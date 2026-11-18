"""Three logical books, revision proposals and run records in one local database."""
import json
import re
import sqlite3
import time
import uuid
from pathlib import Path


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True)


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def strings(value, name):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"{name} must be a list of strings")
    return value


class BookStore:
    BOOKS = ("playbook", "evidence", "learnings", "findings")

    def __init__(self, home=".science-harness"):
        home = Path(home)
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(home / "science.sqlite3"))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT NOT NULL, revision INTEGER NOT NULL, book TEXT NOT NULL,
                content TEXT NOT NULL, created REAL NOT NULL,
                PRIMARY KEY (id, revision)
            );
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, book TEXT NOT NULL, target TEXT,
                base_revision INTEGER, content TEXT NOT NULL, reason TEXT NOT NULL,
                run_id TEXT, status TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, content TEXT NOT NULL
            );
        """)

    def close(self):
        self.connection.close()

    def get(self, entry_id, revision=None):
        condition = "id=?" if revision is None else "id=? AND revision=?"
        parameters = (entry_id,) if revision is None else (entry_id, revision)
        row = self.connection.execute(
            f"SELECT * FROM entries WHERE {condition} ORDER BY revision DESC LIMIT 1",
            parameters,
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown entry: {entry_id}")
        return dict(json.loads(row["content"]), id=row["id"],
                    revision=row["revision"], book=row["book"])

    def frozen_entries(self, snapshot):
        """Read exact accepted revisions, not whatever has changed since run start."""
        return [self.get(entry_id, revision) for entry_id, revision in snapshot.items()]

    def entries(self, book):
        if book not in self.BOOKS:
            raise ValueError(f"Unknown book: {book}")
        rows = self.connection.execute("""
            SELECT e.* FROM entries e
            WHERE book=? AND revision=(SELECT MAX(revision) FROM entries WHERE id=e.id)
            ORDER BY created, id
        """, (book,))
        return [dict(json.loads(r["content"]), id=r["id"],
                     revision=r["revision"], book=r["book"]) for r in rows]

    def validate_citations(self, content, required=False):
        ids = strings(content.get("evidence_ids", []), "evidence_ids")
        if required and not ids:
            raise ValueError("At least one evidence citation is required")
        for entry_id in ids:
            if self.get(entry_id)["book"] != "evidence":
                raise ValueError(f"Not evidence: {entry_id}")
        for entry_id in strings(content.get("contradiction_ids", []), "contradiction_ids"):
            if self.get(entry_id)["book"] != "evidence":
                raise ValueError(f"Not evidence: {entry_id}")

    def validate(self, book, content):
        if book not in self.BOOKS or not isinstance(content, dict):
            raise ValueError("Unknown book or invalid entry")
        if {"id", "book", "revision"} & content.keys():
            raise ValueError("Entry identity is controlled by the store")
        encode(content)
        if book == "playbook":
            text(content.get("title"), "title")
            text(content.get("guidance"), "guidance")
        elif book == "learnings":
            text(content.get("claim"), "claim")
            text(content.get("scope"), "scope")
            if content.get("status") not in ("hypothesis", "supported", "refuted", "retired"):
                raise ValueError("Invalid learning status")
        elif book == "findings":
            text(content.get("title"), "title")
            text(content.get("description"), "description")
            if content.get("severity") not in ("low", "medium", "high", "critical"):
                raise ValueError("Invalid finding severity")
        else:
            text(content.get("source"), "source")
            if "data" not in content:
                raise ValueError("Evidence needs original data")
            if not isinstance(content.get("conditions", {}), dict):
                raise ValueError("conditions must be an object")
        for name in ("assumptions", "uncertainties", "alternatives", "tags"):
            strings(content.get(name, []), name)
        self.validate_citations(
            content, required=book == "findings"
            or (book == "learnings" and content.get("status") == "supported"),
        )

    def _insert(self, book, content, entry_id=None, revision=1):
        if type(revision) is not int or not 1 <= revision <= 2**63 - 1:
            raise ValueError("revision must be a positive SQLite integer")
        if book == "evidence" and revision != 1:
            raise ValueError("Evidence is append-only; imported evidence must have revision 1")
        self.validate(book, content)
        entry_id = entry_id or f"{book[:2]}-{uuid.uuid4().hex}"
        self.connection.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
            (entry_id, revision, book, encode(content), time.time()),
        )
        return entry_id

    def append(self, book, content, entry_id=None, revision=1):
        """Operator import / acquisition API. Models only get propose()."""
        with self.connection:
            if revision != 1 and self.connection.execute(
                    "SELECT 1 FROM entries WHERE id=?", (entry_id,)).fetchone():
                raise ValueError("Import a revision into a fresh store; revise existing knowledge through proposals")
            return self._insert(book, content, entry_id, revision)

    def propose(self, book, content, reason, run_id=None, target=None, base_revision=None):
        if book not in ("playbook", "learnings"):
            raise ValueError("Only reusable knowledge can be revised")
        self.validate(book, content)
        text(reason, "reason")
        if base_revision is not None and (not target or type(base_revision) is not int or base_revision < 1):
            raise ValueError("base_revision requires a target and a positive integer revision")
        base = None
        if target:
            old = self.get(target)
            if old["book"] != book:
                raise ValueError("Cannot change an entry's book")
            if base_revision is not None and old["revision"] != base_revision:
                raise ValueError("Proposal target changed since its pinned base revision")
            base = old["revision"] if base_revision is None else base_revision
        proposal_id = f"pr-{uuid.uuid4().hex}"
        with self.connection:
            self.connection.execute(
                "INSERT INTO proposals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (proposal_id, book, target, base, encode(content), reason,
                 run_id, "pending", time.time()),
            )
        return proposal_id

    def proposals(self):
        return [dict(dict(r), content=json.loads(r["content"])) for r in
                self.connection.execute("SELECT * FROM proposals ORDER BY created")]

    def review(self, proposal_id, approve):
        with self.connection:
            # Lock the revision check and write together, including across CLI processes.
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT * FROM proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if row is None or row["status"] != "pending":
                raise ValueError("Proposal does not exist or is already reviewed")
            entry_id = None
            if approve:
                if row["target"] and self.get(row["target"])["revision"] != row["base_revision"]:
                    raise ValueError("Stale proposal: entry has changed; propose again")
                entry_id = self._insert(
                    row["book"], json.loads(row["content"]), row["target"],
                    (row["base_revision"] or 0) + 1,
                )
            self.connection.execute(
                "UPDATE proposals SET status=? WHERE id=?",
                ("approved" if approve else "rejected", proposal_id),
            )
        return entry_id

    def snapshot(self):
        return {r["id"]: r["revision"] for book in self.BOOKS for r in self.entries(book)}

    def search(self, query, limit=8):
        words = set(re.findall(r"\w+", query.lower()))
        result = {}
        for book in self.BOOKS:
            rows = self.entries(book)
            # Never filter contradictory/refuted claims out of retrieval.
            rows.sort(key=lambda r: len(words & set(re.findall(r"\w+", encode(r).lower()))),
                      reverse=True)
            result[book] = rows[:limit]
        # Retrieved claims must carry the evidence they cite, even outside the top-k.
        cited = {i for book in ("learnings", "findings") for r in result[book]
                 for key in ("evidence_ids", "contradiction_ids") for i in r.get(key, [])}
        present = {r["id"] for r in result["evidence"]}
        result["evidence"].extend(self.get(i) for i in sorted(cited - present))
        return result

    def save_run(self, run):
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?)", (run["id"], encode(run))
            )

    def run(self, run_id):
        row = self.connection.execute("SELECT content FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown run: {run_id}")
        return json.loads(row["content"])
