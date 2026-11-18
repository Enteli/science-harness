"""Native-role SDK transport; every paid dispatch needs a durable reservation."""
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess

from .physics_budget import PilotBudget, price_envelope
from .physics_context import admit_context
from .physics_reference import digest
from .physics_runtime import atomic_json, encode


class CopilotProvider:
    def __init__(self, directory, credits=1000, requests=16, *, budget=None, pinned_version=None,
                 receipt_directory=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.receipt_directory = Path(receipt_directory) if receipt_directory is not None else None
        # All development attempts share this approval, even with new output directories.
        self.budget = budget if budget is not None else PilotBudget(
            Path(__file__).resolve().parents[1] / ".science-harness/physics-development-credits.json",
            credits, requests)
        cli = shutil.which("copilot")
        if not cli:
            raise RuntimeError("Copilot CLI unavailable; no paid fallback")
        version_args = [cli, "--version"] if pinned_version is None else [
            cli, "--prefer-version", pinned_version, "--no-auto-update", "--version"]
        self.version = subprocess.check_output(version_args, text=True).splitlines()[0]
        match = re.fullmatch(r"GitHub Copilot CLI (\d+\.\d+\.\d+(?:-[\w.-]+)?)\.", self.version)
        if not match:
            raise RuntimeError("Cannot pin the installed Copilot runtime version")
        self.cli_version = match[1]
        if pinned_version is not None and self.cli_version != pinned_version:
            raise RuntimeError("The study's pinned Copilot runtime is unavailable")
        pinned = subprocess.check_output(
            [cli, "--prefer-version", self.cli_version, "--no-auto-update", "--version"], text=True)
        if pinned.splitlines()[0] != self.version:
            raise RuntimeError("The requested cached CLI version is unavailable")
        self.stderr = (self.directory / "runtime-stderr.log").open("a")
        self.process = subprocess.Popen(
            ["node", str(Path(__file__).with_name("physics_bridge.mjs")),
             str(self.directory / "runtime")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, text=True, start_new_session=True,
            env={**{key: value for key, value in os.environ.items() if key in {
                "PATH", "HOME", "TMPDIR", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR",
                "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "GH_HOST", "GH_TOKEN",
                "SCIENCE_HARNESS_SDK_ROOT",
            }}, "COPILOT_CLI_PATH": cli, "COPILOT_CLI_VERSION": self.cli_version,
                 "COPILOT_AUTO_UPDATE": "false"})

    def rpc(self, operation, **payload):
        self.process.stdin.write(encode({"operation": operation, **payload}) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 345)
        if not ready:
            raise RuntimeError("Bridge timeout; pending reservation remains held")
        response = self.process.stdout.readline()
        if not response:
            raise RuntimeError("Bridge exited; inspect runtime-stderr.log; no retry")
        return json.loads(response)

    def catalog(self):
        response = self.rpc("catalog")
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def __call__(self, **packet):
        admit_context(packet)
        prepared = self.rpc("prepare", **packet)
        if not prepared["ok"]:
            raise RuntimeError(prepared["error"])
        ready = prepared["result"]
        admission = admit_context(packet, ready)
        models = {row["id"]: row for row in ready["catalog"]["models"]}
        units = {row["modelId"]: row["unit"] for row in ready["catalog"]["units"]}
        quote = price_envelope(models[packet["model"]], units.get(packet["model"]))
        reservation = self.budget.reserve(quote, digest(encode(packet)))
        admission_record = {
            "request_sha256": digest(encode(packet)), "admission": admission,
            "runtime_context": ready["runtime_context"], "context_info": ready["context_info"],
            "pending_text_runtime_tokens": ready["pending_text_runtime_tokens"], "quote": quote,
        }
        atomic_json(self.directory / f"admission-{reservation:04d}.json", admission_record)
        if self.receipt_directory is not None:
            atomic_json(self.receipt_directory / f"admission-{reservation:04d}.json", admission_record)
        response = self.rpc("dispatch", session_key=packet["session_key"],
                            prepared_hash=ready["prepared_hash"])
        response["request_sha256"] = digest(encode(packet))
        response["session_key_sha256"] = digest(encode(packet["session_key"]))
        receipt = self.directory / f"dispatch-{reservation:04d}.json"
        if receipt.exists():
            raise RuntimeError("Refusing to overwrite an original dispatch receipt")
        atomic_json(receipt, response)
        if self.receipt_directory is not None:
            durable = self.receipt_directory / receipt.name
            if durable.exists():
                raise RuntimeError("Refusing to overwrite a persistent original dispatch receipt")
            atomic_json(durable, response)
        if not response["ok"]:
            # Unknown or partially billed failures keep the entire reservation.
            raise RuntimeError(response["error"])
        result = response["result"]
        self.budget.settle(reservation, result)
        result["context_admission"] = admission
        result["runtime_version"] = self.version
        result["credit_reservation_id"] = reservation
        for event in result["usage"]:
            if event["inputTokens"] > 131072 or event["outputTokens"] > 32768:
                raise RuntimeError("Reported usage exceeded the benchmark token allowance")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
        self.process.stdout.close()
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def authorize_terminated_timeout(directory, budget, approval):
    """Keep the full charge; authorize continuation only after a recorded abort/shutdown."""
    state = budget.snapshot()
    authorized = {r["request_id"] for r in state.get("held_continuations", [])}
    pending = [r for r in state["requests"] if r["status"] == "reserved" and r["id"] not in authorized]
    if not pending:
        return []
    directory = Path(directory)
    held = []
    for row in pending:
        receipt_path = directory / f"dispatch-{row['id']:04d}.json"
        if not receipt_path.exists():
            raise RuntimeError("No complete timeout receipt; automatic continuation forbidden")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("ok") or receipt.get("error") != "Model response timeout"
                or receipt.get("request_sha256") != row["request_sha256"]):
            raise RuntimeError("Only a proven terminated timeout can retain a continuation hold")
        key = receipt.get("session_key_sha256", "")
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise RuntimeError("Missing owned-session identity")
        projection = json.loads((directory / "runtime" / (key + ".json")).read_text())
        session_id = projection["sessionId"]
        if not re.fullmatch(r"[a-f0-9-]{36}", session_id) or not projection["pending"]:
            raise RuntimeError("Unexpected owned-session state")
        events_path = directory / "runtime/session-state" / session_id / "events.jsonl"
        text = events_path.read_text()
        events = [json.loads(line) for line in text.splitlines()]
        types = [e["type"] for e in events if not e.get("agentId")]
        starts = [i for i, t in enumerate(types) if t == "assistant.turn_start"]
        if not starts:
            raise RuntimeError("No recorded model turn; timeout remains unresolved")
        tail = types[starts[-1]:]
        if not all(t in tail for t in ("abort", "assistant.turn_end", "session.shutdown")):
            raise RuntimeError("Local turn has not demonstrably terminated")
        if not tail.index("abort") < tail.index("assistant.turn_end") < tail.index("session.shutdown"):
            raise RuntimeError("Unexpected timeout termination order")
        budget.authorize_held_continuation(row["id"], approval, digest(text))
        held.append(row["id"])
    return held
