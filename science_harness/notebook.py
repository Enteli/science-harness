"""Session lab notebook: exact views of the durable ledger, not another evidence book."""
import copy
import json

from .books import encode, strings, text


ENTRY_BYTES = 1800


def completed_step(run):
    ceiling = run["steps"]-1 if run.get("status") == "running" else run["steps"]
    return max((event["step"] for event in run["events"] if event["step"] <= ceiling), default=0)


def snapshot(store, run, step):
    if type(step) is not int or step < 0 or step > completed_step(run):
        raise ValueError("Notebook snapshots contain only completed steps of this investigation")
    events = [event for event in run["events"] if event["step"] <= step]
    if step and not any(event["step"] == step for event in events):
        raise ValueError("Unknown completed notebook step")
    artifacts = [copy.deepcopy(item) for item in run.get("method_artifacts", []) if item["step"] <= step]
    hypotheses = [copy.deepcopy(h) for item in artifacts if item["kind"] == "hypotheses" for h in item["hypotheses"]]
    reviews = [item for item in artifacts if item["kind"] == "review"]
    entries, notes = [], []
    for event in events:
        action = event.get("action", {})
        entry = {"step": event["step"], "source_event": f"event:{event['step']}",
                 "action": copy.deepcopy(action), "inherited": event.get("inherited", False)}
        for key in ("method_phase", "method_artifacts", "evidence_id", "analysis_id", "finding_id",
                    "execution", "error", "preflight_rejected", "accepted", "notebook_entry",
                    "model_role", "review_proposal", "scientific_review", "review_response",
                    "applied_proposal_step", "unperformed_review"):
            if key in event:
                entry[key] = copy.deepcopy(event[key])
        if "lookup" in event:
            entry["lookup_source"] = {key: event["lookup"][key] for key in ("id", "revision", "field")
                                      if key in event["lookup"]}
        entries.append(entry)
        if event.get("notebook_entry"):
            notes.append(dict(copy.deepcopy(event["notebook_entry"]), step=event["step"]))
        elif action.get("action") == "note" and not event.get("error"):
            notes.append({"step": event["step"], "working_state_note": {
                key: copy.deepcopy(action[key]) for key in ("question", "hypotheses", "assumptions", "uncertainties", "resolve")
                if key in action}})
    observations, calculations = [], []
    for key, destination in (("evidence", observations), ("analyses", calculations)):
        for entry_id in run.get(key, []):
            record = store.get(entry_id)
            if record.get("step", 0) > step:
                continue
            destination.append({"id": entry_id, "step": record.get("step"), "source": record["source"],
                                "inherited": record.get("run_id") != run["id"]})
    executed = {item["design_id"] for item in artifacts if item["kind"] == "execution"
                and not item["preflight_rejected"] and item["status"] != "not_started"}
    reviewed = {item["design_id"] for item in reviews}
    return {
        "id": f"notebook:{step}", "revision": 1, "book": "notebook", "kind": "derived",
        "source": "session.lab_notebook", "run_id": run["id"],
        "data": {"goal": run["goal"], "apparatus": run["context"], "through_step": step,
                 "entries": entries, "hypotheses": hypotheses,
                 "designs": [item for item in artifacts if item["kind"] == "design"],
                 "reviews": reviews, "latest_review": reviews[-1] if reviews else None,
                 "notes": notes, "observations": observations, "calculations": calculations,
                 "unreviewed_design_ids": sorted(executed-reviewed),
                 "meaning": "Exact public ledger view. Notes and assessments are interpretations, not new measurements. "
                            "Notebook/event handles navigate; cite underlying observation or calculation IDs."},
    }


def validate_entry(store, run, value):
    from .context import visible_source

    if not isinstance(value, dict) or set(value) - {"title", "body", "evidence_ids"}:
        raise ValueError("notebook_entry requires title, body and optional evidence_ids")
    text(value.get("title"), "notebook title")
    text(value.get("body"), "notebook body")
    if len(encode(value).encode()) > ENTRY_BYTES:
        raise ValueError(f"Notebook entry exceeds {ENTRY_BYTES} bytes; write smaller source-linked entries")
    ids = strings(value.get("evidence_ids", []), "notebook evidence_ids")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate notebook citations")
    for entry_id in ids:
        if visible_source(store, run, entry_id).get("book") != "evidence":
            raise ValueError("Notebook notes cite underlying evidence, not notebook or event handles")
    return copy.deepcopy(value)


def working_context(run, method_view):
    """Keep current claims and predictions exact; offload completed rationale and bulky support lists."""
    step = completed_step(run)
    handle = f"notebook:{step}"
    artifacts = [item for item in run.get("method_artifacts", []) if item["step"] <= step]
    hypotheses = [h for item in artifacts if item["kind"] == "hypotheses" for h in item["hypotheses"]]
    positions = {h["id"]: index for index, h in enumerate(hypotheses)}
    reviews = [item for item in artifacts if item["kind"] == "review"]
    executed = {item["design_id"] for item in artifacts if item["kind"] == "execution"
                and not item["preflight_rejected"] and item["status"] != "not_started"}
    reported = {status: sorted({assessment["hypothesis_id"] for review in reviews for assessment in review["assessments"]
                               if assessment["status"] == status}) for status in ("challenged", "unresolved")}
    index = {
        "id": handle, "revision": 1,
        "unreviewed_design_ids": sorted(executed - {review["design_id"] for review in reviews}),
        "historically_challenged": reported["challenged"], "historically_unresolved": reported["unresolved"],
        "read": "Page data.entries/hypotheses/reviews/notes. Statuses are historical; omission is not resolution. "
                "Cite evidence IDs, not notebook handles.",
        "write": {"action": "note", "notebook_entry": {"title": "string", "body": "interpretation, not measurement",
                                                     "evidence_ids": []}},
        "note_bytes": ENTRY_BYTES,
    }
    if method_view is None:
        return None, index
    view = copy.deepcopy(method_view)
    for position, hypothesis in enumerate(view["hypotheses"]):
        if not hypothesis.get("evidence_ids"):
            continue
        projected = {key: value for key, value in hypothesis.items() if key != "evidence_ids"}
        projected.update(support_count=len(hypothesis["evidence_ids"]),
                         fetch_support={"id": handle, "field": f"data.hypotheses.{positions[hypothesis['id']]}.evidence_ids",
                                        "start": 0, "count": min(50, len(hypothesis["evidence_ids"]))})
        if len(encode(projected)) < len(encode(hypothesis)):
            view["hypotheses"][position] = projected
    plan = view.get("current_design")
    if plan:
        plan["source_event"] = f"event:{plan['step']}"
        for key in ("kind", "run_id", "step"):
            plan.pop(key, None)
        if plan.get("execution"):
            for key in ("kind", "design_id"):
                plan["execution"].pop(key, None)
    previous = view.get("last_review")
    if previous:
        view["last_review"] = {
            "design_id": previous["design_id"], "result": previous["result"],
            "assessments": [{"hypothesis_id": item["hypothesis_id"], "status": item["status"]}
                            for item in previous["assessments"]],
            "fetch_rationale": {"id": handle, "field": "data.latest_review"},
        }
    return view, index


def render(record):
    data = record["data"]
    lines = ["# Session lab notebook", "", f"Run: `{record['run_id']}`", f"Snapshot: `{record['id']}`", "",
             data["meaning"], "", "## Research question", "", data["goal"], "",
             "## Public apparatus contract", "", data["apparatus"], "",
             "## Source index", "", f"Observations: {len(data['observations'])}; calculations: {len(data['calculations'])}.",
             "Original numeric records remain in the evidence ledger under the IDs below.", "",
             "```json", json.dumps({"observations": data["observations"], "calculations": data["calculations"]}, indent=2),
             "```", ""]
    for entry in data["entries"]:
        lines += [f"## Step {entry['step']}: {entry['action'].get('action', 'invalid response')}", "",
                  f"Exact source: `{entry['source_event']}`", "",
                  "```json", json.dumps(entry, indent=2), "```", ""]
    return "\n".join(lines)
