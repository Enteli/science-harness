"""Inspectable public trace exports from immutable operator receipts."""
import copy
import json
from pathlib import Path
import re

from .physics_reference import digest
from .physics_runtime import atomic_json


def fence(text, language=""):
    length = max([3] + [len(m.group()) + 1 for m in re.finditer(r"`+", text)])
    ticks = "`" * length
    return f"{ticks}{language}\n{text}\n{ticks}"


def export_episode(result, directory):
    directory = Path(directory)
    operations = [json.loads(p.read_text()) for p in sorted((directory / "operations").glob("*.json"))]
    public = []
    for original in operations:
        if original["kind"] not in (
                "model", "measurement", "numerics", "notebook", "design_review",
                "public_response_normalization", "action_normalization", "action_validation"):
            continue
        item = copy.deepcopy(original)
        if item["kind"] == "measurement":
            item["input"].pop("noise_seed", None)
            item["input_sha256_meaning"] = "Hash of the original operator input, including its retained noise seed"
        public.append(item)
    evidence = {
        "scope": "public-agent-trace", "manifest": result["manifest"],
        "status": result["status"], "error": result["error"], "requests": result["requests"],
        "operations": public, "notebook": result["notebook"],
        "sources": result["sources"], "observations": result["observations"],
        "executions": [{k: e[k] for k in ("observation_id", "intent", "executed")}
                       for e in result["executions"]],
        "candidates": result["candidates"], "emissions": result["emissions"],
        "final_source_sha256": result["final_source_sha256"],
        "private_reasoning_included": False, "private_grading_included": False,
        "note": "Exact model-visible inputs and public outputs; measurements and fits are observations/analyses, "
                "not SDK shell/MCP tools. Private clean evaluation is separate.",
    }
    if "notebook_memory" in result:
        evidence["notebook_memory"] = copy.deepcopy(result["notebook_memory"])
    if "design_reviews" in result:
        evidence.update({k: result[k] for k in (
            "design_reviews", "scientist_requests", "assistance_requests", "request_roles")})
    elif "request_roles" in result:
        evidence.update({k: result[k] for k in (
            "scientist_requests", "assistance_requests", "request_roles")})
    atomic_json(directory / "public-trace.json", evidence)
    lines = [
        "# Full public agent trace", "",
        f"Model: `{result['manifest']['model']}`; controller: `{result['manifest']['controller']}`; "
        f"selector: `{result['manifest']['selector']}`; status: `{result['status']}`.",
        "", evidence["note"], "",
        "## Operations", "",
    ]
    model_number, measurement_number = 0, 0
    for i, operation in enumerate(public):
        lines += [f"### Operation {i}: {operation['kind']}", "",
                  f"Status: `{operation['status']}`. Elapsed seconds: "
                  f"`{operation.get('elapsed_seconds', 'unresolved')}`.", ""]
        if operation["kind"] == "model":
            model_number += 1
            if "request_roles" in result:
                lines += [f"Role: `{result['request_roles'][model_number - 1]}`.", ""]
            lines += [f"**Model request {model_number}: exact system input**", "",
                      fence(operation["input"]["system"], "text"), ""]
            for j, message in enumerate(operation["input"]["messages"]):
                lines += [f"**Input message {j}: {message['role']}**", "",
                          fence(message["content"], "text"), ""]
            lines += ["**Public output / error**", "", fence(
                operation["result"]["content"] if operation["status"] == "completed"
                else operation.get("error", "Unresolved dispatch"), "text"), ""]
            if operation["status"] == "completed":
                lines += ["**Usage and public message identities**", "", fence(json.dumps({
                    k: operation["result"][k] for k in ("usage", "visible_messages")
                    if k in operation["result"]}, indent=2), "json"), ""]
        else:
            if operation["kind"] == "measurement":
                measurement_number += 1
                lines += [f"**Evidence ID: experiment-{measurement_number:04d}**", ""]
            lines += ["**Exact request**", "", fence(json.dumps(operation["input"], indent=2), "json"),
                      "", "**Output / error**", "", fence(json.dumps(
                          operation.get("result", {"error": operation.get("error")}), indent=2), "json"), ""]
    (directory / "trace.md").write_text("\n".join(lines) + "\n")
    notebook_lines = [
        "# Scientific notebook", "",
        "Public scientist-authored entries, not private reasoning or newly measured evidence. "
        "The complete input/output sequence and observed tool results are in trace.md.", "",
    ]
    if result["manifest"]["controller"] != "science":
        notebook_lines += ["DiscoveryAgent has no notebook intervention. Its full original dialogue is preserved.",
                           "No notes were synthesized for this baseline."]
    cursor = 0
    entry_calls = []
    for entry in result["notebook"]:
        match = next((j for j in range(cursor, len(result["responses"]))
                      if digest(result["responses"][j]["content"]) == entry["original_response_sha256"]), None)
        if match is None:
            raise RuntimeError("Notebook entry lost its exact public source response")
        cursor = match + 1
        entry_calls.append(match + 1)
    for i, (entry, call) in enumerate(zip(result["notebook"], entry_calls), 1):
        next_call = entry_calls[i] if i < len(entry_calls) else len(result["responses"]) + 1
        notebook_lines += [f"## Entry {i}", "",
                           f"Request: {call}; observations available before decision: "
                           f"{entry['decision_observed_count']}.", "",
                           fence(json.dumps(entry["note"], indent=2), "json"),
                           "", f"Original response hash: `{entry['original_response_sha256']}`.", ""]
        for candidate in result["candidates"]:
            if call <= candidate["request_count"] < next_call:
                notebook_lines += ["**Harness observed fit after this response (not private accuracy):**", "",
                                   fence(json.dumps(candidate, indent=2), "json"), ""]
    if "design_reviews" in result:
        notebook_lines += ["## Experiment-design reviews", "",
                           "Drafts and advice are not new measurements or scored law submissions.", ""]
        for review in result["design_reviews"]:
            notebook_lines += [f"### {review['review_id']}", "",
                               fence(json.dumps(review, indent=2), "json"), ""]
    notebook_lines += ["## Exact submitted sources", ""]
    for key, source in result["sources"].items():
        if digest(source) != key:
            raise RuntimeError("Archived source hash changed")
        notebook_lines += [f"### {key}", "", fence(source, "python"), ""]
    (directory / "notebook.md").write_text("\n".join(notebook_lines) + "\n")
    return {"trace_path": "public-trace.json", "trace_markdown": "trace.md",
            "notebook_markdown": "notebook.md", "operations": len(public),
            "notebook_entries": len(result["notebook"]),
            "trace_sha256": digest((directory / "public-trace.json").read_text())}
