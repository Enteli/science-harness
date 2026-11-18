"""Credential-free external-notebook demonstration, not an LLM discovery result."""
import argparse
import json
from pathlib import Path

from science_harness import AcquisitionContract, Observation, Scientist, ScriptedLLM, Tool, ToolRegistry
from science_harness.notebook import render, snapshot


def run(output):
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    def measure(x):
        if type(x) not in (int, float) or not 0 <= x <= 2:
            raise ValueError("x must be a finite number in [0, 2]")
        return Observation({"y": 2 * x}, source="demo.sensor", conditions={"x": x})

    tools = ToolRegistry([Tool(
        "measure", "Read a demonstration sensor at x in [0, 2].", measure,
        acquisition=AcquisitionContract("instrument_call", "One sensor invocation",
                                        ("observations",), lambda **_: {"observations": 1}))])
    actions = [
        {"action": "tool", "tool": "measure", "args": {"x": 1},
         "why": "Obtain one reading.", "discriminates": "No global-law claim from one point.",
         "assumptions": []},
        {"action": "note", "notebook_entry": {
            "title": "Limited evidence", "body": "One point does not identify a global law.",
            "evidence_ids": []}},
        {"action": "fetch", "id": "notebook:2", "field": "data.notes", "start": 0, "count": 1},
        {"action": "finish", "status": "inconclusive",
         "summary": "The notebook was retrieved; additional experiments would be needed.", "updates": []},
    ]
    scientist = Scientist(
        ScriptedLLM([json.dumps(action) for action in actions]), output / "store", tools,
        max_steps=4, memory_policy="notebook", resource_limits={"observations": 1})
    try:
        result = scientist.investigate("Demonstrate durable notes and explicit retrieval.",
                                       context="A toy sensor; no private evaluator is exposed.")
        notebook = output / "lab-notebook.md"
        notebook.write_text(render(snapshot(scientist.store, result, result["steps"])),
                            encoding="utf-8")
        return {"status": result["status"], "observations": len(result["evidence"]),
                "api_calls": 0, "notebook": str(notebook)}
    finally:
        scientist.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="a NEW directory")
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
