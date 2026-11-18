"""Small reference adapters reuse the generic Scientist, never a new loop."""
import json
from pathlib import Path

from ..harness.runner import write_json


class NotebookReferencePolicy:
    """Deterministic integration fixture, not a learned notebook-use policy."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.prefix = 0

    def complete(self, system, user):
        packet = json.loads(user)
        if "lab_notebook" not in packet:
            raise RuntimeError("The notebook reference profile requires the public notebook interface")
        if self.prefix == 0:
            action = {"action": "note", "notebook_entry": {
                "title": "Integration procedure",
                "body": "This deterministic reference will estimate a law from acquired readings; "
                        "it is an integration demonstration, not an LLM discovery result.",
                "evidence_ids": []}}
        elif self.prefix == 1:
            action = {"action": "fetch", "id": packet["lab_notebook"]["id"],
                      "field": "data.notes", "start": 0, "count": 1}
        else:
            return self.delegate.complete(system, user)
        self.prefix += 1
        return json.dumps(action)


class SpringAdapter:
    @staticmethod
    def validate(config):
        value = config.value
        if value["limits"]["credits"] != 0 or value["model"]["id"] != "reference":
            raise ValueError("Reference examples require model id reference and zero credits")
        for arm in config.arms:
            notebook = arm.profile == "notebook-reference-v1" and value["benchmark"] == "newtonbench"
            if (arm.memory != ("external" if notebook else "none")
                    or arm.advisor != "off" or arm.selection != "active"):
                raise ValueError("Reference profiles support only their declared memory and no LLM ablations")
        allowed = {"checkout"} if value["benchmark"] == "newtonbench" else set()
        if set(value.get("parameters", {})) != allowed:
            raise ValueError(f"Reference adapter parameters must be exactly {sorted(allowed)}")
        if allowed and (not isinstance(value["parameters"]["checkout"], str) or
                        not value["parameters"]["checkout"]):
            raise ValueError("Declare the external benchmark checkout")

    def prepare(self, config, output, approved_credits):
        self.config = config

    def testbed(self, cell):
        from ..testbeds.spring import Spring
        return Spring(scenario=cell["world"], seed=cell["seed"])

    def run_episode(self, cell, directory):
        from ..agent import Scientist
        from ..llm import MeteredModel
        from ..tools import AcquisitionContract
        world = self.testbed(cell)
        try:
            for tool in world.tools.tools.values():
                tool.acquisition = AcquisitionContract(
                    "instrument_call", "One started instrument call, including its declared repeated readings",
                    ("observations",), lambda **kwargs: {"observations": 1})
            notebook = cell["arm"]["profile"] == "notebook-reference-v1"
            policy = self.reference_model()
            model = MeteredModel(NotebookReferencePolicy(policy) if notebook else policy)
            scientist = Scientist(
                model, directory / "store", world.tools,
                max_steps=self.config.value["limits"]["requests"],
                memory_policy="notebook" if notebook else "inline",
                submission_validator=getattr(world, "validate_submission", None),
                resource_limits={"observations": self.config.value["limits"]["observations"]})
            try:
                result = scientist.investigate(
                    "Investigate the apparatus, distinguish instrument effects from the "
                    "underlying law, and report what is and is not identifiable.",
                    context=world.context,
                    metadata=dict(world.manifest, agent="baseline", policy_version=1,
                                  policy_profile=cell["arm"]["profile"]))
                result["model_calls"] = model.calls
                try:
                    result["evaluation"] = world.grade(result.get("conclusion") or {})
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    result["evaluation"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
                    scientist.store.save_run(result)
                    raise
                scientist.store.save_run(result)
                if notebook:
                    from ..notebook import render, snapshot
                    (directory / "lab-notebook.md").write_text(
                        render(snapshot(scientist.store, result, result["steps"])), encoding="utf-8")
            finally:
                scientist.close()
            write_json(directory / "result.json", result)
            evaluation = result["evaluation"]
            row = {"status": result["status"], "final_pass": evaluation["passed"],
                    "requests": len(result["model_calls"]), "observations": len(result["evidence"]),
                    "first_hit": None, "capped_observations": None, "censored": False,
                    "metrics": evaluation, "provider": "baseline", "billable_requests": 0,
                    "notebook_reads": sum(str(event.get("lookup", {}).get("id", "")).startswith("notebook:")
                                          for event in result["events"]),
                    "resource_budget": result["resource_budget"],
                    "accounting": {"accounted_credits": "0", "confirmed_credits": "0",
                                   "unresolved_requests": 0}}
        finally:
            world.close()
        if hasattr(world, "worker"):
            row["runtime"] = {"trusted_simulator_worker_drained": world.worker.process.poll() is not None}
        return row

    def reference_model(self):
        from ..baselines import SpringBaseline
        return SpringBaseline()

    def close(self):
        pass


class NewtonBenchAdapter(SpringAdapter):
    def prepare(self, config, output, approved_credits):
        super().prepare(config, output, approved_credits)
        if not Path(config.value["parameters"]["checkout"]).is_dir():
            raise ValueError("Restore the pinned NewtonBench checkout; see docs/second-integration.md")

    def reference_model(self):
        from ..baselines import NewtonPowerBaseline
        return NewtonPowerBaseline()

    def testbed(self, cell):
        from ..testbeds.newtonbench import NewtonBench
        return NewtonBench(Path(self.config.value["parameters"]["checkout"]),
                           difficulty="easy", system=cell["world"], seed=cell["seed"])
