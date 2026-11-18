"""Model-led notebook studies over trusted instruments, using the existing Scientist."""
from decimal import Decimal
from pathlib import Path
import os

from ..harness.runner import write_json


class DiscoveryAdapter:
    model = None

    @staticmethod
    def validate(config):
        value = config.value
        if value["model"]["provider"] != "copilot" or value["limits"]["credits"] <= 0:
            raise ValueError("Live discovery requires Copilot and a fresh positive credit ceiling")
        for arm in config.arms:
            if (arm.memory != "external" or arm.advisor != "off" or arm.selection != "active"
                    or arm.profile != "notebook-discovery-v1"):
                raise ValueError("This qualified discovery profile is external-notebook, active and no review")
        allowed = {"checkout", "cli_version"}
        if value["benchmark"] == "made":
            allowed.add("worker_python")
        if set(value.get("parameters", {})) != allowed:
            raise ValueError(f"Discovery parameters must be exactly {sorted(allowed)}")
        if any(not isinstance(value["parameters"][k], str) or not value["parameters"][k] for k in allowed):
            raise ValueError("Declare nonempty checkout, CLI version and worker interpreter paths")

    def prepare(self, config, output, approved_credits):
        from ..providers.guarded import GuardedCopilotModel, StudyBudget
        from ..harness.runner import source_manifest
        import hashlib
        self.config, self.output = config, output
        if not Path(config.value["parameters"]["checkout"]).is_dir():
            raise ValueError("Restore the pinned external benchmark checkout first")
        self.budget = StudyBudget(output / "credits.json", approved_credits,
                                  len(config.cells()) * config.value["limits"]["requests"], config.value)
        root = Path(__file__).resolve().parents[2]
        for name, expected in source_manifest()["sha256"].items():
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError("Harness source changed during registration")
            copy = output / "source-snapshot" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            with copy.open("xb") as handle:
                handle.write(data)
        self.model = GuardedCopilotModel(output / "transport", config.value["model"]["id"],
                                        self.budget, config.value["parameters"]["cli_version"])
        write_json(output / "driver.json", {"driver_pid": os.getpid(),
                   "bridge_pid": self.model.process.pid, "numerical_concurrency": 1})

    def run_episode(self, cell, directory):
        from ..agent import Scientist
        from ..llm import MeteredModel
        from ..notebook import render, snapshot
        from ..tools import AcquisitionContract
        before = self.model.accounting()
        if before["unresolved_requests"]:
            raise RuntimeError("An earlier paid request is unresolved; cannot start another episode")
        world = self.testbed(cell)
        try:
            write_json(directory / "numeric-worker.json",
                       {"pid": world.worker.process.pid, "role": self.config.value["benchmark"],
                        "native_discoverphysics_worker": False, "concurrency": 1})
            for tool in world.tools.tools.values():
                tool.acquisition = AcquisitionContract(
                    "instrument_call", "One agent-selected instrument invocation",
                    ("observations",), lambda **kwargs: {"observations": 1})
            model = MeteredModel(self.model)
            scientist = Scientist(
                model, directory / "store", world.tools,
                max_steps=self.config.value["limits"]["requests"],
                max_context_bytes=20000, analysis_enabled=True,
                require_predictive_check=cell["world"] == "vanilla_equation",
                memory_policy="notebook", submission_validator=world.validate_submission,
                resource_limits={"observations": self.config.value["limits"]["observations"]})
            try:
                result = scientist.investigate(
                    world.goal, context=world.context,
                    metadata=dict(world.manifest, agent="copilot", model=self.model.model,
                                  policy_profile="notebook-discovery-v1"))
                result["model_calls"] = model.calls
                result["evaluation_status"] = "pending"
                write_json(directory / "result.json", result)
                scientist.store.save_run(result)
                (directory / "lab-notebook.md").write_text(
                    render(snapshot(scientist.store, result, result["steps"])), encoding="utf-8")
                try:
                    evaluation = world.grade(result.get("conclusion") or {})
                    if not isinstance(evaluation, dict) or type(evaluation.get("passed")) is not bool:
                        raise ValueError("The private evaluator returned no boolean verdict")
                except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError, OSError) as exc:
                    result["evaluation_status"] = "failed"
                    result["evaluation"] = {
                        "passed": None, "error": f"{type(exc).__name__}: {exc}"}
                else:
                    result["evaluation_status"] = "completed"
                    result["evaluation"] = evaluation
                scientist.store.save_run(result)
                write_json(directory / "result.json", result)
            finally:
                write_json(directory / "model-calls.json", model.calls)
                scientist.close()
        finally:
            world.close()
        after = self.model.accounting()
        accounting = {
            key: str(Decimal(after[key]) - Decimal(before[key]))
            for key in ("accounted_credits", "confirmed_credits")
        }
        if any(Decimal(value) < 0 for value in accounting.values()):
            raise RuntimeError("Study accounting decreased across an episode with no inherited hold")
        accounting["unresolved_requests"] = after["unresolved_requests"]
        evaluation_failed = result["evaluation_status"] == "failed"
        return {
            "status": "evaluation_failed" if evaluation_failed else result["status"],
            "discovery_status": result["status"],
            "evaluation_status": result["evaluation_status"],
            "final_pass": result["evaluation"]["passed"],
            "censored": evaluation_failed, "stop_study": evaluation_failed,
            "requests": len(model.calls), "observations": len(result["evidence"]),
            "metrics": result["evaluation"], "provider": "copilot",
            "resource_budget": result["resource_budget"],
            "billable_requests": len(model.calls), "accounting": accounting,
            "notebook_reads": sum(str(e.get("lookup", {}).get("id", "")).startswith("notebook:")
                                  for e in result["events"]),
            "runtime": {"trusted_simulator_worker_drained": world.worker.process.poll() is not None},
        }

    def close(self):
        if self.model is not None:
            self.model.close()


class NewtonDiscoveryAdapter(DiscoveryAdapter):
    def testbed(self, cell):
        from ..testbeds.newtonbench import NewtonBench
        world = NewtonBench(self.config.value["parameters"]["checkout"], seed=cell["seed"],
                            law_version="v1")
        world.goal = (
            "Discover a quantitative expression for the unknown stored-energy function U(x) "
            "from your own chosen noisy measurements. Use the external lab notebook to retain "
            "your hypotheses and evidence. Use the trusted analysis tools to estimate coefficients, "
            "freeze a candidate and check predictions on newly acquired observations before "
            "a concluded final submission. Do not assume the textbook law is the apparatus law.")
        return world


class MadeDiscoveryAdapter(DiscoveryAdapter):
    def testbed(self, cell):
        from ..testbeds.made import Made
        parameters = self.config.value["parameters"]
        return Made(parameters["checkout"], parameters["worker_python"], seed=cell["seed"],
                    budget=self.config.value["limits"]["observations"])
