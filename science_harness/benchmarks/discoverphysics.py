"""Native trajectory adapter; the qualified controller and tools stay intact."""
from pathlib import Path
import importlib.util
import json
import shutil
import sys

from ..harness.config import positive
from ..harness.runner import write_json
from ..policies import Arm


def policy_arguments(arm, request_limit):
    return {
        "controller": "discovery" if arm.memory == "none" else "science",
        "selector": arm.selection,
        "notebook_memory": "external" if arm.memory == "external" else "conversation",
        "require_experiment_intent": arm.profile == "memory-advisor-v1",
        "total_request_limit": request_limit if arm.profile == "memory-advisor-v1" else None,
    }


class DiscoverPhysicsAdapter:
    provider = None
    backup = None

    @staticmethod
    def validate(config):
        value, limits = config.value, config.value["limits"]
        parameters = value.get("parameters", {})
        required = {"checkout", "cli_version", "position_scale"}
        if not required <= set(parameters) or set(parameters) - required - {"trace_backup", "action_recovery"}:
            raise ValueError("Native parameters: checkout, cli_version, position_scale; optional trace_backup, action_recovery")
        if "action_recovery" in parameters and type(parameters["action_recovery"]) is not bool:
            raise ValueError("action_recovery must be boolean")
        if "trace_backup" in parameters and (
                not isinstance(parameters["trace_backup"], str) or not parameters["trace_backup"]):
            raise ValueError("trace_backup must name a new backup directory")
        if any(not isinstance(parameters[k], str) or not parameters[k] for k in ("checkout", "cli_version")):
            raise ValueError("Declare the checkout and exact CLI version")
        if value["model"]["id"] not in ("gpt-6.1-sol", "claude-haiku-5.5"):
            raise ValueError("Native physics supports only the qualified model identities")
        if not 2 <= limits["observations"] <= 50 or limits["credits"] <= 0:
            raise ValueError("Native physics requires 2..50 observations and a positive new credit cap")
        scales = parameters["position_scale"]
        if not isinstance(scales, dict) or set(scales) != set(value["worlds"]):
            raise ValueError("Explicit position_scale is required per world; no silent recalibration")
        for world, scale in scales.items():
            positive(scale, f"position_scale.{world}")
        for arm in config.arms:
            if arm.advisor == "design" and (arm.memory == "none" or arm.selection != "active"):
                raise ValueError("Native design review requires a notebook and active selection")
            if arm.profile == "legacy-v1" and (arm.memory == "external" or arm.advisor != "off"):
                raise ValueError("Legacy native profile cannot enable new memory or advice")

    def prepare(self, config, output, approved_credits):
        if sys.platform != "darwin":
            raise RuntimeError("Native generated-code execution requires the qualified macOS sandbox; no unsafe fallback")
        checkout = Path(config.value["parameters"]["checkout"]).resolve()
        if not (checkout / "ScienceAgent").is_dir():
            raise ValueError("Restore the pinned DiscoverPhysics checkout declared in parameters.checkout")
        missing = [name for name in ("numpy", "scipy", "jax", "tiktoken")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError("Install the physics-comparison extra; missing: " + ", ".join(missing))
        if not shutil.which("node") or not shutil.which("copilot"):
            raise RuntimeError("Native physics requires Node, npm ci, and the declared Copilot CLI")
        from ..physics_budget import PilotBudget, number, accounted_cost, price_envelope, upper_cost, CreditLimit
        from ..physics_reference import digest, provenance
        from ..physics_runtime import IsolatedNumerics, NativeWorld, noise_seed, proposal
        from ..physics_transport import CopilotProvider
        self.config, self.output = config, output
        if "trace_backup" in config.value["parameters"]:
            self.backup = Path(config.value["parameters"]["trace_backup"]).resolve()
            primary = output.resolve()
            if self.backup.is_relative_to(primary) or primary.is_relative_to(self.backup):
                raise ValueError("Trace backup must be separate from, not nested in, the primary study")
            self.backup.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.checkout = checkout
        self.provenance = provenance(self.checkout)
        self.numerics = IsolatedNumerics(self.checkout)
        if not all(self.numerics.verify_boundary(Path(__file__)).values()):
            raise RuntimeError("The native numerical isolation boundary failed qualification")
        approval_hash = digest(str(approved_credits) + ":" + str(config.value))

        class StudyBudget(PilotBudget):
            def __init__(ledger, path):
                ledger.path = path
                ledger.credits = number(approved_credits)
                ledger.requests = len(config.cells()) * config.value["limits"]["requests"]

            def initial_state(ledger):
                return {"credit_cap": str(ledger.credits), "request_cap": ledger.requests,
                        "approval": approval_hash, "requests": [], "status": "ready"}

            def validate_approval(ledger, state):
                if state.get("approval") != approval_hash:
                    raise ValueError("Study approval/configuration changed")

        self.budget = StudyBudget(output / "credits.json")
        with self.budget.locked() as state:
            write_json(self.budget.path, state)
        self.accounted_cost = accounted_cost
        receipts = self.backup / "transport" if self.backup is not None else None
        if receipts is not None:
            receipts.mkdir()
        self.provider = CopilotProvider(
            output / "transport", budget=self.budget, receipt_directory=receipts,
            pinned_version=config.value["parameters"]["cli_version"])
        catalog = self.provider.catalog()
        models = {m["id"]: m for m in catalog["models"]}
        model_id = config.value["model"]["id"]
        if model_id not in models:
            raise RuntimeError("The configured model is unavailable; no fallback or inference")
        units = {u["modelId"]: u["unit"] for u in catalog["units"]}
        quote = price_envelope(models[model_id], units.get(model_id))
        reservation = upper_cost(quote)
        self.store("preflight.json", {
            "metadata_only": True, "model": model_id, "quote": quote,
            "required_first_call_reservation": str(reservation),
            "approved_credit_cap": str(self.budget.credits),
            "can_reserve_first_call": reservation <= self.budget.credits,
        })
        if reservation > self.budget.credits:
            raise CreditLimit("Approved cap cannot cover one full provider reservation; no panels or inference started")
        self.store("config.json", config.value)
        self.panels = {}
        for world in config.value["worlds"]:
            simulator = NativeWorld(self.checkout, world)
            for seed in config.value["seeds"]:
                if world == "extra_dimensions":
                    from ..physics_extra_study import suite
                    designs, _ = suite(seed + 1000000)
                else:
                    designs = [proposal(seed + 1000000, i) for i in range(32)]
                truth = [simulator.measure(d, noise_seed(seed + 2000000, i), clean=True)
                         for i, d in enumerate(designs)]
                if world == "extra_dimensions":
                    from ..physics_extra_study import reference_check
                    checked = reference_check(self.numerics, designs, truth)
                else:
                    from ..physics_forecast_study import check_reference
                    checked = check_reference(self.numerics, world, designs, truth)
                self.panels[(world, seed)] = (designs, truth)
                self.store(f"private/{world}-{seed}.json",
                           {"designs": designs, "truth": truth, "reference_check": checked})
        self.store("native-provenance.json", self.provenance)

    def store(self, relative, value):
        for root in (self.output, self.backup):
            if root is not None:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, value)

    def run_episode(self, cell, directory):
        from decimal import Decimal
        from ..physics_comparison import grade, run_case
        from ..physics_design_advisor import NotebookDesignAdvisor
        from ..physics_panel import RecordedPrediction
        from ..physics_runtime import NativeWorld
        from ..physics_trace import export_episode
        from ..providers import usage_by_role
        arm, limits = Arm.parse(cell["arm"]), self.config.value["limits"]
        relative = directory.relative_to(self.output)
        backup = self.backup / relative if self.backup is not None else None
        before = self.budget.snapshot()
        reviewer = NotebookDesignAdvisor(self.provider, request_limit=200) if arm.advisor == "design" else None
        result = run_case(
            self.checkout, NativeWorld(self.checkout, cell["world"]), self.numerics,
            self.provider, directory, model=self.config.value["model"]["id"], seed=cell["seed"],
            request_limit=limits["requests"], observations_limit=limits["observations"],
            validate_final=True, design_advisor=reviewer, trace_directory=backup,
            action_recovery=self.config.value["parameters"].get("action_recovery", False),
            **policy_arguments(arm, limits["requests"]))
        self.store(relative / "result.json", result)
        export_episode(result, directory)
        if backup is not None:
            export_episode(result, backup)
        designs, truth = self.panels[(cell["world"], cell["seed"])]
        scores = grade(result, RecordedPrediction(
            self.numerics, directory / "private-grading",
            backup / "private-grading" if backup is not None else None),
                       designs, truth, self.config.value["parameters"]["position_scale"][cell["world"]], 1.)
        self.store(relative / "scores.json", scores)
        final = scores["final"]
        accounting = self.budget.snapshot()
        self.store("accounting.json", accounting)
        if self.backup is not None:
            self.store("credits-snapshot.json", accounting)
        new_requests = accounting["requests"][len(before["requests"]):]
        if "request_roles" in result:
            roles = result["request_roles"]
        else:
            operations = [json.loads(p.read_text()) for p in
                          sorted((directory / "operations").glob("*-model.json"))]
            roles = []
            base = str(directory.resolve())
            for operation in operations:
                key = operation["input"]["session_key"]
                if key == base:
                    roles.append("scientist")
                elif key.startswith(base + ":aux:") and key[len(base + ":aux:"):].isdigit():
                    roles.append("auxiliary")
                else:
                    raise RuntimeError("Legacy role accounting found an unknown session identity")
            if len(roles) != result["requests"]:
                raise RuntimeError("Legacy role accounting lost its original model-operation receipts")
        confirmed = sum((Decimal(v["credits"]) for k, v in
                         accounting.get("actual_settlements", {}).items()
                         if int(k) >= len(before["requests"])), Decimal(0))
        return {
            "status": result["status"], "requests": result["requests"],
            "observations": len(result["observations"]), "first_hit": scores["first_hit"],
            "capped_observations": scores["capped_observations"],
            "final_pass": bool(final and final["valid"] and final["position_error"] <= .01),
            "censored": result["error_kind"] in ("RuntimeError", "ExecutionFailure", "CreditLimit"),
            "stop_study": result["error_kind"] in ("RuntimeError", "ExecutionFailure", "CreditLimit"),
            "metrics": {"primary": "position_mse/declared_scale_squared<=0.01",
                        "position_scale": self.config.value["parameters"]["position_scale"][cell["world"]],
                        "final": final, "velocity_scale": 1., "velocity_units": "raw squared velocity"},
            "request_roles": roles, "usage_by_role": usage_by_role(result["responses"], roles),
            "accounting": {
                "accounted_credits": str(self.accounted_cost(accounting) - self.accounted_cost(before)),
                "confirmed_credits": str(confirmed),
                "unresolved_requests": sum(r["status"] == "reserved" for r in new_requests),
            },
        }

    def close(self):
        if self.provider is not None:
            self.provider.close()
