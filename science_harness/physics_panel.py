"""Approved paired hosted panel; sealed grading never enters scientist feedback."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .physics_budget import PilotBudget, accounted_cost, price_envelope, upper_cost
from .physics_comparison import CONTROLLERS, MODELS, SELECTORS, grade, run_case
from .physics_reference import digest, provenance
from .physics_runtime import IsolatedNumerics, Journal, NativeWorld, atomic_json, noise_seed, proposal
from .physics_study import fresh_output, reference
from .physics_transport import CopilotProvider, authorize_terminated_timeout
from .physics_validation import check_qualification

SEEDS = {"yukawa": list(range(73142001, 73142006)),
         "oscillator": list(range(73143001, 73143006))}
LEDGER = Path(__file__).resolve().parents[1] / ".science-harness/physics-development-credits.json"
CONTINUE_APPROVAL = (
    "User authorized completion of validation and full80 within2000 cumulativecredits; "
    "continue independent cases after proven terminated timeouts with full reservations retained")


def assignments():
    rows = []
    cells = [(m, c, s) for m in MODELS for c in CONTROLLERS for s in SELECTORS]
    for pair in range(5):
        for world, seeds in SEEDS.items():
            shift = (pair + (world == "oscillator") * 4) % len(cells)
            for model, controller, selector in cells[shift:] + cells[:shift]:
                rows.append({"id": f"{world}-{seeds[pair]}-{model}-{controller}-{selector}",
                             "world": world, "seed": seeds[pair], "model": model,
                             "controller": controller, "selector": selector, "status": "pending"})
    return rows


def states(prediction):
    return np.asarray([[[r["position"], r["velocity"]] for r in rows]
                       for rows in prediction["predictions"]])


def register(checkout, qualification, validation, output):
    qualified = check_qualification(checkout, qualification)
    validated = json.loads(Path(validation).read_text())
    if (validated["provenance"] != provenance(checkout) or validated.get("stopped")
            or any(r["status"] in ("pending", "running", "not_started", "interrupted")
                   for r in validated["short_cases"] + validated["long_probes"])):
        raise RuntimeError("The registered validation stage has not finished")
    budget = PilotBudget(LEDGER, credits=2000, requests=8048).snapshot()
    if not budget.get("scientific_approval"):
        raise RuntimeError("The full scientific panel is not authorized")
    output = fresh_output(output)
    manifest = {
        "scope": "paired-hosted-scientific-panel", "provenance": provenance(checkout),
        "status": "preparing_private_panels", "assignments": assignments(),
        "requests_per_assignment": 100, "observations_per_assignment": 50,
        "evaluation_launches_per_pair": 32, "target_position_error": .01,
        "normalization": qualified["worlds"],
        "qualification_sha256": digest(Path(qualification).read_text()),
        "validation_sha256": digest(Path(validation).read_text()),
        "seed_namespaces": SEEDS,
        "panel_proposal_seed_offset": 1000000,
        "failure_policy": "Scientific nonattainers cost51; unstarted/interrupted cases remain explicitly censored",
        "execution_order": "Alternate worlds within start index; rotate8cells by index and world, fixed before outcomes",
        "final_validation": "observed-only-charged-repair-v1", "panels": {},
        "baseline_request_slots": len(budget["requests"]),
        "credit_cap": 2000, "total_request_cap": 8048,
        "no_model_requests_during_registration": True,
    }
    atomic_json(output / "registration.json", manifest)
    numerics = IsolatedNumerics(checkout)
    numerics.verify_boundary(Path(__file__))
    for world, seeds in SEEDS.items():
        native = NativeWorld(checkout, world)
        for seed in seeds:
            designs = [proposal(seed + 1000000, index) for index in range(32)]
            startup = [proposal(seed + 10000, index) for index in range(2)]
            truth = [native.measure(d, noise_seed(seed + 2000000, i), clean=True)
                     for i, d in enumerate(designs)]
            startup_truth = [native.measure(d, noise_seed(seed, i), clean=True)
                             for i, d in enumerate(startup)]
            panel = {"world": world, "seed": seed, "designs": designs, "truth": truth}
            panel_path = output / "private" / f"{world}-{seed}.json"
            atomic_json(panel_path, panel)
            expected = np.asarray([[r["pos2"], r["velocity2"]] for r in truth + startup_truth])
            expected = expected.transpose(0, 2, 1, 3)
            checked = []
            for fine in (False, True):
                result = numerics.request("predict", reference(world, fine),
                                          parameters={}, queries=designs + startup)
                if not result["ok"]:
                    atomic_json(output / "private" / f"{world}-{seed}-reference-error.json", result)
                    raise RuntimeError("Fixed reference panel failed; no replacement seeds are allowed")
                checked.append(states(result))
            error = np.max(np.abs(checked[1] - expected), axis=(0, 1, 3))
            discrepancy = np.sqrt(np.mean(np.sum((checked[0] - checked[1]) ** 2, axis=-1), axis=1))
            allowed = 1e-5 + 1e-6 * np.sqrt(np.mean(np.sum(checked[1] ** 2, axis=-1), axis=1))
            if np.any(error > 1e-4) or np.any(discrepancy > allowed):
                atomic_json(output / "private" / f"{world}-{seed}-reference-error.json",
                            {"absolute_error": error.tolist(), "discrepancy": discrepancy.tolist(),
                             "allowed": allowed.tolist()})
                raise RuntimeError("Fixed reference/refinement mismatch; do not select an easier panel")
            manifest["panels"][f"{world}-{seed}"] = {
                "path": str(panel_path.relative_to(output)), "sha256": digest(panel_path.read_text()),
                "reference_max_abs_error": error.tolist(),
                "reference_max_discrepancy": discrepancy.max(axis=0).tolist(),
            }
            atomic_json(output / "registration.json", manifest)
    manifest["status"] = "registered"
    atomic_json(output / "registration.json", manifest)
    return manifest


class RecordedPrediction:
    def __init__(self, numerics, directory, mirror=None):
        self.numerics, self.journal = numerics, Journal(directory, mirror)

    def request(self, operation, source, **payload):
        assert operation == "predict"
        request = {"operation": operation, "source": source, **payload}
        return self.journal.call("frozen-prediction", request,
                                 lambda: self.numerics.request(**request))


def run(checkout, qualification, output):
    check_qualification(checkout, qualification)
    output = Path(output).resolve()
    registration = json.loads((output / "registration.json").read_text())
    if registration["status"] != "registered" or registration["provenance"] != provenance(checkout):
        raise RuntimeError("Panel is not registered against this source")
    if (output / "panel.json").exists():
        raise RuntimeError("Do not restart or overwrite an existing scientific execution")
    report = {**registration, "status": "running"}
    atomic_json(output / "panel.json", report)
    budget = PilotBudget(LEDGER, credits=2000, requests=8048)
    numerics = IsolatedNumerics(checkout)
    with CopilotProvider(output / "catalog", credits=2000, requests=8048) as provider:
        catalog = provider.catalog()
    atomic_json(output / "catalog.json", catalog)
    models = {m["id"]: m for m in catalog["models"]}
    units = {u["modelId"]: u["unit"] for u in catalog["units"]}
    quotes = {m: price_envelope(models[m], units[m]) for m in MODELS}
    try:
        for index, row in enumerate(report["assignments"]):
            balance = budget.snapshot()
            used = accounted_cost(balance)
            if (used + upper_cost(quotes[row["model"]]) > 2000
                    or len(balance["requests"]) >= 8048):
                row.update(status="not_started_budget", reason="Full reservation does not fit cumulative cap")
                atomic_json(output / "panel.json", report)
                continue
            directory = output / row["id"]
            row["status"] = "running"
            atomic_json(output / "panel.json", report)
            before = len(balance["requests"])
            with CopilotProvider(directory / "transport", credits=2000, requests=8048) as provider:
                result = run_case(
                    checkout, NativeWorld(checkout, row["world"]), numerics, provider, directory,
                    model=row["model"], controller=row["controller"], selector=row["selector"],
                    seed=row["seed"], request_limit=100, observations_limit=50, validate_final=True)
                atomic_json(directory / "result.json", result)
            row.update(status=result["status"], error=result["error"], error_kind=result["error_kind"],
                       requests=result["requests"], observations=len(result["observations"]),
                       result_path=str((directory / "result.json").relative_to(output)),
                       provider_attempts=len(budget.snapshot()["requests"]) - before)
            if result["error"] == "RuntimeError: Model response timeout":
                row["held_dispatches"] = authorize_terminated_timeout(
                    directory / "transport", budget, CONTINUE_APPROVAL)
                row["infrastructure_censored"] = True
            elif result["error_kind"] in ("RuntimeError", "ExecutionFailure"):
                raise RuntimeError(f"Unresolved infrastructure failure: {result['error']}")
            elif result["error_kind"] == "CreditLimit":
                row["infrastructure_censored"] = True
            panel_ref = registration["panels"][f"{row['world']}-{row['seed']}"]
            panel_path = output / panel_ref["path"]
            if digest(panel_path.read_text()) != panel_ref["sha256"]:
                raise RuntimeError("Private panel changed")
            panel = json.loads(panel_path.read_text())
            normalization = registration["normalization"][row["world"]]
            scores = grade(
                result, RecordedPrediction(numerics, directory / "private-grading"),
                panel["designs"], panel["truth"],
                normalization["position_scale"], normalization["velocity_scale"])
            atomic_json(directory / "scores.json", scores)
            row["score_path"] = str((directory / "scores.json").relative_to(output))
            row["attained"] = scores["attained"]
            row["capped_observations"] = scores["capped_observations"]
            report["budget"] = budget.snapshot()
            atomic_json(output / "panel.json", report)
            print(f"{index+1}/80 {row['id']} {row['status']} attained={scores['attained']}", flush=True)
        report["status"] = "finished" if all(
            r["status"] != "not_started_budget" and not r.get("infrastructure_censored")
            for r in report["assignments"]) else "finished_with_censoring"
    except Exception as exc:
        report["status"], report["error"] = "stopped", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for row in report["assignments"]:
            if row["status"] in ("pending", "running"):
                row["status"] = "not_started" if row["status"] == "pending" else "interrupted"
        report["budget"] = budget.snapshot()
        report["status_counts"] = dict(Counter(r["status"] for r in report["assignments"]))
        atomic_json(output / "panel.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("register", "run"))
    parser.add_argument("--checkout", default=".cache/DiscoverPhysics")
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage == "register":
        if not args.validation:
            parser.error("--validation is required")
        register(args.checkout, args.qualification, args.validation, args.output)
    else:
        run(args.checkout, args.qualification, args.output)


if __name__ == "__main__":
    main()
