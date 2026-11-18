"""Fresh20-episode forecast advisory comparison with a separate approved budget."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .physics_budget import ForecastBudget, accounted_cost, price_envelope, upper_cost
from .physics_comparison import grade, run_case
from .physics_panel import RecordedPrediction
from .physics_planning import CONTRACT, ForecastAdvisor
from .physics_reference import digest, provenance
from .physics_runtime import IsolatedNumerics, NativeWorld, atomic_json, noise_seed, proposal
from .physics_study import fresh_output, reference
from .physics_transport import CopilotProvider, authorize_terminated_timeout
from .physics_validation import check_qualification

SEEDS = {"yukawa": list(range(73144001, 73144006)),
         "oscillator": list(range(73145001, 73145006))}
DEVELOPMENT = {"yukawa": 990301, "oscillator": 990302}
MODEL = "gpt-6.1-sol"
RUNTIME = "1.0.103-0"
BUDGET_PATH = Path(__file__).resolve().parents[1] / ".science-harness/physics-forecast-credits.json"
APPROVAL = "User selected GPT-6.1 Sol only20episodes and3000additionalAIcredits for the two-arm forecast comparison."


def cells():
    rows = []
    for pair in range(5):
        for world, seeds in SEEDS.items():
            arms = ["baseline", "forecast"] if (pair + (world == "oscillator")) % 2 == 0 else [
                "forecast", "baseline"]
            for arm in arms:
                rows.append({"id": f"{world}-{seeds[pair]}-{arm}", "world": world,
                             "seed": seeds[pair], "arm": arm, "status": "pending"})
    return rows


def check_reference(numerics, world, designs, truth):
    expected = np.asarray([[r["pos2"], r["velocity2"]] for r in truth]).transpose(0, 2, 1, 3)
    checked = []
    for fine in (False, True):
        result = numerics.request("predict", reference(world, fine), parameters={}, queries=designs)
        if not result["ok"]:
            raise RuntimeError(f"Fixed {world} reference failed: {result}")
        checked.append(np.asarray([[[r["position"], r["velocity"]] for r in rows]
                                   for rows in result["predictions"]]))
    error = np.max(np.abs(checked[1] - expected), axis=(0, 1, 3))
    discrepancy = np.sqrt(np.mean(np.sum((checked[0] - checked[1])**2, axis=-1), axis=1))
    allowed = 1e-5 + 1e-6 * np.sqrt(np.mean(np.sum(checked[1]**2, axis=-1), axis=1))
    if np.any(error > 1e-4) or np.any(discrepancy > allowed):
        raise RuntimeError(f"Fixed {world} reference/refinement mismatch; do not replace cases")
    return {"reference_max_abs_error": error.tolist(),
            "maximum_refinement_discrepancy": discrepancy.max(axis=0).tolist()}


def qualify(checkout, output):
    output = fresh_output(output)
    report = {
        "scope": "disjoint-forecast-development", "passed": False, "model_requests": 0,
        "provenance": provenance(checkout), "planning_contract": CONTRACT,
        "dependencies": {n: importlib.metadata.version(n)
                         for n in ("numpy", "scipy", "jax", "jaxlib", "requests", "tiktoken")},
        "worlds": {},
    }
    try:
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q",
                                "tests/test_physics_planning.py", "tests/test_physics_budget.py",
                                "tests/test_physics_comparison.py", "tests/test_physics_panel.py",
                                "tests/test_physics_context.py", "tests/test_physics_bridge.py",
                                "tests/test_physics_validation.py", "tests/test_physics_forecast_study.py"],
                               capture_output=True, text=True, timeout=300)
        (output / "tests.log").write_text(tests.stdout + tests.stderr)
        if tests.returncode:
            raise RuntimeError("Forecast qualification failed; inspect tests.log")
        numerics = IsolatedNumerics(checkout)
        report["isolation"] = numerics.verify_boundary(Path(__file__))
        for world, seed in DEVELOPMENT.items():
            native = NativeWorld(checkout, world)
            designs = [proposal(seed, i) for i in range(16)]
            truth = [native.measure(d, noise_seed(seed, i), clean=True) for i, d in enumerate(designs)]
            metrics = check_reference(numerics, world, designs, truth)
            expected = np.asarray([r["pos2"] for r in truth])
            ballistic = np.asarray([[np.asarray(d["pos2"]) + t * np.asarray(d["velocity2"])
                                     for t in (1, 2, 3, 4, 5)] for d in designs])
            vel = np.asarray([r["velocity2"] for r in truth])
            v0 = np.asarray([d["velocity2"] for d in designs])[:, None, :]
            pos_scale = float(np.sqrt(np.mean(np.sum((expected - ballistic)**2, axis=-1))))
            vel_scale = float(np.sqrt(np.mean(np.sum((vel - v0)**2, axis=-1))))
            if not np.isfinite([pos_scale, vel_scale]).all() or min(pos_scale, vel_scale) <= 0:
                raise RuntimeError("Degenerate independent development normalization")
            observed = [{"id": f"experiment-{i+1:04d}", "input": d,
                         "output": native.measure(d, noise_seed(seed, i))}
                        for i, d in enumerate(designs[:4])]
            # Numerical sanity only; these trusted references never enter study committees.
            clean_source = reference(world)
            bad_source = clean_source.replace("return solved.y[:2,-1],", "return solved.y[:2,-1] + 10,")
            archive = [{"source_sha256": digest(s), "final": False}
                       for s in (bad_source, clean_source)]
            sources = {digest(s): s for s in (bad_source, clean_source)}
            advisor = ForecastAdvisor(numerics, output / f"{world}-advisory-sanity", seed)
            advice = advisor.advise(observed, archive, sources)
            records = advisor.records[0]["attempts"]
            if (not records or records[0]["source_sha256"] != digest(clean_source)
                    or not any(m["source_sha256"] == digest(clean_source) for m in advice["members"])
                    or any(m["source_sha256"] == digest(bad_source) for m in advice["members"])):
                raise RuntimeError("Predeclared noise gate failed development sanity; no paid calls")
            atomic_json(output / f"{world}-development.json",
                        {"designs": designs, "truth": truth, "observed_sanity_prefix": observed})
            report["worlds"][world] = {
                **metrics, "seed": seed, "launches": 16,
                "position_scale": pos_scale, "velocity_scale": vel_scale,
                "sanity_correct_source_retained": True, "sanity_offset_source_rejected": True,
            }
        report["passed"] = True
    finally:
        atomic_json(output / "qualification.json", report)
    return report


def register(checkout, qualification, output):
    q = check_qualification(checkout, qualification)
    if q.get("planning_contract") != CONTRACT:
        raise RuntimeError("Planning limits changed since qualification")
    state = ForecastBudget(BUDGET_PATH).snapshot()
    if state["requests"]:
        raise RuntimeError("Register the fresh panel before any study dispatch")
    output = fresh_output(output)
    manifest = {
        "scope": "forecast-advisory-gpt20-v1", "status": "preparing",
        "provenance": provenance(checkout), "model": MODEL, "runtime": RUNTIME,
        "planning_contract": CONTRACT, "normalization": q["worlds"], "seeds": SEEDS,
        "assignments": cells(), "panels": {}, "request_limit": 100, "observations_limit": 50,
        "additional_credit_cap": 3000, "total_request_cap": 2000,
        "primary": "paired difference in capped observations to first normalized position error<=0.01",
        "practical_effect": "Forecast minus baseline mean<=-2observations separately per world; "
                            "report final target attainment and compute/inference cost alongside it",
        "interpretation": "Effect of the complete advisory package, not ranking independent of extra refits/filtering",
        "qualification_sha256": digest(Path(qualification).read_text()),
    }
    atomic_json(output / "registration.json", manifest)
    numerics = IsolatedNumerics(checkout)
    for world, seeds in SEEDS.items():
        native = NativeWorld(checkout, world)
        for seed in seeds:
            designs = [proposal(seed + 1000000, i) for i in range(32)]
            truth = [native.measure(d, noise_seed(seed + 2000000, i), clean=True)
                     for i, d in enumerate(designs)]
            startup = [proposal(seed + 10000, i) for i in range(2)]
            startup_truth = [native.measure(d, noise_seed(seed, i), clean=True) for i, d in enumerate(startup)]
            checked = check_reference(numerics, world, designs + startup, truth + startup_truth)
            panel = output / "private" / f"{world}-{seed}.json"
            atomic_json(panel, {"designs": designs, "truth": truth})
            manifest["panels"][f"{world}-{seed}"] = {
                "path": str(panel.relative_to(output)), "sha256": digest(panel.read_text()), **checked}
            atomic_json(output / "registration.json", manifest)
    manifest["status"] = "registered"
    atomic_json(output / "registration.json", manifest)
    return manifest


def run(checkout, qualification, output):
    check_qualification(checkout, qualification)
    output = Path(output).resolve()
    registered = json.loads((output / "registration.json").read_text())
    if registered["status"] != "registered" or registered["provenance"] != provenance(checkout):
        raise RuntimeError("Study source differs from its fixed registration")
    if (output / "study.json").exists():
        raise RuntimeError("Refusing to overwrite/restart an existing study")
    report = {**registered, "status": "running"}
    atomic_json(output / "study.json", report)
    budget, numerics = ForecastBudget(BUDGET_PATH), IsolatedNumerics(checkout)
    with CopilotProvider(output / "catalog", budget=budget, pinned_version=RUNTIME) as provider:
        catalog = provider.catalog()
    atomic_json(output / "catalog.json", catalog)
    model = next(m for m in catalog["models"] if m["id"] == MODEL)
    unit = next(u["unit"] for u in catalog["units"] if u["modelId"] == MODEL)
    quote = price_envelope(model, unit)
    try:
        for i, row in enumerate(report["assignments"]):
            balance = budget.snapshot()
            if accounted_cost(balance) + upper_cost(quote) > budget.credits:
                row.update(status="not_started_budget", infrastructure_censored=True)
                atomic_json(output / "study.json", report)
                continue
            directory = output / row["id"]
            advisor = ForecastAdvisor(numerics, directory / "planning", row["seed"]) if row["arm"] == "forecast" else None
            row["status"] = "running"
            atomic_json(output / "study.json", report)
            with CopilotProvider(directory / "transport", budget=budget, pinned_version=RUNTIME) as provider:
                result = run_case(
                    checkout, NativeWorld(checkout, row["world"]), numerics, provider, directory,
                    model=MODEL, controller="science", selector="active", seed=row["seed"],
                    request_limit=100, observations_limit=50, validate_final=True, advisor=advisor)
                atomic_json(directory / "result.json", result)
            row.update(status=result["status"], error=result["error"], error_kind=result["error_kind"],
                       requests=result["requests"], observations=len(result["observations"]),
                       result_path=str((directory / "result.json").relative_to(output)),
                       provider_attempts=len(budget.snapshot()["requests"]) - len(balance["requests"]))
            if result["error"] == "RuntimeError: Model response timeout":
                row["held_dispatches"] = authorize_terminated_timeout(
                    directory / "transport", budget, APPROVAL + " Retain full terminated timeout holds.")
                row["infrastructure_censored"] = True
            elif result["error_kind"] in ("RuntimeError", "ExecutionFailure"):
                row["infrastructure_censored"] = True
                raise RuntimeError(result["error"])
            elif result["error_kind"] == "CreditLimit":
                row["infrastructure_censored"] = True
            panel_ref = registered["panels"][f"{row['world']}-{row['seed']}"]
            path = output / panel_ref["path"]
            if digest(path.read_text()) != panel_ref["sha256"]:
                raise RuntimeError("Private evaluation data changed")
            private = json.loads(path.read_text())
            scale = registered["normalization"][row["world"]]
            scores = grade(result, RecordedPrediction(numerics, directory / "private-grading"),
                           private["designs"], private["truth"], scale["position_scale"], scale["velocity_scale"])
            atomic_json(directory / "scores.json", scores)
            row["score_path"] = str((directory / "scores.json").relative_to(output))
            atomic_json(output / "study.json", report)
            print(f"{i+1}/20 {row['id']} {row['status']}", flush=True)
        report["status"] = "finished_with_censoring" if any(
            r.get("infrastructure_censored") for r in report["assignments"]) else "finished"
    except Exception as exc:
        report["status"], report["error"] = "stopped", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for row in report["assignments"]:
            if row["status"] in ("pending", "running"):
                row["status"] = "not_started" if row["status"] == "pending" else "interrupted"
        report["budget"] = budget.snapshot()
        report["accounted_credits_with_holds"] = str(accounted_cost(report["budget"]))
        report["status_counts"] = dict(Counter(r["status"] for r in report["assignments"]))
        atomic_json(output / "study.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify", "register", "run"))
    parser.add_argument("--checkout", default=".cache/DiscoverPhysics")
    parser.add_argument("--qualification")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage == "qualify":
        qualify(args.checkout, args.output)
    elif not args.qualification:
        parser.error("--qualification is required")
    elif args.stage == "register":
        register(args.checkout, args.qualification, args.output)
    else:
        run(args.checkout, args.qualification, args.output)


if __name__ == "__main__":
    main()
