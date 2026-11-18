"""Eight-arm extra-dimension benchmark with persistent full public episode traces."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from .physics_budget import ExtraDimensionsBudget, accounted_cost, price_envelope, upper_cost
from .physics_comparison import CONTROLLERS, MODELS, SELECTORS, grade, run_case
from .physics_panel import RecordedPrediction
from .physics_reference import digest, provenance
from .physics_runtime import IsolatedNumerics, NativeWorld, atomic_json, noise_seed, proposal
from .physics_study import fresh_output
from .physics_trace import export_episode
from .physics_transport import CopilotProvider, authorize_terminated_timeout
from .physics_validation import check_qualification

SEEDS = list(range(73146001, 73146006))
DEVELOPMENT_SEED = 990601
RUNTIME = "1.0.103-0"
APPROVAL = "User selected original8arms,40extra_dimensions episodes,3000additionalcredits, and full persistent traces/notebooks."
REGIMES = ("near", "far")


def reference(fine=True, gravity=False):
    rtol, atol = ("1e-11", "1e-13") if fine else ("1e-9", "1e-11")
    force = ("1 / (2*np.pi*effective)" if gravity else
             "0.25*np.sum(effective/(effective**2+images**2)**1.5)")
    return f"""def discovered_law(pos1,pos2,p1,p2,velocity2,duration,**params):
    import numpy as np
    from scipy.integrate import solve_ivp
    images = np.arange(-20,21)*np.pi
    origin = np.asarray(pos1)
    def rhs(t,state):
        displacement=state[:2]-origin
        radius=np.linalg.norm(displacement)
        effective=np.sqrt(radius**2+0.05**2)
        amplitude={force}
        acceleration=-p1/p2*amplitude*displacement/radius
        return np.concatenate([state[2:],acceleration])
    solved=solve_ivp(rhs,(0,duration),list(pos2)+list(velocity2),method="DOP853",
                     rtol={rtol},atol={atol},max_step=0.05)
    if not solved.success:
        raise ValueError(solved.message)
    return solved.y[:2,-1],solved.y[2:,-1]
"""


def suite(seed, count=16):
    designs, labels = [], []
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4711]))
    for regime, low, high in (("near", .25, 1.0), ("far", 3.0, 10.0)):
        for i in range(count):
            design = proposal(seed + 1000, len(designs))
            radius, angle = rng.uniform(low, high), rng.uniform(-np.pi, np.pi)
            design["pos2"] = [float(radius*np.cos(angle)), float(radius*np.sin(angle))]
            designs.append(design)
            labels.append(regime)
    return designs, labels


def cells():
    choices = [(m, c, s) for m in MODELS for c in CONTROLLERS for s in SELECTORS]
    rows = []
    for i, seed in enumerate(SEEDS):
        order = choices[i:] + choices[:i]
        for model, controller, selector in order:
            rows.append({"id": f"{seed}-{model}-{controller}-{selector}", "seed": seed,
                         "model": model, "controller": controller, "selector": selector,
                         "status": "pending"})
    return rows


def reference_check(numerics, designs, truth):
    expected = np.asarray([[r["pos2"], r["velocity2"]] for r in truth]).transpose(0, 2, 1, 3)
    checked = []
    for fine in (False, True):
        response = numerics.request("predict", reference(fine), parameters={}, queries=designs)
        if not response["ok"]:
            raise RuntimeError(f"Extra-dimension reference failed: {response}")
        checked.append(np.asarray([[[r["position"], r["velocity"]] for r in case]
                                   for case in response["predictions"]]))
    error = np.max(np.abs(checked[1] - expected), axis=(0, 1, 3))
    discrepancy = np.sqrt(np.mean(np.sum((checked[0]-checked[1])**2, axis=-1), axis=1))
    allowed = 1e-5 + 1e-6*np.sqrt(np.mean(np.sum(checked[1]**2, axis=-1), axis=1))
    if np.any(error > 1e-4) or np.any(discrepancy > allowed):
        raise RuntimeError(f"Extra-dimension fixed reference/refinement mismatch: {error.tolist()}")
    return {"maximum_absolute_error": error.tolist(),
            "maximum_refinement_discrepancy": discrepancy.max(axis=0).tolist()}


def position_mse(prediction, truth):
    target = np.asarray([r["pos2"] for r in truth])
    predicted = np.asarray([[r["position"] for r in case] for case in prediction["predictions"]])
    return float(np.mean(np.sum((target-predicted)**2, axis=-1)))


def regime_scores(scores, candidates, directory, panel, position_scale):
    receipts = iter(sorted(Path(directory).glob("*-frozen-prediction.json")))
    target = np.asarray([r["pos2"] for r in panel["truth"]])
    labels = np.asarray(panel["regimes"])
    for scored, candidate in zip(scores["candidates"], candidates):
        if not candidate["fit"].get("ok"):
            continue
        record = json.loads(next(receipts).read_text())
        if digest(record["input"]["source"]) != scored["source_sha256"]:
            raise RuntimeError("Frozen prediction receipt no longer matches candidate")
        prediction = record["result"]
        if not prediction["ok"]:
            continue
        predicted = np.asarray([[r["position"] for r in rows] for rows in prediction["predictions"]])
        with np.errstate(over="ignore", invalid="ignore"):
            errors = np.sum((predicted-target)**2, axis=-1)
        values = {regime: float(np.mean(errors[labels == regime]))/position_scale**2 for regime in REGIMES}
        if not np.isfinite(list(values.values())).all():
            scored["regime_error"] = "Nonfinite squared prediction error"
        else:
            scored["regime_position_error"] = values
            scored["both_regimes_target"] = all(value <= .01 for value in values.values())
    if next(receipts, None) is not None:
        raise RuntimeError("Unexpected extra prediction receipts")


def qualify(checkout, output):
    output = fresh_output(output)
    report = {
        "scope": "extra-dimension-disjoint-development", "passed": False, "model_requests": 0,
        "provenance": provenance(checkout),
        "dependencies": {n: importlib.metadata.version(n)
                         for n in ("numpy", "scipy", "jax", "jaxlib", "requests", "tiktoken")},
        "development_seed": DEVELOPMENT_SEED,
    }
    try:
        files = [
            "tests/test_physics_extra_study.py", "tests/test_physics_trace.py",
            "tests/test_physics_budget.py", "tests/test_physics_comparison.py",
            "tests/test_physics_context.py", "tests/test_physics_bridge.py",
            "tests/test_physics_panel.py", "tests/test_physics_planning.py",
            "tests/test_physics_forecast_study.py", "tests/test_physics_validation.py",
        ]
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q", *files],
                               text=True, capture_output=True, timeout=300)
        (output / "tests.log").write_text(tests.stdout + tests.stderr)
        if tests.returncode:
            raise RuntimeError("Qualification tests failed; see tests.log")
        numerics, native = IsolatedNumerics(checkout), NativeWorld(checkout, "extra_dimensions")
        report["isolation"] = numerics.verify_boundary(Path(__file__))
        designs, labels = suite(DEVELOPMENT_SEED, count=8)
        truth = [native.measure(d, noise_seed(DEVELOPMENT_SEED, i), clean=True)
                 for i, d in enumerate(designs)]
        atomic_json(output / "development.json", {"designs": designs, "regimes": labels, "truth": truth})
        report["reference"] = reference_check(numerics, designs, truth)
        targets = np.asarray([r["pos2"] for r in truth])
        ballistic = np.asarray([[np.asarray(d["pos2"])+t*np.asarray(d["velocity2"])
                                 for t in (1,2,3,4,5)] for d in designs])
        velocity = np.asarray([r["velocity2"] for r in truth])
        initial = np.asarray([d["velocity2"] for d in designs])[:,None,:]
        position_scale = float(np.sqrt(np.mean(np.sum((targets-ballistic)**2, axis=-1))))
        velocity_scale = float(np.sqrt(np.mean(np.sum((velocity-initial)**2, axis=-1))))
        if not np.isfinite([position_scale, velocity_scale]).all() or min(position_scale, velocity_scale) <= 0:
            raise RuntimeError("Degenerate independent development scales")
        wrong = numerics.request("predict", reference(gravity=True), parameters={}, queries=designs)
        if not wrong["ok"]:
            raise RuntimeError("Gravity negative control is not executable")
        report["normalization"] = {"position_scale": position_scale, "velocity_scale": velocity_scale,
                                   "calibration_sha256": digest((output/"development.json").read_text())}
        report["gravity_negative_control_error"] = position_mse(wrong, truth)/position_scale**2
        if report["gravity_negative_control_error"] <= .01:
            raise RuntimeError("Evaluation cannot distinguish the wrong gravity model; no paid run")
        report["passed"] = True
    finally:
        atomic_json(output / "qualification.json", report)
    return report


def register(checkout, qualification, output, archive):
    q = check_qualification(checkout, qualification)
    output, archive = fresh_output(output), fresh_output(archive)
    shutil.copytree(Path(qualification).parent, archive/"qualification")
    manifest = {
        "scope": "extra-dimensions-eight-arms-v1", "status": "preparing",
        "provenance": provenance(checkout), "world": "extra_dimensions",
        "noise_std": .15089155792810338, "runtime": RUNTIME,
        "assignments": cells(), "seeds": SEEDS, "panels": {}, "normalization": q["normalization"],
        "requests_per_episode": 100, "observations_per_episode": 50,
        "panels_per_pair": 32, "near_launches": 16, "far_launches": 16,
        "near_initial_radius": [.25,1.0], "far_initial_radius": [3.0,10.0],
        "primary": "First frozen candidate with overall normalized position MSE<=0.01; nonattainers cost51",
        "regime_guard": "Report near/far scores separately; do not certify a crossover from overall MSE alone",
        "domain_amendment": "Exactly coincident source/probe initial coordinates rejected explicitly before execution",
        "credit_cap": 3000, "total_request_cap": 4000,
        "qualification_sha256": digest(Path(qualification).read_text()),
        "persistent_archive": str(archive),
    }
    numerics, native = IsolatedNumerics(checkout), NativeWorld(checkout, "extra_dimensions")
    atomic_json(output/"registration.json", manifest)
    atomic_json(archive/"registration.json", manifest)
    for seed in SEEDS:
        designs, labels = suite(seed+1000000)
        truth = [native.measure(d, noise_seed(seed+2000000,i), clean=True) for i,d in enumerate(designs)]
        panel = {"designs": designs, "regimes": labels, "truth": truth}
        path = output/"private"/f"{seed}.json"
        atomic_json(path,panel)
        atomic_json(archive/"private-panels"/f"{seed}.json",panel)
        checked = reference_check(numerics,designs,truth)
        startup = [proposal(seed+10000,i) for i in range(2)]
        startup_truth = [native.measure(d,noise_seed(seed,i+1),clean=True) for i,d in enumerate(startup)]
        reference_check(numerics,startup,startup_truth)
        wrong = numerics.request("predict",reference(gravity=True),parameters={},queries=designs)
        if not wrong["ok"]:
            raise RuntimeError("Fixed gravity negative control could not execute")
        wrong_error=position_mse(wrong,truth)/q["normalization"]["position_scale"]**2
        if wrong_error<=.01:
            raise RuntimeError("A fixed panel permits the wrong gravity explanation; no replacement seeds")
        manifest["panels"][str(seed)]={"path":str(path.relative_to(output)),"sha256":digest(path.read_text()),
                                      "gravity_negative_control_error":wrong_error,**checked}
        atomic_json(output/"registration.json",manifest)
        atomic_json(archive/"registration.json",manifest)
    budget=ExtraDimensionsBudget(archive/"budget.json")
    budget.authorize(APPROVAL)
    manifest["status"]="registered"
    atomic_json(output/"registration.json",manifest)
    atomic_json(archive/"registration.json",manifest)
    return manifest


def run(checkout,qualification,output):
    check_qualification(checkout, qualification)
    output = Path(output).resolve()
    registered = json.loads((output / "registration.json").read_text())
    if registered["status"] != "registered" or registered["provenance"] != provenance(checkout):
        raise RuntimeError("Study registration differs from this source")
    if (output / "study.json").exists():
        raise RuntimeError("Do not overwrite or restart a scientific study")
    archive = Path(registered["persistent_archive"])
    budget = ExtraDimensionsBudget(archive / "budget.json")
    report = {**registered, "status": "running"}

    def persist():
        atomic_json(output / "study.json", report)
        atomic_json(archive / "study.json", report)

    persist()
    numerics = IsolatedNumerics(checkout)
    with CopilotProvider(output / "catalog", budget=budget, pinned_version=RUNTIME) as provider:
        catalog = provider.catalog()
    atomic_json(archive / "catalog.json", catalog)
    units = {u["modelId"]: u["unit"] for u in catalog["units"]}
    quotes = {m["id"]: price_envelope(m, units[m["id"]])
              for m in catalog["models"] if m["id"] in MODELS}
    try:
        for i, row in enumerate(report["assignments"]):
            state = budget.snapshot()
            if accounted_cost(state) + upper_cost(quotes[row["model"]]) > 3000:
                row.update(status="not_started_budget", infrastructure_censored=True)
                persist()
                continue
            directory = output / row["id"]
            durable = archive / "episodes" / row["id"]
            row["status"] = "running"
            persist()
            with CopilotProvider(directory / "transport", budget=budget, pinned_version=RUNTIME,
                                 receipt_directory=durable / "receipts") as provider:
                result = run_case(
                    checkout, NativeWorld(checkout, "extra_dimensions"), numerics, provider, directory,
                    model=row["model"], controller=row["controller"], selector=row["selector"],
                    seed=row["seed"], request_limit=100, observations_limit=50, validate_final=True,
                    trace_directory=durable)
                atomic_json(directory / "result.json", result)
                atomic_json(durable / "result.json", result)
            row.update(status=result["status"], error=result["error"], error_kind=result["error_kind"],
                       requests=result["requests"], observations=len(result["observations"]),
                       result_path=str((directory / "result.json").relative_to(output)),
                       persistent_episode=str(durable),
                       provider_attempts=len(budget.snapshot()["requests"]) - len(state["requests"]))
            row["trace"] = export_episode(result, durable)
            if result["error"] == "RuntimeError: Model response timeout":
                row["held_dispatches"] = authorize_terminated_timeout(
                    directory / "transport", budget, APPROVAL + " Retain full terminated timeout holds.")
                row["infrastructure_censored"] = True
            elif result["error_kind"] in ("RuntimeError", "ExecutionFailure"):
                row["infrastructure_censored"] = True
                raise RuntimeError(result["error"])
            elif result["error_kind"] == "CreditLimit":
                row["infrastructure_censored"] = True
            panel_ref = registered["panels"][str(row["seed"])]
            panel_path = output / panel_ref["path"]
            if digest(panel_path.read_text()) != panel_ref["sha256"]:
                raise RuntimeError("Private panel changed")
            panel = json.loads(panel_path.read_text())
            scores = grade(
                result, RecordedPrediction(numerics, directory / "private-grading", durable / "private-grading"),
                panel["designs"], panel["truth"],
                         registered["normalization"]["position_scale"],
                         registered["normalization"]["velocity_scale"])
            regime_scores(scores, result["candidates"], directory / "private-grading", panel,
                          registered["normalization"]["position_scale"])
            atomic_json(directory / "scores.json", scores)
            atomic_json(durable / "scores.json", scores)
            row["score_path"] = str((directory / "scores.json").relative_to(output))
            persist()
            print(f"{i+1}/40 {row['id']} {row['status']}", flush=True)
        report["status"] = "finished_with_censoring" if any(
            r.get("infrastructure_censored") for r in report["assignments"]) else "finished"
    except Exception as exc:
        report["status"], report["error"] = "stopped", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for row in report["assignments"]:
            if row["status"] in ("pending", "running"):
                row["status"] = "not_started" if row["status"] == "pending" else "interrupted"
        report["accounted_credits_with_holds"] = str(accounted_cost(budget.snapshot()))
        report["status_counts"] = dict(Counter(r["status"] for r in report["assignments"]))
        persist()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("qualify","register","run"))
    p.add_argument("--checkout",default=".cache/DiscoverPhysics")
    p.add_argument("--qualification")
    p.add_argument("--archive")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if args.stage=="qualify":
        qualify(args.checkout,args.output)
    elif not args.qualification:
        p.error("--qualification is required")
    elif args.stage=="register":
        if not args.archive:
            p.error("--archive is required for persistent traces")
        register(args.checkout,args.qualification,args.output,args.archive)
    else:
        run(args.checkout,args.qualification,args.output)


if __name__=="__main__":
    main()
