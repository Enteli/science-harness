"""Bounded development qualification/pilot. No final scientific sweep command."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .physics_comparison import CONTROLLERS, MODELS, SELECTORS, run_case
from .physics_reference import digest, provenance
from .physics_runtime import IsolatedNumerics, NativeWorld, atomic_json, noise_seed, proposal
from .physics_transport import CopilotProvider

DEVELOPMENT_SEEDS = {"yukawa": 990001, "oscillator": 990002}


def reference(world, fine=True):
    if world not in DEVELOPMENT_SEEDS:
        raise ValueError("Unregistered reference world")
    force = ("-p1/p2*k1(effective/2)/(4*np.pi)*position/radius" if world == "yukawa" else
             "-5*np.cos(np.pi*t/2)*p1/p2/(2*np.pi*effective)*position/radius")
    rtol, atol = ("1e-11", "1e-13") if fine else ("1e-9", "1e-11")
    return f"""def discovered_law(pos1,pos2,p1,p2,velocity2,duration,**params):
    import numpy as np
    from scipy.special import k1
    from scipy.integrate import solve_ivp
    def rhs(t,state):
        position=state[:2]
        radius=np.linalg.norm(position)
        effective=np.sqrt(radius*radius+0.05**2)
        acceleration={force}
        return np.concatenate([state[2:],acceleration])
    solved=solve_ivp(rhs,(0,duration),list(pos2)+list(velocity2),method='DOP853',
                     rtol={rtol},atol={atol},max_step=0.1)
    if not solved.success:
        raise ValueError(solved.message)
    return solved.y[:2,-1],solved.y[2:,-1]
"""


def fresh_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def qualify(checkout, output):
    output = fresh_output(output)
    report = {"scope": "development-only", "passed": False, "model_requests": 0,
              "provenance": provenance(checkout), "worlds": {},
              "dependencies": {name: importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "jax", "jaxlib", "requests")},
              "python": sys.version}
    started = time.monotonic()
    try:
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_physics_comparison.py",
             "tests/test_physics_budget.py", "tests/test_physics_bridge.py",
             "tests/test_physics_context.py", "tests/test_physics_validation.py",
             "tests/test_physics_panel.py"],
            text=True, capture_output=True, timeout=300)
        (output / "tests.log").write_text(tests.stdout + tests.stderr)
        if tests.returncode:
            raise RuntimeError("Targeted qualification tests failed; see tests.log")
        numerics = IsolatedNumerics(checkout)
        report["isolation"] = numerics.verify_boundary(Path(__file__))
        for world, seed in DEVELOPMENT_SEEDS.items():
            native = NativeWorld(checkout, world)
            designs = [proposal(seed + 10000, i) for i in range(2)]
            designs += [proposal(seed, i) for i in range(14)]
            truth = [native.measure(design, noise_seed(seed, i + 1), clean=True)
                     for i, design in enumerate(designs)]
            atomic_json(output / f"{world}-development.json",
                        {"seed": seed, "designs": designs, "truth": truth})
            checked = []
            for fine in (False, True):
                result = numerics.request("predict", reference(world, fine),
                                          parameters={}, queries=designs)
                if not result["ok"]:
                    atomic_json(output / f"{world}-reference-failure.json", result)
                    raise RuntimeError(f"{world} trusted reference failed")
                checked.append(np.asarray([[[r["position"], r["velocity"]] for r in rows]
                                           for rows in result["predictions"]]))
            expected = np.asarray([[row["pos2"], row["velocity2"]] for row in truth]).transpose(0, 2, 1, 3)
            error = np.max(np.abs(checked[1] - expected), axis=(0, 1, 3))
            discrepancy = np.sqrt(np.mean(np.sum((checked[0] - checked[1]) ** 2, axis=-1), axis=1))
            allowed = 1e-5 + 1e-6 * np.sqrt(np.mean(np.sum(checked[1] ** 2, axis=-1), axis=1))
            if np.any(error > 1e-4) or np.any(discrepancy > allowed):
                raise RuntimeError(f"{world} reference/refinement mismatch; fixed cases not replaced: {error}")
            ballistic = np.asarray([[np.asarray(d["pos2"]) + t * np.asarray(d["velocity2"])
                                     for t in (1, 2, 3, 4, 5)] for d in designs])
            initial_vel = np.asarray([d["velocity2"] for d in designs])[:, None, :]
            position_scale = float(np.sqrt(np.mean(np.sum((expected[:, :, 0] - ballistic) ** 2, axis=-1))))
            velocity_scale = float(np.sqrt(np.mean(np.sum((expected[:, :, 1] - initial_vel) ** 2, axis=-1))))
            if not np.isfinite([position_scale, velocity_scale]).all() or min(position_scale, velocity_scale) <= 0:
                raise RuntimeError("Degenerate development normalization")
            report["worlds"][world] = {
                "seed": seed, "launches": len(designs), "reference_max_abs_error": error.tolist(),
                "maximum_refinement_discrepancy": discrepancy.max(axis=0).tolist(),
                "position_scale": position_scale, "velocity_scale": velocity_scale,
                "data_sha256": digest((output / f"{world}-development.json").read_text()),
                "reference_source_sha256": digest(reference(world)),
            }
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "qualification.json", report)
    return report


def pilot(checkout, qualification, output, credits):
    if credits is None:
        raise ValueError("The explicit approved pilot credit cap is required")
    qualification = Path(qualification).resolve()
    evidence = json.loads(qualification.read_text())
    if not evidence["passed"] or evidence["provenance"] != provenance(checkout):
        raise RuntimeError("Qualification is missing or source hashes changed")
    current_dependencies = {name: importlib.metadata.version(name) for name in evidence["dependencies"]}
    if current_dependencies != evidence["dependencies"]:
        raise RuntimeError("Qualified numerical dependencies changed")
    output = fresh_output(output)
    cells = [{"model": model, "controller": controller, "selector": selector, "status": "pending"}
             for model in MODELS for controller in CONTROLLERS for selector in SELECTORS]
    report = {
        "scope": "development-compatibility-only", "scientific_result": False,
        "final_panel_authorized": False, "approved_credit_cap": credits, "request_cap": 16,
        "qualification_sha256": digest(qualification.read_text()), "provenance": provenance(checkout),
        "world": "yukawa", "seed": DEVELOPMENT_SEEDS["yukawa"],
        "coverage": "Two native decision turns per configuration; NOT a 50-observation investigation",
        "long_history_paid_validation": False, "cases": cells,
    }
    atomic_json(output / "pilot.json", report)
    numerics = IsolatedNumerics(checkout)
    numerics.verify_boundary(Path(__file__))
    with CopilotProvider(output / "transport", credits=credits, requests=16) as provider:
        atomic_json(output / "catalog.json", provider.catalog())
        try:
            for index, cell in enumerate(cells):
                name = f"{index:02d}-{cell['controller']}-{cell['selector']}"
                result = run_case(
                    checkout, NativeWorld(checkout, "yukawa"), numerics, provider, output / name,
                    model=cell["model"], controller=cell["controller"], selector=cell["selector"],
                    seed=report["seed"], request_limit=2)
                atomic_json(output / name / "result.json", result)
                cell.update(status=result["status"], error=result["error"], requests=result["requests"],
                            observation_count=len(result["observations"]), result_path=f"{name}/result.json",
                            candidate_count=len(result["candidates"]),
                            valid_observed_fits=sum(c["fit"].get("ok", False) for c in result["candidates"]))
                report["budget"] = provider.budget.snapshot()
                atomic_json(output / "pilot.json", report)
                print(f"{index + 1}/8 {cell['model']} {cell['controller']}/{cell['selector']}: {cell['status']}", flush=True)
                if result["error_kind"] in ("RuntimeError", "CreditLimit", "ExecutionFailure"):
                    report["stopped"] = result["error"]
                    break
        finally:
            report["budget"] = provider.budget.snapshot()
            for cell in cells:
                if cell["status"] == "pending":
                    cell["status"] = "not_started"
            report["failure_funnel"] = dict(Counter(cell["status"] for cell in cells))
            atomic_json(output / "pilot.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("qualify", "pilot"))
    parser.add_argument("--checkout", default=".cache/DiscoverPhysics")
    parser.add_argument("--output", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--approval-credits", type=float)
    args = parser.parse_args()
    if args.stage == "qualify":
        result = qualify(args.checkout, args.output)
        print(json.dumps({"passed": result["passed"], "model_requests": 0}))
    else:
        if not args.qualification:
            parser.error("--qualification is required for a paid development pilot")
        result = pilot(args.checkout, args.qualification, args.output, args.approval_credits)
        print(json.dumps({"scope": result["scope"], "failure_funnel": result["failure_funnel"]}))


if __name__ == "__main__":
    main()
