"""Trusted simulator worker. Its evaluator and law are never agent tools."""
import contextlib
import json
import math
import random
import sys

from .expression import Expression


def main():
    checkout, config = sys.argv[1], json.loads(sys.argv[2])
    sys.path.insert(0, checkout)
    # Upstream libraries may print import diagnostics; keep stdout a JSON protocol.
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from modules.m9_hooke_law.core import run_experiment_for_module
        from modules.m9_hooke_law.laws import get_ground_truth_law
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    law, _ = get_ground_truth_law(config["difficulty"], config["law_version"])
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "describe":
                result = {"ready": True}
            elif command == "measure":
                result = [float(run_experiment_for_module(
                    noise_level=config["noise"], difficulty=config["difficulty"],
                    system=config["system"], law_version=config["law_version"],
                    x=request["x"], m=request["m"],
                )) for _ in range(request["repeats"])]
                if not all(math.isfinite(v) for v in result):
                    raise ValueError("Simulator returned a nonfinite measurement")
            elif command == "grade":
                result = grade(request.get("expression"), law, config["seed"])
            else:
                raise ValueError(f"Unknown worker command: {command}")
            response = {"result": result}
        except (ValueError, TypeError, KeyError, ArithmeticError, SyntaxError) as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


def grade(expression, law, seed):
    base = {"metric": "custom-newton-predictive-v1", "official_newtonbench_score": False,
            "held_out_points": 500, "tolerance": 0.05}
    try:
        predict = Expression(expression)
        rng = random.Random(seed + 100000)
        xs = [math.exp(rng.uniform(math.log(0.001), 0)) for _ in range(500)]
        targets = [float(law(x)) for x in xs]
        predictions = [predict(x) for x in xs]
        if not all(math.isfinite(y) for y in targets):
            raise ValueError("Nonfinite evaluator target")
        # Pointwise relative error is sign-sensitive and does not discard bad predictions.
        error = math.sqrt(sum(((p - y) / max(abs(y), 1e-12)) ** 2
                              for p, y in zip(predictions, targets)) / len(xs))
        return dict(base, relative_rmse=error, passed=error < base["tolerance"])
    except (ValueError, TypeError, SyntaxError, ArithmeticError) as exc:
        return dict(base, passed=False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
