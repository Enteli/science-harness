"""Isolated worker. Never import this module to execute a candidate on the host."""
import contextlib
import copy
import functools
import hashlib
import io
import json
import math
import re
import resource
import socket
import sys
import time
from typing import Callable, Optional

import numpy as np


class CandidateError(ValueError):
    pass


class Discard:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


def finite_vector(value):
    value = np.asarray(value, dtype=float)
    if value.ndim == 2 and len(value):
        value = value[-1]
    if value.shape != (2,) or not np.isfinite(value).all():
        raise CandidateError("Law must return finite position and velocity vectors")
    return value.copy()


def protected_law(function):
    def call(*args, **kwargs):
        position, velocity = function(*copy.deepcopy(args), **copy.deepcopy(kwargs))
        return finite_vector(position), finite_vector(velocity)
    return call


def predictions(law, designs, parameters, reverse_times=False):
    result = []
    for design in designs:
        rows = []
        times = list(reversed(design["measurement_times"])) if reverse_times else design["measurement_times"]
        for timestamp in times:
            pos, vel = law(
                pos1=[0.0, 0.0], pos2=design["pos2"], p1=design["p1"],
                p2=design["p2"], velocity2=design["velocity2"],
                duration=float(timestamp), **parameters)
            rows.append({"position": pos.tolist(), "velocity": vel.tolist()})
        result.append(rows[::-1] if reverse_times else rows)
    return result


def evaluate(request):
    stage = "compile"
    law_timeout = TimeoutError
    try:
        operation = request["operation"]
        allowed = {"fit", "final_fit", "predict"}
        if operation not in allowed:
            raise CandidateError("Unknown numerical operation")
        if operation != "predict" and ("queries" in request or "truth" in request):
            raise CandidateError("Observed fitting cannot receive held-out data")
        if operation == "predict" and ("observations" in request or "truth" in request):
            raise CandidateError("Frozen prediction cannot receive observations or truth")
        source = request["numerical_source"]
        if hashlib.sha256(source.encode()).hexdigest() != request["numerical_sha256"]:
            raise CandidateError("Trusted numerical source hash mismatch")
        namespace = {"np": np, "functools": functools, "re": re, "time": time,
                     "Callable": Callable, "Optional": Optional}
        exec(compile(source, "<pinned-upstream-numerics>", "exec"), namespace)
        law_timeout = namespace["_LawCallTimeout"]
        fit_budget_exhausted = False

        class ObservedFitBudgetExceeded(namespace["_FitTimeBudgetExceeded"]):
            def __init__(self):
                nonlocal fit_budget_exhausted
                fit_budget_exhausted = True
                super().__init__()

        namespace["_FitTimeBudgetExceeded"] = ObservedFitBudgetExceeded
        law = protected_law(namespace["_compile_law"](request["source"]))
        fit_function = namespace["_compile_fit_parameters"](request["source"])
        spec = namespace["_validate_fit_spec"](fit_function()) if fit_function else []
        collisions = {name for name, _, _ in spec} & {
            "pos1", "pos2", "p1", "p2", "velocity2", "duration"}
        if collisions:
            raise CandidateError(
                "Fitted parameters collide with known law inputs: " + ", ".join(sorted(collisions)))
        declared = {name: {"init": initial, "bounds": bounds} for name, initial, bounds in spec}
        if operation == "predict":
            stage = "frozen_prediction"
            parameters = request["parameters"]
            if (not isinstance(parameters, dict)
                    or set(parameters) != {name for name, _, _ in spec}
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in parameters.values())):
                raise CandidateError("Invalid frozen fitted parameters")
            output = predictions(law, request["queries"], parameters)
            # Check order/call-state dependence without granting another fitting pass.
            reverse = predictions(law, list(reversed(request["queries"])), parameters, True)[::-1]
            if not np.array_equal(
                    np.asarray([[[row["position"], row["velocity"]] for row in case] for case in output]),
                    np.asarray([[[row["position"], row["velocity"]] for row in case] for case in reverse])):
                raise CandidateError("Predictions depend on call order or mutable candidate state")
            return {"ok": True, "stage": stage, "predictions": output,
                    "parameters": parameters, "refitted": False}
        stage = "observed_fit"
        training = copy.deepcopy(request["observations"])
        if not training:
            raise CandidateError("No observed trajectories available")
        if operation == "final_fit" and spec:
            training = namespace["_subsample_training"](training)
        initial = {name: value for name, value, _ in spec}
        loss = namespace["_two_particle_loss"]
        before = float(loss(functools.partial(law, **initial), training))
        if not math.isfinite(before):
            raise CandidateError("Nonfinite initial observed loss")
        parameters = namespace["_fit_law_parameters"](law, spec, training, loss) if spec else {}
        after = float(loss(functools.partial(law, **parameters), training))
        if not math.isfinite(after):
            raise CandidateError("Nonfinite fitted observed loss")
        return {"ok": True, "stage": stage, "loss_before": before, "loss_after": after,
                "fitted_params": parameters, "declared_params": declared,
                "n_training": len(training), "error": None,
                "fit_budget_exhausted": fit_budget_exhausted,
                "fit_policy": "upstream-position-lbfgsb",
                "training_ids": [row["id"] for row in training]}
    except (ValueError, KeyError, IndexError, TypeError, ArithmeticError, TimeoutError,
            SyntaxError, NameError, ImportError, AttributeError, OSError, law_timeout) as exc:
        return {"ok": False, "stage": stage, "failure_kind": "invalid_candidate",
                "error_kind": type(exc).__name__, "error": str(exc)[:300],
                "loss_before": None, "loss_after": None, "fitted_params": {},
                "declared_params": {}, "n_training": 0}


def probe(request):
    result = {}
    for key, action in (
        ("read_denied", lambda: open(request["forbidden"]).read()),
        ("write_denied", lambda: open(request["write_path"], "w").write("forbidden")),
        ("network_denied", lambda: socket.create_connection(("127.0.0.1", request["port"]), 1)),
    ):
        try:
            value = action()
            if hasattr(value, "close"):
                value.close()
            result[key] = False
        except PermissionError:
            result[key] = True
    return result


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (195, 195))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
        result = probe(request) if request["operation"] == "probe" else evaluate(request)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
