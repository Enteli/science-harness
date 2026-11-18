"""Observed-only advisory forecasts; never scientific candidates or new measurements."""
import ast
import copy
import math
import time

import numpy as np

from .physics_reference import digest
from .physics_runtime import Journal, NOISE_STD, TIMES, encode, proposal

CONTRACT = {
    "version": "recent-three-advisory-v1",
    "members": 3, "scan_entries": 12, "refit_attempts": 6, "pool_size": 8,
    "per_coordinate_noise_mse_limit": 9,
    "refit_wall_seconds": 20, "forecast_wall_seconds": 15,
    "update_wall_seconds": 180, "episode_wall_seconds": 1800,
    "maximum_updates": 12,
    "forecast_score": "mean population position variance over times and coordinates / position sigma squared",
    "scan_order": "newest actual fit first; syntactic dedup ignoring comments/docstrings; no training-error ranking",
}

PROMPT = """
ADVISORY FORECAST CALCULATIONS:
Before an ordinary scientific turn, optional advice may describe up to three
recent, syntactically distinct submitted programs, refitted to current acquired
positions in separate planning copies. Advice is not a measurement, truth,
calibrated uncertainty or a hidden score. It may be wrong. Disagreement is only
a heuristic. You remain free to choose ANY legal experiment, including one
outside the public hypothetical pool, and to disregard the advice. Forecasts
do not update your submitted laws/coefficients or select your final law.
Failed or insufficient committees explicitly abstain; do not infer physics
from missing advice. No native simulated outcomes were queried for planning.
"""


def program_key(source):
    class WithoutDocstrings(ast.NodeTransformer):
        def generic_visit(self, node):
            super().generic_visit(node)
            body = getattr(node, "body", None)
            if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:]
            return node

    tree = WithoutDocstrings().visit(ast.parse(source))
    return digest(ast.dump(tree, include_attributes=False))


def noise_mse(fit, observations):
    loss = fit.get("loss_after")
    if (not fit.get("ok") or fit.get("fit_budget_exhausted")
            or isinstance(loss, bool) or not isinstance(loss, (int, float))
            or not math.isfinite(loss) or loss < 0):
        return None
    expected_ids = [o["id"] for o in observations]
    if fit.get("training_ids") != expected_ids or fit.get("n_training") != len(expected_ids):
        raise RuntimeError("Planning fit did not use the exact current acquired prefix")
    if any(len(o["output"]["pos2"]) != len(TIMES) for o in observations):
        raise RuntimeError("Unexpected acquired measurement cadence")
    # Native loss averages squared 2D distances; the gate is per coordinate.
    return loss / (2 * NOISE_STD**2)


def rank_forecasts(forecasts, pool):
    if len(forecasts) < 2:
        return []
    positions = np.asarray([
        [[r["position"] for r in rows] for rows in member["predictions"]]
        for member in forecasts], dtype=float)
    if positions.shape != (len(forecasts), len(pool), len(TIMES), 2) or not np.isfinite(positions).all():
        raise RuntimeError("Worker returned malformed planning predictions")
    with np.errstate(over="ignore", invalid="ignore"):
        disagreement = np.var(positions, axis=0).mean(axis=(1, 2)) / NOISE_STD**2
    ranking = []
    for i, design in enumerate(pool):
        if not math.isfinite(float(disagreement[i])):
            continue
        ranking.append({
            "pool_index": i, "design": design, "disagreement": float(disagreement[i]),
            "position_forecasts": [
                {"source_sha256": member["source_sha256"],
                 "positions": [[float(f"{v:.6g}") for v in state] for state in positions[j, i]]}
                for j, member in enumerate(forecasts)],
        })
    return sorted(ranking, key=lambda row: (-row["disagreement"], row["pool_index"]))


class ForecastAdvisor:
    def __init__(self, numerics, directory, seed, clock=time.monotonic):
        self.numerics, self.journal, self.seed, self.clock = numerics, Journal(directory), seed, clock
        self.elapsed, self.updates, self.last_prefix = 0.0, 0, None
        self.records = []

    def advise(self, observations, candidates, sources):
        prefix = digest(encode({
            "observations": observations,
            "archive": list(dict.fromkeys(
                c["source_sha256"] for c in reversed(candidates) if not c["final"])),
        }))
        if prefix == self.last_prefix:
            return None
        self.last_prefix = prefix
        payload = {
            "contract": CONTRACT, "observations": copy.deepcopy(observations),
            "submissions": [copy.deepcopy(c) for c in candidates if not c["final"]],
            "sources": copy.deepcopy(sources), "seed": self.seed,
            "episode_elapsed_before": self.elapsed, "updates_before": self.updates,
        }
        record = self.journal.call("advisory-update", payload, lambda: self._compute(payload))
        self.elapsed += record["elapsed_seconds"]
        self.updates += record["performed_update"]
        self.records.append(record)
        return record["public"]

    def _compute(self, payload):
        started = self.clock()
        public = {
            "kind": "derived-planning-calculation-not-evidence", "contract": CONTRACT["version"],
            "observed_count": len(payload["observations"]), "members": [],
            "ranking": [], "status": "abstained", "failure_labels": [],
        }
        record = {"public": public, "attempts": [], "forecasts": [],
                  "performed_update": 0, "elapsed_seconds": 0.0}
        if not payload["submissions"]:
            public["reason"] = "No prior actual fit submissions"
            return record
        remaining_episode = CONTRACT["episode_wall_seconds"] - self.elapsed
        if self.updates >= CONTRACT["maximum_updates"] or remaining_episode <= 0:
            public["reason"] = "Frozen episode planning budget exhausted"
            return record
        record["performed_update"] = 1
        deadline = started + min(remaining_episode, CONTRACT["update_wall_seconds"])
        seen, accepted = set(), []
        archive = payload["submissions"][-CONTRACT["scan_entries"]:]
        for candidate in reversed(archive):
            if len(record["attempts"]) >= CONTRACT["refit_attempts"] or len(accepted) >= CONTRACT["members"]:
                break
            source_hash = candidate["source_sha256"]
            source = payload["sources"][source_hash]
            if digest(source) != source_hash:
                raise RuntimeError("Archived planning source changed")
            try:
                key = program_key(source)
            except SyntaxError:
                public["failure_labels"].append({"source_sha256": source_hash, "reason": "SyntaxError"})
                continue
            if key in seen:
                continue
            seen.add(key)
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            fit = self.numerics.request(
                "fit", source, observations=copy.deepcopy(payload["observations"]),
                wall_limit=min(CONTRACT["refit_wall_seconds"], remaining))
            normalized = noise_mse(fit, payload["observations"])
            attempt = {"source_sha256": source_hash, "syntactic_sha256": key,
                       "fit": fit, "per_coordinate_noise_mse": normalized}
            record["attempts"].append(attempt)
            if normalized is not None and normalized <= CONTRACT["per_coordinate_noise_mse_limit"]:
                accepted.append(attempt)
            else:
                public["failure_labels"].append({
                    "source_sha256": source_hash,
                    "reason": fit.get("error") if not fit.get("ok") else "Observed incompatibility gate",
                })
        pool = [proposal(self.seed + 3000000, len(payload["observations"]) * 8 + i)
                for i in range(CONTRACT["pool_size"])]
        for member in accepted:
            remaining = deadline - self.clock()
            if remaining <= 0:
                public["failure_labels"].append({"reason": "Frozen update planning time exhausted"})
                break
            source_hash = member["source_sha256"]
            predicted = self.numerics.request(
                "predict", payload["sources"][source_hash], queries=copy.deepcopy(pool),
                parameters=copy.deepcopy(member["fit"]["fitted_params"]),
                wall_limit=min(CONTRACT["forecast_wall_seconds"], remaining))
            record["forecasts"].append({"source_sha256": source_hash, "result": predicted})
            if predicted.get("ok"):
                public["members"].append({
                    "source_sha256": source_hash, "parameters": member["fit"]["fitted_params"],
                    "per_coordinate_noise_mse": member["per_coordinate_noise_mse"]})
            else:
                public["failure_labels"].append({"source_sha256": source_hash,
                                                "reason": predicted.get("error", "Forecast failure")})
        valid = [{"source_sha256": f["source_sha256"], **f["result"]}
                 for f in record["forecasts"] if f["result"].get("ok")]
        public["ranking"] = rank_forecasts(valid, pool)
        if public["ranking"]:
            public["status"] = "advisory"
        else:
            public["reason"] = "Fewer than two forecastable programs or no finite disagreements"
        record["elapsed_seconds"] = max(0.0, self.clock() - started)
        return record
