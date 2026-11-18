"""Deterministic reference policies, not claims of autonomous scientific discovery."""
import json
import math
import statistics

from .context import expand_observations
from .tools import completed_acquisitions


def tool(name, args, why, discriminates):
    return {"action": "tool", "tool": name, "args": args, "why": why,
            "discriminates": discriminates, "assumptions": []}


def observations(request):
    if "observations" in request or "observation_table" in request:
        observed = expand_observations(request)
        if any(e.get("data_omitted") or e.get("observation", {}).get("truncated") for e in observed):
            raise ValueError("Reference policy requires its small acquisition dataset to fit; "
                             "raise the context budget rather than silently repeat measurements")
        return observed
    return list(completed_acquisitions(request["events"]))


def readout_statistic(observation, field, statistic):
    record = observation["observation"]
    if field in record.get("readout_summaries", {}):
        return record["readout_summaries"][field][statistic]
    readings = record["data"][field]
    return len(readings) if statistic == "n" else getattr(statistics, statistic)(readings)


def finish(summary, evidence, submission, status="concluded", uncertainties=None):
    return {"action": "finish", "status": status, "summary": summary,
            "evidence_ids": evidence, "submission": submission, "assumptions": [],
            "alternatives": [], "uncertainties": uncertainties or [], "updates": []}


class SpringBaseline:
    """Control-calibrated median estimator, or deliberately naive one-point comparator."""
    def __init__(self, naive=False):
        self.naive = naive

    def complete(self, system, user):
        request = json.loads(user)
        obs = observations(request)
        ids = [o["evidence_id"] for o in obs]
        available = "No reference spring" not in request["apparatus"]
        design = [("spring", 0.01)] * 5 if self.naive else [
            ("blank", 0), ("spring", -0.04), ("spring", 0.04),
        ] + ([("reference", -0.04), ("reference", 0.04)] if available else
             [("spring", -0.02), ("spring", 0.02)])
        if len(obs) < len(design):
            specimen, x = design[len(obs)]
            action = tool("measure", {"specimen": specimen, "extension_m": x, "repeats": 15},
                          "Estimate response and distinguish fixed offset from slope.",
                          "Blank and reference results can invalidate a direct stiffness estimate.")
        elif self.naive:
            y = sum(readout_statistic(o, "force_n", "mean") * readout_statistic(o, "force_n", "n")
                    for o in obs) / sum(readout_statistic(o, "force_n", "n") for o in obs)
            action = finish("Uncorrected one-point estimate; assumes calibrated scale and zero bias.",
                            ids, {"stiffness_n_per_m": y / 0.01, "identifiable": True})
            action["assumptions"] = ["Nominal extension equals actual extension", "Zero force offset"]
        elif not available:
            action = finish(
                "Intrinsic stiffness and extension scale are confounded. An independent "
                "extension calibration or known reference spring is required.", ids,
                {"identifiable": False}, "inconclusive",
                ["Only instrument scale times specimen stiffness can be identified."],
            )
        else:
            means = [readout_statistic(o, "force_n", "median") for o in obs]
            scale = (means[4] - means[3]) / (0.08 * 100)
            if scale <= 0:
                action = finish("Reference calibration failed.", ids, {}, "inconclusive",
                                ["Reference response does not identify a positive scale."])
            else:
                stiffness = (means[2] - means[1]) / (0.08 * scale)
                action = finish(
                    "Estimated intrinsic stiffness using signed extensions, a blank and a "
                    "known reference. Replicate medians limit isolated outlier influence.",
                    ids, {"stiffness_n_per_m": stiffness, "offset_n": means[0],
                          "extension_scale": scale, "identifiable": True},
                    uncertainties=["No interval coverage claimed; finite repeated observations."],
                )
                action["assumptions"] = ["Specimens share a fixed affine instrument response",
                                         "Reference stiffness is accurately known"]
                action["updates"] = [{
                    "book": "playbook", "reason": "Preserve the evaluated calibration procedure.",
                    "content": {
                        "title": "Separate instrument gain and offset from specimen response",
                        "when": "An affine measurement instrument may bias a linear response",
                        "guidance": "Use a blank and a known reference at signed inputs; "
                                    "without an independent scale, report confounding.",
                        "scope": "Linear response with shared, stable affine instrument",
                        "evidence_ids": ids,
                        "assumptions": action["assumptions"],
                        "uncertainties": ["Not evaluated for drift or nonlinear instruments"],
                    },
                }]
        return json.dumps(action)


class NewtonPowerBaseline:
    """Fit a power law from observations; restricted to easy vanilla tasks by the CLI."""
    def complete(self, system, user):
        request = json.loads(user)
        obs = observations(request)
        xs = [0.002, 0.007, 0.025, 0.08, 0.25, 0.8]
        if len(obs) < len(xs):
            action = tool("run_experiment", {"x": xs[len(obs)], "m": 1, "repeats": 9},
                          "Measure energy over multiple scales.",
                          "Curvature in log-space would contradict a single power law.")
        else:
            ys = [readout_statistic(o, "readings", "median") for o in obs]
            ids = [o["evidence_id"] for o in obs]
            if any(y <= 0 for y in ys):
                action = finish("Nonpositive energy prevents this power-law estimator.",
                                ids, {}, "inconclusive")
            else:
                lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
                mx, my = statistics.mean(lx), statistics.mean(ly)
                power = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum(
                    (x - mx)**2 for x in lx
                )
                coefficient = math.exp(my - power * mx)
                action = finish("Fitted a power law to median readings across scales.", ids,
                                {"expression": f"{coefficient:.14g} * x ** {power:.14g}"},
                                uncertainties=["Assumes a single positive power law; "
                                               "does not handle transformed system outputs."])
                action["assumptions"] = ["Positive power-law energy in the sampled interval"]
        return json.dumps(action)
