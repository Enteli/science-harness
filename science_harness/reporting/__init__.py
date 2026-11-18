"""Read-only aggregation; no hidden grades are sent back to investigators."""
import csv
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean


def report_study(directory):
    directory = Path(directory)
    study = json.loads((directory / "study.json").read_text())
    if (not isinstance(study, dict) or study.get("schema_version") != 1 or
            not isinstance(study.get("config"), dict) or not isinstance(study.get("rows"), list)):
        raise ValueError("Not a public-harness study manifest; use its original protocol's report")
    rows = study["rows"]
    summary = []
    for world in study["config"]["worlds"]:
        for arm in study["config"]["arms"]:
            group = [r for r in rows if r["world"] == world and r["arm"] == arm["name"]]
            eligible = [r["capped_observations"] for r in group
                        if not r.get("censored", False) and r.get("capped_observations") is not None]
            summary.append({
                "world": world, "arm": arm["name"], "assigned": len(study["config"]["seeds"]),
                "recorded": len(group), "censored": sum(r.get("censored", False) for r in group),
                "final_passes": sum(r["final_pass"] is True for r in group),
                "ungraded": sum(r["final_pass"] is None for r in group),
                "evaluation_errors": sum(r.get("evaluation_status") == "failed" for r in group),
                "mean_capped_observations": mean(eligible) if eligible else None,
                "requests": sum(r["requests"] for r in group),
                "accounted_credits": (str(sum((Decimal(r["accounting"]["accounted_credits"])
                                              for r in group), Decimal(0)))
                                      if group and all("accounting" in r for r in group) else None),
            })
    baseline = study["config"]["arms"][0]["name"]
    pairs = []
    for row in rows:
        if row["arm"] == baseline:
            continue
        reference = next((r for r in rows if r["arm"] == baseline and
                          (r["world"], r["seed"]) == (row["world"], row["seed"])), None)
        if (reference is None or row.get("censored") or reference.get("censored") or
                row.get("capped_observations") is None or reference.get("capped_observations") is None):
            continue
        pairs.append({"world": row["world"], "seed": row["seed"], "baseline": baseline,
                      "arm": row["arm"], "capped_observation_difference":
                      row["capped_observations"] - reference["capped_observations"]})
    for name, data, fields in (
            ("summary.csv", summary, ["world", "arm", "assigned", "recorded", "censored",
                                     "final_passes", "ungraded", "evaluation_errors",
                                     "mean_capped_observations", "requests", "accounted_credits"]),
            ("pairs.csv", pairs, ["world", "seed", "baseline", "arm", "capped_observation_difference"])):
        with (directory / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    role_usage = []
    for group in summary:
        selected = [r for r in rows if (r["world"], r["arm"]) == (group["world"], group["arm"])]
        roles = sorted({role for r in selected for role in r.get("usage_by_role", {})})
        for role in roles:
            data = [r["usage_by_role"][role] for r in selected if role in r.get("usage_by_role", {})]
            row = {"world": group["world"], "arm": group["arm"], "role": role,
                   "attempts": sum(r["attempts"] for r in data)}
            for field in ("input_tokens", "output_tokens"):
                row[field] = sum(r[field] for r in data) if all(r[field] is not None for r in data) else None
            role_usage.append(row)
    report = {"schema_version": 1, "status": study["status"], "summary": summary,
              "paired_baseline": baseline, "pairs": pairs,
              "usage_by_role": role_usage,
              "interpretation": "Repetitions are episodes. Metrics/normalization remain benchmark-specific; do not pool protocol epochs."}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
