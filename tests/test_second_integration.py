import json
import os
from pathlib import Path

import pytest

from science_harness.harness import StudyConfig, run_study
from science_harness.harness.cli import newtonbench_config


@pytest.mark.integration
@pytest.mark.parametrize("seed", [0, 42])
def test_real_newtonbench_notebook_roundtrip_and_private_evaluation(tmp_path, seed):
    checkout = os.environ.get("NEWTONBENCH_PATH")
    if not checkout:
        pytest.skip("Set NEWTONBENCH_PATH to the clean pinned upstream checkout")
    value = newtonbench_config().value
    value["parameters"]["checkout"] = str(Path(checkout).resolve())
    value["seeds"] = [seed]
    output = tmp_path / "newton"
    outcome = run_study(StudyConfig.parse(value), output)
    assert outcome["status"] == "completed" and outcome["episodes"] == 1
    study = json.loads((output / "study.json").read_text())
    row = study["rows"][0]
    assert row["provider"] == "baseline" and row["billable_requests"] == 0
    assert row["requests"] == 9 and row["observations"] == 6 and row["notebook_reads"] == 1
    assert row["metrics"]["measurements"] == 54
    assert row["resource_budget"]["charged"]["observations"] == 6
    assert row["runtime"]["trusted_simulator_worker_drained"] is True
    assert row["metrics"]["official_newtonbench_score"] is False
    assert row["metrics"]["held_out_points"] == 500
    assert row["final_pass"] and row["metrics"]["relative_rmse"] < 0.05
    episode = next((output / "episodes").iterdir())
    result = json.loads((episode / "result.json").read_text())
    assert [event["action"]["action"] for event in result["events"][:2]] == ["note", "fetch"]
    assert len(result["evidence"]) == 6
    public = json.dumps(result["model_calls"])
    notebook = (episode / "lab-notebook.md").read_text()
    assert "Integration procedure" in notebook
    for private_field in ("relative_rmse", "held_out_points", "ground_truth_law"):
        assert private_field not in public and private_field not in notebook
