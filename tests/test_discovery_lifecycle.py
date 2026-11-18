from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from science_harness.books import BookStore
from science_harness.benchmarks.discovery import DiscoveryAdapter, NewtonDiscoveryAdapter
from science_harness.harness import StudyConfig, run_study
from science_harness.harness.cli import discovery_config
from science_harness.providers.guarded import GuardedCopilotModel
from science_harness.tools import ToolRegistry


class FixtureBudget:
    def __init__(self):
        self.state = {"requests": [], "actual_settlements": {}}

    def charge_one(self):
        index = len(self.state["requests"])
        self.state["requests"].append({
            "id": index, "status": "settled", "request_sha256": str(index), "debited_upper": "1"})
        self.state["actual_settlements"][str(index)] = {
            "credits": "1", "request_sha256": str(index)}

    def snapshot(self):
        return deepcopy(self.state)


class FixtureModel:
    model = "fixture"
    accounting = GuardedCopilotModel.accounting

    def __init__(self):
        self.budget = FixtureBudget()
        self.closed = False

    def close(self):
        self.closed = True


class FixtureScientist:
    def __init__(self, model, home, tools, **kwargs):
        self.model, self.store = model, BookStore(home)

    def investigate(self, *args, **kwargs):
        self.model.delegate.budget.charge_one()
        self.model.calls.append({"scope": "unit-fixture-not-a-provider-call"})
        result = {
            "id": "run-fixture", "status": "concluded", "steps": 1, "events": [],
            "evidence": [], "resource_budget": {},
            "conclusion": {"status": "concluded", "submission": {}}}
        self.store.save_run(result)
        return result

    def close(self):
        self.store.close()


class FixtureWorld:
    def __init__(self, fail=False):
        self.tools = ToolRegistry([])
        self.goal, self.context, self.manifest = "fixture", "fixture", {}
        self.worker = SimpleNamespace(process=SimpleNamespace(pid=1000000000, poll=lambda: 0))
        self.fail, self.closed = fail, False

    def validate_submission(self, submission):
        pass

    def grade(self, conclusion):
        if self.fail:
            raise RuntimeError("private evaluator unavailable")
        return {"passed": True}

    def close(self):
        self.closed = True


@pytest.fixture
def fixture_study(monkeypatch):
    models, worlds = [], []
    fail_seeds = set()

    def prepare(self, config, output, approved_credits):
        self.config = config
        self.model = FixtureModel()
        models.append(self.model)

    def testbed(self, cell):
        world = FixtureWorld(cell["seed"] in fail_seeds)
        worlds.append(world)
        return world

    monkeypatch.setattr(DiscoveryAdapter, "prepare", prepare)
    monkeypatch.setattr(NewtonDiscoveryAdapter, "testbed", testbed)
    monkeypatch.setattr("science_harness.agent.Scientist", FixtureScientist)
    monkeypatch.setattr("science_harness.notebook.snapshot", lambda *args: {})
    monkeypatch.setattr("science_harness.notebook.render", lambda *args: "# Fixture\n")
    return models, worlds, fail_seeds


def config(seeds):
    value = discovery_config("newtonbench-discovery").value
    value["seeds"] = seeds
    return StudyConfig.parse(value)


def test_multiple_episodes_report_cost_deltas_not_cumulative_totals(tmp_path, fixture_study):
    models, worlds, _ = fixture_study
    output = tmp_path / "study"
    run_study(config([101, 102]), output, 1500)
    study = json.loads((output / "study.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert [r["accounting"]["accounted_credits"] for r in study["rows"]] == ["1", "1"]
    assert [r["accounting"]["confirmed_credits"] for r in study["rows"]] == ["1", "1"]
    assert report["summary"][0]["accounted_credits"] == "2"
    assert models[0].accounting()["accounted_credits"] == "2"
    assert models[0].closed and all(w.closed for w in worlds)


def test_private_grader_failure_preserves_final_and_an_ungraded_row(tmp_path, fixture_study):
    models, worlds, failures = fixture_study
    failures.add(102)
    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="private evaluator unavailable"):
        run_study(config([101, 102, 103]), output, 1500)
    study = json.loads((output / "study.json").read_text())
    assert study["status"] == "failed"
    assert len(study["rows"]) == 2 and len(study["cells"]) == 3
    row = study["rows"][1]
    assert row["status"] == "evaluation_failed" and row["discovery_status"] == "concluded"
    assert row["final_pass"] is None and row["censored"] and row["stop_study"]
    episode = output / "episodes" / row["id"]
    saved = json.loads((episode / "result.json").read_text())
    assert saved["status"] == "concluded"
    assert saved["conclusion"] == {"status": "concluded", "submission": {}}
    assert saved["evaluation"]["passed"] is None
    assert "private evaluator unavailable" in saved["evaluation"]["error"]
    store = BookStore(episode / "store")
    try:
        assert store.run("run-fixture")["conclusion"] == saved["conclusion"]
    finally:
        store.close()
    summary = json.loads((output / "report.json").read_text())["summary"][0]
    assert summary["assigned"] == 3 and summary["recorded"] == 2
    assert summary["final_passes"] == 1 and summary["ungraded"] == 1
    assert summary["evaluation_errors"] == 1 and summary["censored"] == 1
    assert summary["accounted_credits"] == "2"
    assert models[0].closed and len(worlds) == 2 and all(w.closed for w in worlds)


def test_final_is_exported_before_private_evaluator_runs(tmp_path, fixture_study, monkeypatch):
    output = tmp_path / "study"
    checked = []

    def grade(self, conclusion):
        path = next((output / "episodes").glob("*/result.json"))
        saved = json.loads(path.read_text())
        assert saved["status"] == "concluded" and saved["evaluation_status"] == "pending"
        assert saved["conclusion"] == conclusion
        checked.append(True)
        return {"passed": False}

    monkeypatch.setattr(FixtureWorld, "grade", grade)
    run_study(config([101]), output, 1500)
    assert checked == [True]
    row = json.loads((output / "study.json").read_text())["rows"][0]
    assert row["final_pass"] is False and row["evaluation_status"] == "completed"
    assert not row["censored"]


def test_ungraded_verdict_cannot_hide_missing_error_and_stop_flags(
        tmp_path, fixture_study, monkeypatch):
    models, _, _ = fixture_study
    monkeypatch.setattr(DiscoveryAdapter, "run_episode", lambda *args: {
        "status": "evaluation_failed", "evaluation_status": "failed",
        "requests": 0, "observations": 0, "final_pass": None, "metrics": {},
    })
    with pytest.raises(ValueError, match="unavailable evaluation"):
        run_study(config([101]), tmp_path / "study", 1500)
    assert models[0].closed
