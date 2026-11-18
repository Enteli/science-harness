import json
import os
from pathlib import Path

import pytest

from science_harness.harness import StudyConfig, run_study
from science_harness.harness.cli import discovery_config
from science_harness.physics_budget import CreditLimit, accounted_cost
from science_harness.providers.guarded import StudyBudget
from science_harness.testbeds.made import Made


QUOTE = {"model": "fixture", "input_price": "10", "output_price": "10",
         "batch": "1000", "input_limit": "100", "output_limit": "100"}


def test_guarded_transport_hashes_unicode_like_the_sdk(tmp_path):
    from decimal import Decimal
    import hashlib
    from science_harness.providers.guarded import GuardedCopilotModel
    model = object.__new__(GuardedCopilotModel)
    model.directory, model.index, model.model, model.quote = tmp_path, 0, "fixture", {}
    packet = {"model": "fixture", "system": "Grade the operator \u2207\u00b2", "user": "Measure \u03bb"}
    expected = hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()

    class Budget:
        def reserve(self, quote, request_hash):
            assert request_hash == expected
            return 0

        def settle(self, index, evidence):
            return Decimal(0)

    def rpc(operation, **kwargs):
        if operation == "prepare":
            return {"request_sha256": expected}
        assert kwargs["request_sha256"] == expected
        return {"request_sha256": expected, "usage": [{"inputTokens": 1, "outputTokens": 1}],
                "session_id": "fixture", "content": "{}"}

    model.budget, model.rpc = Budget(), rpc
    assert model.complete(packet["system"], packet["user"]) == "{}"


def receipt(shutdown=True):
    return {
        "shutdown_verified": shutdown,
        "usage": [{"model": "fixture", "availableToolCount": 0, "numToolCalls": 0,
                   "inputTokens": 10, "outputTokens": 10}],
        "metrics": {
            "totalUserRequests": 1, "totalNanoAiu": 100000000,
            "sessionStartTime": "fixture-session", "aiCreditsStatus": "complete",
            "modelMetrics": {"fixture": {"totalNanoAiu": 100000000,
                                         "requests": {"count": 1}, "aiCreditsStatus": "complete"}},
        },
    }


def test_study_local_budget_settles_only_authoritative_fresh_context(tmp_path):
    budget = StudyBudget(tmp_path / "credits.json", 3, 24, {"scope": "new"})
    index = budget.reserve(QUOTE, "request")
    budget.settle(index, receipt())
    state = budget.snapshot()
    assert str(accounted_cost(state)) == "0.1"
    assert state["requests"][0]["reserved"] == "2.000000"
    assert state["requests"][0]["debited_upper"] == "2.000000"
    with pytest.raises(CreditLimit, match="NEW"):
        StudyBudget(budget.path, 3, 24, {"scope": "new"})


def test_missing_shutdown_keeps_full_hold_and_stops_next_dispatch(tmp_path):
    budget = StudyBudget(tmp_path / "credits.json", 3, 24, {"scope": "new"})
    budget.reserve(QUOTE, "request")
    with pytest.raises(CreditLimit, match="shutdown"):
        budget.settle(0, receipt(False))
    assert accounted_cost(budget.snapshot()) == 2
    with pytest.raises(CreditLimit):
        budget.reserve(QUOTE, "another")


@pytest.mark.parametrize("benchmark", ["newtonbench-discovery", "made"])
def test_discovery_profile_is_not_a_reference_and_requires_fresh_approval(tmp_path, benchmark):
    config = discovery_config(benchmark)
    assert config.value["model"]["provider"] == "copilot"
    assert config.arms[0].memory == "external"
    with pytest.raises(ValueError, match="NEW"):
        run_study(config, tmp_path / "unapproved")
    assert not (tmp_path / "unapproved").exists()
    value = json.loads(json.dumps(config.value))
    value["model"] = {"provider": "baseline", "id": "reference"}
    with pytest.raises(ValueError):
        StudyConfig.parse(value)


def test_made_submission_is_structured_not_executable():
    Made.validate_submission({"prototype": "fcc", "a": 1.55})
    for submission in ({"prototype": "fcc", "a": True},
                       {"prototype": "fcc", "a": 0},
                       {"prototype": "fcc", "a": 1.55, "code": "anything"}):
        with pytest.raises(ValueError):
            Made.validate_submission(submission)


def test_made_preserves_virtual_environment_interpreter_path(tmp_path, monkeypatch):
    import sys
    import science_harness.testbeds.made as module
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    seen = {}

    class Worker:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

        def request(self, value):
            import hashlib
            return {"ready": True, "worker_source_sha256": hashlib.sha256(
                Path(module.__file__).with_name("made_worker.py").read_bytes()).hexdigest()}

        def close(self):
            pass

    monkeypatch.setattr(module, "PinnedWorker", Worker)
    world = Made(tmp_path, executable)
    world.close()
    assert seen["python"] == executable.absolute()
    assert seen["python"] != executable.resolve()


def test_fixed_made_source_loader_skips_eager_optional_registries(tmp_path, monkeypatch):
    import importlib
    import sys
    from science_harness.testbeds.made_worker import load_upstream
    names = ("made", "made.data", "made.envs", "made.oracles",
             "made.oracles.classic", "made.utils")
    for name in names:
        directory = tmp_path / "src" / name.replace(".", "/")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("raise RuntimeError('unneeded registry')\n")
        monkeypatch.delitem(sys.modules, name, raising=False)
    oracle = tmp_path / "src/made/oracles/classic"
    (oracle / "probe.py").write_text("VALUE = 17\n")
    try:
        load_upstream(tmp_path)
        module = importlib.import_module("made.oracles.classic.probe")
        assert module.VALUE == 17
        assert module.__file__ == str(oracle / "probe.py")
    finally:
        for name in ("made.oracles.classic.probe", *reversed(names)):
            sys.modules.pop(name, None)


@pytest.mark.integration
def test_real_made_analytic_oracle_private_reference_and_drain():
    checkout, python = os.environ.get("MADE_PATH"), os.environ.get("MADE_PYTHON")
    if not checkout or not python:
        pytest.skip("Set MADE_PATH and MADE_PYTHON for the pinned analytic integration test")
    world = Made(Path(checkout).resolve(), Path(python).absolute(), budget=2)
    try:
        observed = world.measure("fcc", 1.55)
        assert observed.data["readings"][0] < 0
        missing = world.grade({"status": "concluded",
                               "submission": {"prototype": "fcc", "a": 1.57}})
        assert not missing["passed"] and "never measured" in missing["error"]
        score = world.grade({"status": "concluded",
                             "submission": {"prototype": "fcc", "a": 1.55}})
        assert score["passed"] and score["relative_regret"] < 0.01
        assert score["agent_oracle_calls"] == 1
        assert score["private_reference_oracle_calls"] > 300
        assert "reference" not in observed.data
        assert "relative_regret" not in world.context
    finally:
        world.close()
    assert world.worker.process.poll() is not None
