import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from science_harness import benchmarks
from science_harness.benchmarks import BenchmarkSpec, register
from science_harness.cli import main
from science_harness.harness import StudyConfig, load_config
from science_harness.harness.runner import run_study, source_manifest
from science_harness.policies import Arm
from science_harness.providers import usage_by_role
from science_harness.reporting import report_study

ROOT = Path(__file__).resolve().parents[1]


def example(name="spring-offline"):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["spring-offline", "discoverphysics-memory-advisor",
                                  "discoverphysics-external-notebook", "newtonbench-reference"])
def test_examples_validate_without_initializing_resources(name, monkeypatch):
    from science_harness.benchmarks.discoverphysics import DiscoverPhysicsAdapter

    def forbidden(*args, **kwargs):
        pytest.fail("Config validation must never initialize an instrument or provider")
    monkeypatch.setattr(DiscoverPhysicsAdapter, "prepare", forbidden)
    config = StudyConfig.parse(example(name))
    assert config.describe()["dispatches"] == 0
    assert len(config.cells()) == len(config.value["worlds"]) * len(config.value["seeds"]) * len(config.arms)


def test_native_profiles_preserve_legacy_bridge_and_common_budget():
    from science_harness.benchmarks.discoverphysics import policy_arguments
    config = StudyConfig.parse(example("discoverphysics-memory-advisor"))
    legacy, b, _, c, _ = config.arms
    assert policy_arguments(legacy, 100) == {
        "controller": "science", "selector": "active", "notebook_memory": "conversation",
        "require_experiment_intent": False, "total_request_limit": None}
    assert policy_arguments(b, 100)["total_request_limit"] == 100
    assert policy_arguments(c, 100)["notebook_memory"] == "external"
    assert all(policy_arguments(a, 100)["require_experiment_intent"] for a in config.arms[1:])


def test_native_action_recovery_is_explicit_and_boolean():
    value = example("discoverphysics-memory-advisor")
    assert "action_recovery" not in value["parameters"]
    value["parameters"]["action_recovery"] = True
    assert StudyConfig.parse(value).value["parameters"]["action_recovery"] is True
    value["parameters"]["action_recovery"] = "true"
    with pytest.raises(ValueError, match="action_recovery must be boolean"):
        StudyConfig.parse(value)


def test_native_inference_call_cap_can_be_explicitly_extended_without_raising_other_limits():
    value = example("discoverphysics-memory-advisor")
    assert value["limits"]["requests"] == 100
    value["limits"]["requests"] = 500
    config = StudyConfig.parse(value)
    assert config.value["limits"]["requests"] == 500
    assert config.value["limits"]["observations"] == value["limits"]["observations"]
    assert config.value["limits"]["credits"] == value["limits"]["credits"]
    for invalid in (501, True, 500.0):
        value["limits"]["requests"] = invalid
        with pytest.raises(ValueError, match="limits.requests"):
            StudyConfig.parse(value)
    portable = example()
    portable["limits"]["requests"] = 101
    with pytest.raises(ValueError, match="1..100"):
        StudyConfig.parse(portable)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(unknown="not a setting"),
    lambda c: c.update(protocol="the-old-expression-interface"),
    lambda c: c.update(benchmark="../outside"),
    lambda c: c.update(seeds=[True]),
    lambda c: c["limits"].update(requests=True),
    lambda c: c["limits"].update(credits=float("nan")),
    lambda c: c["model"].update(api_key="DO_NOT_PERSIST"),
    lambda c: c["arms"][0].update(name="../outside"),
    lambda c: c["arms"][0].update(advisor="design"),
])
def test_invalid_or_misleading_studies_fail_before_execution(mutation):
    value = example()
    mutation(value)
    with pytest.raises(ValueError):
        StudyConfig.parse(value)


def test_duplicate_budget_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"limits":{"credits":1,"credits":1000}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_config(path)


def test_no_paid_approval_means_no_output_or_model_setup(tmp_path):
    config = StudyConfig.parse(example("discoverphysics-memory-advisor"))
    for approval in (None, "499", "NaN", "not-a-number"):
        with pytest.raises(ValueError):
            run_study(config, tmp_path / "must-not-exist", approval)
        assert not (tmp_path / "must-not-exist").exists()


def test_offline_example_runs_existing_scientist_and_persists_results(tmp_path):
    config = StudyConfig.parse(example())
    output = tmp_path / "study"
    result = run_study(config, output)
    assert result["status"] == "completed" and result["episodes"] == 2
    study = json.loads((output / "study.json").read_text())
    assert all(r["final_pass"] and r["billable_requests"] == 0 for r in study["rows"])
    assert all(r["resource_budget"]["charged"]["observations"] <= 8 for r in study["rows"])
    assert len(list((output / "episodes").glob("*/result.json"))) == 2
    assert (output / "summary.csv").exists() and (output / "pairs.csv").exists()
    assert report_study(output)["summary"][0]["final_passes"] == 2
    with pytest.raises(FileExistsError):
        run_study(config, output)


def test_reference_acquisition_quota_is_enforced_by_existing_core(tmp_path):
    value = example()
    value["seeds"] = [0]
    value["limits"]["observations"] = 1
    run_study(StudyConfig.parse(value), tmp_path / "quota")
    study = json.loads((tmp_path / "quota/study.json").read_text())
    row = study["rows"][0]
    assert row["observations"] == 1 and not row["final_pass"]
    assert row["resource_budget"]["charged"]["observations"] == 1


def test_new_benchmark_registration_needs_no_core_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "_REGISTRY", dict(benchmarks._REGISTRY))
    closed = []

    class Tiny:
        @staticmethod
        def validate(config):
            assert config.value["parameters"] == {"gain": 2}

        def prepare(self, config, output, approved_credits):
            self.config = config

        def run_episode(self, cell, directory):
            return {"status": "completed", "requests": 1, "observations": 1,
                    "final_pass": True, "metrics": {"tiny_error": 0.}}

        def close(self):
            closed.append(True)

    register(BenchmarkSpec("tiny", "tiny-protocol-v1", ("apparatus",), ("baseline",),
                           ("tiny-profile-v1",), "Python", "A test extension"), lambda: Tiny)
    value = example()
    value.update(benchmark="tiny", protocol="tiny-protocol-v1", worlds=["apparatus"],
                 parameters={"gain": 2})
    value["arms"][0]["profile"] = "tiny-profile-v1"
    result = run_study(StudyConfig.parse(value), tmp_path / "extension")
    assert result["episodes"] == 2 and closed == [True]
    with pytest.raises(ValueError, match="already registered"):
        register(benchmarks.get("tiny")[0], lambda: Tiny)


def test_public_cli_lists_and_dry_runs_without_writing(tmp_path, capsys):
    assert main(["bench", "list"]) is None
    assert json.loads(capsys.readouterr().out)["benchmarks"]
    output = tmp_path / "dry"
    main(["bench", "run", str(ROOT / "configs/discoverphysics-memory-advisor.json"),
          "--dry-run", "--output", str(output)])
    printed = capsys.readouterr().out
    result = json.loads(printed)
    assert result["dispatches"] == 0 and not output.exists()
    assert "native-trajectory-v1" in printed


@pytest.mark.parametrize("arguments", [
    ["bench", "list"],
    ["bench", "validate", str(ROOT / "configs/spring-offline.json")],
])
def test_console_entrypoint_exits_zero_for_successful_commands(arguments):
    executed = subprocess.run(
        [sys.executable, "-c", "from science_harness.cli import main; import sys; sys.exit(main(sys.argv[1:]))",
         *arguments], cwd=ROOT, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert isinstance(json.loads(executed.stdout), dict)
    assert executed.stderr == ""


def test_source_manifest_covers_new_modules_not_only_legacy_shims(monkeypatch):
    source = source_manifest()
    assert "science_harness/harness/config.py" in source["sha256"]
    assert "science_harness/benchmarks/discoverphysics.py" in source["sha256"]
    assert "science_harness/physics_comparison.py" in source["sha256"]
    assert all(not name.startswith((".venv", ".cache")) for name in source["sha256"])
    from science_harness.harness import runner
    monkeypatch.setattr(runner.shutil, "which", lambda command: None)
    without_git = runner.source_manifest()
    assert without_git["revision"] is None
    assert without_git["revision_unavailable_reason"] == "Git unavailable"
    assert without_git["sha256"] == source["sha256"]


def test_unreported_or_failed_usage_is_not_silently_zero():
    actual = {"usage": [{"inputTokens": 10, "outputTokens": 3}]}
    result = usage_by_role([actual], ["scientist", "review"])
    assert result["scientist"]["input_tokens"] == 10
    assert result["review"]["input_tokens"] is None
    assert usage_by_role([{"usage": {}}], ["scientist"])["scientist"]["input_tokens"] is None
    generic = usage_by_role([{"usage": {"input_tokens": 7, "output_tokens": None}}], ["scientist"])
    assert generic["scientist"]["input_tokens"] == 7
    assert generic["scientist"]["output_tokens"] is None


@pytest.mark.parametrize("unknown_charge", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("backup", [False, True])
def test_native_adapter_uses_one_fresh_approved_ledger_and_retains_unknown_holds(
        tmp_path, monkeypatch, unknown_charge, legacy, backup):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("jax")
    pytest.importorskip("tiktoken")
    from science_harness import physics_comparison, physics_forecast_study, physics_reference
    from science_harness import physics_runtime, physics_trace, physics_transport
    from science_harness.benchmarks import discoverphysics

    calls = []

    class Numerics:
        def __init__(self, checkout):
            pass

        def verify_boundary(self, path):
            return {"filesystem_denied": True, "network_denied": True}

    class World:
        def __init__(self, checkout, world):
            pass

        def measure(self, design, seed, clean=False):
            assert clean
            assert calls and calls[0].catalog_checked
            return {"pos2": [[0., 0.]] * 5, "velocity2": [[0., 0.]] * 5}

    class Provider:
        closed = False

        def __init__(self, directory, *, budget, pinned_version, receipt_directory=None):
            self.budget = budget
            self.receipt_directory = receipt_directory
            self.catalog_checked = False
            calls.append(self)

        def catalog(self):
            self.catalog_checked = True
            return {"models": [{
                "id": "gpt-6.1-sol", "policy": {"state": "enabled"},
                "billing": {"tokenPrices": {
                    "inputPrice": "1", "outputPrice": "1", "cacheReadPrice": "1",
                    "cacheWritePrice": "1", "batchSize": "1000"}},
                "capabilities": {"limits": {"max_prompt_tokens": 10, "max_output_tokens": 10}},
            }], "units": [{"modelId": "gpt-6.1-sol", "unit": "ai_credits"}]}

        def __call__(self, **packet):
            quote = {"model": packet["model"], "input_price": "1", "output_price": "1",
                     "input_limit": "10", "output_limit": "10", "batch": "1000"}
            reservation = self.budget.reserve(quote, "fixture-only")
            if unknown_charge:
                raise RuntimeError("Unknown charge; keep full hold")
            result = {"usage": [{"model": packet["model"], "inputTokens": 1, "outputTokens": 1,
                                 "availableToolCount": 0, "numToolCalls": 0}], "content": "fixture"}
            self.budget.settle(reservation, result)
            return result

        def close(self):
            self.closed = True

    def fake_case(checkout, world, numerics, provider, directory, **kwargs):
        expected = tmp_path / "backup/episodes" / directory.name if backup else None
        assert kwargs["trace_directory"] == expected
        if legacy:
            assert kwargs["design_advisor"] is None
            assert kwargs["notebook_memory"] == "conversation"
            assert not kwargs["require_experiment_intent"] and kwargs["total_request_limit"] is None
        else:
            assert kwargs["design_advisor"].provider is provider
            assert kwargs["notebook_memory"] == "external"
            assert kwargs["require_experiment_intent"] and kwargs["total_request_limit"] == 100
        result = {"status": "completed", "error_kind": None, "requests": 1,
                  "observations": [{}, {}], "responses": [], "request_roles": ["scientist"]}
        try:
            result["responses"].append(provider(model=kwargs["model"]))
        except RuntimeError:
            result.update(status="failed", error_kind="RuntimeError")
        if legacy:
            result.pop("request_roles")
            (directory / "operations").mkdir()
            (directory / "operations/0000-model.json").write_text(json.dumps({
                "input": {"session_key": str(directory.resolve()) + ":aux:1"}}))
        return result

    def fake_grade(result, *args):
        return {"first_hit": None, "capped_observations": 51, "final": None}

    monkeypatch.setattr(discoverphysics.sys, "platform", "darwin")
    original_which = discoverphysics.shutil.which
    monkeypatch.setattr(discoverphysics.shutil, "which", lambda name:
                        "/fixture/" + name if name in ("node", "copilot") else original_which(name))
    monkeypatch.setattr(physics_reference, "provenance", lambda path: {"test_fixture": True})
    monkeypatch.setattr(physics_runtime, "IsolatedNumerics", Numerics)
    monkeypatch.setattr(physics_runtime, "NativeWorld", World)
    monkeypatch.setattr(physics_transport, "CopilotProvider", Provider)
    monkeypatch.setattr(physics_forecast_study, "check_reference", lambda *args: {"fixture": True})
    monkeypatch.setattr(physics_comparison, "run_case", fake_case)
    monkeypatch.setattr(physics_comparison, "grade", fake_grade)
    monkeypatch.setattr(physics_trace, "export_episode", lambda *args: None)
    value = example("discoverphysics-memory-advisor")
    value["limits"]["credits"] = 500
    (tmp_path / "checkout/ScienceAgent").mkdir(parents=True)
    value["parameters"]["checkout"] = str(tmp_path / "checkout")
    value["worlds"] = ["yukawa"]
    value["parameters"]["position_scale"] = {"yukawa": 1.}
    value["arms"] = [value["arms"][0] if legacy else value["arms"][-1]]
    if backup:
        value["parameters"]["trace_backup"] = str(tmp_path / "backup")
    config = StudyConfig.parse(value)
    output = tmp_path / "native"
    result = run_study(config, output, "500")
    assert calls[0].closed and calls[0].budget.path == output / "credits.json"
    ledger = json.loads((output / "credits.json").read_text())
    assert len(ledger["requests"]) == 1
    assert ledger["requests"][0]["status"] == ("reserved" if unknown_charge else "settled")
    assert result["status"] == ("stopped" if unknown_charge else "completed")
    assert (output / "private/yukawa-12345.json").exists()
    usage = report_study(output)["usage_by_role"][0]
    assert usage["input_tokens"] == (None if unknown_charge else 1)
    assert usage["role"] == ("auxiliary" if legacy else "scientist")
    if backup:
        assert calls[0].receipt_directory == tmp_path / "backup/transport"
        for path in (output / "private").glob("*.json"):
            assert path.read_bytes() == (tmp_path / "backup/private" / path.name).read_bytes()
        episode = next((output / "episodes").iterdir())
        assert (episode / "result.json").read_bytes() == (
            tmp_path / "backup/episodes" / episode.name / "result.json").read_bytes()
        assert (episode / "scores.json").read_bytes() == (
            tmp_path / "backup/episodes" / episode.name / "scores.json").read_bytes()
    if not legacy and not unknown_charge and not backup:
        from science_harness.physics_budget import CreditLimit
        value["limits"]["credits"] = .001
        with pytest.raises(CreditLimit, match="one full provider reservation"):
            run_study(StudyConfig.parse(value), tmp_path / "underfunded", ".001")
        assert calls[-1].closed
        assert not (tmp_path / "underfunded/private").exists()
        assert json.loads((tmp_path / "underfunded/credits.json").read_text())["requests"] == []


def test_reporting_preserves_censors_pairing_and_unknown_role_usage(tmp_path):
    value = example()
    value.update(seeds=[1, 2], arms=[
        {"name": "baseline", "memory": "none", "profile": "reference-v1"},
        {"name": "variant", "memory": "none", "profile": "reference-v1"}])
    rows = []
    for arm, seed, cost, censored in [
            ("baseline", 1, 10, False), ("baseline", 2, 10, False),
            ("variant", 1, 5, False), ("variant", 2, 1, True)]:
        rows.append({"world": "noise", "seed": seed, "arm": arm, "requests": 1,
                     "final_pass": not censored, "capped_observations": cost, "censored": censored,
                     "usage_by_role": {"scientist": {"attempts": 1,
                                                    "input_tokens": None if censored else 10,
                                                    "output_tokens": None if censored else 2}}})
    (tmp_path / "study.json").write_text(json.dumps({
        "schema_version": 1, "status": "stopped", "config": value, "rows": rows}))
    report = report_study(tmp_path)
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["capped_observation_difference"] == -5
    assert report["summary"][1]["censored"] == 1
    assert report["summary"][1]["mean_capped_observations"] == 5
    assert report["usage_by_role"][1]["input_tokens"] is None


def test_new_reporter_refuses_to_reinterpret_a_historical_manifest(tmp_path):
    original = '{"scope":"historical-study","rows":[]}'
    (tmp_path / "study.json").write_text(original)
    with pytest.raises(ValueError, match="original protocol"):
        report_study(tmp_path)
    assert (tmp_path / "study.json").read_text() == original
    assert not (tmp_path / "report.json").exists()
