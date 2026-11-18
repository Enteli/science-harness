import importlib.util
import json
import os
from pathlib import Path

import pytest

from science_harness.harness.cli import discoverphysics_config, main, newtonbench_config
from science_harness.harness.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_init_defaults_to_one_external_notebook_investigation_without_dispatch(tmp_path, capsys):
    path = tmp_path / "study.json"
    main(["bench", "init", "--output", str(path)])
    result = json.loads(capsys.readouterr().out)
    config = load_config(path)
    assert result["dispatches"] == 0 and result["episodes"] == 1
    assert config.arms[0].memory == "external"
    assert config.arms[0].name == "external-notebook"
    assert config.arms[0].advisor == "off" and config.arms[0].selection == "active"
    assert config.value["limits"] == {"requests": 100, "observations": 50, "credits": 1000}
    original = path.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["bench", "init", "--output", str(path), "--memory", "B"])
    assert exc.value.code == 2 and path.read_bytes() == original


@pytest.mark.parametrize("memory,review,random", [
    ("full-history", False, False), ("external-notebook", True, False),
    ("full-history", True, False), ("external-notebook", False, True),
    ("B", False, False), ("C", True, False), ("B", True, False),
    ("B", False, True), ("C", False, True),
])
def test_supported_options_preserve_same_scientific_limits(memory, review, random):
    config = discoverphysics_config(memory, review, random)
    arm = config.arms[0]
    assert arm.memory == ("external" if memory in ("C", "external-notebook") else "inline")
    assert arm.advisor == ("design" if review else "off")
    assert arm.selection == ("random" if random else "active")
    assert config.value["limits"]["observations"] == 50


def test_legacy_memory_aliases_generate_the_same_descriptively_named_config():
    assert discoverphysics_config("C").value == discoverphysics_config("external-notebook").value
    assert discoverphysics_config("B").value == discoverphysics_config("full-history").value


def test_model_settings_use_the_canonical_environment_prefix(monkeypatch):
    from science_harness import OpenAICompatLLM
    monkeypatch.setenv("SCIENCE_HARNESS_MODEL", "example-model")
    monkeypatch.setenv("SCIENCE_HARNESS_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("SCIENCE_HARNESS_API_KEY", "fixture-not-a-real-credential")
    model = OpenAICompatLLM()
    assert model.model == "example-model"
    assert model.base_url == "https://example.test/v1"
    assert model.api_key == "fixture-not-a-real-credential"


def test_newtonbench_init_is_notebook_reference_without_runtime_or_spending(tmp_path, capsys):
    path = tmp_path / "newton.json"
    main(["bench", "init", "--benchmark", "newtonbench", "--output", str(path)])
    assert json.loads(capsys.readouterr().out)["dispatches"] == 0
    config = load_config(path)
    assert config.value == newtonbench_config().value
    assert config.value["limits"]["credits"] == 0
    assert config.value["model"] == {"provider": "baseline", "id": "reference"}
    assert config.arms[0].profile == "notebook-reference-v1"
    with pytest.raises(ValueError, match="only|supports"):
        newtonbench_config(memory="full-history")


def test_notebook_reference_prefix_uses_actual_kernel_storage_and_fetch(tmp_path):
    from science_harness import Scientist, ScriptedLLM
    from science_harness.benchmarks.reference import NotebookReferencePolicy
    model = NotebookReferencePolicy(ScriptedLLM([
        json.dumps({"action": "finish", "status": "inconclusive",
                    "summary": "Only exercise the notebook contract.", "updates": []})]))
    scientist = Scientist(model, tmp_path / "store", max_steps=3, memory_policy="notebook")
    try:
        result = scientist.investigate("Offline notebook prefix.")
        assert result["status"] == "inconclusive"
        assert [e["action"]["action"] for e in result["events"]] == ["note", "fetch", "finish"]
        assert result["events"][1]["lookup"]["value"][0]["title"] == "Integration procedure"
        assert not result["evidence"]
    finally:
        scientist.close()

def test_unsupported_policy_combination_fails_before_creating_config(tmp_path):
    path = tmp_path / "must-not-exist.json"
    with pytest.raises(SystemExit) as exc:
        main(["bench", "init", "--output", str(path), "--review", "--random"])
    assert exc.value.code == 2 and not path.exists()


def test_scripted_notebook_example_uses_real_store_retrieval_and_resource_limits(tmp_path):
    spec = importlib.util.spec_from_file_location("public_notebook_demo", ROOT / "examples/notebook.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    result = demo.run(tmp_path / "demo")
    assert result["api_calls"] == 0 and result["observations"] == 1
    assert result["status"] == "inconclusive"
    notebook = Path(result["notebook"]).read_text()
    assert "One point does not identify a global law." in notebook
    assert '"action": "fetch"' in notebook
    with pytest.raises(FileExistsError):
        demo.run(tmp_path / "demo")


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory permissions")
def test_new_evidence_store_is_private_without_chmod_of_existing_user_directory(tmp_path):
    from science_harness import BookStore
    fresh = tmp_path / "private"
    store = BookStore(fresh)
    store.close()
    assert fresh.stat().st_mode & 0o077 == 0
    existing = tmp_path / "user-owned"
    existing.mkdir(mode=0o755)
    original = existing.stat().st_mode
    store = BookStore(existing)
    store.close()
    assert existing.stat().st_mode == original
