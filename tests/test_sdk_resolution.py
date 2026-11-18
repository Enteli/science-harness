import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "science_harness/providers/sdk.mjs"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is required for SDK module-resolution tests")


def sdk(project, version="1.0.28", tag="project", broken=False):
    project.mkdir(parents=True, exist_ok=True)
    (project / "package.json").write_text('{"private":true}')
    package = project / "node_modules/@github/copilot-sdk"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": "@github/copilot-sdk", "version": version, "main": "./index.cjs"}))
    code = (
        "class CopilotClient { constructor() { throw new Error('SDK_RESOLVED_NO_DISPATCH'); } }\n"
        "const RuntimeConnection = {forStdio() { return {}; }};\n"
        "module.exports = {CopilotClient, RuntimeConnection, fixtureTag:" + json.dumps(tag) + "};\n")
    if broken:
        code = "require('missing-review-transitive-dependency');\n" + code
    (package / "index.cjs").write_text(code)


def loader(tmp_path):
    path = tmp_path / "installed/site-packages/science_harness/providers/sdk.mjs"
    path.parent.mkdir(parents=True)
    shutil.copyfile(LOADER, path)
    return path


def run_loader(path, cwd, explicit=None):
    env = {"PATH": os.environ["PATH"]}
    if explicit is not None:
        env["SCIENCE_HARNESS_SDK_ROOT"] = str(explicit)
    code = (
        f"import {{loadCopilotSdk}} from {json.dumps(path.as_uri())};"
        "const value=loadCopilotSdk();"
        "console.log(JSON.stringify({tag:value.fixtureTag,version:value.sdkResolution.version}));")
    return subprocess.run([NODE, "--input-type=module", "-e", code],
                          cwd=cwd, env=env, text=True, capture_output=True, timeout=20)


def test_external_install_resolves_sdk_in_operator_project(tmp_path):
    path = loader(tmp_path)
    project = tmp_path / "working project"
    sdk(project)
    result = run_loader(path, project)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"tag": "project", "version": "1.0.28"}


def test_explicit_sdk_project_is_used_without_cwd_or_source_fallback(tmp_path):
    path = loader(tmp_path)
    project, chosen = tmp_path / "project", tmp_path / "chosen"
    sdk(project, version="9.9.9")
    sdk(chosen, tag="chosen")
    result = run_loader(path, project, chosen)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["tag"] == "chosen"
    result = run_loader(path, project, tmp_path / "missing")
    assert result.returncode != 0


def test_selected_wrong_version_is_not_silently_replaced(tmp_path):
    path = loader(tmp_path)
    project = tmp_path / "project"
    sdk(project, version="9.9.9")
    sdk(tmp_path / "installed", tag="source")
    result = run_loader(path, project)
    assert result.returncode != 0
    assert "Expected @github/copilot-sdk@1.0.28" in result.stderr


def test_transitive_import_error_does_not_fall_back_to_another_sdk(tmp_path):
    path = loader(tmp_path)
    project = tmp_path / "project"
    sdk(project, broken=True)
    sdk(tmp_path / "installed", tag="source")
    result = run_loader(path, project)
    assert result.returncode != 0
    assert "missing-review-transitive-dependency" in result.stderr


def test_source_installation_remains_a_supported_resolution_root(tmp_path):
    path = loader(tmp_path)
    sdk(tmp_path / "installed", tag="source")
    working = tmp_path / "empty-cwd"
    working.mkdir()
    result = run_loader(path, working)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["tag"] == "source"


@pytest.mark.parametrize("bridge", ["physics_bridge.mjs", "providers/guarded_bridge.mjs"])
def test_actual_installed_bridges_resolve_sdk_before_any_dispatch(tmp_path, bridge):
    path = loader(tmp_path)
    installed = path.parent.parent
    for relative in ("physics_bridge.mjs", "physics_bridge_state.mjs", "providers/guarded_bridge.mjs"):
        destination = installed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "science_harness" / relative, destination)
    project = tmp_path / "project"
    sdk(project)
    result = subprocess.run(
        [NODE, str(installed / bridge), str(tmp_path / "runtime")], cwd=project,
        input="", text=True, capture_output=True, timeout=20,
        env={"PATH": os.environ["PATH"], "COPILOT_CLI_PATH": "/test/no-model-executable",
             "COPILOT_CLI_VERSION": "1.0.103-0"})
    assert result.returncode != 0
    assert "SDK_RESOLVED_NO_DISPATCH" in result.stderr
    assert "ERR_MODULE_NOT_FOUND" not in result.stderr
