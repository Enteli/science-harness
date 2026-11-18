import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_history_survives_json_key_order_without_losing_role_or_content():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the optional Copilot SDK bridge")
    module = Path(__file__).resolve().parents[1] / "science_harness/physics_bridge_state.mjs"
    script = f"""
import assert from 'node:assert/strict';
import {{ hash, historyDelta, visibleResponse }} from {json.dumps(module.as_uri())};
const logical = [
  {{content:'original input',role:'user'}},
  {{role:'assistant',content:'<run_mse_fit>\\nexact source\\n</run_mse_fit>'}}
];
const replay = [
  {{content:'original input',role:'user'}},
  {{content:'<run_mse_fit>\\nexact source\\n</run_mse_fit>',role:'assistant'}},
  {{content:'observed fit only',role:'user'}}
];
assert.deepEqual(historyDelta(replay, logical), [replay[2]]);
assert.equal(hash({{model:'m',system:'s'}}), hash({{system:'s',model:'m'}}));
const changed = structuredClone(replay);
changed[1].content = 'changed source';
assert.throws(() => historyDelta(changed, logical), /prefix changed/);
changed[1] = {{role:'user',content:logical[1].content}};
assert.throws(() => historyDelta(changed, logical), /prefix changed/);
assert.throws(() => historyDelta([...replay, {{role:'assistant',content:'invented'}}], logical),
              /assistant-role/);
const visible = {{messageId:'science',content:'<run_mse_fit>exact code</run_mse_fit>',phase:'commentary'}};
const assembled = visibleResponse([
  {{messageId:'reasoning',content:'',reasoningText:'not a scientific response'}},
  visible, visible,
  {{messageId:'empty-final',content:'',phase:'final_answer'}}
]);
assert.equal(assembled.content, visible.content);
assert.deepEqual(assembled.visible_messages, [visible]);
assert.throws(() => visibleResponse([{{messageId:'empty',content:''}}]), /nonempty/);
assert.throws(() => visibleResponse([
  visible, {{...visible,content:'changed code'}}
]), /Conflicting/);
assert.throws(() => visibleResponse([
  {{messageId:'private',phase:'analysis',content:'not public'}}
]), /reasoning/);
"""
    subprocess.run([node, "--input-type=module", "-e", script], check=True,
                   capture_output=True, text=True, timeout=30)
