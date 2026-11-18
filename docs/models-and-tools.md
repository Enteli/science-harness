# Models and tools

## Models

The portable kernel accepts the existing `LLM` protocol:

```python
from science_harness import Scientist

class MyModel:
    def complete(self, system: str, user: str) -> str:
        # Return the exact public response from a trusted provider.
        # Configure credentials outside the scientific request.
        raise NotImplementedError

scientist = Scientist(MyModel(), home="artifacts/my-agent", memory_policy="notebook")
```

Use `ScriptedLLM` for deterministic, credential-free tests. `OpenAICompatLLM`
supports compatible endpoints, and the optional Copilot transport is separate.
Errors and empty responses must surface; do not silently substitute another model.

**A generic provider integration is not a qualified native benchmark profile.**
The first native DiscoverPhysics binding deliberately accepts only its declared
provider/model/runtime combinations. Qualifying another requires source-frozen
tests of actual request formatting, context/token limits, public-output assembly,
zero-tool behavior, pre-dispatch reservation and complete usage accounting.
Do not remove the identity checks just because a replacement implements `complete`.

Record model and reasoning identity, input/output/context limits and per-role
usage. Reads, corrections and reviews are real requests. Distinguish all-model
request counters from user-initiated counters, and cumulative totals from deltas.
Unknown charges retain their full holds. Tests must cover incomplete, duplicate,
noncontiguous and contradictory receipts, not only an ideal successful response.

### Copilot SDK location

The Node-backed adapters require the pinned `@github/copilot-sdk@1.0.28`
dependency from `package-lock.json`. Run `npm ci` in the operator's project.
The loader checks that project first, then the harness's own installation
ancestors. Resolution does not depend on the Python wheel being installed
inside the npm checkout.

For another trusted npm project, set:

```bash
export SCIENCE_HARNESS_SDK_ROOT=/absolute/path/to/project-containing-package-json
```

An explicit root is exclusive: an absent, incompatible or broken SDK there
fails before runtime startup, rather than silently selecting another install.
The selected SDK version/location is returned in the runtime catalog; the
model-led profiles also retain it with their provider quote.
This controls software dependency resolution, not authentication or permission
to spend credits. Credentials stay out of scientific prompts.

## Trusted tools

```python
from science_harness import AcquisitionContract, Observation, Tool, ToolRegistry

def measure(x):
    if type(x) not in (int, float) or not 0 <= x <= 2:
        raise ValueError("x outside the approved range")
    return Observation({"reading": 2 * x}, source="my.sensor", conditions={"x": x})

tools = ToolRegistry([Tool(
    "measure", "Read the approved sensor", measure,
    acquisition=AcquisitionContract(
        "instrument_call", "One started invocation",
        ("observations",), lambda **_: {"observations": 1}),
)])
```

Tool implementations are trusted application code. The scientist chooses only
registered names and validated arguments. Cost estimation is side-effect-free;
every started acquisition, including a failure, consumes its declared resource.
Sensor repetitions, new preparations and independent specimens are different
units, not interchangeable row counts.

Adding a tool must not grant an arbitrary shell, filesystem or evaluator channel.
For real instruments, add hardware-specific authorization and safety interlocks;
a model prompt is not a laboratory safety system. For generated code, use a
separately qualified execution backend; see `SECURITY.md`.
