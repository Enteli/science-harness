# CLI and run configuration

`my-study.json` is an editable study config, not an output file or special
filename. [See the generated single-run example](../configs/discoverphysics-external-notebook.json).

## Commands

Prefix each command below with `science-harness`:

| Command | Purpose |
|---|---|
| `bench list` | List adapters, profiles and runtime requirements |
| `bench init --output my-study.json` | Create a NEW config; default is DiscoverPhysics/external notebook |
| `bench validate my-study.json` | Validate and enumerate planned investigations |
| `bench run my-study.json --dry-run` | Preview without instruments or model calls |
| `bench run my-study.json --output artifacts/run-1 --approve-credits 1000` | Execute with the exact NEW whole-study credit cap |
| `bench report artifacts/run-1` | Summarize saved results without model calls |

For flags, use e.g. `science-harness bench run --help`.
Only the real `run` starts an experiment; every run needs a new output directory.
Model-led profiles require model access and exact credit approval; zero-API
references do not. The 1,000-credit example matches the default DiscoverPhysics
config, not every profile. Caps are maxima, not expected spending.

## Select a profile

Use `bench init --benchmark NAME --output my-study.json`:

| Name | Scope |
|---|---|
| `discoverphysics` | Native trajectory-law discovery, the default |
| `newtonbench-discovery` | Model-led energy-law discovery |
| `made` | Model-led controlled analytic optimization |
| `newtonbench` | Scripted, zero-API reference, not LLM discovery |

DiscoverPhysics also accepts `--memory full-history`, `--review` or `--random`.
Review + random is unsupported; these options are not qualified for every adapter.

## Key config fields

| Field | Controls |
|---|---|
| `benchmark`, `protocol` | Registered environment and versioned scientific interface |
| `worlds`, `seeds`, `arms` | Explicit task/start/policy combinations; each combination is an investigation |
| `model` | Supported provider and model ID |
| `limits.requests`, `limits.observations` | Per-investigation model and acquisition caps |
| `limits.credits` | Whole-study ceiling shared by every investigation |
| `parameters` | Adapter-specific paths, runtime pins and calibration |

Keep credentials outside JSON. Do not silently change scoring calibration or
edit completed registrations; create a new config and output for a new study.
See the [integration guides](../README.md#executive-summary) and
[native protocol reference](native-protocol-reference.md) for detailed contracts.
