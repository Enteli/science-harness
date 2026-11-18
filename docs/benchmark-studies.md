# DiscoverPhysics: setup and studies

The native adapter preserves DiscoverPhysics's trajectory controller,
Python/SciPy law interface and sealed evaluation. It requires **macOS's qualified
sandbox**, Node, cached Copilot CLI `1.0.103-0` and supported model access.
There is no Linux numerical backend or unsafe host fallback.

## Setup

From the repository root:

```bash
python -m pip install -e '.[dev,physics-comparison]'
npm ci --ignore-scripts
git clone https://github.com/SampsonML/DiscoverPhysics .cache/DiscoverPhysics
git -C .cache/DiscoverPhysics checkout 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5
science-harness bench init --output my-study.json
science-harness bench validate my-study.json
science-harness bench run my-study.json --dry-run
```

Validation/dry-run make no acquisitions, panels or model calls. For SDK location
and credentials, see [models and tools](models-and-tools.md#copilot-sdk-location).

## Run and inspect

The generated config declares one external-notebook investigation, up to 100
all-role model requests, 50 observations and a **new 1,000-credit maximum**:

```bash
science-harness bench run my-study.json \
  --output artifacts/my-study --approve-credits 1000
science-harness bench report artifacts/my-study
```

The approval must match the config. A cap is not an expected charge:
preflight must reserve one full provider envelope before dispatch.
Unknown charges retain their holds; infrastructure failures stop the study
without silent retries. Every run needs a new output directory.

For comparisons, [`discoverphysics-memory-advisor.json`](../configs/discoverphysics-memory-advisor.json)
declares ten investigations and a 1,500-credit **whole-study** ceiling; it does
not promise all ten will fit. Seeds are examples, not reserved assignments.
Scientist and advisor share the ledger. Optional `parameters.trace_backup`
names a new, non-nested backup; keep long-lived records outside disposable worktrees.

## Native profiles

External notebook is the default. Choose `--memory full-history`, `--review`,
or `--random` explicitly. Review is advisory and separately counted; random
changes only the executed experiment, not the scientist's analysis. Review +
random is unsupported.

`legacy-v1` preserves the old controller path. `memory-advisor-v1` adds the
declared memory/intent/review policies. Read, repair and fit-only turns count
against the same request cap; an explicit modern cap may reach 500 without
changing other limits.

See the [protocol reference](native-protocol-reference.md) for configuration
fields, notebook versions, action recovery, exact review behavior and historical
comparability. Different protocol epochs must not be pooled.

## Evaluation and outputs

The primary target is position MSE divided by the declared scale squared,
at most 0.01 across 32 clean launches and five times. Velocity error is secondary.
Private panels are registered and reference-checked before paid dispatch;
they never guide prompts, retrieval, stopping or best-law restoration.

Outputs include `study.json`, `report.json`, `summary.csv`, `pairs.csv` and
per-episode evidence, failures, sources and receipts. Assigned, censored,
first-hit and accepted-final outcomes remain distinct. Completion is not
scientific success.

`science-harness bench list` lists all bindings; [NewtonBench](newtonbench-discovery.md)
and [MADE](made.md) use different instruments and custom metrics.
