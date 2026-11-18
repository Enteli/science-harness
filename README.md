# science-harness

A reusable harness for scientific agents: choose experiments, collect evidence,
analyze results, and keep a durable **external lab notebook**.

Models use trusted tools, not arbitrary host code. Private evaluation stays
outside the scientist's context. A notebook records evidence and interpretation;
it is not another source of truth.

## Executive summary

| Integration | Supported scope | Recorded evidence |
|---|---|---|
| [DiscoverPhysics](docs/benchmark-studies.md) | Native trajectory-law controller; qualified macOS sandbox | Historical study: **47/50** reached the predictive target; **40/50** retained a passing final |
| [NewtonBench](docs/newtonbench-discovery.md) | Generic scientist; restricted energy-law task | One pilot: six acquisitions, **0.20% relative RMSE** on 500 clean held-out points |
| [MADE](docs/made.md) | Generic scientist; controlled CPU crystal optimization | One pilot: 12 queries, **0.0203% energy regret** |

These are separate protocols, not pooled or official leaderboard results.
**MADE is an analytic simulation, not full Materials Project coverage or a new
real material.** External notebooks are the auditable default, not a proven
performance winner over full history; the two pilots made no explicit notebook
fetches. See [historical results and censoring](docs/results.md) and
[pilot outcomes and limits](docs/crossbench-results.md).

## Start without an API

From a source checkout, using a maintained Python release (core supports 3.9+):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python examples/notebook.py --output artifacts/notebook-demo
science-harness bench list
```

No credentials, model calls or benchmark downloads are needed. The scripted
example demonstrates notebook use, not discovery, and ends inconclusively.

## First integration: DiscoverPhysics

[Install the native runtime](docs/benchmark-studies.md), then prepare a study
without running it:

```bash
science-harness bench init --output my-study.json
science-harness bench validate my-study.json
science-harness bench run my-study.json --dry-run
```

`my-study.json` is the editable run configuration:
[see a single-run example](configs/discoverphysics-external-notebook.json).

The default is external notebook with agent-selected experiments. DiscoverPhysics
also supports `--memory full-history`, `--review`, or `--random`. Random changes
experiment selection, not the scientist; review is separately charged.
Review + random is unsupported. NewtonBench/MADE currently qualify only their
declared external-notebook discovery profile.

Inspect the config before spending. Paid execution requires an exact new cap:

```bash
science-harness bench run my-study.json \
  --output artifacts/my-study --approve-credits 1000
```

Caps are maxima, **not expected prices**. Reads, reviews and repairs count;
unknown charges keep their reservations. No silent retries or unsafe host
fallback. Every run needs a new output directory.

## Reuse and extend

- [CLI and configuration](docs/cli.md), [architecture](docs/architecture.md),
  [adding a benchmark](docs/adding-a-benchmark.md),
  and [models/tools](docs/models-and-tools.md).
- [Zero-API NewtonBench reference](docs/second-integration.md), separate from the live pilot.
- [Contributing](CONTRIBUTING.md), [security](SECURITY.md), and [publishing](docs/publishing.md).

Python imports use `science_harness`; the package and CLI use `science-harness`:

```python
from science_harness import Scientist, BookStore, ToolRegistry
```

See [settings and migration](docs/migration.md) for environment variables and legacy aliases.

## License

First-party code is [MIT](LICENSE). Benchmarks, data, weights and providers
retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
