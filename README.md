# science-harness

A small public interface for **science agents with an external lab notebook**:
propose an experiment, acquire evidence, analyze it, record an interpretation,
retrieve earlier work, and decide what to test next.

The default setup is **external notebook: fresh model context plus a durable,
explicitly retrieved notebook**. Full history, advisory review and random
experiment selection are explicit alternatives, not an automatic factorial sweep.
The notebook is an auditable scientific record, not another source of truth.

## Executive summary

science-harness provides auditable scientific-agent infrastructure with three
scoped integrations to public benchmark projects:

| Integration | Implemented interface | Recorded evidence |
|---|---|---|
| [DiscoverPhysics](docs/benchmark-studies.md) | Pinned native-controller bridge for Python trajectory-law discovery | Historical registered study: **47/50** investigations reached the held-out target; **40/50** retained a passing final |
| [NewtonBench](docs/newtonbench-discovery.md) | Generic `Scientist` recovering a restricted stored-energy law | One model-led pilot: six acquisitions, **0.20% relative RMSE** on 500 clean held-out points |
| [MADE](docs/made.md) | Generic `Scientist` optimizing a controlled CPU analytic crystal task | One model-led pilot: 12 queries, **0.0203% energy regret** against a numerical reference |

These are distinct protocols, not a pooled benchmark or official leaderboard
reproduction. DiscoverPhysics retains its qualified native controller;
NewtonBench and MADE use the shared generic scientist loop. MADE's analytic
test is **not the full Materials Project/MADE-paper benchmark or discovery of a
new real material**.

External notebooks make scientific memory explicit and inspectable; the
experiments do **not** establish a general performance advantage over full
history. The small integration pilots made no explicit notebook fetches and
cannot establish a memory-policy effect. Historical studies and their
failure/censoring counts remain separate from package-release checks.
See [detailed historical results](docs/results.md) and
[live integration outcomes and limits](docs/crossbench-results.md).

## Start without an API

Python 3.9+ compatibility; use a maintained Python release for new deployments.
From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python examples/notebook.py --output artifacts/notebook-demo
science-harness bench list
science-harness bench run configs/spring-offline.json --output artifacts/spring-demo
science-harness bench report artifacts/spring-demo
```

These examples need **no credentials, model calls or benchmark downloads**.
The notebook example uses a finite scripted fixture and ends inconclusively;
it demonstrates storage, retrieval and resource limits, not discovery ability.
Every run requires a new output directory.

## First integration: DiscoverPhysics

Create one external-notebook investigation without running it:

```bash
science-harness bench init --output my-study.json
science-harness bench validate my-study.json
science-harness bench run my-study.json --dry-run
```

`init` defaults to external notebook with agent-selected experiments. Choose alternatives explicitly:

```bash
science-harness bench init --output history-study.json --memory full-history
science-harness bench init --output reviewed-study.json --review
science-harness bench init --output random-study.json --random
```

| Choice | What changes |
|---|---|
| External notebook, default | Fresh scientist contexts; durable notebook and bounded reads |
| Full history | Full native conversation history and inline notebook |
| Review | A separately charged, advisory experiment-design critique |
| Random | Only executed experiment selection; the same scientist still analyzes and fits laws |

Review + random is not qualified in this adapter and is rejected, not silently
changed. Policies are versioned; notebook interfaces differ across adapters.

See [native setup and budgets](docs/benchmark-studies.md) before execution.
The native adapter uses the pinned Python/SciPy trajectory interface and a
qualified **macOS sandbox**, Node SDK and Copilot runtime. It is not a Linux
sandbox implementation, arbitrary coding agent, or official-paper reproduction.
No unsafe host-execution fallback is provided. Portable core installation
does not install these optional dependencies.

Paid execution requires a second, explicit approval matching the configuration:

```bash
science-harness bench run my-study.json \
  --output artifacts/my-study --approve-credits 1000
```

The example's 1,000-credit cap is a maximum, **not an expected price**.
Inspect the generated config first. Reads, reviews and repairs count toward
its request limit; unknown charges retain their full reservations. No silent retries.

## Reuse and extend

- [Architecture and scientific boundaries](docs/architecture.md).
- [Add a benchmark through the adapter registry](docs/adding-a-benchmark.md).
- [Integrate models and trusted tools](docs/models-and-tools.md).
- [Completed experimental results and limitations](docs/results.md).
- [Second integration: a real, zero-API NewtonBench notebook reference run](docs/second-integration.md).
- [NewtonBench: an LLM chooses experiments and discovers an expression](docs/newtonbench-discovery.md).
- [MADE: real CPU analytic-oracle integration and controlled crystal optimization](docs/made.md).
- [Two real model-led integration attempts: outcomes and limits](docs/crossbench-results.md).
- [Contributing](CONTRIBUTING.md), [security](SECURITY.md), and [publishing](docs/publishing.md).

The study coordinator does not introduce a second scientist loop. It reuses
`Scientist`, `BookStore` and `ToolRegistry`, or a pinned native controller where
benchmark compatibility requires one. A new benchmark owns its instruments,
action/submission schema and private evaluator; it need not pretend to be physics.

The Python import namespace is **`science_harness`**; the distribution and CLI
are **`science-harness`**. For example:

```python
from science_harness import Scientist, BookStore, ToolRegistry
from science_harness.harness import BenchmarkSpec, register
```

See [the namespace and settings migration](docs/migration.md). Historical
records retain their original labels: B means full history; C means external
notebook. New configs and documentation use the descriptive names. Old B/C CLI
aliases remain accepted for existing scripts.

## License

First-party harness code is **MIT**, unchanged from this project's existing
license. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
Benchmark code, data, model weights, provider access and generated run artifacts
have their own terms. They are not relicensed or bundled by this repository.
