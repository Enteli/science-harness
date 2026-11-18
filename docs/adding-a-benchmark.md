# Adding a benchmark

The public runner is an episode/study coordinator, not a second scientist loop.
Reuse `Scientist`, `BookStore` and `ToolRegistry`, or bind a versioned upstream
protocol when compatibility requires its controller. Keep domain-specific
instrument validation, analysis, submission formats and evaluation in the adapter.

## Code map

`science_harness/harness/` owns public study configuration, the standalone CLI and orchestration;
`benchmarks/` owns domain/runtime bindings; `policies/` declares versioned options;
`providers/` handles public accounting helpers; and `reporting/` aggregates saved
results. The existing `Scientist`, `BookStore`, `ToolRegistry` and `physics_*`
engines remain the execution/compatibility surfaces rather than being replaced by
another loop. See `configs/` for explicit study examples and `tests/` for contracts.

## Minimal interface

Register a trusted adapter before loading its configuration:

```python
from science_harness.harness import BenchmarkSpec, register

register(
    BenchmarkSpec(
        name="my-benchmark", protocol="my-protocol-v1",
        worlds=("apparatus",), providers=("baseline",),
        profiles=("my-profile-v1",),
        runtime="Python plus the benchmark's documented prerequisites",
        description="A bounded scientific task",
    ),
    lambda: MyBenchmarkAdapter,
)
```

The loader returns a class with:

| Method | Contract |
|---|---|
| `validate(config)` | Static/pure validation of benchmark-specific `parameters` and supported policies. No instruments, network, models or grading. |
| `prepare(config, output, approved_credits)` | Check runtime/dependencies, pin scientific provenance, register private evaluation and initialize permitted resources. |
| `run_episode(cell, directory)` | Run one isolated investigation using an existing scientist engine; persist exact evidence and return a normalized result. |
| `close()` | Release resources, including after partial setup/failure. |

Use `load_config()` and `run_study()` from `science_harness.harness` after registration,
or call the CLI from that same launcher process. JSON configs deliberately do not
import arbitrary Python modules or auto-install plugins.

The result has `status`, boolean `final_pass`, integer `requests` and
`observations`, and a benchmark-specific `metrics` object. Optional `first_hit`,
`capped_observations`, `censored`, `usage_by_role`, `accounting` and `stop_study`
support richer comparisons. First-hit values are null when not measured, not
invented from final success. The runner verifies reported request/observation
limits; the engine/transport must enforce them *before* side effects.

If private grading fails after an accepted final, preserve that final before
calling the evaluator. Do not turn an unavailable score into a scientific
failure: return `final_pass: null`, `evaluation_status: "failed"`,
`censored: true`, `stop_study: true`, and a nonempty `metrics.error`.
The runner retains the episode, writes a failed study/report, and surfaces
the error without starting another investigation. Reports count `ungraded`
and `evaluation_errors` separately from `final_passes`.

Optional `accounting` is **episode-local**, not a cumulative study snapshot.
The reporter sums episode rows; record charge deltas (including any new
unresolved hold) so multi-seed/arm costs are not counted repeatedly.

`run_episode` must retain the scientific records, not just return a score.
Use one directory/evidence ledger per investigation. Preserve failed attempts;
do not rerun irreversible acquisitions during recovery.

## Domain independence without false equivalence

An adapter declares acquisition units: one preparation, assay, trajectory or
instrument invocation is not necessarily one independent sensor reading.
Expose trusted acquisition cost contracts through `ToolRegistry` so started
work is charged even when it fails. Keep fitting or hypothesis-test tools
available equally across compared arms.

Policies consume public scientific evidence and drafts. The adapter supplies the
control schema, legal actions, analysis/fitting capabilities and evaluator.
Do not force a catalyst search to submit a trajectory function, or treat a
different prompt/schema as the same benchmark merely because it shares a name.
Unsupported policy combinations must fail validation.

Use `science-harness bench init --output my-study.json` for an external-notebook
DiscoverPhysics configuration. Other adapters must explicitly declare their
own memory/review/selection semantics; a shared option name does not qualify
an unsupported treatment.

The first public bindings qualify notebook/design-review ablations for the
native DiscoverPhysics protocol. The Spring/NewtonBench reference adapters show
the registry/lifecycle over the generic Scientist; they do not claim those same
LLM ablations or complete coverage of the upstream benchmarks.

## Safety and reproducibility checklist

- Inputs from the scientist are untrusted; no arbitrary shell, filesystem paths
  or generated-code execution on the host.
- Private evaluation data/scores cannot enter the evidence store, notebook,
  advisor packet, retrieval path or stopping decision.
- Model calls, reads, repairs and assistance must be included in the declared
  cap; all paid providers need an approved, pre-dispatch reservation mechanism.
- Use a new study ledger, preserve unknown holds, and stop on ambiguous billing.
- Retain exact sources, coefficients, actual conditions, measurements, failure
  classifications and all provider-reported usage. Missing usage is unknown,
  not zero.
- Pin external code/data and declare runtime capabilities. No automatic download
  or paid dispatch during validation/dry-run.
- Add adapter contract tests, an offline example and a declared success metric.
  Preserve old protocol versions instead of rewriting historical outcomes.

`tests/test_public_harness.py` contains a tiny registered adapter proving that a
new benchmark does not require edits to the study runner.
