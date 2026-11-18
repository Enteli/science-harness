# Two bounded model-led integration demonstrations

These are real model-selected experiments, not scripted reference successes.
Each is one registered attempt with Claude Opus 5.5 through Copilot, CLI
`1.0.103-0`, SDK `1.0.28`, seed `20261118`, external-notebook configuration,
at most 24 model decisions and 12 agent-facing measurements. Each had its own
fresh 1,500-credit ceiling. No old study funds or unresolved holds were reused.

See the [machine-readable proof](crossbench-discovery-proof.json) for exact
metrics, submissions and artifact hashes.

| Integration | Model decisions | Agent acquisitions | Result | Actual credits |
|---|---:|---:|---|---:|
| NewtonBench, easy vanilla energy, law `v1` | 13 | 6 (55 sensor readings) | Relative RMSE 0.002011735 on 500 clean held-out points; threshold 0.05 | 52.08208 |
| MADE, controlled analytic crystal task | 13 | 12 | Relative energy regret 0.000203173; threshold 0.01 | 59.37134 |

Both frozen finals passed their declared custom criteria. All **26 requests**
have authoritative, complete charge and owned-session shutdown evidence;
total new spending was **111.45342 Copilot credits**, with no unresolved request.
Investigator-loop wall times were about 132 and 156 seconds respectively.
Those times exclude setup, package validation and the final private evaluator.

## NewtonBench: quantitative law recovery

Using only its selected noisy measurements and trusted analysis, the model
submitted:

```text
463.116581641002 * x ** 0.5008370668751576
```

This is approximately a square-root energy law, not the textbook quadratic.
The model chose displacements spanning 0.001 to 1, used the harness's
`fit_model`, froze a candidate, acquired new observations and performed
observed-data checks before finishing. The hidden-law worker then scored
the submitted expression once on 500 clean held-out values. Relative RMSE
was **0.2012%**, below the declared 5% threshold.

There were three counted protocol errors: the model wrapped proposed final
JSON in Markdown fences, which the strict parser rejected. Subsequent
scientist decisions repaired the format within the same attempt and request
budget. These were not transport retries, discarded runs or free inference.
The natural-language conclusion is a model claim, not an independently
certified description of every training/check detail; the frozen expression,
analysis receipts and private evaluation determine the reported result.

This is one restricted NewtonBench task, not its complete suite, official
symbolic/LLM score, evidence of unique identification or new physical knowledge.
The older [deterministic reference proof](second-integration.md) remains separate.

## MADE: controlled analytic optimization

The model chose the queried prototypes and lattice constants, compared energies
per atom, refined a promising region and submitted an actually measured
**fcc cell with `a=1.55`**. Its energy was **-7.936291 eV/atom**. A separate
post-finish numerical reference search across the three allowed prototypes
found **-7.937904 eV/atom** at approximately `a=1.546379`, giving **0.02032%**
relative energy regret. The model's own conclusion correctly noted that it had
not measured the simple-cubic family or bracketed the bcc minimum.

The real pinned MADE `AnalyticOracle` performed the energy evaluations, through
our documented `Environment` extension. No oracle response was mocked and no
reference optimizer chose the model's experiments. The 350 private reference
oracle calls happened after finish and are separate from the 12 agent queries.
The uncalibrated Lennard-Jones potential and species label Ar define a
controlled simulation, **not a new real material, Materials Project result,
native convex-hull benchmark or MADE-paper leaderboard result**.

## What is and is not established

Both attempts used the existing generic `Scientist`, resource accounting and
durable notebook ledger. Neither made an explicit notebook fetch or manual
notebook-note action; public measurements and calculation context were enough
for these small tasks. Thus these runs do not demonstrate a retrieval benefit,
a memory-policy effect or reliable success rates.

Truth and private scores never entered scientist contexts or public notebooks.
No saved model-generated Python was executed. Each decision used a fresh,
initialized zero-host-tool context, without model substitution or automatic
transport retry. Source snapshots, original packets, noisy measurements,
accepted finals, protocol errors, charges and journals remain preserved.
The two numerical criteria are task-specific and should not be compared as
if they were the same score.
