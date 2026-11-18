# NewtonBench: model-led discovery

This profile uses the real pinned NewtonBench simulator and the existing
`Scientist`, `BookStore` and trusted analysis tools. Unlike the
[deterministic reference proof](second-integration.md), a model chooses the
experiments, notebook operations, analyses and final expression. There is no
reference-policy fallback or built-in answer supplied to the model.

## Scope and setup

The versioned `newtonbench-discovery` integration currently supports one
`m9_hooke_law` configuration: easy, `vanilla_equation`, law version `v1`,
measurement noise 0.01. It is not the complete NewtonBench suite.

```bash
python -m pip install -e '.[analysis,newtonbench]'
npm ci
git clone https://github.com/HKUST-KnowComp/NewtonBench.git .cache/NewtonBench
git -C .cache/NewtonBench checkout 912a4ba5f4356ddd06acc16e44460ca30be4abc2
science-harness bench init --benchmark newtonbench-discovery --output newton-agent.json
science-harness bench validate newton-agent.json
science-harness bench run newton-agent.json --dry-run
```

Inspect the model, cached CLI version and limits before approving a paid run:

```bash
science-harness bench run newton-agent.json \
  --output artifacts/newton-agent --approve-credits 1500
```

The default is one external-notebook run, at most 24 model decisions and
12 started instrument calls. A call may request repeated sensor readings;
those readings are not independent instrument acquisitions. The model can fit
or calculate with the harness-owned safe analysis interface; arbitrary Python
execution is not exposed. A concluded result requires a frozen candidate,
a new acquisition, and an observed-data predictive check before finishing.

## Evaluation and audit

After the investigator finishes, a separate trusted simulator worker evaluates
the frozen arithmetic expression on 500 clean held-out points. The existing
`custom-newton-predictive-v1` relative-RMSE threshold is 0.05. This is not
NewtonBench's official symbolic/LLM judge. No evaluation score, hidden law or
held-out point enters the scientist context or public lab notebook.

Each decision uses a fresh initialized zero-tool Copilot context; only the
generic scientist's public context and explicit notebook retrieval carry
scientific memory. The study keeps source snapshots, full outgoing packets,
provider quotes, conservative pre-dispatch reservations, authoritative
settlements and owned-session shutdown evidence. A failed or ambiguous paid
request retains its full reservation and stops the study; there is no automatic
transport retry, model substitution or best-law restoration.
The pinned Copilot runtime requests up to 32,000 output tokens; its input
allowance is 32,768. This is a declared transport allowance, not a promise that
each short JSON decision will consume that many tokens.

The 1,500-credit example is a new ceiling, not an expected charge or a transfer
from an older experiment. One successful attempt would demonstrate this
restricted integration, not broad discovery ability or a memory-policy effect.
