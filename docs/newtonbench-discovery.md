# NewtonBench: model-led discovery

The generic `Scientist` chooses experiments, analyses and a final energy-law
expression against the real NewtonBench simulator. No answer or reference-policy
fallback is supplied. The [scripted reference proof](second-integration.md) is separate.

## Scope and setup

Scope: `m9_hooke_law`, easy, `vanilla_equation`, law `v1`, noise 0.01.
This is not the complete NewtonBench suite. Run from a source checkout with
cached Copilot CLI `1.0.103-0` and model access:

```bash
python -m pip install -e '.[analysis,newtonbench]'
npm ci
git clone https://github.com/HKUST-KnowComp/NewtonBench.git .cache/NewtonBench
git -C .cache/NewtonBench checkout 912a4ba5f4356ddd06acc16e44460ca30be4abc2
science-harness bench init --benchmark newtonbench-discovery --output newton-agent.json
science-harness bench validate newton-agent.json
science-harness bench run newton-agent.json --dry-run
```

Inspect the generated config, then approve a NEW paid run:

```bash
science-harness bench run newton-agent.json \
  --output artifacts/newton-agent --approve-credits 1500
```

Defaults: external notebook, 24 model decisions, 12 started instrument calls,
1,500-credit maximum. The cap is **not an expected price**. Repeated readings
within a call are not independent acquisitions. Fitting/calculation uses trusted
analysis tools, not arbitrary Python. Before a concluded finish, the model must
freeze a candidate and check it against a new acquisition.

## Evaluation and audit

After finish, the trusted worker evaluates the frozen expression on 500 clean
held-out points. `custom-newton-predictive-v1` requires relative RMSE below 0.05;
it is **not NewtonBench's official symbolic/LLM score**. Hidden laws, points and
grades never enter scientist context or the public notebook.

Every decision uses a fresh, initialized zero-host-tool context. Sources,
packets, measurements, charges and shutdown evidence are retained. Ambiguous
charges keep their reservations and stop the study: no automatic transport
retry, model substitution or best-law restoration. Transport allowances are
32,768 input and 32,000 output tokens, not expected consumption.

See [the recorded pilot](crossbench-results.md) and
[SDK setup](models-and-tools.md#copilot-sdk-location). One successful attempt is
an integration check, not a discovery success rate or memory-policy effect.
