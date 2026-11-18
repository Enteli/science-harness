# Second integration: NewtonBench

This page records the deterministic reference proof. The separate
[model-led discovery profile](newtonbench-discovery.md) uses real model-selected
experiments and is not the reference policy described below.

This is a real end-to-end integration of a **small, explicitly restricted**
NewtonBench task, not a second LLM discovery study or a reproduction of the
complete NewtonBench leaderboard.

## What is exercised

The public adapter registry runs the existing generic `Scientist`, real durable
`BookStore`, one explicit notebook write and retrieval, resource accounting,
the real pinned upstream Hooke-law simulator, a constrained expression
submission and a separate numerical evaluator. No simulator responses are
mocked. The decision policy is a deterministic power-law reference, not an LLM.

Scope: `m9_hooke_law`, easy, `vanilla_equation`, law version `v0`. A fixed family
and sampling strategy make this useful for testing integration, not evidence
of general discovery ability or a memory-policy advantage.

## Reproduce with no model credentials

From a source checkout, using a maintained Python release:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[newtonbench]'
git clone https://github.com/HKUST-KnowComp/NewtonBench.git .cache/NewtonBench
git -C .cache/NewtonBench checkout 912a4ba5f4356ddd06acc16e44460ca30be4abc2
science-harness bench init --benchmark newtonbench --output newton-study.json
science-harness bench validate newton-study.json
science-harness bench run newton-study.json --output artifacts/newton-notebook
science-harness bench report artifacts/newton-notebook
```

The generated config has zero credits and the `notebook-reference-v1` profile.
No API key, hosted model, upstream LLM judge, model weights or separate dataset download
is used. The optional OpenAI package is an upstream import dependency only.
The benchmark code itself is retrieved separately, not bundled.

The simulator worker is a managed **trusted** subprocess on Linux/macOS, not
the generated-Python macOS sandbox used by DiscoverPhysics. Its inputs are
bounded scalar controls; submitted expressions use our restricted arithmetic
interpreter, not unrestricted Python.

## Recorded integration proof

See [the machine-readable proof](second-integration-proof.json). The seeded
reference run made 9 policy decisions: notebook write, notebook read,
6 instrument acquisitions and final submission. The 6 acquisitions contained
54 noisy sensor readings; those are not 54 independent experiments.

Our adapter's `custom-newton-predictive-v1` score uses 500 clean held-out points
and a relative-RMSE threshold of 0.05. It is explicitly **not the official
NewtonBench score**, which includes different evaluation machinery.
Private evaluator output is produced after the investigator finishes and is
not included in model requests or the public lab notebook.

The recorded seed-0 relative RMSE was **0.00378305**, below 0.05. The two
end-to-end integration tests (seeds 0 and 42) passed. This is a pipeline check,
not two independent AI discovery successes.

Raw run records remain under the local output directory; the source bundle
contains only this bounded proof summary and repeatable commands.
The integration tests check worker shutdown. Release validation additionally
exercises the installed `science_harness.testbeds.newton_worker` module from an
empty working directory, rather than relying on imports from the research checkout.

## Why NewtonBench before MADE?

Both repositories' **code** is MIT at the revisions in
[third-party notices](../THIRD_PARTY_NOTICES.md). NewtonBench provides a small
numerical environment that can be demonstrated without model weights, external
data-service credentials or paid inference.

MADE's [controlled CPU analytic integration](made.md) is now separate from this
NewtonBench reference proof. Its standard materials-design workflows
involve data/oracle/generator choices and potentially separately licensed weights
and Materials Project access. MIT code does not clear those dependencies.
NewtonBench remains another physics task: this proof does not establish transfer
to materials design, biology or real laboratory hardware.
