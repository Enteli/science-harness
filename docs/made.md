# MADE: controlled CPU analytic optimization

This integration uses MADE's real `AnalyticOracle` (ASE Lennard-Jones potential)
and `Environment` extension interface at revision
`bffda0c9a1fa1904b35c0ef67b455881c975df86`. The model selects crystal
prototypes and lattice constants, observes energies, records a notebook and
submits a measured candidate through the same generic `Scientist` loop.

**This is an explicitly controlled optimization test, not MADE's Materials
Project/convex-hull paper benchmark and not discovery of a new real material.**
It requires no Materials Project key, external material dataset, pretrained
weights, GPU, generative model or hosted energy oracle.

## Task

The candidate space is a single species labelled Ar, prototypes `sc`, `bcc`
and `fcc`, and cubic lattice constant `a` in [1.2, 3.0]. The potential is not
calibrated to real argon. The upstream oracle evaluates fixed coordinates
without relaxing the structure; the objective is potential energy **per atom**,
not total energy across cells with different atom counts.

This repository supplies a small `Environment` subclass and a controlled local
`PhaseDiagramDataset` subclass; the latter does not call Materials Project.
It does not replace the oracle or return cached/hand-authored energies.

After finish, private numerical searches over all three prototypes establish a
reference minimum. Success requires relative energy regret at most 0.01,
improvement over the declared `sc, a=3.0` baseline, and a final candidate the
scientist actually measured. These are our custom
`controlled-made-lj-regret-v1` criteria, not AF, EF, AUDC or mSUN.
Private reference-oracle calls are reported separately from the agent's
measurement budget; no private optimum or grade is fed back to the scientist.

## Isolated setup

The harness core remains Python 3.9 compatible. MADE's pinned source requires
Python 3.12; keep this environment separate from the native physics dependencies.
For example, with `uv` available:

```bash
python -m pip install -e '.[analysis]'
uv python install 3.12
uv venv --python 3.12 .cache/made-analytic-env
uv pip install --python .cache/made-analytic-env/bin/python -e '.[made]'
git clone --filter=blob:none --sparse --no-checkout \
  https://github.com/diffractivelabs/MADE.git .cache/MADE
git -C .cache/MADE sparse-checkout set --no-cone /src/ /LICENSE /README.md /pyproject.toml
git -C .cache/MADE checkout bffda0c9a1fa1904b35c0ef67b455881c975df86
npm ci
science-harness bench init --benchmark made --output made-agent.json
science-harness bench validate made-agent.json
science-harness bench run made-agent.json --dry-run
```

Inspect the config and authorize a NEW paid run explicitly:

```bash
science-harness bench run made-agent.json \
  --output artifacts/made-agent --approve-credits 1500
```

The default limits are 24 model decisions, 12 agent-facing measurements and
a separate 1,500-credit maximum. The policy is external notebook, active
selection, no reviewer. The fresh zero-tool SDK contexts, accounting and
no-retry behavior are the same as the [NewtonBench discovery profile](newtonbench-discovery.md).

## Import and execution boundary

The pinned MADE package initializers eagerly import unrelated generative and
GPU-oracle stacks, even for an analytic oracle. The trusted worker therefore
registers only six fixed source namespaces and imports the actual, unmodified
upstream analytic oracle, base environment and dataset modules from the clean
pinned checkout. It deliberately does not execute those eager registry
initializers. This compatibility loader is explicit and version-bound, not
a fake oracle, substitute energy formula or altered upstream source file.

The worker interpreter is an operator-configured trusted Python environment,
never a model-selected executable. Install the same harness release into that
environment; startup verifies the worker's source hash. The launcher preserves
the virtual-environment interpreter path and does not inject another Python
version's site-packages into it. On macOS its network access is denied;
credentials are excluded from its environment and CPU thread limits are one.
It is a trusted simulator subprocess, **not an adversarial-code sandbox**.
No model-generated Python is executed.

MADE code is MIT. ASE and other optional dependencies retain their own licenses.
The benchmark code and datasets are not bundled in the source export.
