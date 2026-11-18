# MADE: controlled CPU analytic optimization

The generic `Scientist` chooses crystal cells using MADE's real `AnalyticOracle`
(ASE Lennard-Jones) and `Environment` extension API at
`bffda0c9a1fa1904b35c0ef67b455881c975df86`.

**This is controlled simulated optimization, not MADE's Materials Project/paper
benchmark or a new real material.** No external dataset, API key, pretrained
weights or GPU is needed.

## Task

Choose `sc`, `bcc` or `fcc` and a cubic lattice constant `a` in [1.2, 3.0].
The species label Ar is not a calibration to real argon. Coordinates stay
fixed: minimize potential energy **per atom**, without automatic relaxation.
Our small environment and local-dataset subclasses use actual upstream oracle
energies, not mocked responses or Materials Project access.

After finish, a private numerical search supplies the reference. Success means
relative regret at most 0.01, improvement over `sc, a=3.0`, and submission of an
actually measured candidate. `controlled-made-lj-regret-v1` is our metric, not
AF/EF/AUDC/mSUN. Private reference calls are counted separately from agent
queries; neither the optimum nor the grade reaches the scientist.

## Isolated setup

Use Python 3.12 for the MADE worker, separate from the host/native physics
environment. From a source checkout, with `uv`, Node and cached Copilot CLI
`1.0.103-0` available:

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

Inspect the config before approving a NEW paid run:

```bash
science-harness bench run made-agent.json \
  --output artifacts/made-agent --approve-credits 1500
```

Defaults: external notebook, active selection, no reviewer; 24 model decisions,
12 agent queries and a separate **1,500-credit maximum, not expected spending**.
Fresh zero-host-tool contexts and no-retry accounting follow the
[NewtonBench profile](newtonbench-discovery.md). See
[SDK setup](models-and-tools.md#copilot-sdk-location) and [pilot results](crossbench-results.md).

## Import and execution boundary

The pinned package eagerly imports unrelated generative/GPU stacks. A fixed
six-namespace loader skips those registries while importing the **unmodified**
oracle, base environment and dataset modules.

Install the same harness release in the trusted worker environment; startup
checks its source hash. The launcher preserves its interpreter and site-packages,
strips credentials, limits CPU threads to one, and denies network on macOS.
It is **not an adversarial-code sandbox**; no model-generated Python executes.
MADE code is MIT; ASE and other dependencies keep their own terms. Upstream code
and datasets are not bundled.
