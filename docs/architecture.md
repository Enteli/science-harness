# Architecture

The portable kernel is `Scientist` + `BookStore` + `ToolRegistry`. The public
study API in `science_harness.harness` configures and coordinates investigations;
it does not reason about a particular scientific domain.

```text
configuration -> adapter registry -> explicit study cells
                                      |
                         trusted benchmark adapter
                                      |
                    scientist <-> public evidence/notebook
                       |                    |
                 named tools          bounded retrieval
                       |
              validated acquisitions/analysis

private evaluator -> saved, sealed scientific outcomes -> report
                     (never back into the agent loop)
```

## Small, stable extension points

| Surface | Responsibility |
|---|---|
| `Scientist` | Existing model/tool investigation loop; generic notebook mode is `memory_policy="notebook"` |
| `BookStore` | Append-only evidence, revisioned knowledge and operator-approved promotions |
| `ToolRegistry` | Trusted named instruments, argument binding and acquisition-resource contracts |
| `LLM` | `complete(system, user) -> str`; providers do not define scientific success |
| `BenchmarkAdapter` | Pure validation, preparation, one episode, normalized result, cleanup |
| `Arm` | Explicit, versioned memory/review/selection choices supported by the adapter |

The generic notebook is a durable ledger view with source-linked notes and
explicit search/fetch. The native DiscoverPhysics notebook preserves its own
qualified Markdown/open/grep interface and upstream controller. These are
related design choices, not interchangeable scientific protocols.

## DiscoverPhysics binding

The first native integration retains the existing controller and trajectory
interface, Python/NumPy/SciPy law programs, observation handling, isolated
numerics and sealed predictions. It declares a pinned upstream revision and
explicit normalization. It does not convert other tasks into trajectory fitting.

External-notebook memory supplies a bounded working packet and notebook access
in fresh contexts; full-history memory retains native conversation history.
Notebook claims are not measurements.
Review is advisory and separately counted. Random changes experiment selection,
not the analyst, law proposal, fitting tools or acquired-evidence access.
Exact review/fitting/acquisition timing belongs to the versioned profile.

Operational artifacts record intended versus actual controls, exact programs,
fit coefficients, observed failures, model usage and source hashes. Missing
feedback or billing is not fabricated. Model-request and physical-acquisition
ceilings are separate. No private-score early stopping or automatic best-law restore.

## Evolving the project

Keep benchmark-specific validation and legal scientific actions in adapters;
keep provider-specific transport/usage semantics in providers. Prefer explicit
trusted registration over automatic plugin discovery. Declare unsupported
combinations and fail validation rather than quietly change an experiment.

A model upgrade is a new measured condition, not proof of algorithmic progress.
Pin model identity, reasoning/context settings, provider/runtime versions,
benchmarks, budgets, noise, tools and scoring before comparisons. Package
versions do not replace scientific protocol versions.
