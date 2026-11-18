# Native protocol reference

Configuration, profile versions and evaluation rules for the pinned
DiscoverPhysics adapter. Start with the [setup guide](benchmark-studies.md)
for installation and commands.

## Configuration fields

| Field | Meaning |
|---|---|
| `schema_version` | Public config schema, currently 1 |
| `benchmark`, `protocol` | Registered adapter and its declared scientific interface |
| `worlds`, `seeds` | Explicit paired tasks/starts; arrays contain distinct values |
| `model` | Supported `provider` and `id`; credentials stay in the environment |
| `limits` | Per-episode `requests` and `observations`, whole-study `credits` ceiling |
| `arms` | Explicit named policy combinations; never an implicit factorial expansion |
| `parameters` | Adapter-specific validated settings, such as a pinned checkout and calibration |

Arm fields are `name`, `memory` (`none`, `inline`, `external`), `advisor` (`off`,
`design`), `selection` (`active`, `random`) and a versioned `profile`.
The registry declares which combinations actually exist for each protocol.

## Native profiles

`legacy-v1` leaves new instructions, external memory and reviews disabled.
Memory `none` maps to the native DiscoveryAgent; `inline` to ScienceAgent.
It preserves the original 100-request budget when configured that way.

`memory-advisor-v1` supplies the common experiment-intent guidance, the selected
memory delivery and optional design review. The declared request limit covers
all scientist, lookup, review, commitment and repair calls together. A fresh
advisor is constructed for every investigation. It always uses the same model
as the scientist. Reads happen before the draft; the one post-review commitment
uses the already selected evidence and bounded advice.

Full-history and external-notebook policies use the same notebook-entry schema.
In both reviewed arms, the advisor reads the complete acquired prefix.
The external-notebook scientist commitment receives only its
selected evidence, draft and bounded review, not the advisor's full input.
External memory changes session continuation as well as text delivery; lower
scientist context does not necessarily mean lower total-system cost.

The external notebook records its retrieval protocol version independently.
Version4 accepts plain narration around a read and normalizes repeated read-only
lookups only when their arguments are equivalent after applying the open-page cap.
It preserves the original public output, repeat count and differing original
arguments in the journal. Version3 accepted only exact repeated envelopes.
Different pages, query/revision changes and mixed scientific action tags still fail.
Oversized positive `notebook_open` limits produce an explicitly capped6,000-character
page and cursor, not an unbounded response or a silent correction. These changes
were prompted by the first engineering pilot's protocol failures, tested offline,
and must not be used to rewrite its recorded version2 outcomes. No read allowance,
model-request budget, scientific action or evaluation rule changes with version4.

Version5 adds explicit `reads_remaining` and recoverable read-budget denials.
The live version4 retest served every lookup successfully, but a seventh request
in one decision terminated the investigation. Version5 serves no additional data:
it journals the denial and asks for a scientific decision from already delivered
evidence. Denied attempts and corrections consume the existing total request cap;
repeated noncompliance ends at that cap. This intentionally changes C's overflow
handling, not legacy/full-history behavior, and does not rewrite the failed version4 outcome.
Version5's first paid qualification attempt did not exercise read-denial recovery:
it stopped on a separate duplicated-scientific-packet validation error. Failed
engineering outcomes remain frozen; observed-only operational checks, not private
prediction scores, gate subsequent qualification.

The shared modern experiment-intent/review path also handles an identical
complete proposal emitted in both public commentary and final-answer messages.
Only a single fully valid final proposal with identical complete commentary
copies gets a normalized **validation view**, with a separate journal receipt.
Raw responses, native-role history, source bytes and note provenance stay intact;
the pinned native controller executes the one experiment and optional fit once.
Different proposals, same-phase duplicates, missing transport identities and
partial code fragments are not deduplicated. Legacy B0 remains unchanged.

Optional `parameters.action_recovery: true` selects the separately versioned
`single-design-and-domain-v1` operational policy. The default remains false.
One unambiguous schema-valid experiment object is accepted as its one-element
array, with the same canonical controls used for preflight and execution.
Raw responses/history, Python source and notebook text remain unchanged, and
normalization is journaled separately. Exact source/probe coincidence is an
explicit unsupported launch, rejected before acquiring data or performing a fit;
the harness never moves the probe or changes the simulator force.

An invalid initial experiment draft gets at most one charged scientist correction
within the existing request cap. External-memory correction retains the already
selected evidence and serves no additional notebook reads. A second invalid
action stops explicitly. Review retains its single commitment: invalid commitments
still stop before acquisition rather than introducing another critique round.
The option applies to legacy as well as modern arms only when explicitly selected,
so such runs are a new operational epoch, not exact legacy replications.
Fitting, finalization, noise, private scoring and predictive thresholds are unchanged.
Keep repair diagnostics separate from the original frozen outcomes; selected
failed-case repeats are not replacement observations for a headline comparison.

For selector-only random controls use `selection: "random"` with the same memory
memory/profile settings, without a design advisor. The scientist still performs
all scientific analysis and law fitting; only the executed launch is sampled.
With opt-in action recovery, the prompt explicitly distinguishes unexecuted
proposals from the actual controls returned with each measurement. Unsupported
geometry in an ignored proposal cannot reject an otherwise valid random launch;
the selected launch still receives full validation before measurement.
Reads and fit-only turns neither acquire data nor advance the sampler.
For a fixed seed, full-history/random and external-notebook/random receive the same startup evidence,
random-control stream and per-acquisition noise at common prefixes.
Freeze the sampler/distribution and compare with historical active results
only with explicit source/operational-epoch and remote-service-drift caveats.

Native studies may explicitly declare up to 500 all-role requests per
investigation; defaults and existing registrations remain 100. Increasing
this allowance does not enlarge the experiment limit, context window, notebook
page/read allowances, numerical limits or monetary approval. All lookup,
repair and fit-only responses still count. Apply amendments only at safe
boundaries with a new source/config registration; never rewrite completed
100-call records as 500-call cases. Report call-limit/capacity outcomes separately
from scientific nonattainment. Historical active-versus-random comparisons
with different inference allowances are exploratory, not isolated selector effects.

The example's first arm, `legacy-baseline`, anchors paired reporting. Arm order rotates
by seed. A new output directory is required; incomplete studies are reported as
such rather than silently resumed or retried. Raw protocol journals remain
separate from the common `study.json` summary.

## Evaluation and historical comparability

Native position error is mean squared 2D displacement over 32 held-out launches
and five times, divided by the explicit world's `position_scale` squared.
The primary target remains <=0.01. Velocity scores use scale 1 in this adapter
and are labeled raw squared-velocity errors. Changing the calibration changes the
metric; there is no silent normalization fit.

Private panels are generated/registered and reference-checked before paid
dispatch. They are fresh study panels, not reconstructed historical arrays.
The extra-dimension profile uses balanced near/far starts; an overall target
pass still does not establish accuracy in both regimes.

To compare with original80, keep matching model/world/controller/selector cells,
native revision, measurement noise/cadence, fitting/final-validation behavior and
the original normalization. Pair fresh legacy B0 and new B with the new variants.
Report protocol epochs separately. A common model name cannot eliminate remote
service drift. Original private panels/per-candidate histories that were not
retained cannot be recreated as original evidence.

Summary reports distinguish assigned/recorded/censored episodes, first-hit cost
and retained final accuracy. Paired differences exclude censors and never infer
first-hit curves for adapters that measure only final success. Cache reads still
occupy model context; include all roles when measuring tokens and costs.

The Spring example and narrow NewtonBench reference use their existing metrics,
not the native physics threshold. Results from these interfaces are not an
official-paper leaderboard reproduction or a single pooled score.

## Inspecting a study

Study artifacts retain the exact config and source hashes; original public
messages, notes and programs; actual controls and measurements; fit, read and
review receipts; and failures, censors and costs. Reports include `report.json`,
`summary.csv` and paired `pairs.csv`.

Native frozen predictions and private panels remain separate from agent-visible
evidence. Evaluation does not guide prompts, retrieval, stopping or best-law
restoration. Completion alone is not scientific success.
