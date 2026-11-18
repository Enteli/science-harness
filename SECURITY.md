# Security

## Trust boundaries

- Treat model output, notebook text, retrieved material and benchmark
  observations as untrusted data, never permission to add tools or budgets.
- Register adapters, models and tools only through trusted application code.
  JSON configuration cannot import arbitrary modules or install plugins.
- Generic tools are operator-provided Python functions. Their implementation
  must validate arguments and declare acquisition costs; registering a tool
  does not sandbox that function or make a physical instrument safe.
- Native DiscoverPhysics generated programs run through the qualified macOS
  isolated worker with finite time, memory and thread limits. Unsupported
  platforms fail closed. There is no unsafe host fallback.
- Candidate code and trusted numerical fitting share a worker interpreter.
  This is **not a demonstrated adversarial Python sandbox or VM boundary**.
  Do not execute hostile third-party programs on a valuable workstation.
  Use a disposable, restricted environment with no unrelated data or credentials.
- Private evaluation arrays, scores and validity flags must remain outside
  scientist, notebook-retrieval, reviewer and stopping-policy inputs.

## Credentials and private data

Credentials belong in provider-supported environment/configuration mechanisms,
not study JSON, notebooks, source control or command-line arguments. Only use
trusted provider endpoints and credentials scoped to the intended service.
Never attach authorization headers or SDK authentication files to an issue.

Notebooks and full traces may contain proprietary measurements, prompts and
generated programs. Study output is local and sensitive by default; place it
under ignored `artifacts/`, restrict access, and explicitly review any export.
There is no claim of automatic anonymization or secret removal from run data.
Git ignore rules and the source export's limited token check are defense in
depth, not a complete secret scanner.

## Spending and failure handling

Validation, configuration generation and dry-run do not dispatch models.
Paid studies require an explicit configured cap and matching approval.
Reserve before dispatch, reconcile only complete source-bound accounting
evidence, and keep the full reservation when usage is uncertain.
An aggregate user-initiated counter is not necessarily the total model-call
count; a zero value alone never proves free or absent inference.
Do not automatically retry provider refusals, ambiguous paid requests or
irreversible laboratory actions. A stopped output directory is not resettable
by deleting a marker.

## Dependencies and reporting

Use maintained Python/Node releases; Python 3.9 compatibility is for historical
environments, not an endorsement of an end-of-life interpreter. Pin benchmark
revisions, use `npm ci --ignore-scripts`, and review dependency updates against
protocol/usage contracts. CI has read-only repository permissions, pinned
actions, no model credentials and no paid scientific runs.

Before making a private repository public, enable available secret scanning,
push protection and private vulnerability reporting in its settings. Feature
availability depends on the hosting plan; this project does not enable paid
hosting features automatically.

Report suspected vulnerabilities privately through GitHub private vulnerability
reporting when enabled, or contact the maintainers privately. Do not post
credentials, private datasets or exploit-bearing notebooks in a public issue.
