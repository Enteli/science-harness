# Canonical package and settings

The project now consistently uses:

| Surface | Name |
|---|---|
| Repository, distribution and command | `science-harness` |
| Python package | `science_harness` |
| Module invocation | `python -m science_harness` |
| Default local state directory | `.science-harness/` |
| Model environment setting | `SCIENCE_HARNESS_MODEL` |
| Compatible API endpoint | `SCIENCE_HARNESS_BASE_URL` |
| Compatible API credential | `SCIENCE_HARNESS_API_KEY` |
| Optional Copilot CLI version | `SCIENCE_HARNESS_COPILOT_CLI_VERSION` |
| Optional trusted npm project for the SDK | `SCIENCE_HARNESS_SDK_ROOT` |

The standard provider-specific `OPENAI_API_KEY` fallback is unchanged.
Credentials still belong outside scientific configs and source control.
The portable API remains:

```python
from science_harness import Scientist, BookStore, Observation, Tool, ToolRegistry
from science_harness.harness import BenchmarkSpec, register, load_config, run_study
```

This pre-publication namespace change does not install a compatibility import
package or a second legacy console command. Install the rebuilt distribution
in a fresh environment when checking migration; a previously installed package
can otherwise hide an incomplete rename.

Existing local databases, historical source snapshots, run artifacts and
previous release bundles are not renamed, rewritten or loaded automatically.
Keep them with their recorded source versions. Ignoring the old local-data
directory remains intentional protection against accidentally committing it.
Original copyright and contributor attribution are retained; they are not
active package names or a reassignment of ownership.

The notebook, memory-policy profiles, benchmark versions, model identifiers and
historic scientific results are not retroactively changed by this migration.
