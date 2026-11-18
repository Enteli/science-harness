# Private first, open-source ready

Start the new repository from the clean source export, **not the research Git
history**. A private repository does not make old secrets safe to publish later.
The export contains a hash inventory in `EXPORT-MANIFEST.json`; it does not
contain benchmark checkouts, run data, operational releases or a `.git` directory.
The migration exporter is intentionally not part of the resulting project.

## Before the first push

1. Review every exported path and run a proper secret scan. A filename allowlist
   and limited token check are not a complete content audit.
2. Verify first-party contribution rights, MIT attribution and third-party notices.
   Do not add private datasets, raw provider logs, model weights or generated laws.
3. Run offline tests, build source/wheel distributions and test the installed wheel.
4. Choose the real repository owner/name and confirm **private** visibility.
   Do not reuse the original repository's remote or push its branches/history.
5. Configure protected default branches, required CI review and available private
   security reporting/secret scanning. Do not auto-merge dependency/runtime updates.

After review, from the clean directory:

```bash
git init
git add .
git commit -m "Initial reusable science harness"
# Replace OWNER/REPOSITORY deliberately; this command creates and pushes a PRIVATE repo.
gh repo create OWNER/REPOSITORY --private --source . --push
```

These are operator instructions, not something installation, tests or the
exporter execute. GitHub creation/push and later public visibility changes
require separate authorization.

## Releases

```bash
python -m pip install -e '.[dev,analysis]'
python -m pytest -q -m 'not integration'
python -m build
```

Review both wheel and source archive contents. Core installation stays portable;
the qualified native adapter still needs the documented editable source/Node/
macOS setup. Do not publish wheels that pretend to bundle a sandbox or provider.

Use semantic package versions, explicit scientific protocol/profile versions,
changelogs/release notes and adapter capability tests. Requalify native execution
and usage accounting after SDK/model/backend changes; portable CI is not a paid
benchmark. Publishing to PyPI, changing licenses or making a repository public
is a separate deliberate release decision.
