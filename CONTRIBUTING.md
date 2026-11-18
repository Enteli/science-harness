# Contributing

Keep the public surface small: a scientist loop, durable evidence/notebook,
trusted tools, explicit policies, providers and benchmark adapters. Reuse an
existing engine rather than add another reasoning loop for each experiment.

## Development

```bash
python -m pip install -e '.[dev,analysis]'
python -m pytest -q -m 'not integration'
python -m build
```

Core tests and examples must not require credentials, provider calls, model
downloads or private benchmark records. Use scripted fixtures. Real benchmark
integration tests are opt-in, separately provisioned and never run against
physical instruments or paid providers from an ordinary pull request.

For an adapter, follow [the contract](docs/adding-a-benchmark.md), include an
offline example, and test privacy, failure, resource and lifecycle behavior.
For a provider/tool change, follow [models and tools](docs/models-and-tools.md).

## Compatibility and evidence

Version public configuration schemas, scientific protocols and policy profiles
independently from package releases. Fixes that change prompts, context delivery,
noise, fitting, legal actions, budgets or evaluation are not retroactive edits
to old results. Preserve old traces and report the new source/protocol epoch.
Keep experimental claims separate from transport, capacity and protocol censors.

Dependency update automation opens proposals only. Maintainers must review
model capabilities, tokenizer/usage semantics and sandbox behavior; passing
portable tests alone does not qualify a new native runtime.

Do not commit run directories, private panels, credentials, billing releases,
checkpoint archives, model weights or another project's code without permission.
Contributions must be yours to submit under this project's MIT license; retain
applicable third-party notices and explain origins of borrowed components.
No contributor-rights transfer or blanket relicensing is implied.

Be respectful, review scientific disagreement on evidence, and report security
issues privately as described in `SECURITY.md`.
