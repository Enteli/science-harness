# Licensing and third-party notices

Science Harness remains under the MIT license in `LICENSE`.
This is not a claim that every optional dependency, benchmark, model, input
dataset or generated program is MIT-licensed.

## External benchmarks

| Component | Pinned revision | Upstream license |
|---|---|---|
| [DiscoverPhysics](https://github.com/SampsonML/DiscoverPhysics) | `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5` | [MIT](https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/LICENSE) |
| [NewtonBench](https://github.com/HKUST-KnowComp/NewtonBench) reference and model-led adapters | `912a4ba5f4356ddd06acc16e44460ca30be4abc2` | [MIT](https://github.com/HKUST-KnowComp/NewtonBench/blob/912a4ba5f4356ddd06acc16e44460ca30be4abc2/LICENSE) |
| [MADE](https://github.com/diffractivelabs/MADE), controlled analytic-oracle integration | `bffda0c9a1fa1904b35c0ef67b455881c975df86` | [MIT](https://github.com/diffractivelabs/MADE/blob/bffda0c9a1fa1904b35c0ef67b455881c975df86/LICENSE) |

The repository contains our adapters, not copies of these upstream checkouts,
benchmark datasets or trained weights. Retrieve upstream code separately and
retain its license and notices. Cite the applicable benchmark paper as well as
the software version. An adapter is not an endorsement or leaderboard result.

MADE's **code** is MIT at the revision above. Its default workflows can use
Materials Project data/services and external interatomic-potential or generator
weights. Their access and redistribution terms must be reviewed independently;
the repository's MIT license does not grant those rights. Our optional analytic
profile uses a controlled local task through the real upstream oracle and
environment extension API, not Materials Project data or learned weights.
It is not the standard MADE materials-discovery benchmark.

Python and npm dependencies retain their respective licenses. The optional
ASE dependency, for example, is separately licensed; upstream MIT does not
relicense it. The optional
Copilot SDK is pinned in `package-lock.json`; using a provider also requires its
service terms, permissions and billing account. A software license does not
grant access to a model or a benchmark's separately licensed data.

## Publication policy

Private GitHub visibility controls repository access; MIT does **not** create a
confidentiality obligation. Do not treat either as an NDA. Confirm that all
contributors have the right to publish their changes before sharing.

Keep first-party MIT attribution. Do not add a blanket license header to
downloaded third-party code or assume that model-generated content inherits the
harness license. New vendored components require an explicit origin, version,
license review and preserved notices before inclusion.

MIT is retained for continuity and permissive reuse. Moving to a license with
different patent terms, such as Apache-2.0, is a deliberate future governance
decision, not something a packaging script silently changes. This document is
an inventory and project policy, not legal advice.
