# Services: why parts of LORRAX live outside LORRAX

If you clone this repository you will find, alongside the `src/` tree that is
LORRAX proper, a `services/` directory holding six independently installable
Python packages. They are not vendored third-party code and they are not a
plugin system. They are pieces of LORRAX that were deliberately cut out of it,
given their own package boundary, their own test suite, and a rule that they
may not import LORRAX back. This page explains why that was worth doing, what
the boundary actually enforces, and where to look for each thing.

## The failure class that motivated all of this

Every service in `services/` was extracted after the same accident happened to
the code it replaced. The accident has a recognisable shape: **one operation
existed in several copies, the copies drifted, and the drift did not announce
itself.**

The distributed eigensolver was dispatched from ten modules across four
packages, each with its own ladder of "is this backend available" checks, and
the ladders disagreed. The measured consequence was a run that changed which
library it used, completed with `rc=0`, and reported a quasiparticle gap wrong
by **−161 eV**. The symmetry stack had a conjugation predicate — one boolean —
hand-written at several call sites; applying it against the wrong reference row
costs **183.61 eV** in the off-diagonal self-energy while leaving norms,
hermiticity, traces, electron counts and every printed eqp column *exactly*
unchanged, so nothing in the test suite could see it. The ζ-file reader carried
its layout dispatch in three copies, one of which probed only the legacy
dataset name and therefore silently passed on precisely the production files it
was guarding. The wavefunction loader's per-rank band clamp had a local copy
that diverged on the band axis and wrote real file bands into padding on every
non-divisible geometry — invisible at the divisible defaults the old harness
happened to use. The Coulomb kernel's mini-BZ draw mapped fractional
coordinates through the transposed reciprocal cell, which has the right volume
and the wrong shape; it passed every normalisation check for three months while
biasing the head correction on hexagonal cells by about half its own size.

None of these was a typo. Each was a copy of something that used to be right,
in a place where nothing forced it to stay right. A service is the structural
answer: the operation exists **once**, behind a package boundary that a second
copy has to visibly cross.

## What a door is

Each service package is a *door*. `import distrib_la` and use the names at the
top level; that surface is the whole supported API. Reaching past it —
`from distrib_la.plan import …`, `from symmetry_maps.maps import …` — is a
layering failure, and `tests/test_layering.py` fails on it, with a red twin
that proves the detector can still fire.

The door is not politeness. It is what makes the promises checkable. When
`distrib_la.plan()` hands back a backend name, that name means every guard
passed — platform, known-broken combinations, compiled handler, process
coverage, mesh geometry, divisibility — so the call cannot subsequently fail
for an availability reason. A caller that reached around the door to a private
resolver would get the same computation with none of the promise, and the two
would drift exactly the way the ten dispatch sites did.

The other half of the boundary points inward: a service may not import LORRAX.
`vcoul` speaks `CoulombGeometry`, an explicit k-grid and an explicit `sys_dim`;
it has never heard of a deck key or a `Meta` object. `wfn_loader` contains no
FFI target name, no context handle and no `phdf5` spelling anywhere in its
source. This is enforced by measurement rather than by intent: every service
ships an **import-isolation** suite that launches `python -S` with only that
service on the path, imports the public surface, exercises it, and asserts on
both `sys.modules` and `sys.path` that LORRAX never appeared. Each of those
suites has a red twin — the same detector run against a tree built to be
wrong — because a check that cannot fail is not a check.

## The six packages

| package | what it owns | narrative page | API reference |
|---|---|---|---|
| `lxkit` | the shared foundation: capability gates, the probe vocabulary, jax-version compatibility, the pytest harness | this page | — |
| `distrib_la` | distributed dense `eigh` / `cholesky` / `solve_lu` over a device mesh | [Distributed linear algebra](distributed_linalg.md) | [`services/distrib_la.md`](services/distrib_la.md) |
| `wfn_loader` | reading ψ(G) out of a BerkeleyGW `WFN.h5` | [Wavefunction and ζ I/O](wavefunction_io.md) | [`services/wfn_loader.md`](services/wfn_loader.md) |
| `zeta_loader` | the `zeta_q.h5` format contract and its reads | [Wavefunction and ζ I/O](wavefunction_io.md) | [`services/zeta_loader.md`](services/zeta_loader.md) |
| `symmetry_maps` | IBZ ⇄ full-BZ tables, stars, unfolds, the TRS measurement | [Symmetry](symmetry.md) | [`services/symmetry_maps.md`](services/symmetry_maps.md) |
| `vcoul` | the bare and truncated Coulomb interaction `v(q+G)` | [The Coulomb interaction](coulomb.md) | [`services/vcoul.md`](services/vcoul.md) |

The narrative pages are the front door for a reader: motivation, the shape of
the thing, and the traps. The `docs/services/` pages are the contract — dense,
exhaustive, written for somebody who is about to call the API or change it.
Read the narrative page first and the service page when you need the table.

## lxkit, and why a foundation package exists at all

Five independent services would each have grown their own answer to the same
four questions: how does an environment variable turn into a capability
decision, what does "this library is not usable" mean, which spelling of
`shard_map` does this jax have, and how does a test suite skip honestly. `lxkit`
holds the policy for all four and the tables for none of them.

The one worth understanding is the **ABSENT / BROKEN split**. When a probe says
a handler is unusable, there are two very different situations underneath: the
library was never built on this machine (nothing is wrong, these tests do not
apply, a skip is honest), or the library *is* sitting right there and will not
`dlopen` (something is wrong, and it is exactly what a test suite exists to
catch). Collapsing the two is how a whole host leg was lost on 2026-08-06:
nineteen contract cells reported "19 skipped" next to "0 failed" and the run
looked green. `lxkit.probe` makes the distinction a type, and
`lxkit.testing.absent_or_broken` is the harness arm that acts on it — ABSENT
skips, BUILT-AND-BROKEN **fails**.

`lxkit` is stdlib-only at import time. It does not import jax, and it does not
import pytest outside its harness module, so a service that wants to be
importable on a login-node interpreter can be.

## How the tests are organised

Every service suite is built in tiers, and the tier tells you what it costs to
run and what it can prove:

* **L-a — contract and shape algebra.** Pure logic: refusals fire, vocabularies
  agree, tables come out right. Runs on a laptop in milliseconds and needs
  nothing.
* **L-b — emulated multi-device.** A four-device mesh forced out of one host via
  `XLA_FLAGS`, set by the *service's own* conftest. These **skip** rather than
  assert when fewer than four devices are available, because an emulated mesh
  that quietly became a 1×1 proves nothing.
* **L-c — real multi-process.** `srun -n 4`, genuinely separate processes. The
  check bodies are shared functions called both from pytest and from a `__main__`
  CLI, so the two paths cannot drift.
* **Import isolation**, **skip honesty**, and (for `distrib_la`) an **ELF
  acceptance** tier that reads the shipped `.so` with binutils rather than
  loading it.

Two habits run through all of them and are worth adopting in anything you add.
*Every check ships with the case where it returns FALSE* — the red twin. And
*hostile geometry is mandatory*: non-dividing extents, ragged G-counts, band
counts that do not split evenly across the mesh. Several of the bugs above were
invisible for months because the default fixtures were divisible.

Suites are selected by marker. From the monorepo, `pytest -m distrib_la` (or
`wfn_loader`, `zeta_loader`, `symmetry_maps`, `vcoul`) runs one service; the
umbrella marker is `services`. Standalone, `pytest services/<name>/tests` runs
a service without ever loading the monorepo's `tests/conftest.py`. To deselect,
use `--no-services` or `--only-service=NAME` and **never a second `-m`**:
`pyproject.toml` sets `addopts = "-m 'not extra'"`, and an explicit `-m` on the
command line *replaces* that rather than adding to it, silently re-enabling the
deselected suites. The current census for every leg — pass, fail, skip, per
service, with every red accounted for — is `tests/KNOWN_FAILURES.md`.

## The one piece of plumbing you will trip over

`services/*/src` is not on any `PYTHONPATH` that a launcher sets — `lx` rewrites
the container `PYTHONPATH` to exactly `<checkout>/src`, and the Shifter image
pip-installs nothing. So a LORRAX module that imports a service must call
`ffi._services.ensure_on_path()` on a line strictly above the import.

This is transitional, it has an owner decision behind it, and the thing to know
is the failure mode: **a missing bootstrap is a green-suite, red-cluster
failure.** The service conftests put `services/*/src` on the path during
collection, so pytest passes and the real launch does not. Because sampling
cannot cover that, the coverage is structural instead —
`tests/test_service_path_bootstrap.py` walks the AST of `src/`, enumerates every
module-scope importer of every door, and asserts each one has the bootstrap
above the import. The census is complete and ordered, so a new consumer is a red
cell until it is listed. Do not paper over the failure with a `sys.path.insert`
of your own.

## Where the rest of the code still lives

The services took capabilities, not layers. LORRAX keeps the physics drivers
(`gw/`, `bse/`, `isdf/`, `bandstructure/`, `psp/`, `centroid/`), the deck parser
and configuration, the FFI layer that owns the C++ handlers and the parallel-HDF5
transport (`src/ffi/`, `src/file_io/slab_io.py`), and the translation code that
turns deck- and wavefunction-facing signatures into door calls. When you are
looking for *what LORRAX computes*, look in `src/`. When you are looking for
*how it does one well-defined operation that used to have three copies*, look in
`services/`.

For the wider map — module inventory, call hierarchy, file formats — see
[Codebase](architecture/codebase.md); for what may import what,
[Layers](architecture/layers.md); for what sits underneath a service on the
native side, [The FFI layer](architecture/ffi_layout.md).
