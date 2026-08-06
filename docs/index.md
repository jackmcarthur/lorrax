# LORRAX

**LO**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package — a JAX
multi-GPU/CPU implementation of an $O(N^3)$-scaling GW formalism, accelerated by
Interpolative Separable Density Fitting (ISDF) and real-frequency-axis integration.
The main GW driver is **GWJAX** (`gw.gw_jax`): it reads BerkeleyGW-format plane-wave
DFT wavefunctions (`WFN.h5`) and computes quasiparticle corrections via static COHSEX
or GN-PPM (Godby–Needs Generalized Plasmon Pole) frequency dependence.

- **Input**: DFT wavefunctions on a plane-wave grid, symmetry maps, and k-point sampling
- **Core idea**: replace dense charge-density products with a compact ISDF basis defined by centroids $r_\mu$
- **Outcome**: exchange and screened-exchange self-energy matrix elements and band-edge corrections

## Try it in 60 seconds

The fastest way to confirm LORRAX works on your machine — runs a complete static-COHSEX
calculation end-to-end on a fresh clone with **no GPU and no native (FFI) build**
(the bundled fixture sets `use_ffi_io = false` — the deprecated spelling of
`slab_io = h5py_allgather`, the serial writer — and ships its own wavefunction):

```bash
uv sync
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

See the [Quickstart](quickstart.md) for the worked example, and
[Installation](installation/index.md) for the GPU / distributed / from-source tracks.

## High-level pipeline

LORRAX starts from a BerkeleyGW-format `WFN.h5`; producing one from a crystal is
[Inputs from DFT](preprocessing.md).

1. Charge density from selected bands → choose ISDF points $r_\mu$ via k-means/CVT
2. Read wavefunctions $c_{nk}(G)$, FFT to real space $\psi_{nk}(r)$
3. For each $q$, construct $\zeta_{q,\mu}(r)$ by solving $C_q \zeta_q = Z_q$ by least-squares
4. Compute $V_{q,\mu,\nu}$ from $\zeta_{q,\mu}$ in G-space with the Coulomb kernel $v_q(G)$
5. Build the Green's function $G$ and (optionally) $\chi_0$ and screened interaction $W$
6. Form $\Sigma_{X/SX/COH}$ and project to the band representation $\Sigma_{kij}$

## Where each fact lives {#register}

**This section is the register. One page owns each class of fact; every other
page links here rather than restating it.**

That rule exists because the alternative was measured. `use_collective_write`
and `align_threshold` were corrected in one page and stayed wrong in three
others for ten days, because four pages each carried their own copy. A
restated fact is a fact that will drift. If you are writing documentation and
find yourself explaining something the table below assigns elsewhere, write
one sentence and a link.

| If you want to know… | The owner is | It is authoritative for |
|---|---|---|
| **why the code does something the way it does** | [Design decisions](architecture/decisions.md) | dated, binding owner rulings. Overrides older prose *anywhere* in the tree, including this table's other rows. |
| **where a module may live, and what it may import** | [The three levels](architecture/layers.md) | L1/L2/L3 assignment, the import direction, the sanctioned exceptions, and what deliberately is *not* unified. |
| **what calls what, and where the code is** | [Codebase](architecture/codebase.md) · [Substrate services](architecture/services.md) | the module map and the service-facade call signatures. |
| **how LORRAX reaches a vendor library** | [The FFI layer](architecture/ffi_layout.md) | the five layers, the two build legs, which nvhpc stage selects which communication path, which FFT engine a `.so` actually links, the C++ phdf5 context defaults and their boolean grammar, and how to tell the native-layer failure modes apart. |
| **how a sharded array reaches disk** | [SlabIO](architecture/slab_io.md) | the three tiers and the routing between them, the launcher requirements, the striping and collective-I/O measurements, and the multi-node certification. |
| **how much memory a stage needs** | [Memory model](architecture/memory-model.md) | the per-stage closed forms and the planner's calibration. |
| **how to run in the thousands-of-ranks regime** | [`docs/dev/large_nmu_operation.md`](dev/large_nmu_operation.md) | the LOCAL-vs-DISTRIBUTED plan table, per-stage per-rank scaling, and the fully-distributed deck. |
| **what an environment variable is called and what it defaults to** | [`docs/dev/env_vars.md`](dev/env_vars.md) | **spelling, default, class, and parse grammar — and nothing else.** Machine-enforced by `tests/test_env_registry.py`. Every row's *explanation* lives on the owner page it links to. |
| **what a deck key does** | [Input reference](input_reference.md) | generated from the parser; the deck is the record for anything that changes the numbers. |
| **what a particular run actually resolved** | **the run's own rank-0 startup block** — [annotated, with a real one printed verbatim](environment/overview.md#startup-block) | see the warning below — it outranks every page here. |
| **what the machine provides, and what breaks when a layer is missing** | [Environment overview](environment/overview.md) · [Frontera](environment/machines/frontera.md) · [Perlmutter](environment/machines/perlmutter.md) | the layered dependency tree, the shared JAX configuration, the three CUDA allocators, and the per-machine facts. |
| **why CPU collectives run on `impl=mpi`** | [Collective transports](environment/transports.md) · [`docs/dev/mpi_collectives.md`](dev/mpi_collectives.md) | the gloo corruption evidence and the MPIwrapper recipe. |
| **how to judge whether a claim or a check is any good** | [`docs/dev/QUALITY_PATTERNS.md`](dev/QUALITY_PATTERNS.md) | the ten failure classes and the assessment rubric. Cited by number (`#8`) from other pages. |

> **No page here can tell you what a run resolved.** Several of these knobs
> interact, and two of them (`XLA_PYTHON_CLIENT_ALLOCATOR`,
> `XLA_PYTHON_CLIENT_PREALLOCATE`) are read only *before* backend init, after
> which `os.environ` is a false witness — measured, job 7882443: two runs with
> byte-identical environments and `bytes_limit` 11.805 GB vs 0.000 GB.
> `runtime.initialize_communicator_stack()` prints one rank-0 block naming
> every choice where more than one outcome was possible. Read the block —
> [a real one, annotated](environment/overview.md#startup-block), if you
> have not seen one before.

### Two things about this tree that surprise people

* **`docs/dev/` is not part of the rendered site** (`exclude_docs` in
  `mkdocs.yml`). The environment-variable registry, the quality patterns and
  the large-μ operating guide are repo-only files. They are linked above
  anyway, because on a checkout they are the pages you want.
* **Dates and pins are load-bearing.** Pages that rest on measurement carry a
  verification banner naming the machine, the commit and the date. A statement
  without one is inherited from an earlier pass, not re-measured. Line numbers
  are given so you can find code, never so you can quote it — read the file.

Contributors and coding agents should also read `AGENTS.md` in the repository
root for the module map and coding standards.
