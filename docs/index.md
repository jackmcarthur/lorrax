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

## Try it

The fastest way to confirm LORRAX works on your machine — a complete static-COHSEX
calculation end-to-end on the bundled fixture, which ships its own wavefunction and
needs no GPU:

```bash
uv sync
bash src/ffi/cpp/build_host.sh      # see below: this step is NOT optional
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

!!! warning "There is no longer a pure-JAX path that skips the native build"
    `uv sync` alone is not enough, on any platform, at any process count.
    [Design decisions, 2026-08-01](architecture/decisions.md) made the FFI layer
    **required**: a missing or unloadable library is a refusal at startup, not a
    demotion to a JAX fallback, because the fallback paths were deleted. Verified
    2026-08-10 on a fresh clone at `88f28325`, the command above without the build
    step refuses with `RuntimeError: The required FFTW3-ABI host backend is
    unavailable … Could not locate liblorrax_ffi_host.so`. The former
    `use_ffi_io` and `slab_io` deck keys are now refused: the three I/O tiers
    collapsed to one transport, so a deck cannot select an HDF5 implementation.

    `build_host.sh` needs a SLATE `gpu_backend=none` install and refuses without
    one, naming `src/ffi/cpp/stage/slate_build_perlmutter.sh cpu` as the step
    before it. Budget for that: the 60-second promise this section used to make
    was measured against a tree that no longer exists.

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
| **what calls what, and where the code is** | [Codebase](architecture/codebase.md) | the module map, the class inventory, the call hierarchy and the file formats. |
| **which capabilities are services, and what a caller has to know** | [Substrate services](architecture/services.md) | what makes something a service here, the service inventory, **which services deliberately expose a backend choice and which deliberately hide one, and why**, and the public call signature + contract of each. Pairs with the FFI row below: that one records what is *underneath* a service, this one what a *caller* sees. |
| **how LORRAX reaches a vendor library** | [The FFI layer](architecture/ffi_layout.md) | the five layers, the two build legs, which nvhpc stage selects which communication path, which FFT engine a `.so` actually links, the C++ phdf5 context defaults and their boolean grammar, and how to tell the native-layer failure modes apart. **§3a is the dependency matrix** — one row per routine we call a vendor for, naming the library on each machine, the gate that proves it built right, and whether that gate is passing; **§3b is the list of routines with no check at all.** |
| **how a sharded array reaches disk** | [SlabIO](architecture/slab_io.md) | the tile contract, the caller-facing API **and what a call site may and may not assume of it**, the launcher requirement, the striping and collective-I/O measurements, the restart-read path, the multi-node certification, **the one-owner-per-file rule and the refusal that enforces it**, **the HDF5 operation journal**, and **the three measured failure signatures (S1/S2/S3) with their shared mechanism**. There is one transport; the three tiers and their router were deleted 2026-08-06. |
| **how much memory a stage needs** | [Memory model](architecture/memory-model.md) | the per-stage closed forms and the planner's calibration. |
| **how the face-layout ζ fit moves and shards data** | [Face-ψ ζ fitting](architecture/zeta_fit_face_psi_cct.md) | the `C_q`/`Z_q` constructions, face-Y cache, coupled transverse schedule, and local/distributed solve boundaries. |
| **how to run in the thousands-of-ranks regime** | [`docs/dev/large_nmu_operation.md`](dev/large_nmu_operation.md) | the LOCAL-vs-DISTRIBUTED plan table, per-stage per-rank scaling, and the fully-distributed deck. |
| **what an environment variable is called and what it defaults to** | [`docs/dev/env_vars.md`](dev/env_vars.md) | **spelling, default, class, and parse grammar — and nothing else.** Machine-enforced by `tests/test_env_registry.py`. Every row's *explanation* lives on the owner page it links to. |
| **which JAX generation may run** | [`docs/dev/jax_support.md`](dev/jax_support.md) | the single JAX/JAXLIB 0.9 contract, its package/preflight/runtime enforcement, and the Perlmutter launch pins. Historical run records do not redefine this policy. |
| **what a deck key does** | [Input reference](input_reference.md) | generated from the parser; the deck is the record for anything that changes the numbers. |
| **how MPA samples chi0, fits W and windows Sigma** | [Multipole frequency integration](theory/THEORY_mpa_implementation.md) | the frequency equations, validity domains, window partition, SlabIO boundaries and production-enablement boundary. |
| **how metallic (finite-occupation) screening works** | [Metallic MPA screening](theory/metallic-mpa-screening.md) | the occupation-weight factorization and its cancellation analysis, the metal frequency plan and the origin-row decision, the two `q->0` heads and their order of limits, the occupation-weighted Sigma, the per-iteration QSGW occupation state, **and metallic self-consistency — what one map call rebuilds, the one-omega-reference rule, the entry-solve rule, the `max\|dE\|` stop rule, and the measured damped-vs-undamped convergence**. |
| **how the direct Hartree field is built** | [Direct Hartree field](theory/hartree.md) | sources, G-space solves, zero mode, band matrix, and self-consistent rebuild. |
| **what a particular run actually resolved** | **the driver report** (`kmeans.out` for centroid selection; `gwjax.out` for GW), with the exhaustive runtime inventory behind `LORRAX_DEBUG_PRINT=1` — [annotated startup formats](environment/overview.md#startup-block) | the report records active calculation pathways; debug adds allocator/library/capability forensics. Both outrank static defaults. |
| **what the machine provides, and what breaks when a layer is missing** | [Environment overview](environment/overview.md) · [Frontera](environment/machines/frontera.md) · [Perlmutter](environment/machines/perlmutter.md) | the layered dependency tree, the shared JAX configuration, the three CUDA allocators, and the per-machine facts. |
| **why CPU collectives run on `impl=mpi`** | [Collective transports](environment/transports.md) · [`docs/dev/mpi_collectives.md`](dev/mpi_collectives.md) | the gloo corruption evidence and the MPIwrapper recipe. |
| **how to judge whether a claim or a check is any good** | [`docs/dev/QUALITY_PATTERNS.md`](dev/QUALITY_PATTERNS.md) | the ten failure classes and the assessment rubric. Cited by number (`#8`) from other pages. |
| **how the four-current (bispinor) self-energy treats q→0 and frequency** | [Four-current heads and frequency](theory/four-current-head-corrections.md) | the Γ-cell head of every Lorentz channel (charge `S(ω)` and its Schur fold, the bare TT tensor head, the packed static photon head with its Hall term and its declared omissions), which frequency model each channel carries, what time-reversal breaking changes, and the code owner of each object. |
| **what the four-current (bispinor) layer calls, and what each object's shape and sharding is** | [Four-current wiring](architecture/four_current_wiring.md) | the stage-by-stage map from deck key through `gw_config` resolution, `gw_init` (two centroid sets, the bispinor lift, four ζ fits, the `V_q` tiles), screening (packed `χ_0`, the distributed Dyson, the Γ completion), Σ and the output records — with every object's producer, shape, sharding and **route membership (bare / packed / both)**, the compressed refusal table, and which `gwjax.out` lines identify the route that ran. Physics belongs to the theory row above; module one-liners to the Codebase row. |
| **when a rank/spectrum may be truncated, and what refuses if it may not** | [`docs/dev/rank_truncation_policy.md`](dev/rank_truncation_policy.md) | the one criterion (`common/rank_criterion`), the degeneracy closure (`common/spectral_closure`) and its band-axis twin (`common/band_degeneracy`); the certified κ ceiling and its measurements; **what is refuted as a gate and must not be re-proposed**; the site register with each site's certification status; and the two dials. |

> **No page here can tell you what a run resolved.** Several of these knobs
> interact, and two of them (`XLA_PYTHON_CLIENT_ALLOCATOR`,
> `XLA_PYTHON_CLIENT_PREALLOCATE`) are read only *before* backend init, after
> which `os.environ` is a false witness — measured, job 7882443: two runs with
> byte-identical environments and `bytes_limit` 11.805 GB vs 0.000 GB.
> The production driver report records the active scientific choices from the
> resolved runtime.  When allocator, library or unavailable-capability detail
> matters, rerun with `LORRAX_DEBUG_PRINT=1`; that renders the exhaustive
> measured inventory — [a real one, annotated](environment/overview.md#startup-block),
> if you have not seen one before.

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
