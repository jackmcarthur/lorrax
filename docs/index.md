# LORRAX

**LO**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package — a JAX
multi-GPU/CPU implementation of an $O(N^3)$-scaling GW formalism, accelerated by
Interpolative Separable Density Fitting (ISDF) and real-frequency-axis integration.
The main GW driver is **GWJAX** (`gw.gw_jax`): it reads BerkeleyGW-format plane-wave
DFT wavefunctions (`WFN.h5`) and computes quasiparticle corrections using
static or frequency-dependent screening.

**The production calculation is full-frequency QSGW with the
[shared-pole W](theory/shared-pole-w-model.md), built from residues**, not
GN-PPM: [production QSGW](how-to/production-qsgw.md) gives the recipe, its
error budget and what is still pending.

- **Input**: DFT wavefunctions on a plane-wave grid, symmetry maps, and k-point sampling
- **Core idea**: replace dense charge-density products with a compact ISDF basis defined by centroids $r_\mu$
- **Outcome**: exchange and screened-exchange self-energy matrix elements and band-edge corrections

## Try it

The fastest way to confirm LORRAX works on your machine is to install it,
build the native pair, and run every driver on the bundled test fixture. The
native pair is required on every platform, at every process count
([design decisions](architecture/decisions.md)); a missing library refuses at
startup with `Could not locate liblorrax_ffi_host.so` or `… liblorrax_ffi.so`.

| platform | install and build | then |
|---|---|---|
| Perlmutter, any account | [Installation › Perlmutter](installation/perlmutter.md), steps 1–2 | its step 3 runs the suite |
| Perlmutter, project m4598 | nothing: the `lorrax_A` module supplies the pair | `lx run -N 1 -G 4 -n 4 -- python -m pytest tests/hsuite` |
| Frontera, another site | [Installation](installation/index.md) | [Quickstart](quickstart.md) |

On Perlmutter, the fixture chain alone (one GPU, output on a shared
filesystem) is

```bash
source config/perlmutter/gpu_env.sh
srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node=1 \
  .venv/bin/python -m tests.hsuite.chain --out "$SCRATCH/lorrax_quickstart_$(date +%s)"
```

See the [Quickstart](quickstart.md) for the worked example.

## High-level pipeline

LORRAX starts from a BerkeleyGW-format `WFN.h5`; producing one from a crystal is
[Inputs from DFT](preprocessing.md).

1. [Select ISDF points](theory/centroid-selection.md) from the requested band-pair feature metric.
2. Load reciprocal wavefunctions and sample their centroid faces.
3. [Fit $Z_q$ in G space and solve for $\zeta_q$](architecture/zeta_fit_mubatch.md) in centroid batches.
4. [Build $V_q$](theory/isdf-zeta-vq.md) from the interpolation vectors and Coulomb kernel.
5. Build the Green's function $G$ and (optionally) $\chi_0$ and screened interaction $W$
6. Form the requested self-energy and project to the band representation $\Sigma_{kij}$.

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
| **how to install, and which runtime is the default on which machine** | [Installation](installation/index.md) · [Perlmutter clone](installation/perlmutter.md) · [Perlmutter module](installation/perlmutter-module.md) | the runtime default per platform and the startup refusals with their fix; the clone build on Perlmutter; the module build for maintainers. |
| **how the native FFI pair is built, verified and sealed** | [Building the FFI libraries](building_ffi.md) · [FFI native libraries](installation/ffi-native-libs.md) | the per-site build, the verify contract, sealing and the ABI rule; where each native dependency comes from on a site with no module. The design is the FFI-layer row. |
| **how to run a first calculation** | [Quickstart](quickstart.md) | the bundled fixture chain, the minimal deck, the step order from `WFN.h5` to quasiparticle energies, and the next steps (QSGW, bands, BSE). |
| **how a crystal becomes a `WFN.h5`** | [Inputs from DFT](preprocessing.md) | what `WFN.h5` must contain, the QE and `pw2bgw` namelists, the patched `pw2bgw` for magnetic spinors, and where the tools are on Perlmutter. |
| **what each driver reads, computes and writes** | [Drivers](drivers.md) · [Downfold](downfold.md) | invocation, the flags and keys that change the result, outputs and refusals of each driver in chain order; the downfold input, keys and refusals. |
| **what changed between releases** | `UPGRADE_NOTES.md` (repository root) | every change that moves results, breaks a deck or invalidates an artifact. Every other page states present behaviour only. |
| **how to run the test suite and what a change must pass** | [Contributing](contributing.md) | the suite command, the static gates and the pre-push checklist. |
| **why the code does something the way it does** | [Design decisions](architecture/decisions.md) | dated, binding owner rulings. Overrides older prose *anywhere* in the tree, including this table's other rows. |
| **where a module may live, and what it may import** | [The three levels](architecture/layers.md) | L1/L2/L3 assignment, the import direction, the sanctioned exceptions, and what deliberately is *not* unified. |
| **where a source module or service package lives** | [Codebase](codebase.md) | the one-line inventory of every GW, common, centroid, file-I/O, and service source package. |
| **which capabilities are services** | [Substrate services](architecture/services.md) | the service inventory, layering boundary, and which services expose or hide backend choice. Individual caller contracts live on the service pages below. |
| **how fixed-memory Davidson and shared subspace algebra are planned** | [`planned Davidson`](services/davidson.md) | callable/data API, active storage, scratch planning, status and device contract. |
| **how reusable block orthogonalization is planned** | [`orthogonalization`](services/orthogonalization.md) | CGS2 interval API, native coefficient reductions, aliases and workspace scope. |
| **how Lanczos active storage and reorthogonalization are planned** | [`planned Lanczos`](services/lanczos.md) | Active windows, row-oriented basis, restart and sharding. |
| **how compact pseudopotential coupling is applied** | [`PSP coupling`](services/psp_compact_coupling.md) | Canonical SOC channel blocks, opt-in rollout and memory scope. |
| **how GW fixed-shape kernels avoid unused work** | [`GW kernels`](dev/gw_fixed_shape_kernels.md) | Bracket scan, active pole counts and reusable postprocessing. |
| **what centroid selection optimizes** | [Centroid selection](theory/centroid-selection.md) | Periodic quantization, positive feature Grams, global pivoting costs and the off-grid geometry contract; proposed alternatives are explicitly distinguished from production. |
| **how many centroids the exchange needs for a given Σ band range** | [ISDF exchange accuracy](theory/isdf-exchange-accuracy.md) | the measured Σ_x error of the ISDF basis against an exact plane-wave sum, as a function of N_μ, the band range B and the pair-set rank r(B); the N_μ rule for 1 meV and 0.1 meV; why selection on the wrong pair set fails at any count; the rule that the basis is selected on the band window Σ_c and χ₀ consume, with its selector and fit-window checks. |
| **how GEMM contracts active intervals in fixed distributed buffers** | [`active GEMM ranges`](dev/active_gemm_ranges.md) | interval API, descriptor views, batch support, native workspace and validation. |
| **how distributed dense linear algebra is requested** | [`distrib_la`](services/distrib_la.md) | the top-level API, plan/factor/solve and matmul contracts, backend resolution, layouts, and refusals. |
| **how crystal symmetry data is represented and applied** | [`symmetry_maps`](services/symmetry_maps.md) | the top-level API, canonical maps and actions, storage boundaries, unfold contracts, and TRS checks. |
| **how the space group acts on ψ, G, ζ, V and W, and which tables carry it** | [Symmetry](theory/symmetry.md) · [Symmetry register](architecture/symmetry_register.md) | the Seitz action, parent-k domain and trivial view, antiunitary rows and the pair-transpose rule, orbit-packed centroids (theory); the tables, their builders, shapes, kernels and refusals (register). |
| **how Coulomb kernels and q=0 cell averages are selected** | [`vcoul`](services/vcoul.md) | dimensional kernels, exact slab q=0 default, explicit debug alternatives, tensor-cell sampling, and refusals. |
| **how wavefunctions are loaded** | [`wfn_loader`](services/wfn_loader.md) | header and coefficient surfaces, eager/collective backends, raw-row metadata, and loader validation. |
| **how ζ files are read** | [`zeta_loader`](services/zeta_loader.md) | the `zeta_q.h5` format contract, the header surface, the collective slab read and the local read. |
| **how frequency quadrature rules are obtained** | [`minimax`](services/minimax.md) | the single public `minimax` namespace and its target-specific contracts: screening, MPA damped line/rectangles, Sigma denominator boxes, shared-pole value/derivative response, finite-temperature Matsubara response, GN-PPM odd-node augmentation, and the three domain-limited analytic reciprocal constructors. It owns node selection and certificates; drivers own physical geometry and units. |
| **which kernel computes an operation, on which hardware** | [Kernel operations](architecture/ffi_layout.md#kernel-operations) | every core kernel operation (the χ₀ and Σ k-convolutions, the W/V wedge transform, the ζ-fit and route-G kernels, the k-axis and spatial FFTs, the Green build and band-projection GEMMs, eigh, the Dyson solve, the active subspace, the contour accumulator, spin rotation, slab I/O): where the physics uses it, its NVIDIA and CPU engines with their limits, whether a plain-XLA route exists, its gate and its code. |
| **how LORRAX reaches a vendor library** | [The FFI layer](architecture/ffi_layout.md) | every native target (the kernel catalog), the five layers, the two build legs and their acceptance gates, which library serves each engine on each machine (**§3a is the dependency matrix**), which cuSOLVERMp selects which communication path, which FFT engine the host library binds, the C++ phdf5 defaults, the native failure modes and hard invariants, the k-convolution router (one route per platform, its modes and refusals), and the kernel-lessons register (measured speedups and closed approaches, per family). The `LocalFourierPlan` contract is its [service page](dev/fourier_plan.md). |
| **how a sharded array reaches disk** | [SlabIO](architecture/slab_io.md) | the tile contract, the caller-facing API **and what a call site may and may not assume of it**, close-time error agreement and the commit receipt, **the one-owner-per-file rule and the refusal that enforces it**, the launcher requirement, the striping and collective-I/O rules, the restart-read path, **the HDF5 operation journal**, and SlabIO's refusals. |
| **how a logical array axis becomes a mesh-legal carrier** | [Mesh-padded axes](architecture/padding.md) | the `PaddedAxis` receipt, producer pad / consumer strip rule, spec-derived divisors, restart metadata, axis-family inventory, and the complete remaining-refusal register. |
| **how much memory a stage needs** | [Memory model](architecture/memory-model.md) | the per-rank budget, the per-stage closed forms (route-G charge fit, current-channel tiles, V_q, W, Σ), invisible allocations, and the communication model. |
| **how the charge ζ fit runs** | [μ-batch fit (route G)](architecture/zeta_fit_mubatch.md) | the μ batches, the G-space pair GEMM, the planes and their k-convolution, the Z store, the finalize, and the planner's per-batch and per-rank costs. |
| **how the face-layout ζ fit moves and shards data** | [Face-ψ ζ fitting](architecture/zeta_fit_face_psi_cct.md) | the `C_q` normal equations, factor tiers, the current-channel tile loop and coupled schedule, and local/distributed solve boundaries. |
| **how exact finite-occupation response uses the two-face carrier** | [Fractional χ₀ response face](architecture/fractional_chi0_response_face.md) | the contour kernel and ordered-pair scan, carrier layouts, band-weight supports, schedule, communication, cost and refusals. |
| **how the shared-pole W is sampled, reduced, stored and consumed** | [Shared-pole model](architecture/shared_pole_model.md) | route admission, the response bank (shared-node group rules, line-support panels selected in the producer), sectors, the store schema, the Σ consumer, and the per-phase byte model. The model itself: [shared-pole W](theory/shared-pole-w-model.md). |
| **what the real-space (ISDF-free) GW path is and how to run it** | [Real-space GW](architecture/plane_wave_gw_stages.md) | the opt-in, unwired `gw/plane_wave_pipeline.py` path: the modules, the CLI and its limits, what it was validated against, and which production owner each stage (ψ(G), χ₀, W, Σ_x, Σ_c) reuses. |
| **how to run in the thousands-of-ranks regime** | [`docs/dev/large_nmu_operation.md`](dev/large_nmu_operation.md) | the LOCAL-vs-DISTRIBUTED plan table, per-stage per-rank scaling, and the fully-distributed deck. |
| **what an environment variable is called and what it defaults to** | [`docs/dev/env_vars.md`](dev/env_vars.md) | **spelling, default, class, and parse grammar — and nothing else.** Machine-enforced by `tests/test_env_registry.py`. Every row's *explanation* lives on the owner page it links to. |
| **which JAX generation may run** | [`docs/dev/jax_support.md`](dev/jax_support.md) | the single JAX/JAXLIB 0.9 contract, its package/preflight/runtime enforcement, and the Perlmutter launch pins. Historical run records do not redefine this policy. |
| **what a deck key does** | [Input reference](input_reference.md) | generated from the parser; the deck is the record for anything that changes the numbers. |
| **the equations every GW mode shares, and their cost** | [Theory map](theory/overview.md) · [Core ISDF and GW theory](theory/physics.md) | the chain common to every mode, the size symbols and scalings, and the hand-off to each owning theory page. |
| **how the imaginary-time χ₀ is integrated** | [Minimax quadrature](theory/minimax-quadrature.md) · [response rules](theory/response-laplace.md) | the gapped Laplace χ₀ behind static and PPM screening, the rules it consumes, the derived 1/d time rules on Σ's boxes, and the compact noncrossing response rule. |
| **the q→0 response tensor and the exchange head** | [S-tensor convention](theory/s-tensor-convention.md) · [LT exchange head](theory/lt-exchange-head.md) | the canonical rank-two $S_{ab}(\omega)$; the nonanalytic electron-hole exchange head and longitudinal-transverse splitting. |
| **how HL-PPM is defined** | [HL-GPP derivation](theory/hl-gpp-derivation.md) | the single-pole ansatz of `compute_mode = hl_ppm` and its two fitted parameters. |
| **what the production GW calculation is, its options and its error budget** | [Production QSGW](how-to/production-qsgw.md) | the route (full-frequency QSGW with the shared-pole W), the keys it sets, the owner's production requirements, the error budget in its two classes (controllable to about 1 meV; systematic, reported apart) with measured sizes and scope, and which parts are on main and what is open. The ruling itself: [decisions](architecture/decisions.md#production-gw-route). |
| **how to set up a GW or QSGW run on a metal** | [Metals how-to](how-to/metals.md) | the metal rules (Fermi–Dirac, shared-pole W, the one band support), the choices a request leaves open and their defaults, what the code derives, what refuses, and a worked deck. It links the theory and the SC rules; it does not restate them. |
| **how to get a WFN with every band of the plane-wave basis** | [Complete-basis WFN](how-to/complete-basis-wfn.md) | `psp.run_dense_h`: the dense H_k build and full eigh, the output WFN, the patched `jax_xc` requirement, the `[dense_h:<rule>]` preflight refusals, the per-k memory and time cost, and when to use it. |
| **how the Σ_c band sum is extrapolated past `number_bands_sigma`** | [Band extrapolation](theory/band-extrapolation.md) | where it runs (GN/HL-PPM and the scalar `mpa` stage, shared pole included), the three bracket sums from one pass and their default cuts, the pooled denominator-shell model `spectral_shell` and its fit, the no-tail rule, cost, and the measured Si errors with their scope. Deck keys: the input reference. |
| **how self-consistent GW converges and when it refuses** | [Self-consistency](self_consistency.md) | the QSGW map, band treatment, one-evaluation Anderson and its CONVERGED / STALLED / NOT UNIQUE verdicts, the Σ grid and frozen quadrature across maps, metals, seeding and outputs. |
| **the Sigma quadrature problem and its constraints** | [The Sigma(omega) quadrature problem](theory/sigma-quadrature-problem.md) | separability and per-node cost, product windows, the two error currencies, node laws, acceptance, the grouped rule set per map and its request scope, the SC freeze, and refusals. |
| **how MPA samples chi0, fits W and windows Sigma** | [Multipole frequency integration](theory/THEORY_mpa_implementation.md) | the frequency equations, validity domains, the ordered fit and window evaluation. Rule construction: the Σ quadrature row. |
| **how ordered photon sectors enter frequency Sigma** | [Sector Sigma consumer](dev/sector_sigma_consumer.md) | Manifest, separate endpoint families, common quadrature, instantaneous constant and capacity admission. |
| **how metallic (finite-occupation) screening works** | [Metallic MPA screening](theory/metallic-mpa-screening.md) | Fermi-Dirac occupations, the occupation-weight factorization and its cancellations, occupation-window supports, the two `q->0` heads and their order of limits, and the occupation-weighted Σ. Metallic self-consistency: the Self-consistency row. |
| **how a metal's q→0 head is modelled (scalar and four-current)** | [The metallic q→0 head](theory/metal-q0-head.md) | the anisotropic Fermi-surface Lindhard response of the band velocities, the Taylor-radius intraband/interband split, the Hall part of near-degenerate pairs, the Voronoi-cell average and how the head enters Σ. |
| **how the direct Hartree field is built** | [Direct Hartree field](theory/hartree.md) · [its APIs and schedules](dev/rho_vh_2d_design.md) | sources, G-space solves, zero mode, band matrix, and self-consistent rebuild. |
| **what a particular run actually resolved** | **the driver report** (`kmeans.out` for centroid selection; `gwjax.out` for GW), with the exhaustive runtime inventory behind `LORRAX_DEBUG_PRINT=1` — [annotated startup formats](environment/overview.md#startup-block) | the report records active calculation pathways; debug adds allocator/library/capability forensics. Both outrank static defaults. |
| **what the machine provides, and what breaks when a layer is missing** | [Environment overview](environment/overview.md) · [Frontera](environment/machines/frontera.md) · [Perlmutter](environment/machines/perlmutter.md) | the layered dependency tree, the shared JAX configuration, the three CUDA allocators, and the per-machine facts. |
| **why CPU collectives run on `impl=mpi`** | [Collective transports](environment/transports.md) · [`docs/dev/mpi_collectives.md`](dev/mpi_collectives.md) | the gloo corruption evidence and the MPIwrapper recipe. |
| **how to judge whether a claim or a check is any good** | [`docs/dev/QUALITY_PATTERNS.md`](dev/QUALITY_PATTERNS.md) | the ten failure classes and the assessment rubric. Cited by number (`#8`) from other pages. |
| **developer notes on one mechanism each** | [`device_put` all-gather](dev/device_put_hidden_allgather.md) · [FFI gate contract](dev/ffi_gate_contract.md) · [linalg FFI](dev/linalg_ffi.md) · [band-projection primitive](dev/staged_reshard_primitive.md) · [vendor GEMM handler](dev/vendor_gemm_service.md) | each page names its one source file and owns that mechanism's contract; none is a user page. |
| **how the four-current (bispinor) self-energy treats q→0 and frequency** | [Four-current heads and frequency](theory/four-current-head-corrections.md) | the Γ-cell head of every Lorentz channel (charge `S(ω)` and its Schur fold, the bare TT tensor head, the packed static photon head with its Hall term), the `full_shared_pole` direct bulk head, which frequency model each channel carries, and what time-reversal breaking changes. |
| **how GN-PPM is derived when the imaginary-axis matrix is non-Hermitian** | [Non-Hermitian GN-PPM derivation](dev/notes/DERIVATION_gnppm_nonhermitian.md) | the derivation memo, code correspondence, limiting identities, and test oracles for the even and odd components. |
| **what the four-current (bispinor) layer calls, and what each object's shape and sharding is** | [Four-current wiring](architecture/four_current_wiring.md) | the five routes and their admission, the stage-by-stage objects (ζ fits by channel, `V_q` tiles, packed `χ_0`, the Dyson solve, the Γ completion, Σ) with producer, shape and sharding, the envelope, and the refusals. Physics belongs to the theory row above. |
| **how the BSE is built and solved, and where its screened W comes from** | [BSE](architecture/bse.md) | the inputs and their authentication, the stored or rebuilt static W(0) and its q = 0 head, the Hamiltonian and the trial-stack matvec with its W-term kernel routes, the solvers, the dipoles and absorption, `eigenvectors.h5`, the `w_bse` handoff, the named refusals and the open limits. CLI flags: [drivers](drivers.md#bse-bsebse_jax). |
| **what bispinor (four-current) GW is, its 1/c counting, and which terms each `bispinor_gw` route keeps** | [Bispinor (four-current) GW](theory/bispinor-gw.md) | the kinetic-balance bispinor, the charge and current sources, the Coulomb-gauge propagator, the $1/c$ order of every $\Pi$ and $W$ block and $\Sigma$ term, why `bare_transverse` is the default and exactly what it omits, the route table with its code owners, the two-centroid ISDF, the static Hall term and why it is quantized, the code contracts of the Γ head (`gw.head_correction`), the Dyson solve and the Σ entry, and which route to use. |
| **when a rank/spectrum may be truncated, and what refuses if it may not** | [`docs/dev/rank_truncation_policy.md`](dev/rank_truncation_policy.md) | the one criterion (`common/rank_criterion`), the degeneracy closure (`common/spectral_closure`) and its band-axis twin (`common/band_degeneracy`); the certified κ ceiling and its measurements; **what gates nothing**; the site register with each site's certification status; and the two dials. |

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
