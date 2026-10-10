# LORRAX

LORRAX (**LO**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited-state
package) computes GW quasiparticle energies and Bethe–Salpeter excitons from
plane-wave DFT wavefunctions. It compresses every pair density onto an ISDF
basis of $N_\mu$ interpolation points, so GW costs $O(N^3)$ and BSE $O(N^4)$
in system size, and it integrates χ and Σ on the real frequency axis, so no
analytic continuation is needed. It is written in JAX for many GPUs or CPUs,
reads BerkeleyGW-format `WFN.h5`, and its production calculation is
full-frequency QSGW with the [shared-pole W](theory/shared-pole-w-model.md).
To use it, go in order: [install](installation/index.md), run the
[Quickstart](quickstart.md), then set up [production QSGW](how-to/production-qsgw.md);
the [drivers](drivers.md) and the [input reference](input_reference.md) are
the reference for every step.

## Try it

Install LORRAX and build or load the native FFI pair, then run every driver
on the bundled test fixture. The native pair is required on every platform,
at every process count; a missing library refuses at startup with
`Could not locate liblorrax_ffi_host.so` or `… liblorrax_ffi.so`
([Installation](installation/index.md)).

| platform | install | then |
|---|---|---|
| Perlmutter, a clone | [Perlmutter §1](installation/perlmutter.md#clone), steps 1.1–1.2 | its step 1.3 runs the suite |
| Perlmutter, a published module | `module load` ([Perlmutter §2](installation/perlmutter.md#using-the-module)) | the suite, as below |
| Frontera, another site | [Installation](installation/index.md) | [Quickstart](quickstart.md) |

On Perlmutter, from a clone, inside an allocation `JOBID`:

```bash
source config/perlmutter/gpu_env.sh
srun --jobid=$JOBID -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \
  .venv/bin/python -m pytest tests/hsuite -q -p no:cacheprovider
```

With a module, run the same `srun` line from `$LORRAX_ROOT`, without the
`source` line (the module sets those variables) and with `python` in place of
`.venv/bin/python`. The fixture chain alone, on one GPU with its
output on a shared filesystem, is

```bash
source config/perlmutter/gpu_env.sh
srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node=1 \
  .venv/bin/python -m tests.hsuite.chain --out "$SCRATCH/lorrax_quickstart_$(date +%s)"
```

The [Quickstart](quickstart.md) walks through the same chain step by step.

## High-level pipeline

LORRAX starts from a BerkeleyGW-format `WFN.h5`; producing one from a crystal is
[Inputs from DFT](preprocessing.md).

1. [Select ISDF points](theory/centroid-selection.md) from the requested band-pair feature metric.
2. Load the plane-wave wavefunctions and evaluate them at the centroids
   $r_\mu$, stored as 2-D sharded faces $\psi_{n\mathbf k}(\mu)$ (centroids on
   one mesh axis, bands on the other).
3. [Fit the right-hand side $Z_q$ of the normal equations $C_q\zeta_q = Z_q$ in G space and solve for the interpolation vectors $\zeta_q$](architecture/zeta_fit_mubatch.md) in centroid batches.
4. [Build $V_q$](theory/isdf-zeta-vq.md) from the interpolation vectors and Coulomb kernel.
5. Build the Green's function $G$ and (optionally) $\chi_0$ and screened interaction $W$.
6. Form the requested self-energy and project to the band representation $\Sigma_{kij}$.

## Where each fact lives {#register}

**This section is the register. One page owns each class of fact; every
other page links to the owner rather than restating it.** A fact restated on
two pages drifts: when one copy is corrected, the other stays wrong. If you
find yourself explaining something the table assigns elsewhere, write one
sentence and a link. `tests/test_docs_register.py` checks that every link in
this table resolves and that every published page is in the site navigation.

### Use

| If you want to know… | The owner is | It is authoritative for |
|---|---|---|
| **how to install, and which runtime is the default on which machine** | [Installation](installation/index.md) · [Perlmutter](installation/perlmutter.md) | the runtime default per platform and the startup refusals with their fix; the clone build and the module (use and publication) on Perlmutter. |
| **which JAX generation may run** | [the JAX contract](installation/index.md#jax) | the single JAX/JAXLIB 0.9 contract and its package, preflight and runtime enforcement. |
| **how the native FFI pair is built, verified and sealed** | [Building the FFI libraries](installation/ffi-build.md) | the two legs and why they must agree, the dependencies and where each comes from, the per-site build, the verify contract, sealing, how a run selects the pair, the ABI rule and porting. The design is the FFI-layer row. |
| **how to run a first calculation** | [Quickstart](quickstart.md) | the bundled fixture chain, the minimal deck, the step order from `WFN.h5` to quasiparticle energies, and the next steps (QSGW, bands, BSE). |
| **how a crystal becomes a `WFN.h5`** | [Inputs from DFT](preprocessing.md) | what `WFN.h5` must contain, the QE and `pw2bgw` namelists, the patched `pw2bgw` for magnetic spinors, and where the tools are on Perlmutter. |
| **what the production GW calculation is, its options and its error budget** | [Production QSGW](how-to/production-qsgw.md) | the route (full-frequency QSGW with the shared-pole W), the keys it sets, the project lead's production requirements, the error budget in its two classes (controllable to about 1 meV; systematic, reported apart) with measured sizes and scope, and which parts are on main and what is open. The ruling itself: [decisions](architecture/decisions.md#production-gw-route). |
| **how to set up a GW or QSGW run on a metal** | [Metals how-to](how-to/metals.md) | the metal rules (Fermi–Dirac, shared-pole W, the one band support), the choices a request leaves open and their defaults, what the code derives, what refuses, and a worked deck. It links the theory and the SC rules; it does not restate them. |
| **how self-consistent GW converges and when it refuses** | [Self-consistency](self_consistency.md) | the QSGW map, band treatment, one-evaluation Anderson and its CONVERGED / STALLED / NOT UNIQUE verdicts, the Σ grid and frozen quadrature across maps, metals, seeding and outputs. |
| **how the BSE is built and solved, and where its screened W comes from** | [BSE](architecture/bse.md) | the inputs and their authentication, the stored or rebuilt static W(0) and its q = 0 head, the Hamiltonian and the trial-stack matvec with its W-term kernel routes, the solvers, the dipoles and absorption, `eigenvectors.h5`, the `w_bse` handoff, the named refusals and the open limits. CLI flags: [drivers](drivers.md#bse-bsebse_jax). |
| **how band structures and exciton bands are interpolated** | [Band interpolation and exciton bands](how-to/htransform-and-exciton-bands.md) | htransform's Galerkin basis and f-transform, the QP routes, `bandstructure.dat`, band character and moments (`--color`, `--moments-grid`), exciton bands $E_S(Q)$ with their exchange routes and certification, the flags and the refusals. |
| **how to get a WFN with every band of the plane-wave basis** | [Complete-basis WFN](how-to/complete-basis-wfn.md) | `psp.run_dense_h`: the dense H_k build and full eigh, the output WFN, the patched `jax_xc` requirement, the `[dense_h:<rule>]` preflight refusals, the per-k memory and time cost, and when to use it. |
| **where a dense solve or GEMM runs on P devices, and which stage takes which plan** | [Dense solves and GEMMs on P devices](architecture/dense_linear_algebra.md) | the sizes and the square mesh, the face/batch/local layouts and their exchanges, the SUMMA product (`panel_matmul`, `batch_gram`), the two eigh routes (route (c) and the capacity route) and their shape-priced decision, Newton–Schulz inverse roots, round-off across routes, the portability matrix, the per-stage local-versus-distributed table at large N_μ, and a worked CrI3 P64 example. The budget itself: the memory-model row. |
| **what each driver reads, computes and writes** | [Drivers](drivers.md) · [Downfold](downfold.md) | invocation, the flags and keys that change the result, outputs and refusals of each driver in chain order; the downfold equations, procedure, outputs and refusals. |
| **what a deck key does** | [Input reference](input_reference.md) | generated from the parser; the deck is the record for anything that changes the numbers. |
| **what each GW output file contains, and how LORRAX maps to BerkeleyGW** | [BerkeleyGW users](how-to/berkeleygw-users.md) | BerkeleyGW input keys and their deck counterparts; the format, columns, units, band and k conventions of `eqp0.dat`/`eqp1.dat`, `sigma_diag.dat`, `sigma_mnk.h5`, the restart bundle, `qp_wfn_rotations.h5`, `WFN_qp.h5`, the dipole files and `gwjax.out`; the conventions that move numbers between the codes; the measured agreement; the BSE comparison conventions. |
| **what an environment variable is called and what it defaults to** | [Environment variables](reference/env_vars.md) | **spelling, default, class and parse grammar, and nothing else.** Machine-enforced by `tests/test_env_registry.py`. Every row's explanation lives on the owner page it links to. |
| **what a particular run actually resolved** | **the driver report** (`kmeans.out` for centroid selection; `gwjax.out` for GW), with the exhaustive runtime inventory behind `LORRAX_DEBUG_PRINT=1`: [annotated startup formats](environment/overview.md#startup-block) | the report records the active calculation pathways; debug adds allocator, library and capability forensics. Both outrank static defaults. |
| **what the machine provides, and what breaks when a layer is missing** | [Environment overview](environment/overview.md) · [Frontera](environment/machines/frontera.md) · [Perlmutter](environment/machines/perlmutter.md) | the layered dependency tree, the shared JAX configuration, the GPU memory pool, and the per-machine facts. |
| **why CPU collectives run on `impl=mpi`** | [Collective transports](environment/transports.md) | the gloo corruption evidence, the jaxlib guard and clique warm-up, the thread level and the MPIwrapper recipe. |
| **what changed between releases** | `UPGRADE_NOTES.md` (repository root) | every change that moves results, breaks a deck or invalidates an artifact. Every other page states present behaviour only. |

### Theory

| If you want to know… | The owner is | It is authoritative for |
|---|---|---|
| **the equations every GW mode shares, and their cost** | [Theory map](theory/overview.md) · [Core ISDF and GW theory](theory/physics.md) | the chain common to every mode, the size symbols and scalings, and the hand-off to each owning theory page. |
| **what centroid selection optimizes** | [Centroid selection](theory/centroid-selection.md) | periodic quantization, positive feature Grams, global pivoting costs and the off-grid geometry contract; proposed alternatives are explicitly distinguished from production. |
| **how many centroids the exchange needs for a given Σ band range** | [ISDF exchange accuracy](theory/isdf-exchange-accuracy.md) | the measured Σ_x error of the ISDF basis against an exact plane-wave sum, as a function of N_μ, the band range B and the pair-set rank r(B); the N_μ rule for 1 meV and 0.1 meV; why selection on the wrong pair set fails at any count; the rule that the basis is selected on the band window Σ_c and χ₀ consume, with its selector and fit-window checks. |
| **how the space group acts on ψ, G, ζ, V and W, and which tables carry it** | [Symmetry](theory/symmetry.md) · [Symmetry register](architecture/symmetry_register.md) | the Seitz action, parent-k domain and trivial view, antiunitary rows and the pair-transpose rule, orbit-packed centroids (theory); the tables, their builders, shapes, kernels and refusals (register). |
| **how the direct Hartree field is built** | [Direct Hartree field](theory/hartree.md) · [its APIs and schedules](dev/rho_vh_2d_design.md) | sources, G-space solves, zero mode, band matrix, and self-consistent rebuild. |
| **how the imaginary-time χ₀ is integrated** | [Minimax quadrature](theory/minimax-quadrature.md) · [response rules](theory/response-laplace.md) | the gapped Laplace χ₀ behind static and PPM screening, the rules it consumes, the derived 1/d time rules on Σ's boxes, and the compact noncrossing response rule. |
| **the q→0 response tensor and the exchange head** | [S-tensor convention](theory/s-tensor-convention.md) · [LT exchange head](theory/lt-exchange-head.md) | the canonical rank-two $S_{ab}(\omega)$; the nonanalytic electron-hole exchange head and longitudinal-transverse splitting. |
| **what the velocity operator is, and how parallel-transport links build the QSGW velocity** | [The velocity operator](theory/qp-velocity.md) | $v = \partial_k H$ with the nonlocal and DFT+U terms and their sign; the QSGW term $D_k\Delta H$; links, the covariant derivative and the outer band set; the Marzari–Vanderbilt stencil and collapsed axes; the link error and its per-map bound; the heads (`off`, `dft_velocity`, `parallel_transport`, `interband_commutator`); the four-current case; the `parallel_transport.h5` schema. |
| **how HL-PPM is defined** | [HL-GPP derivation](theory/hl-gpp-derivation.md) | the single-pole ansatz of `compute_mode = hl_ppm` and its two fitted parameters. |
| **how GN-PPM is derived when the imaginary-axis matrix is non-Hermitian** | [Non-Hermitian GN-PPM](theory/gn-ppm-nonhermitian.md) | the two-residue model on a magnet, which residue each Σ branch consumes, the crossing closure's premise, and the ordered multipole fit. |
| **how MPA samples chi0, fits W and windows Sigma** | [Multipole frequency integration](theory/THEORY_mpa_implementation.md) | the frequency equations, validity domains, the ordered fit and window evaluation. Rule construction: the Σ quadrature row. |
| **what the shared-pole W is and how many poles it needs** | [Shared-pole screened interaction](theory/shared-pole-w-model.md) | the model; its implementation is the shared-pole row under Architecture. |
| **the Sigma quadrature problem and its constraints** | [The Sigma(omega) quadrature problem](theory/sigma-quadrature-problem.md) | separability and per-node cost, product windows, the two error currencies, node laws, acceptance, the grouped rule set per map and its request scope, the SC freeze, and refusals. |
| **how the Σ_c band sum is extrapolated past `number_bands_sigma`** | [Band extrapolation](theory/band-extrapolation.md) | where it runs (GN/HL-PPM, the scalar `mpa` stage and both bispinor shared-pole routes), the three bracket sums from one pass and their default cuts, the pooled denominator-shell model `spectral_shell` and its fit, the no-tail rule, cost, and the measured Si errors with their scope. Deck keys: the input reference. |
| **how metallic (finite-occupation) screening works** | [Metallic MPA screening](theory/metallic-mpa-screening.md) | Fermi-Dirac occupations, the occupation-weight factorization and its cancellations, occupation-window supports, the two `q->0` heads and their order of limits, and the occupation-weighted Σ. Metallic self-consistency: the Self-consistency row. |
| **how a metal's q→0 head is modelled (scalar and four-current)** | [The metallic q→0 head](theory/metal-q0-head.md) | the anisotropic Fermi-surface Lindhard response of the band velocities, the Taylor-radius intraband/interband split, the Hall part of near-degenerate pairs, the Voronoi-cell average and how the head enters Σ. |
| **what bispinor (four-current) GW is, its 1/c counting, and which terms each `bispinor_gw` route keeps** | [Bispinor (four-current) GW](theory/bispinor-gw.md) | the carrier of every vertex (the normalized RKB lift) and its one raw-lift exception, the charge and current sources, the Coulomb-gauge propagator, the $1/c$ order of every $\Pi$ and $W$ block and $\Sigma$ term, why `bare_transverse` is the default and exactly what it omits, the route table with its code owners, the two-centroid ISDF, the static Hall term and why it is quantized, the code contracts of the Γ head (`gw.head_correction`), the Dyson solve and the Σ entry, and which route to use. |
| **how the four-current (bispinor) self-energy treats q→0 and frequency** | [Four-current heads and frequency](theory/four-current-head-corrections.md) | the Γ-cell head of every Lorentz channel (charge `S(ω)` and its Schur fold, the bare TT tensor head, the packed static photon head with its Hall term), the `full_shared_pole` direct bulk head, which frequency model each channel carries, and what time-reversal breaking changes. |
| **how the orbital magnetic moment is computed** | `src/psp/orbital_magnetization_THEORY.md` (in the source tree) | the modern-theory sum-over-states moment from the analytic $dH/dk$, its units and sign against the spin moment, the chemical potential and Chern term, metallic occupations, and the QSGW velocity it takes. htransform's `--color orbital` and `--velocity` totals use it ([how-to](how-to/htransform-and-exciton-bands.md)). |

### Architecture

| If you want to know… | The owner is | It is authoritative for |
|---|---|---|
| **why the code does something the way it does** | [Design decisions](architecture/decisions.md) | dated, binding rulings of the project lead. Overrides older prose *anywhere* in the tree, including this table's other rows. |
| **where a module may live, and what it may import** | [The three levels](architecture/layers.md) | L1/L2/L3 assignment, the import direction, the sanctioned exceptions, and what deliberately is *not* unified. |
| **where a source module or service package lives** | [Codebase](codebase.md) | every `src/` package and service: what it owns, its entry points, its level and its importers, and one line per module. |
| **how LORRAX reaches a vendor library** | [The FFI layer](architecture/ffi_layout.md) | every native target (the kernel catalog), the five layers, the one C++ tree and its two legs, which library serves each engine on each machine (**§3a is the dependency matrix**), which cuSOLVERMp selects which communication path, which FFT engine the host library binds, the C++ phdf5 defaults, the native failure modes and hard invariants, and the Local Fourier plan's CUDA leg. The `LocalFourierPlan` contract is its [service page](dev/fourier_plan.md). |
| **which kernel computes an operation, on which hardware** | [Kernel operations](architecture/ffi_layout.md#kernel-operations) | every core kernel operation (the χ₀ and Σ k-convolutions, the W/V wedge transform, the ζ-fit and route-G kernels, the k-axis and spatial FFTs, the Green build and band-projection GEMMs, eigh, the Dyson solve, the active subspace, the contour accumulator, spin rotation, slab I/O): where the physics uses it, its NVIDIA and CPU engines with their limits, whether a plain-XLA route exists, its gate and its code. |
| **how a k-axis convolution or transform is computed** | [k-convolution](architecture/kconv.md) | the pair-convolution operation and why it runs as one fused shared-memory pass, the router (one route per platform), the factories and the twelve modes, the k-box stage, unfold on load, row passes and the `live` operand, scratch and the tile-table load, cost, refusals, what decides each kernel's NVRTC image, the numerical contract, and how to add a mode. |
| **what is compiled, when, and at what cost** | [Compilation](architecture/compilation.md) | the three kinds of compiled code (native kernels, NVRTC images, XLA programs) and the key of each; the NVRTC build service, the cubin cache and the release store; the path of a jitted call to its executable; the persistent compile cache, what makes it cold, its behaviour across ranks and nodes, its pruning, the programs it never stores, the cross-rank compile agreement and the runtime's XLA flags; the measured compile cost on the hsuite and on production decks; what compile threads can and cannot do; the open and rejected levers; and the compile log lines, refusals and notices. What decides a k-convolution image is the k-convolution row; the variables are the environment-variable row. |
| **how a sharded array reaches disk** | [SlabIO](architecture/slab_io.md) | the tile contract, the caller-facing API **and what a call site may and may not assume of it**, close-time error agreement and the commit receipt, **the one-owner-per-file rule and the refusal that enforces it**, the launcher requirement, the striping and collective-I/O rules, the restart-read path, **the per-rank streamed tier** (its files, capacity probe, lifetime and cleanup), **the HDF5 operation journal**, and SlabIO's refusals. |
| **how a logical array axis becomes a mesh-legal carrier** | [Mesh-padded axes](architecture/padding.md) | the `PaddedAxis` receipt, producer pad / consumer strip rule, spec-derived divisors, restart metadata, axis-family inventory, and the complete remaining-refusal register. |
| **how much memory a stage needs** | [Memory model](architecture/memory-model.md) | the per-rank budget, the per-stage closed forms (route-G charge fit, current-channel tiles, V_q, W, Σ), invisible allocations, and the communication model. |
| **when a rank/spectrum may be truncated, and what refuses if it may not** | [Rank truncation](architecture/rank_truncation_policy.md) | the one criterion (`common/rank_criterion`), the degeneracy closure (`common/spectral_closure`) and its band-axis twin (`common/band_degeneracy`); the certified κ ceiling and its measurements; **what gates nothing**; the site register with each site's certification status; and the two dials. |
| **how the charge ζ fit runs** | [μ-batch fit (route G)](architecture/zeta_fit_mubatch.md) | the μ batches, the G-space pair GEMM, the planes and their k-convolution, the Z store, the finalize, and the planner's per-batch and per-rank costs. |
| **how the face-layout ζ fit moves and shards data** | [Face-ψ ζ fitting](architecture/zeta_fit_face_psi_cct.md) | the `C_q` normal equations, factor tiers, the current-channel tile loop and coupled schedule, and local/distributed solve boundaries. |
| **how exact finite-occupation response uses the two-face carrier** | [Fractional χ₀ response face](architecture/fractional_chi0_response_face.md) | the contour kernel and ordered-pair scan, carrier layouts, band-weight supports, schedule, communication, cost and refusals. |
| **how the shared-pole W is sampled, reduced, stored and consumed** | [Shared-pole model](architecture/shared_pole_model.md) | route admission, the response bank (shared-node group rules, line-support panels selected in the producer), the scalar local and face rounds, the store schema, the scalar Σ consumer, and the per-phase byte model. The model itself: [shared-pole W](theory/shared-pole-w-model.md). |
| **how the bispinor (four-current) shared-pole W is built and consumed** | [Shared-pole W for bispinor sectors](architecture/bispinor_shared_pole_w.md) | the CC/TT/CT sector model and its treatment ceiling, the sector carrier, the photon bank, the per-map route (q-local or staged rounds of min(n_q, P) parents), the staged construction (GEMM stages, eigh stacks on route (c) or the mesh), the sector store and residence, the sector Σ consumer (three windows, W(τ) on the irreducible q, the constant, the Γ head, factor placement), and a worked CrI3 P64 byte budget. |
| **what the four-current (bispinor) layer calls, and what each object's shape and sharding is** | [Four-current wiring](architecture/four_current_wiring.md) | the five routes and their admission, the stage-by-stage objects (ζ fits by channel, `V_q` tiles, packed `χ_0`, the Dyson solve, the Γ completion, Σ) with producer, shape and sharding, the envelope, and the refusals. Physics belongs to the theory rows. |
| **how fixed-memory Davidson is planned** | [Planned Davidson](architecture/iterative_eigensolvers.md#planned-davidson) | the callable and data API, fixed-capacity storage, scratch planning, status and the device contract. |
| **how Lanczos active storage and reorthogonalization are planned** | [Planned Lanczos](architecture/iterative_eigensolvers.md#planned-lanczos) | active windows, the row-oriented basis, restart and sharding. |
| **how compact pseudopotential coupling is applied** | [The NSCF Hamiltonian](architecture/nscf_hamiltonian.md) | how `psp.run_nscf` applies $H_k$: the compact nonlocal coupling with its canonical SOC channel blocks, the batched application and the per-k schedule. |
| **what the real-space (ISDF-free) GW path is and how to run it** | [Real-space GW](architecture/plane_wave_gw_stages.md) | the opt-in, unwired `gw/plane_wave_pipeline.py` path: the modules, the CLI and its limits, what it was validated against, and which production owner each stage (ψ(G), χ₀, W, Σ_x, Σ_c) reuses. |

### Services

| If you want to know… | The owner is | It is authoritative for |
|---|---|---|
| **which capabilities are services** | [Substrate services](architecture/services.md) | the service inventory, layering boundary, and which services expose or hide backend choice. Individual caller contracts live on the service pages below. |
| **how distributed dense linear algebra is requested** | [`distrib_la` API](services/distrib_la/api.md) · [backends](services/distrib_la/backends.md) | the layouts, plan/factor/solve and matmul contracts, panel and face products, the polar factor, workspace queries and refusals (API); which library serves each operation, the `linalg` dial, the guard ladder and how to add a backend (backends). |
| **how GEMM contracts active intervals in fixed distributed buffers** | [Active ranges](services/distrib_la/api.md#active-ranges) | the interval API, descriptor views, batch support, native workspace and validation. |
| **how shared subspace algebra is planned** | [Active subspace](services/distrib_la/subspace.md) | the fixed-capacity buffers, the store / projected eigh / project / reconstruct / Gram operations, memory and synchronization. |
| **how reusable block orthogonalization is planned** | [CGS2 orthogonalization](services/distrib_la/subspace.md#cgs2) | the CGS2 interval API, native coefficient reductions, aliases and workspace scope. |
| **how crystal symmetry data is represented and applied** | [`symmetry_maps`](services/symmetry_maps.md) | the top-level API, canonical maps and actions, storage boundaries, unfold contracts, and TRS checks. |
| **how Coulomb kernels and q=0 cell averages are selected** | [`vcoul`](services/vcoul.md) | dimensional kernels, exact slab q=0 default, explicit debug alternatives, tensor-cell sampling, and refusals. |
| **how wavefunctions are loaded** | [`wfn_loader`](services/wfn_loader.md) | header and coefficient surfaces, eager/collective backends, raw-row metadata, and loader validation. |
| **how ζ files are read** | [`zeta_loader`](services/zeta_loader.md) | the `zeta_q.h5` format contract, the header surface, the collective slab read and the local read. |
| **how frequency quadrature rules are obtained** | [`minimax`](services/minimax.md) | the single public `minimax` namespace and its target-specific contracts: screening, MPA damped line/rectangles, Sigma denominator boxes, shared-pole value/derivative response, finite-temperature Matsubara response, GN-PPM odd-node augmentation, and the three domain-limited analytic reciprocal constructors. It owns node selection and certificates; drivers own physical geometry and units. |

### Developer

| If you want to know… | The owner is | It is authoritative for |
|---|---|---|
| **how to run the test suite and what a change must pass** | [Contributing](contributing.md) | the suite command, the static gates and the pre-push checklist. |
| **what each test file checks, and how a test process seals the source closure** | `tests/README.md` (repository) | the `conftest.py` source-closure seal, the static AST suites, the hsuite fixture and one line per focused CPU test. |
| **how to judge whether a claim or a check is any good** | [`docs/dev/QUALITY_PATTERNS.md`](dev/QUALITY_PATTERNS.md) | the ten failure classes and the assessment rubric. Cited by number (`#8`) from other pages. |
| **what the GW driver and `gw.gw_config` guarantee to their callers** | [GW driver and configuration contracts](dev/gw_config_contracts.md) | the driver invariants of `gw.gw_jax` and the parsing, self-energy, screening, layout, band-count, four-current and head contracts of `gw.gw_config`. Deck keys and their defaults are the input reference. |
| **how GW fixed-shape kernels avoid unused work** | [`GW kernels`](dev/gw_fixed_shape_kernels.md) | bracket scan, active pole counts and reusable postprocessing. |
| **developer notes on one mechanism each** | [`device_put` all-gather](dev/device_put_hidden_allgather.md) · [FFI gate contract](dev/ffi_gate_contract.md) · [band-projection primitive](dev/staged_reshard_primitive.md) · [vendor GEMM handler](dev/vendor_gemm_service.md) | each page names its one source file and owns that mechanism's contract; none is a user page. |

`docs/dev/` holds the developer notes linked above. It is not part of the
rendered site (`exclude_docs` in `mkdocs.yml`), so read those pages in a
checkout. Line numbers in any page help you find code; read the file rather
than quoting them. Contributors and coding agents should also read
`AGENTS.md` in the repository root for the module map and coding standards.

> **No page here can tell you what a run resolved.** Several knobs interact,
> and two of them (`XLA_PYTHON_CLIENT_ALLOCATOR`,
> `XLA_PYTHON_CLIENT_PREALLOCATE`) are read only before the backend starts,
> after which `os.environ` is a false witness: two runs with byte-identical
> environments have reported `bytes_limit` 11.805 GB and 0.000 GB. The
> driver report records the active scientific choices from the resolved
> runtime; when allocator, library or capability detail matters, rerun with
> `LORRAX_DEBUG_PRINT=1`, which renders the full measured inventory
> ([an annotated example](environment/overview.md#startup-block)).
