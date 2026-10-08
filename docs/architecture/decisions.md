# Design decisions

Dated, binding rulings from the code owner, newest first. Each entry states the
rule, the reason, and what it licenses deleting. **These override older prose
anywhere in the tree**, including every page the
[register](../index.md#register) names as an owner. Only rulings in force are
listed; a superseded ruling is removed, and git history is the archive. Every
entry is implemented on main, except the code an entry names as not yet
conforming.

The GW driver's phase invariants and the per-function contracts of
`gw.gw_config` are developer reference, not rulings:
[GW driver and configuration contracts](../dev/gw_config_contracts.md).

## 2026-10-08 — One headroom rule sets the device budget and the pool {#headroom}

**Rule (owner).** From the card total `M` alone,
`H = max(8 GB, 0.10·M)`: the XLA pool reserves `M − H` and the planner budget
is that less `max(1 GB, 0.02·M)` (`runtime.device_headroom_bytes`,
`runtime.pool_fraction`, `runtime.planner_budget_bytes`). A positive
`memory_per_device_gb` caps the budget. A start-up reading of the bytes
outside the pool may warn, never size or route
([memory model](memory-model.md#budget)).

**Why.** The frozen fraction (0.89 × 0.90 of the card), measured on one
A100-40, left 4.66 GB outside the pool on a 40 GB card and 9.36 GB on an
80 GB one, while the libraries hold the same bytes on both; the reserved pool
fills at start-up, so later communicator and library contexts get only what
is outside it. A rule of the card total alone is the same on every rank and
every run.

**Deletes.** `runtime.GPU_POOL_FRACTION`, the 0.90 × `bytes_limit`
budget, the nvidia-smi free-memory fallback and the 4 GB defaults, the BFC
fragmentation table, `runtime.aot_memory.RUNTIME_RESERVE_BYTES` and
`ISDF_CHUNK_TARGET_UTILIZATION`.

## 2026-10-08 — XLA is the reference path; a vendor route stays only where it pays {#xla-reference}

**Rule (owner).** XLA is the reference path for every operation on every
platform. A vendor kernel or library stays only when it is ≥ 2× faster on a
production stage or decisive on memory; it sits behind one service; it is
gated against the XLA path on the same device. cuSOLVERMp remains the
capacity route for matrices that do not fit one device.

* The platform is the device vendor, read from the JAX client, never the
  string `gpu`. A platform without a vendor route runs the XLA path.
* A kept vendor route is the default on its platform, and its measured gain
  sits at its Python owner ([kernel lessons](ffi_layout.md#kernel-catalog)).
  The mathdx k-convolutions stay the CUDA default because they are decisive
  on memory: one Σ τ row pass at Ni 20³ P64 holds ≤ 1 GiB of scratch against
  2.5–3.2 GB live on the staged XLA route, and runs 1.7–2.1× faster
  ([why one fused pass](kconv.md#why-fused)).

**Why.** An operation whose only engine is a vendor kernel has no engine on
any other platform: no ROCm GPU runs gwjax, and a CPU mesh without the host
library runs nothing. A vendor route with no XLA twin on its own device has
no reference to be wrong against. Several CUDA routes do not pay for
themselves: the cuBLASMp GEMM was 1.4–4.2× slower than
`distrib_la.panel_matmul` at production shapes.

**Licenses deleting** every vendor kernel, library call and environment
gate that does not meet the bar, and every refusal that exists only because
an operation had no XLA path.

**Not yet conforming.**

* The operations the [FFI layer](ffi_layout.md#kernel-operations) lists
  under *Gaps* have no plain-XLA route.

## 2026-10-05 — A per-round program's shape is decided before the first round {#fixed-round-shapes}

**Rule (owner).** Every argument of a program that runs once per round or
per SC map has a shape and pytree structure fixed for the run, decided from
quantities known before the first round: the recipe's carriers, the deck,
$N_\mu$, $n_b$ and the admitted batch width. A batch loop runs at one width
and pads its last round with a repeated real item behind a `real` count. No
optional pytree member. Where the known bound is wider than the data, the
padding is inert zero columns that the zero-row-safe eigensolver keeps out
of every spectrum. One quantity has no bound below its capacity, the
partner (TRS-odd) selection count: its extent is discovered in SC map 0,
grow-only, and held from map 1 (the cold reference: a pencil at capacity
cost +616 s per CrI3 24×24 map at P64, the map-0 growth about 250 s once).

**Why.** Seven fixes in three weeks (`held_writer_width`,
`cross_span_widths`, the face batch widths, the SC window hold, the partner
panels, the carrier growth, the ragged face tail) each remembered the maximum
seen at one site; the next site that sized a shape from its own data
recompiled again. On CrI3 24×24 at P64 the two 2026-10-05 instances cost
about 120 s per leg and 220–270 s per chain of recompiles that no production
output reported.

**Deletes.** `shared_pole_local.grow_round` and the Ritz carrier history,
the face route's own ragged schedule and the `real != len(own)` refusal;
`parent_rounds` is the one schedule; the one remaining history is the
pencil extent's (`round_tables`, in the SC session).

**Not yet conforming.** The sector writer widths (`held_writer_width`) and
the CT span widths (`cross_span_widths`) still hold a maximum from map 1,
bounded by the pole budget; the writer width is a store carrier, not a
program shape.

## 2026-10-05 — Bispinor Coulomb and Hartree use the normalized four-component wavefunctions {#four-component-carrier}

**Rule (owner).** The Coulomb interaction and the Hartree field are evaluated
with the normalized four-component wavefunctions
$\Psi=[I;X](I+X^\dagger X)^{-1/2}\psi_L$, $X=(\alpha_{\rm FS}/2)\boldsymbol\sigma\cdot\mathbf p$
(the normalized RKB lift; name and references in
[theory](../theory/bispinor-gw.md#lift)), as prior four-component GW does. The current vertices take the same carrier,
so the direct field is the four-current Hartree $V_H[\rho]+\boldsymbol\alpha\cdot\mathbf A[J]$
of one $\Psi$ ([theory](../theory/bispinor-gw.md#lift)).

**Why.** The fully relativistic pseudopotential is a Dirac–Coulomb atom with
no Breit term; it does not stand in for the two-electron relativistic terms.

**Deletes.** The large-block charge carrier $(\psi_L,0)$, its stamp, and the
direct field's separate charge spinor block.

**Not yet conforming.** The scalar Γ-head dipole (`psp.get_dipole_mtxels`)
is built on the raw lift $[\psi_L;X\psi_L]$.

## 2026-10-01 — Per-pass loops are scans over device tables {#scan-pass-loops}

**Rule (owner).** A loop over row passes or tiles runs as one `jax.lax.scan`
whose scanned operand is the per-pass table (window index, first row, last
row). Every per-pass lookup table enters the program as a device operand,
never as an HLO constant. One compiled body serves every pass, so compile
time and program size do not grow with the pass count.

* `gw.subtile_stream.plan_windows` cuts a rank's rows into equal windows
  aligned to whole symmetry orbits; `scan_passes` and `stream_passes` run
  them in one scan; `window_load` cuts the placed unfold tables on the
  device; the mathdx k-convolutions take a `live` rows operand, so the padded
  rows of a short window cost nothing.
* Users: the scalar Σ τ node and the static τ = 0 node
  (`gw.ppm_tau_kernel`), the four-current sector node
  (`gw.mpa.sector_sigma.sector_node`), the charge and four-current χ₀ streams
  (`stream_passes`), and the SUMMA panel loop of
  `distrib_la.panel_matmul`.

**Why.** An unrolled Python loop emits one body, one k-convolution object
and one copy of its tables per pass. Compile time and host memory then grow
linearly with the pass count, which reaches hundreds per rank on
production decks.

**Not yet conforming.** The parent panels of the scalar shared-pole W
synthesis (`gw.mpa.sigma._shared_pole_w_synthesis`) are a static Python loop
inside the window executable; its pole-column chunks are a `fori_loop`.

## 2026-10-01 — Dense per-q linear algebra runs local or on the full mesh, never on a sub-mesh {#no-sub-mesh}

**Rule (owner).** A dense factorization, eigensolve or product over a batch
of per-q (or per-parent) matrices has exactly two plans. **Local:** each rank
solves whole matrices of its own batch slice (for example `distrib_la`'s
batch-layout route, one batched cuSolverDn call per local stack).
**Distributed:** each matrix is distributed over all P ranks
(cuSOLVERMp, ScaLAPACK or SLATE on the world communicator), one matrix after
another. No route splits the mesh into sub-meshes or sub-communicators. The
deck dial `linalg = local | distributed` (`gw.gw_config.resolve_linalg`)
selects the plan.

**Why.** At one q and a large centroid count N_μ the distributed plan is the
only one that holds a matrix, so it must exist; a sub-mesh plan would be a
third plan with its own communicator creation (a world-collective
`MPI_Comm_split`, see the square-mesh ruling below), its own divisibility
contract and its own failure modes, and it gains nothing at that limit.

**Exception: the shared-pole parent solves.** Under `local`, a shared-pole
parent stack whose whole matrices do not fit beside the live stages takes
the distributed plan instead (`gw.shared_pole_execution.whole_parent_execution`
for the bank's line selection, `constructor_execution` for the constructor).
The fit is judged by the capacity ledger against `memory_per_device_gb`, so
this choice reads the budget, unlike the [one-memory-path](#one-memory-path)
and [fixed-tile](#fixed-tile) rulings. It is decided once, before the first
read, from shape prices only: the constructor prices the conservative recipe
pencil (`constructor_side_upper_bound`), never a measured pencil or live
memory, and keeps the route for the whole stage. It is kept because the two
plans solve the same matrices and agree to round-off, and at large N_μ the
distributed plan is the only one that holds a matrix; the budget picks
between two plans that both exist, not between layouts of one stage.

**Licenses deleting** the SLATE per-row sub-communicator context
(`distrib_la._slate._subrow_context_key`, no caller) and the batched SLATE
potrf/trsm handlers that need it (`src/ffi/cpp/slate/batched_{potrf,trsm}_ffi.cc`,
registered but never called).

## 2026-10-01 — LORRAX uses every symmetry operation QE reports {#all-qe-symmetries}

**Rule (owner).** The WFN keeps every operation QE found, including those
composed with time reversal; a deck never needs `no_t_rev` or `nosym`.
LORRAX takes the operations from the WFN header and each one's type
(unitary, or composed with time reversal) from the NSCF's
`data-file-schema.xml` beside `WFN.h5` (`symmetry_maps.qe_schema`), bound only
when the schema's operations and k rows match the WFN's. A full-zone k that
no authorized operation reaches refuses; LORRAX never adds an operation QE
did not record. Time reversal alone (k ↔ −k pairing for every operation) is
used only when QE's data says the reference is nonmagnetic: no row typed
t_rev = 1 and an SCF absolute magnetization below 1e-4 μB/cell; a 2c WFN
without that data has TRS off (`wfn.trs_holds`;
`symmetry_maps.density_symmetry_check.qe_trs_off_reason`).
Recipe: [inputs from DFT](../preprocessing.md#magnetic).

**Why.** A magnet's time-reversal-composed operations are true symmetries;
dropping them enlarges the stored k set (Fe and Ni 20³: 1062 stored k
instead of 641) and the cost of every k sum with it. `WFN.h5` does not record
which operations are composed with time reversal, and applying such an
operation as unitary maps a state onto the wrong partner, so the schema is
the one source of that bit.

**Without a schema** the run prints `SYMMETRY PROVENANCE WARNING` and treats
every header operation as unitary, which is wrong for a magnet whose QE
operations include a time-reversal-composed one.

## 2026-10-01 — A memory price over the budget warns; it never refuses {#warn-not-refuse}

**Rule (owner).** No planner stops a run because a priced or compiled memory
figure exceeds `memory_per_device_gb` or its tile. It prints one
`RuntimeWarning` per stage kind (`common.gpu_utils.warn_over_budget`:
`memory over budget at <stage>: needs X GB/rank, budget Y GB/rank, over by
Z GB; continuing (an OOM is possible)`), takes its smallest size and runs. The
shared-pole capacity ledger records an over-budget row as `FAIL` and admits it
(`gw.shared_pole_recipe.CapacityLedger.reserve`). Correctness gates still
refuse, for example `GATE shared_pole_gram_valid` (`gw.shared_pole_constructor`);
a k grid the mathdx family cannot hold takes the router's XLA backend with one
warning ([router](kconv.md#router)).

**Why.** A price is a model of the allocator and the budget is a user
setting; neither proves that a stage does not fit, so a refusal on them
stops runs the device would hold. The allocator decides exactly, at the
cost of an OOM where the device truly lacks the room.

**Deleted.** Every capacity refusal: `compiled_chunk_capacity`,
`shared_pole_round_capacity`, `zeta-mubatch-capacity`, `vq_tile_budget`,
`gn_ppm_fit_capacity`, the budget arm of `shared_pole_capacity` and the
others the planners carried.

## 2026-09-30 — Streamed loops take one fixed tile sized from their shapes {#fixed-tile}

**Rule (owner).** A loop that streams over k, q, bands, centroids, samples or
rows takes the most units whose per-rank bytes fit one fixed tile,
`runtime.tiles.TILE_BYTES` = 1 GiB (`runtime.tiles.tile_units`). The count
comes from the loop's own shapes. It never reads free device memory and
never reads `memory_per_device_gb`. Mechanics:
[memory model](memory-model.md#budget).

**Why.** Allocator state is rank-local, so a size read from free device
memory differs across ranks; the ranks then compile different loop shapes,
issue different numbers of collectives and deadlock. A size read from the
budget is the same on every rank, but it ties compiled shapes and summation
grouping, and so results at round-off, to a user setting. One GiB per rank
saturates the streaming kernels: on Fe 4³ and Na 8³ only the response sample
group lost more than 10 % per map at 256 MiB.

**A unit is counted in the layout the work runs in.** The response bank's
per-parent dense stages (sample Dyson, line selection, moment Dyson) run one
whole matrix per rank under `linalg = local`, so their unit is a layer of P
parents and a round holds at least P of them, even where one layer's bytes
exceed the tile (`gw.response_bank.parent_span`; about 6·16·n² per rank for a
streamed sample, 6.6 GB at CrI3 24×24). A narrower round leaves P − w ranks
idle (owner 2026-10-04).

**Two sizes follow the budget by design**, through the capacity ledger,
because a larger size is faster and moves no number: the shared-pole
response sample group (`gw.response_bank.response_group_size`) and the
htransform Galerkin whole-state fit (`bandstructure.fh_interp`,
`isdf.galerkin`).

**Not yet conforming.** These planners still size from
`memory_per_device_gb` (the same value on every rank, so they do not
deadlock): the ζ μ-batch planner's batch and G tile
(`gw.gflat_memory_model.plan_zeta_route_g`; its parent chunk and plane-group
width take the fixed tile),
the scalar shared-pole W-synthesis panel schedule (`gw.mpa.sigma`, parent and
pole-column capacities from the ledger), the pair-convolution chunks
(`gw.mixed_basis_pair_convolution._budget_target`), the W-av stage
(`file_io.parallel_transport._write_w_av_stage`), the non-TDA BSE column
chunk (`bse.bse_nontda.dense_col_chunk`), the head Γ GEMM route
(`gw.shared_pole_head`) and the plane-wave screening route
(`gw.plane_wave_screening`). The ruling licenses converting each to
`tile_units`.

**Deleted.** `common.gpu_utils.device_room_bytes`, the allocator read
gathered over processes.

## 2026-09-30 — W is built on the irreducible q and unfolded on the kernel's load {#w-parents}

**Rule (owner).** The screened interaction is formed the way the Green's
function is: only on the irreducible q, by the Green builder's contraction,
and unfolded to the full q grid only on the load of the k-convolution that
reads it. No W(τ), W factor or pole table exists on the full q grid.

* **Scalar shared-pole Σ.** `gw.mpa.sigma._shared_pole_w_synthesis` forms
  $W(q,\tau) = b\,d(\tau)\,b^\dagger$ and its transpose on the irreducible q
  through `gw.greens_function_kernel.build_G` (little-group realization,
  fixed-q projection); `ffi.fft.make_kfft_klead_unfold` (mathdx mode 9)
  unfolds it on the transform's load, and `make_kconv_klead_unfold`
  (mode 7) convolves it with the parent Green.
* **Four-current sector Σ.** `gw.mpa.sector_sigma.ParentW` holds
  $B_A\,d(t)\,B_B^\dagger$ and its antiunitary partner from
  `build_G_parents`; `ffi.fft.make_kconv_lorentz_unfold` (mathdx mode 8,
  target `lorrax_mathdx_kconv_klead_lorentz_wparent`) reads it on its second
  load. The constant $W_\infty - V$ and the photon static classes use the
  same entry.
* **Restart.** The stored $W_0 = V + W_c(0)$
  (`gw.mpa.sigma.shared_pole_static_wc`) holds the q parents with their
  unfold tables; BSE unfolds on load.

**Why.** W has the symmetry of the crystal, as G does. A full-q copy costs
n_q/n_q,irr times the parent bytes per τ node and a scatter to form it,
while the unfold costs nothing extra on a kernel load that already reads
the tile through an index map and phases.

**Licenses deleting** the V_R Lorentz k-convolution targets the native
library still exports for older trees (nothing in `ffi.fft` calls them).

## 2026-09-29 — Production GW is full-frequency QSGW with the shared-pole W {#production-gw-route}

**Rule (owner).** The production GW calculation is full-frequency QSGW with
W built from residues: `compute_mode = mpa`, `sigma_w_model = shared_pole`,
`qp_solver = self_consistent`. GN-PPM is a comparison route, and it refuses
metals. One band count serves χ₀ and Σ, and Σ's band sum is extrapolated by
the pooled `spectral_shell` fit. The error budget has two classes, reported
apart: the Σ quadrature ε, the ISDF basis and the W model are held to about
1 meV; the semicore read, band extrapolation, the scissored tail and W's own
band truncation are systematic and reported with their measured size. The
recipe, its budget and its pending parts are
[production QSGW](../how-to/production-qsgw.md).

**Why.** The owner's interest is the full-frequency calculation ("much more
interested in the full freq shared pole GW calc than GNPPM"). The shared-pole
W is a pole sum with real poles and positive residues by construction
([shared-pole W §4.3](../theory/shared-pole-w-model.md)), the form a Green's
function has; GN-PPM fits one pole per matrix element.

## 2026-09-29 — The current vertex is the q = 0 velocity operator {#current-vertex-q0}

**Rule (owner).** The current vertex uses the q = 0 (long-wavelength)
velocity operator by design; no k/q-dependent nonlocal current vertex (the
finite-transfer Ismail-Beigi–Chang–Louie Γ_NL(k, q)) is planned ("there will
not at any point in the future be a nonlocal k/q dependent current vertex").

**Why.** It keeps one current vertex: the q → 0 velocity $v = i[H, r]$,
which the dipole step computes and authenticates once, serves the
four-current response, its heads and the Hall current alike. A
k/q-dependent nonlocal vertex would need finite-q $V_{\rm NL}$ derivative
tables at every q and a second vertex owner, for a term no route consumes.

**Deleted.** Every to-do for that vertex, and the unused finite-transfer
code: `common.mtxel_sweep.FiniteTransferCurrentEndpoint`, the
`include_transfer_q2` jet, `psp.vnl_ops.ICLVNLTransferJet` and its wrapper,
the third-derivative (`Gppp`, l+3) radial family and the finite-q
`q_cart_bohr_inv` gate.

## 2026-09-28 — No quadrature rule is stored across runs

**Rule (owner).** Every run places its own quadrature rules; nothing is
stored on disk or across processes ("i really don't want any cached rules for
quadratures at all"). Reuse is in process only, as the
[Σ quadrature page §10](../theory/sigma-quadrature-problem.md) describes.

**Why.** It keeps every run's rules a function of that run's own requests.
A stored rule is served through a lookup whose admission tests
(containment, a node-count ceiling, a schema version) must stay correct
across code versions, and a table written by another code version or
machine can serve a rule this run would not build. A cold plan costs about
what a warm plan cost with the table (the widest Na 8³ window builds in
about 1 s, [§10](../theory/sigma-quadrature-problem.md)), and results are
bitwise to a run that read the table, so the store saved nothing worth that
risk.

**Deleted.** The Σ rule table, the run-local rule store, the minimax disk
cache; the deck key `sigma_quadrature_cache_dir` refuses by name, and
`LORRAX_MINIMAX_CACHE_DIR` and `LORRAX_DISABLE_MINIMAX_DISK_CACHE` are not
read.

## 2026-09-25 — GEMM or FFT per axis is decided by measurement, per device

**Rule (owner).** A local transform axis takes a stored-matrix GEMM instead of
the library FFT only where it was measured faster on that device: the rows of
`common.fourier_plan.GEMM_CROSSOVER`, keyed by device kind, give the axis
lengths `N` (separately for a full `N→N` axis and a supported one) at which the
GEMM wins. An unknown device, and CPU, take the FFT on every axis. The choice
is a deterministic function of the device kind, so every rank builds the same
plan, and there is no runtime autotune. Local transforms enter through
`LocalFourierPlan` (service page [`../dev/fourier_plan.md`](../dev/fourier_plan.md));
the route-G plane transform is its `in_gather` form, served by mathdx mode 10
on CUDA ([k-convolution router](kconv.md#router)).

**Why.** A GEMM axis costs `O(N·K)` per line against the FFT's `O(N log N)`,
but on a supported axis it fuses the embedding, the transform and the
restriction into one pass, and at the bounded lengths of a sphere's box that
pass beats the FFT arm's three (A100: 0.45–0.87 of the FFT arm at 16–128;
a full axis never wins there). Full-axis DFT-as-matmul remains a scaling
hazard, `O(N²)` per axis; the table keeps the GEMM to measured, bounded `N`, so
transforms keep their `O(N log N)` scaling.

## 2026-09-24 — One memory path per stage {#one-memory-path}

**Rule (owner).** A stage has one memory layout. It never branches into a
second layout or a low/high-memory mode; a stage that may not fit runs the
same layout in more passes, sized as the
[fixed-tile ruling](#fixed-tile) says. A modest cost is accepted for that.

**Why.** Every second layout is a second code path whose results must agree
with the first and which only the decks that select it ever test.

* **ψ is band-distributed.** One `ParentGreenCarrier`
  (`gw.wavefunction_bundle`) holds two packed raw-parent copies with bands on
  one mesh axis and centroids on the other (`common.wfn_layout.psi_specs`,
  `face`), `2·16·n_par·n_s·μ·N_b/P` bytes per rank, for every spinor extent
  and both centroid families. Canonical files are processor-grid independent
  and read into these faces. The deck key `low_mem_bands` refuses by name. An
  explicit dense `Gij` operand refuses (`GATE explicit_gij_unported`).
* **Band contractions gather panels per call.** A Green build is a batched
  2-D SUMMA (`distrib_la.panel_matmul`): interleaved band panels of at most
  `N_b/p` columns, bounded by one full-k Green tile
  (`gw.greens_function_kernel.green_panel_bytes`), two live, every k in one
  exchange per panel. Inside a sub-tile stream (the Σ τ and χ₀ row passes) a
  rank holds band-complete ψ rows of its own centroid blocks for one dispatch
  (`gw.subtile_stream.band_complete`), because G and χ₀ there exist only as
  streamed sub-tiles of those blocks (owner, 2026-09-30). No band-complete ψ
  copy outlives one dispatch.
* **The ζ back-solve is q-local.** Each whole-tile factor stays on its q
  owners (`16·⌈Q/P⌉·μ²` bytes per rank) and only the right-hand side moves;
  with `Q < P` the ranks past `Q` idle in the solve.

## 2026-09-18 — Headless shared-pole SC is allowed for brute-grid development

`sigma_w_model = shared_pole` with `qp_solver = self_consistent` and
`head_correction = off` runs, with a warning
(`gw_config.warn_headless_shared_pole_self_consistency`) that names the
measured map-1 q = 0 Gram risk. The dense-k grid is the intended convergence
limit: the missing head is a finite-grid term, so the headless map is judged
by its own convergence. The Gram gate is unchanged; a headless run may still
refuse at `GATE shared_pole_gram_valid`. An ordered (time-reversal-broken)
store refuses `head_correction = full` (`GATE shared_pole_head_ordered`) and
carries the direct head (`no_local_fields`) instead.

## 2026-09-06 — The GW carrier is the raw parent

* **One carrier.** `Wavefunctions` carries no full-k copies; its persistent
  GW samples live in `ParentGreenCarrier`. Two optional face fields remain
  for bounded head-star children and numerical oracles; `gw_init` does not
  populate full-k faces.
* **ζ fit.** `fit_zeta_to_h5` accepts only the typed parent plan and its two
  packed faces; the fit kernels (`z_q_from_psi_sm`, `c_q_from_psi_sm`) accept
  only typed parent faces and orbit-tile tables. Charge reuse may omit
  fit-time inputs because it skips the fit. The rectangular downfold Gram
  under `c_q_from_psi_sm` and the BSE/htransform Galerkin services are
  separate consumers and remain.
* **Σ.** Dynamic Σ and invalid-pole static tails take their operands only
  from `wavefunction_bundle.parent_sigma_operands`. GW τ and static-limit
  factories require the typed parent plan; band brackets are masks over the
  resident parents. Bracketed stage timing is unsupported.
* **Antiunitary placement.** The Green GEMM contracts raw parent faces
  (`greens_function_kernel.build_G_parents`), and the typed unfold transports
  the two-point operator to full k. An antiunitary row reads the transposed
  partner rather than a conjugate: `conj(G)` when the band weights are real,
  otherwise a second parent GEMM on the conjugated faces. Energies, masks and
  signed complex-time weights follow the parent index without conjugation;
  quadrature weights are never conjugated. Dynamic band projection returns
  raw parent rows, and the complex-linear band transpose runs once per result,
  after the complete ω accumulation. The Green tiles stay distributed over all
  P ranks.
* **Restart files** carry raw parent faces in logical centroid order.
  `file_io.restart_bundle` owns format admission and the single
  symmetry-service unfold; BSE reads its full-k selected-band ψ through it.
  A file from before the raw-parent format refuses with a regenerate message.
* **Head attribution is opt-in.** `sigma_freq_debug_output` alone enables
  head-attribution diagnostics and their output; the physical Γ completion
  does not depend on it.
* **Ordered Σ residency.** The ordered four-current Σ runs one compiled scan
  and one restore producer per endpoint-family class. The producer returns 1
  CC, 3 CT/TC or 9 TT full-q blocks with centroid axes distributed over all P
  ranks; a TT stack costs `9·nk_tot·M_T_packed²·16/P` bytes per rank. It is
  transient per class, not a cache. Classes are submitted without host
  fences, so a single-class lifetime is not implied.

## 2026-09-06 — Covariant Γ photon completion

The Γ completion obeys the full authenticated little group. It averages the
products of typed transported rank-four factors after the shared coupled head
solve and cubature. The one owner is `head_correction._photon_q0_factor_orbit`;
`StaticPhotonQ0FactorCarrier.family_plans` carries the metadata that gives the
optional diagnostic attribution the same action. No deck key and no dense
projector. One active orbit pair costs `O(group_size·4·packed_extent/√P)`
factor storage per rank; every quadratic update stays distributed on all P
ranks.

## 2026-09-05 — Covariant four-current transport on the parent route

* Four-spinors transform with the symmetry service's `diag(U₂, det(S)·U₂)`.
  Lorentz blocks use the scalar centroid transport followed by `Λ ⊗ Λ`, with
  `Λ = diag(1, polar time-odd Cartesian action)`. Vertices act after the
  child unfold.
* The TT Ward contact is subtracted on the q-IBZ before Dyson and star
  transport (`w_isdf._subtract_static_tt_contact`); on full q the transported
  contact is subtracted, never a constant unphased Γ matrix. A
  centroid-diagonal contact is rejected: real-space locality does not imply
  diagonality in the centroid representation.
* The shared static Σ consumer is `photon_sigma.contract_lorentz_blocks`;
  packed X/SX/COH and bare TT exchange call it with integer Lorentz indices.
  Sectors sum before the band unfold.
* SC density uses raw WFN IBZ rows, file k weights, and the symmetry service
  for scalar-grid and polar-current projection; `qsgw_density.rho_from_wfns`
  takes `sym` because scalar grid pullbacks cannot project a vector current.
  Only completed Hartree band matrices unfold. The transverse DFT parent
  bundle rotates with the same iteration U/E as charge.
* Typed parent transport is required, including on unreduced one-band decks;
  there is no hardware or band-count profitability score. Non-RPA screening
  and full-k restart carriers refuse on this route; neither selects a
  fallback.
* q storage follows the computed q parents (`restart_q_storage = auto | ibz`);
  a naturally unreduced WFN still stores a full q axis.

## 2026-09-05 — One in-memory centroid order (orbit-packed); files canonical

Every centroid axis a gwjax run computes on is in the orbit-packed order of
`common.grouped_layout`: whole symmetry orbits per X/Y shard, with per-shard
zero-pad suffixes. `common.centroid_basis.PackedCentroidBasis` owns it and
`meta.mu_basis` carries it; `meta.n_rmu_padded` is its packed extent. The
parent-k Green contraction, ζ-fit tiles, τ chain, V, χ₀, W and the static
kernels all compute in it, so every symmetry action is rank-local. The order
is converted only where bytes cross a file boundary (readers pack, writers
unpack), by one all-to-all round trip per sharded axis. Files keep the
canonical centroid order at logical extent, so restart files and the MPA
store are processor-grid agnostic, and BSE, htransform and downfold read the
same logical files.

Dense μ solves run at the whole packed extent with C_q's mean physical
diagonal on the pad slots (`meta.mu_solve_extent`,
`PackedCentroidBasis.solve_axis`). A "logical prefix" test on a μ axis is a
defect; use the active mask. `LORRAX_EXTRA_MU_PAD` sizes only the canonical
I/O staging carrier. Charge and current families each have their own packed
basis.

**Admission of non-orbit-closed centroid sets.** The parent plan requires
closure under its typed actions. For a non-closed charge or current set, the
driver switches to `SymMaps.trivial_view()` before building either packed
basis or the q policy: unreduced parents (`n_parent = nk`), identity k
actions and every q row, on the same parent kernels. A WARNING names the file
and recommends orbit-closed kmeans. The WFN loader keeps its authenticated
symmetry for the G-sphere unfold, file energies and file-wedge serialization;
the computational view does not change the physical symmetry verdict.

## 2026-09-04 — One owner for every mesh-padded axis; producers pad and consumers strip

`runtime.padding` owns every mesh-divisibility divisor, carrier extent, pad,
mask, authentication and strip. A producer carries a `PaddedAxis` receipt
beside its plain array; `file_io.tagged_arrays` serializes the same
logical/carrier/divisor receipt at a restart seam. A consumer never infers the
logical extent from the carrier shape and never computes `% p_x`, `% p_y`, a
mesh LCM or a round-up locally. `pad_axis` returns the named
`PadAxisResult(array, logical, padded)` with `fill` keyword-only, so neither
extent can be taken positionally and a signed sentinel cannot be passed by
accident. Contract and axis inventory: [Mesh-padded axes](padding.md).

Dynamic Σ accumulates on the one square carrier derived from both projection
specs; QP, QSGW, SC and output consumers strip by receipt (86 bands on a 4×4
mesh use an 88-band carrier with two exact-zero rows and columns). ζ band
tails, centroid families, q batches and band chunks follow the same rule. The
only remaining refusals concern unsupported process topology or a solve for
which padding is not inert. Distributed and symmetry services may
authenticate a produced carrier at their boundary but may not plan its
extent. A static census over `src/` and service source names every exception
with a disposition and rejects new local pad arithmetic.

## 2026-09-01 — COHSEX with bispinors always carries the q→0 head; `head_correction = off` is debug-only

A static COHSEX calculation with `bispinor = true` always includes the Γ-cell
head corrections; no mode may require them off or drop them silently.
`head_correction = off` exists for brute-force k-grid convergence studies and
prints `WARNING -- DEBUG` into the run record. `no_local_fields` is refused on
every bispinor deck except the shared-pole direct-head routes, because the
coupled solve has no scalar diagnostic head
(`GATE bispinor_head_correction_no_local_fields_unavailable`).

**Why.** The charge head is a leading finite-grid term that decays only as
`N_k^{-1/2}` in 2D. A screened bispinor mode that omits it is less complete
than the scalar route, not more.

`full_static_cohsex` is the one packed screened-current static mode. Under the
default `head_correction = full` its Γ completion inserts `⟨D⟩` into the bare
operator and the charge `S^{00}`/wing head into the screened operator; the
Hall term is optional. Physics: [Four-current heads and
frequency](../theory/four-current-head-corrections.md); wiring:
[Four-current wiring](four_current_wiring.md).

## 2026-08-22 — One mesh-divisibility pad helper, and its result is named

`runtime.padding.pad_axis(A, divisor, *, axis, fill=0.0)` is the only
implementation of the mesh-divisibility pad. It returns
`PadAxisResult(array, logical, padded)`, and a caller reads the extent it
wants by name. `fill` is keyword-only because the BSE ε axis pads with a
signed sentinel (`bse_window.PAD_EPS_GUARD_RY`): a positional fill could sign
the guard by accident and put pad transitions below the optical onset.

**Why named.** Two helpers once returned opposite extents from the same tuple
slot. A call site copied from the wrong one is wrong only when the extent is
not already a mesh multiple, which no mesh-divisible validation run sees. Do
not reintroduce a positional or single-value return, and do not add a second
helper.

## 2026-08-18 — The ζ band chunk is 16

The ζ fit transports ψ in band chunks of 16
(`gw_config.AUTOMATIC_BAND_CHUNK_SIZE`), mesh-rounded and capped at the
logical ζ window. There is no deck key: `band_chunk_size` refuses. The physics band window is the
same in every mode; mesh pad bands are exact zeros. The ψ(r) cache is one
rectangular, all-P band-sharded `lax.scan` result, so a 50-band window stores
64 slots. A ragged tail would split the cache/slice ABI into a second
compiled module family, so the pad stays and the memory model prices it
exactly. The Stage-C estimate keeps the HLO-calibrated concurrent
pair-density slot count (3 on GPU, 4 on CPU; `gflat_memory_model`) until a
compiled-memory query replaces it; a route-derived two-slot estimate is not
licensed.

## 2026-08-10 — `bse/bse_io.py` is split into four modules; the old name is a facade

`bse_window` owns the band window and its padding, `bse_head` the q = 0
rank-1 Coulomb head, `bse_densify` coarse-to-fine interpolation, and
`bse_loading` reading a GW restart into a BSE bundle; each states its
authority rule in its own docstring. `bse_io.py` re-exports every earlier name
as the same function object. Write new code against the owning module.
Retiring the facade is a separate decision that has not been taken.

## 2026-08-10 — The long-range channel criterion is an energy cutoff with a two-shell floor

`vq_interp` fits long-range channel *n* if and only if `(n·|b₃|)² ≤ E_eff`,
with `E_eff = max(E_cut, (2·|b₃|)²·(1 + margin))`. The one site is
`bse/vq_interp.py::lr_fit_degrees`; the superset trim, the fit and the mini-BZ
head-slot guard all read it.

* **Energy, not a shell count.** `|b₃| = 2π/c` shrinks as slab vacuum grows,
  so a fixed shell count covers a shrinking energy window while stage 1 still
  subtracts the full-sphere `V_LR`; unfitted channels are weight subtracted
  and never returned. An energy cutoff makes the channel count follow
  `1/|b₃|`.
* **The two-shell floor** (`FIT_SHELL_FLOOR = 2`). The head slot is
  `argmin_G |Q+G|`, which at a zone boundary rolls onto a neighbouring G; a
  one-shell criterion would leave the rolled channel unfitted and zero the
  head silently.
* **Default `E_CUT_FIT = 1.0` Ry.** It reproduces the MoS2 reference deck's
  channel set exactly and holds the long-range weight lost at 3× vacuum under
  1 %.

`DEG_B26P` is the in-plane degree ladder only, never a channel set.

## 2026-08-06 — There is deliberately no `LORRAX_EXTRA_BAND_PAD`

The pad-invariance test knobs `LORRAX_EXTRA_MU_PAD` (μ) and
`LORRAX_EXTRA_RANK_PAD` (the htransform rank axis) work because every consumer
of their extent reads it from one owner. A band-axis knob is licensed only in
the same form: a `runtime.padding` knob that every band-axis producer honours,
including the ψ loaders, the BSE window and the Σ carrier. A knob that reaches
only some of them reports a pad flip as green for the whole axis. That false
all-clear is worse than no check, because it stops anyone else looking.

## 2026-08-06 — `minimax` is the only screening method; `ctsp` is refused

`screening_method` accepts exactly `minimax`, its default; any other value
raises at parse time. `ctsp` used to parse and run minimax, so the deck, log
and provenance named a method the code never had. The refusal text says so.

## 2026-08-05 — The Lustre stripe count is the aggregator count, so it scales with `nranks`

ROMIO sets `cb_nodes = min(striping_factor, nranks)`, so the stripe count is
the collective-buffering aggregator count, and a fixed count caps aggregation.
With `LORRAX_PHDF5_STRIPE_COUNT` unset, both writers resolve the same policy:
`count = clamp(nranks, 4, 128)` and a striping unit ramped 1 → 4 MiB with the
rank count by exact integer comparison. The one source is
`file_io/_slab_io_ffi.py::_stripe_policy`; `ffi/cpp/phdf5/context.cc`
transcribes it. A negative count ("every OST", the maximum-contention layout)
and a non-integer refuse in both writers.

## 2026-08-05 — An allgather is a refusal, not a fallback; one transport per geometry

A route whose cost is "gather the whole global array onto one rank" cannot
complete at the design size, so it is not a tier the system may fall back to.
There is one sharded-slab transport and no caller selects a tier: no deck
key, env var or argument. On an emulated mesh (`P == 1` with more mesh cells
than processes, `common.collectives.mesh_is_emulated`) `SlabIO` constructs
`file_io._slab_io_serial._SerialBackend` from that predicate, before any
transport is built; it refuses above one process, off a CPU mesh, and outside
modes `w`/`a`/`r`. If MPI cannot bootstrap at P > 1 the run refuses. Tier
history: [`slab_io.md`](slab_io.md#tiers-history).

## 2026-08-04 — Padding is SlabIO's business, not the caller's

A caller states logical shapes only.

* `write_slab(name, A, offset=...)` accepts any `A` and pads internally if the
  backend needs it. `valid_shape` defaults to A's logical extent clipped to the
  dataset and is only the ragged-chunk override.
* `read_slab(name, shape=...)` returns exactly `shape`; a caller never sees a
  pad row.
* Bounds are tested once, on the replicated logical slab
  `offset + valid_shape`, so every rank reaches the same verdict. Testing a
  rank-local offset splits ranks between refusing and entering the collective
  and hangs the communicator.
* No rank skips a collective because of its own error: record it, take part in
  the collective teardown, then raise.
* `create_dataset` on an existing dataset reuses it when logical shape and
  dtype match and refuses otherwise, naming both shapes; it never deletes and
  recreates, and never writes into the old geometry.
* A logical slab that overhangs the dataset is clipped silently (owner
  ruling): inside SlabIO it is indistinguishable from an ordinary pad row. An
  explicit `valid_shape` that contradicts the dataset still refuses.

**Licenses deleting** caller-side pad/unpad arithmetic that exists only to
satisfy SlabIO, and per-call-site `valid_shape` that restates the logical
extent.

## 2026-08-04 — The TRS veto is about k-grid FFT sums, and only those

* **Forbidden:** using time reversal to reduce sums over k for the
  self-energy and related observables evaluated by FFT over the k-grid (χ₀, W,
  Σ). Those convolutions need the whole grid; folding them over ±q is what the
  veto exists for.
* **Allowed and preferred:** time reversal for every sum not done by k-grid
  FFT: the charge density, the IBZ self-consistent update, matrix elements and
  k-weighted accumulations. Unfolding an IBZ WFN to the full grid is an
  expansion and is allowed.

Time reversal is antiunitary with two halves applied in different places, the
`iσ_y·conj` spinor factor (`unfold_psi`) and the negation of the G list (its
caller). Applying one without the other replaces ψ(r) by ψ*(−r), which passes
every norm, orthogonality and ⟨T⟩ check while being wrong by O(100 eV) in
V_loc/V_NL. Every TRS unfold goes through the symmetry service for that
reason.

## 2026-08-01 — Square process meshes only; nonsquare P refuses

Only square 2-D device meshes are supported. A device count that is not a
perfect square refuses at mesh resolution, naming the two nearest square
counts (`common/collectives.py`). Idle-rank truncation is not implemented and
not licensed: under `JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi` communicator
creation is a world-collective `MPI_Comm_split`, `psum_replicate` and the
k-sweep gathers assume mesh size equals process count, and world barriers
would make idle ranks replay the driver in lockstep. Reachable counts are 1,
4, 9, 16, 25, 36, …; 32 GPUs (8 nodes of 4) is not one of them.

**Why square.** Rectangular meshes complicate ScaLAPACK grid geometry and the
divisibility contracts for no measured benefit at our scales.

## 2026-07-30 — D10: fixed-shape ngkmax G loading

Every per-k kernel takes the loader's fixed `(n_k, ngkmax, 3)` G table and its
mask rather than a ragged slice to each k's `ngk`. Every k then presents
identical operand shapes, the kernel lowers once, and the k sweep is dispatches
of one executable that `collectives.sweep_local_k` pipelines behind one host
readback. Padded-vs-ragged agreement is held to `RTOL_D10 = 1e-12` relative,
not bit-exactness, because appended zeros change XLA's reduction blocking.
`psp.dft_operators.generate_gvectors_k` stays only as the ragged reference
route. A code comment cannot mint an "owner decision":
cite an entry here.

## Standing rulings

- **Scaling target.** Thousands of low-memory processes: no `N_μ²`-class
  object may be required to fit on one rank in the large-P limit. Each solve
  family has two plans, a local whole-tile plan and a distributed plan;
  execution schedules of a plan are not new plans.
