# Design decisions

Dated, binding rulings from the code owner, newest first. Each entry states the
rule, the reason, and what it licenses deleting. **These override older prose
anywhere in the tree**, including every page the
[register](../index.md#register) names as an owner. Only rulings in force are
listed; a superseded ruling is removed, and git history is the archive. Every
entry is implemented on main.

The last two sections are agent-tier: the GW driver's binding invariants and
the per-function contracts of `gw.gw_config`, whose one-line docstrings point
here.

## 2026-09-24 — NVIDIA k-convolutions run on nvidia-mathdx, behind one platform router

**Rule.** Physics code calls one backend-agnostic entry per k-axis operation:
the ζ-fit pair and parent-pair convolutions (behind the unchanged
`parent_projector_kconv` seam), the k-leading T·W convolution of Σ and COHSEX,
the k-minor convolution of the BSE ladder rung, and the k-leading and k-minor
transforms (the flat-k transform included). Drivers never name a backend, a
kernel or an environment variable. The router lives in the `ffi/fft.py`
facade (layer 2 of [`ffi_layout.md`](ffi_layout.md) §1) and chooses by
platform only:

* CUDA → one fused nvidia-mathdx (cuFFTDx) kernel family, compiled by NVRTC
  per (mode, grid, n_s, arch), cached in process and on disk
  (content-hashed, atomic, re-verified on load). It is the only NVIDIA route.
* cpu → the FFTW3-ABI plan handlers of the host library.
* Any other platform → a named refusal.

No environment variable selects a route (QUALITY #8). A k-grid axis above
`KCONV_AXIS_MAX = 40` (the fp64 cuFFTDx thread-FFT limit) or a k-row that does
not fit shared memory refuses by name. A missing `nvidia-mathdx` wheel on CUDA
refuses at startup naming `pip install nvidia-mathdx`; the wheel is pinned in
the `cuda12`/`cuda13` extras and the Perlmutter runtime recipe, and both
Perlmutter FFI legs link the one MPI pinned in `config/perlmutter/ffi_mpi.sh`.

**Why.** Library thread FFTs inside one fused shared-memory pass replace a
multi-pass transform/transpose/product chain, so each k-convolution is one
kernel launch; it measured 1.7× (8³) and 4.4× (12×12×1) over the
direct DFT on A100 (CLAIMS 2651). The wheel ships the headers, so making it
the only NVIDIA route costs no build complexity.

**Consequence for BSE.** Every TDA solve uses the stack matvec. The scalar
singlet exchange weight (kernel D + 2V − W for `nspinor = 1`) has one owner,
`bse_preconditioner.exchange_spin_weight`, applied at every encode.

**Licenses deleting** every other k-convolution engine, its env gate and its
route selection, on both platforms.

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

## 2026-09-06 — Both `low_mem_bands` values select parent ψ shardings

`true` selects the `face` layout and `false` the `axis` layout, for every
spinor extent and both centroid families. One `ParentGreenCarrier` holds two
packed raw-parent copies; its static layout chooses their band sharding and
the `distrib_la` matmul plan. The Green builder and the two-GEMM band
projector are shared. Axis band contractions have no collectives; projecting
the tiled centroid operator keeps the X/Y centroid reduce-scatters and the
final psums. Canonical files are processor-grid independent and read into
either layout. The carrier stays resident through screening and Σ, which
consume the same wavefunction bundle. An explicit dense `Gij` operand refuses
under either layout (`GATE low_mem_bands_explicit_gij_unported`).

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
helper. `tests/test_pad_parity_gates.py` pins both
(`test_pad_axis_fill_is_keyword_only_and_signed`, and a source gate against a
second helper).

## 2026-08-18 — The ζ band chunk is 16

The ζ fit transports ψ in band chunks of 16
(`gw_config.AUTOMATIC_BAND_CHUNK_SIZE`), mesh-rounded and capped at the
logical ζ window. There is no deck key: `band_chunk_size` refuses and names
`low_mem_bands`, which selects the carrier. The physics band window is the
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

## 2026-08-01 — FFI backends are required, not optional

The FFI layer (FFT, GEMM, distributed linear algebra, parallel HDF5) is part of
the build. A missing or unloadable FFI library refuses at startup, naming the
library; it never demotes to a slower path. Where a certified FFI path exists,
no duplicate JAX compute path is kept. The required libraries must build and
load at P = 1.

**Licenses deleting** the XLA duplicates of certified FFI paths and every
"if FFI enabled … else jnp" fork. It does not license deleting the
vendor-portability fallbacks inside the handlers (plain-loop CBLAS, FFTW-vs-MKL
resolution); those keep the required layer buildable everywhere.

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
readback. Padded-vs-ragged agreement is gated at `RTOL_D10 = 1e-12` relative,
not bit-exactness, because appended zeros change XLA's reduction blocking
(`tests/test_kin_ion_padded_gvectors.py`, `tests/test_psp_padded_gvectors.py`).
`psp.dft_operators.generate_gvectors_k` stays only as the reference route
those gates compare against. A code comment cannot mint an "owner decision":
cite an entry here.

## Standing rulings

- **Scaling target.** Thousands of low-memory processes: no `N_μ²`-class
  object may be required to fit on one rank in the large-P limit. Each solve
  family has two plans, a local whole-tile plan and a distributed plan;
  execution schedules of a plan are not new plans.
- **No DFT-as-matmul.** Transforms go through the FFT services.
- **Evidence.** Every performance claim carries a CLAIMS row with its job id
  and on-disk artifact.

## GW driver invariants (`gw.gw_jax`)

Binding rules of the driver's phases that no docstring carries.

* **One timing table that sums to the wall.** `timing.reset()` runs at `main()`
  entry, and `timing.report(wall=...)` closes the table, so printed rows plus
  `(untimed)` equal the process wall. The pre-`main()` span
  (`initialize_communicator_stack`, imports) is decomposed from
  `RUNTIME.facts['elapsed']`, not added as extra rows.
- **Refuse before compute.** The `compute_mode × qp_solver` axes and every
  cross-key envelope are resolved and validated at parse time, before the WFN
  read and the ζ fit.
- **The mesh is built once**, above `main()`, with every communicator warmed
  (`nccl_warmup` on GPU, `warm_mesh_cliques` on CPU/MPI). Never call
  `prepare_mesh()` again: a second `Mesh` is a second set of communicators and
  jit caches.
- **HDF5 library instances** are measured from `/proc/self/maps`
  (`file_io.hdf5_owner`), and probed again after each SC store cycle; an
  unsafe inventory (two libhdf5 instances touching one file) always prints.
- **Route and head provenance in the run record.** Which bispinor route ran
  (`packed_bare_transverse_route` returns the first unmet condition), the Γ
  head status, and the resolved ζ-fit edge
  (`gw_init.resolve_zeta_fit_edge`, never the deck value) are printed into
  `gwjax.out`, not left to component chatter.
- **One head resolver.** Every q→0 head sample of a run (COHSEX static head,
  W0 restart head, PPM dynamic head) comes from `head_correction.HeadResolver`.
- **SC iteration 1 is the one-shot.** SC runs skip the one-shot Σ; the first
  map reproduces it (`tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot`).
- **Σ_x gate.** Every Σ_x diagonal entry must be negative
  (`sanity.check_sign`); a positive one is a sign, conjugation or band-index
  slip.
- **Non-finite refusals.** Σ and `kin_ion` are checked before the QP eigh, which
  both one-shot and SC cross; a non-finite value refuses
  (`common.sanity`). `LORRAX_ALLOW_NONFINITE_RESULT=1` is the forensic escape.
- **No gathers for printing.** Rank-0 writers consume bounded `(nk, nb)`
  diagonals extracted collectively; a band-sharded `(nk, nb, nb)` operator is
  never host-converted.
- **Degenerate-set averaging once**, at the H-build seam (BerkeleyGW
  `shiftenergy.f90` convention, off with `no_degen_averaging`). Head-only debug
  columns take the same averaging; `eqp_g0w0` is formed after it.
- **`write_eqp2` never recomputes GW.** It iterates only the built
  full-matrix Σ(ω), rotated into each updated QP basis.

## Configuration contracts (`gw.gw_config`)

Contracts of the functions and classes whose docstrings point here. The deck
keys themselves are in the [input reference](../input_reference.md).

### Parsing

- **`read_lorrax_input`** parses the `[cohsex]` section and strips the QE
  `K_POINTS` block. Parsing is always strict: every unknown key refuses in one
  aggregated error with line numbers (`_deck_key_line`); a retired key gets its
  own report naming the replacement. Inline `#` comments are stripped. Key
  names are case-folded on both sides of the unknown-key check. It records
  which keys the deck named, so an explicit default and an absent key are
  distinguishable where that matters and serialize to the same
  `LorraxConfig` otherwise (`raw_input_keys`). `_print_deck_report` prints the
  hygiene report on rank 0 and stays importable without jax.
- **`LorraxConfig.from_input_file`** resolves the typed record once.
  `runtime_platform` injects `cpu`/`gpu` for a preflight with no device;
  `resolve_hardware=False` leaves an auto memory budget at its zero sentinel
  and makes no device probe. Production callers use the defaults. Only
  `restart = true` enters the restart loader; a file in `tmp/` is not
  permission to reuse it.
- **`env_float`**: unset or blank → default; unparseable → announced, or with
  `refuse=True` (for knobs that gate correctness) a refusal naming the
  variable. Never a silent default.
- **`active_zeta_truncating_knobs`** lists the ζ-fit truncating env knobs in
  force, so the run record says the fit was truncated.
- **Normalizers** (`coerce_compute_mode`, `coerce_screening_diagrams`,
  `_normalize_placement`) accept an enum, its `.value` or a string, in one
  place; a typo raises naming the legal set and never resolves to a default.

### Self-energy and solver axes

- **`ComputeMode`** is the single axis naming the ansatz for W's frequency
  dependence: `x_only`, `cohsex`, `gn_ppm`, `hl_ppm`, `mpa`. The value names
  the ansatz, not the numerics. `full_freq` was rejected. It names a family
  (contour deformation, real-axis quadrature and MPA are all full-frequency),
  so a deck would still need a second axis to pick one. It also reserves the
  name for a future pole-free numerical Σ.
  `is_dynamic` means "this run has an ω axis" (GN/HL-PPM and MPA);
  `ppm_model` is `'gn'`/`'hl'` and None for MPA and the static modes, so a site
  that means "which two-point PPM fit" asks `ppm_model`, never `is_dynamic`.
- **`LorraxConfig.compute_mode`**: `auto` (default) infers from the legacy
  `do_screened`/`use_ppm_sigma`/`ppm_model`; an explicit value overrides.
  Resolving is not permitting: `refuse_unimplemented_compute_mode` runs at
  driver entry and raises `NotImplementedError` (distinct from a typo's
  `ValueError`); `UNIMPLEMENTED_MODES` is empty today.
- **`announce_legacy_sigma_axis_keys`** prints one deprecation note per legacy
  key in `LEGACY_SIGMA_AXIS_KEYS` the deck named and returns them; nothing is
  refused and nothing resolves differently.
- **`SigmaChannel`**: the terms outputs are written from, `X`, `SX`, `COH`,
  `C_OMEGA`; `label` is the prose spelling. `explain_missing_channels` is the
  named-omission clause a writer appends when it declines a channel.
- **`QPSolver`** is orthogonal to `compute_mode`: `one_shot_dft` (default;
  QSGW-Hermitianized Σ_xc at E_DFT diagonalized once), `fixed_point` (diagonal
  on-shell solve; dynamic modes only, a static mode refuses), and
  `self_consistent` (the QSGW loop). `LorraxConfig.qp_solver`: `auto` resolves
  to `self_consistent` on the deprecated `self_consistent = true`, else
  `one_shot_dft`. eqp0/eqp1 use the same at-DFT formula under every solver.
- **`resolve_band_extrapolation`**: `use_band_extrapolation` is the key and
  `sigma_band_extrapolation` a deprecated alias; both named and disagreeing
  refuses. `explicit` selects the behaviour on a non-PPM mode: a defaulted-on
  key auto-disables with a note, an explicitly named one refuses.
  `band_extrapolation_is_consumable` is `ppm_model is not None` on any stage.
- **`sigma_stage_modes`** returns every mode the run dispatches Σ under, in
  order: the staged ladder when `config.sc.stages` exists, else the one
  `compute_mode`. A run-level refusal asks this, never the current stage.
- **`LorraxConfig.omega_grid_ev`**: `n = floor((max − min)/step + 0.5) + 1`;
  the Ry grid is derived by division. With `sigma_omega_patches_ev` the grid is
  the union of patches built by the same formula, and `sigma_omega_min/max_ev`
  become the patch hull. `DynamicSigmaConfig.parsed_omega_patches_ev` refuses
  malformed, unsorted, overlapping or touching patches.

### Screening

- **`HeadCorrection`**: `full` (default; an irreducible direct response is
  completed with its microscopic head/wings exactly once, a micro-reducible
  response is used as is), `no_local_fields` (diagnostic ε head), `off` (no
  special Γ-cell term, for brute-force k convergence).
- **`ScreeningConfig`**: `method` exists only to refuse anything but
  `minimax` (2026-08-06). `diagrams` (`ScreeningDiagrams`) chooses which
  series W sums: `w_rpa` (default, `W = (1 − Vχ₀)⁻¹V`), `w_bse` (ladder W with
  the statically screened direct rung, two-stage: the RPA W(0) is the ladder's
  `W_R`), and `w_rpa_resolvent` (the same resolvent identity with the RPA
  operator, the ladder's `include_w=False` limit, a gate of the resolvent
  machinery against the Dyson route). The fork lives only in
  `gw.screening.compute_screening_model`. It is an enum because the resolvent
  formalism admits more diagram sets.
- **`refuse_unsupported_screening_diagrams`** runs at parse time on resolved
  axes and is a no-op for `w_rpa`. Each non-RPA value has its own table
  (`_W_BSE_REFUSALS`, `_W_RPA_RESOLVENT_REFUSALS`): `x_only`, `hl_ppm`,
  self-consistency and `mc_average_placement != off` refuse for both, and
  `compute_mode = mpa` for `w_rpa_resolvent`. A metallic WFN refuses at the
  stage on its occupations (`{value}_insulators_only`, `gw.screening_bse`);
  `w_bse` also requires a measured time-reversal verdict.
- **`normalize_w_dyson_solver`**: `local`/`auto`/unset → the q-parallel
  per-q dense LU; `distributed` → the 2-D-sharded backsolve through
  `distrib_la`; `lu` → `local` with a deprecation warning; `lstsq` refuses.
  The parser and `w_isdf` share this one vocabulary.

### Layout and linear algebra

- **`resolve_linalg`** interprets `linalg = local | distributed` exactly once
  into `LinalgResolution`; no stage reinterprets the dial. `distributed`
  distributes the W Dyson solve, the transverse LU and the eigensolvers; the ζ
  back-solve is always a whole-tile factor whose `local`/`replicated` tier the
  planner chooses (`zeta_auto_tier`). The internal
  `distributed_lu = 'distributed'` sentinel lowers to cuSolverMp on CUDA and
  ScaLAPACK on CPU.
- **`eigh_backend_choices`** reads `distrib_la.BACKEND_CHOICES`, importable
  without any `.so`; a literal fallback covers a tree without `services/`, and
  `EIGH_CHOICES_SOURCE` records which answered.
  `distrib_la_batched_route_choices` likewise reads the door's batch-route
  vocabulary; `batch_reshard` is the default and `auto` restores the backend's
  scan/stacked route.
- **`MemoryConfig`**: `memory_per_device_gb = 0` auto-detects the GPU;
  `chunk_target_utilization = 0` is the auto sentinel, and a positive
  `ISDF_CHUNK_TARGET_UTILIZATION` overrides the planner's spin-aware default
  after clamping to [0.85, 1.0].

### Band counts

- **`resolve_band_counts`** is the only place band-count precedence exists,
  called once per deck: `nband` is an alias of `number_bands` (both set and
  different → `BandCountConflict`); the umbrella supplies both consumers;
  `number_bands_chi`/`number_bands_sigma` override their own consumer; the
  umbrella and a specific key named with different values refuse, the same
  value is accepted.
- **`BandCounts`** holds `chi`, `sigma` and `isdf = max(chi, sigma)`, the top
  of the loaded ψ window and the ζ-fit window, plus `named`, the keys the deck
  wrote. Nothing downstream re-reads a deck key for a band count;
  `params["nband"]` mirrors `isdf` for tools that read the dict.
  `BandCounts.describe` logs which count won the `max` against the resolved
  ζ-fit edge from `gw_init.resolve_zeta_fit_edge`, never the deck value.
  `zeta_nband` is stored verbatim; its collapse to the default happens only
  in `resolve_zeta_fit_edge`, where the padded edge is known.

### Four-current (bispinor) envelope

- **`BispinorGWMode`** is orthogonal to `ComputeMode`: it selects which Lorentz
  blocks are screened and contracted. Values: `bare_transverse` (default),
  `full_shared_pole`, `full_static_cohsex`. Retired spellings refuse by name in
  `coerce_bispinor_gw_mode`, never aliased.
- **`packed_static_envelope`** is the one table of the packed static photon
  operator's conditions, yielding `(accepted, got, want, klass, why,
  derived_key)`; the key promotion in `from_input_file` and the refusals read
  the same rows. Material class is inferred from WFN occupations, not a row.
- **`packed_bare_transverse_route`** returns `(taken, reason)`:
  `bare_transverse` is the packed static mode with the fifteen current χ
  blocks zero, so the packed Dyson solve is block diagonal and returns
  screened charge COHSEX in CC, bare Breit exchange in TT and zero in CT/TC.
  It is taken exactly inside the envelope its Γ completion is derived for.
- **Route predicates.** `packed_photon_screens_current`: true for
  `full_static_cohsex` (sixteen χ blocks, one packed Dyson solve), false for
  the bare family. `uses_static_photon_response`: both packed static modes.
  `packed_photon_replaces_charge_sigma`: true only for `compute_mode = cohsex`;
  every driver seam asking "may I skip the scalar charge machinery?" asks this.
  `uses_dynamic_packed_photon_route`: charge block on the run's frequency
  model, current blocks at ω = 0. `uses_coupled_photon_head`: the packed
  modes under `head_correction = full`.
- **`incumbent_bispinor_head_record`** returns `(banner, run_record_line)` for
  a bispinor deck off the packed route, so a headless incumbent run carries a
  DEBUG token in `gwjax.out`.
- **`refuse_unsupported_bispinor_gw`** validates the four-current modes and
  requires live direct fields for bispinor QSGW
  (`GATE bispinor_self_consistency_requires_live_four_current`).
  `refuse_unsupported_bispinor_tt_head_correction` guards hand-built configs
  only: `bispinor_tt_head_correction` is not a deck key.
- **`scalar_head_overrides_named`** formats only the scalar-head overrides the
  deck named, for envelope messages.

### Heads, occupations and loop settings

- **`HeadConfig`** holds the q→0 Coulomb-head sources and overrides, consumed
  by `head_correction.HeadResolver`; the BGW vcoul override is diagnostic
  only.
- **`LorraxConfig.occ_broadening_ry`** is the one smearing width every
  occupation solve reads: `occ_smearing_width_ry` (a metal's Fermi-Dirac kBT)
  when declared, else `occ_broadening` converted from eV. `occ_broadening` uses
  BerkeleyGW's MP1 convention, argument `(E − μ)/(2·width)`, half of QE's
  `degauss`. `occ_broadening = 0` selects step occupations; it answers
  whether, not how wide.
- **`_validate_occupation_smearing`**: the metal width must be finite and
  positive; `occ_broadening > 0` beside a metal width refuses
  (`GATE metal_sc_head_update_disabled`).
- **`resolve_mpa_sampling_alpha`** runs after occupations load: fractional
  occupations select 2, integer ones 1; a deck value (1 or 2) wins.
- **`MPAConfig.sample_plan`** returns the double-parallel frequency plan in Ry;
  it is sampling geometry only.
- **`SCConfig`** holds the loop settings read under `qp_solver =
  self_consistent`: `sc_accelerator` accepts only `anderson`
  (`GATE sc_accelerator_anderson_only`), and `eigh` is a layout choice
  (`native` k-sharded batch, `distributed`, or `auto`). The `LORRAX_SC_*` env
  overrides are deprecated and print a note when active. The loop:
  [Self-consistency](../self_consistency.md).
- **`EQP2Config`** is fixed-Σ eigenvalue self-consistency for the opt-in eqp2
  file; it never rebuilds G, χ₀, W or Σ.
- **`BSEConfig`**: `get_centroids_fi` gates the htransform-driven fine-k
  wavefunction recovery (`bandstructure.bse_setup.compute_wfns_fi`).
- **`LorraxConfig`** is the immutable record built once and threaded through
  the driver: top-level system geometry and mode axes, grouped sub-configs
  for the rest.
