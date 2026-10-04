# GW driver and configuration contracts

This page states the binding contracts of the GW driver `gw.gw_jax` and of the
functions and classes in `gw.gw_config`, whose one-line docstrings point here.
It is for developers changing either module; users want the
[input reference](../input_reference.md), which owns every deck key and its
default. Read [design decisions](../architecture/decisions.md) first: its dated
rulings override anything here.

## Driver invariants (`gw.gw_jax`)

- **One timing table that sums to the wall.** `main()` calls `timing.reset()`
  on entry, and `_close_timing` closes the table with `timing.report(wall=...)`,
  so the printed rows plus `(untimed)` equal the process wall. The span before
  `main()` (the communicator stack and imports) is decomposed from
  `RUNTIME.facts["elapsed"]` into `gw_jax.runtime_stack.*` and `gw_jax.imports`
  rows (`_record_process_wall`). A refused run prints the same table before it
  raises (`_report_refused_run`), because it is the only record of where that
  run's wall went.
- **Refuse before compute.** Every check that needs only the deck runs while
  `LorraxConfig.from_input_file` builds the record (`_apply_input_envelope`:
  the screening-diagram, bispinor and metal-q0 refusals), and
  `refuse_unimplemented_compute_mode` and `validate_band_extrapolation` run
  next, all before the WFN is opened. Checks that need the material class,
  which is read from the WFN occupations, run right after the WFN header
  (`validate_material_inputs`), and checks that need the measured time-reversal
  verdict run once the symmetry tables exist; both run before the ζ fit. A
  refusal therefore never costs a ζ fit.
- **The mesh is built once**, by `runtime.initialize_communicator_stack` at
  import, through `common.collectives.prepare_mesh`, which warms every
  communicator (`nccl_warmup` on GPU, `warm_mesh_cliques` on CPU/MPI). The
  driver reads it as `RUNTIME.mesh`. Do not call `prepare_mesh()` again: a
  second `Mesh` is a second set of communicators and of jit caches.
- **HDF5 library instances are measured**, not assumed: `file_io.hdf5_owner.probe`
  reads `/proc/self/maps` at startup and again after each SC map's store cycle
  (`sc_iteration`). It is silent while safe and always prints when two libhdf5
  instances are mapped and one file was written through both.
- **Route and head provenance go into `gwjax.out`.** The run record names the
  head policy, which bispinor route ran (`packed_bare_transverse_route` returns
  the first unmet condition when the packed route is not taken), and the
  ζ-fit band window from the resolved edge (`gw_init.resolve_zeta_fit_edge`,
  never the deck value).
- **One head resolver.** `head_correction.HeadResolver` is built once in
  `_run_gw_stages` and memoizes the q→0 head samples; the static head, the W0
  restart head and the dynamic head all read it, so every stage sees the same
  head.
- **SC map 0 is the one-shot.** Under `qp_solver = self_consistent` the driver
  skips the one-shot Σ (`_run_oneshot_sigma`); the SC loop's first map computes
  it, with $U = I$, so it equals the one-shot bit for bit
  ([self-consistency](../self_consistency.md)).
- **Σ_x sign check.** Every Σ_x diagonal entry must be negative, because Σ_x is
  a negative-definite quadratic form. On the one-shot path `sanity.check_sign`
  prints a sanity failure line for a positive entry (a sign, conjugation or
  band-index slip); `LORRAX_SANITY=strict` turns it into a refusal.
- **Non-finite refusals.** On the one-shot path `common.sanity.refuse_nonfinite`
  refuses a non-finite `kin_ion` or Σ_total before the QP eigh and a non-finite
  E_QP after it, because LAPACK returns silently on a NaN-bearing matrix and
  the run would otherwise exit 0 with NaN eqp files. It ignores
  `LORRAX_SANITY`; `LORRAX_ALLOW_NONFINITE_RESULT=1` downgrades it to a
  warning, for forensic runs that want the artifact on disk.
- **No gathers for printing.** Writers on the reporting rank read bounded
  `(nk, nb)` diagonals (`qsgw_utils.static_sigma_diag_to_host`,
  `dynamic_sigma.extract_sigma_diag_logical`); a band-sharded `(nk, nb, nb)`
  operator is never converted to a host array, and the other ranks keep no
  host copy of the Σ(ω) diagonal.
- **Degenerate-set averaging touches reported diagonals only.**
  `degen_average.average_within_degenerate_sets` (BerkeleyGW's
  `Sigma/shiftenergy.f90` convention, tolerance `degen_avg_tol_ry`, off with
  `no_degen_averaging`) averages Σ_x, Σ_c at E_DFT, the Σ(ω) diagonal and the
  head columns before they are written. Every operator that is diagonalized
  stays unaveraged, because averaging only the diagonal of a degenerate block
  depends on the arbitrary basis inside it.
- **`write_eqp2` never recomputes GW.** `sc_iteration.run_fixed_sigma_evsc`
  iterates the one-shot full-matrix Σ(ω), rotated into each updated QP basis;
  G, χ₀, W and Σ are built once.

## Configuration contracts (`gw.gw_config`)

### Parsing

- **`read_lorrax_input`** reads the `[cohsex]` section and removes the QE
  `K_POINTS` block from it (its band-path segments go to
  `params["kpoints_crystal_b"]`). Parsing is strict: every unknown key refuses
  in one aggregated error with line numbers (`_deck_key_line`), and a retired
  key gets its own report naming the replacement. Key names are case-folded on
  both sides of the unknown-key check. `#` starts a comment; a `;` after a
  value refuses with its line number, because configparser would keep it in
  the value. A deck without `sys_dim` refuses (`GATE sys_dim_required`): it
  selects the Coulomb truncation, and no other key implies it. The parser
  records which keys the deck named (`LorraxConfig.raw_input_keys`), so an
  explicit default and an absent key are distinguishable where that matters.
  `_print_deck_report` prints the hygiene report on rank 0 only.
- **`LorraxConfig.from_input_file`** resolves the typed record once.
  `runtime_platform` (`cpu` or `gpu`) stands in for a device in a preflight
  without one; `resolve_hardware=False` leaves `memory_per_device_gb = 0` at
  its zero sentinel and probes no device. Production callers use the defaults.
  Only `restart = true` enters the restart loader (`gw_init`); an existing
  file in `tmp/` is not permission to reuse it.
- **`env_float`**: unset or blank → the default; an unparseable value prints a
  `LORRAX SANITY` line saying the knob is not in force, or, with
  `refuse=True` (knobs that gate correctness), raises naming the variable. It
  never applies a default silently.
- **`active_zeta_truncating_knobs`** lists the env knobs in force that stop the
  ζ fit early (`ZETA_TRUNCATING_ENV_KNOBS`: `LORRAX_MAX_RCHUNKS`). Its readers
  keep a truncated ζ from being stamped as complete, so a later production run
  cannot reuse it.
- **Normalizers.** `coerce_compute_mode`, `coerce_screening_diagrams`,
  `coerce_head_correction` and `coerce_bispinor_gw_mode` accept the enum, its
  `.value` or a string; `_normalize_placement` (`head_channel.normalize_placement`)
  accepts a string. A typo raises naming the legal set and never resolves to a
  default.

### Self-energy and solver axes

- **`ComputeMode`** names the ansatz for W's frequency dependence: `x_only`,
  `cohsex`, `gn_ppm`, `hl_ppm`, `mpa`. It names the ansatz, not the numerics:
  a generic "full frequency" value would not say which of contour deformation,
  real-axis quadrature or MPA to run. `is_dynamic` means "this run has an ω
  axis" (GN/HL-PPM and MPA). `ppm_model` is `'gn'` or `'hl'` and `None` for MPA
  and the static modes, so a site that means "which two-point PPM fit" asks
  `ppm_model`, never `is_dynamic`.
- **`LorraxConfig.compute_mode`**: `auto` (the default) infers the mode from
  the older `do_screened`, `use_ppm_sigma` and `ppm_model` keys; an explicit
  value wins, and an explicit screened mode beside `do_screened = false`
  refuses. Resolving is not permitting: `refuse_unimplemented_compute_mode`
  runs at driver entry and raises `NotImplementedError` (distinct from a typo's
  `ValueError`) for a mode in `UNIMPLEMENTED_MODES`, which is empty.
- **`announce_legacy_sigma_axis_keys`** prints one deprecation note per key in
  `LEGACY_SIGMA_AXIS_KEYS` that the deck named, with the canonical spelling,
  and returns them; nothing is refused and nothing resolves differently.
- **`SigmaChannel`** lists the terms outputs are written from: `X`, `SX`,
  `COH`, `C_OMEGA`; `label` is the prose spelling. `MODE_SIGMA_CHANNELS` says
  which channels each mode builds, and `explain_missing_channels` is the
  clause a writer appends when it omits a channel the mode does not build.
- **`QPSolver`** is orthogonal to `compute_mode`. `one_shot_dft` (the default)
  diagonalizes the Hermitian QSGW Σ_xc evaluated at E_DFT once;
  `self_consistent` runs the QSGW loop, with Σ at each map's own energies
  ([self-consistency](../self_consistency.md)). No QP root is solved, and
  `qp_solver = fixed_point` refuses by name. `LorraxConfig.qp_solver`: `auto`
  resolves to `self_consistent` when the deprecated `self_consistent = true` is
  set, else to `one_shot_dft`.
- **`resolve_band_extrapolation`**: `use_band_extrapolation` is the key and
  `sigma_band_extrapolation` a deprecated alias; naming both with different
  values refuses. It returns `(enabled, explicit)`. On a run with no stage
  that consumes the key, a defaulted-on key is turned off with a note and an
  explicitly named one refuses (`sigma_dispatch.validate_band_extrapolation`).
  `band_extrapolation_is_consumable` is true when any stage is GN/HL-PPM, or
  is MPA with a Σ_c that runs the scalar pole-sum executor
  (`mpa_sigma_runs_scalar_executor`: scalar decks and both bispinor
  shared-pole routes, `bare_transverse` and `full_shared_pole`; the latter
  brackets its CC class only).
- **`sigma_stage_modes`** returns every mode the run dispatches Σ under, in
  order: the staged ladder when `config.sc.stages` exists, else the one
  `compute_mode`. A run-level refusal asks this, never the current stage.
- **`LorraxConfig.omega_grid_ev`** is the grown support `sc_omega_grid_ev`
  once `gw.qp_support` has set it (the one-shot and every SC map). Before
  that it is the requested grid: `n = floor((max − min)/step + 0.5) + 1`
  points per contiguous range, an unset edge being the sample next to E_F.
  The Ry grid is derived by division. With `sigma_omega_patches_ev` it is the
  union of the `lo:hi` patches built by the same formula;
  `DynamicSigmaConfig.parse_omega_patches_ev` refuses a malformed patch and
  patches that are not ascending and separated by at least one step.
  `lo:hi:eta` triples are coarse windows, parsed and refused
  (`GATE sigma_coarse_window`) by `parse_coarse_windows_ev`.

### Screening

- **`HeadCorrection`**: `full` (the default: an irreducible direct response is
  completed with its microscopic head and wings exactly once, and a
  micro-reducible response is used as is), `no_local_fields` (the diagnostic ε
  head, or the first-order direct four-current Γ head on the bispinor
  shared-pole routes), `off` (no special Γ-cell term, for brute-force k
  convergence).
- **`ScreeningConfig`**: `method` exists only to refuse anything but `minimax`
  ([decisions](../architecture/decisions.md), 2026-08-06). `diagrams`
  (`ScreeningDiagrams`) chooses which series W sums: `w_rpa` (the default,
  $W = (1 - V\chi_0)^{-1}V$), `w_bse` (the ladder W with the statically
  screened direct rung; the RPA W(0) is the ladder's $W_R$), and
  `w_rpa_resolvent` (the same resolvent identity with the RPA operator, the
  ladder without its rung, which checks the resolvent machinery against the
  Dyson route). The fork lives only in `gw.screening.compute_screening_model`.
  It is an enum because the resolvent formalism admits more diagram sets.
- **`refuse_unsupported_screening_diagrams`** runs at parse time on the
  resolved axes and does nothing for `w_rpa`. Each other value has its own
  table (`_W_BSE_REFUSALS`, `_W_RPA_RESOLVENT_REFUSALS`): `x_only`, `hl_ppm`,
  self-consistency and `mc_average_placement != off` refuse for both, and
  `compute_mode = mpa` refuses for `w_rpa_resolvent`. A metallic WFN refuses at
  the stage, on its occupations (`GATE {value}_insulators_only`,
  `gw.screening_bse`), because no deck key declares a metal. `w_bse` also
  requires a measured time-reversal verdict (`GATE w_bse_requires_measured_trs`,
  `screening.refuse_w_bse_without_trs`).
- **`normalize_w_dyson_solver`**: `local`, `auto` or unset → the q-parallel
  per-q dense LU; `distributed` → the 2-D-sharded backsolve through
  `distrib_la`; `lu` → `local` with a deprecation warning; `lstsq` refuses,
  because a rank-deficient $1 - V\chi_0$ means the centroid basis is
  over-complete and a minimum-norm solve would hide it.

### Layout and linear algebra

- **`resolve_linalg`** interprets `linalg = local | distributed` exactly once
  into `LinalgResolution`; no stage reinterprets the dial. `distributed`
  selects the distributed W Dyson solve, the distributed transverse LU
  (`distributed_lu`) and the distributed eigensolvers, including the SC
  eigh; the ζ back-solve is a whole-tile factor applied on its q owners under
  either value.
- **`eigh_backend_choices`** reads `distrib_la.BACKEND_CHOICES`, which imports
  without any `.so`; a literal fallback covers a tree without `services/`, and
  `EIGH_CHOICES_SOURCE` records which answered.
  `distrib_la_batched_route_choices` likewise reads distrib_la's batch-route
  vocabulary. `local` takes the `batch_reshard` route; `distributed` takes
  `auto`, the backend's own scan or stacked route.
- **`MemoryConfig`**: `memory_per_device_gb = 0` auto-detects the device memory
  and takes the minimum over processes (`gpu_utils.minimum_process_budget_gb`),
  so every rank plans the same tile shapes. `chunk_target_utilization = 0` is
  the auto sentinel; a positive `ISDF_CHUNK_TARGET_UTILIZATION` overrides the
  planner's default after clamping to [0.85, 1.0]. Its only reader is the
  ζ μ-batch planner (`gflat_memory_model.plan_zeta_route_g`, called from
  `gw_init`), one of the budget-sized planners the
  [fixed-tile ruling](../architecture/decisions.md#fixed-tile) lists as not
  yet conforming.

### Band counts

- **`resolve_band_counts`** is the only place band-count precedence exists,
  called once per deck: `nband` is an alias of `number_bands` (both set and
  different → `BandCountConflict`); the umbrella supplies both consumers;
  `number_bands_chi` and `number_bands_sigma` override their own consumer; an
  umbrella and a specific key named with different values refuse, and the same
  value is accepted. A key's value is read only when the deck named it, so the
  umbrella's default cannot outrank an explicit alias.
- **`BandCounts`** holds `chi`, `sigma` and `isdf = max(chi, sigma)`, the top
  of the loaded ψ window and of the ζ-fit window, plus `named`, the keys the
  deck wrote. Nothing downstream re-reads a deck key for a band count;
  `params["nband"]` mirrors `isdf` for tools that read the dict.
  `BandCounts.describe` logs which count won the `max`, against the resolved
  ζ-fit edge. `zeta_nband` may only narrow the window (`[1, isdf]`) and is
  stored verbatim; it collapses to "follow the loaded window" only in
  `gw_init.resolve_zeta_fit_edge`, where the mesh-padded edge `b4` is known.

### Four-current (bispinor) envelope

- **`BispinorGWMode`** is orthogonal to `ComputeMode`: it selects which Lorentz
  blocks are screened and contracted. Values: `bare_transverse` (the default),
  `full_shared_pole`, `full_static_cohsex`. Retired spellings refuse by name
  in `coerce_bispinor_gw_mode`, never aliased.
- **`packed_static_envelope`** is the one table of the packed static photon
  operator's conditions. It yields `(accepted, got, want, klass, why,
  derived_key)` rows; the refusals and `packed_bare_transverse_route` read the
  same rows. The material class is inferred from the WFN occupations, not a
  row.
- **`packed_bare_transverse_route`** returns `(taken, reason)`. The packed bare
  route is taken for a `bare_transverse` slab deck (`sys_dim = 2`) inside the
  envelope: one-shot COHSEX (static) or GN/HL-PPM (W₀₀(ω) on the charge block,
  the fifteen current blocks frozen at ω = 0). With the fifteen current χ
  blocks zero, the packed Dyson solve is block diagonal and returns screened
  charge W in CC, bare Breit exchange in TT and zero in CT/TC.
- **Route predicates.** `packed_photon_screens_current`: true only for
  `full_static_cohsex` (sixteen χ blocks, one packed Dyson solve).
  `uses_static_photon_response`: `full_static_cohsex`, or the packed bare
  route. `packed_photon_replaces_charge_sigma`: a packed route under
  `compute_mode = cohsex`; every driver seam asking "may I skip the scalar
  charge machinery?" asks this. `uses_dynamic_packed_photon_route`: a packed
  route under GN/HL-PPM. `uses_coupled_photon_head`: a packed route under
  `head_correction = full`.
- **`incumbent_bispinor_head_record`** returns `(banner, run_record_line)` for
  a bispinor deck off the packed route, so a headless run carries a
  `WARNING -- DEBUG` token in `gwjax.out`.
- **`refuse_unsupported_bispinor_gw`** validates the four-current modes and
  requires live direct fields for bispinor QSGW
  (`GATE bispinor_self_consistency_requires_live_four_current`).
  `refuse_unsupported_bispinor_tt_head_correction` guards hand-built configs
  only: `bispinor_tt_head_correction` is not a deck key.
- **`scalar_head_overrides_named`** formats the scalar-head overrides the deck
  named, for envelope messages.

### Heads, occupations and loop settings

- **`HeadConfig`** holds the q→0 Coulomb-head sources and overrides, read by
  `head_correction.HeadResolver`; the BerkeleyGW vcoul override is diagnostic
  only.
- **`LorraxConfig.occ_broadening_ry`** is the one smearing width every
  occupation solve reads: `occ_smearing_width_ry` (a metal's Fermi-Dirac
  $k_BT$) when set, else `occ_broadening` converted from eV. `occ_broadening`
  uses BerkeleyGW's MP1 convention, argument $(E - \mu)/(2w)$, so it is half of
  QE's `degauss` (`gw.efermi`). `occ_broadening = 0` selects step occupations;
  it answers whether, not how wide.
- **`_validate_occupation_smearing`**: the metal width must be finite and
  positive, and `occ_broadening > 0` beside a metal width refuses
  (`GATE metal_sc_head_update_disabled`), because a metal has one width.
- **`resolve_mpa_sampling_alpha`** runs after the occupations load: fractional
  occupations select 2, integer ones 1; a deck value (1 or 2) wins.
- **`MPAConfig.sample_plan`** returns the double-parallel frequency plan in
  Ry; it is sampling geometry only.
- **`SCConfig`** holds the loop settings read under
  `qp_solver = self_consistent`. `sc_accelerator` accepts only `anderson`
  (`GATE sc_accelerator_anderson_only`). `eigh` comes from the `linalg` dial
  (`resolve_linalg`): `local` gives `auto`, `distributed` gives
  `distributed`; `native` is also accepted. `native` diagonalizes whole
  (nb, nb) tiles batched over k (`distrib_la`'s `batch_reshard` route);
  `distributed` spreads each tile over the mesh; `auto` takes `distributed`
  only on a multi-device mesh where one tile exceeds
  `qsgw_density.BAND_TILE_BUDGET_FRACTION` of the per-device budget and the
  distributed backend resolves, else `native`
  (`sc_iteration._resolve_sc_eigh`). `LORRAX_SC_MAX_ITER`,
  `LORRAX_SC_TOL_EV` and `LORRAX_SC_DUMP_DIR` override
  their deck keys and print a deprecation note when set. The loop: [self-consistency](../self_consistency.md).
- **`EQP2Config`** configures fixed-Σ eigenvalue self-consistency for the
  opt-in `eqp2.dat` (`write_eqp2`); it never rebuilds G, χ₀, W or Σ.
- **`BSEConfig`**: `get_centroids_fi` gates the htransform-driven fine-k
  wavefunction recovery (`bandstructure.bse_setup.compute_wfns_fi`).
- **`LorraxConfig`** is the frozen record built once and threaded through the
  driver: top-level system geometry and mode axes, with grouped sub-configs
  for the rest.
