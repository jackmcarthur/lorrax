# ζ fitting on raw-parent faces

`gw.gw_init.fit_zeta` resolves independent charge/current reuse; every fresh
`gw.isdf_fitting.fit_zeta_to_h5` call requires a typed
`CentroidKUnfoldPlan` and both packed raw-parent faces. There is no fresh
GW full-k fallback, including when the raw parents already span the full
k grid. The temporarily accepted `low_mem_bands=false` deck spelling warns
and uses parents; its policy belongs to [decisions](decisions.md).

## Carriers and band windows

Each centroid family owns its basis, parent plan and two un-conjugated faces:

```text
psi_mun[parent, spin, mu_X, band_Y]
psi_nmu[parent, band_X, spin, mu_Y]
```

The band GEMM merges spin and centroid only at the shared
`common.contract_bands` seam. `distrib_la.gemm_plan` owns the distributed
contraction and its backend communication. Left/right logical band windows
are zero/one weights over the padded loaded extent, so an arbitrary band
edge need not divide either mesh axis. The same weights serve C_q and Z_q.
Pseudoband normalization remains unsupported and is refused before fitting.

The charge and current parent faces are the fit's only centroid ψ inputs.
The loader samples the packed centroid table at raw IBZ k rows, then drops
its temporary loader orientations. `PsiGStore` and the optional all-P-sharded
ψ(r) cache also hold raw parents. A cache-capacity miss streams the same
band chunks through the same transform owner; it does not select different
normal equations.

## Parent projectors, typed transport and vertices

`isdf.core._c_q_face_parent` and `_z_q_face_parent` contract open-spin
projectors on the parents. Every spatial permutation, lattice-wrap phase,
spin rotation and antiunitary action comes from `symmetry_maps` through the
parent plan. See [the symmetry register](symmetry_register.md) for the
four-spinor action and [four-current wiring](four_current_wiring.md) for the
Lorentz convention.

A current vertex acts on the output spin indices after typed transport.
Both C_q and Z_q retain their vertices; a four-component carrier alone is
not the current Gram. The C tail uses `gamma_double_contract`; the Z tail
uses the canonical vertex's output-index permutation and phase. Stored
parent faces are never vertex-folded.

The equal-current solve preserves the paired signed C/Z convention when
regularizing. `_transverse_lu_ridge` owns the shift for all four local or
distributed, hoisted or fused preparations; its literal oracle and default
mode are recorded in the [convention register](symmetry_register.md#integration-closures-2026-09-06).

C_q uses one planned parent-row GEMM per endpoint followed by the typed
operator unfold and IFFT/product/FFT tail. Z_q accumulates parent projectors
across bounded band chunks, then streams output-spin blocks through the
same typed operator action. Its band-block owner broadcast and real-grid
scatter remain communication-bearing operations; local symmetry transport
adds no collective. Zero explicit HLO collectives must never be interpreted
as zero communication inside native GEMM providers.

## Orbit-closed real-grid tiles

A contiguous real-space slab need not be closed under symmetry.
`RealGridOrbitTiles` places whole orbits on each Y owner and supplies the
r-slot indices, local permutations and wraps as runtime operands. Parent
centroids likewise keep each orbit on one owner. Thus both endpoint
symmetry gathers stay local.

The Z kernel carries centroid axes in packed order and its r axis in tile
slot order. The q-selected RHS enters the existing factor/solve owner;
`accumulate_rchunk_to_gflat(r_indices=...)` scatters solved tile slots into
the reciprocal-space accumulator. Every pad slot receives a distinct
out-of-range drop sentinel. A tile's width is bounded below by a whole
orbit on every owner; if this exceeds the requested chunk width the driver
reports the larger live set explicitly.

## One centroid order; canonical files

Every in-memory centroid axis uses its family's `PackedCentroidBasis`.
Whole orbits share a shard, with exact-zero suffix pads on each shard;
these are generally not a global suffix. Dense factors and solves use
`meta.mu_solve_extent`. C_q pads receive C_q's physical mean diagonal,
not an arbitrary unit value that could become the spectral cutoff scale.
Z_q pads stay zero, and the GN-PPM dead-mode selector uses the active-slot
mask. The Dyson matrix already has unit pad entries.

Files retain canonical centroid-file order at logical extent. Readers pack
and writers unpack at the I/O seam only. Canonical staging padding may
differ from runtime packing. The same rule covers ζ, parent restart faces,
V/W and MPA stores, leaving file shapes independent of the processor grid.
The shared BSE/htransform/downfold readers retain their own documented
contracts; they do not gain a GW full-k fallback through this fit API.

## Coupled current schedule

Charge fitting is independent. The three current channels couple only when
all need fresh fits and the planner admits the complete coupled live set.
Partial reuse and a capacity miss fit the missing channels sequentially with
the same parent equations. Each current C_q is prepared separately.

For each tile, `_z_q_face_parent(coupled_mu123=True)` shares the parent
projectors. For each output spin pair it computes the left child transport
and inverse FFT once, then advances the three channel accumulators in the
canonical vertex order. Each channel retains its original spin-pair
reduction order. The single-channel tail remains the sequential fallback.

The coordinator releases one channel's RHS to its solver at a time, in
μ=1→2→3 order. `batch_reshard` and the distributed factor-token route keep
their existing numerical boundaries; no opaque factor is gathered or
concatenated. Explicit backend requests are preserved when coupling does
not fit. Capacity equations and provider envelopes belong to the
[memory model](memory-model.md).

Each channel's G-flat accumulator is spilled to process-local host storage.
Only the active accumulator is restored to device for accumulation. Final
restore, canonical write, close and provenance stamp remain ordered after
all channels finish the tile loop. A reused charge fit may omit its fit-time
plan because no charge fit executes; current reuse remains independent.

`gw_jax.zeta_fit_transverse` measures the outer current-fit schedule once,
including its ordered setup/solves/writes. Per-channel elapsed intervals can
overlap and must not be summed as an isolated ζ_T wall time.

## GW consumers and retained boundaries

The same parent carrier serves screening and Sigma. Completed parent band
operators unfold through the typed band-operator action. Dynamic heads
batch typed child faces in parent order; the SC density rebuild loads its
computational parent G-sphere domain and projects through typed scalar/polar actions. Neither requires a
persistent full-k GW wavefunction carrier. Unsupported non-RPA consumers
and old full-face GW restart stores refuse explicitly.

The GW Zq entry requires typed parents and orbit-tile tables; its old full-k
kernels and full-face Cq are deleted. The legacy
rectangular Cq implementation has a live downfold consumer and remains a
shared service. Galerkin's BSE/htransform fit and generic sample loaders
are separate and retained.

## Verification scope

`tests/test_isdf_zq_parent_parity.py` exercises glide, k reduction,
antiunitary rows and spin mixing against direct NumPy q/band sums, with all Lorentz pairs and short band chunks;
`tests/test_parent_projector_unfold_oracle.py` compares typed projector
transport to the wavefunction loader. Fresh physical fits and canonical
processor-grid round trips are distinct gates from these algebra tests.
The sandbox campaign report `reports/bisp_parent_route_2026-09-05/report.md`
records their job IDs, strict array residuals, QP comparisons and open items.
Historical face-route measurements remain in this page's git history;
they are not evidence that a retired GW route remains selectable.

## Unreduced admission for nonclosed centroid sets

A nonclosed charge or current centroid set selects the service-owned
`SymMaps.trivial_view()` before either family is packed. `parent_k_domain`
then requests loader-unfolded full-k states for the same centroid and G-space
stores; the existing plan has `n_parent = nk`, identity actions and every q
row. The original loader remains authoritative for file energies and the
G-sphere unfold. No deleted full-k Zq or Green kernel is reinstated. The
binding admission ruling is in [decisions.md](decisions.md); the historical
fixtures keep their original centroid coordinates and printed references.


### Zeta solve entry contract (2026-09-06 phase extraction)

Solve for zeta_q given pre-computed system matrix from
:func:`factor_c_q`.

For ``vertex_mu_L == 0`` ``L_q`` is the lower-triangular Cholesky
factor of CCT and the inner solve is two triangular substitutions
(``L y = Z`` then ``L^H ζ = y``).  This is the historical fast
path — bit-identical to the previous implementation.

For ``vertex_mu_L != 0`` the equal-current CCT is a signed Gram;
Gamma2 carries the negative sign shared by its RHS.  Pivoted LU uses
:func:`_transverse_lu_ridge` so regularization preserves the paired sign.
Since
the 2026-08 hoist ``factor_c_q`` computes that LU ONCE per channel on
the local plan (including provider selection with ``batch_reshard``):
``L_q`` carries the packed factors and
``lu_piv`` the permutation, and this routine only APPLIES them per
r-chunk (``lax.linalg.lu_solve`` — bit-identical to the fused
``jnp.linalg.solve``).  On the ScaLAPACK and cuSOLVERMp plans ``L_q``
is instead a :class:`distrib_la.FactorToken`, which carries the
block-cyclic factors and rank-private pivots; this module never opens
that token.
Bunch-Kaufman LDL^T would be the natural Hermitian-
indefinite factorization but JAX doesn't expose it; pivoted LU is
numerically equivalent for our purposes.

Uses q-chunked all-gather strategy: gather B_q matrices at a time,
then solve all B_q systems in parallel using vmap.

Memory trade-off:
- q_chunk_size=1: Minimum memory (one matrix replicated at a time)
- q_chunk_size=nq: Maximum parallelism (all matrices replicated)

Args:
    L_q: (nq, n_rmu, n_rmu) Cholesky factor (μ_L=0) or raw CCT
         (μ_L=1,2,3), sharded P(None, 'x', 'y') — OR a
         :class:`distrib_la.FactorToken` from one of the three
         library-handle routes, which is consumed whole and never
         indexed, resharded or gathered
    Z_q: (nq, n_rmu, n_zchunk) ZCT matrix, sharded P(None, 'x', 'y')
         or P(None, None, ('x','y')) if caller already resharded
    mesh_xy: 2D device mesh
    q_chunk_size: Number of q-points to solve simultaneously (default 1)
    vertex_mu_L: Lorentz vertex index — selects Cholesky-back-solve
                 vs jnp.linalg.solve.  Output sharding is identical
                 in both branches.
    solver_kind: 'auto' (default) defers to :func:`_resolve_solver_kind`;
                 explicit values are 'replicated_cholesky' (mesh-
                 invariant dense factor from :func:`_factor_c_q_replicated`;
                 back-solve shares the 'sharded_cholesky' per-q
                 triangular path — L is replicated, r-columns sharded,
                 so ζ is grid-agnostic), 'sharded_cholesky' (legacy 2D
                 blocked chol + per-q triangular solve), 'lu' (per-q
                 pivoted-LU for transverse channels),
                 'cusolvermp_cholesky' (distributed potrs via FFI),
                 'cusolvermp_lu' (distributed getrf+getrs via FFI
                 for the transverse channels), or
                 'replicated_rank_truncate' (charge rank-truncation:
                 ``L_q`` is the pseudo-inverse factor B, back-solve is
                 the matmul ζ = B(BᴴZ)), or
                 'distributed_rank_truncate' (the
                 ``distributed_zeta_solve='distributed'`` tier: ``L_q``
                 is the truncated pseudo-inverse ``C⁺`` itself, kept
                 2D-sharded, and the back-solve is one stacked GEMM
                 with BOTH operands 2D-sharded — see
                 :func:`_distributed_pinv_apply`).  'replicated_cholesky',
                 'sharded_cholesky' and 'replicated_rank_truncate' all
                 take the general shard_map back-solve branch below
                 (none matches the cuSolverMp/scalapack guards).
    n_rmu_logical: Logical centroid count.  When given and smaller
                 than the padded input extent, every per-q dense
                 solve (pivoted LU AND the per-q triangular
                 back-solve) is μ-SLICED to this extent before the
                 factorisation and the ζ pad rows are zero-filled
                 after.  This is load-bearing for device-count
                 invariance: solving the identity-padded system at
                 the padded extent makes ζ depend deterministically
                 on the pad extent (= on the device count), with
                 O(1) amplification in the near-null transverse
                 modes (reports/device_invariance_2026-07-08/
                 ROOT_CAUSE.md).  ``None`` keeps the padded extent
                 (back-compat for mesh-divisible callers).

Returns:
    zeta_q: (nq, n_rmu, n_zchunk) solution, sharded P(None, ('x','y'), None)
            — μ-axis flat-sharded across the ('x','y') mesh product,
            r-axis replicated.  This is the layout the downstream
            G-flat FFT (``accumulate_rchunk_to_gflat``) wants:
            each rank owns a μ-slab over the full r-extent, so the
            per-rank cuFFT runs locally without resharding.


### Charge/current fit orchestration phase rulings (2026-09-06)

```text
	# Honour cohsex.in ``gamma_contract_mode`` for the γ̃·γ̃ kernel
	# inside the monolithic pair pipeline.  Mode is module-level (the
	# γ̃ contract sits inside shard_map bodies so threading a kwarg
	# through every call would be churn for no benefit).
	# Chunk sizes (band_chunk / chunk_r / q_chunk / gflat_chunk_size) were
	# picked once by ``plan_gflat_chunks`` in the caller and live in
	# ``chunks``; fit_zeta is a pure consumer.
	# Any missing bispinor channel consumes the transverse identity already
	# resolved by the per-channel contract.  Only the fit-specific chunk plan
	# remains here.
		# Chunk-plan the TRANSVERSE channel SEPARATELY from the charge
		# ``chunks`` this function was handed.  μ_T is typically ≈ μ_C/3,
		# and reusing the charge-sized plan unchanged for all three ζ_T
		# fits is exactly the register row this closes: "three ζ_T fits
		# inherit the CHARGE chunk plan (μ_T≈μ_C/3): ~3x extra r-chunks,
		# ~2.7 GB/rank avoidable gather".  ONE call here, after reuse was
		# definitively declined and ahead of the μ_L fit loop; all three
		# Lorentz components share this one transverse-sized plan.
		# The shared Zq transform is coupled, but the three real transverse
		# systems retain their accepted 36-q solve boundaries.  Flattening them
		# to 108 q exposed input-sensitive ~1e-9 arithmetic drift on CrI3 while
		# saving only a small solve dispatch inside a transform-dominated chunk.
		# The deck/planner budget is rank-invariant and no larger than the
		# allocator pool.  Using it for the measured 50% operand ceiling is a
		# conservative static dispatch; process-local memory_stats could differ
		# transiently and must not choose different collective routes by rank.
	# Fresh writers and provenance stamps consume exactly the identity that
	# was tested before planning; there is no second reconstruction here.
	# Device kernels can only record closure/certification findings.  The
	# shared host seam must dispose them after this fresh writer and before its
	# provenance stamp makes the artifact reusable.
	# Stamp what this ζ was fit FOR, so a later run can reuse it.  AFTER
	# the fit (and therefore after ``mark_zeta_done`` inside
	# fit_zeta_to_h5) on purpose: a job killed between the two leaves a
	# complete-but-unstamped file, which _zeta_reuse_ok refits.  Rank 0
	# only, then a barrier so no rank races ahead of the write.
	#
	# EXCEPT when a truncating knob was in force.  ``LORRAX_MAX_RCHUNKS=N``
	# breaks the r-chunk loop after N chunks (gw/isdf_fitting.py) and the
	# writer downstream of the loop still calls ``mark_zeta_done``, so the
	# partial ζ is stamped COMPLETE on disk.  Provenance records the
	# CONFIGURATION, which a later production run in the same directory
	# reproduces exactly — so stamping here would make _zeta_reuse_ok
	# reuse a truncated ζ and produce silently wrong physics from a
	# profiling knob.  Refusing the stamp breaks that chain outright
	# (rule 4 of _zeta_reuse_ok: no provenance ⇒ refit).
	#
	# The writer now consults the SAME knob list before calling
	# ``mark_zeta_done``, so a truncated file also carries
	# ``zeta_is_done=False``.  The two guards stay separate on purpose —
	# provenance answers "may a later run REUSE this", zeta_is_done answers
	# "did the writer FINISH" — and either alone stops the reuse.
			# Non-fatal: the ζ itself is fine, it just won't be reusable.
		# WHERE DID THIS NUMBER COME FROM?  ``fit_zeta_to_h5._track_peak``
		# prefers ``memory_stats()['peak_bytes_in_use']`` and falls back to
		# an nvidia-smi whole-GPU sample when that is 0/absent — and the
		# caller cannot tell which fired.  Under the ``platform`` allocator
		# the arena reports bytes_limit=0 AND peak_bytes_in_use=0
		# (measured, job 7882447), so every figure printed there is the
		# nvidia-smi fallback: the whole card, other processes included.
		#
		# Four bugs in the previous three lines, all silent:
		#   * ``== "platform"`` was case-SENSITIVE while jax lowercases
		#     (jaxlib/xla_client.py:190), so ``=PLATFORM`` printed bare;
		#   * it never matched ``cuda_async``, which is what
		#     ``config/frontera/ffi_env.sh:24`` deploys;
		#   * it called ``platform`` "cuda_async" — three distinct
		#     allocators, and ``platform`` is plain cudaMalloc;
		#   * its ``TF_GPU_ALLOCATOR`` clause was dead (inert for jax).
		# And its premise — "cuda_async under-reports" — was not
		# reproduced: peak_bytes_in_use measured IDENTICAL to BFC.
		#
		# The environment alone cannot answer this: the allocator is fixed
		# at backend init, so a variable set later changes the string and
		# not the client (job 7882443).  Corroborate against the device.
		# Label the line by the LIVE backend.  On a CPU-backend run that
		# lands on a GPU node (JAX_PLATFORMS=cpu with SLURM still exporting
		# CUDA_VISIBLE_DEVICES) the nvidia-smi fallback reads a GPU this
		# run never used, so calling it a "GPU high-water mark" would be a
		# statement about another job.
		# ``budget_gb`` is 0 when no device memory could be detected (the CPU
		# backend reaches this line too).  Print "n/a" rather than divide.
	# Default: no transverse-channel ψ to surface to the caller.
	# ── Bispinor: fit ζ^{μ_L=1,2,3} on the current-density centroid set ──
	# Same kernel as the charge channel, swapping in the γ̃^i vertex.  The
	# automatic coupled schedule shares the expensive face transform; its
	# capacity fallback makes three sequential calls.  Output paths follow the
	# convention zeta_q_mu{1,2,3}.h5 next to zeta_q.h5.
	#
	# Loud-fail guard: if cfg.bispinor=True the transverse ζ fit MUST run,
	# otherwise downstream V_q silently falls back to scalar V_q and then
	# crashes on a full-BZ vs IBZ shape mismatch (ζ_T written by bispinor
	# mode is full-BZ; scalar V_q expects IBZ-only).  See the 2026-05-14
	# CrI3 30 Ry test-bed KNOWN_SANDBOX_ERRORS entry.  The
	# missing-centroids_file_current refusal itself now lives in the
	# bispinor PRE-FLIGHT above (before the charge fit), so this branch
	# gate is only reachable with the path present.
		# ``meta_T``, the centroid table and its μ padding were built ONCE
		# in the bispinor pre-flight (they are inputs to the ζ-reuse
		# decision, which runs before the charge fit).
		# Per-channel cache hygiene.  The 2026-05-04 bispinor branch needed
		# ``jax.clear_caches()`` here because the original ζ-fit cached
		# functions closed over tracers from the enclosing jit (the
		# UnexpectedTracerError surface).  After the 2026-05-08 open-spin
		# consolidation (commit ce28d50), the cached helpers only capture
		# static config (Mesh, shape, kgrid) — no tracer leaks remain.
		#
		# Keeping the surgical drop on ``_fit_one_rchunk_cache`` (whose
		# cache key includes ``id(psi_G_store)`` and would never hit
		# across channels anyway — drop = memory hygiene, not a workaround).
		# The pair-density caches are intentionally preserved so the three
		# transverse channels share the same n_rmu=n_rmu_current compile.
			# The services are process-global.  The coupled schedule serializes
			# this callback with the corresponding preparation/solve so a
			# deferred finding can never be attributed to the next current.
			# Stay silent on the overwhelmingly common empty path; the accepted
			# per-channel gate below still emits its canonical final banner.
					# Transverse ζ IBZ-write activates whenever the
					# bispinor V_q orchestrator iterates IBZ q's — same
					# gate the charge ζ uses,
					# resolved once in the pre-flight so the provenance
					# stamp and this call cannot disagree.
					# Orbit-closure of the transverse centroid set is
					# checked downstream in ``fit_zeta_to_h5``; failure
					# is loud per the bispinor IBZ requirement.
			# Stamp this ζ_T so a later run can reuse it — same ordering
			# and same truncating-knob veto as the charge stamp above
			# (fit → mark_zeta_done → stamp; a run killed between the
			# last two leaves a complete-but-unstamped file, which
			# ``_zeta_reuse_ok`` refits).  The μ_L loop was previously
			# unstamped altogether, which is why bispinor ζ from before
			# 2026-08-04 is never reusable.
					# Starting each successor only after its predecessor has
					# reached the r-loop keeps all preparation collectives in the
					# exact accepted μ order on every process.
```


### Zeta writer setup phase rulings (2026-09-06)

```text
    # P0 — entry of ζ-fit.  Captures the persistent state set up by
    # ``prepare_isdf_and_wavefunctions`` BEFORE ζ-fit starts: ψ at
    # centroids (full [b0, b4) band range, both Y and X transposes),
    # gflat_acc allocation will not have happened yet.  Forms the
    # planner's "Peak C const" baseline.  Round-1 addition.
    # Two μ extents flow through this function (see common/meta.py:38):
    # ``n_rmu`` is the LOGICAL centroid count from the centroid file;
    # ``n_rmu_padded`` rounds up to ``world_size = ∏ p_a`` so any
    # single- or product-axis sharding on the μ dim divides cleanly.
    # ψ is delivered at PADDED extent by ``load_centroids_band_chunked``
    # (Phase 3a) — pad rows zero — and stays there through the
    # in-memory pair-density / CCT chain.  The Cholesky in
    # ``factor_c_q`` slices internally to logical via the
    # ``n_rmu_logical=`` kwarg (Phase 3b-Cholesky) so the factorization
    # sees a non-singular matrix at its true extent.  zeta_q on disk
    # has logical extent (SlabIO clips the padded output against the
    # dataset's extent on write).
    # Dense factor/solve extent: the whole carrier when the run's packed
    # centroid order interleaves its pad slots per shard (identity on the pad
    # diagonal below keeps them inert), the logical prefix otherwise.
    # Band ranges for left and right wavefunctions.
    # Defaults here are (b0,b3) and (b0,b4); gw_jax typically passes (b0,b3) and (b1,b4).
    # The production charge fit uses asymmetric serving windows: L contains
    # all occupied states plus the Sigma conduction window, while R contains
    # the Sigma occupied window plus all empty states.  Complex conjugation
    # swaps those ordered endpoints, so LR alone is not a conjugation-closed
    # training space.  Complete the *normal equations* before factor/solve;
    # no fitted zeta, V, or W is projected downstream.  The q involution is
    # owned by the symmetry service and passed into neutral ``isdf.core``.
    # Full range for loading (max of left and right)
    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    # γ̃^0 = I_4 → vertex_mu_L=0 is the standard spin-traced path.  For
    # vertex_mu_L ∈ {1,2,3} the γ̃^μ vertex is folded into both P_l and
    # P_r so C_q is the proper per-channel interpolation metric for the
    # Lorentz pair density.  CCT^μ for transverse channels is Hermitian
    # indefinite and rank-deficient: TRS in non-magnetic ground states
    # gives near-null transverse-current modes that would be amplified
    # by 10^4–10^6 if we naively LU-solved through them (the original
    # MoS2 σ^B blowup).  The robust solver in :func:`solve_zeta`
    # uses an SVD pseudoinverse with rcond cutoff to drop those null
    # modes instead of inverting through them — the unique min-norm LSQ
    # solution.
    # ── Finalize write_ibz_only BEFORE any IBZ slicing (bug fix) ─────────
    # The IBZ cascade slices C_q/L_q to IBZ rows in STEP 2/3 below, and
    # slices Z_q to IBZ inside the per-r-chunk kernel; the two MUST agree.
    # The orbit-closure auto-fallback can flip write_ibz_only=False when the
    # centroid set isn't closed under the WFN sym group, so it must run HERE
    # — before the C_q slice.  (Previously it ran after factor_c_q, so the
    # charge channel sliced L_q to IBZ, then fell back, leaving L_q at IBZ
    # while Z_q stayed full-BZ → the ``B.shape[0]=nq_full != Nq=nq_ibz``
    # distributed-potrs crash.)  Transverse channels can't fall back (the
    # V_q orchestrator assumes IBZ ζ̃_T), so they loud-fail with a hint.
    #
    # ONE RESOLUTION POINT.  This used to call ``centroid_source_map_and_wrap``
    # for its side effect — build the whole table, throw it away, and read
    # the answer off whether an exception came back — which is both the
    # third spelling of the closure question in ``gw/`` and an
    # exception used as a boolean.  It now asks
    # ``gw.qgrid_symmetry`` the same question every other site asks, and
    # branches on the mode.  The charge channel's fallback line is the
    # SHARED announcement (deduped on the centroid set, so a run that also
    # falls back in V_q says it once, not twice); the transverse channel
    # suppresses it because its consequence is a refusal, not a fallback.
        # ``sym.sym_matrices`` holds the spatial ops; the fractional
        # translations live on WFNReader (BGW WFN.h5 layout).
        # ψ inputs at PADDED n_rmu (Phase 3a's load_centroids contract).
        # Monolithic shard_map pipeline: open-spin pair density + IFFT
        # + γ̃·γ̃ + FFT fused inside one shard_map.  The rank-5
        # P_l/P_r pair density never exists as a global XLA value, so
        # the rank-3 fused-replicated reshape that pegged the kernel
        # peak under the legacy chain cannot form.  γ̃^μ_L applied at
        # the post-IFFT contraction step (charge: identity short-
        # circuit; transverse: (perm, phase) tuple).  Output C_q is
        # rank-3 (k, μ, ν).
        # C_q: (nqx, nqy, nqz, n_rmu_padded, n_rmu_padded) with zero
        # pad rows/cols.
        # Flatten for Cholesky.  Reshape uses padded extent (the
        # in-memory shape); factor_c_q slices to logical
        # internally via ``n_rmu_logical=``.
            # Interleaved pad slots (orbit-packed order): C_q's pad rows and
            # columns are exact zeros.  Put C's own MEAN DIAGONAL (tr C/n per
            # q; the pad rows contribute nothing to the trace) on the pad
            # diagonal: the factor is nonsingular, Z's zero pad rows give
            # zeta_pad = 0, and the pad eigenvalues sit inside the active
            # spectrum -- a unit pad would BE lambda_max when C's scale is
            # small and the rank-truncation cut rcond*lambda_max would then
            # drop real modes (Si leg 20 attempt 1: 39 meV).
        # IBZ cascade for the per-q factor: slice C_q to IBZ rows *before*
        # ``factor_c_q`` runs so Cholesky / LU factors only ``n_q_ibz``
        # blocks instead of all ``n_q_full``.  C_q has the same (n_q, μ, ν)
        # shape as V_q, and Cholesky is per-q independent — slice-then-
        # factor gives bit-equal L_q rows as factor-then-slice.  The
        # downstream solve still produces ζ_q at IBZ, and V_q unfolds via
        # ``symmetry_maps.unfold_isdf_operator`` from IBZ → full BZ.  Same
        # slice helper applies to χ_q for the W_q = (1 − v_q χ_q)^{-1} v_q
        # path once that lands.
        # Commit only the selected q carrier at the outer host boundary.
        # The charge -q completion above still needs all K rows internally,
        # but waiting on C_full before this slice pinned d*(K-Q)*M^2/P bytes
        # through the synchronization point for no consumer.  The slice is a
        # pure row selection, so the selected values are unchanged.
        # Resolve once so the banner reflects what actually runs and
        # downstream callees skip their own 'auto' fallback.  Pass the
        # factor batch (nq = C_q_flat.shape[0], IBZ-sliced above when the
        # cascade fires) and the logical centroid count so the charge
        # resolver can pick the mesh-invariant replicated dense Cholesky for
        # fit-size stacks (see _resolve_solver_kind_charge).
        # ORDER MATTERS (capacity fix 2026-07-29, ladder notes R15.1).
        # The back-solve tier is resolved FIRST because when it is
        # ``distributed`` it REPLACES the charge factor wholesale a few lines
        # below, and the replicated factor's capacity check must therefore not
        # be enforced.  Enforcing it was capping mu at
        # sqrt(4 GiB / 16 B) = 16,384 on the size of a buffer the distributed
        # route never allocates.  ``_resolve_zeta_gather`` does not depend on
        # the factor kind, so the reorder is free.
            # The tier IS a charge-channel route: it replaces the whole
            # factor+back-solve pair, not just the gather granularity of
            # the replicated one.  Overriding here (and NOT inside
            # _resolve_solver_kind_charge) keeps the two knobs
            # single-purpose: `distributed_cholesky` picks the factor
            # LIBRARY, `distributed_zeta_solve` picks whether the factor is
            # ever replicated.  Refuses rather than downgrades if the user
            # also pinned an incompatible factor route.
                # Transverse rank_truncate family (the ONLY way the
                # transverse channel reaches this tier — the ridge
                # family's resolver returns per_q above): replace the
                # local eigh factor with the pzheevd 2D-sharded C⁺.
        # Preserve the fused path's exact ridge scalar for distributed LU.
        # Materializing this tiny (nq,) reduction before factor preparation
        # prevents XLA from choosing a different fused reduction tree whose
        # last-bit change is amplified by the near-null transverse modes.
        # per-q tile: the two structural all_gathers inside ``_per_q_block``
        # move μ²/p_y (row block) + μ² (full tile) — measured, not nominal.
            # Transverse: the factor stage is HOISTED (2026-08) — one
            # pivoted LU per q per CHANNEL instead of per r-chunk.
            # factor_c_q returns (factor, piv): (LU, perm) on the local
            # plan, or a distrib_la FactorToken (ipiv inside it) on either
            # distributed-library plan.
        # A distributed library factor is an OPAQUE token: no ``.shape``
        # to print and no single buffer to block on (a ScaLAPACK token
        # holds factors AND pivots).  Report what it does publish.
    # Pre-compute per-q trace of the CCT ONCE per channel — needed ONLY
    # by the remaining FUSED transverse route (cusolvermp passthrough,
    # piv None), whose per-r-chunk ridge is ``ε·|tr(CCT)|/n``.  On the
    # hoisted routes the ridge is baked into the factor, so the trace
    # (and its per-r-chunk all-reduce, 17 s of GPU stream at MoS2 3×3
    # bispinor) is gone.
        # ``not isinstance(...)`` is the ScaLAPACK hoist: its ridge is
        # baked into the factored matrix, so it needs no trace operand —
        # the condition used to read that off ``lu_piv is not None``, and
        # the pivots live in the token now.  Rank-truncate kinds also
        # return piv=None but carry no fused LU path — no ridge, hence no
        # trace operand (their L_q is C⁺, whose trace would be a
        # different, meaningless quantity here).
            # LOGICAL-block trace only: the identity pad block would
            # contribute exactly +mu_pad to the padded trace, making
            # the LU ridge (ε·|tr|/n) depend on the pad extent — i.e.
            # on the device count.  The slice is a no-op when the
            # extent is already logical.  solve_zeta divides by the
            # logical n to match.
    # Stack raw current Grams before their one factor, retaining its native pivot carrier.
    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget
        # IBZ fractional q-vectors for the G-flat accumulator (Phase C1b).
        # BGW wrap THEN divide by kgrid so the writer's per-q phase
        # matches the V_q kernel's ``apply_bloch_phase`` convention.
    # ---- G-flat on-disk format ---------------------------------
    # The writer accumulates each r-chunk's contribution into a
    # persistent G-flat buffer via
    # ``common.wfn_transforms.accumulate_rchunk_to_gflat`` and writes
    # the final tensor as ``zeta_q_G`` (shape
    # ``(n_q_disk, n_rmu, ngkmax)``).  The full r-space ζ_q is never
    # materialised on disk or as a persistent device buffer.  When
    # ``zeta_cutoff_ry`` is provided we build the per-q WFN.h5-style
    # sphere ``{G : |q+G|² ≤ cutoff}``, pad to a uniform ``ngkmax``
    # with the shared FFT-box pad sentinel (Nyquist-corner cell; the
    # Miller index is ``(-n/2)`` per even axis and ``+(n-1)/2`` per odd
    # one — ``common.gvec_fft_box.fft_box_pad_sentinel``), and
    # store both the coeffs and the per-q components on disk.  Without
    # a cutoff the writer falls back to the full flat-FFT axis
    # (n_G_sph = n_rtot) — slow disk path, kept for sanity checks.
        # Full-BZ q-vectors with BGW wrap, then / kgrid — the convention
        # the per-q sphere below is built in.  (It used to be stated as
        # "what the V_q
        # consumer's disk→G path expects"; that path,
        # ``zeta_loader._do_disk_to_G``, was deleted on 2026-08-07 — this
        # writer bakes the phase in and the reader does no FFT at all.)
    # Build the per-q WFN.h5-style sphere when a cutoff is available.
    # The output is host numpy; the writer threads ``sphere_idx_padded``
    # through ``accumulate_rchunk_to_gflat`` and stashes the components
    # / ngk / cutoff into the isdf_header below.  ``zeta_cutoff_ry``
    # — distinct from V_q's bare-Coulomb cutoff — defines the per-q
    # sphere on disk.  Caller (``gw_init.fit_zeta``) validates
    # ``zeta_cutoff_ry ≥ bare_coulomb_cutoff_ry`` so V_q has every G
    # it needs.
        # The Cartesian reciprocal ROWS off the vcoul door's geometry rather
        # than a hand-written ``blat * bvec``, which ``docs/services/vcoul.md``
        # names as an antipattern for a reason worth restating: a product
        # every caller has to remember to take is a footgun, and the day one
        # of them forgets, every G in this sphere is off by the lattice
        # constant with no shape error to say so.  ``from_wfn`` is duck-typed
        # on ``blat``/``bvec``/``cell_volume``; ``WfnLoader`` binds all three
        # off the mf_header.  Only ``.bvec`` is read here — this site takes no
        # Ω at all, so there is no ``cell_volume`` question to answer.
    # Centroid FFT-grid indices for the isdf_header.  ``centroid_indices``
    # may be a jax.Array on device; pull to host as int32 (n_rmu, 3).
        # WHICHEVER CALL CREATES THE INODE DECIDES THE STRIPE LAYOUT.
        # A Lustre layout is fixed at inode create, and the striping
        # hints live in the MPI_Info that phdf5's H5Fcreate builds — so
        # the inode has to be created by MPI-IO, and everything after
        # inherits what it chose.  (The old ``lfs setstripe`` prestripe
        # helper was deleted 2026-07-31: ``lfs`` is absent in the
        # production container, so the hints are the only lever left.)
        #
        # This used to be ``_replace_inode_for_write`` followed straight
        # by ``copy_mf_header(..., dst_mode='w')`` — i.e. rank-0 SERIAL
        # h5py created the inode, MPI-IO never saw it, and the file took
        # the DIRECTORY DEFAULT while the comment here claimed it got the
        # hints.  MEASURED 2026-08-07: zeta_q.h5 stripe_count=1 against a
        # resolved policy of 4 and a sibling isdf_tensors_*.h5 of 4
        # (``lfs getstripe -c -S``; /pscratch directory default is 1).
        # Cost at 1 rank is exactly zero, which is how it survived — it
        # bites at P>1, where ROMIO sets cb_nodes = min(stripe_count,
        # nranks), so a 1-stripe file pins collective buffering to a
        # SINGLE aggregator however many ranks are writing.
        #
        # So: SlabIO(mode='w') creates and closes the file collectively
        # (that H5Fcreate carries the striping hints), and only THEN does
        # rank-0 h5py append the two header groups with mode='a', which
        # keeps the inode it finds.  Serial h5py appending to a
        # parallel-HDF5-created file is fine — verified through both
        # readers by
        # tests/test_file_io.py::test_zeta_q_inode_gets_striping_policy.
        # Measured cost of the extra collective create+close on an empty
        # file: 4 ms at 1 rank.
        #
        # SlabIO mode='w' runs the same rank-0 unlink + barrier helper
        # (``_replace_inode_for_write``) internally, so the explicit call
        # that used to be here would be doing its job twice.
            # Append both header groups to the inode SlabIO just created.
    # ========== STEP 4b: SlabIO appends zeta_q to the pre-created file ==========
    # zeta_q is stored flat-q: shape (nq, n_rmu, n_rtot) with
    # q_flat = qx*nqy*nqz + qy*nqz + qz.  Flat-q is the ongoing
    # convention across LORRAX; see file_io.slab_io docs.  Chunk by
    # single-q r-slice so per-q reads stay contiguous.
    #
    # Single SlabIO handle reused for both create_dataset and all
    # writes — avoids the ~900 ms cost of a second collective
    # H5Fopen/close pair (measured 2026-04-18 at MoS2 3x3).  The same
    # handle serves BOTH backends: the allgather backend's handle is a
    # cheap rank-0 h5py file object, and routing its final write through
    # ``write_slab`` (instead of a hand-rolled gather + ``[...] =``)
    # applies the shared logical-extent clip — the bypass used
    # to write the PADDED gathered buffer into the logical-shaped
    # dataset and crashed whenever a μ pad existed (PADDING_AUDIT #2).
    #
    # mode='a' (not 'w') so the pre-written mf_header + isdf_header
    # are preserved.  SlabIO's mode='w' inode replace is skipped on 'a'
    # — the inode was already replaced above.
    #
    # Dataset layout ``(nq, n_rtot, n_rmu)`` — NOT ``(nq, n_rmu, n_rtot)``.
    # Rationale: per-r-chunk writes span the full innermost axis (n_rmu)
    # under this layout, so each ``(q, r)`` row is contiguous on disk.
    # Under the old ``(nq, n_rmu, n_rtot)`` layout we'd write n_rchunk <
    # n_rtot on the innermost axis, producing 480K × 1920-B scattered
    # strips per rank per write (measured at 0.18 GB/s on Perlmutter
    # pscratch, 8× slower than contiguous).  Per-q reads (V_q) stay
    # contiguous under this layout too: a 6.6 M-element slab at
    # ``(q, 0, 0)`` is a single contiguous block.  Downstream V_q
    # transposes the returned array on GPU to match the kernel's
    # (n_rmu, n_rtot) expectation — ~50 µs per q, negligible.
        # G-flat layout: ``zeta_q_G`` dataset (n_q_disk, n_rmu, ngkmax)
        # — WFN.h5 ``wfns/coeffs`` style with a fixed ``ngkmax`` padded
        # G axis.  Per-q components live in
        # ``isdf_header/gvec_components`` (already serialised by the
        # write_isdf_header call above).  The row-major axis order keeps one
        # full q slab contiguous without requesting an HDF5 chunk layout.
    # The physics window remains exactly [_bfs, _bfe): masks below use the
    # logical endpoints.  Only the ψ(G) transport range is padded so every
    # per-chunk band shard is P-divisible.  The loader and the pair masks
    # zero/ignore these at-most-P-1 tail bands.
    # Build the host-resident ψ(G) staging store.  It supplies one band
    # chunk at a time while the all-P-sharded ψ(r) cache is built below,
    # then its host tiles are released before the r-chunk loop.
        # Parent route: the r-chunk kernel contracts on the WFN's raw rows
        # and never needs a full-k ψ(G) or ψ(r); the store is 1/(nk/n_parent)
        # of its full-k size and so is the hoisted ψ(r) cache below.
    # The parent route streams orbit-closed real-grid TILES, not contiguous
    # slabs; every tile has one static width <= the planner's chunk_r, so the
    # priced r extent still bounds each Z_q/solve transient.
        # A tile holds whole real-space orbits on each Y owner, so its
        # width is bounded below by n_y x the largest orbit.  When the
        # planner's chunk_r sits under that floor the tiles are WIDER
        # than what was priced: say so, loudly, rather than OOM in
        # silence (planner floor: KNOWN_LORRAX_ISSUES 2026-09-05).
    # Hoist the full-grid ψ(G)->ψ(r) transforms when the memory plan admits
    # the cache.  Otherwise retain the SAME host staging store and let the
    # canonical z_q kernel stream one band chunk through to_rchunk_inner per
    # r chunk.  This is a storage policy only; both routes enter the same
    # pair-density/FFT/solve physics below.
        # Every io_callback has drained.  Release the host coefficient tiles,
        # but retain the store-owned box index and k vectors consumed by the
        # parent kernel metadata until the final close below.
```


### Zeta writer tile-loop phase rulings (2026-09-06)

```text
    # ========== STEP 3: Compute L_q from CCT ==========
    # μ_L=0 (charge): C_q is PSD → 2D-blocked Cholesky factor L_q.
    # μ_L=1,2,3 (transverse): C_q is Hermitian indefinite — skip the
    # factorization and pass the slice through; the per-chunk
    # solve_zeta dispatches to an SVD pseudoinverse with
    # rcond cutoff (drops null transverse-current modes that would
    # otherwise be amplified by 10^4–10^6).
    # ========== STEP 4a: q-IBZ reduction + header writes (rank 0) ==========
    # When ``write_ibz_only=True`` (default), ζ is written for IBZ q's
    # only.  V_q at the full BZ is recovered by the reader / V_q
    # orchestrator using sym data from ``mf_header`` (see report.md
    # §2.4).  The on-disk ``zeta_q`` leading axis is ``n_q_disk``
    # rather than ``n_q_full``; the chunk loop slices
    # ``zeta_chunk[q_irr_full_idx]`` before writing.
    #
    # When ``write_ibz_only=False`` (caller forced full-BZ writes via
    # an unreduced q grid), the full-BZ axis is preserved on
    # disk for back-compatibility.
    #
    # ``write_ibz_only`` was finalized above (before the C_q/L_q IBZ slice)
    # by the orbit-closure auto-fallback, so the on-disk q-axis is IBZ when
    # it is True and full-BZ when it fell back — nothing more to decide here.
    # BGW Brillouin-zone wrap: ``q > kgrid/2 → q − kgrid``.  The writer
    # and V_q consumer share the symmetry service's exact tie convention so
    # the per-q phase
    # ``exp(-2πi (q/kgrid)·r)`` baked into the G-flat output is the
    # convention the consumer expects.
    # ``zeta_q.h5`` carries the BGW-style ``mf_header`` verbatim from
    # the source WFN so any downstream consumer (the new
    # :class:`zeta_loader.ZetaLoader`, or anything else that
    # speaks the WFN.h5 header) sees the same crystal / k-grid / G-grid
    # / symmetry view.  ``isdf_header`` holds ζ-specific metadata only
    # — centroids in FFT-grid + fractional coords, density label,
    # ``vertex_mu_L``.  Everything sym-derivable (q-IBZ list, centroid
    # orbit permutation, G-sphere) is rebuilt at read time via
    # ``SymMaps`` + ``orbit_syms`` and is *not* stored.
    #
    # Sequence: SlabIO(mode='w') replaces the inode (rank-0 unlink +
    # barrier) and CREATES it collectively, so H5Fcreate applies the
    # Lustre striping hints; it closes immediately.  Rank 0 then appends
    # both header groups with h5py mode='a'.  Then SlabIO re-opens with
    # mode='a' so the headers survive and ``create_dataset('zeta_q')``
    # appends rather than truncates.  The create-order matters and is
    # explained at the write_headers section below.
    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    # Uniform band chunks over [b_full_start, b_full_end]: N-1 of
    # size ``band_chunk_size`` plus one remainder chunk.  This gives
    # the read/FFT pipeline and the pair-density einsum exactly
    # TWO compile shapes, regardless of where the L/R endpoints fall.
    # Chunks that straddle an L/R endpoint get handled in the loop
    # below by padding the left-side ``psi_L_bc`` slice with zero
    # bands — the resulting einsum still runs at the uniform
    # ``bc_size``, so it hits the same JIT cache.
    # ========== STEP 6: Loop over chunks ==========
    # Wall-clock totals for the end-of-fit timing line.  ``t_fit_total``
    # covers the fused fit_one_rchunk jit (load + pair + ZCT + solve) —
    # finer-grained breakdown now lives inside the jit and is only
    # observable via xprof, not perf_counter.
    # Driver debug — runtime probe of process-wide HBM at
    # named lifecycle sites.  The module-level ``mem_probe`` helper is
    # reused so the r-chunk loop sites and the gw_init V_q sites all
    # share one source of truth.  HLO's buffer-assignment.txt is per-jit
    # and cannot prove cross-jit liveness — see
    # reports/memory_model_refit_2026-05-17/agent_e_cross_jit_lifetime.md.
    # Per-chunk: ``accumulate_rchunk_to_gflat`` adds the chunk's
    # contribution into the donated ``gflat_acc`` in place; no
    # per-chunk SlabIO write.  The single ``zeta_q_G`` write happens
    # once after the loop.
    # GPU high-water tracker — the all-time ``peak_bytes_in_use``, which is what
    # actually determines OOM (it includes JIT caches + prior-stage allocations,
    # not just the chunk-loop arrays).  Prefer JAX's exact per-rank BFC-arena peak
    # from ``memory_stats()``; fall back to THIS rank's nvidia-smi sample only if
    # that's unavailable.  Two traps this avoids: (1) ``--id=0`` reads a *foreign*
    # GPU on a multi-rank / shared node (``_nvsmi_used_mb_local_gpu`` honours
    # CUDA_VISIBLE_DEVICES); (2) a single post-loop nvidia-smi sample MISSES the
    # peak under the cudaMallocAsync allocator (freed transients already returned),
    # so it is only a last-resort floor, never the reported number when stats work.
    # ---- G-flat accumulator (zero-init, μ-sharded) ----
    # Persistent buffer: (n_q_disk, n_rmu_padded, ngkmax) c128 with
    # μ sharded across ('x', 'y') so each rank holds n_rmu/p per q.
    # Donated to ``accumulate_rchunk_to_gflat`` each iter; in-place add.
    # When the per-q sphere isn't available (no vcoul_cutoff_ry) we
    # fall back to the full flat-FFT axis n_rtot — slow, kept for
    # smoke / sanity tests.
    # μ allocated at PADDED extent so the ('x','y') sharding divides
    # cleanly.  Pad rows are zero because the back-solve produces
    # zeta_pad = 0 (L_q's pad block is identity).
    # Flat-axis chunking inside ``accumulate_rchunk_to_gflat``.  The
    # kernel runs inside a ``shard_map`` over ``('x','y')`` and chunks
    # the per-rank flat ``(n_q · n_mu_local)`` axis into rows-per-
    # scan-iteration of ``chunk_size``.  Memory bound:
    # ``chunk_size · n_rtot · 16 B`` for the per-iteration FFT box.
    #
    # ``gflat_chunk_size = 0`` ⇒ one-shot (fine when the full per-rank
    # box ``N · n_rtot · 16 B`` fits; MoS2 3×3 at 4 ranks: 1.1 GB).
    # For CrI3-class FFT grids set cohsex.in ``gflat_chunk_size`` to
    # an integer; the kernel zero-pads N up to a multiple of the chunk
    # size so any value works (no divisibility constraint on either
    # n_q or n_mu_local).
    # Numpy → replicated, process-locally: ``jax.device_put(numpy, <a
    # multi-process NamedSharding>)`` fires JAX's hidden
    # ``multihost_utils.assert_equal`` all-gather (see
    # ``common.collectives.device_put_process_local``).
    # P1 — pre r-chunk loop, after L_q computed AND gflat_acc allocated.
    # This is the persistent baseline the planner's ``_peak_C_const``
    # should match: centroids (ψ_l/ψ_r in both Y and X transposes), L_q
    # (Cholesky factor at IBZ for charge / pass-through CCT for
    # transverse), and the freshly-zeroed gflat_acc.  Round-1 addition.
            # release_channels() has materialized and synchronized the one
            # concatenated carrier.  Drop each channel's superseded factor
            # and trace before entering the memory-dominant face loop.
    # glibc heap trim hook (see the comment at the call site in the loop
    # below).  DEFAULT ON — one ``malloc_trim(0)`` per r-chunk costs a few
    # ms and is the second half of the workstream-T ramp cure (the first
    # half is ``runtime.tune_glibc_malloc``, applied before ``import jax``).
    # ``LORRAX_MALLOC_TRIM=0`` disables; ``malloc_trim`` is glibc-only, so
    # a missing symbol just disables the hook.
    #
    # The parse used to be a case-SENSITIVE ``not in ("0", "off", "false")``,
    # which missed ``""``, ``"no"`` and every uppercase spelling — so
    # ``LORRAX_MALLOC_TRIM=OFF`` left the hook ON while its documented
    # sibling ``LORRAX_MALLOC_TUNE=OFF`` (runtime._env_falsy) correctly
    # turned OFF.  The two knobs are advertised together in
    # docs/dev/env_vars.md:116-117 and now answer identically.
            # Each pad slot gets its OWN out-of-range sentinel so the
            # scatter's unique-index promise holds with mode='drop'.
            # 6e. IBZ-slice → allgather (or FFI) → HDF5 write.
            # ``zeta_chunk`` is computed at full BZ q (the FFT in
            # ``solve_zeta`` naturally outputs all q's).  We slice to
            # Phase B: ``zeta_chunk`` is already IBZ-shape
            # (n_q_disk, n_rmu, n_rchunk) — the gather happens inside
            # ``fit_one_rchunk`` before the triangular solve.  In
            # full-BZ mode (q_irr_full_idx=None) the kernel returns
            # full-BZ shape.  Accumulate this r-chunk's contribution
            # into ``gflat_acc`` in place; the full ``zeta_q_G`` is
            # written once after the loop.
            # Return glibc's free-but-still-mapped heap to the OS at the
            # end of each r-chunk.  Together with
            # ``runtime.tune_glibc_malloc`` this is the cure for the
            # workstream-T per-r-chunk anonymous-memory ramp: XLA:CPU's
            # transients are ordinary malloc/free, and the memory glibc
            # keeps in its per-thread arenas after ``free`` is what
            # ratchets RSS up chunk after chunk while
            # ``jax.live_arrays()`` stays flat.  MEASURED at MoS2 12x12 /
            # 606c / P=80 / 81 r-chunks: +0.35 GB/rank/chunk -> 0.00.
                # ``rss`` is the leak observable: on CPU the XLA arena is
                # invisible to ``memory_stats()``, so per-chunk anonymous
                # growth only shows up here.  ``live`` is the JAX-side
                # array total — rss growing while live stays flat means
                # the accumulation is NOT a retained jax.Array.
            # LORRAX_MAX_RCHUNKS=N: stop the r-chunk loop after N chunks
            # for profiling/sweeping.  Clean python exit avoids the
            # SLURM step-zombie issue you get from killing the python
            # mid-run.  Off when unset.
            #
            # STRICT PARSE.  The old ``if _max_rchunks and ... >= int(...)``
            # took any non-empty string as a request: ``=0`` — the natural
            # spelling of "no limit" — is a truthy STRING whose int is 0, so
            # ``(chunk_idx+1) >= 0`` fired on the very first chunk and
            # truncated the fit to one r-chunk.  A non-numeric value raised a
            # bare ``invalid literal for int()`` from inside the loop.  Both
            # produce a PARTIAL ζ that the writer still marks complete, so
            # they must refuse up front instead.
    # Sample GPU memory ONCE after the last chunk's jit settles.  The
    # allocator keeps the peak reservation so this reads close to the
    # all-time high water.
    # ---- Write the accumulated G-flat ζ_q ----
    # One collective write of the persistent ``(n_q_disk, n_rmu,
    # ngkmax)`` tensor to disk.
        # Pad slot zero-fill (WFN.h5 ``coeffs = 0`` convention).  The
        # per-q gather inside ``accumulate_rchunk_to_gflat`` read the
        # FFT-box pad sentinel's flat slot into every
        # pad position; those values are physical (not zero) so we
        # mask them here.  Logical slots ``[..., :ngk[q]]`` carry the
        # real coeffs and are untouched.
        # On-disk extent is LOGICAL n_rmu in the CANONICAL centroid order
        # (stated once, at the ``create_dataset`` above).  The buffer is in
        # the run's packed order: convert at this seam (one exchange), then
        # SlabIO clips the trailing canonical pad rows against the dataset's
        # own extent, identically on every backend.
    # Flip ``isdf_header/zeta_is_done`` to True now that every chunk
    # has drained to disk.  Restart paths key off this flag to decide
    # whether the on-disk ζ is trustable; flipping it here (after the
    # global sync above) guarantees every rank's writes are durable.
    #
    # NOT when a truncating knob was in force.  ``LORRAX_MAX_RCHUNKS=N``
    # breaks the r-chunk loop above after N chunks, so the tensor written
    # a few lines up is a PARTIAL ζ — and this stamp is the file's own
    # claim about itself.  ``gw_init`` already refuses to add
    # ``fit_provenance`` in that case, which blocks REUSE; this stops the
    # file lying in the first place.  The two guards are deliberately
    # independent: provenance is about "may a later run reuse this", and
    # ``zeta_is_done`` is about "did the writer finish", which it did not.
    # Both read the one list in ``gw_config.ZETA_TRUNCATING_ENV_KNOBS``.
    # Full teardown after the last cached kernel has completed.  The phdf5
    # reader itself is cached at module level and survives.
    # Per-stage timing breakdown.  ``fit`` is the fused fit_one_rchunk jit;
    # ``H5`` is the allgather+write (or FFI write_slab).  Everything else
    # lives inside the jit — see xprof for the intra-jit breakdown.
    # P3 — exit of ζ-fit.  Captures what's still alive after the chunk
    # loop completes: gflat_acc was del'd above, zeta_chunk freed, but
    # centroids (psi_l/psi_r) and L_q are still referenced by the
    # caller's closure (they were passed in as args).  V_q runs next
    # against this baseline.  Round-1 addition.
    # Return only peak-memory high-water mark; centroid wavefunctions
    # are not returned (see docstring — callers re-load them directly
    # via ``load_centroids_band_chunked``).
```


## Source contracts relocated during the 2026-09-06 compaction

### `src/isdf/core.py` — `<module>`

ISDF core primitives: ψ + centroids -> ζ interpolation vectors.

Neutral array-in / array-out core of the ISDF fit — the composable phases
``c_q_from_psi_sm`` -> ``factor_c_q`` -> ``fit_one_rchunk`` (which fuses
``z_q_from_psi_sm`` + ``solve_zeta``) plus the q=0 Gram building blocks used
by centroid selection.  Depends only on ``common/`` (Meta, timing,
gamma_matrices, fft_helpers, wfn_transforms, psi_G_store) and on the
``distrib_la`` service door (every distributed factor and solve, including
the 2-D blocked Cholesky, which is its ``native2d`` backend).  NO ``gw`` /
LorraxConfig / h5 / V_q packaging lives here — GW and BSE are consumers.

### `src/isdf/core.py` — `host_rss_gb`

This process's resident set size in GB, from ``/proc/self/status``.

The CPU backend returns ``None`` from ``device.memory_stats()``, so
on a CPU mesh the ONLY faithful per-rank memory observable is the
kernel's own RSS accounting.  Cheap (one small read, no JAX calls) —
safe to sample inside the r-chunk loop.  Returns -1.0 where
``/proc`` is unavailable.

### `src/isdf/core.py` — `complete_ordered_pair_normal_equations`

Complete an LR normal equation to the conjugation-closed LR+RL set.

For a Hermitian charge vertex, relabelling ``(n,m,k)`` gives exactly

``N_RL(q) = conj(N_LR(-q))``

for both the CCT metric and every ZCT right-hand-side chunk.  Therefore
this one operation is the normal equation of the concatenated ordered
pair training domain; it is not a projection of fitted zeta, V, or W.
The q permutation is supplied by the symmetry service so this neutral
ISDF layer does not own a second q-grid convention.

### `src/isdf/core.py` — `_conv_kpair_static_gamma`

Host-stable monomial data for the conv_kpair attribute ABI.

The existing XLA arm keeps gamma arrays as runtime operands.  The FFI ABI
uses attributes because every channel's monomial is invariant across the
whole compiled C/Z kernel; including the values in the caller cache key
prevents one Lorentz channel from reusing another's executable.

### `src/isdf/core.py` — `pair_density`

Open-spin pair density P_k,ab(μ, col) = Σ_n ψ*_{n,k,a}(μ) ψ_{n,k,b}(col).

Spin axes (a,b) are kept open; γ̃ is applied downstream at the C_q
or Z_q post-IFFT reduction step.

Inputs:
    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)
    psi_rcol_Y: (nk, nb, ns, n_col) with P(None, None, None, 'y')

Output:
    P_k_ab: (nk, ns, ns, n_rmu, n_col) with P(None, None, None, 'x', 'y')

einsum: ``'kmna,knbr->kabmr'``.

### `src/isdf/core.py` — `pair_density_aot_peak_bytes`

Per-rank compiled peak for the canonical :func:`pair_density`.

This is a planning view of the SAME cached JIT production calls.  It does
not carry a modelling-only einsum: shapes and shardings are passed to the
canonical factory above, then the shared AOT memory service reads XLA's
buffer assignment.  There is no FFT in this kernel, so the service's
cuFFT-workspace term is exactly zero.

### `src/isdf/core.py` — `gram_q0_from_pair`

q=0 valence-conduction pair-product Gram from open-spin pair densities.

``symmetrize=False`` skips the final Hermitian symmetrization (which
requires a SQUARE G) — used by the tiled Gram build in
:mod:`centroid.pivoted_cholesky`, which assembles rectangular edge tiles
and applies the identical 0.5·(G+G^H) once on the full matrix.

Mathematically (q=0 special case of the CCT-over-k structure):

    G(μ,ν) = Σ_k w_k · [Σ_{αβα'β'} γ̃^{μ_L}_{αα'} γ̃^{ν_L}_{ββ'}
                          · P_v_{αβ}(μ,ν;k)*  · P_c_{α'β'}(μ,ν;k)]

γ̃ identity short-circuit: pass ``gamma_L=None`` (and/or
``gamma_R=None``) for charge / left-only / right-only sides.
Both None → Σ_{αβ} P_v* · P_c, the historical pivoted-Cholesky
candidate Gram in open-spin form.  γ̃^μ is monomial — each non-
identity contraction is one ``jnp.take`` + element-wise phase
multiply, not a 4×4 matmul.

Used by :mod:`centroid.pivoted_cholesky`.

Args:
        P_v_k: (nk, ns, ns, n_rows, n_cols) complex, valence open-spin pair
                density (output of :func:`pair_density` on the valence band
                window), sharded ``P(None, None, None, 'x', 'y')``.
        P_c_k: (nk, ns, ns, n_rows, n_cols) complex, conduction window,
                same layout.
        k_weights: (nk,) real, k-point weights (IBZ weights summing to 1,
                or 1/nk_tot for each full-BZ k-point).
        gamma_L, gamma_R: ``(perm, phase)`` tuples or ``None`` (=identity).
        mesh_xy: ('x','y') device mesh, same as the pair densities.

Returns:
        G: (n_rows, n_cols) complex, sharded ``P('x','y')``. Hermitian PSD
                when square and ``symmetrize=True``.

### `src/isdf/core.py` — `transverse_gram_q0_from_pair`

PSD q=0 Gram of the three stacked transverse transition features.

For ``Z_i(a,mn,k) = <psi^R_m(a)|gamma_i|psi^L_n(a)>`` this computes

``G_perp(a,b) = sum_{i=1}^3 sum_{nmk} w_k Z_i(a,mn,k) conj(Z_i(b,mn,k))``.

Since every transverse gamma is Hermitian, this ``Z_i`` is the conjugate
of ``<psi^L|gamma_i|psi^R>``; the pair-density factorisation therefore
uses ``gamma_i^*`` on its first endpoint and ``gamma_i`` on its second.
Components have equal weight and are never normalised separately, so an
orthogonal rotation among the three Cartesian current components leaves
the Gram invariant.  The computation
reuses :func:`gamma_double_contract` and scans one component at a time;
neither band-pair features nor a three-component pair-density stack is
materialised.

### `src/isdf/core.py` — `gram_q0_from_psi_sm`

Fused candidate Gram from two left/right centroid-WFN faces.

Inputs use the same single-axis face convention as
:func:`c_q_from_psi_sm`'s legacy route.  The two band contractions and
the q=0 normal-matrix fold are one compiled program, so their rank-5
pair densities are compiler-internal temporaries rather than committed
outputs of separate dispatches.

``gamma_mode='charge'`` computes the scalar q=0 CCT normal matrix.
``gamma_mode='transverse'`` requires four-component bispinors and computes
the PSD sum ``sum_i Z_i Z_i^H`` used by centroid selection; it is not the
individual indefinite transverse ``C_q^i`` used by the zeta solve.

### `src/isdf/core.py` — `_gram_q0_tiled_from_psi_kernel`

Return one donated executable for a complete tiled q=0 Gram build.

The manual shard-map body owns each rank's complete WFN faces and local
``P('x','y')`` Gram shard.  Its scan walks the caller's fixed square-tile
schedule in column-major order, matching the historical Python loop.  Thus
the four face slices, two pair densities, Gram fold and local insertion are
compiler-internal to one dispatch; only the full WFN faces and the donated
Gram persist across tiles.

### `src/isdf/core.py` — `gram_q0_tiled_from_psi_sm`

Assemble every q=0 candidate-Gram tile in one donated executable.

This is the blocked counterpart of :func:`gram_q0_from_psi_sm`.  ``G_xy``
is a square destination sharded ``P('x','y')`` and is donated.  The WFN
faces keep their full candidate extent and their canonical X/Y layouts;
each scan step slices only the already-owned local shards.  The final
Hermitian fold remains the caller's operation, exactly as in the historical
blocked schedule.

``tile_width`` is the existing global square-tile width, not a tuning
choice made here.  It must divide both mesh axes.  Tail tiles are padded
with exact zeros locally; the final partial row and column are trimmed to
their static in-range shapes before the contiguous destination update.
The transverse route calls the unchanged three-component scan in
:func:`_gram_q0_fold_local`, preserving its component and reduction order.

### `src/isdf/core.py` — `gram_q0_tiled_from_psi_aot_resident_increment_bytes`

Compiled bytes above the already-resident WFN faces and donated G.

This lowers the exact production scan executable.  The caller's live-set
model already counts all arguments (the four complete WFN faces and the
local ``P('x','y')`` destination), so ``resident_increment`` is the relevant
compiler fact: temporary bytes plus any non-aliased output bytes.  Donation
should make the latter zero; the focused P=4 HLO gate asserts the alias.

### `src/isdf/core.py` — `build_psi_r_cache_sm`

Hoist all ψ(G)->ψ(r) transforms out of the outer r-chunk loop.

The returned global array has shape
``(n_bc, nk, bpd_max*P, ns, n_rtot)`` and sharding
``P(None, None, ('x','y'), None, None)``.  Thus every cached coefficient
is owned by exactly one rank; neither the full band window nor an r slab is
replicated.  The leading chunk axis preserves the store's uniform static
shape, including zero pad rows in its last chunk.  This is deliberate: a
ragged final item cannot be the output of the same ``lax.scan`` and would
create a second compiled cache/slice family.  For the 50-band Si window,
bc16 therefore carries 64 slots (28% pad; priced exactly by the planner).

### `src/isdf/core.py` — `_band_chunk_compaction`

Per-chunk Y-side compaction table (and whether it is the identity).

After ``all_to_all('y')`` + ``all_gather('x')`` every rank holds all P
ranks' ``bpd_max`` band slots of one chunk; a chunk narrower than the
uniform carrier leaves its real bands at stride ``bpd_max``.  The table
maps the compact global band position back to that slot.

### `src/isdf/core.py` — `_identity_pad_block_diagonal`

Add identity to the pad-block diagonal of a square N_μ² matrix.

``M`` has shape ``(nq, n_rmu, n_rmu)`` at PADDED μ extent with
zero pad rows/cols (the Phase 3a contract: bilinear in zero-padded
ψ ⇒ M's pad rows/cols are exact zeros).  This helper adds 1 to the
diagonal entries in positions ``[n_rmu_logical, n_rmu)``, leaving
the logical block exactly intact.  Result: ``M_id_pad =
block_diag(M_log, I_pad)`` — block-diagonal with the input's
logical block on top-left and identity on bottom-right.

Why this matters — and the limits of the guarantee:  In EXACT
arithmetic, Cholesky and LU on the identity-padded matrix produce
factorisations whose logical block equals the factorisation of the
un-padded logical-only matrix (the recursions never read across
the zero off-diagonal pad blocks, and ``√1 = 1`` exactly), and the
back-solve with zero-pad-row ``Z`` gives ``y_pad = 0`` with the
logical solve unchanged.  In FLOATING POINT the guarantee is only
approximate, because blocked/tiled implementations regroup partial
sums when the matrix extent changes:

* **Cholesky (charge channel): holds to ≤1e-7 rel** in practice
  (measured ζ_C 5.5e-8 under a pad-extent flip at fixed P; the
  well-conditioned PSD CCT does not amplify the regrouping noise).
* **LU on the near-singular indefinite transverse CCT: does NOT
  hold.**  Shape-dependent LU roundoff is amplified O(1) in the
  near-null modes — each pad extent yields a different,
  per-extent-deterministic ζ_T, with catastrophic resonances at
  some extents (MoS2 668→672: Σ^B tile(2,2) −0.15 → −117.9 eV).
  See ``reports/device_invariance_2026-07-08/ROOT_CAUSE.md``.
  For this reason :func:`solve_zeta` slices the indefinite solve
  back to the LOGICAL extent — the identity pad added here is only
  a non-singularity safety net for the padded buffer, never the
  extent the transverse system is actually solved at.

This is NOT ridge regularisation on C_q (which would corrupt the
logical block).  The identity is added ONLY to the pad-block
diagonal; the logical block is untouched.

Output sharding is ``P(None, 'x', 'y')`` (n_rmu_padded is
mesh-divisible by construction so single-axis sharding on each
μ-dim works at any padded extent).  When ``n_rmu_logical ==
n_rmu`` (no pad), the function is a no-op pass-through with the
sharding constraint reapplied.

### `src/isdf/core.py` — `_replicate_charge_ok`

True when the charge CCT stack ``(nq, n_μ, n_μ)`` c128 fits under the
replication cap — the criterion for the mesh-invariant dense Cholesky
over the grid-dependent distributed cuSolverMp potrf.

Requires both ``nq`` and ``n_rmu`` (the ζ-fit caller passes them from
``C_q.shape[0]`` and ``meta.n_rmu``); ``None`` — direct callers that
don't supply them — keeps the legacy distributed policy so nothing off
the GW ζ-fit path changes behaviour.

### `src/isdf/core.py` — `_replicate_rank_truncate_ok`

True when the rank-truncating charge factor can run replicated.

DIFFERENT CRITERION from :func:`_replicate_charge_ok`, deliberately.
That one gates the *Cholesky* route on the whole ``(nq, μ, μ)`` stack,
which is the right question there.  It is the WRONG question for
``rank_truncate``: :func:`factor_c_q_replicated_batched` already splits
the q axis at its own ``_REPLICATED_FACTOR_MAX_BATCH_BYTES`` bound, so
the replicated transient is ONE q-batch (≤ that bound, plus the eigh's
own workspace) and is FLAT IN nq — it does not grow with the stack.
Testing the stack made the resolver refuse fits that comfortably fit,
e.g. MoS2 12×12 full-BZ (nq=144, μ=2412 → 13.4 GiB stack, but only
~4 GiB replicated at a time), and refusing means losing the §6a
rank-truncation physics cure rather than losing memory.

This can only make the production-default ``rank_truncate`` route
REACHABLE where it previously raised; it never changes a route that
resolves today, and it does not touch the ``cholesky`` branch at all.

NOTE (the real μ ceiling on this route): memory is not what breaks
here.  The factor is a dense whole-tile ``eigh`` per q (~5.5 h at
μ=4k, ~86 h at μ=10k on 28 cores for the FULL nq sweep).  Since
2026-08-01 the plan executes q-parallel above the fold threshold
(:func:`_factor_c_q_replicated_qparallel` — per-rank cost
ceil(nq/P)·μ³, bits unchanged), which divides those walls by
min(P, nq) but cannot touch the SINGLE-q eigh: past ~4k centroids the
route still needs a genuinely distributed eigh (SLATE/ScaLAPACK via
``distrib_la``; cuSOLVERMp is out on a rectangular mesh), not a
bigger cap.

### `src/isdf/core.py` — `_rank_truncate_capacity_error`

THE refusal for a replicated rank-truncating eigh that will not fit.

ONE message for both channels.  The charge branch
(``charge_zeta_solve='rank_truncate'``) and the transverse branch
(``transverse_zeta_solve='rank_truncate'``) allocate the *same* object
— one replicated ``(q_batch, n_mu, n_mu)`` complex128 eigh operand — so
they have the same ceiling and must report it the same way.  Before
2026-08-22 only the charge branch checked it at all; the transverse
resolver returned ``'transverse_rank_truncate'`` unconditionally and
the run died on an allocation, hours in, above ``n_mu_T ~ 16k``
(register: "transverse resolver lacks the charge branch's capacity
gate; OOMs late above mu_T~16k").

REPORT THE QUANTITY THAT ACTUALLY FAILED (DLM campaign 2026-07-29,
jobs 7879700 / 7879689).  Two gates test DIFFERENT things:

    _replicate_charge_ok           whole stack   nq * mu^2 * 16
    _replicate_rank_truncate_ok    one q-batch   batch * mu^2 * 16

The second is the weaker one, so IT is what binds, and the cap that
would clear it is the per-batch figure — not the stack.  The message
this replaced quoted the stack and advised the stack-sized cap (61 /
94 GiB at the two sizes measured), overstating the fix by ~10x: 6 / 10
GiB is what those runs actually needed.

### `src/isdf/core.py` — `_resolve_channel_ladder`

The mesh/CPU/backend decision ladder SHARED by the per-channel
ζ-fit solver resolvers (:func:`_resolve_solver_kind_charge`,
:func:`_resolve_solver_kind_transverse`) — written once so the two
channels cannot drift.

Ladder (identical for both channels):

  * ``override='off'``                 → ``kind_fallback``.
  * ``override`` in ``explicit``       → that handler decides (called
    with ``(px, py)``; owns its own FFI-availability / mesh-geometry
    checks and may raise).  Both channels route EXPLICIT
    ``'cusolvermp'`` (legacy alias ``'on'``) through the distrib_la
    door — platform, compiled-capability, process-coverage and
    true-2D geometry guards — exactly like 'slate'/'scalapack'.  The
    old inline shortcut (``kind_cusolvermp if is_2d else
    kind_fallback``) silently demoted an explicit request on a 1-D
    mesh AND skipped every capability probe, so resolve could promise
    a handler the mesh/build couldn't run (doctrine 3 / quality
    pattern #6; audit fix/zq 2026-07-28).
  * auto (or unrecognised): ``auto_pre()`` first when given (the charge
    channel's replication-cap branch; returns a kind, raises, or
    returns ``None`` to fall through), then ``kind_cusolvermp`` on true
    2D non-CPU meshes (cuSOLVERMp is CUDA-only — never auto-picked on
    a CPU mesh), else ``kind_fallback``.

### `src/isdf/core.py` — `_resolve_solver_kind_charge`

Pick the charge-channel ζ-fit solver: fully-replicated dense
Cholesky (mesh-invariant, the default for fit-size tiles) vs the
distributed cuSolverMp potrf+potrs vs the in-tree shard_map 2D-blocked
Cholesky + per-q triangular solve.

Default policy (2026-07-20): **replicated dense Cholesky** whenever the
CCT stack fits on one device (:func:`_replicate_charge_ok`).  The
distributed cuSolverMp potrf is block-cyclic — its partial-sum
regrouping depends on the process grid ``(px, py)`` — so at large,
mildly rank-deficient n_μ (MoS2 6×6, 1600 centroids) the factor drifts
~0.3% between a 2×2 and a 4×4 grid, and the GN-PPM pole construction
amplifies that into tens-of-eV Σ_c garbage on non-16-GPU meshes.  The
replicated ``jnp.linalg.cholesky`` runs on the whole matrix on every
device (one dense potrf per q), so L_q is bit-identical across device
counts and process grids.  This mirrors the eigh-backend policy in
``bse/vq_interp`` (native batched by default; FFI backends reserved for
tiles too large to replicate).  See
``reports/gw_zeta_mesh_invariance_2026-07-20``.

Above the replication cap the older policy applies: cuSolverMp on
**true 2D meshes** (px≥2 AND py≥2) — it bundles the distributed
Cholesky into one FFI call per q, vs the in-tree ``sharded_cholesky``'s
many small NCCL all-reduces per panel — otherwise the in-tree sharded
path.

Override via cohsex.in ``distributed_cholesky``:
  ``off``        → force the in-tree sharded Cholesky.
  ``cusolvermp`` → force cuSolverMp (legacy alias ``on``).  EXPLICIT
                   choice via the distrib_la door: refuses at
                   resolve time on a non-CUDA mesh, a build without
                   the compiled handler, or a 1-D mesh (block-cyclic
                   layout degenerates) — never a silent fallback
                   (doctrine 3; audit fix/zq 2026-07-28).
  ``slate``      → SLATE ``potrf`` — the portable (Frontier/Aurora)
                   backend.  EXPLICIT choice: fails loudly if the
                   FFI/library is absent or the mesh geometry is the
                   guarded 1×q case (SLATE stride assert; see
                   services/distrib_la/tests/test_distrib_la_contract.py,
                   where that pin now lives) rather than
                   silently running a different backend.
  ``auto`` (default) → replicated dense for fit-size stacks, else
                   cuSolverMp on true 2D / sharded otherwise (neither
                   cuSolverMp nor slate is auto-picked below the cap).

### `src/isdf/core.py` — `_resolve_solver_kind_transverse`

Pick the transverse-channel ζ-fit solver: cuSolverMp distributed
getrf+getrs vs the in-tree per-q ``jnp.linalg.solve`` + ridge.

``transverse_zeta_solve`` (deck key, 2026-08-01) selects the SOLVE
FAMILY first, before any backend ladder:

* ``'ridge'`` (default) — the historical LU+ridge family below,
  byte-identical behaviour.
* ``'rank_truncate'`` — per-q eigh pseudo-inverse of the indefinite
  transverse CCT with an |λ| cut (the charge channel's conditioning
  cure ported to the transverse channel; see
  ``_charge_factor_math``'s ``'transverse_rank_truncate'`` mode).
  Returns ``'transverse_rank_truncate'`` — the LOCAL plan (whole-tile
  replicated eigh, q-parallel at P>1, valid at ANY logical extent on
  ANY mesh).  Its DISTRIBUTED plan (pzheevd at the padded extent) is
  selected by ``distributed_zeta_solve = 'distributed'`` exactly like
  the charge channel — the ζ-fit caller overrides the kind to
  ``'distributed_transverse_rank_truncate'`` after resolving the
  tier.  ``distributed_lu`` names an LU backend this family does not
  run, so an EXPLICIT ``distributed_lu`` request combined with
  ``rank_truncate`` REFUSES here (promise contract) instead of
  silently ignoring one of the two keys.  Since 2026-08-22 the LOCAL
  plan carries the CHARGE branch's capacity gate
  (:func:`_replicate_rank_truncate_ok` →
  :func:`_rank_truncate_capacity_error`), because it allocates the
  same replicated ``(q_batch, μ, μ)`` c128 eigh operand: pass ``nq``
  and ``replicated_factor_used`` to arm it.

The rest of this docstring documents the RIDGE (LU) family.

Default policy (2026-05-12): mirrors the charge-channel resolver —
use cuSolverMp on **true 2D meshes** (px≥2 AND py≥2).  cuSolverMp
0.7.2 fixes the earlier 2D-grid getrf/getrs correctness bug
(validated end-to-end on MoS2 3×3 bispinor at 2×2 mesh; see
``src/ffi/cpp/cusolvermp/batched_solve_lu_ffi.cc`` for history).

Tradeoff: small FFI setup overhead at MoS2 scale (n_rmu=656,
2×2 mesh).  At CrI3 6×6 80 Ry (n_rmu≈1800, 4×4 mesh) the cuSolverMp
path is the right tool.

Override via cohsex.in ``distributed_lu``:
  ``off``        → force per-q ``jnp.linalg.solve``.
  ``cusolvermp`` → force cuSolverMp (legacy alias ``on``).  EXPLICIT
                   choice via the distrib_la door: refuses at
                   resolve time on a non-CUDA mesh, a build without
                   the compiled handler, or a 1-D mesh — never a
                   silent fallback (doctrine 3; audit fix/zq
                   2026-07-28).
  ``scalapack``  → ScaLAPACK ``pXgetrf``+``pXgetrs`` from Cray LibSci
                   — the host/CPU-backend backend (liblorrax_ffi_host).
                   EXPLICIT choice, never auto-picked; fails loudly if
                   the host FFI is absent, and requires a square or
                   1-D mesh (pXgetrf needs square blocks).
  ``auto`` (default) → cuSolverMp on true 2D, legacy otherwise.
  (No ``slate`` value: a SLATE getrf wrapper does not exist yet.)

``n_rmu_logical`` (the LOGICAL transverse centroid count) activates
the resolve-time divisibility contract for the two DISTRIBUTED
backends: the indefinite solve must run at the logical μ extent
(ROOT_CAUSE.md 2026-07-08 — pad-shape LU roundoff is amplified O(1)
in the near-null transverse modes), and the block-cyclic descriptors
need ``n_log % px == n_log % py == 0``.  When they don't divide:

  * EXPLICIT request (``cusolvermp``/``on``/``scalapack``) → raise
    HERE, at resolve time, naming the fix — the promise contract
    (quality pattern #6/#8; the same treatment the charge W solve
    got in the two-plan cleanup).  Before 2026-07-27 this demoted to
    the per-q replicated LU via a ``warnings.warn`` deep inside
    ``solve_zeta`` — the ledgered "silent replicated-LU fallback".
  * ``auto`` resolution → announce the demotion (rank-0 print) and
    return the per-q ``'lu'`` route.

Callers that don't know ``n_rmu_logical`` (pass ``None``) keep the
pure mesh/backend ladder; ``solve_zeta`` retains an announced
call-time demotion as defense in depth for those.

### `src/isdf/core.py` — `_resolve_solver_kind`

Single source of truth for the ``auto`` resolution.  Transverse
channels (γ̃^i, μ_L≠0) take ``_resolve_solver_kind_transverse``;
charge channel takes ``_resolve_solver_kind_charge``.

``n_rmu`` (logical centroid count) and ``nq`` (per-q factor batch =
``C_q.shape[0]``) let the charge resolver pick the mesh-invariant
replicated dense factor for fit-size stacks — and, since 2026-08-22,
let the TRANSVERSE resolver apply the same replicated-eigh capacity
gate (``_rank_truncate_capacity_error``) instead of OOMing late;
``charge_zeta_solve``
(``'rank_truncate'`` | ``'cholesky'``) then picks the rank-revealing
eigh pseudo-inverse vs Cholesky on that route.  The ζ-fit caller passes
all three (``isdf_fitting.fit_zeta_to_h5``).  A concrete ``solver_kind``
is returned unchanged (so ``factor_c_q`` / ``solve_zeta`` re-resolving
the already-resolved kind need not repeat them).

### `src/isdf/core.py` — `_env_override_raw`

THE non-empty-env-wins rule of the deprecated env twins, in ONE
place: the raw env string when it is set and non-blank (that value
wins this release), else ``None`` (the input key is used).  Shared by
the factor sites (:func:`_deprecated_env_float`) and the ζ-provenance
record (:func:`deprecated_env_record` ←
``gw.gw_init._zeta_fit_provenance``) so the two can never drift
(quality pattern #3; audit fix/zq 2026-07-28).

### `src/isdf/core.py` — `deprecated_env_record`

The string ζ-fit provenance records for a deprecated env-twin knob:
the raw env string when the env form wins (the exact rule the factor
sites apply, via :func:`_env_override_raw`), else ``repr(key_value)``.
Byte-identical to the historical inline format in every case that
ever produced a reusable ζ, so existing provenance stamps keep
matching.  (audit fix/zq 2026-07-28)

### `src/isdf/core.py` — `_deprecated_env_float`

Input key is the source of truth; a non-empty env var still overrides,
but prints a deprecation notice on rank 0 (once per process).

Empty/unset env → the key's value, exactly.  This also removes the old
crash on ``LORRAX_ZETA_RCOND=""`` (``float('')``).

### `src/isdf/core.py` — `_resolve_zeta_gather`

Resolve the ζ back-solve TIER — the input key
``distributed_zeta_solve``.

Returns ``'replicated'``, ``'per_q'`` or ``'distributed'``.

* ``replicated`` — today's path: the back-solve all-gathers the whole
  ``(q_batch, μ, μ)`` factor onto every rank, ``nq·μ²·16`` B per rank
  (18.9 GB at MoS2 12×12 / μ=1998 counting the logical-extent copies,
  and it is re-gathered on EVERY r-chunk).
* ``per_q`` — gather ONE ``(μ, μ)`` tile at a time and loop q inside
  the r-chunk.  ``μ²·(1 + 1/p_y)·16`` B (75 MB at μ_pad=2048 on an 8×8
  mesh, 1.8 GB at μ=10k).  Same per-q arithmetic as the batched
  kernel; only the live gathered extent shrinks.  The slice is taken
  INSIDE a ``shard_map`` (``_per_q_block``) — written as a
  ``with_sharding_constraint`` on a traced-``q`` slice it read the
  same way but COMPILED to the full ``(nq, μ, μ)`` gather plus a
  dynamic_slice, which is worse than ``replicated`` and cost 12–40×
  the back-solve wall (scorecard Y.2; do not regress it).
* ``distributed`` — the factor is NEVER gathered.  ``C_q`` is
  eigendecomposed distributed (ScaLAPACK ``pzheevd``), truncated on the
  replicated spectrum, and the truncated pseudo-inverse ``C⁺`` is kept
  2D-sharded; the back-solve is a stacked 2D-sharded GEMM ``C⁺ @ Z``.
  This is the ONLY tier whose eigh ITSELF divides by P — the other two
  run whole-tile dense ``eigh``s per q (q-parallel over devices above
  the replicated plan's fold threshold, so min(P, nq)-scaling since
  2026-08-01; redundant on every rank below it — ~5.5 h at μ=4k,
  ~86 h at μ=10k for the full sweep, /min(P, nq) with the fold).
  EXPLICIT opt-in only: ``auto`` never picks it, because it changes the
  arithmetic (block-cyclic eigh ⇒ a different, equally valid gauge) and
  so is not bit-identical to the other two.
* ``auto`` (default) — ``replicated`` while the gather fits under
  :data:`_ZETA_GATHER_MAX_BYTES`, ``per_q`` above it.  At fixture scale
  (nq=9, μ_pad=64 ⇒ 0.6 MB) that is ``replicated``, i.e. bit-identical
  to the pre-feature path; at MoS2 12×12 / μ=2016 (9.4 GB) it is
  ``per_q``.

``distributed`` additionally REQUIRES (all checked here, at resolve
time, so nothing fails minutes later inside an FFI call):

* ``charge_zeta_solve = 'rank_truncate'`` — the tier IS distributed
  rank truncation, and the spectral cut is the charge channel's
  conditioning cure (ADVICE §6a); a plain distributed inverse would
  silently destroy the physics, so it is refused rather than offered;
* a mesh the ScaLAPACK eigh backend accepts — host devices, one
  process per device, square or 1-D, ``μ_pad`` divisible by both axes
  (``distrib_la.resolve_backend('eigh', 'distributed', …)`` owns that
  ladder and raises with the failed guard named).

On the TRANSVERSE channels (``vertex_mu_L != 0``) ``distributed``
resolves to ``per_q``: the transverse CCT is Hermitian INDEFINITE, so
no eigh-based rank truncation applies to it, and its distributed route
is the already-2D-sharded ``pXgetrf``/``pXgetrs`` pair selected by a
DIFFERENT key (``distributed_lu = scalapack``).  One key drives both
channels, so raising here would kill a bispinor run in the transverse
fit after the charge fit had succeeded.

### `src/isdf/core.py` — `_close_the_cut`

Move a ζ rank cut off any degenerate block it slices, by DROPPING the block.

The device face of ``common/spectral_closure``, wrapped once so all four
ζ truncation sites (charge / transverse × replicated / distributed) get
the same criterion, the same message and the same mode.

THE DIRECTION IS THE MODULE'S DEFAULT, not a choice made here — the
owner's ruling of 2026-08-10, that a cut landing mid-block truncates the
whole block.  So the retained rank comes DOWN, never up, and the
amplification cap ``rank_criterion`` sized it by is satisfied by
construction afterwards.  Nothing at this seam passes ``direction=``: a
site that needs the other one is a finding to report, and the wiring
ratchet in ``tests/test_spectral_closure.py`` asserts no site does.

WHY IT IS SHAPED LIKE THIS.  The cut lives inside a jitted kernel whose
eigenvalues never reach host, so the move has to be pure ``jnp`` — it is,
and it is a cumulative AND over adjacency links with no data-dependent
trip count, so it costs one sort and one cumprod per q against the
``eigh``'s O(n³).  A jitted kernel also cannot raise, which is the
division of labour ``centroid/pivoted_cholesky`` already documents ("a
jitted kernel cannot raise, so it reports and this refuses"): under
``strict`` the firing is recorded through a host callback and refused by
``spectral_closure.raise_if_pending`` at the next host seam, so the flag
means the same thing here as at the host sites.

THE ONE CASE THE DROP DIRECTION ADDS is a block that reaches ``λ_max``,
where dropping it would leave rank zero.  The host face raises on it; a
kernel cannot, so the count is carried out and ``_charge_factor_math``'s
existing zero-rank refusal catches it — which is why that refusal now
names closure as a possible cause.

MESH INVARIANCE.  ``close_keep_mask`` is elementwise plus a sort and a
cumulative product over the SPECTRUM axis, which is never the sharded
axis on any of these routes — the replicated tiers factor whole logical
blocks, and the distributed tier's ``_masks`` runs on the replicated
``lam``.  So the moved mask is bit-identical across device counts, and
the factor keeps the mesh-invariance contract it had before.

### `src/isdf/core.py` — `_certify_the_cut`

GATE the ζ rank cut against the certified regime.  Device face.

The sibling of :func:`_close_the_cut`, and the reason this function
exists at all: that one decides WHERE the cut may land, this one decides
whether the cut was allowed to happen at this conditioning.  Until
2026-08-22 the ζ truncation printed ``n_keep/q`` and ``kappa/q`` and
GATED ON NEITHER — announced-but-ungated truncation, the pattern
``TASTE.md`` (2026-08-15) names as an instrument that measures a defect
and proceeds.

MEASURED, and it is why the threshold is an ABSOLUTE achieved
amplification rather than a drop fraction (register 2026-08-15): Si
4×4×4 SYM/SOC 128-band, ``zeta_rcond = 1e-10``, 1776 centroids on a deck
with ngkmax = 588 — ``n_keep/q = 1469…1472 of 1776`` at
``kappa/q ≈ 9.7–10.0e9``, i.e. sitting on the rcond floor.  Σ_c MAE
**54.4 eV**, max 100.3 eV, **exit 0, no SANITY banner, no refusal**.  The
same deck at 600 centroids does not truncate and gives 0.90 eV.

THE DROP FRACTION IS NOT THE GATE, and must not be re-proposed: MoS2
production discards 33 % of the RANK at the certified rcond and is
right, this deck discards 17 % and is wrong by 54 eV, and Si 960 at
rcond 1e-6 discards 34 % and moves the σ-star spread by 0.005 meV.  The
derivation and the full site register are in
``docs/dev/rank_truncation_policy.md``; the criterion itself, the
ceiling constant and the message live in ``common/rank_criterion``.

``kappa_certified`` is ``None`` for a site no measurement covers (the
transverse channel today).  Then only the discarded-weight finding can
fire, and the log says the ceiling is absent rather than reporting a
clean bill — an absence is not a pass.

A jitted kernel cannot raise, so a firing is recorded through a host
callback and ``rank_criterion.raise_if_pending`` refuses at the next host
seam (``gw_init``, immediately after the fit and before ζ is consumed) —
the same division of labour :func:`_close_the_cut` already documents.

THAT CALLBACK NEEDS A CPU DEVICE IN THE BACKEND, and so does the
``jax.debug.print`` above it.  Measured on jax 0.9.1 / CUDA: with a
GPU-only backend, ``jax.debug.print``, ``jax.debug.callback`` AND
``io_callback`` all raise "failed to find a local CPU device to place the
inputs on".  A LORRAX run never sees that — ``runtime.
initialize_communicator_stack`` sets ``JAX_PLATFORMS="cuda,cpu"`` — but a
bare process that imports this module without booting the runtime does,
which is why the reachability probe in ``tests/test_charge_zeta_route``
runs under ``JAX_PLATFORMS=cpu``.  If that ever changes, this gate and
the existing ζ telemetry lose their host seam together.

COST.  One reduction pass over the spectrum axis per q: three sums and a
min over ``n_log`` values against the ``eigh``'s O(n³).  Unmeasurable,
and it is the only affordable certification at this seam — the honest
one (refit and measure Σ) is the run itself.

THE MODE IS RESOLVED AT TRACE TIME, and the factor jits are cached on a
key that does not include it — the same property :func:`_close_the_cut`
has for ``LORRAX_SPECTRAL_CLOSURE``.  So changing the dial part-way
through ONE process does not retrace an already-compiled factor.  That
is correct for a per-run dial and is stated here rather than discovered:
a test that flips the variable between two calls in one process must
flip it around the FIRST call that compiles the shape.

### `src/isdf/core.py` — `_close_the_cut_padded`

:func:`_close_the_cut` for the distributed tier's PADDED spectrum.

The distributed route never forms the logical block alone: it eighs the
identity-padded matrix ``[C_log 0; 0 I]``, whose spectrum is
``spec(C_log) ∪ {1.0}×(n_pad − n_log)``.  Those pad eigenvalues are
**exactly 1.0 and therefore exactly degenerate with each other**, so a
block walk that reached them would move all ``n_pad − n_log`` of them at
once — admitting them under ``keep_block``, discarding them under the
default ``drop_block``, and in EITHER direction making the retained rank
a function of the DEVICE COUNT.  That is the precise defect this route's
``lam_max`` note exists to prevent, and the one
``rank_criterion.violations()`` reports as ``n_dropped_alignment``.  The
withdrawal below is therefore direction-independent, and so is the gate
on it.

So the pad is withdrawn from the walk before it starts.  ``lam`` is
ascending, so its exact-1.0 entries are contiguous; the first
``n_pad − n_log`` of them are taken as the pad and demoted to magnitude
zero, which puts them below every cut and makes them un-linkable (the
guard never links a pair whose larger member is zero).  If a PHYSICAL
eigenvalue also happens to be exactly 1.0 the choice of which duplicates
to demote is immaterial — the values are identical, so the multiset the
walk sees is the same either way.

Whatever the original cut decided about the pad is preserved: a kept pad
direction inverts to ``1/1.0 = 1`` against an identity block and is inert
by construction, and this guard has no business changing it.

### `src/isdf/core.py` — `_withdraw_identity_pad`

``(spectrum with the identity pad demoted to 0, pad mask)``.

ONE implementation of the pad withdrawal both padded-spectrum guards
need — :func:`_close_the_cut_padded` (so a block walk cannot sweep the
exactly-degenerate pad and make the retained rank a function of the
device count) and :func:`_certify_the_cut` at the distributed charge
site (so the pad is not counted as discarded weight or as dropped
directions).  The mechanism is the one that function's docstring
argues; it lives here so the two cannot drift apart.

### `src/isdf/core.py` — `_charge_factor_math`

The per-q dense factor arithmetic — ONE kernel, shared bit-for-bit
by the all-ranks (replicated) and q-parallel executions of the
replicated plan (:func:`_factor_c_q_replicated`,
:func:`_factor_c_q_replicated_qparallel`).

``C_log``: ``(nqb, n_log, n_log)`` whole LOGICAL tiles; the caller
guarantees they are fully local / replicated per device.  Pure jnp with
NO sharding ops, so the emitted per-q LAPACK calls are identical
wherever it runs — the bit-identity contract of the q-parallel fold.
``mode`` selects the factor exactly as documented on
:func:`_factor_c_q_replicated` (``'rank_truncate'`` | ``'cholesky'``,
charge channel) plus ``'transverse_rank_truncate'`` (bispinor
transverse channels, 2026-08-01): the SAME eigh rank truncation on the
Hermitian INDEFINITE transverse CCT — the cut is on |λ| (both signs
are physical there) and the return value is the EXPLICIT truncated
pseudo-inverse C⁺ = Σ_{|λ|>τ·|λ|_max} vᵢvᵢᴴ/λᵢ, not a B with
BBᴴ = C⁺ (no such Hermitian factor exists for an indefinite C⁺;
explicit C⁺ also halves the per-r-chunk back-solve to ONE matmul —
the same trade the distributed charge tier documents).

### `src/isdf/core.py` — `solve_zeta_charge_dense`

THE producer's charge-ζ solve on ONE whole, unpadded (n_μ, n_μ) tile.

``ζ = C⁺Z`` (``charge_zeta_solve='rank_truncate'``, the production
default) or ``ζ = (C + ridge)⁻¹Z`` through two triangular solves
(``'cholesky'``).  The factor arithmetic is
:func:`_charge_factor_math` — the SAME traced kernel the sharded
producer route runs — and the back-solves are the same two bodies
``solve_zeta`` applies (``_pinv_matmul_logical`` /
``_tri_solve_logical``), written here without the identity pad because
a caller holding one whole tile has no pad to slice.

WHY THIS IS PUBLIC.  ``bse.vq_interp``'s per-Q refit has to solve the
same system the producer solved, or the ζ' it builds differs from ζ in
exactly the near-null subspace the producer discarded — and ``V_Q =
Σ_G conj(ζ(G)) v(q+G) ζ(G)`` is QUADRATIC in it.  Before 2026-08-11 the
refit ran a plain Cholesky with a fixed 1e-14·|tr C| ridge under a
comment claiming it followed a private ridged-Cholesky helper of THIS
module — a symbol that has never existed in this tree — and the tile
identity
``vq_interp.refit_ongrid_null`` read 3.3, 16, 51 and 140 against a
5.0e-02 bracket, monotone in the fraction of directions the producer's
truncation had dropped (4.7 % → 3.289, 58.6 % → 139.9; five parents,
``tests/known_failures/2026-08-11-narrowed-zeta-window-clears-fh-and-the-tile-null-still-refuses.md`` §4).  A second, private re-implementation
of a solve is how that happens; one exported entry point is the fix.

``zeta_rcond`` / ``zeta_ridge`` are taken EXACTLY as given — this
function applies no ``LORRAX_ZETA_RCOND`` / ``LORRAX_ZETA_RIDGE``
override of its own.  Its caller is reproducing a fit that already
happened, and the EFFECTIVE (post-env) values of that fit are recorded
in the ζ file's ``isdf_header/fit_provenance``
(:func:`gw.gw_init._zeta_fit_provenance`); re-applying today's
environment on top would silently solve a different system than the one
on disk.  The producer-side entry points (:func:`_factor_c_q_replicated`
and friends) still apply the deprecated env twins, because there the
deck is what is being resolved.

``rank_log`` defaults to the producer's rule (on for
``rank_truncate``), so a
refit prints the same ``n_keep``/``kappa`` line the fit did and the two
can be read against each other.  jit-safe: everything below is jnp plus
``jax.debug`` callbacks.

### `src/isdf/core.py` — `_factor_c_q_replicated`

Dense, fully REPLICATED factor of the identity-padded charge CCT.

Two selectable conditioners share this ONE replicated (mesh-invariant)
seam — ``charge_zeta_solve`` picks which factor is returned:

* ``'rank_truncate'`` (production default) — rank-revealing ``eigh``
  pseudo-inverse factor ``B`` with ``B Bᴴ = C⁺`` (see the WHY note
  inside).  The back-solve is a matmul ``ζ = B(BᴴZ)``.
* ``'cholesky'`` — the historical lower-triangular Cholesky factor
  ``L`` with ``L Lᴴ = C+ridge``.  Back-solve is two triangular solves.
  Bit-identical to the pre-rank-truncation code (the frozen contract);
  it is the selectable ALTERNATIVE.

Mesh-invariant by construction for BOTH: the factorisation runs on the
fully-replicated LOGICAL block — one dense ``eigh`` / ``cholesky`` per q
on whole tiles — so the factor is bit-identical across device counts and
process grids, unlike the block-cyclic cuSolverMp potrf whose partial-sum
regrouping depends on ``(px, py)``.  This is the single code path for the
``'replicated_cholesky'`` / ``'replicated_rank_truncate'`` auto picks
(fit-size n_μ on any mesh) and every single-device / 1-D-degenerate mesh
(where a dense factor is the only option).

Cholesky ridge (two per-q scalar terms, so both mesh-invariant):

  ridge = [ 1e-14·|tr(C)|  +  zeta_ridge·|tr(C)|/n ] · I

* The hard ``1e-14·|tr(C)|`` FLOOR is unchanged from the historical
  single-device path — it lifts the tiny negative eigenvalues that
  appear with more centroids than band pairs so ``potrf`` stays real.
  With ``zeta_ridge == 0`` (the default) the factor is bit-identical to
  that path (the frozen-golden contract).
* ``zeta_ridge`` (a fraction of the mean diagonal tr(C)/n, default 0) is
  an OPT-IN Tikhonov term that CONDITIONS a near-singular CCT (n_μ
  over-complete for the pair-density rank).  ``rank_truncate`` (the
  default) is the PRINCIPLED cure that supersedes it — drop the near-null
  directions instead of shifting them — so the ridge stays 0 there.
  Tune ε via the ``zeta_ridge`` input key in the deck; the
  ``LORRAX_ZETA_RIDGE`` env form is a DEPRECATED twin (scorecard AV:
  still wins when set non-empty, but loudly — see
  :func:`_deprecated_env_float`) slated for removal.

Factorise at the LOGICAL extent and re-embed identity in the pad block
(√1 = 1 for L; B's pad block is likewise identity and is sliced away in
the back-solve) — see :func:`_identity_pad_block_diagonal`.  The factor
regroups partial sums when the matrix extent changes, so factorising at
the logical (not padded) extent keeps the factor pad-extent-invariant
(the fixed-P invariance gate).

### `src/isdf/core.py` — `factor_c_q_replicated_batched`

:func:`_factor_c_q_replicated` over q in bounded batches.

Per-q independent, so concatenating the batches reproduces the one-shot
call; only the XLA workspace differs.  A single batch (every stack that
already fitted) takes the identical code path it always did.

P>1 SCHEDULE (2026-08-01): above :data:`_QPARALLEL_MIN_NQ_MU3` the
same plan EXECUTES q-parallel (:func:`_factor_c_q_replicated_qparallel`
— q's scattered over all devices, whole tiles per q, bits unchanged)
instead of redundantly on every rank.  This is a fold INTO the
replicated plan, deliberately not a third resolution — see the WHY on
the q-parallel function.

### `src/isdf/core.py` — `_qparallel_factor_ok`

True when the replicated charge factor should EXECUTE q-parallel.

``LORRAX_ZETA_QPARALLEL``: unset/``auto`` → fold above
:data:`_QPARALLEL_MIN_NQ_MU3` (needs >1 device and >1 q to scatter);
``0`` → never (the pre-fold all-ranks execution, kept as the A/B
control); ``1`` → always (the bit-identity gate forces it at fixture
size).  Either way the RESULT is the same bits — this knob selects an
execution schedule, never a numerical route.

### `src/isdf/core.py` — `_factor_c_q_replicated_qparallel`

The replicated charge factor, EXECUTED q-parallel.

WHY THIS IS A FOLD AND NOT A THIRD RESOLUTION: a plan in this family
is a numerical contract — ``replicated`` = whole-tile dense factor,
bit-identical across meshes and device counts; ``distributed`` =
block-cyclic eigh, a different (equally valid) gauge, explicit opt-in.
This path changes only WHICH device runs each per-q factorisation,
never what is computed, so its output is the replicated plan's output
to the bit and it carries no new resolver string, no new input key,
and no new downstream contract.  (Precedent: the W-solve family's
LOCAL plan is likewise q-parallel — scorecard AN.)

Schedule: zero-pad the q axis to the device count, scatter q over the
FLATTENED mesh (``P(('x','y'), None, None)``) through the measured
single-axis staging (``P('x', None, 'y')`` — see gw/w_isdf's
involuntary-remat note), factor each OWNED q as one whole-tile call
into :func:`_charge_factor_math` (per-q ``fori_loop``: the XLA eigh
workspace is bounded by ONE (μ, μ) tile, strictly tighter than
:func:`_replicated_factor_q_chunk`'s batch bound), skip pad q's with a
``lax.cond`` (so the Cholesky branch never factors filler and the
rank log prints no phantom q's), then stage the factors back to
``P(None, 'x', 'y')`` and re-embed the identity μ-pad block.

BIT-IDENTITY to the all-ranks execution, claim by claim:

* the factor is per-q independent — the q-batch split is already
  relied on (``factor_c_q_replicated_batched`` concatenates cap-sized
  batches) and XLA's batched LAPACK wrappers loop per matrix;
* the reshards move exact byte copies (pure data movement);
* the per-q arithmetic is the SAME traced kernel on the same whole
  logical tile (``_charge_factor_math``; the μ-slice/zero-refill is
  the same ``solve_at_logical``; the identity μ-pad re-embed is the
  same helper).

Gate: ``tests/test_zeta_mesh_invariance.py::
test_qparallel_execution_is_bit_identical_to_replicated`` (exact
equality, both modes, non-dividing nq, padded μ).

Observability delta, deliberate: the rank_truncate conditioning log
prints per OWNED q from the owning process (the all-ranks execution
printed every q from every process); fields are unchanged.

### `src/isdf/core.py` — `_certify_transverse_ridge`

CONDITIONING INSTRUMENT for the default (ridge) transverse path.

THE DEFECT THIS CLOSES.  The ridge family is the default transverse
factor and it had **no conditioning instrument at all** — the
``rank_truncate`` family prints ``n_keep/q`` and ``kappa/q``, the ridge
family printed nothing, so on the default path there was no number to
read and no number to gate.  Registered three times (``bispinor``: the
refuted docstring mechanism, the harmful positive ridge above κ~1e12,
and the missing instrument).

WHAT IS MEASURED, and what it is worth.  ``|diag U|`` of the pivoted LU
gives ``kappa_lb = max|u_ii| / min|u_ii|``, which is the standard cheap
conditioning proxy and is a **LOWER bound** in practice, not a
certificate.  That asymmetry is exactly right for a gate that fires when
the number is large: exceeding the ceiling PROVES κ exceeds it, so the
refusal is sound.  Failing to exceed it proves nothing, and the log says
so instead of reporting a clean bill — an absence is not a pass
(``TASTE.md``, "a check that cannot fail is not evidence").

COST.  One diagonal extraction and two reductions over an array the
factor already materialised: ``O(nq · n_log)`` against the LU's
``O(nq · n_log³)``, plus ONE host sync per channel (this runs once per
channel, not per r-chunk).  Priced before enabling, per the owner's
truncation directive.

NOT REACHABLE on the ScaLAPACK transverse plan: its factor is an opaque
``FactorToken`` with no public buffer, by design.  The caller says so
rather than silently skipping.

### `src/isdf/core.py` — `_embed_lu_padded`

Zero-embed per-q LOGICAL LU factors at the padded extent and set
identity on the pad-block diagonal (shape/sharding uniformity only:
the back-solve slices back to the logical block, so the pad content
is never part of any solve — same contract as the charge factor's
identity pad).

### `src/isdf/core.py` — `_factor_c_q_transverse_lu`

LOCAL-plan hoisted transverse factor: per-q pivoted LU of the
ridged LOGICAL block, once per channel.

Returns ``(LU_q, perm_q)``:

* ``LU_q`` ``(nq, n_rmu, n_rmu)`` at PADDED extent, sharded
  ``P(None, 'x', 'y')`` — the packed L/U factors in the logical
  block, identity in the pad block.  Downstream gather tiers
  (replicated / per_q) consume it exactly like the CCT passthrough
  they used to gather: same shape, same sharding, same bytes moved.
* ``perm_q`` ``(nq, n_log)`` int32, replicated — the LU permutation
  for ``lax.linalg.lu_solve``.

Execution schedule mirrors the charge fold
(:func:`_factor_c_q_replicated_qparallel`): q-parallel over the
flattened mesh when :func:`_qparallel_factor_ok` says so (the factor
is per-q independent; scatter/gather reshards are exact byte moves;
the per-q arithmetic is the ONE shared kernel
:func:`_transverse_lu_math`), all-ranks whole-tile execution
otherwise.  Both produce the same bits.

### `src/isdf/core.py` — `_factor_c_q_transverse_distributed_lu`

DISTRIBUTED-plan hoisted transverse LU on the ridged logical block.

Returns a :class:`distrib_la.FactorToken` at ``n = n_log``, the
LOGICAL extent (the resolve contract guarantees
``n_log % px == n_log % py == 0`` on this path).  Inside it are the
block-cyclic provider factors — each rank's shard IS its local block —
and the provider-native, per-rank pivot rows.  ScaLAPACK uses i32;
cuSOLVERMp uses i64.  Both remain private to the service token.

THE TOKEN IS WHY THIS SIGNATURE CHANGED.  It used to hand back
``(LU_q, ipiv_q)`` with "never reshard it, feed it back verbatim"
written in the docstring and re-written at the ``pXgetrs`` call three
frames away.  A comment is not a contract: the pivot vector was an
ordinary ``jax.Array`` that anything could gather, slice or reshard,
and gathering it is silently wrong rather than loud.  The token has
no public factor attribute at all, so there is nothing to reach —
``distrib_la.solve(token, B)`` is the only thing that can consume it,
and it checks B against the extents the factor was made at.

``trace_per_q`` is the materialized trace used by the fused path.  It is
passed explicitly because allowing XLA to fuse the reduction into this
preparation kernel changed its reduction tree on the production CCT;
the near-null transverse solve amplified that tiny ridge change.

### `src/isdf/core.py` — `_distributed_q_batch`

q-batch size bounding the GEMM's gathered transient.

Reuses :data:`_ZETA_GATHER_MAX_BYTES` (``LORRAX_ZETA_GATHER_CAP_GIB``,
4 GiB) because it gates exactly the same thing here as it does for the
other tiers: the live extent of the back-solve's gathered operands.

### `src/isdf/core.py` — `_collective_chunk_bytes`

Upper bound on ONE emitted collective's payload, in bytes.

``LORRAX_COLLECTIVE_CHUNK_MB`` (default 128 MB, see the note above).
``0`` or a negative value disables chunking entirely and restores the
pre-AF single-shot behaviour — kept only so the failure can be
reproduced on demand.

### `src/isdf/core.py` — `_chunk_q`

Largest q-block whose LARGEST single collective fits the budget.

``per_q_collective_bytes`` must be the size of the BIGGEST collective
the block emits per q — not the sum over collectives and not the live
footprint.  The bound is per-instruction because that is what the
transport sees.

### `src/isdf/core.py` — `_chunk_log`

One line per call site naming the emitted per-collective payload.

Mandatory production telemetry: a tier
that silently stopped chunking would otherwise be invisible until it
took a 72-node job down again.  Deduplicated on the tuple, because the
back-solve site is re-entered once per r-chunk (9–81 times).

### `src/isdf/core.py` — `_factor_c_q_distributed_rank_truncate`

Truncated pseudo-inverse ``C⁺``, formed and kept 2D-SHARDED.

Same physics as :func:`_factor_c_q_replicated`'s ``rank_truncate``
branch — drop ``λ < rcond·λ_max``, then ``C⁺ = Σ_{keep} vᵢvᵢᴴ/λᵢ`` —
with two structural differences:

1. the ``eigh`` is DISTRIBUTED (ScaLAPACK ``pzheevd`` over the whole
   mesh), so the O(nq·μ³) factorisation finally divides by P instead of
   running redundantly on every rank;
2. ``C⁺`` is returned EXPLICITLY (not as the factor ``B`` with
   ``BBᴴ = C⁺``).  Explicit costs one extra ``nq·μ³`` at fit time but
   halves the per-r-chunk back-solve: one GEMM ``C⁺Z`` instead of two
   (``B(BᴴZ)``), and the r-chunk loop runs 9–81 times.

PADDED extent, deliberately.  The other charge routes factor at the
LOGICAL extent and re-embed identity, because a blocked factorisation
regroups partial sums when the extent changes.  ScaLAPACK's descriptors
need ``n`` divisible by both mesh axes, which ``n_rmu_logical`` in
general is not and ``n_rmu_padded`` always is — so this route factors
the identity-padded block-diagonal ``[C_log 0; 0 I]``.  That is exact,
not a compromise: the blocks do not mix, so ``C⁺``'s logical block is
``pinv(C_log)`` and its pad block is ``I`` or ``0`` depending on which
side of the cut ``λ = 1`` lands; either way ζ's pad rows come out zero
because Z's pad rows are exactly zero (the bilinear-in-zero-padded-ψ
contract).  The *floating-point* consequence is that ζ from this tier
agrees with the replicated tier to ~κ·ε rather than bit-exactly —
which is already true of any block-cyclic eigh (different gauge), and
is why the tier is explicit opt-in.

``λ`` is replicated by ScaLAPACK's own contract (``W`` is a global
output computed on every process of the grid), so the truncation mask
is computed LOCALLY and is identical on every rank by construction —
no collective, and no chance of a rank-dependent cut.

``indefinite=True`` (2026-08-01) is the TRANSVERSE-channel mode
(``transverse_zeta_solve='rank_truncate'`` +
``distributed_zeta_solve='distributed'``): the transverse CCT is
Hermitian INDEFINITE, so (a) the cut is on ``|λ|`` (both signs are
physical) and (b) the pad block is ZEROED instead of identity —
``[C_log 0; 0 0]`` — so the pad eigenvalues are exactly 0, are
truncated for EVERY τ, and can never contaminate ``σ_max`` (an
identity pad's λ=1 modes could win σ_max on a small-|λ| transverse
spectrum; zeros cannot).  Zero rows/cols stay exact zeros through the
Householder tridiagonalization and deflate exactly, so the pad modes
are inert in the same block-diagonal sense the charge note above
argues — and their ``inv=0`` removes them from C⁺ regardless.  THIS
is what removes the transverse mesh-divisibility constraint: the
eigh runs at the PADDED extent (divisible by both axes by
construction of ``n_rmu_padded``), where the LU family had to refuse
(pad-extent LU roundoff is amplified O(1) through the near-null
modes that rank truncation removes).

### `src/isdf/core.py` — `_distributed_pinv_apply`

ζ = C⁺ Z as a stacked GEMM with BOTH operands 2D-sharded.

``out[q,i,j] = Σ_k C⁺[q,i,k]·Z[q,k,j]`` with ``C⁺`` at
``P(None,'x','y')`` (i on 'x', k on 'y') and ``Z`` at the same spec
(k on 'x', j on 'y') — the classic 2-D block GEMM pairing.  Rank (x,y)
all-gathers C⁺'s row-block along 'y' (full k for its own i rows) and
Z's column-block along 'x' (full k for its own j columns), multiplies
locally, and is done: no psum, and the output lands at
``P(None,'x','y')`` with no further movement.

COMMUNICATION, honestly counted (per rank, per r-chunk, μ_pad=μ,
r = r_chunk, mesh Px×Py):

    this tier   nq·(μ²/Px + μ·r/Py)·16 B   received
    replicated  nq·μ²·16 B                 received (the whole factor)
    per_q       nq·μ²·16 B                 received (same total, lower peak)

At MoS2 12×12 (nq=144, μ=2016, r_chunk=11664, 12×12 mesh) that is
5.3 GB/rank/r-chunk here against 9.4 GB/rank/r-chunk for the other two
— 1.8× less traffic AND a 36.8 MB live transient per q instead of a
65 MB gathered tile (replicated: 9.4 GB).  On top of that this tier
does NOT run ``_reshard_z`` (two all-to-alls moving the whole
``nq·μ·r`` tensor) and skips the first leg of the output reshard,
because Z is consumed in the layout it is built in.

The q axis is batched to bound the gathered transient (see
:func:`_distributed_q_batch`); the GEMM is per-q independent, so the
batching is invisible to the result.

### `src/isdf/core.py` — `factor_c_q`

Compute system-matrix L_q from CCT matrix.

For ``vertex_mu_L == 0`` (standard spin-traced path) the CCT is
Hermitian positive-definite (modulo numerical noise); we run the
optimized 2D blocked Cholesky and return the lower-triangular
factor.  Downstream :func:`solve_zeta` then does two
triangular solves per-q.

For ``vertex_mu_L != 0`` (transverse Lorentz channels γ̃^i, i∈{1,2,3})
the CCT is Hermitian but **indefinite** — Cholesky NaNs; the factor
is a per-q pivoted LU with a stabilising ridge, HOISTED here (once
per channel) since 2026-08-01.  The return value is a PAIR
``(factor, piv)``: the local plan stores ``(LU, perm)`` for
``lax.linalg.lu_solve`` (bit-identical to the fused per-r-chunk
``jnp.linalg.solve`` it replaced), the scalapack plan stores the
block-cyclic provider factors + private per-rank pivots for one
provider back-solve per r-chunk.

Padded-input path (``n_rmu_logical < C_q.shape[-1]``):
n_rmu may be padded to mesh divisibility at the boundary so the
``P(None, 'x', 'y')`` input sharding is admissible at any logical
centroid count (e.g. n_rmu_logical = 661 prime → padded to 672 on
a 4×4 mesh).  By the Phase 3a contract the trailing pad rows/cols
of C_q are exact zeros (bilinear in zero-padded ψ).  We add
identity ONLY to the pad-block diagonal in-place — turning C_q
into a block-diagonal ``[C_log 0; 0 I_pad]`` matrix — and then
run the same sharded Cholesky / LU path the divisible case uses.
Cholesky of an identity-padded matrix produces a factor whose
logical block matches the logical-only factor in exact
arithmetic; in floating point the match is ≤1e-7 rel (blocked
implementations regroup partial sums when the extent changes —
see ``_identity_pad_block_diagonal``).  The pad-block factor is
exactly identity; the back-solve's pad rows of ζ come out as zero
(because Z's pad rows are zero by the same bilinear argument).
For the indefinite transverse channels the exact-arithmetic
guarantee FAILS in floating point (near-null-mode amplification —
ROOT_CAUSE.md 2026-07-08), so ``solve_zeta`` slices that solve
back to the logical extent.  On single-device meshes the dense
Cholesky below also factorises at the logical extent and
re-embeds, making the charge factor pad-extent-invariant at P=1
(the fixed-P invariance gate).

This is NOT ridge regularisation of C_q.  The logical block is
untouched; identity is added ONLY to the pad-block diagonal.

Output sharding is ``P(None, 'x', 'y')`` natively at the padded
extent — no replication, no slice + embed gymnastics, the chol
stays sharded across the mesh.

Args:
    C_q: (nq, n_rmu, n_rmu) CCT matrix at PADDED μ extent, sharded
        ``P(None, 'x', 'y')``.  ``n_rmu == n_rmu_padded`` (== ∏ p_a
        of the device mesh) so the existing 2D-blocked path
        applies.
    mesh_xy: 2D device mesh.
    block_size: Tile block size (auto if None).
    vertex_mu_L: Lorentz vertex index (0 = spin-traced PSD path,
        1/2/3 = transverse indefinite path).
    n_rmu_logical: Logical centroid count.  When given and
        strictly less than ``C_q.shape[-1]``, the pad-block
        diagonal is set to identity before factorisation.
        ``None`` (default) skips the identity-pad: input == output
        extent and the matrix is assumed to be PSD on its full
        extent (legacy mesh-divisible path).
    zeta_rcond: rank-truncation cutoff for the
        ``'replicated_rank_truncate'`` charge factor (drop
        eigenvalues < ``zeta_rcond·λ_max``).  Ignored by the Cholesky
        paths.  Tune via the ``zeta_rcond`` input key in the deck;
        the ``LORRAX_ZETA_RCOND`` env form is a DEPRECATED twin
        (scorecard AV: still wins when set non-empty, but loudly)
        slated for removal.

Returns:
    For ``vertex_mu_L == 0``: L_q ``(nq, n_rmu, n_rmu)`` at PADDED
    extent, sharded ``P(None, 'x', 'y')`` — the Cholesky factor
    (block-diagonal ``[L_log 0; 0 I_pad]``) for the JAX cholesky
    paths, or the rank-revealing pseudo-inverse factor ``B``
    (``B Bᴴ = C⁺``) for ``'replicated_rank_truncate'``.  The two
    LIBRARY cholesky paths (``cusolvermp_cholesky``,
    ``slate_cholesky``) return a :class:`distrib_la.FactorToken`
    instead: their factor is a block-cyclic handle that means nothing
    off the grid that produced it, so it is opaque by construction and
    ``solve_zeta`` feeds it back whole.
    For ``vertex_mu_L ≠ 0``: the PAIR ``(factor, piv)`` described
    above.  ``piv`` is non-None ONLY on the local ``'lu'`` plan, whose
    factor is jax's own ``(LU, perm)``; every distributed factor is a
    :class:`distrib_la.FactorToken` carrying its own pivots.

### `src/isdf/core.py` — `_reshard_zeta_mu_X_r_Y_to_mu_XY`

Reshard (q_, μ_X, r_Y) → (q_, μ_XY, r_) for the cuSolverMp branches.

Single mesh axis ``'y'`` moves from the r-axis to the μ-axis (where
it joins ``'x'`` to form a flat tuple).  All other shardings stay.

Downstream consumer ``accumulate_rchunk_to_gflat`` wants ζ
μ-flat-sharded so the FFT box and gflat-accumulator both live at
``P(None, ('x','y'), None)``; landing ζ in that layout here means
the FFT runs sharding-preserving (no further reshard, no
replicated FFT box).

Note on overhead: tried both ``@jax.jit(donate_argnums=(0,))``
closure-wrapping (matching the ``_reshard_z`` pattern above) and a
module-level decorator with ``static_argnums``.  Neither flipped
XLA's ``is_sync`` flag on the emitted all-to-all from ``true`` to
``false``, and runtime cost was the same either way (~3 ms/call in
the trace).  The bare ``with_sharding_constraint`` is the simplest
form for the same emitted HLO; trace shows the reshard is not on
the critical path at MoS2 3×3 scale.

### `src/isdf/core.py` — `_distributed_backsolve`

RHS pad → distributed back-solve → output reshard → trim.

THE shared frame for every ζ back-solve that keeps the factor
distributed — cuSolverMp ``potrs``, the cuSolverMp/ScaLAPACK
``getrf``+``getrs`` pair, and the ``distributed`` tier's ``C⁺Z``
GEMM.  Those three differ ONLY in ``run``; the three things around
it are identical and used to be written out three times:

1. **NRHS padding.**  Every block-cyclic descriptor (and the GEMM's
   ``'y'``-sharded column block) needs the last axis divisible by
   ``Py``.  ``pad_last_axis_to`` appends zero columns, which give
   exactly zero solution columns, so this is free of arithmetic
   consequences.
2. **The output reshard.**  All three land ζ at ``P(None,'x','y')``
   = ``(q_, μ_X, r_Y)``; the downstream G-flat accumulator wants
   ``(q_, μ_XY, r_)`` so its FFT runs sharding-preserving.  That is
   ONE all-to-all on ``'y'`` (:func:`_reshard_zeta_mu_X_r_Y_to_mu_XY`)
   — half of what the replicated/per_q tiers pay, because their
   shard_map back-solve lands ζ column-sharded over the flat mesh.
3. **The trim** back to the caller's logical column count.

Keeping them here is not only de-duplication: FFI-adjacent
resharding is where this code base has lost the most time (J.9's
silent NaNs from a Z re-layout, T.4's per-r-chunk recompile of one),
so there is exactly one copy to keep right.

``run`` takes the PADDED Z and returns ζ at ``P(None,'x','y')``.

### `src/isdf/core.py` — `_reshard_zeta_r_XY_to_mu_XY`

Reshard (q_, μ_, r_XY) → (q_, μ_XY, r_) for the shard_map branch.

The shard_map triangular-solve naturally lands ζ at
``P(None, None, ('x','y'))`` because the solve is parallelised over
r-columns.  The downstream FFT wants μ-sharded.  Two mesh axes have
to move on the (μ, r) data axes; SPMD's all-to-all planner only
handles one mesh axis at a time, so we stage through the cuSolverMp
intermediate ``P(None, 'x', 'y')`` to keep every step a single-axis
all-to-all primitive ``(a_X, b) → (a, b_X)``:

  Step 1  (q_, μ_, r_XY) → (q_, μ_X, r_Y)   ['x' moves r → μ]
  Step 2  (q_, μ_X, r_Y) → (q_, μ_XY, r_)   ['y' moves r → μ]

### `src/isdf/core.py` — `_factor_nbatch`

The q-axis extent of a ζ factor, whichever kind it is.

``factor_c_q`` returns either a sharded array or an opaque
:class:`distrib_la.FactorToken`, and the token deliberately has no
``.shape``: a ScaLAPACK token holds a ``(nq, n, n)`` factor AND a
``(nq, P·ipiv_len)`` pivot vector, so "the shape" is not one thing and
a property that picked one of them would be a guess dressed as a fact.
The token publishes ``nbatch`` and ``n`` instead, which is what the two
readers in this file ever wanted.

NOT A PYTREE, and that is deliberate.  Registering the token so it
could be a traced ``jax.jit`` argument would let XLA relayout its
leaves at the boundary — and the ipiv is the one operand nothing
re-pins (``distrib_la.solve`` pins B, never the factor), so a
relayouted pivot vector is a silently wrong solve.  "Never reshard it"
is the whole contract; making it traceable is how it would be broken.
The token therefore travels as a Python value, which is exactly how
the production path uses it (``fit_one_rchunk`` calls the un-fused
``z_q_phase`` / ``solve_phase``, never the composed ``@jax.jit``
``_kernel``, which has no caller in this tree).

### `src/isdf/core.py` — `_zeta_logical_solvers._lu_apply_logical`

HOISTED transverse back-solve at the LOGICAL μ extent: apply
the per-q ``(LU, piv)`` factor that ``factor_c_q`` computed once
per channel.  ``jax.scipy.linalg.lu_solve((lu, piv), Z)`` runs
``lu_pivots_to_permutation`` + ``lax_linalg.lu_solve`` — exactly
the arithmetic ``jnp.linalg.solve`` runs after its internal
``lu()`` — so the result is bit-identical to the fused
``_ridge_indef_solve`` path this replaces (the ridge is baked
into the factor).  ``piv`` is built at the logical extent
already; ``solve_at_logical`` slices LU/Z and zero-refills ζ's
pad rows (gate: tests/test_transverse_factor_hoist.py).

### `src/isdf/core.py` — `_zeta_logical_solvers._tri_solve_logical`

Charge-channel two-triangular back-solve at the LOGICAL μ
extent (same ``solve_at_logical`` rationale — the
well-conditioned Cholesky back-solve only wobbles ≤1e-7 under a
pad-extent change, but at fixed shape it is exactly
pad-invariant, which the fixed-P invariance gate requires).
L is the block-diag ``[L_log 0; 0 I]`` factor; its logical
block is exactly the factor of the logical system.

### `src/isdf/core.py` — `_zeta_logical_solvers._pinv_matmul_logical`

Charge rank-truncation back-solve at the LOGICAL μ extent:
ζ = C⁺Z = B(BᴴZ), two matmuls (B is the pseudo-inverse factor,
B Bᴴ = C⁺).  ``solve_at_logical`` slices to the logical block
(dropping the identity pad — never inverted, so no LU/tri
amplification) and zero-refills ζ's pad rows.

### `src/isdf/core.py` — `_zeta_logical_solvers._pinv_apply_T_logical`

Transverse rank-truncation back-solve at the LOGICAL μ
extent: ζ = C⁺Z, ONE matmul (``Cp`` is the explicit truncated
pseudo-inverse of the indefinite transverse CCT).  Same
slice/zero-refill contract as the other whole-tile bodies.

### `src/isdf/core.py` — `_zeta_per_q_kernel._solve_one_q_and_update`

PER-Q tier: gather ONE ``(μ, μ)`` factor tile, solve that q,
scatter into ``zeta_acc``.

``q`` is a traced argument, so every iteration shares one
trace, one compile and one executable, and ``Z_col`` is never
sliced eagerly (an eager slice would materialise ``nq`` extra
``(1, μ, r/P)`` device arrays per r-chunk).  ``donate_argnums``
chains ``zeta_acc`` through the loop the same way
``_solve_batch_and_update`` does.

### `src/isdf/core.py` — `_band_norms_slice`

Slice + clamp the pseudobands weights to a ``(nb,)`` jax array.

Divisor is ``max(1, w_n)``: low-weight pseudobands keep their
sub-unit norm (DOS-preserving), high-weight ones are pulled back
to unit (no dominance), zero-weight windows stay at 1.0 since the
``max(1, 0)=1`` floor avoids a divide-by-zero.  When
``band_norms`` is ``None`` (no pseudobands), returns ``jnp.ones``.

### `src/common/wfn_transforms.py` — `<module>`

Transforms from G-flat ψ to FFT-box / r-space / centroid / r-chunk.

Composes with :class:`wfn_loader.WfnLoader`.  The loader returns
``psi`` in G-flat layout ``(n_k, nb_padded, nspinor, ngkmax)`` c128 and a
``g_index`` from :meth:`WfnLoader.box_index`; this module turns either
pair into the downstream product the consumer actually needs (FFT box,
r-space box, ψ at centroid indices, ψ on a flat-r slab).

Why split this off
------------------
g_flat is ~6-11% of the FFT-box size; band-chunked GW loops that only
need ψ at centroids should never materialise the full FFT box.  Keeping
these as standalone composable functions (rather than methods on the
loader) lets a fused-NUFFT variant land later without changing the
loader API.

Sharding contract
-----------------
Every transform **preserves the band-axis sharding** of its input ``psi``.
The default sharding from ``WfnLoader.load`` is
``P(None, ('x','y'), None, None)`` (band sharded across the 2-D mesh);
outputs add inner axes as ``P(None, ('x','y'), None, None, None, None)``
(FFT-box) or ``P(None, ('x','y'), None, None)`` (r-chunk / r-mu).  No
cross-rank communication is required by any transform.

Replicated ``psi`` (single-rank pytest, or callers passing
``sharding=None`` to the loader) goes through a non-shard_map jit fast
path so the transforms work on a laptop without a mesh.

Public API
----------
* :func:`to_box`   — G-flat → FFT-box  ``(n_k, nb, ns, nx, ny, nz)``
* :func:`to_rbox`  — G-flat → r-space FFT-box  (= IFFT(to_box))
* :func:`to_rmu`   — G-flat → ψ at centroid indices ``(n_k, nb, ns, n_rmu)``
* :func:`to_rchunk` — G-flat → ψ on flat-r slab ``(n_k, nb, ns, r_len)``

All four use the same gather kernel internally; the variants differ only
in what happens after the IFFT.

### `src/common/wfn_transforms.py` — `_cached_gindex_dev`

Cache the ``jnp.asarray(g_arr, dtype=jnp.int32)`` REPLICATED
device buffer by content hash.  See ``_GINDEX_DEV_CACHE`` comment.

Accepts either numpy ``np.ndarray`` or ``jax.Array``; if already a
jax.Array with int32 dtype the call is a no-op pass-through (see
``jnp.asarray``'s identity contract).  Callers that pass a numpy
array end up sharing the device buffer across every cache_key
variant that has the same g_index bytes — the typical case for
centroid-load + ζ-fit + V_q in a single GW run.

**Round 6 canonical-accessor path:** when the caller has access
to ``WfnLoader.box_index_dev(k, mesh)`` (the loader's cached
device-resident sphere index — the canonical buffer shared with
psi_G_store), passing that ``jax.Array`` here returns it
unchanged.  This collapses the multi-buffer steady state observed
in Round 5 (3 distinct ``(nk, nx, ny, nz) i32`` device buffers
with identical content but distinct underlying allocations) to a
single canonical buffer.

### `src/common/wfn_transforms.py` — `_resolve_gindex_dev`

Return ``(g_index_dev_jax, cache_id)`` for either a numpy array
or a ``jax.Array`` g_index — without a device→host roundtrip
when the caller has the canonical buffer in hand.

Round-6 canonical-accessor helper for transform call sites such as
:func:`gflat_to_rmu` and :func:`accumulate_rchunk_to_gflat`.
Two-path logic:

* ``jax.Array`` input → assumed to be the canonical buffer (e.g.
  ``WfnLoader.box_index_dev(k, mesh)``).  Returned unchanged;
  ``cache_id`` uses ``id(g_index)``, stable across the process
  lifetime for the canonical buffer.  ``cast`` to int32 only if
  needed (no-op when dtype already matches).
* ``numpy.ndarray`` input → goes through :func:`_cached_gindex_dev`
  (content-hash dedup → shared device buffer for matching bytes,
  keyed in a private module dict).  ``cache_id`` uses the same
  ``(hash, shape)`` tuple ``_cached_gindex_dev`` would derive.

The ``cache_id`` ends up in the ``_cached_jit`` key for the
closure-bake call sites so two unrelated g_index arrays of the
same shape don't share a compiled closure (which would silently
reuse stale baked-in indices).

Returns
-------
g_index_dev : jax.Array
    Device-resident int32 (nk, nx, ny, nz) (or other) — the
    canonical buffer if input was already a jax.Array, else the
    ``_cached_gindex_dev`` result.
cache_id : Hashable
    Identity tag suitable for use in a ``_cached_jit`` key.

### `src/common/wfn_transforms.py` — `_box_kernel`

psi: (n_k, nb, ns, ngkmax) c128 — band-sharded acceptable.
g_index: (n_k, nx, ny, nz) int32 replicated.
Returns (n_k, nb, ns, nx, ny, nz) c128 with band sharding preserved.

Pure jax — no shard_map.  Sharding propagates by XLA's normal rules:
the gather is over the G-axis (axis 3 of psi after the reshape +
transpose dance), no cross-rank op required.

### `src/common/wfn_transforms.py` — `_spec_of`

Partition spec for ``psi``, always of length ``psi.ndim``.

Returns ``psi.sharding.spec`` padded with trailing ``None`` if
JAX's ``PartitionSpec`` trimmed them off, else an all-None tuple
(fully replicated) for inputs without a ``NamedSharding`` —
e.g. single-device test arrays whose default sharding is
``SingleDeviceSharding``.

### `src/common/wfn_transforms.py` — `_local_box_fft`

Sharded local FFT for the box ``psi.shape[:-1] + (nx, ny, nz)``.

Returns a callable ``f(box) -> (i)fftn(box, axes=(-3, -2, -1))`` whose
output sharding preserves psi's leading layout with the three FFT
axes replicated.  ``kind`` is ``'fftn'`` or ``'ifftn'``.  The caller
is responsible for the mesh — pass a 1×1 trivial mesh for
single-device runs.

### `src/common/wfn_transforms.py` — `_sharding_key`

Hashable signature of psi's sharding (mesh identity + spec).

Used as part of the jit-cache key for the public transforms so two
arrays with identical mesh + PartitionSpec hit the same compiled
XLA module, while a different mesh forces a fresh compile.

``spec`` is normalized to ``psi.ndim`` length: JAX sometimes trims
trailing ``None`` entries on PartitionSpec (so ``P(None, ('x','y'))``
and ``P(None, ('x','y'), None)`` are semantically equal but compare
as different tuples).  Without the normalization the same kernel
recompiles each time JAX hands back a trimmed spec — 11 extra
``_kernel`` compiles observed at MoS2 3×3 bispinor, adding ~7 s of
wall time before this fix.  ``_spec_of`` above already does the
same normalization for the in-body sharding derivation.

### `src/common/wfn_transforms.py` — `to_box`

Scatter G-flat ψ into the FFT box.

Sharding (band axis on ``('x','y')`` or replicated) is preserved.
``g_index`` (output of :meth:`WfnLoader.box_index`) uses sentinel
``ngkmax`` to flag empty FFT-box cells (zero on gather).

### `src/common/wfn_transforms.py` — `to_rbox`

Scatter ψ → FFT box → IFFT to r-space (+ optional Bloch phase).

``norm`` is forwarded to :func:`jnp.fft.ifftn` (``'backward'`` =
``1/N``; ``'ortho'`` = ``1/√N`` on both directions, used by the
centroid pivoted-Cholesky path).  ``kvecs_frac`` (n_k, 3) optionally
applies ``exp(+2πi k·r)`` after the IFFT (set to ``None`` for the
``|ψ|²``-only path).  Output materialises the full FFT box; prefer
:func:`to_rmu`/:func:`to_rchunk` for centroid / slab consumers.

### `src/common/wfn_transforms.py` — `from_rbox`

r-space FFT box → G-sphere.  The RETURN LEG of :func:`to_rbox`.

``psi_r`` is ``(n_k, nb, ns, nx, ny, nz)``; the result is
``(n_k, nb, ns, ngkmax)`` holding the coefficients at the ``ngkmax``
crystal G-vectors in ``gvecs`` ``(ngkmax, 3)``.  Band sharding is
preserved; the three FFT axes stay replicated.

WHY THIS EXISTS.  Every local-potential matrix element
``⟨mk|V(r)|nk⟩`` is a round trip — ``to_rbox`` out, multiply by V(r),
and this function back — and until now only the outbound leg had a
sharded implementation.  ``psp.get_DFT_mtxels`` therefore hand-rolled
the return with a bare ``jnp.fft.fftn``, which is exactly what the
module comment above :func:`_local_box_fft` forbids: on a sharded
tensor XLA's planner is free to insert an all-gather and emit a
global FFT (the CrI3 6×6×1 80 Ry 121 GB OOM).  Routing the return leg
through ``_local_box_fft`` keeps the transform rank-local.

``g_mask`` ``(ngkmax,)`` zeroes pad columns.  Fixed-shape G tables
(owner decision D10) pad every k to ``ngkmax`` with ``(0,0,0)`` rows,
i.e. Γ — which is a REAL G-vector, so an unmasked gather would fold
the Γ coefficient into every pad column instead of zero.  The mask is
mandatory whenever the table is padded, not a tidiness measure.

``norm`` is forwarded to the FFT and must be the inverse convention
of the ``to_rbox`` call it undoes; the caller owns any additional
volume/grid scaling.

### `src/common/wfn_transforms.py` — `to_rmu`

ψ in r-space at the centroid FFT-grid indices ``r_mu``.

``r_mu`` is ``(n_rmu, 3)`` int32 (positions in ``[0, fft_grid[a])``);
other args as :func:`to_rbox`.  Output ``(n_k, nb, nspinor, n_rmu)``
with band-axis sharding preserved.

### `src/common/wfn_transforms.py` — `to_rchunk_inner`

Per-rank-local body of :func:`to_rchunk`: G-flat → FFT-box → IFFT
→ r-slice → optional Bloch phase.

No ``shard_map`` wrapper.  Callable from inside another shard_map's
body or a ``lax.scan`` body — the caller is responsible for any
sharding context.  Inputs and outputs are all per-rank-local arrays.

Path D scaffolding (see
``reports/zeta_rchunk_memory_model_2026-05-13/agent_2_structural_fix.md``
§4b).  Not yet wired into the production fit kernel — the consumer
refactor (§4c, rewrite of ``c_q_from_psi_sm`` / ``z_q_from_psi_sm``
with a ``lax.scan`` over bcs inside their shard_map bodies) is
deferred to a follow-up session.  This helper is checked in early
so it can be unit-tested independently.

Mathematically identical to the body of :func:`to_rchunk`; the only
difference is that this version does not enter a shard_map.

Inputs
------
psi
    Shape ``(..., ngkmax)`` c128 — caller's responsibility to have
    already partitioned data across ranks if running under a
    shard_map.  The leading axes must contain exactly 3 dims before
    the trailing ``ngkmax`` so the post-IFFT reshape lands at
    ``(..., n_rtot)`` — typically ``(nk_local, nb_local, ns,
    ngkmax)``.
g_index
    Shape ``(..., ngkmax)`` int32 — flat box indices for each
    G-vector per k.  Same broadcast contract as :func:`to_rchunk`.
fft_grid
    ``(nx, ny, nz)``.
r0
    Python int or traced int32 scalar — flat-r start index.
r_len
    Static int — width of the r slab.
norm
    FFT normalization, same conventions as ``jnp.fft.ifftn``.
kvecs_frac
    Optional ``(..., 3)`` float64 — when provided, the Bloch
    phase ``exp(+2πi k·r)`` is applied on the sliced slab via
    :func:`apply_bloch_phase_on_slice`.

Output
------
jax.Array
    Shape ``(..., r_len)`` c128.

### `src/common/wfn_transforms.py` — `to_rpoints_inner`

The arbitrary-point twin of :func:`to_rchunk_inner`.

Same body — G-flat → FFT-box → local IFFT → reshape to
``(..., n_rtot)`` → optional Bloch phase — except that the r cells
kept are the arbitrary flat FFT indices ``r_flat_idx`` rather than a
contiguous run.  This is what an orbit-packed real-grid tile needs:
the points of one symmetry orbit are scattered through the flat r
index, so no ``dynamic_slice`` can name them.

Inputs
------
psi
    Shape ``(..., ngkmax)`` c128, with exactly 3 leading axes before
    ``ngkmax`` — same contract as :func:`to_rchunk_inner`.
g_index
    Shape ``(..., ngkmax)`` int32 flat box indices per k.
fft_grid
    ``(nx, ny, nz)``.
r_flat_idx
    ``(R,)`` int32, possibly traced — the flat-r cells to keep, in
    any order.  Indices outside ``[0, n_rtot)`` are pad slots that
    the CALLER masks to zero; this function only keeps the gathers
    in bounds (it clips them, so those cells come back holding some
    other cell's finite value, not garbage and not zero).
norm
    FFT normalization, same conventions as ``jnp.fft.ifftn``.
kvecs_frac
    Optional ``(..., 3)`` float64 — when given, the Bloch phase
    ``exp(+2πi k·r)`` is applied via :func:`apply_bloch_phase_at`.

Output
------
jax.Array
    Shape ``(..., R)`` c128.

### `src/common/wfn_transforms.py` — `take_rchunk_padded`

Take a fixed-width flat-r slab, zero-filling beyond physical r.

``lax.dynamic_slice`` clamps an out-of-bounds start backward.  That is
correct for its API but wrong for a mesh-padded final r carrier: asking
for ``[r0, r0 + r_len)`` must retain those exact logical cells and append
zeros, never substitute earlier physical cells.  The r axis is local and
replicated at both callers, so this bounded gather preserves their
existing low-memory sharding and allocates only the requested slab.

### `src/common/wfn_transforms.py` — `to_rchunk`

ψ in r-space on a contiguous flat-r slab ``[r0, r0 + r_len)``.

Flat-r convention: ``r_flat = rx * ny * nz + ry * nz + rz``.  ``r0``
may be a Python int (bounds-checked) or a traced scalar (caller's
responsibility).  ``allow_padded_tail=True`` permits only the upper end
of that interval to exceed the physical FFT box; the canonical
:func:`take_rchunk_padded` body fills those carrier cells with exact
zeros.  The caller remains responsible for deriving the carrier extent
through :mod:`runtime.padding`.

The full G-flat gather → FFT-box → IFFT → r-slice (→ Bloch phase)
pipeline runs inside one ``shard_map`` region.  Keeping it
inside the manual per-rank region prevents XLA SPMD from
reconstructing the full logical band axis between ``PsiGStore``'s
``io_callback`` and the local FFT — verified to remove ~506 MiB
all-gathers from the HLO at MoS2 3×3 / 4×A100.

### `src/common/wfn_transforms.py` — `gflat_to_rchunk_aot_memory`

AOT memory breakdown of the canonical full-Bloch WFN r-slab program.

This is the planning view of :func:`to_rchunk_inner`, not an FFT-box
proxy.  It compiles the same G-flat gather -> FFT box -> local IFFT ->
flat-r slice -> Bloch-phase program used inside
:class:`common.psi_G_store.PsiGStore`, with the production shardings and
carrier extents, then asks :mod:`runtime.aot_memory` for XLA's complete
buffer peak plus the cuFFT plan workspace.

``PsiGStore`` obtains ``psi_G`` from an ``io_callback`` inside its
``shard_map``.  Here that callback result is represented as a regular
argument of the identical local program.  The AOT service includes
argument bytes in ``total``, so the callback output remains priced while
avoiding a second WFN reader or transform implementation.

A standalone IFFT measurement is insufficient for this decision: its
argument represents an already-built FFT box and therefore cannot see the
gather buffer, retained r-slab, or Bloch-phase/output buffers surrounding
the FFT.  Those buffers were enough for a 32-band CrI3 carrier to pass the
old preflight and then fail its first real cuFFT workspace allocation.

### `src/common/wfn_transforms.py` — `gflat_to_rchunk_aot_peak_bytes`

Per-rank total peak HBM for the canonical WFN r-slab program.

Compatibility view of :func:`gflat_to_rchunk_aot_memory`.  Both APIs use
the same signature-keyed compiled-memory cache, so a planner that needs
the independently placeable cuFFT workspace does not compile twice.

### `src/common/wfn_transforms.py` — `to_rmu_inner`

Per-rank-local body of :func:`to_rmu`: G-flat → FFT-box → IFFT
→ centroid sample → optional Bloch phase.

No ``shard_map`` wrapper.  Callable from inside another shard_map's
body or a ``lax.scan`` body — the caller is responsible for any
sharding context.  Inputs and outputs are all per-rank-local arrays.

Mirror of :func:`to_rchunk_inner` for the centroid-sample direction.
Mathematically identical to the body of :func:`to_rmu`; the only
difference is that this version does not enter a shard_map.

Inputs
------
psi
    Shape ``(..., ngkmax)`` c128 — caller's responsibility to have
    already partitioned data across ranks if running under a
    shard_map.  Leading axes must contain exactly 3 dims before the
    trailing ``ngkmax`` so the post-IFFT reshape lands at
    ``(..., nx, ny, nz)`` — typically ``(nk_local, nb_local, ns,
    ngkmax)``.
g_index
    Shape ``(..., ngkmax)`` int32 — flat box indices for each
    G-vector per k.  Same broadcast contract as :func:`to_rmu`.
fft_grid
    ``(nx, ny, nz)``.
r_mu
    Shape ``(n_rmu, 3)`` int32 — FFT-grid coordinates of the
    centroid sample points.
norm
    FFT normalization, same conventions as ``jnp.fft.ifftn``.
kvecs_frac
    Optional ``(n_k, 3)`` float64 — when provided, the Bloch
    phase ``exp(+2πi k·r)`` is applied to the full FFT box via
    :func:`apply_bloch_phase` before the centroid gather (same as
    :func:`to_rmu`'s body).

Output
------
jax.Array
    Shape ``(..., n_rmu)`` c128.

### `src/common/wfn_transforms.py` — `gflat_to_rmu`

ψ(G-flat) → ψ at centroid grid points, fused over all (k, n).

Inverse-direction mirror of :func:`accumulate_rchunk_to_gflat`:
same scan-inside-shard_map scaffolding (XY-band/μ sharding,
flat-axis pad + scan in chunks of ``cs``, per-row body, truncate
output to N), differing only in (a) FFT direction — IFFT here,
FFT in the ζ writer; (b) phase site — post-gather separable
1D-Bloch here, pre-FFT phase-on-slab in the ζ writer; (c) gather
target — centroid grid cells here, G-sphere cells in the ζ writer.
Together they form the ψ↔ζ G-flat round-trip primitive.

Shapes / shardings (mesh = ``('x', 'y')`` of size ``P = p_x · p_y``)::

    psi_G   : (nk, nb_total, ns, ngkmax)  P(None, ('x','y'), None, None)
    return  : (nk, nb_total, ns, n_rmu)   P(None, ('x','y'), None, None)

``nb_total`` must be divisible by ``mesh.size``.  Each rank owns a
``nb_local = nb_total / P`` block of bands across the full
``(nk, ns, ngkmax)`` extent and writes the same band block to the
output.

Algorithm — inside a single ``shard_map`` over ``('x','y')``:

  1. Flatten per-rank ``(nk, nb_local) → N = nk · nb_local`` rows so
     every row is a single ``(k, n)`` pair.
  2. Zero-pad ``N → ⌈N / cs⌉ · cs`` so the chunk count is exact.
  3. ``lax.scan`` over chunks of ``cs`` rows.  Each iteration:
     a. ``k_row[cs] = (i·cs + arange(cs)) // nb_local`` —
        which k each row belongs to.  Clipped to ``[0, nk)``;
        padding rows land on k = nk - 1 but their data is zero
        (zero-pad) so the centroid samples they produce are
        zero and get truncated in ``out_flat[:N]``.
     b. ``_box_kernel(sub[cs, 1, ns, ngkmax], g_index[k_row])``
        scatters G-sphere coeffs into a per-row FFT box
        ``(cs, 1, ns, nx, ny, nz)``.  Singleton ``nb`` axis is
        squeezed.
     c. ``jnp.fft.ifftn`` on the trailing 3 axes — per-rank-local
        cuFFT, no resharding.
     d. Gather centroid cells: ``rb[:, :, r_mu[:,0], r_mu[:,1],
        r_mu[:,2]]`` → ``(cs, ns, n_rmu)``.
     e. Optional per-row Bloch phase ``exp(+2πi k·r_mu)`` applied
        on the *gathered cells only* (not on the full box) —
        ``apply_bloch_phase``-equivalent under the gather, but
        scratch drops from ``(cs · n_rtot)`` to ``(cs · n_rmu)``.
     f. ``dynamic_update_slice_in_dim`` writes the row block into
        ``out_flat`` at offset ``i·cs``.

Chunking is on the flat ``(k · n_local)`` axis so the chunk size
is a free integer — no divisibility constraint on either nk or
nb_local.  Defaults to one-shot (``cs = N``); the scan compiles to a
single iteration that XLA folds away.  Per-iteration FFT-box
transient is ``cs · ns · n_rtot · 16`` bytes — choose ``chunk_size``
to bound this against the per-rank HBM budget.

Parameters
----------
psi_G
    ``(nk, nb_total, ns, ngkmax)`` c128, band-flat-sharded.
g_index
    ``(nk, nx, ny, nz)`` int32 — flat-FFT-box indices.  Sentinel
    value ``ngkmax`` flags empty box cells (zero on gather).
    Replicated.
r_mu
    ``(n_rmu, 3)`` int32 — FFT-grid coordinates of the centroid
    sample points.  Replicated.
mesh
    Process mesh with named axes ``'x'`` and ``'y'``.
fft_grid
    Static ``(nx, ny, nz)``.
kvecs_frac
    Optional ``(nk, 3)`` fractional k-vectors.  When given, the
    gathered samples are multiplied by ``exp(+2πi k·r_mu)`` per
    ``(k, r_mu)`` pair (separable in x/y/z; pre-computed once per
    k).  ``None`` skips the phase.
k_row_map
    Optional ``(nk,)`` integer row selector.  Row ``k`` of ``psi_G``
    then uses ``g_index[k_row_map[k]]`` and, when present,
    ``kvecs_frac[k_row_map[k]]``.  The selector is a runtime operand so
    every q row shares one executable and the loader's cached full
    FFT-box table is never copied/reordered outside this owner.  The
    default ``None`` preserves the historical row-aligned path.
norm
    Forwarded to :func:`jnp.fft.ifftn`.  Defaults to ``"backward"``
    to match the legacy :func:`to_rmu` default; centroid-load
    callers typically pass ``"ortho"``.
chunk_size
    Rows per scan iteration along the flat ``(k · n_local)``
    axis.  Default ``None`` ⇒ one-shot.  Memory bound:
    ``chunk_size · ns · n_rtot · 16 B`` for the per-iteration FFT
    box.

Returns
-------
``(nk, nb_total, ns, n_rmu)`` c128 with ``P(None, ('x','y'),
None, None)`` sharding.

### `src/common/wfn_transforms.py` — `accumulate_rchunk_to_gflat`

Add ``FFT(pad(phase(rchunk)))[sphere_idx]`` into ``gflat_acc``.

Inverse-direction mirror of :func:`gflat_to_rmu`: same
scan-inside-shard_map scaffolding (XY-band/μ sharding, flat-axis
pad + scan in chunks of ``cs``, per-row body, truncate output to
N), differing only in (a) FFT direction — FFT here, IFFT in the ψ
reader; (b) phase site — pre-FFT phase-on-slab here, post-gather
separable 1D-Bloch in the ψ reader; (c) gather target — G-sphere
cells here, centroid grid cells in the ψ reader.  Together they
form the ψ↔ζ G-flat round-trip primitive — the user-spec mandate
that "ζ and ψ infrastructure are the same except how ngkmax is
padded in the G-sphere."  The padding difference is at the loader
layer (per-q ζ ngk vs per-k ψ ngk), not in this kernel.

Shapes / shardings (mesh = ``('x', 'y')`` of size ``P = p_x · p_y``)::

    rchunk    : (n_q, n_rmu_padded, r_len)   P(None, ('x','y'), None)
    gflat_acc : (n_q, n_rmu_padded, ngkmax)  P(None, ('x','y'), None)

``n_rmu_padded`` must be a multiple of ``mesh.size`` (rounded up at
:class:`common.meta.Meta` construction).  Each rank owns a
``n_mu_local = n_rmu_padded / P`` block of μ-rows over the full
``(n_q, r_len)`` extent.

Algorithm — inside a single ``shard_map`` over ``('x','y')``:

  1. Flatten ``(n_q, n_mu_local) → N = n_q · n_mu_local`` rows.
  2. Zero-pad ``N → ⌈N / cs⌉ · cs`` so the chunk count is exact.
  3. ``lax.scan`` over chunks of ``cs`` rows.  Each iteration:
     a. ``q_row[cs] = (i·cs + arange(cs)) // n_mu_local`` —
        which q each row belongs to.  Clipped to ``[0, n_q)``;
        padding rows land on q = n_q - 1 but their data is zero
        (zero-pad) so the contrib they produce is zero — no
        contamination of ``acc``.
     b. Slab → FFT box via ``dynamic_update_slice_in_dim`` at
        offset ``r0``, or, on the ``r_indices`` path, via a gather
        through the inverse slot table (see the kernel body).
     c. Per-q Bloch phase, if ``qvec_frac`` given: separable
        ``exp(-2πi q · r)`` factors, pre-computed per q at trace
        time, gathered per row by ``q_row``.
     d. ``jnp.fft.fftn`` on the trailing 3 axes — data is
        per-rank-local on the entire FFT box (FFT axes are
        replicated by sharding contract), so plain ``fftn`` runs
        local cuFFT with no resharding.
     e. ``jnp.take_along_axis`` gathers the per-q sphere indices
        (``sphere_idx[q_row]``) along the trailing flat-r axis.
     f. Accumulate into ``acc`` at offset ``i·cs``.

Chunking is on the flat ``(q · μ_local)`` axis so the chunk size
is a free integer — no divisibility constraint on either n_q or
n_mu_local.  Defaults to one-shot (``cs = N``); the scan compiles
to a single iteration that XLA folds away.

Parameters
----------
rchunk
    ``(n_q, n_rmu_padded, r_len)`` c128, μ-flat-sharded.
gflat_acc
    ``(n_q, n_rmu_padded, ngkmax)`` c128, μ-flat-sharded.
    Donated by the inner jit — its buffer is reused in place.
mesh
    Process mesh with named axes ``'x'`` and ``'y'``.
fft_grid
    Static ``(nx, ny, nz)``.
r0
    Python int or jax-scalar — flat-r start of the slab in
    ``[0, nx·ny·nz)``.  Exactly one of ``r0`` / ``r_indices``.
r_indices
    ``(r_len,)`` int32 device array — the arbitrary flat-r cells the
    slab holds, for an orbit-packed real-grid tile whose points are
    not contiguous in the flat r index.  Entries outside
    ``[0, nx·ny·nz)`` are pad slots; the scatter drops them and the
    phase lookup clips them.  Exactly one of ``r0`` / ``r_indices``.
    The indices must be DISTINCT, pad sentinels included (the caller
    gives each pad slot its own out-of-range value): the scatter is
    lowered with ``unique_indices=True`` so no atomic path is needed.
sphere_idx
    ``(n_q, ngkmax)`` int32 flat-FFT indices.  Every q has the
    same ``ngkmax`` axis length with potentially different index
    lists per q; pad slots within a row use a sentinel flat-FFT
    index whose coeffs the caller zeroes post-loop.
qvec_frac
    Optional ``(n_q, 3)`` fractional q-vectors.  When given, the
    FFT box is multiplied by ``exp(-2πi q · r)`` per row
    (separable in x/y/z; pre-computed once per q).
norm
    Forwarded to :func:`jnp.fft.fftn`.
chunk_size
    Rows per scan iteration along the flat ``(q · μ_local)``
    axis.  Default ``None`` ⇒ one-shot.  Memory bound:
    ``chunk_size · n_rtot · 16 B`` for the per-iteration FFT box.

Returns
-------
Updated ``gflat_acc`` (same shape, same sharding).

### `src/common/wfn_transforms.py` — `apply_bloch_phase`

box × exp(sign · 2πi k·r) applied as three separable 1D multiplies.

``box``: trailing shape ``(..., nx, ny, nz)`` c128 (sharding preserved).
    The leading axis must be the k-axis whose length matches
    ``kvecs_frac.shape[0]``.  Any number of intermediate broadcast
    axes are supported (e.g. band, spinor) as long as the spatial
    axes are the last three.
``kvecs_frac``: ``(n_k, 3)`` fractional k-vectors.
``sign``: ``+1`` for the ψ post-IFFT case; ``-1`` for the ζ pre-FFT
    case (``z_q,μ(r) = exp(-2πi q·r) · ζ_q,μ(r)``).

### `src/common/wfn_transforms.py` — `apply_bloch_phase_at`

``slab × exp(sign·2πi k·r)`` at arbitrary flat-r grid points.

The arbitrary-point twin of :func:`apply_bloch_phase_on_slice`, which
is now a thin wrapper over this function.  Its slab cells are the
contiguous run ``[r0, r0 + r_len)``; here they are whatever flat FFT
indices ``flat_idx`` names, in whatever order — the layout an
orbit-packed real-grid tile has, where the points of one symmetry
orbit are scattered through the flat r index.

Flat-r convention matches :func:`to_rchunk`:
``r_flat = rx · ny · nz + ry · nz + rz``.

``slab``: trailing shape ``(..., r_len)``.  Sharding preserved.
``flat_idx``: ``(r_len,)`` int32, possibly traced.  Entries outside
    ``[0, nx·ny·nz)`` are inert carrier cells; their phase lookup is
    clipped into range so every gather stays in bounds, exactly as
    the contiguous version does for a padded slab tail.  Valid
    coordinates are unchanged bit-for-bit.

### `src/common/wfn_transforms.py` — `apply_bloch_phase_on_slice`

``slab × exp(sign·2πi k·r)`` over a contiguous flat-r slab.

Flat-r convention matches :func:`to_rchunk`:
``r_flat = rx · ny · nz + ry · nz + rz``.

Where :func:`apply_bloch_phase` builds the phase over the full FFT
box and relies on the caller to slice the result, this helper
builds the phase only on the requested slab ``[r0, r0 + r_len)``.
Mathematically identical (IFFT + multiply commutes with slicing
along r); operationally important when the slab is much smaller
than the full box — pulls per-r-cell work from
``n_k × nx · ny · nz`` down to ``n_k × r_len``.

The contiguous run is just one index list, so the body is a call to
:func:`apply_bloch_phase_at` with ``r0 + arange(r_len)`` — the same
clip and the same ops it always did.

``slab``: trailing shape ``(..., r_len)``.  Sharding preserved.
``r0``: Python int or a jax scalar (traced).  When traced, callers
    are responsible for the bounds check.
``r_len``: static int — slab length.

### `src/common/wfn_transforms.py` — `_refuse_spinor_zero_fill`

Refuse a ψ spinor-extent mismatch instead of zero-filling it.

This is NOT a mesh-divisibility pad and it never was, which is why it
gets a refusal rather than a parity story.  ``meta.nspinor`` is set
CATEGORICALLY — ``4 if bispinor else wfn.nspinor`` (``common/meta.py``)
— so it is never rounded up to a device count and there is no divisor
to be inert with respect to.  A 2→4 mismatch means exactly one thing:
the caller asked the loader for the 2-component ψ (``bispinor=False``,
the kwarg default) while running under a ``bispinor = true`` deck.

Zero-filling components 2 and 3 is not an inert pad; it DELETES the
small components, whose correct value is ``(α/2)(σ·(k+G)) ψ_L``
(``common/bispinor_init.lift_to_4spinor``).  Every consumer contracts
the spinor axis (``einsum('msg,nsg->mn')``, ``Σ_s|ψ|²``), so the loss
is silent and algebraically well-formed: ρ, V_H and every
⟨mk|V_H|nk⟩ come out built from large components only, ~4e-4 relative
at 30 Ry, under ``build_hartree_potential``'s 1e-3 ∫ρ tolerance.  A
wrong number under every tolerance that would have caught it is the
exact trade the pad-everywhere program exists to avoid, so the branch
refuses.

The fix at a call site is to pass ``bispinor=True`` so the loader
LIFTS (producing real small components), never to widen the array.

### `src/common/wfn_transforms.py` — `load_kpoint_fftbox_local`

One k-point's ψ in the FFT box, **process-local**.

Returns ``(nb - b_lo, nspinor, nx, ny, nz)`` c128 on
``jax.local_devices()[0]`` — see
:meth:`wfn_loader.WfnLoader.load_process_local` for the
single-device contract and why the k-parallel kernels need it.

``b_lo`` selects a band sub-window ``[b_lo, nb)``; the default
``b_lo=0`` reproduces the legacy "first ``nb`` bands" behaviour.
Band sub-windows are what lets the ρ sweep in ``gw.kin_ion_io``
add band-parallelism on top of k-parallelism when the rank count
exceeds ``nk``.

Memory: ``(nb - b_lo) · nspinor · nx·ny·nz · 16 B`` — 0.55 GiB for
the MoS₂ 12×12 at 120 bands, which is why the callers stream k
(and, past P > nk, band chunks) rather than materialising all of it.

### `src/common/wfn_transforms.py` — `load_kpoint_fftbox`

Load a single k-point's wavefunction into the FFT box on GPU.

Returns jax array of shape (nb, nspinor, nx, ny, nz), ~0.55 GiB for 12x12.

Thin back-compat wrapper over :func:`load_kpoint_fftbox_local`;
``sym`` is unused (the loader's full-BZ unfold is internal to
``load_process_local(k=[k_idx])``) and kept for caller-API
compatibility.

Values are unchanged for every existing (single-process) caller: at
``P=1`` ``jax.local_devices()[0] is jax.devices()[0]`` and the
mesh-less loader's ``load(..., sharding=None)`` took the same
``_eager_build`` path ``load_process_local`` takes.  At ``P>1`` the
old body was silently wrong (it boxed a band-SHARDED ψ against a
1×1 mesh pinned to process 0's device); the delegation fixes that.

### `src/common/wfn_transforms.py` — `get_enk_bandrange`

Return band energies and per-band weights for a given band window.

Args:
    wfn: WFNReader providing energies and Fermi level
    sym: SymMaps with mappings between irreducible and full k sets
    bandrange: tuple[int,int] inclusive-exclusive (start, end) bands to extract
    sigma_bandrange: tuple[int,int] band window used to compute weighting
    nspinor: Spinor components widthing the WEIGHTS axis only (2 for
        Pauli, 4 for bispinor); None reads ``wfn.nspinor``.  ``enk``
        does not depend on it — a hardcoded default of 2 was
        silent-wrong for the weights on an nspinor=1 file.

Returns:
    enk: jax.Array of shape (nk_full, nb)
    weights: jax.Array of shape (nk_full, nb * nspinor) with simple val/cond weights

────────────────────────────────────────────────────────────────────────
NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL.
────────────────────────────────────────────────────────────────────────
Everything in this function operates on tiny host-side arrays
(nk × nb ~ a few thousand doubles).  Using ``jnp`` would force each
reduction/where/repeat to be dispatched as its own pjit at trace time
— ~16 standalone pjit compilations per run, for zero runtime benefit
(the arithmetic is ms-scale on host).  Rewriting to ``jnp`` reverses
a deliberate compile-cache trim (commit 31b5961, 2026-04-18).

Only cast to ``jax.Array`` at return so the caller gets the pytree
type it expects.  Do NOT "fix" this back to ``jnp``.
────────────────────────────────────────────────────────────────────────

### `src/common/wfn_transforms.py` — `read_Gvecs_to_devices`

G-space wfns on a 2-D mesh, band-sharded, scattered to FFT box.

Returns ``(global_psi_Gtot, nb_logical)`` where ``global_psi_Gtot``
has shape ``(nk, nb_padded, nspinor, nx, ny, nz)`` sharded
``P(None, ('x','y'), None, None, None, None)``.

The body is thin: :class:`wfn_loader.WfnLoader`
+ :func:`to_box`.  Symmetry unfold, τ-phase, TR conjugation, spinor
rotation, band-axis padding/sharding, and the bispinor lift all happen
inside ``WfnLoader.load``.  ``sym`` is unused (the loader builds its own
SymMaps lazily); kept in the signature so existing callers don't change.

Memory note: this function still materialises the FFT-box representation
for caller back-compat.  The g_flat path (:meth:`WfnLoader.load`
directly) is ~6-11% the size of the FFT box.

### `src/common/wfn_transforms.py` — `load_psi_gflat_padded`

One capped + zero-padded ψ(G-flat) load — THE shared load dance.

Single source for the load → cap-at-file-nbands → zero-pad-band-axis
→ reapply-sharding sequence that was previously triplicated (with
drift) across ``psi_G_store._populate_from_loader``,
``load_centroids_band_chunked`` and ``iter_psi_rchunk_bandwise``.

Past-mnband contract (``common/meta.py:100-117``): ``b_id_4 =
round_up(b_id_4_user, world_size)`` may exceed the file's ``mnband``
(CrI3 6×6 30Ry SOC: mnband=86, world_size=16 ⇒ b_id_4=96).
``WfnLoader.load`` rejects ``b_hi > nbands``; this helper caps the
loader call at ``loader.nbands`` and zero-pads the band axis back up
to ``max(pad_to, b_hi - b_lo)``, preserving ``sharding``.  Pad rows
are physically zero — the same contract every call site already
promised downstream.

Returns ``None`` when the ENTIRE requested window starts at/past the
file's band extent (an all-pad chunk) — the caller decides whether
that is a zero-fill (psi_G_store tiles) or an error (a user window
past EOF on a primary load).

### `src/common/wfn_transforms.py` — `prepare_rchunk_carrier`

Plan and finish the canonical band-sharded r-chunk carrier.

This is the one owner for the r-range validation, runtime-padding
divisor, terminal-tail rule, and staged band-product → r-product move
shared by direct WFN iteration and reusable coefficient sources.  The
transform that produces the already-zero-filled carrier remains the
caller's responsibility.

Returns ``(carrier_extent, output_sharding, finish)``.  ``finish`` checks
the produced extent and commits it to the requested layout; when
``product_r_spec`` is the canonical product-r layout it invokes
:func:`common.staged_reshard.band_to_product_r_reshard`.

### `src/common/wfn_transforms.py` — `iter_psi_rchunk_bandwise`

Generator: yield ``(bc_range, psi_bc_r)`` one band chunk at a time.

By default each result has shape ``(nk, bc, ns, r_end-r_start)`` sharded
``P(None, None, None, 'y')``, preserving the historical contract.  When
``product_r_spec=P(None,None,None,('y','x'))``, the free r axis is first
rounded through :mod:`runtime.padding` to that spec's canonical divisor;
:func:`take_rchunk_padded` emits the exact-zero carrier tail inside the
existing FFT shard-map, and the result moves directly from the input's
product-band layout to the product-r layout by
:func:`common.staged_reshard.band_to_product_r_reshard`.  That route is
two volume-preserving all-to-alls and never constructs the historical
x-replicated r-on-y global carrier.  The caller is responsible for
accumulating contributions (e.g. ``P += einsum(ψ_L_bc, ψ_R_bc)``)
so only one band chunk's r-chunk shard is live at any moment,
decoupling the pair-density peak from the total band count.

``band_chunk_ranges`` lets the caller dictate chunk boundaries —
pass a list to respect left/right pair-density endpoints so every
yielded chunk lies fully inside one (or both) of those ranges and
no out-of-range einsums ever dispatch.  When None, contiguous
chunks of ``band_chunk_size`` are built from ``band_range``.

``band_pad_to`` zero-pads every yielded chunk's band axis up to a
single uniform width (the full chunk width) BEFORE ``to_rchunk``,
so the ``to_rchunk`` shard_map — keyed on ``psi.shape`` — sees ONE
band-dim across the whole sweep and compiles exactly ONCE instead
of once-per-distinct-remainder-width.  The pad rows are physically
zero, so a caller that accumulates a band-contraction (the Galerkin
``UH_bc @ psi`` fold) must slice/zero-pad its contraction operand to
the same width — the extra bands then contribute exactly zero.
``None`` disables the pad (legacy per-chunk-shape behaviour).

Uses :class:`wfn_loader.WfnLoader` + ``to_rchunk``.  ``sym``
is unused (loader builds its own SymMaps).

### `src/common/wfn_transforms.py` — `load_centroids_band_chunked`

Load centroid-sampled wavefunctions using band AND k-point chunking.

Memory-safe version that loops over band chunks (and optionally k-point
chunks) to avoid OOM when loading all bands/k-points at once for FFT.

The FFT box array psi_Gtot_local has shape (nk, nb, nspinor, *fft_grid)
and scales as O(nk * nb * n_rtot). For large k-grids (e.g. 10x10x10 =
1000 k-points), this exceeds GPU memory. K-chunking processes a subset
of k-points at a time, accumulating only the centroid-space outputs
(which are O(nk * nb * n_rmu) — much smaller since n_rmu << n_rtot).

Args:
    wfn: WFNReader
    sym: SymMaps
    meta: Meta object
    centroid_indices: (n_rmu, 3) centroid grid coordinates
    bispinor: Whether to use bispinor
    mesh_xy: Device mesh
    band_range: (b_start, b_end)
    band_chunk_size: Maximum bands in one outer WFN/FFT tile.  A
        positive value smaller than the logical window activates the
        streamed owner; this is the existing GW Stage-A band policy.
    k_chunk_size: Maximum k-points in one outer WFN/FFT tile.  ``None``
        or a non-positive value keeps all k-points inside each band tile.
        A positive value activates k streaming even when the band window
        fits one tile.
    psi_G_flat: Optional already-loaded full G-flat window.  This keeps
        the bulk path regardless of chunk hints because its owner (the
        htransform Galerkin path) deliberately reuses the same allocation
        after centroid sampling.
    k_domain: ``"full_bz"`` (default) or ``"ibz"``.  The latter samples
        the raw WFN parent rows without symmetry unfolding.  It exists for
        parent-k contractions; it is not a second FFT implementation.
    return_ibz_parents: With ``k_domain="full_bz"``, additionally retain
        raw-parent centroid faces while the existing parent-major stream
        already holds each raw WFN row.  Returns
        ``(full_y, full_x, parent_y, parent_x)``.  The four-component lift
        remains on the full-k path until its exact operator representation
        is owned by the symmetry service.

    Both outer remainders are zero-padded through
    :mod:`runtime.padding` to the one fixed physical tile shape.  G-index
    and k-vector values are runtime operands, so all tiles of a window
    reuse one compiled transform family; padding changes scheduling only,
    not the logical return extent or centroid values.

Returns:
    psi_rmu_Y: (nk, nb, ns, n_rmu) with P(None, None, None, 'y')
    psi_rmuT_X: (nk, n_rmu, nb, ns) with P(None, 'x', None, None)

n_rmu divisibility: the kernels here shard the n_rmu axis by a
single mesh axis (``'x'`` alone in psi_rmuT_X, ``'y'`` alone in
psi_rmu_Y) — so n_rmu only needs to divide one axis size, not the
product.  ``mesh.x = mesh.y = 4`` and 668 / 4 = 167 ✓; no padding
needed at this layer.  The undivisibility shows up only at the
V_q read where the trailing axis is sharded by the *product*
``('x', 'y')`` = 16; SlabIO's auto-pad on the on-disk dataset
closes that gap (see ``file_io.slab_io.create_dataset``).
