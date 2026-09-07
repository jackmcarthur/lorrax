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
