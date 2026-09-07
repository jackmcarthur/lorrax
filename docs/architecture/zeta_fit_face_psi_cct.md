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
