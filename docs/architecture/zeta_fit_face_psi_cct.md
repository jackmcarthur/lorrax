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
stream bounded child stars; the SC density rebuild loads IBZ G-sphere
parents and projects through typed scalar/polar actions. Neither requires a
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
