# ζ fitting on raw-parent faces

`gw.gw_init.fit_zeta` decides reuse separately for each centroid family: the
charge family, and under `bispinor` the three current channels. Each fresh fit
calls `gw.isdf_fitting.fit_zeta_to_h5` once per family (`output_files` maps
μ_L to its file: the charge channel alone, or the missing current channels
together), which requires a typed `CentroidKUnfoldPlan` and both packed
raw-parent faces. The fit has no full-k fallback.

This page owns three parts of the fit:

- the normal equations: C_q, the pad diagonal, the LR+RL completion and the
  q selection;
- the factor and back-solve tiers;
- centroid order in memory and on disk.

Every right-hand side runs on [route G](zeta_fit_mubatch.md), which also owns
the per-tile solve seam. The equations are on the
[ISDF page](../theory/isdf-zeta-vq.md).

## Carriers and band windows

Each centroid family owns its basis, its parent plan and two un-conjugated
faces sampled at the raw parents k̄:

```text
psi_mun[parent, spin, mu, band]        psi_nmu[parent, band, spin, mu]
```

`low_mem_bands` ([input reference](../input_reference.md)) selects between two
layouts that solve the same equations:

- **`true`, the face layout (default).** The fit keeps mesh-face copies and
  distributes the band contraction.
- **`false`, the axis layout.** The fit keeps single-axis centroid copies
  with complete bands and contracts bands locally.

Canonical ζ files do not encode the layout. The left and right band windows
are 0/1 weights over the loaded, mesh-padded band extent, so a window edge
need not divide the mesh. C_q and Z_q use the same weights. Pseudobands are
refused (`NotImplementedError`).

## C_q on the parents

`isdf.core.c_q_from_psi_sm` forms the metric with one planned GEMM per raw
parent and endpoint (`distrib_la.gemm_plan`, M = N = μ·ns, K = n_b):

```text
D^X_k̄(μ a, ν b) = Σ_n w^X_n ψ_{n k̄ a}(r_μ) conj ψ_{n k̄ b}(r_ν)          X = L, R
C_q(μ, ν)       = Σ_k Σ_ab D^L_{k,ab}(μ, ν) conj D^R_{k+q,ab}(μ, ν)
```

The k-convolution router's parent-load convolution
(`ffi.fft.make_fused_conv_kparent`, centroid-major) carries out the typed
transport k̄ → k: the centroid permutation, the lattice-wrap phase, the spinor
action and antiunitary conjugation, all read from `symmetry_maps` through the
plan. It then evaluates the k correlation and writes C_q for all N_k q at
`P(None,'x','y')`.

A current channel applies its vertex γ̃^i on the output spin indices after
transport. The stored faces are never vertex-folded. The four-spinor action is
on the [symmetry register](symmetry_register.md), and the Lorentz convention
is in [four-current wiring](four_current_wiring.md).

Cost per fit: 2·n_p·ns²·μ²·n_b MACs of GEMM, divided over the mesh by the
plan, plus O(μ²·ns²·N_k log N_k / P) for the correlation. C_q takes
N_k·μ²·16/P bytes per rank before the q selection.

The GEMM plan is built with `warmup=False`, so C's enclosing JIT runs the
first GEMM ([`distrib_la`](../services/distrib_la.md#gemm-plans-without-dummy-execution)).

## Normal equations

`fit_zeta_to_h5` prepares C_q on all N_k q rows in three steps before
factoring.

1. **Pad diagonal.** Orbit-packed centroids interleave pad slots on every
   shard, so the dense solve extent is the whole carrier
   (`meta.mu_solve_extent`). The pad rows and columns of C_q are exactly
   zero. The fit adds the mean diagonal tr C_q/μ to the pad diagonal
   (`add_pad_diagonal_sharded`, formed on each rank's own tile). The pad
   eigenvalues then lie inside the active spectrum, and a unit pad cannot
   become λ_max and set the rank cut. Z's pad rows stay zero, so ζ is zero on
   the pads.
2. **LR+RL completion.** The charge fit trains on asymmetric windows:
   L holds every occupied state plus the Σ conduction window, and R holds
   the Σ occupied window plus every empty state. Complex conjugation swaps
   the ordered endpoints, so the LR pairs alone are not closed under
   conjugation. Relabelling (n, m, k) gives N_RL(q) = conj N_LR(−q) for C and
   for Z alike. `complete_ordered_pair_normal_equations` therefore adds
   conj C_{−q} to C_q, using the q involution from
   `symmetry_maps.q_negation_index`, and Z receives the same completion. This
   is the normal equation of the concatenated training set, not a projection
   of ζ, V or W.
3. **q selection.** When the run stores the irreducible wedge and the
   centroid set is orbit-closed under the full space group with time reversal
   (`gw.qgrid_symmetry.resolve_qgrid_symmetry_tables`), C_q is sliced to the
   IBZ rows before factoring. When the set is not closed, the charge channel
   stores every q, and a current channel refuses: its V_q assumes IBZ ζ.
   Regenerate the current centroids with
   `centroid.kmeans_cli --density-mode current`.

## Factor and back-solve

**Charge.** `isdf.cplus.factor` computes one dense eigh per q and returns
B = V_keep Λ_keep^{-1/2}, so B Bᴴ = C⁺. It keeps λ > `zeta_rcond`·λ_max
(default 1e-8). The cut is moved off any degenerate multiplet it would split.
A binding cut is certified against κ ≤ 1e8 and against a discarded spectral
weight of at most 1e-3 (`LORRAX_RANK_POLICY` = refuse | warn | off)
([rank-truncation policy](../dev/rank_truncation_policy.md)).

The factor runs replicated, in q batches of at most 4 GiB per rank
(`factor_c_q_replicated_batched`). On P > 1 with Q·μ³ ≥ 5e9 it runs
q-parallel instead, with each rank factoring ceil(Q/P) whole matrices. The
cost is O(Q·μ³), divided by min(P, Q) when q-parallel. A q batch that cannot
be replicated refuses, and the refusal names the per-batch bytes.

**Back-solve tier** (`_resolve_zeta_gather`):

- **`local`.** Rank p holds ceil(Q/P) whole factors (`zeta_factor_resident`
  moves them to batch layout once), and only the right-hand side moves.
- **`replicated`.** The whole stack is replicated on every rank.

The tier is automatic on every layout (`zeta_auto_tier`, read by the resolver
and the planner alike). It is `local` when ceil(Q/P)·P ≤ 2Q, or when Q·μ²·16
exceeds `LORRAX_ZETA_GATHER_CAP_GIB` (default 4 GiB), and `replicated`
otherwise, which covers a few q on many ranks. Both tiers apply the same
whole-tile factor, bit for bit. `linalg` does not select the tier:
`linalg = distributed` distributes the W Dyson solve, the transverse LU and
the eigensolvers, never the ζ back-solve. Route G applies the factor tile
by tile ([finalize](zeta_fit_mubatch.md#finalize-v_q-and-the-head-columns)).

**Current channels.** C_q^i is a Hermitian indefinite, signed Gram. The fit
factors C + δI once per channel with pivoted LU, where
δ = 1e-12·sign(Re tr C)·|tr C|/μ (`_transverse_lu_ridge`). The sign keeps the
pairing (sC + sδI)⁻¹(sZ) = (C + δI)⁻¹Z. The indefinite solve always runs at
the logical extent (`runtime.padding.solve_at_logical`), because pad-extent LU
round-off is amplified O(1) in the near-null current modes.

Route G applies the factor per G tile, so it is always the local whole-tile
JAX LU (`factor_c_q` with `batch_reshard`), laid out on the q owners by the
back-solve tier like the charge factor; a resolved provider LU is hoisted to
it, and `linalg = distributed` no longer selects a block-cyclic token here.
The conditioning instrument κ_lb = max|u_ii|/min|u_ii|, a lower bound on κ,
refuses above 1e12 under the same policy.

## One centroid order; canonical files

Every in-memory centroid axis uses its family's `PackedCentroidBasis`, in
which whole orbits share a shard and each shard ends in exact-zero pads
([mesh-padded axes](padding.md#orbit-packed-runtime-centroids)). Files keep
the canonical centroid-file order at the logical extent. Readers pack, and
writers unpack, at the I/O seam only (`mu_basis.unpack_axis`,
`unpack_operator`). This covers ζ, the parent restart faces, and the V, W and
MPA stores, so no file shape depends on the process grid.

## Unreduced admission for nonclosed centroid sets

When a charge or current centroid set is not orbit-closed, the driver warns
and replaces the symmetry maps with `SymMaps.trivial_view()` before either
family is packed. The view keeps the identity operation only and uses no
antiunitary action; the measured time-reversal verdict is unchanged. `parent_k_domain = "full_bz"`, so the loader supplies every full-zone
k as its own parent. The plan then has n_parent = N_k, identity actions and
every q row. The same parent route and route G run unchanged. The binding
ruling is in [decisions](decisions.md).

## Verification

- **`tests/test_isdf_zq_parent_parity.py`** compares the parent C_q and Z_q
  with direct NumPy q and band sums on typed full-k children. It uses a
  nonsymmorphic glide, spin mixing and antiunitary rows at ns = 1, 2, 4, and
  all three current vertices, at 1e-10.
- **`tests/test_parent_projector_unfold_oracle.py`** checks the parent → child
  projector transport against an independent oracle.
- **`tests/multi_device/zeta_mubatch_p4.py`** gates route G ([route G,
  verification](zeta_fit_mubatch.md#verification)).
