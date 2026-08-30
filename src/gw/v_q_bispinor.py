"""Bispinor V_q^{μ_L, ν_L} orchestrator — thin loop over the unified tile kernel.

Conventions (``common.gamma_matrices``): the stored matrices are
γ̃^μ ≡ γ^0 γ^μ, so γ̃^0 = I_4 and γ̃^i = α^i on 4-spinors, and every
channel density is written with ψ† (never ψ̄):

    ρ^{μ_L}(r) = ψ† γ̃^{μ_L} ψ ≈ Σ_a ζ_{μ_L,a}(r) ρ^{μ_L}(r_a)

The μ_L = 0 channel is the charge density; the μ_L ∈ {1,2,3} channels
are CURRENT densities ψ† α^i ψ (not spin densities — α^i couples the
large and small bispinor components).  We work in Coulomb gauge, where
the photon propagator couples the four channels by

    V^{μ_L,ν_L}_q(μ,ν) = Σ_K  ζ̄_{μ_L,μ}(K) · d^{μ_L,ν_L}(K) · v(K) · ζ_{ν_L,ν}(K)
    d^{0,0}      = 1
    d^{i,j}      = −(δ_ij − K̂_i K̂_j)       (spatial metric × projector)
    d^{0,i}=d^{i,0} = 0                    (Coulomb-gauge cross term)

so out of the 16 (μ_L, ν_L) blocks:
    6 zero by gauge — never computed.
    1 charge-charge (CC) — (0, 0).
    9 transverse-transverse (TT) — (i, j) for i, j ∈ {1, 2, 3}, of which
        3 diagonal (Hermitian on the centroid axes by themselves) and
        6 off-diagonal (3 unique pairs related by Hermitian transpose).

This module handles the **7 unique kernel calls** (CC + 3 TT diagonal +
3 TT off-diagonal upper).  Each tile is computed by exactly the same
``gw.v_q_g_flat._compute_V_q_g_flat_one_tile`` primitive that drives the scalar
charge-only V_q.  After each tile, the result streams to a per-tile
HDF5 dataset and the device array is freed.  Peak GPU memory equals
that of one scalar V_q tile — never the full bispinor object.

Hermitian-redundant tiles ((j, i) for i < j ∈ {1,2,3}) are NOT
materialised on disk.  The reader (:class:`BispinorVqReader`) returns
them on demand by transposing + conjugating the companion tile.

Public surface
--------------
* :func:`compute_V_q_bispinor_g_flat_to_h5` — the orchestrator.
* :class:`BispinorVqReader` — open the output file and fetch any
  (μ_L, ν_L) tile, including the gauge-zero and Hermitian-redundant
  cases, with a uniform interface.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import barrier



# 7 (μ_L, ν_L) tiles for which the unified kernel runs:
UNIQUE_TILES: tuple[tuple[int, int], ...] = (
    (0, 0),                              # CC
    (1, 1), (2, 2), (3, 3),              # TT diagonal
    (1, 2), (1, 3), (2, 3),              # TT off-diagonal upper triangular
)

# 6 gauge-zero tiles: (0, i) and (i, 0) for i ∈ {1, 2, 3}
ZERO_TILES: frozenset = frozenset(
    [(0, i) for i in (1, 2, 3)] + [(i, 0) for i in (1, 2, 3)]
)

# 3 Hermitian-redundant tiles paired with their companions in UNIQUE_TILES.
# Reader retrieves V[j,i] = conj(swapaxes(V[i,j], -1, -2)) for these.
HERMITIAN_PAIRS: dict[tuple[int, int], tuple[int, int]] = {
    (2, 1): (1, 2),
    (3, 1): (1, 3),
    (3, 2): (2, 3),
}

V_QMUNU_FORMAT = "bispinor_lorentz_v2"


def tile_dataset_name(mu_L: int, nu_L: int) -> str:
    """Stable per-tile dataset name in the output HDF5.

    Charge channel uses ``V_qmunu_CC``; transverse uses ``V_qmunu_TT_ij``.
    Reader grep-matches these names so don't rename without updating
    :class:`BispinorVqReader`.
    """
    if (mu_L, nu_L) == (0, 0):
        return "V_qmunu_CC"
    return f"V_qmunu_TT_{mu_L}{nu_L}"


# ---------------------------------------------------------------------------
# Geometry helper: K_cart on the sphere for the transverse projector
# ---------------------------------------------------------------------------


def _tt_head_tensor(
    *, bvec: np.ndarray, cell_volume: float, sys_dim: int, kgrid,
) -> np.ndarray:
    """``T_ab = ⟨v(q) P^T_ab(q̂)⟩_mBZ`` at q=Γ, BARE units.

    The missing q=Γ, G=0 slot of the bare TT tiles — see
    ``_make_per_q_v_builder_for_tile``'s ``tt_head_correction`` docstring
    and ``docs/BISPINOR_DHFB_DESIGN.md`` §11.  One (3,3) tensor call per
    run (not per tile, not per q); callers slice ``T[i, j]``.  Routed
    through the same ``vcoul`` mini-BZ sampler ``q0_average`` uses for the
    charge head's ``vc0`` — no second sampler.  This helper owns the
    positive geometric projector average; the returned TT propagator applies
    the Coulomb-gauge spatial-metric minus once, after the optional head-slot
    replacement, in ``_make_per_q_v_builder_for_tile``.
    """
    from ffi import _services
    _services.ensure_on_path()
    import vcoul
    if int(sys_dim) not in (2, 3):
        raise ValueError(
            f"_tt_head_tensor: tt_head_correction is only implemented for "
            f"sys_dim in (2, 3) (slab / bulk); got sys_dim={sys_dim}.  Box "
            f"truncation (sys_dim=0) never zeros its q=Γ, G=0 slot, so it "
            f"needs no head substitute (vcoul.box_0d.Box0D._v_bare_per_q's "
            f"own docstring) — this should have been refused upstream by "
            f"gw_config's bispinor_tt_head_correction validation.")
    geometry = vcoul.CoulombGeometry(
        bvec=np.asarray(bvec, dtype=np.float64), cell_volume=float(cell_volume))
    kernel = vcoul.get_kernel(sys_dim)
    return np.asarray(
        kernel.q0_average_transverse_tensor(
            geometry, tuple(int(s) for s in kgrid),
            # Bulk v(q)~1/q^2 gives the pure-Sobol estimator infinite
            # variance; Bulk3D's owner requires its analytic sphere term
            # for a production head.  The slab sibling deliberately keeps
            # that 3D term off (its flag only widens the Voronoi fold).
            analytic_sphere=(int(sys_dim) == 3),
        ),
        dtype=np.float64)


def _make_per_q_v_builder_for_tile(
    *,
    mu_L: int, nu_L: int,
    bvec: np.ndarray, cell_volume: float, sys_dim: int,
    vcoul_cutoff_ry: float | None,
    bdot: np.ndarray | None = None,
    eps_K2: float = 1e-30,
    kgrid=None,
    tt_head_correction: bool = False,
):
    """Return ``builder(q_irr_frac, gvec_components) → (n_q, ngkmax) c128``.

    CC tile (μ_L=ν_L=0): bare Coulomb ``v(q+G)`` (real, ≥0).
    TT diagonal (i=j): ``−v(q+G) · (1 − K̂_i²)``.
    TT off-diagonal (i≠j): ``+v(q+G) · K̂_i K̂_j``.

    The ``K̂`` factor uses ``K2_safe = max(|q+G|², eps_K2)`` to keep
    the per-q-Γ slot finite; at K=0 the bare ``v`` is already zero
    (compute_v_q_per_G guards ``denom_zero``), so the product is zero
    regardless of t.  Head correction at q=Γ flows through the CC
    tile's ``g0_acc``; transverse tiles intentionally omit it —
    UNLESS ``tt_head_correction=True``.

    ``tt_head_correction`` (default False, byte-identical to every
    existing deck when off).  The charge structure factor obeys
    ``M_mn(q→0, G=0) → δ_mn``, an exact identity independent of direction,
    so replacing the zeroed CC slot needs only the scalar cell average
    ``⟨v⟩``.  The CURRENT structure factor ``⟨m|α^i|n⟩`` has no such
    limit — it is finite and generically non-diagonal — and the bare
    transverse propagator's projector ``P^T_ij(K̂) = δ_ij − K̂_iK̂_j`` is
    direction-dependent with NO limit as K→0 either.  A single grid point
    (the zeroed q=Γ, G=0 slot) cannot represent either fact, and the
    measured correction is not negligible: on the bi4 (MoS2 4×4) deck the
    missing rank-1 head is comparable in Frobenius norm to the WHOLE
    stored q=Γ TT slab (ratio 0.97/1.04/6.0 for the 11/22/33 tiles) and
    the eqp effect is ≈0.2 meV at 4×4, decaying only as ~1/√N_k — the same
    slow 2D decay that makes the charge head correction mandatory
    (``KNOWN_LORRAX_ISSUES.md``, bispinor row; claim 41, job 7885325).

    When on, the q=Γ, G=0 slot of a TT tile is replaced by the mini-BZ
    Voronoi cell average ``−⟨v(q) P^T_ij(q̂)⟩_mBZ``
    (:func:`_tt_head_tensor`) instead of being left at zero.  This is the
    SAME mechanism the CC
    charge exchange head uses conceptually — a mini-BZ-averaged
    replacement for an otherwise-undefined q→0 grid point — expressed at
    the SAME site the value already flows through: the returned
    ``v_per_G`` table.  No second Dyson/Σ code path is added; the
    existing ``(μ,ν)`` convolution (``gw.v_q_g_flat``) and the existing
    Σ^B kernel (``gw.sigma_x_bispinor``) consume the corrected tile
    exactly as they consume any other value in it, which is what makes
    this a rank-1 update in centroid space after the ``Σ_G`` contraction
    (``ζ(q=Γ,μ,G=0)`` is the only nonzero-weight ζ row at that slot) even
    though nothing here builds ``ζ`` or an outer product explicitly.
    """
    from .compute_vcoul import compute_v_q_per_G

    is_CC = (mu_L == 0 and nu_L == 0)
    if not is_CC:
        if not (1 <= mu_L <= 3 and 1 <= nu_L <= 3):
            raise ValueError(
                f"_make_per_q_v_builder_for_tile: TT tile indices must "
                f"satisfy 1 ≤ μ_L, ν_L ≤ 3; got ({mu_L}, {nu_L}).")
        i, j = mu_L - 1, nu_L - 1
        from ffi import _services
        _services.ensure_on_path()
        from vcoul import COULOMB_GAUGE_TT_SIGN, transverse_projector
        tt_metric_sign = float(COULOMB_GAUGE_TT_SIGN)
    bvec_f = np.asarray(bvec, dtype=np.float64)

    _tt_correction_value = None
    if tt_head_correction and not is_CC:
        if kgrid is None:
            raise ValueError(
                "_make_per_q_v_builder_for_tile: tt_head_correction=True "
                "needs kgrid (the mini-BZ Voronoi cell is defined by the "
                "q-grid).")
        T = _tt_head_tensor(
            bvec=bvec_f, cell_volume=cell_volume, sys_dim=sys_dim, kgrid=kgrid)
        # Bare (T) -> the same "v(q+G)/Ω_cell already applied" convention
        # compute_v_q_per_G's output carries (vcoul.base.CoulombKernel's
        # own Protocol docstring) — divide by cell_volume ONCE, here, not
        # inside the shared vcoul estimator (whose contract is explicitly
        # bare, matching minibz_average/minibz_moment_tensor).
        _tt_correction_value = complex(T[i, j] / float(cell_volume))

    def builder(q_irr_frac, gvec_components):
        v = compute_v_q_per_G(
            q_irr_frac, gvec_components,
            bvec=bvec_f, cell_volume=cell_volume,
            sys_dim=sys_dim, vcoul_cutoff_ry=vcoul_cutoff_ry,
            bdot=bdot,
        )                                                # (n_q, ngkmax) f64
        if is_CC:
            return v.astype(np.complex128)
        # K_cart[q, a, g] = sum_b bvec[b, a] · (q + G)[q, b, g]
        qG_frac = (q_irr_frac[:, :, None]
                    + np.asarray(gvec_components, dtype=np.float64))
        K_cart = np.einsum('ba,qbg->qag', bvec_f, qG_frac)
        K2 = np.sum(K_cart * K_cart, axis=1)             # (n_q, ngkmax)
        is_gamma_slot = K2 <= eps_K2                     # unique (q=Γ,G=0)
        t = transverse_projector(
            np.moveaxis(K_cart, 1, -1), K2, eps_K2=eps_K2)[:, :, i, j]
        # Assemble the positive transverse-projector weight first so the
        # finite-q body and optional mini-BZ replacement share one, and only
        # one, Coulomb-gauge spatial-metric sign below.  Stored currents use
        # j^i = Psi† alpha^i Psi; no vertex or Sigma contraction compensates
        # this propagator sign.
        v_t = (v * t).astype(np.complex128)
        if _tt_correction_value is not None:
            v_t = np.where(is_gamma_slot, _tt_correction_value, v_t)
        return tt_metric_sign * v_t

    return builder


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_V_q_bispinor_g_flat_to_h5(
    *,
    zeta_C_loader,                              # ZetaLoader, charge ζ (G-flat)
    zeta_T_loaders: tuple,                      # length-3 ZetaLoader, μ_L=1,2,3 (G-flat)
    output_h5_path: Path | str,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int,
    n_rmu_C: int,
    n_rmu_T: int,
    bare_coulomb_cutoff_ry: float | None = None,
    bdot: np.ndarray | None = None,
    g_chunk: int | None = None,
    bgw_v_grid_fn=None,                         # only meaningful for the CC tile
    print_fn=print,
    verbose: bool = True,
    # IBZ cascade plumbing (default disabled — must opt in via gw_init).
    sym=None,
    centroid_C_idx: np.ndarray | None = None,
    centroid_T_idx: np.ndarray | None = None,
    use_ibz: bool = False,
    # Bispinor TT (transverse-transverse) q=Γ, G=0 mini-BZ head correction
    # (default off — every existing deck's TT tiles are byte-identical).
    # See _make_per_q_v_builder_for_tile's tt_head_correction docstring.
    tt_head_correction: bool = False,
    # Present only for the explicit mixed-representation comparison.  Leaving
    # these unset preserves the historical bare_transverse artifact exactly.
    bispinor_gw_mode: str | None = None,
    charge_representation: str | None = None,
    spatial_current_representation: str | None = None,
) -> tuple[Path, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Stream the 7 unique bispinor V_q^{μ_L, ν_L} tiles to HDF5 via the
    G-flat per-q + G-chunked path.

    Each tile shares the same orchestration as the scalar charge V_q —
    one q at a time, one per-tile ``v(q+G)`` table, contract via the
    G-chunked kernel into ``(n_q_full, n_rmu_L, n_rmu_R)``, write the
    tile, free the buffer.  Sharing comes from
    :func:`gw.v_q_g_flat._compute_V_q_g_flat_one_tile`; this function
    is just the 7-tile loop + per-tile HDF5 plumbing.

    Each centroid family follows its resolved full-BZ/IBZ path.  The returned
    four-channel tuple is the derived literal-G=0 view produced at the same
    projection seam as V.  It is not persisted beside V: canonical
    ``zeta_q_G`` remains the sole source of truth.  On an IBZ source, the
    symmetry service reconstructs the exact parent G coefficient and applies
    the scalar or polar-vector action; the writer owns no q map, centroid
    action or Cartesian rotation.

    The reader :class:`BispinorVqReader` opens this on-disk format.
    """
    from file_io.slab_io import SlabIO
    import h5py
    import json
    from .v_q_g_flat import _compute_V_q_g_flat_one_tile

    output_h5_path = Path(output_h5_path)
    if len(zeta_T_loaders) != 3:
        raise ValueError(
            f"zeta_T_loaders must be length 3 (μ_L=1,2,3); "
            f"got {len(zeta_T_loaders)}.")
    nq_total = int(np.prod(kgrid))

    # File creation + metadata (rank-0 + collective sync via SlabIO).
    with SlabIO(output_h5_path, mode="w", mesh=mesh_xy,
) as io:
        io.write_attr("kgrid", np.asarray(kgrid, dtype=np.int64))
        io.write_attr("n_rmu_C", np.int64(n_rmu_C))
        io.write_attr("n_rmu_T", np.int64(n_rmu_T))
        io.write_attr("n_q_total", np.int64(nq_total))

    # ------------------------------------------------------------------
    # IBZ cascade plumbing.  When ``use_ibz=True`` the bispinor ζ̃ files
    # were written IBZ-only (see ``gw_init.fit_zeta`` and
    # ``isdf_fitting.fit_zeta_to_h5``); the per-tile kernel iterates the
    # parent IBZ q-list and unfolds post-loop via ``unfold_isdf_operator``.
    # Two centroid sets ⇒ two orbit-closure checks (CC ↔ charge, TT ↔
    # transverse); they may diverge in principle but in practice are
    # generated together by the user's ``kmeans_cli --orbit-aware`` run.
    # If either set's closure check fails, the corresponding tiles fall
    # back to full-BZ.  See derivation §A5.
    #
    # THE FALLBACK IS ANNOUNCED IN ONE PLACE, and it is not here.  This
    # block used to print its own second wording of the same fact, behind
    # ``verbose`` — so a bispinor production run with verbose off degraded
    # both channels silently.  ``gw.qgrid_symmetry`` now speaks once per
    # centroid SET, which is what makes the charge and transverse cases
    # two distinct lines instead of one line printed twice.
    # ------------------------------------------------------------------
    from .v_q_g_flat import _resolve_ibz_q_list

    def _ibz_tables_for(centroid_idx, label):
        if not use_ibz or sym is None or centroid_idx is None:
            return None, False
        tables = _resolve_ibz_q_list(
            sym=sym, centroid_indices=np.asarray(centroid_idx, dtype=np.int32),
            kgrid=kgrid, fft_grid=fft_grid,
            context=f"bispinor g-flat, {label} centroids")
        (_, _, _, _, _sym_perm, _L_table, _ok) = tables
        return tables, _ok

    _ibz_C, _use_ibz_C = _ibz_tables_for(centroid_C_idx, 'charge')
    _ibz_T, _use_ibz_T = _ibz_tables_for(centroid_T_idx, 'transverse')

    # ONE measured q-row policy for every bispinor consumer: CC, TT operator
    # mixing and all four one-leg vectors.  In particular, a ferromagnet's
    # policy contains no antiunitary rows; no tile may fall back to the raw
    # ``sym.sym_idx_q`` table and take a second TRS decision.
    qgrid_policy = None
    if (_use_ibz_C or _use_ibz_T) and sym is not None:
        from .qgrid_symmetry import qgrid_trs_policy_for
        qgrid_policy = qgrid_trs_policy_for(
            sym=sym,
            irr_idx_q=np.asarray(sym.irr_idx_q, dtype=np.int32),
            sym_idx_q=np.asarray(sym.sym_idx_q, dtype=np.int32),
            kgrid=tuple(kgrid),
            n_sym_spatial=int(np.asarray(sym.sym_matrices).shape[0]),
            context="bispinor V / one-leg",
        )

    # Buffer the 6 unique TT tiles post-unfold so we can apply the 3×3
    # Lorentz mixing across them before writing.  CC tile streams straight
    # to disk — no mixing.  Decision: write-time mixing keeps the on-disk
    # format identical to the existing post-mix tiles (downstream Σ^B
    # reads them unchanged via BispinorVqReader.get_tile).  Alternative
    # (read-time mixing inside the reader) would require all 9 TT tiles
    # to be in memory for any single get_tile(i, j) call — write-time
    # mixing keeps the reader's per-tile contract clean.  See derivation
    # §A5 (the algebraic identity for V^{i,j}_full[q]).
    tt_buffer: dict[tuple[int, int], jax.Array] = {}
    tt_g0 = None
    g0_by_channel: list[jax.Array | None] = [None, None, None, None]

    for tile_idx, (mu_L, nu_L) in enumerate(UNIQUE_TILES):
        same_zeta = (mu_L == nu_L)
        loader_L = zeta_C_loader if mu_L == 0 else zeta_T_loaders[mu_L - 1]
        loader_R = (None if same_zeta
                    else (zeta_C_loader if nu_L == 0
                           else zeta_T_loaders[nu_L - 1]))
        n_rmu_L = n_rmu_C if mu_L == 0 else n_rmu_T
        n_rmu_R = n_rmu_C if nu_L == 0 else n_rmu_T
        is_CC = (mu_L == 0 and nu_L == 0)
        # g0 is a one-leg zeta coefficient, so diagonal tiles are its sole
        # producer.  On the transverse IBZ path each streamed source
        # component returns its contributions to all three target Cartesian
        # channels; only those three small carriers are accumulated.
        write_g0 = same_zeta

        # CC tile: charge-centroid orbit closure.  TT tiles: transverse-
        # centroid orbit closure.  These are independent; either may
        # fall back to full-BZ if its centroid set isn't orbit-closed.
        _tile_use_ibz = _use_ibz_C if is_CC else _use_ibz_T
        _tile_sym = sym if _tile_use_ibz else None
        _tile_cent = (centroid_C_idx if is_CC else centroid_T_idx) if _tile_use_ibz else None

        v_builder = _make_per_q_v_builder_for_tile(
            mu_L=mu_L, nu_L=nu_L,
            bvec=bvec, cell_volume=cell_volume, sys_dim=sys_dim,
            vcoul_cutoff_ry=bare_coulomb_cutoff_ry, bdot=bdot,
            kgrid=kgrid, tt_head_correction=tt_head_correction,
        )
        # BGW vcoul overlay only meaningful on the CC tile; transverse
        # tiles are pure projector applications.  Wrap the builder.
        if is_CC and bgw_v_grid_fn is not None:
            _base = v_builder
            nx, ny, nz = (int(s) for s in fft_grid)

            def _v_builder_with_bgw(q_irr_frac, gvec_components,
                                     _base=_base, nx=nx, ny=ny, nz=nz):
                v = np.asarray(_base(q_irr_frac, gvec_components))
                for qi in range(q_irr_frac.shape[0]):
                    v_full = np.asarray(
                        bgw_v_grid_fn(tuple(q_irr_frac[qi]))).reshape(-1)
                    miller = gvec_components[qi]
                    flat = ((miller[0] % nx) * ny * nz
                              + (miller[1] % ny) * nz
                              + (miller[2] % nz))
                    v_at = v_full[flat]
                    v[qi] = np.where(v_at != 0.0, v_at, v[qi])
                return v
            v_builder = _v_builder_with_bgw

        if verbose and jax.process_index() == 0:
            print_fn(f"  [bispinor g-flat] tile "
                     f"{tile_idx + 1}/{len(UNIQUE_TILES)} "
                     f"(μ_L={mu_L}, ν_L={nu_L})  n_rmu_L={n_rmu_L} "
                     f"n_rmu_R={n_rmu_R}  same_zeta={same_zeta}  "
                     f"use_ibz={_tile_use_ibz}")

        V_acc, g0_acc = _compute_V_q_g_flat_one_tile(
            loader_L, loader_R,
            v_per_G_builder=v_builder,
            kgrid=kgrid, fft_grid=fft_grid,
            mesh_xy=mesh_xy,
            g_chunk=g_chunk,
            sym=_tile_sym, centroid_indices=_tile_cent,
            is_charge_cc=is_CC,
            write_g0=write_g0,
            qgrid_policy=qgrid_policy,
            one_leg_action=(
                "polar" if (same_zeta and mu_L != 0 and _tile_use_ibz)
                else "scalar"),
            source_component=(
                mu_L - 1 if (same_zeta and mu_L != 0 and _tile_use_ibz)
                else None),
            timing_label=tile_dataset_name(mu_L, nu_L),
            verbose=verbose,
        )

        if (same_zeta and mu_L != 0 and _use_ibz_T
                and g0_acc is not None):
            tt_g0 = g0_acc if tt_g0 is None else tt_g0 + g0_acc
            del g0_acc
            g0_acc = None

        # Keep the literal-G=0 view in memory at its projection owner.  The
        # transverse IBZ case is filled after its three source components are
        # rotated and accumulated by the symmetry service below.
        if same_zeta and g0_acc is not None:
            g0_by_channel[mu_L] = g0_acc

        if is_CC or not _use_ibz_T:
            # Two cases stream straight to disk per-tile:
            #   * CC tile — never Lorentz-mixed (γ̃^0 = I is invariant).
            #   * TT tiles when the transverse IBZ cascade is off — the
            #     unfold inside ``_compute_V_q_g_flat_one_tile`` was a
            #     no-op (full-BZ path), no mixing needed.  Buffering all
            #     6 TT tiles would inflate peak memory; we preserve the
            #     pre-IBZ "free between tiles" behaviour exactly.
            name = tile_dataset_name(mu_L, nu_L)
            v_logical_shape = (int(V_acc.shape[0]), n_rmu_L, n_rmu_R)
            with SlabIO(output_h5_path, mode='a', mesh=mesh_xy,
) as tile_io:
                tile_io.create_dataset(
                    name, shape=v_logical_shape, dtype=V_acc.dtype)
                tile_io.write_slab(name, V_acc)
            del V_acc, g0_acc
        else:
            # Buffer for post-loop Lorentz mixing (IBZ cascade active on
            # the transverse centroid set).
            tt_buffer[(mu_L, nu_L)] = V_acc
            del g0_acc

    # ------------------------------------------------------------------
    # Lorentz mixing on the TT block.  Only runs when the transverse IBZ
    # cascade is active — the 3×3 mixing is a no-op on full-BZ data
    # because the per-q sym op is identity there.  See derivation §A5.
    # ------------------------------------------------------------------
    if _use_ibz_T and sym is not None and tt_buffer:
        from ffi import _services
        _services.ensure_on_path()
        from symmetry_maps import mix_channels_by_proper_rotation

        # Synthesise the 3 Hermitian-redundant tiles (j, i) for i < j from
        # the stored upper-triangle.  These are needed as INPUTS to the
        # 3×3 contraction; we write only the 6 unique upper-triangle outputs.
        tt_full_in: dict[tuple[int, int], jax.Array] = dict(tt_buffer)
        for (j, i), (i_src, j_src) in HERMITIAN_PAIRS.items():
            tt_full_in[(j, i)] = jnp.conj(
                jnp.swapaxes(tt_buffer[(i_src, j_src)], -1, -2))

        tt_mixed = mix_channels_by_proper_rotation(
            tt_full_in,
            sym_idx=np.asarray(
                qgrid_policy.unfold_sym_idx, dtype=np.int32),
            R_proper_table=np.asarray(sym.R_proper, dtype=np.float64),
            mesh_xy=mesh_xy,
        )

        # Write the 6 unique upper-triangle TT tiles post-mix.
        for (mu_L, nu_L) in UNIQUE_TILES:
            if mu_L == 0 and nu_L == 0:
                continue
            V_mix = tt_mixed[(mu_L, nu_L)]
            name = tile_dataset_name(mu_L, nu_L)
            v_logical_shape = (int(V_mix.shape[0]), n_rmu_T, n_rmu_T)
            with SlabIO(output_h5_path, mode='a', mesh=mesh_xy,
) as tile_io:
                tile_io.create_dataset(
                    name, shape=v_logical_shape, dtype=V_mix.dtype)
                tile_io.write_slab(name, V_mix)
                if mu_L == nu_L:
                    if tt_g0 is None:
                        raise RuntimeError(
                            "bispinor transverse IBZ one-leg accumulation "
                            "is missing despite three diagonal source tiles.")
                    g0_by_channel[mu_L] = tt_g0[mu_L - 1]
        del tt_full_in, tt_mixed
    del tt_buffer, tt_g0

    # Format string + tile-layout JSON — rank-0 post-close write so
    # the BispinorVqReader can h5-open without rank coordination.
    if jax.process_index() == 0:
        with h5py.File(output_h5_path, "a") as f:
            f.create_dataset("v_qmunu_format",
                             data=np.bytes_(V_QMUNU_FORMAT))
            f.attrs["unique_tiles"] = json.dumps(
                [list(t) for t in UNIQUE_TILES])
            f.attrs["zero_tiles"] = json.dumps(
                [list(t) for t in sorted(ZERO_TILES)])
            f.attrs["hermitian_pairs"] = json.dumps(
                [[list(k), list(v)] for k, v in HERMITIAN_PAIRS.items()])
            if bispinor_gw_mode is not None:
                f.attrs["bispinor_gw_mode"] = str(bispinor_gw_mode)
            if charge_representation is not None:
                f.attrs["charge_representation"] = str(
                    charge_representation)
            if spatial_current_representation is not None:
                f.attrs["spatial_current_representation"] = str(
                    spatial_current_representation)
    barrier("v_q_bispinor_g_flat_tile_layout_meta")
    if any(vector is None for vector in g0_by_channel):
        missing = [i for i, vector in enumerate(g0_by_channel)
                   if vector is None]
        raise RuntimeError(
            "bispinor literal-G=0 projection did not produce channels "
            f"{missing}")
    return output_h5_path, tuple(g0_by_channel)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class BispinorVqReader:
    """Open a bispinor V_q HDF5 written by :func:`compute_V_q_bispinor_g_flat_to_h5`.

    Provides a uniform interface over the 16 (μ_L, ν_L) blocks:

    * ``get_tile(μ_L, ν_L)``        — JAX array on the (None, 'x', 'y')
                                       sharding, materialised on demand.
                                       For zero-by-gauge tiles returns
                                       a zeros array sized appropriately.
                                       For Hermitian-redundant tiles
                                       reads the companion + applies
                                       ``conj(swapaxes(.., -1, -2))``.
    Caller manages the lifecycle (use as a context manager).
    """

    def __init__(self, filename: Path | str, mesh_xy: Mesh,
):
        from file_io.slab_io import SlabIO
        import h5py
        self._filename = Path(filename)
        self._mesh = mesh_xy

        # Small metadata scalars are written via SlabIO.write_attr (which
        # creates a dataset in the file).  Read them via h5py — every rank
        # opens its own 'r' handle for a few-byte read; broadcast overhead
        # would dominate.  Validate all metadata before opening SlabIO: its
        # constructor opens a collective PhdfCtx which only __exit__ can
        # close, so a constructor-time refusal must happen first.
        with h5py.File(self._filename, "r") as f:
            def _read_scalar(name):
                d = f[name]
                v = d[()] if d.shape == () else d[:]
                return v
            fmt = _read_scalar("v_qmunu_format")
            if isinstance(fmt, bytes):
                fmt = fmt.decode("utf-8")
            if str(fmt) != V_QMUNU_FORMAT:
                raise ValueError(
                    f"{self._filename}: v_qmunu_format='{fmt}', "
                    f"expected '{V_QMUNU_FORMAT}'.  Wrong file or stale "
                    f"format from a different LORRAX revision."
                )
            self.kgrid = tuple(int(x) for x in _read_scalar("kgrid"))
            self.n_rmu_C = int(_read_scalar("n_rmu_C"))
            self.n_rmu_T = int(_read_scalar("n_rmu_T"))
            self.n_q_total = int(_read_scalar("n_q_total"))

        self._io = SlabIO(self._filename, mode="r", mesh=mesh_xy,
)
        self._io.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._io.__exit__(*exc)

    def _zero_tile(self, mu_L: int, nu_L: int) -> jax.Array:
        n_L = self.n_rmu_C if mu_L == 0 else self.n_rmu_T
        n_R = self.n_rmu_C if nu_L == 0 else self.n_rmu_T
        sharding = NamedSharding(self._mesh, P(None, 'x', 'y'))
        n_L_p, n_R_p = self._padded_shape_LR(n_L, n_R)
        return jax.lax.with_sharding_constraint(
            jnp.zeros((self.n_q_total, n_L_p, n_R_p), dtype=jnp.complex128),
            sharding,
        )

    def _tile_shape(self, mu_L: int, nu_L: int) -> tuple[int, int, int]:
        n_L = self.n_rmu_C if mu_L == 0 else self.n_rmu_T
        n_R = self.n_rmu_C if nu_L == 0 else self.n_rmu_T
        return (self.n_q_total, n_L, n_R)

    def _padded_shape_LR(self, n_L: int, n_R: int) -> tuple[int, int]:
        """Round n_L, n_R up to the total mesh-product (``gx*gy``).  This
        mirrors the write-side μ padding in
        ``gw.v_q_g_flat._compute_V_q_g_flat_one_tile`` (its ``_pad``
        helper, which also pads to ``p_x*p_y``) and matches the
        ψ-side μ extent built by ``load_centroids_band_chunked`` — so a
        single pad here makes Σ^B's V tile broadcast against ψ with no
        further padding step in sigma_x_bispinor.

        Padding to ``gx*gy`` (rather than per-axis ``gx``/``gy``) is also
        what makes sharded reads with spec P(None,'x','y') divide
        cleanly under any 2D mesh factorisation.

        Routed through ``runtime.padding.padded_mu_extent`` so the
        test-only LORRAX_EXTRA_MU_PAD knob stays consistent with the
        ψ-side / write-side extents."""
        from runtime.padding import padded_mu_extent
        proc = int(self._mesh.shape['x']) * int(self._mesh.shape['y'])
        return (padded_mu_extent(int(n_L), proc),
                padded_mu_extent(int(n_R), proc))

    def get_tile(self, mu_L: int, nu_L: int) -> jax.Array:
        """Return V^{μ_L, ν_L}_q as a sharded JAX array (n_q, n_L_padded,
        n_R_padded) c128.  When n_L/n_R aren't divisible by the mesh axis
        size, the trailing μ rows are zero-padded — mirrors the write-side
        μ padding in ``gw.v_q_g_flat._compute_V_q_g_flat_one_tile`` and
        lets Σ^B run at any process count without a runtime divisibility
        error."""
        if not (0 <= mu_L <= 3 and 0 <= nu_L <= 3):
            raise ValueError(f"Lorentz indices must be in {{0..3}}; got "
                             f"({mu_L}, {nu_L}).")
        if (mu_L, nu_L) in ZERO_TILES:
            return self._zero_tile(mu_L, nu_L)
        spec = P(None, 'x', 'y')
        if (mu_L, nu_L) in HERMITIAN_PAIRS:
            companion = HERMITIAN_PAIRS[(mu_L, nu_L)]
            n_L_c, n_R_c = self._tile_shape(*companion)[1:]
            n_L_p, n_R_p = self._padded_shape_LR(n_L_c, n_R_c)
            # Read the companion with its two centroid axes reversed so the
            # physical Hermitian transpose below lands directly on this
            # reader's canonical P(None, 'x', 'y') contract.  Reading in the
            # ordinary orientation and then swapping would return
            # P(None, 'y', 'x') and force every consumer to repair it.
            V_companion = self._io.read_slab(
                tile_dataset_name(*companion),
                shape=(self.n_q_total, n_L_p, n_R_p),
                mesh=self._mesh, partition_spec=P(None, 'y', 'x'))
            # Hermitian: V[j,i](q,μ,ν) = V[i,j](q,ν,μ)*
            return jnp.conj(jnp.swapaxes(V_companion, -1, -2))
        # Direct read (member of UNIQUE_TILES)
        n_L, n_R = self._tile_shape(mu_L, nu_L)[1:]
        n_L_p, n_R_p = self._padded_shape_LR(n_L, n_R)
        return self._io.read_slab(
            tile_dataset_name(mu_L, nu_L),
            shape=(self.n_q_total, n_L_p, n_R_p),
            mesh=self._mesh, partition_spec=spec)

    @property
    def filename(self) -> Path:
        return self._filename
