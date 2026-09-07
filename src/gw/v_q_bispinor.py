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
import json
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
V_QMUNU_INVENTORY_SCHEMA = 1
V_QMUNU_INVENTORY_DATASET = "v_qmunu_unique_tile_inventory"
V_QMUNU_DATA_READY_DATASET = "v_qmunu_data_ready"
V_QMUNU_TILE_DTYPE = np.dtype(np.complex128)


def tile_dataset_name(mu_L: int, nu_L: int) -> str:
    """Stable per-tile dataset name in the output HDF5.

    Charge channel uses ``V_qmunu_CC``; transverse uses ``V_qmunu_TT_ij``.
    Reader grep-matches these names so don't rename without updating
    :class:`BispinorVqReader`.
    """
    if (mu_L, nu_L) == (0, 0):
        return "V_qmunu_CC"
    return f"V_qmunu_TT_{mu_L}{nu_L}"


def _expected_unique_tile_inventory(
    *, n_q_total: int, n_rmu_C: int, n_rmu_T: int,
) -> dict:
    """Return the one canonical receipt for a complete bispinor-V file."""
    tiles = []
    for mu_L, nu_L in UNIQUE_TILES:
        n_L = n_rmu_C if mu_L == 0 else n_rmu_T
        n_R = n_rmu_C if nu_L == 0 else n_rmu_T
        tiles.append({
            "lorentz": [mu_L, nu_L],
            "name": tile_dataset_name(mu_L, nu_L),
            "logical_shape": [n_q_total, n_L, n_R],
            "dtype": V_QMUNU_TILE_DTYPE.name,
        })
    return {"schema": V_QMUNU_INVENTORY_SCHEMA, "tiles": tiles}


def _inventory_json(*, n_q_total: int, n_rmu_C: int,
                    n_rmu_T: int) -> str:
    return json.dumps(
        _expected_unique_tile_inventory(
            n_q_total=n_q_total, n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T),
        sort_keys=True, separators=(",", ":"),
    )


def _validate_unique_tile_datasets(
    h5_file, *, filename: Path, n_q_total: int,
    n_rmu_C: int, n_rmu_T: int,
) -> None:
    """Validate every physical tile without opening a collective reader."""
    expected = _expected_unique_tile_inventory(
        n_q_total=n_q_total, n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T)
    from symmetry_maps import QIRR_TABLE_SUFFIX
    expected_names = {tile["name"] for tile in expected["tiles"]}
    actual_names = {
        name for name in h5_file.keys()
        if name.startswith("V_qmunu_") and not name.endswith(QIRR_TABLE_SUFFIX)
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"{filename}: incomplete bispinor-V unique-tile set; "
            f"missing={missing}, unexpected={unexpected}.")
    for tile in expected["tiles"]:
        dataset = h5_file[tile["name"]]
        expected_shape = tuple(tile["logical_shape"])
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(
                f"{filename}: tile {tile['name']} has storage shape "
                f"{tuple(dataset.shape)}, expected exact logical shape "
                f"{expected_shape}.")
        if np.dtype(dataset.dtype) != V_QMUNU_TILE_DTYPE:
            raise ValueError(
                f"{filename}: tile {tile['name']} has dtype "
                f"{np.dtype(dataset.dtype).name}, expected "
                f"{V_QMUNU_TILE_DTYPE.name}.")


def _publish_unique_tile_inventory(
    h5_file, *, filename: Path, n_q_total: int,
    n_rmu_C: int, n_rmu_T: int,
    bispinor_gw_mode: str | None = None,
    charge_representation: str | None = None,
    spatial_current_representation: str | None = None,
) -> None:
    """Certify a staged tile file, publishing readiness as the last write."""
    h5_file.create_dataset(
        V_QMUNU_DATA_READY_DATASET, data=np.bool_(False))
    _validate_unique_tile_datasets(
        h5_file, filename=filename, n_q_total=n_q_total,
        n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T)
    h5_file.create_dataset(
        "v_qmunu_format", data=np.bytes_(V_QMUNU_FORMAT))
    h5_file.create_dataset(
        V_QMUNU_INVENTORY_DATASET,
        data=np.bytes_(_inventory_json(
            n_q_total=n_q_total, n_rmu_C=n_rmu_C,
            n_rmu_T=n_rmu_T)))
    h5_file.attrs["unique_tiles"] = json.dumps(
        [list(t) for t in UNIQUE_TILES])
    h5_file.attrs["zero_tiles"] = json.dumps(
        [list(t) for t in sorted(ZERO_TILES)])
    h5_file.attrs["hermitian_pairs"] = json.dumps(
        [[list(k), list(v)] for k, v in HERMITIAN_PAIRS.items()])
    if bispinor_gw_mode is not None:
        h5_file.attrs["bispinor_gw_mode"] = str(bispinor_gw_mode)
    if charge_representation is not None:
        h5_file.attrs["charge_representation"] = str(
            charge_representation)
    if spatial_current_representation is not None:
        h5_file.attrs["spatial_current_representation"] = str(
            spatial_current_representation)
    h5_file.flush()
    h5_file[V_QMUNU_DATA_READY_DATASET][()] = np.bool_(True)
    h5_file.flush()


def _tile_logical_shape(
    mu_L: int, nu_L: int, *, n_q_total: int,
    n_rmu_C: int, n_rmu_T: int,
) -> tuple[int, int, int]:
    n_L = n_rmu_C if mu_L == 0 else n_rmu_T
    n_R = n_rmu_C if nu_L == 0 else n_rmu_T
    return (n_q_total, n_L, n_R)


def _require_tile_carrier(
    array, *, name: str, logical_shape: tuple[int, int, int],
) -> None:
    carrier_shape = tuple(int(x) for x in array.shape)
    if (carrier_shape[0] != logical_shape[0]
            or any(got < need for got, need in
                   zip(carrier_shape[1:], logical_shape[1:]))):
        raise ValueError(
            f"{name}: carrier shape {carrier_shape} cannot supply logical "
            f"tile shape {logical_shape}.")
    if np.dtype(array.dtype) != V_QMUNU_TILE_DTYPE:
        raise ValueError(
            f"{name}: carrier dtype {np.dtype(array.dtype).name}, expected "
            f"{V_QMUNU_TILE_DTYPE.name}; refusing a silent precision cast.")


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
    from .v_q_g_flat import _compute_V_q_g_flat_one_tile

    output_h5_path = Path(output_h5_path)
    if len(zeta_T_loaders) != 3:
        raise ValueError(
            f"zeta_T_loaders must be length 3 (μ_L=1,2,3); "
            f"got {len(zeta_T_loaders)}.")
    # Each family stores its canonical q parents with its own centroid tables.
    from .v_q_g_flat import _resolve_ibz_q_list

    if not use_ibz or sym is None:
        raise ValueError("Bispinor V requires q-IBZ storage; rerun with symmetry enabled.")
    _ibz_C = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_C_idx, kgrid=kgrid, fft_grid=fft_grid,
        context="bispinor charge V", return_resolution=True)
    _ibz_T = _resolve_ibz_q_list(
        sym=sym, centroid_indices=centroid_T_idx, kgrid=kgrid, fft_grid=fft_grid,
        context="bispinor current V", return_resolution=True)
    _use_ibz_C, _use_ibz_T = _ibz_C[6], _ibz_T[6]
    if not (_use_ibz_C and _use_ibz_T):
        raise ValueError("Bispinor V requires two orbit-closed centroid families.")
    nq_total = len(_ibz_C[1])
    if len(_ibz_T[1]) != nq_total:
        raise ValueError("Bispinor V families disagree on q-IBZ rows.")
    with SlabIO(output_h5_path, mode="w", mesh=mesh_xy) as io:
        io.write_attr("kgrid", np.asarray(kgrid, dtype=np.int64))
        io.write_attr("n_rmu_C", np.int64(n_rmu_C))
        io.write_attr("n_rmu_T", np.int64(n_rmu_T))
        io.write_attr("n_q_total", np.int64(nq_total))

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

        name = tile_dataset_name(mu_L, nu_L)
        v_logical_shape = _tile_logical_shape(
            mu_L, nu_L, n_q_total=nq_total, n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T)
        _require_tile_carrier(V_acc, name=name, logical_shape=v_logical_shape)
        with SlabIO(output_h5_path, mode="a", mesh=mesh_xy) as tile_io:
            tile_io.create_dataset(name, shape=v_logical_shape, dtype=V_QMUNU_TILE_DTYPE)
            tile_io.write_slab(name, V_acc)
        del V_acc, g0_acc
        barrier("bispinor_V_tile_closed")
        if jax.process_index() == 0:
            from symmetry_maps import QirrTables, stamp_qirr_tensor
            table = _ibz_C if is_CC else _ibz_T
            stamp_qirr_tensor(output_h5_path, name,
                tables=QirrTables(table[2], qgrid_policy.unfold_sym_idx, table[1],
                                  table[4], table[5], qgrid_policy.n_sym_spatial),
                closure_verdict=table[7].verdict, n_rmu_logical=n_rmu_L)
        barrier("bispinor_V_tile_stamped")
    if tt_g0 is not None:
        g0_by_channel[1:] = [tt_g0[i] for i in range(3)]
    del tt_g0

    if any(vector is None for vector in g0_by_channel):
        raise RuntimeError("bispinor literal-G=0 projection did not produce four channels")
    with SlabIO(output_h5_path, mode="a", mesh=mesh_xy) as head_io:
        for channel, vector in enumerate(g0_by_channel):
            name = f"photon_g0_vectors_{channel}"
            logical_mu = n_rmu_C if channel == 0 else n_rmu_T
            head_io.create_dataset(name, shape=(1, logical_mu), dtype=V_QMUNU_TILE_DTYPE)
            head_io.write_slab(name, vector[:1])
    barrier("bispinor_Gamma_vectors_closed")

    # Format, canonical inventory and readiness receipt — rank-0 post-close
    # write so BispinorVqReader can reject a torn file without rank
    # coordination.  data_ready=True is deliberately the final mutation.
    if jax.process_index() == 0:
        with h5py.File(output_h5_path, "a") as f:
            _publish_unique_tile_inventory(
                f, filename=output_h5_path, n_q_total=nq_total,
                n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T,
                bispinor_gw_mode=bispinor_gw_mode,
                charge_representation=charge_representation,
                spatial_current_representation=(
                    spatial_current_representation))
    barrier("v_q_bispinor_g_flat_tile_layout_meta")
    return output_h5_path, tuple(g0_by_channel)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
