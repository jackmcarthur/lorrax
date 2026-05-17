"""Bispinor V_q^{μ_L, ν_L} orchestrator — thin loop over the unified tile kernel.

The bispinor pair density carries a 4-vector Lorentz index from the
Pauli decomposition of ψ†_iσ ψ_jσ′:

    n_iσ,jσ′(r) = Σ_{μ_L=0..3} ζ_{μ_L,a}(r) ⟨σ| τ^{μ_L} |σ′⟩ C^{a}_{ij}
    τ^0 = I,  τ^{1,2,3} = (σ_x, σ_y, σ_z)

The Coulomb kernel in Lorentz gauge couples the four channels by

    V^{μ_L,ν_L}_q(μ,ν) = Σ_K  ζ̄_{μ_L,μ}(K) · t^{μ_L,ν_L}(K) · v(K) · ζ_{ν_L,ν}(K)
    t^{0,0}      = 1
    t^{i,j}      = δ_ij − K̂_i K̂_j        (transverse projector)
    t^{0,i}=t^{i,0} = 0                    (Coulomb-gauge cross term)

so out of the 16 (μ_L, ν_L) blocks:
    6 zero by gauge — never computed.
    1 charge-charge (CC) — (0, 0).
    9 transverse-transverse (TT) — (i, j) for i, j ∈ {1, 2, 3}, of which
        3 diagonal (Hermitian on the centroid axes by themselves) and
        6 off-diagonal (3 unique pairs related by Hermitian transpose).

This module handles the **7 unique kernel calls** (CC + 3 TT diagonal +
3 TT off-diagonal upper).  Each tile is computed by exactly the same
``gw.v_q_tile.compute_V_q_tile`` primitive that drives the scalar
charge-only V_q.  After each tile, the result streams to a per-tile
HDF5 dataset and the device array is freed.  Peak GPU memory equals
that of one scalar V_q tile — never the full bispinor object.

Hermitian-redundant tiles ((j, i) for i < j ∈ {1,2,3}) are NOT
materialised on disk.  The reader (:class:`BispinorVqReader`) returns
them on demand by transposing + conjugating the companion tile.

Public surface
--------------
* :func:`compute_V_q_bispinor_to_h5` — the orchestrator.
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

from .v_q_tile import compute_V_q_tile


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

V_QMUNU_FORMAT = "bispinor_lorentz_v1"


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


def _make_K_cart_on_sphere_fn(sphere_idx, fft_grid, bvec):
    """Return ``vmap(qvec_frac → K_cart on sphere)``.

    For each q (fractional in the k-grid units):
        K_cart(q, G) = (G + q) · b  in cartesian, Bohr⁻¹

    Output shape: (n_G_sph, 3) per q; ``vmap`` lifts to (Q, n_G_sph, 3).
    """
    from .compute_vcoul import fft_integer_axes

    nx, ny, nz = (int(s) for s in fft_grid)
    gx, gy, gz = fft_integer_axes(nx, ny, nz)
    gx_b, gy_b, gz_b = jnp.broadcast_arrays(gx, gy, gz)
    G_int_flat = jnp.stack((gx_b, gy_b, gz_b), axis=-1).reshape(-1, 3).astype(jnp.float64)
    if sphere_idx is not None:
        G_int_sph = G_int_flat[sphere_idx]
    else:
        G_int_sph = G_int_flat
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)

    def get_K_cart(qvec_frac):                     # (3,)
        return (G_int_sph + qvec_frac[None, :]) @ bvec_j   # (n_G_sph, 3)

    return jax.vmap(get_K_cart)


# ---------------------------------------------------------------------------
# Per-tile v(K) · t^{μ_L, ν_L}(K) closure
# ---------------------------------------------------------------------------


def _make_v_per_G_for_tile(
    *,
    base_v_per_G_fn,           # (Q,3) → (Q, n_G_sph) c128 — charge channel v(K)
    K_cart_batch_fn,           # (Q,3) → (Q, n_G_sph, 3)  — required for TT tiles
    mu_L: int,
    nu_L: int,
    eps_K2: float = 1e-30,
):
    """Return the per-G weight callable for tile (μ_L, ν_L).

    For the CC tile the closure is the identity wrapper around
    ``base_v_per_G_fn``.  For TT tiles it multiplies by the transverse
    projector ``δ_ij − K̂_i K̂_j``, where K̂_i = K_i / |K|.

    Caveat: the K=0 limit is gauge-singular; we guard the denominator
    with ``eps_K2`` to keep AD/jit happy.  For TT diagonal tiles the
    G=0 head is suppressed by the (1 − K̂_i²) factor only when K is
    aligned with the i-axis; at generic q→0 the head is non-zero but
    *finite*.  Production sigma_X^B handles the q=0 head via
    ``g0_acc`` only on the CC tile (transverse head is captured in the
    body integral via ``v(K) · t``).
    """
    if mu_L == 0 and nu_L == 0:
        return base_v_per_G_fn

    i = mu_L - 1
    j = nu_L - 1

    if not (0 <= i < 3 and 0 <= j < 3):
        raise ValueError(
            f"Transverse projector defined only for spatial indices "
            f"i, j ∈ {{1, 2, 3}}; got μ_L={mu_L}, ν_L={nu_L}."
        )

    def v_per_G_fn(qvec_np_batch):
        qvec_arr = jnp.asarray(qvec_np_batch, dtype=jnp.float64)
        v = base_v_per_G_fn(qvec_arr)                  # (Q, n_G_sph) c128
        K_cart = K_cart_batch_fn(qvec_arr)             # (Q, n_G_sph, 3) f64
        K2 = jnp.sum(K_cart * K_cart, axis=-1)         # (Q, n_G_sph) f64
        K2_safe = jnp.where(K2 > eps_K2, K2, 1.0)
        Khat_ij = K_cart[..., i] * K_cart[..., j] / K2_safe   # f64
        if i == j:
            t = (1.0 - Khat_ij)
        else:
            t = -Khat_ij
        return (v * t.astype(v.dtype))

    return v_per_G_fn


# ---------------------------------------------------------------------------
# Per-tile v(q+G) builder for the G-flat path
# ---------------------------------------------------------------------------
#
# Returns a ``v_per_G_builder(q_irr_frac, gvec_components) -> (n_q, ngkmax)``
# closure that ``_compute_V_q_g_flat_one_tile`` consumes.  Math is the same
# as ``_make_v_per_G_for_tile`` (legacy shared-sphere); the difference is
# the per-q sphere comes from the writer's ``gvec_components`` table.

def _make_per_q_v_builder_for_tile(
    *,
    mu_L: int, nu_L: int,
    bvec: np.ndarray, cell_volume: float, sys_dim: int,
    vcoul_cutoff_ry: float | None,
    bdot: np.ndarray | None = None,
    eps_K2: float = 1e-30,
):
    """Return ``builder(q_irr_frac, gvec_components) → (n_q, ngkmax) c128``.

    CC tile (μ_L=ν_L=0): bare Coulomb ``v(q+G)`` (real, ≥0).
    TT diagonal (i=j): ``v(q+G) · (1 − K̂_i²)``.
    TT off-diagonal (i≠j): ``v(q+G) · (−K̂_i K̂_j)``.

    The ``K̂`` factor uses ``K2_safe = max(|q+G|², eps_K2)`` to keep
    the per-q-Γ slot finite; at K=0 the bare ``v`` is already zero
    (compute_v_q_per_G guards ``denom_zero``), so the product is zero
    regardless of t.  Head correction at q=Γ flows through the CC
    tile's ``g0_acc``; transverse tiles intentionally omit it.
    """
    from .compute_vcoul import compute_v_q_per_G

    is_CC = (mu_L == 0 and nu_L == 0)
    if not is_CC:
        if not (1 <= mu_L <= 3 and 1 <= nu_L <= 3):
            raise ValueError(
                f"_make_per_q_v_builder_for_tile: TT tile indices must "
                f"satisfy 1 ≤ μ_L, ν_L ≤ 3; got ({mu_L}, {nu_L}).")
        i, j = mu_L - 1, nu_L - 1
    bvec_f = np.asarray(bvec, dtype=np.float64)

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
        K2_safe = np.where(K2 > eps_K2, K2, 1.0)
        Khat_ij = K_cart[:, i] * K_cart[:, j] / K2_safe
        t = (1.0 - Khat_ij) if i == j else -Khat_ij
        return (v * t).astype(np.complex128)

    return builder


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_V_q_bispinor_to_h5(
    *,
    zeta_C_io,                                 # SlabIO for μ_L = 0  (charge ζ)
    zeta_T_ios: tuple,                         # length-3 SlabIO for μ_L = 1, 2, 3
    output_h5_path: Path | str,
    coulomb_kernels,                           # bundle from make_v_munu_chunked_kernel
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec,
    n_rmu_C: int,
    n_rmu_T: int,
    budget_bytes: float | None = None,
    bgw_v_grid_overlay_fn=None,                # only meaningful for CC tile
    print_fn=print,
    verbose: bool = True,
    backend=None,
    use_ffi_io: bool | None = None,
) -> Path:
    """Stream the 7 unique bispinor V_q^{μ_L, ν_L} tiles to HDF5.

    Each tile is computed by ``compute_V_q_tile`` (the same kernel that
    drives the scalar charge-only V_q) and written immediately as its
    own dataset in ``output_h5_path``.  Memory is freed between tiles.

    Parameters
    ----------
    zeta_C_io
        Open ``SlabIO`` to the μ_L=0 (charge) ζ file.
    zeta_T_ios
        Tuple of 3 open ``SlabIO`` handles for μ_L = 1, 2, 3.  All three
        must share the same centroid count ``n_rmu_T`` (the upstream
        4-channel ζ-fit emits one centroid set for the transverse
        channels via the Gordon-current density weight).
    output_h5_path
        Destination file.  Truncated if it exists.
    coulomb_kernels
        Result of :func:`gw.compute_vcoul.make_v_munu_chunked_kernel`.
        Provides ``sphere_idx`` and ``get_sqrt_v_and_phase`` (the same
        callables the scalar V_q uses).
    mesh_xy, kgrid, fft_grid, bvec
        Geometry of the run; mesh is the 2-D ('x', 'y') device mesh.
    n_rmu_C, n_rmu_T
        Centroid counts on the charge / transverse channels.
    budget_bytes
        Per-rank GPU memory budget for the V_q chooser.  The chooser is
        run independently on each tile shape (CC and TT may have
        different (n_rmu_L, n_rmu_R), so q_chunk may differ).

    Returns
    -------
    Path
        ``output_h5_path`` after all 7 unique tiles are flushed.

    On-disk layout (read by :class:`BispinorVqReader`):

        attrs:
            v_qmunu_format         = "bispinor_lorentz_v1"
            kgrid                  = (nkx, nky, nkz) int64[3]
            n_rmu_C, n_rmu_T       = int64
            n_q_total              = nkx · nky · nkz
            unique_tiles           = JSON list of (μ_L, ν_L)
            zero_tiles             = JSON list of (μ_L, ν_L)
            hermitian_pairs        = JSON list of [[j,i],[i,j]]

        datasets:
            V_qmunu_CC             (n_q_total, n_rmu_C, n_rmu_C) c128
            V_qmunu_CC_g0          (n_q_total, n_rmu_C)          c128
            V_qmunu_TT_11          (n_q_total, n_rmu_T, n_rmu_T) c128
            V_qmunu_TT_22          ...
            V_qmunu_TT_33          ...
            V_qmunu_TT_12          ...
            V_qmunu_TT_13          ...
            V_qmunu_TT_23          ...
    """
    from file_io.slab_io import SlabIO
    import h5py
    import json

    output_h5_path = Path(output_h5_path)

    if len(zeta_T_ios) != 3:
        raise ValueError(
            f"zeta_T_ios must be length 3 (μ_L = 1, 2, 3); got {len(zeta_T_ios)}."
        )

    # ----- Static geometry: K_cart on sphere + base v_per_G_fn ---------------
    sphere_idx = coulomb_kernels.sphere_idx
    K_cart_batch_fn = _make_K_cart_on_sphere_fn(sphere_idx, fft_grid, bvec)

    # Base charge v(K) — closures around get_sqrt_v_and_phase, identical
    # to what gw.compute_vcoul.compute_all_V_q builds for the scalar tile.
    _vmapped_sqrt_v_phase = jax.vmap(coulomb_kernels.get_sqrt_v_and_phase)

    def _base_v_per_G_fn(qvec_arr_jax):
        sqrt_v_batch, _ = _vmapped_sqrt_v_phase(qvec_arr_jax)
        return sqrt_v_batch * sqrt_v_batch              # = v(K)

    # ``phase_fn`` is gone — see ``compute_V_q_tile`` docstring.  The
    # per-q FFT-box phase ``exp(-2πi q·r)`` is applied separably by
    # ``apply_bloch_phase`` inside ``_zeta_disk_to_G`` /
    # ``zeta_reader._do_disk_to_G``.

    # ----- Open output (collective) and write metadata ---------------------
    nq_total = int(np.prod(kgrid))

    # First open: write the numeric metadata + truncate any prior file.
    # Per-tile open/close below: kept as a defense-in-depth against any
    # remaining MPI-IO datatype-cache state across different-shape
    # datasets (the SlabIO ``create_dataset`` now drains pending writes
    # so a single open should be safe; per-tile open is still a clean
    # serialisation boundary at negligible cost vs the kernel time).
    with SlabIO(output_h5_path, mode="w", mesh=mesh_xy,
                backend=backend, use_ffi_io=use_ffi_io) as io:
        io.write_attr("kgrid", np.asarray(kgrid, dtype=np.int64))
        io.write_attr("n_rmu_C", np.int64(n_rmu_C))
        io.write_attr("n_rmu_T", np.int64(n_rmu_T))
        io.write_attr("n_q_total", np.int64(nq_total))

    if True:
        for tile_idx, (mu_L, nu_L) in enumerate(UNIQUE_TILES):
            same_zeta = (mu_L == nu_L)
            zeta_L = zeta_C_io if mu_L == 0 else zeta_T_ios[mu_L - 1]
            zeta_R = zeta_C_io if nu_L == 0 else zeta_T_ios[nu_L - 1]

            n_rmu_L = n_rmu_C if mu_L == 0 else n_rmu_T
            n_rmu_R = n_rmu_C if nu_L == 0 else n_rmu_T

            # CC tile: write g0 (head correction).  TT tiles: g0_acc=None
            # is enough because the projector kills the q=0 head when K
            # aligns with the spatial axis; the residual finite-q head
            # is handled inside the kernel's q=0 special-case.
            wants_g0 = (mu_L == 0 and nu_L == 0)

            # The BGW v(q+G) overlay is meaningful only for the
            # charge-channel CC tile; transverse tiles are pure
            # implementation-side projector applications.
            overlay_fn = bgw_v_grid_overlay_fn if (mu_L == 0 and nu_L == 0) else None

            v_per_G_fn = _make_v_per_G_for_tile(
                base_v_per_G_fn=_base_v_per_G_fn,
                K_cart_batch_fn=K_cart_batch_fn,
                mu_L=mu_L, nu_L=nu_L,
            )


            # === Same primitive as the scalar V_q ============================
            # For all tiles: ``g0_acc=None`` and let the driver auto-allocate
            # (it auto-allocates whenever ``same_zeta=True``).  For the TT
            # diagonal tiles this allocates a g0 buffer we won't write; for
            # the CC tile it's exactly the head we want.  The off-diagonal
            # TT tiles have ``same_zeta=False`` → no g0 alloc.
            V_acc, g0_acc = compute_V_q_tile(
                zeta_L_io=zeta_L,
                zeta_R_io=None if same_zeta else zeta_R,
                v_per_G_fn=v_per_G_fn,
                sphere_idx=sphere_idx,
                fft_grid=fft_grid,
                mesh_xy=mesh_xy,
                kgrid=kgrid,
                n_rmu_L=n_rmu_L,
                n_rmu_R=n_rmu_R,
                V_acc=None,
                g0_acc=None,
                chooser_choice=None,
                budget_bytes=budget_bytes,
                bgw_v_grid_overlay_fn=overlay_fn,
                verbose=verbose,
                timing_label=f"V_q_bispinor[{mu_L},{nu_L}]",
            )

            # === Stream this tile to disk ====================================
            # ``compute_V_q_tile`` returns V_acc at the PADDED μ extent
            # (rounded to mesh_xy.shape product) so all interior
            # shardings divide cleanly.  We write only the LOGICAL
            # prefix to disk via SlabIO ``valid_shape`` — the file's
            # on-disk extent is always logical, so it round-trips
            # across any process count regardless of how the writing
            # job padded internally.  Mirror of the band-axis seam at
            # ``common/meta.py:99`` (b_id_4 padded vs b_id_4_user
            # logical for output writers).
            name = tile_dataset_name(mu_L, nu_L)
            v_padded_shape = tuple(int(s) for s in V_acc.shape)
            v_logical_shape = (int(v_padded_shape[0]), n_rmu_L, n_rmu_R)
            with SlabIO(output_h5_path, mode='a', mesh=mesh_xy,
                        backend=backend, use_ffi_io=use_ffi_io) as tile_io:
                tile_io.create_dataset(
                    name, shape=v_logical_shape, dtype=V_acc.dtype)
                tile_io.write_slab(
                    name, V_acc,
                    global_shape=v_logical_shape,
                    valid_shape=v_logical_shape)

                if wants_g0 and g0_acc is not None:
                    g0_padded_shape = tuple(int(s) for s in g0_acc.shape)
                    g0_logical_shape = (int(g0_padded_shape[0]), n_rmu_L)
                    tile_io.create_dataset(
                        f"{name}_g0", shape=g0_logical_shape,
                        dtype=g0_acc.dtype)
                    tile_io.write_slab(
                        f"{name}_g0", g0_acc,
                        global_shape=g0_logical_shape,
                        valid_shape=g0_logical_shape)

            # Free device memory before the next tile.  The kernel cache
            # in v_q_tile holds the compiled artifact for this shape; it
            # may be reused by another tile with the same (n_rmu_L,
            # n_rmu_R, same_zeta, write_g0) signature, so don't drop it.
            del V_acc, g0_acc

    # Format string + tile-layout JSON: rank-0 post-close write.  Strings
    # don't round-trip through SlabIO.write_attr (which doesn't accept
    # variable-length unicode); h5py handles them directly.  We store
    # ``v_qmunu_format`` as a fixed dataset (so the reader can h5-open
    # without rank coordination) and the tile-layout as file attrs.
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
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("v_q_bispinor_tile_layout_meta")
    except Exception:
        pass

    return output_h5_path


# ---------------------------------------------------------------------------
# G-flat bispinor orchestrator
# ---------------------------------------------------------------------------

def compute_V_q_bispinor_g_flat_to_h5(
    *,
    zeta_C_loader,                              # ZetaReader/Loader, charge ζ (G-flat)
    zeta_T_loaders: tuple,                      # length-3 ZetaReader/Loader, μ_L=1,2,3 (G-flat)
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
    backend=None,
    use_ffi_io: bool | None = None,
    print_fn=print,
    verbose: bool = True,
    # IBZ cascade plumbing (default disabled — must opt in via gw_init).
    sym=None,
    centroid_C_idx: np.ndarray | None = None,
    centroid_T_idx: np.ndarray | None = None,
    use_ibz: bool = False,
) -> Path:
    """Stream the 7 unique bispinor V_q^{μ_L, ν_L} tiles to HDF5 via the
    G-flat per-q + G-chunked path.

    Each tile shares the same orchestration as the scalar charge V_q —
    one q at a time, one per-tile ``v(q+G)`` table, contract via the
    G-chunked kernel into ``(n_q_full, n_rmu_L, n_rmu_R)``, write the
    tile, free the buffer.  Sharing comes from
    :func:`gw.v_q_g_flat._compute_V_q_g_flat_one_tile`; this function
    is just the 7-tile loop + per-tile HDF5 plumbing.

    Bispinor ζ files are written full-BZ (see ``gw_init.fit_zeta``
    `write_ibz_only=False` for the transverse μ_L=1..3 channels and
    charge under ``cfg.bispinor=True``), so we don't pass ``sym`` /
    ``centroid_indices``; the helper iterates the full BZ.

    On-disk layout matches :func:`compute_V_q_bispinor_to_h5`
    (legacy) — the reader :class:`BispinorVqReader` opens both
    formats interchangeably.
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
                backend=backend, use_ffi_io=use_ffi_io) as io:
        io.write_attr("kgrid", np.asarray(kgrid, dtype=np.int64))
        io.write_attr("n_rmu_C", np.int64(n_rmu_C))
        io.write_attr("n_rmu_T", np.int64(n_rmu_T))
        io.write_attr("n_q_total", np.int64(nq_total))

    # ------------------------------------------------------------------
    # IBZ cascade plumbing.  When ``use_ibz=True`` the bispinor ζ̃ files
    # were written IBZ-only (see ``gw_init.fit_zeta`` and
    # ``isdf_fitting.fit_zeta_to_h5``); the per-tile kernel iterates the
    # parent IBZ q-list and unfolds post-loop via ``unfold_v_q``.  Two
    # centroid sets ⇒ two orbit-closure checks (CC ↔ charge, TT ↔
    # transverse); they may diverge in principle but in practice are
    # generated together by the user's ``kmeans_cli --orbit-aware`` run.
    # If either set's closure check fails, fall back to full-BZ for the
    # corresponding tiles and log loudly.  See derivation §A5.
    # ------------------------------------------------------------------
    from .v_q_g_flat import _resolve_ibz_q_list

    def _ibz_tables_for(centroid_idx, label):
        if not use_ibz or sym is None or centroid_idx is None:
            return None, False
        tables = _resolve_ibz_q_list(
            sym=sym, centroid_indices=np.asarray(centroid_idx, dtype=np.int32),
            kgrid=kgrid, fft_grid=fft_grid, verbose=False)
        (_, _, _, _, _sym_perm, _L_table, _ok) = tables
        if not _ok and verbose and jax.process_index() == 0:
            print_fn(f"  [bispinor g-flat] {label}-centroid orbit closure "
                     f"failed — falling back to full-BZ for {label} tiles.  "
                     f"Regenerate centroids with ``kmeans_cli --orbit-aware`` "
                     f"to enable the IBZ cascade.")
        return tables, _ok

    _ibz_C, _use_ibz_C = _ibz_tables_for(centroid_C_idx, 'charge')
    _ibz_T, _use_ibz_T = _ibz_tables_for(centroid_T_idx, 'transverse')

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

    for tile_idx, (mu_L, nu_L) in enumerate(UNIQUE_TILES):
        same_zeta = (mu_L == nu_L)
        loader_L = zeta_C_loader if mu_L == 0 else zeta_T_loaders[mu_L - 1]
        loader_R = (None if same_zeta
                    else (zeta_C_loader if nu_L == 0
                           else zeta_T_loaders[nu_L - 1]))
        n_rmu_L = n_rmu_C if mu_L == 0 else n_rmu_T
        n_rmu_R = n_rmu_C if nu_L == 0 else n_rmu_T
        write_g0 = (mu_L == 0 and nu_L == 0)
        is_CC = (mu_L == 0 and nu_L == 0)

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
        )
        # BGW vcoul overlay only meaningful on the CC tile; transverse
        # tiles are pure projector applications.  Wrap the builder.
        if write_g0 and bgw_v_grid_fn is not None:
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
            kgrid=kgrid, fft_grid=fft_grid, bvec=bvec,
            mesh_xy=mesh_xy,
            g_chunk=g_chunk,
            sym=_tile_sym, centroid_indices=_tile_cent,
            write_g0=write_g0,
            timing_label=tile_dataset_name(mu_L, nu_L),
            verbose=verbose,
        )

        if is_CC or not _use_ibz_T:
            # Two cases stream straight to disk per-tile:
            #   * CC tile — never Lorentz-mixed (γ̃^0 = I is invariant).
            #   * TT tiles when the transverse IBZ cascade is off — the
            #     unfold inside ``_compute_V_q_g_flat_one_tile`` was a
            #     no-op (full-BZ path), no mixing needed.  Buffering all
            #     6 TT tiles would inflate peak memory; we preserve the
            #     pre-IBZ "free between tiles" behaviour exactly.
            name = tile_dataset_name(mu_L, nu_L)
            v_padded_shape = tuple(int(s) for s in V_acc.shape)
            v_logical_shape = (int(v_padded_shape[0]), n_rmu_L, n_rmu_R)
            with SlabIO(output_h5_path, mode='a', mesh=mesh_xy,
                        backend=backend, use_ffi_io=use_ffi_io) as tile_io:
                tile_io.create_dataset(
                    name, shape=v_logical_shape, dtype=V_acc.dtype)
                tile_io.write_slab(
                    name, V_acc,
                    global_shape=v_logical_shape,
                    valid_shape=v_logical_shape)
                if write_g0 and g0_acc is not None:
                    g0_padded_shape = tuple(int(s) for s in g0_acc.shape)
                    g0_logical_shape = (int(g0_padded_shape[0]), n_rmu_L)
                    tile_io.create_dataset(
                        f"{name}_g0", shape=g0_logical_shape,
                        dtype=g0_acc.dtype)
                    tile_io.write_slab(
                        f"{name}_g0", g0_acc,
                        global_shape=g0_logical_shape,
                        valid_shape=g0_logical_shape)
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
        from common.symmetry_maps import unfold_v_q_bispinor_lorentz

        # Synthesise the 3 Hermitian-redundant tiles (j, i) for i < j from
        # the stored upper-triangle.  These are needed as INPUTS to the
        # 3×3 contraction; we write only the 6 unique upper-triangle outputs.
        tt_full_in: dict[tuple[int, int], jax.Array] = dict(tt_buffer)
        for (j, i), (i_src, j_src) in HERMITIAN_PAIRS.items():
            tt_full_in[(j, i)] = jnp.conj(
                jnp.swapaxes(tt_buffer[(i_src, j_src)], -1, -2))

        tt_mixed = unfold_v_q_bispinor_lorentz(
            tt_full_in,
            sym_idx=np.asarray(sym.sym_idx_q, dtype=np.int32),
            R_proper_table=np.asarray(sym.R_proper, dtype=np.float64),
            mesh_xy=mesh_xy,
        )

        # Write the 6 unique upper-triangle TT tiles post-mix.
        for (mu_L, nu_L) in UNIQUE_TILES:
            if mu_L == 0 and nu_L == 0:
                continue
            V_mix = tt_mixed[(mu_L, nu_L)]
            name = tile_dataset_name(mu_L, nu_L)
            v_padded_shape = tuple(int(s) for s in V_mix.shape)
            v_logical_shape = (int(v_padded_shape[0]), n_rmu_T, n_rmu_T)
            with SlabIO(output_h5_path, mode='a', mesh=mesh_xy,
                        backend=backend, use_ffi_io=use_ffi_io) as tile_io:
                tile_io.create_dataset(
                    name, shape=v_logical_shape, dtype=V_mix.dtype)
                tile_io.write_slab(
                    name, V_mix,
                    global_shape=v_logical_shape,
                    valid_shape=v_logical_shape)
        del tt_full_in, tt_mixed
    del tt_buffer

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
    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("v_q_bispinor_g_flat_tile_layout_meta")
    except Exception:
        pass
    return output_h5_path


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class BispinorVqReader:
    """Open a bispinor V_q HDF5 written by :func:`compute_V_q_bispinor_to_h5`.

    Provides a uniform interface over the 16 (μ_L, ν_L) blocks:

    * ``get_tile(μ_L, ν_L)``        — JAX array on the (None, 'x', 'y')
                                       sharding, materialised on demand.
                                       For zero-by-gauge tiles returns
                                       a zeros array sized appropriately.
                                       For Hermitian-redundant tiles
                                       reads the companion + applies
                                       ``conj(swapaxes(.., -1, -2))``.
    * ``get_g0_CC()``                — charge-channel q=0 head
                                       (n_rmu_C,) c128 or None.

    Caller manages the lifecycle (use as a context manager).
    """

    def __init__(self, filename: Path | str, mesh_xy: Mesh,
                 backend=None, use_ffi_io: bool | None = None):
        from file_io.slab_io import SlabIO
        import h5py
        self._filename = Path(filename)
        self._mesh = mesh_xy
        self._io = SlabIO(self._filename, mode="r", mesh=mesh_xy,
                          backend=backend, use_ffi_io=use_ffi_io)
        self._io.__enter__()

        # Small metadata scalars are written via SlabIO.write_attr (which
        # creates a dataset in the file).  Read them via h5py — every rank
        # opens its own 'r' handle for a few-byte read; broadcast overhead
        # would dominate.
        with h5py.File(self._filename, "r") as f:
            def _read_scalar(name):
                d = f[name]
                v = d[()] if d.shape == () else d[:]
                return v
            fmt = _read_scalar("v_qmunu_format")
            if isinstance(fmt, bytes):
                fmt = fmt.decode("utf-8")
            self.kgrid = tuple(int(x) for x in _read_scalar("kgrid"))
            self.n_rmu_C = int(_read_scalar("n_rmu_C"))
            self.n_rmu_T = int(_read_scalar("n_rmu_T"))
            self.n_q_total = int(_read_scalar("n_q_total"))

        if str(fmt) != V_QMUNU_FORMAT:
            raise ValueError(
                f"{self._filename}: v_qmunu_format='{fmt}', "
                f"expected '{V_QMUNU_FORMAT}'.  Wrong file or stale "
                f"format from a different LORRAX revision."
            )

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
        mirrors the write-side ``_round_up_to_mesh`` at
        v_q_tile.py:1116-1118 (which pads to ``p_x*p_y``) and matches the
        ψ-side μ extent built by ``load_centroids_band_chunked`` — so a
        single pad here makes Σ^B's V tile broadcast against ψ with no
        further padding step in sigma_x_bispinor.

        Padding to ``gx*gy`` (rather than per-axis ``gx``/``gy``) is also
        what makes sharded reads with spec P(None,'x','y') divide
        cleanly under any 2D mesh factorisation."""
        proc = int(self._mesh.shape['x']) * int(self._mesh.shape['y'])
        def _pad(n):
            rem = n % proc
            return n + ((proc - rem) if rem else 0)
        return _pad(int(n_L)), _pad(int(n_R))

    def get_tile(self, mu_L: int, nu_L: int) -> jax.Array:
        """Return V^{μ_L, ν_L}_q as a sharded JAX array (n_q, n_L_padded,
        n_R_padded) c128.  When n_L/n_R aren't divisible by the mesh axis
        size, the trailing μ rows are zero-padded — mirrors the write-side
        invariant in v_q_tile._round_up_to_mesh and lets Σ^B run at any
        process count without a runtime divisibility error."""
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
            V_companion = self._io.read_slab(
                tile_dataset_name(*companion),
                shape=(self.n_q_total, n_L_p, n_R_p),
                valid_shape=(self.n_q_total, n_L_c, n_R_c),
                mesh=self._mesh, partition_spec=spec)
            # Hermitian: V[j,i](q,μ,ν) = V[i,j](q,ν,μ)*
            return jnp.conj(jnp.swapaxes(V_companion, -1, -2))
        # Direct read (member of UNIQUE_TILES)
        n_L, n_R = self._tile_shape(mu_L, nu_L)[1:]
        n_L_p, n_R_p = self._padded_shape_LR(n_L, n_R)
        return self._io.read_slab(
            tile_dataset_name(mu_L, nu_L),
            shape=(self.n_q_total, n_L_p, n_R_p),
            valid_shape=(self.n_q_total, n_L, n_R),
            mesh=self._mesh, partition_spec=spec)

    def get_g0_CC(self) -> jax.Array | None:
        """q=0 head for the charge channel only.  None if absent."""
        try:
            n_L = int(self.n_rmu_C)
            n_L_p, _ = self._padded_shape_LR(n_L, n_L)
            return self._io.read_slab(
                tile_dataset_name(0, 0) + "_g0",
                shape=(self.n_q_total, n_L_p),
                valid_shape=(self.n_q_total, n_L),
                mesh=self._mesh,
                partition_spec=P(None, 'x'))
        except Exception:
            return None

    @property
    def filename(self) -> Path:
        return self._filename
