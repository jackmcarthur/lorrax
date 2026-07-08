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
            kgrid=kgrid, fft_grid=fft_grid,
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
    """Open a bispinor V_q HDF5 written by :func:`compute_V_q_bispinor_g_flat_to_h5`.

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
