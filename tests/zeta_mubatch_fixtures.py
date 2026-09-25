"""Fixtures for the route-G μ-batch gate (``tests/multi_device/zeta_mubatch_p4.py``).

A-cubic (diamond-H2, 48 operations incl. glides, 3 parents of 8 k, 48
centroids, real SymMaps from the committed WFN) with synthetic parent states,
and the order-two glide group with spin rotation and an antiunitary row
(ns = 2 or 4).  ``_children`` realizes the full-BZ children by the typed
r-space action (the transport C_q and the faces use).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_TOL = 1.0e-12
_RED = 1.0e-3
_HERE = Path(__file__).resolve().parent


def _crand(rng, *shape):
    import numpy as np
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


def _put(x, sharding):
    """Host array onto a (possibly multi-process) sharding, every process
    holding the same host copy."""
    import jax
    import numpy as np
    x = np.asarray(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])


def _host(x):
    """Global value of a (possibly multi-process) array on every process."""
    import numpy as np
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _grid_points(fft_grid):
    import numpy as np
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in fft_grid), indexing="ij")
    return np.stack([ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)],
                    axis=1).astype(np.int32)


def _acubic_fixture(mesh, rng):
    """Real A-cubic symmetry tables, synthetic parents on its 12^3 grid."""
    import numpy as np
    from file_io import WfnLoader
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan

    root = _HERE / "core" / "fixtures" / "A-cubic"
    with WfnLoader(root / "WFN.h5", backend="eager",
                   qe_schema=root / "data-file-schema.xml") as loader:
        sym = loader.symmetry()
        fft_grid = tuple(int(v) for v in loader.fft_grid)
        k_parent = np.asarray(loader.kvecs(k=sym.parent_k_domain))
        kgrid = tuple(int(v) for v in loader.kgrid)
    frac = np.loadtxt(root / "centroids_frac_48.txt")
    cent_idx = (np.rint(frac * np.asarray(fft_grid)) % np.asarray(fft_grid)).astype(np.int32)
    plan = build_centroid_k_unfold_plan(sym, cent_idx, fft_grid, mesh,
                                        nspinor=1, parent_k_frac=k_parent)
    fg = np.asarray(fft_grid)
    cent_flat = cent_idx[:, 0] * fg[1] * fg[2] + cent_idx[:, 1] * fg[2] + cent_idx[:, 2]
    nb = 8
    psi_parent = _crand(rng, plan.n_parent, nb, 1, int(fg.prod()))
    return dict(plan=plan, fft_grid=fft_grid, kgrid=kgrid, cent_flat=cent_flat,
                psi_parent=psi_parent, kfull=np.asarray(sym.unfolded_kpts),
                ops=np.asarray(sym.sym_matrices)[:plan.n_sym_spatial],
                tnp=np.asarray(sym.translations)[:plan.n_sym_spatial],
                band_chunks=((0, nb),), left=(0, 5), right=(2, 8), b_target=24,
                r_s_target=160, tile_width=192,
                rows=np.asarray(sym.active_symmetry_rows), spinor_action=sym.spinor_action)


def _glide_fixture(mesh, rng, ns, *, translated_anti=False, theta=0.7):
    """The order-two glide group with spin mixing and an antiunitary row.

    ``translated_anti`` sends k2 through glide followed by time reversal,
    so the Fourier transport must conjugate the nonzero glide phase.
    ``theta`` is the glide's spin rotation exp(-iθσ_x); only θ = π/2 makes it a
    representation (glide² = E needs U² = ±1), which covariant operands require.
    """
    import numpy as np
    from types import SimpleNamespace
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import centroid_source_map_and_wrap, spinor_rotation_for_sym_row

    fft_grid = (4, 4, 4)
    kgrid = (2, 2, 1)
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), swap])
    tnp = np.asarray([[0.0, 0.0, 0.0], [np.pi, np.pi, 0.0]])
    kints = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    kfrac = kints / np.asarray(kgrid, dtype=np.float64)
    irr = np.asarray([0, 1, 1, 2], dtype=np.int32)
    sym_rows = np.asarray([0, 0, 3 if translated_anti else 1, 2], dtype=np.int32)
    parent_k = kfrac[[0, 1, 3]]
    U1 = np.asarray([[np.cos(theta), -1j * np.sin(theta)],
                     [-1j * np.sin(theta), np.cos(theta)]])
    U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1])

    def spinor_action(rows, *, nspinor):
        return spinor_rotation_for_sym_row(U_spatial, np.asarray(rows), 2,
                                           nspinor=nspinor, R_cart=ops)

    sym = SimpleNamespace(sym_matrices=ops, translations=tnp, irr_idx_k=irr,
                          sym_idx_k=sym_rows, spinor_action=spinor_action,
                          unfolded_kpts=kfrac, kirr_fullids=np.asarray([0, 1, 3]))
    grid = _grid_points(fft_grid)
    perm_g, _ = centroid_source_map_and_wrap(grid, ops, tnp, fft_grid, extend_trs=True)
    cent = []
    for seed in (0, 5, 22, 27, 41, 50, 60, 63, 9, 14):
        orbit = sorted({int(perm_g[s, seed]) for s in range(4)})
        if len(cent) + len(orbit) <= 8 and not any(c in cent for c in orbit):
            cent.extend(orbit)
        if len(cent) == 8:
            break
    cent_flat = np.asarray(sorted(cent))
    plan = build_centroid_k_unfold_plan(sym, grid[cent_flat], fft_grid, mesh,
                                        nspinor=ns, parent_k_frac=parent_k)
    nb = 8
    psi_parent = _crand(rng, 3, nb, ns, 64)
    return dict(plan=plan, fft_grid=fft_grid, kgrid=kgrid, cent_flat=cent_flat,
                psi_parent=psi_parent, kfull=kfrac, ops=ops, tnp=tnp,
                band_chunks=((0, nb),), left=(0, 5), right=(2, 8), b_target=4,
                r_s_target=6, tile_width=16,
                rows=np.arange(4, dtype=np.int32), spinor_action=spinor_action)


def _children(fx):
    """Full-BZ children by the typed action (the (★) ψ unfold) on the grid."""
    import numpy as np
    from symmetry_maps import centroid_source_map_and_wrap

    plan, fg = fx["plan"], fx["fft_grid"]
    perm_g, L_g = centroid_source_map_and_wrap(
        _grid_points(fg), fx["ops"], fx["tnp"], fg, extend_trs=True)
    psi = fx["psi_parent"]
    out = np.empty((plan.n_full,) + psi.shape[1:], dtype=np.complex128)
    for k in range(plan.n_full):
        p, s = int(plan.irr_idx[k]), int(plan.sym_idx[k])
        val = psi[p][:, :, perm_g[s]] * np.exp(
            2j * np.pi * (L_g[s].astype(np.float64) @ plan.k_parent_frac[p]))[None, None, :]
        if s >= plan.n_sym_spatial:
            val = np.conj(val)
        out[k] = np.einsum("ac,ncr->nar", plan.spin_action_full[k], val)
    return out


def _rel(a, b):
    import numpy as np
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
