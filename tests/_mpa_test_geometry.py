"""One copy of the orbit-closed glide geometry the MPA suites share.

``tests/test_mpa_store.py`` and ``tests/test_mpa_fit_driver.py`` build
the same synthetic wedge: a centroid set that is the UNION OF A SEED
LIST'S ORBITS under a non-symmorphic glide, so
``verify_centroid_orbit_closure`` passes as a property of the builder
and not of a file somebody might regenerate.  The glide is chosen
(rather than a symmorphic group) so ``L_table`` is non-zero and VARIES
across μ, which keeps the umklapp phase live in the unfold arm — the
phase is ``exp(2πi q·(L_μ − L_ν))``, a DIFFERENCE, so a set whose
centroids share one L makes it identically 1 however large L is.

Parameterized by the seed list: each suite keeps its own seeds (with
the comments explaining their orbit structure) and passes them to
:func:`closed_centroid_set` / :func:`geometry`.  The standalone copy in
``tests/multi_device/mpa_fit_stream_gate.py`` is deliberate — that gate
must run without pytest or this package.

:class:`HostSlabIO` is the one serial-h5py stand-in for the collective
``SlabIO``: it exercises the collective ``mpa_store`` API in a single
host process at logical extents (PHDF5 itself has cluster tests).
Call-census tests subclass it with thin recording wrappers.
"""

from __future__ import annotations

import numpy as np

#: The synthetic FFT grid the centroid set lives on.
FFT = np.array([12, 12, 12], dtype=np.int64)

#: A GLIDE: {σ_z | τ = (1/2, 0, 0)}.  Order two — applying it twice
#: gives {I | (1, 0, 0)} ≡ the identity mod the lattice — so {I, g} is a
#: group and the orbits really close.  τ×grid = (6, 0, 0) is integer on
#: the 12-grid, so images land on grid points.
SYMS = np.stack([np.eye(3, dtype=np.int64),
                 np.diag([1, 1, -1]).astype(np.int64)])
TNP = np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])

#: 5 full-BZ q folding onto 3 IBZ parents, two of them through the
#: TIME-REVERSED rows (index >= n_sym_spatial = 2), so the antiunitary
#: branch of the unfold is live.
IRR = np.array([0, 1, 1, 2, 2], dtype=np.int32)
SYM = np.array([0, 1, 0, 3, 0], dtype=np.int32)
N_SYM_SPATIAL = 2
N_Q_IBZ = 3
N_Q_FULL = 5

#: IBZ q's with non-zero components on every axis, so exp(2πi q·L) is
#: not accidentally 1.
Q_IRR = np.array([[0.0, 0.0, 0.0],
                  [1 / 3, 0.0, 1 / 4],
                  [0.0, 1 / 3, 1 / 3]])


def closed_centroid_set(seeds):
    """The union of the seeds' orbits — closed under the group by
    definition, so the closure verdict is a property of this function."""
    S = np.asarray(SYMS, dtype=np.float64)
    rinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    tint = np.rint(np.asarray(TNP, dtype=np.float64) / (2.0 * np.pi)
                   * FFT).astype(np.int64)
    imgs = set()
    for r in np.asarray(seeds, dtype=np.int64):
        for s in range(S.shape[0]):
            imgs.add(tuple(((rinv[s] @ r + tint[s]) % FFT).tolist()))
    return np.array(sorted(imgs), dtype=np.int32)


def geometry(seeds):
    """``(tables, verdict, n_mu)`` for a seed list's closed set."""
    from symmetry_maps import (centroid_source_map_and_wrap,
                               verify_centroid_orbit_closure)
    from symmetry_maps import qirr_store as QS

    cent = closed_centroid_set(seeds)
    verdict = verify_centroid_orbit_closure(
        cent.astype(np.float64) / FFT, SYMS, tnp=TNP, fft_grid=FFT)
    assert verdict.closed, verdict.describe()
    perm, L = centroid_source_map_and_wrap(
        cent, SYMS, TNP, FFT, validate=True, extend_trs=True)
    tables = QS.QirrTables(IRR, SYM, Q_IRR, perm, L, N_SYM_SPATIAL)
    return tables, verdict, int(cent.shape[0])


class HostSlabIO:
    """Serial-h5py stand-in for the collective ``SlabIO``.

    Reads land as jax arrays placed by ``partition_spec`` on the mesh
    the reader handed in; writes clip against ``valid_shape`` when
    given (the fit-store geometry) and against ``global_shape``
    otherwise (the W-slab geometry), so device-dependent padding never
    reaches the file — the same two contracts the production class
    honours.
    """

    def __init__(self, path, *, mode, mesh):
        import h5py
        self.file = h5py.File(path, mode)
        self.mesh = mesh

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()

    def create_dataset(self, name, *, shape, dtype, **_):
        if name not in self.file:
            self.file.create_dataset(name, shape=shape, dtype=dtype)

    def read_slab(self, name, *, shape, offset, valid_shape,
                  partition_spec, **_):
        import jax
        from jax.sharding import NamedSharding

        out = np.zeros(shape, dtype=self.file[name].dtype)
        extent = tuple(min(valid_shape[d], shape[d],
                           self.file[name].shape[d] - offset[d])
                       for d in range(len(shape)))
        dst = tuple(slice(0, n) for n in extent)
        src = tuple(slice(offset[d], offset[d] + extent[d])
                    for d in range(len(shape)))
        out[dst] = self.file[name][src]
        return jax.device_put(
            out, NamedSharding(self.mesh, partition_spec))

    def write_slab(self, name, value, *, offset, global_shape=None,
                   valid_shape=None, **_):
        import jax

        host = np.asarray(jax.device_get(value))
        bound = valid_shape if valid_shape is not None else global_shape
        extent = tuple(min(bound[d], host.shape[d],
                           self.file[name].shape[d] - offset[d])
                       for d in range(host.ndim))
        dst = tuple(slice(offset[d], offset[d] + extent[d])
                    for d in range(host.ndim))
        src = tuple(slice(0, n) for n in extent)
        self.file[name][dst] = host[src]
