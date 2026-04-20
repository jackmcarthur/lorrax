"""Memory-efficient WFN.h5 reader using the parallel-HDF5 FFI.

Unlike :class:`common.wfnreader.WFNReader` this reader does **not**
slurp ``wfns/coeffs`` into host memory at open time.  Coefficients are
pulled directly onto device, on demand, via the phdf5 FFI's compound
``H5S_SELECT_OR`` hyperslab read — suitable for WFN files that don't
fit in host RAM.

Scope
-----
**v1 — nosym only** (``ntran == 1``).  With nosym each file k-point
IS a full-BZ k-point so no unfolding is needed.  Symmetric-file
unfolding (on-device ``U_spinor`` × τ-phase × Umklapp applied to the
reader output) is a planned follow-up.

Usage
-----
::

    from common.phdf5_wfn_reader import PhdfWfnReader

    with PhdfWfnReader("WFN.h5", mesh=mesh) as wfn:
        psi_G = wfn.coeffs_gspace(band_range=(0, 80))              # all k
        psi_G = wfn.coeffs_gspace(band_range=(0, 80),
                                  k_ids=[5, 12, 0, 3])              # subset
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi.phdf5 import open_file, close_file
from ffi.phdf5.read import read_kchunk_union_sharded
from common.gvec_fft_box import (
    build_g_index_for_fft_box, make_fft_box_kernel)


__all__ = ["PhdfWfnReader"]


# WFN.h5 ``wfns/coeffs`` has shape (mnband, nspinor, ngktot, 2) = (band,
# spinor, packed-G, re/im).  The n_kchunk axis goes between spinor and
# G so HDF5's row-major SELECT_OR iteration matches memspace iteration
# — see ffi/phdf5/cpp/read_ffi.cc and ffi/phdf5/read.py.
_COEFFS = "wfns/coeffs"
_KCHUNK_AXIS = 2


class PhdfWfnReader:
    # --------------------------- lifecycle --------------------------------
    def __init__(self, path: str, *, mesh: Mesh):
        self.path = str(path)
        self.mesh = mesh
        self._read_header()
        self._open_parallel_file()
        self._g_index = self._build_g_index()

    def close(self) -> None:
        if self._ctx_handle:
            close_file(self._ctx_handle)
            self._ctx_handle = 0

    def __enter__(self) -> "PhdfWfnReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # --------------------------- user API --------------------------------
    def coeffs_gspace(
        self,
        band_range: tuple[int, int],
        *,
        k_ids: Sequence[int] | None = None,
    ) -> jax.Array:
        """G-space FFT box for one band chunk.

        Returns shape ``(len(k_ids), nb_padded, nspinor, nx, ny, nz)``
        complex128, sharded ``P(None, ('x','y'), None, None, None, None)``.
        ``nb_padded == band_range[1] - band_range[0]`` must divide the
        world size.  ``k_ids=None`` means all file k-points in file
        order.  Non-sorted ``k_ids`` are supported — the reader sorts
        internally for the HDF5 union read and permutes the output k
        axis back to the caller's order.
        """
        b_lo, b_hi = band_range
        nb = b_hi - b_lo
        world = self._world_size
        if nb % world:
            raise ValueError(f"band count {nb} not divisible by world={world}")
        bands_per_rank = nb // world

        k_ids = (np.arange(self.nkpts, dtype=np.int32) if k_ids is None
                 else np.asarray(k_ids, dtype=np.int32))
        n_kchunk = len(k_ids)

        # Row-major SELECT_OR iteration requires ascending file offsets.
        file_order = np.argsort(
            self.kpt_starts[k_ids], kind="stable").astype(np.int32)
        k_file_sorted = k_ids[file_order]

        offsets, counts = self._hyperslab_table(
            b_lo, bands_per_rank, k_file_sorted)
        g_index = jax.device_put(
            jnp.asarray(self._g_index[k_file_sorted]), self._rep4d)

        cnk_slab = self._reader(n_kchunk, bands_per_rank)(offsets, counts)
        psi_G = self._fft_box_kernel(n_kchunk, bands_per_rank)(
            cnk_slab, g_index)

        if not np.array_equal(file_order, np.arange(n_kchunk)):
            caller_order = np.argsort(file_order, kind="stable")
            psi_G = jnp.take(psi_G, jnp.asarray(caller_order), axis=0)
        return psi_G

    # --------------------------- setup phases -----------------------------
    def _read_header(self) -> None:
        import h5py
        with h5py.File(self.path, "r") as f:
            self.ntran = int(f["mf_header/symmetry/ntran"][()])
            if self.ntran != 1:
                raise NotImplementedError(
                    f"PhdfWfnReader v1 supports nosym files only "
                    f"(ntran == 1); got ntran={self.ntran}.")
            self.nbands = int(f["mf_header/kpoints/mnband"][()])
            self.nspinor = int(f["mf_header/kpoints/nspinor"][()])
            self.nkpts = int(f["mf_header/kpoints/nrk"][()])
            self.ngk = np.asarray(
                f["mf_header/kpoints/ngk"][:], dtype=np.int64)
            self.kpoints = np.asarray(
                f["mf_header/kpoints/rk"][:], dtype=np.float64)
            self.kgrid = np.asarray(
                f["mf_header/kpoints/kgrid"][:], dtype=np.int32)
            self.fft_grid = tuple(
                int(v) for v in f["mf_header/gspace/FFTgrid"][:])
            self.gvecs_all = np.asarray(
                f["wfns/gvecs"][:], dtype=np.int32)
            self.ecutwfc = float(f["mf_header/kpoints/ecutwfc"][()])
            self.ecutrho = float(f["mf_header/gspace/ecutrho"][()])

        self.ngkmax = int(self.ngk.max())
        self.ngktot = int(self.ngk.sum())
        self.kpt_starts = np.concatenate(
            [[0], np.cumsum(self.ngk)[:-1]]).astype(np.int64)

        self._world_size = (
            int(self.mesh.shape["x"]) * int(self.mesh.shape["y"]))
        self._rep2d = NamedSharding(self.mesh, P(None, None))
        self._rep4d = NamedSharding(self.mesh, P(None, None, None, None))

    def _open_parallel_file(self) -> None:
        self._ctx_handle = open_file(self.path, mesh=self.mesh, mode="r")

    def _build_g_index(self) -> np.ndarray:
        gvecs_per_k = [
            self.gvecs_all[self.kpt_starts[k]:self.kpt_starts[k] + self.ngk[k]]
            for k in range(self.nkpts)
        ]
        return build_g_index_for_fft_box(
            gvecs_per_k, self.fft_grid, self.ngkmax)

    # --------------------------- helpers ----------------------------------
    def _hyperslab_table(
        self, b_lo: int, bands_per_rank: int, k_file_sorted: np.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """(offsets, counts) for ``read_kchunk_union_sharded``.

        offsets[k] = (b_lo, 0, kpt_starts[k], 0) — file origin of this k's
        coefficient slab.  counts[k] = (bands_per_rank, nspinor, ngk[k], 2)
        — per-rank slab shape; using the *actual* ngk keeps per-k slabs
        disjoint even with variable ngk.
        """
        offsets = np.stack([
            [b_lo, 0, int(self.kpt_starts[k]), 0]
            for k in k_file_sorted
        ], axis=0).astype(np.int64)
        counts = np.stack([
            [bands_per_rank, self.nspinor, int(self.ngk[k]), 2]
            for k in k_file_sorted
        ], axis=0).astype(np.int64)
        return (
            jax.device_put(jnp.asarray(offsets), self._rep2d),
            jax.device_put(jnp.asarray(counts), self._rep2d),
        )

    # Cached jit callables — compile once per (n_kchunk, bands_per_rank).
    # Other compile-key inputs (ngkmax, nspinor, fft_grid, mesh, dtype)
    # are fixed for the reader's lifetime.
    @lru_cache(maxsize=16)
    def _reader(self, n_kchunk: int, bands_per_rank: int):
        return read_kchunk_union_sharded(
            self._ctx_handle, _COEFFS,
            n_kchunk=n_kchunk,
            kchunk_axis=_KCHUNK_AXIS,
            file_global_shape=(
                self.nbands, self.nspinor, self.ngktot, 2),
            per_rank_file_shape=(
                bands_per_rank, self.nspinor, self.ngkmax, 2),
            dtype=np.float64,
            mesh=self.mesh,
            file_partition_spec=P(("x", "y"), None, None, None),
        )

    @lru_cache(maxsize=16)
    def _fft_box_kernel(self, n_kchunk: int, bands_per_rank: int):
        return make_fft_box_kernel(
            mesh=self.mesh,
            nk=n_kchunk,
            ngkmax=self.ngkmax,
            nb_padded=bands_per_rank * self._world_size,
            nspinor=self.nspinor,
            fft_grid=self.fft_grid,
        )
