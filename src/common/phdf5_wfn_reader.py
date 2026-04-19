"""Memory-efficient WFN.h5 reader using the parallel-HDF5 FFI.

Unlike :class:`common.wfnreader.WFNReader` this reader does **not**
slurp ``wfns/coeffs`` into host memory at open time.  Coefficients are
pulled directly onto device, on demand, via the phdf5 FFI's compound
``H5S_SELECT_OR`` hyperslab read — suitable for WFN files that don't
fit in host RAM.

Scope
-----
**v1 — non-symmetric files only.**  Asserts ``ntran == 1`` at open.
With nosym, each k-point in the file is a full-BZ k-point and no
post-read symmetry unfolding is needed (identity spinor rotation, no
Umklapp, no fractional-translation phase).  Adding symmetric-file
support is an on-device ``U_spinor × phase × Umklapp`` kernel applied
to the output of this reader — planned but out of scope here.

Usage
-----
::

    from common.phdf5_wfn_reader import PhdfWfnReader

    with PhdfWfnReader("WFN.h5", mesh=mesh) as wfn:
        psi_G = wfn.load_band_chunk_gspace(band_range=(0, 80))
        # psi_G: (nk, nb_padded, nspinor, nx, ny, nz)
        # sharded P(None, ('x','y'), None, None, None, None)
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
    build_within_k_inv_map, make_gather_fft_box_kernel)


__all__ = ["PhdfWfnReader"]


# WFN.h5 layout: the coeffs dataset has shape (mnband, nspinor, ngktot, 2)
# along these axes, in this order.
_COEFFS_DATASET = "wfns/coeffs"
_AXIS_BAND = 0
_AXIS_SPINOR = 1
_AXIS_G = 2
_AXIS_RE_IM = 3
_N_FILE_DIMS = 4
# The n_kchunk axis should sit right before the G axis in the read
# output so HDF5's row-major SELECT_OR iteration order matches
# memspace iteration — see cpp/read_ffi.cc note 2 and
# ffi/phdf5/read.py::read_kchunk_union_sharded.
_KCHUNK_AXIS_IN_OUTPUT = 2


# =============================================================================
#  Reader
# =============================================================================
class PhdfWfnReader:
    """Lazy WFN.h5 reader.  Holds file metadata on host; streams
    coefficients through the phdf5 FFI."""

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------
    def __init__(self, path: str, *, mesh: Mesh):
        self.path = str(path)
        self.mesh = mesh
        self._read_metadata()
        self._assert_nosym()
        self._open_parallel_file()
        self._precompute_inv_map()

    def close(self) -> None:
        if self._ctx_handle != 0:
            close_file(self._ctx_handle)
            self._ctx_handle = 0

    def __enter__(self) -> "PhdfWfnReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    #  User-facing API
    # ------------------------------------------------------------------
    def load_band_chunk_gspace(
        self,
        band_range: tuple[int, int],
        *,
        k_subset: Sequence[int] | None = None,
    ) -> jax.Array:
        """Load one band chunk of the G-space FFT box.

        Parameters
        ----------
        band_range : (b_lo, b_hi)
            Half-open band range to load.  Must be divisible by
            ``world_size`` so the band axis shards evenly.
        k_subset : sequence of int, optional
            Physical k-indices (any order).  ``None`` (default) → all
            k-points.  The reader pre-sorts by file offset for the
            compound-hyperslab read and permutes the output k axis
            back to the caller's order.

        Returns
        -------
        psi_G : ``jax.Array`` of shape
            ``(n_kchunk, nb_padded, nspinor, nx, ny, nz)``, dtype
            complex128, sharded
            ``P(None, ('x','y'), None, None, None, None)``.
        """
        nb = band_range[1] - band_range[0]
        bands_per_rank = self._validate_band_divisibility(nb)

        k_physical = self._resolve_k_subset(k_subset)
        n_kchunk = len(k_physical)

        sorted_order, inverse_order = self._sort_by_file_offset(k_physical)
        k_sorted = k_physical[sorted_order]

        offsets_dev, counts_dev = self._build_offset_count_tables(
            band_range, bands_per_rank, k_sorted)

        inv_dev = self._get_inv_subset_on_device(k_sorted)

        # Dispatch: one H5Dread for all per-k windows, then one gather
        # into the (nk, nb, ns, nx, ny, nz) FFT box.
        union_reader = self._get_union_reader(n_kchunk, bands_per_rank)
        gather_kernel = self._get_gather_kernel(n_kchunk, bands_per_rank)

        slab_real_imag = union_reader(offsets_dev, counts_dev)
        psi_G = gather_kernel(slab_real_imag, inv_dev)

        if not np.array_equal(sorted_order, np.arange(n_kchunk)):
            psi_G = jnp.take(psi_G, jnp.asarray(inverse_order), axis=0)
        return psi_G

    # ------------------------------------------------------------------
    #  Setup phases — called once from __init__, kept separate for
    #  readability.
    # ------------------------------------------------------------------
    def _read_metadata(self) -> None:
        """Read header arrays via plain h5py (all small)."""
        import h5py
        with h5py.File(self.path, "r") as f:
            self.ntran = int(f["mf_header/symmetry/ntran"][()])
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

    def _assert_nosym(self) -> None:
        if self.ntran != 1:
            raise NotImplementedError(
                f"PhdfWfnReader v1 supports nosym WFN files only "
                f"(ntran == 1); got ntran={self.ntran}.  "
                f"Symmetric-file support (on-device U_spinor + "
                f"Umklapp + τ phase) is a planned follow-up.")

    def _open_parallel_file(self) -> None:
        self._ctx_handle = open_file(self.path, mesh=self.mesh, mode="r")

    def _precompute_inv_map(self) -> None:
        """Build the ``inv[k, nx, ny, nz]`` → g_within_slab map once."""
        gvecs_per_k = [
            self.gvecs_all[self.kpt_starts[k]:self.kpt_starts[k] + self.ngk[k]]
            for k in range(self.nkpts)
        ]
        self._inv_map_host = build_within_k_inv_map(
            gvecs_per_k, self.fft_grid, self.ngkmax)

    # ------------------------------------------------------------------
    #  Per-call helpers
    # ------------------------------------------------------------------
    def _validate_band_divisibility(self, nb: int) -> int:
        world = int(self.mesh.shape["x"]) * int(self.mesh.shape["y"])
        if nb % world != 0:
            raise ValueError(
                f"band count {nb} not divisible by world={world}; "
                f"pad the band range at the caller to a multiple of world.")
        return nb // world

    def _resolve_k_subset(
        self, k_subset: Sequence[int] | None,
    ) -> np.ndarray:
        if k_subset is None:
            return np.arange(self.nkpts, dtype=np.int32)
        ks = np.asarray(k_subset, dtype=np.int32)
        if ks.min() < 0 or ks.max() >= self.nkpts:
            raise ValueError(
                f"k_subset out of range: min={int(ks.min())} "
                f"max={int(ks.max())} vs nkpts={self.nkpts}")
        return ks

    def _sort_by_file_offset(
        self, k_physical: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sorted_order, inverse_order)`` s.t.
        ``k_physical[sorted_order]`` visits ``kpt_starts`` ascending,
        matching the SELECT_OR union's row-major iteration order."""
        file_offsets = self.kpt_starts[k_physical]
        sorted_order = np.argsort(file_offsets, kind="stable").astype(np.int32)
        inverse_order = np.argsort(sorted_order, kind="stable").astype(np.int32)
        return sorted_order, inverse_order

    def _build_offset_count_tables(
        self,
        band_range: tuple[int, int],
        bands_per_rank: int,
        k_sorted: np.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Per-k file-offset + per-rank-count tables, on device replicated."""
        b_lo = band_range[0]
        offsets = np.stack([
            np.array(
                [b_lo, 0, int(self.kpt_starts[k]), 0], dtype=np.int64)
            for k in k_sorted
        ], axis=0)
        counts = np.stack([
            np.array(
                [bands_per_rank, self.nspinor, int(self.ngk[k]), 2],
                dtype=np.int64)
            for k in k_sorted
        ], axis=0)

        replicated_2d = NamedSharding(self.mesh, P(None, None))
        return (
            jax.device_put(jnp.asarray(offsets), replicated_2d),
            jax.device_put(jnp.asarray(counts), replicated_2d),
        )

    def _get_inv_subset_on_device(self, k_sorted: np.ndarray) -> jax.Array:
        inv_subset = self._inv_map_host[k_sorted]
        return jax.device_put(
            jnp.asarray(inv_subset),
            NamedSharding(self.mesh, P(None, None, None, None)))

    # ------------------------------------------------------------------
    #  Jitted-callable cache — keyed on (n_kchunk, bands_per_rank).
    #  The compile signature also depends on ngkmax, nspinor, fft_grid,
    #  mesh, dtype; these are fixed for the reader's lifetime so they
    #  don't need to be in the key.
    # ------------------------------------------------------------------
    @lru_cache(maxsize=16)
    def _get_union_reader(
        self, n_kchunk: int, bands_per_rank: int,
    ):
        return read_kchunk_union_sharded(
            self._ctx_handle, _COEFFS_DATASET,
            n_kchunk=n_kchunk,
            kchunk_axis=_KCHUNK_AXIS_IN_OUTPUT,
            file_global_shape=(
                self.nbands, self.nspinor, self.ngktot, 2),
            per_rank_file_shape=(
                bands_per_rank, self.nspinor, self.ngkmax, 2),
            dtype=np.float64,
            mesh=self.mesh,
            file_partition_spec=P(("x", "y"), None, None, None),
        )

    @lru_cache(maxsize=16)
    def _get_gather_kernel(
        self, n_kchunk: int, bands_per_rank: int,
    ):
        world = int(self.mesh.shape["x"]) * int(self.mesh.shape["y"])
        nb_padded = bands_per_rank * world
        return make_gather_fft_box_kernel(
            mesh=self.mesh,
            nk=n_kchunk,
            ngkmax=self.ngkmax,
            nb_padded=nb_padded,
            nspinor=self.nspinor,
            fft_grid=self.fft_grid,
        )
