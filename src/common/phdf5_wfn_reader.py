"""Memory-efficient WFN.h5 reader using the parallel-HDF5 FFI.

Unlike :class:`common.wfnreader.WFNReader` this reader does **not**
slurp ``wfns/coeffs`` into host memory at open time.  Coefficients are
pulled directly onto device, on demand, via the phdf5 FFI's compound
``H5S_SELECT_OR`` hyperslab read — suitable for WFN files that don't
fit in host RAM.

Scope
-----
Handles both nosym and symmetric WFN files.  For symmetric files, a
per-rank unfold kernel applies the ``U_spinor × τ-phase ×
time-reversal-conjugation`` transform on device after the union read;
the reader dedupes by irreducible-BZ k so each file k-point is read
at most once per band chunk.

Usage
-----
::

    from common.phdf5_wfn_reader import PhdfWfnReader

    with PhdfWfnReader("WFN.h5", mesh=mesh) as wfn:
        psi_G = wfn.coeffs_gspace(band_range=(0, 80))              # all full-BZ k
        psi_G = wfn.coeffs_gspace(band_range=(0, 80),
                                  k_ids=[5, 12, 0, 3])              # subset
"""
from __future__ import annotations

import types
from functools import lru_cache
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from functools import partial

from ffi.phdf5 import open_file, close_file
from ffi.phdf5.read import read_kchunk_union_sharded
from common.gvec_fft_box import (
    build_g_index_for_fft_box, make_fft_box_kernel)
from common.symmetry_maps import SymMaps


__all__ = ["PhdfWfnReader"]


# WFN.h5 ``wfns/coeffs`` has shape (mnband, nspinor, ngktot, 2) = (band,
# spinor, packed-G, re/im).  The n_kchunk axis goes between spinor and
# G so HDF5's row-major SELECT_OR iteration matches memspace iteration
# — see ffi/phdf5/cpp/read_ffi.cc and ffi/phdf5/read.py.
_COEFFS = "wfns/coeffs"
_KCHUNK_AXIS = 2


class PhdfWfnReader:
    # =====================================================================
    #  Lifecycle
    # =====================================================================
    def __init__(self, path: str, *, mesh: Mesh):
        self.path = str(path)
        self.mesh = mesh
        self._read_header()
        self._build_symmetry_tables()
        self._open_parallel_file()
        self._g_index = self._build_g_index()
        self._device_put_static_tables()

    def close(self) -> None:
        if self._ctx_handle:
            close_file(self._ctx_handle)
            self._ctx_handle = 0

    def __enter__(self) -> "PhdfWfnReader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # =====================================================================
    #  User API
    # =====================================================================
    def coeffs_gspace(
        self,
        band_range: tuple[int, int],
        *,
        k_ids: Sequence[int] | None = None,
        bispinor: bool = False,
    ) -> jax.Array:
        """G-space FFT box for one band chunk, at a set of full-BZ k-points.

        Returns shape ``(len(k_ids), nb_padded, ns_out, nx, ny, nz)``
        complex128, sharded ``P(None, ('x','y'), None, None, None, None)``,
        where ``ns_out = self.nspinor`` for the default 2-spinor read
        and ``ns_out = 4`` when ``bispinor=True`` (kinetic-balance lift
        ψ_S = (α/2)(σ·(k+G)) ψ_L applied per FFT-box bin; mirrors
        ``common.load_wfns.get_small_psi_component``).
        ``nb_padded = band_range[1] - band_range[0]`` must divide the
        world size.  ``k_ids=None`` means all full-BZ k-points in
        index order.

        ``k_ids`` are indices into the full-BZ (``0..nk_full-1``).  The
        reader figures out which irreducible-BZ k-points to read from
        the file, dedupes, reads once per unique IBZ k, then applies
        the relevant symmetry unfold (``U_spinor × τ-phase × optional
        TR conjugation``) on device.  k_ids may be in any order — the
        output k axis is returned in the caller's requested order.
        """
        if bispinor and self.nspinor != 2:
            raise ValueError(
                f"bispinor lift requires a 2-spinor source wfn; "
                f"file has nspinor={self.nspinor}.")
        b_lo, b_hi = band_range
        nb = b_hi - b_lo
        if nb % self._world_size:
            raise ValueError(
                f"band count {nb} not divisible by world={self._world_size}")
        bands_per_rank = nb // self._world_size

        k_ids = (np.arange(self.nk_full, dtype=np.int32) if k_ids is None
                 else np.asarray(k_ids, dtype=np.int32))
        n_k = len(k_ids)

        # Which IBZ k-points do we need to read?  Dedupe and sort by
        # file offset (the ascending-file-offset sort is required by
        # H5S_SELECT_OR; see read_kchunk_union_sharded docstring).
        ibz_per_id = self._ibz_per_full_k[k_ids]
        unique_ibz, ibz_inv = np.unique(ibz_per_id, return_inverse=True)
        file_order = np.argsort(
            self.kpt_starts[unique_ibz], kind="stable").astype(np.int32)
        ibz_file_sorted = unique_ibz[file_order]
        # Position of each k_id in ibz_file_sorted.
        position_in_reads = np.argsort(file_order, kind="stable")[ibz_inv]
        position_in_reads = position_in_reads.astype(np.int32)

        # Read unique IBZ k slabs.
        offsets, counts = self._hyperslab_table(
            b_lo, bands_per_rank, ibz_file_sorted)
        n_reads = len(ibz_file_sorted)
        cnk_at_ibz = self._reader(n_reads, bands_per_rank)(offsets, counts)

        # Stage the k_ids to device once; the unfold + gather kernels
        # consume slices of the pre-staged full-BZ tables via jnp.take.
        k_ids_dev = jax.device_put(jnp.asarray(k_ids), self._rep1d)

        # Unfold to full-BZ: expand along k axis + τ phase + U_spinor +
        # optional TR conjugation, all on device.  Skipped entirely for
        # nosym files (ntran == 1), where file-k == full-BZ-k,
        # ``position_in_reads`` is identity, U is identity, phase is 1,
        # and no TR conjugation applies — all of which the ``fft_box``
        # kernel handles directly from the union-read output.
        if self.ntran == 1:
            cnk_at_full = cnk_at_ibz
        else:
            U_k, phase_k, tr_mask_k = self._sym_tables_for_ids(k_ids_dev)
            cnk_at_full = self._unfold_kernel(n_reads, n_k, bands_per_rank)(
                cnk_at_ibz, U_k, phase_k, tr_mask_k, position_in_reads)

        # Slice the full-BZ-k g_index down to the caller's k_ids on
        # device, then scatter into FFT box via the shared gather kernel.
        g_index_for_ids = jnp.take(self._g_index_dev, k_ids_dev, axis=0)
        psi_G = self._fft_box_kernel(n_k, bands_per_rank)(
            cnk_at_full, g_index_for_ids)
        if bispinor:
            # Apply kinetic-balance lift to lift the 2-spinor ψ_L to a
            # 4-spinor (ψ_L, ψ_S) using the FFT-bin (k+G)_cart.  Output
            # spinor axis grows from 2 → 4.
            kfrac_for_ids = jnp.take(self._kfrac_dev, k_ids_dev, axis=0)
            psi_G = self._bispinor_lift_kernel(n_k, bands_per_rank)(
                psi_G, kfrac_for_ids)
        return psi_G

    # =====================================================================
    #  Setup phases
    # =====================================================================
    def _read_header(self) -> None:
        import h5py
        with h5py.File(self.path, "r") as f:
            hdr = f["mf_header"]
            self.nbands = int(hdr["kpoints/mnband"][()])
            self.nspinor = int(hdr["kpoints/nspinor"][()])
            self.nkpts_ibz = int(hdr["kpoints/nrk"][()])
            self.ngk = np.asarray(hdr["kpoints/ngk"][:], dtype=np.int64)
            self.kpoints_ibz = np.asarray(
                hdr["kpoints/rk"][:], dtype=np.float64)
            self.kgrid = np.asarray(
                hdr["kpoints/kgrid"][:], dtype=np.int32)
            self.shift = np.asarray(
                hdr["kpoints/shift"][:], dtype=np.float64)
            self.fft_grid = tuple(
                int(v) for v in hdr["gspace/FFTgrid"][:])

            self.ntran = int(hdr["symmetry/ntran"][()])
            self.sym_matrices_all = np.asarray(
                hdr["symmetry/mtrx"][:], dtype=np.int32)  # (48, 3, 3)
            self.translations_all = np.asarray(
                hdr["symmetry/tnp"][:], dtype=np.float64)  # (48, 3)

            # crystal basis vectors — SymMaps needs bvec for the
            # crystal-to-cartesian rotation of spinor axes.
            self.bvec = np.asarray(
                hdr["crystal/bvec"][:], dtype=np.float64)
            # alat — needed by the bispinor kinetic-balance lift
            # ψ_S = (α/2)(σ·(k+G)) ψ_L, where (k+G)_cart = 2π/alat (k+G)·bvec.
            self.alat = float(hdr["crystal/alat"][()])

            self.gvecs_all = np.asarray(f["wfns/gvecs"][:], dtype=np.int32)
            self.ecutwfc = float(hdr["kpoints/ecutwfc"][()])
            self.ecutrho = float(hdr["gspace/ecutrho"][()])

        self.ngkmax = int(self.ngk.max())
        self.ngktot = int(self.ngk.sum())
        self.kpt_starts = np.concatenate(
            [[0], np.cumsum(self.ngk)[:-1]]).astype(np.int64)

        self._world_size = (
            int(self.mesh.shape["x"]) * int(self.mesh.shape["y"]))
        self._rep1d = NamedSharding(self.mesh, P(None))
        self._rep2d = NamedSharding(self.mesh, P(None, None))
        self._rep4d = NamedSharding(self.mesh, P(None, None, None, None))

    def _build_symmetry_tables(self) -> None:
        """Precompute per-full-BZ-k quantities for the unfold kernel.

        Builds ``SymMaps`` from a stub with just the fields SymMaps
        reads, then collects the per-nk lookups into flat arrays.
        """
        wfn_stub = types.SimpleNamespace(
            ntran=self.ntran,
            sym_matrices=self.sym_matrices_all[:self.ntran],
            translations=self.translations_all[:self.ntran],
            kpoints=self.kpoints_ibz,
            kgrid=self.kgrid,
            shift=self.shift,
            nkpts=self.nkpts_ibz,
            bvec=self.bvec,
        )
        self._sym = SymMaps(wfn_stub)
        self.nk_full = int(self._sym.nk_tot)

        self._ibz_per_full_k = np.asarray(
            self._sym.irk_to_k_map, dtype=np.int32)
        self._sym_idx_per_full_k = np.asarray(
            self._sym.irk_sym_map, dtype=np.int32)
        self._tr_mask_per_full_k = (
            self._sym_idx_per_full_k >= self.ntran).astype(np.bool_)

        # Per-nk 2x2 U_spinor.
        self._U_spinor_per_full_k = np.asarray(
            self._sym.U_spinor[self._sym_idx_per_full_k], dtype=np.complex128)
        # Per-nk fractional-translation phase on the kbar's ngkmax grid.
        # Zeros past ngk[kbar] (the cnk values are zero there anyway).
        # For TR-symmetry nk's the physical phase is "1" (the
        # legacy SymMaps returns None in the TR branch), matching the
        # nosym handling; the TR conj is applied separately in the
        # unfold kernel.
        self._phase_per_full_k = self._compute_phases_all_full_k()

    def _compute_phases_all_full_k(self) -> np.ndarray:
        """``phase[nk_full, g]`` = ``exp(-i (S·G_bar) · τ)`` zero-padded
        to ngkmax.  For TR-symmetry nk's, unit phase."""
        phase = np.ones((self.nk_full, self.ngkmax), dtype=np.complex128)
        for nk in range(self.nk_full):
            sym_idx = int(self._sym_idx_per_full_k[nk])
            if sym_idx >= self.ntran:
                continue                           # TR: unit phase
            tau = self.translations_all[sym_idx]
            if not np.any(np.abs(tau) > 1e-12):
                continue                           # symmorphic: unit phase
            ibz = int(self._ibz_per_full_k[nk])
            g_bar = self.gvecs_all[
                self.kpt_starts[ibz]:self.kpt_starts[ibz] + int(self.ngk[ibz])]
            rotated = (self._sym.sym_mats_k[sym_idx].astype(np.int32) @ g_bar.T).T
            phase[nk, :g_bar.shape[0]] = np.exp(
                -1j * rotated.astype(np.float64) @ tau)
        return phase

    def _open_parallel_file(self) -> None:
        self._ctx_handle = open_file(self.path, mesh=self.mesh, mode="r")

    def _device_put_static_tables(self) -> None:
        """Move the four per-full-BZ-k lookup tables onto the devices
        once at init — they don't depend on the band range or the
        k-chunk, so staging them on every call would be a 5.8 MB /
        2.2 MB / 288 kB / 9 B H2D per call for MoS2 3×3.  With
        ``k_ids`` a numpy array we then take a JAX ``take`` on device
        to slice the requested rows, which is ~µs on-GPU instead of
        ms-scale per-call H2D."""
        U_sharding    = NamedSharding(self.mesh, P(None, None, None))
        phase_sharding = NamedSharding(self.mesh, P(None, None))
        tr_sharding   = NamedSharding(self.mesh, P(None))
        self._g_index_dev = jax.device_put(
            jnp.asarray(self._g_index), self._rep4d)
        self._U_dev = jax.device_put(
            jnp.asarray(self._U_spinor_per_full_k), U_sharding)
        self._phase_dev = jax.device_put(
            jnp.asarray(self._phase_per_full_k), phase_sharding)
        self._tr_mask_dev = jax.device_put(
            jnp.asarray(self._tr_mask_per_full_k), tr_sharding)
        # Full-BZ k-points (fractional, replicated) for the bispinor lift.
        self._kfrac_dev = jax.device_put(
            jnp.asarray(self._sym.unfolded_kpts, dtype=jnp.float64),
            self._rep2d)

    def _build_g_index(self) -> np.ndarray:
        """``g_index[nk_full, nx, ny, nz]`` mapping each FFT-box cell to
        the g-position within the IBZ slab that maps here under the
        full-BZ k's symmetry operation."""
        gvecs_per_full_k = [
            self._rotated_gvecs_for_full_k(nk) for nk in range(self.nk_full)
        ]
        return build_g_index_for_fft_box(
            gvecs_per_full_k, self.fft_grid, self.ngkmax)

    def _rotated_gvecs_for_full_k(self, nk: int) -> np.ndarray:
        """G-vectors that appear at full-BZ k = nk, as integer
        reciprocal-lattice coordinates — i.e. ``S · G_bar − kg0`` where
        G_bar are the IBZ k's stored gvecs, S is the symmetry rotation
        on gvecs (``sym_mats_k``), and kg0 is BGW's Umklapp vector."""
        sym_idx = int(self._sym_idx_per_full_k[nk])
        ibz = int(self._ibz_per_full_k[nk])
        g_bar = self.gvecs_all[
            self.kpt_starts[ibz]:self.kpt_starts[ibz] + int(self.ngk[ibz])]
        sym_krep = self._sym.sym_mats_k[sym_idx].astype(np.int32)
        kg0 = self._umklapp_vector(nk, sym_idx, ibz, sym_krep)
        return (sym_krep @ g_bar.T).T - kg0[None, :]

    def _umklapp_vector(
        self, nk: int, sym_idx: int, ibz: int, sym_krep: np.ndarray,
    ) -> np.ndarray:
        """BGW's kg0 such that k_full = S·k_ibz + kg0, as int32."""
        if sym_idx >= self.ntran:
            q_full = sym_krep @ self.kpoints_ibz[ibz]
            q_inzone = q_full % 1.0
            q_inzone[q_inzone > 0.9999] = 0.0
            return (q_inzone - q_full).astype(np.int32)
        k_full = self._sym.unfolded_kpts[nk]
        return np.rint(
            k_full - sym_krep @ self.kpoints_ibz[ibz]).astype(np.int32)

    # =====================================================================
    #  Per-call helpers
    # =====================================================================
    def _hyperslab_table(
        self, b_lo: int, bands_per_rank: int, ibz_file_sorted: np.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """(offsets, counts) for the union read: one row per unique IBZ
        k in ascending file-offset order.  Count uses the actual ngk per
        IBZ k so the compound hyperslabs are disjoint by construction.
        """
        offsets = np.stack([
            [b_lo, 0, int(self.kpt_starts[ibz]), 0]
            for ibz in ibz_file_sorted
        ], axis=0).astype(np.int64)
        counts = np.stack([
            [bands_per_rank, self.nspinor, int(self.ngk[ibz]), 2]
            for ibz in ibz_file_sorted
        ], axis=0).astype(np.int64)
        return (
            jax.device_put(jnp.asarray(offsets), self._rep2d),
            jax.device_put(jnp.asarray(counts), self._rep2d),
        )

    def _sym_tables_for_ids(
        self, k_ids_dev: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Slice the pre-device-put full-BZ sym tables down to the
        caller's k_ids on the device.  ``k_ids_dev`` is the replicated
        int32 index array."""
        return (
            jnp.take(self._U_dev,       k_ids_dev, axis=0),
            jnp.take(self._phase_dev,   k_ids_dev, axis=0),
            jnp.take(self._tr_mask_dev, k_ids_dev, axis=0),
        )

    # =====================================================================
    #  Jitted-callable cache
    #  Keys: (n_reads, bands_per_rank) for the reader; (n_k, bpr) for
    #  the unfold and FFT-box kernels.  Other compile-key fixed for the
    #  reader's lifetime.
    # =====================================================================
    @lru_cache(maxsize=16)
    def _reader(self, n_reads: int, bands_per_rank: int):
        return read_kchunk_union_sharded(
            self._ctx_handle, _COEFFS,
            n_kchunk=n_reads,
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
    def _unfold_kernel(
        self, n_reads: int, n_k: int, bands_per_rank: int,
    ):
        return _make_unfold_kernel(
            mesh=self.mesh,
            n_reads=n_reads,
            n_k=n_k,
            ngkmax=self.ngkmax,
            bands_per_rank=bands_per_rank,
            nspinor=self.nspinor,
        )

    @lru_cache(maxsize=16)
    def _fft_box_kernel(self, n_k: int, bands_per_rank: int):
        return make_fft_box_kernel(
            mesh=self.mesh,
            nk=n_k,
            ngkmax=self.ngkmax,
            nb_padded=bands_per_rank * self._world_size,
            nspinor=self.nspinor,
            fft_grid=self.fft_grid,
        )

    @lru_cache(maxsize=16)
    def _bispinor_lift_kernel(self, n_k: int, bands_per_rank: int):
        return _make_bispinor_lift_kernel(
            mesh=self.mesh,
            n_k=n_k,
            bands_per_rank=bands_per_rank,
            fft_grid=self.fft_grid,
            bvec=self.bvec,
            alat=self.alat,
        )


# =============================================================================
#  Unfold kernel — takes the re/im-packed union-read output at IBZ k-points
#  and produces the equivalent re/im-packed slab at the full-BZ k-points,
#  with per-nk τ phase, U_spinor rotation, and optional TR conjugation applied.
# =============================================================================
def _make_unfold_kernel(
    mesh: Mesh, n_reads: int, n_k: int, ngkmax: int,
    bands_per_rank: int, nspinor: int,
):
    def _per_rank(
        cnk_at_ibz,          # (bpr, ns, n_reads, ngkmax, 2) f64
        U_per_k,             # (n_k, ns, ns) c128
        phase_per_k,         # (n_k, ngkmax) c128
        tr_mask_per_k,       # (n_k,) bool
        position_in_reads,   # (n_k,) int32 — which IBZ slab serves each k
    ):
        # Re/im pack → complex.
        cnk = cnk_at_ibz[..., 0] + 1j * cnk_at_ibz[..., 1]   # (bpr, ns, n_reads, ngkmax)

        # Expand along the k axis: pick, for each full-BZ k, the IBZ slab
        # that it unfolds from.
        cnk = jnp.take(cnk, position_in_reads, axis=2)       # (bpr, ns, n_k, ngkmax)

        # Time-reversal conjugation: applied BEFORE the τ phase and
        # U_spinor rotation to match common.symmetry_maps' get_cnk_fullzone_batch.
        # For TR nk's, phase_per_k was set to 1 so the subsequent multiply
        # is a no-op.
        cnk = jnp.where(
            tr_mask_per_k[None, None, :, None], jnp.conj(cnk), cnk)

        # τ phase per k, per g.
        cnk = cnk * phase_per_k[None, None, :, :]

        # U_spinor rotation: spinor_out[a] = Σ_b U[k, a, b] · spinor_in[b].
        cnk = jnp.einsum("kac,bckg->bakg", U_per_k, cnk)

        # Repack for the downstream FFT-box kernel (which expects re/im).
        return jnp.stack([cnk.real, cnk.imag], axis=-1)

    sharded = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(
            P(("x", "y"), None, None, None, None),   # cnk_at_ibz
            P(None, None, None),                      # U_per_k
            P(None, None),                            # phase_per_k
            P(None),                                  # tr_mask_per_k
            P(None),                                  # position_in_reads
        ),
        out_specs=P(("x", "y"), None, None, None, None),
        check_rep=False,
    )
    return jax.jit(sharded)


# =============================================================================
#  Bispinor kinetic-balance lift in FFT-box G-space
#  ψ_S = (α/2) (σ·(k+G)) ψ_L on each FFT-bin, mirrors
#  ``common.bispinor_init.get_small_psi_component`` but vectorised over the
#  dense FFT box (bins past ngk[k] have ψ_L = 0 so contribute zero).
# =============================================================================
def _make_bispinor_lift_kernel(
    *, mesh: Mesh, n_k: int, bands_per_rank: int,
    fft_grid: tuple[int, int, int], bvec: np.ndarray, alat: float,
):
    nx, ny, nz = (int(v) for v in fft_grid)
    bvec_jax = jnp.asarray(bvec, dtype=jnp.float64)
    halfalpha = jnp.complex128(0.00364867628215)
    tpi_over_alat = float(2.0 * np.pi / alat)

    def _per_rank(psi_L_box, k_frac_per_k):
        # psi_L_box: (n_k, bpr, 2, nx, ny, nz) c128, band axis sharded on (x,y)
        # k_frac_per_k: (n_k, 3) f64, replicated
        gx = jnp.fft.fftfreq(nx, d=1.0 / nx).astype(jnp.float64)
        gy = jnp.fft.fftfreq(ny, d=1.0 / ny).astype(jnp.float64)
        gz = jnp.fft.fftfreq(nz, d=1.0 / nz).astype(jnp.float64)
        # K_frac per (k, bin): k + G.  Build axis-by-axis to keep the
        # intermediate small (no full meshgrid).
        # K_cart_axis[i] = (k_frac[:, 0] * bvec[0, i]
        #                   + (Gx[None, :, None, None] + ...) * bvec[..., i])
        # We compute K_x, K_y, K_z scalars per (k, bin) directly.
        kx = (k_frac_per_k[:, 0:1, None, None] * bvec_jax[0, 0]
              + k_frac_per_k[:, 1:2, None, None] * bvec_jax[1, 0]
              + k_frac_per_k[:, 2:3, None, None] * bvec_jax[2, 0]
              + gx[None, :, None, None] * bvec_jax[0, 0]
              + gy[None, None, :, None] * bvec_jax[1, 0]
              + gz[None, None, None, :] * bvec_jax[2, 0])
        ky = (k_frac_per_k[:, 0:1, None, None] * bvec_jax[0, 1]
              + k_frac_per_k[:, 1:2, None, None] * bvec_jax[1, 1]
              + k_frac_per_k[:, 2:3, None, None] * bvec_jax[2, 1]
              + gx[None, :, None, None] * bvec_jax[0, 1]
              + gy[None, None, :, None] * bvec_jax[1, 1]
              + gz[None, None, None, :] * bvec_jax[2, 1])
        kz = (k_frac_per_k[:, 0:1, None, None] * bvec_jax[0, 2]
              + k_frac_per_k[:, 1:2, None, None] * bvec_jax[1, 2]
              + k_frac_per_k[:, 2:3, None, None] * bvec_jax[2, 2]
              + gx[None, :, None, None] * bvec_jax[0, 2]
              + gy[None, None, :, None] * bvec_jax[1, 2]
              + gz[None, None, None, :] * bvec_jax[2, 2])
        Kx = (tpi_over_alat * kx).astype(jnp.complex128)   # (n_k, nx, ny, nz)
        Ky = (tpi_over_alat * ky).astype(jnp.complex128)
        Kz = (tpi_over_alat * kz).astype(jnp.complex128)
        # σ·K matrix elements: σ_z=Kz, σ_+=Kx+iKy, σ_−=Kx−iKy.
        Km = (Kx - 1j * Ky)[:, None, ...]   # (n_k, 1, nx, ny, nz) for broadcast over bands
        Kp = (Kx + 1j * Ky)[:, None, ...]
        Kz_b = Kz[:, None, ...]
        psi_L_0 = psi_L_box[:, :, 0, ...]
        psi_L_1 = psi_L_box[:, :, 1, ...]
        psi_S_0 = halfalpha * (Kz_b * psi_L_0 + Km * psi_L_1)
        psi_S_1 = halfalpha * (Kp * psi_L_0 - Kz_b * psi_L_1)
        # (n_k, bpr, 4, nx, ny, nz)
        return jnp.stack([psi_L_0, psi_L_1, psi_S_0, psi_S_1], axis=2)

    sharded = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(
            P(None, ("x", "y"), None, None, None, None),
            P(None, None),
        ),
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_rep=False,
    )
    return jax.jit(sharded)
