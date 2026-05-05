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
    ) -> jax.Array:
        """G-space FFT box for one band chunk, at a set of full-BZ k-points.

        Returns shape ``(len(k_ids), nb_padded, nspinor, nx, ny, nz)``
        complex128, sharded ``P(None, ('x','y'), None, None, None, None)``.
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
        b_lo, b_hi = band_range
        nb = b_hi - b_lo
        if nb % self._world_size:
            raise ValueError(
                f"band count {nb} not divisible by world={self._world_size} — "
                f"caller must pre-pad band_range (see common.meta.Meta.from_system "
                f"for the driver-level pad).")

        # Common path: file holds every requested band.
        if b_hi <= self.nbands:
            return self._coeffs_gspace_aligned(b_lo, b_hi, k_ids)

        # File-short path: pad target exceeds wfn.nbands.  Split the
        # request into a fast collective bulk read of the largest
        # world-aligned in-file slice + a small replicated remainder
        # (≤ world-1 in-file "tail" bands plus pure-zero padding).
        # The remainder is handled by ``_coeffs_gspace_replicated_tail``;
        # all three pieces are concatenated along the band axis and
        # resharded to the canonical layout.
        n_in_file = max(0, int(self.nbands) - int(b_lo))
        n_aligned = (n_in_file // self._world_size) * self._world_size
        return self._coeffs_gspace_with_short_tail(
            b_lo, b_hi, n_aligned, k_ids)

    def _coeffs_gspace_aligned(
        self,
        b_lo: int,
        b_hi: int,
        k_ids: Sequence[int] | None,
    ) -> jax.Array:
        """Fast path: collective FFI read assuming ``b_hi <= self.nbands``
        and ``(b_hi - b_lo) % world == 0``.  This is the body the
        public ``coeffs_gspace`` ran before pad-past-file support; the
        sharded output layout is identical."""
        nb = b_hi - b_lo
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
        return psi_G

    def _coeffs_gspace_with_short_tail(
        self,
        b_lo: int,
        b_hi: int,
        n_aligned: int,
        k_ids: Sequence[int] | None,
    ) -> jax.Array:
        """File-short path.  Bulk: collective FFI read of ``[b_lo,
        b_lo + n_aligned)``.  Tail: replicated h5py read of
        ``[b_lo + n_aligned, min(b_hi, self.nbands))`` (≤ world-1 bands).
        Zero-pad: ``[min(b_hi, self.nbands), b_hi)`` (no read).  Stitch
        via ``jnp.concatenate`` along the band axis + reshard to the
        canonical ``P(None, ('x','y'), None, None, None, None)`` layout.

        Cost: one extra h5py read (≤ world-1 bands × n_k × ngkmax —
        kilobytes) per band-chunk that crosses the file boundary, plus
        one all-to-all-shape reshard at the concat boundary.  Negligible
        compared to the bulk FFI read."""
        nb_total = b_hi - b_lo
        n_in_file = max(0, int(self.nbands) - int(b_lo))
        n_tail = n_in_file - n_aligned
        n_zero = nb_total - n_in_file
        assert n_tail >= 0 and n_zero >= 0
        assert n_aligned + n_tail + n_zero == nb_total

        if k_ids is None:
            k_ids = np.arange(self.nk_full, dtype=np.int32)
        else:
            k_ids = np.asarray(k_ids, dtype=np.int32)
        n_k = len(k_ids)

        nx, ny, nz = (int(v) for v in self.fft_grid)
        ns = int(self.nspinor)
        rep_shard = NamedSharding(self.mesh, P())
        final_shard = NamedSharding(
            self.mesh, P(None, ('x', 'y'), None, None, None, None))

        parts = []

        # 1. Bulk via FFI (skip if n_aligned == 0).
        if n_aligned > 0:
            parts.append(
                self._coeffs_gspace_aligned(b_lo, b_lo + n_aligned, k_ids))

        # 2. Replicated tail via h5py + on-device unfold + fft_box.
        if n_tail > 0:
            parts.append(self._coeffs_gspace_replicated_tail(
                b_lo + n_aligned, b_lo + n_aligned + n_tail, k_ids))

        # 3. Pure-zero pad past wfn.nbands.
        if n_zero > 0:
            zero_block = jax.device_put(
                jnp.zeros((n_k, n_zero, ns, nx, ny, nz), dtype=jnp.complex128),
                rep_shard)
            parts.append(zero_block)

        full = jnp.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        return jax.lax.with_sharding_constraint(full, final_shard)

    def _coeffs_gspace_replicated_tail(
        self,
        b_lo: int,
        b_hi: int,
        k_ids: np.ndarray,
    ) -> jax.Array:
        """Read ``[b_lo, b_hi)`` (1 ≤ count < world bands) via h5py, then
        run the same unfold + fft-box pipeline as the FFI path but
        replicated across all ranks.  Handles both nosym (ntran == 1,
        identity unfold) and sym (ntran > 1, full TR conjugation + τ
        phase + U_spinor rotation) files.  Output shape
        ``(n_k, b_hi - b_lo, nspinor, nx, ny, nz)`` complex128, sharded
        ``P()`` (replicated)."""
        import h5py
        n_tail = int(b_hi - b_lo)
        n_k = len(k_ids)
        nx, ny, nz = (int(v) for v in self.fft_grid)
        ns = int(self.nspinor)

        # Mirror the bulk path's IBZ-k dedup so the unfold inputs share
        # the same layout as the sharded ``_make_unfold_kernel`` body.
        # For nosym files this just decays to the identity (n_reads =
        # n_k, position_in_reads = identity, U = I, phase = 1, tr_mask
        # all False).
        ibz_per_id = self._ibz_per_full_k[k_ids]
        unique_ibz, ibz_inv = np.unique(ibz_per_id, return_inverse=True)
        file_order = np.argsort(
            self.kpt_starts[unique_ibz], kind="stable").astype(np.int32)
        ibz_file_sorted = unique_ibz[file_order]
        position_in_reads = np.argsort(
            file_order, kind="stable")[ibz_inv].astype(np.int32)
        n_reads = int(len(ibz_file_sorted))

        # Read each unique IBZ k's tail bands via h5py.  All ranks read
        # independently — small (≤ world-1 bands × n_reads × ngkmax,
        # kilobytes) so concurrent file opens don't contend.
        cnk_at_ibz_np = np.zeros(
            (n_tail, ns, n_reads, self.ngkmax, 2), dtype=np.float64)
        with h5py.File(self.path, "r") as f:
            coeffs = f["wfns/coeffs"]
            for ri, ibz in enumerate(ibz_file_sorted):
                ngk = int(self.ngk[ibz])
                start = int(self.kpt_starts[ibz])
                cnk_at_ibz_np[:, :, ri, :ngk, :] = coeffs[
                    b_lo:b_hi, :, start:start + ngk, :]

        cnk_at_ibz = jnp.asarray(cnk_at_ibz_np)

        # Apply unfold IBZ → full BZ.  Same body as the sharded
        # ``_make_unfold_kernel._per_rank``; we just don't wrap it in
        # shard_map because the input is already replicated.
        if self.ntran == 1:
            # Nosym fast path: identity unfold = jnp.take along the
            # k axis.  Skip the U/phase/TR multiplies entirely.
            cnk_at_full = jnp.take(
                cnk_at_ibz, jnp.asarray(position_in_reads), axis=2)
        else:
            k_ids_dev = jax.device_put(jnp.asarray(k_ids), self._rep1d)
            U_k, phase_k, tr_mask_k = self._sym_tables_for_ids(k_ids_dev)
            position_dev = jax.device_put(
                jnp.asarray(position_in_reads), self._rep1d)
            cnk_at_full = _unfold_replicated_body(
                cnk_at_ibz, U_k, phase_k, tr_mask_k, position_dev)
        # cnk_at_full: (n_tail, ns, n_k, ngkmax, 2) f64

        # Re/im → complex, transpose to (n_k, n_tail, ns, ngkmax) for
        # the per-k gather.
        cnk = cnk_at_full[..., 0] + 1j * cnk_at_full[..., 1]
        cnk = jnp.transpose(cnk, (2, 0, 1, 3))
        # Append a zero G-slot so the sentinel index ngkmax gathers zero
        # — same trick as ``make_fft_box_kernel`` uses.
        zero_slot = jnp.zeros((n_k, n_tail, ns, 1), dtype=jnp.complex128)
        cnk_padded = jnp.concatenate([cnk, zero_slot], axis=-1)
        # Shape: (n_k, n_tail, ns, ngkmax+1)

        # Per-k gather via vmap.  ``g_index_dev`` lives on device with
        # shape (n_k_full, nx, ny, nz); slice down to the caller's k_ids.
        g_index_for_ids = jnp.take(
            self._g_index_dev, jnp.asarray(k_ids), axis=0)   # (n_k, nx, ny, nz)

        def _gather_one_k(cnk_k_padded, g_idx_k):
            # cnk_k_padded: (n_tail, ns, ngkmax+1); g_idx_k: (nx, ny, nz)
            # → (n_tail, ns, nx, ny, nz)
            return jnp.take(cnk_k_padded, g_idx_k, axis=2)
        gathered = jax.vmap(_gather_one_k, in_axes=(0, 0))(
            cnk_padded, g_index_for_ids)
        # Shape: (n_k, n_tail, ns, nx, ny, nz) — canonical layout.
        return jax.device_put(
            gathered, NamedSharding(self.mesh, P()))

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


@jax.jit
def _unfold_replicated_body(
    cnk_at_ibz,           # (nb, ns, n_reads, ngkmax, 2) f64 — replicated
    U_per_k,              # (n_k, ns, ns) c128
    phase_per_k,          # (n_k, ngkmax) c128
    tr_mask_per_k,        # (n_k,) bool
    position_in_reads,    # (n_k,) int32 — IBZ slab serving each full-BZ k
):
    """Replicated equivalent of ``_make_unfold_kernel._per_rank``.

    Same math (TR conjugation → τ phase → U_spinor rotation → re/im
    repack) but runs on a fully replicated band axis with no
    ``shard_map`` wrapper.  Used for the small "tail" of the
    pad-past-file path in :meth:`PhdfWfnReader._coeffs_gspace_replicated_tail`.
    Bands are independent across the unfold body, so the replicated
    output is bit-equivalent to running the sharded kernel with the
    same inputs gathered across ranks."""
    cnk = cnk_at_ibz[..., 0] + 1j * cnk_at_ibz[..., 1]
    # (nb, ns, n_reads, ngkmax) → (nb, ns, n_k, ngkmax)
    cnk = jnp.take(cnk, position_in_reads, axis=2)
    cnk = jnp.where(
        tr_mask_per_k[None, None, :, None], jnp.conj(cnk), cnk)
    cnk = cnk * phase_per_k[None, None, :, :]
    cnk = jnp.einsum("kac,bckg->bakg", U_per_k, cnk)
    return jnp.stack([cnk.real, cnk.imag], axis=-1)
