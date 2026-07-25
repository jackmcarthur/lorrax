"""``WfnLoader`` — single entry point for ψ(G) loading.

Replaces the {WFNReader + PhdfWfnReader + SymMaps.get_cnk_fullzone[_batch] +
SymMaps.get_gvecs_kfull + load_wfns.read_Gvecs_to_devices + load_kpoint_fftbox}
mess with one class.  Caller never thinks about backend, symmetry unfold,
padding, or τ-phase.

Returns ψ in **G-flat** layout (per-k, per-band, per-spinor, ngk_max_pad-padded).
Downstream consumers compose with :mod:`common.wfn_transforms` (P3 landing
target) to get FFT-box / r-space / centroid-gather outputs.  This split
keeps the loader cheap (g_flat is ~6-11% of g_box) so band-chunk loops
never have to materialise the FFT box just to access a real-space slice.

Backends
--------
``backend='auto'`` (default) picks the lightest path that works:

- **eager** (host h5py + numpy unfold + ``device_put``): single-process,
  CPU JAX, or small files.
- **phdf5** (collective parallel-HDF5 FFI + on-device unfold): multi-rank
  GPU + 2-D mesh + FFI .so loadable.  Reuses the same union-read +
  unfold kernel that powered the legacy ``PhdfWfnReader.coeffs_gspace``
  path, but stops one step short of the FFT-box scatter so the output
  stays G-flat (the loader's defining layout).

Both backends produce **byte-identical** output for the same ``(bands, k,
sharding, bispinor)`` request — that's the P2 test contract.

Public surface
--------------
* :class:`MfHeader` attributes — same names :class:`WFNReader` exposes
  (``nkpts``, ``nbands``, ``nspinor``, ``kgrid``, ``fft_grid``, ``bvec``,
  ``sym_matrices``, ``translations``, …).  Drop-in source.
* :meth:`load` — one call: ψ array for a (band_range, k) window.
* :meth:`bands` — band-chunked iterator for GW driver loops.
* :meth:`gvecs` — cached G-vector lists per k-set.
* :meth:`ngk_valid` — per-k logical ngk (for callers that care).
* :meth:`get_gvec_nk` — deprecated thin shim for legacy vcoul / qp_wfn.

P-roadmap
---------
- P1 (this commit): eager backend; bit-match legacy WFNReader+SymMaps.
- P2: phdf5 backend; bit-match eager.
- P3: ``common/wfn_transforms.py`` (to_box / to_rbox / to_rmu / to_rchunk).
- P4: migrate consumers; delete SymMaps unfold helpers + load_wfns helpers.
- P5: delete WFNReader + PhdfWfnReader.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path
from typing import Iterator, Literal, Sequence

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .mf_header import bind_mf_attrs, kpt_starts, read_mf_header_from_file


__all__ = ["WfnLoader"]


KSpec = Sequence[int] | Literal["ibz", "full_bz"]


def _build_phdf5_clamped_counts(
    *,
    world: int,
    bands_per_rank: int,
    b_lo_logical: int,
    mnband_file: int,
    n_reads: int,
    ngk_per_ibz_read: Sequence[int],
    ns: int,
) -> np.ndarray:
    """Per-rank ``counts`` table for the kchunk-union phdf5 read,
    clamped to the on-disk ``mnband_file`` band extent.

    The C++ ``read_kchunk_union`` handler computes each rank's
    band-axis file offset as
    ``offset_band = b_lo_logical + rank_coord_band * bands_per_rank``,
    where ``rank_coord_band = coord_x * p_y + coord_y`` for a 2-D
    ``('x','y')`` mesh of shape ``(p_x, p_y)``.  Without clamping,
    the tail rank can read past ``mnband_file`` whenever
    ``(b_hi_logical - b_lo_logical)`` rounded up to ``world *
    bands_per_rank`` extends past the file extent — H5Dread then
    fails with "selection + offset not within extent".

    This helper returns a ``(world * n_reads, 4) int64`` table whose
    rank ``r``-slice ``[r*n_reads:(r+1)*n_reads, :]`` has the band-axis
    count clamped to ``max(0, min(bands_per_rank, mnband_file -
    (b_lo_logical + r * bands_per_rank)))``.  Ranks fully past EOF get
    band_cnt=0 — their pinned-buffer pre-zero (in the C++ worker)
    becomes a zero-filled rank tile, which is the correct semantics for
    band-pad rows.

    Sharded on the leading axis by ``('x','y')`` so each rank's
    shard_map-local view is the ``(n_reads, 4)`` slice for its own
    rank.  Rank-flattening matches the C++: outer loop over ``r``
    in 0..world-1 corresponds to ``coord_x = r // p_y,
    coord_y = r % p_y`` (leftmost-is-slowest in JAX's convention).
    """
    counts = np.zeros((world, n_reads, 4), dtype=np.int64)
    for r in range(int(world)):
        file_off_band = int(b_lo_logical) + r * int(bands_per_rank)
        avail = max(0, int(mnband_file) - file_off_band)
        band_cnt = min(int(bands_per_rank), avail)
        for ki in range(int(n_reads)):
            counts[r, ki, 0] = band_cnt
            counts[r, ki, 1] = int(ns)
            counts[r, ki, 2] = int(ngk_per_ibz_read[ki])
            counts[r, ki, 3] = 2
    return counts.reshape(int(world) * int(n_reads), 4)


class WfnLoader:
    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str | Path,
        *,
        mesh: Mesh | None = None,
        backend: Literal["auto", "eager", "phdf5"] = "auto",
    ) -> None:
        self._path = str(path)
        self._filename = self._path  # legacy WFNReader compat
        self._mesh = mesh

        if backend == "auto":
            backend = self._auto_pick_backend()
        if backend not in ("eager", "phdf5"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "phdf5" and mesh is None:
            raise ValueError(
                "WfnLoader: backend='phdf5' requires a Mesh; pass mesh=...")
        self.backend = backend

        self._file = h5.File(self._path, "r")

        # phdf5 collective context (lazy on first load).  Held here so
        # the file is kept open for the loader's lifetime.
        self._phdf5_ctx: int | None = None
        self._phdf5_static_dev: dict | None = None

        # mf_header surface — same names WFNReader exposes (drop-in compat).
        hdr = read_mf_header_from_file(self._file)
        bind_mf_attrs(self, hdr)
        # Cartesian-→-crystal: same expression legacy WFNReader exposed.
        self.atom_crys = np.einsum(
            'ij,kj->ki', np.linalg.inv(self.avec).T, self.atom_positions)

        # Derived band-fill metadata — same names WFNReader exposed.
        # ``ifmax`` is the 1-based index of the highest occupied band.
        if np.size(self.ifmax) > 0:
            self.nelec = int(np.max(self.ifmax))
        else:
            self.nelec = int(np.sum(self.occs[0, 0] > 0.5))
        _nb = int(self.energies.shape[-1])
        _occ_idx = max(0, min(self.nelec - 1, _nb - 1))
        self.vbm = float(np.max(self.energies[:, :, _occ_idx]))
        if _occ_idx + 1 < _nb:
            self.cbm = float(np.min(self.energies[:, :, _occ_idx + 1]))
            self.efermi = 0.5 * (self.vbm + self.cbm)
        else:
            self.cbm = float(self.vbm)
            self.efermi = float(self.vbm)

        # Eager-backend state.  Only the eager path reads coeffs host-side;
        # keep the dataset HANDLE (no read) and hyperslab the requested
        # ``[b_lo:b_hi, :, start:end, :]`` block per-call in ``_eager_build``.
        # The phdf5 backend reads coeffs collectively via the FFI, so it
        # never touches this — slurping the whole ``(nb, ns, ngktot, 2)`` f64
        # array unconditionally was pure waste + a latent OOM (a second,
        # whole-array read on the exact multi-rank path where memory is
        # tightest).  ``_gvecs_raw`` (ngktot,3) + ``_kpt_starts`` are cheap
        # index metadata both backends use — keep those eager.
        self._coeffs_ds = (
            self._file["wfns/coeffs"] if self.backend == "eager" else None)
        self._gvecs_raw = self._file["wfns/gvecs"][:]     # (ngktot, 3) int
        # kpt_starts = cumulative (exclusive prefix) sum of ngk.
        self._kpt_starts = kpt_starts(self.ngk)

        # Lazy state.
        self._sym = None
        self._gvecs_cache: dict[tuple, np.ndarray] = {}
        self._ngk_valid_cache: dict[tuple, np.ndarray] = {}
        # Device-resident g_index cache.  Sphere/g_index is a function of
        # ``(k_set, fft_grid)`` only — identical across charge + transverse
        # bispinor channels and across V_q tiles.  Caching the device copy
        # here (not in ``psi_G_store._g_index_dev``, which dies with each
        # ``fit_zeta_to_h5`` instance) deduplicates the (nk, nx, ny, nz)
        # int32 buffer across the full GW pipeline.  Pre-fix:
        # ``jax.device_put`` allocated a fresh REPLICATED buffer per
        # ``psi_G_store`` construction → 1 buffer/channel × 4 channels =
        # 4-8 leaked buffers ≈ 1.3 GB/rank wasted (agent_h §3 Finding 3).
        # Post-fix: single shared device buffer per ``(k_cache_key, mesh)``.
        # Keyed by ``(k_cache_key, id(mesh))`` so two ``WfnLoader``s on
        # different meshes (rare) get distinct buffers; same loader+mesh
        # always returns the same ``jax.Array``.
        self._gvecs_dev_cache: dict[tuple, "jax.Array"] = {}

    # ------------------------------------------------------------------
    def close(self) -> None:
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._file = None
        ctx = getattr(self, "_phdf5_ctx", None)
        if ctx is not None:
            try:
                from ffi.phdf5 import close_file
                close_file(ctx)
            except Exception:
                pass
            self._phdf5_ctx = None

    def __enter__(self) -> "WfnLoader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    def _auto_pick_backend(self) -> str:
        """Pick the lightest backend that works.

        Rules:
          * Mesh missing → eager (single-device / laptop / pytest).
          * Single-process JAX → eager (no benefit from collective FFI).
          * Multi-process JAX + mesh provided + CUDA FFI .so loadable →
            phdf5.  The probe must pin the CUDA library: the phdf5 FFI
            lives only there, and a bare get_lib() on a CPU backend would
            "succeed" by loading the slate-only host library and then
            crash at the first open_file.
          * Anything else → eager.
        """
        if self._mesh is None:
            return "eager"
        try:
            if int(jax.process_count()) <= 1:
                return "eager"
        except Exception:
            return "eager"
        try:
            from ffi.common.ffi_loader import get_lib
            get_lib("CUDA")
            from ffi.phdf5 import open_file as _of  # noqa: F401
            return "phdf5"
        except Exception:
            return "eager"

    # ------------------------------------------------------------------
    # k-set resolution
    # ------------------------------------------------------------------
    def _ensure_sym(self):
        from common.symmetry_maps import SymMaps
        if self._sym is None:
            self._sym = SymMaps(self._sym_wfn_stub())
        return self._sym

    def _sym_wfn_stub(self):
        """Minimal namespace SymMaps's __init__ reads; lets us construct
        SymMaps without holding a circular ref to the loader."""
        return types.SimpleNamespace(
            ntran=int(self.ntran),
            sym_matrices=self.sym_matrices,
            translations=self.translations,
            kpoints=self.kpoints,
            kgrid=self.kgrid,
            shift=self.shift,
            nkpts=int(self.nkpts),
            bvec=self.bvec,
            avec=self.avec,
            atom_types=self.atom_types,
            atom_positions=self.atom_positions,
            atom_crys=np.einsum(
                "ij,kj->ki",
                np.linalg.inv(self.avec).T, self.atom_positions),
            fft_grid=self.fft_grid,
        )

    def _resolve_k(self, k: KSpec) -> tuple[np.ndarray, bool]:
        """Resolve a k-spec to (k_idxs, unfold).

        - ``'ibz'``: raw IBZ — returns np.arange(nkpts), unfold=False.
        - ``'full_bz'``: full-BZ unfold — returns np.arange(sym.nk_tot),
          unfold=True.
        - explicit list: interpreted as **full-BZ indices**, unfold=True.
        """
        if isinstance(k, str):
            if k == "ibz":
                return np.arange(self.nkpts, dtype=np.int32), False
            if k == "full_bz":
                sym = self._ensure_sym()
                return np.arange(int(sym.nk_tot), dtype=np.int32), True
            raise ValueError(f"unknown k-spec {k!r}")
        return np.asarray(k, dtype=np.int32), True

    def _k_cache_key(self, k: KSpec) -> tuple:
        if isinstance(k, str):
            return (k,)
        return ("list", tuple(int(v) for v in k))

    # ------------------------------------------------------------------
    # G-vector and ngk_valid accessors
    # ------------------------------------------------------------------
    def gvecs(self, *, k: KSpec = "full_bz") -> np.ndarray:
        """Return ``(n_k, ngkmax, 3)`` int32 — G-vector list per k, zero-padded
        beyond logical ngk.  Cached per k-set."""
        key = self._k_cache_key(k)
        if key in self._gvecs_cache:
            return self._gvecs_cache[key]

        k_idxs, unfold = self._resolve_k(k)
        out = np.zeros((len(k_idxs), int(self.ngkmax), 3), dtype=np.int32)

        if not unfold:
            for j, ik in enumerate(k_idxs):
                start = int(self._kpt_starts[int(ik)])
                end = start + int(self.ngk[int(ik)])
                out[j, : end - start] = self._gvecs_raw[start:end]
        else:
            sym = self._ensure_sym()
            for j, nk in enumerate(k_idxs):
                # Inlines the former ``sym.get_gvecs_kfull`` body — see
                # the unfold derivation in ``test_wfn_loader_eager.py``
                # ``test_gvecs_full_bz_matches_legacy``.  Each full-BZ k
                # rotates its IBZ-source G-list by ``sym_krep`` and
                # subtracts the BGW umklapp ``kg0``.
                nk_int = int(nk)
                sym_idx = int(sym.sym_idx_k[nk_int])
                kbar = int(sym.irr_idx_k[nk_int])
                sym_krep = np.asarray(
                    sym.sym_mats_k[sym_idx], dtype=np.int32)
                start = int(self._kpt_starts[kbar])
                end = start + int(self.ngk[kbar])
                k_gvecs = self._gvecs_raw[start:end]
                Gkk = sym.get_umklapp_vector(
                    self, nk_int, sym_idx, kbar, sym_krep)
                g_rot = np.einsum('ij,kj->ki', sym_krep, k_gvecs) - Gkk
                out[j, : g_rot.shape[0]] = g_rot
        self._gvecs_cache[key] = out
        return out

    def ngk_valid(self, *, k: KSpec = "full_bz") -> np.ndarray:
        """Per-k logical ngk (without pad).  Host numpy int32."""
        key = self._k_cache_key(k)
        if key in self._ngk_valid_cache:
            return self._ngk_valid_cache[key]
        k_idxs, unfold = self._resolve_k(k)
        if not unfold:
            out = np.asarray(self.ngk[k_idxs], dtype=np.int32)
        else:
            sym = self._ensure_sym()
            ibz_per_full = np.asarray(sym.irr_idx_k, dtype=np.int32)
            out = np.asarray(self.ngk[ibz_per_full[k_idxs]], dtype=np.int32)
        self._ngk_valid_cache[key] = out
        return out

    def get_gvec_nk(self, ik: int) -> np.ndarray:
        """Deprecated shim for legacy vcoul.py / qp_wfn.py callers.

        Returns the (ngk[ik], 3) IBZ G-list for a single k.  New code
        should use ``loader.gvecs(k='ibz')[ik, :loader.ngk_valid(k='ibz')[ik]]``
        — but vcoul reads one k at a time, so the shim stays for one
        release."""
        start = int(self._kpt_starts[int(ik)])
        end = start + int(self.ngk[int(ik)])
        return self._gvecs_raw[start:end]

    # ------------------------------------------------------------------
    # G-flat → FFT-box index (zero-sentinel gather table)
    # ------------------------------------------------------------------
    def box_index(self, *, k: KSpec = "full_bz") -> np.ndarray:
        """Return ``(n_k, nx, ny, nz)`` int32 — for each FFT-box cell, the
        index along the ψ(G) axis to gather from.  Empty cells take the
        sentinel value ``ngkmax``; downstream transforms append a zero
        slot at that position so empty cells gather zero (see
        :func:`common.wfn_transforms.to_box`).

        Cached per (k-set, ``self.fft_grid``).  Reuses
        :func:`common.gvec_fft_box.build_g_index_for_fft_box` so the
        algorithm lives in one place.
        """
        cache_key = ("box_index", *self._k_cache_key(k))
        if cache_key in self._gvecs_cache:
            return self._gvecs_cache[cache_key]

        from common.gvec_fft_box import build_g_index_for_fft_box

        gvecs = self.gvecs(k=k)                                # (n_k, ngkmax, 3)
        ngk_v = self.ngk_valid(k=k)                            # (n_k,)
        # Strip pad rows back to per-k logical extent so the index
        # builder doesn't see zero-padded gvecs (which would map to
        # (0,0,0) and clobber the real Γ slot).
        gvecs_per_k = [
            gvecs[j, : int(ngk_v[j])] for j in range(int(gvecs.shape[0]))
        ]
        g_index = build_g_index_for_fft_box(
            gvecs_per_k, tuple(int(s) for s in self.fft_grid),
            int(self.ngkmax))
        self._gvecs_cache[cache_key] = g_index
        return g_index

    def box_index_dev(
        self,
        *,
        k: KSpec = "full_bz",
        mesh: "Mesh | None" = None,
        sharding: "NamedSharding | PartitionSpec | None" = None,
    ) -> "jax.Array":
        """Return ``box_index(k=k)`` as a REPLICATED ``jax.Array`` on
        ``mesh`` — but only do the ``device_put`` once per ``(k, mesh)``.

        Fixes the sphere-idx replicated leak (agent_h §3 Finding 3):
        every fresh ``psi_G_store._populate_from_loader`` used to call
        ``jax.device_put(loader.box_index("full_bz"), ...)`` which
        allocated a NEW REPLICATED ``(nk, nx, ny, nz) int32`` buffer
        per channel (0.16 GB/rank each).  After 4 bispinor channels +
        a couple of one-off calls the leak grew to 8 buffers ≈ 1.3
        GB/rank.  Caching the ``jax.Array`` here means **every caller
        for the same (k, mesh) gets the same device buffer**; the
        XLA allocator references it once and Python GC retains it
        through the loader's lifetime.

        Parameters
        ----------
        k : KSpec
            Same as :meth:`box_index` (defaults to ``"full_bz"`` — the
            only path that's ever been observed to leak in production).
        mesh : Mesh, optional
            Device mesh to replicate over.  If absent, falls back to
            ``self._mesh`` (the loader's own mesh, set at construction).
            Raises if both are absent — there's no sensible single-device
            fallback for a sharding-aware accessor.
        sharding : NamedSharding | PartitionSpec, optional
            For callers that need a non-default replicated layout.
            Default ``None`` ⇒ ``NamedSharding(mesh, P(None, None, None, None))``
            (the only layout used in production; centralised here so
            future shape changes (e.g. 5-D ψ for bispinor) update one
            site instead of three).
        """
        from jax.sharding import NamedSharding, PartitionSpec as P
        mesh = mesh if mesh is not None else self._mesh
        if mesh is None:
            raise ValueError(
                "box_index_dev: pass `mesh=` or construct the WfnLoader "
                "with a mesh; cannot device_put without one.")
        # Build the cache key.  ``id(mesh)`` is fine because the mesh
        # outlives the loader in every production driver (the mesh is
        # built once at top-of-main and threaded through every kernel).
        cache_key = ("box_index_dev", *self._k_cache_key(k), id(mesh))
        if cache_key in self._gvecs_dev_cache:
            return self._gvecs_dev_cache[cache_key]
        # Resolve the requested sharding (default = replicated 4-axis).
        if sharding is None:
            sharding = NamedSharding(mesh, P(None, None, None, None))
        elif isinstance(sharding, P):
            sharding = NamedSharding(mesh, sharding)
        # Pass numpy directly to ``device_put`` — the per-process-local
        # placement path avoids the all-reduce that ``jnp.asarray`` ->
        # device_put(replicated) would otherwise trigger (same comment
        # as in ``psi_G_store._populate_from_loader``).
        g_idx_np = self.box_index(k=k)
        dev = jax.device_put(g_idx_np, sharding)
        self._gvecs_dev_cache[cache_key] = dev
        return dev

    # ------------------------------------------------------------------
    # Sharding + padding
    # ------------------------------------------------------------------
    def _default_sharding(
        self,
        sharding: PartitionSpec | None,
        *,
        n_k: int,
    ) -> tuple[NamedSharding | None, int]:
        """Return ``(named_sharding, p_band)`` for the (k, nb, ns, ngk) layout.

        - If caller passed ``sharding`` explicitly, use it; ``p_band`` is
          the product of mesh axes on the band dim (for padding).
        - Else default: if mesh available and multi-device, shard band
          on ``('x','y')`` (the GW production layout); if not, replicate.
        """
        if sharding is None:
            if self._mesh is None or len(self._mesh.devices.flat) <= 1:
                return None, 1                   # replicated
            sharding = P(None, ("x", "y"), None, None)

        if self._mesh is None:
            return None, 1                       # replicated even with explicit spec
        named = NamedSharding(self._mesh, sharding)
        # Compute band-axis pad factor from the spec's band-dim entry.
        spec = list(sharding)
        band_axes = spec[1] if len(spec) > 1 else None
        if band_axes is None:
            p_band = 1
        elif isinstance(band_axes, str):
            p_band = int(self._mesh.shape[band_axes])
        else:
            p_band = 1
            for a in band_axes:
                p_band *= int(self._mesh.shape[a])
        return named, int(p_band)

    # ------------------------------------------------------------------
    # The main load
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # phdf5 backend — collective FFI read + on-device unfold
    # ------------------------------------------------------------------
    def _ensure_phdf5_static(self) -> dict:
        """Lazy build of the per-full-BZ-k symmetry tables on device.

        Builds once per ``WfnLoader`` instance; subsequent ``load()``
        calls reuse the staged arrays.  Mirrors
        ``PhdfWfnReader._build_symmetry_tables`` +
        ``_compute_phases_all_full_k`` + ``_device_put_static_tables``
        but without the FFT-box machinery (which lives downstream in
        ``wfn_transforms``).
        """
        if self._phdf5_static_dev is not None:
            return self._phdf5_static_dev

        from ffi.phdf5 import open_file

        sym = self._ensure_sym()
        nk_full = int(sym.nk_tot)
        ngkmax = int(self.ngkmax)
        ibz_per_full = np.asarray(sym.irr_idx_k, dtype=np.int32)[:nk_full]
        sym_idx_per_full = np.asarray(sym.sym_idx_k, dtype=np.int32)[:nk_full]
        n_tran = int(sym.sym_matrices.shape[0])
        tr_mask = (sym_idx_per_full >= n_tran).astype(np.bool_)

        # Per-k spinor rotation matrix, single-sourced via the ψ-unfold
        # spinor rule (spatial rows → ``sym.U_spinor[s]``; TRS rows →
        # ``iσ_y · conj(sym.U_spinor[s − ntran])``, the T = iσ_y K rule).
        # See ``common.symmetry_maps.{unfold_psi,trs_augment_U}`` and
        # ``reports/trs_sym_audit_2026-05-14`` Sites #5–#7.  ``sym.U_spinor``
        # is length ``ntran`` (PR3); the TRS half is built inside the helper.
        from common.symmetry_maps import trs_augment_U
        U_per = trs_augment_U(
            sym.U_spinor, sym_idx_per_full, n_tran)                      # (nk_full, 2, 2)

        # τ-phase per full-BZ k on the ibz-source ngkmax-padded G-list.
        # Use the SAME formula for spatial and TRS rows:
        #   phase = exp(-i (sym_mats_k[s] · G_kbar) · τ_{s_spatial})
        # For TRS rows ``sym_mats_k[s] = -S_spatial`` so the formula yields
        # ``exp(+i (S_spatial · G_kbar) · τ)``, which is ``conj`` of the
        # spatial-row phase. Combined with the downstream kernel's
        # ``where(tr_mask, conj(cnk), cnk)`` step, the per-element TRS rule
        # ``ψ_full = (iσ_y · conj(U)) · conj(ψ_kbar) · conj(phase_spatial)``
        # is reproduced. Pre-PR3 the TRS rows were set to 1 (skipped the
        # phase entirely) — that bug fired on non-symmorphic non-inversion
        # bispinor systems.
        from common.symmetry_maps import tau_phase_row
        phase = np.ones((nk_full, ngkmax), dtype=np.complex128)
        for nk in range(nk_full):
            s = int(sym_idx_per_full[nk])
            s_spatial = s - n_tran if s >= n_tran else s
            ibz = int(ibz_per_full[nk])
            ngk_k = int(self.ngk[ibz])
            start = int(self._kpt_starts[ibz])
            g_bar = self._gvecs_raw[start:start + ngk_k]
            ph = tau_phase_row(
                sym.sym_mats_k[s], sym.translations[s_spatial], g_bar)
            if ph is not None:
                phase[nk, :ngk_k] = ph

        # Open the phdf5 collective context lazily.
        if self._phdf5_ctx is None:
            self._phdf5_ctx = open_file(self._path, mesh=self._mesh, mode="r")

        # Device-stage the static tables once.  Sharding choice mirrors
        # ``PhdfWfnReader._device_put_static_tables`` — all replicated
        # since they're per-full-BZ-k metadata, not per-band data.
        rep0 = NamedSharding(self._mesh, P())
        rep1 = NamedSharding(self._mesh, P(None))
        rep2 = NamedSharding(self._mesh, P(None, None))
        rep3 = NamedSharding(self._mesh, P(None, None, None))
        # Pass numpy directly to ``device_put``; ``jnp.asarray`` wrapped
        # in this position used to single-device-stage the host array
        # first, forcing device_put → replicated to fire an all-reduce
        # broadcast.  See psi_G_store: same fix.
        self._phdf5_static_dev = {
            "ibz_per_full": jax.device_put(ibz_per_full, rep1),
            "sym_idx_per_full": jax.device_put(sym_idx_per_full, rep1),
            "tr_mask_per_full": jax.device_put(tr_mask, rep1),
            "U_per_full": jax.device_put(U_per, rep3),
            "phase_per_full": jax.device_put(phase, rep2),
            "n_tran": n_tran,
            "nk_full": nk_full,
        }
        return self._phdf5_static_dev

    def _phdf5_build(
        self,
        *,
        b_lo: int,
        b_hi: int,
        k_idxs: np.ndarray,
        unfold: bool,
        nb_padded: int,
        out_sharding: NamedSharding,
    ) -> jax.Array:
        """Collective FFI read + on-device unfold → G-flat ψ.

        Output: ``(n_k, nb_padded, nspinor, ngkmax)`` c128 sharded as
        ``out_sharding`` (typically ``P(None, ('x','y'), None, None)``).
        """
        from ffi.phdf5.read import read_kchunk_union_sharded

        static = self._ensure_phdf5_static()
        ctx = self._phdf5_ctx
        assert ctx is not None
        p_x = int(self._mesh.shape["x"])
        p_y = int(self._mesh.shape["y"])
        world = p_x * p_y
        ns = int(self.nspinor)
        ngkmax = int(self.ngkmax)
        ngktot = int(np.sum(self.ngk))
        nb = nb_padded
        if nb % world:
            raise ValueError(
                f"_phdf5_build: nb_padded={nb} not divisible by world={world}; "
                "this is a loader bug — _default_sharding should have padded.")
        bands_per_rank = nb // world
        b_lo_logical = int(b_lo)
        b_hi_logical = int(b_hi)

        # Determine the union of IBZ k-points to read, and the position
        # of each requested k in that union.
        if unfold:
            ibz_per_full = np.asarray(static["ibz_per_full"])
            ibz_per_k = ibz_per_full[np.asarray(k_idxs, dtype=np.int32)]
        else:
            # k_idxs are already IBZ indices.
            ibz_per_k = np.asarray(k_idxs, dtype=np.int32)

        ibz_unique_sorted = np.unique(ibz_per_k).astype(np.int32)
        n_reads = int(ibz_unique_sorted.size)
        # ``position_in_reads[j]`` = where ibz_per_k[j] sits in the
        # ascending-sorted union (the dim along which read_kchunk_union
        # returns its concatenated output).
        position_in_reads = np.searchsorted(
            ibz_unique_sorted, ibz_per_k).astype(np.int32)
        n_k = int(len(k_idxs))

        # Hyperslab offsets/counts for the union read.
        # Dataset layout: (mnband, nspinor, ngktot, 2) f64; kchunk_axis=2.
        offsets = np.stack([
            [b_lo_logical, 0, int(self._kpt_starts[ibz]), 0]
            for ibz in ibz_unique_sorted
        ], axis=0).astype(np.int64)
        # Per-rank counts: the C++ adds ``rank_coord_band * bands_per_rank``
        # to ``offsets[:, 0]`` to get each rank's band-axis file offset.
        # When ``mnband`` is not divisible by the global pad
        # (``world * (b_hi_logical-b_lo_logical-mnband-tail)``), the
        # tail-padded ranks would otherwise read past the on-disk band
        # extent.  Build a per-rank ``counts`` table that clamps the
        # band-axis count so each rank's [offset, offset+count) stays
        # inside the file's [0, mnband) extent.  Ranks fully past the
        # extent get count=0 on the band axis → no H5Dread bytes
        # contributed; the C++ pre-zeros the pinned buffer so the
        # rank's tile reads as exactly zero.
        mnband_file = int(self.nbands)
        ngk_per_ibz_read = tuple(int(self.ngk[ibz]) for ibz in ibz_unique_sorted)
        counts_global = _build_phdf5_clamped_counts(
            world=world,
            bands_per_rank=bands_per_rank,
            b_lo_logical=b_lo_logical,
            mnband_file=mnband_file,
            n_reads=n_reads,
            ngk_per_ibz_read=ngk_per_ibz_read,
            ns=ns,
        )

        # The (pure) per-rank band-axis cap doubles as a precondition
        # for the pad-past-file case: if ``b_hi_logical > mnband_file``,
        # the per-rank clamp above produces band_cnt < bands_per_rank
        # for the tail-rank(s); the C++ honours that (count is per-rank
        # and ≤ per_rank_max[0]=bands_per_rank).  This makes the prior
        # NotImplementedError-on-pad-past-file branch obsolete; the
        # phdf5 backend now matches the eager backend's zero-fill
        # behaviour on past-file pads.  We leave a sanity check for the
        # extreme case where ``b_lo`` itself starts past EOF
        # (nonsensical request that ``WfnLoader.load`` rejects upstream
        # at lines 678-681, but defended here too).
        if b_lo_logical >= mnband_file:
            raise ValueError(
                f"_phdf5_build: b_lo={b_lo_logical} >= mnband={mnband_file}; "
                "entire band window past file extent")

        rep1 = NamedSharding(self._mesh, P(None))
        rep2 = NamedSharding(self._mesh, P(None, None))
        counts_sharding = NamedSharding(self._mesh, P(("x", "y"), None))
        # Numpy → replicated/sharded device_put; bare numpy skips the
        # single-device staging that triggers an all-reduce broadcast.
        offsets_dev = jax.device_put(offsets, rep2)
        counts_dev = jax.device_put(counts_global, counts_sharding)
        position_in_reads_dev = jax.device_put(position_in_reads, rep1)

        reader = read_kchunk_union_sharded(
            ctx, "wfns/coeffs",
            n_kchunk=n_reads,
            kchunk_axis=2,
            file_global_shape=(int(self.nbands), ns, ngktot, 2),
            per_rank_file_shape=(bands_per_rank, ns, ngkmax, 2),
            dtype=np.float64,
            mesh=self._mesh,
            file_partition_spec=P(("x", "y"), None, None, None),
            count_partition_spec=P(("x", "y"), None),
        )
        cnk_at_ibz = reader(offsets_dev, counts_dev)
        # cnk_at_ibz layout: per-rank
        # (bands_per_rank, ns, n_reads, ngkmax, 2) f64 sharded
        # P(('x','y'), None, None, None, None).

        # On-device unfold + transpose to WfnLoader's G-flat layout.
        unfold_jit = _phdf5_unfold_kernel(
            self._mesh, n_reads=n_reads, n_k=n_k, bands_per_rank=bands_per_rank,
            nspinor=ns, ngkmax=ngkmax, unfold=unfold)

        if unfold:
            U_k = jnp.take(static["U_per_full"],
                           jnp.asarray(k_idxs, dtype=jnp.int32), axis=0)
            phase_k = jnp.take(static["phase_per_full"],
                                jnp.asarray(k_idxs, dtype=jnp.int32), axis=0)
            tr_mask_k = jnp.take(static["tr_mask_per_full"],
                                  jnp.asarray(k_idxs, dtype=jnp.int32), axis=0)
            psi = unfold_jit(cnk_at_ibz, U_k, phase_k, tr_mask_k,
                              position_in_reads_dev)
        else:
            psi = unfold_jit(cnk_at_ibz, position_in_reads_dev)

        # psi shape after the kernel: (n_k, nb_padded, ns, ngkmax) c128
        # with band-axis sharding propagated from the read.
        return jax.lax.with_sharding_constraint(psi, out_sharding)

    # ------------------------------------------------------------------
    def load(
        self,
        *,
        bands: tuple[int, int],
        k: KSpec = "full_bz",
        sharding: PartitionSpec | None = None,
        bispinor: bool = False,
    ) -> jax.Array:
        """ψ(G) for a (band_range, k-set) window.

        Returns ``(n_k, nb_padded, nspinor_out, ngkmax)`` complex128.

        Padding contract:
        * Band axis pad rows are zero-filled.
        * G axis pad rows are zero-filled; the matching ``gvecs(k=...)``
          rows beyond ``ngk_valid(k=...)`` are zero (used as no-op
          scatter indices).
        * For ``k='full_bz'`` (or explicit list): symmetry unfold +
          τ-phase + TR conjugation applied internally.
        * For ``k='ibz'``: raw WFN-file IBZ slab; no unfold.

        ``bispinor=True`` lifts the small spinor components via
        ``(α/2) σ·(k+G) ψ_L``.  ``nspinor_out`` is then 4; else 2 (or
        the file's ``nspinor``).  Requires the WFN file to have
        ``nspinor == 2`` (BGW Pauli convention); ``ValueError``
        otherwise.
        """
        if bispinor and int(self.nspinor) != 2:
            raise ValueError(
                f"WfnLoader.load(bispinor=True) requires a 2-spinor WFN; "
                f"file has nspinor={int(self.nspinor)}.")

        b_lo, b_hi = int(bands[0]), int(bands[1])
        nb_logical = b_hi - b_lo
        if nb_logical <= 0:
            raise ValueError(f"empty band range: {bands}")
        if b_lo < 0 or b_hi > int(self.nbands):
            raise ValueError(
                f"band range {bands} out of [0, {self.nbands}); use "
                f"bands_pad_to-style external padding for over-file requests")

        k_idxs, unfold = self._resolve_k(k)
        named_sharding, p_band = self._default_sharding(
            sharding, n_k=len(k_idxs))
        from runtime.padding import round_up
        nb_padded = round_up(nb_logical, p_band)

        # Backends produce 2-spinor ψ in the canonical layout
        # ``(n_k, nb_padded, nspinor, ngkmax)`` c128.  The optional
        # bispinor lift below promotes the spinor axis 2 → 4 in one
        # k-vectorised pass (eager: numpy → jnp; phdf5: stays on
        # device).  Eager output is host-staged at the end via
        # ``device_put``; the bispinor-True case routes through
        # ``jnp`` for the lift but follows the same staging path.
        if self.backend == "phdf5":
            # phdf5 path requires a NamedSharding to express the
            # collective output layout.  Caller passed sharding=None on
            # a phdf5 loader → replicate output (rare in production but
            # used by bit-equality tests).  Spinor axis size is 4 when
            # bispinor=True.
            ns_out = 4 if bispinor else int(self.nspinor)
            if named_sharding is None:
                named_sharding = NamedSharding(
                    self._mesh, P(*([None] * 4)))
            elif bispinor:
                # Rebuild the sharding for the post-lift shape — only
                # the spinor axis count changes; partition spec is the
                # same string set.
                pass  # NamedSharding doesn't care about exact dim sizes
            psi = self._phdf5_build(
                b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
                nb_padded=nb_padded, out_sharding=named_sharding)
            if bispinor:
                psi = self._apply_bispinor_lift(
                    psi, k=k, k_idxs=k_idxs, unfold=unfold,
                    sharding=named_sharding)
            return psi

        if bispinor:
            # Bispinor lift path (rarer): full build.  Process-local bispinor
            # is future work (the lift appends per-band small components).
            psi_np = self._eager_build(
                b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
                nb_padded=nb_padded)
            psi_j = jnp.asarray(psi_np)
            psi_j = self._apply_bispinor_lift(
                psi_j, k=k, k_idxs=k_idxs, unfold=unfold, sharding=None)
            if named_sharding is None:
                return psi_j
            return jax.device_put(psi_j, named_sharding)

        if named_sharding is not None and int(jax.process_count()) > 1:
            # (§5b) Build ONLY this process's band shard and assemble via the
            # slab-io process-local idiom -- no rank materialises the full
            # (n_k, nb_padded, ns, ngkmax) host array, which grows with
            # world_size and OOMs past ~16 nodes.  Byte-identical to the full
            # build + device_put below.
            return self._eager_build_process_local(
                b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
                nb_padded=nb_padded, named_sharding=named_sharding)

        # Single-process / replicated: the full host build is fine (no OOM).
        psi_np = self._eager_build(
            b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
            nb_padded=nb_padded)
        if named_sharding is None:
            return jnp.asarray(psi_np)
        return jax.device_put(psi_np, named_sharding)

    def _eager_build_process_local(
        self,
        *,
        b_lo: int,
        b_hi: int,
        k_idxs: np.ndarray,
        unfold: bool,
        nb_padded: int,
        named_sharding: NamedSharding,
    ) -> jax.Array:
        """(§5b) Process-local eager load: each rank builds only its band shard.

        Reuses the slab-io helpers verbatim -- ``_local_shard_and_global_offset``
        to learn THIS rank's band block from a cheap sharded zero proto, and
        ``jax.make_array_from_single_device_arrays`` to assemble -- so no rank
        allocates the full (n_k, nb_padded, ns, ngkmax) host array.  The only
        WFN-specific part is the symmetry unfold, which stays in ``_eager_build``.

        Assumes the BAND axis (1) is the only sharded axis (the centroid-load
        spec ``P(None, ('x','y'), None, None)``); asserts the rest are
        replicated so a different spec fails loud rather than silently wrong.
        """
        from ._slab_io_mpi_host import _local_shard_and_global_offset
        ns = int(self.nspinor)
        ngkmax = int(self.ngkmax)
        n_k = int(len(k_idxs))
        nb_logical = int(b_hi) - int(b_lo)
        global_shape = (n_k, int(nb_padded), ns, ngkmax)

        # Cheap directly-sharded zero proto (each device makes its own zero
        # shard -- no full host/device allocation), then ask JAX which slab
        # this rank owns.  Same trick as ``_slab_io_mpi_host.read_slab``.
        proto = jax.jit(
            lambda: jnp.zeros(global_shape, dtype=jnp.complex128),
            out_shardings=named_sharding)()
        local_zero, offset = _local_shard_and_global_offset(proto)
        local_shape = tuple(int(x) for x in local_zero.shape)
        del proto, local_zero
        if (offset[0], offset[2], offset[3]) != (0, 0, 0) or \
                local_shape[0] != n_k or local_shape[2] != ns or \
                local_shape[3] != ngkmax:
            raise ValueError(
                "process-local eager load supports only band-axis sharding "
                f"(P(None, band, None, None)); got offset={offset} "
                f"local_shape={local_shape} for global {global_shape}.")
        local_nb = local_shape[1]
        band_off = int(offset[1])

        # Real bands in this rank's block map to its FRONT [0:n_real); the rest
        # (pad bands, or blocks entirely past nb_logical) stay zero.
        real_hi = min(band_off + local_nb, nb_logical)
        n_real = max(0, real_hi - band_off)
        if n_real > 0:
            local_np = self._eager_build(
                b_lo=int(b_lo) + band_off, b_hi=int(b_lo) + band_off + n_real,
                k_idxs=k_idxs, unfold=unfold, nb_padded=local_nb)
        else:
            local_np = np.zeros(local_shape, dtype=np.complex128)

        local_arr = jax.device_put(local_np, jax.local_devices()[0])
        return jax.make_array_from_single_device_arrays(
            global_shape, named_sharding, [local_arr])

    # ------------------------------------------------------------------
    # Iterator: band chunks
    # ------------------------------------------------------------------
    def bands(
        self,
        b_lo: int,
        b_hi: int,
        *,
        chunk: int,
        k: KSpec = "full_bz",
        sharding: PartitionSpec | None = None,
        bispinor: bool = False,
    ) -> Iterator[tuple[tuple[int, int], jax.Array]]:
        """Yield ``((bc_lo, bc_hi), psi)`` for a chunked sweep over bands."""
        if chunk <= 0:
            raise ValueError(f"chunk must be positive, got {chunk}")
        for bc_lo in range(int(b_lo), int(b_hi), int(chunk)):
            bc_hi = min(bc_lo + int(chunk), int(b_hi))
            yield (bc_lo, bc_hi), self.load(
                bands=(bc_lo, bc_hi), k=k, sharding=sharding,
                bispinor=bispinor)

    # ------------------------------------------------------------------
    # Bispinor lift (G-flat)
    # ------------------------------------------------------------------
    def _apply_bispinor_lift(
        self,
        psi_2: jax.Array,
        *,
        k: KSpec,
        k_idxs: np.ndarray,
        unfold: bool,
        sharding: NamedSharding | None,
    ) -> jax.Array:
        """ψ (2-spinor) → ψ (4-spinor) by appending the small components.

        Computes the lower-2 spinor components via
        ``(α/2) σ·(k+G) ψ_L`` on G-flat directly — no FFT box.  Sharding
        propagates through (the lift is k-vectorised + band-broadcast;
        no cross-rank op).  Matches the legacy
        :func:`common.bispinor_init.get_small_psi_component` byte-for-byte
        but is vectorised across k.

        Pad rows of ψ are zero → small components of pad rows are also
        zero (clean propagation, no per-k mask needed).

        ``unfold`` tells us which kvec table to use: raw IBZ
        (``wfn.kpoints``, k_idxs are IBZ indices) vs full-BZ
        (``sym.unfolded_kpts``, k_idxs are full-BZ indices).
        """
        gvecs = np.asarray(self.gvecs(k=k))                  # (n_k, ngkmax, 3) int
        if unfold:
            sym = self._ensure_sym()
            kvecs_np = np.asarray(
                sym.unfolded_kpts, dtype=np.float64)[
                    np.asarray(k_idxs, dtype=np.int32)]
        else:
            kvecs_np = np.asarray(
                self.kpoints, dtype=np.float64)[
                    np.asarray(k_idxs, dtype=np.int32)]
        bvec = np.asarray(self.bvec, dtype=np.float64)
        return _bispinor_lift_kernel(
            psi_2,
            jnp.asarray(gvecs, dtype=jnp.float64),
            jnp.asarray(kvecs_np),
            jnp.asarray(bvec),
            sharding=sharding,
        )

    # ------------------------------------------------------------------
    # Eager unfold core
    # ------------------------------------------------------------------
    def _eager_build(
        self,
        *,
        b_lo: int,
        b_hi: int,
        k_idxs: np.ndarray,
        unfold: bool,
        nb_padded: int,
    ) -> np.ndarray:
        """Compose the (n_k, nb_padded, nspinor, ngkmax) host slab.

        Algorithm (matches SymMaps.get_cnk_fullzone_batch +
        WFNReader.get_cnk_batch byte-for-byte under the same inputs):

          1. For each requested k, look up (sym_idx, kbar_idx, sym_krep).
          2. Read raw IBZ band-block from ``self._coeffs_raw`` at
             ``kpt_starts[kbar]:..+ngk[kbar]``; convert (re, im) → c128.
          3. If TRS (sym_idx >= ntran): conjugate.
             Else apply ``exp(-i (S·G_bar)·τ)`` per-G phase.
          4. Apply U_spinor[sym_idx] rotation.
          5. Place into output at ``[j, :nb_logical, :, :ngk]``; rest zero.
        """
        nb_logical = b_hi - b_lo
        ns = int(self.nspinor)
        ngkmax = int(self.ngkmax)
        out = np.zeros((len(k_idxs), nb_padded, ns, ngkmax),
                       dtype=np.complex128)

        if not unfold:
            # Raw IBZ — bypass sym entirely.
            for j, ik in enumerate(k_idxs):
                ibz = int(ik)
                start = int(self._kpt_starts[ibz])
                end = start + int(self.ngk[ibz])
                raw = self._coeffs_ds[b_lo:b_hi, :, start:end, :]
                out[j, :nb_logical, :, : end - start] = (
                    raw[..., 0] + 1j * raw[..., 1])
            return out

        # Full-BZ unfold path.
        from common.symmetry_maps import unfold_psi
        sym = self._ensure_sym()
        ntran = int(sym.sym_matrices.shape[0])
        for j, nk in enumerate(k_idxs):
            nk_int = int(nk)
            sym_idx = int(sym.sym_idx_k[nk_int])
            kbar = int(sym.irr_idx_k[nk_int])
            ngk_k = int(self.ngk[kbar])

            start = int(self._kpt_starts[kbar])
            end = start + ngk_k
            raw = self._coeffs_ds[b_lo:b_hi, :, start:end, :]
            cnk = raw[..., 0] + 1j * raw[..., 1]                    # (nb, ns, ngk_k)
            g_bar = self._gvecs_raw[start:end]                      # (ngk_k, 3)
            cnk = unfold_psi(
                cnk,
                sym_idx=sym_idx,
                g_kbar=g_bar,
                sym_mats_k=sym.sym_mats_k,
                translations=self.translations,
                U_spinor_spatial=sym.U_spinor,
            )
            out[j, :nb_logical, :, :ngk_k] = cnk
        return out


# ---------------------------------------------------------------------------
# Bispinor small-component lift kernel
# ---------------------------------------------------------------------------
#
# 2-spinor ψ (..., 2, ngkmax) → 4-spinor ψ (..., 4, ngkmax) via
# (α/2) σ · (k+G) ψ_L applied to the upper components.  The constant + σ·p
# contraction live once in :func:`common.bispinor_init.lift_to_4spinor`
# (same math as :func:`common.bispinor_init.get_small_psi_component`, but
# vectorised across k).  The loader only owns the jit-cache-by-sharding
# wrapper below.


@functools.lru_cache(maxsize=None)
def _get_bispinor_lift_jit(sharding: NamedSharding | None):
    """Cache one jit'd copy of the bispinor lift per output sharding.

    Without this, each call to ``_bispinor_lift_kernel`` traces every
    inner ``jnp`` op (``broadcast_in_dim``, ``_take``, ``concatenate``…)
    through pjit's per-op cache.  The parent function id is stable but
    JAX's "tracing context" comparison fires on calls from different
    enclosing scopes, blowing the cache.  Wrapping the whole body in a
    single ``jax.jit`` collapses all inner ops into one cached compile.
    """
    from common.bispinor_init import lift_to_4spinor

    @jax.jit
    def _kernel(psi_2, gvecs, kvecs, bvec):
        out = lift_to_4spinor(psi_2, gvecs, kvecs, bvec)
        if sharding is not None:
            out = jax.lax.with_sharding_constraint(out, sharding)
        return out
    return _kernel


def _bispinor_lift_kernel(
    psi_2: jax.Array,
    gvecs: jax.Array,
    kvecs: jax.Array,
    bvec: jax.Array,
    *,
    sharding: NamedSharding | None,
) -> jax.Array:
    """Append small components → 4-spinor ψ.

    psi_2: (n_k, nb, 2, ngkmax) c128
    gvecs: (n_k, ngkmax, 3)  float64 (already cast)
    kvecs: (n_k, 3)          float64
    bvec : (3, 3)            float64
    """
    return _get_bispinor_lift_jit(sharding)(psi_2, gvecs, kvecs, bvec)


# ---------------------------------------------------------------------------
# phdf5 unfold-and-relayout kernel (module-level so the JIT cache
# survives multiple ``load()`` calls at the same shape signature)
# ---------------------------------------------------------------------------
#
# ``@functools.lru_cache`` keys on the static shape signature; the
# closure ``_per_rank`` is defined INSIDE the cached factory so its
# Python ``id()`` is stable per cache entry.  Repeat invocations with
# the same signature reuse the same closure, which lets JAX's
# trace-cache hit on every inner op (``_where``, ``_take``,
# ``broadcast_in_dim`` …) instead of re-tracing them on each call.

@functools.lru_cache(maxsize=None)
def _phdf5_unfold_kernel(
    mesh: Mesh,
    *,
    n_reads: int,
    n_k: int,
    bands_per_rank: int,
    nspinor: int,
    ngkmax: int,
    unfold: bool,
):
    """Jitted shard_map that takes the FFI's re/im-packed IBZ read and
    returns G-flat ψ in WfnLoader's output layout.

    Steps inside the kernel (mirroring ``PhdfWfnReader._make_unfold_kernel``
    but rearranged to emit ``(n_k, nb_padded, ns, ngkmax)`` c128 directly
    instead of the FFT-box-bound ``(bpr, ns, n_k, ngkmax, 2)``):

      1. Re/im → c128.
      2. ``jnp.take(axis=2, indices=position_in_reads)`` expands the
         IBZ-union axis to the full requested k-set.
      3. If ``unfold``: apply ``where(tr_mask, conj, identity)`` then
         multiply by τ-phase then ``U_spinor`` spinor rotation.  If not
         ``unfold`` (IBZ raw mode): skip step 3.
      4. Transpose ``(bpr, ns, n_k, ngkmax) → (n_k, bpr, ns, ngkmax)``;
         the rank's ``bpr`` slab becomes the local shard of the global
         band axis.

    Output sharding ``P(None, ('x','y'), None, None)`` on global shape
    ``(n_k, nb_padded, ns, ngkmax)``.
    """
    if unfold:
        def _per_rank(cnk_at_ibz, U_per_k, phase_per_k, tr_mask_per_k,
                       position_in_reads):
            cnk = cnk_at_ibz[..., 0] + 1j * cnk_at_ibz[..., 1]
            cnk = jnp.take(cnk, position_in_reads, axis=2)
            cnk = jnp.where(
                tr_mask_per_k[None, None, :, None], jnp.conj(cnk), cnk)
            cnk = cnk * phase_per_k[None, None, :, :]
            cnk = jnp.einsum("kac,bckg->bakg", U_per_k, cnk)
            # (bpr, ns, n_k, ngkmax) → (n_k, bpr, ns, ngkmax)
            return jnp.transpose(cnk, (2, 0, 1, 3))

        in_specs = (
            P(("x", "y"), None, None, None, None),     # cnk_at_ibz
            P(None, None, None),                        # U_per_k
            P(None, None),                              # phase_per_k
            P(None),                                    # tr_mask_per_k
            P(None),                                    # position_in_reads
        )
    else:
        # IBZ raw: skip unfold; just take(position_in_reads) and convert.
        def _per_rank(cnk_at_ibz, position_in_reads):  # type: ignore[misc]
            cnk = cnk_at_ibz[..., 0] + 1j * cnk_at_ibz[..., 1]
            cnk = jnp.take(cnk, position_in_reads, axis=2)
            return jnp.transpose(cnk, (2, 0, 1, 3))

        in_specs = (
            P(("x", "y"), None, None, None, None),     # cnk_at_ibz
            P(None),                                    # position_in_reads
        )

    out_specs = P(None, ("x", "y"), None, None)        # (n_k, bpr→band, ns, ngkmax)
    return jax.jit(shard_map(
        _per_rank, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
        check_rep=False,
    ))


# ===========================================================================
# AsyncWfnReader — prefetched ψ(G) reads via a daemon worker thread.
# ===========================================================================

class AsyncWfnReader:
    """Wrap a :class:`WfnLoader` with a background-thread read queue.

    Both HDF5 backends release the GIL during their host-side read
    (h5py via ``H5Dread`` for the eager backend; the phdf5 FFI for
    the collective backend), so a single daemon thread issuing
    ``loader.load`` overlaps cleanly with whatever the main thread
    runs against the previous result.  Mirrors the write-side pattern
    in :mod:`file_io._slab_io_ffi` (``_dispatch_loop`` at lines
    ~339-410), now sharing a common
    :class:`common.async_io.AsyncDispatcher` backbone.

    Typical usage::

        with AsyncWfnReader(loader, maxsize=2) as reader:
            reader.submit(bands=bcs[0], k="full_bz", sharding=S, bispinor=False)
            for i, bc in enumerate(bcs):
                if i + 1 < len(bcs):
                    reader.submit(bands=bcs[i+1], k="full_bz", sharding=S, bispinor=False)
                psi_G_flat = reader.get()
                # ... downstream compute on psi_G_flat ...

    ``submit`` blocks if ``maxsize`` reads are already in flight
    (back-pressure: each pending read pins a ψ-tile buffer; default
    ``maxsize=2`` lets one read overlap with one downstream compute
    without piling unbounded queues).
    """

    def __init__(
        self,
        loader: "WfnLoader",
        *,
        name: str = "wfn-async-read",
        maxsize: int = 2,
    ) -> None:
        from common.async_io import AsyncDispatcher
        import queue as _queue
        self._loader = loader
        self._results: _queue.Queue = _queue.Queue()
        self._dispatcher = AsyncDispatcher(name=name, maxsize=maxsize)

    def submit(self, *, bands, k, sharding=None, bispinor: bool = False) -> None:
        """Queue a ``loader.load(bands=..., k=..., sharding=..., bispinor=...)``.

        Worker thread runs ``loader.load`` and calls
        ``.block_until_ready()`` on the result before publishing — that
        forces the FFI / h5py read to actually drain on the worker
        thread, so when ``get()`` returns the buffer is fully resident
        and the main thread can dispatch downstream ops without an
        implicit host-side stall.
        """
        loader = self._loader
        results = self._results

        def _task() -> None:
            psi = loader.load(
                bands=bands, k=k, sharding=sharding, bispinor=bispinor)
            jax.block_until_ready(psi)
            results.put(psi)

        self._dispatcher.submit(_task)

    def get(self):
        """Pop the next submitted read's result in FIFO order (blocks)."""
        return self._results.get()

    def drain(self) -> None:
        """Wait for every submitted read to complete."""
        self._dispatcher.drain()

    def close(self) -> None:
        self._dispatcher.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


__all__ = ["WfnLoader", "AsyncWfnReader"]
