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
  CPU JAX, or small files.  This is the only backend P1 implements.
- **phdf5** (collective parallel-HDF5 FFI + on-device unfold): multi-rank
  GPU, large files that don't fit in host RAM.  P2.

Both backends produce **byte-identical** output for the same ``(bands, k,
sharding, bispinor)`` request.

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

import types
from pathlib import Path
from typing import Iterator, Literal, Sequence

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .mf_header import read_mf_header_from_file


__all__ = ["WfnLoader"]


KSpec = Sequence[int] | Literal["ibz", "full_bz"]


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
        self._mesh = mesh

        if backend == "auto":
            backend = self._auto_pick_backend()
        if backend == "phdf5":
            raise NotImplementedError(
                "WfnLoader: phdf5 backend lands in P2; use backend='eager'."
            )
        if backend != "eager":
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

        self._file = h5.File(self._path, "r")

        # mf_header surface — same names WFNReader exposes (drop-in compat).
        hdr = read_mf_header_from_file(self._file)
        self.version = hdr.version
        self.flavor = hdr.flavor
        self.nspin = hdr.nspin
        self.nspinor = hdr.nspinor
        self.nkpts = hdr.nkpts
        self.nbands = hdr.nbands
        self.ngkmax = hdr.ngkmax
        self.ecutwfc = hdr.ecutwfc
        self.kgrid = hdr.kgrid
        self.shift = hdr.shift
        self.ngk = hdr.ngk
        self.ifmin = hdr.ifmin
        self.ifmax = hdr.ifmax
        self.kweights = hdr.kweights
        self.kpoints = hdr.kpoints
        self.energies = hdr.energies
        self.occs = hdr.occs
        self.ng = hdr.ng
        self.ecutrho = hdr.ecutrho
        self.fft_grid = hdr.fft_grid
        self.ntran = hdr.ntran
        self.cell_symmetry = hdr.cell_symmetry
        self.sym_matrices = hdr.sym_matrices
        self.translations = hdr.translations
        self.cell_volume = hdr.cell_volume
        self.recip_volume = hdr.recip_volume
        self.alat = hdr.alat
        self.blat = hdr.blat
        self.nat = hdr.nat
        self.avec = hdr.avec
        self.bvec = hdr.bvec
        self.adot = hdr.adot
        self.bdot = hdr.bdot
        self.atom_types = hdr.atom_types
        self.atom_positions = hdr.atom_positions

        # Eager-backend state: slurp wfns/* into host RAM.  Same memory
        # behaviour as the legacy WFNReader.
        self._coeffs_raw = self._file["wfns/coeffs"][:]   # (nb, ns, ngktot, 2) f64
        self._gvecs_raw = self._file["wfns/gvecs"][:]     # (ngktot, 3) int
        # kpt_starts = cumulative sum of ngk.
        self._kpt_starts = np.zeros(self.nkpts, dtype=np.int64)
        for ik in range(1, self.nkpts):
            self._kpt_starts[ik] = self._kpt_starts[ik - 1] + int(self.ngk[ik - 1])

        # Lazy state.
        self._sym = None
        self._gvecs_cache: dict[tuple, np.ndarray] = {}
        self._ngk_valid_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    def close(self) -> None:
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._file = None

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
        # P1 only has eager.  P2 will check for FFI .so + multi-rank + mesh.
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
                g_rot = sym.get_gvecs_kfull(self, int(nk))  # reuses existing impl
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
            ibz_per_full = np.asarray(sym.irk_to_k_map, dtype=np.int32)
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

    @staticmethod
    def _pad_to(n: int, multiple: int) -> int:
        rem = n % multiple
        return n + (multiple - rem) if rem else n

    # ------------------------------------------------------------------
    # The main load
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
        the file's ``nspinor``).  P1 raises ``NotImplementedError`` for
        ``bispinor=True``; lands in P2 alongside phdf5.
        """
        if bispinor:
            raise NotImplementedError(
                "WfnLoader: bispinor lift lands in P2; in P1 use the "
                "legacy load_wfns path for current-density runs.")

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
        nb_padded = self._pad_to(nb_logical, p_band)

        psi_np = self._eager_build(
            b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
            nb_padded=nb_padded)

        if named_sharding is None:
            return jnp.asarray(psi_np)
        return jax.device_put(jnp.asarray(psi_np), named_sharding)

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
                raw = self._coeffs_raw[b_lo:b_hi, :, start:end, :]
                out[j, :nb_logical, :, : end - start] = (
                    raw[..., 0] + 1j * raw[..., 1])
            return out

        # Full-BZ unfold path.
        sym = self._ensure_sym()
        ntran = int(sym.sym_matrices.shape[0])
        U_per = np.asarray(sym.U_spinor)
        for j, nk in enumerate(k_idxs):
            nk_int = int(nk)
            sym_idx = int(sym.irk_sym_map[nk_int])
            kbar = int(sym.irk_to_k_map[nk_int])
            sym_krep = np.asarray(sym.sym_mats_k[sym_idx], dtype=np.int32)
            ngk_k = int(self.ngk[kbar])

            start = int(self._kpt_starts[kbar])
            end = start + ngk_k
            raw = self._coeffs_raw[b_lo:b_hi, :, start:end, :]
            cnk = raw[..., 0] + 1j * raw[..., 1]                    # (nb, ns, ngk_k)

            if sym_idx >= ntran:
                cnk = np.conj(cnk)
            else:
                tau = np.asarray(self.translations[sym_idx], dtype=np.float64)
                if np.any(np.abs(tau) > 1e-12):
                    g_bar = self._gvecs_raw[start:end]              # (ngk_k, 3)
                    rotated = (sym_krep @ g_bar.T).T                # (ngk_k, 3)
                    phase = np.exp(-1j * rotated.astype(np.float64) @ tau)
                    cnk = cnk * phase[None, None, :]

            cnk = np.einsum("jk,nkl->njl", U_per[sym_idx], cnk)     # spinor rotate
            out[j, :nb_logical, :, :ngk_k] = cnk
        return out
