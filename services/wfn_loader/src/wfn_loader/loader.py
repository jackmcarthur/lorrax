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
There are TWO, and ``backend='auto'`` (default) picks the lightest that
works:

- **eager** (host h5py + numpy unfold + ``device_put``): single-process,
  mesh-less, or an explicit ``LORRAX_WFN_BACKEND=eager``.  At P>1 with a
  mesh it still reads only THIS rank's band block (``_eager_build_process_local``).
- **phdf5** (collective parallel-HDF5 FFI + on-device unfold): multi-rank
  + 2-D mesh + an FFI .so exporting the kchunk-union read handler on
  either platform.  Reuses the same union-read + unfold kernel that
  powered the legacy ``PhdfWfnReader.coeffs_gspace`` path, but stops one
  step short of the FFT-box scatter so the output stays G-flat (the
  loader's defining layout).

Both backends produce **byte-identical** output for the same ``(bands, k,
sharding, bispinor)`` request — that's the P2 test contract, and it is
the only reason ``LORRAX_WFN_BACKEND`` is safe to expose at all
(``docs/architecture/services.md``).

There was a third, ``phdf5_host``, deleted 2026-08-06: a duplicate compute
path over the eager backend's own POSIX transport, auto-selected by a
missing ``.so`` — which the 2026-08-01 ruling makes a refusal, not a
demotion.  Full history: ``docs/services/wfn_loader.md`` (Backends).  The
two refusal doors below are the tombstone and are load-bearing.

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

P-roadmap — STATUS, not plan (2026-08-07, wave-1 wfn_loader branch)
------------------------------------------------------------------
- P1 DONE: eager backend, bit-matching the legacy WFNReader+SymMaps path.
- P2 DONE: phdf5 backend, bit-identical to eager — ``np.array_equal``, no
  atol, at world 4 on hostile geometry, CPU and CUDA.
- P3 DONE: ``common/wfn_transforms.py`` owns to_box / to_rbox / to_rmu /
  to_rchunk.  It is a CONSUMER of this loader, not part of it (the
  service boundary decision), which is why the old ``load_wfns`` helpers
  live THERE rather than being deleted.
- P4 DONE for every consumer this branch owns: the step-3 replumb moved
  lorrax onto the door, 45 old-path import edges over 36 files → 3 over
  3 (converted delta 42).  The ``SymMaps`` unfold helpers are gone (see
  ``common/symmetry_maps.py``'s head comment for where they went).  The
  three remaining edges ride sibling wave-1 branches by ruling, not by
  oversight.
- P5 NOT DONE, deliberately.  ``PhdfWfnReader`` is gone, but ``WFNReader``
  is a live ALIAS of this class (the same class object, not a subclass)
  and ``src/file_io/wfn_loader.py`` is a transitional SHIM re-exporting
  the door's own objects.  Both STAY until the phase-wide cleanup commit
  after all four wave-1 branches land (coordination ruling 2) — the other
  branches are written against the old spellings and have not rebased.
  That cleanup is the gate; nothing here may delete either early.

Docs: ``docs/services/wfn_loader.md`` (API, contract, backends, measured
baselines); ``services/wfn_loader/docs/DESIGN.md`` (why).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
import types
from pathlib import Path
from typing import Iterator, Literal, Sequence

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
from ._shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._collectives import device_put_process_local


__all__ = [
    "IBZRows", "WfnLoader", "WfnProvenance", "read_wfn_provenance",
    "uniform_band_windows",
]


@dataclass(frozen=True)
class IBZRows:
    """Explicit raw WFN-file IBZ rows, without symmetry unfolding."""

    rows: tuple[int, ...]


KSpec = Sequence[int] | IBZRows | Literal["ibz", "full_bz"]


@dataclass(frozen=True)
class _OccupationSummary:
    nelec: int
    state_capacity: float
    num_electrons: float
    exact_integer: bool
    density_band_stop: int


def _occupation_summary(mf) -> _OccupationSummary:
    """Canonical immutable occupation metadata derived from one MfHeader."""
    nspin = int(mf.nspin)
    nspinor = int(mf.nspinor)
    nkpts = int(mf.nkpts)
    nbands = int(mf.nbands)
    if min(nspin, nspinor, nkpts, nbands) <= 0:
        raise ValueError(
            "WFN occupation dimensions must be positive; got "
            f"nspin={nspin}, nspinor={nspinor}, nkpts={nkpts}, "
            f"nbands={nbands}.")
    occs = np.asarray(mf.occs, dtype=np.float64)
    if np.size(mf.ifmax) > 0:
        nelec = int(np.max(mf.ifmax))
    else:
        nelec = int(np.sum(occs[0, 0] > 0.5))
    if not 0 <= nelec <= nbands:
        raise ValueError(
            f"WFN ifmax implies band boundary {nelec}, outside [0,{nbands}].")
    if not np.all(np.isfinite(occs)):
        raise ValueError("WFN occupations must be finite.")
    weights = np.asarray(mf.kweights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if (weights.shape != (nkpts,)
            or not np.all(np.isfinite(weights)) or np.any(weights < 0.0)
            or not np.isfinite(weight_sum) or weight_sum <= 0.0):
        raise ValueError(
            "WFN k-point weights must be finite, nonnegative, and have "
            f"positive sum; got shape={weights.shape}, sum={weight_sum}.")
    expected_shape = (nspin, nkpts, nbands)
    if occs.shape != expected_shape:
        raise ValueError(
            f"WFN occupations have shape {occs.shape}, expected "
            f"{expected_shape}.")
    capacity = 2.0 / (float(nspin) * float(nspinor))
    num_electrons = capacity * float(np.einsum(
        "k,skb->", weights / weight_sum, occs, optimize=True))
    exact = bool(np.all(occs[:, :, :nelec] == 1.0)
                 and np.all(occs[:, :, nelec:] == 0.0))
    if exact:
        density_stop = nelec
    else:
        nonzero = np.flatnonzero(np.any(occs != 0.0, axis=(0, 1)))
        density_stop = 0 if nonzero.size == 0 else int(nonzero[-1]) + 1
    if density_stop <= 0:
        raise ValueError(
            "physical WFN density has no occupied states; exact Hartree "
            "requires at least one exactly nonzero occupation.")
    return _OccupationSummary(
        nelec, capacity, num_electrons, exact, density_stop)


@dataclass(frozen=True)
class WfnProvenance:
    """Lightweight WFN identity/occupation view; no G or psi payloads."""
    path: str
    energies: np.ndarray
    kpoints: np.ndarray
    nelec: int
    nspinor: int
    nbands: int
    num_electrons: float
    occupations_are_exact_integer: bool
    physical_density_band_stop: int


def read_wfn_provenance(path: str) -> WfnProvenance:
    """Read one canonical MfHeader into the post-hoc authentication view."""
    from file_io.mf_header import read_mf_header_from_file
    resolved = str(Path(path).expanduser().resolve())
    with h5.File(resolved, "r") as handle:
        mf = read_mf_header_from_file(handle)
    summary = _occupation_summary(mf)
    energies = np.array(mf.energies, dtype=np.float64, copy=True)
    kpoints = np.array(mf.kpoints, dtype=np.float64, copy=True)
    energies.flags.writeable = kpoints.flags.writeable = False
    return WfnProvenance(
        resolved, energies, kpoints, summary.nelec, int(mf.nspinor),
        int(mf.nbands), summary.num_electrons,
        summary.exact_integer, summary.density_band_stop)


def uniform_band_windows(b_lo: int, b_hi: int, width: int) -> list:
    """Fixed-width ``(lo, mask)`` windows covering a band range once.

    The final window overlaps instead of shortening, so every consumer sees
    one compiled FFT shape; its 0/1 mask removes the overlap exactly.
    """
    b_lo, b_hi = int(b_lo), int(b_hi)
    span = b_hi - b_lo
    if span <= 0:
        return []
    width = min(max(1, int(width)), span)
    out, counted = [], b_lo
    while counted < b_hi:
        lo = min(counted, b_hi - width)
        mask = np.zeros(width, dtype=np.float64)
        mask[counted - lo:] = 1.0
        out.append((lo, mask))
        counted = lo + width
    return out


class WfnLoader:
    uniform_band_windows = staticmethod(uniform_band_windows)

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
        self._backend_was_auto = (backend == "auto")

        if backend == "auto":
            backend = self._auto_pick_backend()
            # The auto-pick is invisible otherwise, and it is a REAL fork
            # in behaviour (collective MPI-IO FFI read vs h5py union read
            # vs per-rank eager) selected by whether an FFI .so happens to
            # be on LD_LIBRARY_PATH.  Two GPU campaigns differed only in
            # this and nothing in either log said so (scorecard AG).
            self._announce_backend(backend)
        if backend == "phdf5_host":
            raise ValueError(
                "WfnLoader: backend='phdf5_host' was deleted 2026-08-06.  It "
                "was not a third transport — it read with the same "
                "independent POSIX h5py hyperslabs the 'eager' backend "
                "already uses at P>1, and was auto-selected only by a "
                "missing FFI .so, which the 2026-08-01 ruling makes a "
                "refusal rather than a demotion.  Use backend='phdf5' (the "
                "collective FFI read) or backend='eager'.")
        if backend not in ("eager", "phdf5"):
            raise ValueError(
                f"unknown backend {backend!r}; accepted: "
                f"'auto', 'eager', 'phdf5'")
        if backend == "phdf5" and mesh is None:
            raise ValueError(
                f"WfnLoader: backend={backend!r} requires a Mesh; pass mesh=...")
        self.backend = backend

        self._file = h5.File(self._path, "r")

        # The collective read handle (lazy on first load).  Held here so
        # the file is kept open for the loader's lifetime.
        self._slab_io = None
        self._phdf5_static_dev: dict | None = None

        # mf_header surface — same names WFNReader exposes (drop-in compat).
        from file_io.mf_header import (
            bind_mf_attrs, kpt_starts, read_mf_header_from_file)
        hdr = read_mf_header_from_file(self._file)
        bind_mf_attrs(self, hdr)
        # Cartesian-→-crystal: same expression legacy WFNReader exposed.
        self.atom_crys = np.einsum(
            'ij,kj->ki', np.linalg.inv(self.avec).T, self.atom_positions)

        # Immutable occupation metadata is derived by the same bounded header
        # helper used by the lightweight post-hoc provenance reader.
        occupation = _occupation_summary(hdr)
        self.nelec = occupation.nelec
        self._occupation_state_capacity = occupation.state_capacity
        self.num_electrons = occupation.num_electrons
        self._occupations_are_exact_integer = occupation.exact_integer
        self._physical_density_band_stop = occupation.density_band_stop
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

        # Two-component DFT-reference TRS check (default on).  It compares
        # occupied one-particle density operators in G space, using only raw
        # k/-k pairs, spatial-only unfolding, or TRIM closure.  A state made
        # with an antiunitary row is never evidence.  SymMaps consumes the
        # verdict before selecting any time-reversal row.
        self.trs_reference = None
        # Compatibility name retained while report consumers migrate.
        self.density_symmetry = None
        self.trs_holds = True
        self._run_density_symmetry_check()

    # ------------------------------------------------------------------
    # Public identity (DESIGN DECISION 3)
    # ------------------------------------------------------------------
    @property
    def path(self) -> str:
        """The WFN.h5 this loader was constructed on.

        Consumers that re-open a SECOND loader/reader off the same file
        (``file_io/qp_wfn.py``, ``centroid/{charge,current}_density.py``,
        ``psp/dft_operators.py``) were reaching into ``_filename``; this is
        the same string under a public name.  ``_filename`` stays as a
        compat attribute — it costs one assignment in ``__init__`` and the
        legacy ``WFNReader`` spelling is still exported by the shim for
        the sibling wave-1 branches.
        """
        return self._path

    @property
    def occupations_are_exact_integer(self) -> bool:
        """Whether the complete WFN table is exactly ``[1...1,0...0]``.

        ``nelec=max(ifmax)`` is only nominal: smearing tails can extend past
        it, so the complete stored table participates in this predicate.
        """
        return bool(self._occupations_are_exact_integer)

    @property
    def occupation_state_capacity(self) -> float:
        """Electrons represented by one unit WFN occupation."""
        return float(self._occupation_state_capacity)

    @property
    def physical_density_band_stop(self) -> int:
        """Exclusive exact support for a physical WFN density quadrature."""
        return int(self._physical_density_band_stop)

    def physical_density_occupations(
        self,
        *,
        k: str,
        unit_as_none: bool = False,
    ) -> np.ndarray | None:
        """Canonical ``(nk, nb)`` occupation operand for physical density.

        Full-BZ rows use the same cached ``SymMaps.irr_idx_k`` as the
        wavefunction unfold.  ``unit_as_none`` preserves the exact insulating
        reduction.  Collinear ``nspin=2`` is refused because the coefficient
        carrier has no explicit spin-channel axis.
        """
        stop = self.physical_density_band_stop
        if int(self.nspin) != 1:
            raise ValueError(
                "physical_density_occupations: WfnLoader has no explicit "
                "collinear-spin wavefunction axis, so it cannot pair nspin="
                f"{int(self.nspin)} occupations with its psi carrier.")
        occs = np.asarray(self.occs, dtype=np.float64)
        if k == "file":
            selected = occs[0, :, :stop]
        elif k == "full_bz":
            sym = self.symmetry()
            source_rows = np.asarray(
                sym.irr_idx_k, dtype=np.int64)
            if (source_rows.shape != (int(sym.nk_tot),)
                    or np.any(source_rows < 0)
                    or np.any(source_rows >= int(self.nkpts))):
                raise ValueError(
                    "physical_density_occupations: cached full-BZ source-row "
                    f"map is invalid for nk_file={int(self.nkpts)}: "
                    f"shape={source_rows.shape}.")
            selected = occs[0, source_rows, :stop]
        else:
            raise ValueError(
                "physical_density_occupations: k must be 'file' or "
                f"'full_bz', got {k!r}.")
        selected = np.ascontiguousarray(selected, dtype=np.float64)
        if not np.all(np.isfinite(selected)):
            raise ValueError(
                "physical_density_occupations: WFN weights must be finite.")
        if k == "full_bz":
            unfolded_electrons = (self.occupation_state_capacity
                                  * float(np.sum(selected))
                                  / float(selected.shape[0]))
            tol = (256.0 * np.finfo(np.float64).eps
                   * max(1.0, abs(self.num_electrons), float(stop)))
            if not np.isclose(unfolded_electrons, self.num_electrons,
                              rtol=0.0, atol=tol):
                raise ValueError(
                    "full-BZ occupation unfold changes fixed N: "
                    f"{unfolded_electrons:.16e} != "
                    f"{self.num_electrons:.16e} (tol={tol:.3e}).")
        if unit_as_none and np.array_equal(
                selected, np.ones_like(selected)):
            return None
        return selected

    @property
    def kpt_starts(self) -> np.ndarray:
        """Exclusive prefix sum of ``ngk`` — the per-k offset into the
        flat ``wfns/coeffs`` / ``wfns/gvecs`` G axis.

        Public for the two-component reference check, which reads bounded
        raw coefficient slabs without invoking a full-BZ loader path.
        """
        return self._kpt_starts

    def _run_density_symmetry_check(self) -> None:
        """Check the two-component DFT reference before TRS unfolding."""
        from symmetry_maps import cached_density_symmetry_check
        report = cached_density_symmetry_check(self)
        if report is None:
            return
        self.trs_reference = report
        self.density_symmetry = report
        self.trs_holds = bool(report.trs_holds)

    # ------------------------------------------------------------------
    def close(self) -> None:
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._file = None
        sio = getattr(self, "_slab_io", None)
        if sio is not None:
            try:
                sio.close()
            except Exception:
                pass
            self._slab_io = None

    def __enter__(self) -> "WfnLoader":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    _BACKEND_ANNOUNCED: set = set()

    def _announce_backend(self, backend: str) -> None:
        """Debug-print the auto-picked backend once per (backend, world).

        Rank-0 only, once per distinct decision — so a driver that opens
        several loaders does not spam, but a run whose read path changed
        because an ``.so`` appeared on ``LD_LIBRARY_PATH`` says so.
        """
        from runtime import debug_print_enabled
        if not debug_print_enabled():
            return
        try:
            world = int(jax.process_count())
            if int(jax.process_index()) != 0:
                return
        except Exception:
            world = 1
        key = (backend, world)
        if key in WfnLoader._BACKEND_ANNOUNCED:
            return
        WfnLoader._BACKEND_ANNOUNCED.add(key)
        why = {
            "eager": "single-process or mesh-less: host h5py read per rank",
            "phdf5": "collective MPI-IO read through the phdf5 FFI .so",
        }.get(backend, "")
        print(f"  [WfnLoader] read backend = {backend} "
              f"(auto, {world} process{'es' if world != 1 else ''}) — {why}",
              flush=True)

    def _auto_pick_backend(self) -> str:
        """Pick the lightest backend that works.

        Rules:
          * ``LORRAX_WFN_BACKEND`` env set → honour it verbatim (escape
            hatch for A/B testing / forcing eager).
          * Mesh missing → eager (single-device / laptop / pytest).
          * Single-process JAX → eager (no benefit from a collective /
            union read).
          * Multi-process JAX + mesh + CUDA FFI .so loadable → phdf5
            (collective MPI-IO read on GPU).  The probe pins the CUDA
            library explicitly.
          * Multi-process JAX + mesh + host FFI .so exposes the phdf5 read
            handlers → phdf5 (collective MPI-IO read on CPU).  Same single
            read path — the phdf5 C++ read core is shared and only its
            device-staging tail switches (cudaMemcpyAsync H2D vs a host
            memcpy); the ffi_call resolves to the host handler by lowering
            platform and open_file routes the collective lifecycle to the
            host lib (liblorrax_ffi_host.so built with the phdf5 subpackage).
          * Multi-process JAX + mesh, no phdf5-capable FFI .so on EITHER
            platform → **REFUSE**, quoting ``probe_target``'s three-way
            reason for both.  No demotion to the retired ``phdf5_host``
            tier (``docs/services/wfn_loader.md``); an operator who wants
            the host read asks by name, ``LORRAX_WFN_BACKEND=eager``,
            which is checked above and never reaches the probe.
        """
        import os
        # A mesh-less loader can only run eager — there is no device mesh
        # to express a sharded read against.  This dominates the env
        # override: forcing a sharded backend onto a metadata-only,
        # mesh-less loader (htransform builds one before its mesh exists)
        # must not raise.
        if self._mesh is None:
            return "eager"
        forced = os.environ.get("LORRAX_WFN_BACKEND", "").strip().lower()
        if forced == "eager":
            return "eager"
        if forced == "phdf5":
            return forced           # mesh present (checked above) → viable
        if forced == "phdf5_host":
            # A deleted spelling must not resolve to something else: an
            # operator who exported this asked for a specific read path,
            # and silently giving them the FFI one (or eager) is how an
            # A/B measures the wrong arm.
            raise ValueError(
                "LORRAX_WFN_BACKEND=phdf5_host names a backend deleted on "
                "2026-08-06 (it was the eager backend's own POSIX h5py "
                "transport with a different unfold kernel, auto-selected "
                "by a missing FFI .so).  Set 'eager' for the host read or "
                "'phdf5' for the collective FFI read, or unset it.")
        try:
            if int(jax.process_count()) <= 1:
                return "eager"
        except Exception:
            return "eager"
        # GPU first, then CPU: the SAME collective MPI-IO read path, served
        # by whichever platform's FFI library can serve a slab read.  The
        # question "can this platform read a slab" belongs to the door that
        # does the reading, so this ladder asks
        # ``slab_io.probe_read_availability`` and nothing here names an FFI
        # target.  That probe wraps ``ffi_loader.probe_target``, which
        # distinguishes the three ways a target can be unusable — which is
        # why the refusal below quotes its reason verbatim instead of
        # reducing it to a bool — and it is per-platform UNCACHED, because a
        # ladder is exactly what a platform-blind cache would poison.
        from file_io.slab_io import probe_read_availability
        reasons = []
        for _plat in ("CUDA", "cpu"):
            usable, why = probe_read_availability(_plat)
            if usable:
                return "phdf5"
            reasons.append(f"  {_plat}: {why}")
        raise RuntimeError(
            "WfnLoader: no FFI library can serve the collective WFN read "
            f"(jax.process_count()={int(jax.process_count())}, mesh "
            f"{tuple(self._mesh.devices.shape)}), and there is no second "
            "transport to demote to — the h5py 'phdf5_host' tier was "
            "deleted 2026-08-06 (why: docs/services/wfn_loader.md, "
            "Backends; decisions.md 2026-08-01).\n"
            + "\n".join(reasons)
            + "\nRepair the library named above, or set "
              "LORRAX_WFN_BACKEND=eager to take the per-rank host read "
              "deliberately (it is byte-identical, and at P>1 it still "
              "reads only this rank's band block).")

    def adopt_mesh(self, mesh) -> str:
        """Late-bind the device mesh and re-run the auto backend pick.

        For the driver whose mesh cannot exist at construction time:
        kmeans sizes its mesh from the FFT grid THIS file declares (and
        from the charge density read against it), so the loader is
        necessarily built mesh-less and, at P>1, lands on the per-rank
        eager read even though every ψ load it will do is mesh-wide
        (scorecard BD.2).  Calling this right after ``dist.build_mesh``
        gives it the same collective phdf5 route htransform picks.

        Deliberately narrow: only a MULTI-PROCESS run, and only a loader
        constructed with ``backend="auto"`` that resolved to ``eager``,
        switches — an explicit ``backend=`` request (A/B forcing) is
        never overridden, a loader already on a sharded backend keeps
        its mesh, and a single-process run (however many host devices)
        keeps the mesh-less replicated-load contract its callers were
        built against (no band-axis mesh padding appears that was not
        there before).  The switch is safe mid-life because the phdf5
        collective context is created lazily on the first ``load`` and
        the eager state kept so far (the ``coeffs`` dataset HANDLE — no
        data) remains valid for ``load_process_local``.  Returns the
        backend now in force.

        MAY RAISE since 2026-08-06: it re-runs :meth:`_auto_pick_backend`,
        which refuses at P>1 rather than demoting, and startup — this is
        called right after ``dist.build_mesh`` — is the intended place for
        that (``docs/services/wfn_loader.md``).
        """
        if mesh is None or self._mesh is not None:
            return self.backend
        if not self._backend_was_auto or self.backend != "eager":
            return self.backend
        try:
            if int(jax.process_count()) <= 1:
                return self.backend
        except Exception:
            return self.backend
        self._mesh = mesh
        backend = self._auto_pick_backend()
        if backend != self.backend:
            self._announce_backend(backend)
            self.backend = backend
        return self.backend

    # ------------------------------------------------------------------
    # k-set resolution
    # ------------------------------------------------------------------
    def symmetry(self):
        """The loader's ``SymMaps``, built on first use and cached.

        Public accessor (DESIGN DECISION 3).  Idempotent: every call after
        the first returns the SAME object, which is what the two external
        consumers (``common/wfn_transforms.py``, ``common/psi_G_store.py``)
        rely on — they hand it straight to kernels keyed on its identity.
        """
        from symmetry_maps import SymMaps
        if self._sym is None:
            self._sym = SymMaps(self._sym_wfn_stub())
        return self._sym

    #: Compat alias.  Internal call sites and the sibling wave-1 branches
    #: still spell ``_ensure_sym``; one line keeps them working.
    _ensure_sym = symmetry

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
            # CHECKED (not inferred) time-reversal verdict — see
            # ``_run_density_symmetry_check``.  ``SymMaps`` refuses to
            # select time-reversal rows when this is False, whatever the
            # ``ntran``/k-weight flags imply.
            trs_holds=bool(getattr(self, "trs_holds", True)),
        )

    def _resolve_k(self, k: KSpec) -> tuple[np.ndarray, bool]:
        """Resolve a k-spec to (k_idxs, unfold).

        - ``'ibz'``: raw IBZ — returns np.arange(nkpts), unfold=False.
        - ``'full_bz'``: full-BZ unfold — returns np.arange(sym.nk_tot),
          unfold=True.
        - explicit list: interpreted as **full-BZ indices**, unfold=True.
        """
        if isinstance(k, IBZRows):
            return np.asarray(k.rows, dtype=np.int32), False
        if isinstance(k, str):
            if k == "ibz":
                return np.arange(self.nkpts, dtype=np.int32), False
            if k == "full_bz":
                sym = self._ensure_sym()
                return np.arange(int(sym.nk_tot), dtype=np.int32), True
            raise ValueError(f"unknown k-spec {k!r}")
        return np.asarray(k, dtype=np.int32), True

    def _k_cache_key(self, k: KSpec) -> tuple:
        if isinstance(k, IBZRows):
            return ("ibz_rows", tuple(int(v) for v in k.rows))
        if isinstance(k, str):
            return (k,)
        return ("list", tuple(int(v) for v in k))

    # ------------------------------------------------------------------
    # G-vector and ngk_valid accessors
    # ------------------------------------------------------------------
    def kvecs(self, *, k: KSpec = "full_bz") -> np.ndarray:
        """Return the fractional k representatives paired with ``gvecs``.

        This is the coordinate half of the loader's G-flat gauge contract.
        For raw IBZ rows it returns the WFN file's own ``kpoints``; for a
        full-BZ request it returns ``SymMaps.unfolded_kpts`` in the exact
        requested row order. Consumers that form ``k+G`` or apply a Bloch
        phase must take both tables from this loader: rebuilding k from an
        integer grid can choose a different reciprocal-lattice image without
        applying the compensating shift to G.
        """
        k_idxs, unfold = self._resolve_k(k)
        if unfold:
            table = np.asarray(
                self._ensure_sym().unfolded_kpts, dtype=np.float64)
        else:
            table = np.asarray(self.kpoints, dtype=np.float64)
        out = np.asarray(table[np.asarray(k_idxs, dtype=np.int32)],
                         dtype=np.float64)
        if out.shape != (len(k_idxs), 3) or not np.all(np.isfinite(out)):
            raise ValueError(
                "WfnLoader.kvecs: resolved k table must be finite with shape "
                f"({len(k_idxs)}, 3); got {out.shape}.")
        return np.ascontiguousarray(out)

    def gvecs(self, *, k: KSpec = "full_bz") -> np.ndarray:
        """Return ``(n_k, ngkmax, 3)`` int32 — G-vector list per k, padded
        beyond logical ``ngk`` with the FFT-box **pad sentinel**.

        The pad rows are ``common.gvec_fft_box.fft_box_pad_sentinel(
        self.fft_grid)`` — the Nyquist-corner Miller index — NOT zeros.
        Zeros are the Miller index of Γ, a physical component of every
        G-sphere, so a consumer that dropped the ``ngk_valid`` mask used
        to add ``ngkmax − ngk`` extra copies of ψ(Γ) without any symptom.
        The sentinel is a cell no physical G occupies (enforced by
        :func:`~common.gvec_fft_box.pad_gvecs_to_sentinel`, which refuses
        the table otherwise), so the same mistake is now *detectable* —
        see ``gw.kin_ion_io.get_kin_ion_k``'s refusal.

        This is the SAME padded representation ``zeta_q.h5`` stores in
        ``isdf_header/gvec_components``, built by the same routine; the
        two on-disk layouts (ragged ψ, ngkmax-rectangular ζ) differ only
        in how they are read, not in what they become.

        Cached per k-set.
        """
        from common.gvec_fft_box import pad_gvecs_to_sentinel
        key = self._k_cache_key(k)
        if key in self._gvecs_cache:
            return self._gvecs_cache[key]

        k_idxs, unfold = self._resolve_k(k)

        if not unfold:
            rows = []
            for ik in k_idxs:
                start = int(self._kpt_starts[int(ik)])
                end = start + int(self.ngk[int(ik)])
                rows.append(self._gvecs_raw[start:end])
        else:
            sym = self._ensure_sym()
            rows = []
            for nk in k_idxs:
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
                rows.append(np.einsum('ij,kj->ki', sym_krep, k_gvecs) - Gkk)

        # ``ngkmax=self.ngkmax`` (the FILE's max), not max(len(rows)):
        # an explicit k-subset may not contain the widest k, but every
        # ψ buffer in this loader is cut to the file's ngkmax.
        out, _ = pad_gvecs_to_sentinel(
            rows, tuple(int(s) for s in self.fft_grid),
            ngkmax=int(self.ngkmax))
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
        algorithm lives in one place — and hands it the loader's OWN
        rectangular ``(n_k, ngkmax, 3)`` table plus ``ngk_valid``, so the
        builder does a masked scatter with no ragged Python k-loop
        between the two.
        """
        # ``fft_grid`` is in the key because the g_index is a function of
        # (k-set, fft_grid) and the docstring above has always said so —
        # it just wasn't true.  A loader's ``fft_grid`` is read-only in
        # practice, so this has never fired; it costs one tuple.
        cache_key = ("box_index", *self._k_cache_key(k),
                     tuple(int(s) for s in self.fft_grid))
        if cache_key in self._gvecs_cache:
            return self._gvecs_cache[cache_key]

        from common.gvec_fft_box import build_g_index_for_fft_box

        g_index = build_g_index_for_fft_box(
            self.gvecs(k=k), tuple(int(s) for s in self.fft_grid),
            int(self.ngkmax), ngk_valid=self.ngk_valid(k=k))
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
        cache_key = ("box_index_dev", *self._k_cache_key(k),
                     tuple(int(s) for s in self.fft_grid), id(mesh))
        if cache_key in self._gvecs_dev_cache:
            return self._gvecs_dev_cache[cache_key]
        # Resolve the requested sharding (default = replicated 4-axis).
        if sharding is None:
            sharding = NamedSharding(mesh, P(None, None, None, None))
        elif isinstance(sharding, P):
            sharding = NamedSharding(mesh, sharding)
        # Process-local placement (``common.collectives``): every rank
        # builds the SAME index table from the same file, so each may
        # simply declare its own shard.  ``jax.device_put(numpy,
        # multi_process_named_sharding)`` would instead fire JAX's silent
        # ``multihost_utils.assert_equal`` → ``process_allgather``, which
        # for THIS table is ``P·nk·n_rtot·4`` B on every rank: 6.45 GB at
        # P=64, 14.5 GB projected at P=144 (scorecard Y.3, the larger of
        # the two "P-LINEAR loader allgathers").  It was the single
        # biggest collective in a ζ-fit at 276 centroids.
        g_idx_np = self.box_index(k=k)
        dev = device_put_process_local(g_idx_np, sharding)
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
        # Band-axis pad factor, derived from the spec by the SHARED helper.
        # ``common.mtxel_sweep`` derives its own with the same call, so the
        # sweep can consume this loader's psi without re-padding; a second
        # local copy of this arithmetic is how the two would silently drift.
        from runtime.padding import spec_divisor
        return named, spec_divisor(self._mesh, sharding, 1)

    # ------------------------------------------------------------------
    # Shared sharded-read scaffolding (used by the phdf5 path AND the
    # §5b process-local eager path — single-sourced so the "learn my
    # band block / assemble sharded" idiom is written exactly once)
    # ------------------------------------------------------------------
    def _kplan(
        self, k_idxs: np.ndarray, unfold: bool,
    ) -> tuple[np.ndarray, int, np.ndarray, int]:
        """Union-read bookkeeping for the phdf5 collective read.

        Maps the requested k-set to the ascending-sorted union of IBZ
        source k-points (each read from disk exactly once) and each
        request's position in that union — the axis along which the
        union read concatenates its output.

        Returns ``(ibz_unique_sorted, n_reads, position_in_reads, n_k)``.
        """
        if unfold:
            static = self._ensure_phdf5_static()
            ibz_per_full = np.asarray(static["ibz_per_full"])
            ibz_per_k = ibz_per_full[np.asarray(k_idxs, dtype=np.int32)]
        else:
            # k_idxs are already IBZ indices.
            ibz_per_k = np.asarray(k_idxs, dtype=np.int32)
        ibz_unique_sorted = np.unique(ibz_per_k).astype(np.int32)
        n_reads = int(ibz_unique_sorted.size)
        # ``position_in_reads[j]`` = where ibz_per_k[j] sits in the
        # ascending-sorted union.
        position_in_reads = np.searchsorted(
            ibz_unique_sorted, ibz_per_k).astype(np.int32)
        return ibz_unique_sorted, n_reads, position_in_reads, int(len(k_idxs))

    def _assemble_process_local(
        self,
        *,
        global_shape: tuple[int, ...],
        sharding: NamedSharding,
        dtype,
        sharded_axis: int,
        fill_local,
    ) -> jax.Array:
        """Learn THIS rank's block of a band-sharded global array and
        assemble it from a per-rank host build — the scaffold
        :meth:`_eager_build_process_local` runs on.

        Steps (the slab-io process-local idiom, written once):

          1. Materialise a cheap sharded zero proto via the lru-cached
             :func:`_sharded_zero_proto_fn` (each device makes only its
             own zero shard; compiles ONCE per (shape, dtype, sharding)
             signature — no per-call lambda re-lowering).
          2. ``_local_shard_and_global_offset`` reports the local slab's
             shape + global offset.
          3. Validate that ONLY ``sharded_axis`` is sharded — a stray
             non-band spec fails loud rather than silently wrong.
          4. ``fill_local(axis_offset, local_shape) -> np.ndarray``
             builds the rank's host block (caller owns zero-fill of pad
             rows / past-EOF rows).
          5. ``jax.make_array_from_single_device_arrays`` assembles the
             global array — no rank materialises the full host slab.
        """
        from ._collectives import _local_shard_and_global_offset
        global_shape = tuple(int(s) for s in global_shape)
        proto = _sharded_zero_proto_fn(global_shape, dtype, sharding)()
        local_zero, offset = _local_shard_and_global_offset(proto)
        local_shape = tuple(int(x) for x in local_zero.shape)
        del proto, local_zero
        if any(
            ax != sharded_axis
            and (int(offset[ax]) != 0 or local_shape[ax] != global_shape[ax])
            for ax in range(len(global_shape))
        ):
            raise ValueError(
                f"process-local load supports only axis-{sharded_axis} "
                f"(band) sharding; got offset={tuple(int(o) for o in offset)} "
                f"local_shape={local_shape} for global {global_shape}.")
        local_np = fill_local(int(offset[sharded_axis]), local_shape)
        local_arr = jax.device_put(local_np, jax.local_devices()[0])
        return jax.make_array_from_single_device_arrays(
            global_shape, sharding, [local_arr])

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

        Touches NO FFI: these are numpy tables staged process-locally, so
        the collective handle lives in :meth:`_ensure_slab_io` instead of
        being a side effect of building them (it used to be one, gated on
        a ``self.backend`` test that made the pure table build un-callable
        without an ``.so``; ``docs/services/wfn_loader.md``).
        """
        if self._phdf5_static_dev is not None:
            return self._phdf5_static_dev

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
        # ``nspinor`` is NOT optional here.  ``sym.U_spinor`` is always
        # (ntran, 2, 2) — it is built from the CARTESIAN rotations and
        # knows nothing about how many components psi has — so on a scalar
        # (nspinor=1) WFN the un-told helper hands back a 2x2; the former
        # unfold einsum BROADCASTED the size-1 spinor axis instead of raising,
        # and psi came back 2-component holding ``U[a,0]+U[a,1]`` times
        # itself.  Told ``nspinor``, the helper returns the 1x1 identity and
        # the static service application is a genuine no-op.  Registered
        # nspinor=1 loader defect, fixed 2026-08-09;
        # see ``tests/KNOWN_FAILURES.md``.
        from symmetry_maps import trs_augment_U
        U_per = trs_augment_U(
            sym.U_spinor, sym_idx_per_full, n_tran,
            nspinor=int(self.nspinor))                        # (nk_full, ns, ns)

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
        from symmetry_maps import tau_phase_row
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

        # Device-stage the static tables once.  Sharding choice mirrors
        # ``PhdfWfnReader._device_put_static_tables`` — all replicated
        # since they're per-full-BZ-k metadata, not per-band data.
        rep0 = NamedSharding(self._mesh, P())
        rep1 = NamedSharding(self._mesh, P(None))
        rep2 = NamedSharding(self._mesh, P(None, None))
        rep3 = NamedSharding(self._mesh, P(None, None, None))
        # Process-local placement, NOT ``jax.device_put(numpy, sharding)``.
        # On a multi-process mesh the latter fires JAX's silent
        # ``multihost_utils.assert_equal`` → ``process_allgather(tiled=True)``
        # per table (see ``common.collectives.device_put_process_local``).
        # ``phase`` is the second of scorecard Y.3's two P-LINEAR loader
        # allgathers: ``P·nk·ngkmax·16`` = 1.27 GB/rank at P=64, 2.85 GB
        # projected at P=144.  Every rank computes these tables from the
        # same file with the same code, so they are bit-identical by
        # construction and the assertion buys nothing.
        self._phdf5_static_dev = {
            "ibz_per_full": device_put_process_local(ibz_per_full, rep1),
            "sym_idx_per_full": device_put_process_local(
                sym_idx_per_full, rep1),
            "tr_mask_per_full": device_put_process_local(tr_mask, rep1),
            "U_per_full": device_put_process_local(U_per, rep3),
            "phase_per_full": device_put_process_local(phase, rep2),
            "n_tran": n_tran,
            "nk_full": nk_full,
        }
        return self._phdf5_static_dev

    def _ensure_slab_io(self):
        """The SlabIO handle this loader reads psi through, opened once.

        Held for the loader's lifetime; :meth:`close` closes it.  SlabIO's
        constructor runs the collective open's real guards — the capability
        probe, the MPI-world check against the LIVE communicator, and the
        read-side stripe-layout announcement — which this loader used to
        hand-copy from ``_FfiBackend.__init__`` because it called
        ``ffi.io.open_file`` itself.  It no longer does; the door is the
        only phdf5 opener.

        ``ffi.io`` caches one ``PhdfCtx`` per PATH, so a loader and a
        SlabIO on the same file share one collective context and one
        H5Fopen (measured on the step-0 legs: both handles printed equal).
        """
        if self._slab_io is None:
            from file_io.slab_io import SlabIO
            self._slab_io = SlabIO(self._path, mode="r", mesh=self._mesh)
        return self._slab_io

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
        """Collective read through the slab_io door + on-device unfold.

        Output: ``(n_k, nb_padded, nspinor, ngkmax)`` c128 sharded as
        ``out_sharding`` (typically ``P(None, ('x','y'), None, None)``).

        The read is ONE call: n_reads windows of the SAME padded slab
        shape, one per IBZ source k, in one collective H5Dread.  This
        method builds the request — where each window starts and how much
        of the slab is real — and states nothing about hyperslabs, ranks
        or FFI targets; ``SlabIO.read_slabs`` owns all of that, including
        the per-rank clip that used to be a second copy of the arithmetic
        living here (see its docstring, and _slab_io_ffi's
        ``_derive_window_counts``).
        """
        self._ensure_phdf5_static()
        p_x = int(self._mesh.shape["x"])
        p_y = int(self._mesh.shape["y"])
        world = p_x * p_y
        ns = int(self.nspinor)
        ngkmax = int(self.ngkmax)
        nb = nb_padded
        if nb % world:
            raise ValueError(
                f"_phdf5_build: nb_padded={nb} not divisible by world={world}; "
                "this is a loader bug — _default_sharding should have padded.")
        bands_per_rank = nb // world
        b_lo_logical = int(b_lo)
        b_hi_logical = int(b_hi)
        mnband_file = int(self.nbands)

        # ``load`` refuses ``b_hi > mnband``, so the min() is the
        # file-extent backstop for a direct caller of _phdf5_build.  The
        # per-rank clip inside the door then handles the padded case: when
        # ``b_lo + nb_padded > band_extent`` the tail rank(s) get a band
        # count below bands_per_rank and the rank past the end gets 0,
        # which is how the collective path matches the eager backend's
        # zero-fill on BOTH kinds of pad row (past the logical window, and
        # past the file).  Only the extreme case is ours to refuse:
        # ``b_lo`` itself past EOF, which ``load``'s band-range check
        # rejects upstream and this defends anyway.
        band_extent = min(b_hi_logical, mnband_file)
        if b_lo_logical >= mnband_file:
            raise ValueError(
                f"_phdf5_build: b_lo={b_lo_logical} >= mnband={mnband_file}; "
                "entire band window past file extent")

        # Union-read k-plan (shared with the host twin via _kplan).
        ibz_unique_sorted, n_reads, position_in_reads, n_k = self._kplan(
            k_idxs, unfold)

        # One window per IBZ source k, in ascending file order (which is
        # what _kplan's sort buys — the door requires the windows disjoint
        # and ascending).  Dataset layout: (mnband, nspinor, ngktot, 2) f64,
        # so the window axis goes at 2, immediately before the G axis that
        # varies across windows.
        offsets = np.stack([
            [b_lo_logical, 0, int(self._kpt_starts[ibz]), 0]
            for ibz in ibz_unique_sorted
        ], axis=0).astype(np.int64)
        # ...and how much of the padded slab is REAL in each: the band
        # rows up to the logical window end, this k's own ngk on the G
        # axis, everything on the replicated axes.  Stating extents is the
        # whole request; the clip against them, per rank, is the door's.
        valid_shapes = np.stack([
            [band_extent - b_lo_logical, ns, int(self.ngk[ibz]), 2]
            for ibz in ibz_unique_sorted
        ], axis=0).astype(np.int64)

        cnk_at_ibz = self._ensure_slab_io().read_slabs(
            "wfns/coeffs",
            shape=(nb, ns, ngkmax, 2),
            offsets=offsets,
            valid_shapes=valid_shapes,
            partition_spec=P(("x", "y"), None, None, None),
            window_axis=2,
            dtype=np.float64,
        )
        # cnk_at_ibz layout: per-rank
        # (bands_per_rank, ns, n_reads, ngkmax, 2) f64 sharded
        # P(('x','y'), None, None, None, None).

        return self._phdf5_unfold_and_shard(
            cnk_at_ibz, k_idxs=k_idxs, unfold=unfold, n_reads=n_reads,
            n_k=n_k, bands_per_rank=bands_per_rank, ns=ns, ngkmax=ngkmax,
            position_in_reads=position_in_reads, out_sharding=out_sharding)

    def _phdf5_unfold_and_shard(
        self, cnk_at_ibz, *, k_idxs, unfold, n_reads, n_k, bands_per_rank,
        ns, ngkmax, position_in_reads, out_sharding,
    ) -> jax.Array:
        """Tail of the phdf5 read: on-device symmetry unfold (+ transpose
        to WfnLoader's G-flat layout) of the union buffer ``cnk_at_ibz``
        ``(bands_per_rank, ns, n_reads, ngkmax, 2)`` sharded
        ``P(('x','y'), None, None, None, None)``.

        Kept a separate method from :meth:`_phdf5_build` because it is
        the only piece of the collective path that runs without an
        ``.so``, so it is the piece a single-process test can pin against
        ``_eager_build`` (``tests/test_wfn_loader_eager.py``).
        """
        static = self._ensure_phdf5_static()
        unfold_jit = _phdf5_unfold_kernel(
            self._mesh, n_reads=n_reads, n_k=n_k, bands_per_rank=bands_per_rank,
            nspinor=ns, ngkmax=ngkmax, unfold=unfold)
        rep1 = NamedSharding(self._mesh, P(None))
        position_in_reads_dev = device_put_process_local(
            np.asarray(position_in_reads, dtype=np.int32), rep1)

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
        bispinor_lift: str = "raw",
    ) -> jax.Array:
        """ψ(G) for a (band_range, k-set) window.

        Returns ``(n_k, nb_padded, nspinor_out, ngkmax)`` complex128.

        Padding contract:
        * Band axis pad rows are zero-filled.
        * G axis pad rows are zero-filled; the matching ``gvecs(k=...)``
          rows beyond ``ngk_valid(k=...)`` hold the FFT-box pad sentinel
          (:func:`common.gvec_fft_box.fft_box_pad_sentinel`).  Zero
          COEFFICIENT + sentinel G-VECTOR is the whole contract: the
          coefficient makes the slot inert in any contraction, the
          sentinel makes an unmasked slot detectable rather than
          silently aliased onto Γ.
        * For ``k='full_bz'`` (or explicit list): symmetry unfold +
          τ-phase + TR conjugation applied internally.
        * For ``k='ibz'``: raw WFN-file IBZ slab; no unfold.

        ``bispinor=True`` lifts the small spinor components via the selected
        canonical ``bispinor_lift`` representation. ``nspinor_out`` is then 4; else 2 (or
        the file's ``nspinor``).  Requires the WFN file to have
        ``nspinor == 2`` (BGW Pauli convention); ``ValueError``
        otherwise.
        """
        if bispinor and int(self.nspinor) != 2:
            raise ValueError(
                f"WfnLoader.load(bispinor=True) requires a 2-spinor WFN; "
                f"file has nspinor={int(self.nspinor)}.")
        if not bispinor and str(bispinor_lift).strip().lower() != "raw":
            raise ValueError(
                "bispinor_lift selects a four-spinor transform and requires "
                "bispinor=True")

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
                    psi, k=k, sharding=named_sharding,
                    representation=bispinor_lift)
            return psi

        if bispinor:
            # Bispinor lift path (rarer): full build.  Process-local bispinor
            # is future work (the lift appends per-band small components).
            psi_np = self._eager_build(
                b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
                nb_padded=nb_padded)
            psi_j = jnp.asarray(psi_np)
            psi_j = self._apply_bispinor_lift(
                psi_j, k=k, sharding=None,
                representation=bispinor_lift)
            if named_sharding is None:
                return psi_j
            # Process-local shard-out.  ``jax.device_put`` of an
            # UNCOMMITTED array onto a multi-process sharding takes the
            # same hidden ``assert_equal`` branch as the numpy case — and
            # here the operand is the whole ψ window, so the assertion
            # would gather ``P × nk·nb·ns·ngkmax·16``.
            return device_put_process_local(psi_j, named_sharding)

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

    def load_process_local(
        self,
        *,
        bands: tuple[int, int],
        k: KSpec = "full_bz",
        bispinor: bool = False,
        bispinor_lift: str = "raw",
    ) -> jax.Array:
        """ψ(G) for THIS PROCESS ALONE — a **single-device** ``jax.Array``.

        Layout contract
        ---------------
        Returns ``(n_k, nb, nspinor_out, ngkmax)`` c128 committed to
        ``jax.local_devices()[0]``.  ``nb = bands[1] - bands[0]``
        exactly: **no mesh-divisibility band padding**, because nothing
        about this array is global.

        How it differs from :meth:`load` — and why that matters
        -------------------------------------------------------
        :meth:`load` always returns a *global* array: every rank must
        request the SAME ``(bands, k)`` window and each ends up owning a
        band shard of one logical object.  That is the right primitive
        for the GW pipeline, and the wrong one for any kernel whose
        parallelism is over k, because rank *r* asking for ``k=[7]``
        while rank *s* asks for ``k=[9]`` builds a global array whose
        shards are pieces of different physical objects.

        This method is the other primitive: the array it returns is
        addressable by this process only, so each rank may load a
        DIFFERENT ``(bands, k)`` window and run ordinary ``jax.jit``
        computations on it with no collective, no barrier and no
        cross-rank shape agreement.  Combining the per-rank results is
        then an explicit, auditable step (one ``psum`` / one
        ``process_allgather``) rather than something XLA's SPMD
        partitioner infers.

        Used by the exact-V_H kernel (``gw.kin_ion_io``), whose ρ sweep
        and ⟨mk|V_H|nk⟩ sweep are both partitioned over k.

        Identical values to ``load(..., sharding=None)`` on a mesh-less
        loader — same ``_eager_build``, same symmetry unfold — so a P=1
        run is bit-for-bit what the serial path produced.
        """
        if bispinor and int(self.nspinor) != 2:
            raise ValueError(
                f"load_process_local(bispinor=True) requires a 2-spinor WFN; "
                f"file has nspinor={int(self.nspinor)}.")
        if not bispinor and str(bispinor_lift).strip().lower() != "raw":
            raise ValueError(
                "bispinor_lift selects a four-spinor transform and requires "
                "bispinor=True")
        b_lo, b_hi = int(bands[0]), int(bands[1])
        if b_hi <= b_lo:
            raise ValueError(f"empty band range: {bands}")
        if b_lo < 0 or b_hi > int(self.nbands):
            raise ValueError(
                f"band range {bands} out of [0, {self.nbands})")

        k_idxs, unfold = self._resolve_k(k)
        # The phdf5 backend never opens the coeffs dataset (it reads it
        # collectively through the FFI), but the host build below needs
        # the handle.  Opening it lazily costs one h5py lookup and keeps
        # this method usable from a driver whose loader is phdf5-backed.
        if self._coeffs_ds is None:
            self._coeffs_ds = self._file["wfns/coeffs"]

        psi_np = self._eager_build(
            b_lo=b_lo, b_hi=b_hi, k_idxs=k_idxs, unfold=unfold,
            nb_padded=b_hi - b_lo)
        psi = jax.device_put(psi_np, jax.local_devices()[0])
        if bispinor:
            psi = self._apply_bispinor_lift(
                psi, k=k, sharding=None,
                representation=bispinor_lift)
        return psi

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

        Runs on the shared :meth:`_assemble_process_local` scaffold (cached
        sharded-zero proto → ``_local_shard_and_global_offset`` → per-rank
        host build → ``jax.make_array_from_single_device_arrays``) so no rank
        allocates the full (n_k, nb_padded, ns, ngkmax) host array.  The only
        WFN-specific part is the symmetry unfold, which stays in ``_eager_build``.

        Assumes the BAND axis (1) is the only sharded axis (the centroid-load
        spec ``P(None, ('x','y'), None, None)``); asserts the rest are
        replicated so a different spec fails loud rather than silently wrong.
        """
        ns = int(self.nspinor)
        ngkmax = int(self.ngkmax)
        n_k = int(len(k_idxs))
        nb_logical = int(b_hi) - int(b_lo)

        def _fill(band_off: int, local_shape: tuple[int, ...]) -> np.ndarray:
            # Real bands in this rank's block map to its FRONT [0:n_real);
            # the rest (pad bands, or blocks past nb_logical) stay zero.
            local_nb = local_shape[1]
            real_hi = min(band_off + local_nb, nb_logical)
            n_real = max(0, real_hi - band_off)
            if n_real > 0:
                return self._eager_build(
                    b_lo=int(b_lo) + band_off,
                    b_hi=int(b_lo) + band_off + n_real,
                    k_idxs=k_idxs, unfold=unfold, nb_padded=local_nb)
            return np.zeros(local_shape, dtype=np.complex128)

        return self._assemble_process_local(
            global_shape=(n_k, int(nb_padded), ns, ngkmax),
            sharding=named_sharding,
            dtype=jnp.complex128,
            sharded_axis=1,
            fill_local=_fill)

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
        bispinor_lift: str = "raw",
    ) -> Iterator[tuple[tuple[int, int], jax.Array]]:
        """Yield ``((bc_lo, bc_hi), psi)`` for a chunked sweep over bands."""
        if chunk <= 0:
            raise ValueError(f"chunk must be positive, got {chunk}")
        for bc_lo in range(int(b_lo), int(b_hi), int(chunk)):
            bc_hi = min(bc_lo + int(chunk), int(b_hi))
            yield (bc_lo, bc_hi), self.load(
                bands=(bc_lo, bc_hi), k=k, sharding=sharding,
                bispinor=bispinor, bispinor_lift=bispinor_lift)

    # ------------------------------------------------------------------
    # Bispinor lift (G-flat)
    # ------------------------------------------------------------------
    def _apply_bispinor_lift(
        self,
        psi_2: jax.Array,
        *,
        k: KSpec,
        sharding: NamedSharding | None,
        representation: str = "raw",
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

        ``k`` is resolved once by :meth:`kvecs`, the same public loader door
        whose representatives are paired with :meth:`gvecs`.
        """
        gvecs = np.asarray(self.gvecs(k=k))                  # (n_k, ngkmax, 3) int
        kvecs_np = self.kvecs(k=k)
        # WFN stores bvec in reciprocal-lattice units and blat=2π/alat in
        # bohr⁻¹.  This file-format boundary is the one place the bispinor
        # lift converts to the Cartesian momentum its API requires.
        bvec_cart_bohr = (
            float(self.blat) * np.asarray(self.bvec, dtype=np.float64))
        return _bispinor_lift_kernel(
            psi_2,
            jnp.asarray(gvecs, dtype=jnp.float64),
            jnp.asarray(kvecs_np),
            jnp.asarray(bvec_cart_bohr),
            sharding=sharding, representation=representation,
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
        from symmetry_maps import unfold_psi
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
def _get_bispinor_lift_jit(
    sharding: NamedSharding | None,
    representation: str = "raw",
):
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
    def _kernel(psi_2, gvecs, kvecs, bvec_cart_bohr):
        out = lift_to_4spinor(
            psi_2, gvecs, kvecs, bvec_cart_bohr,
            representation=representation)
        if sharding is not None:
            out = jax.lax.with_sharding_constraint(out, sharding)
        return out
    return _kernel


def _bispinor_lift_kernel(
    psi_2: jax.Array,
    gvecs: jax.Array,
    kvecs: jax.Array,
    bvec_cart_bohr: jax.Array,
    *,
    sharding: NamedSharding | None,
    representation: str = "raw",
) -> jax.Array:
    """Append small components → 4-spinor ψ.

    psi_2: (n_k, nb, 2, ngkmax) c128
    gvecs: (n_k, ngkmax, 3)  float64 (already cast)
    kvecs: (n_k, 3)          float64
    bvec_cart_bohr : (3, 3)  float64, reciprocal rows in bohr⁻¹
    """
    return _get_bispinor_lift_jit(sharding, str(representation).strip().lower())(
        psi_2, gvecs, kvecs, bvec_cart_bohr)


# ---------------------------------------------------------------------------
# Sharded-zero proto factory (module-level so the JIT cache survives
# across ``load()`` calls; used by the host union read to learn which
# band block THIS rank owns without re-lowering a fresh lambda per call)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _sharded_zero_proto_fn(global_shape: tuple, dtype, sharding):
    """A cached jitted ``() -> zeros(global_shape) @ sharding``.

    Keyed by (shape, dtype, sharding) so it compiles exactly once per
    signature.  Each device materialises only its own zero shard — no
    full host/device allocation — and JAX's ``.addressable_shards`` then
    reports the local slab's global offset, which is how the host read
    discovers its band block for ``make_array_from_single_device_arrays``.
    """
    return jax.jit(lambda: jnp.zeros(global_shape, dtype=dtype),
                   out_shardings=sharding)


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
        from symmetry_maps import apply_spinor_rotation

        def _per_rank(cnk_at_ibz, U_per_k, phase_per_k, tr_mask_per_k,
                       position_in_reads):
            cnk = cnk_at_ibz[..., 0] + 1j * cnk_at_ibz[..., 1]
            cnk = jnp.take(cnk, position_in_reads, axis=2)
            cnk = jnp.where(
                tr_mask_per_k[None, None, :, None], jnp.conj(cnk), cnk)
            cnk = cnk * phase_per_k[None, None, :, :]
            # Normalize to spinor-last so symmetry_maps owns the same static
            # ns=1/ns=2 application as the eager host unfold.  U's inserted
            # singleton axes align k without materializing a broadcast.
            cnk_last = jnp.transpose(cnk, (0, 2, 3, 1))
            cnk_last = apply_spinor_rotation(
                U_per_k[None, :, None, :, :], cnk_last)
            # (bpr, n_k, ngkmax, ns) → (n_k, bpr, ns, ngkmax)
            return jnp.transpose(cnk_last, (1, 0, 3, 2))

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
        check_vma=False,
    ))
