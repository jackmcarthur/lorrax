"""``ZetaLoader`` — single entry point for ζ(r) loading from ``zeta_q.h5``.

Mirrors :class:`file_io.wfn_loader.WfnLoader`: one ``.load`` call covers
the common windows, ``q`` strings give symbolic ranges, header attrs
match the legacy :class:`ZetaReader` 1:1.

API
---

::

    with ZetaLoader(path, mesh=mesh_xy) as loader:
        # mf_header + isdf_header attrs — same names ZetaReader exposes.
        loader.nspin, loader.fft_grid, loader.n_rmu, loader.vertex_mu_L

        # On-disk q-layout (IBZ-only or full-BZ) — detected from disk shape.
        loader.q_layout           # 'ibz' | 'full_bz'
        loader.n_q_on_disk        # rows in ``zeta_q``
        loader.n_q_full           # ∏ kgrid (always)
        loader.zeta_is_done       # restart guard, False until writer's mark.

        # Slice-style read.  q='ibz' returns every on-disk q; q='full_bz'
        # returns the symmetry-unfolded full-BZ ζ (see Pass-2 note below).
        # ``layout='r_space'`` is the default (legacy callers).  Once V_q
        # is fully on the G-flat path, the default flips to ``'G_flat'``.
        zeta = loader.load(
            q='ibz',                                 # 'ibz' | 'full_bz' | seq[int]
            mu=(0, n_rmu),                           # half-open μ range; None = all
            sharding=P(None, None, ('x', 'y')),      # default for r_space
            layout='r_space',
        )

Backends
--------
* ``eager``   — host h5py read, then ``jax.device_put``.  Single-process,
  CPU JAX, or small files.
* ``phdf5``   — collective FFI read via :class:`SlabIO`.  Multi-rank GPU
  + 2-D mesh; same path the existing ``ZetaReader`` uses.
* ``auto``    — ``phdf5`` if a multi-process mesh is present, else
  ``eager``.

Pass-2 plan
-----------
``q='full_bz'`` is rejected with ``NotImplementedError`` in Pass-1.
The mathematically-correct unfold is ``ζ_full[q, r, μ] = ζ_ibz[i(q),
S_{s(q)}·r + τ_{s(q)}, π_{s(q)}^{-1}(μ)]``  (eq. 3 of
``reports/zeta_ibz_2026-05-11/report.md``).  The r-permutation table
needs ``compute_rgrid_sym_perm`` to land before this can be wired —
adding that in a follow-up commit alongside the ``q='full_bz'`` test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .mf_header import read_mf_header_from_file
from .isdf_header import read_isdf_header_from_file
from .slab_io import SlabIO


__all__ = ["ZetaLoader"]


QSpec = Sequence[int] | Literal["ibz", "full_bz"]
LayoutSpec = Literal["r_space", "G_flat"]


class ZetaLoader:
    """Reader for ``zeta_q.h5`` with a :class:`WfnLoader`-shaped surface."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str | Path,
        *,
        mesh: Mesh | None = None,
        backend: Literal["auto", "eager", "phdf5"] = "auto",
        mode: str = "r",
    ) -> None:
        self._path = str(path)
        self._mesh = mesh

        if backend == "auto":
            backend = self._auto_pick_backend()
        if backend not in ("eager", "phdf5"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "phdf5" and mesh is None:
            raise ValueError(
                "ZetaLoader: backend='phdf5' requires a Mesh; pass mesh=...")
        self.backend = backend

        # Read both headers in one open; reused by load() for shape probes.
        with h5.File(self._path, "r") as f:
            mf = read_mf_header_from_file(f)
            isdf = read_isdf_header_from_file(f)
            zeta_shape = tuple(int(x) for x in f["zeta_q"].shape)
        self._mf = mf
        self._isdf = isdf

        # mf_header attribute surface — same names ZetaReader exposes
        # (drop-in source for callers).
        self.version = mf.version
        self.flavor = mf.flavor
        self.nspin = mf.nspin
        self.nspinor = mf.nspinor
        self.nkpts = mf.nkpts
        self.nbands = mf.nbands
        self.ngkmax = mf.ngkmax
        self.ecutwfc = mf.ecutwfc
        self.kgrid = mf.kgrid
        self.shift = mf.shift
        self.ngk = mf.ngk
        self.ifmin = mf.ifmin
        self.ifmax = mf.ifmax
        self.kweights = mf.kweights
        self.kpoints = mf.kpoints
        self.energies = mf.energies
        self.occs = mf.occs
        self.ng = mf.ng
        self.ecutrho = mf.ecutrho
        self.fft_grid = mf.fft_grid
        self.ntran = mf.ntran
        self.cell_symmetry = mf.cell_symmetry
        self.sym_matrices = mf.sym_matrices
        self.translations = mf.translations
        self.cell_volume = mf.cell_volume
        self.recip_volume = mf.recip_volume
        self.alat = mf.alat
        self.blat = mf.blat
        self.nat = mf.nat
        self.avec = mf.avec
        self.bvec = mf.bvec
        self.adot = mf.adot
        self.bdot = mf.bdot
        self.atom_types = mf.atom_types
        self.atom_positions = mf.atom_positions

        # isdf_header attribute surface.
        self.density = isdf.density
        self.vertex_mu_L = isdf.vertex_mu_L
        self.r_mu_fft_idx = isdf.r_mu_fft_idx
        self.r_mu_crystal = isdf.r_mu_crystal
        self.n_rmu = int(isdf.n_rmu)
        self.zeta_is_done = bool(isdf.zeta_is_done)

        # On-disk q-axis classification.
        self.n_q_on_disk = int(zeta_shape[0])
        self.n_rtot_disk = int(zeta_shape[1])
        self.n_rmu_disk = int(zeta_shape[2])
        self.n_q_full = int(np.prod(self.kgrid))
        self.q_layout: Literal["ibz", "full_bz"] = (
            "full_bz" if self.n_q_on_disk == self.n_q_full else "ibz")

        # SlabIO handle (held open for the loader's lifetime so the
        # phdf5 FFI ctx is reused across reads — same pattern as the
        # existing ZetaReader).  SlabIO's own backend autoselect picks
        # the right path from the mesh; ``backend`` on ZetaLoader is
        # currently advisory (eager vs phdf5 affects the validation
        # above, not yet the slab read).
        self._slab_io: SlabIO | None = SlabIO(
            self._path, mode=mode, mesh=mesh)

    # ------------------------------------------------------------------
    def _auto_pick_backend(self) -> str:
        # Multi-rank GPU + mesh ⇒ phdf5; everything else ⇒ eager.
        if self._mesh is not None and jax.process_count() > 1:
            return "phdf5"
        return "eager"

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._slab_io is not None:
            self._slab_io.close()
            self._slab_io = None

    def __enter__(self) -> "ZetaLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def n_rtot(self) -> int:
        nx, ny, nz = (int(s) for s in self.fft_grid)
        return nx * ny * nz

    @property
    def slab_io(self) -> SlabIO:
        """Underlying SlabIO handle for callers that still need the raw
        ``read_slab('zeta_q', ...)`` contract during migration."""
        if self._slab_io is None:
            raise RuntimeError("ZetaLoader: file already closed")
        return self._slab_io

    # ------------------------------------------------------------------
    # The load contract
    # ------------------------------------------------------------------
    def load(
        self,
        *,
        q: QSpec = "ibz",
        mu: Sequence[int] | tuple[int, int] | slice | None = None,
        sharding: P | None = None,
        layout: LayoutSpec = "r_space",
        qvec_frac: jax.Array | None = None,
        sphere_idx: jax.Array | None = None,
        valid_mu: int | None = None,
    ) -> jax.Array:
        """Read a ζ window.

        Parameters
        ----------
        q
            ``'ibz'``      — every row on disk (works for both IBZ and
                              full-BZ on-disk layouts).
            ``'full_bz'``  — symmetry-unfolded full-BZ ζ.  Not yet
                              implemented (Pass-2; raises
                              ``NotImplementedError``).
            ``Sequence[int]`` — explicit row indices into the on-disk
                              q-axis.  For IBZ-on-disk layouts these are
                              IBZ-row indices.
        mu
            ``(mu_lo, mu_hi)`` half-open range, ``slice``, or ``None``
            (full μ axis).
        sharding
            Output partition spec.  Defaults to the layout-appropriate
            spec (``P(None, None, ('x','y'))`` for ``r_space``,
            ``P(None, ('x','y'), None)`` for ``G_flat``).  Pass ``None``
            to keep the default; pass an explicit ``PartitionSpec`` to
            override.
        layout
            ``'r_space'``   — ``(Q, n_rtot, μ)`` complex128 (default).
            ``'G_flat'``    — ``(Q, μ/p_prod, n_G_sph)`` complex128.
                                Requires ``qvec_frac`` + ``sphere_idx``.
        qvec_frac, sphere_idx, valid_mu
            Forwarded to the G-flat post-processing (FFT + sphere
            gather); ignored on ``r_space``.
        """
        if layout not in ("r_space", "G_flat"):
            raise ValueError(f"layout must be 'r_space' or 'G_flat'; got {layout!r}")
        if self._slab_io is None:
            raise RuntimeError("ZetaLoader: file already closed")

        # --- q axis ---------------------------------------------------
        q_indices = self._resolve_q(q)

        # --- μ axis ---------------------------------------------------
        mu_lo, mu_hi = self._resolve_mu(mu)
        mu_count = mu_hi - mu_lo
        valid_count = mu_count if valid_mu is None else int(valid_mu)

        # --- Default sharding for layout ------------------------------
        if sharding is None:
            sharding = (P(None, None, ('x', 'y')) if layout == 'r_space'
                        else P(None, ('x', 'y'), None))

        # --- r-space read (the disk-native layout) --------------------
        zeta_r = self._read_r_space(
            q_indices=q_indices,
            mu_lo=mu_lo, mu_count=mu_count, valid_mu=valid_count,
            partition_spec=sharding if layout == 'r_space'
                            else P(None, None, ('x', 'y')),
        )

        if layout == 'r_space':
            return zeta_r

        # layout == 'G_flat' — same pipeline as the legacy ZetaReader.
        if qvec_frac is None or sphere_idx is None:
            raise ValueError(
                "ZetaLoader.load(layout='G_flat') requires both "
                "``qvec_frac`` and ``sphere_idx`` (used for the per-q "
                "FFT-box phase and sphere gather).")
        from .zeta_reader import _do_disk_to_G
        nx, ny, nz = (int(s) for s in self.fft_grid)
        n_G_sph = int(np.asarray(sphere_idx).shape[0])
        return _do_disk_to_G(
            zeta_r, qvec_frac,
            mesh_xy=self._mesh, fft_shape=(nx, ny, nz),
            n_G_sph=n_G_sph,
            sphere_idx=jnp.asarray(sphere_idx, dtype=jnp.int32),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_q(self, q: QSpec) -> np.ndarray:
        if isinstance(q, str):
            if q == 'ibz':
                return np.arange(self.n_q_on_disk, dtype=np.int32)
            if q == 'full_bz':
                if self.q_layout == 'full_bz':
                    return np.arange(self.n_q_on_disk, dtype=np.int32)
                raise NotImplementedError(
                    "ZetaLoader.load(q='full_bz') for an IBZ-on-disk file "
                    "requires the r-grid symmetry permutation; "
                    "see the Pass-2 plan in the module docstring.  Use "
                    "q='ibz' + post-V_q unfold "
                    "(gw.v_q_tile._unfold_v_q_ibz_to_full) for now.")
            raise ValueError(f"q string must be 'ibz' or 'full_bz'; got {q!r}")
        arr = np.asarray(q, dtype=np.int32)
        if arr.ndim != 1:
            raise ValueError(f"q indices must be 1-D; got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError("q indices must be non-empty")
        if int(arr.min()) < 0 or int(arr.max()) >= self.n_q_on_disk:
            raise ValueError(
                f"q indices out of [0, {self.n_q_on_disk}); got "
                f"min={int(arr.min())}, max={int(arr.max())}")
        return arr

    def _resolve_mu(self, mu) -> tuple[int, int]:
        if mu is None:
            return 0, self.n_rmu_disk
        if isinstance(mu, slice):
            start = 0 if mu.start is None else int(mu.start)
            stop = self.n_rmu_disk if mu.stop is None else int(mu.stop)
            if mu.step not in (None, 1):
                raise ValueError(f"mu slice must have step 1; got {mu.step}")
            return start, stop
        lo, hi = mu
        return int(lo), int(hi)

    def _read_r_space(
        self,
        *,
        q_indices: np.ndarray,
        mu_lo: int,
        mu_count: int,
        valid_mu: int,
        partition_spec: P,
    ) -> jax.Array:
        """Issue the underlying SlabIO read for an r-space ζ slab.

        Two cases:
        * Contiguous ``q_indices`` (sorted, step-1) → single
          ``read_slab`` with offset+count.
        * Non-contiguous indices → fall back to a per-row read and
          concatenate on host.  Slow path; primarily for diagnostic
          use.  Hot callers should pass a contiguous range.
        """
        if _is_contiguous(q_indices):
            q_offset = int(q_indices[0])
            q_count = int(q_indices.size)
            return self._slab_io.read_slab(
                'zeta_q',
                shape=(q_count, self.n_rtot_disk, mu_count),
                valid_shape=(q_count, self.n_rtot_disk, valid_mu),
                dtype=np.complex128,
                offset=(q_offset, 0, mu_lo),
                mesh=self._mesh,
                partition_spec=partition_spec,
            )

        # Non-contiguous: per-row read.  Cost = n_rows × open-stream
        # overhead; ok for small batches (e.g. caller-selected k-points
        # for diagnostics) but not the V_q hot loop.
        rows = []
        for qi in q_indices:
            row = self._slab_io.read_slab(
                'zeta_q',
                shape=(1, self.n_rtot_disk, mu_count),
                valid_shape=(1, self.n_rtot_disk, valid_mu),
                dtype=np.complex128,
                offset=(int(qi), 0, mu_lo),
                mesh=self._mesh,
                partition_spec=partition_spec,
            )
            rows.append(row)
        return jnp.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_contiguous(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return True
    return bool(np.all(np.diff(arr) == 1))
