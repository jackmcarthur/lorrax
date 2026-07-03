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

from functools import partial
from pathlib import Path
from typing import Literal, Sequence

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .mf_header import bind_mf_attrs, read_mf_header_from_file
from .isdf_header import bind_isdf_attrs, read_isdf_header_from_file
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
        # Dataset name depends on layout: ``zeta_q`` (r-space, legacy)
        # vs ``zeta_q_G`` (G-flat — WFN.h5 ``coeffs`` style, padded to
        # ``ngkmax``).
        with h5.File(self._path, "r") as f:
            mf = read_mf_header_from_file(f)
            isdf = read_isdf_header_from_file(f)
            _ds_name = ('zeta_q_G' if isdf.zeta_layout == 'G_flat'
                         else 'zeta_q')
            zeta_shape = tuple(int(x) for x in f[_ds_name].shape)
        self._mf = mf
        self._isdf = isdf
        self._zeta_dataset_name = _ds_name

        # mf_header attribute surface — same names ZetaReader exposes
        # (drop-in source for callers).
        bind_mf_attrs(self, mf)

        # isdf_header attribute surface.
        bind_isdf_attrs(self, isdf)

        # On-disk q-axis classification.  Dataset shape differs by layout:
        #   r-space: (n_q_disk, n_rtot,        n_rmu) — n_rtot at axis 1
        #   G-flat:  (n_q_disk, n_rmu_padded,  ngkmax) — μ at axis 1
        # so ``n_rtot_disk`` is meaningful only for the r-space case.
        self.n_q_on_disk = int(zeta_shape[0])
        if self.zeta_layout == 'G_flat':
            self.n_rtot_disk = None
            self.n_rmu_disk = int(zeta_shape[1])
        else:
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
        q_indices, need_unfold = self._resolve_q(q)

        # --- μ axis ---------------------------------------------------
        mu_lo, mu_hi = self._resolve_mu(mu)
        mu_count = mu_hi - mu_lo
        valid_count = mu_count if valid_mu is None else int(valid_mu)

        # --- Default sharding for layout ------------------------------
        if sharding is None:
            sharding = (P(None, None, ('x', 'y')) if layout == 'r_space'
                        else P(None, ('x', 'y'), None))

        # --- Disk-native G-flat: read direct, no FFT ------------------
        # The current writer always produces G-flat on disk
        # — slab on disk is ``(Q, μ, ngkmax)``, WFN.h5 ``coeffs`` style.
        # Reading ``layout='r_space'`` would require an inverse FFT
        # we don't support yet; ``layout='G_flat'`` returns the slab as-is
        # (caller's sphere_idx / qvec_frac are accepted for API
        # symmetry but only used to scatter into the consumer's
        # shared sphere downstream).
        if self.zeta_layout == 'G_flat':
            if layout == 'r_space':
                raise NotImplementedError(
                    "ZetaLoader.load(layout='r_space') on a G-flat "
                    "on-disk file would require an inverse FFT; not "
                    "implemented.  Consume the file with "
                    "layout='G_flat' instead.")
            if need_unfold:
                raise NotImplementedError(
                    "ZetaLoader.load(q='full_bz') on a G-flat "
                    "on-disk file: IBZ→full unfold for G-flat ζ_q "
                    "is not yet wired (needs rotation of per-q "
                    "components + the R·V·Rᵀ transverse path).  "
                    "Use q='ibz' and unfold in the V_q consumer.")
            return self._read_g_flat_disk(
                q_indices=q_indices,
                mu_lo=mu_lo, mu_count=mu_count, valid_mu=valid_count,
                partition_spec=sharding,
            )

        # --- r-space read (the disk-native layout) --------------------
        zeta_r = self._read_r_space(
            q_indices=q_indices,
            mu_lo=mu_lo, mu_count=mu_count, valid_mu=valid_count,
            partition_spec=sharding if layout == 'r_space'
                            else P(None, None, ('x', 'y')),
        )

        # --- IBZ → full-BZ unfold (q='full_bz' on an IBZ-on-disk file) ---
        if need_unfold:
            zeta_r = self._unfold_q_full_bz(
                zeta_r, mu_lo=mu_lo, mu_count=mu_count,
                partition_spec=(sharding if layout == 'r_space'
                                else P(None, None, ('x', 'y'))),
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
    def _resolve_q(self, q: QSpec) -> tuple[np.ndarray, bool]:
        """Resolve ``q`` into ``(disk_row_indices, need_full_bz_unfold)``.

        ``need_full_bz_unfold = True`` means the caller asked for
        ``q='full_bz'`` against an IBZ-on-disk file; the ``.load`` path
        will then expand the IBZ rows via the symmetry tables.
        """
        if isinstance(q, str):
            if q == 'ibz':
                return np.arange(self.n_q_on_disk, dtype=np.int32), False
            if q == 'full_bz':
                if self.q_layout == 'full_bz':
                    # Disk is already full-BZ; one row per q.
                    return np.arange(self.n_q_on_disk, dtype=np.int32), False
                # IBZ on disk; we need every IBZ row (the unfold reads them all).
                return np.arange(self.n_q_on_disk, dtype=np.int32), True
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
        return arr, False

    def _resolve_mu(self, mu) -> tuple[int, int]:
        # μ extent on disk: r-space layout puts μ at axis 2,
        # G-flat layout puts μ at axis 1; both use ``self.n_rmu_disk``
        # which is set layout-aware in __init__.
        n_rmu_disk = int(self.n_rmu_disk)
        if mu is None:
            return 0, n_rmu_disk
        if isinstance(mu, slice):
            start = 0 if mu.start is None else int(mu.start)
            stop = n_rmu_disk if mu.stop is None else int(mu.stop)
            if mu.step not in (None, 1):
                raise ValueError(f"mu slice must have step 1; got {mu.step}")
            return start, stop
        lo, hi = mu
        return int(lo), int(hi)

    def _ensure_sym(self):
        """Lazily build a :class:`common.symmetry_maps.SymMaps` from our
        mf_header attributes.  Cached on the instance."""
        sym = getattr(self, "_sym_cache", None)
        if sym is not None:
            return sym
        from common.symmetry_maps import SymMaps
        sym = SymMaps(self)
        self._sym_cache = sym
        return sym

    def _full_bz_unfold_tables(self):
        """Build the host-side tables used by :meth:`_unfold_q_full_bz`:
        ``(full_to_irr_idx, full_to_irr_sym, r_perm, mu_perm)``.

        Raises ``NotImplementedError`` if the IBZ wedge requires
        time-reversal symmetry to reach some full-BZ q — TR support
        will land in a follow-up alongside the spinor-conjugate path.
        """
        cached = getattr(self, "_full_bz_tables", None)
        if cached is not None:
            return cached
        from centroid.orbit_syms import (
            compute_centroid_sym_perm, compute_rgrid_sym_perm)

        sym = self._ensure_sym()
        ntran = int(self.ntran)
        # The eager q-IBZ tables on SymMaps use sym_mats_k which includes TR
        # (the trailing ntran entries are -sym_mats_k).  TR mapping is
        # not yet handled — bail loudly if any q needs it.
        full_to_irr_idx = sym.irr_idx_q
        full_to_irr_sym = sym.sym_idx_q
        if int(np.max(full_to_irr_sym)) >= ntran:
            tr_q = int(np.argmax(full_to_irr_sym >= ntran))
            raise NotImplementedError(
                f"ZetaLoader.load(q='full_bz'): full-BZ q[{tr_q}] needs "
                f"time-reversal symmetry to reach its IBZ parent "
                f"(sym index {int(full_to_irr_sym[tr_q])} ≥ ntran={ntran}).  "
                f"TR maps ζ(r) → ζ*(r) and is not yet wired into the "
                f"unfold.  Workaround: regenerate the IBZ with TR off "
                f"or fall back to ``q='ibz'`` + post-V_q unfold.")

        # Sanity: the IBZ row indices on disk match sym.q_irr_full_idx by
        # construction — the writer stores rows in q_irr_full_idx order
        # (isdf_fitting.py:1689); the reader reads them in the same order.

        r_perm = compute_rgrid_sym_perm(
            sym.sym_matrices, sym.translations, self.fft_grid)
        mu_perm, _mu_L = compute_centroid_sym_perm(
            self.r_mu_fft_idx, sym.sym_matrices,
            sym.translations, self.fft_grid)

        out = (full_to_irr_idx.astype(np.int32),
               full_to_irr_sym.astype(np.int32),
               r_perm.astype(np.int32),
               mu_perm.astype(np.int32))
        self._full_bz_tables = out
        return out

    def _unfold_q_full_bz(
        self,
        zeta_ibz: jax.Array,
        *,
        mu_lo: int,
        mu_count: int,
        partition_spec: P,
    ) -> jax.Array:
        """Expand IBZ ζ to full-BZ ζ via r/μ permutation gathers.

        Math (eq. 3 of ``reports/zeta_ibz_2026-05-11/report.md``)::

            ζ_full[q, r_new, μ_new] = ζ_ibz[i(q),
                                             r_perm[s(q), r_new],
                                             inv_mu_perm[s(q), μ_new]]

        No τ-phase: ζ inside the V_q bilinear contracts out the phase
        (the user-facing ZetaLoader returns the same convention).  The
        gather runs inside a jit cached by output shape + sharding.
        """
        full_to_irr_idx, full_to_irr_sym, r_perm, mu_perm = (
            self._full_bz_unfold_tables())
        n_q_full = int(self.n_q_full)
        n_rtot = int(self.n_rtot_disk)

        # Slice mu_perm columns to the requested μ window.
        mu_slice = slice(mu_lo, mu_lo + mu_count)
        # inv_mu[s, μ_new] = μ_old such that mu_perm[s, μ_old] = μ_new.
        inv_mu_full = np.argsort(mu_perm, axis=-1).astype(np.int32)  # (n_sym, n_rmu)
        # When μ window != full μ axis, inv_mu needs to be clipped to
        # the requested μ_new range AND map back into the SAME window
        # on the IBZ side (otherwise we'd be gathering out-of-window).
        # For the common case ``mu = None`` (full μ), no slicing.
        if mu_count != int(mu_perm.shape[1]):
            raise NotImplementedError(
                "ZetaLoader.load(q='full_bz', mu=<partial>): partial-μ "
                "unfold isn't supported yet — μ permutation can mix "
                "in-window and out-of-window indices.  Pass mu=None "
                "for the whole μ axis, or use q='ibz' + post-V_q unfold.")
        inv_mu = inv_mu_full

        idx_j = jnp.asarray(full_to_irr_idx)            # (n_q_full,)
        sym_j = jnp.asarray(full_to_irr_sym)            # (n_q_full,)
        r_perm_j = jnp.asarray(r_perm)                  # (n_sym, n_rtot)
        inv_mu_j = jnp.asarray(inv_mu)                  # (n_sym, n_rmu)

        out_sharding = NamedSharding(self._mesh, partition_spec)

        @partial(jax.jit, out_shardings=out_sharding)
        def _unfold(z):
            # z: (n_q_ibz, n_rtot, n_rmu); pick parent rows first.
            z_at_irr = z[idx_j]                          # (n_q_full, n_rtot, n_rmu)
            r_gather = r_perm_j[sym_j]                   # (n_q_full, n_rtot)
            mu_gather = inv_mu_j[sym_j]                  # (n_q_full, n_rmu)
            z_r = jnp.take_along_axis(
                z_at_irr, r_gather[:, :, None], axis=1)
            z_full = jnp.take_along_axis(
                z_r, mu_gather[:, None, :], axis=2)
            return z_full

        return _unfold(zeta_ibz)

    def _read_g_flat_disk(
        self,
        *,
        q_indices: np.ndarray,
        mu_lo: int,
        mu_count: int,
        valid_mu: int,
        partition_spec: P,
    ) -> jax.Array:
        """Read a G-flat ζ slab ``(Q, μ, ngkmax)`` from ``zeta_q_G``.

        Pad slots at ``j ≥ ngk[q]`` are zero by writer construction,
        so the caller can ignore them.  Per-q sphere lookup tables
        (``self.gvec_components``, ``self.ngk_per_q``) tell the V_q
        consumer where each disk-axis entry lives in Miller-index space.
        """
        ngkmax = int(self.ngkmax_zeta)
        if _is_contiguous(q_indices):
            q_offset = int(q_indices[0])
            q_count = int(q_indices.size)
            return self._slab_io.read_slab(
                'zeta_q_G',
                shape=(q_count, mu_count, ngkmax),
                valid_shape=(q_count, valid_mu, ngkmax),
                dtype=np.complex128,
                offset=(q_offset, mu_lo, 0),
                mesh=self._mesh,
                partition_spec=partition_spec,
            )

        rows = []
        for qi in q_indices:
            row = self._slab_io.read_slab(
                'zeta_q_G',
                shape=(1, mu_count, ngkmax),
                valid_shape=(1, valid_mu, ngkmax),
                dtype=np.complex128,
                offset=(int(qi), mu_lo, 0),
                mesh=self._mesh,
                partition_spec=partition_spec,
            )
            rows.append(row)
        return jnp.concatenate(rows, axis=0)

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
