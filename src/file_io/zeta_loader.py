"""``ZetaLoader`` — the single reader for ``zeta_q.h5``.

One class covers both read surfaces (they were previously split across a
``ZetaReader``/``ZetaLoader`` pair with duplicated header/lifecycle code;
merged 2026-07-09):

* **Slab API** (the production V_q reader of record):
  :meth:`read_zeta_r_slab` / :meth:`read_zeta_G_slab` — explicit
  ``(q_offset, q_count, mu_offset, mu_count)`` windows; a ``mu_count``
  past the on-disk extent comes back zero-filled (SlabIO's business,
  decisions.md 2026-08-04).
* **Load API** (WfnLoader-shaped, test bench + future consumers):
  :meth:`load` — symbolic ``q='ibz' | 'full_bz' | seq[int]`` ranges,
  μ slices, and the IBZ→full-BZ symmetry unfold of ζ(r).

Header surface: eager :class:`MfHeader` attributes (``nspin``,
``kgrid``, ``fft_grid``, ``sym_matrices``, …) and :class:`IsdfHeader`
attributes (``vertex_mu_L``, ``r_mu_fft_idx``, ``n_rmu``,
``zeta_layout``, …) — same names ``WFNReader`` exposes, so a
``ZetaLoader`` drops in wherever a ``WFNReader`` was used for header
information.

On-disk layouts: legacy r-space files store ``zeta_q`` shape
``(n_q, n_rtot, n_rmu)``; G-flat files store ``zeta_q_G`` shape
``(n_q, n_rmu_padded, ngkmax)`` (WFN.h5 ``coeffs`` style, per-q sphere
from ``isdf_header/gvec_components``).  IBZ-only q-axes are detected
from the disk shape (``q_layout``); full-BZ callers unfold post-V_q via
:func:`common.symmetry_maps.unfold_v_q` (or ``load(q='full_bz')`` for
the r-space ζ itself).

I/O backend: all data reads go through one :class:`SlabIO` handle, held
open for the loader's lifetime to amortise open/close on the FFI
value every other SlabIO consumer uses (``None`` = SlabIO's own
auto-route).  Use as a context manager or call :meth:`close`.

HEADER-ONLY MODE (``mesh=None``).  The header surface above is read with
plain serial h5py inside ``__init__`` and needs no transport at all, but
until now ``mesh=None`` reached ``SlabIO(mesh=None)`` and raised, and a
stack without the phdf5 FFI could not even ask this file how many q it
holds — so a caller that wanted ONLY the layout contract (``ngk``,
``gvec_components``, ``zeta_cutoff_ry``, the mf_header crystal block)
had to open the file a second time with its own h5py and re-derive it.
That second reader is the thing this class exists to prevent.  With
``mesh=None`` the loader skips the SlabIO open entirely: every header
attribute works, and :attr:`slab_io` / :meth:`load` / the two
``read_*_slab`` methods refuse, naming the missing mesh.  Passing a mesh
is UNCHANGED in every respect (eager SlabIO open, same collective, same
refusal when the FFI is absent).
"""
from __future__ import annotations

import os
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
    """Reader for ``zeta_q.h5`` produced by ``isdf_fitting.fit_zeta_to_h5``."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str | Path,
        *,
        mesh: Mesh | None = None,
        mode: str = "r",
    ) -> None:
        self._path = str(path)
        self._mesh = mesh

        # Read both headers + the ζ dataset shape in one open.
        with h5.File(self._path, "r") as f:
            mf = read_mf_header_from_file(f)
            isdf = read_isdf_header_from_file(f)
            _ds_name = ('zeta_q_G' if isdf.zeta_layout == 'G_flat'
                        else 'zeta_q')
            zeta_shape = tuple(int(x) for x in f[_ds_name].shape)
        self._mf = mf
        self._isdf = isdf
        self._zeta_dataset_name = _ds_name
        self._zeta_disk_shape = zeta_shape

        # mf_header + isdf_header attribute surfaces (WFNReader-shaped).
        bind_mf_attrs(self, mf)
        bind_isdf_attrs(self, isdf)

        # On-disk q-axis classification.  Dataset shape differs by layout:
        #   r-space: (n_q_disk, n_rtot,       n_rmu)  — n_rtot at axis 1
        #   G-flat:  (n_q_disk, n_rmu_padded, ngkmax) — μ at axis 1
        self.n_q_on_disk = int(zeta_shape[0])
        if self.zeta_layout == 'G_flat':
            # ``n_rtot_disk`` is kept meaningful (= ∏ fft_grid) for code
            # that probes the r-extent; n_G_sph_disk is the per-q padded
            # sphere size.
            nx, ny, nz = (int(s) for s in self.fft_grid)
            self.n_rtot_disk = nx * ny * nz
            self.n_rmu_disk = int(zeta_shape[1])
            self.n_G_sph_disk = int(zeta_shape[2])
        else:
            self.n_rtot_disk = int(zeta_shape[1])
            self.n_rmu_disk = int(zeta_shape[2])
            self.n_G_sph_disk = None
        self.n_q_full = int(np.prod(self.kgrid))
        self.q_layout: Literal["ibz", "full_bz"] = (
            "full_bz" if self.n_q_on_disk == self.n_q_full else "ibz")

        # ── Read-side provenance gate ────────────────────────────────
        # ``zeta_is_done`` is the writer's completeness flag: stamped
        # False before the first chunk, flipped True by ``mark_zeta_done``
        # only after the last one drains.  Until now NOTHING read it, so a
        # ζ left behind by a job that died mid-write (this happened
        # repeatedly during the 2026-07 campaign — SIGABRT at V_q entry,
        # RESOURCE_EXHAUSTED in Stage C) was indistinguishable from a
        # complete one and its undefined trailing q-blocks flowed straight
        # into V_q → W → Σ with rc=0.  Refuse at open instead.
        _allow_partial = bool(int(
            os.environ.get("LORRAX_ALLOW_PARTIAL_ZETA", "0") or "0"))
        if mode == "r" and not bool(self.zeta_is_done) and not _allow_partial:
            raise ValueError(
                f"{self._path} has isdf_header/zeta_is_done=False: the ζ fit "
                f"that wrote it did not finish, so its trailing q-blocks are "
                f"undefined.  Reading it would produce a physically "
                f"meaningless V_q / W / Σ without any error.  Delete the file "
                f"and re-run the fit (restart=false), or point at a complete "
                f"ζ.  Set LORRAX_ALLOW_PARTIAL_ZETA=1 to override for "
                f"debugging.")
        # Header-vs-dataset agreement.  The two are written by different
        # calls (``write_isdf_header`` then the SlabIO append); a crash
        # between them, or a stale header surviving a rewrite, leaves a
        # file whose centroid table describes a different basis than its
        # ζ block.  In G-flat layout μ is the on-disk axis 1 (padded to
        # the mesh), so the header count is a floor, not an equality.
        _n_rmu_header = int(self.n_rmu)
        if self.zeta_layout == 'G_flat':
            if self.n_rmu_disk < _n_rmu_header:
                raise ValueError(
                    f"{self._path} is inconsistent: isdf_header lists "
                    f"{_n_rmu_header} centroids but zeta_q_G has only "
                    f"{self.n_rmu_disk} μ rows.  The header and the ζ block "
                    f"were written by different runs — the file is corrupt.")
        elif self.n_rmu_disk != _n_rmu_header:
            raise ValueError(
                f"{self._path} is inconsistent: isdf_header lists "
                f"{_n_rmu_header} centroids but zeta_q has {self.n_rmu_disk}. "
                f"The header and the ζ block were written by different runs.")

        # SlabIO handle (held open for the loader's lifetime so the
        # phdf5 FFI ctx is reused across reads).  ``mesh=None`` is
        # HEADER-ONLY mode: no transport is opened, so nothing here
        # probes the FFI and every data read refuses instead (see the
        # module docstring and :attr:`slab_io`).
        self._slab_io: SlabIO | None = (
            None if mesh is None else SlabIO(self._path, mode=mode, mesh=mesh))
        self._header_only = mesh is None

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
    def has_slab_io(self) -> bool:
        """True when this loader can serve DATA reads (a mesh was given
        and the SlabIO handle is still open).  Header attributes work
        either way."""
        return self._slab_io is not None

    @property
    def slab_io(self) -> SlabIO:
        """Underlying SlabIO handle for callers that still need the raw
        ``read_slab('zeta_q', ...)`` contract during migration."""
        if self._slab_io is None:
            if self._header_only:
                raise RuntimeError(
                    f"ZetaLoader({self._path!r}) was opened HEADER-ONLY "
                    f"(mesh=None), so it has no transport and cannot read ζ. "
                    f"Re-open it with mesh=<the run's mesh_xy> to get the "
                    f"SlabIO tile path.")
            raise RuntimeError("ZetaLoader: file already closed")
        return self._slab_io

    # ------------------------------------------------------------------
    # Slab API (production V_q reader of record)
    # ------------------------------------------------------------------
    def read_zeta_r_slab(
        self,
        *,
        q_offset: int,
        q_count: int,
        mu_offset: int,
        mu_count: int,
        mesh: Mesh | None = None,
        partition_spec=P(None, None, ('x', 'y')),
    ) -> jax.Array:
        """Read an r-space ζ slab.

        Returns ``(q_count, n_rtot, mu_count)`` complex128, sharded
        per ``partition_spec``.  Ask for the μ extent you want to
        consume: SlabIO fills what the dataset covers and zeroes the
        rest (decisions.md 2026-08-04), so a padded ``mu_count`` needs
        no further argument.
        """
        if mesh is None:
            mesh = self._mesh
        n_rtot = self.n_rtot_disk
        return self.slab_io.read_slab(
            'zeta_q',
            shape=(int(q_count), int(n_rtot), int(mu_count)),
            dtype=np.complex128,
            offset=(int(q_offset), 0, int(mu_offset)),
            mesh=mesh,
            partition_spec=partition_spec,
        )

    def read_zeta_G_slab(
        self,
        *,
        q_offset: int,
        q_count: int,
        mu_offset: int,
        mu_count: int,
        qvec_batch_frac: jax.Array,
        sphere_idx: jax.Array | None,
        mesh: Mesh | None = None,
    ) -> jax.Array:
        """Read ζ in G-flat layout.

        G-flat-on-disk files return the ``(Q, μ, ngkmax)`` slab directly
        (per-q phase already baked in by the writer).  Legacy r-space
        files go through: r-space slab read → per-q Bloch phase → 3D FFT
        (μ-sharded) → sphere gather (:func:`_do_disk_to_G`).

        Returns
        -------
        zeta_G : jax.Array
            Shape ``(q_count, μ_per_rank, n_G_sph)`` complex128,
            sharded ``P(None, ('x','y'), None)``.

        Parameters
        ----------
        q_offset, q_count : int
            Slab range along the on-disk q axis.  Caller-managed
            indexing — under IBZ-only layouts the offset is the IBZ
            index.
        mu_offset, mu_count : int
            Slab range along the μ axis (centroid axis).
        qvec_batch_frac : jax.Array
            ``(Q, 3)`` fractional q-vectors in kgrid units (BGW
            wrapped-to-(-nk/2, nk/2) divided by kgrid).  Used to apply
            the per-q FFT-box phase separably on the r-space path;
            ignored for G-flat-on-disk files.
        sphere_idx : jax.Array | None
            Flat-FFT indices that define the G-sphere (or None to
            keep the full FFT box).
        mesh : Mesh | None
            Override of the loader's stored mesh.
        """
        if mesh is None:
            mesh = self._mesh

        nx, ny, nz = (int(s) for s in self.fft_grid)
        n_rtot = nx * ny * nz
        if sphere_idx is not None:
            n_G_sph = int(np.asarray(sphere_idx).shape[0])
        else:
            n_G_sph = n_rtot
        sphere_jx = (jnp.asarray(sphere_idx, dtype=jnp.int32)
                     if sphere_idx is not None else None)

        if self.zeta_layout == 'G_flat':
            # File is already G-flat.  Read the (q_count, mu_count,
            # ngkmax) slab directly.  ``qvec_batch_frac`` is ignored
            # — the per-q phase is already baked into the on-disk
            # tensor by the writer.  The G-axis on disk is now a
            # **per-q** WFN.h5-style sphere of size ``ngkmax``
            # (positions vary per q via ``isdf_header/gvec_components``),
            # so a single shared ``sphere_idx`` can NOT be used to
            # narrow with one ``jnp.take`` — that would pick the same
            # disk position for every q, which is per-q wrong.
            # The proper per-q scatter to a consumer's shared sphere
            # via the components table belongs in the V_q wrapper and
            # is not implemented here yet.
            n_G_sph_disk = int(self.n_G_sph_disk)
            zeta_g_disk = self.slab_io.read_slab(
                self._zeta_dataset_name,
                shape=(int(q_count), int(mu_count), n_G_sph_disk),
                dtype=np.complex128,
                offset=(int(q_offset), int(mu_offset), 0),
                mesh=mesh,
                partition_spec=P(None, ('x', 'y'), None),
            )
            if sphere_jx is not None and n_G_sph != n_G_sph_disk:
                raise NotImplementedError(
                    "ZetaLoader.read_zeta_G_slab: on-disk per-q sphere "
                    f"(ngkmax={n_G_sph_disk}) ≠ caller's shared sphere "
                    f"(n_G_sph={n_G_sph}).  Per-q → shared-sphere "
                    "scatter via gvec_components is not yet wired into "
                    "the V_q hot loop; pass sphere_idx=None to consume "
                    "the raw slab, or refit with the r-space writer.")
            return zeta_g_disk

        # Legacy 'r_space' path: read r-space slab + FFT + sphere gather.
        zeta_disk = self.read_zeta_r_slab(
            q_offset=q_offset, q_count=q_count,
            mu_offset=mu_offset, mu_count=mu_count,
            mesh=mesh,
        )

        return _do_disk_to_G(
            zeta_disk, qvec_batch_frac,
            mesh_xy=mesh, fft_shape=(nx, ny, nz),
            n_G_sph=n_G_sph, sphere_idx=sphere_jx,
        )

    # ------------------------------------------------------------------
    # Load API (WfnLoader-shaped)
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
    ) -> jax.Array:
        """Read a ζ window.

        Parameters
        ----------
        q
            ``'ibz'``      — every row on disk (works for both IBZ and
                              full-BZ on-disk layouts).
            ``'full_bz'``  — symmetry-unfolded full-BZ ζ (r-space files
                              only; G-flat unfold not yet wired).
            ``Sequence[int]`` — explicit row indices into the on-disk
                              q-axis.  For IBZ-on-disk layouts these are
                              IBZ-row indices.
        mu
            ``(mu_lo, mu_hi)`` half-open range, ``slice``, or ``None``
            (full μ axis).
        sharding
            Output partition spec.  Defaults to the layout-appropriate
            spec (``P(None, None, ('x','y'))`` for ``r_space``,
            ``P(None, ('x','y'), None)`` for ``G_flat``).
        layout
            ``'r_space'``   — ``(Q, n_rtot, μ)`` complex128 (default).
            ``'G_flat'``    — ``(Q, μ/p_prod, n_G_sph)`` complex128.
                                Requires ``qvec_frac`` + ``sphere_idx``.
        qvec_frac, sphere_idx
            Forwarded to the G-flat post-processing (FFT + sphere
            gather); ignored on ``r_space``.
        """
        if layout not in ("r_space", "G_flat"):
            raise ValueError(f"layout must be 'r_space' or 'G_flat'; got {layout!r}")
        _ = self.slab_io  # refuse now, naming header-only vs closed

        # --- q axis ---------------------------------------------------
        q_indices, need_unfold = self._resolve_q(q)

        # --- μ axis ---------------------------------------------------
        mu_lo, mu_hi = self._resolve_mu(mu)
        mu_count = mu_hi - mu_lo

        # --- Default sharding for layout ------------------------------
        if sharding is None:
            sharding = (P(None, None, ('x', 'y')) if layout == 'r_space'
                        else P(None, ('x', 'y'), None))

        # --- Disk-native G-flat: read direct, no FFT ------------------
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
                mu_lo=mu_lo, mu_count=mu_count,
                partition_spec=sharding,
            )

        # --- r-space read (the disk-native layout) --------------------
        zeta_r = self._read_r_space(
            q_indices=q_indices,
            mu_lo=mu_lo, mu_count=mu_count,
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

        # layout == 'G_flat' — same disk→G pipeline as read_zeta_G_slab.
        if qvec_frac is None or sphere_idx is None:
            raise ValueError(
                "ZetaLoader.load(layout='G_flat') requires both "
                "``qvec_frac`` and ``sphere_idx`` (used for the per-q "
                "FFT-box phase and sphere gather).")
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

        # Slice mu_perm columns to the requested μ window.
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
            return self.slab_io.read_slab(
                'zeta_q_G',
                shape=(q_count, mu_count, ngkmax),
                dtype=np.complex128,
                offset=(q_offset, mu_lo, 0),
                mesh=self._mesh,
                partition_spec=partition_spec,
            )

        rows = []
        for qi in q_indices:
            row = self.slab_io.read_slab(
                'zeta_q_G',
                shape=(1, mu_count, ngkmax),
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
            return self.slab_io.read_slab(
                'zeta_q',
                shape=(q_count, self.n_rtot_disk, mu_count),
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
            row = self.slab_io.read_slab(
                'zeta_q',
                shape=(1, self.n_rtot_disk, mu_count),
                dtype=np.complex128,
                offset=(int(qi), 0, mu_lo),
                mesh=self._mesh,
                partition_spec=partition_spec,
            )
            rows.append(row)
        return jnp.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Internal: r-space slab → G-flat — module-level so the jitted helpers
# cache across calls and don't recompile per-ZetaLoader instance.
# ---------------------------------------------------------------------------

_disk_to_G_cache: dict = {}


def _do_disk_to_G(
    zeta_disk: jax.Array,
    qvec_batch_frac: jax.Array,
    *,
    mesh_xy: Mesh,
    fft_shape: tuple[int, int, int],
    n_G_sph: int,
    sphere_idx: jax.Array | None,
) -> jax.Array:
    """r-space ζ slab + per-q fractional q-vec → G-flat ζ.

    Pipeline: transpose → reshape to ``(Q, μ, nx, ny, nz)`` → apply
    separable per-q Bloch phase ``exp(-2πi q·r)`` via
    :func:`common.wfn_transforms.apply_bloch_phase` → 3D FFT (μ-sharded)
    → sphere gather.

    Cached by ``(mesh_xy, shape, n_G_sph, sphere_idx)``.
    """
    from common.fft_helpers import make_sharded_fftn_3d
    from common.wfn_transforms import apply_bloch_phase

    nx, ny, nz = (int(s) for s in fft_shape)
    n_rtot = nx * ny * nz
    Q, n_rtot_in, mu_total = (int(s) for s in zeta_disk.shape)
    if n_rtot_in != n_rtot:
        raise ValueError(
            f"_do_disk_to_G: ζ slab n_rtot={n_rtot_in} disagrees with "
            f"FFT grid product {n_rtot}.")

    key = (
        id(mesh_xy), Q, mu_total, nx, ny, nz, int(n_G_sph),
        id(sphere_idx),
    )
    fn = _disk_to_G_cache.get(key)
    if fn is None:
        blk_xy_sh = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
        blk_xy_5d_sh = NamedSharding(
            mesh_xy, P(None, ('x', 'y'), None, None, None))
        local_fftn = make_sharded_fftn_3d(
            mesh_xy, P(None, ('x', 'y'), None, None, None),
            P(None, ('x', 'y'), None, None, None))
        qvec_sh = NamedSharding(mesh_xy, P(None, None))
        zeta_disk_sh = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))

        @partial(
            jax.jit,
            in_shardings=(zeta_disk_sh, qvec_sh),
            out_shardings=blk_xy_sh,
        )
        def _f(z, qvec_frac):
            # (Q, n_rtot, mu) → (Q, mu, n_rtot) → (Q, mu, nx, ny, nz)
            z = jax.lax.with_sharding_constraint(
                jnp.transpose(z, (0, 2, 1)), blk_xy_sh)
            Qd, mu_d, _ = z.shape
            z5 = z.reshape(Qd, mu_d, nx, ny, nz)
            z5 = apply_bloch_phase(z5, qvec_frac, (nx, ny, nz), sign=-1)
            z5 = jax.lax.with_sharding_constraint(z5, blk_xy_5d_sh)
            box = local_fftn(z5)
            box = jax.lax.with_sharding_constraint(
                box.reshape(Qd, mu_d, n_rtot), blk_xy_sh)
            if sphere_idx is not None:
                z_G = jnp.take(box, sphere_idx, axis=-1)
            else:
                z_G = box
            return jax.lax.with_sharding_constraint(z_G, blk_xy_sh)

        _disk_to_G_cache[key] = _f
        fn = _f

    return fn(zeta_disk, qvec_batch_frac)


def _is_contiguous(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return True
    return bool(np.all(np.diff(arr) == 1))
