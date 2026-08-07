"""``ZetaLoader`` — the reader for ``zeta_q.h5``.

One class covers both read surfaces (they were previously split across a
``ZetaReader``/``ZetaLoader`` pair with duplicated header/lifecycle code;
merged 2026-07-09):

* **Slab API** (the production V_q reader of record):
  :meth:`read_zeta_G_slab` — an explicit
  ``(q_offset, q_count, mu_offset, mu_count)`` window; a ``mu_count``
  past the on-disk extent comes back zero-filled (SlabIO's business,
  decisions.md 2026-08-04).
* **Load API** (WfnLoader-shaped, test bench + future consumers):
  :meth:`load` — symbolic ``q='ibz' | seq[int]`` ranges and μ slices.

Header surface: eager :class:`MfHeader` attributes (``nspin``,
``kgrid``, ``fft_grid``, ``sym_matrices``, …) and :class:`IsdfHeader`
attributes (``vertex_mu_L``, ``r_mu_fft_idx``, ``n_rmu``,
``zeta_layout``, …) — same names ``WFNReader`` exposes, so a
``ZetaLoader`` drops in wherever a ``WFNReader`` was used for header
information.

ONE DATA LAYOUT, SINCE 2026-08-07.  G-flat files store ``zeta_q_G``
shape ``(n_q, n_rmu_padded, ngkmax)`` (WFN.h5 ``coeffs`` style, per-q
sphere from ``isdf_header/gvec_components``), and that is the only
layout any DATA method reads.  The legacy r-space surface — the
``zeta_q`` ``(n_q, n_rtot, n_rmu)`` slab read, the FFT + sphere-gather
disk→G pipeline, and the IBZ→full-BZ ζ(r) symmetry unfold — was deleted
with the extraction: ``fit_zeta_to_h5`` hardcodes
``zeta_layout='G_flat'`` (``gw/isdf_fitting.py``), so no writer in the
tree has emitted ``r_space`` since the G-flat migration and nothing
outside this module ever called those paths.  __init__ STILL OPENS
r-space files, because the header surface is layout-independent and
several callers legitimately want only that; it is the data methods that
refuse, each naming the removal and the refit.  IBZ-only q-axes are
detected from the disk shape (``q_layout``) and full-BZ callers unfold
POST-V_q via :func:`common.symmetry_maps.unfold_v_q`, which is what
production has always done.

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
attribute works, and :attr:`slab_io` / :meth:`load` /
:meth:`read_zeta_G_slab` refuse, naming the missing mesh.  Passing a
mesh is UNCHANGED in every respect (eager SlabIO open, same collective,
same refusal when the FFI is absent).

THE HOST-TREE IMPORTS ARE LAZY, AND THAT IS THE WAVE-1B SEAM.  This
module's only module-scope third-party imports are h5py, numpy and jax:
``import zeta_loader`` is clean on a machine with no LORRAX checkout at
all.  The four things it does need from the host tree —
``file_io.mf_header``, ``file_io.isdf_header``, ``file_io.slab_io`` and
``common.gvec_fft_box`` — are imported at CALL time by the four helpers
below, each of which refuses by naming the missing module and the
surface that still works without it.  Those four are not a permanent
dependency: ``slab_io`` and the header binders are scheduled for
extraction as services of their own (wave 1b), at which point the lazy
imports become declared package dependencies and these helpers go.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import h5py as h5
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

if TYPE_CHECKING:                       # pragma: no cover — typing only
    from file_io.slab_io import SlabIO


__all__ = ["ZetaLoader"]


QSpec = Sequence[int] | Literal["ibz", "full_bz"]


# ---------------------------------------------------------------------------
# The host-tree seam.  Four call-time imports, four named refusals.
# ---------------------------------------------------------------------------

def _host_tree_refusal(module: str, names: str, needed_for: str) -> str:
    """The one refusal sentence, so the four helpers cannot drift apart."""
    return (
        f"zeta_loader needs {names} from {module!r}, and that module is not "
        f"importable here.  {module!r} is a LORRAX HOST-TREE module, not a "
        f"dependency of this package: the data path ({needed_for}) reads its "
        f"metadata and its bytes through the host tree until slab_io and the "
        f"header binders are extracted as services of their own (wave 1b), "
        f"and this import is that seam.  Put <lorrax>/src on sys.path — "
        f"ffi._services.ensure_on_path() is what the monorepo does — or use "
        f"the standalone surface (probe_zeta_file / write_g0_mu), which is "
        f"pure h5py+numpy and needs none of this.")


def _mf_header_binders():
    """``(bind_mf_attrs, read_mf_header_from_file)`` — the mf_header surface."""
    try:
        from file_io.mf_header import bind_mf_attrs, read_mf_header_from_file
    except ImportError as exc:                                  # noqa: BLE001
        raise ImportError(_host_tree_refusal(
            "file_io.mf_header", "bind_mf_attrs / read_mf_header_from_file",
            "every ZetaLoader open, header-only included")) from exc
    return bind_mf_attrs, read_mf_header_from_file


def _isdf_header_binders():
    """``(bind_isdf_attrs, read_isdf_header_from_file)`` — the ζ metadata."""
    try:
        from file_io.isdf_header import (
            bind_isdf_attrs, read_isdf_header_from_file)
    except ImportError as exc:                                  # noqa: BLE001
        raise ImportError(_host_tree_refusal(
            "file_io.isdf_header",
            "bind_isdf_attrs / read_isdf_header_from_file",
            "every ZetaLoader open, header-only included")) from exc
    return bind_isdf_attrs, read_isdf_header_from_file


def _slab_io_class():
    """The :class:`SlabIO` transport — needed only when a mesh is passed."""
    try:
        from file_io.slab_io import SlabIO
    except ImportError as exc:                                  # noqa: BLE001
        raise ImportError(_host_tree_refusal(
            "file_io.slab_io", "SlabIO",
            "every DATA read; a mesh=None loader never reaches it")) from exc
    return SlabIO


def _gvec_fft_box_helpers():
    """``(fft_box_pad_sentinel, pad_gvecs_to_sentinel)`` — the G-list pad."""
    try:
        from common.gvec_fft_box import (
            fft_box_pad_sentinel, pad_gvecs_to_sentinel)
    except ImportError as exc:                                  # noqa: BLE001
        raise ImportError(_host_tree_refusal(
            "common.gvec_fft_box",
            "fft_box_pad_sentinel / pad_gvecs_to_sentinel",
            "gvecs(), the validated per-q G-list accessor")) from exc
    return fft_box_pad_sentinel, pad_gvecs_to_sentinel


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

        bind_mf_attrs, read_mf_header_from_file = _mf_header_binders()
        bind_isdf_attrs, read_isdf_header_from_file = _isdf_header_binders()

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
        self._slab_io: "SlabIO | None" = (
            None if mesh is None
            else _slab_io_class()(self._path, mode=mode, mesh=mesh))
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

    def _refuse_unless_g_flat(self, what: str) -> None:
        """Refuse a DATA read on anything but a G-flat file.

        Called by every data method, never by the header surface: opening
        an r-space ζ for its mf_header/isdf_header attributes stays legal
        and is what several callers do.
        """
        if self.zeta_layout == 'G_flat':
            return
        raise ValueError(
            f"{self._path} has zeta_layout={self.zeta_layout!r}, and "
            f"ZetaLoader.{what} reads G-flat ζ only.  The r-space DATA "
            f"surface (read_zeta_r_slab, the FFT + sphere-gather disk→G "
            f"pipeline, and the IBZ→full-BZ ζ(r) unfold) was REMOVED on "
            f"2026-08-07: gw.isdf_fitting.fit_zeta_to_h5 hardcodes "
            f"zeta_layout='G_flat', so nothing in the tree has written an "
            f"r-space ζ since the G-flat migration.  Refit this file with "
            f"the G-flat writer.  The HEADER surface of this loader still "
            f"works on r-space files — it is only the ζ block that has no "
            f"reader.")

    @property
    def slab_io(self) -> "SlabIO":
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
    # G-vector accessors — the SAME surface WfnLoader exposes
    # ------------------------------------------------------------------
    def gvecs(self, *, q: QSpec = 'ibz') -> np.ndarray:
        """Return ``(n_q, ngkmax, 3)`` int32 — the per-q G-list, padded.

        Deliberately shaped, named and padded exactly like
        :meth:`file_io.wfn_loader.WfnLoader.gvecs`.  The two files store
        the G-axis differently — WFN.h5 flattens a RAGGED axis and needs
        ``kpt_starts``, ``zeta_q.h5`` is already ``ngkmax``-rectangular —
        but that difference belongs to the READ, not to what a consumer
        holds afterwards.  Both end here, in one representation built by
        :func:`common.gvec_fft_box.pad_gvecs_to_sentinel`.

        On-disk, ``isdf_header/gvec_components`` is ``(n_q, 3, ngkmax)``
        (the WFN.h5 ``(3, ng)`` component order with a leading q axis);
        this transposes it and re-runs the shared validation, so a
        corrupt or hand-edited components table — one whose pad rows are
        not the sentinel, or whose sphere reaches the FFT box corner —
        is caught at READ time rather than trusted.

        Raises for ``zeta_layout == 'r_space'``: those files carry no
        per-q sphere, and since 2026-08-07 they have no ζ reader either
        (see :meth:`_refuse_unless_g_flat`).
        """
        if self.gvec_components is None:
            raise ValueError(
                f"{self._path} has zeta_layout={self.zeta_layout!r} and no "
                f"isdf_header/gvec_components, so there is no per-q G-list "
                f"to return.  Only G-flat ζ carries one; refit this file "
                f"with gw.isdf_fitting's G-flat writer.")
        fft_box_pad_sentinel, pad_gvecs_to_sentinel = _gvec_fft_box_helpers()
        rows, _unfold = self._resolve_q(q)
        comps = np.asarray(self.gvec_components, dtype=np.int32)[rows]
        grid = tuple(int(s) for s in self.fft_grid)
        ngk = self.ngk_valid(q=q)
        src = np.ascontiguousarray(comps.transpose(0, 2, 1))

        # The components table means nothing without the grid it was
        # built on, and that grid is stored SEPARATELY (mf_header/gspace/
        # FFTgrid).  Nothing has ever checked they agree.  They do agree
        # iff the on-disk pad rows are THIS grid's sentinel, so check
        # exactly that — otherwise ``pad_gvecs_to_sentinel`` would
        # silently rewrite the pad rows and hide the disagreement.
        sentinel, _ = fft_box_pad_sentinel(grid)
        for j in range(src.shape[0]):
            n = int(ngk[j])
            if n >= src.shape[1]:
                continue
            if not np.array_equal(
                    src[j, n:], np.broadcast_to(sentinel, (src.shape[1] - n, 3))):
                raise ValueError(
                    f"{self._path}: isdf_header/gvec_components row q={j} has "
                    f"pad slots [{n}:{src.shape[1]}] that are not the pad "
                    f"sentinel {tuple(int(v) for v in sentinel)} for "
                    f"mf_header FFTgrid {grid}.  The components table and the "
                    f"header's FFT grid disagree, so every G in this file is "
                    f"being read on the wrong grid.  Refit the ζ, or fix the "
                    f"writer that produced them.")

        gvecs, _ = pad_gvecs_to_sentinel(src, grid, ngk_valid=ngk)
        return gvecs

    def ngk_valid(self, *, q: QSpec = 'ibz') -> np.ndarray:
        """Per-q logical sphere size (without pad).  Host numpy int32.

        Twin of :meth:`file_io.wfn_loader.WfnLoader.ngk_valid`.
        """
        if self.ngk_per_q is None:
            raise ValueError(
                f"{self._path} has zeta_layout={self.zeta_layout!r} and no "
                f"isdf_header/ngk, so it has no per-q logical extent.")
        rows, _unfold = self._resolve_q(q)
        return np.asarray(self.ngk_per_q, dtype=np.int32)[rows]

    # ------------------------------------------------------------------
    # Slab API (production V_q reader of record)
    # ------------------------------------------------------------------
    def read_zeta_G_slab(
        self,
        *,
        q_offset: int,
        q_count: int,
        mu_offset: int,
        mu_count: int,
        mesh: Mesh | None = None,
    ) -> jax.Array:
        """Read ζ in G-flat layout.  THE production V_q read.

        One ``read_slab`` of the ``(Q, μ, ngkmax)`` window — the per-q
        FFT-box phase is already baked into the on-disk tensor by the
        writer, so there is no post-processing here at all.  A
        ``mu_count`` past the on-disk μ extent comes back zero-filled
        (SlabIO's business, decisions.md 2026-08-04), so a caller that
        pads μ to a mesh product states the extent it wants to consume
        and passes nothing else.

        Returns
        -------
        zeta_G : jax.Array
            Shape ``(q_count, μ_per_rank, ngkmax)`` complex128,
            sharded ``P(None, ('x','y'), None)``.

        Parameters
        ----------
        q_offset, q_count : int
            Slab range along the on-disk q axis.  Caller-managed
            indexing — under IBZ-only layouts the offset is the IBZ
            index.
        mu_offset, mu_count : int
            Slab range along the μ axis (centroid axis).
        mesh : Mesh | None
            Override of the loader's stored mesh.

        TWO ARGUMENTS ARE GONE as of 2026-08-07 (design D3).
        ``qvec_batch_frac`` was ignored on the only live path and
        ``v_q_g_flat.py`` passed a ``(Q, 3)`` zeros array purely to
        satisfy the signature; ``sphere_idx``'s only legal value was
        ``None``, because the on-disk G axis is a PER-Q sphere whose
        positions vary with q (``isdf_header/gvec_components``) and a
        single shared index would pick the same disk position for every
        q, which is per-q wrong.  The per-q → shared-sphere scatter
        belongs in the V_q wrapper and is not implemented here.
        """
        self._refuse_unless_g_flat("read_zeta_G_slab")
        if mesh is None:
            mesh = self._mesh

        return self.slab_io.read_slab(
            self._zeta_dataset_name,
            shape=(int(q_count), int(mu_count), int(self.n_G_sph_disk)),
            dtype=np.complex128,
            offset=(int(q_offset), int(mu_offset), 0),
            mesh=mesh,
            partition_spec=P(None, ('x', 'y'), None),
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
    ) -> jax.Array:
        """Read a ζ window.  ``(Q, μ, ngkmax)`` complex128, G-flat only.

        Parameters
        ----------
        q
            ``'ibz'``      — every row on disk (works for both IBZ and
                              full-BZ on-disk layouts).
            ``'full_bz'``  — refuses on an IBZ-on-disk file (see below);
                              on a full-BZ-on-disk file it is ``'ibz'``.
            ``Sequence[int]`` — explicit row indices into the on-disk
                              q-axis.  For IBZ-on-disk layouts these are
                              IBZ-row indices.
        mu
            ``(mu_lo, mu_hi)`` half-open range, ``slice``, or ``None``
            (full μ axis).
        sharding
            Output partition spec; defaults to ``P(None, ('x','y'),
            None)``, the shape the V_q consumer wants.

        THREE ARGUMENTS ARE GONE as of 2026-08-07 (design D3).
        ``layout`` had one legal value once the r-space data surface was
        removed, and ``qvec_frac``/``sphere_idx`` only ever fed the
        deleted disk→G pipeline.  The one production caller
        (``bse/vq_interp.py``) passed ``layout='G_flat'`` explicitly and
        the other two never.
        """
        # ORDER: the two refusals that are facts about the FILE AND THE
        # REQUEST come before the one that is a fact about the STACK, so
        # a caller on a machine with no phdf5 FFI still gets told that it
        # asked for something this reader does not do — rather than being
        # told about a transport it was never going to reach.
        self._refuse_unless_g_flat("load")

        # --- q axis ---------------------------------------------------
        q_indices, need_unfold = self._resolve_q(q)
        if need_unfold:
            raise NotImplementedError(
                "ZetaLoader.load(q='full_bz'): IBZ→full unfold for "
                "G-flat ζ_q is not wired (it needs rotation of the "
                "per-q components table + the R·V·Rᵀ transverse path).  "
                "Use q='ibz' and unfold POST-V_q, via "
                "common.symmetry_maps.unfold_v_q — which is what "
                "production does, because V_q is bilinear in ζ and the "
                "unfold is a centroid double-permute there.")

        _ = self.slab_io  # refuse now, naming header-only vs closed

        # --- μ axis ---------------------------------------------------
        mu_lo, mu_hi = self._resolve_mu(mu)
        mu_count = mu_hi - mu_lo

        if sharding is None:
            sharding = P(None, ('x', 'y'), None)

        return self._read_g_flat_disk(
            q_indices=q_indices,
            mu_lo=mu_lo, mu_count=mu_count,
            partition_spec=sharding,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_q(self, q: QSpec) -> tuple[np.ndarray, bool]:
        """Resolve ``q`` into ``(disk_row_indices, need_full_bz_unfold)``.

        ``need_full_bz_unfold = True`` means the caller asked for
        ``q='full_bz'`` against an IBZ-on-disk file.  :meth:`load`
        refuses on it, naming the post-V_q unfold; the flag is kept
        rather than folded into the string test because ``'full_bz'``
        against a full-BZ-on-disk file is legal and means ``'ibz'``, and
        that distinction is exactly what this returns.
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


def _is_contiguous(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return True
    return bool(np.all(np.diff(arr) == 1))
