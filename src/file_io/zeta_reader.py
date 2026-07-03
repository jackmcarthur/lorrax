"""``ZetaReader`` — wfn_reader-shaped reader for ``zeta_q.h5``.

Surface modelled on :class:`WFNReader`:

* Eager :class:`MfHeader` attributes (``nspin``, ``nkpts``, ``kgrid``,
  ``fft_grid``, ``bvec``, ``sym_matrices``, ``translations``, …) — same
  names ``WFNReader`` exposes, so callers can drop a ``ZetaReader`` in
  wherever they used a ``WFNReader`` for header information.
* Eager :class:`IsdfHeader` attributes (``density``, ``vertex_mu_L``,
  ``r_mu_fft_idx``, ``r_mu_crystal``, ``n_rmu``).
* Lazy ζ-data reads via two methods:

  - :meth:`read_zeta_r_slab` — legacy r-space read.  Returns a sharded
    JAX array with shape ``(Q, n_rtot, μ)``, layout
    ``P(None, None, ('x','y'))``.  Wraps the existing
    :class:`SlabIO.read_slab` contract.

  - :meth:`read_zeta_G_slab` — **G-flat read**: reads the same r-space
    slab, multiplies by the per-q phase ``exp(i q·r)``, runs the
    3-D FFT on the μ-sharded slab, and gathers onto the G-sphere.
    Returns ``(Q, μ/p_prod, n_G_sph)`` sharded
    ``P(None, ('x','y'), None)``.

The G-flat path is the new V_q kernel input — the FFT moves out of the
kernel and into the reader (eq. 3 of
``reports/zeta_ibz_2026-05-11/report.md``, §3.3).  Compute cost is
identical; the kernel just becomes a v(K)-multiply + einsum +
G=0-index lookup.

The reader transparently handles IBZ-only on-disk layouts (the writer's
default after C2): ``read_zeta_q_count`` reports the actual on-disk q
count.  Callers that loop over IBZ q's index directly into this axis;
callers that loop over full-BZ q's must do the centroid-double-permute
unfold post-V_q (see :func:`common.symmetry_maps.unfold_v_q`).

Lifecycle
---------
The reader owns a :class:`SlabIO` handle.  Use as a context manager (or
call :meth:`close` explicitly) — the underlying file is held open
across reads to amortise the open/close cost on the FFI backend (see
``isdf_fitting.py:1656`` for the historical motivation).
"""

from __future__ import annotations

from pathlib import Path
from functools import partial
from typing import Sequence

import h5py as h5
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .mf_header import bind_mf_attrs, read_mf_header
from .isdf_header import bind_isdf_attrs, read_isdf_header
from .slab_io import SlabIO


class ZetaReader:
    """Reader for ``zeta_q.h5`` produced by ``isdf_fitting.fit_zeta_to_h5``."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        path: str | Path,
        *,
        mesh: Mesh,
        backend=None,
        mode: str = "r",
    ):
        self._path = str(path)
        self._mesh = mesh

        # --- Headers ----------------------------------------------------
        mf = read_mf_header(self._path)
        isdf = read_isdf_header(self._path)
        self._mf = mf
        self._isdf = isdf

        # WFNReader-shape mf_header attributes (1:1 from MfHeader).
        bind_mf_attrs(self, mf)

        # isdf_header attributes.
        bind_isdf_attrs(self, isdf)

        # ---- Capture on-disk q-axis size (IBZ vs full-BZ) ---------------
        # We poke the file directly here rather than going through SlabIO —
        # SlabIO is a write-or-collective-read object, and we only want
        # the dataset shape metadata.  Reopened for the actual data
        # reads via the SlabIO handle below.
        # Dataset name depends on layout: legacy r-space files store
        # ``zeta_q`` shape (n_q, n_rtot, n_rmu); G-flat files store
        # ``zeta_q_G`` shape (n_q, n_rmu, n_G_sph).
        _ds_name = ('zeta_q_G' if self.zeta_layout == 'G_flat'
                    else 'zeta_q')
        with h5.File(self._path, "r") as f:
            self._zeta_disk_shape = tuple(int(x) for x in f[_ds_name].shape)
        self._zeta_dataset_name = _ds_name
        self.n_q_on_disk = self._zeta_disk_shape[0]
        if self.zeta_layout == 'G_flat':
            # G_flat layout: shape (n_q, n_rmu, n_G_sph).  The
            # ``n_rtot_disk`` attribute is kept for r-space callers;
            # we expose n_G_sph for G-flat consumers and leave
            # n_rtot_disk = n_rtot (computed from fft_grid) for
            # compatibility with code that probes the on-disk r-extent.
            nx, ny, nz = (int(s) for s in self.fft_grid)
            self.n_rtot_disk = nx * ny * nz
            self.n_rmu_disk = self._zeta_disk_shape[1]
            self.n_G_sph_disk = self._zeta_disk_shape[2]
        else:
            self.n_rtot_disk = self._zeta_disk_shape[1]
            self.n_rmu_disk = self._zeta_disk_shape[2]
            self.n_G_sph_disk = None

        # ---- SlabIO handle ---------------------------------------------
        self._slab_io = SlabIO(self._path, mode=mode, mesh=mesh,
                                backend=backend)

    def close(self):
        if self._slab_io is not None:
            self._slab_io.close()
            self._slab_io = None

    def __enter__(self) -> "ZetaReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def slab_io(self) -> SlabIO:
        """Underlying SlabIO handle.  Exposed for callers that still need
        the raw ``read_slab('zeta_q', ...)`` contract while the rest of
        the V_q stack is migrating to G-flat."""
        return self._slab_io

    @property
    def n_rtot(self) -> int:
        nx, ny, nz = (int(s) for s in self.fft_grid)
        return nx * ny * nz

    # ------------------------------------------------------------------
    # Read (r-space, legacy)
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
        valid_mu: int | None = None,
    ) -> jax.Array:
        """Read an r-space ζ slab.

        Returns ``(q_count, n_rtot, mu_count)`` complex128, sharded
        per ``partition_spec``.  Trailing μ pad slots are zero-filled
        when ``mu_offset + mu_count`` exceeds the logical extent and
        ``valid_mu`` is set.
        """
        if mesh is None:
            mesh = self._mesh
        n_rtot = self.n_rtot_disk
        valid_count = (mu_count if valid_mu is None else valid_mu)
        return self._slab_io.read_slab(
            'zeta_q',
            shape=(int(q_count), int(n_rtot), int(mu_count)),
            valid_shape=(int(q_count), int(n_rtot), int(valid_count)),
            dtype=np.complex128,
            offset=(int(q_offset), 0, int(mu_offset)),
            mesh=mesh,
            partition_spec=partition_spec,
        )

    # ------------------------------------------------------------------
    # Read (G-flat — the new V_q kernel input)
    # ------------------------------------------------------------------
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
        valid_mu: int | None = None,
    ) -> jax.Array:
        """Read ζ in G-flat layout.

        Pipeline:
        1. Read r-space slab ``(Q, n_rtot, μ)`` via :class:`SlabIO`
           (same async-dispatch semantics as today).
        2. Transpose to ``(Q, μ, n_rtot)``; reshape to
           ``(Q, μ, nx, ny, nz)``; apply separable per-q Bloch phase
           ``exp(-2πi q·r)`` via :func:`common.wfn_transforms.apply_bloch_phase`.
        3. 3D FFT over the spatial axes (μ-sharded).
        4. Sphere gather to ``(Q, μ/p_prod, n_G_sph)``.

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
            the per-q FFT-box phase separably.  Replaces the legacy
            ``(Q, 1, nx, ny, nz)`` ``phase_batch`` argument — the 4D
            phase is gone; scratch memory drops from ``Q·nx·ny·nz``
            to ``Q·(nx+ny+nz)``.
        sphere_idx : jax.Array | None
            Flat-FFT indices that define the G-sphere (or None to
            keep the full FFT box).
        mesh : Mesh | None
            Override of the reader's stored mesh.
        valid_mu : int | None
            Logical μ extent if smaller than ``mu_count`` (pad-aware
            reads).
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
            valid_count = (mu_count if valid_mu is None else int(valid_mu))
            zeta_g_disk = self._slab_io.read_slab(
                self._zeta_dataset_name,
                shape=(int(q_count), int(mu_count), n_G_sph_disk),
                valid_shape=(int(q_count), int(valid_count), n_G_sph_disk),
                dtype=np.complex128,
                offset=(int(q_offset), int(mu_offset), 0),
                mesh=mesh,
                partition_spec=P(None, ('x', 'y'), None),
            )
            if sphere_jx is not None and n_G_sph != n_G_sph_disk:
                raise NotImplementedError(
                    "ZetaReader.read_zeta_G_slab: on-disk per-q sphere "
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
            mesh=mesh, valid_mu=valid_mu,
        )

        return _do_disk_to_G(
            zeta_disk, qvec_batch_frac,
            mesh_xy=mesh, fft_shape=(nx, ny, nz),
            n_G_sph=n_G_sph, sphere_idx=sphere_jx,
        )


# ---------------------------------------------------------------------------
# Internal: r-space slab → G-flat — module-level so the jitted helpers
# cache across calls and don't recompile per-ZetaReader instance.
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


__all__ = ['ZetaReader']
