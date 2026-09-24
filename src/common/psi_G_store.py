"""Host-resident ψ(G-flat) staging for reusable ψ(r)-chunk sources.

The one-time cache builder consumes ψ(G) one band chunk at a time, and the
ISDF fit later slices ψ(r) by band chunk and r chunk.  Holding ψ in full FFT-box
representation on host costs ``nb · ns · nx · ny · nz`` complex128 per
rank — for CrI3-class systems that's tens of GB.  Holding ψ in G-flat
representation instead costs ``nb · ns · ngkmax`` per rank, which is
~6-11% of the box for typical GW grids.

This rewrite (P4c) replaces the legacy g_box host-cache with a G-flat
staging pipeline:

* :class:`PsiGStore` stores per-rank tiles of shape
  ``(nk, nb_local, ns, ngkmax)`` instead of ``(nk, nb_local, ns, nx,
  ny, nz)``.
* :meth:`PsiGStore.read_local_band_chunk` returns one bc's per-rank
  band slab via ``io_callback``, padded to ``(nk, _bpd_max, ns,
  ngkmax)`` so the enclosing ``lax.scan`` body sees a static return
  shape.  :func:`isdf.core.build_psi_r_cache_sm` iterates band chunks via
  ``lax.scan`` inside its ``shard_map`` body, pulling one bc per iteration
  via the slicer.  The resulting ψ(r) cache is band-flat-sharded over the
  full mesh.

The store populates once and can either feed the one-time ψ(r) cache build or
serve repeated r chunks through :meth:`PsiGStore.iter_rchunk_bandwise`.  Its
host footprint is one band shard per process-addressable mesh cell; the
process total is the exact sum of those local tiles (one tile in the usual
one-rank-per-GPU launch).

The reader adapters (legacy h5py vs phdf5) collapse to a single
:class:`wfn_loader.WfnLoader` whose ``backend='auto'`` picks the
right path.
"""
from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from common.shard_map import shard_map
from common.wfn_layout import band_sphere_spec
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from runtime.padding import spec_divisor


def _zero_user_band_pad_in_shard(
    shard_data: np.ndarray,
    *,
    bc_range: tuple[int, int],
    shard_band_slice: slice,
    user_band_stop: int,
) -> np.ndarray:
    """Zero locally-owned padded bands inside one ψ(G-flat) shard.

    ``Meta.b_id_4`` may be larger than the user's requested ``nband`` so
    the band axis divides the device mesh.  The loader can still return
    real DFT coefficients for those padded slots when the WFN file has
    enough bands.  The centroid loader zeros them after extraction; this
    helper applies the same contract to the host ψ(G-flat) cache used
    by the r-chunk ζ fit.
    """
    b0, _ = (int(bc_range[0]), int(bc_range[1]))
    s0 = 0 if shard_band_slice.start is None else int(shard_band_slice.start)
    s1 = shard_data.shape[1] if shard_band_slice.stop is None else int(shard_band_slice.stop)
    step = 1 if shard_band_slice.step is None else int(shard_band_slice.step)
    if step != 1:
        raise ValueError(
            f"ψ(G-flat) shard band slice must be contiguous; got {shard_band_slice!r}")

    local_global_bands = b0 + np.arange(s0, s1, dtype=np.int64)
    pad_mask = local_global_bands >= int(user_band_stop)
    if not np.any(pad_mask):
        return shard_data

    out = np.array(shard_data, copy=True)
    out[:, pad_mask, :, :] = 0.0
    return out


def _mesh_device_coords(mesh: Mesh) -> dict:
    """Map only this process's addressable devices to global mesh cells."""
    global_coords = {
        id(dev): tuple(int(i) for i in idx)
        for idx, dev in np.ndenumerate(np.asarray(mesh.devices))
    }
    coords = {}
    for dev in mesh.local_devices:
        dev_id = id(dev)
        if dev_id not in global_coords:
            raise RuntimeError(
                "PsiGStore: a Mesh.local_devices entry is absent from "
                "Mesh.devices")
        coords[dev_id] = global_coords[dev_id]
    if not coords:
        raise RuntimeError(
            "PsiGStore: this process owns no addressable device in the mesh")
    return coords


def assert_band_chunks_divisible(band_chunk_ranges, world_size: int) -> None:
    """Refuse a band chunk whose width the per-rank tile would floor-divide.

    THE FLOOR DIVISION IS THE DEFECT SITE, so this is the refusal site.
    :class:`PsiGStore` splits each chunk's bands across all ``P`` ranks as
    ``(b_hi - b_lo) // P``; a width that is not a multiple of ``P`` loses
    ``width % P`` bands right there, in the store's own band accounting,
    and every consumer downstream then works on a store that is short
    those bands with the right shape, the right dtype and no other
    symptom.

    MEASURED (JID 57187694,
    ``reports/zeta_residue_2026-08-17/evidence/baseline_p4.log``): on the
    80 Ry scalar-Si deck the P=4 baseline's logical 50-band window split
    16+16+16+2; the last chunk gives ``bpd = 2 // 4 = 0`` and its two
    bands contributed nothing to ``z_q`` — at rc=0.

    ``gw.isdf_fitting`` fixed the PRODUCTION path on 2026-08-17 by padding
    the transport range up to a ``P`` multiple (``_bfe_transport``), and
    ``isdf.core.z_q_from_psi_sm`` carries an equivalent consumer-side
    check.  Neither makes this one redundant: a guard living in ONE
    consumer is a guard the next consumer does not have, and the padding
    lives in ONE producer.  This function is on the object that performs
    the division, which is the only place every caller must pass through.

    ``ValueError`` and not ``assert``: the fix is a user input key
    (``band_chunk_size``), and an assert vanishes under ``python -O``,
    re-arming exactly the silent band-dropping it guards.
    """
    p = int(world_size)
    if p <= 0:
        raise ValueError(f"world_size must be positive, got {world_size!r}")
    from runtime.padding import authenticate_padded_axis
    for i, (b_lo, b_hi) in enumerate(band_chunk_ranges):
        authenticate_padded_axis(
            int(b_hi) - int(b_lo), int(b_hi) - int(b_lo), p,
            name=f"PsiGStore band chunk {i} [{int(b_lo)}, {int(b_hi)})")


class PsiGStore:
    """Host-resident ψ(G-flat) staging and reusable r-chunk source.

    Per locally-addressable mesh cell ``(x, y)`` owns one contiguous
    host tile of shape ``(nk, nb_local, ns, ngkmax)``.  The band axis
    inside each tile is ordered by band-chunk (bc) — block 0 holds
    bc 0's local bands, block 1 holds bc 1's local bands, and so on.
    For CrI3-scale, the new shape is ~14× smaller than the legacy
    g_box ``(nk, nb_local, ns, nx, ny, nz)`` shape.

    :meth:`read_local_band_chunk` is the public per-iter host-tile slicer used
    by the ``io_callback`` inside the cache builder's ``lax.scan`` body.
    It returns one bc's per-rank slab padded to
    ``(nk, _bpd_max, ns, ngkmax)`` so the scan body sees a static
    output shape every iteration.
    """

    def __init__(
        self,
        *,
        loader,
        mesh_xy: Mesh,
        band_chunk_ranges: tuple[tuple[int, int], ...],
        meta,
        bispinor: bool = False,
        bispinor_lift: str = "raw",
        band_pad_to: int | None = None,
        k_domain: str = "full_bz",
    ):
        self.loader = loader
        self.mesh = mesh_xy
        self.band_chunk_ranges = tuple(tuple(bc) for bc in band_chunk_ranges)
        self.meta = meta
        self.bispinor = bool(bispinor)
        self.bispinor_lift = str(bispinor_lift)
        # Which k rows the store holds: the unfolded full BZ (the production
        # ζ-fit source) or the WFN file's own raw parent rows.  A parent-k
        # consumer wants the IBZ rows themselves, not their images, and the
        # loader's k table, box index and coefficient rows must all come from
        # the same domain or k and G fall out of gauge.
        self.k_domain = str(k_domain)
        if self.k_domain not in ("full_bz", "ibz"):
            raise ValueError(
                "PsiGStore: k_domain must be 'full_bz' or 'ibz'; got "
                f"{k_domain!r}")

        nk = (int(loader.nkpts) if self.k_domain == "ibz"
              else int(meta.nk_tot))
        ns = int(meta.nspinor)
        ngkmax = int(loader.ngkmax)
        p = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)

        logical_widths = tuple(
            int(b_hi) - int(b_lo) for b_lo, b_hi in self.band_chunk_ranges)
        if band_pad_to is not None:
            band_pad_to = int(band_pad_to)
            if band_pad_to <= 0:
                raise ValueError(
                    f"PsiGStore: band_pad_to must be positive, got "
                    f"{band_pad_to}")
            too_wide = [
                (bc, width) for bc, width in zip(
                    self.band_chunk_ranges, logical_widths)
                if width > band_pad_to
            ]
            if too_wide:
                raise ValueError(
                    "PsiGStore: band_pad_to is smaller than a logical band "
                    f"chunk: band_pad_to={band_pad_to}, chunks={too_wide}")
            transport_ranges = tuple(
                (int(b_lo), int(b_lo) + band_pad_to)
                for b_lo, _ in self.band_chunk_ranges)
        else:
            transport_ranges = self.band_chunk_ranges
        self._band_pad_to = band_pad_to

        # Per-bc local band count: bands_per_device for ONE bc.  Used to
        # compute the per-rank tile's band-axis offsets (bc-stacked
        # ordering); the per-rank tile's full band axis is contiguous
        # across all bcs and lives at ``self._per_rank_shape[1]``.
        # ``band_pad_to`` is a transport carrier: a short logical final chunk
        # remains the range the caller sees, while ``load_psi_gflat_padded``
        # supplies exact-zero rows so the store can shard a uniform width.
        assert_band_chunks_divisible(transport_ranges, p)
        bpd_per_bc = [
            (int(b_hi) - int(b_lo)) // p
            for b_lo, b_hi in transport_ranges
        ]
        self._bpd_per_bc = tuple(bpd_per_bc)
        # Padded uniform per-bc local band count — used by
        # ``read_local_band_chunk`` so an ``io_callback`` inside a
        # ``lax.scan`` body sees a static return shape regardless of
        # which bc the traced index resolves to.  Round 6 Phase 2
        # restoration of the field originally added in commit
        # ``cdd0fba`` (deleted in ``5cadd4b`` when the flat-axis path
        # took over).  ``io_callback`` REQUIRES static ``out_sds`` at
        # trace time; ``_bpd_max`` is the closure-static value the
        # caller closes into ``ShapeDtypeStruct``.
        self._bpd_max = max(bpd_per_bc) if bpd_per_bc else 0
        offsets = [0]
        for bpd in bpd_per_bc:
            offsets.append(offsets[-1] + bpd)
        self._bc_band_offsets = tuple(offsets)
        self._nb_local = offsets[-1]
        self._per_rank_shape = (nk, self._nb_local, ns, ngkmax)

        self._dtype = jnp.complex128
        self._coords = _mesh_device_coords(mesh_xy)
        # host_tiles[(x, y)] = one contiguous numpy array of shape
        # _per_rank_shape, populated once below.
        self._host_tiles: dict = {}

        # Cache the box index (g_index) and Bloch-phase ingredients on
        # device once — they're shared across every cache-builder callback.
        self._g_index_dev: jax.Array | None = None
        self._kvecs_frac_dev: jax.Array | None = None
        self._rchunk_kernel_cache: dict[int, object] = {}
        self._closed = False

        self._populate_from_loader()
        expected_host_bytes = (
            len(self._coords) * self._per_rank_shape_bytes())
        if self.host_cache_bytes != expected_host_bytes:
            raise RuntimeError(
                "PsiGStore: process-local host cache allocation drifted "
                f"from its exact bound: allocated={self.host_cache_bytes} "
                f"bytes, expected={expected_host_bytes} bytes for "
                f"{len(self._coords)} addressable mesh cells")
        if jax.process_index() == 0:
            host_gb = self.host_cache_bytes / 1e9
            print(f"  ψ(G-flat) host cache: {host_gb:.2f} GB/process resident")

    # ---------------------------------------------------------------------
    # Population — pulls from the WfnLoader, scatters into per-(x,y) tiles.
    # ---------------------------------------------------------------------
    def _populate_from_loader(self) -> None:
        """One ``loader.load(bands=bc)`` per band-chunk, then split the
        returned sharded jax.Array into per-(x, y) tiles on host.

        ``loader.load`` is a collective on the FFI backend or a
        broadcast-then-device-put on the eager backend; either way each
        rank's local shard of the (band-sharded) output is what we
        actually need to copy into the host tile.
        """
        # Allocate tiles on first population.
        for (x, y) in self._coords.values():
            if (x, y) not in self._host_tiles:
                self._host_tiles[(x, y)] = np.empty(
                    self._per_rank_shape, dtype=np.complex128)

        # NOTE: this read is deliberately SYNCHRONOUS.  A prefetching
        # async wfn reader (a worker thread issuing ``loader.load(bc+1)``
        # against bc[i]'s shard_to_host copy) was implemented, measured,
        # and deleted 2026-07-25: at MoS2 3×3 scale xprof shows
        # H2D/compute overlap_frac = 0.000 even with depth-2 prefetch —
        # XLA's stream scheduler does not pipeline our H2D against
        # compute.  If the async-reader story comes back at larger scale
        # (CrI3), rebuild it on ``common.async_io.AsyncDispatcher``, which
        # is still here and drives the SlabIO write side.
        from common import timing
        from common.wfn_transforms import load_psi_gflat_padded
        sharding_spec = band_sphere_spec()
        for bc_idx, bc_range in enumerate(self.band_chunk_ranges):
            bc_start, bc_end = int(bc_range[0]), int(bc_range[1])
            b_lo = self._bc_band_offsets[bc_idx]
            b_hi = self._bc_band_offsets[bc_idx + 1]
            # Past-mnband zero-pad — the cap-at-file-nbands + zero-pad +
            # reshard dance is single-sourced in
            # :func:`common.wfn_transforms.load_psi_gflat_padded` (same
            # contract as ``load_centroids_band_chunked``, commit
            # 2129fad).  ``None`` return = the entire bc range starts
            # at/past ``loader.nbands``: only band-pad rows the
            # ``_zero_user_band_pad_in_shard`` post-step would zero out
            # anyway, so skip the load entirely and zero-fill the tile span
            # directly.  This also covers a uniformly padded transport
            # carrier whose logical final chunk lies wholly beyond EOF.
            with timing.section("psi_G_store.populate.loader_load"):
                psi_G_bc = load_psi_gflat_padded(
                    self.loader, (bc_start, bc_end), mesh_xy=self.mesh,
                    bispinor=self.bispinor, k=self.k_domain,
                    pad_to=self._band_pad_to, sharding=sharding_spec,
                    bispinor_lift=self.bispinor_lift)
                if psi_G_bc is not None:
                    jax.block_until_ready(psi_G_bc)
            if psi_G_bc is None:
                if b_hi - b_lo > 0:
                    for (x, y) in self._coords.values():
                        self._host_tiles[(x, y)][:, b_lo:b_hi, :, :] = 0
                continue
            with timing.section("psi_G_store.populate.shard_to_host"):
                for shard in psi_G_bc.addressable_shards:
                    x, y = self._coords[id(shard.device)]
                    tile = self._host_tiles[(x, y)]
                    shard_band_slice = shard.index[1]
                    data = _zero_user_band_pad_in_shard(
                        np.asarray(shard.data),
                        bc_range=bc_range,
                        shard_band_slice=shard_band_slice,
                        user_band_stop=int(getattr(self.meta, "b_id_4_user", self.meta.b_id_4)),
                    )
                    tile[:, b_lo:b_hi, :, :] = data
            del psi_G_bc  # release device memory before next bc

        # Stage box_index + kvecs once on device; reused across every
        # fetch.  These don't depend on the band range or r-chunk.
        if self._g_index_dev is None:
            # ``WfnLoader.box_index_dev`` deduplicates the device-resident
            # ``(nk, nx, ny, nz) int32`` g_index across every
            # ``psi_G_store`` instance that shares the same loader+mesh.
            # Without this dedupe, every ``fit_zeta_to_h5`` channel
            # (charge + 3 transverse on bispinor) device_put'd a fresh
            # REPLICATED buffer (0.16 GB/rank each), accumulating to
            # ~1.3 GB/rank wasted by V_q time (agent_h §3 Finding 3).
            self._g_index_dev = self.loader.box_index_dev(
                k=self.k_domain, mesh=self.mesh)
            # k and G are one gauge contract owned by WfnLoader. Rebuilding
            # k from integer grid labels can pick a different reciprocal-
            # lattice image than ``box_index``'s G table (notably on an
            # identity-only WFN whose stored full grid is centered).
            kvecs_frac = self.loader.kvecs(k=self.k_domain)
            # Process-local placement — see
            # ``common.collectives.device_put_process_local``: on a
            # multi-process mesh ``jax.device_put(numpy, sharding)``
            # fires JAX's hidden ``assert_equal`` all-gather.
            from common.collectives import device_put_process_local
            self._kvecs_frac_dev = device_put_process_local(
                kvecs_frac,
                NamedSharding(self.mesh, P(None, None)))

    def _clear_tiles(self) -> None:
        self._host_tiles.clear()

    def release_host_tiles(self) -> None:
        """Release coefficient tiles after their callbacks have drained.

        The device-resident box index and k vectors remain valid for cached
        r-space consumers.  :meth:`close` is the sole full teardown.
        """
        self._rchunk_kernel_cache.clear()
        self._clear_tiles()

    # ---------------------------------------------------------------------
    # Per-rank host-tile slice for one bc, padded to a static shape.
    # ---------------------------------------------------------------------
    # Round 6 Phase 2 restoration of the helper originally added in
    # commit ``cdd0fba`` and removed in ``5cadd4b`` when the (now-buggy)
    # flat-axis ``psi_G_device_full`` path took over.  The production consumer
    # is the io_callback inside ``build_psi_r_cache_sm``'s ``lax.scan`` body;
    # ``z_q_from_psi_sm`` retains a compatibility-only direct consumer.
    #
    # Static-shape contract: ``io_callback`` requires its ``out_sds`` to
    # be static at trace time, AND ``lax.scan`` requires the body output
    # shape to be uniform across iters.  ``_bpd_max = max(bpd_per_bc)``
    # is closure-static at ``__init__``; short-final-bc bands are
    # zero-padded to the same shape every iter.  The downstream L/R
    # band-mask zeros out pad rows so they contribute mathematically zero
    # to the pair-density einsum.
    #
    # NOTE the ``np.zeros`` (NOT ``np.empty``) on the pad-row buffer:
    # the math-neutrality of pad rows depends on them being EXACTLY
    # zero.  ``np.empty`` would leave garbage that the L/R mask might
    # zero-out at the einsum but could still pollute IFFT precision.

    @property
    def local_band_chunk_shape(self) -> tuple[int, int, int, int]:
        """Per-device host callback shape ``(nk, b_local, ns, ngkmax)``.

        This is the only shape a consumer needs in order to declare an
        ``io_callback`` result.  Keeping it public prevents r-chunk sources
        from reaching into ``_bpd_max`` or ``_per_rank_shape``.
        """
        nk, _, ns, ngkmax = self._per_rank_shape
        return (int(nk), int(self._bpd_max), int(ns), int(ngkmax))

    @property
    def host_cache_bytes(self) -> int:
        """Exact bytes in process-addressable coefficient host tiles."""
        return sum(int(tile.nbytes) for tile in self._host_tiles.values())

    @property
    def band_chunk_carrier(self) -> int:
        """Uniform global band width yielded for every logical chunk."""
        p = spec_divisor(self.mesh, band_sphere_spec(), axis=1)
        return int(self._bpd_max) * int(p)

    def read_local_band_chunk(self, x_idx, y_idx, bc_idx) -> np.ndarray:
        """Return one local band-chunk carrier from this process's host tile.

        Parameters
        ----------
        x_idx, y_idx
            ``jax.lax.axis_index('x') / ('y')`` int32 scalars (resolved
            to Python ints inside the io_callback host fn).
        bc_idx
            Traced int32 scalar in ``[0, len(band_chunk_ranges))``.

        Returns
        -------
        np.ndarray
            Shape ``(nk, _bpd_max, ns, ngkmax)`` c128.  The first
            ``self._bpd_per_bc[bc]`` band rows hold the real bc data;
            the remaining ``_bpd_max - bpd_per_bc[bc]`` rows are zero
            (math-neutral when consumed under a band mask).

        Lifetime contract: host tiles must remain valid for the full
        duration of the enclosing kernel jit because ``io_callback`` fires
        asynchronously inside ``lax.scan``.  ``isdf_fitting.py`` blocks on
        the completed ψ(r) cache before closing this store.
        """
        if getattr(self, "_closed", False):
            raise RuntimeError(
                "PsiGStore.read_local_band_chunk: the store is closed")
        if not self._host_tiles:
            raise RuntimeError(
                "PsiGStore.read_local_band_chunk: host tiles were released")
        x, y, bc = int(x_idx), int(y_idx), int(bc_idx)
        if not 0 <= bc < len(self.band_chunk_ranges):
            raise ValueError(
                f"read_local_band_chunk: bc_idx={bc} not in "
                f"[0, {len(self.band_chunk_ranges)})")
        tile = self._host_tiles[(x, y)]
        b_lo = self._bc_band_offsets[bc]
        b_hi = self._bc_band_offsets[bc + 1]
        nk, _, ns, ngkmax = tile.shape
        out = np.zeros((nk, self._bpd_max, ns, ngkmax), dtype=tile.dtype)
        out[:, : b_hi - b_lo, :, :] = tile[:, b_lo:b_hi, :, :]
        return out

    def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx) -> np.ndarray:
        """Compatibility adapter; new consumers use the public method."""
        return self.read_local_band_chunk(x_idx, y_idx, bc_idx)

    def _rchunk_kernel(self, n_r_carrier: int):
        """One cached host-store → band-sharded r-carrier executable."""
        n_r_carrier = int(n_r_carrier)
        fn = self._rchunk_kernel_cache.get(n_r_carrier)
        if fn is not None:
            return fn

        from common.wfn_transforms import to_rchunk_inner

        store = self
        fft_grid = tuple(int(s) for s in self.meta.fft_grid)
        out_sds = jax.ShapeDtypeStruct(
            self.local_band_chunk_shape, jnp.complex128)

        def _read_host(x_idx, y_idx, bc_idx):
            return store.read_local_band_chunk(x_idx, y_idx, bc_idx)

        @partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(P(None, None), P(None, None), P(), P()),
            out_specs=band_sphere_spec(),
            check_vma=False,
        )
        def _local(g_index_dev, kvecs_frac_dev, r_start, bc_idx):
            x_idx = jax.lax.axis_index('x')
            y_idx = jax.lax.axis_index('y')
            psi_G_bc = io_callback(
                _read_host, out_sds, x_idx, y_idx, bc_idx, ordered=False)
            return to_rchunk_inner(
                psi_G_bc, g_index_dev, fft_grid,
                r_start, n_r_carrier, norm="ortho",
                kvecs_frac=kvecs_frac_dev)

        def _call(g_index_dev, kvecs_frac_dev, r_start, bc_idx):
            return _local(g_index_dev, kvecs_frac_dev, r_start, bc_idx)

        rep = NamedSharding(self.mesh, P())
        _run = jax.jit(
            _call,
            in_shardings=(
                NamedSharding(self.mesh, P(None, None)),
                NamedSharding(self.mesh, P(None, None)), rep, rep),
            out_shardings=NamedSharding(self.mesh, band_sphere_spec()),
        )
        self._rchunk_kernel_cache[n_r_carrier] = _run
        return _run

    def iter_rchunk_bandwise(
        self,
        r_start: int,
        r_end: int,
        *,
        product_r_spec: P,
    ):
        """Yield cached-WFN ``(band_range, ψ_band(r_chunk))`` pairs.

        Coefficients were read exactly once when this store was constructed.
        Each iteration pulls the process-local G-flat tile through
        ``io_callback``, calls the canonical ``to_rchunk_inner`` FFT/Bloch
        transform, then takes the canonical staged product-band → product-r
        exchange.  No WFN reader or symmetry/FFT formula lives here.

        ``product_r_spec`` is explicit so a consumer's Q layout cannot drift
        from its source.  The one supported contract is
        ``P(None,None,None,('y','x'))``.  Only a terminal logical slab may be
        padded; its carrier tail is exact zero.

        The caller must finish consuming returned arrays before ``close()``;
        an ``io_callback`` may still be in flight while a JAX array is pending.
        """
        if self._closed:
            raise RuntimeError("PsiGStore.iter_rchunk_bandwise: store is closed")
        if not self._host_tiles:
            raise RuntimeError(
                "PsiGStore.iter_rchunk_bandwise: host tiles were released")
        from common.wfn_transforms import prepare_rchunk_carrier
        r_start = int(r_start)
        r_end = int(r_end)
        r_axis, _, finish_r_carrier = prepare_rchunk_carrier(
            self.mesh,
            r_start=r_start,
            r_end=r_end,
            n_rtot=self.meta.n_rtot,
            product_r_spec=product_r_spec,
        )
        kernel = self._rchunk_kernel(r_axis.carrier)
        r_start_dev = jnp.asarray(r_start, dtype=jnp.int32)
        for bc_idx, bc_range in enumerate(self.band_chunk_ranges):
            psi_band_r = kernel(
                self.g_index, self.kvecs_frac, r_start_dev,
                jnp.asarray(bc_idx, dtype=jnp.int32))
            psi_product_r = finish_r_carrier(psi_band_r)
            yield tuple(int(v) for v in bc_range), psi_product_r

    @property
    def g_index(self) -> jax.Array:
        """Replicated ``(nk_tot, nx, ny, nz)`` int32 box-index tensor.

        Staged on device by ``_populate_from_loader``.  Used by
        ``gflat_to_rchunk``.
        """
        if self._g_index_dev is None:
            raise RuntimeError(
                "g_index: store population did not stage the box index")
        return self._g_index_dev

    @property
    def kvecs_frac(self) -> jax.Array:
        """Replicated ``(nk_tot, 3)`` float64 fractional k-vectors."""
        if self._kvecs_frac_dev is None:
            raise RuntimeError(
                "kvecs_frac: store population did not stage k vectors")
        return self._kvecs_frac_dev

    def close(self) -> None:
        """Release all store resources; the shared loader remains caller-owned."""
        self.release_host_tiles()
        self._g_index_dev = None
        self._kvecs_frac_dev = None
        self._closed = True

    def __enter__(self) -> "PsiGStore":
        if self._closed:
            raise RuntimeError("PsiGStore.__enter__: store is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _per_rank_shape_bytes(self) -> int:
        return int(np.prod(self._per_rank_shape)) * 16  # complex128


def build_psi_G_store(
    *,
    wfn,
    mesh_xy: Mesh,
    meta,
    band_chunk_ranges,
    bispinor: bool = False,
    bispinor_lift: str = "raw",
    band_pad_to: int | None = None,
    k_domain: str = "full_bz",
) -> PsiGStore:
    """Construct the one ψ(G-flat) host store.

    Single backend choice: :class:`wfn_loader.WfnLoader`.
    ``backend='auto'`` picks the FFI phdf5 path when multi-rank GPU +
    mesh + .so present; falls back to eager h5py otherwise.  CPU and
    single-process tests get the eager path automatically.

    ``band_pad_to`` supplies a uniform, exactly-zero-padded transport width
    while preserving the logical ``band_chunk_ranges`` exposed by the store.
    It must be at least every logical chunk width and divisible by the
    band-sharding mesh product.

    ``k_domain='ibz'`` builds the store over the WFN file's own raw parent
    rows (``loader.nkpts`` of them) instead of the unfolded full BZ, for a
    consumer that works at parent k.
    """
    loader = wfn  # reuse top-level WfnLoader; opening a second one would
                  # re-slurp wfns/coeffs into host RAM.
    return PsiGStore(
        loader=loader, mesh_xy=mesh_xy,
        band_chunk_ranges=band_chunk_ranges, meta=meta,
        bispinor=bispinor, bispinor_lift=bispinor_lift,
        band_pad_to=band_pad_to, k_domain=k_domain)


# ===========================================================================
# ONE ψ(G) read: the G-slot store and the centroid faces (loader tables
# 2026-09-23, route G).
# ===========================================================================

class ParentPsiG(NamedTuple):
    """The product of :func:`load_parent_psi_G` — one pass over WFN.h5.

    ``psi_G``        ``(n_k, nb_c, ns, ngk_c)`` c128 at
                     ``P(None, None, None, ('x','y'))`` — every band of one
                     ``ngk_c/P`` G-slot slice per rank (``placement='device'``);
                     else ``None``.
    ``host_tile``    this process's ``(n_k, nb_c, ns, ngk_c/P)`` slice of the
                     same array on host (``placement='host'``); else ``None``.
    ``sphere_index`` ``(n_k, ngk_c)`` int32, replicated: the flat FFT-box cell
                     of every G slot, ``n_rtot + g`` on a pad slot — THE one
                     table (:func:`common.gvec_fft_box.build_sphere_box_index`).
                     Rank ``p`` owns slots ``[p·ngk_c/P, (p+1)·ngk_c/P)``,
                     ``p = x·P_y + y``.
    ``kvecs_frac``   ``(n_k, 3)`` the loader's k representatives for these rows.
    ``band_range``   the logical ``[b0, b1)``; bands ``[b1, b0+nb_c)`` are zero.
    ``faces``        ``(psi_y, psi_x)`` exactly as
                     :func:`common.wfn_transforms.load_centroids_band_chunked`
                     returns them, or ``None`` without centroids.
    """
    psi_G: "jax.Array | None"
    host_tile: "np.ndarray | None"
    sphere_index: jax.Array
    kvecs_frac: np.ndarray
    band_range: tuple
    faces: "tuple | None"


def _pad_sphere_index(sphere_index: np.ndarray, width: int, n_rtot: int) -> np.ndarray:
    """Widen the loader's sphere index to a G-slot carrier (pads ``n_rtot + g``)."""
    nk, ngk = sphere_index.shape
    out = np.broadcast_to(n_rtot + np.arange(width, dtype=np.int64),
                          (nk, width)).astype(np.int32)
    out[:, :ngk] = sphere_index
    return out


@lru_cache(maxsize=None)
def _gslot_face_kernel(mesh: Mesh, fft_grid: tuple, nk: int, bc_w: int, ns: int,
                       ngk_c: int, mu_pad: int, mu_t: int, with_faces: bool):
    """Band-sharded ψ chunk → G-slot chunk (+ centroid DFT into the faces).

    Per rank: ONE all-to-all moves the chunk from ``bands_XY`` to
    ``G_XY``.  Then, per k,

        X[b,s,μ] = Σ_{G ∈ my slots} ψ[k,b,s,G] · e^{2πi (k+G)·r_μ} / √N_r

    is a GEMM over the local G slots, psum'ed over the mesh, and each face
    keeps its own μ slice (``'y'`` for ψ_y, ``'x'`` for ψ_x).  The phase is
    formed from integer residues ``(G_a r_a mod n_a)/n_a``, the twiddles the
    FFT itself uses.
    """
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    XY = ('x', 'y')
    P_ = int(mesh.size)
    py = int(mesh.shape['y'])
    px = int(mesh.shape['x'])
    ngk_l = ngk_c // P_
    mu_y, mu_x = mu_pad // py, mu_pad // px
    n_t = mu_pad // mu_t
    inv_sqrt_n = 1.0 / np.sqrt(float(n_rtot))

    def local(psi, sidx, kvecs, r_mu, w_mu, acc_y, acc_x, b0):
        # (nk, bc_w/P, ns, ngk_c) → (nk, bc_w, ns, ngk_l): the one all-to-all.
        g = jax.lax.all_to_all(psi, XY, split_axis=3, concat_axis=1, tiled=True)
        if not with_faces:
            return g, acc_y, acc_x
        p = jax.lax.axis_index('x') * py + jax.lax.axis_index('y')
        idx = jax.lax.dynamic_slice_in_dim(sidx, p * ngk_l, ngk_l, axis=1)
        valid = idx < n_rtot
        c = jnp.where(valid, idx, 0)
        G = jnp.stack([c // (ny * nz), (c // nz) % ny, c % nz], axis=-1)  # (nk, ngk_l, 3)
        grid = jnp.asarray((nx, ny, nz), dtype=jnp.int32)
        r_t = r_mu.reshape(n_t, mu_t, 3)
        w_t = w_mu.reshape(n_t, mu_t)

        def one_k(carry, kk):
            ay, ax_ = carry
            gk, Gk, vk, kv = g[kk], G[kk], valid[kk], kvecs[kk]
            a = gk.reshape(bc_w * ns, ngk_l)

            # ponytail: the phase tile e^{2πi(k+G)·r_μ} is recomputed for
            # every band chunk (one sincos per G·μ) instead of cached across
            # chunks or built separably from 1-D tables: one direct formula.
            def one_tile(args):
                r, w = args                                   # (mu_t, 3), (mu_t,)
                frac = jnp.sum(
                    (jnp.mod(Gk[:, None, :] * r[None, :, :], grid)
                     ).astype(jnp.float64) / grid.astype(jnp.float64), axis=-1)
                E = jnp.where(vk[:, None], jnp.exp(2j * jnp.pi * frac), 0)
                bloch = jnp.exp(2j * jnp.pi * jnp.sum(
                    kv[None, :] * r.astype(jnp.float64)
                    / grid.astype(jnp.float64), axis=-1)) * w * inv_sqrt_n
                return (a @ E) * bloch[None, :]                # (bc_w·ns, mu_t)

            X = jax.lax.map(one_tile, (r_t, w_t))              # (n_t, bc_w·ns, mu_t)
            X = jnp.moveaxis(X, 0, 1).reshape(bc_w, ns, mu_pad)
            X = jax.lax.psum(X, XY)
            y0 = jax.lax.axis_index('y') * mu_y
            x0 = jax.lax.axis_index('x') * mu_x
            Xy = jax.lax.dynamic_slice_in_dim(X, y0, mu_y, axis=2)
            Xx = jnp.conj(jax.lax.dynamic_slice_in_dim(X, x0, mu_x, axis=2)
                          ).transpose(2, 0, 1)
            z = jnp.int32(0)
            ay = jax.lax.dynamic_update_slice(ay, Xy[None], (kk, b0, z, z))
            ax_ = jax.lax.dynamic_update_slice(ax_, Xx[None], (kk, z, b0, z))
            return (ay, ax_), None

        (acc_y, acc_x), _ = jax.lax.scan(
            one_k, (acc_y, acc_x), jnp.arange(nk, dtype=jnp.int32), unroll=1)
        return g, acc_y, acc_x

    rep = P()
    fn = shard_map(
        local, mesh=mesh,
        in_specs=(P(None, XY, None, None), rep, rep, rep, rep,
                  P(None, None, None, 'y'), P(None, 'x', None, None), rep),
        out_specs=(P(None, None, None, XY), P(None, None, None, 'y'),
                   P(None, 'x', None, None)),
        check_vma=False)
    return jax.jit(fn, donate_argnums=(5, 6))


@lru_cache(maxsize=None)
def _gslot_insert_kernel(mesh: Mesh):
    spec = NamedSharding(mesh, P(None, None, None, ('x', 'y')))

    @partial(jax.jit, donate_argnums=(0,), out_shardings=spec)
    def insert(store, chunk, b0):
        z = jnp.int32(0)
        return jax.lax.dynamic_update_slice(store, chunk, (z, b0, z, z))
    return insert


def load_parent_psi_G(
    *,
    wfn,
    mesh_xy: Mesh,
    meta,
    band_range: tuple[int, int],
    band_chunk: int,
    centroid_indices=None,
    placement: str = "device",
    bispinor: bool = False,
    bispinor_lift: str = "raw",
    k_domain: str = "ibz",
    mu_tile_bytes: int = 256 * 2**20,
    print_fn=print,
) -> ParentPsiG:
    """Read ψ(G) of the raw parents ONCE; return the G-slot store and the faces.

    Each band chunk is read band-sharded (the loader's one collective union
    read, ``WfnLoader.load``), moved to G slots by one all-to-all, sampled at
    the centroids by a direct DFT (a GEMM over the local G slots), and then
    either kept on device (``placement='device'``), copied to this process's
    host tile (``'host'``, the streaming case), or dropped (``'none'``: faces
    only).  Nothing is read twice and nothing goes device → host → device.

    ``band_chunk`` is the planner's band width (padded to the mesh here).
    ``centroid_indices`` ``(n_rmu, 3)``; with ``meta.mu_basis`` the packed
    table and its active mask are used, as in ``load_centroids_band_chunked``,
    and the faces leave in the same layouts and extents.

    Per-rank bytes: the store ``n_k·nb_c·ns·ngk_c·16/P``, the in-flight chunk
    twice (read + all-to-all), the replicated sphere index ``n_k·ngk_c·4``
    and one ``ngk_c/P × μ_tile`` phase tile (``mu_tile_bytes``).
    """
    from common import timing
    from common.collectives import device_put_process_local
    from common.wfn_transforms import (
        load_psi_gflat_padded, _centroid_sampling_geometry,
        _centroid_sampling_shardings, _centroid_face_kernels)
    from runtime.padding import padded_axis, mesh_divisor

    if placement not in ("device", "host", "none"):
        raise ValueError(
            f"load_parent_psi_G: placement must be 'device', 'host' or "
            f"'none'; got {placement!r}")
    loader = wfn
    P_ = int(mesh_divisor(mesh_xy))
    fft_grid = tuple(int(v) for v in meta.fft_grid)
    n_rtot = int(np.prod(fft_grid))
    b0, b1 = (int(v) for v in band_range)
    w = padded_axis(int(band_chunk), P_, name="ψ(G) band chunk").carrier
    nb_c = padded_axis(b1 - b0, P_, name="ψ(G) band carrier").carrier
    ngk_c = padded_axis(int(loader.ngkmax), P_, name="ψ(G) G-slot carrier").carrier
    k_spec = "ibz" if k_domain == "ibz" else "full_bz"
    nk = int(loader.nkpts) if k_spec == "ibz" else int(meta.nk_tot)

    sidx_np = _pad_sphere_index(loader.box_index(k=k_spec), ngk_c, n_rtot)
    kvecs = loader.kvecs(k=k_spec)
    rep = NamedSharding(mesh_xy, P())
    sidx = device_put_process_local(sidx_np, NamedSharding(mesh_xy, P(None, None)))
    kvecs_dev = device_put_process_local(kvecs, NamedSharding(mesh_xy, P(None, None)))

    with_faces = centroid_indices is not None
    finish = None
    if with_faces:
        (_, _, _, _, _, _, _, mu_basis, mu_active_mask, n_rmu, cidx_np, _) = (
            _centroid_sampling_geometry(
                (b0, b0 + nb_c), centroid_indices, k_spec, meta, None, False,
                None, loader))
        (_, _, _, _, out_Y, out_X, stage_Y, stage_X, mu_pad, _) = (
            _centroid_sampling_shardings(mesh_xy, meta, mu_basis, n_rmu, loader))
        ngk_l = ngk_c // P_
        mu_t = max(1, min(int(mu_pad), int(mu_tile_bytes) // (16 * max(ngk_l, 1))))
        while mu_pad % mu_t:
            mu_t -= 1
        r_mu = np.zeros((int(mu_pad), 3), dtype=np.int32)
        r_mu[:n_rmu] = cidx_np
        w_mu = np.zeros((int(mu_pad),), dtype=np.float64)
        w_mu[:n_rmu] = 1.0 if mu_active_mask is None else mu_active_mask
        r_mu_dev = device_put_process_local(r_mu, rep)
        w_mu_dev = device_put_process_local(w_mu, rep)

        @partial(jax.jit, out_shardings=(out_Y, out_X))
        def _zero_faces():
            return (jnp.zeros((nk, nb_c, int(meta.nspinor), int(mu_pad)), jnp.complex128),
                    jnp.zeros((nk, int(mu_pad), nb_c, int(meta.nspinor)), jnp.complex128))
        acc_y, acc_x = _zero_faces()
        _, finish = _centroid_face_kernels(
            b0, meta, mu_active_mask, n_rmu, int(mu_pad), b1 - b0, out_X, out_Y,
            stage_X, stage_Y)
    else:
        mu_pad, mu_t = P_, 1
        r_mu_dev = device_put_process_local(np.zeros((P_, 3), np.int32), rep)
        w_mu_dev = device_put_process_local(np.zeros((P_,), np.float64), rep)
        acc_y, acc_x = jax.jit(
            lambda: (jnp.zeros((1, 1, 1, P_), jnp.complex128),
                     jnp.zeros((1, P_, 1, 1), jnp.complex128)),
            out_shardings=(NamedSharding(mesh_xy, P(None, None, None, 'y')),
                           NamedSharding(mesh_xy, P(None, 'x', None, None))))()

    ns = int(meta.nspinor) if bispinor else int(loader.nspinor)
    store = host_tile = None
    gspec = NamedSharding(mesh_xy, P(None, None, None, ('x', 'y')))
    if placement == "device":
        store = jax.jit(lambda: jnp.zeros((nk, nb_c, ns, ngk_c), jnp.complex128),
                        out_shardings=gspec)()
    elif placement == "host":
        if len(mesh_xy.local_devices) != 1:
            raise ValueError("load_parent_psi_G(placement='host') wants one "
                             "device per process (the LORRAX launch).")
        host_tile = np.zeros((nk, nb_c, ns, ngk_c // P_), np.complex128)
    insert = _gslot_insert_kernel(mesh_xy)
    t_read = t_xform = t_keep = 0.0
    import time as _time
    for lo in range(b0, b0 + nb_c, w):
        wc = min(w, b0 + nb_c - lo)     # the tail chunk: a P multiple, ≤ w
        t0 = _time.perf_counter()
        with timing.section("psi_G_store.gslot.read"):
            chunk = load_psi_gflat_padded(
                loader, (lo, min(lo + wc, b1)), mesh_xy=mesh_xy,
                bispinor=bispinor, pad_to=wc, k=k_spec,
                sharding=band_sphere_spec(), bispinor_lift=bispinor_lift)
            if chunk is None:          # wholly past the file's bands: zeros
                continue
            if int(chunk.shape[-1]) < ngk_c:
                chunk = jax.jit(
                    lambda a: jnp.pad(a, ((0, 0), (0, 0), (0, 0),
                                          (0, ngk_c - a.shape[-1]))),
                    out_shardings=NamedSharding(mesh_xy, band_sphere_spec()))(chunk)
            jax.block_until_ready(chunk)
        t1 = _time.perf_counter()
        kern = _gslot_face_kernel(mesh_xy, fft_grid, nk, wc, ns, ngk_c,
                                  int(mu_pad), int(mu_t), with_faces)
        with timing.section("psi_G_store.gslot.a2a_faces"):
            g_chunk, acc_y, acc_x = kern(chunk, sidx, kvecs_dev, r_mu_dev,
                                         w_mu_dev, acc_y, acc_x,
                                         jnp.int32(lo - b0))
            del chunk
            jax.block_until_ready((g_chunk, acc_y, acc_x))
        t2 = _time.perf_counter()
        with timing.section("psi_G_store.gslot.keep"):
            if placement == "device":
                store = insert(store, g_chunk, jnp.int32(lo - b0))
                jax.block_until_ready(store)
            elif placement == "host":
                (shard,) = g_chunk.addressable_shards
                host_tile[:, lo - b0:lo - b0 + wc] = np.asarray(shard.data)
            del g_chunk
        t3 = _time.perf_counter()
        t_read += t1 - t0
        t_xform += t2 - t1
        t_keep += t3 - t2
    faces = None
    if with_faces:
        faces = finish(acc_y, acc_x, nk)
        jax.block_until_ready(faces)
    print_fn(f"  ψ(G) one-read: {nk} k × {nb_c} bands × {ngk_c} G slots "
             f"({(b1 - b0)} logical) in chunks of {w}; read {t_read:.2f}s, "
             f"all-to-all{' + centroid DFT' if with_faces else ''} "
             f"{t_xform:.2f}s, keep[{placement}] {t_keep:.2f}s")
    return ParentPsiG(store, host_tile, sidx, kvecs, (b0, b1), faces)
