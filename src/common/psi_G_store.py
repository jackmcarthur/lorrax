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

from functools import partial
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
            in_specs=(P(None, None, None, None), P(None, None), P(), P()),
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
                NamedSharding(self.mesh, P(None, None, None, None)),
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

    # ---------------------------------------------------------------------
    # Transform-once sources.  ``iter_rchunk_bandwise`` above repeats the
    # full-box IFFT for every r chunk; the sources below transform each
    # G-flat row ONCE over the whole grid (the same ``to_rchunk_inner`` at
    # ``r0 = 0``, ``r_len = n_rtot``), hold that band-sharded ψ(r) slab, and
    # then cut it into the caller's r chunks through the same canonical
    # ``prepare_rchunk_carrier`` exchange.  A consumer therefore pays one
    # IFFT and one band->r all-to-all per row per pass.
    # ---------------------------------------------------------------------
    def _band_owner_map(self):
        """``{device: (flat band-block owner, flat row-block index)}`` for
        the band-sphere layout, from JAX's own sharding index maps."""
        p = spec_divisor(self.mesh, band_sphere_spec(), axis=1)
        nk, _, ns, ngkmax = self._per_rank_shape
        band_map = NamedSharding(self.mesh, band_sphere_spec()) \
            .devices_indices_map((1, p, 1, 1))
        row_map = NamedSharding(self.mesh, P(("x", "y"), None, None)) \
            .devices_indices_map((p, 1, 1))
        return {dev: (int(band_map[dev][1].start or 0),
                      int(row_map[dev][0].start or 0))
                for dev in band_map}

    def state_row_owners(self, states, *, band_start: int, band_count: int):
        """Host tile address of stacked states ``s = k*band_count + b``.

        Returns ``(owner, k, tile_band)``: the flat band-block owner of each
        state (the device whose host tile holds it) and its row in that tile.
        """
        states = np.asarray(states, dtype=np.int64).reshape(-1)
        nk = int(self._per_rank_shape[0])
        band_start, band_count = int(band_start), int(band_count)
        k_idx = states // band_count
        band = band_start + states % band_count
        lo = np.asarray([int(b0) for b0, _ in self.band_chunk_ranges])
        hi = np.asarray([int(b1) for _, b1 in self.band_chunk_ranges])
        bc = np.searchsorted(hi, band, side="right")
        if (np.any(states < 0) or np.any(k_idx >= nk)
                or np.any(bc >= len(lo)) or np.any(band < lo[np.minimum(
                    bc, len(lo) - 1)])):
            raise ValueError(
                "state_row_owners: a requested state lies outside the "
                f"store's k extent {nk} or bands {self.band_chunk_ranges}")
        bpd = np.asarray(self._bpd_per_bc, dtype=np.int64)[bc]
        within = band - lo[bc]
        return (within // bpd, k_idx,
                np.asarray(self._bc_band_offsets)[bc] + within % bpd)

    def gather_state_rows(self, states, *, band_start: int, band_count: int,
                          row_multiple: int = 1):
        """Stage selected stacked states ``s = k*band_count + b`` on device.

        Each device takes the requested rows that its own host tile already
        holds, so this is a host copy plus one H2D placement with no
        collective.  Returns ``(rows, row_k, row_state)``: ``rows`` has shape
        ``(P*n_loc, ns, ngkmax)`` on ``P(('x','y'),None,None)`` (exact-zero
        pad rows), ``row_k`` the ``(P*n_loc,)`` int32 k index of every row
        slot on ``P(('x','y'))``, and ``row_state`` the host ``(P*n_loc,)``
        stacked-state label of every slot (``-1`` = pad), identical on every
        process.  ``n_loc`` is the largest per-device count, padded through
        ``runtime.padding`` to a multiple of ``row_multiple``.
        """
        if self._closed or not self._host_tiles:
            raise RuntimeError("PsiGStore.gather_state_rows: store is closed")
        from runtime.padding import padded_axis
        states = np.asarray(states, dtype=np.int64).reshape(-1)
        _, _, ns, ngkmax = self._per_rank_shape
        band_count = int(band_count)
        owner, k_idx, tile_band = self.state_row_owners(
            states, band_start=band_start, band_count=band_count)

        owners = self._band_owner_map()
        counts = {o: int(np.count_nonzero(owner == o))
                  for o, _ in owners.values()}
        n_loc = padded_axis(
            max(1, max(counts.values())), int(row_multiple),
            name="PsiGStore gathered-row carrier").carrier
        p = len(owners)
        row_state = np.full(p * n_loc, -1, dtype=np.int64)
        for o, q in owners.values():
            pick = np.flatnonzero(owner == o)
            row_state[q * n_loc:q * n_loc + pick.size] = states[pick]
        row_k = np.where(row_state >= 0, row_state // band_count, 0)

        row_sharding = NamedSharding(self.mesh, P(("x", "y"), None, None))
        blocks = []
        for dev in row_sharding.addressable_devices:
            o, q = owners[dev]
            pick = np.flatnonzero(owner == o)
            block = np.zeros((n_loc, ns, ngkmax), dtype=np.complex128)
            tile = self._host_tiles[self._coords[id(dev)]]
            block[:pick.size] = tile[k_idx[pick], tile_band[pick]]
            blocks.append(jax.device_put(block, dev))
        rows = jax.make_array_from_single_device_arrays(
            (p * n_loc, ns, ngkmax), row_sharding, blocks)
        from common.collectives import device_put_process_local
        row_k_dev = device_put_process_local(
            row_k.astype(np.int32), NamedSharding(self.mesh, P(("x", "y"))))
        return rows, row_k_dev, row_state

    def _rows_fullr_kernel(self, n_rows: int, fft_rows: int):
        """Rows ψ(G) → band-sharded full-grid ψ(r), ``fft_rows`` at a time."""
        key = ("rows", int(n_rows), int(fft_rows))
        fn = self._rchunk_kernel_cache.get(key)
        if fn is not None:
            return fn
        from common.wfn_transforms import to_rchunk_inner
        p = spec_divisor(self.mesh, band_sphere_spec(), axis=1)
        n_loc, t = int(n_rows) // p, int(fft_rows)
        if n_loc * p != int(n_rows) or t <= 0 or n_loc % t:
            raise ValueError(
                f"_rows_fullr_kernel: {n_rows} rows over {p} devices must "
                f"split into whole {fft_rows}-row transform tiles")
        fft_grid = tuple(int(s) for s in self.meta.fft_grid)
        n_rtot = int(self.meta.n_rtot)
        _, _, ns, ngkmax = self._per_rank_shape

        @partial(
            shard_map, mesh=self.mesh,
            in_specs=(P(("x", "y"), None, None), P(("x", "y")),
                      P(None, None, None, None), P(None, None)),
            out_specs=band_sphere_spec(), check_vma=False)
        def _local(rows, row_k, g_index, kvecs_frac):
            def _tile(args):
                psi_t, k_t = args
                return to_rchunk_inner(
                    psi_t[:, None], g_index[k_t], fft_grid, 0, n_rtot,
                    norm="ortho", kvecs_frac=kvecs_frac[k_t])[:, 0]
            out = jax.lax.map(_tile, (rows.reshape(n_loc // t, t, ns, ngkmax),
                                      row_k.reshape(n_loc // t, t)))
            return out.reshape(1, n_loc, ns, n_rtot)

        fn = jax.jit(
            _local,
            in_shardings=(
                NamedSharding(self.mesh, P(("x", "y"), None, None)),
                NamedSharding(self.mesh, P(("x", "y"))),
                NamedSharding(self.mesh, P(None, None, None, None)),
                NamedSharding(self.mesh, P(None, None))),
            out_shardings=NamedSharding(self.mesh, band_sphere_spec()))
        self._rchunk_kernel_cache[key] = fn
        return fn

    def _bandchunk_fullr_kernel(self, k_tile: int):
        """One band chunk's host tile → band-sharded full-grid ψ(r).

        ``k_tile`` k rows share one FFT box; it must divide ``nk``.
        """
        key = ("bandchunk", int(k_tile))
        fn = self._rchunk_kernel_cache.get(key)
        if fn is not None:
            return fn
        from common.wfn_transforms import to_rchunk_inner
        store = self
        nk, bpd, ns, ngkmax = self.local_band_chunk_shape
        kt = int(k_tile)
        if kt <= 0 or nk % kt:
            raise ValueError(
                f"_bandchunk_fullr_kernel: k_tile={kt} must divide nk={nk}")
        fft_grid = tuple(int(s) for s in self.meta.fft_grid)
        n_rtot = int(self.meta.n_rtot)
        out_sds = jax.ShapeDtypeStruct((nk, bpd, ns, ngkmax), jnp.complex128)

        def _read_host(x_idx, y_idx, bc_idx):
            return store.read_local_band_chunk(x_idx, y_idx, bc_idx)

        @partial(
            shard_map, mesh=self.mesh,
            in_specs=(P(None, None, None, None), P(None, None), P()),
            out_specs=band_sphere_spec(), check_vma=False)
        def _local(g_index, kvecs_frac, bc_idx):
            psi_G = io_callback(
                _read_host, out_sds, jax.lax.axis_index('x'),
                jax.lax.axis_index('y'), bc_idx, ordered=False)

            def _tile(args):
                psi_t, g_t, kv_t = args
                return to_rchunk_inner(
                    psi_t, g_t, fft_grid, 0, n_rtot, norm="ortho",
                    kvecs_frac=kv_t)
            split = lambda a: a.reshape(nk // kt, kt, *a.shape[1:])
            out = jax.lax.map(
                _tile, (split(psi_G), split(g_index), split(kvecs_frac)))
            return out.reshape(nk, bpd, ns, n_rtot)

        rep = NamedSharding(self.mesh, P())
        fn = jax.jit(
            _local,
            in_shardings=(
                NamedSharding(self.mesh, P(None, None, None, None)),
                NamedSharding(self.mesh, P(None, None)), rep),
            out_shardings=NamedSharding(self.mesh, band_sphere_spec()))
        self._rchunk_kernel_cache[key] = fn
        return fn

    def _slice_rchunk_kernel(self, r_carrier: int):
        """Band-sharded full-grid slab → one padded r-chunk carrier (local)."""
        key = ("slice", int(r_carrier))
        fn = self._rchunk_kernel_cache.get(key)
        if fn is not None:
            return fn
        from common.wfn_transforms import take_rchunk_padded
        layout = NamedSharding(self.mesh, band_sphere_spec())

        @partial(
            shard_map, mesh=self.mesh,
            in_specs=(band_sphere_spec(), P()), out_specs=band_sphere_spec(),
            check_vma=False)
        def _local(full, r0):
            return take_rchunk_padded(full, r0, int(r_carrier))

        fn = jax.jit(_local, in_shardings=(layout, NamedSharding(
            self.mesh, P())), out_shardings=layout)
        self._rchunk_kernel_cache[key] = fn
        return fn

    def _iter_fullr_rchunks(self, psi_full, r_chunk_ranges, product_r_spec):
        from common.wfn_transforms import prepare_rchunk_carrier
        for r_idx, (r0, r1) in enumerate(r_chunk_ranges):
            key = ("carrier", int(r0), int(r1), tuple(product_r_spec))
            if key not in self._rchunk_kernel_cache:
                self._rchunk_kernel_cache[key] = prepare_rchunk_carrier(
                    self.mesh, r_start=int(r0), r_end=int(r1),
                    n_rtot=self.meta.n_rtot, product_r_spec=product_r_spec)
            r_axis, _, finish = self._rchunk_kernel_cache[key]
            slab = self._slice_rchunk_kernel(r_axis.carrier)(
                psi_full, jnp.asarray(int(r0), dtype=jnp.int32))
            last = finish(slab)
            del slab
            yield r_idx, last
        # Every reader of ``psi_full`` has run once its last chunk exists, so
        # the caller's next full-grid slab cannot coexist with this one.
        jax.block_until_ready(last)

    def iter_rows_rchunks(self, rows, row_k, r_chunk_ranges, *,
                          product_r_spec: P, fft_rows: int):
        """Yield ``(r_idx, ψ_rows(r_chunk))`` from rows transformed once.

        ``rows``/``row_k`` come from :meth:`gather_state_rows`.  Every yielded
        slab has shape ``(1, P*n_loc, ns, r_carrier)`` on ``product_r_spec``
        (rows in the gathered slot order, r over the full mesh product).
        """
        if self._closed:
            raise RuntimeError("PsiGStore.iter_rows_rchunks: store is closed")
        kernel = self._rows_fullr_kernel(int(rows.shape[0]), int(fft_rows))
        psi_full = kernel(rows, row_k, self.g_index, self.kvecs_frac)
        yield from self._iter_fullr_rchunks(
            psi_full, r_chunk_ranges, product_r_spec)

    def iter_bandchunks_rchunks(self, r_chunk_ranges, *,
                                product_r_spec: P, k_tile: int):
        """Yield ``(band_range, r_idx, ψ_band(r_chunk))``, one IFFT per state.

        The loop-inverted twin of :meth:`iter_rchunk_bandwise`: band chunks
        outside, r chunks inside, and each band chunk transformed once over
        the whole grid.  Slabs have the same shape and layout as that
        iterator's.
        """
        if self._closed or not self._host_tiles:
            raise RuntimeError(
                "PsiGStore.iter_bandchunks_rchunks: store is closed")
        kernel = self._bandchunk_fullr_kernel(int(k_tile))
        for bc_idx, bc_range in enumerate(self.band_chunk_ranges):
            psi_full = kernel(self.g_index, self.kvecs_frac,
                              jnp.asarray(bc_idx, dtype=jnp.int32))
            for r_idx, slab in self._iter_fullr_rchunks(
                    psi_full, r_chunk_ranges, product_r_spec):
                yield tuple(int(v) for v in bc_range), r_idx, slab
            del psi_full

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
