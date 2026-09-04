"""Parity and read-count gate for ``PsiGStore``'s public r-chunk source."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import numpy as np
import pytest

from jax.sharding import Mesh, PartitionSpec as P

from common.psi_G_store import _mesh_device_coords, build_psi_G_store
from common.wfn_transforms import iter_psi_rchunk_bandwise
from wfn_loader import WfnLoader  # noqa: E402


# Reuse the service-owned synthetic BerkeleyGW WFN writer.  A second local
# HDF5 fixture would be a second file-format implementation.
_SVC_TESTS = str(Path(__file__).resolve().parents[1]
                 / "services" / "wfn_loader" / "tests")
if _SVC_TESTS not in sys.path:
    sys.path.insert(0, _SVC_TESTS)
from test_wfn_loader_contract import _synth_wfn  # noqa: E402


class _CountingLoader:
    """Transparent loader proxy whose one extra fact is ``load_calls``."""

    def __init__(self, loader):
        self._loader = loader
        self.load_calls: list[tuple[int, int]] = []

    def load(self, *, bands, **kwargs):
        self.load_calls.append(tuple(int(v) for v in bands))
        return self._loader.load(bands=bands, **kwargs)

    def __getattr__(self, name):
        return getattr(self._loader, name)


def _mesh():
    devices = jax.devices()
    if len(devices) >= 4:
        return Mesh(np.asarray(devices[:4]).reshape(2, 2), ('x', 'y'))
    return Mesh(np.asarray(devices[:1]).reshape(1, 1), ('x', 'y'))


def _meta(loader):
    sym = loader.symmetry()
    fft_grid = tuple(int(v) for v in loader.fft_grid)
    return SimpleNamespace(
        nk_tot=int(sym.nk_tot),
        nspinor=int(loader.nspinor),
        fft_grid=fft_grid,
        n_rtot=int(np.prod(fft_grid)),
        kgrid=tuple(int(v) for v in loader.kgrid),
        b_id_4_user=int(loader.nbands),
        b_id_4=int(loader.nbands),
    )


def _collect(iterator):
    out = []
    for band_range, psi_r in iterator:
        psi_r.block_until_ready()
        out.append((band_range, np.asarray(psi_r), psi_r.sharding.spec))
    return out


def test_mesh_device_coords_excludes_nonaddressable_cells():
    """A process allocates no host tile for another process's mesh cell."""
    dev00, dev01, dev10, dev11 = (object() for _ in range(4))
    mesh = SimpleNamespace(
        devices=np.asarray([[dev00, dev01], [dev10, dev11]], dtype=object),
        local_devices=(dev10,),
    )
    assert _mesh_device_coords(mesh) == {id(dev10): (1, 0)}


def test_store_source_matches_direct_iterator_without_rereads(tmp_path):
    """Two r slabs consume one coefficient read per band chunk, not per slab.

    The short final band chunk exercises ``band_pad_to``.  On a forced P4
    process the terminal three-cell r slab also exercises the exact-zero
    product-r carrier tail.
    """
    mesh = _mesh()
    wfn_path = _synth_wfn(tmp_path)
    product_r_spec = P(None, None, None, ('y', 'x'))
    band_ranges = ((0, 4), (4, 6))

    with WfnLoader(wfn_path, mesh=mesh, backend="eager") as loader:
        meta = _meta(loader)
        counted = _CountingLoader(loader)
        with build_psi_G_store(
            wfn=counted,
            mesh_xy=mesh,
            meta=meta,
            band_chunk_ranges=band_ranges,
            band_pad_to=4,
        ) as source:
            # Population is the only coefficient-I/O phase.
            assert counted.load_calls == [(0, 4), (4, 6)]
            assert source.band_chunk_carrier == 4
            assert len(source._host_tiles) == len(mesh.local_devices)
            expected_host_bytes = (
                len(mesh.local_devices)
                * int(np.prod(source._per_rank_shape))
                * np.dtype(np.complex128).itemsize
            )
            assert source.host_cache_bytes == expected_host_bytes

            cached_main = _collect(source.iter_rchunk_bandwise(
                0, 8, product_r_spec=product_r_spec))
            cached_tail = _collect(source.iter_rchunk_bandwise(
                meta.n_rtot - 3, meta.n_rtot,
                product_r_spec=product_r_spec))

            # Re-entering the source at a different r offset performs FFTs and
            # staged exchanges, but never returns to WfnLoader.load/PHDF5.
            assert counted.load_calls == [(0, 4), (4, 6)]

            direct_main = _collect(iter_psi_rchunk_bandwise(
                loader, None, meta, mesh, (0, 6), 0, 8, False,
                band_chunk_ranges=list(band_ranges), band_pad_to=4,
                product_r_spec=product_r_spec))
            direct_tail = _collect(iter_psi_rchunk_bandwise(
                loader, None, meta, mesh, (0, 6),
                meta.n_rtot - 3, meta.n_rtot, False,
                band_chunk_ranges=list(band_ranges), band_pad_to=4,
                product_r_spec=product_r_spec))

            for cached, direct in zip(
                    cached_main + cached_tail, direct_main + direct_tail):
                assert cached[0] == direct[0]
                assert cached[2] == product_r_spec == direct[2]
                np.testing.assert_array_equal(cached[1], direct[1])

            if mesh.size == 4:
                # logical terminal width 3 -> carrier width 4 on product-r P4
                assert cached_tail[0][1].shape[-1] == 4
                assert np.all(cached_tail[0][1][..., 3] == 0)

            # Production hoisted-cache lifecycle: callbacks have drained, so
            # host coefficient tiles can go away while the canonical paired
            # transform metadata remains owned by this store.
            g_index = source.g_index
            kvecs_frac = source.kvecs_frac
            source.release_host_tiles()
            assert source.host_cache_bytes == 0
            assert source.g_index is g_index
            assert source.kvecs_frac is kvecs_frac
            with pytest.raises(RuntimeError, match="host tiles were released"):
                source.read_local_band_chunk(0, 0, 0)

        # Negative discriminator: final close is still the full teardown.
        with pytest.raises(RuntimeError, match="did not stage the box index"):
            _ = source.g_index
        with pytest.raises(RuntimeError, match="did not stage k vectors"):
            _ = source.kvecs_frac
