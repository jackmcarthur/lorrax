"""Physical plane-wave oracle for signed/rebased and shifted WFN grids.

Compare the actual centroid, streaming and cached loading entry points with
sum_G c(G) exp(2*pi*i*(k+G).r), evaluated independently from the input file.
No TRS is assumed: the synthetic spinors are arbitrary complex coefficients.
"""
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from file_io import WfnLoader
from common.wfn_transforms import (
    iter_psi_rchunk_bandwise, load_centroids_band_chunked,
)
from common.psi_G_store import PsiGStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'services/wfn_loader/tests'))
from test_wfn_loader_contract import _synth_wfn


def check_negative_k_loading(tmp_path, mesh, shift, backend='eager'):
    """Run on a local CPU mesh or a real distributed GPU mesh."""
    path = tmp_path / 'WFN.h5'
    alt = tmp_path / 'WFN_rebased.h5'
    from jax.experimental.multihost_utils import sync_global_devices
    if jax.process_index() == 0:
        path = Path(_synth_wfn(tmp_path))
        # Unique Gs well inside the FFT box, even after the integer rebasing.
        gs = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [-1,0,0],
                       [0,-1,0], [0,0,-1], [1,1,0], [-1,0,1]], dtype=np.int32)
        with h5py.File(path, 'r+') as f:
            f['wfns/gvecs'][...] = np.concatenate((gs[:7], gs))
            k = f['mf_header/kpoints/rk'][...]
            k += np.array(shift) / np.array([1,1,2])
            f['mf_header/kpoints/rk'][...] = k
            f['mf_header/kpoints/shift'][...] = shift
            packed = f['wfns/coeffs'][...]
            coeff = packed[...,0] + 1j*packed[...,1]
        shutil.copyfile(path, alt)
        jumps = np.array([[1,-1,0], [0,0,1]], dtype=np.int32)
        with h5py.File(alt, 'r+') as f:
            f['mf_header/kpoints/rk'][...] = k-jumps
            f['wfns/gvecs'][...] = np.concatenate((gs[:7]+jumps[0], gs+jumps[1]))
    sync_global_devices('negative-k-fixture-written')
    with h5py.File(path, 'r') as f:
        k = f['mf_header/kpoints/rk'][...]
        gs = f['wfns/gvecs'][7:]
        packed = f['wfns/coeffs'][...]
        coeff = packed[...,0] + 1j*packed[...,1]
    grid = (8,8,8)
    start, stop = 57, 73
    coords = np.stack(np.unravel_index(np.arange(start, stop), grid), axis=1)
    r = coords / np.array(grid)
    ref = np.stack([
        np.einsum('bsg,gr->bsr', coeff[:4,:,a:b],
                  np.exp(2j*np.pi*(g+k[i])@r.T))/np.sqrt(np.prod(grid))
        for i,(a,b,g) in enumerate(((0,7,gs[:7]), (7,16,gs)))])
    meta = SimpleNamespace(nk_tot=2, nspinor=2, fft_grid=grid, n_rtot=512,
                           kgrid=(1,1,2), b_id_4_user=4, b_id_4=4,
                           memory_per_device_gb=1)
    for source in (path, alt):
        with WfnLoader(str(source), mesh=mesh, backend=backend) as loader:
            sym = loader.symmetry()
            np.testing.assert_allclose(sym.unfolded_kpts, k, atol=1e-14)
            # Assert preservation of physical momenta at the loader boundary.
            full_g = loader.gvecs(k='full_bz')
            for i,n in enumerate((7,9)):
                np.testing.assert_allclose(full_g[i,:n]+sym.unfolded_kpts[i],
                                           gs[:n]+k[i], atol=1e-14)
            chunks = iter_psi_rchunk_bandwise(loader, sym, meta, mesh,
                        (0,4), start, stop, False, band_chunk_size=4)
            _, streamed = next(chunks)
            mu, _ = load_centroids_band_chunked(loader, sym, meta, coords,
                                               False, mesh, (0,4))
            store = PsiGStore(loader=loader, mesh_xy=mesh,
                             band_chunk_ranges=((0,4),), meta=meta)
            _, cached = next(store.iter_rchunk_bandwise(start, stop, product_r_spec=P(None,None,None,('y','x'))))
            for name, result in [('streamed', streamed), ('centroids',mu),
                                 ('cached',cached)]:
                from jax.experimental.multihost_utils import process_allgather
                actual = np.asarray(process_allgather(result, tiled=True))
                np.testing.assert_allclose(actual, ref, atol=2e-12, rtol=2e-12,
                                           err_msg=f'{name}: {source.name}, shift={shift}')
            store.close()


@pytest.mark.parametrize('shift', [(0.,0.,0.), (0.,0.,0.5), (0.25,0.5,0.5)])
def test_negative_k_loading(tmp_path, shift):
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1,1), ('x','y'))
    check_negative_k_loading(tmp_path, mesh, shift)
