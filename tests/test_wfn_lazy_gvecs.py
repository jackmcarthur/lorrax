"""File-backed G vectors and exactly compatible, bounded TRS content hashing."""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from ffi import _services
_services.ensure_on_path()
from symmetry_maps import density_symmetry_check as check
from wfn_loader import WfnLoader


def _legacy_digest(loader):
    digest = hashlib.sha256()
    for name in check._STAMP_FIELDS:
        value = np.ascontiguousarray(np.asarray(getattr(loader, name)))
        digest.update(f'{name}|{value.dtype.str}|{value.shape}|'.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class _BoundedDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shape, self.dtype = dataset.shape, dataset.dtype
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('whole raw-G materialization is forbidden')

    def __getitem__(self, rows):
        value = self.dataset[rows]
        self.reads.append(value.nbytes)
        assert value.nbytes <= 8 << 20
        return value


def test_streamed_stamp_bytes_match_legacy_and_detect_content_changes(tmp_path):
    path = tmp_path/'raw_g.h5'
    for dtype in ('<i4', '>i4', '<i8', '>i8'):
        with h5py.File(path, 'w') as file:
            raw = (np.arange(800003*3, dtype=np.int64).reshape(-1, 3)-193).astype(dtype)
            dataset = file.create_dataset('gvecs', data=raw)
            values = {name:np.array([1, 3, 7], dtype='>i8') for name in check._STAMP_FIELDS}
            values.update(path=str(path), _gvecs_raw=dataset,
                          energies=np.arange(18, dtype='>f8').reshape(3, 6)[:, ::2],
                          occs=np.array([], dtype='<f8'), alat=np.array(2.5),
                          atom_positions=np.zeros((0, 3), dtype=np.float64))
            loader = SimpleNamespace(**values)
            expected = _legacy_digest(loader)
            assert check._content_digest(loader) == expected
            guarded = _BoundedDataset(dataset)
            loader._gvecs_raw = guarded
            assert check._content_digest(loader) == expected
            assert len(guarded.reads) > 1
            file.flush()
            before_stat = os.stat(path)
            before = check._stamp_identity(loader, 2, 1e-6, 12)
            dataset[0, 0] += 1
            file.flush()
            os.utime(path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
            after = check._stamp_identity(loader, 2, 1e-6, 12)
            assert before is not None and after is not None
            assert before['content_sha256'] != after['content_sha256']
            assert check._identity_digest(before) != check._identity_digest(after)
            assert {k:v for k,v in before.items() if k != 'content_sha256'} == {
                   k:v for k,v in after.items() if k != 'content_sha256'}


def test_loader_keeps_raw_g_file_backed_and_symmetry_stub_survives_close():
    for fixture in ('fixture_na', 'fixture'):
        path = Path(__file__).parent/'hsuite'/fixture/'WFN.h5'
        with WfnLoader(path, backend='eager') as loader:
            assert isinstance(loader._gvecs_raw, h5py.Dataset)
            assert check._content_digest(loader) == _legacy_digest(loader)
            observed = _BoundedDataset(loader._gvecs_raw)
            loader._gvecs_raw = observed
            loader._parent_g_host_cache = None
            first = loader.get_gvec_nk(0)
            assert loader.get_gvec_nk(0) is first
            assert len(observed.reads) == 1
            if int(loader.nkpts) > 1:
                loader.get_gvec_nk(1)
                loader.get_gvec_nk(0)
                assert len(observed.reads) == 3
            with h5py.File(path) as raw:
                starts = np.concatenate(([0], np.cumsum(np.asarray(loader.ngk))))
                for k in (0, int(loader.nkpts)-1):
                    expected = raw['wfns/gvecs'][starts[k]:starts[k+1]]
                    np.testing.assert_array_equal(loader.get_gvec_nk(k), expected)
            sym = loader.symmetry()
            kpoints = np.asarray(sym.unfolded_kpts).copy()
        assert loader._parent_g_host_cache is None
        np.testing.assert_array_equal(sym.unfolded_kpts, kpoints)
