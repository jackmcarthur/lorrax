"""Parent-row velocity reads preserve the typed polar action and band window."""
from types import SimpleNamespace

import h5py
import numpy as np

from gw.qsgw_head import read_authenticated_dipole_velocity
from symmetry_maps import SymMaps


def test_parent_velocity_ignores_nonparent_payload_and_slices_bands(tmp_path, monkeypatch):
    import runtime

    monkeypatch.setattr(runtime, "initialize_communicator_stack", lambda **kw: None)
    import psp.get_dipole_mtxels as owner

    sym = object.__new__(SymMaps)
    sym.nk_red, sym.nk_tot = 1, 2
    sym.kirr_fullids = np.asarray([0])
    sym.irr_idx_k = np.asarray([0, 0])
    sym.sym_idx_k = np.asarray([0, 1])
    sym.sym_matrices = np.eye(3)[None]
    sym.sym_mats_k = np.stack([np.eye(3), -np.eye(3)])
    sym.translations = np.zeros((1, 3))
    sym.R_cart = np.eye(3)[None]
    velocity = np.full((3, 2, 4, 4), np.nan, dtype=np.complex128)
    velocity[:, 0, 1:3, 1:3] = np.asarray([1+2j, 3-4j, 5+6j])[:, None, None]
    path = tmp_path / 'dipole.h5'
    with h5py.File(path, 'w') as h5:
        h5['dipole_cart'] = velocity
    monkeypatch.setattr(owner, 'check_dipole_provenance', lambda *a, **k: True)
    monkeypatch.setattr(owner, 'resolve_vnl_velocity_sign', lambda *a: 1)
    got = read_authenticated_dipole_velocity(
        path, wfn=SimpleNamespace(symmetry=lambda: sym),
        meta=SimpleNamespace(nspinor=2, b_id_0=1, b_id_4_chi_user=3),
        config=SimpleNamespace(nval=1, ncond=1, nband=4, vnl_velocity_sign=''))
    assert got.shape == (3, 2, 2, 2)
    np.testing.assert_array_equal(got[:, 0], velocity[:, 0, 1:3, 1:3])
    np.testing.assert_array_equal(got[:, 1], -velocity[:, 0, 1:3, 1:3].conj())
