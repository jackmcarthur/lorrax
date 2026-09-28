"""Parent-row velocity reads preserve the typed polar action and band window."""
from types import SimpleNamespace

import h5py
import pytest
import numpy as np

from gw.qsgw_head import read_authenticated_dipole_velocity
from symmetry_maps import SymMaps


def test_parent_velocity_ignores_nonparent_payload_and_slices_bands(tmp_path, monkeypatch):
    import runtime

    monkeypatch.setattr(runtime, "initialize_communicator_stack", lambda **kw: None)
    import psp.get_dipole_mtxels as owner
    from file_io import restart_bundle as reader

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
    monkeypatch.setattr(reader, 'check_dipole_provenance', lambda *a, **k: True)
    monkeypatch.setattr(owner, 'resolve_vnl_velocity_sign', lambda *a: 1)
    import jax
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    got = read_authenticated_dipole_velocity(
        path, wfn=SimpleNamespace(symmetry=lambda: sym),
        meta=SimpleNamespace(nspinor=2, b_id_0=1, b_id_4_chi_user=3),
        config=SimpleNamespace(nval=1, ncond=1, nband=4, vnl_velocity_sign=''),
        mesh=mesh)
    assert got.shape == (3, 2, 2, 2)
    np.testing.assert_array_equal(got[:, 0], velocity[:, 0, 1:3, 1:3])
    np.testing.assert_array_equal(got[:, 1], -velocity[:, 0, 1:3, 1:3].conj())


@pytest.mark.parametrize("nb", [3, 5])
def test_parent_window_read_is_the_h5py_slice_on_a_four_device_mesh(tmp_path, nb):
    """The collective reader against direct h5py slices, element for element.

    Four emulated CPU devices on a 2x2 mesh (SlabIO refuses a one-process
    multi-GPU mesh; the P4 FFI tier is ``tests/multi_device/
    dipole_parent_window_p4.py``): nb = 3 pads to 4 band rows (rank 3 reads
    none), nb = 5 pads to 8 (rank 2 reads one row, rank 3 none).  The parents
    are unsorted, so the ascending-read permutation and its inverse run, and
    the band window starts at 1.  Everything outside the parent rows and the
    window is NaN, so a wrong row or a pad row leaking in cannot pass.
    """
    import jax
    from jax.sharding import Mesh
    from file_io.restart_bundle import read_dipole_parent_window

    devs = jax.devices()
    if len(devs) < 4 or devs[0].platform != "cpu":
        pytest.skip("needs four emulated CPU devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))
    nk, nb_file, b0 = 6, 8, 1
    rows = [5, 0, 3]
    rng = np.random.default_rng(nb)
    velocity = np.full((3, nk, nb_file, nb_file), np.nan + 1j * np.nan)
    for r in rows:
        velocity[:, r, b0:b0 + nb, b0:b0 + nb] = (
            rng.standard_normal((3, nb, nb))
            + 1j * rng.standard_normal((3, nb, nb)))
    path = tmp_path / "dipole.h5"
    with h5py.File(path, "w") as h5:
        h5["dipole_cart"] = velocity
    want = np.stack([velocity[:, r, b0:b0 + nb, b0:b0 + nb] for r in rows])
    got = read_dipole_parent_window(path, rows, b0, b0 + nb, nk_full=nk, mesh=mesh)
    assert got.dtype == np.complex128 and got.shape == (len(rows), 3, nb, nb)
    np.testing.assert_array_equal(got, want)
