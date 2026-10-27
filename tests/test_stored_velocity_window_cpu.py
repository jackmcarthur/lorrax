"""A large logical dipole file is read only on the guarded fit window."""
from types import SimpleNamespace

import runtime
import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from bandstructure.orbital import stored_velocity
from file_io import dipole
from file_io.commit_state import set_commit_state
from file_io.slab_io import SlabIO

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('side', (2, 3))
def test_stored_velocity_reads_fit_before_gather_not_whole_file(tmp_path, monkeypatch, side):
    if jax.device_count() < side*side:
        pytest.skip('requires four or nine virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    path = tmp_path/'large-logical-dipole.h5'
    nb, lo, hi, nk = 20000, 10001, 10006, 2
    i, j = np.arange(lo, hi)[:, None], np.arange(lo, hi)[None, :]
    field = np.stack([np.stack([k+.1*a+.001*i+.00001*j
                               +1j*(.3+.02*a+.0001*i-.0003*j)
                               for k in range(nk)]) for a in range(3)])
    energy = np.stack([np.arange(nb)*.1+.01*k for k in range(nk)])
    with h5py.File(path, 'w') as f:
        f.attrs['format_version'] = 2
        # Sparse storage: the fixture never allocates or writes the full field.
        ds = f.create_dataset('dipole_cart', shape=(3, nk, nb, nb),
            chunks=(1, 1, 5, 5), dtype=np.complex128, fillvalue=99+17j)
        ds[:, :, lo:hi, lo:hi] = field
        set_commit_state(f, True)
    assert path.stat().st_size < 1 << 20
    # Authentication has separate real-file controls. Stub only its metadata
    # here; the actual format reader, SlabIO and parent gather remain intact.
    monkeypatch.setattr(dipole, 'velocity_stamp', lambda *a, **k: ('v_DFT', 'dft', energy))
    sym = SimpleNamespace(kirr_fullids=np.array([1, 0]), nk_tot=nk)
    reads = []
    original = SlabIO.read_slabs

    def read(self, name, **kw):
        assert name == 'dipole_cart'
        assert kw['shape'][-1] == hi-lo and kw['shape'][-2] <= hi-lo+side*side-1
        np.testing.assert_array_equal(np.asarray(kw['offsets'])[:, 2:], lo)
        np.testing.assert_array_equal(np.asarray(kw['valid_shapes'])[:, 2:], hi-lo)
        reads.append(kw['shape'])
        return original(self, name, **kw)

    monkeypatch.setattr(SlabIO, 'read_slabs', read)
    parents, energies, label = stored_velocity(path, wfn=None, wfn_path='metadata-fixture',
        sym=sym, mesh=mesh, band_range=(lo, hi))
    assert reads and label == 'v_DFT'
    np.testing.assert_array_equal(parents, np.moveaxis(field[:, [1, 0]], 0, 1))
    # Edge weighting still sees the first unfitted band; only the square
    # operator payload is narrowed, not its inexpensive energy metadata.
    np.testing.assert_array_equal(energies, energy[[1, 0]])
    assert energies[0, hi] == energy[1, hi]


def test_fitted_operator_absolute_origin_preserves_interpolated_dipoles():
    from tests.test_bse_kgrid_dipoles_cpu import (
        _model, _states, _grid, _densify, _exact_rotated, _rel, KC)
    # This model has ten physical orbital nodes. P1 is its legal rank carrier;
    # the separate real-reader controls exercise distributed P4/P9 I/O.
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    h = _model()
    c, e, v = _states(h, _grid(KC))
    ordinary, fine = _densify(mesh, c, e, v)
    shifted, _ = _densify(mesh, c, e, v, band_offset=10001)
    np.testing.assert_allclose(shifted['d_alpha'], ordinary['d_alpha'], atol=1e-13, rtol=1e-12)
    np.testing.assert_array_equal(shifted['de_cv'], ordinary['de_cv'])
    reference, _ = _exact_rotated(h, fine)
    assert _rel(shifted['d_alpha'], reference) < .02
    assert shifted['on_grid_rel'] < 1e-12

