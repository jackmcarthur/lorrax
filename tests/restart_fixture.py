"""Canonical parent bundles for small restart-reader transport tests."""
import h5py
import jax
import numpy as np
from jax.sharding import NamedSharding


def canonicalize_fixture(path):
    """Express a synthetic full-grid fixture as explicit identity parent rows."""
    from file_io.tagged_arrays import BAND_WINDOW_SCHEMA_VERSION
    with h5py.File(path, 'r+') as f:
        for old, new in [('psi_full_y', 'psi_parent_y'),
                         ('psi_full_y_transverse', 'psi_parent_y_transverse')]:
            if old in f:
                psi = np.asarray(f[old])
                del f[old]
                f[new] = psi
                f[new + '_mun'] = psi.transpose(0, 2, 3, 1)
        if 'psi_parent_y' not in f:
            nb = int(f['band_window'][-1]) if 'band_window' in f else 1
            mu = int(f['V_qmunu'].shape[-1]) if 'V_qmunu' in f else 1
            f['psi_parent_y'] = np.zeros((1, nb, 1, mu), complex)
        psi = f['psi_parent_y']
        nk, nb, _, mu = psi.shape
        if 'psi_parent_y_mun' not in f:
            f['psi_parent_y_mun'] = np.asarray(psi).transpose(0, 2, 3, 1)
        defaults = dict(psi_parent_k_rows=np.arange(nk),
                        band_window=np.array([0, 0, 1, nb, nb]),
                        band_window_schema=BAND_WINDOW_SCHEMA_VERSION,
                        enk_full=np.zeros((nk, nb)), kgrid=np.array([nk, 1, 1]),
                        n_rmu_logical=mu)
        for name, value in defaults.items():
            if name not in f:
                f[name] = value
        if 'band_window_carrier' not in f:
            f['band_window_carrier'] = np.asarray(f['band_window'])
        if 'band_window_split' not in f:
            f['band_window_split'] = np.repeat(f['band_window'][-1], 2)
        for name in ('V_qmunu', 'W0_qmunu', 'S_qmunu'):
            if name in f and f[name].ndim > 3:
                a, attrs = np.asarray(f[name]), dict(f[name].attrs)
                del f[name]
                f[name] = a.reshape(-1, mu, mu)
                f[name].attrs.update(attrs)


def identity_parent_transport(monkeypatch):
    """Replace only transport and symmetry for synthetic identity-parent data."""
    from file_io import restart_bundle, slab_io

    class HostSlabIO:
        def __init__(self, path, *, mode='r', mesh=None):
            self.f = h5py.File(path, mode)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            self.f.close()
        def read_slab(self, name, *, shape, dtype, offset=None, mesh,
                      partition_spec):
            off = offset or (0,) * len(shape)
            src = np.asarray(self.f[name][tuple(
                slice(o, o+n) for o, n in zip(off, shape))])
            out = np.zeros(shape, dtype=dtype)
            out[tuple(slice(0, n) for n in src.shape)] = src
            return jax.device_put(out, NamedSharding(mesh, partition_spec))

    monkeypatch.setattr(slab_io, 'SlabIO', HostSlabIO)
    monkeypatch.setattr(restart_bundle, 'unfold_parent_faces',
                        lambda faces, *args, **kwargs: faces)
