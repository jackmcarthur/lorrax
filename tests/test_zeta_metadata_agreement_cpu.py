"""Real fitting orchestration agrees metadata failures before its next stage."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize('failure', [None, 'copy', 'header', 'complete'])
def test_fit_metadata_uses_the_shared_transaction_owner(monkeypatch, tmp_path, failure):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from common import collectives
    from file_io import isdf_header, mf_header, slab_io
    from gw import isdf_fitting as fit
    from common import coulomb_sphere
    import distrib_la
    from vcoul import CoulombGeometry

    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.array(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    target = tmp_path/'zeta.h5'
    events, agreements = [], []
    original_header = isdf_header.write_isdf_header
    original_complete = isdf_header.mark_zeta_done
    original_agree = collectives.agree_io_error

    class Transport:
        def __init__(self, path, *, mode, mesh):
            assert mode == 'w'
            with h5py.File(path, 'w'):
                pass
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            events.append('transport_closed')

    def copy(_source, path, *, dst_mode):
        assert events[-1] == 'transport_closed' and dst_mode == 'a'
        events.append('copy')
        if failure == 'copy':
            raise OSError('injected source header copy failure')
        with h5py.File(path, 'a') as f:
            f.create_group('mf_header')

    def header(*a, **k):
        events.append('header')
        if failure == 'header':
            raise OSError('injected ISDF header failure')
        return original_header(*a, **k)

    def complete(*a, **k):
        events.append('complete')
        if failure == 'complete':
            raise OSError('injected completion failure')
        return original_complete(*a, **k)

    def agree(error, *, path, stage):
        agreements.append((stage, error))
        return original_agree(error, path=path, stage=stage)

    def batches(**_k):
        events.append('batch')
        return {1: SimpleNamespace(pending_file=None)}, 1, 1

    monkeypatch.setattr(slab_io, 'SlabIO', Transport)
    monkeypatch.setattr(mf_header, 'copy_mf_header', copy)
    monkeypatch.setattr(isdf_header, 'write_isdf_header', header)
    monkeypatch.setattr(isdf_header, 'mark_zeta_done', complete)
    monkeypatch.setattr(collectives, 'agree_io_error', agree)
    monkeypatch.setattr(distrib_la, 'gemm_plan', lambda *_a, **_k:
        SimpleNamespace(describe=lambda: 'metadata fixture'))
    monkeypatch.setattr(CoulombGeometry, 'from_wfn', lambda _w:
        SimpleNamespace(bvec=np.eye(3)))
    monkeypatch.setattr(coulomb_sphere, 'compute_per_q_bare_coulomb_components', lambda **_k:
        dict(sphere_idx_padded=np.zeros((1, 1), np.int32),
             gvec_components_padded=np.zeros((1, 3, 1), np.int32),
             ngk_per_q=np.ones(1, np.int32), ngkmax=1))
    monkeypatch.setattr(fit, 'c_q_from_psi_sm', lambda **_k: jnp.eye(4)[None].astype(complex))
    monkeypatch.setattr(fit, 'factor_c_q', lambda c, *_a, **_k:
        (c, jnp.zeros((1, 4), jnp.int32)))
    monkeypatch.setattr(fit, 'zeta_factor_resident', lambda a, b, *_a, **_k: (a, b))
    monkeypatch.setattr(fit, '_fit_mubatch', batches)
    meta = SimpleNamespace(n_rmu=4, n_rmu_padded=4, n_rtot=8, nk_tot=1,
        kgrid=(1, 1, 1), fft_grid=(2, 2, 2), sys_dim=3,
        b_id_0=0, b_id_3=1, b_id_4=2)
    kwargs = dict(wfn=SimpleNamespace(_filename='fixture WFN'),
        sym=SimpleNamespace(kvecs_asints=np.zeros((1, 3), np.int32)), meta=meta,
        centroid_indices=np.zeros((4, 3), np.int32), mesh_xy=mesh,
        output_files={1: str(target)}, k_unfold_plan=SimpleNamespace(
            n_parent=1, n_centroid_packed=4), psi_mun_parent=jnp.zeros((1, 4, 4, 2)),
        psi_nmu_parent=jnp.zeros((1, 2, 4, 4)), mubatch_plan=object(),
        write_ibz_only=False, zeta_cutoff_ry=1.)
    if failure is None:
        fit.fit_zeta_to_h5(**kwargs)
        assert [s for s, _ in agreements] == ['zeta_fit_headers', 'zeta_fit_complete']
        assert all(error is None for _, error in agreements)
        with h5py.File(target, 'r') as f:
            assert bool(f['isdf_header/zeta_is_done'][()])
    else:
        stage = 'zeta_fit_complete' if failure == 'complete' else 'zeta_fit_headers'
        with pytest.raises(RuntimeError, match='GATE io_global_commit:.*stage='+stage):
            fit.fit_zeta_to_h5(**kwargs)
        assert agreements[-1][0] == stage and isinstance(agreements[-1][1], OSError)
        assert ('batch' in events) == (failure == 'complete')
        with h5py.File(target, 'r') as f:
            assert ('isdf_header/zeta_is_done' not in f
                    or not bool(f['isdf_header/zeta_is_done'][()]))
