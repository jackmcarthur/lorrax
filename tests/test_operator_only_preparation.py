"""Fresh screened-operator scope and full-band reconstruction selection."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _config():
    from gw.gw_config import BispinorGWMode, HeadCorrection, QPSolver, ScreeningDiagrams
    return SimpleNamespace(restart=False, qp_solver=QPSolver.ONE_SHOT_DFT,
        density_self_consistent=False, bispinor=True,
        bispinor_gw=BispinorGWMode.COULOMB_ONLY,
        head=SimpleNamespace(correction=HeadCorrection.OFF),
        occ_smearing_width_ry=None, sigma=SimpleNamespace(w_model='shared_pole'),
        screening=SimpleNamespace(occ_broadening_ev=0., diagrams=ScreeningDiagrams.W_RPA))


def test_fresh_scalar_operator_scope_accepts_pauli_and_rkb_carriers():
    from gw.gw_init import _validate_operator_only_preparation
    cfg = _config()
    _validate_operator_only_preparation(cfg, True)
    cfg.bispinor = False
    _validate_operator_only_preparation(cfg, True)
    # The normal public call retains its historical downstream admission.
    _validate_operator_only_preparation(object(), False)


@pytest.mark.parametrize('change', ['restart', 'SC', 'density', 'head',
    'current', 'smearing', 'broadening'])
def test_unproved_operator_scope_refuses_before_any_source_access(change):
    from gw.gw_config import BispinorGWMode, HeadCorrection, QPSolver
    from gw.gw_init import prepare_isdf_and_wavefunctions
    cfg = _config()
    if change == 'restart':
        cfg.restart = True
    elif change == 'SC':
        cfg.qp_solver = QPSolver.SELF_CONSISTENT
    elif change == 'density':
        cfg.density_self_consistent = True
    elif change == 'head':
        cfg.head.correction = HeadCorrection.FULL
    elif change == 'current':
        cfg.bispinor_gw = BispinorGWMode.BARE_TRANSVERSE
    elif change == 'smearing':
        cfg.occ_smearing_width_ry = .01
    else:
        cfg.screening.occ_broadening_ev = .01
    class ForbiddenSource:
        def __getattribute__(self, name):
            raise AssertionError('Source accessed before scope admission: '+name)
    with pytest.raises(ValueError, match='operator_only_preparation'):
        prepare_isdf_and_wavefunctions(cfg=cfg, wfn=ForbiddenSource(), sym=None,
            meta=None, centroid_indices=None, band_slices=None, mesh_xy=None,
            tmp_dir=None, tensors_filename=None, print0=lambda *a: None,
            operator_only=True)


def test_scope_requires_an_explicit_boolean():
    from gw.gw_init import _validate_operator_only_preparation
    with pytest.raises(TypeError, match='explicit boolean'):
        _validate_operator_only_preparation(_config(), 1)


@pytest.mark.parametrize('operator_only', [False, True])
def test_public_preparation_forwards_scope_to_the_fresh_owner(monkeypatch, operator_only):
    from gw import gw_init
    from gw.gw_config import ComputeMode
    cfg = _config()
    cfg.paths = SimpleNamespace(atomic_reconstruction_dir=None)
    cfg.compute_mode = ComputeMode.MPA
    cfg.sys_dim = 2
    reached = []
    class OwnerReached(Exception):
        pass
    def stop_before_io(*args, **kwargs):
        reached.append(kwargs['operator_only'])
        raise OwnerReached
    monkeypatch.setattr(gw_init, '_prepare_fresh_isdf', stop_before_io)
    with pytest.raises(OwnerReached):
        gw_init.prepare_isdf_and_wavefunctions(cfg=cfg, wfn=object(), sym=None,
            meta=SimpleNamespace(b_id_4_user=198, mu_basis=None),
            centroid_indices=None, band_slices=SimpleNamespace(b0=0), mesh_xy=None,
            tmp_dir=None, tensors_filename=None, print0=lambda *a: None,
            operator_only=operator_only)
    assert reached == [operator_only]


@pytest.mark.parametrize('layout', ['source', 'face', 'coefficients'])
def test_full_band_rotation_precedes_window_selection_and_zeros_ghosts(layout):
    from gw.isdf_augmentation import _band_rotation_kernel
    from psp.reconstruction_overlap import rotate_band_rows
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    specs = dict(source=P(None,None,None,('x','y')),
        face=P(None,'x',None,'y'), coefficients=P(None,('x','y'),None))
    rng = np.random.default_rng(159)
    shape = (2, 8, 4, 5) if layout != 'coefficients' else (2, 8, 7)
    values = rng.normal(size=shape)+1j*rng.normal(size=shape)
    mixing = rng.normal(size=(2,8,8))+1j*rng.normal(size=(2,8,8))
    factor = np.eye(8)[None]+.04*(mixing+mixing.conj().transpose(0,2,1))
    with mesh:
        rows = jax.device_put(jnp.asarray(values), NamedSharding(mesh, specs[layout]))
        A = jax.device_put(jnp.asarray(factor), NamedSharding(mesh, P()))
        actual = _band_rotation_kernel(mesh, 1, layout, 8, 0, 6)(rows, A)
        full = _band_rotation_kernel(mesh, 1, layout, 8, 0, 8)(rows, A)
        expected = np.asarray(rotate_band_rows(rows, A, band_axis=1))
        actual = np.asarray(actual)
        np.testing.assert_array_equal(actual[:,:6], np.asarray(full)[:,:6])
        np.testing.assert_allclose(actual[:,:6], expected[:,:6], atol=2e-15, rtol=2e-15)
        assert not np.any(actual[:,6:])
        prematurely_cropped = np.asarray(rotate_band_rows(jnp.asarray(values[:,:6]),
            jnp.asarray(factor[:,:6,:6]), band_axis=1))
        assert np.max(abs(actual[:,:6]-prematurely_cropped)) > .01
