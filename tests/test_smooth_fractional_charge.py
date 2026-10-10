"""Signed-G smooth samples and their explicitly declared scalar fit metric."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _config(tmp_path, extra=''):
    from gw.gw_config import LorraxConfig
    deck = tmp_path / 'cohsex.in'
    deck.write_text('[cohsex]\nnval=1\nncond=2\nnumber_bands=7\n'
        'qp_solver=one_shot_dft\nlinalg=local\ncompute_mode=x_only\n'
        'sys_dim=3\nbispinor=true\nbispinor_gw=coulomb_only\nhead_correction=off\n' + extra)
    return LorraxConfig.from_input_file(str(deck), resolve_hardware=False,
                                      print_fn=lambda *_: None)


def _provenance(cfg, conditioning=None):
    from gw.gw_init import _zeta_fit_provenance
    return _zeta_fit_provenance(
        wfn=SimpleNamespace(_filename='', ecutwfc=80., ecutrho=320.),
        meta=SimpleNamespace(n_rmu=4, nspinor_wfnfile=2,
            fft_grid=np.array([8,8,8]), current_basis_rows=None),
        cfg=cfg, band_range_left=(0,4), band_range_right=(0,8),
        logical_band_stop=7, zeta_cutoff=80., zeta_vcoul_cutoff=80.,
        write_ibz_only=True, band_norms=None,
        charge_fit_conditioning=conditioning)


@pytest.mark.parametrize('value,expected', [('none', None), ('unit_diagonal', 'unit_diagonal')])
def test_parsed_metric_and_canonical_provenance(tmp_path, value, expected):
    from gw.gw_init import _resolve_charge_fit_conditioning
    cfg = _config(tmp_path, 'charge_fit_conditioning='+value+'\n')
    assert cfg.backend.charge_fit_conditioning == expected
    assert _resolve_charge_fit_conditioning(cfg) == expected
    stamp = json.loads(_provenance(cfg))
    if expected is None:
        assert 'charge_fit_conditioning' not in stamp
        assert _provenance(cfg) == _provenance(_config(tmp_path))
    else:
        assert stamp['charge_fit_conditioning'] == expected
        assert 'atomic_augmentation_identity' not in stamp


@pytest.mark.parametrize('value', ['true', 'false', '1', 'equilibrated', 'nan'])
def test_invalid_metric_input_refuses(tmp_path, value):
    with pytest.raises(ValueError, match='charge_fit_conditioning'):
        _config(tmp_path, 'charge_fit_conditioning='+value+'\n')


def test_metric_resolves_manifest_once_and_refuses_invalid_or_current(tmp_path):
    from gw.gw_init import _resolve_charge_fit_conditioning
    from gw.gw_config import BispinorGWMode
    cfg = _config(tmp_path)
    manifest = {'charge_fit': {'conditioning': 'unit_diagonal'}}
    assert _resolve_charge_fit_conditioning(cfg, manifest) == 'unit_diagonal'
    explicit = _config(tmp_path, 'charge_fit_conditioning=unit_diagonal\n')
    assert _resolve_charge_fit_conditioning(explicit, manifest) == 'unit_diagonal'
    assert json.loads(_provenance(cfg, 'unit_diagonal'))['charge_fit_conditioning'] == 'unit_diagonal'
    with pytest.raises(ValueError, match='charge_fit_conditioning'):
        _resolve_charge_fit_conditioning(explicit, {'charge_fit': {'conditioning': 'other'}})
    current = SimpleNamespace(**cfg.__dict__)
    current.bispinor_gw = BispinorGWMode.BARE_TRANSVERSE
    current.backend = explicit.backend
    with pytest.raises(ValueError, match='only scalar charge'):
        _resolve_charge_fit_conditioning(current, manifest)


@pytest.mark.parametrize('stored,requested,accepted', [
    (None,None,True), ('unit_diagonal','unit_diagonal',True),
    (None,'unit_diagonal',False), ('unit_diagonal',None,False)])
def test_restart_metric_has_authenticated_exact_join(tmp_path, monkeypatch, stored, requested, accepted):
    from gw import gw_init
    import file_io.restart_bundle as restart
    stamp = _provenance(_config(tmp_path), stored)
    monkeypatch.setattr(restart, 'read_charge_zeta_provenance', lambda _: {
        'charge_zeta_identity': 'bound', 'charge_zeta_provenance': stamp})
    monkeypatch.setattr(gw_init, 'charge_zeta_identity', lambda *a, **kw: 'bound')
    call = lambda: gw_init._require_restart_charge_fit_weights('unused', None,
        wfn=object(), wfn_fingerprint_binding='unchanged', tmp_dir=str(tmp_path),
        expected_conditioning=requested)
    if accepted:
        call()
    else:
        with pytest.raises(ValueError, match='restart_charge_fit_conditioning'):
            call()


def test_legacy_restart_cannot_inherit_an_unmarked_metric(tmp_path, monkeypatch):
    from gw import gw_init
    import file_io.restart_bundle as restart
    monkeypatch.setattr(restart, 'read_charge_zeta_provenance', lambda _: {
        'charge_zeta_identity': None, 'charge_zeta_provenance': None})
    kwargs = dict(wfn=object(), wfn_fingerprint_binding='same', tmp_dir=str(tmp_path))
    gw_init._require_restart_charge_fit_weights('unused', None, **kwargs)
    with pytest.raises(ValueError, match='legacy restart'):
        gw_init._require_restart_charge_fit_weights('unused', None,
            expected_conditioning='unit_diagonal', **kwargs)


@pytest.mark.parametrize('change', ['restart', 'SC', 'density', 'head', 'current', 'smearing', 'broadening'])
def test_fractional_scope_guard_precedes_source_access(tmp_path, change):
    from gw.gw_config import BispinorGWMode, HeadCorrection, QPSolver
    from gw.gw_init import prepare_band_metadata
    cfg = SimpleNamespace(**_config(tmp_path).__dict__)
    cfg.qp_solver = QPSolver.ONE_SHOT_DFT
    if change == 'restart': cfg.restart = True
    elif change == 'SC': cfg.qp_solver = QPSolver.SELF_CONSISTENT
    elif change == 'density': cfg.density_self_consistent = True
    elif change == 'head': cfg.head = SimpleNamespace(correction=HeadCorrection.FULL)
    elif change == 'current': cfg.bispinor_gw = BispinorGWMode.BARE_TRANSVERSE
    elif change == 'smearing': cfg.occ_smearing_width_ry = .01
    else: cfg.screening = SimpleNamespace(occ_broadening_ev=.01)
    class ForbiddenSource:
        def __getattribute__(self, name):
            raise AssertionError('Source accessed before fractional admission: '+name)
    with pytest.raises(ValueError, match='smooth_fractional_charge'):
        prepare_band_metadata(None, cfg, None, 0, lambda *_:None, None,
            ForbiddenSource(), coordinate_kind='fractional')


def test_admitted_smooth_fractional_scope_needs_no_atomic_manifest(tmp_path):
    from gw.gw_init import _validate_smooth_fractional_charge
    cfg = _config(tmp_path)
    assert cfg.paths.atomic_reconstruction_dir is None
    _validate_smooth_fractional_charge(cfg)


@pytest.mark.parametrize('ns', [2,4])
def test_signed_miller_sampling_matches_literal_nongamma_field_and_ghosts(ns):
    from common.psi_G_store import _gslot_face_kernel, _gslot_faces_kernel
    from common.wfn_layout import PSI_NMU_ACC_SPEC, PSI_MUNT_ACC_SPEC
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1,1), ('x','y'))
    grid = (8,10,12); nk=2; nb=4; ng=6; nm=6
    G = np.array([[0,0,0],[-1,2,-3],[2,-3,1],[-3,1,4],[3,-4,-5],[71,82,93]], np.int32)
    geometry = np.broadcast_to(np.c_[G, [1,1,1,1,1,0]], (nk,ng,4)).copy()
    k = np.array([[0,0,0],[1/3,-1/3,0.]])
    r = np.array([[.123,.317,.731],[.019,.815,.415],[.515,.275,.177],
                  [.375,.2,.25],[.8,.1,.7],[.99,.99,.99]])
    active = np.array([1,1,1,1,1,0.])
    rng = np.random.default_rng(411+ns)
    coeff = rng.normal(size=(nk,nb,ns,ng))+1j*rng.normal(size=(nk,nb,ns,ng))
    coeff[:,:,-1,-1] = 19.+23j  # invalid G is observably populated
    def run(kind, coordinates, geom):
        with mesh:
            put = lambda a,s: jax.device_put(jnp.asarray(a), NamedSharding(mesh,s))
            z = np.zeros((nk,nb,ns,nm), np.complex128)
            _, y, x = _gslot_face_kernel(mesh,grid,nk,nb,ns,ng,nm,2,True,
                coordinate_kind=kind)(put(coeff,P(None,('x','y'),None,None)),
                put(geom,P()),put(k,P()),put(coordinates,P()),put(active,P()),
                put(z,PSI_NMU_ACC_SPEC),put(z,PSI_MUNT_ACC_SPEC),jnp.int32(0))
            y,x = _gslot_faces_kernel(mesh)(y,x)
            return np.asarray(y),np.asarray(x)
    actual, companion = run('fractional',r,geometry)
    truth = np.einsum('knsg,kgm->knsm',coeff[...,:5],
        np.exp(2j*np.pi*np.einsum('kgi,mi->kgm',k[:,None]+G[None,:5],r)))
    truth *= active[None,None,None,:]/np.sqrt(np.prod(grid))
    np.testing.assert_allclose(actual,truth,atol=4e-15,rtol=4e-14)
    np.testing.assert_array_equal(companion,actual.conj().transpose(0,3,1,2))
    assert not np.any(actual[...,-1])
    def wrong_field(Gwrong,rwrong,kfactor=True):
        phase = np.einsum('gi,mi->gm',Gwrong[:5],rwrong)
        phase = phase[None]+(np.einsum('ki,mi->km',k,rwrong)[:,None] if kfactor else 0)
        return np.einsum('knsg,kgm->knsm',coeff[...,:5],np.exp(2j*np.pi*phase)) / np.sqrt(np.prod(grid))
    for wrong in (wrong_field(G % grid,r),wrong_field(G,np.rint(r*grid)/grid),wrong_field(G,r,False)):
        assert np.max(abs(actual[...,:5]-wrong[...,:5])) > .01
    # Grid points reduce to the unchanged FFT residue path with the same k.
    ri = np.array([[1,2,3],[0,8,5],[4,3,2],[3,2,3],[6,1,8],[7,9,11]],np.int32)
    box = np.ravel_multi_index((G[:5] % grid).T,grid)
    slots = np.broadcast_to(np.r_[box,np.prod(grid)],(nk,ng)).copy()
    old,_ = run('fft_indices',ri,slots)
    exact,_ = run('fractional',ri/np.asarray(grid),geometry)
    np.testing.assert_allclose(old,exact,atol=4e-15,rtol=4e-14)
