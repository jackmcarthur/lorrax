"""Physical Nyquist labels, unique FFT support, and conditional restart identity."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from common.coulomb_sphere import compute_per_q_bare_coulomb_components
from common.gvec_fft_box import pad_gvecs_to_sentinel
from vcoul import (bare_coulomb_sphere_mask, bare_coulomb_sphere_indices,
                   bare_coulomb_sphere_rows, fft_box_miller)


B_AGI = .5115967582820649 * np.asarray([[-1., -1., 1.],
                                      [1., 1., 1.], [-1., 1., -1.]])


def _key(rows):
    return sorted(map(tuple, rows.tolist()))


@pytest.mark.parametrize('grid', [(50, 50, 50), (51, 51, 51)])
def test_legal_cross_k_plane_wave_has_the_correct_physical_label(grid):
    q = np.asarray([-1., 1., 1.]) / 3.
    G = np.asarray([25, 12, 12])
    assert np.sum(((q + G) @ B_AGI)**2) < 320.
    slot = np.ravel_multi_index(tuple(G % grid), grid)
    rows = bare_coulomb_sphere_rows(grid, B_AGI, q[None], 320.)
    where = np.flatnonzero(rows['idx_per_q'][0] == slot)
    assert len(where) == 1
    assert np.array_equal(rows['gvecs_per_q'][0][where[0]], G)
    if grid[0] == 50:
        old, oldG = bare_coulomb_sphere_mask(grid, B_AGI, q[None], 320.)
        assert np.array_equal(oldG[slot], [-25, 12, 12])
        assert not old[0, slot]
        assert rows['sphere_convention'] == 'q_aware_unique_fft_image_v1'
    else:
        assert rows['sphere_convention'] is None


@pytest.mark.parametrize('q', [[0., 0., 0.], [-1/3, 1/3, 1/3],
                                [1/3, -1/3, 0.5], [0.5, 0.5, 0.5]])
def test_actual_small_cutoff_is_bitwise_legacy_including_padding(q):
    q = np.asarray([q]);grid = (50, 50, 50)
    old, G = bare_coulomb_sphere_mask(grid, B_AGI, q, 80.)
    legacy = bare_coulomb_sphere_indices(grid, B_AGI, q, 80.)
    new = bare_coulomb_sphere_rows(grid, B_AGI, q, 80.)
    assert new['sphere_convention'] is None
    assert np.array_equal(new['idx_per_q'][0], legacy['idx_per_q'][0])
    assert np.array_equal(new['gvecs_per_q'][0], G[old[0]])
    padded, ng = pad_gvecs_to_sentinel([G[old[0]]], grid)
    actual = compute_per_q_bare_coulomb_components(grid, B_AGI, q, 80.)
    assert actual['sphere_convention'] is None
    assert np.array_equal(actual['gvec_components_padded'], padded.transpose(0, 2, 1))
    assert np.array_equal(actual['ngk_per_q'], ng)


@pytest.mark.parametrize('grid', [(8, 8, 8), (9, 9, 9), (8, 9, 10)])
def test_skew_lattice_exact_sphere_and_canonical_pair_reversal(grid):
    B = np.asarray([[1., 0., 0.], [.2, 1.1, 0.], [-.1, .1, 1.2]])
    qs = np.asarray([[0., 0., 0.], [-.5, 1/3, -.25], [.5, -1/3, .25]])
    # Radius is below every coordinate Nyquist, yet reaches an even face
    # on the smallest grid. Wide integer enumeration is independent.
    cutoff = 12.
    wide = np.stack(np.meshgrid(*[np.arange(-6, 7)]*3,indexing='ij'), -1).reshape(-1,3)
    rows = bare_coulomb_sphere_rows(grid, B, qs, cutoff)
    for q, G, slots in zip(qs, rows['gvecs_per_q'], rows['idx_per_q']):
        exact = wide[np.sum(((wide+q) @ B)**2, axis=1) <= cutoff]
        assert _key(G) == _key(exact)
        assert np.array_equal(slots, np.sort(slots))
        assert len(np.unique(slots)) == len(slots)
        assert np.array_equal(slots, np.ravel_multi_index((G % grid).T,grid))
    assert _key(-rows['gvecs_per_q'][1]) == _key(rows['gvecs_per_q'][2])


@pytest.mark.parametrize('cutoff', [16., 17.])
def test_ambiguous_support_is_an_explicit_implementation_refusal(cutoff):
    with pytest.raises(ValueError, match='GATE coulomb-sphere-unique-fft-image.*conservative'):
        bare_coulomb_sphere_rows((8, 8, 8), np.eye(3), [[0,0,0]], cutoff)
    assert bare_coulomb_sphere_rows((9, 9, 9), np.eye(3), [[0,0,0]], cutoff)['ngkmax'] > 0


@pytest.mark.parametrize('case', ['noncanonical', 'singular', 'nonfinite', 'cutoff'])
def test_geometry_refusals(case):
    B = np.eye(3);q = np.zeros((1,3));cutoff = 1.
    if case == 'noncanonical':q[0,0] = .6
    elif case == 'singular':B[1] = B[0]
    elif case == 'nonfinite':B[0,0] = np.nan
    else:cutoff = np.inf
    with pytest.raises(ValueError):
        bare_coulomb_sphere_rows((8,8,8), B, q, cutoff)


def test_positive_nyquist_keeps_the_existing_sentinel_collision_guard():
    with pytest.raises(ValueError, match='PHYSICAL.*sentinel'):
        pad_gvecs_to_sentinel([np.asarray([[3,3,3]]),
                              np.asarray([[0,0,0],[1,0,0]])], (6,6,6))


def _provenance(cfg, marker=None):
    from gw import gw_init
    return gw_init._zeta_fit_provenance(
        wfn=SimpleNamespace(_filename='', ecutwfc=80., ecutrho=320.),
        meta=SimpleNamespace(n_rmu=4,nspinor_wfnfile=2,fft_grid=np.array([8,8,8]),
                             current_basis_rows=None),
        cfg=cfg,band_range_left=(0,4),band_range_right=(0,8),logical_band_stop=7,
        zeta_cutoff=320.,zeta_vcoul_cutoff=320.,write_ibz_only=True,band_norms=None,
        fft_sphere_convention=marker)


def test_only_changed_physical_tables_change_restart_identity(tmp_path, monkeypatch):
    from gw import gw_init
    from gw.gw_config import LorraxConfig
    import file_io.restart_bundle as restart
    import common.parallel_transport as transport
    deck = tmp_path / 'gw.in'
    deck.write_text('[cohsex]\nnval=1\nncond=2\nnumber_bands=7\n'
                    'compute_mode=x_only\nqp_solver=one_shot_dft\nlinalg=local\n'
                    'sys_dim=3\nbispinor=true\nbispinor_gw=coulomb_only\n')
    cfg = LorraxConfig.from_input_file(str(deck),resolve_hardware=False,print_fn=lambda *_:None)
    old = _provenance(cfg);new = _provenance(cfg,'q_aware_unique_fft_image_v1')
    assert 'fft_sphere_convention' not in json.loads(old)
    assert old == _provenance(cfg,None)
    difference = json.loads(new);difference.pop('fft_sphere_convention')
    assert difference == json.loads(old)
    cents = np.arange(12,dtype=np.int32).reshape(4,3);path=tmp_path/'zeta.h5';path.touch();holder={'stamp':old}
    monkeypatch.setattr(restart,'read_isdf_header',lambda _:SimpleNamespace(
        zeta_is_done=True,fit_provenance=holder['stamp'],r_mu_fft_idx=cents,
        coordinate_kind='fft_indices',centroid_coordinates=cents))
    monkeypatch.delenv('LORRAX_FORCE_REFIT',raising=False)
    monkeypatch.setattr(transport,'wfn_fingerprint',lambda _:'fixed-source')
    assert gw_init._zeta_reuse_ok(str(path),old,cents,print_fn=lambda *_:None)
    assert not gw_init._zeta_reuse_ok(str(path),new,cents,print_fn=lambda *_:None)
    assert gw_init.charge_zeta_identity(old,wfn=object()) != gw_init.charge_zeta_identity(new,wfn=object())
    holder['stamp']=new
    assert gw_init._zeta_reuse_ok(str(path),new,cents,print_fn=lambda *_:None)
    assert not gw_init._zeta_reuse_ok(str(path),old,cents,print_fn=lambda *_:None)
    with pytest.raises(ValueError,match='unknown physical FFT-sphere'):_provenance(cfg,'unknown')


def test_provenance_uses_this_channels_actual_q_domain(monkeypatch):
    from gw import gw_init
    import gw.v_q_g_flat as qowner
    import common.coulomb_sphere as sphere
    import vcoul
    seen=[];sym=SimpleNamespace(q_irr_full_idx=np.asarray([0]));basis=SimpleNamespace(coordinate_kind='fractional')
    meta=SimpleNamespace(sys_dim=3,kgrid=(2,2,2),fft_grid=(8,8,8),mu_basis=basis)
    def resolve(**kwargs):
        seen.append(kwargs)
        return (None, np.asarray([[0,0,0]] if kwargs['sym'] is sym else [[-.5,0,0]]))
    monkeypatch.setattr(qowner,'_resolve_ibz_q_list',resolve)
    monkeypatch.setattr(vcoul.CoulombGeometry,'from_wfn',lambda _:SimpleNamespace(bvec=np.eye(3)))
    def produce(grid,B,q,cutoff,**kwargs):
        return {'sphere_convention':None if not np.any(q) else 'q_aware_unique_fft_image_v1'}
    monkeypatch.setattr(sphere,'compute_per_q_bare_coulomb_components',produce)
    assert gw_init._zeta_sphere_convention(object(),sym,meta,np.zeros((1,3)),1.,write_ibz_only=True) is None
    assert gw_init._zeta_sphere_convention(object(),sym,meta,np.zeros((1,3)),1.,write_ibz_only=False) == 'q_aware_unique_fft_image_v1'
    assert seen[0]['sym'] is sym and seen[1]['sym'] is None
    assert all(x['coordinate_kind']=='fractional' for x in seen)
