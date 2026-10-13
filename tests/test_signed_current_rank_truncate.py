"""Physical channel signs, null-space projection and paired OWN solves."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_current_sign_comes_from_actual_cartesian_and_circular_matrices():
    from isdf.core import current_fit_solver_kind
    from symmetry_maps import CIRCULAR_CURRENT_ROWS
    assert [current_fit_solver_kind(c,None) for c in (1,2,3)] == [
        'positive_rank_truncate','negative_rank_truncate','positive_rank_truncate']
    assert [current_fit_solver_kind(c,CIRCULAR_CURRENT_ROWS) for c in (1,2,3)] == [
        'positive_rank_truncate']*3
    with pytest.raises(ValueError,match='real or imaginary'):
        current_fit_solver_kind(1,np.exp(1j*np.pi/4)*CIRCULAR_CURRENT_ROWS)


@pytest.mark.parametrize('sign,kind',[(1,'positive_rank_truncate'),(-1,'negative_rank_truncate')])
def test_signed_psd_factor_removes_null_modes_with_same_PW_and_local_owner(sign,kind):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from isdf.core import factor_c_q,zeta_factor_resident
    from isdf.zeta_mubatch import _logical_solve,_local_coefficient_solve
    mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    rng=np.random.default_rng(139);n=6;pad=8;Q=2
    V=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
    lam=np.asarray([0.,1e-12,1e-4,.02,.3,1.])
    positive=(V*lam)@V.conj().T
    C=np.full((Q,pad,pad),2e7+5e7j,dtype=complex)
    C[:,:n,:n]=sign*positive
    put=lambda value,spec:jax.device_put(value,NamedSharding(mesh,spec))
    B=factor_c_q(put(C,P(None,'x','y')),mesh,vertex_mu_L=2,
                 n_rmu_logical=n,solver_kind=kind,zeta_rcond=1e-8)
    B,_=zeta_factor_resident(B,None,mesh,solver_kind=kind)
    rhs=rng.normal(size=(Q,pad,7))+1j*rng.normal(size=(Q,pad,7))
    inverse=sign*(V[:,2:]/lam[2:])@V[:,2:].conj().T
    expected=np.einsum('mn,qnf->qmf',inverse,rhs[:,:n])
    # The same canonical solve is used on the PW tile and both local RHSs.
    pw=jax.jit(jax.vmap(_logical_solve(kind,n)))(B,put(rhs,P(('x','y'),None,None)))
    local=_local_coefficient_solve(mesh,kind,n,Q)(B,put(rhs,P(('x','y'),None,None)))
    np.testing.assert_allclose(np.asarray(pw)[:,:n],expected,rtol=3e-11,atol=3e-9)
    np.testing.assert_array_equal(np.asarray(local),np.asarray(pw))
    assert not np.asarray(local)[:,n:].any()
    projected=np.einsum('mn,qnf->qmf',V[:,2:]@V[:,2:].conj().T,rhs[:,:n])
    np.testing.assert_allclose(np.einsum('mn,qnf->qmf',sign*positive,np.asarray(local)[:,:n]),
                               projected,rtol=3e-11,atol=3e-11)


@pytest.mark.parametrize('defect',['negative','skew','zero'])
def test_signed_current_rejects_non_PSD_source_instead_of_dropping_it(defect):
    import jax
    import jax.numpy as jnp
    from isdf import cplus
    from common import rank_criterion
    C=np.diag([1.,.2,.03,.004]).astype(complex)
    if defect=='negative':C[3,3]=-.1
    elif defect=='skew':C[0,1]=.01j
    else:C[:]=0
    rank_criterion.raise_if_pending()
    B=jax.jit(lambda c:cplus.factor(c,rcond=1e-8,rank_log=False,n_log=4,
                                    require_psd=True))(jnp.asarray(C[None]))
    B.block_until_ready();jax.effects_barrier()
    with pytest.raises(rank_criterion.RankPolicyError,match='signed current PSD'):
        rank_criterion.raise_if_pending()


def test_explicit_LU_keeps_its_original_factor_and_signed_ridge():
    import jax
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from isdf.core import factor_c_q,_transverse_lu_math
    mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
    C=-np.diag([1.,.2,.03,.004]).astype(complex)[None]
    array=jax.device_put(C,NamedSharding(mesh,P(None,'x','y')))
    lu,piv=factor_c_q(array,mesh,vertex_mu_L=2,solver_kind='lu')
    expected=jax.jit(jax.vmap(lambda c:_transverse_lu_math(c,4)))(array)
    np.testing.assert_array_equal(np.asarray(lu),np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(piv),np.asarray(expected[1]))


def test_new_current_identity_preserves_charge_stamp_and_authenticates_sign(tmp_path):
    import json
    from gw.gw_config import LorraxConfig
    from gw.gw_init import _zeta_fit_provenance
    deck=tmp_path/'current.in'
    deck.write_text('[cohsex]\nnval=1\nncond=2\nnumber_bands=7\n'
                    'qp_solver=one_shot_dft\nlinalg=local\ncompute_mode=x_only\n'
                    'sys_dim=3\nbispinor=true\nbispinor_gw=bare_transverse\n')
    cfg=LorraxConfig.from_input_file(str(deck),resolve_hardware=False,print_fn=lambda *_:None)
    options=dict(wfn=SimpleNamespace(_filename='',ecutwfc=80.,ecutrho=320.),
        meta=SimpleNamespace(n_rmu=4,nspinor_wfnfile=2,fft_grid=(8,8,8),current_basis_rows=None),
        cfg=cfg,band_range_left=(0,4),band_range_right=(0,8),logical_band_stop=7,
        zeta_cutoff=80.,zeta_vcoul_cutoff=80.,write_ibz_only=True,band_norms=None)
    legacy=dict(n_rmu=4,centroids_md5='same-points',solver_kind='lu')
    current=dict(legacy,current_solver_family='signed_psd_rank_truncate')
    stamp=lambda channel,identity:_zeta_fit_provenance(**options,vertex_mu_L=channel,
                                                     transverse_identity=identity)
    assert stamp(0,legacy)==stamp(0,current)
    old=json.loads(stamp(2,legacy));new=json.loads(stamp(2,current))
    assert old['transverse_zeta_solve']=='ridge' and old['transverse_zeta_rcond'] is None
    assert new['transverse_zeta_solve']=='signed_psd_rank_truncate'
    assert new['transverse_zeta_rcond']==new['zeta_rcond']
    assert new['transverse_solver_kind']=='negative_rank_truncate'
    assert new!=old


def test_stream_uses_and_authenticates_actual_signed_PSD_factor():
    import runpy
    from pathlib import Path
    owners=runpy.run_path(str(Path(__file__).with_name('test_streamed_current_coefficients.py')))
    fixture,make_stream,fill=(owners[name] for name in ('fixture','make_stream','fill'))
    f=fixture()
    for c,z in enumerate(f.zetas):
        sign=-1 if c==1 else 1
        lam,V=np.linalg.eigh(sign*f.C[c])
        z.L_q=f.put(V/np.sqrt(lam)[:,None,:])
        z.lu_piv=None
        z.solver_kind='negative_rank_truncate' if sign==-1 else 'positive_rank_truncate'
    banks=fill(make_stream(f),f);actual=banks.authenticate(f.zetas)
    for c,pair in enumerate(actual):
        for k,value in enumerate(pair):
            np.testing.assert_allclose(np.asarray(value),f.expected[c][k].reshape(value.shape),
                                       rtol=3e-13,atol=3e-15)
    f.zetas[1].solver_kind='positive_rank_truncate'
    with pytest.raises(ValueError,match='owner changed'):
        banks.authenticate(f.zetas)
