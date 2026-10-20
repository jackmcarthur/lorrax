"""Independent dielectric/inverse witnesses; no fitted-pole admission or GPU."""
from types import SimpleNamespace

import numpy as np
import pytest


def _native(epsilon, root_scale=1.):
    import jax
    from jax.sharding import Mesh
    from gw.response_bank import response_algebra
    from gw.shared_pole_recipe import CapacityLedger
    jax.config.update('jax_enable_x64',True)
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    n=len(epsilon);meta=SimpleNamespace(nk_tot=36,nspin=1,nspinor_wfnfile=2,nspinor=2,n_rmu=n)
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=2**30)
    meta.shared_pole_capacity.live_stages=()
    config=SimpleNamespace(backend=SimpleNamespace(linalg='local'))
    root=root_scale*np.diag(np.linspace(.8,1.2,n)).astype(complex)
    pref=1/6
    raw=np.linalg.solve(root,(np.eye(n)-epsilon)/pref)@np.linalg.inv(root)
    dyson,_,_,receipt=response_algebra(meta,config,mesh_xy=mesh,n=n,ordered=True)
    matrix_only=dyson.dielectric_matrix(root[None],raw[None])
    eps,inverse=dyson.dielectric(root[None],raw[None])
    np.testing.assert_array_equal(matrix_only,eps)
    np.testing.assert_allclose(eps[0],epsilon,rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(inverse[0],np.linalg.inv(epsilon),rtol=3e-13,atol=3e-13)
    assert receipt['prefactor']==pref
    return dyson,np.asarray(eps),np.asarray(inverse),root,raw


def _metrics(epsilon,inverse,mask):
    import jax
    from gw.response_bank import dielectric_physical_metrics,dielectric_inverse_bounds
    jax.config.update('jax_enable_x64',True)
    metrics={k:float(v) for k,v in dielectric_physical_metrics(epsilon,mask,
        matmul=lambda a,b:a@b,inverse=inverse).items()}
    bound=dielectric_inverse_bounds(metrics['epsilon_fro'],metrics['inverse_fro'],
        metrics['inverse_residual_fro'],int(np.count_nonzero(mask)),
        padding_defect=max(metrics['padding_defect'],metrics['inverse_padding_defect']))
    return metrics,bound


def test_native_dielectric_uses_current_full_complex_response_without_hermitization():
    eps=np.array([[.7+.2j,.4-.3j,0.],[.03j,1.2-.1j,.2],[.08,0.,.9+.1j]])
    dyson,actual,inverse,root,raw=_native(eps)
    assert np.linalg.norm(actual[0]-actual[0].conj().T)>.3
    expected=root@(np.linalg.inv(eps)-np.eye(3))@root
    np.testing.assert_allclose(dyson.value(root[None],raw[None])[0],expected,rtol=3e-13,atol=3e-13)
    metrics,bounds=_metrics(actual,inverse,np.ones(3,bool))
    smallest=np.linalg.svd(eps,compute_uv=False)[-1]
    assert bounds['resolved'] and bounds['sigma_min_lower']<=smallest<=bounds['sigma_min_upper']


def test_nonnormal_inverse_bounds_use_logical_mask_not_identity_padding_or_coulomb_rank():
    # Non-contiguous centroid padding and one perfectly physical identity
    # direction. The latter must NOT be discarded as a Coulomb-null mode.
    mask=np.array([True,False,True,False,True])
    physical=np.array([[1.,90.+30j,0.],[0.,1.,.2j],[0.,0.,1.]])
    eps=np.eye(5,dtype=complex);eps[np.ix_(mask,mask)]=physical
    _,actual,inverse,_,_=_native(eps)
    metrics,bounds=_metrics(actual,inverse,mask)
    assert bounds['dimension']==3 and bounds['resolved']
    assert metrics['padding_defect']<1e-14 and metrics['inverse_padding_defect']<1e-14
    assert metrics['epsilon_fro']==pytest.approx(np.linalg.norm(physical),rel=2e-14)
    assert metrics['inverse_fro']==pytest.approx(np.linalg.norm(np.linalg.inv(physical)),rel=2e-14)
    smallest=np.linalg.svd(physical,compute_uv=False)[-1]
    assert bounds['sigma_min_lower']<=smallest<=bounds['sigma_min_upper']
    # A visibly coupled/nonidentity pad is not allowed to produce a logical
    # certificate even if an inverse exists for the entire padded matrix.
    bad=eps.copy();bad[1,0]=.03j
    metrics,bounds=_metrics(bad[None],np.linalg.inv(bad)[None],mask)
    assert not bounds['resolved'] and bounds['reason']=='padding is not decoupled identity'


def test_near_true_dielectric_null_and_unresolved_inverse_refuse_false_precision():
    from gw.response_bank import dielectric_inverse_bounds
    for gap in (1e-5,1e-10):
        eps=np.diag([gap,1.,1.]).astype(complex)
        metrics,bounds=_metrics(eps[None],np.linalg.inv(eps)[None],np.ones(3,bool))
        assert bounds['resolved'] and bounds['sigma_min_lower']<=gap<=bounds['sigma_min_upper']
    eps=np.diag([1e-16,1.,1.]).astype(complex)
    _,bounds=_metrics(eps[None],np.linalg.inv(eps)[None],np.ones(3,bool))
    assert not bounds['resolved'] # inverse-product roundoff allowance reaches one
    assert not dielectric_inverse_bounds(1.,1.,1.1,3)['resolved']
    assert not dielectric_inverse_bounds(1.,np.inf,0.,3)['resolved']


def test_huge_screened_background_does_not_certify_a_dielectric_pole():
    eps=np.diag([.7,1.2,1.4]).astype(complex)
    dyson,actual,inverse,root,raw=_native(eps,root_scale=1e5)
    background=np.asarray(dyson.value(root[None],raw[None]))[0]
    assert np.linalg.norm(background)>1e9
    _,bounds=_metrics(actual,inverse,np.ones(3,bool))
    assert bounds['resolved'] and bounds['sigma_min_lower']>.3 # nowhere near singular
    # A fake weak fitted pole may differ by <1% on this huge background.
    # Such total-W agreement cannot overturn the direct dielectric witness.
    fake_term=np.zeros((3,3),complex);fake_term[0,0]=.004*np.linalg.norm(background)
    assert np.linalg.norm((background+fake_term)-background)/np.linalg.norm(background)==pytest.approx(.004)


def test_upper_endpoint_crossing_is_exact_without_rho_energy_commutation_or_central_order():
    from tests.test_full_density_blocks_oracle_cpu import case
    from tests._full_microscopic_oracle import ordered_pr
    psi,energy,rho,partners=case()
    comm=rho*energy[:,None,:]-energy[:,:,None]*rho
    assert np.linalg.norm(comm)>.01
    sites=np.array([-.3+.23j,.19+.44j]);gamma=.043;nk=len(partners)
    for q,partner in enumerate(partners):
        raw=ordered_pr(psi,energy,rho,partner,sites,gamma)
        mirror=ordered_pr(psi,energy,rho,partners[(-q)%nk],-sites.conjugate(),gamma).conjugate()
        np.testing.assert_allclose(raw,mirror,rtol=3e-13,atol=3e-15)
