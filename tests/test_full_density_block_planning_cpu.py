"""Native coherent block scalar planning, no spatial reduction."""
import numpy as np
import pytest


def test_public_remote_laplace_pair_rows_have_both_retarded_signs():
    from gw.response_bank import full_density_sample_rules
    sites = np.array([.2+.4j,.7+.5j])
    gamma = .03
    rules = full_density_sample_rules(np.array([[-4.,0.,1.]]),sites,gamma_ry=gamma,
        interval_ry=(2.,5.),paired=True,held={})
    assert all(r['construction']=='continuum_noncrossing_laplace' for r in rules)
    d = np.linspace(2.,5.,541)
    actual = np.zeros((2,len(sites),len(d)),complex)
    for r in rules:
        for side in (0,1):
            t=r['times'].conj() if side else r['times']
            actual[side] += r['weights'][side]@np.exp(-(d[:,None]-r['reference_ry'])*t).T
    z=sites+1j*gamma
    np.testing.assert_allclose(actual[0],1/(z[:,None]-d),rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(actual[1],-1/(z[:,None]+d),rtol=1e-8,atol=1e-10)


def test_nonstationary_support_blocks_keep_measured_crossing_intervals():
    from gw.response_bank import response_blocks
    energy = np.array([[-2.,-.1,.4,.8],[-1.9,-.3,.3,.5]])
    f=np.array([[1.,1.,1.,0.],[1.,1.,1.,0.]])
    u=np.array([[0.,1.,1.,1.],[0.,1.,1.,1.]])
    blocks=response_blocks(energy,f,u,(1,3),allow_crossing=True)
    assert {b['name'] for b in blocks}=={'FF','FA','AF','AA'}
    assert next(b for b in blocks if b['name']=='AF')['lo']==pytest.approx(.1)
    # Current global dispersion can overlap even when every row is ordered.
    energy[1,3]=.35
    with pytest.raises(ValueError,match='response_blocks_interval'):
        response_blocks(energy,f,u,(1,3))
    blocks=response_blocks(energy,f,u,(1,3),allow_crossing=True)
    assert next(b for b in blocks if b['name']=='AF')['lo']<0
