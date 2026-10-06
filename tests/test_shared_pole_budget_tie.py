"""Independent exact Gram-eigenspace closure at the pole budget boundary."""
import jax.numpy as jnp
import numpy as np
import pytest
from gw.shared_pole_reduction import _within_budget


@pytest.mark.parametrize("values,budget",[
    ([1.,2.,3.,4.,5.],2),([1.,2.,3.,3.,4.],2),
    ([1.,3.,3.,3.],2),([3.,3.,3.,3.],2),
    ([1.,1.,2.,2.,3.,3.],3),([1.,1.,1.,3.,3.,3.],3),
])
def test_exact_blocks_are_kept_all_or_none_under_cap(values,budget):
    g=np.asarray(values);mask=np.asarray(_within_budget(jnp.asarray(g[None]),budget))[0]
    assert mask.sum()<=budget
    for value in np.unique(g):
        block=mask[g==value]
        assert np.any(block)==np.all(block)


def test_unfittable_top_multiplet_drops_whole_block():
    for g in ([1.,3.,3.,3.],[3.,3.,3.,3.]):
        assert not np.any(np.asarray(_within_budget(jnp.asarray([g]),2)))


def test_partial_edge_closure_keeps_unrelated_top_state():
    mask=np.asarray(_within_budget(jnp.asarray([[1.,2.,3.,3.,4.]]),2))[0]
    np.testing.assert_array_equal(mask,[False,False,False,False,True])


def test_closed_projector_is_invariant_under_tied_eigenbasis_rotation():
    rng=np.random.default_rng(4211)
    rotation=np.eye(4);rotation[1:,1:]=np.linalg.qr(rng.normal(size=(3,3)))[0]
    mask=np.asarray(_within_budget(jnp.asarray([[1.,3.,3.,3.]]),2))[0]
    p=np.diag(mask.astype(float))
    np.testing.assert_allclose(p,rotation@p@rotation.T,atol=2e-15,rtol=0)


def test_none_or_sufficient_cap_preserves_legacy_all_entries():
    gamma=jnp.asarray([[1.,3.,3.,3.]])
    assert _within_budget(gamma,None) is True
    assert _within_budget(gamma,4) is True
