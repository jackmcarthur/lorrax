"""Positive endpoint weights retain coverage and one normal-equation metric."""
import numpy as np
import pytest

from gw.isdf_fitting import fitting_band_weights, fitting_weight_options


def test_unit_weight_preserves_origin_and_transport_masks():
    expected = (np.array([1,1,1,1,0,0,0,0],float),
                np.array([0,0,1,1,1,1,0,0],float))
    legacy = fitting_band_weights(8,(3,7),(5,9))
    explicit = fitting_band_weights(8,(3,7),(5,9),occupied_stop=6,occupied_weight=1.)
    for old,new,wanted in zip(legacy,explicit,expected):
        np.testing.assert_array_equal(old,wanted)
        np.testing.assert_array_equal(new,old)
    assert fitting_weight_options(None) == {}


@pytest.mark.parametrize('weight',[np.nan,np.inf,.99,-4,True,[4.],1+1j])
def test_invalid_weight_refuses(weight):
    with pytest.raises(ValueError,match='occupied endpoint weight'):
        fitting_band_weights(8,(0,4),(0,6),occupied_stop=2,occupied_weight=weight)


@pytest.mark.parametrize('stop',[True,2.,-1,None,5])
def test_nonunit_weight_requires_integer_occupied_coverage(stop):
    with pytest.raises(ValueError,match='occupied'):
        fitting_band_weights(8,(0,4),(0,6),occupied_stop=stop,occupied_weight=4.)


def test_explicit_policy_rejects_partial_or_extra_fields():
    for policy in ({'occupied_weight':4.},{'occupied_stop':2},
                   {'occupied_stop':2,'occupied_weight':4.,'empty_weight':1.},[2,4]):
        with pytest.raises(ValueError,match='exactly'):
            fitting_weight_options(policy)


def test_all_spin_terms_and_lr_rl_match_literal_positive_pair_loss():
    rng = np.random.default_rng(463)
    nk,nb,ns,nmu,nr = 3,5,4,7,6
    def fields(shape):
        return (rng.normal(size=shape)+1j*rng.normal(size=shape))/4
    mu = fields((nk,nb,ns,nmu))
    ae,ps = fields((nk,nb,ns,nr)),fields((nk,nb,ns,nr))
    wl,wr = map(np.asarray,fitting_band_weights(nb,(0,3),(0,5),
        occupied_stop=2,occupied_weight=4.))
    assert wl[2]*wr[4] == 1. and wl[0]*wr[0] == 16.
    assert np.all(wr > 0), 'every original right endpoint remains covered'

    def literal(q):
        C,Z = np.zeros((nmu,nmu),complex),np.zeros((nmu,nr),complex)
        for k in range(nk):
            kp = (k+q)%nk
            for m in range(nb):
                for n in range(nb):
                    pair = np.sum(mu[k,m].conj()*mu[kp,n],axis=0)
                    delta = np.sum(ae[k,m].conj()*ae[kp,n]-ps[k,m].conj()*ps[kp,n],axis=0)
                    C += wl[m]*wr[n]*np.outer(pair.conj(),pair)
                    Z += wl[m]*wr[n]*np.outer(pair.conj(),delta)
        return C,Z

    def projectors(q):
        C,Z = np.zeros((nmu,nmu),complex),np.zeros((nmu,nr),complex)
        for k in range(nk):
            kp = (k+q)%nk
            for i in range(ns):
                for j in range(ns):
                    lm = np.einsum('m,mx,my->xy',wl,mu[k,:,i],mu[k,:,j].conj())
                    rn = np.einsum('n,nx,ny->xy',wr,mu[kp,:,i].conj(),mu[kp,:,j])
                    C += lm*rn
                    for sign,right in ((1,ae),(-1,ps)):
                        l = np.einsum('m,mx,mr->xr',wl,mu[k,:,i],right[k,:,j].conj())
                        r = np.einsum('n,nx,nr->xr',wr,mu[kp,:,i].conj(),right[kp,:,j])
                        Z += sign*l*r
        return C,Z

    for q in range(nk):
        literal_rows,other = literal(q),literal((-q)%nk)
        projected,projected_other = projectors(q),projectors((-q)%nk)
        for expected,partner,got,got_partner in zip(literal_rows,other,projected,projected_other):
            expected,got = expected+partner.conj(),got+got_partner.conj()
            assert np.linalg.norm(got-expected)/np.linalg.norm(expected) < 8e-15
        assert np.linalg.eigvalsh(literal_rows[0])[0] > 0
