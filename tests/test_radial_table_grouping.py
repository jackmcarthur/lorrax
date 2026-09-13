"""Numerical contracts for grouped projector Hankel tables."""
import numpy as np
import pytest
from psp.species import SpeciesData
from psp.radial_tables import build_all_tables
from psp.radial import radial_jax

@pytest.mark.parametrize('projectors,second,third', [(False,False,False),(True,False,False),(True,True,False),(True,True,True)])
def test_grouped_tables_match_independent_rows(monkeypatch, projectors, second, third):
    r=np.linspace(.01,12,121)
    ls=np.array([2,0,1,2,0])  # deliberately interleaved channels
    beta=np.stack([(i+1)*r**l*np.exp(-r) for i,l in enumerate(ls)])
    sp=SpeciesData('Si',4.,14,r,np.full_like(r,r[1]-r[0]),-8/r,
                   np.exp(-r*r),True,len(ls),beta,ls,np.eye(len(ls)),81)
    flags=dict(projectors=projectors,second_derivatives=second,third_derivatives=third)
    got=build_all_tables([sp],4.,65,**flags)
    kernel=radial_jax.spherical_hankel_table_batch_jax
    def separate(l,r,rows,q,w):
        return np.concatenate([np.asarray(kernel(l,r,row[None,:],q,w)) for row in rows])
    monkeypatch.setattr(radial_jax,'spherical_hankel_table_batch_jax',separate)
    expected=build_all_tables([sp],4.,65,**flags)
    assert got.keys()==expected.keys()
    for key,value in got.items():
        if value is None:
            assert expected[key] is None
        else:
            np.testing.assert_allclose(value,expected[key],rtol=2e-12,atol=2e-11,err_msg=key)
    if not projectors:
        assert got['proj_tables'] is None and got['deriv_tables'] is None

def test_third_derivative_requires_second():
    with pytest.raises(ValueError,match='requires second'):
        build_all_tables([],4.,65,third_derivatives=True)
