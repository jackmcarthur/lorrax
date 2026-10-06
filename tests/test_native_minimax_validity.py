"""Ghost metadata cannot widen a physical static quadrature interval."""
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest
from gw.wavefunction_bundle import BandSlices,Wavefunctions
from gw import minimax_screening as owner


@pytest.mark.parametrize('ghost',[1e300,-1e300])
def test_native_static_interval_uses_only_physical_energies(monkeypatch,ghost):
    e=np.asarray([[-2.,-.8,.6,1.8,ghost],[-1.9,-.7,.5,-ghost,ghost]])
    valid=np.asarray([[True,True,True,True,False],[True,True,True,False,False]])
    f=np.asarray([[1.,1.,0.,0.,0.],[1.,1.,0.,0.,0.]])
    s=BandSlices.from_band_edges(0,0,2,3,5)
    w=Wavefunctions(enk=jnp.asarray(e),occ=jnp.asarray(f),slices=s,valid_kn=jnp.asarray(valid))
    seen=[]
    def solve(low,high,**kw):
        seen.append((low,high));return SimpleNamespace()
    monkeypatch.setattr(owner,'solve_laplace_minimax_interval',solve)
    _,reference=owner.build_static_quadrature(w,SimpleNamespace(energy_reference='midgap',target_error=1e-6,max_nodes=64))
    assert reference==pytest.approx(-.1,abs=1e-15)
    np.testing.assert_allclose(seen,[(1.2,3.8)],atol=1e-15)


def test_uniform_none_and_explicit_alltrue_interval_parity(monkeypatch):
    e=jnp.asarray([[-2.,-.8,.6,1.8],[-1.9,-.7,.5,1.7]])
    f=jnp.asarray([[1.,1.,0.,0.],[1.,1.,0.,0.]])
    s=BandSlices.from_band_edges(0,0,2,3,4)
    seen=[]
    def solve(low,high,**kw):seen.append((low,high));return SimpleNamespace()
    monkeypatch.setattr(owner,'solve_laplace_minimax_interval',solve)
    refs=[]
    for valid in (None,jnp.ones_like(e,dtype=bool)):
        _,ref=owner.build_static_quadrature(Wavefunctions(enk=e,occ=f,slices=s,valid_kn=valid),
            SimpleNamespace(energy_reference='midgap',target_error=1e-6,max_nodes=64))
        refs.append(ref)
    assert refs[0]==refs[1]
    assert seen[0]==seen[1]
