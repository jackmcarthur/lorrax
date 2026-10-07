"""CPU4 NEW-block admission guards, independent of response/contraction math."""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from gw import contour_reference as cd


def operands(mesh):
    face=NamedSharding(mesh,P(None,"x","y"))
    return (jax.device_put(np.eye(4,dtype=np.complex128)[None],face),
            jax.device_put(np.arange(32,dtype=np.complex128).reshape(1,8,4),face))


@pytest.mark.parametrize("slot",range(4))
def test_proper_subset_refuses_before_projection(monkeypatch,slot):
    assert len(jax.devices())==4
    mesh=Mesh(np.asarray([jax.devices()[slot]]).reshape(1,1),("x","y"))
    def forbidden(*args,**kw):raise AssertionError("proper subset reached numerical projection")
    monkeypatch.setattr(cd,"project_interaction_diagonal",forbidden)
    w,p=operands(mesh)
    with pytest.raises(ValueError,match="every global JAX device"):
        cd.project_interaction_block(w,p,mesh=mesh,n_targets=2,prefactor=.1,
            scalar_replication_bound_bytes=256)


@pytest.mark.parametrize("permutation",[(0,1,2,3),(3,2,1,0)])
def test_full_square_membership_reaches_unchanged_numerical_owner(monkeypatch,permutation):
    assert len(jax.devices())==4
    mesh=Mesh(np.asarray(jax.devices())[list(permutation)].reshape(2,2),("x","y"))
    reached=[]
    def diagonal(w,p,**kw):reached.append((w,p,kw));return p,object()
    sentinel=object()
    monkeypatch.setattr(cd,"project_interaction_diagonal",diagonal)
    monkeypatch.setattr(cd,"_target_block_program",lambda m,n:lambda p,r,f:sentinel)
    w,p=operands(mesh)
    product,block=cd.project_interaction_block(w,p,mesh=mesh,n_targets=2,prefactor=.1,
        scalar_replication_bound_bytes=256)
    assert product is p and block is sentinel and len(reached)==1
    assert reached[0][2]["mesh"] is mesh


def test_full_rectangular_grid_refuses_before_projection(monkeypatch):
    assert len(jax.devices())==4
    mesh=Mesh(np.asarray(jax.devices()).reshape(1,4),("x","y"))
    def forbidden(*args,**kw):raise AssertionError("rectangular mesh reached projection")
    monkeypatch.setattr(cd,"project_interaction_diagonal",forbidden)
    w,p=operands(mesh)
    with pytest.raises(ValueError,match="square X/Y"):
        cd.project_interaction_block(w,p,mesh=mesh,n_targets=2,prefactor=.1,
            scalar_replication_bound_bytes=256)
