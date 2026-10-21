"""Tiny literal passive branch controls, not a replacement collision model."""
import numpy as np
import pytest
import jax

from tdagw.collision import branch_operators


def _dag(a):
    return a.conj().swapaxes(-1, -2)


def _width(a):
    return 1j*(a-_dag(a))


def _reference(eta=.2, degenerate=False):
    """Closed reciprocal three-level model, arbitrary common energy units."""
    e=np.array([-1., -1. if degenerate else -.5, 1.])
    rho=np.diag([1., 1., 0.])
    vertex=np.array([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]])
    grid={space:np.zeros((3, 3, 3), complex) for space in ('cond', 'val')}
    for f, omega in enumerate(e):
        for n in range(3):
            outer=np.outer(vertex[:, n], vertex[:, n])
            grid['cond'][f]+=outer*(1-rho[n,n])/(omega-e[n]-1+1j*eta)
            grid['val'][f]+=outer*rho[n,n]/(omega-e[n]+1+1j*eta)
    row={b:np.einsum('iij->ij',v) for b,v in grid.items()}
    col={b:np.einsum('jij->ij',v) for b,v in grid.items()}
    gamma=_width(col['cond']+col['val']); injection=_width(col['val'])
    collision=injection-(gamma@rho+rho@gamma)/2
    return e,rho,grid,row,col,gamma,injection,collision


def test_reciprocal_passive_trace_zero_model_has_actual_outward_projector_direction():
    e,p,grid,row,col,g,d,c=_reference()
    assert np.trace(c).real==0.
    np.testing.assert_allclose(p@p,p,atol=0.)
    np.testing.assert_allclose(np.diag(e)@p-p@np.diag(e),0.,atol=0.)
    for values in grid.values():
        assert np.linalg.eigvalsh(_width(values)).min()>-1.e-15
    eig=np.linalg.eigvalsh((g-d)[:2,:2])
    np.testing.assert_allclose(eig,[-.031543335232515776,.1393841276126106],atol=1.e-15)
    assert np.linalg.eigvalsh(c[:2,:2]).max()>0.
    assert d[2,2].real>0.
    assert np.linalg.det((g-d)[:2,:2]).real<0.


def test_actual_advanced_hole_compatibility_recovers_retarded_column():
    _,_,_,row,col,g,d,_=_reference()
    cond=np.stack([row['cond'],col['cond']])
    val=np.stack([_dag(col['val']),_dag(row['val'])])
    with jax.enable_x64(True):
        actual_g,actual_d,_=map(np.asarray,branch_operators(cond,val))
    np.testing.assert_allclose(actual_g,g,atol=2.e-16)
    np.testing.assert_allclose(actual_d,d,atol=2.e-16)


def test_hole_conversion_and_outer_density_order_cannot_change_filled_boundary():
    _,p,_,_,col,g,d,c=_reference()
    rng=np.random.default_rng(711)
    changed_hole=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3))
    changed_g=_width(col['cond']+changed_hole); changed_d=_width(changed_hole)
    np.testing.assert_allclose((changed_g-changed_d)[:2,:2],(g-d)[:2,:2],atol=1.e-15)
    q=np.eye(3)-p
    a=1j*(col['val']@q-col['cond']@p); literal=a+_dag(a)
    a=1j*(q@col['val']-p@col['cond']); alternate=a+_dag(a)
    np.testing.assert_allclose(literal[:2,:2],alternate[:2,:2],atol=0.)
    np.testing.assert_allclose(literal[2:,2:],alternate[2:,2:],atol=0.)
    np.testing.assert_allclose(literal[:2,:2],c[:2,:2],atol=1.e-16)


def test_row_swap_retains_negative_boundary_while_pv_free_diagnostic_is_not_a_cure_theorem():
    _,_,_,row,col,_,_,_=_reference()
    wc,wr=_width(col['cond']),_width(row['cond'])
    assert np.linalg.eigvalsh(wc[:2,:2]).min()<0.
    assert np.linalg.eigvalsh(wr[:2,:2]).min()<0.
    # A rank-one channel's unequal diagonal absorption also makes its
    # arithmetic spectral shell matrix indefinite, even with PV deleted.
    assert np.linalg.eigvalsh(((wc+wr)/2)[:2,:2]).min()<0.


def test_commutator_and_consistent_unitary_rotation_preserve_boundary_sign():
    _,p,_,_,_,g,d,c=_reference()
    rng=np.random.default_rng(844)
    a=rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)); h=(a+_dag(a))/2
    comm=-1j*(h@p-p@h)
    np.testing.assert_allclose(comm[:2,:2],0.,atol=0.)
    np.testing.assert_allclose(comm[2:,2:],0.,atol=0.)
    q,_=np.linalg.qr(a); basis=_dag(q)[:,:2]
    rotated=_dag(q)@(c+comm)@q
    np.testing.assert_allclose(np.linalg.eigvalsh(_dag(basis)@rotated@basis),
                               np.linalg.eigvalsh(c[:2,:2]),atol=1.e-15)


@pytest.mark.parametrize('eta',[.2,1.e-8])
def test_degenerate_energy_control_is_safe_but_small_eta_alone_is_not(eta):
    _,_,_,_,_,g,d,c=_reference(eta=eta)
    assert abs(np.trace(c))<1.e-16
    assert np.linalg.eigvalsh((g-d)[:2,:2]).min()<-.03
    _,_,_,_,_,g,d,c=_reference(eta=eta,degenerate=True)
    assert abs(np.trace(c))<1.e-16
    assert np.linalg.eigvalsh((g-d)[:2,:2]).min()>-1.e-15
