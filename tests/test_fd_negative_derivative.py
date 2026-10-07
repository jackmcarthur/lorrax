"""Independent FD surface-weight and native carrier controls."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from gw.efermi import fd_occupations, fd_negative_derivative, OccupationState


def decimal_surface(x, width):
    with localcontext() as c:
        c.prec=80
        t=Decimal(str(-abs(x))).exp()
        return float(t/(1+t)**2/Decimal(str(width)))


def test_peak_units_sign_evenness_and_analytic_tail_without_occupation_cut():
    width=.02;mu=0.;x=np.asarray([-100.,-40.,-3.,0.,3.,40.,100.])
    energy=mu+width*x
    actual=np.asarray(fd_negative_derivative(energy[None],mu,width))[0]
    expected=np.asarray([decimal_surface(a,width) for a in x])
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=0.)
    assert actual[3]==1/(4*width) and np.all(actual>0)
    assert np.array_equal(actual,actual[::-1])
    stored=np.asarray(fd_occupations(energy[None],mu,width))
    assert stored[0,0]==1. and stored[0,1]==1.
    assert np.all(actual[:2]>0.) # subtraction of rounded 1-f would erase these.


@pytest.mark.parametrize('x',[-40.,-3.,0.,3.,40.])
def test_centered_continuous_law_derivative_has_quadratic_refinement(x):
    width=.02;mu=.125;energy=mu+width*x
    analytic=float(fd_negative_derivative([[energy]],mu,width)[0,0]);errors=[]
    for h in width*np.asarray([.05,.025,.0125]):
        if x<0:
            # Differentiate the independently evaluated complement in the
            # occupied tail, where the stored f table rounds to one.
            plus=float(fd_occupations([[-energy-h]],-mu,width)[0,0])
            minus=float(fd_occupations([[-energy+h]],-mu,width)[0,0])
            finite=(plus-minus)/(2*h)
        else:
            plus=float(fd_occupations([[energy+h]],mu,width)[0,0])
            minus=float(fd_occupations([[energy-h]],mu,width)[0,0])
            finite=-(plus-minus)/(2*h)
        errors.append(abs(finite-analytic)/analytic)
    assert 3.8<errors[0]/errors[1]<4.2
    assert 3.8<errors[1]/errors[2]<4.2
    assert errors[-1]<3e-5


def test_energy_unit_rescaling_changes_derivative_inversely():
    e=np.asarray([[-.5,0.,.3]]);a=np.asarray(fd_negative_derivative(e,.125,.02))
    np.testing.assert_allclose(np.asarray(fd_negative_derivative(2*e,.25,.04)),a/2,rtol=0,atol=0)


def test_fixed_n_table_is_unchanged_and_uses_its_exact_mu_and_width():
    e=np.asarray([[-.4,-.03,.07,1e300],[.02,.06,.2,-1e300]])
    valid=np.asarray([[1,1,1,0],[1,1,1,0]],bool)
    state=OccupationState.solve_smearing(e,[.5,.5],3.25,.02,state_capacity=2.,family='fd',valid_kn=valid)
    before=np.asarray(state.f_kn).copy();digest=state.occ_hash
    derivative=np.asarray(fd_negative_derivative(e,state.mu_ry,state.smearing_width_ry,valid_kn=valid))
    assert np.array_equal(np.asarray(state.f_kn),before) and state.occ_hash==digest
    assert np.array_equal(derivative[~valid],np.zeros((~valid).sum()))
    np.testing.assert_allclose(derivative[valid],np.asarray([decimal_surface((v-state.mu_ry)/state.smearing_width_ry,state.smearing_width_ry) for v in e[valid]]),rtol=2e-14,atol=0)


def test_mask_suppresses_nonfinite_ghosts_before_evaluation_and_is_dynamic():
    e=np.asarray([[np.nan,.125,np.inf,-np.inf]])
    valid=np.asarray([[0,1,0,0]],bool)
    a=np.asarray(fd_negative_derivative(e,.125,.02,valid_kn=valid))
    assert np.array_equal(a,np.asarray([[0.,12.5,0.,0.]]))
    b=np.asarray(fd_negative_derivative([[.125,.125,.125,.125]],.125,.02,valid_kn=~valid))
    assert np.array_equal(b,np.asarray([[12.5,0.,12.5,12.5]]))


@pytest.mark.parametrize('width',[0.,-1.,np.nan,np.inf])
def test_invalid_width_refuses(width):
    with pytest.raises(ValueError,match='kBT'):fd_negative_derivative([[0.]],0.,width)


@pytest.mark.parametrize('mu',[np.nan,np.inf,-np.inf])
def test_invalid_mu_refuses(mu):
    with pytest.raises(ValueError,match='chemical potential'):fd_negative_derivative([[0.]],mu,.02)


@pytest.mark.parametrize('energy',[[],[1.],np.zeros((1,1,1)),np.zeros((0,2))])
def test_invalid_energy_shape_refuses(energy):
    with pytest.raises(ValueError,match='E_kn'):fd_negative_derivative(energy,0.,.02)


@pytest.mark.parametrize('valid',[np.asarray([[1]],np.int32),np.asarray([True]),np.asarray([[True,False]])])
def test_invalid_native_mask_refuses(valid):
    with pytest.raises(ValueError,match='physical_band_validity'):fd_negative_derivative([[0.]],0.,.02,valid_kn=valid)
