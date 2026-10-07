"""Independent null/isotropic/axial/radial contracts for the bulk sphere."""
import numpy as np
import pytest
import file_io  # noqa: F401 -- canonical service path bootstrap
from vcoul import Bulk3D, CoulombGeometry
from vcoul import bulk_3d as owner


@pytest.mark.parametrize('s', [0, -.04, -.04-.001j])
def test_solid_angle_isotropic(s):
    np.testing.assert_allclose(owner._interband_sphere_factor([s*np.eye(3)]),
        [1/(1-8*np.pi*s)], rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize('c', [5.0, 5.0+.3j])
def test_axial_dielectric_against_closed_form_and_rotation(c):
    a=2.0;eps=np.diag([a,a,c]).astype(complex)
    exact=np.arctan(np.sqrt((c-a)/a))/np.sqrt(a*(c-a))
    q,_=np.linalg.qr(np.array([[.4,.3,.7],[.8,-.2,.1],[.3,.5,-.2]]))
    cases=[eps,q@eps@q.T]
    np.testing.assert_allclose(owner._interband_sphere_factor(
        [(np.eye(3)-x)/(8*np.pi) for x in cases]), [exact,exact],
        rtol=2e-11,atol=2e-12)


def test_antisymmetric_tensor_does_not_enter_longitudinal_response():
    s=np.diag([-.02,-.04,-.06]).astype(complex)
    anti=np.array([[0,.05j,.03],[-.05j,0,-.02],[-.03,.02,0]])
    np.testing.assert_allclose(owner._interband_sphere_factor([s,s+anti]),
        np.repeat(owner._interband_sphere_factor([s]),2),rtol=2e-12,atol=2e-12)


def fixture_geometry_and_draw(monkeypatch):
    g=CoulombGeometry(2*np.pi*np.eye(3),1.)
    points=np.array([[.05,0,0],[.7,.7,0],[.7,.6,.2],[.3,.2,.1]])
    monkeypatch.setattr(owner,'_sample_q0_minibz_qpoints',lambda *a,**k:[points])
    return g,points


@pytest.mark.parametrize('s', [0,-.04,-.04-.001j])
def test_public_single_and_multi_row_use_same_corrected_average(monkeypatch,s):
    g,_=fixture_geometry_and_draw(monkeypatch);kernel=Bulk3D()
    v,w=kernel.q0_average(g,(4,4,4),S_cart=s*np.eye(3),analytic_sphere=True)
    v2,ws=kernel.q0_average_screened(g,(4,4,4),S_carts=[s*np.eye(3)],analytic_sphere=True)
    np.testing.assert_allclose([w,ws[0]], [v/(1-8*np.pi*s)]*2,rtol=2e-12,atol=2e-12)
    np.testing.assert_equal(v,v2)


@pytest.mark.parametrize('k2', [.3,300.,3e9,.3+.02j])
def test_thomas_fermi_sphere_against_independent_radial_integral(monkeypatch,k2):
    g,points=fixture_geometry_and_draw(monkeypatch);R=np.pi/4
    v,w=Bulk3D().q0_average(g,(4,4,4),static_kappa2=k2,analytic_sphere=True)
    u,weights=np.polynomial.legendre.leggauss(192)
    r=R*(u+1)/2
    # Direct integral of 8pi/(r²+k²), radial Jacobian and solid angle,
    # divided by the physical mini-BZ volume (2pi)^3/64.
    sphere=(8*np.pi*4*np.pi/(2*np.pi)**3*64)*np.sum(weights*R/2*r*r/(r*r+k2))
    q2=np.sum(points*points,axis=1)
    outside=np.sum(np.where(q2>R*R,8*np.pi/(q2+k2),0))/len(points)
    np.testing.assert_allclose(w,outside+sphere,rtol=2e-12,atol=2e-12)
    assert np.isfinite(v)


def test_extra_finite_q_response_refuses_unmatched_sphere(monkeypatch):
    g,_=fixture_geometry_and_draw(monkeypatch);k=Bulk3D()
    with pytest.raises(NotImplementedError,match='sphere_extra_chi_unavailable'):
        k.q0_average(g,(4,4,4),S_cart=np.zeros((3,3)),extra_chi=lambda q:0,analytic_sphere=True)
    with pytest.raises(NotImplementedError,match='sphere_extra_chi_unavailable'):
        k.q0_average_screened(g,(4,4,4),S_carts=[np.zeros((3,3))],extra_chi_rows=lambda q:0,analytic_sphere=True)


def test_bad_tensor_refuses_before_angular_quadrature():
    with pytest.raises(ValueError,match='sphere_tensor'):
        owner._interband_sphere_factor([np.full((3,3),np.nan)])
