"""Finite two-endpoint RPA with explicitly declared correlated boundary data.

These CPU references cover only pure isospectral projector rotations. The
positive Gaussian covariance is quasibosonic, not the exact finite-fermion
Pauli covariance at a pure Slater rho. It is not determined by a lesser Wick
bubble alone. No production runtime or general mixed-rho closure is tested.
Energies use hbar=1; arrays are tiny and unsharded.
"""

import numpy as np
import pytest
import scipy.integrate as sint
import scipy.linalg as sla


_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def _cross(r):
    """Return [r]_cross, with [r]_cross x=r cross x for real r[3]."""
    x, y, z = r
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _kato(reference, target):
    """Direct polar rotation of pure projectors; refuse orthogonal support."""
    for p in (reference, target):
        if (np.linalg.norm(p-p.conj().T) > 1e-12 or np.linalg.norm(p@p-p) > 1e-12
                or abs(np.trace(p)-1) > 1e-12):
            raise ValueError("pure rank-one projectors required")
    identity = np.eye(2)
    overlap = target @ reference + (identity-target) @ (identity-reference)
    values, vectors = sla.eigh(overlap.conj().T @ overlap)
    if values.min() <= 1e-10:
        raise ValueError("principal angle reaches 90 degrees")
    return overlap @ (vectors*(1/np.sqrt(values))) @ vectors.conj().T


def _rotation(unitary):
    """Adjoint real rotation O_ab=Tr(sigma_a U sigma_b U.H)/2."""
    rotated = np.einsum("ij,bjk,lk->bil", unitary, _PAULI, unitary.conj())
    return (0.5*np.einsum("aij,bji->ab", _PAULI, rotated)).real


def _prepare(theta):
    """Rotate reference RPA vacuum and mean by one declared direct rotation."""
    reference = np.diag([0.0, 1.0]).astype(complex)
    kick = sla.expm(-0.5j*theta*_PAULI[1])
    target = kick @ reference @ kick.conj().T
    rotation = _rotation(_kato(reference, target))
    omega = np.sqrt(2.0)  # Delta=1, g=1/2, interaction=1.
    covariance = np.array([[1/omega, -1j, 0], [1j, omega, 0], [0, 0, 0]])
    return rotation @ np.array([0.0, 0.0, -1.0]), rotation @ covariance @ rotation.T


def _evolve(rinitial, times):
    """Integrate dot F=D(t)F with free rho(t), direct-RPA feedback, and F(0)=I."""
    vertex = np.array([np.sqrt(0.5), 0.0, 0.0])
    free = _cross(np.array([0.0, 0.0, 1.0]))

    def background(t):
        x, y, z = rinitial
        return np.array([np.cos(t)*x-np.sin(t)*y, np.sin(t)*x+np.cos(t)*y, z])

    def rhs(t, flat):
        drift = free + 2*np.outer(np.cross(vertex, background(t)), vertex)
        return (drift @ flat.reshape(3, 3)).ravel()

    solution = sint.solve_ivp(rhs, (0.0, float(times[-1])), np.eye(3).ravel(),
                              method="DOP853", rtol=2e-11, atol=2e-13, dense_output=True)
    assert solution.success
    return solution.sol(times).T.reshape(-1, 3, 3), np.array([background(t) for t in times])


def test_correlated_reference_ground_kernel_matches_analytic_two_endpoint_rpa():
    """The correlated boundary removes the historical positive lesser branch."""
    times = np.linspace(0.0, 6.0, 61)
    rinitial, covariance = _prepare(0.0)
    fundamental, _ = _evolve(rinitial, times)
    output = np.sqrt(0.5)*fundamental[:, 0, :]
    greater_covariance = output @ covariance @ output.T
    omega, residue = np.sqrt(2.0), 0.5/np.sqrt(2.0)
    lag = times[:, None]-times[None, :]
    np.testing.assert_allclose(greater_covariance, residue*np.exp(-1j*omega*lag), atol=3e-11)
    # Physical endpoint ordering: Wgreater=-i<Q(t)Q(s)>, Wless=-i<Q(s)Q(t)>.
    wgreater, wless = -1j*greater_covariance, -1j*greater_covariance.T
    wretarded = np.tril(wgreater-wless)  # Bare delta(t-s) is excluded.
    np.testing.assert_allclose(wless, -1j*residue*np.exp(1j*omega*lag), atol=3e-11)
    np.testing.assert_allclose(wretarded, np.tril(-2*residue*np.sin(omega*lag)), atol=3e-11)
    np.testing.assert_allclose(wretarded-wretarded.conj().T, wgreater-wless, atol=2e-14)
    basis = np.column_stack((np.exp(-1j*omega*times), np.exp(1j*omega*times)))
    weights = sla.lstsq(basis, 1j*wless[:, 0])[0]
    assert abs(weights[0]) < 2e-11
    np.testing.assert_allclose(weights[1], residue, atol=2e-11)
    # Bare Slater Wick covariance has unit transverse variances. This RPA
    # squeezed state is additional quasiboson boundary information.
    reference = np.diag([0.0, 1.0])
    wick = np.array([[np.trace(reference @ a @ (np.eye(2)-reference) @ b)
                      for b in _PAULI] for a in _PAULI])
    assert np.linalg.norm(covariance-wick) > 0.5
    assert covariance[1, 1].real > 1  # Not an exact finite-fermion Pauli variance.


def test_kato_preparation_covariance_guard_and_physical_stabilizer_ambiguity():
    """The direct rotation is covariant, but rho does not identify a real kick."""
    reference = np.diag([0.0, 1.0]).astype(complex)
    kick = sla.expm(-0.35j*_PAULI[1])
    target = kick @ reference @ kick.conj().T
    direct = _kato(reference, target)
    np.testing.assert_allclose(direct @ reference @ direct.conj().T, target, atol=2e-14)
    np.testing.assert_allclose(_kato(reference, reference), np.eye(2), atol=2e-14)
    gauge = sla.expm(-0.31j*(_PAULI[0]+0.4*_PAULI[2]))
    transformed = _kato(gauge @ reference @ gauge.conj().T, gauge @ target @ gauge.conj().T)
    np.testing.assert_allclose(transformed, gauge @ direct @ gauge.conj().T, atol=2e-14)
    with pytest.raises(ValueError, match="90 degrees"):
        _kato(reference, np.eye(2)-reference)
    with pytest.raises(ValueError, match="pure"):
        _kato(reference, np.eye(2)/2)
    # A reference-projector stabilizer changes squeezed fluctuation orientation
    # while leaving the final fermionic density exactly the same.
    alternate = direct @ sla.expm(-0.3j*_PAULI[2])
    np.testing.assert_allclose(alternate @ reference @ alternate.conj().T, target, atol=2e-14)
    _, boundary = _prepare(0.0)
    correlated_direct = _rotation(direct) @ boundary @ _rotation(direct).T
    correlated_alternate = _rotation(alternate) @ boundary @ _rotation(alternate).T
    assert np.linalg.norm(correlated_direct-correlated_alternate) > 0.5


@pytest.mark.parametrize("direction", [-1, 1])
def test_coherent_endpoint_kernel_preserves_commutator_psd_and_nonstationarity(direction):
    """Transport both endpoints instead of imposing a stationary lag extension."""
    times = direction*np.linspace(0.0, 6.0, 61)
    rinitial, boundary = _prepare(0.7)
    fundamental, background = _evolve(rinitial, times)
    vertex = np.array([np.sqrt(0.5), 0.0, 0.0])
    free = _cross(np.array([0.0, 0.0, 1.0]))
    jinitial = -2j*_cross(rinitial)
    np.testing.assert_allclose(boundary-boundary.T, jinitial, atol=2e-14)
    for f, r in zip(fundamental, background, strict=True):
        j = -2j*_cross(r)
        drift = free + 2*np.outer(np.cross(vertex, r), vertex)
        np.testing.assert_allclose(drift @ j+j @ drift.T, -2j*_cross(free @ r), atol=2e-14)
        np.testing.assert_allclose(f @ jinitial @ f.T, j, atol=4e-10)
        np.testing.assert_allclose(r @ f, rinitial, atol=4e-10)
    # Every block uses the same positive correlated boundary covariance.
    stacked = fundamental[::5].reshape(-1, 3)
    endpoint_kernel = stacked @ boundary @ stacked.T
    assert sla.eigvalsh(endpoint_kernel).min() > -2e-11
    output = np.sqrt(0.5)*fundamental[:, 0, :]
    charge_kernel = output @ boundary @ output.T
    assert sla.eigvalsh(charge_kernel).min() > -2e-12
    assert np.linalg.norm(charge_kernel[1:, 1:]-charge_kernel[:-1, :-1])/np.linalg.norm(charge_kernel) > 0.01


def test_retarded_volterra_converges_to_independent_fundamental_transport():
    """Direct free-spin bubble plus causal Dyson converges at second order."""
    rinitial, boundary = _prepare(0.7)
    errors = []
    for count in (81, 161, 321):
        times = np.linspace(0.0, 6.0, count)
        dt = times[1]-times[0]
        fundamental, _ = _evolve(rinitial, times)
        output = np.sqrt(0.5)*fundamental[:, 0, :]
        covariance = output @ boundary @ output.T
        exact = np.tril(-1j*(covariance-covariance.T))
        # Free bubble -i v.T R_free(t-s) J(s) v. Here 2g=Delta=1,
        # Q=sqrt(g)sigma_x, so only conserved background r_z enters.
        bare = np.tril(rinitial[2]*np.sin(times[:, None]-times[None, :]))
        # Kernels vanish on the diagonal; trapezoid endpoint terms vanish.
        discrete = sla.solve_triangular(np.eye(count)-dt*bare, bare, lower=True)
        errors.append(float(np.max(abs(discrete-exact))))
    assert errors[-1] < 2e-4
    assert errors[0]/errors[1] > 3.8
    assert errors[1]/errors[2] > 3.8
