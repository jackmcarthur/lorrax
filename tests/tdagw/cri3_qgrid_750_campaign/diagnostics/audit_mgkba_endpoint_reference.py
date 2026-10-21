"""Small independent endpoint/stationarity diagnostic; no production imports.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
The 3-level electron/one passive boson model retains complete one-sided
complex column amplitudes and their full Hermitian self-energy correction.
This script diagnoses approximations; it implements no material closure.
The internal phase energies E remain fixed while the coherent Hamiltonian
contains diag(E)+MH[rho]. There is no instantaneous-H diagonalization or
self-consistent quasiparticle/GW ground-state solve in this model analogue.
"""
import json

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import root


E = np.array([0., .6, 1.3])
OMEGA = .4
M = np.zeros((3, 3), complex)
M[:2, 2] = [.2, .3j]
M[2, :2] = M[:2, 2].conj()
IDENTITY = np.eye(3)
PROJECTOR = np.diag([1., 1., 0.])


def adjoint(a):
    return a.conj().swapaxes(-1, -2)


def branch_columns(rho, eta, coupling=1.):
    vertex = coupling*M
    particle = np.einsum('am,mn,bn,bn->ab', vertex, IDENTITY-rho, vertex.conj(),
        1/(E[:, None]-E[None, :]-OMEGA+1j*eta))
    hole = np.einsum('am,mn,bn,bn->ab', vertex, rho, vertex.conj(),
        1/(E[:, None]-E[None, :]+OMEGA+1j*eta))
    return particle, hole


def raw_rhs(rho, eta, coupling=1.):
    particle, hole = branch_columns(rho, eta, coupling)
    total = particle+hole
    mh = .5*(total+adjoint(total))
    gamma = 1j*(total-adjoint(total))
    injection = 1j*(hole-adjoint(hole))
    h = np.diag(E)+mh
    return -1j*(h@rho-rho@h)+injection-.5*(gamma@rho+rho@gamma)


def coordinates():
    basis = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        x = np.zeros((3, 3), complex)
        x[a, b] = x[b, a] = 1/np.sqrt(2)
        basis.append(x)
        x = np.zeros((3, 3), complex)
        x[a, b], x[b, a] = 1j/np.sqrt(2), -1j/np.sqrt(2)
        basis.append(x)
    basis.extend((np.diag([1., -1., 0.])/np.sqrt(2),
                  np.diag([1., 1., -2.])/np.sqrt(6)))
    return np.asarray(basis)


BASIS = coordinates()


def pack(a):
    return np.einsum('jab,ba->j', BASIS, a).real


def unpack(x):
    # Exact N=2 parameterization, no density clipping or trace correction.
    return 2*IDENTITY/3+np.einsum('j,jab->ab', x, BASIS)


def stationary(eta, coupling=1.):
    fun = lambda x: pack(raw_rhs(unpack(x), eta, coupling))
    solution = root(fun, pack(PROJECTOR), tol=1e-11)
    rho = unpack(solution.x)
    step = 1e-6
    jacobian = np.column_stack([(fun(solution.x+step*np.eye(8)[j])
        -fun(solution.x-step*np.eye(8)[j]))/(2*step) for j in range(8)])
    return dict(eta=eta, coupling=coupling, root_success=bool(solution.success),
        full_rhs_residual_fro=float(np.linalg.norm(raw_rhs(rho, eta, coupling))),
        trace=float(np.trace(rho).real), eigenvalues=np.linalg.eigvalsh(rho).tolist(),
        diagonal=rho.diagonal().real.tolist(),
        offdiag_abs_max=float(np.max(abs(rho-np.diag(rho.diagonal())))),
        tangent_jacobian_real_range=[float(np.linalg.eigvals(jacobian).real.min()),
                                     float(np.linalg.eigvals(jacobian).real.max())])


def fermion_operator(a, b):
    """c_a^dagger c_b on the exact N=2 fermion sector."""
    states = [3, 5, 6]
    operator = np.zeros((3, 3), complex)
    for column, state in enumerate(states):
        if not state & (1 << b):
            continue
        sign = (-1)**((state & ((1 << b)-1)).bit_count())
        intermediate = state ^ (1 << b)
        if intermediate & (1 << a):
            continue
        sign *= (-1)**((intermediate & ((1 << a)-1)).bit_count())
        operator[states.index(intermediate | (1 << a)), column] = sign
    return operator


def exact_ground_state(coupling=1., cutoff=12):
    operators = np.asarray([[fermion_operator(a, b) for b in range(3)] for a in range(3)])
    electron_h = sum(E[a]*operators[a, a] for a in range(3))
    vertex = np.einsum('ab,abij->ij', coupling*M, operators)
    annihilation = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    h = (np.kron(electron_h, np.eye(cutoff))
         +np.kron(IDENTITY, OMEGA*np.diag(np.arange(cutoff)))
         +np.kron(vertex, annihilation+adjoint(annihilation)))
    energies, vectors = np.linalg.eigh(h)
    gs = vectors[:, 0]
    rho = np.asarray([[np.vdot(gs, np.kron(operators[b, a], np.eye(cutoff))@gs)
                       for b in range(3)] for a in range(3)])
    return dict(coupling=coupling, boson_cutoff=cutoff, energy=float(energies[0]),
        density_eigenvalues=np.linalg.eigvalsh(rho).tolist(),
        density_diagonal=rho.diagonal().real.tolist(),
        trace=float(np.trace(rho).real),
        raw_rhs_at_eta_005_fro=float(np.linalg.norm(raw_rhs(rho, .05, coupling))))


def endpoint_test(rho, eta=.05):
    """Direct KBE tau quadrature versus analytic column amplitudes."""
    def term(tau, wrong_external_order=False):
        phase = np.diag(np.exp(-1j*E*tau))
        lesser = 1j*rho@phase
        greater = -1j*(IDENTITY-rho)@phase
        sigma_less = M@lesser@adjoint(M)*np.exp(1j*OMEGA*tau)
        sigma_greater = M@greater@adjoint(M)*np.exp(-1j*OMEGA*tau)
        if wrong_external_order:
            reverse_less = 1j*rho@adjoint(phase)
            reverse_greater = -1j*(IDENTITY-rho)@adjoint(phase)
        else:
            reverse_less = 1j*adjoint(phase)@rho
            reverse_greater = -1j*adjoint(phase)@(IDENTITY-rho)
        value = sigma_less@reverse_greater-sigma_greater@reverse_less
        return np.exp(-eta*tau)*(value+adjoint(value))
    integral, error = quad_vec(term, 0., np.inf, epsabs=1e-11, epsrel=1e-11)
    wrong, _ = quad_vec(lambda tau: term(tau, True), 0., np.inf,
                        epsabs=1e-11, epsrel=1e-11)
    p, h = branch_columns(rho, eta)
    ai, ao = 1j*h, 1j*p
    raw = ai@(IDENTITY-rho)+(IDENTITY-rho)@adjoint(ai)-ao@rho-rho@adjoint(ao)
    return dict(column_integral_error_max=float(abs(integral-raw).max()),
        quadrature_error_estimate=float(error),
        wrong_negative_lag_order_error_max=float(abs(wrong-raw).max()),
        raw_trace=float(np.trace(raw).real),
        wrong_negative_lag_order_trace=float(np.trace(wrong).real))


def spectral_only_test(rho, eta=.05):
    # Complete matrices at every common external energy, prior to shell sampling.
    particle = np.asarray([M@(IDENTITY-rho)@np.diag(1/(omega-E-OMEGA+1j*eta))@adjoint(M)
                           for omega in E])
    hole = np.asarray([M@rho@np.diag(1/(omega-E+OMEGA+1j*eta))@adjoint(M) for omega in E])
    spectral = []
    for branch in (particle, hole):
        k = 1j*(branch-adjoint(branch))
        spectral.append(.5*(k[np.arange(3), np.arange(3), :]
                            +k[np.arange(3), :, np.arange(3)].T))
    kp, kh = spectral
    c = kh-.5*((kp+kh)@rho+rho@(kp+kh))
    return dict(raw_fresh_trace=float(np.trace(raw_rhs(rho, eta)).real),
        spectral_only_fresh_trace=float(np.trace(c).real))


def dressing_test(rho, eta=.05):
    """Exact matrix divided-difference identity, not a closure validation."""
    fs, pvs = [], []
    difference = E[None, :]-E[:, None]
    nondegenerate = ~np.eye(3, dtype=bool)
    for density, pole in ((IDENTITY-rho, OMEGA), (rho, -OMEGA)):
        rs, derivatives = [], []
        for omega in E:
            denominator = omega-E-pole+1j*eta
            value = M@density@np.diag(1/denominator)@adjoint(M)
            derivative = M@density@np.diag(-1/denominator**2)@adjoint(M)
            rs.append(.5*(value+adjoint(value)))
            derivatives.append(.5*(derivative+adjoint(derivative)))
        rs, derivatives = np.asarray(rs), np.asarray(derivatives)
        row, column = rs[np.arange(3), np.arange(3), :], rs[np.arange(3), :, np.arange(3)].T
        f = np.zeros((3, 3), complex)
        f[nondegenerate] = (column-row)[nondegenerate]/difference[nondegenerate]
        f[np.arange(3), np.arange(3)] = derivatives[np.arange(3), np.arange(3), np.arange(3)]
        fs.append(f)
        pvs.append(1j*(column-row))
    fp, fh = fs
    gp, dh = pvs[0]+pvs[1], pvs[1]
    c_pv = dh-.5*(gp@rho+rho@gp)
    ft = fp+fh
    z = fh-.5*(ft@rho+rho@ft)
    energy = np.diag(E)
    commutator = energy@rho-rho@energy
    free_dressing = -1j*(energy@z-z@energy)
    coherent_remainder = -.5j*(ft@commutator+commutator@ft)
    reconstructed = rho-z
    return dict(identity_error_max=float(abs(c_pv-free_dressing-coherent_remainder).max()),
        commutator_only_error_fro=float(np.linalg.norm(c_pv-free_dressing)),
        coherent_remainder_fro=float(np.linalg.norm(coherent_remainder)),
        dressing_fro=float(np.linalg.norm(z)), dressing_trace=float(np.trace(z).real),
        rho_minus_Z_eigenvalues=np.linalg.eigvalsh(reconstructed).tolist(),
        limitation='rho-Z is only a first-order candidate density reconstruction; no PSD, exact-GS, transport or driven-coherence equivalence is asserted.')


def main():
    rng = np.random.default_rng(91)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    coherent = u@np.diag([.95, .7, .35])@adjoint(u)
    p, h = branch_columns(PROJECTOR, .05)
    ai, ao = 1j*h, 1j*p
    return dict(model=dict(energies=E.tolist(), omega=OMEGA,
        vertex_02=[.2, 0.], vertex_12=[0., .3], particle_count=2),
        projector=dict(raw_trace=float(np.trace(raw_rhs(PROJECTOR, .05)).real),
            filled_blocking_eigenvalues=np.linalg.eigvalsh((ao+adjoint(ao))[:2, :2]).tolist(),
            empty_injection=float((ai+adjoint(ai))[2, 2].real)),
        coherent_endpoint=endpoint_test(coherent),
        spectral_only_number_test=spectral_only_test(coherent),
        projector_PV_dressing=dressing_test(PROJECTOR),
        coherent_PV_dressing=dressing_test(coherent),
        nonlinear_stationary=[stationary(eta) for eta in (.1, .05, .01, .001)],
        weak_coupling_stationary=[stationary(.01, coupling) for coupling in (.1, .3, 1.)],
        exact_number_conserving_fermion_boson_ground_states=[
            exact_ground_state(coupling, cutoff) for coupling, cutoff in ((1., 12), (1., 18), (.1, 12))],
        limitations='Toy diagnostic only, with FIXED internal phase energies E and coherent H=diag(E)+MH[rho]. No instantaneous-H diagonalization or self-consistent quasiparticle/GW solve. Nonlinear Markov stationary density is not assumed a ground state. Exact fermion-boson comparison is a model analogue and does not rule out a genuine self-consistent correlated-ground-state preparation. No spectral/density clipping, fit reduction, material calculation or production prescription change.')


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
