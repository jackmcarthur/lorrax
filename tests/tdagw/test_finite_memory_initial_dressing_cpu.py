"""Tiny exact number-conserving model: finite memory versus bare Markov start.

This is a perturbative explanation test, not a production closure or a proof
that an arbitrary stationary Markov density is an interacting ground state.
All energies/times use hbar=1. The oscillator is undamped; eta is an Abel
regulator taken to zero, not a finite physical linewidth in the exact model.
"""
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh


ENERGIES = np.array([-1., -.5, 1.])
OMEGA = 1.
PROJECTOR = np.diag([1., 1., 0.])


def _model(g, *, cutoff=6, phase=0.):
    """H=E.n+Omega b†b+g(c† M c)(b+b†), restricted to N=2."""
    annihilation = []
    for orbital in range(3):
        op = np.zeros((8, 8))
        for state in range(8):
            if (state >> orbital) & 1:
                sign = (-1) ** ((state & ((1 << orbital)-1)).bit_count())
                op[state ^ (1 << orbital), state] = sign
        annihilation.append(op)
    density_ops = np.array([[annihilation[j].T @ annihilation[i]
                             for j in range(3)] for i in range(3)])
    vertex = np.zeros((3, 3), complex)
    vertex[:2, 2] = [1., np.exp(1j*phase)]
    vertex[2, :2] = vertex[:2, 2].conj()
    electronic_h = sum(ENERGIES[i]*density_ops[i, i] for i in range(3))
    interaction = sum(vertex[i, j]*density_ops[j, i]
                      for i in range(3) for j in range(3))
    boson = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    h = (np.kron(electronic_h, np.eye(cutoff))
         + np.kron(np.eye(8), OMEGA*boson.T @ boson)
         + g*np.kron(interaction, boson+boson.T))
    sector = np.array([state*cutoff+m for state in range(8)
                       if state.bit_count() == 2 for m in range(cutoff)])
    ix = np.ix_(sector, sector)
    ops = np.array([[np.kron(op, np.eye(cutoff))[ix] for op in row]
                    for row in density_ops])
    initial = np.zeros(len(sector))
    initial[np.flatnonzero(sector == 3*cutoff)[0]] = 1.
    return h[ix], ops, initial, vertex


def _density(state, ops):
    """rho_ij=<c†_j c_i>, not a quasiparticle occupation relabelling."""
    return np.einsum('a,ijab,b->ij', state.conj(), ops, state)


def _markov_rhs(rho, g, vertex, *, eta=0.):
    """Literal Ain=i Rh, Aout=i Rp, plus the zeroth-order free commutator.

    No production branch, rate, projection or resolvent helper is imported.
    The coherent correlation self-energy is included once by this literal
    Ain/Aout expression. Replacing the free phase energies by their O(g²)
    self-consistent shifts changes this comparison only at O(g⁴). This is a
    fixed-E toy, not a solved instantaneous Hs=hHF+M_H fixed point.
    """
    q = np.eye(3)-rho
    rp = np.empty((3, 3), complex)
    rh = np.empty_like(rp)
    for a in range(3):
        for b in range(3):
            rp[a, b] = g*g*sum((vertex @ q)[a, n]*vertex[n, b]
                /(ENERGIES[b]-ENERGIES[n]-OMEGA+1j*eta) for n in range(3))
            rh[a, b] = g*g*sum((vertex @ rho)[a, n]*vertex[n, b]
                /(ENERGIES[b]-ENERGIES[n]+OMEGA+1j*eta) for n in range(3))
    ain, aout = 1j*rh, 1j*rp
    collision = ain @ q+q @ ain.conj().T-aout @ rho-rho @ aout.conj().T
    return -1j*(ENERGIES[:, None]-ENERGIES[None, :])*rho+collision


def _second_order(g, vertex, time):
    delta = ENERGIES[2]+OMEGA-ENERGIES[:2]
    alpha = g*vertex[:2, 2]*(1.-np.exp(1j*delta*time))/delta
    hole = np.outer(alpha, alpha.conj())
    rho = PROJECTOR.astype(complex)
    rho[:2, :2] -= hole
    rho[2, 2] += np.trace(hole)
    beta = g*vertex[:2, 2]/delta
    return rho, np.outer(beta, beta.conj())


def _finite_memory_rhs(time, value, g, vertex):
    """Order-g² finite-memory equation, with the free commutator retained."""
    rho2 = value.reshape((3, 3))
    delta = ENERGIES[2]+OMEGA-ENERGIES[:2]
    f = -np.expm1(-1j*delta*time)/(1j*delta)
    weights = g*g*np.outer(vertex[:2, 2], vertex[:2, 2].conj())
    source = np.zeros((3, 3), complex)
    source[:2, :2] = -weights*(f[None, :]+f.conj()[:, None])
    source[2, 2] = -np.trace(source[:2, :2])
    return (-1j*(ENERGIES[:, None]-ENERGIES[None, :])*rho2+source).ravel()


def evidence():
    """Return small scalar receipts; the calculation is wholly CPU/local."""
    rows = []
    time = .8
    for g in (.1, .05, .025, .0125):
        h, ops, initial, vertex = _model(g)
        energies, vectors = eigh(h)
        state = vectors @ (np.exp(-1j*energies*time)*(vectors.conj().T @ initial))
        quench = _density(state, ops)
        ground = _density(vectors[:, 0], ops)
        rho2, b = _second_order(g, vertex, time)
        ground2 = PROJECTOR.astype(complex)
        ground2[:2, :2] -= b
        ground2[2, 2] += np.trace(b)
        bare_rhs = _markov_rhs(PROJECTOR, g, vertex)
        rows.append(dict(g=g,
            exact_quench_vs_finite_memory_fro=float(np.linalg.norm(quench-rho2)),
            exact_ground_vs_dressed_order2_fro=float(np.linalg.norm(ground-ground2)),
            bare_markov_rhs_fro=float(np.linalg.norm(bare_rhs)),
            exact_ground_markov_rhs_fro=float(np.linalg.norm(_markov_rhs(ground, g, vertex))),
            bare_filled_blocking_min=float(np.linalg.eigvalsh(-bare_rhs[:2, :2])[0]),
            exact_quench_density_eigenvalues=np.linalg.eigvalsh(quench).tolist(),
            exact_ground_density_eigenvalues=np.linalg.eigvalsh(ground).tolist(),
            trace_error=float(max(abs(np.trace(quench)-2.), abs(np.trace(ground)-2.)))))
    g = .025
    _, _, _, vertex = _model(g)
    ground2, b = _second_order(g, vertex, 0.)
    ground2[:2, :2] -= b
    ground2[2, 2] += np.trace(b)
    free_ground2 = -1j*(ENERGIES[:, None]-ENERGIES[None, :])*ground2
    # Keep only order g²: the collision is evaluated at P. Finite eta adds
    # Lorentzian tails, so the eta=0 GS stationarity proof no longer applies.
    tails = [dict(eta=eta, leading_dressed_rhs_fro=float(np.linalg.norm(
        _markov_rhs(PROJECTOR, g, vertex, eta=eta)+free_ground2)))
        for eta in (0., .001, .005, .01, .2)]
    return dict(status='PASS', scope='Perturbative finite-memory/initial-dressing toy; not material qualification',
                time=time, energies=ENERGIES.tolist(), omega=OMEGA, oscillator_cutoff=6,
                number_sector=2, eta=0., rows=rows, finite_eta_leading_order_tails=tails,
                tail_coupling=g, instantaneous_hs_fixed_point_solved=False)


def test_exact_quench_and_correlated_ground_have_expected_perturbative_scaling():
    rows = evidence()['rows']
    for row in rows:
        assert row['trace_error'] < 2e-14
        for key in ('exact_quench_density_eigenvalues', 'exact_ground_density_eigenvalues'):
            assert min(row[key]) >= -2e-14
            assert max(row[key]) <= 1.+2e-14
    for first, second in zip(rows[:-1], rows[1:]):
        assert abs(first['bare_markov_rhs_fro']/second['bare_markov_rhs_fro']-4.) < 1e-12
        for key in ('exact_quench_vs_finite_memory_fro', 'exact_ground_vs_dressed_order2_fro',
                    'exact_ground_markov_rhs_fro'):
            assert 15.7 < first[key]/second[key] < 16.3
    assert rows[0]['bare_filled_blocking_min'] < -6e-4
    tails = evidence()['finite_eta_leading_order_tails']
    assert tails[0]['leading_dressed_rhs_fro'] < 1e-18
    assert 1.99 < tails[3]['leading_dressed_rhs_fro']/tails[2]['leading_dressed_rhs_fro'] < 2.01


def test_finite_memory_gram_and_initial_slippage_with_complex_vertex():
    g, time = .03, .8
    h, ops, initial, vertex = _model(g, phase=.37)
    rho2, b = _second_order(g, vertex, time)
    energies, vectors = eigh(h)
    state = vectors @ (np.exp(-1j*energies*time)*(vectors.conj().T @ initial))
    assert np.linalg.norm(_density(state, ops)-rho2) < 1.4e-6
    solution = solve_ivp(_finite_memory_rhs, (0., time), np.zeros(9, complex),
                         args=(g, vertex), rtol=2e-11, atol=1e-14)
    assert np.max(abs(solution.y[:, -1].reshape(3, 3)+PROJECTOR-rho2)) < 2e-13
    u = np.diag(np.exp(-1j*ENERGIES[:2]*time))
    bare_hole = b-u @ b @ u.conj().T
    assert np.linalg.eigvalsh(bare_hole)[0] < -1e-5
    # Markov evolution h(t)=U h(0) U†+B−UBU†. The 2B slip gives
    # B+UBU†, whereas the genuinely adiabatic correlated start h(0)=B
    # is stationary at this perturbative order. Neither is an arbitrary GS fit.
    slipped_hole = u @ (2*b) @ u.conj().T+bare_hole
    assert np.linalg.eigvalsh(slipped_hole)[0] >= -1e-18
    assert np.max(abs(u @ b @ u.conj().T+bare_hole-b)) < 1e-18


def test_oscillator_cutoff_does_not_control_the_order2_result():
    densities = []
    for cutoff in (6, 8):
        h, ops, _, _ = _model(.1, cutoff=cutoff)
        _, vectors = eigh(h)
        densities.append(_density(vectors[:, 0], ops))
    assert np.max(abs(densities[0]-densities[1])) < 1e-12


if __name__ == '__main__':
    print(json.dumps(evidence(), indent=2, allow_nan=False))
