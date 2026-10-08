"""Closed-form tests of ordered CD, including fractional and remote states."""
import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw import contour_reference as cd


def _plant(eta_ev=.25, n=192, step_ev=.03125, fractional=True, ordered=True):
    rng = np.random.default_rng(20261006)
    eta, h = eta_ev / RYD_TO_EV, step_ev / RYD_TO_EV
    omega = np.asarray([.7, 3., 18., 75.]) / RYD_TO_EV
    b = rng.normal(size=(2, 4, 3, 2)) + 1j*rng.normal(size=(2, 4, 3, 2))
    residue = .0001 * np.einsum("psiq,psjq->psij", b, b.conj())
    if not ordered:
        residue[1] = residue[0].conj()
    p = rng.normal(size=(1, 2, 5, 3)) + 1j*rng.normal(size=(1, 2, 5, 3))
    energy = np.asarray([-80., -5., 0., 4., 100.]) / RYD_TO_EV
    occupation = np.asarray([1., .8, .5, .2, 0.]) if fractional else np.asarray([1., 1., 1., 0., 0.])
    # Own-energy singularities, off-node queries, semicore and high conduction.
    external = np.asarray([[-80., -5., -4.999, 100.], [0., 4., 4.001, 100.01]]) / RYD_TO_EV
    x = external[None, :, :, None] - energy[None, None, None, :]
    f = np.broadcast_to(occupation, x.shape)
    contract = lambda a: np.einsum("kalm,mn,kbln->kabl", p.conj(), a, p)
    rp = np.stack([contract(a) for a in residue[0]])
    rm = np.stack([contract(a.T) for a in residue[1]])
    exact = sum(np.einsum("kabl,kael->kabe", rp[s], (1-f)/(x-om+1j*eta))
                + np.einsum("kabl,kael->kabe", rm[s], f/(x+om-1j*eta))
                for s, om in enumerate(omega))
    real = np.arange(round(181. / step_ev) + 1) * h
    u, w = cd.imaginary_rule(n, eta, scale=10. / RYD_TO_EV)
    z = np.concatenate((real + 1j*eta, 1j*u))
    samples = []
    for partner in (False, True):
        a, b_ = (rm, rp) if partner else (rp, rm)
        value = np.zeros((len(z),) + rp.shape[1:], complex)
        derivative = value.copy()
        for s, om in enumerate(omega):
            zp = (z-om)[:, None, None, None, None]
            zm = (z+om)[:, None, None, None, None]
            value += -a[s]/zp + b_[s]/zm
            derivative += a[s]/zp**2 - b_[s]/zm**2
        derivative /= 2*z[:, None, None, None, None]
        samples.extend((value, derivative))
    schedule = dict(real_nodes=real, imaginary=[dict(indices=np.arange(len(real), len(z)), nodes=u, weights=w)])
    return samples, x, f, schedule, eta, exact


@pytest.mark.parametrize("fractional,ordered", [(False, False), (True, False), (False, True), (True, True)])
def test_ordered_cd_closed_form(fractional, ordered):
    samples, x, f, schedule, eta, exact = _plant(fractional=fractional, ordered=ordered)
    got = next(iter(cd.integrate_ordered(*samples, x, f, schedule, eta=eta).values()))
    assert np.max(np.abs(got-exact)) / np.max(np.abs(exact)) < 2e-7


def test_cd_partner_orientation_negative_control():
    samples, x, f, schedule, eta, exact = _plant()
    wrong = next(iter(cd.integrate_ordered(samples[0], samples[1], samples[0], samples[1],
                                          x, f, schedule, eta=eta).values()))
    assert np.max(np.abs(wrong-exact)) / np.max(np.abs(exact)) > 1e-3


def test_infinite_rule_and_residue_coverage():
    eta = .25 / RYD_TO_EV
    for n in (64, 128, 256):
        u, w = cd.imaginary_rule(n, eta, scale=eta)
        assert not np.any(u == eta)
        # Independent analytic integral to infinity, including the map's tail.
        assert abs(np.sum(w/(u*u+eta*eta)) - np.pi/(2*eta)) < 1e-8
    samples, x, f, schedule, eta, _ = _plant()
    shorter = schedule["real_nodes"][:100]
    with pytest.raises(ValueError, match="coverage"):
        cd.real_part(*(a[:len(shorter)] for a in samples), x, f, shorter, eta=eta)


def test_rejects_nonuniform_grid_and_noninteger_coarsening():
    samples, x, f, schedule, eta, _ = _plant(n=32)
    nodes = schedule["real_nodes"].copy()
    nodes[3] *= 1.01
    with pytest.raises(ValueError, match="uniform"):
        cd.real_part(*(a[:len(nodes)] for a in samples), x, f, nodes, eta=eta)
    with pytest.raises(ValueError, match="integer"):
        cd.real_part(*(a[:len(nodes)] for a in samples), x, f, schedule["real_nodes"],
                     eta=eta, spacing=1.5*(nodes[1]-nodes[0]))


def _one_pole_matrix(eta_ev, occupation, *, n_imaginary=192):
    """An ordered PSD Lehmann model, built independently of CD routines.

    Both residue matrices are Hermitian positive definite and genuinely
    complex. The second parent reverses them, as required by the q/-q
    particle-hole relation. No response fit or shared-pole evaluator supplies
    the expected self-energy.
    """
    eta = eta_ev / RYD_TO_EV
    omega = .4
    rp = np.asarray([[.04, .006 + .009j], [.006 - .009j, .027]])
    rm = np.asarray([[.025, -.008 + .004j], [-.008 - .004j, .035]])
    rp = rp[None, :, :, None]
    rm = rm[None, :, :, None]
    queries = np.asarray([-.6, -.4, -.1, -1e-9, 0., 1e-9, .1, .4, .6])
    x = np.broadcast_to(queries[None, None, :, None], (1, 2, len(queries), 1))
    f = np.full(x.shape, occupation)
    # η/8 misses the closed-form tolerance at a pole by several ppm;
    # refine the actual interpolation grid rather than relaxing the gate.
    h = eta / 32
    real = np.arange(int(np.ceil(np.max(np.abs(x)) / h)) + 2) * h
    u, w = cd.imaginary_rule(n_imaginary, eta, scale=omega)
    z = np.concatenate((real + 1j * eta, 1j * u))
    samples = []
    for a, b in ((rp, rm), (rm, rp)):
        zp = (z - omega)[:, None, None, None, None]
        zm = (z + omega)[:, None, None, None, None]
        value = -a / zp + b / zm
        derivative_z = a / zp**2 - b / zm**2
        samples.extend((value, derivative_z / (2 * z[:, None, None, None, None])))
    schedule = dict(real_nodes=real, imaginary=[dict(
        indices=np.arange(len(real), len(z)), nodes=u, weights=w)])
    return samples, x, f, schedule, eta, omega, rp, rm


@pytest.mark.parametrize("eta_ev", [.05, .25, 1.])
@pytest.mark.parametrize("occupation", [0., .25, .5, 1.])
@pytest.mark.parametrize("analytic_convention", ["time_ordered_fractional", "retarded"])
def test_ordered_single_pole_own_energy_and_fractional_sheets(eta_ev, occupation, analytic_convention):
    samples, x, f, schedule, eta, omega, rp, rm = _one_pole_matrix(eta_ev, occupation)
    # Closed-form denominators declare the analytic sheet independently of
    # the numerical integration. Retarded uses the upper sheet for both
    # occupation branches; historical time-ordered uses the lower for holes.
    occupied_sheet = -1 if analytic_convention == "time_ordered_fractional" else 1
    expected = (np.einsum("kabl,kael->kabe", rp, (1 - f) / (x - omega + 1j * eta))
                + np.einsum("kabl,kael->kabe", rm, f / (x + omega + occupied_sheet * 1j * eta)))
    got = next(iter(cd.integrate_ordered(*samples, x, f, schedule, eta=eta,
                                       analytic_convention=analytic_convention).values()))
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 2e-7


def test_time_ordered_imaginary_part_is_not_a_retarded_linewidth():
    samples, x, f, schedule, eta, omega, rp, rm = _one_pole_matrix(.25, .5)
    time_ordered = next(iter(cd.integrate_ordered(*samples, x, f, schedule, eta=eta).values()))
    retarded = (np.einsum("kabl,kael->kabe", rp, (1 - f) / (x - omega + 1j * eta))
                + np.einsum("kabl,kael->kabe", rm, f / (x + omega + 1j * eta)))
    integrated_retarded = next(iter(cd.integrate_ordered(*samples, x, f, schedule, eta=eta,
                                                        analytic_convention="retarded").values()))
    assert np.max(np.abs(integrated_retarded - retarded)) / np.max(np.abs(retarded)) < 2e-7
    # Diagonal real parts coincide in this finite pole model; occupied
    # imaginary parts, hence the inferred linewidth, demonstrably differ.
    for band in range(2):
        assert np.max(np.abs(time_ordered[0, band, band].real
                             - retarded[0, band, band].real)) < 2e-7 * np.max(np.abs(retarded))
    assert np.max(np.abs(time_ordered.imag - retarded.imag)) > .1
    assert np.all(np.imag(np.diagonal(retarded[0], axis1=0, axis2=1)) <= 0)


def test_infinite_rule_scale_refinement_checks_a_distinct_tail():
    eta = .25 / RYD_TO_EV
    beta = 30. / RYD_TO_EV
    values = []
    for n in (128, 256):
        u, w = cd.imaginary_rule(n, eta, scale=10. / RYD_TO_EV)
        values.append(np.sum(w / (u*u + beta*beta)))
    exact = np.pi / (2 * beta)
    assert abs(values[-1] - exact) < 1e-10 * exact
    assert abs(values[-1] - values[0]) < 1e-10 * exact


def _exact_node_one_pole(eta_ev, occupation, convention, *, masked=False,
                         omit_node=None, perturb_node=None):
    """A complete CD integral whose real samples have no interpolation error.

    This plant evaluates an independently specified ordered Lehmann model
    at each crossing. It deliberately supplies nonzero invalid coefficients
    when masked, so a successful result requires every contour term to honor
    the physical-band mask rather than relying on zero padded wavefunctions.
    """
    _, x, f, _, eta, omega, rp, rm = _one_pole_matrix(eta_ev, occupation)
    valid, n_active = None, None
    if masked:
        x = x + np.asarray([0., .03, .19])[None, None, None, :]
        f = np.broadcast_to(np.asarray([occupation, .7, .2]), x.shape)
        scale = np.asarray([1., 7., 1000.])[None, None, None, :]
        rp, rm = rp * scale, rm * scale
        valid = np.asarray([True, False, True])[None, None, None, :]
        n_active = 2
    physical = np.ones(x.shape, bool)
    if valid is not None:
        physical &= np.broadcast_to(valid, x.shape)
        physical &= np.arange(x.shape[-1])[None, None, None, :] < n_active

    def value(z, partner=False):
        a, b = (rm, rp) if partner else (rp, rm)
        return -a / (z - omega) + b / (z + omega)

    def slope(z):
        return (rp / (z - omega)**2 - rm / (z + omega)**2) / (2 * z)

    kwargs = dict(eta=eta, analytic_convention=convention,
                  n_active=n_active, band_valid=valid)
    total, cp, cm, beta = cd.anchor_part(value(1j * eta), slope(1j * eta),
                                        x, f, **kwargs)
    u, w = cd.imaginary_rule(192, eta, scale=omega)
    for ui, wi in zip(u, w):
        total += cd.imag_remainder_node(value(1j * ui), ui, wi, x, f,
                                        cp, cm, beta, **kwargs)
    residue_weight = np.where(x < 0, f, 1 - f)
    nodes = np.unique(np.abs(x)[physical & (residue_weight != 0)])
    for i, node in enumerate(nodes):
        if i == omit_node:
            continue
        declared = np.nextafter(node, np.inf) if i == perturb_node else node
        total += cd.real_residue_node(value(node + 1j * eta),
                                      value(node + 1j * eta, partner=True),
                                      x, f, declared, **kwargs)
    occupied_sheet = -1 if convention == "time_ordered_fractional" else 1
    expected = (np.einsum("kabl,kael->kabe", rp,
                           np.where(physical, (1 - f) / (x - omega + 1j * eta), 0))
                + np.einsum("kabl,kael->kabe", rm,
                            np.where(physical, f / (x + omega + occupied_sheet * 1j * eta), 0)))
    return total, expected, nodes


@pytest.mark.parametrize("eta_ev", [.05, .25, 1.])
@pytest.mark.parametrize("occupation", [0., .25, .5, 1.])
@pytest.mark.parametrize("convention", ["time_ordered_fractional", "retarded"])
def test_exact_crossing_integral_matches_independent_ordered_poles(eta_ev, occupation, convention):
    got, expected, nodes = _exact_node_one_pole(eta_ev, occupation, convention)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 2e-7
    if occupation != 1:
        assert 0. in nodes


@pytest.mark.parametrize("convention", ["time_ordered_fractional", "retarded"])
def test_exact_crossings_mask_nonzero_invalid_coefficients_in_all_contour_terms(convention):
    got, expected, _ = _exact_node_one_pole(.25, .25, convention, masked=True)
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) < 2e-7


@pytest.mark.parametrize("failure", ["omit", "perturb"])
def test_exact_crossing_coverage_is_not_rounded_or_silently_complete(failure):
    # Every sample is otherwise exact; dropping the x=0 crossing or moving its
    # declared node by one representable float must expose the missing term.
    kwargs = {"omit_node": 0} if failure == "omit" else {"perturb_node": 0}
    got, expected, nodes = _exact_node_one_pole(.25, .25, "retarded", **kwargs)
    assert nodes[0] == 0.
    assert np.max(np.abs(got - expected)) / np.max(np.abs(expected)) > 1e-3


def test_exact_crossing_rejects_invalid_frequency_and_nonboolean_mask():
    samples, x, f, _, eta, _, _, _ = _one_pole_matrix(.25, .25)
    for node in (-1., np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            cd.real_residue_node(samples[0][0], samples[2][0], x, f, node, eta=eta)
    with pytest.raises(ValueError, match="boolean"):
        cd.real_residue_node(samples[0][0], samples[2][0], x, f, 0., eta=eta,
                             band_valid=np.ones(x.shape, np.int32))
