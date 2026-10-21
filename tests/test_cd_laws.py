"""CPU pins of the CD laws against the Fe harness of record (ordered_cd.py / selector_jets.py).

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE names the clause; unit and CPU cells).
The harness laws run in their own layout ([k, bra, ket, l] with a per-bra E grid) with a
bra-independent x, so both layouts describe one Sigma_c law; equality to 1e-12 relative.
"""
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fixtures_cd"))
import ordered_cd as ref       # noqa: E402  (Fe harness copy, sha in fixtures_cd/ordered_cd.sha256)
import selector_jets as jets   # noqa: E402

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp        # noqa: E402
from gw.cd import laws         # noqa: E402
from gw.cd.rule import tan_mapped_gauss_rule  # noqa: E402

EV = 13.605693122994


def _case(seed=0, k=2, m=3, l=4, e=5, h_ev=0.0625):
    rng = np.random.default_rng(seed)
    eta = 0.25 / EV
    x = rng.uniform(-3.0, 3.0, (k, e, l)) / EV
    f = np.where(rng.uniform(size=(k, l)) < 0.5, 1.0, 0.0)
    f[0, 0] = 0.37                          # one fractional occupation
    def c(): return rng.normal(size=(k, m, m, l)) + 1j * rng.normal(size=(k, m, m, l))
    a, d, at, dt = c(), c(), c(), c()
    xh = np.broadcast_to(x[:, None], (k, m, e, l))
    fh = np.broadcast_to(f[:, None, None, :], (k, m, e, l))
    return eta, x, f, a, d, at, dt, xh, fh, h_ev / EV


def _ours(out_ref):
    """ref [k, bra, ket, E] -> ours [E, k, bra, ket]."""
    return np.moveaxis(out_ref, -1, 0)


def test_imag_node_matches_ordered_cd():
    eta, x, f, a, *_, xh, fh, _h = _case()
    u, w = 0.7 / EV, 0.3 / EV
    mine = np.asarray(laws.imag_raw_node(jnp.asarray(a), u, w, jnp.asarray(x), jnp.asarray(f), eta))
    rm, rp = ref.imag_raw_node(a, u, w, xh, fh, eta=eta)
    np.testing.assert_allclose(mine, _ours(rm + rp), rtol=1e-12, atol=1e-14)


def test_anchor_and_subtraction_match_ordered_cd():
    eta, x, f, a, d, *_, xh, fh, _h = _case()
    beta = tuple(float(b) for b in ref.BETA)   # the harness unit constant (13.605693122994), not common.units
    total, cp, cm = laws.anchor_part(jnp.asarray(a), jnp.asarray(d), jnp.asarray(x), jnp.asarray(f), eta, beta)
    rt, rcp, rcm = ref.anchor_part(a, d, xh, fh, jets, eta=eta)
    np.testing.assert_allclose(np.asarray(total), _ours(rt), rtol=1e-12, atol=1e-13)
    for mine, theirs in zip(cp + cm, rcp + rcm):
        np.testing.assert_allclose(np.asarray(mine), theirs, rtol=1e-12)
    u, w = tan_mapped_gauss_rule(12, eta)
    sums = laws.imag_anchor_sums(jnp.asarray(u), jnp.asarray(w), jnp.asarray(x), jnp.asarray(f), eta, beta)
    rs = ref.imag_anchor_sums(list(u), list(w), xh, fh, eta=eta)
    sub = laws.anchor_subtraction(cp, cm, sums)
    rsub = ref.imag_anchor_subtraction(rcp, rcm, rs)
    np.testing.assert_allclose(np.asarray(sub), _ours(rsub), rtol=1e-12, atol=1e-13)


def test_real_knots_match_ordered_cd_real_part():
    eta, x, f, a, d, at, dt, xh, fh, h = _case()
    j_max = int(np.ceil(np.abs(x).max() / h)) + 2
    real_ev = np.arange(j_max) * h * EV
    k, m, _, l = a.shape
    rng = np.random.default_rng(1)
    def c(): return rng.normal(size=(j_max, k, m, m, l)) + 1j * rng.normal(size=(j_max, k, m, m, l))
    vals, ders, vals_t, ders_t = c(), c(), c(), c()
    mine = 0
    for j in range(j_max):
        z = real_ev[j] / EV + 1j * eta
        mine = mine + np.asarray(laws.real_knot(j, z, jnp.asarray(vals[j]), jnp.asarray(ders[j]),
                                                jnp.asarray(vals_t[j]), jnp.asarray(ders_t[j]),
                                                jnp.asarray(x), jnp.asarray(f), h))
    theirs = ref.real_part(vals, ders, vals_t, ders_t, xh, fh, real_ev, jets, eta=eta, n_active=l,
                           h=h * EV, base_ev=h * EV)
    np.testing.assert_allclose(mine, _ours(theirs), rtol=1e-12, atol=1e-13)


def test_tan_rule_is_the_fe_harness_split_rule():
    n, eta_ev = 96, 0.25
    u, w = tan_mapped_gauss_rule(n, eta_ev)
    low = round(n / 5)
    parts = []
    for count, left, right in ((low, 0.0, np.pi / 4), (n - low, np.pi / 4, np.pi / 2)):
        q, wq = np.polynomial.legendre.leggauss(count)
        theta = (right - left) * q / 2 + (right + left) / 2
        parts.append((eta_ev * np.tan(theta), wq * (right - left) * eta_ev / (2 * np.cos(theta) ** 2)))
    np.testing.assert_array_equal(u, np.concatenate([p[0] for p in parts]))
    np.testing.assert_array_equal(w, np.concatenate([p[1] for p in parts]))
    d = 0.5
    np.testing.assert_allclose(np.sum(w / (u ** 2 + d ** 2)), np.pi / (2 * d), rtol=1e-9)


# --------------------------------------------------------------------------- dSigma/dE laws (*_dx)
def _dx_case(seed=3, k=2, m=3, l=5, e=4, h_ev=0.0625, n_nodes=12):
    """Random contractions; x off every knot and off 0 by >= 0.1 h, so a central difference stays on one branch."""
    rng = np.random.default_rng(seed)
    eta, h = 0.25 / EV, h_ev / EV
    x = rng.choice([-1.0, 1.0], (k, e, l)) * (rng.integers(0, 30, (k, e, l)) + rng.uniform(0.1, 0.9, (k, e, l))) * h
    f = np.where(rng.uniform(size=(k, l)) < 0.5, 1.0, 0.0)
    f[0, 0] = 0.37
    j_max = int(np.ceil(np.abs(x).max() / h)) + 2

    def c(*shape):
        return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    knots = [c(4, k, m, m, l) for _ in range(j_max)]                 # a, d, at, dt per knot
    u, w = (np.asarray(v) / EV for v in tan_mapped_gauss_rule(n_nodes, eta * EV))
    return dict(eta=eta, h=h, x=x, f=jnp.asarray(f), a=c(k, m, m, l), a0=c(k, m, m, l), d0=c(k, m, m, l),
                knots=knots, u=u, w=w, beta=tuple(float(b) for b in laws.BETA_RY))


def _terms(cs, x, derivative=False):
    """The three CD term families [E, k, m, n] at x (``derivative``: their *_dx laws on the same operators)."""
    x, f, eta, h, beta = jnp.asarray(x), cs["f"], cs["eta"], cs["h"], cs["beta"]
    kernel = laws.imag_node_kernels_dx if derivative else laws.imag_node_kernels
    imag = sum(laws.apply(*laws.imag_node_terms(cs["a"], u, w, x, f, eta, kernel=kernel))
               for u, w in zip(cs["u"], cs["w"]))
    total, cp, cm = laws.anchor_part(cs["a0"], cs["d0"], x, f, eta, beta)
    if derivative:
        total = laws.anchor_part_dx(cp, cm, x, f, eta, beta)
    anchor = total - laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(cs["u"], cs["w"], x, f, eta, beta,
                                                                           kernel=kernel))
    real = 0
    for j, (a, d, at, dt) in enumerate(cs["knots"]):
        z = j * h + 1j * eta
        for A, D, neg in ((a, d, False), (at, dt, True)):
            sel = laws.real_knot_select(j, laws.hermite_branch_coefficients(x, f, h, neg, derivative=derivative))
            real = real + laws.apply(laws.real_knot_operators(z, A, D), sel)
    return {name: np.asarray(v) for name, v in dict(imag=imag, anchor=anchor, real=real).items()}


def test_dx_laws_are_the_derivatives_of_the_terms():
    """Each family's *_dx law equals its central difference (x shifted by +-1e-6 Ry, on one branch)."""
    cs, dE = _dx_case(), 1e-6
    exact = _terms(cs, cs["x"], derivative=True)
    plus, minus = _terms(cs, cs["x"] + dE), _terms(cs, cs["x"] - dE)
    for name in exact:
        fd = (plus[name] - minus[name]) / (2 * dE)
        rel = np.abs(exact[name] - fd).max() / np.abs(exact[name]).max()
        assert rel < 1e-6, (name, rel)


def test_dx_laws_take_the_right_branch_at_x_zero():
    """At x = 0 every family's *_dx law is its right derivative (second-order one-sided difference on x >= 0).

    1e-8 Ry: the imaginary rule has nodes at u ~ eta, where the (x + i eta - i u) denominator is ~1e-3 Ry wide.
    """
    cs, dE = _dx_case(seed=4), 1e-8
    zero = np.zeros_like(cs["x"])
    exact = _terms(cs, zero, derivative=True)
    f0, f1, f2 = (_terms(cs, zero + s * dE) for s in (0, 1, 2))
    for name in exact:
        fd = (-3 * f0[name] + 4 * f1[name] - f2[name]) / (2 * dE)
        rel = np.abs(exact[name] - fd).max() / np.abs(exact[name]).max()
        assert rel < 1e-5, (name, rel)


def test_row_apply_binds_E_to_the_bra():
    cs = _dx_case()
    m = int(cs["a"].shape[1])
    x = jnp.asarray(cs["x"][:, :m])                                   # [k, E = m, l]
    w = laws.imag_node_kernels(cs["u"][0], cs["w"][0], x, cs["f"], cs["eta"])[0]
    full = np.asarray(laws.apply(cs["a"], w))                         # [E, k, m, n]
    rows = np.asarray(laws.row_apply(cs["a"], w))                     # [k, m, n]
    np.testing.assert_allclose(rows, np.stack([full[i, :, i] for i in range(m)], axis=1), rtol=1e-13, atol=1e-13)
