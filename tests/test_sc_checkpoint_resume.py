"""SC checkpoint continuation on toy inputs (CPU only, seconds).

1. ``mixing.acceleration.anderson_nojit`` on an affine map: a run stopped
   after an evaluation and resumed from the ``AndersonState`` its ``on_eval``
   saw evaluates the same iterates, bit for bit, as the run that was not
   stopped -- also from the state one map behind the newest output, which is
   what a run that used up its budget leaves (its last map writes no state,
   so the continuation evaluates that map again). The map is chosen so the
   best pair leaves a depth-2 window and the secant fallback fires, so both
   restore branches run.
   The first step is half a plain step: on an affine map the second point
   is the same as after a full first step, and along a Picard eigenvalue
   of -2.5 the map-1 residual falls instead of growing 2.5x.
2. ``gw.sigma_box_plan._fit_fixed_sc_rules``: a held Sigma rule that the
   resumed map refuses becomes an escape refit, not a crash.
3. ``gw.sc_iteration._fit_band_carrier``: a cube read on a non-square mesh
   (another padded extent per band axis) returns to the square carrier.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
CPU = jax.devices("cpu")[0]


class _Stop(Exception):
    pass


def _affine(seed):
    """An expansive affine map: eigenvalues down to -30, -10 or -3 make a new
    residual exceed the window (fallback) and keep the best pair in the past."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((18, 18)))
    m = q @ np.diag(np.linspace((-30.0, -10.0, -3.0)[seed % 3], 0.8, 18)) @ q.T
    c = rng.standard_normal(18) + 1j * rng.standard_normal(18)
    return m, c


def _run(seed, maxit, *, stop_at=None, resume=None):
    from mixing.acceleration import anderson_nojit
    m, c = _affine(seed)
    inputs, states, lines = [], [], []

    def residual(x):
        inputs.append(np.asarray(x))
        return (m @ x.reshape(-1) + c).reshape(x.shape) - x

    def on_eval(state):
        states.append(state)
        if stop_at is not None and len(state.res) - 1 == stop_at:
            raise _Stop

    with jax.default_device(CPU):
        x0 = jax.numpy.zeros((2, 3, 3), complex)
        try:
            anderson_nojit(residual, x0, m=2, maxit=maxit, tol=0.0,
                           print_fn=lines.append, on_eval=on_eval, resume=resume)
        except _Stop:
            pass
    return inputs, states, lines


def test_anderson_resume_is_the_same_trajectory():
    seed = next((s for s in range(96)
                 if (lambda r: any("fallback" in line for line in r[2])
                     and any(st.best is not None for st in r[1]))(_run(s, 10))), None)
    assert seed is not None, "no toy map exercised the fallback and an out-of-window best"
    full, states, _ = _run(seed, 10)
    for k in range(len(states)):  # stopped after evaluation k, resumed
        _, kept, _ = _run(seed, 10, stop_at=k)
        resumed, _, _ = _run(seed, 10, resume=lambda st=kept[-1]: st)
        assert len(resumed) == len(full) - (k + 1)
        assert all(np.array_equal(a, b) for a, b in zip(full[k + 1:], resumed))
    short, kept, _ = _run(seed, 4)  # budget used up: newest state is one map behind
    assert len(short) == 5 and len(kept[-1].res) == 4
    resumed, _, _ = _run(seed, 10, resume=lambda: kept[-1])
    assert all(np.array_equal(a, b) for a, b in zip(full[4:], resumed))


def test_half_first_step():
    from mixing.acceleration import AndersonState, anderson_nojit
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((18, 18)))
    m = q @ np.diag(np.linspace(-2.5, 0.8, 18)) @ q.T
    c = rng.standard_normal(18) + 1j * rng.standard_normal(18)
    jnp = jax.numpy

    def run(resume=None):
        seen = []

        def residual(x):
            seen.append(np.asarray(x))
            return (m @ x.reshape(-1) + c).reshape(x.shape) - x
        with jax.default_device(CPU):
            out = anderson_nojit(residual, jnp.zeros((2, 3, 3), complex), m=4,
                                 maxit=2, tol=0.0, resume=resume)
        return seen, np.asarray(out.residual_norms)

    half, res = run()
    assert res[1] <= res[0]                      # a full step: 2.5x res[0]
    x0 = half[0]
    f0 = (m @ x0.reshape(-1) + c).reshape(x0.shape) - x0
    x1 = x0 + f0                                 # the full first step
    f1 = (m @ x1.reshape(-1) + c).reshape(x1.shape) - x1
    stack = np.zeros((4,) + x0.shape, complex)
    stack[0] = x0
    fstack = np.zeros_like(stack)
    fstack[0] = f0
    state = AndersonState(jnp.asarray(stack), jnp.asarray(fstack), 1, 1,
                          jnp.asarray(x1), jnp.asarray(f1), None,
                          (float(np.linalg.norm(f0)), float(np.linalg.norm(f1))), False)
    full, _ = run(resume=lambda: state)
    assert np.linalg.norm(full[0] - half[2]) <= 1e-10 * np.linalg.norm(half[2])


def test_refused_held_rule_is_an_escape(monkeypatch):
    from gw import sigma_box_plan as sbp

    def refuse(spec, rule, eps, **_):
        raise RuntimeError("factored log growth exceeds the cap")
    monkeypatch.setattr(sbp, "_BOX_RULE_BUILDER", lambda box, eps, mass_cap: object())
    monkeypatch.setattr(sbp, "_accept_rule", refuse)
    monkeypatch.setattr(sbp, "_sc_padded_box_spec", lambda spec, eta, **_: dict(
        spec, sc_state_pad_ev=(2.0, 2.0), sc_certified_states_ry=(0.0, 1.0),
        sc_certified_poles_ry=(0.0, 1.0, 0.0, 0.0), sc_certified_omega_ry=(0.0, 1.0)))
    monkeypatch.setattr(sbp, "fit_sigma_box_specs", lambda specs, eta, **_: (
        [dict(node_count=7, rule_box=spec["box"], relative=True, rule_source="built")
         for spec in specs], []))
    spec = dict(name="w", box=(0.1, 0.2, 0.01, 0.01), kind="sign_definite_positive")
    session = dict(call_count=1, eta_ry=0.01, eps=1e-4, rules={"w": dict(
        held_rule=dict(rule_box=(0.05, 0.3, 0.01, 0.02), analytic_line=False,
                       node_digest="0"))})
    fits, _, receipt = sbp._fit_fixed_sc_rules([spec], 0.01, eps=1e-4, scope=None,
                                               session=session)
    assert receipt["rebuilt"] == ("w",) and fits[0]["node_count"] == 7
    assert "held rule refused" in dict(receipt["recompute_reasons"])["w"]
    assert "held_rule" not in session["rules"]["w"]


def test_band_carrier_from_a_non_square_read():
    from gw.sc_iteration import _fit_band_carrier
    sharding = jax.sharding.SingleDeviceSharding(CPU)
    rng = np.random.default_rng(0)
    logical = rng.standard_normal((2, 3, 5, 5))
    read = np.zeros((2, 3, 6, 8))
    read[..., :5, :5] = logical
    with jax.default_device(CPU):
        out = np.asarray(_fit_band_carrier(jax.numpy.asarray(read), 5, 6, sharding))
        square = jax.numpy.asarray(read[..., :6, :6])
        assert _fit_band_carrier(square, 5, 6, sharding) is square
    assert out.shape == (2, 3, 6, 6)
    assert np.array_equal(out[..., :5, :5], logical) and not out[..., 5:, :].any()
    assert not out[..., :, 5:].any()


def test_deck_digest_reads_lines_as_the_parser_does(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import common.parallel_transport as pt
    from gw.sc_iteration import _sc_checkpoint_identity
    monkeypatch.setattr(pt, "wfn_fingerprint", lambda wfn: "wfn")

    def digest(text):
        deck = tmp_path / "deck.in"
        deck.write_text(text)
        inputs = SimpleNamespace(config=SimpleNamespace(input_file=str(deck), bispinor=False),
                                 wfn=None, wfn_fingerprint_binding=None)
        return _sc_checkpoint_identity(inputs, (1, 2, 2), 20)["deck_sha256"]
    base = digest("[cohsex]\nnband = 60\nsc_max_iter = 30\n")
    assert digest("[cohsex]\nnband=60  # more bands later\nSC_MAX_ITER: 40\n") == base
    assert digest("[cohsex]\nnband = 61\nsc_max_iter = 30\n") != base
