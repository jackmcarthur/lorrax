"""The CT round's held CC/TT span widths: grow-only, logged, and inert."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


def _sector(counts, cap):
    active = np.zeros((len(counts), cap), bool)
    for parent, count in enumerate(counts):
        active[parent, :count] = True
    return dict(signed=(None, None, active))


def test_span_widths_are_held_from_the_first_bound_round_and_grow_only():
    from gw.shared_pole_sectors import cross_span_widths

    # No binding (one-shot, SC map 0): each round's own ladder width.
    one_shot = SimpleNamespace()
    assert cross_span_widths(one_shot, [_sector([1100, 1000], 2344), _sector([1500, 1400], 2692)]) == (
        [1152, 1536], [1152, 1536])
    assert not hasattr(one_shot, "shared_pole_rank_capacity")

    held = {}
    meta = SimpleNamespace(shared_pole_rank_capacity=held)
    # Map 1, round 1 sets the capacity silently.
    assert cross_span_widths(meta, [_sector([1100, 1000], 2344), _sector([1500, 1400], 2692)]) == (
        [1152, 1536], [1152, 1536])
    assert held == {"CC": 1152, "TT": 1536}
    # Round 2: CC crosses a ladder step (logged), TT drops one (held).
    assert cross_span_widths(meta, [_sector([1200, 1000], 2344), _sector([1300, 1400], 2692)]) == (
        [1280, 1408], [1280, 1536])
    assert held["_events"] == ["shared-pole CT span (CC): retained rank 1200 exceeds the held "
                               "width 1152; grown to 1280"]
    held.pop("_events")
    # Later maps: ranks fall back, the widths never shrink, nothing is logged.
    assert cross_span_widths(meta, [_sector([1100, 1000], 2344), _sector([1300, 1400], 2692)]) == (
        [1152, 1408], [1280, 1536])
    assert "_events" not in held and held["CC"] == 1280 and held["TT"] == 1536
    # A held width never exceeds the sector's signed carrier.
    assert cross_span_widths(meta, [_sector([1100, 1000], 1200), _sector([1300], 1400)])[1] == [1200, 1400]


def test_held_span_columns_are_exact_zeros_that_leave_the_ct_model_unchanged():
    """Compaction past the live width adds zero Y/c columns; the joint pencil
    and the zero-row-safe eigensolver keep them out of the CT model."""
    from gw.shared_pole_local import _mm, zero_row_safe_eigh
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_sectors import (_compact_sector_equations, joint_sector_pencil,
                                        reduce_sector_pencil)

    adj = lambda a: np.swapaxes(a.conj(), -1, -2)
    # A retained span with one active column among four, stored out of order.
    y = np.zeros((1, 2, 4), complex)
    y[0, :, 2] = [1., .5j]
    c = np.zeros((1, 3, 4), complex)
    c[0, :, 2] = [.3, .1j, -.2]
    mu = np.array([[1., 1., .6, 1.]])
    active = np.array([[False, False, True, False]])
    narrow = _compact_sector_equations(y, (c, mu, active), width=1)
    wide = _compact_sector_equations(y, (c, mu, active), width=3)
    np.testing.assert_array_equal(np.asarray(wide[0])[..., :1], np.asarray(narrow[0]))
    assert not np.any(np.asarray(wide[0])[..., 1:])
    assert not np.any(np.asarray(wide[1][0])[..., 1:])
    assert not np.any(np.asarray(wide[1][2])[..., 1:])

    # The two-state CT plant of tests/multi_device/shared_pole_sectors_p4.py
    # (ordered data), solved at the live widths and with held padding.
    yc = np.array([[1.], [0.]], complex)
    yt = np.array([[.3], [np.sqrt(.91)]], complex)
    cc = np.array([[1., .2j], [.1, .4]], complex)
    tt = np.array([[.3j, 1.], [.6, .2j], [.2, -.1j]], complex)
    value = np.array([[.6, .15j], [-.15j, -.4]], complex)
    vc = (adj(yc) @ value @ yc).real.diagonal()
    vt = (adj(yt) @ value @ yt).real.diagonal()
    cross = (adj(yc) @ yt, adj(yc) @ value @ yt)
    eigh = zero_row_safe_eigh(jnp.linalg.eigh)

    def model(pad_c, pad_t):
        span = lambda k, pad: np.pad(np.eye(k, dtype=complex), ((0, 0), (0, pad)))
        values = lambda v, pad: np.pad(v, (0, pad), constant_values=1.)
        charge = (span(1, pad_c), values(vc, pad_c), cc @ yc, tt @ yc)
        current = (span(1, pad_t), values(vt, pad_t), tt @ yt, cc @ yt)
        batch = lambda t: tuple(jnp.asarray(a)[None] for a in t)
        pencil = joint_sector_pencil(batch(charge), batch(current), batch(cross), matmul=_mm)
        return reduce_sector_pencil(pencil, eigh=eigh, matmul=_mm, gates=gates)

    (c0, t0, lam0, active0), diag0 = model(0, 0)
    (c1, t1, lam1, active1), diag1 = model(2, 3)
    assert bool(jnp.all(diag1["gram_valid"])) and bool(jnp.all(diag1["retained_metric_positive"]))
    assert int(diag0["retained_rank"][0]) == int(diag1["retained_rank"][0]) == 2
    assert not np.any(np.asarray(c1)[..., ~np.asarray(active1)[0]])
    np.testing.assert_allclose(np.sort(np.asarray(lam1)[active1]), np.sort(np.asarray(lam0)[active0]),
                               rtol=0, atol=1e-13)
    for z in (.7 + .2j, 1.3 + .6j, 2j):
        evaluate = lambda c, t, lam, act: (c * (act / (z * lam - 1))[:, None, :]) @ adj(np.asarray(t))
        exact = cc @ np.linalg.solve(z * value - np.eye(2), adj(tt))
        held = evaluate(*map(np.asarray, (c1, t1, lam1, active1)))[0]
        live = evaluate(*map(np.asarray, (c0, t0, lam0, active0)))[0]
        assert np.linalg.norm(held - live) / np.linalg.norm(exact) < 1e-13
        assert np.linalg.norm(held - exact) / np.linalg.norm(exact) < 1e-12
