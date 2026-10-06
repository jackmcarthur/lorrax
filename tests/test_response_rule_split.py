"""Response-rule groups and their closed-form fallback (CPU only, under a minute).

1. A sample set whose pencil is wider than both halves' (the narrowest
   imaginary height in one half, line sites to 42 eV in the other) is built as
   its halves without a whole-group try; a set reaching 9.44 eV is tried whole.
2. That 9.44 eV set (main's Si 4^3 reach, 22 samples) is one shared group on
   the real pencil.
3. A line sample near the interval top, which no pencil geometry fits within
   the coefficient-mass gate, builds on the closed-form imaginary-axis set.
"""
import numpy as np

RY = 13.605693122994
LO, HI = 0.0509122803681414, 3.856008262454254  # Si 4^3 response interval, Ry
IMAG_EV = (1.0, 1.6489319412704257, 2.718976546941855, 7.392833462819855,
           12.19027923333651, 20.10094080085413)


def _samples(top_ev, count):
    """Imaginary ladder by Im z, then line sites at height 2.6 eV by Re z (Ry)."""
    line = np.geomspace(2.6, top_ev, count)
    return np.array([1j*u for u in IMAG_EV] + [x + 2.6j for x in line])/RY


def test_wide_pencil_builds_halves_untried(monkeypatch):
    import file_io  # noqa: F401  (service path bootstrap)
    from minimax import complex_response as cr

    tried = []

    def shared(lo, hi, poles, tol, previous=None, decay_rate=0., patience=None):
        tried.append(len(poles))
        return (np.array([1.+1.j]),
                [(np.zeros((1, 2), complex), np.zeros(2), np.zeros(2))]*len(poles))

    monkeypatch.setattr(cr, "_shared_times", shared)
    # 6 imaginary (one pole each) + 32 line sites to 41.78 eV (two poles each).
    wide = cr.response_group_rules(LO, HI, _samples(41.78, 32))
    assert tried == [6 + 2*13, 2*19]
    assert [len(rule["members"]) for rule in wide] == [19, 19]
    tried.clear()
    cr.response_group_rules(LO, HI, _samples(9.44, 16))
    assert tried == [6 + 2*16]


def test_narrow_reach_is_one_shared_group():
    import file_io  # noqa: F401
    import minimax

    rules = minimax.response_group_rules(LO, HI, _samples(9.44, 16))
    assert len(rules) == 1 and rules[0]["members"] == list(range(22))
    assert float(rules[0]["sampled_error"].max()) <= 1e-8


def test_edge_sample_builds_on_the_imaginary_axis():
    """The Si 4^3 +-5 eV top line site (44.8 + 2.6i eV, 85 % up the interval)
    was refused: the pencil's decaying nodes need coefficient mass above the
    gate at every geometry. The closed-form imaginary-axis set builds it, and
    any line pole up to the interval top, with O(1) mass."""
    import file_io  # noqa: F401
    import minimax
    from minimax import complex_response as cr

    eta = 2.6/RY
    for top_ev in (44.8, 52.4):
        poles = cr._poles(np.array([top_ev/RY + 1j*eta]))
        times, fits = cr._imaginary_times(LO, HI, poles, 5e-9, 0., cr.RESPONSE_NODE_CAPACITY)
        assert np.all(times.real == 0) and len(times) == 140
        assert max(float(fit[2][0]) for fit in fits) < 10
    rules = minimax.response_group_rules(LO, HI, np.array([44.8/RY + 1j*eta]))
    assert len(rules) == 1 and rules[0]["count"] == 140
    assert float(rules[0]["sampled_error"].max()) <= 1e-8
