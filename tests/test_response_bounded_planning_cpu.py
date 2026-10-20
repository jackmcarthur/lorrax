"""Bounded full-band scalar planning; no GPU or physical model approximation."""
import numpy as np
import pytest


def test_actual_material_bounded_metadata_schedule_has_no_idle_speculation():
    import file_io  # noqa: F401
    import minimax
    from gw.response_bank import _rule_round
    from common.units import RYD_TO_EV
    # Exact source recipe from the stopped full978mu/36q/160band proof.
    line = [2.603465, 3.1841689626499474, 3.747068452569397, 4.325418971660813,
            4.929710806852976, 5.5817974982184895, 6.310801821061985, 7.117718278047358,
            7.998410349756846, 8.96586711568583, 10.003863332525258, 11.066140273119055,
            12.132880677314066, 13.21084935284863, 14.368833926780045,
            15.747201394547194, 18.53]
    held_line = [6.714260049554671, 12.671865015081348]
    imag = [.9841477243747115, 3.9681687955523497, 16., 1.9761741547944567, 7.968105215723346]
    sites = np.array([complex(x, 2.6) for x in line]+[1j*x for x in imag[:3]]+
                     [complex(x, 2.6) for x in held_line]+[1j*x for x in imag[3:]]+
                     [complex(x, .1) for x in sorted(line+held_line)])/RYD_TO_EV
    extent = 6.105066515798275+4./RYD_TO_EV
    tree = minimax.ResponseGroupTree(-extent, extent, sites+1j*.05/RYD_TO_EV,
        panels=True, max_pencil_bytes=4*1024**3, max_panel_count=32)
    assert len(sites) == 43
    plan = _rule_round([tree], [{}], 4)
    assert list(map(len, plan)) == [1, 0, 0, 0]  # formerly [1,45,1,1]
    assert plan[0] == [(0, tree.root)]
    assert tree.pencil_shapes(tree.root) == []  # METADATA ONLY, no QR


def test_bounded_needed_only_schedule_keeps_serial_numeric_rule_bytes():
    import file_io  # noqa: F401
    import minimax
    from gw.response_bank import _rule_round
    tree = minimax.ResponseGroupTree(-1., 1., [.2+.08j, .4+.1j], panels=True,
                                    max_pencil_bytes=1, max_panel_count=32)
    serial = tree.rules()
    known = [{}]
    while (plan := _rule_round([tree], known, 4)) is not None:
        needed = set(tree.pending(known[0]))
        assert set(n for tasks in plan for _, n in tasks) == needed
        for tasks in plan:
            for _, node in tasks:
                known[0][node] = tree.attempt(node)
    for a, b in zip(serial, tree.rules(known[0])):
        for key in ('t','value','derivative','count','sampled_error','coefficient_mass','members','reference_ry'):
            assert np.asarray(a[key]).tobytes() == np.asarray(b[key]).tobytes(), key


def test_shared_panel_checks_retain_literal_old_probe_errors_and_mass():
    import file_io  # noqa: F401
    from minimax import complex_response as cr
    poles = np.array([.2+.08j, -.2+.08j, .4+.08j, .1+.13j])
    old = cr._imaginary_panels(-1., 1., poles, 5e-9, 0., 384, 32)
    new = cr._imaginary_panels(-1., 1., poles, 5e-9, 0., 384, 32, shared_checks=True)
    assert old is not None and new is not None
    for a, b in zip(old[0], new[0]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(old[1], new[1]):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=2e-14, rtol=0)
        np.testing.assert_array_equal(a[2], b[2])


def test_bounded_group_panels_never_enter_hankel(monkeypatch):
    import file_io  # noqa: F401: public service bootstrap
    import minimax
    from minimax import complex_response as cr
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized Hankel must not allocate/prepare")
    monkeypatch.setattr(cr, "_shared_times", forbidden)
    tree = minimax.ResponseGroupTree(-1., 1., [.2+.08j, .4+.1j], panels=True,
                                    max_pencil_bytes=1, max_panel_count=32)
    assert tree.pencil_shapes(tree.root) == []
    rules = tree.rules()
    assert all(r["construction"] == "bounded_imaginary_panels" for r in rules)
    assert all(r["members"] == [0, 1] for r in rules)
    # Independent value and d/d(z²), forward AND reverse summation.
    d = np.linspace(-1., 1., 731)
    for j, z in enumerate(tree.z):
        for side, p in enumerate((z, -z)):
            value, derivative = 0., 0.
            for rule in rules:
                count = rule["count"]
                t = rule["t"][:count]
                if side:
                    t = t.conj()
                basis = np.exp(-(d[:, None]-rule["reference_ry"])*t)
                value += basis@rule["value"][j, side, :count]
                derivative += basis@rule["derivative"][j, side, :count]
            np.testing.assert_allclose(value, 1/(d-p), atol=1e-7)
            np.testing.assert_allclose(derivative, (1 if side == 0 else -1)/(2*z*(d-p)**2), atol=1e-6)


def test_bounded_single_failure_cannot_escape_to_separate_pencil(monkeypatch):
    import file_io  # noqa: F401
    import minimax
    from minimax import complex_response as cr
    monkeypatch.setattr(cr, "_imaginary_panels", lambda *a, **k: None)
    monkeypatch.setattr(cr, "_shared_times", lambda *a, **k: pytest.fail("Hankel reached"))
    with pytest.raises(ValueError, match="planning capacity"):
        minimax.response_group_rules(-1., 1., [.2+.01j], panels=True,
                                      max_pencil_bytes=1, max_panel_count=1)


def test_full_density_rules_collective_owner_and_exact_containment_reuse(monkeypatch):
    import gw.response_bank as rb
    original = rb._gather_group_rules
    calls = []
    def gather(requests, factory):
        calls.append(requests)
        return original(requests, factory)
    monkeypatch.setattr(rb, "_gather_group_rules", gather)
    # Force bounded panels: still exercise public owner and actual scalar tests.
    import minimax
    real_tree = minimax.ResponseGroupTree
    def factory(*a, **kw):
        kw["max_pencil_bytes"] = 1
        return real_tree(*a, **kw)
    monkeypatch.setattr(minimax, "ResponseGroupTree", factory)
    held = {}
    energy, sites = np.array([[-.5, .5]]), np.array([.2+.08j])
    first = rb.full_density_sample_rules(energy, sites, gamma_ry=.02, held=held)
    same = rb.full_density_sample_rules(energy*.99, sites, gamma_ry=.02, held=held)
    assert same is first and len(calls) == 1
    changed = rb.full_density_sample_rules(energy, sites+.01j, gamma_ry=.02, held=held)
    assert changed is not first and len(calls) == 2
