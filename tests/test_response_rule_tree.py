"""Response-rule tree: attempts evaluated anywhere give the serial rules (CPU, ~1 min).

1. ``ResponseGroupTree.rules(attempts)`` with attempts evaluated out of walk
   order, plus speculative attempts the walk never reaches, returns the same
   bytes as ``response_group_rules`` (a real Si split: 6 imaginary + 32 line
   samples to 41.78 eV, built as halves).
2. The bank's round planner (``gw.response_bank._rule_round``) on 1-8
   processes always schedules every node the walk needs, schedules nothing
   else on one process, and reaches the serial rules; its multi-process
   gather (threads standing in for processes) returns the serial rules.
3. A refusal is raised only where the serial walk reaches it.
"""
import threading

import numpy as np
import pytest

RY = 13.605693122994
LO, HI = 0.0509122803681414, 3.856008262454254  # Si 4^3 response interval, Ry
IMAG_EV = (1.0, 1.6489319412704257, 2.718976546941855, 7.392833462819855,
           12.19027923333651, 20.10094080085413)
KEYS = ("t", "value", "derivative", "count", "sampled_error", "coefficient_mass",
        "members", "reference_ry")


def _samples(top_ev, count):
    line = np.geomspace(2.6, top_ev, count)
    return np.array([1j*u for u in IMAG_EV] + [x + 2.6j for x in line])/RY


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for k in KEYS:
            assert np.asarray(x[k]).tobytes() == np.asarray(y[k]).tobytes(), k


def _fake(monkeypatch, fails):
    """Cheap stand-in numerics: a multi-sample node whose members meet ``fails`` fails."""
    from minimax import complex_response as cr

    def shared(lo, hi, poles, tol, previous=None, decay_rate=0., patience=None, pencil=None):
        if patience is not None and fails(poles):
            return None
        return (np.array([poles.sum()]), [(np.ones((1, 2), complex), np.zeros(2), np.zeros(2))]*len(poles))
    monkeypatch.setattr(cr, "_shared_times", shared)
    monkeypatch.setattr(cr, "_imaginary_times", lambda *a, **k: None)


def test_out_of_order_attempts_give_serial_bytes():
    import file_io  # noqa: F401  (service path bootstrap)
    import minimax

    z = _samples(41.78, 32)
    serial = minimax.response_group_rules(LO, HI, z)
    tree = minimax.ResponseGroupTree(LO, HI, z)
    attempts = {}
    while pending := tree.pending(attempts):
        for node in reversed(pending):
            attempts[node] = tree.attempt(node)
            for half in tree.children(node):  # speculative, mostly unreached
                if tree.tried(half) and half not in attempts:
                    attempts[half] = tree.attempt(half)
    assert len(attempts) > len(serial)
    _same(tree.rules(attempts), serial)


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8])
def test_round_planner_reaches_serial_rules(monkeypatch, world):
    import file_io  # noqa: F401
    import minimax
    from gw.response_bank import _rule_round

    z = _samples(41.78, 32)
    _fake(monkeypatch, lambda poles: poles.real.max() > 20/RY)  # line sites above 20 eV fail
    serial = minimax.response_group_rules(LO, HI, z)
    tree = minimax.ResponseGroupTree(LO, HI, z)
    known = [{}]
    while (plan := _rule_round([tree], known, world)) is not None:
        need = set(tree.pending(known[0]))
        tasks = [node for tasks in plan for _, node in tasks]
        assert need <= set(tasks) and len(tasks) == len(set(tasks))
        if world == 1:
            assert set(tasks) == need
        for node in tasks:
            known[0][node] = tree.attempt(node)
    _same(tree.rules(known[0]), serial)
    assert len(serial) > 2


def test_threaded_gather_returns_serial_rules(monkeypatch):
    import file_io  # noqa: F401
    import minimax
    from common import collectives
    from gw.response_bank import _gather_group_rules

    z = _samples(41.78, 32)
    _fake(monkeypatch, lambda poles: poles.real.max() > 20/RY)
    serial = minimax.response_group_rules(LO, HI, z)
    world, local = 3, threading.local()
    barrier, slots = threading.Barrier(world), [None]*world

    def gather(x, tiled=False):
        slots[local.rank] = np.asarray(x).copy()
        barrier.wait()
        out = np.stack(slots)
        barrier.wait()
        return out
    monkeypatch.setattr(collectives, "process_count", lambda: world)
    monkeypatch.setattr(collectives, "process_rank", lambda: local.rank)
    monkeypatch.setattr(collectives, "all_gather_processes", gather)
    out = [None]*world

    def run(rank):
        local.rank = rank
        out[rank] = _gather_group_rules([list(range(len(z)))],
                                        lambda m: minimax.ResponseGroupTree(LO, HI, z[m]))
    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for rules in out:
        _same(rules[0], serial)


def test_refusal_raised_only_where_reached(monkeypatch):
    import file_io  # noqa: F401
    import minimax
    from gw import response_bank

    z = _samples(9.44, 4)  # one shared group: the singletons are never reached
    tree = lambda m: minimax.ResponseGroupTree(LO, HI, z[m])
    t = tree(list(range(len(z))))
    real_round, real_attempt = response_bank._rule_round, minimax.ResponseGroupTree.attempt

    def attempt(self, node):
        if len(node) == 1:
            raise ValueError("response exponential fit failed: stand-in")
        return real_attempt(self, node)
    monkeypatch.setattr(minimax.ResponseGroupTree, "attempt", attempt)

    def speculate(trees, known, world):  # also attempt one unreached singleton
        plan = real_round(trees, known, world)
        if plan is not None and not known[0]:
            plan[0].append((0, (0,)))
        return plan
    monkeypatch.setattr(response_bank, "_rule_round", speculate)
    rules = response_bank._gather_group_rules([list(range(len(z)))], tree)
    assert [r["members"] for r in rules[0]] == [list(t.root)]
    # Reached: a set whose whole-group fit fails walks down to the refused singletons.
    monkeypatch.setattr(minimax.ResponseGroupTree, "_attempt", lambda self, m: None)
    with pytest.raises(ValueError, match="stand-in"):
        response_bank._gather_group_rules([list(range(len(z)))], tree)


def _perturbed_pencils(zps, size, columns, steps):
    """The reference pencils with 1e-12 relative noise: another source's rounding."""
    from minimax import complex_response as cr
    rng = np.random.default_rng(7)
    for s, shift in cr.host_pencils(zps, size, columns, steps):
        yield s*(1 + 1e-12*rng.standard_normal(s.shape)), shift*(1 + 1e-12*rng.standard_normal(shift.shape))


def test_accepted_geometry_is_decided_on_the_reference_pencil():
    """A pencil source with other rounding: an accepted node set is the reference's bytes."""
    import file_io  # noqa: F401  (service path bootstrap)
    import minimax

    z = _samples(12.0, 4)
    _same(minimax.response_group_rules(LO, HI, z, pencil=_perturbed_pencils),
          minimax.response_group_rules(LO, HI, z))



def test_resolved_acceptance_stands_on_the_source():
    """A source that declares it resolves its counts keeps its own acceptance (same counts)."""
    import functools
    import file_io  # noqa: F401  (service path bootstrap)
    import minimax

    z = _samples(12.0, 4)
    resolved = functools.partial(_perturbed_pencils)
    resolved.sigma_error = 0.
    ours, ref = minimax.response_group_rules(LO, HI, z, pencil=resolved), minimax.response_group_rules(LO, HI, z)
    assert [r["count"] for r in ours] == [r["count"] for r in ref]
    assert any(np.asarray(a["t"]).tobytes() != np.asarray(b["t"]).tobytes() for a, b in zip(ours, ref))
