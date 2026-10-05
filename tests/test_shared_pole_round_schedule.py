"""Every shared-pole round program sees one shape per run (CPU, seconds).

Rows 2 and 3 of the 2026-10-05 compile audit: a ragged face tail and a
selection-sized pencil extent each recompiled every round program. One
fixed-width schedule (``parent_rounds``) serves the local, face, scalar and
rerun routes, and ``round_tables`` sizes the pencil at the panels' capacity,
known before the first round.
"""
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp


def _schedule(execution, batch_width, nq=7):
    from gw.shared_pole_execution import sector_round_schedule
    config = SimpleNamespace(backend=SimpleNamespace(linalg="local"))
    return sector_round_schedule(None, {"n_q_irr": nq}, None, config, SimpleNamespace(size=4),
                                 execution=execution, batch_width=batch_width)


def test_ragged_tail_traces_once():
    from gw.shared_pole_local import parent_rounds
    from gw.shared_pole_sectors import face_rerun_rounds
    rounds = parent_rounds(7, 3)
    assert [(r[0], r[1]) for r in rounds] == [([0, 1, 2], 3), ([3, 4, 5], 3), ([6, 6, 6], 1)]
    assert all(r[2].tolist() == [0, 1, 2] for r in rounds)
    assert [(r[0], r[1]) for r in parent_rounds([4, 5, 6, 7], 3)] == [([4, 5, 6], 3), ([7, 7, 7], 1)]
    # The face, local and rerun routes all take fixed-width rounds.
    for execution, width in (("face", 3), ("local", 4)):
        sched = _schedule(execution, 3)
        assert {len(r[0]) for r in sched} == {width}
        assert [r[3] for r in sched] == [execution] * len(sched)
        assert [q for r in sched for q in r[0][:r[1]]] == list(range(7))
    assert [(r[0], r[1], r[3]) for r in face_rerun_rounds([4, 5, 6, 7], 4, 3)] == [
        ([4, 5, 6], 3, "face"), ([7, 7, 7], 1, "face")]
    # A round program keyed on the batch's leading axis traces once over the schedule.
    traces = []

    @jax.jit
    def round_program(batch):
        traces.append(batch.shape)
        return jnp.linalg.eigh(batch)[0].sum()

    for ids, real, slots, _ in _schedule("face", 3):
        round_program(jnp.stack([jnp.eye(4) * (q + 1) for q in ids]))
    assert len(traces) == 1


def test_recipe_panel_widths_are_known_before_round_one():
    """Panels of a selection narrower than its recipe width are padded to that width; a wider one keeps its own."""
    from types import SimpleNamespace
    from gw.shared_pole_local import recipe_panel_widths
    recipe = {"imaginary_width": 100, "line_direction_cap": 25}
    roles = [{"role": "imaginary:0"}, {"role": "imaginary:0", "conjugate": True},
             {"role": "line:1"}, {"role": "line:1", "conjugate": True},
             {"role": "imaginary:0", "mirror": True}, {"role": "line:1", "mirror": True}]
    panel = lambda w: (0j, SimpleNamespace(shape=(4, 432, w)))
    states = [panel(64), panel(64), panel(20), panel(20), panel(64), panel(112)]
    widths = recipe_panel_widths(roles, states, recipe, column_extent=lambda w: -(-w // 8) * 8, logical_n=432)
    assert widths == [104, 104, 32, 32, 104, 112]
    # No line cap: the logical extent bounds the line panels.
    widths = recipe_panel_widths(roles[2:3], states[2:3], {"imaginary_width": 100}, column_extent=int, logical_n=432)
    assert widths == [432]


def test_infinity_block_wider_than_the_recipe_keeps_its_own_carrier():
    """An M1 selection closed over a multiplet past the recipe width is never cut (Na, a metal)."""
    import pytest
    from types import SimpleNamespace
    from gw.shared_pole_local import recipe_infinity_width, pad_states
    recipe = {"infinity_width": 54}
    extent = lambda w: -(-w // 8) * 8
    narrow = (SimpleNamespace(shape=(4, 432, 40)),)
    wide = (SimpleNamespace(shape=(4, 432, 72)),)
    assert recipe_infinity_width(narrow, recipe, column_extent=extent, logical_n=432) == 56
    assert recipe_infinity_width(wide, recipe, column_extent=extent, logical_n=432) == 72
    assert recipe_infinity_width(wide, recipe, column_extent=extent, logical_n=60) == 72
    # A carrier below a panel is refused by name, never a negative pad.
    block = (jnp.ones((1, 4, 72)),)
    with pytest.raises(ValueError, match="GATE shared_pole_carrier"):
        pad_states([], [], block, 56)
    assert pad_states([], [], block, 72)[1][0].shape == (1, 4, 72)


def test_pencil_extent_fixed_before_round_one():
    """Selections growing parent by parent give one pencil side, the panels' capacity."""
    from runtime.padding import ladder_extent
    from gw.shared_pole_local import parent_rounds, round_tables
    extent = lambda w: ladder_extent(int(w))
    rng = np.random.default_rng(0)
    nq, states, width = 61, 4, 3
    per_parent = (rng.uniform(0.4, 1.0, nq) * 2000).astype(int)
    per_parent[[1, 2, 3, 13, 19]] = [1300, 1600, 1900, 2050, 2150]   # later parents select more
    widths = [2304] * states
    sides, traces = [], []

    @jax.jit
    def round_program(active):
        traces.append(active.shape)
        return active.sum()

    for ids, real, _ in parent_rounds(nq, width):
        counts = np.repeat(per_parent[ids][:, None], states, axis=1)
        tables = round_tables(counts, widths, [0j] * states, [0] * width, 0, column_extent=extent,
                              ordered=True, odd_moments=False)
        sides.append(tables["active"].shape[-1])
        # Each slot's live columns are its own selection; the rest is inert padding.
        assert tables["active"].sum(axis=1).tolist() == counts.sum(axis=1).tolist()
        assert tables["own"].tolist() == [sum(extent(c) for c in row[:states // 2]) for row in counts]
        round_program(jnp.asarray(tables["active"]))
    assert set(sides) == {2 * (states // 2) * 2304} and len(traces) == 1
