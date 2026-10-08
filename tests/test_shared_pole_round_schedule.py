"""Shared-pole rounds: one fixed-width schedule, and a pencil extent that grows only in map 0 (CPU).

Rows 2 and 3 of the 2026-10-05 compile audit: a ragged face tail recompiled every
round program, and a selection-sized extent recompiled them on every growth in
every map. One schedule (``parent_rounds``) serves the local, staged and scalar
routes; ``round_tables`` grows the extent in map 0 and holds it from map 1.
"""
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp


def test_ragged_tail_traces_once():
    from gw.shared_pole_local import parent_rounds
    rounds = parent_rounds(7, 3)
    assert [(r[0], r[1]) for r in rounds] == [([0, 1, 2], 3), ([3, 4, 5], 3), ([6, 6, 6], 1)]
    assert all(r[2].tolist() == [0, 1, 2] for r in rounds)
    assert [(r[0], r[1]) for r in parent_rounds([4, 5, 6, 7], 3)] == [([4, 5, 6], 3), ([7, 7, 7], 1)]
    # A round program keyed on the batch's leading axis traces once over the schedule.
    traces = []

    @jax.jit
    def round_program(batch):
        traces.append(batch.shape)
        return jnp.linalg.eigh(batch)[0].sum()

    for ids, real, slots in parent_rounds(7, 3):
        round_program(jnp.stack([jnp.eye(4) * (q + 1) for q in ids]))
    assert len(traces) == 1


def test_pencil_extent_grows_in_map_zero_then_holds():
    """The extent grows with the selections seen in map 0 (one compile per growth, never above
    the panels' capacity) and a later map with the same history compiles nothing."""
    from runtime.padding import ladder_extent
    from gw.shared_pole_local import parent_rounds, round_tables
    extent = lambda w: ladder_extent(int(w))
    rng = np.random.default_rng(0)
    nq, states, width = 61, 4, 3
    per_parent = (rng.uniform(0.4, 1.0, nq) * 2000).astype(int)
    per_parent[[1, 2, 3, 13, 19]] = [1300, 1600, 1900, 2050, 2150]   # later parents select more
    widths = [2304] * states
    history, traces = {}, []

    @jax.jit
    def round_program(active):
        traces.append(active.shape)
        return active.sum()

    def sc_map():
        sides = []
        for ids, real, _ in parent_rounds(nq, width):
            counts = np.repeat(per_parent[ids][:, None], states, axis=1)
            tables = round_tables(counts, widths, [0j] * states, [0] * width, 0, column_extent=extent,
                                  ordered=True, odd_moments=False, key=("sector", "CC", 8), history=history)
            sides.append(tables["active"].shape[-1])
            # Each slot's live columns are its own selection; the rest is inert padding.
            assert tables["active"].sum(axis=1).tolist() == counts.sum(axis=1).tolist()
            assert tables["own"].tolist() == [sum(extent(c) for c in row[:states // 2]) for row in counts]
            round_program(jnp.asarray(tables["active"]))
        return sides

    sides = sc_map()
    assert sides == sorted(sides) and max(sides) <= 2 * (states // 2) * 2304
    growths = len(set(sides))
    assert 1 < growths == len(traces) <= 8
    assert sc_map() == [sides[-1]] * len(sides) and len(traces) == growths
    # Without a history every round takes its own selection's extent.
    bare = round_tables(np.full((width, states), 100), widths, [0j] * states, [0] * width, 0,
                        column_extent=extent, ordered=True, odd_moments=False)
    assert bare["active"].shape[-1] == 2 * extent(2 * extent(100))
