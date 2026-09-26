"""Round column tables by hand, and one round executable reused across changed inputs.

CPU (pytest, 2x2 host mesh) and P4 (``python test_shared_pole_callable_reuse.py OUT.json``):
* ``round_tables``: each state keeps its carrier with the inert tail inside, in state order; the zero
  column fills a slot to the round extent (the carrier of the largest selection, at most every full panel); ordered originals and mirrors pack as two halves of one
  extent (paired per slot); the infinity block is doubled on an odd-moment ordered round and absent
  without odd moments; ``own`` is the spectrum length at the slot's own extent;
* ``own_extent_receipts`` drops exactly the round padding's zeros; ``zero_row_safe_eigh`` returns
  the zero-padded spectrum and a valid eigenbasis;
* ``round_program``: a round with new supports, panels and counts at the same round extent runs the
  same executable, every slot solving at the round side; the retained-column kernel is reused;
* ``ladder_extent``: eighth-octave carriers, capped;
* ``canonical_factors`` pads round blocks to the widest and places parents in canonical order;
* ``parent_rounds`` depth: two parents per rank in one round equal the same parents in two
  one-per-rank rounds, synthetic slots exact zeros.
"""
from pathlib import Path
import json
import os
import numpy as np


def _extent(width):
    return 2 * ((width + 1) // 2)


def check_tables():
    from gw.shared_pole_local import own_extent_receipts, round_tables
    counts = np.array([[3, 7], [7, 3], [3, 3], [0, 0]])
    t = round_tables(counts, (8, 8), (2 + 1j, 3 + 1j), [3, 4, 3, 0], 4, column_extent=_extent,
                     ordered=False, odd_moments=False)
    fill = 16
    assert t["order"].tolist() == [[0, 1, 2, 3, *range(8, 16)], [*range(8), 8, 9, 10, 11],
                                   [0, 1, 2, 3, 8, 9, 10, 11] + [fill] * 4, [fill] * 12]
    assert t["points"][0].tolist() == [2 + 1j] * 4 + [3 + 1j] * 8
    assert t["points"][2, 8:].tolist() == [0] * 4
    finite = t["active"][:, :12]
    assert finite[0].tolist() == [True] * 3 + [False] + [True] * 7 + [False]
    assert finite[2].tolist() == [True] * 3 + [False] + [True] * 3 + [False] * 5 and not finite[3].any()
    assert t["active"][:, 12:].sum(axis=1).tolist() == [3, 4, 3, 0]
    from runtime.padding import ladder_extent
    wide = round_tables(counts, (32, 32), (2 + 1j, 3 + 1j), [3, 4, 3, 0], 4,
                        column_extent=lambda w: _extent(ladder_extent(w)), ordered=False, odd_moments=False)
    assert wide["order"].shape == (4, 12)  # selection 12 on the ladder, below the 64 capacity
    assert [ladder_extent(n) for n in (15, 17, 33, 1025, 2047)] == [15, 18, 36, 1152, 2048]
    assert ladder_extent(1025, 1100) == 1100
    # Fe 4^3 bispinor SC regression: an extent function that saturates below
    # the round's summed carriers (the old rank-capped ladder) must not shrink
    # the round extent below the selection.
    capped = round_tables(np.array([[28, 28], [20, 20]]), (28, 28), (1j, 2j), [2, 2], 2,
                          column_extent=lambda w: _extent(ladder_extent(w, 40)),
                          ordered=False, odd_moments=False)
    assert capped["order"].shape == (2, 56) and capped["order"][0].tolist() == list(range(56))
    try:
        round_tables(np.array([[9, 8]]), (8, 8), (1j, 2j), [0], 0, column_extent=_extent,
                     ordered=False, odd_moments=False)
        raise AssertionError("a carrier wider than its panel must refuse")
    except ValueError as error:
        assert "GATE shared_pole_round_tables" in str(error)
    assert t["own"].tolist() == [16, 16, 12, 0]
    assert t["extents"].tolist() == [[12, 4], [12, 4], [8, 4], [0, 0]]
    paired = np.column_stack((counts, counts))
    for odd, blocks in ((True, 2), (False, 0)):
        o = round_tables(paired, (8,) * 4, (1j, 2j, -1j, -2j), [3, 4, 3, 0], 4, column_extent=_extent,
                         ordered=True, odd_moments=odd)
        originals, mirrors = o["order"][:, :12], o["order"][:, 12:]
        assert np.array_equal(mirrors, np.where(originals == 32, 32, originals + 16))
        assert np.array_equal(o["points"][:, 12:], -o["points"][:, :12])
        assert np.array_equal(o["active"][:, 12:24], o["active"][:, :12])
        assert o["active"].shape[-1] == 24 + 4 * blocks
        if odd:
            assert np.array_equal(o["active"][:, 24:28], o["active"][:, 28:])
        assert o["own"].tolist() == ([16, 16, 12, 0] if odd else [12, 12, 8, 0])
    row, = own_extent_receipts({"gram_spectrum_relative": np.array([[-1e-9, 0, 0, 0, .5, 1]]),
                                "metric_inverse_root_residual_fro": np.array([2.0])}, [4])
    assert row["gram_spectrum_relative"].tolist() == [[-1e-9, 0, .5, 1]]
    assert row["gram_min_relative"].tolist() == [-1e-9]
    assert row["metric_inverse_root_residual_relative"].tolist() == [1.0]
    import jax.numpy as jnp
    from gw.shared_pole_local import zero_row_safe_eigh
    rng = np.random.default_rng(5)
    x = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    a = np.zeros((10, 10), complex)
    live = [0, 2, 3, 5, 6, 9]
    a[np.ix_(live, live)] = x @ x.conj().T - 20 * np.eye(6)  # indefinite live block, zero rows between
    values, vectors = map(np.asarray, zero_row_safe_eigh(jnp.linalg.eigh)(jnp.asarray(a)))
    tol = 1e3 * np.finfo(values.dtype).eps * np.abs(a).max()  # x64 or not, as the runtime chose
    assert np.allclose(values, np.linalg.eigvalsh(a), atol=tol)
    assert np.allclose(vectors @ np.diag(values) @ vectors.conj().T, a, atol=tol)


def check_reuse(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la as D
    from gw.shared_pole_local import _batch_put, canonical_factors, reduce_round, round_program, round_tables
    from distrib_la.polar import _retained_column_kernel
    ranks, n = 4, 8
    rng = np.random.default_rng(88)
    c = rng.normal(size=(n, 12)) * .3
    poles = np.linspace(.2, 3., 12)
    m1, m3 = c @ c.T / 2, (c * poles) @ c.T / 2
    qi = np.linalg.eigh(m1)[1][:, -2:]
    infinity = tuple(_batch_put(mesh, np.broadcast_to(a, (ranks,) + a.shape).astype(complex))
                     for a in (qi, m1 @ qi, m3 @ qi))
    base_native = D.plan("eigh", mesh, n=n, backend="off", batched_route="batch_reshard").native_fn
    solved_sides = set()

    def native(a):
        solved_sides.add(a.shape[-1])
        return base_native(a)

    def round_(nodes, counts, seed):
        local = np.random.default_rng(seed)
        states = []
        for a, s in enumerate(nodes):
            q = np.linalg.qr(local.normal(size=(ranks, n, n)))[0].astype(complex)
            q *= np.arange(n)[None, None, :] < counts[:, a, None, None]
            w, dw = (c / (s - poles)) @ c.T, (-c / (s - poles) ** 2) @ c.T
            states.append((s, *(_batch_put(mesh, m @ q) for m in (np.eye(n), w, dw))))
        tables = round_tables(counts, (n,) * len(nodes), nodes, [2] * ranks, 2, column_extent=_extent,
                              ordered=False, odd_moments=False)
        return reduce_round(states, infinity, tables, real=ranks, mesh_xy=mesh, native_eigh=native,
                            ordered=False, odd_moments=False, keep_budget=None)

    program = round_program(mesh, native, False, False, None, False, None)
    first = round_((-.3 + .2j, -.9 + .1j), np.array([[3, 7], [7, 3], [3, 3], [7, 7]]), 1)
    jax.block_until_ready(first)
    compiled = program._cache_size()
    assert compiled > 0
    assert solved_sides == {18}, solved_sides
    second = round_((-.4 + .3j, -1.1 + .2j), np.array([[7, 3], [3, 7], [3, 3], [8, 7]]), 2)
    jax.block_until_ready(second)
    assert program._cache_size() == compiled
    assert float(jnp.max(jnp.abs(second[0][1] - first[0][1]))) > 0
    blocks = [np.arange(2 * 8 * 2).reshape(2, 8, 2) + 1j, np.arange(8 * 4).reshape(1, 8, 4) + 100.]
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    placed = canonical_factors(mesh, (2, 0, 1))(*(jax.make_array_from_callback(b.shape, face, lambda i, b=b: b[i])
                                                 for b in blocks))
    want = np.concatenate([np.pad(blocks[0], ((0, 0), (0, 0), (0, 2))), blocks[1]])[[2, 0, 1]][:, :, None, :]
    from jax.experimental import multihost_utils
    host = np.asarray(multihost_utils.process_allgather(placed, tiled=True))
    assert placed.sharding.spec == P(None, 'x', None, 'y') and np.array_equal(host, want)
    assert canonical_factors(mesh, (2, 0, 1)) is canonical_factors(mesh, (2, 0, 1))
    host = np.stack([np.full((8, 8), k + 1, complex) for k in range(3)])
    panels = jax.make_array_from_callback(host.shape, NamedSharding(mesh, P(None, 'x', 'y')), lambda i: host[i])
    select = _retained_column_kernel(mesh, True, 4)
    counts = jax.device_put(np.array([2, 3, 4], np.int64), NamedSharding(mesh, P()))
    first_selection = select(panels, counts)
    jax.block_until_ready(first_selection)
    selection_specializations = select._cache_size()
    changed = select(panels, jax.device_put(np.array([4, 2, 3], np.int64), NamedSharding(mesh, P())))
    jax.block_until_ready(changed)
    assert select._cache_size() == selection_specializations
    for i, count in enumerate((4, 2, 3)):
        assert float(jnp.max(jnp.abs(changed[i, :, :count] - (i + 1)))) == 0
        if count < 4:
            assert float(jnp.max(jnp.abs(changed[i, :, count:]))) == 0
    return dict(status='PASS', scope='round tables by hand; round executable and retained-column reuse; canonical factor placement',
                round_specializations=compiled, selection_specializations=selection_specializations)


def check_depth(mesh):
    """Two parents per rank in one round equal the same parents in two one-per-rank rounds.

    Slots 0..7 of one depth-2 round (rank r owns slots 2r, 2r+1; slots 6, 7
    synthetic) against rounds 0..3 and 4..7 (slots 6, 7 synthetic) at the same
    round extent: every real slot's model, vectors and diagnostics agree, and
    every synthetic slot is the skipped slot (no active column). Returns the largest
    phase-free real-slot difference and whether every leaf kept its bits.
    """
    import jax
    import distrib_la as D
    from jax.experimental import multihost_utils
    from gw.shared_pole_local import _batch_put, parent_rounds, reduce_round, round_tables
    assert [row[:2] for row in parent_rounds(13, 4, 4)] == [(list(range(13)) + [12] * 3, 13)]
    assert [row[1] for row in parent_rounds(13, 4)] == [4, 4, 4, 1]
    ranks, n = 4, 8
    rng = np.random.default_rng(89)
    c = rng.normal(size=(n, 12)) * .3
    poles = np.linspace(.2, 3., 12)
    m1, m3 = c @ c.T / 2, (c * poles) @ c.T / 2
    qi = np.linalg.eigh(m1)[1][:, -2:]
    native = D.plan("eigh", mesh, n=n, backend="off", batched_route="batch_reshard").native_fn
    nodes = (-.3 + .2j, -.9 + .1j)
    # Both halves reach the full round extent (a [7, 7] slot), so the rounds share one side.
    counts = np.array([[3, 7], [7, 3], [3, 3], [7, 7], [7, 7], [3, 5], [0, 0], [0, 0]])
    q = [np.linalg.qr(rng.normal(size=(2 * ranks, n, n)))[0].astype(complex)
         * (np.arange(n)[None, None, :] < counts[:, a, None, None]) for a in range(2)]

    def run(rows, real):
        states = []
        for a, s in enumerate(nodes):
            w, dw = (c / (s - poles)) @ c.T, (-c / (s - poles) ** 2) @ c.T
            states.append((s, *(_batch_put(mesh, np.ascontiguousarray((m @ q[a])[rows]))
                                for m in (np.eye(n), w, dw))))
        infinity = tuple(_batch_put(mesh, np.broadcast_to(v, (len(rows),) + v.shape).astype(complex))
                         for v in (qi, m1 @ qi, m3 @ qi))
        tables = round_tables(counts[rows], (n, n), nodes, [2] * len(rows), 2, column_extent=_extent,
                              ordered=False, odd_moments=False)
        out = reduce_round(states, infinity, tables, real=real, mesh_xy=mesh, native_eigh=native,
                           ordered=False, odd_moments=False, keep_budget=None)
        return jax.tree.map(lambda a: np.asarray(multihost_utils.process_allgather(a, tiled=True)), out)

    deep = run(np.arange(2 * ranks), 6)
    halves = (run(np.arange(ranks), ranks), run(np.arange(ranks, 2 * ranks), 2))
    worst, bitwise = 0.0, True
    for slot in range(2 * ranks):
        here = jax.tree.map(lambda a: a[slot], deep)
        if slot >= 6:
            # A synthetic slot is the skipped slot: no active column, zero factor and
            # diagnostics, inactive poles 1 and the identity permutation after the sort.
            (b, poles, active), _, _, (reduction, zero, retained, permutation) = here
            assert not np.any(b) and not np.any(active) and np.all(poles == 1), slot
            assert all(not np.any(leaf) for leaf in jax.tree.leaves((reduction, zero, retained))), slot
            assert np.array_equal(permutation, np.arange(permutation.size)), slot
            continue
        there = jax.tree.map(lambda a: a[slot % ranks], halves[slot // ranks])
        bitwise = bitwise and all(np.array_equal(a, b) for a, b in
                                  zip(jax.tree.leaves(here), jax.tree.leaves(there)))
        (b1, l1, a1), _, _, diagnostics1 = here
        (b2, l2, a2), _, _, diagnostics2 = there
        assert np.array_equal(a1, a2), slot
        # Eigenvector phases are the solver's choice: compare the phase-free model
        # b b^H and b Lambda b^H, the poles and the diagnostics (not the permutation).
        pairs = [(b1 @ b1.conj().T, b2 @ b2.conj().T), ((b1 * l1) @ b1.conj().T, (b2 * l2) @ b2.conj().T),
                 (l1, l2), *zip(jax.tree.leaves(diagnostics1[:3]), jax.tree.leaves(diagnostics2[:3]))]
        for x, y in pairs:
            assert x.shape == y.shape and x.dtype == y.dtype
            if x.dtype == bool:
                assert np.array_equal(x, y), slot
            else:
                scale = max(1.0, float(np.max(np.abs(y), initial=0.0)))
                worst = max(worst, float(np.max(np.abs(x - y), initial=0.0)) / scale)
    assert worst <= 1e-10, worst
    return dict(status='PASS', scope='depth-2 round vs two depth-1 rounds, unordered 8x8 fixture',
                max_relative_difference=worst, bitwise=bitwise)


def test_round_tables_and_executable_reuse():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    check_tables()
    mesh = Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    check_reuse(mesh)
    check_depth(mesh)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')

    def main():
        import sys
        import jax
        from common.collectives import resolve_mesh, barrier
        assert jax.process_count() == 4
        check_tables()
        row = check_reuse(resolve_mesh())
        row['depth'] = check_depth(resolve_mesh())
        row['job_step'] = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(row, indent=2) + '\n')
        barrier('callable-reuse-gate')
    run_main_and_finalize(main)
