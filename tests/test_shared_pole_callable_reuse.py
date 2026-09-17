"""Round column tables by hand, and one round executable reused across changed inputs.

CPU (pytest, 2x2 host mesh) and P4 (``python test_shared_pole_callable_reuse.py OUT.json``):
* ``round_tables``: each state keeps its carrier with the inert tail inside, in state order; the zero
  column fills a slot to the round extent; ordered originals and mirrors pack as two halves of one
  extent (paired per slot); the infinity block is doubled on an odd-moment ordered round and absent
  without odd moments; ``own`` is the spectrum length at the slot's own extent;
* ``own_extent_receipts`` drops exactly the round padding's zeros;
* ``round_program``: a round with new supports, panels and counts at the same extents runs the same
  executable; the public-factor and retained-column kernels are reused.
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
    assert t["own"].tolist() == [16, 16, 12, 0]
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


def check_reuse(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la as D
    from gw.shared_pole_directions import _public_factor_kernel
    from gw.shared_pole_local import _batch_put, reduce_round, round_program, round_tables
    from distrib_la.polar import _retained_column_kernel
    ranks, n = 4, 8
    rng = np.random.default_rng(88)
    c = rng.normal(size=(n, 12)) * .3
    poles = np.linspace(.2, 3., 12)
    m1, m3 = c @ c.T / 2, (c * poles) @ c.T / 2
    qi = np.linalg.eigh(m1)[1][:, -2:]
    infinity = tuple(_batch_put(mesh, np.broadcast_to(a, (ranks,) + a.shape).astype(complex))
                     for a in (qi, m1 @ qi, m3 @ qi))
    native = D.plan("eigh", mesh, n=n, backend="off", batched_route="batch_reshard").native_fn

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

    program = round_program(mesh, native, False, False, None)
    first = round_((-.3 + .2j, -.9 + .1j), np.array([[3, 7], [7, 3], [3, 3], [7, 7]]), 1)
    jax.block_until_ready(first)
    compiled = program._cache_size()
    second = round_((-.4 + .3j, -1.1 + .2j), np.array([[7, 3], [3, 7], [3, 3], [8, 7]]), 2)
    jax.block_until_ready(second)
    assert program._cache_size() == compiled
    assert float(jnp.max(jnp.abs(second[0][1] - first[0][1]))) > 0
    assert _public_factor_kernel(mesh) is _public_factor_kernel(mesh)
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
    return dict(status='PASS', scope='round tables by hand; round executable, public-factor and retained-column reuse',
                round_specializations=compiled, selection_specializations=selection_specializations)


def test_round_tables_and_executable_reuse():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    check_tables()
    check_reuse(Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y')))


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
        row['job_step'] = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(row, indent=2) + '\n')
        barrier('callable-reuse-gate')
    run_main_and_finalize(main)
