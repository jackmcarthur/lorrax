"""Independent P4 complex operator transport with changing q metadata.

Run as a compute-node SCRIPT. Includes both antiunitary rules, different
endpoints, exact padding, bounded overlapping q tiles, and nested tracing.
"""
import runtime
import numpy as np


def tables(n, logical, side, local):
    perm = np.tile(np.arange(n, dtype=np.int32), (4, 1))
    ranges = [(a, min(logical, a+n//side)) for a in range(0, n, n//side)] if local else [(0, logical)]
    for lo, hi in ranges:
        if hi > lo:
            perm[1, lo:hi] = np.roll(np.arange(lo, hi), 3)
    perm[3] = perm[1]
    wraps = np.zeros((4, n, 3), np.int64)
    wraps[1, :logical] = np.random.default_rng(n).integers(-2, 3, (logical, 3))
    wraps[3] = -wraps[1]
    local_perm = ((perm.reshape(4, side, n//side)
                   - np.arange(side)[None, :, None]*(n//side)).reshape(4, n)
                  if local else None)
    return perm, wraps, local_perm


def literal(v, pair, idx, ops, q, left, right, logical, rule):
    out = []
    for parent, op in zip(idx, ops):
        anti = op >= 2
        a, la, _ = left
        b, lb, _ = right
        pm = np.exp(2j*np.pi*(la[op] @ q[parent]))
        pn = np.exp(2j*np.pi*(lb[op] @ q[parent]))
        source = pair[parent].T if anti and rule == 'pair_transpose' else v[parent]
        x = source[a[op, :, None], b[op, None, :]]
        if anti and rule == 'pair_transpose':
            x = pm.conj()[:, None]*x*pn[None, :]
        else:
            x = pm[:, None]*x*pn.conj()[None, :]
            if anti:
                x = x.conj()
        x[logical[0]:] = 0
        x[:, logical[1]:] = 0
        out.append(x)
    return np.asarray(out)


def main():
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from symmetry_maps import maps

    assert mesh.size == 4
    side = int(mesh.shape['x'])
    source = np.random.default_rng(815)
    v = source.normal(size=(5, 16, 24)) + 1j*source.normal(size=(5, 16, 24))
    pair = source.normal(size=(5, 24, 16)) + 1j*source.normal(size=(5, 24, 16))
    vdev = device_put_process_local(v, NamedSharding(mesh, P(None, 'x', 'y')))
    pairdev = device_put_process_local(pair, NamedSharding(mesh, P(None, 'y', 'x')))
    saved = maps._get_unfold_isdf_operator_jit

    def bounded(**kwargs):
        # Two q rows per tile; seven rows require the overlapping final tile.
        return saved(**kwargs, q_tile_bytes=2*16*(16//side)*(24//side))

    maps._get_unfold_isdf_operator_jit = bounded
    errors = []
    try:
        for local in (False, True):
            left, right = tables(16, 13, side, local), tables(24, 19, side, local)
            common = dict(sym_perm=left[0], L_table=left[1],
                right_sym_perm=right[0], right_L_table=right[1],
                left_logical_extent=13, right_logical_extent=19,
                mesh_xy=mesh, n_sym_spatial=2,
                axis_local_sym_perm=left[2], right_axis_local_sym_perm=right[2])
            for rule in ('conj', 'pair_transpose'):
                for query in range(3):
                    idx = np.roll(np.array([4, 0, 1, 3, 2, 1, 4], np.int32), query)
                    ops = np.roll(np.array([1, 3, 0, 2, 1, 3, 0], np.int32), -query)
                    q = source.uniform(-.5, .5, (5, 3))
                    kwargs = dict(common, irr_idx=idx, sym_idx=ops, q_irr_frac=q,
                                  trs_rule=rule)
                    expected = literal(v, pair, idx, ops, q, left, right, (13, 19), rule)
                    def evaluate(a, b):
                        return maps.unfold_isdf_operator(a, **kwargs,
                            **({'trs_pair_q_ibz': b} if rule == 'pair_transpose' else {}))
                    # A cached transport first created under an outer trace
                    # must also serve later direct calls with other q values.
                    result = (jax.jit(evaluate)(vdev, pairdev) if query == 0
                              else evaluate(vdev, pairdev))
                    actual = np.asarray(gather_to_host(result))
                    relative = float(np.max(np.abs(actual-expected))/np.max(np.abs(expected)))
                    assert relative < 2e-13, (local, rule, query, relative)
                    assert np.all(actual[:, 13:] == 0) and np.all(actual[:, :, 19:] == 0)
                    errors.append(relative)
    finally:
        maps._get_unfold_isdf_operator_jit = saved
    print('PASS changing-q operator transport:', len(errors), 'cases; max relative', max(errors), flush=True)


if __name__ == '__main__':
    main()
