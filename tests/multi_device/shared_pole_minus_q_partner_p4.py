"""P4 line-sample states with the minus-q partner W_q(-conj z), diagonal and rectangular.

The producer selects a fitted line sample's directions from W(z) itself and forms every action the
constructor reads (``gw.shared_pole_directions.line_sample_states``/``line_sample_mirrors``, the
rectangular ``_local_cross_action_program``), and the constructor reads them back from their stored
panel (``line_panel_states``). Oracle: a planted signed model W(node) = L diag(1/(node mu - 1)) R^H
with dW/ds = L diag(-mu/(node mu - 1)^2/(2 node)) R^H, evaluated at z, conj z, -z and -conj z.
Checks: the directions are the right singular vectors of W(z) above the cutoff (projector and count
per parent), each state's output and d/dz action equal the oracle on its directions, the stored panel
returns the same states with the conj z direction being the z state's output, and the CT/TC actions
equal the rectangular oracle. RED TWIN: a mirror formed from the +z sample instead of W(-conj z) misses.
"""


def main():
    import argparse
    import json
    import os
    from pathlib import Path
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, resolve_mesh
    from gw.shared_pole_capacity import constructor_eigenplan
    from gw.shared_pole_directions import (_round_kernels, line_panel_states, line_sample_mirrors,
                                           line_sample_states)
    from gw.shared_pole_sectors import _local_cross_action_program
    from runtime import initialize_communicator_stack, finalize_process

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    initialize_communicator_stack()
    mesh = resolve_mesh()
    assert jax.process_count() == 4 and mesh.size == 4
    rng = np.random.default_rng(477)
    z = .73 + .41j
    nodes = (z, z.conjugate(), -z, -z.conjugate())
    n, k = 6, 4
    mu = np.array([.4, .9, -.5, -1.2])
    adj = lambda a: np.swapaxes(a.conj(), -1, -2)
    # Four parents, each its own model; parent 3 is a synthetic round slot.
    square = rng.normal(size=(4, n, k)) + 1j * rng.normal(size=(4, n, k))
    left = rng.normal(size=(4, 2, k)) + 1j * rng.normal(size=(4, 2, k))

    def value(node, l, r):
        weight = 1 / (node * mu - 1)
        slope = -mu / (node * mu - 1) ** 2 / (2 * node)
        return (l * weight) @ adj(r), (l * slope) @ adj(r)

    batch = NamedSharding(mesh, P(('x', 'y')))
    put = lambda a: jax.make_array_from_callback(a.shape, batch, lambda idx: a[idx])
    stack = lambda a: put(np.asarray(a)[:, None].copy())
    recipe = dict(z_ry=np.array([z]), distinct_id=np.array([0]), role=np.array([0], np.int8),
                  held=np.array([False]), fit_ids=np.array([0]), direction_cutoff=1e-3,
                  multiplet_relative_tolerance=1e-6, line_direction_cap=None)
    extent = lambda width: 2 * ((width + 1) // 2)
    eig, svd = constructor_eigenplan(mesh, n, 'local'), constructor_eigenplan(mesh, 2 * n, 'local')
    W, dW = value(z, square, square)
    line = line_sample_states(stack(W), stack(dW), recipe, sid=0, ordered=True, real=3, mesh_xy=mesh,
                              eigh_plan=eig, svd_plan=svd, column_extent=extent, logical_n=n)
    counts = line['counts'].tolist()
    expected = [int(np.sum(s > 1e-3 * s[0])) for s in np.linalg.svd(W[:3], compute_uv=False)]
    assert counts == expected + [0], (counts, expected)
    partner = value(-z.conjugate(), square, square)
    mirrors = line_sample_mirrors(line, (stack(partner[0]), stack(partner[1])), recipe, sid=0, mesh_xy=mesh)
    states = line['states'] + mirrors
    host = lambda a: np.asarray(gather_to_host(a))
    q = host(states[0][1])
    errors = []
    for p in range(3):
        v = np.linalg.svd(W[p])[2].conj().T[:, :counts[p]]
        errors.append(float(np.linalg.norm(q[p][:, :counts[p]] @ adj(q[p][:, :counts[p]]) - v @ adj(v))))
    for node, state in zip(nodes, states):
        assert state[0] == node, (state[0], node)
        exact, derivative = value(node, square, square)
        direction = host(state[1])[:3]
        errors.append(float(np.max(np.abs(host(state[2])[:3] - exact[:3] @ direction))))
        errors.append(float(np.max(np.abs(host(state[3])[:3] - 2 * node * derivative[:3] @ direction))))
    assert max(errors) < 1e-11, errors
    # The stored panel returns the same states, the conj z direction being the z output.
    kernels = _round_kernels(mesh, 'batch')
    panels = kernels.stack(states[0][1], *[a for st in states for a in st[2:]])
    originals, stored_mirrors, widths = line_panel_states(panels, line['counts'], recipe, sid=0,
                                                          ordered=True, mesh_xy=mesh)
    assert originals[1][1] is originals[0][2] and stored_mirrors[1][1] is originals[0][2]
    assert widths == tuple(counts)
    for a, b in zip(originals + stored_mirrors, states):
        assert a[0] == b[0] and all(bool(np.array_equal(host(x), host(y))) for x, y in zip(a[1:], b[1:]))
    # CT/TC actions on the diagonal directions: forward rectangle (rows of the other family,
    # columns of this one) and reverse, then the minus-q partner's pair for the mirror states.
    rect = lambda node: (value(node, left, square), value(node, square, left))
    (forward, dforward), (reverse, dreverse) = rect(z)
    (mforward, dmforward), (mreverse, dmreverse) = rect(-z.conjugate())
    direct = tuple(stack(a) for a in (forward, reverse, dforward, dreverse))
    minus = (None,) * 4 + tuple(stack(a) for a in (mforward, mreverse, dmforward, dmreverse))
    cross = []
    sample = np.int32(0)
    for index, (node, state) in enumerate(zip(nodes, states)):
        mirror, conjugate = index >= 2, index % 2 == 1
        program = _local_cross_action_program(mesh, mirror, False, conjugate)
        output, action = program(minus if mirror else direct, state[1], np.asarray(node), sample)
        exact, derivative = value(node, left, square)
        direction = host(state[1])[:3]
        cross.append(float(np.max(np.abs(host(output)[:3] - exact[:3] @ direction))))
        cross.append(float(np.max(np.abs(host(action)[:3] - 2 * node * derivative[:3] @ direction))))
    assert max(cross) < 1e-11, cross
    # RED TWIN: the mirror from the +z sample (no minus-q partner) misses.
    wrong = line_sample_mirrors(line, (stack(W), stack(dW)), recipe, sid=0, mesh_xy=mesh)
    red = float(np.linalg.norm(host(wrong[0][2])[:3] - host(mirrors[0][2])[:3]))
    assert red > 1e-3, red
    result = dict(status='PASS', job=os.environ.get('SLURM_JOB_ID'), step=os.environ.get('SLURM_STEP_ID'),
                  diagonal_max=max(errors), cross_max=max(cross), counts=counts,
                  wrong_positive_sample_red=red,
                  scope='P4 producer line-sample states and minus-q partner actions at nonzero complex '
                        'frequency; no bank producer, moments or material model')
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finalize_process()


if __name__ == '__main__':
    main()
