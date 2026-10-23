"""P4 mixed-parent/carrier oracle for the ordered pencil and paired Gram.

The exact signed rational model has positive M, so H=X^H M X and its
paired vv block are PSD. Physical entries are identified through owner
tables, including interior padding and tile-interleaved infinity blocks.
"""


def main():
    import argparse
    import json
    from pathlib import Path
    import runtime
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import device_budget_bytes
    from gw.tangential_directions import port_extent
    from gw.tangential_execution import face_matmul, face_reduce_round, face_ritz_carrier
    from gw.tangential_local import round_tables, pack_panels
    from gw.tangential_pencil import (assemble_ordered_tangential_pencil,
        interleave_tables, join_vectors, split_vectors)
    from gw.tangential_reduction import paired_members
    from gw.tangential_recipe import shared_real_pole_gates_ordered_v1 as gates
    import distrib_la

    assert mesh.size == 4
    device_budget_bytes()
    face, rep = NamedSharding(mesh, P(None, 'x', 'y')), NamedSharding(mesh, P())
    put = lambda a, sh=face: device_put_process_local(np.ascontiguousarray(a), sh)
    host = lambda a: np.asarray(gather_to_host(a))
    extent, py = port_extent(mesh), int(mesh.shape['y'])
    rng = np.random.default_rng(20261023)
    n, poles, parents = 32, 24, 5
    nodes = (.7j, 1.9j, -.7j, -1.9j)
    counts = np.tile(np.array([[2, 3], [5, 1], [0, 6], [4, 7], [7, 2]]), (1, 2))
    icount = np.array([1, 2, 3, 2, 3])
    width, iw = extent(int(counts.max())), extent(int(icount.max()))
    widths = [width] * len(nodes)
    models, states, infinity = [], [], []

    def orthogonal(columns):
        return np.linalg.qr(rng.normal(size=(n, columns)) + 1j*rng.normal(size=(n, columns)))[0]

    # W=C(zS-M)^-1 C^H, S=diag(+1,-1), M=diag(omega,omega)>0.
    # Conjugate residue partners give W(-z)=W(z)^T with an odd channel.
    directions = [[orthogonal(width), orthogonal(width)] for _ in range(parents)]
    qi = np.zeros((parents, n, iw), complex)
    moments = [np.zeros((parents, n, iw), complex) for _ in range(4)]
    for p in range(parents):
        omega = np.linspace(.6, 3.7, poles) + .013*p
        b = (rng.normal(size=(n, poles)) + 1j*rng.normal(size=(n, poles))) / np.sqrt(n)
        c = np.concatenate((b, np.conj(b)), axis=1)
        metric, mass = np.r_[np.ones(poles), -np.ones(poles)], np.tile(omega, 2)
        models.append((c, metric, mass))
        qi[p, :, :icount[p]] = orthogonal(int(icount[p]))
        for k in range(4):
            coefficient = (c*((metric*mass)**k*metric)[None]) @ np.conj(c.T)
            moments[k][p] = .5*coefficient @ qi[p]
    infinity = (qi, *moments)
    for a, z in enumerate(nodes):
        q = np.zeros((parents, n, width), complex)
        o, d = np.zeros_like(q), np.zeros_like(q)
        for p, (c, metric, mass) in enumerate(models):
            q[p, :, :counts[p, a]] = directions[p][a % 2][:, :counts[p, a]]
            denominator = z*metric-mass
            w = (c/denominator[None]) @ np.conj(c.T)
            derivative = (c*(-metric/denominator**2)[None]) @ np.conj(c.T)
            o[p], d[p] = w @ q[p], derivative @ q[p]
        states.append((z, q, o, d))

    key = ('mixed-width-oracle', n, True, True)
    def table(c, inf, history):
        return round_tables(c, widths, nodes, inf, iw, column_extent=extent,
                            ordered=True, odd_moments=True, key=key, history=history)
    small_history, large_history = {}, {}
    table(counts, icount, small_history)
    table(np.full_like(counts, width), icount, large_history)
    targets = [next(iter(h.values()))[0] for h in (small_history, large_history)]
    assert targets[1] > targets[0]

    mm = face_matmul(mesh)
    def raw(points, active, q, o, d, inf):
        pencil = assemble_ordered_tangential_pencil([(points, q, o, d)], inf,
                                                   matmul=mm, matrix_sharding=face)
        paired = paired_members(pencil, active, gates=gates, matrix_sharding=face)
        return pencil[:3], paired['h_vv'], paired['diagonal_ok'], paired['paired']
    raw = jax.jit(raw, out_shardings=((face, face, face), face, rep, rep))
    records, baseline, paired_baseline = [], None, None
    for carrier, history in zip(targets, (small_history, large_history)):
        for batch in (1, parents):
            t = table(counts[:batch], icount[:batch], dict(history))
            ft = interleave_tables(t, py)
            finite, side = t['order'].shape[1], t['active'].shape[1]
            half = finite//2
            owner = join_vectors(split_vectors(np.arange(side)[None], (half, half, iw, iw), 1), py)[0]
            inverse_owner = np.argsort(owner)
            s = [(z, *(put(v[:batch]) for v in fields)) for z, *fields in states]
            inf = tuple(put(v[:batch]) for v in infinity)
            packed = pack_panels(tuple([st[k] for st in s] for k in (1, 2, 3)),
                                 ft['order'], widths, mesh_xy=mesh, layout='face')
            result = raw(put(ft['points'], rep), put(ft['active'], rep), *packed, inf)
            (gg, hh, oo), vv, diagonal_ok, paired = jax.tree.map(host, result)
            assert np.all(diagonal_ok) and np.all(paired)
            offsets = np.r_[0, np.cumsum(widths)]
            principal, labels = [], []
            expected_errors, paired_errors, expected_min = [], [], []
            for p in range(batch):
                c, metric, mass = models[p]
                x = np.zeros((2*poles, side), complex)
                tags, tag_by_column = [], {}
                for col in np.flatnonzero(t['active'][p]):
                    if col < finite:
                        source = int(t['order'][p, col])
                        state = int(np.searchsorted(offsets, source, side='right')-1)
                        within = source-offsets[state]
                        x[:, col] = (np.conj(c.T) @ states[state][1][p, :, within]) / (t['points'][p, col]*metric-mass)
                        tag = ('state', state, int(within))
                    else:
                        block, within = divmod(col-finite, iw)
                        v = np.conj(c.T) @ qi[p, :, within]
                        x[:, col] = metric*v if block == 0 else mass*v
                        tag = ('infinity', block, int(within))
                    tags.append(tag); tag_by_column[int(col)] = tag
                active = np.flatnonzero(t['active'][p])
                face_active = inverse_owner[active]
                assert not np.array_equal(active, np.arange(len(active)))
                gx = np.conj(x.T) @ (metric[:, None]*x)
                hx = np.conj(x.T) @ (mass[:, None]*x)
                ox = c @ x
                err = 0.
                for got, wanted in ((gg[p], gx), (hh[p], hx)):
                    selected = got[np.ix_(face_active, face_active)]
                    reference = wanted[np.ix_(active, active)]
                    np.testing.assert_allclose(selected, reference, rtol=3e-11, atol=3e-11)
                    err = max(err, float(np.max(np.abs(selected-reference))))
                np.testing.assert_allclose(oo[p][:, face_active], ox[:, active], rtol=3e-11, atol=3e-11)
                eig = np.linalg.eigvalsh(hx[np.ix_(active, active)])
                assert eig.min() > -1e-10*eig.max()
                inactive = inverse_owner[np.flatnonzero(~t['active'][p])]
                assert np.all(gg[p, inactive] == 0) and np.all(hh[p, inactive] == 0)
                assert np.all(oo[p][:, inactive] == 0)
                expected_errors.append(err); expected_min.append(float(eig.min()))
                # The paired v columns are finite differences plus k0.
                # Their own interleaving joins finite and infinity pieces.
                live_v = np.r_[t['active'][p, :half], t['active'][p, finite:finite+iw]]
                v = np.zeros((2*poles, half+iw), complex)
                for col in np.flatnonzero(t['active'][p, :half]):
                    v[:, col] = (x[:, col]-x[:, half+col])/(2*t['points'][p, col])
                v[:, half:] = x[:, finite:finite+iw]
                hv = np.conj(v.T) @ (mass[:, None]*v)
                scale = np.where(live_v, 1/np.sqrt(np.maximum(np.real(np.diag(hv)), 1e-300)), 0)
                hv *= scale[:, None]*scale[None, :]
                v_owner = join_vectors(split_vectors(np.arange(half+iw)[None], (half, iw), 1), py)[0]
                v_active = np.flatnonzero(live_v)
                v_face = np.argsort(v_owner)[v_active]
                actual_v = vv[p][np.ix_(v_face, v_face)]
                expected_v = hv[np.ix_(v_active, v_active)]
                v_labels = [tag_by_column[int(j) if j < half else finite+int(j)-half]
                            for j in v_active]
                np.testing.assert_allclose(actual_v, expected_v, rtol=4e-11, atol=4e-11)
                assert np.linalg.eigvalsh(expected_v).min() > -1e-10
                paired_errors.append(float(np.max(np.abs(actual_v-expected_v))))
                v_inactive = np.argsort(v_owner)[np.flatnonzero(~live_v)]
                assert np.all(vv[p, v_inactive] == 0)
                if p == 0:
                    principal = hh[p][np.ix_(face_active, face_active)]
                    labels = tags
                    if paired_baseline is None: paired_baseline = (v_labels, actual_v.copy())
                    assert v_labels == paired_baseline[0]
                    np.testing.assert_allclose(actual_v, paired_baseline[1], rtol=4e-11, atol=4e-11)
            # q0's physical principal block is compared by owner labels;
            # the extra parent/carrier cells are never assumed a prefix.
            if baseline is None: baseline = (labels, principal.copy())
            assert labels == baseline[0]
            np.testing.assert_allclose(principal, baseline[1], rtol=3e-11, atol=3e-11)
            distrib_la.new_stack_routes()
            reduced = face_reduce_round(s, inf, t, mesh=mesh, budget=None, ordered=True,
                odd_moments=True, keep_budget=8, eigh_backend='auto', room=1<<30,
                admit=False, carrier=face_ritz_carrier(mesh, 8))
            model, _, _, diagnostics = jax.tree.map(host, reduced)
            red = diagnostics[0]
            assert np.all(red['gram_valid']), red['gram_min_relative']
            assert np.all(red['orientation_paired']) and np.all(red['retained_metric_positive'])
            assert np.all(red['infinite_weight_ok']) and np.all(diagnostics[1]['zero_policy'])
            b, values, live = model
            retained = (values[0][live[0]], b[0][:, live[0]] @ np.conj(b[0][:, live[0]].T))
            if len(records) == 0: retained_reference = retained
            np.testing.assert_allclose(retained[0], retained_reference[0], rtol=2e-8, atol=2e-9)
            np.testing.assert_allclose(retained[1], retained_reference[1], rtol=2e-7, atol=2e-8)
            records.append(dict(batch=batch, finite_carrier=int(carrier), side=int(side),
                own=np.asarray(t['own']).tolist(), raw_principal_max_error=expected_errors,
                paired_principal_max_error=paired_errors,
                expected_H_min=expected_min, gram_min_relative=red['gram_min_relative'].tolist(),
                retained_rank=red['retained_rank'].tolist(), routes=distrib_la.new_stack_routes()))
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(dict(verdict='PASS',records=records), indent=2)+'\n')
    print('MIXED_PARENT_GRAM PASS', jax.process_index(), flush=True)


if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
