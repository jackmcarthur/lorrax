"""P4 native CCT stream against independent NumPy normal equations.

Run as a compute-node script with --out. Covers scalar/Pauli/Dirac spinors,
antiunitary transport, wraps, orbit/padding boundaries, q selection/completion,
real/complex channel padding, admission before allocation and runtime metadata
reuse. The injected small room tests the automatic route only; it is not a
physical capacity measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path

def numpy_reference(psi, wl, wr, plan, kgrid, gl=0, gr=0):
    import numpy as np
    from common.gamma_matrices import gamma_perm_phase_host
    nk, ns, mu = (plan.n_full, plan.nspinor, plan.n_centroid_packed)

    def projector(w):
        return np.einsum('pamn,n,pbvn->pambv', psi, w, psi.conj())

    def unfold(d):
        out = np.empty((nk, ns, mu, mu, ns), complex)
        for k in range(nk):
            p, op = (int(plan.irr_idx[k]), int(plan.sym_idx[k]))
            idx = plan.sym_perm[op]
            x = np.take(np.take(d[p], idx, axis=1), idx, axis=3)
            phase = np.exp(2j * np.pi * (plan.L_table[op] @ plan.k_parent_frac[p]))
            x = x * phase[None, :, None, None] * phase.conj()[None, None, None, :]
            if op >= plan.n_sym_spatial:
                x = x.conj()
            u = plan.spin_action_full[k]
            out[k] = np.einsum('ac,cmen,be->amnb', u, x, u.conj()).conj()
        return out
    left, right = (unfold(projector(wl)), unfold(projector(wr)))
    if ns == 4:
        (pl, fl), (pr, fr) = map(gamma_perm_phase_host, (gl, gr))
        right = np.take(np.take(right, pl, axis=1), pr, axis=4) * fl[None, :, None, None, None] * fr[None, None, None, None, :]
    z = (np.fft.ifftn(left.reshape(tuple(kgrid) + left.shape[1:]), axes=(0, 1, 2), norm='forward').conj() * np.fft.ifftn(right.reshape(tuple(kgrid) + right.shape[1:]), axes=(0, 1, 2), norm='forward')).sum(axis=(3, 6))
    return np.fft.fftn(z, axes=(0, 1, 2), norm='forward').reshape(nk, mu, mu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    o = ap.parse_args()
    import runtime
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host, barrier
    from common.grouped_layout import build_square_grouped_shard_layout
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    from isdf.core import c_q_from_psi_sm
    from distrib_la import gemm_plan
    from symmetry_maps import q_negation_index
    import isdf.core as owner
    assert mesh.size == 4
    o.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(762193)
    kgrid = (3, 2, 1)
    nk = 6
    np_ = 3
    nb = 8
    neg = np.asarray(q_negation_index(kgrid), np.int32)
    labels = np.arange(19) // 2
    layout = build_square_grouped_shard_layout(labels, tuple((mesh.shape[a] for a in ('x', 'y'))))
    mu = layout.axis.n_padded
    canonical_perm = np.tile(np.arange(19), (4, 1))
    for i in range(0, 18, 2):
        canonical_perm[1:, i:i + 2] = [i + 1, i]
    sym_perm = layout.axis.pack_permutations_host(canonical_perm)
    wraps = rng.integers(-1, 2, (4, mu, 3)).astype(np.float64)
    wraps[0] = 0
    wraps[:, ~layout.axis.active_mask] = 0
    irr = np.array([0, 1, 2, 0, 1, 2], np.int32)
    sym = np.array([0, 0, 0, 1, 2, 3], np.int32)
    kp = rng.random((np_, 3))
    cases = []
    for ns in (1, 2, 4):
        u = []
        for k in range(nk):
            if ns == 1:
                z = np.ones((1, 1), complex)
            else:
                a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
                z = np.linalg.qr(a)[0]
                if ns == 4:
                    z = np.block([[z, np.zeros((2, 2))], [np.zeros((2, 2)), (-1 if k % 2 else 1) * z]])
            if k < 3:
                z = np.eye(ns, dtype=complex)
            u.append(z)
        plan = CentroidKUnfoldPlan(mesh, layout, irr, sym, sym_perm, wraps, kp, np.asarray(u), 2, ns)
        psi = rng.normal(size=(np_, ns, mu, nb)) + 1j * rng.normal(size=(np_, ns, mu, nb))
        psi[:, :, ~layout.axis.active_mask, :] = 0
        pm = device_put_process_local(psi, NamedSharding(mesh, P(None, None, 'x', 'y')))
        pn = device_put_process_local(psi.transpose(0, 3, 1, 2), NamedSharding(mesh, P(None, 'x', None, 'y')))
        wl = np.array([1, 1, 1, 1, 0, 0, 0, 0.0])
        wr = np.array([0, 0, 1, 1, 1, 1, 1, 1.0])
        gemm = gemm_plan(mesh, m=ns * mu, n=ns * mu, k=nb, nq=np_, dtype=np.complex128, layout='face', warmup=False)
        for gl, gr in ((0, 0),) if ns < 4 else ((0, 0), (1, 2), (2, 3)):
            dense = numpy_reference(psi, wl, wr, plan, kgrid, gl, gr)
            full = c_q_from_psi_sm(pm, pn, wl, wr, k_unfold_plan=plan, kgrid=kgrid, mesh_xy=mesh, gemm=gemm, gamma_L=gl, gamma_R=gr)
            old_h = np.asarray(gather_to_host(full))
            scale = max(float(np.abs(dense).max()), 1.0)
            np.testing.assert_allclose(old_h, dense, atol=3e-11 * scale, rtol=3e-12)
            for rows, completion in ((None, False), (np.array([4, 0, 2], np.int32), True)):
                from isdf.core import _c_q_parent_program
                from common.gamma_matrices import gamma_perm_phase_host
                if ns == 4:
                    (pl, fl), (pr, fr) = map(gamma_perm_phase_host, (gl, gr))
                    gamma = (np.stack([pl[:2] % 2, pr[:2] % 2]), np.stack([fl, fr]), np.asarray([gl != 0, gr != 0], np.int32))
                else:
                    gamma = (np.zeros((2, ns), np.int32), np.ones((2, ns), np.complex128), np.zeros(2, np.int32))
                qrows = np.arange(nk, dtype=np.int32) if rows is None else rows
                run, zero, meta = _c_q_parent_program(plan, kgrid, gemm, pm.shape, pm.dtype, len(qrows), completion, 3)
                aval = jax.ShapeDtypeStruct(meta['output_shape'], pm.dtype, sharding=meta['sharding'])
                args = (pm, pn, wl, wr, gamma, qrows, neg[qrows] if completion else np.empty((0,), np.int32))
                exe = run.lower(aval, *args).compile()
                meta = {k: v for k, v in meta.items() if k != 'sharding'}
                stats = exe.memory_analysis()
                assert stats.alias_size_in_bytes == stats.output_size_in_bytes
                result = exe(zero(), *args)
                result.block_until_ready()
                got = np.asarray(gather_to_host(result))
                expected = dense if not completion else (dense + np.conj(dense[neg]))[rows]
                np.testing.assert_allclose(got, expected, atol=3e-11 * scale, rtol=3e-12)
                cases.append(dict(ns=ns, gl=gl, gr=gr, completion=completion, output_rows=rows.tolist() if rows is not None else None, max_abs=float(np.abs(got - expected).max()), **meta, compiled=dict(argument=int(stats.argument_size_in_bytes), output=int(stats.output_size_in_bytes), temporary=int(stats.temp_size_in_bytes), aliased=int(stats.alias_size_in_bytes))))
                from lxkit import memory
                import isdf.core as core
                old_room = memory.room
                memory.room = lambda *a, **kw: 65536
                try:
                    automatic = c_q_from_psi_sm(pm, pn, wl, wr, k_unfold_plan=plan, kgrid=kgrid, mesh_xy=mesh, gemm=gemm, gamma_L=gl, gamma_R=gr, q_rows=rows, q_neg_idx=neg if completion else None)
                    np.testing.assert_allclose(np.asarray(gather_to_host(automatic)), expected, atol=3e-11 * scale, rtol=3e-12)
                    if rows is not None:
                        remapped = rows[::-1].copy()
                        n_before = len(core._isdf_pipeline_cache)
                        other = c_q_from_psi_sm(pm, pn, wl, wr, k_unfold_plan=plan, kgrid=kgrid, mesh_xy=mesh, gemm=gemm, gamma_L=gl, gamma_R=gr, q_rows=remapped, q_neg_idx=neg)
                        np.testing.assert_allclose(np.asarray(gather_to_host(other)), expected[::-1], atol=3e-11 * scale, rtol=3e-12)
                        assert len(core._isdf_pipeline_cache) == n_before, 'q values multiplied program cache'
                    cases[-1]['automatic_room_bytes'] = 65536
                    decisions = [value for key, values in core._isdf_pipeline_cache.items() if isinstance(key, tuple) and key[0] == 'cct_room' and (key[1][1] is plan) and (key[1][-2:] == (len(expected), completion)) for room, tests, value in values if all((test(65536) == out if r == room else 65536 == room for test, out, r in tests))]
                    assert decisions
                    chosen = decisions[-1]
                    cases[-1]['auto_selected'] = {k: v for k, v in chosen[2].items() if k != 'sharding'}
                    cases[-1]['auto_compiled_new_including_initial_output'] = chosen[3]
                finally:
                    memory.room = old_room
                if ns == 4 and gl == 0:
                    cct_count = sum((isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], tuple) and (k[0][0] == 'cct_parent_columns') for k in core._isdf_pipeline_cache))
                    cases[-1]['dirac_program_count_initial'] = cct_count
                elif ns == 4:
                    count = sum((isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], tuple) and (k[0][0] == 'cct_parent_columns') for k in core._isdf_pipeline_cache))
                    assert count == cct_count, 'gamma values multiplied program cache'
                del automatic
                del result
        del pm, pn
    from gw.isdf_fitting import add_pad_diagonal_sharded
    from isdf.core import complete_ordered_pair_normal_equations
    pad_cases = []
    sh = NamedSharding(mesh, P(None, 'x', 'y'))
    for weights in ((1.0, -0.3), (0.7 + 0.2j, -0.1j)):
        raw = []
        for i in range(2):
            c = rng.normal(size=(nk, mu, mu)) + 1j * rng.normal(size=(nk, mu, mu))
            c[:, ~layout.axis.active_mask, :] = 0
            c[:, :, ~layout.axis.active_mask] = 0
            raw.append(c)
        complex_weights = any((complex(w).imag != 0 for w in weights))
        if complex_weights:
            old = sum((w * (c + np.conj(c[neg])) for w, c in zip(weights, raw)))
            old = add_pad_diagonal_sharded(device_put_process_local(old, sh), layout.axis.active_mask, 19.0, mesh_xy=mesh)
        else:
            old = add_pad_diagonal_sharded(device_put_process_local(sum((w * c for w, c in zip(weights, raw))), sh), layout.axis.active_mask, 19.0, mesh_xy=mesh)
            old = complete_ordered_pair_normal_equations(old, neg)
        new = sum((w * (c + np.conj(c[neg])) for w, c in zip(weights, raw)))
        new = add_pad_diagonal_sharded(device_put_process_local(new, sh), layout.axis.active_mask, 19.0, mesh_xy=mesh)
        delta = float(np.max(np.abs(np.asarray(gather_to_host(old)) - np.asarray(gather_to_host(new)))))
        assert delta < 1e-12, delta
        pad_cases.append(dict(complex_weights=complex_weights, max_abs=delta))
    barrier('cct_oracle.done')
    if jax.process_index() == 0:
        (o.out / 'result.json').write_text(json.dumps(dict(scope='Synthetic exact existing-owner CCT stream; no physical claim', production=True, pad_order_cases=pad_cases, impl_sha256=hashlib.sha256(Path(owner.__file__).read_bytes()).hexdigest(), cases=cases), indent=2) + '\n')
        print('CCT_STREAM_ORACLE PASS', len(cases), 'native full C and independent NumPy Fourier+typed-transport reference', flush=True)
if __name__ == '__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
