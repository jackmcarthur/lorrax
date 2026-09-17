"""SVD-selected real/complex planted measures through production port owners."""
from pathlib import Path
import json
import os
import numpy as np


def check(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la as D
    from shared_pole_round_helpers import round_states
    from gw.shared_pole_gates import shared_pole_passivity, shared_pole_reciprocity
    from gw.shared_pole_pencil import assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_shared_pole_pencil
    from gw.shared_pole_local import _batch_put, batch_to_face, face_rows, reduce_round, round_tables
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates
    from runtime.padding import padded_axis

    rng = np.random.default_rng(918)
    real_c = rng.normal(size=(8, 64)) * .04
    complex_c = real_c + 1j*rng.normal(size=(8, 64)) * .04
    poles = np.linspace(.1, 4., 64)
    points = (.5+.4j, 1.9+.6j)
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    def put(a):
        a = np.asarray(a, np.complex128)[None]
        return jax.make_array_from_callback(a.shape, face, lambda i: a[i])
    def extent(width):
        return padded_axis(width, mesh, name='closure_port',
                           specs=((P('x', 'y'), 0), (P('x', 'y'), 1))).carrier
    ep = D.plan('eigh', mesh, n=8, backend='off', batched_route='batch_reshard')
    sp = D.plan('eigh', mesh, n=16, backend='off', batched_route='batch_reshard')
    def mm(a, b, **kwargs):
        # Provider-free staged route: 'auto' would resolve and probe a 2-D
        # provider, which needs one JAX process per mesh cell.
        return D.matmul(a, b, mesh=mesh, backend='off',
                        batched_route='batch_reshard', **kwargs)
    adj = lambda a: a.conj().swapaxes(-1, -2)
    rows = []
    for case, c in (('real_residues', real_c), ('complex_residues', complex_c)):
        def sample(s):
            return (c/(s-poles)) @ adj(c), (-c/(s-poles)**2) @ adj(c)
        recipe = dict(fit_ids=[0, 1], distinct_id=[0, 1], role=[0, 0],
                      held=[False, False], z_ry=np.sqrt(points), direction_cutoff=.8,
                      multiplet_relative_tolerance=1e-6)
        ranks = jax.device_count()
        round_, counts, roles = round_states(
            mesh, lambda slot, i: sample(points[i]), recipe, n=8, eig=ep, svd=sp, extent=extent, batch=True)
        assert round_[1][1] is round_[0][2] and round_[3][1] is round_[2][2]
        m1, m3 = c @ adj(c)/2, (c*poles) @ adj(c)/2
        qi = np.linalg.eigh(m1)[1][:, -2:]
        infinity = tuple(put(a) for a in (qi, m1 @ qi, m3 @ qi))
        tables = round_tables(counts, [st[1].shape[-1] for st in round_], [st[0] for st in round_],
                              [2] * ranks, 2, column_extent=extent, ordered=False, odd_moments=False)
        model, _, vectors, (diagnostics, zero, retained, _) = reduce_round(
            round_, tuple(_batch_put(mesh, np.broadcast_to(a, (ranks,) + a.shape).copy())
                          for a in (qi, m1 @ qi, m3 @ qi)),
            tables, real=ranks, mesh_xy=mesh, native_eigh=ep.native_fn, ordered=False, odd_moments=False,
            keep_budget=None)
        to_face, first = batch_to_face(mesh), face_rows(mesh, (0,))
        states = [(st[0], *(first(to_face(a)) for a in st[1:])) for st in round_]
        counts = counts[:1]
        b, t, active = first(to_face(model[0])), vectors[0][:1], vectors[1][:1]
        got = mm(b*jnp.where(active, 1/(-.37-t), 0)[:, None, :], b, transb='C')
        reference = put(sample(-.37)[0])
        green = shared_pole_reciprocity(got, reference, gates=gates)
        assert bool(jnp.all(green['passed']))
        assert bool(jnp.all(diagnostics['gram_valid']))
        assert bool(jnp.all(zero['zero_policy']))
        assert bool(jnp.all(jnp.where(active, t > 0, True)))
        assert all(float(jnp.max(a)) < 1e-10 for a in retained.values())
        passive = shared_pole_passivity((b,t,active), put(np.eye(8)), eta_ry=.25,
                                        matmul=mm, eigh=ep.batched, gates=gates)
        assert bool(jnp.all(passive['passivity']))
        # Independent latent projection from NumPy singular subspaces;
        # no production face is gathered to the host.
        latent = []
        for i, point in enumerate(points):
            w = sample(point)[0]
            q = adj(np.linalg.svd(w)[2])[:, :counts[0, 2*i]]
            latent.extend(((adj(c) @ q)/(point-poles)[:, None],
                           (adj(c) @ (w @ q))/(point.conjugate()-poles)[:, None]))
        x = np.concatenate([*latent, adj(c) @ qi], axis=-1)
        basis = np.linalg.svd(x, full_matrices=False)[0][:, :int(counts.sum()+2)]
        tr = adj(basis) @ (poles[:, None]*basis)
        out = c @ basis
        expected = out @ np.linalg.solve(-.37*np.eye(tr.shape[0])-tr, adj(out))
        projection = float(jnp.linalg.norm(got-put(expected))/jnp.linalg.norm(put(expected)))
        assert projection < 1e-10, projection
        # Red twin: precisely the former same-Q policy, same selected Q.
        old_states = []
        for i in (0, 2):
            q = states[i][1]
            for s in (states[i][0], states[i+1][0]):
                w, dw = (put(a) for a in sample(s))
                old_states.append((s, q, mm(w,q), mm(dw,q)))
        pencil = assemble_shared_pole_pencil(old_states, infinity, matmul=mm)
        ep_old = D.plan('eigh',mesh,n=pencil[0].shape[-1],backend='off',batched_route='batch_reshard')
        old_active = jax.device_put(tables['active'][:1], NamedSharding(mesh, P()))
        old, _, _ = reduce_shared_pole_pencil(pencil, old_active,
                            eigh=ep_old.batched, matmul=mm, gates=gates)
        ob, ot, om = old
        bad = mm(ob*jnp.where(om, 1/(-.37-ot), 0)[:, None, :], ob, transb='C')
        red = shared_pole_reciprocity(bad, reference, gates=gates)
        imag = float(jnp.linalg.norm(got.imag)/jnp.linalg.norm(got))
        bad_imag = float(jnp.linalg.norm(bad.imag)/jnp.linalg.norm(bad))
        if case == 'real_residues':
            # The same predicate, same threshold, same reference: green passes
            # only because the ports changed. Red must be applicable and fail.
            assert bool(jnp.all(green['applicable'])) and bool(jnp.all(red['applicable']))
            assert not bool(jnp.any(red['passed']))
            assert imag < 1e-11 and bad_imag > 1e-4
        else:
            assert not bool(jnp.any(green['applicable'])) and imag > .01
        rows.append(dict(case=case, imaginary_relative=imag,
                         same_q_imaginary_relative=bad_imag, projection_relative=projection,
                         same_q_gates={k: np.asarray(v).tolist() for k, v in red.items()},
                         pencil_side=int(tables['active'].shape[-1]),
                         minimum_pole=float(jnp.min(jnp.where(active,t,jnp.inf))),
                         gates={k:np.asarray(v).tolist() for k,v in green.items()}))
    return dict(status='PASS', rows=rows,
                scope='P4 production round selection and round program; planted real and complex measures; former same-Q red twin')


def test_conjugate_closure():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    mesh = Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    check(mesh)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys
        import jax
        from common.collectives import resolve_mesh, barrier
        assert jax.process_count() == 4
        result = check(resolve_mesh())
        result['job_step'] = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
        barrier('conjugate-closure-gate')
    run_main_and_finalize(main)
