"""Complex-residue pair transpose and bounded endpoint-routing gates."""
from functools import partial
import numpy as np


def check_shared_pole_unfold(mesh, profile=False):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import symmetry_maps as S
    import distrib_la as D
    from symmetry_maps._shard_map import shard_map
    rng = np.random.default_rng(1911)
    c = rng.normal(size=(2, 12, 1, 5)) + 1j*rng.normal(size=(2, 12, 1, 5))
    active = np.ones(12, dtype=bool); active[[2, 8]] = False
    c[:, ~active] = 0
    d = rng.normal(size=(2, 5)) + 1j*rng.normal(size=(2, 5))
    q = np.array([[.125, .25, -.125], [.25, -.125, .375]])
    irr, sym = np.array([1, 0, 1, 0]), np.arange(4)
    perm = np.tile(np.arange(12), (4, 1))
    for row in (1, 3):
        perm[row, [0, 1, 6, 7]] = [1, 0, 7, 6]
    wraps = rng.integers(-2, 3, (4, 12, 3))
    spin = np.ones((4, 1, 1), dtype=complex)
    def put(a, spec):
        sh = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(a.shape, sh, lambda i: a[i])
    def err(a, ref):
        return float(jnp.max(jnp.abs(a - put(ref, P(None, 'x', 'y')))))
    def oracle(p):
        child = c[irr[:, None], p[sym]]
        phase = np.exp(2j*np.pi*np.einsum('qi,qmi->qm', q[irr], wraps[sym]))
        child *= phase[:, :, None, None]
        child[sym >= 2] = child[sym >= 2].conj()
        cc = child[:, :, 0, :]
        return (cc*d[irr, None, :]) @ cc.conj().transpose(0, 2, 1)
    cx = put(c, P(None, 'x', None, None)); cy = put(c, P(None, 'y', None, None))
    w, wt = jax.jit(lambda x, y, weights, lo, hi: D.contract_faces(
        x, y, weights, lo, hi, mesh=mesh, return_transpose=True))(
            cx, cy, put(d, P()), put(np.zeros(2, dtype=int), P()), put(np.full(2, 5), P()))
    left = S.certify_endpoint_locality(perm, mesh=mesh, mesh_axis='x', active_mask=active)
    right = S.certify_endpoint_locality(perm, mesh=mesh, mesh_axis='y', active_mask=active)
    assert left['is_local'] and right['is_local']
    @partial(shard_map, mesh=mesh, in_specs=(P(None, 'x', 'y'),)*2,
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def unfold(a, at):
        return S.unfold_operator_local(
            a, irr_idx=irr, sym_idx=sym, q_irr_frac=q,
            left_local_perm=left['local_perm'], right_local_perm=right['local_perm'],
            left_L_table=wraps, right_L_table=wraps, n_sym_spatial=2,
            trs_rule='pair_transpose', transposed_parent_local=at)
    compiled = jax.jit(unfold).lower(w, wt).compile()
    if profile:
        import ctypes
        cudart = ctypes.CDLL('libcudart.so.13')
        jax.block_until_ready((w, wt))
        assert cudart.cudaProfilerStart() == 0
        for _ in range(25):
            jax.block_until_ready(compiled(w, wt))
        assert cudart.cudaProfilerStop() == 0
    got = compiled(w, wt)
    local_error = err(got, oracle(perm))
    assert local_error < 3e-12, local_error
    hlo = compiled.as_text().lower()
    assert not any(x in hlo for x in ('all-gather', 'all-to-all', 'all-reduce', 'collective-permute'))
    # Conjugating W would conjugate the time weight too. This twin shares
    # every source/phase table but deliberately supplies that wrong partner.
    wrong = compiled(w, w.conj())
    red_error = err(wrong, oracle(perm))
    assert red_error > .1, red_error
    # Identity group (no TR, no wraps): original parent byte/value parity.
    @partial(shard_map, mesh=mesh, in_specs=P(None, 'x', 'y'),
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def identity(a):
        return S.unfold_operator_local(
            a, irr_idx=np.arange(2), sym_idx=np.zeros(2, dtype=int), q_irr_frac=q,
            left_local_perm=left['local_perm'][:1], right_local_perm=right['local_perm'][:1],
            left_L_table=np.zeros((1, 12, 3)), right_L_table=np.zeros((1, 12, 3)),
            n_sym_spatial=1, trs_rule='pair_transpose', transposed_parent_local=a)
    identity_error = float(jnp.max(jnp.abs(jax.jit(identity)(w)-w)))
    assert identity_error == 0
    nonlocal_perm = perm.copy()
    nonlocal_perm[1, [0, 6]] = nonlocal_perm[1, [6, 0]]
    nonlocal_perm[3, [1, 7]] = nonlocal_perm[3, [7, 1]]
    cert = S.certify_endpoint_locality(nonlocal_perm, mesh=mesh, mesh_axis='x', active_mask=active)
    assert not cert['is_local'] and cert['local_perm'] is None
    fallback = jnp.zeros_like(got)
    costs = []
    route_reuse = []
    for a, b in ((0, 3), (3, 5)):
        faces = []
        for axis in ('x', 'y'):
            width = ((b-a+int(mesh.size)-1)//int(mesh.size))*int(mesh.size)
            panel = np.pad(c[..., a:b], ((0,0),(0,0),(0,0),(0,width-(b-a))))
            spec = P(None, axis, None, 'y' if axis == 'x' else 'x')
            f = put(panel, spec)
            child, cost = S.unfold_endpoint_panel(
                f, irr_idx=irr, sym_idx=sym, q_irr_frac=q,
                source_perm=nonlocal_perm, L_table=wraps, spin_action_full=spin,
                n_sym_spatial=2, active_mask=active, mesh=mesh, mesh_axis=axis,
                max_live_bytes=1_000_000)
            # Bind metadata once, as the Sigma time loop does. Distinct
            # factor inputs must reuse the same compiled route and cannot
            # be replaced by a cached child-factor value.
            traces = []
            def repeated_route(factors):
                traces.append(1)
                return S.unfold_endpoint_panel(
                    factors, irr_idx=irr, sym_idx=sym, q_irr_frac=q,
                    source_perm=nonlocal_perm, L_table=wraps,
                    spin_action_full=spin, n_sym_spatial=2,
                    active_mask=active, mesh=mesh, mesh_axis=axis,
                    max_live_bytes=1_000_000)[0]
            sharding = NamedSharding(mesh, spec)
            reused = jax.jit(repeated_route, in_shardings=sharding,
                             out_shardings=sharding)
            reuse_errors = []
            with jax.log_compiles(True):
                for scale in (1., 2., -1., 3., .5):
                    changed = put(panel*scale, spec)
                    repeated = reused(changed)
                    jax.block_until_ready(repeated)
                    reuse_errors.append(float(jnp.max(jnp.abs(repeated-scale*child))))
            assert len(traces) == 1, traces
            assert max(reuse_errors) < 3e-12, reuse_errors
            route_reuse.append(dict(axis=axis, k_panel=b-a, calls=5,
                                    traces=len(traces), errors=reuse_errors))
            faces.append(child); costs.append(cost)
        # The service test contracts the tiny result with an independent
        # dense oracle; production uses the planned G face GEMM.
        dm = put(np.pad(d[irr, a:b], ((0,0),(0,width-(b-a)))), P())
        fallback += jax.jit(lambda x,y,w: (x[:,:,0,:]*w[:,None,:]) @
                            y[:,:,0,:].conj().swapaxes(-1,-2),
                            out_shardings=NamedSharding(mesh,P(None,'x','y')))(*faces,dm)
    fallback_error = err(fallback, oracle(nonlocal_perm))
    assert fallback_error < 3e-12, fallback_error
    try:
        S.unfold_endpoint_panel(put(panel, P(None,'x',None,'y')), irr_idx=irr, sym_idx=sym, q_irr_frac=q,
            source_perm=nonlocal_perm, L_table=wraps, spin_action_full=spin,
            n_sym_spatial=2, active_mask=active, mesh=mesh, mesh_axis='x', max_live_bytes=1)
    except ValueError as ex:
        assert 'max_live_bytes' in str(ex)
    else:
        raise AssertionError('missing capacity refusal')
    return dict(status='PASS', local_error=local_error, identity_error=identity_error,
                wrong_conjugation_error=red_error, fallback_error=fallback_error,
                crossing_count=cert['crossing_count'], panel_costs=costs,
                route_reuse=route_reuse)


def test_shared_pole_unfold():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    check_shared_pole_unfold(Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y')))
