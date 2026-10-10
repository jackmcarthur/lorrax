"""Host-level glue lowers one program per shape, not one per op (CPU, seconds).

An eager jnp op outside any jit compiles a program of its own, per shape, in
every process.  Each test counts MLIR lowerings over repeated calls of one
helper and holds the values (and the sharding) to the op-by-op spelling:

1. ``runtime.padding.pad_to_axis`` / ``pad_square``: one program per shape;
   inside a jit they inline (the same values as eager).
2. ``symmetry_maps.slice_q_full_to_ibz``: one gather per shape, any rows.
3. ``symmetry_maps.little_group_covariance_residual``: one program for all
   sampled (parent, op) pairs.
4. ``gw.mpa.sector_sigma._unfold_w_rows``: one program for both branches.
   On the CPU it agrees with the eager shard_map it replaces to 4 ulp, not bit
   for bit, because the fused spin rotation's complex multiply-adds are
   code-generated differently from op-by-op dispatch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

_LOWERINGS = []
jax.monitoring.register_event_time_span_listener(
    lambda event, *a, **k: _LOWERINGS.append(1)
    if event == "/jax/core/compile/jaxpr_to_mlir_module_duration" else None)


def _lowerings(fn):
    jax.clear_caches()
    n0 = len(_LOWERINGS)
    out = fn()
    jax.block_until_ready(out)
    return len(_LOWERINGS) - n0, out


def _mesh(px=2, py=2):
    return Mesh(np.array(jax.devices()[:px * py]).reshape(px, py), ("x", "y"))


def _rnd(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_pad_helpers_one_program_and_inline():
    from runtime.padding import pad_square, pad_to_axis, padded_axis
    rng = np.random.default_rng(0)
    mesh = _mesh()
    tag = padded_axis(6, 4, name="band")                  # carrier 8
    A = jax.device_put(_rnd(rng, 4, 8, 6), NamedSharding(mesh, P(None, ("x", "y"), None)))
    wide = jax.device_put(_rnd(rng, 4, 8, 12), NamedSharding(mesh, P(None, "x", "y")))

    def calls():
        return [pad_to_axis(A, tag, axis=2, fill=f) for f in (0.0, 3.0, -1e10)]
    n, outs = _lowerings(calls)
    assert n == 1, f"pad_to_axis at one shape lowered {n} programs"
    for f, out in zip((0.0, 3.0, -1e10), outs):
        want = np.concatenate([np.asarray(A), np.full((4, 8, 2), f)], axis=2)
        assert np.array_equal(np.asarray(out), want)
    n, out = _lowerings(lambda: pad_to_axis(wide, tag, axis=2, fill=2.0))
    assert n == 1
    assert np.array_equal(np.asarray(out)[..., :6], np.asarray(wide)[..., :6])
    assert np.all(np.asarray(out)[..., 6:] == 2.0)
    assert out.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)

    S = _rnd(rng, 3, 6, 6)
    n, sq = _lowerings(lambda: [pad_square(S, tag, pad_diagonal=1.0),
                                pad_square(S + 1, tag, pad_diagonal=1.0)])
    assert n == 1, f"pad_square at one shape lowered {n} programs"
    want = np.zeros((3, 8, 8), complex)
    want[:, :6, :6] = S
    want[:, 6, 6] = want[:, 7, 7] = 1.0
    assert np.array_equal(np.asarray(sq[0]), want)
    traced = jax.jit(lambda s: pad_square(s, tag, pad_diagonal=1.0))(S)
    assert np.array_equal(np.asarray(traced), want)
    # Inline: the caller's jaxpr holds the pad ops, not a call of the helper's program.
    eqns = jax.make_jaxpr(lambda s: pad_to_axis(s, tag, axis=2))(S).jaxpr.eqns
    assert "_pad_body" not in {str(e.params.get("name")) for e in eqns}


def test_ibz_slice_one_gather_any_rows():
    from symmetry_maps import slice_q_full_to_ibz
    rng = np.random.default_rng(1)
    mesh = _mesh()
    out_sh = NamedSharding(mesh, P(None, "x", "y"))
    V = jax.device_put(_rnd(rng, 8, 4, 4), out_sh)
    rows = ([0, 3, 5], [1, 2, 7], [6, 0, 4])
    n, outs = _lowerings(lambda: [slice_q_full_to_ibz(V, r, out_sharding=out_sh) for r in rows])
    assert n == 1, f"slice_q_full_to_ibz at one shape lowered {n} programs"
    for r, out in zip(rows, outs):
        assert np.array_equal(np.asarray(out), np.asarray(V)[r])
        assert out.sharding.is_equivalent_to(out_sh, 3)


def test_covariance_residual_one_program():
    from symmetry_maps import little_group_covariance_residual
    rng = np.random.default_rng(2)
    n_mu, grid = 4, (2, 2, 1)
    eye = np.eye(3, dtype=np.int64)
    flips = [np.diag(d).astype(np.int64) for d in ((1, 1, 1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1))]
    S = np.concatenate([np.stack(flips), -np.stack(flips)])
    perm = np.stack([np.roll(np.arange(n_mu), k) for k in range(4)] * 2).astype(np.int32)
    L = np.zeros((8, n_mu, 3))
    L[1, 0] = (1, 0, 0)
    reps = np.arange(4)
    q = np.stack(np.unravel_index(reps, grid), axis=1) / np.asarray(grid)
    V = jnp.asarray(_rnd(rng, 4, n_mu, n_mu))
    kw = dict(q_irr_frac=q, q_irr_full_idx=reps, sym_mats_k=S, sym_perm=perm, L_table=L,
              kgrid=grid, n_sym_spatial=4, parents="all")
    assert np.array_equal(S[0], eye)
    n, got = _lowerings(lambda: little_group_covariance_residual(V, **kw))
    assert got["n_ops_sampled"] == 12
    assert n == 1, f"{got['n_ops_sampled']} pairs lowered {n} programs"
    Vh, ref = np.asarray(V), []
    for p in range(4):
        for s in range(1, 4):
            ph = np.exp(1j * (2 * np.pi * (L[s] @ q[p])))
            img = Vh[p][perm[s]][:, perm[s]] * (ph[:, None] * np.conj(ph)[None, :])
            ref.append(np.max(np.abs(img - Vh[p])))
    assert got["max_abs"] == pytest.approx(max(ref), rel=1e-12)
    assert got["scale"] == float(np.max(np.abs(Vh)))


def _old_unfold_w_rows(W, Wt, tables, rows, mesh_xy):
    """The eager shard_map over baked host tables this lane replaced (reference only)."""
    from symmetry_maps import local_unfold_load_tables
    from symmetry_maps.maps import _rotate_open_spin_centroid_operator
    from jax import shard_map
    rows = np.asarray(rows)
    t0 = tables._replace(**{f: getattr(tables, f)[rows] for f in
                            ("row", "trs", "lsrc", "rsrc", "mph", "nph", "spin")})
    na, nb, n_par = int(W.shape[2]), int(W.shape[4]), int(W.shape[0])

    def local(w, wt):
        t = local_unfold_load_tables(t0)
        flat = lambda a: a.reshape(a.shape[0], a.shape[1] * na, a.shape[3] * nb)
        g, gt = flat(w), flat(wt)
        src = jnp.concatenate((g, gt), axis=0)[t.row + n_par * t.trs]
        nl = int(g.shape[2])
        idx = (jnp.maximum(t.lsrc, 0)[:, :, None] * nl + jnp.maximum(t.rsrc, 0)[:, None, :])
        V = jnp.take_along_axis(src.reshape(src.shape[0], -1), idx.reshape(src.shape[0], -1),
                                axis=1).reshape(src.shape)
        V = t.mph[:, :, None] * V * t.nph[:, None, :]
        V = jnp.where((t.lsrc >= 0)[:, :, None] & (t.rsrc >= 0)[:, None, :], V, 0)
        O = _rotate_open_spin_centroid_operator(
            V.reshape(V.shape[0], V.shape[1] // na, na, V.shape[2] // nb, nb), np.asarray(t0.spin))
        return O.reshape(O.shape[0], O.shape[1] * na, O.shape[3] * nb)
    spec = P(None, "x", None, "y", None)
    return shard_map(local, mesh=mesh_xy, in_specs=(spec, spec), out_specs=P(None, "x", "y"),
                     check_vma=False)(W, Wt)


def test_unfold_w_rows_one_program_bitwise():
    from gw.mpa.sector_sigma import _unfold_w_rows
    from symmetry_maps import UnfoldLoadTables
    rng = np.random.default_rng(3)
    mesh = _mesh()
    nq, n_par, m, ns = 6, 3, 4, 2
    ml = m * ns // 2                                       # merged endpoints per shard

    def tables(seed):
        r = np.random.default_rng(seed)
        q, _ = np.linalg.qr(_rnd(r, nq, ns, ns))
        q[:, 0, 1] = 0.0                                   # a structural zero to skip
        return UnfoldLoadTables(
            row=r.integers(0, n_par, nq).astype(np.int32),
            trs=(r.random(nq) < 0.5).astype(np.int32),
            lsrc=r.integers(-1, ml, (nq, m * ns)).astype(np.int32),
            rsrc=r.integers(-1, ml, (nq, m * ns)).astype(np.int32),
            mph=np.exp(2j * np.pi * r.random((nq, m * ns))),
            nph=np.exp(2j * np.pi * r.random((nq, m * ns))),
            spin=q.astype(np.complex128), n_parent=n_par, mesh_shape=(2, 2))
    sh = NamedSharding(mesh, P(None, "x", None, "y", None))
    W, Wt = (jax.device_put(_rnd(rng, n_par, m, ns, m, ns), sh) for _ in range(2))
    rows = np.array([0, 2, 5])
    branches = (tables(10), tables(11))
    n, got = _lowerings(lambda: [_unfold_w_rows(W, Wt, t, rows, mesh) for t in branches])
    assert n == 1, f"two unfold branches lowered {n} programs"
    for t, out in zip(branches, got):
        got_h, want = np.asarray(out), np.asarray(_old_unfold_w_rows(W, Wt, t, rows, mesh))
        assert np.array_equal(got_h == 0, want == 0)      # the structural zeros and the masks
        assert np.max(np.abs(got_h - want)) <= 4 * np.finfo(float).eps * np.max(np.abs(want))
        assert out.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
