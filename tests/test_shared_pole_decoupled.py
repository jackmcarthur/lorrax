"""The staged face route and its face products against the local round (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
A synthetic ordered pencil: four parents, each a full-rank positive real-pole model
W(z) = C diag(1/(z mu - 1)) C^H (k = n poles, so the half-pencil Gram is nonsingular)
with its exact moments M_k = sum_j c_j c_j^H mu_j^-(k+1), sampled at two imaginary
supports and their mirrors. The local round program (one whole parent per rank, the
numerical control of every route) and the decoupled face route (stage programs over
sub-batches, each eigh once over every parent, products by ``panel_matmul``) must
agree on the retained poles and factors to round-off.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _model(rng, n, k):
    mu = np.sort(rng.uniform(0.5, 4.0, k))                       # inverse poles, Ry^-2
    c = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    return mu, c / np.sqrt(n)


def _w(mu, c, z):
    return (c * (1 / (z * mu - 1))[None, :]) @ np.conj(c.T)


def _dw(mu, c, z):
    return (c * (-mu / (z * mu - 1) ** 2)[None, :]) @ np.conj(c.T)


def _moment(mu, c, k):
    return (c * (mu ** -(k + 1))[None, :]) @ np.conj(c.T)


def _orthonormal(rng, n, w):
    q, _ = np.linalg.qr(rng.normal(size=(n, w)) + 1j * rng.normal(size=(n, w)))
    return q


def _pencil(mesh, seed=0, n=24, k=24, width=4, iw=2, q=4):
    """States, infinity panels and tables of ``q`` parents in the ordered paired layout."""
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import round_tables
    rng = np.random.default_rng(seed)
    extent = port_extent(mesh)
    nodes = [0.7j, 1.9j]
    models = [_model(rng, n, k) for _ in range(q)]
    directions = [[_orthonormal(rng, n, width) for _ in nodes] for _ in range(q)]
    states = []
    for sign in (1, -1):
        for a, z in enumerate(nodes):
            zz = sign * z
            qs = np.stack([directions[p][a] for p in range(q)])
            os_ = np.stack([_w(*models[p], zz) @ directions[p][a] for p in range(q)])
            ds = np.stack([_dw(*models[p], zz) @ directions[p][a] for p in range(q)])
            states.append((np.full(q, zz, np.complex128), qs, os_, ds))
    moments = [np.stack([_moment(*models[p], kk) for p in range(q)]) for kk in range(4)]
    qi = np.stack([np.linalg.eigh((moments[1][p] + np.conj(moments[1][p].T)) / 2)[1][:, -iw:] for p in range(q)])
    infinity = (qi, *(np.einsum("pij,pjk->pik", m, qi) for m in moments))
    counts = np.full((q, len(states)), width)
    widths = [extent(width)] * len(states)
    tables = round_tables(counts, widths, [s[0][0] for s in states], [iw] * q, extent(iw),
                          column_extent=extent, ordered=True, odd_moments=True)
    return states, infinity, tables, models


def _pad(a, w):
    return np.pad(a, ((0, 0), (0, 0), (0, w - a.shape[-1])))


def test_decoupled_face_matches_local():
    """The decoupled face route (stage programs over sub-batches, eighs over every parent)
    against the local round on the synthetic pencil; the face products are panel_matmul."""
    mesh = _mesh()
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_execution import face_reduce_decoupled, face_ritz_carrier
    from gw.shared_pole_local import BATCH, reduce_round
    from gw.shared_pole_capacity import _local_eigenplan
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    extent = port_extent(mesh)
    states, infinity, tables, models = _pencil(mesh)
    q = len(models)
    width, iw = extent(4), extent(2)
    gram_keep = shared_real_pole_gates_ordered_v1["normalized_gram_keep"]["sector_threshold"]
    budget = 8
    side = int(tables["active"].shape[-1])

    def place(sharding):
        put = lambda a: jax.device_put(np.asarray(a), sharding)
        return ([(s[0], put(_pad(s[1], width)), put(_pad(s[2], width)), put(_pad(s[3], width))) for s in states],
                tuple(put(_pad(a, iw)) for a in infinity))
    ls, li = place(NamedSharding(mesh, P(BATCH)))
    local = reduce_round(ls, li, tables, real=q, mesh_xy=mesh, native_eigh=_local_eigenplan(mesh, side).native_fn,
                         ordered=True, odd_moments=True, keep_budget=budget, retain_span=True, gram_keep=gram_keep)
    fs, fi = place(NamedSharding(mesh, P(None, "x", "y")))
    for sub in (q, 2, 1):
        plan = _local_eigenplan(mesh, side)
        held = list(fs)
        dec = face_reduce_decoupled(held, fi, tables, mesh=mesh, eigh_plans=(plan, plan, plan), width=sub,
                                    ordered=True, odd_moments=True, keep_budget=budget, retain_span=True,
                                    gram_keep=gram_keep, carrier=face_ritz_carrier(mesh, budget))
        assert all(len(s) == 3 for s in held)          # the dW Q panels were released
        ld, dd = (jax.tree.map(np.asarray, r[3][0]) for r in (local, dec))
        assert np.all(dd["paired_metric_inverse_root_residual_relative"] < 1e-12), sub
        assert np.array_equal(ld["paired_metric_inverse_root_iterations"],
                              dd["paired_metric_inverse_root_iterations"]), sub
        for key in ("gram_spectrum_relative", "gram_min_relative", "paired_min_relative", "retained_metric_relative"):
            assert np.allclose(ld[key], dd[key], rtol=1e-8, atol=1e-12), (sub, key)
        for key in ("retained_rank", "paired_rank", "positive_count", "gram_valid", "orientation_paired"):
            assert np.array_equal(ld[key], dd[key]), (sub, key)
        lb, lp, la = (np.asarray(a) for a in local[0])
        db, dp, da = (np.asarray(a) for a in dec[0])
        assert dec[0][0].sharding.spec == P(None, "x", "y")
        assert np.array_equal(la, da)
        assert np.allclose(np.where(la, lp, 0), np.where(da, dp, 0), rtol=1e-8, atol=1e-10), sub
        for p in range(q):
            assert np.allclose(lb[p] @ np.conj(lb[p].T), db[p] @ np.conj(db[p].T), rtol=1e-7, atol=1e-9), (sub, p)
        # The face span's rows are in the face's tile-interleaved pencil order: position k of
        # the face holds row standard[k] of the whole-matrix order.
        from gw.shared_pole_pencil import join_vectors, split_vectors
        f = int(tables["order"].shape[-1])
        standard = join_vectors(split_vectors(np.arange(side)[None], (f // 2, f // 2, (side - f) // 2, (side - f) // 2), 1),
                                int(mesh.shape["y"]))[0]
        ly, dy = np.asarray(local[4]), np.zeros_like(np.asarray(dec[4]))
        dy[:, standard] = np.asarray(dec[4])
        for p in range(q):
            assert np.allclose(ly[p] @ np.conj(ly[p].T), dy[p] @ np.conj(dy[p].T), rtol=1e-7, atol=1e-9), (sub, p)


def test_face_products_match_matmul():
    """``panel_matmul`` in every transpose form, wide and two-column panels, with and
    without the latency-hiding option, against ``jnp.matmul`` on the CPU 2x2 face."""
    mesh = _mesh()
    from distrib_la import panel_matmul
    rng = np.random.default_rng(3)
    face = NamedSharding(mesh, P(None, "x", "y"))
    q, m, k, n = 3, 8, 12, 4
    op = {"N": lambda a: a, "T": lambda a: np.swapaxes(a, -1, -2), "C": lambda a: np.conj(np.swapaxes(a, -1, -2))}
    for ta in "NTC":
        for tb in "NTC":
            a = rng.normal(size=(q, m, k)) + 1j * rng.normal(size=(q, m, k))
            b = rng.normal(size=(q, k, n)) + 1j * rng.normal(size=(q, k, n))
            # op(stored) is the product's operand: stored = op^-1(a)
            sa = a if ta == "N" else (np.swapaxes(a, -1, -2) if ta == "T" else np.conj(np.swapaxes(a, -1, -2)))
            sb = b if tb == "N" else (np.swapaxes(b, -1, -2) if tb == "T" else np.conj(np.swapaxes(b, -1, -2)))
            assert np.allclose(op[ta](sa), a) and np.allclose(op[tb](sb), b)
            want = a @ b
            for columns in (k, 2):
                per_column = 16 * q * (m // 2 + n // 2)
                for options in (None, {"xla_gpu_enable_latency_hiding_scheduler": True}):
                    got = panel_matmul(jax.device_put(sa, face), jax.device_put(sb, face), mesh=mesh,
                                       panel_bytes=per_column * 2 * columns, transa=ta, transb=tb,
                                       compiler_options=options)
                    assert got.sharding.spec == P(None, "x", "y")
                    assert np.allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12), (ta, tb, columns)


def test_paired_schur_cut_takes_the_sector_keep():
    """The Schur cut of the paired stage uses the caller's keep (the ordered sector threshold),
    as the H'_vv cut does: H_r = diag(S, I) with S = diag(gamma_r), B = 0, so the restricted
    metric is the identity and the paired rank counts gamma_r above the cut plus the kept span."""
    from gw.shared_pole_local import _mm
    from gw.shared_pole_reduction import paired_stage
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    gamma_r = np.array([[1e-8 / 2, 1e-6, 0.5]])
    c = gamma_r.shape[-1]
    eye = np.eye(c, dtype=np.complex128)[None]
    zero = np.zeros_like(eye)
    h_r = np.block([[np.diag(gamma_r[0]).astype(np.complex128), zero[0]], [zero[0], eye[0]]])[None]
    stage = dict(kept=np.ones((1, c), bool), b_r=zero, h_r=jnp.asarray(h_r), g_r=jnp.asarray(np.eye(2 * c, dtype=np.complex128)[None]),
                 schur=jnp.asarray(np.diag(gamma_r[0]).astype(np.complex128)[None]))
    for keep, want in ((gates["normalized_gram_keep"]["sector_threshold"], 1), (None, 2)):
        out = paired_stage(dict(stage), jnp.asarray(gamma_r), jnp.asarray(eye), matmul=_mm, gates=gates, gram_keep=keep)
        assert int(np.asarray(out["count_r"])[0]) == want + c, (keep, np.asarray(out["count_r"]))
        assert float(np.asarray(out["paired_metric_residual_relative"])[0]) < 1e-12


def test_decoupled_cross_matches_the_joint_reduction():
    """face_cross_decoupled (stages over sub-batches, each eigh once over the stack) equals
    reduce_sector_pencil (one program per round) on synthetic definite joint pencils."""
    mesh = _mesh()
    from gw.shared_pole_execution import face_cross_decoupled
    from gw.shared_pole_local import _mm
    from gw.shared_pole_sectors import reduce_sector_pencil
    from gw.shared_pole_capacity import _local_eigenplan
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    rng = np.random.default_rng(5)
    q, k, nc, nt = 5, 8, 6, 4
    herm = lambda a: (a + np.conj(np.swapaxes(a, -1, -2))) / 2
    a = rng.normal(size=(q, k, k)) + 1j * rng.normal(size=(q, k, k))
    metric = herm(a @ np.conj(np.swapaxes(a, -1, -2))) / k + 0.1 * np.eye(k)
    value = herm(rng.normal(size=(q, k, k)) + 1j * rng.normal(size=(q, k, k)))
    oc = rng.normal(size=(q, nc, k)) + 1j * rng.normal(size=(q, nc, k))
    ot = rng.normal(size=(q, nt, k)) + 1j * rng.normal(size=(q, nt, k))
    face = NamedSharding(mesh, P(None, "x", "y"))
    eig = lambda m: tuple(np.linalg.eigh(np.asarray(m)))
    want, wd = reduce_sector_pencil(tuple(jnp.asarray(x) for x in (metric, value, oc, ot)),
                                    eigh=lambda m: tuple(jnp.asarray(v) for v in eig(m)), matmul=_mm, gates=gates)
    plan = _local_eigenplan(mesh, k)
    for width in (q, 2):
        got, gd = face_cross_decoupled([jax.device_put(x, face) for x in (metric, value, oc, ot)],
                                       mesh=mesh, eigh_plans=(plan, plan), width=width)
        assert np.array_equal(np.asarray(want[3]), np.asarray(got[3]))
        assert np.allclose(np.asarray(want[2]), np.asarray(got[2]), rtol=1e-10, atol=1e-12)
        for p in range(q):            # factors up to a phase per column: compare c_C c_T^H
            lw = np.asarray(want[0][p]) @ np.conj(np.asarray(want[1][p]).T)
            lg = np.asarray(got[0][p]) @ np.conj(np.asarray(got[1][p]).T)
            assert np.allclose(lw, lg, rtol=1e-8, atol=1e-10), (width, p)
        for key in ("gram_valid", "retained_metric_positive", "retained_rank"):
            assert np.array_equal(np.asarray(wd[key]), np.asarray(gd[key])), key


def test_release_selection_panels_frees_them_and_keeps_the_models():
    """After a round's CT pencils, the sectors' (Q, O, infinity) panels and spans are dropped
    (their bytes return before the CT eighs); the models a sub-batch slice reads are unchanged."""
    import gc
    import weakref
    from gw.shared_pole_sectors import release_selection_panels, slice_sector
    mesh = _mesh()
    face = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(3)
    put = lambda shape: jax.device_put(jnp.asarray(rng.normal(size=shape)), face)
    sector = dict(model=(put((4, 8, 8)), put((4, 8, 8))), signed=(put((4, 8, 8)),),
                  coefficients=put((4, 8, 8)), states=[(0, put((4, 8, 8)), put((4, 8, 8)))],
                  infinity=(put((4, 8, 8)),), tables={'t': np.arange(4)}, vectors=(put((4, 8, 8)),),
                  diagnostics={'d': np.arange(4.0)})
    before = slice_sector(sector, [1, 3], mesh)
    panels = [weakref.ref(a) for a in (*sector['states'][0][1:], *sector['infinity'], sector['coefficients'])]
    release_selection_panels([sector])
    gc.collect()
    assert all(ref() is None for ref in panels)
    assert sector['states'] is None and sector['infinity'] is None and sector['coefficients'] is None
    after = [np.asarray(a)[[1, 3]] for a in jax.tree.leaves((sector['model'], sector['signed']))]
    for a, b in zip(jax.tree.leaves((before['model'], before['signed'])), after):
        assert np.array_equal(np.asarray(a), b)


def test_scalar_face_round_matches_local(monkeypatch):
    """The scalar route's face round (one program, ``face_reduce_round``), whose pencil columns
    enter the face's tile-interleaved order, against the local round on the synthetic pencil."""
    mesh = _mesh()
    import gw.shared_pole_execution as ex
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_local import BATCH, reduce_round
    from gw.shared_pole_capacity import _local_eigenplan
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    # The face round's eigh stacks on route (c), as the staged route's run on CPU.
    monkeypatch.setattr(ex, "face_eigh", lambda mesh_, n, room=None: _local_eigenplan(mesh_, int(n)))
    extent = port_extent(mesh)
    states, infinity, tables, models = _pencil(mesh, seed=2)
    q = len(models)
    width, iw = extent(4), extent(2)
    gram_keep = shared_real_pole_gates_ordered_v1["normalized_gram_keep"]["sector_threshold"]
    side = int(tables["active"].shape[-1])

    def place(sharding):
        put = lambda a: jax.device_put(np.asarray(a), sharding)
        return ([(s[0], put(_pad(s[1], width)), put(_pad(s[2], width)), put(_pad(s[3], width))) for s in states],
                tuple(put(_pad(a, iw)) for a in infinity))
    ls, li = place(NamedSharding(mesh, P(BATCH)))
    local = reduce_round(ls, li, tables, real=q, mesh_xy=mesh, native_eigh=_local_eigenplan(mesh, side).native_fn,
                         ordered=True, odd_moments=True, keep_budget=8, gram_keep=gram_keep)
    fs, fi = place(NamedSharding(mesh, P(None, "x", "y")))
    face = ex.face_reduce_round(fs, fi, tables, mesh=mesh, budget=None, ordered=True, odd_moments=True,
                                keep_budget=8, admit=False, gram_keep=gram_keep,
                                carrier=ex.face_ritz_carrier(mesh, 8))
    (lb, lp, la), (fb, fp, fa) = ([np.asarray(a) for a in r[0]] for r in (local, face))
    assert np.array_equal(la, fa)
    assert np.allclose(np.where(la, lp, 0), np.where(fa, fp, 0), rtol=1e-8, atol=1e-10)
    for p in range(q):
        assert np.allclose(lb[p] @ np.conj(lb[p].T), fb[p] @ np.conj(fb[p].T), rtol=1e-7, atol=1e-9), p
