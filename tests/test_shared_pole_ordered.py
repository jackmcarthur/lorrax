"""Ordered (time-reversal-broken) shared-pole model on exact particle-hole plants.

Plants are Wc(z) = C (z s3 - M)^-1 C^H with M = diag(dE) + L L^H > 0. The
gates mirror the TRMODEL planted oracle (sandbox run 422_trmodel_20260915):
full-order exactness with real poles, the positive-pole carrier with its odd
term, projected z-moments m0..m3, equality with today's even construction on
time-reversal-symmetric data, the conjugate-partner dedupe, generic-q assembly
across the q/-q parents, and Sigma's two synthesized orientations against the
Lehmann sums.
"""
import numpy as np

adj = lambda a: a.conj().swapaxes(-1, -2)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
ZS = (.7 + .2j, -1.3 + .05j, 2.0j, -.4 - .9j, 3.1 + .6j)


class _Plant:
    def __init__(self, M, C):
        h = M.shape[0] // 2
        self.M, self.C = M, C
        self.s3 = np.diag(np.r_[np.ones(h), -np.ones(h)]).astype(complex)
        assert np.linalg.eigvalsh(M)[0] > 0

    def F(self, z):
        return self.C @ np.linalg.solve(z * self.s3 - self.M, adj(self.C))

    def dF(self, z):
        r = np.linalg.solve(z * self.s3 - self.M, adj(self.C))
        return -self.C @ np.linalg.solve(z * self.s3 - self.M, self.s3 @ r)

    def moment(self, k):
        x = self.s3 @ adj(self.C)
        for _ in range(k):
            x = self.s3 @ (self.M @ x)
        return self.C @ x

    def modes(self):
        """Exact poles and PSD residue magnitudes (C x)(C x)^H/|x^H s3 x|."""
        w, x = np.linalg.eig(self.s3 @ self.M)
        norm = np.einsum("in,ij,jn->n", x.conj(), self.s3, x).real
        cx = self.C @ x
        return w.real, [np.outer(cx[:, n], cx[:, n].conj()) / abs(norm[n]) for n in range(len(w))]


def _trim(rng, nt, nport, eps):
    """q = -q plant: L = [l; conj l] keeps A Hermitian, B symmetric; eps breaks TRS."""
    dE = np.linspace(.3, 2.5, nt)
    l = (rng.normal(size=(nt, nt)) + 1j * eps * rng.normal(size=(nt, nt))) * .25
    L = np.vstack([l, l.conj()])
    c = (rng.normal(size=(nport, nt)) + 1j * eps * rng.normal(size=(nport, nt))) * .3
    return _Plant(np.diag(np.r_[dE, dE]).astype(complex) + L @ adj(L), np.hstack([c, c.conj()]))


def _generic_pair(rng, nt, nport):
    """q != -q plant and its exact particle-hole partner at -q."""
    dE = rng.uniform(.3, 2.5, 2 * nt)
    L = (rng.normal(size=(2 * nt, nt)) + .4j * rng.normal(size=(2 * nt, nt))) * .25
    M = np.diag(dE).astype(complex) + L @ adj(L)
    C = (rng.normal(size=(nport, 2 * nt)) + 1j * rng.normal(size=(nport, 2 * nt))) * .3
    tx = np.block([[np.zeros((nt, nt)), np.eye(nt)], [np.eye(nt), np.zeros((nt, nt))]])
    return _Plant(M, C), _Plant(tx @ M.conj() @ tx, C.conj() @ tx)


def _ops():
    import jax.numpy as jnp

    def mm(a, b, transa="N", transb="N"):
        op = lambda x, t: x if t == "N" else jnp.conj(jnp.swapaxes(x, -1, -2))
        return op(a, transa) @ op(b, transb)
    return mm, jnp.linalg.eigh


def _put(x):
    import jax.numpy as jnp
    return jnp.asarray(np.asarray(x, complex)[None])


def _states(plant, points, ordered=True):
    """(node, Q, WQ, action). Ordered: X(z) on Q and X(conj z) on O = W Q per point,
    then their particle-hole mirrors X(-z), X(-conj z) on the same directions."""
    points = tuple(points) if isinstance(points, (list, tuple)) else (points,)
    out, mirrors = [], []
    for z in points:
        W = plant.F(z)
        Q = adj(np.linalg.svd(W)[2])
        O = W @ Q
        if ordered:
            dW = plant.dF(z)
            out += [(z, Q, O, dW @ Q), (np.conj(z), O, adj(W) @ O, adj(dW) @ O)]
            zb = -np.conj(z)
            mirrors += [(-z, Q, plant.F(-z) @ Q, plant.dF(-z) @ Q),
                        (zb, O, plant.F(zb) @ O, plant.dF(zb) @ O)]
        else:
            dW = plant.dF(z) / (2 * z)
            out += [(z**2, Q, O, dW @ Q), (np.conj(z)**2, O, adj(W) @ O, adj(dW) @ O)]
    return [(complex(n), _put(q), _put(o), _put(d)) for n, q, o, d in out + mirrors]


def _ordered(plant, points, r_inf=1):
    import jax.numpy as jnp
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    mm, eigh = _ops()
    qi = np.linalg.eigh(plant.moment(1))[1][:, -r_inf:]
    infinity = tuple(_put(a) for a in (qi, *(plant.moment(k) / 2 @ qi for k in range(4))))
    pencil = assemble_ordered_shared_pole_pencil(_states(plant, points), infinity, matmul=mm)
    active = jnp.ones((1, pencil[0].shape[-1]), bool)
    model, signed, diag = reduce_ordered_shared_pole_pencil(
        pencil, active, eigh=eigh, matmul=mm, gates=gates)
    return model, signed, diag, infinity


def _physical(rng, nt, nport):
    """RPA plant: Phi = [rho, conj rho], M = diag(D, D) + Phi^H V Phi, C = V Phi."""
    D = rng.uniform(.3, 2.5, nt)
    rho = (rng.normal(size=(nport, nt)) + 1j * rng.normal(size=(nport, nt))) * .4
    phi = np.hstack([rho, rho.conj()])
    a = rng.normal(size=(nport, nport)) + 1j * rng.normal(size=(nport, nport))
    V = a @ adj(a) / nport + .1 * np.eye(nport)
    M = np.diag(np.r_[D, D]).astype(complex) + adj(phi) @ V @ phi
    return _Plant(M, V @ phi), D, V


def _signed_value(signed, z):
    c, mu, kept = (np.asarray(a[0]) for a in signed)
    return (c[:, kept] / (z * mu[kept] - 1)) @ adj(c[:, kept])


def test_ordered_full_order_exact_with_real_poles_and_paired_carrier():
    from gw.shared_pole_gates import ordered_shared_pole_value, signed_shared_pole_passivity
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    mm, eigh = _ops()
    plant = _trim(np.random.default_rng(20260915), 6, 3, eps=.4)
    model, signed, diag, _ = _ordered(plant, .9 + .35j)
    assert bool(diag["gram_valid"][0]) and bool(diag["infinite_weight_ok"][0])
    assert int(diag["positive_count"][0]) == int(diag["negative_count"][0]) == 6
    exact = np.sort(np.linalg.eigvals(plant.s3 @ plant.M).real)
    mu = np.asarray(signed[1][0])[np.asarray(signed[2][0])]
    assert np.max(abs(np.sort(1 / mu) - exact) / abs(exact)) < 1e-10
    for z in ZS:
        assert rel(_signed_value(signed, z), plant.F(z)) < 1e-12
        # q = -q: the parent is its own partner; the odd term uses the same vector.
        value = np.asarray(ordered_shared_pole_value(model, model, z, matmul=mm)[0])
        assert rel(value, plant.F(z)) < 1e-12
    b = np.asarray(model[0][0])[:, np.asarray(model[2][0])]
    assert np.linalg.norm((b @ adj(b)).imag) > .1 * np.linalg.norm(b @ adj(b))
    eta = .25
    f = plant.F(1j * eta)
    top = np.linalg.eigvalsh(-(f + adj(f)) / 2)[-1]
    passive = signed_shared_pole_passivity(
        signed, _put(np.eye(3) * np.sqrt(.9 / top)), eta_ry=eta, matmul=mm, eigh=eigh, gates=gates)
    assert bool(passive["passivity"][0])
    assert float(passive["passivity_antihermitian_relative"][0]) > 1e-3


def test_ordered_projected_moments_at_reduced_order():
    from gw.shared_pole_gates import ordered_moment_identity
    mm, _ = _ops()
    plant = _trim(np.random.default_rng(7), 12, 3, eps=.4)
    _, signed, diag, infinity = _ordered(plant, .9 + .35j, r_inf=2)
    assert int(diag["positive_count"][0]) + int(diag["negative_count"][0]) < 24
    rows = ordered_moment_identity(signed, infinity, matmul=mm)
    assert float(rows["m1"][0]) < 1e-12 and float(rows["m3"][0]) < 1e-12
    c, mu, kept = (np.asarray(a[0]) for a in signed)
    qi = np.asarray(infinity[0][0])
    a = adj(qi) @ c[:, kept]
    scale = np.linalg.norm(adj(qi) @ plant.moment(1) @ qi)
    for k in range(4):
        model = (a * mu[kept] ** -(k + 1)) @ adj(a)
        assert np.linalg.norm(model - adj(qi) @ plant.moment(k) @ qi) < 1e-11 * scale
    assert np.linalg.norm(plant.moment(0)) > 1e-3


def test_ordered_pole_bound_covers_rpa_poles():
    from gw.shared_pole_gates import ordered_pole_bound_ry
    mm, eigh = _ops()
    for seed in (31, 32, 33):
        plant, D, V = _physical(np.random.default_rng(seed), 6, 4)
        lam, U = np.linalg.eigh(V)
        hinv = (U / np.sqrt(lam)) @ adj(U)
        bound = float(np.asarray(ordered_pole_bound_ry(
            _put(plant.moment(1) / 2), _put(hinv), energy_span_ry=D.max(), gap_ry=D.min(),
            matmul=mm, eigh=eigh))[0])
        assert np.max(np.abs(np.linalg.eigvals(plant.s3 @ plant.M))) <= bound
    assert np.isinf(np.asarray(ordered_pole_bound_ry(
        _put(plant.moment(1) / 2), _put(hinv), energy_span_ry=D.max(), gap_ry=0.0, matmul=mm, eigh=eigh))[0])


def test_ordered_equals_even_at_an_active_keep_cut_on_time_reversal_symmetric_data():
    """Two supports over-span the latent space, so the relative keep cut removes
    directions; the paired-basis cut keeps exactly the even route's span (P3)."""
    import jax.numpy as jnp
    from gw.shared_pole_gates import apply_shared_pole_zero_policy
    from gw.shared_pole_pencil import assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_shared_pole_pencil
    from gw.shared_pole_recipe import (shared_real_pole_gates_v1_r3b as gates,
                                       shared_real_pole_gates_ordered_v1 as ordered_gates)
    mm, eigh = _ops()
    plant = _trim(np.random.default_rng(17), 6, 4, eps=0.)
    points = (.9 + .35j, 1.7 + .6j)
    ordered_model, _, diag, _ = _ordered(plant, points)
    ordered_model, _ = apply_shared_pole_zero_policy(ordered_model, gates=ordered_gates)
    qi = np.linalg.eigh(plant.moment(1))[1][:, -1:]
    infinity = tuple(_put(a) for a in (qi, plant.moment(1) / 2 @ qi, plant.moment(3) / 2 @ qi))
    pencil = assemble_shared_pole_pencil(_states(plant, points, ordered=False), infinity, matmul=mm)
    even_model, even_diag, _ = reduce_shared_pole_pencil(
        pencil, jnp.ones(pencil[0].shape[:2], bool), eigh=eigh, matmul=mm, gates=gates)
    even_model, _ = apply_shared_pole_zero_policy(even_model, gates=gates)
    assert int(even_diag["retained_rank"][0]) < pencil[0].shape[-1]
    assert int(diag["retained_rank"][0]) == int(even_diag["retained_rank"][0])
    (bo, po, ao), (be, pe, ae) = ((np.asarray(a[0]) for a in m) for m in (ordered_model, even_model))
    assert int(ao.sum()) == int(ae.sum())
    for z in ZS:
        ordered = (bo[:, ao] / (z**2 - po[ao])) @ adj(bo[:, ao])
        even = (be[:, ae] / (z**2 - pe[ae])) @ adj(be[:, ae])
        assert rel(ordered, even) < 1e-10


def test_ordered_equals_even_construction_on_time_reversal_symmetric_data():
    import jax.numpy as jnp
    from gw.shared_pole_pencil import assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates
    mm, eigh = _ops()
    plant = _trim(np.random.default_rng(11), 12, 3, eps=0.)
    point = .9 + .35j
    _, signed, _, _ = _ordered(plant, point)
    qi = np.linalg.eigh(plant.moment(1))[1][:, -1:]
    infinity = tuple(_put(a) for a in (qi, plant.moment(1) / 2 @ qi, plant.moment(3) / 2 @ qi))
    pencil = assemble_shared_pole_pencil(_states(plant, point, ordered=False), infinity, matmul=mm)
    (b, poles, act), _, _ = reduce_shared_pole_pencil(
        pencil, jnp.ones(pencil[0].shape[:2], bool), eigh=eigh, matmul=mm, gates=gates)
    b, poles, act = (np.asarray(a[0]) for a in (b, poles, act))
    b, poles = b[:, act], poles[act]
    for z in ZS:
        even = (b / (z**2 - poles)) @ adj(b)
        assert rel(_signed_value(signed, z), even) < 1e-12
        assert rel(_signed_value(signed, -z), even) < 1e-12


def _round_setup():
    import jax
    from jax.sharding import Mesh, PartitionSpec as P
    from lxkit.testing import require_devices
    import distrib_la as D
    from runtime.padding import padded_axis
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    extent = lambda width: padded_axis(width, mesh, name="round_port",
                                       specs=((P("x", "y"), 0), (P("x", "y"), 1))).carrier
    ep = D.plan("eigh", mesh, n=8, backend="off", batched_route="batch_reshard")
    sp = D.plan("eigh", mesh, n=16, backend="off", batched_route="batch_reshard")
    return mesh, extent, ep, sp


def _round_infinity(mesh, plants, orders):
    """Infinity panels (Q_inf, M_k Q_inf) of each slot's plant in batch layout, two directions."""
    from gw.shared_pole_local import _batch_put
    qi = [np.linalg.eigh(p.moment(1))[1][:, -2:] for p in plants]
    return tuple(_batch_put(mesh, np.stack([q if k is None else p.moment(k) / 2 @ q for p, q in zip(plants, qi)]))
                 for k in (None, *orders))


def test_round_program_pairs_mirrors_on_parents_of_different_sides():
    """Round selection and the round program on a q = -q TR-broken round of two parents with
    different pencil sides (slots 2, 3 synthetic; every slot its own partner, identity realization):
    each mirror X(-node) sits on its original's directions, an imaginary support mirrors on its own
    sample, the conjugate O panels are the next state's directions, every real slot is in the paired
    layout and its signed model reproduces its plant. RED TWIN: unmirrored nodes are refused."""
    import jax
    from shared_pole_round_helpers import round_states
    from gw.shared_pole_local import reduce_round, round_tables
    mesh, extent, ep, sp = _round_setup()
    n = 8
    plants = [_trim(np.random.default_rng(41), 6, n, eps=.4), _trim(np.random.default_rng(42), 3, n, eps=.4)]
    slot_plants = [plants[0], plants[1], plants[1], plants[1]]
    nodes = [.9 + .35j, 1.7 + .6j, .5j]
    recipe = dict(fit_ids=[0, 1, 2], distinct_id=[0, 1, 2], role=[0, 0, 1], held=[False] * 3,
                  z_ry=[dict(real=v.real, imag=v.imag) for v in nodes], direction_cutoff=1e-3,
                  imaginary_width=4, multiplet_relative_tolerance=1e-6)
    states, counts, roles = round_states(
        mesh, lambda slot, i: (slot_plants[slot].F(nodes[i]), slot_plants[slot].dF(nodes[i]) / (2 * nodes[i])),
        recipe, n=n, eig=ep, svd=sp, extent=extent, ordered=True, real=2, batch=True)
    half = len(states) // 2
    assert len(states) == 12
    for i in range(half):
        assert states[half + i][0] == -states[i][0] and states[half + i][1] is states[i][1]
    assert states[1][1] is states[0][2] and states[3][1] is states[2][2]
    assert all(row.get("mirror") for row in roles[0][half:]) and not any(row.get("mirror") for row in roles[0][:half])
    infinity = _round_infinity(mesh, slot_plants, range(4))
    tables = round_tables(counts, [st[1].shape[-1] for st in states], [st[0] for st in states], [2, 2, 0, 0], 2,
                          column_extent=extent, ordered=True, odd_moments=True)
    assert tables["own"][0] != tables["own"][1]
    run = lambda t: reduce_round(states, infinity, t, real=2, mesh_xy=mesh, native_eigh=ep.native_fn,
                                 ordered=True, odd_moments=True, keep_budget=None)
    _, signed, (diag, _, _, _) = run(tables)
    diag = jax.tree.map(np.asarray, diag)
    assert diag["orientation_paired"][:2].all() and diag["gram_valid"][:2].all()
    c, mu, kept = (np.asarray(a) for a in signed)
    for slot, plant in enumerate(plants):
        cs, ms = c[slot][:, kept[slot]], mu[slot][kept[slot]]
        for zz in ZS:
            assert rel((cs / (zz * ms - 1)) @ adj(cs), plant.F(zz)) < 1e-10
    points = tables["points"].copy()
    points[:, points.shape[1] // 2:] = points[:, :points.shape[1] // 2]
    _, _, (bad, _, _, _) = run(dict(tables, points=points))
    assert not np.asarray(bad["orientation_paired"])[:2].any()


def test_dedupe_drops_duplicate_partners_and_equals_even_on_symmetric_data():
    """On a TRS plant the imaginary-role and Re z = 0 partners of W Q lie in span(Q); the dedupe drops them, and the
    ordered model built through the round selection and the round program equals the even model at the production
    cut (the planted analogue of the MoS2 P3 gate)."""
    import jax
    from shared_pole_round_helpers import round_states
    from gw.shared_pole_local import reduce_round, round_tables
    mesh, extent, ep, sp = _round_setup()
    n = 8
    plant = _trim(np.random.default_rng(43), 6, n, eps=0.)
    nodes = [.9 + .35j, 1.7 + .6j, .5j, .35j]
    recipe = dict(fit_ids=list(range(4)), distinct_id=list(range(4)), role=[0, 0, 1, 0], held=[False] * 4,
                  z_ry=[dict(real=v.real, imag=v.imag) for v in nodes], direction_cutoff=1e-3,
                  imaginary_width=4, multiplet_relative_tolerance=1e-6)
    models, ranks = {}, {}
    for label, ordered in (("even", False), ("ordered", True)):
        states, counts, _ = round_states(
            mesh, lambda slot, i: (plant.F(nodes[i]), plant.dF(nodes[i]) / (2 * nodes[i])), recipe,
            n=n, eig=ep, svd=sp, extent=extent, ordered=ordered, batch=True)
        if ordered:
            assert len(states) == 12
        infinity = _round_infinity(mesh, [plant] * 4, range(4) if ordered else (1, 3))
        tables = round_tables(counts, [st[1].shape[-1] for st in states], [st[0] for st in states], [2] * 4, 2,
                              column_extent=extent, ordered=ordered, odd_moments=True)
        model, _, (diag, zero, _, _) = reduce_round(states, infinity, tables, real=4, mesh_xy=mesh,
                                                    native_eigh=ep.native_fn, ordered=ordered, odd_moments=True,
                                                    keep_budget=None)
        assert bool(np.asarray(zero["zero_policy"]).all())
        b, poles, act = (np.asarray(a)[0] for a in model)
        models[label] = (b[:, act], poles[act])
        ranks[label] = int(np.asarray(diag["retained_rank"])[0])
    assert ranks["ordered"] == ranks["even"] < tables["active"].shape[-1]
    assert models["ordered"][0].shape[-1] == models["even"][0].shape[-1]
    for zz in ZS:
        value = lambda m: (m[0] / (zz**2 - m[1])) @ adj(m[0])
        assert rel(value(models["ordered"]), value(models["even"])) < 1e-10


def test_generic_q_positive_halves_assemble_the_galerkin_model():
    from gw.shared_pole_gates import ordered_shared_pole_value
    mm, _ = _ops()
    q, mq = _generic_pair(np.random.default_rng(13), 12, 3)
    for z in ZS:
        assert rel(mq.F(z), q.F(-z).T) < 1e-13
    model_q, signed_q, _, _ = _ordered(q, .9 + .35j)
    model_m, _, _, _ = _ordered(mq, .9 + .35j)
    for z in ZS:
        stored = np.asarray(ordered_shared_pole_value(model_q, model_m, z, matmul=mm)[0])
        assert rel(stored, _signed_value(signed_q, z)) < 1e-12
    # RED TWIN: the identity partner (q's own model in the -q slot) is not the Galerkin model.
    worst = max(rel(np.asarray(ordered_shared_pole_value(model_q, model_q, z, matmul=mm)[0]),
                    _signed_value(signed_q, z)) for z in ZS)
    assert worst > 1e-2, worst


def test_sigma_orientations_are_lehmann_sums_of_complex_residues():
    """synthesize_shared_pole_parents on an ordered q = -q model: Wplus is the
    positive-frequency Lehmann sum, Wtranspose the negative-frequency one."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit.testing import require_devices
    from gw.mpa.sigma import synthesize_shared_pole_parents
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    plant = _trim(np.random.default_rng(20260915), 6, 4, eps=.4)
    model, _, _, _ = _ordered(plant, .9 + .35j)
    active = np.asarray(model[2][0])
    b = np.asarray(model[0][0])[:, active]
    poles2 = np.asarray(model[1][0])[active]
    assert b.shape[-1] == 6 and np.linalg.norm((b @ adj(b)).imag) > .1 * np.linalg.norm(b @ adj(b))
    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec))
    gemm = jax.jit(lambda x, y: x @ y, out_shardings=NamedSharding(mesh, P(None, "x", "y")))
    w, residues = plant.modes()
    for tau in (.7 + .2j, -.3j):
        E = .3
        lplus = sum(r * np.exp(-1j * (wn - E) * tau) for wn, r in zip(w, residues) if wn > 0)
        lminus = sum(r * np.exp(-1j * (-wn - E) * tau) for wn, r in zip(w, residues) if wn < 0)
        plus, transposed = jax.jit(lambda x, y, p, r: synthesize_shared_pole_parents(
            x, y, p, r, E, tau, mesh_xy=mesh, gemm=gemm))(
            put(b[None, :, None, :], P(None, "x", None, "y")),
            put(b[None, :, None, :], P(None, "y", None, "x")),
            put(poles2[None], P()), put(np.array([[0, 6]], np.int32), P()))
        assert rel(np.asarray(plus)[0], lplus) < 1e-12
        assert rel(np.asarray(transposed)[0], lminus) < 1e-12
        assert rel(lplus.T, lplus) > 1e-2


def test_sigma_hole_branch_routes_minus_q_transpose_at_generic_q():
    """Ordered store on a (3,1,1) grid: q = 1/3 and -q = 2/3 are independent
    parents, Gamma is a q = -q parent. Conduction windows (W_+) and valence
    windows through ``shared_pole_hole_kernel`` equal the exact plant's positive-
    and negative-frequency Lehmann sums, i.e. R_-(q) = R_+(-q)^T."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit.testing import require_devices
    from gw.mpa.sigma import (
        shared_pole_hole_kernel, synthesize_shared_pole_parents)
    from symmetry_maps import q_negation_index
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(29)
    gamma = _trim(rng, 6, 4, eps=.4)
    q, mq = _generic_pair(rng, 6, 4)
    plants = (gamma, q, mq)
    minus_q = q_negation_index((3, 1, 1))
    assert minus_q.tolist() == [0, 2, 1]
    width, factors, poles, bounds = 8, [], [], []
    for plant in plants:
        model, _, _, _ = _ordered(plant, .9 + .35j)
        active = np.asarray(model[2][0])
        b, p2 = np.asarray(model[0][0])[:, active], np.asarray(model[1][0])[active]
        assert b.shape[-1] <= width
        factors.append(np.pad(b, ((0, 0), (0, width - b.shape[-1]))))
        poles.append(np.pad(p2, (0, width - p2.shape[-1]), constant_values=1.))
        bounds.append([0, b.shape[-1]])
    f = np.stack(factors)[:, :, None, :]
    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec))
    gemm = jax.jit(lambda x, y: x @ y, out_shardings=NamedSharding(mesh, P(None, "x", "y")))
    E, tau = .3, .7 + .2j
    plus, _ = jax.jit(lambda x, y, p, r: synthesize_shared_pole_parents(
        x, y, p, r, E, tau, mesh_xy=mesh, gemm=gemm))(
        put(f, P(None, "x", None, "y")), put(f, P(None, "y", None, "x")),
        put(np.stack(poles), P()), put(np.asarray(bounds, np.int32), P()))
    valence = shared_pole_hole_kernel(mesh)(plus, put(minus_q, P()))
    assert valence.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
    plus, valence = np.asarray(plus), np.asarray(valence)
    for row, plant in enumerate(plants):
        w, residues = plant.modes()
        lplus = sum(r * np.exp(-1j * (wn - E) * tau) for wn, r in zip(w, residues) if wn > 0)
        lminus = sum(r * np.exp(-1j * (-wn - E) * tau) for wn, r in zip(w, residues) if wn < 0)
        assert rel(plus[row], lplus) < 1e-12
        assert rel(valence[row], lminus) < 1e-12
    assert rel(valence[1], plus[1].T) > 1e-2


def test_two_component_ordered_store_synthesizes_lehmann_sums(tmp_path):
    """Two-component magnet store (N_spinor = 2, time reversal broken) on a (3,1,1)
    grid with independent q/-q parents and Gamma. The charge operator is mu x mu, so
    the stored factor keeps spin axis 1 and the header records the source N_spinor.
    Written and read back through the store, Sigma's synthesis gives W_+(q) and the
    hole kernel W_+(-q)^T; both equal the exact plants' Lehmann sums (< 1e-12)."""
    import jax
    from types import SimpleNamespace
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit.testing import require_devices
    from common.centroid_basis import PackedCentroidBasis
    from symmetry_maps import QirrTables, centroid_source_map_and_wrap, q_negation_index
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from gw.shared_pole_recipe import CapacityLedger, shared_real_pole_v1_r3b
    from gw.qgrid_symmetry import shared_pole_operator_realizer
    from gw.mpa.sigma import (
        shared_pole_hole_kernel, synthesize_shared_pole_parents)
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    rotations = np.eye(3, dtype=np.int32)[None]
    sym = SimpleNamespace(sym_matrices=rotations, translations=np.zeros((1, 3)),
                          trs_allowed=False, active_symmetry_rows=np.arange(1, dtype=np.int32),
                          operation_typing_source="planted two-component magnet fixture")
    sym.operation_rows = lambda rows: (
        np.asarray([rotations[0] * (-1 if r >= 1 else 1) for r in rows]),
        np.zeros((len(rows), 3)), np.asarray(rows) >= 1)
    sym.spinor_action = lambda rows, nspinor: np.ones((len(rows), 1, 1), np.complex128)
    cents = np.asarray([[1, 0, 0], [0, 1, 0], [2, 0, 0], [0, 2, 0], [3, 1, 0], [1, 3, 0], [2, 2, 0]], np.int32)
    grid = (4, 4, 1)
    basis = PackedCentroidBasis.build(cents, sym, grid, mesh)
    perm, wraps = centroid_source_map_and_wrap(cents, rotations, sym.translations,
                                               np.asarray(grid, np.int32), extend_trs=True)
    qt = QirrTables(irr_idx_q=np.arange(3, dtype=np.int32), sym_idx_q=np.zeros(3, np.int32),
                    q_irr_frac=np.asarray([[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]]),
                    sym_perm=perm, L_table=wraps, n_sym_spatial=1)
    meta = SimpleNamespace(mu_basis=basis, nspinor=2, nspinor_wfnfile=2, nkx=3, nky=1, nkz=1,
                           fft_grid=grid, nk_tot=3, n_rmu=len(cents))
    assert store.charge_representation(meta)
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=1 << 30)
    meta.shared_pole_capacity.reserve("fixture_live_bound", resident_bytes_per_rank=4096,
                                      workspace_bytes_per_rank=0)
    meta.shared_pole_capacity.live_stages = ("fixture_live_bound",)
    tables = {"qirr": qt, "q_irr_full_idx": np.arange(3, dtype=np.int64), "sym": sym}
    recipe = {"version": "shared_real_pole_v1_r3b", "gate_version": "shared_real_pole_gates_ordered_v1",
              "operator_realization": shared_real_pole_v1_r3b["operator_realization"]}
    identity = {key: "planted-" + key for key in store._IDENTITY_KEYS}

    rng = np.random.default_rng(31)
    n = basis.n_logical
    gamma = _trim(rng, 6, n, eps=.4)
    q, mq = _generic_pair(rng, 6, n)
    plants = (gamma, q, mq)
    width, factors, poles, counts = 8, [], [], []
    for plant in plants:
        model, _, _, _ = _ordered(plant, .9 + .35j)
        active = np.asarray(model[2][0])
        b, p2 = np.asarray(model[0][0])[:, active], np.asarray(model[1][0])[active]
        order = np.argsort(p2)
        b, p2 = b[:, order], p2[order]
        assert b.shape[-1] <= width
        factors.append(np.pad(b, ((0, 0), (0, width - b.shape[-1]))))
        poles.append(np.pad(p2, (0, width - p2.shape[-1]), constant_values=1.))
        counts.append(b.shape[-1])
    packed = basis.pack_host(np.stack(factors)[:, :, None, :], axis=1)
    put = lambda x, spec: jax.make_array_from_callback(
        np.shape(x), NamedSharding(mesh, spec), lambda idx: np.asarray(x)[idx])
    path = tmp_path / "model_two_component_ordered.h5"
    written = store.write_shared_pole_model(
        path, put(packed, P(None, "x", None, "y")), put(np.stack(poles), P(None, "y")),
        np.asarray(counts, np.int64), q_span=(0, 3), meta=meta, tables=tables, recipe=recipe,
        receipts={"identity": identity, "scope": "planted two-component"}, ordered=True)
    assert written["finalized"]
    header = store.validate_shared_pole_model(path, expected_identity=identity, mesh_xy=mesh,
                                              capacity=meta.shared_pole_capacity)
    assert header["representation"] == "scalar-ordered-ph" and header["nspinor"] == 2
    with SlabIO(path, mode="r", mesh=mesh) as io:
        b_X, b_Y, p2, K = store.read_shared_pole_faces(io, (0, 3), meta=meta, header=header)
    assert b_X.shape[2] == 1 and np.asarray(K).tolist() == counts
    gemm = jax.jit(lambda x, y: x @ y, out_shardings=NamedSharding(mesh, P(None, "x", "y")))
    E, tau = .3, .7 + .2j
    bounds = np.stack([np.zeros(3), np.asarray(counts)], axis=1).astype(np.int32)
    plus, transposed = jax.jit(lambda x, y, p, r: synthesize_shared_pole_parents(
        x, y, p, r, E, tau, mesh_xy=mesh, gemm=gemm))(b_X, b_Y, p2, put(bounds, P()))
    # The Sigma realization gate admits the two-component charge store.
    realize = shared_pole_operator_realizer(meta, header, q_full_idx=np.arange(3), mesh_xy=mesh)
    realized = realize(plus, transposed)[0]
    valence = shared_pole_hole_kernel(mesh)(plus, put(q_negation_index((3, 1, 1)), P()))
    unpack = jax.jit(lambda a: basis.unpack_operator(a, spec=P(None, "x", "y")))
    plus, realized, valence = (np.asarray(unpack(a))[:, :n, :n] for a in (plus, realized, valence))
    for row, plant in enumerate(plants):
        w, residues = plant.modes()
        lplus = sum(r * np.exp(-1j * (wn - E) * tau) for wn, r in zip(w, residues) if wn > 0)
        lminus = sum(r * np.exp(-1j * (-wn - E) * tau) for wn, r in zip(w, residues) if wn < 0)
        assert rel(plus[row], lplus) < 1e-12
        assert rel(realized[row], lplus) < 1e-12
        assert rel(valence[row], lminus) < 1e-12
    assert rel(valence[1], plus[1].T) > 1e-2


def test_debug_even_part_kernel_equals_union_store_synthesis():
    """LORRAX_DEBUG_SHARED_POLE_EVEN_PART: the even part W^even_q = [W_q + W_-q^T]/2 of an ordered store is the
    union store {b(q)/sqrt2 at Omega(q)} and {conj b(-q)/sqrt2 at Omega(-q)}. Sigma's synthesis of that union store
    (both branches) equals shared_pole_even_part_kernel applied to the original store's full-q W_+; exclude_q0 keeps
    the original branch W at q = 0."""
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from lxkit.testing import require_devices
    from gw.mpa.sigma import (
        shared_pole_even_part_kernel, shared_pole_hole_kernel, synthesize_shared_pole_parents)
    from symmetry_maps import q_negation_index
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(41)
    q, mq = _generic_pair(rng, 6, 4)
    plants = (_physical(rng, 6, 4)[0], q, mq)
    minus = q_negation_index((3, 1, 1)).tolist()
    models = []
    for plant in plants:
        model, _, _, _ = _ordered(plant, .9 + .35j)
        active = np.asarray(model[2][0]).astype(bool)
        models.append((np.asarray(model[0][0])[:, active], np.asarray(model[1][0])[active]))
    union = [(np.concatenate((models[p][0], np.conj(models[minus[p]][0])), axis=-1) / np.sqrt(2),
              np.concatenate((models[p][1], models[minus[p]][1]))) for p in range(3)]
    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh, spec))
    gemm = jax.jit(lambda x, y: x @ y, out_shardings=NamedSharding(mesh, P(None, "x", "y")))
    E, tau = .3, .7 + .2j

    def synth(parent_models):
        width = max(b.shape[-1] for b, _ in parent_models)
        width += width % 2
        f = np.stack([np.pad(b, ((0, 0), (0, width - b.shape[-1]))) for b, _ in parent_models])[:, :, None, :]
        p2 = np.stack([np.pad(p, (0, width - p.shape[-1]), constant_values=1.) for _, p in parent_models])
        bounds = np.asarray([[0, b.shape[-1]] for b, _ in parent_models], np.int32)
        plus, _ = jax.jit(lambda x, y, pp, r: synthesize_shared_pole_parents(x, y, pp, r, E, tau, mesh_xy=mesh, gemm=gemm))(
            put(f, P(None, "x", None, "y")), put(f, P(None, "y", None, "x")), put(p2, P()), put(bounds, P()))
        return plus

    hole = shared_pole_hole_kernel(mesh)
    mq_dev = put(np.asarray(minus, np.int32), P())
    plus, plus_u = synth(models), synth(union)
    for valence in (False, True):
        want = np.asarray(hole(plus_u, mq_dev) if valence else plus_u)
        got = np.asarray(shared_pole_even_part_kernel(mesh, exclude_q0=False)(plus, mq_dev, valence))
        assert rel(got, want) < 1e-12
        got_ex = np.asarray(shared_pole_even_part_kernel(mesh, exclude_q0=True)(plus, mq_dev, valence))
        own = np.asarray(hole(plus, mq_dev) if valence else plus)
        assert np.array_equal(got_ex[0], own[0]) and rel(got_ex[1:], want[1:]) < 1e-12
    assert rel(np.asarray(plus)[1], want[1]) > 1e-3


def test_debug_even_part_switch_refuses(monkeypatch):
    import pytest
    from gw.mpa.sigma import debug_shared_pole_even_part
    monkeypatch.delenv("LORRAX_DEBUG_SHARED_POLE_EVEN_PART", raising=False)
    assert debug_shared_pole_even_part(True) is None
    monkeypatch.setenv("LORRAX_DEBUG_SHARED_POLE_EVEN_PART", "bogus")
    with pytest.raises(ValueError, match="all or exclude_q0"):
        debug_shared_pole_even_part(True)
    monkeypatch.setenv("LORRAX_DEBUG_SHARED_POLE_EVEN_PART", "all")
    with pytest.raises(ValueError, match="time-reversal-symmetric"):
        debug_shared_pole_even_part(False)
    assert debug_shared_pole_even_part(True) == "all"
