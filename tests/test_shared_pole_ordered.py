"""Ordered (time-reversal-broken) shared-pole model on exact particle-hole plants.

Plants are Wc(z) = C (z s3 - M)^-1 C^H with M = diag(dE) + L L^H > 0. The
gates mirror the TRMODEL planted oracle (sandbox run 422_trmodel_20260915):
full-order exactness with real poles, the positive-pole carrier with its odd
term, projected z-moments m0..m3, equality with today's even construction on
time-reversal-symmetric data, generic-q assembly across the q/-q parents, and
Sigma's two synthesized orientations against the Lehmann sums.
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


def _states(plant, point, ordered=True):
    """(node, Q, WQ, action) for both orientations (ordered) and conjugate partners."""
    out = []
    for z in ((point, -np.conj(point)) if ordered else (point,)):
        W = plant.F(z)
        dW = plant.dF(z) if ordered else plant.dF(z) / (2 * z)
        node = z if ordered else z**2
        Q = adj(np.linalg.svd(W)[2])
        O = W @ Q
        out += [(node, Q, O, dW @ Q), (np.conj(node), O, adj(W) @ O, adj(dW) @ O)]
    return [(complex(n), _put(q), _put(o), _put(d)) for n, q, o, d in out]


def _ordered(plant, point):
    import jax.numpy as jnp
    from gw.shared_pole_constructor import (
        assemble_ordered_shared_pole_pencil, reduce_ordered_shared_pole_pencil)
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    mm, eigh = _ops()
    qi = np.linalg.eigh(plant.moment(1))[1][:, -1:]
    infinity = tuple(_put(a) for a in (qi, *(plant.moment(k) / 2 @ qi for k in range(4))))
    pencil = assemble_ordered_shared_pole_pencil(_states(plant, point), infinity, matmul=mm)
    active = jnp.ones(pencil[0].shape[:2], bool)
    model, signed, diag = reduce_ordered_shared_pole_pencil(
        pencil, active, eigh=eigh, matmul=mm, gates=gates)
    return model, signed, diag, infinity


def _signed_value(signed, z):
    c, mu, kept = (np.asarray(a[0]) for a in signed)
    return (c[:, kept] / (z * mu[kept] - 1)) @ adj(c[:, kept])


def test_ordered_full_order_exact_with_real_poles_and_paired_carrier():
    from gw.shared_pole_constructor import ordered_shared_pole_value, signed_shared_pole_passivity
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
    from gw.shared_pole_constructor import ordered_moment_identity
    mm, _ = _ops()
    plant = _trim(np.random.default_rng(7), 12, 3, eps=.4)
    _, signed, diag, infinity = _ordered(plant, .9 + .35j)
    assert int(diag["positive_count"][0]) + int(diag["negative_count"][0]) < 24
    rows = ordered_moment_identity(signed, infinity, matmul=mm)
    assert max(float(v[0]) for v in rows.values()) < 1e-12
    assert np.linalg.norm(plant.moment(0)) > 1e-3


def test_ordered_equals_even_construction_on_time_reversal_symmetric_data():
    import jax.numpy as jnp
    from gw.shared_pole_constructor import assemble_shared_pole_pencil, reduce_shared_pole_pencil
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


def test_generic_q_positive_halves_assemble_the_galerkin_model():
    from gw.shared_pole_constructor import ordered_shared_pole_value
    mm, _ = _ops()
    q, mq = _generic_pair(np.random.default_rng(13), 12, 3)
    for z in ZS:
        assert rel(mq.F(z), q.F(-z).T) < 1e-13
    model_q, signed_q, _, _ = _ordered(q, .9 + .35j)
    model_m, _, _, _ = _ordered(mq, .9 + .35j)
    for z in ZS:
        stored = np.asarray(ordered_shared_pole_value(model_q, model_m, z, matmul=mm)[0])
        assert rel(stored, _signed_value(signed_q, z)) < 1e-12


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
        shared_pole_hole_kernel, shared_pole_minus_q_index, synthesize_shared_pole_parents)
    require_devices(4, "cpu")
    mesh = Mesh(np.asarray(jax.devices("cpu")[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(29)
    gamma = _trim(rng, 6, 4, eps=.4)
    q, mq = _generic_pair(rng, 6, 4)
    plants = (gamma, q, mq)
    minus_q = shared_pole_minus_q_index((3, 1, 1))
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
