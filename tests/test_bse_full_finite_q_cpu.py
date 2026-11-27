"""The full BSE at a finite pair momentum, and the head tensor in the coupling block, on CPU.

Payload: synthetic, k grid 2x2x2, 4v x 4c spinor bands, N_mu 24, 128 transitions.  Four orbital
sets: conduction at k+Q and at k, valence at k and at k+Q (independent random arrays in the
general case; the same arrays at Q = 0).  ``W_R`` is real with ``W_R[M, N, R] = W_R[N, M, -R]``,
``V(Q)`` is complex Hermitian positive definite, the screened head is the exact diagonal shift and
the head tensor ``(D_head, M_head)`` is on in the cases that say so.

Reference: ``K = [[A, B], [B^H, A']]`` assembled in NumPy from the DEFINITION, with no line of
``src/``: the Galerkin matrix of the kernel ``D + K^x + K^head - K^d`` on the 2N pair functions

    resonant      phi_t(r, r') = psi_c(k+Q, r) conj(psi_v(k, r')),
    antiresonant  phi_t(r, r') = psi_v(k+Q, r) conj(psi_c(k, r')),

with ``K^d`` the multiplication by ``W`` (a k convolution on the pair's k label), ``K^x`` the
exchange through the pair densities ``phi_t(r, r)`` and ``V(Q)``, and ``K^head`` the same with
the dipoles (``d`` for a resonant pair, ``conj(d)`` for an antiresonant one) and ``M_head``.

1. The production blocks (``bse_nontda._materialize_K`` over the coupled applier and the two
   operand sets) equal the reference to 1e-12 on one device and on the emulated 2x2 mesh, with
   and without the head tensor; ``K`` is Hermitian, ``B`` has no symmetry, and the spectrum has
   no ``+-`` pairs.  Red twins: a partner with the two orbital sets exchanged, and an applier
   without the second exchange vertex, miss the reference.
2. The dense route and the Davidson route (``solve_bse_nontda_sharded`` with a partner) return
   the roots of positive type of ``numpy.linalg.eig(Sigma K)``, with ``X^H X - Y^H Y = 1``, on
   one device and the 2x2 mesh.  The Davidson's pair calls are printed.
3. ``Q = 0``: a partner equal to the payload's own orbitals reproduces the paired route's roots,
   and the head tensor in the paired route (dense and Davidson) matches the reference, whose
   ``B`` is complex symmetric with it.
4. A lower row that does not close the pencil refuses (``GATE nontda_lower_row``), and the
   matrix-free Lanczos refuses a partner.

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_full_finite_q_cpu.py -s``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from bse import bse_nontda  # noqa: E402
from bse.bse_ring_comm import make_bse_shardings  # noqa: E402
from tests import bse_full_oracle as orc  # noqa: E402

KGRID = (2, 2, 2)
NK = 8
NC = NV = 4
NS = 2
NMU = 24
N = NC * NV * NK
HEAD_RY = 0.013                       # the exact screened head, a diagonal shift (Ry)
RY_EV = 13.6056980659


def _mesh(side):
    if jax.device_count() < side * side:
        pytest.skip("four host devices required (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))


def _orbitals(rng, nb):
    return (rng.standard_normal((NK, nb, NS, NMU))
            + 1j * rng.standard_normal((NK, nb, NS, NMU))) / np.sqrt(NS * NMU)


def _operands(seed=3, *, q0=False):
    """Host operands.  ``q0``: the four orbital sets collapse to two and V is real symmetric."""
    rng = np.random.default_rng(seed)
    c_plus, v, c, v_plus = (_orbitals(rng, nb) for nb in (NC, NV, NC, NV))
    ec_plus = np.sort(0.40 + 0.10 * rng.random((NK, NC)), axis=1)
    ec = np.sort(0.42 + 0.10 * rng.random((NK, NC)), axis=1)
    ev = -np.sort(0.06 * rng.random((NK, NV)), axis=1)
    ev_plus = -np.sort(0.07 * rng.random((NK, NV)), axis=1)
    W0 = 0.3 * rng.standard_normal((NMU, NMU) + KGRID)
    W_minus = np.roll(np.flip(W0, axis=(2, 3, 4)), 1, axis=(2, 3, 4))
    W_R = 0.5 * (W0 + np.transpose(W_minus, (1, 0, 2, 3, 4)))        # real, W(r, r') = W(r', r)
    Vh = (rng.standard_normal((NMU, NMU)) + 1j * rng.standard_normal((NMU, NMU))) / NMU
    V = 2.0 * (Vh @ Vh.conj().T) + 0.05 / NMU * np.eye(NMU)          # complex Hermitian PD
    D_head = 0.05 * (rng.standard_normal((NK, NC, NV, 3)) + 1j * rng.standard_normal((NK, NC, NV, 3)))
    Mh = rng.standard_normal((3, 3))
    M_head = Mh @ Mh.T + 0.5 * np.eye(3)                              # real symmetric
    if q0:
        c, v_plus, ec, ev_plus = c_plus, v, ec_plus, ev
        V = np.real(V)
        V = 0.5 * (V + V.T)
    return dict(c_plus=c_plus, v=v, c=c, v_plus=v_plus, ec_plus=ec_plus, ev=ev, ec=ec,
                ev_plus=ev_plus, W_R=W_R, W_q=np.fft.fftn(W_R, axes=(2, 3, 4), norm="ortho"),
                V=V, D_head=D_head, M_head=M_head)


def _payload(mesh, ops, *, head_tensor=False, partner=True, swap_partner=False):
    from lxkit import device_put_process_local
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))  # noqa: E731
    x, y = P(None, None, None, "x"), P(None, None, None, "y")
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(ops["c_plus"]), ops["v"])
    data = dict(
        psi_c_X=put(ops["c_plus"], x), psi_c_Y=put(ops["c_plus"], y),
        psi_v_X=put(ops["v"], x), psi_v_Y=put(ops["v"], y),
        eps_c=put(ops["ec_plus"], P(None, None)), eps_v=put(ops["ev"], P(None, None)),
        W_q=put(ops["W_q"], P("x", "y", None, None, None)), V_q0=put(ops["V"], P("x", "y")),
        M=put(M, P(None, "x", "y", None)), q0_head="diagonal", q0_head_ry=HEAD_RY,
        nkx=KGRID[0], nky=KGRID[1], nkz=KGRID[2], n_cond=NC, n_val=NV,
        n_cond_pad=NC, n_val_pad=NV, n_rmu=NMU, densified=False)
    if head_tensor:
        data["head_tensor"] = (put(ops["D_head"], P()), put(ops["M_head"], P()))
    if partner:
        pc, pv = ("c_plus", "v") if swap_partner else ("c", "v_plus")
        data["partner"] = dict(
            psi_c_X=put(ops[pc], x), psi_c_Y=put(ops[pc], y),
            psi_v_X=put(ops[pv], x), psi_v_Y=put(ops[pv], y),
            eps_c=put(ops["ec"], P(None, None)), eps_v=put(ops["ev_plus"], P(None, None)))
    return data


# ---------------------------------------------------------------------------
# The reference: K from the definition (NumPy only)
# ---------------------------------------------------------------------------

def _k_index_difference():
    """``diff[k, k']`` = flat index of ``k - k'`` on the grid (C order, periodic)."""
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in KGRID], indexing="ij"), axis=-1).reshape(NK, 3)
    d = (grid[:, None, :] - grid[None, :, :]) % np.array(KGRID)
    return np.ravel_multi_index((d[..., 0], d[..., 1], d[..., 2]), KGRID)


def reference_K(ops, *, head_tensor=False, w_x=1.0):
    """``K`` (2N x 2N) on the pair functions, flat index (c, v, k) in C order, resonant first."""
    Wq = ops["W_q"].reshape(NMU, NMU, NK)
    diff = _k_index_difference()
    # pair functions F[k, c, v, t, mu, s, nu] at their own k
    F_res = np.einsum("kctm,kvsn->kcvtmsn", ops["c_plus"], np.conj(ops["v"]))
    F_anti = np.einsum("kvtm,kcsn->kcvtmsn", ops["v_plus"], np.conj(ops["c"]))
    F = np.stack([F_res, F_anti])                                    # (2, k, c, v, t, mu, s, nu)
    dens = np.einsum("pkcvsmsm->pkcvm", F)                           # phi_t(r, r): t = s, mu = nu
    d = np.conj(ops["D_head"])                                       # the dipole of a resonant pair
    g = np.stack([d, np.conj(d)])                                    # (2, k, c, v, 3)
    de = np.stack([(ops["ec_plus"] - HEAD_RY)[:, :, None] - ops["ev"][:, None, :],
                   (ops["ec"] - HEAD_RY)[:, :, None] - ops["ev_plus"][:, None, :]])   # (2, k, c, v)
    n_t = NC * NV
    K = np.zeros((2, NK, n_t, 2, NK, n_t), dtype=np.complex128)
    for p in range(2):
        for q in range(2):
            for k in range(NK):
                Fi = F[p, k].reshape(n_t, NS, NMU, NS, NMU)
                for kp in range(NK):
                    Fj = F[q, kp].reshape(n_t, NS, NMU, NS, NMU)
                    Kd = np.einsum("atmsn,mn,btmsn->ab", np.conj(Fi), Wq[:, :, diff[k, kp]], Fj,
                                   optimize=True) / NK
                    Kx = w_x / NK * np.conj(dens[p, k].reshape(n_t, NMU)) @ ops["V"] \
                        @ dens[q, kp].reshape(n_t, NMU).T
                    K[p, k, :, q, kp, :] = Kx - Kd
                    if head_tensor:
                        K[p, k, :, q, kp, :] += w_x / NK * np.conj(g[p, k].reshape(n_t, 3)) \
                            @ ops["M_head"] @ g[q, kp].reshape(n_t, 3).T
                if p == q:
                    K[p, k, :, p, k, :] += np.diag(de[p, k].reshape(n_t))
    # (p, k, cv) -> (p, c, v, k): the solver's flat order
    K = K.reshape(2, NK, NC, NV, 2, NK, NC, NV).transpose(0, 2, 3, 1, 4, 6, 7, 5)
    return K.reshape(2 * N, 2 * N)


def _production_K(mesh, ops, **kw):
    sh = make_bse_shardings(mesh)
    data = _payload(mesh, ops, **kw)
    pair, up, low = bse_nontda._operator(data, mesh, sh, include_W=True)
    A, B, B_low, A_low = bse_nontda._materialize_K(pair, up, low, sh, NC, NV, NK)
    return np.block([[A, B], [B_low, A_low]])


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------------------
# 1. the operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("head_tensor", [False, True])
def test_finite_q_blocks_match_the_definition(side, head_tensor):
    ops = _operands()
    ref = reference_K(ops, head_tensor=head_tensor)
    assert _rel(ref, ref.conj().T) <= 1e-13                          # the reference is Hermitian
    got = _production_K(_mesh(side), ops, head_tensor=head_tensor)
    err = _rel(got, ref)
    assert err <= 1e-12, err
    B = ref[:N, N:]
    assert np.linalg.norm(B - B.T) > 0.1 * np.linalg.norm(B)         # no B = B^T at finite Q
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    lam = np.linalg.eigvals(sig[:, None] * ref).real
    pos, neg = np.sort(lam[lam > 0]), np.sort(-lam[lam < 0])
    assert pos.size == neg.size == N and np.max(np.abs(pos - neg)) > 1e-3   # no +- pairs
    print(f"\nMEASURED finite-Q K, {side}x{side}, head tensor {head_tensor}: production against the "
          f"definition {err:.2e} (relative, max norm)")


def test_red_twins_miss_the_definition():
    ops = _operands()
    ref = reference_K(ops)
    mesh = _mesh(1)
    assert _rel(_production_K(mesh, ops, swap_partner=True), ref) > 1e-2
    # an applier that reads M where the antiresonant vertex belongs (the Q = 0 rule)
    sh = make_bse_shardings(mesh)
    data = _payload(mesh, ops)
    pair, up, low = bse_nontda._operator(data, mesh, sh, include_W=True)
    up_bad = up[:9] + (up[8],) + up[10:]
    A, B, B_low, A_low = bse_nontda._materialize_K(pair, up_bad, low, sh, NC, NV, NK)
    assert _rel(np.block([[A, B], [B_low, A_low]]), ref) > 1e-3
    with pytest.raises(ValueError, match="GATE nontda_lower_row"):
        bse_nontda.solve_nontda_definite_pencil(A, B, 4, lower=(B_low, A_low))


# ---------------------------------------------------------------------------
# 2. the two routes against eig
# ---------------------------------------------------------------------------

def _host_pairs(evecs, n):
    Z = np.asarray(evecs).reshape(n, 2, N)
    return Z[:, 0].T, Z[:, 1].T


@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("head_tensor", [False, True])
def test_finite_q_routes_match_eig(side, head_tensor):
    ops = _operands()
    mesh = _mesh(side)
    K = reference_K(ops, head_tensor=head_tensor)
    ref = orc.solve_general_pencil(K)
    assert ref["definite"]
    n_eig = 6
    om_d, ev_d, _ = bse_nontda.solve_bse_nontda_sharded(
        _payload(mesh, ops, head_tensor=head_tensor), mesh, n_eig=n_eig)
    om_d = np.asarray(om_d)
    assert np.max(np.abs(om_d - ref["omega"][:n_eig])) <= 1e-11
    X, Y = _host_pairs(ev_d, n_eig)
    assert np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(n_eig))) <= 1e-10
    assert np.max(orc.general_residuals(K, om_d, X, Y)) <= 1e-11

    om_v, ev_v, _ = bse_nontda.solve_bse_nontda_sharded(
        _payload(mesh, ops, head_tensor=head_tensor), mesh, n_eig=n_eig, solver="davidson")
    om_v = np.asarray(om_v)
    info = bse_nontda.LAST_DAVIDSON["info"]
    assert info["paired"] is False and info["status_name"] == "CONVERGED"
    err = float(np.max(np.abs(om_v - ref["omega"][:n_eig])))
    assert err <= 1e-9, err
    X, Y = _host_pairs(ev_v, n_eig)
    assert np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(n_eig))) <= 1e-8
    assert np.max(orc.general_residuals(K, om_v, X, Y)) <= 1e-5
    assert tuple(ev_v.sharding.spec) == (None, None, "x", "y", None)
    print(f"\nMEASURED finite-Q routes, {side}x{side}, head tensor {head_tensor}: dense against eig "
          f"{np.max(np.abs(om_d - ref['omega'][:n_eig])) * RY_EV:.1e} eV; Davidson {err * RY_EV:.1e} eV, "
          f"{info['iterations']} iterations, {info['pair_calls']} pair calls "
          f"({info['pair_calls_per_root']:.1f} per root), lam_min {info['lam_min']:.4f} Ry "
          f"(lam_min(K) = {ref['k_min']:.4f})")


# ---------------------------------------------------------------------------
# 3. Q = 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("head_tensor", [False, True])
def test_q0_partner_reproduces_the_paired_route(head_tensor):
    ops = _operands(q0=True)
    mesh = _mesh(2)
    K = reference_K(ops, head_tensor=head_tensor)
    A, B = K[:N, :N], K[:N, N:]
    orc.assert_bse_structure(A, B, tol=1e-12)                        # B = B^T with the head tensor too
    assert _rel(K[N:, N:], A.conj()) <= 1e-12
    ref = orc.reference_roots(A, B)
    n_eig = 6
    # the paired route (no partner): dense A, B and both solvers
    sh = make_bse_shardings(mesh)
    pair, args = bse_nontda._full_matvec_and_args(
        _payload(mesh, ops, head_tensor=head_tensor, partner=False), mesh, sh, include_W=True)
    A_p, B_p = bse_nontda._materialize_A_B(pair, args, sh, NC, NV, NK)
    assert _rel(A_p, A) <= 1e-12 and _rel(B_p, B) <= 1e-12 * np.max(np.abs(A)) / np.max(np.abs(B))
    roots = {}
    for label, partner, solver in (("paired dense", False, "dense"), ("paired davidson", False, "davidson"),
                                   ("partner dense", True, "dense"), ("partner davidson", True, "davidson")):
        om, _, _ = bse_nontda.solve_bse_nontda_sharded(
            _payload(mesh, ops, head_tensor=head_tensor, partner=partner), mesh, n_eig=n_eig, solver=solver)
        roots[label] = np.asarray(om)
        tol = 1e-11 if solver == "dense" else 1e-9
        assert np.max(np.abs(roots[label] - ref["omega"][:n_eig])) <= tol, label
        if solver == "davidson":
            assert bse_nontda.LAST_DAVIDSON["info"]["paired"] is (not partner)
            roots[label + " calls"] = bse_nontda.LAST_DAVIDSON["info"]["pair_calls"]
    print(f"\nMEASURED Q = 0, head tensor {head_tensor}: paired Davidson {roots['paired davidson calls']} "
          f"pair calls, unpaired (partner = own orbitals) {roots['partner davidson calls']}")


# ---------------------------------------------------------------------------
# 4. refusals
# ---------------------------------------------------------------------------

def test_matrix_free_lanczos_refuses_a_partner():
    mesh = _mesh(1)
    with pytest.raises(ValueError, match="GATE nontda_route_option"):
        bse_nontda.solve_bse_nontda_sharded(_payload(mesh, _operands()), mesh, n_eig=4,
                                            solver="matrixfree")
