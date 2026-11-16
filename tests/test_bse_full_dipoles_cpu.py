"""Full-BSE oscillator strengths (``full_bse = true`` with ``dipole_file``) on CPU, four host devices.

The payload is synthetic and time-reversal symmetric: k grid 2x2x2 (every k is its own
time-reversal image), 4v x 4c spinor bands in Kramers pairs (psi_{2j+1} = i sigma_y
conj(psi_{2j}) at every centroid, degenerate energies), N_mu 24, so the transition dimension
is 128.  V is real symmetric positive definite and W_R is real with W_R[M, N, R] =
W_R[N, M, -R], as W(r, r') = W(r', r) gives, so A is Hermitian and B complex symmetric (both
checked on the assembled blocks).  The dipoles are d_t = <c|O|v>/(E_c - E_v) of a time-odd Hermitian spin
operator O = sum_a f_a(r) sigma_a, so d_{Tc,Tv} = -conj(d_{c,v}) up to the Kramers signs.  A
and B are the production operator, ``bse_nontda._materialize_A_B`` over the fused pair
applier.  The contraction under test is
``<0|r|S> ~ sum_t X_t conj(d_t) - sum_t Y_t d_t`` (``absorption_common.exciton_dipole_projections``).

1. Coupling off: with B = 0 the dense full solve returns the TDA energies (1e-12 relative),
   Y = 0 and the TDA's strengths per multiplet (1e-10); the TDA eigenvectors with Y = 0
   through the full contraction give the TDA dipoles and spectrum exactly.  B = 0 with a
   complex A takes the pencil, not the real product route.
2. f-sum rule: with the local-field kernel alone and time reversal, sum_S Omega_S |d_S|^2 =
   sum_t dE_t |d_t|^2 over all N roots (1e-10 relative).  The TDA breaks it by exactly
   d^H K^x d, and so does the other sign of the Y term (the red twin).  With W on, the sum is
   d^H A d + Re(d^H B conj(d)), the first moment of the paired spectrum for any kernel.
3. Time-reversal partner roots: from a general eig of the full H (2N roots, no metric
   assumed), each root at +Omega and its partner at -Omega carry the same strength (1e-9 of
   the largest), and the pencil's roots carry the eig's strengths (1e-9).
4. The production route: ``solve_bse_nontda_sharded`` returns (X, Y) on the mesh's transition axes
   with X^H X - Y^H Y = 1, and ``exciton_dipoles_distributed`` on it equals the host
   contraction (1e-12).
5. A real H with a degenerate pair: ``solve_nontda_product`` returns Sigma-orthonormal pairs
   (1e-10) at the pencil's energies; ``_h_is_real`` takes it for a real H only.
6. ``parse_args`` accepts ``full_bse = true`` with ``dipole_file`` and builds the spectrum
   settings for it; Haydock still needs the TDA.

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_full_dipoles_cpu.py``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from bse import bse_jax  # noqa: E402
from bse.absorption_common import (build_dipole_vector_bse,  # noqa: E402
                                   eps2_from_exciton_dipoles,
                                   exciton_dipole_projections,
                                   exciton_dipoles_distributed)
from bse.bse_nontda import (_full_matvec_and_args, _h_is_real,  # noqa: E402
                            _materialize_A_B, solve_bse_nontda_sharded,
                            solve_nontda_definite_pencil, solve_nontda_product)
from bse.bse_ring_comm import make_bse_shardings  # noqa: E402

KGRID = (2, 2, 2)
NK = 8
NC = NV = 4
NS = 2
NMU = 24
N = NC * NV * NK
SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def _mesh():
    if jax.device_count() < 4:
        pytest.skip("four host devices required (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _kramers(rng, nb):
    """(nk, nb, NS, NMU) spinors with psi[:, 2j+1] = i sigma_y conj(psi[:, 2j]): (a, b) -> (b*, -a*)."""
    half = (rng.standard_normal((NK, nb // 2, NS, NMU))
            + 1j * rng.standard_normal((NK, nb // 2, NS, NMU))) / np.sqrt(NS * NMU)
    psi = np.empty((NK, nb, NS, NMU), dtype=np.complex128)
    psi[:, 0::2] = half
    psi[:, 1::2, 0] = np.conj(half[:, :, 1])
    psi[:, 1::2, 1] = -np.conj(half[:, :, 0])
    return psi


def _operands(seed=7):
    rng = np.random.default_rng(seed)
    pc, pv = _kramers(rng, NC), _kramers(rng, NV)
    ec = np.repeat(np.sort(0.40 + 0.08 * rng.random((NK, NC // 2)), axis=1), 2, axis=1)  # Ry
    ev = np.repeat(-np.sort(0.06 * rng.random((NK, NV // 2)), axis=1), 2, axis=1)
    # W(r, r') = W(r', r), real: W_R[M, N, R] = W_R[N, M, -R], so A is Hermitian and B symmetric
    W0 = 0.3 * rng.standard_normal((NMU, NMU) + KGRID)
    W_minus = np.roll(np.flip(W0, axis=(2, 3, 4)), 1, axis=(2, 3, 4))
    W_R = 0.5 * (W0 + np.transpose(W_minus, (1, 0, 2, 3, 4)))
    W_q = np.fft.fftn(W_R, axes=(2, 3, 4), norm="ortho")
    Vh = rng.standard_normal((NMU, NMU)) / NMU
    V = 2.0 * (Vh @ Vh.T) + 0.05 / NMU * np.eye(NMU)                # real symmetric PD
    f = rng.standard_normal((3, 3, NMU))                            # O_alpha(mu) = sum_a f sigma_a
    O = np.einsum("pam,ast->pmst", f, SIGMA)
    v = np.einsum("kcsm,pmst,kvtm->pkcv", np.conj(pc), O, pv)      # <c|O|v>, Hermitian, time-odd
    d_kcv = v / (ec[:, :, None] - ev[:, None, :])[None]             # (3, nk, nc, nv)
    return dict(pc=pc, pv=pv, ec=ec, ev=ev, W_q=W_q, V=V, d_kcv=d_kcv,
                d=np.transpose(d_kcv, (0, 2, 3, 1)))                # (3, nc, nv, nk): solver axes


def _payload(mesh, ops):
    from lxkit import device_put_process_local
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))  # noqa: E731
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(ops["pc"]), ops["pv"])
    return dict(
        psi_c_X=put(ops["pc"], P(None, None, None, "x")), psi_c_Y=put(ops["pc"], P(None, None, None, "y")),
        psi_v_X=put(ops["pv"], P(None, None, None, "x")), psi_v_Y=put(ops["pv"], P(None, None, None, "y")),
        eps_c=put(ops["ec"], P(None, None)), eps_v=put(ops["ev"], P(None, None)),
        W_q=put(ops["W_q"], P("x", "y", None, None, None)), V_q0=put(ops["V"], P("x", "y")),
        M=put(M, P(None, "x", "y", None)),
        nkx=KGRID[0], nky=KGRID[1], nkz=KGRID[2], n_cond=NC, n_val=NV,
        n_cond_pad=NC, n_val_pad=NV, n_rmu=NMU, densified=False)


def _dense(mesh, ops, include_W):
    """The production A and B (N x N, flat (c, v, k)) from the fused pair applier."""
    sh = make_bse_shardings(mesh)
    pair, args = _full_matvec_and_args(_payload(mesh, ops), mesh, sh, include_W=include_W)
    A, B = _materialize_A_B(pair, args, sh, NC, NV, NK)
    assert np.linalg.norm(A - A.conj().T) <= 1e-12 * np.linalg.norm(A)      # the physical structure
    assert np.linalg.norm(B - B.T) <= 1e-12 * np.linalg.norm(B)
    return A, B


def _dipoles(Z, d, sign=-1.0):
    """(n, 3) contraction of Z = [X; Y] (2N, n) columns; ``sign = +1`` is the red twin."""
    X = Z[:N].T.reshape(-1, NC, NV, NK)
    Y = Z[N:].T.reshape(-1, NC, NV, NK)
    if sign < 0:
        return exciton_dipole_projections(X, d, Y=Y)
    return exciton_dipole_projections(X, d) + exciton_dipole_projections(Y, np.conj(d))


def _groups(omega, tol=1e-9):
    """Index groups of (numerically) degenerate roots, in ascending order of omega."""
    order = np.argsort(omega)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(omega[i] - omega[cur[-1]]) < tol * max(1.0, abs(omega[i])):
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    return groups + [cur]


def _group_strength(Z, d, idx):
    """sum_alpha |o G^-1 o^H| of one eigenspace: o_i = contraction of z_i, G = Z^H Sigma Z.

    Basis-free, so it holds for a general eig's unnormalised, non-orthogonal vectors."""
    Zg = Z[:, idx]
    G = Zg[:N].conj().T @ Zg[:N] - Zg[N:].conj().T @ Zg[N:]
    o = _dipoles(Zg, d)                                              # (n_g, 3)
    return float(sum(abs(o[:, a] @ np.linalg.solve(G, o[:, a].conj())) for a in range(3)))


def _first_moment(omega, Z, d, sign=-1.0):
    return float(np.sum(omega * np.sum(np.abs(_dipoles(Z, d, sign)) ** 2, axis=1)))


def test_coupling_off_equals_tda():
    mesh = _mesh()
    ops = _operands()
    A, _ = _dense(mesh, ops, include_W=True)
    B0 = np.zeros_like(A)
    assert not _h_is_real(A, B0)                     # complex A, B = 0: the pencil
    omega, Z = solve_nontda_definite_pencil(A, B0, N)
    w, U = np.linalg.eigh(0.5 * (A + A.conj().T))
    assert np.max(np.abs(omega - w)) <= 1e-12 * np.max(np.abs(w))
    assert np.max(np.abs(Z[N:])) <= 1e-12
    Xt = U.T.reshape(N, NC, NV, NK)
    tda = exciton_dipole_projections(Xt, ops["d"])
    for g in _groups(w):
        s_full = _group_strength(Z, ops["d"], g)
        s_tda = float(np.sum(np.abs(tda[g]) ** 2))
        assert abs(s_full - s_tda) <= 1e-10 * max(1.0, s_tda)          # K^{-1/2} round-off
    # the same TDA states as a paired vector with Y = 0: the full path reproduces the TDA exactly
    full = exciton_dipole_projections(Xt, ops["d"], Y=np.zeros_like(Xt))
    assert np.array_equal(full, tda)
    om = np.linspace(0.3, 0.7, 41)
    args = (1e-3, 270.0, NK, 1, NS)
    assert np.array_equal(eps2_from_exciton_dipoles(om, w, full, *args),
                          eps2_from_exciton_dipoles(om, w, tda, *args))


def test_f_sum_rule_full_equation():
    mesh = _mesh()
    ops = _operands()
    d = ops["d"]
    dE = (ops["ec"].T[:, None, :] - ops["ev"].T[None, :, :]).reshape(-1)    # flat (c, v, k)
    window = float(np.sum(dE * np.sum(np.abs(d.reshape(3, -1)) ** 2, axis=0)))
    A, B = _dense(mesh, ops, include_W=False)                               # local fields only
    omega, Z = solve_nontda_definite_pencil(A, B, N)
    full = _first_moment(omega, Z, d)
    assert abs(full - window) <= 1e-10 * window, (full, window)
    # the TDA breaks the rule by d^H K^x d (K^x = A - D), the other sign by 2 Im-part terms
    w, U = np.linalg.eigh(0.5 * (A + A.conj().T))
    tda = float(np.sum(w * np.sum(np.abs(exciton_dipole_projections(
        U.T.reshape(N, NC, NV, NK), d)) ** 2, axis=1)))
    Kx = A - np.diag(dE)
    excess = sum(float(np.real(d[a].reshape(-1).conj() @ Kx @ d[a].reshape(-1))) for a in range(3))
    assert abs((tda - window) - excess) <= 1e-10 * window
    assert excess > 1e-3 * window
    twin = _first_moment(omega, Z, d, sign=+1.0)
    assert abs(twin - window) > 1e-3 * window
    # any kernel: the first moment of the paired spectrum is d^H A d + Re(d^H B conj(d))
    A, B = _dense(mesh, ops, include_W=True)
    omega, Z = solve_nontda_definite_pencil(A, B, N)
    moment = sum(float(np.real(d[a].reshape(-1).conj() @ A @ d[a].reshape(-1)
                               + d[a].reshape(-1).conj() @ B @ d[a].reshape(-1).conj()))
                 for a in range(3))
    assert abs(_first_moment(omega, Z, d) - moment) <= 1e-10 * abs(moment)


def test_time_reversal_partner_roots_carry_equal_strength():
    mesh = _mesh()
    ops = _operands()
    d = ops["d"]
    A, B = _dense(mesh, ops, include_W=True)
    H = np.block([[A, B], [-B.conj(), -A.conj()]])
    lam, Zg = np.linalg.eig(H)
    assert np.max(np.abs(lam.imag)) <= 1e-10 * np.max(np.abs(lam.real))
    lam = lam.real
    pos = [g for g in _groups(lam) if lam[g[0]] > 0]
    neg = [g for g in _groups(lam) if lam[g[0]] < 0]
    assert len(pos) == len(neg)
    neg_at = np.array([lam[g[0]] for g in neg])
    partner = [neg[int(np.argmin(np.abs(neg_at + lam[g[0]])))] for g in pos]
    assert max(abs(lam[g[0]] + lam[h[0]]) for g, h in zip(pos, partner)) <= 1e-10
    assert all(len(g) == len(h) for g, h in zip(pos, partner))
    s_pos = np.array([_group_strength(Zg, d, g) for g in pos])
    s_neg = np.array([_group_strength(Zg, d, h) for h in partner])
    assert np.max(np.abs(s_pos - s_neg)) <= 1e-9 * np.max(s_pos)
    omega, Z = solve_nontda_definite_pencil(A, B, N)
    s_pen = np.array([_group_strength(Z, d, g) for g in _groups(omega)])
    assert s_pen.shape == s_pos.shape and np.max(np.abs(s_pen - s_pos)) <= 1e-9 * np.max(s_pos)


def test_production_route_on_mesh():
    mesh = _mesh()
    ops = _operands()
    n_eig = 12
    omega, evecs, _ = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=n_eig)
    assert isinstance(evecs.sharding, NamedSharding)
    assert tuple(evecs.sharding.spec) == (None, None, "x", "y", None)     # Davidson's root layout
    host = np.asarray(evecs)
    X, Y = host[:, 0], host[:, 1]
    norm = (np.sum(np.abs(X.reshape(n_eig, -1)) ** 2, axis=1)
            - np.sum(np.abs(Y.reshape(n_eig, -1)) ** 2, axis=1))
    assert np.max(np.abs(norm - 1.0)) <= 1e-10
    want = exciton_dipole_projections(X, ops["d"], Y=Y)
    # the dipole block committed on the transition axes, as the driver's window dipoles are
    from lxkit import device_put_process_local
    d_block = device_put_process_local(build_dipole_vector_bse(ops["d_kcv"]),
                                       NamedSharding(mesh, P(None, "x", "y", None)))
    got = exciton_dipoles_distributed(evecs, d_block, n_eig)
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))


def test_real_product_route_multiplets():
    rng = np.random.default_rng(3)
    n = 24
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    e = np.sort(0.4 + 0.2 * rng.random(n))
    b = 0.01 * rng.standard_normal(n)
    e[3], b[3] = e[2], b[2]                           # one exactly degenerate pair of H
    A = (Q * e) @ Q.T
    B = (Q * b) @ Q.T
    S = rng.standard_normal((n, n))
    assert _h_is_real(A, B) and not _h_is_real(A + 1e-3j * (S - S.T), B)
    omega, Z = solve_nontda_product(A, B, n)
    om_p, _ = solve_nontda_definite_pencil(A.astype(complex), B.astype(complex), n)
    assert np.max(np.abs(omega - om_p)) <= 1e-12
    G = Z[:n].conj().T @ Z[:n] - Z[n:].conj().T @ Z[n:]
    assert np.max(np.abs(G - np.eye(n))) <= 1e-10
    H = np.block([[A, B], [-B, -A]])
    assert np.max(np.abs(H @ Z - Z * omega[None, :])) <= 1e-10


def _deck(tmp_path, lines):
    path = tmp_path / "full.in"
    path.write_text("[cohsex]\nsys_dim = 3\nnval = 5\nncond = 5\nnumber_bands = 100\n"
                    "centroids_file = centroids_frac.txt\n"
                    "number_val_bands_fine = 4\nnumber_cond_bands_fine = 4\n"
                    + "".join(f"{line}\n" for line in lines))
    return str(path)


def test_parse_accepts_full_bse_with_dipoles(tmp_path):
    _, args = bse_jax.parse_args(["-i", _deck(tmp_path, [
        "full_bse = true", "dipole_file = dipole.h5", "write_eigenvectors = -1"])])
    assert args.tda is False and args.dipole is not None
    assert bse_jax._spectrum_settings(args) is not None
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _deck(tmp_path, [
            "full_bse = true", "dipole_file = dipole.h5", "haydock", "number_iterations = 10"])])
