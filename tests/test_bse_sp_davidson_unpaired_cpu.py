"""The unpaired form of the structure-preserving Davidson (``sp_davidson(paired=False)``) against a
dense ``eig``: the general pencil ``K z = omega Sigma z`` with ``K = [[A, B], [B^H, A_low]]`` and no
conjugation symmetry, which is the full BSE at a finite pair momentum.

The solver sees only ``K`` on pair rows ``[X; Y]``; here ``K`` is dense, 240 transitions (window
``(4, 4, 15)``, energies in Ry).  Reference: ``numpy.linalg.eig`` of ``Sigma K`` with every root
classified by type (``tests/bse_full_oracle.solve_general_pencil``), and the closed-form roots of
rotated direct sums of 2 x 2 pencils.

1. Definite pencils, one device and the emulated 2x2 mesh: the lowest positive-type roots within
   1e-10 eV, ``X_i^H X_j - Y_i^H Y_j = delta_ij``, residuals of ``Sigma K z = omega z``, the reported
   ``lam_min`` at or above ``lam_min(K)``.  The spectrum has no ``+-`` pairs and the test says so.
2. Exactly degenerate roots (multiplets cut and uncut by the block) and a restart under a small
   capacity give the same roots.
3. At a symmetric pencil (``A_low = conj(A)``, ``B = B^T``) the unpaired form returns the paired
   form's roots; the pair calls of both are printed.
4. ``K`` not positive definite: the default refuses (``GATE sp_davidson_definite``);
   ``on_indefinite='continue'`` returns a complex root, a real root of positive type at negative
   energy and the stable roots above them, and keeps a negative-type root at positive energy out of
   the returned block.  Red twin: the definite pencil does not refuse.
5. Operator integrity: a ``K`` that is not Hermitian refuses (``GATE sp_davidson_operator``).

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_sp_davidson_unpaired_cpu.py -s``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from lxkit import device_put_process_local  # noqa: E402
from solvers import bse_sp_davidson as spd  # noqa: E402
from tests import bse_full_oracle as orc  # noqa: E402

N = 240
WINDOW = (4, 4, N // 16)
RY_EV = 13.6056980659
ROOT_TOL_EV = 1e-10
N_EIG, MARGIN = 6, 4
TOL = 1e-9


def _mesh(side):
    if jax.device_count() < side * side:
        pytest.skip("four host devices required (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))


def _apply(data, Z):
    """``K Z`` on pair rows ``(b, 2, *WINDOW)``; ``data[0] = K`` replicated."""
    flat = Z.reshape(Z.shape[0], -1)
    return (flat @ data[0].T).reshape(Z.shape)


def _start(n_block, seed=0, diag=None):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_block, N)) + 1j * rng.standard_normal((n_block, N))
    if diag is not None:
        low = np.argsort(diag)[:n_block - 2]
        X0[:n_block - 2] = 0.0
        X0[np.arange(n_block - 2), low] = 1.0
    return X0


def _solve(K, mesh, *, n_eig=N_EIG, margin=MARGIN, precond=False, **kw):
    """One unpaired solve.  Returns ``omega``, host ``X``, ``Y`` as (N, n_block) columns, ``info``."""
    n_block = n_eig + margin
    d_up, d_low = np.real(np.diag(K))[:N], np.real(np.diag(K))[N:]
    X0 = _start(n_block, diag=d_up if precond else None).reshape((n_block,) + WINDOW)
    if mesh is None:
        Kd, Du, Dl, X0d = (jnp.asarray(a) for a in (K, d_up.reshape(WINDOW), d_low.reshape(WINDOW), X0))
    else:
        rep, dsh = NamedSharding(mesh, P()), NamedSharding(mesh, P("x", "y", None))
        Kd = device_put_process_local(K, rep)
        Du = device_put_process_local(d_up.reshape(WINDOW), dsh)
        Dl = device_put_process_local(d_low.reshape(WINDOW), dsh)
        X0d = device_put_process_local(X0, NamedSharding(mesh, P(None, "x", "y", None)))
    divide = spd.sp_diagonal_precond(epsilon_shift=1e-3)
    kw.setdefault("tol", TOL)
    kw.setdefault("max_iter", 400)
    kw.setdefault("verbose", False)
    omega, X, Y, info = spd.sp_davidson(
        _apply, X0d, data=(Kd, Du, Dl), n_converge=n_eig, paired=False,
        precondition=(lambda d, RX, RY, om: divide((d[1], d[2]), RX, RY, om)) if precond else None, **kw)
    if mesh is not None:
        assert tuple(X.sharding.spec) == (None, "x", "y", None)
        assert tuple(Y.sharding.spec) == (None, "x", "y", None)
    return omega, np.asarray(X).reshape(n_block, N).T, np.asarray(Y).reshape(n_block, N).T, info


def _check(K, ref, omega, X, Y, *, n_eig=N_EIG, first=0, label=""):
    sel = slice(first, first + n_eig)
    om = np.real(np.asarray(omega)[sel])
    err_ev = float(np.max(np.abs(om - ref["omega"][:n_eig])) * RY_EV)
    assert err_ev <= ROOT_TOL_EV, (label, err_ev)
    gram = float(np.max(np.abs(orc.sigma_gram(X[:, sel], Y[:, sel]) - np.eye(n_eig))))
    assert gram <= 1e-10, (label, gram)
    resid = float(np.max(orc.general_residuals(K, om, X[:, sel], Y[:, sel])))
    assert resid <= 1e-7, (label, resid)
    return err_ev, gram, resid


@pytest.fixture(scope="module")
def dominant():
    A, B, A_low = orc.dominant_general_pencil(n=N, seed=5)
    K = orc.general_K(A, B, A_low)
    ref = orc.solve_general_pencil(K)
    assert ref["definite"] and len(ref["omega"]) == N
    return K, ref


# ---------------------------------------------------------------------------
# 1. definite pencils
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", [None, 2])
def test_definite_general_pencil_matches_dense_eig(dominant, side):
    K, ref = dominant
    mesh = None if side is None else _mesh(side)
    omega, X, Y, info = _solve(K, mesh, precond=True)
    assert info["status_name"] == "CONVERGED" and info["paired"] is False
    err_ev, gram, resid = _check(K, ref, omega, X, Y, label="dominant")
    assert info["lam_min"] >= ref["k_min"] - 1e-12
    # No +- pairs: the negative roots are not the mirror of the positive ones.
    lam = np.linalg.eigvals(np.concatenate([np.ones(N), -np.ones(N)])[:, None] * K).real
    neg = np.sort(-lam[lam < 0])
    assert np.max(np.abs(neg[:N_EIG] - ref["omega"][:N_EIG])) > 1e-3
    print(f"\nMEASURED unpaired dominant ({'one device' if side is None else '2x2'}): root error "
          f"{err_ev:.2e} eV, Sigma-gram {gram:.1e}, residual {resid:.1e}; {info['iterations']} "
          f"iterations, {info['pair_calls']} pair calls ({info['pair_calls_per_root']:.1f} per root), "
          f"{info['restarts']} restarts")


def test_mode_pencil_closed_form_roots():
    rng = np.random.default_rng(11)
    a = 0.4 + 0.6 * np.sort(rng.random(N))
    a_low = 0.5 + 0.6 * rng.random(N)
    b = 0.05 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    A, B, A_low = orc.general_mode_pencil(a, a_low, b, seed=12)
    K = orc.general_K(A, B, A_low)
    plus, _ = orc.mode_roots(a, a_low, b)
    exact = np.sort(plus.real)
    omega, X, Y, info = _solve(K, None)
    assert info["status_name"] == "CONVERGED"
    err = float(np.max(np.abs(np.real(omega[:N_EIG]) - exact[:N_EIG])) * RY_EV)
    assert err <= ROOT_TOL_EV, err
    _check(K, orc.solve_general_pencil(K), omega, X, Y, label="modes")
    print(f"\nMEASURED unpaired rotated modes: {err:.2e} eV from the closed form, "
          f"{info['pair_calls']} pair calls")


# ---------------------------------------------------------------------------
# 2. degenerate roots, restart
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_eig,margin", [(4, 4), (6, 6)])
def test_degenerate_multiplets(n_eig, margin):
    """Fourfold multiplets; ``n_eig = 6`` cuts the second one, the block holds whole multiplets
    (a block that cuts one stalls, flagged, in both forms of the solver)."""
    mult, groups = 4, N // 4
    rng = np.random.default_rng(21)
    ag = 0.4 + 0.1 * np.arange(groups) + 0.02 * rng.random(groups)
    al = 0.45 + 0.1 * np.arange(groups) + 0.02 * rng.random(groups)
    bg = 0.03 * (rng.standard_normal(groups) + 1j * rng.standard_normal(groups))
    A, B, A_low = orc.general_mode_pencil(np.repeat(ag, mult), np.repeat(al, mult), np.repeat(bg, mult),
                                          seed=22)
    K = orc.general_K(A, B, A_low)
    exact = np.sort(orc.mode_roots(np.repeat(ag, mult), np.repeat(al, mult), np.repeat(bg, mult))[0].real)
    omega, X, Y, info = _solve(K, None, n_eig=n_eig, margin=margin)
    assert info["status_name"] == "CONVERGED"
    assert np.max(np.abs(np.real(omega[:n_eig]) - exact[:n_eig])) * RY_EV <= ROOT_TOL_EV
    gram = np.max(np.abs(orc.sigma_gram(X[:, :n_eig], Y[:, :n_eig]) - np.eye(n_eig)))
    assert gram <= 1e-10, gram


def test_restart_under_small_capacity(dominant):
    """The default capacity (the six-block floor here, 60 pair vectors) restarts; a basis that
    holds the whole solve does not; the roots are the same."""
    K, ref = dominant
    full = _solve(K, None, precond=True, capacity=800)
    small = _solve(K, None, precond=True)
    assert full[3]["capacity"] == 400 and small[3]["capacity"] == 60
    assert small[3]["status_name"] == "CONVERGED" and small[3]["restarts"] > full[3]["restarts"] == 0
    _check(K, ref, small[0], small[1], small[2], label="restart")
    _check(K, ref, full[0], full[1], full[2], label="no restart")
    print(f"\nMEASURED unpaired restart: capacity {small[3]['capacity']} -> {small[3]['restarts']} "
          f"restarts, {small[3]['pair_calls']} pair calls (capacity {full[3]['capacity']}: "
          f"{full[3]['restarts']}, {full[3]['pair_calls']})")


# ---------------------------------------------------------------------------
# 3. a symmetric pencil: unpaired == paired
# ---------------------------------------------------------------------------

def test_symmetric_pencil_unpaired_equals_paired():
    p = orc.dominant_pencil(n=N)
    A, B = p["A"], p["B"]
    K = orc.general_K(A, B, A.conj())
    ref = orc.reference_roots(A, B)
    omega, X, Y, info = _solve(K, None, precond=True)
    assert info["status_name"] == "CONVERGED"
    assert np.max(np.abs(np.real(omega[:N_EIG]) - ref["omega"][:N_EIG])) * RY_EV <= ROOT_TOL_EV

    def pair(data, Xr, s):
        flat = Xr.reshape(Xr.shape[0], -1)
        return (flat @ data[0].T + s[:, None] * (jnp.conj(flat) @ data[1].T)).reshape(Xr.shape)

    n_block = N_EIG + MARGIN
    d = np.real(np.diag(A))
    X0 = jnp.asarray(_start(n_block, diag=d).reshape((n_block,) + WINDOW))
    divide = spd.sp_diagonal_precond(epsilon_shift=1e-3)
    om_p, _, _, info_p = spd.sp_davidson(
        pair, X0, data=(jnp.asarray(A), jnp.asarray(B), jnp.asarray(d.reshape(WINDOW))),
        n_converge=N_EIG, tol=TOL, max_iter=400, verbose=False,
        precondition=lambda dd, RX, RY, om: divide(dd[2], RX, RY, om))
    assert np.max(np.abs(np.real(om_p[:N_EIG]) - np.real(omega[:N_EIG]))) * RY_EV <= ROOT_TOL_EV
    print(f"\nMEASURED symmetric pencil, 6 roots: unpaired {info['iterations']} iterations, "
          f"{info['pair_calls']} pair calls; paired {info_p['iterations']} iterations, "
          f"{info_p['pair_calls']} pair calls (ratio {info['pair_calls'] / info_p['pair_calls']:.2f})")


# ---------------------------------------------------------------------------
# 4. K not positive definite
# ---------------------------------------------------------------------------

def _indefinite():
    """Modes 0..2 are special: a complex pair, a positive-type root at omega < 0, a negative-type
    root at omega > 0; the rest are stable."""
    rng = np.random.default_rng(31)
    a = 0.4 + 0.6 * np.sort(rng.random(N))
    a_low = 0.45 + 0.6 * rng.random(N)
    b = 0.04 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    a[0], a_low[0], b[0] = 0.05, 0.07, 0.10          # |b| > (a + a')/2: complex pair
    a[1], a_low[1], b[1] = 0.02, 0.12, 0.06          # a a' < |b|^2 < ((a+a')/2)^2, a < a'
    a[2], a_low[2], b[2] = 0.12, 0.02, 0.06          # the same with a > a': negative type, omega > 0
    A, B, A_low = orc.general_mode_pencil(a, a_low, b, seed=32)
    plus, minus = orc.mode_roots(a, a_low, b)
    return orc.general_K(A, B, A_low), plus, minus


def test_indefinite_refuses_by_default(dominant):
    K, _, _ = _indefinite()
    with pytest.raises(ValueError, match="GATE sp_davidson_definite"):
        _solve(K, None)
    omega, *_rest, info = _solve(dominant[0], None, precond=True)      # red twin
    assert info["status_name"] == "CONVERGED" and info["indefinite_at"] is None


def test_indefinite_continue_separates_special_roots():
    K, plus, minus = _indefinite()
    ref = orc.solve_general_pencil(K)
    assert not ref["definite"]
    assert len(ref["complex"]) == 1 and len(ref["negative"]) == 1 and len(ref["mirror"]) == 1
    assert abs(plus[1].real - ref["negative"][0]) < 1e-12 and plus[1].real < 0
    assert abs(minus[2].real - ref["mirror"][0]) < 1e-12 and minus[2].real > 0
    omega, X, Y, info = _solve(K, None, on_indefinite="continue", margin=6)
    assert info["status_name"] == "CONVERGED"
    assert info["n_unstable"] == 1 and info["n_negative"] == 1
    assert abs(omega[0] - ref["complex"][0]) * RY_EV <= 1e-9
    assert abs(omega[1] - ref["negative"][0]) * RY_EV <= 1e-9
    stable = np.real(omega[2:2 + N_EIG])
    assert np.max(np.abs(stable - ref["omega"][:N_EIG])) * RY_EV <= ROOT_TOL_EV
    # The negative-type root at omega > 0 lies below the stable ones and is not returned.
    assert ref["mirror"][0] < ref["omega"][0]
    assert np.min(np.abs(np.real(omega) - ref["mirror"][0])) > 1e-6
    gram = np.max(np.abs(orc.sigma_gram(X[:, 2:2 + N_EIG], Y[:, 2:2 + N_EIG]) - np.eye(N_EIG)))
    assert gram <= 1e-9, gram
    print(f"\nMEASURED unpaired continue: complex root {omega[0]:.6f}, negative {omega[1].real:.6f} Ry, "
          f"stable error {np.max(np.abs(stable - ref['omega'][:N_EIG])) * RY_EV:.1e} eV, "
          f"{info['pair_calls']} pair calls")


# ---------------------------------------------------------------------------
# 5. operator integrity
# ---------------------------------------------------------------------------

def test_non_hermitian_K_refuses(dominant):
    K = dominant[0].copy()
    rng = np.random.default_rng(41)
    K[:N, N:] += 1e-3 * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))   # B only
    with pytest.raises(ValueError, match="GATE sp_davidson_operator"):
        _solve(K, None, precond=True)
