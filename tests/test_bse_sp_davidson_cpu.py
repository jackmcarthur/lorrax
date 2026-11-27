"""The structure-preserving Davidson (``solvers.bse_sp_davidson``) against the dense oracle.

The solver sees only a pair applier ``pair(X, s) = A X + s B conj(X)``; here ``A`` and ``B`` are the
synthetic BSE-structured pencils of ``tests/bse_full_oracle.py`` at 240 transitions (energies in Ry,
window ``(4, 4, 15)``), applied densely.  The pencils are rotated by a random unitary, so their diagonal
carries nothing and the Davidson runs as a restarted block Krylov method; the dominant pencil is not
rotated and uses the paired diagonal preconditioner.  Reference: the oracle's Cholesky solve for a
positive definite ``K = [[A, B], [B*, A*]]``, ``numpy.linalg.eig`` of ``H`` otherwise.

1. Roots within 1e-10 eV of the oracle on the dominant, nearly singular, exactly degenerate and clustered
   pencils, on one device and on the emulated 2x2 mesh (vectors sharded ``P(None, 'x', 'y', None)``): the
   pairs are Sigma-orthonormal (``X_i^H X_j - Y_i^H Y_j = delta_ij``), solve ``H z = omega z``, span the
   oracle's eigenspace in every multiplet, and ``lam_min`` is at or above ``lam_min(K)``.  An unsharded
   start block takes the one-device plan and gives the same roots.
2. Soft modes (``lam_min(K)`` = 1e-6, 1e-9, 1e-11 Ry): the soft root and the roots above it, and the
   stability margin the solve reports.
3. ``K`` not positive definite.  The default refuses (``GATE sp_davidson_definite``) with the measured
   ``lam_min`` and the option that continues; ``on_indefinite='continue'`` returns the imaginary root of
   the oracle's indefinite pencil and the real roots above it.  The same for a real root of positive type
   at negative energy, a complex quadruplet and two imaginary pairs.  Red twin: the definite pencils do
   not refuse.
4. Operator integrity: a coupling block that is not symmetric, or a resonant block that is not Hermitian,
   refuses (``GATE sp_davidson_operator``); the correct operator measures round-off.
5. The Tamm-Dancoff features: margin roots (the stop reads the lowest ``n_converge``), soft locking and
   the looser margin lock (fewer pair calls, same roots), restart under a capacity (the same roots from a
   third of the basis), the stall exit and the iteration cap as returned statuses that are not CONVERGED,
   and the refusals of a rank-deficient start block, a non-finite operator and bad arguments.
6. Accounting: the applier is called with ``s = +1`` only, the rows it receives add up to
   ``info['pair_calls']``, of which ``2 n_block`` are the start block; a second solve under the same
   ``plan_key`` reuses the kernels (no new trace of the applier).  The optimized HLO of every kernel
   on the 2x2 mesh holds its collective table (coefficient reductions only) and no array larger than
   a rank's tile of the basis.
7. The projected solves: the Cholesky route against a general eigensolve of the same pencil, the
   unstable sector's separation, and rank 0's host results reaching every rank through one all-reduce
   (two ranks emulated).

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_sp_davidson_cpu.py -s``; about a minute on two threads.
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
ROOT_TOL_EV = 1e-10          # the brief's bar on every root against the oracle
N_EIG, MARGIN = 6, 4
TOL = 1e-9                   # residual stop, Ry: the eigenvalue error is second order in it


def _mesh(side):
    if jax.device_count() < side * side:
        pytest.skip("four host devices required (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))


def _pair(data, X, s):
    """``A X + s B conj(X)`` per row on the solver's layout; ``data = (A, B, ...)`` replicated."""
    flat = X.reshape(X.shape[0], -1)
    return (flat @ data[0].T + s[:, None] * (jnp.conj(flat) @ data[1].T)).reshape(X.shape)


def _start(n_block, seed=0, diag=None):
    """Host start block (n_block, N): unit vectors at the lowest diagonal entries when ``diag`` is
    given (the BSE route's kind of start), complex Gaussians otherwise."""
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n_block, N)) + 1j * rng.standard_normal((n_block, N))
    if diag is not None:
        low = np.argsort(diag)[:n_block - 2]
        X0[:n_block - 2] = 0.0
        X0[np.arange(n_block - 2), low] = 1.0
    return X0


def _solve(A, B, mesh, *, n_eig=N_EIG, margin=MARGIN, precond=False, X0=None, pair=_pair, **kw):
    """One solve on ``mesh`` (None: an unsharded start block).  Returns ``omega``, host ``X``, ``Y``
    as (N, n_block) columns, and ``info``."""
    n_block = n_eig + margin
    diag = np.real(np.diag(A))
    X0 = _start(n_block, diag=diag if precond else None) if X0 is None else X0
    X0 = X0.reshape((n_block,) + WINDOW)
    if mesh is None:
        put = jnp.asarray
        Ad, Bd, Dd, X0d = put(A), put(B), put(diag.reshape(WINDOW)), put(X0)
    else:
        rep = NamedSharding(mesh, P())
        Ad, Bd = device_put_process_local(A, rep), device_put_process_local(B, rep)
        Dd = device_put_process_local(diag.reshape(WINDOW), NamedSharding(mesh, P("x", "y", None)))
        X0d = device_put_process_local(X0, NamedSharding(mesh, P(None, "x", "y", None)))
    divide = spd.sp_diagonal_precond(epsilon_shift=1e-3)
    kw.setdefault("tol", TOL)
    kw.setdefault("max_iter", 400)
    kw.setdefault("verbose", False)
    omega, X, Y, info = spd.sp_davidson(
        pair, X0d, data=(Ad, Bd, Dd), n_converge=n_eig,
        precondition=(lambda d, RX, RY, om: divide(d[2], RX, RY, om)) if precond else None, **kw)
    if mesh is not None:
        assert isinstance(X.sharding, NamedSharding) and tuple(X.sharding.spec) == (None, "x", "y", None)
        assert tuple(Y.sharding.spec) == (None, "x", "y", None)
    return omega, np.asarray(X).reshape(n_block, N).T, np.asarray(Y).reshape(n_block, N).T, info


@pytest.fixture(scope="module")
def pencils():
    """The oracle's five pencils at N = 240, with their reference roots."""
    made = [orc.dominant_pencil(n=N), orc.near_singular_pencil(n=N, eps=1e-4),
            orc.degenerate_pencil(n_groups=N // 4, mult=4), orc.clustered_pencil(n_clusters=N // 4, size=4),
            orc.indefinite_pencil(n=N)]
    out = {}
    for p in made:
        orc.assert_bse_structure(p["A"], p["B"], tol=1e-13)
        p["ref"] = orc.reference_roots(p["A"], p["B"])
        out[p["name"]] = p
    return out


def _check_pairs(p, omega, X, Y, info, *, n_eig=N_EIG, first=0):
    """Roots, normalisation, residuals and multiplet eigenspaces of roots ``[first, first + n_eig)``."""
    A, B, ref = p["A"], p["B"], p["ref"]
    sel = slice(first, first + n_eig)
    om = np.real(np.asarray(omega)[sel])
    err_ev = np.max(np.abs(om - ref["omega"][:n_eig])) * RY_EV
    assert err_ev <= ROOT_TOL_EV, (p["name"], err_ev)
    Xs, Ys = X[:, sel], Y[:, sel]
    gram = np.max(np.abs(orc.sigma_gram(Xs, Ys) - np.eye(n_eig)))
    assert gram <= 1e-10, (p["name"], gram)
    resid = np.max(orc.eigpair_residuals(A, B, om, Xs, Ys))
    assert resid <= 1e-7, (p["name"], resid)            # the stop is 1e-9 Ry on ||H z - omega z|| / ||z||
    for g in orc.degenerate_groups(om, rtol=1e-5):
        whole = [h for h in orc.degenerate_groups(ref["omega"], rtol=1e-5) if h[0] == g[0]][0]
        if len(whole) != len(g):
            continue                                     # a multiplet cut by n_eig: a partial eigenspace
        d = orc.sigma_projector_distance(Xs[:, g], Ys[:, g], ref["X"][:, g], ref["Y"][:, g])
        assert d <= 1e-5, (p["name"], g, d)
    return err_ev, gram, resid


# ---------------------------------------------------------------------------
# 1. definite pencils against the oracle, one device and the 2x2 mesh
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", (1, 2))
@pytest.mark.parametrize("name", ("dominant", "near_singular", "degenerate", "clustered"))
def test_definite_pencils_match_the_oracle(pencils, name, side):
    p = pencils[name]
    omega, X, Y, info = _solve(p["A"], p["B"], _mesh(side), precond=(name == "dominant"), capacity=N)
    assert info["status"] == spd.CONVERGED, info["status_name"]
    assert omega.dtype == np.float64 and np.all(np.diff(omega[:N_EIG]) >= 0)
    err_ev, gram, resid = _check_pairs(p, omega, X, Y, info)
    assert info["n_unstable"] == 0 and info["indefinite_at"] is None
    assert info["lam_min"] >= p["k_min"] * (1 - 1e-9) and info["sigma_err"] <= 1e-10
    assert info["metric_asym"] <= 1e-12 and info["orth_err"] <= 1e-10
    if name in ("degenerate", "clustered"):
        # the lowest multiplet (4 roots) is returned whole: the block holds a cut one in its margin
        assert np.max(np.abs(omega[:4] - omega[0])) <= (1e-12 if name == "degenerate" else 1e-6)
    print(f"MEASURED {name} {side}x{side}: max|omega - oracle| = {err_ev:.2e} eV, Sigma Gram {gram:.1e}, "
          f"residual {resid:.1e}, {info['iterations']} iterations, {info['pair_calls']} pair calls "
          f"({info['pair_calls_per_root']:.1f} per root), {info['restarts']} restarts, "
          f"lam_min {info['lam_min']:.6e} (K: {p['k_min']:.6e})", flush=True)


def test_unsharded_start_block_takes_the_one_device_plan(pencils):
    p = pencils["dominant"]
    om_plain, X, Y, info = _solve(p["A"], p["B"], None, precond=True)
    om_mesh, *_ = _solve(p["A"], p["B"], _mesh(1), precond=True)
    assert info["status"] == spd.CONVERGED
    _check_pairs(p, om_plain, X, Y, info)
    assert np.max(np.abs(om_plain - om_mesh)[:N_EIG]) <= 1e-12


# ---------------------------------------------------------------------------
# 2. soft modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k_min", (1e-6, 1e-9, 1e-11))
def test_soft_mode_and_stability_margin(k_min):
    p = orc.near_singular_pencil(n=N, eps=k_min)
    omega, X, Y, info = _solve(p["A"], p["B"], _mesh(2), capacity=N)
    assert info["status"] == spd.CONVERGED
    err = np.abs(omega[:N_EIG] - p["omega_exact"][:N_EIG])
    y2 = float(np.sum(np.abs(Y[:, 0]) ** 2))
    print(f"MEASURED soft mode lam_min(K) = {k_min:g} Ry: omega_1 = {omega[0]:.6e} Ry, error {err[0]:.2e} Ry, "
          f"roots above {err[1:].max():.2e} Ry, |Y_1|^2 = {y2:.3e}, reported lam_min {info['lam_min']:.6e} Ry, "
          f"Sigma error {info['sigma_err']:.1e}", flush=True)
    assert np.max(err[1:]) * RY_EV <= ROOT_TOL_EV
    # the soft root itself: its absolute accuracy degrades like 1/omega_1 (omega^2 is what is conditioned)
    assert err[0] <= 1e-9
    assert abs(info["lam_min"] - k_min) <= 1e-4 * k_min + 1e-15       # the margin, with no extra pair call
    assert info["lam_min"] > info["pd_floor"] and info["indefinite_at"] is None
    assert info["sigma_err"] <= 1e-9 and y2 > 10.0


# ---------------------------------------------------------------------------
# 3. K not positive definite
# ---------------------------------------------------------------------------

def test_indefinite_refuses_by_default_with_the_measured_margin(pencils):
    p = pencils["indefinite"]
    with pytest.raises(ValueError, match="GATE sp_davidson_definite") as exc:
        _solve(p["A"], p["B"], _mesh(2), capacity=N,
               refusal_notes={"definite": "CALLER NOTE: energies are in Ry."})
    msg = str(exc.value)
    import re
    lam = float(re.search(r"lam_min = (-?[0-9.]+e[+-]\d+)", msg).group(1))
    assert p["k_min"] * (1 + 1e-6) <= lam <= 0.0          # a certificate: lam_min(K) <= lam_min(M_r) <= 0
    assert "got:" in msg and "want:" in msg and "Why:" in msg
    assert "on_indefinite='continue'" in msg and "CALLER NOTE: energies are in Ry." in msg
    # red twin: a positive definite K does not refuse under the same default (test 1 ran those)
    assert pencils["dominant"]["definite"] and pencils["indefinite"]["k_min"] < 0


@pytest.mark.parametrize("side", (1, 2))
def test_indefinite_continued_solve_returns_the_imaginary_root(pencils, side):
    p = pencils["indefinite"]
    omega, X, Y, info = _solve(p["A"], p["B"], _mesh(side), capacity=N, on_indefinite="continue")
    assert info["status"] == spd.CONVERGED and info["n_unstable"] == 1 and info["n_negative"] == 0
    assert info["indefinite_at"] is not None and info["general_solve"] == 0
    assert np.iscomplexobj(omega)
    gamma = p["imag_exact"][0]
    err_im = abs(omega[0].imag - gamma)
    print(f"MEASURED indefinite {side}x{side}: unstable root {omega[0]:.10f} Ry (exact i {gamma:.10f}), "
          f"error {err_im * RY_EV:.2e} eV, lam_min {info['lam_min']:.6e} (K: {p['k_min']:.6e}), found at "
          f"iteration {info['indefinite_at']}, {info['pair_calls']} pair calls", flush=True)
    assert err_im * RY_EV <= ROOT_TOL_EV and abs(omega[0].real) <= 1e-10
    assert p["k_min"] * (1 + 1e-6) <= info["lam_min"] < 0
    # the unstable pair: zero Sigma norm, unit length, an eigenvector of H at i gamma
    x, y = X[:, 0], Y[:, 0]
    assert abs(np.vdot(x, x) - np.vdot(y, y)) <= 1e-8 and abs(np.vdot(x, x) + np.vdot(y, y) - 1) <= 1e-10
    z = np.concatenate([x, y])
    assert np.linalg.norm(orc.bse_H(p["A"], p["B"]) @ z - omega[0] * z) <= 1e-7
    # the real roots above it, Sigma-orthonormal
    _check_pairs(p, omega, X, Y, info, first=1)


def _rotated_diag(a, b, seed=7):
    rng = np.random.default_rng(seed)
    return orc.rotate_unitary(np.diag(a).astype(complex), np.diag(b).astype(complex),
                              orc.random_unitary(a.size, rng))


def _unstable_cases():
    """Three pencils whose K has two negative eigenvalues; each with its special roots."""
    rng = np.random.default_rng(11)
    a = 0.4 + 0.02 * np.arange(N) + 0.003 * rng.random(N)
    b = 0.03 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    # (a) one transition at negative energy: a real root of positive type at -sqrt(a^2 - |b|^2)
    a1, b1 = a.copy(), b.copy()
    a1[0], b1[0] = -0.05, 0.01
    neg = (*_rotated_diag(a1, b1), np.array([-np.sqrt(0.05 ** 2 - 0.01 ** 2)]), 0, 1)
    # (b) a complex quadruplet: (A - B)(A + B) = g^2 [[0, 1], [-1, 0]] on a 2x2 block
    g = 0.05
    A0, B0 = np.diag(a).astype(complex), np.diag(b).astype(complex)
    A0[:2, :2] = g * np.array([[0.5, 0.5], [0.5, -0.5]])
    B0[:2, :2] = g * np.array([[-0.5, 0.5], [0.5, 0.5]])
    quad = (*orc.rotate_unitary(A0, B0, orc.random_unitary(N, rng)),
            np.array([g * (1 + 1j) / np.sqrt(2)]), 1, 0)
    # (c) two imaginary pairs: real A + B positive definite, A - B with two negative eigenvalues
    q1, _ = np.linalg.qr(rng.standard_normal((N, N)))
    q2, _ = np.linalg.qr(rng.standard_normal((N, N)))
    lam = np.concatenate([[-1e-3, -2.5e-3], 0.3 + 0.7 * rng.random(N - 2)])
    mu = 0.5 + rng.random(N)
    Mm, Pp = (q1 * lam) @ q1.T, (q2 * mu) @ q2.T
    ph = (q2 * np.sqrt(mu)) @ q2.T
    w2 = np.sort(np.linalg.eigvalsh(ph @ Mm @ ph))
    two = (*orc.rotate_unitary(0.5 * (Pp + Mm), 0.5 * (Pp - Mm), orc.random_unitary(N, rng)),
           1j * np.sqrt(-w2[:2]), 2, 0)
    return {"negative_real": neg, "quadruplet": quad, "two_imaginary": two}


@pytest.mark.parametrize("case", ("negative_real", "quadruplet", "two_imaginary"))
def test_unstable_sectors_beyond_one_imaginary_pair(case):
    A, B, special, n_un, n_neg = _unstable_cases()[case]
    orc.assert_bse_structure(A, B, tol=1e-13)
    omega, X, Y, info = _solve(A, B, _mesh(2), capacity=N, on_indefinite="continue")
    assert info["status"] == spd.CONVERGED
    assert (info["n_unstable"], info["n_negative"]) == (n_un, n_neg), info
    n_sp = n_un + n_neg
    found = np.asarray(omega)[:n_sp]
    err = max(np.min(np.abs(found - s)) for s in special)
    lam = np.linalg.eigvals(orc.bse_H(A, B))
    stable = np.sort(lam[(np.abs(lam.imag) < 1e-9) & (lam.real > 0)].real)
    Z = np.concatenate([X, Y], axis=0)[:, :n_sp + N_EIG]
    R = np.linalg.norm(orc.bse_H(A, B) @ Z - Z * np.asarray(omega)[None, :n_sp + N_EIG], axis=0)
    # every stable root returned is a root of H (the negative-type partner of a negative root is not listed)
    got = np.real(np.asarray(omega)[n_sp:n_sp + N_EIG])
    err_stable = max(np.min(np.abs(stable - w)) for w in got)
    print(f"MEASURED {case}: special roots {np.array2string(found, precision=8)} within {err * RY_EV:.1e} eV "
          f"of the exact ones, stable roots within {err_stable * RY_EV:.1e} eV, residual {R.max():.1e}, "
          f"lam_min {info['lam_min']:.4e}", flush=True)
    assert err * RY_EV <= ROOT_TOL_EV and err_stable * RY_EV <= ROOT_TOL_EV and R.max() <= 1e-7
    assert np.all(np.diff(got) >= 0) and got[0] > 0
    Xs, Ys = X[:, n_un:n_sp + N_EIG], Y[:, n_un:n_sp + N_EIG]     # real roots, the negative one included
    assert np.max(np.abs(orc.sigma_gram(Xs, Ys) - np.eye(Xs.shape[1]))) <= 1e-10
    if case == "negative_real":
        # the partner at +0.049 has negative Sigma norm: it is a root of H but not an excitation
        assert np.min(np.abs(got - abs(special[0]))) > 1e-3


# ---------------------------------------------------------------------------
# 4. operator integrity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("broken", ("B_not_symmetric", "A_not_hermitian"))
def test_broken_operator_structure_refuses(pencils, broken):
    p = pencils["dominant"]
    rng = np.random.default_rng(1)
    R = rng.standard_normal((N, N))
    R = 1e-3 * (R - R.T) / np.sqrt(N)                                  # real antisymmetric
    A, B = (p["A"], p["B"] + R) if broken == "B_not_symmetric" else (p["A"] + R, p["B"])
    with pytest.raises(ValueError, match="GATE sp_davidson_operator") as exc:
        _solve(A, B, _mesh(2), precond=True, refusal_notes={"operator": "CALLER NOTE."})
    msg = str(exc.value)
    import re
    asym = float(re.search(r"got: ([0-9.]+e[+-]\d+) relative", msg).group(1))
    assert asym > 1e-6 and "A = A^H and B = B^T" in msg and "CALLER NOTE." in msg
    assert "after 20 pair calls" in msg                                  # refused on the start block
    # the same broken operator runs (and is measured on every block) when the gate is off; the correct
    # one is round-off
    omega, _, _, info = _solve(A, B, _mesh(2), precond=True, metric_tol=None, max_iter=3)
    assert info["metric_asym"] >= asym * (1 - 1e-3) and info["status"] != spd.CONVERGED
    _, _, _, good = _solve(p["A"], p["B"], _mesh(2), precond=True, max_iter=3)
    assert good["metric_asym"] <= 1e-12


# ---------------------------------------------------------------------------
# 5. the Tamm-Dancoff features
# ---------------------------------------------------------------------------

def test_locking_margin_and_restart(pencils):
    p = pencils["dominant"]
    mesh = _mesh(2)
    ref = p["ref"]["omega"][:N_EIG]
    runs = {
        "default": _solve(p["A"], p["B"], mesh, precond=True),                           # capacity 20 n_eig
        "no_locking": _solve(p["A"], p["B"], mesh, precond=True, lock_converged=False),
        "margin_lock": _solve(p["A"], p["B"], mesh, precond=True, margin_lock=1e4),
        "tight_memory": _solve(p["A"], p["B"], mesh, precond=True, capacity=1),          # raised to 6 blocks
        "no_margin": _solve(p["A"], p["B"], mesh, precond=True, margin=0),
    }
    for name, (omega, X, Y, info) in runs.items():
        assert info["status"] == spd.CONVERGED, name
        assert np.max(np.abs(omega[:N_EIG] - ref)) * RY_EV <= ROOT_TOL_EV, name
        print(f"MEASURED {name}: {info['iterations']} iterations, {info['pair_calls']} pair calls, "
              f"{info['restarts']} restarts, capacity {info['capacity']}, block {info['n_block']}", flush=True)
    calls = {name: run[3]["pair_calls"] for name, run in runs.items()}
    info = runs["default"][3]
    assert info["capacity"] == spd.DEFAULT_CAPACITY_FACTOR * N_EIG and info["n_block"] == N_EIG + MARGIN
    assert info["n_tested"] == N_EIG and np.all(info["converged"][:N_EIG])
    # soft locking and the margin lock save pair calls; they change no root
    assert calls["no_locking"] > calls["default"] > calls["margin_lock"]
    assert np.all(info["root_corrections"] <= info["iterations"])
    assert np.all(runs["no_locking"][3]["root_corrections"] == runs["no_locking"][3]["iterations"] - 1)
    # the memory cap: a third of the basis, more restarts, the same roots
    tight = runs["tight_memory"][3]
    assert tight["capacity"] == spd.MIN_CAPACITY_BLOCKS * (N_EIG + MARGIN) < info["capacity"]
    assert tight["restarts"] > info["restarts"] >= 1 and tight["active_size"] <= tight["capacity"]
    assert runs["no_margin"][3]["n_block"] == N_EIG


def test_unconverged_stops_are_returned_not_raised(pencils):
    p = pencils["clustered"]
    mesh = _mesh(2)
    omega, X, Y, info = _solve(p["A"], p["B"], mesh, capacity=N, max_iter=5)
    assert info["status"] == spd.ITERATION_LIMIT and info["iterations"] == 5
    assert not np.all(info["converged"][:N_EIG]) and np.max(info["residuals"][:N_EIG]) > TOL
    omega, X, Y, info = _solve(p["A"], p["B"], mesh, tol=1e-15, stall_patience=3, max_iter=400)
    assert info["status"] == spd.STALLED and info["status_name"] == "STALLED"
    assert info["iterations"] < 400                                     # the stall exit, not the cap


def test_whole_space_block_converges_or_reports_no_directions():
    """A block as large as the space: the first basis ``[X0, i X0]`` is the whole real form, so the
    Ritz values are the spectrum; a tolerance below rounding then ends ``NO_DIRECTIONS`` (no room,
    no independent correction), never ``CONVERGED``."""
    p = orc.dominant_pencil(n=16)
    ref = orc.reference_roots(p["A"], p["B"])
    rng = np.random.default_rng(2)
    X0 = jnp.asarray((rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))).reshape(16, 4, 4, 1))
    data = (jnp.asarray(p["A"]), jnp.asarray(p["B"]))
    omega, X, Y, info = spd.sp_davidson(_pair, X0, data=data, tol=1e-10, verbose=False)
    assert info["status"] == spd.CONVERGED and info["iterations"] == 1
    assert info["capacity"] == 32 == info["active_size"] and info["pair_calls"] == 32
    assert np.max(np.abs(omega - ref["omega"])) * RY_EV <= ROOT_TOL_EV      # all 16 roots
    assert abs(info["lam_min"] - p["k_min"]) <= 1e-12                         # the whole K: lam_min exactly
    omega, X, Y, info = spd.sp_davidson(_pair, X0, data=data, tol=1e-19, verbose=False)
    assert info["status"] == spd.NO_DIRECTIONS and info["status_name"] == "NO_DIRECTIONS"
    assert np.max(np.abs(omega - ref["omega"])) * RY_EV <= ROOT_TOL_EV      # the roots are there all the same


def test_bad_inputs_refuse(pencils):
    p = pencils["dominant"]
    mesh = _mesh(1)
    X0 = _start(N_EIG + MARGIN)
    X0[1] = X0[0]                                                       # a dependent start block
    with pytest.raises(ValueError, match="rank deficient"):
        _solve(p["A"], p["B"], mesh, X0=X0)
    nan_pair = lambda data, X, s: _pair(data, X, s) * jnp.nan          # noqa: E731
    with pytest.raises(FloatingPointError, match="non-finite"):
        _solve(p["A"], p["B"], mesh, pair=nan_pair)
    for bad in (dict(tol=0.0), dict(tol=float("nan")), dict(max_iter=0), dict(margin_lock=0.5),
                dict(on_indefinite="ignore")):
        with pytest.raises(ValueError, match="sp_davidson"):
            _solve(p["A"], p["B"], mesh, **bad)
    with pytest.raises(ValueError, match="n_converge"):
        spd.sp_davidson(_pair, jnp.asarray(_start(4).reshape((4,) + WINDOW)), n_converge=5)
    with pytest.raises(TypeError, match="complex128"):
        spd.sp_davidson(_pair, jnp.zeros((4,) + WINDOW))


# ---------------------------------------------------------------------------
# 6. accounting and kernel reuse
# ---------------------------------------------------------------------------

def test_pair_calls_are_counted_and_kernels_reused(pencils):
    from jax.experimental import io_callback
    p = pencils["dominant"]
    seen = {"rows": 0, "signs": set(), "traces": 0}

    def note(rows, lo, hi):
        seen["rows"] += int(rows)
        seen["signs"].update((float(lo), float(hi)))

    def counted(data, X, s):
        seen["traces"] += 1                                             # a trace, not a call
        io_callback(note, None, jnp.int32(X.shape[0]), jnp.min(s), jnp.max(s), ordered=True)
        return _pair(data, X, s)

    spd._KERNELS.clear()
    omega, X, Y, info = _solve(p["A"], p["B"], _mesh(1), precond=True, pair=counted, plan_key="accounting")
    assert info["status"] == spd.CONVERGED
    assert seen["signs"] == {1.0}                                       # s = +1 only: K(x) = -i M(i x)
    assert seen["rows"] == info["pair_calls"]
    assert info["pair_calls_start"] == 2 * (N_EIG + MARGIN)
    assert info["pair_calls_per_root"] == info["pair_calls"] / N_EIG
    assert info["pair_calls"] <= info["pair_calls_start"] + 2 * int(np.sum(info["root_corrections"]))
    traces, rows = seen["traces"], seen["rows"]
    assert 1 <= traces <= 6                                             # one per power-of-two width
    omega2, *_, info2 = _solve(p["A"], p["B"], _mesh(1), precond=True, pair=counted, plan_key="accounting")
    assert seen["traces"] == traces and len(spd._KERNELS) == 1          # the planned kernels were reused
    assert seen["rows"] == 2 * rows and info2["pair_calls"] == info["pair_calls"]
    assert np.array_equal(omega, omega2)
    spd._KERNELS.clear()


def test_kernels_keep_vectors_sharded_and_reduce_coefficients_only():
    """The optimized HLO of every kernel on the 2x2 mesh: its collectives, and no array larger than
    a rank's own tile of the basis.

    Geometry chosen so a replicated block would show: capacity 24, block 2 x 4, 512 transitions, so a
    rank's basis tile has 24 x 128 = 3072 elements, a whole block 8 x 512 = 4096 and a whole basis
    12288.  The reductions are ``lxkit.ordered_psum`` of coefficient matrices (an all-gather of the
    per-rank partials, summed in rank order) and two all-reduces of row norms.
    """
    import collections
    import re
    mesh = _mesh(2)
    sh = NamedSharding(mesh, P(None, "x", "y", None))
    rep = NamedSharding(mesh, P())
    nb, vs, cap = 4, (4, 4, 32), 24
    n = int(np.prod(vs))
    alg = spd._algebra(nb, cap, vs, sh)
    ops = spd._operator_kernels(alg, lambda d, X, s: 2.0 * X, spd.sigma_gradient, nb, vs, sh)
    rng = np.random.default_rng(0)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)   # noqa: E731
    W, MW = (device_put_process_local(cplx(cap, *vs), sh) for _ in range(2))
    Pb, X0 = device_put_process_local(cplx(2 * nb, *vs), sh), device_put_process_local(cplx(nb, *vs), sh)
    Cc, om = device_put_process_local(cplx(cap, 2 * nb), rep), device_put_process_local(cplx(nb), rep)
    mask = device_put_process_local(np.ones(2 * nb, bool), rep)
    i32 = np.int32
    cases = {                                       # kernel: (callable, operands, collectives)
        "first": (alg.first, (X0, i32(8)), {"all-gather": 2}),                       # two block Grams
        "ortho": (alg.ortho, (W, Pb, mask, i32(12), i32(8)), {"all-gather": 6}),     # 4 basis + 2 block Grams
        "extend": (alg.extend, (W, MW, Pb, Pb, i32(12), i32(8)), {"all-gather": 2}),  # the two panels
        "restart": (alg.restart, (W, MW, Cc, i32(12), i32(8), Pb), {}),
        "sigma_gram": (alg.sigma_gram, (X0, X0), {"all-gather": 1}),
        "ritz": (ops.ritz, ((), W, MW, Cc, om, i32(12), Pb), {"all-reduce": 2}),     # row norms
        "apply": (ops.apply(4), ((), Pb, Pb, i32(2)), {}),                           # the applier's own only
    }
    tile = cap * n // 4
    for name, (fn, args, want) in cases.items():
        text = fn.lower(*args).compile().as_text()
        found = collections.Counter(re.findall(
            r"= [^\n]*?\b(all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter)(?:-start)?\(", text))
        shapes = re.findall(r"^\s*(?:ROOT\s+)?%[^\n=]+ = (?:c128|f64)\[([0-9,]+)\]", text, re.MULTILINE)
        sizes = [int(np.prod(tuple(map(int, s.split(","))))) for s in shapes]
        print(f"MEASURED kernel {name}: collectives {dict(found)}, largest array {max(sizes)} elements "
              f"(a rank's basis tile {tile}, a whole block {2 * nb * n}, a whole basis {cap * n})", flush=True)
        assert dict(found) == want, (name, dict(found))
        assert sizes and max(sizes) <= tile, (name, max(sizes))       # nothing vector-sized is replicated


# ---------------------------------------------------------------------------
# 7. the projected solves and the rank-0 agreement
# ---------------------------------------------------------------------------

def _projected(seed, m=24, negative=0):
    """A projected pencil ``(M_r, S)`` from a real-orthonormal basis of a small BSE pencil."""
    rng = np.random.default_rng(seed)
    n = 40
    p = orc.indefinite_pencil(n=n, delta=1e-2) if negative else orc.dominant_pencil(n=n)
    W = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    G = np.real(W.conj() @ W.T)
    e, U = np.linalg.eigh(G)
    W = ((U / np.sqrt(e)).T @ W)                                        # Re(W W^H) = 1
    MW = W @ p["A"].T + W.conj() @ p["B"].T
    return np.real(W.conj() @ MW.T), np.imag(W.conj() @ W.T)


def test_projected_solves():
    import scipy.linalg as sla
    Mr, S = _projected(0)
    Mr, S = 0.5 * (Mr + Mr.T), 0.5 * (S - S.T)
    omega, C = spd.solve_projected_definite(Mr, S, 5)
    w = sla.eig(Mr, 1j * S, right=False)
    w = np.sort(w[np.isfinite(w) & (w.real > 0)].real)
    assert np.max(np.abs(omega - w[:5])) <= 1e-12 and np.all(np.diff(omega) >= 0)
    assert np.max(np.abs(C.conj().T @ (2j * S) @ C - np.eye(5))) <= 1e-12       # X^H X - Y^H Y = 1, Eq. (4)
    assert np.max(np.abs(Mr @ C - (1j * S) @ C * omega[None, :])) <= 1e-12
    with pytest.raises(np.linalg.LinAlgError):
        spd.solve_projected_definite(Mr - 10.0 * np.eye(Mr.shape[0]), S, 5)
    # an indefinite projected pencil (the whole 40-transition space): the unstable sector is separated,
    # the stable roots come from the Cholesky route on the complement, and the rotation keeps both
    pen = orc.indefinite_pencil(n=40, delta=1e-2)
    ref = orc.reference_roots(pen["A"], pen["B"])
    Wf = np.concatenate([np.eye(40), 1j * np.eye(40)]).astype(complex)       # the real form of C^40
    Mr = np.real(Wf.conj() @ (Wf @ pen["A"].T + Wf.conj() @ pen["B"].T).T)
    S = np.imag(Wf.conj() @ Wf.T)
    sol = spd._solve_projected(0.5 * (Mr + Mr.T), 0.5 * (S - S.T), 8, 1.0, definite=False)
    assert (sol["n_unstable"], sol["n_negative"], sol["general"]) == (1, 0, False)
    assert abs(sol["omega"][0] - 1j * ref["imag"][0]) <= 1e-10
    assert np.max(np.abs(sol["omega"][1:].real - ref["omega"][:7])) <= 1e-10
    assert sol["keep"].shape == (80, 2)                                   # c(+i gamma) and c(-i gamma)
    step = spd._projected_step(0.5 * (Mr + Mr.T), 0.5 * (S - S.T), 8, 100, solve_indefinite=True)
    om, C, present, Q, counts, reals = step
    assert C.shape == (100, 8) and Q.shape == (100, 16) and present.all()
    assert list(counts) == [0, 1, 1, 16, 0] and reals[0] < 0 < reals[1]
    Qm = Q[:80, :16]
    assert np.max(np.abs(Qm.T @ Qm - np.eye(16))) <= 1e-12
    assert np.max(np.abs(Qm @ (Qm.T @ C[:80]) - C[:80])) <= 1e-9          # Eq. (7): the Ritz block survives
    refused = spd._projected_step(0.5 * (Mr + Mr.T), 0.5 * (S - S.T), 8, 100, solve_indefinite=False)
    assert not refused[2].any() and refused[4][0] == 0 and refused[5][0] == reals[0]


def test_general_solve_is_the_fallback_when_the_complement_is_not_definite(monkeypatch):
    """If the Cholesky route fails on the complement of the unstable sector, the stable roots come
    from the general solve (flagged ``general``); they are the same roots."""
    pen = orc.indefinite_pencil(n=40, delta=1e-2)
    ref = orc.reference_roots(pen["A"], pen["B"])
    Wf = np.concatenate([np.eye(40), 1j * np.eye(40)]).astype(complex)
    Mr = np.real(Wf.conj() @ (Wf @ pen["A"].T + Wf.conj() @ pen["B"].T).T)
    S = np.imag(Wf.conj() @ Wf.T)
    Mr, S = 0.5 * (Mr + Mr.T), 0.5 * (S - S.T)

    def fails(*a, **k):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(spd, "solve_projected_definite", fails)
    sol = spd._solve_projected(Mr, S, 8, 1.0, definite=False)
    assert sol["general"] is True and (sol["n_unstable"], sol["n_negative"]) == (1, 0)
    assert abs(sol["omega"][0] - 1j * ref["imag"][0]) <= 1e-10
    assert np.max(np.abs(sol["omega"][1:].real - ref["omega"][:7])) <= 1e-9
    C = sol["C"][:, 1:]
    assert np.max(np.abs(np.diag(C.conj().T @ (2j * S) @ C) - 1.0)) <= 1e-10    # each root normalised
    step = spd._projected_step(Mr, S, 8, 100, solve_indefinite=True)
    assert list(step[4]) == [0, 1, 1, 16, 1]                                     # ... and counted


def test_host_results_are_rank_zeros_on_every_rank(monkeypatch):
    """Two emulated ranks whose host solves differ: both end with rank 0's arrays, bit for bit."""
    rng = np.random.default_rng(3)
    mine = [rng.standard_normal(5) + 1j * rng.standard_normal(5), rng.standard_normal((4, 3)) + 0j,
            np.array([True, False, True]), rng.standard_normal((4, 6)),
            np.array([1, 7, 2, 12, 0], np.int64), np.array([-3.5e-4, 1e-13])]
    # a rank whose LAPACK rounded otherwise: other floats, other counts, other flags
    other = [~a if a.dtype == bool else a + 1 if a.dtype == np.int64 else a + 1e-9 for a in mine]
    sent = {}

    def fake_psum(local, mesh):
        sent[len(sent)] = np.array(local)
        return sent[0] + 0.0 * np.asarray(local)                         # rank 0's plus the others' zeros

    monkeypatch.setattr(spd, "process_count", lambda: 2)
    monkeypatch.setattr(spd, "psum_replicate", fake_psum)
    results = {}
    for rank, arrays in ((0, mine), (1, other)):
        monkeypatch.setattr(spd, "process_rank", lambda rank=rank: rank)
        results[rank] = spd._from_rank0(arrays, mesh=object())
    assert not np.any(sent[1])                                           # rank 1 contributed zeros only
    for a, r0, r1 in zip(mine, results[0], results[1]):
        assert r0.dtype == a.dtype and r1.dtype == a.dtype and r0.shape == a.shape
        assert np.array_equal(r0, a) and np.array_equal(r1, a)
    # one process, or no mesh: nothing is sent
    monkeypatch.setattr(spd, "process_count", lambda: 1)
    assert spd._from_rank0(mine, mesh=object()) is mine
    monkeypatch.setattr(spd, "process_count", lambda: 2)
    assert spd._from_rank0(mine, mesh=None) is mine
