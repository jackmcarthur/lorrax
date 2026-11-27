"""The full-BSE Davidson route (``full_bse_solver = davidson``) on the smallest realistic payload.

``bse_nontda.solve_bse_nontda_sharded(solver='davidson')`` runs ``solvers.bse_sp_davidson`` over the
fused pair applier with the paired exact diagonal and the Tamm-Dancoff start block.  The payload is the
synthetic time-reversal-symmetric spinor one of ``tests/test_bse_full_dipoles_cpu.py`` (k grid 2x2x2,
4v x 4c, N_mu 24: 128 transitions), energies in Ry; the reference is the dense route on the same
payload and the oracle on the blocks it assembles.

1. On a 1x1 mesh and on the emulated 2x2 mesh the Davidson's lowest 8 roots equal the dense route's to
   1e-8 Ry at the default 0.01 meV stop; the pairs lie on the mesh's transition axes, are
   Sigma-orthonormal and solve ``H z = omega z`` of the assembled blocks; at a tight stop ``X`` and ``Y``
   equal the dense route's to 1e-8, and the per-state dipoles follow.
2. The preconditioner's diagonal is the diagonal of the assembled ``A`` (1e-12), and the diagonal of
   ``H`` is ``[D; -D]``.
3. Pair calls per root of this route, of the matrix-free Lanczos and A applications per root of the
   Tamm-Dancoff Davidson (``MEASURED`` lines).  The Tamm-Dancoff default route returns the roots and
   takes the iterations, matvecs and restarts recorded at the base commit of this branch (b66181c).
4. Gates, each with a red twin: an iteration cap refuses (``GATE nontda_davidson_converged``) and so
   does a broken normalisation; a non-symmetric coupling block refuses (``GATE sp_davidson_operator``);
   an operator shifted until ``K`` has one negative eigenvalue refuses with the measured ``lam_min``
   and names ``full_bse_instability = continue`` (``GATE sp_davidson_definite``), and with
   ``on_indefinite='continue'`` returns the positive roots, keeps the imaginary one and refuses a block
   without room for it (``GATE nontda_davidson_block``); an option of another route, an unknown option,
   a ``head_tensor`` payload, an Olsen correction and ``keep_W_R`` refuse.
5. The driver: ``full_bse = true`` alone still runs the dense solve; the two new input keys parse, the
   keys the route cannot honour refuse, the SETTINGS rows and the run header name the solver that runs,
   the dense size refusal names the new route, and ``bse_jax._preview_lanczos`` writes ``eigenvectors.h5``
   with both ``X`` and ``Y`` through the existing writer (equal to the dense run's file); a capped run
   refuses before it writes.

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_full_davidson_cpu.py -s``; under a minute on two threads.
"""
import logging
import os
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
h5py = pytest.importorskip("h5py")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from bse import bse_io, bse_jax, bse_lanczos, bse_nontda, bse_stack_matvec, bse_window  # noqa: E402
from bse.absorption_common import (build_dipole_vector_bse, exciton_dipole_projections,  # noqa: E402
                                   exciton_dipoles_distributed)
from bse.bse_nontda import (_full_matvec_and_args, _materialize_A_B, refuse_unconverged_davidson,  # noqa: E402
                            solve_bse_nontda_sharded)
from bse.bse_ring_comm import make_bse_shardings  # noqa: E402
from tests import bse_full_oracle as orc  # noqa: E402
from tests.test_bse_default_solver import _toy_deck  # noqa: E402
from tests.test_bse_full_dipoles_cpu import NC, NK, NV, N, _operands, _payload  # noqa: E402

N_EIG = 8
ROOT_TOL = 1e-8                    # Ry, against the dense route
RY_MEV = 13605.6980659

#: The Tamm-Dancoff default route on this payload (1x1 mesh, 8 roots + 8 margin, ``max_iter = 64``),
#: recorded at the base commit of this branch (b66181c, JAX 0.12 CPU, before any change here).
TDA_BASELINE = {
    "roots": [0.4012768911428358, 0.40657190506810853, 0.40805524399724896, 0.4086414601978401,
              0.4095790248183601, 0.4114727008394151, 0.4127210510168713, 0.41385265338381444],
    "iterations": 14, "matvecs": 207, "restarts": 3}


def _mesh(side):
    if jax.device_count() < side * side:
        pytest.skip("four host devices required (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:side * side]).reshape(side, side), ("x", "y"))


@pytest.fixture(scope="module")
def ops():
    return _operands()


@pytest.fixture(scope="module")
def dense(ops):
    """Per mesh side: the dense route's roots and vectors, and the blocks it assembles."""
    out = {}
    for side in (1, 2):
        if jax.device_count() < side * side:
            continue
        mesh = _mesh(side)
        omega, evecs, _ = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG)
        sh = make_bse_shardings(mesh)
        pair, args = _full_matvec_and_args(_payload(mesh, ops), mesh, sh, include_W=True)
        A, B = _materialize_A_B(pair, args, sh, NC, NV, NK)
        out[side] = {"omega": np.asarray(omega), "evecs": np.asarray(evecs), "A": A, "B": B}
    return out


def _columns(evecs):
    """Host ``(n, 2, nc, nv, nk)`` pairs as ``X``, ``Y`` of shape (N, n)."""
    host = np.asarray(evecs)
    n = host.shape[0]
    return host[:, 0].reshape(n, N).T, host[:, 1].reshape(n, N).T


def _davidson(mesh, ops, **kw):
    omega, evecs, n_iter = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG,
                                                    solver="davidson", **kw)
    return np.asarray(omega), evecs, int(n_iter), dict(bse_nontda.LAST_DAVIDSON)


# ---------------------------------------------------------------------------
# 1. the route against the dense solve, 1x1 and 2x2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", (1, 2))
def test_davidson_route_matches_dense(ops, dense, side):
    mesh = _mesh(side)
    ref = dense[side]
    omega, evecs, n_iter, run = _davidson(mesh, ops)
    info = run["info"]
    err = float(np.max(np.abs(omega - ref["omega"])))
    assert omega.shape == (N_EIG,) and np.all(np.diff(omega) >= 0)
    assert err <= ROOT_TOL, err
    assert isinstance(evecs.sharding, NamedSharding)
    assert tuple(evecs.sharding.spec) == (None, None, "x", "y", None)       # the dense route's layout
    assert evecs.shape == (N_EIG, 2, NC, NV, NK) and n_iter == info["iterations"]
    X, Y = _columns(evecs)
    gram = float(np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(N_EIG))))
    resid = float(np.max(orc.eigpair_residuals(ref["A"], ref["B"], omega, X, Y)))
    assert gram <= 1e-10 and info["sigma_err"] <= 1e-10
    assert resid <= 1e-5                                                     # the 0.01 meV stop: 7.3e-7 Ry
    assert info["status_name"] == "CONVERGED" and info["n_unstable"] == 0 and run["n_special"] == 0
    k_min = float(np.linalg.eigvalsh(orc.bse_K(ref["A"], ref["B"]))[0])
    assert k_min * (1 - 1e-9) <= info["lam_min"] <= 2.0 * k_min             # an upper bound of lam_min(K)
    assert info["metric_asym"] <= 1e-12
    print(f"MEASURED davidson route {side}x{side}: max|omega - dense| = {err:.3e} Ry, Sigma Gram {gram:.1e}, "
          f"eigenpair residual {resid:.1e}, {info['iterations']} iterations, {info['pair_calls']} pair calls "
          f"({info['pair_calls'] / N_EIG:.1f} per root), {info['restarts']} restarts, lam_min "
          f"{info['lam_min']:.6f} Ry (lam_min(K) {k_min:.6f} Ry)", flush=True)

    # a tight stop: X and Y themselves against the dense route's (fix each state's phase)
    omega_t, evecs_t, _, run_t = _davidson(mesh, ops, davidson_tol_mev=1e-7)
    Xt, Yt = _columns(evecs_t)
    Xd, Yd = _columns(ref["evecs"])
    assert np.max(np.abs(np.diff(ref["omega"]))) > 0 and np.min(np.diff(ref["omega"])) > 1e-4   # no multiplet
    phase = np.sum(Xd.conj() * Xt, axis=0) - np.sum(Yd.conj() * Yt, axis=0)                     # <dense|Sigma|dav>
    assert np.max(np.abs(np.abs(phase) - 1.0)) <= 1e-10
    dX = float(np.max(np.abs(Xt - Xd * phase[None, :])))
    dY = float(np.max(np.abs(Yt - Yd * phase[None, :])))
    err_t = float(np.max(np.abs(omega_t - ref["omega"])))
    print(f"MEASURED davidson route {side}x{side}, 1e-7 meV stop: max|omega - dense| = {err_t:.3e} Ry, "
          f"max|X - X_dense| = {dX:.2e}, max|Y - Y_dense| = {dY:.2e} (largest |Y| {np.max(np.abs(Yd)):.2e}), "
          f"{run_t['info']['pair_calls']} pair calls", flush=True)
    assert err_t <= 1e-10 and dX <= 1e-8 and dY <= 1e-8
    # the per-state dipoles through the production contraction equal the host contraction
    from lxkit import device_put_process_local
    d_block = device_put_process_local(build_dipole_vector_bse(ops["d_kcv"]),
                                       NamedSharding(mesh, P(None, "x", "y", None)))
    got = exciton_dipoles_distributed(evecs_t, d_block, N_EIG)
    host = np.asarray(evecs_t)
    want = exciton_dipole_projections(host[:, 0], ops["d"], Y=host[:, 1])
    assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want))
    dense_d = exciton_dipole_projections(ref["evecs"][:, 0], ops["d"], Y=ref["evecs"][:, 1])
    assert np.max(np.abs(np.abs(got) - np.abs(dense_d))) <= 1e-7 * np.max(np.abs(dense_d))


def test_route_options_are_honoured(ops, dense):
    """Each option the route reads changes the run and not the answer; W off follows ``include_W``."""
    mesh = _mesh(2)
    ref = dense[2]["omega"]
    base = _davidson(mesh, ops)[3]["info"]
    runs = {
        "bare diagonal": dict(davidson_precond="bare"),
        "capacity 96": dict(davidson_m_max=96),
        "margin 2": dict(davidson_margin=2),
        "margin locked at 1 meV": dict(davidson_margin_tol_mev=1.0),
        "eps shift 1e-5": dict(davidson_eps_shift_Ry=1e-5),
        "no random starts": dict(davidson_n_random_init=0),
    }
    for name, kw in runs.items():
        omega, _, _, run = _davidson(mesh, ops, **kw)
        info = run["info"]
        assert np.max(np.abs(omega - ref)) <= ROOT_TOL, name
        print(f"MEASURED option {name}: {info['iterations']} iterations, {info['pair_calls']} pair calls, "
              f"{info['restarts']} restarts, block {info['n_block']}, capacity {info['capacity']}", flush=True)
        runs[name] = info
    assert runs["capacity 96"]["capacity"] == 96 < base["capacity"] == 160
    assert runs["capacity 96"]["restarts"] > base["restarts"]
    assert runs["margin 2"]["n_block"] == N_EIG + 2 and base["n_block"] == N_EIG + 8
    assert runs["margin locked at 1 meV"]["pair_calls"] < base["pair_calls"]
    assert runs["bare diagonal"]["pair_calls"] != base["pair_calls"]
    # W off: the route applies the operator of include_W = False (the dense route's, not the W roots)
    omega_rpa, *_ = _davidson(mesh, ops, include_W=False)
    dense_rpa, _, _ = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, include_W=False)
    assert np.max(np.abs(omega_rpa - np.asarray(dense_rpa))) <= ROOT_TOL
    assert np.max(np.abs(np.asarray(dense_rpa) - ref)) > 1e-4


def test_padded_window_stays_on_the_physical_transitions(ops):
    """Three conduction bands padded to four on the 2x2 mesh (zero orbitals, the loader's 1e3 Ry
    sentinel energy): the start block has no weight on the pad, the solve never puts any there, and
    the roots are the dense route's."""
    mesh = _mesh(2)

    def padded():
        data = _payload(mesh, dict(ops, pc=np.concatenate([ops["pc"][:, :3], np.zeros_like(ops["pc"][:, :1])], axis=1),
                                   ec=np.concatenate([ops["ec"][:, :3], np.full((NK, 1), 1.0e3)], axis=1)))
        return dict(data, n_cond=3)

    dense_omega, dense_evecs, _ = solve_bse_nontda_sharded(padded(), mesh, n_eig=N_EIG)
    omega, evecs, _ = solve_bse_nontda_sharded(padded(), mesh, n_eig=N_EIG, solver="davidson")
    info = bse_nontda.LAST_DAVIDSON["info"]
    err = float(np.max(np.abs(np.asarray(omega) - np.asarray(dense_omega))))
    host = np.asarray(evecs)
    print(f"MEASURED padded window (3 of 4 conduction bands physical): max|omega - dense| = {err:.3e} Ry, "
          f"largest pad amplitude {np.max(np.abs(host[:, :, 3])):.1e}, {info['pair_calls']} pair calls, "
          f"lam_min {info['lam_min']:.6f} Ry", flush=True)
    assert err <= ROOT_TOL
    assert np.max(np.abs(host[:, :, 3])) == 0.0                              # exactly zero on the pad
    assert np.max(np.abs(np.asarray(dense_evecs)[:, :, 3])) <= 1e-12
    X, Y = _columns(evecs)
    assert np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(N_EIG))) <= 1e-10
    assert info["lam_min"] < 1.0                                             # the pad's 1e3 Ry never entered


# ---------------------------------------------------------------------------
# 2. the preconditioner's diagonal
# ---------------------------------------------------------------------------

def test_preconditioner_diagonal_is_the_diagonal_of_the_full_operator(ops, dense):
    from bse.bse_davidson_helpers import (bse_pair_diagonal_precond, build_bse_exact_diagonal,
                                          w_q0_for_precond)
    mesh = _mesh(2)
    sh = make_bse_shardings(mesh)
    data = _payload(mesh, ops)
    w_q0 = w_q0_for_precond(data)
    assert w_q0 is w_q0_for_precond(data) and w_q0.shape == (24, 24)         # stashed once
    pair, args = _full_matvec_and_args(data, mesh, sh, include_W=True)
    diag_sh = NamedSharding(mesh, P("x", "y", None))
    D = build_bse_exact_diagonal(args[4], args[5], args[0], args[3], w_q0, args[8], args[7], NK,
                                 sharding=diag_sh, memo=False)
    A, B = dense[2]["A"], dense[2]["B"]
    D_host = np.asarray(D).reshape(-1)
    assert np.max(np.abs(D_host - np.real(np.diag(A)))) <= 1e-12 * np.max(np.abs(np.diag(A)))
    H = orc.bse_H(A, B)
    assert np.max(np.abs(np.diag(H) - np.concatenate([D_host, -D_host]))) <= 1e-12       # [D; -D]: B adds nothing
    with pytest.raises(ValueError, match="already been consumed"):
        w_q0_for_precond({"W_q": None})
    # the paired division, Eq. (6): t_X = R_X / (D - omega + eps), t_Y = -R_Y / (D + omega + eps)
    pre = bse_pair_diagonal_precond(args[4], args[5], epsilon_shift=1e-3, sharding=diag_sh, diag_H=D)
    rng = np.random.default_rng(0)
    RX, RY = (rng.standard_normal((2, NC, NV, NK)) + 1j * rng.standard_normal((2, NC, NV, NK)) for _ in range(2))
    om = np.array([0.40 + 0.0j, 0.05j])
    tX, tY = pre.apply(pre.data, jnp.asarray(RX), jnp.asarray(RY), jnp.asarray(om))
    Dh = np.asarray(D)[None]
    assert np.max(np.abs(np.asarray(tX) - RX / (Dh - om[:, None, None, None] + 1e-3))) <= 1e-13
    assert np.max(np.abs(np.asarray(tY) + RY / (Dh + om[:, None, None, None] + 1e-3))) <= 1e-13
    bare = bse_pair_diagonal_precond(args[4], args[5], epsilon_shift=0.0, sharding=diag_sh)
    dE = np.asarray(args[4]).T[:, None, :] - np.asarray(args[5]).T[None, :, :]
    tX, _ = bare.apply(bare.data, jnp.asarray(RX), jnp.asarray(RY), jnp.asarray(om))
    assert np.max(np.abs(np.asarray(tX) - RX / (dE[None] - om[:, None, None, None]))) <= 1e-13


# ---------------------------------------------------------------------------
# 3. call counts, and the Tamm-Dancoff default route
# ---------------------------------------------------------------------------

def _tda(ops):
    """The Tamm-Dancoff default route on a 1x1 mesh: roots and ``solvers.davidson.LAST_RUN`` counts."""
    from solvers.davidson import LAST_RUN
    mesh = _mesh(1)
    ev, vec, n_iter = bse_lanczos.solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, max_iter=64)
    return np.asarray(ev), {"iterations": int(n_iter), "matvecs": int(LAST_RUN["mv"][0]),
                            "restarts": int(LAST_RUN["restarts"])}, vec


def test_tamm_dancoff_default_route_is_untouched(ops):
    """Same roots as before this branch (``TDA_BASELINE``), at no more than 1.5x its recorded cost (the
    counts were recorded on one machine; the roots are what the route must reproduce)."""
    ev, counts, vec = _tda(ops)
    assert np.max(np.abs(ev - np.array(TDA_BASELINE["roots"]))) <= 1e-10
    for key in ("iterations", "matvecs", "restarts"):
        assert counts[key] <= 1.5 * TDA_BASELINE[key], (key, counts)
    assert vec.shape == (N_EIG, 1, NC, NV, NK)                                # X only, as before
    # the default arguments still dispatch the Tamm-Dancoff Davidson, and the full default is dense
    import inspect
    sig = inspect.signature(bse_lanczos.solve_bse_sharded).parameters
    assert sig["tda"].default is True and sig["solver_kind"].default == "davidson"
    assert sig["full_solver"].default == "dense" and sig["full_on_indefinite"].default == "refuse"


def test_pair_calls_per_root_table(ops, dense, caplog):
    """Pair calls per root: this route, the matrix-free Lanczos, and the Tamm-Dancoff Davidson."""
    caplog.set_level(logging.INFO)
    rows = []
    for side in (1, 2):
        mesh = _mesh(side)
        _, _, _, run = _davidson(mesh, ops)
        rows.append((f"SP Davidson {side}x{side}, defaults", run["info"]["pair_calls"],
                     f"{run['info']['iterations']} iterations, {run['info']['restarts']} restarts"))
        for label, kw in (("defaults (m_max 140 > N: the whole window)", {}),
                          ("m_max 56, n_keep 24", dict(mf_m_max=56, mf_n_keep=24))):
            caplog.clear()
            om, _, _ = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, solver="matrixfree", **kw)
            text = " ".join(r.getMessage() for r in caplog.records)
            m = re.search(r"(\d+) fused pair\s+applications .*?(\d+)/(\d+) restart cycles", text)
            assert np.max(np.abs(np.asarray(om) - dense[side]["omega"])) <= ROOT_TOL
            rows.append((f"SP Lanczos {side}x{side}, {label}", int(m.group(1)), f"{m.group(2)} restart cycles"))
    ev, counts, _ = _tda(ops)
    rows.append(("TD Davidson 1x1, defaults (A applications)", counts["matvecs"],
                 f"{counts['iterations']} iterations, {counts['restarts']} restarts"))
    for name, calls, note in rows:
        print(f"MEASURED calls | {name} | {calls} | {calls / N_EIG:.1f} per root | {note}", flush=True)
    by = {name: calls for name, calls, _ in rows}
    spd, td = by["SP Davidson 1x1, defaults"], by["TD Davidson 1x1, defaults (A applications)"]
    # cost bounds at 1.5x the counts recorded on one machine (436 pair calls here, 207 Tamm-Dancoff applications)
    assert spd <= 1.5 * 436 and td <= 1.5 * 207, (spd, td)
    assert by["SP Davidson 2x2, defaults"] <= 1.5 * 436


# ---------------------------------------------------------------------------
# 4. gates and their red twins
# ---------------------------------------------------------------------------

def test_iteration_cap_refuses(ops):
    mesh = _mesh(2)
    bse_nontda.LAST_DAVIDSON.clear()
    with pytest.raises(RuntimeError, match="GATE nontda_davidson_converged") as exc:
        _davidson(mesh, ops, davidson_max_iter=3)
    assert not bse_nontda.LAST_DAVIDSON                                      # nothing recorded as a result
    # red twin: the same route without the cap returns (test_davidson_route_matches_dense)


def test_stop_line_gate_predicates():
    """``refuse_unconverged_davidson`` on the solver's ``info``: a pass and its two refusals."""
    from solvers.bse_sp_davidson import CONVERGED, STALLED
    info = dict(status=CONVERGED, status_name="CONVERGED", n_tested=4, iterations=9, pair_calls=120,
                restarts=1, residuals=np.full(6, 1e-9), converged=np.ones(6, bool), lam_min=0.3,
                sigma_err=2e-15)
    line = refuse_unconverged_davidson(info, n_eig=4, tol_mev=0.01, budget=50)
    assert isinstance(line, str) and "4/4" in line
    with pytest.raises(RuntimeError, match="GATE nontda_davidson_converged"):
        refuse_unconverged_davidson(dict(info, status=STALLED, status_name="STALLED",
                                         converged=np.array([1, 1, 0, 1, 1, 1], bool)),
                                    n_eig=4, tol_mev=0.01, budget=50)
    with pytest.raises(RuntimeError, match="GATE nontda_davidson_converged"):
        refuse_unconverged_davidson(dict(info, sigma_err=1e-3), n_eig=4, tol_mev=0.01, budget=50)
    with pytest.raises(RuntimeError, match="GATE nontda_davidson_converged"):
        refuse_unconverged_davidson(dict(info, sigma_err=float("nan")), n_eig=4, tol_mev=0.01, budget=50)


def _patched_applier(monkeypatch, change):
    """``build_bse_stack_pair_matvec`` whose applier adds ``change(X, s)`` to its output."""
    orig = bse_stack_matvec.build_bse_stack_pair_matvec

    def build(*a, **kw):
        pair = orig(*a, **kw)

        def changed(X, s, *args):
            out = pair(X, s, *args)
            return out + change(X, jnp.asarray(s)).astype(out.dtype)
        return changed

    monkeypatch.setattr(bse_stack_matvec, "build_bse_stack_pair_matvec", build)


def test_non_symmetric_coupling_block_refuses(ops, monkeypatch):
    mesh = _mesh(2)
    rng = np.random.default_rng(1)
    S = rng.standard_normal((N, N))
    Rasym = jnp.asarray(0.05 * (S - S.T) / np.sqrt(N))                      # B -> B + R, R^T = -R
    _patched_applier(monkeypatch, lambda X, s: (
        s.reshape((-1,) + (1,) * 3) * (jnp.conj(X).reshape(X.shape[0], -1) @ Rasym.T).reshape(X.shape)))
    with pytest.raises(ValueError, match="GATE sp_davidson_operator") as exc:
        _davidson(mesh, ops)
    msg = str(exc.value)
    asym = float(re.search(r"got: ([0-9.]+e[+-]\d+) relative", msg).group(1))
    print(f"MEASURED broken-B refusal: Re(P^H M P) asymmetry {asym:.3e} (gate 1e-08)", flush=True)
    assert asym > 1e-4
    # red twin: the unpatched applier measures round-off (test_davidson_route_matches_dense)


@pytest.fixture(scope="module")
def unstable(dense):
    """The payload's blocks with ``A -> A - sigma``: ``K`` gets one negative eigenvalue, -2e-4 Ry."""
    A, B = dense[2]["A"], dense[2]["B"]
    k = np.linalg.eigvalsh(orc.bse_K(A, B))
    sigma = float(k[0]) + 2e-4
    assert k[1] - sigma > 1e-3                                              # exactly one negative eigenvalue
    ref = orc.reference_roots(A - sigma * np.eye(N), B)
    assert ref["definite"] is False and ref["imag"].shape == (1,)
    return {"sigma": sigma, "ref": ref, "k_min": -2e-4}


def test_unstable_reference_state_refuses_then_continues(ops, unstable, monkeypatch, caplog):
    mesh = _mesh(2)
    sigma, ref = unstable["sigma"], unstable["ref"]
    _patched_applier(monkeypatch, lambda X, s: -sigma * X)                  # A -> A - sigma
    # default: refuse, with the measured margin and the key that continues
    with pytest.raises(ValueError, match="GATE sp_davidson_definite") as exc:
        _davidson(mesh, ops)
    msg = str(exc.value)
    lam = float(re.search(r"lam_min = (-?[0-9.]+e[+-]\d+)", msg).group(1))
    assert unstable["k_min"] * (1 + 1e-6) <= lam <= 0.0
    # continue: the positive roots, and the imaginary one kept aside
    caplog.set_level(logging.INFO)
    omega, evecs, _, run = _davidson(mesh, ops, on_indefinite="continue")
    info = run["info"]
    err = float(np.max(np.abs(omega - ref["omega"][:N_EIG])))
    gamma = complex(run["omega_ry"][0])
    err_im = abs(gamma.imag - ref["imag"][0])
    print(f"MEASURED unstable payload: lam_min(K) = {unstable['k_min']:.1e} Ry, refused at lam_min = {lam:.3e} Ry; "
          f"continued: unstable root {gamma.imag * RY_MEV:.6f} i meV (dense eig {ref['imag'][0] * RY_MEV:.6f} i, "
          f"error {err_im:.2e} Ry), positive roots within {err:.2e} Ry of the dense eig, "
          f"{info['pair_calls']} pair calls, lam_min {info['lam_min']:.3e} Ry", flush=True)
    assert err <= ROOT_TOL and err_im <= ROOT_TOL and abs(gamma.real) <= 1e-9
    assert run["n_special"] == 1 and info["n_unstable"] == 1 and info["n_negative"] == 0
    assert omega.dtype == np.float64 and np.all(omega > 0) and np.all(np.diff(omega) >= 0)
    X, Y = _columns(evecs)
    assert np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(N_EIG))) <= 1e-9
    special = run["special_vectors"]
    assert special.shape == (1, 2, NC, NV, NK) and tuple(special.sharding.spec) == (None, None, "x", "y", None)
    xs, ys = _columns(special)
    assert abs(np.vdot(xs, xs) - np.vdot(ys, ys)) <= 1e-6                    # zero Sigma norm
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "unstable root 1" in text and "the reference state is not a minimum" in text
    line, rows = bse_jax._full_davidson_report()                            # the driver's RESULTS rows
    assert rows[0][0] == "stability" and "NOT positive definite" in rows[0][1] and len(rows) == 2
    assert rows[1][0] == "unstable root 1" and f"{gamma.imag * RY_MEV:+.6f} i meV" in rows[1][1]
    assert "not written" in rows[1][1] and line == run["line"]
    # a block with no room for the unstable root next to n_eig positive ones refuses
    with pytest.raises(RuntimeError, match="GATE nontda_davidson_block"):
        _davidson(mesh, ops, on_indefinite="continue", davidson_margin=0)
    # red twin of the refusal: the unshifted payload is positive definite and returns
    # (test_davidson_route_matches_dense, same default)


def test_options_the_route_cannot_honour_refuse(ops):
    mesh = _mesh(1)
    cases = [("dense", dict(davidson_tol_mev=0.02), "davidson_tol_mev (read by solver='davidson')"),
             ("dense", dict(on_indefinite="continue"), "on_indefinite (read by solver='davidson')"),
             ("dense", dict(mf_m_max=56), "mf_m_max (read by solver='matrixfree')"),
             ("davidson", dict(mf_n_keep=24), "mf_n_keep (read by solver='matrixfree')"),
             ("matrixfree", dict(davidson_margin=2), "davidson_margin (read by solver='davidson')")]
    for solver, kw, want in cases:
        with pytest.raises(ValueError, match="GATE nontda_route_option"):
            solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, solver=solver, **kw)
    with pytest.raises(ValueError, match="GATE nontda_route_option"):
        solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, solver="davidson", davidson_tolmev=1.0)
    with pytest.raises(ValueError, match="GATE nontda_route_option"):
        solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, solver="lanczos")
    with pytest.raises(ValueError, match="GATE nontda_head_tensor"):
        solve_bse_nontda_sharded(dict(_payload(mesh, ops), head_tensor=(None, None)), mesh, n_eig=N_EIG,
                                 solver="davidson")
    # through solve_bse_sharded: what the Tamm-Dancoff Davidson has and this route has not
    for kw, want in ((dict(davidson_olsen=True), "davidson_olsen"), (dict(keep_W_R=True), "keep_W_R"),
                     (dict(full_solver="trlan"), "full_solver='trlan'")):
        kw = {"full_solver": "davidson", **kw}
        with pytest.raises(ValueError, match="GATE nontda_route_option"):
            bse_lanczos.solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, tda=False, **kw)
    with pytest.raises(ValueError, match="GATE nontda_route_option"):
        bse_lanczos.solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, tda=False,
                                      full_on_indefinite="continue")


# ---------------------------------------------------------------------------
# 5. the driver
# ---------------------------------------------------------------------------

def test_full_bse_default_is_still_the_dense_solve(ops, dense):
    mesh = _mesh(2)
    bse_nontda.LAST_DAVIDSON.clear()
    ev, vec, n_iter = bse_lanczos.solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, tda=False)
    assert np.array_equal(np.asarray(ev), dense[2]["omega"]) and int(n_iter) == 0
    assert np.array_equal(np.asarray(vec), dense[2]["evecs"])
    assert not bse_nontda.LAST_DAVIDSON                                      # no Davidson ran
    _, args = bse_jax.parse_args(["-i", _toy_deck(["full_bse = true"])])
    assert args.full_bse_solver == "dense" and args.full_bse_instability == "refuse"
    # through solve_bse_sharded the new route gives the route's own result
    ev_d, vec_d, n_it = bse_lanczos.solve_bse_sharded(
        _payload(mesh, ops), mesh, n_eig=N_EIG, tda=False, full_solver="davidson", max_iter=64)
    assert np.max(np.abs(np.asarray(ev_d) - dense[2]["omega"])) <= ROOT_TOL and int(n_it) > 0
    assert bse_nontda.LAST_DAVIDSON["info"]["iterations"] == int(n_it)


def _parse(lines):
    return bse_jax.parse_args(["-i", _toy_deck(lines)])[1]


def test_input_keys_and_what_they_refuse(capsys):
    full = ["full_bse = true", "full_bse_solver = davidson"]
    args = _parse(full + ["full_bse_instability = continue", "davidson_margin = 4", "davidson_m_max = 200",
                          "davidson_precond = bare", "davidson_eps_shift = 1e-4",
                          "davidson_margin_tol_mev = 1", "davidson_tol_mev = 0.005", "solver_max_iter = 50"])
    assert (args.tda, args.full_bse_solver, args.full_bse_instability) == (False, "davidson", "continue")
    assert (args.davidson_margin, args.davidson_m_max, args.davidson_precond) == (4, 200, "bare")
    refused = [
        (full + ["davidson_olsen = true"], "`davidson_olsen` is not read by full_bse_solver = davidson"),
        (full + ["trlan_m_max = 40"], "`trlan_m_max` is not read by full_bse_solver = davidson"),
        (full + ["lanczos_rtol = 1e-6"], "`lanczos_rtol` is not read by full_bse_solver = davidson"),
        (["full_bse = true", "full_bse_instability = continue"],
         "`full_bse_instability` is read by full_bse_solver = davidson only"),
        (["full_bse = true", "davidson_margin = 4"], "set full_bse_solver = davidson for the full-BSE Davidson"),
        (["full_bse_solver = davidson"], "`full_bse_solver` is read with full_bse = true only"),
        (["full_bse_instability = continue"], "`full_bse_instability` is read with full_bse = true only"),
        (["full_bse = true", "full_bse_solver = lanczos"], "takes one of dense, davidson"),
        (["feast = true", "full_bse = true", "full_bse_solver = davidson"],
         "`full_bse_solver` is not read by the feast route"),
        (full + ["haydock", "dipole_file = dipole.h5"], "it needs the TDA"),
    ]
    for lines, want in refused:
        with pytest.raises(SystemExit):
            _parse(lines)
        assert want in capsys.readouterr().err, lines
    # the keys are registered where the GW input reader looks
    from gw.gw_config import BSE_DECK_KEYS
    assert {"full_bse_solver", "full_bse_instability"} <= set(BSE_DECK_KEYS) & set(bse_jax.BSE_KEYS)


def test_settings_rows_and_size_refusal_name_the_solver_that_runs():
    def rows(lines):
        return {r.key: r for r in bse_jax.settings_rows(_parse(lines), source=lambda *keys: "deck")}
    dav = rows(["full_bse = true", "full_bse_solver = davidson"])
    assert dav["full_bse"].value == "on" and "solver" not in dav
    assert dav["full_bse_solver"].value == "davidson"
    assert "structure-preserving Davidson" in dav["full_bse_solver"].effect
    assert "paired as [D; -D]" in dav["full_bse_solver"].effect and "dense" not in dav["full_bse_solver"].value
    assert dav["davidson_tol_mev"].value == "0.01"
    assert dav["full_bse_instability"].value == "refuse" and "refuses" in dav["full_bse_instability"].effect
    cont = rows(["full_bse = true", "full_bse_solver = davidson", "full_bse_instability = continue"])
    assert "continues" in cont["full_bse_instability"].effect
    dense_rows = rows(["full_bse = true"])                                   # unchanged
    assert dense_rows["solver"].value == "dense" and "full_bse_solver" not in dense_rows
    assert "davidson_tol_mev" not in dense_rows and "full_bse_instability" not in dense_rows
    tda = rows([])                                                           # unchanged
    assert tda["solver"].value == "davidson" and "exact diagonal preconditioner" in tda["solver"].effect
    assert "full_bse_solver" not in tda and "full_bse_instability" not in tda
    # the dense size refusal names the route that has no size limit
    with pytest.raises(ValueError, match="_DENSE_N_MAX"):
        bse_nontda.refuse_dense_window(8, 8, 512, _mesh(1))
    # and the Davidson route is not stopped by it before the load
    loads = []

    def stub_loader(*a, **k):
        loads.append(1)
        raise KeyError("the stubbed loader was reached")

    real, real_find = bse_io.load_bse_data_from_restart_sharded, bse_jax._find_restart_file
    bse_io.load_bse_data_from_restart_sharded = stub_loader
    bse_jax._find_restart_file = lambda path: path
    try:
        with pytest.raises(ValueError, match="_DENSE_N_MAX"):
            bse_jax._preview_lanczos("toy.in", NV, NC, n_eig=N_EIG, tda=False, kgrid=(16, 16, 2))
        assert loads == []                                                   # the dense gate fires before it
        with pytest.raises(KeyError, match="the stubbed loader was reached"):
            bse_jax._preview_lanczos("toy.in", NV, NC, n_eig=N_EIG, tda=False, kgrid=(16, 16, 2),
                                     full_bse_solver="davidson")
        assert loads == [1]                                                  # no size gate on the Davidson
    finally:
        bse_io.load_bse_data_from_restart_sharded, bse_jax._find_restart_file = real, real_find


def _fake_wfn(path):
    with h5py.File(path, "w") as f:
        f["mf_header/versionnumber"] = np.int32(1)
        f["mf_header/crystal/celvol"] = 270.1
        k = f.create_group("mf_header/kpoints")
        k["nrk"], k["rk"] = np.int32(NK), np.zeros((NK, 3))
        k["el"], k["kgrid"] = np.zeros((1, NK, 8)), np.array([2, 2, 2], np.int32)
        k["nspinor"], k["nspin"] = np.int32(2), np.int32(1)


def _drive(tmp, mesh, ops, monkeypatch, lines, **kw):
    """``bse_jax._preview_lanczos`` in ``tmp`` with the loader, the restart lookup, the WFN path and
    the BerkeleyGW root files stubbed; returns the files written."""
    args = _parse(["full_bse = true", "write_eigenvectors = -1"] + lines)
    _fake_wfn(os.path.join(tmp, "WFN.h5"))
    monkeypatch.setattr(bse_io, "load_bse_data_from_restart_sharded", lambda restart, **k: _payload(mesh, ops))
    monkeypatch.setattr(bse_jax, "_find_restart_file", lambda path: path)
    monkeypatch.setattr(bse_jax, "create_mesh_2d", lambda: mesh)
    monkeypatch.setattr(bse_jax, "_write_all_roots", lambda *a, **k: ["eigenvalues.dat"])
    monkeypatch.setattr(bse_window, "_parse_wfn_path", lambda path: os.path.join(tmp, "WFN.h5"))
    monkeypatch.chdir(tmp)
    bse_jax._preview_lanczos(
        args.input, NV, NC, n_eig=N_EIG, write_eigs=args.write_eigs, tda=args.tda,
        davidson_tol_mev=args.davidson_tol_mev, davidson_margin=args.davidson_margin,
        davidson_m_max=args.davidson_m_max, davidson_precond=args.davidson_precond,
        full_bse_solver=args.full_bse_solver, full_bse_instability=args.full_bse_instability, **kw)
    return sorted(os.listdir(tmp))


def _read_pairs(path):
    """Eigenvalues (eV) and the ``X``, ``Y`` datasets as complex (n, nk, nc, nv) arrays."""
    with h5py.File(path, "r") as f:
        ev = f["exciton_data/eigenvalues"][()]
        X = f["exciton_data/eigenvectors"][()]
        Y = f["exciton_data/eigenvectors_deexcitation"][()]
        use_tda = int(f["exciton_header/params/use_tda"][()])
    return ev, (X[0, ..., 0, 0] + 1j * X[0, ..., 0, 1]), (Y[0, ..., 0, 0] + 1j * Y[0, ..., 0, 1]), use_tda


def test_driver_writes_both_halves_and_says_what_runs(ops, monkeypatch, tmp_path, caplog):
    mesh = _mesh(2)
    caplog.set_level(logging.INFO)
    out = {}
    for solver in ("davidson", "dense"):
        tmp = tmp_path / solver
        tmp.mkdir()
        caplog.clear()
        files = _drive(str(tmp), mesh, ops, monkeypatch, [f"full_bse_solver = {solver}"])
        assert "eigenvectors.h5" in files
        out[solver] = (*_read_pairs(str(tmp / "eigenvectors.h5")), " ".join(r.getMessage() for r in caplog.records))
    ev_dav, X, Y, use_tda, text = out["davidson"]
    ev_dense, Xd, Yd, _, text_dense = out["dense"]
    assert use_tda == 0 and X.shape == (N_EIG, NK, NC, NV) and Y.shape == X.shape
    assert np.max(np.abs(ev_dav - ev_dense)) <= ROOT_TOL * 13.6056980659
    flat = lambda a: a.reshape(N_EIG, -1)                                    # noqa: E731
    norm = np.sum(np.abs(flat(X)) ** 2, axis=1) - np.sum(np.abs(flat(Y)) ** 2, axis=1)
    assert np.max(np.abs(norm - 1.0)) <= 1e-10 and np.max(np.abs(Y)) > 1e-4   # Y is there, and normalised
    overlap = np.abs(np.sum(flat(Xd).conj() * flat(X), axis=1) - np.sum(flat(Yd).conj() * flat(Y), axis=1))
    assert np.max(np.abs(overlap - 1.0)) <= 1e-6                             # the same states as the dense file
    # the run header and the stop line name the solver that ran
    assert "Full BSE: structure-preserving Davidson on the 128-state window" in text
    assert "full-BSE Davidson converged: 8/8 roots with residual < 0.01 meV" in text
    assert "pair calls" in text and "dense structure-preserving solve" not in text
    assert "Full BSE: dense structure-preserving solve of the 128-state window" in text_dense
    assert "Davidson" not in text_dense.replace("the Davidson, Lanczos and TRLan settings are not read", "")
    line, rows = bse_jax._full_davidson_report()
    assert line.startswith("full-BSE Davidson converged") and rows[0][0] == "stability"
    assert "positive definite" in rows[0][1] and "NOT" not in rows[0][1] and len(rows) == 1


def test_capped_driver_run_refuses_before_writing(ops, monkeypatch, tmp_path):
    mesh = _mesh(2)
    with pytest.raises(RuntimeError, match="GATE nontda_davidson_converged"):
        _drive(str(tmp_path), mesh, ops, monkeypatch, ["full_bse_solver = davidson"], max_lanczos_iter=2)
    assert sorted(os.listdir(tmp_path)) == ["WFN.h5"]                        # nothing was written
