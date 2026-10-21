"""The non-TDA matrix-free route (``bse_nontda``, ``solver='matrixfree'``) stops on convergence
and refuses without it: three invariants nothing else reaches (the route is opt-in, and the
hsuite has no non-TDA row).

On a definite toy pencil (N = 160, A Hermitian with a clustered low edge, B complex symmetric,
K = [[A, B], [B*, A*]] positive definite; energies in Ry):

1. ``sdy_lanczos_eig(tol=0.01 meV)`` stops after fewer restart cycles than its cap with every
   root's ``resid / 2 omega`` below the tolerance, and its lowest 4 omega agree with the dense
   eigenvalues of [[A, B], [-B*, -A*]] to 0.01 meV.
2. The fixed-count route (``tol=None``) at the same cycle count returns the same omega (1e-12 Ry):
   the convergence-driven loop runs the existing restart program, it does not change it.
3. ``GATE nontda_converged`` (``bse_nontda.refuse_unconverged_sdy``): the cold cycle alone
   (cap 0) refuses; the converged run passes.

A script on a 1x1 mesh (login node, <= 4 threads):
``JAX_PLATFORMS=cpu python -m tests.test_bse_nontda_matrix_free``.  It skips under pytest.
"""
N = 160
N_EIG = 4
M_MAX, N_KEEP, CAP = 40, 14, 50
TOL_MEV = 0.01
RY_MEV = 13605.6980659


def _pencil(seed=5):
    import numpy as np
    rng = np.random.default_rng(seed)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    e = np.sort(np.concatenate([0.30 + 0.004 * rng.random(6), 0.31 + 0.6 * rng.random(N - 6)]))
    C = cplx(N, N) / N
    A = np.diag(e) + 0.02 * (C + C.conj().T)
    S = cplx(N, N) / N
    B = 0.02 * (S + S.T)
    H = np.block([[A, B], [-B.conj(), -A.conj()]])
    w = np.linalg.eigvals(H)
    exact = np.sort(w.real[w.real > 0])[:N_EIG]
    assert np.min(np.linalg.eigvalsh(np.block([[A, B], [B.conj(), A.conj()]]))) > 0
    return A, B, exact


def run_cases():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from solvers.bse_sp_lanczos import sdy_lanczos_eig
    from bse.bse_nontda import refuse_unconverged_sdy
    A, B, exact = _pencil()
    Aj, Bj = jnp.asarray(A), jnp.asarray(B)
    F = lambda z: z @ Aj.T + jnp.conj(z) @ Bj.T
    G = lambda z: z @ Aj.T - jnp.conj(z) @ Bj.T

    def solve(n_restarts, tol):
        f = jax.jit(lambda: sdy_lanczos_eig(F, G, (N,), n_eig=N_EIG, m_max=M_MAX, n_keep=N_KEEP,
                                            n_restarts=n_restarts, tol=tol))
        om, _, _, dg = f()
        return np.asarray(om), {k: np.asarray(v) for k, v in dg.items()}

    tol_ry = TOL_MEV / RY_MEV
    om, dg = solve(CAP, tol_ry)
    cycles = int(dg["cycles"])
    out = {"cycles": cycles,
           "max_resid_omega_meV": float(np.max(dg["resid_omega"]) * RY_MEV),
           "sdy_vs_dense_meV": float(np.max(np.abs(np.sort(om) - exact)) * RY_MEV)}
    om_fixed, _ = solve(cycles, None)
    out["tol_vs_fixed_Ry"] = float(np.max(np.abs(om - om_fixed)))
    _, dg0 = solve(0, tol_ry)
    try:
        refuse_unconverged_sdy(dg0, n_eig=N_EIG, tol_mev=TOL_MEV, n_restarts=0)
        out["gate"] = False
    except RuntimeError as e:
        out["gate"] = "GATE nontda_converged" in str(e)
    out["gate"] = out["gate"] and "converged" in refuse_unconverged_sdy(
        dg, n_eig=N_EIG, tol_mev=TOL_MEV, n_restarts=CAP)
    return out


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    if tuple(stack.mesh.devices.shape) != (1, 1):
        print(f"SKIP: needs a 1x1 mesh, got {tuple(stack.mesh.devices.shape)}", flush=True)
        runtime.finalize_process(0)
        return 0
    out = run_cases()
    checks = {
        f"1 convergence-driven SDY: 1 <= cycles < {CAP}, resid/2w < {TOL_MEV} meV, <= {TOL_MEV} meV from dense":
            1 <= out["cycles"] < CAP and out["max_resid_omega_meV"] < TOL_MEV
            and out["sdy_vs_dense_meV"] <= TOL_MEV,
        "2 fixed-count route at the same cycle count returns the same omega (1e-12 Ry)":
            out["tol_vs_fixed_Ry"] <= 1e-12,
        "3 GATE nontda_converged refuses the cold cycle alone, passes the converged run": out["gate"],
    }
    ok = all(checks.values())
    if jax.process_index() == 0:
        for name, good in checks.items():
            print(f"{'ok  ' if good else 'FAIL'} {name}", flush=True)
        print("BSE_NONTDA_MATRIX_FREE " + ("PASS" if ok else "FAIL") + " " +
              " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()),
              flush=True)
    rc = 0 if ok else 1
    runtime.finalize_process(rc)
    return rc


def test_nontda_matrix_free_runs_as_a_script():
    """Collected by pytest: the cases start their own runtime, so they run as a script."""
    import pytest
    pytest.skip("run as a script: python -m tests.test_bse_nontda_matrix_free")


if __name__ == "__main__":
    raise SystemExit(main())
