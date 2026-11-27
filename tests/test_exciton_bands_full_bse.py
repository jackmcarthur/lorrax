"""``exciton_bands`` with ``full_bse = true``: the full BSE at every Q of a path.

A synthetic spinor path (3 Q, k grid 2x2x2, 4v x 4c, N_mu 24: 128 transitions per Q).  Each Q has
its own conduction and valence states at k + Q; the grid's states at k, the screened kernel (real in
R space with ``W_R[M, N, R] = W_R[N, M, -R]``) and the head shift are shared.  The reference is the
NumPy assembly of ``K = [[A, B], [B^H, A']]`` from the pair-function definition
(``tests/test_bse_full_finite_q_cpu.reference_K``) and ``numpy.linalg.eig`` of ``Sigma K``.

1. ``build_path_solver_davidson(full=dict(solver='dense'))`` gives each Q's lowest 4 roots of
   positive type within 1e-10 Ry of ``eig``; ``solver='davidson'`` within 0.01 meV, with one
   planned kernel set for the whole path and the stability margin of every Q at or above
   ``lam_min(K)``.
2. A Q whose ``K`` is not positive definite refuses naming the Q row (``GATE sp_davidson_definite``);
   ``on_indefinite='continue'`` returns the positive roots above the unstable one and counts it.
3. ``build_pair_stacks`` slices a joint-window bundle into the conduction and valence stacks, with
   the valence pad at ``-PAD_EPS_GUARD_RY``.
4. The deck keys: ``full_bse`` parses; ``full_bse_solver`` without it, ``solver = block_lanczos``
   with it, and the ``bse.bse_jax`` keys this driver cannot honour (``qp_rotations_file``,
   ``highest_occupied_band``) refuse.  The Davidson's tuning keys (``davidson_olsen``,
   ``davidson_margin``, ``davidson_eps_shift``, ``davidson_precond``, ``davidson_m_max``,
   ``davidson_margin_tol_mev``) are read and reach the solve (same roots, a block of n_eig + 4);
   they refuse under ``solver = block_lanczos`` and under the dense full solve.

A script on a 1x1 mesh: ``JAX_PLATFORMS=cpu python -m tests.test_exciton_bands_full_bse``.  It skips
under pytest (importing the driver starts the runtime).
"""
NQ = 3
N_EIG = 4
RY_MEV = 13605.6980659


def _path_operands():
    import numpy as np
    from tests import test_bse_full_finite_q_cpu as fq
    base = fq._operands(seed=3)
    per_q = [base] + [dict(base, **{k: fq._operands(seed=s)[k]
                                   for k in ("c_plus", "v_plus", "ec_plus", "ev_plus", "V")})
                      for s in (5, 9)]
    refs = [fq.reference_K(o) for o in per_q]
    return fq, np, base, per_q, refs


def _device(mesh, fq, np, base, per_q, *, shift=0.0):
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))  # noqa: E731
    stack = lambda key: np.stack([o[key] for o in per_q])                                    # noqa: E731
    x5, y5 = P(None, None, None, None, "x"), P(None, None, None, None, "y")
    x4, y4 = P(None, None, None, "x"), P(None, None, None, "y")
    head = fq.HEAD_RY
    path = (put(stack("c_plus"), x5), put(stack("c_plus"), y5), put(stack("ec_plus") - head - shift, P()),
            put(stack("V"), P(None, "x", "y")), put(base["v"], x4), put(base["v"], y4),
            put(base["ev"], P()), put(base["W_R"].astype(complex), P("x", "y", None, None, None)))
    partner = dict(psi_vQ_X=put(stack("v_plus"), x5), psi_vQ_Y=put(stack("v_plus"), y5),
                   eps_vQ=put(stack("ev_plus"), P()),
                   psi_c_X=put(base["c"], x4), psi_c_Y=put(base["c"], y4),
                   eps_c=put(base["ec"] - head - shift, P()))
    return path, partner, put(base["W_q"][:, :, 0, 0, 0], P("x", "y"))


def run_cases(mesh):
    from bse import exciton_bands as exb
    from solvers import bse_sp_davidson as spd
    from tests import bse_full_oracle as orc
    fq, np, base, per_q, refs = _path_operands()
    exact = [orc.solve_general_pencil(K) for K in refs]
    want = np.stack([e["omega"][:N_EIG] for e in exact])
    path, partner, W_q0 = _device(mesh, fq, np, base, per_q)
    out = {}
    geom = (mesh, *fq.KGRID, fq.NC, fq.NV, fq.NC, fq.NV)

    dense = exb.build_path_solver_davidson(*geom, n_eig=N_EIG, max_iter=200, tol_mev=0.01, W_q0=W_q0,
                                           full=dict(solver="dense", on_indefinite="refuse"))
    out["dense_vs_eig_ry"] = float(np.max(np.abs(
        np.asarray(exb._gather_host(dense(*path, partner=partner)[0])).real - want)))

    kernels0 = len(spd._KERNELS)
    dav = exb.build_path_solver_davidson(*geom, n_eig=N_EIG, max_iter=200, tol_mev=0.01, W_q0=W_q0,
                                         full=dict(solver="davidson", on_indefinite="refuse"))
    ev = np.asarray(exb._gather_host(dav(*path, partner=partner)[0])).real
    out["davidson_vs_eig_mev"] = float(np.max(np.abs(ev - want)) * RY_MEV)
    out["kernel_sets"] = len(spd._KERNELS) - kernels0
    stab = np.asarray(dav.stability)
    out["stability_rows"] = int(stab.shape[0])
    out["margin_above_lam_min"] = bool(np.all(stab[:, 0] >= np.array([e["k_min"] for e in exact]) - 1e-12)
                                       and np.all(stab[:, 1] == 0))

    # the Tamm-Dancoff solve of the same path is another answer (the coupling is not small here)
    tda = exb.build_path_solver_davidson(*geom, n_eig=N_EIG, max_iter=200, tol_mev=0.01, W_q0=W_q0)
    ev_tda = np.asarray(exb._gather_host(tda(*path)[0])).real
    out["tda_differs_mev"] = float(np.max(np.abs(ev_tda - want)) * RY_MEV)

    # 2. an unstable Q: lower every conduction energy until the softest K is just indefinite
    shift = min(e["k_min"] for e in exact) + 0.004
    K_un = [K - shift * np.eye(K.shape[0]) for K in refs]
    un = [orc.solve_general_pencil(K) for K in K_un]
    out["unstable_reference"] = [bool(not u["definite"]) for u in un]
    path_u, partner_u, _ = _device(mesh, fq, np, base, per_q, shift=shift)
    try:
        dav(*path_u, partner=partner_u)
        out["unstable_refuses"] = False
    except (RuntimeError, ValueError) as exc:
        out["unstable_refuses"] = "GATE sp_davidson_definite" in str(exc) and "Q row" in str(exc)
    cont = exb.build_path_solver_davidson(*geom, n_eig=N_EIG, max_iter=400, tol_mev=0.01, W_q0=W_q0,
                                          full=dict(solver="davidson", on_indefinite="continue"))
    try:
        ev_c = np.asarray(exb._gather_host(cont(*path_u, partner=partner_u)[0])).real
        want_c = np.stack([u["omega"][:N_EIG] for u in un])
        out["continue_vs_eig_mev"] = float(np.max(np.abs(ev_c - want_c)) * RY_MEV)
        got = np.asarray(cont.stability)[:, 1].astype(int).tolist()
        out["continue_special_counts"] = got
        out["continue_counts_match"] = got == [len(u["complex"]) + len(u["negative"]) for u in un]
    except (RuntimeError, ValueError) as exc:
        out["continue_vs_eig_mev"] = float("inf")
        out["continue_error"] = str(exc)[:300]
        out["continue_counts_match"] = False

    # 3. the joint-window bundle
    import types
    import jax.numpy as jnp
    nk, nb, pad_v, pad_c = fq.NK, fq.NV + fq.NC, fq.NV + 2, fq.NC + 1
    rng = np.random.default_rng(0)
    psi = rng.standard_normal((NQ * nk, nb, fq.NS, fq.NMU)) + 0j
    eps = rng.standard_normal((NQ * nk, nb))
    bundle = types.SimpleNamespace(psi_rmu_Y=jnp.asarray(psi), psi_rmuT_X=None, enk_full=jnp.asarray(eps),
                                   n_mu_logical=fq.NMU)
    cX, cY, ec, vX, vY, evq = exb.build_pair_stacks(bundle, NQ, nk, fq.NV, fq.NC, pad_v, pad_c,
                                                    fq.NMU, fq.NMU + 2, mesh)
    p5 = psi.reshape(NQ, nk, nb, fq.NS, fq.NMU)
    e3 = eps.reshape(NQ, nk, nb)
    out["pair_stacks"] = bool(
        np.array_equal(np.asarray(cX)[:, :, :fq.NC, :, :fq.NMU], p5[:, :, fq.NV:])
        and np.array_equal(np.asarray(vY)[:, :, :fq.NV, :, :fq.NMU], p5[:, :, :fq.NV])
        and np.all(np.asarray(vX)[:, :, fq.NV:] == 0) and np.all(np.asarray(cY)[..., fq.NMU:] == 0)
        and np.array_equal(np.asarray(ec)[:, :, :fq.NC], e3[:, :, fq.NV:])
        and np.array_equal(np.asarray(evq)[:, :, :fq.NV], e3[:, :, :fq.NV])
        and np.all(np.asarray(evq)[:, :, fq.NV:] == -exb.PAD_EPS_GUARD_RY)
        and np.all(np.asarray(ec)[:, :, fq.NC:] == exb.PAD_EPS_GUARD_RY)
        and cX.shape == (NQ, nk, pad_c, fq.NS, fq.NMU + 2) and vX.shape == (NQ, nk, pad_v, fq.NS, fq.NMU + 2))

    # 4. the deck keys
    p = exb.build_parser()

    def refuses(lines, check=False):
        try:
            _deck_args(p, lines, check=check)
            return False
        except SystemExit:
            return True
    a = _deck_args(p, ["full_bse = true", "full_bse_solver = davidson", "full_bse_instability = continue",
                       "davidson_olsen = true", "davidson_margin = 4", "davidson_eps_shift = 0",
                       "davidson_precond = bare", "davidson_m_max = 120", "davidson_margin_tol_mev = 1"])
    a0 = _deck_args(p, [])
    tuned = exb._davidson_tuning(a)
    # the tuning keys reach the solve: the same roots with Olsen, a bare diagonal and 4 margin roots
    tuned_solver = exb.build_path_solver_davidson(*geom, n_eig=N_EIG, max_iter=200, tol_mev=0.01, W_q0=W_q0,
                                                  full=dict(solver="davidson", on_indefinite="refuse"),
                                                  davidson=tuned)
    ev_t = np.asarray(exb._gather_host(tuned_solver(*path, partner=partner)[0])).real
    out["tuned_vs_eig_mev"] = float(np.max(np.abs(ev_t - want)) * RY_MEV)
    from bse import bse_nontda
    out["tuned_block"] = int(bse_nontda.LAST_DAVIDSON["info"]["n_block"])
    out["deck"] = bool(
        a.full_bse and a.full_solver == "davidson" and a.full_instability == "continue"
        and tuned == dict(davidson_precond="bare", davidson_olsen=True, davidson_eps_shift_Ry=0.0,
                          davidson_margin=4, davidson_margin_tol_Ry=1.0 / RY_MEV, davidson_m_max=120)
        and exb._davidson_tuning(a0) == dict(davidson_precond="exact", davidson_olsen=False,
                                             davidson_eps_shift_Ry=1e-3, davidson_margin=8,
                                             davidson_margin_tol_Ry=None, davidson_m_max=None)
        and out["tuned_vs_eig_mev"] <= 0.01 and out["tuned_block"] == N_EIG + 4
        and refuses(["solver = block_lanczos", "davidson_olsen = true"], check=True)
        and refuses(["full_bse = true", "davidson_margin = 4"], check=True)       # dense reads none
        and not a0.full_bse and a0.full_solver == "dense"
        and refuses(["full_bse_solver = davidson"])
        and refuses(["full_bse = true", "solver = block_lanczos"])
        and refuses(["full_bse = true", "full_bse_instability = continue"])      # dense has no such option
        and refuses(["qp_rotations_file = rot.h5"]) and refuses(["highest_occupied_band = 4"])
        and not refuses(["full_bse = true"]) and not refuses(["haydock = true"]))
    return out


def _deck_args(p, lines, check=False):
    import tempfile
    from bse import exciton_bands as exb
    fd = tempfile.NamedTemporaryFile("w", suffix=".in", delete=False)
    fd.write("[cohsex]\nsys_dim = 3\nnval = 5\nncond = 5\ncentroids_file = centroids_frac.txt\n"
             "number_val_bands_fine = 3\nnumber_cond_bands_fine = 3\n"
             + "".join(f"{line}\n" for line in lines))
    fd.close()
    args = p.parse_args(["-i", fd.name])
    settings_ap = exb.deck_settings(p, args)
    if check:
        exb.check_solver_flags(settings_ap, args)
    return args


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    mesh = stack.mesh
    if tuple(mesh.devices.shape) != (1, 1):
        print(f"SKIP: needs a 1x1 mesh, got {tuple(mesh.devices.shape)}", flush=True)
        runtime.finalize_process(0)
        return 0
    out = run_cases(mesh)
    checks = {
        "1a dense full BSE per Q <= 1e-10 Ry from eig of the definition": out["dense_vs_eig_ry"] <= 1e-10,
        "1b Davidson full BSE per Q <= 0.01 meV from eig; one kernel set for the path":
            out["davidson_vs_eig_mev"] <= 0.01 and out["kernel_sets"] == 1,
        "1c one stability row per Q, each margin >= lam_min(K)":
            out["stability_rows"] == NQ and out["margin_above_lam_min"],
        "1d the Tamm-Dancoff path is a different answer (> 0.1 meV)": out["tda_differs_mev"] > 0.1,
        "2a an indefinite K refuses naming the Q row": any(out["unstable_reference"]) and out["unstable_refuses"],
        "2b continue returns the positive roots above the special ones (<= 0.01 meV) and counts them":
            out["continue_vs_eig_mev"] <= 0.01 and out["continue_counts_match"],
        "3 build_pair_stacks slices and pads the joint window": out["pair_stacks"],
        "4 deck keys: full_bse parses; foreign and unserved keys refuse": out["deck"],
    }
    ok = all(checks.values())
    if jax.process_index() == 0:
        for name, good in checks.items():
            print(f"{'ok  ' if good else 'FAIL'} {name}", flush=True)
        print("EXCITON_BANDS_FULL_BSE " + ("PASS" if ok else "FAIL") + " " +
              " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()),
              flush=True)
    rc = 0 if ok else 1
    runtime.finalize_process(rc)
    return rc


def test_exciton_bands_full_bse_is_a_script():
    import pytest
    pytest.skip("run as a script: python -m tests.test_exciton_bands_full_bse")


if __name__ == "__main__":
    raise SystemExit(main())
