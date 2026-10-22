"""``exciton_bands``' default per-Q solver is ``bse_lanczos``' Davidson: four invariants the
hsuite cannot reach.

On a synthetic spinor Q path (3 Q, k grid 2x2x1, 3v x 3c, N_mu 16: transition dimension 36 per
Q), the screened kernel real in R space (so every H_Q is Hermitian):

1. ``build_path_solver_davidson`` (the CLI default) gives each Q's lowest 2 within 0.01 meV of
   ``eigvalsh`` of H_Q assembled from the same stack matvec, as does the full-Krylov block
   Lanczos scan (``solver = block_lanczos``, 36 steps of width 1) to 1e-6 meV.
2. One planned Davidson serves the whole path (``solvers.davidson`` plan cache, one entry).
3. ``GATE davidson_converged``: the path capped at 2 iterations refuses at Q row 0, before the
   driver's write.
4. The deck keys: ``solver`` defaults to davidson, ``solver_max_iter`` to auto; ``block_size``
   refuses under davidson and parses under ``solver = block_lanczos``.

A script on a 1x1 mesh (login node, <= 4 threads):
``JAX_PLATFORMS=cpu python -m tests.test_exciton_bands_davidson``.  It skips under pytest.
"""
KGRID = (2, 2, 1)
NC = NV = 3
NS = 2
NMU = 16
NQ = 3
N_EIG = 2
TOL_MEV = 0.01
RY_MEV = 13605.6980659


def _host_operands(seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    nk = int(np.prod(KGRID))
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    pc = cplx(NQ, nk, NC, NS, NMU) / np.sqrt(NS * NMU)              # psi_c(k+Q)
    pv = cplx(nk, NV, NS, NMU) / np.sqrt(NS * NMU)
    ec = np.sort(0.40 + 0.08 * rng.random((NQ, nk, NC)), axis=2)     # Ry
    ev = -np.sort(0.06 * rng.random((nk, NV)), axis=1)
    W_R = 0.6 * rng.standard_normal((NMU, NMU) + KGRID)              # real in R: H Hermitian
    W_q = np.fft.fftn(W_R, axes=(2, 3, 4), norm="ortho")
    A = cplx(NQ, NMU, NMU) / NMU
    V = 0.4 * np.einsum("qab,qcb->qac", A, A.conj())                 # V_Q Hermitian, >= 0
    return dict(pc=pc, pv=pv, ec=ec, ev=ev, W_R=W_R.astype(complex), W_q=W_q, V=V)


def _device(mesh, ops):
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    return dict(
        cX=put(ops["pc"], P(None, None, None, None, "x")), cY=put(ops["pc"], P(None, None, None, None, "y")),
        ec=put(ops["ec"], P()), V=put(ops["V"], P(None, "x", "y")),
        vX=put(ops["pv"], P(None, None, None, "x")), vY=put(ops["pv"], P(None, None, None, "y")),
        ev=put(ops["ev"], P()), W_R=put(ops["W_R"], P("x", "y", None, None, None)),
        W_q0=put(ops["W_q"][:, :, 0, 0, 0], P("x", "y")))


def _dense_lowest(mesh, ops):
    """Each Q's lowest N_EIG of H_Q assembled column by column from the stack matvec."""
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from bse.bse_stack_matvec import build_bse_stack_matvec
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    nk = int(np.prod(KGRID))
    n = NC * NV * nk
    mv = build_bse_stack_matvec(mesh, *KGRID, kernel="bse")
    X = put(np.eye(n, dtype=complex).reshape(n, NC, NV, nk), P(None, "x", "y", None))
    out = []
    for q in range(NQ):
        M = np.einsum("kcsm,kvsm->kcvm", np.conj(ops["pc"][q]), ops["pv"])
        HX = mv(X, put(ops["pc"][q], P(None, None, None, "x")), put(ops["pc"][q], P(None, None, None, "y")),
                put(ops["pv"], P(None, None, None, "x")), put(ops["pv"], P(None, None, None, "y")),
                put(ops["ec"][q], P()), put(ops["ev"], P()), put(ops["W_R"], P("x", "y", None, None, None)),
                put(ops["V"][q], P("x", "y")), put(M, P(None, "x", "y", None)))
        H = np.asarray(multihost_utils.process_allgather(HX, tiled=True)).reshape(n, n).T
        out.append(np.linalg.eigvalsh(0.5 * (H + H.conj().T))[:N_EIG])
    return np.stack(out)


def run_cases(mesh):
    import numpy as np
    from bse import exciton_bands as exb
    import importlib
    dav = importlib.import_module("solvers.davidson")   # the package re-exports the function
    ops = _host_operands()
    d = _device(mesh, ops)
    exact = _dense_lowest(mesh, ops)
    path = (d["cX"], d["cY"], d["ec"], d["V"], d["vX"], d["vY"], d["ev"], d["W_R"])
    n_phys = NC * NV * int(np.prod(KGRID))
    out = {}

    plans0 = sum(1 for k in dav._PLANS if k[0][0] == "exciton_bands")
    p = exb.build_parser()
    a0 = _deck_args(p, [])
    solve = exb.build_path_solver_davidson(
        mesh, *KGRID, NC, NV, NC, NV, n_eig=N_EIG, max_iter=max(30, min(200, n_phys // 2)),
        tol_mev=a0.davidson_tol_mev, W_q0=d["W_q0"])
    ev_d = np.asarray(exb._gather_host(solve(*path)[0])).real
    out["exciton_plans"] = sum(1 for k in dav._PLANS if k[0][0] == "exciton_bands") - plans0
    lanczos = exb.build_path_solver(mesh, *KGRID, NC, NV, NC, NV, n_eig=N_EIG, block_size=1,
                                    max_iter=n_phys)
    ev_l = np.asarray(exb._gather_host(lanczos(*path)[0])).real
    out["davidson_vs_dense_meV"] = float(np.max(np.abs(ev_d - exact)) * RY_MEV)
    out["lanczos_vs_dense_meV"] = float(np.max(np.abs(ev_l - exact)) * RY_MEV)

    capped = exb.build_path_solver_davidson(mesh, *KGRID, NC, NV, NC, NV, n_eig=N_EIG, max_iter=2,
                                            tol_mev=TOL_MEV, W_q0=d["W_q0"])
    try:
        capped(*path)
        out["capped_refuses_at_row_0"] = False
    except RuntimeError as e:
        out["capped_refuses_at_row_0"] = ("GATE davidson_converged" in str(e)
                                          and "at Q row 0" in str(e))

    def refuses(lines):
        try:
            _deck_args(p, lines, check=True)
            return False
        except SystemExit:
            return True
    out["cli"] = (a0.solver == "davidson" and a0.max_iter is None
                  and refuses(["block_size = 4"])
                  and not refuses(["solver = block_lanczos", "block_size = 4"]))
    return out


def _deck_args(p, lines, check=False):
    """The driver's settings for a toy deck holding ``lines`` (BSE deck keys)."""
    import tempfile
    from bse import exciton_bands as exb
    fd = tempfile.NamedTemporaryFile("w", suffix=".in", delete=False)
    fd.write("[cohsex]\nsys_dim = 3\nnumber_val_bands_fine = 3\nnumber_cond_bands_fine = 3\n"
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
        f"1 Davidson per Q <= {TOL_MEV} meV from dense; full-Krylov Lanczos <= 1e-6 meV":
            out["davidson_vs_dense_meV"] <= TOL_MEV and out["lanczos_vs_dense_meV"] <= 1e-6,
        "2 one planned Davidson for the whole path": out["exciton_plans"] == 1,
        "3 GATE davidson_converged refuses a 2-iteration cap at Q row 0": out["capped_refuses_at_row_0"],
        "4 deck: davidson default, block_size refused unless solver = block_lanczos": out["cli"],
    }
    ok = all(checks.values())
    if jax.process_index() == 0:
        for name, good in checks.items():
            print(f"{'ok  ' if good else 'FAIL'} {name}", flush=True)
        print("EXCITON_BANDS_DAVIDSON " + ("PASS" if ok else "FAIL") + " " +
              " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()),
              flush=True)
    rc = 0 if ok else 1
    runtime.finalize_process(rc)
    return rc


def test_exciton_bands_davidson_runs_as_a_script():
    """Collected by pytest: the cases start their own runtime, so they run as a script."""
    import pytest
    pytest.skip("run as a script: python -m tests.test_exciton_bands_davidson")


if __name__ == "__main__":
    raise SystemExit(main())
