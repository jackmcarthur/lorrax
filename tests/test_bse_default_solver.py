"""The default lowest-N BSE solve and the one-process Haydock: five invariants the hsuite
cannot reach.

On a synthetic spinor payload (k grid 2x2x2, 4v x 4c, N_mu 24: transition dimension 128) in
the loader's shardings, the screened kernel real in R space (so H is Hermitian):

1. ``bse_jax``'s CLI defaults (``--solver davidson --davidson-precond exact
   --davidson-tol-mev 0.01``) through ``bse_lanczos.solve_bse_sharded`` solve 8 + 8 margin
   roots, stop CONVERGED on the lowest 8 and return those 8, within 0.01 meV of a full-Krylov
   Lanczos (120 of 128 steps) and of ``eigvalsh`` of the operator assembled from the same
   matvec.
2. ``GATE davidson_converged``: the driver (``bse_jax._preview_lanczos``, loader stubbed)
   capped at 2 iterations refuses, and its run directory stays empty.
3. ``keep_W_R``: ``absorption_haydock.haydock_coefficients`` on the eigensolve's payload
   equals the standalone route (a fresh payload) to 1e-12.
4. ``bse_jax --lanczos --dipole-position-fix`` parses and reaches the loader as
   ``velocity_position_fix=True``.
5. ``GATE cpu_davidson_threads`` refuses four emulated CPU devices on 4 host threads and
   passes them on 16.

A script on a 1x1 mesh (login node, <= 4 threads) or a 2x2 mesh of four host devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``, at least 16 host threads):
``JAX_PLATFORMS=cpu python -m tests.test_bse_default_solver``.  It skips under pytest.
"""
KGRID = (2, 2, 2)
NC = NV = 4
NS = 2
NMU = 24
N_EIG = 8
TOL_MEV = 0.01
RY_MEV = 13605.6980659


def _host_operands(seed=11):
    import numpy as np
    rng = np.random.default_rng(seed)
    nk = int(np.prod(KGRID))
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    pc = cplx(nk, NC, NS, NMU) / np.sqrt(NS * NMU)
    pv = cplx(nk, NV, NS, NMU) / np.sqrt(NS * NMU)
    ec = np.sort(0.40 + 0.08 * rng.random((nk, NC)), axis=1)        # Ry
    ev = -np.sort(0.06 * rng.random((nk, NV)), axis=1)
    W_R = 0.6 * rng.standard_normal((NMU, NMU) + KGRID)               # real in R: H Hermitian
    W_q = np.fft.fftn(W_R, axes=(2, 3, 4), norm="ortho")
    A = cplx(NMU, NMU) / NMU
    V = 0.4 * (A @ A.conj().T)
    d = cplx(3, NC, NV, nk)                                           # seed block (pol, c, v, k)
    return dict(pc=pc, pv=pv, ec=ec, ev=ev, W_q=W_q, V=V, d=d)


def _payload(mesh, ops):
    """Fresh device arrays in the loader's layout (W_q is donated by the solve)."""
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(ops["pc"]), ops["pv"])
    return dict(
        psi_c_X=put(ops["pc"], P(None, None, None, "x")), psi_c_Y=put(ops["pc"], P(None, None, None, "y")),
        psi_v_X=put(ops["pv"], P(None, None, None, "x")), psi_v_Y=put(ops["pv"], P(None, None, None, "y")),
        eps_c=put(ops["ec"], P(None, None)), eps_v=put(ops["ev"], P(None, None)),
        W_q=put(ops["W_q"], P("x", "y", None, None, None)), V_q0=put(ops["V"], P("x", "y")),
        M=put(M, P(None, "x", "y", None)),
        nkx=KGRID[0], nky=KGRID[1], nkz=KGRID[2], n_cond=NC, n_val=NV,
        n_cond_pad=NC, n_val_pad=NV, n_rmu=NMU, densified=False)


def _dense(mesh, ops):
    """H assembled column by column from the stack matvec (the operator both solvers see)."""
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from bse.absorption_haydock import ensure_W_R
    from bse.bse_stack_matvec import build_bse_stack_matvec
    data = _payload(mesh, ops)
    W_R = ensure_W_R(data, mesh)
    nk = int(np.prod(KGRID))
    n = NC * NV * nk
    eye = np.eye(n, dtype=complex).reshape(n, NC, NV, nk)
    mv = build_bse_stack_matvec(mesh, *KGRID, kernel="bse")
    X = device_put_process_local(eye, NamedSharding(mesh, P(None, "x", "y", None)))
    HX = mv(X, data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
            data["eps_c"], data["eps_v"], W_R, data["V_q0"], data["M"])
    return np.asarray(multihost_utils.process_allgather(HX, tiled=True)).reshape(n, n).T


def run_cases(mesh):
    import numpy as np
    from bse import bse_jax
    from bse.bse_lanczos import solve_bse_sharded
    from bse.absorption_haydock import haydock_coefficients
    from solvers.davidson import LAST_RUN
    from solvers.davidson_fixed import CONVERGED
    ops = _host_operands()
    H = _dense(mesh, ops)
    exact = np.linalg.eigvalsh(0.5 * (H + H.conj().T))[:N_EIG]
    p = bse_jax.build_parser()
    defaults = {k: p.get_default(k) for k in ("solver", "davidson_precond", "davidson_tol_mev")}

    data = _payload(mesh, ops)
    ev_d, _, it_d = solve_bse_sharded(
        data, mesh, n_eig=N_EIG, max_iter=200, solver_kind=defaults["solver"],
        davidson_precond=defaults["davidson_precond"],
        davidson_tol_Ry=defaults["davidson_tol_mev"] / RY_MEV, keep_W_R=True)
    ev_d = np.asarray(ev_d).real
    out = {"davidson_converged": int(LAST_RUN["status"]) == CONVERGED,
           "davidson_iterations": int(it_d), "davidson_matvecs": int(LAST_RUN["mv"][0]),
           "davidson_roots": f"{len(ev_d)}+{len(LAST_RUN['res'][0]) - len(ev_d)}"}
    ev_l, _, _ = solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG,
                                   max_iter=H.shape[0] - N_EIG, solver_kind="lanczos", block_size=1)
    ev_l = np.asarray(ev_l).real
    out["davidson_vs_lanczos_meV"] = float(np.max(np.abs(ev_d - ev_l)) * RY_MEV)
    out["davidson_vs_dense_meV"] = float(np.max(np.abs(ev_d - exact)) * RY_MEV)
    out["lanczos_vs_dense_meV"] = float(np.max(np.abs(ev_l - exact)) * RY_MEV)

    a1, b1, n1 = haydock_coefficients(data, mesh, ops["d"], 20)                # eigensolve's W_R
    a2, b2, n2 = haydock_coefficients(_payload(mesh, ops), mesh, ops["d"], 20)   # standalone
    out["haydock_combined_vs_standalone"] = float(max(np.max(np.abs(a1 - a2)), np.max(np.abs(b1 - b2)),
                                                      np.max(np.abs(n1 - n2))))
    out.update(_driver_cases(mesh, ops))
    return out


def _driver_cases(mesh, ops):
    """The driver path on the synthetic payload (loader and restart lookup stubbed): the
    parsed ``--lanczos --dipole-position-fix`` reaches the loader, and a Davidson capped
    at 2 iterations refuses before anything is written; then the CPU threads gate."""
    import os
    import tempfile
    from types import SimpleNamespace
    import jax
    import numpy as np
    from bse import bse_io, bse_jax, bse_lanczos
    _, args = bse_jax.parse_args(["-i", "toy.in", "--lanczos", "--tda", "--bse", "--write-eigs",
                                  "--dipole", "toy_dipole.h5", "--dipole-position-fix"])
    seen = {}

    def load(restart_file, **kw):
        seen.update(kw)
        return _payload(mesh, ops)

    stubs = [(bse_io, "load_bse_data_from_restart_sharded", load),
             (bse_jax, "_find_restart_file", lambda path: path), (bse_jax, "create_mesh_2d", lambda: mesh)]
    stubs = [(mod, name, getattr(mod, name), new) for mod, name, new in stubs]
    cwd, gate = os.getcwd(), ""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            for mod, name, _, new in stubs:
                setattr(mod, name, new)
            os.chdir(tmp)
            bse_jax._preview_lanczos(args.input, NV, NC, n_eig=N_EIG, write_eigs=args.write_eigs,
                                     dipole_file=args.dipole,
                                     dipole_position_fix=args.dipole_position_fix, max_lanczos_iter=2)
        except RuntimeError as exc:
            gate = str(exc)
        finally:
            os.chdir(cwd)
            for mod, name, old, _ in stubs:
                setattr(mod, name, old)
        written = sorted(os.listdir(tmp))
    fake = SimpleNamespace(devices=np.empty((2, 2), dtype=object))
    for i in range(4):
        fake.devices.flat[i] = SimpleNamespace(platform="cpu", process_index=jax.process_index())
    real = os.sched_getaffinity

    def threads_gate(n):
        os.sched_getaffinity = lambda pid: set(range(n))
        try:
            bse_lanczos._refuse_cpu_davidson_threads(fake)
            return ""
        except RuntimeError as exc:
            return str(exc)
        finally:
            os.sched_getaffinity = real
    return {"dipole_position_fix_reaches_loader": seen.get("velocity_file") == "toy_dipole.h5"
            and seen.get("velocity_position_fix") is True,
            "capped_davidson_refuses_unwritten": "GATE davidson_converged" in gate
            and "/2 iterations" in gate and written == [],
            "cpu_threads_gate": "GATE cpu_davidson_threads" in threads_gate(4) and threads_gate(16) == ""}


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    mesh = stack.mesh
    if tuple(mesh.devices.shape) not in ((1, 1), (2, 2)):
        print(f"SKIP: needs a 1x1 or 2x2 mesh, got {tuple(mesh.devices.shape)}", flush=True)
        runtime.finalize_process(0)
        return 0
    out = run_cases(mesh)
    checks = {
        f"1 Davidson (CLI defaults, {N_EIG} + 8 margin roots) CONVERGED, <= {TOL_MEV} meV "
        "from full-Krylov Lanczos and dense":
            out["davidson_converged"] and out["davidson_roots"] == f"{N_EIG}+8"
            and out["davidson_vs_lanczos_meV"] <= TOL_MEV
            and out["davidson_vs_dense_meV"] <= TOL_MEV and out["lanczos_vs_dense_meV"] <= 1e-6,
        "2 GATE davidson_converged refuses a run capped at 2 iterations, nothing written":
            out["capped_davidson_refuses_unwritten"],
        "3 Haydock on the eigensolve's payload == standalone (1e-12)":
            out["haydock_combined_vs_standalone"] <= 1e-12,
        "4 --lanczos --dipole-position-fix parsed and reaches the loader":
            out["dipole_position_fix_reaches_loader"],
        "5 GATE cpu_davidson_threads: 4 devices refuse on 4 threads, run on 16": out["cpu_threads_gate"],
    }
    ok = all(checks.values())
    if jax.process_index() == 0:
        for name, good in checks.items():
            print(f"{'ok  ' if good else 'FAIL'} {name}", flush=True)
        print(f"BSE_DEFAULT_SOLVER mesh {tuple(mesh.devices.shape)} " + ("PASS" if ok else "FAIL") + " " +
              " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()),
              flush=True)
    rc = 0 if ok else 1
    runtime.finalize_process(rc)
    return rc


def test_default_solver_runs_as_a_script():
    """Collected by pytest: the solver cases start their own runtime, so they run as a script."""
    import pytest
    pytest.skip("run as a script: python -m tests.test_bse_default_solver")


if __name__ == "__main__":
    raise SystemExit(main())
