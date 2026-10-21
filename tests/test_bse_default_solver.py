"""The default lowest-N BSE solve (Davidson, exact diagonal, residual stop) reproduces
the Lanczos reference; the in-process Haydock reuses the eigensolve's payload.

On a synthetic spinor payload (k grid 2x2x2, 4v x 4c, N_mu 24: transition dimension
128) built on a 2x2 mesh with the loader's shardings, the screened kernel real in R
space (so H is Hermitian) and the exchange V positive:

* ``bse_jax``'s CLI defaults (``--solver davidson --davidson-precond exact
  --davidson-tol-mev 0.01``) through ``bse_lanczos.solve_bse_sharded`` give the lowest
  8 eigenvalues of the operator to 0.01 meV, against a full-Krylov single-vector
  Lanczos (120 of 128 steps, full CGS2) on the same payload and against ``eigvalsh`` of the
  operator assembled column by column from the same matvec; the Davidson stop is
  CONVERGED.  The control variant, the old fixed 30-step Lanczos default, misses the
  0.01 meV mark.  (The matvec count is not a criterion here: on a 128-dimensional toy a
  full Krylov space is cheap.  The Si 8^3 counts are in the track report.)
* ``keep_W_R``: a second solve on the kept ``W_R`` repeats the first bit for bit, and
  ``absorption_haydock.haydock_coefficients`` on the eigensolve's payload equals the
  standalone route (a fresh payload, ``W_q`` transformed there) to 1e-12, and the
  three-term recurrence of the dense operator to 1e-9.
* The solver-flag guard refuses a Lanczos-only flag under the default solver, and
  ``--haydock-iter`` refuses the RPA kernel (the recursion is D + V - W).
* The driver (``bse_jax._preview_lanczos`` on the synthetic payload, loader stubbed):
  ``--lanczos --dipole-position-fix`` parses and reaches the loader as
  ``velocity_position_fix=True``; a Davidson run capped at 2 iterations refuses with
  ``GATE davidson_converged`` and writes no file.  ``GATE cpu_davidson_threads`` refuses
  four emulated devices on 4 host threads and passes them on 16.

CPU, one process, on a 1x1 mesh (login node, <= 4 threads) or a 2x2 mesh of four host
devices: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4
python3 tests/test_bse_default_solver.py``.  The 2x2 case needs at least 16 host threads
(4 per device): the planned Davidson's host callbacks and the four devices' collective
rendezvous share the CPU client's pool, and at 4 threads an all-to-all waits until XLA's
rendezvous termination timeout (measured); ``GATE cpu_davidson_threads`` refuses that.  The host-only checks (``test_*``) run under pytest
anywhere; the mesh cases skip there.
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
    from bse.bse_preconditioner import compute_pair_amplitude
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(ops["pc"]), ops["pv"])
    return dict(
        psi_c_X=put(ops["pc"], P(None, None, None, "x")), psi_c_Y=put(ops["pc"], P(None, None, None, "y")),
        psi_v_X=put(ops["pv"], P(None, None, None, "x")), psi_v_Y=put(ops["pv"], P(None, None, None, "y")),
        eps_c=put(ops["ec"], P(None, None)), eps_v=put(ops["ev"], P(None, None)),
        W_q=put(ops["W_q"], P("x", "y", None, None, None)), V_q0=put(ops["V"], P("x", "y")),
        M=put(M, P(None, "x", "y", None)),
        nkx=KGRID[0], nky=KGRID[1], nkz=KGRID[2], n_cond=NC, n_val=NV,
        n_cond_pad=NC, n_val_pad=NV, densified=False)


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


def _haydock_dense(H, d, n_iter):
    """The three-term recurrence of absorption_haydock on a dense H (one seed per row)."""
    import numpy as np
    out_a, out_b = [], []
    for seed in d.reshape(d.shape[0], -1):
        s, sp, b = seed / np.linalg.norm(seed), np.zeros_like(seed), 0.0
        al, be = [], []
        for _ in range(n_iter):
            hs = H @ s
            a = np.vdot(s, hs).real
            r = hs - a * s - b * sp
            b = np.linalg.norm(r)
            sp, s = s, r / b
            al.append(a); be.append(b)
        out_a.append(al); out_b.append(be)
    return np.array(out_a), np.array(out_b)


def run_cases(mesh):
    import numpy as np
    from bse import bse_jax
    from bse.bse_lanczos import solve_bse_sharded
    from bse.absorption_haydock import haydock_coefficients
    from solvers.davidson import LAST_RUN
    from solvers.davidson_fixed import CONVERGED
    ops = _host_operands()
    H = _dense(mesh, ops)
    herm = float(np.max(np.abs(H - H.conj().T)))
    exact = np.linalg.eigvalsh(0.5 * (H + H.conj().T))[:N_EIG]
    n_phys = H.shape[0]
    p = bse_jax.build_parser()
    defaults = {k: p.get_default(k) for k in ("solver", "davidson_precond", "davidson_tol_mev")}
    out = {"defaults": defaults, "herm_Ry": herm}

    data = _payload(mesh, ops)
    ev_d, _, it_d = solve_bse_sharded(
        data, mesh, n_eig=N_EIG, max_iter=200, solver_kind=defaults["solver"],
        davidson_precond=defaults["davidson_precond"],
        davidson_tol_Ry=defaults["davidson_tol_mev"] / RY_MEV, keep_W_R=True)
    ev_d = np.asarray(ev_d).real
    out["davidson_status_converged"] = int(LAST_RUN["status"]) == CONVERGED
    out["davidson_matvecs"] = int(LAST_RUN["mv"][0])
    out["davidson_iterations"] = int(it_d)
    ev_d2, _, _ = solve_bse_sharded(
        data, mesh, n_eig=N_EIG, max_iter=200, solver_kind=defaults["solver"],
        davidson_precond=defaults["davidson_precond"],
        davidson_tol_Ry=defaults["davidson_tol_mev"] / RY_MEV)
    out["second_solve_on_kept_W_R_bitwise"] = bool(np.array_equal(np.asarray(ev_d2).real, ev_d))

    ev_l, _, _ = solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, max_iter=n_phys - N_EIG,
                                   solver_kind="lanczos", block_size=1)
    ev_l = np.asarray(ev_l).real
    ev_l30, _, _ = solve_bse_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, max_iter=30,
                                     solver_kind="lanczos", block_size=1)
    ev_l30 = np.asarray(ev_l30).real
    out["davidson_vs_lanczos_meV"] = float(np.max(np.abs(ev_d - ev_l)) * RY_MEV)
    out["davidson_vs_dense_meV"] = float(np.max(np.abs(ev_d - exact)) * RY_MEV)
    out["lanczos_vs_dense_meV"] = float(np.max(np.abs(ev_l - exact)) * RY_MEV)
    out["control_lanczos30_vs_dense_meV"] = float(np.max(np.abs(ev_l30 - exact)) * RY_MEV)

    n_iter = 20
    a1, b1, n1 = haydock_coefficients(data, mesh, ops["d"], n_iter)          # eigensolve's W_R
    a2, b2, n2 = haydock_coefficients(_payload(mesh, ops), mesh, ops["d"], n_iter)   # standalone
    ad, bd = _haydock_dense(H, ops["d"], n_iter)
    out["haydock_combined_vs_standalone"] = float(max(np.max(np.abs(a1 - a2)), np.max(np.abs(b1 - b2)),
                                                      np.max(np.abs(n1 - n2))))
    out["haydock_vs_dense"] = float(max(np.max(np.abs(a1 - ad)), np.max(np.abs(b1 - bd))))
    a3, _, _ = haydock_coefficients(_payload(mesh, ops), mesh, ops["d"][:1], n_iter)
    out["haydock_x_only_equals_row0"] = float(np.max(np.abs(a3[0] - a2[0])))

    from bse.absorption_haydock import parse_polarizations
    pols_ok = (parse_polarizations("xyz"), parse_polarizations("x"), parse_polarizations("zx")) == \
        ((0, 1, 2), (0,), (2, 0))
    for bad in ("", "xx", "xw", "b1"):
        try:
            parse_polarizations(bad)
            pols_ok = False
        except ValueError as exc:
            pols_ok = pols_ok and "GATE haydock_polarizations" in str(exc)
    out["haydock_polarizations_parse_and_refuse"] = pols_ok
    import contextlib
    import io
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            bse_jax.parse_args(["-i", "x.in", "--lanczos", "--tda", "--bse", "--lanczos-rtol", "1e-6"])
        out["lanczos_flag_refused_under_davidson"] = False
    except SystemExit:
        out["lanczos_flag_refused_under_davidson"] = "--lanczos-rtol is not read by --solver davidson" in err.getvalue()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            bse_jax.parse_args(["-i", "x.in", "--lanczos", "--tda", "--write-eigs", "--dipole", "d.h5",
                                "--haydock-iter", "10"])
        out["haydock_refused_without_bse"] = False
    except SystemExit:
        out["haydock_refused_without_bse"] = "it needs --bse" in err.getvalue()
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
    print(f"[driver] {gate}", flush=True)
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
            "capped_davidson_refuses": "GATE davidson_converged" in gate and "/2 iterations" in gate,
            "refusal_writes_nothing": written == [],
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
        "CLI defaults davidson/exact/0.01 meV": out["defaults"] == {
            "solver": "davidson", "davidson_precond": "exact", "davidson_tol_mev": TOL_MEV},
        "operator Hermitian (<1e-12 Ry)": out["herm_Ry"] < 1e-12,
        "Davidson CONVERGED": out["davidson_status_converged"],
        f"Davidson vs Lanczos <= {TOL_MEV} meV": out["davidson_vs_lanczos_meV"] <= TOL_MEV,
        f"Davidson vs dense <= {TOL_MEV} meV": out["davidson_vs_dense_meV"] <= TOL_MEV,
        "Lanczos reference vs dense <= 1e-6 meV": out["lanczos_vs_dense_meV"] <= 1e-6,
        f"control variant: 30-step Lanczos misses {TOL_MEV} meV": out["control_lanczos30_vs_dense_meV"] > TOL_MEV,
        "second solve on kept W_R bitwise": out["second_solve_on_kept_W_R_bitwise"],
        "Haydock on eigensolve payload == standalone (1e-12)": out["haydock_combined_vs_standalone"] <= 1e-12,
        "Haydock vs dense recurrence (1e-9)": out["haydock_vs_dense"] <= 1e-9,
        "Haydock x-only == row 0 of xyz (1e-12)": out["haydock_x_only_equals_row0"] <= 1e-12,
        "Lanczos-only flag refused under default solver": out["lanczos_flag_refused_under_davidson"],
        "Haydock polarizations parse and refuse": out["haydock_polarizations_parse_and_refuse"],
        "--haydock-iter refused on the RPA kernel": out["haydock_refused_without_bse"],
        "--lanczos --dipole-position-fix parsed and reaches the loader":
            out["dipole_position_fix_reaches_loader"],
        "GATE davidson_converged refuses a run capped at 2 iterations": out["capped_davidson_refuses"],
        "the refusal writes no file": out["refusal_writes_nothing"],
        "GATE cpu_davidson_threads: 4 devices refuse on 4 threads, run on 16": out["cpu_threads_gate"],
    }
    ok = all(checks.values())
    if jax.process_index() == 0:
        for name, good in checks.items():
            print(f"{'ok  ' if good else 'FAIL'} {name}", flush=True)
        print(f"BSE_DEFAULT_SOLVER mesh {tuple(mesh.devices.shape)} " + ("PASS" if ok else "FAIL") + " " +
              " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()
                       if k != "defaults"), flush=True)
    rc = 0 if ok else 1
    runtime.finalize_process(rc)
    return rc


def test_default_solver_runs_as_a_script():
    """Collected by pytest: the solver cases start their own runtime, so they run as a script."""
    import pytest
    pytest.skip("run as a script: python3 tests/test_bse_default_solver.py")


def test_lorentzian_blocks_equal_the_whole_kernel(monkeypatch):
    import numpy as np
    from bse import absorption_common as ac
    rng = np.random.default_rng(3)
    om, E, w = np.linspace(0, 1, 37), rng.random(101), rng.random((101, 3))
    L = (0.01 / np.pi) / ((om[:, None] - E[None, :]) ** 2 + 0.01 ** 2)
    monkeypatch.setattr(ac, "LORENTZ_BLOCK_ELEMENTS", 37 * 7)           # 15 blocks, ragged tail
    np.testing.assert_allclose(ac.lorentzian_broaden(om, E, w, 0.01), L @ w, rtol=1e-13)
    np.testing.assert_allclose(ac.lorentzian_broaden(om, E, w[:, 0], 0.01), L @ w[:, 0], rtol=1e-13)


def test_jdos_one_kernel_three_polarizations():
    import numpy as np
    from bse.absorption_common import jdos_from_transitions, lorentzian_broaden
    rng = np.random.default_rng(4)
    d = rng.standard_normal((3, 5, 2, 3)) + 1j * rng.standard_normal((3, 5, 2, 3))
    de = 0.3 + rng.random((5, 2, 3))
    om = np.linspace(0.0, 1.0, 50)
    got = jdos_from_transitions(d, de, om, 270.0, 5, 0.01, 1, 2)
    pref = 16.0 * np.pi ** 2 / (270.0 * 5 * 1 * 2)
    want = np.stack([pref * lorentzian_broaden(om, de.ravel(), np.abs(d[a]).ravel() ** 2, 0.01)
                     for a in range(3)], axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-13)


if __name__ == "__main__":
    raise SystemExit(main())
