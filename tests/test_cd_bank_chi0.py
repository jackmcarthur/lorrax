"""CPU pins of the CD chi0 provider (``gw.cd.sigma.bank_chi0``, rules ``gw.cd.chi0_rules``): the
shared-pole bank's own response quadrature serving chi0 at the contour-deformation points, on an
emulated four-device 2x2 mesh.  The exact ordered Lehmann sum
(``w_isdf.compute_chi0_direct_fractional(ordered=True)``, the provider CD used before) is a reference only.

A. Bank route (``windows=None``, the bank's single request): wiring identity (a shuffled chunk with a
   repeat equals the bank's contiguous-group evaluation, bitwise) and accuracy at the rule tolerance,
   insulator and fractional-occupation metal, 1e-8 and 1e-10.
B. CD route (one request per memory window): the imaginary window on the ordered Laplace rule, the
   knot windows on the bank tree at the CD capacity and gate; accuracy at 1e-8; a saved rule file
   loads to the same chi bitwise; a file for another tolerance refuses (GATE cd_chi0_rules_identity).
C. Even-only Laplace rows (the small-u route on a time-reversal-authenticated deck): exact on a
   time-reversal-symmetric (real psi) fixture; the TR-broken error is reported.
D. Precompute CLI (``python -m gw.cd.chi0_rules``): two partial window builds merged into a file the
   loader accepts.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; device-count logic only).
"""
import json
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r"""
    import json, os, subprocess, sys, tempfile
    from types import SimpleNamespace
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from gw import response_bank as rb
    from gw.cd import chi0_rules as cr
    from gw.cd.sigma import bank_chi0
    from gw.w_isdf import compute_chi0_direct_fractional
    assert jax.device_count() == 4
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    put = lambda a, sp: jax.device_put(a, NamedSharding(mesh, sp))
    quiet = lambda *a, **k: None
    rng = np.random.default_rng(11)
    grid, nb, nbl, s, n, nocc = (2, 2, 1), 12, 10, 1, 8, 4
    nk = int(np.prod(grid))
    psi_c = (rng.standard_normal((nk, nb, s, n)) + 1j * rng.standard_normal((nk, nb, s, n))) / 3
    psi_r = rng.standard_normal((nk, nb, s, n)) / 3 + 0j          # real: time-reversal symmetric on this grid
    ev = np.concatenate([rng.uniform(-1.0, -0.2, (nk, nocc)), rng.uniform(0.2, 1.5, (nk, nb - nocc))], axis=1)
    e0 = np.sort(ev, axis=1)
    ijk = np.stack(np.unravel_index(np.arange(nk), grid), axis=-1)
    dif = (ijk[None, :, :] - ijk[:, None, :]) % np.asarray(grid)
    KMQ = np.ravel_multi_index((dif[..., 0], dif[..., 1], dif[..., 2]), grid).astype(np.int64)
    q_ids = np.array([0, 1, 3])
    meta = SimpleNamespace(nk_tot=nk, nkx=grid[0], nky=grid[1], nkz=grid[2], kgrid=grid, nspinor=s, nspin=1,
                           b_id_4_chi_user=nbl, mu_basis=SimpleNamespace(n_packed=n, n_logical=n))

    def fixture(e, f, psi):
        return SimpleNamespace(enk=put(e, P(None, None)), occ=put(f, P(None, None)), layout="face",
                               green_parent=None, psi_mun=put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC),
                               psi_nmu=put(psi, PSI_NMU_SPEC),
                               slices=SimpleNamespace(b0=0, b4_logical=nbl, nb_full=nb))

    def exact(wfns, f, z, mu=0.0):
        return np.asarray(compute_chi0_direct_fractional(
            wfns, z, meta, mesh, occupation_state=SimpleNamespace(f_kn=f, mu_ry=mu, smearing_family="step"),
            kminq_rows=KMQ[q_ids], nb_logical=nbl, ordered=True))

    def rel(got, ref):
        return float((np.abs(got - ref).max(axis=(1, 2, 3)) / np.abs(ref).max(axis=(1, 2, 3))).max())

    def execute(kernel, args, stage, runtime_bytes=0):
        result = rb._compiled(kernel, args)(*args)
        return kernel.finish(args, result) if hasattr(kernel, "finish") else result

    def contiguous(chi, slots):
        # the bank's own evaluation of chi's rule groups: contiguous slot slices (response_quadrature "groups")
        ref, rec = {}, {"correlation_count": 0}
        for g in chi.groups:
            for a in range(0, len(g["members"]), slots):
                sub = dict(g, members=list(g["members"])[a:a + slots],
                           **{k: np.asarray(g[k])[a:a + slots] for k in ("value", "derivative", "sampled_error",
                                                                          "coefficient_mass")})
                carry = np.asarray(rb.integrate_response_group(fx, meta, mesh, chi.rules, sub, q_ids=tuple(q_ids),
                                                               execute=execute, receipt=rec, ordered=True))
                for r, m in enumerate(sub["members"]):
                    ref[complex(chi.points[int(m)])] = carry[2 * r]
        return ref

    out = {}
    knots = np.array([-0.1, -0.05, 0.0, 0.45, 0.9, 1.6])
    # ---- A: bank route
    for name in ("insulator", "metal"):
        f = np.zeros((nk, nb)); f[:, :nocc] = 1.0
        e = e0.copy()
        if name == "metal":
            e[:, nocc] = np.array([-0.15, -0.05, 0.05, 0.15]); f[:, nocc] = np.array([0.9, 0.7, 0.3, 0.1])
            z = np.concatenate([1j * np.array([0.3, 0.8, 2.0]), knots + 0.3j])     # repeats 0.3j
        else:
            z = np.concatenate([1j * np.array([0.04, 0.3, 2.0]), knots + 0.05j])
        fx = fixture(e, f, psi_c)
        ref_exact = exact(fx, f, z)
        for tol in (1e-8, 1e-10):
            chi, note = bank_chi0(fx, meta, mesh, z, q_ids=q_ids, mu_ry=0.0, rule_tolerance=tol, slots=3,
                                  print_fn=quiet)
            err = rel(np.asarray(chi(z)), ref_exact)
            ref = contiguous(chi, 3)
            order = [7, 0, 4, 4, 8, 2, 5]
            pick = np.asarray(chi(z[order]))
            ident = float(max(np.abs(pick[d] - ref[complex(z[i])]).max() / np.abs(ref[complex(z[i])]).max()
                              for d, i in enumerate(order)))
            out[f"A_{name}_{tol:.0e}"] = dict(err=err, identity=ident, groups=len(chi.groups), points=note["points"],
                                              nodes=note["rule_nodes"], partial_at_mu=note["partial_at_mu"])
    # ---- B: CD route on windows, file round trip, identity refusal
    f = np.zeros((nk, nb)); f[:, :nocc] = 1.0
    fx = fixture(e0, f, psi_c)
    imag = [1j * np.array([0.04, 0.3, 2.0])]
    real = [knots[:4] + 0.05j, knots[4:] + 0.05j]
    points, windows = cr.window_members(imag, real)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "rules.pkl")
    chi, note = bank_chi0(fx, meta, mesh, points, q_ids=q_ids, mu_ry=0.0, rule_tolerance=1e-8, slots=4,
                          windows=windows, rule_capacity=192, save_path=path, print_fn=quiet)
    got = np.asarray(chi(points))
    providers = sorted({g.get("provider") for g in chi.groups})
    chi2, note2 = bank_chi0(fx, meta, mesh, points, q_ids=q_ids, mu_ry=0.0, rule_tolerance=1e-8, slots=4,
                            windows=windows, rule_capacity=192, rules_file=path, print_fn=quiet)
    same = float(np.abs(np.asarray(chi2(points)) - got).max())
    try:
        bank_chi0(fx, meta, mesh, points, q_ids=q_ids, mu_ry=0.0, rule_tolerance=1e-10, slots=4,
                  windows=windows, rule_capacity=192, rules_file=path, print_fn=quiet)
        refused = "accepted"
    except ValueError as exc:
        refused = "GATE cd_chi0_rules_identity" if "GATE cd_chi0_rules_identity" in str(exc) else str(exc)[:200]
    out["B"] = dict(err=rel(got, exact(fx, f, points)), providers=providers, groups=len(chi.groups),
                    census=len(note["rule_groups"]), file_vs_build=same, refusal=refused, source=note2["source"][:6])
    # ---- C: even-only Laplace rows, time-reversal-symmetric vs broken fixture
    zc = 1j * np.array([0.04, 0.3, 2.0])
    for name, psi in (("trs", psi_r), ("broken", psi_c)):
        fx = fixture(e0, f, psi)
        plan, _, _ = cr.sample_plan(fx, meta, zc, mu_ry=0.0, tolerance=1e-8)
        support = rb.response_support(fx, meta, plan, {}, print_fn=quiet)
        pid = cr.plan_identity(support, False)
        g = cr._laplace_group(zc, [0, 1, 2], pid, 1e-8, 384, ordered=False)
        rules = cr.stream_rules(dict(plan=pid, rule_groups=[g]), support, 3)
        carry = np.asarray(rb.integrate_response_group(fx, meta, mesh, rules, g, q_ids=tuple(q_ids),
                                                       execute=execute, receipt={"correlation_count": 0}, ordered=True))
        out[f"C_even_{name}"] = dict(err=rel(carry[0:6:2], exact(fx, f, zc)), nodes=int(g["count"]))
    # ---- D: precompute CLI, two partial builds merged
    np.save(os.path.join(tmp, "e.npy"), e0)
    base = [sys.executable, "-m", "gw.cd.chi0_rules", "--energies", os.path.join(tmp, "e.npy"), "--nocc", str(nocc),
            "--nb-chi", str(nbl), "--b4-logical", str(nbl), "--eta-ev", "0.68", "--knot-step-ev", "1.36",
            "--imaginary-nodes", "6", "--knots", "5", "--ndev", "4", "--nq-irr", "3", "--nk", "4",
            "--knot-chunk", "4", "--tolerance", "1e-8", "--capacity", "192"]
    runs = []
    for i, sel in enumerate(("imag:0,real:0", "imag:1,real:1")):
        r = subprocess.run(base + ["--windows", sel, "--out", os.path.join(tmp, f"part{i}.pkl")],
                           capture_output=True, text=True, timeout=600)
        runs.append(r.returncode)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:])
    m = subprocess.run([sys.executable, "-m", "gw.cd.chi0_rules", "--merge", os.path.join(tmp, "part0.pkl"),
                        os.path.join(tmp, "part1.pkl"), "--out", os.path.join(tmp, "merged.pkl")],
                       capture_output=True, text=True, timeout=300)
    merged = cr.load_rules(os.path.join(tmp, "merged.pkl")) if m.returncode == 0 else None
    try:
        cr.load_rules(os.path.join(tmp, "part0.pkl")); partial = "accepted"
    except ValueError as exc:
        partial = "refused" if "partial" in str(exc) else str(exc)[:120]
    out["D"] = dict(runs=runs, merge=m.returncode, groups=None if merged is None else len(merged["rule_groups"]),
                    windows=None if merged is None else len(merged["identity"]["windows"]),
                    providers=None if merged is None else sorted({g.get("provider") for g in merged["rule_groups"]}),
                    partial_load=partial)
    print("CDBANKCHI0", json.dumps(out))
""")


def _run():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=1500)
    assert run.returncode == 0 and "CDBANKCHI0" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
    line = [l for l in run.stdout.splitlines() if l.startswith("CDBANKCHI0")][-1]
    return json.loads(line.split(" ", 1)[1])


def test_cd_bank_chi0_routes_files_and_accuracy():
    out = _run()
    print(out)
    for key, row in out.items():
        if key.startswith("A_"):
            tol = float(key.split("_")[2])
            assert row["identity"] <= 1e-13 and row["err"] <= 100 * tol, (key, row)
    assert out["A_metal_1e-08"]["partial_at_mu"] and not out["A_insulator_1e-08"]["partial_at_mu"]
    assert out["A_metal_1e-08"]["points"] == 8 and out["A_insulator_1e-08"]["points"] == 9, out
    b = out["B"]
    assert b["err"] <= 1e-6 and b["providers"] == ["laplace", "pencil"] and b["census"] == b["groups"], b
    assert b["file_vs_build"] == 0.0 and b["refusal"] == "GATE cd_chi0_rules_identity" and b["source"] == "loaded", b
    assert out["C_even_trs"]["err"] <= 1e-6, out["C_even_trs"]
    d = out["D"]
    assert d["runs"] == [0, 0] and d["merge"] == 0 and d["partial_load"] == "refused" and d["windows"] == 4, d
    assert "pencil" in d["providers"] and "laplace" in d["providers"], d
