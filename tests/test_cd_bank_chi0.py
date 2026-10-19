"""CPU pins of the CD chi0 provider (``gw.cd.sigma.bank_chi0``): the shared-pole bank's own response
quadrature serving chi0 at the contour-deformation points, on an emulated four-device 2x2 mesh.

1. Wiring identity: ``bank_chi0`` on a shuffled chunk with repeats, spanning several rule groups, equals
   the bank's own contiguous-group evaluation (``response_support`` -> ``response_quadrature`` ->
   ``integrate_response_group`` value rows) of the same rule set.
2. Accuracy: ``bank_chi0`` at imaginary nodes i u and real knots x + i eta (ghost knots x < 0 included)
   agrees with the exact ordered Lehmann sum (``w_isdf.compute_chi0_direct_fractional(ordered=True)``,
   the provider CD used before; here only a reference) at the rule tolerance, for an insulator and a
   fractional-occupation metal, at cd_chi0_rule_tolerance 1e-8 and 1e-10.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; device-count logic only).
"""
import json
import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(r"""
    import json
    from types import SimpleNamespace
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from gw import response_bank as rb
    from gw.cd.sigma import bank_chi0
    from gw.w_isdf import compute_chi0_direct_fractional
    assert jax.device_count() == 4
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    put = lambda a, sp: jax.device_put(a, NamedSharding(mesh, sp))
    rng = np.random.default_rng(11)
    grid, nb, nbl, s, n, nocc = (2, 2, 1), 12, 10, 1, 8, 4
    nk = int(np.prod(grid))
    psi = (rng.standard_normal((nk, nb, s, n)) + 1j * rng.standard_normal((nk, nb, s, n))) / 3
    ev = np.concatenate([rng.uniform(-1.0, -0.2, (nk, nocc)), rng.uniform(0.2, 1.5, (nk, nb - nocc))], axis=1)
    e = np.sort(ev, axis=1)
    ijk = np.stack(np.unravel_index(np.arange(nk), grid), axis=-1)
    dif = (ijk[None, :, :] - ijk[:, None, :]) % np.asarray(grid)
    KMQ = np.ravel_multi_index((dif[..., 0], dif[..., 1], dif[..., 2]), grid).astype(np.int64)
    q_ids = np.array([0, 1, 3])
    knots = np.array([-0.1, -0.05, 0.0, 0.45, 0.9, 1.6])        # ghost knots x < 0 included
    # insulator: CD-like small Im z; metal: Im z >= 0.3 Ry, where the bank's metallic rule fits (at
    # Im z = 0.04 Ry its exponential fit refuses: GATE cd_chi0_rule).  The metal set repeats 0.3j
    # (node u = 0.3 and knot x = 0 at eta = 0.3): one evaluation serves both.
    points = dict(insulator=np.concatenate([1j * np.array([0.04, 0.3, 2.0]), knots + 0.05j]),
                  metal=np.concatenate([1j * np.array([0.3, 0.8, 2.0]), knots + 0.3j]))
    meta = SimpleNamespace(nk_tot=nk, nkx=grid[0], nky=grid[1], nkz=grid[2], kgrid=grid, nspinor=s, nspin=1,
                           b_id_4_chi_user=nbl, mu_basis=SimpleNamespace(n_packed=n, n_logical=n))

    def fixture(f):
        return SimpleNamespace(enk=put(e, P(None, None)), occ=put(f, P(None, None)), layout="face",
                               green_parent=None, psi_mun=put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC),
                               psi_nmu=put(psi, PSI_NMU_SPEC),
                               slices=SimpleNamespace(b0=0, b4_logical=nbl, nb_full=nb))

    def execute(kernel, args, stage, runtime_bytes=0):
        result = rb._compiled(kernel, args)(*args)
        return kernel.finish(args, result) if hasattr(kernel, "finish") else result

    out = {}
    for name in ("insulator", "metal"):
        f = np.zeros((nk, nb)); f[:, :nocc] = 1.0
        mu = 0.0
        if name == "metal":                       # band nocc crosses mu with fractional occupations
            e_m = e.copy(); e_m[:, nocc] = np.array([-0.15, -0.05, 0.05, 0.15])
            f[:, nocc] = np.array([0.9, 0.7, 0.3, 0.1])
            e_use = e_m
        else:
            e_use = e
        e_save = e.copy(); e[...] = e_use
        wfns = fixture(f)
        z = points[name]
        exact = np.asarray(compute_chi0_direct_fractional(
            wfns, z, meta, mesh, occupation_state=SimpleNamespace(f_kn=f, mu_ry=mu, smearing_family="step"),
            kminq_rows=KMQ[q_ids], nb_logical=nbl, ordered=True))
        scale = np.abs(exact).max(axis=(1, 2, 3))
        for tol in (1e-8, 1e-10):
            chi, note = bank_chi0(wfns, meta, mesh, z, q_ids=q_ids, mu_ry=mu, rule_tolerance=tol, slots=3,
                                  print_fn=lambda *a, **k: None)
            got = np.asarray(chi(z))
            err = float((np.abs(got - exact).max(axis=(1, 2, 3)) / scale).max())
            # identity: the bank's own contiguous groups (response_quadrature "groups") of the same rule set
            rules, rec = chi.rules, {"correlation_count": 0}
            ref = {}
            for g in rules["groups"]:
                carry = np.asarray(rb.integrate_response_group(wfns, meta, mesh, rules, g, q_ids=tuple(q_ids),
                                                               execute=execute, receipt=rec, ordered=True))
                for r, m in enumerate(g["members"]):
                    ref[complex(chi.points[int(m)])] = carry[2 * r]
            order = [7, 0, 4, 4, 8, 2, 5]               # shuffled, one repeat, several rule groups
            pick = np.asarray(chi(z[order]))
            ident = float(max(np.abs(pick[d] - ref[complex(z[i])]).max() / np.abs(ref[complex(z[i])]).max()
                              for d, i in enumerate(order)))
            out[f"{name}_{tol:.0e}"] = dict(err=err, identity=ident, groups=len(note["rule_groups"]), points=note["points"],
                                            nodes=note["rule_nodes"], passes=note["passes"],
                                            partial_at_mu=note["partial_at_mu"])
        e[...] = e_save
    print("CDBANKCHI0", json.dumps(out))
""")


def _run():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=900)
    assert run.returncode == 0 and "CDBANKCHI0" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
    line = [l for l in run.stdout.splitlines() if l.startswith("CDBANKCHI0")][-1]
    return json.loads(line.split(" ", 1)[1])


def test_cd_bank_chi0_identity_and_accuracy():
    out = _run()
    print(out)
    for key, row in out.items():
        tol = float(key.split("_")[1])
        assert row["identity"] <= 1e-13, (key, row)
        assert row["err"] <= 100 * tol, (key, row)
    assert out["metal_1e-08"]["partial_at_mu"] and not out["insulator_1e-08"]["partial_at_mu"]
    assert out["metal_1e-08"]["points"] == 8 and out["insulator_1e-08"]["points"] == 9, out
