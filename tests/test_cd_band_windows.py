"""CPU pins of the CD chi0 band windows (deck ``cd_chi0_band_windows = auto``; ``gw.cd.chi0_rules``
``band_windows`` / ``band_plan`` / ``stream_rules(band_window=)``, ``gw.cd.sigma.bank_chi0``).

1. Partition (numpy): on random insulating tables and several reaches, the windows' band ranges cover
   every (occupied, empty) band pair exactly once, the band-window weights sum to f u exactly, every
   remote window starts above max|Re z| + margin, refs[1] - refs[0] = lo; a metal refuses
   (GATE cd_chi0_band_windows).
2. Toy deck on an emulated four-device 2x2 mesh (semicore, valence, low/high empty bands; two knot
   windows whose reaches give different cuts, six band-window definitions): every band-window rule
   group certified at the tolerance; chi0 against the exact ordered Lehmann reference
   (``compute_chi0_direct_fractional(ordered=True)``, a reference only) at 1e-8; a saved file loads to
   the same chi bitwise; a windowed file refuses an ``off`` run and vice versa
   (GATE cd_chi0_rules_identity); the precompute CLI with ``--band-windows auto``: two partial builds
   merged into a file the loader accepts.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; device-count logic only).
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from gw.cd import chi0_rules as cr


def _toy_energies(rng, nk):
    """[nk, 12] Ry: semicore 0, valence 1-3, empty 4-8 (spaced), high empty 9-11; sorted by construction."""
    lo = np.array([-3.1, -1.0, -0.6, -0.3, 0.2, 0.45, 0.8, 1.1, 1.4, 3.0, 3.8, 4.8])
    return lo[None, :] + rng.uniform(0.0, 0.1, (nk, lo.size))


def test_band_windows_partition_and_cuts():
    rng = np.random.default_rng(5)
    checked = 0
    for trial in range(40):
        nk, nb = int(rng.integers(1, 6)), int(rng.integers(6, 30))
        nocc = int(rng.integers(1, nb - 1))
        gap = rng.uniform(0.05, 0.5)
        e = np.sort(np.concatenate([rng.uniform(-6.0, -gap / 2, (nk, nocc)),
                                    rng.uniform(gap / 2, 8.0, (nk, nb - nocc))], axis=1), axis=1)
        e[:, :nocc] = np.minimum(e[:, :nocc], -gap / 2)
        f = np.zeros((nk, nb)); f[:, :nocc] = rng.uniform(0.5, 1.0, (nk, nocc))
        u = np.zeros((nk, nb)); u[:, nocc:] = rng.uniform(0.5, 1.0, (nk, nb - nocc))
        for reach in (0.0, 0.3, 1.5, 4.0, 20.0):
            z = np.array([reach + 0.05j, -0.5 * reach + 0.05j])
            margin = float(rng.uniform(0.1, 1.0))
            wins = cr.band_windows(e, f, u, z, margin)
            seen = np.zeros((nb, nb), int)
            total = np.zeros((nk, nb, nb))
            for w in wins:
                (oa, ob), (ea, eb) = w["occ"], w["emp"]
                seen[oa:ob, ea:eb] += 1
                fw = np.zeros_like(f); fw[:, oa:ob] = f[:, oa:ob]
                uw = np.zeros_like(u); uw[:, ea:eb] = u[:, ea:eb]
                total += fw[:, :, None] * uw[:, None, :]
                assert w["refs"][1] - w["refs"][0] == w["lo"] and w["lo"] > 0 and w["hi"] > w["lo"], w
                if w["kind"] != "crossing":
                    assert w["lo"] > abs(z.real).max() + margin, (w, reach, margin)
            want = np.zeros((nb, nb), int); want[:nocc, nocc:] = 1
            assert np.array_equal(seen, want), (trial, reach, wins)
            assert np.array_equal(total, f[:, :, None] * u[:, None, :])
            checked += 1
    assert checked == 200
    # the toy deck of the device test: cuts at the designed bands
    e = _toy_energies(np.random.default_rng(0), 4)
    f = np.zeros_like(e); f[:, :4] = 1.0
    u = np.zeros_like(e); u[:, 4:] = 1.0
    a = cr.band_windows(e, f, u, np.array([-0.1 + 0.05j, 0.6 + 0.05j]), 0.3)
    b = cr.band_windows(e, f, u, np.array([0.9 + 0.05j, 1.6 + 0.05j]), 0.3)
    assert [(w["kind"], w["occ"], w["emp"]) for w in a] == [
        ("crossing", [2, 4], [4, 6]), ("remote_valence", [2, 4], [6, 12]), ("remote_semicore", [0, 2], [4, 12])], a
    assert [(w["kind"], w["occ"], w["emp"]) for w in b] == [
        ("crossing", [1, 4], [4, 9]), ("remote_valence", [1, 4], [9, 12]), ("remote_semicore", [0, 1], [4, 12])], b
    # a metal (a band both occupied and empty) refuses
    f2, u2 = f.copy(), u.copy(); f2[:, 4] = 0.3
    with pytest.raises(ValueError, match="GATE cd_chi0_band_windows"):
        cr.band_windows(e, f2, u2, np.array([0.6 + 0.05j]), 0.3)


SCRIPT = textwrap.dedent(r"""
    import json, os, subprocess, sys, tempfile, time
    from types import SimpleNamespace
    import numpy as np, jax
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
    grid, nb, s, n, nocc = (2, 2, 1), 12, 1, 8, 4
    nbl = nb
    nk = int(np.prod(grid))
    psi = (rng.standard_normal((nk, nb, s, n)) + 1j * rng.standard_normal((nk, nb, s, n))) / 3
    base = np.array([-3.1, -1.0, -0.6, -0.3, 0.2, 0.45, 0.8, 1.1, 1.4, 3.0, 3.8, 4.8])
    e0 = base[None, :] + np.random.default_rng(0).uniform(0.0, 0.1, (nk, nb))
    ijk = np.stack(np.unravel_index(np.arange(nk), grid), axis=-1)
    dif = (ijk[None, :, :] - ijk[:, None, :]) % np.asarray(grid)
    KMQ = np.ravel_multi_index((dif[..., 0], dif[..., 1], dif[..., 2]), grid).astype(np.int64)
    q_ids = np.array([0, 1, 3])
    meta = SimpleNamespace(nk_tot=nk, nkx=grid[0], nky=grid[1], nkz=grid[2], kgrid=grid, nspinor=s, nspin=1,
                           b_id_4_chi_user=nbl, mu_basis=SimpleNamespace(n_packed=n, n_logical=n))
    f = np.zeros((nk, nb)); f[:, :nocc] = 1.0
    fx = SimpleNamespace(enk=put(e0, P(None, None)), occ=put(f, P(None, None)), layout="face",
                         green_parent=None, psi_mun=put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC),
                         psi_nmu=put(psi, PSI_NMU_SPEC), slices=SimpleNamespace(b0=0, b4_logical=nbl, nb_full=nb))

    def exact(z):
        return np.asarray(compute_chi0_direct_fractional(
            fx, z, meta, mesh, occupation_state=SimpleNamespace(f_kn=f, mu_ry=0.0, smearing_family="step"),
            kminq_rows=KMQ[q_ids], nb_logical=nbl, ordered=True))

    def rel(got, ref):
        return float((np.abs(got - ref).max(axis=(1, 2, 3)) / np.abs(ref).max(axis=(1, 2, 3))).max())

    out = {}
    eta, margin, tol, cap = 0.05, 0.3, 1e-8, 192
    imag = [1j * np.array([0.04, 0.3, 2.0])]
    real = [np.array([-0.1, 0.3, 0.45, 0.6]) + 1j * eta, np.array([0.9, 1.2, 1.6]) + 1j * eta]
    points, windows = cr.window_members(imag, real)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "bw.pkl")
    kw = dict(q_ids=q_ids, mu_ry=0.0, rule_tolerance=tol, slots=4, windows=windows, rule_capacity=cap,
              print_fn=quiet)
    t0 = time.monotonic()
    chi, note = bank_chi0(fx, meta, mesh, points, save_path=path, band_windows=True, remote_margin_ry=margin, **kw)
    got = np.asarray(chi(points))
    wall = time.monotonic() - t0
    ref = exact(points)
    band = chi.payload["identity"]["band_windows"]
    rel_tol = tol / note["occupation_amplitude"]
    bw_groups = [g for g in chi.groups if g.get("band_window") is not None]
    per_point = {}
    for g in bw_groups:
        for m in g["members"]:
            per_point.setdefault(int(m), []).append(band["defs"][g["band_window"]]["kind"])
    knot_members = sorted(int(m) for w in windows if w["kind"] == "real" for m in w["members"])
    order = [5, 0, 7, 7, 9, 2]
    pick = np.asarray(chi(points[order]))
    out["toy"] = dict(
        err=rel(got, ref), err_knots=rel(got[knot_members], ref[knot_members]),
        err_imag=rel(got[[0, 1, 2]], ref[[0, 1, 2]]), shuffle=float(np.abs(pick - got[order]).max()),
        defs=[[d["kind"], d["occ"], d["emp"]] for d in band["defs"]], ranges=band["ranges"],
        providers=sorted({g.get("provider") for g in chi.groups}),
        bw_providers=sorted({g.get("provider") for g in bw_groups}),
        max_sampled=float(max(np.max(g["sampled_error"]) for g in bw_groups)), rel_tol=rel_tol,
        nodes={k: v for k, v in note["band_windows"]["nodes"].items()},
        windows_per_knot=sorted({len(v) for m, v in per_point.items()}),
        knots_all_windows=sorted(per_point) == knot_members, census=len(note["rule_groups"]),
        groups=len(chi.groups), census_labels=sorted({r["band_window"].split(":")[0] for r in note["rule_groups"]}),
        wall_s=wall)
    # file round trip, identity refusals both ways
    chi2, note2 = bank_chi0(fx, meta, mesh, points, rules_file=path, band_windows=True, remote_margin_ry=margin, **kw)
    out["file"] = dict(file_vs_build=float(np.abs(np.asarray(chi2(points)) - got).max()), source=note2["source"][:6])
    def refusal(**extra):
        try:
            bank_chi0(fx, meta, mesh, points, **extra, **kw)
            return "accepted"
        except ValueError as exc:
            return "GATE cd_chi0_rules_identity" if "GATE cd_chi0_rules_identity" in str(exc) else str(exc)[:200]
    out["file"]["off_run_on_windowed_file"] = refusal(rules_file=path)
    out["file"]["other_margin"] = refusal(rules_file=path, band_windows=True, remote_margin_ry=0.25)
    off_path = os.path.join(tmp, "off.pkl")
    np.save(os.path.join(tmp, "e.npy"), e0)
    cli = [sys.executable, "-m", "gw.cd.chi0_rules", "--energies", os.path.join(tmp, "e.npy"), "--nocc", str(nocc),
           "--nb-chi", str(nbl), "--b4-logical", str(nbl), "--eta-ev", "0.68", "--knot-step-ev", "1.36",
           "--imaginary-nodes", "6", "--knots", "5", "--ndev", "4", "--nq-irr", "3", "--nk", "4",
           "--knot-chunk", "4", "--tolerance", "1e-8", "--capacity", "512", "--band-windows", "auto",
           "--remote-margin-ev", str(margin * 13.6056980659)]
    runs, logs = [], []
    for i, sel in enumerate(("imag:0,real:0", "imag:1,real:1")):
        r = subprocess.run(cli + ["--windows", sel, "--out", os.path.join(tmp, f"part{i}.pkl")],
                           capture_output=True, text=True, timeout=900)
        runs.append(r.returncode)
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-3000:])
    m = subprocess.run([sys.executable, "-m", "gw.cd.chi0_rules", "--merge", os.path.join(tmp, "part0.pkl"),
                        os.path.join(tmp, "part1.pkl"), "--out", os.path.join(tmp, "merged.pkl")],
                       capture_output=True, text=True, timeout=300)
    if m.returncode:
        print(m.stdout[-2000:], m.stderr[-3000:])
    merged = cr.load_rules(os.path.join(tmp, "merged.pkl")) if m.returncode == 0 else None
    out["cli"] = dict(runs=runs, merge=m.returncode,
                      route=None if merged is None else merged["identity"]["route"],
                      defs=None if merged is None else len(merged["identity"]["band_windows"]["defs"]),
                      providers=None if merged is None else sorted({g.get("provider") for g in merged["rule_groups"]}),
                      windows=None if merged is None else len(merged["identity"]["windows"]))
    print("CDBANDWIN", json.dumps(out))
""")


def _run():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=2400)
    assert run.returncode == 0 and "CDBANDWIN" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
    line = [l for l in run.stdout.splitlines() if l.startswith("CDBANDWIN")][-1]
    return json.loads(line.split(" ", 1)[1])


def test_cd_band_windows_chi0_files_and_cli():
    out = _run()
    print(out)
    t = out["toy"]
    assert t["defs"] == [["crossing", [2, 4], [4, 6]], ["remote_valence", [2, 4], [6, 12]],
                         ["remote_semicore", [0, 2], [4, 12]], ["crossing", [1, 4], [4, 9]],
                         ["remote_valence", [1, 4], [9, 12]], ["remote_semicore", [0, 1], [4, 12]]], t
    assert t["ranges"] == {"crossing": [[1, 4], [4, 9]], "remote_valence": [[1, 4], [6, 12]],
                           "remote_semicore": [[0, 2], [4, 12]]}, t
    assert t["bw_providers"] == ["imaginary", "laplace"] and t["windows_per_knot"] == [3] and t["knots_all_windows"], t
    assert t["max_sampled"] <= t["rel_tol"] and t["census"] == t["groups"], t
    assert t["err"] <= 1e-6 and t["shuffle"] == 0.0, t
    assert set(t["census_labels"]) == {"all", "crossing", "remote_valence", "remote_semicore"}, t
    fl = out["file"]
    assert fl["file_vs_build"] == 0.0 and fl["source"] == "loaded", fl
    assert fl["off_run_on_windowed_file"] == "GATE cd_chi0_rules_identity", fl
    assert fl["other_margin"] == "GATE cd_chi0_rules_identity", fl
    c = out["cli"]
    assert c["runs"] == [0, 0] and c["merge"] == 0 and c["route"] == "cd-windowed" and c["windows"] == 4, c
    assert "imaginary" in c["providers"] and c["defs"] >= 1, c
