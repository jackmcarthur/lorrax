"""The on-site Breit sidecar generator (``psp.onsite_breit_sidecar``): round trip, null sidecar,
radial reduction, the four-spinor tensor against the spin-other-orbit tensor, refusals.

A CPU/unit cell (no driver runs).  Run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_onsite_breit_sidecar.py -q -p no:cacheprovider

Data (``tests/data/onsite_breit/``): the v2 sidecars ``Fe/Cr/Cl.onsite_breit.json``
(``make_v2_fixtures.py``); the all-electron radial tables ``Fe_3d_ae.dat``,
``Cr_3d_ae.dat``, ``Cl_3p_ae.dat``; the two CrCl3 UPFs (the pseudo functions and
projectors of Cr and Cl); ``v1/``: the earlier spin-other-orbit sidecars, whose tensors
were computed by a Gaussian fit of each radial function and analytic two-electron
integrals, and ``v1/reference_values.json`` (the post-processing code ``onsite_ops`` on
those tensors).  The Fe UPF is not shipped: the Fe pseudo function is the v1 file's own
projector (the plain j-average, which equals the aligned one for that file up to
normalisation).

The spin-other-orbit tensor ``shell_tensor`` is an INDEPENDENT construction kept in the
generator for these checks only: the four-spinor direct field must reproduce its d1 for
a pure spin source and the spin-dependent part of its d2 for a pure orbital source.  It
is checked here against the v1 files (fit-limited, ``ROUNDTRIP_RTOL``: max element
relative to max|Lam|; measured 2.7e-8 / 2.7e-4, 1.5e-6 / 1.7e-7, 4.3e-9 / 1.3e-6 for
Fe, Cr, Cl AE / PS) and its mean field against ``onsite_ops``.
"""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from psp import onsite_breit as ob  # noqa: E402
from psp import onsite_breit_sidecar as gen  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "onsite_breit")
UPF = {"Cr": os.path.join(DATA, "CrCl3_Cr.upf"), "Cl": os.path.join(DATA, "CrCl3_Cl.upf")}
AE = {"Fe": ("Fe_3d_ae.dat", "3D", 2), "Cr": ("Cr_3d_ae.dat", "3D", 2), "Cl": ("Cl_3p_ae.dat", "3P", 1)}
ROUNDTRIP_RTOL = {("Fe", "ae"): 1e-7, ("Fe", "ps"): 5e-4, ("Cr", "ae"): 5e-6, ("Cr", "ps"): 1e-6,
                  ("Cl", "ae"): 1e-7, ("Cl", "ps"): 5e-6}
ALPHA_REF = 1.0 / 137.035999084                  # onsite_ops (v1/reference_values.json)
SCALE_REF = (ALPHA_REF / ob.ALPHA) ** 2
HA_TO_MEV = ob.HA_TO_RY * ob.RY_TO_MEV
# N_L (L = 0..2l+1) and K of the normalised all-electron tables, by psp.onsite_breit_sidecar.radial_breit
# (computed 2026-11-18; the Cl values are cross-checked by brute-force Slater integrals in (e)).
PINNED_AE = {
    "Fe": ([0.8322733116240699, 0.5228637529023386, 0.3705253122595672, 0.28339096667090413,
            0.2280815735372457, 0.1902517594528042], 1.0946963154186873),
    "Cr": ([0.4348736290000259, 0.2689970737435238, 0.18920365470084352, 0.144144318453966,
            0.11575439735940532, 0.0964250097777085], 0.5528898457393364),
    "Cl": ([0.20855548560364764, 0.11251501457481475, 0.07868241872003391, 0.06094314286700329],
           0.2504733543723137),
}


def _p(name):
    return os.path.join(DATA, name)


def _lam(block):
    return np.asarray(block["lam_re"]) + 1j * np.asarray(block["lam_im"])


def _v1(el):
    with open(_p(f"v1/{el}.onsite_breit.json")) as f:
        return json.load(f)["shells"][0]


def _committed(el):
    with open(_p(f"{el}.onsite_breit.json")) as f:
        return json.load(f)["shells"][0]


def _fake_wfn():
    return SimpleNamespace(nspinor=2, atom_types=np.array([24, 17, 17]),
                           atom_crys=np.array([[0.0, 0.0, 0.0], [0.31, 0.12, 0.27], [0.69, 0.88, 0.73]]),
                           blat=2 * np.pi / 9.0, bvec=np.eye(3), cell_volume=9.0 ** 3, ecutwfc=20.0)


def _radial(el, kind):
    """Prepared (r, u) of a species' AE table or PS function (Cr, Cl: the UPF; Fe: the v1 projector)."""
    name, label, l = AE[el]
    if kind == "ae":
        r, u = gen.read_radial_table(_p(name))
    elif el == "Fe":
        pj = _v1("Fe")["projector"]
        r, u = np.asarray(pj["r"]), np.asarray(pj["chi"])
    else:
        sh = gen.upf_shell(UPF[el], label)
        r, u = sh["r"], sh["ps"]
    return gen.prepare_radial(r, u)[:2]


def _herm(g, n):
    A = g.normal(size=(n, n)) + 1j * g.normal(size=(n, n))
    return (A + A.conj().T) / 2


@pytest.fixture(scope="module")
def shells():
    """(el, kind) -> (l, r, u, N, K, G, Lam) for the AE and PS function of Fe, Cr, Cl."""
    out = {}
    for el, (_, _, l) in AE.items():
        for kind in ("ae", "ps"):
            r, u = _radial(el, kind)
            N, K = gen.radial_breit(r, u, l)
            out[(el, kind)] = (l, r, u, N, K, ob.shell_breit_tensor(l, N, K), gen.shell_tensor(r, u, l))
    return out


# ---------------------------------------------------------------- (a) round trip
def test_roundtrip_cli_and_committed_sidecars(tmp_path, shells):
    """The CLI rewrites the committed v2 files bit for bit (N, K, projector); the check tensor reproduces the
    v1 files' independent integrals (fit-limited); the files authenticate in a run's setup."""
    written = {}
    for el in ("Cr", "Cl"):
        name, label, l = AE[el]
        out = str(tmp_path / f"{el}.onsite_breit.json")
        assert gen.main(["--upf", UPF[el], "--shell", label.lower(), "--ae", _p(name), "-o", out]) == 0
        written[el] = out
        with open(out) as f:
            raw = json.load(f)
        assert raw["format"] == "lorrax.onsite_breit_sidecar/v2"
        new, old, v1 = raw["shells"][0], _committed(el), _v1(el)
        assert new["projector"]["r"] == v1["projector"]["r"] and new["projector"]["chi"] == v1["projector"]["chi"]
        assert new["projector"] == old["projector"] and new["l"] == l
        for kind in ("ae", "ps"):
            assert new[kind]["N"] == old[kind]["N"] and new[kind]["K"] == old[kind]["K"]
            assert "lam_re" not in new[kind] and len(new[kind]["N"]) == 2 * l + 2
            for k in ("x_closed_ls_mev", "x_closed_shift_mev", "b1_mev_per_muB", "I0", "r3"):
                assert new[kind][k] == old[kind][k], (el, kind, k)
            # r3, I0 against the v1 files (their trapezoid quadrature: <= 6e-5)
            assert abs(new[kind]["r3"] / v1[kind]["r3"] - 1) < 2e-5 and abs(new[kind]["I0"] / v1[kind]["I0"] - 1) < 1e-4
            # the reader's recomputed closed-shell coefficients equal the written ones
            rd = ob.read_sidecar(out)["shells"][0][kind]
            assert abs(rd["x_closed_ls_ha_check"] * HA_TO_MEV / new[kind]["x_closed_ls_mev"] - 1) < 1e-12
            assert abs(rd["x_closed_shift_ha_check"] * HA_TO_MEV / new[kind]["x_closed_shift_mev"] - 1) < 1e-12
    # the committed Fe v2 file holds the N, K of its two radial functions
    for kind in ("ae", "ps"):
        assert np.allclose(_committed("Fe")[kind]["N"], shells[("Fe", kind)][3], rtol=1e-14, atol=0)
    # the check tensor against the v1 files' Gaussian-fit integrals
    for (el, kind), rec in shells.items():
        lam = rec[6]
        lam0 = _lam(_v1(el)[kind]).reshape(lam.shape)
        rel = float(np.abs(lam - lam0).max() / np.abs(lam0).max())
        print(f"\n[a] {el} {kind}: max|Lam - Lam_v1| / max|Lam| = {rel:.2e} (tolerance {ROUNDTRIP_RTOL[(el, kind)]:g})")
        assert rel < ROUNDTRIP_RTOL[(el, kind)]
    setup = ob.build_onsite_setup(_fake_wfn(), UPF, written, projector="ortho-atomic",
                                  double_counting="paw", exchange="paw")
    assert setup.R_os == 10 + 2 * 6
    assert json.loads(setup.provenance)["scheme"] == "lorrax.onsite_breit/v2"


# ---------------------------------------------------------------- (b) null sidecar
def test_null_sidecar_is_an_exact_zero(tmp_path):
    out = str(tmp_path / "Cr.null.onsite_breit.json")
    assert gen.main(["--upf", UPF["Cr"], "--shell", "3D", "--null", "-o", out]) == 0
    with open(out) as f:
        raw = json.load(f)
    sh = raw["shells"][0]
    assert raw["provenance"]["null_gate"].startswith("Cr") and sh["ae"]["reference"].startswith("NULL GATE")
    assert all(sh["ae"][k] == sh["ps"][k] for k in ("N", "K", "I0", "r3", "x_closed_ls_mev", "x_closed_shift_mev"))
    assert sh["ps"]["N"] == _committed("Cr")["ps"]["N"]                        # its PS block is the full file's
    kern = ob.shell_kernels(ob.read_sidecar(out)["shells"][0])
    g = np.random.default_rng(4)
    A = g.normal(size=(10, 10)) + 1j * g.normal(size=(10, 10))
    D = A @ A.conj().T / 10
    for dc, ex in (("paw", "paw"), ("paw", "off")):
        pc = ob.onsite_potential(D, kern, double_counting=dc, exchange=ex)
        assert np.array_equal(pc["total"], np.zeros_like(pc["total"]))       # h_AE - h_PS = 0, bit for bit
        assert np.abs(pc["dir_ae"]).max() > 0                                  # the pieces themselves are not zero
    with pytest.raises(SystemExit):                                            # --null excludes --ae
        gen.main(["--upf", UPF["Cr"], "--shell", "3D", "--null", "--ae", _p(AE["Cr"][0])])


# ---------------------------------------------------------------- (c) identities, inputs and refusals
def test_block_checks_inputs_and_refusals(tmp_path):
    """verify_breit on l = 1, 2, 3; no fit: an exact function gives the same numbers on two grids;
    ONCVPSP plot blocks; l = 0 and bad inputs refuse; a corrupted generated file refuses."""
    r, u = gen.read_radial_table(_p(AE["Cr"][0]))
    for l in (1, 2, 3):
        blk, info = gen.tensor_block(r, u, l, "Cr 3d radial function as an l shell")
        print(f"\n[c] l = {l}: " + ", ".join(f"{k} {v:.1e}" for k, v in info.items() if isinstance(v, float)))
        assert info["pair_symmetry_rel"] < 1e-14 and info["hermiticity_rel"] < 1e-14
        assert info["closed_shell_dir_over_x"] < 1e-12 and info["closed_shell_resid"] < 1e-12
        # The two constructions integrate different (analytically equal) radial integrands; for the 3d
        # function used as an l = 3 shell the difference reaches 3e-10 (measured), inside SOURCE_RTOL.
        tol = 1e-12 if l < 3 else gen.SOURCE_RTOL
        assert info["orbital_source_vs_d2"] < tol and info["spin_source_vs_d1"] < tol
        assert abs(info["b1_tensor_over_formula_minus_1"]) < 1e-12 and abs(info["I0_marvin_minus_N1"]) < 1e-14
        assert info["x_closed_ls_ha"] > 0 and len(blk["N"]) == 2 * l + 2 and blk["I0"] == blk["N"][1]
    # u = r^3 exp(-r): a uniform grid and a log grid give the same N, K (both converged; measured 3.1e-9 at
    # N_5, falling at fourth order: 2.0e-10 and 1.2e-11 on grids refined twice and four times)
    rl = np.exp(np.linspace(np.log(1e-5), np.log(60.0), 2001))
    ru = np.linspace(0.0, 60.0, 6001)
    Nl, Kl = gen.radial_breit(*gen.prepare_radial(rl, rl ** 3 * np.exp(-rl))[:2], 2)
    Nu, Ku = gen.radial_breit(*gen.prepare_radial(ru, ru ** 3 * np.exp(-ru))[:2], 2)
    assert np.abs(Nl / Nu - 1).max() < 1e-8 and abs(Kl / Ku - 1) < 1e-10
    # u ~ r^3 exp(-r), normalised: I0 = N_1 = 11/960 in closed form
    # (gamma(7, x) = 6! [1 - exp(-x) sum_k<=6 x^k/k!]), r3 = <r^-3> = 1/15
    m = gen.radial_moments(*gen.prepare_radial(rl, rl ** 3 * np.exp(-rl))[:2])
    assert abs(m["I0"] * 960 / 11 - 1) < 1e-9 and abs(m["r3"] * 15 - 1) < 1e-9 and abs(Nl[1] * 960 / 11 - 1) < 1e-9
    # K = int u^4 / r^2 with u^2 = r^6 exp(-2r) / (6!/2^7): K = 10!/4^11 / (6!/2^7)^2
    assert abs(Kl / (3628800.0 / 4.0 ** 11 / (720.0 / 128.0) ** 2) - 1) < 1e-9
    # ONCVPSP plot blocks: both kappa blocks of (n, l), sign-aligned j-average; a test-configuration repeat is ignored
    rr = np.exp(np.linspace(np.log(0.01), np.log(6.0), 400))
    hi, lo = rr ** 3 * np.exp(-rr), -0.9 * rr ** 3 * np.exp(-1.1 * rr)
    lines = ["  some header", ""]
    for kap, f in ((-3, hi), (2, lo), (-3, 7 * hi)):
        lines += [f"n= 3,  l= 2  kap={kap:2d}, all-electron wave function, pseudo w-f", ""]
        lines += [f"&    -2  {a:12.6f}  {b:12.6f}  {0.0:12.6f}" for a, b in zip(rr, f)] + ["", ""]
    p = tmp_path / "fake.out"
    p.write_text("\n".join(lines) + "\n")
    r1, u1 = gen.read_oncvpsp_ae(str(p), 3, 2)
    six = np.vectorize(lambda x: float(f"{x:12.6f}"))
    r0 = six(rr)
    want = (3 * six(hi) - 2 * six(lo)) / 5
    assert np.array_equal(r1, r0) and np.abs(u1 - want).max() < 1e-15
    with pytest.raises(ValueError, match="no all-electron plot blocks"):
        gen.read_oncvpsp_ae(str(p), 4, 2)
    with pytest.raises(ValueError, match="1 <= l <= 3"):
        gen.tensor_block(r, u, 0, "s shell")
    with pytest.raises(ValueError, match="strictly increasing"):
        gen.prepare_radial(r[::-1], u[::-1])
    with pytest.raises(ValueError, match="labelled '4F'"):
        gen.upf_shell(UPF["Cr"], "4F")
    with pytest.raises(ValueError, match="all-electron radial function"):
        gen.make_sidecar(UPF["Cr"], "3D")
    # a generated file whose radial numbers are corrupted is refused by the reader on the re-read
    side = gen.make_sidecar(UPF["Cl"], "3p", ae=gen.read_radial_table(_p(AE["Cl"][0])), date="2026-01-01")
    bad = copy.deepcopy(side)
    bad["shells"][0]["ae"]["N"][1] *= 1.001
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):
        gen.write_sidecar(bad, str(tmp_path / "bad.json"))
    gen.write_sidecar(side, str(tmp_path / "ok.json"))


# ---------------------------------------------------------------- (d) the W_L formulas, analytic u'
def _brute_W(r, u, du, L):
    """W_L[a, b] = iint f_a(r1) f_b(r2) r_<^L / r_>^(L+1) by direct quadrature, f1 = u u' - u^2/r, f2 = u^2/r."""
    f = [u * du - u ** 2 * gen._pow(r, -1.0), u ** 2 * gen._pow(r, -1.0)]
    W = np.zeros((2, 2))
    for j in range(2):
        P = gen.radial_integral(r, f[j] * gen._pow(r, float(L)), cumulative=True)
        Q = gen.radial_integral(r, f[j] * gen._pow(r, -(L + 1.0)), cumulative=True)
        v = gen._pow(r, -(L + 1.0)) * P + gen._pow(r, float(L)) * (Q[-1] - Q)
        for i in range(2):
            W[i, j] = gen.radial_integral(r, f[i] * v)
    return W


def test_W_formulas_against_brute_force_hydrogenic():
    """Integration by parts: W_L of (N_L, K) equals the direct double integral with the analytic u'
    of u = r^(l+1) exp(-a r), every L = 0..2l+1, l = 1, 2, 3."""
    r = np.concatenate([[0.0], np.geomspace(1e-6, 60.0, 6000)])
    worst = 0.0
    for l, a in ((1, 1.3), (2, 1.7), (3, 2.1)):
        nrm = np.sqrt(gen.radial_integral(r, (r ** (l + 1) * np.exp(-a * r)) ** 2))
        u = r ** (l + 1) * np.exp(-a * r) / nrm
        du = ((l + 1) * r ** l - a * r ** (l + 1)) * np.exp(-a * r) / nrm
        N, K = gen.radial_breit(r, u, l)
        for L in range(2 * l + 2):
            Wf, Wb = ob.breit_radial_W(L, N, K), _brute_W(r, u, du, L)
            err = float(np.abs(Wb - Wf).max() / np.abs(Wf).max())
            worst = max(worst, err)
            assert np.allclose(Wf, Wf.T, rtol=0, atol=0)
    # measured 2.4e-9 (worst at l = 3; l <= 2: 3e-10): the brute-force reference's own floor, which does not
    # fall under refinement (5e-8, 2.4e-9, 8e-9 at 3000, 6000, 12000 points); a wrong formula is O(1)
    print(f"\n[d] worst relative W_L error (formula vs brute force) {worst:.1e}")
    assert worst < 1e-8


# ---------------------------------------------------------------- (e) pinned radial numbers
def test_pinned_radial_numbers_and_brute_force_cl():
    """N_L, K of the Fe/Cr/Cl all-electron tables (pinned), and the Cl W_L cross-checked by brute force
    with a cubic-spline u' (the tabulated function has no analytic derivative; measured 5e-9)."""
    for el, (Np, Kp) in PINNED_AE.items():
        r, u = _radial(el, "ae")
        N, K = gen.radial_breit(r, u, AE[el][2])
        print(f"\n[e] {el} AE: N_L = {np.array2string(N, precision=6)}, K = {K:.6f}")
        assert np.abs(N / np.array(Np) - 1).max() < 1e-12 and abs(K / Kp - 1) < 1e-12
        assert _committed(el)["ae"]["N"] == list(N) and _committed(el)["ae"]["K"] == K
    from scipy.interpolate import CubicSpline
    r, u = _radial("Cl", "ae")
    i = np.arange(len(r), dtype=float)
    du = CubicSpline(i, u)(i, 1) / CubicSpline(i, r)(i, 1)
    N, K = gen.radial_breit(r, u, 1)
    worst = max(float(np.abs(_brute_W(r, u, du, L) - ob.breit_radial_W(L, N, K)).max()
                      / np.abs(ob.breit_radial_W(L, N, K)).max()) for L in range(4))
    print(f"[e] Cl AE brute-force W_L worst relative error {worst:.1e}")
    assert worst < 1e-7


# ---------------------------------------------------------------- (f) four-spinor tensor vs spin-other-orbit tensor
def test_source_resolved_identities_against_spin_other_orbit(shells):
    """The one current-current integral contains the spin-other-orbit operator exactly:
    orbital source D = 1 (x) D0: the spin-dependent part of h_dir is that of d2;
    spin source D = sum_c sigma_c (x) D_c: the spin-independent part of h_dir is that of d1;
    closed shell: the l.s coefficient of h_x is that of x1 + x2.  Also reported: the shares of
    orbit-orbit (spin-independent, orbital source) and spin-spin (spin-dependent, spin source)."""
    worst = {"orb": 0.0, "spin": 0.0, "ls": 0.0}
    for (el, kind), (l, r, u, N, K, G, lam) in shells.items():
        n = 2 * l + 1
        g = np.random.default_rng(5)
        for _ in range(2):
            D0 = np.kron(np.eye(2), _herm(g, n))
            Ds = sum(np.kron(s, _herm(g, n)) for s in ob._SIGMA)
            hd = ob.breit_meanfield(G, D0)["dir"]
            new0, newc = gen._spin_parts(hd, n)
            _, oldc = gen._spin_parts(gen.meanfield_pieces(lam, D0)["d2"], n)
            orb = max(np.linalg.norm(a - b) for a, b in zip(newc, oldc)) / np.linalg.norm(hd)
            oo = np.linalg.norm(np.kron(np.eye(2), new0)) / np.linalg.norm(hd)
            hd = ob.breit_meanfield(G, Ds)["dir"]
            new0, newc = gen._spin_parts(hd, n)
            old0, _ = gen._spin_parts(gen.meanfield_pieces(lam, Ds)["d1"], n)
            spin = np.linalg.norm(new0 - old0) / np.linalg.norm(hd)
            ss = np.linalg.norm(sum(np.kron(s, x) for s, x in zip(ob._SIGMA, newc))) / np.linalg.norm(hd)
            worst["orb"], worst["spin"] = max(worst["orb"], orb), max(worst["spin"], spin)
        x = gen.meanfield_pieces(lam, np.eye(2 * n))
        ls = ob.shell_operators(l)["ls"]
        c_old = float((np.vdot(ls, x["x1"] + x["x2"]) / np.vdot(ls, ls)).real)
        c_new = ob.closed_shell_coefficients(G, l)["c"]
        worst["ls"] = max(worst["ls"], abs(c_new / c_old - 1))
        print(f"\n[f] {el} {kind}: orbital source vs d2 {orb:.1e}, spin source vs d1 {spin:.1e}, closed-shell l.s "
              f"{c_new / c_old - 1:+.1e}; orbit-orbit share {oo:.3f}, spin-spin share {ss:.3f}")
    assert worst["orb"] < 1e-12 and worst["spin"] < 1e-12 and worst["ls"] < 1e-11


# ---------------------------------------------------------------- (g) the check tensor against onsite_ops
def test_check_tensor_against_onsite_ops():
    """The spin-other-orbit pieces (gen.meanfield_pieces) of the v1 tensors against the post-processing code
    onsite_ops (v1/reference_values.json): closed shell, b1, the atomic limit and a non-collinear regression."""
    with open(_p("v1/reference_values.json")) as f:
        ref = json.load(f)
    for el in ("Fe", "Cr", "Cl"):
        s = _v1(el)
        l, n = s["l"], 2 * s["l"] + 1
        o = ob.shell_operators(l)
        for kind in ("ae", "ps"):
            lam = _lam(s[kind]).reshape((3,) + (n,) * 4)
            rv = ref["shells"][f"{el}.onsite_breit.json:{s['label']}:{kind}"]
            h = gen.meanfield_pieces(lam, np.eye(2 * n))
            c = float((np.vdot(o["ls"], h["x1"] + h["x2"]) / np.vdot(o["ls"], o["ls"])).real)
            assert abs(c * SCALE_REF / rv["x_closed_ha"] - 1) < 1e-10
            d1 = gen.meanfield_pieces(lam, np.diag([0.9] * n + [0.3] * n))["d1"]
            b = float((np.vdot(o["L"][2], d1) / np.vdot(o["L"][2], o["L"][2])).real) / (n * 0.6)
            assert abs(b * SCALE_REF / rv["b1_tensor_ha"] - 1) < 1e-10
    lz = ob._lmats(2)[2]
    Lz = ob.shell_operators(2)["L"][2]
    for fx, nup in (("Cr3plus_uhf3dup", 3), ("Fe2plus_uhf3dup", 5)):
        lam = _lam(_v1(fx)["ae"]).reshape((3,) + (5,) * 4)
        h = gen.meanfield_pieces(lam, np.diag([nup / 5] * 5 + [0.0] * 5))
        b = float((np.vdot(Lz, h["d1"]) / np.vdot(Lz, Lz)).real)
        lhd = 2.5 * 2 * float((np.vdot(lz, (h["x1"] + h["x2"])[:5, :5]) / np.vdot(lz, lz)).real)
        t6 = ref["t6"][fx]
        assert abs(b * SCALE_REF / t6["b_ha"] - 1) < 1e-10 and abs(lhd * SCALE_REF / t6["lhalf_dxi_ha"] - 1) < 1e-10
    rg = ref["regression"]
    D = np.array(rg["D_re"]) + 1j * np.array(rg["D_im"])
    h = gen.meanfield_pieces(_lam(_v1("Fe")["ae"]).reshape((3,) + (5,) * 4), D)
    errs = {k: float(np.linalg.norm(h[k] * SCALE_REF - (np.array(rg[f"{k}_re"]) + 1j * np.array(rg[f"{k}_im"])))
                     / np.linalg.norm(np.array(rg[f"{k}_re"]) + 1j * np.array(rg[f"{k}_im"]))) for k in h}
    print(f"\n[g] regression vs onsite_ops (Fe 3D AE, v1): " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    assert max(errs.values()) < 1e-10
