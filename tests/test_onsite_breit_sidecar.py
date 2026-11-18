"""The on-site Breit sidecar generator (``psp.onsite_breit_sidecar``): round trip, null sidecar, identities.

A CPU/unit cell (no driver runs).  Run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest tests/test_onsite_breit_sidecar.py -q -p no:cacheprovider

Data (``tests/data/onsite_breit/``): the committed sidecars ``Fe/Cr/Cl.onsite_breit.json``,
whose tensors were computed by a Gaussian fit of each radial function and analytic
two-electron integrals; the all-electron radial tables ``Fe_3d_ae.dat``,
``Cr_3d_ae.dat``, ``Cl_3p_ae.dat`` those sidecars were made from; and the two CrCl3 UPFs
(the pseudo functions and projectors of Cr and Cl).  The Fe UPF is not shipped: the Fe
pseudo function is taken from the sidecar's own projector (the plain j-average, which
equals the aligned one for that file up to normalisation).

The generator integrates the radial functions directly (no fit), so the round trip is
limited by the committed files' fit error, measured per block (``ROUNDTRIP_RTOL``, max
element relative to max|Lam|; measured 2.7e-8 / 2.7e-4, 1.5e-6 / 1.7e-7, 4.3e-9 / 1.3e-6
for Fe, Cr, Cl AE / PS).  ``I0`` and ``I2`` of the committed files carry the trapezoid
error of their quadrature (up to 6e-5 relative); the module logs meV with
``common.units.RYD_TO_EV`` (3.6e-7 relative from the committed files' Rydberg).
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


def _p(name):
    return os.path.join(DATA, name)


def _lam(block):
    return np.asarray(block["lam_re"]) + 1j * np.asarray(block["lam_im"])


def _committed(el):
    with open(_p(f"{el}.onsite_breit.json")) as f:
        return json.load(f)["shells"][0]


def _fake_wfn():
    return SimpleNamespace(nspinor=2, atom_types=np.array([24, 17, 17]),
                           atom_crys=np.array([[0.0, 0.0, 0.0], [0.31, 0.12, 0.27], [0.69, 0.88, 0.73]]),
                           blat=2 * np.pi / 9.0, bvec=np.eye(3), cell_volume=9.0 ** 3, ecutwfc=20.0)


# ---------------------------------------------------------------- (a) round trip against the committed sidecars
def test_roundtrip_against_committed_sidecars(tmp_path):
    """Same radial inputs -> the committed tensors (fit-limited), the projector bit for bit; CLI path for Cr, Cl."""
    worst = {}
    written = {}
    for el in ("Cr", "Cl"):
        name, label, l = AE[el]
        out = str(tmp_path / f"{el}.onsite_breit.json")
        assert gen.main(["--upf", UPF[el], "--shell", label.lower(), "--ae", _p(name), "-o", out]) == 0
        written[el] = out
        new, old = ob.read_sidecar(out)["shells"][0], _committed(el)
        with open(out) as f:
            raw = json.load(f)["shells"][0]
        assert raw["projector"]["r"] == old["projector"]["r"] and raw["projector"]["chi"] == old["projector"]["chi"]
        assert raw["projector"]["construction"] == old["projector"]["construction"] and new["l"] == l
        for kind in ("ae", "ps"):
            worst[(el, kind)] = (new[kind]["lam"], old[kind])
    # Fe: AE from its table, PS from the committed projector (no Fe UPF in the test data)
    sc = _committed("Fe")
    name, label, l = AE["Fe"]
    for kind, (r, u) in (("ae", gen.read_radial_table(_p(name))),
                         ("ps", (np.asarray(sc["projector"]["r"]), np.asarray(sc["projector"]["chi"])))):
        blk, _ = gen.tensor_block(r, u, l, f"Fe {kind}")
        worst[("Fe", kind)] = (_lam(blk).reshape((3,) + (2 * l + 1,) * 4), sc[kind])
    for (el, kind), (lam, old) in worst.items():
        lam0 = _lam(old).reshape(lam.shape)
        rel = float(np.abs(lam - lam0).max() / np.abs(lam0).max())
        print(f"\n[a] {el} {kind}: max|Lam - Lam_committed| / max|Lam| = {rel:.2e} (tolerance {ROUNDTRIP_RTOL[(el, kind)]:g})")
        assert rel < ROUNDTRIP_RTOL[(el, kind)]
    # scalar summaries of the CLI-written files
    for el, path in written.items():
        new, old = ob.read_sidecar(path)["shells"][0], _committed(el)
        for kind in ("ae", "ps"):
            n, o = new[kind], old[kind]
            assert abs(n["r3"] / o["r3"] - 1) < 2e-5 and abs(n["I0"] / o["I0"] - 1) < 1e-4
            assert abs(n["b1_mev_per_muB"] / o["b1_mev_per_muB"] - 1) < 1e-4
            assert abs(n["x_closed_mev"] / o["x_closed_mev"] - 1) < 1e-5
            # the field the reader recomputes from Lam equals the written one
            assert abs(n["x_closed_ha_check"] * ob.HA_TO_RY * ob.RY_TO_MEV / n["x_closed_mev"] - 1) < 1e-12
    # the generated Cr + Cl sidecars authenticate against the UPFs in the run's setup
    setup = ob.build_onsite_setup(_fake_wfn(), UPF, written, projector="ortho-atomic",
                                  double_counting="paw", exchange="paw")
    assert setup.R_os == 10 + 2 * 6


# ---------------------------------------------------------------- (b) null sidecar
def test_null_sidecar_is_an_exact_zero(tmp_path):
    out = str(tmp_path / "Cr.null.onsite_breit.json")
    assert gen.main(["--upf", UPF["Cr"], "--shell", "3D", "--null", "-o", out]) == 0
    with open(out) as f:
        raw = json.load(f)
    sh = raw["shells"][0]
    assert raw["provenance"]["null_gate"].startswith("Cr") and sh["ae"]["reference"].startswith("NULL GATE")
    assert all(sh["ae"][k] == sh["ps"][k] for k in ("lam_re", "lam_im", "I0", "r3", "I2", "x_closed_mev"))
    full = gen.make_sidecar(UPF["Cr"], "3D", ae=gen.read_radial_table(_p(AE["Cr"][0])))
    assert sh["ps"]["lam_re"] == full["shells"][0]["ps"]["lam_re"]          # its PS block is the full file's
    kern = ob.shell_kernels(ob.read_sidecar(out)["shells"][0])
    g = np.random.default_rng(4)
    A = g.normal(size=(10, 10)) + 1j * g.normal(size=(10, 10))
    D = A @ A.conj().T / 10
    pc = ob.onsite_potential(D, kern, double_counting="paw", exchange="paw")
    assert np.array_equal(pc["total"], np.zeros_like(pc["total"]))           # h_AE - h_PS = 0, bit for bit
    with pytest.raises(SystemExit):                                            # --null excludes --ae
        gen.main(["--upf", UPF["Cr"], "--shell", "3D", "--null", "--ae", _p(AE["Cr"][0])])


# ---------------------------------------------------------------- (c) identities, inputs and refusals
def test_tensor_identities_inputs_and_refusals(tmp_path):
    """Closed shell (the reader's gate, here at 1e-12), classical limit, no fit: an exact function
    reproduces the tensor independently of the grid; ONCVPSP plot blocks; l = 0 and bad inputs refuse."""
    r, u = gen.read_radial_table(_p(AE["Cr"][0]))
    for l in (1, 2, 3):
        blk, info = gen.tensor_block(r, u, l, "Cr 3d radial function as an l shell")
        assert info["closed_shell_d_over_x"] < 1e-12 and info["closed_shell_ls_resid"] < 1e-12
        assert info["hermiticity_rel"] < 1e-12 and abs(info["b1_tensor_over_formula_minus_1"]) < 1e-12
        assert info["x_closed_ha"] > 0
    # u = r^3 exp(-r): a uniform grid and a log grid give the same tensor (both converged)
    rl = np.exp(np.linspace(np.log(1e-5), np.log(60.0), 2001))
    ru = np.linspace(0.0, 60.0, 6001)
    lam_l = gen.shell_tensor(*gen.prepare_radial(rl, rl ** 3 * np.exp(-rl))[:2], 2)
    lam_u = gen.shell_tensor(*gen.prepare_radial(ru, ru ** 3 * np.exp(-ru))[:2], 2)
    assert np.abs(lam_l - lam_u).max() < 1e-9 * np.abs(lam_l).max()
    # u ~ r^3 exp(-r), normalised: I0 = int u^2 r^-3 int_0^r u^2 = 11/960 in closed form
    # (gamma(7, x) = 6! [1 - exp(-x) sum_k<=6 x^k/k!]), r3 = <r^-3> = 1/15
    m = gen.radial_moments(*gen.prepare_radial(rl, rl ** 3 * np.exp(-rl))[:2])
    assert abs(m["I0"] * 960 / 11 - 1) < 1e-9 and abs(m["r3"] * 15 - 1) < 1e-9
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
    # a generated file whose tensor is corrupted is refused by the reader's closed-shell gate
    side = gen.make_sidecar(UPF["Cl"], "3p", ae=gen.read_radial_table(_p(AE["Cl"][0])), date="2026-01-01")
    bad = copy.deepcopy(side)
    i = int(np.argmax(np.abs(bad["shells"][0]["ae"]["lam_re"])))
    bad["shells"][0]["ae"]["lam_re"][i] *= 1.001
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):
        gen.write_sidecar(bad, str(tmp_path / "bad.json"))
    gen.write_sidecar(side, str(tmp_path / "ok.json"))
