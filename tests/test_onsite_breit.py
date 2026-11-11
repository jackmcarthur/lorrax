"""On-site Breit operator (``psp.onsite_breit``): identities, limits, refusals, kernels.

A CPU/unit cell, exempt from the four-GPU rule (no driver runs; the band
kernels are checked on a single-device mesh, and on an emulated 2x2 host mesh
when ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` provides four CPU
devices).  Run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_THREAD_TIMEOUT=1 \\
        python -m pytest tests/test_onsite_breit.py -q -p no:cacheprovider

Data (``tests/data/onsite_breit/``): the species sidecars ``Fe/Cr/Cl.onsite_breit.json``
and the atomic-limit fixtures written by ``breit_salvage/onsite_sidecar/make_sidecar.py``
(sha256 in its README), ``reference_values.json`` (numbers of the validated
post-processing code ``onsite_ops`` on the same tensors, with recipes), and the
two CrCl3 UPFs the Cr/Cl sidecars authenticate against.

The reference numbers were computed with alpha = 1/137.035999084; the module
uses the repo's ``common.bispinor_init.ALPHA_FS`` (2 x 0.00364867628215), which
differs by 6.9e-10 relative.  Comparisons at 1e-10 rescale by (alpha_ref/alpha)^2.
"""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from psp import onsite_breit as ob  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "onsite_breit")
ALPHA_REF = 1.0 / 137.035999084
SCALE_REF = (ALPHA_REF / ob.ALPHA) ** 2          # module value -> reference value
HA_TO_MEV = ob.HA_TO_RY * ob.RY_TO_MEV


def _path(name):
    return os.path.join(DATA, name)


@pytest.fixture(scope="module")
def sc():
    return {el: ob.read_sidecar(_path(f"{el}.onsite_breit.json")) for el in ("Fe", "Cr", "Cl")}


@pytest.fixture(scope="module")
def ref():
    with open(_path("reference_values.json")) as f:
        return json.load(f)


def _tensors(sc):
    """(name, l, lam) of every AE and PS tensor of the three species."""
    return [(f"{el} {s['label']} {k}", s["l"], s[k]["lam"])
            for el, d in sc.items() for s in d["shells"] for k in ("ae", "ps")]


def _herm(n, seed):
    g = np.random.default_rng(seed)
    A = g.normal(size=(n, n)) + 1j * g.normal(size=(n, n))
    return A @ A.conj().T / n


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _coef(op, M):
    return float((np.vdot(op, M) / np.vdot(op, op)).real)


def _rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q * np.linalg.det(q)


def _all_pieces(lam, D):
    h = ob.meanfield_pieces(lam, D)
    Kd, Kx = ob.meanfield_kernels(lam)
    h["dir"] = np.einsum("prsq,sq->pr", Kd, D)
    h["x"] = np.einsum("prsq,sq->pr", Kx, D)
    return h


# ---------------------------------------------------------------- (a) closed shell
def test_closed_shell_limit(sc, ref):
    """D = 1: d1 = d2 = 0, x1 + x2 = c l.s (a pure delta-xi), c = the sidecar's x_closed_mev, c > 0."""
    worst_d, worst_res = 0.0, 0.0
    for el, d in sc.items():
        for s in d["shells"]:
            l, n2 = s["l"], 2 * (2 * s["l"] + 1)
            ls = ob.shell_operators(l)["ls"]
            kern = ob.shell_kernels(s)
            for kind in ("ae", "ps"):
                h = ob.meanfield_pieces(s[kind]["lam"], np.eye(n2))
                x = h["x1"] + h["x2"]
                c = _coef(ls, x)
                res = _rel(x, c * ls)
                dmax = max(np.abs(h["d1"]).max(), np.abs(h["d2"]).max()) / np.abs(x).max()
                worst_d, worst_res = max(worst_d, dmax), max(worst_res, res)
                c_mev = c * HA_TO_MEV * SCALE_REF
                rv = ref["shells"][f"{el}.onsite_breit.json:{s['label']}:{kind}"]
                print(f"\n[a] {el} {s['label']} {kind}: max|d1,d2|/|x| = {dmax:.1e}, |x - c l.s|/|x| = {res:.1e}, "
                      f"c = {c * HA_TO_MEV:+.6f} meV (sidecar {s[kind]['x_closed_mev']:+.6f})")
                assert c > 0 and res < 1e-12 and dmax < 1e-12
                assert abs(c_mev / s[kind]["x_closed_mev"] - 1) < 1e-10
                assert abs(c * SCALE_REF / rv["x_closed_ha"] - 1) < 1e-10
            # the assembled potential at D = 1 (onsite_only, full) is the AE exchange delta-xi in Ry
            pc = ob.onsite_potential(np.eye(n2), kern, double_counting="onsite_only", exchange="full")
            c_ae = s["ae"]["x_closed_ha_check"]
            assert _rel(pc["total"], ob.HA_TO_RY * c_ae * ls) < 1e-12
    print(f"\n[a] worst: max|d|/|x| {worst_d:.1e}, l.s residual {worst_res:.1e}")


# ---------------------------------------------------------------- (b) Hermiticity, linearity
def test_hermiticity_and_linearity(sc):
    worst_h, worst_lin = 0.0, 0.0
    for name, l, lam in _tensors(sc):
        n2 = 2 * (2 * l + 1)
        D1, D2 = _herm(n2, 1), _herm(n2, 2)
        h1, h2 = _all_pieces(lam, D1), _all_pieces(lam, D2)
        h12 = _all_pieces(lam, 0.7 * D1 - 1.3 * D2)
        for k in h1:
            # x1 and x2 are not Hermitian one by one (only their sum is); d1, d2, dir, x are
            if k not in ("x1", "x2"):
                worst_h = max(worst_h, np.linalg.norm(h1[k] - h1[k].conj().T) / np.linalg.norm(h1[k]))
            worst_lin = max(worst_lin, _rel(h12[k], 0.7 * h1[k] - 1.3 * h2[k]))
        assert _rel(h1["dir"], h1["d1"] + h1["d2"]) < 1e-13 and _rel(h1["x"], h1["x1"] + h1["x2"]) < 1e-13
    print(f"\n[b] max rel anti-Hermitian part {worst_h:.1e}; max rel linearity defect {worst_lin:.1e}")
    assert worst_h < 1e-12 and worst_lin < 1e-12


# ---------------------------------------------------------------- (c) time reversal
def test_time_reversal(sc):
    """h[Theta D Theta^-1] = Theta h[D] Theta^-1, Theta = U_T K; Theta^2 = -1 on the shell."""
    worst = 0.0
    for name, l, lam in _tensors(sc):
        UT = ob.time_reversal(l)
        assert np.abs(UT @ UT.conj() + np.eye(len(UT))).max() < 1e-14      # Theta^2 = -1
        for seed in range(3):
            D = _herm(len(UT), 10 + seed)
            h, hT = _all_pieces(lam, D), _all_pieces(lam, UT @ D.conj() @ UT.conj().T)
            for k in h:
                worst = max(worst, _rel(hT[k], UT @ h[k].conj() @ UT.conj().T))
        # a magnetic shell is not invariant (the check can fail): moment flips sign
        o = ob.shell_operators(l)
        Dz = np.diag([0.9] * (2 * l + 1) + [0.3] * (2 * l + 1)).astype(complex)
        DzT = UT @ Dz.conj() @ UT.conj().T
        assert np.trace(DzT @ o["S"][2]).real == pytest.approx(-np.trace(Dz @ o["S"][2]).real)
    print(f"\n[c] max rel time-reversal violation {worst:.1e}")
    assert worst < 1e-12


# ---------------------------------------------------------------- (d) rotations
def test_rotation_covariance_and_moment_direction(sc):
    from scipy.linalg import expm
    worst, worst_zx = 0.0, 0.0
    for l in (1, 2):
        o = ob.shell_operators(l)
        for seed in range(3):                  # M is exp(-i theta n.J), J = L + S (active, CS phases)
            R = _rotation(100 + seed)
            w, v = np.linalg.eig(R)
            ax = np.real(v[:, np.argmin(np.abs(w - 1))])
            th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
            th = th if np.dot(ax, [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) >= 0 else -th
            J = [a + b for a, b in zip(o["L"], o["S"])]
            M = ob.rotation_matrix(l, R)
            assert np.abs(M - expm(-1j * th * sum(a * j for a, j in zip(ax, J)))).max() < 1e-10
    for name, l, lam in _tensors(sc):
        n = 2 * l + 1
        o = ob.shell_operators(l)
        for seed in range(3):
            M = ob.rotation_matrix(l, _rotation(200 + seed))
            D = _herm(2 * n, 30 + seed)
            h0, h1 = _all_pieces(lam, D), _all_pieces(lam, M @ D @ M.conj().T)
            for k in h0:
                worst = max(worst, _rel(h1[k], M @ h0[k] @ M.conj().T))
        # moment along z vs along x: R_y(+90 deg) takes z to x
        Ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        M = ob.rotation_matrix(l, Ry)
        occ = np.random.default_rng(5).uniform(0.1, 0.9, size=2 * n)
        Dz = np.diag(occ).astype(complex)                              # collinear, non-spherical, m || z
        Dx = M @ Dz @ M.conj().T
        Sx = [np.trace(Dx @ s).real for s in o["S"]]
        assert abs(Sx[1]) < 1e-12 and abs(Sx[2]) < 1e-12 and Sx[0] == pytest.approx(np.trace(Dz @ o["S"][2]).real)
        hz, hx = _all_pieces(lam, Dz), _all_pieces(lam, Dx)
        for k in hz:
            worst_zx = max(worst_zx, _rel(hx[k], M @ hz[k] @ M.conj().T))
        Dsph = np.diag([0.9] * n + [0.3] * n).astype(complex)          # spherical collinear: d1 = b L.m
        d1z = ob.meanfield_pieces(lam, Dsph)["d1"]
        d1x = ob.meanfield_pieces(lam, M @ Dsph @ M.conj().T)["d1"]
        bz, bx = _coef(o["L"][2], d1z), _coef(o["L"][0], d1x)
        assert _rel(d1z, bz * o["L"][2]) < 1e-10 and _rel(d1x, bx * o["L"][0]) < 1e-10
        assert abs(bx / bz - 1) < 1e-12
    print(f"\n[d] max rel covariance error (general axes, all pieces) {worst:.1e}; z -> x {worst_zx:.1e}")
    assert worst < 1e-12 and worst_zx < 1e-12


# ---------------------------------------------------------------- (e) classical and atomic limits
def test_classical_atomic_limits_and_regression(sc, ref):
    # spherical collinear shell: d1 = b L_z, b = b1 (N_up - N_dn), b1 = -(alpha^2/2) I0 (fit-limited)
    for el, d in sc.items():
        for s in d["shells"]:
            l, n = s["l"], 2 * s["l"] + 1
            Lz = ob.shell_operators(l)["L"][2]
            for kind in ("ae", "ps"):
                d1 = ob.meanfield_pieces(s[kind]["lam"], np.diag([0.9] * n + [0.3] * n))["d1"]
                b = _coef(Lz, d1) / (n * 0.6)
                b_cls = -ob.ALPHA ** 2 / 2 * s[kind]["I0"]
                rv = ref["shells"][f"{el}.onsite_breit.json:{s['label']}:{kind}"]
                print(f"\n[e] {el} {s['label']} {kind}: b1 = {b * HA_TO_MEV:+.6f} meV/muB, classical "
                      f"{b_cls * HA_TO_MEV:+.6f} (rel {b / b_cls - 1:+.1e}), sidecar {s[kind]['b1_mev_per_muB']:+.6f}")
                assert _rel(d1, _coef(Lz, d1) * Lz) < 1e-10
                assert abs(b / b_cls - 1) < 5e-4
                assert abs(b * SCALE_REF / rv["b1_tensor_ha"] - 1) < 1e-10
    # one electron: no self-interaction (h_aa = 0, and <v|h|v> = 0 for a random spin-orbital)
    for name, l, lam in _tensors(sc):
        n2 = 2 * (2 * l + 1)
        worst, scale = 0.0, 0.0
        for a in range(n2):
            D = np.zeros((n2, n2)); D[a, a] = 1.0
            h = ob.meanfield_pieces(lam, D)
            worst = max(worst, abs(sum(h.values())[a, a]))
            scale = max(scale, np.abs(h["d1"]).max(), np.abs(h["x1"]).max())
        g = np.random.default_rng(3)
        v = g.normal(size=n2) + 1j * g.normal(size=n2)
        v /= np.linalg.norm(v)
        e = v.conj() @ sum(ob.meanfield_pieces(lam, np.outer(v, v.conj())).values()) @ v
        assert worst < 1e-13 * scale and abs(e) < 1e-13 * scale
    # atomic limit (ONSITE_BREIT 1.6 T6): Cr3+ b = -0.9499, (l+1/2) dxi_x = +1.0988; Fe2+ -2.2577, +2.6109 meV
    lz = ob._lmats(2)[2]
    Lz = ob.shell_operators(2)["L"][2]
    for fx, nup, b_doc, x_doc in (("Cr3plus_uhf3dup", 3, -0.9499, 1.0988), ("Fe2plus_uhf3dup", 5, -2.2577, 2.6109)):
        s = ob.read_sidecar(_path(f"{fx}.onsite_breit.json"))["shells"][0]
        h = ob.meanfield_pieces(s["ae"]["lam"], np.diag([nup / 5] * 5 + [0.0] * 5))
        b = _coef(Lz, h["d1"])
        lhd = 2.5 * 2 * _coef(lz, (h["x1"] + h["x2"])[:5, :5])
        t6 = ref["t6"][fx]
        print(f"\n[e] {fx}: b = {b * HA_TO_MEV:+.5f} meV (doc {b_doc:+.4f}), (l+1/2) dxi_x = {lhd * HA_TO_MEV:+.5f} "
              f"meV (doc {x_doc:+.4f}); vs onsite_ops {b * SCALE_REF / t6['b_ha'] - 1:+.1e}, "
              f"{lhd * SCALE_REF / t6['lhalf_dxi_ha'] - 1:+.1e}")
        assert abs(b * HA_TO_MEV / b_doc - 1) < 1e-3 and abs(lhd * HA_TO_MEV / x_doc - 1) < 1e-3
        assert abs(b * SCALE_REF / t6["b_ha"] - 1) < 1e-10 and abs(lhd * SCALE_REF / t6["lhalf_dxi_ha"] - 1) < 1e-10
    # regression vector: a written-out non-collinear D through the Fe 3D AE tensor, against onsite_ops
    rg = ref["regression"]
    D = np.array(rg["D_re"]) + 1j * np.array(rg["D_im"])
    h = ob.meanfield_pieces(sc["Fe"]["shells"][0]["ae"]["lam"], D)
    errs = {k: _rel(h[k] * SCALE_REF, np.array(rg[f"{k}_re"]) + 1j * np.array(rg[f"{k}_im"])) for k in h}
    print(f"\n[e] regression vs onsite_ops (Fe 3D AE): " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    assert max(errs.values()) < 1e-10


# ---------------------------------------------------------------- (f) double counting
def test_double_counting_tables(sc):
    s = sc["Cr"]["shells"][0]
    kern = ob.shell_kernels(s)
    D = _herm(10, 7)
    raw = {k: ob.HA_TO_RY * v for k, v in
           (("dae", _all_pieces(s["ae"]["lam"], D)["dir"]), ("dps", _all_pieces(s["ps"]["lam"], D)["dir"]),
            ("xae", _all_pieces(s["ae"]["lam"], D)["x"]), ("xps", _all_pieces(s["ps"]["lam"], D)["x"]))}
    want = {("onsite_only", "full"): raw["dae"] + raw["xae"],
            ("onsite_only", "paw"): raw["dae"] + raw["xae"] - raw["xps"],
            ("onsite_only", "off"): raw["dae"],
            ("paw", "full"): raw["dae"] - raw["dps"] + raw["xae"],
            ("paw", "paw"): raw["dae"] - raw["dps"] + raw["xae"] - raw["xps"],
            ("paw", "off"): raw["dae"] - raw["dps"]}
    for (dc, ex), w in want.items():
        pc = ob.onsite_potential(D, kern, double_counting=dc, exchange=ex)
        assert _rel(pc["total"], w) < 1e-13
        assert _rel(pc["dir_ae"], raw["dae"]) < 1e-13 and _rel(pc["x_ps"], raw["xps"]) < 1e-13
    # pairings: the legal one passes, the illegal one refuses
    assert ob.resolve_onsite_keys("onsite_only", "full", bispinor=False, tt_exchange=False) == ("onsite_only", "full")
    assert ob.resolve_onsite_keys("paw", "paw", bispinor=True, tt_exchange=True) == ("paw", "paw")
    for args, kw in ((("onsite_only", "full"), dict(bispinor=True)), (("paw", "full"), dict(bispinor=False)),
                     (("paw", "paw"), dict(tt_exchange=False)), (("paw", "full"), dict(tt_exchange=True)),
                     ((None, "full"), {}), (("onsite_only", None), {}), (("both", "full"), {}),
                     (("paw", "on"), {})):
        with pytest.raises(ValueError, match="GATE onsite_breit_keys"):
            ob.resolve_onsite_keys(*args, **kw)
    with pytest.raises(ValueError, match="GATE onsite_breit_keys"):
        ob.onsite_potential(D, kern, double_counting="onsite_only", exchange="")
    # a sidecar without ps: onsite_only/full and off pass, any paw refuses
    fx = ob.shell_kernels(ob.read_sidecar(_path("Cr3plus_uhf3dup.onsite_breit.json"))["shells"][0])
    pc = ob.onsite_potential(D, fx, double_counting="onsite_only", exchange="full")
    assert pc["dir_ps"] is None and pc["x_ps"] is None
    for dc, ex in (("paw", "full"), ("onsite_only", "paw")):
        with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):
            ob.onsite_potential(D, fx, double_counting=dc, exchange=ex)


# ---------------------------------------------------------------- setup helpers (CrCl3-like cell)
A_LAT = 9.0


def _fake_wfn(nspinor=2):
    return SimpleNamespace(
        nspinor=nspinor, atom_types=np.array([24, 17, 17]),
        atom_crys=np.array([[0.0, 0.0, 0.0], [0.31, 0.12, 0.27], [0.69, 0.88, 0.73]]),
        blat=2 * np.pi / A_LAT, bvec=np.eye(3), cell_volume=A_LAT ** 3, ecutwfc=20.0)


UPFS = {"Cr": _path("CrCl3_Cr.upf"), "Cl": _path("CrCl3_Cl.upf")}
SIDECARS = {"Cr": _path("Cr.onsite_breit.json"), "Cl": _path("Cl.onsite_breit.json")}


@pytest.fixture(scope="module")
def setup():
    return ob.build_onsite_setup(_fake_wfn(), UPFS, SIDECARS, projector="ortho-atomic",
                                 double_counting="onsite_only", exchange="full")


# ---------------------------------------------------------------- (g) sidecar refusals
def test_sidecar_refusals(tmp_path, setup):
    with open(_path("Cl.onsite_breit.json")) as f:
        raw = json.load(f)

    def write(obj, name):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return str(p)

    assert ob.read_sidecar(write(raw, "ok.json"))["element"] == "Cl"                 # control
    bad = copy.deepcopy(raw); bad["format"] = "lorrax.onsite_breit_sidecar/v0"
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):
        ob.read_sidecar(write(bad, "fmt.json"))
    bad = copy.deepcopy(raw)                                                          # corrupted tensor
    i_big = int(np.argmax(np.abs(bad["shells"][0]["ae"]["lam_re"])))
    bad["shells"][0]["ae"]["lam_re"][i_big] *= 1.001
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar.*\n.*closed shell"):
        ob.read_sidecar(write(bad, "corrupt.json"))
    bad = copy.deepcopy(raw); bad["shells"][0]["ae"]["lam_im"] = bad["shells"][0]["ae"]["lam_im"][:-3]
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):
        ob.read_sidecar(write(bad, "shape.json"))
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):                  # ps missing when needed
        ob.read_sidecar(_path("Fe2plus_uhf3dup.onsite_breit.json"), require_ps=True)
    ob.read_sidecar(_path("Fe2plus_uhf3dup.onsite_breit.json"))                          # control
    # chi authentication against the UPF: a 1e-6 relative change of chi refuses, the original builds
    bad = copy.deepcopy(raw); bad["shells"][0]["projector"]["chi"] = [
        x * (1 + 1e-6) for x in bad["shells"][0]["projector"]["chi"]]
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar.*\n.*chi"):
        ob.build_onsite_setup(_fake_wfn(), UPFS, {"Cr": SIDECARS["Cr"], "Cl": write(bad, "chi.json")},
                              projector="ortho-atomic", double_counting="onsite_only", exchange="full")
    assert setup.R_os == 10 + 2 * 6                                                       # control (fixture)
    # projector / WFN refusals, each against the control above
    for kw, wfn, upfs in ((dict(projector="atomic"), _fake_wfn(), UPFS),
                          (dict(projector="ortho-atomic"), _fake_wfn(nspinor=1), UPFS),
                          (dict(projector="ortho-atomic"), _fake_wfn(), {"Cr": UPFS["Cr"]})):
        with pytest.raises(ValueError, match="GATE onsite_breit_projector"):
            ob.build_onsite_setup(wfn, upfs, {"Cr": SIDECARS["Cr"]}, double_counting="onsite_only",
                                  exchange="full", **kw)
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):                    # paw needs ps
        ob.build_onsite_setup(_fake_wfn(), UPFS, [_path("Cr3plus_uhf3dup.onsite_breit.json")],
                              projector="ortho-atomic", double_counting="paw", exchange="full")
    with pytest.raises(ValueError, match="GATE onsite_breit_sidecar"):                    # projector type none
        ob.build_onsite_setup(_fake_wfn(), UPFS, [_path("Cr3plus_uhf3dup.onsite_breit.json")],
                              projector="ortho-atomic", double_counting="onsite_only", exchange="full")


# ---------------------------------------------------------------- (h) basis map and band kernels
def _real_L(l):
    """<Y^r_i| L_c |Y^r_j> by quadrature, L = -i r x grad on the QE real harmonics (exact polynomials)."""
    from psp.radial.solid_harmonics import solid_harmonics_jax
    x, wx = np.polynomial.legendre.leggauss(2 * l + 4)
    nphi = 4 * l + 8
    ct, ph = np.repeat(x, nphi), np.tile(2 * np.pi * np.arange(nphi) / nphi, len(x))
    st = np.sqrt(1 - ct ** 2)
    xyz = np.stack([st * np.cos(ph), st * np.sin(ph), ct], axis=1)
    w = np.repeat(wx, nphi) * 2 * np.pi / nphi
    f = lambda K: solid_harmonics_jax(l, K)                                     # noqa: E731
    Y = np.asarray(f(jnp.asarray(xyz)))
    grad = np.stack([np.asarray(jax.jvp(f, (jnp.asarray(xyz),), (jnp.asarray(np.eye(3)[c][None].repeat(len(xyz), 0)),))[1])
                     for c in range(3)])                                         # (3, n, N)
    rxg = np.stack([xyz[:, 1] * grad[2] - xyz[:, 2] * grad[1], xyz[:, 2] * grad[0] - xyz[:, 0] * grad[2],
                    xyz[:, 0] * grad[1] - xyz[:, 1] * grad[0]])
    return [np.einsum("iN,jN,N->ij", Y, -1j * rxg[c], w) for c in range(3)]


def _gtable(kvec, ngk, ngkmax, B, seed):
    """The ngk lowest |k+G| Miller indices, then pad rows (a far G, masked)."""
    r = np.arange(-6, 7)
    G = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T
    q = np.linalg.norm((G + kvec) @ B, axis=1)
    G = G[np.argsort(q, kind="stable")][:ngk]
    pad = np.tile([[6, -5, 4]], (ngkmax - ngk, 1))
    return np.concatenate([G, pad]).astype(np.int32), np.r_[np.ones(ngk), np.zeros(ngkmax - ngk)]


def _reference_P(psi, gv, gm, kv, setup, mask=True):
    """numpy: masked atomic rows, Loewdin over all rows, projection, W^dagger per spin block."""
    from psp import vnl_ops
    nk, nb = psi.shape[:2]
    out = np.zeros((nk, setup.R_os, nb), complex)
    for k in range(nk):
        Z = np.asarray(vnl_ops.build_vnl_kdata_from_kvec(kv[k], gv[k], setup.rows).Z)
        if mask:
            Z = Z * gm[k][None, :]
        lam, V = np.linalg.eigh(np.conj(Z) @ Z.T)
        # phi~_r = sum_r' phi_r' (O^-1/2)_{r'r}, O_rr' = <Z_r|Z_r'>: all rows orthonormal
        Zt = (V @ np.diag(lam ** -0.5) @ V.conj().T).T @ Z
        assert np.abs(np.conj(Zt) @ Zt.T - np.eye(len(Zt))).max() < 1e-10
        for sa in setup.shell_atoms:
            n = 2 * sa["l"] + 1
            Pr = np.einsum("iG,nsG->isn", np.conj(Zt[list(sa["rows"])]), psi[k])  # (n, 2, nb)
            for s in range(2):
                out[k, sa["start"] + s * n: sa["start"] + (s + 1) * n] = sa["W"].conj().T @ Pr[:, s]
    return out


def _meshes():
    devs = jax.devices()
    out = [Mesh(np.array(devs[:1]).reshape(1, 1), ("x", "y"))]
    if len(devs) >= 4:
        out.append(Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y")))
    return out


def test_basis_map_and_band_kernels(setup):
    # W: unitary, and it carries the real-basis L onto the CS matrices (L_z -> diag(m))
    for l in range(4):
        W = ob.real_to_complex_ylm(l)
        assert np.abs(W.conj().T @ W - np.eye(2 * l + 1)).max() < 1e-12
        Lr, Lc = _real_L(l), ob._lmats(l)
        for c in range(3):
            assert np.abs(W.conj().T @ Lr[c] @ W - Lc[c]).max() < 1e-12
        assert np.abs(W.conj().T @ Lr[2] @ W - np.diag(np.arange(-l, l + 1))).max() < 1e-12
    # synthetic projection table against the numpy reference (mask + Loewdin + W)
    wfn = _fake_wfn()
    B = wfn.blat * wfn.bvec
    rng = np.random.default_rng(11)
    nk, ngkmax = 3, 90
    kv = rng.uniform(-0.5, 0.5, size=(nk, 3))
    tabs = [_gtable(kv[k], 70 + 7 * k, ngkmax, B, k) for k in range(nk)]
    gv = np.stack([t[0] for t in tabs]); gm = np.stack([t[1] for t in tabs])
    for mesh in _meshes():
        for nb in (8, 6):
            psi = (rng.normal(size=(nk, nb, 2, ngkmax)) + 1j * rng.normal(size=(nk, nb, 2, ngkmax))) * gm[:, None, None, :]
            ref_P = _reference_P(psi, gv, gm, kv, setup)
            Pt = ob.onsite_projections(jnp.asarray(psi), gv, gm, kv, setup, mesh=mesh)
            err = _rel(np.asarray(Pt), ref_P)
            red = _rel(_reference_P(psi, gv, gm, kv, setup, mask=False), ref_P)         # the mask matters
            print(f"\n[h] mesh {dict(mesh.shape)} nb={nb}: |P - P_ref|/|P_ref| = {err:.1e} (unmasked rows: {red:.1e})")
            assert Pt.shape == (nk, setup.R_os, nb) and err < 1e-12 and red > 1e-3
            assert Pt.sharding.is_fully_replicated
            # density matrix with a random unitary U (replicated, and sharded on the mesh)
            w = np.full(nk, 1.0 / nk)
            f = rng.uniform(0, 1, size=(nk, nb))
            U = np.linalg.qr(rng.normal(size=(nk, nb, nb)) + 1j * rng.normal(size=(nk, nb, nb)))[0]
            PU = np.einsum("kan,kni->kai", ref_P, U)
            D_ref = np.einsum("k,ki,kai,kbi->ab", w, f, PU, PU.conj())
            D_id = ob.onsite_density_matrix(Pt, w, f, setup, mesh=mesh)
            assert _rel(D_id, np.einsum("k,kn,kan,kbn->ab", w, f, ref_P, ref_P.conj())) < 1e-12
            for spec in (P(), P(None, "x", "y"), P(None, "y", "x")):
                Uj = jax.device_put(jnp.asarray(U), NamedSharding(mesh, spec))
                D = ob.onsite_density_matrix(Pt, w, f, setup, U=Uj, mesh=mesh)
                assert _rel(D, D_ref) < 1e-12 and np.abs(D - D.conj().T).max() < 1e-14
            # band matrix: Hermitian for Hermitian h, equal to the numpy reference, replicated
            h, _ = ob.onsite_block_potential(D_ref, setup)
            assert np.abs(h - h.conj().T).max() < 1e-14 * np.abs(h).max() + 1e-300
            hb = h + _herm(setup.R_os, 3) * 1e-4                    # plus a generic Hermitian block
            dH = ob.onsite_band_matrix(Pt, hb, mesh=mesh)
            dH_ref = np.einsum("kan,ab,kbm->knm", ref_P.conj(), hb, ref_P)
            assert _rel(np.asarray(dH), dH_ref) < 1e-12 and dH.sharding.is_fully_replicated
            assert np.abs(np.asarray(dH) - np.conj(np.swapaxes(np.asarray(dH), 1, 2))).max() < 1e-14 * np.abs(dH_ref).max()
    lines = ob.onsite_report_lines(D_ref, ob.onsite_block_potential(D_ref, setup)[1], setup)
    assert len(lines) == len(setup.shell_atoms) * 6 and "Tr D" in lines[0]
    print("\n" + "\n".join(lines[:6]))
