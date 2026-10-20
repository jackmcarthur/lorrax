"""``tdagw.td_system``: the TD system of a GW bundle, and the observables on it (CPU).

1. ``position_from_velocity`` inverts ``v = i[H, r]``: off-diagonal r back to
   1e-13, Hermitian, zero diagonal; ``mask`` / ``cut`` / ``cv`` treat the
   same-occupation pairs as documented and count them; a band inversion and
   a non-Hermitian velocity refuse.
2. ``bundle_header`` round-trips through the writer, ``read_header`` and
   ``header_operators`` (itype 1: r as built, no intraband term); energies
   Ry -> eV -> Ha with TDGW's Hartree.
3. ``bse_dipole_receipt`` reproduces a ``bse_jax``-convention ⟨0|r|S⟩
   (stored as −i⟨0|r|S⟩) exactly; the bra/ket-swapped r (red twin) does not
   and exchanges I+ and I−.
4. Helicity end to end (header -> deck -> propagation -> exciton_projection):
   on a three-band chiral system σ+ = E0 (cos ωt, +sin ωt, 0) excites only the
   transition with ⟨c|x + iy|v⟩ ≠ 0 and σ− only the other; the swapped header
   (red twin) exchanges them.  The amplitude is first-order perturbation
   theory, |ρ_cv|² = E0² |r_x ± i r_y|² π τ² / 4, the carrier density is
   Σ|c_S|²/N_k, and c_S rotates as exp(−iΩt) after the pulse.
5. Deck refusals of the bundle route and of the new measurements.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import os

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from common.units import RYD_TO_EV  # noqa: E402
from tdagw import td_system as ts  # noqa: E402
from tdagw import tdgw_io  # noqa: E402

RNG = np.random.default_rng(5)
HBAR_EV_AS = 658.2119569        # ħ in eV·as


def _herm(n, scale=1.0):
    a = RNG.standard_normal((n, n)) + 1j * RNG.standard_normal((n, n))
    return scale * 0.5 * (a + a.conj().T)


def _planted(nk=3, nb=6, n_val=3):
    """Gapped spectra, a planted Hermitian position X and v = i[H, X] (eigenbasis)."""
    e = np.sort(RNG.uniform(-1.0, -0.2, (nk, n_val)), axis=1)
    e = np.concatenate([e, np.sort(RNG.uniform(0.3, 1.2, (nk, nb - n_val)), axis=1)], axis=1)
    x = np.stack([np.stack([_herm(nb) for _ in range(nk)]) for _ in range(3)])
    v = 1j * (e[None, :, :, None] - e[None, :, None, :]) * x
    return e, x, v


def test_position_from_velocity_inverts_the_commutator():
    e, x, v = _planted()
    r, rec = ts.position_from_velocity(v, e, 3, rule="cut", gap_ry=1e-6)
    off = ~np.eye(6, dtype=bool)
    assert np.abs((r - x)[..., off]).max() / np.abs(x).max() < 1e-13
    assert np.abs(np.diagonal(r, axis1=-2, axis2=-1)).max() == 0.0
    assert np.abs(r - np.conj(np.swapaxes(r, -1, -2))).max() == 0.0
    assert rec["vv"]["pairs"] == 3 * 3 and rec["cc"]["pairs"] == 3 * 3
    assert rec["cv_pairs"] == 3 * 9 and rec["vv"]["below_gap"] == 0
    r_cv, _ = ts.position_from_velocity(v, e, 3, rule="cv", gap_ry=1e-6)
    assert np.abs(r_cv[:, :, :3, :3]).max() == 0 and np.abs(r_cv[:, :, 3:, 3:]).max() == 0
    assert np.array_equal(r_cv[:, :, 3:, :3], r[:, :, 3:, :3])


def test_pair_rules_on_a_near_degenerate_pair_and_the_gates():
    e, x, v = _planted(nk=2)
    e[:, 1] = e[:, 0] + 1e-3 / RYD_TO_EV          # a valence pair 1 meV apart
    v = 1j * (e[None, :, :, None] - e[None, :, None, :]) * x
    gap = 0.05 / RYD_TO_EV
    r_cut, rc = ts.position_from_velocity(v, e, 3, rule="cut", gap_ry=gap)
    r_mask, rm = ts.position_from_velocity(v, e, 3, rule="mask", gap_ry=gap)
    assert rc["vv"]["below_gap"] >= 2 and rc["vv"]["min_gap_ev"] == pytest.approx(1e-3, rel=1e-6)
    assert np.abs(r_cut[:, :, 0, 1]).max() == 0.0
    factor = -np.expm1(-(1e-3 / 0.05) ** 2 / 2)
    assert np.allclose(r_mask[:, :, 0, 1], x[:, :, 0, 1] * factor, rtol=1e-10, atol=0)
    assert np.allclose(r_mask[:, :, 3:, :3], x[:, :, 3:, :3], rtol=1e-12, atol=0)
    bad = e.copy()
    bad[0, 3] = bad[0, 2] - 0.01                     # conduction below valence at k = 0
    with pytest.raises(ValueError, match="GATE td_cv_gap"):
        ts.position_from_velocity(v, bad, 3, rule="mask", gap_ry=gap)
    with pytest.raises(ValueError, match="GATE td_velocity_hermitian"):
        ts.position_from_velocity(v + 1e-3j * np.abs(v).max(), e, 3, gap_ry=gap)  # i c J is anti-Hermitian


def _structure():
    return dict(rprim=np.diag([8.0, 8.0, 30.0]), atomic_numbers=np.array([1]),
                atomic_positions_red=np.zeros((1, 3)), fftgrid=np.array([8, 8, 30]))


def _kgrid():
    k = np.stack(np.meshgrid(np.arange(2) / 2, np.arange(2) / 2, [0.0], indexing="ij"), -1)
    return k.reshape(-1, 3), (2, 2, 1)


def test_bundle_header_round_trip(tmp_path):
    e, x, v = _planted(nk=4)
    r, _ = ts.position_from_velocity(v, e, 3, gap_ry=1e-6)
    kf, kg = _kgrid()
    hdr = ts.bundle_header(structure=_structure(), k_frac=kf, kgrid=kg, bands=(10, 16),
                           nspinor=2, e_qp_ry=e, n_val=3, r=r)
    hdr["td_system"] = dict(route="test", gap=np.float64(0.05))
    tdgw_io.init_output(str(tmp_path / "tdgw.h5"), hdr, t0=0.0, tf=1.0, mdt=0.5,
                        mtsteps=np.array([0.0, 0.5]), measurements=[])
    back = tdgw_io.read_header(str(tmp_path))
    assert back["bands"] == (11, 17, 1) and back["itype"] == 1
    assert np.allclose(back["e_kn"], e * RYD_TO_EV / tdgw_io.HA_IN_EV, rtol=0, atol=1e-15)
    assert np.array_equal(back["f_kn"], np.repeat([[1.0] * 3 + [0.0] * 3], 4, axis=0))
    host = tdgw_io.header_operators(back, geometry="slab", esmear_ha=0.05 / 27.2,
                                    intraband=True)
    assert host["intra"] is None and host["v0_intra"] is None
    assert np.array_equal(host["r_inter"], r)


def _write_eigenvectors(path, A, kpts, *, eig_ev, dipoles=None):
    """A minimal BerkeleyGW-layout eigenvectors.h5 (TDA, one Q), valence flipped as on disk."""
    import h5py
    a = np.asarray(A)[:, :, :, ::-1]                                # BGW: v = 0 at the VBM
    n, nk, nc, nv = a.shape
    with h5py.File(path, "w") as f:
        ex = f.create_group("exciton_data")
        ex["eigenvalues"] = np.asarray(eig_ev, float)
        ev = np.zeros((1, n, nk, nc, nv, 1, 2))
        ev[0, :, :, :, :, 0, 0], ev[0, :, :, :, :, 0, 1] = a.real, a.imag
        ex["eigenvectors"] = ev
        if dipoles is not None:
            d = np.zeros((1, n, 3, 2))
            d[0, :, :, 0], d[0, :, :, 1] = dipoles.real, dipoles.imag
            ex["dipoles"] = d
        p = f.create_group("exciton_header/params")
        for key, val in (("nc", nc), ("nv", nv), ("ns", 1), ("spin_kernel", 3), ("use_tda", 1)):
            p[key] = val
        kp = f.create_group("exciton_header/kpoints")
        kp["nk"] = nk
        kp["kpts"] = np.asarray(kpts, float).T.copy()        # bse_jax layout (3, nk)


def test_bse_dipole_receipt_and_the_swapped_twin(tmp_path):
    e, x, v = _planted(nk=4)
    r, _ = ts.position_from_velocity(v, e, 3, gap_ry=1e-6)
    kf, _ = _kgrid()
    A = RNG.standard_normal((5, 4, 3, 3)) + 1j * RNG.standard_normal((5, 4, 3, 3))
    A /= np.linalg.norm(A.reshape(5, -1), axis=1)[:, None, None, None]
    # bse_jax stores sum_t A_t conj(d_t), d = <c|v|v>/dE = i <c|r|v>
    d = 1j * np.moveaxis(r[:, :, 3:, :3], 0, -1)                   # (nk, nc, nv, 3)
    stored = np.einsum("skcv,kcvx->sx", A, np.conj(d))
    path = str(tmp_path / "eigenvectors.h5")
    _write_eigenvectors(path, A, kf, eig_ev=np.arange(5.0), dipoles=stored)
    rec = ts.bse_dipole_receipt(r, path, 3, kf)
    assert rec["dipole_rel_dev"] < 1e-14 and rec["i_plus_rel_dev"] < 1e-13
    twin = ts.bse_dipole_receipt(np.swapaxes(r, -1, -2), path, 3, kf)
    assert twin["dipole_rel_dev"] > 0.1


# ---------------------------------------------------------------------------
# Helicity, amplitude and phase on a chiral three-band system
# ---------------------------------------------------------------------------

# t0 = 5 tau: the drive switches on at f(0) = e^-25.  At t0 = 3 tau (f(0) = e^-9) the
# switch-on alone leaves the counter-rotating channel at E0 f(0) d / (sqrt2 (Omega + w))
# = 1.1e-8 (population ratio 7.8e-12), reproduced by an independent 3-level integration.
E_GAP_EV, D_BOHR, E0, T0_AS, TAU_AS = 2.0, 1.0, 1e7, 20000.0, 4000.0


def _chiral_header(swap=False):
    """1 valence + 2 degenerate conduction bands; c+ couples to x+iy, c- to x-iy."""
    kf, kg = _kgrid()
    nk = kf.shape[0]
    e = np.zeros((nk, 3))
    e[:, 1:] = E_GAP_EV / RYD_TO_EV
    r = np.zeros((3, nk, 3, 3), complex)
    r[:2, :, 1, 0] = (D_BOHR / np.sqrt(2) * np.array([1.0, -1j]))[:, None]   # <c+|r|v>
    r[:2, :, 2, 0] = (D_BOHR / np.sqrt(2) * np.array([1.0, 1j]))[:, None]    # <c-|r|v>
    r = r + np.conj(np.swapaxes(r, -1, -2))
    if swap:
        r = np.swapaxes(r, -1, -2)
    return ts.bundle_header(structure=_structure(), k_frac=kf, kgrid=kg, bands=(0, 3),
                            nspinor=2, e_qp_ry=e, n_val=1, r=r), kf


def _drive(sign):
    w = E_GAP_EV / HBAR_EV_AS
    env = f"{E0} * np.exp(-((t - {T0_AS}) / {TAU_AS})**2)"
    return {"E_x": f"{env} * np.cos({w!r} * t)",
            "E_y": f"{sign} * {env} * np.sin({w!r} * t)", "E_z": 0}


def _run_chiral(tmp_path, swap=False):
    import h5py
    from common.collectives import resolve_mesh
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function
    hdr, kf = _chiral_header(swap)
    hdir = tmp_path / ("hdr_swap" if swap else "hdr")
    hdir.mkdir()
    tdgw_io.init_output(str(hdir / "tdgw.h5"), hdr, t0=0.0, tf=1.0, mdt=0.5,
                        mtsteps=np.array([0.0, 0.5]), measurements=[])
    A = np.zeros((2, 4, 2, 1), complex)
    A[0, :, 0, 0] = A[1, :, 1, 0] = 0.5                              # S+ = c+ v, S- = c- v
    evf = str(tmp_path / "chiral_eigenvectors.h5")
    _write_eigenvectors(evf, A, kf, eig_ev=[E_GAP_EV, E_GAP_EV])
    out = tmp_path / ("out_swap" if swap else "out")
    raw = {"control": {"calc_type": "IP",
                       "timesteps": {"t0": 0, "tf": 40000, "dt_measure": 100, "ntcheck": 1000},
                       "ivp": {"rtol": 1e-11, "atol": 1e-18},
                       "files": {"outdir": str(out), "load_header_from": str(hdir)}},
           "system": {"ibndmin": 1, "ibndmax": 3, "ivbm": 1, "geometry": "bulk"},
           "measurements": {"types": ["Efield", "carrier_density", "exciton_projection"],
                            "2": {"eigenvectors": evf}},
           "lorrax": {"trials": [_drive(+1), _drive(-1)]}}
    deck = parse_td_deck(raw, source="chiral")
    mesh = resolve_mesh()
    system = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, system, mesh)
    outs = stages.open_outputs(deck, system)
    stages.propagate(deck, system, model, efield_function(deck.trials), outs, mesh,
                     say=lambda s: None)
    res = []
    for path in outs.paths:
        with h5py.File(path, "r") as f:
            td = f["tdgw_data/time_domain"]
            res.append(dict(t=td["exciton_projection/times"][()],
                            c=td["exciton_projection/data"][()],
                            n=td["carrier_density/data"][()]))
    return res


@pytest.fixture(scope="module")
def chiral(tmp_path_factory):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices")
    tmp = tmp_path_factory.mktemp("chiral")
    return _run_chiral(tmp), _run_chiral(tmp, swap=True)


def test_circular_drives_select_the_helicity_and_the_swap_exchanges_it(chiral):
    (plus, minus), (plus_sw, minus_sw) = chiral
    pop = lambda r: np.abs(r["c"][-1]) ** 2                         # noqa: E731
    assert pop(plus)[1] < 1e-12 * pop(plus)[0]                      # sigma+ -> S+ only
    assert pop(minus)[0] < 1e-12 * pop(minus)[1]                    # sigma- -> S- only
    assert pop(plus_sw)[0] < 1e-12 * pop(plus_sw)[1]                # twin: exchanged
    assert pop(minus_sw)[1] < 1e-12 * pop(minus_sw)[0]


def test_first_order_amplitude_density_and_phase(chiral):
    (plus, _), _ = chiral
    au_e = 5.14220674763e11                                         # V/m per a.u.
    tau = TAU_AS / tdgw_io.T_IN_ATTOSEC
    rho2 = (E0 / au_e) ** 2 * 2 * D_BOHR ** 2 * np.pi * tau ** 2 / 4  # |r_x + i r_y|^2 = 2 d^2
    c_plus = plus["c"][-1, 0]
    assert abs(c_plus) ** 2 == pytest.approx(4 * rho2, rel=2e-4)    # c = sum_k (1/2) rho_cv
    n_c, n_v = plus["n"][-1]
    assert n_c == pytest.approx(abs(c_plus) ** 2 / 4, rel=1e-4) and n_v == pytest.approx(n_c, rel=1e-12)
    # after the pulse, c_S(t) ~ exp(-i Omega t)
    late = plus["t"] > (T0_AS + 3.5 * TAU_AS) / tdgw_io.T_IN_ATTOSEC  # residual drive erfc(3.5) ~ 7e-7
    ph = np.unwrap(np.angle(plus["c"][late, 0]))
    omega = (E_GAP_EV / tdgw_io.HA_IN_EV)
    slope = np.polyfit(plus["t"][late], ph, 1)[0]
    assert slope == pytest.approx(-omega, rel=1e-6)


def test_bundle_deck_refusals():
    from tdagw.td_config import parse_td_deck
    base = {"control": {"calc_type": "IP", "timesteps": {"t0": 0, "tf": 10, "dt_measure": 1},
                        "files": {"outdir": "/tmp/x"}},
            "system": {"ibndmin": 1, "ibndmax": 4, "ivbm": 2, "geometry": "slab"},
            "measurements": {"types": ["Efield"]}}

    def deck(**lorrax):
        raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        raw["lorrax"] = lorrax
        return raw

    with pytest.raises(ValueError, match="GATE td_bundle_input"):
        parse_td_deck(deck(system="bundle", bundle_input="gw.in"))
    raw = deck(system="bundle", bundle_input="gw.in", dipole_file="dipole.h5")
    raw["control"] = dict(raw["control"], files={"outdir": "/tmp/x", "load_header_from": "/tmp/h"})
    with pytest.raises(ValueError, match="GATE td_header_source"):
        parse_td_deck(raw)
    raw = deck(eqp_file="eqp1.dat")
    raw["control"] = dict(raw["control"], files={"outdir": "/tmp/x", "load_header_from": "/tmp/h"})
    with pytest.raises(ValueError, match="GATE td_bundle_key"):
        parse_td_deck(raw)
    with pytest.raises(ValueError, match="GATE td_pair_rule"):
        parse_td_deck(deck(system="bundle", bundle_input="gw.in", dipole_file="d.h5",
                           pair_rule="soft"))
    raw = deck(system="bundle", bundle_input="gw.in", dipole_file="d.h5")
    raw["measurements"] = {"types": ["exciton_projection"]}
    with pytest.raises(ValueError, match="GATE td_exciton_projection"):
        parse_td_deck(raw)
    ok = parse_td_deck(deck(system="bundle", bundle_input="gw.in", dipole_file="d.h5"))
    assert ok.values["lorrax.pair_gap_ev"] == 0.05 and ok.values["lorrax.system"] == "bundle"
    assert os.path.basename(ok.values["lorrax.bundle_input"]) == "gw.in"


def test_describe_prints_the_bundle_receipts(tmp_path):
    """The run header carries the builder's pair counts and the gauge receipt."""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices")
    import dataclasses
    from types import SimpleNamespace
    from common.collectives import resolve_mesh
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    hdr, _ = _chiral_header()
    hdir = tmp_path / "hdr"
    hdir.mkdir()
    tdgw_io.init_output(str(hdir / "tdgw.h5"), hdr, t0=0.0, tf=1.0, mdt=0.5,
                        mtsteps=np.array([0.0, 0.5]), measurements=[])
    raw = {"control": {"calc_type": "IP", "timesteps": {"t0": 0, "tf": 200, "dt_measure": 100},
                       "files": {"outdir": str(tmp_path / "out"), "load_header_from": str(hdir)}},
           "system": {"ibndmin": 1, "ibndmax": 3, "ivbm": 1, "geometry": "bulk"},
           "measurements": {"types": ["Efield"]},
           "lorrax": {"trials": [_drive(+1)]}}
    deck = parse_td_deck(raw, source="describe")
    mesh = resolve_mesh()
    system = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, system, mesh)
    e, _, v = _planted(nk=2)
    _, prec = ts.position_from_velocity(v, e, 3, rule="mask", gap_ry=0.05 / RYD_TO_EV)
    rec = dict(densified=False, energy_source="eqp", velocity=dict(route="coarse dipole.h5 (whole window)"),
               pairs=prec, bse_dipoles=dict(n_states=2, dipole_rel_dev=1e-15, i_plus_rel_dev=2e-15,
                                            i_minus_rel_dev=0.0))
    system = (system._replace(receipts=rec) if hasattr(system, "_replace")
              else dataclasses.replace(system, receipts=rec))
    text = stages.describe(deck, system, model, SimpleNamespace(mesh_shape=(2, 2), n_devices=4, platform="cpu"))
    assert "TD system       : native bundle" in text and "pair rule mask at 0.05 eV" in text
    assert f"cc' below: {prec['cc']['below_gap']} of {prec['cc']['pairs']}" in text
    assert "gauge receipt   : <0|r|S> of 2 BSE states" in text and "rel dev 1.00e-15" in text



def test_window_position_from_route_b_is_used_as_given():
    """r_window (Route B, exact position of the fine states) enters without a v/dE division."""
    e, x, _ = _planted(nk=2)
    gap = 0.05 / RYD_TO_EV
    r, rec = ts.position_from_window_r(x, e, 3, rule="mask", gap_ry=gap)
    xd = x.copy(); xd[:, :, np.arange(6), np.arange(6)] = 0.0
    assert np.allclose(r, xd, rtol=0, atol=1e-15)                          # mask applied upstream, not twice
    assert rec["rule"].startswith("mask") and rec["vv"]["pairs"] == 2 * 3 and rec["cc"]["pairs"] == 2 * 3
    r_cv, _ = ts.position_from_window_r(x, e, 3, rule="cv", gap_ry=gap)
    assert np.abs(r_cv[:, :, :3, :3]).max() == 0 and np.abs(r_cv[:, :, 3:, 3:]).max() == 0
    assert np.allclose(r_cv[:, :, 3:, :3], x[:, :, 3:, :3], atol=1e-15)
    e2 = e.copy(); e2[:, 1] = e2[:, 0] + 1e-3 / RYD_TO_EV
    r_cut, rc = ts.position_from_window_r(x, e2, 3, rule="cut", gap_ry=gap)
    assert np.abs(r_cut[:, :, 0, 1]).max() == 0 and rc["vv"]["below_gap"] >= 2
    with pytest.raises(ValueError, match="GATE td_position_hermitian"):
        ts.position_from_window_r(x + 1e-3j * np.abs(x).max(), e, 3, gap_ry=gap)
    bad = e.copy(); bad[0, 3] = bad[0, 2] - 0.01
    with pytest.raises(ValueError, match="GATE td_cv_gap"):
        ts.position_from_window_r(x, bad, 3, gap_ry=gap)
    with pytest.raises(ValueError, match="GATE td_pair_gap_route_b"):
        ts.position_from_window_r(x, e, 3, rule="mask", gap_ry=0.1 / RYD_TO_EV)
