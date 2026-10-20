"""``densified_dipoles(qp_states=True)``: the exact position in fine QP states (CPU, 2x2 host mesh).

With ``--eqp`` the fine states diagonalize the interpolated QP f(H), which
mixes the velocity's DFT bands off the coarse grid; ``v/ΔE`` of their
Rayleigh-quotient energies is then not their position (CrI3 8→12 against the
native 12x12: Σ|d|² 0.986, onset 0.87, while Σ|v|² is 1.0004).  On the toy
slab of ``tests/test_bse_kgrid_dipoles_cpu.py`` (imported, not copied; smooth
model, hop 0.02) with band-dependent QP shifts that rotate the window's two
valence bands and its two conduction bands against each other:

1. GREEN: ``qp_states=True`` gives ⟨c̃|r̂|ṽ⟩ of the fine states to the
   htransform's interpolation floor, against the toy's exact position in its
   full orbital eigenbasis; RED twin: the Rayleigh-quotient ``v/ΔE``
   (``d_alpha_rq``, the pre-fix dipoles) is more than 5x further off.
2. The DFT path (``qp_states`` False) keeps ``d_alpha = v/ΔE`` bitwise.
3. ``r_window`` is Hermitian (zero diagonal on the DFT path: no intraband
   position); on the QP path ``i`` times its c←v block IS ``d_alpha``; off the
   diagonal it matches the exact window position.
4. The pair rule is ``tdagw.td_system.position_from_velocity``'s.
5. The rotation's record: small valence↔conduction mixing, window sum kept.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

_TOY = pathlib.Path(__file__).with_name("test_bse_kgrid_dipoles_cpu.py")
_spec = importlib.util.spec_from_file_location("_bse_kgrid_dipoles_toy", _TOY)
toy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(toy)

from bandstructure.bse_setup import compute_wfns_fi  # noqa: E402
from bse.bse_densify import (DENSIFIED_POSITION_FORM,  # noqa: E402
                             _position_from_velocity_rule, densified_dipoles)
from ffi.fft import xla_reference  # noqa: E402

NV, LO, HI = toy.N_VAL, toy.B_MIN, toy.B_MAX
HOP, SHIFT = 0.02, 0.1


def _qp(e, kc, s=SHIFT):
    """Band-dependent QP energies: a conduction scissor plus opposite k-dependent
    shifts of the window's two valence and two conduction bands."""
    x, y = np.cos(2 * np.pi * kc[:, 0]), np.cos(2 * np.pi * kc[:, 1])
    d = np.zeros_like(e)
    d[:, 4:] += 0.3
    d[:, 2] += s * x * y
    d[:, 3] -= s * x * y
    d[:, 4] += s * x
    d[:, 5] -= s * x
    return e + d


def _run(mesh, c, e, v, qp, **kw):
    kc = toy._grid(toy.KC)
    with xla_reference():
        fine = compute_wfns_fi(
            ctilde=jnp.asarray(c), B_at_mu=jnp.ones((c.shape[2], 1, 4), complex),
            enk_sigma=jnp.asarray(qp.T), kgrid_co=toy.KC, kgrid_fi=toy.KF,
            band_window_fi=(LO, HI), mesh_xy=mesh, return_coeffs=True)
        dip = densified_dipoles(
            (v, e, "toy dH/dk", "toy.h5"), sym=toy._sym(kc), fh_energies=qp,
            basis=SimpleNamespace(ctilde=jnp.asarray(c), band_range=(0, toy.NB)),
            fine=fine, coarse_grid=toy.KC, n_val=NV, window_start=LO,
            a_band_index=None, mesh_xy=mesh, log_fn=lambda *a: None, **kw)
    return dip, fine


def _exact_position(h, fine):
    """⟨ĩ|r̂|j̃⟩ (2, nq, nb, nb) of the fine states, from ALL orbital eigenstates."""
    co = np.asarray(fine.coeffs_fi)                       # (nq, orbital, window)
    out = []
    for i, k in enumerate(fine.q_frac):
        ph = lambda r: np.exp(2j * np.pi * (k[0] * r[0] + k[1] * r[1]))  # noqa: E731
        hk = sum(ph(r) * x for r, x in h.items())
        dk = [sum(1j * r[a] * ph(r) * x for r, x in h.items()) for a in (0, 1)]
        e, u = np.linalg.eigh(hk)
        vm = np.stack([u.conj().T @ d @ u for d in dk])
        de = e[:, None] - e[None, :]
        off = ~np.eye(len(e), dtype=bool)
        r = np.where(off[None], vm / (1j * np.where(off, de, 1.0))[None], 0.0)
        w = u.conj().T @ co[i]                             # ⟨exact n|fine m⟩
        out.append(np.einsum("nm,anl,lk->amk", w.conj(), r, w))
    return np.stack(out, 1)


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(scope="module")
def runs():
    mesh, h = toy._mesh(), toy._model(hop=HOP)
    kc = toy._grid(toy.KC)
    c, e, v = toy._states(h, kc)
    qp = _qp(e, kc)
    out = {"h": h}
    out["qp"], out["fine_qp"] = _run(mesh, c, e, v, qp, full_window=True, qp_states=True)
    out["qp_off"], _ = _run(mesh, c, e, v, qp, full_window=True)      # guard off: pre-fix
    out["dft"], out["fine_dft"] = _run(mesh, c, e, v, e, full_window=True)
    out["dft_default"], _ = _run(mesh, c, e, v, e)
    return out


def test_qp_states_dipoles_are_the_exact_position_and_the_rq_twin_is_not(runs):
    exact = 1j * _exact_position(runs["h"], runs["fine_qp"])[:, :, NV:, :NV]
    fixed = _rel(runs["qp"]["d_alpha"][:2], exact)
    rq = _rel(runs["qp"]["d_alpha_rq"][:2], exact)
    floor = _rel(runs["dft"]["d_alpha"][:2],
                 1j * _exact_position(runs["h"], runs["fine_dft"])[:, :, NV:, :NV])
    # Measured: floor 0.24 %, fixed 0.41 %, Rayleigh-quotient twin 4.95 %.
    assert floor < 0.005 and fixed < 0.01
    assert rq > 0.03 and rq > 5.0 * fixed
    # The guard off (qp_states False) IS the pre-fix conversion.
    assert np.array_equal(runs["qp_off"]["d_alpha"], runs["qp"]["d_alpha_rq"])
    assert runs["qp"]["route"].endswith(DENSIFIED_POSITION_FORM)
    assert DENSIFIED_POSITION_FORM not in runs["qp_off"]["route"]


def test_dft_path_keeps_v_over_dE_bitwise(runs):
    for key in ("dft", "dft_default"):
        dip = runs[key]
        assert "rfix" not in dip and "d_alpha_rq" not in dip
    a, b = runs["dft"], runs["dft_default"]
    assert np.array_equal(a["d_alpha"], b["d_alpha"]) and np.array_equal(a["de_cv"], b["de_cv"])
    vw, ew = a["v_window"], a["e_dft_window"]
    de = ew[:, NV:, None] - ew[:, None, :NV]
    assert np.array_equal(vw[:, :, NV:, :NV] / de[None], a["d_alpha"])
    assert set(a) - set(b) == {"v_window", "e_dft_window", "r_window", "pair_rule", "pair_gap_ev"}


def test_r_window_is_the_window_position(runs):
    nb = HI - LO
    off = ~np.eye(nb, dtype=bool)
    for key, fine_key in (("qp", "fine_qp"), ("dft", "fine_dft")):
        rw = runs[key]["r_window"]
        assert rw.shape == (3, np.prod(toy.KF), nb, nb)
        assert np.abs(rw - np.conj(np.swapaxes(rw, -1, -2))).max() <= 1e-14 * np.abs(rw).max()
        if key == "dft":       # no intraband position in the velocity's eigenbasis;
            assert np.all(rw[:, :, np.arange(nb), np.arange(nb)] == 0)  # QP: rotated
        exact = _exact_position(runs["h"], runs[fine_key])
        assert _rel(rw[:2][..., off], exact[..., off]) < 0.01
    assert np.array_equal(runs["qp"]["d_alpha"], 1j * runs["qp"]["r_window"][:, :, NV:, :NV])
    assert _rel(1j * runs["dft"]["r_window"][:, :, NV:, :NV], runs["dft"]["d_alpha"]) < 1e-12


def test_pair_rule_is_tdagws():
    td_system = pytest.importorskip("tdagw.td_system")
    rng = np.random.default_rng(7)
    nq, nb, nv = 4, 6, 3
    a = rng.standard_normal((3, nq, nb, nb)) + 1j * rng.standard_normal((3, nq, nb, nb))
    v = a + np.conj(np.swapaxes(a, -1, -2))
    e = np.sort(rng.random((nq, nb)), axis=1) * 0.02
    e[:, nv:] += 0.5                                      # a gap above the rule scale
    e[:, 1] = e[:, 0] + 1e-4                              # a near-degenerate vv' pair
    gap = 0.05 / 13.6056980659
    for rule in ("mask", "cut", "cv"):
        mine = _position_from_velocity_rule(v, e, nv, rule=rule, gap_ry=gap)
        theirs, _ = td_system.position_from_velocity(v, e, nv, rule=rule, gap_ry=gap)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-13 * np.abs(theirs).max())


def test_rotation_record_and_refusal(runs):
    rf = runs["qp"]["rfix"]
    assert rf["applied"] and rf["pair_rule"] == "mask"
    assert rf["max_offblock_mixing"] < 1e-2                  # measured 1.6e-3
    assert abs(rf["window_sum_eig_over_fixed"] - 1.0) < 1e-4  # measured 1.8e-5
    assert rf["lowest_dE_rq_minus_eig_meV_max"] > 0.0
    with pytest.raises(ValueError, match="GATE densified_pair_rule"):
        mesh, h = toy._mesh(), toy._model(hop=HOP)
        kc = toy._grid(toy.KC)
        c, e, v = toy._states(h, kc)
        _run(mesh, c, e, v, e, pair_rule="nearest")
