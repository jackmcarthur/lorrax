"""LORRAX_DIAG_BISPINOR_CARRIER: each mode lifts charge and current as named.

normalized_rkb (default): one isometric carrier; raw: norm 1 + (alpha/2)^2|K|^2
per plane wave; pauli: charge = [psi_L; 0] exactly, current = raw lift.  Every
mode carries its own provenance stamp and an unknown value refuses.
"""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _lifts(monkeypatch, mode):
    from common.bispinor_init import HALFALPHA, lift_to_4spinor
    from common.four_current_model import resolve_four_current_representation
    if mode is None:
        monkeypatch.delenv("LORRAX_DIAG_BISPINOR_CARRIER", raising=False)
    else:
        monkeypatch.setenv("LORRAX_DIAG_BISPINOR_CARRIER", mode)
    rng = np.random.default_rng(7)
    nk, nb, ng = 2, 3, 40
    psi = rng.normal(size=(nk, nb, 2, ng)) + 1j * rng.normal(size=(nk, nb, 2, ng))
    gvecs = rng.integers(-6, 7, size=(nk, ng, 3)).astype(np.float64)
    kvecs = rng.uniform(-0.5, 0.5, size=(nk, 3))
    bvec = 1.16 * np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    rep = resolve_four_current_representation(True, "bare_transverse")
    charge = np.asarray(lift_to_4spinor(psi, gvecs, kvecs, bvec,
                                        representation=rep.charge_lift))
    current = np.asarray(lift_to_4spinor(psi, gvecs, kvecs, bvec,
                                         representation=rep.current_lift))
    K2 = np.sum(((gvecs + kvecs[:, None, :]) @ bvec) ** 2, axis=-1)
    gain = (1.0 + HALFALPHA ** 2 * K2)[:, None, :]          # raw norm per G
    pw = lambda x: np.sum(np.abs(x) ** 2, axis=2)          # spinor-summed density per G
    return rep, psi, charge, current, gain, pw


def test_diag_carrier_modes(monkeypatch):
    stamps = {}
    for mode in (None, "normalized_rkb", " RAW ", "pauli"):
        rep, psi, charge, current, gain, pw = _lifts(monkeypatch, mode)
        name = (mode or "normalized_rkb").strip().lower()
        stamps[name] = rep.charge_representation
        assert rep.charge_bispinor and rep.current_bispinor and rep.scalar_head_bispinor
        if name == "normalized_rkb":
            assert rep.charge_lift == rep.current_lift == "normalized_rkb"
            np.testing.assert_allclose(pw(charge), pw(psi), rtol=1e-13)
            np.testing.assert_array_equal(charge, current)
        elif name == "raw":
            assert rep.charge_lift == rep.current_lift == "raw"
            np.testing.assert_array_equal(charge[:, :, :2], psi)
            np.testing.assert_allclose(pw(charge), gain * pw(psi), rtol=1e-13)
        else:
            assert (rep.charge_lift, rep.current_lift) == ("large", "raw")
            np.testing.assert_array_equal(charge[:, :, :2], psi)
            assert not np.any(charge[:, :, 2:])
            np.testing.assert_array_equal(current[:, :, :2], psi)
            np.testing.assert_allclose(pw(current), gain * pw(psi), rtol=1e-13)
    assert stamps == {
        "normalized_rkb": "normalized_rkb_four_current_v1",
        "raw": "raw_kinetic_balance_four_current_diag_v1",
        "pauli": "large_block_pauli_charge_v1"}


def test_diag_carrier_refusals(monkeypatch):
    from common.bispinor_init import kinetic_balance_lift_jet
    from common.four_current_model import resolve_four_current_representation
    monkeypatch.setenv("LORRAX_DIAG_BISPINOR_CARRIER", "isometric")
    with pytest.raises(ValueError, match="LORRAX_DIAG_BISPINOR_CARRIER"):
        resolve_four_current_representation(True, "bare_transverse")
    # A scalar run never reads the switch.
    assert resolve_four_current_representation(False, None).charge_lift is None
    with pytest.raises(ValueError, match="large-block"):
        kinetic_balance_lift_jet(np.zeros((1, 2, 3), complex), np.zeros((3, 3)),
                                 representation="large")
