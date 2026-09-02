"""The MPA scalar-head record for ``head_correction=off``."""

from types import SimpleNamespace

import numpy as np

from gw.mpa import model


def test_head_off_publishes_exact_zero_without_calling_pole_solver(monkeypatch):
    """An absent physical channel is not sent through an ill-posed fit."""
    captured = {}

    def refuse_fit(*args, **kwargs):
        raise AssertionError("the scalar pole solver must not see zero data")

    def capture_write(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(model.fit_driver, "fit_scalar_samples", refuse_fit)
    monkeypatch.setattr(
        model.mpa_store, "write_head_fit_collective", capture_write)
    samples = tuple(
        SimpleNamespace(vc0=0.0j, wcoul0=0.0j) for _ in range(8))
    z = np.linspace(0.1, 0.8, 8).astype(np.complex128) * (1.0 + 1.0j)

    model._fit_head_samples(
        "fit.h5", samples, z, 4, "grid", object(),
        model="head_off_zero", solve="loewner")

    args = captured["args"]
    np.testing.assert_array_equal(args[1], z)
    np.testing.assert_array_equal(args[2], np.zeros_like(z))
    np.testing.assert_array_equal(args[3], np.full(4, -1.0j))
    np.testing.assert_array_equal(args[4], np.zeros(4, np.complex128))
    assert captured["kwargs"]["model"] == "head_off_zero"
    assert captured["kwargs"]["fit_max_abs_residual"] == 0.0
    assert captured["kwargs"]["fit_provenance"] == {
        "solve_mode": "exact_zero", "n_valid": 4}


def test_head_off_zero_refuses_nonzero_samples(monkeypatch):
    """The exact-zero provenance cannot hide an accidentally active head."""
    monkeypatch.setattr(
        model.mpa_store, "write_head_fit_collective",
        lambda *args, **kwargs: None)
    samples = (SimpleNamespace(vc0=0.0j, wcoul0=1.0j),)

    try:
        model._fit_head_samples(
            "fit.h5", samples, np.asarray([0.2j]), 1, "grid", object(),
            model="head_off_zero", solve="loewner")
    except ValueError as exc:
        assert "nonzero scalar-head sample" in str(exc)
    else:
        raise AssertionError("nonzero head data was stamped as head_off_zero")
