"""A redundant Loewner pencil certifies its retained solve and full residual."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gw.mpa import fit_driver, model, pade_fit
from gw.mpa.sampling import double_parallel_grid

jax.config.update("jax_enable_x64", True)


def _measure(z, omega, residue):
    return np.sum(2 * omega * residue / (
        np.asarray(z)[..., None] ** 2 - omega ** 2), axis=-1)


def _samples(values):
    return [SimpleNamespace(vc0=0j, wcoul0=value) for value in values]


def test_exact_lower_order_heads_keep_full_witness_and_predict_held_points():
    """Two/four positive poles need not fill an eight-pole requested pencil."""
    z = np.asarray(double_parallel_grid(8, 4., energy_unit="Ry"))
    held = np.r_[1j * np.geomspace(.001, 30, 31),
                 np.linspace(.01, 3.1, 73) + .03j]
    for count in (2, 4):
        omega = np.array([.07, .23, .9, 2.7])[:count]
        residue = np.array([.03, .17, .8, .3])[:count]
        values = _measure(z, omega, residue)
        fitted = fit_driver.fit_scalar_samples(values, z, 8)
        assert fitted["condition"] > 1 / fitted["rcond"]
        assert fitted["rank_retained"] == fitted["n_valid"] == count
        assert 1 <= fitted["condition_retained"] <= 1 / fitted["rcond"]
        assert fitted["backward_error"] < np.sqrt(np.finfo(float).eps)
        head = model.fit_head_samples(_samples(values), z, 8,
            model="qsgw_schur_loewner", solve="loewner",
            moment_m1=np.sum(residue / omega ** 2))
        assert head["provenance"]["condition_full"] == fitted["condition"]
        assert head["provenance"]["rank_retained"] == count
        assert head["diagnostics"]["fit_condition"] == fitted["condition_retained"]
        assert head["diagnostics"]["fit_backward_error"] == fitted["backward_error"]
        actual = _measure(held, head["Omega_p"], head["B_p"])
        relative = np.linalg.norm(actual - _measure(held, omega, residue)) / np.linalg.norm(
            _measure(held, omega, residue))
        assert relative < 1e-9


def test_incompatible_discarded_space_and_rank_zero_still_refuse():
    """A well-conditioned retained block cannot hide a bad full equation."""
    z = np.asarray(double_parallel_grid(8, 4., energy_unit="Ry"))
    # f(x)=x gives rank-one L, but sL has a component outside range(L).
    # A strictly proper pole model cannot solve this original equation.
    fitted = fit_driver.fit_scalar_samples(z ** 2, z, 8)
    assert fitted["rank_retained"] == 1
    assert fitted["condition_retained"] == pytest.approx(1.)
    assert fitted["backward_error"] > .1
    with pytest.raises(ValueError, match="condition/backward-error gate"):
        model.fit_head_samples(_samples(z ** 2), z, 8,
            model="qsgw_schur_loewner", solve="loewner")
    with pytest.raises(ValueError, match="rejected every pole"):
        model.fit_head_samples(_samples(np.ones_like(z)), z, 8,
            model="qsgw_schur_loewner", solve="loewner")


def test_exact_null_singular_values_and_raw_infinity_metadata(monkeypatch, tmp_path):
    """An exact null direction is reported honestly and stays JSON/HDF safe."""
    import h5py
    from file_io.tangential_store import _json

    with monkeypatch.context() as patch:
        patch.setattr(pade_fit, "_loewner_pencil", lambda *args: (
            jnp.diag(jnp.array([2., 0.], dtype=complex)),
            jnp.diag(jnp.array([.5, 0.], dtype=complex))))
        result = pade_fit._loewner_roots(jnp.ones(4), jnp.arange(4), 2, 1e-13)
        _, full_condition, _, _, backward, retained_condition, rank = result
        assert np.isposinf(full_condition)
        assert int(rank) == 1 and float(retained_condition) == 1.
        assert float(backward) == 0.
        patch.setattr(pade_fit, "_loewner_pencil", lambda *args: (
            jnp.zeros((2, 2), dtype=complex), jnp.eye(2, dtype=complex)))
        result = pade_fit._loewner_roots(jnp.ones(4), jnp.arange(4), 2, 1e-13)
        assert int(result[-1]) == 0 and np.isposinf(result[-2])
        assert float(result[4]) == 1.

    # Inject only the diagnostic's exact-null representation into a real
    # lower-rank fit; poles, residues, retained solve and residual stay real.
    real_fit = fit_driver.fit_scalar_samples
    def exact_null_diagnostic(*args, **kwargs):
        return dict(real_fit(*args, **kwargs), condition=float(full_condition))
    monkeypatch.setattr(fit_driver, "fit_scalar_samples", exact_null_diagnostic)
    z = np.asarray(double_parallel_grid(8, 4., energy_unit="Ry"))
    values = _measure(z, np.array([.07, .23]), np.array([.03, .17]))
    head = model.fit_head_samples(_samples(values), z, 8,
        model="qsgw_schur_loewner", solve="loewner")
    assert np.isfinite(head["diagnostics"]["fit_condition"])
    assert json.loads(_json(head))["provenance"]["condition_full"] == "inf"
    with h5py.File(tmp_path / "head.h5", "w") as stream:
        for key, value in head["provenance"].items():
            stream.attrs["fit_" + key] = value
    with h5py.File(tmp_path / "head.h5", "r") as stream:
        assert stream.attrs["fit_condition_full"] == "inf"
        assert stream.attrs["fit_rank_retained"] == 2


def test_other_scalar_solvers_keep_their_original_condition_certificate():
    z = np.asarray(double_parallel_grid(2, 4., energy_unit="Ry"))
    values = _measure(z, np.array([.23, .9]), np.array([.17, .8]))
    for solve in ("companion", "thiele"):
        fitted = fit_driver.fit_scalar_samples(values, z, 2, solve=solve)
        head = model.fit_head_samples(_samples(values), z, 2,
            model="qsgw_schur_loewner", solve=solve)
        assert "condition_retained" not in fitted
        assert "condition_scope" not in head["provenance"]
        assert head["diagnostics"]["fit_condition"] == fitted["condition"]
