"""Focused gates for orbital-magnetization's device spin reduction."""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from psp.orbital_magnetization import _spin_z_per_band


def _host_spin_z(psi_G):
    psi = np.asarray(psi_G, dtype=np.complex128)
    return (
        np.abs(psi[:, 0]) ** 2 - np.abs(psi[:, 1]) ** 2
    ).sum(axis=1).real


def test_device_spin_z_matches_host_for_hostile_complex_unnormalized_spinors():
    rng = np.random.default_rng(20260830)
    psi = (rng.normal(size=(9, 2, 73))
           + 1j * rng.normal(size=(9, 2, 73))).astype(np.complex128)
    # Exercise non-normalised bands, very unequal components, complex phases,
    # exact zeros, and magnitudes spanning twelve orders without changing the
    # declared c128 calculation.
    scales = np.asarray(
        [1.0e-6, 1.0e6, 3.0e-3, 4.0e2, 1.0, 9.0e-5, 7.0e3, 2.0, 0.25])
    phases = np.exp(1j * np.linspace(-2.7, 2.4, psi.shape[0]))
    psi *= (scales * phases)[:, None, None]
    psi[2, 0] *= 1.0e4
    psi[5, 1] *= 1.0e-4
    psi[7, :, 11:19] = 0.0

    got = np.asarray(_spin_z_per_band(jnp.asarray(psi)))
    expected = _host_spin_z(psi)

    assert got.shape == (psi.shape[0],)
    assert got.dtype == np.float64
    assert np.all(np.isreal(got))
    np.testing.assert_allclose(got, expected, rtol=8.0e-15, atol=1.0e-12)


def test_device_spin_z_respects_zeroed_g_padding():
    rng = np.random.default_rng(83002)
    psi = (rng.normal(size=(5, 2, 41))
           + 1j * rng.normal(size=(5, 2, 41))).astype(np.complex128)
    mask = np.arange(psi.shape[-1]) < 29
    masked = psi * mask[None, None, :]

    got = np.asarray(_spin_z_per_band(jnp.asarray(masked)))
    expected = _host_spin_z(psi[..., :29])
    np.testing.assert_allclose(got, expected, rtol=8.0e-15, atol=1.0e-13)


@pytest.mark.parametrize("shape", [(4, 1, 13), (4, 3, 13), (4, 2, 3, 5)])
def test_device_spin_z_refuses_non_two_spinor_carriers(shape):
    with pytest.raises(ValueError, match=r"shape \(nb,2,nG\)"):
        _spin_z_per_band(jnp.zeros(shape, dtype=jnp.complex128))


def test_orbmag_routes_do_not_materialize_full_spinor_carriers_on_host():
    source_path = (Path(__file__).resolve().parents[1]
                   / "src" / "psp" / "orbital_magnetization.py")
    tree = ast.parse(source_path.read_text())
    forbidden = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr == "asarray"
                and node.args):
            continue
        operand = ast.unparse(node.args[0])
        if operand in {"psi_G", "U_val_G", "psi_ibz[i]"}:
            forbidden.append((node.lineno, operand))

    assert forbidden == [], (
        "orbital-magnetization must transfer only the reduced (nb,) spin "
        f"result, never a full wavefunction carrier: {forbidden}")

    route_calls = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in {"velocity_at_k", "run_ibz", "run_sternheimer_orbmag"}:
            continue
        route_calls[node.name] = sum(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_spin_z_per_band"
            for child in ast.walk(node))
    assert route_calls == {
        "velocity_at_k": 1,
        "run_ibz": 1,
        "run_sternheimer_orbmag": 1,
    }
