"""Bit-identity and measured certificates for causal crossing tables."""

import numpy as np
import pytest

import minimax


_CELLS = {
    (10.0, 1.0e-4): (
        60, 2.0088175279399608e-5, 1.0000000000006994,
        "61483406d5cd05b55e4d4bf179d1dcc524894c950d578c993280ed7941b07966"),
    (10.0, 1.0e-5): (
        71, 2.374999991228144e-6, 1.0000000000000726,
        "6b22b610c1cdd0a3e15d44697f856bdaa470c9f121b70883eee91b83c5087a71"),
    (20.0, 1.0e-4): (
        64, 2.3125000005563834e-5, 0.9999999999999817,
        "6c3b24ae7ac8d7426ec0ac946818ed96ab8b529d6bea76b5860c7500ce4f10eb"),
    (20.0, 1.0e-5): (
        78, 2.000000034363403e-6, 0.999999999999898,
        "1b9b72e32392513240880247b8572b1aa2983e7fe5266fc9185bd4b3b40feef2"),
    (40.0, 1.0e-4): (
        122, 2.3124999951162906e-5, 1.000000000003113,
        "3f65843794eda55312b04d58f01278d69a1c555e22fb38222eaa711f3bed85cf"),
    (40.0, 1.0e-5): (
        147, 2.249999992431917e-6, 1.0000000000000548,
        "6fafaac3e9a2589b856b3f9a7c2ec8972f3bbeb533507138feca48f31ce8b8b8"),
    (64.0, 1.0e-4): (
        189, 2.3124999969925675e-5, 1.0000000000029436,
        "78e5f87fa9d919c88be21455594e53e418c1657e184e7f7ac864069823f30a06"),
    (64.0, 1.0e-5): (
        228, 2.3749999430444646e-6, 1.0000000000063576,
        "0c13e6bd0521ffd9bd9f3fab7600df76ed6935e49126602ab882136bc425692e"),
    (96.0, 1.0e-4): (
        277, 2.375000006660244e-5, 0.99999999999975,
        "382fd50232f8dde7a4505fc2fde7bc8149c1780292a975d45f40daccdbce857b"),
    (96.0, 1.0e-5): (
        335, 2.25000005671383e-6, 0.9999999999998421,
        "f7c289f4c72da65dbb506b5e07f499b98208dae02f2548f84430b4dd6f8aa486"),
}


@pytest.mark.parametrize(
    "cell, expected", _CELLS.items(),
    ids=[f"A{A:g}-eps{eps:.0e}" for A, eps in _CELLS],
)
def test_causal_crossing_payloads_are_bit_pinned_and_measured(cell, expected):
    """Every shipped cell retains its numerical payload and certificate."""
    A, error_bound = cell
    nodes, dense_error, kappa0, payload_sha = expected
    entries = {
        (entry.range_max, entry.error_bound): entry
        for entry in minimax.catalog().for_family("crossing_causal")
    }
    assert set(entries) == set(_CELLS)
    entry = entries[cell]
    tau, alpha, certified_error, loaded_kappa0, _table_hash = (
        minimax.load_table(entry))

    assert entry.node_count == tau.size == alpha.size == nodes
    assert entry.raw["measured_family_error"] == dense_error
    assert certified_error == entry.raw["max_error"] <= error_bound
    assert loaded_kappa0 == entry.raw["kappa0"] == kappa0
    assert minimax.payload_sha256(tau, alpha) == payload_sha
    assert entry.raw["payload_sha256"] == payload_sha
    assert np.all(tau > 0.0)
    assert np.array_equal(alpha, -1.0j * np.real(1.0j * alpha))
    assert np.all(np.real(1.0j * alpha) > 0.0)
    assert set(entry.raw["provenance"]) == {
        "tool", "tool_sha256", "generator_commit", "numpy", "scipy",
        "python", "certifier", "backend_sha256",
    }
