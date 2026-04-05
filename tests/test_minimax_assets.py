from __future__ import annotations

import numpy as np

from gw import minimax_screening as ms


def test_find_shipped_table_entry_prefers_smallest_range_then_loosest_error(monkeypatch):
    catalog = {
        "tables": [
            {
                "family": "noncrossing",
                "range_max": 100.0,
                "error_bound": 2.0e-7,
                "node_count": 30,
                "file": "noncrossing/strict.npz",
            },
            {
                "family": "noncrossing",
                "range_max": 100.0,
                "error_bound": 1.0e-6,
                "node_count": 28,
                "file": "noncrossing/loose.npz",
            },
            {
                "family": "noncrossing",
                "range_max": 200.0,
                "error_bound": 1.0e-6,
                "node_count": 20,
                "file": "noncrossing/larger_range.npz",
            },
        ],
    }
    monkeypatch.setattr(ms, "_load_shipped_minimax_catalog", lambda: catalog)

    entry = ms._find_shipped_table_entry(
        "noncrossing",
        range_value=80.0,
        target_error=1.0e-6,
        max_nodes=64,
    )

    assert entry is not None
    assert entry["file"] == "noncrossing/loose.npz"


def test_find_shipped_table_entry_filters_crossing_metadata(monkeypatch):
    catalog = {
        "tables": [
            {
                "family": "crossing",
                "target_kind": "fermi",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-3,
                "node_count": 50,
                "file": "crossing/wrong_kind.npz",
            },
            {
                "family": "crossing",
                "target_kind": "hgl",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-2,
                "node_count": 50,
                "file": "crossing/wrong_epsq.npz",
            },
            {
                "family": "crossing",
                "target_kind": "hgl",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-3,
                "node_count": 70,
                "file": "crossing/too_many_nodes.npz",
            },
            {
                "family": "crossing",
                "target_kind": "hgl",
                "range_max": 80.0,
                "error_bound": 1.0e-6,
                "eps_q": 1.0e-3,
                "node_count": 48,
                "file": "crossing/good.npz",
            },
        ],
    }
    monkeypatch.setattr(ms, "_load_shipped_minimax_catalog", lambda: catalog)

    entry = ms._find_shipped_table_entry(
        "crossing",
        range_value=60.0,
        target_error=1.0e-6,
        max_nodes=64,
        target_kind="hgl",
        eps_q=1.0e-3,
    )

    assert entry is not None
    assert entry["file"] == "crossing/good.npz"


def test_solve_laplace_minimax_interval_uses_shipped_table_and_rescales(monkeypatch):
    tau_hat = np.array([1.0, 2.0], dtype=np.float64)
    alpha_hat = np.array([0.25, 0.5], dtype=np.float64)
    err_hat = 5.0e-7

    monkeypatch.setattr(
        ms,
        "_pick_shipped_table",
        lambda *args, **kwargs: (tau_hat, alpha_hat, err_hat),
    )

    def _unexpected_solver(*args, **kwargs):
        raise AssertionError("Exact noncrossing solver should not run when a shipped table matches.")

    monkeypatch.setattr(ms, "_solve_noncrossing_scaled_cached", _unexpected_solver)

    quad = ms.solve_laplace_minimax_interval(2.0, 20.0, target_error=1.0e-6, max_nodes=64)

    np.testing.assert_allclose(quad.tau, tau_hat / 2.0)
    np.testing.assert_allclose(quad.alpha, alpha_hat / 2.0)
    assert quad.max_error == err_hat / 2.0


def test_solve_phase_minimax_bandwidth_falls_back_when_no_shipped_table(monkeypatch):
    tau_hat = np.array([0.5, 1.5], dtype=np.float64)
    alpha_hat = np.array([0.1, 0.2], dtype=np.float64)
    err_hat = 9.0e-7

    monkeypatch.setattr(ms, "_pick_shipped_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ms,
        "_solve_crossing_scaled_cached",
        lambda *args, **kwargs: (tau_hat, alpha_hat, err_hat),
    )

    quad = ms.solve_phase_minimax_bandwidth(
        83.0,
        target_error=1.0e-6,
        max_nodes=500,
        eps_q=1.0e-3,
        target_kind="hgl",
    )

    np.testing.assert_allclose(quad.tau, tau_hat)
    np.testing.assert_allclose(quad.alpha, alpha_hat)
    assert quad.max_error == err_hat
    assert quad.target_kind == "hgl"
