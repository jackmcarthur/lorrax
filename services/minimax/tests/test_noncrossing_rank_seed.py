"""Deterministic rank seeding for on-demand noncrossing rules."""

import numpy as np

from minimax import solver


def _rank_solver(calls, first_passing):
    def solve(rank, _range):
        calls.append(rank)
        error = 1.0e-7 if rank >= first_passing else 1.0e-3
        value = np.asarray([float(rank)])
        return value, -value, error

    return solve


def test_rank_seed_skips_the_known_failing_prefix(monkeypatch):
    calls = []
    monkeypatch.setattr(solver, "predict_N_noncrossing", lambda _r, _e: 8)
    monkeypatch.setattr(solver, "_nc_solve_varpro", _rank_solver(calls, 9))

    tau, weights, rank, error = solver.noncrossing_grids(
        100.0, 1.0e-6, N_start=2, N_max=20)

    assert calls == [8, 9]
    assert rank == 9 and error == 1.0e-7
    np.testing.assert_array_equal(tau, [9.0])
    np.testing.assert_array_equal(weights, [-9.0])


def test_passing_rank_seed_walks_down_to_the_first_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(solver, "predict_N_noncrossing", lambda _r, _e: 8)
    monkeypatch.setattr(solver, "_nc_solve_varpro", _rank_solver(calls, 7))

    _tau, _weights, rank, error = solver.noncrossing_grids(
        100.0, 1.0e-6, N_start=2, N_max=20)

    assert calls == [8, 7, 6]
    assert rank == 7 and error == 1.0e-7
