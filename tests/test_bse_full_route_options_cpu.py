"""What ``full_bse = true`` says it runs, and which eigensolver options it refuses.

``bse_lanczos.solve_bse_sharded(tda=False)`` (its own default, and ``full_bse_solver = dense`` in an
input file) returns before it reads any Davidson, Lanczos or TRLan
option: that route is the dense structure-preserving solve, not an iteration.  The deck layer
(``bse_jax``) must therefore (1) print that route in the SETTINGS table, not "davidson ... exact
diagonal preconditioner", and (2) refuse an eigensolver key set away from its default, the unread-key
rule of the eigensolve route.  The TDA default is unchanged.

``JAX_PLATFORMS=cpu python -m pytest tests/test_bse_full_route_options_cpu.py``.
"""
import pytest

pytest.importorskip("jax")

from bse import bse_jax  # noqa: E402
from tests.test_bse_default_solver import _toy_deck  # noqa: E402


def _rows(lines):
    """``{key: Setting}`` of the SETTINGS table a deck with ``lines`` gives."""
    _, args = bse_jax.parse_args(["-i", _toy_deck(lines)])
    return {row.key: row for row in bse_jax.settings_rows(args, source=lambda *keys: "default")}


def test_full_bse_settings_name_the_dense_route():
    full = _rows(["full_bse = true", "full_bse_solver = dense"])
    assert full["solver"].value == "dense" and full["solver"].source == "derived"
    text = " ".join(f"{row.value} {row.effect}" for row in full.values())
    assert "preconditioner" not in text and "Krylov" not in text
    assert "davidson_tol_mev" not in full and "lanczos_rtol" not in full
    tda = _rows([])                                   # the default route is unchanged
    assert tda["solver"].value == "davidson" and "exact diagonal preconditioner" in tda["solver"].effect
    assert "davidson_tol_mev" in tda


@pytest.mark.parametrize("line, key", [
    ("davidson_precond = bare", "davidson_precond"), ("davidson_margin = 4", "davidson_margin"),
    ("davidson_olsen = true", "davidson_olsen"), ("trlan_m_max = 40", "trlan_m_max"),
    ("lanczos_rtol = 1e-6", "lanczos_rtol")])
def test_full_bse_refuses_unread_eigensolver_keys(line, key, capsys):
    """An eigensolver key set away from its default refuses under full_bse; the TDA run still reads it
    (``solver``, ``block_size``, ``davidson_tol_mev`` and the iteration cap are deck keys other BSE drivers
    share, which the existing unread-key rule leaves alone on every route)."""
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", "full_bse_solver = dense", line])])
    err = capsys.readouterr().err
    assert f"`{key}`" in err and "not read by full_bse_solver = dense" in err
    if key.startswith("davidson"):
        bse_jax.parse_args(["-i", _toy_deck([line])])             # the default route reads it
        bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", line])])   # and so does the default full solver


@pytest.mark.parametrize("route", ("full_bse = true\nfull_bse_solver = dense", "full_bse = true",
                                   "full_bse = true\nfull_bse_solver = davidson"))
@pytest.mark.parametrize("line, key, instead", [
    ("solver = block_lanczos", "solver", "full_bse_solver"), ("solver = trlan", "solver", "full_bse_solver"),
    ("block_size = 4", "block_size", "Lanczos block")])
def test_full_bse_refuses_the_shared_keys_no_full_route_reads(route, line, key, instead, capsys):
    """``solver`` and ``block_size`` are shared with the other BSE drivers, which the unread-key rule
    leaves alone; neither full-BSE route reads them (``block_size`` used to divide the Davidson iteration
    cap).  The refusal names what to set instead, and the Tamm-Dancoff run still reads both."""
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck(route.split("\n") + [line])])
    err = capsys.readouterr().err
    assert f"`{key}`" in err and "not read by full_bse = true" in err and instead in err
    bse_jax.parse_args(["-i", _toy_deck([line])])


@pytest.mark.parametrize("line, key", [("solver_max_iter = 50", "solver_max_iter"),
                                       ("davidson_tol_mev = 0.1", "davidson_tol_mev")])
def test_dense_full_bse_refuses_the_unread_stop_keys(line, key, capsys):
    """The dense solve has no iteration: ``solver_max_iter`` and ``davidson_tol_mev`` set by the input
    file refuse, and ``full_bse_solver = davidson`` (which reads both) takes them."""
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", "full_bse_solver = dense", line])])
    err = capsys.readouterr().err
    assert f"`{key}`" in err and "dense solve has no iteration" in err and "full_bse_solver = davidson" in err
    bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", "full_bse_solver = davidson", line])])
    bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", line])])    # the default full solver reads both
    bse_jax.parse_args(["-i", _toy_deck([line])])                       # and the Tamm-Dancoff run
