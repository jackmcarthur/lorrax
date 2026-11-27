"""What ``full_bse = true`` says it runs, and which eigensolver options it refuses.

``bse_lanczos.solve_bse_sharded(tda=False)`` returns before it reads any Davidson
option: the full route is the dense structure-preserving solve, not an iteration.  The deck layer
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
    full = _rows(["full_bse = true"])
    assert full["solver"].value == "dense" and full["solver"].source == "derived"
    text = " ".join(f"{row.value} {row.effect}" for row in full.values())
    assert "preconditioner" not in text and "Krylov" not in text
    assert "davidson_tol_mev" not in full
    tda = _rows([])                                   # the default route is unchanged
    assert tda["solver"].value == "davidson" and "exact diagonal preconditioner" in tda["solver"].effect
    assert "davidson_tol_mev" in tda


@pytest.mark.parametrize("line, key", [
    ("davidson_precond = bare", "davidson_precond"), ("davidson_margin = 4", "davidson_margin"),
    ("davidson_olsen = true", "davidson_olsen")])
def test_full_bse_refuses_unread_eigensolver_keys(line, key, capsys):
    """An eigensolver key set away from its default refuses under full_bse; the TDA run still reads it
    (``solver``, ``davidson_tol_mev`` and the iteration cap are deck keys other BSE drivers
    share, which the existing unread-key rule leaves alone on every route)."""
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", line])])
    err = capsys.readouterr().err
    assert f"`{key}`" in err and "not read by full_bse = true" in err
    if key.startswith("davidson"):
        bse_jax.parse_args(["-i", _toy_deck([line])])             # the default route reads it


@pytest.mark.parametrize("line, key", [("solver_max_iter = 50", "solver_max_iter"),
                                       ("davidson_tol_mev = 0.1", "davidson_tol_mev")])
def test_dense_full_bse_refuses_the_unread_stop_keys(line, key, capsys):
    """The dense solve has no iteration: ``solver_max_iter`` and ``davidson_tol_mev`` set by the input
    file refuse, and ``full_bse_solver = davidson`` (which reads both) takes them."""
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", line])])
    err = capsys.readouterr().err
    assert f"`{key}`" in err and "dense solve has no iteration" in err and "full_bse_solver = davidson" in err
    bse_jax.parse_args(["-i", _toy_deck(["full_bse = true", "full_bse_solver = davidson", line])])
    bse_jax.parse_args(["-i", _toy_deck([line])])                       # and the Tamm-Dancoff run
