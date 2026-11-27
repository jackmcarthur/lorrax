"""The input keys of the retired FEAST route refuse by name.

An input key no driver reads is left alone (``bse.bse_deck``), so without
``RETIRED_KEYS`` a ``feast = true`` input would silently run the Tamm-Dancoff
Davidson eigensolve.  ``GATE bse_retired_key`` refuses it in the ``bse_jax``
parser, with the key, what it selected and what to do instead.

``JAX_PLATFORMS=cpu python -m pytest tests/test_bse_retired_keys_feast_cpu.py``.
"""
import pytest

pytest.importorskip("jax")

from bse import bse_jax  # noqa: E402
from bse.bse_deck import DeckError, RETIRED_KEYS, refuse_retired_keys  # noqa: E402
from tests.test_bse_default_solver import _toy_deck  # noqa: E402

FEAST_KEYS = (
    "feast", "feast_n_lanczos", "feast_buffer", "feast_n_quad1", "feast_n_quad2",
    "feast_quadrature", "feast_ritz_count", "feast_tol_mev", "feast_iter",
    "feast_window1", "feast_window2", "gmres_max_iter", "gmres_tol", "gmres_seed",
    "gmres_fp32")


def test_feast_true_refuses_in_the_bse_jax_parser(capsys):
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck(["feast = true", "full_bse = true"])])
    err = capsys.readouterr().err
    assert "GATE bse_retired_key" in err and "`feast`" in err
    assert "retired 2026-11-27" in err and "full_bse_solver = davidson" in err


@pytest.mark.parametrize("key", FEAST_KEYS)
def test_every_feast_key_is_retired(key):
    assert key in RETIRED_KEYS
    with pytest.raises(DeckError, match=f"GATE bse_retired_key: `{key}`"):
        refuse_retired_keys({key: "1"}, {key})


def test_the_kpm_density_of_states_route_is_untouched():
    _, args = bse_jax.parse_args(["-i", _toy_deck(["kpm_dos = true", "kpm_window_count = 3"])])
    assert args.kpm_dos and args.kpm_window_count == 3
    assert not any(key.startswith("kpm_") for key in RETIRED_KEYS)
