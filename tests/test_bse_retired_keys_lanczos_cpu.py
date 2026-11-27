"""Retired Lanczos routes of the BSE refuse by name (``GATE bse_retired_key``).

The input-file reader leaves a key a driver does not read alone, so after the thick-restart
Lanczos (``solver = trlan``, ``trlan_m_max``, ``trlan_n_keep``) was retired on 2026-11-27 an
input file that still names it would run the Davidson in silence.  Each retired key and each
retired ``solver`` value refuses in ``bse_jax``'s parser and in ``bse.exciton_bands.deck_settings``.

``JAX_PLATFORMS=cpu python -m pytest tests/test_bse_retired_keys_lanczos_cpu.py``.
"""
import tempfile

import pytest

pytest.importorskip("jax")

from bse import bse_deck, bse_jax  # noqa: E402
from bse import exciton_bands as exb  # noqa: E402
from tests.test_bse_default_solver import _toy_deck  # noqa: E402

#: retired input line -> the key the refusal names
RETIRED_LINES = [
    ("trlan_m_max = 40", "trlan_m_max"),
    ("trlan_n_keep = 12", "trlan_n_keep"),
    ("solver = trlan", "solver = trlan"),
]


def _exciton_bands_deck(lines):
    fd = tempfile.NamedTemporaryFile("w", suffix=".in", delete=False)
    fd.write("[cohsex]\nsys_dim = 3\nnval = 5\nncond = 5\ncentroids_file = centroids_frac.txt\n"
             "number_val_bands_fine = 3\nnumber_cond_bands_fine = 3\n"
             + "".join(f"{line}\n" for line in lines))
    fd.close()
    return fd.name


@pytest.mark.parametrize("line, named", RETIRED_LINES)
def test_bse_jax_refuses_a_retired_key_or_solver(line, named, capsys):
    with pytest.raises(SystemExit):
        bse_jax.parse_args(["-i", _toy_deck([line])])
    err = capsys.readouterr().err
    assert "GATE bse_retired_key" in err and named in err and "retired 2026-11-27" in err


@pytest.mark.parametrize("line, named", RETIRED_LINES)
def test_exciton_bands_refuses_a_retired_key_or_solver(line, named, capsys):
    ap = exb.build_parser()
    args = ap.parse_args(["-i", _exciton_bands_deck([line])])
    with pytest.raises(SystemExit):
        exb.deck_settings(ap, args)
    err = capsys.readouterr().err
    assert "GATE bse_retired_key" in err and named in err


def test_the_davidson_default_and_the_named_davidson_still_parse():
    for lines in ([], ["solver = davidson"]):
        route, args = bse_jax.parse_args(["-i", _toy_deck(lines)])
        assert route == "lanczos" and args.solver == "davidson"


def test_every_retired_key_has_a_remedy():
    for key, (what, fix) in bse_deck.RETIRED_KEYS.items():
        assert what and fix, key
