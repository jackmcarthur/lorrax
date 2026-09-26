"""bse_jax's CLI is strict: unknown, retired and route-ignored flags refuse.

Before 2026-09-25 the driver parsed with ``parse_known_args`` and rebuilt an
argv for ``bse_feast.main``: ``--nval 8`` ran 4v4c, and the default FEAST
route dropped ``--eqp``, ``--n-eig``, ``--n-occ`` and the solver flags, so a
QP run came back on DFT energies (KNOWN_LORRAX_ISSUES, lane IC).  The routes
now take ``bse_feast.run(settings)`` / ``bse_kpm.run(settings)``.
"""
from __future__ import annotations

import pytest


def _bse_jax():
    from bse import bse_jax
    return bse_jax


def _refusal(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        _bse_jax().parse_args(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


def test_a_typo_flag_refuses(capsys):
    err = _refusal(capsys, ["-i", "d.in", "--nval", "8"])
    assert "--nval" in err


def test_a_retired_flag_refuses_by_name(capsys):
    err = _refusal(capsys, ["-i", "d.in", "--ring-timing"])
    assert "--ring-timing is retired" in err


def test_the_feast_route_refuses_the_flags_it_would_drop(capsys):
    err = _refusal(capsys, ["-i", "d.in", "--eqp", "eqp1.dat",
                            "--n-eig", "12"])
    assert "--eqp" in err and "--n-eig" in err and "feast route" in err


def test_the_lanczos_route_refuses_feast_flags(capsys):
    err = _refusal(capsys, ["-i", "d.in", "--lanczos", "--gmres-tol", "0.5"])
    assert "--gmres-tol" in err and "lanczos route" in err


def test_the_routes_accept_their_own_flags():
    route, args = _bse_jax().parse_args(
        ["-i", "d.in", "--lanczos", "--eqp", "eqp1.dat", "--n-eig", "12",
         "--band-degeneracy", "off"])
    assert route == "lanczos" and args.eqp == "eqp1.dat" and args.n_eig == 12
    route, _ = _bse_jax().parse_args(["-i", "d.in", "--gmres-tol", "0.5"])
    assert route == "feast"
    route, _ = _bse_jax().parse_args(["-i", "d.in", "--kpm-dos",
                                      "--kpm-n-moments", "50"])
    assert route == "kpm"


def test_the_delegate_settings_take_the_parsers_defaults():
    from bse import bse_feast, bse_kpm
    s = bse_feast.settings("d.in", n_quad2=12)
    assert s.n_quad2 == 12 and s.n_lanczos_max == 50 and s.feast_iter == 2
    with pytest.raises(TypeError, match="unknown settings"):
        bse_feast.settings("d.in", n_quad3=1)
    k = bse_kpm.settings("d.in", n_windows=3)
    assert k.n_windows == 3 and k.n_energy_pts == 2000
