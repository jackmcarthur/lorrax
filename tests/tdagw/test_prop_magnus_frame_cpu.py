"""magnus2's two frames (CPU, ~1 min): whole rho (default) and interaction picture.

The whole-rho exponential midpoint (``time_integrators.magnus_midpoint_step``,
dSigma at the average of the end states) conserves the post-pulse energy
E = sum_k w_k [Tr(E^QP rho) + (1/2) Tr(dSigma Delta rho)] up to rounding; the
interaction-picture step on Delta rho (``ip_magnus_step``) splits the band
energies from the coupling and drifts at second order in h.  On the TDGW
fixture (Gamma = 0, its pulse x10, over by 400 as), 400-4000 as:

F1 whole_rho: |E(t) - E(400 as)| <= 1e-9 of the excitation energy at 5 and 2.5 as
F2 interaction: a drift above rounding that falls 4x when h halves (O(h^2))
F3 the default frame of magnus2 is whole_rho (the same run)
R1 magnus_frame = 'whole_rho' with cfm4 or dp54 refuses (td_magnus_frame)
"""
import json
import os
import subprocess
import sys

import pytest

from tdagw.td_config import parse_td_deck

_HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run([sys.executable, os.path.join(_HERE, "_frame_case.py"),
                          str(tmp_path_factory.mktemp("frame")), "10"],
                         capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, out.stderr[-4000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[7:])


def test_whole_rho_conserves_the_energy(frames):
    for dt in ("5", "2.5"):
        assert frames[f"whole_rho_{dt}"]["drift"] <= 1e-9, frames


def test_interaction_frame_drifts_at_second_order(frames):
    d5, d25 = frames["interaction_5"]["drift"], frames["interaction_2.5"]["drift"]
    assert d5 > 10.0 * frames["whole_rho_5"]["drift"], frames
    assert 3.5 <= d5 / d25 <= 4.5, (d5, d25)


def test_default_frame_is_whole_rho(frames):
    assert frames["default_5"]["drift"] == frames["whole_rho_5"]["drift"], frames


@pytest.mark.parametrize("method", ["cfm4", "dp54"])
def test_whole_rho_refused_outside_magnus2(method):
    raw = {"control": {"calc_type": "IP", "timesteps": {"t0": 0, "tf": 10, "dt_measure": 5},
                       "files": {"outdir": "/tmp/x", "load_header_from": "/tmp/y"}},
           "system": {"ibndmin": 1, "ibndmax": 2, "geometry": "slab"},
           "measurements": {"types": ["polarization"]},
           "lorrax": {"method": method, "magnus_frame": "whole_rho"}}
    if method == "cfm4":
        raw["lorrax"]["dt"] = 5.0
    with pytest.raises(ValueError) as e:
        parse_td_deck(raw, source="frame")
    assert str(e.value).startswith("GATE td_magnus_frame"), str(e.value)
