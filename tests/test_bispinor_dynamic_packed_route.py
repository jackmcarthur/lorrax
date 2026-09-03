"""The DYNAMIC packed four-current route (phase 3, minimal form).

``W_packed(omega) = diag(W_00(omega), W_TT, W_CT)``: the charge block carries
the run's plasmon-pole model, the twelve current blocks are the ``omega = 0``
packed response.  These cells pin the grammar that selects the route and the
block selection that keeps ONE Sigma consumer serving both packed routes.
They are parse-time and argument-validation cells only; the numbers are the
end-to-end gates in ``reports/bisp_n_dynamic_packed_2026-09-01/report.md``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

_PACKED_BARE = """\
bispinor = true
bispinor_gw = bare_transverse
sys_dim = 2
qp_solver = one_shot_dft
low_mem_bands = true
w_dyson_solver = distributed
restart = false
head_correction = full
"""

#: The explicit non-distributed negative control.  An unnamed solver is
#: derived to ``distributed`` from ``packed_static_envelope``.
_PACKED_BARE_LOCAL_SOLVER = _PACKED_BARE.replace(
    "w_dyson_solver = distributed\n", "w_dyson_solver = local\n")

_PPM_KEYS = """\
use_ppm_sigma = true
ppm_omega_p = 2.0
use_band_extrapolation = false
"""


def _config(tmp_path, extra="", name="dynamic_packed.in"):
    from gw.gw_config import LorraxConfig
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Which route a deck takes
# ---------------------------------------------------------------------------

def test_gn_ppm_bispinor_takes_the_packed_route(tmp_path):
    from gw.gw_config import (packed_bare_transverse_route,
                              packed_photon_replaces_charge_sigma,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    cfg = _config(
        tmp_path, _PACKED_BARE + _PPM_KEYS + "compute_mode = gn_ppm\n",
        name="gnppm_packed.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert taken, reason
    assert "DYNAMIC" in reason, reason
    assert uses_static_photon_response(cfg)
    assert uses_dynamic_packed_photon_route(cfg)
    # THE LOAD-BEARING HALF: the scalar charge owner survives, so the driver
    # still builds the head samples, the {static, probe} role W's and the
    # scalar band-diagonal q->0 bare-X head that the CC channel needs.
    assert not packed_photon_replaces_charge_sigma(cfg)


def test_cohsex_bispinor_still_replaces_the_charge_sigma(tmp_path):
    from gw.gw_config import (packed_photon_replaces_charge_sigma,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    cfg = _config(
        tmp_path, _PACKED_BARE + "compute_mode = cohsex\n",
        name="cohsex_packed.in")
    assert uses_static_photon_response(cfg)
    assert packed_photon_replaces_charge_sigma(cfg)
    assert not uses_dynamic_packed_photon_route(cfg)


def test_mpa_bispinor_does_not_take_the_packed_route(tmp_path):
    """MPA has no independent static-role W; it must not be served silently."""
    from gw.gw_config import (PACKED_PHOTON_COMPUTE_MODES, ComputeMode,
                              packed_bare_transverse_route,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    assert ComputeMode.MPA not in PACKED_PHOTON_COMPUTE_MODES
    cfg = _config(
        tmp_path, _PACKED_BARE + "compute_mode = mpa\n", name="mpa_packed.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert not taken
    assert "compute_mode = mpa" in reason
    assert not uses_static_photon_response(cfg)
    assert not uses_dynamic_packed_photon_route(cfg)


def test_scalar_decks_are_untouched_by_the_new_grammar(tmp_path):
    from gw.gw_config import (packed_photon_replaces_charge_sigma,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    cfg = _config(
        tmp_path, _PPM_KEYS + "compute_mode = gn_ppm\nsys_dim = 2\n",
        name="scalar_gnppm.in")
    assert cfg.bispinor is False
    assert not uses_static_photon_response(cfg)
    assert not uses_dynamic_packed_photon_route(cfg)
    assert not packed_photon_replaces_charge_sigma(cfg)


def test_the_hand_tt_overlay_is_refused_on_the_dynamic_packed_route(tmp_path):
    """The completion carries the TT head here too, so the overlay double counts."""
    with pytest.raises(ValueError) as exc:
        _config(
            tmp_path,
            _PACKED_BARE + _PPM_KEYS + "compute_mode = gn_ppm\n"
            "bispinor_tt_head_correction = true\n",
            name="gnppm_overlay.in")
    message = str(exc.value)
    # Since lane L the key is not a deck key at all: the parse-time tombstone
    # fires before the route's own double-count gate could, on any value.
    assert "bispinor_tt_head_correction" in message
    assert "REMOVED" in message


def test_the_route_is_not_taken_when_its_dyson_solver_would_refuse(tmp_path):
    """A route predicate must not claim a deck its screening owner refuses.

    ``w_isdf.compute_static_photon_response``'s first statement refuses
    anything but ``dyson_solver = 'distributed'`` (the packed solve has no
    local plan, and ``distrib_la`` additionally needs a true 2-D mesh).  The
    An unnamed solver is derived to ``distributed`` from the envelope table.
    This explicit ``local`` arm proves a user request is preserved and keeps
    the deck on the incumbent route rather than being silently overwritten.
    """
    from gw.gw_config import (packed_bare_transverse_route,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    cfg = _config(
        tmp_path, _PACKED_BARE_LOCAL_SOLVER + _PPM_KEYS + "compute_mode = gn_ppm\n",
        name="gnppm_auto_solver.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert not taken
    assert "w_dyson_solver" in reason, reason
    assert not uses_static_photon_response(cfg)
    assert not uses_dynamic_packed_photon_route(cfg)


def test_unnamed_solver_is_derived_for_the_dynamic_packed_route(tmp_path):
    from gw.gw_config import LorraxConfig, uses_dynamic_packed_photon_route

    path = tmp_path / "gnppm_derived_solver.in"
    path.write_text(
        _BASE
        + _PACKED_BARE.replace("w_dyson_solver = distributed\n", "")
        + _PPM_KEYS
        + "compute_mode = gn_ppm\n")
    lines = []
    cfg = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.backend.w_dyson_solver == "distributed"
    assert uses_dynamic_packed_photon_route(cfg)
    assert any("packed_static_envelope" in line
               and "w_dyson_solver was not named" in line for line in lines)


def test_dynamic_hall_gate_admits_trs_zero_and_announces_broken_tr_omission(
        tmp_path):
    """The dynamic route is never silent about the Faraday Gamma head."""
    from gw.w_isdf import _gate_dynamic_hall_head

    cfg = _config(
        tmp_path, _PACKED_BARE + _PPM_KEYS + "compute_mode = gn_ppm\n",
        name="gnppm_faraday_gate.in")

    class _Hall:
        sigma_H = np.asarray((4.0e-8, -5.0e-8, 6.0e-8),
                             dtype=np.float64)

        def sigma_H_at(self, frequency):
            assert frequency == 2.0j
            return np.asarray((1.0e-6, -2.0e-6, 3.0e-6),
                              dtype=np.complex128)

    records = []
    record = _gate_dynamic_hall_head(
        cfg, trs_allowed=True, coupled_head=True,
        hall_transaction=object(),
        print_fn=lambda text, **kwargs: records.append(text))
    assert record.startswith("EXACT ZERO")
    assert records == [
        "Faraday head : EXACT ZERO (SymMaps.trs_allowed=true; "
        "no Hall producer or consumer path taken)"]

    records.clear()
    record = _gate_dynamic_hall_head(
        cfg, trs_allowed=False, coupled_head=True,
        hall_transaction=_Hall(),
        print_fn=lambda text, **kwargs: records.append(text))
    assert record == (
        "ABSENT (unowned rank-four probe-frequency CT/TC Sigma insertion; "
        "max|sigma_H(i*omega_p)| = 3.00000000000000008e-06 bohr^-1; "
        "static sigma_H = [4e-08, -5e-08, 6e-08] bohr^-1)")
    assert records == ["WARNING Faraday head : " + record]

    records.clear()
    record = _gate_dynamic_hall_head(
        cfg, trs_allowed=False, coupled_head=True, hall_transaction=None,
        print_fn=lambda text, **kwargs: records.append(text))
    assert "unmeasured (static_gauge_hall_file unnamed)" in record
    assert records == ["WARNING Faraday head : " + record]


# ---------------------------------------------------------------------------
# One consumer, two block selections
# ---------------------------------------------------------------------------

def test_photon_sigma_block_selection_is_validated():
    """A misspelled selection must refuse, not silently sum all sixteen."""
    from gw.photon_sigma import (PHOTON_BLOCKS_ALL, PHOTON_BLOCKS_CURRENT,
                                 compute_static_photon_sigma)
    assert PHOTON_BLOCKS_ALL == "all"
    assert PHOTON_BLOCKS_CURRENT == "current"
    with pytest.raises(ValueError) as exc:
        compute_static_photon_sigma(
            wfns_charge=None, wfns_transverse=None, Gij=None,
            V_packed=None, W_packed=None, photon_layout=None, meta=None,
            mesh_xy=None, blocks="transverse")
    assert "block selection" in str(exc.value)


def test_the_dispatch_asks_for_the_current_blocks_only():
    """The dynamic branch must not re-sum the CC block the scalar Sigma owns."""
    import inspect

    from gw import sigma_dispatch
    source = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    assert "blocks=PHOTON_BLOCKS_CURRENT" in source
    # and the scalar bare-X call in that branch must not fold Sigma^B twice
    assert "bispinor_v_q_path=None," in source

    driver = (pathlib.Path(__file__).resolve().parents[1]
              / "src" / "gw" / "gw_jax.py").read_text()
    assert '"Faraday head   : "' in driver
    assert "photon_response.faraday_head_record" in driver


# ---------------------------------------------------------------------------
# In-band import canary (KNOWN_SANDBOX_ERRORS.md: `lx` can bind another tree)
# ---------------------------------------------------------------------------

def test_this_file_grades_the_checkout_it_lives_in():
    import importlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for name in ("gw.gw_config", "gw.sigma_dispatch", "gw.photon_sigma",
                 "gw.cohsex_sigma"):
        module = importlib.import_module(name)
        origin = pathlib.Path(module.__file__).resolve()
        if repo not in origin.parents:
            offenders.append(f"{name} -> {origin}")
    assert not offenders, (
        f"these modules came from outside {repo}: " + "; ".join(offenders))
