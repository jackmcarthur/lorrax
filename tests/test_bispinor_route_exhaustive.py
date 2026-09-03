"""Every supported bispinor deck class has the packed photon owner.

This is the executable capability ruling for retiring the former
``sigma_x_bispinor`` route.  The parse-time grid and the later
screened-restart door are tested separately from the measured-TRS Hall door:
TRS is a property of the loaded WFN, not a deck key.
"""

from __future__ import annotations

import inspect
import itertools
import re

import numpy as np
import pytest


_FAMILIES = ("bare_transverse", "full_static_cohsex")
_COMPUTE_MODES = ("cohsex", "gn_ppm", "hl_ppm", "mpa", "x_only")
_DIMENSIONS = (2, 3)
_RESTARTS = (False, True)
_QP_SOLVERS = ("one_shot_dft", "self_consistent")
_DYSON_SOLVERS = ("distributed", "local")
_HEAD_POLICIES = ("full", "off", "no_local_fields")
_SCREENING_DIAGRAMS = ("w_rpa", "w_bse", "w_rpa_resolvent")

# Exactly the deck-class matrix in docs/input_reference.md.  Runtime facts
# that are not deck axes (a multi-process local solve and measured broken TR)
# have their own focused gates below.
_EXPECTED_DECK_GATE_IDS = {
    "bispinor_head_correction_no_local_fields_unavailable",
    "bispinor_screened_packed_restart_storage_unimplemented",
    "bispinor_screened_x_only_has_no_screened_operand",
    "bispinor_screening_diagrams_require_packed_operand",
    "packed_bare_cohsex_self_consistency_unimplemented",
    "packed_bare_x_only_self_consistency_unimplemented",
    "packed_screened_mpa_static_role_unimplemented",
    "packed_screened_self_consistency_unimplemented",
}


def _gate_id(error: BaseException) -> str:
    match = re.search(r"\bGATE ([a-z0-9_]+):", str(error))
    assert match is not None, f"unnamed bispinor refusal: {error}"
    return match.group(1)


def _deck(*, family, compute_mode, sys_dim, restart, qp_solver,
          w_dyson_solver, head_correction, screening_diagrams):
    return "".join((
        "[cohsex]\n",
        "nval = 2\n",
        "ncond = 2\n",
        "nband = 10\n",
        "memory_per_device_gb = 4.0\n",
        "strict_keys = true\n",
        "bispinor = true\n",
        f"bispinor_gw = {family}\n",
        f"compute_mode = {compute_mode}\n",
        f"sys_dim = {sys_dim}\n",
        f"restart = {str(restart).lower()}\n",
        f"qp_solver = {qp_solver}\n",
        f"w_dyson_solver = {w_dyson_solver}\n",
        f"head_correction = {head_correction}\n",
        f"screening_diagrams = {screening_diagrams}\n",
    ))


def test_every_deck_class_is_packed_or_refuses_by_the_matrix(tmp_path):
    """Cross the complete public class grid through the driver's doors."""
    from gw.gw_config import (
        BispinorGWMode,
        LorraxConfig,
        packed_bare_transverse_route,
        packed_photon_screens_current,
        refuse_screened_photon_restart_storage,
        refuse_unsupported_bispinor_gw,
        refuse_unsupported_screening_diagrams,
        uses_static_photon_response,
    )

    refused = set()
    served = 0
    path = tmp_path / "bispinor_route_grid.in"
    axes = itertools.product(
        _FAMILIES,
        _COMPUTE_MODES,
        _DIMENSIONS,
        _RESTARTS,
        _QP_SOLVERS,
        _DYSON_SOLVERS,
        _HEAD_POLICIES,
        _SCREENING_DIAGRAMS,
    )
    for values in axes:
        kwargs = dict(zip((
            "family", "compute_mode", "sys_dim", "restart", "qp_solver",
            "w_dyson_solver", "head_correction", "screening_diagrams",
        ), values))
        path.write_text(_deck(**kwargs))
        try:
            config = LorraxConfig.from_input_file(
                str(path), print_fn=lambda *args, **kw: None)
            # These are the same validation doors reached by config parsing
            # and driver preparation.  Calling them explicitly makes a
            # future change in parse ordering unable to weaken this ruling.
            refuse_unsupported_screening_diagrams(config)
            refuse_unsupported_bispinor_gw(config)
            refuse_screened_photon_restart_storage(
                config, nq=2, n_charge_padded=3, n_current_padded=5)
        except (NotImplementedError, ValueError) as error:
            refused.add(_gate_id(error))
            continue

        served += 1
        assert uses_static_photon_response(config), kwargs
        if config.bispinor_gw is BispinorGWMode.BARE_TRANSVERSE:
            taken, reason = packed_bare_transverse_route(config)
            assert taken, (kwargs, reason)
            assert not packed_photon_screens_current(config)
        else:
            assert packed_photon_screens_current(config)

    assert served > 0
    assert refused == _EXPECTED_DECK_GATE_IDS


def test_measured_broken_tr_hl_refuses_before_a_photon_body_is_opened(
        tmp_path):
    """HL has a real-axis probe, not the authenticated imaginary Hall one."""
    from gw.gw_config import LorraxConfig
    from gw.w_isdf import _gate_dynamic_hall_head

    path = tmp_path / "magnetic_hl.in"
    path.write_text(_deck(
        family="bare_transverse", compute_mode="hl_ppm", sys_dim=2,
        restart=False, qp_solver="one_shot_dft",
        w_dyson_solver="distributed", head_correction="full",
        screening_diagrams="w_rpa"))
    config = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *args, **kw: None)

    class Hall:
        sigma_H = np.asarray((0.0, 0.0, 2.0e-8), dtype=np.float64)

        def sigma_H_at(self, frequency):
            return np.asarray((0.0, 0.0, 1.0e-8), dtype=np.complex128)

    with pytest.raises(
            NotImplementedError,
            match="GATE dynamic_hall_head_hl_imaginary_probe"):
        _gate_dynamic_hall_head(
            config, trs_allowed=False, coupled_head=True,
            hall_transaction=Hall(), print_fn=lambda *args, **kw: None)


def test_the_sigma_dispatch_has_no_nonpacked_bispinor_fallback():
    """Red twin: route predicates are insufficient while a fallback exists."""
    from gw import cohsex_sigma, sigma_dispatch

    dispatch_signature = inspect.signature(sigma_dispatch.compute_sigma_xc)
    dispatch_source = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    cohsex_signature = inspect.signature(cohsex_sigma.compute_cohsex_sigma)
    exchange_signature = inspect.signature(cohsex_sigma.compute_sigma_x)

    assert "bispinor_v_q_path" not in dispatch_signature.parameters
    assert "bispinor_v_q_path" not in dispatch_source
    assert "return_transverse" not in dispatch_source
    for signature in (cohsex_signature, exchange_signature):
        assert "wfns_transverse" not in signature.parameters
        assert "bispinor_v_q_path" not in signature.parameters
