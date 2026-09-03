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

#: The explicit local plan.  The bare family performs no packed solve, so
#: this value controls only its scalar charge W and is legal on a 1x1 mesh.
_PACKED_BARE_LOCAL_SOLVER = _PACKED_BARE.replace(
    "w_dyson_solver = distributed\n", "w_dyson_solver = local\n")

_PACKED_SCREENED = _PACKED_BARE.replace(
    "bispinor_gw = bare_transverse\n",
    "bispinor_gw = full_static_cohsex\n")

_PACKED_BARE_SC = _PACKED_BARE.replace(
    "qp_solver = one_shot_dft\n", "qp_solver = self_consistent\n")

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


def test_mpa_bare_bispinor_takes_absent_charge_packed_route(tmp_path):
    """MPA needs no static W because the current consumer skips packed CC."""
    from gw.gw_config import (PACKED_PHOTON_COMPUTE_MODES, ComputeMode,
                              packed_bare_transverse_route,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    assert ComputeMode.MPA in PACKED_PHOTON_COMPUTE_MODES
    cfg = _config(
        tmp_path, _PACKED_BARE + "compute_mode = mpa\n", name="mpa_packed.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert taken, reason
    assert "DYNAMIC" in reason
    assert uses_static_photon_response(cfg)
    assert uses_dynamic_packed_photon_route(cfg)


@pytest.mark.parametrize("mode", ["gn_ppm", "hl_ppm", "mpa"])
def test_self_consistent_bare_dynamic_modes_take_packed_route(tmp_path, mode):
    from gw.gw_config import (packed_bare_transverse_route,
                              uses_dynamic_packed_photon_route)

    extra = _PACKED_BARE_SC + f"compute_mode = {mode}\n"
    if mode != "mpa":
        extra += _PPM_KEYS
    cfg = _config(tmp_path, extra, name=f"sc_{mode}.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert taken, reason
    assert "self-consistent" in reason
    assert uses_dynamic_packed_photon_route(cfg)


def test_screened_mpa_refuses_missing_static_role_by_name(tmp_path):
    with pytest.raises(ValueError) as exc:
        _config(
            tmp_path, _PACKED_SCREENED + "compute_mode = mpa\n",
            name="screened_mpa.in")
    assert "GATE packed_screened_mpa_static_role_unimplemented" in str(exc.value)


def test_screened_self_consistency_refuses_per_map_cost_by_name(tmp_path):
    with pytest.raises(ValueError) as exc:
        _config(
            tmp_path,
            _PACKED_SCREENED.replace(
                "qp_solver = one_shot_dft\n", "qp_solver = self_consistent\n")
            + _PPM_KEYS + "compute_mode = gn_ppm\n",
            name="screened_sc.in")
    assert "GATE packed_screened_self_consistency_unimplemented" in str(exc.value)


def test_bare_cohsex_self_consistency_refuses_charge_rebuild_by_name(tmp_path):
    with pytest.raises(ValueError) as exc:
        _config(
            tmp_path, _PACKED_BARE_SC + "compute_mode = cohsex\n",
            name="bare_cohsex_sc.in")
    assert "GATE packed_bare_cohsex_self_consistency_unimplemented" in str(exc.value)


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


def test_bare_route_accepts_local_solver_for_one_gpu(tmp_path):
    """Bare current builds no packed Dyson system; local serves scalar CC."""
    from gw.gw_config import (packed_bare_transverse_route,
                              uses_dynamic_packed_photon_route,
                              uses_static_photon_response)
    cfg = _config(
        tmp_path, _PACKED_BARE_LOCAL_SOLVER + _PPM_KEYS + "compute_mode = gn_ppm\n",
        name="gnppm_auto_solver.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert taken, reason
    assert cfg.backend.w_dyson_solver == "local"
    assert uses_static_photon_response(cfg)
    assert uses_dynamic_packed_photon_route(cfg)


def test_unnamed_solver_resolves_local_for_the_dynamic_packed_route(tmp_path):
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
    assert cfg.backend.w_dyson_solver == "local"
    assert uses_dynamic_packed_photon_route(cfg)
    assert any("packed_static_envelope" in line
               and "w_dyson_solver was not named" in line for line in lines)
    assert any("bare packed operator skips its packed solve" in line
               for line in lines)


def test_x_only_bare_family_takes_packed_route(tmp_path):
    from gw.gw_config import (ComputeMode, PACKED_PHOTON_COMPUTE_MODES,
                              packed_bare_transverse_route,
                              uses_static_photon_response)

    cfg = _config(
        tmp_path, _PACKED_BARE_LOCAL_SOLVER + "compute_mode = x_only\n",
        name="x_only_packed.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert ComputeMode.X_ONLY in PACKED_PHOTON_COMPUTE_MODES
    assert taken, reason
    assert "scalar charge X plus packed current X" in reason
    assert uses_static_photon_response(cfg)


def test_x_only_screened_family_refuses_by_exact_gate(tmp_path):
    deck = (_PACKED_BARE_LOCAL_SOLVER
            .replace("bispinor_gw = bare_transverse",
                     "bispinor_gw = full_static_cohsex")
            + "compute_mode = x_only\n")
    with pytest.raises(
            ValueError,
            match="GATE bispinor_screened_x_only_has_no_screened_operand"):
        _config(tmp_path, deck, name="x_only_screened.in")


@pytest.mark.parametrize(
    "mode,extra",
    [("cohsex", ""), ("gn_ppm", _PPM_KEYS), ("x_only", "")],
)
def test_restart_does_not_move_served_bare_classes_off_packed_route(
        tmp_path, mode, extra):
    from gw.gw_config import packed_bare_transverse_route

    deck = _PACKED_BARE_LOCAL_SOLVER.replace(
        "restart = false", "restart = true")
    cfg = _config(
        tmp_path, deck + extra + f"compute_mode = {mode}\n",
        name=f"restart_{mode}.in")
    taken, reason = packed_bare_transverse_route(cfg)
    assert taken, reason


def test_x_only_dispatch_uses_packed_current_blocks_not_sigma_b_fallback():
    import inspect

    from gw import sigma_dispatch
    source = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    marker = "and mode is ComputeMode.X_ONLY"
    assert marker in source
    branch = source[source.index(marker):source.index(
        "elif uses_dynamic_packed_photon_route")]
    assert "blocks=PHOTON_BLOCKS_CURRENT" in branch
    assert "bispinor_v_q_path=None" in branch


@pytest.mark.parametrize("diagrams", ["w_bse", "w_rpa_resolvent"])
def test_bispinor_ladder_classes_refuse_at_dispatch_door(
        tmp_path, diagrams):
    with pytest.raises(ValueError) as exc:
        _config(
            tmp_path,
            _PACKED_BARE_LOCAL_SOLVER
            + "compute_mode = cohsex\n"
            + f"screening_diagrams = {diagrams}\n",
            name=f"{diagrams}.in")
    assert "GATE bispinor_screening_diagrams_require_packed_operand" in str(
        exc.value)


def test_screened_restart_refusal_prints_exact_storage_size(tmp_path):
    from gw.gw_config import refuse_screened_photon_restart_storage

    screened = _PACKED_BARE_LOCAL_SOLVER.replace(
        "bispinor_gw = bare_transverse",
        "bispinor_gw = full_static_cohsex").replace(
            "restart = false", "restart = true")
    cfg = _config(
        tmp_path, screened + "compute_mode = cohsex\n",
        name="screened_restart.in")
    with pytest.raises(ValueError) as exc:
        refuse_screened_photon_restart_storage(
            cfg, nq=9, n_charge_padded=16, n_current_padded=20)
    message = str(exc.value)
    assert "GATE bispinor_screened_packed_restart_storage_unimplemented" in message
    assert str(2 * 9 * (16 + 3 * 20) ** 2 * 16) in message


def test_screened_local_solver_is_a_served_packed_plan(tmp_path):
    from gw.gw_config import uses_static_photon_response

    screened = _PACKED_BARE_LOCAL_SOLVER.replace(
        "bispinor_gw = bare_transverse",
        "bispinor_gw = full_static_cohsex")
    cfg = _config(
        tmp_path, screened + "compute_mode = cohsex\n",
        name="screened_local.in")
    assert cfg.backend.w_dyson_solver == "local"
    assert uses_static_photon_response(cfg)


def test_screened_local_solver_factors_the_full_packed_direct_sum():
    """The scalar ``meta.n_rmu`` must not truncate C+T1+T2+T3."""
    import inspect

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from gw.w_isdf import compute_static_photon_response, solve_w

    source = inspect.getsource(compute_static_photon_response)
    screened = source[source.index("if screen_current:"):
                      source.index("else:\n        # chi_TT")]
    assert "n_rmu_logical=sum(layout.logical_extents)" in screened

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    class ScalarMeta:
        nk_tot = 1
        nspin = 1
        nspinor_wfnfile = 1
        n_rmu = 1

    V = np.diag([0.10, 0.20, 0.30, 0.40]).astype(np.complex128)[None]
    chi = np.diag([-0.25, -0.20, -0.15, -0.10]).astype(np.complex128)[None]
    got = np.asarray(solve_w(
        jnp.asarray(V), jnp.asarray(chi), ScalarMeta(), mesh,
        n_rmu_logical=4, dyson_solver="local"))
    pref = 2.0
    want = np.linalg.solve(
        np.eye(4, dtype=np.complex128) - pref * V[0] @ chi[0], V[0])
    np.testing.assert_allclose(got[0], want, rtol=2.0e-13, atol=2.0e-13)
    assert np.all(np.diag(got[0])[1:] != 0.0)


def test_screened_local_solver_refuses_on_a_multi_process_mesh(
        tmp_path, monkeypatch):
    from gw.w_isdf import compute_static_photon_response

    screened = (_PACKED_BARE_LOCAL_SOLVER
                .replace("bispinor_gw = bare_transverse",
                         "bispinor_gw = full_static_cohsex")
                .replace("head_correction = full", "head_correction = off"))
    cfg = _config(
        tmp_path, screened + "compute_mode = cohsex\n",
        name="screened_local_p4.in")
    monkeypatch.setattr("gw.w_isdf.jax.process_count", lambda: 4)

    class Mesh:
        class Devices:
            shape = (2, 2)
        devices = Devices()

    with pytest.raises(
            ValueError,
            match="GATE bispinor_screened_packed_local_requires_1x1"):
        compute_static_photon_response(
            None, None, None, None, None, Mesh(),
            screen_current=True, W_charge=None, config=cfg,
            dyson_solver="local")


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
    assert "charge_block_state=photon_response.charge_block_state" in source


def test_mpa_finalizer_keeps_the_packed_head_diagnostics():
    """MPA and PPM publish the same current-sector Gamma receipt."""
    import inspect

    from gw import sigma_dispatch
    source = inspect.getsource(sigma_dispatch.compute_sigma_xc)
    mpa_branch = source.split(
        "if mode is ComputeMode.MPA:", 1)[1].split(
            "# ── THE EXHAUSTIVENESS SEAM", 1)[0]
    assert "photon_head_sigma_diag_tskn_ry=photon_head_sigma_diag" in mpa_branch
    assert "photon_head_sigma_basis=photon_head_sigma_basis" in mpa_branch


def test_absent_charge_state_has_no_packed_w_and_aliases_current_to_v():
    """The layout is named absence, never a zero CC tile posing as W."""
    import inspect

    from gw import photon_sigma, w_isdf
    from gw.head_correction import STATIC_PHOTON_CHARGE_BLOCK_ABSENT

    assert STATIC_PHOTON_CHARGE_BLOCK_ABSENT == "absent_dynamic_scalar_owner"
    build_source = inspect.getsource(w_isdf.compute_static_photon_response)
    consume_source = inspect.getsource(photon_sigma.compute_static_photon_sigma)
    assert "W_packed = None" in build_source
    assert "V_AB if charge_absent" in consume_source
    assert "charge_absent and blocks != PHOTON_BLOCKS_CURRENT" in consume_source


def test_sc_map_rotates_and_recontracts_current_bundle():
    import inspect

    from gw import sc_iteration
    source = inspect.getsource(sc_iteration.gw_iteration_map)
    assert "wfns_transverse_qp = rotate_wavefunctions" in source
    assert "wfns_transverse=wfns_transverse_qp" in source
    assert "photon_response=inputs.photon_response" in source
    assert "SC packed current map" in source


def test_dynamic_run_record_names_the_absent_charge_layout():
    """Production logs must not describe a nonexistent static CC Dyson."""
    text = (pathlib.Path(__file__).resolve().parents[1]
            / "src" / "gw" / "gw_jax.py").read_text(encoding="utf-8")
    assert "packed dynamic current-only photon operator" in text
    assert "CC block absent; W_CURRENT = V_CURRENT" in text
    assert "_bare_taken and uses_dynamic_packed_photon_route(config)" in text


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
