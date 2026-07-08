"""Unit tests for the ``qp_solver`` config axis and the ``SCConfig`` group.

Covers the auto-resolution contract of ``LorraxConfig.qp_solver``
(G0W0_SC_TOGGLE_DESIGN.md §2a):

1. Default → ``ONE_SHOT_DFT`` (standard G0W0).
2. Legacy ``self_consistent = true`` → ``SELF_CONSISTENT`` (deprecated,
   honored).
3. Legacy ``sigma_at_dft_energies = true`` → ``ONE_SHOT_DFT`` (the orphan
   flag's intended meaning IS the default; deprecated, honored).
4. Explicit ``qp_solver`` overrides the legacy flags.
5. Validation: ``fixed_point`` × static mode → error;
   ``fixed_point`` / ``self_consistent`` × dynamic × ``kij_stream`` → error
   (previously that pair silently degraded the eigh outputs to static Σ).
6. ``SCConfig``: sc_* input keys parsed; ``LORRAX_SC_*`` envs honored as
   deprecated overrides; knob validation.

All tests run on a throwaway input file — no WFN, no GPU, no jit.
"""
from __future__ import annotations

import pytest

from gw.gw_config import LorraxConfig, QPSolver


BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra: str = "", name: str = "cohsex_qp.in"):
    path = tmp_path / name
    path.write_text(BASE_INPUT + extra)
    return LorraxConfig.from_input_file(str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# auto-resolution
# ---------------------------------------------------------------------------

def test_default_is_one_shot_dft(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_legacy_self_consistent_resolves_and_warns(tmp_path):
    with pytest.warns(DeprecationWarning, match="self_consistent"):
        cfg = _config(tmp_path, "self_consistent = true\n")
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT


def test_legacy_sigma_at_dft_energies_resolves_and_warns(tmp_path):
    with pytest.warns(DeprecationWarning, match="sigma_at_dft_energies"):
        cfg = _config(tmp_path, "sigma_at_dft_energies = true\n")
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_explicit_overrides_legacy(tmp_path):
    with pytest.warns(DeprecationWarning):
        cfg = _config(
            tmp_path, "self_consistent = true\nqp_solver = one_shot_dft\n")
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_explicit_fixed_point_dynamic(tmp_path):
    cfg = _config(
        tmp_path, "compute_mode = gn_ppm\nqp_solver = fixed_point\n")
    assert cfg.qp_solver is QPSolver.FIXED_POINT


def test_unknown_value_raises(tmp_path):
    cfg = _config(tmp_path, "qp_solver = bogus\n")
    with pytest.raises(ValueError, match="qp_solver"):
        cfg.qp_solver


# ---------------------------------------------------------------------------
# validation of inconsistent axis combinations
# ---------------------------------------------------------------------------

def test_fixed_point_static_mode_rejected(tmp_path):
    cfg = _config(
        tmp_path, "compute_mode = cohsex\nqp_solver = fixed_point\n")
    with pytest.raises(ValueError, match="fixed_point"):
        cfg.qp_solver


@pytest.mark.parametrize("solver", ["fixed_point", "self_consistent"])
def test_kij_stream_dynamic_rejected(tmp_path, solver):
    cfg = _config(
        tmp_path,
        f"compute_mode = gn_ppm\nqp_solver = {solver}\n"
        "sigma_omega_accumulation = kij_stream\n")
    with pytest.raises(ValueError, match="kij_stream"):
        cfg.qp_solver


def test_kij_stream_one_shot_dft_ok(tmp_path):
    # one_shot_dft × streamed is fine: the at-DFT diag comes from the h5.
    cfg = _config(
        tmp_path,
        "compute_mode = gn_ppm\nqp_solver = one_shot_dft\n"
        "sigma_omega_accumulation = kij_stream\n")
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_kij_stream_static_self_consistent_ok(tmp_path):
    # The accumulation knob is never read in static modes; SC-COHSEX
    # must not be rejected on account of it.
    cfg = _config(
        tmp_path,
        "compute_mode = cohsex\nqp_solver = self_consistent\n"
        "sigma_omega_accumulation = kij_stream\n")
    assert cfg.qp_solver is QPSolver.SELF_CONSISTENT


# ---------------------------------------------------------------------------
# SCConfig
# ---------------------------------------------------------------------------

def test_sc_defaults(tmp_path):
    sc = _config(tmp_path).sc
    assert (sc.max_iter, sc.tol_ev, sc.accelerator, sc.history_depth,
            sc.mixing, sc.dump_dir) == (20, 1.0e-4, "rcrop", 5, 1.0, None)


def test_sc_input_keys(tmp_path):
    sc = _config(
        tmp_path,
        "sc_max_iter = 7\nsc_tol_ev = 1e-6\nsc_accelerator = linear\n"
        "sc_history_depth = 3\nsc_mixing = 0.5\nsc_dump_dir = sc_hist\n").sc
    assert (sc.max_iter, sc.tol_ev, sc.accelerator, sc.history_depth,
            sc.mixing, sc.dump_dir) == (7, 1.0e-6, "linear", 3, 0.5, "sc_hist")


def test_sc_env_overrides_deprecated(tmp_path, monkeypatch):
    monkeypatch.setenv("LORRAX_SC_MAX_ITER", "3")
    monkeypatch.setenv("LORRAX_SC_TOL_EV", "1e-10")
    monkeypatch.setenv("LORRAX_SC_ACCEL", "linear")
    monkeypatch.setenv("LORRAX_SC_DEPTH", "2")
    monkeypatch.setenv("LORRAX_SC_MIXING", "0.25")
    monkeypatch.setenv("LORRAX_SC_DUMP_DIR", "/tmp/sc_dump")
    lines: list[str] = []
    path = tmp_path / "cohsex_env.in"
    path.write_text(BASE_INPUT + "sc_max_iter = 99\n")
    sc = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a)))
    ).sc
    assert (sc.max_iter, sc.tol_ev, sc.accelerator, sc.history_depth,
            sc.mixing, sc.dump_dir) == (3, 1.0e-10, "linear", 2, 0.25,
                                        "/tmp/sc_dump")
    assert any("deprecated env override" in l for l in lines)


def test_sc_bad_accelerator_rejected(tmp_path):
    with pytest.raises(ValueError, match="sc_accelerator"):
        _config(tmp_path, "sc_accelerator = bogus\n")
