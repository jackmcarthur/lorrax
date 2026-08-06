"""Unit tests for the ``qp_solver`` config axis and the ``SCConfig`` group.

Covers the auto-resolution contract of ``LorraxConfig.qp_solver``
(G0W0_SC_TOGGLE_DESIGN.md §2a):

1. Default → ``ONE_SHOT_DFT`` (standard G0W0).
2. Legacy ``self_consistent = true`` → ``SELF_CONSISTENT`` (deprecated,
   honored).
3. Legacy ``sigma_at_dft_energies = true`` → ``ONE_SHOT_DFT`` (the orphan
   flag's intended meaning IS the default; deprecated, honored).
4. Explicit ``qp_solver`` overrides the legacy flags.
5. Validation: ``fixed_point`` × static mode → error; the removed
   ``sigma_omega_accumulation = kij_stream`` spelling → parse-time error.
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


def test_kij_stream_refused_at_parse(tmp_path):
    # kij_stream was REMOVED (2026-07-31).  A removed VALUE of a known
    # key must refuse at parse with the removal named — never silently
    # reroute an old deck (announce-or-refuse).
    with pytest.raises(ValueError, match="REMOVED"):
        _config(
            tmp_path,
            "compute_mode = gn_ppm\n"
            "sigma_omega_accumulation = kij_stream\n")


def test_kij_stream_refused_even_in_static_modes(tmp_path):
    # The removed value refuses regardless of compute_mode — a removed
    # spelling must not survive in decks just because the knob would
    # not have been read.
    with pytest.raises(ValueError, match="REMOVED"):
        _config(
            tmp_path,
            "compute_mode = cohsex\nqp_solver = self_consistent\n"
            "sigma_omega_accumulation = kij_stream\n")


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


# ---------------------------------------------------------------------------
# Distributed-linalg axes: distributed_cholesky / distributed_lu
# (portable backend names; the legacy cusolvermp_charge/cusolvermp_lu
# aliases were REMOVED 2026-07-31 — unknown deck keys are ignored)
#
# The parse outcome is JAX-backend-dependent (explicit cusolvermp REFUSES
# on a CPU backend — doctrine 3, audit fix/zq 2026-07-28; 'auto' LU
# demotes there with an announcement), so each cell PINS the backend via
# monkeypatch instead of inheriting whatever the test box runs, and the
# slab_io routers are stubbed so no capability probe / MPI init runs at
# parse time.
# ---------------------------------------------------------------------------

def _pin_backend(monkeypatch, backend_name: str):
    import jax
    import gw.gw_config as gw_config
    from gw.gw_config import SlabIOBackend
    monkeypatch.setattr(jax, "default_backend", lambda: backend_name)
    monkeypatch.setattr(gw_config, "_route_cpu_slab_io",
                        lambda print_fn: SlabIOBackend.H5PY_ALLGATHER)
    monkeypatch.setattr(gw_config, "_route_gpu_slab_io",
                        lambda print_fn: SlabIOBackend.H5PY_ALLGATHER)


def test_distributed_linalg_defaults(tmp_path, monkeypatch):
    _pin_backend(monkeypatch, "gpu")
    cfg = _config(tmp_path)
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "auto"


def test_distributed_cholesky_slate_accepted(tmp_path):
    cfg = _config(tmp_path, "distributed_cholesky = slate\n")
    assert cfg.backend.distributed_cholesky == "slate"


def test_distributed_lu_slate_rejected(tmp_path):
    # No SLATE getrf wrapper exists yet — the value must fail loudly.
    with pytest.raises(ValueError, match="distributed_lu"):
        _config(tmp_path, "distributed_lu = slate\n")


def test_removed_legacy_aliases_are_inert(tmp_path, monkeypatch):
    # cusolvermp_charge / cusolvermp_lu were REMOVED (2026-07-31): they
    # are unknown keys now and must neither DeprecationWarning nor steer
    # the portable keys (the pre-removal alias honored 'on' → cusolvermp
    # here).  They DO appear in the aggregated unknown-key stdout report
    # like any other unknown key — that is the point of the removal.
    _pin_backend(monkeypatch, "gpu")
    cfg = _config(tmp_path, "cusolvermp_charge = on\ncusolvermp_lu = off\n")
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "auto"


def test_distributed_cholesky_invalid_value_raises(tmp_path):
    with pytest.raises(ValueError, match="distributed_cholesky"):
        _config(tmp_path, "distributed_cholesky = scalapack\n")


@pytest.mark.parametrize("key", ["distributed_cholesky", "distributed_lu"])
def test_explicit_cusolvermp_on_cpu_refuses(tmp_path, monkeypatch, key):
    # Doctrine 3: an explicit CUDA-only backend on a CPU JAX backend
    # REFUSES at parse time — it is not rewritten to 'off' (which silently
    # ran a different solver than the input file names).  Substring
    # contract, not exact text.
    _pin_backend(monkeypatch, "cpu")
    with pytest.raises(ValueError, match="CUDA-only"):
        _config(tmp_path, f"{key} = cusolvermp\n")


# ---------------------------------------------------------------------------
# Unknown-key check (read_lorrax_input): a deck key that is neither in
# _DEFAULTS nor covered by a legacy branch is reported in ONE aggregated
# rank-0 stdout warning and ignored; ``strict_keys = true`` upgrades the
# report to a ValueError naming every unknown key.  These cases call
# ``read_lorrax_input`` directly — the parsing path needs no jax.
# ---------------------------------------------------------------------------

def test_unknown_key_warns_on_stdout_and_still_parses(tmp_path, capsys):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_unknown.in"
    p.write_text(BASE_INPUT + "x_only = true\nuse_chunked_isdf = false\n")
    params = read_lorrax_input(str(p))
    out = capsys.readouterr().out
    # One aggregated report naming each key with its line number.
    assert "unrecognized deck key" in out
    assert "x_only (line 6)" in out
    assert "use_chunked_isdf (line 7)" in out
    assert "ignored" in out
    # ...and the deck still parses: known keys land, unknown ones don't.
    assert params["nval"] == 2
    assert "x_only" not in params


def test_strict_keys_true_raises_naming_every_unknown_key(tmp_path):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_strict.in"
    p.write_text(BASE_INPUT + "strict_keys = true\n"
                 "x_only = true\nbogus_knob = 3\n")
    with pytest.raises(ValueError, match="strict_keys") as exc:
        read_lorrax_input(str(p))
    # ALL unknown keys named at once, not just the first.
    assert "x_only" in str(exc.value)
    assert "bogus_knob" in str(exc.value)


def test_mixed_case_key_is_recognised_not_reported_unknown(tmp_path, capsys):
    """``do_G0`` is honoured, so it must not also be called unrecognised.

    configparser folds option names to lower case; ``do_G0`` is the only
    key in _DEFAULTS that is not already lower case.  Before the fold was
    applied to the recognition test, this deck BOTH steered the run
    (``params["do_G0"] is False``) and printed ``do_g0 ... not a
    recognized deck key`` -- and ``strict_keys = true`` refused it.
    Checked for the failure signature (reported-unknown) rather than for
    a success marker.
    """
    from gw.gw_config import read_lorrax_input
    for spelling in ("do_G0", "do_g0"):
        p = tmp_path / f"deck_{spelling}.in"
        p.write_text(BASE_INPUT + f"{spelling} = false\n")
        params = read_lorrax_input(str(p))
        out = capsys.readouterr().out
        assert params["do_G0"] is False, (
            f"{spelling}: value not honoured -- test premise is wrong")
        assert "not a recognized deck key" not in out, (
            f"{spelling}: honoured key reported as unrecognised:\n{out}")


def test_mixed_case_key_survives_strict_keys(tmp_path):
    """The same key must not be REFUSED by the strict gate."""
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_strict_case.in"
    p.write_text(BASE_INPUT + "strict_keys = true\ndo_G0 = false\n")
    params = read_lorrax_input(str(p))     # must not raise
    assert params["do_G0"] is False


def test_screening_method_ctsp_refuses_and_names_minimax(tmp_path):
    """``ctsp`` must REFUSE, not resolve to minimax in silence.

    It parsed and ran minimax for months because nothing reads
    ``ScreeningConfig.method``.  The refusal has to name the supported
    method, per the explicit-request-fails-loudly convention.
    """
    with pytest.raises(ValueError, match="minimax") as exc:
        _config(tmp_path, "screening_method = ctsp\n")
    assert "ctsp" in str(exc.value)


def test_screening_method_minimax_and_default_still_parse(tmp_path):
    """The supported value, and omitting the key, both stay clean."""
    assert (_config(tmp_path, "screening_method = minimax\n")
            .screening.method == "minimax")
    assert _config(tmp_path, "").screening.method == "minimax"


def test_legacy_keys_keep_their_dedicated_messages(tmp_path, capsys):
    # Keys with an explicit legacy branch (dedicated DeprecationWarning or
    # refusal) must NOT also be reported as unknown — one key, one message.
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_legacy.in"
    p.write_text(BASE_INPUT + "chunk_size = 8\noutput_file = out.dat\n")
    with pytest.warns(DeprecationWarning) as rec:
        read_lorrax_input(str(p))
    messages = [str(w.message) for w in rec]
    assert any("chunk_size" in m for m in messages)
    assert any("output_file" in m for m in messages)
    assert "unrecognized deck key" not in capsys.readouterr().out
    # The refusal branch also stays dedicated (no unknown-key rewording).
    q = tmp_path / "deck_refused.in"
    q.write_text(BASE_INPUT + "use_shipped_minimax_tables = true\n")
    with pytest.raises(ValueError, match="regenerate_minimax_tables"):
        read_lorrax_input(str(q))


def test_auto_on_cpu_chol_passes_lu_demotes_announced(tmp_path, monkeypatch):
    # 'auto' MAY demote, announced: distributed_cholesky=auto passes
    # through (it carries the replicated rank-truncation route on CPU);
    # distributed_lu=auto demotes to 'off' and says so.
    _pin_backend(monkeypatch, "cpu")
    lines: list[str] = []
    path = tmp_path / "cohsex_auto_cpu.in"
    path.write_text(BASE_INPUT)
    cfg = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "off"
    assert any("distributed_lu=auto" in l and "off" in l for l in lines)
