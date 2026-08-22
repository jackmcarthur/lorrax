"""Unit tests for the ``qp_solver`` config axis and the ``SCConfig`` group.

Covers the auto-resolution contract of ``LorraxConfig.qp_solver``
(G0W0_SC_TOGGLE_DESIGN.md §2a):

1. Default → ``ONE_SHOT_DFT`` (standard G0W0).
2. The deleted ``self_consistent`` BOOLEAN refuses, naming
   ``qp_solver = self_consistent``.
3. The deleted ``sigma_at_dft_energies`` is ignored with a word; the
   run it named is the default.
4. ``qp_solver = auto`` is gone with the boolean it read.
5. Validation: ``fixed_point`` × static mode → error; the removed
   ``sigma_omega_accumulation = kij_stream`` spelling → parse-time error.
6. ``SCConfig``: sc_* input keys parsed and DECK-ONLY (the six
   ``LORRAX_SC_*`` env twins were deleted in 0.1.0); knob validation.

All tests run on a throwaway input file — no WFN, no GPU, no jit.
"""
from __future__ import annotations

import pathlib
import re

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
    path.write_text(_with_compute_mode(BASE_INPUT + extra))
    return LorraxConfig.from_input_file(str(path), print_fn=lambda *a, **k: None)


def _with_compute_mode(body: str, mode: str = "cohsex") -> str:
    """Supply ``compute_mode`` unless the case under test names its own.

    The key is REQUIRED as of 0.1.0.  ``cohsex`` is what the deleted
    ``compute_mode = auto`` resolved to for a deck that named none of the
    three legacy flags, so every case that does not care about the mode
    keeps the behaviour it was written against.
    """
    if re.search(r"^\s*compute_mode\s*[=:]", body, re.M):
        return body
    head, sep, rest = body.partition("\n")
    return f"{head}{sep}compute_mode = {mode}\n{rest}"


# ---------------------------------------------------------------------------
# auto-resolution
# ---------------------------------------------------------------------------

def test_default_is_one_shot_dft(tmp_path):
    cfg = _config(tmp_path)
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT


def test_the_self_consistent_boolean_refuses_naming_qp_solver(tmp_path):
    """It was a KEY named exactly like another key's VALUE.

    Refused, not ignored: it chose between one-shot and a QSGW loop.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "self_consistent = true\n")
    msg = str(exc.value)
    assert "self_consistent" in msg
    assert "qp_solver = self_consistent" in msg


def test_sigma_freq_debug_output_refuses_naming_the_file_key(tmp_path):
    """Folded into the filename (empty = off) in 0.1.0.

    REFUSED rather than translated: ``true`` has to become a FILENAME, and
    picking one for the deck would be the parser inventing an output path.
    """
    from gw.gw_config import read_lorrax_input

    p = tmp_path / "deck_freqdbg.in"
    p.write_text(_with_compute_mode(BASE_INPUT)
                 + "sigma_freq_debug_output = true\n")
    with pytest.raises(ValueError) as exc:
        read_lorrax_input(str(p))
    assert "sigma_freq_debug_file = sigma_freq_debug.dat" in str(exc.value)


def test_the_debug_table_is_off_by_default_and_on_by_naming_a_file(tmp_path):
    assert _config(tmp_path).debug.sigma_freq_debug_file == ""
    cfg = _config(tmp_path, "sigma_freq_debug_file = sfd.dat\n")
    assert cfg.debug.sigma_freq_debug_file.endswith("sfd.dat")
    assert not hasattr(cfg.debug, "sigma_freq_debug_output")


@pytest.mark.parametrize("key, env", [
    ("r_chunk_size", "LORRAX_R_CHUNK"),
    ("gflat_chunk_size", "LORRAX_GFLAT_CHUNK"),
    ("vq_g_chunk_size", "LORRAX_VQ_G_CHUNK"),
])
def test_the_demoted_chunk_keys_name_their_env_twin(tmp_path, key, env):
    """Demoted, not deleted: the override seam is unchanged.

    Warn-and-ignore rather than refuse — an HBM schedule moves bytes
    between stages and changes no number, so an old deck should keep
    running and be told where the dial went.
    """
    from gw.gw_config import read_lorrax_input

    p = tmp_path / "deck_chunk.in"
    p.write_text(_with_compute_mode(BASE_INPUT) + f"{key} = 64\n")
    with pytest.warns(DeprecationWarning, match=key):
        params = read_lorrax_input(str(p))
    assert key not in params


@pytest.mark.parametrize("env, field", [
    ("LORRAX_R_CHUNK", "r_chunk_override"),
    ("LORRAX_GFLAT_CHUNK", "gflat_chunk_size"),
    ("LORRAX_VQ_G_CHUNK", "vq_g_chunk_size"),
])
def test_the_env_twin_reaches_the_same_memory_field(tmp_path, monkeypatch,
                                                    env, field):
    assert getattr(_config(tmp_path).memory, field) == 0
    monkeypatch.setenv(env, "64")
    assert getattr(_config(tmp_path).memory, field) == 64


def test_a_non_integer_chunk_env_announces_and_falls_back(tmp_path,
                                                          monkeypatch):
    """``int(float("12.5"))`` would be a silently different schedule."""
    from gw.gw_config import env_int, reset_env_announce_state

    reset_env_announce_state()
    lines: list[str] = []
    monkeypatch.setenv("LORRAX_R_CHUNK", "12.5")
    assert env_int("LORRAX_R_CHUNK", 0, print_fn=lines.append) == 0
    assert "LORRAX SANITY" in "\n".join(lines)
    assert "LORRAX_R_CHUNK" in "\n".join(lines)


def test_there_is_no_bse_config_group(tmp_path):
    """Five keys, one dataclass, zero readers.

    ``bandstructure.htransform`` re-parses the deck and reads the raw
    params dict; ``bse_setup`` takes the values as arguments.  The frozen
    group was a second, quieter copy — deleted in 0.1.0.  The KEYS stay.
    """
    from gw import gw_config

    assert not hasattr(gw_config, "BSEConfig")
    cfg = _config(tmp_path, "wfn_fi_q_chunk = 12\n")
    assert not hasattr(cfg, "bse")
    for key in ("get_centroids_fi", "wfn_fi_min", "wfn_fi_max", "kgrid_fi",
                "wfn_fi_q_chunk"):
        assert key in gw_config._DEFAULTS


def test_the_q_chunk_refusal_survived_the_group(tmp_path):
    """The one load-bearing line ``BSEConfig`` carried."""
    with pytest.raises(ValueError, match="wfn_fi_q_chunk"):
        _config(tmp_path, "wfn_fi_q_chunk = -3\n")


def test_sigma_at_dft_energies_is_ignored_and_says_so(tmp_path):
    """Deleted 0.1.0.  It was parsed and never read for its whole life.

    Warn-and-ignore rather than refuse (the ``slab_io`` rule): it never
    changed a number, so an old deck keeps running and is told what it
    lost.  The run it describes is the DEFAULT.
    """
    with pytest.warns(DeprecationWarning, match="sigma_at_dft_energies"):
        cfg = _config(tmp_path, "sigma_at_dft_energies = true\n")
    assert cfg.qp_solver is QPSolver.ONE_SHOT_DFT
    assert not hasattr(cfg.sigma, "sigma_at_dft_energies")


def test_qp_solver_auto_is_no_longer_a_value(tmp_path):
    """``auto`` existed only to read the deleted boolean."""
    cfg = _config(tmp_path, "qp_solver = auto\n")
    with pytest.raises(ValueError, match="auto"):
        cfg.qp_solver


@pytest.mark.parametrize("solver", ["one_shot_dft", "self_consistent"])
def test_explicit_qp_solver_values_resolve(tmp_path, solver):
    assert _config(tmp_path,
                   f"qp_solver = {solver}\n").qp_solver is QPSolver(solver)


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


def test_bispinor_with_self_consistent_refuses_naming_the_dropped_term(
        tmp_path):
    """The SC path never threads ``wfns_transverse``, so Σ^B vanishes.

    Checked for the FAILURE signature: the message must name Σ^B, because
    "unsupported combination" would not tell an operator that the run they
    already have is missing a term.
    """
    cfg = _config(tmp_path, "bispinor = true\nqp_solver = self_consistent\n")
    with pytest.raises(ValueError) as exc:
        cfg.qp_solver
    msg = str(exc.value)
    assert "bispinor" in msg and "self_consistent" in msg
    assert "Σ^B" in msg and "DROPPED" in msg


@pytest.mark.parametrize("solver", ["one_shot_dft", "fixed_point"])
def test_bispinor_with_the_non_sc_solvers_still_resolves(tmp_path, solver):
    """RED TWIN: only the SC leg is refused."""
    cfg = _config(tmp_path,
                  f"bispinor = true\ncompute_mode = gn_ppm\n"
                  f"qp_solver = {solver}\n")
    assert cfg.qp_solver is QPSolver(solver)


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
    # ``accelerator`` left this record when ``sc_accelerator`` was retired
    # (2026-08-14); rCROP is the only accelerator.  See test_staged_sc.py
    # for the retirement's own cells.
    sc = _config(tmp_path).sc
    assert (sc.max_iter, sc.tol_ev, sc.history_depth,
            sc.mixing, sc.dump_dir) == (20, 1.0e-4, 5, 1.0, None)
    assert not hasattr(sc, "accelerator")


def test_sc_input_keys(tmp_path):
    sc = _config(
        tmp_path,
        "sc_max_iter = 7\nsc_tol_ev = 1e-6\n"
        "sc_history_depth = 3\nsc_mixing = 0.5\nsc_dump_dir = sc_hist\n").sc
    assert (sc.max_iter, sc.tol_ev, sc.history_depth,
            sc.mixing, sc.dump_dir) == (7, 1.0e-6, 3, 0.5, "sc_hist")


def test_the_sc_env_twins_no_longer_outrank_the_deck(tmp_path, monkeypatch):
    """All six DELETED in 0.1.0, with the ``_sc_env`` shim.

    An env var that outranks the deck on a physics knob makes a run
    unreproducible from its own input file.  Checked for the FAILURE
    signature — every twin exported at once, and the DECK's values must
    come out — rather than for the absence of a helper.

    ``sc_accelerator`` is in the deck below on purpose: the KEY was
    retired by the staged-SC merge (warn-and-ignore, rCROP is the only
    accelerator) and the ENV twin was deleted here, so the deck line must
    neither steer anything nor stop the parse.
    """
    for name, val in (("LORRAX_SC_MAX_ITER", "3"),
                      ("LORRAX_SC_TOL_EV", "1e-10"),
                      ("LORRAX_SC_ACCEL", "linear"),
                      ("LORRAX_SC_DEPTH", "2"),
                      ("LORRAX_SC_MIXING", "0.25"),
                      ("LORRAX_SC_DUMP_DIR", "/tmp/sc_dump")):
        monkeypatch.setenv(name, val)
    lines: list[str] = []
    path = tmp_path / "cohsex_env.in"
    path.write_text(_with_compute_mode(
        BASE_INPUT + "sc_max_iter = 99\nsc_accelerator = rcrop\n"))
    with pytest.warns(DeprecationWarning, match="sc_accelerator"):
        sc = LorraxConfig.from_input_file(
            str(path),
            print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))),
        ).sc
    assert (sc.max_iter, sc.tol_ev, sc.history_depth,
            sc.mixing, sc.dump_dir) == (99, 1.0e-4, 5, 1.0, None)
    assert not any("deprecated env override" in l for l in lines)
    # The env twin must not even be ANNOUNCED: it is not read, and a
    # line about it would be a reader by another name.
    assert not any("LORRAX_SC_ACCEL" in l for l in lines)


def test_the_sc_env_shim_is_gone_from_the_source():
    """No reader may survive the row's deletion from the registry."""
    import ast

    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "gw" / "gw_config.py").read_text()
    tree = ast.parse(src)
    # A comment naming the deleted shim is history, not a reader; a
    # DEFINITION of it is the thing this cell exists to catch.
    assert not [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_sc_env"]
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for name in ("LORRAX_SC_MAX_ITER", "LORRAX_SC_TOL_EV", "LORRAX_SC_ACCEL",
                 "LORRAX_SC_DEPTH", "LORRAX_SC_MIXING", "LORRAX_SC_DUMP_DIR"):
        assert name not in strings


# ---------------------------------------------------------------------------
# Distributed-linalg axes: distributed_cholesky / distributed_lu
# (portable backend names; the legacy cusolvermp_charge/cusolvermp_lu
# aliases were REMOVED 2026-07-31 — unknown deck keys are ignored)
#
# The parse outcome is JAX-backend-dependent (explicit cusolvermp REFUSES
# on a CPU backend — doctrine 3, audit fix/zq 2026-07-28; 'auto' LU
# demotes there with an announcement), so each cell PINS the backend via
# monkeypatch instead of inheriting whatever the test box runs, and the
# parse time no longer runs any transport capability probe (2026-08-06).
# ---------------------------------------------------------------------------

def _pin_backend(monkeypatch, backend_name: str):
    import jax
    monkeypatch.setattr(jax, "default_backend", lambda: backend_name)
    # The two slab_io router stubs that used to live here are gone with the
    # routers (2026-08-06).  Parse time no longer runs any transport
    # capability probe or MPI init, so there is nothing left to stub.


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
    # ``strict_keys = false`` because the default flipped to true for
    # 0.1.0: what is under test here is that the removed aliases do not
    # STEER, which needs the deck to parse at all.
    _pin_backend(monkeypatch, "gpu")
    cfg = _config(tmp_path, "strict_keys = false\n"
                  "cusolvermp_charge = on\ncusolvermp_lu = off\n")
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
    # ``strict_keys = false`` is now the OPT-IN (the default flipped to
    # true for 0.1.0), so the lenient arm has to ask for itself.
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_unknown.in"
    p.write_text(BASE_INPUT + "strict_keys = false\n"
                 "x_only = true\nuse_chunked_isdf = false\n")
    params = read_lorrax_input(str(p))
    out = capsys.readouterr().out
    # One aggregated report naming each key with its line number.
    assert "unrecognized deck key" in out
    assert "x_only (line 7)" in out
    assert "use_chunked_isdf (line 8)" in out
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


# ---------------------------------------------------------------------------
# compute_mode is REQUIRED, and the three legacy ansatz flags it used to be
# inferred from REFUSE.  They are refused rather than warn-and-ignored
# because each of them selected a PHYSICS ARM.
# ---------------------------------------------------------------------------

def test_compute_mode_is_required(tmp_path):
    from gw.gw_config import LorraxConfig

    p = tmp_path / "deck_nomode.in"
    p.write_text(BASE_INPUT)          # deliberately NOT _with_compute_mode
    with pytest.raises(ValueError) as exc:
        LorraxConfig.from_input_file(str(p), print_fn=lambda *a, **k: None)
    msg = str(exc.value)
    assert "compute_mode is REQUIRED" in msg
    for mode in ("x_only", "cohsex", "gn_ppm", "hl_ppm", "mpa"):
        assert mode in msg


def test_a_typo_in_compute_mode_fails_at_parse_not_at_first_read(tmp_path):
    """An omission and a typo must fail at the same moment."""
    from gw.gw_config import LorraxConfig

    p = tmp_path / "deck_typo.in"
    p.write_text(_with_compute_mode(BASE_INPUT, mode="cohsexx"))
    with pytest.raises(ValueError, match="cohsexx"):
        LorraxConfig.from_input_file(str(p), print_fn=lambda *a, **k: None)


def test_auto_is_no_longer_a_compute_mode_value(tmp_path):
    from gw.gw_config import LorraxConfig

    p = tmp_path / "deck_auto.in"
    p.write_text(_with_compute_mode(BASE_INPUT, mode="auto"))
    with pytest.raises(ValueError, match="auto"):
        LorraxConfig.from_input_file(str(p), print_fn=lambda *a, **k: None)


@pytest.mark.parametrize("line, needle", [
    ("do_screened = false", "compute_mode = x_only"),
    ("do_screened = true", "compute_mode = x_only"),
    ("use_ppm_sigma = true", "compute_mode = gn_ppm"),
    ("ppm_model = hl", "compute_mode = hl_ppm"),
])
def test_retired_ansatz_flags_refuse_naming_their_replacement(
        tmp_path, line, needle):
    from gw.gw_config import read_lorrax_input

    p = tmp_path / "deck_retired_mode.in"
    p.write_text(_with_compute_mode(BASE_INPUT) + line + "\n")
    with pytest.raises(ValueError) as exc:
        read_lorrax_input(str(p))
    msg = str(exc.value)
    assert line.split("=")[0].strip() in msg
    assert needle in msg


def test_all_three_retired_flags_are_named_at_once(tmp_path):
    """One report, not three runs of trial and error."""
    from gw.gw_config import read_lorrax_input

    p = tmp_path / "deck_retired_all.in"
    p.write_text(_with_compute_mode(BASE_INPUT)
                 + "do_screened = true\nuse_ppm_sigma = true\n"
                   "ppm_model = gn\n")
    with pytest.raises(ValueError) as exc:
        read_lorrax_input(str(p))
    msg = str(exc.value)
    assert all(k in msg for k in
               ("do_screened", "use_ppm_sigma", "ppm_model"))


# ---------------------------------------------------------------------------
# UNIMPLEMENTED_OPTIONS — declared-but-not-built option VALUES, refused at
# parse time.  Both rows previously died deep in a kernel, after the WFN
# read and the whole screening stage.
# ---------------------------------------------------------------------------

def test_ppm_invalid_mode_imaginary_refuses_at_parse_time(tmp_path):
    """It used to die inside the Σ^c kernel (ppm_sigma.py:877)."""
    with pytest.raises(NotImplementedError) as exc:
        _config(tmp_path, "ppm_invalid_mode = imaginary\n")
    msg = str(exc.value)
    assert "ppm_invalid_mode = imaginary" in msg
    assert "complex-Omega" in msg


@pytest.mark.parametrize("value",
                         ["zero", "2ry", "static_limit", "skip", "infinity"])
def test_the_other_invalid_pole_values_still_parse(tmp_path, value):
    """RED TWIN: only the one value with no code path is refused."""
    assert (_config(tmp_path, f"ppm_invalid_mode = {value}\n")
            .ppm.invalid_mode == value)


def test_unimplemented_option_values_stay_legal_vocabulary(tmp_path):
    """Parseable, refused — the same contract UNIMPLEMENTED_MODES has.

    A config echo, a layering test or an operator reading a deck back must
    be able to NAME the value; what must not happen is spending a run on
    it.  So the validators that own the vocabulary still accept both.
    """
    from gw.gw_config import MPAConfig, PPMConfig, UNIMPLEMENTED_OPTIONS

    assert set(UNIMPLEMENTED_OPTIONS) == {
        ("mpa_material_class", "metal"),
        ("ppm_invalid_mode", "imaginary"),
    }
    # Neither validator refuses the value it owns.
    PPMConfig(omega_p=2.0, fallback_omega=2.0,
              head_omega_h_ry=None, probe_chi_reuse="off",
              sigma_target_error=1e-6, sigma_max_nodes=64,
              invalid_mode="imaginary")
    MPAConfig(n_poles=8, material_class="metal", sampling_alpha=1,
              varpi_near_ry=0.2, varpi_far_ry=2.0, pole_batch_size=4, sigma_sector_target_error=6.5e-4,
              sigma_crossing_target_error=2.0e-3, sigma_max_nodes=96)


# ---------------------------------------------------------------------------
# Parse-time band-window / dimensionality ranges.  ``zeta_nband`` was the
# only one of the five ever checked here; the other four each died far
# downstream, in the vocabulary of whatever site tripped over them.
# ---------------------------------------------------------------------------

def _geom_config(tmp_path, *, nval=2, ncond=2, nband=10, sys_dim=2):
    path = tmp_path / "cohsex_geom.in"
    path.write_text(
        f"[cohsex]\ncompute_mode = cohsex\n"
        f"nval = {nval}\nncond = {ncond}\nnband = {nband}\n"
        f"sys_dim = {sys_dim}\nmemory_per_device_gb = 4.0\n")
    return LorraxConfig.from_input_file(str(path),
                                        print_fn=lambda *a, **k: None)


@pytest.mark.parametrize("kwargs, needle", [
    ({"nval": -1}, "nval=-1"),
    ({"ncond": -3}, "ncond=-3"),
    ({"nband": 0}, "nband=0"),
    ({"nband": -10}, "nband=-10"),
    ({"sys_dim": 1}, "sys_dim=1"),
    ({"sys_dim": 4}, "sys_dim=4"),
])
def test_geometry_ranges_refuse_at_parse_time(tmp_path, kwargs, needle):
    with pytest.raises(ValueError, match=re.escape(needle)):
        _geom_config(tmp_path, **kwargs)


@pytest.mark.parametrize("sys_dim", [0, 2, 3])
def test_legal_sys_dim_values_still_parse(tmp_path, sys_dim):
    assert _geom_config(tmp_path, sys_dim=sys_dim).sys_dim == sys_dim


def test_zero_is_a_legal_nval_and_ncond(tmp_path):
    """The boundary is >= 0, not > 0: nval = 0 puts b1 at n_occ."""
    cfg = _geom_config(tmp_path, nval=0, ncond=0)
    assert (cfg.nval, cfg.ncond) == (0, 0)


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


def test_every_tracked_fixture_deck_has_no_dead_keys(tmp_path):
    """No tracked ``*.in`` fixture may carry a key nothing reads.

    The guard for a cleanup that would otherwise silently rot: 7 of 7
    fixtures once carried keys absent from _DEFAULTS (``x_only`` in six of
    them -- a VALUE of ``compute_mode``, never a key -- plus
    ``use_chunked_isdf``, ``sigma_debug_split_contrib``, ``max_r_chunks``,
    ``profile_qloop``, ``profile_trace_dir``).  Every one parsed clean and
    steered nothing.

    ``strict_keys`` is forced ON here regardless of the shipped default,
    so this stays a real gate if that default never flips.  The knob is
    injected right after the ``[cohsex]`` header: appending at EOF can
    land past the section end when a K_POINTS block follows, which would
    silently test nothing.
    """
    import re
    from gw.gw_config import read_lorrax_input

    repo = pathlib.Path(__file__).resolve().parent.parent
    decks = sorted(p for p in repo.glob("tests/**/*.in")
                   if re.search(r"^\s*\[cohsex\]", p.read_text(errors="replace"),
                                re.I | re.M))
    assert decks, "no [cohsex] fixture decks found -- glob is wrong"

    offenders = {}
    for deck in decks:
        out, injected = [], False
        for ln in deck.read_text().splitlines(keepends=True):
            if re.match(r"\s*strict_keys\s*[=:]", ln, re.I):
                continue
            out.append(ln)
            if not injected and ln.strip().lower().startswith("[cohsex]"):
                out.append("strict_keys = true\n")
                injected = True
        probe = tmp_path / deck.name
        probe.write_text("".join(out))
        try:
            read_lorrax_input(str(probe))
        except ValueError as exc:
            if "strict_keys" in str(exc):
                offenders[deck.relative_to(repo)] = str(exc).splitlines()[1:]
            else:
                raise
    assert not offenders, (
        "tracked fixture decks carry keys that nothing reads:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items()))


# ---------------------------------------------------------------------------
# Renamed-key aliases (``_ALIASES``).  The table ships EMPTY — the rename
# phases populate it — so these tests inject a row rather than pinning a
# spelling that does not exist yet.  What is under test is the mechanism:
# an old spelling parses as the new key, says so once, and a deck naming
# both with different values refuses.
# ---------------------------------------------------------------------------

#: Same as ``BASE_INPUT`` minus ``nband`` — the alias tests set that key
#: through the alias, and configparser refuses a duplicate option.
ALIAS_BASE_INPUT = """\
[cohsex]
nval = 2
ncond = 2
memory_per_device_gb = 4.0
"""


@pytest.fixture
def _alias_row(monkeypatch):
    """One ``old_nband -> nband`` row, live for the duration of a test."""
    from gw import gw_config
    monkeypatch.setitem(gw_config._ALIASES, "old_nband", "nband")
    return ("old_nband", "nband")


def test_alias_old_spelling_parses_as_the_new_key(tmp_path, capsys,
                                                  _alias_row):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_alias.in"
    p.write_text(ALIAS_BASE_INPUT + "old_nband = 42\n")
    with pytest.warns(DeprecationWarning, match="old_nband"):
        params = read_lorrax_input(str(p))
    assert params["nband"] == 42
    out = capsys.readouterr().out
    # One rank-0 line naming BOTH spellings; not an "unrecognized" report.
    assert "old_nband (line 5): parsed as 'nband'" in out
    assert "not a recognized deck key" not in out


def test_alias_survives_strict_keys(tmp_path, _alias_row):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_alias_strict.in"
    p.write_text(ALIAS_BASE_INPUT + "strict_keys = true\nold_nband = 42\n")
    with pytest.warns(DeprecationWarning):
        params = read_lorrax_input(str(p))     # must not raise
    assert params["nband"] == 42


def test_alias_both_spellings_same_value_is_not_ambiguous(tmp_path,
                                                          _alias_row):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_alias_agree.in"
    p.write_text(ALIAS_BASE_INPUT + "old_nband = 42\nnband = 42\n")
    with pytest.warns(DeprecationWarning):
        params = read_lorrax_input(str(p))
    assert params["nband"] == 42


def test_alias_both_spellings_different_values_refuse(tmp_path, _alias_row):
    from gw.gw_config import read_lorrax_input
    p = tmp_path / "deck_alias_conflict.in"
    p.write_text(ALIAS_BASE_INPUT + "old_nband = 42\nnband = 7\n")
    with pytest.raises(ValueError) as exc:
        read_lorrax_input(str(p))
    # Both spellings AND both values named — the deck author has to be able
    # to see which line to delete.
    assert "old_nband" in str(exc.value) and "nband" in str(exc.value)
    assert "42" in str(exc.value) and "7" in str(exc.value)


def test_alias_table_ships_empty(tmp_path):
    """Seeded empty: a row here is a rename phase's deliberate gesture."""
    from gw.gw_config import _ALIASES
    assert _ALIASES == {}


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


@pytest.mark.parametrize("key, line", [
    ("sc_accelerator", "sc_accelerator = rcrop"),
    ("sigma_omega_batch_size", "sigma_omega_batch_size = 8"),
])
def test_the_staged_sc_merge_retirements_warn_rather_than_vanish(
        tmp_path, key, line):
    """Two keys the staged-SC merge deleted from ``_DEFAULTS``.

    A ``_LEGACY_DECK_KEYS`` row alone is NOT enough: the row only exempts
    the key from the unknown-key check, so a key with a row and no branch
    is dropped in SILENCE — which under ``strict_keys = true`` is strictly
    worse than the refusal it replaced.  Each therefore needs a branch,
    and this cell is the failure signature for its absence.
    """
    from gw.gw_config import read_lorrax_input, _LEGACY_DECK_KEYS

    assert key in _LEGACY_DECK_KEYS, (
        f"{key} was deleted from _DEFAULTS; without a legacy row it "
        f"REFUSES under the 0.1.0 strict_keys default")
    p = tmp_path / f"deck_{key}.in"
    p.write_text(BASE_INPUT + line + "\n")
    with pytest.warns(DeprecationWarning, match=key):
        read_lorrax_input(str(p))


def test_sc_accelerator_linear_refuses_by_name(tmp_path):
    """The asymmetry is the point: ``rcrop`` warns, ``linear`` refuses.

    ``linear`` named a DIFFERENT fixed-point iteration, so running rCROP
    under that name would be a mode substitution — the same class as the
    ``screening_method = ctsp`` spelling that silently ran minimax.
    """
    from gw.gw_config import read_lorrax_input

    p = tmp_path / "deck_accel_linear.in"
    p.write_text(BASE_INPUT + "sc_accelerator = linear\n")
    with pytest.raises(ValueError, match="sc_accelerator"):
        read_lorrax_input(str(p))


def test_auto_on_cpu_chol_passes_lu_demotes_announced(tmp_path, monkeypatch):
    # 'auto' MAY demote, announced: distributed_cholesky=auto passes
    # through (it carries the replicated rank-truncation route on CPU);
    # distributed_lu=auto demotes to 'off' and says so.
    _pin_backend(monkeypatch, "cpu")
    lines: list[str] = []
    path = tmp_path / "cohsex_auto_cpu.in"
    path.write_text(_with_compute_mode(BASE_INPUT))
    cfg = LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert cfg.backend.distributed_cholesky == "auto"
    assert cfg.backend.distributed_lu == "off"
    assert any("distributed_lu=auto" in l and "off" in l for l in lines)
