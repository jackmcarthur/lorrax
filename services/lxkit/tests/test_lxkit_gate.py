"""``lxkit.gate`` — grammar, announcement discipline, and the refusing probe.

PURE STDLIB.  No pytest, no jax: these cells are the proof that a service's
policy layer resolves on a machine with nothing installed.  Run either way::

    python -m pytest services/lxkit/tests/test_lxkit_gate.py
    python3 services/lxkit/tests/test_lxkit_gate.py
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from _lxkit_harness import FakeMesh, env, raises, run_module   # noqa: I001

from lxkit.gate import (
    FFI_PLATFORM_MAP, Gate, MODE_HELP, MODE_SPELLINGS, announce_once,
    dial_key, mesh_ffi_platform, platform_from_env, rank0, rank_id,
    reset_gate_state,
)

ENV = "LXKIT_TEST_DIAL"


def _gate(**kw) -> Gate:
    """A gate with the shape every LORRAX dial has, overridable per cell."""
    base = dict(env=ENV, target="lxkit_test_target", platforms=("cpu",),
                modes=("off", "on"), default="on", off_label="native jnp")
    base.update(kw)
    return Gate(**base)


def _say(fn):
    """Run ``fn``; return (result, everything it printed)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


# ---------------------------------------------------------------------------
# Vocabulary — two-valued, and `auto` is gone
# ---------------------------------------------------------------------------

def test_the_vocabulary_is_two_valued_and_has_no_auto():
    """`auto` was deleted with decisions.md 2026-08-01.  A token with no
    resolver branch would behave as `on` silently, which is the shape the
    ruling forbids."""
    assert set(MODE_SPELLINGS) == {"off", "on"}
    assert "auto" not in MODE_SPELLINGS
    assert set(MODE_HELP) == set(MODE_SPELLINGS)


def test_every_spelling_resolves_to_its_mode():
    for mode, spellings in MODE_SPELLINGS.items():
        for spelling in spellings:
            for cased in (spelling, spelling.upper(), f"  {spelling} "):
                with env(**{ENV: cased}):
                    assert _gate().mode() == mode, (mode, cased)


def test_unset_and_whitespace_take_the_declared_default():
    for value in (None, "", "   "):
        with env(**{ENV: value}):
            assert _gate(default="on").mode() == "on"
            assert _gate(default="off").mode() == "off"


# ---------------------------------------------------------------------------
# Grammar errors fall to the DEFAULT, not to `off`
# ---------------------------------------------------------------------------

def test_a_grammar_error_takes_the_DEFAULT_not_off():
    """docs/services.md:546 says these fall to `off`.  The code says DEFAULT
    and the code is right (gate.py's own docstring argues why): an
    unparseable value is not evidence of intent either way, so take the
    direction that cannot break a run — and for a gate whose native twin was
    deleted, `off` is itself a refusal, so falling back to it would turn a
    typo into a dead run.  Pinned here so the stale doc cannot be 'fixed'
    into the code."""
    with env(**{ENV: "Y"}):
        result, said = _say(_gate(default="on").mode)
        assert result == "on"
        assert "not a recognized value" in said
        reset_gate_state()
        result, said = _say(_gate(default="off").mode)
    assert result == "off", "the fallback IS self.default, not 'off' by name"
    assert "OFF" in said


def test_a_grammar_error_announces_the_accepted_spellings_once():
    with env(**{ENV: "maybe"}):
        g = _gate()
        _, first = _say(g.mode)
        _, second = _say(g.mode)
    assert "maybe" in first and "0/off/false/no" in first
    assert "1/on/true/yes" in first
    assert second == "", "the grammar note must not repeat every read"


def test_a_stale_auto_export_is_a_grammar_error_not_a_mode():
    with env(**{ENV: "auto"}):
        result, said = _say(_gate(default="on").mode)
    assert result == "on"
    assert "not a recognized value" in said


def test_a_mode_the_gate_does_not_declare_is_a_grammar_error():
    """`modes=("on",)` means `=0` is not this gate's vocabulary."""
    with env(**{ENV: "0"}):
        result, said = _say(_gate(modes=("on",), default="on").mode)
    assert result == "on"
    assert "not a recognized value" in said


# ---------------------------------------------------------------------------
# __post_init__ — a mode with no resolver branch cannot be declared
# ---------------------------------------------------------------------------

def test_post_init_refuses_a_mode_with_no_resolver_branch():
    with raises(ValueError, match="not in the gate vocabulary"):
        _gate(modes=("on", "auto"))


def test_post_init_refuses_a_default_outside_the_vocabulary():
    with raises(ValueError, match="is not one of"):
        _gate(default="auto")


def test_post_init_fires_at_construction_not_at_the_first_env_read():
    """Otherwise mode() KeyErrors on whichever rank reads the variable
    first — a per-rank crash for a per-process declaration error."""
    with raises(ValueError):
        _gate(modes=("sometimes",))


# ---------------------------------------------------------------------------
# announce_once — validate BEFORE burning the key
# ---------------------------------------------------------------------------

def test_announce_once_prints_once_per_key():
    assert _say(lambda: announce_once(("k",), "hello"))[0] is True
    printed, said = _say(lambda: announce_once(("k",), "hello"))
    assert printed is False and said == ""


def test_reset_gate_state_rearms_the_key():
    _say(lambda: announce_once(("k",), "hello"))
    reset_gate_state()
    printed, said = _say(lambda: announce_once(("k",), "hello"))
    assert printed is True and "hello" in said


def test_a_bad_scope_is_rejected_without_burning_the_key():
    """A rejected call must not consume the key, or the retry after the fix
    is silently swallowed."""
    with raises(ValueError, match="announce scope"):
        announce_once(("k",), "hello", scope="everyone")
    printed, said = _say(lambda: announce_once(("k",), "hello"))
    assert printed is True and "hello" in said


def test_an_empty_message_is_rejected_without_burning_the_key():
    """A blank announcement tells nobody anything while looking like the
    doctrine was honored."""
    for blank in ("", "   "):
        with raises(ValueError, match="empty message"):
            announce_once(("k",), blank)
    printed, _ = _say(lambda: announce_once(("k",), "hello"))
    assert printed is True


# ---------------------------------------------------------------------------
# Rank discipline
# ---------------------------------------------------------------------------

def test_rank_id_reads_the_launcher_vars_in_order():
    with env(SLURM_PROCID="3", PMI_RANK="7", OMPI_COMM_WORLD_RANK="9"):
        assert rank_id() == 3
    with env(SLURM_PROCID=None, PMI_RANK="7", OMPI_COMM_WORLD_RANK="9"):
        assert rank_id() == 7
    with env(SLURM_PROCID=None, PMI_RANK=None, OMPI_COMM_WORLD_RANK="9"):
        assert rank_id() == 9


def test_an_unparseable_rank_is_unknown_not_a_crash():
    with env(SLURM_PROCID="nope"):
        assert rank_id() is None
    # An EMPTY launcher var is not a rank: it is skipped, and with no other
    # launcher set the answer is the jax fallback (0) or None.
    with env(SLURM_PROCID="", PMI_RANK=None, OMPI_COMM_WORLD_RANK=None):
        assert rank_id() in (None, 0)


def test_rank0_speaks_when_the_rank_is_unknown():
    with env(SLURM_PROCID="0"):
        assert rank0() is True
    with env(SLURM_PROCID="1"):
        assert rank0() is False


def test_a_rank_invariant_announcement_is_rank0_only():
    with env(SLURM_PROCID="2"):
        printed, said = _say(lambda: announce_once(("k",), "geometry"))
    assert printed is False and said == ""


def test_a_rank_local_announcement_speaks_from_the_rank_it_happened_on():
    """The env is per-process and a probe failure is usually one rank's
    LD_LIBRARY_PATH.  Tagged on ranks >= 1 so it is attributable, untagged
    on rank 0 so the harnesses' grep line stays byte-identical."""
    with env(SLURM_PROCID="2"):
        printed, said = _say(
            lambda: announce_once(("k",), "my env is wrong", scope="local"))
    assert printed is True and said.strip() == "[rank 2] my env is wrong"
    reset_gate_state()
    with env(SLURM_PROCID="0"):
        _, said = _say(
            lambda: announce_once(("k",), "my env is wrong", scope="local"))
    assert said.strip() == "my env is wrong"


# ---------------------------------------------------------------------------
# Platform vocabulary — a PARAMETER, not a hard-coded table
# ---------------------------------------------------------------------------

def test_the_default_platform_map_is_the_cpu_CUDA_table():
    assert mesh_ffi_platform(FakeMesh("cpu")) == "cpu"
    assert mesh_ffi_platform(FakeMesh("gpu")) == "CUDA"
    assert mesh_ffi_platform(FakeMesh("cuda")) == "CUDA"


def test_an_unmapped_platform_passes_through_so_a_refusal_can_name_it():
    """'tpu' reads better than 'unknown'."""
    assert mesh_ffi_platform(FakeMesh("tpu")) == "tpu"
    assert mesh_ffi_platform(FakeMesh("rocm")) == "rocm"


def test_a_service_supplies_its_own_platform_vocabulary():
    rocm = {"rocm": "HIP", "cpu": "cpu"}
    assert mesh_ffi_platform(FakeMesh("rocm"), rocm) == "HIP"
    assert mesh_ffi_platform(FakeMesh("gpu"), rocm) == "gpu"
    assert FFI_PLATFORM_MAP["gpu"] == "CUDA", "default table unchanged"


def test_platform_from_env_reads_JAX_PLATFORMS_without_touching_jax():
    with env(JAX_PLATFORMS=None):
        assert platform_from_env() == "CUDA"
        assert platform_from_env("cpu") == "cpu"
    with env(JAX_PLATFORMS="cpu"):
        assert platform_from_env() == "cpu"
    with env(JAX_PLATFORMS="cuda,cpu"):
        assert platform_from_env() == "CUDA"
    with env(JAX_PLATFORMS="gpu"):
        assert platform_from_env() == "CUDA"


# ---------------------------------------------------------------------------
# The probe — the DEFAULT REFUSES
# ---------------------------------------------------------------------------

def test_an_unwired_gate_refuses_rather_than_assuming_the_handler_is_there():
    """A permissive default would turn 'nobody wired a probe' into 'the
    handler is fine' — the silent downgrade lxkit.gate exists to forbid."""
    g = _gate()
    assert g.probe is None
    with raises(RuntimeError, match="no probe is wired"):
        g.require(FakeMesh("cpu"))


def test_the_refusal_names_the_target_the_platform_and_the_fix():
    msg = ""
    try:
        _gate().require(FakeMesh("cpu"))
    except RuntimeError as exc:
        msg = str(exc)
    assert "lxkit_test_target" in msg and "'cpu'" in msg
    assert "probe=" in msg, "the message must name the way out"


def test_a_wired_probe_is_the_one_that_is_asked():
    seen = []

    def probe(target, platform):
        seen.append((target, platform))
        return True, "available"

    assert _gate(probe=probe).require(FakeMesh("cpu")) == "cpu"
    assert seen == [("lxkit_test_target", "cpu")]


def test_a_failed_probe_refuses_quoting_the_reason_verbatim():
    """'the .so would not load' and 'the .so has no such handler' have
    different fixes; a bare bool tells somebody to rebuild a library that
    is fine."""
    reason = "loaded /x/lib.so but it does not export lrx_thing"
    try:
        _gate(probe=lambda t, p: (False, reason)).require(FakeMesh("cpu"))
    except RuntimeError as exc:
        assert reason in str(exc)
    else:
        raise AssertionError("a failed probe must refuse")


def test_an_out_of_scope_platform_refuses_before_the_probe_runs():
    def probe(target, platform):
        raise AssertionError("must not probe an out-of-scope platform")

    with raises(RuntimeError, match="this dial exists on cpu only"):
        _gate(probe=probe).require(FakeMesh("gpu"))


def test_the_receipt_is_announced_once_on_rank0():
    g = _gate(probe=lambda t, p: (True, "available"),
              resolved_msg={"cpu": "[lxkit] using {target}"})
    _, first = _say(lambda: g.require(FakeMesh("cpu")))
    _, second = _say(lambda: g.require(FakeMesh("cpu")))
    assert "[lxkit] using lxkit_test_target" in first
    assert second == ""


# ---------------------------------------------------------------------------
# resolve / enforce
# ---------------------------------------------------------------------------

def test_off_resolves_to_None_without_probing_anything():
    def probe(target, platform):
        raise AssertionError("an off gate must not load a library")

    with env(**{ENV: "0"}):
        g = _gate(probe=probe)
        result, said = _say(lambda: g.resolve(FakeMesh("cpu")))
    assert result is None and said == ""


def test_an_out_of_scope_platform_demotes_with_an_announcement():
    result, said = _say(lambda: _gate().resolve(FakeMesh("gpu")))
    assert result is None
    assert "OFF" in said and "'gpu'" in said


def test_silence_on_demote_requires_a_declared_reason():
    g = _gate(silent_platform_demote="the native GPU lowering IS the path")
    result, said = _say(lambda: g.resolve(FakeMesh("gpu")))
    assert result is None and said == ""


def test_a_gate_that_wrote_no_demote_message_still_announces():
    """An unset field can never turn a demote silent — silence requires
    silent_platform_demote, which must carry its reason."""
    assert _gate().platform_demote_msg == ""
    _, said = _say(lambda: _gate().resolve(FakeMesh("tpu")))
    assert said.strip() != ""


def test_enforce_off_with_a_retained_native_path_announces_the_opt_out():
    with env(**{ENV: "0"}):
        result, said = _say(lambda: _gate().enforce(FakeMesh("cpu")))
    assert result is None
    assert "debug opt-out" in said and "UNCERTIFIED" in said


def test_enforce_off_where_the_native_twin_was_deleted_refuses():
    with env(**{ENV: "0"}):
        with raises(RuntimeError, match="nothing to opt out to"):
            _gate(off_policy="refuse").enforce(FakeMesh("cpu"))


def test_enforce_is_where_a_missing_library_refuses_at_startup():
    with raises(RuntimeError, match="no probe is wired"):
        _gate().enforce(FakeMesh("cpu"))


# ---------------------------------------------------------------------------
# tier 1 stays lexical, and dial_key aggregates it
# ---------------------------------------------------------------------------

def test_enabled_answers_from_the_env_alone():
    def probe(target, platform):
        raise AssertionError("enabled() must not probe: it is a cache key")

    g = _gate(probe=probe)
    with env(**{ENV: "1"}):
        assert g.enabled() is True
    with env(**{ENV: "0"}):
        assert g.enabled() is False


def test_dial_key_tracks_every_gate_it_is_given():
    a = _gate(env="LXKIT_A")
    b = _gate(env="LXKIT_B")
    with env(LXKIT_A="1", LXKIT_B="0"):
        assert dial_key(a, b) == (("LXKIT_A", True), ("LXKIT_B", False))
    with env(LXKIT_A="0", LXKIT_B="0"):
        assert dial_key(a, b) == (("LXKIT_A", False), ("LXKIT_B", False))
    assert dial_key() == ()


def test_dial_key_changes_when_a_dial_flips_midprocess():
    """The whole reason it exists: a kernel cache that omits the dials
    serves a stale backend after a mid-process flag flip."""
    g = _gate()
    with env(**{ENV: "1"}):
        hot = dial_key(g)
    with env(**{ENV: "0"}):
        assert dial_key(g) != hot


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals())))
