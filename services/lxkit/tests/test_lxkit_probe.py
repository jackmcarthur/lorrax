"""``lxkit.probe`` — the ABSENT/BROKEN split and the three-way reason.

PURE STDLIB; runs under pytest or a bare interpreter (see _lxkit_harness).
"""

from __future__ import annotations

from _lxkit_harness import FakeMesh, raises, run_module        # noqa: I001

from lxkit.gate import Gate
from lxkit.probe import (
    AVAILABLE, LibraryNotBuilt, LibraryUnusable, ProbeResult, missing_symbol,
    not_loadable, unknown_target,
)


# ---------------------------------------------------------------------------
# ProbeResult IS the (ok, reason) pair Gate.probe is declared to return
# ---------------------------------------------------------------------------

def test_a_probe_result_unpacks_as_the_pair_a_gate_expects():
    r = ProbeResult(False, "because")
    ok, reason = r
    assert (ok, reason) == (False, "because")
    assert isinstance(r, tuple) and r == (False, "because")


def test_a_probe_returning_a_ProbeResult_drives_a_gate_unchanged():
    """The two surfaces compose without an adapter — which is the point of
    making it a NamedTuple rather than a dataclass."""
    def _gate(probe):
        return Gate(env="LXKIT_PROBE_TEST", target="t", platforms=("cpu",),
                    modes=("off", "on"), default="on", off_label="native",
                    probe=probe)

    assert _gate(lambda t, p: AVAILABLE).require(FakeMesh("cpu")) == "cpu"
    with raises(RuntimeError, match="does not export lrx_t"):
        _gate(lambda t, p: missing_symbol(t, "lrx_t", "/x/lib.so")
              ).require(FakeMesh("cpu"))


def test_available_is_the_positive_answer_and_carries_a_reason():
    assert AVAILABLE.ok is True and AVAILABLE.reason == "available"


# ---------------------------------------------------------------------------
# ABSENT vs BROKEN — two types because they want opposite responses
# ---------------------------------------------------------------------------

def test_absent_and_broken_are_distinct_types_neither_catching_the_other():
    """Collapsing them is how a broken dependency closure gets reported as a
    platform skip (19 cells, 2026-08-06)."""
    assert not issubclass(LibraryNotBuilt, LibraryUnusable)
    assert not issubclass(LibraryUnusable, LibraryNotBuilt)
    try:
        raise LibraryUnusable("present and will not dlopen")
    except LibraryNotBuilt:
        raise AssertionError("BROKEN must not be caught as ABSENT")
    except LibraryUnusable:
        pass


def test_each_type_keeps_the_builtin_that_states_its_kind():
    """ABSENT is a filesystem fact, BROKEN is a loader fact — so callers
    that only speak builtins still get the distinction."""
    assert issubclass(LibraryNotBuilt, FileNotFoundError)
    assert issubclass(LibraryUnusable, OSError)


# ---------------------------------------------------------------------------
# The three-way reason taxonomy (ffi_loader.probe_target:438-497 shape)
# ---------------------------------------------------------------------------

def test_all_three_reasons_are_negative_and_carry_text():
    for r in (unknown_target("t", "cpu", ("a", "b")),
              not_loadable("cpu", OSError("libfftw3.so.mpi31.3: not found")),
              missing_symbol("t", "lrx_t", "/x/lib.so")):
        assert r.ok is False
        assert r.reason.strip()


def test_unknown_target_names_what_the_library_does_know():
    r = unknown_target("lrx_typo", "CUDA", ("lrx_eigh", "lrx_potrf"))
    assert "unknown target" in r.reason
    assert "lrx_typo" in r.reason
    assert "lrx_eigh" in r.reason and "lrx_potrf" in r.reason


def test_not_loadable_never_tells_anyone_to_rebuild():
    """THE defect this taxonomy removes: a missing LD_LIBRARY_PATH entry
    reported as 'not compiled' silently downgraded an Nx1 mesh to native
    (wk_P G4, 2026-07-25).  The handler may well be compiled."""
    r = not_loadable("cpu", OSError("libfftw3.so.mpi31.3: not found"),
                     target="lrx_pgemm",
                     hint="LORRAX_FFI_HOST_SO pins the .so; run `ldd <so>`.")
    assert "could not be loaded" in r.reason
    assert "libfftw3.so.mpi31.3" in r.reason and "OSError" in r.reason
    assert "rebuild" not in r.reason.lower()
    assert "says nothing about whether lrx_pgemm is compiled" in r.reason
    assert "ldd <so>" in r.reason


def test_missing_symbol_is_the_only_reason_that_says_rebuild():
    """The genuine partial-build case — and the ONLY one."""
    r = missing_symbol("lrx_eigh", "lrx_scalapack_eigh", "/opt/lib.so",
                       build_hint="config/perlmutter/build_ffi_host.sh")
    assert "does not export lrx_scalapack_eigh" in r.reason
    assert "/opt/lib.so" in r.reason
    assert "Rebuild with config/perlmutter/build_ffi_host.sh" in r.reason


def test_the_three_reasons_are_distinguishable_from_each_other():
    """A harness must be able to route on them; three strings that read the
    same would be a bare bool with extra steps."""
    reasons = {unknown_target("t", "cpu").reason,
               not_loadable("cpu", OSError("x")).reason,
               missing_symbol("t", "s", "/p").reason}
    assert len(reasons) == 3


def test_the_optional_service_specific_text_is_omitted_when_absent():
    """lxkit ships the SHAPE; the tables, env-var names and build hints are
    the service's.  With none supplied the reason must still stand alone."""
    assert "known:" not in unknown_target("t", "cpu").reason
    assert "Rebuild" not in missing_symbol("t", "s", "/p").reason
    assert "NOTE" not in not_loadable("cpu", OSError("x")).reason


if __name__ == "__main__":
    raise SystemExit(run_module(dict(globals())))
