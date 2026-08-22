"""Unit tests for the stage-boundary sanity gates (``common.sanity``).

RUNNABLE ON THE LOGIN NODE with plain ``python3`` — no jax, no h5py::

    cd /work2/08271/jackmc/frontera/wt-F
    PYTHONPATH=src python3 tests/test_sanity_gates.py

(also collects under pytest).  Everything here exercises the numpy code
path of ``common.sanity`` plus the pure-python eqp file verifier; the
device path is covered by ``tests/test_sanity_gates_jax.py``, which needs
a jax build.

Why these tests exist
---------------------
These gates are the thing that is supposed to fire when *nothing else
does*.  They are therefore the one piece of the codebase whose failure
mode is total silence — so their own coverage has to be direct: does the
check return False on corrupt input, True on healthy input, and nothing
at all when switched off.
"""
import importlib.util
import os
import sys
import tempfile
import types

import numpy as np

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)


def _load_isolated(mod_name, relpath):
    """Import one source file WITHOUT executing its package ``__init__``.

    ``src/common/__init__.py`` imports jax, and ``gw/eqp_bgw.py`` imports
    h5py — neither exists in the login-node interpreter.  The modules
    under test need neither (``common.sanity`` imports jax lazily, inside
    the device branch; ``gw.eqp_bgw``'s writer and verifier are pure text
    + numpy).  Loading them directly by path is what lets this file run
    on the login node against the *real* shipped code rather than a copy.
    """
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_SRC, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Package shims so intra-package imports inside the loaded modules resolve
# to the isolated copies rather than re-triggering the real __init__.
_common_pkg = sys.modules.setdefault("common", types.ModuleType("common"))
_common_pkg.__path__ = [os.path.join(_SRC, "common")]
sanity = _load_isolated("common.sanity", "common/sanity.py")
_common_pkg.sanity = sanity
collectives = _load_isolated("common.collectives", "common/collectives.py")
_common_pkg.collectives = collectives

if "h5py" not in sys.modules:
    try:
        import h5py  # noqa: F401
    except ImportError:
        # eqp_bgw imports h5py at module scope for its post-hoc CLI
        # orchestrator; the writer/verifier under test never touch it.
        sys.modules["h5py"] = types.ModuleType("h5py")

_units = _load_isolated("common.units", "common/units.py")
_common_pkg.units = _units
_provenance = _load_isolated("common.provenance", "common/provenance.py")
_common_pkg.provenance = _provenance
_gw_pkg = sys.modules.setdefault("gw", types.ModuleType("gw"))
_gw_pkg.__path__ = [os.path.join(_SRC, "gw")]
eqp_bgw = _load_isolated("gw.eqp_bgw", "gw/eqp_bgw.py")
_gw_pkg.eqp_bgw = eqp_bgw


class _Log:
    """Collecting print_fn so tests can assert on what the user would see."""

    def __init__(self):
        self.lines = []

    def __call__(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))

    @property
    def text(self):
        return "\n".join(self.lines)

    @property
    def failures(self):
        return [ln for ln in self.lines if "LORRAX SANITY FAILURE" in ln]


def _with_level(level, fn):
    """Run ``fn`` with LORRAX_SANITY set to ``level`` (None ⇒ unset)."""
    prev = os.environ.get("LORRAX_SANITY")
    if level is None:
        os.environ.pop("LORRAX_SANITY", None)
    else:
        os.environ["LORRAX_SANITY"] = level
    try:
        return fn()
    finally:
        if prev is None:
            os.environ.pop("LORRAX_SANITY", None)
        else:
            os.environ["LORRAX_SANITY"] = prev


def _with_allow_nonfinite(fn):
    """Run ``fn`` with the NAMED forensic escape for ``refuse_nonfinite`` on."""
    key = "LORRAX_ALLOW_NONFINITE_RESULT"
    prev = os.environ.get(key)
    os.environ[key] = "1"
    try:
        return fn()
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


# ---------------------------------------------------------------------------
# Level switch
# ---------------------------------------------------------------------------

def test_level_switch():
    assert _with_level(None, sanity.sanity_enabled) is True
    assert _with_level("1", sanity.sanity_enabled) is True
    for off in ("0", "off", "OFF", "false", "no"):
        assert _with_level(off, sanity.sanity_enabled) is False, off
    assert _with_level("strict", sanity.sanity_strict) is True
    assert _with_level("1", sanity.sanity_strict) is False


def test_off_is_a_true_noop():
    """Disabled checks must return True and print NOTHING, even on garbage."""
    log = _Log()
    bad = np.array([np.nan, np.inf, 1.0])

    def run():
        assert sanity.check_finite("x", bad, print_fn=log) is True
        assert sanity.check_sign("x", np.array([1.0]), print_fn=log) is True
        assert sanity.check_positive("x", -1.0, print_fn=log) is True
        assert sanity.check_count("x", 1, 2, print_fn=log) is True

    _with_level("0", run)
    assert log.lines == [], log.text


def test_strict_raises():
    def run():
        try:
            sanity.check_finite("x", np.array([np.nan]), print_fn=_Log())
        except sanity.SanityError:
            return True
        return False
    assert _with_level("strict", run) is True


# ---------------------------------------------------------------------------
# check_finite
# ---------------------------------------------------------------------------

def test_check_finite_clean():
    log = _Log()
    assert sanity.check_finite("V", np.arange(12.0).reshape(3, 4),
                               print_fn=log) is True
    assert log.failures == []


def test_check_finite_catches_nan_and_inf():
    log = _Log()
    a = np.ones((4, 4), dtype=np.complex128)
    a[1, 1] = np.nan
    a[2, 3] = np.inf
    a[0, 2] = 1.0 + 1j * np.nan          # imaginary-part NaN must count too
    assert sanity.check_finite("W", a, print_fn=log) is False
    assert len(log.failures) == 1
    assert "3 non-finite entries of 16" in log.failures[0]
    assert "2 NaN" in log.failures[0]


def test_check_finite_magnitude_ceiling():
    log = _Log()
    a = np.array([1.0, 1e30])
    assert sanity.check_finite("V", a, expect_max_abs=1e12, print_fn=log) is False
    assert "exceeds the sanity ceiling" in log.failures[0]
    # Finite and within the ceiling ⇒ pass.
    assert sanity.check_finite("V", np.array([1.0, 5.0]),
                               expect_max_abs=1e12, print_fn=_Log()) is True


def test_check_finite_empty_is_ok():
    assert sanity.check_finite("V", np.zeros((0, 3)), print_fn=_Log()) is True


# ---------------------------------------------------------------------------
# check_hermitian
# ---------------------------------------------------------------------------

def test_check_hermitian_true_and_false():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    herm = 0.5 * (m + m.conj().T)
    assert sanity.check_hermitian("V", herm, print_fn=_Log()) is True

    log = _Log()
    broken = herm.copy()
    broken[0, 3] += 0.5                       # break one off-diagonal
    assert sanity.check_hermitian("V", broken, print_fn=log) is False
    assert "is NOT Hermitian" in log.failures[0]


def test_check_hermitian_scale_relative():
    """A tiny absolute deviation on a huge matrix must NOT fire."""
    m = np.eye(4) * 1e9
    m[0, 1] = 1e-3
    m[1, 0] = 1e-3 + 1e-9                     # rel ~1e-18
    assert sanity.check_hermitian("V", m, print_fn=_Log()) is True


def test_check_hermitian_ignores_non_square():
    assert sanity.check_hermitian("x", np.zeros((3, 5)), print_fn=_Log()) is True


# ---------------------------------------------------------------------------
# check_positive / check_in_range / check_sign
# ---------------------------------------------------------------------------

def test_check_positive():
    assert sanity.check_positive("tr V", 6.9e9, print_fn=_Log()) is True
    log = _Log()
    assert sanity.check_positive("tr V", -1.0, print_fn=log) is False
    assert "is not positive" in log.failures[0]
    log = _Log()
    assert sanity.check_positive("tr V", 1.0,
                                 magnitude_hint=(1e8, 1e10),
                                 print_fn=log) is False
    assert "outside the expected magnitude" in log.failures[0]


def test_check_sign_matches_sigma_x_semantics():
    sig_x = np.array([[-12.0, -9.5, -8.1]])
    assert sanity.check_sign("sigx", sig_x, print_fn=_Log()) is True
    log = _Log()
    bad = np.array([[-12.0, +3.0, -8.1]])
    assert sanity.check_sign("sigx", bad, print_fn=log) is False
    assert "1 of 3 entries are not negative" in log.failures[0]


def test_check_in_range_reports_worst():
    log = _Log()
    # The -136 eV QP-gap signature: finite, plausible dtype, absurd value.
    vals = np.array([-2.1, -136.0, 1.4])
    assert sanity.check_in_range("eqp shift", vals, -20.0, 20.0,
                                 unit="eV", print_fn=log) is False
    assert "1 of 3 entries outside" in log.failures[0]
    assert sanity.check_in_range("eqp shift", np.array([-2.1, 1.4]),
                                 -20.0, 20.0, print_fn=_Log()) is True


def test_check_shape_and_count():
    assert sanity.check_shape("x", np.zeros((4, 5)), (4, None),
                              print_fn=_Log()) is True
    log = _Log()
    assert sanity.check_shape("x", np.zeros((4, 5)), (4, 6),
                              print_fn=log) is False
    assert "has shape (4, 5)" in log.failures[0]
    assert sanity.check_count("rows", 10, 10, print_fn=_Log()) is True
    log = _Log()
    assert sanity.check_count("rows", 9, 10, print_fn=log) is False
    assert "got 9, expected 10" in log.failures[0]


# ---------------------------------------------------------------------------
# The eqp writer's structural float count (the check that caught the NaN bug)
# ---------------------------------------------------------------------------

def _write_eqp(path, e_dft, e_qp, kpts, nspin=1):
    return eqp_bgw.write_bgw_eqp(path, kpts, e_dft, e_qp, band_offset=4, nspin=nspin)


def test_eqp_writer_roundtrip_counts():
    verify_eqp_file = eqp_bgw.verify_eqp_file
    nk, nb = 3, 5
    kpts = np.linspace(0, 0.5, nk * 3).reshape(nk, 3)
    e_dft = np.arange(nk * nb, dtype=np.float64).reshape(nk, nb)
    e_qp = e_dft + 0.25
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "eqp0.dat")
        _write_eqp(p, e_dft, e_qp, kpts)
        log = _Log()
        assert verify_eqp_file(p, nk=nk, nb=nb, print_fn=log) is True
        assert log.failures == []
        # Independent count: 3 floats per k-header + 2 per band row.
        with open(p) as fh:
            body = [ln for ln in fh if not ln.startswith("#") and ln.strip()]
        assert len(body) == nk + nk * nb


def test_eqp_verifier_catches_nan_column():
    """A NaN in E_QP keeps the line count but drops the finite-float count.

    This is precisely the 2026-07 SUMMA-back-solve signature: the file
    looks complete, the run exited 0, and only counting floats revealed it.
    """
    verify_eqp_file = eqp_bgw.verify_eqp_file
    nk, nb = 2, 4
    kpts = np.zeros((nk, 3))
    e_dft = np.ones((nk, nb))
    e_qp = np.ones((nk, nb))
    e_qp[1, 2] = np.nan
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "eqp1.dat")
        # Write through the NAMED forensic escape so we get the corrupt
        # file on disk, then verify it with checking ON — mirrors "the run
        # already finished; is its output trustworthy?".  ``LORRAX_SANITY=0``
        # no longer buys this: that switch is a COST escape and since
        # 2026-08-22 the writer's finiteness gate is a refusal that ignores
        # it (``sanity.refuse_nonfinite``), because rc=0 on a 7176-of-7176
        # NaN E_QP column is the defect, not a tuning preference.
        _with_allow_nonfinite(lambda: _write_eqp(p, e_dft, e_qp, kpts))
        log = _Log()
        ok = _with_level("1", lambda: verify_eqp_file(
            p, nk=nk, nb=nb, print_fn=log))
        assert ok is False
        assert any("finite-float count" in f for f in log.failures), log.text
        # Line structure is intact — which is exactly why this was silent.
        with open(p) as fh:
            body = [ln for ln in fh if not ln.startswith("#") and ln.strip()]
        assert len(body) == nk + nk * nb


def test_eqp_verifier_catches_truncated_file():
    verify_eqp_file = eqp_bgw.verify_eqp_file
    nk, nb = 2, 4
    kpts = np.zeros((nk, 3))
    e = np.ones((nk, nb))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "eqp0.dat")
        _write_eqp(p, e, e, kpts)
        with open(p) as fh:
            lines = fh.readlines()
        with open(p, "w") as fh:
            fh.writelines(lines[:-3])         # simulate a truncated write
        log = _Log()
        assert verify_eqp_file(p, nk=nk, nb=nb, print_fn=log) is False
        assert any("band rows" in f for f in log.failures), log.text


def test_eqp_writer_gate_fires_on_nan_input():
    """The writer must REFUSE a non-finite QP column at EVERY sanity level.

    Not merely under ``LORRAX_SANITY=strict``.  The bcc-Fe run of
    2026-08-15 wrote 7176 of 7176 NaN E_QP entries through this call and
    exited **rc=0** in 883 s at the default level (JID 57051742, CLAIMS
    204); a warning does not reach an exit code, and ``LORRAX_SANITY=0``
    is a COST escape, not permission to ship a NaN.
    """
    nk, nb = 2, 3
    kpts = np.zeros((nk, 3))
    e = np.ones((nk, nb))
    bad = e.copy()
    bad[0, 0] = np.inf
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "eqp0.dat")

        def run():
            try:
                _write_eqp(p, e, bad, kpts)
            except sanity.NonFiniteResultError:
                return True
            return False

        for level in ("strict", "1", "0", None):
            assert _with_level(level, run) is True, level

        # NOT VOID: the same writer accepts the same shapes when they are
        # finite, so the refusal above is about the values and not about
        # the call.
        assert _with_level(None, lambda: _write_eqp(p, e, e, kpts))

        # And the NAMED escape is the only thing that lets it through.
        assert _with_level(
            None, lambda: _with_allow_nonfinite(
                lambda: _write_eqp(p, e, bad, kpts)))


# ---------------------------------------------------------------------------
# The eqp mean-field gate — REGRESSION for the miscalibrated first version
# ---------------------------------------------------------------------------

def _capture(fn):
    """Run ``fn`` capturing stdout (the gate's default print_fn)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


def test_eqp_gate_silent_on_large_but_legitimate_qp_shift():
    """A deep semicore state has |eqp0 - E_DFT| >> 50 eV and is CORRECT.

    This is the exact false positive the first version of this gate
    produced: it bracketed the raw QP shift at +/-50 eV and fired on 72 of
    120 states of the cohsex_debug fixture -- a run that reproduced its
    reference eqp to 1e-6 eV in the same job.  Delta = Sigma_xc - V_xc, and
    for semicore bands both terms are of order 100 eV, so the shift is
    legitimately huge while the mean-field identity stays perfectly
    physical.  The gate must key on the identity, not the shift.
    """
    nk, nb = 2, 3
    e_dft = np.array([[-59.16, -33.86, -3.08]] * nk)
    v_h = np.array([[254.64, 337.41, 161.70]] * nk)
    implied_vxc = np.full((nk, nb), -18.0)          # healthy
    kin_ion = e_dft - v_h - implied_vxc
    sigma_x = np.array([[-103.0, -88.0, -21.0]] * nk)   # bare exchange
    sigma_c = np.zeros((nk, nb))

    def run():
        eqp_bgw._warn_on_unphysical_implied_vxc(
            e_dft_ev=e_dft, kin_ion_diag_ev=kin_ion, hartree_diag_ev=v_h)
        return eqp_bgw.compute_eqp_diag(
            kin_ion_diag_ev=kin_ion, hartree_diag_ev=v_h,
            sigma_x_diag_ev=sigma_x, sigma_c_at_dft_diag_ev=sigma_c,
            e_dft_ev=e_dft)
    (eqp0, _), out = _with_level("1", lambda: _capture(run))

    shift = eqp0 - e_dft
    assert np.max(np.abs(shift)) > 50.0, (
        "test is not exercising the large-shift regime")
    assert "LORRAX SANITY FAILURE" not in out, (
        f"gate fired on a healthy run with a legitimately large QP shift:\n{out}")


def test_eqp_gate_fires_on_broken_mean_field():
    """A broken kin_ion + V_H cancellation must still be caught."""
    nk, nb = 2, 3
    e_dft = np.array([[-59.16, -33.86, -3.08]] * nk)
    v_h = np.array([[254.64, 337.41, 161.70]] * nk)
    # 46 eV of ISDF error in V_H -> implied Vxc leaves the physical window,
    # which is the 2026-07 production signature.
    kin_ion = e_dft - v_h - np.full((nk, nb), -18.0) + 60.0
    sigma_x = np.array([[-103.0, -88.0, -21.0]] * nk)

    def run():
        eqp_bgw._warn_on_unphysical_implied_vxc(
            e_dft_ev=e_dft, kin_ion_diag_ev=kin_ion, hartree_diag_ev=v_h)
    _, out = _with_level("1", lambda: _capture(run))
    assert "LORRAX SANITY FAILURE" in out, out
    assert "implied Vxc" in out, out


def test_compute_eqp_diag_does_not_duplicate_the_mean_field_gate():
    """The shared math function must stay silent about H0.

    On the live driver path ``gw_output.write_results`` already runs
    ``_warn_on_unphysical_h0`` on these same arrays immediately before
    calling the writer.  Putting the check in ``compute_eqp_diag`` too
    made a single broken H0 report itself twice, in two different
    wordings — verified against the cohsex_debug fixture, where both
    fired on the identical 86 of 120 states.  The gate belongs to
    ``make_eqp_bgw`` (the post-hoc CLI, which has no other guard).
    """
    nk, nb = 2, 3
    e_dft = np.array([[-59.16, -33.86, -3.08]] * nk)
    v_h = np.array([[254.64, 337.41, 161.70]] * nk)
    kin_ion = e_dft - v_h - np.full((nk, nb), -18.0) + 60.0   # broken H0

    def run():
        return eqp_bgw.compute_eqp_diag(
            kin_ion_diag_ev=kin_ion, hartree_diag_ev=v_h,
            sigma_x_diag_ev=np.zeros((nk, nb)),
            sigma_c_at_dft_diag_ev=np.zeros((nk, nb)),
            e_dft_ev=e_dft)
    _, out = _with_level("1", lambda: _capture(run))
    assert "LORRAX SANITY FAILURE" not in out, (
        f"compute_eqp_diag duplicated the driver-side H0 guard:\n{out}")


def test_eqp_gate_window_matches_gw_output():
    """The writer-side window must not drift from the driver-side guard."""
    lo, hi = eqp_bgw._implied_vxc_window_ev()
    assert (lo, hi) == (-50.0, 2.0), (lo, hi)


# ---------------------------------------------------------------------------
# common.collectives.barrier — single-process semantics
# ---------------------------------------------------------------------------

def test_barrier_is_noop_without_jax_multihost():
    """In a 1-process context the barrier must skip, not raise."""
    assert collectives.process_count() >= 1
    assert collectives.barrier("unit_test_barrier", print_fn=_Log()) is False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:               # noqa: BLE001 - test runner
            import traceback
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())


# ---------------------------------------------------------------------------
# refuse_nonfinite — the guard the ONE-SHOT path skips
# ---------------------------------------------------------------------------
#
# The SC path already refuses a non-finite spectrum, on its second map call,
# through ``sc_iteration._solve_head_occupations -> efermi.OccupationState``.
# A one-shot run never reaches a second map call, so on bcc Fe every element
# of ``sigma_c_kij_ev`` (27,067,872 of 27,067,872) and every one of 7176 E_QP
# entries came back NaN behind rc=0.  These cells pin the refusal that closes
# that path, and — as important — pin that it does NOT fire on health.

def test_refuse_nonfinite_is_silent_on_a_clean_array():
    log = _Log()
    sanity.refuse_nonfinite("V", np.arange(12.0).reshape(3, 4), print_fn=log)
    sanity.refuse_nonfinite("empty", np.zeros((0, 3)), print_fn=log)
    assert log.lines == [], log.text


def test_refuse_nonfinite_raises_regardless_of_the_sanity_level():
    """``LORRAX_SANITY`` buys back the COST of a reduction, never a NaN."""
    a = np.ones((3, 4))
    a[1, 2] = np.nan

    def run():
        try:
            sanity.refuse_nonfinite("Sigma_c", a, print_fn=_Log())
        except sanity.NonFiniteResultError:
            return True
        return False

    for level in ("strict", "1", "0", "off", None):
        assert _with_level(level, run) is True, level


def test_refuse_nonfinite_counts_nan_and_inf_separately():
    a = np.ones((2, 3), dtype=np.complex128)
    a[0, 0] = np.nan
    a[0, 1] = np.inf
    a[1, 1] = 1.0 + 1j * np.nan          # an imaginary-part NaN counts too
    log = _Log()
    try:
        sanity.refuse_nonfinite("W", a, print_fn=log)
    except sanity.NonFiniteResultError as exc:
        assert "3 non-finite entries of 6" in str(exc), str(exc)
        assert "2 NaN, 1 Inf" in str(exc), str(exc)
    else:
        raise AssertionError("a NaN-bearing result was accepted")
    assert any("LORRAX SANITY FAILURE" in ln for ln in log.lines), log.text


def test_the_named_escape_downgrades_it_to_the_loud_warning():
    a = np.array([np.nan, 1.0])
    log = _Log()
    _with_allow_nonfinite(
        lambda: sanity.refuse_nonfinite("E_qp", a, print_fn=log))
    assert any("LORRAX SANITY FAILURE" in ln for ln in log.lines), log.text
    assert any("LORRAX_ALLOW_NONFINITE_RESULT" in ln for ln in log.lines)
