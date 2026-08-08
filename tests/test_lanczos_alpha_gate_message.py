"""The alpha-Hermiticity gate must say what it measured and where to look.

Why this file exists
--------------------
``solvers/lanczos.py``'s alpha-Hermiticity gate was correct, deterministic
and ignored for ten days.  It was ignored because its message named the
gloo ``psum_scatter`` corruption as "the known cause on this stack" and
told the reader to re-run under
``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` before believing any
eigenvalue -- advice that is falsified three ways on the only deck that
has ever fired the gate (the residual is bit-identical across fourteen
P=4 runs, it is present at P=1 where there is no collective at all, and
it reproduces on a single-process login-node dense probe).  The real
cause is the mini-BZ Coulomb head injected at the slot labelled Miller
(0,0,0), a label that is not equivariant under q -> -q.

Evidence: ``~/lorrax_bse_perf_2026-08-08/HERMITICITY_INVESTIGATION.md``.

A gate's message is part of the gate.  A wrong message costs exactly what
a wrong tolerance costs, and nothing in the tree could see it, so these
cells pin the message the way the constant above it is pinned: two
content assertions on the text the human actually receives, and a RED
TWIN that reinstates the pre-2026-08-08 wording and requires both
assertions to fail against it.  Without the twin these are string
compares that would pass against any sufficiently long paragraph.
"""
import os
import sys

import pytest

_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from solvers import lanczos  # noqa: E402


# The measured triple from the real control arm, so the cells are pinned to
# a number a human has seen rather than to a synthetic one:
# /pscratch/sd/j/jackm/vcoul_head_0808/_reports/bse4p_control_400.log
#   max|A-Aᴴ|/max|A| = 1.155e-06 (abs 6.268e-07 on scale 5.425e-01)
_DEV_FAIL = 6.268e-07
_SCALE = 5.425e-01
_WORST = 153

# The verbatim wording this branch retired, kept here and NOWHERE else so
# the red twin measures the real regression rather than a paraphrase of it.
_PRE_20260808_CAUSE = (
    " means the matvec did not return H*q -- the "
    "operator, not the algorithm, is wrong.  Known cause on this stack: a "
    "silent reduce-scatter corruption (jax.lax.psum_scatter under "
    "JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo returns wrong data in ~5% of "
    "executions, always output segment 0; see "
    "wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md).  Re-run under "
    "JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi (clean in 584/584) before "
    "believing any eigenvalue from this solve.  Other candidates: a "
    "non-Hermitian W/V tile fed to the matvec, or a mis-transposed shard."
)


def _fire_the_gate(capsys, monkeypatch, cause=None):
    """Run the failing branch of the gate and return what the human sees."""
    monkeypatch.setenv("LORRAX_SANITY", "1")   # warn, never raise
    if cause is not None:
        monkeypatch.setattr(lanczos, "_ALPHA_CAUSE", cause)
    ok = lanczos._report_alpha_herm(
        "lanczos_eig_jit", "vec", _DEV_FAIL, _SCALE, _WORST)
    assert ok is False, "a 1.155e-06 residual must not pass a 1e-9 gate"
    return capsys.readouterr().out


def _assert_states_the_measured_quantity(text):
    """The message must say WHAT was measured, in the reader's own units."""
    assert "max|A-Aᴴ|/max|A|" in text          # the ratio, spelled out
    assert "1.155e-06" in text                       # rel, as computed
    assert "6.268e-07" in text and "5.425e-01" in text   # abs and scale
    assert "j=153" in text                           # where it peaked
    assert "max_j|Im alpha_j|" in text               # the definition of dev
    assert "max_j|alpha_j|" in text                  # the definition of scale


def _assert_points_at_the_head_slot_mechanism(text):
    """The head-equivariance mechanism must lead; the collective must not."""
    assert "Miller (0,0,0)" in text
    assert "q -> -q" in text
    assert "argmin|q+G|" in text
    # The collective may still be listed -- it is a real bug -- but it is a
    # last resort behind an explicit determinism test, not the headline.
    assert "psum_scatter" in text, "do not delete a real cause, demote it"
    assert text.index("Miller (0,0,0)") < text.index("psum_scatter")
    assert "DETERMINISM IS THE DISCRIMINATOR" in text
    # and the specific instruction that misdirected readers is gone
    assert "before believing any eigenvalue" not in text
    assert "Known cause on this stack: a silent reduce-scatter" not in text


def test_alpha_gate_message_states_the_measured_quantity(capsys, monkeypatch):
    _assert_states_the_measured_quantity(_fire_the_gate(capsys, monkeypatch))


def test_alpha_gate_message_points_at_the_head_slot(capsys, monkeypatch):
    _assert_points_at_the_head_slot_mechanism(
        _fire_the_gate(capsys, monkeypatch))


def test_message_assertions_can_fail__RED_TWIN(capsys, monkeypatch):
    """Reinstate the retired wording; BOTH gates above must reject it.

    This is what makes the two cells above measurements rather than
    tautologies: the same predicates, run against the text this branch
    replaced, have to fail.
    """
    text = _fire_the_gate(capsys, monkeypatch, cause=_PRE_20260808_CAUSE)

    # the old text never defined dev or scale for the reader
    with pytest.raises(AssertionError):
        _assert_states_the_measured_quantity(text)
    # ... and it led with the collective
    with pytest.raises(AssertionError):
        _assert_points_at_the_head_slot_mechanism(text)

    # positive control: the parts the old text DID carry are still there,
    # so the twin fails for the reason claimed and not because the harness
    # silently produced an empty string.
    assert "max|A-Aᴴ|/max|A|" in text and "1.155e-06" in text
    assert "psum_scatter" in text


def test_alpha_gate_ok_line_reports_the_ratio_and_the_tolerance(
        capsys, monkeypatch):
    """The passing branch is the one a landed fix must produce.

    3.165e-14 is what si_bse_debug measures once the head is injected at
    argmin|q+G| -- the fixed arm of the A/B this branch reproduces.
    """
    monkeypatch.setenv("LORRAX_SANITY", "1")
    ok = lanczos._report_alpha_herm(
        "lanczos_eig_jit", "vec", 3.165e-14 * _SCALE, _SCALE, 23)
    assert ok is True
    out = capsys.readouterr().out
    assert "alpha non-Hermitian part / max|alpha| = 3.165e-14" in out
    assert "(tol 1e-09, worst j=23)  OK" in out


def test_the_tolerance_is_still_the_derived_one():
    """No landing gets to relax it: the fixed arm clears 1e-9 by 4.5 orders."""
    assert lanczos.ALPHA_HERM_RTOL == 1e-9
