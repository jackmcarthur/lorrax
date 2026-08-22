"""``interp_along_omega`` may not clamp an OUTPUT to a grid endpoint silently.

THE DEFECT.  ``interp_along_omega`` applied ``np.clip`` to every requested
evaluation energy, unconditionally and with nothing said.  The SC
Hamiltonian path counts its clipped cells and routes out-of-range bands
through the scissor partition; the final ``eqp0.dat`` / ``eqp1.dat`` at-DFT
writer applied the same edge clamp without refusing, masking or stamping
which states were uncovered.

MEASURED on the exact-origin Na bandstructure run
(``runs/Na/02_soc48b_qsgw_mpa/49_origin_exact_fresh_np_ladder_20260818/
01_np10_nested_batch8``, log lines 5078-5113): the deck requests only
``[-5, +26] eV`` relative to fixed-N mu while bands 5--8 lie near -25.07 eV,
the log reports ``QSGW: 10142 clipped (41.3%)``, and at Gamma band 5's
stored ``sigC = 2.791702837+0.013253924i`` is bit-for-shown-digits the
``omega = -5 eV`` ENDPOINT of ``sigma_mnk.h5``, not Sigma at its DFT energy.
The wide-window control moves that band from -32.511754 to -28.524225 eV
(+3.987528 eV), leaving -0.156302 eV against BGW's Eqp0 = -28.367923 eV.

THE FIX IS A NAMED POLICY WITH NO DEFAULT, plus a count that is always
reported.  These cells pin both halves, and the bit-identity of the
``clamp`` arm against the historical arithmetic.
"""
from __future__ import annotations

import ast
import glob
import os

import numpy as np
import pytest

from gw.qsgw_utils import (OUT_OF_RANGE_POLICIES, interp_along_omega,
                           omega_coverage, resolve_out_of_range_policy)


_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")

_ENV = "LORRAX_OMEGA_OUT_OF_RANGE"


def _values(n_omega=9, nk=3, nb=4, seed=0):
    rng = np.random.default_rng(seed)
    omega = np.linspace(-4.0, 4.0, n_omega)
    vals = (rng.normal(size=(n_omega, nk, nb))
            + 1j * rng.normal(size=(n_omega, nk, nb)))
    return omega, vals


# ---------------------------------------------------------------------------
# The policy is required, and unknown values refuse
# ---------------------------------------------------------------------------

def test_out_of_range_has_no_default():
    """A default would let a new OUTPUT call site clamp by accident, which
    is exactly how the at-DFT writer came to ship 41.3 % endpoint values."""
    omega, vals = _values()
    with pytest.raises(TypeError):
        interp_along_omega(vals, omega, np.zeros((3, 4)))


def test_an_unknown_policy_refuses():
    omega, vals = _values()
    with pytest.raises(ValueError, match="out_of_range"):
        interp_along_omega(vals, omega, np.zeros((3, 4)),
                           out_of_range="edge")


def test_every_call_site_in_src_names_its_policy():
    """AST census, not a grep: the argument is only load-bearing if no call
    site can omit it, and a keyword in a comment is not a call site."""
    missing = []
    for path in glob.glob(os.path.join(_SRC, "**", "*.py"), recursive=True):
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (getattr(node.func, "id", None)
                    or getattr(node.func, "attr", None))
            if name != "interp_along_omega":
                continue
            if "out_of_range" not in {kw.arg for kw in node.keywords}:
                missing.append(f"{os.path.relpath(path, _SRC)}:{node.lineno}")
    assert not missing, (
        f"interp_along_omega call sites with no out_of_range policy: "
        f"{missing}")


# ---------------------------------------------------------------------------
# Coverage, and what each policy does with the uncovered cells
# ---------------------------------------------------------------------------

def test_coverage_counts_only_the_states_outside_the_grid():
    omega, _ = _values()
    ev = np.array([[-9.0, 0.0, 3.9, 4.0],
                   [-4.0, 1.0, 2.0, 12.0]])
    covered, n_out, frac = omega_coverage(omega, ev)
    # The two ENDPOINTS are covered: they were sampled.
    assert covered[0, 3] and covered[1, 0]
    assert n_out == 2 and covered.sum() == 6
    assert frac == pytest.approx(2.0 / 8.0)


def test_clamp_is_bit_identical_to_the_historical_arithmetic():
    """The default arm must not move a single existing number.

    Reproduced here from the pre-2026-08-22 body rather than asserted
    against a stored constant, so this stays a statement about the two
    implementations and not about one machine's rounding.
    """
    omega, vals = _values(seed=3)
    ev = np.array([[-9.0, 0.0, 3.9, 4.4],
                   [-4.0, 1.0, 2.0, 12.0]])
    got = interp_along_omega(vals, omega, ev, out_of_range="clamp")

    n_omega = omega.size
    eval_clamped = np.clip(ev, float(omega[0]), float(omega[-1]))
    idx_hi = np.clip(np.searchsorted(omega, eval_clamped, side="left"),
                     1, n_omega - 1)
    idx_lo = idx_hi - 1
    denom = np.where(omega[idx_hi] > omega[idx_lo],
                     omega[idx_hi] - omega[idx_lo], 1.0)
    w_hi = (eval_clamped - omega[idx_lo]) / denom
    k_idx = np.arange(ev.shape[0])[:, None]
    n_idx = np.arange(ev.shape[1])[None, :]
    want = ((1.0 - w_hi) * vals[idx_lo, k_idx, n_idx]
            + w_hi * vals[idx_hi, k_idx, n_idx])
    np.testing.assert_array_equal(got, want)


def test_clamp_really_does_return_the_endpoint():
    """The observable that makes the defect visible: an out-of-grid state
    comes back as the grid's own edge value, which is a finite, plausible
    number and not Sigma at that energy."""
    omega, vals = _values(seed=5)
    ev = np.full((3, 4), -50.0)
    got = interp_along_omega(vals, omega, ev, out_of_range="clamp")
    np.testing.assert_allclose(got, vals[0])


def test_mask_marks_the_uncovered_cells_non_finite():
    omega, vals = _values(seed=7)
    ev = np.array([[-9.0, 0.0, 3.9, 4.0],
                   [-4.0, 1.0, 2.0, 12.0]])
    got = interp_along_omega(vals, omega, ev, out_of_range="mask")
    covered, _, _ = omega_coverage(omega, ev)
    assert np.all(np.isnan(got[~covered]))
    assert np.all(np.isfinite(got[covered]))


def test_refuse_names_the_count_the_fraction_and_the_worst():
    omega, vals = _values(seed=9)
    ev = np.array([[-9.0, 0.0, 3.9, 4.0],
                   [-4.0, 1.0, 2.0, 12.0]])
    with pytest.raises(ValueError) as exc:
        interp_along_omega(vals, omega, ev, out_of_range="refuse",
                           context="unit")
    text = str(exc.value)
    assert "2 of 8" in text and "25.0%" in text
    assert "+12.000" in text
    assert "sigma_omega_patches_ev" in text or "sigma_omega_min_ev" in text


def test_a_fully_covered_request_is_never_reported_and_never_refused():
    """NOT-VOID control.  Without it every assertion above could be
    satisfied by a policy that fires unconditionally."""
    omega, vals = _values(seed=11)
    ev = np.zeros((3, 4))
    lines = []
    for policy in OUT_OF_RANGE_POLICIES:
        got = interp_along_omega(vals, omega, ev, out_of_range=policy,
                                 context="unit", print_fn=lines.append)
        assert np.all(np.isfinite(got))
    assert lines == [], lines


def test_the_report_line_is_emitted_exactly_once_and_carries_the_numbers():
    omega, vals = _values(seed=13)
    ev = np.array([[-9.0, 0.0, 3.9, 4.0],
                   [-4.0, 1.0, 2.0, 12.0]])
    lines = []
    interp_along_omega(vals, omega, ev, out_of_range="clamp",
                       context="Sigma_c at E_DFT", print_fn=lines.append)
    assert len(lines) == 1, lines
    assert "Sigma_c at E_DFT" in lines[0]
    assert "2 of 8" in lines[0] and "25.0%" in lines[0]
    assert "policy=clamp" in lines[0]


# ---------------------------------------------------------------------------
# The env override
# ---------------------------------------------------------------------------

def test_the_policy_override_parses_and_refuses_a_typo(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert resolve_out_of_range_policy() == "clamp"
    for good in OUT_OF_RANGE_POLICIES:
        monkeypatch.setenv(_ENV, good)
        assert resolve_out_of_range_policy() == good
        monkeypatch.setenv(_ENV, good.upper())
        assert resolve_out_of_range_policy() == good
    for bad in ("edge", "nearest", "1", "off"):
        monkeypatch.setenv(_ENV, bad)
        with pytest.raises(ValueError, match=_ENV):
            resolve_out_of_range_policy()
