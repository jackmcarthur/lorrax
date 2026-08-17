"""Metallic Sigma refuses before ``number_bands_sigma`` truncates supp(f)."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from gw import w_isdf
from gw.band_extrapolation import trivial_plan
from gw.mpa.sigma import _branches
from gw.ppm_pipeline import compute_ppm_sigma_pipeline
from gw.ppm_sigma import compute_sigma_c_ppm_omega_grid


_ENK = np.asarray([[-2.0, -1.0, -0.1, 0.1, 1.0, 2.0]])
_OMEGA = np.asarray([-0.2, 0.4])


def _wfns(sigma_stop):
    slices = SimpleNamespace(
        b0=0,
        b2=2,
        b4=sigma_stop,
        sigma_sum=slice(0, sigma_stop),
        nb_sigma_sum=sigma_stop,
    )
    return SimpleNamespace(
        enk=_ENK,
        occ=np.asarray([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        slices=slices,
    )


def _state(f_kn):
    return SimpleNamespace(f_kn=np.asarray([f_kn]), mu_ry=0.0)


def test_ppm_partition_seam_refuses_a_truncated_metal_support(monkeypatch):
    """The refusal fires before PPM reads any pole/kernel operand."""
    monkeypatch.delenv(w_isdf.OCCUPATION_SUPPORT_ENV, raising=False)
    wfns = _wfns(3)
    state = _state([1.0, 1.0, 0.4, 0.2, 0.0, 0.0])
    plan = trivial_plan(3, 2, 3)

    with pytest.raises(w_isdf.OccupationSupportTruncationError) as exc:
        compute_sigma_c_ppm_omega_grid(
            wfns,
            object(),
            SimpleNamespace(b_id_4_sigma_user=3),
            None,
            ppm_cfg=None,
            sigma_cfg=None,
            quad=None,
            omega_grid_ry=np.asarray([0.0]),
            occupation_state=state,
            plan=plan,
        )
    text = str(exc.value)
    assert "number_bands_sigma=3" in text
    assert "required minimum number_bands_sigma=4" in text
    assert "occupation_support_bandwidth=4" in text
    assert "LORRAX_OCCUPATION_SUPPORT=allow-truncated" in text
    assert "ppm_sigma bracket partition" in text


def test_mpa_override_warns_with_the_same_numbers_and_continues(monkeypatch):
    monkeypatch.setenv(
        w_isdf.OCCUPATION_SUPPORT_ENV,
        w_isdf.OCCUPATION_SUPPORT_ALLOW_TRUNCATED,
    )
    lines = []
    branches = _branches(
        _wfns(3),
        _OMEGA,
        0.0,
        occupation_state=_state([1.0, 1.0, 0.4, 0.2, 0.0, 0.0]),
        print_fn=lines.append,
    )
    assert branches
    text = "\n".join(lines)
    assert "WARNING" in text
    assert "number_bands_sigma=3" in text
    assert "required minimum number_bands_sigma=4" in text
    assert "occupation_support_bandwidth=4" in text
    assert "Continuing because LORRAX_OCCUPATION_SUPPORT=allow-truncated" in text


def test_a_sigma_sum_containing_supp_f_is_silent(monkeypatch):
    """The unoccupied tail may remain truncated; only supp(f) is required."""
    monkeypatch.delenv(w_isdf.OCCUPATION_SUPPORT_ENV, raising=False)
    lines = []
    branches = _branches(
        _wfns(4),
        _OMEGA,
        0.0,
        occupation_state=_state([1.0, 1.0, 0.4, 0.2, 0.0, 0.0]),
        print_fn=lines.append,
    )
    assert branches
    assert lines == []
    assert all(np.asarray(branch.E_A).shape[-1] == 4 for branch in branches)


def test_insulating_branches_never_evaluate_the_guard(monkeypatch):
    """``occupation_state=None`` returns through the incumbent path first."""
    calls = {"n": 0}

    def forbidden(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("occupation support evaluated on an insulator")

    monkeypatch.setattr(w_isdf, "assert_sigma_contains_occupation_support",
                        forbidden)
    wfns = _wfns(4)
    branches = _branches(wfns, _OMEGA, 0.0, occupation_state=None)
    assert calls["n"] == 0
    assert all(branch.band_weight is None for branch in branches)
    np.testing.assert_array_equal(
        np.asarray(branches[0].base_mask_A),
        np.asarray(wfns.occ[:, :4]) <= 0.5,
    )


def test_negative_mp1_wrong_side_extremum_sets_the_required_minimum(
    monkeypatch,
):
    """A one-sided ``f > 0`` bound would miss band 4; ``abs(f)`` keeps it."""
    monkeypatch.delenv(w_isdf.OCCUPATION_SUPPORT_ENV, raising=False)
    f = np.asarray([[1.0, 1.0, 0.4, -0.0355, 0.0, 0.0]])
    assert np.flatnonzero(np.any(f > 0.0, axis=0))[-1] + 1 == 3
    with pytest.raises(w_isdf.OccupationSupportTruncationError) as exc:
        w_isdf.assert_sigma_contains_occupation_support(
            _ENK, f, slice(0, 3), where="wrong-side fixture")
    text = str(exc.value)
    assert "|f| != 0 support is [0, 4)" in text
    assert "required minimum number_bands_sigma=4" in text


def _has_state_guarded_support_call(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_metal_test = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "occupation_state"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )
        if not is_metal_test:
            continue
        for child in node.body:
            for candidate in ast.walk(child):
                if (isinstance(candidate, ast.Call)
                        and isinstance(candidate.func, ast.Name)
                        and candidate.func.id
                        == "assert_sigma_contains_occupation_support"):
                    return True
    return False


@pytest.mark.parametrize(
    "fn",
    [compute_ppm_sigma_pipeline, compute_sigma_c_ppm_omega_grid],
)
def test_both_ppm_seams_guard_only_a_metal_state(fn):
    """AST proof: both pre-existing bracket seams dominate the new call."""
    assert _has_state_guarded_support_call(fn)


def test_an_unknown_override_is_a_refusal_not_a_silent_default(monkeypatch):
    monkeypatch.setenv(w_isdf.OCCUPATION_SUPPORT_ENV, "please")
    with pytest.raises(
        w_isdf.OccupationSupportTruncationError,
        match="Unrecognised LORRAX_OCCUPATION_SUPPORT",
    ):
        w_isdf.assert_sigma_contains_occupation_support(
            _ENK,
            np.asarray([[1.0, 1.0, 0.4, 0.2, 0.0, 0.0]]),
            slice(0, 3),
        )
