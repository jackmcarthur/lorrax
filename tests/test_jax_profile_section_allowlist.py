"""A profiler session must be openable section by section, not all-or-nothing.

RED TWIN for the sigma-scaling lane's multi-node tracing block (2026-08-09).

``ISDF_JAX_PROFILE_DIR`` used to be a single switch: set it, and every
``jax_profile.trace_section`` call site in the run opened a profiler session.
That is fine on one node and it is fatal on several.  Tracing the production
GN-PPM Sigma deck at a 3x3 mesh over three nodes segfaults rank 0 inside the
phdf5 collective close of ``zeta_fit`` -- reproduced three times, always at the
same place, always ~20 s in, and never on a 2x2 single-node mesh.  The section
that crashes is not the section anyone wanted; the sigma tau kernel is nine
sections later.  With one switch there was no way to ask for the second and
not the first, so the tau kernel simply could not be traced above P=4.

``ISDF_JAX_PROFILE_SECTIONS`` is that ask: a comma-separated allowlist of
substrings.  Unset, every section traces, which is what every existing caller
already relies on.  Set, only the named sections open a session.

The load-bearing cell is ``test_a_named_section_excludes_the_others``: on the
pre-fix tree ``_trace_path`` consults only the directory variable, so it hands
back a path for ``zeta_fit`` no matter what the allowlist says, and that cell
fails.  The two cells around it pin the default (trace everything) and the
disabled state (trace nothing), because an allowlist that quietly changed
either of those would be a worse bug than the one it fixes.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _load_jax_profile():
    """Import common.jax_profile without dragging the package in."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    spec = importlib.util.find_spec("common.jax_profile")
    if spec is None:  # pragma: no cover - the module is in-tree
        pytest.skip("common.jax_profile not importable in this environment")
    import common.jax_profile as jp
    return jp


def test_unset_traces_every_section(monkeypatch, tmp_path):
    """The default is unchanged: no allowlist means every section traces."""
    jp = _load_jax_profile()
    monkeypatch.setenv("ISDF_JAX_PROFILE_DIR", str(tmp_path))
    monkeypatch.delenv("ISDF_JAX_PROFILE_SECTIONS", raising=False)
    for section in ("zeta_fit", "chi0_W", "sigma_tau___E_F_cond", "V_q_compute"):
        assert jp._section_selected(section) is True


def test_a_named_section_excludes_the_others(monkeypatch, tmp_path):
    """THE RED: naming sigma_tau must not open a session at zeta_fit.

    This is the cell that fails on the pre-fix tree, where ``_trace_path``
    reads only the directory variable.
    """
    jp = _load_jax_profile()
    monkeypatch.setenv("ISDF_JAX_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("ISDF_JAX_PROFILE_SECTIONS", "sigma_tau")
    assert jp._trace_path("sigma_tau___E_F_cond") is not None
    assert jp._trace_path("zeta_fit") is None
    assert jp._trace_path("chi0_W") is None
    assert jp._trace_path("V_q_compute") is None


def test_several_sections_and_whitespace(monkeypatch, tmp_path):
    """The list is comma-separated, substring-matched, whitespace-tolerant."""
    jp = _load_jax_profile()
    monkeypatch.setenv("ISDF_JAX_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("ISDF_JAX_PROFILE_SECTIONS", " sigma_tau , chi0_W ")
    assert jp._trace_path("sigma_tau___E_F_val") is not None
    assert jp._trace_path("chi0_W_probe") is not None
    assert jp._trace_path("zeta_fit") is None


def test_an_empty_allowlist_is_not_an_empty_selection(monkeypatch, tmp_path):
    """An empty or whitespace value means "unset", not "trace nothing".

    A shell that exports the variable with no value must not silently turn
    every trace off -- that failure mode is invisible (a run that produces no
    artifacts and no error), which is exactly the class of defect the P21
    landing removed from this same file.
    """
    jp = _load_jax_profile()
    monkeypatch.setenv("ISDF_JAX_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("ISDF_JAX_PROFILE_SECTIONS", "   ")
    assert jp._trace_path("zeta_fit") is not None


def test_no_directory_still_means_no_trace(monkeypatch):
    """The allowlist does not enable tracing on its own."""
    jp = _load_jax_profile()
    monkeypatch.delenv("ISDF_JAX_PROFILE_DIR", raising=False)
    monkeypatch.setenv("ISDF_JAX_PROFILE_SECTIONS", "sigma_tau")
    assert jp._trace_path("sigma_tau___E_F_cond") is None


def test_trace_section_is_a_no_op_for_an_excluded_section(monkeypatch, tmp_path):
    """The context manager still runs its body, and writes no directory."""
    jp = _load_jax_profile()
    monkeypatch.setenv("ISDF_JAX_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("ISDF_JAX_PROFILE_SECTIONS", "sigma_tau")
    ran = []
    with jp.trace_section("zeta_fit"):
        ran.append(True)
    assert ran == [True]
    assert not any(p.name.startswith("zeta_fit") for p in tmp_path.iterdir())
