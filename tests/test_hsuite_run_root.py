"""The hsuite run root comes from the runtime's site-scratch rule, before any driver.

``tests.hsuite.rank_session.run_root`` must give the documented pytest command a
scratch directory without extra environment: a caller's ``SCRATCH`` wins, else
the site profile's ``[paths] scratch`` (``runtime.apply_site_scratch``), else a
refusal that names the remedy -- never a silent fall-back to HOME.  Plain Python.
"""
import os

import pytest

import runtime
from tests.hsuite import rank_session

_SITE_VARS = ("SCRATCH", "LORRAX_SITE_PROFILE", "NERSC_HOST", "LMOD_SYSTEM_NAME",
              "MEMBERWORK", "SLURM_JOB_ACCOUNT")


@pytest.fixture
def fresh(monkeypatch):
    for name in _SITE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "_SITE_SCRATCH_DONE", [])
    monkeypatch.setattr(rank_session, "_ROOT", None)
    return monkeypatch


def test_a_callers_scratch_wins(fresh, tmp_path):
    fresh.setenv("SCRATCH", str(tmp_path))
    fresh.setenv("LMOD_SYSTEM_NAME", "frontier")
    assert rank_session.run_root() == tmp_path / ".cache" / "lorrax" / "hsuite"
    assert rank_session.ROOT == rank_session.run_root()


def test_the_site_profile_fills_scratch_when_unset(fresh, tmp_path):
    (tmp_path / "mat280").mkdir()
    fresh.setenv("LMOD_SYSTEM_NAME", "frontier")
    fresh.setenv("MEMBERWORK", str(tmp_path))
    fresh.setenv("SLURM_JOB_ACCOUNT", "mat280")
    root = rank_session.run_root()
    assert root == tmp_path / "mat280" / ".cache" / "lorrax" / "hsuite"
    assert os.environ["SCRATCH"] == str(tmp_path / "mat280")      # the drivers see the same


def test_no_scratch_and_no_profile_refuses_naming_the_remedy(fresh):
    with pytest.raises(RuntimeError) as err:
        rank_session.run_root()
    text = str(err.value)
    assert "GATE hsuite_run_root" in text and "the site profile declares no scratch" in text
    assert "Export SCRATCH=" in text and "[paths] scratch" in text
    assert rank_session._ROOT is None                              # nothing was chosen


def test_a_profile_rule_that_does_not_expand_refuses(fresh, tmp_path):
    fresh.setenv("LMOD_SYSTEM_NAME", "frontier")                   # no job account: a login shell
    fresh.setenv("MEMBERWORK", str(tmp_path))
    with pytest.raises(RuntimeError, match="does not expand here"):
        rank_session.run_root()
