"""Launcher decisions shared with the deployed ``lx`` front door."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile

try:
    from _lxkit_harness import raises, run_module
except ModuleNotFoundError:
    from ._lxkit_harness import raises, run_module

from lxkit.launcher_policy import (
    LauncherPolicyError,
    apply_cache_policy,
    export_allocation_pin,
    newest_first,
    resolve_allocation_pin,
    runtime_origin,
    select_source_root,
    square_mesh,
    validate_geometry,
)


def test_jobid_has_one_happy_path_and_legacy_compatibility():
    assert resolve_allocation_pin("57920327", {}).source == "--jid"
    assert resolve_allocation_pin(None, {"SLURM_JOBID": "57920327"}).jid \
        == "57920327"
    assert resolve_allocation_pin(None, {"SLURM_JOB_ID": "57920327"}).jid \
        == "57920327"
    assert resolve_allocation_pin(None, {}).jid is None


def test_jobid_conflicts_and_bad_values_refuse():
    with raises(LauncherPolicyError, match="LX-JID-CONFLICT"):
        resolve_allocation_pin(None, {
            "SLURM_JOBID": "57920327", "SLURM_JOB_ID": "57924704"})
    with raises(LauncherPolicyError, match="LX-JID-CONFLICT"):
        resolve_allocation_pin("57920327", {"SLURM_JOBID": "57924704"})
    with raises(LauncherPolicyError, match="LX-BADJID"):
        resolve_allocation_pin("latest", {})


def test_internal_jobid_export_feeds_both_old_helpers():
    env = {}
    export_allocation_pin(env, "57920327")
    assert env == {"SLURM_JOBID": "57920327", "SLURM_JOB_ID": "57920327"}


def test_cache_default_is_cold_and_explicit_values_win():
    env = {}
    assert apply_cache_policy(env) == "default cold"
    assert env["ISDF_JAX_CACHE_DIR"] == ""

    env = {"ISDF_JAX_CACHE_DIR": "/run/warm-cache"}
    assert apply_cache_policy(env) == "explicit warm"
    assert env["ISDF_JAX_CACHE_DIR"] == "/run/warm-cache"

    env = {"ISDF_JAX_CACHE_DIR": ""}
    assert apply_cache_policy(env) == "explicit cold"


def test_gpu_count_is_per_node_and_p16_geometry_is_valid():
    geometry = validate_geometry(4, 4, 16)
    assert geometry.total_gpus == 16
    assert square_mesh(geometry.ranks) == (4, 4)

    with raises(LauncherPolicyError, match="-G is per node"):
        validate_geometry(4, 16, 16)


def test_non_square_mesh_refuses_by_name():
    with raises(LauncherPolicyError, match="LX-NONSQUARE-MESH"):
        square_mesh(8)


@dataclass(frozen=True)
class _Allocation:
    jobid: str


def test_newest_allocation_is_first():
    rows = [_Allocation("57920327"), _Allocation("57924704")]
    assert [row.jobid for row in newest_first(rows)] == ["57924704", "57920327"]


def _make_checkout(root: Path) -> None:
    (root / "src" / "runtime").mkdir(parents=True)
    (root / "src" / "gw").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='lorrax'\n")
    (root / "src" / "runtime" / "__init__.py").write_text("")
    (root / "src" / "gw" / "__init__.py").write_text("")


def test_source_selection_prefers_cwd_checkout_and_keeps_old_pin():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cwd_tree = tmp_path / "cwd_tree"
        site_tree = tmp_path / "site_tree"
        old_tree = tmp_path / "old_tree"
        for root in (cwd_tree, site_tree, old_tree):
            _make_checkout(root)
        run_dir = cwd_tree / "nested" / "run"
        run_dir.mkdir(parents=True)

        root, reason = select_source_root(run_dir, site_tree)
        assert root == cwd_tree.resolve()
        assert reason == "cwd checkout"
        assert runtime_origin(root) == (
            cwd_tree / "src" / "runtime" / "__init__.py").resolve()

        root, reason = select_source_root(run_dir, site_tree, old_tree)
        assert root == old_tree.resolve()
        assert reason == "LORRAX_CHECKOUT compatibility"


if __name__ == "__main__":
    sys.exit(run_module(globals()))
