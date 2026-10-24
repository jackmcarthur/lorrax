"""scripts/runtime_image.sh: pack once, restore once per node at the same path.

Checks the properties a run depends on: four concurrent restores on one node
extract once; every file comes back at the recorded absolute path with its
nanosecond mtime (the closure gate's no-rehash fast path compares mtime_ns);
an image a process still holds is never cleaned, an unheld one is.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runtime_image.sh"

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("flock") and shutil.which("tar")),
    reason="needs bash, flock and GNU tar")


def _run(*args, env=None, check=True):
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], env=env,
                          capture_output=True, text=True, check=check)


def _make_tree(root: Path) -> dict[str, int]:
    (root / "lorrax" / "lib").mkdir(parents=True)
    mt = {}
    for i in range(20):
        p = root / "lorrax" / "lib" / f"f{i}.so"
        p.write_bytes(os.urandom(1000 + i))
        os.utime(p, ns=(1_792_812_330_754_568_034 + i, 1_792_812_330_754_568_034 + i))
        mt[str(p)] = p.stat().st_mtime_ns
    return mt


def test_pack_restore_concurrent_and_clean(tmp_path):
    tree = tmp_path / "node" / "rt-a"
    mtimes = _make_tree(tree)
    archive = tmp_path / "store" / "rt-a.tar"
    archive.parent.mkdir()
    _run("pack", tree, archive)
    assert archive.is_file() and Path(f"{archive}.meta").is_file()
    shutil.rmtree(tree)

    env = dict(os.environ)
    procs = [subprocess.Popen(["bash", str(SCRIPT), "restore", str(archive)],
                              env={**env, "SLURM_LOCALID": str(r)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for r in range(4)]
    outs = [p.communicate() for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    assert all(o[0].strip() == str(tree) for o in outs)
    assert sum("restored" in o[1] for o in outs) <= 1          # one extraction (rank 0 reports)
    assert not list(tree.parent.glob(".rt-a.tmp.*"))
    for path, ns in mtimes.items():
        assert Path(path).stat().st_mtime_ns == ns, path

    # A second image of the same user on the node: held -> kept, unheld -> cleaned.
    other = tmp_path / "node" / "rt-b"
    _make_tree(other)
    (other / ".lorrax-image").write_text("b\n")
    foreign = tmp_path / "node" / "venv-x"                      # another tool's lock + dir
    foreign.mkdir(); (tmp_path / "node" / ".venv-x.lock").touch()
    lock = tmp_path / "node" / ".rt-b.use.lock"
    with open(lock, "a") as held:                               # a running process's shared lock
        fcntl.flock(held, fcntl.LOCK_SH)
        _run("clean", archive)
        assert other.is_dir()
    _run("clean", archive)
    assert not other.exists() and tree.is_dir() and foreign.is_dir()


def test_concurrent_runs_while_ranks_hold_the_image(tmp_path):
    """Four ranks `run` at once and keep running: a rank that restores must not
    wait for the others to exit (each holds its use lock for its lifetime)."""
    tree = tmp_path / "node" / "rt-d"
    _make_tree(tree)
    sel = tree / "lorrax" / "src" / "ffi" / "cpp" / "select_gpu.sh"
    sel.parent.mkdir(parents=True)
    sel.write_text('#!/usr/bin/env bash\nexec "$@"\n'); sel.chmod(0o755)
    archive = tmp_path / "rt-d.tar"
    _run("pack", tree, archive)
    shutil.rmtree(tree)
    procs = [subprocess.Popen(["bash", str(SCRIPT), "run", str(archive), "--", "sleep", "3"],
                              env={**os.environ, "SLURM_LOCALID": str(r), "SLURM_PROCID": str(r)},
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for r in range(4)]
    try:
        outs = [p.communicate(timeout=30) for p in procs]
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        pytest.fail("a rank waited on the image lock held by running ranks")
    assert all(p.returncode == 0 for p in procs), outs
    assert sum("restored" in o[1] for o in outs) == 1


def test_reused_release_id_refuses(tmp_path):
    tree = tmp_path / "node" / "rt-c"
    _make_tree(tree)
    archive = tmp_path / "rt-c.tar"
    _run("pack", tree, archive)
    assert _run("pack", tree, archive, check=False).returncode != 0   # never repacked
    (tree / ".lorrax-image").write_text("some-other-id\n")
    r = _run("restore", archive, check=False)
    assert r.returncode != 0 and "release id" in r.stderr
