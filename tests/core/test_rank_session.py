"""Real process regression for rank-private pytest paths and shared drivers."""
import os
from pathlib import Path
import subprocess
import sys


CHILD = r'''
import os
from pathlib import Path
import sys
import time
import subprocess
from types import SimpleNamespace
from core import rank_session as session

session.ROOT = Path(sys.argv[1])
rank = int(os.environ["SLURM_PROCID"])
# Delay the writer: followers must not return before preparation completes.
def prepare(source, target):
    time.sleep(0.2)
    target.mkdir(parents=True)
    (target / "ready").write_text("complete")
    return target

run = session.stage(Path("A"), prepare)
assert (run / "ready").read_text() == "complete"
paths = session.exchange(str(run))
assert len(set(paths)) == 1
result = session.completed(SimpleNamespace(returncode=0, stdout=f"rank{rank}", stderr=""))
assert result.stdout == "rank0"
try:
    session.completed(SimpleNamespace(returncode=int(rank == 2), stdout="red twin", stderr=""))
except AssertionError as exc:
    assert "driver rank 2 failed" in str(exc)
else:
    raise AssertionError("a peer failure did not propagate")
assert session.exchange(rank) == [0, 1, 2, 3]
def timed_child():
    if rank == 2:
        raise subprocess.TimeoutExpired(["fixture"], 1, output=b"partial output")
    return SimpleNamespace(returncode=0, stdout="completed", stderr="")
try:
    session.run_child(timed_child, run / "timeout")
except AssertionError as exc:
    assert "driver rank 2 failed" in str(exc) and "partial output" in str(exc)
else:
    raise AssertionError("a peer timeout did not propagate")
assert (run / f"timeout.rank{rank}.log").is_file()
'''


def test_four_pytests_share_staging_and_propagate_peer_failure(tmp_path):
    """Four OS processes, delayed preparation, repeated rendezvous, red twin."""
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, SLURM_JOB_ID="unit", SLURM_STEP_ID="unit",
               SLURM_NTASKS="4", JAX_PROCESS_COUNT="4")
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("JAX_PROCESS_INDEX", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "tests"), str(root / "src"), env.get("PYTHONPATH", "")])
    children = []
    try:
        for rank in range(4):
            children.append(subprocess.Popen(
                [sys.executable, "-c", CHILD, str(tmp_path / "shared")],
                env=dict(env, SLURM_PROCID=str(rank)),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        outputs = [child.communicate(timeout=30) for child in children]
        for child, (stdout, stderr) in zip(children, outputs):
            assert child.returncode == 0, stdout + stderr
        assert len(list((tmp_path / "shared").glob("A-*"))) == 1
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
