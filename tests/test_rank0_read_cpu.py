"""``rank0_read``: one process opens a committed file, every process gets it. CPU only."""
import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent("""
    import json, os, sys
    import h5py, numpy as np
    import jax
    rank, port, path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=3, process_id=rank)
    from common import collectives
    from common.collectives import rank0_read, rank0_transaction
    collectives._RANK0_READ_CHUNK = 1000          # force a multi-chunk payload
    opens = []
    real = h5py.File
    def counted(*a, **k):
        opens.append(a[0]); return real(*a, **k)
    h5py.File = counted
    out = {}
    def read():
        with h5py.File(path, "r") as f:
            return dict(header=json.loads(bytes(f["header_json"][()])), shape=list(f["x"].shape))
    out["value"] = rank0_read(path, stage="t.read", read=read)
    def bad():
        raise ValueError("GATE shared_pole_store: refused on rank 0")
    try:
        rank0_read(path, stage="t.bad", read=bad)
    except ValueError as exc:
        out["error"] = str(exc)
    def nested():
        return rank0_read(path, stage="t.inner", read=lambda: [1, (2, 3)])
    out["nested"] = rank0_read(path, stage="t.outer", read=nested)
    box = {}
    rank0_transaction(path, stage="t.txn", write=lambda: box.update(
        inner=rank0_read(path, stage="t.in_write", read=lambda: "local")))
    out["in_write"] = box.get("inner")
    out["after"] = rank0_read(path, stage="t.after", read=lambda: rank)
    out["opens"] = len(opens)
    print("RESULT " + json.dumps(out), flush=True)
""")


def test_one_reader_shares_value_errors_and_nesting(tmp_path):
    import h5py
    import numpy as np
    path = tmp_path / "bank.h5"
    header = {"masks": ["x" * 40] * 200, "nan": float("nan")}   # ~9 kB > one 1000 B chunk
    with h5py.File(path, "w") as f:
        f["header_json"] = np.bytes_(json.dumps(header))
        f["x"] = np.zeros((4, 3))
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    procs = [subprocess.Popen([sys.executable, str(script), str(r), port, str(path)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
             for r in range(3)]
    outs = [p.communicate(timeout=240)[0] for p in procs]
    results = []
    for out in outs:
        line = [l for l in out.splitlines() if l.startswith("RESULT ")]
        assert line, out
        results.append(json.loads(line[0][7:]))
    for rank, got in enumerate(results):
        assert got["value"]["shape"] == [4, 3]
        assert got["value"]["header"]["masks"] == header["masks"]
        assert got["error"] == "GATE shared_pole_store: refused on rank 0"
        assert got["nested"] == [1, [2, 3]]
        assert got["after"] == 0                       # rank 0's value on every rank
        assert got["opens"] == (1 if rank == 0 else 0)  # only rank 0 opened the file
    assert results[0]["in_write"] == "local" and results[1]["in_write"] is None


def test_single_process_reads_directly():
    from common.collectives import rank0_read
    assert rank0_read("p", stage="s", read=lambda: (1, {"a": (2,)})) == [1, {"a": [2]}]
    with pytest.raises(KeyError):
        rank0_read("p", stage="s", read=lambda: {}["missing"])
