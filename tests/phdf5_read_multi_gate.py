"""phdf5 ``PhdfReadMulti`` gate: SlabIO.read_multi against read_slab (the reference route).

Run on every rank of a launch (``srun -n P``), any leg:
    python tests/phdf5_read_multi_gate.py OUT_DIR [--platform gpu|cpu] [--bench N]
Rank 0 writes the fixture with h5py; every rank then opens it with SlabIO and
checks, per addressable shard, that read_multi returns exactly read_slab's
bytes and the numpy reference (zero past ``valid_shape``): 4-D complex128 line
panels of ragged widths in batch and face layouts, mixed dtypes in one call, a
padded extent, a zero-size request, and the shared-dataset case (one dataset
twice). It also counts, per call, the descriptor puts and the Python
``block_until_ready`` calls of both routes, and times ``--bench`` rounds of a
35-panel read both ways. Prints ``GATE <case> PASS|FAIL``; exit 1 on any FAIL.
"""
import os
import sys
import time

import runtime

_PLATFORM = sys.argv[sys.argv.index("--platform") + 1] if "--platform" in sys.argv else "gpu"
runtime.bootstrap(platform=_PLATFORM)

import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402
import numpy as np                           # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402

from common.collectives import barrier       # noqa: E402
from file_io import _slab_io_ffi as sff      # noqa: E402
from file_io.slab_io import SlabIO           # noqa: E402

OUT = sys.argv[1]
BENCH = int(sys.argv[sys.argv.index("--bench") + 1]) if "--bench" in sys.argv else 0
rank, nproc = jax.process_index(), jax.process_count()
devs = np.asarray(jax.devices())
px = 2 if devs.size % 2 == 0 and devs.size > 1 else 1
mesh = Mesh(devs.reshape(px, devs.size // px), ("x", "y"))
PATH = os.path.join(OUT, f"read_multi_gate_P{devs.size}.h5")
NQ, FIELDS, ROWS = 8, 5, 13
WIDTHS = [3, 7, 1, 6, 2, 5, 4, 8] + [3 + (k % 5) for k in range(27)]   # 35 panels


def ref(k, shape, dtype):
    n = int(np.prod(shape))
    x = (np.arange(n, dtype=np.float64) + 1000.0 * k).reshape(shape)
    if np.issubdtype(dtype, np.complexfloating):
        return (x + 1j * (0.5 - x)).astype(dtype)
    return x.astype(dtype)


def fixture():
    import h5py
    os.makedirs(OUT, exist_ok=True)
    tmp = PATH + f".{os.getpid()}.tmp"
    with h5py.File(tmp, "w") as f:
        for k, w in enumerate(WIDTHS):
            f[f"line_{k:03d}"] = ref(k, (NQ, FIELDS, ROWS, w), np.complex128)
        f["real_f64"] = ref(100, (NQ, FIELDS, ROWS, 4), np.float64)
        f["index_i64"] = ref(101, (NQ, 2, 3, 4), np.int64)
        f["empty"] = np.zeros((NQ, FIELDS, ROWS, 0), np.complex128)
    os.replace(tmp, PATH)


if rank == 0:
    fixture()
barrier("read_multi_gate.fixture")

ndev = devs.size
BATCH = P(("x", "y"), None, None, None)
FACE = P(None, None, "x", "y")


def pad(n, d):
    return -(-n // d) * d


def requests(spec, names_widths, q0, count):
    out = []
    for k, name, w, dtype, fields, rows in names_widths:
        if spec == BATCH:
            shape = (count, fields, rows, w)
        else:
            shape = (count, fields, pad(rows, px), pad(w, ndev // px))
        out.append(dict(name=name, shape=shape, offset=(q0, 0, 0, 0),
                        valid_shape=(count, fields, rows, w), dtype=dtype, k=k))
    return out


def expected(req):
    k, (count, fields, rows, w) = req["k"], req["valid_shape"]
    full = ref(k, (NQ, fields, rows, w), np.dtype(req["dtype"]))
    q0 = req["offset"][0]
    e = np.zeros(req["shape"], np.dtype(req["dtype"]))
    e[:count, :fields, :rows, :w] = full[q0:q0 + count]
    return e


def same(a, b):
    return all(np.array_equal(np.asarray(x.data), np.asarray(y.data))
               for x, y in zip(a.addressable_shards, b.addressable_shards))


def matches_ref(a, e):
    return all(np.array_equal(np.asarray(s.data), e[s.index]) for s in a.addressable_shards)


class Count:
    def __init__(self):
        self.puts = self.syncs = 0

    def __enter__(self):
        from jax._src import array as _arr
        self._arr = _arr
        self._put, self._bur = sff.device_put_process_local, _arr.ArrayImpl.block_until_ready
        cnt = self

        def put(*a, **k):
            cnt.puts += 1
            return self._put(*a, **k)

        def bur(x):
            cnt.syncs += 1
            return self._bur(x)
        sff.device_put_process_local = put
        self._arr.ArrayImpl.block_until_ready = bur
        return self

    def __exit__(self, *exc):
        sff.device_put_process_local = self._put
        self._arr.ArrayImpl.block_until_ready = self._bur


fails = []


def verdict(case, ok, detail=""):
    ok = bool(ok)
    if rank == 0 or not ok:
        print(f"GATE {case} {'PASS' if ok else 'FAIL'} rank={rank} {detail}", flush=True)
    if not ok:
        fails.append(case)


lines = [(k, f"line_{k:03d}", w, np.complex128, FIELDS, ROWS) for k, w in enumerate(WIDTHS)]
mixed = [lines[1], (100, "real_f64", 4, np.float64, FIELDS, ROWS),
         (101, "index_i64", 4, np.int64, 2, 3), lines[2]]
cases = [("line35_batch", BATCH, lines, 0, ndev),
         ("line35_batch_q4", BATCH, lines, NQ - ndev, ndev),
         ("line35_face", FACE, lines, 1, 3),
         ("mixed_dtype_face", FACE, mixed, 2, 5),
         ("mixed_dtype_batch", BATCH, mixed, 0, ndev),
         ("same_dataset_twice", BATCH, [lines[3], lines[3]], 0, ndev),
         ("repeated_parent_face", FACE, [lines[4], lines[5], lines[4]], 6, 1)]
with SlabIO(PATH, mode="r", mesh=mesh) as io:
    for case, spec, nw, q0, count in cases:
        reqs = requests(spec, nw, q0, count)
        with Count() as cm:
            multi = io.read_multi([{k: v for k, v in r.items() if k != "k"} for r in reqs],
                                  partition_spec=spec)
        with Count() as cs:
            single = [io.read_slab(r["name"], shape=r["shape"], offset=r["offset"],
                                   valid_shape=r["valid_shape"], dtype=r["dtype"],
                                   partition_spec=spec) for r in reqs]
        ok = all(m.shape == s.shape and m.dtype == s.dtype and m.sharding == s.sharding
                 and same(m, s) and matches_ref(m, expected(r))
                 for m, s, r in zip(multi, single, reqs))
        verdict(case, ok, f"n={len(reqs)} multi(puts={cm.puts}, syncs={cm.syncs}) "
                          f"read_slab(puts={cs.puts}, syncs={cs.syncs})")
    # overlapping windows of one dataset: two disjoint waves, same bytes as read_slab
    reqs = requests(FACE, [lines[6]], 1, 3) + requests(FACE, [lines[6]], 2, 3)
    with Count() as cm:
        multi = io.read_multi([{k: v for k, v in r.items() if k != "k"} for r in reqs],
                              partition_spec=FACE)
    verdict("overlap_waves", cm.puts == 2 and all(matches_ref(m, expected(r))
                                                  for m, r in zip(multi, reqs)),
            f"puts={cm.puts}")
    # zero-size request: zeros, no I/O, beside a real one
    reqs = requests(BATCH, [lines[0]], 0, ndev) + [dict(
        name="empty", shape=(ndev, FIELDS, ROWS, 0), offset=(0, 0, 0, 0),
        valid_shape=(ndev, FIELDS, ROWS, 0), dtype=np.complex128, k=0)]
    multi = io.read_multi([{k: v for k, v in r.items() if k != "k"} for r in reqs],
                          partition_spec=BATCH)
    verdict("zero_size", multi[1].shape == (ndev, FIELDS, ROWS, 0)
            and matches_ref(multi[0], expected(reqs[0])))
    # refusal: one call shares one rank
    try:
        io.read_multi([dict(name="line_000", shape=(ndev, FIELDS, ROWS, 3), offset=(0, 0, 0, 0)),
                       dict(name="line_001", shape=(ndev, FIELDS, ROWS * 7), offset=(0, 0, 0))],
                      partition_spec=BATCH)
        verdict("refuses_mixed_rank", False)
    except ValueError as exc:
        verdict("refuses_mixed_rank", "one call shares one" in str(exc))
    if BENCH:
        reqs = [{k: v for k, v in r.items() if k != "k"} for r in requests(BATCH, lines, 0, ndev)]
        for route in ("read_slab", "read_multi"):
            walls = []
            for it in range(BENCH + 1):
                t0 = time.perf_counter()
                if route == "read_multi":
                    vals = io.read_multi(reqs, partition_spec=BATCH)
                else:
                    vals = [io.read_slab(r["name"], shape=r["shape"], offset=r["offset"],
                                         valid_shape=r["valid_shape"], dtype=r["dtype"],
                                         partition_spec=BATCH) for r in reqs]
                jax.block_until_ready(vals)
                walls.append(time.perf_counter() - t0)
            w = np.asarray(walls[1:]) * 1e3
            if rank == 0:
                print(f"BENCH {route} n=35 P={ndev} warm rounds={BENCH} median_ms={np.median(w):.2f} "
                      f"min_ms={w.min():.2f} first_ms={walls[0]*1e3:.1f}", flush=True)
print(f"GATE_DONE rank={rank} fails={len(fails)}", flush=True)
sys.exit(1 if fails else 0)
