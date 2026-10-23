"""phdf5 gate: SlabIO.read_multi against read_slab, and the posted control channel (handler ABI 7).

Run on every rank of a launch (``srun -n P``), any leg:
    python tests/phdf5_read_multi_gate.py OUT_DIR [--platform gpu|cpu] [--bench N]
Rank 0 writes the fixture with h5py; every rank then opens it with SlabIO and
checks, per addressable shard, that read_multi returns exactly read_slab's
bytes and the numpy reference (zero past ``valid_shape``): 4-D complex128 line
panels of ragged widths in batch and face layouts, mixed dtypes in one call, a
padded extent, a zero-size request, and the shared-dataset case (one dataset
twice). It also counts, per call, the descriptor puts and the Python
``block_until_ready`` calls of both routes, and times ``--bench`` rounds of a
35-panel read both ways. Posted control (ABI 7): queued same-shape writes to
different offsets and back-to-back async reads of one geometry must land where
posted (a keying or ordering fault swaps rows), no descriptor reaches the
device, and no control stays pending. Prints ``GATE <case> PASS|FAIL``; exit 1
on any FAIL.
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
        self.puts = self.syncs = self.posts = 0

    def __enter__(self):
        from jax._src import array as _arr
        self._arr = _arr
        self._put, self._bur = sff.device_put_process_local, _arr.ArrayImpl.block_until_ready
        self._post = sff._posted
        cnt = self

        def post(*a, **k):
            cnt.posts += 1
            return self._post(*a, **k)

        def put(*a, **k):
            cnt.puts += 1
            return self._put(*a, **k)

        def bur(x):
            cnt.syncs += 1
            return self._bur(x)
        sff.device_put_process_local = put
        sff._posted = post
        self._arr.ArrayImpl.block_until_ready = bur
        return self

    def __exit__(self, *exc):
        sff.device_put_process_local = self._put
        sff._posted = self._post
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
        ok = ok and cm.puts == 0 and cs.puts == 0
        verdict(case, ok, f"n={len(reqs)} multi(posts={cm.posts}, puts={cm.puts}, syncs={cm.syncs}) "
                          f"read_slab(posts={cs.posts}, puts={cs.puts}, syncs={cs.syncs})")
    # Disjoint windows of the SAME dataset still need distinct native calls.
    # An exact duplicate aliases the first result; another dataset can share a wave.
    reqs = (requests(FACE, [lines[6]], 6, 1) + requests(FACE, [lines[6]], 0, 1)
            + requests(FACE, [lines[5]], 2, 1) + requests(FACE, [lines[6]], 6, 1))
    print(f"GATE disjoint_same_dataset START rank={rank}", flush=True)
    with Count() as cm:
        multi = io.read_multi([{k: v for k, v in r.items() if k != "k"} for r in reqs],
                              partition_spec=FACE)
    single = [io.read_slab(r["name"], shape=r["shape"], offset=r["offset"],
                           valid_shape=r["valid_shape"], dtype=r["dtype"],
                           partition_spec=FACE) for r in reqs]
    verdict("disjoint_same_dataset", cm.posts == 2 and cm.puts == 0
            and multi[0] is multi[3] and all(same(m, s) and matches_ref(m, expected(r))
                for m, s, r in zip(multi, single, reqs)), f"posts={cm.posts} puts={cm.puts}")
    # Overlapping windows of one dataset: distinct waves, same bytes as read_slab.
    reqs = requests(FACE, [lines[6]], 1, 3) + requests(FACE, [lines[6]], 2, 3)
    with Count() as cm:
        multi = io.read_multi([{k: v for k, v in r.items() if k != "k"} for r in reqs],
                              partition_spec=FACE)
    verdict("overlap_waves", cm.posts == 2 and cm.puts == 0 and all(
        matches_ref(m, expected(r)) for m, r in zip(multi, reqs)), f"posts={cm.posts} puts={cm.puts}")
    # back-to-back async reads of ONE geometry at different offsets, no wait between
    reqs_by_q = [requests(BATCH, [lines[7]], q, ndev)[0] for q in (0, NQ - ndev) * 8]
    vals = [io.read_multi([{k: v for k, v in r.items() if k != "k"}], partition_spec=BATCH)[0]
            for r in reqs_by_q]
    verdict("posted_reads_back_to_back", all(matches_ref(v, expected(r))
                                             for v, r in zip(vals, reqs_by_q)), f"n={len(vals)}")
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
# posted writes: 8 same-shape slabs to different offsets, queued with no sync between
WPATH = os.path.join(OUT, f"posted_write_gate_P{ndev}.h5")
wshape = (NQ, FIELDS, ROWS, 6)
full = ref(200, wshape, np.complex128)
row_spec = P(("x", "y"), None, None, None)
with SlabIO(WPATH, mode="w", mesh=mesh) as wio:
    wio.create_dataset("w", shape=wshape, dtype=np.complex128)
    with Count() as cw:
        for q0 in range(0, NQ, ndev):
            blk = full[q0:q0 + ndev]
            A = jax.make_array_from_callback(blk.shape, jax.sharding.NamedSharding(mesh, row_spec),
                                             lambda idx, blk=blk: blk[idx])
            wio.write_slab("w", A, offset=(q0, 0, 0, 0))
        for k in range(1, 4):         # the same array to three offsets
            wio.write_slab("w", A, offset=(NQ - ndev, 0, 0, 0))
with SlabIO(WPATH, mode="r", mesh=mesh) as rio:
    back = rio.read_slab("w", shape=wshape, offset=(0, 0, 0, 0), dtype=np.complex128,
                         partition_spec=P(None, None, None, None))
verdict("posted_writes_queued", np.array_equal(np.asarray(back.addressable_shards[0].data), full)
        and cw.puts == 0, f"writes={NQ // ndev + 3} puts={cw.puts}")
# ordering fence (ABI 7): reads and a write issued behind ~100 ms of queued
# device work.  Without the fence the H2D lands before the destination's fill
# (the fill then zeroes it) and the write D2H reads its operand before it exists.
heavy_n = 3072 if jax.devices()[0].platform != "cpu" else 384
from common.collectives import device_put_process_local
hx = device_put_process_local(np.full((heavy_n, heavy_n), 1.0 / heavy_n),
                              jax.sharding.NamedSharding(mesh, P()))


@jax.jit
def _busy(a):
    for _ in range(30):
        a = jnp.tanh(a @ a)          # bounded: the busy values stay finite
    return a


with SlabIO(PATH, mode="r", mesh=mesh) as io:
    rq = requests(BATCH, [lines[5]], 0, ndev)[0]
    jax.block_until_ready(_busy(hx))
    busy = _busy(hx)                      # queued, not waited
    one = io.read_slab(rq["name"], shape=rq["shape"], offset=rq["offset"],
                       valid_shape=rq["valid_shape"], dtype=rq["dtype"], partition_spec=BATCH)
    busy = _busy(hx)
    many = io.read_multi([{k: v for k, v in r.items() if k != "k"}
                          for r in requests(BATCH, lines[:8], 0, ndev)], partition_spec=BATCH)
    ok = matches_ref(one, expected(rq)) and all(
        matches_ref(m, expected(r)) for m, r in zip(many, requests(BATCH, lines[:8], 0, ndev)))
    jax.block_until_ready(busy)
verdict("posted_reads_behind_queue", ok)
# read_slabs (k-chunk union reader, asynchronous worker) on posted control behind the
# queue: two disjoint field windows of real_f64, the window axis before the fields.
WIN = [(0, 2), (2, 2)]
ref64 = ref(100, (NQ, FIELDS, ROWS, 4), np.float64)
# Expected exactly: ONE post, of kind read_kchunk_union (through ffi.io, which
# posts for the union reader; SlabIO's _posted carries none), no placement, and
# no control left pending once the read has landed.
from ffi.common import ffi_loader as _fl
from ffi.io import _platform_for_mesh as _pfm
union_posts = []
_post0 = _fl.phdf5_post_control


def _counted_post(address, kind, values, *, platform):
    union_posts.append(kind)
    return _post0(address, kind, values, platform=platform)


with SlabIO(PATH, mode="r", mesh=mesh) as io:
    jax.block_until_ready(_busy(hx))
    busy = _busy(hx)
    _fl.phdf5_post_control = _counted_post
    try:
        with Count() as c_us:
            u = io.read_slabs("real_f64", shape=(ndev, 2, ROWS, 4),
                              offsets=[(0, f0, 0, 0) for f0, _ in WIN],
                              valid_shapes=[(min(ndev, NQ), nf, ROWS, 4) for _, nf in WIN],
                              partition_spec=BATCH, window_axis=1)
    finally:
        _fl.phdf5_post_control = _post0
    want = np.stack([ref64[:ndev, f0:f0 + nf] for f0, nf in WIN], axis=1)
    okU = matches_ref(u, want)
    jax.block_until_ready((u, busy))
    pend_u = _fl.phdf5_pending_controls(platform=_pfm(mesh))
verdict("read_slabs_posted_behind_queue",
        okU and union_posts == ["read_kchunk_union"] and c_us.posts == 0
        and pend_u == 0 and c_us.puts == 0,
        f"posts={union_posts} slabio_posts={c_us.posts} pending={pend_u} puts={c_us.puts}")
RPATH = os.path.join(OUT, f"posted_race_gate_P{ndev}.h5")
blk = full[:ndev]
B0 = jax.make_array_from_callback(blk.shape, jax.sharding.NamedSharding(mesh, row_spec),
                                  lambda idx: blk[idx])
slow = jax.jit(lambda b, a: b + 0.0 * jnp.sum(_busy(a)).astype(b.dtype))
with SlabIO(RPATH, mode="w", mesh=mesh) as wio:
    wio.create_dataset("w", shape=blk.shape, dtype=np.complex128)
    jax.block_until_ready(B0)
    wio.write_slab("w", slow(B0, hx), offset=(0, 0, 0, 0))   # operand still being produced
with SlabIO(RPATH, mode="r", mesh=mesh) as rio:
    back = rio.read_slab("w", shape=blk.shape, offset=(0, 0, 0, 0), dtype=np.complex128,
                         partition_spec=P(None, None, None, None))
verdict("posted_write_behind_queue", np.array_equal(np.asarray(back.addressable_shards[0].data), blk))
# File-order pieces (writes) and file rows (reads) with slice starts computed in the
# piece programs from a device-resident counter: no placement per piece.  A small
# piece budget forces the plans on small arrays: case A cuts lead-axis pieces whose
# last one is clamped back inside the extent; case B cuts several row pieces, the
# last one short.  P=1 never cuts pieces (one rank's write is one run).
from jax.sharding import NamedSharding as _NS


def _file_order_case(tag, shape, budget, min_run):
    spec = P(None, None, None, tuple(mesh.axis_names))
    full = ref(7, shape, np.complex128)
    A = jax.make_array_from_callback(shape, _NS(mesh, spec), lambda idx: full[idx])
    path = os.path.join(OUT, f"file_order_{tag}_P{ndev}.h5")
    saved = sff._file_order_plan.__defaults__
    orig = (sff.device_put_process_local, sff.device_put_process_local_cached)
    puts = [0]

    def counted(f):
        def g(*a, **k):
            puts[0] += 1
            return f(*a, **k)
        return g
    sff._file_order_plan.__defaults__ = (budget, min_run)
    sff.device_put_process_local, sff.device_put_process_local_cached = map(counted, orig)
    try:
        plan = sff._file_order_plan(shape, shape, sff._spec_axes(spec, len(shape)), 16, ndev,
                                    tuple(mesh.axis_names), shape)
        with SlabIO(path, mode="w", mesh=mesh) as w:
            w.create_dataset(tag, shape=shape, dtype=np.complex128)
            w.write_slab(tag, A, offset=(0,) * len(shape))
        with SlabIO(path, mode="r", mesh=mesh) as r:
            back = r.read_slab(tag, shape=shape, offset=(0,) * len(shape),
                               dtype=np.complex128, partition_spec=spec)
            ok_back = matches_ref(back, full)
    finally:
        sff._file_order_plan.__defaults__ = saved
        sff.device_put_process_local, sff.device_put_process_local_cached = orig
    barrier(f"read_multi_gate.file_order_{tag}")
    ok_raw = True
    if rank == 0:
        import h5py
        with h5py.File(path, "r") as f:
            ok_raw = bool(np.array_equal(f[tag][()], full))
    if ndev == 1:
        verdict(f"file_order_{tag}", plan is None and ok_back and ok_raw, "P=1: no pieces")
    else:
        verdict(f"file_order_{tag}", isinstance(plan, tuple) and ok_back and ok_raw and puts[0] == 0,
                f"plan={plan} back={ok_back} raw={ok_raw} puts={puts[0]}")


_file_order_case("lead_clamped", (5, 3, 8, 48), 3584, 512)      # steps (1, 2): 2 -> 1
_file_order_case("rows_short", (5, 3, 64, 48), 8192, 512)       # rows 40: 0, 40 (24 real)
# write_slab(require_finite=True): the writer checks the bytes it holds.  A rank whose
# block has a NaN writes nothing, still enters the collective, and every rank raises
# at close; the finite dataset round-trips and no NaN reaches the file.
FPATH = os.path.join(OUT, f"finite_gate_P{ndev}.h5")
bad_blk = blk.copy()
bad_blk.flat[0] = np.nan
_rows = jax.sharding.NamedSharding(mesh, row_spec)
GOOD = jax.make_array_from_callback(blk.shape, _rows, lambda idx: blk[idx])
BAD = jax.make_array_from_callback(blk.shape, _rows, lambda idx: bad_blk[idx])
with SlabIO(FPATH, mode="w", mesh=mesh) as fio:
    fio.create_dataset("good", shape=blk.shape, dtype=np.complex128)
    fio.create_dataset("bad", shape=blk.shape, dtype=np.complex128)
    fio.write_slab("good", GOOD, offset=(0, 0, 0, 0), require_finite=True)
raised = None
try:
    with SlabIO(FPATH, mode="a", mesh=mesh) as fio:
        fio.write_slab("bad", BAD, offset=(0, 0, 0, 0), require_finite=True)
except Exception as exc:                                  # noqa: BLE001
    raised = f"{type(exc).__name__}: {exc}"
verdict("write_require_finite_refuses", raised is not None and "slab_io_write_finite" in raised,
        (raised or "no error")[:200].replace("\n", " "))
# The refused transaction leaves the file uncommitted: SlabIO will not reopen it.
reopened = None
try:
    with SlabIO(FPATH, mode="r", mesh=mesh) as rio:
        pass
except Exception as exc:                                  # noqa: BLE001
    reopened = str(exc)
verdict("write_require_finite_unpublished",
        reopened is not None and "not globally committed" in reopened, (reopened or "reopened")[:120])
# The bytes themselves (raw h5py, rank 0 after every rank's teardown): the finite
# dataset landed whole, and the refusing rank's block holds no NaN.
barrier("read_multi_gate.finite_raw")
if rank == 0:
    import h5py
    with h5py.File(FPATH, "r") as f:
        g, b = f["good"][()], f["bad"][()]
    verdict("write_require_finite_passes_finite", np.array_equal(g, blk))
    verdict("write_require_finite_no_nan_on_disk", bool(np.isfinite(b).all()))
barrier("read_multi_gate.finite_raw_done")
from ffi.common import ffi_loader
from ffi.io import _platform_for_mesh
pending = ffi_loader.phdf5_pending_controls(platform=_platform_for_mesh(mesh))
verdict("no_pending_controls", pending == 0, f"pending={pending}")
print(f"GATE_DONE rank={rank} fails={len(fails)}", flush=True)
sys.exit(1 if fails else 0)
