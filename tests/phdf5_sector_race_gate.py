"""CT-sector read race gate (lane slabmr): the _read_photon_bank_sector pattern behind a busy queue.

python tests/phdf5_sector_race_gate.py OUT [--trials N] [--mib M]   (every rank; any leg; uses only SlabIO.read_slab/read_slabs)
Per trial: queue ~100 ms of device work, then read `side` column unions of row windows (read_slabs, the
k-chunk union reader: asynchronous handler, H2D on the file's worker) from a fresh, large dataset and consume
them at once on the device (reshape + concat, as the sector reader does) and on the host. Counts trials whose
values differ from numpy, and all-zero results (the bisp_sc CT symptom). Control: the same rectangles by
synchronous read_slab. Prints SECTOR <case> wrong=<n>/<N> allzero=<n>/<N>.
"""
import os, sys, time
import runtime
runtime.bootstrap(platform="gpu")
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.collectives import barrier, device_put_process_local
from file_io.slab_io import SlabIO
OUT = sys.argv[1]; TRIALS = int(sys.argv[sys.argv.index("--trials") + 1]) if "--trials" in sys.argv else 20
MIB = int(sys.argv[sys.argv.index("--mib") + 1]) if "--mib" in sys.argv else 256
rank = jax.process_index(); devs = np.asarray(jax.devices()); ndev = devs.size
px = 2 if ndev % 2 == 0 and ndev > 1 else 1
mesh = Mesh(devs.reshape(px, ndev // px), ("x", "y"))
side = px                                   # photon layout: mesh_side
c, t = 6, 4; stride = c + 3 * t; d = side * stride
nsample = max(1, (MIB << 20) // (16 * d * d * 4))   # dataset ~MIB MiB for 4 parents
nq = 4
prefix = (nq, nsample)
FACE = P(None, None, "x", "y")
def ref(shape, k):
    n = int(np.prod(shape)); x = (np.arange(n, dtype=np.float64) + 1e3 * k + 1.0).reshape(shape)
    return (x + 1j * (0.25 * x + 1.0)).astype(np.complex128)
hx = device_put_process_local(np.full((3072, 3072), 1.0 / 3072), NamedSharding(mesh, P()))
@jax.jit
def busy(a):
    for _ in range(30):
        a = jnp.tanh(a @ a)
    return a
jax.block_until_ready(busy(hx))
concat = jax.jit(lambda *cols: jnp.concatenate(cols, axis=-1))
sector = ("C", "T")
widths = tuple(c if f == "C" else 3 * t for f in sector); starts = tuple(0 if f == "C" else c for f in sector)
res = {"union_device": [0, 0], "union_host": [0, 0], "sync_device": [0, 0]}
for trial in range(TRIALS):
    path = os.path.join(OUT, f"sector_{trial}_P{ndev}.h5")
    full = ref(prefix + (d, d), trial)
    if rank == 0:
        import h5py
        os.makedirs(OUT, exist_ok=True)
        tmp = path + ".tmp"
        with h5py.File(tmp, "w") as f:
            f["bank"] = full
        os.replace(tmp, path)
    barrier(f"sector_gate.fixture.{trial}")
    # numpy expectation of the sector rectangle (C rows x T columns), mesh-major as the reader builds it
    rows = np.concatenate([full[..., r * stride + starts[0]: r * stride + starts[0] + widths[0], :]
                           for r in range(side)], axis=-2)
    expect = np.concatenate([rows[..., col * stride + starts[1]: col * stride + starts[1] + widths[1]]
                             for col in range(side)], axis=-1)
    shape = prefix + widths
    with SlabIO(path, mode="r", mesh=mesh) as io:
        q = busy(hx)                                        # queued, not waited
        cols = []
        for col in range(side):
            offs = [(0, 0, r * stride + starts[0], col * stride + starts[1]) for r in range(side)]
            arr = io.read_slabs("bank", shape=shape, offsets=np.asarray(offs, np.int64),
                                valid_shapes=np.asarray([shape] * side, np.int64), partition_spec=FACE,
                                window_axis=len(prefix), dtype=np.complex128)
            cols.append(arr.reshape(prefix + (side * widths[0], widths[1])))
        got = concat(*cols)                                 # device consumer, no wait
        host0 = np.asarray(cols[0].addressable_shards[0].data)   # host consumer of the first column
        q2 = busy(hx)
        sync = [io.read_slab("bank", shape=prefix + (widths[0], widths[1]),
                             offset=(0, 0, starts[0], starts[1]), dtype=np.complex128,
                             partition_spec=FACE)]
        jax.block_until_ready((got, sync, q, q2))
    for key, arr, exp in (("union_device", got, expect),
                          ("union_host", None, None),
                          ("sync_device", sync[0], full[..., starts[0]:starts[0] + widths[0], starts[1]:starts[1] + widths[1]])):
        if key == "union_host":
            sh = cols[0].addressable_shards[0]
            e = expect[..., :, :widths[1]][sh.index]
            v = host0
        else:
            sh = arr.addressable_shards[0]; e = exp[sh.index]; v = np.asarray(sh.data)
        res[key][0] += int(not np.array_equal(v, e)); res[key][1] += int(not np.any(v))
    if rank == 0:
        os.remove(path)
for key, (wrong, zero) in res.items():
    tot = np.asarray([wrong, zero]); print(f"SECTOR {key} rank={rank} wrong={wrong}/{TRIALS} allzero={zero}/{TRIALS} nsample={nsample}", flush=True)
