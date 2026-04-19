"""Minimal reproducer for the H5Dread out-of-range failure.

Reads a tiny hyperslab of wfns/coeffs via the phdf5 FFI and prints
what each rank is seeing.  Verifies the FFI per-rank offset math
against the dataset extent.
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import sys
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init():
    if os.environ.get(_DIST_SENTINEL):
        return
    n = int(os.environ.get("SLURM_NTASKS", "1"))
    if n > 1:
        try: jax.distributed.initialize()
        except Exception: pass
    os.environ[_DIST_SENTINEL] = "1"
_maybe_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from file_io.slab_io import SlabIO


def _log(s):
    print(f"[rank {jax.process_index()}] {s}", flush=True)


def main():
    wfn_path = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5"

    # First, rank-0 plain-h5py inspection of the dataset extent.
    if jax.process_index() == 0:
        import h5py
        with h5py.File(wfn_path, "r") as f:
            ds = f["wfns/coeffs"]
            print(f"[h5py] wfns/coeffs: shape={ds.shape}, dtype={ds.dtype}")
            print(f"[h5py] wfns/gvecs : shape={f['wfns/gvecs'].shape}, dtype={f['wfns/gvecs'].dtype}")

    world = jax.process_count()
    assert world == 4
    p, q = 2, 2
    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    # Try the SIMPLEST possible read: full wfns/coeffs of the 80-band
    # prefix, all G, all spinor, all real/imag.  Sharded on band axis.
    # If THIS fails, the issue is in the sharding encoding or bounds math.
    _log(f"mesh axis_names = {mesh.axis_names}")
    _log(f"local devices = {[d.id for d in jax.local_devices()]}")
    _log(f"all devices = {[d.id for d in jax.devices()]}")

    with SlabIO(wfn_path, mode="r", mesh=mesh, use_ffi_io=True) as io:
        # Test 1: single chunky read with zero offset — shape (80, 2, 17559, 2).
        # Global 80/4 = 20 bands per rank.
        _log("--- test 1: full-G read, zero offset ---")
        try:
            arr = io.read_slab(
                "wfns/coeffs",
                shape=(80, 2, 17559, 2),
                dtype=np.float64,
                offset=(0, 0, 0, 0),
                partition_spec=P(("x", "y"), None, None, None),
            )
            _log(f"test 1 OK: shape={arr.shape}, sharding={arr.sharding.spec}")
            for sh in arr.addressable_shards:
                _log(f"  my shard index = {sh.index}, data.shape={sh.data.shape}")
        except Exception as e:
            _log(f"test 1 FAILED: {type(e).__name__}: {e}")

        # Test 2: single-k slab read — shape (80, 2, 1963, 2), offset (0,0,0,0).
        _log("--- test 2: ngkmax-sized read at k=0, zero offset ---")
        try:
            arr = io.read_slab(
                "wfns/coeffs",
                shape=(80, 2, 1963, 2),
                dtype=np.float64,
                offset=(0, 0, 0, 0),
                partition_spec=P(("x", "y"), None, None, None),
            )
            _log(f"test 2 OK: shape={arr.shape}")
        except Exception as e:
            _log(f"test 2 FAILED: {type(e).__name__}: {e}")

        # Test 3: separate-axes sharding P('x','y',None,None) — 80 bands
        # sharded 2-way on x (40/rank-row) then 2 spinor sharded on y (1/rank-col).
        # This matches what phdf5_multi_offset_test uses, just to sanity-check
        # that the per-rank-offset math works on this file at all.
        _log("--- test 3: separate-axes sharding P('x','y',None,None) ---")
        try:
            arr = io.read_slab(
                "wfns/coeffs",
                shape=(80, 2, 1963, 2),
                dtype=np.float64,
                offset=(0, 0, 0, 0),
                partition_spec=P("x", "y", None, None),
            )
            _log(f"test 3 OK: shape={arr.shape}")
        except Exception as e:
            _log(f"test 3 FAILED: {type(e).__name__}: {e}")

        # Test 5: time the plain single-offset read of wfns/coeffs at full
        # shape (80, 2, 17559, 2) f64 across many iters.  Pure FFI cost,
        # no post-processing.
        import time
        for trial in range(3):
            t0 = time.perf_counter()
            arr = io.read_slab(
                "wfns/coeffs",
                shape=(80, 2, 17559, 2),
                dtype=np.float64,
                offset=(0, 0, 0, 0),
                partition_spec=P(("x", "y"), None, None, None),
            )
            jax.block_until_ready(arr)
            dt = time.perf_counter() - t0
            bytes_per_rank = 20 * 2 * 17559 * 2 * 8
            _log(f"test 5.{trial}: full single-offset read  {dt*1e3:.1f} ms  "
                 f"({bytes_per_rank/dt/1e6:.1f} MB/s/rank, "
                 f"{4*bytes_per_rank/dt/1e6:.1f} MB/s global)")

        # Test 4: per-k slab reads with real offsets — this is what the
        # main test loop does.  Find the k that blows up.
        ngk = np.array([1947, 1963, 1963, 1963, 1917, 1963, 1963, 1963, 1917])
        kpt_starts = np.concatenate([[0], np.cumsum(ngk)[:-1]])
        ngktot = int(kpt_starts[-1] + ngk[-1])
        ngkmax = int(ngk.max())
        for k in range(len(ngk)):
            if kpt_starts[k] + ngkmax <= ngktot:
                slab_start = int(kpt_starts[k])
            else:
                slab_start = int(ngktot - ngkmax)
            _log(f"--- test 4.{k}: slab_start={slab_start} ngkmax={ngkmax} "
                 f"(range [{slab_start}, {slab_start+ngkmax}) vs ngktot={ngktot}) ---")
            try:
                arr = io.read_slab(
                    "wfns/coeffs",
                    shape=(80, 2, ngkmax, 2),
                    dtype=np.float64,
                    offset=(0, 0, slab_start, 0),
                    partition_spec=P(("x", "y"), None, None, None),
                )
                _log(f"test 4.{k} OK: shape={arr.shape}")
            except Exception as e:
                _log(f"test 4.{k} FAILED: {type(e).__name__}: {str(e)[:200]}")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
