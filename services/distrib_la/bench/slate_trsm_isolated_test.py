"""Isolated trsm test with hand-built L (no cholesky in the loop).

Build a known lower-tri L and known X in numpy, compute B = L @ X,
shard L and B, run distributed_trsm with various uplo/op combos, see
which gives X back.

For n=4 with 2x2 mesh we can also print actual numeric values.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST):
        return
    try:
        from ffi.common.ffi_loader import get_lib, platform_from_env
        # Explicit platform: get_lib(None) would initialize the XLA backend
        # (jax.default_backend()) BEFORE jax.distributed.initialize below.
        get_lib(platform_from_env()).lrx_slate_init_mpi()
    except Exception as _e:
        pass
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize(local_device_ids=[0])
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()
from distrib_la import backend_module                                # noqa: E402

distributed_trsm = backend_module("slate").distributed_trsm


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = args.n
    world = jax.process_count()
    if world != 4:
        _log("need 4 procs"); return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2),
                axis_names=("x", "y"))
    sharding = NamedSharding(mesh, P("x", "y"))

    # Hand-build L (lower triangular) + X random + B = L @ X.
    rng = np.random.default_rng(args.seed)
    L_np = np.tril(rng.standard_normal((n, n)))
    # Make diag positive and not too small for numerical sanity.
    np.fill_diagonal(L_np, np.abs(np.diag(L_np)) + 1.0)
    X_true_np = rng.standard_normal((n, n))
    B_np = L_np @ X_true_np

    L_j = jax.device_put(jnp.asarray(L_np), sharding)
    B_j = jax.device_put(jnp.asarray(B_np), sharding)

    if jax.process_index() == 0 and n <= 8:
        print("L_np ="); print(L_np)
        print("X_true_np ="); print(X_true_np)
        print("B_np ="); print(B_np)
        print()

    for side in ("L", "R"):
        for uplo in ("L", "U"):
            for op in ("N", "T"):
                X = distributed_trsm(L_j, B_j, mesh=mesh,
                                     side=side, uplo=uplo, op=op)
                jax.block_until_ready(X)
                X_full = multihost_utils.process_allgather(X)
                if jax.process_index() == 0:
                    X_np = np.asarray(X_full)
                    err = float(np.max(np.abs(X_np - X_true_np)))
                    err_t = float(np.max(np.abs(X_np.T - X_true_np)))
                    print(f"  side={side} uplo={uplo} op={op}:  "
                          f"|X-Xt|={err:.3e}  |X.T-Xt|={err_t:.3e}",
                          flush=True)
                    if n <= 4 and side == "R" and uplo == "U" and op == "N":
                        print("    X="); print(X_np)
    return 0


if __name__ == "__main__":
    sys.exit(main())
