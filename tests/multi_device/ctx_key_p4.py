"""P=4 gate: the cuBLASMp/cuSolverMp ``ctx_key`` attribute and its native registry.

Four processes, one GPU each, 2x2 mesh.

1. A cuBLASMp ``gemm_plan`` matches ``jnp`` on rank-varying operands, and
   its lowered module carries ``ctx_key = <key>`` (the same on every rank,
   equal to ``loader.context_key`` of the configuration) and no
   ``ctx_handle``: the module text is a pure function of the configuration.
2. The registry refuses a second live context for a bound key and a
   different configuration under a bound key (a collision); an identical
   re-bind is accepted.
3. Red twin: after the contexts are torn down, the same compiled plan
   refuses instead of dereferencing a freed context.

Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/ctx_key_p4.py``.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

TAG = "[ctx-key-p4]"


def _put(x, sharding):
    return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])


def main() -> int:
    from distrib_la import gemm_plan
    from distrib_la import loader, _cusolvermp
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(11)
    nq, m, k, n = 2, 8, 12, 4
    A = rng.standard_normal((nq, m, k)) + 1j * rng.standard_normal((nq, m, k))
    B = rng.standard_normal((nq, k, n)) + 1j * rng.standard_normal((nq, k, n))
    plan = gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=jnp.complex128,
                     backend="cublasmp", warmup=True)
    got = np.asarray(multihost_utils.process_allgather(plan(_put(A, sh), _put(B, sh)), tiled=True))
    ref = np.einsum("qmk,qkn->qmn", A, B)
    bad = []
    rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    if not rel < 1e-13:
        bad.append(f"gemm rel {rel:.2e}")

    config = "lorrax-ctx/v1|cusolvermp|2x2|row"
    key = loader.context_key(config)
    text = jax.jit(lambda a, b: plan(a, b)).lower(_put(A, sh), _put(B, sh)).as_text()
    keys = np.asarray(multihost_utils.process_allgather(np.asarray([plan.ctx_key])))
    if plan.ctx_key != key or f"ctx_key = {key}" not in text or "ctx_handle" in text:
        bad.append(f"module carries ctx_key {plan.ctx_key} (want {key}) / ctx_handle present="
                   f"{'ctx_handle' in text}")
    if len(set(int(v) for v in keys.ravel())) != 1:
        bad.append(f"ctx_key differs across ranks: {keys.ravel().tolist()}")

    lib = loader.get_lib("CUDA")
    handle = _cusolvermp.get_or_init_context(mesh, col_major=False)
    err = ctypes.create_string_buffer(512)
    same = lib.lrx_ctx_bind(key, handle, config.encode(), err, 512)
    second = lib.lrx_ctx_bind(key, handle + 64, config.encode(), err, 512)
    msg_second = err.value.decode()
    collide = lib.lrx_ctx_bind(key, handle, (config + "|other").encode(), err, 512)
    msg_collide = err.value.decode()
    if same != 0 or second != 1 or "different live context" not in msg_second \
            or collide != 1 or "collision" not in msg_collide:
        bad.append(f"registry: identical={same} second={second} ({msg_second!r}) "
                   f"collision={collide} ({msg_collide!r})")

    _cusolvermp._atexit_teardown()
    try:
        jax.block_until_ready(plan(_put(A, sh), _put(B, sh)))
        bad.append("red twin: a plan run after teardown did not refuse")
        red = "did not fire"
    except Exception as exc:                                     # noqa: BLE001
        red = f"refused: {type(exc).__name__}: {str(exc)[:160]}"
    if jax.process_index() == 0:
        print(TAG, f"gemm rel={rel:.2e} ctx_key={plan.ctx_key} ranks={keys.ravel().tolist()}",
              flush=True)
        print(TAG, f"teardown red twin {red}", flush=True)
        print(TAG, "FAIL: " + "; ".join(bad) if bad else "PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    # main() returns the gate verdict; run_main_and_finalize carries it into the
    # process exit (a bare finalize_process call exits 0 whatever main returned).
    run_main_and_finalize(main)
