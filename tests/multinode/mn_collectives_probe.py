#!/usr/bin/env python
"""Multi-node collective probe for LORRAX's communicator stack (no solver).

One rank per GPU (launch through src/ffi/cpp/select_gpu.sh).  Brings up the
runtime either through LORRAX (`--stack lorrax`: runtime.bootstrap(), the
one-GPU-per-process gate, the NCCL clique warm-up) or as plain JAX
(`--stack jax`: jax.distributed.initialize() only, no LORRAX import before the
backend starts, no network planner), then times and checks, on a (nodes, gpus)
mesh inside shard_map:

  psum, psum_scatter, all_gather, all_to_all, ppermute,
  lxkit.ordered_psum, lxkit.ordered_psum_scatter     (over the full mesh and the
                                                       inter-node axis 'x')
  common.collectives.barrier                          (comm7 coordination barrier)

per-rank buffer sizes 1 KiB .. 256 MiB, dtypes c128 and f64.  Every result is
checked bit-exactly (integer-valued inputs with power-of-two rank weights).

Hangs: each operation runs under faulthandler.dump_traceback_later(--op-timeout,
exit=True), so a hung collective prints every thread's stack and exits rc=1 on
that rank instead of consuming the allocation.  Per-rank JSONL records go to
--out/rank<k>.jsonl, written and flushed after every measurement.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import socket
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--stack", choices=("lorrax", "jax"), required=True)
p.add_argument("--out", required=True)
p.add_argument("--sizes", default="1K,4K,16K,64K,256K,1M,4M,16M,64M,256M")
p.add_argument("--dtypes", default="c128,f64")
p.add_argument("--ops", default="psum,psum_scatter,all_gather,all_to_all,ppermute,"
                                "ordered_psum,ordered_psum_scatter")
p.add_argument("--axes", default="xy,x", help="comma list of axis sets: xy (all), x (inter-node), y (intra)")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--op-timeout", type=float, default=120.0)
p.add_argument("--barriers", type=int, default=50)
p.add_argument("--startup", choices=("bootstrap", "full"), default="bootstrap",
               help="lorrax stack: bootstrap() only, or every initialize_communicator_stack step "
                    "that does not need a square mesh or the FFI (5b-5d, 7, 7b) around the sweep")
p.add_argument("--fake-mesh", default="", help="single-process test only: e.g. 2x4 (CPU devices)")
args = p.parse_args()

T0 = time.time()
RANK = int(os.environ.get("SLURM_PROCID", "0"))
os.makedirs(args.out, exist_ok=True)
_LOG = open(os.path.join(args.out, f"rank{RANK}.jsonl"), "a", buffering=1)


def rec(**kw):
    kw.setdefault("rank", RANK)
    kw.setdefault("t", round(time.time() - T0, 3))
    _LOG.write(json.dumps(kw) + "\n")
    _LOG.flush()


def say(*a):
    if RANK == 0:
        print(f"[probe {time.time() - T0:8.1f}s]", *a, flush=True)


ENV_KEYS = ("NCCL_", "FI_", "SLURM_NETWORK", "SLURM_STEP_NUM_NODES", "CUDA_VISIBLE_DEVICES",
            "JAX_", "XLA_", "LD_LIBRARY_PATH", "NERSC_HOST", "LORRAX_SITE_PROFILE", "MPICH_GPU")
rec(event="env_before", host=socket.gethostname(),
    env={k: v for k, v in os.environ.items() if k.startswith(ENV_KEYS)})

faulthandler.enable()
faulthandler.dump_traceback_later(600, exit=True)          # start-up watchdog
if args.stack == "lorrax":
    import runtime                                          # noqa: E402  (before jax, by contract)
    runtime.bootstrap()
    from runtime import _enforce_gpu_placement
    _enforce_gpu_placement()
    from runtime.network_env import network_configuration
    decision = network_configuration()
else:
    import jax                                              # noqa: E402
    jax.config.update("jax_enable_x64", True)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        jax.distributed.initialize(local_device_ids=[0] if vis and "," not in vis else None)
    decision = None
faulthandler.cancel_dump_traceback_later()

import numpy as np                                          # noqa: E402
import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from jax import lax                                         # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402
try:
    from jax import shard_map
except ImportError:                                          # pragma: no cover
    from jax.experimental.shard_map import shard_map

NP = jax.process_count()
ND = jax.device_count()
rec(event="env_after_init", decision=decision,
    env={k: v for k, v in os.environ.items() if k.startswith(ENV_KEYS)},
    local_devices=[str(d) for d in jax.local_devices()], n_devices=ND, n_proc=NP,
    jax=jax.__version__)
say(f"stack={args.stack} P={ND} processes={NP} jax={jax.__version__} decision={decision}")

# ---- mesh: x = node, y = GPU within node (devices in process order) -----
devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
hosts = {}
# process -> host: exchange through the coordination KV store (no device op).
from jax._src import distributed as _dist                   # noqa: E402
client = _dist.global_state.client
if client is not None:
    client.key_value_set(f"mnprobe:host:{jax.process_index()}", socket.gethostname())
    for i in range(NP):
        hosts[i] = client.blocking_key_value_get(f"mnprobe:host:{i}", 120_000)
else:
    hosts[0] = socket.gethostname()
nodes = sorted(set(hosts.values()), key=lambda h: min(i for i in hosts if hosts[i] == h))
NX, NY = len(nodes), ND // len(nodes)
if NX * NY != ND or any(hosts[d.process_index] != nodes[k // NY] for k, d in enumerate(devs)):
    NX, NY = 1, ND
if NP == 1 and args.fake_mesh:
    NX, NY = (int(v) for v in args.fake_mesh.split("x"))
mesh = Mesh(np.array(devs).reshape(NX, NY), ("x", "y"))
say(f"mesh {NX}x{NY}; nodes={nodes}")
rec(event="mesh", nx=NX, ny=NY, hosts=hosts)

ordered_psum = ordered_psum_scatter = None
try:
    from lxkit import ordered_psum, ordered_psum_scatter   # noqa: F811
except Exception as exc:                                    # 3a9e3cd67 has no lxkit.collectives
    rec(event="ordered_unavailable", err=f"{type(exc).__name__}: {exc}")
    say(f"lxkit ordered reductions unavailable: {exc}")

AXES = {"xy": ("x", "y"), "x": ("x",), "y": ("y",)}
SPEC = P(("x", "y"))


def _bytes(s):
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(s[:-1]) * mult[s[-1]] if s[-1] in mult else int(s)


def lin(ax):
    """Row-major index of this device over the axis tuple, and the axis size."""
    idx, n = 0, 1
    for a in ax:
        sz = lax.axis_size(a)
        idx = idx * sz + lax.axis_index(a)
        n *= sz
    return idx, n


def grank():
    return lax.axis_index("x") * NY + lax.axis_index("y")


def group_weight(ax):
    """sum over this device's group (members along `ax`) of 2**global_rank."""
    # global rank = ix*NY + iy; the group varies the axes in `ax`, others fixed.
    ix, iy = lax.axis_index("x"), lax.axis_index("y")
    xs = jnp.arange(NX) if "x" in ax else ix[None]
    ys = jnp.arange(NY) if "y" in ax else iy[None]
    r = (xs[:, None] * NY + ys[None, :]).reshape(-1)
    return jnp.sum(2.0 ** r)


def weight_of(ax, j):
    """2**global_rank of the j-th member (row-major over `ax`) of my group."""
    ix, iy = lax.axis_index("x"), lax.axis_index("y")
    if ax == ("x", "y"):
        r = j
    elif ax == ("x",):
        r = j * NY + iy
    else:
        r = ix * NY + j
    return 2.0 ** r


def base(n, dt):
    a = (jnp.arange(n) % 1021).astype(jnp.float64)
    return (a - 1j * (a + 1.0)).astype(dt) if jnp.issubdtype(dt, jnp.complexfloating) else a.astype(dt)


def make_input(op, nel, dt, ax):
    n_ax = int(np.prod([mesh.shape[a] for a in ax]))

    def gen(_):
        w = 2.0 ** grank()
        if op == "all_to_all":
            j = jnp.arange(n_ax, dtype=jnp.float64)[:, None]
            x = (w * base(nel // n_ax, dt)[None, :] + j).reshape(-1)
        else:
            x = w * base(nel, dt)
        return x.astype(dt)
    f = jax.jit(shard_map(gen, mesh=mesh, in_specs=SPEC, out_specs=SPEC, check_vma=False))
    seed = jnp.zeros((ND,), jnp.int32, device=NamedSharding(mesh, SPEC))
    return f(seed), n_ax


def kernel(op, ax, n_ax):
    if op == "psum":
        return lambda x: lax.psum(x, ax)
    if op == "ordered_psum":
        return lambda x: ordered_psum(x, ax)
    if op == "psum_scatter":
        return lambda x: lax.psum_scatter(x, ax, scatter_dimension=0, tiled=True)
    if op == "ordered_psum_scatter":
        return lambda x: ordered_psum_scatter(x, ax, scatter_dimension=0, tiled=True)
    if op == "all_gather":
        return lambda x: lax.all_gather(x, ax, tiled=True)
    if op == "all_to_all":
        return lambda x: lax.all_to_all(x.reshape(n_ax, -1), ax, 0, 0, tiled=True).reshape(-1)
    if op == "ppermute":
        def f(x):
            # ring shift +1 along the (row-major) group; the perm is over the
            # flattened axis tuple.
            return lax.ppermute(x, ax, perm=[(i, (i + 1) % n_ax) for i in range(n_ax)])
        return f
    raise ValueError(op)


def checker(op, ax, n_ax, nel, dt):
    def chk(y):
        me, _ = lin(ax)
        if op in ("psum", "ordered_psum"):
            exp = group_weight(ax) * base(nel, dt)
        elif op in ("psum_scatter", "ordered_psum_scatter"):
            m = nel // n_ax
            exp = group_weight(ax) * lax.dynamic_slice(base(nel, dt), (me * m,), (m,))
        elif op == "all_gather":
            exp = jnp.concatenate([weight_of(ax, j) * base(nel, dt) for j in range(n_ax)])
        elif op == "all_to_all":
            m = nel // n_ax
            exp = jnp.concatenate([weight_of(ax, j) * base(m, dt) + me for j in range(n_ax)])
        elif op == "ppermute":
            exp = weight_of(ax, (me - 1) % n_ax) * base(nel, dt)
        err = jnp.max(jnp.abs(y - exp.astype(dt))).astype(jnp.float64)
        return err[None]
    return jax.jit(shard_map(chk, mesh=mesh, in_specs=SPEC, out_specs=SPEC, check_vma=False))


def local(a):
    return float(np.asarray(a.addressable_shards[0].data).reshape(-1)[0])


def busbw_factor(op, n):
    if op in ("psum", "ordered_psum"):
        return 2.0 * (n - 1) / n
    if op == "ppermute":
        return 1.0
    return (n - 1) / n


def loaded_libs():
    out = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                pth = line.split()[-1]
                if "nccl" in pth or "libfabric" in pth or "ofi" in pth:
                    out.add(pth)
    except OSError:
        pass
    return sorted(out)


FULL = args.stack == "lorrax" and args.startup == "full"
if FULL:
    faulthandler.dump_traceback_later(args.op_timeout, exit=True)
    t = time.time()
    runtime._enforce_supported_jax(say)                      # 5b
    runtime._enforce_cpu_mpi_thread_multiple(say)            # 5c
    from common.jax_compile_cache import install_compile_agreement, ensure_jax_compile_cache
    install_compile_agreement()                              # 5d
    try:                                                     # 6: record what the square-mesh policy says
        from common.collectives import resolve_mesh
        resolve_mesh(axis_names=("x", "y"))
        rec(event="resolve_mesh", status="ok")
    except Exception as exc:                                 # noqa: BLE001
        rec(event="resolve_mesh", status="refused", err=f"{type(exc).__name__}: {exc}"[:600])
        say(f"resolve_mesh refused (expected for non-square P): {str(exc)[:200]}")
    faulthandler.cancel_dump_traceback_later()

# ---- NCCL clique warm-up (LORRAX's own, on the lorrax stack) ------------
faulthandler.dump_traceback_later(args.op_timeout, exit=True)
t = time.time()
if args.stack == "lorrax":
    runtime.nccl_warmup(mesh)
else:
    x = jnp.ones((NX, NY), jnp.float64, device=NamedSharding(mesh, P("x", "y")))
    jax.jit(jnp.sum)(x).block_until_ready()
faulthandler.cancel_dump_traceback_later()
rec(event="warmup", s=round(time.time() - t, 3), libs=loaded_libs())
say(f"warm-up {time.time() - t:.2f}s; libs={loaded_libs()}")

if FULL:
    faulthandler.dump_traceback_later(args.op_timeout, exit=True)
    err = None
    try:
        ensure_jax_compile_cache()                           # 7 (P>1 agreement over the coordination service)
    except Exception as exc:                                 # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    runtime._install_pjrt_log_filter()
    from runtime.vendor_kernels import arm_runtime_kernel_stores, start_blas_host_warmup
    arm_runtime_kernel_stores(proc_idx=int(jax.process_index()), say=say)   # 7b
    runtime.default_blas_threads()
    from runtime.env_guard import freeze_process_environment
    freeze_process_environment("mn_collectives_probe step 7b")
    start_blas_host_warmup(proc_idx=int(jax.process_index()), say=say)
    faulthandler.cancel_dump_traceback_later()
    rec(event="full_startup", s=round(time.time() - t, 3), cache_error=err,
        cache_dir=str(jax.config.jax_compilation_cache_dir))
    say(f"full startup steps done in {time.time() - t:.1f}s; cache_error={err}; "
        f"cache_dir={jax.config.jax_compilation_cache_dir}")

# ---- comm7 barrier -------------------------------------------------------
if args.barriers:
    try:
        from common import collectives as _coll
        faulthandler.dump_traceback_later(args.op_timeout, exit=True)
        _coll.barrier("mnprobe.first")
        ts = []
        for i in range(args.barriers):
            t = time.perf_counter()
            _coll.barrier("mnprobe.loop")
            ts.append(time.perf_counter() - t)
        faulthandler.cancel_dump_traceback_later()
        rec(event="barrier", n=len(ts), med_us=1e6 * float(np.median(ts)),
            max_us=1e6 * float(np.max(ts)), status="ok")
        say(f"collectives.barrier x{len(ts)}: median {1e6*np.median(ts):.0f} us, max {1e6*np.max(ts):.0f} us")
    except Exception as exc:
        faulthandler.cancel_dump_traceback_later()
        rec(event="barrier", status="error", err=f"{type(exc).__name__}: {exc}")
        say(f"collectives.barrier ERROR {type(exc).__name__}: {exc}")

# ---- primitive sweep -----------------------------------------------------
DT = {"c128": jnp.complex128, "f64": jnp.float64}
sizes = [_bytes(s) for s in args.sizes.split(",") if s]
n_fail = 0
for axname in args.axes.split(","):
    ax = AXES[axname]
    n_ax = int(np.prod([mesh.shape[a] for a in ax]))
    if n_ax == 1:
        say(f"axes {axname}: size 1, skipped")
        continue
    for op in args.ops.split(","):
        if op.startswith("ordered") and ordered_psum is None:
            rec(event="op", op=op, axes=axname, status="unavailable")
            continue
        for dname in args.dtypes.split(","):
            dt = DT[dname]
            isz = jnp.dtype(dt).itemsize
            for nbytes in sizes:
                nel = nbytes // isz
                if nel % n_ax:
                    continue
                key = dict(event="op", op=op, axes=axname, n=n_ax, dtype=dname, bytes=nbytes)
                faulthandler.dump_traceback_later(args.op_timeout, exit=True)
                say(f"start {op} axes={axname} {dname} {nbytes}B")
                rec(status="start", **key)
                try:
                    x, _ = make_input(op, nel, dt, ax)
                    x.block_until_ready()
                    f = jax.jit(shard_map(kernel(op, ax, n_ax), mesh=mesh, in_specs=SPEC,
                                          out_specs=SPEC, check_vma=False))
                    t = time.perf_counter()
                    y = f(x)
                    y.block_until_ready()
                    t_first = time.perf_counter() - t
                    reps = args.reps if nbytes <= (64 << 20) else max(2, args.reps // 2)
                    ts = []
                    for _ in range(reps):
                        t = time.perf_counter()
                        y = f(x)
                        y.block_until_ready()
                        ts.append(time.perf_counter() - t)
                    err = local(checker(op, ax, n_ax, nel, dt)(y))
                    tm = float(np.median(ts))
                    algbw = (nbytes * (n_ax if op == "all_gather" else 1)) / tm
                    status = "ok" if err == 0.0 else "WRONG"
                    n_fail += status != "ok"
                    rec(status=status, err=err, t_first_s=t_first, t_med_s=tm, t_min_s=min(ts),
                        algbw_GBps=algbw / 1e9, busbw_GBps=algbw * busbw_factor(op, n_ax) / 1e9, **key)
                    if RANK == 0:
                        say(f"  {op:22s} {axname:2s} {dname:4s} {nbytes:>10d}B  {1e6*tm:10.1f} us  "
                            f"busbw {algbw*busbw_factor(op, n_ax)/1e9:7.2f} GB/s  err={err:g}  "
                            f"first={t_first:.2f}s")
                    del x, y
                except Exception as exc:
                    n_fail += 1
                    rec(status="error", err=f"{type(exc).__name__}: {exc}"[:2000], **key)
                    say(f"  ERROR {op} {axname} {dname} {nbytes}: {type(exc).__name__}: {str(exc)[:500]}")
                faulthandler.cancel_dump_traceback_later()

rec(event="done", n_fail=n_fail, libs=loaded_libs())
if client is not None:
    try:
        client.wait_at_barrier("mnprobe:end", 300_000)
    except Exception as exc:                                # noqa: BLE001
        rec(event="end_barrier", err=f"{type(exc).__name__}: {exc}")
say(f"DONE n_fail={n_fail} wall={time.time() - T0:.1f}s")
_LOG.close()
sys.stdout.flush()
os._exit(0 if n_fail == 0 else 3)
