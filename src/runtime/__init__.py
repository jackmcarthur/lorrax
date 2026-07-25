"""Centralised JAX env-var setup + jax.distributed initialization.

Every LORRAX driver (gw.gw_jax, psp.run_nscf, centroid.kmeans_isdf, the
phdf5 plumbing tests, ...) needs the same three things:

  * JAX_ENABLE_X64 = 1 and a sensible JAX_PLATFORMS default
  * ``jax.distributed.initialize()`` called exactly once per process with
    the SLURM-aware argument pattern that actually works on Cray MPICH
    (explicit ``local_device_ids`` derived from CUDA_VISIBLE_DEVICES;
    explicit coordinator from SLURM_NODELIST when the no-args default
    hangs)
  * A fallback to CPU when the GPU backend is unavailable (common in
    sandbox tests without a live CUDA context)

This module owns all three.  Each driver should do::

    from runtime import bootstrap
    bootstrap()          # BEFORE the driver's own ``import jax``
    import jax

:func:`bootstrap` bundles the canonical three-call header
(``set_default_env`` → ``init_jax_distributed`` →
``fallback_to_cpu_if_no_gpu_backend``).  It is importable without pulling
in jax (this module only imports jax lazily, inside functions), and it
sets the env vars before anything imports jax — so as long as the CLI
calls it above its own ``import jax``, the before-import contract holds.
Drivers with a non-standard header (e.g. no CPU fallback) can still call
the three pieces individually.

Previously five different modules had their own copies of this logic,
drifting apart over time (gw.gw_jax had the SLURM-coordinator fallback;
psp.run_nscf and centroid.kmeans_isdf didn't; the phdf5 tests had yet
another flavour).  The sentinel-env-var guard (``_LORRAX_JAX_DISTRIBUTED_DONE``)
now persists across re-imports so the re-entry path
``python -m gw.gw_jax`` → ``gw_init`` imports ``gw.gw_jax`` again and
previously double-initialised no longer does.
"""
from __future__ import annotations

import os
import subprocess


_DISTRIBUTED_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"

__all__ = [
    "bootstrap",
    "set_default_env",
    "init_jax_distributed",
    "fallback_to_cpu_if_no_gpu_backend",
]


def bootstrap(*, platform: str = "gpu") -> None:
    """Canonical CLI bootstrap: env defaults + distributed init + CPU fallback.

    One call replaces the three-call header every LORRAX CLI used to
    carry.  MUST run before the caller's own ``import jax``:
    :func:`set_default_env` only works if jax has not been imported yet
    (jax reads its env at import time).  The jax imports *inside*
    :func:`init_jax_distributed` / :func:`fallback_to_cpu_if_no_gpu_backend`
    happen after the env is set, so they are safe.

    Idempotent (each piece guards itself); no-op-ish in single-process
    runs.  ``platform`` forwards to :func:`set_default_env`.
    """
    set_default_env(platform=platform)
    init_jax_distributed()
    fallback_to_cpu_if_no_gpu_backend()


def set_default_env(*, platform: str = "gpu") -> None:
    """Set LORRAX's canonical JAX env defaults.

    Must be called BEFORE ``import jax`` — JAX reads these at import time.
    Uses ``setdefault`` so any caller-provided override wins.

    ``platform="gpu"`` (default) sets ``JAX_PLATFORMS="cuda,cpu"`` so
    JAX tries CUDA and falls back to CPU.  ``platform="cpu"`` forces CPU.
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if platform == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    elif platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        raise ValueError(f"platform must be 'gpu' or 'cpu', got {platform!r}")


def _resolve_proc_count() -> int:
    """Process count: JAX_PROCESS_COUNT → JAX_NUM_PROCESSES → SLURM_NTASKS → 1."""
    return int(os.environ.get(
        "JAX_PROCESS_COUNT",
        os.environ.get(
            "JAX_NUM_PROCESSES",
            os.environ.get("SLURM_NTASKS", "1"))))


def _resolve_proc_id() -> int:
    """Process index: JAX_PROCESS_INDEX → SLURM_PROCID → 0."""
    return int(os.environ.get(
        "JAX_PROCESS_INDEX",
        os.environ.get("SLURM_PROCID", "0")))


def _resolve_coordinator_address() -> str:
    """Coordinator address for jax.distributed.

    JAX_COORDINATOR_ADDRESS overrides everything.  Otherwise resolve the
    first host of SLURM_NODELIST via ``scontrol show hostnames`` and
    append port 12355.  Final fallback: SLURMD_NODENAME / HOSTNAME /
    'localhost' + port 12355.
    """
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coord:
        return coord
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, check=True,
            )
            first_host = result.stdout.strip().split("\n")[0]
            return f"{first_host}:12355"
        except Exception:
            pass
    host = (os.environ.get("SLURMD_NODENAME")
            or os.environ.get("HOSTNAME")
            or "localhost")
    return f"{host}:12355"


def init_jax_distributed() -> None:
    """Call ``jax.distributed.initialize()`` idempotently.

    Safe to call multiple times — the ``_LORRAX_JAX_DISTRIBUTED_DONE``
    env sentinel persists across re-imports within a process (module-
    level Python globals don't, which is why the previous per-driver
    copies sometimes double-initialised when ``python -m gw.gw_jax``
    pulled ``gw.gw_jax`` in again through ``gw_init``).

    The Cray MPICH stack on Perlmutter runs each rank with
    ``CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`` — exactly one GPU per
    process.  ``jax.distributed.initialize()`` with no args then hangs
    in the topology exchange because it assumes each process owns
    *all* local GPUs.  We pass ``local_device_ids`` explicitly,
    derived from CUDA_VISIBLE_DEVICES.  First try that; on failure
    fall back to the explicit ``(coordinator_address, num_processes,
    process_id)`` form.

    ``JAX_COORDINATOR_ADDRESS``, when set, SKIPS the auto-detected form and
    goes straight to the explicit one.  Auto-detection derives the coordinator
    port from ``SLURM_JOB_ID``, so every step of one allocation lands on the
    SAME port: two concurrent runs in a shared interactive allocation (two
    agents attached to one salloc, or one agent's two launches) join each
    other's coordinator and die with ``ABORTED: task N unexpectedly tried to
    connect with a different incarnation``, or hang until srun SIGKILLs them.
    Set a per-launch address (``--env=JAX_COORDINATOR_ADDRESS=$HOST:$PORT``
    with a port unique to the launch) to keep the runs independent.
    """
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return

    proc_count = _resolve_proc_count()
    if proc_count <= 1:
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        return

    import jax

    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_local = len([x for x in cv.split(",") if x.strip()]) if cv else 0
    init_kwargs = {"local_device_ids": list(range(n_local))} if n_local else {}
    if not os.environ.get("JAX_COORDINATOR_ADDRESS"):
        try:
            jax.distributed.initialize(**init_kwargs)
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            return
        except Exception:
            pass

    # ``local_device_ids`` matters on BOTH paths: without it the explicit form
    # assumes each process owns every local GPU and dies with
    # "CUDA_ERROR_INVALID_DEVICE: invalid device ordinal" under the one-GPU-
    # per-process binding select_gpu.sh sets up.
    jax.distributed.initialize(
        coordinator_address=_resolve_coordinator_address(),
        num_processes=proc_count,
        process_id=_resolve_proc_id(),
        **init_kwargs,
    )
    os.environ[_DISTRIBUTED_SENTINEL] = "1"


def nccl_warmup(mesh_xy) -> None:
    """Pre-initialise every NCCL communicator we'll need later.

    First call on a new NCCL communicator pays ``ncclCommInitRank`` cost
    (~1-2 s on A100) — topology discovery.  Each unique ``replica_groups``
    pattern is a separate communicator.  Our mesh uses three patterns:

      * full-mesh psum     — ``{{0,1,2,3}}``   (used by ``jnp.mean``,
                                                 reductions with no axis
                                                 arg, etc.)
      * 'x'-axis psum      — ``{{0,1},{2,3}}`` (sigma reduce-scatter x
                                                 stage; also triggered by
                                                 any axis-'x' psum)
      * 'y'-axis psum      — ``{{0,2},{1,3}}`` (sigma reduce-scatter y
                                                 stage; any axis-'y' psum)

    Firing a dummy psum on each pattern at driver init moves the
    multi-second first-call cost off whatever timed section would
    otherwise have hit it (most recently: a 1.9 s single ``all-reduce-start``
    inside ``jit(_mean)/reduce_sum`` during the sigma phase, traced via
    the profiling stack).  No-op in single-process mode.
    """
    import jax
    import jax.numpy as jnp
    if jax.process_count() <= 1:
        return
    from jax.sharding import NamedSharding, PartitionSpec as P
    # Each (axis_spec) shape below has its reduction emit a distinct NCCL
    # communicator at XLA lower time.  ``jnp.sum`` on an array sharded
    # over the given axes lowers to the right psum; ``jax.lax.psum`` isn't
    # callable from top-level jit (needs shard_map/pmap context), so we
    # route the warmup through the implicit-reduction path instead.
    shape2d = tuple(mesh_xy.shape[ax] for ax in mesh_xy.axis_names)
    warm_specs = [(shape2d, P(*mesh_xy.axis_names))]       # full-mesh psum
    for ax in mesh_xy.axis_names:
        n_ax = int(mesh_xy.shape[ax])
        warm_specs.append(((n_ax,), P(ax)))                # per-axis psum
    for shape, spec in warm_specs:
        sharding = NamedSharding(mesh_xy, spec)
        x = jax.device_put(jnp.ones(shape, dtype=jnp.float64), sharding)
        _ = jax.jit(jnp.sum)(x).block_until_ready()


def _gpu_is_present() -> bool:
    """True if an NVIDIA GPU is actually visible to this process.

    Used to decide whether a JAX GPU-backend init failure is a benign
    "no GPU here, run on CPU" (login/CPU node) or a genuine GPU-init
    failure that must NOT be masked (GPU node with a driver/library
    problem).  Signals, cheapest first:

      * ``CUDA_VISIBLE_DEVICES=""`` → explicitly masked, no GPU.
      * any ``/dev/nvidia[0-9]*`` device node (or ``/dev/nvidiactl``) →
        a GPU is physically present on this node.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False
    import glob
    return bool(glob.glob("/dev/nvidia[0-9]*")) or os.path.exists("/dev/nvidiactl")


def fallback_to_cpu_if_no_gpu_backend() -> None:
    """If ``jax.devices()`` fails because no GPU backend came up, retry on CPU.

    Two failure strings count as "no GPU backend":

      * ``Unknown backend: 'gpu'``            — JAX_PLATFORMS unset / 'gpu'
        with no CUDA runtime (sandbox / test contexts).
      * ``Unable to initialize backend 'cuda'`` (and the 'gpu'/'rocm'
        variants) — what JAX raises when ``JAX_PLATFORMS='cuda,cpu'`` (the
        value ``set_default_env()`` sets) is tried on a CPU node.

    We downgrade to CPU ONLY when no GPU is actually present
    (:func:`_gpu_is_present`): on a real GPU node a cuda-init failure is a
    genuine error and is re-raised rather than silently masked by a
    catastrophically-slow CPU run.  On downgrade we clear JAX_PLATFORM_NAME,
    force JAX_PLATFORMS=cpu, and drop any cached (failed) backend so the
    caller's next ``jax.devices()`` re-initialises cleanly on CPU.
    """
    import jax
    caught = None
    try:
        jax.devices()
        return
    except RuntimeError as exc:
        caught = exc            # bind to a name the except block won't delete
        msg = str(exc)
    no_gpu_backend = (
        "Unknown backend: 'gpu'" in msg
        or "Unable to initialize backend 'cuda'" in msg
        or "Unable to initialize backend 'gpu'" in msg
        or "Unable to initialize backend 'rocm'" in msg)
    if not (no_gpu_backend and not _gpu_is_present()):
        # Genuine failure (real GPU-init error on a GPU node, or a
        # non-backend RuntimeError): re-raise the ORIGINAL exception.  A
        # bare ``raise`` here would throw "No active exception to re-raise"
        # since the except block has exited.
        raise caught
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:                       # drop the half-initialised cuda backend cache
        jax.clear_backends()
    except Exception:
        pass
