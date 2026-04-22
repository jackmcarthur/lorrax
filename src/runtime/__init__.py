"""LORRAX process startup: JAX env defaults + jax.distributed init.

Centralises what used to live as three near-duplicate blocks at the top
of ``gw/gw_jax.py``, ``psp/run_nscf.py``, and ``centroid/kmeans_isdf.py``.

Usage at every entry point:

    from runtime import set_default_env
    set_default_env()          # MUST precede `import jax`

    import jax
    # ...

    from runtime import init_jax_distributed
    init_jax_distributed()     # no-op for single-process runs

This module does not import ``jax`` at top level, so ``set_default_env``
can run before the first ``import jax`` in the caller.  Allocator env
vars (``XLA_PYTHON_CLIENT_*``, ``TF_GPU_ALLOCATOR``) are deliberately NOT
set here — the Perlmutter modulefile owns those so there is a single
source of truth.  Override them in the shell if you need to.
"""
from __future__ import annotations

import os


__all__ = [
    "set_default_env",
    "init_jax_distributed",
    "fallback_to_cpu_if_no_gpu_backend",
]


def set_default_env(*, platform: str = "gpu") -> None:
    """Set JAX env defaults.  Idempotent; must run before ``import jax``.

    Parameters
    ----------
    platform : {"gpu", "cpu"}
        "gpu" → ``JAX_PLATFORMS=cuda,cpu`` (CUDA preferred, CPU fallback);
        "cpu" → ``JAX_PLATFORM_NAME=cpu`` (forces host execution, used by
        CPU-only benchmarks and tests).
    """
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    if platform == "cpu":
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    elif platform == "gpu":
        if ("JAX_PLATFORMS" not in os.environ
                and "JAX_PLATFORM_NAME" not in os.environ):
            os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    else:
        raise ValueError(
            f"set_default_env(platform={platform!r}): expected 'gpu' or 'cpu'")


# Single process-wide sentinel.  Replaces three per-module sentinels (one
# with a different name by accident, which would have double-initialised
# if centroid and gw_jax ran in the same process).
_DISTRIBUTED_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
_COORDINATOR_PORT = 12355


def _resolve_coordinator_address() -> str:
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coord:
        return coord

    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        import subprocess
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, check=True,
            )
            first_host = result.stdout.strip().splitlines()[0]
            return f"{first_host}:{_COORDINATOR_PORT}"
        except Exception:
            pass

    host = (os.environ.get("SLURMD_NODENAME")
            or os.environ.get("HOSTNAME")
            or "localhost")
    return f"{host}:{_COORDINATOR_PORT}"


def _resolve_local_device_ids() -> list[int] | None:
    # Under the Cray MPICH stack on Perlmutter, each rank runs on exactly
    # one GPU via CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (set by select_gpu.sh
    # in lxrun).  JAX's no-args distributed.initialize() assumes each
    # process owns *all* local GPUs, then hangs in the topology exchange
    # ("GetKeyValue() timed out with key: cuda:local_topology/cuda/1")
    # because each rank is looking for peers that don't exist on its side.
    # Passing local_device_ids explicitly fixes that.
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cv:
        return None
    n_local = len([x for x in cv.split(",") if x.strip()])
    return list(range(n_local)) if n_local else None


def init_jax_distributed() -> None:
    """Initialise ``jax.distributed`` if running under srun with >1 task.

    Idempotent via an env-var sentinel: safe across re-imports (which
    happen when a script runs as ``__main__`` and is then re-imported as
    a package submodule by a child).  No-op for single-process runs.
    """
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return

    proc_count = int(os.environ.get(
        "JAX_PROCESS_COUNT",
        os.environ.get("JAX_NUM_PROCESSES",
                       os.environ.get("SLURM_NTASKS", "1"))))

    if proc_count <= 1:
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        return

    import jax

    local_ids = _resolve_local_device_ids()
    init_kwargs = {"local_device_ids": local_ids} if local_ids else {}
    try:
        jax.distributed.initialize(**init_kwargs)
        os.environ[_DISTRIBUTED_SENTINEL] = "1"
        return
    except Exception:
        pass

    proc_id = int(os.environ.get(
        "JAX_PROCESS_INDEX", os.environ.get("SLURM_PROCID", "0")))
    jax.distributed.initialize(
        coordinator_address=_resolve_coordinator_address(),
        num_processes=proc_count,
        process_id=proc_id,
    )
    os.environ[_DISTRIBUTED_SENTINEL] = "1"


def fallback_to_cpu_if_no_gpu_backend() -> None:
    """If the GPU backend is unavailable, switch env to force CPU.

    Preserves the behaviour historically embedded in ``gw_jax.py`` for
    local/WSL development where ``JAX_PLATFORMS=cuda,cpu`` is set but no
    CUDA is present.
    """
    import jax
    try:
        jax.devices()
    except RuntimeError as exc:
        if "Unknown backend: 'gpu'" in str(exc):
            os.environ.pop("JAX_PLATFORM_NAME", None)
            os.environ["JAX_PLATFORMS"] = "cpu"
        else:
            raise
