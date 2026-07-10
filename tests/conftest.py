"""Shared pytest setup for the LORRAX_A test suite.

JAX must be configured for x64 BEFORE the first ``import jax`` in the
process, otherwise ``jnp.complex128`` silently degrades to complex64
(see jax-ml/jax#current-gotchas).  Pytest collects all test modules
into one process, so the first import wins — set the env here.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

# ---------------------------------------------------------------------------
# pytest-xdist: pin each worker to its own GPU (gw0 → GPU 0, gw1 → GPU 1, …)
# so the e2e regression gates — subprocess launchers that each need ONE
# GPU — run N-wide on an N-GPU node instead of serially on GPU 0.  Must
# run before the worker's first CUDA/JAX init, which is why it lives at
# conftest module scope.  This OVERRIDES any pre-set CUDA_VISIBLE_DEVICES
# (SLURM gres sets "0,1,2,3" for the task): without the override each
# worker — and every gate subprocess it launches — sees all N GPUs and
# runs the gate on an N-device mesh, which breaks the 1-GPU-frozen
# references.  Mapping goes through the existing list so SLURM's device
# selection is respected.  No-op without xdist.
# ---------------------------------------------------------------------------
_wid = os.environ.get("PYTEST_XDIST_WORKER", "")
if _wid.startswith("gw"):
    _preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if _preset:
        _devs = [d for d in _preset.split(",") if d != ""]
    else:
        try:
            import subprocess as _sp
            _n = len(_sp.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True,
                timeout=10).stdout.strip().splitlines())
        except Exception:
            _n = 0
        _devs = [str(i) for i in range(_n)]
    if _devs:
        os.environ["CUDA_VISIBLE_DEVICES"] = _devs[int(_wid[2:]) % len(_devs)]
