"""Stage the module venv's ``nvidia/`` tree to node-local storage, once per node.

Imported by ``sitecustomize.py`` beside it, which the lorrax module puts first on
``PYTHONPATH``, so it runs at every interpreter start in a module shell, before
anything imports ``nvidia`` or JAX.  Standard library only.

WHY.  The module venv lives on ``/global/common`` (read-only DVS on compute nodes).
Measured on one node (nid002077, cold hsuite P4, a822be861): the module took 1352 s
against 773 s for a ``$SCRATCH`` clone of the same commit, with the same XLA compile
and NVRTC build counts; XLA compile 674 vs 390 s, NVRTC 335 vs 122 s.  The CUDA
libraries (nvrtc, nvJitLink, cublas, nccl) and the MathDx headers are what the
module maps from ``/global/common`` and the clone from ``$SCRATCH``.

WHAT.  Inside a Slurm job, on each node, the first rank to take the node lock copies
``<venv site-packages>/nvidia`` (and a ``lorrax-launch/libcudart.so`` link to the
copy's CUDA runtime) to ``<root>/venv-<key>/`` under the first usable root of
``LORRAX_STAGE_ROOTS`` (``/tmp/lorrax-$USER``, node-local tmpfs, then
``$SCRATCH/.cache/lorrax/stage``), writes a ``COMPLETE`` marker and renames the tree
into place; every other rank waits on the lock and reuses it.  The tree is then put
first on ``sys.path`` (``nvidia`` is a namespace package, so JAX's plugin, NVRTC and
the MathDx header root resolve to the copy), and the modulefile already lists the
copy's library directories ahead of the venv's on ``LD_LIBRARY_PATH``.
``<key>`` is the SHA-256 of the venv's ``lorrax-locked-requirements.txt`` (exact
wheel versions, so the same bytes); the native closure gate accepts the copies as
byte-identical files, and the compile caches are keyed on content
(``common.jax_compile_cache``), so a staged process hits the same caches.

``LORRAX_STAGE_LOCAL=off`` disables it; any failure leaves the venv paths in place,
with one line on stderr.
"""
from __future__ import annotations

import fcntl
import os
import shutil
import sys
import time

_OFF = {"0", "off", "false", "no"}


def _say(msg: str) -> None:
    if os.environ.get("SLURM_LOCALID", "0") == "0":
        print(f"[lorrax stage] {msg}", file=sys.stderr, flush=True)


def _tree_bytes(path: str) -> int:
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def _stage_into(root: str, src_nvidia: str, key: str, need: int) -> str | None:
    """``<root>/venv-<key>/site-packages``, staged or reused; None when unusable."""
    final = os.path.join(root, f"venv-{key}")
    site = os.path.join(final, "site-packages")
    if os.path.isfile(os.path.join(final, "COMPLETE")):
        return site
    try:
        os.makedirs(root, mode=0o700, exist_ok=True)
    except OSError:
        return None
    if not os.access(root, os.W_OK):
        return None
    with open(os.path.join(root, f"venv-{key}.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)          # the other ranks of the node wait here
        if os.path.isfile(os.path.join(final, "COMPLETE")):
            return site
        if shutil.disk_usage(root).free < need + (1 << 30):
            return None
        t0 = time.time()
        tmp = f"{final}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)  # a torn tree of a killed stager
        try:
            shutil.copytree(src_nvidia, os.path.join(tmp, "site-packages", "nvidia"),
                            symlinks=True)
            launch = os.path.join(tmp, "lorrax-launch")
            os.makedirs(launch)
            os.symlink(os.path.join(final, "site-packages", "nvidia", "cu13", "lib",
                                    "libcudart.so.13"), os.path.join(launch, "libcudart.so"))
            open(os.path.join(tmp, "COMPLETE"), "w").close()
            os.rename(tmp, final)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        _say(f"staged {need / 1e9:.1f} GB of nvidia/ to {final} in {time.time() - t0:.1f} s")
    return site


def _use(site: str) -> None:
    """Put the staged tree first and bind its CUDA runtime before anything else can.

    JAX's CUDA plugin resolves its ``libcudart.so.13`` NEEDED entry through its own
    RUNPATH (the venv's ``nvidia/cu13/lib``), not through ``nvidia`` on ``sys.path``;
    loading the staged runtime first makes that SONAME resolve to the copy, so one
    process never maps both (the native closure gate refuses two copies)."""
    import ctypes
    if site not in sys.path:
        sys.path.insert(0, site)
    os.environ["LORRAX_STAGED_SITE"] = site
    cudart = os.path.join(site, "nvidia", "cu13", "lib", "libcudart.so.13")
    if os.path.isfile(cudart):
        ctypes.CDLL(cudart, mode=ctypes.RTLD_GLOBAL)


def stage() -> None:
    if os.environ.get("LORRAX_STAGE_LOCAL", "on").strip().lower() in _OFF:
        return
    if os.environ.get("LORRAX_STAGED_SITE"):              # an inherited, already-staged tree
        site = os.environ["LORRAX_STAGED_SITE"]
        if os.path.isdir(site):
            _use(site)
        return
    venv_site = os.environ.get("LORRAX_STAGE_VENV_SITE", "")
    key = os.environ.get("LORRAX_STAGE_KEY", "")
    roots = [r for r in os.environ.get("LORRAX_STAGE_ROOTS", "").split(":") if r]
    if not (os.environ.get("SLURM_JOB_ID") and venv_site and key and roots):
        return                                            # a login shell, or no module
    src = os.path.join(venv_site, "nvidia")
    if not os.path.isdir(src):
        return
    for root in roots:                                    # reuse without walking the venv
        if os.path.isfile(os.path.join(root, f"venv-{key}", "COMPLETE")):
            _use(os.path.join(root, f"venv-{key}", "site-packages"))
            return
    need = _tree_bytes(src)
    for root in roots:
        site = _stage_into(root, src, key, need)
        if site is not None:
            _use(site)
            return
    _say(f"no usable stage root in {roots}; the venv's nvidia/ stays in use")
