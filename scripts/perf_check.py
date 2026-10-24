"""LORRAX install performance self-check: PASS, or FAIL with the cause and the fix.

    srun -N 1 -n 4 --gpus-per-node=4 <your production binding flags> \\
        $LORRAX_ROOT/src/ffi/cpp/select_gpu.sh python $LORRAX_ROOT/scripts/perf_check.py

Run it with the SAME srun line (binding flags, OMP_NUM_THREADS, modules) your
production runs use: it judges the install, the node and the launch together.
On a login node, or anywhere without the GPUs a run will use, it refuses to
certify, runs only the static checks, and prints the command to run it properly
(exit 3).  The installer calls it as its last step.

Why it exists.  A published module (venv + source on a read-only network
filesystem) imported fine on a login node and ran 2x slower on compute nodes;
nothing failed, every gate passed.  A slow install is not an error anywhere
else, so it has to be measured, against a reference, on the node that runs.

Components (each a PASS/WARN/FAIL with one line of cause and fix):

  1 filesystem  per-open latency of ~500 real install files (CUDA and mathdx
                headers, libdevice, jax/numpy/scipy and LORRAX sources), every
                rank at once, warm median (a page cache cannot hide a per-open
                server round trip, the DVS mechanism); warm os.walk+lstat rate
                of the venv; cold ``import`` of runtime, jax, the backend and a
                driver; .pyc presence on a read-only tree; the mount and options.
  2 caches      JAX compile cache and NVRTC cubin cache: location writable, on a
                fast filesystem, content-keyed namespace (``_k3``), cubin store
                images for this GPU's sm_XX; a second process must hit the
                persistent cache for every program the first compiled.
  3 kernels     the FFI CUDA leg carries SASS for this sm_XX (fatbin parse), the
                NVRTC path produces a cubin, PTX-JIT traps (CUDA_FORCE_PTX_JIT,
                ...), whether XLA execs ptxas/nvlink or compiles in-process.
  4 binding     per rank: affinity mask, physical cores, SMT siblings, the
                OpenMP team libgomp will start, OpenBLAS/MKL threads, overlap
                between ranks.
  5 libraries   the FFI legs' recorded dependency closure enforced
                (lxkit.native_provider, the same API the runtime gate uses) and
                its cost; duplicate CUDA libraries mapped; LD_LIBRARY_PATH traps.
  6 benchmark   per GPU: cold XLA compile of a fixed program set (fresh cache),
                the warm re-run, one NVRTC kernel build, one FFI call, GEMM and
                FFT, against per-model references (no reference: components 1-5
                still judge).

Exit codes: 0 PASS (or CPU-only PASS), 1 FAIL, 3 not certified (wrong place),
2 the check itself broke.  Design and calibration: docs/installation/perf_check.md.
Stdlib only at module scope; every jax/LORRAX import happens in a probe
subprocess so its cost is measured, not hidden.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve()
#: The install's source root: LORRAX_ROOT (a module or lorrax-env.sh sets it), else this tree.
ROOT = Path(os.environ.get("LORRAX_ROOT") or HERE.parents[1]).resolve()

# ---------------------------------------------------------------------------
# References and thresholds (calibration: docs/installation/perf_check.md)
# ---------------------------------------------------------------------------
#: Per-GPU-model references, measured on a fast install (git clone + uv venv on
#: Lustre $SCRATCH) on Perlmutter, 4 ranks x 1 A100, --cpus-per-task=16
#: --threads-per-core=1, OMP_NUM_THREADS=14.  Values are medians of n>=3 runs;
#: ``spread`` is the measured max/min - 1 of the slowest-rank value.
REFERENCE: dict = {
    # job 60189229, nid200276, 2026-10-23, claude/ri-selfcheck 0a873fadf on robust-install (JAX
    # 0.12.0.1), slowest rank of 4, n=3 (ref1-3): compile_cold 7.63/7.64/7.97 s (spread 4.4%),
    # warm 0.687/0.699/0.690, NVRTC 5.78/5.80/6.62 (14.5%; the first run on the node is the
    # high one), GEMM 19.22-19.29, FFT 1.023-1.027 ms; import 3.7/3.7/18.6 s (first touch).
    "NVIDIA A100-SXM4-80GB": {
        "import_s": 18.6, "compile_cold_s": 7.64, "compile_warm_s": 0.693,
        "nvrtc_s": 5.83, "gemm_tflops": 19.26, "fft_ms": 1.025, "host_gemm_s": None,
    },
    # job 60190769, nid002136, 2026-10-23, 581a81831, n=3: compile_cold 7.68/7.75/8.02 s,
    # warm 0.696-0.703, NVRTC 5.75/6.78/7.03 (22%), GEMM 19.28-19.29, FFT 1.240-1.241 ms,
    # import 3.7/3.8/10.6 s (first touch).
    "NVIDIA A100-SXM4-40GB": {
        "import_s": 10.6, "compile_cold_s": 7.75, "compile_warm_s": 0.699,
        "nvrtc_s": 6.78, "gemm_tflops": 19.29, "fft_ms": 1.240, "host_gemm_s": None,
    },
}
#: A metric fails when it is worse than reference by more than this fraction...
SLOWER_FAIL = 0.15
#: ...and by more than this absolute amount (its unit; keeps small metrics from failing
#: on jitter).  host_gemm_s is reported, never judged: its thread count is the launch's
#: OpenBLAS choice (select_gpu.sh defaults OPENBLAS_NUM_THREADS=1), judged by component 4.
ABS_SLACK_S = {"import_s": 2.0, "compile_cold_s": 0.5, "compile_warm_s": 0.1,
               "nvrtc_s": 1.0, "fft_ms": 0.1}
#: Filesystem, primary: open()+close() of ~500 real install files on every rank at once,
#: median of the SECOND (warm) pass, slowest rank.  Compute-node medians (agent D, ri_diagnose;
#: this check's own table in docs/installation/perf_check.md): module on DVS 3.5 ms, $HOME 10.6,
#: CFS 9.6, $SCRATCH Lustre 0.24, node /tmp 0.005; every filesystem ~0.07 ms on a login node.
OPEN_PASS_MS = 0.5
OPEN_FAIL_MS = 1.0
OPEN_PER_GROUP = 120
OPEN_WALL_BUDGET_S = 25.0
#: Filesystem, secondary: warm os.walk + lstat of the venv (module 952/s, clone 19983/s).
WALK_FAIL_PER_S = 5000.0
WALK_MAX_ENTRIES = 20000
WALK_BUDGET_S = 8.0
#: NVRTC images for this GPU in the run's cache + store: fewer than MIN = no store (FAIL);
#: fewer than FULL = incomplete (WARN; the hsuite grids alone are ~60 images, the full
#: build_cubin_store table 389).
MIN_STORE_IMAGES = 10
FULL_STORE_IMAGES = 50
#: Filesystem types that are slow for many-small-file/exec workloads when they
#: hold the runtime (diagnosis wording only; numbers decide).
NETWORK_FS = {"dvs", "nfs", "nfs4", "gpfs", "lustre", "cifs", "smb3", "fuse.sshfs",
              "fuse.squashfuse", "beegfs", "wekafs", "ceph", "fuse.gcsfuse"}

STATUS_ORDER = {"PASS": 0, "SKIP": 0, "INFO": 0, "WARN": 1, "FAIL": 2}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def say(msg: str) -> None:
    print(f"[perf_check] {msg}", flush=True)


def env_int(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() else None


def parse_cpulist(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi) + 1) if hi else {int(lo)})
    return out


def mounts() -> list[tuple[str, str, str, str]]:
    """(mount point, fstype, source, options), longest mount point first."""
    rows = []
    try:
        with open("/proc/self/mountinfo") as fh:
            for line in fh:
                left, _, right = line.partition(" - ")
                f = left.split()
                r = right.split()
                if len(f) >= 6 and len(r) >= 2:
                    mp = f[4].replace("\\040", " ")
                    opts = ",".join(o for o in (f[5], r[2] if len(r) > 2 else "") if o)
                    rows.append((mp, r[0], r[1], opts))
    except OSError:
        pass
    rows.sort(key=lambda t: len(t[0]), reverse=True)
    return rows


_MOUNTS: list | None = None


def fs_of(path: str | os.PathLike) -> dict:
    """{mount, fstype, source, readonly} for a path (realpath, longest prefix)."""
    global _MOUNTS
    if _MOUNTS is None:
        _MOUNTS = mounts()
    p = os.path.realpath(str(path))
    for mp, fstype, src, opts in _MOUNTS:
        if p == mp or p.startswith(mp.rstrip("/") + "/") or mp == "/":
            ro = None
            try:
                ro = bool(os.statvfs(p if os.path.exists(p) else mp).f_flag & os.ST_RDONLY)
            except OSError:
                pass
            return {"mount": mp, "fstype": fstype, "source": src, "readonly": ro,
                    "options": opts[:300]}
    return {"mount": "?", "fstype": "?", "source": "?", "readonly": None, "options": ""}


def describe_fs(path) -> str:
    f = fs_of(path)
    return f"{f['fstype']} at {f['mount']}" + (" (read-only)" if f["readonly"] else "")




# ---------------------------------------------------------------------------
# context: where are we running?
# ---------------------------------------------------------------------------
def gpu_inventory() -> list[dict]:
    """GPUs visible to this process: [{name, cc}] via nvidia-smi, else /dev/nvidia*."""
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout
            gpus = []
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0]:
                    gpus.append({"name": parts[0], "cc": parts[1]})
            if gpus:
                return gpus
        except (OSError, subprocess.SubprocessError):
            pass
    return [{"name": "?", "cc": "?"} for d in sorted(glob.glob("/dev/nvidia[0-9]*"))]


def context() -> dict:
    host = socket.gethostname()
    step = os.environ.get("SLURM_STEP_ID", os.environ.get("SLURM_STEPID", ""))
    in_step = bool(re.fullmatch(r"\d+", step or "")) and "SLURM_PROCID" in os.environ
    login_name = bool(re.search(r"(^|[-_.])login\d*|^login", host)) or bool(
        re.fullmatch(r"(login|uan|frontend|head)\d*", host.split(".")[0]))
    return {
        "host": host,
        "job": os.environ.get("SLURM_JOB_ID"),
        "in_step": in_step,
        "rank": env_int("SLURM_PROCID") or 0,
        "nranks": env_int("SLURM_NTASKS") or 1,
        "local_rank": env_int("SLURM_LOCALID") or 0,
        "login_name": login_name,
    }


def node_topology() -> dict:
    """Logical CPU -> (package, core) and NUMA node; physical core count."""
    cpus = {}
    for d in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        n = int(d.rsplit("cpu", 1)[1])
        try:
            core = int(open(f"{d}/topology/core_id").read())
            pkg = int(open(f"{d}/topology/physical_package_id").read())
        except (OSError, ValueError):
            continue
        cpus[n] = (pkg, core)
    numa = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        try:
            for c in parse_cpulist(open(f"{d}/cpulist").read()):
                numa[c] = int(d.rsplit("node", 1)[1])
        except (OSError, ValueError):
            continue
    return {"cpus": cpus, "numa": numa, "physical": len(set(cpus.values())) or os.cpu_count(),
            "logical": len(cpus) or os.cpu_count()}


def suggested_srun(ngpu: int, topo: dict, entry: str) -> str:
    phys = topo.get("physical") or 64
    per = max(2, phys // max(ngpu, 1))
    run = launch_info().get("lorrax_run")
    # A runtime-image install: lorrax-run restores the image and execs select_gpu.sh itself.
    sel, py = (run, "python") if run else (ROOT / "src/ffi/cpp/select_gpu.sh", shlex.quote(sys.executable))
    return (f"srun -N 1 -n {ngpu} -c {per} --threads-per-core=1 --cpu-bind=threads "
            f"--gpus-per-node={ngpu} env OMP_NUM_THREADS={max(1, per - 2)} "
            f"{sel} {py} {entry}")


# ---------------------------------------------------------------------------
# install file set (bounded walks of the install's own directories only)
# ---------------------------------------------------------------------------
def site_packages() -> Path:
    import sysconfig
    return Path(sysconfig.get_paths()["purelib"])


def bounded_files(root: Path, suffixes: tuple, max_files: int, max_depth: int = 8) -> list[str]:
    """Files under ``root`` (an install directory we own) with these suffixes."""
    out: list[str] = []
    root = Path(root)
    if not root.is_dir():
        return out
    base_depth = len(root.parts)
    for dirpath, dirs, files in os.walk(root):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for f in files:
            if not suffixes or f.endswith(suffixes):
                out.append(os.path.join(dirpath, f))
                if len(out) >= max_files:
                    return out
    return out


def open_sample() -> list[tuple[str, str]]:
    """``(group, path)`` of ~500 real files a run opens, from the install's own tree.

    A cold LORRAX process opens ~10^4 such files serially: NVRTC re-reads every CUDA and
    mathdx header per kernel build, XLA opens libdevice once per compiled module, Python
    opens every imported source/bytecode file (ri_rootcause: on a DVS mount each successful
    open is a synchronous server round trip, 5-10 ms, 170 ms under 4-rank load)."""
    sp = site_packages()
    groups = (
        ("cuda headers", [p for d in sorted(glob.glob(str(sp / "nvidia/*/include")))
                          for p in bounded_files(Path(d), (".h", ".hpp", ".cuh", ".inl"), 60)],
         OPEN_PER_GROUP),
        ("mathdx headers", bounded_files(sp / "nvidia/mathdx/include", (".hpp", ".h", ".cuh", ".inl"),
                                         OPEN_PER_GROUP), OPEN_PER_GROUP),
        ("site-packages", [p for pkg in ("jax", "numpy", "scipy")
                           for p in bounded_files(sp / pkg, (".py",), OPEN_PER_GROUP)], OPEN_PER_GROUP),
        ("source", bounded_files(ROOT / "src", (".py",), OPEN_PER_GROUP), OPEN_PER_GROUP),
        ("libdevice", glob.glob(str(sp / "nvidia/*/nvvm/libdevice/*.bc")), 4),
    )
    out = []
    for name, files, cap in groups:
        files = sorted(files)
        step = max(1, len(files) // cap) if files else 1
        out += [(name, p) for p in files[::step][:cap]]
    return out


def _stats(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    s = sorted(ms)
    return {"n": len(s), "median_ms": s[len(s) // 2], "p90_ms": s[int(0.9 * (len(s) - 1))],
            "max_ms": s[-1], "total_s": sum(s) / 1e3}


def probe_open_latency(sample: list[tuple[str, str]]) -> dict:
    """open()+close() latency of the sample, twice (the second pass shows a warm page
    cache does not hide it); per group and overall, plus the mount each group is on."""
    res: dict = {"passes": [], "groups": {}}
    t0 = time.perf_counter()
    for k in range(2):
        per: dict = {}
        for group, p in sample:
            if time.perf_counter() - t0 > OPEN_WALL_BUDGET_S:
                break
            t1 = time.perf_counter()
            try:
                fd = os.open(p, os.O_RDONLY)
                os.close(fd)
            except OSError:
                continue
            per.setdefault(group, []).append((time.perf_counter() - t1) * 1e3)
        res["passes"].append({"all": _stats([v for vs in per.values() for v in vs]),
                              **{g: _stats(vs) for g, vs in per.items()}})
    for group, p in sample:
        if group not in res["groups"]:
            res["groups"][group] = {"example": p, **fs_of(p)}
    res["wall_s"] = time.perf_counter() - t0
    return res


def probe_path_lookups(n: int = 20) -> dict:
    """Failed-lookup (ENOENT) latency of every LD_LIBRARY_PATH and PATH directory.

    The dynamic loader of every process -- each ptxas/nvlink XLA execs per compiled module,
    each dlopen -- tries every LD_LIBRARY_PATH entry for every library it resolves (libc,
    libm, ...).  An entry on a slow mount turns that into thousands of failed opens: a
    cudart shim directory on DVS $HOME cost 5495 failed opens and +24 s of one stage
    (ri_rootcause).  Judged: the same missing name looked up again and again, as the loader of
    every new process does for libc.so.6 & co. (a mount that caches negative lookups answers
    locally; DVS $HOME did not).  Also reported: first lookups of unique names."""
    def _t(p: str) -> float:
        t0 = time.perf_counter()
        try:
            os.close(os.open(p, os.O_RDONLY))
        except OSError:
            pass
        return (time.perf_counter() - t0) * 1e3
    out: dict = {}
    for var in ("LD_LIBRARY_PATH", "PATH"):
        rows = []
        for d in dict.fromkeys(x for x in os.environ.get(var, "").split(":") if x):
            if not os.path.isdir(d):
                rows.append({"dir": d, "missing": True})
                continue
            first = [_t(os.path.join(d, f"libperfcheck-{os.getpid()}-{i}.so.1")) for i in range(n // 2)]
            same = os.path.join(d, "libperfcheck-absent.so.1")
            _t(same)
            again = [_t(same) for _ in range(n)]
            rows.append({"dir": d, **_stats(again), "first_median_ms": _stats(first).get("median_ms"),
                         **fs_of(d)})
        out[var] = rows
    return out


def cache_dirs() -> dict[str, str]:
    """The persistent caches a run reads and writes per program and per kernel (the same
    locations ``static_caches`` checks for writability)."""
    explicit = (os.environ.get("ISDF_JAX_CACHE_DIR") or "").strip()
    return {"jax_compile": explicit or user_cache_root("jax_compile"),
            "cubin": user_cache_root("kconv_mathdx")}


def probe_cache_latency(n: int = 20) -> dict:
    """Per-file cost of the cache directories, as a run pays it: open()+close() of up to 60
    existing entries (two passes, the second judged) and the same missing name looked up
    again and again (the compile cache tests every program's key before compiling it; the
    test suite compiles ~2400 programs, and every NVRTC image is one file).  A cache on a
    DVS or NFS home costs 3-10 ms per lookup on a compute node (ri_diagnose)."""
    out: dict = {}
    for label, d in cache_dirs().items():
        probe = d
        while probe and not os.path.isdir(probe):
            probe = os.path.dirname(probe)
        files = bounded_files(Path(d), (), 60, max_depth=4) if os.path.isdir(d) else []
        passes = []
        for _ in range(2):
            ms = []
            for p in files:
                t0 = time.perf_counter()
                try:
                    os.close(os.open(p, os.O_RDONLY))
                except OSError:
                    continue
                ms.append((time.perf_counter() - t0) * 1e3)
            passes.append(_stats(ms))
        absent = os.path.join(probe or "/", f".perfcheck-absent-{socket.gethostname()}")
        miss = []
        for _ in range(n + 1):
            t0 = time.perf_counter()
            try:
                os.stat(absent)
            except OSError:
                pass
            miss.append((time.perf_counter() - t0) * 1e3)
        out[label] = {"path": d, "probed": probe, "open": passes[-1], "open_first": passes[0],
                      "miss": _stats(miss[1:]), **fs_of(probe or "/"),
                      "explicit": label == "jax_compile" and bool((os.environ.get("ISDF_JAX_CACHE_DIR") or "").strip())}
    return out


def launch_info() -> dict:
    """How this rank was launched: through the runtime image's lorrax-run or not."""
    image = os.environ.get("LORRAX_IMAGE_ROOT") or ""
    prefix = os.environ.get("LORRAX_PREFIX") or ""
    run = os.path.join(prefix, "bin", "lorrax-run") if prefix else ""
    return {"image_root": image or None,
            "image_realpath": os.path.realpath(image) if image else None,
            "image_fs": fs_of(image) if image else None,
            "prefix": prefix or None, "runtime": os.environ.get("LORRAX_RUNTIME"),
            "lorrax_run": run if run and os.path.isfile(run) else None}


def probe_walk(root: str) -> dict:
    """Entries/s of os.walk + lstat over the venv, twice; the second (warm) pass is judged.
    Python's import system does exactly this kind of directory scanning and stat-ing."""
    reps = []
    for _ in range(2):
        t0 = time.perf_counter()
        n = 0
        for d, ds, fs in os.walk(root):
            for x in ds + fs:
                try:
                    os.lstat(os.path.join(d, x))
                except OSError:
                    pass
                n += 1
            if n >= WALK_MAX_ENTRIES or time.perf_counter() - t0 > WALK_BUDGET_S:
                break
        dt = time.perf_counter() - t0
        reps.append({"entries": n, "s": dt, "per_s": n / dt if dt > 0 else None})
    return {"root": root, "reps": reps, **fs_of(root)}


# ---------------------------------------------------------------------------
# probes run in a fresh interpreter (their cost is the measurement)
# ---------------------------------------------------------------------------
PROBE_IMPORT = r"""
import json, os, sys, time
t0 = time.perf_counter()
out = {}
err = None
try:
    import runtime  # noqa: F401  first, as every driver does (it tunes OpenBLAS before numpy)
except Exception as e:
    err = f"{type(e).__name__}: {e}"
out["runtime_s"] = time.perf_counter() - t0
t1 = time.perf_counter()
import numpy  # noqa: F401
import jax
out["jax_s"] = time.perf_counter() - t1
t1 = time.perf_counter()
try:
    devs = jax.devices()
    out["devices"] = [d.device_kind for d in devs]
    out["platform"] = devs[0].platform
    out["cc"] = getattr(devs[0], "compute_capability", None)
except Exception as e:
    out["devices_error"] = f"{type(e).__name__}: {e}"
out["backend_s"] = time.perf_counter() - t1
t1 = time.perf_counter()
try:
    import gw.gw_jax  # noqa: F401  the driver import set
except Exception as e:
    err = err or f"{type(e).__name__}: {e}"
out["lorrax_s"] = time.perf_counter() - t1
out["lorrax_error"] = err
out["total_s"] = time.perf_counter() - t0
# .pyc census of every imported module
import sysconfig
roots = tuple(os.path.realpath(p) + "/" for p in (
    sys.argv[1], sysconfig.get_paths()["purelib"], sysconfig.get_paths()["stdlib"]))
missing, have = [], 0
for name, mod in list(sys.modules.items()):
    spec = getattr(mod, "__spec__", None)
    origin = getattr(spec, "origin", None) if spec else None
    cached = getattr(spec, "cached", None) if spec else None
    if not origin or not origin.endswith(".py") or not cached:
        continue
    if not os.path.realpath(origin).startswith(roots):
        continue                       # not the install's (a site's sitecustomize, ...)
    if os.path.exists(cached):
        have += 1
    else:
        missing.append(origin)
out["pyc"] = {"have": have, "missing": len(missing), "missing_sample": missing[:8],
              "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
              "pycache_prefix": sys.pycache_prefix,
              "missing_dirs_writable": sorted({str(os.access(os.path.dirname(m), os.W_OK)) for m in missing})}
out["modules"] = len(sys.modules)
maps = set()
try:
    for line in open("/proc/self/maps"):
        p = line.split(None, 5)
        if len(p) == 6 and ".so" in p[5]:
            maps.add(os.path.realpath(p[5].strip()))
except OSError:
    pass
out["mapped"] = sorted(maps)
print("PROBE_JSON " + json.dumps(out), flush=True)
"""

PROBE_BENCH = r"""
import json, os, re, sys, threading, time
warm = sys.argv[1] == "warm"
out = {"warm": warm}
T = time.perf_counter
t0 = T()
import runtime  # noqa: F401  (BLAS spin tuning, as a run does)
try:
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure(print_fn=lambda *a, **k: None)
except Exception as e:
    out["source_closure_error"] = f"{type(e).__name__}: {e}"
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
out["import_s"] = T() - t0
dev = jax.local_devices()[0]
out["device_kind"] = dev.device_kind
out["platform"] = dev.platform
out["cc"] = getattr(dev, "compute_capability", None)
gpu = dev.platform in ("gpu", "cuda", "rocm")

# --- threads exactly as the run's startup block reports them ------------------
try:
    out["blas_auto"] = runtime.default_blas_threads()
    out["thread_env"] = runtime._thread_env()
except Exception as e:
    out["thread_env_error"] = f"{type(e).__name__}: {e}"

# --- 5: FFI legs + closure enforcement (the runtime gate's API) ---------------
lib = {}
if gpu:
    try:
        from ffi.common import ffi_loader
        from lxkit import native_provider as NP
        for leg in ("CUDA", "cpu"):
            t1 = T()
            try:
                ffi_loader.get_lib(leg)
                lib[leg] = {"open_s": T() - t1, "path": ffi_loader.loaded_lib_path(leg)}
                try:
                    ident = NP.native_closure_identity(leg)
                    exp = NP.expected_closure_identity(lib[leg]["path"], leg)
                    lib[leg]["closure"] = ident[:12]
                    lib[leg]["closure_ok"] = (ident == exp)
                except Exception as e:
                    lib[leg]["closure_error"] = f"{type(e).__name__}: {e}"
            except Exception as e:
                lib[leg] = {"open_s": T() - t1, "error": f"{type(e).__name__}: {e}"}
    except Exception as e:
        lib["error"] = f"{type(e).__name__}: {e}"
out["ffi"] = lib

# --- 2: the real persistent compile cache path (ISDF_JAX_CACHE_DIR = probe dir)
try:
    from common import jax_compile_cache as JCC
    JCC.ensure_jax_compile_cache()
    out["cache_dir"] = JCC._STATE.dir
    out["cache_enabled"] = bool(JCC._STATE.enabled)
    out["namespace"] = JCC.cache_namespace()
except Exception as e:
    out["cache_error"] = f"{type(e).__name__}: {e}"
events = {}
try:
    import jax.monitoring as M
    def _ev(name, *a, **k):
        if "cache" in name:
            events[name] = events.get(name, 0) + 1
    M.register_event_listener(_ev)
except Exception as e:
    out["monitoring_error"] = str(e)

# --- ptxas child-process sampler -----------------------------------------------
children = set()
stop = threading.Event()
def _sample():
    me = os.getpid()
    while not stop.is_set():
        try:
            for t in os.listdir(f"/proc/{me}/task"):
                try:
                    for c in open(f"/proc/{me}/task/{t}/children").read().split():
                        try:
                            children.add(os.path.realpath(f"/proc/{c}/exe"))
                        except OSError:
                            pass
                except OSError:
                    pass
        except OSError:
            pass
        stop.wait(0.003)
th = threading.Thread(target=_sample, daemon=True)
th.start()

# --- 6: the fixed compile set: 12 program shapes x 3 sizes = 36 programs ----------
def f_gemm(a, b): return (a @ b @ a.conj().T).trace()
def f_fft(x): return jnp.abs(jnp.fft.ifftn(jnp.fft.fftn(x) * 2.0)).sum()
def f_eigh(h): w, U = jnp.linalg.eigh(h); return w.sum() + jnp.abs(U[:, :4]).sum()
def f_scan(x):
    def body(cc, xi):
        cc = jnp.tanh(cc * 1.01 + xi)
        return cc, jnp.exp(-cc * cc).sum()
    return lax.scan(body, x[:64], x.reshape(-1, 64))[1].sum()
def f_cg(A, b):
    n = A.shape[0]
    A = A.real @ A.real.T + n * jnp.eye(n)
    def cond(s): return s[3] < 25
    def body(s):
        xk, r, p, k = s
        Ap = A @ p
        al = (r @ r) / (p @ Ap)
        xk = xk + al * p
        rn = r - al * Ap
        p = rn + (rn @ rn) / (r @ r) * p
        return xk, rn, p, k + 1
    return lax.while_loop(cond, body, (jnp.zeros(n), b, b, 0))[0].sum()
def f_sort(x):
    i = jnp.argsort(x)
    return jnp.take(x, i)[::3].sum() + jnp.cumsum(x).max() + jnp.searchsorted(jnp.sort(x), 0.5)
def f_elem(x):
    y = jnp.sin(x) * jnp.exp(-x * x) + jnp.log1p(jnp.abs(x)) / (1 + jnp.sqrt(jnp.abs(x)))
    z = jnp.where(y > 0, jnp.arctan2(y, x + 1.5), jax.scipy.special.erfc(x))
    return (z * jnp.cos(3 * z) + jnp.expm1(-jnp.abs(z))).sum()
def f_solve(A, b):
    n = A.shape[0]
    As =jnp.stack([A.real + (k + n) * jnp.eye(n) for k in range(4)])
    return jax.vmap(jnp.linalg.solve)(As, jnp.stack([b] * 4)).sum()
def f_einsum(a, b): return jnp.einsum("ij,jk,kl->il", a, b, a).real.sum() + jnp.einsum("ii", b).real
def f_scatter(x):
    idx = (jnp.arange(x.size) * 7) % 1000
    return jnp.zeros(1000).at[idx].add(x).sum() + jnp.zeros(1000).at[idx].max(x).sum()
def f_grad(w):
    def loss(w):
        hcur = jnp.tanh(w[:64].reshape(8, 8) @ jnp.ones(8))
        return (hcur ** 2).sum() + jnp.sin(w).sum()
    return jax.grad(loss)(w).sum()
def f_qr(a): q, r = jnp.linalg.qr(a); return jnp.abs(jnp.diag(r)).sum() + jnp.abs(q[0]).sum()
key = np.random.default_rng(0)
SET = []
for n, nv, n3 in ((96, 2048, 16), (160, 4096, 24), (224, 8192, 32)):
    c = key.standard_normal((n, n)) + 1j * key.standard_normal((n, n))
    h = c + c.conj().T
    v = key.standard_normal(nv)
    x3 = key.standard_normal((n3, n3, n3)) + 0j
    SET += [(f"gemm{n}", f_gemm, (c, c)), (f"fft{n3}", f_fft, (x3,)), (f"eigh{n}", f_eigh, (h,)),
            (f"scan{nv}", f_scan, (v,)), (f"cg{n}", f_cg, (c, v[:n])), (f"sort{nv}", f_sort, (v,)),
            (f"elem{nv}", f_elem, (v,)), (f"solve{n}", f_solve, (c, v[:n])),
            (f"einsum{n}", f_einsum, (c, h)), (f"scatter{nv}", f_scatter, (v,)),
            (f"grad{nv}", f_grad, (v,)), (f"qr{n}", f_qr, (c,))]
args_dev = {nm: tuple(jax.device_put(a, dev) for a in args) for nm, _, args in SET}
per = {}
tc = T()
for nm, f, _ in SET:
    t1 = T()
    exe = jax.jit(f).lower(*args_dev[nm]).compile()
    per[nm] = T() - t1
    jax.block_until_ready(exe(*args_dev[nm]))
out["compile_wall_s"] = T() - tc
out["compile_s"] = sum(per.values())
out["programs"] = len(per)
out["s_per_program"] = out["compile_s"] / len(per)
out["compile_per"] = per

# --- 6/3: one NVRTC kernel (a k-grid no store ships) + one FFI call ------------
nvrtc = {}
if gpu:
    try:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from ffi import fft as F
        mesh = Mesh(np.asarray(jax.local_devices()[:1]).reshape(1, 1), ("x", "y"))
        nvrtc["router"] = F.require_kconv(mesh, announce=False)
        nvrtc["cubin_dir"] = F.cubin_cache_dir()
        before = set(os.listdir(nvrtc["cubin_dir"])) if os.path.isdir(nvrtc["cubin_dir"]) else set()
        kg = (7, 3, 1)
        nk, m = 21, 2
        X = jax.device_put(np.ones((nk, m, m), np.complex128), NamedSharding(mesh, P(None, "x", "y")))
        f = jax.jit(F.make_kfft_klead(mesh, kg, P(None, None, None, "x", "y"), kind="fftn", norm="ortho"))
        t1 = T()
        jax.block_until_ready(f(X))
        nvrtc["first_call_s"] = T() - t1
        t1 = T()
        for _ in range(20):
            y = f(X)
        jax.block_until_ready(y)
        nvrtc["ffi_call_ms"] = (T() - t1) / 20 * 1e3
        after = set(os.listdir(nvrtc["cubin_dir"]))
        nvrtc["new_cubins"] = sorted(a for a in after - before if a.endswith(".cubin"))
    except Exception as e:
        nvrtc["error"] = f"{type(e).__name__}: {e}"[:600]
out["nvrtc"] = nvrtc
out["cache_events"] = dict(events)
try:
    st = JCC.compile_cache_stats()
    st.pop("keys", None)
    out["cache_stats"] = st
except Exception as e:
    out["cache_stats_error"] = str(e)
stop.set()
th.join(1)
out["children"] = sorted(children)

# --- 6: device compute + 4: host GEMM ------------------------------------------
if gpu and not warm:
    try:
        N = 8192
        a = jax.device_put(jnp.ones((N, N), jnp.float64) * 0.5, dev)
        mm = jax.jit(lambda a: a @ a)
        jax.block_until_ready(mm(a))
        t1 = T()
        for _ in range(5):
            r = mm(a)
        jax.block_until_ready(r)
        out["gemm_tflops"] = 5 * 2 * N ** 3 / (T() - t1) / 1e12
        z = jax.device_put(jnp.ones((256, 256, 256), jnp.complex128), dev)
        ff = jax.jit(jnp.fft.fftn)
        jax.block_until_ready(ff(z))
        t1 = T()
        for _ in range(10):
            r = ff(z)
        jax.block_until_ready(r)
        out["fft_ms"] = (T() - t1) / 10 * 1e3
    except Exception as e:
        out["compute_error"] = f"{type(e).__name__}: {e}"
if not warm:
    A = np.random.default_rng(1).standard_normal((2048, 2048))
    A @ A
    t1 = T()
    for _ in range(3):
        A @ A
    out["host_gemm_s"] = (T() - t1) / 3
try:
    import ctypes
    g = ctypes.CDLL("libgomp.so.1")
    out["omp_max_threads"] = int(g.omp_get_max_threads())
except OSError:
    out["omp_max_threads"] = None
out["total_s"] = T() - t0
print("PROBE_JSON " + json.dumps(out, default=str), flush=True)
"""


PROBE_STORE = r"""
import json, os, sys, time
out = {}
import runtime  # noqa: F401
try:
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure(print_fn=lambda *a, **k: None)
except Exception as e:
    out["source_closure_error"] = f"{type(e).__name__}: {e}"
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
try:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from ffi import fft as F
    mesh = Mesh(np.asarray(jax.local_devices()[:1]).reshape(1, 1), ("x", "y"))
    out["cubin_dir"] = F.cubin_cache_dir()          # the run's real image cache, seeded from the store
    out["store"] = str(F.CUBIN_STORE)
    t0 = time.perf_counter()
    out["router"] = F.require_kconv(mesh, announce=False)   # runs the (2,1,1) probe kernel
    kg, nk, m = (3, 3, 3), 27, 2                             # an hsuite grid (bcc Na)
    X = jax.device_put(np.ones((nk, m, m), np.complex128), NamedSharding(mesh, P(None, "x", "y")))
    f = jax.jit(F.make_kfft_klead(mesh, kg, P(None, None, None, "x", "y"), kind="fftn", norm="ortho"))
    jax.block_until_ready(f(X))
    out["s"] = time.perf_counter() - t0
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:600]
print("PROBE_JSON " + json.dumps(out, default=str), flush=True)
"""


def kconv_log(text: str) -> list[tuple[str, str]]:
    """[(hit|built, kgrid)] from the k-convolution handlers' own log lines."""
    out = []
    for m in re.finditer(r"\[kconv_mathdx\] (disk-cache hit|NVRTC (?:re)?built)[^\n]*?kgrid=\(([0-9,]+)\)", text):
        out.append(("hit" if m.group(1).startswith("disk") else "built", m.group(2)))
    return out


def count_images(dirs: list[str], sm: int | None) -> dict:
    """Images per directory and how many are for ``sm`` (flat directories; no walk)."""
    res = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            res[d] = {"n": 0, "for_sm": 0, "present": False}
            continue
        names = [e.path for e in os.scandir(d) if e.name.endswith(".cubin")]
        archs = [cubin_arch(p) for p in names[:2000]]
        res[d] = {"n": len(names), "for_sm": sum(1 for a in archs if a == sm) if sm else len(names),
                  "present": True}
    return res


def run_probe(code: str, args: list[str], env: dict, timeout: float) -> tuple[dict, str, float]:
    """Run a probe in a fresh interpreter; (json, stderr+stdout tail, wall)."""
    t0 = time.perf_counter()
    try:
        cp = subprocess.run([sys.executable, "-c", code, *args], env=env, capture_output=True,
                            text=True, timeout=timeout, cwd=tempfile.gettempdir())
        text = (cp.stdout or "") + (cp.stderr or "")
    except subprocess.TimeoutExpired as e:
        text = f"TIMEOUT after {timeout:.0f} s\n" + ((e.stdout or b"").decode(errors="replace")
                                                    if isinstance(e.stdout, bytes) else str(e.stdout or ""))
        return {"error": f"timeout {timeout:.0f} s"}, text[-4000:], time.perf_counter() - t0
    wall = time.perf_counter() - t0
    for line in reversed((cp.stdout or "").splitlines()):
        if line.startswith("PROBE_JSON "):
            try:
                return json.loads(line[len("PROBE_JSON "):]), text[-6000:], wall
            except json.JSONDecodeError:
                break
    return {"error": f"probe exited {cp.returncode} without a result"}, text[-6000:], wall


def nvrtc_times(text: str) -> list[float]:
    """NVRTC build seconds from the FFI's own log lines."""
    return [float(m) / 1e3 for m in re.findall(r"NVRTC (?:re)?built[^\n]*? in ([0-9.]+) ms", text)]


# ---------------------------------------------------------------------------
# component 3 helpers: fatbin/cubin parsing
# ---------------------------------------------------------------------------
def elf_sections(path: str) -> dict[str, tuple[int, int]]:
    """{name: (offset, size)} of a 64-bit little-endian ELF."""
    with open(path, "rb") as fh:
        head = fh.read(64)
        if head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
            return {}
        shoff, = struct.unpack_from("<Q", head, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", head, 0x3A)
        fh.seek(shoff)
        tab = fh.read(shentsize * shnum)
        secs = [struct.unpack_from("<IIQQQQIIQQ", tab, i * shentsize) for i in range(shnum)]
        so, ss = secs[shstrndx][4], secs[shstrndx][5]
        fh.seek(so)
        names = fh.read(ss)
    out = {}
    for s in secs:
        nm = names[s[0]:names.index(b"\0", s[0])].decode(errors="replace")
        out[nm] = (s[4], s[5])
    return out


FATBIN_MAGIC = 0xBA55ED50


def parse_fatbin(blob: bytes) -> list[tuple[str, int]]:
    """[(kind, sm)] of every entry in a .nv_fatbin blob: kind 'sass' (ELF) or 'ptx'."""
    out = []
    off = 0
    while off + 16 <= len(blob):
        magic, _ver, hsize, fsize = struct.unpack_from("<IHHQ", blob, off)
        if magic != FATBIN_MAGIC:
            off += 8                      # sections are 8-aligned between blobs
            continue
        e = off + hsize
        end = e + fsize
        while e + 32 <= min(end, len(blob)):
            kind, _u, ehsize, size = struct.unpack_from("<HHIQ", blob, e)
            if ehsize == 0:
                break
            arch, = struct.unpack_from("<I", blob, e + 28)
            out.append(({1: "ptx", 2: "sass"}.get(kind, f"kind{kind}"), arch))
            e += ehsize + size
        off = end
    return out


def fatbin_inventory(path: str) -> dict:
    try:
        secs = elf_sections(path)
    except (OSError, struct.error, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if ".nv_fatbin" not in secs:
        return {"entries": [], "note": "no .nv_fatbin section"}
    o, s = secs[".nv_fatbin"]
    with open(path, "rb") as fh:
        fh.seek(o)
        blob = fh.read(s)
    ent = parse_fatbin(blob)
    sass = sorted({a for k, a in ent if k == "sass"})
    ptx = sorted({a for k, a in ent if k == "ptx"})
    return {"sass": sass, "ptx": ptx, "n": len(ent)}


def cubin_arch(path: str) -> int | None:
    """sm_XX of a kernel image: the ``_sm<NN>_`` field of LORRAX's image names
    (``kconv_m3_2x1x1_ns1_c64_sm80_<hash>.cubin``, an ``LRXK``-framed file, common/rtc_build.h),
    else the e_flags arch byte of a bare EM_CUDA ELF, either ABI."""
    m = re.search(r"_sm(\d{2,3})[a-z]?_", os.path.basename(path))
    if m:
        return int(m.group(1))
    try:
        with open(path, "rb") as fh:
            h = fh.read(64)
    except OSError:
        return None
    if h[:4] != b"\x7fELF" or len(h) < 52:
        return None
    machine, = struct.unpack_from("<H", h, 18)
    if machine != 190:
        return None
    flags, = struct.unpack_from("<I", h, 0x30)
    lo, hi = flags & 0xFF, (flags >> 8) & 0xFF
    # pre-CUDA-12.8 ABI: arch in the low byte; newer: in bits 8-15.
    for cand in (hi, lo):
        if 30 <= cand <= 200:
            return cand
    return None


# ---------------------------------------------------------------------------
# component 4: binding (per rank) and its judgement (over ranks)
# ---------------------------------------------------------------------------
def probe_binding(topo: dict) -> dict:
    mask = sorted(os.sched_getaffinity(0))
    cores = {topo["cpus"].get(c) for c in mask if c in topo["cpus"]}
    numa = sorted({topo["numa"].get(c) for c in mask if c in topo["numa"]})
    # What runtime.default_blas_threads() would give OpenBLAS when nothing is exported:
    # mask size // SMT siblings of cpu0 (it undercounts a mask that already excludes siblings).
    try:
        sib = len(parse_cpulist(open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list").read()))
    except OSError:
        sib = 1
    return {"mask": mask, "logical": len(mask), "physical": len(cores) or len(mask),
            "openblas_auto": max(1, len(mask) // max(sib, 1)),
            "numa": numa, "host": socket.gethostname(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
            "node_physical": topo["physical"], "node_logical": topo["logical"]}


#: A thread team larger than the rank's physical cores but within its logical CPUs (SMT
#: siblings in the mask, e.g. the docs srun line with OMP_NUM_THREADS unset) -- severity.
SMT_TEAM_SEVERITY = "WARN"


def _ranks_text(rs: list[int]) -> str:
    rs = sorted(rs)
    return f"rank {rs[0]}" if len(rs) == 1 else "ranks " + ",".join(map(str, rs))


def judge_binding(ranks: list[dict]) -> dict:
    """FAIL: overlapping masks, a thread team larger than the rank's CPUs, fewer than two
    physical cores; SMT_TEAM_SEVERITY: a team larger than the physical cores (SMT siblings);
    WARN: team == cores (no room for MPI progress / Python).  Messages are grouped over ranks."""
    found: dict = {}          # (severity, text) -> [ranks]
    rows: list = []
    by_host: dict = {}
    per_rank_cores = []
    for r in ranks:
        b = r.get("binding") or {}
        if not b:
            continue
        by_host.setdefault(b["host"], []).append((r["rank"], set(b["mask"])))
        bench = r.get("bench_cold") or {}
        omp_env = b.get("OMP_NUM_THREADS")
        omp = bench.get("omp_max_threads") or (int(omp_env) if (omp_env or "").isdigit() else b["logical"])
        te = bench.get("thread_env") or {}
        # OpenBLAS reads OPENBLAS_NUM_THREADS, then OMP_NUM_THREADS; unset, the runtime sets it.
        ob = (te.get("OPENBLAS_NUM_THREADS") or b.get("OPENBLAS_NUM_THREADS") or omp_env
              or b.get("openblas_auto"))
        ob = int(ob) if str(ob or "").strip().isdigit() else None
        mkl = b.get("MKL_NUM_THREADS")
        mkl = int(mkl) if (mkl or "").isdigit() else None
        phys, logical = b["physical"], b["logical"]
        per_rank_cores.append(phys)
        rows.append((f"{logical} CPUs = {phys} physical cores"
                     f"{' (SMT siblings in mask)' if logical > phys else ''}, OpenMP team {omp}"
                     f"{'' if omp_env else ' (OMP_NUM_THREADS unset: libgomp uses every CPU of the mask)'}"
                     f", OpenBLAS {ob}, MKL {mkl}"
                     + (f"; NUMA nodes {b['numa']}" if len(b.get("numa") or []) > 1 else ""), r["rank"]))
        for name, n in (("OpenMP", omp), ("OpenBLAS", ob), ("MKL", mkl)):
            if n and n > logical:
                found.setdefault(("FAIL", f"oversubscription: a {n}-thread {name} team on {logical} CPUs "
                                  f"({phys} physical cores)"), []).append(r["rank"])
            elif n and n > phys:
                found.setdefault((SMT_TEAM_SEVERITY, f"oversubscription: a {n}-thread {name} team on {phys} "
                                  f"physical cores (SMT siblings)"), []).append(r["rank"])
        if phys < 2:
            found.setdefault(("FAIL", f"bad binding: {phys} physical core(s); the MPI progress thread "
                              "competes with compute"), []).append(r["rank"])
        elif omp == phys:
            found.setdefault(("WARN", f"OpenMP team = all {phys} cores; leave 2 for MPI progress and "
                              f"Python (OMP_NUM_THREADS={max(1, phys - 2)})"), []).append(r["rank"])
    for host, lst in by_host.items():
        pairs = [(a[0], b_[0], len(a[1] & b_[1])) for i, a in enumerate(lst) for b_ in lst[i + 1:]
                 if a[1] & b_[1]]
        if pairs:
            found.setdefault(("FAIL", f"bad binding: ranks on {host} share CPUs ({len(pairs)} overlapping "
                              f"pairs, e.g. ranks {pairs[0][0]} and {pairs[0][1]} share {pairs[0][2]}): no "
                              "--cpu-bind, or -c missing"), []).extend([r for p in pairs for r in p[:2]])
    out = {"fails": [], "warns": [], "rows": []}
    for (sev, text), rs in found.items():
        msg = f"{_ranks_text(list(set(rs)))}: {text}"
        (out["fails"] if sev == "FAIL" else out["warns"]).append(msg)
    grouped: dict = {}
    for text, rk in rows:
        grouped.setdefault(text, []).append(rk)
    out["rows"] = [f"{_ranks_text(rs)}: {t}" for t, rs in grouped.items()]
    node = (ranks[0].get("binding") or {}) if ranks else {}
    nper = max(1, len(by_host.get(node.get("host"), [])))
    per = max(2, int(node.get("node_physical") or 64) // nper)
    out["fix"] = (f"launch with -c {per} --threads-per-core=1 --cpu-bind=threads and OMP_NUM_THREADS="
                  f"{per - 2} ({nper} ranks on {node.get('node_physical')} physical cores; "
                  "docs/environment/machines/perlmutter.md#cpu)")
    return out


# ---------------------------------------------------------------------------
# judgement
# ---------------------------------------------------------------------------
def comp(name: str, status: str, lines: list[str], metrics: dict | None = None) -> dict:
    return {"name": name, "status": status, "lines": lines, "metrics": metrics or {}}


def worst(*sts: str) -> str:
    return max(sts, key=lambda s: STATUS_ORDER.get(s, 0)) if sts else "PASS"


def vs_ref(metric: str, value, ref: dict, higher_better: bool = False):
    """(status, text) of one metric against its reference."""
    r = (ref or {}).get(metric)
    if value is None:
        return "SKIP", f"{metric}: not measured"
    if r is None:
        return "INFO", f"{metric} {value:.3g} (no reference)"
    if higher_better:
        bad = value < r / (1 + SLOWER_FAIL)
        ratio = r / value if value else float("inf")
    else:
        bad = value > r * (1 + SLOWER_FAIL) and value - r > ABS_SLACK_S.get(metric, 0.0)
        ratio = value / r if r else float("inf")
    return ("FAIL" if bad else "PASS"), f"{metric} {value:.3g} vs ref {r:.3g} ({ratio:.2f}x)"


def open_verdict(opens: list[dict]) -> dict:
    """The slowest rank by second-pass (warm) open median, with its per-group stats."""
    pick = None
    for o in opens:
        passes = o.get("passes") or []
        if len(passes) < 2 or not (passes[1].get("all") or {}).get("n"):
            continue
        if pick is None or passes[1]["all"]["median_ms"] > pick["passes"][1]["all"]["median_ms"]:
            pick = o
    if pick is None:
        return {}
    p1, p2 = pick["passes"][0], pick["passes"][1]
    return {"warm": p2["all"], "first": p1.get("all") or {},
            "groups": {g: v for g, v in p2.items() if g != "all" and v.get("n")},
            "where": pick.get("groups") or {}}


STAGE_FIX = ("stage the runtime node-local (see manual: docs/installation/perf_check.md#filesystem)")


def stage_fix(launch: dict | None) -> str:
    """The filesystem fix for how this check was launched (``launch_info``)."""
    launch = launch or {}
    if launch.get("image_root"):
        fs = launch.get("image_fs") or {}
        return (f"lorrax-run ran the runtime image from {launch.get('image_realpath')} ({fs.get('fstype')} at "
                f"{fs.get('mount')}), a shared copy: this node had no node-local directory with room (see the "
                "[lorrax image] line); free about 9 GB in /tmp, or set LORRAX_IMAGE_ROOTS to a node-local "
                "directory with room")
    if launch.get("lorrax_run"):
        return (f"launch with lorrax-run, which runs every job from a node-local copy of the install: "
                f"srun ... {launch['lorrax_run']} python ... (not the tree's select_gpu.sh)")
    return ("reinstall with config/perlmutter/install_deps.sh (a runtime image every job restores "
            "node-local), or keep the install on a fast filesystem; " + STAGE_FIX)


def judge_lookups(lookups: list[dict]) -> tuple[str, list[str]]:
    """Slow failed lookups in LD_LIBRARY_PATH (FAIL) or PATH (WARN) directories, slowest rank."""
    worst_by: dict = {}
    for lk in lookups:
        for var in ("LD_LIBRARY_PATH", "PATH"):
            for row in lk.get(var) or []:
                if row.get("missing") or not row.get("n"):
                    continue
                # Judged on the MEAN of the repeated misses (their total is what every exec
                # pays): DVS $HOME measured median 0.2-0.7 ms but p90 1.3-7.6 ms.
                row = {**row, "median_ms": row.get("total_s", 0) * 1e3 / row["n"]}
                key = (var, row["dir"])
                if key not in worst_by or row["median_ms"] > worst_by[key]["median_ms"]:
                    worst_by[key] = row
    st, lines = "PASS", []
    for var in ("LD_LIBRARY_PATH", "PATH"):
        rows = [(d, r) for (v, d), r in worst_by.items() if v == var]
        slow = sorted(((d, r) for d, r in rows if r["median_ms"] >= OPEN_FAIL_MS),
                      key=lambda t: -t[1]["median_ms"])
        if slow:
            # PATH is searched only for bare command names (XLA execs ptxas by absolute path), but
            # every rank's shell and its tools search it: a warning.
            st = worst(st, "FAIL" if var == "LD_LIBRARY_PATH" else "WARN")
            names = "; ".join(f"{d} ({r.get('fstype')}, {r['median_ms']:.1f} ms)" for d, r in slow)
            who = ("every exec (ptxas/nvlink per XLA module) and dlopen probes each of them for each library"
                   if var == "LD_LIBRARY_PATH" else
                   "every bare command name a rank's shell or tools run probes each of them")
            lines.append(f"slow {var}: {len(slow)} of {len(rows)} entries cost >= {OPEN_FAIL_MS} ms per failed "
                         f"lookup, and {who}: {names}. Drop them, or stage the libraries node-local")
        elif rows:
            mx = max(r["median_ms"] for _, r in rows)
            lines.append(f"{var}: {len(rows)} entries, failed lookup <= {mx:.3f} ms each")
    return st, lines


def judge_filesystem(fs0: dict, opens: list[dict], imports: list[dict], ref: dict,
                     lookups: list[dict] | None = None) -> dict:
    lines, st = [], "PASS"
    fix = stage_fix(fs0.get("launch"))
    s_lk, l_lk = judge_lookups(lookups or [])
    v = open_verdict(opens)
    warm = v.get("warm") or {}
    walk = ((fs0.get("walk") or {}).get("reps") or [{}, {}])[-1]
    met = {"open_warm_median_ms": warm.get("median_ms"), "open_warm_p90_ms": warm.get("p90_ms"),
           "open_first_median_ms": (v.get("first") or {}).get("median_ms"),
           "walk_per_s": walk.get("per_s")}
    imp = [i.get("total_s") for i in imports if i.get("total_s") is not None]
    met["import_s"] = max(imp) if imp else None
    s_imp, t_imp = vs_ref("import_s", met["import_s"], ref)
    med = warm.get("median_ms")
    if med is not None and med >= OPEN_FAIL_MS:
        st = "FAIL"
        g, s = max((v.get("groups") or {"all": warm}).items(), key=lambda t: t[1]["median_ms"])
        w = (v.get("where") or {}).get(g) or next(iter((v.get("where") or {}).values()), {})
        lines.append(f"slow filesystem: {med:.2f} ms per open (warm median, slowest rank; worst group {g} "
                     f"{s['median_ms']:.2f} ms) on {w.get('fstype')} at {w.get('mount')}; a cold run opens "
                     f"~10^4 install files serially; {fix}")
    elif med is not None and med > OPEN_PASS_MS:
        st = "WARN"
        lines.append(f"open latency {med:.2f} ms (warm median) is between PASS {OPEN_PASS_MS} and FAIL "
                     f"{OPEN_FAIL_MS} ms")
    wps = walk.get("per_s")
    if wps is not None and wps < WALK_FAIL_PER_S:
        w = fs0.get("walk") or {}
        if st != "FAIL":
            lines.insert(0, f"slow filesystem: venv walk+lstat {wps:.0f} entries/s < {WALK_FAIL_PER_S:.0f} on "
                         f"{w.get('fstype')} at {w.get('mount')}; {fix}")
        st = "FAIL"
    if s_imp == "FAIL":
        st = "FAIL"
        lines.append(f"slow cold import: {t_imp}")
    per_group = ", ".join(f"{g} {s['median_ms']:.3f}" for g, s in (v.get("groups") or {}).items())
    lines.append(f"open+close ms, slowest rank: warm median {fmt(med, '.3f')} p90 {fmt(warm.get('p90_ms'), '.2f')} "
                 f"(first touch {fmt(met['open_first_median_ms'], '.3f')}); per group warm: {per_group}")
    lines.append(f"venv walk+lstat {fmt(wps, '.0f')} entries/s (warm, {walk.get('entries')} entries); "
                 f"slowest-rank cold import {fmt(met['import_s'], '.1f')} s [{t_imp}]")
    seen = set()
    for g, w in (v.get("where") or {}).items():
        if w.get("mount") in seen:
            continue
        seen.add(w.get("mount"))
        lines.append(f"  mount {w.get('mount')}: {w.get('fstype')} {w.get('source')} "
                     f"[{(w.get('options') or '')[:110]}]")
    # .pyc: a read-only tree without bytecode recompiles every imported module in every process
    pfx = fs0.get("pycache_prefix")
    for i in imports[:1]:
        pyc = i.get("pyc") or {}
        if pyc.get("missing", 0) > 20:
            ro = "False" in (pyc.get("missing_dirs_writable") or []) or pyc.get("dont_write_bytecode")
            msg = (f"missing .pyc: {pyc['missing']} of {pyc['missing'] + pyc['have']} imported install "
                   f"modules have no bytecode (e.g. {', '.join(pyc.get('missing_sample', [])[:1])})")
            if ro:
                st = "FAIL"
                lines.insert(1 if lines and lines[0].startswith("slow filesystem") else 0,
                             msg + " on a read-only tree, so every process byte-compiles them: run "
                             "`python -m compileall -q <install>` at install time, or export "
                             "PYTHONPYCACHEPREFIX=<node-local dir, e.g. /tmp/$USER/pycache>")
            else:
                st = worst(st, "WARN")
                lines.append(msg + "; the first run writes them")
    if pfx:
        lines.append(f"PYTHONPYCACHEPREFIX={pfx.get('path')} on {pfx.get('fstype')}")
        if pfx.get("fstype") in NETWORK_FS:
            st = worst(st, "WARN")
            lines.append("PYTHONPYCACHEPREFIX is on a network filesystem: point it at node-local storage")
    if l_lk:
        if s_lk == "FAIL" and st != "FAIL":
            lines = l_lk + lines
        else:
            lines += l_lk
        st = worst(st, s_lk)
    return comp("1 filesystem", st, lines, met)


def fmt(v, spec: str) -> str:
    return "?" if v is None else format(v, spec)


def cache_misses(probe: dict) -> tuple[int | None, int | None]:
    """(misses, hits) of a probe's persistent-cache lookups: the runtime's own counters
    (``compile_cache_stats``: XLA compiles despite the cache, cache hits), else JAX's
    monitoring events."""
    s = probe.get("cache_stats") or {}
    if s.get("enabled") and s.get("probes") is not None:
        return int(s.get("compiles") or 0), int(s.get("hits") or 0)
    ev = probe.get("cache_events") or {}
    if not ev:
        return None, None
    return (sum(v for k, v in ev.items() if "miss" in k),
            sum(v for k, v in ev.items() if k.endswith("cache_hits")))


def judge_cache_latency(lats: list[dict]) -> tuple[str, list[str]]:
    """Per-file cost of the cache directories, slowest rank: the warm open median of existing
    entries or the mean repeated failed lookup, whichever is larger (same thresholds as the
    install's files)."""
    worst_by: dict = {}
    for lat in lats:
        for label, row in lat.items():
            miss = row.get("miss") or {}
            miss_ms = miss.get("total_s", 0) * 1e3 / miss["n"] if miss.get("n") else None
            vals = [v for v in ((row.get("open") or {}).get("median_ms"), miss_ms) if v is not None]
            if not vals:
                continue
            row = {**row, "cost_ms": max(vals), "miss_ms": miss_ms}
            if label not in worst_by or row["cost_ms"] > worst_by[label]["cost_ms"]:
                worst_by[label] = row
    st, lines = "PASS", []
    slow = {k: r for k, r in worst_by.items() if r["cost_ms"] > OPEN_PASS_MS}
    for label, r in slow.items():
        sev = "FAIL" if r["cost_ms"] >= OPEN_FAIL_MS else "WARN"
        st = worst(st, sev)
        knob = "ISDF_JAX_CACHE_DIR" if r.get("explicit") else "LORRAX_CACHE_ROOT"
        root = r["path"] if r.get("explicit") else os.path.dirname(r["path"].rstrip("/"))
        lines.insert(0, f"slow cache root: the {label} cache {r['path']} is on {r.get('fstype')} at {r.get('mount')}, "
                     f"{r['cost_ms']:.2f} ms per open or lookup (slowest rank; open "
                     f"{fmt((r.get('open') or {}).get('median_ms'), '.3f')} ms, failed lookup "
                     f"{fmt(r.get('miss_ms'), '.3f')} ms); a run looks up every XLA program it compiles (~2400 in the "
                     f"test suite) and opens every NVRTC image there. Fix: export {knob}=<a directory on a fast "
                     f"filesystem, e.g. Lustre scratch> in the launch shell (now {root}), then rerun lorrax-postinstall "
                     "so the kernel store is built there")
    if worst_by and not slow:
        lines.append("cache directories: " + ", ".join(
            f"{k} {r['cost_ms']:.3f} ms per open/lookup ({r.get('fstype')})" for k, r in worst_by.items()))
    return st, lines


def judge_caches(caches: dict, ranks: list[dict]) -> dict:
    lines, st = [], "PASS"
    for label in ("jax_compile", "cubin"):
        c = caches.get(label) or {}
        if not c:
            continue
        if not c.get("writable"):
            st = "FAIL"
            lines.append(f"cold or misnamed caches: {label} cache {c.get('path')} is not writable "
                         f"({c.get('error')}); every run recompiles. Point SCRATCH (or "
                         "ISDF_JAX_CACHE_DIR) at a writable fast filesystem")
        elif c.get("fstype") in ("nfs", "nfs4", "dvs", "gpfs") or (c.get("write_s") or 0) > 0.5:
            st = worst(st, "WARN")
            lines.append(f"{label} cache {c.get('path')} is on {c.get('fstype')} "
                         f"(1 MB write+fsync+read {fmt(c.get('write_s'), '.3f')} s)")
    st_lat, l_lat = judge_cache_latency([r.get("cache_latency") or {} for r in ranks])
    st = worst(st, st_lat)
    lines += l_lat
    ns = caches.get("namespace")
    if ns and not ns.endswith("_k3"):
        st = worst(st, "WARN")
        lines.append(f"compile-cache namespace {ns} is not content-keyed (k3): a module candidate and "
                     "its published copy will not share entries")
    r0 = next((r for r in ranks if r.get("rank") == 0), {})
    spr = r0.get("store_probe")
    run = ((r0.get("fs") or {}).get("launch") or {}).get("lorrax_run")
    build_cmd = (f"lorrax-postinstall (or srun -N 1 -n 1 -c 128 --gpus-per-node=4 {run} --all-gpus python "
                 f"{ROOT}/scripts/build_cubin_store.py --jobs 16)" if run else
                 f"srun -N 1 -n 1 --gpus-per-node=4 {shlex.quote(sys.executable)} "
                 f"{ROOT}/scripts/build_cubin_store.py --jobs 16")
    if spr is not None:
        sm = r0.get("sm")
        imgs = spr.get("images") or {}
        n_sm = sum(v.get("for_sm", 0) for v in imgs.values())
        log = spr.get("log") or []
        probe_built = [k for s, k in log if s == "built" and k == "2,1,1"]
        built = [k for s, k in log if s == "built"]
        where = ", ".join(f"{d}: {v.get('for_sm', 0)}/{v.get('n', 0)}" for d, v in imgs.items())
        if spr.get("error"):
            st = "FAIL"
            lines.append(f"cubin store probe failed: {spr['error'][:200]}")
        elif n_sm < MIN_STORE_IMAGES:
            st = "FAIL"
            lines.insert(0, f"cubin store missing: {n_sm} NVRTC images for sm_{sm} ({where}); every rank "
                         "NVRTC-builds each kernel on first use (~6 s each, ~10^4 header opens; the module's "
                         f"stage measured 259 s without images, 59 s with): run {build_cmd}")
        elif probe_built:
            st = "FAIL"
            lines.insert(0, f"cold or misnamed caches: {n_sm} images for sm_{sm} present but the router's probe "
                         "kernel was NVRTC-built, not loaded: the image key (toolchain, headers) does not "
                         f"match this install; rebuild: {build_cmd}")
        else:
            if n_sm < FULL_STORE_IMAGES:
                st = worst(st, "WARN")
                lines.append(f"cubin store incomplete: {n_sm} images for sm_{sm} (the full table is 389); "
                             f"kernels outside it NVRTC-build on first use: run {build_cmd}")
            lines.append(f"cubin images for sm_{sm}: {where}; probe kernels "
                         f"{'loaded from the cache' if not built else 'partly built: ' + ','.join(built)} "
                         f"({fmt(spr.get('s'), '.1f')} s)")
    # second-process cache hits
    for r in ranks:
        w = r.get("bench_warm") or {}
        misses, hits = cache_misses(w)
        if w and not w.get("error"):
            if misses:
                st = "FAIL"
                lines.append(f"cold or misnamed caches: rank {r['rank']}: a second process missed the "
                             f"persistent compile cache {misses} times (hits {hits}); the cache key is "
                             "unstable or the entries are not persisted")
                break
    if not lines:
        lines.append("compile cache and cubin cache writable on fast storage"
                     + (f"; namespace {ns}" if ns else ""))
    return comp("2 caches", st, lines)


def judge_kernels(kern: dict, ranks: list[dict], sm: int | None) -> dict:
    lines, st = [], "PASS"
    traps = kern.get("env_traps") or []
    for t in traps:
        st = "FAIL" if t["severity"] == "FAIL" else worst(st, "WARN")
        lines.append(f"JIT/PTX-only kernels: {t['text']}" if t["severity"] == "FAIL" else t["text"])
    for leg, inv in (kern.get("fatbin") or {}).items():
        if inv.get("error"):
            lines.append(f"{leg}: fatbin unreadable ({inv['error']})")
            continue
        if sm is not None and inv.get("n"):
            if sm not in inv.get("sass", []):
                st = "FAIL"
                lines.append(f"JIT/PTX-only kernels: {leg} has SASS for sm_{inv.get('sass')} and PTX for "
                             f"{inv.get('ptx')}, none for this GPU's sm_{sm}; the driver JIT-compiles every "
                             "kernel at load. Rebuild the FFI for this architecture")
            else:
                lines.append(f"{os.path.basename(leg)}: SASS sm_{sm} present ({inv['n']} fatbin entries)")
    for r in ranks[:1]:
        nv = (r.get("bench_cold") or {}).get("nvrtc") or {}
        if nv.get("error"):
            st = "FAIL"
            lines.append(f"NVRTC path broken: {nv['error'][:200]}")
        elif nv:
            archs = nv.get("new_cubin_archs") or []
            if not nv.get("new_cubins"):
                st = worst(st, "WARN")
                lines.append("NVRTC probe produced no new cubin (served from a store or not on mathdx: "
                             f"router {nv.get('router')})")
            elif sm is not None and archs and sm not in archs:
                st = "FAIL"
                lines.append(f"NVRTC cubin built for sm_{archs}, device is sm_{sm}")
            else:
                lines.append(f"NVRTC cubin built (sm_{archs})")
    px = kern.get("ptx_compiler") or {}
    if px:
        lines.append(px.get("text", ""))
        if px.get("severity") == "WARN":
            st = worst(st, "WARN")
    return comp("3 kernels", st, [l for l in lines if l])


def judge_libraries(ranks: list[dict], libs_static: dict) -> dict:
    lines, st = [], "PASS"
    r0 = (ranks[0].get("bench_cold") or {}) if ranks else {}
    ffi = r0.get("ffi") or {}
    for leg in ("CUDA", "cpu"):
        f = ffi.get(leg)
        if not f:
            continue
        if f.get("error"):
            st = "FAIL"
            lines.append(f"FFI {leg} leg refused: {f['error'][:300]}")
        elif f.get("closure_ok") is False:
            st = "FAIL"
            lines.append(f"FFI {leg} leg: mapped closure {f.get('closure')} differs from its record")
        else:
            lines.append(f"FFI {leg} leg opened + closure enforced in {f['open_s']:.2f} s "
                         f"({f.get('path')})")
            if f["open_s"] > 5.0:
                st = worst(st, "WARN")
                lines.append(f"slow closure gate: opening the {leg} leg took {f['open_s']:.1f} s "
                             "(library rehash on a slow filesystem?)")
    for d in libs_static.get("duplicates", []):
        st = "FAIL"
        lines.append(f"duplicate CUDA library mapped: {d}")
    for t in libs_static.get("ld_traps", []):
        st = worst(st, t["severity"])
        lines.append(t["text"])
    if not lines:
        lines.append("no FFI legs opened (none built?)")
    return comp("5 libraries", st, lines)


def judge_bench(ranks: list[dict], ref: dict | None, kind: str) -> dict:
    lines, st = [], "PASS"
    vals: dict = {}
    for r in ranks:
        c = r.get("bench_cold") or {}
        w = r.get("bench_warm") or {}
        if c.get("error"):
            why = [x.strip() for x in (c.get("log_tail") or "").splitlines()
                   if re.search(r"(Error|INTERNAL|Exception):", x)]
            return comp("6 benchmark", "FAIL", [f"rank {r['rank']} benchmark probe failed: {c['error']}"
                                                + (f" ({why[-1][:240]})" if why else "")])
        nv = c.get("nvrtc") or {}
        cand = {"compile_cold_s": c.get("compile_s"), "compile_warm_s": w.get("compile_s"),
                "s_per_program": c.get("s_per_program"),
                "nvrtc_s": max(nv.get("build_s") or [0]) or None,
                "gemm_tflops": c.get("gemm_tflops"), "fft_ms": c.get("fft_ms"),
                "host_gemm_s": c.get("host_gemm_s")}
        for k, v in cand.items():
            if v is None:
                continue
            if k == "gemm_tflops":
                vals[k] = min(vals.get(k, v), v)
            else:
                vals[k] = max(vals.get(k, v), v)
    if not ref:
        lines.append(f"no reference for {kind}: benchmark reported, not judged")
        st = "INFO"
    for k in ("compile_cold_s", "s_per_program", "compile_warm_s", "nvrtc_s", "gemm_tflops", "fft_ms",
              "host_gemm_s"):
        if k == "s_per_program":          # the same measurement as compile_cold_s, per program
            if vals.get(k) is not None:
                lines.append(f"cold XLA compile throughput {vals[k]:.3f} s/program (slowest rank)")
            continue
        s, t = vs_ref(k, vals.get(k), ref or {}, higher_better=(k == "gemm_tflops"))
        if ref and s == "FAIL":
            st = "FAIL"
        lines.append(("SLOW " if s == "FAIL" else "") + t)
    return comp("6 benchmark", st, lines, vals)


def verdict(components: list[dict]) -> str:
    return worst(*(c["status"] for c in components))


# ---------------------------------------------------------------------------
# rank-0 static checks
# ---------------------------------------------------------------------------
def env_traps() -> list[dict]:
    t = []
    if os.environ.get("CUDA_FORCE_PTX_JIT", "") not in ("", "0"):
        t.append({"severity": "FAIL", "text": "CUDA_FORCE_PTX_JIT is set: the driver ignores SASS and "
                  "JIT-compiles every kernel from PTX (or fails where no PTX is shipped); unset it"})
    if os.environ.get("CUDA_DISABLE_PTX_JIT", "") not in ("", "0"):
        t.append({"severity": "WARN", "text": "CUDA_DISABLE_PTX_JIT is set: kernels without SASS for "
                  "this GPU will fail to load"})
    if os.environ.get("CUDA_CACHE_DISABLE", "") not in ("", "0"):
        t.append({"severity": "WARN", "text": "CUDA_CACHE_DISABLE is set: any PTX JIT repeats in every "
                  "process"})
    if os.environ.get("CUDA_MODULE_LOADING", "").upper() == "EAGER":
        t.append({"severity": "WARN", "text": "CUDA_MODULE_LOADING=EAGER: every kernel of every library "
                  "loads at context creation"})
    xf = os.environ.get("XLA_FLAGS", "")
    for flag, why in (("xla_gpu_force_compilation_parallelism=1", "serial XLA compilation"),
                      ("xla_gpu_autotune_level=4", "exhaustive autotuning"),
                      ("xla_dump_to", "HLO dumping")):
        if flag in xf:
            t.append({"severity": "WARN", "text": f"XLA_FLAGS has {flag}: {why}"})
    if os.environ.get("JAX_DISABLE_JIT", "") not in ("", "0", "false", "False"):
        t.append({"severity": "FAIL", "text": "JAX_DISABLE_JIT is set: everything runs op by op"})
    ccp = os.environ.get("CUDA_CACHE_PATH")
    if ccp and fs_of(ccp)["fstype"] in ("nfs", "nfs4", "dvs"):
        t.append({"severity": "WARN", "text": f"CUDA_CACHE_PATH {ccp} is on {fs_of(ccp)['fstype']}"})
    return t


CUDA_FAMILIES = ("libcudart.so", "libcublas.so", "libcublasLt.so", "libcufft.so", "libcusolver.so",
                 "libcusparse.so", "libnccl.so", "libnvrtc.so", "libnvJitLink.so", "libcudnn.so")


def libs_static(mapped: list[str]) -> dict:
    dup = []
    for fam in CUDA_FAMILIES:
        paths = sorted({p for p in mapped if os.path.basename(p).startswith(fam)
                        and "builtins" not in p})
        if len(paths) > 1:
            dup.append(f"{fam}: {', '.join(paths)}")
    traps = []
    ld = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]
    missing = [d for d in ld if not os.path.isdir(d)]
    if missing:
        traps.append({"severity": "INFO", "text": f"LD_LIBRARY_PATH names {len(missing)} missing "
                      f"directories ({', '.join(missing[:3])})"})
    # slow LD_LIBRARY_PATH entries are measured (failed-lookup latency) in component 1
    for d in ld:
        hits = [f for f in CUDA_FAMILIES if glob.glob(os.path.join(d, f + ".*"))]
        if hits and "site-packages" not in d and "lorrax-launch" not in d and "cudart-hmem" not in d \
                and not any(os.path.realpath(m).startswith(os.path.realpath(d) + "/") for m in mapped):
            traps.append({"severity": "INFO", "text": f"LD_LIBRARY_PATH {d} provides {', '.join(hits)} "
                          "(not mapped by this process: harmless now, a trap for other launchers)"})
    return {"duplicates": dup, "ld_traps": traps}


def cache_location_check(path: str) -> dict:
    out = {"path": path, **fs_of(path if os.path.exists(path) else os.path.dirname(path) or "/")}
    try:
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, f".perfcheck.{socket.gethostname()}.{os.getpid()}")
        t0 = time.perf_counter()
        with open(fn, "wb") as fh:
            fh.write(os.urandom(1 << 20))
            fh.flush()
            os.fsync(fh.fileno())
        with open(fn, "rb") as fh:
            fh.read()
        out["write_s"] = time.perf_counter() - t0
        os.unlink(fn)
        out["writable"] = True
    except OSError as e:
        out["writable"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def user_cache_root(name: str) -> str:
    """``lxkit.user_cache_dir(name)``'s rule (mirrored: this file imports nothing of LORRAX):
    LORRAX_CACHE_ROOT, else ``$SCRATCH/.cache/lorrax`` when SCRATCH is a writable directory,
    else the XDG cache."""
    explicit = os.environ.get("LORRAX_CACHE_ROOT", "").strip()
    if explicit:
        return os.path.join(explicit, name)
    s = os.environ.get("SCRATCH", "")
    if s and os.path.isdir(s) and os.access(s, os.W_OK | os.X_OK):
        return os.path.join(s, ".cache", "lorrax", name)
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "lorrax", name)


def static_caches(sm: int | None, namespace: str | None) -> dict:
    explicit = os.environ.get("ISDF_JAX_CACHE_DIR")
    jcc = explicit.strip() if explicit and explicit.strip() else user_cache_root("jax_compile")
    out = {"jax_compile": cache_location_check(jcc),
           "cubin": cache_location_check(user_cache_root("kconv_mathdx")),
           "namespace": namespace}
    store = ROOT / "cubin_store"
    if store.is_dir():
        imgs = [str(p) for p in store.iterdir() if p.name.endswith(".cubin")]
        archs = {}
        for p in imgs[:400]:
            a = cubin_arch(p)
            archs[a] = archs.get(a, 0) + 1
        out["store"] = {"present": True, "path": str(store), "n": len(imgs), "archs": archs,
                        "sm": sm, "for_arch": (sm in archs) if sm is not None else bool(imgs)}
    else:
        out["store"] = {"present": False, "path": str(store)}
    return out


def ptx_compiler_info(children: list[str]) -> dict:
    """How XLA turned PTX into SASS in the cold compile probe: the ptxas children it
    exec'd (one per compiled module, each a file open + exec on the install's filesystem),
    or none (in-process libnvptxcompiler)."""
    ptxas = [c for c in children if os.path.basename(c).startswith(("ptxas", "nvlink"))]
    xf = os.environ.get("XLA_FLAGS", "")
    if ptxas:
        p = ptxas[0]
        f = fs_of(p)
        txt = f"XLA execs {', '.join(sorted({os.path.basename(c) for c in ptxas}))} per compiled module: {p} on {f['fstype']}"
        sev = "WARN" if f["fstype"] in NETWORK_FS - {"lustre"} else "INFO"
        return {"method": "ptxas", "path": p, "text": txt, "severity": sev}
    if "--xla_gpu_enable_libnvptxcompiler=true" in xf or children == []:
        return {"method": "libnvptxcompiler", "text": "XLA compiled PTX in-process (libnvptxcompiler; no "
                "ptxas child observed)", "severity": "INFO"}
    return {"method": "?", "text": f"compile children: {children[:3]}", "severity": "INFO"}




# ---------------------------------------------------------------------------
# rank coordination through a shared directory
# ---------------------------------------------------------------------------
def mark(out: Path, stage: str, rank: int) -> None:
    (out / f".{stage}.{rank}").write_text("")


def wait_all(out: Path, stage: str, nranks: int, timeout: float = 900.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all((out / f".{stage}.{r}").exists() for r in range(nranks)):
            return True
        time.sleep(0.2)
    return False


# ---------------------------------------------------------------------------
# main paths
# ---------------------------------------------------------------------------
#: The launcher's world variables (runtime._WORLD_SIZE_ENVS / _WORLD_RANK_ENVS): a probe is
#: one process on its rank's GPU, never a member of the step's world.
WORLD_ENVS = ("JAX_PROCESS_COUNT", "JAX_NUM_PROCESSES", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE",
              "PMI_SIZE", "SLURM_NTASKS", "SLURM_NPROCS", "JAX_PROCESS_INDEX", "OMPI_COMM_WORLD_RANK",
              "MV2_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID")


def single_process_env(base_env) -> dict:
    return {k: v for k, v in base_env.items() if k not in WORLD_ENVS}


def probe_env(base: Path, rank: int) -> dict:
    """Environment for a bench probe: fresh compile cache and fresh cubin cache."""
    env = single_process_env(os.environ)
    env["ISDF_JAX_CACHE_DIR"] = str(base / "jax")
    # The per-user cache root (lxkit.user_cache_root takes SCRATCH only if it is a writable
    # directory, so it must exist): a fresh, empty NVRTC image cache.
    (base / "scratch").mkdir(parents=True, exist_ok=True)
    env["SCRATCH"] = str(base / "scratch")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def rank_main(args, ctx: dict, topo: dict, out: Path) -> dict:
    rank, nranks = ctx["rank"], ctx["nranks"]
    res: dict = {"rank": rank, "host": ctx["host"], "binding": probe_binding(topo)}
    # 1. open latency on every rank at once (as a run's ranks start), then rank 0's venv walk.
    sample = open_sample()
    mark(out, "start", rank)
    wait_all(out, "start", nranks)
    res["open"] = probe_open_latency(sample)
    res["open"]["sample"] = len(sample)
    res["lookups"] = probe_path_lookups()
    res["cache_latency"] = probe_cache_latency()
    mark(out, "open", rank)
    wait_all(out, "open", nranks)
    if rank == 0:
        res["fs"] = {"paths": {label: {"path": p, **fs_of(p)} for label, p in (
            ("site-packages", str(site_packages())), ("source", str(ROOT)),
            ("python", os.path.realpath(sys.executable)))}}
        res["fs"]["launch"] = launch_info()
        res["fs"]["walk"] = probe_walk(str(site_packages()))
        pfx = os.environ.get("PYTHONPYCACHEPREFIX")
        res["fs"]["pycache_prefix"] = {"path": pfx, **fs_of(pfx)} if pfx else None
    mark(out, "fs", rank)
    wait_all(out, "fs", nranks)
    ienv = single_process_env(os.environ)
    if args.cpu_only:
        ienv["JAX_PLATFORMS"] = "cpu"     # a module's "cuda,cpu" would refuse on a CPU node
    imp, text, wall = run_probe(PROBE_IMPORT, [str(ROOT)], ienv, timeout=600)
    imp["wall_s"] = wall
    res["imports"] = imp
    mark(out, "imports", rank)
    wait_all(out, "imports", nranks)
    gpu = (imp.get("platform") in ("gpu", "cuda", "rocm")) and not args.cpu_only
    res["gpu"] = gpu
    if gpu:
        base = Path(tempfile.mkdtemp(prefix=f"lorrax_perfcheck_r{rank}_",
                                     dir=args.tmp or os.environ.get("TMPDIR") or "/tmp"))
        env = probe_env(base, rank)
        cold, text_c, wall_c = run_probe(PROBE_BENCH, ["cold"], env, timeout=900)
        cold["wall_s"] = wall_c
        cold.setdefault("nvrtc", {})["build_s"] = nvrtc_times(text_c)
        cd = (cold.get("nvrtc") or {}).get("cubin_dir")
        cold["nvrtc"]["new_cubin_archs"] = sorted({a for a in (
            cubin_arch(os.path.join(cd, c)) for c in cold["nvrtc"].get("new_cubins", [])) if a}) if cd else []
        if cold.get("error"):
            cold["log_tail"] = text_c[-3000:]
        res["bench_cold"] = cold
        mark(out, "cold", rank)
        wait_all(out, "cold", nranks)
        warm, text_w, wall_w = run_probe(PROBE_BENCH, ["warm"], env, timeout=600)
        warm["wall_s"] = wall_w
        if warm.get("error"):
            warm["log_tail"] = text_w[-3000:]
        res["bench_warm"] = warm
        if rank == 0:
            # The run's real NVRTC image cache (real SCRATCH): does a probe kernel load from it?
            senv = single_process_env(os.environ)
            senv["ISDF_JAX_CACHE_DIR"] = str(base / "jax_store")
            sp_, text_s, wall_s = run_probe(PROBE_STORE, [], senv, timeout=600)
            sp_["wall_s"] = wall_s
            sp_["log"] = kconv_log(text_s)
            if sp_.get("error"):
                sp_["log_tail"] = text_s[-2000:]
            res["store_probe"] = sp_
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
        else:
            res["probe_dir"] = str(base)
    if rank == 0:
        sm = None
        kinds = imp.get("devices") or []
        cc = imp.get("cc") or (res.get("bench_cold") or {}).get("cc")
        if cc:
            try:
                sm = int(str(cc).replace(".", ""))
            except ValueError:
                sm = None
        res["sm"] = sm
        res["device_kind"] = kinds[0] if kinds else None
        bc = res.get("bench_cold") or {}
        res["caches"] = static_caches(sm, bc.get("namespace"))
        spr = res.get("store_probe")
        if spr and not spr.get("error"):
            spr["images"] = count_images([spr.get("cubin_dir"), spr.get("store")], sm)
        fat = {}
        for leg in ("CUDA",):
            p = ((bc.get("ffi") or {}).get(leg) or {}).get("path") or os.environ.get("LORRAX_FFI_SO")
            if p and os.path.isfile(p):
                fat[p] = fatbin_inventory(p)
        children = bc.get("children") or []
        res["kernels"] = {"env_traps": env_traps(), "fatbin": fat,
                          "ptx_compiler": ptx_compiler_info(children) if gpu else {}}
        res["libs_static"] = libs_static(imp.get("mapped") or [])
    (out / f"rank{rank}.json").write_text(json.dumps(res, indent=1, default=str))
    mark(out, "done", rank)
    return res


def judge_all(ranks: list[dict], cpu_only: bool) -> tuple[str, list[dict], dict]:
    r0 = next(r for r in ranks if r["rank"] == 0)
    kind = r0.get("device_kind") or "?"
    ref = REFERENCE.get(kind)
    ref = ref if ref and any(v is not None for v in ref.values()) else None
    comps = [judge_filesystem(r0.get("fs") or {}, [r.get("open") or {} for r in ranks],
                              [r.get("imports") or {} for r in ranks], ref or {},
                              [r.get("lookups") or {} for r in ranks]),
             judge_caches(r0.get("caches") or {}, ranks)]
    gpu = all(r.get("gpu") for r in ranks)
    if gpu:
        comps.append(judge_kernels(r0.get("kernels") or {}, ranks, r0.get("sm")))
    else:
        comps.append(comp("3 kernels", "SKIP", ["no GPU in this step: skipped"]))
    b = judge_binding(ranks)
    fix = [f"fix: {b['fix']}"] if (b["fails"] or b["warns"]) else []
    comps.append(comp("4 binding", "FAIL" if b["fails"] else ("WARN" if b["warns"] else "PASS"),
                      [*b["fails"], *b["warns"], *fix, *b["rows"]]))
    if gpu:
        comps.append(judge_libraries(ranks, r0.get("libs_static") or {}))
        comps.append(judge_bench(ranks, ref, kind))
    else:
        comps.append(comp("5 libraries", "SKIP", ["no GPU in this step: FFI CUDA leg not opened"]))
        comps.append(comp("6 benchmark", "SKIP", ["no GPU in this step: skipped"]))
    return verdict(comps), comps, {"device_kind": kind, "reference": bool(ref), "gpu": gpu}



def report(v: str, comps: list[dict], meta: dict, ctx: dict, wall: float) -> str:
    out = [f"LORRAX perf check on {ctx['host']} ({ctx['nranks']} ranks, {meta.get('device_kind')}"
           f"{'' if meta.get('reference') else ', no reference'}) in {wall:.0f} s"]
    for c in comps:
        out.append(f"  [{c['status']:4}] {c['name']}")
        for line in c["lines"]:
            out.append(f"         {line}")
    fails = [c for c in comps if c["status"] == "FAIL"]
    if v == "FAIL":
        out.append("VERDICT: FAIL -- a run on this install/launch is expected to be slower than reference:")
        for c in fails:
            slow = [x for x in c["lines"] if x.startswith("SLOW ")]
            out.append(f"  {c['name']}: {'; '.join(slow) if slow else c['lines'][0]}")
    elif not meta.get("gpu"):
        out.append(f"VERDICT: {'PASS' if v != 'FAIL' else 'FAIL'} (CPU only: GPU components 3, 5, 6 skipped; "
                   "not a GPU certification)")
    else:
        out.append(f"VERDICT: {'PASS' if v in ('PASS', 'INFO', 'SKIP') else 'PASS with warnings'}")
    return "\n".join(out)


def static_main(args, ctx: dict, topo: dict, reason: str) -> int:
    """Login node / no GPU / not in a step: static checks + the command to run."""
    ngpu = args.gpus or 4
    entry = shlex.quote(str(HERE))
    say(f"NOT CERTIFIED: {reason}.")
    caches = static_caches(None, None)
    fat = {}
    for p in [os.environ.get("LORRAX_FFI_SO", "")] + glob.glob(str(ROOT / "src/ffi/cpp/build*/liblorrax_ffi.so")):
        if p and os.path.isfile(p):
            fat[p] = fatbin_inventory(p)
    for label in ("jax_compile", "cubin"):
        c = caches[label]
        say(f"  {label} cache {c['path']}: {'writable' if c.get('writable') else 'NOT writable'} on "
            f"{c.get('fstype')}")
    for p, inv in fat.items():
        say(f"  {p}: SASS {inv.get('sass')} PTX {inv.get('ptx')}")
    for t in env_traps():
        say(f"  {t['severity']}: {t['text']}")
    say(f"  install: {describe_fs(site_packages())} (site-packages), {describe_fs(ROOT)} (source)")
    o = probe_open_latency(open_sample())
    a = (o["passes"][-1].get("all") or {}) if o.get("passes") else {}
    if a.get("n"):
        say(f"  open latency HERE: {a['median_ms']:.3f} ms warm median over {a['n']} install files "
            f"({describe_fs(site_packages())}). Login nodes look fast on every filesystem (Perlmutter: "
            "~0.07 ms, while /global/common is a DVS mount on compute nodes at ~3.5 ms per open), so "
            "this does not certify compute nodes.")
    acct = os.environ.get("SBATCH_ACCOUNT") or os.environ.get("SLURM_ACCOUNT") or "<account>"
    srun = suggested_srun(ngpu, topo if not ctx["login_name"] else {"physical": 64}, entry)
    if ctx["job"] and not ctx["in_step"]:
        say("Inside an allocation: run the check as a step with your production launch line, e.g.")
        say(f"  {srun}")
    else:
        say("Run it on a GPU compute node with the srun line your production runs use, e.g.")
        say(f"  salloc -N 1 -C gpu -q interactive -t 00:15:00 --gpus-per-node={ngpu} -A {acct} \\")
        say(f"    {srun}")
    say("(srun ... perf_check.py --cpu-only on a CPU node checks components 1, 2 and 4 only.)")
    return 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="result directory (default $SCRATCH/.cache/lorrax/perfcheck/<job>.<step>)")
    ap.add_argument("--cpu-only", action="store_true", help="components 1, 2, 4 only")
    ap.add_argument("--gpus", type=int, help="GPUs per node a run will use (for the printed command)")
    ap.add_argument("--keep", action="store_true", help="keep the probe cache directories")
    ap.add_argument("--tmp", help="parent of the fresh probe caches (default $TMPDIR or /tmp)")
    ap.add_argument("--json", help="also write the verdict JSON here (rank 0)")
    ap.add_argument("--rejudge", metavar="DIR", help="re-judge a finished check's rank*.json with the "
                    "current thresholds and references (no measurement)")
    args = ap.parse_args(argv)
    if args.rejudge:
        ranks = [json.loads(p.read_text()) for p in sorted(Path(args.rejudge).glob("rank*.json"))]
        v, comps, meta = judge_all(ranks, False)
        print(report(v, comps, meta, {"host": ranks[0].get("host"), "nranks": len(ranks)}, 0.0))
        return 1 if v == "FAIL" else 0
    t0 = time.perf_counter()
    ctx = context()
    topo = node_topology()
    gpus = gpu_inventory()
    if not ctx["in_step"]:
        if ctx["login_name"] or not ctx["job"]:
            return static_main(args, ctx, topo, f"{ctx['host']} is not a compute-node job step "
                               "(no SLURM step, or a login host)")
        return static_main(args, ctx, topo, "not inside a job step")
    if not gpus and not args.cpu_only:
        if ctx["rank"] == 0:
            say("no GPU visible in this step: running the CPU-only components (1, 2, 4)")
        args.cpu_only = True
    job = f"{ctx['job']}.{os.environ.get('SLURM_STEP_ID', os.environ.get('SLURM_STEPID', '0'))}"
    out = Path(args.out or user_cache_root("perfcheck")) / (job if not args.out else "")
    out.mkdir(parents=True, exist_ok=True)
    if ctx["rank"] == 0:
        say(f"results in {out}")
    try:
        rank_main(args, ctx, topo, out)
    except Exception as e:                                         # noqa: BLE001
        import traceback
        (out / f"rank{ctx['rank']}.json").write_text(json.dumps(
            {"rank": ctx["rank"], "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}))
        mark(out, "done", ctx["rank"])
        say(f"rank {ctx['rank']}: the check itself failed: {type(e).__name__}: {e}")
        if ctx["rank"] != 0:
            return 2
    if ctx["rank"] != 0:
        return 0
    if not wait_all(out, "done", ctx["nranks"], timeout=1200):
        say("not every rank finished; judging the ranks that did")
    ranks = []
    for r in range(ctx["nranks"]):
        try:
            ranks.append(json.loads((out / f"rank{r}.json").read_text()))
        except (OSError, json.JSONDecodeError):
            pass
    broken = [r for r in ranks if r.get("error")]
    if broken or not any(r["rank"] == 0 for r in ranks):
        for r in broken:
            say(f"rank {r['rank']} broke: {r['error']}\n{r.get('tb', '')}")
        return 2
    v, comps, meta = judge_all(ranks, args.cpu_only)
    wall = time.perf_counter() - t0
    text = report(v, comps, meta, ctx, wall)
    print(text, flush=True)
    doc = {"verdict": v, "components": comps, "meta": meta, "wall_s": wall, "host": ctx["host"],
           "job": job}
    (out / "verdict.json").write_text(json.dumps(doc, indent=1, default=str))
    (out / "report.txt").write_text(text + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1, default=str))
    complete = len(ranks) == ctx["nranks"]
    stamp_install(doc, v, meta, complete and not args.cpu_only)
    return 1 if v == "FAIL" else 0


PROBE_STAMP = r"""
import sys
import runtime
print("STAMP " + str(runtime.install_verification_stamp(sys.argv[1])), flush=True)
"""


def stamp_install(doc: dict, v: str, meta: dict, eligible: bool) -> None:
    """Rank 0, after every rank's result is judged: a GPU PASS writes the verdict JSON at
    ``runtime.install_verification_stamp(device kind)`` (the runtime warns INSTALL NOT
    VERIFIED at every GPU startup until it exists); a FAIL removes it."""
    if not meta.get("gpu") or not meta.get("device_kind"):
        return
    try:
        cp = subprocess.run([sys.executable, "-c", PROBE_STAMP, str(meta["device_kind"])],
                            env=single_process_env(os.environ), capture_output=True, text=True,
                            timeout=300, cwd=tempfile.gettempdir())
        line = next((x for x in cp.stdout.splitlines() if x.startswith("STAMP ")), None)
    except (OSError, subprocess.SubprocessError):
        line = None
    if not line:
        say("this install's runtime has no install_verification_stamp: nothing recorded")
        return
    stamp = Path(line[len("STAMP "):].strip())
    if v == "FAIL":
        if stamp.exists():
            stamp.unlink()
            say(f"removed the verification stamp {stamp}")
        return
    if not eligible:
        say("not every rank reported: the install is not stamped as verified")
        return
    stamp.parent.mkdir(parents=True, exist_ok=True)
    tmp = stamp.with_name(f".{stamp.name}.{socket.gethostname()}.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, stamp)
    say(f"install verified for {meta['device_kind']}: {stamp}")


if __name__ == "__main__":
    sys.exit(main())
