"""Per-rank launch defaults for a plain ``srun python -m <driver>``, and the stale-override refusal.

THE FIRST THING A DRIVER PROCESS RUNS (``runtime/__init__.py`` calls :func:`apply` before any
other import), so a user who knows nothing about ``lorrax-run`` or ``select_gpu.sh`` gets what
those wrappers give::

    srun -N 1 -n 4 --gpus-per-node=4 python -m gw.gw_jax -i gw.in

1. GPU pinning (:func:`pin_gpu`).  One rank per GPU: the LOCALID-th entry of the list Slurm
   assigned the step, exactly ``src/ffi/cpp/select_gpu.sh``'s rule (``LORRAX_GPU_DEVICE``,
   then ``CUDA_VISIBLE_DEVICES``, ``SLURM_STEP_GPUS``, ``SLURM_JOB_GPUS``).  Only in a Slurm
   step with more than one task on this node, and never when the launcher already pinned the
   rank (one visible device, or ``ROCR_VISIBLE_DEVICES`` cut to one by ``--gpu-bind``).
2. CPU threads (:func:`thread_defaults`).  ``select_gpu.sh``'s shape: the affinity mask cut to
   one hardware thread per physical core, ``OMP_NUM_THREADS`` = cores - 2 (at least 1),
   OpenBLAS/MKL teams 1.  The plain step gives each Perlmutter rank 32 hyperthreads on 16
   cores, and 32-thread OpenMP teams cost ~10 s of a cold hsuite (ri_rootcause).  A caller
   that exported ``OMP_NUM_THREADS`` owns the threading, and a rank ``select_gpu.sh`` already
   set up (``LORRAX_RANK_THREADS``) is left alone.
3. Stale overrides (:func:`stale_overrides`).  An environment variable that points this
   process at ANOTHER tree than the one it imports -- an FFI library pin for another handler
   ABI, a pin to a file that is gone, a second LORRAX source directory on ``PYTHONPATH`` --
   refuses before anything is loaded, with the exact ``unset``/``export`` that fixes it.
   (Frontier, 2026-10: an ABI-10 ``LORRAX_FFI_HOST_SO`` inherited by sbatch from an old
   shell.)  ``LORRAX_ROOT`` naming another tree only warns: Python never reads it, but lx
   and the self-check do.  ``LORRAX_CHECKOUT`` is :mod:`runtime.source_closure`'s.
4. The import tree's file-open latency (:func:`import_tree_latency`): the venv (CUDA/MathDx
   headers, libdevice, ptxas, the Python packages) on a metadata-slow mount is the 3.5x
   slowdown of 2026-10-23 (DVS: every open a server round trip).  Refuses at the
   self-check's FAIL line, warns at its WARN line; the source tree alone only warns.

Steps 1-2 apply to any process that imports ``runtime`` inside a Slurm step with several
tasks on the node (a driver, or ``srun -n 4 python -m pytest tests/hsuite``): one device per
process is the only model LORRAX runs.  Steps 3-4 are for driver processes only
(``python -m <LORRAX package>...`` or a ``lorrax-*`` entry point).  ``LORRAX_RANK_SETUP=0``
turns all of it off, ``LORRAX_PREFLIGHT=warn`` turns the refusals into warnings.  Standard library only: this runs before numpy, jax or any
native library is mapped, which is what lets the thread and device variables bind.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

__all__ = ["apply", "pin_gpu", "thread_defaults", "stale_overrides", "import_tree_latency",
           "is_driver_process", "refuse_or_warn"]

#: Top-level packages of ``src/``: ``python -m <one of these>.<module>`` is a LORRAX driver.
DRIVER_PACKAGES = frozenset({"bandstructure", "bse", "centroid", "gw", "isdf", "postprocess",
                             "psp", "tdagw", "file_io", "common"})

#: The FFI library pins (``lxkit.legs``: ``so_env`` of each leg).
FFI_PIN_VARS = ("LORRAX_FFI_SO", "LORRAX_FFI_HOST_SO", "LORRAX_FFI_ROCM_SO")

#: Median open() latency lines, the self-check's (``scripts/perf_check.py``, per group).
OPEN_WARN_MS, OPEN_FAIL_MS = 0.5, 1.0

_TREE = Path(__file__).resolve().parents[2]          # <tree>/src/runtime/rank_setup.py
_ABI_HEADER = Path("src/ffi/cpp/common/lorrax_ffi_abi.h")
_DONE: dict = {}


def _say(msg: str) -> None:
    """One copy per step: rank 0 (or an unlaunched process) prints."""
    if os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")) == "0":
        print(msg, file=sys.stderr, flush=True)


def is_driver_process(argv=None) -> bool:
    """``python -m <LORRAX package>...`` or a ``lorrax-*`` console script; never pytest."""
    argv = list(getattr(sys, "orig_argv", None) or sys.argv) if argv is None else list(argv)
    for i, arg in enumerate(argv):
        if arg == "-m" and i + 1 < len(argv):
            return argv[i + 1].split(".")[0] in DRIVER_PACKAGES
        if arg.startswith("-m") and len(arg) > 2:
            return arg[2:].split(".")[0] in DRIVER_PACKAGES
    prog = os.path.basename(argv[1] if len(argv) > 1 and argv[0].endswith(("python", "python3"))
                            else (argv[0] if argv else ""))
    return prog.startswith("lorrax-") and "pytest" not in prog


def _in_slurm_step(env) -> bool:
    return any(str(env.get(k, "")).isdigit() for k in ("SLURM_STEP_ID", "SLURM_STEPID"))


def _tasks_on_this_node(env) -> int:
    """Tasks of this step on this node, from ``SLURM_STEP_TASKS_PER_NODE`` (``4(x2),3``)."""
    spec = env.get("SLURM_STEP_TASKS_PER_NODE") or env.get("SLURM_TASKS_PER_NODE") or ""
    counts: list[int] = []
    for part in filter(None, spec.split(",")):
        m = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", part.strip())
        if not m:
            counts = []
            break
        counts += [int(m.group(1))] * int(m.group(2) or 1)
    node = int(env.get("SLURM_NODEID", "0") or 0)
    if counts and node < len(counts):
        return counts[node]
    try:
        return -(-int(env["SLURM_STEP_NUM_TASKS"]) // max(1, int(env.get("SLURM_STEP_NUM_NODES", 1))))
    except (KeyError, ValueError):
        return 1


def _entries(value: str | None) -> list[str]:
    return [x for x in (value or "").split(",") if x != ""] if value else []


def pin_gpu(env=os.environ) -> str | None:
    """Export the one device this rank uses (``select_gpu.sh``'s rule); the line saying so."""
    if not _in_slurm_step(env) or _tasks_on_this_node(env) <= 1:
        return None
    if env.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        return None
    local = int(env.get("SLURM_LOCALID", "0") or 0)
    if env.get("LORRAX_GPU_DEVICE"):
        lst = _entries(env["LORRAX_GPU_DEVICE"])
        if local >= len(lst):
            refuse_or_warn([f"LORRAX_GPU_DEVICE={env['LORRAX_GPU_DEVICE']!r} has no entry for local "
                            f"rank {local}: list one device per rank on this node, or "
                            "`unset LORRAX_GPU_DEVICE`"], force=True)
        env["CUDA_VISIBLE_DEVICES"] = lst[local]
        return f"local rank {local} -> device {lst[local]} (LORRAX_GPU_DEVICE)"
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if var in env and len(_entries(env[var])) == 1:
            return None                                   # the launcher pinned this rank already
    for var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        lst = _entries(env.get(var))
        if local < len(lst):
            out = var if var in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES") else "CUDA_VISIBLE_DEVICES"
            env[out] = lst[local]
            return f"local rank {local} -> {out}={lst[local]} (from {var})"
    return None


def _physical_core_mask(cpus: set[int]) -> tuple[set[int], bool]:
    """The lowest hardware thread of each physical core in ``cpus``; ``False`` without topology."""
    keep: set[int] = set()
    for cpu in sorted(cpus):
        try:
            sib = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
        except OSError:
            return set(cpus), False
        lower = False
        for part in sib.strip().split(","):
            lo, _, hi = part.partition("-")
            if any(c in cpus for c in range(int(lo), int(hi or lo) + 1) if c < cpu):
                lower = True
        if not lower:
            keep.add(cpu)
    return keep, True


def thread_defaults(env=os.environ) -> str | None:
    """``select_gpu.sh``'s thread shape for a rank of a Slurm step; the line saying what ran."""
    if not _in_slurm_step(env) or env.get("LORRAX_RANK_THREADS"):
        return None
    if env.get("OMP_NUM_THREADS"):
        env["LORRAX_RANK_THREADS"] = (f"runtime: OMP_NUM_THREADS={env['OMP_NUM_THREADS']} exported "
                                      "by the caller; mask and threads unchanged")
        return None
    try:
        cpus = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        env["LORRAX_RANK_THREADS"] = "runtime: no affinity mask readable; threads unchanged"
        return None
    keep, topo = _physical_core_mask(cpus)
    if not topo:
        n, how = len(cpus), f"mask of {len(cpus)} CPUs left alone (no sysfs core topology)"
    elif keep == cpus:
        n, how = len(cpus), f"{len(cpus)} CPUs, one per core already"
    else:
        try:
            os.sched_setaffinity(0, keep)
            n, how = len(keep), f"mask cut from {len(cpus)} CPUs to {len(keep)}, one per physical core"
        except OSError:
            n, how = len(cpus), f"mask of {len(cpus)} CPUs left alone (sched_setaffinity refused)"
    env["OMP_NUM_THREADS"] = str(n - 2 if n > 3 else 1)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["LORRAX_RANK_THREADS"] = (f"runtime: {how}; OMP {env['OMP_NUM_THREADS']}, OpenBLAS "
                                  f"{env['OPENBLAS_NUM_THREADS']}, MKL {env['MKL_NUM_THREADS']}")
    return env["LORRAX_RANK_THREADS"]


def cudart_shim(env=os.environ) -> None:
    """The node-local ``libcudart.so`` link ``gpu_env.sh`` named (``<link>=<target>``)."""
    spec = env.get("LORRAX_CUDART_SHIM", "")
    if "=" not in spec:
        return
    link, target = spec.split("=", 1)
    try:
        if os.readlink(link) == target:
            return
    except OSError:
        pass
    try:
        os.makedirs(os.path.dirname(link), exist_ok=True)
        tmp = f"{link}.{os.getpid()}"
        os.symlink(target, tmp)
        os.replace(tmp, link)
    except OSError:
        pass


def _abi_of_tree(root: Path) -> int | None:
    try:
        m = re.search(r"#define\s+LORRAX_FFI_ABI_VERSION\s+(\d+)", (root / _ABI_HEADER).read_text())
        return int(m.group(1)) if m else None
    except OSError:
        return None


def _tree_of(path: Path) -> Path | None:
    """The LORRAX source tree a built library or a directory lies in (bounded walk up)."""
    for parent in list(path.parents)[:8]:
        if (parent / _ABI_HEADER).is_file() and (parent / "src" / "runtime" / "__init__.py").is_file():
            return parent
    return None


def _bundle_abi(lib: Path) -> int | None:
    """``ffi_abi`` of a sealed bundle manifest beside the library (lxkit ``_discover_manifest``)."""
    import json
    for d in (lib.parent, lib.parent.parent):
        try:
            return int(json.loads((d / "lorrax_ffi_bundle.json").read_text()).get("ffi_abi"))
        except (OSError, ValueError, TypeError):
            continue
    return None


def _venv_tree(prefix: str | None = None) -> Path | None:
    """The LORRAX tree whose ``.venv`` this interpreter is (``<tree>/.venv``), if any."""
    p = Path(prefix or sys.prefix)
    if p.name == ".venv" and (p.parent / "src" / "runtime" / "__init__.py").is_file():
        return p.parent
    return None


def stale_overrides(env=os.environ, tree: Path = _TREE,
                    venv_tree: Path | None | str = "auto") -> tuple[list[str], list[str]]:
    """``(refusals, warnings)``: environment overrides that point at another tree or ABI."""
    if venv_tree == "auto":
        venv_tree = _venv_tree()
    refuse: list[str] = []
    warn: list[str] = []
    abi = _abi_of_tree(tree)
    for var in FFI_PIN_VARS:
        value = env.get(var, "").strip()
        if not value:
            continue
        lib = Path(value)
        if not lib.is_file():
            refuse.append(f"{var}={value} is not a file (a pin left from another build): `unset {var}`")
            continue
        other = _tree_of(lib.resolve())
        found = _bundle_abi(lib) or (_abi_of_tree(other) if other and other != tree else None)
        if abi is not None and found is not None and found != abi:
            refuse.append(f"{var}={value} is an ABI-{found} build"
                          f"{f' from {other}' if other and other != tree else ''}; this tree "
                          f"({tree}) speaks ABI {abi}: `unset {var}`")
    pp = [x for x in env.get("PYTHONPATH", "").split(":") if x]
    others = []
    for entry in pp:
        p = Path(entry).expanduser()
        if (p / "runtime" / "__init__.py").is_file() and (p / "gw").is_dir():
            if p.resolve() != (tree / "src").resolve():
                others.append(entry)
    if others:
        kept = [x for x in pp if x not in others]
        fix = f"export PYTHONPATH={':'.join(kept)}" if kept else "unset PYTHONPATH"
        refuse.append(f"PYTHONPATH names another LORRAX source tree ({', '.join(others)}) than the "
                      f"one this process imports ({tree / 'src'}), so modules can come from both: `{fix}`")
    elif (venv_tree is not None and Path(venv_tree).resolve() != tree.resolve()
          and any(Path(x).expanduser().resolve() == (tree / "src").resolve() for x in pp)
          and Path(env.get("LORRAX_CHECKOUT", "") or "/nonexistent").resolve() != tree.resolve()):
        kept = [x for x in pp if Path(x).expanduser().resolve() != (tree / "src").resolve()]
        fix = f"export PYTHONPATH={':'.join(kept)}" if kept else "unset PYTHONPATH"
        refuse.append(f"PYTHONPATH makes the interpreter of {venv_tree} import the source of another tree "
                      f"({tree / 'src'}): `{fix}` (or, for a deliberate overlay, "
                      f"`export LORRAX_CHECKOUT={tree}`)")
    root = env.get("LORRAX_ROOT", "").strip()
    if root:
        r = Path(root).expanduser()
        if not r.exists():
            warn.append(f"LORRAX_ROOT={root} does not exist (left from an old install): `unset LORRAX_ROOT`")
        elif (r / "src" / "runtime" / "__init__.py").is_file() and r.resolve() != tree.resolve():
            warn.append(f"LORRAX_ROOT={root} is another LORRAX tree than this process imports ({tree}); "
                        "lx and lorrax-postinstall would use it: `unset LORRAX_ROOT` (or `module load` "
                        "the install you mean)")
    return refuse, warn


def _open_ms(paths: list[str]) -> float | None:
    times = []
    for p in paths:
        t0 = time.perf_counter()
        try:
            os.close(os.open(p, os.O_RDONLY))
        except OSError:
            continue
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2] if times else None


def _sample(root: Path, suffixes: tuple[str, ...], cap: int) -> list[str]:
    """Up to ``cap`` files of ``root`` (bounded: two directory levels, our own tree only)."""
    out: list[str] = []
    try:
        dirs = [root] + sorted(d for d in root.iterdir() if d.is_dir())[:40]
    except OSError:
        return out
    for d in dirs:
        try:
            out += [str(f) for f in sorted(d.iterdir())[:20] if f.suffix in suffixes]
        except OSError:
            continue
        if len(out) >= cap:
            break
    return out[:cap]


def import_tree_latency(tree: Path = _TREE) -> tuple[list[str], list[str]]:
    """``(refusals, warnings)`` from the median open() latency of the venv and the source."""
    refuse: list[str] = []
    warn: list[str] = []
    venv = Path(sys.prefix)
    site = next(iter(sorted(venv.glob("lib/python3*/site-packages"))), None)
    groups = []
    if site is not None:
        groups.append(("venv", site, _sample(site / "nvidia", (".h", ".hpp", ".py"), 12)
                       + _sample(site / "jax", (".py",), 12) + _sample(site / "numpy", (".py",), 6)))
    groups.append(("source", tree / "src", _sample(tree / "src", (".py",), 24)))
    for name, where, files in groups:
        ms = _open_ms(files)
        if ms is None:
            continue
        line = (f"the {name} at {where} answers open() in {ms:.2f} ms (median of {len(files)}); "
                f"a cold run opens ~10^4 such files, so this filesystem costs ~{ms * 10:.0f} s per run")
        if name == "venv" and ms >= OPEN_FAIL_MS:
            refuse.append(line + ": run from the install's runtime image (`module load` it, or "
                          "`source <prefix>/lorrax-env.sh`), or install on a fast filesystem")
        elif ms >= OPEN_WARN_MS:
            warn.append(line)
    return refuse, warn


def refuse_or_warn(refusals: list[str], warnings: list[str] = (), *, force: bool = False) -> None:
    """Print each finding on one line; exit 2 on a refusal unless ``LORRAX_PREFLIGHT=warn``."""
    soft = os.environ.get("LORRAX_PREFLIGHT", "").strip().lower() == "warn" and not force
    for w in warnings:
        _say(f"[lorrax preflight] WARNING: {w}")
    for r in refusals:
        _say(f"[lorrax preflight] {'WARNING' if soft else 'REFUSED'}: {r}")
    if refusals and not soft:
        _say("[lorrax preflight] (LORRAX_PREFLIGHT=warn runs anyway)")
        sys.stdout.flush()
        os._exit(2)


def apply() -> None:
    """Steps 1-4 of the module docstring, once per process."""
    if _DONE or os.environ.get("LORRAX_RANK_SETUP", "").strip().lower() in ("0", "off", "false", "no"):
        return
    _DONE["at"] = time.time()
    if _in_slurm_step(os.environ) and _tasks_on_this_node(os.environ) > 1:
        _DONE["pin"] = pin_gpu()
        _DONE["threads"] = thread_defaults()
        cudart_shim()
    if not is_driver_process():
        return
    if "threads" not in _DONE:
        _DONE["threads"] = thread_defaults()
        cudart_shim()
    refuse, warn = stale_overrides()
    r2, w2 = import_tree_latency()
    refuse_or_warn(refuse + r2, warn + w2)
