#!/usr/bin/env python3
"""Dependency smoke tests: every build of a LORRAX dependency ends here, and fails by name.

    dep_check.py blas  <libopenblas.so> [--expect-threads N]
    dep_check.py elpa  <prefix> --blas <libopenblas.so the closure must bind> [--solve]
    dep_check.py mpi-allreduce          (under `srun -n >=2`, cpu_mpi_env.sh sourced)
    dep_check.py nvidia [--site-packages DIR] [--ffi <liblorrax_ffi.so>]
    dep_check.py cholesky               (under `srun -n 4` on one GPU node: the LORRAX runtime)
    dep_check.py ffi   <liblorrax_ffi*.so> [--arch sm_80] [--root DIR ...]

Each check prints ``[dep-check] <dependency>: OK <what was shown>`` or one line
``[dep-check] FAILED <dependency>: <what is wrong> -- fix: <the fix>`` and exits 1.  A check
that cannot run here (no MPI launcher on a login node) prints ``SKIPPED`` with where it runs
instead, and exits 0.

What each one adds to the gates that already exist (it does not repeat them):
* blas -- the LAPACK symbols the legs call, whether the library honours its thread variable
  (a sequential build reports 1 under OPENBLAS_NUM_THREADS=2, a threaded one 2), and a small
  dgemm and dsyev checked in pure Python (no second BLAS involved).  Frontier
  build_openblas_seq.sh.
* elpa -- the closure in an EMPTY environment (``env -i``: the OpenBLAS the library binds
  must be the expected one, found through its own RUNPATH/RPATH), ``ldd -r`` in the closure
  record's directories (no missing library, no undefined symbol), and with --solve a 64x64 ELPA
  1-stage eigensolve against LAPACK dsyev from the same OpenBLAS.  Frontier
  build_elpa_hip.sh (the closure record and NEEDED gates stay there) and verify.sh.
* mpi-allreduce -- JAX's CPU collectives through MPItrampoline -> MPIwrapper -> Cray MPICH, the
  exact consumer path: a cross-process allgather on >= 2 ranks.  The MPI ABI version
  handshake is build_mpiwrapper.sh's own (mpiabi_version); this is the live half.
* nvidia -- cuSOLVERMp, cuBLASMp and NCCL: the header the FFI compiles against states the
  installed wheel's version, header and library are files of that one wheel (its RECORD),
  NCCL's live ncclGetVersion equals NCCL_VERSION_CODE, and (--ffi) the FFI binds that copy.
  (cusolverMpGetVersion needs a live handle; the wheel pins are build_ffi_cuda.sh's.)
* cholesky -- a 64x64 distributed cuSOLVERMp Cholesky on the run's mesh against numpy.
  lorrax-postinstall.
* ffi -- the CUDA leg's cubin architectures (cuobjdump) include the site GPU's, and no
  library of the closure resolves outside the install (from $HOME, CFS, /global/common or
  $SCRATCH, unless under --root).  ABI, symbols, one MPI/HDF5/BLAS and the closure record
  stay scripts/verify_ffi_build.sh's.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_TAG = "[dep-check]"


class Failed(Exception):
    def __init__(self, dep: str, what: str, fix: str):
        super().__init__(f"{_TAG} FAILED {dep}: {what} -- fix: {fix}")


def ok(dep: str, what: str) -> None:
    print(f"{_TAG} {dep}: OK {what}", flush=True)


def skipped(dep: str, why: str) -> None:
    print(f"{_TAG} {dep}: SKIPPED {why}", flush=True)


def _run(cmd, env=None, timeout=120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=timeout)


def _dynamic(lib: str) -> dict:
    """``readelf -d``: NEEDED list, RPATH, RUNPATH."""
    out = _run(["readelf", "-d", lib]).stdout
    get = lambda tag: [l.split("[", 1)[1].rstrip("]") for l in out.splitlines() if f"({tag})" in l]
    return {"needed": get("NEEDED"), "rpath": get("RPATH"), "runpath": get("RUNPATH")}


def _ldd(lib: str, env=None, relocs=False) -> tuple[dict, list[str]]:
    """``{soname: path or None}`` and the ``undefined symbol`` lines of ``ldd [-r]``."""
    p = _run(["ldd"] + (["-r"] if relocs else []) + [lib], env=env)
    libs, undef = {}, []
    for line in (p.stdout + p.stderr).splitlines():
        line = line.strip()
        if "undefined symbol" in line:
            undef.append(line)
        m = re.match(r"(\S+) => (not found|(\S+))", line)
        if m:
            libs[m.group(1)] = None if m.group(2) == "not found" else m.group(3)
    return libs, undef


def _bound(so: str) -> tuple[dict, str]:
    """``{soname: path}`` the leg binds at run time, and where that came from: its closure
    record (``<so>.closure.json``, the exact files the loader preloads before dlopen, so
    LD_LIBRARY_PATH cannot change them) when there is one, else ``ldd``."""
    record = Path(so + ".closure.json")
    if record.is_file():
        import json
        libs = json.loads(record.read_text()).get("libraries", [])
        return {os.path.basename(e["path"]): e["path"] for e in libs if e.get("path")}, "closure record"
    return _ldd(so)[0], "ldd"


# --------------------------------------------------------------------- blas --
_BLAS_SYMBOLS = ("dgemm_", "zgemm_", "dsyev_", "zheev_", "dpotrf_", "dgetrf_", "dgesv_")


def _matmul(a, b, m, n, k):          # column-major reference, pure Python
    return [sum(a[i + p * m] * b[p + j * k] for p in range(k)) for j in range(n) for i in range(m)]


def check_blas(lib: str, expect_threads: int | None) -> None:
    dep = f"OpenBLAS ({lib})"
    try:
        h = ctypes.CDLL(lib)
    except OSError as exc:
        raise Failed(dep, f"does not load: {exc}", "rebuild it, or point the build at the right file")
    missing = [s for s in _BLAS_SYMBOLS if not hasattr(h, s)]
    if missing:
        raise Failed(dep, f"lacks {', '.join(missing)}", "build with LAPACK (`make ... libs netlib shared`, "
                     "NO_LAPACK unset); the legs call these")
    probe = ("import ctypes,sys; h=ctypes.CDLL(sys.argv[1]); "
             "f=getattr(h,'openblas_get_num_threads',None); g=getattr(h,'openblas_get_parallel',None); "
             "print(f() if f else -1, g() if g else -1)")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    got = _run([sys.executable, "-c", probe, lib], env=env).stdout.split()
    nthreads, parallel = (int(got[0]), int(got[1])) if len(got) == 2 else (-1, -1)
    want = expect_threads if expect_threads is not None else (1 if parallel == 0 else 2)
    if nthreads != -1 and nthreads != want:
        raise Failed(dep, f"under OPENBLAS_NUM_THREADS=2 it runs {nthreads} threads (parallel mode "
                     f"{parallel}), expected {want}", "build with the USE_THREAD/USE_OPENMP the site "
                     "file states (sequential: USE_THREAD=0 USE_OPENMP=0)")
    m, n, k = 7, 5, 6
    a = [((3 * i + 1) % 11) / 7.0 - 0.5 for i in range(m * k)]
    b = [((5 * i + 2) % 13) / 9.0 - 0.3 for i in range(k * n)]
    c = (ctypes.c_double * (m * n))()
    i = lambda v: ctypes.byref(ctypes.c_int(v))
    d = lambda v: ctypes.byref(ctypes.c_double(v))
    h.dgemm_(ctypes.c_char_p(b"N"), ctypes.c_char_p(b"N"), i(m), i(n), i(k), d(1.0),
             (ctypes.c_double * len(a))(*a), i(m), (ctypes.c_double * len(b))(*b), i(k), d(0.0), c, i(m))
    err = max(abs(x - y) for x, y in zip(c, _matmul(a, b, m, n, k)))
    if err > 1e-12:
        raise Failed(dep, f"dgemm differs from the reference by {err:.2e}", "the library is miscompiled "
                     "for this CPU (TARGET=); rebuild for the node's architecture")
    nn = 6
    sym = [1.0 / (1 + abs(r - s)) + (nn if r == s else 0) for s in range(nn) for r in range(nn)]
    w = (ctypes.c_double * nn)()
    work = (ctypes.c_double * 256)()
    info = ctypes.c_int(0)
    am = (ctypes.c_double * len(sym))(*sym)
    h.dsyev_(ctypes.c_char_p(b"V"), ctypes.c_char_p(b"U"), i(nn), am, i(nn), w, work, i(256),
             ctypes.byref(info))
    resid = 0.0
    for j in range(nn):
        v = [am[r + j * nn] for r in range(nn)]
        av = [sum(sym[r + s * nn] * v[s] for s in range(nn)) for r in range(nn)]
        resid = max(resid, max(abs(av[r] - w[j] * v[r]) for r in range(nn)))
    if info.value != 0 or resid > 1e-12 or abs(sum(w) - sum(sym[r + r * nn] for r in range(nn))) > 1e-12:
        raise Failed(dep, f"dsyev info={info.value}, max |Av - wv| = {resid:.2e}", "LAPACK in this "
                     "build is broken; rebuild (and run its own `make tests`)")
    ok(dep, f"{len(_BLAS_SYMBOLS)} BLAS/LAPACK symbols; {nthreads} thread(s) under OPENBLAS_NUM_THREADS=2 "
            f"(parallel mode {parallel}); dgemm err {err:.1e}; dsyev residual {resid:.1e}")


# --------------------------------------------------------------------- elpa --
_ELPA_C = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <mpi.h>
#include <elpa/elpa.h>
typedef void (*dsyev_t)(const char*, const char*, const int*, double*, const int*, double*, double*,
                        const int*, int*);
int main(int argc, char **argv) {
    int n = 64, err = 0, provided = 0, lwork = 64 * 66, info = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    double *a = malloc(sizeof(double) * n * n), *b = malloc(sizeof(double) * n * n);
    double *z = malloc(sizeof(double) * n * n), *ev = malloc(sizeof(double) * n);
    double *w = malloc(sizeof(double) * n), *work = malloc(sizeof(double) * lwork);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            a[i + j * n] = b[i + j * n] = 1.0 / (1.0 + abs(i - j)) + (i == j ? 0.01 * i : 0.0);
    void *blas = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    dsyev_t dsyev = blas ? (dsyev_t)dlsym(blas, "dsyev_") : NULL;
    if (!dsyev) { printf("DSYEV_MISSING %s\n", dlerror()); return 3; }
    dsyev("N", "U", &n, b, &n, w, work, &lwork, &info);
    if (elpa_init(20171201) != ELPA_OK) { printf("ELPA_INIT_FAILED\n"); return 4; }
    elpa_t h = elpa_allocate(&err);
    elpa_set(h, "na", n, &err);            elpa_set(h, "nev", n, &err);
    elpa_set(h, "local_nrows", n, &err);   elpa_set(h, "local_ncols", n, &err);
    elpa_set(h, "nblk", 16, &err);
    elpa_set(h, "mpi_comm_parent", (int)MPI_Comm_c2f(MPI_COMM_WORLD), &err);
    elpa_set(h, "process_row", 0, &err);   elpa_set(h, "process_col", 0, &err);
    if (elpa_setup(h) != ELPA_OK) { printf("ELPA_SETUP_FAILED\n"); return 5; }
    elpa_set(h, "solver", ELPA_SOLVER_1STAGE, &err);
    elpa_eigenvectors(h, a, ev, z, &err);
    if (err != ELPA_OK) { printf("ELPA_SOLVE_FAILED %d\n", err); return 6; }
    double d = 0.0;
    for (int i = 0; i < n; i++) d = fmax(d, fabs(ev[i] - w[i]));
    printf("MAXDIFF %.3e INFO %d\n", d, info);
    MPI_Finalize();
    return 0;
}
"""


def _elpa_lib(prefix: Path) -> Path:
    libs = sorted(p for p in (prefix / "lib").glob("libelpa*.so*") if "-gdb" not in p.name)
    real = [p for p in libs if not p.is_symlink()] or libs
    if not real:
        raise Failed(f"ELPA ({prefix})", "no lib/libelpa*.so", "the build did not install; read its log")
    return real[0]


def check_elpa(prefix: str, blas: str, solve: bool) -> None:
    pfx = Path(prefix).resolve()
    lib = _elpa_lib(pfx)
    dep = f"ELPA ({lib})"
    dyn = _dynamic(str(lib))
    search = f"RUNPATH {dyn['runpath']}" if dyn["runpath"] else f"RPATH {dyn['rpath']}" if dyn["rpath"] else "no RUNPATH/RPATH"
    bare, _ = _ldd(str(lib), env={"PATH": "/usr/bin:/bin"})          # env -i
    want = os.path.realpath(blas)
    got = {k: v for k, v in bare.items() if k.startswith(("libopenblas", "libblas", "liblapack", "libsci"))}
    if not got:
        raise Failed(dep, f"binds no BLAS by name ({search})", "link ELPA against the leg's OpenBLAS "
                     "(-L$P/lib -lopenblas, rpath $PREFIX/lib)")
    wrong = {k: v for k, v in got.items() if v is None or os.path.realpath(v) != want}
    if wrong:
        raise Failed(dep, f"in an empty environment (env -i) its BLAS resolves to {wrong} through {search}, "
                     f"not {want}", "build under the same environment the run uses and link with "
                     "-Wl,-rpath,<prefix>/lib so the private OpenBLAS copy is the one bound; never let a "
                     "build-time RPATH name another OpenBLAS")
    # The run's search path: the directories the build's closure record names
    # (LORRAX_ELPA_CLOSURE.json: private, pinned runtime and venv ROCm dirs) before the caller's.
    run_env = dict(os.environ)
    record = pfx / "LORRAX_ELPA_CLOSURE.json"
    where = "this environment"
    if record.is_file():
        import json
        dirs = list(dict.fromkeys(os.path.dirname(e["path"]) for e in json.loads(record.read_text())["libraries"]))
        run_env["LD_LIBRARY_PATH"] = ":".join(dirs + [x for x in run_env.get("LD_LIBRARY_PATH", "").split(":") if x])
        where = f"the closure record's {len(dirs)} directories"
    full, undef = _ldd(str(lib), env=run_env, relocs=True)
    missing = [k for k, v in full.items() if v is None]
    if missing or undef:
        raise Failed(dep, f"ldd -r in {where}: missing {missing or 'none'}; "
                     f"{len(undef)} undefined symbol(s){': ' + undef[0] if undef else ''}",
                     "rebuild the prefix (its closure record no longer resolves), or load the site's "
                     "runtime modules (ROCm, cray-mpich) the record pins")
    ok(dep, f"{search}; env -i binds {list(got.values())[0]}; ldd -r clean in {where} ({len(full)} libraries)")
    if not solve:
        return
    cc = os.environ.get("LORRAX_DEP_CHECK_CC") or shutil.which("cc") or shutil.which("mpicc")
    inc = sorted(pfx.glob("include/elpa*"))
    if not cc or not inc:
        raise Failed(dep, f"cannot build the eigensolve test (compiler {cc}, include {inc})",
                     "load the programming environment that built ELPA (cc wrapper)")
    with tempfile.TemporaryDirectory(prefix="dep_check_elpa_") as td:
        src, exe = Path(td) / "t.c", Path(td) / "t"
        src.write_text(_ELPA_C)
        name = re.sub(r"^lib|\.so.*$", "", lib.name)
        p = _run([cc, "-O1", str(src), "-o", str(exe), f"-I{inc[0]}", f"-L{lib.parent}", f"-l{name}",
                  f"-Wl,-rpath,{lib.parent}", "-ldl", "-lm"], timeout=300)
        if p.returncode != 0:
            raise Failed(dep, f"the 64x64 test does not link: {(p.stderr or p.stdout).strip().splitlines()[-1:]}",
                         "the installed headers and library disagree; rebuild the prefix")
        cmd = [str(exe), want]
        if os.environ.get("SLURM_JOB_ID") and not os.environ.get("SLURM_STEP_ID") and shutil.which("srun"):
            cmd = ["srun", "-N", "1", "-n", "1", "--overlap"] + cmd
        r = _run(cmd, env=run_env, timeout=300)
    m = re.search(r"MAXDIFF (\S+) INFO (-?\d+)", r.stdout)
    if not m:
        text = (r.stdout + r.stderr).strip()
        if not os.environ.get("SLURM_JOB_ID") and re.search(r"PMI|pmi|MPI.*[Ii]nit", text):
            skipped(dep, "64x64 eigensolve: MPI cannot start on this node without a launcher; "
                         "it runs in the site's verify.sh")
            return
        raise Failed(dep, f"64x64 eigensolve did not finish (rc {r.returncode}): {text.splitlines()[-1:] }",
                     "run it by hand with LD_DEBUG=libs to see what it loads")
    diff = float(m.group(1))
    if diff > 1e-10 or m.group(2) != "0":
        raise Failed(dep, f"64x64 1-stage eigenvalues differ from LAPACK dsyev by {diff:.2e} (info {m.group(2)})",
                     "ELPA's kernels or its BLAS are wrong for this CPU; rebuild with the generic kernel")
    ok(dep, f"64x64 1-stage eigensolve matches LAPACK dsyev to {diff:.1e}")


# ------------------------------------------------------------ mpi-allreduce --
def check_mpi_allreduce() -> None:
    dep = f"MPIwrapper/MPItrampoline ({os.environ.get('MPITRAMPOLINE_LIB', 'MPITRAMPOLINE_LIB unset')})"
    if not os.environ.get("MPITRAMPOLINE_LIB"):
        raise Failed(dep, "MPITRAMPOLINE_LIB is not set", "source config/perlmutter/cpu_mpi_env.sh first")
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if ntasks < 2:
        raise Failed(dep, "needs >= 2 ranks", "run under `srun -n 2`")
    try:
        import jax
        import numpy as np
        jax.config.update("jax_cpu_collectives_implementation", "mpi")
        jax.distributed.initialize()
        from jax.experimental import multihost_utils
        got = np.asarray(multihost_utils.process_allgather(np.array([jax.process_index() + 1.0])))
    except Exception as exc:                                  # noqa: BLE001
        raise Failed(dep, f"cross-process collective failed: {type(exc).__name__}: {str(exc)[:300]}",
                     "rebuild the adapter (config/perlmutter/build_mpiwrapper.sh) against the loaded "
                     "cray-mpich, and check `MPITRAMPOLINE_VERBOSE=1`")
    want = ntasks * (ntasks + 1) / 2
    if abs(float(got.sum()) - want) > 1e-12:
        raise Failed(dep, f"allgather of rank+1 summed to {got.sum()} on {ntasks} ranks, want {want}",
                     "the adapter's MPI handles disagree with the MPI it wraps: rebuild it")
    if jax.process_index() == 0:
        ok(dep, f"{ntasks}-rank JAX CPU collective (allgather) through MPItrampoline: sum {got.sum():g}")


# ------------------------------------------------------------------- nvidia --
_NV = {   # library: (header, macro prefix, library glob, distribution, live version symbol)
    "cuSOLVERMp": ("cusolverMp.h", "CUSOLVERMP", "libcusolverMp.so*", "nvidia_cusolvermp", None),
    "cuBLASMp": ("cublasmp.h", "CUBLASMP", "libcublasmp.so*", "nvidia_cublasmp", None),
    "NCCL": ("nccl.h", "NCCL", "libnccl.so*", "nvidia_nccl", "ncclGetVersion"),
}


def _header_version(header: Path, pfx: str) -> tuple[int, int, int] | None:
    text = header.read_text(errors="replace")
    parts = []
    for part in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"#define\s+{pfx}_(?:VER_)?{part}\s+(\d+)", text)
        if not m:
            return None
        parts.append(int(m.group(1)))
    return tuple(parts)


def _wheel(sp: Path, dist: str):
    """``(version, files)`` of the installed wheel whose name starts with ``dist``."""
    for info in sorted(sp.glob(f"{dist}*.dist-info")):
        version = info.name[len(info.name.split("-")[0]) + 1:-len(".dist-info")]
        try:
            files = {str((sp / line.split(",")[0]).resolve()) for line in
                     (info / "RECORD").read_text().splitlines() if line}
        except OSError:
            files = set()
        return version, files
    return None, set()


def check_nvidia(site: str | None, ffi: str | None) -> None:
    """Header version == the wheel's version, header and library from that one wheel (its RECORD),
    NCCL's live ``ncclGetVersion`` == ``NCCL_VERSION_CODE``, and (--ffi) the FFI binds that file."""
    if site is None:
        import sysconfig
        site = sysconfig.get_paths()["purelib"]
    sp = Path(site)
    resolved, how = _bound(ffi) if ffi else ({}, "")
    for dep, (hdr, pfx, libglob, dist, live) in _NV.items():
        version, files = _wheel(sp, dist)
        headers = sorted(p for p in (sp / "nvidia").glob(f"**/include/{hdr}"))
        libs = sorted(p for p in (sp / "nvidia").glob(f"**/lib/{libglob}") if p.is_file() and not p.is_symlink())
        if version is None or not headers or not libs:
            raise Failed(dep, f"wheel {dist}* {version or 'not installed'}, header {hdr} "
                         f"{'found' if headers else 'missing'}, library {libglob} {'found' if libs else 'missing'} "
                         f"under {sp}", "uv sync --extra cuda13 (the wheels the FFI is built against)")
        hv = _header_version(headers[0], pfx)
        wv = tuple(int(x) for x in re.findall(r"\d+", version)[:3])
        if hv is None or hv != wv:
            raise Failed(dep, f"header {headers[0]} says {hv}, the installed wheel is {version}",
                         f"reinstall the wheel (uv sync --reinstall-package {dist.replace('_', '-')}-cu13) "
                         "and rebuild the CUDA leg")
        stray = [str(p) for p in (headers[0], libs[0]) if files and str(p.resolve()) not in files]
        if stray:
            raise Failed(dep, f"{stray} not installed by the {version} wheel (another release copied in)",
                         "remove the copied file and reinstall the wheel")
        shown = f"header {'.'.join(map(str, hv))} == wheel {version}; header and {libs[0].name} from it"
        if live:
            try:
                v = ctypes.c_int(0)
                rc = getattr(ctypes.CDLL(str(libs[0])), live)(ctypes.byref(v))
            except (OSError, AttributeError) as exc:
                raise Failed(dep, f"{libs[0]} does not load or has no {live}: {exc}", "reinstall the wheel")
            code = hv[0] * 10000 + hv[1] * 100 + hv[2]
            if rc != 0 or v.value != code:
                raise Failed(dep, f"{libs[0]} reports {v.value} (rc {rc}), its header {code}",
                             "reinstall the wheel and rebuild the CUDA leg")
            shown += f"; live {live} {v.value}"
        if resolved:
            stem = libs[0].name.split(".so")[0]
            hit = {k: v for k, v in resolved.items() if k.startswith(stem + ".so")}
            if hit and any(v is None or os.path.realpath(v) != os.path.realpath(libs[0]) for v in hit.values()):
                raise Failed(dep, f"the FFI {ffi} binds {hit} ({how}), not the venv's {libs[0]}",
                             "rebuild the CUDA leg with only the venv's wheels on its search path")
            shown += f"; the FFI binds it ({how})" if hit else ""
        ok(dep, shown)


# ----------------------------------------------------------------- cholesky --
def check_cholesky() -> None:
    dep = "cuSOLVERMp distributed Cholesky"
    from runtime import initialize_communicator_stack
    rt = initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    try:
        from distrib_la._cusolvermp import batched_distributed_cholesky, cholesky_handle_to_natural_L
        n = 64
        rng = np.random.default_rng(7)
        b = rng.standard_normal((n, n))
        a = (b @ b.T + n * np.eye(n))[None]
        sh = NamedSharding(rt.mesh, P(None, "x", "y"))
        A = jax.make_array_from_callback(a.shape, sh, lambda idx: a[idx])
        L = cholesky_handle_to_natural_L(batched_distributed_cholesky(A, mesh=rt.mesh))
        Lh = np.asarray(multihost_utils.process_allgather(L, tiled=True))[0]
        err = float(np.max(np.abs(Lh - np.linalg.cholesky(a[0]))) / np.max(np.abs(a[0])))
    except Exception as exc:                                  # noqa: BLE001
        raise Failed(dep, f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}",
                     "see the startup block above for the FFI/NCCL/cuSOLVERMp the run resolved; "
                     "rebuild the CUDA leg (config/perlmutter/build_ffi_cuda.sh)")
    if err > 1e-12:
        raise Failed(dep, f"64x64 factor differs from numpy by {err:.2e} (relative)",
                     "cuSOLVERMp / NCCL mismatch: rebuild the CUDA leg against the venv's wheels")
    if jax.process_index() == 0:
        ok(dep, f"64x64 on mesh {dict(rt.mesh.shape)}: relative error {err:.1e}")


# ---------------------------------------------------------------------- ffi --
_SLOW_ROOTS = ("/global/", "/pscratch/", "/home/", "/global/homes/")


def check_ffi(so: str, arch: str | None, roots: list[str]) -> None:
    dep = f"LORRAX FFI ({so})"
    real = os.path.realpath(so)
    allowed = [os.path.realpath(r) for r in roots] + [str(d) for d in list(Path(real).parents)[:6]
                                                       if (d / "src").is_dir() or d.name == ".venv"]
    home = os.path.realpath(os.path.expanduser("~"))
    libs, how = _bound(so)
    outside = []
    for name, path in libs.items():
        if path is None:
            continue
        rp = os.path.realpath(path)
        slow = rp.startswith(_SLOW_ROOTS) or rp.startswith(home + "/")
        if slow and not any(rp.startswith(a + "/") for a in allowed):
            outside.append(f"{name} -> {rp}")
    if outside:
        raise Failed(dep, f"resolves {len(outside)} librar{'y' if len(outside) == 1 else 'ies'} outside the "
                     f"install: {'; '.join(outside[:3])}", "remove those directories from LD_LIBRARY_PATH "
                     "(a shim on $HOME cost +24 s per stage) and rebuild so the closure names the install's")
    shown = f"{len(libs)} libraries ({how}) inside the install or the system"
    if arch:
        tool = shutil.which("cuobjdump") or next(iter(glob.glob(os.path.join(
            sys.prefix, "lib/python3*/site-packages/nvidia/cu*/bin/cuobjdump"))), None)
        if tool is None:
            skipped(dep, "cubin architectures: no cuobjdump here")
        else:
            out = _run([tool, "--list-elf", so]).stdout
            arches = sorted(set(re.findall(r"\.(sm_\d+a?)\.", out)))
            if arch not in arches:
                raise Failed(dep, f"carries cubins for {arches or 'no architecture'}, not {arch}",
                             f"rebuild with CMAKE_CUDA_ARCHITECTURES including {arch.split('_')[1]}")
            shown += f"; cubins {arches}"
    ok(dep, shown)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dep_check.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("blas"); s.add_argument("lib"); s.add_argument("--expect-threads", type=int)
    s = sub.add_parser("elpa"); s.add_argument("prefix"); s.add_argument("--blas", required=True)
    s.add_argument("--solve", action="store_true")
    sub.add_parser("mpi-allreduce")
    s = sub.add_parser("nvidia"); s.add_argument("--site-packages"); s.add_argument("--ffi")
    sub.add_parser("cholesky")
    s = sub.add_parser("ffi"); s.add_argument("so"); s.add_argument("--arch")
    s.add_argument("--root", action="append", default=[])
    a = ap.parse_args(argv)
    try:
        if a.cmd == "blas":
            check_blas(a.lib, a.expect_threads)
        elif a.cmd == "elpa":
            check_elpa(a.prefix, a.blas, a.solve)
        elif a.cmd == "mpi-allreduce":
            check_mpi_allreduce()
        elif a.cmd == "nvidia":
            check_nvidia(a.site_packages, a.ffi)
        elif a.cmd == "cholesky":
            check_cholesky()
        elif a.cmd == "ffi":
            check_ffi(a.so, a.arch, a.root)
    except Failed as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
