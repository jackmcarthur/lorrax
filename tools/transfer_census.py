"""Transfer / sync / compile / donation census of one LORRAX run (diagnostic; nothing in src/ imports it).

    python tools/transfer_census.py run OUT_DIR SCRIPT.py [args...]      # or: run OUT_DIR -m gw.gw_jax -i cohsex.in
    python tools/transfer_census.py summarize OUT_DIR [--rank 0]         # -> OUT_DIR/census_rank<r>.md

``run`` imports the LORRAX runtime first (GATE blas_spin), installs the hooks below, then executes the
target with runpy; every process writes OUT_DIR/rank<SLURM_PROCID>.json at exit.  Hooks (JAX 0.9.1):
  d2h        ArrayImpl._value  -- every host materialization of a jax.Array (np.asarray, float/bool/int,
             device_get) when no host copy is cached; bytes = size*itemsize
  h2d_*      the numpy entries of pxla.shard_arg_handlers -- classified by the JAX frames on the stack:
             h2d_device_put (jax.device_put / lxkit device_put_process_local), h2d_jnp_array (jnp.asarray/
             array of host data), h2d_jit_arg (a host array or scalar passed to a jitted program)
  sync:*     ArrayImpl.block_until_ready and jax.block_until_ready
  compile:*  jax_log_compiles records (tracing, 'Compiling' = executable-cache miss, persistent hits)
  warn:*     every Python warning (the 'donated buffers were not usable' check)
Each event is keyed by (kind, common.timing.current_path(), innermost LORRAX frames).
XCENSUS_NO_FASTPATH=1 (default) routes every jit call through JAX's Python dispatch (the C++ fast path
would hide repeat-call host arguments); results stay bit-identical, dispatch is slower (~2-3 % wall on Fe).
Note: JAX 0.9.1 does not read JAX_TRANSFER_GUARD_* from the environment; XCENSUS_GUARD=log sets the
C++ guard through jax.config instead (no Python stack; a count cross-check only).
"""
from __future__ import annotations

import atexit, collections, json, logging, os, sys, threading, time, warnings

T0 = time.time()
_lock = threading.Lock()
_ev: dict = {}
_in = threading.local()
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_OUT = None


def _stage():
    try:
        from common import timing
        return "/".join(timing.current_path()) or "-"
    except Exception:
        return "?"


def _site(skip, depth=5):
    f = sys._getframe(skip); out = []; jaxf = []
    while f is not None and len(out) < depth:
        fn = f.f_code.co_filename
        if fn.startswith(_ROOT) and not fn.endswith("transfer_census.py"):
            out.append(f"{os.path.relpath(fn, _ROOT)}:{f.f_lineno}:{f.f_code.co_name}")
        elif f"{os.sep}jax{os.sep}" in fn and len(jaxf) < 40:
            jaxf.append(f.f_code.co_name)
        f = f.f_back
    return tuple(out), jaxf


def _rec(kind, nbytes=0, shape=None, extra=None, skip=3):
    if getattr(_in, "busy", False):
        return
    _in.busy = True
    try:
        site, jaxf = _site(skip)
        if kind == "d2h" and "device_get" in jaxf:
            kind = "d2h_explicit"
        if kind == "h2d":
            if "device_put" in jaxf and not any(n in jaxf for n in ("asarray", "array")):
                kind = "h2d_device_put"
            elif any(n in jaxf for n in ("asarray", "array")):
                kind = "h2d_jnp_array"
            else:
                kind = "h2d_jit_arg"
        st, t = _stage(), time.time() - T0
        with _lock:
            r = _ev.setdefault((kind, st, site), dict(n=0, bytes=0, t0=t, t1=t, shapes=[], extra=[]))
            r["n"] += 1; r["bytes"] += int(nbytes); r["t1"] = t
            if shape is not None and len(r["shapes"]) < 6 and shape not in r["shapes"]:
                r["shapes"].append(shape)
            if extra is not None and len(r["extra"]) < 12:
                r["extra"].append(extra)
    finally:
        _in.busy = False


class _CompileLog(logging.Handler):
    KEYS = (("Finished tracing + transforming", "compile:trace"), ("Compiling ", "compile:miss"),
            ("Persistent compilation cache hit", "compile:persistent_hit"),
            ("Finished XLA compilation", "compile:xla"))

    def emit(self, r):
        try:
            m = r.getMessage()
        except Exception:
            return
        for key, kind in self.KEYS:
            if key in m:
                el = None
                if m.rstrip().endswith("sec"):
                    try:
                        el = float(m.rstrip().split(" in ")[-1].split()[0])
                    except Exception:
                        pass
                _rec(kind, extra=(m[:100], el), skip=8)
                return
        if r.levelno >= logging.WARNING and "Finished jaxpr to MLIR" not in m:
            sys.stderr.write(f"[jax] {r.name}: {m}\n")


def install(out_dir):
    global _OUT
    _OUT = out_dir
    import jax, numpy as np
    from jax._src import array as A, api, pjit
    from jax._src.interpreters import pxla
    getter = A.ArrayImpl._value.fget

    def _value(self):
        if self._npy_value is None:
            _rec("d2h", int(self.size) * self.dtype.itemsize, shape=(tuple(self.shape), str(self.dtype)))
        return getter(self)
    A.ArrayImpl._value = property(_value)
    bur = A.ArrayImpl.block_until_ready

    def _bur(self):
        _rec("sync:block_until_ready", shape=(tuple(self.shape), str(self.dtype)))
        return bur(self)
    A.ArrayImpl.block_until_ready = _bur
    abur = api.block_until_ready

    def _abur(x):
        _rec("sync:jax.block_until_ready")
        _in.busy = True
        try:
            return abur(x)
        finally:
            _in.busy = False
    api.block_until_ready = jax.block_until_ready = _abur
    wrapped = {}
    for t, h in list(pxla.shard_arg_handlers.items()):
        if t is np.ndarray or (isinstance(t, type) and issubclass(t, np.generic)) or t.__name__ == "TypedNdArray":
            if h not in wrapped:
                def mk(h):
                    def w(xs, shardings, layouts, *a):
                        for x in xs:
                            _rec("h2d", int(np.asarray(x).nbytes),
                                 shape=(tuple(np.shape(x)), str(getattr(x, "dtype", type(x).__name__))))
                        return h(xs, shardings, layouts, *a)
                    return w
                wrapped[h] = mk(h)
            pxla.shard_arg_handlers[t] = wrapped[h]
    if os.environ.get("XCENSUS_NO_FASTPATH", "1") == "1":
        pjit._get_fastpath_data = lambda *a, **k: None
        pxla.MeshExecutable.create_cpp_call = lambda self, params: None
    if os.environ.get("XCENSUS_GUARD"):
        jax.config.update("jax_transfer_guard_host_to_device", os.environ["XCENSUS_GUARD"])
        jax.config.update("jax_transfer_guard_device_to_host", os.environ["XCENSUS_GUARD"])
    jax.config.update("jax_log_compiles", True)
    lg = logging.getLogger("jax"); lg.addHandler(_CompileLog()); lg.propagate = False
    if lg.level == logging.NOTSET or lg.level > logging.WARNING:
        lg.setLevel(logging.WARNING)
    show = warnings.showwarning

    def _show(message, category, filename, lineno, file=None, line=None):
        _rec("warn:" + category.__name__, extra=str(message)[:200], skip=3)
        return show(message, category, filename, lineno, file, line)
    warnings.showwarning = _show
    warnings.simplefilter("always")
    atexit.register(dump)


def dump():
    if _OUT is None:
        return
    os.makedirs(_OUT, exist_ok=True)
    r = int(os.environ.get("SLURM_PROCID", "0"))
    rows = [dict(kind=k[0], stage=k[1], site=list(k[2]), **v) for k, v in list(_ev.items())]
    tmp = f"{_OUT}/.rank{r}.json"
    with open(tmp, "w") as fh:
        json.dump(dict(wall=time.time() - T0, rows=rows), fh, default=str)
    os.replace(tmp, f"{_OUT}/rank{r}.json")


# ---------------------------------------------------------------- summarize
#: Thin wrappers whose caller is the real site.
_HELPERS = ("placement.py:", "_slab_io_ffi.py:1296:", "shared_pole_directions.py:34:", "<lambda>", "<genexpr>",
            "collectives.py:634:replicate_to_mesh")


def _attributed(site):
    i = 0
    while i < len(site) - 1 and any(h in site[i] for h in _HELPERS):
        i += 1
    return site[i] if site else "-"


def _line(site):
    if site == "-":
        return ""
    path, ln, _ = site.rsplit(":", 2)
    try:
        with open(os.path.join(_ROOT, path)) as fh:
            for i, text in enumerate(fh, 1):
                if i == int(ln):
                    return text.strip()[:100]
    except OSError:
        pass
    return ""


def summarize(out_dir, rank=0):
    d = json.load(open(f"{out_dir}/rank{rank}.json"))
    rows = d["rows"]
    tot = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = r["kind"] if not r["kind"].startswith("compile") else r["kind"]
        tot[k][0] += r["n"]; tot[k][1] += r["bytes"]
    md = [f"# Transfer census rank {rank}: {out_dir}", f"wall {d['wall']:.1f} s", "",
          "| kind | events | MB |", "|---|---:|---:|"]
    md += [f"| {k} | {v[0]} | {v[1] / 1e6:.2f} |" for k, v in sorted(tot.items())]

    def table(title, pred, key=_attributed):
        agg = {}
        for r in rows:
            if not pred(r["kind"]):
                continue
            s = key(r["site"]) if r["site"] else "-"
            a = agg.setdefault(s, dict(n=0, b=0, kinds=set(), stages=set()))
            a["n"] += r["n"]; a["b"] += r["bytes"]; a["kinds"].add(r["kind"]); a["stages"].add(r["stage"])
        md.extend(["", f"## {title}", "", "| events | MB | kind | site | source | stages |", "|---:|---:|---|---|---|---|"])
        for s, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
            stages = "; ".join(sorted(x.replace("gw_jax.", "") for x in a["stages"]))[:160]
            md.append(f"| {a['n']} | {a['b'] / 1e6:.3f} | {','.join(sorted(a['kinds']))} | `{s}` | "
                      f"`{_line(s).replace('|', '/')}` | {stages} |")
    table("Host-to-device (every row is a violation of 'data reaches devices through SlabIO')",
          lambda k: k.startswith("h2d"))
    table("Device-to-host", lambda k: k.startswith("d2h"))
    table("Explicit syncs", lambda k: k.startswith("sync"))
    table("Executable-cache misses ('Compiling'), by first LORRAX frame", lambda k: k == "compile:miss",
          key=lambda s: s[0])
    table("Warnings", lambda k: k.startswith("warn"), key=lambda s: s[0])
    path = f"{out_dir}/census_rank{rank}.md"
    open(path, "w").write("\n".join(md) + "\n")
    return path


def main(argv):
    if len(argv) >= 2 and argv[0] == "summarize":
        rank = int(argv[argv.index("--rank") + 1]) if "--rank" in argv else 0
        print(summarize(argv[1], rank)); return 0
    if len(argv) < 3 or argv[0] != "run":
        print(__doc__); return 2
    for p in [os.path.join(_ROOT, "src")] + [os.path.join(_ROOT, "services", s, "src")
                                            for s in sorted(os.listdir(os.path.join(_ROOT, "services")))]:
        if p not in sys.path:
            sys.path.insert(0, p)
    import runtime  # noqa: F401  -- GATE blas_spin: the runtime precedes numpy/jax
    install(os.path.abspath(argv[1]))
    import runpy
    if argv[2] == "-m":
        sys.argv = [argv[3]] + argv[4:]
        runpy.run_module(argv[3], run_name="__main__", alter_sys=True)
    else:
        sys.argv = argv[2:]
        runpy.run_path(argv[2], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
