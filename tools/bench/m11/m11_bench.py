"""Mode-11 (chi unfold) microbenchmark + bitwise dump, one GPU, several cases per process.
usage: m11_bench.py OUTDIR VAR CASE...   CASE = name:nx,ny,nz:mx:my:tables:partners:live:nout:complete:acc0:write
  live '' or lo-hi; acc0 zero|rand; write 0|1 (acc=None).  Env M11_REPS (20), M11_NOSAVE."""
import json, sys, time, os
import runtime
RUNTIME = runtime.initialize_communicator_stack()
import numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/cs_claude/sigma_depth2/wt_kc_occ/tools/bench")
import kconv_bench as KB
import jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)
import ffi.fft as F
from symmetry_maps import UnfoldLoadTables
from symmetry_maps.unfold_load import DeviceLoadTables
OUT, VAR = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
REPS = int(os.environ.get("M11_REPS", "20"))
dev = lambda a: jax.device_put(np.asarray(a))
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
ns = 2
for case in sys.argv[3:]:
    name, grid, mx, my, tab, partners, live, nout, complete, acc0m, write = case.split(":")
    n = tuple(int(v) for v in grid.split(",")); nk = int(np.prod(n))
    partners, nout, complete, write = int(partners), int(nout), int(complete), int(write)
    rng = np.random.default_rng(5)
    cz = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex128)
    if tab:
        z = np.load(tab); T = {k: z[k] for k in z.files}; T["n_parent"] = int(T["n_parent"])
        mx = int(T["lsrc"].shape[1]) // ns; my = int(T["rsrc"].shape[1]) // ns
    else:
        mx, my = int(mx), int(my)
        T = KB.synth_tables(n, mx, my, ns, "group", 1)
    npar = T["n_parent"]
    t = UnfoldLoadTables(row=T["row"], trs=T["trs"], lsrc=T["lsrc"], rsrc=T["rsrc"], mph=T["mph"], nph=T["nph"],
                         spin=T["spin"], n_parent=npar, mesh_shape=(1, 1))
    load = DeviceLoadTables(*(dev(a) for a in (t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin, t.spin)))
    kc = F.make_kconv_chi_unfold(mesh, n, t, n_out=nout, complete=bool(complete), norm="ortho")
    Gv, Gc, Gvt, Gct = (dev(cz(npar, mx, ns, my, ns)) for _ in range(4))
    alpha = dev(cz(nout))
    kw = dict(load=load)
    if live: kw["live"] = dev(np.array([int(v) for v in live.split("-")], np.int32))
    part = (Gvt, Gct) if partners else ()
    shape = (nout, nk, mx, my)
    acc0 = cz(*shape) if acc0m == "rand" else np.zeros(shape, np.complex128)
    qsel = np.random.default_rng(11).integers(0, nk, 118).astype(np.int32)   # 118 rows, some repeated
    if write == 1:
        f = jax.jit(lambda acc: kc(None, Gv, Gc, alpha, *part, **kw))
    elif write == 2:                          # the fused q-row store
        qd = dev(qsel)
        f = jax.jit(lambda acc: kc(None, Gv, Gc, alpha, *part, rows=qd, **kw))
    elif write == 3:                          # reference: accumulate on zeros, mode 3, the row gather
        from common.fft_helpers import make_flat_k_fftn
        from jax.sharding import PartitionSpec as P
        fftn = make_flat_k_fftn(mesh, n, P(None, None, None, "x", "y"), norm="ortho")
        qd = dev(qsel)
        f = jax.jit(lambda acc: jnp.take(fftn(kc(jnp.zeros(shape, jnp.complex128), Gv, Gc, alpha, *part, **kw)[0]),
                                         qd, axis=0)[None])
    else:
        f = jax.jit(lambda acc: kc(acc, Gv, Gc, alpha, *part, **kw), donate_argnums=0)
    out = np.asarray(f(jnp.asarray(acc0)))
    if not os.environ.get("M11_NOSAVE"): np.savez(f"{OUT}/{VAR}.{name}.npz", acc=out)
    st = jnp.asarray(acc0)
    st = f(st); jax.block_until_ready(st); st = f(st); jax.block_until_ready(st)
    t0 = time.perf_counter()
    for _ in range(REPS): st = f(st)
    jax.block_until_ready(st)
    ms = 1e3 * (time.perf_counter() - t0) / REPS
    del st
    rec = dict(var=VAR, name=name, grid=list(n), mx=mx, my=my, n_parent=npar, partners=partners, nout=nout,
               complete=complete, live=live, write=write, acc0=acc0m, ms=round(ms, 4),
               tables=tab.split("/")[-1] or "synthetic", finite=bool(np.isfinite(out).all()),
               absmax=float(np.abs(out).max()))
    print("M11B " + json.dumps(rec), flush=True)
    with open(f"{OUT}/{VAR}.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
