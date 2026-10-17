"""Run gw.gw_jax on a deck, capture the Sigma-tau kconv load tables (rank 0), then exit.

Patches common.fft_helpers.make_kfft_klead_unfold / make_kconv_klead_unfold and
gw.subtile_stream.plan_windows to record their host arguments; after the first
tau (non-static) mode-7 factory call every rank exits.
"""
import os, sys, runpy, pickle
import runtime  # noqa: F401  (before numpy: the OpenBLAS spin gate)
import numpy as np
OUT = os.environ["KC_DUMP_OUT"]
import jax
import common.fft_helpers as fh
import ffi.fft as ff
import gw.subtile_stream as ss
rec = {"windows": [], "kfft": [], "kconv": [], "gplan": []}
import gw.centroid_k_unfold as cku
_ult = cku.CentroidKUnfoldPlan.unfold_load_tables
def unfold_load_tables(self, right_plan=None):
    t = _ult(self, right_plan)
    if right_plan is None:
        rec["gplan"].append(dict(tables={k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in t._asdict().items()},
                                 parent_full_rows=np.asarray(self.parent_full_rows)))
    return t
cku.CentroidKUnfoldPlan.unfold_load_tables = unfold_load_tables
_pw = ss.plan_windows
def plan_windows(local_rows, row_bytes, cuts):
    R, w = _pw(local_rows, row_bytes, cuts)
    rec["windows"].append(dict(local_rows=int(local_rows), R=int(R), windows=[tuple(int(v) for v in x) for x in w]))
    return R, w
ss.plan_windows = plan_windows
def tab(t):
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in t._asdict().items()}
_k9 = fh.make_kfft_klead_unfold
def make_kfft_klead_unfold(mesh, kgrid, tables, **kw):
    rec["kfft"].append(dict(kgrid=tuple(kgrid), tables=tab(tables), kw=kw,
                            mesh=(int(mesh.shape["x"]), int(mesh.shape["y"]))))
    return _k9(mesh, kgrid, tables, **kw)
_k7 = fh.make_kconv_klead_unfold
def make_kconv_klead_unfold(mesh, kgrid, tables, **kw):
    rec["kconv"].append(dict(kgrid=tuple(kgrid), tables=tab(tables),
                             kw={k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in kw.items()},
                             mesh=(int(mesh.shape["x"]), int(mesh.shape["y"]))))
    fn = _k7(mesh, kgrid, tables, **kw)
    tau = bool(rec["kfft"]) and int(rec["kfft"][-1]["tables"].get("conj_trs", 0)) == 0
    if (tau and rec["windows"]) or len(rec["kconv"]) >= 4:
        if jax.process_index() == 0:
            with open(OUT + ".tmp", "wb") as f:
                pickle.dump(rec, f)
            os.replace(OUT + ".tmp", OUT)
            print("KC_DUMP saved", OUT, len(rec["kconv"]), flush=True)
        sys.stdout.flush(); os._exit(0)
    return fn
fh.make_kconv_klead_unfold = make_kconv_klead_unfold
fh.make_kfft_klead_unfold = make_kfft_klead_unfold
ff.make_kconv_klead_unfold = make_kconv_klead_unfold
ff.make_kfft_klead_unfold = make_kfft_klead_unfold
sys.argv = ["gw.gw_jax"] + sys.argv[1:]
runpy.run_module("gw.gw_jax", run_name="__main__")
