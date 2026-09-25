"""P=4 gate: the mixed-basis pair convolution on CUDA, both backends, against materialized references.

``gw.mixed_basis_pair_convolution.MixedBasisPairConvolution`` on a 2x2 mesh of four
processes, one GPU each:

* the CPU suite's cases (``tests/test_mixed_basis_pair_convolution.py``): random
  operands with pad-slot garbage (n_s = 1, 2; one chunk and forced r'/batch/k/q
  chunks), the A-cubic tables (48 operations with glides) and the glide group with
  spin mixing and an antiunitary row (n_s = 2, 4), each against the dense supercell
  reference at 1e-11 of max|X|, with their red twins (> 1e-3);
* route predicates: ``backend='router'`` must resolve on this mesh with the mathdx
  k-convolution (``kconv_backend == 'mathdx'``), ``'xla'`` must be the forced fallback;
* fast path against fallback on a larger random case (k-grid 4x4x4, box 12^3, no
  dense reference): 1e-12 of max|X|;
* ``--wfn WFN.h5`` adds a real magnetic crystal (Fe, n_s = 2, antiunitary rows): its
  SymMaps with metric spheres at a reduced cutoff on a 6^3 box, parents against the
  full grid from the r-space action.

Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/mixed_basis_pair_conv_p4.py [--wfn PATH]``.
Prints one ``[pairconv-p4] PASS``/``FAIL`` line per check and ``ALL PASS`` at the end.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import mixed_basis_pair_conv_cases as cases  # noqa: E402
import test_mixed_basis_pair_convolution as t  # noqa: E402

TAG = "[pairconv-p4]"
FAILS = []


def say(msg):
    if jax.process_index() == 0:
        print(f"{TAG} {msg}", flush=True)


def check(name, ok, detail):
    say(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        FAILS.append(name)


def fe_case(mesh, wfn_path, ns_box=(6, 6, 6)):
    """Fe's magnetic SymMaps on a 6^3 grid; spheres |k+G|^2 <= 1.05 min|b|^2."""
    from file_io import WfnLoader
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    import zeta_mubatch_fixtures as fixtures
    with WfnLoader(wfn_path, backend="eager") as w:
        sym = w.symmetry()
        b = np.asarray(w.bvec, dtype=np.float64)
        kgrid = tuple(int(v) for v in w.kgrid)
        kpar = np.asarray(w.kvecs(k=sym.parent_k_domain))
        ns = int(w.nspinor)
    grid = fixtures._grid_points(ns_box)
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    plan = build_centroid_k_unfold_plan(sym, grid, ns_box, mesh, nspinor=ns, parent_k_frac=kpar)
    fx = dict(plan=plan, fft_grid=ns_box, kgrid=kgrid, kfull=np.asarray(sym.unfolded_kpts),
              ops=np.asarray(sym.sym_matrices)[:n_sp], tnp=np.asarray(sym.translations)[:n_sp])
    return t.symmetry_case(fx, ecut=1.05 * float(np.min(np.einsum("ij,ij->i", b, b))),
                           metric=b @ b.T, box=ns_box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", default=None)
    args = ap.parse_args()
    from ffi.fft import kconv_backend
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    check("route", kconv_backend(mesh) == "mathdx", f"kconv_backend={kconv_backend(mesh)}")

    # ---- random operands, dense reference ---------------------------------
    for ns in (1, 2):
        c = t.random_case(ns)
        ref, ks = t._references(c)
        tr = SphereTransport.identity(SphereSet(c["sph"], c["ngk"], c["kfrac"]), ns)
        for backend in ("router", "xla"):
            for chunks in (None, (3, 4, 2, 1)):
                conv = t._conv(mesh, c["kgrid"], c["fft_grid"], c["sph"], c["ngk"], c["kfrac"],
                               c["out"], transport=tr, backend=backend, chunks=chunks,
                               budget_bytes=int(1e10))
                e = cases.rel(t._run(conv, c["A"], c["C"]), ref)
                check(f"random ns={ns} {backend} chunks={chunks}",
                      e <= t.TOL and conv.backend == backend,
                      f"rel {e:.2e} (n_c {conv.chunks.n_c}, J {conv.chunks.J}, kc {conv.chunks.kc}, "
                      f"qc {conv.chunks.qc}; k-sum vs dense {cases.rel(ks, ref):.1e})")

    # ---- symmetry: typed parents against the full grid ---------------------
    sym_cases = [("A-cubic ns=1", t.acubic_case(mesh), "conj_phase"),
                 ("glide ns=2", t.glide_case(mesh, 2), "no_anti"),
                 ("glide ns=4", t.glide_case(mesh, 4), "no_anti")]
    if args.wfn:
        sym_cases.append((f"Fe ns=2 ({os.path.basename(os.path.dirname(os.path.dirname(args.wfn)))})",
                          fe_case(mesh, args.wfn), "no_anti"))
    for name, c, red in sym_cases:
        for backend in ("router", "xla"):
            r = t._symmetry_check(mesh, c, backend)
            check(f"{name} {backend}", r["full"] <= t.TOL and r["parent"] <= t.TOL
                  and r["red"][red] > 1e-3,
                  f"full {r['full']:.2e}, parent {r['parent']:.2e}, anti rows {r['anti']}, "
                  f"red twins {', '.join(f'{k} {v:.1e}' for k, v in r['red'].items())}, "
                  f"leak {c['leak']:.1e}")

    # ---- fast path against fallback at a larger size ------------------------
    for ns in (1, 2):
        kgrid, box = (4, 4, 4), (12, 12, 12)
        kfrac = cases.kgrid_frac(kgrid)
        metric = np.asarray([[1.0, 0.2, 0.0], [0.2, 1.1, 0.1], [0.0, 0.1, 0.9]])
        sph, ngk = cases.spheres(kfrac, metric, 4.2, span=4)
        rng = np.random.default_rng(11 + ns)
        A = cases.random_green(rng, len(kfrac), sph.shape[1], ngk, ns)
        C = cases.random_green(rng, len(kfrac), sph.shape[1], ngk, ns)
        qsel = np.arange(0, 64, 5)
        out = (*cases.spheres(kfrac[qsel], metric, 3.0, span=4), kfrac[qsel])
        tr = SphereTransport.identity(SphereSet(sph, ngk, kfrac), ns)
        got = {}
        for backend in ("router", "xla"):
            conv = t._conv(mesh, kgrid, box, sph, ngk, kfrac, out, transport=tr, backend=backend,
                           budget_bytes=int(8e9))
            got[backend] = t._run(conv, A, C)
            say(conv.describe())
        e = cases.rel(got["router"], got["xla"])
        check(f"fast vs fallback ns={ns} k 4^3 box 12^3 M={sph.shape[1]}", e <= 1e-12, f"rel {e:.2e}")

    say("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        finalize_process(rc)
