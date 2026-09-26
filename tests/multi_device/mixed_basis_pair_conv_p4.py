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
* the router plan's compiled stages hold only the planned collectives: one all-to-all
  per slab and expand, two in the final stage, none in the streamed middle;
* ``--wfn WFN.h5`` adds a real magnetic crystal (Fe, n_s = 2, antiunitary rows): its
  SymMaps with metric spheres at a reduced cutoff on a 6^3 box, parents against the
  full grid from the r-space action;
* the r'-column wedge (``ColumnWedge``) on covariant operands (parent band sets closed
  under their little groups), both backends: glide n_s = 2 (full group; {E, Θ·glide},
  whose antiunitary branch carries every non-identity column), A-cubic (48 operations),
  complex weights with partners on the unitary rows, and with ``--wfn`` Fe's 16 rows
  (8 antiunitary) and its 8 unitary rows at n_s = 1 and 2; each against the dense
  reference and against the dense-column plan at 1e-11, with red twins (the antiunitary
  conjugation or the lattice-wrap phase dropped) missing by > 1e-3.

* Σ, the second caller (``product='scalar'``: A = G on the ψ sphere, B = W on the χ
  sphere, the output on the ψ sphere), both backends: random operands (n_s = 1, 2, W rows
  at the other representative of the ±½ planes, forced chunks); the glide group with G at
  its parents and W at its q-IBZ (W's conj rule; red twin: W's antiunitary flag dropped);
  the r'-wedge on covariant G and W (red twin: the output's spin sandwich dropped); with
  ``--wfn`` Fe at n_s = 1 and 2 on the same checks, all 16 rows and the 8 unitary ones;
* the expand at the k-parents (p'→r' on the parents, each child a column gather of its
  parent's transform at mtrx·(r' − τ) with the Bloch/wrap phase e^{-2πi k̄·y}): its red twins,
  the lattice wrap dropped and the identity column map, must miss by > 1e-3 on the typed
  parents (A-cubic, glide, Fe) and on Fe's wedge checks for both products;
* W's antiunitary rule at fixed τ on the TR-broken Fe group: χ₀(τ) of covariant Greens at
  every full-grid q equals the conj-rule unfold of χ₀ at the parents (red twin: no conj).

Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/mixed_basis_pair_conv_p4.py [--wfn PATH] [--only sigma]``.
Prints one ``[pairconv-p4] PASS``/``FAIL`` line per check and ``ALL PASS`` at the end.
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402  (after the runtime: its BLAS thread setting must come first)

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


def fe_fixture(mesh, wfn_path, ns_box=(6, 6, 6), ns=None):
    """Fe's magnetic SymMaps on a 6^3 grid as a fixture dict (``ns`` overrides the spinor width)."""
    from file_io import WfnLoader
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    import zeta_mubatch_fixtures as fixtures
    with WfnLoader(wfn_path, backend="eager") as w:
        sym = w.symmetry()
        b = np.asarray(w.bvec, dtype=np.float64)
        kgrid = tuple(int(v) for v in w.kgrid)
        kpar = np.asarray(w.kvecs(k=sym.parent_k_domain))
        ns = int(w.nspinor) if ns is None else int(ns)
    grid = fixtures._grid_points(ns_box)
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    plan = build_centroid_k_unfold_plan(sym, grid, ns_box, mesh, nspinor=ns, parent_k_frac=kpar)
    return dict(plan=plan, fft_grid=ns_box, kgrid=kgrid, kfull=np.asarray(sym.unfolded_kpts),
                ops=np.asarray(sym.sym_matrices)[:n_sp], tnp=np.asarray(sym.translations)[:n_sp],
                rows=np.asarray(sym.active_symmetry_rows), spinor_action=sym.spinor_action), b


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


def _fe_box(c, w):
    """The smallest box from 6³ up that keeps Σ (ψ ⊕ χ → ψ) and χ₀ alias-free on these spheres."""
    from gw.mixed_basis_pair_convolution import SphereSet, alias_free_margin
    sp = SphereSet(c["sph"], c["ngk"], c["kfrac"]).recentred().union_support()
    ws = SphereSet(w["wsp"], w["wngk"], w["wfrac"]).recentred().union_support()
    for n in range(6, 13):
        if np.all(alias_free_margin((n,) * 3, sp, ws, sp) >= 1) and np.all(
                alias_free_margin((n,) * 3, sp, sp, ws) >= 1):
            return (n, n, n)
    raise RuntimeError("no alias-free box up to 12^3")


def sigma_checks(mesh, args):
    """Σ = G ⊙ W, the second caller of the same plan (product='scalar')."""
    import zeta_mubatch_fixtures as fixtures
    from ffi.fft import kconv_backend  # noqa: F401  (route already checked)
    from gw.mixed_basis_pair_convolution import SphereTransport
    # ---- random operands, dense reference ---------------------------------
    for ns in (1, 2):
        c = t.sigma_random_case(ns)
        ref = t._sigma_ref(c)
        g_op, w_op = t._identity_ops(c)
        for backend in ("router", "xla"):
            for chunks in (None, (3, 4, 2, 1)):
                conv = t._sigma_conv(mesh, c["kgrid"], c["fft_grid"], g_op, w_op, c["out"],
                                     backend=backend, chunks=chunks, budget_bytes=int(1e10))
                e = cases.rel(t._run_sigma(conv, c["G"], c["W"]), ref)
                check(f"sigma random ns={ns} {backend} chunks={chunks}",
                      e <= t.TOL and conv.backend == backend,
                      f"rel {e:.2e} (n_c {conv.chunks.n_c}, J {conv.chunks.J}, kc {conv.chunks.kc}, "
                      f"qc {conv.chunks.qc})")
    # the red twin: without the time-reversed transport the plan forms A ⊙ conj B
    c = t.sigma_random_case(2)
    ref = t._sigma_ref(c)
    g_op, w_op = t._identity_ops(c)
    keep = SphereTransport.time_reversed
    SphereTransport.time_reversed = lambda self, sphere: self
    try:
        conv = t._sigma_conv(mesh, c["kgrid"], c["fft_grid"], g_op, w_op, c["out"], backend="router",
                             budget_bytes=int(1e10))
        e = cases.rel(t._run_sigma(conv, c["G"], c["W"]), ref)
    finally:
        SphereTransport.time_reversed = keep
    check("sigma red twin: no time reversal (router)", e > 1e-3, f"rel {e:.1e}")
    census = conv.collective_census()
    want = {"middle": {}, "final": {"all-to-all": 2}, "slab left": {"all-to-all": 1},
            "slab right": {"all-to-all": 1}, "expand left": {"all-to-all": 1},
            "expand right": {"all-to-all": 1}}
    check("sigma collective census (router, compiled HLO)", census == want, str(census))

    # ---- symmetry: G at the parents, W at the q-IBZ; the wedge on covariant operands ----
    cg, wg = t.sigma_glide_case(mesh, covariant=False)
    cv, wv = t.sigma_glide_case(mesh, covariant=True)
    refg = refv = None
    for backend in ("router", "xla"):
        r = t._sigma_symmetry_check(mesh, cg, wg, backend, twins=("no_anti_W",), ref=refg)
        refg = r["ref_arr"]
        check(f"sigma glide ns=2 parents {backend}",
              r["anti"] and r["parent"] <= t.TOL and r["w_tile"] <= 1e-13 and r["red"]["no_anti_W"] > 1e-3,
              f"parents vs dense ref {r['parent']:.2e}; W tile unfold {r['w_tile']:.1e}; red no_anti_W "
              f"{r['red']['no_anti_W']:.1e}")
        for rows in ((0, 1, 2, 3), (0, 3)):
            r = t._sigma_symmetry_check(mesh, cv, wv, backend, wedge_rows=rows, twins=("no_spin",),
                                        ref=refv)
            refv = r["ref_arr"]
            check(f"sigma wedge glide ns=2 rows {rows} {backend}",
                  r["parent"] <= t.TOL and r["wedge_ref"] <= t.TOL and r["wedge_dense"] <= t.TOL
                  and r["red"]["no_spin"] > 1e-3,
                  f"wedge vs dense ref {r['wedge_ref']:.2e}, vs dense-column plan {r['wedge_dense']:.2e}; "
                  f"{r['orbits']} orbits of {r['nr']}; red no_spin {r['red']['no_spin']:.1e}")
    if not args.wfn:
        return
    for ns_fe in (1, 2):
        fxf, bf = fe_fixture(mesh, args.wfn, ns=ns_fe)
        metric = bf @ bf.T
        e1 = float(np.min(np.einsum("ij,ij->i", bf, bf)))
        c = t.covariant_case(fxf, ecut=1.05 * e1, metric=metric, box=(6, 6, 6), nb=2)
        kf = c["kfrac"]
        qsel = [0, 1, len(kf) - 1]
        c["out"] = (*cases.spheres(kf[qsel], metric, 1.05 * e1), kf[qsel])
        c["out_full"] = (*cases.spheres(kf, metric, 1.05 * e1), kf)
        w = t.w_parents_and_children(c, ecut_w=0.8 * e1, metric=metric, covariant=True)
        c["fft_grid"] = _fe_box(c, w)
        rows = np.asarray(fxf["rows"])
        n_sp = len(fxf["ops"])
        cr = t.conj_rule_check(mesh, c, ecut_w=0.8 * e1, metric=metric, backend="router")
        check(f"W conj rule at fixed tau, Fe ns={ns_fe} (TR-broken)",
              cr["n_anti"] > 0 and cr["rule"] <= t.TOL and cr["red"] > 1e-3,
              f"chi0 full grid vs conj-rule unfold {cr['rule']:.2e} over {cr['n_anti']} antiunitary "
              f"children; red (no conj) {cr['red']:.1e}")
        ref = None
        for backend in ("router", "xla"):
            for name, rw in ((f"all {len(rows)} rows", rows), ("unitary rows", rows[rows < n_sp])):
                twins = ("no_anti_W",) + (("no_spin",) if ns_fe > 1 else ()) + t.EXPAND_TWINS
                r = t._sigma_symmetry_check(mesh, c, w, backend, wedge_rows=rw, twins=twins, ref=ref)
                ref = r["ref_arr"]
                ok = (r["parent"] <= t.TOL and r["wedge_ref"] <= t.TOL and r["wedge_dense"] <= t.TOL
                      and r["w_tile"] <= 1e-13 and all(v > 1e-3 for v in r["red"].values()))
                check(f"sigma Fe ns={ns_fe} {name} {backend} box {c['fft_grid']}", ok,
                      f"parents vs dense ref {r['parent']:.2e}; wedge vs ref {r['wedge_ref']:.2e}, vs "
                      f"dense-column {r['wedge_dense']:.2e}; {r['orbits']} orbits of {r['nr']}; W tile "
                      f"{r['w_tile']:.1e}; red {', '.join(f'{k} {v:.1e}' for k, v in r['red'].items())}; "
                      f"leak {c['leak']:.1e}/{w['wleak']:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", default=None)
    ap.add_argument("--only", choices=("all", "sigma"), default="all")
    args = ap.parse_args()
    from ffi.fft import kconv_backend
    from gw.mixed_basis_pair_convolution import SphereSet, SphereTransport
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    check("route", kconv_backend(mesh) == "mathdx", f"kconv_backend={kconv_backend(mesh)}")
    if args.only == "sigma":
        sigma_checks(mesh, args)
        say("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
        return 0 if not FAILS else 1

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
                  and r["red"][red] > 1e-3 and all(r["red"][k] > 1e-3 for k in t.EXPAND_TWINS),
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
        for backend in ("xla", "router"):
            conv = t._conv(mesh, kgrid, box, sph, ngk, kfrac, out, transport=tr, backend=backend,
                           budget_bytes=int(8e9))
            got[backend] = t._run(conv, A, C)
            say(conv.describe())
        e = cases.rel(got["router"], got["xla"])
        check(f"fast vs fallback ns={ns} k 4^3 box 12^3 M={sph.shape[1]}", e <= 1e-12, f"rel {e:.2e}")

    # ---- the r'-column wedge on covariant operands ---------------------------
    import zeta_mubatch_fixtures as fixtures
    wcases = []
    fxg = fixtures._glide_fixture(mesh, np.random.default_rng(2), 2, translated_anti=True,
                                  theta=np.pi / 2)     # a spin representation: glide² = E
    cg = t.covariant_case(fxg, ecut=1.3, metric=np.eye(3), box=(6, 6, 5))
    wcases += [("glide ns=2 full group", cg, (0, 1, 2, 3), ("no_wrap",)),
               ("glide ns=2 {E, anti glide}", cg, (0, 3), ("no_conj",))]
    fxa = fixtures._acubic_fixture(mesh, np.random.default_rng(1))
    from file_io import WfnLoader
    root = fixtures._HERE / "core" / "fixtures" / "A-cubic"
    with WfnLoader(root / "WFN.h5", backend="eager", qe_schema=root / "data-file-schema.xml") as w:
        ba = np.asarray(w.bvec, dtype=np.float64)
    ca = t.covariant_case(fxa, ecut=1.05 * float(np.min(np.einsum("ij,ij->i", ba, ba))),
                          metric=ba @ ba.T, box=(8, 8, 8), nb=2)
    wcases.append(("A-cubic ns=1", ca, fxa["rows"], ("no_wrap",)))
    cp = t.covariant_case(fxg, ecut=1.3, metric=np.eye(3), box=(6, 6, 5), complex_weights=True)
    wcases.append(("glide ns=2 complex weights + partners, unitary rows", cp, (0, 1), ()))
    if args.wfn:
        for ns_fe in (1, 2):
            fxf, bf = fe_fixture(mesh, args.wfn, ns=ns_fe)
            cf = t.covariant_case(fxf, ecut=1.05 * float(np.min(np.einsum("ij,ij->i", bf, bf))),
                                  metric=bf @ bf.T, box=(6, 6, 6), nb=2)
            n_sp = len(fxf["ops"])
            rows = np.asarray(fxf["rows"])
            wcases += [(f"Fe ns={ns_fe} all {len(rows)} rows", cf, rows,
                        ("no_conj", "no_wrap") + t.EXPAND_TWINS),
                       (f"Fe ns={ns_fe} unitary rows", cf, rows[rows < n_sp], t.EXPAND_TWINS)]
    for name, c, rows, twins in wcases:
        for backend in ("router", "xla"):
            r = t._wedge_check(mesh, c, backend, rows, twins=twins)
            ok = (r["ref"] <= t.TOL and r["dense"] <= t.TOL and r["dense_ref"] <= t.TOL
                  and all(v > 1e-3 for v in r["red"].values()))
            check(f"wedge {name} {backend}", ok,
                  f"vs dense ref {r['ref']:.2e}, vs dense-column plan {r['dense']:.2e} "
                  f"(dense plan vs ref {r['dense_ref']:.2e}); {r['orbits']} orbits of {r['nr']} "
                  f"columns; red twins {', '.join(f'{k} {v:.1e}' for k, v in r['red'].items()) or '-'}; "
                  f"leak {c['leak']:.1e}")
    try:
        t._wedge_check(mesh, cp, "xla", (0, 3))
        check("wedge refuses partners on antiunitary rows", False, "no refusal")
    except ValueError as e:
        check("wedge refuses partners on antiunitary rows", "GATE pairconv-wedge-partner" in str(e),
              str(e)[:80])

    # ---- collective census of the fast path's compiled stages ---------------
    census = conv.collective_census()
    want = {"middle": {}, "final": {"all-to-all": 2}, "slab left": {"all-to-all": 1},
            "slab right": {"all-to-all": 1}, "expand left": {"all-to-all": 1},
            "expand right": {"all-to-all": 1}}
    check("collective census (router, compiled HLO)", census == want, str(census))

    sigma_checks(mesh, args)
    say("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    run_main_and_finalize(main)       # keeps a failure's traceback, then the ordered exit
