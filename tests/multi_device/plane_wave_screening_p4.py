"""P4 GPU gate for ``gw.plane_wave_screening`` (one process per GPU).

Runs the CPU suite's checks (``tests/test_plane_wave_screening.py``) on a 2×2 CUDA
mesh (χ's factor against the band sum, v_q(G) against the ISDF owner and the closed
forms, the Dyson solve under ``linalg = local`` and ``distributed`` against explicit
inverses, the Γ body/fold/head, the MPA tiles against one dense fit, the antiunitary
rule on the glide group) and, with ``--wfn``, the antiunitary rule on a real magnetic
crystal: Fe's SymMaps (antiunitary rows, TR broken), covariant n_s = 2 operands on a
6³ box, χ at every full-grid q, W at z and z̄, the scalar typed transport on the χ
sphere.

Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/plane_wave_screening_p4.py [--wfn PATH]``.
Prints one ``[pw-screening-p4] PASS``/``FAIL`` line per check and ``ALL PASS`` at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import numpy as np  # noqa: E402  (after the runtime: its BLAS thread setting must come first)

import jax  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

TAG = "[pw-screening-p4]"
FAILS = []


def say(msg):
    if jax.process_index() == 0:
        print(f"{TAG} {msg}", flush=True)


def run(name, fn, *a, **kw):
    try:
        out = fn(*a, **kw)
        say(f"PASS {name}")
        return out
    except Exception as e:                    # a failed assertion or a raise: both are FAIL
        say(f"FAIL {name}: {type(e).__name__}: {e}")
        if jax.process_index() == 0:
            traceback.print_exc()
        FAILS.append(name)
        return None


def fe_fixture(mesh, wfn_path, ns, box=(6, 6, 6)):
    from file_io import WfnLoader
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    import zeta_mubatch_fixtures as fixtures
    with WfnLoader(wfn_path, backend="eager") as w:
        sym = w.symmetry()
        b = np.asarray(w.bvec, dtype=np.float64)
        kgrid = tuple(int(v) for v in w.kgrid)
        kpar = np.asarray(w.kvecs(k=sym.parent_k_domain))
    grid = fixtures._grid_points(box)
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    plan = build_centroid_k_unfold_plan(sym, grid, box, mesh, nspinor=ns, parent_k_frac=kpar)
    return dict(plan=plan, fft_grid=box, kgrid=kgrid, kfull=np.asarray(sym.unfolded_kpts),
                ops=np.asarray(sym.sym_matrices)[:n_sp], tnp=np.asarray(sym.translations)[:n_sp],
                rows=np.asarray(sym.active_symmetry_rows), spinor_action=sym.spinor_action), b


def fe_antiunitary(mesh, wfn):
    import test_mixed_basis_pair_convolution as t
    import test_plane_wave_screening as s
    from ffi import _services
    _services.ensure_on_path()
    import vcoul
    # a 10^3 grid and ψ spheres through the |b|^2·8/3 shell (the χ sphere at 0.8 of it holds
    # the G = 0, |b|^2 and 4/3|b|^2 shells: a 1-slot Γ sphere would make every check trivial)
    box = (10, 10, 10)
    fx2, b = fe_fixture(mesh, wfn, 2, box=box)
    fx1, _ = fe_fixture(mesh, wfn, 1, box=box)
    c = t.covariant_case(fx2, ecut=2.9 * float(np.min(np.einsum("ij,ij->i", b, b))),
                         metric=b @ b.T, box=box, nb=2)
    say(f"Fe case: ψ slots {c['sph'].shape[1]}, χ slots at Γ "
        f"{int(c['out_full'][1][np.flatnonzero(np.all(np.abs(c['out_full'][2]) < 1e-12, axis=1))[0]])}, "
        f"closure leak {c['leak']:.1e}")
    assert c["leak"] <= 1e-12, c["leak"]
    geo = vcoul.CoulombGeometry(bvec=b, cell_volume=1.0)
    r = s.antiunitary_check(mesh, fx2, fx1, c, geo)
    say(f"Fe antiunitary ({r['n_anti']} anti rows of {len(c['kfrac'])}): χ(Γ) (r,r') asymmetry "
        f"{r['asym']:.2e}; τ rule {r['tau']:.1e} (no-conj twin {r['tau_red']:.1e}); z rule "
        f"{r['z']:.1e} (parent-at-z twin {r['z_red']:.1e})")
    assert r["n_anti"] > 0 and r["asym"] > 1e-3
    assert r["tau"] <= 1e-10 and r["z"] <= 1e-10
    assert r["tau_red"] > 1e-3 and r["z_red"] > 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", default=None)
    args = ap.parse_args()
    import test_plane_wave_screening as s
    say(f"devices {jax.device_count()} on {jax.process_count()} processes; platform "
        f"{jax.devices()[0].platform}")
    for ns in (1, 2):
        run(f"chi factor vs band sum ns={ns}", s.test_chi_factor_matches_the_band_sum, 4, ns)
    for sd in (3, 2):
        run(f"v_q(G) vs ISDF owner and closed form sys_dim={sd}",
            s.test_coulomb_matches_the_isdf_owner_and_the_closed_form, sd)
    for linalg in ("local", "distributed"):
        run(f"Dyson vs explicit inverses linalg={linalg}", s.test_dyson_matches_explicit_inverses, 4,
            linalg=linalg)
    run("Γ body, fold, head", s.test_gamma_body_fold_and_head)
    run("Γ head, slab", s.test_gamma_head_slab)
    for n_mesh in (4,):
        run("minimax rule on pair sums vs band sum", s.test_minimax_rule_on_pair_sums_matches_the_band_sum,
            n_mesh)
    run("MPA tiles vs dense fit", s.test_pole_fit_tiles_equal_one_dense_fit, 4)
    run("antiunitary rule, glide", s.test_antiunitary_rule_glide)
    if args.wfn:
        mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
        run(f"antiunitary rule, Fe ({os.path.basename(os.path.dirname(os.path.dirname(args.wfn)))})",
            fe_antiunitary, mesh, args.wfn)
    say("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    run_main_and_finalize(main)
