"""Microbench for the vcoul service — claims-style baselines.

Times the two operationally relevant costs (both one-shot setup costs,
not steady-state loops): the per-q ``v(q+G)`` table at Si-fixture-like
shapes for each live kernel, and the 3D body-head table at the production
``nmc=2**18``, plus the Sobol q0 average for reference.  Writes a
claims-style JSON (machine, op, shape, seconds over reps) to
``baselines/<tag>.json``.

Usage:  python bench_vcoul.py <tag>     (e.g. wsl_cpu, perlmutter_login)
"""
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np


def _time(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"reps": reps, "seconds": float(np.median(ts)),
            "seconds_min": float(min(ts)), "seconds_max": float(max(ts))}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "local"
    import jax
    import vcoul

    si_c = 0.612323844648
    bvec = si_c * np.array([[1., 1., -1.], [-1., 1., 1.], [1., -1., 1.]])
    celvol = abs(8.0 * np.pi ** 3 / np.linalg.det(bvec))
    geom = vcoul.CoulombGeometry(bvec=bvec, cell_volume=celvol)

    rng = np.random.RandomState(0)
    n_q, ngk = 64, 3000                      # Si-fixture scale
    qf = (rng.randint(0, 4, (n_q, 3)) / 4.0)
    gvec = rng.randint(-8, 9, (n_q, 3, ngk)).astype(np.float64)

    rows = []
    for sd in (3, 2):
        k = vcoul.get_kernel(sd)
        rows.append({"op": "v_qG_table", "sys_dim": sd,
                     "shape": [n_q, ngk],
                     **_time(lambda k=k: vcoul.v_qG_table(
                         k, qf, gvec, geometry=geom))})

    # The body head is a function of K now, so the honest unit of work is
    # "build the draw, then value the 64 argmin slots" — which is what a
    # 4x4x4 production run actually does, and what the retired per-q table
    # builder used to do in one call.
    def _body_head():
        fn = vcoul.build_v_head_miniBZ_fn_3d((4, 4, 4), bvec, celvol)
        kg = np.asarray((4, 4, 4), dtype=np.float64)
        K = []
        for qx in range(4):
            for qy in range(4):
                for qz in range(4):
                    qw = np.array([qx, qy, qz], dtype=np.float64)
                    qw = np.where(qw > kg / 2, qw - kg, qw)
                    q_cart = (qw / kg) @ bvec
                    if float(q_cart @ q_cart) >= 1e-12:
                        K.append(q_cart)
        fn(np.asarray(K))
    rows.append({"op": "build_v_head_miniBZ_fn_3d + 63 slots",
                 "shape": [4, 4, 4], "nmc": 2 ** 18,
                 **_time(_body_head, reps=3)})

    def _q0():
        b = vcoul.minibz_voronoi_batches(bvec, (4, 4, 4), nsamples=2 ** 18,
                                         method="sobol", qmc_reps=10, nmax=3)
        q0sph2 = vcoul.minibz_inscribed_sphere_r2(bvec, (4, 4, 4))
        vcoul.minibz_average(np.zeros(3), [np.asarray(x) for x in b],
                             kind="bulk_3d", celvol=celvol, n_kpts=64,
                             q0sph2=q0sph2, analytic_sphere=True)
    rows.append({"op": "q0_sobol_analytic_head", "shape": [4, 4, 4],
                 "nsamples": 2 ** 18, "qmc_reps": 10, **_time(_q0)})

    out = {
        "machine": tag,
        "hostname": platform.node(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "recorded": datetime.now(timezone.utc).astimezone().isoformat(),
        "rows": rows,
    }
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "baselines", f"{tag}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
