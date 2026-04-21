"""Evaluate a saved fit at a (sys, knobs, mesh) point.

Example:

    lxrun python3 -m gw.aot_memory_model.predict_cli \\
        --kernel chi0_tau_step --tag si222_tiny \\
        --kgrid 4,4,4 --n_rmu 2400 --n_b_v 20 --n_b_c 276 --n_s 2 \\
        --mesh 2,2
"""
from __future__ import annotations

import argparse

from .core import SysDims, Knobs, MeshSpec, get_kernel, load_fit, predict_peak


def _parse_tuple(s, n=None, cast=int):
    parts = [cast(x) for x in s.split(",")]
    if n is not None and len(parts) != n:
        raise ValueError(f"Expected {n}-tuple, got {parts}")
    return tuple(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--kgrid", required=True, help="e.g. 4,4,4")
    ap.add_argument("--n_rmu", type=int, required=True)
    ap.add_argument("--n_s", type=int, default=1)
    ap.add_argument("--nspinor", type=int, default=1)
    ap.add_argument("--n_b_v", type=int, default=0)
    ap.add_argument("--n_b_c", type=int, default=0)
    ap.add_argument("--n_b", type=int, default=0)
    ap.add_argument("--n_r", type=int, default=0)
    ap.add_argument("--mesh", required=True, help="p_x,p_y, e.g. 2,2")
    ap.add_argument("--knob", action="append", default=[],
                    help="name=int  (repeatable)")
    args = ap.parse_args(argv)

    sys = SysDims(
        kgrid=_parse_tuple(args.kgrid, 3), n_rmu=args.n_rmu,
        n_s=args.n_s, nspinor=args.nspinor,
        n_b_v=args.n_b_v, n_b_c=args.n_b_c, n_b=args.n_b, n_r=args.n_r,
    )
    mesh_vals = _parse_tuple(args.mesh, 2)
    mesh = MeshSpec(p_x=mesh_vals[0], p_y=mesh_vals[1])
    knob_kv = {}
    for spec in args.knob:
        k, v = spec.split("=")
        knob_kv[k] = int(v)
    knobs = Knobs.of(**knob_kv)

    kernel = get_kernel(args.kernel)
    fit = load_fit(args.kernel, tag=args.tag)
    peak = predict_peak(fit, kernel, sys, knobs, mesh)

    print(f"\nPrediction for kernel={args.kernel} fit_tag={args.tag}:")
    print(f"  sys={sys}")
    print(f"  mesh={mesh}  knobs={knobs}")
    print(f"  peak = {peak / 1e9:.3f} GB = {peak:.3e} bytes")
    print(f"  (gamma correction = {fit.gamma:.3f})")
    print(f"\nPer-primitive contribution:")
    for name, coef in zip(fit.feature_names, fit.coefs):
        bytes_i = coef * kernel.PRIMITIVES[name](sys, knobs, mesh)
        print(f"  β[{name:12s}] = {coef:8.3f} × {kernel.PRIMITIVES[name](sys, knobs, mesh)/1e9:7.3f} GB "
              f"= {bytes_i/1e9:7.3f} GB")
    print(f"  intercept                       = {fit.intercept/1e9:7.3f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
