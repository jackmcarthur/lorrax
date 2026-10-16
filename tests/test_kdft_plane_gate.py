"""The ROCM k-box mode 6 (``lorrax_kdft_kconv_plane``, the port of the mathdx plane mode) against the
XLA route on the same GCD: ``ffi.fft.make_fused_conv_kplane`` built by default (the kdft target on a
ROCm mesh; a case that silently took XLA fails) and inside ``ffi.fft.xla_reference()``, applied to the
same random D, F with random permutations and phases in {1, i, -1, -i}; ``max|a - b| <= 1e-12 max|b|``.
Cases: ns 1, 2, 4; grids 1x1x1 to 6^3 (one output column per tile at 6^3, ns 2); a partial last tile;
and the refusal by name (``GATE kdft-kconv-plane-tile``, ``GATE kdft-axis``).  ``--bench``: per call,
kdft vs the XLA route, at hsuite- and Fe-shaped tiles.  ROCm only: ``python -m tests.test_kdft_plane_gate``.
"""
import argparse
import sys
import time

TOL = 1e-12


def _timed(fn, *args, reps=10):
    import jax
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(reps):
        y = fn(*args)
    jax.block_until_ready(y)
    return (time.perf_counter() - t0) / reps * 1e3


def _case(rng, kg, ns, g, c, p, dev):
    import jax
    import numpy as np
    nk = int(np.prod(kg))
    rnd = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    D = jax.device_put(rnd(nk, g, ns, 2 * c, ns, p), dev)
    F = jax.device_put(np.exp(2j * np.pi * rng.random((nk, g, p))), dev)
    q = np.asarray([1, 1j, -1, -1j])
    tabs = dict(perm_l=rng.permutation(ns), phase_l=q[rng.integers(0, 4, ns)],
                perm_r=rng.permutation(ns), phase_r=q[rng.integers(0, 4, ns)])
    return D, F, tabs


def gate_cases(mesh):
    import jax
    import numpy as np
    from ffi import fft as F_
    from tests.test_kconv_xla_gate import _rel
    dev = jax.local_devices()[0]
    rng = np.random.default_rng(20261016)
    out = {}
    for kg, ns, g, c, p in (((4, 4, 4), 2, 3, 7, 11), ((5, 5, 1), 2, 2, 9, 13), ((3, 2, 2), 1, 2, 5, 7),
                            ((2, 2, 2), 4, 2, 3, 5), ((1, 1, 1), 2, 2, 4, 3), ((6, 6, 6), 2, 1, 3, 5),
                            ((8, 8, 1), 2, 2, 5, 9)):
        tag = f"{kg[0]}x{kg[1]}x{kg[2]} ns{ns} g{g} c{c} p{p}"
        if F_.rocm_plane_target(mesh, kg, ns) is None:
            out[f"{tag} ROUTE"] = float("inf")
            continue
        D, Fp, tabs = _case(rng, kg, ns, g, c, p, dev)
        f_k = F_.make_fused_conv_kplane(mesh, kg, ns, **tabs)
        with F_.xla_reference():
            f_x = F_.make_fused_conv_kplane(mesh, kg, ns, **tabs)
        out[tag] = _rel(jax.jit(f_k)(D, Fp), jax.jit(f_x)(D, Fp))
    for kg, ns, gate in (((41, 1, 1), 2, "kdft-axis"), ((10, 10, 10), 4, "kdft-kconv-plane-tile")):
        why = F_.kdft_plane_refusal(kg, ns)
        out[f"refusal {kg} ns{ns} -> GATE {gate}"] = 0.0 if f"GATE {gate}" in why else float("inf")
    return out


def bench(mesh, shapes, reps):
    import jax
    import numpy as np
    from ffi import fft as F_
    dev = jax.local_devices()[0]
    rng = np.random.default_rng(7)
    print("bench  k-grid   ns   g    c    p |  XLA ms   kdft ms   (x)", flush=True)
    for kg, ns, g, c, p in shapes:
        D, Fp, tabs = _case(rng, kg, ns, g, c, p, dev)
        f_k = jax.jit(F_.make_fused_conv_kplane(mesh, kg, ns, **tabs))
        with F_.xla_reference():
            f_x = jax.jit(F_.make_fused_conv_kplane(mesh, kg, ns, **tabs))
        tx, tk = _timed(f_x, D, Fp, reps=reps), _timed(f_k, D, Fp, reps=reps)
        print(f"bench  {kg[0]}x{kg[1]}x{kg[2]:<3d} {ns:3d} {g:3d} {c:4d} {p:4d} | {tx:8.3f} {tk:8.3f} {tx / tk:6.2f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--shapes", default="4x4x4:2:8:78:40,4x4x4:2:16:156:40,5x5x1:2:4:24:30,8x8x1:2:8:90:40")
    args = ap.parse_args()
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    from ffi.gate import mesh_ffi_platform
    mesh = stack.mesh
    if mesh_ffi_platform(mesh) != "ROCM":
        print(f"SKIP: needs a ROCm mesh, got {mesh_ffi_platform(mesh)}", flush=True)
        runtime.finalize_process(0)
        return 0
    if args.bench:
        shapes = []
        for s in args.shapes.split(","):
            g, *rest = s.split(":")
            shapes.append((tuple(int(v) for v in g.split("x")), *(int(v) for v in rest)))
        bench(mesh, shapes, args.reps)
        runtime.finalize_process(0)
        return 0
    out = gate_cases(mesh)
    bad = {k: v for k, v in out.items() if not v <= TOL}
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:50s} rel {v:.3e}", flush=True)
        print(f"KDFT_PLANE_GATE {'PASS' if not bad else 'FAIL'} (mode 6 vs xla_reference, {len(out)} checks, "
              f"tol {TOL:g}, {jax.process_count()} process(es))", flush=True)
    runtime.finalize_process(0 if not bad else 1)
    return 0 if not bad else 1


def test_kdft_plane_gate():
    import pytest
    pytest.skip("needs a ROCm device: python -m tests.test_kdft_plane_gate")


if __name__ == "__main__":
    sys.exit(main())
