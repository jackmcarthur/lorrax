"""The ROCM outer-product k-convolution against the XLA route on the same GCD: the BSE W term's
outer load (``lorrax_kdft_kconv_klead_outer``) and its fused decode
(``lorrax_kdft_kconv_klead_outer_decode``), ports of ``cpp/cufft/kconv_outer_cuda_ffi.cc``.

Every kept vendor route is gated against the XLA path on the same device
(``docs/architecture/decisions.md#xla-reference``).  Each case builds
``ffi.fft.make_local_kconv_klead_outer`` (and ``make_local_kconv_klead_outer_decode``) twice, by
default (the kdft target on a ROCm mesh; a case that silently took another route fails) and inside
``ffi.fft.xla_reference()`` (the einsum encode, the XLA k transforms, the einsum decode), applies
both to the same random operands on this process's first device (``LORRAX_KDFT_OUTER=all``: every
grid the kernels fit, past the measured routing rule, whose refusals are checked by name) and requires
``max|a - b| <= 1e-12 * max|b|`` (the family's tolerance).  Cases: K not a multiple of 4, conj_r
both ways, x and y extents that leave partial tiles, the tile rule's arms (8 x 8 at 5x5x1, smaller
boxes at 8x8x1, 12x12x1, 6^3, one column at 12^3), a grid of one point, a decode split into phases
(slots and the item counter), and the router's refusals by name (``GATE kdft-kconv-outer-tile``,
``GATE kdft-axis``, ``GATE kdft-kconv-outer-decode-tile``).

ROCm: ``python -m tests.test_kdft_outer_gate`` on any mesh (one GCD suffices).  ``--bench``: one GCD,
per call, the native route against the route ROCm took before it (XLA encode einsum + kdft mode 2
[+ XLA decode einsum]) at production-shaped tiles.  Under pytest it skips (it needs a ROCm device).
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


def _operands(rng, kg, na, mx, nb, my, K, nc, dev):
    import jax
    import numpy as np
    nk = int(np.prod(kg))
    rnd = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    put = lambda a: jax.device_put(np.asarray(a, np.complex128), dev)
    return (put(rnd(nk, na, mx, K)), put(rnd(nk, K, nb, my)), put(rnd(mx, my, nk)),
            put(rnd(nk, nc, na, mx)))


def gate_cases(mesh):
    import os
    import jax
    import numpy as np
    from ffi import fft as F
    from tests.test_kconv_xla_gate import _rel
    dev = jax.local_devices()[0]
    rng = np.random.default_rng(20261016)
    out = {}
    # The measured rule's refusals by name (default routing), then every kernel arm (A/B all).
    os.environ.pop("LORRAX_KDFT_OUTER", None)
    for kg, nc, gate in (((8, 8, 1), None, "kdft-kconv-outer-box"), ((5, 5, 1), 4, "kdft-kconv-outer-decode-slower")):
        why = F.rocm_outer_refusal(mesh, kg, "kconv_klead_outer_decode" if nc else "kconv_klead_outer", nc)
        out[f"default rule {kg} -> GATE {gate}"] = 0.0 if why and f"GATE {gate}" in why else float("inf")
    out["default rule (5, 5, 1) outer load -> kdft"] = 0.0 if F.rocm_outer_refusal(
        mesh, (5, 5, 1), "kconv_klead_outer") == "" else float("inf")
    os.environ["LORRAX_KDFT_OUTER"] = "all"
    #  kg            na  mx  nb  my   K  nc  conj
    cases = (((5, 5, 1), 2, 13, 2, 11, 3, 5, True),
             ((5, 5, 1), 2, 16, 2, 16, 8, 6, False),
             ((8, 8, 1), 2, 9, 2, 19, 7, 4, True),
             ((12, 12, 1), 1, 10, 2, 9, 5, 3, False),
             ((4, 4, 4), 2, 7, 2, 12, 4, 5, True),
             ((3, 2, 2), 2, 5, 1, 6, 1, 2, False),
             ((6, 6, 6), 1, 6, 1, 9, 2, 2, True),
             ((12, 12, 12), 1, 3, 1, 4, 2, 1, False),
             ((1, 1, 1), 2, 6, 2, 5, 3, 2, True),
             ((5, 5, 1), 2, 200, 1, 8, 4, 3, True))       # one item, 50 pairs: a phased decode
    for kg, na, mx, nb, my, K, nc, conj in cases:
        tag = f"{kg[0]}x{kg[1]}x{kg[2]} a{na} mx{mx} b{nb} my{my} K{K} nc{nc}{' conj' if conj else ''}"
        if F.rocm_outer_target(mesh, kg, "kconv_klead_outer") is None or \
                F.rocm_outer_target(mesh, kg, "kconv_klead_outer_decode", nc) is None:
            out[f"{tag} ROUTE"] = float("inf")             # every gate case must take the kdft target
            continue
        L, R, V, Pc = _operands(rng, kg, na, mx, nb, my, K, nc, dev)
        f_k = F.make_local_kconv_klead_outer(mesh, kg, norm="ortho", mult=0.7)
        p_k, a_k = F.make_local_kconv_klead_outer_decode(mesh, kg, norm="ortho")
        with F.xla_reference():
            f_x = F.make_local_kconv_klead_outer(mesh, kg, norm="ortho", mult=0.7)
            p_x, a_x = F.make_local_kconv_klead_outer_decode(mesh, kg, norm="ortho")
        u_k = jax.jit(lambda l, r, v: f_k(l, r, v, conj_r=conj))(L, R, V)
        u_x = jax.jit(lambda l, r, v: f_x(l, r, v, conj_r=conj))(L, R, V)
        out[f"{tag} U"] = _rel(u_k, u_x)
        d_k = jax.jit(lambda l, r, v, pc: a_k(l, r, v, p_k(pc), conj_r=conj))(L, R, V, Pc)
        d_x = jax.jit(lambda l, r, v, pc: a_x(l, r, v, p_x(pc), conj_r=conj))(L, R, V, Pc)
        out[f"{tag} A"] = _rel(d_k, d_x)
    # The refusals by name (the router keeps these on the unfused chain).
    for kg, nc, gate in (((16, 16, 16), None, "kdft-kconv-outer-tile"), ((41, 1, 1), None, "kdft-axis"),
                         ((8, 8, 8), 64, "kdft-kconv-outer-decode-tile")):
        why = F.rocm_outer_refusal(mesh, kg, "kconv_klead_outer_decode" if nc else "kconv_klead_outer", nc)
        out[f"refusal {kg} -> GATE {gate}"] = 0.0 if why and f"GATE {gate}" in why else float("inf")
    return out


def bench(mesh, shapes, reps):
    import os
    os.environ["LORRAX_KDFT_OUTER"] = "all"           # time the kernels on every shape
    import jax
    import jax.numpy as jnp
    import numpy as np
    from ffi import fft as F
    dev = jax.local_devices()[0]
    rng = np.random.default_rng(7)
    print("bench  k-grid    na  mx   nb  my   K  nc |  U: before ms  native ms  (x) |  A: before ms  native ms  (x)",
          flush=True)
    for kg, na, mx, nb, my, K, nc in shapes:
        L, R, V, Pc = _operands(rng, kg, na, mx, nb, my, K, nc, dev)
        conv = F.make_local_kconv_klead(mesh, kg, norm="ortho")           # kdft mode 2: the route before
        f_k = F.make_local_kconv_klead_outer(mesh, kg, norm="ortho")
        p_k, a_k = F.make_local_kconv_klead_outer_decode(mesh, kg, norm="ortho")

        def before_u(l, r, v):
            return conv(jnp.einsum("kaxK,kKby->kaxby", l, jnp.conj(r)), jnp.moveaxis(v, -1, 0))

        def before_a(l, r, v, pc):
            return jnp.einsum("kctM,ktMsN->kcsN", jnp.conj(pc), before_u(l, r, v))
        tb_u = _timed(jax.jit(before_u), L, R, V, reps=reps)
        tk_u = _timed(jax.jit(lambda l, r, v: f_k(l, r, v, conj_r=True)), L, R, V, reps=reps)
        tb_a = _timed(jax.jit(before_a), L, R, V, Pc, reps=reps)
        tk_a = _timed(jax.jit(lambda l, r, v, pc: a_k(l, r, v, p_k(pc), conj_r=True)), L, R, V, Pc, reps=reps)
        print(f"bench  {kg[0]}x{kg[1]}x{kg[2]:<4d} {na:3d} {mx:4d} {nb:3d} {my:4d} {K:3d} {nc:3d} | "
              f"{tb_u:12.3f} {tk_u:10.3f} {tb_u / tk_u:5.2f} | {tb_a:12.3f} {tk_a:10.3f} {tb_a / tk_a:5.2f}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--shapes", default="5x5x1:2:96:2:96:8:8,8x8x1:2:362:2:362:8:14,8x8x1:2:724:2:724:8:14,"
                                        "4x4x4:2:156:2:156:8:8")
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
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:60s} rel {v:.3e}", flush=True)
        print(f"KDFT_OUTER_GATE {'PASS' if not bad else 'FAIL'} (outer load + fused decode vs xla_reference, "
              f"{len(out)} checks, tol {TOL:g})", flush=True)
    runtime.finalize_process(0 if not bad else 1)
    return 0 if not bad else 1


def test_kdft_outer_gate():
    import pytest
    pytest.skip("needs a ROCm device: python -m tests.test_kdft_outer_gate")


if __name__ == "__main__":
    sys.exit(main())
