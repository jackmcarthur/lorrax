"""The ROCM k-box unfold (mode 7, ``lorrax_kdft_kconv_klead_unfold``) against the XLA route on the same device.

Every kept vendor route is gated against the XLA path on the same device
(``docs/architecture/decisions.md#xla-reference``).  Each case builds
``ffi.fft.make_kconv_klead_unfold`` twice on this job's mesh, by default (the k-box stage on a
ROCm mesh) and inside ``ffi.fft.xla_reference()`` (the service's unfold in XLA, the transpose,
kdft mode 2, the row selection), applies both to the same random operands and random valid
tables (``tests/test_kconv_xla_gate._tables``: parent rows, antiunitary rows with a partner tile,
-1 sources, unit phases, a random spin unitary per k) and requires
``max|a - b| <= 1e-12 * max|b|``, the family's tolerance.  Cases: the partner tile, conj(G)
partners, an x block, placed tables with live rows; ns 1, 2, 4; several grids; and the router's
refusal of a grid the single arm cannot hold (``GATE kdft-kbox-residency`` -> XLA).

ROCm: ``python -m tests.test_kdft_kbox_gate`` on any mesh (one GCD or a 2x2 P=4 mesh).
``--bench``: one GCD, the k-box route against the XLA route per call at production-shaped
tiles, and the XLA route's parts (the unfold gather, the transpose, kdft mode 2).  Under pytest
it skips (it needs a ROCm device).
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


def gate_cases(mesh):
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax
    from ffi import fft as F
    from lxkit import device_put_process_local
    from symmetry_maps import device_load_tables
    from tests.test_kconv_xla_gate import _gather, _rel, _tables

    rng = np.random.default_rng(20261015)
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    rnd = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    out = {}
    for kg, ns, m in (((3, 2, 2), 2, 6), ((5, 5, 1), 2, 6), ((4, 4, 4), 2, 5), ((3, 2, 2), 1, 6),
                      ((3, 2, 2), 4, 3), ((6, 6, 6), 2, 4), ((8, 8, 1), 1, 7)):
        nk = int(np.prod(kg))
        mu = m * px
        n_parent = max(2, nk // 4)
        tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (px, py))
        rows = sorted(rng.choice(nk, size=min(nk, 5), replace=False).tolist())
        G = put(rnd(n_parent, mu, ns, m * py, ns), P(None, "x", None, "y", None))
        Gt = put(rnd(n_parent, mu, ns, m * py, ns), P(None, "x", None, "y", None))
        W = put(rnd(nk, mu, m * py), P(None, "x", "y"))
        gl = device_load_tables(tg, mesh)
        live = put(np.asarray([1, m - 1], np.int32), P())
        tag = f"{kg[0]}x{kg[1]}x{kg[2]} ns{ns}"
        if F.rocm_kbox_target(mesh, kg, "kconv_klead_unfold", ns) is None:
            out[f"{tag} ROUTE"] = float("inf")        # every gate grid must take the k-box stage
            continue
        build = lambda: F.make_kconv_klead_unfold(mesh, kg, tg, store_rows=rows, norm="ortho", mult=0.7)
        f_kbox = build()
        with F.xla_reference():
            f_xla = build()
        for name, kw, gt in (("partner", {}, Gt), ("conj partner", dict(conj_partner=True), None),
                             ("x block", dict(rows=(1, 2, 3, 2) if m >= 6 else (0, 1, 2, 2)), Gt),
                             ("placed, live", dict(load=gl, live=live), Gt)):
            a = _gather(jax.block_until_ready(f_kbox(G, gt, W, **kw)))
            b = _gather(jax.block_until_ready(f_xla(G, gt, W, **kw)))
            out[f"{tag} {name}"] = _rel(a, b)
    # a grid the single arm cannot hold goes to XLA with the named gate (12^3 ns 2: 119.9 KB group)
    why = F.kdft_kbox_refusal((12, 12, 12), 2)
    out["refusal 12^3 ns2 -> GATE kdft-kbox-residency"] = 0.0 if why.startswith("GATE kdft-kbox-residency") else 1.0
    return out


def bench(mesh, grids, m, ns, reps):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import fft as F
    from lxkit import device_put_process_local
    from symmetry_maps import apply_unfold_load_tables_local, device_load_tables
    from tests.test_kconv_xla_gate import _rel, _tables

    rng = np.random.default_rng(7)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    rnd = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s))
    print(f"kbox-bench device {jax.devices()[0].device_kind} mesh {tuple(mesh.devices.shape)} "
          f"tile mx=my={m} ns={ns} (complex128)", flush=True)
    for kg in grids:
        nk = int(np.prod(kg))
        n_parent = max(2, nk // 8)
        tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (1, 1))
        rows = sorted(rng.choice(nk, size=n_parent, replace=False).tolist())
        G = put(rnd(n_parent, m, ns, m, ns), P(None, "x", None, "y", None))
        Gt = put(rnd(n_parent, m, ns, m, ns), P(None, "x", None, "y", None))
        W = put(rnd(nk, m, m), P(None, "x", "y"))
        gl = device_load_tables(tg, mesh)
        why = F.kdft_kbox_refusal(kg, ns)
        build = lambda: F.make_kconv_klead_unfold(mesh, kg, tg, store_rows=rows, norm="ortho", mult=1.0)
        with F.xla_reference():
            f_ref = build()
        j_ref = jax.jit(lambda g, gt, w, l: f_ref(g, gt, w, load=l))
        t_ref = _timed(j_ref, G, Gt, W, gl, reps=reps)
        # the XLA route's parts on the same operands (one rank: the local tables are the tables)
        flat = lambda a: a.reshape(n_parent, m * ns, m * ns)
        j_unf = jax.jit(lambda g, gt: apply_unfold_load_tables_local(flat(g), flat(gt), tg, np.asarray(tg.spin)))
        t_unf = _timed(j_unf, G, Gt, reps=reps)
        j_unt = jax.jit(lambda g, gt: jnp.transpose(apply_unfold_load_tables_local(
            flat(g), flat(gt), tg, np.asarray(tg.spin)), (0, 2, 1, 4, 3)))
        t_unt = _timed(j_unt, G, Gt, reps=reps)
        conv = F.make_local_kconv_klead(mesh, kg, norm="ortho", mult=1.0)
        T = jnp.asarray(rnd(nk, ns, m, ns, m))
        j_conv = jax.jit(lambda t, w: conv(t, w))
        t_conv = _timed(j_conv, T, jnp.asarray(rnd(nk, m, m)), reps=reps)
        line = (f"kbox-bench kgrid={kg[0]}x{kg[1]}x{kg[2]} nk={nk} n_parent={n_parent} ns={ns} m={m} | XLA route "
                f"{t_ref:.3f} ms (unfold {t_unf:.3f}, unfold+transpose {t_unt:.3f}, kdft mode 2 {t_conv:.3f})")
        if why:
            print(line + f" | k-box: refused ({why[:60]})", flush=True)
            continue
        f_kb = build()
        j_kb = jax.jit(lambda g, gt, w, l: f_kb(g, gt, w, load=l))
        t_kb = _timed(j_kb, G, Gt, W, gl, reps=reps)
        err = _rel(np.asarray(j_kb(G, Gt, W, gl)), np.asarray(j_ref(G, Gt, W, gl)))
        gbytes = (2 * n_parent * (ns * m) ** 2 + nk * m * m) * 16 / 1e9     # G read, V read, U written
        print(line + f" | k-box {t_kb:.3f} ms  x{t_ref / t_kb:.2f}  rel err {err:.2e}  "
              f"(compulsory HBM {gbytes:.3f} GB -> {gbytes / t_kb * 1e3:.0f} GB/s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--grids", default="5x5x1,8x8x1,12x12x1,4x4x4,6x6x6,8x8x8")
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
        grids = [tuple(int(v) for v in g.split("x")) for g in args.grids.split(",")]
        bench(mesh, grids, args.m, args.ns, args.reps)
        runtime.finalize_process(0)
        return 0
    out = gate_cases(mesh)
    bad = {k: v for k, v in out.items() if not v <= TOL}
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:44s} rel {v:.3e}", flush=True)
        print(f"KDFT_KBOX_GATE {'PASS' if not bad else 'FAIL'} (k-box mode 7 vs xla_reference on "
              f"{tuple(mesh.devices.shape)}): {len(out) - len(bad)}/{len(out)} within {TOL:g}", flush=True)
    rc = 1 if bad else 0
    runtime.finalize_process(rc)
    return rc


def test_kdft_kbox_gate_needs_rocm():
    """Collected by pytest: the gate needs a ROCm device, so it runs as a script."""
    import pytest
    pytest.skip("run as a script on a ROCm mesh: python -m tests.test_kdft_kbox_gate")


if __name__ == "__main__":
    sys.exit(main())
