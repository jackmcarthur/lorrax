"""The ROCM k-box modes against the XLA route on the same device: mode 7
(``lorrax_kdft_kconv_klead_unfold``), mode 9 (``lorrax_kdft_kfft_klead_unfold``) and mode 11
(``lorrax_kdft_kconv_chi_unfold``, ``lorrax_kdft_kconv_chi_vertex``).

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
Modes 9 and 11 (:func:`gate_cases_rspace`) the same way: ``make_kfft_klead_unfold`` with the
partner tile, the conjugate rule, unequal endpoint widths and placed tables with live rows;
``make_kconv_chi_unfold`` with partner tiles and the completion, conj(G) partners, placed tables
with live rows; ``make_kconv_chi_vertex`` with a sign_c and with conj(G) partners, placed, live;
grids up to 7^3 (mode 11 at ns 2) and an axis of 40 (each mode's own bound).

ROCm: ``python -m tests.test_kdft_kbox_gate`` on any mesh (one GCD or a 2x2 P=4 mesh).
``--bench``: one GCD, the k-box route against the XLA route per call at production-shaped
tiles, and the XLA route's parts (the unfold gather, the transpose, kdft mode 2); ``--mode 9`` or
``--mode 11`` benches those modes the same way (synthetic valid tables).  Under pytest
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



def gate_cases_rspace(mesh):
    """Modes 9 and 11 against xla_reference(): ``{case: rel_error}``."""
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    import jax
    from ffi import fft as F
    from lxkit import device_put_process_local
    from symmetry_maps import device_load_tables
    from tests.test_kconv_xla_gate import _gather, _rel, _tables

    rng = np.random.default_rng(20261016)
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    rnd = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    unitary = lambda nk, n: np.linalg.qr(rnd(nk, n, n))[0].astype(np.complex128)
    g_spec, acc_spec = P(None, "x", None, "y", None), P(None, None, "x", "y")
    out = {}

    def both(build, run):
        f_kb = build()
        with F.xla_reference():
            f_x = build()
        return _gather(jax.block_until_ready(run(f_kb))), _gather(jax.block_until_ready(run(f_x)))

    # mode 9: (grid, n_l, n_r, m)
    for kg, nl_, nr_, m in (((3, 2, 2), 1, 1, 6), ((5, 5, 1), 1, 1, 6), ((4, 4, 4), 2, 2, 4),
                            ((3, 2, 2), 2, 3, 4), ((8, 8, 8), 1, 1, 4), ((12, 12, 12), 1, 1, 2),
                            ((40, 8, 1), 1, 1, 3), ((6, 6, 1), 4, 4, 2)):
        nk = int(np.prod(kg))
        n_parent = max(2, nk // 4)
        tag = f"m9 {kg[0]}x{kg[1]}x{kg[2]} w{nl_}x{nr_}"
        if F.rocm_kbox_target(mesh, kg, "kfft_klead_unfold", nl_, nr_) is None:
            out[f"{tag} ROUTE"] = float("inf")
            continue
        base = _tables(rng, nk, n_parent, m, m, nl_, nl_, nr_, (px, py))
        base = base._replace(spin=unitary(nk, nl_), spin_r=None if nr_ == nl_ else unitary(nk, nr_))
        Wp = put(rnd(n_parent, m * px * nl_, m * py * nr_), P(None, "x", "y"))
        Wt = put(rnd(n_parent, m * px * nl_, m * py * nr_), P(None, "x", "y"))
        live = put(np.asarray([1, m - 1], np.int32), P())
        for name, tg, kw in (("partner", base, dict(Wt=Wt)), ("conj rule", base._replace(conj_trs=1), {}),
                             ("placed, live", base, dict(Wt=Wt, load=device_load_tables(base, mesh), live=live))):
            a, b = both(lambda: F.make_kfft_klead_unfold(mesh, kg, tg, norm="ortho"), lambda f: f(Wp, **kw))
            out[f"{tag} {name}"] = _rel(a, b)

    # mode 11: (grid, ns, m)
    vl = (((0, 1), (1, 1j)), ((1, 0), (-1, 1j)))
    vr = (((1, 0), (1j, -1)), ((0, 1), (1, -1j)))
    for kg, ns, m in (((3, 2, 2), 2, 6), ((5, 5, 1), 2, 6), ((4, 4, 4), 2, 4), ((3, 2, 2), 1, 6),
                      ((3, 2, 2), 4, 3), ((7, 7, 7), 2, 2), ((8, 8, 1), 1, 6), ((40, 8, 1), 2, 2)):
        nk = int(np.prod(kg))
        n_parent = max(2, nk // 4)
        tag = f"m11 {kg[0]}x{kg[1]}x{kg[2]} ns{ns}"
        if F.rocm_kbox_target(mesh, kg, "kconv_chi_unfold", ns) is None:
            out[f"{tag} ROUTE"] = float("inf")
            continue
        tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (px, py))
        mu, nu = m * px, m * py
        G, Gt, Gc, Gct = (put(rnd(n_parent, mu, ns, nu, ns), g_spec) for _ in range(4))
        n_out = 2
        alpha = put(rnd(n_out), P())
        acc0 = rnd(n_out, nk, mu, nu)
        gl = device_load_tables(tg, mesh)
        live = put(np.asarray([1, m - 1], np.int32), P())
        chi = lambda complete: (lambda: F.make_kconv_chi_unfold(mesh, kg, tg, n_out=n_out, complete=complete,
                                                                norm="ortho"))
        a, b = both(chi(True), lambda f: f(put(acc0, acc_spec), G, Gc, alpha, Gt, Gct))
        out[f"{tag} partner, complete"] = _rel(a, b)
        a, b = both(chi(False), lambda f: f(put(acc0, acc_spec), G, Gc, alpha))
        out[f"{tag} conj partner"] = _rel(a, b)
        a, b = both(chi(True), lambda f: f(put(acc0, acc_spec), G, Gc, alpha, load=gl, live=live))
        out[f"{tag} conj partner, complete, placed, live"] = _rel(a, b)
        if ns == 2:
            sign = np.where(rng.random(nk) < 0.5, -1.0, 1.0)
            sign[:2] = (1.0, -1.0)
            acc_v = rnd(len(vl) * len(vr), nk, mu, nu)
            vtx = lambda sg: (lambda: F.make_kconv_chi_vertex(mesh, kg, tg, left_vertices=vl, right_vertices=vr,
                                                              sign_c=sg, norm="ortho"))
            a, b = both(vtx(sign), lambda f: f(put(acc_v, acc_spec), G, Gc, Gt, Gct))
            out[f"{tag} vertex sign_c"] = _rel(a, b)
            a, b = both(vtx(None), lambda f: f(put(acc_v, acc_spec), G, Gc, load=gl, live=live))
            out[f"{tag} vertex conj partner, placed, live"] = _rel(a, b)
    # grids the single arms cannot hold go to XLA with the named gates
    checks = {"refusal m11 8^3 ns2 -> GATE kdft-kbox-residency": F.kdft_kbox_chi_refusal((8, 8, 8), 2),
              "refusal m9 16^3 w1x1 -> GATE kdft-kbox-residency": F.kdft_kbox_wedge_refusal((16, 16, 16), 1, 1),
              "refusal m11 41x2x1 -> GATE kdft-axis": F.kdft_kbox_chi_refusal((41, 2, 1), 1),
              "refusal m9 41x2x1 -> GATE kdft-axis": F.kdft_kbox_wedge_refusal((41, 2, 1), 1, 1)}
    for name, why in checks.items():
        out[name] = 0.0 if why.startswith(name.split("-> ")[1]) else 1.0
    return out


def bench_rspace(mesh, mode, grids, m, ns, reps):
    """Modes 9 / 11 per call on one GCD: the k-box route against the XLA route (synthetic tables)."""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import fft as F
    from lxkit import device_put_process_local
    from symmetry_maps import device_load_tables
    from tests.test_kconv_xla_gate import _rel, _tables

    rng = np.random.default_rng(9 if mode == 9 else 11)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    rnd = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s))
    w = 1 if mode == 9 else ns
    print(f"kbox-bench mode {mode} device {jax.devices()[0].device_kind} tile mx=my={m} "
          f"{'widths 1x1' if mode == 9 else f'ns={ns}'} (complex128)", flush=True)
    for kg in grids:
        nk = int(np.prod(kg))
        n_parent = max(2, nk // 8)
        tg = _tables(rng, nk, n_parent, m, m, w, w, w, (1, 1))
        gl = device_load_tables(tg, mesh)
        if mode == 9:
            why = F.kdft_kbox_wedge_refusal(kg, 1, 1)
            Wp = put(rnd(n_parent, m, m), P(None, "x", "y"))
            Wt = put(rnd(n_parent, m, m), P(None, "x", "y"))
            build = lambda: F.make_kfft_klead_unfold(mesh, kg, tg, norm="ortho")
            jit_of = lambda f: jax.jit(lambda a, b, l: f(a, b, load=l))
            args = (Wp, Wt, gl)
            gbytes = (n_parent * m * m + nk * m * m) * 16 / 1e9          # W read, Y written
        else:
            why = F.kdft_kbox_chi_refusal(kg, ns)
            G = put(rnd(n_parent, m, ns, m, ns), P(None, "x", None, "y", None))
            Gc = put(rnd(n_parent, m, ns, m, ns), P(None, "x", None, "y", None))
            alpha = put(rnd(1), P())
            acc = put(rnd(1, nk, m, m), P(None, None, "x", "y"))
            build = lambda: F.make_kconv_chi_unfold(mesh, kg, tg, n_out=1, complete=False, norm="ortho")
            jit_of = lambda f: jax.jit(lambda c, a, b, al, l: f(c, a, b, al, load=l))
            args = (acc, G, Gc, alpha, gl)
            gbytes = (2 * n_parent * (ns * m) ** 2 + 2 * nk * m * m) * 16 / 1e9   # G pair read, acc r+w
        with F.xla_reference():
            f_ref = build()
        j_ref = jit_of(f_ref)
        t_ref = _timed(j_ref, *args, reps=reps)
        line = (f"kbox-bench mode={mode} kgrid={kg[0]}x{kg[1]}x{kg[2]} nk={nk} n_parent={n_parent} m={m} | "
                f"XLA route {t_ref:.3f} ms")
        if why:
            print(line + f" | k-box: refused ({why[:70]})", flush=True)
            continue
        j_kb = jit_of(build())
        t_kb = _timed(j_kb, *args, reps=reps)
        err = _rel(np.asarray(j_kb(*args)), np.asarray(j_ref(*args)))
        print(line + f" | k-box {t_kb:.3f} ms  x{t_ref / t_kb:.2f}  rel err {err:.2e}  "
              f"(compulsory HBM {gbytes:.3f} GB -> {gbytes / t_kb * 1e3:.0f} GB/s)", flush=True)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--ns", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--grids", default="5x5x1,8x8x1,12x12x1,4x4x4,6x6x6,8x8x8")
    ap.add_argument("--mode", default="all", choices=("7", "9", "11", "all"),
                    help="which k-box modes to gate (all) or bench (7, 9 or 11)")
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
        if args.mode in ("9", "11"):
            bench_rspace(mesh, int(args.mode), grids, args.m, args.ns, args.reps)
        else:
            bench(mesh, grids, args.m, args.ns, args.reps)
        runtime.finalize_process(0)
        return 0
    out = {}
    if args.mode in ("7", "all"):
        out.update(gate_cases(mesh))
    if args.mode in ("9", "11", "all"):
        out.update(gate_cases_rspace(mesh))
    bad = {k: v for k, v in out.items() if not v <= TOL}
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:52s} rel {v:.3e}", flush=True)
        print(f"KDFT_KBOX_GATE {'PASS' if not bad else 'FAIL'} (k-box modes {args.mode} vs xla_reference on "
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
