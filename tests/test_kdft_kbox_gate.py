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
each arm on its grids by the rule (``ffi.fft.kdft_kbox_arm``: the split arm at 8^3-20^3, the nk
bound 40x16x16 and 40x40x1, mode 11's pair chunks through the scratch intermediate) and the split
arm forced on the single arm's grids (``LORRAX_KDFT_KBOX_ARM``); the refusals (``GATE kdft-axis``,
``GATE kdft-nk``, ``GATE kdft-kbox-residency``) and the arm rule are checked by name.

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



# Modes 9 / 11 gate grids: (grid, widths or ns, m per rank, forced arm or None for the rule).
# The rule's split grids (ffi.fft.kdft_kbox_arm): two passes at 6^3-10^3 and 2-D grids, three
# passes at 12^3-20^3, n_s = 4 at 10^3, 40x16x16 (the nk bound KDFT_KBOX_RSPACE_NK_MAX, axis 40);
# 40x40x1 (mode 9) and 24x24x1 (mode 11, KDFT_KBOX_CHI_2D_AXIS_MAX) the large 2-D grids.  The single arm's grids again with the split arms forced
# (LORRAX_KDFT_KBOX_ARM = split / split2 / split3), so each split variant meets every width, spin
# and vertex form, and the two-pass arm meets a three-pass grid.
M9_CASES = [((3, 2, 2), 1, 1, 6, None), ((5, 5, 1), 1, 1, 6, None), ((4, 4, 4), 2, 2, 4, None),
            ((3, 2, 2), 2, 3, 4, None), ((7, 7, 7), 1, 1, 4, None), ((6, 6, 6), 2, 2, 2, None),
            ((40, 8, 1), 1, 1, 3, None), ((6, 6, 1), 4, 4, 2, None),
            ((8, 8, 8), 1, 1, 4, None), ((10, 10, 10), 1, 1, 3, None), ((12, 12, 12), 1, 1, 3, None),
            ((12, 12, 12), 2, 3, 2, None), ((16, 16, 16), 1, 1, 2, None), ((20, 20, 20), 1, 1, 2, None),
            ((40, 16, 16), 1, 1, 2, None), ((40, 40, 1), 1, 1, 3, None), ((9, 7, 5), 2, 2, 2, None),
            ((3, 2, 2), 2, 3, 4, "split"), ((4, 4, 4), 2, 2, 4, "split"), ((6, 6, 1), 4, 4, 2, "split"),
            ((5, 5, 1), 1, 1, 6, "split"), ((40, 8, 1), 1, 1, 3, "split"), ((10, 10, 10), 4, 4, 2, None),
            ((3, 2, 2), 2, 3, 4, "split3"), ((4, 4, 4), 2, 2, 4, "split3"), ((6, 6, 6), 2, 2, 2, "split3"),
            ((12, 12, 12), 1, 1, 3, "split2")]
M11_CASES = [((3, 2, 2), 2, 6, None), ((5, 5, 1), 2, 6, None), ((4, 4, 4), 2, 4, None), ((3, 2, 2), 1, 6, None),
             ((3, 2, 2), 4, 3, None), ((7, 7, 7), 2, 2, None), ((8, 8, 1), 1, 6, None), ((24, 8, 1), 2, 2, None),
             ((8, 8, 8), 2, 4, None), ((10, 10, 10), 2, 2, None), ((12, 12, 12), 2, 2, None),
             ((13, 13, 13), 1, 2, None), ((16, 16, 16), 1, 2, None), ((20, 20, 20), 2, 2, None),
             ((40, 16, 16), 1, 2, None), ((24, 24, 1), 2, 3, None), ((16, 16, 1), 1, 3, None), ((9, 7, 5), 2, 2, None),
             ((3, 2, 2), 2, 6, "split"), ((3, 2, 2), 1, 6, "split"), ((3, 2, 2), 4, 3, "split"),
             ((5, 5, 1), 2, 6, "split"), ((7, 7, 7), 2, 2, "split2"), ((6, 6, 6), 4, 2, "split"),
             ((10, 10, 10), 4, 2, None), ((16, 16, 16), 1, 2, "split2"), ((3, 2, 2), 2, 6, "split3"),
             ((3, 2, 2), 1, 6, "split3"), ((3, 2, 2), 4, 3, "split3"), ((6, 6, 6), 2, 2, "split3")]


def _arm_ok(forced, taken):
    """A forced case must take its arm ("split": either split arm, the rule's passes)."""
    if not forced:
        return taken is not None
    return taken.startswith("split") if forced == "split" else taken == {"split2": "split"}.get(forced, forced)


class _Arm:
    """LORRAX_KDFT_KBOX_ARM for the k-box build and calls inside the block (None: the rule)."""
    def __init__(self, arm):
        self.arm = arm

    def __enter__(self):
        import os
        self.old = os.environ.get("LORRAX_KDFT_KBOX_ARM")
        if self.arm:
            os.environ["LORRAX_KDFT_KBOX_ARM"] = self.arm
        else:
            os.environ.pop("LORRAX_KDFT_KBOX_ARM", None)

    def __exit__(self, *exc):
        import os
        if self.old is None:
            os.environ.pop("LORRAX_KDFT_KBOX_ARM", None)
        else:
            os.environ["LORRAX_KDFT_KBOX_ARM"] = self.old


def gate_cases_rspace(mesh, which=("9", "11"), big=True):
    """Modes 9 and 11 against xla_reference(): ``{case: rel_error}``.  Each case's tag names the
    arm the handler takes (``ffi.fft.kdft_kbox_arm`` under the case's override)."""
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
    t0 = time.perf_counter()

    import os
    skip = int(os.environ.get("KBOX_GATE_SKIP_RSPACE", "0"))   # resume: skip the first N grid entries
    only = os.environ.get("KBOX_GATE_ONLY_RSPACE")              # one grid entry per process (index)
    only = None if only is None else int(only)

    class _Out(dict):                      # one progress line per case (rank 0), as it completes
        def __setitem__(self, k, v):
            super().__setitem__(k, v)
            if jax.process_index() == 0:
                print(f"gate-case [{time.perf_counter() - t0:7.1f} s] {k} rel {v:.2e}", flush=True)
    out = _Out()

    def both(arm, build, run):
        with _Arm(arm):
            f_kb = build()
            a = _gather(jax.block_until_ready(run(f_kb)))
        with F.xla_reference():
            f_x = build()
        return a, _gather(jax.block_until_ready(run(f_x)))

    def small(kg):
        return big or int(np.prod(kg)) <= 1000

    # mode 9: (grid, n_l, n_r, m, arm)
    for ci, (kg, nl_, nr_, m, arm) in enumerate(M9_CASES if "9" in which else ()):
        if not small(kg) or ci < skip or (only is not None and ci != only):
            continue
        nk = int(np.prod(kg))
        n_parent = max(2, nk // 4)
        with _Arm(arm):
            taken = F.kdft_kbox_arm(9, kg, nl_, nr_)[0]
            routed = F.rocm_kbox_target(mesh, kg, "kfft_klead_unfold", nl_, nr_)
        tag = f"m9 {kg[0]}x{kg[1]}x{kg[2]} w{nl_}x{nr_} {taken}{'(forced)' if arm else ''}"
        if routed is None or not _arm_ok(arm, taken):
            out[f"{tag} ROUTE"] = float("inf")
            continue
        base = _tables(rng, nk, n_parent, m, m, nl_, nl_, nr_, (px, py))
        base = base._replace(spin=unitary(nk, nl_), spin_r=None if nr_ == nl_ else unitary(nk, nr_))
        Wp = put(rnd(n_parent, m * px * nl_, m * py * nr_), P(None, "x", "y"))
        Wt = put(rnd(n_parent, m * px * nl_, m * py * nr_), P(None, "x", "y"))
        live = put(np.asarray([1, max(2, m - 1)], np.int32), P())   # >= 1 live row at m = 2
        for name, tg, kw in (("partner", base, dict(Wt=Wt)), ("conj rule", base._replace(conj_trs=1), {}),
                             ("placed, live", base, dict(Wt=Wt, load=device_load_tables(base, mesh), live=live))):
            a, b = both(arm, lambda: F.make_kfft_klead_unfold(mesh, kg, tg, norm="ortho"), lambda f: f(Wp, **kw))
            out[f"{tag} {name}"] = _rel(a, b)

    # mode 11: (grid, ns, m, arm)
    vl = (((0, 1), (1, 1j)), ((1, 0), (-1, 1j)))
    vr = (((1, 0), (1j, -1)), ((0, 1), (1, -1j)))
    vl3 = vl + (((0, 1), (-1j, -1)),)
    vr3 = vr + (((1, 0), (1, 1)),)
    for ci, (kg, ns, m, arm) in enumerate(M11_CASES if "11" in which else ()):
        if not small(kg) or ci < skip or (only is not None and ci != only):
            continue
        nk = int(np.prod(kg))
        n_parent = max(2, nk // 4)
        with _Arm(arm):
            taken = F.kdft_kbox_arm(11, kg, ns)[0]
            routed = F.rocm_kbox_target(mesh, kg, "kconv_chi_unfold", ns)
        tag = f"m11 {kg[0]}x{kg[1]}x{kg[2]} ns{ns} {taken}{'(forced)' if arm else ''}"
        if routed is None or not _arm_ok(arm, taken):
            out[f"{tag} ROUTE"] = float("inf")
            continue
        tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (px, py))
        mu, nu = m * px, m * py
        G, Gt, Gc, Gct = (put(rnd(n_parent, mu, ns, nu, ns), g_spec) for _ in range(4))
        n_out = 2
        alpha = put(rnd(n_out), P())
        acc0 = rnd(n_out, nk, mu, nu)
        gl = device_load_tables(tg, mesh)
        live = put(np.asarray([1, max(2, m - 1)], np.int32), P())   # >= 1 live row at m = 2
        chi = lambda complete, sb=None: (lambda: F.make_kconv_chi_unfold(
            mesh, kg, tg, n_out=n_out, complete=complete, norm="ortho", scratch_bytes=sb))
        a, b = both(arm, chi(True), lambda f: f(put(acc0, acc_spec), G, Gc, alpha, Gt, Gct))
        out[f"{tag} partner, complete"] = _rel(a, b)
        a, b = both(arm, chi(False), lambda f: f(put(acc0, acc_spec), G, Gc, alpha))
        out[f"{tag} conj partner"] = _rel(a, b)
        a, b = both(arm, chi(True), lambda f: f(put(acc0, acc_spec), G, Gc, alpha, load=gl, live=live))
        out[f"{tag} conj partner, complete, placed, live"] = _rel(a, b)
        if taken.startswith("split") and m * m >= 4:
            # several pair chunks (3 pairs each, a remainder) through the scratch intermediate
            sb = 3 * nk * 2 * ns * ns * 16
            a, b = both(arm, chi(False, sb), lambda f: f(put(acc0, acc_spec), G, Gc, alpha, Gt, Gct, live=live))
            out[f"{tag} chunks of 3 pairs, live"] = _rel(a, b)
        if ns == 2:
            sign = np.where(rng.random(nk) < 0.5, -1.0, 1.0)
            sign[:2] = (1.0, -1.0)
            acc_v = rnd(len(vl) * len(vr), nk, mu, nu)
            vtx = lambda sg, L=vl, R=vr: (lambda: F.make_kconv_chi_vertex(
                mesh, kg, tg, left_vertices=L, right_vertices=R, sign_c=sg, norm="ortho"))
            a, b = both(arm, vtx(sign), lambda f: f(put(acc_v, acc_spec), G, Gc, Gt, Gct))
            out[f"{tag} vertex sign_c"] = _rel(a, b)
            a, b = both(arm, vtx(None), lambda f: f(put(acc_v, acc_spec), G, Gc, load=gl, live=live))
            out[f"{tag} vertex conj partner, placed, live"] = _rel(a, b)
            if taken.startswith("split") and nk <= 1000:
                # 3 x 3 = 9 channels: two pencil passes of a pair's 8 lanes (lanes past the last idle)
                acc_9 = rnd(len(vl3) * len(vr3), nk, mu, nu)
                a, b = both(arm, vtx(sign, vl3, vr3), lambda f: f(put(acc_9, acc_spec), G, Gc, Gt, Gct))
                out[f"{tag} vertex 3x3 sign_c"] = _rel(a, b)
    # grids beyond a mode's verified range go to XLA with the named gates (within axis 40 every grid
    # has an arm: the three-pass z tile of one n_s = 4 group is <= 41 * 16 * 16 B)
    checks = {"refusal m11 40x16x17 -> GATE kdft-nk": F.kdft_kbox_chi_refusal((40, 16, 17), 1),
              "refusal m9 40x16x17 -> GATE kdft-nk": F.kdft_kbox_wedge_refusal((40, 16, 17), 1, 1),
              "refusal m11 41x2x1 -> GATE kdft-axis": F.kdft_kbox_chi_refusal((41, 2, 1), 1),
              "refusal m9 41x2x1 -> GATE kdft-axis": F.kdft_kbox_wedge_refusal((41, 2, 1), 1, 1),
              "refusal m11 40x40x1 ns2 -> GATE kdft-kbox-chi-2d": F.kdft_kbox_chi_refusal((40, 40, 1), 2),
              "refusal m11 32x32x1 ns1 -> GATE kdft-kbox-chi-2d": F.kdft_kbox_chi_refusal((32, 32, 1), 1),
              "refusal m11 40x16x16 ns4 -> served": "served" if not F.kdft_kbox_chi_refusal((40, 16, 16), 4) else "x"}
    for name, why in checks.items():
        out[name] = 0.0 if why.startswith(name.split("-> ")[1]) else 1.0
    # the arm rule (the production grids of this lane)
    arms = {"arm m9 5x5x1 w1x1 -> single": F.kdft_kbox_arm(9, (5, 5, 1), 1, 1)[0],
            "arm m9 12x12x1 w1x1 -> split": F.kdft_kbox_arm(9, (12, 12, 1), 1, 1)[0],
            "arm m9 10^3 w1x1 -> split": F.kdft_kbox_arm(9, (10, 10, 10), 1, 1)[0],
            "arm m9 16^3 w1x1 -> split3": F.kdft_kbox_arm(9, (16, 16, 16), 1, 1)[0],
            "arm m11 6^3 ns2 -> single": F.kdft_kbox_arm(11, (6, 6, 6), 2)[0],
            "arm m11 7^3 ns2 -> split": F.kdft_kbox_arm(11, (7, 7, 7), 2)[0],
            "arm m11 8^3 ns2 -> split": F.kdft_kbox_arm(11, (8, 8, 8), 2)[0],
            "arm m11 12^3 ns2 -> split3": F.kdft_kbox_arm(11, (12, 12, 12), 2)[0],
            "arm m11 13^3 ns1 -> split3": F.kdft_kbox_arm(11, (13, 13, 13), 1)[0],
            "arm m11 16^3 ns4 -> split3": F.kdft_kbox_arm(11, (16, 16, 16), 4)[0],
            "arm m11 24x24x1 ns2 -> split": F.kdft_kbox_arm(11, (24, 24, 1), 2)[0],
            "arm m11 16x16x1 ns1 -> split": F.kdft_kbox_arm(11, (16, 16, 1), 1)[0],
            "arm m11 12x12x1 ns2 -> single": F.kdft_kbox_arm(11, (12, 12, 1), 2)[0]}
    for name, got in arms.items():
        out[name] = 0.0 if got == name.split("-> ")[1] else 1.0
    return out


def bench_rspace(mesh, mode, grids, m, ns, reps, arm=None):
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
        t0 = time.perf_counter()
        say = lambda what: print(f"kbox-bench   [{time.perf_counter() - t0:7.1f} s] {kg} {what}", flush=True)
        with F.xla_reference():
            f_ref = build()
        j_ref = jit_of(f_ref)
        say("XLA route built")
        t_ref = _timed(j_ref, *args, reps=reps)
        say("XLA route timed")
        line = (f"kbox-bench mode={mode} kgrid={kg[0]}x{kg[1]}x{kg[2]} nk={nk} n_parent={n_parent} m={m} | "
                f"XLA route {t_ref:.3f} ms")
        if why:
            print(line + f" | k-box: refused ({why[:70]})", flush=True)
            continue
        with _Arm(arm):
            taken = F.kdft_kbox_arm(mode, kg, w if mode == 9 else ns, w if mode == 9 else None)
            j_kb = jit_of(build())
            jax.block_until_ready(j_kb(*args))
        say("k-box first call (build + image) done")
        with _Arm(arm):
            t_kb = _timed(j_kb, *args, reps=reps)
        err = _rel(np.asarray(j_kb(*args)), np.asarray(j_ref(*args)))
        print(line + f" | k-box[{taken[0]} tile {taken[1]}] {t_kb:.3f} ms  x{t_ref / t_kb:.2f}  rel err {err:.2e}  "
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
    ap.add_argument("--small", action="store_true", help="gate: modes 9/11 grids with nk <= 1000 only")
    ap.add_argument("--arm", default="", choices=("", "single", "split"),
                    help="bench: LORRAX_KDFT_KBOX_ARM for the k-box route (default: the rule)")
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
            bench_rspace(mesh, int(args.mode), grids, args.m, args.ns, args.reps, args.arm or None)
        else:
            bench(mesh, grids, args.m, args.ns, args.reps)
        runtime.finalize_process(0)
        return 0
    out = {}
    if args.mode in ("7", "all"):
        out.update(gate_cases(mesh))
    if args.mode in ("9", "11", "all"):
        which = ("9", "11") if args.mode == "all" else (args.mode,)
        out.update(gate_cases_rspace(mesh, which, big=not args.small))
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
