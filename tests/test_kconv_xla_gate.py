"""The vendor k-convolution routes against the XLA backend on the same devices.

Every kept vendor route is gated against the XLA path on the same device
(``docs/architecture/decisions.md#xla-reference``).  Each case builds one
router factory twice on the same 2x2 mesh, once by default (mathdx on CUDA,
the host plan route on cpu) and once inside ``ffi.fft.xla_reference()``,
applies both to the same random operands and requires
``max|a - b| <= 1e-12 * max|b|``.  The unfold modes read
random but valid load tables (parent rows, antiunitary rows with a partner
tile, -1 sources, unit phases, a random spin unitary per k).

Modes: 0 pair, 2 + 3 the Sigma k-leading convolution (prep + apply), 3 and 5
the k-leading and k-minor transforms, 4 the BSE k-minor convolution, 7 the
raw-parent Sigma convolution, 8 the four-current Sigma convolution (W from its
parents; plain, and on placed tables with live rows), 9 the wedge transform, 10
the plane FFT, 11 the chi0 node (partner tiles, conj(G) partners, and the
four-current channel vertices with a Dirac-parity sign_c that is -1 on some k
rows, on placed tables with live rows).  CUDA: four ranks, one GPU each,
``lx run ... -n 4 -- python3 tests/test_kconv_xla_gate.py``.  cpu: one process
with four host devices, ``JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 tests/test_kconv_xla_gate.py``.
Under pytest it skips.

ROCm (the kdft routes, ``ffi.fft._ROCM_KLEAD_TARGETS``; ``kconv_backend`` stays ``'xla'``
there): the same cases, then :func:`kdft_cases` -- the k-minor split arm and the k-leading
multi-pass plan on grids to the verified bound (nk <= 10240: 16^3, 20^3, 97x97, 16x16x30,
128x64, 128x80, odd and prime axes), modes 5, 4 (both layouts), 3 and 2, complex128 and
complex64, the router's decision on each grid (a grid that silently took XLA fails), and the
named refusals past the bound.  One GCD or a 2x2 mesh:
``prun.sh <lane> <name> 1|4 <wt> -- python3 tests/test_kconv_xla_gate.py``.

CUDA also runs :func:`arm_cases`: one grid per arm the mathdx handler keeps (``ARM_CASES``,
``docs/architecture/kconv.md#arms``), each against the XLA reference, with the arm the build log
line names (``[kconv_mathdx] ... arm=<name>``, rank 0) asserted on an A100 (opt-in ``ARM_OPTIN``);
mode 11 there also runs ``acc=None`` and the q-row store ``rows=`` (against the XLA call on zeros,
its flat-k forward transform and the row gather), and an out-of-range ``rows=`` must refuse.

The router's fallback (``ffi.fft.mathdx_refusal``) is gated too.  CUDA: with
``KCONV_AXIS_MAX`` forced to 1, every factory takes the XLA backend and must
match mathdx on the same grid.  cpu: with the mesh reported as CUDA, a k axis of
48 takes the XLA backend and must match the XLA reference exactly.
"""
import contextlib
import os
import sys

TOL = 1e-12
TOL_C64 = 1e-5          # complex64 operands (both routes in single precision)
#: The kdft gate grids (ROCm): split-arm and large-nk shapes to the verified bound
#: (ffi.fft.KDFT_KMINOR_NK_MAX = 10240), odd / prime / unit axes, and two grids whose
#: out_layout-1 call takes the split arm inside the single arm's range (12^3, 16x16x15).
KDFT_GRIDS = ((16, 16, 16), (20, 20, 20), (97, 97, 1), (16, 16, 30), (128, 64, 1), (128, 80, 1),
              (64, 64, 1), (32, 32, 10), (13, 17, 19), (7, 11, 61), (101, 101, 1), (1, 128, 80),
              (3, 127, 10), (12, 12, 12), (16, 16, 15))
KDFT_GRIDS_C64 = ((16, 16, 16), (97, 97, 1), (20, 20, 20))
#: Past the bound: the router keeps these on XLA and names the gate.
KDFT_REFUSED = (((128, 81, 1), "GATE kdft-nk"), ((129, 1, 1), "GATE kdft-axis"))
#: The CUDA arm list (kconv.md#arms): (mode, k-grid, n_s, the arm the build log names on an A100).
#: "chain" is mode 7's stage / transform3 / store chain, reached through the legacy
#: ``lorrax_mathdx_kconv_klead_unfold_block`` target at an output spin block d < n_s.  Mode 9:
#: single below N_k = 128 (4^3), w2 from it (6^3, the bottom of its range, and 8^3).
ARM_OPTIN = 166912
ARM_CASES = (
    (7, (6, 6, 6), 2, "m7f"), (7, (6, 6, 6), 1, "m7f"), (7, (4, 4, 4), 4, "tt"), (7, (6, 6, 6), 2, "chain"),
    (7, (14, 14, 14), 2, "m7s"), (7, (14, 14, 14), 1, "arm2"), (7, (9, 9, 9), 4, "arm2"),
    (7, (22, 22, 22), 1, "arm1"),
    (9, (4, 4, 4), 1, "single"), (9, (6, 6, 6), 1, "w2"), (9, (8, 8, 8), 1, "w2"),
    (9, (14, 14, 14), 1, "split"),
    (11, (8, 8, 8), 2, "m11f"), (11, (8, 8, 8), 2, "m11q"), (11, (4, 4, 4), 1, "single"),
    (11, (4, 4, 4), 4, "tt"), (11, (11, 11, 11), 2, "split"),
)


def _rel(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _gather(x):
    from jax.experimental import multihost_utils
    return multihost_utils.process_allgather(x, tiled=True)


def _tables(rng, nk, n_parent, m_l, m_r, ns, n_l, n_r, mesh_shape, trs_rule="pair_transpose"):
    import numpy as np
    from symmetry_maps import UnfoldLoadTables
    px, py = mesh_shape
    row = rng.integers(0, n_parent, nk).astype(np.int32)
    trs = (rng.random(nk) < 0.4).astype(np.int32)
    lsrc = rng.integers(-1, m_l * n_l, (nk, px * m_l * n_l)).astype(np.int32)
    rsrc = rng.integers(-1, m_r * n_r, (nk, py * m_r * n_r)).astype(np.int32)
    phase = lambda shape: np.exp(2j * np.pi * rng.random(shape))
    q, _ = np.linalg.qr(rng.standard_normal((nk, ns, ns)) + 1j * rng.standard_normal((nk, ns, ns)))
    return UnfoldLoadTables(row=row, trs=trs, lsrc=lsrc, rsrc=rsrc,
                            mph=phase(lsrc.shape), nph=phase(rsrc.shape), spin=q.astype(np.complex128),
                            n_parent=n_parent, mesh_shape=(px, py),
                            conj_trs=int(trs_rule == "conj"))


@contextlib.contextmanager
def _axis_max(n):
    """``ffi.fft.KCONV_AXIS_MAX`` set to ``n`` inside the block (the fallback's trigger)."""
    from ffi import fft as F
    saved, F.KCONV_AXIS_MAX = F.KCONV_AXIS_MAX, n
    try:
        yield
    finally:
        F.KCONV_AXIS_MAX = saved


@contextlib.contextmanager
def _as_cuda():
    """The mesh reported as an NVIDIA one, so the router's CUDA decision runs on host devices."""
    import ffi.gate as G
    saved, G.mesh_ffi_platform = G.mesh_ffi_platform, lambda mesh, *a, **k: "CUDA"
    try:
        yield
    finally:
        G.mesh_ffi_platform = saved


def run_cases(mesh, kg=(3, 2, 2), arm=None, skip=()):
    """Every case on ``mesh``; returns ``{name: rel_error}``.

    Each factory is built twice: by default, then inside ``arm`` (default
    :func:`ffi.fft.xla_reference`)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax import shard_map
    from ffi import fft as F

    rng = np.random.default_rng(20261008)
    nk = int(np.prod(kg))
    arm = F.xla_reference if arm is None else arm
    ns = 2
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    m = 6                                   # local centroids per shard
    mu = m * px

    def put(a, spec):
        from lxkit import device_put_process_local
        return device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))

    def rnd(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def both(build, call):
        """``call(build())`` by default and under the XLA reference; (mathdx, xla)."""
        a = _gather(jax.block_until_ready(call(build())))
        with arm():
            fn = build()
        b = _gather(jax.block_until_ready(call(fn)))
        return a, b

    out = {}
    # mode 3, 5: the flat-k transforms
    X = put(rnd(nk, mu, mu), P(None, "x", "y"))
    for kind in ("ifftn", "fftn"):
        a, b = both(lambda: F.make_kfft_klead(mesh, kg, P(None, None, None, "x", "y"), kind=kind,
                                              norm="ortho"), lambda f: f(X))
        out[f"mode3 kfft_klead {kind}"] = _rel(a, b)
    Xm = put(rnd(mu, mu)[..., None, None, None] * rnd(1, 1, *kg), P("x", "y", None, None, None))
    a, b = both(lambda: F.make_kfft_kminor(mesh, kg, P("x", "y", None, None, None), kind="ifftn",
                                           norm="backward"), lambda f: f(Xm))
    out["mode5 kfft_kminor"] = _rel(a, b)

    # mode 2 + 3: the Sigma k-leading stored-kernel convolution
    T = put(rnd(nk, ns, mu, ns, mu), P(None, None, "x", None, "y"))
    Wk = put(rnd(nk, mu, mu), P(None, "x", "y"))
    def klead():
        return F.make_kconv_klead(mesh, kg, P(None, None, None, None, "x", None, "y"),
                                  P(None, None, None, "x", "y"), norm="ortho", mult=0.7)
    a, b = both(klead, lambda c: c.apply(T, c.prep(Wk)))
    out["mode2 kconv_klead"] = _rel(a, b)

    # mode 4: the BSE k-minor convolution
    Xb = put(rnd(2, mu, mu, 2, 3, nk), P(None, "x", "y", None, None, None))
    Kr = put(rnd(mu, mu, nk), P("x", "y", None))
    a, b = both(lambda: F.make_kconv_kminor(mesh, kg, P(None, "x", "y", None, None, None),
                                            P("x", "y", None), norm="ortho"),
                lambda f: f(Xb, Kr))
    out["mode4 kconv_kminor"] = _rel(a, b)

    # mode 0: the pair convolution, rank-local inside the caller's shard_map
    A = put(rnd(*kg, ns, 4, mu, ns), P(None, None, None, None, "x", "y", None))
    B = put(rnd(*kg, ns, 4, mu, ns), P(None, None, None, None, "x", "y", None))
    spec0 = P(None, None, None, None, "x", "y", None)
    def pair():
        return F.make_fused_conv_kpair(mesh, kg, perm_l=[1, 0], phase_l=[1, 1j],
                                       perm_r=[0, 1], phase_r=[-1, 1])
    a, b = both(pair, lambda f: shard_map(f, mesh=mesh, in_specs=(spec0, spec0),
                                          out_specs=P(None, None, None, "x", "y"),
                                          check_vma=False)(A, B))
    out["mode0 kconv_pair"] = _rel(a, b)

    # mode 10: the plane FFT read from its cylinder (no k grid: not a fallback case)
    nb, nc = 24, 30
    occ = rng.random(nb * nc) < 0.6
    n_col = int(occ.sum())
    pfc = np.full(nb * nc, n_col, np.int64)
    pfc[occ] = np.arange(n_col)
    Fc = put(rnd(4, 3, n_col), P(None, None, None))
    if "mode10" not in skip:
        a, b = both(lambda: F.make_plane_fft_gather(mesh, pfc, n_col, (nb, nc)),
                    lambda f: shard_map(f, mesh=mesh, in_specs=(P(),), out_specs=P(),
                                        check_vma=False)(Fc))
        out["mode10 plane_fft_gather"] = _rel(a, b)

    # the unfold modes: random valid tables, n_parent parent rows, antiunitary partners
    n_parent = 5
    tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (px, py))
    G = put(rnd(n_parent, mu, ns, mu, ns), P(None, "x", None, "y", None))
    Gt = put(rnd(n_parent, mu, ns, mu, ns), P(None, "x", None, "y", None))

    # mode 9: an interaction's operand read from its q wedge (scalar endpoints), compared
    # through the convolution that consumes it (the plan backend keeps it in k space)
    tw = _tables(rng, nk, n_parent, m, m, 1, 1, 1, (px, py))._replace(
        spin=np.ones((nk, 1, 1), np.complex128))
    Wp = put(rnd(n_parent, mu, mu), P(None, "x", "y"))
    Wt = put(rnd(n_parent, mu, mu), P(None, "x", "y"))
    a, b = both(lambda: (F.make_kfft_klead_unfold(mesh, kg, tw, norm="ortho"), klead()),
                lambda fs: fs[1].apply(T, fs[0](Wp, Wt)))
    out["mode9 kfft_klead_unfold"] = _rel(a, b)

    # mode 7: the Sigma convolution read from the raw-parent Green, stored at some rows
    rows = [0, 3, 7, 11]
    def unfold7():
        return (F.make_kconv_klead_unfold(mesh, kg, tg, store_rows=rows, norm="ortho", mult=0.7),
                klead())
    a, b = both(unfold7, lambda fs: fs[0](G, Gt, fs[1].prep(Wk)))
    out["mode7 kconv_klead_unfold"] = _rel(a, b)

    # mode 11: one chi0 node read from the raw-parent Green pair
    n_out = 2
    alpha = put(rnd(n_out), P())
    Gc = put(rnd(n_parent, mu, ns, mu, ns), P(None, "x", None, "y", None))
    Gct = put(rnd(n_parent, mu, ns, mu, ns), P(None, "x", None, "y", None))
    acc0 = rnd(n_out, nk, mu, mu)
    a, b = both(lambda: F.make_kconv_chi_unfold(mesh, kg, tg, n_out=n_out, complete=True,
                                                norm="ortho"),
                lambda f: f(put(acc0, P(None, None, "x", "y")), G, Gc, alpha, Gt, Gct))
    out["mode11 kconv_chi_unfold"] = _rel(a, b)
    # the same node with conj(G) partners (a Green of real weights) and no completion
    a, b = both(lambda: F.make_kconv_chi_unfold(mesh, kg, tg, n_out=n_out, complete=False,
                                                norm="ortho"),
                lambda f: f(put(acc0, P(None, None, "x", "y")), G, Gc, alpha))
    out["mode11 kconv_chi_unfold conj partner"] = _rel(a, b)

    # The production calls read placed tables and a padded pass's live rows.
    from symmetry_maps import device_load_tables
    gl = device_load_tables(tg, mesh)
    live = put(np.asarray([1, m - 1], np.int32), P())

    # mode 11 with channel vertices (the four-current stream): the Dirac parity sign_c is -1 on
    # some k rows and acts on Gc's k rows before the transform (gw.w_isdf._photon_chi_kconvs)
    vl = (((0, 1), (1, 1j)), ((1, 0), (-1, 1j)))
    vr = (((1, 0), (1j, -1)), ((0, 1), (1, -1j)))
    sign = np.where(rng.random(nk) < 0.5, -1.0, 1.0)
    sign[:2] = (1.0, -1.0)
    acc_v = rnd(len(vl) * len(vr), nk, mu, mu)

    def vertex():
        return F.make_kconv_chi_vertex(mesh, kg, tg, left_vertices=vl, right_vertices=vr,
                                       sign_c=sign, norm="ortho")
    a, b = both(vertex, lambda f: f(put(acc_v, P(None, None, "x", "y")), G, Gc, Gt, Gct))
    out["mode11 kconv_chi_vertex sign_c"] = _rel(a, b)
    a, b = both(vertex, lambda f: f(put(acc_v, P(None, None, "x", "y")), G, Gc, load=gl, live=live))
    out["mode11 kconv_chi_vertex conj partner, placed, live"] = _rel(a, b)

    # mode 8: the four-current Sigma convolution, W read from its parents through its own tables
    # (Lorentz endpoint actions of the vertices' widths)
    tw8 = _tables(rng, nk, n_parent, m, m, len(vl), len(vl), len(vr), (px, py))
    W8 = put(rnd(n_parent, mu, len(vl), mu, len(vr)), P(None, "x", None, "y", None))
    W8t = put(rnd(n_parent, mu, len(vl), mu, len(vr)), P(None, "x", None, "y", None))
    wl8 = device_load_tables(tw8, mesh)

    def lorentz():
        return F.make_kconv_lorentz_unfold(mesh, kg, tg, w_tables=tw8, left_vertices=vl,
                                           right_vertices=vr, store_rows=rows, norm="ortho", mult=0.7)
    a, b = both(lorentz, lambda f: f(G, Gt, W8, W8t))
    out["mode8 kconv_lorentz_unfold"] = _rel(a, b)
    a, b = both(lorentz, lambda f: f(G, Gt, W8, W8t, load=gl, w_load=wl8, live=live))
    out["mode8 kconv_lorentz_unfold placed, live"] = _rel(a, b)
    return out


@contextlib.contextmanager
def _stderr_into(lines):
    """fd 2 into ``lines`` inside the block (the C++ build log is unbuffered fprintf)."""
    import tempfile
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tf:
        os.dup2(tf.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tf.seek(0)
            lines.extend(tf.read().decode(errors="replace").splitlines())


def _logged_arms(lines, mode, kg, ns):
    """The ``arm=`` names of the build log lines of (mode, k-grid, n_s)."""
    import re
    pat = re.compile(r"\[kconv_mathdx\] .* mode=(\d+) kgrid=\((\d+),(\d+),(\d+)\) ns=(\d+) variant=(\d+) arm=(\S+)")
    want = (mode, *kg, ns)
    return [m.group(7) for m in map(pat.search, lines)
            if m and tuple(int(m.group(i)) for i in range(1, 6)) == want]


def arm_cases(mesh):
    """One case per mathdx arm (``ARM_CASES``): ``({name: rel_error}, [failures])``.  The arm
    named by the build log is checked on process 0 when the device's opt-in memory is the A100's."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import fft as F
    from symmetry_maps import local_unfold_load_tables

    rng = np.random.default_rng(20261017)
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    m, n_parent, mult = 6, 5, 0.7
    mu = m * px
    check_arm = jax.process_index() == 0 and F._optin_smem_bytes() == ARM_OPTIN
    out, wrong = {}, []

    def put(a, spec):
        from lxkit import device_put_process_local
        return device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))

    def rnd(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    def both(build, call):
        a = _gather(jax.block_until_ready(call(build())))
        with F.xla_reference():
            fn = build()
        return a, _gather(jax.block_until_ready(call(fn)))

    g_spec, w_spec, u_spec = P(None, "x", None, "y", None), P(None, "x", "y"), P(None, None, "x", None, "y")
    for mode, kg, ns, arm in ARM_CASES:
        nk = int(np.prod(kg))
        name = f"arm {arm} mode{mode} {'x'.join(map(str, kg))} ns{ns}"
        lines = []
        with _stderr_into(lines):
            if mode == 7:
                tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (px, py))
                G = put(rnd(n_parent, mu, ns, mu, ns), g_spec)
                Gt = put(rnd(n_parent, mu, ns, mu, ns), g_spec)
                Wk = put(rnd(nk, mu, mu), w_spec)
                rows = [0, 3, 7, 11]

                def build(tg=tg, kg=kg, rows=rows):
                    return (F.make_kconv_klead_unfold(mesh, kg, tg, store_rows=rows, norm="ortho", mult=mult),
                            F.make_kconv_klead(mesh, kg, P(None, None, None, None, "x", None, "y"),
                                               P(None, None, None, "x", "y"), norm="ortho", mult=mult))
                a, b = both(build, lambda fs: fs[0](G, Gt, fs[1].prep(Wk)))
                if arm == "chain":            # d = 1 of n_s = 2 through the legacy _block target
                    _, kout = F._store_row_map(rows, nk, "gate")
                    scale = F.ffi_fft_scale("ifftn", "ortho", nk) * F.ffi_fft_scale("fftn", "ortho", nk) * mult
                    fs = build()
                    wp = fs[1].prep(Wk)

                    def local(g, gt, v, tg=tg, kg=kg, kout=kout, scale=scale):
                        t = local_unfold_load_tables(tg)
                        n_par, mx, _, my, _ = (int(s) for s in g.shape)
                        flat = lambda x: x.reshape(n_par, mx * ns, my * ns)
                        o = jax.ShapeDtypeStruct((len(rows), 1, mx, 1, my), g.dtype)
                        return jax.ffi.ffi_call("lorrax_mathdx_kconv_klead_unfold_block", o)(
                            flat(g), flat(gt), t.row, t.trs, t.lsrc, t.rsrc, t.mph, t.nph, t.spin,
                            jnp.asarray(kout), v, nkx=np.int64(kg[0]), nky=np.int64(kg[1]), nkz=np.int64(kg[2]),
                            scale=np.float64(scale), conj_src=np.int64(0), spin_block=np.int64(1),
                            a0=np.int64(1), b0=np.int64(0), **F._mathdx_common())
                    blk = _gather(jax.block_until_ready(
                        F._sharded(local, mesh, (g_spec, g_spec, w_spec), u_spec)(G, Gt, wp)))
                    a, b = np.asarray(blk), np.asarray(a)[:, 1:2, :, 0:1, :]   # against m7f's full group
            elif mode == 9:
                tw = _tables(rng, nk, n_parent, m, m, 1, 1, 1, (px, py))._replace(
                    spin=np.ones((nk, 1, 1), np.complex128))
                Wp, Wt = put(rnd(n_parent, mu, mu), w_spec), put(rnd(n_parent, mu, mu), w_spec)
                a, b = both(lambda tw=tw, kg=kg: F.make_kfft_klead_unfold(mesh, kg, tw, norm="ortho"),
                            lambda f: f(Wp, Wt))
            else:
                tg = _tables(rng, nk, n_parent, m, m, ns, ns, ns, (px, py))
                Gv, Gvt, Gc, Gct = (put(rnd(n_parent, mu, ns, mu, ns), g_spec) for _ in range(4))
                n_out = 1 if arm == "m11q" else 2
                alpha = put(rnd(n_out), P())
                acc0 = rnd(n_out, nk, mu, mu)
                build = lambda tg=tg, kg=kg, n_out=n_out: F.make_kconv_chi_unfold(
                    mesh, kg, tg, n_out=n_out, complete=True, norm="ortho")
                if arm != "m11q":
                    a, b = both(build, lambda f: f(put(acc0, P(None, None, "x", "y")), Gv, Gc, alpha, Gvt, Gct))
                    out[name + " acc=None"] = _rel(*both(build, lambda f: f(None, Gv, Gc, alpha, Gvt, Gct)))
                else:
                    q = [0, 5, 17, nk - 1]
                    fn = build()
                    if not fn.q_store:
                        wrong.append(f"{name}: fn.q_store is False")
                        continue
                    a = np.asarray(_gather(jax.block_until_ready(fn(None, Gv, Gc, alpha, Gvt, Gct, rows=q))))
                    with F.xla_reference():
                        fx = build()
                        fft = F.make_kfft_klead(mesh, kg, P(None, None, None, "x", "y"), kind="fftn", norm="ortho")
                    z = fx(put(np.zeros((1, nk, mu, mu), np.complex128), P(None, None, "x", "y")), Gv, Gc, alpha, Gvt, Gct)
                    b = np.take(np.asarray(_gather(jax.block_until_ready(fft(z[0])))), q, axis=0)[None]
                    try:
                        fn(None, Gv, Gc, alpha, Gvt, Gct, rows=[nk])
                        wrong.append(f"{name}: rows=[{nk}] did not refuse")
                    except ValueError:
                        pass
        out[name] = _rel(a, b)
        if check_arm:
            got = _logged_arms(lines, mode, kg, ns)
            if arm not in got:
                wrong.append(f"{name}: the build log names {got or 'no build'} (an image built earlier "
                             f"in this process logs nothing)")
    from jax.experimental import multihost_utils
    n_wrong = int(np.max(multihost_utils.process_allgather(np.int32(len(wrong)))))
    if n_wrong and not wrong:
        wrong.append("arm checks failed on process 0")
    return out, wrong


def kdft_cases(mesh, grids=KDFT_GRIDS, grids_c64=KDFT_GRIDS_C64):
    """The ROCm kdft routes on large and split-arm grids: ``({name: (rel_error, tol)}, [decision
    failures])``.  Each factory is built by default (kdft) and inside ``ffi.fft.xla_reference()``
    on the same mesh; X carries d3*d4 > 1 rows, so the out_layout-1 store is the general
    permutation, and enough rows per shard for several row-chunk squares."""
    import warnings
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import fft as F
    from lxkit import device_put_process_local

    rng = np.random.default_rng(20261015)
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    mu, nu = 6 * px, 6 * py
    out, wrong = {}, []

    def put(a, spec):
        return device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))

    def rnd(dt, *shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)

    def both(build, call):
        a = _gather(jax.block_until_ready(call(build())))
        with F.xla_reference():
            fn = build()
        return a, _gather(jax.block_until_ready(call(fn)))

    for dt, gl in ((np.complex128, grids), (np.complex64, grids_c64)):
        tol, tag = (TOL, "") if dt == np.complex128 else (TOL_C64, " c64")
        for kg in gl:
            g = "x".join(map(str, kg))
            nk = int(np.prod(kg))
            for fac in ("kfft_kminor", "kconv_kminor", "kfft_klead", "kconv_klead"):
                if F.rocm_klead_target(mesh, kg, fac) is None:
                    wrong.append(f"{g} {fac} left kdft")
            Xm = put(rnd(dt, 3, mu, nu, *kg), P(None, "x", "y", None, None, None))
            for kind in ("ifftn", "fftn"):
                a, b = both(lambda: F.make_kfft_kminor(mesh, kg, P(None, "x", "y", None, None, None),
                                                       kind=kind, norm="ortho"), lambda f: f(Xm))
                out[f"kdft mode5 {g} {kind}{tag}"] = (_rel(a, b), tol)
            Xb = put(rnd(dt, 2, mu, nu, 2, 3, nk), P(None, "x", "y", None, None, None))
            Kr = put(np.exp(2j * np.pi * rng.random((mu, nu, nk))).astype(dt), P("x", "y", None))
            for ol in (0, 1):
                a, b = both(lambda: F.make_kconv_kminor(mesh, kg, P(None, "x", "y", None, None, None),
                                                        P("x", "y", None), norm="ortho", mult=0.7,
                                                        out_layout=ol), lambda f: f(Xb, Kr))
                out[f"kdft mode4 {g} out_layout {ol}{tag}"] = (_rel(a, b), tol)
            if dt != np.complex128:
                continue
            X3 = put(rnd(dt, nk, mu, nu), P(None, "x", "y"))
            a, b = both(lambda: F.make_kfft_klead(mesh, kg, P(None, None, None, "x", "y"), kind="ifftn",
                                                  norm="ortho"), lambda f: f(X3))
            out[f"kdft mode3 {g} ifftn"] = (_rel(a, b), tol)
            T = put(rnd(dt, nk, 2, mu, 2, nu), P(None, None, "x", None, "y"))
            Wk = put(rnd(dt, nk, mu, nu), P(None, "x", "y"))
            def klead():
                return F.make_kconv_klead(mesh, kg, P(None, None, None, None, "x", None, "y"),
                                          P(None, None, None, "x", "y"), norm="ortho", mult=0.7)
            a, b = both(klead, lambda c: c.apply(T, c.prep(Wk)))
            out[f"kdft mode2 {g}"] = (_rel(a, b), tol)
    for kg, gate in KDFT_REFUSED:
        for fac in ("kfft_kminor", "kconv_kminor", "kfft_klead"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tgt = F.rocm_klead_target(mesh, kg, fac)
            msg = " ".join(str(x.message) for x in w)
            rule = (F.kdft_kminor_refusal if "kminor" in fac else F.kdft_refusal)(kg)
            # the router announces once, on process 0; every rank reads its rule
            if tgt is not None or gate not in rule or (jax.process_index() == 0 and gate not in msg):
                wrong.append(f"{kg} {fac}: want XLA with {gate}, got target {tgt} / {rule[:80]!r} / {msg[:80]!r}")
    return out, wrong


def _fallback_cases(mesh, backend):
    """The router's XLA fallback: ``({name: rel_error}, [decision failures])``."""
    from ffi import fft as F
    wrong = []
    if backend == "mathdx":
        if F.kconv_backend(mesh, (3, 2, 2)) != "mathdx":
            wrong.append("(3,2,2) left mathdx")
        with _axis_max(1):
            if F.kconv_backend(mesh, (3, 2, 2)) != "xla":
                wrong.append("forced axis cap kept mathdx")
        out = run_cases(mesh, arm=lambda: _axis_max(1), skip=("mode10",))
        saved = F._probe_kconv_compile
        def failing(mesh):
            raise RuntimeError("GATE mathdx-probe: forced by the gate")
        F._probe_kconv_compile = failing
        try:
            if F.require_kconv(mesh, announce=False) != "xla" or F.kconv_backend(mesh, (3, 2, 2)) != "xla":
                wrong.append("a failed probe kept mathdx")
        finally:
            F._probe_kconv_compile = saved
            F._MATHDX_DOWN.clear()
        return {f"fallback {k}": v for k, v in out.items()}, wrong
    with _as_cuda():
        if F.kconv_backend(mesh, (48, 1, 1)) != "xla":
            wrong.append("axis 48 kept mathdx")
        out = run_cases(mesh, kg=(48, 1, 1), skip=("mode10",))
    return {f"axis48 {k}": v for k, v in out.items()}, wrong


def main() -> int:
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    from ffi import fft as F
    mesh = stack.mesh
    backend = F.kconv_backend(mesh)
    if backend == "xla" and F.rocm_klead_target(mesh, (3, 2, 2), "kconv_kminor") is not None:
        return _main_kdft(mesh, runtime)
    if tuple(mesh.devices.shape) != (2, 2) or backend == "xla":
        print(f"SKIP: needs a 2x2 mesh with a vendor route, got backend {backend} on a "
              f"{tuple(mesh.devices.shape)} mesh", flush=True)
        runtime.finalize_process(0)
        return 0
    out = run_cases(mesh)
    extra, wrong = _fallback_cases(mesh, backend)
    out.update(extra)
    if backend == "mathdx":
        extra, wrong_arm = arm_cases(mesh)
        out.update(extra)
        wrong += wrong_arm
    bad = {k: v for k, v in out.items() if not v <= TOL}
    if jax.process_index() == 0:
        for k, v in out.items():
            print(f"{'ok  ' if v <= TOL else 'FAIL'} {k:36s} rel {v:.3e}", flush=True)
        for w in wrong:
            print(f"FAIL decision: {w}", flush=True)
        print(f"KCONV_XLA_GATE {'PASS' if not (bad or wrong) else 'FAIL'} ({backend} vs xla, "
              f"and the XLA fallback): {len(out) - len(bad)}/{len(out)} cases within {TOL:g}, "
              f"{len(wrong)} decision failures", flush=True)
    rc = 1 if (bad or wrong) else 0
    runtime.finalize_process(rc)
    return rc


def _main_kdft(mesh, runtime) -> int:
    """The ROCm gate: :func:`run_cases` (kdft against XLA at the small grid) and
    :func:`kdft_cases`, on one GCD or a 2x2 mesh."""
    import jax
    shape = tuple(mesh.devices.shape)
    if shape not in ((1, 1), (2, 2)):
        print(f"SKIP: the kdft gate runs on one GCD or a 2x2 mesh, got {shape}", flush=True)
        runtime.finalize_process(0)
        return 0
    out = {k: (v, TOL) for k, v in run_cases(mesh).items()}
    extra, wrong = kdft_cases(mesh)
    out.update(extra)
    bad = {k for k, (v, tol) in out.items() if not v <= tol}
    if jax.process_index() == 0:
        for k, (v, tol) in out.items():
            print(f"{'ok  ' if k not in bad else 'FAIL'} {k:44s} rel {v:.3e} (tol {tol:g})", flush=True)
        for w in wrong:
            print(f"FAIL decision: {w}", flush=True)
        print(f"KCONV_XLA_GATE {'PASS' if not (bad or wrong) else 'FAIL'} (kdft vs xla on {shape}): "
              f"{len(out) - len(bad)}/{len(out)} cases within tolerance, {len(wrong)} decision failures",
              flush=True)
    rc = 1 if (bad or wrong) else 0
    runtime.finalize_process(rc)
    return rc


def test_kconv_xla_gate_needs_p4_cuda():
    """Collected by pytest: the gate needs its own 2x2 mesh, so it runs as a script."""
    import pytest
    pytest.skip("run as a script on a 2x2 mesh: python3 tests/test_kconv_xla_gate.py")


if __name__ == "__main__":
    sys.exit(main())
