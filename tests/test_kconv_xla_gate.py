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
raw-parent Sigma convolution, 9 the wedge transform, 10 the plane FFT, 11 the
chi0 node.  CUDA: four ranks, one GPU each,
``lx run ... -n 4 -- python3 tests/test_kconv_xla_gate.py``.  cpu: one process
with four host devices, ``JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_force_host_platform_device_count=4 python3 tests/test_kconv_xla_gate.py``.
Under pytest it skips.

The router's fallback (``ffi.fft.mathdx_refusal``) is gated too.  CUDA: with
``KCONV_AXIS_MAX`` forced to 1, every factory takes the XLA backend and must
match mathdx on the same grid.  cpu: with the mesh reported as CUDA, a k axis of
48 takes the XLA backend and must match the XLA reference exactly.
"""
import contextlib
import os
import sys

TOL = 1e-12


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
    return out


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
    if tuple(mesh.devices.shape) != (2, 2) or backend == "xla":
        print(f"SKIP: needs a 2x2 mesh with a vendor route, got backend {backend} on a "
              f"{tuple(mesh.devices.shape)} mesh", flush=True)
        runtime.finalize_process(0)
        return 0
    out = run_cases(mesh)
    extra, wrong = _fallback_cases(mesh, backend)
    out.update(extra)
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


def test_kconv_xla_gate_needs_p4_cuda():
    """Collected by pytest: the gate needs its own 2x2 mesh, so it runs as a script."""
    import pytest
    pytest.skip("run as a script on a 2x2 mesh: python3 tests/test_kconv_xla_gate.py")


if __name__ == "__main__":
    sys.exit(main())
