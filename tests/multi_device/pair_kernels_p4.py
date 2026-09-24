"""P=4 parity gate for the μ-batch pair kernels (src/isdf/pair_kernels.py, conv_kparent).

Four processes, one GPU each, 2x2 mesh.  Every rank works on its own data
(rank-varying operands sharded on a leading rank axis or on the G slice);
the batch b and the column extent are uneven logical sizes padded to their
mesh carriers through ``runtime.padding``.

1. Pair GEMM: ``pair_projectors_lr`` against the current per-chunk einsum
   (zeta_mubatch.py plane route) and a host NumPy sum, route-G shapes
   (ψ(G) sharded on its G slice); pad centroid slots and pad G columns must
   be exactly zero.  Red twin: the column store left unconjugated.
2. conv_kparent: the native arm of ``parent_projector_kconv`` against its
   XLA arm on an identity plan (plus a host NumPy
   ``Z_q = Σ_k Σ_ab D^L_k conj D^R_{k+q}``) and on the glide ns=2 plan (spin
   mixing, an antiunitary row) with in-tile gather tables.  Red twin: the
   native arm with the right gather rolled by one slot.

Parity at 1e-13 max-abs relative; each red twin must miss by > 1e-3.
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/pair_kernels_p4.py``.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, finalize_process  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from common.shard_map import shard_map  # noqa: E402

TAG = "[pair-kernels-p4]"
TOL, RED = 1.0e-13, 1.0e-3
XY = ("x", "y")


def _crand(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _put(x, sharding):
    x = np.asarray(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])


def _host(x):
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def gemm_case(mesh, rng):
    """Pair GEMM at route-G shapes: uneven b, uneven G slice, two band chunks."""
    from runtime.padding import padded_axis, pad_to_axis
    from isdf.pair_kernels import pair_projectors_lr
    nk, ns, nb, n_bc = 3, 2, 13, 2
    b_tag = padded_axis(37, mesh, name="batch b")               # 37 -> 40
    g_tag = padded_axis(45, mesh, name="G slice", spec=P(XY), axis=0)   # 45 -> 48
    bc_w = -(-nb // n_bc)                                        # 7: one pad band
    X = _crand(rng, nk, nb, ns, b_tag.logical)
    C = _crand(rng, nk, nb, ns, g_tag.logical)                   # ψ(G) coefficients
    w_l, w_r = rng.random(nb), rng.random(nb)
    ref_l = np.einsum('knam,kndg->kamdg', X * w_l[None, :, None, None], np.conj(C))
    ref_r = np.einsum('knam,kndg->kamdg', X * w_r[None, :, None, None], np.conj(C))

    def chunks(A):
        A = np.concatenate([A, np.zeros(A.shape[:1] + (n_bc * bc_w - nb,) + A.shape[2:])], 1)
        return np.ascontiguousarray(np.moveaxis(A.reshape(nk, n_bc, bc_w, *A.shape[2:]), 1, 0))
    xb = np.asarray(pad_to_axis(jnp.asarray(chunks(X)), b_tag, axis=-1))
    cb = np.asarray(pad_to_axis(jnp.asarray(chunks(C)), g_tag, axis=-1))
    wl, wr = (np.r_[w, np.zeros(n_bc * bc_w - nb)].reshape(n_bc, bc_w) for w in (w_l, w_r))
    rep, gsh = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, None, None, None, XY))
    out = P(None, None, None, None, XY)

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P(), P(None, None, None, None, XY), P(), P()),
             out_specs=(out,) * 6, check_vma=False)
    def run(xb_, cb_, wl_, wr_):
        D_l, D_r = pair_projectors_lr(xb_, lambda bc: jnp.conj(cb_[bc]), wl_, wr_)
        # The current code (zeta_mubatch.py, plane route): conj on ψ, one einsum per side.
        def body(carry, bc):
            x = jnp.transpose(xb_[bc], (0, 2, 3, 1))              # (k, a, m, n)
            yc = jnp.conj(cb_[bc])
            return (carry[0] + jnp.einsum('kamn,knbr->kambr', x * wl_[bc], yc),
                    carry[1] + jnp.einsum('kamn,knbr->kambr', x * wr_[bc], yc)), None
        z0 = jnp.zeros_like(D_l)
        (C_l, C_r), _ = jax.lax.scan(body, (z0, z0), jnp.arange(n_bc), unroll=1)
        R_l, R_r = pair_projectors_lr(xb_, lambda bc: cb_[bc], wl_, wr_)  # red: store not conjugated
        return D_l, D_r, C_l, C_r, R_l, R_r

    D_l, D_r, C_l, C_r, R_l, R_r = (_host(v) for v in run(
        _put(xb, rep), _put(cb, gsh), _put(wl, rep), _put(wr, rep)))
    lb, lg = b_tag.logical, g_tag.logical
    cut = (slice(None), slice(None), slice(0, lb), slice(None), slice(0, lg))
    pad = float(max(np.max(np.abs(D[:, :, lb:])) for D in (D_l, D_r))
                + max(np.max(np.abs(D[..., lg:])) for D in (D_l, D_r)))
    return dict(
        case="pair_gemm_routeG", b=f"{lb}->{b_tag.carrier}", G=f"{lg}->{g_tag.carrier}",
        n_bc=n_bc, bc_w=bc_w,
        vs_current=max(_rel(D_l, C_l), _rel(D_r, C_r)),
        vs_numpy=max(_rel(D_l[cut], ref_l), _rel(D_r[cut], ref_r)),
        pad_zone_max=pad,
        red_wrong_conj=min(_rel(R_l[cut], ref_l), _rel(R_r[cut], ref_r)))


def _identity_plan(kgrid, mesh, ns=2):
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import spinor_rotation_for_sym_row
    import test_zeta_mubatch_sym_parity as parity
    ops = np.eye(3, dtype=np.int64)[None]
    kfrac = np.asarray(list(np.ndindex(kgrid))) / np.asarray(kgrid, float)
    nk = kfrac.shape[0]
    U = np.eye(2, dtype=np.complex128)[None]
    sym = SimpleNamespace(
        sym_matrices=ops, translations=np.zeros((1, 3)), irr_idx_k=np.arange(nk, dtype=np.int32),
        sym_idx_k=np.zeros(nk, np.int32), unfolded_kpts=kfrac, kirr_fullids=np.arange(nk),
        spinor_action=lambda rows, *, nspinor: spinor_rotation_for_sym_row(
            U, np.asarray(rows), 2, nspinor=nspinor, R_cart=ops))
    fg = (4, 4, 4)
    return build_centroid_k_unfold_plan(sym, parity._grid_points(fg)[:8], fg, mesh,
                                        nspinor=ns, parent_k_frac=kfrac)


def kconv_case(mesh, rng, which):
    """native conv_kparent vs its XLA arm on one uneven (b, cols) tile per rank."""
    from runtime.padding import padded_axis
    from ffi.fft import conv_kpair_plan, make_fused_conv_kparent, CONV_KPARENT_GATE
    from isdf.core import parent_projector_kconv, _conv_kpair_static_gamma
    import test_zeta_mubatch_sym_parity as parity
    if which == "identity":
        kgrid = (3, 4, 2)
        plan = _identity_plan(kgrid, mesh)
    else:
        fx = parity._glide_fixture(mesh, rng, 2)
        plan, kgrid = fx["plan"], tuple(fx["kgrid"])
    ns, n_par, nk = int(plan.nspinor), int(plan.n_parent), int(np.prod(kgrid))
    rows = int(np.asarray(plan.sym_perm).shape[0])
    b_tag = padded_axis(37, mesh, name="owner b")               # 37 -> 40
    c_tag = padded_axis(45, mesh, name="plane points")          # 45 -> 48
    b, c = b_tag.carrier, c_tag.carrier
    P_ = 4
    D = _crand(rng, 2, P_, n_par, ns, b, ns, c)
    D[:, :, :, :, b_tag.logical:] = 0
    D[..., c_tag.logical:] = 0

    def gather(n_log, n_car):
        if which == "identity":
            return (np.broadcast_to(np.arange(n_car), (rows, n_car)).astype(np.int32),
                    np.zeros((rows, n_car, 3)))
        perm = np.stack([np.r_[rng.permutation(n_log), np.arange(n_log, n_car)]
                         for _ in range(rows)]).astype(np.int32)
        return perm, rng.integers(-1, 2, (rows, n_car, 3)).astype(float)
    lp, lL = gather(b_tag.logical, b)
    rp, rL = gather(c_tag.logical, c)
    arm, reason = conv_kpair_plan(mesh, kgrid, ns, (b, c), gate=CONV_KPARENT_GATE)
    p_l, ph_l = _conv_kpair_static_gamma(None, ns)
    native = make_fused_conv_kparent(mesh, kgrid, ns, (b, c), perm_l=p_l, phase_l=ph_l,
                                     perm_r=p_l, phase_r=ph_l)
    if native is None:
        raise SystemExit(f"{TAG} route predicate failed: conv_kparent arm={arm} ({reason})")
    vtx = (np.arange(ns), np.ones(ns, dtype=np.complex128))
    rep, sh = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, XY))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P(None, XY), P(), P(), P(), P()),
             out_specs=(P(XY),) * 3, check_vma=False)
    def run(D_, lp_, lL_, rp_, rL_):
        D_l, D_r = D_[0, 0], D_[1, 0]
        kw = dict(plan=plan, left_perm=lp_, left_L=lL_, right_L=rL_, kgrid=kgrid,
                  vertex_l=vtx, vertex_r=vtx)
        Zn = parent_projector_kconv(D_l, D_r, right_perm=rp_, pair_kernel=native, **kw)
        Zx = parent_projector_kconv(D_l, D_r, right_perm=rp_, pair_kernel=None, **kw)
        Zr = parent_projector_kconv(D_l, D_r, right_perm=jnp.roll(rp_, 1, axis=1),
                                    pair_kernel=native, **kw)
        return Zn[None], Zx[None], Zr[None]

    Zn, Zx, Zr = (_host(v) for v in run(_put(D, sh), *(_put(v, rep) for v in (lp, lL, rp, rL))))
    rec = dict(case=f"conv_kparent_{which}", kgrid=list(kgrid), n_parent=n_par, rows=rows,
               b=f"{b_tag.logical}->{b}", cols=f"{c_tag.logical}->{c}", arm=arm,
               native_vs_xla=_rel(Zn, Zx),
               pad_zone_max=float(np.max(np.abs(Zn[:, :, b_tag.logical:]))
                                  + np.max(np.abs(Zn[..., c_tag.logical:]))),
               red_rolled_right=_rel(Zr, Zx))
    if which == "identity":   # Z_q = Σ_k Σ_ab D^L_k conj D^R_{k+q} (identity plan)
        kint = np.asarray(list(np.ndindex(kgrid)))
        ref = np.zeros_like(Zx)
        for q in range(nk):
            kq = np.ravel_multi_index(((kint + kint[q]) % kgrid).T, kgrid)
            ref[:, q] = np.einsum('pkambr,pkambr->pmr', D[0], np.conj(D[1][:, kq]))
        rec["native_vs_numpy"] = _rel(Zn, ref)
    return rec


def main() -> int:
    import json
    devs = np.asarray(jax.devices()).reshape(2, 2)
    mesh = Mesh(devs, XY)
    rng = np.random.default_rng(20260923)
    recs = [gemm_case(mesh, rng), kconv_case(mesh, rng, "identity"),
            kconv_case(mesh, rng, "glide")]
    bad = []
    for r in recs:
        for k, v in r.items():
            if k.startswith(("vs_", "native_vs")) and not v <= TOL:
                bad.append(f"{r['case']}.{k}={v:.2e} > {TOL}")
            if k == "pad_zone_max" and v != 0.0:
                bad.append(f"{r['case']}.pad_zone_max={v:.2e} != 0")
            if k.startswith("red_") and not v > RED:
                bad.append(f"{r['case']}.{k}={v:.2e} <= {RED} (red twin did not fire)")
        if jax.process_index() == 0:
            print(TAG, json.dumps(r), flush=True)
    if jax.process_index() == 0:
        print(TAG, "FAIL: " + "; ".join(bad) if bad else "PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    rc = main()
    finalize_process()
    sys.exit(rc)
