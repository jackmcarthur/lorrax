"""P=4 parity gate: the ISDF k-convolution fallback tails on the flat-k FFT plan (common.fft_helpers.local_kfft3).

Four processes, one GPU each, 2x2 mesh, rank-varying operands.  Each of the
three call sites runs its non-native tail (formerly jnp.fft, now the library
FFT plan; selected by pair_kernel=None or its gate =off) and the native
conv_kpair / conv_kparent arm (gate =on) on the same inputs:

1. ``isdf.core.parent_projector_kconv`` (μ-batch and r-chunk ζ fit):
   identity plans on 3x4x2 and the awkward 7x9x5 grid (plus a host NumPy
   ``Z_q = Σ_k Σ_ab D^L_k conj D^R_{k+q}``) and the glide ns=2 plan (spin
   mixing, an antiunitary row); uneven tile b 37->40, cols 45->48 through
   ``runtime.padding``.
2. ``isdf.core.c_q_downfold`` (CCT/ZCT post-pair convolution), ns=2, 3x2x2,
   plus the dense pair sum of tests/test_isdf_zq_parent_parity.py.
3. ``isdf.core.c_q_from_psi_sm`` (face-parent CCT), glide ns=2 plan.

Parity 1e-13 (plan vs native), 1e-12 vs NumPy.  Red twins (each must miss by
> 1e-3): the plan tail fed a right gather rolled by one slot (1), an R-side
operand rolled by one k (2, 3).
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/kconv_plan_p4.py``.
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

TAG = "[kconv-plan-p4]"
TOL, TOL_REF, RED = 1.0e-13, 1.0e-12, 1.0e-3
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


def _arm(env, value):
    os.environ[env] = value


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


def kparent_case(mesh, rng, which):
    """parent_projector_kconv: plan tail (default) vs native conv_kparent."""
    from runtime.padding import padded_axis
    from ffi.fft import make_fused_conv_kparent
    from isdf.core import parent_projector_kconv, _conv_kpair_static_gamma
    import test_zeta_mubatch_sym_parity as parity
    if which == "glide":
        fx = parity._glide_fixture(mesh, rng, 2)
        plan, kgrid = fx["plan"], tuple(fx["kgrid"])
    else:
        kgrid = {"identity": (3, 4, 2), "awkward": (7, 9, 5)}[which]
        plan = _identity_plan(kgrid, mesh)
    ns, n_par, nk = int(plan.nspinor), int(plan.n_parent), int(np.prod(kgrid))
    rows = int(np.asarray(plan.sym_perm).shape[0])
    b_tag = padded_axis(37, mesh, name="owner b")
    c_tag = padded_axis(45, mesh, name="plane points")
    b, c = b_tag.carrier, c_tag.carrier
    D = _crand(rng, 2, 4, n_par, ns, b, ns, c)
    D[:, :, :, :, b_tag.logical:] = 0
    D[..., c_tag.logical:] = 0

    def gather(n_log, n_car):
        if which != "glide":
            return (np.broadcast_to(np.arange(n_car), (rows, n_car)).astype(np.int32),
                    np.zeros((rows, n_car, 3)))
        perm = np.stack([np.r_[rng.permutation(n_log), np.arange(n_log, n_car)]
                         for _ in range(rows)]).astype(np.int32)
        return perm, rng.integers(-1, 2, (rows, n_car, 3)).astype(float)
    lp, lL = gather(b_tag.logical, b)
    rp, rL = gather(c_tag.logical, c)
    _arm("LORRAX_CONV_KPARENT_FFI", "on")
    p_l, ph_l = _conv_kpair_static_gamma(None, ns)
    native = make_fused_conv_kparent(mesh, kgrid, ns, (b, c), perm_l=p_l, phase_l=ph_l,
                                     perm_r=p_l, phase_r=ph_l)
    _arm("LORRAX_CONV_KPARENT_FFI", "off")
    if native is None:
        raise SystemExit(f"{TAG} the native reference arm did not build")
    vtx = (np.arange(ns), np.ones(ns, dtype=np.complex128))
    rep, sh = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, XY))

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P(None, XY), P(), P(), P(), P()),
             out_specs=(P(XY),) * 3, check_vma=False)
    def run(D_, lp_, lL_, rp_, rL_):
        D_l, D_r = D_[0, 0], D_[1, 0]
        kw = dict(plan=plan, left_perm=lp_, left_L=lL_, right_L=rL_, kgrid=kgrid,
                  vertex_l=vtx, vertex_r=vtx)
        Zp = parent_projector_kconv(D_l, D_r, right_perm=rp_, **kw)
        Zn = parent_projector_kconv(D_l, D_r, right_perm=rp_, pair_kernel=native, **kw)
        Zr = parent_projector_kconv(D_l, D_r, right_perm=jnp.roll(rp_, 1, axis=1), **kw)
        return Zp[None], Zn[None], Zr[None]

    Zp, Zn, Zr = (_host(v) for v in run(_put(D, sh), *(_put(v, rep) for v in (lp, lL, rp, rL))))
    rec = dict(case=f"parent_projector_kconv_{which}", kgrid=list(kgrid), n_parent=n_par,
               b=f"{b_tag.logical}->{b}", cols=f"{c_tag.logical}->{c}",
               plan_vs_native=_rel(Zp, Zn),
               pad_zone_max=float(np.max(np.abs(Zp[:, :, b_tag.logical:]))
                                  + np.max(np.abs(Zp[..., c_tag.logical:]))),
               red_rolled_right=_rel(Zr, Zn))
    if which != "glide":
        kint = np.asarray(list(np.ndindex(kgrid)))
        ref = np.zeros_like(Zn)
        for q in range(nk):
            kq = np.ravel_multi_index(((kint + kint[q]) % kgrid).T, kgrid)
            ref[:, q] = np.einsum('pkambr,pkambr->pmr', D[0], np.conj(D[1][:, kq]))
        rec["ref_plan_vs_numpy"] = _rel(Zp, ref)
    return rec


def downfold_case(mesh, rng):
    """c_q_downfold (CCT/ZCT): plan tail (default) vs native conv_kpair."""
    from isdf.core import c_q_downfold
    from test_isdf_zq_parent_parity import _dense_pair_rhs
    kgrid, ns, n_rmu, nb = (3, 2, 2), 2, 8, 6
    nk = int(np.prod(kgrid))
    psi = _crand(rng, nk, nb, ns, n_rmu)
    bra = psi.conj().transpose(0, 3, 1, 2)                       # (k, μ, n, s)
    row = NamedSharding(mesh, P(None, "x", None, None))
    col = NamedSharding(mesh, P(None, None, None, "y"))
    (l0, l1), (r0, r1) = (0, 4), (2, 6)
    args = (_put(bra[:, :, l0:l1], row), _put(psi[:, l0:l1], col),
            _put(bra[:, :, r0:r1], row), _put(psi[:, r0:r1], col))

    def call(env, a):
        _arm("LORRAX_CONV_KPAIR_FFI", env)
        return _host(c_q_downfold(*a, kgrid=kgrid, mesh_xy=mesh))
    Cp = call("off", args)
    Cn = call("on", args)
    red = args[:3] + (_put(np.roll(psi[:, r0:r1], 1, axis=0), col),)
    Cr = call("off", red)
    _arm("LORRAX_CONV_KPAIR_FFI", "off")
    idx = np.arange(nb)
    ref = _dense_pair_rhs(psi, np.arange(n_rmu), kgrid,
                          ((idx >= l0) & (idx < l1)).astype(float),
                          ((idx >= r0) & (idx < r1)).astype(float), 0, 0)
    return dict(case="c_q_downfold", kgrid=list(kgrid), ns=ns, plan_vs_native=_rel(Cp, Cn),
                ref_plan_vs_numpy=_rel(Cp, ref), red_rolled_k=_rel(Cr, Cn))


def face_parent_case(mesh, rng):
    """c_q_from_psi_sm (face-parent CCT): plan tail (default) vs native conv_kparent."""
    from isdf.core import c_q_from_psi_sm
    import test_zeta_mubatch_sym_parity as parity
    fx = parity._glide_fixture(mesh, rng, 2)
    plan, kgrid = fx["plan"], tuple(fx["kgrid"])
    n_par, ns, mu, nb = int(plan.n_parent), 2, int(plan.n_centroid_packed), 6
    mun = _crand(rng, n_par, ns, mu, nb)                 # ψ_{n k̄ s}(r_μ), packed μ
    nmu = np.ascontiguousarray(mun.transpose(0, 3, 1, 2))  # the same ψ, (p, n, s, μ)

    def gemm(a, b):
        return jnp.einsum("qmk,qkn->qmn", a, b, optimize=True)
    gemm.in_sharding_a = NamedSharding(mesh, P(None, "x", "y"))
    gemm.in_sharding_b = gemm.in_sharding_a
    s_mun = NamedSharding(mesh, P(None, None, "x", "y"))
    s_nmu = NamedSharding(mesh, P(None, "x", None, "y"))
    w_l = jnp.asarray((np.arange(nb) < 4).astype(float))
    w_r = jnp.asarray((np.arange(nb) >= 2).astype(float))

    def call(env, nmu_host):
        _arm("LORRAX_CONV_KPARENT_FFI", env)
        return _host(c_q_from_psi_sm(
            _put(mun, s_mun), _put(nmu_host, s_nmu), w_l, w_r, k_unfold_plan=plan,
            kgrid=kgrid, mesh_xy=mesh, gemm=gemm))
    Cp = call("off", nmu)
    Cn = call("on", nmu)
    Cr = call("off", np.roll(nmu, 1, axis=0))
    _arm("LORRAX_CONV_KPARENT_FFI", "off")
    return dict(case="c_q_from_psi_sm", kgrid=list(kgrid), n_parent=n_par, mu=mu,
                plan_vs_native=_rel(Cp, Cn), red_rolled_k=_rel(Cr, Cn))


def main() -> int:
    import json
    only = sys.argv[1:]
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), XY)
    rng = np.random.default_rng(20260923)
    cases = dict(identity=lambda: kparent_case(mesh, rng, "identity"),
                 awkward=lambda: kparent_case(mesh, rng, "awkward"),
                 glide=lambda: kparent_case(mesh, rng, "glide"),
                 downfold=lambda: downfold_case(mesh, rng),
                 face_parent=lambda: face_parent_case(mesh, rng))
    recs = [fn() for name, fn in cases.items() if not only or name in only]
    bad = []
    for r in recs:
        for k, v in r.items():
            lim = TOL if k == "plan_vs_native" else TOL_REF if k.startswith("ref_") else None
            if lim is not None and not v <= lim:
                bad.append(f"{r['case']}.{k}={v:.2e} > {lim}")
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
