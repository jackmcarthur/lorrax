"""P=4 gate for the route-G μ-batch kernel and Z store (docs/architecture/zeta_fit_mubatch.md).

Four processes, one GPU each, 2x2 mesh, on the fixtures of
``tests/zeta_mubatch_fixtures.py`` (glide group with spin mixing and an
antiunitary row, ns = 2; A-cubic, 48 operations, ns = 1) plus a ragged deck
where no axis divides the mesh (one operation, box (5, 5, 7) so N_r = 175 and
7 planes, 7 bands, 7 centroids, a 3x1x1 k grid with Q = 2 stored q, and a
ζ sphere that fills no whole G tile).  conj ψ(G) of the full zone (the typed
children) is sharded over G slots; the kernel's Z_q(μ, G) -- G-space pair
GEMM, one all-to-all to the μ owners, planes (cylinder, axis DFT, 2D FFT),
the k-convolution on the identity plan, LR+RL completion, forward plane FFT
and axis-phase accumulation -- goes through the write-once store (pinned host
tiles and a slab_io file) and both read layouts, and must match the dense sum
over the full-BZ children,

    Z_q(μ, G) = FFT_r[e^{-iq·r} (Z_q + conj Z_{-q})(μ, r)],
    Z_q(μ, r) = Σ_k Σ_ab D^L_{k,ab}(μ,r) conj D^R_{k+q,ab}(μ,r),

at 1e-12 (max-abs relative).  Red twin (TASTE 21): the ζ-sphere axis index
shifted by one must miss by more than 1e-3.
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/zeta_mubatch_p4.py``.
"""
from __future__ import annotations

import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import zeta_mubatch_fixtures as parity  # noqa: E402
from test_isdf_zq_parent_parity import _dense_pair_rhs  # noqa: E402

TAG = "[zeta-mubatch-p4]"
TOL, RED = 1.0e-12, 1.0e-3


def _ragged_fixture(mesh, rng):
    """One operation; no axis divisible by P = 4 (see the module docstring)."""
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import spinor_rotation_for_sym_row
    fg, kgrid, ns, nb, n_mu = (5, 5, 7), (3, 1, 1), 2, 7, 7
    ops = np.eye(3, dtype=np.int64)[None]
    kfrac = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) / np.asarray(kgrid, float)
    U = np.eye(2, dtype=np.complex128)[None]
    sym = SimpleNamespace(
        sym_matrices=ops, translations=np.zeros((1, 3)), irr_idx_k=np.arange(3, dtype=np.int32),
        sym_idx_k=np.zeros(3, np.int32), unfolded_kpts=kfrac, kirr_fullids=np.arange(3),
        spinor_action=lambda rows, *, nspinor: spinor_rotation_for_sym_row(
            U, np.asarray(rows), 2, nspinor=nspinor, R_cart=ops))
    grid = parity._grid_points(fg)
    cent_flat = np.sort(rng.choice(int(np.prod(fg)), n_mu, replace=False))
    plan = build_centroid_k_unfold_plan(sym, grid[cent_flat], fg, mesh, nspinor=ns,
                                        parent_k_frac=kfrac)
    return dict(plan=plan, fft_grid=fg, kgrid=kgrid, cent_flat=cent_flat,
                psi_parent=parity._crand(rng, 3, nb, ns, int(np.prod(fg))),
                kfull=kfrac, ops=ops, tnp=np.zeros((1, 3)), left=(0, 5), right=(2, nb),
                b_target=4)


def run_case(case, fx, mesh, scratch):
    from isdf import zeta_mubatch as zmb
    from gw.centroid_k_unfold import orbit_mu_batches
    from common.collectives import device_put_process_local as put
    from runtime.padding import pad_to_axis, padded_axis

    plan, fg, kgrid = fx["plan"], tuple(fx["fft_grid"]), tuple(fx["kgrid"])
    psi = fx["psi_parent"]
    nb, ns, n_rtot = psi.shape[1:]
    nk = int(np.prod(kgrid))
    mu_pad = int(plan.n_centroid_packed)
    idx = np.arange(nb)
    w_l = ((idx >= fx["left"][0]) & (idx < fx["left"][1])).astype(float)
    w_r = ((idx >= fx["right"][0]) & (idx < fx["right"][1])).astype(float)
    kint = np.asarray(list(np.ndindex(kgrid)))
    q_neg = np.ravel_multi_index(((-kint) % kgrid).T, kgrid).astype(np.int32)
    q_sel = np.arange(nk - 1, dtype=np.int32)                  # Q_pad > Q
    qf = kint[q_sel] / np.asarray(kgrid, float)
    G = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in fg), indexing="ij"),
                 -1).reshape(-1, 3)
    sph = [np.flatnonzero(np.sum((G + q) ** 2, 1) < 3.1) for q in qf]
    ngk = max(s.size for s in sph)
    sphere = np.stack([np.r_[s, np.repeat(s[:1], ngk - s.size)] for s in sph]).astype(np.int32)

    # Dense reference from the full-BZ children (no LORRAX kernel).
    Z = _dense_pair_rhs(parity._children(fx), fx["cent_flat"], kgrid, w_l, w_r, 0)
    Z = (Z + np.conj(Z[q_neg]))[q_sel]
    x = parity._grid_points(fg) / np.asarray(fg, float)
    Z = Z * np.exp(-2j * np.pi * qf @ x.T)[:, None, :]
    Z = np.fft.fftn(Z.reshape(Z.shape[:2] + fg), axes=(-3, -2, -1)).reshape(Z.shape)
    ref = plan.layout.axis.pack_host(
        np.take_along_axis(Z, sphere[:, None, :], axis=2), axis=1)

    q_axis = padded_axis(len(q_sel), 4, name="stored q rows")
    g_axis = padded_axis(ngk, 8, name="ζ-sphere G tiles")
    put_rep = lambda a: put(np.asarray(a), NamedSharding(mesh, P()))
    worst = 0.0

    # Route G: conj ψ(G) of the raw PARENTS sharded over G slots; the pair
    # GEMM on the parents, one all-to-all to the whole-orbit owners, the typed
    # unfold of the pair projectors there, then the planes.
    from common.wfn_transforms import psi_cylinder_tables
    kfull = np.asarray(fx["kfull"], dtype=np.float64)
    kpar = np.asarray(plan.k_parent_frac, dtype=np.float64)
    n_par = int(psi.shape[0])
    xg = parity._grid_points(fg) / np.asarray(fg, float)
    u = psi * np.exp(-2j * np.pi * (xg @ kpar.T).T)[:, None, None, :]
    cG = np.fft.fftn(u.reshape(*u.shape[:3], *fg), axes=(-3, -2, -1),
                     norm="ortho").reshape(u.shape)
    s_ax = padded_axis(n_rtot, 4, name="ψ sphere slots")
    cbar = np.conj(np.asarray(pad_to_axis(cG, s_ax, axis=3)))
    miller = G.astype(np.int32)
    g3 = np.asarray(pad_to_axis(np.broadcast_to(miller, (n_par,) + miller.shape).copy(),
                                s_ax, axis=1))
    sph_id = np.broadcast_to(np.arange(n_rtot, dtype=np.int32), (n_par, n_rtot)).copy()
    pslot, phase, anti = zmb.typed_child_G_tables(
        plan, fft_grid=fg, sphere_par=sph_id,
        gvec_child=np.broadcast_to(miller, (nk,) + miller.shape),
        ngk_child=np.full(nk, n_rtot), k_child=kfull)
    axis = int(np.argmax(fg))
    cyl = psi_cylinder_tables(np.broadcast_to(np.arange(n_rtot, dtype=np.int32),
                                              (nk, n_rtot)).copy(), fg, axis, ngkmax=n_rtot)
    zt = zmb.zeta_plane_tables(G[sphere].transpose(0, 2, 1).astype(np.int64),
                               np.full(len(q_sel), ngk), fg, axis, g_axis)
    plan_id = zmb.identity_kplan(kfull, parity._grid_points(fg)[fx["cent_flat"]], fg, mesh, ns)
    canon = np.asarray(plan.layout.axis.packed_to_canonical)
    cbar_d = parity._put(cbar, NamedSharding(mesh, P(None, None, None, ("x", "y"))))
    g3_d = parity._put(g3, NamedSharding(mesh, P(None, ("x", "y"), None)))
    ops = (put_rep(w_l), put_rep(w_r), put_rep(kpar))
    unf = tuple(put_rep(a) for a in (plan.irr_idx.astype(np.int32),
                                     plan.sym_idx.astype(np.int32), anti,
                                     plan.spin_action_full, pslot, phase, kfull))
    tabs = (tuple(put_rep(a) for a in cyl), tuple(put_rep(a) for a in zt))
    rank = NamedSharding(mesh, P(("x", "y")))

    def run_g(tabs, placement="host"):
        ob = zmb.owner_orbit_batches(plan, mu_pad, 4,
                                     c_target=max(1, int(fx["b_target"]) // 4))
        kern = zmb.make_route_g_kernel(
            mesh=mesh, plan_id=plan_id, kgrid=kgrid, fft_grid=fg, ns=ns, b=ob.b,
            q_sel=q_sel, q_axis=q_axis, q_neg=q_neg, qvec_frac=qf,
            n_col=int(cyl[0].shape[1]), n_s=int(cyl[0].shape[2]), n_pg=2, axis=axis,
            n_src=n_par)
        zs = zmb.ZStore(mesh=mesh, q_axis=q_axis, mu_pad=mu_pad, g_axis=g_axis, b=ob.b,
                        placement=placement, n_batch=ob.n_batch,
                        packed_from_slot=ob.slot_of_packed,
                        scratch_path=os.path.join(scratch, f"{case}_zstore.h5"))
        for beta in range(ob.n_batch):
            slots = ob.mu[beta]
            live = (slots >= 0).astype(np.float64)
            xmu = xg[fx["cent_flat"][canon[np.clip(slots, 0, None)]]] * live[:, None]
            lt = (parity._put(ob.left_perm[beta], rank), parity._put(ob.left_L[beta], rank))
            zs.write_batch(beta, kern(cbar_d, *ops, g3_d, put_rep(xmu), put_rep(live),
                                      *tabs, unf, lt))
        out = {lay: np.concatenate([parity._host(zs.read_tile(i, layout=lay))
                                    for i in range(zs.n_Gt)], axis=2)[:len(q_sel), :, :ngk]
               for lay in ("q", "g")}
        zs.close()
        return out

    for placement in ("host", "disk"):
        for lay, got in run_g(tabs, placement).items():
            e = parity._rel(got, ref)
            worst = max(worst, e)
            if jax.process_index() == 0:
                print(f"{TAG} {case:<11s} store={placement:<5s} read={lay}  rel={e:.2e}",
                      flush=True)
            if not e <= TOL:
                raise SystemExit(f"{TAG} FAIL {case} {placement}/{lay}: {e:.3e}")
    bad = (tabs[0], (tabs[1][0], put_rep(zt[1] + 1), tabs[1][2]))   # axis index off by one
    red_g = parity._rel(run_g(bad)["q"], ref)
    if jax.process_index() == 0:
        print(f"{TAG} {case} route G red twin (ζ axis index shifted by one): "
              f"rel={red_g:.2e}", flush=True)
    if not red_g > RED:
        raise SystemExit(f"{TAG} FAIL {case}: route G red twin did not fire ({red_g:.3e})")

    return worst


def main():
    if jax.process_count() != 4 or jax.device_count() != 4:
        raise SystemExit(f"{TAG} FAIL: needs 4 processes x 1 GPU")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    scratch = os.path.join(os.environ["SCRATCH"], "zeta_mubatch_p4_" + os.environ.get(
        "SLURM_JOB_ID", "0") + "_" + os.environ.get("SLURM_STEP_ID", "0"))
    if jax.process_index() == 0:
        os.makedirs(scratch, exist_ok=True)
    multihost_utils.sync_global_devices("zeta_mubatch_p4 scratch")
    worst = 0.0
    try:
        for case in ("glide_ns2", "acubic_ns1", "ragged_ns2"):
            rng = np.random.default_rng(2026_09_23)
            fx = (parity._glide_fixture(mesh, rng, 2) if case == "glide_ns2"
                  else parity._acubic_fixture(mesh, rng) if case == "acubic_ns1"
                  else _ragged_fixture(mesh, rng))
            worst = max(worst, run_case(case, fx, mesh, scratch))
    finally:
        multihost_utils.sync_global_devices("zeta_mubatch_p4 done")
        if jax.process_index() == 0:
            shutil.rmtree(scratch, ignore_errors=True)
    if jax.process_index() == 0:
        print(f"{TAG} PASS worst={worst:.2e} (tol {TOL:.0e})", flush=True)


if __name__ == "__main__":
    run_main_and_finalize(main)   # a failure keeps its traceback and a nonzero status
