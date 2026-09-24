"""P=4 gate for the μ-batch batch kernel and Z store (docs/architecture/zeta_fit_mubatch.md).

Four processes, one GPU each, 2x2 mesh, on the fixtures of
``tests/test_zeta_mubatch_sym_parity.py`` (glide group with spin mixing and an
antiunitary row, ns = 2; A-cubic, 48 operations, ns = 1) plus a ragged deck
where no axis divides the mesh (one operation, box (5, 5, 7) so N_r = 175 and
7 planes, 7 bands, 7 centroids, a 3x1x1 k grid with Q = 2 stored q, and a
ζ sphere that fills no whole G tile): every pad is exercised.
For every ψ source (block cache; plane regeneration from resident or host
ψ(G)), both row owners (q-owned chunks, μ-owned rows) and every store
placement (device, host, slab_io disk) and read layout, the kernel's
Z_q(μ, G) -- parent-k pair GEMM, typed unfold and k-convolution, LR+RL
completion, the r->row transpose, local full-box FFTs, the sphere gather, the
write-once store and the packed read-back -- must match the dense sum over
the full-BZ children,

    Z_q(μ, G) = FFT_r[e^{-iq·r} (Z_q + conj Z_{-q})(μ, r)],
    Z_q(μ, r) = Σ_k Σ_ab conj D^L_{k,ab}(μ,r) D^R_{k+q,ab}(μ,r),

at 1e-12 (max-abs relative).  Red twin (TASTE 21): a box_from_slot table
rolled by one grid point must miss by more than 1e-3.
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

import test_zeta_mubatch_sym_parity as parity  # noqa: E402
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
                b_target=3, r_s_target=30)          # q-owned batches of 3 < P


class _Store:
    """The PsiGStore surface the kernels read: raw-parent ψ(G) on the whole box."""

    def __init__(self, psi, plan, fg, bcr, mesh):
        from common.collectives import device_put_process_local
        from runtime.padding import round_up
        n_parent, nb, ns, n_rtot = psi.shape
        # PsiGStore's transport carrier: each chunk padded to a multiple of P
        # with exact-zero bands (band_slot_tables drops them by index).
        width = max(round_up(hi - lo, 4) for lo, hi in bcr)
        psi = np.concatenate([psi, np.zeros((n_parent, bcr[-1][0] + width - nb, ns, n_rtot),
                                            psi.dtype)], axis=1)
        kv = np.asarray(plan.k_parent_frac)
        x = parity._grid_points(fg) / np.asarray(fg, float)
        u = psi * np.exp(-2j * np.pi * (x @ kv.T).T)[:, None, None, :]
        self.psi_G = np.fft.fftn(u.reshape(*u.shape[:3], *fg), axes=(-3, -2, -1),
                                 norm="ortho").reshape(u.shape)
        self.band_chunk_ranges = tuple(bcr)
        self._bpd_per_bc = tuple(round_up(hi - lo, 4) // 4 for lo, hi in bcr)
        self.local_band_chunk_shape = (n_parent, max(self._bpd_per_bc), ns, n_rtot)
        self.band_chunk_carrier = 4 * max(self._bpd_per_bc)
        rep = NamedSharding(mesh, P())
        self.g_index = device_put_process_local(np.ascontiguousarray(np.broadcast_to(
            np.arange(n_rtot, dtype=np.int32).reshape(fg), (n_parent,) + tuple(fg))), rep)
        self.kvecs_frac = device_put_process_local(kv, rep)

    def read_local_band_chunk(self, x, y, bc):
        p = int(x) * 2 + int(y)
        lo, _ = self.band_chunk_ranges[int(bc)]
        bpd = self._bpd_per_bc[int(bc)]
        out = np.zeros(self.local_band_chunk_shape, complex)
        out[:, :bpd] = self.psi_G[:, lo + p * bpd:lo + (p + 1) * bpd]
        return out


def run_case(case, fx, mesh, scratch):
    from isdf import zeta_mubatch as zmb
    from isdf.core import build_psi_G_resident_sm
    from gw.centroid_k_unfold import orbit_mu_batches
    from common.collectives import device_put_process_local as put
    from runtime.padding import pad_axis, pad_to_axis, padded_axis

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

    bcr = ((0, 4), (4, nb))                   # the last chunk is short when nb < 8
    store = _Store(psi, plan, fg, bcr, mesh)
    rep = NamedSharding(mesh, P())
    # The face band carrier is padded once, by name, as production's is.
    face = pad_axis(plan.layout.axis.pack_host(
        psi[:, :, :, fx["cent_flat"]].transpose(0, 2, 3, 1), axis=2), 4, axis=3,
        name="centroid face bands")
    nb_face = face.padded
    w_pad = lambda w: pad_axis(w, 4, axis=0, name="face band weights").array
    tables = tuple(put(np.asarray(a), rep) for a in zmb.band_slot_tables(
        store, band_start=0, nb_face=nb_face, weight_l=w_pad(w_l), weight_r=w_pad(w_r)))
    face = parity._put(face.array, NamedSharding(mesh, P(None, None, "x", "y")))
    q_axis = padded_axis(len(q_sel), 4, name="stored q rows")
    g_axis = padded_axis(ngk, 8, name="ζ-sphere G tiles")
    sph_pad = np.asarray(pad_to_axis(sphere, g_axis, axis=1))
    rank = NamedSharding(mesh, P(("x", "y")))

    def run(route, source, rows, placements, roll=False):
        mb = orbit_mu_batches(plan, mu_pad, 4 if rows == "mu" else 1,
                              b_target=int(fx["b_target"]))
        rs, pts, perm, wraps, planes = zmb.r_blocks(
            plan, fg, 4, route=route, r_s_target=int(fx["r_s_target"]))
        box = zmb.box_from_slots(pts, n_rtot)
        geo = (put(pts, rep), parity._put(perm, rank), parity._put(wraps, rank),
               put(planes, rep), put(np.roll(box, 1) if roll else box, rep),
               put(sph_pad, rep))
        cyl = None if source == "cache" else zmb.plane_cylinder(store, rs)
        src = (zmb.build_psi_block_cache(store, mesh=mesh, rs=rs, points=pts)
               if source == "cache" else build_psi_G_resident_sm(store, mesh_xy=mesh)
               if source == "resident" else jnp.zeros((1,), jnp.complex128))
        kern = zmb.make_batch_kernel(
            mesh=mesh, rs=rs, plan=plan, kgrid=kgrid, fft_grid=fg, ns=ns, b=mb.b,
            n_bc=len(bcr), bc_w=int(tables[0].shape[1]), nb_face=nb_face, q_sel=q_sel,
            q_axis=q_axis,
            q_neg=q_neg, sphere_idx=sph_pad, qvec_frac=qf, row_chunk=3,
            source=source, rows=rows, psi_G_store=store, cylinder=cyl,
            k_chunk=1 if source == "host" else None)
        stores = {pl: zmb.ZStore(
            mesh=mesh, q_axis=q_axis, mu_pad=mu_pad, g_axis=g_axis, b=mb.b,
            placement=pl, rows=rows, n_batch=mb.n_batch,
            packed_from_slot=mb.packed_to_slot(mu_pad),
            scratch_path=os.path.join(scratch, f"{case}_{source}_{rows}.h5"))
            for pl in placements}
        cyl_ops = cyl if cyl is not None else tuple(jnp.zeros((1,), jnp.int32) for _ in range(3))
        for beta in range(mb.n_batch):
            X = zmb.gather_batch_centroids(face, mb.mu[beta], mesh=mesh)
            out = kern(src, X, *tables, store.kvecs_frac, cyl_ops, geo,
                       (put(mb.left_perm[beta], rep), put(mb.left_L[beta], rep)))
            for zs in stores.values():
                zs.write_batch(beta, out)
        res = {}
        for pl, zs in stores.items():
            for layout in ("q",) if rows == "q" else ("q", "g"):
                t = [parity._host(zs.read_tile(i, layout=layout)) for i in range(zs.n_Gt)]
                res[pl, layout] = np.concatenate(t, axis=2)[:len(q_sel), :, :ngk]
            zs.close()
        return rs, mb, res

    worst = 0.0
    for route, source in (("cache", "cache"), ("planes", "resident"), ("planes", "host")):
        for rows in ("q", "mu"):
            rs, mb, res = run(route, source, rows, ("host", "disk"))
            for (pl, layout), got in res.items():
                e = parity._rel(got, ref)
                worst = max(worst, e)
                if jax.process_index() == 0:
                    print(f"{TAG} {case:<11s} src={source:<8s} rows={rows:<2s} "
                          f"store={pl:<6s} read={layout} n_sub={rs.n_sub} r_s={rs.r_s} "
                          f"b={mb.b}x{mb.n_batch}  rel={e:.2e}", flush=True)
                if not e <= TOL:
                    raise SystemExit(f"{TAG} FAIL {case} {source}/{rows}/{pl}/{layout}: {e:.3e}")
    _, _, res = run("cache", "cache", "q", ("host",), roll=True)
    red = parity._rel(res["host", "q"], ref)
    if jax.process_index() == 0:
        print(f"{TAG} {case} red twin (box_from_slot rolled by one): rel={red:.2e}", flush=True)
    if not red > RED:
        raise SystemExit(f"{TAG} FAIL {case}: red twin did not fire ({red:.3e})")
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
