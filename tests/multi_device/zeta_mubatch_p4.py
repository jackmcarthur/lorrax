"""P=4 gate for the route-G μ-batch kernel and Z store (docs/architecture/zeta_fit_mubatch.md).

Four processes, one GPU each, 2x2 mesh, on the fixtures of
``tests/zeta_mubatch_fixtures.py`` (glide group with spin mixing and an
antiunitary row, ns = 2; a translated antiunitary glide, ns = 4;
A-cubic, 48 operations, ns = 1) plus a ragged deck
where no axis divides the mesh (one operation, box (5, 5, 7) so N_r = 175 and
7 planes, 7 bands, 7 centroids, a 3x1x1 k grid with Q = 2 stored q, and a
ζ sphere that fills no whole G tile), and the glide group at ns = 4 with the
three current vertices γ̃^{1,2,3} in ONE kernel (the bispinor current fit:
general U on the four-spinor, an antiunitary row, no LR+RL completion).  conj ψ(G) of the full zone (the typed
children) is sharded over G slots; the kernel's Z_q(μ, G) -- G-space pair
GEMM, one all-to-all to the μ owners, planes (cylinder, axis DFT, 2D FFT),
the plane k-convolution (the identity plan, phase and L/R split on load), LR+RL completion, forward plane FFT
and axis-phase accumulation -- goes through the write-once store (pinned host
tiles and a slab_io file) and its q-local read, and must match the dense sum
over the full-BZ children,

    Z_q(μ, G) = FFT_r[e^{-iq·r} (Z_q + conj Z_{-q})(μ, r)],
    Z_q(μ, r) = Σ_k Σ_ab D^L_{k,ab}(μ,r) conj D^R_{k+q,ab}(μ,r),

at 1e-12 (max-abs relative), with γ̃^{μ_L} on both endpoints' output spins
for a current channel.  Red twin (TASTE 21): the ζ-sphere axis index shifted
by one must miss by more than 1e-3; for the currents, channel 1 compared
against the γ̃^2 reference must miss too.  The transverse solve seam
(``_logical_solve('lu')``: the sign-aware ridged LU of an indefinite C) is
checked against a dense solve.
The G-space table is also checked on unequal parent-sphere extents against
an independently Fourier-transformed r-space action.  Reversing the glide
phase is its red twin.
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


def run_case(case, fx, mesh, scratch, vertices=(0,)):
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

    # Dense reference from the full-BZ children (no LORRAX kernel).  The charge
    # fit completes LR+RL; the currents train on LR alone.
    charge = tuple(vertices) == (0,)
    x = parity._grid_points(fg) / np.asarray(fg, float)

    def reference(v):
        Z = _dense_pair_rhs(parity._children(fx), fx["cent_flat"], kgrid, w_l, w_r, v)
        Z = (Z + np.conj(Z[q_neg]))[q_sel] if charge else Z[q_sel]
        Z = Z * np.exp(-2j * np.pi * qf @ x.T)[:, None, :]
        Z = np.fft.fftn(Z.reshape(Z.shape[:2] + fg), axes=(-3, -2, -1)).reshape(Z.shape)
        return plan.layout.axis.pack_host(
            np.take_along_axis(Z, sphere[:, None, :], axis=2), axis=1)
    refs = [reference(v) for v in vertices]

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
    canon = np.asarray(plan.layout.axis.packed_to_canonical)
    cbar_d = parity._put(cbar, NamedSharding(mesh, P(None, None, None, ("x", "y"))))
    g3_d = parity._put(g3, NamedSharding(mesh, P(None, ("x", "y"), None)))
    ops = (put_rep(w_l), put_rep(w_r), put_rep(kpar))
    unf = tuple(put_rep(a) for a in (plan.irr_idx.astype(np.int32),
                                     plan.sym_idx.astype(np.int32), anti,
                                     plan.spin_action_full, pslot, phase, kfull))
    tabs = (tuple(put_rep(a) for a in cyl[:2]), tuple(put_rep(a) for a in zt))
    rank = NamedSharding(mesh, P(("x", "y")))

    def run_g(tabs, placement="host", c_out=None, n_blk=1):
        """One store per channel; returns [{"q": Z}] in ``vertices`` order.
        ``c_out=1, n_blk=2`` streams each owner's rows through the planes one
        at a time, the plane axis in two blocks."""
        ob = zmb.owner_orbit_batches(plan, mu_pad, 4,
                                     c_target=max(1, int(fx["b_target"]) // 4))
        kern = zmb.make_route_g_kernel(
            mesh=mesh, kgrid=kgrid, fft_grid=fg, ns=ns, b=ob.b,
            q_sel=q_sel, q_axis=q_axis, q_neg=q_neg if charge else None, qvec_frac=qf,
            n_col=int(cyl[0].shape[1]), n_s=int(cyl[0].shape[2]),
            plane_from_col=np.asarray(cyl[2]), n_pg=2, axis=axis,
            n_src=n_par, vertices=vertices, c_out=c_out, n_blk=n_blk)
        stores = [zmb.ZStore(mesh=mesh, q_axis=q_axis, mu_pad=mu_pad, g_axis=g_axis,
                             b=ob.b, placement=placement, n_batch=ob.n_batch,
                             packed_from_slot=ob.slot_of_packed,
                             scratch_path=os.path.join(scratch, f"{case}_zstore{v}.h5"))
                  for v in vertices]
        for beta in range(ob.n_batch):
            slots = ob.mu[beta]
            live = (slots >= 0).astype(np.float64)
            xmu = xg[fx["cent_flat"][canon[np.clip(slots, 0, None)]]] * live[:, None]
            lt = (parity._put(ob.left_perm[beta], rank), parity._put(ob.left_L[beta], rank))
            rows = kern(cbar_d, *ops, g3_d, put_rep(xmu), put_rep(live), *tabs, unf, lt)
            for zs, r in zip(stores, rows):
                zs.write_batch(beta, r)
        out = [{"q": np.concatenate([parity._host(zs.read_tile(i))
                                     for i in range(zs.n_Gt)], axis=2)[:len(q_sel), :, :ngk]}
               for zs in stores]
        for zs in stores:
            zs.close()
        return out

    for placement, c_out, n_blk in (("host", None, 1), ("disk", None, 1), ("host", 1, 2)):
        for v, ref, got_v in zip(vertices, refs, run_g(tabs, placement, c_out, n_blk)):
            for lay, got in got_v.items():
                e = parity._rel(got, ref)
                worst = max(worst, e)
                if jax.process_index() == 0:
                    print(f"{TAG} {case:<11s} μ_L={v} store={placement:<5s} read={lay} "
                          f"c_out={c_out} n_blk={n_blk}  rel={e:.2e}", flush=True)
                if not e <= TOL:
                    raise SystemExit(f"{TAG} FAIL {case} μ_L={v} {placement}/{lay} "
                                     f"c_out={c_out} n_blk={n_blk}: {e:.3e}")
    bad = (tabs[0], (tabs[1][0], put_rep(zt[1] + 1), tabs[1][2]))   # axis index off by one
    red_g = parity._rel(run_g(bad)[0]["q"], refs[0])
    if jax.process_index() == 0:
        print(f"{TAG} {case} route G red twin (ζ axis index shifted by one): "
              f"rel={red_g:.2e}", flush=True)
    if not red_g > RED:
        raise SystemExit(f"{TAG} FAIL {case}: route G red twin did not fire ({red_g:.3e})")
    if len(vertices) > 1:
        # The vertex is load-bearing: channel 1's Z against channel 2's reference.
        red_v = parity._rel(run_g(tabs)[0]["q"], refs[1])
        if jax.process_index() == 0:
            print(f"{TAG} {case} vertex red twin (μ_L={vertices[0]} vs the "
                  f"μ_L={vertices[1]} reference): rel={red_v:.2e}", flush=True)
        if not red_v > RED:
            raise SystemExit(f"{TAG} FAIL {case}: vertex red twin did not fire ({red_v:.3e})")

    return worst


def _check_finish(mesh):
    """The finalize's one-collective reshard (q-local all-to-all) against the
    host reference, both μ splits."""
    from isdf import zeta_mubatch as zmb
    rng = np.random.default_rng(7)
    Q, Q_pad, a, b = 3, 4, 8, 4
    worst = 0.0
    shape, spec = (Q_pad, a, b), P(("x", "y"), None, None)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    want = x[:Q]
    for split in ("xy", "mu"):
        got = parity._host(zmb._to_mu_owner(mesh, Q, split)(
            parity._put(x, NamedSharding(mesh, spec))))
        e = parity._rel(got, want)
        worst = max(worst, e)
        if jax.process_index() == 0:
            print(f"{TAG} finish split={split}  rel={e:.2e}", flush=True)
        if not e <= TOL:
            raise SystemExit(f"{TAG} FAIL finish {split}: {e:.3e}")
    return worst


def _check_transverse_seam(mesh):
    """Route G's current-channel solve, ζ = (C + δI)⁻¹ Z for an INDEFINITE C,
    through the hoisted LU factor and ``zeta_mubatch._v_tile_kernel``'s
    ``'lu'`` seam, against a dense solve."""
    from isdf import zeta_mubatch as zmb
    from isdf.core import (_factor_c_q_transverse_lu, _transverse_lu_ridge,
                           zeta_factor_resident)
    from runtime.padding import padded_axis
    rng = np.random.default_rng(11)
    Q, mu, g = 3, 8, 8
    A = rng.standard_normal((Q, mu, mu)) + 1j * rng.standard_normal((Q, mu, mu))
    lam = np.linspace(-2.0, 3.0, mu)                    # both signs: indefinite
    Qm = np.linalg.qr(A)[0]
    C = np.einsum("qij,j,qkj->qik", Qm, lam, Qm.conj())
    Z = rng.standard_normal((Q, mu, g)) + 1j * rng.standard_normal((Q, mu, g))
    ridge = np.asarray(_transverse_lu_ridge(np.trace(C, axis1=1, axis2=2), mu))
    want = np.linalg.solve(C + ridge[:, None, None] * np.eye(mu), Z)
    LU, piv = _factor_c_q_transverse_lu(
        parity._put(C, NamedSharding(mesh, P(None, "x", "y"))), mesh, mu)
    worst = 0.0
    F = zeta_factor_resident(LU, piv, mesh, solver_kind="lu")
    q_axis = padded_axis(Q, 4, name="seam q rows")
    Zt = parity._put(np.concatenate([Z, np.zeros((q_axis.carrier - Q, mu, g))]),
                     NamedSharding(mesh, P(("x", "y"), None, None)))
    ngk = np.r_[np.full(Q, g), np.zeros(q_axis.carrier - Q)].astype(np.int32)
    ops = (parity._put(np.zeros((q_axis.carrier, g), complex),
                       NamedSharding(mesh, P(("x", "y"), None))),
           parity._put(ngk, NamedSharding(mesh, P(("x", "y")))),
           parity._put(np.zeros((q_axis.carrier, 1), np.int32),
                       NamedSharding(mesh, P(("x", "y"), None))))
    step = zmb._v_tile_kernel(mesh, "lu", mu, g, debug_m=False, with_v=False)
    acc = zmb._zero_accumulators(mesh, q_axis.carrier, mu, 1,
                                 debug_m=False, with_v=False)
    got = parity._host(step(F, Zt, *ops, jnp.int32(0), *acc)[3])[:Q]
    e = parity._rel(got, want)
    worst = max(worst, e)
    if jax.process_index() == 0:
        print(f"{TAG} transverse seam  rel={e:.2e}", flush=True)
    if not e <= 1e-10:
        raise SystemExit(f"{TAG} FAIL transverse seam: {e:.3e}")
    # Red twin: the PSD seam (cplus) on the same indefinite C drops half of it.
    from isdf import cplus
    lam_c, V = np.linalg.eigh(C)
    B = V * np.where(lam_c > 1e-8 * lam_c[:, -1:], 1 / np.sqrt(np.abs(lam_c)), 0)[:, None, :]
    red = parity._rel(np.asarray(cplus.apply(jnp.asarray(B), jnp.asarray(Z))), want)
    if jax.process_index() == 0:
        print(f"{TAG} transverse seam red twin (PSD cut on the indefinite C): "
              f"rel={red:.2e}", flush=True)
    if not red > RED:
        raise SystemExit(f"{TAG} FAIL transverse seam red twin did not fire ({red:.3e})")
    return worst


def _check_typed_g_spheres(mesh):
    """G-space tables against the independent r-space action, with ragged spheres.

    The translated antiunitary child has a non-real nonsymmorphic phase.
    Each parent has a different nonzero G count; inactive child slots must
    map to the zero sentinel rather than another parent's live coefficient.
    """
    from isdf.zeta_mubatch import typed_child_G_tables

    rng = np.random.default_rng(240925)
    fx = parity._glide_fixture(mesh, rng, 4, translated_anti=True)
    plan, fg = fx["plan"], tuple(fx["fft_grid"])
    N = int(np.prod(fg))
    nb, ns = fx["psi_parent"].shape[1:3]
    kpar = np.asarray(plan.k_parent_frac)
    x = parity._grid_points(fg) / np.asarray(fg, float)
    g = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1 / n) for n in fg),
                             indexing="ij"), -1).reshape(N, 3).astype(np.int32)
    cpar = np.zeros((plan.n_parent, nb, ns, N), np.complex128)
    sphere = np.full((plan.n_parent, 7), N, np.int32)
    for p, count in enumerate((3, 5, 7)):
        slots = rng.choice(N, count, replace=False)
        sphere[p, :count] = slots
        cpar[p, :, :, slots] = parity._crand(rng, count, nb, ns)
    u = np.fft.ifftn(cpar.reshape(*cpar.shape[:-1], *fg),
                     axes=(-3, -2, -1), norm="ortho").reshape(cpar.shape)
    fx["psi_parent"] = u * np.exp(2j * np.pi * (kpar @ x.T))[:, None, None, :]
    children = parity._children(fx)
    kfull = np.asarray(fx["kfull"])
    uc = children * np.exp(-2j * np.pi * (kfull @ x.T))[:, None, None, :]
    cchild = np.fft.fftn(uc.reshape(*uc.shape[:-1], *fg),
                         axes=(-3, -2, -1), norm="ortho").reshape(uc.shape)
    counts = np.array([np.count_nonzero(np.max(np.abs(row), axis=(0, 1)) > 1e-10)
                       for row in cchild], np.int32)
    width = int(counts.max()) + 2
    gc = np.zeros((plan.n_full, width, 3), np.int32)
    child_slots = []
    for k, row in enumerate(cchild):
        slots = np.flatnonzero(np.max(np.abs(row), axis=(0, 1)) > 1e-10)
        child_slots.append(slots)
        gc[k, :len(slots)] = g[slots]
    src, phase, anti = typed_child_G_tables(
        plan, fft_grid=fg, sphere_par=sphere, gvec_child=gc,
        ngk_child=counts, k_child=kfull)
    worst, red = 0.0, 0.0
    for k, slots in enumerate(child_slots):
        p = int(plan.irr_idx[k])
        value = np.take(cpar[p], sphere[p, src[k, :len(slots)]], axis=-1)
        with_phase = value * phase[k, :len(slots)]
        if anti[k]:
            with_phase = np.conj(with_phase)
        pred = np.einsum("ab,nbg->nag", plan.spin_action_full[k], with_phase)
        ref = np.take(cchild[k], slots, axis=-1)
        worst = max(worst, parity._rel(pred, ref))
        if int(plan.sym_idx[k]) == 3:
            wrong = value * np.conj(phase[k, :len(slots)])
            wrong = np.einsum("ab,nbg->nag", plan.spin_action_full[k],
                              np.conj(wrong))
            red = max(red, parity._rel(wrong, ref))
        assert np.all(src[k, len(slots):] == sphere.shape[1])
        assert np.all(phase[k, len(slots):] == 0)
    if jax.process_index() == 0:
        print(f"{TAG} typed G ragged/anti rel={worst:.2e} wrong-phase={red:.2e}",
              flush=True)
    if worst > TOL or red <= RED:
        raise SystemExit(f"{TAG} FAIL typed G ragged/anti: {worst:.3e}, red {red:.3e}")
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
        worst = max(worst, _check_typed_g_spheres(mesh))
        for case in ("glide_ns2", "acubic_ns1", "ragged_ns2", "glide_ns4_T",
                     "glide_anti_ns4_T"):
            rng = np.random.default_rng(2026_09_23)
            fx = (parity._glide_fixture(mesh, rng, 2) if case == "glide_ns2"
                  else parity._glide_fixture(mesh, rng, 4,
                                             translated_anti=case == "glide_anti_ns4_T")
                  if case.endswith("_T")
                  else parity._acubic_fixture(mesh, rng) if case == "acubic_ns1"
                  else _ragged_fixture(mesh, rng))
            worst = max(worst, run_case(case, fx, mesh, scratch,
                                        vertices=(1, 2, 3) if case.endswith("_T") else (0,)))
        worst = max(worst, _check_finish(mesh))
        worst = max(worst, _check_transverse_seam(mesh))
    finally:
        multihost_utils.sync_global_devices("zeta_mubatch_p4 done")
        if jax.process_index() == 0:
            shutil.rmtree(scratch, ignore_errors=True)
    if jax.process_index() == 0:
        print(f"{TAG} PASS worst={worst:.2e} (tol {TOL:.0e})", flush=True)


if __name__ == "__main__":
    run_main_and_finalize(main)   # a failure keeps its traceback and a nonzero status
