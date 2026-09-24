"""The μ-batch layout on raw parent k reproduces the r-chunk ζ fit.

Parity harness for docs/architecture/zeta_fit_mubatch.md ("Symmetry: parent
k"), until the batch loop consumes these modules.  On a 2x2 emulated CPU
mesh (four ranks) it runs the new layout with production pieces only:

* centroids in whole-orbit batches (``orbit_mu_batches``), replicated;
* the grid in orbit-closed rank blocks (``orbit_r_blocks``), one run per
  flat ``('x','y')`` rank, sub-blocks scanned;
* pair projectors on the raw parents only, unfolded and k-convolved by
  ``isdf.core.parent_projector_kconv`` with the batch-local and block-local
  tables,

and compares, at the same (q, μ, r):

L2  the unfolded full-BZ pair projector against D built from the full-BZ
    children of the typed wavefunction action (the r-chunk path's own
    intermediate, as the μ-batch agent's full-BZ loop consumes it);
L3  Z_q(μ, r) against the incumbent r-chunk kernel (``z_q_from_psi_sm`` on
    ``plan.real_grid_tiles``) and against direct NumPy k/band sums;
L4  ζ_q = C_q⁺ Z_q (``solve_zeta_charge_dense``, rank_truncate, one C_q for
    both) and a V_q contraction on the box (charge cases only).

Red twins (must be visibly wrong): a permuted batch table, a permuted block
table, and a split orbit forced through with an identity fallback.

Fixtures: A-cubic (diamond-H2, 48 operations incl. glides, 3 parents of 8
k, 48 centroids, real SymMaps from the committed WFN) with synthetic parent
states; the order-two glide group of ``test_isdf_zq_parent_parity`` with
spin rotation and an antiunitary row (ns = 2; ns = 4 with a current vertex).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_TOL = 1.0e-12
_RED = 1.0e-3
_CASES = ("acubic_ns1", "glide_ns2", "glide_ns4_v1")
_HERE = Path(__file__).resolve().parent


def _crand(rng, *shape):
    import numpy as np
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


def _put(x, sharding):
    """Host array onto a (possibly multi-process) sharding, every process
    holding the same host copy."""
    import jax
    import numpy as np
    x = np.asarray(x)
    return jax.make_array_from_callback(x.shape, sharding, lambda i: x[i])


def _host(x):
    """Global value of a (possibly multi-process) array on every process."""
    import numpy as np
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _grid_points(fft_grid):
    import numpy as np
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in fft_grid), indexing="ij")
    return np.stack([ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)],
                    axis=1).astype(np.int32)


def _acubic_fixture(mesh, rng):
    """Real A-cubic symmetry tables, synthetic parents on its 12^3 grid."""
    import numpy as np
    from file_io import WfnLoader
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan

    root = _HERE / "core" / "fixtures" / "A-cubic"
    with WfnLoader(root / "WFN.h5", backend="eager",
                   qe_schema=root / "data-file-schema.xml") as loader:
        sym = loader.symmetry()
        fft_grid = tuple(int(v) for v in loader.fft_grid)
        k_parent = np.asarray(loader.kvecs(k=sym.parent_k_domain))
        kgrid = tuple(int(v) for v in loader.kgrid)
    frac = np.loadtxt(root / "centroids_frac_48.txt")
    cent_idx = (np.rint(frac * np.asarray(fft_grid)) % np.asarray(fft_grid)).astype(np.int32)
    plan = build_centroid_k_unfold_plan(sym, cent_idx, fft_grid, mesh,
                                        nspinor=1, parent_k_frac=k_parent)
    fg = np.asarray(fft_grid)
    cent_flat = cent_idx[:, 0] * fg[1] * fg[2] + cent_idx[:, 1] * fg[2] + cent_idx[:, 2]
    nb = 8
    psi_parent = _crand(rng, plan.n_parent, nb, 1, int(fg.prod()))
    return dict(plan=plan, fft_grid=fft_grid, kgrid=kgrid, cent_flat=cent_flat,
                psi_parent=psi_parent, kfull=np.asarray(sym.unfolded_kpts),
                ops=np.asarray(sym.sym_matrices)[:plan.n_sym_spatial],
                tnp=np.asarray(sym.translations)[:plan.n_sym_spatial],
                band_chunks=((0, nb),), left=(0, 5), right=(2, 8), b_target=24,
                r_s_target=160, tile_width=192)


def _glide_fixture(mesh, rng, ns):
    """The order-two glide group with spin mixing and an antiunitary row."""
    import numpy as np
    from types import SimpleNamespace
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import centroid_source_map_and_wrap, spinor_rotation_for_sym_row

    fft_grid = (4, 4, 4)
    kgrid = (2, 2, 1)
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), swap])
    tnp = np.asarray([[0.0, 0.0, 0.0], [np.pi, np.pi, 0.0]])
    kints = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    kfrac = kints / np.asarray(kgrid, dtype=np.float64)
    irr = np.asarray([0, 1, 1, 2], dtype=np.int32)
    sym_rows = np.asarray([0, 0, 1, 2], dtype=np.int32)      # k3: time reversal
    parent_k = kfrac[[0, 1, 3]]
    theta = 0.7
    U1 = np.asarray([[np.cos(theta), -1j * np.sin(theta)],
                     [-1j * np.sin(theta), np.cos(theta)]])
    U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1])

    def spinor_action(rows, *, nspinor):
        return spinor_rotation_for_sym_row(U_spatial, np.asarray(rows), 2,
                                           nspinor=nspinor, R_cart=ops)

    sym = SimpleNamespace(sym_matrices=ops, translations=tnp, irr_idx_k=irr,
                          sym_idx_k=sym_rows, spinor_action=spinor_action,
                          unfolded_kpts=kfrac, kirr_fullids=np.asarray([0, 1, 3]))
    grid = _grid_points(fft_grid)
    perm_g, _ = centroid_source_map_and_wrap(grid, ops, tnp, fft_grid, extend_trs=True)
    cent = []
    for seed in (0, 5, 22, 27, 41, 50, 60, 63, 9, 14):
        orbit = sorted({int(perm_g[s, seed]) for s in range(4)})
        if len(cent) + len(orbit) <= 8 and not any(c in cent for c in orbit):
            cent.extend(orbit)
        if len(cent) == 8:
            break
    cent_flat = np.asarray(sorted(cent))
    plan = build_centroid_k_unfold_plan(sym, grid[cent_flat], fft_grid, mesh,
                                        nspinor=ns, parent_k_frac=parent_k)
    nb = 8
    psi_parent = _crand(rng, 3, nb, ns, 64)
    return dict(plan=plan, fft_grid=fft_grid, kgrid=kgrid, cent_flat=cent_flat,
                psi_parent=psi_parent, kfull=kfrac, ops=ops, tnp=tnp,
                band_chunks=((0, nb),), left=(0, 5), right=(2, 8), b_target=4,
                r_s_target=6, tile_width=16)


def _children(fx):
    """Full-BZ children by the typed action (the (★) ψ unfold) on the grid."""
    import numpy as np
    from symmetry_maps import centroid_source_map_and_wrap

    plan, fg = fx["plan"], fx["fft_grid"]
    perm_g, L_g = centroid_source_map_and_wrap(
        _grid_points(fg), fx["ops"], fx["tnp"], fg, extend_trs=True)
    psi = fx["psi_parent"]
    out = np.empty((plan.n_full,) + psi.shape[1:], dtype=np.complex128)
    for k in range(plan.n_full):
        p, s = int(plan.irr_idx[k]), int(plan.sym_idx[k])
        val = psi[p][:, :, perm_g[s]] * np.exp(
            2j * np.pi * (L_g[s].astype(np.float64) @ plan.k_parent_frac[p]))[None, None, :]
        if s >= plan.n_sym_spatial:
            val = np.conj(val)
        out[k] = np.einsum("ac,ncr->nar", plan.spin_action_full[k], val)
    return out


def _incumbent_rchunk(fx, mesh, w_l, w_r, vertex):
    """Z_q(μ, r) of the production r-chunk kernel over plan.real_grid_tiles."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from isdf.core import build_psi_r_cache_sm, z_q_from_psi_sm

    plan, fg, kgrid = fx["plan"], fx["fft_grid"], fx["kgrid"]
    psi = fx["psi_parent"]
    n_parent, nb, ns, n_rtot = psi.shape
    PX = PY = 2
    x_frac = _grid_points(fg) / np.asarray(fg, dtype=np.float64)
    kv = np.asarray(plan.k_parent_frac)
    u = psi * np.exp(-2j * np.pi * (x_frac @ kv.T).T)[:, None, None, :]
    psi_G = np.fft.fftn(u.reshape(*u.shape[:3], *fg), axes=(-3, -2, -1),
                        norm="ortho").reshape(u.shape)
    bcr = fx["band_chunks"]

    class _Store:
        band_chunk_ranges = bcr
        meta = SimpleNamespace(fft_grid=fg, nk_tot=n_parent, nspinor=ns)
        _bpd = max(hi - lo for lo, hi in bcr) // (PX * PY)
        local_band_chunk_shape = (n_parent, _bpd, ns, n_rtot)
        band_chunk_carrier = _bpd * PX * PY
        g_index = _put(np.broadcast_to(
            np.arange(n_rtot, dtype=np.int32).reshape(fg), (n_parent,) + tuple(fg)),
            NamedSharding(mesh, P(None, None, None, None)))
        kvecs_frac = _put(kv, NamedSharding(mesh, P(None, None)))

        def read_local_band_chunk(self, x_idx, y_idx, bc_idx):
            r = int(x_idx) * PY + int(y_idx)
            lo, hi = bcr[int(bc_idx)]
            bpd = (hi - lo) // (PX * PY)
            out = np.zeros(self.local_band_chunk_shape, dtype=np.complex128)
            out[:, :bpd] = psi_G[:, lo + r * bpd:lo + (r + 1) * bpd]
            return out

        _slice_local_tile_bc = read_local_band_chunk

    store = _Store()
    cache = jax.block_until_ready(build_psi_r_cache_sm(store, mesh_xy=mesh))
    face = plan.layout.axis.pack_host(
        psi[:, :, :, fx["cent_flat"]].transpose(0, 2, 3, 1), axis=2)
    psi_mun = _put(face, NamedSharding(mesh, P(None, None, "x", "y")))
    rep = NamedSharding(mesh, P())
    tiles = plan.real_grid_tiles(target_width=fx["tile_width"])
    Z = np.zeros((plan.n_full, plan.n_centroid_logical, n_rtot), dtype=np.complex128)
    for t in range(tiles.n_tiles):
        perm, wraps = tiles.source_tables(t)
        r_index = tiles.r_index[t]
        z = z_q_from_psi_sm(
            psi_G_store=store, psi_r_cache=cache, band_chunk_ranges=bcr,
            kgrid=kgrid, mesh_xy=mesh, psi_mun=psi_mun,
            weight_l=_put(w_l, rep), weight_r=_put(w_r, rep),
            k_unfold_plan=plan, gamma_L=vertex, gamma_R=vertex,
            tile_r_index=_put(r_index.astype(np.int32), rep),
            tile_local_perm=_put(perm, rep), tile_wraps=_put(wraps, rep))
        z = plan.layout.axis.unpack_host(_host(z), axis=1)
        act = r_index >= 0
        Z[:, :, r_index[act]] = z[:, :, act]
    return Z


def _batch_layout(fx, mesh, mb, rb, w_l, w_r, vertex, *, left_perm=None,
                  right_perm=None, want_projector=False):
    """The μ-batch layout on parent k: returns Z (nk, μ_canonical, N_r) and,
    with ``want_projector``, the unfolded D (nk, ns, μ, ns, N_r)."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from common.gamma_matrices import gamma_perm_phase
    from isdf.core import _conv_kpair_static_gamma, parent_projector_kconv
    from ffi.fft import make_fused_conv_kparent
    from symmetry_maps import open_spin_block_coefficient, unfold_operator_local

    plan, kgrid = fx["plan"], fx["kgrid"]
    psi = fx["psi_parent"]
    n_parent, nb, ns, n_rtot = psi.shape
    nk = plan.n_full
    XY = ("x", "y")
    left_perm = mb.left_perm if left_perm is None else left_perm
    right_perm = rb.local_perm if right_perm is None else right_perm
    canon = np.asarray(plan.layout.axis.packed_to_canonical)
    pts = rb.points
    Y = psi[..., np.clip(pts, 0, None)] * (pts >= 0)          # (kp, n, s, P, S, r)
    Y = np.ascontiguousarray(Y.transpose(3, 4, 0, 1, 2, 5))
    shard = NamedSharding(mesh, P(XY))
    rep = NamedSharding(mesh, P())
    Y_d = _put(Y, shard)
    rp_d = _put(np.asarray(right_perm, dtype=np.int32), shard)
    rL_d = _put(np.asarray(rb.wraps, dtype=np.float64), shard)
    wl = jnp.asarray(w_l)
    wr = jnp.asarray(w_r)
    # The pair kernel exactly as _z_q_face_parent builds it: the native
    # conv_kparent arm where the gate resolves it (CUDA auto), else None.
    p_l, ph_l = _conv_kpair_static_gamma(None, ns)
    pair_kernel = make_fused_conv_kparent(
        mesh, kgrid, ns, (mb.b, rb.r_s), perm_l=p_l, phase_l=ph_l,
        perm_r=p_l, phase_r=ph_l)
    vtx = ((jnp.arange(ns), jnp.ones(ns)) if vertex == 0
           else tuple(jnp.asarray(v) for v in gamma_perm_phase(vertex)))
    coef = [open_spin_block_coefficient(plan.spin_action_full, a, b)
            for a in range(ns) for b in range(ns)]

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(P(), P(XY), P(), P(), P(XY), P(XY)),
             out_specs=(P(XY), P(XY)), check_vma=False)
    def run(X, Yl, lp, lL, rp, rL):
        Yl, rp, rL = Yl[0], rp[0], rL[0]

        def body(carry, s):
            yc = jnp.conj(Yl[s])                                # (kp, n, s, r)
            D_l = jnp.einsum("kamn,knbr->kambr", X * wl, yc)
            D_r = jnp.einsum("kamn,knbr->kambr", X * wr, yc)
            Z = parent_projector_kconv(
                D_l, D_r, plan=plan, left_perm=lp, left_L=lL,
                right_perm=rp[s], right_L=rL[s], kgrid=kgrid,
                vertex_l=vtx, vertex_r=vtx, pair_kernel=pair_kernel)
            if not want_projector:
                return carry, (Z, jnp.zeros((1,), Z.dtype))
            blocks = []
            for i in range(ns * ns):
                acc = 0
                for c in range(ns):
                    for d in range(ns):
                        acc = acc + coef[i][:, c, d][:, None, None] * unfold_operator_local(
                            D_l[:, c, :, d, :], irr_idx=plan.irr_idx, sym_idx=plan.sym_idx,
                            q_irr_frac=plan.k_parent_frac, left_local_perm=lp,
                            left_L_table=lL, right_local_perm=rp[s], right_L_table=rL[s],
                            n_sym_spatial=plan.n_sym_spatial,
                            left_mesh_axis=None, right_mesh_axis=None)
                blocks.append(acc)
            return carry, (Z, jnp.stack(blocks, axis=1))

        _, (Zs, Ds) = jax.lax.scan(body, 0, jnp.arange(Yl.shape[0]), unroll=1)
        return Zs[None], Ds[None]

    Z = np.zeros((nk, plan.n_centroid_logical, n_rtot), dtype=np.complex128)
    D = (np.zeros((nk, ns, plan.n_centroid_logical, ns, n_rtot), dtype=np.complex128)
         if want_projector else None)
    act_r = pts >= 0
    for beta in range(mb.n_batch):
        slots = mb.mu[beta]
        c_idx = np.where(slots >= 0, canon[np.clip(slots, 0, None)], -1)
        flat = fx["cent_flat"][np.clip(c_idx, 0, None)]
        X = psi[..., flat] * (c_idx >= 0)                       # (kp, n, s, b)
        X = np.ascontiguousarray(X.transpose(0, 2, 3, 1))
        Zb, Db = run(_put(X, rep), Y_d, _put(np.asarray(left_perm[beta], np.int32), rep),
                     _put(np.asarray(mb.left_L[beta], np.float64), rep), rp_d, rL_d)
        Zb = _host(Zb)                                           # (P, S, nk, b, r)
        if want_projector:
            Db = _host(Db)
        live = np.flatnonzero(c_idx >= 0)
        for p in range(rb.n_ranks):
            for s in range(rb.n_sub):
                r_live = np.flatnonzero(act_r[p, s])
                sub = Zb[p, s][:, live][:, :, r_live]
                Z[:, c_idx[live][:, None], pts[p, s, r_live][None, :]] = sub
                if want_projector:
                    Dps = Db[p, s].reshape(nk, ns, ns, -1, rb.r_s)
                    for a in range(ns):
                        for b in range(ns):
                            D[:, a, c_idx[live][:, None], b, pts[p, s, r_live][None, :]] = \
                                Dps[:, a, b][:, live][:, :, r_live]
    return Z, D, ("native" if pair_kernel is not None else "xla")


def _zeta_and_vq(fx, Z, C):
    """ζ_q = C_q⁺ Z_q (the producer's dense charge solve) and V_q on the box."""
    import numpy as np
    import jax.numpy as jnp
    from isdf.core import solve_zeta_charge_dense

    fg = np.asarray(fx["fft_grid"])
    kq = np.asarray([[i, j, k] for i in range(fx["kgrid"][0]) for j in range(fx["kgrid"][1])
                     for k in range(fx["kgrid"][2])]) / np.asarray(fx["kgrid"])
    x = _grid_points(fg) / fg
    G = np.stack(np.meshgrid(*(np.fft.fftfreq(n, 1.0 / n) for n in fg),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    zeta, vq = [], []
    for q in range(Z.shape[0]):
        z = np.asarray(solve_zeta_charge_dense(
            jnp.asarray(C[q]), jnp.asarray(Z[q]), charge_zeta_solve="rank_truncate",
            zeta_rcond=1e-10, rank_log=False))
        zg = np.fft.fftn((z * np.exp(-2j * np.pi * x @ kq[q]))
                         .reshape(-1, *fg), axes=(1, 2, 3)).reshape(z.shape[0], -1)
        v = 1.0 / (np.sum((G + kq[q]) ** 2, axis=1) + 0.5)
        zeta.append(z)
        vq.append(np.einsum("mg,g,ng->mn", zg.conj(), v, zg))
    return np.asarray(zeta), np.asarray(vq)


def _rel(a, b):
    import numpy as np
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _worker(case: str) -> int:
    import numpy as np
    import jax
    from jax.sharding import Mesh

    devs = jax.devices()
    if len(devs) < 4:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0
    mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(2026_09_23)
    if case == "acubic_ns1":
        fx, vertex = _acubic_fixture(mesh, rng), 0
    else:
        ns = 2 if case == "glide_ns2" else 4
        fx, vertex = _glide_fixture(mesh, rng, ns), (1 if case.endswith("_v1") else 0)
    print(json.dumps(parity_receipt(case, fx, mesh, vertex)))
    return 0


def parity_receipt(case: str, fx: dict, mesh, vertex: int) -> dict:
    """All levels and red twins for one fixture dict (see the ``_*_fixture``
    builders); a sandbox harness may pass its own fixture."""
    import numpy as np
    sys.path.insert(0, str(_HERE))
    from test_isdf_zq_parent_parity import _dense_pair_rhs
    from gw.centroid_k_unfold import orbit_mu_batches, orbit_r_blocks

    plan = fx["plan"]
    nb = fx["psi_parent"].shape[1]
    idx = np.arange(nb)
    w_l = np.where((idx >= fx["left"][0]) & (idx < fx["left"][1]), 1.0, 0.0)
    w_r = np.where((idx >= fx["right"][0]) & (idx < fx["right"][1]), 1.0, 0.0)
    fg = fx["fft_grid"]
    kin = np.rint(fx["kfull"] * np.asarray(fx["kgrid"])).astype(int) % np.asarray(fx["kgrid"])
    assert np.array_equal(np.ravel_multi_index(kin.T, fx["kgrid"]), np.arange(plan.n_full)), \
        "full-k rows must be the C-order grid"

    mb = orbit_mu_batches(plan, plan.n_centroid_packed, 4, b_target=fx["b_target"])
    rb = orbit_r_blocks(plan, fg, 4, r_s_target=fx["r_s_target"], route="cache")
    children = _children(fx)
    cent = fx["cent_flat"]
    receipt = dict(
        case=case, n_parent=plan.n_parent, n_full=plan.n_full,
        n_sym=int(plan.n_sym_spatial), rows=np.unique(plan.sym_idx).tolist(),
        antiunitary_rows=int(np.sum(plan.sym_idx >= plan.n_sym_spatial)),
        n_batch=mb.n_batch, b=mb.b, n_sub=rb.n_sub, r_s=rb.r_s,
        naive_child_parent_gap=float(
            np.max(np.abs(children - fx["psi_parent"][plan.irr_idx]))
            / np.max(np.abs(children))))

    # L2: unfolded projector vs D from the full-BZ children.
    Z_new, D_new, arm = _batch_layout(fx, mesh, mb, rb, w_l, w_r, vertex, want_projector=True)
    receipt["tail_arm"] = arm
    D_ref = np.einsum("knam,knbr,n->kambr", children[:, :, :, cent], children.conj(), w_l)
    receipt["L2_projector_rel"] = _rel(D_new, D_ref)

    # L3: Z vs the incumbent r-chunk kernel and the dense sums.
    Z_inc = _incumbent_rchunk(fx, mesh, w_l, w_r, vertex)
    Z_ref = _dense_pair_rhs(children, cent, fx["kgrid"], w_l, w_r, vertex)
    receipt["L3_Z_vs_rchunk_rel"] = _rel(Z_new, Z_inc)
    receipt["L3_Z_vs_dense_rel"] = _rel(Z_new, Z_ref)
    receipt["L3_rchunk_vs_dense_rel"] = _rel(Z_inc, Z_ref)

    # L4: ζ and V_q through one C_q (the incumbent's, Z at the centroids).
    if vertex == 0:
        C = Z_inc[:, :, cent]
        zeta_new, vq_new = _zeta_and_vq(fx, Z_new, C)
        zeta_inc, vq_inc = _zeta_and_vq(fx, Z_inc, C)
        receipt["L4_zeta_rel"] = _rel(zeta_new, zeta_inc)
        receipt["L4_vq_rel"] = _rel(vq_new, vq_inc)

    # Red twins.  (a) permuted batch table on a non-identity row in use.
    row = next(int(r) for r in np.unique(plan.sym_idx)
               if not np.array_equal(mb.left_perm[0, r], np.arange(mb.b)))
    bad = mb.left_perm.copy()
    live = np.flatnonzero(mb.mu[0] >= 0)
    bad[0, row, live[[0, 1]]] = bad[0, row, live[[1, 0]]]
    Z_bad, _, _ = _batch_layout(fx, mesh, mb, rb, w_l, w_r, vertex, left_perm=bad)
    receipt["red_left_perm_rel"] = _rel(Z_bad, Z_inc)
    # (b) permuted block table.
    p_s = next((p, s) for p in range(rb.n_ranks) for s in range(rb.n_sub)
               if not np.array_equal(rb.local_perm[p, s, row], np.arange(rb.r_s)))
    bad_r = rb.local_perm.copy()
    moved = np.flatnonzero(bad_r[p_s][row] != np.arange(rb.r_s))[:2]
    bad_r[p_s][row, moved] = bad_r[p_s][row, moved[::-1]]
    Z_bad, _, _ = _batch_layout(fx, mesh, mb, rb, w_l, w_r, vertex, right_perm=bad_r)
    receipt["red_right_perm_rel"] = _rel(Z_bad, Z_inc)
    # (c) a split orbit forced through: sources outside the batch fall back
    # to the slot itself (what a clipped gather would do).
    if mb.n_batch >= 2:
        mu = mb.mu.copy()
        a, b = int(np.flatnonzero(mu[0] >= 0)[0]), int(np.flatnonzero(mu[1] >= 0)[0])
        mu[0, a], mu[1, b] = mu[1, b], mu[0, a]
        perm = np.asarray(plan.sym_perm)
        forced = np.tile(np.arange(mb.b, dtype=np.int32), (mb.n_batch, perm.shape[0], 1))
        L = np.zeros(mb.left_L.shape, dtype=np.int32)
        for beta in range(mb.n_batch):
            slot_of = {int(m): j for j, m in enumerate(mu[beta]) if m >= 0}
            for r in np.unique(plan.sym_idx):
                for j, m in enumerate(mu[beta]):
                    if m >= 0:
                        forced[beta, r, j] = slot_of.get(int(perm[r, m]), j)
                        L[beta, r, j] = plan.L_table[r, m]
        split = mb._replace(mu=mu, left_L=L)
        Z_bad, _, _ = _batch_layout(fx, mesh, split, rb, w_l, w_r, vertex, left_perm=forced)
        receipt["red_split_orbit_rel"] = _rel(Z_bad, Z_inc)
    return receipt


def _run_worker(case: str, timeout: int = 1200) -> dict:
    env = dict(os.environ)
    env.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", LORRAX_CONV_KPAIR_FFI="off")
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_force_host_platform_device_count=4").strip()
    src = str(_HERE.parent / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run([sys.executable, __file__, "worker", case], env=env,
                         capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, f"{case} rc={res.returncode}\n{res.stdout}\n{res.stderr}"
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"no receipt\n{res.stdout}\n{res.stderr}"
    return json.loads(lines[-1])


@pytest.mark.parametrize("case", _CASES)
def test_mubatch_parent_k_layout_matches_the_rchunk_fit(case):
    out = _run_worker(case)
    if "skip" in out:
        pytest.skip(out["skip"])
    # Route receipt: raw parents fewer than full k, the typed action is not
    # the identity, and several batches/sub-blocks run.
    assert out["n_parent"] < out["n_full"] and len(out["rows"]) > 1, out
    assert out["naive_child_parent_gap"] > 0.1, out
    assert out["n_batch"] >= 2 and out["n_sub"] >= 2, out
    if case.startswith("glide"):
        assert out["antiunitary_rows"] > 0, out
    for key in ("L2_projector_rel", "L3_Z_vs_rchunk_rel", "L3_Z_vs_dense_rel",
                "L4_zeta_rel", "L4_vq_rel"):
        if key in out:
            assert out[key] < _TOL, (key, out)
    for key in ("red_left_perm_rel", "red_right_perm_rel", "red_split_orbit_rel"):
        assert out[key] > _RED, (key, out)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        sys.exit(_worker(sys.argv[2]))
    raise SystemExit("usage: python test_zeta_mubatch_sym_parity.py worker <case>")
