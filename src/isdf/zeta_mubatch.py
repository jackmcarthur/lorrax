"""μ-batch ζ fit, route G: Z_q(G) by centroid batches; see docs/architecture/zeta_fit_mubatch.md.

Per μ batch ``B``, with conj ψ(G) of the full zone sharded over G slots:

    X_B = ψ_{nks}(r_μ)                        one psum of the ranks' partial DFTs
    D̃^X_k(a, μ, b, G) = Σ_n w^X_n X_{nka}(r_μ) conj c_{nkb}(G)   (G-space GEMM, local)
    one all-to-all: G split → μ owner (rank p owns batch slots p·c + [0, c))
    on the owner, per plane group: cylinder → planes → D(k, μ, r_plane);
        Z_q(μ, r) = ffi.fft.make_fused_conv_kplane(D, Bloch phase), the pair
        convolution on the identity plan read from the plane FFT output;
        LR+RL completion; e^{-iq·r}, 2D FFT, ζ-sphere columns, axis phase
        accumulate Z_q(μ, G)

``Z_q(G)`` is written once into :class:`ZStore` (pinned host tiles or a
slab_io file); no accumulator over r and no distributed FFT exist.  ``C_q⁺``
acts on μ only, so it is applied afterwards on the sphere, G tile by G tile:
``ζ_q(G) = C_q⁺ Z_q(G)`` (:class:`ZetaG`, through :mod:`isdf.cplus`).
"""
from __future__ import annotations

import math
import time
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from common.fft_helpers import local_fftn3
from common.wfn_transforms import _plane_geometry
from runtime.padding import axis_mask, pad_to_axis, padded_axis, strip_axis

_XY = ('x', 'y')
_kernel_cache: dict = {}


def _mesh_size(mesh: Mesh) -> int:
    return int(mesh.shape['x']) * int(mesh.shape['y'])


def _mesh_id(mesh: Mesh) -> tuple:
    return (tuple(mesh.axis_names), tuple(int(v) for v in mesh.devices.shape),
            tuple(int(d.id) for d in mesh.devices.flat))


# ---------------------------------------------------------------------------
# Route G: the pair GEMM in G space, one all-to-all to the μ owners, planes
# on the owner (docs/architecture/zeta_fit_mubatch.md, "Route G")
# ---------------------------------------------------------------------------

class OwnerOrbitBatches(NamedTuple):
    """μ batches whose every owner (``c`` slots per rank) holds whole orbits."""
    mu: np.ndarray            # (n_batch, P·c) packed centroid per slot, −1 pad
    left_perm: np.ndarray     # (n_batch, P, n_rows, c) owner-local source slot
    left_L: np.ndarray        # (n_batch, P, n_rows, c, 3) lattice wrap
    c: int
    n_batch: int
    slot_of_packed: np.ndarray  # (μ_pad,) store slot β·P·c + p·c + j, −1 pad

    @property
    def b(self) -> int:
        return int(self.mu.shape[1])


def owner_orbit_batches(plan, mu_pad: int, n_ranks: int, *, c_target: int):
    """Whole-orbit bins of about ``c_target`` (one per owner), ``P`` bins a batch.

    The owner unfolds the left endpoint of its pair projectors inside its own
    bin, so a bin must hold whole orbits
    (:func:`gw.centroid_k_unfold.orbit_mu_batches` with one rank).  Empty
    bins pad the last batch.
    """
    from gw.centroid_k_unfold import orbit_mu_batches
    P_ = int(n_ranks)
    bins = orbit_mu_batches(plan, int(mu_pad), 1, b_target=max(1, int(c_target)))
    c = int(bins.b)
    n_bins = int(bins.n_batch)
    n_batch = -(-n_bins // P_)
    pad = n_batch * P_ - n_bins
    n_rows = int(bins.left_perm.shape[1])
    mu = np.concatenate([bins.mu, np.full((pad, c), -1, bins.mu.dtype)])
    lp = np.concatenate([bins.left_perm, np.broadcast_to(
        np.arange(c, dtype=bins.left_perm.dtype), (pad, n_rows, c))])
    lL = np.concatenate([bins.left_L, np.zeros((pad, n_rows, c, 3), bins.left_L.dtype)])
    return OwnerOrbitBatches(
        mu=mu.reshape(n_batch, P_ * c),
        left_perm=lp.reshape(n_batch, P_, n_rows, c).astype(np.int32),
        left_L=lL.reshape(n_batch, P_, n_rows, c, 3).astype(np.int32),
        c=c, n_batch=n_batch,
        slot_of_packed=np.asarray(bins.packed_to_slot(int(mu_pad)), dtype=np.int32))


def best_owner_orbit_batches(plan, mu_pad: int, n_ranks: int, *, c_max: int):
    """The whole-orbit batching with the least padded work, bins of at most ``c_max``.

    Bins hold whole orbits, so the planned ``c = b/P`` can pack badly (CrI3
    8x8 P16, p4r_cri3_p16: c = 19 packs 8 batches of 18-slot bins, c = 12
    the same 8 batches of 12).  Each candidate c ≤ ``c_max`` is packed and
    costed ``n_batch·(c + 1)``: a batch pays its per-centroid owner work plus
    about one centroid's worth of fixed cost (collectives, launches).
    ponytail: a linear scan over c and a fixed-cost guess of one centroid;
    move the choice into the planner (with the orbit sizes) if either binds.
    """
    best = None
    for c in range(max(1, int(c_max)), 0, -1):
        mb = owner_orbit_batches(plan, mu_pad, n_ranks, c_target=c)
        if best is not None and mb.c > c_max:
            continue                      # an orbit wider than c: no gain
        cost = mb.n_batch * (mb.c + 1)
        if best is None or cost < best[0]:
            best = (cost, mb)
    return best[1]


def typed_child_G_tables(plan, *, fft_grid, sphere_par, gvec_child,
                         ngk_child, k_child):
    """The r-space typed transport of :func:`typed_children_psi_G` as G-space
    tables, its exact Fourier image.

    The typed child is ``ψ_k(x) = U_k T[ψ_p(S x − t)]`` with the snapped
    offset ``t = round(N·S·τ)/N`` (the grid permutation's), so on the child's
    sphere ``c_k(G') = U_k T[c_p(G) e^{-2πi (k̄+G)·t}]`` with
    ``S^T(k̄+G) = ±(k + G')`` (+ unitary, − antiunitary rows, where T
    conjugates).  ``sphere_par (n_parent, ngk_par)`` is the parents' sphere
    index (slot → flat box cell, ``≥ N_r`` on a pad slot;
    :func:`common.gvec_fft_box.build_sphere_box_index`).  Returns ``(pslot
    (nk, ngk_c) int32`` parent slot of each child slot (``ngk_par`` for pad
    slots), ``phase (nk, ngk_c)`` ``e^{-2πi (k̄+G)·t}``, ``anti (nk,) bool)``.
    """
    fg = np.asarray(fft_grid, dtype=np.int64)
    S_all = np.asarray(plan.spatial_ops, dtype=np.int64)
    tau = np.asarray(plan.translations, dtype=np.float64) / (2.0 * np.pi)
    n_sym = int(plan.n_sym_spatial)
    kp = np.asarray(plan.k_parent_frac, dtype=np.float64)
    kc = np.asarray(k_child, dtype=np.float64)
    gvc = np.asarray(gvec_child, dtype=np.int64)
    nk, ngk_c = int(gvc.shape[0]), int(gvc.shape[1])
    sph = np.asarray(sphere_par, dtype=np.int64)
    n_par, ngk_par = (int(v) for v in sph.shape)
    N = int(np.prod(fg))
    box = np.full((n_par, N), ngk_par, dtype=np.int64)      # flat cell → slot
    for p_ in range(n_par):
        live_p = sph[p_] < N
        box[p_, sph[p_][live_p]] = np.flatnonzero(live_p)
    pslot = np.full((nk, ngk_c), ngk_par, dtype=np.int32)
    phase = np.zeros((nk, ngk_c), dtype=np.complex128)
    anti = np.zeros(nk, dtype=bool)
    for k in range(nk):
        p, s = int(plan.irr_idx[k]), int(plan.sym_idx[k])
        S = S_all[s % n_sym]
        anti[k] = s >= n_sym
        t = np.rint(fg * (S @ tau[s % n_sym])) / fg
        live = np.arange(ngk_c) < int(ngk_child[k])
        K = (kc[k][None, :] + gvc[k]) * (-1.0 if anti[k] else 1.0)   # = S^T (k̄+G)
        kg = np.linalg.solve(S.T.astype(np.float64), K.T).T            # k̄ + G
        G = np.rint(kg - kp[p][None, :]).astype(np.int64)
        if np.max(np.abs((kg - kp[p]) - G)[live], initial=0.0) > 1e-6:
            raise ValueError(f"typed_child_G_tables: child k={k} is not an image "
                             f"of parent {p} under row {s}")
        flat = (((G % fg) * np.array([fg[1] * fg[2], fg[2], 1])).sum(-1))
        sl = box[p][flat]
        if np.any(sl[live] >= int(ngk_par)):
            raise ValueError(f"typed_child_G_tables: child k={k} has a G outside "
                             f"parent {p}'s sphere")
        pslot[k] = np.where(live, sl, int(ngk_par))
        phase[k] = np.where(live, np.exp(-2j * np.pi * ((kp[p] + G) @ t)), 0.0)
    return pslot, phase, anti


def zeta_plane_tables(gvec_components, ngk_per_q, fft_grid, axis, g_axis):
    """The ζ sphere as a cylinder: its in-plane columns ``zc`` (union over the
    stored q), its axis values ``za`` (mod ``n_a``), and per ``(q, slot)`` the
    flat index ``col·n_za + a`` into ``(zc, za)``, at the store's G-tile
    carrier width (pad slots index 0; ``ngk`` masks them downstream)."""
    n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(fft_grid, axis)
    gv = np.asarray(gvec_components, dtype=np.int64)          # (Q, 3, ngk)
    live = np.arange(gv.shape[-1])[None, :] < np.asarray(ngk_per_q)[:, None]
    col = (gv[:, b_ax] % n_b) * n_c + gv[:, c_ax] % n_c
    ga = gv[:, axis] % n_a
    zc, zc_i = np.unique(np.where(live, col, col[:, :1]), return_inverse=True)
    za, za_i = np.unique(np.where(live, ga, ga[:, :1]), return_inverse=True)
    flat = np.where(live, zc_i.reshape(col.shape) * za.size + za_i.reshape(ga.shape), 0)
    return (zc.astype(np.int32), za.astype(np.int32),
            np.asarray(pad_to_axis(flat.astype(np.int32), g_axis, axis=1)))


def _spin_sandwich(U, d):
    """``(U ⊗ Ū) d`` on the two spin axes (0 and 3) of ``d (s, x, m, s', j)``.

    Written as ``ns²`` elementwise terms rather than an einsum: with a
    contraction length of ``ns`` (2 or 4) XLA lowers the einsum to eight tiny
    cuBLAS GEMMs per child k, whereas the elementwise form fuses into the
    surrounding gathers (docs/dev/QUALITY_PATTERNS.md §11).
    """
    ns = int(d.shape[0])
    Uc = jnp.conj(U)
    left = jnp.stack([sum(U[a, c] * d[c] for c in range(ns)) for a in range(ns)])
    return jnp.stack([sum(left[:, :, :, e] * Uc[b, e] for e in range(ns))
                      for b in range(ns)], axis=3)


def make_route_g_kernel(*, mesh: Mesh, kgrid, fft_grid, ns: int, b: int,
                        q_sel, q_axis, q_neg, qvec_frac, n_col: int, n_s: int,
                        n_pg: int, axis: int, n_src: int, vertices=(0,),
                        stop_at: str | None = None):
    """Compile-once executable for one μ batch on route G.

    Returns ``fn(psi_bar, w_l, w_r, kvecs, g3, xmu, live, cyl, zt, unf, lt) -> rows``,
    one ``(Q, b, N_G)`` array per vertex of ``vertices`` at ``P(None, ('x','y'),
    None)`` (μ-owned; rank p owns the batch slots ``p·c + [0, c)``, ``c = b/P``).
    ``vertices`` are the Lorentz channels μ_L of γ̃^{μ_L}: ``(0,)`` for the
    charge fit, ``(1, 2, 3)`` for the three current channels, which share
    every stage but the k-convolution and the accumulate:

    1. ``X_B = ψ(r_μ)`` by a direct DFT of each rank's ψ G slice, one psum;
    2. the pair GEMM in G space on the rank's slice
       (:func:`isdf.pair_kernels.pair_projectors_lr`, stored ``conj c``);
    3. ONE all-to-all, G split → μ owner (``[L | R]`` rows owner-major);
    4. on the owner, per group of ``n_pg`` planes normal to ``axis``: the
       sphere → cylinder gather, the axis DFT onto the planes, the 2D FFT
       and Bloch phase → ``D(k, μ, r_plane)``; the k-convolution
       (``ffi.fft.make_fused_conv_kplane``: the pair convolution on the
       identity plan, reading the 2D FFT output in place and applying the
       Bloch phase and the L/R split on load; the vertex γ̃^{μ_L} on both
       endpoints' output spins, ``(perm, phase)`` attributes of the load);
       LR+RL completion; ``e^{-iq·r}``, forward 2D FFT, the ζ-sphere
       columns and the axis phase accumulate ``Z^{μ_L}_q(μ, G)``.

    Operands: ``psi_bar (n_src, nb, ns, ngk_pad)`` = conj ψ(G) of the raw
    parents, G sharded ``P(None, None, None, ('x','y'))``; ``g3 (n_src,
    ngk_pad, 3)`` Miller index of each parent slot, same G sharding;
    ``kvecs (n_src, 3)``; ``xmu (b, 3)`` fractional coordinates of the batch
    slots and ``live (b,)`` their mask; ``cyl`` =
    :func:`common.wfn_transforms.psi_cylinder_tables` of the CHILDREN's
    spheres; ``zt`` = :func:`zeta_plane_tables`.  ``unf = (irr (nk,), sym
    (nk,), anti (nk,), U (nk, ns, ns), pslot (nk, ngk_c), phase (nk,
    ngk_c), k_child (nk, 3))`` is the typed transport in G space
    (:func:`typed_child_G_tables`) and ``lt = (left_perm (P, n_rows, c),
    left_L (P, n_rows, c, 3))`` each owner's whole-orbit centroid tables,
    sharded over the ranks: the owner unfolds the parents' pair projectors
    ``D̃_k = (U⊗Ū) T[D̃_k̄(perm μ, pslot G) e^{2πi L·k̄} conj(phase)]``,
    the exact Fourier image of the r-space typed transport C_q and the
    faces use.

    ``stop_at`` (debug split timers only: ``'x'``, ``'gemm'``, ``'a2a'``,
    ``'planes'``, ``'kconv'``) truncates after that stage with a checksum.
    """
    from isdf.core import _conv_kpair_static_gamma
    from isdf.pair_kernels import pair_projectors_lr
    from ffi.fft import make_fused_conv_kplane
    from common.gamma_matrices import gamma_perm_phase
    P_ = _mesh_size(mesh)
    if b % P_:
        raise ValueError(f"make_route_g_kernel: batch {b} must be a multiple of P={P_}")
    c = b // P_
    nk = int(np.prod(kgrid))
    N = int(np.prod(fft_grid))
    n_a, (n_b, n_c), (b_ax, c_ax) = _plane_geometry(fft_grid, axis)
    ps = n_b * n_c
    pl_ax = padded_axis(n_a, int(n_pg), name="route-G plane groups")
    n_grp = pl_ax.carrier // int(n_pg)
    q_sel = np.asarray(q_sel, dtype=np.int32)
    Q = int(q_axis.logical)
    # None: equal L/R windows, the pair equations are already symmetric.
    q_neg = None if q_neg is None else np.asarray(q_neg, dtype=np.int32)
    qv = np.asarray(qvec_frac, dtype=np.float64)
    vertices = tuple(int(v) for v in vertices)
    if not vertices or any(v not in (0, 1, 2, 3) for v in vertices):
        raise ValueError(f"make_route_g_kernel: vertices {vertices} must be μ_L in 0..3")
    # One k-convolution per channel: C_q and Z_q put γ̃^{μ_L} on both
    # endpoints' output spins after the typed unfold (isdf.core.c_q_from_psi_sm),
    # which the router takes as the static (perm, phase) of its load.
    pair_kernels = []
    for v in vertices:
        p_v, ph_v = _conv_kpair_static_gamma(None if v == 0 else gamma_perm_phase(v), ns)
        pair_kernels.append(make_fused_conv_kplane(mesh, kgrid, ns, perm_l=p_v, phase_l=ph_v,
                                                   perm_r=p_v, phase_r=ph_v))
    n_v = len(vertices)
    ib = (np.arange(ps) // n_c).astype(np.float64)
    ic = (np.arange(ps) % n_c).astype(np.float64)
    key = ('route_g', _mesh_id(mesh), tuple(kgrid), tuple(fft_grid), ns, b,
           hash(q_sel.tobytes()), q_axis,
           None if q_neg is None else hash(q_neg.tobytes()), hash(qv.tobytes()),
           int(n_col), int(n_s), int(n_pg), int(axis), int(n_src), vertices, stop_at)
    hit = _kernel_cache.get(key)
    if hit is not None:
        return hit

    G_ = P(None, None, None, _XY)
    R_ = P(_XY)

    @partial(shard_map, mesh=mesh,
             in_specs=(G_, P(), P(), P(), P(None, _XY, None), P(), P(), P(), P(),
                       P(), (R_, R_)),
             out_specs=(P(None, _XY, None),) * n_v, check_vma=False)
    def _local(psi_bar, w_l, w_r, kvecs, g3, xmu, live, cyl, zt, unf, lt):
        ci, cax, pfc = cyl
        zc, za, zflat = zt
        irr, sym, anti, U, pslot, phase, kch = unf
        lperm, lL = lt[0][0], lt[1][0]                    # this owner's orbits
        # 1. X_B = ψ_{nks}(r_μ) = Σ_G c e^{2πi(k+G)·x_μ}/√N, one psum
        kg = kvecs[:, None, :] + g3.astype(jnp.float64)            # (k, Gp, 3)
        ph = jnp.exp(2j * jnp.pi * jnp.einsum('kgd,md->kgm', kg, xmu))
        X = jnp.einsum('knsg,kgm->knsm', jnp.conj(psi_bar), ph) / np.sqrt(N)
        X = jax.lax.psum(X, _XY) * live[None, None, None, :]
        n_g = int(zflat.shape[-1])
        chk = lambda a: (jnp.zeros((Q, c, n_g), jnp.complex128) + jnp.sum(jnp.abs(a)),) * n_v
        if stop_at == 'x':
            return chk(X)
        # 2. pair GEMM in G space on this rank's slice (one band chunk)
        D_l, D_r = pair_projectors_lr(X[None], lambda bc: psi_bar,
                                      w_l[None], w_r[None])     # (k, s, b, s, Gp)
        if stop_at == 'gemm':
            return chk(jnp.sum(jnp.abs(D_l)) + jnp.sum(jnp.abs(D_r)))
        # 3. one all-to-all: G split -> μ owners, [L | R] owner-major
        D = jnp.stack([D_l, D_r], axis=2).reshape(n_src, ns, 2, P_, c, ns, -1)
        D = jnp.moveaxis(D, 3, 2).reshape(n_src, ns, P_ * 2 * c, ns, -1)
        D = jax.lax.all_to_all(D, _XY, split_axis=2, concat_axis=4, tiled=True)
        if stop_at == 'a2a':
            return chk(D)
        D = jnp.concatenate([D, jnp.zeros(D.shape[:4] + (1,), D.dtype)], axis=-1)
        ngk1 = int(D.shape[-1])

        # The axis DFT of every k onto all planes, once per batch: the D
        # cylinder (k, plane, s, μ, s, column), plane axis padded to whole groups.
        pa_all = jnp.exp(-2j * jnp.pi * cax.astype(jnp.float64)[:, None]
                         * jnp.arange(pl_ax.carrier)[None, :] / n_a)
        pa_all = pa_all * (jnp.arange(pl_ax.carrier) < n_a)[None, :]

        def cyl_k(_, k):
            # the typed unfold of the parent's pair projectors to child k
            p, s_ = irr[k], sym[k]
            d = D[p].reshape(ns, 2, c, ns, ngk1)
            wl = jnp.exp(2j * jnp.pi * (lL[s_].astype(jnp.float64) @ kvecs[p]))
            d = jnp.take(d, lperm[s_], axis=2) * wl[None, None, :, None, None]
            d = jnp.take(d, pslot[k], axis=-1) * jnp.conj(phase[k])
            d = jnp.where(anti[k], jnp.conj(d), d)
            d = _spin_sandwich(U[k], d)
            d = jnp.concatenate([d.reshape(ns, 2 * c, ns, -1),
                                 jnp.zeros((ns, 2 * c, ns, 1), d.dtype)], -1)
            cy = jnp.take(d, jnp.clip(ci[k], 0, int(d.shape[-1]) - 1).reshape(-1), axis=-1)
            cy = cy.reshape(ns, 2 * c, ns, n_col, n_s)
            return None, jnp.einsum('asbcj,jp->pasbc', cy, pa_all)

        _, Fa = jax.lax.scan(cyl_k, None, jnp.arange(nk, dtype=jnp.int32), unroll=1)
        del D
        if stop_at == 'planes':
            return chk(Fa)
        n_zc, n_za = int(zc.shape[0]), int(za.shape[0])

        def group(acc, gi):
            a0 = gi * n_pg + jnp.arange(n_pg)
            on = (a0 < n_a).astype(jnp.float64)
            F = jax.lax.dynamic_slice_in_dim(Fa, gi * n_pg, n_pg, axis=1)
            # Empty plane cells carry the out-of-range column n_col: a zero
            # fill in the gather itself, no padded copy of the cylinder.
            st = jnp.take(F, pfc, axis=-1, mode='fill', fill_value=0).reshape(
                nk, n_pg, ns, 2 * c, ns, n_b, n_c)
            d = local_fftn3(st, axes=(-2, -1), norm='backward')        # Σ e^{-iG·r}
            bl = jnp.exp(-2j * jnp.pi * (
                kch[:, axis][:, None, None] * a0[None, :, None] / n_a
                + kch[:, b_ax][:, None, None] * ib[None, None, :] / n_b
                + kch[:, c_ax][:, None, None] * ic[None, None, :] / n_c)) / np.sqrt(N)
            qin = jnp.exp(-2j * jnp.pi * (jnp.asarray(qv[:, b_ax])[:, None] * ib[None, :] / n_b
                                          + jnp.asarray(qv[:, c_ax])[:, None] * ic[None, :] / n_c))
            # The axis transform onto the ζ cylinder: one matmul over the
            # group's planes, e^{-2πi (q_a + G_a) a/n_a}.
            E = jnp.exp(-2j * jnp.pi * (jnp.asarray(qv[:, axis])[:, None, None]
                                        + za.astype(jnp.float64)[None, None, :])
                        * a0[None, :, None] / n_a) * on[None, :, None]   # (Q, n_pg, n_za)
            d = d.reshape(nk, n_pg, ns, 2 * c, ns, ps)
            out = []
            for pair_kernel, acc_v in zip(pair_kernels, acc):
                # The k-convolution reads d where the FFT left it: the Bloch
                # phase, the L | R split of the 2c slots and the vertex happen
                # on its load.
                Z = pair_kernel(d, bl)                                  # (nk, c, n_pg·ps)
                if stop_at == 'kconv':
                    out.append(acc_v + jnp.sum(jnp.abs(Z)))
                    continue
                if q_neg is not None:
                    Z = Z + jnp.conj(jnp.take(Z, jnp.asarray(q_neg), axis=0))
                Z = jnp.take(Z, jnp.asarray(q_sel), axis=0).reshape(Q, c, n_pg, ps)
                Z = (Z * qin[:, None, None, :]).reshape(Q, c, n_pg, n_b, n_c)
                Fz = local_fftn3(Z, axes=(-2, -1), norm='backward').reshape(Q, c, n_pg, ps)
                out.append(acc_v + jnp.einsum('qcpj,qpg->qcjg', jnp.take(Fz, zc, axis=-1), E))
            return tuple(out), None

        acc, _ = jax.lax.scan(
            group, (jnp.zeros((Q, c, n_zc, n_za), jnp.complex128),) * n_v,
            jnp.arange(n_grp, dtype=jnp.int32), unroll=1)
        if stop_at == 'kconv':
            return tuple(jnp.zeros((Q, c, n_g), jnp.complex128) + jnp.sum(jnp.abs(a))
                         for a in acc)
        return tuple(jnp.take_along_axis(a.reshape(Q, c, n_zc * n_za),
                                         zflat[:, None, :], axis=-1) for a in acc)

    fn = jax.jit(_local)
    _kernel_cache[key] = fn
    return fn


# ---------------------------------------------------------------------------
# The Z store: write-once batches, streamed G tiles
# ---------------------------------------------------------------------------

def _reorder_rank_major(x, P_: int, n_batch: int, c: int, mu_pad: int, axis: int):
    """μ axis in (rank, batch, j) order → packed (batch, rank, j), cut to ``mu_pad``."""
    sh = x.shape
    x = x.reshape(*sh[:axis], P_, n_batch, c, *sh[axis + 1:])
    x = jnp.swapaxes(x, axis, axis + 1)
    x = x.reshape(*sh[:axis], P_ * n_batch * c, *sh[axis + 1:])
    return jax.lax.slice_in_dim(x, 0, mu_pad, axis=axis)


class ZStore:
    """Write-once ``Z_q(μ, G)``; the GPU holds only the batch being written.

    Tile-major: G is cut into ``n_Gt`` tiles of ``G_tile`` slots, so a batch
    write and a G-tile read are both a few large contiguous blocks (the
    owner's ``(q_tile, μ_batch, G_tile)`` chunking; SlabIO datasets are
    contiguous, so the tiling lives in the dataset shape).  One resource,
    placed by the memory planner:

    * ``'disk'``: a slab_io scratch dataset ``(n_Gt, Q, n_batch·b, G_tile)``
      in packed μ order.  A batch is one hyperslab at μ offset ``β·b``; a
      G tile is one hyperslab.
    * ``'host'``: one exact-bytes numpy block ``(n_Gt, Q, n_batch, c,
      G_tile)`` per local device: each rank keeps the rows it computed, and
      a G tile is one contiguous read.

    Rows are μ-owned (rank p owns batch slots ``p·c + [0, c)``, ``c = b/P``),
    as route G produces them.

    ponytail: the store never lives on the device (owner simplification,
    2026-09-23).  A small deck whose whole store would fit beside the batch
    (CrI3 8x8 P16: 1.1 GB/rank) pays one host round trip, ~1 s; the
    upgrade path is a device placement here, priced by the planner.

    :meth:`read_tile` returns ``Z[:, :, tile]`` either q-local
    ``(Q_pad, μ_pad, G_tile)`` at ``P(('x','y'), None, None)`` or G-split
    ``(Q, μ_pad, G_tile)`` at ``P(None, None, ('x','y'))``; the host/device
    tiers reach that layout with one all-to-all per tile.
    """

    def __init__(self, *, mesh: Mesh, q_axis, mu_pad: int, g_axis, b: int,
                 placement: str, scratch_path: str | None = None,
                 packed_from_slot=None, n_batch: int | None = None):
        """``q_axis``: stored q rows (carrier a multiple of P); ``g_axis``: the
        ζ sphere cut into whole G tiles (divisor = ``G_tile``).  Both are
        ``runtime.padding.PaddedAxis`` records the kernel and the finalize share."""
        self.mesh = mesh
        self.P = _mesh_size(mesh)
        self.q_axis, self.g_axis = q_axis, g_axis
        if q_axis.divisor != self.P:
            raise ValueError(f"ZStore: {q_axis} must divide over P={self.P}")
        self.Q, self.mu_pad, self.n_G = q_axis.logical, int(mu_pad), g_axis.carrier
        self.b, self.c = int(b), int(b) // self.P
        if self.b % self.P:
            raise ValueError(f"ZStore: batch {b} must be a multiple of P={self.P} "
                             "(each rank owns b/P of its rows)")
        self.g_tile = g_axis.divisor
        self.n_Gt = g_axis.carrier // g_axis.divisor
        self.n_batch = (-(-self.mu_pad // self.b) if n_batch is None
                        else int(n_batch))
        self.Q_pad = q_axis.carrier
        self.placement = str(placement)
        # The store's μ axis is batch-slot order (β·b + j); readers gather the
        # packed carrier from it.  Contiguous batches make this a prefix.
        pfs = (np.arange(self.mu_pad, dtype=np.int32) if packed_from_slot is None
               else np.asarray(packed_from_slot, dtype=np.int32))
        if pfs.shape != (self.mu_pad,):
            raise ValueError(f"ZStore: packed_from_slot has shape {pfs.shape}")
        self._pfs = pfs
        self.bytes_written = 0
        self.bytes_read = 0
        self.t_write = 0.0
        self.t_read = 0.0
        if self.placement == 'host':
            # One tile-major numpy block per addressable device, exact bytes
            # (Q·n_batch·c·N_G·16 per rank): a G tile's rows for every batch
            # are one contiguous (Q, n_batch·c, G_tile) read.  ponytail:
            # pageable, not the pinned HostTileStore -- VI3 12x12 P16 (33.4
            # GB/rank of Z) was host-OOM-killed on it (p4v_vi3_p16_whole); the
            # pinned tier returns when the planner can price its overhead.
            # np.zeros commits pages on first write, so a batch never
            # written (a truncated debug fit) costs nothing and reads zero.
            self._host = {dev.id: np.zeros(
                (self.n_Gt, self.Q, self.n_batch, self.c, self.g_tile), np.complex128)
                for dev in mesh.local_devices}
        elif self.placement == 'disk':
            from file_io.slab_io import SlabIO
            if scratch_path is None:
                raise ValueError("ZStore(disk) needs scratch_path")
            self.path = str(scratch_path)
            self._io = SlabIO(self.path, mode='w', mesh=mesh)
            self._io.create_dataset(
                'Z', shape=(self.n_Gt, self.Q, self.n_batch * self.b,
                            self.g_tile), dtype=np.complex128)
        else:
            raise ValueError(f"ZStore: unknown placement {placement!r}")

    # -- write ------------------------------------------------------------
    def write_batch(self, beta: int, rows: jax.Array) -> None:
        """``rows (Q, b, n_Gt·G_tile)`` at ``P(None, ('x','y'), None)`` for batch ``beta``."""
        t0 = time.perf_counter()
        if self.placement == 'host':
            # The next batch is already dispatched, so this D2H overlaps it.
            for shard in rows.addressable_shards:
                d = np.asarray(shard.data).reshape(self.Q, self.c, self.n_Gt,
                                                   self.g_tile)
                self._host[shard.device.id][:, :, int(beta)] = d.transpose(2, 0, 1, 3)
        else:
            tiled = _tile_rows(self.mesh, self.Q, self.b, self.n_Gt, self.g_tile)(rows)
            self._io.write_slab('Z', tiled, offset=(0, 0, int(beta) * self.b, 0))
            jax.block_until_ready(tiled)
        self.bytes_written += self.Q * self.b * self.n_Gt * self.g_tile * 16
        self.t_write += time.perf_counter() - t0

    def sync(self) -> None:
        """Drain this store's queued (asynchronous, collective) disk writes.

        Two stores writing at once put two SlabIO worker threads' collective
        MPI-IO on the wire together, in an order no rank agrees on; a
        multi-channel fit drains each store before the next one writes.
        """
        if self.placement == 'disk':
            self._io.sync_writes()

    # -- read -------------------------------------------------------------
    def read_tile(self, t: int, *, layout: str) -> jax.Array:
        """G tile ``t``: ``layout='q'`` q-local, ``'g'`` G-split (see class doc)."""
        if layout not in ('q', 'g'):
            raise ValueError(f"ZStore.read_tile: layout {layout!r}")
        t0 = time.perf_counter()
        t = int(t)
        if self.placement == 'disk':
            self._io.sync_writes()
            if layout == 'q':
                raw = self._io.read_slab(
                    'Z', shape=(1, self.Q_pad, self.n_batch * self.b, self.g_tile),
                    offset=(t, 0, 0, 0), mesh=self.mesh,
                    partition_spec=P(None, _XY, None, None))
            else:
                raw = self._io.read_slab(
                    'Z', shape=(1, self.Q, self.n_batch * self.b, self.g_tile),
                    offset=(t, 0, 0, 0), mesh=self.mesh,
                    partition_spec=P(None, None, None, _XY))
            out = _disk_tile(self.mesh, layout, self.n_batch * self.b)(raw)
        else:
            nc = self.n_batch * self.c
            local = _host_tile_to_device(
                self.mesh, P(None, _XY, None), (self.Q, self.P * nc, self.g_tile),
                {dev: self._host[dev.id][t].reshape(self.Q, nc, self.g_tile)
                 for dev in self.mesh.local_devices})
            out = _rows_to_layout(self.mesh, layout, self.P, self.n_batch,
                                  self.c, self.n_batch * self.b, self.q_axis)(local)
        # Store slot order → packed centroid order (a prefix for contiguous
        # batches, a gather for orbit batches).
        out = _slots_to_packed(self.mesh, layout)(out, jnp.asarray(self._pfs))
        # Not blocked: the caller prefetches tile t+1 behind tile t's work.
        self.bytes_read += self.Q * self.mu_pad * self.g_tile * 16
        self.t_read += time.perf_counter() - t0
        return out

    def close(self) -> None:
        if self.placement == 'host':
            self._host.clear()
        else:
            self._io.close()
            if jax.process_index() == 0:
                import os
                try:
                    os.unlink(self.path)
                except OSError:
                    pass

    def receipt(self) -> str:
        return (f"  Z store: placement={self.placement}, μ-owned rows, "
                f"(Q={self.Q}, μ={self.mu_pad}, "
                f"N_G={self.n_G}) as {self.n_Gt} tiles x {self.g_tile}, "
                f"{self.n_batch} batches of {self.b}; written "
                f"{self.bytes_written / 1e9:.2f} GB in {self.t_write:.2f} s, "
                f"read {self.bytes_read / 1e9:.2f} GB in {self.t_read:.2f} s "
                f"(global volumes)")


def _tile_rows(mesh, Q, b, n_Gt, g_tile):
    """``(Q, b, n_Gt·G_tile)`` μ-sharded → ``(n_Gt, Q, b, G_tile)`` μ-sharded (local)."""
    key = ('tile_rows', _mesh_id(mesh), Q, b, n_Gt, g_tile)
    fn = _kernel_cache.get(key)
    if fn is None:
        @partial(shard_map, mesh=mesh, in_specs=(P(None, _XY, None),),
                 out_specs=P(None, None, _XY, None), check_vma=False)
        def _f(r):
            q, c, _ = r.shape
            return jnp.transpose(r.reshape(q, c, n_Gt, g_tile), (2, 0, 1, 3))
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


def _rows_to_layout(mesh, layout, P_, n_batch, c, mu_pad, q_axis):
    """Rank-local rows ``(Q, P·n_batch·c, g)`` (μ in rank-major order) → q-local or G-split."""
    key = ('rows_to_layout', _mesh_id(mesh), layout, P_, n_batch, c, mu_pad, q_axis)
    fn = _kernel_cache.get(key)
    if fn is None:
        out_spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(P(None, _XY, None),),
                 out_specs=out_spec, check_vma=False)
        def _f(x):                                        # (Q, n_batch·c, g)
            if layout == 'q':
                x = pad_to_axis(x, q_axis, axis=0)
                x = jax.lax.all_to_all(x, _XY, split_axis=0, concat_axis=1,
                                       tiled=True)        # (Q_pad/P, P·n_b·c, g)
            else:
                x = jax.lax.all_to_all(x, _XY, split_axis=2, concat_axis=1,
                                       tiled=True)        # (Q, P·n_b·c, g/P)
            return _reorder_rank_major(x, P_, n_batch, c, mu_pad, axis=1)
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


def _disk_tile(mesh, layout, mu_pad):
    key = ('disk_tile', _mesh_id(mesh), layout, mu_pad)
    fn = _kernel_cache.get(key)
    if fn is None:
        in_spec = (P(None, _XY, None, None) if layout == 'q'
                   else P(None, None, None, _XY))
        out_spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec,
                 check_vma=False)
        def _f(raw):
            return jax.lax.slice_in_dim(raw[0], 0, mu_pad, axis=1)
        fn = jax.jit(_f)
        _kernel_cache[key] = fn
    return fn


# ---------------------------------------------------------------------------
# ζ held as (Z store, C⁺): formed tile by tile, never materialized whole
# ---------------------------------------------------------------------------


def shell_positions(shell_slots, sel):
    """Positions in the kept columns ``shell_slots (Q, n)`` of the sphere
    slots ``sel (Q, j)``; a slot the pass did not keep is a refusal."""
    kept = np.asarray(shell_slots, dtype=np.int32)
    sel = np.asarray(sel, dtype=np.int32)
    pos = np.zeros_like(sel)
    for q in range(sel.shape[0]):
        for j in range(sel.shape[1]):
            hit = np.flatnonzero(kept[q] == sel[q, j])
            if hit.size == 0:
                raise ValueError(
                    "GATE zeta-mubatch-shell: got head slot "
                    f"{int(sel[q, j])} at stored q {q}, want a slot the V_q "
                    f"pass kept ({kept[q].tolist()}); why: that pass keeps "
                    "only the columns its head consumers named, so this "
                    "consumer would read a ζ column that was never formed.")
            pos[q, j] = hit[0]
    return pos


class ZetaG:
    """ζ_q(G) = C_q⁺ Z_q(G) held as its :class:`ZStore` and the C⁺ factor.

    Presents the metadata a G-flat ζ reader presents (``n_rmu``,
    ``zeta_layout``, ``gvec_components``) so the V_q and head-channel
    consumers take it in place of a ``ZetaLoader``; their data comes through
    :meth:`contract_v`, which streams G tiles once: C⁺ applied on each tile,
    ``V_q += conj(ζ) diag(v_q) ζᵀ`` accumulated, ζ kept at the columns the
    caller names (the head consumers' slots), and each ζ tile written only
    when a file is wanted.  On the ``local`` tier every
    rank owns whole q's (the factor's R4 batch layout) and the pass moves no
    data but the store read; the only resident accumulator is V (Q·μ²/P).
    """

    zeta_layout = 'G_flat'

    def __init__(self, store, *, mesh, L_q, lu_piv, solver_kind,
                 zeta_gather, batched_route, n_rmu_solve, n_rmu, mu_basis,
                 ngk_per_q, gvec_components, path, print_fn=print):
        from isdf.core import FactorToken
        if isinstance(L_q, FactorToken):
            raise ValueError(
                "ZetaG: route G applies a whole-tile factor on each G tile; got a "
                "block-cyclic FactorToken (factor the channel with "
                "distrib_la_batched_route='batch_reshard').")
        self.store = store
        self.print_fn = print_fn        # the fit's report sink (V_q receipt)
        self.mesh = mesh
        self.L_q, self.lu_piv = L_q, lu_piv
        # The hoisted transverse LU travels as (LU, pivots): the 'lu' seam.
        self.solver_kind = 'lu' if lu_piv is not None else str(solver_kind)
        self.zeta_gather = str(zeta_gather)
        self.batched_route = str(batched_route)
        self.n_rmu_solve = int(n_rmu_solve)
        self.n_rmu = int(n_rmu)
        self.n_rmu_disk = int(n_rmu)
        self.mu_basis = mu_basis
        self.ngk_per_q = np.asarray(ngk_per_q, dtype=np.int32)
        self.gvec_components = np.asarray(gvec_components, dtype=np.int32)
        self.ngkmax = int(self.gvec_components.shape[-1])
        self.path = str(path)
        self.shell_slots = self.shell = None
        self.receipt = ""

    @property
    def q_local(self) -> bool:
        """The solve tier decides the finalize layout: q-local reads each G
        tile onto q owners (one all-to-all per tile from μ-owned rows)."""
        return self.zeta_gather == 'local'

    @property
    def factor(self):
        """The factor operand of :func:`_logical_solve`: B, C⁺ or (LU, pivots)."""
        return self.L_q if self.lu_piv is None else (self.L_q, self.lu_piv)

    def write_file(self, zeta_io, *, print_fn=None):
        """Stream every G tile once and write ζ = C⁻¹Z into ``zeta_q_G`` (no V)."""
        Q = self.store.Q
        self.contract_v(np.zeros((Q, self.ngkmax), np.complex128),
                        keep=np.zeros((Q, 1), np.int32), zeta_io=zeta_io,
                        print_fn=print_fn, with_v=False)


    # -- the one pass ---------------------------------------------------
    def contract_v(self, v_table, *, keep, zeta_io=None, print_fn=None, with_v=True):
        """Stream every G tile once; return V (Q, μ_pad, μ_pad) at ``P(None,'x','y')``.

        ``v_table`` is ``(Q, ngkmax)`` v(q+G) on the stored sphere.  ``keep``
        ``(Q, n)`` names the sphere slots the head consumers read (slot 0
        first); ζ at them comes back as ``self.shell (Q, μ_pad, n)``.  V and
        the shell are in the canonical (file) centroid order.  With
        ``zeta_io`` the masked ζ tiles are also written to ``zeta_q_G``;
        ``with_v=False`` (:meth:`write_file`) forms and writes ζ only.
        """
        t0 = time.perf_counter()
        print_fn = print_fn or self.print_fn
        st = self.store
        # Pad q rows and G-tile slots carry v = 0, ngk = 0: inert in V.
        qa, ga = st.q_axis, st.g_axis
        v = np.asarray(pad_to_axis(pad_to_axis(
            jnp.asarray(v_table, dtype=jnp.complex128), qa, axis=0), ga, axis=1))
        ngk = np.asarray(pad_to_axis(jnp.asarray(self.ngk_per_q, jnp.int32), qa, axis=0))
        keep = np.asarray(keep, dtype=np.int32)
        if keep.ndim != 2 or keep.shape[0] != st.Q:
            raise ValueError(
                f"ZetaG.contract_v: keep must be (Q={st.Q}, n) slots; got "
                f"{keep.shape}.")
        sl = np.asarray(pad_to_axis(jnp.asarray(keep, jnp.int32), qa, axis=0))
        from common.collectives import device_put_process_local
        if self.q_local:
            layout = 'q'
            v_dev = device_put_process_local(
                v, NamedSharding(self.mesh, P(_XY, None)))
            ngk_dev = device_put_process_local(
                ngk, NamedSharding(self.mesh, P(_XY)))
            sl_dev = device_put_process_local(
                sl, NamedSharding(self.mesh, P(_XY, None)))
        else:
            layout = 'g'
            rep = NamedSharding(self.mesh, P())
            v_dev = device_put_process_local(np.asarray(strip_axis(v, qa, axis=0)), rep)
            ngk_dev = device_put_process_local(np.asarray(strip_axis(ngk, qa, axis=0)), rep)
            sl_dev = device_put_process_local(np.asarray(strip_axis(sl, qa, axis=0)), rep)
        dbg = _debug_enabled() and with_v
        step = _v_tile_kernel(self.mesh, layout, self.solver_kind,
                              self.n_rmu_solve, st.g_tile, debug_m=dbg, with_v=with_v)
        mu = int(st.mu_pad)
        V, M, shell = _zero_accumulators(self.mesh, layout, st.Q_pad, st.Q, mu,
                                         int(sl.shape[1]), debug_m=dbg, with_v=with_v)
        L_arg = self.factor
        if layout == 'g':
            L_arg = jax.tree.map(lambda a: jax.lax.with_sharding_constraint(
                a, NamedSharding(self.mesh, P())), L_arg)
        nxt = st.read_tile(0, layout=layout)
        for t in range(st.n_Gt):
            # Tile t+1 is read while tile t is contracted.
            Zt = nxt
            nxt = st.read_tile(t + 1, layout=layout) if t + 1 < st.n_Gt else None
            V, M, shell, zt = step(L_arg, Zt, v_dev, ngk_dev, sl_dev,
                                   jnp.int32(t), V, M, shell)
            if zeta_io is not None:
                self._write_tile(zeta_io, zt, t * st.g_tile)
                if st.placement == 'disk':
                    # The next store read enters another HDF5 handle; the
                    # asynchronous ζ write must reach MPI-IO first on every
                    # rank (SlabIO.sync_writes).
                    zeta_io.sync_writes()
            del Zt, zt
        if not dbg:
            M = None
        if not with_v:
            V = None
            self.receipt = (f"  μ-batch ζ pass: {st.n_Gt} G tiles, {layout}-layout, "
                            f"{time.perf_counter() - t0:.2f}s (store read "
                            f"{st.t_read:.2f}s); zeta file written")
            return None
        V = _finish_v(self.mesh, layout, st.Q)(V)
        shell = _finish_shell(self.mesh, layout, st.Q)(shell)
        if self.mu_basis is not None:
            V = self.mu_basis.unpack_operator(V)
            shell = self.mu_basis.unpack_axis(shell, 1)
        self.shell, self.shell_slots = shell, keep
        if M is not None:
            V_cmc = _v_from_m(self.mesh, layout, self.solver_kind,
                              self.n_rmu_solve)(L_arg, M)
            V_cmc = _finish_v(self.mesh, layout, st.Q)(V_cmc)
            if self.mu_basis is not None:
                V_cmc = self.mu_basis.unpack_operator(V_cmc)
            d = float(jnp.linalg.norm(V_cmc - V) / jnp.linalg.norm(V))
            if jax.process_index() == 0:
                print_fn(f"  μ-batch V check: conj(C+) M conj(C+) vs zeta-first "
                         f"rel {d:.3e} (production keeps zeta-first)")
        peak = (jax.local_devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
        self.receipt = (f"  μ-batch V_q: {st.n_Gt} G tiles, {layout}-layout, "
                        f"{time.perf_counter() - t0:.2f}s (store read "
                        f"{st.t_read:.2f}s); zeta file "
                        f"{'written' if zeta_io is not None else 'not written'}; "
                        f"device peak so far {peak / 1e9:.2f} GB")
        if jax.process_index() == 0:
            print_fn(self.receipt)
        return V

    def _write_tile(self, zeta_io, zt, g0):
        """One masked ζ tile into ``zeta_q_G`` (canonical μ order, clipped)."""
        if zt.sharding.spec[0] is not None:     # q-local → the writer's layout
            zt = _to_mu_owner(self.mesh, 'q', self.store.Q, 'mu')(zt)
        else:
            zt = zt[:self.store.Q]
        if self.mu_basis is not None:
            zt = self.mu_basis.unpack_axis(zt, 1)
        zeta_io.write_slab('zeta_q_G', zt, offset=(0, 0, int(g0)))

    def head_columns(self, sel):
        """Positions in :attr:`shell` of the stored-sphere slots ``sel (Q, j)``."""
        return shell_positions(self.shell_slots, sel)

    def close(self):
        self.store.close()
        self.L_q = self.lu_piv = self.shell = None


def _debug_enabled() -> bool:
    from runtime import debug_print_enabled
    return bool(debug_print_enabled())


def _logical_solve(solver_kind: str, n_log: int):
    """ζ_q = C_q⁻¹ Z_q at the logical μ extent, through the channel's conditioning seam.

    The charge Gram is PSD: C⁺ = B Bᴴ, the λ > rcond·λ_max eigh cut
    (:mod:`isdf.cplus`).  A current channel's Gram C^μ is Hermitian
    INDEFINITE -- that cut would drop its whole negative half -- so the
    currents keep their own seam from :mod:`isdf.core`, the same arithmetic
    as every transverse fit: the sign-aware ridged pivoted LU factored once
    per channel (``'lu'``, operand ``(LU, pivots)``, κ_lb certified at
    factor time).
    """
    from isdf import cplus
    from isdf.core import _zeta_logical_solvers
    from runtime.padding import solve_at_logical
    n_log = int(n_log)
    if solver_kind == 'lu':
        lu_apply = _zeta_logical_solvers(n_log)[1]
        return lambda F, Z: lu_apply(F[0], F[1], Z)
    return lambda B, Z: solve_at_logical(cplus.apply, n_log, (B,), Z)


def _factor_specs(solver_kind: str, layout: str):
    """In-specs of the factor operand: whole q tiles (q-local) or replicated."""
    tile = P(_XY, None, None) if layout == 'q' else P()
    if solver_kind == 'lu':
        return (tile, P(_XY, None) if layout == 'q' else P())
    return tile


def _acc_specs(layout):
    """Accumulator layout: q-local blocks, or per-rank partial sums over local G."""
    return P(_XY, None, None) if layout == 'q' else P(_XY, None, None, None)


def _v_tile_kernel(mesh, layout, solver_kind, n_log, g_tile, *, debug_m, with_v=True):
    """One G tile: ζ = C⁻¹Z, V += conj(ζ) v ζᵀ, shell gather (and M in debug).

    ``layout='q'``: F (Q_pad, μ, μ) and Z (Q_pad, μ, g) q-local; V, shell and
    M accumulate on the q owner.  ``layout='g'``: F replicated, Z G-split;
    each rank accumulates its partial sums over its G columns into a leading
    rank axis, reduced once by :func:`_finish_v`.  ``with_v=False`` forms ζ
    only (the accumulators pass through untouched).
    """
    key = ('v_tile', _mesh_id(mesh), layout, solver_kind, int(n_log),
           int(g_tile), bool(debug_m), bool(with_v))
    fn = _kernel_cache.get(key)
    if fn is not None:
        return fn
    one = _logical_solve(solver_kind, n_log)
    acc = _acc_specs(layout)
    f_spec = _factor_specs(solver_kind, layout)
    if layout == 'q':
        in_specs = (f_spec, P(_XY, None, None), P(_XY, None),
                    P(_XY), P(_XY, None), P(), acc, acc, acc)
        z_spec = P(_XY, None, None)
    else:
        in_specs = (f_spec, P(None, None, _XY), P(), P(), P(), P(), acc, acc, acc)
        z_spec = P(None, None, _XY)

    @partial(shard_map, mesh=mesh, in_specs=in_specs,
             out_specs=(acc, acc, acc, z_spec), check_vma=False)
    def k(F, Z, v, ngk, sl, t, V, M, S):
        n_g = Z.shape[-1]
        g_idx = t * g_tile + jnp.arange(n_g, dtype=jnp.int32)
        if layout == 'g':
            g_idx = g_idx + jax.lax.axis_index(_XY) * n_g
        mask = g_idx[None, :] < ngk[:, None]                    # (q, g)
        zeta = jnp.where(mask[:, None, :], jax.vmap(one)(F, Z), 0)
        if not with_v:
            return V, M, S, zeta
        vt = jnp.where(mask, jnp.take(v, jnp.clip(g_idx, 0, v.shape[-1] - 1),
                                      axis=1), 0)
        dV = jnp.einsum('qmg,qg,qng->qmn', jnp.conj(zeta), vt, zeta)
        hit = (sl[:, None, :] == g_idx[None, :, None]).astype(zeta.dtype)
        dS = jnp.einsum('qmg,qgs->qms', zeta, hit)
        if layout == 'g':
            dV, dS = dV[None], dS[None]
        V = V + dV
        S = S + dS
        if debug_m:
            dM = jnp.einsum('qmg,qg,qng->qmn', jnp.conj(Z), vt, Z)
            M = M + (dM[None] if layout == 'g' else dM)
        return V, M, S, zeta

    fn = jax.jit(k, donate_argnums=(6, 7, 8))
    _kernel_cache[key] = fn
    return fn


def _pair_tile_kernel(mesh, layout, g_tile, pairs):
    """One G tile of several ζ's: ``V^{ab} += conj(ζ^a) v^{ab} ζ^bᵀ`` per pair,
    and each ζ's shell gather — the accumulation half of :func:`_v_tile_kernel`
    over ζ tiles it formed with ``with_v=False``."""
    pairs = tuple((int(a), int(b)) for a, b in pairs)
    key = ('pair_tile', _mesh_id(mesh), layout, int(g_tile), pairs)
    fn = _kernel_cache.get(key)
    if fn is not None:
        return fn
    n_z = 1 + max(max(p) for p in pairs)
    acc = _acc_specs(layout)
    if layout == 'q':
        z_spec, v_spec, n_spec, s_spec = (P(_XY, None, None), P(None, _XY, None),
                                          P(_XY), P(_XY, None))
    else:
        z_spec, v_spec, n_spec, s_spec = P(None, None, _XY), P(), P(), P()
    in_specs = ((z_spec,) * n_z, v_spec, n_spec, s_spec, P(),
                (acc,) * len(pairs), (acc,) * n_z)

    @partial(shard_map, mesh=mesh, in_specs=in_specs,
             out_specs=((acc,) * len(pairs), (acc,) * n_z), check_vma=False)
    def k(Z, v, ngk, sl, t, V, S):
        n_g = Z[0].shape[-1]
        g_idx = t * g_tile + jnp.arange(n_g, dtype=jnp.int32)
        if layout == 'g':
            g_idx = g_idx + jax.lax.axis_index(_XY) * n_g
        mask = g_idx[None, :] < ngk[:, None]                    # (q, g)
        cols = jnp.clip(g_idx, 0, v.shape[-1] - 1)
        V_out = []
        for i, (a, b) in enumerate(pairs):
            vt = jnp.where(mask, jnp.take(v[i], cols, axis=1), 0)
            dV = jnp.einsum('qmg,qg,qng->qmn', jnp.conj(Z[a]), vt, Z[b])
            V_out.append(V[i] + (dV[None] if layout == 'g' else dV))
        hit = (sl[:, None, :] == g_idx[None, :, None]).astype(Z[0].dtype)
        S_out = []
        for a in range(n_z):
            dS = jnp.einsum('qmg,qgs->qms', Z[a], hit)
            S_out.append(S[a] + (dS[None] if layout == 'g' else dS))
        return tuple(V_out), tuple(S_out)

    fn = jax.jit(k, donate_argnums=(5, 6))
    _kernel_cache[key] = fn
    return fn


def contract_v_group(zetas, pairs, v_tables, *, keep, print_fn=None):
    """Several V tiles over several ζ's in ONE pass over the G tiles.

    ``zetas`` are :class:`ZetaG` of one fit (one store geometry and centroid
    basis: the current family's three channels).  On each G tile every ζ^a =
    C_a⁺ Z^a is formed once (the ``with_v=False`` half of
    :meth:`ZetaG.contract_v`), then ``V^{ab}_q += conj(ζ^a) diag(v^{ab}_q)
    ζ^{bT}`` for every ``(a, b)`` in ``pairs`` with its own ``v_tables[i]``
    ``(Q, ngkmax)``, and each ζ keeps its ``keep`` columns (``.shell``), as
    :meth:`ZetaG.contract_v` does.  Returns ``[V^{ab}]`` in ``pairs`` order,
    ``(Q, μ_pad, μ_pad)`` at ``P(None,'x','y')`` in canonical centroid order.
    This is what spares the four-current V_q the three ζ_T files: the file
    route wrote each ζ^a and read it back to form the same sums.
    """
    t0 = time.perf_counter()
    z0 = zetas[0]
    print_fn = print_fn or z0.print_fn
    st0 = z0.store
    for z in zetas[1:]:
        st = z.store
        if (st.Q, st.mu_pad, st.g_tile, st.n_Gt) != (st0.Q, st0.mu_pad, st0.g_tile, st0.n_Gt) \
                or z.q_local != z0.q_local or z.mu_basis is not z0.mu_basis:
            raise ValueError(
                "contract_v_group: every ζ must share one store geometry, "
                "solve tier and centroid basis (one fit's channels).")
    if not z0.q_local:
        # G-split accumulators are whole (Q, μ, μ) per rank: one per pair is
        # what the fit's plan priced, not six (VI3 12×12 current: 2.9 GB each).
        raise ValueError(
            "contract_v_group: needs the q-local solve tier (q-sharded V "
            "accumulators); the G-split tier forms its tiles from ζ files.")
    qa, ga = st0.q_axis, st0.g_axis
    v = np.stack([np.asarray(pad_to_axis(pad_to_axis(
        jnp.asarray(t, dtype=jnp.complex128), qa, axis=0), ga, axis=1))
        for t in v_tables])                               # (n_pair, Q_pad, G)
    ngk = np.asarray(pad_to_axis(jnp.asarray(z0.ngk_per_q, jnp.int32), qa, axis=0))
    keep = np.asarray(keep, dtype=np.int32)
    if keep.ndim != 2 or keep.shape[0] != st0.Q:
        raise ValueError(
            f"contract_v_group: keep must be (Q={st0.Q}, n) slots; got {keep.shape}.")
    sl = np.asarray(pad_to_axis(jnp.asarray(keep, jnp.int32), qa, axis=0))
    from common.collectives import device_put_process_local
    mesh = z0.mesh
    if z0.q_local:
        layout = 'q'
        v_dev = device_put_process_local(v, NamedSharding(mesh, P(None, _XY, None)))
        ngk_dev = device_put_process_local(ngk, NamedSharding(mesh, P(_XY)))
        sl_dev = device_put_process_local(sl, NamedSharding(mesh, P(_XY, None)))
    else:
        layout = 'g'
        rep = NamedSharding(mesh, P())
        v_dev = device_put_process_local(np.asarray(strip_axis(v, qa, axis=1)), rep)
        ngk_dev = device_put_process_local(np.asarray(strip_axis(ngk, qa, axis=0)), rep)
        sl_dev = device_put_process_local(np.asarray(strip_axis(sl, qa, axis=0)), rep)
    form = [_v_tile_kernel(mesh, layout, z.solver_kind, z.n_rmu_solve, st0.g_tile,
                           debug_m=False, with_v=False) for z in zetas]
    stub = [_zero_accumulators(mesh, layout, st0.Q_pad, st0.Q, int(st0.mu_pad), 1,
                               debug_m=False, with_v=False) for _ in zetas]
    step = _pair_tile_kernel(mesh, layout, st0.g_tile, pairs)
    mu = int(st0.mu_pad)
    V = tuple(_zero_accumulators(mesh, layout, st0.Q_pad, st0.Q, mu, 1,
                                 debug_m=False)[0] for _ in pairs)
    S = tuple(_zero_accumulators(mesh, layout, st0.Q_pad, st0.Q, mu,
                                 int(sl.shape[1]), debug_m=False)[2] for _ in zetas)
    factors = []
    for z in zetas:
        f = z.factor
        if layout == 'g':
            f = jax.tree.map(lambda a: jax.lax.with_sharding_constraint(
                a, NamedSharding(mesh, P())), f)
        factors.append(f)
    v_form = v_dev[0]                  # unread by the ζ-forming half
    # No look-ahead: each Z tile is read just before its ζ is formed and
    # dropped after, so a G tile holds one Z tile and the ζ tiles — the six
    # tile-sized buffers the fit's plan sized g_tile for (MuBatchPlan).
    for t in range(st0.n_Gt):
        zt = []
        for i, z in enumerate(zetas):
            Zt = z.store.read_tile(t, layout=layout)
            a, m, sh, zi = form[i](factors[i], Zt, v_form, ngk_dev, sl_dev,
                                   jnp.int32(t), *stub[i])
            stub[i] = (a, m, sh)
            zt.append(zi)
            del Zt
        V, S = step(tuple(zt), v_dev, ngk_dev, sl_dev, jnp.int32(t), V, S)
        del zt
    out = []
    for Vp in V:
        Vp = _finish_v(mesh, layout, st0.Q)(Vp)
        out.append(z0.mu_basis.unpack_operator(Vp) if z0.mu_basis is not None else Vp)
    for z, Sa in zip(zetas, S):
        Sa = _finish_shell(mesh, layout, st0.Q)(Sa)
        z.shell = z0.mu_basis.unpack_axis(Sa, 1) if z0.mu_basis is not None else Sa
        z.shell_slots = keep
    receipt = (f"  μ-batch V_q group: {len(zetas)} ζ, {len(pairs)} tiles, "
               f"{st0.n_Gt} G tiles, {layout}-layout, "
               f"{time.perf_counter() - t0:.2f}s (store read "
               f"{sum(z.store.t_read for z in zetas):.2f}s); no ζ file")
    if jax.process_index() == 0:
        print_fn(receipt)
    return out


def _zero_accumulators(mesh, layout, Q_pad, Q, mu, n_sh, *, debug_m, with_v=True):
    P_ = _mesh_size(mesh)
    sh = NamedSharding(mesh, _acc_specs(layout))
    if layout == 'q':
        vs, ss = (Q_pad, mu, mu), (Q_pad, mu, n_sh)
    else:
        vs, ss = (P_, Q, mu, mu), (P_, Q, mu, n_sh)
    z = lambda shape: jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                              out_shardings=sh)()
    stub = lambda shape: z(shape[:1] + (1,) * (len(shape) - 1))
    if not with_v:                   # ζ only: no V, M or shell is formed
        return stub(vs), stub(vs), stub(ss)
    return z(vs), (z(vs) if debug_m else stub(vs)), z(ss)


def _to_mu_owner(mesh, layout, Q, split):
    """q-local ``(Q_pad, a, b)`` or per-rank partial sums ``(P, Q, a, b)`` →
    ``(Q, a, b)`` with μ split, in ONE collective (all-to-all or
    reduce-scatter over ('x','y')).

    ``split='xy'``: a over 'x' and b over 'y' (``P(None,'x','y')``, V);
    ``split='mu'``: a over ('x','y') (``P(None,('x','y'),None)``, shell, ζ tile).
    """
    key = ('to_mu_owner', _mesh_id(mesh), layout, int(Q), split)
    fn = _kernel_cache.get(key)
    if fn is not None:
        return fn
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    P_ = px * py
    out = P(None, 'x', 'y') if split == 'xy' else P(None, _XY, None)

    def blocks(x):
        """(n, a, b) → (P, n, a', b'): the block rank p = x·py + y owns, first."""
        n, a, b = x.shape
        if split == 'xy':
            return x.reshape(n, px, a // px, py, b // py).transpose(
                1, 3, 0, 2, 4).reshape(P_, n, a // px, b // py)
        return x.reshape(n, P_, a // P_, b).transpose(1, 0, 2, 3)

    @partial(shard_map, mesh=mesh, in_specs=(_acc_specs(layout),),
             out_specs=out, check_vma=False)
    def f(x):
        if layout == 'q':            # (Q_pad/P, a, b): the q block this rank owns
            x = jax.lax.all_to_all(blocks(x), _XY, 0, 0, tiled=True)
            x = x.reshape((-1,) + x.shape[2:])         # q blocks in rank order
        else:                        # (1, Q, a, b): this rank's partial sum
            x = jax.lax.psum_scatter(blocks(x[0]), _XY, scatter_dimension=0,
                                     tiled=True)[0]
        return x[:Q]
    fn = jax.jit(f)
    _kernel_cache[key] = fn
    return fn


def _finish_v(mesh, layout, Q):
    """Accumulator → ``(Q, μ, μ)`` at ``P(None, 'x', 'y')``."""
    return _to_mu_owner(mesh, layout, Q, 'xy')


def _finish_shell(mesh, layout, Q):
    """Shell accumulator → ``(Q, μ, n_shell)`` at ``P(None, ('x','y'), None)``."""
    return _to_mu_owner(mesh, layout, Q, 'mu')


def _v_from_m(mesh, layout, solver_kind, n_log):
    """Debug check: V = conj(C⁺) M conj(C⁺) by the same per-q solver (two applications)."""
    key = ('v_from_m', _mesh_id(mesh), layout, solver_kind, int(n_log))
    fn = _kernel_cache.get(key)
    if fn is None:
        one = _logical_solve(solver_kind, n_log)
        acc = _acc_specs(layout)
        f_spec = _factor_specs(solver_kind, layout)

        @partial(shard_map, mesh=mesh, in_specs=(f_spec, acc), out_specs=acc,
                 check_vma=False)
        def k(F, M):
            if layout == 'g':
                M = jax.lax.psum(M[0], _XY)
            X = jax.vmap(one)(F, jnp.conj(M))                 # C⁻¹ M*
            Vc = jnp.conj(jax.vmap(one)(F, jnp.conj(jnp.swapaxes(X, -1, -2))))
            if layout == 'g':
                Vc = jnp.where(jax.lax.axis_index(_XY) == 0, Vc, 0)[None]
            return Vc
        fn = jax.jit(k)
        _kernel_cache[key] = fn
    return fn


def _host_tile_to_device(mesh, spec, shape, local_tiles):
    """This process's host tiles → one sharded array, through the collectives service.

    ``common.collectives.restore_from_host`` is the one place this process
    places host shards on its own devices (no cross-process traffic).
    """
    from common.collectives import HostSpill, restore_from_host
    return restore_from_host(HostSpill(
        shape=tuple(int(v) for v in shape),
        sharding=NamedSharding(mesh, spec),
        shards=[(dev, np.ascontiguousarray(a)) for dev, a in local_tiles.items()]))


def _slots_to_packed(mesh, layout):
    """Store slot order → packed centroid order on the (unsharded) μ axis."""
    key = ('slots_to_packed', _mesh_id(mesh), layout)
    fn = _kernel_cache.get(key)
    if fn is None:
        spec = P(_XY, None, None) if layout == 'q' else P(None, None, _XY)

        @partial(shard_map, mesh=mesh, in_specs=(spec, P()), out_specs=spec,
                 check_vma=False)
        def _g(x, pfs):
            # -1: a layout pad centroid, in no batch; its Z row is exactly zero.
            y = jnp.take(x, jnp.clip(pfs, 0, None), axis=1)
            live = (pfs >= 0)[None, :, None]
            return jnp.where(live, y, 0)
        fn = jax.jit(_g)
        _kernel_cache[key] = fn
    return fn
