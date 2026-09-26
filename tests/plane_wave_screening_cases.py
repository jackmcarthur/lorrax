"""Cases and dense references for ``gw.plane_wave_screening`` (the CPU suite and the P4 gate).

Nothing here imports the module under test.  The references are numpy evaluations of
the definitions: a BerkeleyGW M-matrix band sum for χ, the closed-form Coulomb
kernels, explicit per-q inverses for the Dyson solve and the bordered Γ inverse, and a
host-side typed transport for the antiunitary rule.
"""
from __future__ import annotations

import numpy as np

import mixed_basis_pair_conv_cases as cases


# ---------------------------------------------------------------------------
# a small orthorhombic crystal with random orthonormal bands
# ---------------------------------------------------------------------------

LAT = np.diag([4.3, 4.7, 5.1])                      # Bohr, rows a1 a2 a3


def bvec_rows(lat=LAT):
    return 2.0 * np.pi * np.linalg.inv(lat).T       # rows b1 b2 b3 (1/Bohr)


def orthonormal_bands(rng, kfrac, lat, ecut, nb, ns, *, span=3):
    """ψ coefficients ``c (N_k, nb, ns, width)`` orthonormal over (G, spin) per k, the sphere
    ``|k+G|² ≤ ecut`` (Cartesian, Ry), and ``(gvecs, ngk)``."""
    b = bvec_rows(lat)
    sph, ngk = cases.spheres(kfrac, b @ b.T, ecut, span=span)
    nk, w = len(kfrac), sph.shape[1]
    c = np.zeros((nk, nb, ns, w), np.complex128)
    for k in range(nk):
        n = int(ngk[k]) * ns
        if n < nb:
            raise ValueError("orthonormal_bands: sphere smaller than the band count")
        a = rng.standard_normal((n, nb)) + 1j * rng.standard_normal((n, nb))
        qm, _ = np.linalg.qr(a)
        c[k, :, :, :ngk[k]] = qm.T.reshape(nb, ns, int(ngk[k]))
    return c, sph, ngk


def band_sum_chi(c, E, nv, sph, ngk, kfrac, kgrid, out_gvecs, out_ngk, out_frac, *, tau,
                 cell_volume, s):
    """χ_q(G, G'; τ) = −s/(N_k Ω) Σ_k Σ_{c,v} conj M_cv(G) M_cv(G') e^{-(E_c,k+q − E_v,k) τ},
    M_cv(k, q, G) = ⟨c, k+q| e^{i(q+G)·r} |v, k⟩ from plane-wave coefficients (BerkeleyGW)."""
    nb = c.shape[1]
    return band_sum_response(c, E, np.arange(nv, nb), np.arange(nv), sph, ngk, kfrac, kgrid,
                             out_gvecs, out_ngk, out_frac, lambda dE: np.exp(-dE * tau),
                             cell_volume=cell_volume, s=s)


def band_sum_response(c, E, upper, lower, sph, ngk, kfrac, kgrid, out_gvecs, out_ngk, out_frac,
                      wfun, *, cell_volume, s):
    """−s/(N_k Ω) Σ_k Σ_{a∈upper, b∈lower} conj M_ab(G) M_ab(G') wfun(E_a,k+q − E_b,k),
    M_ab(k, q, G) = ⟨a, k+q| e^{i(q+G)·r} |b, k⟩ (band index sets ``upper`` at k+q, ``lower`` at k)."""
    kg = np.asarray(kgrid)
    nk = len(kfrac)
    kint = np.rint(np.asarray(kfrac) * kg).astype(np.int64)
    look = [{tuple(g): i for i, g in enumerate(sph[k, :ngk[k]])} for k in range(nk)]
    nq, wo = out_gvecs.shape[0], out_gvecs.shape[1]
    up, lo = np.asarray(upper), np.asarray(lower)
    chi = np.zeros((nq, wo, wo), np.complex128)
    for iq, q in enumerate(np.asarray(out_frac)):
        for k in range(nk):
            kq = np.asarray(kfrac[k]) + q
            kj_int = np.rint(kq * kg).astype(np.int64) % kg
            j = int(np.flatnonzero(np.all(kint % kg == kj_int, axis=1))[0])
            g0 = np.rint(kq - kfrac[j]).astype(np.int64)
            M = np.zeros((len(up), len(lo), wo), np.complex128)          # (a, b, G)
            for ig in range(int(out_ngk[iq])):
                G = out_gvecs[iq, ig]
                for ip in range(int(ngk[k])):
                    t = look[j].get(tuple(sph[k, ip] + G + g0))
                    if t is None:
                        continue
                    # Σ_α conj c_a,k+q(p+G+g0, α) c_b,k(p, α)
                    M[:, :, ig] += np.einsum("ca,va->cv", np.conj(c[j, up][:, :, t]), c[k, lo][:, :, ip])
            wgt = wfun(E[j, up][:, None] - E[k, lo][None, :])
            chi[iq] += np.einsum("cvg,cv,cvh->gh", np.conj(M), wgt, M)
    return -s / (nk * float(cell_volume)) * chi


# ---------------------------------------------------------------------------
# closed-form Coulomb, dense Dyson and the bordered Γ inverse
# ---------------------------------------------------------------------------

def coulomb_closed_form(q_frac, gvecs, ngk, bvec, *, sys_dim):
    """8π/|q+G|² (slab: × (1 − e^{-z_c|K∥|} cos(K_z z_c)), z_c = π/b3z), zero at q+G = 0 and pads."""
    K = (np.asarray(q_frac)[:, None, :] + gvecs) @ np.asarray(bvec)
    k2 = np.sum(K * K, axis=-1)
    live = np.arange(gvecs.shape[1])[None, :] < np.asarray(ngk)[:, None]
    v = np.where(k2 > 1e-12, 8.0 * np.pi / np.where(k2 > 1e-12, k2, 1.0), 0.0)
    if sys_dim == 2:
        zc = np.pi / float(np.asarray(bvec)[2, 2])
        v = v * (1.0 - np.exp(-zc * np.hypot(K[..., 0], K[..., 1])) * np.cos(K[..., 2] * zc))
    return np.where(live, v, 0.0)


def dense_dyson(v, chi):
    """W_q = (I − diag(v_q) χ_q)⁻¹ diag(v_q), q by q, explicit inverse."""
    out = np.zeros_like(chi)
    for q in range(chi.shape[0]):
        V = np.diag(v[q]).astype(np.complex128)
        out[q] = np.linalg.inv(np.eye(chi.shape[1]) - V @ chi[q]) @ V
    return out


def bordered_head(q_cart, S, Y, Z, chi_bb, v_b):
    """The full Γ-neighbourhood inverse at a small Cartesian q (head slot 0, body 1..):
    χ(q) = [[qᵀSq, qᵀY], [Zq, χ_bb]], v = diag(8π/q², v_b); returns W (1+n_b, 1+n_b)."""
    q = np.asarray(q_cart, np.float64)
    n = chi_bb.shape[0]
    chi = np.zeros((n + 1, n + 1), np.complex128)
    chi[0, 0] = q @ S @ q
    chi[0, 1:] = q @ Y
    chi[1:, 0] = Z @ q
    chi[1:, 1:] = chi_bb
    v = np.concatenate([[8.0 * np.pi / float(q @ q)], v_b]).astype(np.complex128)
    return np.linalg.inv(np.eye(n + 1) - v[:, None] * chi) * v[None, :]


# ---------------------------------------------------------------------------
# the typed transport of a two-point operator on sphere slots (host)
# ---------------------------------------------------------------------------

def transport_operator(par, src, phase, anti):
    """O_child[p, p'] = ph(p) S[src p, src p'] conj ph(p') (unitary), and its conjugate on an
    antiunitary row (the partner is conj S: a two-point operator obeying O(gx, gx') = conj O(x, x')).
    ``par (w, w)`` the parent operator, ``src``/``phase (w,)`` the child's slot tables."""
    img = phase[:, None] * par[np.ix_(src, src)] * np.conj(phase)[None, :]
    return np.conj(img) if anti else img


# ---------------------------------------------------------------------------
# a time-reversal-broken magnetic group: {E, Θ·glide}
# ---------------------------------------------------------------------------

def glide_tr_broken_fixture(mesh, ns, *, theta=np.pi / 2):
    """``zeta_mubatch_fixtures._glide_fixture``'s crystal with the group {E, Θ·glide} only.

    The glide fixture's own group holds Θ itself (rows E, glide, Θ, Θ·glide), so its χ
    is (r, r')-symmetric and cannot tell z from z̄.  Here Θ·glide carries k(0,½) to
    k(½,0) and every other k is its own parent; the spin rotation exp(−iθσ_x) at θ = π/2
    is a representation."""
    from types import SimpleNamespace
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from symmetry_maps import centroid_source_map_and_wrap, spinor_rotation_for_sym_row
    import zeta_mubatch_fixtures as fixtures
    fft_grid = (4, 4, 4)
    kgrid = (2, 2, 1)
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    ops = np.stack([np.eye(3, dtype=np.int64), swap])
    tnp = np.asarray([[0.0, 0.0, 0.0], [np.pi, np.pi, 0.0]])
    kints = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    kfrac = kints / np.asarray(kgrid, dtype=np.float64)
    irr = np.asarray([0, 1, 1, 2], dtype=np.int32)
    sym_rows = np.asarray([0, 0, 3, 0], dtype=np.int32)          # k(½,0) = Θ·glide k(0,½)
    parent_k = kfrac[[0, 1, 3]]
    U1 = np.asarray([[np.cos(theta), -1j * np.sin(theta)],
                     [-1j * np.sin(theta), np.cos(theta)]])
    U_spatial = np.stack([np.eye(2, dtype=np.complex128), U1])

    def spinor_action(rows, *, nspinor):
        return spinor_rotation_for_sym_row(U_spatial, np.asarray(rows), 2,
                                           nspinor=nspinor, R_cart=ops)

    sym = SimpleNamespace(sym_matrices=ops, translations=tnp, irr_idx_k=irr,
                          sym_idx_k=sym_rows, spinor_action=spinor_action,
                          unfolded_kpts=kfrac, kirr_fullids=np.asarray([0, 1, 3]))
    grid = fixtures._grid_points(fft_grid)
    perm_g, _ = centroid_source_map_and_wrap(grid, ops, tnp, fft_grid, extend_trs=True)
    cent = []
    for seed in (0, 5, 22, 27, 41, 50, 60, 63, 9, 14):
        orbit = sorted({int(perm_g[s, seed]) for s in range(4)})
        if len(cent) + len(orbit) <= 8 and not any(c in cent for c in orbit):
            cent.extend(orbit)
        if len(cent) == 8:
            break
    plan = build_centroid_k_unfold_plan(sym, grid[np.asarray(sorted(cent))], fft_grid, mesh,
                                        nspinor=ns, parent_k_frac=parent_k)
    return dict(plan=plan, fft_grid=fft_grid, kgrid=kgrid, kfull=kfrac, ops=ops, tnp=tnp,
                rows=np.asarray([0, 3], dtype=np.int32), spinor_action=spinor_action)
