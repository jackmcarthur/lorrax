"""Cases and materialized references for ``gw.mixed_basis_pair_convolution``.

Shared by the CPU suite (``tests/test_mixed_basis_pair_convolution.py``) and the
P4 GPU gate (``tests/multi_device/mixed_basis_pair_conv_p4.py``).  Nothing here
imports the kernel's internals: the references are dense numpy evaluations of the
definition in the module docstring.

* ``dense_reference``: a_k(r, r') per k by explicit Fourier matrices, A(r+R, r') on
  the whole supercell by an explicit k sum, the product, and X_q(G, G') by explicit
  Fourier matrices and an explicit R sum.
* ``ksum_reference``: the direct k-sum convolution
  X̂_q(r, r') = N_k⁻¹ Σ_k a_k conj(c_{k-q}), then the same output matrices.
* ``dense_scalar_reference``: the Σ product A^{αβ}(r+R, r') B(r+R, r') on the same
  materialized supercell, each operand on its own spheres.
* ``band_sum_sigma`` / ``band_sum_chi``: the physical normalization, Σ_nn'(k) and
  χ_q(G, G') as BGW band sums over explicit plane-wave matrix elements (no box).

Symmetry cases realize the full-grid operands from parent ψ by the r-space typed
action on the grid (``zeta_mubatch_fixtures._children``: the transport C_q and the
faces use), projected back onto each child's sphere, so the parent input's G-space
transport is checked against code it shares nothing with.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# spheres and operands
# ---------------------------------------------------------------------------

def kgrid_frac(kgrid):
    """``(N_k, 3)`` C-order grid points in [0, 1)."""
    ii = np.stack(np.meshgrid(*(np.arange(n) for n in kgrid), indexing="ij"), -1).reshape(-1, 3)
    return ii / np.asarray(kgrid, dtype=np.float64)


def spheres(frac, metric, ecut, *, span=3, width=None):
    """``|k+G|²_metric ≤ ecut`` for each row of ``frac``: ``(gvecs (n, width, 3), ngk)``.

    Slots in (|k+G|², G) order; pad rows hold a Miller index far outside every sphere."""
    rng = np.arange(-span, span + 1)
    G = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
    rows = []
    for k in np.asarray(frac, float):
        kg = k[None, :] + G
        e = np.einsum("gi,ij,gj->g", kg, metric, kg)
        sel = np.flatnonzero(e <= ecut + 1e-9)
        order = np.lexsort((G[sel, 2], G[sel, 1], G[sel, 0], np.round(e[sel], 9)))
        rows.append(G[sel[order]])
    w = max(len(r) for r in rows) if width is None else int(width)
    out = np.full((len(rows), w, 3), 99, np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out, np.asarray([len(r) for r in rows], np.int64)


def random_green(rng, n, width, ngk, ns, *, garbage=0.0):
    """``(n, width, ns, width, ns)`` random tiles; pad slots hold ``garbage``·random."""
    A = rng.standard_normal((n, width, ns, width, ns)) + 1j * rng.standard_normal((n, width, ns, width, ns))
    live = np.arange(width)[None, :] < np.asarray(ngk)[:, None]
    mask = live[:, :, None, None, None] & live[:, None, None, :, None]
    return np.where(mask, A, garbage * A)


def green_from_psi(c, weights):
    """``Σ_n c_n(p α) w_n conj c_n(p' β)`` for ``c (n, nb, ns, width)``: ``(n, width, ns, width, ns)``."""
    return np.einsum("knap,n,knbq->kpaqb", c, weights, np.conj(c))


def pad_tiles(A, carrier):
    """Zero-pad the two slot axes of ``(n, w, ns, w, ns)`` to ``carrier``."""
    w = A.shape[1]
    return np.pad(A, ((0, 0), (0, carrier - w), (0, 0), (0, carrier - w), (0, 0)))


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------

def _grid(fft_grid):
    fg = np.asarray(fft_grid)
    ii = np.stack(np.meshgrid(*(np.arange(n) for n in fg), indexing="ij"), -1).reshape(-1, 3)
    return ii / fg


def _bloch_kernel(A, gvecs, ngk, frac, fft_grid):
    """a_k(r α, r' β) = Σ_pp' e^{i(k+p)·r} A_k e^{-i(k+p')·r'}: ``(n, ns, ns, N_r, N_r)``."""
    r = _grid(fft_grid)
    n, ns = A.shape[0], A.shape[2]
    out = np.empty((n, ns, ns, r.shape[0], r.shape[0]), np.complex128)
    for k in range(n):
        g = gvecs[k, :ngk[k]]
        E = np.exp(2j * np.pi * r @ (frac[k][None, :] + g).T)          # (N_r, ngk)
        for a in range(ns):
            for b in range(ns):
                out[k, a, b] = E @ A[k, :ngk[k], a, :ngk[k], b] @ E.conj().T
    return out


def _output(Xq_rr, out_gvecs, out_ngk, out_frac, fft_grid):
    """X_q(G, G') = Σ_{r,r'} e^{-i(q+G)·r} X̂_q(r, r') e^{i(q+G')·r'}: ``(n_q, width, width)``."""
    r = _grid(fft_grid)
    nq, w = out_gvecs.shape[0], out_gvecs.shape[1]
    X = np.zeros((nq, w, w), np.complex128)
    for i in range(nq):
        g = out_gvecs[i, :out_ngk[i]]
        O = np.exp(-2j * np.pi * (out_frac[i][None, :] + g) @ r.T)      # (ngk, N_r)
        X[i, :out_ngk[i], :out_ngk[i]] = O @ Xq_rr[i] @ O.conj().T
    return X


def dense_reference(A, C, sphere_gvecs, sphere_ngk, kfrac, kgrid, fft_grid,
                    out_gvecs, out_ngk, out_frac):
    """The definition on the materialized supercell (small cases only)."""
    nk = int(np.prod(kgrid))
    a = _bloch_kernel(A, sphere_gvecs, sphere_ngk, kfrac, fft_grid)
    c = _bloch_kernel(C, sphere_gvecs, sphere_ngk, kfrac, fft_grid)
    R = kgrid_frac(kgrid) * np.asarray(kgrid)                            # integer lattice vectors
    ph = np.exp(2j * np.pi * kfrac @ R.T)                               # (k, R)
    AR = np.einsum("kR,kabrs->Rabrs", ph, a) / nk
    CR = np.einsum("kR,kabrs->Rabrs", ph, c) / nk
    XR = np.einsum("Rabrs,Rabrs->Rrs", AR, np.conj(CR))
    qph = np.exp(-2j * np.pi * out_frac @ R.T)                          # (q, R)
    Xq = np.einsum("qR,Rrs->qrs", qph, XR)
    return _output(Xq, out_gvecs, out_ngk, out_frac, fft_grid)


def ksum_reference(A, C, sphere_gvecs, sphere_ngk, kfrac, kgrid, fft_grid,
                   out_gvecs, out_ngk, out_frac):
    """The direct k-sum convolution X̂_q = N_k⁻¹ Σ_k a_k conj(c_{k-q}), same output matrices."""
    kg = np.asarray(kgrid)
    nk = int(np.prod(kg))
    a = _bloch_kernel(A, sphere_gvecs, sphere_ngk, kfrac, fft_grid)
    c = _bloch_kernel(C, sphere_gvecs, sphere_ngk, kfrac, fft_grid)
    kint = np.rint(kfrac * kg).astype(int) % kg
    Xq = []
    for q in out_frac:
        qi = np.rint(q * kg).astype(int)
        kp = np.ravel_multi_index(((kint - qi) % kg).T, tuple(kg))
        Xq.append(np.einsum("kabrs,kabrs->rs", a, np.conj(c[kp])) / nk)
    return _output(np.asarray(Xq), out_gvecs, out_ngk, out_frac, fft_grid)


def dense_scalar_reference(A, a_gvecs, a_ngk, a_frac, B, b_gvecs, b_ngk, b_frac, kgrid, fft_grid,
                           out_gvecs, out_ngk, out_frac):
    """X_k(p α, p' β) = Σ_R Σ_{r,r'} e^{-i(k+p)·(r+R)} A^{αβ}(r+R, r') B(r+R, r') e^{i(k+p')·r'}
    (B one spin channel) on the materialized supercell: ``(n_k, width, ns, width, ns)``."""
    nk = int(np.prod(kgrid))
    a = _bloch_kernel(A, a_gvecs, a_ngk, a_frac, fft_grid)
    b = _bloch_kernel(B, b_gvecs, b_ngk, b_frac, fft_grid)[:, 0, 0]
    R = kgrid_frac(kgrid) * np.asarray(kgrid)
    AR = np.einsum("kR,kabrs->Rabrs", np.exp(2j * np.pi * a_frac @ R.T), a) / nk
    BR = np.einsum("kR,krs->Rrs", np.exp(2j * np.pi * b_frac @ R.T), b) / nk
    Xq = np.einsum("qR,Rabrs->qabrs", np.exp(-2j * np.pi * out_frac @ R.T), AR * BR[:, None, None])
    ns, w = A.shape[2], out_gvecs.shape[1]
    X = np.zeros((len(out_frac), w, ns, w, ns), np.complex128)
    for i in range(ns):
        for j in range(ns):
            X[:, :, i, :, j] = _output(Xq[:, i, j], out_gvecs, out_ngk, out_frac, fft_grid)
    return X


def _density(c1, g1, n1, k1, c2, g2, n2, k2, G, q):
    """ρ(G) = Σ_{p'',α} c1(p''+G+g₀, α) conj c2(p'', α), g₀ = q + k2 − k1 (integer): the plane-wave
    coefficient of ψ1 ψ2* at q + G (``c (nb, ns, width)``); returns ``(nb1, nb2, len(G))``."""
    g0 = np.rint(np.asarray(q) + np.asarray(k2) - np.asarray(k1)).astype(np.int64)
    look = {tuple(v): i for i, v in enumerate(g1[:n1])}
    out = np.zeros((c1.shape[0], c2.shape[0], len(G)), np.complex128)
    for gi, Gv in enumerate(G):
        pairs = [(look[t], i2) for i2, v in enumerate(g2[:n2])
                 if (t := tuple(v + Gv + g0)) in look]
        if pairs:
            i1, i2 = (np.asarray(v) for v in zip(*pairs))
            out[:, :, gi] = np.einsum("nap,map->nm", c1[:, :, i1], np.conj(c2[:, :, i2]))
    return out


def band_sum_sigma(c, weights, gvecs, ngk, kfrac, W, w_gvecs, w_ngk, w_frac, kgrid, k_rows):
    """Σ_nn'(k) = −N_k⁻¹ Σ_{q,m} g_m Σ_{GG'} ρ_nm(G) W_q(G, G') conj ρ_n'm(G'),
    ρ_nm(G) = ⟨nk| e^{i(q+G)·r} |m, k−q⟩, with ψ = Σ_p c(p) e^{i(k+p)·r} (Ω = 1):
    ``(len(k_rows), nb, nb)``."""
    kg = np.asarray(kgrid)
    nk = int(np.prod(kg))
    kint = np.rint(kfrac * kg).astype(int) % kg
    qint = np.rint(w_frac * kg).astype(int) % kg
    out = []
    for k in k_rows:
        S = np.zeros((c.shape[1], c.shape[1]), np.complex128)
        for q in range(nk):
            kp = int(np.ravel_multi_index((kint[k] - qint[q]) % kg, tuple(kg)))
            G = w_gvecs[q, :w_ngk[q]]
            rho = np.conj(_density(c[k], gvecs[k], ngk[k], kfrac[k], c[kp], gvecs[kp], ngk[kp],
                                   kfrac[kp], G, w_frac[q]))                # ρ_nm(G), (n, m, G)
            Wq = W[q, :w_ngk[q], 0, :w_ngk[q], 0]
            S += np.einsum("nmg,gh,Nmh,m->nN", rho, Wq, np.conj(rho), weights)
        out.append(-S / nk)
    return np.asarray(out)


def band_sum_chi(c, wc, wv, gvecs, ngk, kfrac, kgrid, out_gvecs, out_ngk, out_frac):
    """BGW's χ_q(G, G') = N_k⁻¹ Σ_{k,n,m} wc_n wv_m ρ_nm(G) conj ρ_nm(G'), ρ_nm = ψ_nk ψ*_{m,k−q} at
    q + G (Ω = 1), the χ₀ of A = Σ_n c wc c†, C = Σ_m c wv c†: ``(n_q, width, width)``."""
    kg = np.asarray(kgrid)
    nk = int(np.prod(kg))
    kint = np.rint(kfrac * kg).astype(int) % kg
    X = np.zeros((len(out_frac), out_gvecs.shape[1], out_gvecs.shape[1]), np.complex128)
    for i, q in enumerate(out_frac):
        qi = np.rint(q * kg).astype(int)
        G = out_gvecs[i, :out_ngk[i]]
        for k in range(nk):
            kp = int(np.ravel_multi_index((kint[k] - qi) % kg, tuple(kg)))
            rho = _density(c[k], gvecs[k], ngk[k], kfrac[k], c[kp], gvecs[kp], ngk[kp], kfrac[kp], G, q)
            X[i, :out_ngk[i], :out_ngk[i]] += np.einsum("nmg,n,m,nmh->gh", rho, wc, wv, np.conj(rho))
    return X / nk


def unfold_tile(transport, S_par, anti_conj=True):
    """The transport's definition on one-channel tiles (numpy, the docstring's formula):
    ``O_k[p, p'] = mph_k(p) · 𝒯 S_{row(k)}[src_k(p), src_k(p')] · nph_k(p')`` with 𝒯 = conj on
    an antiunitary row; ``S_par (n_par, w, w)``; returns ``(N_k, w, w)`` (pad slots zero)."""
    nk, w = transport.src.shape
    out = np.zeros((nk, w, w), np.complex128)
    for k in range(nk):
        live = np.flatnonzero(transport.phase[k] != 0)
        src, ph = transport.src[k, live], transport.phase[k, live]
        blk = S_par[transport.row[k]][np.ix_(src, src)]
        if transport.anti[k] and anti_conj:
            blk, ph = np.conj(blk), np.conj(ph)
        out[k][np.ix_(live, live)] = ph[:, None] * blk * np.conj(ph)[None, :]
    return out


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------------------
# symmetry: parent ψ → children by the r-space typed action
# ---------------------------------------------------------------------------

def children_psi(fx, c_par, par_gvecs, par_ngk, child_gvecs, child_ngk):
    """Children's sphere coefficients from parent coefficients through the r-space typed action.

    ``c_par (n_par, nb, ns, width)``; the parents' Bloch functions are sampled on
    the fixture grid, moved by ``zeta_mubatch_fixtures._children`` and projected
    onto each child's sphere.  Returns ``(c_child (N_k, nb, ns, width), leak)``,
    ``leak`` the largest relative norm outside the child sphere (≈ 0 iff the grid
    and the spheres are closed under the group)."""
    import zeta_mubatch_fixtures as fixtures
    plan, fg = fx["plan"], tuple(fx["fft_grid"])
    r = _grid(fg)
    kpar = np.asarray(plan.k_parent_frac)
    kful = np.asarray(fx["kfull"])
    n_par, nb, ns, w = c_par.shape
    psi = np.zeros((n_par, nb, ns, r.shape[0]), np.complex128)
    for p in range(n_par):
        g = par_gvecs[p, :par_ngk[p]]
        E = np.exp(2j * np.pi * r @ (kpar[p][None, :] + g).T)
        psi[p] = np.einsum("rg,nsg->nsr", E, c_par[p, :, :, :par_ngk[p]])
    psi_k = fixtures._children(dict(fx, psi_parent=psi))                 # (N_k, nb, ns, N_r)
    out = np.zeros((kful.shape[0], nb, ns, w), np.complex128)
    leak = 0.0
    for k in range(kful.shape[0]):
        u = psi_k[k] * np.exp(-2j * np.pi * r @ kful[k])[None, None, :]
        cf = np.fft.fftn(u.reshape(nb, ns, *fg), axes=(-3, -2, -1)).reshape(nb, ns, -1) / r.shape[0]
        g = child_gvecs[k, :child_ngk[k]] % np.asarray(fg)
        flat = (g[:, 0] * fg[1] + g[:, 1]) * fg[2] + g[:, 2]
        out[k, :, :, :child_ngk[k]] = cf[:, :, flat]
        off = np.ones(cf.shape[-1], bool)
        off[flat] = False
        leak = max(leak, float(np.sqrt(np.sum(np.abs(cf[:, :, off]) ** 2) / np.sum(np.abs(cf) ** 2))))
    return out, leak


# ---------------------------------------------------------------------------
# covariant operands: parent band sets closed under their little groups
# ---------------------------------------------------------------------------

def little_group(rows, ops, n_sym, k):
    """The rows of ``rows`` that fix ``k``: ``±S^T k ≡ k`` (mod 1), − on an antiunitary row."""
    out = []
    for r in np.asarray(rows).tolist():
        S = np.asarray(ops[r % n_sym], dtype=np.float64)
        img = (-1.0 if r >= n_sym else 1.0) * (S.T @ np.asarray(k, float))
        d = img - np.asarray(k, float)
        if np.max(np.abs(d - np.rint(d))) < 1e-8:
            out.append(int(r))
    return out


def close_under_little_groups(fx, c_par, par_gvecs, par_ngk, *, rows, spinor_action, ns):
    """Each parent's bands and their images under every row of its little group, by the
    r-space typed action (``children_psi`` with the parent as its own child): the parent
    Green of these bands is little-group invariant, so its typed unfold is covariant.

    ``c_par (n_par, nb, ns, width)``; returns ``(c_closed (n_par, nb·n_max, ns, width), leak)``
    with zero bands padding the parents whose little group is smaller than the largest."""
    from types import SimpleNamespace
    plan = fx["plan"]
    kpar = np.asarray(plan.k_parent_frac)
    n_sym = int(plan.n_sym_spatial)
    n_par, nb, _, w = c_par.shape
    groups = [little_group(rows, fx["ops"], n_sym, kpar[p]) for p in range(n_par)]
    n_max = max(len(g) for g in groups)
    out = np.zeros((n_par, nb * n_max, ns, w), np.complex128)
    leak = 0.0
    for p, g in enumerate(groups):
        U = (np.ones((len(g), 1, 1), np.complex128) if ns == 1
             else np.asarray(spinor_action(np.asarray(g, np.int32), nspinor=ns)))
        sub = SimpleNamespace(n_full=len(g), irr_idx=np.full(len(g), p, np.int32),
                              sym_idx=np.asarray(g, np.int32), k_parent_frac=kpar,
                              n_sym_spatial=n_sym, spin_action_full=U)
        img, lk = children_psi(dict(fx, plan=sub, kfull=np.repeat(kpar[p:p + 1], len(g), axis=0)),
                               c_par, par_gvecs, par_ngk,
                               np.repeat(par_gvecs[p:p + 1], len(g), axis=0),
                               np.repeat(np.asarray(par_ngk)[p:p + 1], len(g)))
        leak = max(leak, lk)
        out[p, :nb * len(g)] = img.reshape(len(g) * nb, ns, w)
    return out, leak
