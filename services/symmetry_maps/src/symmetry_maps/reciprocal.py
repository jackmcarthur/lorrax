"""Reciprocal-space tables and local spin action for parent pair operators.

The caller provides the same typed centroid plan used in real space. This
module owns its Fourier image, including magnetic antiunitary operations.
"""
import numpy as np
import jax.numpy as jnp


def typed_child_G_tables(plan, *, fft_grid, sphere_par, gvec_child,
                         ngk_child, k_child):
    """The r-space typed ψ action (:func:`symmetry_maps.unfold_wavefunction_local`)
    as G-space tables, its exact Fourier image.

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


def _pair_spinor_action(U, d):
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



def unfold_reciprocal_pair_local(d, *, centroid_perm, centroid_wraps,
                                 k_parent, g_slots, g_phase, anti, spin):
    """Apply the typed reciprocal action to a local pair-operator slab.

    ``d`` has spin axes 0 and 3 in ``(s, x, mu, s, G)``. Whole centroid
    orbits are local; G slots carry the scalar wavefunction translation
    phase. Centroid and G phases precede magnetic conjugation, then both
    spin indices transform by U and its conjugate. No group is inferred.
    """
    wl = jnp.exp(2j * jnp.pi * (centroid_wraps.astype(jnp.float64) @ k_parent))
    d = jnp.take(d, centroid_perm, axis=2) * wl[None, None, :, None, None]
    d = jnp.take(d, g_slots, axis=-1) * jnp.conj(g_phase)
    d = jnp.where(anti, jnp.conj(d), d)
    return _pair_spinor_action(spin, d)
