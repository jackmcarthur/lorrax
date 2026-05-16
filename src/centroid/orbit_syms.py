"""Real-space symmetry helpers for orbit-aware k-means.

Bridges between LORRAX's ``SymMaps`` (which stores rotations and the
BGW-convention fractional translations) and the row-vector fractional
convention used by the kmeans kernels.

Conventions
-----------
For a row-vector fractional position ``r``:

    image_row(r, s) = r @ Rinv[s].T  +  tau[s]                 (mod 1)
    fold_back(δ, s) = δ @ R[s].T

* ``R[s]``   = ``sym.R_grid[s]``      — acts on G-vectors (col convention)
* ``Rinv[s]`` = ``sym.Rinv_grid[s]``  — acts on real-space r (col convention)
* ``tau[s]`` = ``wfn.translations[s] / (2π)`` — BGW sign already baked in.

This is the same convention used by ``sym.validate_atomic_symmetries`` and
``charge_density._symmetrise_density``; see ``symmetry_maps.py:339-345``
for the canonical example.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────
# Build sym table (host)
# ─────────────────────────────────────────────────────────────────────────

def build_real_space_syms(wfn, sym, validate: bool = True):
    """Spatial-only sym data for r-space orbit construction.

    Parameters
    ----------
    wfn, sym : LORRAX WFNReader / SymMaps.
    validate : if True, run ``sym.validate_atomic_symmetries(wfn)`` and
        raise if it returns failures.

    Returns
    -------
    R, Rinv : (n_sym, 3, 3) int32 jax arrays.
    tau     : (n_sym, 3) float64 jax array, BGW sign convention.
    """
    if validate:
        failures = sym.validate_atomic_symmetries(wfn)
        if failures:
            raise RuntimeError(
                f"sym data fails atomic-orbit closure: {failures[:3]}"
            )
    n_sym = int(wfn.ntran)
    R = jnp.asarray(np.asarray(sym.R_grid[:n_sym], dtype=np.int32))
    Rinv = jnp.asarray(np.asarray(sym.Rinv_grid[:n_sym], dtype=np.int32))
    tau = jnp.asarray(
        np.asarray(wfn.translations[:n_sym], dtype=np.float64) / (2.0 * np.pi)
    )
    return R, Rinv, tau


# ─────────────────────────────────────────────────────────────────────────
# Orbit utilities
# ─────────────────────────────────────────────────────────────────────────

@jax.jit
def orbit_images(reps: jnp.ndarray,
                 Rinv: jnp.ndarray,
                 tau: jnp.ndarray) -> jnp.ndarray:
    """Apply BGW r-action ``r' = Rinv·r + τ`` (mod 1) to every rep.

    BGW's space-group action on real-space points is
    ``r' = mtrx⁻¹ · r + τ`` where ``mtrx = wfn.sym_matrices`` and
    ``τ = wfn.translations / (2π)``.  Pass ``Rinv = inv(mtrx)`` (=
    ``sym.Rinv_grid``).  Verified by ``validate_atomic_symmetries`` on
    Si Fd-3m: 96/96 atom mappings pass with this convention,
    48/96 with the wrong-direction ``mtrx · r + τ`` action.

    For symmorphic systems (CrI3, MoS2: τ=0) the choice of matrix
    direction is moot — both produce the same orbit set because the
    group is closed under inversion.  For non-symmorphic systems
    (Si Fd-3m glides), only the BGW convention closes the orbit
    properly.
    """
    return jax.vmap(lambda Ri, t: (reps @ Ri.T + t) % 1.0)(Rinv, tau)


_CANON_INV = jnp.int64(10**12)
"""Integer precision for canonicalisation lex keys: 12 digits → 1 ppm
resolution on fractional coords ([0, 1)). Coarser than fp64 noise (1e-15)
by ~3 orders of magnitude — enough to absorb single-op drift without
collapsing distinct orbit members."""


def _orbit_lex_winner(images: jnp.ndarray) -> jnp.ndarray:
    """For each rep, the index s* of the lex-smallest orbit image.

    images: (n_sym, n_rep, 3) fp64 in [0, 1).
    Returns: (n_rep,) int32 — index in [0, n_sym).

    Uses integer lex ordering (`jnp.lexsort` on int64 triples) instead of a
    floating composite key. Robust against fp noise; deterministic.
    """
    keys = jnp.round(images * _CANON_INV).astype(jnp.int64)  # (n_sym, n_rep, 3)
    # lexsort: rightmost key is primary. We want primary = x, secondary = y,
    # tertiary = z, so pass (z, y, x). Result is sorted indices along axis 0.
    n_rep = images.shape[1]
    def winner_for_rep(i):
        order = jnp.lexsort((keys[:, i, 2], keys[:, i, 1], keys[:, i, 0]))
        return order[0].astype(jnp.int32)
    return jax.vmap(winner_for_rep)(jnp.arange(n_rep))


@jax.jit
def canonicalize_orbit(reps: jnp.ndarray,
                       Rinv: jnp.ndarray,
                       tau: jnp.ndarray) -> jnp.ndarray:
    """Map each rep to the lex-smallest member of its orbit (integer-key
    lex). Idempotent. Static shape (n_rep, 3)."""
    images = orbit_images(reps, Rinv, tau)                 # (n_sym, n_rep, 3)
    best_s = _orbit_lex_winner(images)                      # (n_rep,)
    return jnp.take_along_axis(
        images, best_s[None, :, None], axis=0
    )[0]


def unfold_orbit_unique_with_id(reps_np: np.ndarray,
                                Rinv: np.ndarray,
                                tau: np.ndarray,
                                tol: float = 1e-6,
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Unfold reps into all distinct orbit images; also return ``orbit_id``
    — a per-candidate integer that's the same for two candidates iff they
    lie in the same **physical** orbit under the WFN's sym group.

    Pass ``Rinv = inv(wfn.sym_matrices) = sym.Rinv_grid``.  BGW's
    r-action is ``r' = Rinv · r + τ``; this matches the direction used
    by ``orbit_images``, ``compute_centroid_sym_perm``, and
    ``validate_atomic_symmetries``.

    orbit_id is the integer encoding of each candidate's canonical
    (lex-smallest) orbit member, then run through ``np.unique`` to make
    them dense [0, n_orbits). This is the right key for the orbit-aware
    pivoted Cholesky: two kmeans reps that drifted into the same physical
    orbit during Lloyd get identical orbit_ids, so PC sees them as one
    orbit (not two).
    """
    Rinv = np.asarray(Rinv); tau = np.asarray(tau)
    inv = np.int64(round(1.0 / tol))         # int! avoid fp64-precision loss at 1e18

    # 1. Unfold + dedupe at fp tolerance.
    # ``r @ Rinv[s].T + τ[s]`` = Rinv·r + τ in column form (BGW r-action).
    images = (np.einsum('ri,sji->srj', reps_np, Rinv) + tau[:, None, :]) % 1.0
    flat = images.reshape(-1, 3)
    keys = np.round(flat * inv).astype(np.int64) % inv
    _, first_idx = np.unique(keys, axis=0, return_index=True)
    flat = flat[np.sort(first_idx)]

    # 2. For each unique candidate, compute its canonical (lex-min) orbit
    #    member, then dense-encode the canonical triples as orbit_id.
    cand_imgs = (np.einsum('ci,sji->scj', flat, Rinv) + tau[:, None, :]) % 1.0
    cand_keys = np.round(cand_imgs * inv).astype(np.int64) % inv     # (n_sym, n, 3)
    # Lex via np.lexsort on (z, y, x) — primary key first in argument order
    # is leftmost; lexsort treats the LAST key as primary. So pass (z, y, x)
    # so x is primary, y secondary, z tertiary. Returns sorted indices along axis 0.
    n = flat.shape[0]
    s_idx = np.array([
        np.lexsort((cand_keys[:, i, 2], cand_keys[:, i, 1], cand_keys[:, i, 0]))[0]
        for i in range(n)
    ])
    canonical_keys = cand_keys[s_idx, np.arange(n)]                  # (n, 3)
    _, orbit_id = np.unique(canonical_keys, axis=0, return_inverse=True)
    return flat, orbit_id.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────
# Centroid orbit permutation π_s : r_{π_s(μ)} = S_s r_μ + τ_s  (mod 1)
# ─────────────────────────────────────────────────────────────────────────

def compute_centroid_sym_perm(
    r_mu_fft_idx: np.ndarray,
    sym_matrices: np.ndarray,
    translations: np.ndarray,
    fft_grid: np.ndarray | tuple[int, int, int],
    *,
    validate: bool = True,
    extend_trs: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (α_s, L_s) for the BGW-convention r-action ``r' = inv(mtrx)·r + τ``.

    For each target centroid μ and sym op s, compute the **source**
    centroid α(μ) and integer real-space lattice wrap L_μ such that

        y_μ = mtrx · (x_μ − τ) = x_{α(μ)} + L_μ,     L_μ ∈ ℤ³.

    Here ``mtrx = sym_matrices[s]``, ``x_μ = centroid frac coords``,
    ``τ = translations[s] / (2π)``.  ``mtrx`` is BGW's stored matrix
    (acts on G-vectors; BGW's r-action is ``r' = mtrx⁻¹·r + τ``).
    Confirmed by ``validate_atomic_symmetries``: 96/96 Si atom mappings
    pass with this convention; the wrong direction ``mtrx·r + τ`` gives
    48/96 and breaks Si Fd-3m closure (see
    ``reports/trs_sym_audit_2026-05-14/SYMMETRY_CONVENTIONS.md``).

    The user-math formula for the V_q unfold then reads:

        V_full[q1, μ', ν'] = exp(2π i q · (L_{μ'} − L_{ν'}))
                             · V_ibz[parent, α(μ'), α(ν')]

    where q = IBZ parent q in fractional reciprocal coords.  ``unfold_v_q``
    consumes ``sym_perm = α`` and ``L_table = L`` directly (no argsort).
    For symmorphic systems (τ=0), both α and its inverse give the same
    orbit set (group closed under inversion), so the choice of "α" vs
    "π" direction is irrelevant; for non-symmorphic systems (Si Fd-3m
    glides) the α direction is required for the orbit to close.

    Parameters
    ----------
    r_mu_fft_idx
        (n_rmu, 3) int32 — centroid positions as integer FFT-grid
        indices ``[0, FFTgrid[a])``.
    sym_matrices
        (n_sym, 3, 3) int — BGW ``mtrx``.  These act on G-vectors
        (column convention); for r-space we use the inverse.  We
        compute ``Rinv = inv(S)`` here so callers can pass either
        ``wfn.sym_matrices[:ntran]`` or a pre-sliced ``R_grid``.
    translations
        (n_sym, 3) float — BGW ``tnp``.  Fractional translation is
        ``τ_frac = translations / (2π)``.
    fft_grid
        (3,) int — FFT grid extents.  Centroid positions must be
        commensurate with this grid AND with the τ × fft_grid product
        (otherwise the rounded image won't land on a grid point —
        that's the orbit-closure failure mode).
    validate
        If True, asserts every row of the result is a permutation
        ``[0, n_rmu)``.  Set False only for offline diagnostics where
        the closure failure is the thing you want to inspect.
    extend_trs
        If True, return a ``(2·n_sym, n_rmu)`` table whose rows
        ``[n_sym:]`` duplicate rows ``[:n_sym]``.  This is the
        TRS-augmented variant: under time-reversal symmetry, real-space
        centroid coordinates r_μ are unchanged (TRS acts on momenta and
        complex-conjugates ψ, but leaves r fixed), so the permutation
        for a TRS-augmented op ``K ∘ {S | τ}`` coincides with the
        permutation for the bare spatial op ``{S | τ}``.  Pass
        ``extend_trs=True`` whenever the caller's ``full_to_irr_sym``
        values may exceed ``n_sym`` (i.e. come from
        ``SymMaps.irr_idx_q / sym_idx_q`` which uses the
        TRS-augmented ``sym_mats_k``).  See Agent 1's scope report at
        ``reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md``
        Site #1 for the bug this option closes.

    Returns
    -------
    sym_perm
        ``(n_sym, n_rmu)`` int32 by default; ``(2·n_sym, n_rmu)`` int32
        when ``extend_trs=True``.  ``sym_perm[s, μ] = ν`` iff
        ``r_ν ≡ S_s r_μ + τ_s`` on the FFT grid.  For
        ``s ∈ [n_sym, 2·n_sym)`` the TRS-augmented row duplicates
        ``s - n_sym`` (TRS keeps r fixed).  The ζ-leg complex
        conjugation under TRS — for V_q bilinear in ζ this becomes
        ``V_{TRS-q, μ, ν} = conj(V_{q, μ, ν}) = V_{q, ν, μ}`` (last
        equality by V_q Hermiticity) — is applied at the V_q-unfold
        level (see ``gw.v_q_tile._unfold_v_q_ibz_to_full``), NOT in
        ``sym_perm`` itself.
    L_table
        ``(n_sym, n_rmu, 3)`` int8 by default; ``(2·n_sym, n_rmu, 3)``
        when ``extend_trs=True``.  ``L_table[s, μ] = floor(S r_μ + τ)``
        — the integer real-space lattice vector by which the image
        exits the unit cell.  TRS rows duplicate spatial rows (r is
        fixed under TRS).  Used by ``unfold_v_q`` to build the
        umklapp phase ``exp(2π i q · (L_μ − L_ν))``.

    Raises
    ------
    RuntimeError
        If orbit closure fails: i.e. for some (s, μ) the image of r_μ
        under {S_s | τ_s} doesn't land on any other centroid in the
        table.  Caller should regenerate the centroid set with
        ``kmeans_cli`` in orbit-aware mode, or pass identity-only sym.
    """
    fft_grid_np = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    idx = np.asarray(r_mu_fft_idx, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(
            f"r_mu_fft_idx must be (n_rmu, 3); got {idx.shape}")
    n_rmu = int(idx.shape[0])

    S = np.asarray(sym_matrices, dtype=np.int64)
    if S.ndim != 3 or S.shape[1:] != (3, 3):
        raise ValueError(
            f"sym_matrices must be (n_sym, 3, 3); got {S.shape}")
    n_sym = int(S.shape[0])

    tau_frac = (np.asarray(translations, dtype=np.float64)[:n_sym]
                / (2.0 * np.pi))                              # (n_sym, 3)

    # Convert centroid FFT indices to fractional coords for the
    # transformation, then back to FFT indices after wrap.
    r_frac = idx.astype(np.float64) / fft_grid_np[None, :]    # (n_rmu, 3)

    # User-spec source-centroid + lattice-wrap decomposition:
    #   y_μ = mtrx · (x_μ − τ) = x_{α(μ)} + L_μ, with L_μ ∈ ℤ³.
    # BGW r-action ``r' = mtrx⁻¹ · r + τ`` ⇒ the SOURCE of x_μ under this
    # action is mtrx·(x_μ−τ); the integer part is the lattice wrap that
    # produces the umklapp phase exp(2π i q · L_μ) on ζ; the fractional
    # part mod 1 is the centroid index α(μ) we permute by.  See
    # SYMMETRY_CONVENTIONS.md.
    r_shifted = r_frac[None, :, :] - tau_frac[:, None, :]            # (n_sym, n_rmu, 3)
    # images_raw[s, μ, i] = (mtrx[s] · r_shifted[s, μ])_i = sum_j S[s,i,j] r_shifted[s,μ,j]
    images_raw = np.einsum('sij,srj->sri', S.astype(np.float64), r_shifted)
    # Snap to FFT-grid integers BEFORE floor.  Centroids live at
    # multiples of 1/fft_grid; mtrx and τ are commensurate (BGW guarantee),
    # so images_raw is also a multiple of 1/fft_grid up to 1e-17
    # floating-point noise.  Naive ``np.floor`` flips an L component
    # from 0 → -1 whenever the true integer part is 0 but a tiny
    # negative noise hits np.floor's discontinuity — which produces a
    # spurious exp(±iπ/2) phase in unfold_v_q.  Snapping fixes this
    # cleanly; verified at ISDF noise floor on Si Fd-3m (24³ FFT,
    # non-symmorphic τ) where the previous code gave 14/64 q's with
    # rel err ~0.8 due to this exact off-by-one.
    images_int = np.rint(images_raw * fft_grid_np[None, None, :]).astype(np.int64)
    grid_per_axis = fft_grid_np[None, None, :]
    L_wrap = (np.floor_divide(images_int, grid_per_axis)).astype(np.int8)
    images_int_mod = images_int - L_wrap.astype(np.int64) * grid_per_axis
    images = images_int_mod.astype(np.float64) / grid_per_axis.astype(np.float64)

    # Snap back to FFT-grid integers.  If τ × FFTgrid isn't integer to
    # roundoff, this rounding will land on a half-grid point and the
    # subsequent dict lookup will fail — that's the right error signal.
    img_idx = np.rint(images * fft_grid_np[None, None, :]).astype(np.int64)
    img_idx = img_idx % fft_grid_np[None, None, :]              # wrap residual

    # Build a fast lookup from FFT-grid triple → centroid index.
    radix1 = fft_grid_np[1] * fft_grid_np[2]
    radix2 = fft_grid_np[2]
    def _flat(idx_arr):
        return idx_arr[..., 0] * radix1 + idx_arr[..., 1] * radix2 \
               + idx_arr[..., 2]
    cent_flat = _flat(idx)                                       # (n_rmu,)
    img_flat = _flat(img_idx)                                    # (n_sym, n_rmu)

    flat_to_mu = -np.ones(int(fft_grid_np.prod()), dtype=np.int64)
    flat_to_mu[cent_flat] = np.arange(n_rmu, dtype=np.int64)

    sym_perm = flat_to_mu[img_flat]                              # (n_sym, n_rmu)

    if validate:
        bad = (sym_perm < 0)
        if bad.any():
            bad_s, bad_mu = np.where(bad)
            ex_s, ex_mu = int(bad_s[0]), int(bad_mu[0])
            ex_idx = img_idx[ex_s, ex_mu].tolist()
            raise RuntimeError(
                f"compute_centroid_sym_perm: centroid orbit closure "
                f"failed.  sym {ex_s} maps centroid μ={ex_mu} "
                f"(at fft_idx {idx[ex_mu].tolist()}) to fft_idx "
                f"{ex_idx}, which is NOT in the centroid table.  "
                f"Total failures: {int(bad.sum())} / {n_sym * n_rmu}.  "
                f"Regenerate centroids with orbit-aware kmeans or fall "
                f"back to identity-only sym."
            )
        # Each row should be a permutation.  Cheap O(n_sym · n_rmu) check.
        for s in range(n_sym):
            if np.unique(sym_perm[s]).size != n_rmu:
                raise RuntimeError(
                    f"compute_centroid_sym_perm: sym_perm[{s}] is not a "
                    f"permutation — two distinct centroids map to the "
                    f"same image under sym {s}.  Likely cause: τ × "
                    f"fft_grid is not integer, so the rounded image "
                    f"collides with a different centroid.  Check "
                    f"``validate_atomic_symmetries`` on the WFN.")

    sym_perm = sym_perm.astype(np.int32)
    if extend_trs:
        # TRS keeps r fixed; the augmented rows duplicate the spatial
        # rows for BOTH sym_perm and L_wrap.  Doubling here makes the
        # tables index-compatible with the TRS-augmented
        # ``full_to_irr_sym`` values returned by ``SymMaps.irr_idx_q /
        # sym_idx_q`` (which range over ``[0, 2·ntran)``).  Without
        # this, a downstream gather of ``inv_perm[s]`` for ``s ≥
        # ntran`` silently clips to the last spatial row under JAX
        # ``mode='promise_in_bounds'``, producing wrong V_q at every
        # TRS-folded q (the headline bug — see
        # ``reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md``).
        sym_perm = np.concatenate([sym_perm, sym_perm.copy()], axis=0)
        L_wrap = np.concatenate([L_wrap, L_wrap.copy()], axis=0)
    return sym_perm, L_wrap


# ─────────────────────────────────────────────────────────────────────────
# Full FFT-grid orbit permutation — used by ZetaLoader's q='full_bz' unfold.
# ─────────────────────────────────────────────────────────────────────────

def compute_rgrid_sym_perm(
    sym_matrices: np.ndarray,
    translations: np.ndarray,
    fft_grid: np.ndarray | tuple[int, int, int],
    *,
    validate: bool = True,
) -> np.ndarray:
    """Build the per-sym r-grid permutation table over the FULL FFT grid.

    Returns ``sym_perm[s, r_new] = r_old`` such that
    ``r_{r_new} ≡ S_s · r_{r_old} + τ_s`` on the FFT grid, where ``r_*``
    indexes the flat FFT grid in C-order
    (``r_flat = i_x * ny * nz + i_y * nz + i_z``).  This is the gather
    table the caller wants when expanding an IBZ-q ζ tensor onto the
    full BZ: ``ζ_full[q, r_new, μ] = ζ_ibz[i(q), sym_perm[s(q), r_new],
    π_{s(q)}^{-1}(μ)]``.

    Math
    ----
    Same convention as :func:`compute_centroid_sym_perm`: BGW's
    ``mtrx`` acts on G-vectors; real-space r transforms by
    ``Rinv = inv(S)``.  We compute the forward image
    ``r' = Rinv · r + τ`` for every grid point, snap back to the FFT
    grid, and invert the resulting permutation so the output is a
    pull-back (``sym_perm[s, r_new] = r_old``) suitable for
    ``take_along_axis(zeta, sym_perm[s, :], axis=r)``.

    Failure mode
    ------------
    The orbit-closure assertion is automatic for the full FFT grid IF
    ``τ × fft_grid`` is integer (i.e. the fractional translation lands
    on a discrete grid point).  When it doesn't, the rounding step here
    will collide images — that's the right error signal.  In practice
    QE outputs satisfy this commensurability; if a future workflow
    breaks it, the loader will refuse rather than silently corrupt.

    Parameters
    ----------
    sym_matrices, translations, fft_grid
        Same meaning as :func:`compute_centroid_sym_perm`.  ``translations``
        is BGW's ``tnp`` (``τ_frac = tnp / (2π)``).
    validate
        If True, asserts every row of the result is a permutation of
        ``[0, n_rtot)``.

    Returns
    -------
    sym_perm
        ``(n_sym, n_rtot) int32`` — gather indices along the flat-r axis.
    """
    fg = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
    n_rtot = nx * ny * nz

    S = np.asarray(sym_matrices, dtype=np.int64)
    if S.ndim != 3 or S.shape[1:] != (3, 3):
        raise ValueError(
            f"sym_matrices must be (n_sym, 3, 3); got {S.shape}")
    n_sym = int(S.shape[0])
    tau_frac = (np.asarray(translations, dtype=np.float64)[:n_sym]
                / (2.0 * np.pi))                                # (n_sym, 3)

    # Enumerate every grid point as (i_x, i_y, i_z) in flat C-order.
    ix, iy, iz = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    r_idx = np.stack([ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)],
                       axis=1).astype(np.int64)                  # (n_rtot, 3)
    r_frac = r_idx.astype(np.float64) / fg[None, :]              # (n_rtot, 3)

    # Real-space transform uses Rinv = inv(S).
    Rinv = np.rint(np.linalg.inv(S)).astype(np.int64)            # (n_sym, 3, 3)

    # images[s, r] = r @ Rinv[s].T + τ[s]  (mod 1)
    images = (np.einsum('rj,sij->sri', r_frac, Rinv.astype(np.float64))
              + tau_frac[:, None, :])
    images = images - np.floor(images)                            # in [0, 1)

    img_idx = np.rint(images * fg[None, None, :]).astype(np.int64)
    img_idx = img_idx % fg[None, None, :]                         # wrap residual

    radix1 = ny * nz
    radix2 = nz
    img_flat = (img_idx[..., 0] * radix1
                + img_idx[..., 1] * radix2
                + img_idx[..., 2])                                # (n_sym, n_rtot)

    # ``img_flat[s, r_old]`` is the destination index ``r_new`` of
    # the forward image: ``r_{r_new} ≡ S_s · r_{r_old} + τ_s``.
    # Invert to get the gather table: ``sym_perm[s, r_new] = r_old``.
    sym_perm = np.empty_like(img_flat)
    base = np.arange(n_rtot, dtype=np.int64)
    for s in range(n_sym):
        sym_perm[s, img_flat[s]] = base

    if validate:
        for s in range(n_sym):
            if np.unique(sym_perm[s]).size != n_rtot:
                raise RuntimeError(
                    f"compute_rgrid_sym_perm: sym_perm[{s}] is not a "
                    f"permutation of [0, n_rtot={n_rtot}).  Likely cause: "
                    f"τ × fft_grid is not integer for sym {s} "
                    f"(τ_frac={tau_frac[s].tolist()}, "
                    f"τ×fft_grid={(tau_frac[s] * fg).tolist()}).  Off-grid "
                    f"fractional translations are not yet supported by the "
                    f"ζ full-BZ unfold path."
                )

    return sym_perm.astype(np.int32)


