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

import dataclasses
import hashlib

import numpy as np
import jax
import jax.numpy as jnp

from ._compat import deprecated_alias


# ─────────────────────────────────────────────────────────────────────────
# Charge-density point group (recovery of symmetry a reduced WFN dropped)
# ─────────────────────────────────────────────────────────────────────────

def recover_symmorphic_density_point_group(
    avec: np.ndarray,
    charge_density: np.ndarray,
    *,
    tol_rho: float = 1e-3,
) -> np.ndarray:
    """Symmorphic (τ=0) point group that leaves the charge density invariant.

    Why this exists — the V_H C3-symmetry fix
    -----------------------------------------
    ⟨nk|V_H|nk⟩ is built as an ISDF **centroid quadrature** (``gw/
    cohsex_sigma.py:hartree``: ``Σ_{μν} ρ_{mn}(k,r_μ)·V_q[0](μ,ν)·ρ(r_ν)``).
    A raw centroid sum is only point-group symmetric across a k-star if the
    centroid set ``{r_μ}`` is closed under the point group.  Because V_H is
    a large matrix element (~500 eV — the full electron-electron Hartree,
    ~30× Σ_x), the centroid-placement-dependent quadrature error is
    amplified into a multi-eV C3 split of ⟨nk|V_H|nk⟩ across equivalent k,
    which corrupts the QP dispersion (Vxc = E_dft − kin_ion − V_H).

    Orbit-closing the centroids fixes it — but the orbit must be closed
    under the **crystal** point group, and the WFN's stored symmetry list
    can be *smaller* than that.  A non-collinear SOC MoS₂ WFN, for example,
    stores only ``ntran=2`` = {E, σ_h} even though the crystal — and the
    charge density — are fully D3h symmetric.

    (CORRECTED 2026-08-04, owner.  This comment previously blamed pw2bgw for
    dropping the C3 rotations "because the BGW ``mtrx`` format can't carry
    their spinor rotation".  That is wrong on both counts: BGW carries spinor
    rotations, and pw2bgw does not drop the ops.  A short stored list comes
    from the symmetries allowed in the QE input that produced the deck, so
    the fix is upstream in that input, not here.  This routine remains useful
    for a deck whose stored list is already short — it recovers the group
    from the density rather than from the file — but it is a recovery path,
    not a workaround for a format limitation.)
    Closing centroids under the stored {E, σ_h} leaves V_H C3-broken.

    The physically-correct group to close under is the one that leaves the
    ground-state **charge density** invariant: for a non-magnetic crystal
    that is the full crystal point group; for a magnetically-ordered
    crystal it is the (smaller) magnetic point group.  Testing invariance
    of the QE-symmetrized ρ(r) directly recovers the former without
    over-closing the latter — a bare atom-position test would wrongly add
    the ops broken by magnetic order.

    Method
    ------
    1. Enumerate the holohedry: integer matrices ``M`` (entries in
       {−1,0,1}, valid for a primitive-cell fractional basis of any of the
       7 crystal systems) with ``Mᵀ G M = G`` where ``G = avec·avecᵀ`` is
       the real-space metric.  These are the r-action (BGW ``Rinv``)
       matrices of every metric-preserving point operation.
    2. Keep the ``M`` under which the density grid is invariant:
       ``ρ[M·n mod N] ≈ ρ[n]`` for every FFT-grid index ``n`` (M maps the
       grid to itself because it is integer).  Non-symmorphic ops (τ≠0)
       fail this τ=0 test and are conservatively omitted — safe, because a
       WFN that genuinely carries non-symmorphic ops already exposes them
       (see :func:`real_space_action_tables`, which keeps the larger group).

    Parameters
    ----------
    avec : (3, 3) real
        Real-space lattice vectors as rows (any length unit — only the
        metric shape matters).
    charge_density : (Nx, Ny, Nz) real
        Point-group-symmetrized ground-state density on the FFT grid
        (e.g. QE's ``charge-density.hdf5``).
    tol_rho : float
        Relative tolerance ``max|ρ[Mn]−ρ[n]| / max|ρ|`` for calling an op
        a density symmetry.

    Returns
    -------
    Rinv : (n_op, 3, 3) int32
        The r-action matrices of the recovered symmorphic density point
        group (τ = 0).  Always contains the identity; closed under the
        group operation by construction.
    """
    A = np.asarray(avec, dtype=np.float64)
    G = A @ A.T
    # Scale-relative tolerance: ``avec`` carries ~1e-7 float roundoff (e.g.
    # √3/2 ≈ 0.8660253) that propagates into G, so an exact-integer test on
    # Mᵀ G M is too brittle.  A true point op reproduces G to that roundoff
    # (~1e-6·|G|); a non-op differs by O(|G|).  1e-4·max|G| cleanly separates
    # them for any lattice (alat-normalized or Bohr; hexagonal off-diagonal
    # ±0.5 vs a wrong op's O(1) mismatch).
    gtol = 1e-4 * float(np.max(np.abs(G)))

    # 1. Holohedry: integer M with Mᵀ G M = G.
    import itertools
    holo = []
    for entries in itertools.product((-1, 0, 1), repeat=9):
        M = np.array(entries, dtype=np.int64).reshape(3, 3)
        if abs(round(np.linalg.det(M))) != 1:
            continue
        if np.max(np.abs(M.T @ G @ M - G)) <= gtol:
            holo.append(M)

    # 2. Density-invariance filter on the FFT grid.
    rho = np.asarray(charge_density, dtype=np.float64)
    N = np.asarray(rho.shape, dtype=np.int64)
    scale = float(np.max(np.abs(rho))) or 1.0
    ix, iy, iz = np.meshgrid(np.arange(N[0]), np.arange(N[1]), np.arange(N[2]),
                             indexing="ij")
    n_idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)  # (P,3)
    rho_flat = rho.ravel()
    keep = []
    for M in holo:
        img = (n_idx @ M.T) % N[None, :]                # r' = M·n  (mod grid)
        img_flat = img[:, 0] * (N[1] * N[2]) + img[:, 1] * N[2] + img[:, 2]
        if np.max(np.abs(rho_flat[img_flat] - rho_flat)) <= tol_rho * scale:
            keep.append(M)
    if not keep:                                        # identity always works
        keep = [np.eye(3, dtype=np.int64)]
    return np.asarray(keep, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────
# Build sym table (host)
# ─────────────────────────────────────────────────────────────────────────

def real_space_action_tables(wfn, sym, validate: bool = True, *,
                             charge_density=None):
    """Spatial-only sym data for r-space orbit construction.

    Parameters
    ----------
    wfn, sym : LORRAX WFNReader / SymMaps.
    validate : if True, run ``sym.validate_atomic_symmetries(wfn)`` and
        raise if it returns failures.
    charge_density : (Nx, Ny, Nz) real, optional
        The k-means weighting density.  When supplied, the symmorphic
        point group that leaves it invariant is recovered
        (:func:`recover_symmorphic_density_point_group`); if that group
        strictly contains the WFN's stored group, it is used for centroid
        orbit closure instead.  This recovers point-group symmetry a
        reduced WFN (e.g. non-collinear SOC, which stores only {E, σ_h})
        dropped, so ⟨nk|V_H|nk⟩ becomes C3-symmetric across the k-star.
        The downstream ζ / V_q IBZ cascade continues to use the WFN's
        stored group; only the centroid *set* is closed under the larger
        (density) group — which is all a scalar quadrature needs.

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
    Rinv_wfn = np.asarray(sym.Rinv_grid[:n_sym], dtype=np.int32)
    tau_wfn = np.asarray(wfn.translations[:n_sym], dtype=np.float64) / (2.0 * np.pi)

    if charge_density is not None:
        Rinv_rho = recover_symmorphic_density_point_group(
            np.asarray(wfn.avec), charge_density)
        wfn_symmorphic = np.allclose(tau_wfn, 0.0, atol=1e-6)
        rho_set = {M.tobytes() for M in Rinv_rho}
        wfn_in_rho = all(M.tobytes() in rho_set for M in Rinv_wfn)
        # Only adopt the density group when it is a strict superset of the
        # WFN group (i.e. the WFN symmetry is reduced).  Requiring the WFN
        # ops ⊆ density group AND symmorphic guards against ever *losing* a
        # non-symmorphic op the WFN carries but the τ=0 detector can't see.
        if wfn_symmorphic and wfn_in_rho and Rinv_rho.shape[0] > n_sym:
            print(f"  [orbit] WFN stores {n_sym} sym op(s); recovered "
                  f"{Rinv_rho.shape[0]}-op symmorphic point group from the "
                  f"charge density — closing centroids under the larger group "
                  f"(fixes V_H k-star symmetry).")
            Rinv = Rinv_rho
            R = np.rint(np.linalg.inv(Rinv)).astype(np.int32)
            tau = np.zeros((Rinv.shape[0], 3), dtype=np.float64)
            return (jnp.asarray(R), jnp.asarray(Rinv), jnp.asarray(tau))

    R = jnp.asarray(np.asarray(sym.R_grid[:n_sym], dtype=np.int32))
    Rinv = jnp.asarray(Rinv_wfn)
    tau = jnp.asarray(tau_wfn)
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
    by ``orbit_images``, ``centroid_source_map_and_wrap``, and
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

def centroid_source_map_and_wrap(
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

    where q = IBZ parent q in fractional reciprocal coords.
    ``unfold_isdf_operator`` consumes ``sym_perm = α`` and
    ``L_table = L`` directly (no argsort).
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
        level (see ``symmetry_maps.unfold_isdf_operator``), NOT in
        ``sym_perm`` itself.
    L_table
        ``(n_sym, n_rmu, 3)`` int8 by default; ``(2·n_sym, n_rmu, 3)``
        when ``extend_trs=True``.  ``L_table[s, μ] = floor(S r_μ + τ)``
        — the integer real-space lattice vector by which the image
        exits the unit cell.  TRS rows duplicate spatial rows (r is
        fixed under TRS).  Used by ``unfold_isdf_operator`` to build the
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
    # spurious exp(±iπ/2) phase in unfold_isdf_operator.  Snapping fixes this
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
                f"centroid_source_map_and_wrap: centroid orbit closure "
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
                    f"centroid_source_map_and_wrap: sym_perm[{s}] is not a "
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
# Orbit closure, as a MEASUREMENT you can hold — the public door diagnostic
# ─────────────────────────────────────────────────────────────────────────

#: Closure tolerance, in fractional coordinates, and why this number.
#:
#: The committed centroid files are written to SIX DECIMALS, so a set that
#: is exactly closed still measures a residual of ~1e-6 — that is the text
#: file's rounding floor, not a defect (MEASURED on
#: ``si_cohsex_debug/centroids_frac_144.txt``: worst 1.000e-06 over all 48
#: ops).  A set that is genuinely NOT closed misses by a fraction of a
#: cell: 1.318e-01 on the 960 set and 1.718e-01 on the 480 set, which on
#: the 24³ grid those centroids live on is ~3-4 grid steps.  1e-5 sits an
#: order of magnitude above the rounding floor and four orders below the
#: real failures, so no plausible file lands in the gap.
CLOSURE_TOL_DEFAULT = 1.0e-5

#: Decimals used to build the exact-match lookup key.  Matches the
#: precision the centroid files are written at; images that hit a key are
#: scored against that one centroid instead of against all of them.
_CLOSURE_KEY_DECIMALS = 6

#: Image rows scored per pairwise block.  Bounds the temporary at
#: ``_CLOSURE_BLOCK · n_centroids · 3 · 8`` bytes (~70 MB at 3000
#: centroids) instead of letting the (n_sym, n_mu, n_mu, 3) tensor exist.
_CLOSURE_BLOCK = 1024

#: How far off a grid point a coordinate may sit, IN GRID STEPS, before
#: ``fft_grid`` is called a mismatch rather than rounding.  The committed
#: centroid files carry six decimals, so ``0.083333 × 24 = 1.999992`` —
#: MEASURED 8.000e-06 of a step on ``centroids_frac_960.txt``, which is the
#: file's own precision and not a defect.  A genuine grid mismatch (τ on
#: the wrong denominator, centroids from another FFT box) misses by an
#: appreciable fraction of a step, and anything past 0.5 collides with a
#: different grid point under ``np.rint``.  1e-2 is three orders above the
#: rounding and fifty times below the collision.
_GRID_COMMENSURATE_TOL = 1.0e-2


@dataclasses.dataclass(frozen=True)
class CentroidClosureVerdict:
    """What :func:`verify_centroid_orbit_closure` measured, and its verdict.

    A record, not a wrapper: it carries numbers a caller would otherwise
    recompute, in the same spirit as :class:`DensitySymmetryReport` — the
    service's other measurement-with-a-verdict.  The point of holding the
    per-op residual vector rather than a single bool is that the failure
    mode this guards against is *partial*: 47 of 48 ops violating with the
    identity clean is a completely different diagnosis from one op
    violating, and a bool cannot tell them apart.

    Attributes
    ----------
    closed
        ``True`` iff every image of every centroid under every op lands on
        a centroid within ``tol``.  THE PREREQUISITE for IBZ storage: if
        this is False the permutation α does not exist and the tensor
        cannot be unfolded at all (spec §2).
    tol
        The fractional-coordinate tolerance the verdict was taken at.
    n_sym, n_centroids
        Table extents the measurement ran over.
    worst_residual
        Largest per-image residual over all (op, centroid) pairs, in
        fractional coordinates, under :attr:`metric`.
    worst_op
        The op index attaining :attr:`worst_residual`.
    violating_ops
        EVERY op index whose worst residual exceeds ``tol``, ascending.
    residual_by_op
        ``(n_sym,)`` float64 — each op's worst residual.  Read-only.
    centroid_hash
        Strong hash of the centroid set, prefixed ``g:`` when it was taken
        over integer FFT-grid indices and ``f:`` when over fractional
        coordinates.  The prefix exists so a hash taken under one rule can
        never compare equal to one taken under the other.
    metric
        Names the residual: the minimum-image Euclidean distance in
        fractional coordinates.  Recorded on the verdict because it is the
        thing a future reader would otherwise have to guess when comparing
        a stored number against a fresh one.
    """

    closed: bool
    tol: float
    n_sym: int
    n_centroids: int
    worst_residual: float
    worst_op: int
    violating_ops: tuple[int, ...]
    residual_by_op: np.ndarray
    centroid_hash: str
    metric: str = "min_image_euclidean_frac"

    @property
    def n_violating(self) -> int:
        return len(self.violating_ops)

    def describe(self) -> str:
        """The loud form: every violating op index with its residual.

        Spec §6 gate 1 asks the refusal to name "the offending op index and
        residual" — plural, because the production failure is 47 ops, and a
        message that named only the first would read as a single bad op.
        """
        head = (
            f"centroid orbit closure: "
            f"{'CLOSED' if self.closed else 'NOT CLOSED'} — "
            f"{self.n_violating}/{self.n_sym} ops violating at tol="
            f"{self.tol:.1e}, worst {self.worst_residual:.3e} on op "
            f"{self.worst_op} ({self.n_centroids} centroids, metric "
            f"{self.metric}, centroids {self.centroid_hash})")
        if self.closed:
            return head
        rows, cur = [], []
        for s in self.violating_ops:
            cur.append(f"s={s}:{float(self.residual_by_op[s]):.3e}")
            if len(cur) == 6:
                rows.append("  " + "  ".join(cur))
                cur = []
        if cur:
            rows.append("  " + "  ".join(cur))
        return "\n".join([head, "  offending ops (index:worst residual):"]
                         + rows)

    def as_attr(self) -> str:
        """One line, for an HDF5 attr.  Stable enough to compare on."""
        return (f"{'closed' if self.closed else 'not_closed'} "
                f"tol={self.tol:.3e} worst={self.worst_residual:.6e} "
                f"worst_op={self.worst_op} "
                f"violating={self.n_violating}/{self.n_sym} "
                f"metric={self.metric}")

    def raise_if_not_closed(self, context: str = "") -> None:
        """REFUSE on a non-closed set.  Never warn and continue.

        A q_irr file written against a non-closed centroid set is silently
        unrecoverable — there is no α to invert with — so this is the one
        place the branch is taken, and it raises.
        """
        if self.closed:
            return
        where = f"{context}: " if context else ""
        raise RuntimeError(where + self.describe())

    def __str__(self) -> str:                      # pragma: no cover - repr
        return self.describe()


def _centroid_hash(frac: np.ndarray, fft_grid: np.ndarray | None) -> str:
    """sha256 of the centroid set, canonicalised so it is reproducible.

    On the grid when a grid is given (integers — exactly reproducible from
    any float representation that snaps to the same points), on the wrapped
    fractional coordinates rounded to :data:`_CLOSURE_KEY_DECIMALS`
    otherwise.  The two rules produce differently-PREFIXED digests on
    purpose: a ``g:`` hash and an ``f:`` hash of the same point set are not
    interchangeable and must never silently compare equal.
    """
    h = hashlib.sha256()
    if fft_grid is not None:
        grid = np.asarray(fft_grid, dtype=np.int64).reshape(3)
        idx = np.rint(np.asarray(frac, dtype=np.float64) * grid[None, :])
        idx = (idx.astype(np.int64) % grid[None, :])
        h.update(np.ascontiguousarray(idx, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(grid, dtype=np.int64).tobytes())
        return "g:" + h.hexdigest()
    can = np.round(np.asarray(frac, dtype=np.float64) % 1.0,
                   _CLOSURE_KEY_DECIMALS) % 1.0
    h.update(np.ascontiguousarray(can, dtype=np.float64).tobytes())
    return "f:" + h.hexdigest()


def _min_image_residual(images: np.ndarray, cent: np.ndarray) -> np.ndarray:
    """Per-image distance to the NEAREST centroid, minimum-image, in frac.

    Exact-key fast path first (a closed set hits every key and never pays
    for a distance matrix), chunked pairwise for the misses only.
    """
    n_img = int(images.shape[0])
    out = np.zeros(n_img, dtype=np.float64)
    if n_img == 0:
        return out

    def _key(a):
        k = np.rint((np.asarray(a) % 1.0) * 10 ** _CLOSURE_KEY_DECIMALS)
        return k.astype(np.int64) % (10 ** _CLOSURE_KEY_DECIMALS)

    ckey = _key(cent)
    lut = {}
    for i in range(int(cent.shape[0])):
        lut.setdefault(tuple(ckey[i].tolist()), i)
    ikey = _key(images)
    hit_to = np.full(n_img, -1, dtype=np.int64)
    for i in range(n_img):
        j = lut.get(tuple(ikey[i].tolist()), -1)
        hit_to[i] = j
    hit = hit_to >= 0
    if hit.any():
        d = images[hit] - cent[hit_to[hit]]
        d -= np.rint(d)
        out[hit] = np.sqrt((d * d).sum(axis=-1))
    miss = np.flatnonzero(~hit)
    for beg in range(0, miss.size, _CLOSURE_BLOCK):
        blk = miss[beg:beg + _CLOSURE_BLOCK]
        d = images[blk][:, None, :] - cent[None, :, :]
        d -= np.rint(d)
        out[blk] = np.sqrt((d * d).sum(axis=-1)).min(axis=1)
    return out


def verify_centroid_orbit_closure(
    centroids_frac,
    sym_matrices,
    *,
    tnp=None,
    tau=None,
    fft_grid=None,
    tol: float = CLOSURE_TOL_DEFAULT,
) -> CentroidClosureVerdict:
    """Is the centroid set closed under the space group?  MEASURE and say.

    Reconstructing ``W(Sq)`` from ``W(q)`` in the ISDF basis is a
    permutation of the (μ, ν) indices, and that permutation exists only if
    every symmetry maps the centroid set into itself.  This is the door
    diagnostic for that prerequisite: it asks, for every op ``s`` and every
    centroid ``x_μ``,

        y_μ = mtrx_s · (x_μ − τ_s)   (mod 1)

    — the same source-map decomposition :func:`centroid_source_map_and_wrap`
    builds α from — and reports how far ``y_μ`` lands from the nearest
    member of the set.  :func:`centroid_source_map_and_wrap` REFUSES on the
    same condition; this function is the form you can hold, compare and
    stamp into a file without catching an exception to learn the answer.

    THE 2π.  ``tnp`` in the WFN is ``2π·τ``.  Dividing by 2π is the whole
    difference between "every set looks unclosed, including the ones that
    are fine" and the truth, and it cost the author of
    ``SPEC_qirr_restart_tensors.md`` an hour and a wrong first answer.
    **This function is the one place in the service where that division
    lives**, and the reason it cannot go wrong here is that there is no
    positional slot for the translations at all: you must name which
    convention you are holding, ``tnp=`` (BGW's stored ``2π·τ``, which is
    what ``wfn.translations`` and ``mf_header/symmetry/tnp`` are) or
    ``tau=`` (already divided).  Exactly one, and passing neither or both
    is a ``ValueError``.  A caller who guesses gets a refusal, never a
    plausible wrong verdict.

    There is one extra guard, and it is asymmetric because only one
    direction is detectable: fractional translations are in [0, 1), so a
    ``tau=`` whose components exceed 1 is a ``tnp`` wearing the wrong
    keyword and is refused by name.  The reverse — a ``tnp=`` that is
    really a τ — cannot be detected, because a symmorphic deck has
    ``tnp = 0 = τ`` and every value in between is legal.

    Parameters
    ----------
    centroids_frac
        ``(n_mu, 3)`` float — centroid positions in fractional
        coordinates.  Wrapped to [0, 1) internally; the caller's array is
        never modified.
    sym_matrices
        ``(n_sym, 3, 3)`` int — BGW ``mtrx``, the same array
        :func:`centroid_source_map_and_wrap` takes.  Only the first
        ``n_sym`` rows that ``tnp``/``tau`` covers are used, so a
        TRS-augmented ``sym_mats_k`` may be passed as long as the
        translations match it row for row.
    tnp
        ``(n_sym, 3)`` float — BGW's stored ``2π·τ``.  Mutually exclusive
        with ``tau``.
    tau
        ``(n_sym, 3)`` float — fractional translations, already divided by
        2π.  Mutually exclusive with ``tnp``.
    fft_grid
        ``(3,)`` int, optional.  Does NOT change the residual: the metric
        stays the minimum-image Euclidean distance in fractional
        coordinates so the number is comparable with or without a grid.
        What it changes is (a) the centroid hash, which becomes the exact
        integer-index digest, and (b) an extra commensurability refusal —
        centroids that do not sit on the grid, or a τ that does not, are
        the mechanism by which a set "fails closure" for a reason that has
        nothing to do with the group.
    tol
        Fractional-coordinate tolerance; see :data:`CLOSURE_TOL_DEFAULT`
        for why the default is 1e-5 and not something tighter.

    Returns
    -------
    CentroidClosureVerdict
        Carries the verdict, every violating op index, the per-op worst
        residual and the centroid hash.  It does not raise on a non-closed
        set — call :meth:`CentroidClosureVerdict.raise_if_not_closed` for
        that, which is what the q_irr writer does.

    Examples
    --------
    MEASURED 2026-08-07 on the committed fixtures (24³ FFT grid, 48 ops)::

        si_cohsex_debug/centroids_frac_960.txt  worst 1.318e-01  47/48
        si_cohsex_debug/centroids_frac_144.txt  worst 1.000e-06   0/48
        si_bse_debug/centroids_frac_480.txt     worst 1.718e-01  47/48

    reproducing the table in ``SPEC_qirr_restart_tensors.md`` §2 exactly.
    """
    if (tnp is None) == (tau is None):
        raise ValueError(
            "verify_centroid_orbit_closure: pass exactly one of tnp= or "
            "tau=.  ``tnp`` is BGW's stored 2π·τ (mf_header/symmetry/tnp, "
            "wfn.translations); ``tau`` is the fractional translation "
            "itself.  There is no positional slot for either, on purpose: "
            "dividing by 2π at the wrong moment makes every centroid set "
            "look unclosed, including the ones that are fine.")
    if tnp is not None:
        tau_frac = np.asarray(tnp, dtype=np.float64) / (2.0 * np.pi)
    else:
        tau_frac = np.asarray(tau, dtype=np.float64)
        if tau_frac.size and float(np.abs(tau_frac).max()) > 1.0 + 1e-9:
            raise ValueError(
                f"verify_centroid_orbit_closure: tau= has a component of "
                f"magnitude {float(np.abs(tau_frac).max()):.6f}, but a "
                f"fractional translation lives in [0, 1).  These look like "
                f"BGW ``tnp`` = 2π·τ — pass them as tnp= and let this "
                f"function do the division.")
    if tau_frac.ndim != 2 or tau_frac.shape[1] != 3:
        raise ValueError(
            f"verify_centroid_orbit_closure: translations must be "
            f"(n_sym, 3); got {tau_frac.shape}")

    cent = np.asarray(centroids_frac, dtype=np.float64)
    if cent.ndim != 2 or cent.shape[1] != 3:
        raise ValueError(
            f"verify_centroid_orbit_closure: centroids_frac must be "
            f"(n_mu, 3); got {cent.shape}")
    cent = cent % 1.0
    n_mu = int(cent.shape[0])
    if n_mu == 0:
        raise ValueError(
            "verify_centroid_orbit_closure: empty centroid set; there is "
            "nothing for the group to act on.")

    S = np.asarray(sym_matrices, dtype=np.float64)
    if S.ndim != 3 or S.shape[1:] != (3, 3):
        raise ValueError(
            f"verify_centroid_orbit_closure: sym_matrices must be "
            f"(n_sym, 3, 3); got {S.shape}")
    n_sym = int(tau_frac.shape[0])
    if int(S.shape[0]) < n_sym:
        raise ValueError(
            f"verify_centroid_orbit_closure: {int(S.shape[0])} sym "
            f"matrices but {n_sym} translation rows; they index the same "
            f"op list and must agree.")
    S = S[:n_sym]

    grid = None
    if fft_grid is not None:
        grid = np.asarray(fft_grid, dtype=np.int64).reshape(3)
        if np.any(grid <= 0):
            raise ValueError(
                f"verify_centroid_orbit_closure: fft_grid must be "
                f"positive; got {grid.tolist()}")
        # Commensurability.  A centroid or a τ that is off-grid produces a
        # closure failure that says nothing about the GROUP — the rounded
        # image simply cannot land on a grid point.  Separate the two
        # diagnoses here rather than letting the residual carry both.
        for label, arr in (("centroids_frac", cent),
                           ("tau (= tnp/2π)", tau_frac % 1.0)):
            scaled = arr * grid[None, :]
            off = float(np.abs(scaled - np.rint(scaled)).max())
            if off > _GRID_COMMENSURATE_TOL:
                raise ValueError(
                    f"verify_centroid_orbit_closure: {label} is not "
                    f"commensurate with fft_grid {grid.tolist()} — worst "
                    f"off-grid offset {off:.3e} of a grid step.  That is a "
                    f"grid mismatch, not a closure verdict; fix the grid "
                    f"or omit fft_grid to score in pure fractional "
                    f"coordinates.")

    # y_μ = mtrx · (x_μ − τ)  (mod 1) — the SOURCE map, matching
    # centroid_source_map_and_wrap's decomposition y_μ = x_{α(μ)} + L_μ.
    residual_by_op = np.zeros(n_sym, dtype=np.float64)
    for s in range(n_sym):
        shifted = cent - tau_frac[s][None, :]
        images = (shifted @ S[s].T) % 1.0
        residual_by_op[s] = float(_min_image_residual(images, cent).max())

    residual_by_op.setflags(write=False)
    violating = tuple(int(s) for s in np.flatnonzero(residual_by_op > tol))
    worst_op = int(np.argmax(residual_by_op))
    return CentroidClosureVerdict(
        closed=(len(violating) == 0),
        tol=float(tol),
        n_sym=n_sym,
        n_centroids=n_mu,
        worst_residual=float(residual_by_op[worst_op]),
        worst_op=worst_op,
        violating_ops=violating,
        residual_by_op=residual_by_op,
        centroid_hash=_centroid_hash(cent, grid),
    )


# ─────────────────────────────────────────────────────────────────────────
# The ONE resolution point — verdict, mode, tables, reason, in one object
# ─────────────────────────────────────────────────────────────────────────

#: The consequence sentence, written once.  Every announcement of a
#: fallback carries this exact clause so that a log grep for it finds
#: every degraded run, whatever site resolved it.
FULL_BZ_CONSEQUENCE = (
    "q-grid symmetry reduction disabled; solving on the full BZ — "
    "restart tensors stay full-BZ; see verify_centroid_orbit_closure")


@dataclasses.dataclass(frozen=True)
class QgridSymmetryResolution:
    """What the q-grid reduction RESOLVED to, and why — one object.

    Before this existed, every consumer of the centroid permutation
    tables called :func:`centroid_source_map_and_wrap` inside its own
    ``try``/``except RuntimeError`` and, on the refusal, set its own
    private flag back to full-BZ.  Four sites, four spellings, and the
    only trace of the decision on a production run was a line one of them
    printed when ``verbose`` happened to be true — which at the W solve it
    is not.  That is the silent degradation the owner named: the run is
    ~8× slower and its restart tensors are 8× larger than the design
    intends, and nothing says so.

    This record replaces the exception-as-control-flow with an answer.
    :func:`resolve_qgrid_symmetry` takes the closure verdict ONCE, decides
    the mode from it, builds the tables when the mode allows them, and
    hands back everything a caller could want to know — including the
    sentence it should print.  Callers branch on :attr:`use_ibz`; nobody
    catches anything.

    Attributes
    ----------
    mode
        ``"ibz"`` or ``"full_bz"``.  The resolved mode, not a request.
    verdict
        The :class:`CentroidClosureVerdict` the decision was made on —
        always present, including on the ``"ibz"`` path, because the
        numbers are what a stamp or a report wants and recomputing them
        is how two answers to one question get born.
    sym_perm, L_table
        The tables, or ``None`` when ``mode == "full_bz"``.  Shapes are
        exactly what :func:`centroid_source_map_and_wrap` returns for the
        ``extend_trs`` that was asked for.
    reason
        Empty on the ``"ibz"`` path.  On ``"full_bz"``, one line naming
        WHY — either the closure verdict's own summary or the table
        builder's refusal (see :func:`resolve_qgrid_symmetry` for the
        second, rarer arm).
    n_sym_spatial
        The spatial op count the tables were built over, so a consumer
        that must interpret a TRS-augmented row index does not have to
        re-derive it from a shape.
    context
        Free text naming the call site, e.g. ``"V_q / W q-grid
        reduction"``.  It appears in the announcement, because "the
        centroid set is not closed" is a different operational fact
        depending on which centroid set is meant.
    """

    mode: str
    verdict: CentroidClosureVerdict
    sym_perm: np.ndarray | None
    L_table: np.ndarray | None
    reason: str
    n_sym_spatial: int
    context: str = ""

    @property
    def use_ibz(self) -> bool:
        """The one predicate callers branch on."""
        return self.mode == "ibz"

    @property
    def announce_key(self) -> tuple:
        """Dedup key for a once-per-run announcement.

        Keyed on the centroid set, not on the call site: the V_q pass,
        the W Dyson solve and every SC iteration resolve the SAME set and
        must speak once between them.  A bispinor deck carries two
        genuinely different sets (charge and transverse) whose closure
        can differ, and those are two different facts, so they get two
        keys and two lines.
        """
        return ("qgrid_symmetry_fallback", self.verdict.centroid_hash)

    def tables(self) -> tuple[np.ndarray, np.ndarray]:
        """``(sym_perm, L_table)``, or a refusal naming the mode.

        The accessor exists so that a caller which has already branched
        wrong gets an error that says which decision it ignored, instead
        of a ``TypeError`` on ``None`` several frames downstream.
        """
        if not self.use_ibz:
            raise RuntimeError(
                f"QgridSymmetryResolution.tables(): mode is "
                f"{self.mode!r}, so there are no unfold tables — the "
                f"centroid set does not admit them.  {self.reason}")
        return self.sym_perm, self.L_table

    def announcement(self) -> str | None:
        """The loud line, or ``None`` when nothing degraded.

        Returns the text; it does NOT print.  The service has no rank and
        no once-per-run memory — both belong to the process that runs the
        deck — so the announcement is composed here (where the numbers
        are) and emitted there (where rank 0 is).  The monorepo side of
        that seam is ``gw.qgrid_symmetry.resolve_qgrid_symmetry_tables``,
        which hands this string to ``ffi.gate.announce_once`` under
        :attr:`announce_key`.
        """
        if self.use_ibz:
            return None
        where = self.context or "q-grid symmetry reduction"
        return (
            f"\n  *** LORRAX q-grid symmetry: FALLBACK at {where} — the "
            f"centroid set is not orbit-closed.\n"
            f"      why:         {self.reason}\n"
            f"      consequence: {FULL_BZ_CONSEQUENCE}.\n"
            f"      fix:         regenerate the centroid set with the "
            f"orbit-aware k-means mode (centroid.kmeans_cli); the "
            f"per-op\n"
            f"                   residuals are on the verdict this "
            f"line was taken from. ***\n")

    def __str__(self) -> str:                      # pragma: no cover - repr
        tail = "" if self.use_ibz else f" — {self.reason}"
        return (f"q-grid symmetry: mode={self.mode} "
                f"(n_sym_spatial={self.n_sym_spatial}){tail}")


def resolve_qgrid_symmetry(
    r_mu_fft_idx,
    sym_matrices,
    *,
    tnp=None,
    tau=None,
    fft_grid,
    extend_trs: bool = True,
    tol: float = CLOSURE_TOL_DEFAULT,
    context: str = "",
) -> QgridSymmetryResolution:
    """Resolve the q-grid reduction ONCE: verdict → mode → tables.

    THE DEFECT THIS CLOSES.  ``centroid_source_map_and_wrap`` refuses on a
    non-closed centroid set, correctly — but a refusal is only as loud as
    the ``except`` that receives it, and in this tree the receivers each
    degraded the whole run to the full BZ and said nothing a production
    log would show.  The q_irr work makes that fallback expensive
    (restart tensors stay 8× larger) and the Σ star spread makes it
    inaccurate (16.9 meV against 0.7 on a closed set), so it stops being
    a quiet default and becomes an announced, single-sited decision.

    Callers do not catch anything.  They call this, read
    :attr:`~QgridSymmetryResolution.use_ibz`, and — if they are the
    process that owns rank 0 — emit
    :meth:`~QgridSymmetryResolution.announcement` once.

    TWO WAYS TO LAND ON ``full_bz``, and they are different diagnoses:

    1. The verdict says the set is not closed.  This is the production
       case on the 960- and 480-centroid decks (47 of 48 ops violating),
       and the reason carries the worst op and its residual.
    2. The verdict says CLOSED and ``centroid_source_map_and_wrap`` still
       refuses.  The closure measurement scores the minimum-image
       distance in fractional coordinates; the table builder additionally
       needs the image to land on THIS FFT grid and needs each row to be
       a bijection.  A τ that is not commensurate with the grid, or two
       centroids that collide under rounding, fail the second without
       failing the first.  That refusal is caught HERE — the one place in
       the tree that catches it — and reported as its own reason, because
       "your centroids are not closed" would be a wrong diagnosis for it.

    Anything else raises: a bad shape or a mis-keyed translation is a
    programming error, and resolving it to a slower-but-running mode is
    how a convention bug survives to production.

    Parameters
    ----------
    r_mu_fft_idx
        ``(n_rmu, 3)`` int — centroid positions as integer FFT-grid
        indices, exactly what :func:`centroid_source_map_and_wrap` takes.
    sym_matrices
        ``(n_sym, 3, 3)`` int — BGW ``mtrx``, spatial ops only.
    tnp, tau
        The translations, under :func:`verify_centroid_orbit_closure`'s
        exclusive-keyword contract: ``tnp`` is BGW's stored ``2π·τ``,
        ``tau`` is the fractional translation.  Exactly one, named.  This
        function does no dividing of its own — it forwards, so the 2π
        still lives in exactly one place.
    fft_grid
        ``(3,)`` int — the grid ``r_mu_fft_idx`` indexes.  Required: the
        centroids are integers against it and their fractional
        coordinates cannot be recovered without it.
    extend_trs
        Forwarded to :func:`centroid_source_map_and_wrap`.  Default ``True``
        — every q-axis consumer in this tree indexes the tables with the
        TRS-augmented ``sym_idx_q``, and the one historical bug from
        getting this wrong was a silent clipped gather.
    tol
        Closure tolerance; see :data:`CLOSURE_TOL_DEFAULT`.
    context
        Names the call site for the announcement.

    Returns
    -------
    QgridSymmetryResolution
    """
    idx = np.asarray(r_mu_fft_idx)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(
            f"resolve_qgrid_symmetry: r_mu_fft_idx must be (n_rmu, 3); "
            f"got {idx.shape}")
    grid = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    if np.any(grid <= 0):
        raise ValueError(
            f"resolve_qgrid_symmetry: fft_grid must be positive; got "
            f"{grid.tolist()}")

    S = np.asarray(sym_matrices)
    n_sym_spatial = int(np.asarray(S).shape[0])

    # The verdict, taken in PURE FRACTIONAL coordinates.  ``fft_grid`` is
    # deliberately not forwarded: its only effects there are the
    # integer-index hash and a commensurability REFUSAL, and a refusal at
    # this point would convert a deck that has always degraded quietly
    # into a deck that raises.  The residual metric is identical with and
    # without it (the function says so), and the grid question is asked —
    # and answered with a message about the grid — by the table builder
    # below, which is the code that actually needs commensurability.
    cent_frac = idx.astype(np.float64) / grid[None, :].astype(np.float64)
    verdict = verify_centroid_orbit_closure(
        cent_frac, S, tnp=tnp, tau=tau, tol=tol)

    if not verdict.closed:
        head = verdict.describe().splitlines()[0]
        return QgridSymmetryResolution(
            mode="full_bz", verdict=verdict, sym_perm=None, L_table=None,
            reason=head, n_sym_spatial=n_sym_spatial, context=context)

    # Closed.  Build the tables.  ``validate=True`` is not redundant with
    # the verdict — see arm 2 in the docstring.
    try:
        sym_perm, L_table = centroid_source_map_and_wrap(
            idx.astype(np.int32),
            sym_matrices=S,
            translations=(np.asarray(tnp) if tnp is not None
                          else np.asarray(tau) * (2.0 * np.pi)),
            fft_grid=grid.astype(np.int32),
            validate=True,
            extend_trs=extend_trs,
        )
    except RuntimeError as exc:
        first = (str(exc.args[0]).splitlines()[0] if exc.args else str(exc))
        return QgridSymmetryResolution(
            mode="full_bz", verdict=verdict, sym_perm=None, L_table=None,
            reason=(f"closure measured CLOSED (worst "
                    f"{verdict.worst_residual:.3e} on op "
                    f"{verdict.worst_op}) but the permutation table "
                    f"refused on this FFT grid: {first}"),
            n_sym_spatial=n_sym_spatial, context=context)

    return QgridSymmetryResolution(
        mode="ibz", verdict=verdict, sym_perm=sym_perm, L_table=L_table,
        reason="", n_sym_spatial=n_sym_spatial, context=context)


# ─────────────────────────────────────────────────────────────────────────
# Full FFT-grid orbit permutation — used by ZetaLoader's q='full_bz' unfold.
# ─────────────────────────────────────────────────────────────────────────

def fft_grid_pullback_perm(
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
    Same convention as :func:`centroid_source_map_and_wrap`: BGW's
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
        Same meaning as :func:`centroid_source_map_and_wrap`.  ``translations``
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
                    f"fft_grid_pullback_perm: sym_perm[{s}] is not a "
                    f"permutation of [0, n_rtot={n_rtot}).  Likely cause: "
                    f"τ × fft_grid is not integer for sym {s} "
                    f"(τ_frac={tau_frac[s].tolist()}, "
                    f"τ×fft_grid={(tau_frac[s] * fg).tolist()}).  Off-grid "
                    f"fractional translations are not yet supported by the "
                    f"ζ full-BZ unfold path."
                )

    return sym_perm.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────
# Pre-sweep spellings — MODULE-LEVEL half of the compat layer
# ─────────────────────────────────────────────────────────────────────────
# See the twin block at the foot of :mod:`symmetry_maps.maps` and the
# retirement gate in :mod:`symmetry_maps._compat`.  The pair below is the
# one the sweep existed for: identical names, opposite directions, a
# recorded 4 eV silent failure for confusing them.

#: DEPRECATED — :func:`centroid_source_map_and_wrap` (SOURCE map + wrap).
compute_centroid_sym_perm = deprecated_alias(
    centroid_source_map_and_wrap, "compute_centroid_sym_perm")
#: DEPRECATED — :func:`fft_grid_pullback_perm` (PULL-BACK permutation).
compute_rgrid_sym_perm = deprecated_alias(
    fft_grid_pullback_perm, "compute_rgrid_sym_perm")
#: DEPRECATED — :func:`real_space_action_tables`.
build_real_space_syms = deprecated_alias(
    real_space_action_tables, "build_real_space_syms")
