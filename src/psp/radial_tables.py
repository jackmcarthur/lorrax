"""psp/radial_tables.py — Build all radial Hankel tables from species data.

Single pass: SpeciesData → q-grid tables for V_loc_sr, ρ_core, and β projectors.
All transforms go through one weighted Hankel kernel with different l.
Tables ship to GPU as JAX arrays via jnp.asarray.

THIS MODULE OWNS THE QE RADIAL QUADRATURE CONVENTIONS (QE 7.4, verified):

* ``_qe_simpsn_weights`` — upflib/simpsn.f90 composite-Simpson weights,
  including the even-n branch.
* ``qe_vloc_radial_scheme`` — local-potential / core-charge integrals run
  over ``msh`` points: the mesh truncated at r > 10 bohr and FORCED ODD
  (Modules/read_pseudo.f90:175-186), with simpsn weights.  Used by QE for
  V_loc (upflib/vloc_mod.f90 init_tab_vloc, both the q>0 tables and the
  q=0 alpha-Z branch) AND for the NLCC core charge (upflib/rhoc_mod.f90
  init_tab_rhc).
* ``qe_beta_radial_scheme`` — β-projector integrals run over EXACTLY
  ``upf%kkbeta`` points (upflib/beta_mod.f90 init_tab_beta), simpsn
  weights — kkbeta is often even, which is why the even-n branch exists.

``psp.radial.build_projectors_qe`` imports these; do not re-implement the
weights anywhere else.  Full-mesh generic-Simpson (or bare-rab rectangle)
integration is the defect class behind the 2026-08-28 KIH diagonal
offsets (-0.08 meV rectangle constant, +0.068 meV FR full-mesh tail,
+0.004 meV β cutoff mismatch) — see tests/test_scalar_psp.py red twins.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, erf as scipy_erf

from psp.species import SpeciesData


# ---------------------------------------------------------------------------
# Core Hankel transform — the one kernel everything flows through
# ---------------------------------------------------------------------------

def _hankel_weighted(l: int, r: np.ndarray, f_r: np.ndarray, q: np.ndarray,
                     w: np.ndarray) -> np.ndarray:
    """∫ dr r² f(r) j_l(qr) with explicit quadrature weights ``w``.

    ``w`` must already include the radial measure (rab) — callers obtain
    it from ``qe_vloc_radial_scheme`` / ``qe_beta_radial_scheme`` so the
    integration mesh and weights always travel together.

    Parameters
    ----------
    l : angular momentum (0, 1, 2, ...)
    r, f_r, w : (n,) radial grid, function, and weights (already truncated)
    q : (n_q,) momentum grid

    Returns
    -------
    table : (n_q,)
    """
    qr = q[:, None] * r[None, :]                     # (n_q, n)
    j_l = spherical_jn(l, qr)                        # (n_q, n)
    return np.sum(j_l * (r * r * f_r)[None, :] * w[None, :], axis=1)


# ---------------------------------------------------------------------------
# Table builders — QE-convention twins of the batched build_all_tables rows
# ---------------------------------------------------------------------------

def vloc_sr_table(sp: SpeciesData, q: np.ndarray) -> np.ndarray:
    """Short-range local potential: ∫ [V_loc(r) + Z·e²·erf(r)/r] j₀(qr) r² dr.

    QE convention: simpsn over the r ≤ 10 truncated odd mesh
    (upflib/vloc_mod.f90 init_tab_vloc).
    """
    e2 = 2.0
    msh, w = qe_vloc_radial_scheme(sp.r, sp.rab)
    r = sp.r[:msh]
    safe_r = np.where(r > 0, r, 1.0)
    erf_over_r = np.where(r > 0, scipy_erf(r) / safe_r, 2.0 / np.sqrt(np.pi))
    v_sr = sp.vloc_r[:msh] + sp.z_valence * e2 * erf_over_r
    return _hankel_weighted(0, r, v_sr, q, w)


def core_charge_table(sp: SpeciesData, q: np.ndarray) -> np.ndarray:
    """NLCC core density: ∫ ρ_core(r) j₀(qr) r² dr.

    QE convention: simpsn over the SAME msh mesh as V_loc
    (upflib/rhoc_mod.f90 init_tab_rhc integrates ``DO ir = 1, msh(nt)``).
    """
    msh, w = qe_vloc_radial_scheme(sp.r, sp.rab)
    return _hankel_weighted(0, sp.r[:msh], sp.rho_core_r[:msh], q, w)


def projector_table(sp: SpeciesData, ip: int, q: np.ndarray) -> np.ndarray:
    """Beta projector form factor: ∫ (β(r)/r) j_l(qr) r² dr.

    QE convention: simpsn over exactly kkbeta points
    (upflib/beta_mod.f90 init_tab_beta).
    """
    kkb, w = qe_beta_radial_scheme(sp.r, sp.rab, sp.kkbeta)
    return _hankel_weighted(int(sp.proj_l[ip]), sp.r[:kkb],
                            sp.beta_r[ip][:kkb], q, w)


def projector_deriv_table(sp: SpeciesData, ip: int, q: np.ndarray) -> np.ndarray:
    """Analytic q-derivative of the **reduced** form factor  G_l(q) ≡ F_l(q)/q^l.

    Starting from  F_l(q) = ∫ (β(r)/r) · j_l(qr) · r² dr  and the spherical-
    Bessel recurrence  j_l'(x) = −j_{l+1}(x) + (l/x)·j_l(x)  the q-derivative
    of the reduced form factor collapses (after the l·F_l/q^{l+1} terms cancel)
    to the clean expression

        dG_l/dq(q)  =  −  H_{l+1}(β; q)  /  q^l          (q > 0)

    where  H_{l+1}(β; q) ≡ ∫ β(r) · j_{l+1}(qr) · r² dr  (the l+1 Hankel of β).

    This is the radial-form-factor derivative that mature DFT codes (QE, Abinit,
    VASP) tabulate analytically to avoid the FD / interpolation-slope inaccuracies
    that plague autodiff-through-``_table_interp`` paths.

    Analyticity at q = 0:  j_{l+1}(qr) ∼ (qr)^{l+1}/(2l+3)!!  so  H_{l+1}(β; q) ∼
    q^{l+1} — the ratio → 0 as q → 0 (set exactly to zero at the q=0 grid point,
    matching the evenness of G_l(q)).

    Quadrature: QE β convention — simpsn over kkbeta points, matching the
    forward table (``projector_table`` / ``build_all_tables``).
    """
    l = int(sp.proj_l[ip])
    kkb, w = qe_beta_radial_scheme(sp.r, sp.rab, sp.kkbeta)
    beta = sp.beta_r[ip][:kkb] * sp.r[:kkb]  # sp.beta_r stores β(r)/r → restore β(r)
    H_lp1 = _hankel_weighted(l + 1, sp.r[:kkb], beta, q, w)
    if l == 0:
        return -H_lp1
    deriv = np.empty_like(H_lp1)
    mask = q > 0
    deriv[mask] = -H_lp1[mask] / q[mask] ** l
    deriv[~mask] = 0.0
    return deriv


def alpha_z(sp: SpeciesData, vol: float) -> float:
    """G=0 local potential: (4π/Ω) ∫ r·[r·V_loc(r) + Z·e²] dr.

    QE convention (upflib/vloc_mod.f90 init_tab_vloc, q=0 branch): simpsn
    over msh points — the r ≤ 10 truncated, forced-odd mesh.  Integrating
    the FULL mesh instead picks up the r > 10 tail where the tabulated
    vloc deviates from -Z·e²/r in the last digits — a pseudo-DEPENDENT
    constant on every KIH diagonal (+0.068 meV for the FR Si pseudo,
    -0.0008 meV for the SR one; measured 2026-08-28).
    """
    e2 = 2.0
    msh, w = qe_vloc_radial_scheme(sp.r, sp.rab)
    integrand = sp.r[:msh] * (sp.r[:msh] * sp.vloc_r[:msh] + sp.z_valence * e2)
    return float(4.0 * np.pi * np.sum(integrand * w) / vol)


# ---------------------------------------------------------------------------
# Build all tables for all species in one pass
# ---------------------------------------------------------------------------

def _simpson_weights(n_r: int) -> np.ndarray:
    """Simpson 1/3 quadrature weights for a uniform n_r-point grid."""
    sw = np.ones(n_r, dtype=np.float64)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[0] = sw[-1] = 1.0 / 3.0
    return sw


def _qe_simpsn_weights(n: int) -> np.ndarray:
    """QE's composite-Simpson weights, upflib/simpsn.f90 EXACTLY.

    Odd n: the standard 1/3, 4/3, 2/3, ..., 4/3, 1/3.  Even n: QE's
    even-mesh branch — the last point gets weight 0 and the one before it
    net 1/3 (interior 2/3 minus the closing 1/3), i.e. the standard odd
    rule on the first n-1 points.  β integrals in QE run over exactly
    ``upf%kkbeta`` points (often even — 196 for the PseudoDojo Si UPFs),
    so matching QE's V_NL requires this branch, not a parity fix-up.
    """
    w = np.zeros(n, dtype=np.float64)
    i = np.arange(2, n)                      # 1-based interior 2..n-1
    w[1:n - 1] = np.abs((i % 2) - 2) * 2.0   # even i -> 4, odd i -> 2
    if n % 2 == 1:
        w[0] += 1.0
        w[n - 1] += 1.0
    else:
        w[0] += 1.0
        w[n - 2] -= 1.0
    return w / 3.0


def qe_vloc_radial_scheme(
    r: np.ndarray,
    rab: np.ndarray | None,
) -> tuple[int, np.ndarray]:
    """QE's radial quadrature for local-potential/core-charge integrals.

    Returns ``(msh, weights)``; integrate ``f[:msh] · weights``.

    Matches QE exactly (verified against QE 7.4 sources):
    - Modules/read_pseudo.f90:175-186 — the integration mesh is truncated at
      ``rcut = 10`` bohr (msh = 1-based index of the first r > 10, or the full
      mesh) and then FORCED ODD: ``msh = 2*((msh+1)/2) - 1``.
    - upflib/simpsn.f90 — composite-Simpson weights on the (odd) truncated
      mesh, times ``rab``.

    Consumed by upflib/vloc_mod.f90 init_tab_vloc (q>0 tables AND the q=0
    alpha-Z branch) and upflib/rhoc_mod.f90 init_tab_rhc (NLCC).

    A bare ``rab`` (rectangle-rule) weighting is NOT equivalent: by
    Euler–Maclaurin its error against Simpson is ~ -(h²/12)·f'(0), and the
    alpha-Z integrand r·(r·V_loc + Z·e²) has f'(0) = Z·e² exactly — a
    pseudo-independent, band-independent constant that showed up as a
    -0.08 meV offset on every KIH diagonal (Si, h = 0.01).  The full-mesh
    tail (r > 10, where tabulated vloc deviates from -Z·e²/r in the last
    digits) added a further pseudo-DEPENDENT constant (+0.068 meV for the
    FR Si pseudo).  Both vanish with QE's scheme.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    above = np.nonzero(r > 10.0)[0]
    msh = int(above[0]) + 1 if above.size else n     # 1-based, as in QE
    msh = 2 * ((msh + 1) // 2) - 1                   # forced odd
    w = _qe_simpsn_weights(msh)
    if rab is not None:
        w = w * np.asarray(rab, dtype=float)[:msh]
    else:
        w = w * (float(r[1] - r[0]) if n > 1 else 1.0)
    return msh, w


def qe_beta_radial_scheme(
    r: np.ndarray,
    rab: np.ndarray | None,
    kkbeta: int,
) -> tuple[int, np.ndarray]:
    """QE's radial quadrature for β-projector integrals.

    Returns ``(kkb, weights)``; integrate ``f[:kkb] · weights``.

    upflib/beta_mod.f90 init_tab_beta: ``simpson(upf%kkbeta, aux, rab)``
    — EXACTLY kkbeta points (max cutoff_radius_index over the betas; often
    EVEN, 196 for the PseudoDojo Si UPFs, hence the even-n simpsn branch),
    NOT the full mesh.  ONCVPSP l=2 betas end on ~1e-4 nonzero values at
    the cutoff index, where full-mesh generic weights disagree with QE —
    measured as a +0.004 meV constant on every V_NL diagonal.

    ``kkbeta <= 0`` falls back to the full mesh (UPF without cutoff index).
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    kkb = int(kkbeta) if kkbeta else n
    kkb = min(max(kkb, 1), n)
    w = _qe_simpsn_weights(kkb)
    if rab is not None:
        w = w * np.asarray(rab, dtype=float)[:kkb]
    else:
        w = w * (float(r[1] - r[0]) if n > 1 else 1.0)
    return kkb, w


def build_all_tables(
    species_list: list[SpeciesData],
    q_max: float,
    n_q: int = 4000,
) -> dict:
    """Build Hankel tables for all species on a uniform q-grid.

    Per-species the per-projector forward (F_l) and analytic-deriv
    raw-Hankel (H_{l+1}) tables are produced in two batched JAX kernel
    invocations — one per unique l (resp. l+1).  All Bessel evaluation
    + integrand reduction lives on GPU; only the (n_proj_s, n_q)
    result moves back to host.  Replaces a per-projector scipy.special
    .spherical_jn call site (~10 s total on MoS2) with ~1 s of GPU
    work + persistent-cached compile.

    Returns dict with:
      q : (n_q,) uniform grid
      dq : float
      vloc : (n_species, n_q) V_loc short-range tables  (l=0 Hankel)
      nlcc : (n_species, n_q) core density tables       (l=0 Hankel)
      has_vloc, has_nlcc : (n_species,) bool
      proj_tables   : list of (n_proj_s, n_q) — F_l(q) per species
      deriv_tables  : list of (n_proj_s, n_q) — raw H_{l+1}(β; q) per
                       species, the *unscaled* integral that
                       projector_deriv_table normalises by /q^l.  Caller
                       (build_vnl_setup) does the q^l division.
    """
    import jax.numpy as jnp
    from psp.radial.radial_jax import spherical_hankel_table_batch_jax

    q = np.linspace(0.0, max(q_max, 1e-8), n_q)
    n_sp = len(species_list)
    q_j = jnp.asarray(q, dtype=jnp.float64)

    vloc = np.zeros((n_sp, n_q), dtype=np.float64)
    nlcc = np.zeros((n_sp, n_q), dtype=np.float64)
    has_vloc = np.ones(n_sp, dtype=bool)
    has_nlcc = np.zeros(n_sp, dtype=bool)
    proj_tables = []
    deriv_tables = []

    e2 = 2.0
    for i, sp in enumerate(species_list):
        n_r = len(sp.r)

        # ── vloc + core (l=0 single rows, batched together) ──
        # QE convention for BOTH: simpsn over the r ≤ 10 truncated odd
        # mesh (msh) — upflib/vloc_mod.f90 init_tab_vloc and
        # upflib/rhoc_mod.f90 init_tab_rhc integrate ``DO ir = 1, msh``.
        # Full-mesh generic-Simpson weights here put the +0.068 meV
        # (FR Si) alpha-Z-class tail constant on the scf-potential lane.
        msh, w0_np = qe_vloc_radial_scheme(sp.r, sp.rab)
        r0 = sp.r[:msh]
        r0_j = jnp.asarray(r0, dtype=jnp.float64)
        w0_j = jnp.asarray(w0_np, dtype=jnp.float64)
        v_sr_rows = [None, None]
        safe_r = np.where(r0 > 0, r0, 1.0)
        erf_over_r = np.where(r0 > 0, scipy_erf(r0) / safe_r,
                              2.0 / np.sqrt(np.pi))
        v_sr_rows[0] = sp.vloc_r[:msh] + sp.z_valence * e2 * erf_over_r
        v_sr_rows[1] = sp.rho_core_r[:msh] if sp.has_nlcc else np.zeros(msh)
        l0_block = spherical_hankel_table_batch_jax(
            0, r0_j, jnp.asarray(np.stack(v_sr_rows), dtype=jnp.float64),
            q_j, w0_j,
        )
        l0_np = np.asarray(l0_block)
        vloc[i] = l0_np[0]
        if sp.has_nlcc:
            nlcc[i] = l0_np[1]
            has_nlcc[i] = True

        # ── Projector forward + deriv tables, grouped by l (resp. l+1) ──
        # β integrals use QE's convention (upflib/beta_mod.f90 init_tab_beta):
        # simpsn weights over EXACTLY kkbeta points, NOT the full mesh.
        # ONCVPSP l=2 betas end on ~1e-4 nonzero values at the cutoff
        # index, where full-mesh generic weights disagree with QE —
        # measured as a +0.004 meV constant on every V_NL diagonal.
        kkb, wb_np = qe_beta_radial_scheme(
            sp.r, sp.rab, int(getattr(sp, "kkbeta", 0)))
        rb_j = jnp.asarray(sp.r[:kkb], dtype=jnp.float64)
        wb_j = jnp.asarray(wb_np, dtype=jnp.float64)

        n_proj = sp.n_proj
        ls = np.asarray(sp.proj_l, dtype=int)
        F_table = np.zeros((n_proj, n_q), dtype=np.float64)
        H_table = np.zeros((n_proj, n_q), dtype=np.float64)

        # Forward F_l: integrand is (β/r), Bessel order l_p
        beta_over_r = np.asarray(sp.beta_r, dtype=np.float64)[:, :kkb]  # (n_proj, kkb)
        # Deriv raw H_{l+1}: integrand is β(r) = (β/r)·r, Bessel order l_p+1
        beta_full = beta_over_r * sp.r[None, :kkb]                      # (n_proj, kkb)

        for l_val in np.unique(ls):
            idx = np.where(ls == l_val)[0]
            F_block = spherical_hankel_table_batch_jax(
                int(l_val), rb_j,
                jnp.asarray(beta_over_r[idx], dtype=jnp.float64),
                q_j, wb_j,
            )
            F_table[idx] = np.asarray(F_block)

        for l_val in np.unique(ls + 1):
            idx = np.where(ls + 1 == l_val)[0]
            H_block = spherical_hankel_table_batch_jax(
                int(l_val), rb_j,
                jnp.asarray(beta_full[idx], dtype=jnp.float64),
                q_j, wb_j,
            )
            H_table[idx] = np.asarray(H_block)

        proj_tables.append(F_table)
        deriv_tables.append(H_table)

    return dict(q=q, dq=float(q[1] - q[0]) if n_q > 1 else 1.0,
                vloc=vloc, nlcc=nlcc,
                has_vloc=has_vloc, has_nlcc=has_nlcc,
                proj_tables=proj_tables,
                deriv_tables=deriv_tables)
