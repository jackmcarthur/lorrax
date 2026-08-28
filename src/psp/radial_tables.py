"""psp/radial_tables.py — Build all radial Hankel tables from species data.

Single pass: SpeciesData → q-grid tables for V_loc_sr, ρ_core, and β projectors.
All transforms go through one Hankel function with different l.
Tables ship to GPU as JAX arrays via jnp.asarray.
"""
from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, erf as scipy_erf

from psp.species import SpeciesData


# ---------------------------------------------------------------------------
# Core Hankel transform — the one function everything flows through
# ---------------------------------------------------------------------------

def hankel_l(l: int, r: np.ndarray, f_r: np.ndarray, q: np.ndarray,
             rab: np.ndarray) -> np.ndarray:
    """∫ dr r² f(r) j_l(qr) rab(r)  — Hankel transform with Simpson weights.

    Parameters
    ----------
    l : angular momentum (0, 1, 2, ...)
    r, f_r, rab : (n_r,) radial grid, function, and integration weights
    q : (n_q,) momentum grid

    Returns
    -------
    table : (n_q,)
    """
    n_r = len(r)
    # Simpson weights: 1/3, 4/3, 2/3, 4/3, ..., 4/3, 1/3
    sw = np.ones(n_r)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[0] = sw[-1] = 1.0 / 3.0
    w = sw * rab

    qr = q[:, None] * r[None, :]                     # (n_q, n_r)
    j_l = spherical_jn(l, qr)                        # (n_q, n_r)
    return np.sum(j_l * (r * r * f_r)[None, :] * w[None, :], axis=1)


# ---------------------------------------------------------------------------
# Table builders — thin wrappers over hankel_l
# ---------------------------------------------------------------------------

def vloc_sr_table(sp: SpeciesData, q: np.ndarray) -> np.ndarray:
    """Short-range local potential: ∫ [V_loc(r) + Z·e²·erf(r)/r] j₀(qr) r² dr."""
    e2 = 2.0
    safe_r = np.where(sp.r > 0, sp.r, 1.0)
    erf_over_r = np.where(sp.r > 0, scipy_erf(sp.r) / safe_r, 2.0 / np.sqrt(np.pi))
    v_sr = sp.vloc_r + sp.z_valence * e2 * erf_over_r
    return hankel_l(0, sp.r, v_sr, q, sp.rab)


def core_charge_table(sp: SpeciesData, q: np.ndarray) -> np.ndarray:
    """NLCC core density: ∫ ρ_core(r) j₀(qr) r² dr."""
    return hankel_l(0, sp.r, sp.rho_core_r, q, sp.rab)


def projector_table(sp: SpeciesData, ip: int, q: np.ndarray) -> np.ndarray:
    """Beta projector form factor: ∫ (β(r)/r) j_l(qr) r² dr."""
    return hankel_l(int(sp.proj_l[ip]), sp.r, sp.beta_r[ip], q, sp.rab)


def projector_deriv_table(sp: SpeciesData, ip: int, q: np.ndarray) -> np.ndarray:
    """Analytic q-derivative of the **reduced** form factor  G_l(q) ≡ F_l(q)/q^l.

    Starting from  F_l(q) = ∫ (β(r)/r) · j_l(qr) · r² dr  and the spherical-
    Bessel recurrence  j_l'(x) = −j_{l+1}(x) + (l/x)·j_l(x)  the q-derivative
    of the reduced form factor collapses (after the l·F_l/q^{l+1} terms cancel)
    to the clean expression

        dG_l/dq(q)  =  −  H_{l+1}(β; q)  /  q^l          (q > 0)

    where  H_{l+1}(β; q) ≡ ∫ β(r) · j_{l+1}(qr) · r² dr  (= ``hankel_l(l+1, β)``).

    This is the radial-form-factor derivative that mature DFT codes (QE, Abinit,
    VASP) tabulate analytically to avoid the FD / interpolation-slope inaccuracies
    that plague autodiff-through-``_table_interp`` paths.

    Analyticity at q = 0:  j_{l+1}(qr) ∼ (qr)^{l+1}/(2l+3)!!  so  H_{l+1}(β; q) ∼
    q^{l+1} — the ratio → 0 as q → 0 (set exactly to zero at the q=0 grid point,
    matching the evenness of G_l(q)).
    """
    l = int(sp.proj_l[ip])
    beta = sp.beta_r[ip] * sp.r              # sp.beta_r stores β(r)/r → restore β(r)
    H_lp1 = hankel_l(l + 1, sp.r, beta, q, sp.rab)
    if l == 0:
        return -H_lp1
    deriv = np.empty_like(H_lp1)
    mask = q > 0
    deriv[mask] = -H_lp1[mask] / q[mask] ** l
    deriv[~mask] = 0.0
    return deriv


def alpha_z(sp: SpeciesData, vol: float) -> float:
    """G=0 local potential: (4π/Ω) ∫ r·[r·V_loc(r) + Z·e²] rab dr."""
    e2 = 2.0
    integrand = sp.r * (sp.r * sp.vloc_r + sp.z_valence * e2)
    # Simpson weights
    n_r = len(sp.r)
    sw = np.ones(n_r)
    sw[1:-1:2] = 4.0 / 3.0
    sw[2:-1:2] = 2.0 / 3.0
    sw[0] = sw[-1] = 1.0 / 3.0
    return float(4.0 * np.pi * np.sum(integrand * sw * sp.rab) / vol)


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
        weights_np = _simpson_weights(n_r) * sp.rab
        r_j = jnp.asarray(sp.r, dtype=jnp.float64)
        w_j = jnp.asarray(weights_np, dtype=jnp.float64)

        # ── vloc + core (l=0 single rows, batched together) ──
        v_sr_rows = [None, None]
        safe_r = np.where(sp.r > 0, sp.r, 1.0)
        erf_over_r = np.where(sp.r > 0, scipy_erf(sp.r) / safe_r,
                              2.0 / np.sqrt(np.pi))
        v_sr_rows[0] = sp.vloc_r + sp.z_valence * e2 * erf_over_r
        v_sr_rows[1] = sp.rho_core_r if sp.has_nlcc else np.zeros(n_r)
        l0_block = spherical_hankel_table_batch_jax(
            0, r_j, jnp.asarray(np.stack(v_sr_rows), dtype=jnp.float64),
            q_j, w_j,
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
        kkb = int(getattr(sp, "kkbeta", 0)) or n_r
        kkb = min(kkb, n_r)
        wb_np = _qe_simpsn_weights(kkb) * sp.rab[:kkb]
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
