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


def _odd_double_factorial(n: int) -> float:
    """Return ``n!!`` for positive odd ``n`` (and ``(-1)!! = 1``)."""
    value = 1
    for factor in range(n, 0, -2):
        value *= factor
    return float(value)


def projector_reduced_origin(sp: SpeciesData, ip: int) -> float:
    r"""Exact regular value ``lim_(q->0) F_l(q)/q^l`` as a radial moment."""
    l = int(sp.proj_l[ip])
    weights = _simpson_weights(len(sp.r)) * sp.rab
    moment = np.sum(
        np.asarray(sp.beta_r[ip], dtype=np.float64)
        * sp.r ** (l + 2) * weights)
    return float(moment / _odd_double_factorial(2 * l + 1))


def _reduced_projector_second_derivative(
    q: np.ndarray,
    l: int,
    H_lp1: np.ndarray,
    J_lp2: np.ndarray,
    origin_moment: float,
) -> np.ndarray:
    r"""Canonical Bessel-recurrence algebra for ``d2(F_l/q^l)/dq2``.

    ``H_lp1 = int beta(r) j_(l+1)(qr) r^2 dr`` and
    ``J_lp2 = int r beta(r) j_(l+2)(qr) r^2 dr``.  For ``q > 0``,

    ``G_l'' = J_(l+2)/q^l - H_(l+1)/q^(l+1)``.

    The two terms have a finite analytic limit at the origin.  Evaluating
    that limit as the radial moment, instead of subtracting singular terms,
    is essential for the exact Gamma-point Hessian.
    """
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(H_lp1, dtype=np.float64)
    nonzero = q > 0.0
    out[nonzero] = (
        J_lp2[nonzero] / q[nonzero] ** l
        - H_lp1[nonzero] / q[nonzero] ** (l + 1)
    )
    out[~nonzero] = -origin_moment / _odd_double_factorial(2 * l + 3)
    return out


def projector_second_deriv_table(
    sp: SpeciesData, ip: int, q: np.ndarray,
) -> np.ndarray:
    r"""Analytic ``d2(F_l/q^l)/dq2`` with the exact regular origin moment."""
    l = int(sp.proj_l[ip])
    beta = np.asarray(sp.beta_r[ip], dtype=np.float64) * sp.r
    H_lp1 = hankel_l(l + 1, sp.r, beta, q, sp.rab)
    J_lp2 = hankel_l(l + 2, sp.r, beta * sp.r, q, sp.rab)
    weights = _simpson_weights(len(sp.r)) * sp.rab
    origin_moment = float(np.sum(beta * sp.r ** (l + 3) * weights))
    return _reduced_projector_second_derivative(
        q, l, H_lp1, J_lp2, origin_moment)


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


def build_all_tables(
    species_list: list[SpeciesData],
    q_max: float,
    n_q: int = 4000,
    *,
    second_derivatives: bool = False,
) -> dict:
    """Build Hankel tables for all species on a uniform q-grid.

    Per-species the per-projector forward (F_l) and analytic-deriv
    raw-Hankel (H_{l+1}) tables are produced in two batched JAX kernel
    families — one per unique l (resp. l+1).  Contact setup adds the
    opt-in l+2 family.  All Bessel evaluation
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
      reduced_origins : list of (n_proj_s,) — exact regular
                       ``lim_(q->0) F_l/q^l`` radial moments.
      deriv_tables  : list of (n_proj_s, n_q) — raw H_{l+1}(β; q) per
                       species, the *unscaled* integral that
                       projector_deriv_table normalises by /q^l.  Caller
                       (build_vnl_setup) does the q^l division.
      second_deriv_tables : list of (n_proj_s, n_q) or None — analytic
                       ``d2(F_l/q^l)/dq2``, including the exact regular
                       radial-moment value at q=0.  Built only when
                       ``second_derivatives=True`` so ordinary Hamiltonian
                       setup does not pay the extra l+2 Bessel pass.
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
    reduced_origins = []
    deriv_tables = []
    second_deriv_tables = [] if second_derivatives else None

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
        n_proj = sp.n_proj
        ls = np.asarray(sp.proj_l, dtype=int)
        F_table = np.zeros((n_proj, n_q), dtype=np.float64)
        H_table = np.zeros((n_proj, n_q), dtype=np.float64)

        # Forward F_l: integrand is (β/r), Bessel order l_p
        beta_over_r = np.asarray(sp.beta_r, dtype=np.float64)        # (n_proj, n_r)
        # Deriv raw H_{l+1}: integrand is β(r) = (β/r)·r, Bessel order l_p+1
        beta_full = beta_over_r * sp.r[None, :]                       # (n_proj, n_r)

        for l_val in np.unique(ls):
            idx = np.where(ls == l_val)[0]
            F_block = spherical_hankel_table_batch_jax(
                int(l_val), r_j,
                jnp.asarray(beta_over_r[idx], dtype=jnp.float64),
                q_j, w_j,
            )
            F_table[idx] = np.asarray(F_block)

        for l_val in np.unique(ls + 1):
            idx = np.where(ls + 1 == l_val)[0]
            H_block = spherical_hankel_table_batch_jax(
                int(l_val), r_j,
                jnp.asarray(beta_full[idx], dtype=jnp.float64),
                q_j, w_j,
            )
            H_table[idx] = np.asarray(H_block)

        if second_derivatives:
            J_table = np.zeros((n_proj, n_q), dtype=np.float64)
            # Second derivative recurrence: J_{l+2} uses r*beta(r).
            for l_val in np.unique(ls + 2):
                idx = np.where(ls + 2 == l_val)[0]
                J_block = spherical_hankel_table_batch_jax(
                    int(l_val), r_j,
                    jnp.asarray(beta_full[idx] * sp.r[None, :],
                                dtype=jnp.float64),
                    q_j, w_j,
                )
                J_table[idx] = np.asarray(J_block)

            Gpp_table = np.empty_like(F_table)
            for ip, l_val in enumerate(ls):
                origin_moment = float(np.sum(
                    beta_full[ip] * sp.r ** (int(l_val) + 3) * weights_np))
                Gpp_table[ip] = _reduced_projector_second_derivative(
                    q, int(l_val), H_table[ip], J_table[ip], origin_moment)

        proj_tables.append(F_table)
        reduced_origins.append(np.asarray([
            projector_reduced_origin(sp, ip) for ip in range(n_proj)
        ], dtype=np.float64))
        deriv_tables.append(H_table)
        if second_derivatives:
            second_deriv_tables.append(Gpp_table)

    return dict(q=q, dq=float(q[1] - q[0]) if n_q > 1 else 1.0,
                vloc=vloc, nlcc=nlcc,
                has_vloc=has_vloc, has_nlcc=has_nlcc,
                proj_tables=proj_tables,
                reduced_origins=reduced_origins,
                deriv_tables=deriv_tables,
                second_deriv_tables=second_deriv_tables)
