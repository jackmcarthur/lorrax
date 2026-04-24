"""Helpers for the q=0, G=G'=0 Coulomb head.

The modern GWJAX paths keep the head separate from the ISDF body tensors:

- Dynamic GN-PPM uses scalar head samples ``(v_h, W_h(0), W_h(iω_p))``.
- Static COHSEX uses exact band-diagonal head shifts for ``Σ^X``, ``Σ^SX``,
  ``Σ^(SX-X)``, and ``Σ^COH``.

This module centralizes:

- head source resolution (`override`, `epshead`, `s_tensor`)
- scalar GN-PPM head fitting
- exact static COHSEX head terms
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class HeadSample:
    """Resolved q=0 Coulomb head sample at one frequency."""

    vc0: complex
    wcoul0: complex
    source: str
    omega: complex


@dataclass(frozen=True)
class HeadGNParams:
    """Fitted GN parameters for the scalar Coulomb head."""

    omega_h_sq: float
    omega_h: float
    B_h: float
    R_h: float
    wc_head_0: float
    wc_head_iwp: float
    vc0: float
    omega_p: float


@dataclass(frozen=True)
class StaticHeadTerms:
    """Exact static q=0 head terms for bare X / SX / COHSEX.

    All values are diagonal-in-band shifts in Rydberg atomic units.
    The head contributes equally at every k-point, with the Brillouin-zone
    average carried by the explicit ``1 / N_k`` factor.
    """

    sigma_x_diag: jnp.ndarray
    sigma_sx_diag: jnp.ndarray
    sigma_sx_minus_x_diag: jnp.ndarray
    sigma_coh_diag: jnp.ndarray
    vc0: complex
    wcoul0: complex
    wc_head_0: complex
    source: str


def _representative_entry(diag: jnp.ndarray) -> complex:
    """Return a representative diagonal value for diagnostics."""

    arr = np.asarray(diag).reshape(-1)
    if arr.size == 0:
        return 0.0 + 0.0j
    nz = np.flatnonzero(np.abs(arr) > 0.0)
    idx = int(nz[0]) if nz.size else 0
    return complex(arr[idx])


def resolve_head_override(params, omega) -> HeadSample | None:
    """Return explicit head overrides when both v and W are provided."""

    omega_val = complex(omega)
    vhead_override = params.get("vhead")
    w_key = "whead_0freq" if abs(omega_val) <= 1.0e-14 else "whead_imfreq"
    whead_override = params.get(w_key)
    if vhead_override is None or whead_override is None:
        return None
    source = "override" if abs(omega_val) <= 1.0e-14 else f"override(omega={omega_val} Ry)"
    return HeadSample(
        vc0=complex(vhead_override),
        wcoul0=complex(whead_override),
        source=source,
        omega=omega_val,
    )


def resolve_head_sample(params, input_dir, wfn, sym, meta, print_fn, omega) -> HeadSample:
    """Resolve a q=0 head sample using overrides and configured source order."""

    override = resolve_head_override(params, omega)
    if override is not None:
        return override

    want_source = str(params.get("wcoul0_source", "s_tensor")).strip().lower()
    if want_source not in ("epshead", "s_tensor"):
        print_fn(f"Unknown wcoul0_source={want_source}; defaulting to 's_tensor'")
        want_source = "s_tensor"

    omega_val = complex(omega)
    eta = float(params.get("wcoul0_eta", 0.0) or 0.0)
    eps0_path = os.path.join(input_dir, "eps0mat.h5")
    dipole_path = os.path.join(input_dir, "dipole.h5")

    def from_epshead() -> HeadSample | None:
        if not os.path.exists(eps0_path):
            return None
        try:
            if abs(omega_val) > 1.0e-14:
                print_fn(
                    f"wcoul0_source=epshead is static-only; using epshead(0) for omega={omega_val} Ry"
                )
            from file_io.epsmat_reader import EPSReader
            from gw.vcoul import compute_q0_averages

            eps0 = EPSReader(eps0_path)
            vc0_mean, wcoul0 = compute_q0_averages(
                wfn,
                jnp.asarray(eps0.epshead, dtype=jnp.complex128),
                meta,
                S_cart=None,
            )
            source = "epshead(0)" if abs(omega_val) > 1.0e-14 else "epshead"
            return HeadSample(
                vc0=complex(vc0_mean),
                wcoul0=complex(wcoul0),
                source=source,
                omega=omega_val,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            print_fn(f"epshead wcoul0 failed: {exc}")
            return None

    def from_s_tensor() -> HeadSample | None:
        if not os.path.exists(dipole_path):
            print_fn(f"dipole.h5 not found at {dipole_path}; cannot build S(omega) wcoul0")
            return None
        from common.chi_from_dipole import read_dipole_h5, compute_S_omega
        from gw.vcoul import compute_q0_averages

        dipole_cart, deltaE = read_dipole_h5(dipole_path)
        nk_tot = int(sym.nk_tot)
        nb = int(dipole_cart.shape[2])
        nelec = int(wfn.nelec)
        occ = np.zeros((nk_tot, nb), dtype=float)
        occ[:, :max(0, min(nelec, nb))] = 1.0
        f_nk = jnp.asarray(occ, dtype=jnp.float64)
        omega_grid = jnp.asarray([omega_val], dtype=jnp.complex128)
        S_cart_omega = compute_S_omega(
            dipole_cart,
            deltaE,
            f_nk,
            float(wfn.cell_volume),
            int(sym.nk_tot),
            int(wfn.nspin),
            int(wfn.nspinor),
            omega_grid,
            eta=eta,
        )[0]
        vc0_mean, wcoul0 = compute_q0_averages(
            wfn,
            jnp.asarray(0.0, dtype=jnp.float64),
            meta,
            S_cart=S_cart_omega,
        )
        source = "s_tensor" if abs(omega_val) <= 1.0e-14 else f"s_tensor(omega={omega_val} Ry)"
        return HeadSample(
            vc0=complex(vc0_mean),
            wcoul0=complex(wcoul0),
            source=source,
            omega=omega_val,
        )

    source_order = [want_source] + [s for s in ("epshead", "s_tensor") if s != want_source]
    for source in source_order:
        result = from_epshead() if source == "epshead" else from_s_tensor()
        if result is not None:
            return result

    raise RuntimeError(
        "Failed to resolve q=0 Coulomb head: neither explicit overrides nor supported sources are available."
    )


def format_head_sample_diagnostics(head: HeadSample, *, include_screened: bool = True) -> str:
    """Return a compact diagnostic summary for one resolved head sample."""

    lines = [
        "",
        "-" * 72,
        "  FINITE-SIZE CORRECTIONS",
        "-" * 72,
        f"  Head source: {head.source}",
        f"  v(q→0)  = {head.vc0.real:12.3f} a.u.  (bare Coulomb head)",
    ]
    if include_screened:
        if abs(head.omega) > 1.0e-14:
            lines.append(f"  Head frequency ω = {head.omega} Ry")
        lines.append(f"  W(q→0)  = {head.wcoul0.real:12.3f} a.u.  (screened Coulomb head)")
        lines.append(
            f"  ΔW      = {(head.wcoul0.real - head.vc0.real):12.3f} a.u.  (screening correction)"
        )
    return "\n".join(lines)


def format_head_pair_diagnostics(head_static: HeadSample, head_imag: HeadSample) -> str:
    """Return a compact summary of static and imaginary-frequency head samples."""

    lines = [
        "",
        "-" * 72,
        "  FINITE-SIZE CORRECTIONS",
        "-" * 72,
        f"  Head source (ω=0):    {head_static.source}",
        f"  Head source (ω=iωp):  {head_imag.source}",
        f"  v(q→0)               = {head_static.vc0.real:12.3f} a.u.  (bare Coulomb head)",
        f"  W(q→0, ω=0)          = {head_static.wcoul0.real:12.3f} a.u.",
        f"  W(q→0, ω=iωp)        = {head_imag.wcoul0.real:12.3f} a.u.  [ω={head_imag.omega} Ry]",
        f"  W^c(q→0, ω=0)        = {(head_static.wcoul0.real - head_static.vc0.real):12.3f} a.u.",
        f"  W^c(q→0, ω=iωp)      = {(head_imag.wcoul0.real - head_imag.vc0.real):12.3f} a.u.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic GN-PPM scalar head
# ---------------------------------------------------------------------------

def fit_head_gn(
    vc0: float,
    wcoul0_static: float,
    wcoul0_imfreq: float,
    omega_p_ry: float,
) -> HeadGNParams:
    """Fit a scalar GN pole from two W^c head samples."""

    w1 = wcoul0_static - vc0
    w2 = wcoul0_imfreq - vc0
    omega_2_sq = -(omega_p_ry ** 2)

    denom = w1 - w2
    if abs(denom) < 1.0e-30:
        return HeadGNParams(
            omega_h_sq=1.0,
            omega_h=1.0,
            B_h=0.0,
            R_h=0.0,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=omega_p_ry,
        )

    omega_h_sq = -w2 * omega_2_sq / denom
    B_h = -w1 * omega_h_sq

    if omega_h_sq <= 0.0:
        omega_h = abs(omega_h_sq) ** 0.5 if omega_h_sq != 0.0 else 1.0
        R_h = B_h / (2.0 * omega_h) if omega_h > 1.0e-30 else 0.0
        return HeadGNParams(
            omega_h_sq=omega_h_sq,
            omega_h=omega_h,
            B_h=B_h,
            R_h=R_h,
            wc_head_0=w1,
            wc_head_iwp=w2,
            vc0=vc0,
            omega_p=omega_p_ry,
        )

    omega_h = omega_h_sq ** 0.5
    R_h = B_h / (2.0 * omega_h)
    return HeadGNParams(
        omega_h_sq=omega_h_sq,
        omega_h=omega_h,
        B_h=B_h,
        R_h=R_h,
        wc_head_0=w1,
        wc_head_iwp=w2,
        vc0=vc0,
        omega_p=omega_p_ry,
    )


def fit_head_gn_from_samples(
    head_static: HeadSample,
    head_imag: HeadSample,
    *,
    omega_p_ry: float,
) -> HeadGNParams:
    """Fit the scalar GN head from resolved static and imaginary-frequency samples."""

    return fit_head_gn(
        vc0=float(head_static.vc0.real),
        wcoul0_static=float(head_static.wcoul0.real),
        wcoul0_imfreq=float(head_imag.wcoul0.real),
        omega_p_ry=omega_p_ry,
    )


_RY2EV = 13.6056980659


# ---------------------------------------------------------------------------
# Exact static COHSEX head
# ---------------------------------------------------------------------------

def compute_static_head_terms(
    *,
    vc0: complex,
    wcoul0_static: complex,
    occ: np.ndarray | jnp.ndarray,
    cell_volume: float,
    nk_tot: int,
    source: str = "unknown",
) -> StaticHeadTerms:
    """Build exact static COHSEX head terms in band space.

    Parameters
    ----------
    vc0
        Bare Coulomb head ``v_h`` in atomic units.
    wcoul0_static
        Static screened Coulomb head ``W_h(omega=0)`` in atomic units.
    occ
        Occupation mask for the active band window, shape ``(nb,)`` with values
        in ``{0, 1}``.
    cell_volume
        Cell volume in atomic units.
    nk_tot
        Total number of k-points in the full Brillouin-zone average.
    source
        Human-readable source tag for diagnostics.

    Returns
    -------
    StaticHeadTerms
        Exact diagonal head pieces for:
        ``Sigma^X``, ``Sigma^SX``, ``Sigma^{SX-X}``, and ``Sigma^COH``.
    """

    occ_arr = jnp.asarray(occ, dtype=jnp.complex128)
    ones = jnp.ones_like(occ_arr, dtype=jnp.complex128)
    pref = jnp.asarray(1.0 / (float(cell_volume) * float(nk_tot)), dtype=jnp.complex128)

    v_h = jnp.asarray(vc0, dtype=jnp.complex128)
    w_h = jnp.asarray(wcoul0_static, dtype=jnp.complex128)
    wc_h = w_h - v_h

    sigma_x_diag = -(v_h * pref) * occ_arr
    sigma_sx_diag = -(w_h * pref) * occ_arr
    sigma_sx_minus_x_diag = -(wc_h * pref) * occ_arr
    sigma_coh_diag = 0.5 * (wc_h * pref) * ones

    return StaticHeadTerms(
        sigma_x_diag=sigma_x_diag,
        sigma_sx_diag=sigma_sx_diag,
        sigma_sx_minus_x_diag=sigma_sx_minus_x_diag,
        sigma_coh_diag=sigma_coh_diag,
        vc0=complex(vc0),
        wcoul0=complex(wcoul0_static),
        wc_head_0=complex(wcoul0_static) - complex(vc0),
        source=source,
    )


def compute_static_head_terms_from_sample(
    head: HeadSample,
    *,
    occ: np.ndarray | jnp.ndarray,
    cell_volume: float,
    nk_tot: int,
) -> StaticHeadTerms:
    """Build exact static COHSEX head terms from a resolved head sample."""

    return compute_static_head_terms(
        vc0=head.vc0,
        wcoul0_static=head.wcoul0,
        occ=occ,
        cell_volume=cell_volume,
        nk_tot=nk_tot,
        source=head.source,
    )


def format_static_head_diagnostics(head: StaticHeadTerms) -> str:
    """Return a concise summary of the exact static COHSEX head terms."""

    x_occ = _representative_entry(head.sigma_x_diag)
    sx_occ = _representative_entry(head.sigma_sx_diag)
    sxmx_occ = _representative_entry(head.sigma_sx_minus_x_diag)
    coh_all = _representative_entry(head.sigma_coh_diag)
    lines = [
        "",
        "-" * 72,
        "  STATIC HEAD TERMS (exact COHSEX / BGW-style)",
        "-" * 72,
        f"  Head source: {head.source}",
        f"  v_h(q→0)           = {head.vc0.real:12.6f} a.u.",
        f"  W_h(q→0, ω=0)      = {head.wcoul0.real:12.6f} a.u.",
        f"  W_h^c              = {head.wc_head_0.real:12.6f} a.u.",
        f"  Σ^X head (occ)     = {x_occ.real:12.6e} Ry",
        f"  Σ^SX head (occ)    = {sx_occ.real:12.6e} Ry",
        f"  Σ^(SX-X) head(occ) = {sxmx_occ.real:12.6e} Ry",
        f"  Σ^COH head (all)   = {coh_all.real:12.6e} Ry",
    ]
    return "\n".join(lines)


def expand_band_diagonal_to_kij(diag: jnp.ndarray, nk_tot: int) -> jnp.ndarray:
    """Broadcast a band-diagonal shift to a dense ``(nk, nb, nb)`` matrix."""

    diag_arr = jnp.asarray(diag, dtype=jnp.complex128)
    nb = int(diag_arr.shape[0])
    eye = jnp.eye(nb, dtype=jnp.complex128)
    one_k = eye[None, :, :] * diag_arr[None, :, None]
    return jnp.broadcast_to(one_k, (int(nk_tot), nb, nb))


def static_head_terms_to_kij(
    head: StaticHeadTerms,
    *,
    nk_tot: int,
    do_screened: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Expand exact static head shifts to dense ``(k, i, j)`` matrices.

    Parameters
    ----------
    head
        Exact static head terms from :func:`compute_static_head_terms`.
    nk_tot
        Total number of k-points in the full-zone average.
    do_screened
        If ``True``, return the screened-exchange head ``Sigma^SX``.
        If ``False``, return the bare-exchange head ``Sigma^X``.

    Returns
    -------
    sigma_sx_kij, sigma_coh_kij
        Dense diagonal matrices shaped ``(nk_tot, nb, nb)`` suitable for adding
        directly to the static COHSEX matrices in GWJAX.
    """

    sx_diag = head.sigma_sx_diag if do_screened else head.sigma_x_diag
    return (
        expand_band_diagonal_to_kij(sx_diag, nk_tot),
        expand_band_diagonal_to_kij(head.sigma_coh_diag, nk_tot),
    )


def compute_head_sigma_diagonal(
    head: HeadGNParams,
    energies_dft_ry: np.ndarray,
    occ: np.ndarray,
    cell_volume: float,
) -> np.ndarray:
    """Compute the simple on-shell diagonal head shift."""

    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return np.zeros_like(energies_dft_ry)
    occ_arr = np.asarray(occ, dtype=np.float64)
    return (head.R_h / (head.omega_h * cell_volume)) * (2.0 * occ_arr - 1.0)


def compute_head_sigma_at_omega(
    head: HeadGNParams,
    energies_dft_ry: np.ndarray,
    omega_eval_ry: np.ndarray,
    occ: np.ndarray,
    cell_volume: float,
) -> np.ndarray:
    """Compute the scalar head self-energy at arbitrary frequencies."""

    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return np.zeros_like(energies_dft_ry, dtype=np.complex128)

    eps = np.asarray(energies_dft_ry, dtype=np.float64)
    omega = np.asarray(omega_eval_ry, dtype=np.float64)
    f = np.asarray(occ, dtype=np.float64)
    eta = 1.0e-6
    occ_term = f / (omega - eps + head.omega_h - 1j * eta)
    emp_term = (1.0 - f) / (omega - eps - head.omega_h + 1j * eta)
    return (head.R_h / cell_volume) * (occ_term + emp_term)


def compute_ppm_head_sigma_kij(
    head: HeadGNParams,
    *,
    omega_grid_ry: np.ndarray,
    enk_ry: np.ndarray,
    efermi_ry: float,
    n_occ: int,
    cell_volume: float,
    nk_tot: int,
    eta: float = 1.0e-6,
) -> np.ndarray:
    """q→0, G=G'=0 head contribution to PPM ``Σ^c_kij(ω)``.

    At q=0, ``M_{nm}(k, q→0, G=0) = δ_{nm}``, so the head only enters the
    band-diagonal ``(i, i)`` of the PPM ``Σ^c`` matrix.  With the GN pole
    extracted in :func:`fit_head_gn` (``R_h = B_h / (2 Ω_h)``,
    ``B_h = -W^c(0) · Ω_h²``):

        Σ^c_n^head(ω - E_F) =
            +R_h / (V_cell · N_k) · [
                  f_n     / (ω - ε_n + Ω_h - iη)
                + (1-f_n) / (ω - ε_n - Ω_h + iη)
            ]

    where ω, ε_n are taken in the same E_F-relative convention (the difference
    ω - ε_n is invariant under that shift).  In the static limit ω → ε_n
    this reduces to ``-W^c(0) / (2 V_cell N_k)`` for occupied bands and
    ``+W^c(0) / (2 V_cell N_k)`` for empty bands, matching the COHSEX
    static-head pieces (``Σ^{SX-X} + Σ^COH``) built by
    :func:`compute_static_head_terms`.

    Parameters
    ----------
    head
        Fitted GN head pole.
    omega_grid_ry
        Σ^c frequency grid (relative to E_F), shape ``(n_omega,)`` in Ry.
    enk_ry
        Absolute band energies for the σ window, shape ``(nk, nb)`` in Ry.
    efermi_ry
        Fermi level in Ry (subtracted from ``enk_ry`` to get ``ε - E_F``).
    n_occ
        Number of occupied bands at the bottom of the σ window
        (``f_n = 1`` for ``n < n_occ``, else ``0``).
    cell_volume, nk_tot
        Unit-cell volume and full-zone k-point count.
    eta
        Imaginary regularization for the retarded poles.

    Returns
    -------
    sigma_kij : np.ndarray, shape ``(n_omega, nk, nb, nb)``, dtype complex128
        Diagonal-in-band head contribution; off-diagonals are zero.
    """

    omega = np.asarray(omega_grid_ry, dtype=np.float64).reshape(-1)
    enk = np.asarray(enk_ry, dtype=np.float64)
    if enk.ndim != 2:
        raise ValueError("enk_ry must be 2D (nk, nb)")
    n_omega = int(omega.size)
    nk, nb = enk.shape
    out = np.zeros((n_omega, nk, nb, nb), dtype=np.complex128)
    if abs(head.R_h) < 1.0e-30 or abs(head.omega_h) < 1.0e-30:
        return out

    eps_rel = enk - float(efermi_ry)                                # (nk, nb)
    f = np.zeros((nb,), dtype=np.float64)
    f[: max(0, min(int(n_occ), nb))] = 1.0
    delta = omega[:, None, None] - eps_rel[None, :, :]              # (nω, nk, nb)
    occ_term = f[None, None, :] / (delta + head.omega_h - 1j * eta)
    emp_term = (1.0 - f[None, None, :]) / (delta - head.omega_h + 1j * eta)
    sigma_diag = (head.R_h / (float(cell_volume) * float(nk_tot))) * (occ_term + emp_term)

    idx = np.arange(nb)
    out[:, :, idx, idx] = sigma_diag
    return out


def format_head_diagnostics(head: HeadGNParams, cell_volume: float) -> str:
    """Return a short multiline diagnostic summary for the scalar head fit."""

    lines = [
        "",
        "-" * 72,
        "  HEAD CORRECTION (scalar GN, separate from ISDF body)",
        "-" * 72,
        f"  v(q→0)             = {head.vc0:12.3f} a.u.",
        f"  W^c(q→0, ω=0)      = {head.wc_head_0:12.3f} a.u.",
        f"  W^c(q→0, ω=iωp)    = {head.wc_head_iwp:12.3f} a.u.  [ωp={head.omega_p:.4f} Ry]",
        f"  Ω_h²               = {head.omega_h_sq:12.6f} Ry²",
        f"  Ω_h                = {head.omega_h:12.6f} Ry  ({head.omega_h * _RY2EV:.6f} eV)",
        f"  B_h                = {head.B_h:12.6f} Ry² · a.u.",
        f"  R_h                = {head.R_h:12.6f} Ry · a.u.",
    ]
    if abs(head.omega_h) > 1.0e-30:
        lines.append(
            f"  R_h / (Ω_h · vol)  = {head.R_h / (head.omega_h * cell_volume):12.6e} (Ry)"
        )
    else:
        lines.append("  R_h / (Ω_h · vol)  = 0.0 (degenerate)")
    return "\n".join(lines)
