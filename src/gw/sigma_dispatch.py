"""Mode-orthogonal Σ_xc dispatch.

A single entry point :func:`compute_sigma_xc` that the QSGW iteration
map calls regardless of compute mode (X_ONLY, COHSEX, GN_PPM, HL_PPM).
The dispatch decides which Σ kernel runs internally; the iteration map
sees one signature and one result type.

Returned :class:`SigmaResult` always contains ``v_h_kij_ry``,
``sigma_x_kij_ry``, and a single ``sigma_xc_kij_ry`` representing the
total exchange-correlation contribution to ``H_QP = kin_ion + V_H +
Σ_xc``.  PPM-mode-only diagnostics (full ω-grid Σ_c, on-shell diagonals,
head decomposition) live as optional fields and are populated only when
the mode produces them.

This module owns *no compute* of its own — every kernel lives under
``cohsex_sigma`` (static channels), ``ppm_pipeline`` (dynamic Σ_c) or
``qsgw_utils`` (the QSGW Hermitisation).  It only orchestrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from .gw_config import ComputeMode


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SigmaResult:
    """Outputs of one full Σ pipeline call.

    Always populated
    ----------------
    v_h_kij_ry           : (nk, nb, nb)   Hartree (replicated)
    sigma_x_kij_ry       : (nk, nb, nb)   Bare exchange (replicated)
    sigma_xc_kij_ry      : (nk, nb, nb)   Exchange-correlation total going
                                          into ``H_QP = kin_ion + V_H + Σ_xc``.
                                          Static modes: Σ_SX + Σ_COH (with
                                          head).  PPM modes: Σ_x + Σ_c^QSGW.

    Static-mode-only (None in PPM)
    ------------------------------
    sigma_sx_kij_ry      : (nk, nb, nb)   Σ_SX with head
    sigma_coh_kij_ry     : (nk, nb, nb)   Σ_COH with head

    PPM-only (None in static)
    -------------------------
    sigma_c_omega_kij_ry      : (nω, nk, nb, nb), sharded P(None,None,'x','y')
                                Full ω-grid Σ_c (post-head); drives eqp1
                                Z-factor central difference.
    sigma_c_at_dft_diag_ev    : (nk, nb)  diag(Σ_c) at E_DFT (eV).
    omega_dft_rel_ev          : (nk, nb)  E_DFT − E_F (eV).
    omega_grid_ev             : (nω,)     ω-grid in eV.
    omega_grid_ry             : (nω,)     ω-grid in Ry.
    head_sigma_diag_w_kn_ry   : (nω, nk, nb)  PPM analytic head diagonal.
    sigma_omega_h5_path       : str       on-disk Σ_c(ω) HDF5 path.
    """

    v_h_kij_ry: jax.Array
    sigma_x_kij_ry: jax.Array
    sigma_xc_kij_ry: jax.Array
    sigma_sx_kij_ry: jax.Array | None = None
    sigma_coh_kij_ry: jax.Array | None = None
    sigma_c_omega_kij_ry: jax.Array | None = None
    sigma_c_at_dft_diag_ev: np.ndarray | None = None
    omega_dft_rel_ev: np.ndarray | None = None
    omega_grid_ev: np.ndarray | None = None
    omega_grid_ry: np.ndarray | None = None
    head_sigma_diag_w_kn_ry: np.ndarray | None = None
    sigma_omega_h5_path: str | None = None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def compute_sigma_xc(
    mode: ComputeMode,
    *,
    wfns,
    V_q: jax.Array,
    W_by_role: dict,
    e_qp_ev: np.ndarray | None,
    static_head_terms,
    head_resolver,
    quad,
    e_ref: float,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    band_slices,
    input_dir: str,
    print_fn: Callable = print,
) -> SigmaResult:
    """One-line entry point: build the full Σ_xc + V_H given the current
    wfn bundle and screened W's.

    Parameters
    ----------
    mode
        Compute-mode pivot.  Determines which Σ kernel chain runs and
        which roles in ``W_by_role`` are consulted.
    wfns
        ``Wavefunctions`` bundle in the *current* QP basis (or DFT basis
        for the iter-0 / one-shot call).
    V_q
        Bare Coulomb in flat-q ISDF basis.
    W_by_role
        Screened-Coulomb dict produced by
        :func:`gw.screening.compute_screening`, keyed by symbolic role.
        Conventional roles consumed here:

        * ``"static"`` — W(ω = 0).  Used by COHSEX (Σ_SX, Σ_COH) and as
          the ω-zero anchor for the PPM two-point fit.
        * ``"probe"``  — W at the GN/HL probe frequency.  Used by PPM
          for the second fit point.

        ``X_ONLY`` ignores ``W_by_role`` entirely.  Adding a new mode
        means picking the role labels it needs in
        :func:`gw.screening.screening_requests_for` and reading them
        here — no plumbing changes elsewhere.
    e_qp_ev
        Per-(k, n) QP energies (eV) used by the PPM QSGW build to evaluate
        Σ_c(E_m, E_n).  Required for PPM modes; ignored for static.
    static_head_terms, head_resolver
        q→0 head plumbing; ``static_head_terms`` is None when ``do_G0`` is
        false in the config.
    quad, e_ref
        Static minimax quadrature for χ₀; produced by
        ``w_isdf.build_static_quadrature`` once per W solve.
    config, meta, mesh_xy, sym, wfn, band_slices, input_dir
        Standard driver scaffolding.
    print_fn
        Rank-0-only print.

    Returns
    -------
    :class:`SigmaResult` populated per the mode.
    """
    from .cohsex_sigma import compute_cohsex_sigma, compute_v_h_sigma_x
    from .ppm_pipeline import compute_ppm_sigma_pipeline
    from .qsgw_utils import build_qsgw_sigma_xc

    # Static channels: sig_h (V_H) and sig_x (bare exchange) are needed
    # by every mode; sig_sx / sig_coh use W(ω=0) and only matter for
    # COHSEX.  Route to a separate top-level entry point for the
    # V-only path so PPM / X_ONLY modes never invoke the W-touching
    # kernels and the two paths each get their own jit-cached graph.
    W_static = W_by_role.get("static", V_q)
    if mode is ComputeMode.COHSEX:
        cohsex = compute_cohsex_sigma(
            wfns, V_q, W_static, meta, mesh_xy,
            Gij=None,                            # default DFT-occ projector
            do_screened=True,
            static_head_terms=static_head_terms,
            compute_bare_x=True,
        )
    else:
        cohsex = compute_v_h_sigma_x(
            wfns, V_q, meta, mesh_xy,
            Gij=None,
            static_head_terms=static_head_terms,
        )
    sig_h = cohsex["sig_h"]
    sig_x = cohsex["sig_x"]
    sig_sx = cohsex["sig_sx"]                    # zero placeholders for V-only path
    sig_coh = cohsex["sig_coh"]

    if mode is ComputeMode.X_ONLY:
        sigma_xc = sig_x
        return SigmaResult(
            v_h_kij_ry=sig_h,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=sigma_xc,
        )
    if mode is ComputeMode.COHSEX:
        sigma_xc = sig_sx + sig_coh
        return SigmaResult(
            v_h_kij_ry=sig_h,
            sigma_x_kij_ry=sig_x,
            sigma_xc_kij_ry=sigma_xc,
            sigma_sx_kij_ry=sig_sx,
            sigma_coh_kij_ry=sig_coh,
        )

    # Dynamic PPM modes: need W_static + W_probe.
    if e_qp_ev is None:
        raise ValueError(
            f"compute_sigma_xc: PPM mode {mode!r} requires e_qp_ev "
            "(QP energies for the QSGW Σ_c evaluation).")
    if "probe" not in W_by_role:
        raise KeyError(
            f"compute_sigma_xc: PPM mode {mode!r} requires "
            f"W_by_role['probe'] (set by screening_requests_for).")

    ppm_outputs = compute_ppm_sigma_pipeline(
        wfns=wfns,
        V_q=V_q,
        W_static_q=W_static, W_probe_q=W_by_role["probe"],
        sig_x=sig_x, sig_h=sig_h,
        quad=quad, e_ref=e_ref,
        config=config, meta=meta, mesh_xy=mesh_xy,
        head_resolver=head_resolver,
        band_slices=band_slices, wfn=wfn, sym=sym,
        input_dir=input_dir,
        print_fn=print_fn,
    )

    # QSGW Σ_xc^QSGW evaluated at e_qp_ev.  Static Σ_x is added inside
    # the kernel, so the result already includes Σ_x.
    omega_grid_ev = np.asarray(
        ppm_outputs.ppm_options.omega_grid_ev, dtype=np.float64)
    efermi_ry = float(wfn.efermi)
    e_qp_rel_ev = np.asarray(e_qp_ev, dtype=np.float64) - efermi_ry * RYD_TO_EV
    sig_x_rep = jax.device_put(jnp.asarray(sig_x),
        NamedSharding(mesh_xy, P(None, None, None)))
    sigma_xc_qsgw, qsgw_diag = build_qsgw_sigma_xc(
        ppm_outputs.sigma_c_omega, sig_x_rep,
        omega_grid_ev, e_qp_rel_ev, mesh_xy,
    )
    print_fn(f"  QSGW: {int(qsgw_diag['n_clipped'])} clipped "
             f"({100*qsgw_diag['frac_clipped']:.1f}%)")

    return SigmaResult(
        v_h_kij_ry=sig_h,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_qsgw,
        sigma_c_omega_kij_ry=ppm_outputs.sigma_c_omega,
        sigma_c_at_dft_diag_ev=ppm_outputs.sigma_c_at_dft_ev,
        omega_dft_rel_ev=ppm_outputs.omega_dft_rel_ev,
        omega_grid_ev=ppm_outputs.ppm_options.omega_grid_ev,
        omega_grid_ry=ppm_outputs.ppm_options.omega_grid_ry,
        head_sigma_diag_w_kn_ry=ppm_outputs.head_sigma_diag_w_kn_ry,
        sigma_omega_h5_path=ppm_outputs.sigma_omega_h5_path,
    )


__all__ = ["SigmaResult", "compute_sigma_xc"]
