"""Driver-level orchestration of the GN/HL plasmon-pole Σ^c(ω) path.

This module wires together the steps that ``gw_jax.main`` previously had
inlined as ~200 lines of bookkeeping:

    χ₀(probe ω) → W(probe ω) → 2-point PPM fit (B_q, Ω_q)
        → precompile + run Σ^c(ω, k, m, n)
        → analytic q→0 head injection
        → diag-Σ_c interpolation at DFT energies
        → write sigma_mnk.h5

The math kernels live in ``gw.ppm_sigma`` (per-τ kernel + accumulators)
and ``gw.head_correction`` (head fits + analytic head Σ).  This module
only sequences them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from common.units import RYD_TO_EV
from common.load_wfns import get_enk_bandrange
import common.timing as timing

from .gw_config import ComputeMode, LorraxConfig
from .gw_driver_helpers import (
    PPMSigmaRuntimeOptions,
    build_ppm_sigma_runtime_options,
    profile_section,
)
from .head_correction import HeadResolver
from .ppm_sigma import (
    compute_sigma_c_ppm_omega_grid,
    fit_ppm,
    precompile_sigma,
)
from .w_isdf import (
    build_imag_quadrature,
    build_real_quadrature,
    compute_chi0,
    solve_w,
)


@dataclass(frozen=True)
class PPMOutputs:
    """Frequency-dependent PPM Σ^c results returned to the GW driver."""

    sigma_c_omega: jax.Array | None        # (n_omega, nk, nb, nb)  Ry, or None if streamed
    sigma_c_at_dft_ev: np.ndarray | None   # (nk, nb)  diag(Σ_c) at E_DFT
    sigma_xc_at_dft_ev: np.ndarray | None  # (nk, nb)  diag(Σ_x) + diag(Σ_c) at E_DFT
    omega_dft_rel_ev: np.ndarray | None    # (nk, nb)  E_DFT - E_F  (eV)
    efermi_dft_ev: float | None
    sigma_omega_h5_path: str
    ppm_options: PPMSigmaRuntimeOptions


def _build_probe_quadrature(quad, config, *, print_fn):
    """Return (probe_omega, quad_probe) for the GN (imag) or HL (real) path."""
    is_hl = config.compute_mode is ComputeMode.HL_PPM
    if is_hl:
        probe_omega = complex(float(config.ppm_omega_p), 0.0)
        quad_probe = build_real_quadrature(
            quad, float(config.ppm_omega_p),
            config.minimax_config, print_fn=print_fn,
        )
    else:
        probe_omega = 1j * float(config.ppm_omega_p)
        quad_probe = build_imag_quadrature(
            quad, config.ppm_omega_p,
            config.minimax_config, print_fn=print_fn,
        )
    return probe_omega, quad_probe


def _fit_head_correction(
    head_resolver: HeadResolver, *,
    config: LorraxConfig,
    meta,
    probe_omega: complex,
    print_fn,
):
    """Fit the GN-PPM scalar head from the user-selected source."""
    from .head_correction import (
        fit_head_hl_analytic_from_sample,
        fit_head_ppm_from_samples,
        fit_head_with_fixed_omega_from_sample,
        format_head_diagnostics,
    )

    head_static = head_resolver.at(0.0 + 0.0j)
    omega_h_override = config.ppm_head_omega_h_ry
    is_hl = config.compute_mode is ComputeMode.HL_PPM

    if omega_h_override is not None:
        # User-supplied head pole Ω_h (e.g. BGW's analytic value).  Static
        # W^c(0) head is still LORRAX's — see fit_head_with_fixed_omega.
        head_gn = fit_head_with_fixed_omega_from_sample(
            head_static, omega_h_ry=float(omega_h_override))
        print_fn(
            f"  PPM head: Ω_h override = {float(omega_h_override):.6f} Ry "
            f"({float(omega_h_override) * RYD_TO_EV:.4f} eV)"
        )
    elif is_hl:
        # BGW-style analytic head pole: Ω_h² = ω_p² / (1 − ε_head⁻¹).
        # ω_p² = 16π · N_e / V_cell in Ry² (Hartree-AU energies → Ry²
        # has factor 4 → 16π).
        omega_p_sq_ry = 16.0 * float(np.pi) * float(meta.nelec) / float(meta.cell_volume)
        head_gn = fit_head_hl_analytic_from_sample(
            head_static, omega_p_sq_ry=omega_p_sq_ry)
        print_fn(
            f"  HL head: ω_p (analytic, BGW-style) = "
            f"{omega_p_sq_ry**0.5:.6f} Ry "
            f"({(omega_p_sq_ry**0.5) * RYD_TO_EV:.4f} eV)"
        )
    else:
        head_probe = head_resolver.at(probe_omega)
        head_gn = fit_head_ppm_from_samples(
            head_static, head_probe, probe_omega=probe_omega)

    print_fn(format_head_diagnostics(head_gn, cell_volume=meta.cell_volume))
    return head_gn


def _inject_analytic_head(
    sigma_c_omega: jax.Array | None, head_gn, *,
    ppm_options: PPMSigmaRuntimeOptions,
    band_slices,
    wfn, sym, meta,
    print_fn,
) -> jax.Array | None:
    """Add the analytic q→0, G=G'=0 head to Σ^c(ω) on rank 0."""
    if sigma_c_omega is None:
        return None
    from .head_correction import compute_ppm_head_sigma_kij

    enk_full, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    enk_full_np = np.asarray(enk_full, dtype=np.float64)
    n_occ = min(meta.nelec, enk_full_np.shape[1])
    vbm_ry = float(np.max(enk_full_np[:, :n_occ]))
    cbm_ry = (
        float(np.min(enk_full_np[:, n_occ:]))
        if n_occ < enk_full_np.shape[1] else vbm_ry
    )
    efermi_ry = 0.5 * (vbm_ry + cbm_ry)

    head_sigma_kij_ry = compute_ppm_head_sigma_kij(
        head_gn,
        omega_grid_ry=np.asarray(ppm_options.omega_grid_ry, dtype=np.float64),
        enk_ry=enk_full_np,
        efermi_ry=efermi_ry,
        n_occ=n_occ,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
    )
    head_max_ev = float(np.max(np.abs(head_sigma_kij_ry))) * RYD_TO_EV
    on_shell_occ = (
        -head_gn.R_h
        / (head_gn.omega_h * meta.cell_volume * meta.nk_tot)
        * RYD_TO_EV
    )
    print_fn(
        f"  Σ_c head shift: max|Σ^head_diag| = {head_max_ev:.4f} eV "
        f"(on-shell occ band → {on_shell_occ:+.4f} eV)"
    )
    return sigma_c_omega + jnp.asarray(head_sigma_kij_ry, dtype=jnp.complex128)


def _eval_sigma_c_at_dft_energies(
    sigma_c_omega: jax.Array | None,
    sigma_omega: 'object', *,                          # noqa: F821 (forward decl)
    ppm_options: PPMSigmaRuntimeOptions,
    sig_x: jax.Array,
    band_slices, wfn, sym, meta,
    print_fn,
):
    """On rank 0, interpolate diag(Σ_c)(ω) at each DFT energy.

    Returns ``(sigma_c_at_dft_ev, sigma_xc_at_dft_ev, omega_dft_rel_ev,
    efermi_dft_ev)``.  Off-rank-0 returns ``(None, None, None, None)``.
    """
    if meta.rank != 0:
        return None, None, None, None

    import h5py

    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range,
        band_slices.sigma_range, nspinor=meta.nspinor)
    enk_dft_ev = np.asarray(enk_dft) * RYD_TO_EV
    n_occ = min(meta.nelec, enk_dft_ev.shape[1])
    vbm_ev = float(np.max(enk_dft_ev[:, :n_occ]))
    cbm_ev = (
        float(np.min(enk_dft_ev[:, n_occ:]))
        if n_occ < enk_dft_ev.shape[1] else vbm_ev
    )
    efermi_dft_ev = 0.5 * (vbm_ev + cbm_ev)
    omega_dft_rel_ev = enk_dft_ev - efermi_dft_ev
    print_fn(
        f"  E_F(midgap) = {efermi_dft_ev:.6f} eV  "
        f"(VBM={vbm_ev:.6f}, CBM={cbm_ev:.6f})"
    )

    omega_ev = np.asarray(ppm_options.omega_grid_ev, dtype=np.float64)
    if sigma_c_omega is not None:
        sig_c_diag = np.diagonal(
            np.asarray(sigma_c_omega), axis1=2, axis2=3) * RYD_TO_EV
    else:
        with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5:
            sig_c_diag = np.diagonal(
                np.asarray(h5["sigma_c_kij_ry"], dtype=np.complex128),
                axis1=2, axis2=3,
            ) * RYD_TO_EV

    nk, nb = sig_c_diag.shape[1], sig_c_diag.shape[2]
    sigma_c_at_dft_ev = np.array([
        [
            complex(
                np.interp(omega_dft_rel_ev[ik, ib], omega_ev,
                          np.real(sig_c_diag[:, ik, ib])),
                np.interp(omega_dft_rel_ev[ik, ib], omega_ev,
                          np.imag(sig_c_diag[:, ik, ib])),
            )
            for ib in range(nb)
        ]
        for ik in range(nk)
    ], dtype=np.complex128)

    sig_x_diag_ev = np.diagonal(np.asarray(sig_x), axis1=1, axis2=2) * RYD_TO_EV
    sigma_xc_at_dft_ev = sig_x_diag_ev + sigma_c_at_dft_ev
    return (
        sigma_c_at_dft_ev,
        sigma_xc_at_dft_ev,
        omega_dft_rel_ev,
        efermi_dft_ev,
    )


def _write_sigma_omega_h5(
    sigma_c_omega: jax.Array | None,
    sigma_omega: 'object', *,                          # noqa: F821
    ppm_options: PPMSigmaRuntimeOptions,
    sig_x: jax.Array,
    sig_h: jax.Array,
    config: LorraxConfig,
    input_dir: str,
    meta,
    mesh_xy,
) -> str:
    """Write the canonical sigma_mnk.h5 file (one writer, two backends)."""
    import os
    from file_io import (
        copy_sigma_kij_h5_to_omega_h5,
        write_sigma_omega_h5,
    )

    out_path = config.sigma_omega_h5_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(input_dir, out_path)

    if sigma_c_omega is not None:
        # SlabIO handles rank-0 dispatch internally; both backends need
        # all ranks to enter, so no ``if rank == 0`` guard.
        write_sigma_omega_h5(
            out_path, ppm_options.omega_grid_ev, None,
            sigma_c_kij_ev=RYD_TO_EV * sigma_c_omega,
            sigma_sx_kij_ev=RYD_TO_EV * sig_x,
            hartree_kij_ev=RYD_TO_EV * sig_h,
            mesh=mesh_xy,
            use_ffi_io=bool(config.use_ffi_io),
        )
    elif meta.rank == 0 and sigma_omega.sigma_kij_h5_path:
        copy_sigma_kij_h5_to_omega_h5(
            sigma_omega.sigma_kij_h5_path,
            out_path,
            ppm_options.omega_grid_ev,
            sigma_sx_kij_ev=sig_x,
            hartree_kij_ev=sig_h,
            omega_batch_size=ppm_options.sigma_omega_batch_size,
        )
    return out_path


def compute_ppm_sigma_pipeline(
    *,
    wfns,
    V_q: jax.Array,
    W_q: jax.Array,
    sig_x: jax.Array,
    sig_h: jax.Array,
    quad,
    e_ref,
    config: LorraxConfig,
    meta,
    mesh_xy,
    head_resolver: HeadResolver,
    band_slices,
    wfn,
    sym,
    input_dir: str,
    print_fn=print,
) -> PPMOutputs:
    """Run the GN/HL-PPM dynamic Σ^c(ω) pipeline end-to-end.

    Sequences (with timing.section + xprof annotations):

        1. Build the probe-frequency quadrature (HL: real ω; GN: iω_p).
        2. Compute χ₀(probe) and W(probe) via the static screening solver.
        3. Two-point PPM pole fit (B_q, Ω_q).
        4. Precompile + run Σ^c(ω, k, m, n) over the windowed minimax grid.
        5. Inject the analytic q→0 head correction.
        6. Interpolate diag(Σ_c) at DFT energies (rank-0 only).
        7. Write sigma_mnk.h5 (eV units).
    """
    from . import w_isdf  # for ensure_compilation_cache + cache hit timings

    if not config.do_screened:
        raise ValueError("PPM Σ^c pipeline requires do_screened=true.")
    if config.self_consistent:
        raise NotImplementedError(
            "PPM Σ^c pipeline does not yet support self_consistent=true."
        )

    ppm_options = build_ppm_sigma_runtime_options(config, input_dir=input_dir)
    label = "HL-PPM" if config.compute_mode is ComputeMode.HL_PPM else "GN-PPM"
    from .gw_output import print_section
    print_section(f"{label} + FREQUENCY-INTEGRATED SIGMA", print_fn)

    with timing.section("gw_jax.ppm_sigma"):
        # Step 1–2: probe-frequency W
        probe_omega, quad_probe = _build_probe_quadrature(
            quad, config, print_fn=print_fn)
        chi0_probe = compute_chi0(
            wfns, quad_probe, meta, mesh_xy, energy_reference=e_ref)
        # Block BEFORE solve_w so chi0_probe compute time isn't folded
        # into the W-solve wall, and so solve_w can donate χ₀'s buffer.
        chi0_probe.block_until_ready()
        Wiwp_q = solve_w(
            V_q, chi0_probe, meta, mesh_xy,
            memory_mode=config.isdf_memory_mode,
        )
        del chi0_probe
        Wiwp_q.block_until_ready()

        # Step 3: PPM pole fit
        ppm = fit_ppm(
            W_q, Wiwp_q, V_q, probe_omega, mesh_xy,
            fallback_omega=config.ppm_fallback_omega,
            n_nodes_static=quad.node_count,
            print_fn=print_fn,
            model_label=label,
        )
        del Wiwp_q

        # Step 4: precompile + run Σ^c(ω, k, m, n)
        with timing.section("sigma.compile"):
            precompile_sigma(wfns, ppm, meta, mesh_xy)
        with timing.section("sigma.exec"), profile_section("sigma_ppm", print_fn=print_fn):
            sigma_omega = compute_sigma_c_ppm_omega_grid(
                wfns, ppm, meta, mesh_xy, ppm_options,
                sigma_window_quad=config.sigma_quadrature_config,
                print_fn=print_fn,
            )
        sigma_c_omega = sigma_omega.sigma_c_kij  # None if streamed

        # Step 5: q→0 head injection (analytic, mini-BZ-averaged)
        head_gn = _fit_head_correction(
            head_resolver, config=config, meta=meta,
            probe_omega=probe_omega, print_fn=print_fn,
        )
        sigma_c_omega = _inject_analytic_head(
            sigma_c_omega, head_gn,
            ppm_options=ppm_options, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta, print_fn=print_fn,
        )

        # Step 6: diag(Σ_c) at DFT energies (rank-0 only)
        (sigma_c_at_dft_ev,
         sigma_xc_at_dft_ev,
         omega_dft_rel_ev,
         efermi_dft_ev) = _eval_sigma_c_at_dft_energies(
            sigma_c_omega, sigma_omega,
            ppm_options=ppm_options, sig_x=sig_x,
            band_slices=band_slices, wfn=wfn, sym=sym, meta=meta,
            print_fn=print_fn,
        )

        # Step 7: write sigma_mnk.h5
        sigma_omega_h5_path = _write_sigma_omega_h5(
            sigma_c_omega, sigma_omega,
            ppm_options=ppm_options, sig_x=sig_x, sig_h=sig_h,
            config=config, input_dir=input_dir,
            meta=meta, mesh_xy=mesh_xy,
        )

    return PPMOutputs(
        sigma_c_omega=sigma_c_omega,
        sigma_c_at_dft_ev=sigma_c_at_dft_ev,
        sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
        omega_dft_rel_ev=omega_dft_rel_ev,
        efermi_dft_ev=efermi_dft_ev,
        sigma_omega_h5_path=sigma_omega_h5_path,
        ppm_options=ppm_options,
    )
