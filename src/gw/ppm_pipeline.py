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
from common.wfn_transforms import get_enk_bandrange
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
    # Diagonal of the analytic q→0 head added to ``sigma_c_omega`` — kept
    # separately for diagnostic printing (the head is band-diagonal so the
    # decomposition is lossless).  ``None`` when no head was injected.
    head_sigma_diag_w_kn_ry: np.ndarray | None = None


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
    omega_h_override = config.ppm.head_omega_h_ry
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
    sigma_kij_h5_path: str | None,
    print_fn,
) -> tuple[jax.Array | None, np.ndarray | None]:
    """Add the analytic q→0, G=G'=0 head to Σ^c(ω) on rank 0.

    Two accumulation paths, one head.  When ``sigma_c_omega`` is in memory
    (KIJ_HOST) the head is added to the tensor and returned.  When Σ_c was
    streamed to an h5 (KIJ_STREAM) the streamed accumulator wrote a
    *head-less* ``sigma_c_kij_ry``; here we RMW-add the identical analytic
    head into that dataset on rank 0 so both paths land the same Σ_c
    (Bug B, reports/sigma_ppm_tighten_2026-07-04).  The head is computed
    once (same call, same ω-grid) and consumed by whichever path applies.

    Returns
    -------
    sigma_c_omega_with_head, head_sigma_diag_w_kn_ry
        Post-head Σ_c (same shape as input, or ``None`` in streamed mode
        where the correction lands in the h5 instead) and the band-diagonal
        of the head-only contribution ``(nω, nk, nb)`` in Ry (head is
        diagonal in band so this is a lossless decomposition).
    """
    from .head_correction import compute_ppm_head_sigma_kij

    enk_full, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    enk_full_np = np.asarray(enk_full, dtype=np.float64)
    # Canonical mid-gap E_F from WFNReader (computed once at WFN load
    # over the full set of bands stored in WFN.h5; band-window-independent).
    efermi_ry = float(wfn.efermi)
    n_occ = min(meta.nelec, enk_full_np.shape[1])

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
    head_diag_w_kn_ry = np.diagonal(
        np.asarray(head_sigma_kij_ry), axis1=2, axis2=3)

    if sigma_c_omega is None:
        # Streamed mode: RMW-add the head into the sigma_kij h5.  A head-less
        # streamed Σ_c is a silent wrong answer, so refuse to proceed without
        # a file to correct rather than dropping the head.
        if sigma_kij_h5_path is None:
            raise ValueError(
                "streamed Σ_c requested (sigma_c_kij is None) but no "
                "sigma_kij h5 path is available; the analytic q→0 head "
                "cannot be injected and a head-less Σ_c would be silently "
                "wrong. Set sigma_kij_h5_file, or use "
                "sigma_omega_accumulation=kij (in-memory).")
        if meta.rank == 0:
            import h5py
            head_np = np.asarray(head_sigma_kij_ry, dtype=np.complex128)
            n_omega = int(head_np.shape[0])
            # The head is diagonal-in-band; head_np's off-diagonals are exactly
            # zero, so adding the full (nω, nk, nb, nb) tile touches only the
            # band-diagonal — bit-identical to the in-memory path's tensor add.
            # ω-batch the read-modify-write to match the chunked dataset.
            batch = int(max(1, getattr(ppm_options, 'sigma_omega_batch_size', 4)))
            with h5py.File(sigma_kij_h5_path, "r+") as h5:
                if bool(h5.attrs.get("head_injected", False)):
                    print_fn(
                        f"  Σ_c head: {sigma_kij_h5_path} already carries "
                        "head_injected=True; skipping RMW to avoid double-add.")
                else:
                    dset = h5["sigma_c_kij_ry"]
                    for ibeg in range(0, n_omega, batch):
                        iend = min(ibeg + batch, n_omega)
                        dset[ibeg:iend] = dset[ibeg:iend] + head_np[ibeg:iend]
                    h5.attrs["head_injected"] = True
        return None, head_diag_w_kn_ry

    return (
        sigma_c_omega + jnp.asarray(head_sigma_kij_ry, dtype=jnp.complex128),
        head_diag_w_kn_ry,
    )


def _eval_sigma_c_at_dft_energies(
    sigma_c_omega: jax.Array | None,
    sigma_omega: 'object', *,                          # noqa: F821 (forward decl)
    ppm_options: PPMSigmaRuntimeOptions,
    sig_x: jax.Array,
    band_slices, wfn, sym, meta, mesh_xy,
    print_fn,
):
    """Interpolate diag(Σ_c)(ω) at each DFT energy on all ranks.

    Pulls the replicated diagonal of Σ_c(ω, k, m, n) via
    :func:`qsgw_utils.extract_sigma_diag_replicated` (cheap allgather of
    the diagonal only, ~MB) so the result is consistent across ranks —
    required by the post-Σ flow in ``gw_jax`` which now runs on all
    ranks.  The streamed-mode fallback (``sigma_c_omega is None``) uses
    a single rank-0 h5 read followed by an MPI broadcast.

    Returns ``(sigma_c_at_dft_ev, sigma_xc_at_dft_ev, omega_dft_rel_ev,
    efermi_dft_ev)``, all replicated.
    """
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range,
        band_slices.sigma_range, nspinor=meta.nspinor)
    enk_dft_ev = np.asarray(enk_dft) * RYD_TO_EV
    # Single source of truth for the mid-gap E_F: WFNReader computes it
    # once at WFN load (``wfn.efermi`` in Ry).  Don't recompute here.
    vbm_ev = float(wfn.vbm) * RYD_TO_EV
    cbm_ev = float(wfn.cbm) * RYD_TO_EV
    efermi_dft_ev = float(wfn.efermi) * RYD_TO_EV
    omega_dft_rel_ev = enk_dft_ev - efermi_dft_ev
    print_fn(
        f"  E_F(midgap) = {efermi_dft_ev:.6f} eV  "
        f"(VBM={vbm_ev:.6f}, CBM={cbm_ev:.6f})"
    )

    omega_ev = np.asarray(ppm_options.omega_grid_ev, dtype=np.float64)
    if sigma_c_omega is not None:
        from .qsgw_utils import extract_sigma_diag_replicated
        sig_c_diag = (
            np.asarray(extract_sigma_diag_replicated(sigma_c_omega, mesh_xy))
            * RYD_TO_EV)
    else:
        # Streamed mode: rank-0 reads, then broadcast to all ranks.
        import h5py
        if meta.rank == 0:
            with h5py.File(sigma_omega.sigma_kij_h5_path, "r") as h5:
                sig_c_diag = np.diagonal(
                    np.asarray(h5["sigma_c_kij_ry"], dtype=np.complex128),
                    axis1=2, axis2=3,
                ) * RYD_TO_EV
        else:
            sig_c_diag = np.zeros(
                (omega_ev.size, *enk_dft_ev.shape), dtype=np.complex128)
        from jax.experimental.multihost_utils import broadcast_one_to_all
        sig_c_diag = np.asarray(broadcast_one_to_all(sig_c_diag))

    from .qsgw_utils import interp_along_omega
    sigma_c_at_dft_ev = interp_along_omega(
        sig_c_diag, omega_ev, omega_dft_rel_ev)

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

    out_path = config.paths.sigma_omega_h5_file
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
            backend=config.backend.slab_io,
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
    W_static_q: jax.Array,
    W_probe_q: jax.Array,
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
    write_sigma_omega_h5: bool = True,
    print_fn=print,
) -> PPMOutputs:
    """Run the GN/HL-PPM dynamic Σ^c(ω) pipeline given pre-computed W's.

    Both ``W_static_q`` (W at ω=0) and ``W_probe_q`` (W at the GN-PPM
    iω_p / HL-PPM Ω) must be supplied by the caller.  In the SC
    iteration map the caller is :func:`gw.screening.compute_screening`
    which evaluates them once per iteration; in one-shot main() the
    same helper is invoked at the screening seam.  Decoupling the
    probe-frequency χ₀+W solve from this pipeline lets future Σ
    schemes (CD, spectral, …) share the same screening planner.

    Sequences (with timing.section + xprof annotations):

        1. Two-point PPM pole fit (B_q, Ω_q) from (W_static, W_probe).
        2. Precompile + run Σ^c(ω, k, m, n) over the windowed minimax grid.
        3. Inject the analytic q→0 head correction.
        4. Interpolate diag(Σ_c) at DFT energies (rank-0 only).
        5. Write sigma_mnk.h5 (eV units).
    """
    from . import w_isdf  # for ensure_compilation_cache + cache hit timings

    if not config.do_screened:
        raise ValueError("PPM Σ^c pipeline requires do_screened=true.")

    ppm_options = build_ppm_sigma_runtime_options(config, input_dir=input_dir)
    label = "HL-PPM" if config.compute_mode is ComputeMode.HL_PPM else "GN-PPM"
    from .gw_output import print_section
    print_section(f"{label} + FREQUENCY-INTEGRATED SIGMA", print_fn)

    with timing.section("gw_jax.ppm_sigma"):
        # Probe frequency for the PPM fit — recovered from the configured
        # ω_p (real-axis Ω for HL, iω_p for GN).  The screening planner
        # used the same convention to pick W_probe_q's evaluation point.
        is_hl = config.compute_mode is ComputeMode.HL_PPM
        probe_omega = (
            complex(float(config.ppm.omega_p), 0.0) if is_hl
            else 1j * float(config.ppm.omega_p)
        )

        # Step 1: PPM pole fit
        ppm = fit_ppm(
            W_static_q, W_probe_q, V_q, probe_omega, mesh_xy,
            fallback_omega=config.ppm.fallback_omega,
            n_nodes_static=quad.node_count,
            print_fn=print_fn,
            model_label=label,
        )

        # Step 2: precompile + run Σ^c(ω, k, m, n)
        with timing.section("sigma.compile"):
            precompile_sigma(wfns, ppm, meta, mesh_xy)
        with timing.section("sigma.exec"), profile_section("sigma_ppm", print_fn=print_fn):
            sigma_omega = compute_sigma_c_ppm_omega_grid(
                wfns, ppm, meta, mesh_xy, ppm_options,
                sigma_window_quad=config.sigma_quadrature_config,
                print_fn=print_fn,
            )
        sigma_c_omega = sigma_omega.sigma_c_kij  # None if streamed

        # Step 3: q→0 head injection (analytic, mini-BZ-averaged)
        head_gn = _fit_head_correction(
            head_resolver, config=config, meta=meta,
            probe_omega=probe_omega, print_fn=print_fn,
        )
        sigma_c_omega, head_sigma_diag_w_kn_ry = _inject_analytic_head(
            sigma_c_omega, head_gn,
            ppm_options=ppm_options, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta,
            sigma_kij_h5_path=sigma_omega.sigma_kij_h5_path,
            print_fn=print_fn,
        )

        # Step 4: diag(Σ_c) at DFT energies (replicated across ranks)
        (sigma_c_at_dft_ev,
         sigma_xc_at_dft_ev,
         omega_dft_rel_ev,
         efermi_dft_ev) = _eval_sigma_c_at_dft_energies(
            sigma_c_omega, sigma_omega,
            ppm_options=ppm_options, sig_x=sig_x,
            band_slices=band_slices, wfn=wfn, sym=sym, meta=meta,
            mesh_xy=mesh_xy,
            print_fn=print_fn,
        )

        # Step 5: write sigma_mnk.h5 (skipped on intermediate SC iters
        # — the SC driver writes once at convergence using the captured
        # sigma_c_omega from the final SigmaResult).
        if write_sigma_omega_h5:
            sigma_omega_h5_path = _write_sigma_omega_h5(
                sigma_c_omega, sigma_omega,
                ppm_options=ppm_options, sig_x=sig_x, sig_h=sig_h,
                config=config, input_dir=input_dir,
                meta=meta, mesh_xy=mesh_xy,
            )
        else:
            import os
            sigma_omega_h5_path = config.paths.sigma_omega_h5_file
            if not os.path.isabs(sigma_omega_h5_path):
                sigma_omega_h5_path = os.path.join(
                    input_dir, sigma_omega_h5_path)

    return PPMOutputs(
        sigma_c_omega=sigma_c_omega,
        sigma_c_at_dft_ev=sigma_c_at_dft_ev,
        sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
        omega_dft_rel_ev=omega_dft_rel_ev,
        efermi_dft_ev=efermi_dft_ev,
        sigma_omega_h5_path=sigma_omega_h5_path,
        ppm_options=ppm_options,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
    )
