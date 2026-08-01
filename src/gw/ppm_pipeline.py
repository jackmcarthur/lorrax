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
from common.jax_profile import profile_section
from .head_correction import HeadResolver
from .ppm_sigma import (
    compute_sigma_c_ppm_omega_grid,
    fit_ppm,
)
from .ppm_tau_kernel import precompile_sigma


@dataclass(frozen=True)
class PPMOutputs:
    """Frequency-dependent PPM Σ^c results returned to the GW driver."""

    sigma_c_omega: jax.Array | None        # (n_omega, nk, nb, nb)  Ry
    sigma_c_at_dft_ev: np.ndarray | None   # (nk, nb)  diag(Σ_c) at E_DFT
    sigma_xc_at_dft_ev: np.ndarray | None  # (nk, nb)  diag(Σ_x) + diag(Σ_c) at E_DFT
    omega_dft_rel_ev: np.ndarray | None    # (nk, nb)  E_DFT - E_F  (eV)
    efermi_dft_ev: float | None
    sigma_omega_h5_path: str
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
    sigma_c_omega: jax.Array, head_gn, *,
    config: LorraxConfig,
    band_slices,
    wfn, sym, meta,
    print_fn,
) -> tuple[jax.Array, np.ndarray | None]:
    """Add the analytic q→0, G=G'=0 head to Σ^c(ω).

    One head, added to the in-memory tensor (replicated add, or a
    rank-local band-diagonal add on the sharded layout).  A head-less
    Σ_c is a silent wrong answer (Bug B,
    reports/sigma_ppm_tighten_2026-07-04); the streamed (kij_stream)
    h5-RMW arm of this function was REMOVED 2026-07-31 with the mode.

    Returns
    -------
    sigma_c_omega_with_head, head_sigma_diag_w_kn_ry
        Post-head Σ_c (same shape as input) and the band-diagonal of the
        head-only contribution ``(nω, nk, nb)`` in Ry (head is diagonal
        in band so this is a lossless decomposition).
    """
    from .head_correction import compute_ppm_head_sigma_diag

    enk_full, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    enk_full_np = np.asarray(enk_full, dtype=np.float64)
    # Canonical mid-gap E_F from WFNReader (computed once at WFN load
    # over the full set of bands stored in WFN.h5; band-window-independent).
    efermi_ry = float(wfn.efermi)
    n_occ = min(meta.nelec, enk_full_np.shape[1])

    # The head is band-DIAGONAL; compute that (nω, nk, nb) representation
    # once.  The dense (nω, nk, nb, nb) tensor — n_ω·nk·nb²·16 B per rank,
    # a full-cube-sized transient — is embedded from it ONLY on the path
    # that adds densely (the in-memory replicated add); the
    # sharded layout injects the diagonal rank-locally and never
    # materializes the dense head anywhere.  The dense embed below is
    # bit-identical to the historical compute_ppm_head_sigma_kij output
    # (that function now embeds this same array — single source of truth).
    head_sigma_diag_ry = compute_ppm_head_sigma_diag(
        head_gn,
        omega_grid_ry=np.asarray(config.omega_grid_ry, dtype=np.float64),
        enk_ry=enk_full_np,
        efermi_ry=efermi_ry,
        n_occ=n_occ,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
    )

    def _embed_dense(diag_w_kn: np.ndarray) -> np.ndarray:
        n_w, nk_h, nb_h = diag_w_kn.shape
        dense = np.zeros((n_w, nk_h, nb_h, nb_h), dtype=np.complex128)
        idx = np.arange(nb_h)
        dense[:, :, idx, idx] = diag_w_kn
        return dense

    # max|dense| == max|diag| (off-diagonals are exact zeros; |diag| >= 0),
    # so the diagnostic is unchanged on every path.
    head_max_ev = float(np.max(np.abs(head_sigma_diag_ry))) * RYD_TO_EV
    on_shell_occ = (
        -head_gn.R_h
        / (head_gn.omega_h * meta.cell_volume * meta.nk_tot)
        * RYD_TO_EV
    )
    print_fn(
        f"  Σ_c head shift: max|Σ^head_diag| = {head_max_ev:.4f} eV "
        f"(on-shell occ band → {on_shell_occ:+.4f} eV)"
    )
    head_diag_w_kn_ry = np.asarray(head_sigma_diag_ry)


    from .qsgw_utils import add_band_diag_sharded, is_band_sharded_sigma_omega
    if is_band_sharded_sigma_omega(sigma_c_omega):
        # Sharded layout (sigma_omega_layout=sharded): rank-local add of the
        # band-diagonal head onto each rank's (m_X, n_Y) tile — zero
        # communication, no dense head anywhere.  Element-for-element the
        # same IEEE add as the dense path performs on the diagonal (its
        # off-diagonal adds are exact +0.0).
        return (
            add_band_diag_sharded(sigma_c_omega, head_diag_w_kn_ry),
            head_diag_w_kn_ry,
        )

    return (
        sigma_c_omega + jnp.asarray(_embed_dense(head_diag_w_kn_ry),
                                    dtype=jnp.complex128),
        head_diag_w_kn_ry,
    )


def _eval_sigma_c_at_dft_energies(
    sigma_c_omega: jax.Array, *,
    config: LorraxConfig,
    sig_x: jax.Array,
    band_slices, wfn, sym, meta, mesh_xy,
    print_fn,
):
    """Interpolate diag(Σ_c)(ω) at each DFT energy on all ranks.

    Pulls the replicated diagonal of Σ_c(ω, k, m, n) via
    :func:`qsgw_utils.extract_sigma_diag_replicated` (cheap allgather of
    the diagonal only, ~MB) so the result is consistent across ranks —
    required by the post-Σ flow in ``gw_jax`` which now runs on all
    ranks.

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

    omega_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
    from .qsgw_utils import extract_sigma_diag_replicated
    sig_c_diag = (
        np.asarray(extract_sigma_diag_replicated(sigma_c_omega, mesh_xy))
        * RYD_TO_EV)

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
    sigma_c_omega: jax.Array, *,
    sig_x: jax.Array,
    sig_h: jax.Array,
    config: LorraxConfig,
    input_dir: str,
    meta,
    mesh_xy,
) -> str:
    """Write the canonical sigma_mnk.h5 file (one writer, two backends)."""
    import os
    from file_io import write_sigma_omega_h5

    out_path = config.paths.sigma_omega_h5_file
    if not os.path.isabs(out_path):
        out_path = os.path.join(input_dir, out_path)

    from .qsgw_utils import is_band_sharded_sigma_omega
    if is_band_sharded_sigma_omega(sigma_c_omega):
        # Sharded layout: derive the eV tensors with the OUTPUT sharding
        # pinned to the cube's own (m_X, n_Y) tiling, so the partitioner
        # cannot resolve the sharded+replicated elementwise mix by
        # gathering the cube (pattern #4: make the constraint
        # structural).  Same expression, same operand order as the
        # writer's own derivation on the replicated path:
        # total = (Ry→eV Σ_c + Ry→eV Σ_x[None]) + Ry→eV V_H[None] —
        # bit-identical elementwise.  SlabIO's per-rank hyperslab
        # writers (PHDF5_FFI / PHDF5_HOST) then write each rank's tile
        # directly; the h5py_allgather backend is refused at P>1 for
        # this layout at driver start.
        shd = sigma_c_omega.sharding

        def _ev_tensors(c_ry, x_ry, h_ry):
            c_ev = RYD_TO_EV * c_ry
            total = (c_ev + (RYD_TO_EV * x_ry)[None, ...]) \
                + (RYD_TO_EV * h_ry)[None, ...]
            return total, c_ev

        total_ev, sigma_c_ev = jax.jit(
            _ev_tensors, out_shardings=(shd, shd))(
                sigma_c_omega, sig_x, sig_h)
        write_sigma_omega_h5(
            out_path, config.omega_grid_ev, total_ev,
            sigma_c_kij_ev=sigma_c_ev,
            sigma_sx_kij_ev=RYD_TO_EV * sig_x,
            hartree_kij_ev=RYD_TO_EV * sig_h,
            mesh=mesh_xy,
            backend=config.backend.slab_io,
        )
        return out_path
    # SlabIO handles rank-0 dispatch internally; both backends need
    # all ranks to enter, so no ``if rank == 0`` guard.
    write_sigma_omega_h5(
        out_path, config.omega_grid_ev, None,
        sigma_c_kij_ev=RYD_TO_EV * sigma_c_omega,
        sigma_sx_kij_ev=RYD_TO_EV * sig_x,
        hartree_kij_ev=RYD_TO_EV * sig_h,
        mesh=mesh_xy,
        backend=config.backend.slab_io,
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
            n_mu_logical=int(meta.n_rmu),
        )

        # Step 2: precompile + run Σ^c(ω, k, m, n)
        with timing.section("sigma.compile"):
            precompile_sigma(wfns, ppm, meta, mesh_xy)
        with timing.section("sigma.exec"), profile_section("sigma_ppm", print_fn=print_fn):
            sigma_omega = compute_sigma_c_ppm_omega_grid(
                wfns, ppm, meta, mesh_xy,
                ppm_cfg=config.ppm,
                quad=config.sigma_quadrature_config,
                omega_grid_ry=config.omega_grid_ry,
                print_fn=print_fn,
            )
        sigma_c_omega = sigma_omega.sigma_c_kij

        # Step 3: q→0 head injection (analytic, mini-BZ-averaged)
        head_gn = _fit_head_correction(
            head_resolver, config=config, meta=meta,
            probe_omega=probe_omega, print_fn=print_fn,
        )
        sigma_c_omega, head_sigma_diag_w_kn_ry = _inject_analytic_head(
            sigma_c_omega, head_gn,
            config=config, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta,
            print_fn=print_fn,
        )

        # Step 4: diag(Σ_c) at DFT energies (replicated across ranks)
        (sigma_c_at_dft_ev,
         sigma_xc_at_dft_ev,
         omega_dft_rel_ev,
         efermi_dft_ev) = _eval_sigma_c_at_dft_energies(
            sigma_c_omega,
            config=config, sig_x=sig_x,
            band_slices=band_slices, wfn=wfn, sym=sym, meta=meta,
            mesh_xy=mesh_xy,
            print_fn=print_fn,
        )

        # Step 5: write sigma_mnk.h5 (skipped on intermediate SC iters
        # — the SC driver writes once at convergence using the captured
        # sigma_c_omega from the final SigmaResult).
        if write_sigma_omega_h5:
            sigma_omega_h5_path = _write_sigma_omega_h5(
                sigma_c_omega,
                sig_x=sig_x, sig_h=sig_h,
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
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
    )
