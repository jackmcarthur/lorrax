"""Driver-level orchestration of the GN/HL plasmon-pole Σ^c(ω) path.

This module wires together the steps that ``gw_jax.main`` previously had
inlined as ~200 lines of bookkeeping:

    χ₀(probe ω) → W(probe ω) → 2-point PPM fit (B_q, Ω_q)
        → precompile + run Σ^c(ω, k, m, n)
        → analytic q→0 head construction

The math kernels live in ``gw.ppm_sigma`` (per-τ kernel + accumulators)
and ``gw.head_correction`` (head fits + analytic head Σ).  This module
only sequences them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange
import common.timing as timing

from .band_extrapolation import (
    fit_band_extrapolation,
    format_extrapolation_report,
    plan_band_brackets,
)
from .gw_config import LorraxConfig
from .head_correction import HeadResolver
from .ppm_sigma import (
    compute_sigma_c_ppm_omega_grid,
    fit_ppm,
)
from .ppm_tau_kernel import precompile_sigma


@dataclass(frozen=True)
class PPMOutputs:
    """Ansatz-specific PPM outputs handed to the dynamic finalizer."""

    sigma_c_body_omega: jax.Array          # (n_omega, nk, nb, nb)  Ry
    # Kept separate: injection is shared by every dynamic ansatz in
    # ``dynamic_sigma.add_head_sigma_diag``.
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
    # ``ppm_model``, not ``is HL_PPM``: the final arm of this chain is the
    # GN two-point head fit, so reading the mode as a boolean "is it HL"
    # hands GN's fit to every non-HL mode there will ever be.  The pipeline
    # entry (``compute_ppm_sigma_pipeline``) has already refused a mode
    # with no pole model, so 'gn' / 'hl' are the only two values here.
    is_hl = config.compute_mode.ppm_model == "hl"

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


def _compute_analytic_head_diag(
    head_gn, *,
    config: LorraxConfig,
    band_slices,
    wfn, sym, meta,
    print_fn,
) -> np.ndarray:
    """Compute the analytic q→0, G=G'=0 PPM head diagonal.

    A head-less Σ_c is a silent wrong answer (Bug B,
    reports/sigma_ppm_tighten_2026-07-04).  Injection is deliberately left
    to the ansatz-neutral dynamic-Sigma finalizer.
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

    # The head is band-diagonal; compute that lossless (nω, nk, nb)
    # representation once.  The neutral injection seam avoids a dense head
    # entirely for the sharded layout.
    head_sigma_diag_ry = compute_ppm_head_sigma_diag(
        head_gn,
        omega_grid_ry=np.asarray(config.omega_grid_ry, dtype=np.float64),
        enk_ry=enk_full_np,
        efermi_ry=efermi_ry,
        n_occ=n_occ,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
    )

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
    return np.asarray(head_sigma_diag_ry)


def _band_count_point(cube, i: int):
    """``cube[i]`` with the TRAILING (ω, k, m, n) sharding preserved.

    The Σ cube's leading axis is the band count and is replicated, so
    dropping it is shard-local — but ``sigma_omega_layout=sharded``'s whole
    contract is that consumers read the layout off the array itself
    (``qsgw_utils.is_band_sharded_sigma_omega``), and a bare ``cube[i]``
    leaves that to XLA's propagation through a slice+reshape.  Restate it.
    """
    sharding = getattr(cube, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return cube[i]
    spec = tuple(sharding.spec)
    if len(spec) != int(getattr(cube, "ndim", 0)):
        return cube[i]
    out = NamedSharding(sharding.mesh, P(*spec[1:]))
    return jax.jit(lambda a: a[i], out_shardings=out)(cube)


def _report_band_extrapolation(
    sigma_omega, head_sigma_diag_w_kn_ry, *,
    plan, config, band_slices, wfn, sym, meta, mesh_xy, print_fn,
) -> None:
    """Log the three band-count Σ_c's, the fit and its diagnostics.

    Reads the band DIAGONAL of each cumulative point at the SAME external
    evaluation energy — E_DFT − E_F, on the SAME ω grid — so nothing but the
    band count differs between the three.  The analytic q→0 head is added to
    every point identically (it is a band-diagonal ω-dependent term with no
    unoccupied-state sum, hence bracket-independent), so the reported values
    are the physical Σ_c rather than the body alone; being a common offset it
    shifts S_∞ and S₃ together and leaves ``Δ_tail`` unchanged.

    Σ_c only.  Σ_x is a sum over OCCUPIED states and has no slow unoccupied
    tail; extrapolating Σ_total would fit a constant as if it converged.
    """
    from .qsgw_utils import extract_sigma_diag_replicated, interp_along_omega

    cube = sigma_omega.sigma_c_kij
    enk_dft, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    enk_ev = np.asarray(enk_dft) * RYD_TO_EV
    omega_eval_ev = enk_ev - float(wfn.efermi) * RYD_TO_EV
    omega_grid_ev = np.asarray(config.omega_grid_ev, dtype=np.float64)
    head = (None if head_sigma_diag_w_kn_ry is None
            else np.asarray(head_sigma_diag_w_kn_ry))

    points = []
    for i in range(cube.shape[0]):
        diag_w_kn = np.asarray(
            extract_sigma_diag_replicated(_band_count_point(cube, i), mesh_xy))
        if head is not None:
            diag_w_kn = diag_w_kn + head
        # Ry -> eV here, exactly where ``eval_sigma_c_at_dft_energies`` does
        # it, so the reported numbers are in the same unit as every other Σ
        # line in the log and the formatter needs no scale of its own.
        points.append(interp_along_omega(
            diag_w_kn * RYD_TO_EV, omega_grid_ev, omega_eval_ev))
    fit = fit_band_extrapolation(sigma_omega.band_counts, np.stack(points))

    # THE STATES A GW RUN IS FOR.  The band edges, located from the actual
    # eigenvalues over the QP window rather than assumed to sit at index
    # n_occ-1 / n_occ (spin-orbit doubling and the k-star ordering both move
    # them).  An aggregate over the whole QP window is reported too, labelled
    # as the envelope it is: Σ_c at the top of the window is the largest and
    # least converged quantity in the run, and a max is its number, not the
    # calculation's.
    n_occ = int(band_slices.b2 - band_slices.b0)
    occ, unocc = enk_ev[:, :n_occ], enk_ev[:, n_occ:]
    states = []
    if occ.size:
        kv, nv = np.unravel_index(int(np.argmax(occ)), occ.shape)
        states.append((f"VBM  k={kv} n={nv}  E={occ[kv, nv]:.4f} eV",
                       (int(kv), int(nv))))
    if unocc.size:
        kc, nc = np.unravel_index(int(np.argmin(unocc)), unocc.shape)
        states.append((f"CBM  k={kc} n={nc + n_occ}  "
                       f"E={unocc[kc, nc]:.4f} eV",
                       (int(kc), int(nc + n_occ))))
    print_fn(format_extrapolation_report(plan, fit, states=states))


def compute_ppm_sigma_pipeline(
    *,
    wfns,
    V_q: jax.Array,
    W_static_q: jax.Array,
    W_probe_q: jax.Array,
    quad,
    config: LorraxConfig,
    meta,
    mesh_xy,
    head_resolver: HeadResolver,
    band_slices,
    wfn,
    sym,
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
        3. Construct the analytic q→0 head correction.

    The ansatz-neutral finalizer injects that head, interpolates, writes and
    builds the QSGW matrix.
    """
    if not config.do_screened:
        raise ValueError("PPM Σ^c pipeline requires do_screened=true.")

    # THE POLE MODEL IS THE ENTRY CONDITION.  This module is the two-point
    # plasmon-pole fit and everything below it — the probe frequency, the
    # head fit, the printed label — reads the mode as "HL, or else GN".
    # A mode with no pole model at all (MPA, and anything after it) must
    # therefore be turned away HERE, at one seam, rather than collecting a
    # refusal at each of those three reads.  ``sigma_dispatch`` refuses it
    # before this call; this is the invariant restated where it is relied
    # upon, for the benefit of any other caller.
    ppm_model = config.compute_mode.ppm_model
    if ppm_model is None:
        raise NotImplementedError(
            f"compute_ppm_sigma_pipeline: compute_mode = "
            f"{config.compute_mode.value} is not a plasmon-pole model, so "
            f"the two-point PPM Σ^c pipeline is not its Σ stage.  Running it "
            f"anyway would fit two W samples with a GN pole and report the "
            f"result as this mode's Σ_c(ω).")

    label = "HL-PPM" if ppm_model == "hl" else "GN-PPM"
    from .gw_output import print_section
    print_section(f"{label} + FREQUENCY-INTEGRATED SIGMA", print_fn)

    with timing.section("gw_jax.ppm_sigma"):
        # Probe frequency for the PPM fit — recovered from the configured
        # ω_p (real-axis Ω for HL, iω_p for GN).  The screening planner
        # used the same convention to pick W_probe_q's evaluation point.
        is_hl = ppm_model == "hl"
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
        #
        # The band-bracket plan is resolved HERE, once, before anything is
        # compiled: it fixes the kernel's G-build count, the AOT signature
        # and the Σ cube's leading extent, so it must be the same object all
        # three see.  ``sigma_band_extrapolation = false`` (the default)
        # gives the trivial one-bracket plan and the whole path below is
        # bit-identical to the un-bracketed code.
        s = wfns.slices
        plan = plan_band_brackets(
            enabled=bool(config.sigma.band_extrapolation),
            enk_ry=np.asarray(wfns.enk[:, s.full]),
            n_occ=int(s.b2 - s.b0),
            nb_logical=int(meta.b_id_4_user or s.b4) - int(s.b0),
            nb_padded=int(s.nb_full),
        )
        if plan.enabled:
            print_fn(
                f"  Σc band extrapolation: ON — {plan.n_brackets} disjoint "
                f"band brackets {plan.bounds} against ONE W(τ) per τ; "
                f"band counts {plan.counts} (requested {plan.requested}).")
            # Emitted HERE and not only in the report block at the end: a
            # planner fallback is a fact about the run that the operator
            # should see before Σ is spent, not after.
            for note in plan.notes:
                print_fn(f"  Σc band extrapolation: {note}")
        with timing.section("sigma.compile"):
            precompile_sigma(wfns, ppm, meta, mesh_xy, brackets=plan.bounds)
        with timing.section("sigma.exec"):
            sigma_omega = compute_sigma_c_ppm_omega_grid(
                wfns, ppm, meta, mesh_xy,
                ppm_cfg=config.ppm,
                sigma_cfg=config.sigma,
                quad=config.sigma_quadrature_config,
                omega_grid_ry=config.omega_grid_ry,
                plan=plan,
                print_fn=print_fn,
            )
        # THE BLAST RADIUS STOPS HERE.  ``sigma_omega.sigma_c_kij`` carries
        # the leading band-count axis; everything downstream of this line —
        # the head injection, the eqp interpolation, sigma_mnk.h5, the QSGW
        # build — is shared with MPA and COHSEX and is deliberately left at
        # the shape it has always had.  The last element IS the ordinary
        # full-band Σ_c (the cumulative sum's final term), so at
        # n_bracket = 1 this index is the identity.
        sigma_c_body_omega = _band_count_point(
            sigma_omega.sigma_c_kij, sigma_omega.sigma_c_kij.shape[0] - 1)

        # Step 3: q→0 head construction (analytic, mini-BZ-averaged)
        head_gn = _fit_head_correction(
            head_resolver, config=config, meta=meta,
            probe_omega=probe_omega, print_fn=print_fn,
        )
        head_sigma_diag_w_kn_ry = _compute_analytic_head_diag(
            head_gn,
            config=config, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta,
            print_fn=print_fn,
        )

        # Step 4: the band-convergence extrapolation report.  After the head,
        # because the head is part of the Σ_c being reported; before the
        # return, because the cube's leading axis does not survive it.
        if plan.enabled:
            _report_band_extrapolation(
                sigma_omega, head_sigma_diag_w_kn_ry,
                plan=plan, config=config, band_slices=band_slices,
                wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh_xy,
                print_fn=print_fn,
            )

    return PPMOutputs(
        sigma_c_body_omega=sigma_c_body_omega,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
    )
