"""Driver-level orchestration of the multipole (MPA) Σ^c(ω) path.

    staged fit store → per-pole Σ^c(ω, k, m, n) pass loop
        → q→0 head from the store's own pole axis
        → diag-Σ_c interpolation at DFT energies
        → write sigma_mnk.h5

WHY A MODULE OF ITS OWN, AND NOT A SECOND FUNCTION IN ``ppm_pipeline``.
Three reasons, in the order they matter.

The first is that ``ppm_pipeline`` refuses this mode by name at its own
entry, and says why in as many words: everything in that file — the
probe frequency, the head fit, the printed label — reads the compute
mode as "HL, or else GN", so a mode with no pole model is turned away
at one seam.  Putting a second pipeline behind that refusal would make
the file contradict its own entry condition.

The second is that the two schemes' Σ stages share NO kernel.  The
two-point path fits two W samples to a real pole and integrates it; this
one reads complex poles off disk and integrates them one pass at a time
through ``gw.mpa.sigma_pass``, whose whole apparatus (the width split,
the Laplace bucketing, the composite complex-pole rules) the two-point
path does not import.  ``tests/test_mpa_sigma_costs_gnppm_nothing.py``
turns that separation into a gate — the two-point path must not so much
as IMPORT the complex-pole routing — and a shared module would put the
import one attribute lookup away from it.

The third is what is genuinely shared, which is the part AFTER the cube
exists: the head injection mechanics (``qsgw_utils.add_band_diag``), the
at-DFT interpolation and the writer.  Those are imported here and called
verbatim, so steps 3–5 have one implementation between the two schemes
and only the Σ_c producer and the head PRODUCER differ.

WHAT THIS MODULE REFUSES, AND WHERE
------------------------------------
Both of the store properties that kept ``compute_mode = mpa`` on
``gw_config.UNIMPLEMENTED_MODES`` are still refused, by the code that
owns each of them and not by a second copy here:

* a WEDGE-SHAPED pole axis — ``sigma_pass.refuse_wedge_pole_slab``,
  because unfolding a pole field is not the operation that unfolds W and
  the conjugation on the time-reversed members is uncertified;
* a HEADLESS store — ``mpa_store.read_head_poles``, because a Σ_c with
  no q→0 term is finite, smooth and wrong by ~200 meV on silicon.

This module calls the head reader FIRST, before the pass loop, for one
reason: the pass loop is the expensive stage and the head refusal is a
property of the file that can be established in milliseconds.  A run
that is going to be refused for having no head should be refused before
it integrates fourteen passes, not after.  ``sigma_pass`` still makes
its own wedge refusal inside the loop where the extent is read; neither
refusal is weakened by the other existing.
"""

from __future__ import annotations

import os

import numpy as np

import common.timing as timing
from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange

from .gw_config import LorraxConfig
from .ppm_pipeline import (
    PPMOutputs,
    _eval_sigma_c_at_dft_energies,
    _write_sigma_omega_h5,
)

#: How far the stored pole model may miss the stored samples before the
#: head is refused, as a RELATIVE residual on ``W_c``.  Loose on purpose:
#: this gate is not a convergence check on the fit (the store carries its
#: own backward error for that) but a check that the poles and the samples
#: beside them came from ONE fit and are being read in the spelling the
#: producer wrote.  The failures it is aimed at miss by orders of
#: magnitude — a stored ``W`` instead of ``W - v`` is out by ``v_head``,
#: ~3300 Ry on silicon; a sign or factor-of-two error in the model is out
#: by O(1) — so a threshold anywhere in this decade separates them from a
#: merely imperfect fit.
HEAD_SAMPLE_REL_TOL = 1.0e-6


def _resolve_fit_store(config: LorraxConfig, input_dir: str) -> str:
    """The deck's ``mpa_fit_file``, resolved and required to exist.

    Refused HERE as well as at config parse time, and the two are not
    redundant: the parse-time refusal is what saves the operator's
    allocation, this one is what makes any other caller of the pipeline
    (a test, a future driver, the SC map) safe without having to remember
    that the first one exists.
    """
    from .gw_config import refuse_missing_mpa_fit_store

    path = refuse_missing_mpa_fit_store(config, context="compute_mpa_sigma_pipeline")
    if not os.path.isabs(path):
        path = os.path.join(input_dir, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"compute_mpa_sigma_pipeline: mpa_fit_file = {path!r} does not "
            f"exist.  The multipole Σ reads its screening from this store "
            f"and from nowhere else — there is no screening-stage W it "
            f"could fall back to, and falling back to the bare Coulomb "
            f"would return an unscreened Σ_c under the multipole name.")
    return path


def _inject_mpa_head(
    sigma_c_omega, head, *,
    config: LorraxConfig,
    band_slices,
    wfn, sym, meta,
    pole_energy_unit: str,
    print_fn,
):
    """Add the q→0, G=G'=0 head built from the store's own pole axis.

    THE SAME SHAPE AS ``ppm_pipeline._inject_analytic_head``, and
    deliberately so: the band window, the canonical mid-gap E_F, the
    occupation split and the layout-aware add are taken from the same
    places by the same calls.  What differs is one line — the producer of
    the ``(nω, nk, nb)`` diagonal, which is ``gw.mpa.sigma_head`` here and
    a two-point Godby-Needs fit there.

    Returns ``(sigma_c_omega_with_head, head_sigma_diag_w_kn_ry)``.
    """
    from .mpa.sigma_head import (
        head_poles_in_ry, head_sample_residual, mpa_head_sigma_diag)
    from .qsgw_utils import add_band_diag

    # THE GATE THAT CAN FAIL, run before anything is built from the poles.
    # See ``sigma_head.head_sample_residual`` for what it does and does
    # NOT pin — in particular it is blind to a common rescaling of the
    # pole axis, which is why the f-sum line below is printed beside it.
    resid = head_sample_residual(head, pole_energy_unit=pole_energy_unit)
    if resid["max_rel_residual"] > HEAD_SAMPLE_REL_TOL:
        raise ValueError(
            f"compute_mpa_sigma_pipeline: the fit store's head poles do not "
            f"reproduce the head samples stored beside them — worst "
            f"relative residual {resid['max_rel_residual']:.3e} over "
            f"{resid['n_samples']} samples, against a tolerance of "
            f"{HEAD_SAMPLE_REL_TOL:.1e}.  The store keeps the 2*n_p samples "
            f"WITH the n_p poles precisely so this can be asked; a miss "
            f"this large is not a poorly converged fit, it is the two "
            f"halves of the head axis not belonging to each other, or "
            f"head_w holding W rather than the correlation part W - v.  "
            f"Rebuild the head leg (gw.mpa.head_dipole.build_head_axis + "
            f"mpa_store.write_head_axis) rather than loosening this.")
    if resid["z0_index"] is not None:
        print_fn(
            f"  MPA head gate: model(z=0) = {resid['z0_model'].real:.6f} Ry "
            f"against the stored sample {resid['z0_sample'].real:.6f} Ry, "
            f"|Δ| = {resid['z0_residual']:.3e} Ry "
            f"({resid['z0_rel_residual']:.3e} relative); worst over all "
            f"{resid['n_samples']} samples "
            f"{resid['max_rel_residual']:.3e} relative")
    else:
        print_fn(
            f"  MPA head gate: this sample grid carries no exact z = 0 "
            f"(the metals protocol shifts the origin to i*1e-5), so the "
            f"gate is the whole axis: worst relative residual "
            f"{resid['max_rel_residual']:.3e} over {resid['n_samples']} "
            f"samples")
    _announce_head_fsum(head, meta=meta,
                        pole_energy_unit=pole_energy_unit, print_fn=print_fn)

    enk_full, _ = get_enk_bandrange(
        wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
        nspinor=meta.nspinor)
    enk_full_np = np.asarray(enk_full, dtype=np.float64)
    efermi_ry = float(wfn.efermi)
    n_occ = min(meta.nelec, enk_full_np.shape[1])

    head_diag_w_kn_ry = mpa_head_sigma_diag(
        head,
        omega_grid_ry=np.asarray(config.omega_grid_ry, dtype=np.float64),
        enk_ry=enk_full_np,
        efermi_ry=efermi_ry,
        n_occ=n_occ,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
        pole_energy_unit=pole_energy_unit,
    )
    Om_ry, _ = head_poles_in_ry(head, pole_energy_unit=pole_energy_unit)
    head_max_ev = float(np.max(np.abs(head_diag_w_kn_ry))) * RYD_TO_EV
    print_fn(
        f"  Σ_c head shift: max|Σ^head_diag| = {head_max_ev:.4f} eV "
        f"from {Om_ry.size} head poles of set '{head['label']}', "
        f"Re Ω_p = "
        + ", ".join(f"{float(np.real(w)) * RYD_TO_EV:.4f}" for w in Om_ry[:4])
        + (" ..." if Om_ry.size > 4 else "") + " eV "
        f"(poles read as {pole_energy_unit})")
    return (add_band_diag(sigma_c_omega, head_diag_w_kn_ry),
            head_diag_w_kn_ry)


def _announce_head_fsum(head, *, meta, pole_energy_unit, print_fn):
    """Print the f-sum residual — THE gate that is sensitive to the unit.

    ``sum_p 2 Omega_p B_p = v_head * omega_p**2`` scales as the SQUARE of
    a common rescaling of the pole axis, so it separates a Hartree pole
    set from a Rydberg one by a factor of four, which the sample residual
    cannot do at all (see ``sigma_head.head_sample_residual``).

    ANNOUNCED AND NOT REFUSED, because a large residual here has two
    causes and this function cannot tell them apart: the fit failing to
    capture the leading asymptote (a method statement) and the deck's
    transition manifold not saturating the sum rule (a deck statement).
    The committed Si fixture over-saturates by 1.63 for a reason that is
    known, measured and owner-scoped — a sign on the nonlocal velocity
    commutator in ``common.mtxel_sweep.dipole_operator``, written up in
    ``gw.mpa.head_dipole.head_fsum_from_transitions`` — so a refusal here
    would refuse the anchor deck for somebody else's defect.  A factor of
    four is a different order of number from 1.63 and is what a reader
    should look for.
    """
    vhead = head.get("vhead", None)
    if vhead is None:
        print_fn(
            "  MPA head f-sum: the store carries no mpa_head_vhead, so the "
            "unit-sensitive check is unavailable (the sample residual above "
            "is blind to a common rescaling of the pole axis)")
        return
    try:
        from .mpa.head_dipole import fsum_residual
        from .mpa.sigma_head import head_poles_in_ry

        Om, Bp = head_poles_in_ry(head, pole_energy_unit=pole_energy_unit)
        rep = fsum_residual(Om, Bp, complex(vhead),
                            float(meta.nelec), float(meta.cell_volume))
    except Exception as exc:                    # pragma: no cover - advisory
        print_fn(f"  MPA head f-sum: unavailable ({type(exc).__name__}: {exc})")
        return
    print_fn(
        f"  MPA head f-sum (set '{head['label']}', poles read as "
        f"{pole_energy_unit}): "
        f"Σ_p 2Ω_pB_p = {rep['fit_sum_2_Omega_B'].real:.6e} against "
        f"v_head·ω_p² = {rep['v_head_times_omega_p_squared'].real:.6e} Ry³, "
        f"relative residual {rep['rel_residual']:.4f}.  This one scales as "
        f"the SQUARE of the pole-axis unit, so ~4x (or ~1/4) is the "
        f"signature of a Ha/Ry mix-up; O(1) is the deck's f-sum saturation")


def compute_mpa_sigma_pipeline(
    *,
    wfns,
    sig_x,
    sig_h,
    config: LorraxConfig,
    meta,
    mesh_xy,
    band_slices,
    wfn,
    sym,
    input_dir: str,
    fit_store_path: str | None = None,
    write_sigma_omega_h5: bool = True,
    print_fn=print,
) -> PPMOutputs:
    """Run the multipole Σ^c(ω) pipeline against a staged fit store.

    Mirrors :func:`gw.ppm_pipeline.compute_ppm_sigma_pipeline` steps 2–5
    and has no step 1: the pole fit is not part of this stage.  The MPA
    scheme's screening lives ENTIRELY in the fit store — ``2*n_p`` complex
    samples of W, fitted to ``n_p`` complex poles, written by the fit
    driver — so this pipeline takes no ``W_by_role`` and asks the
    screening stage for nothing.

        2. ``sigma_pass.compute_mpa_sigma_c_omega_grid`` — one pass per
           pole, ascending, through the same device τ loop, accumulator
           and sharded tile sink the two-point path uses.
        3. q→0 head injection from the store's ``__mpahead`` pole axis.
        4. Interpolate diag(Σ_c) at the DFT energies (replicated).
        5. Write sigma_mnk.h5 (eV units).

    ``fit_store_path`` overrides the deck key; it exists for tests and for
    a caller that has already resolved the path, and ``None`` (the
    default) means "ask the deck", which is the production route.
    """
    from file_io import mpa_store
    from .gw_output import print_section
    from .mpa.sigma_pass import compute_mpa_sigma_c_omega_grid

    if not config.do_screened:
        raise ValueError(
            "compute_mpa_sigma_pipeline: the multipole Σ^c is a SCREENED "
            "self-energy — its poles are a fit of W_c — so do_screened=false "
            "and compute_mode=mpa are contradictory instructions rather than "
            "a cheaper run.")

    # THE POLE MODEL IS THE ENTRY CONDITION, read the other way round from
    # ``compute_ppm_sigma_pipeline``'s.  That pipeline refuses a mode with
    # no two-point pole model; this one refuses a mode that HAS one,
    # because a GN or HL mode reaching here would have its Σ_c built from
    # whatever fit store the deck happened to name — a complete, finite
    # answer for a screening it never computed.
    if config.compute_mode.ppm_model is not None:
        raise NotImplementedError(
            f"compute_mpa_sigma_pipeline: compute_mode = "
            f"{config.compute_mode.value} is a two-point plasmon-pole model, "
            f"and its Σ stage is gw.ppm_pipeline.compute_ppm_sigma_pipeline. "
            f"Running it here would build its Σ_c from a multipole fit store "
            f"that its own screening never produced.")

    # SELF-CONSISTENCY IS REFUSED, BY NAME, AND THE REASON IS NOT
    # PLUMBING.  The QSGW loop rotates the band basis and re-solves W from
    # the rotated orbitals each iteration (``sc_iteration`` calls
    # ``compute_screening`` inside the map).  This scheme's W is not
    # solved in the loop at all — it is a fit read off disk — so an SC run
    # would rotate ψ every iteration against a screening frozen at the DFT
    # orbitals and converge, smoothly, to the fixed point of a
    # self-energy nobody asked for.  Nothing about the shapes would
    # object.  Closing it means re-fitting the store per iteration, which
    # is a campaign and not a branch.
    from .gw_config import QPSolver

    if config.qp_solver is QPSolver.SELF_CONSISTENT:
        raise NotImplementedError(
            "compute_mpa_sigma_pipeline: qp_solver = self_consistent with "
            "compute_mode = mpa is refused.  The QSGW loop re-solves W from "
            "the rotated orbitals every iteration; the multipole poles are "
            "fitted once, before the run, against the DFT screening, and "
            "this pipeline has no way to refit them in the loop.  An SC run "
            "would therefore iterate ψ against a frozen W and converge to "
            "the fixed point of a self-energy that is neither G0W0 nor "
            "QSGW, with no shape or finiteness check able to see it.  Use "
            "qp_solver = one_shot_dft or fixed_point.")

    path = fit_store_path
    if path is None:
        path = _resolve_fit_store(config, input_dir)

    print_section("MULTIPOLE (MPA) FREQUENCY-INTEGRATED SIGMA", print_fn)

    with timing.section("gw_jax.mpa_sigma"):
        # THE HEAD IS READ FIRST, and its refusal is the cheap one.  A
        # store with no ``__mpahead`` group is refused BY NAME here
        # (``read_head_poles``), in milliseconds, rather than after the
        # pass loop has integrated every pole.  The wedge refusal stays
        # inside the pass loop, where the extent it compares against is
        # read.
        with timing.section("mpa.head_read"):
            head = mpa_store.read_head_poles(
                path, label=(config.mpa_head_label or None))
        ledger = mpa_store.fit_completion_ledger(path)
        # THE HEAD SET IS NAMED IN THE LOG, ALWAYS, and not only when it
        # is unusual.  A store may carry several q→0 head sets — the
        # velocity-commutator sign is an open owner decision and both
        # conventions are fitted side by side — and they differ by ~3 eV
        # in the head pole.  A head number that cannot be attributed to a
        # convention is the thing that arrangement exists to prevent, so
        # the label and its group go in the same line as the store path.
        print_fn(
            f"  MPA fit store: {path}\n"
            f"    {ledger['n_p']} poles x {ledger['n_q']} q x "
            f"{ledger['n_mu']} centroids, complete={ledger['complete']}, "
            f"worst backward error {ledger['backward_error_max']}\n"
            f"    q→0 head set '{head['label']}' (group {head['group']}): "
            f"{head['n_p']} poles fitted to "
            f"{np.asarray(head['z']).size} samples, written "
            f"{head['written_utc']}")

        # Step 2: the per-pole Σ^c(ω, k, m, n) pass loop.  ``quad`` is the
        # Σ-side quadrature config (``config.sigma_quadrature_config``),
        # the same object step 2 of the two-point pipeline passes — NOT
        # the χ₀ minimax quadrature, which this scheme never builds
        # because it never solves a χ₀.
        with timing.section("sigma.exec"):
            sigma_c_omega, records = compute_mpa_sigma_c_omega_grid(
                wfns, path, meta, mesh_xy,
                ppm_cfg=config.ppm,
                quad=config.sigma_quadrature_config,
                omega_grid_ry=config.omega_grid_ry,
                print_fn=print_fn,
            )

        # Step 3: q→0 head injection from the store's own pole axis.
        sigma_c_omega, head_sigma_diag_w_kn_ry = _inject_mpa_head(
            sigma_c_omega, head,
            config=config, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta,
            pole_energy_unit=str(config.mpa_pole_energy_unit),
            print_fn=print_fn,
        )

        # Step 4: diag(Σ_c) at DFT energies — the two-point path's own
        # function, called verbatim.  It reads the cube, the ω grid and
        # ``sig_x``, none of which know which scheme produced them.
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

        # Step 5: write sigma_mnk.h5 — likewise verbatim.
        if write_sigma_omega_h5:
            sigma_omega_h5_path = _write_sigma_omega_h5(
                sigma_c_omega,
                sig_x=sig_x, sig_h=sig_h,
                config=config, input_dir=input_dir,
                meta=meta, mesh_xy=mesh_xy,
                sym=sym, print_fn=print_fn,
            )
        else:
            sigma_omega_h5_path = config.paths.sigma_omega_h5_file
            if not os.path.isabs(sigma_omega_h5_path):
                sigma_omega_h5_path = os.path.join(
                    input_dir, sigma_omega_h5_path)

    del records          # the pass report is printed by the pass loop
    return PPMOutputs(
        sigma_c_omega=sigma_c_omega,
        sigma_c_at_dft_ev=sigma_c_at_dft_ev,
        sigma_xc_at_dft_ev=sigma_xc_at_dft_ev,
        omega_dft_rel_ev=omega_dft_rel_ev,
        efermi_dft_ev=efermi_dft_ev,
        sigma_omega_h5_path=sigma_omega_h5_path,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
    )
