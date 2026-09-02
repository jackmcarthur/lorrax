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
import os

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange
from common.collectives import barrier, process_rank
import common.timing as timing

from .band_extrapolation import (
    BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT,
    SpectralShellExtrapolationFailed,
    build_band_ladder,
    fit_band_extrapolation_spectral,
    format_spectral_report,
    plane_wave_band_count,
    spectral_h5_payload,
    assert_brackets_match_ols_abscissae,
    extrapolation_h5_payload,
    extrapolation_weights,
    fit_band_extrapolation,
    format_extrapolation_report,
    static_limit_tail_ruling,
    plan_band_brackets,
    sc_tolerance_ruling,
)
from .gw_config import LorraxConfig
from .head_correction import HeadResolver
from .ppm_sigma import (
    compute_sigma_c_ppm_omega_grid,
    fit_ppm,
)


@dataclass(frozen=True)
class PPMOutputs:
    """Ansatz-specific PPM outputs handed to the dynamic finalizer."""

    sigma_c_body_omega: jax.Array          # (n_omega, nk, nb, nb)  Ry
    # Kept separate: injection is shared by every dynamic ansatz in
    # ``dynamic_sigma.add_head_sigma_diag``.
    head_sigma_diag_w_kn_ry: np.ndarray | None = None
    # The band-extrapolation fit, as ``{"arrays": {...}, "attrs": {...}}``
    # ready for ``sigma_mnk.h5``.  None when the feature is off, which is
    # what keeps the default write path byte-identical.  It rides HERE
    # rather than being written inside the pipeline because the Σ file is
    # created once, by the shared dynamic finalizer, and a second writer
    # for one dataset group is a second place for the star extraction and
    # the k stamp to disagree.
    band_extrapolation: dict | None = None
    # The UN-EXTRAPOLATED N₃ body cube, present only when the feature is on.
    # ``sigma_c_body_omega`` above is then the EXTRAPOLATED Σ_c and is what
    # drives E_nk; this one exists so the driver can diagonalize the ordinary
    # full-band Σ once per iteration too and report the correction at the eqp
    # level, where it is a statement about the answer rather than about Σ.
    # None when the feature is off — the field is then not merely unused but
    # absent from every downstream branch, which is what keeps that path
    # bit-identical.
    sigma_c_body_omega_unextrap: jax.Array | None = None


def _fit_head_correction(
    head_resolver: HeadResolver, *,
    config: LorraxConfig,
    meta,
    probe_omega: complex,
    print_fn,
    iteration_head=None,
):
    """Fit the GN-PPM scalar head from the user-selected source."""
    from .head_correction import (
        fit_head_hl_analytic_from_sample,
        fit_head_ppm_from_samples,
        fit_head_with_fixed_omega_from_sample,
        format_head_diagnostics,
    )

    head_source = iteration_head if iteration_head is not None else head_resolver
    head_static = head_source.at(0.0 + 0.0j)
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
        head_probe = head_source.at(probe_omega)
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
    iteration_head=None,
) -> np.ndarray:
    """Compute the analytic q→0, G=G'=0 PPM head diagonal.

    A head-less Σ_c is a silent wrong answer (Bug B,
    reports/sigma_ppm_tighten_2026-07-04).  Injection is deliberately left
    to the ansatz-neutral dynamic-Sigma finalizer.
    """
    from .head_correction import (compute_ppm_head_sigma_diag,
                                  on_shell_occupied_head_sigma_ry)

    occupations = None
    if iteration_head is None:
        enk_full, _ = get_enk_bandrange(
            wfn, sym, band_slices.sigma_range, band_slices.sigma_range,
            nspinor=meta.nspinor)
        enk_full_np = np.asarray(enk_full, dtype=np.float64)
        # Canonical DFT mid-gap E_F for the one-shot path.
        efermi_ry = float(wfn.efermi)
    else:
        # The velocity, transition denominators, occupations, and analytic
        # head Sigma must describe ONE iteration.  Re-loading DFT energies
        # here would quietly mix bases after the S tensor was updated.
        enk_full_np = np.asarray(
            iteration_head.sigma_energies_ry, dtype=np.float64)
        efermi_ry = float(iteration_head.efermi_ry)
        expected = (int(meta.nk_tot), int(meta.nb_sigma))
        if enk_full_np.shape != expected:
            raise ValueError(
                "iteration head active energies must have shape "
                f"{expected}, got {enk_full_np.shape}.")
        occupations = np.asarray(
            iteration_head.sigma_occupations, dtype=np.float64)
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
        occupations=occupations,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
    )

    # max|dense| == max|diag| (off-diagonals are exact zeros; |diag| >= 0),
    # so the diagnostic is unchanged on every path.
    head_max_ev = float(np.max(np.abs(head_sigma_diag_ry))) * RYD_TO_EV
    # Derived from the SAME kernel that built the tensor above, not from a
    # second spelling of its closed form: the hand-written ``-R_h/(...)``
    # that used to live here had drifted in sign against both the kernel
    # and the named ``sig_c_head(Edft).Re`` output column (register row
    # `ppm_pipeline.py:193-201`; JID 57243214 measured log -0.8071 eV vs
    # tensor +0.807048 eV on the same occupied state).
    on_shell_occ = on_shell_occupied_head_sigma_ry(
        head_gn,
        cell_volume=float(meta.cell_volume),
        nk_tot=int(meta.nk_tot),
    ) * RYD_TO_EV
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


def _extrapolated_point(cube, weights):
    """``S_extrap`` over the leading bracket axis: ``sum_b w_b * cube[b]``.

    THE OPERATION THAT MAKES THE EXTRAPOLATED Σ A LEGITIMATE HAMILTONIAN.
    ``weights`` are REAL and sum to 1 under BOTH estimators, and both
    properties are load-bearing rather than incidental.

    TWO SHAPES, ONE INVARIANT.

    ``(3,)`` — ``band_index_only``.  The ordinary-least-squares coefficients
    from ``band_extrapolation.extrapolation_weights`` depend only on the three
    band COUNTS, so this is one fixed affine combination applied identically
    to every (ω, k, i, j) element.  **This branch is unchanged and is the
    bit-for-bit path**: it must stay a single ``tensordot`` in exactly this
    order, because that is what the byte-identity gate against the previous
    default compares.

    ``(3, nk, nb)`` — ``spectral_shell``.  The estimator solves one exponent
    per EXTERNAL state, so its coefficients carry the state shape.  The Σ
    element ``(i, j)`` has two external states, and the coefficient applied
    there is the MEAN of the two, ``w_ij = ½(w_i + w_j)``.  That is forced,
    not chosen: it is the unique symmetric rule that is exact on the band
    diagonal (where the estimator is defined and where it was measured) and
    that preserves ``sum_b w_b = 1``.

    Hermiticity survives EITHER shape EXACTLY, not approximately.  Each
    cumulative bracket point is Hermitian in (i, j); a real scalar times a
    complex number commutes with conjugation bit-for-bit in IEEE arithmetic;
    ``½(w_i + w_j)`` equals ``½(w_j + w_i)`` to the last bit because IEEE
    addition is commutative; and the three-term reduction is performed in the
    same order for element (i, j) and element (j, i).  So
    ``S[j, i] == conj(S[i, j])`` to the last bit, and the next SC iteration's
    eigenvectors stay consistent with its own eigenvalues.
    ``tests/test_band_extrapolation.py::
    test_extrapolated_sigma_is_hermitian_to_machine_precision`` is the gate.

    Sharding is restated on the way out for the same reason
    :func:`_band_count_point` restates it: the leading axis is dropped, and
    ``sigma_omega_layout=sharded``'s contract is that consumers read the
    layout off the array rather than trusting XLA to propagate it through a
    reduction.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim not in (1, 3):
        raise ValueError(
            f"_extrapolated_point: weights must be (3,) [band_index_only] or "
            f"(3, nk, nb) [spectral_shell], got shape {w.shape}")

    def _combine(a):
        if w.ndim == 1:
            return jnp.tensordot(jnp.asarray(w, dtype=a.dtype), a, axes=(0, 0))
        # Per-state coefficients.  Symmetrised to (nk, nb, nb) ONE BRACKET AT
        # A TIME: the full (3, nk, nb, nb) block would be three times the
        # footprint of the thing it multiplies, and on a large deck that is
        # hundreds of MB of nothing.
        acc = None
        for b in range(w.shape[0]):
            wb = jnp.asarray(w[b], dtype=a.dtype)          # (nk, nb)
            wsym = 0.5 * (wb[:, :, None] + wb[:, None, :])  # (nk, nb, nb)
            term = a[b] * wsym[None, ...]
            acc = term if acc is None else acc + term
        return acc

    sharding = getattr(cube, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return _combine(cube)
    spec = tuple(sharding.spec)
    if len(spec) != int(getattr(cube, "ndim", 0)):
        return _combine(cube)
    out = NamedSharding(sharding.mesh, P(*spec[1:]))
    return jax.jit(_combine, out_shardings=out)(cube)


def _report_band_extrapolation(
    sigma_omega, head_sigma_diag_w_kn_ry, *,
    plan, config, band_slices, wfn, sym, meta, mesh_xy, print_fn,
) -> dict:
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

    Returns ``(h5_payload, weights)``.  The WEIGHTS come back from here rather
    than being derived at the call site because under ``spectral_shell`` they
    are per-state and are a function of the FIT — deriving them twice would be
    deriving the estimator twice, and the two copies could disagree.  The
    caller applies them to the Σ cube; ``band_extrapolation.extrapolation_
    weights`` is still used for ``band_index_only`` so that path stays the
    identical arithmetic it always was.
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
        # ``clamp``, named.  These three points are a band-count FIT input,
        # and the fit's own trust verdict is what judges them; masking an
        # uncovered state to nan here would poison the fit for the covered
        # ones through the same least squares.  The count is reported once
        # at the output path, on the same grid and the same eval energies.
        points.append(interp_along_omega(
            diag_w_kn * RYD_TO_EV, omega_grid_ev, omega_eval_ev,
            out_of_range="clamp"))
    s_at_counts = np.stack(points)

    # ── WHICH ESTIMATOR ─────────────────────────────────────────────────
    # Read here and nowhere else.  Both estimators consume the SAME
    # ``s_at_counts`` produced above, so the fork is post-processing: nothing
    # about the compute, the brackets or the three points depends on it.
    estimator = str(getattr(config.sigma, "band_extrapolation_estimator",
                            BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT))
    spectral = estimator == "spectral_shell"
    ladder = None
    if spectral:
        # THE LADDER IS DFT-ONLY, AND IT IS BUILT FROM THE WFN'S OWN
        # EIGENVALUES AND k WEIGHTS — the mean field, never the three Σ
        # values.  Absolute band indexing (band 1 = the WFN's first band),
        # because the Weyl counting law counts from the bottom of the band
        # manifold; ``b0`` carries the Σ sum's offset into it.
        ladder = build_band_ladder(
            enk_ry=np.asarray(wfn.energies[0], dtype=np.float64),
            kweights=np.asarray(wfn.kweights, dtype=np.float64),
            n_target=plane_wave_band_count(wfn.ngk, int(wfn.nspinor)),
            b0=int(band_slices.b0),
        )
        fit = fit_band_extrapolation_spectral(
            sigma_omega.band_counts, s_at_counts, ladder)
    else:
        fit = fit_band_extrapolation(sigma_omega.band_counts, s_at_counts)

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
    print_fn((format_spectral_report if spectral
              else format_extrapolation_report)(plan, fit, states=states))

    # ── THE PER-STATE REFUSAL ───────────────────────────────────────────
    # Raised AFTER the report block, so the operator sees the shells, the
    # ladder and every state's D2/D3 before the message about why the run
    # stopped.  spectral_shell never clips an exponent and never substitutes
    # a value for a failed state; see
    # ``band_extrapolation.SpectralShellExtrapolationFailed``.
    if spectral and fit.n_failed:
        raise SpectralShellExtrapolationFailed(fit.failure_report())

    # ── WHAT THIS RUN DOES WITH THE NUMBER ──────────────────────────────
    # Until 2026-08-16 the fit was reported and then discarded: S(N₃) drove
    # the Hamiltonian and S_inf lived in a log line and four h5 datasets.  It
    # now drives E_nk, and a reader of this block must not have to infer that
    # from the absence of a statement.
    print_fn(
        f"     [driving] E_nk for this iteration is built from the "
        f"EXTRAPOLATED Sigma_c ({estimator}), not S(N3).  Sigma is "
        f"extrapolated FIRST and diagonalized SECOND: the estimator is a REAL "
        f"linear combination of the three cumulative band-bracket sums, "
        f"symmetrised over the two external states of each matrix element, so "
        f"the result is exactly Hermitian and is a legitimate static "
        f"self-energy.  Extrapolating EIGENVALUES instead would produce a "
        f"spectrum belonging to no Hamiltonian.")

    # ── THE TOLERANCE RULING ────────────────────────────────────────────
    # Printed HERE, beside the fit that sets the bar, and once per SC
    # iteration because this pipeline runs once per SC iteration.  See
    # ``band_extrapolation.sc_tolerance_ruling`` for why this warns rather
    # than refusing — in short, ``sc_tol_ev`` defaults to 0.1 meV against a
    # bar of tens of meV, so a refusal would fire on the shipped default, and
    # the two numbers answer different questions anyway.
    tol_ev = getattr(getattr(config, "sc", None), "tol_ev", None)
    if tol_ev is not None:
        _, ruling = sc_tolerance_ruling(fit, float(tol_ev))
        print_fn(ruling)

    # ── THE PER-MODE CONTAMINANT THE PER-compute_mode GUARD CANNOT SEE ───
    # ``ppm_invalid_mode = "static_limit"`` (the shipping default) puts an
    # analytic static-COHSEX term inside this Σ_c.  The PPM-only guard in
    # ``sigma_dispatch`` exists precisely to keep a static Coulomb hole out of
    # this fit, and it cannot fire here because the ``compute_mode`` genuinely
    # IS ``gn_ppm`` — the contamination is per-MODE, one logical ISDF mode at a
    # time, underneath the seam that guard checks.  So the check lives HERE,
    # beside the fit it qualifies, and triggers on the term rather than on the
    # mode.  ``static_coh_at_counts`` is None when the run has no invalid poles
    # or is not extrapolating, and there is then nothing to say.
    static_coh = getattr(sigma_omega, "static_coh_at_counts", None)
    if static_coh is not None:
        # Same band diagonal, same states, same unit as ``points`` above: the
        # term is ω-INDEPENDENT, so no interpolation onto ``omega_eval_ev`` is
        # possible or needed and the diagonal is taken directly.
        static_diag = np.einsum(
            'bkii->bki', np.asarray(static_coh)) * RYD_TO_EV
        _, static_ruling, _ = static_limit_tail_ruling(fit, static_diag)
        print_fn(static_ruling)

    # The arrays are already in eV on the band diagonal, which is the unit
    # and the layout ``sigma_mnk.h5`` wants, so no scale is applied here.
    if spectral:
        return spectral_h5_payload(plan, fit), fit.weights()
    return (extrapolation_h5_payload(plan, fit),
            extrapolation_weights(sigma_omega.band_counts))


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
    iteration_head=None,
    occupation_state=None,
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

    ``occupation_state`` is the iteration's
    :class:`gw.efermi.OccupationState`, carried through to the one
    occupation projector this pipeline builds (the invalid-pole
    static-COHSEX term in ``ppm_sigma``).  ``None`` — every insulating
    deck — keeps the integer projector bit-for-bit.
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
        q_neg = None
        if not is_hl:
            from ffi import _services
            _services.ensure_on_path()
            from symmetry_maps import q_negation_index
            q_neg = q_negation_index(tuple(int(v) for v in meta.kgrid))
        ppm = fit_ppm(
            W_static_q, W_probe_q, V_q, probe_omega, mesh_xy,
            fallback_omega=config.ppm.fallback_omega,
            n_nodes_static=quad.node_count,
            print_fn=print_fn,
            model_label=label,
            n_mu_logical=int(meta.n_rmu),
            q_neg_index=q_neg,
            # User-ruled GN variant: re-anchor the exact 0.2% tails at the fit
            # owner before the incumbent exact-pane planner sees the reduced
            # support.  This is lossy versus BGW finite-pole parity; HL is a
            # different real-axis model and is deliberately unchanged.
            coarsen_extreme_tails=not is_hl,
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
        # THE CUTS ARE WITHIN THE Σ COUNT, NOT THE χ COUNT.  ``b_id_4_sigma``
        # / ``sigma_sum``, never ``b_id_4_user`` / ``full``: the latter pair
        # is the LOADED extent = max(chi, sigma), so on a default-scheme deck
        # running χ at 248 and Σ at 100 they would bracket at
        # ~(198, 223, 248) — three points on a curve this run never evaluates
        # — instead of ~(80, 90, 100).  The conduction-coordinate scheme
        # likewise defines n_occ, n_cond and its DFT ladder inside this same
        # Σ window.  Identical counts on an unsplit deck.
        # Pinned by tests/test_band_extrapolation_sigma_count.py.
        plan = plan_band_brackets(
            enabled=bool(config.sigma.band_extrapolation),
            enk_ry=np.asarray(wfns.enk[:, s.sigma_sum]),
            n_occ=int(s.b2 - s.b0),
            nb_logical=int(meta.b_id_4_sigma_user or s.b4) - int(s.b0),
            nb_padded=int(s.nb_sigma_sum),
            bracket_scheme=str(
                config.sigma.band_extrapolation_bracket_scheme),
        )
        # ...AND THE COMMENT ABOVE IS NOT ENOUGH, SO THIS IS CHECKED.  A plan
        # built from the wrong count is invisible in every weight-level
        # diagnostic — the OLS coefficients depend only on the abscissae's
        # RATIOS and the fractions are the same 0.80/0.90/1.00 of whichever
        # count, so a wrong-count run is Hermitian, converges, and prints
        # ordinary numbers.  This is the last place that can see it: here the
        # plan and the band slices its brackets will slice are both in scope.
        # Before the shared MPA planner opens the fit store, so a mismatch
        # costs neither I/O nor a compile.
        assert_brackets_match_ols_abscissae(
            plan, s, meta=meta, where="ppm_pipeline plan seam")
        if plan.enabled:
            print_fn(
                f"  Σc band extrapolation: ON — bracket scheme "
                f"{plan.bracket_scheme}; {plan.n_brackets} disjoint "
                f"band brackets {plan.bounds} against ONE W(τ) per τ; "
                f"band counts {plan.counts} (requested {plan.requested}).")
            # Emitted HERE and not only in the report block at the end: a
            # planner fallback is a fact about the run that the operator
            # should see before Σ is spent, not after.
            for note in plan.notes:
                print_fn(f"  Σc band extrapolation: {note}")
        from .sigma_box_plan import resolve_sigma_box_cache_dir
        quadrature_cache_dir = resolve_sigma_box_cache_dir(
            config.sigma.quadrature_cache_dir, config.input_dir)
        fit_dir = os.path.join(config.input_dir, "tmp", "mpa")
        if process_rank() == 0:
            os.makedirs(fit_dir, exist_ok=True)
        barrier("ppm_mpa_fit_directory_ready")
        fit_store_path = os.path.join(fit_dir, "mpa_fit_oneshot.h5")
        with timing.section("sigma.exec"):
            sigma_omega = compute_sigma_c_ppm_omega_grid(
                wfns, ppm, meta, mesh_xy,
                ppm_cfg=config.ppm,
                sigma_cfg=config.sigma,
                mpa_cfg=config.mpa,
                omega_grid_ry=config.omega_grid_ry,
                ansatz=config.compute_mode,
                fit_store_path=fit_store_path,
                screening_diagrams=config.screening.diagrams,
                quadrature_cache_dir=quadrature_cache_dir,
                occupation_state=occupation_state,
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
        sigma_c_body_omega_n3 = _band_count_point(
            sigma_omega.sigma_c_kij, sigma_omega.sigma_c_kij.shape[0] - 1)
        # OFF: ``sigma_c_body_omega`` IS the N₃ point and the second cube is
        # None, so the object graph below is exactly what it always was.
        # ON: both are replaced at the report seam below.
        #
        # The combination itself happens AFTER the report block below, for
        # one reason: under ``spectral_shell`` the coefficients ARE the fit,
        # and the fit is built there.  Deriving them here as well would be
        # deriving the estimator twice.  Nothing between here and there reads
        # ``sigma_c_body_omega``, and for ``band_index_only`` the arithmetic
        # is byte-for-byte what it was when it happened at this line.
        sigma_c_body_omega = sigma_c_body_omega_n3
        sigma_c_body_omega_unextrap = None

        # Step 3: q→0 head construction (analytic, mini-BZ-averaged)
        head_gn = _fit_head_correction(
            head_resolver, config=config, meta=meta,
            probe_omega=probe_omega, print_fn=print_fn,
            iteration_head=iteration_head,
        )
        head_sigma_diag_w_kn_ry = _compute_analytic_head_diag(
            head_gn,
            config=config, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta,
            iteration_head=iteration_head,
            print_fn=print_fn,
        )

        # Step 4: the band-convergence extrapolation report.  After the head,
        # because the head is part of the Σ_c being reported; before the
        # return, because the cube's leading axis does not survive it.
        extrap_payload = None
        if plan.enabled:
            extrap_payload, extrap_weights = _report_band_extrapolation(
                sigma_omega, head_sigma_diag_w_kn_ry,
                plan=plan, config=config, band_slices=band_slices,
                wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh_xy,
                print_fn=print_fn,
            )
            # ── WHICH Σ DRIVES THE ITERATION ────────────────────────────
            # The EXTRAPOLATED Σ_c, so the band-sum tail is included in the
            # E_nk the SC loop converges.  The un-extrapolated N₃ cube is
            # kept beside it — not as a fallback, but so the driver can
            # diagonalize BOTH once per iteration and report the eqp-level
            # correction side by side.  Extrapolating Σ and then
            # diagonalizing is the only order that yields a Hermitian
            # operator; see ``_extrapolated_point``.
            sigma_c_body_omega = _extrapolated_point(
                sigma_omega.sigma_c_kij, extrap_weights)
            sigma_c_body_omega_unextrap = sigma_c_body_omega_n3

    return PPMOutputs(
        sigma_c_body_omega=sigma_c_body_omega,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
        band_extrapolation=extrap_payload,
        sigma_c_body_omega_unextrap=sigma_c_body_omega_unextrap,
    )
