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

import dataclasses

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

from common.units import RYD_TO_EV
from common.wfn_transforms import get_enk_bandrange
import common.timing as timing

from .band_extrapolation import (
    BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT,
    build_band_ladder,
    fit_band_extrapolation_spectral,
    format_spectral_report,
    plane_wave_band_count,
    spectral_h5_payload,
    assert_brackets_match_ols_abscissae,
    static_limit_tail_ruling,
    plan_band_brackets,
    pooled_state_mask,
    sc_tolerance_ruling,
)
from .dynamic_sigma import sigma_on_shell_request
from .gw_config import LorraxConfig
from .head_correction import HeadResolver
from .ppm_sigma import (
    compute_sigma_c_ppm_omega_grid,
    fit_pole_model,
    pole_fit_plan,
)
from runtime.padding import PaddedAxis


@dataclass(frozen=True)
class PPMOutputs:
    """Ansatz-specific PPM outputs handed to the dynamic finalizer."""

    sigma_c_body_omega: object             # ppm_sigma.OnShellSigma (or, sigma_omega_grid_debug, the (n_omega, nk, nb, nb) cube): the Sigma_c body, Ry
    band_axis: PaddedAxis | None = None
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
    probe_hermiticity_residual: float | None = None
    odd_even_residue_ratio: float | None = None
    #: The body's omega reference (Ry), from ``SigmaOmegaResult.efermi_ry``.
    efermi_ry: float | None = None


#: Open :func:`observe_window_sigma_source` blocks, innermost last.
_WINDOW_SOURCE_OBSERVERS: list = []


class observe_window_sigma_source:
    """Collect, from the GN/HL-PPM Sigma stage that runs inside the block, what a window Sigma reuses.

    TD-aGW's frozen-screening collision term (docs/theory/td_agw_dynamical.md
    §1.5, §9 item 5) evaluates ``gw.ppm_sigma.window_sigma_c`` on the GW run's
    own carrier and pole model.  A GN-PPM run persists W(0) but neither
    W(i w_p) nor its poles, so the TD driver runs the GW driver in its own
    process and reads them here, at the return of
    :func:`compute_ppm_sigma_pipeline`: no stage is re-implemented and the
    driver's outputs are unchanged (with no open block the seam is one empty
    list test).  ``with observe_window_sigma_source() as src: gw_jax.main(...)``
    leaves in ``src``: ``wfns`` (the Sigma carrier), ``meta``, ``config``,
    ``poles`` (``PPMBuildResult``), ``head`` (``HeadSigmaDiag`` or None),
    ``efermi_ry``, ``request`` (the run's ``OnShellRequest`` or None), ``V_q``,
    ``quad``.  The device fields of the poles and the carrier are copied at the
    seam (a later stage may donate the originals); the last Sigma stage of the
    block wins.  ``screening=True`` also leaves ``fit_plan`` (the run's
    :func:`gw.ppm_sigma.pole_fit_plan`) and ``W_supports`` (copies of the fit's
    two W samples, W(0) and W(probe) on the fit's q set, taken right after the
    fit): the held inputs of the TD-aGW screening update
    (``gw.ppm_sigma.screening_update``, docs/theory/td_agw_dynamical.md Eq. 6).
    """

    def __init__(self, screening: bool = False):
        self.source = {}
        self.screening = bool(screening)

    def __enter__(self):
        _WINDOW_SOURCE_OBSERVERS.append(self)
        return self.source

    def __exit__(self, *exc):
        _WINDOW_SOURCE_OBSERVERS.remove(self)
        return False


#: The fields only a ``screening=True`` observer receives.
_SCREENING_FIELDS = ("fit_plan", "W_supports")


def _publish_window_source(**fields):
    """Hand the Sigma stage's objects to every open :class:`observe_window_sigma_source`."""
    poles = fields["poles"]
    copy = lambda a: None if a is None else jnp.copy(a)
    fields["poles"] = dataclasses.replace(
        poles, B_q=copy(poles.B_q), Omega_q=copy(poles.Omega_q),
        valid_mask_q=copy(poles.valid_mask_q), Wc0_q=copy(poles.Wc0_q),
        B_odd_q=copy(poles.B_odd_q))
    fields["wfns"] = jax.tree.map(
        lambda a: jnp.copy(a) if isinstance(a, jax.Array) else a, fields["wfns"])
    for obs in _WINDOW_SOURCE_OBSERVERS:
        obs.source.update({k: v for k, v in fields.items()
                           if obs.screening or k not in _SCREENING_FIELDS})


def _fit_head_correction(
    head_resolver: HeadResolver, *,
    config: LorraxConfig,
    meta,
    num_electrons: float,
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
        # has factor 4 → 16π).  N_e is the WFN's electron count, not
        # meta.nelec: that counts occupied bands, N_e/2 on a scalar WFN.
        omega_p_sq_ry = 16.0 * float(np.pi) * float(num_electrons) / float(meta.cell_volume)
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
    body_efermi_ry=None,
) -> np.ndarray:
    """Compute the analytic q→0, G=G'=0 PPM head diagonal.

    ``body_efermi_ry`` is the omega reference of the Sigma body this head is
    added to (``SigmaOmegaResult.efermi_ry``).  The head Sigma(omega) lives
    on the same omega grid, so it must be measured from the same energy.

    A head-less Σ_c is a silent wrong answer (Bug B,
    reports/sigma_ppm_tighten_2026-07-04).  Injection is deliberately left
    to the ansatz-neutral dynamic-Sigma finalizer.
    """
    from .head_correction import HeadSigmaDiag, on_shell_occupied_head_sigma_ry

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
    if body_efermi_ry is not None:
        efermi_ry = float(body_efermi_ry)
    n_occ = min(meta.nelec, enk_full_np.shape[1])

    # The head is band-diagonal and kept as its closed form, the one MPA's
    # head is (``HeadSigmaDiag``): the grid finalize evaluates its rows where
    # it adds them, the on-shell one its value and derivative at the
    # evaluation energies.  Its cells are ``compute_ppm_head_sigma_diag``'s
    # arithmetic (one real pole Omega_h - i*eta, residue R_h, the same
    # occupation step), so the grid numbers are unchanged.
    if occupations is None:
        occupations = np.zeros((enk_full_np.shape[1],), dtype=np.float64)
        occupations[: max(0, min(int(n_occ), enk_full_np.shape[1]))] = 1.0
    live = abs(head_gn.R_h) >= 1.0e-30 and abs(head_gn.omega_h) >= 1.0e-30
    head_sigma_diag_ry = HeadSigmaDiag.build(
        omega_grid_ry=np.asarray(config.omega_grid_ry, dtype=np.float64),
        enk_ry=enk_full_np, efermi_ry=efermi_ry, occupations=occupations,
        poles_ry=np.asarray([head_gn.omega_h - 1j * 1.0e-6] if live else [], np.complex128),
        residues_ry=np.asarray([head_gn.R_h] if live else [], np.float64),
        cell_volume=float(meta.cell_volume), nk_tot=int(meta.nk_tot))

    # max|dense| == max|diag| (off-diagonals are exact zeros; |diag| >= 0),
    # so the diagnostic is unchanged on every path.  Band blocks bound the
    # host temporary.
    nb_head = int(enk_full_np.shape[1])
    head_max_ev = max((float(np.max(np.abs(head_sigma_diag_ry.band_rows(b, min(b + 64, nb_head)))))
                       for b in range(0, nb_head, 64)), default=0.0) * RYD_TO_EV
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
    return head_sigma_diag_ry


def _combine_extrapolation(first, rest, w):
    """Apply symmetric per-state weights to the live points, in count order."""
    w = jnp.asarray(w, dtype=first.dtype)
    # Form only one (nk, nb, nb) weight matrix at a time. The small static
    # point loop preserves the established elementwise summation order.
    acc = None
    for b, a in enumerate((first, *rest)):
        wb = w[b]
        wsym = 0.5 * (wb[:, :, None] + wb[:, None, :])
        term = a * wsym[None, ...]
        acc = term if acc is None else acc + term
    return acc


@lru_cache(maxsize=16)
def _extrapolation_kernel(sharding, donate):
    """The ordered affine combination; ``donate``: into the first point's buffer."""
    return jax.jit(_combine_extrapolation, out_shardings=sharding,
                   donate_argnums=(0,) if donate else ())


def band_count_point(cube, i: int):
    """Count ``i`` of a ``BandCountCube`` on the full BZ (an ``OnShellSigma``).

    On the grid, ``(ω, k, m, n)``: a fresh buffer at ``P(None, None, 'x',
    'y')``; consumers read the layout off the array
    (``qsgw_utils.is_band_sharded_sigma_omega``).
    """
    return cube.point(i)


def _extrapolated_point(cube, weights, top=None):
    """``S_extrap = sum_b w_b * S(N_b)`` over the points that carry weight.

    THE OPERATION THAT MAKES THE EXTRAPOLATED Σ A LEGITIMATE HAMILTONIAN.
    ``weights`` are REAL and sum to 1, and both properties are load-bearing
    rather than incidental.

    ``(3, nk, nb)`` — ``spectral_shell``.  The estimator solves one tail
    ratio per EXTERNAL state, so its coefficients carry the state shape.  The
    Σ element ``(i, j)`` has two external states, and the coefficient applied
    there is the MEAN of the two, ``w_ij = ½(w_i + w_j)``.  That is forced,
    not chosen: it is the unique symmetric rule that is exact on the band
    diagonal (where the estimator is defined and where it was measured) and
    that preserves ``sum_b w_b = 1``.

    Hermiticity survives EXACTLY, not approximately.  Each
    cumulative bracket point is Hermitian in (i, j); a real scalar times a
    complex number commutes with conjugation bit-for-bit in IEEE arithmetic;
    ``½(w_i + w_j)`` equals ``½(w_j + w_i)`` to the last bit because IEEE
    addition is commutative; and the reduction is performed in the
    same order for element (i, j) and element (j, i).  So
    ``S[j, i] == conj(S[i, j])`` to the last bit, and the next SC iteration's
    eigenvectors stay consistent with its own eigenvalues.

    MEMORY.  The matrix of a point is needed only where its weight is not
    zero: N₁ and N₃ (N₂ fixes Ω from its band diagonal, which is all the
    sweep keeps of it).  The first live point gives its buffer to the
    result, so the cube no longer serves that point afterwards; ``top`` is
    the caller's N₃ cube (the raw twin), read and kept.  Two full-BZ cubes
    are resident: the two this stage returns.
    """
    w = np.asarray(weights, dtype=np.float64)
    n_count, carrier = int(cube.shape[0]), int(cube.shape[-1])
    if w.ndim != 3 or w.shape[0] != n_count:
        raise ValueError(
            f"_extrapolated_point: weights must be the per-state "
            f"({n_count}, nk, nb) of spectral_shell, got shape {w.shape}")
    if w.shape[-1] != carrier:
        # Per-state weights are fitted on the LOGICAL bands; the cube keeps
        # its padded band carrier, whose extra rows/columns are zero.  Zero
        # weights there keep them zero without stripping the cube.
        if w.shape[-1] > carrier:
            raise ValueError(
                f"_extrapolated_point: {w.shape[-1]} logical bands exceed the "
                f"cube's band carrier {carrier}")
        w = np.pad(w, ((0, 0), (0, 0), (0, carrier - w.shape[-1])))
    live = [b for b in range(n_count) if np.any(w[b])]
    if not live:
        raise ValueError(
            "_extrapolated_point: every weight is zero; they must sum to 1")
    between = [b for b in live if cube.diagonal_only(b)]
    if between:
        raise ValueError(
            f"_extrapolated_point: band counts {between} carry weight, but "
            f"the sweep keeps only their band diagonal (BandCountCube)")
    if cube.request is not None:
        return _on_shell_combination(cube, live, w[live])
    if cube.fresh:
        return _streamed_combination(cube, live, w[live])
    first = cube.point(live[0])
    rest = tuple(
        top if (top is not None and b == n_count - 1) else cube.point(b)
        for b in live[1:])
    # ``first`` is the caller's own N3 cube only where no state has a tail;
    # it is then read, not taken.
    return _extrapolation_kernel(first.sharding, first is not top)(
        first, rest, w[live])


def _on_shell_combination(cube, live, w):
    """``sum_b w_b * S(N_b)`` of an on-shell cube: its pair and its slots.

    The pair is combined by :func:`_combine_extrapolation`, with the two pair
    slots on its leading axis; each
    diagonal slot of state ``(k, n)`` takes ``w_b[k, n]``, the pair rule's
    ``½(w_n + w_n)`` on the diagonal.  Every count's slots are kept
    (``BandCountCube.slot_diagonals``), so the derivative and the eqp tables
    of the extrapolated Sigma are combinations too.
    """
    from .ppm_sigma import OnShellSigma
    points = [cube.point(b) for b in live]
    pair = _extrapolation_kernel(points[0].pair.sharding, cube.fresh)(
        points[0].pair, tuple(p.pair for p in points[1:]), w)
    slots = _slot_combination_fn(points[0].slots.sharding)(
        tuple(p.slots for p in points), jnp.asarray(w))
    return OnShellSigma(pair=pair, slots=slots, request=points[0].request,
                        mesh_xy=points[0].mesh_xy)


@lru_cache(maxsize=8)
def _slot_combination_fn(sharding):
    def combine(slots, w):
        acc = None
        for b, a in enumerate(slots):
            term = a * w[b][None, :, :, None]
            acc = term if acc is None else acc + term
        return acc
    return jax.jit(combine, out_shardings=sharding)


#: omega slabs of one streamed combination (a slab's unfolded counts are
#: its only temporaries).
_COMBINE_SLABS = 8


def _streamed_combination(cube, live, w):
    """``sum_b w_b * S(N_b)`` written omega slab by omega slab from the wedge.

    Each slab unfolds its live counts, combines them in count order
    (:func:`_combine_extrapolation`, the same elementwise arithmetic) and is
    written into the one output cube, so no full-BZ count is ever resident.
    """
    members = tuple(cube._matrix(b) for b in live)
    n_omega = int(members[0].shape[0])
    step = -(-n_omega // _COMBINE_SLABS)
    shape = (n_omega, int(cube.nk), *members[0].shape[2:])
    sharding = NamedSharding(cube.mesh_xy, P(None, None, "x", "y"))
    out = jnp.zeros(shape, jnp.complex128, device=sharding)
    run = _combination_slab_fn(cube.unfold, step,
                               0 if cube.term is None else cube.term.ndim, sharding)
    for lo in range(0, n_omega, step):
        out = run(out, members, cube.term, w, jnp.asarray(
            min(lo, n_omega - step), dtype=jnp.int32))
    return out


@lru_cache(maxsize=16)
def _combination_slab_fn(unfold, step, term_ndim, sharding):
    """One ω slab of the combination; ``term_ndim`` 0 (no term), 3 (one term
    for every ω) or 4 (a term per ω, sliced with the slab)."""
    def slab(out, members, term, w, lo):
        parts = []
        for m in members:
            part = unfold(jax.lax.dynamic_slice_in_dim(m, lo, step, axis=0))
            if term_ndim == 3:
                part = part + term[None, ...]
            elif term_ndim == 4:
                part = part + jax.lax.dynamic_slice_in_dim(term, lo, step, axis=0)
            parts.append(part)
        value = _combine_extrapolation(parts[0], tuple(parts[1:]), w)
        return jax.lax.dynamic_update_slice_in_dim(out, value, lo, axis=0)
    return jax.jit(slab, donate_argnums=(0,), out_shardings=sharding)


def _grid_fit_points(cube, head_sigma_diag_w_kn_ry, sigma_omega, mesh_xy, omega_grid_ev,
                     omega_eval_ev):
    """The band-count fit's points from a grid ``BandCountCube`` (``sigma_omega_grid_debug``).

    The cube lives on the padded band carrier; the head, the DFT energies and
    the fit are logical.  The reader strips at this consumer boundary and
    adds the head at the two omega slots each (k, n) reads: no (n_omega, nk,
    nb) diagonal is made on the host.  Ry -> eV here, exactly where
    ``eval_sigma_c_at_dft_energies`` does it.  ``clamp``, named: these points
    are a band-count FIT input, and the fit's own trust verdict judges them.
    """
    from .qsgw_utils import interp_sigma_diag_along_omega
    # The closed-form head is read at the slots; a host array as it is.
    head = (head_sigma_diag_w_kn_ry
            if head_sigma_diag_w_kn_ry is None
            or hasattr(head_sigma_diag_w_kn_ry, "at_slots")
            else np.asarray(head_sigma_diag_w_kn_ry))
    return [interp_sigma_diag_along_omega(
        cube.diagonal(i), mesh_xy, omega_grid_ev, omega_eval_ev,
        band_axis=sigma_omega.band_axis, add_w_kn=head, scale=RYD_TO_EV)
        for i in range(cube.shape[0])]


def _report_band_extrapolation(
    sigma_omega, head_sigma_diag_w_kn_ry, *,
    e_state_ev, plan, config, band_slices, wfn, mesh_xy, print_fn,
) -> dict:
    """Log the three band-count Σ_c's, the fit and its diagnostics.

    ``e_state_ev`` [k, n] (eV, absolute) is the energy each row of the cube
    is read at: the caller's ``e_qp_ev``, E_DFT in a one-shot and the map's
    read energies in SC.  Reads the band DIAGONAL of each cumulative point at
    that SAME external evaluation energy − E_F, under the same read rule —
    so nothing but the band count differs between the three.  The analytic q→0
    head is added to every point identically (it is a band-diagonal ω-dependent term with no
    unoccupied-state sum, hence bracket-independent), so the reported values
    are the physical Σ_c rather than the body alone; being a common offset it
    shifts S_∞ and S₃ together and leaves ``Δ_tail`` unchanged.

    Σ_c only.  Σ_x is a sum over OCCUPIED states and has no slow unoccupied
    tail; extrapolating Σ_total would fit a constant as if it converged.

    Returns ``(h5_payload, weights)``.  The WEIGHTS come back from here rather
    than being derived at the call site because they are per-state and are a
    function of the FIT — deriving them twice would be deriving the estimator
    twice, and the two copies could disagree.  The caller applies them to the
    Σ cube.
    """
    cube = sigma_omega.sigma_c_kij
    # THE ROW'S OWN ENERGY, NOT THE DFT ENERGY OF ITS SORTED INDEX.  The
    # cube's rows are the states of the bundle Sigma was built in; in SC they
    # are the input's sorted QP states.  Where a QP singlet and doublet swap
    # order against DFT, the DFT energy by index gave the two members of one
    # doublet two tail ratios and split it (bcc Na 3^3, map 1: 2.5 meV at
    # k = (1/3,1/3,1/3)), and the next W broke the crystal symmetry
    # (GATE shared_pole_gram_valid at map 2).  One-shot and SC map 0 read E_DFT.
    enk_ev = np.asarray(e_state_ev, dtype=np.float64)
    omega_eval_ev = enk_ev - (float(wfn.efermi) if sigma_omega.efermi_ry is None
                              else float(sigma_omega.efermi_ry)) * RYD_TO_EV
    if cube.request is not None:
        # The fit reads each state where H reads it: the pair table's diagonal
        # (``sigma_out_of_grid``) with the closed-form head added there, in eV
        # like every other Σ line in the log.
        points = [cube.diagonal(i).with_head(head_sigma_diag_w_kn_ry).diagonal(
            "pair", sigma_omega.band_axis) * RYD_TO_EV for i in range(cube.shape[0])]
    else:
        points = _grid_fit_points(cube, head_sigma_diag_w_kn_ry, sigma_omega, mesh_xy,
                                  np.asarray(config.omega_grid_ev, dtype=np.float64),
                                  omega_eval_ev)
    s_at_counts = np.stack(points)

    # ── THE ESTIMATOR ───────────────────────────────────────────────────
    # ``spectral_shell`` is the only value the config accepts; it is read
    # here for the log line below.
    estimator = str(getattr(config.sigma, "band_extrapolation_estimator",
                            BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT))
    # THE LADDER IS DFT-ONLY, AND IT IS BUILT FROM THE WFN'S OWN EIGENVALUES
    # AND k WEIGHTS — the mean field, never the three Σ values.  Absolute
    # band indexing (band 1 = the WFN's first band), because the Weyl
    # counting law counts from the bottom of the band manifold; ``b0``
    # carries the Σ sum's offset into it.
    ladder = build_band_ladder(
        enk_ry=np.asarray(wfn.energies[0], dtype=np.float64),
        kweights=np.asarray(wfn.kweights, dtype=np.float64),
        n_target=plane_wave_band_count(wfn.ngk, int(wfn.nspinor)),
        b0=int(band_slices.b0),
    )
    # The pooled (beta, Omega) fit reads each external state's DFT energy on
    # the ladder's own (absolute eV) reference: the model's denominator is
    # E_A - E_i + Omega.  It is pooled over the states within +-10 eV of E_F.
    fit = fit_band_extrapolation_spectral(
        sigma_omega.band_counts, s_at_counts, ladder, e_state_ev=enk_ev,
        fit_mask=pooled_state_mask(omega_eval_ev))

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
    print_fn(format_spectral_report(plan, fit, states=states))

    # States without a usable exponent keep S(N3) (``band_extrapolation``,
    # "NO USABLE EXPONENT").  The block above names them; this one line is
    # the count the production log keeps.
    if fit.n_failed:
        print_fn("WARNING: " + fit.failure_report().splitlines()[0])

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
    return spectral_h5_payload(plan, fit), fit.weights()


def plan_sigma_band_brackets(config, wfns, meta, *, print_fn, where):
    """The Σ band-bracket plan of one dynamic stage, validated and logged.

    One owner for every pole-sum Σ (GN/HL-PPM and scalar MPA): the plan fixes
    the τ kernel's Green-build count, its AOT signature and the Σ cube's
    leading extent, so it is resolved once, before anything compiles.
    ``use_band_extrapolation = false`` gives the trivial one-bracket plan.
    """
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
        plan, s, meta=meta, where=where)
    if plan.enabled:
        print_fn(
            f"  Σc band extrapolation: ON — bracket scheme "
            f"{plan.bracket_scheme}; {plan.n_brackets} disjoint "
            f"band brackets {plan.bounds} against ONE W(τ) per τ; "
            f"band counts {plan.counts} (requested {plan.requested}).")
        # Emitted HERE and not only in the report block at the end: a
        # planner fallback is a fact about the run that the operator
        # should see before Σ is spent, not after.  ``WARNING:`` is what
        # keeps it in the production log's warning block.
        for note in plan.notes:
            print_fn(f"WARNING: Σc band extrapolation: {note}")
    return plan


def extrapolate_sigma_body(
    sigma_omega, head_sigma_diag_w_kn_ry, *,
    e_state_ev, plan, config, band_slices, wfn, mesh_xy, print_fn, top=None,
):
    """``(extrapolated body, h5 payload, raw twin)`` from a bracketed Σ_c cube.

    ``sigma_omega`` carries the cumulative band-count axis
    (``SigmaOmegaResult``).  The fit reads the band diagonal with the q→0
    head added (:func:`_report_band_extrapolation`); the weights it returns
    combine the three cumulative cubes (:func:`_extrapolated_point`).  The
    EXTRAPOLATED Σ_c drives E_nk; the caller keeps the un-extrapolated N₃
    cube (:func:`band_count_point`) beside it so the driver
    diagonalizes both and reports the eqp-level correction.  ``top`` is that
    N₃ cube when the caller already holds it.  Extrapolating Σ
    and then diagonalizing is the only order that yields a Hermitian operator.

    THE RAW TWIN.  Its readers by default are the raw columns (sigC_raw,
    eqp0_raw, eqp1_raw), which read its band diagonal, so the twin returned
    is N₃'s diagonal slots (a pair-less ``OnShellSigma``; on the grid
    ``BandDiagonalSlots`` of N₃): one pair (or Σ(ω) cube) leaves this stage.
    N₃'s pair (the second QSGW matrix and the per-map "extrapolation effect
    on E_nk" eigvalsh block) is kept only under ``sigma_freq_debug_output``.
    The caller drops its own ``top`` reference.
    """
    cube = sigma_omega.sigma_c_kij
    last = int(cube.shape[0]) - 1
    extrap_payload, extrap_weights = _report_band_extrapolation(
        sigma_omega, head_sigma_diag_w_kn_ry, e_state_ev=e_state_ev,
        plan=plan, config=config, band_slices=band_slices,
        wfn=wfn, mesh_xy=mesh_xy, print_fn=print_fn)
    if bool(config.debug.sigma_freq_debug_output):
        twin = top if top is not None else band_count_point(cube, last)
    else:
        twin, top = cube.diagonal(last), None
    extrapolated = _extrapolated_point(cube, extrap_weights, top=top)
    cube.release()
    return extrapolated, extrap_payload, twin


def refuse_hl_ppm_without_trs(config, sym) -> None:
    """``GATE hl_ppm_requires_measured_trs``, before any basis, ζ or V.

    HL fits one residue on a real-axis probe, which cannot carry the
    anti-Hermitian residue of a time-reversal-broken store; GN's ordered
    fit splits W(iω_p) into both halves.
    """
    if config.compute_mode.ppm_model != "hl":
        return
    from .screening import _trs_verdict
    if not _trs_verdict(sym):
        raise ValueError(
            "GATE hl_ppm_requires_measured_trs: compute_mode = hl_ppm keeps "
            "one residue, but SymMaps.trs_allowed is false, so W(ω) has an "
            "anti-Hermitian residue the real-axis probe cannot carry.\n"
            "  want: compute_mode = gn_ppm (its ordered fit carries both "
            "halves)\n"
            "  doc:  docs/theory/gn-ppm-nonhermitian.md §6.")


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
    fixed_quadrature_session=None,
    e_qp_ev=None,
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

    ``e_qp_ev`` [k, n] (eV) is the energy each Σ row is read at, which the
    band extrapolation reads its states at too.

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
        # ORDERED ORIENTATIONS: GN on a deck whose MEASURED time-reversal
        # verdict is false.  The same predicate that routed the probe χ₀
        # through the ordered kernel in ``gw.screening.compute_screening``
        # (one measurement, ``SymMaps.trs_allowed``, read the same way) now
        # tells the fit to split W(iω_p) into its Hermitian and
        # anti-Hermitian halves.  HL keeps the single-residue fit: a
        # real-axis probe cannot carry the odd residue
        # (``docs/theory/gn-ppm-nonhermitian.md`` §6), so an
        # ordered store refuses HL at setup (``refuse_hl_ppm_without_trs``).
        from .screening import _trs_verdict
        ordered = bool((not is_hl) and (_trs_verdict(sym) is False))
        # GN fits on the q wedge (owner, 2026-09-25): W(0), W(iω_p) and V
        # share the run's wedge, every count weights a row by its star and
        # the tail closure reads -q through the unfold tables.  HL (a
        # real-axis probe, solved on the full zone) and an unreduced q grid
        # fit on the full zone.
        from .cohsex_sigma import interaction_operator
        W0_op = interaction_operator(W_static_q)
        Wp_op = interaction_operator(W_probe_q)
        V_op = interaction_operator(V_q)
        on_wedge = (not is_hl and not W0_op.is_whole_zone()
                    and Wp_op.same_wedge(W0_op, any_rule=True))
        if on_wedge:
            fit_inputs = (W0_op.values, Wp_op.values, V_op.restrict(W0_op).values)
            q_wedge = dataclasses.replace(W0_op, values=None, load=None)
            wedge_kw = dict(q_star=W0_op.star_sizes(),
                            q_partner=W0_op.minus_q_partner(meta.kgrid, mesh_xy),
                            q_wedge=q_wedge)
        else:
            fit_inputs = (W0_op.unfold(mesh_xy), Wp_op.unfold(mesh_xy), V_op.unfold(mesh_xy))
            wedge_kw = {}
        # The fit's plan and the fit are the TD-aGW screening update's too
        # (``ppm_sigma.screening_update``, docs/theory/td_agw_dynamical.md Eq. 6).
        fit_plan = pole_fit_plan(
            fit_inputs[2], probe_omega, mesh_xy,
            wedge=wedge_kw,
            fallback_omega=config.ppm.fallback_omega,
            n_nodes_static=quad.node_count,
            model_label=label,
            n_mu_logical=int(meta.n_rmu),
            mu_active_mask=getattr(meta, 'mu_active_mask', None),
            q_neg_index=q_neg,
            # User-ruled GN variant: re-anchor the exact 0.2% tails at the fit
            # owner before the box planner sees the reduced
            # support.  This is lossy versus BGW finite-pole parity; HL is a
            # different real-axis model and is deliberately unchanged.
            coarsen_extreme_tails=not is_hl,
            ordered=ordered,
            solve_extent=int(getattr(meta, 'mu_solve_extent', meta.n_rmu)),
        )
        ppm = fit_pole_model(fit_inputs[0], fit_inputs[1], fit_plan, print_fn=print_fn)
        screening_source = {}
        if any(obs.screening for obs in _WINDOW_SOURCE_OBSERVERS):
            screening_source = dict(fit_plan=fit_plan, W_supports=(
                jnp.copy(fit_inputs[0]), jnp.copy(fit_inputs[1])))

        # Step 2: precompile + run Σ^c(ω, k, m, n)
        #
        # The band-bracket plan is resolved HERE, once, before anything is
        # compiled: it fixes the kernel's G-build count, the AOT signature
        # and the Σ cube's leading extent, so it must be the same object all
        # three see.  ``use_band_extrapolation = false`` gives the trivial
        # one-bracket plan and the whole path below is bit-identical to the
        # un-bracketed code.
        plan = plan_sigma_band_brackets(
            config, wfns, meta, print_fn=print_fn,
            where="ppm_pipeline plan seam")
        with timing.section("sigma.exec"):
            sigma_omega = compute_sigma_c_ppm_omega_grid(
                wfns, ppm, meta, mesh_xy,
                ppm_cfg=config.ppm,
                sigma_cfg=config.sigma,
                mpa_cfg=config.mpa,
                # The omega grid only under sigma_omega_grid_debug; on shell
                # the windows are planned on the request's frequencies.
                omega_grid_ry=(config.omega_grid_ry
                               if bool(config.debug.sigma_omega_grid_debug) else None),
                ansatz=config.compute_mode,
                screening_diagrams=config.screening.diagrams,
                occupation_state=occupation_state,
                plan=plan,
                fixed_quadrature_session=(
                    None if fixed_quadrature_session is None else
                    fixed_quadrature_session.setdefault("primary", {})),
                request_for=(
                    None if bool(config.debug.sigma_omega_grid_debug) else
                    lambda efermi_ry: sigma_on_shell_request(
                        config, e_eval_ev=e_qp_ev, wfn=wfn, sym=sym, meta=meta,
                        band_slices=band_slices, efermi_ry=efermi_ry, print_fn=print_fn)),
                print_fn=print_fn,
            )
        # THE BLAST RADIUS STOPS HERE.  ``sigma_omega.sigma_c_kij`` carries
        # the band-count points; everything downstream of this line —
        # the head injection, the eqp interpolation, sigma_mnk.h5, the QSGW
        # build — is shared with MPA and COHSEX and is deliberately left at
        # the shape it has always had.  The last element IS the ordinary
        # full-band Σ_c (the cumulative sum's final term), so at
        # n_bracket = 1 this index is the identity.
        sigma_c_body_omega_n3 = (
            None if plan.enabled and not bool(
                config.debug.sigma_freq_debug_output) else
            band_count_point(sigma_omega.sigma_c_kij,
                             sigma_omega.sigma_c_kij.shape[0] - 1))
        # OFF: ``sigma_c_body_omega`` IS the N₃ point and the second cube is
        # None, so the object graph below is exactly what it always was.
        # ON: both are replaced at the report seam below.
        #
        # The combination itself happens AFTER the report block below, for
        # one reason: the coefficients ARE the fit, and the fit is built
        # there.  Deriving them here as well would be deriving the estimator
        # twice.  Nothing between here and there reads
        # ``sigma_c_body_omega``.
        sigma_c_body_omega = sigma_c_body_omega_n3
        sigma_c_body_omega_unextrap = None

        # Step 3: q→0 head construction (analytic, mini-BZ-averaged)
        head_gn = _fit_head_correction(
            head_resolver, config=config, meta=meta,
            num_electrons=float(wfn.num_electrons),
            probe_omega=probe_omega, print_fn=print_fn,
            iteration_head=iteration_head,
        )
        head_sigma_diag_w_kn_ry = _compute_analytic_head_diag(
            head_gn,
            config=config, band_slices=band_slices,
            wfn=wfn, sym=sym, meta=meta,
            iteration_head=iteration_head,
            body_efermi_ry=sigma_omega.efermi_ry,
            print_fn=print_fn,
        )

        # Step 4: the band-convergence extrapolation report.  After the head,
        # because the head is part of the Σ_c being reported; before the
        # return, because the cube's leading axis does not survive it.
        extrap_payload = None
        if plan.enabled:
            (sigma_c_body_omega, extrap_payload,
             sigma_c_body_omega_unextrap) = extrapolate_sigma_body(
                sigma_omega, head_sigma_diag_w_kn_ry, e_state_ev=e_qp_ev,
                plan=plan, config=config, band_slices=band_slices,
                wfn=wfn, mesh_xy=mesh_xy, print_fn=print_fn,
                top=sigma_c_body_omega_n3,
            )
        del sigma_c_body_omega_n3

    if _WINDOW_SOURCE_OBSERVERS:              # TD-aGW's window Sigma (observe_window_sigma_source)
        _publish_window_source(
            wfns=wfns, meta=meta, config=config, poles=ppm, head=head_sigma_diag_w_kn_ry,
            efermi_ry=sigma_omega.efermi_ry, request=getattr(sigma_c_body_omega, "request", None),
            V_q=V_q, quad=quad, **screening_source)
    return PPMOutputs(
        sigma_c_body_omega=sigma_c_body_omega,
        band_axis=sigma_omega.band_axis,
        head_sigma_diag_w_kn_ry=head_sigma_diag_w_kn_ry,
        band_extrapolation=extrap_payload,
        sigma_c_body_omega_unextrap=sigma_c_body_omega_unextrap,
        probe_hermiticity_residual=ppm.probe_hermiticity_residual,
        odd_even_residue_ratio=ppm.odd_even_residue_ratio,
        efermi_ry=sigma_omega.efermi_ry,
    )
