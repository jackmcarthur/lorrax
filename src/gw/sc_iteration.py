"""Self-consistent QSGW iteration map.

A single ``state → state`` step :func:`gw_iteration_map` and a small
Python-loop driver :func:`run_self_consistency` that wraps it.  The
state is :class:`SCState` carrying ``H_qp_dft_mnk`` in the **original
DFT basis** (so the iteration carry has a fixed coordinate system; rcrop
Anderson mixing composes meaningfully).  Every iteration:

1. Diagonalize ``H_qp_dft`` → ``(E_qp, U_qp)`` where
   ``U_qp[k, m, n] = ⟨DFT_m | QP_n⟩``.
2. Rotate the **original** DFT wfn bundle to the new QP basis via
   :func:`wavefunction_bundle.rotate_wavefunctions` (no cumulative
   U-product, no drift).
3. Recompute χ₀ → W → Σ_xc with the rotated wfns
   (:func:`sigma_dispatch.compute_sigma_xc`, mode-orthogonal).
4. Rotate ``(V_H + Σ_xc)`` back to the DFT basis and form
   ``H_qp_dft = kin_ion_dft + (V_H + Σ_xc)_dft``.

The iteration map is a pure function: ``state → state``.  The body has
no closure capture of mutable bundles; it composes trivially with rcrop
Anderson mixing or future ``jax.lax.scan`` migration.

Active / inactive partition
---------------------------
The carry ``H_qp_dft`` is sized ``(nk, nb_active, nb_active)`` where
the **active subspace** is ``band_slices.sigma = [b0, b3)`` — the bands
``kin_ion.h5`` was generated for and the bands :mod:`cohsex_sigma` /
:mod:`ppm_pipeline` compute Σ for.  Bands above ``b3`` keep their DFT
ψ throughout SC iteration.  Iteration 1 also keeps their DFT energies
exactly; later iterations apply the current conduction scissor to the
logical sum-band tail ``[b3, b4_user)`` before rebuilding χ₀ and Σ.
Mesh-padding slots ``[b4_user, b4)`` remain untouched.

Robustness assumptions for the active-space partition:

- **Insulator with sorted DFT bands**: robust.  ψ rotation within the
  active subspace preserves orthonormality with the inactive bands
  (block-diagonal U on nb_full).
- **Active block aligned with kin_ion file**: validated by the shape
  match ``kin_ion.shape[1:] == (nb_sigma, nb_sigma)`` at iteration
  init time.
- **Metals or near-gap-closure systems**: NOT robust — rotation may
  push an active "valence" band above the active "conduction" band's
  energy, or above an inactive band's energy.  ``occ`` is rebuilt
  per-band-vs-efermi so it stays correct, but downstream consumers
  (chi0's slices.val/cond split) assume a strict val/cond ordering.
  Add a re-sort + re-occupy step here if/when metals are supported.
- **Carry over multiple iterations**: ``U_qp`` is recomputed from the
  carry each iteration, so there's no accumulated U-product drift.

TODO (per design discussion 2026-05-08): inactive bands above ``b3``
that are themselves entirely within the Σ_c(ω) grid bounds at every k
should receive a *diagonal* Σ correction at each SC iteration (no
off-diagonals — they're never mixed with active bands).  Bands fully
outside the ω-grid keep the scissor extrapolation.  The "best
determined Σ for an inactive band that straddles the ω-grid edge after
SC updates" is undecided; flagged for a separate design pass.
"""

from __future__ import annotations

import functools as _functools
import math as _math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing
from common.collectives import barrier, device_put_process_local
from common.units import RYD_TO_EV
from .band_partition import BandPartition, apply_band_partition
from .efermi import (OCCUPATION_CLAMP_TOL_DEFAULT
                     as _OCCUPATION_CLAMP_TOL_DEFAULT, OccupationState)
from .gw_config import ComputeMode
from .scissor import (ScissorFit, apply_conduction_scissor_to_tail,
                      classify_scissor_bands, fit_scissor)
from .sigma_dispatch import SigmaResult, compute_sigma_xc
from .wavefunction_bundle import (
    BandSlices, Wavefunctions, rotate_wavefunctions)


# ---------------------------------------------------------------------------
# Convergence: max|dE| over the NON-SCISSORED bands
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvergenceVerdict:
    """One convergence decision, with the numbers it was made from.

    ``max_abs_ev`` IS THE TEST; the two RMS figures are reported only.
    They travel together, and print together, because they are different
    tests and the looser one has historically been the one on screen: an
    RMS can sit comfortably under a cutoff that a single band is far
    above.
    """

    converged: bool
    max_abs_ev: float        # L-inf over the non-scissored bands -- THE test
    rms_protected_ev: float  # RMS over the same set   -- reported only
    rms_all_ev: float        # RMS over ALL active bands -- the legacy number
    n_protected: int
    n_total: int
    worst_k: int
    worst_band: int
    cutoff_ev: float

    def summary(self) -> str:
        """The log line.  Says which number is the test and which is not."""
        return (
            f"max|dE| = {self.max_abs_ev:.6f} eV over {self.n_protected} "
            f"non-scissored bands (CRITERION, cutoff {self.cutoff_ev:.6f} eV; "
            f"worst k={self.worst_k} band={self.worst_band}) | "
            f"RMS_nonscissored = {self.rms_protected_ev:.6f} eV, "
            f"RMS_all({self.n_total}) = {self.rms_all_ev:.6f} eV "
            f"(diagnostics, NOT the criterion) | "
            f"{'CONVERGED' if self.converged else 'not converged'}")


def protected_band_convergence(
    e_new_ev: np.ndarray,
    e_prev_ev: np.ndarray,
    protected_mask: np.ndarray,
    in_range_mask: np.ndarray,
    cutoff_ev: float,
) -> ConvergenceVerdict:
    """Has every NON-SCISSORED band moved by less than ``cutoff_ev``?

    ``e_*_ev`` are ``(nk, nb_active)`` eV.  The test set is
    ``protected_mask | in_range_mask``, both from the ITERATION-LOCAL
    :class:`~gw.band_partition.BandPartition` -- never a band-index
    window written down somewhere else.  A frozen window keeps testing
    the bands that WERE in range; the set that matters is the one this
    iteration actually treated as physical.

    WHY THE UNION AND NOT ``protected_mask`` ALONE.  The partition is
    THREE-way and only the third category is scissored:
    ``apply_band_partition`` substitutes alpha*E_DFT + beta exactly where
    ``in_range_mask`` is False.  A band in range but NOT protected keeps
    its own Sigma-derived diagonal and merely loses off-diagonal mixing --
    a genuine degree of freedom.  Today ``run_sc_driver`` builds both
    masks from the same ``in_range``, so the two spellings agree; that is
    a property of that ONE line, not of this predicate.

    WHY SCISSORED BANDS ARE EXCLUDED.  Their energies are alpha*E_DFT +
    beta with the coefficients refitted each iteration FROM the in-range
    corrections, so including them re-counts in-range drift through the
    fit rather than measuring anything new.

    THE TEST IS L-INFINITY, NOT RMS.  "All these bands change by less than
    the cutoff" is a statement about the WORST band.  The RMS figures come
    back in the verdict for reporting and are never compared to the cutoff.
    """
    e_new = np.asarray(e_new_ev, dtype=np.float64)
    e_prev = np.asarray(e_prev_ev, dtype=np.float64)
    if e_new.shape != e_prev.shape:
        raise ValueError(
            f"protected_band_convergence: energy shapes disagree, "
            f"{e_new.shape} vs {e_prev.shape}.")
    prot = np.asarray(protected_mask, dtype=bool).reshape(-1)
    inr = np.asarray(in_range_mask, dtype=bool).reshape(-1)
    for name, m in (("protected_mask", prot), ("in_range_mask", inr)):
        if m.shape[0] != e_new.shape[1]:
            raise ValueError(
                f"protected_band_convergence: {name} has {m.shape[0]} "
                f"bands but the energies have {e_new.shape[1]}.  The masks "
                f"must be the iteration's own band partition over the SAME "
                f"active window.")
    mask = prot | inr

    delta = np.abs(e_new - e_prev)
    rms_all = float(np.sqrt(np.mean(delta ** 2)))
    n_protected = int(mask.sum())
    if n_protected == 0:
        # Refuse rather than declare victory over the empty set: "max over
        # {} < cutoff" is vacuously true.
        raise ValueError(
            "protected_band_convergence: the band partition leaves ZERO "
            "non-scissored bands (protected | in_range is empty), so there "
            "is no set to converge on -- every active band's diagonal would "
            "be a scissor value.  Widen sigma_omega_min_ev / "
            "sigma_omega_max_ev so some active band lies on the Sigma grid.")

    d_prot = delta[:, mask]
    max_abs = float(d_prot.max())
    rms_prot = float(np.sqrt(np.mean(d_prot ** 2)))
    flat = int(np.argmax(d_prot))
    worst_k, worst_col = divmod(flat, d_prot.shape[1])
    worst_band = int(np.flatnonzero(mask)[worst_col])
    return ConvergenceVerdict(
        converged=bool(max_abs < float(cutoff_ev)),
        max_abs_ev=max_abs,
        rms_protected_ev=rms_prot,
        rms_all_ev=rms_all,
        n_protected=n_protected,
        n_total=int(e_new.shape[1]),
        worst_k=int(worst_k),
        worst_band=worst_band,
        cutoff_ev=float(cutoff_ev),
    )


class _Converged(Exception):
    """Internal: stop the rCROP solve because the criterion was met.

    rCROP's loop has no convergence callback, and the criterion is an
    L-infinity norm on EIGENVALUES (eV) rather than anything the solver
    can express, so it is signalled out of ``residual_fn``.  Carries the
    state so the caller need not reconstruct it.
    """

    def __init__(self, state: "SCState", verdict: ConvergenceVerdict):
        super().__init__(verdict.summary())
        self.state = state
        self.verdict = verdict


# ---------------------------------------------------------------------------
# State + inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SCInputs:
    """Quantities held constant across self-consistent iterations.

    The wfn bundle here is the **original DFT bundle** — the iteration
    map rotates copies of it on demand and never mutates it.

    ``partition`` is the active-subspace band classification
    (protected / non-protected-in-range / out-of-range).  Default
    ``BandPartition.all_protected(nb_active)`` reduces the masking step
    to the identity, so existing one-shot paths are unchanged until
    the partition is configured deliberately.

    ``e_dft_active_kn_ry`` and ``valence_mask_active_kn`` feed the
    per-iteration scissor refit; they are constant across iterations
    (DFT band identities + occupation labels don't move).
    """

    wfns_dft: Wavefunctions
    V_q: jax.Array
    kin_ion_dft: jax.Array
    # ``gw.head_channel.HeadChannel`` or None.  Carried per-iteration because
    # every SC iteration re-solves W from the SAME V_q, so the Coulomb
    # placement has to travel with it or iteration 2 would quietly revert to
    # the default placement while iteration 1 used the deck's.
    head_channel: object
    quad: object             # static minimax quadrature for χ₀
    e_ref: float
    static_head_terms: object | None
    head_resolver: object
    config: object
    meta: object
    mesh_xy: Mesh
    sym: object
    wfn: object              # WFNReader (for vbm/efermi anchor + paths)
    centroid_indices: object  # ISDF centroid set — IBZ resolve for the static W
    band_slices: BandSlices
    input_dir: str
    partition: BandPartition
    e_dft_active_kn_ry: jax.Array      # (nk, nb_active) DFT energies for scissor fit
    valence_mask_active_kn: jax.Array  # (nk, nb_active) bool — for scissor val/cond split
    #: IBZ ⇄ full-BZ map for BAND-INDEX quantities.  ``None`` ⇒ the loop
    #: runs entirely on the full BZ, which is what every result before
    #: 2026-08-04 did and what the one-shot equivalence gate pins.  When
    #: given, H / E / U / the carried state live on the IBZ and only Σ is
    #: built on the full BZ — Σ comes from an FFT over the k-grid, which
    #: needs the whole grid (decisions.md, TRS veto scope).
    kstar: object | None = None
    #: Validated, device-resident Berry connection + exact DFT velocity.
    #: None preserves the historical fixed-DFT head path exactly.
    parallel_transport: object | None = None
    print_fn: Callable = print


@dataclass(frozen=True)
class SCOutputs:
    """Output-only records captured from one map call.

    Purely for the final output writers (``run_sc_driver``'s finalize,
    ``dump_sigma_omega_h5_final``, ``dump_qp_wfn_artifacts``, the per-map
    snapshot); nothing here feeds the next fixed-point evaluation.

    ``sigma_result`` and ``sigma_basis_U`` are on the FULL BZ and must
    AGREE, because they are consumed together by ``run_sc_driver``'s
    final rotate-back — Σ is a k-grid FFT, so its k-set is not
    negotiable, and the U stored beside it follows.  ``sigma_basis_U``
    is the DFT→QP unitary that DEFINED the basis ``sigma_result`` was
    computed in (the eigh of the *previous* carry) — the writer must
    rotate Σ back to DFT with THIS U, not the converged U of the final
    carry: the two agree only at convergence, and using the converged U
    mis-rotates Σ_x/V_H whenever the loop stops before the fixed point
    (maximally so at max_iter=1, where the correct U is the identity).
    """

    sigma_result: SigmaResult
    sigma_basis_U: jax.Array         # (nk, nb, nb) ⟨DFT_m|QP_n⟩, full BZ
    scissor_fit: ScissorFit | None
    tail_scissor_fit: ScissorFit | None


@dataclass(frozen=True)
class SCState:
    """State carried across self-consistent iterations.

    K-SET INVARIANT (with ``SCInputs.kstar``): ``H_qp_dft`` is on the
    LOOP's k-set — the IBZ when a map is given — while ``outputs``
    carries full-BZ objects (see :class:`SCOutputs`).

    The primary iteration carry is ``H_qp_dft``.  When metallic occupations
    are enabled, ``occupation_state`` is the ONE per-iteration
    ``gw.efermi.OccupationState`` (padded parallel-transport head window,
    fixed-N μ, family, width, hash) plus its derived tetrahedron
    Fermi-surface table: iteration i's first-half consumers read the values
    produced only at the END of iteration i-1.  They are deliberately not
    installed in the wavefunction bundle: under mpa_material_class = metal
    the state is threaded explicitly into screening (build_mpa_fit) and
    Sigma (compute_sigma_xc); insulating decks pass None and keep the
    historical occupations bit-exactly.

    ``outputs`` (an ``SCOutputs`` record) is purely for the final output
    writers; it does not feed the next iteration.  Its ``sigma_basis_U``
    is the DFT→QP unitary that DEFINED the basis ``sigma_result`` was
    computed in (the eigh of the *previous* carry) — the writer must
    rotate Σ back to DFT with THIS U, not the converged U of the final
    carry: the two agree only at convergence, and using the converged U
    mis-rotates Σ_x/V_H whenever the loop stops before the fixed point
    (maximally so at max_iter=1, where the correct U is the identity).
    """

    H_qp_dft: jax.Array              # (nk, nb_active, nb_active) Ry, DFT basis
    iteration: int
    # DIAGNOSTIC continuity only (mu-drift log): the map ENTRY-solves its
    # own occupation state from the spectrum of the H it is handed, so
    # correctness never depends on these two fields — F(H) is a self-map
    # of H alone, which is what rCROP's trial/accept trajectory assumes.
    occupation_state: OccupationState | None = None  # full-BZ, padded PT head manifold
    head_surface_weight_kn: jax.Array | None = None  # Nk * tetrahedron delta(E-mu)
    outputs: SCOutputs | None = None


# ---------------------------------------------------------------------------
# Initial state from DFT
# ---------------------------------------------------------------------------

def make_initial_state_from_dft(inputs: SCInputs) -> SCState:
    """``H_qp_dft^(0) = diag(E_DFT)`` on the active subspace.

    Iteration 1's ``eigh`` of a diagonal matrix returns ``(E_DFT, U=I)``
    so the first Σ-pipeline call uses the unrotated DFT wfns and "one
    iteration of QSGW" reduces exactly to one-shot G0W0 at E=E_DFT.
    """
    from common.wfn_transforms import get_enk_bandrange
    enk_dft, _ = get_enk_bandrange(
        inputs.wfn, inputs.sym,
        inputs.band_slices.sigma_range, inputs.band_slices.sigma_range,
        nspinor=inputs.meta.nspinor)
    enk_dft_ry = np.asarray(enk_dft, dtype=np.float64)
    nk, nb_active = enk_dft_ry.shape
    # Per-k diagonal of E_DFT_kn — broadcast cast to complex128 for the
    # iteration carry.
    H0 = (enk_dft_ry[:, :, None] * np.eye(nb_active)[None, :, :]).astype(
        np.complex128)
    # The carried state lives on whatever k-set the loop runs on.  With a
    # k-star map that is the IBZ, so H0 is selected here ONCE rather than
    # every iteration; diag(E_DFT) is star-consistent by construction, so
    # the selection is exact and the iteration-0 shortcut below (which
    # requires H0 to be exactly diagonal) still fires.
    ks = getattr(inputs, "kstar", None)
    if ks is not None and not ks.is_identity:
        H0 = ks.select(H0)
    rep = NamedSharding(inputs.mesh_xy, P(None, None, None))
    # Process-local replication (plain ``jax.device_put`` of host numpy
    # onto a multi-process sharding fires JAX's hidden ``assert_equal``
    # all-gather, P × nk × nb² × 16 B — scorecard AA.1).  ``H0`` is a
    # pure function of the DFT energies, identical on every rank;
    # ``LORRAX_CHECK_REPLICA=1`` re-arms the assertion.
    occ_state, head_surface_weight_kn = _solve_head_occupations(
        inputs, inputs.wfns_dft.enk)
    if occ_state is not None:
        inputs.print_fn(
            "  SC head occupations: initialized BGW-MP1 state at "
            f"mu={occ_state.mu_ry * RYD_TO_EV:.8f} eV "
            f"[occ_hash {occ_state.occ_hash}]")
        if _material_class(inputs) == "metal":
            # Metal-run startup gate: re-solving on the WFN's OWN stored
            # eigenvalue/weight table must reproduce its stored occupations,
            # or the smearing family/width does not match the deck that made
            # the WFN.  ``config.occ_broadening_ry`` is the deck's ONE width
            # and already carries BGW's convention, so QE's 'mp' matches it
            # at occ_smearing_width_ry = degauss/2 — this gate is what
            # catches a deck that got that factor wrong.
            from .efermi import (OccupationState as _OS,
                                 assert_wfn_occupation_consistency)
            from psp.get_DFT_mtxels import spin_degeneracy_factor
            w_ibz = np.asarray(inputs.wfn.kweights, dtype=np.float64)
            w_ibz = w_ibz / w_ibz.sum()
            capacity = float(spin_degeneracy_factor(inputs.wfn))
            check_state = _OS.solve_mp1(
                np.asarray(inputs.wfn.energies[0], dtype=np.float64),
                w_ibz,
                float(inputs.wfn.num_electrons),
                inputs.config.occ_broadening_ry,
                state_capacity=capacity,
                clamp_tol=float(inputs.config.occupation_clamp_tol))
            deviation = assert_wfn_occupation_consistency(
                check_state, np.asarray(inputs.wfn.occs[0]), w_ibz,
                state_capacity=capacity,
                num_electrons=float(inputs.wfn.num_electrons))
            inputs.print_fn(
                "  SC head occupations: WFN consistency max|Δf| = "
                f"{deviation:.3e}, mu_check = "
                f"{check_state.mu_ry * RYD_TO_EV:.8f} eV (wfn.efermi "
                f"{inputs.wfn.efermi * RYD_TO_EV:.6f} eV is the midgap/VBM "
                "convention, reported not asserted)")
    return SCState(
        H_qp_dft=device_put_process_local(H0, rep),
        iteration=0,
        occupation_state=occ_state,
        head_surface_weight_kn=head_surface_weight_kn,
    )


def _material_class(inputs) -> str:
    """The deck's declared MPA material class; 'insulator' when absent."""
    mpa = getattr(inputs.config, "mpa", None)
    return str(getattr(mpa, "material_class", "insulator"))


def _solve_head_occupations(
    inputs: SCInputs,
    energies_kn_ry,
) -> tuple[OccupationState | None, jax.Array | None]:
    """Solve the per-iteration fixed-N MP1 occupation state.

    Returns ``(OccupationState, surface_weight_kn)`` — the one occupation
    record the head consumes today and chi/Σ consume at Wave-2 wiring, plus
    its derived tetrahedron Fermi-surface table.  The state's ``f_kn`` is
    full-BZ because the parallel-transport velocity and head contraction are
    full-BZ, padded to the PT storage width; padding is explicit and inert
    (exact zeros move neither the count nor the hash's meaning) and the
    solver sees exactly ``nb_logical`` physical bands.  A zero width returns
    ``(None, None)`` so the historical step-occupation path is bit-for-bit
    untouched.
    """
    # ``occ_broadening`` is the DIAL (0 ⇒ step occupations, historical path
    # untouched); ``config.occ_broadening_ry`` is the WIDTH.  They are the
    # same key on an insulating deck and cross-checked on a metal one.
    width_ev = float(inputs.config.screening.occ_broadening_ev)
    pt = getattr(inputs, "parallel_transport", None)
    if width_ev == 0.0 or pt is None:
        return None, None

    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    if energies.ndim != 2:
        raise ValueError(
            f"QSGW head occupation energies must be (nk,nb), got {energies.shape}.")
    nb_logical = int(pt.nb_logical)
    # The velocity is the one large dataset BOTH head modes carry; the
    # connection is absent under sc_head_update = dft_velocity.  They are
    # written with identical shape, so the storage width is unchanged.
    nb_storage = int(pt.velocity_dft_cart.shape[-1])
    if not (0 < nb_logical <= nb_storage <= int(energies.shape[1])):
        raise ValueError(
            "QSGW head occupation manifold must satisfy 0 < logical <= "
            f"storage <= energy bands; got {nb_logical}, {nb_storage}, "
            f"{energies.shape[1]}.")

    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from .efermi import assert_fixed_n, solve_mp1_occupations

    nk = int(energies.shape[0])
    kweights = np.full(nk, 1.0 / float(nk), dtype=np.float64)
    capacity = float(spin_degeneracy_factor(inputs.wfn))
    target_electrons = float(inputs.wfn.num_electrons)
    width_ry = inputs.config.occ_broadening_ry
    mu_ry, occ_logical = solve_mp1_occupations(
        energies[:, :nb_logical],
        kweights,
        target_electrons,
        width_ry,
        state_capacity=capacity,
        clamp_tol=float(inputs.config.occupation_clamp_tol),
    )
    occ_kn = jnp.pad(
        occ_logical,
        ((0, 0), (0, nb_storage - nb_logical)),
        mode="constant",
        constant_values=0.0,
    )
    occ_state = OccupationState(
        f_kn=occ_kn,
        mu_ry=float(mu_ry),
        smearing_family="mp1",
        smearing_width_ry=float(width_ry),
        n_electrons=target_electrons,
    )
    # The pad bands are exact zeros, so the padded table satisfies the same
    # fixed-N invariant the logical solve does.
    assert_fixed_n(occ_state, kweights, state_capacity=capacity)
    # The Drude weight is a Fermi-surface integral, not a coarse-grid
    # sampling of the MP1 derivative.  On sodium 8^3 the latter moves the
    # plasma frequency from 6.09 to 7.68 eV.  Tetrahedra use MP1 only to
    # determine the fixed-N chemical potential; no fitted plasma frequency
    # enters the response.
    from .fermi_surface import (
        star_symmetrize_weights,
        tetrahedron_delta_weights,
    )

    surface_logical = tetrahedron_delta_weights(
        np.asarray(energies[:, :nb_logical], dtype=np.float64),
        np.asarray(inputs.sym.unfolded_kpts, dtype=np.float64),
        tuple(int(x) for x in inputs.wfn.kgrid),
        float(mu_ry),
        symmetry_matrices=np.asarray(inputs.sym.sym_mats_k),
    )
    # POST-INTEGRATION STAR SYMMETRIZATION.  The six tetrahedra all share one
    # hardcoded (1,1,1) body diagonal, so the weight table is NOT star
    # covariant: measured on this deck's converged state, 4 of 48 crystal ops
    # leave it invariant, N(E_F) varies by 0.0425 states/Ry/cell inside a
    # single star, and the Drude tensor comes out 2.68 percent anisotropic on
    # cubic sodium.  Averaging over each star restores the crystal point
    # group exactly and moves neither sum_kn w_kn nor the Drude trace.  It
    # belongs HERE rather than inside the quadrature because this is where
    # the star labels live -- `sym.irr_idx_k` is already in hand, and the
    # quadrature is deliberately symmetry-free.  O(nk*nb).
    surface_logical = star_symmetrize_weights(
        surface_logical, np.asarray(inputs.sym.irr_idx_k))
    # The distributed contraction owns an explicit uniform 1/Nk.  Convert
    # normalized-BZ tetrahedron weights to its per-grid-point interface.
    surface_kn = jnp.pad(
        jnp.asarray(surface_logical * float(nk), dtype=jnp.float64),
        ((0, 0), (0, nb_storage - nb_logical)),
        mode="constant",
        constant_values=0.0,
    )
    return occ_state, surface_kn


def _assert_index_mask_matches_classes(inputs: SCInputs, classes) -> None:
    """Refuse if the step mask and the three-way rule disagree.

    On a deck with STEP occupations the two are the same statement: with no
    band crossing E_F, "below the lowest crossing band" degenerates to
    "fully occupied", which is the ``arange(nb) < meta.nelec`` cut that
    ``run_sc_driver`` freezes into ``valence_mask_active_kn``.  Saying so as
    a refusal rather than a comment is what keeps the insulating path
    byte-identical as the metal path grows: if the two ever part, the
    insulating scissor has silently changed class on some band, and that is
    exactly the failure this whole change is about.
    """
    nb = int(inputs.e_dft_active_kn_ry.shape[1])
    want = np.arange(nb) < int(inputs.meta.nelec)
    got = np.arange(nb) < int(classes.valence_stop)
    if int(classes.n_crossing) or not np.array_equal(want, got):
        raise ValueError(
            "SC scissor: a step-occupation ('fixed' family) deck must "
            "reproduce the frozen index mask exactly, but the three-way "
            f"classification gives {classes.summary()} against "
            f"meta.nelec = {int(inputs.meta.nelec)} over {nb} active bands.")


# ---------------------------------------------------------------------------
# Iteration map
# ---------------------------------------------------------------------------

def _make_kshard_eigh(mesh_xy: Mesh, *, eigvalsh_only: bool,
                      u_spec: P | None = None):
    """Return a jit'd eigh that briefly k-shards the input over the mesh
    so each device only does its slice of the per-k diagonalisations,
    then allgathers the eigenvalues (and U if requested) back to
    replicated.  Pure perf hint — the math is identical to running
    ``vmap(eigh)`` on the replicated input.

    ``nk`` NEED NOT divide ``mesh_xy.size``.  ``with_sharding_constraint``
    is a layout hint and GSPMD shards the k axis unevenly when it has to —
    some devices simply get one fewer k.  (This docstring previously
    asserted the opposite; job 7889742 ran the ``sc_on_ibz`` arm green at
    P=4 with ``nk_irr = 10``, and ``_run_linear_mixing`` calls the
    eigvalsh kernel on that ``(10, nb, nb)`` carry — ``10 % 4 != 0``.
    ``dsc_demo/ibz44v.7889742.out:26``.)  What it costs at ``nk <
    mesh.size`` is idle devices, not a failure.
    """
    rep_E = NamedSharding(mesh_xy, P(None, None))
    # ``u_spec`` chooses where U LANDS.  The SC loop asks for
    # ``qsgw_density.band_rotation_spec`` (``P(None,'x','y')``, so no rank
    # holds a full (nb, nb)); the default replicates it and is kept for
    # ``final_qp_eigenstates``, whose only consumers are host writers.
    # Parametrised rather than copied: the eigh itself, the k-shard hint
    # and the hermitisation are identical and must not drift.
    rep_U = NamedSharding(mesh_xy,
                          P(None, None, None) if u_spec is None else u_spec)
    k_shard_3d = NamedSharding(mesh_xy, P(('x', 'y'), None, None))

    # Replication is an ENFORCED output contract (out_shardings), not a
    # body hint: at P=4 the trailing with_sharding_constraint was dropped
    # by the partitioner and the host read local-shard-plus-zeros — 22 of
    # 29 IBZ rows exactly 0.0 in every eqp snapshot, max|dE| = VBM to six
    # decimals, and the MP1 mu solved on a three-quarters-zero table
    # (QUALITY_PATTERNS §4: the optimized HLO is the only ground truth).
    if eigvalsh_only:
        @_functools.partial(jax.jit, out_shardings=rep_E)
        def _f(H):
            H_k = jax.lax.with_sharding_constraint(H, k_shard_3d)
            H_h = 0.5 * (H_k + jnp.conj(jnp.swapaxes(H_k, -1, -2)))
            return jax.vmap(jnp.linalg.eigvalsh)(H_h)
        return _f
    else:
        @_functools.partial(jax.jit, out_shardings=(rep_E, rep_U))
        def _f(H):
            H_k = jax.lax.with_sharding_constraint(H, k_shard_3d)
            H_h = 0.5 * (H_k + jnp.conj(jnp.swapaxes(H_k, -1, -2)))
            E, U = jax.vmap(jnp.linalg.eigh)(H_h)
            return E, U
        return _f


# Kernel cache.  The eigh is keyed by ``(mesh, u_spec)`` because ``u_spec``
# changes its output sharding and therefore its lowering; the eigvalsh has
# no U and is cached on its own key so a second U layout does not retrace
# it.  Re-used across all SC iterations, so the JIT cost is paid once.
_KSHARD_EIGH_CACHE: dict[tuple, object] = {}


def _kshard_eigh_kernels(mesh_xy: Mesh, u_spec: P | None = None) -> tuple:
    key = (id(mesh_xy), u_spec)
    eigh = _KSHARD_EIGH_CACHE.get(key)
    if eigh is None:
        eigh = _make_kshard_eigh(mesh_xy, eigvalsh_only=False, u_spec=u_spec)
        _KSHARD_EIGH_CACHE[key] = eigh
    ev_key = (id(mesh_xy), "eigvalsh")
    eigvalsh = _KSHARD_EIGH_CACHE.get(ev_key)
    if eigvalsh is None:
        eigvalsh = _make_kshard_eigh(mesh_xy, eigvalsh_only=True)
        _KSHARD_EIGH_CACHE[ev_key] = eigvalsh
    return eigh, eigvalsh


def _midgap_efermi(E: jax.Array, n_occ: int) -> jax.Array:
    """Fixed-band-cut midgap E_F from ascending eigenvalues.

    One spelling, two callers (:func:`_diagonalize_and_get_efermi` and
    :func:`gw_iteration_map`), because which EIGH ran and which E_F rule
    applies are now independent decisions and the rule must not be
    duplicated inside one of the eigh branches.

    Valid for an insulator with a fixed occupied count; the general
    answer, and the one the IBZ needs, is
    :func:`gw.efermi.fermi_level_step` with star weights.
    """
    vbm = jnp.max(E[:, :n_occ])
    cbm = jnp.where(n_occ < E.shape[1], jnp.min(E[:, n_occ:]), vbm)
    return 0.5 * (vbm + cbm)


def _diagonalize_and_get_efermi(
    H: jax.Array, n_occ: int, mesh_xy: Mesh, u_spec: P | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Hermitise + eigh + midgap E_F.  Returns (E, U, efermi_ry).

    Per-k eighs are briefly k-sharded over the device mesh so each
    device only does ``nk / mesh_size`` of them.  The midgap reduction
    runs on the gathered E (small, replicated).

    ``u_spec`` is where U lands; ``None`` replicates it.  Pass
    ``qsgw_density.band_rotation_spec()`` when the CALLER's consumers are
    the device-side rotations, and leave it ``None`` when they are host
    writers — see the two call sites.
    """
    eigh_kshard, _ = _kshard_eigh_kernels(mesh_xy, u_spec)
    E, U = eigh_kshard(H)
    return E, U, _midgap_efermi(E, n_occ)


# The largest share of the per-device memory budget one (nb, nb) tile is
# allowed to take before the native eigh stops being acceptable.
#
# The native path is a k-sharded BATCH: each device runs whole per-k
# eighs, so it materialises the input tile, the eigenvector tile and
# LAPACK's workspace — call it three tiles — on ONE device, on top of ψ,
# the FFT boxes and the ω-cube.  Capping ONE tile at 1% of the budget
# therefore caps the eigh's single-device footprint near 3%.
#
# Derived from bytes and the budget rather than from a band count, so it
# tracks the device it runs on.  Where 1% puts the switch, against the
# budgets ``gw_config`` actually resolves:
#
#   80 GB GPU   → budget 72 GB (0.9·bytes_limit)      → nb ≈ 6.7e3
#   CLX node    → budget 169 GB (0.9·RAM / n_devices) → nb ≈ 1.0e4
#   8 GB device → budget 7.2 GB                       → nb ≈ 2.2e3
#
# which is the band the owner ruling names — robustness at 1e4+ bands
# over speed at 1e3, where the native batch solves ndev matrices at once
# and wins by roughly ndev (``distrib_la.resolve``, eigh ``auto``
# policy).  3% was the first choice and was wrong on the CPU arm: the
# CPU budget is the whole node's RAM divided by the JAX device count, so
# with several ranks per node it over-counts, and 3% of 169 GB puts the
# switch past nb = 1.8e4 — it would not have fired on the nb = 1e4 case
# the distributed eigh exists for.
_SC_EIGH_TILE_BUDGET_FRACTION = 0.01


def _resolve_sc_eigh(nb: int, mesh_xy: Mesh, config, *, print_fn) -> str:
    """``"native"`` or ``"distributed"`` for this iteration's eigh.

    A LAYOUT decision and nothing else.  It used to be a side effect of
    ``density_self_consistent`` — a physics knob, defaulting to False —
    so the only eigh that keeps no whole ``(nb, nb)`` tile on one rank
    was unreachable on the default path.  ``config.sc.eigh`` selects it
    now; the E_F rule stays where it was, with ``density_self_consistent``.

    ``"auto"`` picks ``distributed`` only when both hold:

    * the mesh has more than one device — on one device "distributed" is
      the same tile with an FFI call around it;
    * one tile exceeds :data:`_SC_EIGH_TILE_BUDGET_FRACTION` of the
      per-device budget.

    and then only if the distributed backend actually resolves on this
    mesh.  ``resolve_backend`` is the probe: it raises at RESOLVE time
    naming the failed guard (platform, uncompiled handler, mesh geometry,
    divisibility), so ``auto`` degrades to native with the reason printed
    rather than failing inside the eigh.  An explicit request is not
    probed — it must raise.

    DIVISIBILITY IS NO LONGER A CONDITION (2026-08-06).  It used to be a
    third clause here, and an explicit ``sc_eigh = distributed`` used to
    raise on an indivisible ``nb``, because ``distributed_eigh_bands``
    padded both band axes to the divisor and did **not** undo the pad —
    returning ``(nk, nb_pad)`` / ``(nk, nb_pad, nb_pad)``, a silent shape
    change the carry and every band-indexed operand beside it would not
    match.  That callee now pads with a large diagonal sentinel and slices
    back BY COUNT, so it returns the LOGICAL extent at any ``nb``.  The
    refusal had nothing left to protect.  Note what actually changed: the
    old objection was a SHAPE objection, not a spectral one — a sentinel
    alone would not have answered it, and zero-padding without the slice
    would still be wrong (pad eigenvalues at exactly 0.0 sort into the
    middle of a Ry spectrum and move band order, ``_midgap_efermi`` and
    the occupations).  It took both halves.

    The backend probe is asked about ``round_up(nb, pad_div)`` — the
    extent the eigh actually runs at — not about ``nb``.  Probing ``nb``
    would trip ``resolve.py``'s own divisibility guard and degrade every
    indivisible window to native, i.e. reinstate the lifted refusal by
    accident.
    """
    requested = str(getattr(getattr(config, "sc", None), "eigh", "auto"))
    if requested == "native":
        return "native"

    from common.mtxel_sweep import band_sphere_spec
    from runtime.padding import spec_divisor, round_up

    ndev = int(mesh_xy.size)
    px, py = (int(mesh_xy.shape[a]) for a in mesh_xy.axis_names)
    pad_div = spec_divisor(mesh_xy, band_sphere_spec(), 1)
    # ``distributed_eigh_bands`` pads to this and slices BACK by count, so
    # an indivisible nb is no longer a reason to refuse or to degrade — it
    # is just a pad.  The backend probe below must therefore be asked about
    # the extent the eigh actually runs at, not about nb.
    nb_solve = round_up(nb, pad_div)

    if requested == "distributed":
        return "distributed"

    tile_b = float(nb) * float(nb) * 16.0
    budget_b = float(getattr(getattr(config, "memory", None),
                             "per_device_gb", 0.0)) * 1e9
    big = budget_b > 0.0 and tile_b > _SC_EIGH_TILE_BUDGET_FRACTION * budget_b
    if ndev <= 1 or not big:
        return "native"

    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import resolve_backend
    try:
        resolve_backend("eigh", "distributed", mesh_xy, n=nb_solve)
    except Exception as exc:                                  # noqa: BLE001
        print_fn(
            f"  SC eigh: auto wanted the distributed eigh (one (nb, nb) tile "
            f"is {tile_b / 2**30:.3f} GiB, over "
            f"{_SC_EIGH_TILE_BUDGET_FRACTION:.0%} of the "
            f"{budget_b / 1e9:.1f} GB/device budget) but the backend refused "
            f"— {type(exc).__name__}: {exc}.  Falling back to the k-sharded "
            f"native batch, which puts that whole tile on ONE device.")
        return "native"
    return "distributed"


def _band_rotation_spec() -> P:
    """``gw.qsgw_density.band_rotation_spec()``, resolved lazily.

    Reused rather than re-spelled — a second literal ``P(None,'x','y')``
    in this file would be exactly the drift that spec is a function to
    avoid, and a spec that disagreed would not raise, it would insert a
    silent reshard between the eigh and every ψ rotation.  Lazy because
    every other reference to ``gw.qsgw_density`` in this module is too:
    its import graph adds the FFT and matrix-element helpers, which most
    importers of ``sc_iteration`` do not need.
    """
    from gw.qsgw_density import band_rotation_spec
    return band_rotation_spec()


def _place(x, mesh: Mesh, spec: P | None = None) -> jax.Array:
    """``x`` as a global array on ``mesh`` at ``spec`` (default replicated).

    Three input kinds reach the U consumers and each needs a different
    route; a single ``jnp.asarray`` is wrong for two of them.

    * an already-correctly-placed ``jax.Array`` — ``jax.device_put`` onto
      the same sharding is a no-op, so the default SC path pays nothing;
    * a ``jax.Array`` at another mesh layout, which is what
      ``qsgw_density.distributed_eigh_bands`` and
      ``_make_kshard_eigh(u_spec=...)`` both emit — ``device_put`` reshards
      it on the device.  ``jnp.asarray`` would leave it where it was and
      ``np.asarray`` raises "Fetching value for jax.Array that spans
      non-addressable (non process local) devices" at P>1 (measured,
      job 7889419);
    * a HOST array, which is what the k-star broadcast produces on a
      reduced k-set — ``device_put_process_local``, because plain
      ``jnp.asarray`` builds a SINGLE-DEVICE array (an operand-sharding
      error against the mesh-sharded operands at P>1) and plain
      ``jax.device_put`` fires JAX's hidden replica ``assert_equal``
      all-gather (common.collectives header).

    ``spec=None`` means replicated, which is still what the ω-grid and
    eqp writers want.  The U consumers ask for
    ``qsgw_density.band_rotation_spec`` instead — see
    :func:`_rotate_to_dft_basis`.
    """
    nd = int(np.ndim(x))
    sh = NamedSharding(mesh, P(*([None] * nd)) if spec is None else spec)
    if isinstance(x, jax.Array):
        return jax.device_put(x, sh)
    return device_put_process_local(x, sh)


@_functools.partial(jax.jit, static_argnames=("mesh",))
def _rotate_to_dft_basis(O_qp: jax.Array, U: jax.Array, *,
                         mesh: Mesh) -> jax.Array:
    """``O_DFT[m, n] = Σ_pq U[m, p] · O_QP[p, q] · U[n, q]^*`` per k.

    Two calls into ``qsgw_density.rotate_band_matrix``, i.e. the SAME
    primitive ``rotate_bands`` uses, applied once per index.  U STAYS AT
    ``band_rotation_spec`` — no rank holds a full (nb, nb) of it, and
    the (nk, nb, nb) intermediate is sharded too.

    ONLY THE RESULT IS PINNED REPLICATED, and it has to be: the SC carry
    is ``kin_ion + this``, and ``_run_rcrop``, ``_run_linear_mixing`` and
    ``_scissor_E_qp_for_outofrange`` all read the carry back on the host,
    which raises the non-addressable-devices error on a sharded array at
    P>1.  ``O_qp`` arrives replicated from ``compute_sigma_xc`` for the
    same reason.  So this seam still holds two replicated (nk, nb, nb)
    objects; what it no longer holds is the two U-shaped ones (U itself
    and the rotation's intermediate), which is 2/4 of its former peak.
    Making the carry itself distributed is a separate change and would
    have to move those three host readbacks first.

    THIS REPLACES A GATHERED U, AND THAT WAS A DELIBERATE CHOICE ONCE.
    The previous form pinned U replicated and did the whole contraction
    locally, measured at nk=16/nb=128 on a 2×2 mesh (job 7889423):

        U sharded, layout inferred    3 collectives   7.63 ms  out P(None,'x')
        U sharded, result pinned      4 collectives  12.35 ms  out replicated
        U pinned replicated (old)     2 collectives  10.11 ms  out replicated

    It is the right trade only while U fits: replicated U is 9.2 GB/rank
    at nk=144/nb=2000, which is the scaling target's refusal case, and
    the same owner ruling that put ``distributed_eigh_bands`` on this
    layout (2026-08-04, robustness at 1e4+ bands over speed at 1e3)
    applies here.  Measured at nk=8/nb=32 (job 7889851), per-rank U:
    0.1250 MiB replicated → 0.0312 MiB at 2×2 and 0.0078 MiB at 4×4,
    exactly px·py; module argument bytes 0.2500 → 0.1562 (2×2) →
    0.1328 MiB (4×4).

    Gathering U also made the result bit-identical to the fully-replicated
    path; the distributed form reassociates the band sums and does not —
    5.4e-16 relative at worst over both directions and P ∈ {1, 4, 16}
    (``tests/multi_device/band_rotate_gate.py``, job 7889851), against an
    explicit host rotation at 5.6e-16.
    """
    from gw.qsgw_density import rotate_band_matrix

    out = rotate_band_matrix(O_qp, U, mesh=mesh, to_qp=False)
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, P(None, None, None)))


# ---------------------------------------------------------------------------
# Density self-consistency: rebuild V_H from the CURRENT orbitals
# ---------------------------------------------------------------------------
#
# OFF BY DEFAULT (``config.density_self_consistent``).  With it off this
# module is byte-identical to before, which is what keeps
# tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot
# meaningful.

_PSI_G_CACHE: dict = {}


def _dft_psi_sphere(inputs):
    """DFT ψ(G) on the SC k-set, loaded ONCE and cached.

    The SC bundle carries ψ at ISDF CENTROIDS, which cannot reconstruct
    ρ on the FFT grid, so the density rebuild needs the G-sphere.  ψ_DFT
    is constant across iterations — only U moves — so this is one read per
    run, not one per iteration.

    THE BAND RANGE IS GLOBAL, THE CARRY'S EXTENT IS b0-RELATIVE.
    ``WfnLoader.load`` indexes the file's bands, ``[0, wfn.nbands)``
    (``wfn_loader.py:1158-1165``), while ``kin_ion_dft`` is
    ``(nk, nb_sigma, nb_sigma)`` with ``nb_sigma = b3 − b0``
    (``load_kin_ion_submatrix``, and the shape check in this module's
    header note).  This used to read ``bands=(0, kin_ion_dft.shape[1])``,
    which is the correct window only while ``b0 == 0`` and silently the
    WRONG BANDS otherwise — ``[0, nb_sigma)`` instead of ``[b0, b3)``.
    V_H is an O(400 Ry) term and the band count would still be right, so
    neither the electron-count check in ``rho_from_wfns`` nor any norm or
    hermiticity check downstream would see it.  Take the range from
    ``band_slices.sigma_range``, which IS the global pair, and never
    reconstruct one from an extent.
    """
    from jax.sharding import NamedSharding as _NS
    from common.mtxel_sweep import band_sphere_spec

    b_lo, b_hi = inputs.band_slices.sigma_range
    nb_sigma = int(inputs.kin_ion_dft.shape[1])
    if (b_hi - b_lo) != nb_sigma:
        raise ValueError(
            f"_dft_psi_sphere: band_slices.sigma_range={(b_lo, b_hi)} spans "
            f"{b_hi - b_lo} bands but the SC carry is {nb_sigma} wide.  These "
            f"describe the same active subspace and a mismatch means one of "
            f"them is b0-relative where the other is global.")
    # Key on the GLOBAL RANGE, not on its width: two windows of equal
    # extent at different b0 are different ψ and must not share a cache
    # entry.
    key = (id(inputs.wfn), b_lo, b_hi)
    hit = _PSI_G_CACHE.get(key)
    if hit is None:
        # ONE-TIME ROW.  The section is entered every iteration but only
        # the first does work, so ``vh.psi_load``'s count is the iteration
        # count and its total is the single WFN.h5 read.  It exists so
        # that read shows as its own row instead of as unexplained SELF
        # time on ``vh.rebuild``.
        with timing.section("vh.psi_load"):
            spec = band_sphere_spec()
            psi = inputs.wfn.load(
                bands=(b_lo, b_hi), k="full_bz", sharding=spec)
            bidx = np.asarray(inputs.wfn.box_index(k="full_bz"))
        hit = (psi, bidx)
        _PSI_G_CACHE[key] = hit
    return hit


def _kstar(inputs):
    """The loop's k-star map; identity when symmetry is not in use.

    Returning an identity map rather than ``None`` is what lets
    ``gw_iteration_map`` be written ONCE: ``select``/``broadcast`` are
    no-ops on it, so the full-BZ path is the same code, not a branch.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import KStarMap
    ks = getattr(inputs, "kstar", None)
    if ks is not None:
        return ks
    return KStarMap.identity(int(inputs.kin_ion_dft.shape[0]))


# THE DENSITY-SC ROW, AND ITS FOUR CHILDREN.  ``vh.rebuild`` is the whole
# rebuild; under it the timing tree carries ``vh.psi_load``,
# ``vh.rotate_bands``, ``vh.rho``, ``vh.poisson`` and ``mtxel.sweep``.
# No ``watch`` on the parent: every child that produces a device array
# blocks on it (see each one's note), so the last statement of this
# function has already been synchronised and the inclusive time is real.
# What is left in the parent's SELF column is host work only —
# ``padded_gvectors`` and the two scalars ``gw.efermi`` brings back (E_F
# and the electron count).  ``fermi_level_step`` and ``step_occupations``
# are jit kernels, so E is no longer read back here.
#
# MEASURED, MoS2 4x4 / nb=128 / nk=16 (IBZ 10) / N_mu=785 / P=1, 5
# iterations, job 7889362.  Per iteration, against a 62.4 s SC iteration:
#
#   vh.rebuild        8.90 s   14.3 %
#     mtxel.sweep     5.72 s    9.2 %   <m|V_H|n> over 16 k x 128 bands
#     vh.rho          2.96 s    4.7 %   16 k x 128 band inverse FFTs
#     vh.poisson      0.12 s    0.2 %
#     vh.rotate_bands 0.04 s    0.1 %
#     vh.psi_load     0.05 s    0.1 %   (0.272 s once, amortised)
#     self            0.01 s
#
# The same deck with ``density_self_consistent`` off runs the driver in
# 269.6 s against 312.2 s (53.9 vs 62.4 s/iteration, same job), so the
# difference confirms the rows: +8.5 s/iteration measured end to end
# against the +8.90 s the tree reports.  The cost is NOT the rotation the
# module docstrings worry about — it is the matrix-element sweep, which
# alone exceeds this deck's whole chi0+W screening (5.51 s/iteration).
@timing.timed("vh.rebuild")
def rebuild_hartree_dft_basis(inputs, U_qp, E_qp_ry):
    """⟨m|V_H[ρ_i]|n⟩ in the DFT basis, from iteration i's own orbitals.

    The cycle this closes::

        H_i --eigh--> E, U_qp --+--> psi_qp = rotate(psi_dft, U_qp)
                                +--> occ = f(E);  rho_i = sum_n f|psi_qp|^2
                                +--> V_H[rho_i],  W_i,  Sigma_i
        H_{i+1} = T + V_loc + V_NL + V_H[rho_i] + Sigma[psi_i, W_i]

    V_H and Sigma are built from the SAME iteration-i orbitals and both
    land in H_{i+1}, so the fixed point is rho = rho[psi(H)] with H
    containing V_H[rho].  ``kin_ion_dft`` stays pristine T+V_loc+V_NL, so
    V_H arrives only through ``delta_h`` and cannot double-count.

    RETURNED IN THE DFT BASIS, AND CONTRACTED THERE.  psi is rotated for
    RHO -- rho needs the occupied QP orbitals -- but the matrix elements
    are contracted with the UNROTATED psi_dft, because H is assembled in
    the DFT basis around a pristine ``kin_ion_dft``.  A DFT-basis V_H adds
    to it directly, with NO rotation at all.  Contracting with psi_qp
    instead would put V_H in the QP basis only for the caller to rotate it
    straight back, an nk*nb^3 round trip for the same number.

    No mixing: straight rho_out feedback, by owner ruling (2026-08-04).
    """
    from gw.efermi import (fermi_level_step, occupied_band_count,
                           step_occupations)
    from gw.qsgw_density import rotate_bands, hartree_from_orbitals
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from common.mtxel_sweep import (SweepGeometry, local_potential_operator,
                                    sweep_matrix_elements)
    from psp.dft_operators import padded_gvectors

    psi_G, bidx = _dft_psi_sphere(inputs)
    nk, nb = int(psi_G.shape[0]), int(psi_G.shape[1])
    kweights = np.full(nk, 1.0 / nk)          # full BZ => uniform, no star

    # E stays on the device: both are jit kernels over ``E`` (``gw.efermi``
    # header) and only E_F and the degeneracy flag cross.
    if _material_class(inputs) == "metal":
        # Metal ρ: fixed-N MP1 occupations solved on THIS iteration's QP
        # spectrum (W3 update point).  The step path below cannot represent
        # a metal (partial fill / degenerate-manifold refusal), and the
        # constructor owns the fixed-N invariant.
        width_ev = float(inputs.config.screening.occ_broadening_ev)
        if width_ev <= 0.0:
            raise ValueError(
                "mpa_material_class = metal requires screening occ_broadening "
                "> 0 to solve fixed-N MP1 occupations for the density; got "
                f"{width_ev!r} eV.")
        occ_state = OccupationState.solve_mp1(
            E_qp_ry, kweights, float(inputs.wfn.num_electrons),
            inputs.config.occ_broadening_ry,
            state_capacity=float(spin_degeneracy_factor(inputs.wfn)),
            clamp_tol=float(inputs.config.occupation_clamp_tol))
        e_f = occ_state.mu_ry
        occ = occ_state.f_kn
        inputs.print_fn(
            "    V_H rebuild: metal MP1 occupations, "
            f"mu={e_f * RYD_TO_EV:.8f} eV [occ_hash {occ_state.occ_hash}]")
    else:
        e_f = fermi_level_step(E_qp_ry, kweights, float(inputs.meta.nelec))
        occ = step_occupations(E_qp_ry, e_f)

    # Rotated orbitals: needed for rho, NOT for the contraction below.
    psi_qp = rotate_bands(psi_G, U_qp, mesh=inputs.mesh_xy)
    f_spin = spin_degeneracy_factor(inputs.wfn)
    grid = tuple(int(v) for v in inputs.wfn.fft_grid)
    _rho, V_H_r = hartree_from_orbitals(
        psi_qp, occ, kweights, inputs.wfn, mesh=inputs.mesh_xy,
        box_index=bidx, fft_grid=grid,
        truncation_2d=bool(getattr(inputs.config, "sys_dim", 3) == 2),
        spin_degeneracy=f_spin,
        # ``occupied_band_count`` rather than a second einsum: same
        # contraction, and it is the number ``fermi_level_step`` targets.
        expected_electrons=f_spin * occupied_band_count(occ, kweights),
        print_fn=inputs.print_fn)

    gtab = padded_gvectors(inputs.wfn, k="full_bz")
    geom = SweepGeometry(mesh=inputs.mesh_xy, fft_grid=grid,
                         ngkmax=int(psi_G.shape[3]), nb=nb,
                         ns=int(psi_G.shape[2]), nk=nk,
                         cell_volume=float(inputs.wfn.cell_volume))
    H_vh = sweep_matrix_elements(
        psi_G, operator=local_potential_operator(geom, V_H_r), geom=geom,
        gvecs=gtab.gvecs, gmask=gtab.mask, box_index=bidx,
        kvecs=np.asarray(inputs.sym.unfolded_kpts))
    return H_vh, e_f


def _residency_census(named, print_fn) -> None:
    """One line per named array: global bytes, bytes addressable HERE, spec.

    The SC loop's (nk, nb, nb)-class objects are what decide how many bands
    a rank can hold, and a replicated one is invisible in any aggregate
    number — every rank simply has its own full copy.  Reading the bytes
    off ``addressable_shards`` is the only way to tell "sharded" from
    "sharded in the docstring": a ``device_put`` that silently fell back to
    replicated reports the global size here.

    Printed once per run (iteration 0), so it costs one host round trip of
    metadata and no device work.
    """
    print_fn("    SC residency census (global / addressable here):")
    for name, a in named:
        if a is None:
            continue
        try:
            local = sum(sh.data.nbytes for sh in a.addressable_shards)
            spec = getattr(getattr(a, "sharding", None), "spec", "?")
        except AttributeError:
            local, spec = int(np.asarray(a).nbytes), "host"
        print_fn(
            f"      {name:22s} {tuple(a.shape)!s:18s} "
            f"{a.nbytes / 2**20:10.3f} / {local / 2**20:9.3f} MiB  "
            f"1/{a.nbytes / max(local, 1):<4.0f} {spec}")


# Refusal threshold on ``KStarMap.spread_rel`` of Σ + V_H, relative.
#
# MEASURED on healthy mos2_4x4 runs, every ``k-star:`` line recorded on
# scratch: 1.470e-12 … 7.559e-12 on the linear/rCROP arms
# (dsc_demo/ibz44v.7889742.out:26,28,30; dsc44.7889362.out:121-129) and
# 2.303e-11 … 1.178e-10 on the density-SC arms, which is where the
# largest value sits (dsc_demo/dev/dev1_p1.7889590.out:11,19 at P=1;
# dsc44.7889362.out:13,23 at P=4).  The largest observed is 1.178e-10.
#
# 1e-6 is four decades above that and four decades below the failure it
# exists for: a gauge mismatch puts a phase on the off-diagonals of the
# doubled stars, so its residual is a fixed FRACTION of max|Σ| — O(1e-2)
# relative or worse — while every mechanism that legitimately grows this
# number (a larger k-grid, more bands, a longer float64 accumulation)
# grows it by decades, not by eight.  Do not tighten it toward the
# observed maximum: the cost of a false refusal on a 40-node job is a
# dead run, and the check discriminates just as well from here.
_KSTAR_SPREAD_TOL = 1.0e-6


def _check_kstar_spread(kstar, delta_h_qp, *, print_fn) -> float:
    """Enforce the star spread of Σ + V_H before selecting the IBZ rows.

    ``KStarMap.spread``/``spread_rel`` is documented as the only check
    that catches a gauge mismatch introduced upstream — hermiticity, the
    norm and the electron count all survive one — and the value was
    formatted into a log line and dropped.  A number that is only printed
    is not a check, and this one is the sole guard on the two-k-set seam.

    REFUSES, does not warn: the whole point is that nothing downstream
    notices.  A warning on rank 0 of a 64-rank job is a line in a log
    somebody reads after the run produced numbers.

    One reduction and one 16-byte host read, which the iteration pays
    already (the accelerators read the eigenvalues back every call).
    """
    spread = float(kstar.spread_rel(delta_h_qp))
    print_fn(
        f"    k-star: Σ+V_H residual {spread:.3e} rel "
        f"over {kstar.nk_full}->{kstar.nk_irr} k ({kstar.reduction:.2f}x)")
    # ``not (x <= tol)`` and not ``x > tol``: NaN must refuse.
    if not (spread <= _KSTAR_SPREAD_TOL):
        raise ValueError(
            f"k-star spread of Σ+V_H is {spread:.6e} relative, above the "
            f"refusal threshold {_KSTAR_SPREAD_TOL:.1e}.  Members of a star "
            f"must carry the same Σ up to round-off; they do not, so the "
            f"full-BZ Σ and the IBZ carry are in different gauges and "
            f"selecting the star representatives would silently keep the "
            f"wrong one.  Healthy runs on this deck measure ≤ 1.2e-10.  "
            f"Suspect the wavefunction rotation or the symmetry map, not "
            f"convergence: hermiticity, the norm and the electron count all "
            f"survive this fault.")
    return spread


def _check_sigma_stage(sigma_result: SigmaResult, *, print_fn) -> None:
    """The Σ stage gates, once per SC iteration.

    ``gw_jax`` applies these four to the Σ it builds, inside
    ``if qp_solver is not QPSolver.SELF_CONSISTENT:`` — so the ONE path
    that rebuilds Σ_x in a rotated band basis, 2·max_iter + 1 times, was
    the one path with no gate on it.  They belong here rather than there
    anyway: they are per-iteration invariants, and the SC loop is the
    place a band-index or conjugation slip can enter, because it is the
    only place the Σ basis is not the DFT one.  Spelling is deliberately
    identical to ``gw_jax``'s so there is one set of messages to grep.

    Σ_x[i,i] = −Σ_{m∈occ} ⟨im|V|mi⟩ is a negative-definite quadratic form
    in a positive-semidefinite kernel, so the sign and the −200…0 eV
    bracket hold in ANY orthonormal band basis; nothing here assumes the
    DFT one.  The bracket is loose on purpose (bare exchange runs
    −40…−5 eV on the production decks): it catches a unit or
    basis-normalisation slip, not physics.

    COST.  Four device reductions to ≤ 4 scalars, hence four host syncs
    per iteration.  The iteration already synchronises three times
    (``eigvalsh_kshard``, the k-star spread, the scissor's ``np.asarray``
    of the H diagonal), so this adds reductions, not a new class of
    stall.  The diagonal is taken ON DEVICE: ``gw_jax`` writes
    ``np.diagonal(np.asarray(sigma_x))``, which pulls the whole
    (nk, nb, nb) to the host — 9.2 GB at nk=144/nb=2000 — and that would
    be the most expensive thing in the iteration.

    NOT applied here: ``average_within_degenerate_sets``.  It needs the
    band energies on the host and only narrows the diagonal's spread, so
    omitting it makes both gates harder to pass, never easier.

    NOT covered: a NaN confined to Σ_c.  It reaches ``sigma_xc_kij_ry``
    but not ``sigma_x_kij_ry``, and ``rcrop_nojit``'s ``res <= tol`` is
    False for NaN, so such a run still costs the full ``maxit``.  Stated
    because it is the remaining hole on this surface, not because it is
    fixed here.
    """
    from common import sanity

    sanity.check_finite("Σ_x", sigma_result.sigma_x_kij_ry, print_fn=print_fn)
    sanity.check_finite("V_H", sigma_result.v_h_kij_ry, print_fn=print_fn)
    sig_x_diag_ev = jnp.real(jnp.diagonal(
        sigma_result.sigma_x_kij_ry, axis1=1, axis2=2)) * RYD_TO_EV
    sanity.check_sign("Σ_x diagonal (eV)", sig_x_diag_ev,
                      expect="negative", print_fn=print_fn)
    sanity.check_in_range("Σ_x diagonal (eV)", sig_x_diag_ev,
                          -200.0, 0.0, unit="eV", print_fn=print_fn)


def gw_iteration_map(state: SCState, inputs: SCInputs) -> SCState:
    """One self-consistent QSGW step in the DFT basis.

    Pure function — no side effects on ``inputs.wfns_dft``.  All
    derived quantities (E_qp, U_qp, efermi) are recomputed each call;
    the only carried state is ``H_qp_dft`` on the active subspace.

    Screening is mode-orthogonal: each iteration asks
    :func:`gw.screening.compute_screening_model` for the configured Σ
    scheme's screening representation and hands the resulting
    ``{role → W_q}`` dict to
    :func:`gw.sigma_dispatch.compute_sigma_xc`.  No ``compute_chi0``
    call lives here directly — adding a new Σ scheme that wants extra
    W frequencies is purely a screening + compute_sigma_xc change.
    """
    from .screening import compute_screening_model, screening_requests_for

    n_occ = int(inputs.meta.nelec)
    E_qp_ry = U_qp = None
    if state.iteration == 0:
        # The canonical initial carry (``make_initial_state_from_dft``)
        # is EXACTLY diag(E_DFT); its eigensystem is (E_DFT, I) by
        # construction.  Do NOT run eigh on it: LAPACK roundtrips the
        # eigenvalues at ~1 ulp, and the GN-PPM two-point fit amplifies
        # ulp-scale enk noise to O(0.1–1 eV) in Σ_c(ω) via near-threshold
        # pole modes (measured on the MoS2 3×3 fixture: +1 ulp on every
        # WFN energy → max|ΔΣ_c| = 1.28 eV; same ill-conditioning family
        # as the Fix-3 on-pole census sensitivity in
        # reports/device_invariance_2026-07-08/ROOT_CAUSE.md).  The exact
        # eigensystem keeps SC-iteration-1 ≡ one-shot G0W0 bit-exactly
        # (gated by tests/test_invariance_gates.py::
        # test_sc_iteration1_equals_one_shot).
        #
        # TWO MORE THINGS THE BYPASS PINS THAT ``eigh`` DOES NOT PROMISE.
        # (a) ORDER: eigh returns eigenvalues ASCENDING, while every
        # band-indexed operand here — ``e_dft_active_kn_ry``,
        # ``valence_mask_active_kn``, ``slices.val``/``cond`` — is in the
        # WFN's band order.  They coincide only while E_DFT is sorted at
        # every k; the bypass makes the band labelling identity by
        # construction instead of by coincidence.
        # (b) GAUGE IN A DEGENERATE MANIFOLD: for repeated eigenvalues the
        # eigenvector basis is arbitrary up to a unitary on the degenerate
        # block, and LAPACK does not promise the identity even for an
        # exactly diagonal input.  Any such mixing rotates ψ, and Σ's
        # off-diagonals are not invariant under it.  Returning U = I
        # removes the dependence rather than relying on it.
        #
        # The predicate below is EXACT (bitwise all-zero off-diagonal),
        # not a tolerance, so it cannot fire on a carry that is merely
        # nearly diagonal, and a non-finite diagonal makes the difference
        # non-zero and falls through to eigh.  ``.real`` on the diagonal is
        # exact for the only producer of iteration 0
        # (``make_initial_state_from_dft`` writes a real diagonal).
        H_np = np.asarray(state.H_qp_dft)
        nb = H_np.shape[1]
        diag = np.diagonal(H_np, axis1=1, axis2=2)
        if not np.any(H_np - diag[:, :, None] * np.eye(nb)[None]):
            E_np = np.ascontiguousarray(diag.real)
            vbm = E_np[:, :n_occ].max()
            cbm = E_np[:, n_occ:].min() if n_occ < nb else vbm
            efermi_ry = 0.5 * (vbm + cbm)
            rep2 = NamedSharding(inputs.mesh_xy, P(None, None))
            # U AT band_rotation_spec, NOT REPLICATED.  This is the same
            # (nk, nb, nb) object ``distributed_eigh_bands`` emits sharded
            # at every iteration ≥ 1 (9.2 GB replicated at nb=2000/nk=144),
            # and iteration 0 was the last producer still handing a
            # replicated one to ``rotate_wavefunctions`` /
            # ``qsgw_density.rotate_bands``.  Sharding it is free for both:
            # ``rotate_bands`` takes this layout as-is (measured 115.7 ms
            # against 117.6 ms replicated, same three collectives, argument
            # 41 MiB against 44 MiB — job 7889424), and
            # ``rotate_wavefunctions`` reshards to ``band_mix_spec``
            # whichever it gets.  ``_rotate_to_dft_basis`` contracts in
            # this layout directly and no longer gathers it.
            # Process-local placement — see the H_qp_dft note above (same
            # hidden assert_equal; same rank-invariance argument, and here
            # each rank stages only its own nb²/(px·py) block).
            E_qp_ry = device_put_process_local(E_np, rep2)
            U_qp = device_put_process_local(
                np.broadcast_to(
                    np.eye(nb, dtype=np.complex128), H_np.shape),
                NamedSharding(inputs.mesh_xy, _band_rotation_spec()))
    if E_qp_ry is None:
        # TWO INDEPENDENT DECISIONS, and they used to be one condition.
        #
        # (a) WHICH EIGH -- a LAYOUT question, answered by
        # ``_resolve_sc_eigh`` from ``config.sc.eigh``.  The k-sharded
        # batch gives each device nk/P of the per-k diagonalisations but
        # still lands one WHOLE (nb, nb) tile on one device -- 1.6 GB at
        # nb=1e4; ``distributed_eigh_bands`` spreads each tile over the
        # mesh instead (owner ruling 2026-08-04: robustness at 1e4+ bands
        # over speed at 1e3, where the native batch wins by ~ndev).  Until
        # 2026-08-05 the distributed one was reachable ONLY by turning on
        # ``density_self_consistent``, a physics knob defaulting to False,
        # so the default -- and only shipped -- configuration had no way
        # to ask for it.
        #
        # (b) WHICH E_F RULE -- a PHYSICS question, and it stays with
        # ``density_self_consistent``: the k-weighted step routine there,
        # the fixed-band-cut midgap otherwise.  Moving it would change
        # numbers; moving (a) does not.
        #
        # Both eigh branches return U at ``band_rotation_spec``, so
        # everything below is layout-blind.  The k-sharded one used to
        # allgather U back to replicated by default; every consumer here
        # is a device-side rotation that either wants
        # ``band_rotation_spec`` outright (``qsgw_density.rotate_bands``)
        # or reshards from whatever it is given (``rotate_wavefunctions``
        # → ``band_mix_spec``), and the two matrix rotations
        # (``_rotate_to_dft_basis`` and ``sigma_dispatch``'s V_H basis
        # change) contract in this layout through
        # ``qsgw_density.rotate_band_matrix``.  Per-rank U drops by px·py
        # — 4.00 MiB → 1.00 MiB at nk=16/nb=128 on a 2×2 mesh (job
        # 7889423), and it is the (nk, nb, nb) object that reaches 9.2 GB
        # at nb=2000/nk=144.
        nb_carry = int(state.H_qp_dft.shape[1])
        eigh_kind = _resolve_sc_eigh(
            nb_carry, inputs.mesh_xy, inputs.config,
            print_fn=inputs.print_fn)
        # ``<= 1``, NOT ``== 0``.  Iteration 0 reaches this block only when
        # the carry is not exactly diagonal, which the canonical
        # ``make_initial_state_from_dft`` never produces — so an
        # ``== 0`` guard printed nothing on any real run (job 7890020,
        # every arm).  Iteration 1 is the first that actually runs an
        # eigh; both are allowed so a non-canonical initial carry still
        # reports.
        if state.iteration <= 1:
            inputs.print_fn(
                f"    SC eigh: {eigh_kind} (nb={nb_carry}, one (nb, nb) tile "
                f"= {nb_carry * nb_carry * 16 / 2**20:.2f} MiB, "
                f"sc_eigh="
                f"{getattr(getattr(inputs.config, 'sc', None), 'eigh', 'auto')})")
        if eigh_kind == "distributed":
            from gw.qsgw_density import distributed_eigh_bands
            E_qp_ry, U_qp = distributed_eigh_bands(
                state.H_qp_dft, mesh=inputs.mesh_xy,
                distrib_la_batched_route=(
                    getattr(inputs.config.backend,
                            "distrib_la_batched_route", "auto")))
        else:
            eigh_kshard, _ = _kshard_eigh_kernels(
                inputs.mesh_xy, _band_rotation_spec())
            E_qp_ry, U_qp = eigh_kshard(state.H_qp_dft)

        if bool(getattr(inputs.config, "density_self_consistent", False)):
            from gw.efermi import fermi_level_step

            from .scissor import k_star_weights
            # THE SECOND REDUCTION OVER k IN THIS LOOP, and it needs the
            # same star weights the scissor does: the electron count is
            # Σ_k w_k Σ_n f_nk, so on the IBZ each star must carry its
            # multiplicity.  ``1/nk`` here would count the 6 doubled stars
            # of mos2_4x4 once each and put E_F in the wrong place.
            # ``fermi_level_step`` wants weights summing to 1 over its own
            # k-set (efermi.py:50-53), hence the divide by nk_full.
            w_k = k_star_weights(_kstar(inputs))
            efermi_ry = fermi_level_step(
                E_qp_ry, w_k / float(w_k.sum()), float(n_occ))
        else:
            efermi_ry = _midgap_efermi(E_qp_ry, n_occ)

    # Rotate the active subspace of the DFT bundle to this iteration's QP
    # basis.  Bands outside ``slices.sigma`` always keep their DFT ψ.  From
    # iteration 2 onward, logical conduction-sum bands above b3 receive the
    # current active-space conduction scissor in ENERGY only; iteration 1
    # keeps the historical DFT ladder exactly, preserving the one-shot gate.
    # THE ONE PLACE THE TWO k-SETS MEET.  H, E and U live on the IBZ; the
    # bundle, W and Σ live on the full BZ because Σ is an FFT over the
    # k-grid and needs the whole grid.  ``broadcast`` is an index gather
    # plus a conjugation on time-reversed members -- a band index is
    # symmetry-inert, so no umklapp phase or centroid permutation enters
    # (see symmetry_maps.maps, above star_select).
    # The broadcast is a device gather and keeps the operand's sharding
    # (``symmetry_maps``, ``_row_out_sharding``), so U_full arrives
    # at ``band_rotation_spec`` — what ``rotate_bands`` and
    # ``rotate_wavefunctions`` want — and U never crosses to the host.  It
    # needs no ``_place`` first, unlike the host-numpy form it replaces.
    ks = _kstar(inputs)
    U_full = U_qp if ks.is_identity else ks.broadcast(U_qp)
    E_full = E_qp_ry if ks.is_identity else ks.broadcast(E_qp_ry)

    # ENTRY-SOLVED metallic occupations: one MP1 state per map CALL, from
    # the spectrum of the H actually being mapped.  This makes the
    # iteration a genuine self-map F(H) = Sigma[H, occ(H)] — the contract
    # _run_rcrop's own header states ("gw_iteration_map reads
    # state.iteration and state.H_qp_dft and nothing else") and the one
    # rCROP's trial/accept trajectory requires: every F(H) evaluation,
    # trial or accepted, gets occupations consistent with ITS H by
    # construction.  The previous flow solved this same quantity at
    # END-of-iteration and carried it into the NEXT call — a
    # one-generation lag (audit A5) that was exact only on the
    # mixing=1 linear trajectory, which is why rCROP was refused on
    # metallic decks.  At a fixed point H* = F(H*) the two rules
    # coincide, so the converged answer is unchanged.  Insulating decks
    # (occ_broadening = 0) return (None, None): bit-identical path.
    #
    # SOLVED HERE, ABOVE THE TAIL SCISSOR, so that ONE occupation state
    # serves the whole map call — including BOTH scissor fits, which now
    # classify their val/cond/crossing bands from it.  The ladder handed
    # over is the same one ``rotate_wavefunctions`` assembles for
    # ``wfns_qp.enk`` (``wavefunction_bundle.py:616-619``): the DFT ladder
    # with the active block replaced by this iteration's eigenvalues.  The
    # only columns that can differ from the old spelling
    # (``_solve_head_occupations(inputs, wfns_qp.enk)``, called after the
    # rotation) are the sum-band tail ``[b3, b4_user)`` on iterations that
    # ran the tail scissor — bands hundreds of eV above mu, where f is
    # exactly 0 under either ladder.  Reading them from the tail scissor
    # was also the circular half: the tail scissor's own band classes now
    # come from this state.
    # Replicated, like the bundle's own enk (``rotate_wavefunctions``
    # constrains it to ``P(None, None)`` right after the same update), so
    # the host-side MP1 solve cannot meet an array that spans
    # non-addressable devices.
    with inputs.mesh_xy:
        enk_entry = jax.lax.with_sharding_constraint(
            jnp.asarray(inputs.wfns_dft.enk).at[
                :, inputs.band_slices.sigma].set(
                    jnp.asarray(E_full, dtype=inputs.wfns_dft.enk.dtype)),
            NamedSharding(inputs.mesh_xy, P(None, None)))
    entry_occ_state, entry_surface_weight_kn = _solve_head_occupations(
        inputs, enk_entry)
    # THE THREE-WAY SCISSOR CLASSIFICATION, once per map call.  Valence is
    # everything below the lowest Fermi-crossing band, conduction everything
    # above the highest, and the crossing bands enter NEITHER fit (owner
    # ruling 2026-08-16; the damage the old occupied-index mask did on
    # sodium is quantified in gw/scissor.py's header and claim 0212).
    # ``None`` on an insulating deck: the caller keeps the index mask, which
    # IS this rule when nothing crosses.
    scissor_classes = None
    if entry_occ_state is not None:
        scissor_classes = classify_scissor_bands(entry_occ_state.f_kn)
        if str(entry_occ_state.smearing_family) == "fixed":
            _assert_index_mask_matches_classes(inputs, scissor_classes)
        inputs.print_fn(
            f"    SC scissor classes: {scissor_classes.summary()}")

    # ENERGY-ONLY SCISSOR FOR THE SUM-BAND TAIL.  No new iteration state:
    # the fit is derived from the current carry's eigenspectrum and the
    # immutable active DFT ladder.  The logical stop is b4_user, not padded
    # b4; apply_conduction_scissor_to_tail copies padding bit-for-bit.  The
    # optional ladder is consumed by rotate_wavefunctions, which remains the
    # single owner of occupation rebuilding after an energy change.
    enk_base = None
    tail_fit = None
    tail_start = int(inputs.band_slices.sigma.stop)
    logical_stop = (
        int(inputs.meta.b_id_4_user) - int(inputs.band_slices.b0))
    if state.iteration > 0 and logical_stop > tail_start:
        from .scissor import k_star_weights

        e_dft_fit = inputs.e_dft_active_kn_ry
        valence_fit = inputs.valence_mask_active_kn
        if not ks.is_identity:
            e_dft_fit = ks.select(e_dft_fit)
            valence_fit = ks.select(valence_fit)
        e_dft_fit_ev = np.asarray(e_dft_fit, dtype=np.float64) * RYD_TO_EV
        fit_mask_kn = np.broadcast_to(
            np.asarray(inputs.partition.in_range_mask, dtype=bool)[None, :],
            e_dft_fit_ev.shape)
        # SAME three-way classification as the active-window scissor below.
        # It matters here too: the tail law that gets applied is the
        # CONDUCTION one, and under the old index mask a Fermi-crossing
        # band above the occupied cut (sodium's band 10 of nval = 10) was a
        # conduction sample.
        valence_kn = np.asarray(valence_fit, dtype=bool)
        if scissor_classes is not None:
            valence_kn, crossing_kn = scissor_classes.masks(e_dft_fit_ev.shape)
            fit_mask_kn = fit_mask_kn & ~crossing_kn
        tail_fit = fit_scissor(
            E_dft_kn_ev=e_dft_fit_ev,
            E_qp_kn_ev=(
                np.asarray(E_qp_ry, dtype=np.float64) * RYD_TO_EV),
            valence_mask_kn=valence_kn,
            fit_mask_kn=fit_mask_kn,
            k_weights=k_star_weights(ks),
        )
        enk_base_ev = apply_conduction_scissor_to_tail(
            np.asarray(inputs.wfns_dft.enk, dtype=np.float64) * RYD_TO_EV,
            tail_fit,
            tail_start=tail_start,
            logical_stop=logical_stop,
        )
        enk_base = device_put_process_local(
            enk_base_ev / RYD_TO_EV,
            NamedSharding(inputs.mesh_xy, P(None, None)))
        inputs.print_fn(
            f"    SC sum-band tail: scissored [{tail_start}, "
            f"{logical_stop}) with conduction "
            f"alpha={tail_fit.alpha_c:+.4f}, "
            f"beta={tail_fit.beta_c_ev:+.4f} eV "
            f"(n={tail_fit.n_fit_c}, w={tail_fit.w_fit_c:.0f})")

    wfns_qp = rotate_wavefunctions(
        inputs.wfns_dft, U_full,
        enk_active_new=E_full, enk_base=enk_base,
        efermi=float(efermi_ry),
        mesh_xy=inputs.mesh_xy,
        active_slice=inputs.band_slices.sigma,
    )

    # (``entry_occ_state`` was solved above, before the tail scissor that
    # feeds ``enk_base`` — see the block after ``E_full``.)

    # DENSITY SELF-CONSISTENCY (opt-in).  V_H[rho_i] from THIS iteration's
    # orbitals, alongside Sigma_i and from the same U_qp, both feeding
    # H_{i+1}.  Off by default, so the one-shot equivalence gate holds.
    v_h_dft_new = None
    if bool(getattr(inputs.config, "density_self_consistent", False)):
        # rho is built from FULL-BZ psi (uniform weights, no star sum),
        # so it takes the broadcast U and E; the matrix it returns is
        # selected to the IBZ to match delta_h_dft.
        v_h_dft_new, e_f_ry = rebuild_hartree_dft_basis(
            inputs, U_full, E_full)
        if not ks.is_identity:
            v_h_dft_new = ks.select(v_h_dft_new)
        inputs.print_fn(
            f"    density-SC: rebuilt V_H from iteration {state.iteration} "
            f"orbitals (E_F = {e_f_ry:.6f} Ry)")

    # Per-mode screening plan.  The q->0 head uses this exact frequency/role
    # table so a Schur-folded probe can never drift from the body W it folds.
    requests = screening_requests_for(
        inputs.config.compute_mode, inputs.config)

    mpa_mode = inputs.config.compute_mode is ComputeMode.MPA
    mpa_plan = None
    if mpa_mode:
        from .mpa import sample_plan
        from .mpa.model import make_mpa_plan

        mpa_plan = make_mpa_plan(inputs.config, inputs.quad)
        mpa_z = np.asarray(sample_plan.plan_z(mpa_plan), dtype=np.complex128)
        head_omegas = [complex(value) for value in mpa_z]
        # The metallic MPA grid starts just above the origin.  Preserve a
        # separate exact-static sample for do_G0; it is nonanalytic and must
        # never enter the Loewner sample vector.
        if bool(inputs.config.do_G0) and not np.any(mpa_z == 0.0 + 0.0j):
            head_omegas.append(0.0 + 0.0j)
    else:
        head_omegas = (
            [complex(req.omega_ry) for req in requests]
            if requests else [0.0 + 0.0j])

    # Per-iteration QSGW q->0 head.  The opt-in map is stationary even for
    # accelerators that evaluate one carry repeatedly: at iteration zero
    # DeltaH=0 and U=I, so this reconstructs the DFT head through the same
    # prevalidated path used thereafter.  Saved A/v and the carried H build
    # the direct tensor.  The already-resident, already-rotated centroid
    # wavefunctions build q-linear wings; they are not reloaded here.
    iteration_head = None
    iteration_head_response = None
    iteration_static_head_terms = inputs.static_head_terms
    pt = getattr(inputs, "parallel_transport", None)
    if pt is not None:
        from .head_correction import compute_static_head_terms_from_sample
        from .qsgw_head import (
            assemble_delta_head_manifold,
            build_iteration_head_response,
            finalize_iteration_head_samples,
        )

        nb_storage = int(pt.velocity_dft_cart.shape[-1])
        if int(wfns_qp.enk.shape[1]) < nb_storage:
            raise ValueError(
                "parallel-transport head storage has "
                f"{nb_storage} padded bands, but the SC wavefunction bundle "
                f"has only {wfns_qp.enk.shape[1]}.")
        # ``sc_head_update = dft_velocity`` carries no connection, so DeltaH
        # has no consumer: skip its assembly rather than building an
        # O(nk*nb_storage^2) manifold and a spectral derivative for a term
        # that is then dropped.  Everything below this point is shared.
        connection_cart = pt.connection_cart
        delta_head = None
        if connection_cart is not None:
            H_active_full = (
                state.H_qp_dft if ks.is_identity
                else ks.broadcast(state.H_qp_dft))
            e_dft_active = inputs.e_dft_active_kn_ry
            nb_active = int(H_active_full.shape[-1])
            h_dft_active = (
                e_dft_active[:, :, None]
                * jnp.eye(nb_active, dtype=jnp.complex128)[None, :, :])
            delta_active = H_active_full - h_dft_active
            tail_diagonal = (wfns_qp.enk[:, :nb_storage]
                             - inputs.wfns_dft.enk[:, :nb_storage])
            delta_head = assemble_delta_head_manifold(
                delta_active, tail_diagonal, nb_storage=nb_storage,
                mesh=inputs.mesh_xy)

        head_occ_kn = wfns_qp.occ[:, :nb_storage]
        head_efermi_ry = float(efermi_ry)
        head_surface_weight_kn = None
        if entry_occ_state is not None:
            # Entry-solved THIS call from wfns_qp.enk, so the shapes are
            # right by construction; the invariant worth pinning is that
            # the head consumes the SAME mu the entry solve produced —
            # exact equality, not a tolerance (the rewiring regression
            # this guards is a consumer drifting back to a carried or
            # midgap reference).
            head_occ_kn = entry_occ_state.f_kn
            head_efermi_ry = float(entry_occ_state.mu_ry)
            assert head_efermi_ry == float(entry_occ_state.mu_ry)
            head_surface_weight_kn = entry_surface_weight_kn

        iteration_head_response = build_iteration_head_response(
            delta_head,
            connection_cart,
            pt.velocity_dft_cart,
            U_full,
            wfns_qp.enk[:, :nb_storage],
            head_occ_kn,
            np.asarray(head_omegas, dtype=np.complex128),
            surface_weight_qp_kn=head_surface_weight_kn,
            mesh=inputs.mesh_xy,
            kgrid=tuple(int(n) for n in inputs.wfn.kgrid),
            bvec_cart=pt.reciprocal_lattice_cart,
            nb_logical=int(pt.nb_logical),
            sigma_energies_ry=np.asarray(E_full, dtype=np.float64),
            efermi_ry=head_efermi_ry,
            wfn=inputs.wfn,
            meta=inputs.meta,
            config=inputs.config,
            # Y and Z are built directly from the two centroid-sharded
            # wavefunction copies.  Their band-pair tiles are distributed
            # over the full Px*Py mesh and frequency-blocked in each ring.
            wfns_qp=wfns_qp,
            eta_ry=(0.0 if mpa_mode else None),
        )
        velocity_kind = (
            "QSGW covariant velocity" if connection_cart is not None
            else "QP-rotated DFT p-matrix velocity")
        if mpa_mode:
            inputs.print_fn(
                "    SC head: occupation-aware QSGW response plus sharded "
                f"ISDF wings on the exact MPA z grid ({velocity_kind}, "
                f"nb={pt.nb_logical}, fit samples={mpa_z.size})")
        else:
            inputs.print_fn(
                f"    SC head: {velocity_kind} + current-basis wings "
                "from saved parallel transport/current centroid bundle "
                f"(nb={pt.nb_logical}, samples={len(head_omegas)})")


    # Per-mode screening: solve W at every frequency the Sigma scheme needs.
    # XLA cache hits on iteration ≥ 2 (same shapes, new values).
    # Metal-only threading: the insulating routes stay on the None branch
    # (bit-exact bool selectors).  The ENTRY-solved state feeds chi, the
    # head and Sigma — one mu per map call, from this call's spectrum.
    metal_occ_state = (entry_occ_state
                       if _material_class(inputs) == "metal" else None)
    W_by_role = compute_screening_model(
        inputs.config.compute_mode, wfns_qp, inputs.V_q,
        quad=inputs.quad, e_ref=inputs.e_ref, sym=inputs.sym,
        centroid_indices=inputs.centroid_indices, config=inputs.config,
        meta=inputs.meta, mesh_xy=inputs.mesh_xy,
        run_dir=os.path.join(inputs.input_dir, "tmp", "mpa"),
        label=f"sc_{state.iteration:04d}",
        head_resolver=inputs.head_resolver,
        head_channel=getattr(inputs, 'head_channel', None),
        wfn=inputs.wfn,
        mpa_plan=mpa_plan,
        iteration_head_response=iteration_head_response,
        occupation_state=metal_occ_state,
        print_fn=inputs.print_fn)

    # MPA owns a shared complex-frequency model rather than the finite role
    # table above.  Its fit mapping is not a body-W role mapping; without
    # ordinary requests finalize the direct head only.
    if iteration_head_response is not None and mpa_mode:
        iteration_head = W_by_role.get("iteration_head")
        if iteration_head is None:
            raise RuntimeError(
                "MPA screening did not return its sampled QSGW head")
        if bool(inputs.config.do_G0):
            iteration_static_head_terms = compute_static_head_terms_from_sample(
                iteration_head.at(0.0 + 0.0j),
                occ=np.asarray(head_occ_kn[:, :inputs.meta.nb_sigma]),
                cell_volume=float(inputs.meta.cell_volume),
                nk_tot=int(inputs.meta.nk_tot),
            )
    elif iteration_head_response is not None:
        head_body_by_role = W_by_role if requests else None
        iteration_head = finalize_iteration_head_samples(
            iteration_head_response,
            wfn=inputs.wfn,
            meta=inputs.meta,
            config=inputs.config,
            mesh=inputs.mesh_xy,
            requests=requests,
            W_by_role=head_body_by_role,
        )
        if head_body_by_role:
            inputs.print_fn(
                "    SC head: folded q-linear wings through screened "
                "W_body(Gamma) before the mini-BZ average")
        if bool(inputs.config.do_G0):
            iteration_static_head_terms = compute_static_head_terms_from_sample(
                iteration_head.at(0.0 + 0.0j),
                occ=np.asarray(head_occ_kn[:, :inputs.meta.nb_sigma]),
                cell_volume=float(inputs.meta.cell_volume),
                nk_tot=int(inputs.meta.nk_tot),
            )

    # Under mpa_material_class = metal the finite-q body above went through
    # build_mpa_fit(occupation_state=...) — fractional contour lines and the
    # divided-difference origin rows.  Insulating decks keep the historical
    # valence/conduction cut (occupation_state=None).

    # Σ_xc dispatch — mode-orthogonal.  ``write_sigma_omega_h5=False``
    # so intermediate SC iterations don't thrash sigma_mnk.h5; the
    # converged tensor is written once after run_self_consistency
    # returns (see ``dump_sigma_omega_h5_final``).
    # Same metal-only threading as the screening step above: Σ_x/SX
    # diag(f), Σ_c branch weights, and the metal E_F reference.
    sigma_result = compute_sigma_xc(
        inputs.config.compute_mode,
        occupation_state=metal_occ_state,
        wfns=wfns_qp, V_q=inputs.V_q, W_by_role=W_by_role,
        # FULL-BZ E, for the same reason as hartree_basis_rotation above:
        # every operand compute_sigma_xc sees is on the full BZ.
        e_qp_ev=np.asarray(E_full) * RYD_TO_EV,
        static_head_terms=iteration_static_head_terms,
        head_resolver=inputs.head_resolver,
        quad=inputs.quad,
        config=inputs.config, meta=inputs.meta, mesh_xy=inputs.mesh_xy,
        sym=inputs.sym, wfn=inputs.wfn,
        band_slices=inputs.band_slices,
        input_dir=inputs.input_dir,
        # The stored/gspace V_H lives in the DFT basis; this is the U that
        # takes it into the basis ``wfns_qp`` is expressed in.
        # FULL-BZ U: everything compute_sigma_xc touches -- wfns_qp, the
        # resolved external V_H, every Sigma channel -- is on the full BZ.
        # The IBZ U_qp would mismatch resolve_external_hartree's k axis
        # (measured: einsum 'k' 10 vs 16).  Selection to the IBZ happens
        # once, below, after this returns.
        hartree_basis_rotation=U_full,
        omit_v_h=v_h_dft_new is not None,
        iteration_head=iteration_head,
        write_sigma_omega_h5=False,
        print_fn=inputs.print_fn,
    )
    _check_sigma_stage(sigma_result, print_fn=inputs.print_fn)

    # Rotate (V_H + Σ_xc) back to DFT basis and form the *full* QSGW H
    # (as if every band were protected); the partition step below masks
    # off non-protected off-diagonals and overrides out-of-range
    # diagonals with the per-iteration scissor.
    # Σ_xc is genuinely built in the QP basis (from ``wfns_qp``) and must
    # be rotated back.  V_H is not: under density-SC it arrives already in
    # the DFT basis and adds to the pristine ``kin_ion_dft`` with no
    # rotation, which is the whole reason it is contracted with ψ_dft.
    # ``sigma_result.v_h_kij_ry`` is zero in that case (``omit_v_h``).
    delta_h_qp = sigma_result.v_h_kij_ry + sigma_result.sigma_xc_kij_ry
    if not ks.is_identity:
        # Σ ARRIVES ON THE FULL BZ AND IS SELECTED HERE.  Selection is a
        # row take, not a symmetry operation -- these ARE the IBZ k.  The
        # star spread is the free check that the two k-sets agree, and the
        # only one that catches a gauge mismatch upstream; hermiticity,
        # the norm and the electron count all survive one.
        # ``spread_rel`` does both reductions (residual and scale) in one
        # compiled module and brings back 16 bytes, where the two-call form
        # read this (nk, nb, nb) array back twice to print one line.  Its
        # scalar read still synchronises, but the iteration synchronises
        # anyway in ``_run_linear_mixing`` / ``_run_rcrop``.
        # ENFORCED, not printed -- see ``_check_kstar_spread``.
        _check_kstar_spread(ks, delta_h_qp, print_fn=inputs.print_fn)
        delta_h_qp = ks.select(delta_h_qp)
    delta_h_dft = _rotate_to_dft_basis(delta_h_qp, U_qp, mesh=inputs.mesh_xy)
    if v_h_dft_new is not None:
        delta_h_dft = delta_h_dft + v_h_dft_new
    H_qp_dft_full = inputs.kin_ion_dft + delta_h_dft
    if state.iteration == 0:
        _residency_census(
            (("kin_ion_dft", inputs.kin_ion_dft),
             ("H_qp_dft (carry in)", state.H_qp_dft),
             ("U_qp", U_qp),
             ("U_full", U_full),
             ("sigma_xc_kij_ry", sigma_result.sigma_xc_kij_ry),
             ("sigma_x_kij_ry", sigma_result.sigma_x_kij_ry),
             ("v_h_kij_ry", sigma_result.v_h_kij_ry),
             # THE ω-CUBE.  (nω, nk, nb, nb) and the largest object the
             # loop carries; it is the one whose retention across
             # iterations ``residual_fn`` now drops, so its measured
             # per-rank size is the size of that saving.
             ("sigma_c_omega_kij_ry", sigma_result.sigma_c_omega_kij_ry),
             ("delta_h_qp", delta_h_qp),
             ("delta_h_dft", delta_h_dft),
             ("H_qp_dft_full", H_qp_dft_full)),
            inputs.print_fn)
    # The scissor and partition operands are indexed by k and were built
    # on the full BZ; take their IBZ rows so every operand of the H
    # assembly is on one k-set.  The masks are band-only, so selecting
    # rows cannot change what they mean.
    e_dft_act = inputs.e_dft_active_kn_ry
    val_mask = inputs.valence_mask_active_kn
    if not ks.is_identity:
        e_dft_act = ks.select(e_dft_act)
        val_mask = ks.select(val_mask)
    # ``ks``, NOT A WEIGHT ARRAY.  The scissor refit is a REDUCTION over k
    # and the only one in the carry, so it needs star multiplicities, not
    # just the right k-set (§7 of the scaffold labels operands by k-set;
    # that rule is incomplete).  Handing the callee the SAME map that did
    # the ``select`` three lines up is what makes the weights impossible
    # to get out of step with the rows.
    scissor_E_qp_kn_ry, scissor_fit = _scissor_E_qp_for_outofrange(
        H_qp_dft_full, e_dft_act, val_mask,
        inputs.partition.in_range_mask, ks,
        band_classes=scissor_classes,
        print_fn=inputs.print_fn,
    )
    H_qp_dft_new = apply_band_partition(
        H_qp_dft_full,
        protected_mask=inputs.partition.protected_mask,
        in_range_mask=inputs.partition.in_range_mask,
        scissor_E_qp_kn=scissor_E_qp_kn_ry,
    )

    # The occupation state was ENTRY-solved from this call's own spectrum
    # and consumed by this call's chi/head/Sigma; no second solve happens
    # here.  The carry below is DIAGNOSTIC continuity only (mu drift
    # between consecutive map inputs) — the next call re-solves at its
    # own entry, whatever trajectory (linear, rCROP trial/accept) handed
    # it its H.
    if entry_occ_state is not None:
        drift = (abs(entry_occ_state.mu_ry - state.occupation_state.mu_ry)
                 if state.occupation_state is not None else float("nan"))
        inputs.print_fn(
            "    SC occupations: entry-solved BGW-MP1 state, "
            f"mu={entry_occ_state.mu_ry * RYD_TO_EV:.8f} eV, "
            f"|dmu|={drift * RYD_TO_EV:.3e} eV vs previous map input, "
            f"width={inputs.config.occ_broadening_ry:.10f} Ry "
            f"(degauss {2.0 * inputs.config.occ_broadening_ry:.10f} Ry), "
            f"occ_hash {entry_occ_state.occ_hash}")

    # Keep at most one complete MPA screening model on disk.  The current
    # map has now built its replacement, consumed it through Sigma, passed
    # the Sigma gates and assembled H.  Only at this point is the preceding
    # map's pair safe to unlink.  The newest pair survives convergence for
    # restart/debugging; ordinary screening modes never enter this branch.
    if inputs.config.compute_mode is ComputeMode.MPA:
        from .mpa.model import retain_iteration_artifacts
        retain_iteration_artifacts(
            os.path.join(inputs.input_dir, "tmp", "mpa"),
            f"sc_{state.iteration:04d}",
            print_fn=inputs.print_fn,
        )
        # AFTER this iteration's complete store cycle (allocate → fit →
        # finalize → head → Σ → retain), which is the point at which the
        # per-iteration churn audit A1 measures has actually happened.
        # Cheap: one /proc/self/maps read and a dict walk.  It reports the
        # mapped-libhdf5 count and, per file, how many times each HDF5
        # library instance opened it — the number that grows with
        # iteration count and that A1 predicts is the segfault's driver.
        from file_io.hdf5_owner import probe as _hdf5_probe
        _hdf5_probe(f"sc_{state.iteration:04d}", print_fn=inputs.print_fn)

    return SCState(
        H_qp_dft=H_qp_dft_new,
        iteration=state.iteration + 1,
        occupation_state=entry_occ_state,
        head_surface_weight_kn=entry_surface_weight_kn,
        outputs=SCOutputs(
            sigma_result=sigma_result,
            # FULL-BZ U.  These two fields are consumed TOGETHER by
            # run_sc_driver's final rotate-back, so they must share a
            # k-set, and ``sigma_result``'s is the full BZ by
            # construction -- compute_sigma_xc runs there.  Storing the
            # IBZ U_qp here is the mismatch that raised 'k' 10 vs 16.
            sigma_basis_U=U_full,
            scissor_fit=scissor_fit,
            tail_scissor_fit=tail_fit,
        ),
    )


# ---------------------------------------------------------------------------
# Per-iteration scissor refit for non-protected out-of-range bands
# ---------------------------------------------------------------------------

def _scissor_E_qp_for_outofrange(
    H_qp_dft_full: jax.Array,
    e_dft_kn_ry: jax.Array,
    valence_mask_kn: jax.Array,
    in_range_mask: jax.Array,
    kstar,
    band_classes=None,
    print_fn=None,
) -> tuple[jax.Array, object | None]:
    """Return ``E_QP_scissor[k, n]`` for use as the diagonal of bands
    that are out of the ω-grid range.

    Mechanism: take the diagonal of ``H_qp_dft_full`` (the candidate
    QP energies if the iteration kept all off-diagonals), restrict to
    in-range bands as the scissor's reference set, fit α/β per
    val/cond, then evaluate ``E_QP = α·E_DFT + β`` for every (k, n).
    The masking primitive will use this only at out-of-range entries.

    Short-circuits to ``E_DFT`` (no correction) when every band is
    in-range — the all-protected default — so the per-iteration cost
    is one ``np.diagonal`` call.

    ``kstar`` IS REQUIRED AND IS THE MAP THAT PRODUCED THESE ROWS.  The
    fit is a least squares over every (k, n) sample, so it is a reduction
    over k and its answer depends on how often each k appears.  With the
    loop on the IBZ each star appears once but stands for
    ``multiplicity`` full-BZ points; fitting those rows unweighted gave a
    different α/β, a different scissor diagonal on the 98/128 out-of-range
    bands of mos2_4x4, and eqp0 differing from the full-BZ arm by 0.386 eV
    max / 0.037 eV rms at 3 iterations (job 7889373) while Σ itself was
    bit-identical (job 7889375).  Taking the map instead of a weight array
    means the weights cannot be omitted, and cannot be built from a
    different k-set than the rows.  On an identity map
    ``k_star_weights`` returns ones and the arithmetic is unchanged.

    ``band_classes`` (a ``scissor.ScissorBandClasses``, or None) OVERRIDES
    ``valence_mask_kn``.  It carries the metal's three-way split — valence
    below the lowest Fermi-crossing band, conduction above the highest,
    crossing bands in neither fit class.  ``valence_mask_kn`` remains the
    insulating path's two-way "occupied at DFT occupation" index mask,
    which on a metal is the bug this argument exists to fix: it filed
    sodium's Fermi-crossing Kramers pair as VALENCE, and in the ``[-5,+5]``
    window those two bands were the val fit's ONLY in-window samples
    (n_v = 1024, α = 0.9100), whose extrapolation to the 2s semicore was
    wrong by 17.5 eV and wrong in sign against BerkeleyGW (claim 0212).
    Excluding them empties the class there, and an empty class is the
    identity (``bf57701b``) — a refusal to extrapolate rather than a
    Fermi-surface-anchored line.
    """
    from .scissor import k_star_weights

    e_dft_np = np.asarray(e_dft_kn_ry, dtype=np.float64)
    in_range = np.asarray(in_range_mask, dtype=bool)
    # Fast path: nothing to extrapolate.
    if bool(in_range.all()):
        return e_dft_kn_ry, None

    H_diag_np = np.real(np.asarray(jnp.diagonal(
        H_qp_dft_full, axis1=1, axis2=2)))
    in_range_kn = np.broadcast_to(
        in_range[None, :], e_dft_np.shape).astype(bool)
    # THE THREE-WAY CLASSIFICATION.  ``band_classes`` is None on an
    # insulating deck and the historical two-way index mask is used
    # verbatim — the step mask IS this rule when no band crosses E_F, so
    # that path is byte-identical, not merely equivalent.
    valence_kn = np.asarray(valence_mask_kn, dtype=bool)
    crossing_kn = None
    fit_mask_kn = in_range_kn
    if band_classes is not None:
        valence_kn, crossing_kn = band_classes.masks(e_dft_np.shape)
        fit_mask_kn = in_range_kn & ~crossing_kn
    fit = fit_scissor(
        e_dft_np * RYD_TO_EV,
        H_diag_np * RYD_TO_EV,
        valence_mask_kn=valence_kn,
        fit_mask_kn=fit_mask_kn,
        k_weights=k_star_weights(kstar),
    )
    if print_fn is not None:
        # The two arms' agreement is readable here: ``n`` differs with the
        # k-set, ``w`` must not.
        print_fn(f"    SC scissor: {fit.summary()}")
    # ΔE = (α − 1) · E + β; E_QP = E_DFT + ΔE.  The SAME boundary indices
    # that split the fit split the prediction, so a band cannot be fit as
    # one class and extrapolated as another.
    delta_ev = fit.predict(
        e_dft_np * RYD_TO_EV, valence_kn, crossing_mask=crossing_kn)
    return jnp.asarray((e_dft_np + delta_ev / RYD_TO_EV)), fit


def _write_sc_eqp_snapshot(
    inputs: SCInputs,
    state_out: SCState,
    e_output_kn_ev: np.ndarray,
    *,
    call_index: int,
    role: str,
    rms_ev: float,
    rms2_ev: float,
) -> str | None:
    """Write one small BGW-shaped record of a completed SC map call.

    The QP column is ``eigvalsh(F(H_in))``.  The active-band scissor is fit
    to that same output candidate; the sum-band tail scissor is explicitly
    labelled as the input law that was used to build this map's chi/W/Sigma.
    rCROP trial outputs are useful diagnostics but are not accepted iterates.
    This is not a second implementation of BGW's final ``eqp0`` / ``eqp1``
    equations; those remain solely in :mod:`gw.eqp_bgw`.
    """
    from common.collectives import process_rank
    from .eqp_bgw import write_bgw_eqp

    if process_rank() != 0:
        return None

    irr = np.asarray(inputs.sym.kirr_fullids, dtype=np.int64)
    kpoints = np.asarray(inputs.sym.unfolded_kpts, dtype=np.float64)[irr]
    e_dft_full = np.asarray(
        inputs.e_dft_active_kn_ry, dtype=np.float64) * RYD_TO_EV
    e_dft = e_dft_full[irr]
    e_output = np.asarray(e_output_kn_ev, dtype=np.float64)
    # The loop is either full-BZ or already in the KStarMap's irreducible
    # row order.  ``kirr_fullids`` selects only in the former case.
    if getattr(inputs, "kstar", None) is None:
        e_output = e_output[irr]

    active_scissored = np.flatnonzero(
        ~np.asarray(inputs.partition.in_range_mask, dtype=bool))
    band_offset = int(inputs.band_slices.sigma.start)
    active_labels = ",".join(
        str(band_offset + int(i) + 1) for i in active_scissored)
    active_labels = active_labels or "none"
    tail_start = int(inputs.band_slices.sigma.stop)
    tail_stop = int(inputs.meta.b_id_4_user) - band_offset

    active_fit = state_out.outputs.scissor_fit
    tail_fit = state_out.outputs.tail_scissor_fit
    comments = (
        f"SC map={int(call_index):04d} role={role}; columns are "
        "E_DFT reference and eigvalsh(F(H_in)) map output; rCROP trial "
        "outputs are not accepted iterates",
        f"map_output_RMS_dE_prev_output={float(rms_ev):.9e} eV; "
        f"map_output_RMS_dE_two_calls={float(rms2_ev):.9e} eV; "
        "these are map-call diagnostics, not accepted-iterate residuals",
        f"active_scissored_bands_1based={active_labels}",
        ("active_scissor=none (all active bands lie on the Sigma grid)"
         if active_fit is None else
         "output_candidate_active_scissor: " + active_fit.summary()),
        (f"sum_band_tail=[{tail_start + band_offset + 1},"
         f"{tail_stop + band_offset}] 1-based; input_tail_scissor="
         + ("none (DFT energies)" if tail_fit is None else
            f"E_QP={tail_fit.alpha_c:+.10e}*E_DFT"
            f"{tail_fit.beta_c_ev:+.10e} eV; "
            f"n={tail_fit.n_fit_c}, w={tail_fit.w_fit_c:.0f}, "
            f"rmse={tail_fit.rmse_c_ev:.10e} eV")),
    )
    path = os.path.join(
        inputs.input_dir, f"eqp0_iter{int(call_index):04d}.dat")
    write_bgw_eqp(
        path, kpoints, e_dft, e_output,
        band_offset=band_offset, nspin=1, comments=comments,
    )
    inputs.print_fn(f"  SC map energies: {path}")
    return path


def _clear_sc_eqp_snapshots(input_dir: str, *, print_fn=print) -> None:
    """Remove only managed per-map text snapshots from an earlier run."""
    from .qsgw_utils import remove_managed

    removed = remove_managed(
        input_dir, r"eqp0_iter[0-9]{4}\.dat\Z",
        barrier_tag="sc.eqp_snapshots.clear", print_fn=print_fn)
    if removed:
        print_fn(f"  SC map energies: cleared {len(removed)} stale snapshots")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_self_consistency(
    state_init: SCState,
    inputs: SCInputs,
    *,
    max_iter: int = 1,
    tol_ev: float = 1.0e-4,
    accelerator: str = "rcrop",
    history_depth: int = 5,
    mixing: float = 1.0,
) -> tuple[SCState, list[float]]:
    """Iterate ``gw_iteration_map`` until ``max_iter`` or RMS ΔE < ``tol_ev``.

    The iteration carry holds only ``H_qp_dft``; convergence is judged
    on the **eigenvalues** of consecutive H matrices (recomputed each
    iteration via the same k-sharded eigvalsh kernel as the main map)
    so the carry never gets out of sync with a separately-tracked E.

    Parameters
    ----------
    accelerator
        ``"rcrop"`` (default) — Anderson-style restart-CROP acceleration
        from :mod:`mixing.acceleration`.  Order ``history_depth``.
        Required for QSGW on dense band manifolds: the Jacobian's
        cycle-direction eigenvalue is typically ≲ −3 for systems with
        many bands near the gap (PPM ω-grid stiffness), which means a
        plain fixed-point hits a 2-cycle and even α=0.5 linear damping
        only shrinks the cycle amplitude rather than killing it.
        ``"linear"`` — plain α-mixing with damping ``mixing``.  Useful
        for diagnosis (very small α reaches the fixed point monotonically
        but is slow).
    history_depth
        rCROP history depth (only used when ``accelerator="rcrop"``).
        ``m=5`` is BGW's QSGW default.
    mixing
        Linear damping coefficient when ``accelerator="linear"``.

    Returns
    -------
    state_final
        Last :class:`SCState` produced.
    rms_history
        RMS ΔE_n (eV) at each iteration ≥ 1; empty list when
        ``max_iter == 1`` (one-shot G0W0).
    """
    print_fn = inputs.print_fn
    _clear_sc_eqp_snapshots(inputs.input_dir, print_fn=print_fn)
    _, eigvalsh_kshard = _kshard_eigh_kernels(inputs.mesh_xy)
    # E-history dump dir from config.sc (LORRAX_SC_DUMP_DIR env is a
    # deprecated override, applied at config construction).
    _dump_dir = inputs.config.sc.dump_dir

    # One-shot fast path: no acceleration needed.
    if max_iter == 1:
        state_new = gw_iteration_map(state_init, inputs)
        e_initial_ev = (
            np.asarray(eigvalsh_kshard(state_init.H_qp_dft)) * RYD_TO_EV)
        e_new_ev = (
            np.asarray(eigvalsh_kshard(state_new.H_qp_dft)) * RYD_TO_EV)
        rms = float(np.sqrt(np.mean((e_new_ev - e_initial_ev) ** 2)))
        _write_sc_eqp_snapshot(
            inputs, state_new, e_new_ev,
            call_index=0, role="one_shot", rms_ev=rms,
            rms2_ev=float("nan"),
        )
        return state_new, []

    if accelerator == "rcrop":
        return _run_rcrop(
            state_init, inputs,
            max_iter=max_iter, tol_ev=tol_ev,
            history_depth=history_depth,
            eigvalsh_kshard=eigvalsh_kshard,
            print_fn=print_fn,
            dump_dir=_dump_dir,
        )
    if accelerator == "linear":
        return _run_linear_mixing(
            state_init, inputs,
            max_iter=max_iter, tol_ev=tol_ev, mixing=mixing,
            eigvalsh_kshard=eigvalsh_kshard,
            print_fn=print_fn,
            dump_dir=_dump_dir,
        )
    raise ValueError(
        f"run_self_consistency: unknown accelerator={accelerator!r} "
        f"(expected 'rcrop' or 'linear').")


def _run_linear_mixing(
    state_init: SCState, inputs: SCInputs, *,
    max_iter: int, tol_ev: float, mixing: float,
    eigvalsh_kshard, print_fn, dump_dir,
) -> tuple[SCState, list[float]]:
    """Plain α-mixing fixed point.  Diagnostic / accelerator-control path.

    Converges on the SAME criterion as the rCROP path, and for the same
    reason it is output-vs-input: testing the MIXED iterate against its
    predecessor is exactly the mixing != 1 trap -- at small alpha the
    mixed iterate barely moves whatever F does, so the loop would
    "converge" by damping rather than by solving.
    """
    state = state_init
    rms_history: list[float] = []
    E_prev_ev = np.asarray(eigvalsh_kshard(state.H_qp_dft)) * RYD_TO_EV
    _e_history: list[np.ndarray] = [E_prev_ev.copy()]
    _protected = np.asarray(
        inputs.partition.protected_mask, dtype=bool).reshape(-1)
    _in_range = np.asarray(
        inputs.partition.in_range_mask, dtype=bool).reshape(-1)
    if mixing != 1.0:
        print_fn(f"  SC mixing α = {mixing:.3f} (linear)")

    for it in range(max_iter):
        # DROP ITERATION i-1's SigmaResult BEFORE BUILDING ITERATION i's.
        # See the note in ``_run_rcrop.residual_fn``; the shape is the
        # same here — ``state`` is both the loop carry and the argument
        # to the map, so without this rebind both generations of the
        # ω-cube are live for the whole of ``gw_iteration_map``.  The
        # LAST iteration's is kept: it leaves the loop in ``state_new``.
        state = SCState(
            H_qp_dft=state.H_qp_dft,
            iteration=state.iteration,
            occupation_state=state.occupation_state,
            head_surface_weight_kn=state.head_surface_weight_kn,
        )
        state_new = gw_iteration_map(state, inputs)
        E_candidate_ev = (
            np.asarray(eigvalsh_kshard(state_new.H_qp_dft)) * RYD_TO_EV)
        if mixing != 1.0:
            H_mixed = (
                mixing * state_new.H_qp_dft
                + (1.0 - mixing) * state.H_qp_dft
            )
            state_new = SCState(
                H_qp_dft=H_mixed,
                iteration=state_new.iteration,
                occupation_state=state_new.occupation_state,
                head_surface_weight_kn=state_new.head_surface_weight_kn,
                outputs=state_new.outputs,
            )
        E_new_ev = np.asarray(eigvalsh_kshard(state_new.H_qp_dft)) * RYD_TO_EV
        rms = float(np.sqrt(np.mean((E_new_ev - E_prev_ev) ** 2)))
        rms_history.append(rms)
        _e_history.append(E_new_ev.copy())
        rms2 = (
            float(np.sqrt(np.mean((E_new_ev - _e_history[-3]) ** 2)))
            if len(_e_history) >= 3 else float("nan"))
        print_fn(
            f"  SC iter {state_new.iteration}: "
            f"RMS ΔE_{{k,k-1}} = {rms:.6f} eV, "
            f"ΔE_{{k,k-2}} = {rms2:.6f} eV"
        )
        _write_sc_eqp_snapshot(
            inputs, state_new, E_candidate_ev,
            call_index=it, role="linear", rms_ev=rms, rms2_ev=rms2,
        )
        # SAME CRITERION AS rCROP.  ``E_candidate_ev`` is the UNMIXED map
        # output F(H); ``E_prev_ev`` is that call's input.  This used to
        # break on ``rms < tol_ev`` -- an RMS over ALL active bands
        # including the scissored ones, both the looser test and a
        # different set.
        verdict = protected_band_convergence(
            E_candidate_ev, E_prev_ev, _protected, _in_range, tol_ev)
        print_fn(f"    SC convergence: {verdict.summary()}")
        state = state_new
        E_prev_ev = E_new_ev
        if verdict.converged:
            break

    _maybe_dump_e_history(dump_dir, _e_history, print_fn)
    return state, rms_history


def _run_rcrop(
    state_init: SCState, inputs: SCInputs, *,
    max_iter: int, tol_ev: float, history_depth: int,
    eigvalsh_kshard, print_fn, dump_dir,
) -> tuple[SCState, list[float]]:
    """rCROP (Anderson-style) accelerated fixed point.

    Wraps :func:`mixing.acceleration.rcrop_nojit` around the iteration
    map.  rCROP makes **two** ``gw_iteration_map`` calls per
    rCROP-iteration (one for the trial step, one for the
    real-residual evaluation); ``max_iter`` here is the rCROP iteration
    count, not the underlying pipeline call count.

    Convergence tolerance is converted from per-band RMS ΔE (eV) to a
    L2-norm-of-residual on H (Ry) the rCROP solver expects::

        ‖H_new − H_old‖_2 / √(nk · nb²) ≈ RMS-per-element ≈ RMS ΔE / RYD_TO_EV

    RESIDENCY BUDGET, because it is the number that decides the deck size.
    The solver holds 2·``history_depth`` copies of the carry plus a window
    of 2·(m+1).  With m = 5, complex128, at the production shape nk=144,
    nb=2000 (one copy = 9.22 GB)::

        Xhist + Fhist   2·m·nk·nb²·16 B        92.2 GB  whole solve
        Xw + Fw         2·(m+1)·nk·nb²·16 B   110.6 GB  per iteration

    The history entries keep the carry's own (nk, nb, nb) shape at
    ``qsgw_density.band_rotation_spec`` — bra band on 'x', ket band on 'y',
    k replicated — stacked on a LEADING history axis that is never sharded.
    Per rank that is the above over ``mesh.size``.  ``nk`` is the LOOP's
    k-set, so under ``sc_on_ibz`` it is the IBZ: measured n = 163840
    (nk=10) against 262144 (nk=16) on mos2_4x4, job 7889876.

    The accelerator's only collective is one (m+1, m+1) Gram; the update is
    an elementwise combination over the history axis.  What is NOT free is
    the seam here: ``gw_iteration_map`` needs a REPLICATED carry (it adds a
    replicated ``kin_ion_dft`` and, at iteration 0, reads the carry on the
    host to test exact diagonality), so ``residual_fn`` gathers one
    (nk, nb, nb) per call and reshards the residual back.  Distributing the
    carry itself is a separate change and needs that iteration-0 readback
    (:628) and ``kin_ion``'s replicated load to move first.
    """
    from mixing.acceleration import rcrop_nojit

    H0 = state_init.H_qp_dft
    nk, nb, _ = H0.shape
    n_elem = nk * nb * nb
    mesh = inputs.mesh_xy
    print_fn(
        f"  SC rCROP: history_depth={history_depth}, "
        f"max_iter={max_iter}, tol={tol_ev:.1e} eV/band-RMS")
    # PAD, DO NOT DEGRADE.  ``band_rotation_spec`` puts the two band axes
    # on the two mesh axes, so it needs px | nb and py | nb — the same
    # condition every other user of that spec is under.  What used to be
    # here fell back to an UNSHARDED history when nb did not divide, i.e.
    # to the 92.2 GB-on-one-device wall the residency budget in the
    # docstring exists to describe.  Zero-padding both band axes up to the
    # divisor keeps ONE shape and ONE layout for every nb instead, which
    # is also the difference between one compiled executable and a
    # recompile per ragged band count.
    #
    # PARITY CONTRACT — MEASURED (job 56389339, artifacts under
    # ~/software/pad_artifacts_2026-08-06), and it is not the contract you
    # would assume, nor the one the first pass at this change assumed.
    # Three separate claims, because they have three different answers:
    #
    #   1. THE PAD MODES ARE EXACTLY INERT.  H reaches ``gw_iteration_map``
    #      and ``eigvalsh_kshard`` only at the LOGICAL extent
    #      (``_to_carry`` slices first), so no spurious zero eigenvalue is
    #      ever admitted to the RMS-ΔE history, and the pad zone is
    #      bit-for-bit 0.0 after 12 rCROP iterations — 60, 992 and 3072 pad
    #      elements, on 4 GPUs and on 4- and 16-device CPU meshes.  Checked
    #      at the bottom of this function rather than asserted here.
    #
    #   2. A DIVISIBLE EXTENT IS BYTE-IDENTICAL to the pre-pad code, and
    #      SHARDING THE HISTORY IS ITSELF BIT-EXACT.  Both measured 0.0
    #      difference, bit-identical, in every configuration tried.  That
    #      second one matters: it means the pad is the ONLY thing in this
    #      change that moves a number, and it removes the excuse that the
    #      drift below hides under a pre-existing sharded/unsharded floor.
    #      There is no such floor here — that comparison is exact.
    #
    #   3. THE PAD IS NOT BIT-EXACT, AND THE DRIFT IS NOT A FIXED FEW-eps
    #      GAUGE.  rCROP's two primitives are full-array REDUCTIONS (the
    #      (m+1, m+1) Gram, the residual 2-norm), so the extra zero terms
    #      change how XLA GROUPS the nonzero ones.  That seeds a
    #      reduction-order error which rCROP then amplifies, and the seed
    #      grows with the reduction length: after ONE iteration it is 0.2
    #      eps at nk·nb² = 243 and 39.9 eps at nk·nb² = 29768.  How far it
    #      then grows is a property of the TRAJECTORY, not of the pad — on
    #      a contracting one it stayed ≤ 8.3 eps through 12 iterations; on
    #      a stalled one (residual plateaued, history Gram near-degenerate)
    #      it reached 2.9e5 eps at 12 iterations and 9.2e6 eps at 16.
    #      Quoting a single eps figure for this change is therefore wrong.
    #
    #      WHAT BOUNDS IT IS THE RESIDUAL, NOT eps.  Across both regimes
    #      and every iteration count 1–16, |ΔH| stayed ≤ 6.1e-8 of the
    #      per-element residual norm: the padded and unpadded runs are the
    #      same iterate to within ~1e-8 of how far either still is from its
    #      own fixed point.  So this is a gauge in the sense that matters —
    #      it cannot move a converged answer by more than the convergence
    #      criterion — but it is NOT a 3-eps effect, and a test that pins
    #      this path to a few ULPs will fail at production shapes.
    #
    # An nb that already divides pads by zero rows: ``pad_axis_to`` returns
    # the SAME array, so the production path is byte-identical to before.
    from runtime.padding import pad_axis_to, round_up, spec_divisor

    spec = _band_rotation_spec()
    px, py = (int(mesh.shape[a]) for a in mesh.axis_names)
    # From the SPEC, not from px and py directly.  A band axis the spec
    # replicates needs no pad at all, and ``spec_divisor`` is the single
    # place that mapping lives (``runtime.padding``); re-deriving it from
    # the mesh here is how the loader and the sweep would drift apart.
    # ONE extent for BOTH axes, so the carry stays square — the residual is
    # H_out − H_in and the re-Hermitisation below both need that.
    band_div = _math.lcm(spec_divisor(mesh, spec, 1),
                         spec_divisor(mesh, spec, 2))
    nb_pad = round_up(nb, band_div)

    def _pad_bands(A):
        A, _ = pad_axis_to(A, band_div, axis=1)
        A, _ = pad_axis_to(A, band_div, axis=2)
        return A

    entry_sh = NamedSharding(mesh, spec)
    x0 = jax.device_put(_pad_bands(H0), entry_sh)
    # MEASURED, not derived from the shape and the mesh.  A
    # ``device_put`` that fell back to replicated would print the full
    # size here, and that is the failure mode that would make this a
    # silent no-op.
    local_b = sum(sh.data.nbytes for sh in x0.addressable_shards)
    print_fn(
        f"  SC rCROP residency: carry {tuple(H0.shape)} (nk={nk} on the "
        f"loop's k-set), n={n_elem} logical, mesh {px}x{py}; bands "
        f"{nb}→{nb_pad} (band divisor {band_div}, "
        f"+{100.0 * ((float(nb_pad) / nb) ** 2 - 1.0):.2f}% elements); entry "
        f"{x0.nbytes / 2**20:.2f} MiB global / {local_b / 2**20:.2f} MiB "
        f"addressable here; history 2x{history_depth} entries = "
        f"{2.0 * history_depth * x0.nbytes / 2**30:.4f} GiB global / "
        f"{2.0 * history_depth * local_b / 2**30:.4f} GiB here")

    # THE SEAM, and the only reshard in the loop.  History entries live at
    # ``entry_sh`` at the PADDED band extent; ``gw_iteration_map`` needs the
    # carry REPLICATED at the LOGICAL one.  The band extent is the only
    # thing that crosses this seam — nk never changes, and at nb_pad == nb
    # both directions collapse to exactly the pre-pad spelling.
    def _to_carry(A):
        return _place(A if nb_pad == nb else A[:, :nb, :nb], mesh)

    def _to_entry(A):
        return jax.device_put(_pad_bands(A), entry_sh)

    # Bookkeeping for per-iteration printing + final SCOutputs capture.
    _e_history: list[np.ndarray] = [
        np.asarray(eigvalsh_kshard(H0)) * RYD_TO_EV]
    _last_outputs: list = [None]
    _occ_state: list = [state_init.occupation_state]
    _head_surface_weight: list = [state_init.head_surface_weight_kn]
    _iter_idx = [0]
    rms_history: list[float] = []
    _protected = np.asarray(
        inputs.partition.protected_mask, dtype=bool).reshape(-1)
    _in_range = np.asarray(
        inputs.partition.in_range_mask, dtype=bool).reshape(-1)

    def residual_fn(H_in: jnp.ndarray) -> jnp.ndarray:
        # SHARDED IN, REPLICATED CARRY, SHARDED OUT.  The gather is one
        # (nk, nb, nb) per call and is the price of the map's replicated
        # carry; the residual goes straight back to the entry layout, so the
        # history never holds a replicated copy.
        H = _to_carry(H_in)
        # rCROP's mixing combinations don't preserve Hermitisation
        # exactly (numeric drift); re-Hermitise before feeding the
        # iteration map so eigh stays well-defined.
        H = 0.5 * (H + jnp.conj(jnp.swapaxes(H, -1, -2)))
        E_in = np.asarray(eigvalsh_kshard(H)) * RYD_TO_EV
        # DROP ITERATION i-1's SigmaResult BEFORE BUILDING ITERATION i's.
        # ``gw_iteration_map`` reads ``state.iteration`` and
        # ``state.H_qp_dft`` and nothing else, so the previous result was
        # passed in and held in this cell for the whole call for no
        # reader.  Its ``sigma_c_omega_kij_ry`` is the largest object on
        # the SC path and, at the default ``sigma_omega_layout =
        # "replicated"``, does not shrink with P: 2751 MB/rank at nb=512
        # (``gw_config.py``), so holding two generations was a
        # P-independent doubling of the peak.  Only the LAST one has a
        # consumer -- ``dump_sigma_omega_h5_final`` -- and it survives:
        # this cell is refilled below and the loop exits with it.
        _last_outputs[0] = None
        state_in = SCState(
            H_qp_dft=H,
            iteration=_iter_idx[0],
            occupation_state=_occ_state[0],
            head_surface_weight_kn=_head_surface_weight[0],
        )
        state_out = gw_iteration_map(state_in, inputs)
        _last_outputs[0] = state_out.outputs
        _occ_state[0] = state_out.occupation_state
        _head_surface_weight[0] = state_out.head_surface_weight_kn
        # Track per-call eigenvalue RMS so the user sees progress in the
        # same shape the linear path prints.
        E_new = np.asarray(eigvalsh_kshard(state_out.H_qp_dft)) * RYD_TO_EV
        # THE CRITERION: the fixed-point residual of THIS call, output
        # against that same call's input.  Not the difference between
        # successive accepted iterates -- under rCROP the accepted
        # iterate is a mixed combination of the history, so that
        # difference can be driven small by damping while F still has no
        # fixed point.  ||F(H) - H|| makes no reference to the iteration
        # that produced H, so the accelerator cannot flatter it.
        _verdict = protected_band_convergence(
            E_new, E_in, _protected, _in_range, tol_ev)
        rms = float(np.sqrt(np.mean((E_new - _e_history[-1]) ** 2)))
        rms_history.append(rms)
        _e_history.append(E_new.copy())
        rms2 = (
            float(np.sqrt(np.mean((E_new - _e_history[-3]) ** 2)))
            if len(_e_history) >= 3 else float("nan"))
        print_fn(
            f"  SC rCROP call {len(rms_history)}: "
            f"RMS ΔE_{{k,k-1}} = {rms:.6f} eV, "
            f"ΔE_{{k,k-2}} = {rms2:.6f} eV"
        )
        call_index = _iter_idx[0]
        role = ("initial" if call_index == 0 else
                "trial" if call_index % 2 else "accepted_input_map")
        _write_sc_eqp_snapshot(
            inputs, state_out, E_new,
            call_index=call_index, role=role, rms_ev=rms, rms2_ev=rms2,
        )
        print_fn(f"    SC convergence: {_verdict.summary()}")
        _iter_idx[0] += 1
        # Non-trial calls only: there the INPUT is the accepted iterate
        # (rcrop_nojit's ``f_new = residual_fn(x_new)``), so this is the
        # residual AT the iterate the loop would return.  A trial call's
        # residual is the residual at a probe point -- a diagnostic.
        if role != "trial" and _verdict.converged:
            raise _Converged(
                SCState(H_qp_dft=state_out.H_qp_dft,
                        iteration=_iter_idx[0],
                        outputs=state_out.outputs),
                _verdict)
        return _to_entry(state_out.H_qp_dft - H)

    # rCROP HAS NO STOPPING AUTHORITY.  ``tol=0.0`` below is not a
    # disarmed threshold; it is the statement that this solver's job is
    # to ACCELERATE and the caller's is to decide convergence, using the
    # exact L-infinity eigenvalue test.  That test is free: the map
    # already diagonalises H to get QP energies.
    #
    # What used to be here was
    # ``tol_resid = sqrt(n_elem) * tol_ev / RYD_TO_EV``, DELETED rather
    # than repaired.  The ``sqrt(n_elem)`` was the whole defect: the
    # comment above it said it converted a per-band ENERGY tolerance into
    # a per-element RMS on the MATRIX, but for Hermitian H, Weyl gives
    #
    #     |dlambda_i| <= ||dH||_2 <= ||dH||_F = ||f||_2
    #
    # so the only sound conversion is ``tol_resid = tol_ry``, with no
    # sqrt at all -- conservative, possibly stopping later than strictly
    # necessary, never early.  Multiplying by sqrt(n_elem) loosens the
    # bound by exactly the factor that destroys it.
    #
    # THE ARITHMETIC CLOSES EXACTLY.  MEASURED on Si 4x4x4 SYM/SOC at
    # P=4: the carry is (64, 24, 24) -- the loop runs full-BZ under
    # ``sc_on_ibz = false`` -- so n_elem = 36864 and sqrt(n_elem) = 192.
    # At a 2 meV request the threshold became 2.8223e-02 Ry; rCROP
    # stopped at ||f||_2 = 2.3618e-02 Ry reporting converged=True; and
    # max|dE| over the non-scissored bands at that call was 0.120477 eV,
    # 60.2x the cutoff.  The sound threshold, 1.4700e-04 Ry, sits 161x
    # BELOW the residual it stopped at, so it would not have stopped.
    # Weyl slack ||dH||_F / max|dlambda| measured 2.67, and
    # 192 / 2.67 = 72.0x is the predicted looseness at threshold against
    # 60.2x observed at the actual stop.  Nothing is unaccounted for.
    #
    # The deleted comment's care about LOGICAL vs PADDED ``n_elem`` was
    # correct about padding and beside the point: it fine-tuned a factor
    # that should not have existed.  Local precision can disguise a
    # global error, so the factor is gone rather than corrected.

    try:
        result = rcrop_nojit(
            residual_fn,
            # THE CARRY ITSELF, not a flattened copy of it.
            x0,
            m=history_depth,
            maxit=max_iter,
            tol=0.0,   # see above: rCROP does not decide convergence
            print_fn=None,  # we print our own RMS-ΔE history above
            entry_sharding=entry_sh,
        )
    except _Converged as stop:
        # The criterion fired inside the map.  Return the accepted
        # iterate that met it, NOT rCROP's internal x: the two differ by
        # one trial step and only this one carries an SCOutputs for the
        # writers.  The pad-inertness check below reads the solver's
        # final x, which this path never reaches -- say so rather than
        # skip it silently.
        print_fn(
            f"  SC rCROP stopped by the convergence criterion after "
            f"{_iter_idx[0]} map calls: {stop.verdict.summary()}")
        print_fn(
            "  SC rCROP pad inertness: NOT CHECKED (early stop -- the "
            "check reads the solver's final x, not reached on this path)")
        _maybe_dump_e_history(dump_dir, _e_history, print_fn)
        return stop.state, rms_history

    print_fn(
        f"  SC rCROP done: {result.iterations} iterations WITHOUT meeting "
        f"the {tol_ev:.3e} eV criterion (budget exhausted), final "
        f"‖residual‖₂ = {float(result.residual_norms[-1]):.4e} Ry -- a "
        f"DIAGNOSTIC: rCROP has no stopping rule of its own")

    # INERTNESS, CHECKED — a DIFFERENT claim from the parity one at the top
    # of this function, and this pair has been measured to come apart: the
    # pad modes can be bit-for-bit 0.0 while the reduction order still moves
    # the answer by eps-scale.  Neither substitutes for the other, so the
    # cheap one runs in-line.  Two slices and a max, once per solve, and it
    # is a failure signature rather than a success marker — a nonzero here
    # means something wrote into the pad zone, which would make every
    # statement above about the pad wrong.
    if nb_pad != nb:
        pad_max = max(
            float(jnp.max(jnp.abs(result.x[:, nb:, :]))),
            float(jnp.max(jnp.abs(result.x[:, :, nb:]))))
        print_fn(
            f"  SC rCROP pad inertness: {nb_pad - nb} pad bands per axis, "
            f"max|H| over the pad zone = {pad_max:.3e} "
            f"(exactly 0.0: {pad_max == 0.0})")

    # Final state: use the last x from rCROP (Hermitised) and the last
    # captured SCOutputs so the writer downstream has the full
    # frequency-grid Σ_c, head pieces, etc.
    # Back to the REPLICATED carry layout: every consumer of the returned
    # state (``_scissor_E_qp_for_outofrange``, ``final_qp_eigenstates``,
    # the h5 writers) reads it back on the host.
    H_final = _to_carry(result.x)
    H_final = 0.5 * (H_final + jnp.conj(jnp.swapaxes(H_final, -1, -2)))
    state_final = SCState(
        H_qp_dft=H_final,
        iteration=_iter_idx[0],
        occupation_state=_occ_state[0],
        head_surface_weight_kn=_head_surface_weight[0],
        outputs=_last_outputs[0],
    )
    _maybe_dump_e_history(dump_dir, _e_history, print_fn)
    return state_final, rms_history


def _maybe_dump_e_history(
    dump_dir: str | None,
    e_history: list[np.ndarray],
    print_fn,
) -> None:
    if not dump_dir:
        return
    from common.collectives import process_rank
    if process_rank() == 0:
        os.makedirs(dump_dir, exist_ok=True)
        np.save(os.path.join(dump_dir, "e_history_kn_ev.npy"),
                np.stack(e_history, axis=0))
        print_fn(
            f"  SC dump: saved {len(e_history)} eigenvalue snapshots to "
            f"{dump_dir}/e_history_kn_ev.npy (shape (map, k, n))"
        )
    barrier("sc.e_history.write", print_fn=print_fn)


def load_head_velocity_source(
    config,
    input_dir: str,
    *,
    mesh,
    wfn,
    meta,
    print_fn=print,
):
    """Resolve ``sc_head_update`` to the head's velocity source, or None.

    The ONE place the mode string turns into an object.  Both metal modes
    read the artifact ``get_dipole_mtxels --parallel-transport`` writes;
    they differ in how much of it they need:

    ``parallel_transport``
        the whole thing — Berry connection, the exact DFT velocity, and the
        completed velocity-identity validation, all through
        ``load_parallel_transport_head``.

    ``dft_velocity``
        the exact DFT p-matrix velocity stage ONLY, through
        ``load_dft_velocity_head``.  ``load_parallel_transport_head`` is
        not called, not imported, and not reachable on this path — which is
        the point: it currently dies on its first rank-0 scalar read
        (claim 0188 blocker 1) and repairing that belongs to the
        parallel-transport route, not to this one.

    Returns None for ``off``, which preserves the fixed-DFT head exactly.
    """
    from gw.gw_config import METAL_HEAD_UPDATES

    mode = str(config.sc.head_update)
    if mode not in METAL_HEAD_UPDATES:
        return None
    if not bool(config.do_G0):
        raise ValueError(
            f"sc_head_update={mode} requires do_G0=true; "
            "otherwise the rebuilt head has no consumer.")

    from file_io.paths import resolve_input_path

    pt_path = resolve_input_path(
        input_dir, config.paths.parallel_transport_file)
    if mode == "dft_velocity":
        from .qsgw_head import load_dft_velocity_head

        source = load_dft_velocity_head(
            pt_path, mesh=mesh, wfn=wfn, meta=meta)
        print_fn(
            "  SC head: loaded the exact DFT p-matrix velocity stage from "
            f"{pt_path} (nb={source.nb_logical}); no Berry connection, so "
            "the ΔH covariant velocity correction is OFF and the head runs "
            "on DFT velocities rotated into each iteration's QP basis "
            "(claim 0183 parks the covariant upgrade)")
        return source

    from .qsgw_head import load_parallel_transport_head

    source = load_parallel_transport_head(
        pt_path, mesh=mesh, wfn=wfn, meta=meta)
    vgate = source.validation
    print_fn(
        "  SC head: loaded validated parallel transport from "
        f"{pt_path} (nb={source.nb_logical}, "
        f"velocity max_abs={vgate['max_abs']:.3e}, "
        f"max_rel={vgate['max_rel']:.3e})")
    return source


def run_sc_driver(
    wfns,
    V_q: jax.Array,
    kin_ion: jax.Array,
    *,
    head_channel=None,
    quad,
    e_ref: float,
    static_head_terms,
    head_resolver,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    centroid_indices,
    band_slices: BandSlices,
    input_dir: str,
    enk_dft,
    print_fn: Callable = print,
) -> tuple[SigmaResult, jax.Array, list[float], bool]:
    """Self-consistent QSGW, driver-facing: DFT inputs in, DFT-basis Σ out.

    Wraps the whole SC machinery — band partition (protected / in-range /
    scissored, from the ω-grid window), :class:`SCInputs` assembly,
    :func:`run_self_consistency`, the post-SC artifact dumps (WFN_qp.h5 /
    qp_wfn_rotations.h5 / converged sigma_mnk.h5) — and returns exactly
    what the driver's post-Σ seam consumes:

    Returns
    -------
    sigma_result : SigmaResult
        The LAST iteration's Σ, **rotated back to the DFT basis** with
        the basis-of-record U (``state.outputs.sigma_basis_U`` — the
        unitary that defined the basis the last ``compute_sigma_xc``
        call ran in; the converged U is one iteration ahead and agrees
        only at the fixed point — worst case ``max_iter=1``, where the
        correct U is the identity, which is the case
        ``tests/test_invariance_gates.py::test_sc_iteration1_equals_one_shot``
        runs).
        ``sigma_omega_h5_path`` points at the converged single-write
        sigma_mnk.h5; ``efermi_dft_ev`` is filled for every mode — the
        dynamic finalize's OWN omega reference (the one its grid and its
        interpolation used, and the one stamped into that file) wherever
        there is one, ``wfn.efermi`` only for the static modes that never
        set it.
    sigma_total : (nk, nb, nb) Ry
        Σ_xc + V_H in the DFT basis — the eigh operand
        ``H_QP = kin_ion + sigma_total``.
    rms_history : list[float]
        RMS ΔE (eV) per iteration.
    rotations_written : bool
        True when this call wrote ``qp_wfn_rotations.h5`` (the
        ``config.debug.write_wfn_h5`` artifact dump ran).  The driver's
        generic writer reads this fact instead of re-deriving the
        predicate, so it cannot overwrite the authoritative SC file.
    """
    import dataclasses

    from .band_partition import BandPartition
    from .scissor import classify_bands_in_grid

    # THE b0 == 0 ASSUMPTION, MADE EXPLICIT.  Every occupancy in this
    # module indexes the ACTIVE window with a GLOBAL band count:
    # ``val_mask_active`` below, ``n_occ`` in ``gw_iteration_map``, the
    # ``E[:, :n_occ]`` midgap in ``_diagonalize_and_get_efermi``, and the
    # ``fermi_level_step`` target in ``rebuild_hartree_dft_basis``.  All
    # four are ``meta.nelec``, which counts from band 0, while the window
    # starts at ``b0``.  They coincide only at ``b0 == 0``.  ``Meta`` fixes
    # b0=0 today; importantly, ``nval`` moves b1 and does NOT move b0.  If a
    # future caller supplies a truncated active window, these masks would
    # silently mark the wrong bands occupied and the density-SC rebuild
    # would omit the bands below b0 from ρ — an O(400 Ry) V_H error with no
    # local symptom, since ``rho_from_wfns`` checks only the electron count
    # it was handed.  Refuse instead of computing it.
    b0_sigma, b3_sigma = band_slices.sigma_range
    if int(b0_sigma) != 0:
        raise NotImplementedError(
            f"run_sc_driver: the SC active window starts at b0={int(b0_sigma)} "
            f"(sigma_range={(int(b0_sigma), int(b3_sigma))}), but every "
            f"occupancy here is meta.nelec={int(meta.nelec)} indexed into the "
            f"window, i.e. counted from band 0.  Self-consistency on a deck "
            f"with b0 != 0 needs the occupancies re-expressed relative to b0 "
            f"and the density rebuild extended to the bands below b0; neither "
            f"is implemented.  Restore an active window beginning at band 0, "
            f"or use qp_solver = one_shot_dft.  Changing nval cannot fix "
            f"this: nval moves b1, not b0.")

    e_dft_active_kn_ry = jnp.asarray(np.asarray(enk_dft, dtype=np.float64))
    nb_active = e_dft_active_kn_ry.shape[1]
    val_mask_active = jnp.broadcast_to(
        jnp.arange(nb_active) < int(meta.nelec),
        e_dft_active_kn_ry.shape)

    # In-range mask: bands whose E_DFT lies inside [σ_ω_min, σ_ω_max]
    # at *every* k.  Bands outside the ω-grid get the per-iteration
    # scissor (otherwise their Σ_c is clamped at the grid edge → the
    # QSGW H-build feeds garbage diagonals that explode the iteration).
    # ONE ω reference: the window must be anchored where the Σ grid is.
    # Metallic decks build the grid against the fixed-N μ; judging the
    # window against wfn.efermi (midgap, +2.79 eV on sodium) emptied the
    # partition — 0/48 in range — so every band was scissored by a fit
    # with zero samples and H_qp came back all-zero.
    if getattr(config, "occ_smearing_family", None):
        from psp.get_DFT_mtxels import spin_degeneracy_factor
        from .efermi import solve_mp1_occupations
        _e_part = np.asarray(enk_dft, dtype=np.float64)
        # enk_dft here is the unfolded full-BZ table: uniform weights,
        # the same convention as _solve_head_occupations.
        _mu_ry, _ = solve_mp1_occupations(
            _e_part,
            np.full(_e_part.shape[0], 1.0 / _e_part.shape[0]),
            float(wfn.num_electrons),
            float(config.occ_broadening_ry),
            state_capacity=float(spin_degeneracy_factor(wfn)),
            clamp_tol=float(config.occupation_clamp_tol))
        efermi_ev = float(_mu_ry) * RYD_TO_EV
    else:
        efermi_ev = float(wfn.efermi) * RYD_TO_EV
    omega_min_ev = float(config.sigma.omega_min_ev) + efermi_ev
    omega_max_ev = float(config.sigma.omega_max_ev) + efermi_ev
    e_dft_ev = np.asarray(enk_dft, dtype=np.float64) * RYD_TO_EV
    band_in_grid, _ = classify_bands_in_grid(
        e_dft_ev, omega_min_ev, omega_max_ev)
    in_range = jnp.asarray(band_in_grid, dtype=bool)
    # Default protected = in-range: these bands carry full off-diag Σ.
    # Out-of-range bands take the scissor, no off-diag mixing.
    print_fn(
        f"  SC partition: protected/in-range = {int(band_in_grid.sum())}"
        f"/{int(band_in_grid.size)} bands"
    )
    partition = BandPartition(
        protected_mask=in_range, in_range_mask=in_range)
    partition.warn_if_protected_outside_grid(print_fn=print_fn)

    # THE k-STAR MAP.  Built UNCONDITIONALLY, because it has two
    # independent jobs and only the first is optional:
    #
    #  1. ``config.sc_on_ibz`` -- run the LOOP reduced.  Opt-in; absent,
    #     H / E / U and the carried state stay on the full BZ exactly as
    #     before.  Σ is built on the full BZ either way (it comes from an
    #     FFT over the k-grid; decisions.md 2026-08-04, TRS veto scope).
    #  2. the post-SC writers -- ``dump_qp_wfn_artifacts`` puts each one
    #     on ITS OWN k-set, and ``write_qp_wfn_h5`` wants the WFN file's
    #     IBZ whatever the loop did (qp_wfn.py:136).  On a deck whose WFN
    #     stores a reduced k-set that is a REDUCTION, not a broadcast, so
    #     the map is needed with ``sc_on_ibz`` off as well.  Omitting it
    #     there is the crash "U shape (16,128,128) inconsistent with
    #     (nk=10, nb_active=128)".
    #
    # Construction is two numpy index arrays plus a ``np.unique``.
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import KStarMap
    kstar_io = KStarMap.from_sym(sym, int(wfn.ntran))
    kstar = None
    if bool(getattr(config, "sc_on_ibz", False)):
        if kstar_io.is_identity:
            print_fn("  SC: sc_on_ibz requested but every k-star is a "
                     "singleton on this deck; running on the full BZ.")
        else:
            kstar = kstar_io
            print_fn(f"  SC: {kstar!r} — H/E/U on the IBZ, Σ on the full BZ")
            # Device gather — ``np.asarray`` here would raise "spans
            # non-addressable devices" the moment ``kin_ion`` arrives
            # sharded, and it is the same (nk, nb, nb) object as U.
            kin_ion = kstar.select(kin_ion)

    parallel_transport = load_head_velocity_source(
        config, input_dir, mesh=mesh_xy, wfn=wfn, meta=meta,
        print_fn=print_fn)

    inputs = SCInputs(
        wfns_dft=wfns, V_q=V_q, kin_ion_dft=kin_ion,
        head_channel=head_channel,
        quad=quad, e_ref=e_ref,
        static_head_terms=static_head_terms,
        head_resolver=head_resolver,
        config=config, meta=meta, mesh_xy=mesh_xy,
        sym=sym, wfn=wfn, centroid_indices=centroid_indices,
        band_slices=band_slices, input_dir=input_dir,
        partition=partition,
        e_dft_active_kn_ry=e_dft_active_kn_ry,
        valence_mask_active_kn=val_mask_active,
        kstar=kstar,
        parallel_transport=parallel_transport,
        print_fn=print_fn,
    )
    state_init = make_initial_state_from_dft(inputs)
    # Loop knobs from ``config.sc`` (the LORRAX_SC_* env vars are
    # deprecated overrides, applied at config construction).
    sc = config.sc
    print_fn(f"  SC: mode={config.compute_mode.value}, max_iter={sc.max_iter}, "
             f"tol={sc.tol_ev:.1e} eV, accel={sc.accelerator}"
             + (f", depth={sc.history_depth}" if sc.accelerator == "rcrop"
                else f", α={sc.mixing:.2f}"))
    state_final, rms_history = run_self_consistency(
        state_init, inputs,
        max_iter=sc.max_iter, tol_ev=sc.tol_ev,
        accelerator=sc.accelerator,
        history_depth=sc.history_depth,
        mixing=sc.mixing,
    )
    sigma_result = state_final.outputs.sigma_result
    print_fn(
        f"  SC done: {len(rms_history)} GW map calls"
        + (f", final RMS ΔE = {rms_history[-1]:.4e} eV"
            if rms_history else " (one-shot)"))

    # Post-SC dumps: WFN_qp.h5 (drop-in BSE / restart input),
    # qp_wfn_rotations.h5 ((U, E_qp) companion), and the converged
    # sigma_mnk.h5 (intermediate iterations skipped the H5 write, so
    # this is the single end-of-run write).  WFN_qp.h5 uses the eigh of
    # ``state_final.H_qp_dft`` — the converged DFT-basis H — so its
    # eigenvalues + U are the *true* QP eigenstates of the SC fixed
    # point (the driver's post-Σ-seam eigh differs slightly because the
    # SC carry applies the band partition).
    rotations_written = False
    if config.debug.write_wfn_h5:
        dump_qp_wfn_artifacts(
            state_final, n_occ=int(meta.nelec), mesh_xy=mesh_xy,
            kstar=kstar_io, state_on_ibz=kstar is not None,
            wfn=wfn, sym=sym, band_slices=band_slices, kgrid=meta.kgrid,
            logical_band_stop=int(meta.b_id_4_user),
            output_dir=input_dir, print_fn=print_fn,
            clamp_tol=float(config.occupation_clamp_tol),
        )
        rotations_written = True
    sigma_omega_h5_path = dump_sigma_omega_h5_final(
        state_final, config=config, meta=meta, mesh_xy=mesh_xy,
        input_dir=input_dir, sym=sym, print_fn=print_fn,
    )

    # Rotate every QP-basis SigmaResult field back to the DFT basis.
    # The Σ matrices live in the basis of the wfn bundle the last
    # ``compute_sigma_xc`` call ran in — the basis DEFINED by
    # ``state.outputs.sigma_basis_U``.  Downstream driver code (H build +
    # eigh, writer, freq_debug) is written for DFT-basis matrices
    # (kin_ion is DFT basis throughout).
    # PLACED ONCE, FOR ALL FIVE ROTATIONS, at ``band_rotation_spec`` —
    # the layout ``_rotate_to_dft_basis`` contracts in and the layout
    # every producer of this array already emits, so on the default SC
    # path this is a no-op ``device_put``.  It is NOT dead: ``jnp.asarray``
    # alone is wrong for the HOST U the k-star broadcast leaves on a
    # reduced k-set — it builds a SINGLE-DEVICE array, which is an
    # operand-sharding error against the mesh-sharded Σ at P>1 rather
    # than a slow success — and plain ``jax.device_put`` of a host array
    # fires the hidden replica ``assert_equal`` all-gather.  ``_place``
    # routes each kind correctly; only the spec changed.
    U = _place(state_final.outputs.sigma_basis_U, mesh_xy,
               _band_rotation_spec())
    sig_h = _rotate_to_dft_basis(sigma_result.v_h_kij_ry, U, mesh=mesh_xy)
    sig_x = _rotate_to_dft_basis(sigma_result.sigma_x_kij_ry, U, mesh=mesh_xy)
    sigma_xc_dft = _rotate_to_dft_basis(
        sigma_result.sigma_xc_kij_ry, U, mesh=mesh_xy)
    sigma_total = sigma_xc_dft + sig_h
    sigma_result_dft = dataclasses.replace(
        sigma_result,
        v_h_kij_ry=sig_h,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_dft,
        sigma_sx_kij_ry=(
            _rotate_to_dft_basis(sigma_result.sigma_sx_kij_ry, U, mesh=mesh_xy)
            if sigma_result.sigma_sx_kij_ry is not None else None),
        sigma_coh_kij_ry=(
            _rotate_to_dft_basis(sigma_result.sigma_coh_kij_ry, U, mesh=mesh_xy)
            if sigma_result.sigma_coh_kij_ry is not None else None),
        sigma_omega_h5_path=sigma_omega_h5_path,
        # ONE omega reference, fifth site, ACTUALLY ON THE WRITER'S PATH.
        # This line used to read ``float(wfn.efermi) * RYD_TO_EV``
        # unconditionally, which OVERWROTE the reference the finalize
        # interpolated and built its grid with — the fixed-N mu on a metal —
        # with the loader's mid-gap ½(VBM+CBM) of the DFT spectrum.  That
        # value is what ``gw_jax`` puts in ``GWResults.efermi_ev`` and what
        # ``gw_output.write_results`` forms ``e_dft_rel_ev`` from, so the
        # run's eqp0/eqp1.dat sampled Sigma_c(omega) 2.932 eV away from the
        # loop's own zero on the sodium deck and clipped at the grid edge —
        # non-rigidly, hence the jagged final band (claim 0202 §2; the
        # mechanism there is attributed to ``final_qp_eigenstates``' printed
        # midgap 4.166867 eV, and the measurement says otherwise: the files
        # reassemble to 3.6e-4 eV at wfn.efermi = 4.440137 eV and to 3.7 eV
        # at the loop's mu = 1.507789 eV, so THIS line is the mechanism and
        # 32564eb7's fix never reached the writer — ``run_sc_driver``
        # discards the efermi ``dump_qp_wfn_artifacts`` returns).
        # Static modes never fill it (only the dynamic finalize does), which
        # is what the unconditional fill was for; keep that and nothing more.
        # Insulating dynamic decks are byte-identical either way — there the
        # finalize's own reference IS ``wfn.efermi``.
        efermi_dft_ev=(sigma_result.efermi_dft_ev
                       if sigma_result.efermi_dft_ev is not None
                       else float(wfn.efermi) * RYD_TO_EV),
    )
    return sigma_result_dft, sigma_total, rms_history, rotations_written


def final_qp_eigenstates(
    state: SCState, *, n_occ: int, mesh_xy: Mesh,
    state_capacity: float | None = None,
    clamp_tol: float = _OCCUPATION_CLAMP_TOL_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diagonalise the converged ``state.H_qp_dft`` and return the QP eigenstates.

    ON THE STATE'S OWN k-SET, and no k-set argument.  Placing the result
    on a CONSUMER's k-set belongs to the consumer's call site
    (:func:`dump_qp_wfn_artifacts`), because the consumers do not share
    one: ``write_qp_wfn_h5`` wants the WFN file's IBZ (qp_wfn.py:136)
    while ``write_qp_rotations_h5`` wants the full BZ (its
    ``kpoints_crys`` is ``sym.unfolded_kpts`` in the canonical writer,
    gw_output.py:865-875, and ``rotate_wfn_to_qp.py:159`` indexes
    ``U_mnk`` with a full-BZ index).  A single k-set kwarg here can only
    be right for one of them; it used to broadcast for both, and that is
    what made the WFN writer fail on an IBZ loop
    (ibz_self_consistency_scaffold.md §7 row 3).

    ``efermi_ry`` is k-set independent: every full-BZ k shares its star's
    eigenvalues, so the midgap over the IBZ and over the full BZ are the
    same number.

    Returned arrays are host-side numpy (not jax.Array) since the
    typical consumers (WFN_qp.h5 writer, eqp.dat tooling) operate on
    NumPy.  Use this once after :func:`run_self_consistency` to extract
    the (E_qp_ry, U_qp, efermi_ry) needed for downstream rotation +
    serialisation.

    THE ONE PLACE THAT KEEPS THE REPLICATED U, deliberately.  The SC loop
    asks ``_diagonalize_and_get_efermi`` for U at
    ``qsgw_density.band_rotation_spec`` because its consumers are device
    rotations; this function's only consumers are ``np.asarray`` two lines
    below and the two h5py writers behind it, which need the whole
    ``(nk, nb, nb)`` on the host on rank 0 whatever the device layout is.
    Sharding U here would buy nothing and add a gather, and the host read
    would then need the same guard ``gw_iteration_map``'s k-star broadcast
    carries.

    Returns
    -------
    enk_qp_ry : (nk, nb_active) float64
    U_kmn     : (nk, nb_active, nb_active) complex128, ``U[k, m, n] = ⟨DFT_m | QP_n⟩``
    efermi_ry : float, midgap of the converged eigenvalues
    """
    E_ry, U, efermi_ry = _diagonalize_and_get_efermi(
        state.H_qp_dft, n_occ, mesh_xy)
    # ONE omega reference, fifth site (the final writers): the midgap rule
    # is the insulating convention. A metallic run's eqp/sigma writers
    # evaluated Sigma_c at midgap — 2.66 eV above the loop's fixed-N mu on
    # sodium — distorting eqp0/eqp1.dat non-rigidly while the converged
    # iterates were right (claim 0202 §2). The metallic reference is the
    # fixed-N MP1 mu solved on THESE converged eigenvalues, same solver,
    # width and capacity as the loop; uniform weights on the state's own
    # k-set, the _solve_head_occupations convention.
    if (state.occupation_state is not None
            and str(state.occupation_state.smearing_family) == "mp1"):
        if state_capacity is None:
            raise ValueError(
                "final_qp_eigenstates: a metallic occupation_state needs "
                "state_capacity (spin_degeneracy_factor(wfn)) to place the "
                "final mu; got None. The caller has the WFN in scope.")
        from .efermi import solve_mp1_occupations
        _E_np = np.asarray(E_ry, dtype=np.float64)
        _st = state.occupation_state
        _mu_ry, _ = solve_mp1_occupations(
            _E_np,
            np.full(_E_np.shape[0], 1.0 / _E_np.shape[0]),
            float(_st.n_electrons),
            float(_st.smearing_width_ry),
            state_capacity=float(state_capacity),
            # SAME clamp as the loop's solve, for the same reason
            # ``state_capacity`` is a kwarg here: ``SCState`` carries the
            # occupation TABLE, not the solver settings that made it, and
            # two mu's from two differently-parameterised solves is the
            # shadow-accounting failure this module exists to avoid.
            clamp_tol=float(clamp_tol),
        )
        efermi_ry = float(_mu_ry)
    return (
        np.asarray(E_ry, dtype=np.float64),
        np.asarray(U, dtype=np.complex128),
        float(efermi_ry),
    )


def _on_kset(arrays, *, kstar, have_ibz: bool, want_ibz: bool):
    """Move band-index arrays between the IBZ and the full BZ.

    ``kstar.select`` / ``kstar.broadcast`` only.  A hand-rolled gather is
    wrong on any TRS-reduced deck: Θ is antiunitary, so
    ``O(-k) = conj(O(k))`` and not ``O(k)`` (symmetry_maps.py:1740-1751);
    assuming equality is off by 3.6e-01 relative, job 7889235.
    """
    if kstar is None or kstar.is_identity or have_ibz == want_ibz:
        return [np.asarray(a) for a in arrays]
    op = kstar.select if want_ibz else kstar.broadcast
    return [np.asarray(op(np.asarray(a))) for a in arrays]


def dump_sigma_omega_h5_final(
    state: SCState, *,
    config,
    meta,
    mesh_xy: Mesh,
    input_dir: str,
    sym=None,
    print_fn: Callable = print,
) -> str | None:
    """Write the converged ``sigma_mnk.h5`` once after SC convergence.

    Pulls the full ω-grid Σ_c tensor from ``state.outputs.sigma_result``
    (which the iteration map captures from each
    :func:`compute_sigma_xc` call but does NOT write to disk during SC
    iterations — see the ``write_sigma_omega_h5=False`` flag in
    :func:`gw_iteration_map`).  Replaces ~30 redundant per-iteration
    writes with a single end-of-run write.

    Returns the on-disk path (or ``None`` for static modes that didn't
    populate a Σ_c(ω) tensor).
    """
    sigma_result = (
        state.outputs.sigma_result if state.outputs is not None else None)
    if sigma_result is None or sigma_result.sigma_c_omega_kij_ry is None:
        return None
    from .dynamic_sigma import write_sigma_omega
    from .qsgw_utils import write_qsgw_sigma_cube

    # ``sym`` TURNS THE k_irr EXTRACTION ON, and the ordering it needs is
    # already the ordering this function has: Sigma arrives on the full BZ
    # (``H/E/U on the IBZ, Sigma on the full BZ``), the accumulation is
    # complete and the kernel has exited by the time the SC loop reaches
    # convergence, and only then is this called.  The writer measures the
    # star spread on those complete rows before dropping any.
    path = write_sigma_omega(
        sigma_result.sigma_c_omega_kij_ry,
        sig_x=sigma_result.sigma_x_kij_ry,
        sig_h=sigma_result.v_h_kij_ry,
        config=config, input_dir=input_dir,
        meta=meta, mesh_xy=mesh_xy,
        # THE CONVERGED CUBE'S OWN ω REFERENCE, carried on the result the
        # cube came from.  Recomputing it here would be a second opinion
        # about the axis this file writes, and on the metal path the two
        # conventions differ by a measured 2.79 eV (audit A2).
        omega_reference_ev=sigma_result.efermi_dft_ev,
        omega_reference_provenance=sigma_result.omega_reference_provenance,
        sym=sym, print_fn=print_fn,
    )
    print_fn(f"  Σ_c(ω) tensor: {path}")
    # THE QSGW CUBE, WRITTEN WHERE IT IS STILL IN ITS OWN BASIS.  Under
    # self-consistency ``sigma_xc_kij_ry`` is Σ_x + Σ_c^QSGW built from
    # ``wfns_qp``, i.e. the QP basis — the same basis the Σ_c(ω) cube
    # above was written in (``sigma_dispatch.SIGMA_BASIS_FIELDS`` says
    # why that one is never rotated).  ``run_sc_driver`` rotates
    # ``sigma_xc_kij_ry`` back to the DFT basis a few frames below this
    # call, and appending it AFTER that would put one DFT-basis matrix in
    # a file of QP-basis ones with matching shape, dtype and stamp.
    # Nothing downstream would notice; this seam is the reason it cannot
    # happen.  Full BZ either way, so the k_irr extraction that ran on
    # the cubes above runs on this one identically.
    write_qsgw_sigma_cube(
        path, sigma_result.sigma_xc_kij_ry,
        config=config, print_fn=print_fn)
    return path


def dump_qp_wfn_artifacts(
    state: SCState, *,
    n_occ: int,
    mesh_xy: Mesh,
    kstar=None,                          # IBZ <-> full BZ map (KStarMap)
    state_on_ibz: bool = False,          # k-set ``state.H_qp_dft`` is on
    wfn,                                 # WFNReader (source of base coeffs + crystal)
    sym,                                 # SymMaps (full-BZ k-list + kirr_fullids)
    band_slices,
    logical_band_stop: int | None = None,
    kgrid,                               # (nkx, nky, nkz)
    output_dir: str,
    print_fn: Callable = print,
    clamp_tol: float = _OCCUPATION_CLAMP_TOL_DEFAULT,
) -> tuple[str, str, float]:
    """Post-SC artifact dump: WFN_qp.h5 + qp_wfn_rotations.h5.

    Diagonalises the converged ``state.H_qp_dft`` once, then writes:

    * ``WFN_qp.h5`` — full BGW-format wavefunction file with active-block
      ψ rotated by ``U`` and active-block energies replaced by ``E_qp``;
      the logical conduction-sum tail keeps DFT orbitals and receives the
      last map's energy-only scissor; other bands keep their DFT values.
      Drop-in replacement for downstream BSE / restart paths that read
      a WFN.h5.
    * ``qp_wfn_rotations.h5`` — small companion file containing just
      ``(U, E_qp)`` for tools that prefer to apply the rotation
      themselves.

    THE TWO WRITERS ARE ON DIFFERENT k-SETS and neither is the loop's:

    * ``write_qp_wfn_h5`` — the WFN FILE's k-set, ``wfn.nkpts``, checked
      at qp_wfn.py:136.  A WFN file stores the IBZ by BGW convention and
      this writer copies the source file's ``kpoints``/``mtrx``/``tnp``
      through unchanged, so its ``U`` must be the rotation of the stored
      ψ at the stored k.  ``KStarMap.select`` delivers exactly that only
      because the row it takes — the first full-BZ member of each star —
      is the stored k itself, reached by the IDENTITY operation.
      MEASURED on mos2_4x4 (job 7889366): ``kirr_fullids`` =
      [0,1,2,4,5,6,7,8,9,10] is strictly increasing (so ``select``'s row
      order is ``wfn.kpoints`` order), ``sym_idx_k[kirr_fullids]`` is 0
      at all 10 (so no member is a rotated or time-reversed image),
      ``max|unfolded_kpts[kirr_fullids] − wfn.kpoints|`` = 5.6e-17, and
      ``select(broadcast(A)) − A`` = 0 exactly.

      ONE OF THOSE TWO PROPERTIES IS NOW ENFORCED AND THE OTHER STILL IS
      NOT (fix/kirr-fullids-2026-08-08).  ``kirr_fullids`` no longer reads
      the star labels; it matches ``wfn.kpoints`` against the full grid
      directly and raises if a stored k is not on it, so
      ``unfolded_kpts[kirr_fullids] == wfn.kpoints`` holds on every deck
      by construction — and on three of the four in-tree decks it did NOT
      hold before, which is what that change fixed.  The IDENTITY-operation
      property is a separate fact and remains a property of the deck: it
      holds on ``si_cohsex_debug`` and on mos2_4x4, and does not hold on
      the 3x3x1 decks, where the register-don't-touch op-selection policy
      assigns a rotation (on ``cohsex_debug``, a time-reversal row) to some
      wedge rows whose k is nevertheless exactly right.  So this writer's
      "U is the STORED ψ's rotation" claim still needs the probe on a new
      symmetry group; what no longer needs it is the k itself.
    * ``write_qp_rotations_h5`` — the FULL BZ.  Its ``kpoints_crys``
      labels the rows of ``U_mnk``; the canonical writer of this same
      file passes ``sym.unfolded_kpts`` there (gw_output.py:865-875) and
      the consumer indexes ``U_mnk`` by full-BZ index
      (postprocess/rotate_wfn_to_qp.py:159).  In an SC run this is the sole
      owner: the later generic result writer is told not to overwrite it
      from its slightly different post-Sigma eigensolve.

    ``state_on_ibz`` says which k-set the loop ran on (``config.sc_on_ibz``)
    and ``kstar`` is the map; both writers are then reached by
    :func:`_on_kset` from wherever the state is.

    ``logical_band_stop`` is the unpadded end of the sum-band ladder.  It
    is required only when the final map used an energy-only tail scissor.

    Both files are rank-0-only writes (h5py is single-writer); a
    multihost barrier follows so the caller can rely on both files
    existing on every rank when this function returns.

    Returns ``(qp_wfn_path, qp_rotations_path, efermi_ry)``.
    """
    from file_io.qp_wfn import write_qp_rotations_h5, write_qp_wfn_h5

    from psp.get_DFT_mtxels import spin_degeneracy_factor
    enk_loop_ry, U_loop, efermi_ry = final_qp_eigenstates(
        state, n_occ=n_occ, mesh_xy=mesh_xy,
        state_capacity=float(spin_degeneracy_factor(wfn)),
        clamp_tol=float(clamp_tol))
    enk_irr_ry, U_irr = _on_kset(
        (enk_loop_ry, U_loop), kstar=kstar,
        have_ibz=state_on_ibz, want_ibz=True)
    enk_full_ry, U_full = _on_kset(
        (enk_loop_ry, U_loop), kstar=kstar,
        have_ibz=state_on_ibz, want_ibz=False)
    # State the two k-sets rather than letting a mismatch surface as a
    # shape error two frames down (that is how this was found: "U shape
    # (16, 128, 128) inconsistent with (nk=10, nb_active=128)").
    nk_irr, nk_full = int(U_irr.shape[0]), int(U_full.shape[0])
    nk_wfn = int(wfn.nkpts)
    # THE FILE DECIDES WHICH PLACEMENT THE WFN WRITER GETS, not the BGW
    # convention that a WFN stores the IBZ.  ``write_qp_wfn_h5`` copies the
    # source file's kpoints/mtrx/tnp through unchanged, so its U must be the
    # rotation of the stored ψ at the stored k — whichever k-set the file
    # happens to hold.  mos2_4x4's WFN holds the full BZ (9), not its 5-point
    # IBZ, and hard-wiring the IBZ placement refused a run that is fine.
    placements = {nk_irr: (enk_irr_ry, U_irr), nk_full: (enk_full_ry, U_full)}
    if nk_wfn not in placements or nk_full != int(sym.unfolded_kpts.shape[0]):
        raise ValueError(
            f"dump_qp_wfn_artifacts: k-set placement failed — loop nk="
            f"{int(U_loop.shape[0])} (on_ibz={state_on_ibz}) gave "
            f"WFN_qp nk={sorted(placements)} (need wfn.nkpts={nk_wfn}) and "
            f"rotations nk={nk_full} (need full BZ "
            f"{int(sym.unfolded_kpts.shape[0])}); kstar={kstar!r}")
    enk_wfn_ry, U_wfn = placements[nk_wfn]
    print_fn(f"  QP dump k-sets: WFN_qp {nk_wfn} (WFN file, "
             f"{'IBZ' if nk_wfn == nk_irr else 'full BZ'}), "
             f"rotations {nk_full} (full BZ), loop {int(U_loop.shape[0])}")
    qp_wfn_path = os.path.join(output_dir, "WFN_qp.h5")
    qp_rot_path = os.path.join(output_dir, "qp_wfn_rotations.h5")
    if jax.process_index() == 0:
        enk_full_base_ry = None
        tail_fit = (state.outputs.tail_scissor_fit
                    if state.outputs is not None else None)
        if tail_fit is not None:
            if logical_band_stop is None:
                raise ValueError(
                    "dump_qp_wfn_artifacts: logical_band_stop is required "
                    "when the final SC map used a tail scissor.")
            base_ev = np.asarray(
                wfn.energies[0], dtype=np.float64) * RYD_TO_EV
            enk_full_base_ry = apply_conduction_scissor_to_tail(
                base_ev, tail_fit,
                tail_start=int(band_slices.b3),
                logical_stop=int(logical_band_stop),
            ) / RYD_TO_EV
        write_qp_wfn_h5(
            qp_wfn_path, wfn=wfn,
            U_kmn=U_wfn, enk_active_qp_ry=enk_wfn_ry,
            band_start=band_slices.b0, band_stop=band_slices.b3,
            enk_full_base_ry=enk_full_base_ry,
        )
        write_qp_rotations_h5(
            qp_rot_path,
            U_mnk=U_full,
            E_qp_nk=enk_full_ry * 0.5,                     # Ry → Hartree
            band_start=band_slices.b0, band_stop=band_slices.b3,
            kpoints_crys=np.asarray(sym.unfolded_kpts, dtype=np.float64),
            nkx=int(kgrid[0]), nky=int(kgrid[1]), nkz=int(kgrid[2]),
            kpoints_reduced=np.asarray(wfn.kpoints, dtype=np.float64),
            kirr_to_kfull=np.asarray(sym.kirr_fullids, dtype=np.int32),
        )
    barrier("qp_wfn_h5_write")
    print_fn(f"  QP WFN:       {qp_wfn_path}")
    print_fn(f"  QP rotations: {qp_rot_path}")
    _ref_kind = ("fixed-N mu" if (state.occupation_state is not None
                 and str(state.occupation_state.smearing_family) == "mp1")
                 else "midgap")
    print_fn(f"  Final E_F ({_ref_kind}, eV): {efermi_ry * RYD_TO_EV:.6f}")
    return qp_wfn_path, qp_rot_path, efermi_ry


__all__ = [
    "SCInputs",
    "SCOutputs",
    "SCState",
    "gw_iteration_map",
    "load_head_velocity_source",
    "make_initial_state_from_dft",
    "run_self_consistency",
    "run_sc_driver",
    "final_qp_eigenstates",
    "dump_qp_wfn_artifacts",
    "dump_sigma_omega_h5_final",
]
