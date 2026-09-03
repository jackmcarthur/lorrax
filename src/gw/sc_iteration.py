"""Self-consistent QSGW iteration map.

A single ``state → state`` step :func:`gw_iteration_map`, an inner
Python-loop driver :func:`run_self_consistency`, and the two-level outer
orchestrator in :mod:`gw.sc_two_level`.  The
state is :class:`SCState` carrying ``H_qp_dft_mnk`` in the **original
DFT basis** (so the iteration carry has a fixed coordinate system; rcrop
Anderson mixing composes meaningfully).  Every iteration:

1. Diagonalize ``H_qp_dft`` → ``(E_qp, U_qp)`` where
   ``U_qp[k, m, n] = ⟨DFT_m | QP_n⟩``.
2. Rotate the **original** DFT wfn bundle to the new QP basis via
   :func:`wavefunction_bundle.rotate_wavefunctions` (no cumulative
   U-product, no drift).
3. At an outer boundary recompute χ₀ → W and the pole model; within an inner
   solve reuse that screening transaction and rebuild only G → Σ_xc with the
   rotated wfns (:func:`sigma_dispatch.compute_sigma_xc`, mode-orthogonal).
4. Between expensive Sigma rebuilds, converge rotations and energies against
   the fixed Sigma_c(omega) table with :func:`run_fixed_sigma_evsc`; the outer
   screening transaction also caches its immutable W-side time factors.
5. Rotate ``(V_H + Σ_xc)`` back to the DFT basis and form
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
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing
from common.collectives import barrier, device_put_process_local
from common.units import RYD_TO_EV
from .band_partition import (
    BandPartition, apply_band_partition, build_omega_band_partition)
from .efermi import (OCCUPATION_CLAMP_TOL_DEFAULT
                     as _OCCUPATION_CLAMP_TOL_DEFAULT, OccupationState)
from .gw_config import ComputeMode, HeadCorrection
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


@dataclass(frozen=True)
class FixedSigmaEVSCResult:
    """Converged fixed-Sigma eigenvalue-self-consistent QP ladder.

    ``U_dft_to_qp`` uses the driver-wide column convention
    ``U[k,m,n] = <DFT_m|QP_n>``.  It is retained on the two-dimensional
    band mesh; only the much smaller energy ladder is brought to the host.
    """

    energies_ry: np.ndarray
    U_dft_to_qp: jax.Array
    H_qp_dft_ry: jax.Array
    iterations: int
    residual_ev: float
    converged: bool = True


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
    ``efermi_dft_ry`` is the matching immutable DFT reference: the SC map's
    fixed-band-cut midgap on an insulator, the initial fixed-N MP1 chemical
    potential on a metal.
    The low-valence policy compares it with each map candidate's Fermi level
    so the whole out-of-range valence tail preserves ``E_band - E_F``.
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
    # Driver-owned screening seam.  The ordinary driver passes the canonical
    # producer; a certified-fit harness passes its already-resolved reuse
    # provider.  Keeping the callable explicit makes the SC loop use the same
    # source owner as the one-shot stage instead of importing a second path.
    screening_model_fn: Callable
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
    # Canonical DFT Fermi reference for the active scissor.  On an insulator
    # this is the SC map's own fixed-band-cut midgap (not the loader's possible
    # VBM convention); on a metal it is the ONE fixed-N MP1 solve already used
    # to anchor the Sigma window/partition.  Retained here so an SC map never
    # re-solves the old endpoint of its Fermi displacement.
    efermi_dft_ry: float
    material_class: str
    #: IBZ ⇄ full-BZ map for BAND-INDEX quantities.  ``None`` ⇒ the loop
    #: runs entirely on the full BZ, which is what every result before
    #: 2026-08-04 did and what the one-shot equivalence gate pins.  When
    #: given, H / E / U / the carried state live on the IBZ and only Σ is
    #: built on the full BZ — Σ comes from an FFT over the k-grid, which
    #: needs the whole grid (decisions.md, TRS veto scope).
    kstar: object | None = None
    #: Validated, device-resident nearest-neighbour links + exact DFT
    #: velocity.  None preserves the historical fixed-DFT head path exactly.
    parallel_transport: object | None = None
    #: Immutable DFT S/Y/Z response used by ``head_correction=full`` when
    #: ``sc_head_update=off``.  Built once on the shared screening plan, not
    #: rebuilt in every map call; the resident W still supplies the one fold.
    fixed_dft_head_response: object | None = None
    #: One-entry source-resolution cache for certified-fit reuse.  It is
    #: populated at map 0 only after the exact JAX-built entry occupation
    #: state exists; later maps read the path as immutable grid provenance.
    screening_seed_cache: dict | None = None
    #: Frozen χ0/W/pole-model transaction for a two-level inner solve.
    #: ``None`` builds screening from this map's orbitals; otherwise the map
    #: rebuilds G and Sigma from the retained model without entering the
    #: screening producer.  Set only by :mod:`gw.sc_two_level`.
    frozen_screening: SCMapScreeningArtifacts | None = None
    #: Activate the nested fixed-Sigma-table cycle and W(tau) cache.  False
    #: on the max_iter=1 identity path, even though that path also uses this
    #: input bundle.
    two_level_enabled: bool = False
    #: Run-local measured cost ledger.  Host orchestration only, never a jit
    #: operand; shared across the dataclass replacements made by the loops.
    two_level_cost: dict | None = None
    print_fn: Callable = print
    # Selected ladder/iteration/verdict lines go to the driver's production
    # record.  Component chatter remains on ``print_fn`` and can still be
    # sunk when ``debug=false``.
    record_fn: Callable | None = None


def _sc_buffer_mask(inputs: SCInputs) -> np.ndarray:
    """Boolean mask for the symmetric buffer outside the named SC window."""
    n_buffer = int(inputs.config.sc.buffer_nbands)
    nb = int(inputs.band_slices.sigma.stop)
    if n_buffer == 0:
        return np.zeros(nb, dtype=bool)

    band_ids = np.arange(
        int(inputs.band_slices.b0), int(inputs.band_slices.b3),
        dtype=np.int64)
    core_lo = int(inputs.meta.nelec) - int(inputs.config.nval)
    core_hi = int(inputs.meta.nelec) + int(inputs.config.ncond)
    expected_lo = core_lo - n_buffer
    expected_hi = core_hi + n_buffer
    if (int(inputs.band_slices.b1), int(inputs.band_slices.b3)) != (
            expected_lo, expected_hi):
        raise ValueError(
            "SC buffer execution edges disagree with the named window: "
            f"got b1/b3={(inputs.band_slices.b1, inputs.band_slices.b3)}, "
            f"expected {(expected_lo, expected_hi)} from nval/ncond="
            f"{(inputs.config.nval, inputs.config.ncond)} plus "
            f"sc_buffer_nbands={n_buffer}.")
    return ((band_ids >= expected_lo) & (band_ids < core_lo)) | (
        (band_ids >= core_hi) & (band_ids < expected_hi))


def _apply_sc_buffer_partition(
    partition: BandPartition,
    inputs: SCInputs,
) -> BandPartition:
    """Make diagonal/carry buffers diagonal-only while retaining their Sigma."""
    buffer = _sc_buffer_mask(inputs)
    if not buffer.any() or inputs.config.sc.buffer_mode == "one_sided":
        return partition
    in_range = np.asarray(partition.in_range_mask, dtype=bool).reshape(-1)
    outside = np.flatnonzero(buffer & ~in_range)
    if outside.size:
        raise ValueError(
            "SC diagonal buffer left the sampled Sigma(omega) domain at "
            f"local band(s) {outside.tolist()}; endpoint-clamped Sigma is not "
            "a diagonal buffer. Widen sigma_omega_min/max_ev or reduce "
            "sc_buffer_nbands.")
    protected = np.asarray(
        partition.protected_mask, dtype=bool).reshape(-1).copy()
    protected[buffer] = False
    return BandPartition(
        protected_mask=jnp.asarray(protected),
        in_range_mask=partition.in_range_mask)


@jax.jit
def _carry_sc_buffer_diagonal(H_new, H_input, buffer_mask):
    """Replace buffer diagonals by the preceding map input's references."""
    nb = H_new.shape[-1]
    idx = jnp.arange(nb)
    diagonal = jnp.where(
        buffer_mask[None, :],
        jnp.diagonal(H_input, axis1=-2, axis2=-1),
        jnp.diagonal(H_new, axis1=-2, axis2=-1))
    return H_new.at[:, idx, idx].set(diagonal)


@dataclass(frozen=True)
class SCMapScreeningArtifacts:
    """Screening artifacts owned by one completed QSGW map evaluation.

    These values belong to the same map evaluation, and the static
    terms are the exact ones passed to that map's Sigma build.  Under FULL,
    the head and terms are folded through this map's exact static/probe W
    role table.  Under NLF they deliberately remain independent of the W
    body.  Keeping the artifacts together prevents a restart writer from
    pairing a final head with a seed/previous W.  Only the static W is
    retained; probe-frequency W is not restart-owned and must not survive
    the map on the historical single-level path.  The two-level path retains
    the complete role mapping only for the lifetime of its current inner
    solve, then releases it before the next outer producer allocates W.

    ``sigma_model`` is the complete role mapping consumed by
    :func:`gw.sigma_dispatch.compute_sigma_xc`: resident static/probe W for
    PPM or the disk-backed pole-store path for MPA.  Keeping that one mapping
    is what makes a frozen-screening inner loop reuse the exact model rather
    than reconstructing a second representation.  ``static_w`` aliases its
    static role for restart persistence and stays in the producer's
    two-dimensional mesh sharding.  The driver drops both after persistence
    and before the post-SC artifact builds.
    """

    static_w: object | None
    iteration_head: object | None
    static_head_terms: object | None
    sigma_model: object | None = None
    #: Device-resident W-side time factors, scoped to this outer W model.
    w_time_factor_cache: dict | None = None


def _retarget_frozen_iteration_head(
    head,
    *,
    energies_ry,
    occupations,
    efermi_ry: float,
):
    """Attach the current Green-function state to frozen head samples.

    The samples (and therefore their fitted screening poles) belong to the
    outer step and remain immutable.  Their band-energy/occupation fields are
    Green-function inputs used only when evaluating the head self-energy, so
    an inner map replaces those fields with its current QP state.  This is the
    head analogue of rebuilding the body Sigma from a frozen W store.
    """
    if head is None:
        return None
    return replace(
        head,
        sigma_energies_ry=np.asarray(energies_ry, dtype=np.float64),
        sigma_occupations=np.asarray(occupations, dtype=np.float64),
        efermi_ry=float(efermi_ry),
    )


@dataclass(frozen=True)
class SCExactHartree:
    """Caller-owned direct field rebuilt from one SC map's orbitals.

    Every matrix is on the full BZ in the original DFT basis.  Keeping the
    scalar and transverse pieces separate preserves ``sigma_mnk.h5``'s
    decomposition; :attr:`total` is the only combination that enters the
    Hamiltonian.
    """

    scalar_dft: jax.Array
    transverse_dft: jax.Array | None
    efermi_ry: float

    @property
    def total(self):
        if self.transverse_dft is None:
            return self.scalar_dft
        return self.scalar_dft + self.transverse_dft


@_functools.partial(jax.jit, donate_argnums=(0,))
def _add_exact_scalar_hartree(base, scalar):
    """Add a caller-owned scalar direct field into a dead matrix buffer."""
    return base + scalar


@_functools.partial(jax.jit, donate_argnums=(0,))
def _add_exact_four_current_hartree(base, scalar, transverse):
    """Fuse V_H + H_T into ``base`` without materialising their sum."""
    return base + scalar + transverse


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
    screening: SCMapScreeningArtifacts
    exact_hartree_dft: SCExactHartree | None = None
    fixed_sigma_cycle_converged: bool = True


@dataclass(frozen=True)
class SCDriverResult:
    """Final, basis-labelled products returned to :mod:`gw.gw_jax`."""

    sigma_result_dft: SigmaResult
    sigma_total_dft: jax.Array
    rms_history_ev: list[float]
    rotations_written: bool
    static_head_terms_dft: object | None
    head_sigma_diag_dft_w_kn_ry: object | None


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
    # The previous map's structural band decision.  Re-anchoring still uses
    # this map's spectrum and Fermi level; this one-bit memory supplies only
    # the Schmitt deadband at the moving window edge.
    partition: BandPartition | None = None
    # DIAGNOSTIC continuity only (mu-drift log): the map ENTRY-solves its
    # own occupation state from the spectrum of the H it is handed, so
    # correctness never depends on these two fields — F(H) is a self-map
    # of H alone, which is what rCROP's trial/accept trajectory assumes.
    occupation_state: OccupationState | None = None  # full-BZ, padded PT head manifold
    head_surface_weight_kn: jax.Array | None = None  # Nk * tetrahedron delta(E-mu)
    outputs: SCOutputs | None = None
    convergence_verdict: ConvergenceVerdict | None = None


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
        if not np.isclose(
            float(occ_state.mu_ry), float(inputs.efermi_dft_ry),
            rtol=0.0, atol=1.0e-12,
        ):
            raise ValueError(
                "SC DFT Fermi references disagree: the partition/window "
                f"stored {float(inputs.efermi_dft_ry):.12e} Ry but the "
                "canonical initial OccupationState solved "
                f"{float(occ_state.mu_ry):.12e} Ry. The low-valence rigid "
                "shift requires one DFT endpoint; do not choose between "
                "two chemical potentials.")
        inputs.print_fn(
            "  SC head occupations: initialized BGW-MP1 state at "
            f"mu={occ_state.mu_ry * RYD_TO_EV:.8f} eV "
            f"[occ_hash {occ_state.occ_hash}]")
        if inputs.material_class == "metal":
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
    else:
        initial_efermi_ry = float(_midgap_efermi(
            jnp.asarray(enk_dft_ry), int(inputs.meta.nelec)))
        if not np.isclose(
            initial_efermi_ry, float(inputs.efermi_dft_ry),
            rtol=0.0, atol=1.0e-12,
        ):
            raise ValueError(
                "SC DFT midgap references disagree: the initial active "
                f"spectrum gives {initial_efermi_ry:.12e} Ry but SCInputs "
                f"stored {float(inputs.efermi_dft_ry):.12e} Ry.")
    return SCState(
        H_qp_dft=device_put_process_local(H0, rep),
        iteration=0,
        partition=inputs.partition,
        occupation_state=occ_state,
        head_surface_weight_kn=head_surface_weight_kn,
    )


def _solve_occupation_state(
    inputs: SCInputs,
    energies_kn_ry,
) -> OccupationState | None:
    """Solve the canonical per-iteration fixed-N MP1 occupation state.

    The state's ``f_kn`` is full-BZ because the parallel-transport velocity
    and head contraction are full-BZ, padded to the PT storage width; padding
    is explicit and inert (exact zeros move neither the count nor the hash's
    meaning) and the solver sees exactly ``nb_logical`` physical bands.  A
    zero width returns ``None`` so the historical step-occupation path is
    bit-for-bit untouched.

    This is split from :func:`_solve_head_occupations` so the output-candidate
    Fermi used by the low-valence scissor can call the SAME fixed-N solver
    without also building an unused tetrahedron surface table.  There is one
    implementation of the chemical-potential policy and two spectra that
    legitimately need it: the map input (chi/head/Sigma) and F(H)'s candidate
    output (the no-lag valence anchor).
    """
    # ``occ_broadening`` is the head DIAL (0 means the historical insulating
    # path).  A declared MPA metal still needs the fixed-N state when the
    # head is off: chi, Sigma and the Fermi reference consume it.  The
    # physical WIDTH is the single ``config.occ_broadening_ry`` value.
    width_ev = float(inputs.config.screening.occ_broadening_ev)
    pt = getattr(inputs, "parallel_transport", None)
    metal = inputs.material_class == "metal"
    if not metal and (width_ev == 0.0 or pt is None):
        return None

    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    if energies.ndim != 2:
        raise ValueError(
            f"QSGW head occupation energies must be (nk,nb), got {energies.shape}.")
    if pt is None:
        # Head-off MPA metal: the current DFT/QP energy ladder is already the
        # canonical body carrier.  No padding or velocity-storage width
        # exists, so solve every band in that ladder directly.
        nb_logical = int(energies.shape[1])
        nb_storage = nb_logical
    else:
        nb_logical = int(pt.nb_logical)
        # The velocity is the one large dataset BOTH head modes carry; the
        # finite links are absent under sc_head_update = dft_velocity.  They
        # are written with identical shape, so the storage width is unchanged.
        nb_storage = int(pt.velocity_dft_cart.shape[-1])
    if not (0 < nb_logical <= nb_storage <= int(energies.shape[1])):
        raise ValueError(
            "QSGW head occupation manifold must satisfy 0 < logical <= "
            f"storage <= energy bands; got {nb_logical}, {nb_storage}, "
            f"{energies.shape[1]}.")

    from .efermi import assert_fixed_n, solve_mp1_occupations

    nk = int(energies.shape[0])
    kweights = np.full(nk, 1.0 / float(nk), dtype=np.float64)
    # WfnLoader owns this value.  Importing the full pseudopotential CLI only
    # to read the same attribute pulled XML parser dependencies into the SC
    # map and obscured that this is immutable WFN provenance.
    capacity = float(inputs.wfn.occupation_state_capacity)
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
    return occ_state


def _certified_seed_occupation_state(
        inputs: SCInputs, energies_kn_ry, certified_fit,
        solved_state: OccupationState) -> OccupationState:
    """Replay a certified fit's exact occupation provenance on its ladder.

    GPU reductions can choose adjacent float64 roots for the same fixed-N
    problem.  That changes the byte hash even when ``mu`` differs by one ulp.
    A reused fit already stores the authoritative root and occupation hash.
    Re-evaluate MP1 at that stored root on the current entry spectrum, then
    require the reconstructed table to reproduce the stored hash exactly.
    A different spectrum therefore still refuses; only root-reduction noise
    is removed from the cross-run provenance gate.
    """
    if solved_state is None:
        raise ValueError(
            "certified metallic MPA reuse requires an entry occupation state")

    from file_io.mpa_store import (
        assert_occupation_stamps, read_occupation_stamps)
    from .efermi import assert_fixed_n, mp1_occupations

    stamps = read_occupation_stamps(certified_fit)
    if stamps is None:
        raise ValueError(
            "certified MPA fit reuse has no occupation provenance stamps")
    if str(stamps["smearing_family"]) != "mp1":
        raise ValueError(
            "certified metallic MPA fit must carry MP1 occupations; got "
            f"{stamps['smearing_family']!r}")

    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    nb_logical = int(solved_state.f_kn.shape[1])
    if energies.ndim != 2 or int(energies.shape[1]) < nb_logical:
        raise ValueError(
            "certified MPA occupation replay needs the complete logical "
            f"energy ladder ({nb_logical} bands); got {tuple(energies.shape)}")
    occ_kn = mp1_occupations(
        energies[:, :nb_logical], float(stamps["mu_ry"]),
        float(stamps["smearing_width_ry"]),
        clamp_tol=float(inputs.config.occupation_clamp_tol))
    replayed = OccupationState(
        f_kn=occ_kn,
        mu_ry=float(stamps["mu_ry"]),
        smearing_family="mp1",
        smearing_width_ry=float(stamps["smearing_width_ry"]),
        n_electrons=float(stamps["occ_nelec"]),
    )
    assert_occupation_stamps(
        certified_fit, replayed, where="certified MPA fit replay")
    kweights = np.full(
        int(energies.shape[0]), 1.0 / float(energies.shape[0]),
        dtype=np.float64)
    assert_fixed_n(
        replayed, kweights,
        state_capacity=float(inputs.wfn.occupation_state_capacity))
    if replayed.occ_hash != solved_state.occ_hash:
        inputs.print_fn(
            "    SC occupations: replayed certified root "
            f"mu={replayed.mu_ry:.17g} Ry; exact stored occ_hash "
            f"{replayed.occ_hash} (live root hash was "
            f"{solved_state.occ_hash})")
    return replayed


def _solve_head_occupations(
    inputs: SCInputs,
    energies_kn_ry,
) -> tuple[OccupationState | None, jax.Array | None]:
    """Solve the iteration occupation state and its head surface weights."""
    occ_state = _solve_occupation_state(inputs, energies_kn_ry)
    if occ_state is None:
        return None, None

    energies = jnp.asarray(energies_kn_ry, dtype=jnp.float64)
    pt = inputs.parallel_transport
    if pt is None:
        # A head-off MPA metal still needs the occupation state for chi,
        # Sigma and its fixed-N Fermi reference.  It has no velocity carrier
        # and therefore no Drude surface table to construct.
        return occ_state, None
    nb_logical = int(pt.nb_logical)
    nb_storage = int(pt.velocity_dft_cart.shape[-1])
    nk = int(energies.shape[0])
    mu_ry = float(occ_state.mu_ry)
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


def _partitioned_candidate_efermi(
    H_partitioned: jax.Array,
    *,
    inputs: SCInputs,
    kstar,
    n_occ: int,
    use_mp1: bool,
) -> float:
    """Canonical Fermi level of one already-partitioned map candidate.

    Insulators use the map's existing fixed-band-cut midgap convention.
    Metals route the candidate spectrum through the same
    :func:`_solve_occupation_state` fixed-N MP1 implementation used at map
    entry; no second chemical-potential policy is spelled here.
    """
    _, eigvalsh_candidate = _kshard_eigh_kernels(inputs.mesh_xy)
    E_candidate_ry = eigvalsh_candidate(H_partitioned)
    if not use_mp1:
        return float(_midgap_efermi(E_candidate_ry, n_occ))

    E_candidate_full = (
        E_candidate_ry if kstar.is_identity
        else kstar.broadcast(E_candidate_ry))
    with inputs.mesh_xy:
        enk_candidate = jax.lax.with_sharding_constraint(
            jnp.asarray(inputs.wfns_dft.enk).at[
                :, inputs.band_slices.sigma].set(
                    jnp.asarray(
                        E_candidate_full,
                        dtype=inputs.wfns_dft.enk.dtype)),
            NamedSharding(inputs.mesh_xy, P(None, None)))
    candidate_occ_state = _solve_occupation_state(inputs, enk_candidate)
    if candidate_occ_state is None:
        raise RuntimeError(
            "SC low-valence Fermi anchor: the map entry had an MP1 "
            "occupation state but the candidate spectrum did not.")
    return float(candidate_occ_state.mu_ry)


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
    ``density_self_consistent`` — a physics knob whose scalar-QSGW default
    is False —
    so the only eigh that keeps no whole ``(nb, nb)`` tile on one rank
    was unreachable on the default path.  ``config.sc.eigh`` selects it
    now; the E_F rule stays where it was, with ``density_self_consistent``.

    ``"native"`` is still distributed over the k batch, but the staging
    and local eigh now belong to ``distrib_la``'s ``batch_reshard`` route.
    ``"auto"`` picks a one-tile-distributed backend only when both hold:

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

    from common.wfn_layout import band_sphere_spec
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


def _sc_eigh_bands(H: jax.Array, *, kind: str, mesh_xy: Mesh, config):
    """One repeated-eigh door for SC-QSGW and fixed-Sigma evSC.

    ``distrib_la`` owns both routes.  A large tile uses its explicitly
    distributed backend; a fit-size stack uses its ``batch_reshard`` route,
    which stages k over the mesh and runs native tiles concurrently.  Both
    return eigenvectors as columns at ``P(None,'x','y')`` and share the
    logical-band padding/slicing seam in :mod:`gw.qsgw_density`.
    """
    from .qsgw_density import distributed_eigh_bands

    if kind == "distributed":
        backend_config = getattr(config, "backend", None)
        return distributed_eigh_bands(
            H, mesh=mesh_xy, distrib_la_backend="distributed",
            distrib_la_batched_route=getattr(
                backend_config, "distrib_la_batched_route", "batch_reshard"))
    if kind == "native":
        return distributed_eigh_bands(
            H, mesh=mesh_xy, distrib_la_backend="off",
            distrib_la_batched_route="batch_reshard")
    raise ValueError(
        f"_sc_eigh_bands: kind must be 'native' or 'distributed', got "
        f"{kind!r}.")


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


_SIGMA_OMEGA_ROTATE_CACHE: dict[
    tuple[int, tuple[int, ...], bool, bool], Callable] = {}


def _rotate_sigma_omega_cube(
    sigma_c_omega_ry: jax.Array,
    U_dft_to_qp: jax.Array,
    *,
    mesh: Mesh,
    to_qp: bool,
) -> jax.Array:
    """Rotate every frequency row of a correlation-operator cube.

    ``to_qp=True`` computes ``U^dagger Sigma_DFT(omega) U``;
    ``to_qp=False`` computes ``U Sigma_QP(omega) U^dagger``.  The frequency
    axis is scanned instead of joining one giant contraction, so the
    transient rotation holds one ``(nk, nb, nb)`` row.  A band-sharded input
    stays ``P(None,None,'x','y')`` throughout and a replicated input stays
    replicated.  No path gathers a sharded full cube to a device or host.

    This one helper serves both fixed-Sigma eigenvalue iteration and the
    final QP-to-DFT output rotation.  Keeping both directions here pins the
    index convention and the bounded-memory schedule in one place.
    """
    from .qsgw_utils import is_band_sharded_sigma_omega

    shape = tuple(int(v) for v in sigma_c_omega_ry.shape)
    sharded = is_band_sharded_sigma_omega(sigma_c_omega_ry)
    key = (id(mesh), shape, bool(to_qp), bool(sharded))
    fn = _SIGMA_OMEGA_ROTATE_CACHE.get(key)
    if fn is None:
        from .qsgw_density import rotate_band_matrix

        cube_spec = (P(None, None, "x", "y") if sharded
                     else P(None, None, None, None))
        row_spec = (_band_rotation_spec() if sharded
                    else P(None, None, None))
        cube_sh = NamedSharding(mesh, cube_spec)
        row_sh = NamedSharding(mesh, row_spec)
        rotation_sh = NamedSharding(mesh, _band_rotation_spec())

        @jax.jit
        def _kernel(cube, U):
            cube = jax.lax.with_sharding_constraint(cube, cube_sh)
            U = jax.lax.with_sharding_constraint(U, rotation_sh)

            def _one(_carry, sigma_kij):
                rotated = rotate_band_matrix(
                    sigma_kij, U, mesh=mesh, to_qp=bool(to_qp))
                rotated = jax.lax.with_sharding_constraint(
                    rotated, row_sh)
                return None, rotated

            _, out = jax.lax.scan(_one, None, cube, unroll=1)
            return jax.lax.with_sharding_constraint(out, cube_sh)

        fn = _kernel
        _SIGMA_OMEGA_ROTATE_CACHE[key] = fn
    return fn(sigma_c_omega_ry, U_dft_to_qp)


def _sigma_c_at_dft_diag_from_dft_cube(
    sigma_c_omega_dft_ry: jax.Array,
    sigma_result: SigmaResult,
    *,
    mesh: Mesh,
    print_fn: Callable = print,
) -> np.ndarray:
    """Interpolate the DFT-basis cube diagonal at the DFT energies.

    ``sigma_result.sigma_c_at_dft_diag_ev`` was formed before the SC
    finalize, from the diagonal of the QP-basis cube.  A diagonal cannot be
    similarity-transformed by itself.  After the full cube has undergone the
    one sanctioned QP-to-DFT rotation, extract its much smaller diagonal and
    repeat the canonical output interpolation on the already-recorded omega
    axis and DFT evaluation points.  The full cube is never gathered to host.

    Parameters
    ----------
    sigma_c_omega_dft_ry
        ``(n_omega, n_k, n_band, n_band)`` correlation operator in the DFT
        output basis.  It may be band-sharded.
    sigma_result
        Last-map result carrying the omega grid and DFT-relative evaluation
        energies used by the original interpolation.
    mesh
        Device mesh that owns the cube.

    Returns
    -------
    np.ndarray
        ``(n_k, n_band)`` complex correlation diagonal at ``E_DFT``, in eV
        and in the DFT output basis.
    """
    from .qsgw_utils import (
        extract_sigma_diag_replicated,
        interp_along_omega,
        resolve_out_of_range_policy,
    )

    if (sigma_result.omega_grid_ev is None
            or sigma_result.omega_dft_rel_ev is None):
        raise ValueError(
            "a dynamic SC output cube needs its omega grid and DFT-relative "
            "evaluation energies to rebuild Sigma_c(E_DFT)")
    diagonal_ev = np.asarray(extract_sigma_diag_replicated(
        sigma_c_omega_dft_ry, mesh)) * RYD_TO_EV
    return interp_along_omega(
        diagonal_ev,
        np.asarray(sigma_result.omega_grid_ev, dtype=np.float64),
        np.asarray(sigma_result.omega_dft_rel_ev, dtype=np.float64),
        out_of_range=resolve_out_of_range_policy(),
        context="DFT-basis Sigma_c at E_DFT after SC finalize",
        print_fn=print_fn,
    )


@_functools.partial(jax.jit, static_argnames=("mesh", "to_qp"))
def _rotate_fixed_matrix(A: jax.Array, U_dft_to_qp: jax.Array, *,
                         mesh: Mesh, to_qp: bool) -> jax.Array:
    """Shared distributed matrix basis change, kept two-axis sharded."""
    from .qsgw_density import rotate_band_matrix

    out = rotate_band_matrix(
        A, U_dft_to_qp, mesh=mesh, to_qp=bool(to_qp))
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, _band_rotation_spec()))


def run_fixed_sigma_evsc(
    sigma_result: SigmaResult,
    kin_ion_dft_ry: jax.Array,
    e_dft_kn_ry,
    *,
    config,
    meta,
    band_slices: BandSlices,
    wfn,
    mesh_xy: Mesh,
    sc_context: dict | None = None,
    print_fn: Callable = print,
) -> FixedSigmaEVSCResult:
    """Iterate a fixed full-matrix Sigma(omega) table to eigenvalue SC.

    This is the single fixed-table engine used by both the opt-in
    ``eqp2.dat`` treatment and the innermost two-level QSGW cycle.
    Screening, W, and every self-energy diagram are fixed for the duration
    of the call.  Its fixed-point variable is the Hermitian QP Hamiltonian
    in the ORIGINAL DFT basis.  For an input ``H_p^DFT`` it diagonalizes

    ``H_p^DFT U_p = U_p E_p``

    rotates the stored cube as
    ``Sigma_c,p(omega) = U_p^dagger Sigma_c,DFT(omega) U_p``.  The EQP2
    consumer forms the historical half-sum

    ``Sigma_eff,p[m,n] = 1/2 [Sigma_p,mn(E_p,m)
                              + Sigma_p,mn(E_p,n)]^h``;

    the nested SC consumer instead evaluates its diagonal at ``E_p,m`` and
    every off-diagonal at ``E_F``, using the same shared interpolation owner,

    and returns ``F(H_p)^DFT = H0_DFT + U_p Sigma_eff,p U_p^dagger``.
    Keeping the fixed-point carry in one coordinate system is what makes the
    rCROP residual meaningful: eigenvector phases and degenerate-subspace
    gauges never enter the object being mixed.

    The frequency table itself is never recomputed.  Protected/in-range
    states must remain covered by it; optional out-of-range states are
    removed from the Sigma Hamiltonian and replaced by the same no-lag
    semicore/conduction scissor policy as the ordinary SC driver.  Therefore
    an internal endpoint clamp can occur only in matrix entries that the
    partition immediately discards.  Convergence is the maximum absolute
    eigenvalue difference between ``F(H)`` and ``H`` over the protected or
    otherwise non-scissored states.  ``eqp2_accelerator=rcrop`` accelerates
    that fixed-basis map; ``linear`` is its unaccelerated Picard fallback.
    """
    from common import sanity
    from .qsgw_utils import (
        assert_omega_grid_covers, build_qsgw_sigma_xc, omega_coverage)

    sigma_c_dft = sigma_result.sigma_c_omega_kij_ry
    sigma_x_dft = sigma_result.sigma_x_kij_ry
    v_h_dft = sigma_result.v_h_kij_ry
    omega_ev = sigma_result.omega_grid_ev
    omega_ry = sigma_result.omega_grid_ry
    efermi_ev = sigma_result.efermi_dft_ev
    missing = [name for name, value in (
        ("sigma_c_omega_kij_ry", sigma_c_dft),
        ("sigma_x_kij_ry", sigma_x_dft),
        ("v_h_kij_ry", v_h_dft),
        ("omega_grid_ev", omega_ev),
        ("omega_grid_ry", omega_ry),
        ("efermi_dft_ev", efermi_ev),
    ) if value is None]
    if missing:
        raise ValueError(
            "write_eqp2=true requires the one-shot dynamic full-matrix "
            "Sigma(omega) result, but these fields are absent: "
            + ", ".join(missing))

    sc_mode = sc_context is not None
    if sc_mode:
        required_context = (
            "inputs", "H_seed_dft_ry", "sigma_basis_U", "partition",
            "band_classes", "exact_hartree_dft")
        absent = [name for name in required_context
                  if name not in sc_context]
        if absent:
            raise ValueError(
                "SC fixed-table context is missing: " + ", ".join(absent))
        sc_inputs = sc_context["inputs"]
        ks = _kstar(sc_inputs)
        basis_U = sc_context["sigma_basis_U"]

        # The Sigma producer returns its table in the QP basis of the G that
        # built it.  The fixed-table engine's carry is always in the original
        # DFT basis, so rotate each immutable operator back exactly once.
        sigma_c_dft = _rotate_sigma_omega_cube(
            sigma_c_dft, basis_U, mesh=mesh_xy, to_qp=False)
        sigma_x_dft = _rotate_fixed_matrix(
            sigma_x_dft, basis_U, mesh=mesh_xy, to_qp=False)
        if not bool(sigma_result.hartree_omitted):
            v_h_dft = _rotate_fixed_matrix(
                v_h_dft, basis_U, mesh=mesh_xy, to_qp=False)

        if not ks.is_identity:
            # KStarMap selects a leading k axis.  The cube's k axis is one,
            # so expose it as leading only for this index selection.
            sigma_c_dft = jnp.swapaxes(
                ks.select(jnp.swapaxes(sigma_c_dft, 0, 1)), 0, 1)
            sigma_x_dft = ks.select(sigma_x_dft)
            if not bool(sigma_result.hartree_omitted):
                v_h_dft = ks.select(v_h_dft)

    e_dft_ry = np.asarray(e_dft_kn_ry, dtype=np.float64)
    if e_dft_ry.ndim != 2:
        raise ValueError(
            f"run_fixed_sigma_evsc: E_DFT must be (nk,nb), got "
            f"{e_dft_ry.shape}.")
    nk, nb = (int(v) for v in e_dft_ry.shape)
    b0 = int(band_slices.b0)
    nb_sigma = int(band_slices.nb_sigma)
    if nb != nb_sigma:
        raise ValueError(
            "run_fixed_sigma_evsc: E_DFT is not the canonical Sigma band "
            f"window: nb={nb}, BandSlices.sigma has {nb_sigma} bands.")
    n_occ = int(meta.nelec) - b0
    if not (0 < n_occ < nb):
        raise ValueError(
            "run_fixed_sigma_evsc: the fixed-Sigma ladder needs an occupied "
            "and an unoccupied frontier inside the active window; got "
            f"b0={b0}, meta.nelec={int(meta.nelec)}, nb={nb}.")
    expected_cube = (len(np.asarray(omega_ev)), nk, nb, nb)
    if tuple(sigma_c_dft.shape) != expected_cube:
        raise ValueError(
            "run_fixed_sigma_evsc: full Sigma cube and DFT ladder disagree; "
            f"got {tuple(sigma_c_dft.shape)} and {e_dft_ry.shape}, expected "
            f"{expected_cube}.")

    sanity.refuse_nonfinite("eqp2 E_DFT", e_dft_ry, print_fn=print_fn)
    sanity.refuse_nonfinite(
        "eqp2 fixed Sigma_c(omega)", sigma_c_dft, print_fn=print_fn)

    rotation_spec = _band_rotation_spec()
    matrix_sharding = NamedSharding(mesh_xy, rotation_spec)
    ident = np.broadcast_to(
        np.eye(nb, dtype=np.complex128)[None, :, :], (nk, nb, nb)).copy()
    U_seed = _place(ident, mesh_xy, rotation_spec)
    if sc_mode and bool(sigma_result.hartree_omitted):
        exact_hartree_dft = sc_context["exact_hartree_dft"]
        if exact_hartree_dft is None:
            raise ValueError(
                "SC fixed-table cycle received hartree_omitted=true "
                "without its caller-owned exact Hartree operator")
        h0_dft = (jnp.asarray(kin_ion_dft_ry)
                  + jnp.asarray(exact_hartree_dft.total))
    else:
        h0_dft = jnp.asarray(kin_ion_dft_ry) + jnp.asarray(v_h_dft)
    h0_dft = _place(h0_dft, mesh_xy, rotation_spec)
    sigma_c_dft = _place(
        sigma_c_dft, mesh_xy, P(None, None, "x", "y"))
    sigma_x_dft = _place(sigma_x_dft, mesh_xy, rotation_spec)
    omega_ev = np.asarray(omega_ev, dtype=np.float64)
    omega_ry = np.asarray(omega_ry, dtype=np.float64)
    efermi_ev = float(efermi_ev)
    if sc_mode:
        partition = sc_context["partition"]
    else:
        e_dft_full_ry = np.asarray(wfn.energies[0], dtype=np.float64)
        partition = build_omega_band_partition(
            e_dft_ry, e_dft_full_ry,
            band_offset=b0,
            omega_min_abs_ev=float(omega_ev[0]) + efermi_ev,
            omega_max_abs_ev=float(omega_ev[-1]) + efermi_ev,
            label="EQP2", print_fn=print_fn)
    protected = np.asarray(
        partition.protected_mask, dtype=bool).reshape(-1)
    in_range = np.asarray(
        partition.in_range_mask, dtype=bool).reshape(-1)
    required_kn = np.broadcast_to(
        (protected | in_range)[None, :], e_dft_ry.shape)
    valence_mask_kn = (
        np.asarray(sc_inputs.valence_mask_active_kn, dtype=bool)
        if sc_mode else np.broadcast_to(
            (np.arange(nb) + b0 < int(meta.nelec))[None, :],
            e_dft_ry.shape))
    if sc_mode and not ks.is_identity:
        valence_mask_kn = np.asarray(ks.select(valence_mask_kn), dtype=bool)

    # EQP2 changes eigenvalues only, but metals still need the same fixed-N
    # chemical potential and three-way valence/crossing/conduction split as
    # the main SC map.  On an insulator this closure is never called and the
    # established fixed-band-cut midgap path remains exact.
    use_mp1 = bool(
        sc_context.get("occupation_state") is not None
        if sc_mode else getattr(config, "occ_smearing_family", None))
    if use_mp1 and not sc_mode:
        from psp.get_DFT_mtxels import spin_degeneracy_factor
        from .efermi import solve_mp1_occupations

        _state_capacity = float(spin_degeneracy_factor(wfn))
        _kweights = np.full(nk, 1.0 / nk, dtype=np.float64)

        def _occupation_state(e_kn_ry):
            return solve_mp1_occupations(
                np.asarray(e_kn_ry, dtype=np.float64), _kweights,
                float(wfn.num_electrons),
                float(config.occ_broadening_ry),
                state_capacity=_state_capacity,
                clamp_tol=float(config.occupation_clamp_tol))

        efermi_dft_scissor_ry, _ = _occupation_state(e_dft_ry)
    elif not sc_mode:
        efermi_dft_scissor_ry = float(
            _midgap_efermi(jnp.asarray(e_dft_ry), n_occ))
    else:
        efermi_dft_scissor_ry = float(sc_inputs.efermi_dft_ry)

    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import KStarMap
    kstar = ks if sc_mode else KStarMap.identity(nk)
    controls = config.sc if sc_mode else config.eqp2
    tol_ev = float(controls.tol_ev)
    max_iter = int(controls.max_iter)
    accelerator = str(controls.accelerator)
    history_depth = int(controls.history_depth)
    cycle_label = "SC fixed table" if sc_mode else "EQP2"
    eigh_kind = _resolve_sc_eigh(nb, mesh_xy, config, print_fn=print_fn)

    # The seed's eigensystem is exactly the DFT input basis.  Using this
    # explicit diagonal matrix (and its explicit identity U on call zero)
    # makes the first map evaluation exactly the requested one-shot
    # Sigma_mn(E_m^DFT)+Sigma_mn(E_n^DFT) construction, including inside a
    # degenerate DFT manifold where a generic eigh may choose another gauge.
    if sc_mode:
        H_seed = _place(
            sc_context["H_seed_dft_ry"], mesh_xy, rotation_spec)
    else:
        H_seed_host = np.zeros((nk, nb, nb), dtype=np.complex128)
        H_seed_host[:, np.arange(nb), np.arange(nb)] = e_dft_ry
        H_seed = _place(H_seed_host, mesh_xy, rotation_spec)

    def _eigh(H):
        H = _place(H, mesh_xy, rotation_spec)
        return _sc_eigh_bands(
            H, kind=eigh_kind, mesh_xy=mesh_xy, config=config)

    def _candidate_efermi(H):
        if sc_mode:
            return _partitioned_candidate_efermi(
                H, inputs=sc_inputs, kstar=kstar, n_occ=n_occ,
                use_mp1=use_mp1)
        e_candidate, _ = _eigh(H)
        if use_mp1:
            mu_ry, _ = _occupation_state(e_candidate)
            return float(mu_ry)
        return float(_midgap_efermi(e_candidate, n_occ))

    map_calls = [0]
    last_fixed_map = [None]

    def _fixed_map(H_dft):
        call_index = map_calls[0]
        H_dft = _place(H_dft, mesh_xy, rotation_spec)
        H_dft = 0.5 * (
            H_dft + jnp.conj(jnp.swapaxes(H_dft, -1, -2)))
        if call_index == 0 and not sc_mode:
            e_in_ry = e_dft_ry
            U_dft_to_qp = U_seed
        else:
            e_in_j, U_dft_to_qp = _eigh(H_dft)
            e_in_ry = np.asarray(e_in_j, dtype=np.float64)

        if use_mp1 and not sc_mode:
            from .scissor import classify_scissor_bands
            _, f_in = _occupation_state(e_in_ry)
            band_classes = classify_scissor_bands(f_in)
            if call_index == 0:
                print_fn(
                    f"    EQP2 scissor classes: {band_classes.summary()}")
        elif sc_mode:
            band_classes = sc_context["band_classes"]
        else:
            band_classes = None

        e_rel_ev = e_in_ry * RYD_TO_EV - efermi_ev
        covered, n_out, _ = omega_coverage(omega_ev, e_rel_ev)
        required_out = required_kn & ~covered
        n_required_out = int(np.count_nonzero(required_out))
        if n_required_out:
            bad = e_rel_ev[required_out]
            worst = float(bad.flat[int(np.argmax(np.abs(bad)))])
            raise ValueError(
                "GATE eqp2_omega_coverage: fixed-Sigma eigenvalue "
                f"self-consistency requested Sigma at {n_required_out}/"
                f"{int(np.count_nonzero(required_kn))} protected/non-scissored "
                "energies outside the sampled "
                f"grid [{omega_ev[0]:+.3f}, {omega_ev[-1]:+.3f}] eV "
                f"(worst {worst:+.3f} eV).  An endpoint clamp would not be "
                "Sigma(E), so eqp2 is refused.  Widen "
                "sigma_omega_min_ev / sigma_omega_max_ev or add a "
                "sigma_omega_patches_ev patch.")
        assert_omega_grid_covers(
            e_rel_ev / RYD_TO_EV, required_kn & covered, omega_ry,
            context=f"eqp2 map call {call_index + 1}")

        # Both Sigma operands are expressed in the basis whose energies label
        # their m,n indices.  The full cube is rotated from the ORIGINAL DFT
        # basis on every call, not successively, so there is no cumulative
        # basis drift.
        if call_index == 0:
            sigma_x_qp = sigma_x_dft
            sigma_c_qp = sigma_c_dft
        else:
            sigma_x_qp = _rotate_fixed_matrix(
                sigma_x_dft, U_dft_to_qp, mesh=mesh_xy, to_qp=True)
            sigma_c_qp = _rotate_sigma_omega_cube(
                sigma_c_dft, U_dft_to_qp, mesh=mesh_xy, to_qp=True)

        sigma_xc_qp, diagnostics = build_qsgw_sigma_xc(
            sigma_c_qp, sigma_x_qp, omega_ev, e_rel_ev, mesh_xy,
            replicated_output=False,
            offdiagonal_efermi_ev=(0.0 if sc_mode else None))
        if int(diagnostics["n_clipped"]) != int(n_out):
            raise RuntimeError(
                "run_fixed_sigma_evsc: omega_coverage/build_qsgw_sigma_xc "
                f"disagree ({n_out} outside versus "
                f"{int(diagnostics['n_clipped'])} clipped).")
        if call_index == 0 and n_out:
            print_fn(
                f"    EQP2: {n_out}/{e_rel_ev.size} optional out-of-grid "
                "Sigma evaluations are edge-clamped only inside matrix "
                "entries discarded by the partition, then replaced by the "
                "semicore/conduction scissor.")
        sigma_xc_dft = _rotate_fixed_matrix(
            sigma_xc_qp, U_dft_to_qp, mesh=mesh_xy, to_qp=False)
        H_full = h0_dft + sigma_xc_dft
        H_full = 0.5 * (
            H_full + jnp.conj(jnp.swapaxes(H_full, -1, -2)))
        if sc_mode:
            # Match the main SC map's accidental-degeneracy and named-buffer
            # policy.  The shared scissor helper below remains the sole owner
            # of the three-way band replacement.
            if not bool(getattr(config, "no_degen_averaging", False)):
                from .degen_average import average_matrix_diagonal
                delta_h = average_matrix_diagonal(
                    H_full - jnp.asarray(kin_ion_dft_ry),
                    energies_kn_ry=e_dft_ry,
                    tol_ry=(float(config.sc.exact_degeneracy_tol_ev)
                            / RYD_TO_EV),
                    mesh_xy=mesh_xy)
                H_full = jnp.asarray(kin_ion_dft_ry) + delta_h
            buffer_mask = np.asarray(
                sc_context.get("buffer_mask", np.zeros(nb, dtype=bool)),
                dtype=bool)
            if (buffer_mask.any()
                    and config.sc.buffer_mode == "carry"):
                H_full = _carry_sc_buffer_diagonal(
                    H_full, H_dft, jnp.asarray(buffer_mask, dtype=bool))
        H_out, _ = _apply_scissor_partition_policy(
            H_full, e_dft_ry, valence_mask_kn, partition, kstar,
            efermi_dft_ry=efermi_dft_scissor_ry,
            n_occ=n_occ,
            candidate_efermi_fn=_candidate_efermi,
            band_classes=band_classes,
            scissor_fit=(sc_context.get("scissor_fit")
                         if sc_mode else None),
            use_valence_fit=(
                config.sc.tail_fit == "buffer_edges" if sc_mode else False),
            label=("SC fixed table" if sc_mode else "EQP2"),
            print_fn=print_fn)
        if sc_mode and not kstar.is_identity:
            _check_kstar_spread(
                kstar, kstar.broadcast(H_out), print_fn=print_fn)
        H_out = _place(H_out, mesh_xy, rotation_spec)
        sanity.refuse_nonfinite(
            f"eqp2 H map call {call_index + 1}", H_out,
            print_fn=print_fn)

        e_out_j, U_out = _eigh(H_out)
        e_out_ry = np.asarray(e_out_j, dtype=np.float64)
        sanity.refuse_nonfinite(
            f"eqp2 E map call {call_index + 1}", e_out_ry,
            print_fn=print_fn)
        verdict = protected_band_convergence(
            e_out_ry * RYD_TO_EV, e_in_ry * RYD_TO_EV,
            protected, in_range, tol_ev)
        residual_ev = float(verdict.max_abs_ev)
        map_calls[0] += 1
        last_fixed_map[0] = (H_out, e_out_ry, U_out, residual_ev)
        return H_out, e_out_ry, U_out, residual_ev

    print_fn(
        f"[ {cycle_label} | fixed Sigma(omega), screening unchanged | "
        f"criterion max|dE| over non-scissored states "
        f"<= {tol_ev * 1e3:.3f} meV | "
        f"accelerator {accelerator} | max_iter {max_iter} ]")

    def _verify_final_map(H, *, source: str):
        """Evaluate once after the putative final rotation/scissor."""
        H_check, e_check, U_check, residual_check = _fixed_map(H)
        print_fn(
            f"[ {cycle_label} | final post-rotation verification ({source}) | "
            f"max|dE| {residual_check * 1e3:.6f} meV | "
            f"{'VERIFIED' if residual_check <= tol_ev else 'resume'} ]")
        return H_check, e_check, U_check, residual_check

    if accelerator == "linear":
        H = H_seed
        for iteration in range(1, max_iter + 1):
            H, e_out_ry, U_out, residual_ev = _fixed_map(H)
            print_fn(
                f"[ {cycle_label} | Picard {iteration:02d}/{max_iter:02d} | "
                f"max|dE| {residual_ev * 1e3:.6f} meV | "
                f"{'CONVERGED' if residual_ev <= tol_ev else 'continue'} ]")
            if residual_ev <= tol_ev:
                H, e_out_ry, U_out, residual_ev = _verify_final_map(
                    H, source=f"Picard {iteration:02d}")
                if residual_ev <= tol_ev:
                    return FixedSigmaEVSCResult(
                        energies_ry=e_out_ry, U_dft_to_qp=U_out,
                        H_qp_dft_ry=H,
                        iterations=map_calls[0], residual_ev=residual_ev)
    else:
        from mixing.acceleration import rcrop_nojit

        class _EQP2Converged(Exception):
            def __init__(self, H, energies, rotations, residual):
                self.H = H
                self.energies = energies
                self.rotations = rotations
                self.residual = residual

        last_accepted_residual = [float("inf")]
        residual_calls = [0]

        def _residual_fn(H):
            call_index = residual_calls[0]
            H_in = 0.5 * (H + jnp.conj(jnp.swapaxes(H, -1, -2)))
            H_in = _place(H_in, mesh_xy, rotation_spec)
            H_out, e_out, U_out, residual = _fixed_map(H_in)
            residual_calls[0] += 1
            role = ("initial" if call_index == 0 else
                    "trial" if call_index % 2 else "accepted")
            print_fn(
                f"[ {cycle_label} | rCROP residual {call_index + 1:02d} | {role} | "
                f"max|dE| {residual * 1e3:.6f} meV | "
                f"{'CONVERGED' if residual <= tol_ev else 'continue'} ]")
            if role != "trial":
                last_accepted_residual[0] = residual
                if residual <= tol_ev:
                    H_check, e_check, U_check, residual_check = _verify_final_map(
                        H_out, source=f"rCROP {role} {call_index + 1:02d}")
                    last_accepted_residual[0] = residual_check
                    if residual_check <= tol_ev:
                        raise _EQP2Converged(
                            H_check, e_check, U_check, residual_check)
            return _place(H_out - H_in, mesh_xy, rotation_spec)

        try:
            rcrop_nojit(
                _residual_fn, H_seed, m=history_depth, maxit=max_iter,
                # The physical L-infinity eigenvalue test above owns stopping.
                tol=0.0, print_fn=None, entry_sharding=matrix_sharding)
        except _EQP2Converged as stop:
            return FixedSigmaEVSCResult(
                energies_ry=np.asarray(stop.energies, dtype=np.float64),
                U_dft_to_qp=stop.rotations,
                H_qp_dft_ry=stop.H,
                iterations=map_calls[0],
                residual_ev=float(stop.residual),
            )
        residual_ev = last_accepted_residual[0]

    if sc_mode and last_fixed_map[0] is not None:
        H_last, e_last, U_last, residual_ev = last_fixed_map[0]
        print_fn(
            f"[ {cycle_label} | STALLED after {map_calls[0]} map "
            f"evaluations at max|dE|={residual_ev * 1e3:.6f} meV; "
            "returning the last evaluated table map so frozen-W Sigma can "
            "be rebuilt ]")
        return FixedSigmaEVSCResult(
            energies_ry=np.asarray(e_last, dtype=np.float64),
            U_dft_to_qp=U_last, H_qp_dft_ry=H_last,
            iterations=map_calls[0], residual_ev=float(residual_ev),
            converged=False)

    raise RuntimeError(
        "eqp2 fixed-Sigma eigenvalue self-consistency did not converge: "
        f"accepted max|dE|={residual_ev * 1e3:.6f} meV after "
        f"{map_calls[0]} map evaluations (accelerator={accelerator}, "
        f"max_iter={max_iter}), above the {tol_ev * 1e3:.6f} meV cutoff.  "
        "No eqp2 file was written; raise eqp2_max_iter or choose "
        "eqp2_accelerator=linear only after inspecting the map history and "
        "Sigma(omega) grid.")


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
# OFF BY DEFAULT for scalar QSGW (``config.density_self_consistent``);
# config resolution enables it whenever bispinor QSGW is requested without
# an explicit setting.  With it off this module is byte-identical to before,
# which is what keeps
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
    from common.wfn_layout import band_sphere_spec

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
    from common.four_current_model import resolve_four_current_representation
    representation = resolve_four_current_representation(
        bool(inputs.config.bispinor), inputs.config.bispinor_gw)
    carrier_bispinor = bool(representation.current_bispinor)
    carrier_lift = representation.current_lift or "raw"
    # The device placement is mesh-specific.  Keeping the mesh in the key
    # prevents a later calculation in the same process from reusing a buffer
    # whose devices belong to an earlier runtime.
    key = (id(inputs.wfn), id(inputs.mesh_xy), b_lo, b_hi,
           carrier_bispinor, carrier_lift)
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
                bands=(b_lo, b_hi), k="full_bz", sharding=spec,
                bispinor=carrier_bispinor,
                bispinor_lift=carrier_lift)
            # Use the loader's canonical resident index.  rho_from_wfns and
            # mtxel_sweep both accept a jax.Array and then their jnp.asarray
            # calls are identity operations.  Caching the host table here
            # made each SC map independently stage the same replicated
            # nk*Ngrid*i32 buffer.
            bidx = inputs.wfn.box_index_dev(
                k="full_bz", mesh=inputs.mesh_xy)
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


# THE DENSITY-SC ROW, AND ITS CHILDREN.  ``vh.rebuild`` is the whole
# rebuild; under it the timing tree carries ``vh.psi_load``,
# ``vh.rho``, ``vh.poisson`` and ``mtxel.sweep``;
# four-current runs also carry ``vh.transverse_field`` while scalar and
# transverse matrix elements ride the same packed sweep.
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
def rebuild_hartree_dft_basis(inputs, U_qp, E_qp_ry) -> SCExactHartree:
    """Exact direct Hartree in the DFT basis from iteration-i orbitals.

    The cycle this closes::

        H_i --eigh--> E, U_qp --+--> psi_qp = rotate(psi_dft, U_qp)
                                +--> occ = f(E);  rho_i = sum_n f|psi_qp|^2
                                +--> V_H[rho_i],  W_i,  Sigma_i
        H_{i+1} = T + V_loc + V_NL + V_H[rho_i] + Sigma[psi_i, W_i]

    V_H and Sigma are built from the SAME iteration-i orbitals and both
    land in H_{i+1}, so the fixed point is rho = rho[psi(H)] with H
    containing V_H[rho].  ``kin_ion_dft`` stays pristine T+V_loc+V_NL, so
    V_H arrives only through ``delta_h`` and cannot double-count.

    RETURNED IN THE DFT BASIS, AND CONTRACTED THERE.  The density scan applies
    ``U_qp`` without materialising the full rotated wavefunction; matrix
    elements use the unrotated DFT orbitals because H is assembled around a
    pristine DFT-basis ``kin_ion_dft``.

    No mixing: straight rho_out feedback, by owner ruling (2026-08-04).
    """
    from gw.efermi import (fermi_level_step, occupied_band_count,
                           step_occupations)
    from gw.qsgw_density import rho_from_wfns
    from common.four_current_model import resolve_four_current_representation
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from psp.get_DFT_mtxels import build_hartree_potential
    from common.mtxel_sweep import (SweepGeometry,
                                    four_current_potential_operator,
                                    local_potential_operator,
                                    sweep_matrix_elements)
    from psp.dft_operators import padded_gvectors

    psi_G, bidx = _dft_psi_sphere(inputs)
    nk, nb = int(psi_G.shape[0]), int(psi_G.shape[1])
    kweights = np.full(nk, 1.0 / nk)          # full BZ => uniform, no star

    # E stays on the device: both are jit kernels over ``E`` (``gw.efermi``
    # header) and only E_F and the degeneracy flag cross.
    if inputs.material_class == "metal":
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

    f_spin = spin_degeneracy_factor(inputs.wfn)
    grid = tuple(int(v) for v in inputs.wfn.fft_grid)
    representation = resolve_four_current_representation(
        bool(inputs.config.bispinor), inputs.config.bispinor_gw)
    include_current = bool(representation.current_bispinor)
    charge_ns = (int(psi_G.shape[2]) if representation.charge_bispinor
                 else int(inputs.wfn.nspinor))
    fields = rho_from_wfns(
        psi_G, occ, kweights, U=U_qp, mesh=inputs.mesh_xy,
        box_index=bidx, fft_grid=grid,
        cell_volume=float(inputs.wfn.cell_volume),
        spin_degeneracy=f_spin,
        include_dirac_current=include_current,
        charge_nspinor=charge_ns)
    rho_r = fields[0] if include_current else fields
    expected_electrons = f_spin * occupied_band_count(occ, kweights)
    with timing.section("vh.poisson"):
        V_H_r = build_hartree_potential(
            rho_r, inputs.wfn,
            truncation_2d=bool(getattr(inputs.config, "sys_dim", 3) == 2),
            expected_electrons=expected_electrons,
            print_fn=inputs.print_fn)

    V_T_r = None
    if include_current:
        from ffi import _services
        _services.ensure_on_path()
        from symmetry_maps import project_polar_fft_field
        from psp.dft_operators import transverse_potential_from_current
        from vcoul import COULOMB_GAUGE_TT_SIGN

        current_raw = np.asarray(fields[1:], dtype=np.float64)
        ngrid = int(np.prod(grid))
        current_g0 = np.sum(current_raw, axis=(-3, -2, -1)) / np.sqrt(ngrid)
        current_scale = max(
            float(np.linalg.norm(current_raw)), np.finfo(np.float64).tiny)
        inputs.print_fn(
            "    SC Dirac-current G=0 diagnostic (J=j/c): "
            f"||J0||={float(np.linalg.norm(current_g0)):.6e}, "
            f"||J||={current_scale:.6e}, "
            f"ratio={float(np.linalg.norm(current_g0)) / current_scale:.6e}; "
            "periodic TT sets G=0 to zero")
        projection = project_polar_fft_field(current_raw, inputs.sym)
        inputs.print_fn(
            "    SC Dirac-current symmetry projection: "
            f"rows={projection.n_symmetry_rows} "
            f"(antiunitary={projection.n_antiunitary_rows}), "
            f"movement={projection.relative_movement:.6e}, "
            f"residual={projection.relative_residual:.6e} <= "
            f"{projection.relative_residual_tolerance:.6e}")
        with timing.section("vh.transverse_field"):
            V_T_r = transverse_potential_from_current(
                jnp.asarray(projection.field, dtype=jnp.float64),
                jnp.asarray(inputs.wfn.bdot, dtype=jnp.float64),
                jnp.asarray(inputs.wfn.bvec, dtype=jnp.float64),
                float(inputs.wfn.blat),
                bool(getattr(inputs.config, "sys_dim", 3) == 2),
                tt_metric_sign=float(COULOMB_GAUGE_TT_SIGN))

    gtab = padded_gvectors(inputs.wfn, k="full_bz")
    psi_charge = psi_G[:, :, :charge_ns, :]
    geom_matrix = SweepGeometry(
        mesh=inputs.mesh_xy, fft_grid=grid,
        ngkmax=int(psi_G.shape[3]), nb=nb,
        ns=(int(psi_G.shape[2]) if V_T_r is not None
            else int(psi_charge.shape[2])), nk=nk,
        cell_volume=float(inputs.wfn.cell_volume))
    if V_T_r is not None:
        H_pair = sweep_matrix_elements(
            psi_G,
            operator=four_current_potential_operator(
                geom_matrix, V_H_r, V_T_r,
                charge_nspinor=charge_ns),
            geom=geom_matrix,
            gvecs=gtab.gvecs, gmask=gtab.mask, box_index=bidx,
            kvecs=gtab.kvecs)
        H_scalar, H_transverse = H_pair[:, 0], H_pair[:, 1]
        del H_pair
    else:
        H_scalar = sweep_matrix_elements(
            psi_charge,
            operator=local_potential_operator(geom_matrix, V_H_r),
            geom=geom_matrix,
            gvecs=gtab.gvecs, gmask=gtab.mask, box_index=bidx,
            kvecs=gtab.kvecs)
        H_transverse = None
    return SCExactHartree(
        scalar_dft=H_scalar,
        transverse_dft=H_transverse,
        efermi_ry=float(e_f))


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


def _report_extrapolation_eqp_shift(
    H_extrap, H_unextrap, *, mesh_xy, n_occ, iteration, print_fn) -> None:
    """Diagonalize both Σ's H and report the correction AT THE EQP LEVEL.

    WHY A SECOND DIAGONALIZATION IS WORTH ITS COST.  The extrapolation report
    in ``gw.ppm_pipeline`` states the correction to Σ_c on the band diagonal.
    That is not the same number as the correction to E_nk, and the difference
    is not a detail: Σ's off-diagonals move too, so the eigenvalue shift is
    the diagonal shift PLUS the eigenvector response, and near a small gap the
    second term is not small.  The quantity an operator acts on is the eqp
    level, so the eqp level is what gets reported beside its un-extrapolated
    twin.

    TWO DIAGONALIZATIONS, NOT FOUR.  Both are ``eigvalsh`` — eigenvalues only,
    no eigenvectors, because nothing here feeds back into the iteration.  The
    rejected alternative is diagonalizing each of the three band brackets and
    extrapolating the EIGENVALUES, which is four diagonalizations AND is
    wrong: eigenvalues are not linear in the matrix, so an extrapolated
    spectrum is the spectrum of no Hamiltonian (see
    ``band_extrapolation.extrapolation_weights``).

    THE H's HERE ARE PRE-PARTITION, and that is the right pair to compare.
    ``apply_band_partition`` masks off-diagonals of non-protected bands and
    overwrites out-of-range diagonals with the per-iteration scissor; on the
    protected, in-range bands — the QP window, which is what this report is
    about — it changes nothing.  Comparing the pre-partition pair keeps the
    two arms differing in exactly ONE thing, the band-sum tail in Σ_c, rather
    than also differing in a scissor refitted separately on each.
    """
    _, eigvalsh = _kshard_eigh_kernels(mesh_xy, _band_rotation_spec())
    e_ext = np.asarray(eigvalsh(H_extrap), dtype=np.float64) * RYD_TO_EV
    e_n3 = np.asarray(eigvalsh(H_unextrap), dtype=np.float64) * RYD_TO_EV
    d = e_ext - e_n3
    nb = e_ext.shape[1]
    n_occ = max(0, min(int(n_occ), nb))

    lines = [
        f"    -- band-extrapolation effect on E_nk, iteration "
        f"{iteration} (two eigvalsh: extrapolated vs N3) --",
    ]
    if n_occ and n_occ < nb:
        # eigvalsh returns ASCENDING eigenvalues, so the band cut at n_occ is
        # the VBM/CBM split on both arms by construction.
        vbm_e, cbm_e = e_ext[:, n_occ - 1].max(), e_ext[:, n_occ].min()
        vbm_3, cbm_3 = e_n3[:, n_occ - 1].max(), e_n3[:, n_occ].min()
        lines += [
            f"       VBM   S(N3) = {vbm_3:+11.6f}  ->  S_inf = "
            f"{vbm_e:+11.6f} eV   ({vbm_e - vbm_3:+.6f})",
            f"       CBM   S(N3) = {cbm_3:+11.6f}  ->  S_inf = "
            f"{cbm_e:+11.6f} eV   ({cbm_e - cbm_3:+.6f})",
            f"       gap   S(N3) = {cbm_3 - vbm_3:+11.6f}  ->  S_inf = "
            f"{cbm_e - vbm_e:+11.6f} eV   "
            f"({(cbm_e - vbm_e) - (cbm_3 - vbm_3):+.6f})",
        ]
    lines.append(
        f"       over all (k, band): mean {d.mean():+.6f}  "
        f"RMS {float(np.sqrt(np.mean(d ** 2))):.6f}  "
        f"max |dE| {float(np.max(np.abs(d))):.6f} eV")
    # THE SIGN IS THE DIAGNOSTIC.  The 1/N law was measured to decay FASTER
    # than the data over the sampled window, so the fit projects more
    # remaining tail than exists and lands BELOW the truth on every state
    # (module docstring, 2026-08-16).  A correction that is not one-signed
    # here is therefore a statement about this deck, and worth seeing.
    frac_neg = float(np.mean(d < 0.0))
    lines.append(
        f"       {100 * frac_neg:.1f} % of states moved DOWN.  The 1/N form "
        f"is known to undershoot one-signed against a measured S(508); a "
        f"MIXED sign here means this deck's three points are not in the "
        f"regime the form was calibrated on.")
    print_fn("\n".join(lines))


def _sc_head_frequency_plan(
        config, quad, *, material_class, certified_fit=None, mesh_xy=None):
    """Single frequency plan for SC body W and every head provenance arm.

    A live screening build takes its ceiling from ``quad``.  A certified-fit
    reuse has no quadrature, so its already-resolved fit path must be supplied
    and the stored scalar-head samples become the exact plan provenance.
    """
    from .screening import screening_requests_for

    requests = screening_requests_for(config.compute_mode, config)
    mpa_plan = None
    if config.compute_mode is ComputeMode.MPA:
        from .mpa import sample_plan
        if quad is None:
            if certified_fit is None:
                raise ValueError(
                    "MPA SC frequency planning has no screening quadrature "
                    "and no certified fit supplied by the screening reuse "
                    "provider. Build screening normally or return the "
                    "already-resolved certified fit path from that provider.")
            if mesh_xy is None:
                raise ValueError(
                    "MPA certified-fit frequency planning requires mesh_xy "
                    "for the collective head-fit provenance read")
            from .mpa.model import make_mpa_plan_from_fit
            mpa_plan = make_mpa_plan_from_fit(
                config, certified_fit, mesh_xy=mesh_xy,
                material_class=material_class)
        else:
            from .mpa.model import make_mpa_plan
            mpa_plan = make_mpa_plan(
                config, quad, material_class=material_class)
        mpa_z = np.asarray(sample_plan.plan_z(mpa_plan), dtype=np.complex128)
        head_omegas = [complex(value) for value in mpa_z]
        # The metallic MPA grid starts just above the origin.  Preserve a
        # separate exact-static sample for do_G0; it is nonanalytic and must
        # never enter the Loewner sample vector.
        if bool(config.do_G0) and not np.any(mpa_z == 0.0 + 0.0j):
            head_omegas.append(0.0 + 0.0j)
        return requests, mpa_plan, head_omegas
    head_omegas = (
        [complex(req.omega_ry) for req in requests]
        if requests else [0.0 + 0.0j])
    return requests, None, head_omegas


def _partition_hysteresis_margin_ev(inputs: SCInputs) -> float:
    """Run-derived Schmitt margin for the re-anchored Sigma window."""
    # Half a sampled omega bin is unresolved by the grid.  Metallic
    # occupations also deliberately smear the Fermi edge over their physical
    # width.  A classification must move beyond both resolutions before it
    # can remove structure that the preceding map retained.
    return max(
        0.5 * float(inputs.config.sigma.omega_step_ev),
        float(inputs.config.occ_broadening_ry) * RYD_TO_EV,
    )


def _state_partition(state: SCState, inputs: SCInputs) -> BandPartition:
    """Current partition, with compatibility for synthetic bare states."""
    return state.partition if state.partition is not None else inputs.partition


def _include_fixed_table_verdict(
    state: SCState,
    verdict: ConvergenceVerdict,
    inputs: SCInputs,
) -> ConvergenceVerdict:
    """Prevent an outer/inner stop while the innermost table cycle stalled."""
    if not bool(getattr(inputs, "two_level_enabled", False)) or state.outputs is None:
        return verdict
    if bool(getattr(state.outputs, "fixed_sigma_cycle_converged", True)):
        return verdict
    if verdict.converged:
        inputs.print_fn(
            "  SC convergence held: the Sigma-rebuild residual is below "
            "cutoff but its innermost fixed-Sigma table cycle stalled")
    return replace(verdict, converged=False)


def gw_iteration_map(state: SCState, inputs: SCInputs) -> SCState:
    """One self-consistent QSGW step in the DFT basis.

    Pure function — no side effects on ``inputs.wfns_dft``.  All
    derived quantities (E_qp, U_qp, efermi) are recomputed each call.  The
    carried state is ``H_qp_dft`` plus the preceding protected-band decision;
    the latter supplies only edge hysteresis and never freezes the Fermi
    anchor or current-spectrum classification.

    Screening is mode-orthogonal: an outer call asks
    :func:`gw.screening.compute_screening_model` for the configured Σ
    scheme's screening representation and hands the resulting
    ``{role → W_q}`` dict to
    :func:`gw.sigma_dispatch.compute_sigma_xc`; an inner call receives that
    same mapping through ``inputs.frozen_screening`` and skips the producer.
    No ``compute_chi0`` call lives here directly — adding a new Σ scheme that
    wants extra W frequencies is purely a screening + compute_sigma_xc
    change.
    """
    n_occ = int(inputs.meta.nelec)
    frozen_screening = inputs.frozen_screening
    if (frozen_screening is not None
            and frozen_screening.sigma_model is None):
        raise ValueError(
            "GATE sc_frozen_screening_model_missing: a two-level inner "
            "map requested frozen screening but SCMapScreeningArtifacts "
            "contains no Sigma screening model")
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
        # ``_resolve_sc_eigh`` from ``config.sc.eigh``.  distrib_la's
        # native ``batch_reshard`` route gives each device nk/P of the
        # per-k diagonalisations but still lands one WHOLE (nb, nb) tile on
        # one device -- 1.6 GB at nb=1e4; its distributed backend spreads
        # each tile over the mesh instead (owner ruling 2026-08-04:
        # robustness at 1e4+ bands over speed at 1e3, where the native
        # batch wins by ~ndev).  Until
        # 2026-08-05 the distributed one was reachable ONLY by turning on
        # ``density_self_consistent``, a physics knob defaulting to False for
        # scalar QSGW,
        # so the default -- and only shipped -- configuration had no way
        # to ask for it.
        #
        # (b) WHICH E_F RULE -- a PHYSICS question, and it stays with
        # ``density_self_consistent``: the k-weighted step routine there,
        # the fixed-band-cut midgap otherwise.  Moving it would change
        # numbers; moving (a) does not.
        #
        # Both distrib_la routes return U at ``band_rotation_spec``, so
        # everything below is layout-blind.  The k-batched one used to
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
        E_qp_ry, U_qp = _sc_eigh_bands(
            state.H_qp_dft, kind=eigh_kind, mesh_xy=inputs.mesh_xy,
            config=inputs.config)

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

    # ------------------------------------------------------------------
    # RE-ANCHOR THE WINDOW ON THIS ITERATION'S FERMI LEVEL.
    #
    # The Sigma grid is built as mu +- the deck's half-width and the MPA
    # branches measure every state energy from the SAME mu
    # (``mpa.sigma._branches``: energy = enk - occupation_state.mu_ry, with
    # a hard refusal if the two disagree).  The band partition -- which
    # bands are trusted on that grid and which are handed to the scissor --
    # therefore has to be rebuilt against the same pair, or it answers a
    # question about a grid the run no longer has.
    #
    # It used to be built ONCE, before the loop, from the DFT spectrum and
    # the DFT mu.  Measured on the signed +-5 eV sodium deck, mu moves
    # +1.352 eV in ONE map (E_F(DFT) = +1.646762 -> E_F(F(H)) = +2.998851),
    # which left 2 of 24 bands in range, put 10 protected bands OUTSIDE the
    # grid, and gave the scissor fit zero qualifying samples so it returned
    # the identity and those bands took no correction at all.
    #
    # On the first map the spectrum IS the DFT spectrum and mu IS the DFT
    # mu, so this reproduces the frozen partition exactly; it only starts
    # to differ once the spectrum has actually moved.
    # ------------------------------------------------------------------
    partition = inputs.partition
    if entry_occ_state is not None:
        _mu_ev = float(entry_occ_state.mu_ry) * RYD_TO_EV
        # E_full is this map's active table on the full BZ -- the same rows
        # and bands as inputs.e_dft_active_kn_ry, which is what the partition
        # masks are applied to.
        _e_active_now = np.asarray(E_full, dtype=np.float64)
        partition = build_omega_band_partition(
            _e_active_now, np.asarray(enk_entry, dtype=np.float64),
            band_offset=int(inputs.band_slices.b0),
            omega_min_abs_ev=float(inputs.config.sigma.omega_min_ev) + _mu_ev,
            omega_max_abs_ev=float(inputs.config.sigma.omega_max_ev) + _mu_ev,
            previous_partition=(state.partition
                                if state.partition is not None
                                else inputs.partition),
            hysteresis_margin_ev=_partition_hysteresis_margin_ev(inputs),
            label=f"SC map {int(state.iteration)} (mu-anchored)",
            print_fn=inputs.print_fn)
    partition = _apply_sc_buffer_partition(partition, inputs)
    buffer_mask = _sc_buffer_mask(inputs)
    if buffer_mask.any():
        buffer_ids = np.flatnonzero(buffer_mask) + int(inputs.band_slices.b0) + 1
        inputs.print_fn(
            f"    SC window buffer: mode={inputs.config.sc.buffer_mode}, "
            f"bands={_band_ranges(buffer_mask, band_offset=int(inputs.band_slices.b0))} "
            f"(n={buffer_ids.size}); named core="
            f"{int(inputs.meta.nelec) - int(inputs.config.nval) + 1}-"
            f"{int(inputs.meta.nelec) + int(inputs.config.ncond)}")

    # Resolve an external MPA seed before any occupation consumer runs.  The
    # provider remains the sole owner of path resolution; production then
    # replays and asserts the stored occupation provenance on this exact
    # entry ladder.  Later maps retain only the fit's sample geometry and
    # rebuild screening from their live occupations.
    mpa_mode = inputs.config.compute_mode is ComputeMode.MPA
    screening_reuse = None
    certified_fit = None
    if mpa_mode and inputs.quad is None and frozen_screening is None:
        if inputs.config.head.correction is HeadCorrection.FULL:
            raise ValueError(
                "MPA certified-fit reuse with no screening quadrature cannot "
                "serve head_correction=full, whose current-iteration head "
                "must be folded through newly sampled W")
        cache = inputs.screening_seed_cache
        certified_fit = (cache.get("mpa_fit") if cache is not None else None)
        if state.iteration == 0:
            screening_reuse = (
                cache.get("mapping") if cache is not None else None)
            if screening_reuse is None:
                screening_reuse = inputs.screening_model_fn(
                    inputs.config.compute_mode, inputs.wfns_dft, inputs.V_q,
                    quad=inputs.quad, e_ref=inputs.e_ref, sym=inputs.sym,
                    centroid_indices=inputs.centroid_indices,
                    config=inputs.config, meta=inputs.meta,
                    mesh_xy=inputs.mesh_xy,
                    run_dir=os.path.join(inputs.input_dir, "tmp", "mpa"),
                    label=f"sc_{state.iteration:04d}",
                    head_resolver=inputs.head_resolver,
                    head_channel=getattr(inputs, 'head_channel', None),
                    wfn=inputs.wfn, mpa_plan=None,
                    iteration_head_response=None,
                    occupation_state=entry_occ_state,
                    material_class=inputs.material_class,
                    print_fn=inputs.print_fn)
                if isinstance(screening_reuse, dict):
                    certified_fit = screening_reuse.get("mpa_fit")
                if cache is not None and certified_fit is not None:
                    cache.update(
                        mapping=screening_reuse, mpa_fit=certified_fit)
            if certified_fit is None:
                raise ValueError(
                    "MPA screening reuse provider returned no certified "
                    "mpa_fit path")
            entry_occ_state = _certified_seed_occupation_state(
                inputs, enk_entry, certified_fit, entry_occ_state)
            if inputs.parallel_transport is not None:
                raise ValueError(
                    "certified MPA occupation replay with a live head "
                    "surface table is not implemented")
            entry_surface_weight_kn = None
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
    if logical_stop > tail_start:
        from .scissor import k_star_weights

        e_dft_fit = inputs.e_dft_active_kn_ry
        valence_fit = inputs.valence_mask_active_kn
        if not ks.is_identity:
            e_dft_fit = ks.select(e_dft_fit)
            valence_fit = ks.select(valence_fit)
        e_dft_fit_ev = np.asarray(e_dft_fit, dtype=np.float64) * RYD_TO_EV
        fit_mask_kn = np.broadcast_to(
            np.asarray(partition.in_range_mask, dtype=bool)[None, :],
            e_dft_fit_ev.shape)
        if inputs.config.sc.tail_fit == "buffer_edges":
            fit_mask_kn = fit_mask_kn & np.broadcast_to(
                buffer_mask[None, :], e_dft_fit_ev.shape)
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
            # The sum-band tail is not part of the rotated QP subspace. Its
            # update is a rigid edge scissor defined by the lowest accidental-
            # degeneracy manifold, not by a user-expanded near-degenerate/SOC
            # scale.  A resolved 1.7 meV pair is physics and remains two
            # distinct samples; only <=0.1 meV may be grouped here.
            conduction_frontier_tol_ev=(
                float(inputs.config.sc.exact_degeneracy_tol_ev)
                if inputs.config.sc.tail_fit == "frontier" else None),
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
            f"(n={tail_fit.n_fit_c}, w={tail_fit.w_fit_c:.0f}, "
            f"policy={inputs.config.sc.tail_fit})")

    wfns_qp = rotate_wavefunctions(
        inputs.wfns_dft, U_full,
        enk_active_new=E_full, enk_base=enk_base,
        efermi=float(efermi_ry),
        mesh_xy=inputs.mesh_xy,
        active_slice=inputs.band_slices.sigma,
    )

    # (``entry_occ_state`` was solved above, before the tail scissor that
    # feeds ``enk_base`` — see the block after ``E_full``.)

    # LIVE DENSITY SELF-CONSISTENCY (scalar opt-in, bispinor required).
    # V_H[rho_i] from THIS iteration's orbitals, alongside Sigma_i and from
    # the same U_qp, both feeding
    # H_{i+1}.  Off by default for scalar QSGW, so the one-shot equivalence
    # gate holds; bispinor QSGW config resolution enables the live path.
    exact_hartree_dft = None
    if bool(getattr(inputs.config, "density_self_consistent", False)):
        # rho is built from FULL-BZ psi (uniform weights, no star sum),
        # so it takes the broadcast U and E; the matrix it returns is
        # selected to the IBZ to match delta_h_dft.
        exact_hartree_dft = rebuild_hartree_dft_basis(
            inputs, U_full, E_full)
        from common import sanity as _sanity
        _sanity.check_finite(
            "V_H[SC] scalar", exact_hartree_dft.scalar_dft,
            print_fn=inputs.print_fn)
        if exact_hartree_dft.transverse_dft is not None:
            _sanity.check_finite(
                "H_T[SC] transverse", exact_hartree_dft.transverse_dft,
                print_fn=inputs.print_fn)
        inputs.print_fn(
            f"    density-SC: rebuilt exact scalar"
            f"{' + transverse' if exact_hartree_dft.transverse_dft is not None else ''} "
            f"Hartree from iteration {state.iteration} orbitals "
            f"(E_F = {exact_hartree_dft.efermi_ry:.6f} Ry)")

    # Same-run metal threading: the ENTRY-solved state feeds chi, the head
    # and Sigma — one mu per map call, from this call's spectrum.
    metal_occ_state = (entry_occ_state
                       if inputs.material_class == "metal" else None)

    def _screening(mpa_plan, iteration_head_response, *, producer=None,
                   quad_override=None):
        """Call the driver-owned producer/reuse provider at one map seam."""
        used_producer = (inputs.screening_model_fn
                         if producer is None else producer)
        used_quad = inputs.quad if quad_override is None else quad_override
        return used_producer(
            inputs.config.compute_mode, wfns_qp, inputs.V_q,
            quad=used_quad, e_ref=inputs.e_ref, sym=inputs.sym,
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
            material_class=inputs.material_class,
            print_fn=inputs.print_fn)

    # Per-mode screening plan.  The q->0 head uses this exact frequency/role
    # table so a Schur-folded probe can never drift from the body W it folds.
    if frozen_screening is None:
        requests, mpa_plan, head_omegas = _sc_head_frequency_plan(
            inputs.config, inputs.quad, certified_fit=certified_fit,
            mesh_xy=inputs.mesh_xy, material_class=inputs.material_class)
    else:
        requests, mpa_plan, head_omegas = (), None, ()

    # Per-iteration QSGW q->0 head.  The opt-in map is stationary even for
    # accelerators that evaluate one carry repeatedly: at iteration zero
    # DeltaH=0 and U=I, so this reconstructs the DFT head through the same
    # prevalidated path used thereafter.  Saved A/v and the carried H build
    # the direct tensor.  The already-resident, already-rotated centroid
    # wavefunctions build q-linear wings; they are not reloaded here.
    iteration_head = None
    iteration_head_response = None
    iteration_static_head_terms = inputs.static_head_terms
    head_occ_kn = None
    pt = getattr(inputs, "parallel_transport", None)
    fixed_dft_full_head = inputs.fixed_dft_head_response is not None
    if pt is not None or fixed_dft_full_head:
        from .head_correction import compute_static_head_terms_from_sample
        from .qsgw_head import finalize_iteration_head_samples
    if frozen_screening is not None:
        head_efermi_ry = (
            float(entry_occ_state.mu_ry)
            if entry_occ_state is not None else float(efermi_ry))
        iteration_head = _retarget_frozen_iteration_head(
            frozen_screening.iteration_head,
            energies_ry=E_full,
            occupations=wfns_qp.occ[:, inputs.band_slices.sigma],
            efermi_ry=head_efermi_ry,
        )
        iteration_static_head_terms = frozen_screening.static_head_terms
        if iteration_head is not None and bool(inputs.config.do_G0):
            from .head_correction import compute_static_head_terms_from_sample
            iteration_static_head_terms = compute_static_head_terms_from_sample(
                iteration_head.at(0.0 + 0.0j),
                occ=np.asarray(
                    iteration_head.sigma_occupations,
                    dtype=np.float64),
                cell_volume=float(inputs.meta.cell_volume),
                nk_tot=int(inputs.meta.nk_tot),
            )
        inputs.print_fn(
            "    SC inner: reused frozen screening/pole model; rebuilt G "
            "and Sigma on the current QP spectrum")
    elif pt is not None:
        from .qsgw_head import (
            assemble_delta_head_manifold,
            build_iteration_head_response,
        )

        nb_storage = int(pt.velocity_dft_cart.shape[-1])
        if int(wfns_qp.enk.shape[1]) < nb_storage:
            raise ValueError(
                "parallel-transport head storage has "
                f"{nb_storage} padded bands, but the SC wavefunction bundle "
                f"has only {wfns_qp.enk.shape[1]}.")
        # ``sc_head_update = dft_velocity`` carries no links, so DeltaH
        # has no consumer: skip its assembly rather than building an
        # O(nk*nb_storage^2) manifold and finite-link derivative for a term
        # that is then dropped.  Everything below this point is shared.
        forward_links = pt.forward_links
        delta_head = None
        if forward_links is not None:
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
            forward_links,
            pt.forward_neighbors,
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
            "QSGW finite-link covariant velocity" if forward_links is not None
            else "QP-rotated DFT p-matrix velocity")
        if mpa_mode:
            # The fit-sample count comes from the RETURNED plan --
            # ``mpa_z`` is a local inside ``_sc_head_frequency_plan`` and
            # was never visible here (KNOWN_LORRAX_ISSUES 2026-08-19 row;
            # every sc_head_update=dft_velocity + compute_mode=mpa
            # iteration raised NameError at this line before W sampling).
            # ``plan_z`` excludes the separately appended exact-static
            # G=0 head sample, which is exactly the count meant here.
            from .mpa import sample_plan as _sample_plan
            inputs.print_fn(
                "    SC head: occupation-aware QSGW response plus sharded "
                f"ISDF wings on the exact MPA z grid ({velocity_kind}, "
                f"nb={pt.nb_logical}, "
                f"fit samples={len(_sample_plan.plan_z(mpa_plan))})")
        else:
            inputs.print_fn(
                f"    SC head: {velocity_kind} + current-basis wings "
                "from saved parallel transport/current centroid bundle "
                f"(nb={pt.nb_logical}, samples={len(head_omegas)})")

    elif fixed_dft_full_head:
        # ``sc_head_update=off`` freezes the DFT direct response; it does not
        # turn a requested macroscopic W head back into the no-local-field
        # epsilon head.  Reuse the immutable DFT S/Y/Z on this iteration's
        # exact frequency plan, then fold it once through this iteration's W.
        # This branch keeps the public full/no-LF decision independent of the
        # QSGW velocity-update choice.  Only the small 3x3/vector response is
        # retained; body W remains 2-D sharded as before.
        iteration_head_response = inputs.fixed_dft_head_response
        head_occ_kn = np.asarray(
            iteration_head_response.sigma_occupations, dtype=np.float64)
        if tuple(iteration_head_response.omegas) != tuple(head_omegas):
            raise ValueError(
                "fixed DFT head response does not match the current "
                "screening frequency plan")
        inputs.print_fn(
            "    SC head: fixed DFT direct response plus matching wings "
            "on the current screening frequency plan; local fields fold "
            "once through this iteration's W (sc_head_update=off)")


    # Per-mode screening: solve W at every frequency the Sigma scheme needs.
    # XLA cache hits on iteration ≥ 2 (same shapes, new values).
    # The pre-plan reuse call already returned this map's complete fit mapping.
    # A live producer runs here, after the current head response exists.
    if frozen_screening is not None:
        W_by_role = frozen_screening.sigma_model
    elif screening_reuse is not None:
        W_by_role = screening_reuse
    elif mpa_mode and inputs.quad is None:
        # The external fit is valid only for the occupation state stamped in
        # it.  Once the SC spectrum changes, keep its exactly reconstructed
        # sample geometry but run the canonical producer on this map's live
        # occupations.  ``build_mpa_fit`` needs only the original frequency
        # ceiling from the quadrature object when an explicit plan is given.
        from types import SimpleNamespace
        from .mpa import sample_plan as _sample_plan
        from .screening import compute_screening_model as _live_screening
        plan_z = np.asarray(
            _sample_plan.plan_z(mpa_plan), dtype=np.complex128)
        stored_ceiling = SimpleNamespace(x_max=float(np.max(plan_z.real)))
        W_by_role = _screening(
            mpa_plan, iteration_head_response,
            producer=_live_screening, quad_override=stored_ceiling)
    else:
        W_by_role = _screening(mpa_plan, iteration_head_response)

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
    w_time_factor_cache = (
        frozen_screening.w_time_factor_cache
        if frozen_screening is not None else
        {} if inputs.two_level_enabled else None)
    sigma_wall_start = time.perf_counter()
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
        omit_v_h=exact_hartree_dft is not None,
        iteration_head=iteration_head,
        material_class=inputs.material_class,
        write_sigma_omega_h5=False,
        # Map 1 stays on the historical half-sum exactly, preserving the
        # one-shot identity gate.  Every later dynamic SC map uses the
        # owner-selected diagonal-at-E_m / off-diagonal-at-E_F law.
        qsgw_offdiagonal_efermi_ev=(
            None if int(state.iteration) == 0 else 0.0),
        frozen_screening_model=frozen_screening is not None,
        w_time_factor_cache=w_time_factor_cache,
        print_fn=inputs.print_fn,
    )
    sigma_wall = time.perf_counter() - sigma_wall_start
    if inputs.two_level_cost is not None:
        walls = inputs.two_level_cost.setdefault("sigma_walls_s", [])
        walls.append(float(sigma_wall))
        inputs.print_fn(
            f"[ SC cost | Sigma(omega) evaluation {len(walls)} | "
            f"wall={sigma_wall:.6f} s | "
            f"screening={'frozen' if frozen_screening is not None else 'refit'} ]")
    if bool(sigma_result.hartree_omitted) != (exact_hartree_dft is not None):
        raise RuntimeError(
            "SigmaResult Hartree-omission receipt disagrees with the "
            "density-SC direct-field owner")
    _check_sigma_stage(sigma_result, print_fn=inputs.print_fn)

    # Rotate (V_H + Σ_xc) back to DFT basis and form the *full* QSGW H
    # (as if every band were protected); the partition step below masks
    # off non-protected off-diagonals and overrides out-of-range
    # diagonals with the per-iteration scissor.
    # Σ_xc is genuinely built in the QP basis (from ``wfns_qp``) and must
    # be rotated back.  V_H is not: under density-SC it arrives already in
    # the DFT basis and adds to the pristine ``kin_ion_dft`` with no
    # rotation, which is the whole reason it is contracted with ψ_dft.
    # Under density-SC the direct field is caller-owned and SigmaResult uses
    # a scalar-zero sentinel rather than allocating a full zero band matrix.
    delta_h_qp = (
        sigma_result.sigma_xc_kij_ry
        if exact_hartree_dft is not None
        else sigma_result.v_h_kij_ry + sigma_result.sigma_xc_kij_ry)
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
        # MOVED, 2026-08-16: the star-spread enforcement used to run HERE,
        # on the raw Sigma+V_H, and that is a different object from the one
        # that ships.  ``apply_band_partition`` below zeroes every
        # off-diagonal outside protected x protected, so it is the LAST thing
        # that can break the star relation -- and until the partition was
        # promoted to whole multiplets it could, by treating two members of a
        # degenerate manifold differently.  Checking before it ran meant the
        # gate certified an object the loop then modified.  See the call after
        # ``apply_band_partition``.
        delta_h_qp = ks.select(delta_h_qp)
    delta_h_dft = _rotate_to_dft_basis(delta_h_qp, U_qp, mesh=inputs.mesh_xy)
    exact_hartree_terms = None
    if exact_hartree_dft is not None:
        scalar = exact_hartree_dft.scalar_dft
        transverse = exact_hartree_dft.transverse_dft
        if not ks.is_identity:
            scalar = ks.select(scalar)
            if transverse is not None:
                transverse = ks.select(transverse)
        exact_hartree_terms = (scalar, transverse)
        delta_h_dft = (
            _add_exact_scalar_hartree(delta_h_dft, scalar)
            if transverse is None else
            _add_exact_four_current_hartree(
                delta_h_dft, scalar, transverse))
    # Symmetric averaging is confined to accidental/exact degeneracies.  It
    # is intentionally NOT tied to ``degen_avg_tol_ry``: that general output
    # convention may be user-expanded, while QSGW state identity must never
    # erase a resolved SOC splitting to make a trajectory converge.  Average
    # the correction, not H itself, so the immutable DFT splitting survives.
    e_dft_map = inputs.e_dft_active_kn_ry
    if not ks.is_identity:
        e_dft_map = ks.select(e_dft_map)
    if not bool(getattr(inputs.config, "no_degen_averaging", False)):
        from .degen_average import average_matrix_diagonal
        delta_h_dft = average_matrix_diagonal(
            delta_h_dft,
            energies_kn_ry=np.asarray(e_dft_map, dtype=np.float64),
            tol_ry=(float(inputs.config.sc.exact_degeneracy_tol_ev)
                    / RYD_TO_EV),
            mesh_xy=inputs.mesh_xy,
        )
    H_qp_dft_full = inputs.kin_ion_dft + delta_h_dft
    if (buffer_mask.any()
            and inputs.config.sc.buffer_mode == "carry"
            and int(state.iteration) > 0):
        # Map zero earns a Sigma-derived reference for every buffer state.
        # Later maps carry that reference instead of snapping the states back
        # to DFT (or repeatedly reevaluating it).  Applying this on map zero
        # would merely freeze the buffer at DFT and would not test option (c).
        H_qp_dft_full = _carry_sc_buffer_diagonal(
            H_qp_dft_full, state.H_qp_dft,
            jnp.asarray(buffer_mask, dtype=bool))

    # ── THE UN-EXTRAPOLATED TWIN ────────────────────────────────────────
    # Present only when ``use_band_extrapolation`` drove this stage's Σ.
    # Assembled through the SAME three steps as the carry above — k-select,
    # rotate to the DFT basis with the SAME U, add the same density-SC V_H —
    # so the two Hamiltonians differ in exactly one thing: whether Σ_c
    # carries the extrapolated band tail.  It is diagonalized, reported and
    # dropped; it never reaches the carry, rCROP or any artifact.
    sigma_xc_unextrap = getattr(
        sigma_result, "sigma_xc_kij_ry_unextrap", None)
    if sigma_xc_unextrap is not None:
        delta_h_qp_n3 = (
            sigma_xc_unextrap
            if exact_hartree_dft is not None
            else sigma_result.v_h_kij_ry + sigma_xc_unextrap)
        if not ks.is_identity:
            delta_h_qp_n3 = ks.select(delta_h_qp_n3)
        delta_h_dft_n3 = _rotate_to_dft_basis(
            delta_h_qp_n3, U_qp, mesh=inputs.mesh_xy)
        if exact_hartree_terms is not None:
            scalar, transverse = exact_hartree_terms
            delta_h_dft_n3 = (
                _add_exact_scalar_hartree(delta_h_dft_n3, scalar)
                if transverse is None else
                _add_exact_four_current_hartree(
                    delta_h_dft_n3, scalar, transverse))
        if not bool(getattr(inputs.config, "no_degen_averaging", False)):
            delta_h_dft_n3 = average_matrix_diagonal(
                delta_h_dft_n3,
                energies_kn_ry=np.asarray(e_dft_map, dtype=np.float64),
                tol_ry=(float(inputs.config.sc.exact_degeneracy_tol_ev)
                        / RYD_TO_EV),
                mesh_xy=inputs.mesh_xy,
            )
        _report_extrapolation_eqp_shift(
            H_qp_dft_full, inputs.kin_ion_dft + delta_h_dft_n3,
            mesh_xy=inputs.mesh_xy, n_occ=n_occ,
            iteration=state.iteration, print_fn=inputs.print_fn)

    if state.iteration == 0:
        _residency_census(
            (("kin_ion_dft", inputs.kin_ion_dft),
             ("H_qp_dft (carry in)", state.H_qp_dft),
             ("U_qp", U_qp),
             ("U_full", U_full),
             ("sigma_xc_kij_ry", sigma_result.sigma_xc_kij_ry),
             ("sigma_x_kij_ry", sigma_result.sigma_x_kij_ry),
             ("v_h_kij_ry", sigma_result.v_h_kij_ry),
             ("V_H[SC] scalar", (
                 exact_hartree_dft.scalar_dft
                 if exact_hartree_dft is not None else None)),
             ("H_T[SC] transverse", (
                 exact_hartree_dft.transverse_dft
                 if exact_hartree_dft is not None else None)),
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

    # The same no-lag semicore/conduction policy is used by fixed-Sigma
    # EQP2.  ``ks`` is deliberately passed rather than spelling weights:
    # the fit is a reduction over k and must use the multiplicities of the
    # exact rows selected above.
    H_qp_dft_new, scissor_fit = _apply_scissor_partition_policy(
        H_qp_dft_full, e_dft_act, val_mask, partition, ks,
        efermi_dft_ry=float(inputs.efermi_dft_ry),
        n_occ=n_occ,
        candidate_efermi_fn=lambda H: _partitioned_candidate_efermi(
            H, inputs=inputs, kstar=ks, n_occ=n_occ,
            use_mp1=entry_occ_state is not None),
        band_classes=scissor_classes,
        scissor_fit=tail_fit,
        use_valence_fit=(inputs.config.sc.tail_fit == "buffer_edges"),
        label="SC", print_fn=inputs.print_fn)
    fixed_sigma_cycle_converged = True
    if inputs.two_level_enabled:
        fixed_exact_hartree = None
        if exact_hartree_terms is not None:
            scalar, transverse = exact_hartree_terms
            fixed_exact_hartree = SCExactHartree(
                scalar_dft=scalar, transverse_dft=transverse,
                efermi_ry=float(exact_hartree_dft.efermi_ry))
        fixed_cycle = run_fixed_sigma_evsc(
            sigma_result, inputs.kin_ion_dft,
            np.asarray(e_dft_act, dtype=np.float64),
            config=inputs.config, meta=inputs.meta,
            band_slices=inputs.band_slices, wfn=inputs.wfn,
            mesh_xy=inputs.mesh_xy,
            sc_context={
                "inputs": inputs,
                # The expensive Sigma evaluation above already produced the
                # first fixed-table map.  Start at that candidate and spend
                # only the additional cheap rotate/interpolate/eigh maps.
                "H_seed_dft_ry": H_qp_dft_new,
                "sigma_basis_U": U_full,
                "partition": partition,
                "band_classes": scissor_classes,
                "scissor_fit": tail_fit,
                "buffer_mask": buffer_mask,
                "occupation_state": entry_occ_state,
                "exact_hartree_dft": fixed_exact_hartree,
            },
            print_fn=inputs.print_fn,
        )
        H_qp_dft_new = fixed_cycle.H_qp_dft_ry
        fixed_sigma_cycle_converged = bool(fixed_cycle.converged)
        inputs.print_fn(
            "[ SC fixed table receipt | "
            f"maps={fixed_cycle.iterations}, "
            f"max|dE|={fixed_cycle.residual_ev * 1e3:.6f} meV, "
            f"converged={fixed_cycle.converged} ]")
    # THE STAR-SPREAD GATE, ON THE OBJECT THAT SHIPS.  It ran before the
    # partition until 2026-08-16, which certified a matrix the loop then
    # rewrote.  The partition is precisely the operation that could break the
    # star relation -- a protected mask whose edge fell inside a degenerate
    # multiplet gave one member off-diagonal Sigma and the other a scalar
    # scissor -- so it is the one thing the check most needed to be after.
    #
    # This is a STRENGTHENING and it may turn red on a deck that passed
    # before; that redness is correct and should be read as the partition
    # breaking symmetry, not as this gate misfiring.  The mask is now promoted
    # to whole multiplets at construction, which is what makes it pass.
    #
    # Full-BZ operand: the check needs every star member, and H_qp_dft_new is
    # on the loop's k-set, so it is unfolded through the same map that
    # reduced it.
    if not ks.is_identity:
        _check_kstar_spread(
            ks, ks.broadcast(H_qp_dft_new), print_fn=inputs.print_fn)

    # The occupation state CARRIED below is the ENTRY solve consumed by this
    # call's chi/head/Sigma.  The low-valence no-lag gate above also invokes
    # that solver on provisional/final output spectra, but those scalar
    # anchors are policy checks, not screening states and are deliberately
    # dropped.  The carry remains DIAGNOSTIC continuity only (mu drift between
    # consecutive map inputs) — the next call re-solves at its own entry,
    # whatever trajectory (linear, rCROP trial/accept) handed it its H.
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
    if (inputs.config.compute_mode is ComputeMode.MPA
            and frozen_screening is None):
        from .mpa.model import retain_iteration_artifacts
        retain_iteration_artifacts(
            os.path.join(inputs.input_dir, "tmp", "mpa"),
            f"sc_{state.iteration:04d}",
            print_fn=inputs.print_fn,
        )
        # AFTER this iteration's complete store cycle (allocate → fit →
        # finalize → head → Σ → retain), which is the point at which the
        # per-iteration churn audit A1 measures has actually happened.
        # Cheap: one /proc/self/maps read and a dict walk.  It stays quiet
        # while safe; an A1-unsafe condition reports the mapped-libhdf5
        # count and the written paths touched through both libraries.
        from file_io.hdf5_owner import probe as _hdf5_probe
        _hdf5_probe(f"sc_{state.iteration:04d}", print_fn=inputs.print_fn)

    return SCState(
        H_qp_dft=H_qp_dft_new,
        iteration=state.iteration + 1,
        partition=partition,
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
            screening=SCMapScreeningArtifacts(
                # Restart owns only static W.  Retaining the full role table
                # would keep a second large probe-frequency W alive for no
                # consumer in GN/HL-PPM.
                static_w=W_by_role.get("static"),
                iteration_head=iteration_head,
                static_head_terms=iteration_static_head_terms,
                sigma_model=W_by_role,
                w_time_factor_cache=w_time_factor_cache,
            ),
            exact_hartree_dft=exact_hartree_dft,
            fixed_sigma_cycle_converged=fixed_sigma_cycle_converged,
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
    *,
    band_classes=None,
    scissor_fit: ScissorFit | None = None,
    fermi_displacement_ry: float,
    use_valence_fit: bool = False,
    print_fn=None,
) -> tuple[jax.Array, object | None]:
    """Return ``E_QP_scissor[k, n]`` for use as the diagonal of bands
    that are out of the ω-grid range.

    Mechanism: use ``scissor_fit`` when supplied; otherwise take the
    diagonal of ``H_qp_dft_full`` (the candidate QP energies if the
    iteration kept all off-diagonals), restrict to in-range bands as the
    scissor's reference set, and fit α/β per val/cond.  The shared-fit
    form lets the active and sum-band sides of ``b3`` use the exact same
    law from the map input.  The conduction candidate remains
    ``E_QP = α_c·E_DFT + β_c``.  The valence fit is diagnostic only: every
    valence candidate is ``E_DFT + fermi_displacement_ry``, so its binding
    relative to the candidate Fermi level is unchanged.  The masking
    primitive will use these candidates only at out-of-range entries.

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
    from .scissor import k_star_weights, qsgw_out_of_range_energies

    e_dft_np = np.asarray(e_dft_kn_ry, dtype=np.float64)
    in_range = np.asarray(in_range_mask, dtype=bool)
    # Fast path: nothing to extrapolate.
    if bool(in_range.all()):
        return e_dft_kn_ry, None

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
    fit = scissor_fit
    if fit is None:
        H_diag_np = np.real(np.asarray(jnp.diagonal(
            H_qp_dft_full, axis1=1, axis2=2)))
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
        print_fn(f"    SC scissor: {fit.summary()}; valence regression is "
                 "diagnostic only")
    # The SAME boundary indices that split the fit split the application, so
    # a band cannot be fit as one class and extrapolated as another.  Crossing
    # bands stay at E_DFT.  In practice they are protected/in-range, but the
    # identity is the honest no-information fallback if one is not.
    if use_valence_fit:
        e_dft_ev = e_dft_np * RYD_TO_EV
        out_ev = e_dft_ev + fit.predict(
            e_dft_ev, valence_kn, crossing_mask=crossing_kn)
    else:
        out_ev = qsgw_out_of_range_energies(
            e_dft_np * RYD_TO_EV,
            fit,
            valence_kn,
            fermi_displacement_ev=float(fermi_displacement_ry) * RYD_TO_EV,
            crossing_mask_kn=crossing_kn,
        )
    return jnp.asarray(out_ev / RYD_TO_EV), fit


def _apply_scissor_partition_policy(
    H_qp_dft_full: jax.Array,
    e_dft_kn_ry,
    valence_mask_kn,
    partition: BandPartition,
    kstar,
    *,
    efermi_dft_ry: float,
    n_occ: int,
    candidate_efermi_fn: Callable[[jax.Array], float],
    band_classes=None,
    scissor_fit: ScissorFit | None = None,
    use_valence_fit: bool = False,
    label: str = "SC",
    print_fn=print,
) -> tuple[jax.Array, ScissorFit | None]:
    """Apply the shared semicore/conduction policy to one full H map.

    In-range states keep their Sigma-derived Hamiltonian.  Out-of-range
    conduction states take the supplied shared map-input fit when present,
    otherwise the historical fit derived from this map output.  The SC
    loop supplies the same fit already used by the sum-band tail, so moving
    ``b3`` cannot switch one physical band between two update laws.
    Out-of-range valence states preserve ``E - E_F`` using this map output's
    own Fermi level.  The provisional/final Fermi check makes that anchor
    non-circular.  Fixed-Sigma EQP2 keeps the output-fit form.
    """
    scissor_provisional_ry, scissor_fit = _scissor_E_qp_for_outofrange(
        H_qp_dft_full, e_dft_kn_ry, valence_mask_kn,
        partition.in_range_mask, kstar,
        band_classes=band_classes,
        scissor_fit=scissor_fit,
        fermi_displacement_ry=0.0,
        use_valence_fit=use_valence_fit,
        print_fn=print_fn,
    )
    scissor_E_qp_kn_ry = scissor_provisional_ry
    candidate_efermi_ry = None

    if scissor_fit is not None:
        in_range_np = np.asarray(
            partition.in_range_mask, dtype=bool).reshape(-1)
        if band_classes is not None and band_classes.n_crossing:
            frontier = np.arange(
                int(band_classes.valence_stop),
                int(band_classes.conduction_start))
        else:
            frontier = np.asarray(
                [max(0, n_occ - 1), min(n_occ, in_range_np.size - 1)],
                dtype=np.int64)
        bad_frontier = frontier[~in_range_np[frontier]]
        if bad_frontier.size:
            raise ValueError(
                f"{label} low-valence Fermi anchor requires the complete "
                "Fermi-crossing/frontier manifold inside the Sigma window; "
                f"out-of-range active band(s) {bad_frontier.tolist()} would "
                "make E_F(F(H)) depend on the scissor being anchored. Widen "
                "sigma_omega_min/max_ev (and preserve whole multiplets).")

        H_fermi_probe = apply_band_partition(
            H_qp_dft_full,
            protected_mask=partition.protected_mask,
            in_range_mask=partition.in_range_mask,
            scissor_E_qp_kn=scissor_provisional_ry,
        )
        candidate_efermi_ry = float(candidate_efermi_fn(H_fermi_probe))
        fermi_displacement_ry = (
            candidate_efermi_ry - float(efermi_dft_ry))

        from .scissor import qsgw_out_of_range_energies
        e_dft_np = np.asarray(e_dft_kn_ry, dtype=np.float64)
        valence_kn = np.asarray(valence_mask_kn, dtype=bool)
        crossing_kn = None
        if band_classes is not None:
            valence_kn, crossing_kn = band_classes.masks(e_dft_np.shape)
        if use_valence_fit:
            e_dft_ev = e_dft_np * RYD_TO_EV
            candidate_ev = e_dft_ev + scissor_fit.predict(
                e_dft_ev, valence_kn, crossing_mask=crossing_kn)
        else:
            candidate_ev = qsgw_out_of_range_energies(
                e_dft_np * RYD_TO_EV,
                scissor_fit,
                valence_kn,
                fermi_displacement_ev=(
                    fermi_displacement_ry * RYD_TO_EV),
                crossing_mask_kn=crossing_kn,
            )
        scissor_E_qp_kn_ry = jnp.asarray(candidate_ev / RYD_TO_EV)
        print_fn(
            f"    {label} low-valence anchor: "
            f"E_F(DFT)={float(efermi_dft_ry) * RYD_TO_EV:+.6f} eV, "
            f"E_F(F(H))={candidate_efermi_ry * RYD_TO_EV:+.6f} eV, "
            f"dE_F={fermi_displacement_ry * RYD_TO_EV:+.6f} eV")

    H_partitioned = apply_band_partition(
        H_qp_dft_full,
        protected_mask=partition.protected_mask,
        in_range_mask=partition.in_range_mask,
        scissor_E_qp_kn=scissor_E_qp_kn_ry,
    )
    if scissor_fit is not None:
        final_efermi_ry = float(candidate_efermi_fn(H_partitioned))
        if not np.isclose(
            final_efermi_ry, candidate_efermi_ry,
            rtol=0.0, atol=1.0e-10,
        ):
            raise ValueError(
                f"{label} low-valence Fermi anchor became circular: final "
                f"E_F={final_efermi_ry * RYD_TO_EV:+.9f} eV differs from "
                f"the provisional anchor "
                f"{candidate_efermi_ry * RYD_TO_EV:+.9f} eV by "
                f"{(final_efermi_ry - candidate_efermi_ry) * RYD_TO_EV:+.3e} "
                "eV. A scissored tail entered the frontier; widen the Sigma "
                "window rather than anchoring through it.")
    return H_partitioned, scissor_fit


def _refuse_empty_map_output(e_output_kn_ev: np.ndarray, *,
                             call_index: int, role: str) -> None:
    """Refuse a map-output column that is identically zero, or not a number.

    THE COLUMN THIS FILE DOCUMENTS IS ``eigvalsh(F(H_in))``.  A Hermitian
    QP Hamiltonian on a real deck has no all-zero spectrum: every band
    energy would have to be exactly 0 Ry at every k.  So an all-zero column
    is not a small answer, it is an ABSENT one — and it is indistinguishable
    on disk from a converged one.

    MEASURED 2026-08-15 on the sodium 48b one-shot metallic arms
    (`runs/Na/02_soc48b_qsgw_mpa/01_lorrax_metal_mpa/r4_np*/eqp0_iter0000.dat`,
    4 arms x 1392 rows, all four byte-identical apart from the timestamp):
    the column was written as zeros and the writer's own header diagnostic
    came out as `map_output_RMS_dE_prev_output = 2.467681671e+01 eV`, which
    is simply RMS|E_DFT| — the signature of subtracting zero.  The campaign's
    `r6_residual.py` computes `d = e_qp - prev` from exactly these files, so
    it would have read a zero residual and reported a FALSE converged from
    call 1 onward, and `r4_grid_floor.py` returned a `0.000000000e+00`
    "floor" from the same source.

    Non-finite is refused here too, and for the same reason as
    ``sanity.refuse_nonfinite``: an eqp snapshot is a shipped artifact.  The
    writer below (`eqp_bgw.write_bgw_eqp`) also refuses non-finite columns;
    this fires first so the message names the MAP CALL rather than the file.
    """
    arr = np.asarray(e_output_kn_ev, dtype=np.float64)
    where = f"SC map {int(call_index):04d} (role={role})"
    if arr.size == 0:
        raise ValueError(
            f"{where}: the map-output spectrum is EMPTY. "
            f"eqp0_iter{int(call_index):04d}.dat documents its second column "
            f"as eigvalsh(F(H_in)); there is nothing to write, and an empty "
            f"snapshot is read downstream as a zero residual.")
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"{where}: the map-output spectrum has {n_bad} non-finite "
            f"entries of {arr.size}.  Refusing to write "
            f"eqp0_iter{int(call_index):04d}.dat: a residual script reading "
            f"it cannot distinguish NaN from converged, and the exit code "
            f"will not tell you either.")
    if not np.any(arr):
        raise ValueError(
            f"{where}: the map-output spectrum is identically ZERO over all "
            f"{arr.shape} entries.  That is not a spectrum — eigvalsh of a "
            f"real QP Hamiltonian is not all-zero — so the map produced no "
            f"output and the snapshot is refused rather than written.  "
            f"MEASURED signature: the header's "
            f"map_output_RMS_dE_prev_output then comes back as RMS|E_DFT| "
            f"(24.68 eV on the sodium 48b one-shot metallic arm), and every "
            f"downstream residual reads zero, i.e. FALSE converged.")


def _record_sc(inputs: SCInputs, line: str) -> None:
    """Send one selected SC line to the production record."""
    record_fn = getattr(inputs, "record_fn", None)
    fallback = getattr(inputs, "print_fn", None)
    if record_fn is not None:
        record_fn(line)
    elif fallback is not None:
        fallback(line)


def _sc_z_factors(inputs: SCInputs, state_out: SCState) -> np.ndarray:
    """Return per-state Z for the exact dynamic Sigma built by one map.

    The Sigma cube and evaluation ladder are full-BZ quantities in the
    current QP basis.  Diagonal extraction is collective for a band-sharded
    cube, so every rank calls this before the snapshot writer's rank gate.
    Static modes have zero frequency derivative and therefore return one.
    """
    sigma = state_out.outputs.sigma_result
    e_eval = np.asarray(sigma.e_eval_ev, dtype=np.float64)
    cube = sigma.sigma_c_omega_kij_ry
    omega = sigma.omega_grid_ev
    if cube is None or omega is None:
        return np.ones_like(e_eval, dtype=np.float64)

    from .eqp_bgw import compute_z_factor_from_omega_grid
    from .qsgw_utils import extract_sigma_diag_replicated

    sigma_c_diag_ev = np.asarray(
        extract_sigma_diag_replicated(cube, inputs.mesh_xy),
        dtype=np.complex128,
    ) * RYD_TO_EV
    _, z_factor = compute_z_factor_from_omega_grid(
        sigma_c_omega_diag_ev=sigma_c_diag_ev,
        omega_rel_ev=np.asarray(omega, dtype=np.float64),
        e_dft_rel_ev=e_eval - float(sigma.efermi_dft_ev),
    )
    return np.asarray(z_factor, dtype=np.float64)


def _write_sc_z_snapshot(
    path: str,
    kpoints: np.ndarray,
    e_eval_ev: np.ndarray,
    z_factor: np.ndarray,
    *,
    band_offset: int,
    call_index: int,
    role: str,
) -> None:
    """Write one compact per-map Z table on the canonical file wedge."""
    if e_eval_ev.shape != z_factor.shape:
        raise ValueError(
            "SC Z snapshot shape mismatch: evaluation energies "
            f"{e_eval_ev.shape}, Z {z_factor.shape}")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(
            "# LORRAX SC Z factors; columns: ispin iband "
            "E_eval_eV Z\n")
        stream.write(f"# SC map={call_index:04d} role={role}\n")
        for ik, kpt in enumerate(np.asarray(kpoints, dtype=np.float64)):
            stream.write(
                f"{kpt[0]:15.9f} {kpt[1]:15.9f} {kpt[2]:15.9f} "
                f"{z_factor.shape[1]:7d}\n")
            for ib in range(z_factor.shape[1]):
                stream.write(
                    f"{1:8d} {band_offset + ib + 1:8d} "
                    f"{e_eval_ev[ik, ib]:16.9f} "
                    f"{z_factor[ik, ib]:16.9f}\n")


def _dump_sc_rotation(
    inputs: SCInputs,
    state_out: SCState,
    *,
    call_index: int,
) -> str | None:
    """Persist the exact DFT→QP rotation consumed by one SC map.

    ``sigma_basis_U`` is already the full-BZ rotation paired with this map's
    Sigma result.  The dump is opt-in through ``sc_dump_dir`` and is small for
    diagnostic windows.  Every rank joins the gather and the post-write
    barrier; only rank zero writes the ``.npy`` file.
    """
    dump_dir = inputs.config.sc.dump_dir
    if not dump_dir:
        return None

    from common.collectives import gather_to_host, process_rank

    rotation = gather_to_host(state_out.outputs.sigma_basis_U)
    path = os.path.join(
        dump_dir, f"rotation_iter{int(call_index):04d}.npy")
    if process_rank() == 0:
        os.makedirs(dump_dir, exist_ok=True)
        np.save(path, np.asarray(rotation, dtype=np.complex128))
    barrier(f"sc.rotation.{int(call_index):04d}.write",
            print_fn=inputs.print_fn)
    return path


def _band_ranges(mask, *, band_offset: int) -> str:
    """Format a one-dimensional band mask as compact 1-based ranges."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool)) + band_offset + 1
    if indices.size == 0:
        return "none"
    ranges = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ",".join(
        str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges)


def _write_sc_eqp_snapshot(
    inputs: SCInputs,
    state_out: SCState,
    e_output_kn_ev: np.ndarray,
    *,
    call_index: int,
    role: str,
    rms_ev: float,
    rms2_ev: float,
    prev_output_role: str,
    verdict: ConvergenceVerdict,
) -> str | None:
    """Write one small BGW-shaped record of a completed SC map call.

    The QP column is ``eigvalsh(F(H_in))``.  The active-band and sum-band
    scissor are the same input law used to build this map's chi/W/Sigma;
    recording both ranges makes the closure across ``b3`` auditable.
    rCROP trial outputs are useful diagnostics but are not accepted iterates.
    This is not a second implementation of BGW's final ``eqp0`` / ``eqp1``
    equations; those remain solely in :mod:`gw.eqp_bgw`.

    WHICH NUMBER IS THE RESIDUAL, AND WHICH IS NOT.  ``verdict`` is
    :func:`protected_band_convergence` on THIS call's output against THIS
    call's own input, over the non-scissored set — the fixed-point residual
    the driver actually stops on.  It is stamped, labelled as the criterion,
    with its max-abs beside its RMS.

    ``rms_ev`` / ``rms2_ev`` are the historical output-vs-previous-output
    diagnostics and they are stamped as such, now WITH the previous call's
    role.  That role is the whole reason the old stamp misled: under rCROP
    the preceding call alternates trial / accepted, and a trial step sits
    near its accepted neighbour by construction, so the number understates
    the accepted-iterate residual.  MEASURED 2026-08-14 by re-analysing an
    accepted MPA QSGW run's retained snapshots
    (`mpa_si_output_lifecycle_0814/run/qp_convergence/eqp0_iter0000..0004.dat`):
    the ledger figure **2.6571 meV** is a trial-to-accepted all-band RMS,
    while the last accepted-to-accepted step (0002 -> 0004) moved
    max|dE| = **50.87 meV** over the protected bands — a factor ~19, made of
    2.6x (RMS to max-abs) and 5.5x (trial pair to accepted pair).  That run
    was NOT converged at 2 meV, and the file said it was.  Both numbers are
    kept because both are real; only one of them is the criterion, and now
    the file says which.

    THE TWO WEDGES MEET HERE, and they are not the same size.  This writer
    is a ``.dat`` writer, so its rows are the **file wedge** —
    ``wfn.kpoints``, what BerkeleyGW means by the IBZ.  Under
    ``sc_on_ibz`` the loop runs on the **star wedge** (one row per orbit),
    which is SMALLER on two of the three committed decks: 4 vs 3 on
    ``cohsex_debug``, 9 vs 5 on ``gnppm_debug``, and 8 = 8 on
    ``si_cohsex_debug`` — the deck most gates run.  Handing the loop's rows
    straight to this writer was the ``e_qp shape (5, 46) does not match
    e_dft (9, 46)`` crash, and it would not have raised on Si.

    So the star-wedge operand goes back through the full BZ and is then
    reduced to the file wedge, by name, in that order.  Both hops are
    symmetry service calls; nothing here holds an index table.
    """
    from common.collectives import process_rank
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import (
        reduce_full_bz_to_file_wedge, unfold_star_wedge_to_full_bz)

    from .eqp_bgw import write_bgw_eqp

    # BEFORE the rank gate, on purpose.  ``e_output_kn_ev`` is a replicated
    # host array, so the check is bit-identical on every rank and costs a
    # reduction over (nk, nb); running it here makes the refusal COLLECTIVE
    # instead of a rank-0 exception that leaves P-1 peers in the next
    # collective.
    e_output = np.asarray(e_output_kn_ev, dtype=np.float64)
    _refuse_empty_map_output(e_output, call_index=call_index, role=role)
    # AND THE NON-FINITE REFUSAL, HERE, FOR THE SAME REASON.  8c59cb3d put
    # ``sanity.refuse_nonfinite`` inside ``write_bgw_eqp``, which this
    # function calls BELOW the rank gate three lines down -- so on a NaN
    # spectrum rank 0 would raise while P-1 peers walked into the next SC
    # collective, turning a clean refusal into a hang.  That is exactly the
    # hazard the paragraph above avoids for the empty-column check, and the
    # cost argument is the same: ``e_output`` is a replicated host array, so
    # the verdict is bit-identical on every rank.  The in-writer refusal
    # stays as defence in depth for the terminal writers (``gw_output`` at
    # end of run, where a rank-0 raise has no collective left to desert),
    # but in the SC loop it can no longer be the first to fire.
    from common import sanity
    sanity.refuse_nonfinite(
        f"SC map output E (call {int(call_index)}, role {role})", e_output,
        print_fn=inputs.print_fn,
        detail="this is the column the eqp snapshot reports as the map "
               "output, and its RMS stamp is the convergence criterion.")

    z_factor_full = _sc_z_factors(inputs, state_out)
    sanity.refuse_nonfinite(
        f"SC Z factor (call {int(call_index)}, role {role})", z_factor_full,
        print_fn=inputs.print_fn,
        detail="this is the renormalization table written beside the map "
               "spectrum; non-finite Z is a dynamic-Sigma pathology")

    rotation_path = _dump_sc_rotation(
        inputs, state_out, call_index=call_index)

    if process_rank() != 0:
        return None

    sym = inputs.sym

    def _to_file_wedge(a):
        return np.asarray(
            reduce_full_bz_to_file_wedge(sym, np.asarray(a)),
            dtype=np.float64)

    kpoints = _to_file_wedge(
        np.asarray(sym.unfolded_kpts, dtype=np.float64))
    e_dft = _to_file_wedge(
        np.asarray(inputs.e_dft_active_kn_ry, dtype=np.float64) * RYD_TO_EV)
    if getattr(inputs, "kstar", None) is None:
        e_output = _to_file_wedge(e_output)          # loop ran full-BZ
    else:
        e_output = _to_file_wedge(                    # loop ran star-wedge
            unfold_star_wedge_to_full_bz(sym, e_output))
    e_eval = _to_file_wedge(
        np.asarray(state_out.outputs.sigma_result.e_eval_ev,
                   dtype=np.float64))
    z_factor = _to_file_wedge(z_factor_full)

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
        # THE CRITERION, FIRST, AND NAMED AS SUCH.  Output against this
        # call's OWN input, over the non-scissored set: the pair the driver
        # stops on.  It is stamped max-abs-first because max-abs is the
        # test and the RMS is 2.6x smaller on a real run.
        f"map_fixedpoint_max_abs_dE_protected_ev="
        f"{float(verdict.max_abs_ev):.9e}; "
        f"map_fixedpoint_RMS_dE_protected_ev="
        f"{float(verdict.rms_protected_ev):.9e}; "
        f"n_protected={int(verdict.n_protected)} of {int(verdict.n_total)}; "
        f"cutoff_ev={float(verdict.cutoff_ev):.9e}; "
        f"converged={bool(verdict.converged)} "
        "(THIS is the convergence criterion: F(H_in) against H_in)",
        f"map_output_RMS_dE_prev_output={float(rms_ev):.9e} eV "
        f"(prev call role={prev_output_role}); "
        f"map_output_RMS_dE_two_calls={float(rms2_ev):.9e} eV; "
        "these are map-call diagnostics over ALL active bands, not "
        "accepted-iterate residuals — a trial neighbour understates them",
        f"active_scissored_bands_1based={active_labels}",
        ("active_scissor=none (all active bands lie on the Sigma grid)"
         if active_fit is None else
         "shared_map_input_active_scissor: " + active_fit.summary()),
        (f"sum_band_tail=[{tail_start + band_offset + 1},"
         f"{tail_stop + band_offset}] 1-based; shared_input_tail_scissor="
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
    z_path = os.path.join(
        inputs.input_dir, f"z_factor_iter{int(call_index):04d}.dat")
    _write_sc_z_snapshot(
        z_path, kpoints, e_eval, z_factor,
        band_offset=band_offset, call_index=int(call_index), role=role)

    n_occ = int(inputs.meta.nelec) - band_offset
    gap_ev = float(np.min(e_output[:, n_occ])
                   - np.max(e_output[:, n_occ - 1]))
    z_min = float(np.min(z_factor))
    z_max = float(np.max(z_factor))
    z_bad = int(np.count_nonzero((z_factor <= 0.0) | (z_factor > 1.0)))
    partition = _state_partition(state_out, inputs)
    protected = np.asarray(partition.protected_mask, dtype=bool)
    in_range = np.asarray(partition.in_range_mask, dtype=bool)
    # Keep the established map-artifact line schema intact: the canonical
    # convergence parser keys the following verdict to this exact line.  Z is
    # a sibling artifact, not a suffix that makes the map line unparsable.
    _record_sc(inputs, f"  SC map energies: {path}")
    _record_sc(inputs, f"  SC map Z factors: {z_path}")
    if rotation_path is not None:
        _record_sc(inputs, f"  SC map rotation: {rotation_path}")
    _record_sc(
        inputs,
        f"    SC iteration: call={int(call_index):04d} role={role} "
        f"gap={gap_ev:.9f} eV max|dE|={float(verdict.max_abs_ev):.9e} eV "
        f"Z=[{z_min:.9f},{z_max:.9f}] bad_Z={z_bad}/{z_factor.size} "
        f"active={band_offset + 1}-{band_offset + e_output.shape[1]} "
        f"protected={_band_ranges(protected, band_offset=band_offset)} "
        f"in_range={_band_ranges(in_range, band_offset=band_offset)}")
    return path


def _clear_sc_eqp_snapshots(input_dir: str, *, print_fn=print) -> None:
    """Remove only managed per-map text snapshots from an earlier run."""
    from .qsgw_utils import remove_managed

    removed = remove_managed(
        input_dir, r"(?:eqp0|z_factor)_iter[0-9]{4}\.dat\Z",
        barrier_tag="sc.eqp_snapshots.clear", print_fn=print_fn)
    if removed:
        print_fn(f"  SC map energies: cleared {len(removed)} stale snapshots")


def _clear_sc_rotation_snapshots(
    dump_dir: str | None,
    *,
    print_fn=print,
) -> None:
    """Remove only rotation snapshots managed by ``sc_dump_dir``."""
    if not dump_dir:
        return
    from .qsgw_utils import remove_managed

    removed = remove_managed(
        dump_dir, r"rotation_iter[0-9]{4}\.npy\Z",
        barrier_tag="sc.rotation_snapshots.clear", print_fn=print_fn)
    if removed:
        print_fn(f"  SC map rotations: cleared {len(removed)} stale snapshots")


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
    reset_diagnostics: bool = True,
    snapshot_offset: int = 0,
) -> tuple[SCState, list[float]]:
    """Run one fixed-screening-capable inner solve of ``gw_iteration_map``.

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

    ``max_iter`` bounds this inner solve.  The public two-level driver uses
    ``SCConfig.max_iter`` independently as both the number of outer W refits
    and each inner solve's cap; ``reset_diagnostics`` and ``snapshot_offset``
    let those solves share one non-overwriting per-map artifact sequence.
    Direct callers retain the historical defaults.
    """
    print_fn = inputs.print_fn
    if reset_diagnostics:
        _clear_sc_eqp_snapshots(inputs.input_dir, print_fn=print_fn)
    _, eigvalsh_kshard = _kshard_eigh_kernels(inputs.mesh_xy)
    # E-history dump dir from config.sc (LORRAX_SC_DUMP_DIR env is a
    # deprecated override, applied at config construction).
    _dump_dir = inputs.config.sc.dump_dir
    if reset_diagnostics:
        _clear_sc_rotation_snapshots(_dump_dir, print_fn=print_fn)

    # One-shot fast path: no acceleration needed.
    if max_iter == 1:
        state_new = gw_iteration_map(state_init, inputs)
        e_initial_ev = (
            np.asarray(eigvalsh_kshard(state_init.H_qp_dft)) * RYD_TO_EV)
        e_new_ev = (
            np.asarray(eigvalsh_kshard(state_new.H_qp_dft)) * RYD_TO_EV)
        rms = float(np.sqrt(np.mean((e_new_ev - e_initial_ev) ** 2)))
        verdict = protected_band_convergence(
            e_new_ev, e_initial_ev,
            np.asarray(_state_partition(state_new, inputs).protected_mask,
                       dtype=bool).reshape(-1),
            np.asarray(_state_partition(state_new, inputs).in_range_mask,
                       dtype=bool).reshape(-1),
            tol_ev)
        verdict = _include_fixed_table_verdict(state_new, verdict, inputs)
        state_new = replace(state_new, convergence_verdict=verdict)
        _write_sc_eqp_snapshot(
            inputs, state_new, e_new_ev,
            call_index=int(snapshot_offset), role="one_shot", rms_ev=rms,
            rms2_ev=float("nan"),
            # There is no previous map call; the "previous output" the RMS
            # is against is the DFT seed spectrum, and saying so is what
            # stops it being read as a convergence residual.
            prev_output_role="dft_seed",
            verdict=verdict,
        )
        _record_sc(inputs, f"    SC convergence: {verdict.summary()}")
        return state_new, []

    if accelerator == "rcrop":
        return _run_rcrop(
            state_init, inputs,
            max_iter=max_iter, tol_ev=tol_ev,
            history_depth=history_depth,
            eigvalsh_kshard=eigvalsh_kshard,
            print_fn=print_fn,
            dump_dir=_dump_dir,
            snapshot_offset=int(snapshot_offset),
        )
    if accelerator == "linear":
        return _run_linear_mixing(
            state_init, inputs,
            max_iter=max_iter, tol_ev=tol_ev, mixing=mixing,
            eigvalsh_kshard=eigvalsh_kshard,
            print_fn=print_fn,
            dump_dir=_dump_dir,
            snapshot_offset=int(snapshot_offset),
        )
    raise ValueError(
        f"run_self_consistency: unknown accelerator={accelerator!r} "
        f"(expected 'rcrop' or 'linear').")


def _run_linear_mixing(
    state_init: SCState, inputs: SCInputs, *,
    max_iter: int, tol_ev: float, mixing: float,
    eigvalsh_kshard, print_fn, dump_dir, snapshot_offset: int = 0,
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
    #: Map OUTPUTS (pre-mix candidates), which is what the eqp snapshots
    #: hold.  Separate from ``_e_history`` (accepted MIXED iterates) because
    #: under ``mixing != 1`` those are two different sequences and the file
    #: was stamped from the wrong one.
    _out_history: list[np.ndarray] = [E_prev_ev.copy()]
    if mixing != 1.0:
        print_fn(f"  SC mixing α = {mixing:.3f} (linear)")

    last_evaluated: SCState | None = None
    for it in range(max_iter):
        # DROP ITERATION i-1's SigmaResult BEFORE BUILDING ITERATION i's.
        # See the note in ``_run_rcrop.residual_fn``; the shape is the
        # same here — ``state`` is both the loop carry and the argument
        # to the map, so without this rebind both generations of the
        # ω-cube are live for the whole of ``gw_iteration_map``.  The
        # last completed map's payload is retained separately below.
        state = SCState(
            H_qp_dft=state.H_qp_dft,
            iteration=state.iteration,
            partition=state.partition,
            occupation_state=state.occupation_state,
            head_surface_weight_kn=state.head_surface_weight_kn,
        )
        map_input = state
        state_map = gw_iteration_map(map_input, inputs)
        E_candidate_ev = (
            np.asarray(eigvalsh_kshard(state_map.H_qp_dft)) * RYD_TO_EV)
        # Outputs, W and head all describe the MAP INPUT.  Record that exact
        # evaluated point before constructing an unevaluated mixed candidate.
        # This is the state returned on convergence or budget exhaustion.
        last_evaluated = SCState(
            H_qp_dft=map_input.H_qp_dft,
            iteration=state_map.iteration,
            partition=_state_partition(state_map, inputs),
            occupation_state=state_map.occupation_state,
            head_surface_weight_kn=state_map.head_surface_weight_kn,
            outputs=state_map.outputs,
        )
        if mixing != 1.0:
            H_next = (
                mixing * state_map.H_qp_dft
                + (1.0 - mixing) * map_input.H_qp_dft
            )
        else:
            H_next = state_map.H_qp_dft
        state_next = SCState(
            H_qp_dft=H_next,
            iteration=state_map.iteration,
            partition=_state_partition(state_map, inputs),
            occupation_state=state_map.occupation_state,
            head_surface_weight_kn=state_map.head_surface_weight_kn,
        )
        E_new_ev = np.asarray(eigvalsh_kshard(state_next.H_qp_dft)) * RYD_TO_EV
        rms = float(np.sqrt(np.mean((E_new_ev - E_prev_ev) ** 2)))
        rms_history.append(rms)
        _e_history.append(E_new_ev.copy())
        rms2 = (
            float(np.sqrt(np.mean((E_new_ev - _e_history[-3]) ** 2)))
            if len(_e_history) >= 3 else float("nan"))
        print_fn(
            f"  SC iter {state_map.iteration}: "
            f"RMS ΔE_{{k,k-1}} = {rms:.6f} eV, "
            f"ΔE_{{k,k-2}} = {rms2:.6f} eV"
        )
        # SAME CRITERION AS rCROP.  ``E_candidate_ev`` is the UNMIXED map
        # output F(H); ``E_prev_ev`` is that call's input.  This used to
        # break on ``rms < tol_ev`` -- an RMS over ALL active bands
        # including the scissored ones, both the looser test and a
        # different set.
        verdict = protected_band_convergence(
            E_candidate_ev, E_prev_ev,
            np.asarray(_state_partition(state_map, inputs).protected_mask,
                       dtype=bool),
            np.asarray(_state_partition(state_map, inputs).in_range_mask,
                       dtype=bool),
            tol_ev)
        verdict = _include_fixed_table_verdict(state_map, verdict, inputs)
        # THE SNAPSHOT'S STAMPS COME FROM THE MAP-OUTPUT HISTORY, NOT THE
        # MIXED ONE.  The column written is ``E_candidate_ev`` (pre-mix), so
        # a stamp computed from ``E_new_ev`` (post-mix) described a
        # different array than the file holds — wrong exactly when mixing is
        # on, which is exactly when someone is watching it.  ``_out_history``
        # holds map OUTPUTS, so ``map_output_RMS_dE_prev_output`` now means
        # what its name says on both accelerators.
        cand_rms = float(np.sqrt(np.mean(
            (E_candidate_ev - _out_history[-1]) ** 2)))
        _out_history.append(E_candidate_ev.copy())
        cand_rms2 = (
            float(np.sqrt(np.mean((E_candidate_ev - _out_history[-3]) ** 2)))
            if len(_out_history) >= 3 else float("nan"))
        _write_sc_eqp_snapshot(
            inputs, state_map, E_candidate_ev,
            call_index=int(snapshot_offset) + it, role="linear",
            rms_ev=cand_rms, rms2_ev=cand_rms2,
            prev_output_role=("dft_seed" if it == 0 else "linear"),
            verdict=verdict,
        )
        _record_sc(inputs, f"    SC convergence: {verdict.summary()}")
        last_evaluated = replace(
            last_evaluated, convergence_verdict=verdict)
        if verdict.converged:
            state = last_evaluated
            break
        state = state_next
        E_prev_ev = E_new_ev
        if it + 1 < max_iter:
            # ``state`` carries no outputs, but Python loop locals otherwise
            # retain the completed map and ``last_evaluated`` while the NEXT
            # Sigma/W is built.  Drop both large-payload owners now; the final
            # iteration keeps them for the return below.
            last_evaluated = None
            state_map = None

    _maybe_dump_e_history(dump_dir, _e_history, print_fn)
    if last_evaluated is None:
        raise RuntimeError("linear self-consistency completed no map calls")
    # At budget exhaustion ``state`` is the next mixed candidate and has not
    # been evaluated.  Final artifacts must instead use the last map input and
    # that input's exact Sigma/W/head payload.
    return last_evaluated, rms_history


def _run_rcrop(
    state_init: SCState, inputs: SCInputs, *,
    max_iter: int, tol_ev: float, history_depth: int,
    eigvalsh_kshard, print_fn, dump_dir, snapshot_offset: int = 0,
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
    # An nb that already divides pads by zero rows: ``pad_axis`` returns
    # the SAME array, so the production path is byte-identical to before.
    from runtime.padding import pad_axis, round_up, spec_divisor

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
        A = pad_axis(A, band_div, axis=1).array
        A = pad_axis(A, band_div, axis=2).array
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
    _last_input_H: list = [None]
    _last_verdict: list[ConvergenceVerdict | None] = [None]
    _occ_state: list = [state_init.occupation_state]
    _head_surface_weight: list = [state_init.head_surface_weight_kn]
    _iter_idx = [int(state_init.iteration)]
    _call_idx = [0]
    rms_history: list[float] = []
    _partition: list[BandPartition | None] = [state_init.partition]

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
        _last_input_H[0] = H
        E_in = np.asarray(eigvalsh_kshard(H)) * RYD_TO_EV
        # DROP ITERATION i-1's SigmaResult BEFORE BUILDING ITERATION i's.
        # ``gw_iteration_map`` reads ``state.iteration`` and
        # ``state.H_qp_dft`` and nothing else, so the previous result was
        # passed in and held in this cell for the whole call for no
        # reader.  Its ``sigma_c_omega_kij_ry`` is the largest object on
        # the SC path and, on the explicit ``sigma_omega_layout =
        # "replicated"`` control, does not shrink with P: 2751 MB/rank at
        # nb=512
        # (``gw_config.py``), so holding two generations was a
        # P-independent doubling of the peak.  Only the LAST one has a
        # consumer -- ``dump_sigma_omega_h5_final`` -- and it survives:
        # this cell is refilled below and the loop exits with it.
        _last_outputs[0] = None
        state_in = SCState(
            H_qp_dft=H,
            iteration=_iter_idx[0],
            partition=_partition[0],
            occupation_state=_occ_state[0],
            head_surface_weight_kn=_head_surface_weight[0],
        )
        state_out = gw_iteration_map(state_in, inputs)
        _last_outputs[0] = state_out.outputs
        _partition[0] = _state_partition(state_out, inputs)
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
            E_new, E_in,
            np.asarray(_state_partition(state_out, inputs).protected_mask,
                       dtype=bool),
            np.asarray(_state_partition(state_out, inputs).in_range_mask,
                       dtype=bool),
            tol_ev)
        _verdict = _include_fixed_table_verdict(
            state_out, _verdict, inputs)
        _last_verdict[0] = _verdict
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
        call_index = _call_idx[0]

        def _role_of(idx):
            return ("initial" if idx == 0 else
                    "trial" if idx % 2 else "accepted_input_map")

        role = _role_of(call_index)
        _write_sc_eqp_snapshot(
            inputs, state_out, E_new,
            call_index=int(snapshot_offset) + call_index,
            role=role, rms_ev=rms, rms2_ev=rms2,
            # NAMING THE PREVIOUS CALL'S ROLE IS THE FIX.  ``rms`` is
            # measured against ``_e_history[-1]``, i.e. the immediately
            # preceding MAP CALL, which under rCROP alternates trial and
            # accepted.  A trial step sits near its accepted neighbour by
            # construction, so this pair understates the accepted-iterate
            # residual (measured ~19x on the 2026-08-14 MPA QSGW run).  The
            # criterion is stamped beside it from ``_verdict``.
            prev_output_role=("dft_seed" if call_index == 0
                              else _role_of(call_index - 1)),
            verdict=_verdict,
        )
        _record_sc(inputs, f"    SC convergence: {_verdict.summary()}")
        _iter_idx[0] += 1
        _call_idx[0] += 1
        # Non-trial calls only: there the INPUT is the accepted iterate
        # (rcrop_nojit's ``f_new = residual_fn(x_new)``), so this is the
        # residual AT the iterate the loop would return.  A trial call's
        # residual is the residual at a probe point -- a diagnostic.
        if role != "trial" and _verdict.converged:
            raise _Converged(
                SCState(H_qp_dft=H,
                        iteration=_iter_idx[0],
                        partition=_state_partition(state_out, inputs),
                        occupation_state=state_out.occupation_state,
                        head_surface_weight_kn=state_out.head_surface_weight_kn,
                        outputs=state_out.outputs,
                        convergence_verdict=_verdict),
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
        # map INPUT that met it, NOT F(input) and not rCROP's stale internal
        # x: only this input was accepted and evaluated with the SCOutputs
        # retained for the writers.  The pad-inertness check below reads the
        # final x, which this path never reaches -- say so rather than
        # skip it silently.
        print_fn(
            f"  SC rCROP stopped by the convergence criterion after "
            f"{_call_idx[0]} map calls: {stop.verdict.summary()}")
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

    # Final state: use the last EVALUATED map input and its exact outputs.
    # ``result.x`` is a new mixed candidate that has not passed through the
    # GW map, so pairing it with ``_last_outputs`` would make the terminal
    # energy ladder and Sigma belong to different states.
    # Back to the REPLICATED carry layout: every consumer of the returned
    # state (``_scissor_E_qp_for_outofrange``, ``final_qp_eigenstates``,
    # the h5 writers) reads it back on the host.
    H_final = _last_input_H[0]
    if H_final is None or _last_verdict[0] is None:
        raise RuntimeError("rCROP completed without an evaluated SC map")
    state_final = SCState(
        H_qp_dft=H_final,
        iteration=_iter_idx[0],
        partition=_partition[0],
        occupation_state=_occ_state[0],
        head_surface_weight_kn=_head_surface_weight[0],
        outputs=_last_outputs[0],
        convergence_verdict=_last_verdict[0],
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


#: Floor below which a retained link's Löwdin-overlap singular value marks
#: its window edge as cutting a hybridized (non-separable) manifold —
#: PLAN.md D3(a).  Calibrated against the measured Na 8^3 SOC-48-band
#: collapse (proposal_1, Sec 1.2 item 2 / KNOWN_LORRAX_ISSUES.md, the
#: ``file_io/parallel_transport.py:407-415`` register row's sibling
#: finding): a manifold the transport holonomy independently flagged
#: (``max|S-1| = 1.9906``) carried retained singular values of ``5.462e-2``
#: at band 45 and ``4.476e-8`` at band 48, against a healthy link's
#: near-unitary overlap (near 1.0).  0.5 is a first, round, principled
#: floor — half the fidelity of a clean link — not yet independently
#: calibrated against a second material or system size; PLAN.md flags this
#: exact number as needing its own calibration run.
_LINK_HYBRIDIZATION_FLOOR: float = 0.5


def _refuse_unsupported_link_stencil(kgrid, *, where: str) -> None:
    """D3(c): per-axis stencil support, checked before ANY artifact read.

    Mirrors the producer-side gate
    (``file_io.parallel_transport.write_parallel_transport_artifact``) at
    driver-entry altitude, off the SAME threshold
    (``common.parallel_transport.undersampled_link_axes`` /
    ``MIN_STENCIL_POINTS`` — one name, not a second literal ``5`` here) —
    "before any allocation" (PLAN.md pipeline step 3): a deck whose links
    could never have been written is refused here, before this driver opens
    the artifact at all, rather than surfacing as a shape/refusal deep
    inside the read.  Never called for ``sc_head_update=dft_velocity``,
    which carries no stencil requirement on any axis (PLAN.md D2).
    """
    from common.parallel_transport import (MIN_STENCIL_POINTS,
                                           undersampled_link_axes)

    grid = tuple(int(n) for n in np.asarray(kgrid).reshape(3))
    bad_axes = undersampled_link_axes(grid)
    if not bad_axes:
        return
    raise ValueError(
        "GATE pt_head_stencil_unsupported: "
        f"{where} requires the nearest-neighbour link/fourth-order "
        "connection stencil along every Cartesian mesh direction.\n"
        f"  got:  kgrid={grid}, undersampled axes {', '.join(bad_axes)}\n"
        f"  want: >={MIN_STENCIL_POINTS} mesh points on every axis, or a "
        "head mode that needs no links\n"
        "  fix:  for a genuinely lower-dimensional deck (a collapsed axis "
        "with kgrid[i]=1, e.g. a slab), set sc_head_update=dft_velocity — "
        "it reads the exact DFT p-matrix velocity alone and needs no "
        "stencil on any axis; for an undersampled but periodic axis "
        "(1 < kgrid[i] < 5), densify the mesh along it\n"
        "  why:  the fourth-order +/-2 connection stencil differentiates "
        "along k; a direction with too few points cannot support it, and "
        "it is never fabricated as an analytic zero (KNOWN_LORRAX_ISSUES.md, "
        "file_io/parallel_transport.py row, fix note)\n"
        "  doc:  reports/metal_head_pt_pipelines_2026-08-23/PLAN.md, "
        "pipeline step 3(c)")


def _refuse_degenerate_window_edge(
    enk_full_ry: np.ndarray, nb_logical: int, *, where: str,
    trs_measured=None, mode: str = "strict",
) -> None:
    """D3(b): independent multiplet/TRIM degeneracy check on the head window.

    "Independent" of D3(a): pure DFT energies, no link/singular-value data —
    so, unlike D3(a), it applies to BOTH metal head modes.  Reuses
    :func:`common.band_degeneracy.check_band_window`, the SAME "minimum gap
    over k across a boundary" primitive this module already applies to the
    SC active window elsewhere (``BandPartition.report_multiplet_splits`` /
    ``promoted_to_multiplets``), rather than a second notion of "clean
    boundary" for the head window specifically — CLAUDE.md: single source of
    truth; rebuilding a symmetry/degeneracy primitive is banned.

    A Kramers pair under measured TRS is EXACTLY degenerate at its TRIM k,
    which is a special case of "the tightest gap this boundary has anywhere
    in the BZ" — exactly what ``boundary_min_gaps`` already computes over
    the FULL k-mesh.  There is no separate TRIM-only code path for that
    reason: a bespoke TRIM-point enumeration would be a second, narrower
    implementation of the same min-over-k question this reuse already
    answers everywhere, including at every TRIM.  ``trs_measured`` (PLAN.md:
    "TRS is measured, never assumed") is threaded through for the message
    only, naming why a near-zero gap here is expected rather than alarming.
    """
    from common.band_degeneracy import check_band_window

    e = np.asarray(enk_full_ry, dtype=np.float64)
    nb = int(nb_logical)
    if e.ndim != 2 or e.shape[1] <= nb:
        # No bands beyond the window are on hand, so the window's own top
        # edge cannot be told apart from a clean cut (band_degeneracy's own
        # ``boundary_min_gaps`` docstring: a window cannot see the cut that
        # produced it).  An honest scope limit, not a pass — callers should
        # prefer to pass the WFN's full loaded spectrum, not an
        # already-windowed slice, for exactly this reason.
        return
    trs_note = ("TRS measured to hold" if trs_measured
                else "TRS measured NOT to hold" if trs_measured is not None
                else "TRS not measured")
    check_band_window(
        e, 0, nb, mode=mode,
        where=f"{where}, active window top edge ({trs_note}; a Kramers "
              "pair is exactly degenerate at its TRIM k under measured "
              "TRS)")


def _refuse_hybridized_window_edge(
    singular_values: np.ndarray, nb_logical: int, *, where: str,
    floor: float = _LINK_HYBRIDIZATION_FLOOR,
) -> None:
    """D3(a): refuse a window whose edge cuts a hybridized manifold.

    Reads the link overlap singular values
    ``initialize_parallel_transport_artifact`` already writes and no
    consumer read before this fix (PLAN.md D3(a) —
    ``file_io.parallel_transport.SINGULAR_VALUES_DATASET``).  "Retained"
    means WITHIN the window: with ``nb_logical`` bands kept and the array
    descending along its last axis (the dataset's own ``ordering``
    attribute), the minimum retained singular value is exactly the LAST
    kept one, ``singular_values[..., :nb_logical].min()`` — no comparison to
    bands outside the window is needed or available (the artifact carries
    no bands beyond ``nb_logical``; see ``load_parallel_transport_head``'s
    ``band_stop == expected_nb`` gate).
    """
    sv = np.asarray(singular_values, dtype=np.float64)
    kept = sv[..., :int(nb_logical)]
    if kept.size == 0:
        return
    worst = float(np.min(kept))
    if worst > float(floor):
        return
    ik, idir, rank0 = (int(x) for x in np.unravel_index(
        int(np.argmin(kept)), kept.shape))
    direction = "xyz"[idir]
    raise ValueError(
        "GATE pt_head_window_hybridized: "
        f"{where}: the active window's retained link singular values dip "
        f"to {worst:.6e} (floor {float(floor):.3g}) — the window edge cuts "
        "a manifold the link construction cannot resolve as separable "
        "bands.\n"
        f"  got:  min retained singular value {worst:.6e} at source-k row "
        f"{ik}, direction {direction!r}, retained rank {rank0 + 1} of "
        f"{int(nb_logical)}\n"
        f"  want: every retained link singular value > {float(floor):.3g}\n"
        "  fix:  snap the active window outward (more bands) until the cut "
        "lands past the collapsing rank, or narrow it below that rank\n"
        "  why:  a near-zero link singular value means the Löwdin overlap "
        "between this band and its rotated k+b neighbour is far from "
        "unitary — the band mixes with a partner the window excludes, so "
        "the finite-link covariant derivative built across the cut is not "
        "the derivative of a well-defined single band\n"
        "  doc:  reports/metal_head_pt_pipelines_2026-08-23/PLAN.md, "
        "pipeline step 3(a)")


def load_head_velocity_source(
    config,
    input_dir: str,
    *,
    mesh,
    sym,
    wfn,
    meta,
    print_fn=print,
):
    """Resolve ``sc_head_update`` to the head's velocity source, or None.

    The ONE place the mode string turns into an object.  Both metal modes
    read the artifact ``get_dipole_mtxels --parallel-transport`` writes;
    they differ in how much of it they need:

    ``parallel_transport``
        the whole thing — nearest-neighbour links, the exact DFT velocity,
        and the completed velocity-identity validation, all through
        ``load_parallel_transport_head``.

    ``dft_velocity``
        the exact DFT p-matrix velocity stage ONLY, through
        ``load_dft_velocity_head``.  ``load_parallel_transport_head`` is
        not called, not imported, and not reachable on this path; the mode
        therefore has no finite-link derivative of Delta H.

    Returns None for ``off``, which preserves the fixed-DFT head exactly.

    THREE PREFLIGHT REFUSALS run here, before the expensive per-iteration
    head machinery ever sees this source (PLAN.md pipeline step 3 /
    ``reports/metal_head_pt_pipelines_2026-08-23/PLAN.md`` D3):

    (c) per-axis stencil support — ``parallel_transport`` only, before the
        artifact is even opened;
    (b) independent multiplet/TRIM degeneracy at the active window's top
        edge — both modes, pure DFT energies;
    (a) link singular-value hybridization at the active window's top edge —
        ``parallel_transport`` only, needs the links this mode alone reads.
    """
    from gw.gw_config import METAL_HEAD_UPDATES

    mode = str(config.sc.head_update)
    if mode not in METAL_HEAD_UPDATES:
        return None
    if not bool(config.do_G0):
        raise ValueError(
            f"sc_head_update={mode} requires do_G0=true; "
            "otherwise the rebuilt head has no consumer.")

    nb_active = int(meta.b_id_4_user)
    if not hasattr(sym, "trs_allowed"):
        raise ValueError(
            "GATE sc_head_needs_measured_trs: load_head_velocity_source "
            "requires SymMaps.trs_allowed; the supplied symmetry object "
            "has no verdict.")
    trs_measured = bool(sym.trs_allowed)
    _refuse_degenerate_window_edge(
        np.asarray(wfn.energies)[0], nb_active,
        where=f"sc_head_update={mode}", trs_measured=trs_measured)

    from file_io.paths import resolve_input_path

    pt_path = resolve_input_path(
        input_dir, config.paths.parallel_transport_file)
    if mode == "dft_velocity":
        from .qsgw_head import load_dft_velocity_head

        source = load_dft_velocity_head(
            pt_path, mesh=mesh, wfn=wfn, meta=meta)
        print_fn(
            "  SC head: loaded the exact DFT p-matrix velocity stage from "
            f"{pt_path} (nb={source.nb_logical}); no finite links, so "
            "the ΔH covariant velocity correction is OFF and the head runs "
            "on DFT velocities rotated into each iteration's QP basis "
            "(claim 0183 parks the covariant upgrade)")
        return source

    _refuse_unsupported_link_stencil(
        wfn.kgrid, where=f"sc_head_update={mode}")

    from .qsgw_head import load_parallel_transport_head

    source = load_parallel_transport_head(
        pt_path, mesh=mesh, sym=sym, wfn=wfn, meta=meta)
    _refuse_hybridized_window_edge(
        source.singular_values, source.nb_logical,
        where=f"sc_head_update={mode}")
    vgate = source.validation
    print_fn(
        "  SC head: loaded validated parallel transport from "
        f"{pt_path} (nb={source.nb_logical}, "
        "head-response rel="
        f"{vgate['head_response_relative_frobenius']:.3e}, "
        "DFT transition overlap="
        f"{vgate['transition_overlap_real']:+.6f}"
        f"{vgate['transition_overlap_imag']:+.2e}j; "
        f"full-matrix max_abs={vgate['max_abs']:.3e} diagnostic only)")
    return source


def _rotate_head_diagonal_to_dft(
    diag, U_dft_to_qp, *, mesh, weight_dft_qp=None,
):
    """Return the DFT-basis diagonal of a QP-diagonal head operator.

    If ``U[m,n] = <DFT_m|QP_n>``, then
    ``diag(U diag(h) U†)_m = sum_n |U[m,n]|^2 h_n``.  ``diag`` may carry
    no k axis, a k axis, or a leading frequency axis.  This avoids forming a
    dense matrix solely for an opt-in per-band diagnostic.
    """
    # Explicit replicated placement: a host ``jnp.asarray`` would create a
    # single-device array and then collide with the mesh-sharded U operand on
    # multi-process runs (the same seam documented by ``_place`` itself).
    d = _place(diag, mesh)
    weight = (jnp.real(U_dft_to_qp * jnp.conj(U_dft_to_qp))
              if weight_dft_qp is None else weight_dft_qp)
    if d.ndim == 1:
        out = jnp.einsum("kmn,n->km", weight, d)
        return _place(out, mesh)
    if d.ndim == 2:
        out = jnp.einsum("kmn,kn->km", weight, d)
        return _place(out, mesh)
    if d.ndim == 3:
        out = jnp.einsum("kmn,wkn->wkm", weight, d)
        return _place(out, mesh)
    raise ValueError(
        "head diagonal must be (nb,), (nk,nb), or (nomega,nk,nb); "
        f"got shape {tuple(d.shape)}")


def _rotate_static_head_terms_to_dft(
    head, U_dft_to_qp, *, mesh, weight_dft_qp,
):
    """Rotate every per-band static head contribution for final diagnostics."""
    if head is None:
        return None
    import dataclasses

    return dataclasses.replace(
        head,
        sigma_x_diag=_rotate_head_diagonal_to_dft(
            head.sigma_x_diag, U_dft_to_qp, mesh=mesh,
            weight_dft_qp=weight_dft_qp),
        sigma_sx_diag=_rotate_head_diagonal_to_dft(
            head.sigma_sx_diag, U_dft_to_qp, mesh=mesh,
            weight_dft_qp=weight_dft_qp),
        sigma_sx_minus_x_diag=_rotate_head_diagonal_to_dft(
            head.sigma_sx_minus_x_diag, U_dft_to_qp, mesh=mesh,
            weight_dft_qp=weight_dft_qp),
        sigma_coh_diag=_rotate_head_diagonal_to_dft(
            head.sigma_coh_diag, U_dft_to_qp, mesh=mesh,
            weight_dft_qp=weight_dft_qp),
        source=f"{head.source}; diagonal rotated QP->DFT",
    )


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
    screening_model_fn=None,
    config,
    meta,
    mesh_xy: Mesh,
    sym,
    wfn,
    centroid_indices,
    band_slices: BandSlices,
    input_dir: str,
    tensors_filename: str,
    enk_dft,
    material_class: str,
    print_fn: Callable = print,
    record_fn: Callable | None = None,
) -> SCDriverResult:
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
    static_head_terms_dft : StaticHeadTerms or None
        The LAST map's exact static head terms, with their diagonal rotated
        from that map's QP basis to the DFT basis used by the final debug
        table.  Built only when that table is enabled.  Restart persistence
        is completed inside this function so its W body and head samples
        cannot be separated at the caller boundary.
    """
    import dataclasses

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
        efermi_dft_scissor_ry = float(_mu_ry)
    else:
        efermi_ev = float(wfn.efermi) * RYD_TO_EV
        # The Sigma grid keeps its established loader reference above.  The
        # LOW-VALENCE DISPLACEMENT is a different invariant: it compares the
        # SC map's DFT and candidate Fermi levels, so both endpoints must use
        # the map's fixed-band-cut midgap convention.  Reuse its sole helper;
        # do not silently subtract a loader VBM from a candidate midgap.
        efermi_dft_scissor_ry = float(
            _midgap_efermi(e_dft_active_kn_ry, int(meta.nelec)))
    omega_min_ev = float(config.sigma.omega_min_ev) + efermi_ev
    omega_max_ev = float(config.sigma.omega_max_ev) + efermi_ev
    # One constructor owns the all-k window predicate and whole-multiplet
    # promotion.  EQP2 uses the same constructor below; neither ladder can
    # silently invent a different protected subspace.
    partition = build_omega_band_partition(
        e_dft_active_kn_ry, np.asarray(wfn.energies[0], dtype=np.float64),
        band_offset=int(band_slices.b0),
        omega_min_abs_ev=omega_min_ev,
        omega_max_abs_ev=omega_max_ev,
        label="SC", print_fn=print_fn)

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
        config, input_dir, mesh=mesh_xy, sym=sym, wfn=wfn, meta=meta,
        print_fn=print_fn)
    fixed_dft_head_response = None
    if (parallel_transport is None
            and config.head.correction is HeadCorrection.FULL):
        # ``sc_head_update=off`` freezes this direct DFT response.  Build it
        # once, on the same single-sourced frequency plan every map consumes,
        # then fold it through each iteration's resident W exactly once.
        from .qsgw_head import build_dft_head_response
        _, _, fixed_head_omegas = _sc_head_frequency_plan(
            config, quad, material_class=material_class)
        fixed_dft_head_response = build_dft_head_response(
            wfns, np.asarray(fixed_head_omegas, dtype=np.complex128),
            input_dir=input_dir, mesh=mesh_xy, wfn=wfn, meta=meta,
            config=config)
        print_fn(
            "  SC head: cached fixed DFT direct response and wings for "
            f"{len(fixed_head_omegas)} frequency sample(s); each map folds "
            "them once through its resident W.")

    inputs = SCInputs(
        wfns_dft=wfns, V_q=V_q, kin_ion_dft=kin_ion,
        head_channel=head_channel,
        quad=quad, e_ref=e_ref,
        static_head_terms=static_head_terms,
        head_resolver=head_resolver,
        screening_model_fn=screening_model_fn,
        config=config, meta=meta, mesh_xy=mesh_xy,
        sym=sym, wfn=wfn, centroid_indices=centroid_indices,
        band_slices=band_slices, input_dir=input_dir,
        partition=partition,
        e_dft_active_kn_ry=e_dft_active_kn_ry,
        valence_mask_active_kn=val_mask_active,
        # One SC convention at both endpoints: fixed-band midgap on an
        # insulator, the existing initial fixed-N _mu_ry on a metal.
        efermi_dft_ry=efermi_dft_scissor_ry,
        material_class=material_class,
        kstar=kstar,
        parallel_transport=parallel_transport,
        fixed_dft_head_response=fixed_dft_head_response,
        screening_seed_cache={},
        two_level_enabled=int(config.sc.max_iter) > 1,
        two_level_cost=(
            {"sigma_walls_s": [], "w_refits": 0}
            if int(config.sc.max_iter) > 1 else None),
        print_fn=print_fn,
        record_fn=record_fn,
    )
    state_init = make_initial_state_from_dft(inputs)
    # Loop knobs from ``config.sc`` (the LORRAX_SC_* env vars are
    # deprecated overrides, applied at config construction).
    sc = config.sc
    _record_sc(
        inputs,
        f"  SC: mode={config.compute_mode.value}, max_iter={sc.max_iter}, "
        f"tol={sc.tol_ev:.1e} eV, accel={sc.accelerator}, "
        f"exact_degeneracy_tol={sc.exact_degeneracy_tol_ev:.1e} eV, "
        f"tail_fit={sc.tail_fit}, buffer={sc.buffer_nbands}/edge, "
        f"buffer_mode={sc.buffer_mode}"
        + (f", depth={sc.history_depth}" if sc.accelerator == "rcrop"
           else f", α={sc.mixing:.2f}"))
    from .sc_two_level import run_two_level_self_consistency
    state_final, rms_history = run_two_level_self_consistency(
        state_init, inputs,
        max_iter=sc.max_iter, tol_ev=sc.tol_ev,
        accelerator=sc.accelerator,
        history_depth=sc.history_depth,
        mixing=sc.mixing,
    )
    verdict = state_final.convergence_verdict
    if verdict is None:
        raise RuntimeError(
            "GATE sc_missing_convergence_verdict: the last evaluated SC map "
            "returned no fixed-point verdict")
    if sc.max_iter == 1:
        _record_sc(
            inputs, "  SC verdict: ONE-MAP DIAGNOSTIC (convergence was not "
            f"requested); {verdict.summary()}")
    elif not verdict.converged:
        _record_sc(
            inputs, f"  SC verdict: NOT CONVERGED after {len(rms_history)} "
            f"GW map calls; {verdict.summary()}")
        raise RuntimeError(
            "GATE sc_fixed_point_not_converged: the SC iteration budget was "
            "exhausted without satisfying the protected-band fixed-point "
            f"criterion. {verdict.summary()} Per-map eqp0_iterNNNN.dat and "
            "z_factor_iterNNNN.dat diagnostics were retained; no terminal "
            "QP result is reported as converged.")
    else:
        _record_sc(
            inputs, f"  SC verdict: CONVERGED after {len(rms_history)} GW "
            f"map calls; {verdict.summary()}")
    sigma_result = state_final.outputs.sigma_result
    screening = state_final.outputs.screening
    requires_iteration_head = (
        config.head.correction is HeadCorrection.FULL
        or str(config.sc.head_update) != "off")
    if requires_iteration_head and screening.iteration_head is None:
        raise RuntimeError(
            "GATE sc_final_map_requires_iteration_head: "
            "screening.iteration_head got: None; want: accepted final-map "
            "head samples for "
            f"head_correction={config.head.correction.value!r}, "
            f"sc_head_update={config.sc.head_update!r}; why: final QSGW "
            "artifacts must carry the response from the accepted map.")
    requires_static_head_terms = (
        bool(config.do_G0)
        and config.head.correction is not HeadCorrection.OFF)
    if requires_static_head_terms and screening.static_head_terms is None:
        raise RuntimeError(
            "GATE sc_final_map_requires_static_head_terms: "
            "screening.static_head_terms got: None; want: static Sigma-head "
            f"terms for do_G0={bool(config.do_G0)!r}, "
            f"head_correction={config.head.correction.value!r}; why: the "
            "accepted final map must supply every requested Sigma term.")
    _record_sc(
        inputs,
        f"  SC done: {len(rms_history)} GW map calls"
        + (f", final RMS ΔE = {rms_history[-1]:.4e} eV"
            if rms_history else " (one-shot)"))
    if screening.static_head_terms is not None:
        from .head_correction import format_static_head_diagnostics
        print_fn("  SC final map: " + format_static_head_diagnostics(
            screening.static_head_terms))

    # ONE FINAL-MAP RESTART WRITE.  No seed W is persisted before the loop:
    # a failed SC run therefore cannot leave W0_ready=true, and a successful
    # one writes the static W body together with the head samples produced by
    # the exact same accepted map.  ``persist_w0_and_head`` consumes the
    # matching fixed-name pre-unfold capture when IBZ storage is selected.
    from .restart_q_storage import take_pre_unfold
    from .screening import driver_persists_w0
    try:
        if bool(config.do_screened) and driver_persists_w0(
                config.compute_mode, config):
            if screening.static_w is None:
                raise RuntimeError(
                    "GATE persist_sc_requires_final_static_w: "
                    "screening.static_w got: None; want: the accepted "
                    "final-map static W body; why: restart persistence must "
                    "not write head samples without their matching W.")
            from .gw_output import persist_w0_and_head
            with timing.section("gw_jax.persist_w0"):
                persist_w0_and_head(
                    screening.static_w,
                    tensors_filename=tensors_filename,
                    head_resolver=head_resolver,
                    iteration_head=screening.iteration_head,
                    config=config, meta=meta, mesh_xy=mesh_xy,
                    sym=sym, centroid_indices=centroid_indices,
                    print_fn=print_fn,
                )
    finally:
        # The writer consumes this on an IBZ restart write.  On disabled,
        # absent-file, or non-persisting paths the producer capture still
        # owns the large pre-unfold W wedge, so the final-map owner must tear
        # it down explicitly before post-SC artifacts are built.
        take_pre_unfold("W0_qmunu")
    # W0 is the only large object in the final-map payload.  Drop it before
    # WFN/sigma artifact construction; the tiny head/term provenance remains.
    screening = dataclasses.replace(
        screening, static_w=None, sigma_model=None,
        w_time_factor_cache=None)
    state_final = dataclasses.replace(
        state_final,
        outputs=dataclasses.replace(
            state_final.outputs,
            screening=screening,
        ),
    )

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
            output_dir=input_dir,
            qp_rotations_k_storage=config.qp_rotations_k_storage,
            print_fn=print_fn,
            clamp_tol=float(config.occupation_clamp_tol),
        )
        rotations_written = True
    sigma_omega_h5_path = dump_sigma_omega_h5_final(
        state_final, config=config, meta=meta, mesh_xy=mesh_xy,
        input_dir=input_dir, sym=sym,
        exact_hartree_dft=state_final.outputs.exact_hartree_dft,
        sigma_basis_U=state_final.outputs.sigma_basis_U,
        print_fn=print_fn,
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
    sigma_c_omega_dft = (
        _rotate_sigma_omega_cube(
            sigma_result.sigma_c_omega_kij_ry, U,
            mesh=mesh_xy, to_qp=False)
        if sigma_result.sigma_c_omega_kij_ry is not None else None)
    sigma_c_at_dft_dft = (
        _sigma_c_at_dft_diag_from_dft_cube(
            sigma_c_omega_dft, sigma_result,
            mesh=mesh_xy, print_fn=print_fn)
        if sigma_c_omega_dft is not None else None)
    # These are diagnostics only.  Avoid an extra U-sized |U|^2 temporary and
    # five distributed contractions on the default path where no debug table
    # consumes them; when enabled, build the weight once and share it.
    static_head_terms_dft = None
    head_sigma_diag_dft = None
    if bool(config.debug.sigma_freq_debug_output):
        head_weight_dft_qp = jnp.real(U * jnp.conj(U))
        static_head_terms_dft = _rotate_static_head_terms_to_dft(
            state_final.outputs.screening.static_head_terms, U, mesh=mesh_xy,
            weight_dft_qp=head_weight_dft_qp)
        if sigma_result.head_sigma_diag_w_kn_ry is not None:
            head_sigma_diag_dft = _rotate_head_diagonal_to_dft(
                sigma_result.head_sigma_diag_w_kn_ry, U, mesh=mesh_xy,
                weight_dft_qp=head_weight_dft_qp)
    exact_hartree_dft = state_final.outputs.exact_hartree_dft
    if exact_hartree_dft is None:
        sig_h = _rotate_to_dft_basis(
            sigma_result.v_h_kij_ry, U, mesh=mesh_xy)
        v_h_scalar = _rotate_to_dft_basis(
            sigma_result.v_h_scalar_kij_ry, U, mesh=mesh_xy)
        h_transverse = (
            _rotate_to_dft_basis(
                sigma_result.h_transverse_kij_ry, U, mesh=mesh_xy)
            if sigma_result.h_transverse_kij_ry is not None else None)
    else:
        # Already contracted with the unrotated DFT orbitals.  Rotating this
        # through the Sigma-basis U would be a second, erroneous basis change.
        v_h_scalar = exact_hartree_dft.scalar_dft
        h_transverse = exact_hartree_dft.transverse_dft
        sig_h = exact_hartree_dft.total
    sig_x = _rotate_to_dft_basis(sigma_result.sigma_x_kij_ry, U, mesh=mesh_xy)
    sigma_xc_dft = _rotate_to_dft_basis(
        sigma_result.sigma_xc_kij_ry, U, mesh=mesh_xy)
    sigma_total = sigma_xc_dft + sig_h
    sigma_result_dft = dataclasses.replace(
        sigma_result,
        v_h_kij_ry=sig_h,
        v_h_scalar_kij_ry=v_h_scalar,
        h_transverse_kij_ry=h_transverse,
        hartree_omitted=False,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_dft,
        sigma_c_omega_kij_ry=sigma_c_omega_dft,
        sigma_c_at_dft_diag_ev=sigma_c_at_dft_dft,
        sigma_sx_kij_ry=(
            _rotate_to_dft_basis(sigma_result.sigma_sx_kij_ry, U, mesh=mesh_xy)
            if sigma_result.sigma_sx_kij_ry is not None else None),
        sigma_coh_kij_ry=(
            _rotate_to_dft_basis(sigma_result.sigma_coh_kij_ry, U, mesh=mesh_xy)
            if sigma_result.sigma_coh_kij_ry is not None else None),
        sigma_lorentz_skij_ry=(
            jnp.stack([
                _rotate_to_dft_basis(
                    sigma_result.sigma_lorentz_skij_ry[sector], U,
                    mesh=mesh_xy)
                for sector in range(3)
            ]) if sigma_result.sigma_lorentz_skij_ry is not None else None),
        # The un-extrapolated N₃ twin travels with its partner or not at all.
        # It is None on every non-extrapolating run; when it is present,
        # leaving it in the QP basis while ``sigma_xc_kij_ry`` beside it is
        # rotated is exactly the silent-basis-mismatch this seam exists to
        # prevent — and it would show up as a "band-extrapolation correction"
        # that was mostly the basis difference.
        sigma_xc_kij_ry_unextrap=(
            _rotate_to_dft_basis(
                sigma_result.sigma_xc_kij_ry_unextrap, U, mesh=mesh_xy)
            if sigma_result.sigma_xc_kij_ry_unextrap is not None else None),
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
    return SCDriverResult(
        sigma_result_dft=sigma_result_dft,
        sigma_total_dft=sigma_total,
        rms_history_ev=rms_history,
        rotations_written=rotations_written,
        static_head_terms_dft=static_head_terms_dft,
        head_sigma_diag_dft_w_kn_ry=head_sigma_diag_dft,
    )


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


def _loop_arrays_on_full_bz(arrays, *, kstar, state_on_ibz: bool):
    """The loop's band-index arrays, on the FULL BZ, whichever wedge it ran on.

    ONE DIRECTION ONLY, and that is the point.  The loop runs either on the
    full BZ or on the STAR wedge (``KStarMap``, one row per orbit); the full
    BZ is the k-set every consumer here can be reached from, because the
    ``.dat``/WFN writers want the FILE wedge (``wfn.kpoints``) and the two
    wedges are DIFFERENT SIZES on two of the three committed decks.  Going
    star-wedge → full BZ → file wedge is the only route that is right on all
    of them; a direct star→file hop does not exist and must not be invented.

    ``kstar.broadcast`` only.  A hand-rolled gather is wrong on any
    TRS-reduced deck: Θ is antiunitary, so ``O(-k) = conj(O(k))`` and not
    ``O(k)`` (symmetry_maps.py:1740-1751); assuming equality is off by
    3.6e-01 relative, job 7889235.
    """
    if kstar is None or kstar.is_identity or not state_on_ibz:
        return [np.asarray(a) for a in arrays]
    return [np.asarray(kstar.broadcast(np.asarray(a))) for a in arrays]


def dump_sigma_omega_h5_final(
    state: SCState, *,
    config,
    meta,
    mesh_xy: Mesh,
    input_dir: str,
    sym=None,
    exact_hartree_dft: SCExactHartree | None = None,
    sigma_basis_U=None,
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
    if exact_hartree_dft is not None:
        if sigma_basis_U is None:
            raise ValueError(
                "dump_sigma_omega_h5_final: caller-owned exact Hartree "
                "requires the DFT-to-Sigma basis rotation")
        # The dynamic cube stays in the last map's QP basis.  The live direct
        # field was contracted directly in DFT basis, so rotate its two
        # components INTO that QP basis once for this file rather than
        # mixing bases or repeating the rotation on every SC iteration.
        U = _place(sigma_basis_U, mesh_xy, _band_rotation_spec())
        scalar_qp = _rotate_fixed_matrix(
            exact_hartree_dft.scalar_dft, U, mesh=mesh_xy, to_qp=True)
        transverse_qp = (
            _rotate_fixed_matrix(
                exact_hartree_dft.transverse_dft, U,
                mesh=mesh_xy, to_qp=True)
            if exact_hartree_dft.transverse_dft is not None else None)
        total_qp = (scalar_qp if transverse_qp is None
                    else scalar_qp + transverse_qp)
        import dataclasses
        sigma_result = dataclasses.replace(
            sigma_result,
            v_h_kij_ry=total_qp,
            v_h_scalar_kij_ry=scalar_qp,
            h_transverse_kij_ry=transverse_qp,
            hartree_omitted=False)
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
        v_h_scalar=sigma_result.v_h_scalar_kij_ry,
        h_transverse=sigma_result.h_transverse_kij_ry,
        config=config, input_dir=input_dir,
        meta=meta, mesh_xy=mesh_xy,
        # THE CONVERGED CUBE'S OWN ω REFERENCE, carried on the result the
        # cube came from.  Recomputing it here would be a second opinion
        # about the axis this file writes, and on the metal path the two
        # conventions differ by a measured 2.79 eV (audit A2).
        omega_reference_ev=sigma_result.efermi_dft_ev,
        omega_reference_provenance=sigma_result.omega_reference_provenance,
        # AND THE SPECTRUM IT WAS EVALUATED AT, carried on the same result
        # for the same reason.  Under self-consistency this is the converged
        # QP spectrum, NOT E_DFT, and the file used to say nothing -- so a
        # from-disk `eqp_bgw.make_eqp_bgw` reassembly of an SC run silently
        # re-centred eqp1's linearization at E_DFT, which is a different
        # calculation.  Relative to this cube's own omega reference, one
        # subtraction, the same one the finalize made.
        # Both halves or neither: the relative spectrum is meaningless
        # without the reference it is relative to, and a half-stamped file
        # would be worse than an unstamped one -- a reader cannot tell that
        # the zero is missing.
        eval_energies_rel_ev=(
            None if (sigma_result.e_eval_ev is None
                     or sigma_result.efermi_dft_ev is None)
            else np.asarray(sigma_result.e_eval_ev, dtype=np.float64)
            - float(sigma_result.efermi_dft_ev)),
        eval_energies_provenance="self_consistent_qp",
        omega_coverage=sigma_result.omega_coverage,
        sym=sym, print_fn=print_fn,
    )
    print_fn(f"  Σ_c(ω) tensor: {path}")
    # THE QSGW CUBE, WRITTEN WHERE IT IS STILL IN ITS OWN BASIS.  Under
    # self-consistency ``sigma_xc_kij_ry`` is Σ_x + Σ_c^QSGW built from
    # ``wfns_qp``, i.e. the QP basis — the same basis the Σ_c(ω) cube
    # above was written in.  ``run_sc_driver`` rotates its separate output
    # copy
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
    qp_rotations_k_storage: str = "auto",
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

    * ``write_qp_wfn_h5`` — the **FILE WEDGE**, ``wfn.kpoints``, checked
      at qp_wfn.py:136.  This writer copies the source file's
      ``kpoints``/``mtrx``/``tnp`` through unchanged, so its ``U`` must be
      the rotation of the STORED ψ at the STORED k, in the stored ORDER.
      ``reduce_full_bz_to_file_wedge`` is the definition of that k-set —
      ``kirr_fullids`` no longer reads the star labels, it matches
      ``wfn.kpoints`` against the full grid directly and raises if a stored
      k is not on it (fix/kirr-fullids-2026-08-08), so
      ``unfolded_kpts[kirr_fullids] == wfn.kpoints`` holds on every deck by
      construction.

      IT IS NOT ``KStarMap.select``, and the difference is the two wedges.
      ``select`` keeps one row per ORBIT — the STAR wedge.  MEASURED over
      every committed deck, 2026-08-15 (``file wedge`` / ``star wedge`` /
      ``wfn.nkpts``): si_cohsex_debug 8/8/8, si_bse_debug 8/8/8,
      hbn_cohsex_debug 18/18/18, **cohsex_debug 4/3/4**,
      **gnppm_debug 9/5/9**, **bispinor_debug 9/5/9**.

      What the size-matching this replaces actually did, deck by deck:
      where the two wedges coincide it picked ``select`` and was right;
      on gnppm/bispinor ``wfn.nkpts`` equals ``nk_tot``, so it picked the
      full BZ — right there too, because ``kirr_fullids`` measures as the
      identity ``[0..8]`` on those decks; and on ``cohsex_debug``,
      ``wfn.nkpts = 4`` is neither the star wedge (3) nor the full BZ (9),
      so it REFUSED a run that is perfectly well defined.  The named
      reduction is right on all six without a size argument.

      The remaining un-enforced property is separate and unchanged: that
      each stored k is reached from its orbit parent by the IDENTITY
      operation.  It holds on ``si_cohsex_debug`` and on mos2_4x4 and does
      NOT hold on the 3×3×1 decks (``sym_idx_k[kirr_fullids]`` measures
      ``[0, 12, 0, 0]`` on ``cohsex_debug``, 12 = pure time reversal).
      That claim still needs a probe on a new symmetry group; what no
      longer needs one is the k itself, or its order.
    * ``write_qp_rotations_h5`` — the FULL BZ.  Its ``kpoints_crys``
      labels the rows of ``U_mnk``; the canonical writer of this same
      file passes ``sym.unfolded_kpts`` there (gw_output.py:865-875) and
      the consumer indexes ``U_mnk`` by full-BZ index
      (postprocess/rotate_wfn_to_qp.py:159).  In an SC run this is the sole
      owner: the later generic result writer is told not to overwrite it
      from its slightly different post-Sigma eigensolve.

    ``state_on_ibz`` says which k-set the loop ran on (``config.sc_on_ibz``)
    and ``kstar`` is the map.  The loop's rows reach the full BZ through
    :func:`_loop_arrays_on_full_bz` and the file wedge through the service;
    there is no star-wedge → file-wedge hop, because there is no such
    operation.

    ``logical_band_stop`` is the unpadded end of the sum-band ladder.  It
    is required only when the final map used an energy-only tail scissor.

    Both files are rank-0-only writes (h5py is single-writer); a
    multihost barrier follows so the caller can rely on both files
    existing on every rank when this function returns.

    Returns ``(qp_wfn_path, qp_rotations_path, efermi_ry)``.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import reduce_full_bz_to_file_wedge

    from file_io.qp_wfn import write_qp_rotations_h5, write_qp_wfn_h5

    from psp.get_DFT_mtxels import spin_degeneracy_factor
    enk_loop_ry, U_loop, efermi_ry = final_qp_eigenstates(
        state, n_occ=n_occ, mesh_xy=mesh_xy,
        state_capacity=float(spin_degeneracy_factor(wfn)),
        clamp_tol=float(clamp_tol))
    enk_full_ry, U_full = _loop_arrays_on_full_bz(
        (enk_loop_ry, U_loop), kstar=kstar, state_on_ibz=state_on_ibz)
    # Full BZ → file wedge, by name.  One reduction, two arrays; the k
    # labels ``write_qp_wfn_h5`` writes are ``wfn.kpoints`` and this is the
    # selection that produces exactly those rows in exactly that order.
    enk_wfn_ry, U_wfn = (
        np.asarray(reduce_full_bz_to_file_wedge(sym, np.asarray(a)))
        for a in (enk_full_ry, U_full))
    nk_full = int(U_full.shape[0])
    nk_wfn_got, nk_wfn_want = int(U_wfn.shape[0]), int(wfn.nkpts)
    # State the two k-sets rather than letting a mismatch surface as a
    # shape error two frames down (that is how this was found: "U shape
    # (16, 128, 128) inconsistent with (nk=10, nb_active=128)").
    if (nk_wfn_got != nk_wfn_want
            or nk_full != int(sym.unfolded_kpts.shape[0])):
        raise ValueError(
            f"dump_qp_wfn_artifacts: k-set placement failed — loop nk="
            f"{int(U_loop.shape[0])} (on_ibz={state_on_ibz}) gave "
            f"WFN_qp nk={nk_wfn_got} (need wfn.nkpts={nk_wfn_want}) and "
            f"rotations nk={nk_full} (need full BZ "
            f"{int(sym.unfolded_kpts.shape[0])}); kstar={kstar!r}")
    print_fn(f"  QP dump k-sets: WFN_qp {nk_wfn_got} (file wedge), "
             f"rotations {nk_full} (full BZ), "
             f"loop {int(U_loop.shape[0])}"
             f"{' (star wedge)' if state_on_ibz else ' (full BZ)'}")
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
        # The tables come from the SERVICE's own accessor, never re-spelled
        # here: ``n_sym_spatial`` is derived from ``sym_mats_k`` rather than
        # from the WFN header, and that derivation is the one the unfold
        # side uses to decide which rows get conjugated.
        from ffi import _services as _svc
        _svc.ensure_on_path()
        import symmetry_maps as _sm
        write_qp_rotations_h5(
            qp_rot_path,
            U_mnk=U_full,
            E_qp_nk=enk_full_ry * 0.5,                     # Ry → Hartree
            band_start=band_slices.b0, band_stop=band_slices.b3,
            kpoints_crys=np.asarray(sym.unfolded_kpts, dtype=np.float64),
            nkx=int(kgrid[0]), nky=int(kgrid[1]), nkz=int(kgrid[2]),
            kpoints_reduced=np.asarray(wfn.kpoints, dtype=np.float64),
            kirr_to_kfull=np.asarray(sym.kirr_fullids, dtype=np.int32),
            k_storage=str(qp_rotations_k_storage),
            star_tables=_sm.star_tables_of(sym),
            source_wfn=wfn,
            print_fn=print_fn,
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
    "FixedSigmaEVSCResult",
    "SCInputs",
    "SCOutputs",
    "SCState",
    "gw_iteration_map",
    "load_head_velocity_source",
    "make_initial_state_from_dft",
    "run_self_consistency",
    "run_sc_driver",
    "run_fixed_sigma_evsc",
    "final_qp_eigenstates",
    "dump_qp_wfn_artifacts",
    "dump_sigma_omega_h5_final",
]
