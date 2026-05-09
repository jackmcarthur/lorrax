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
ψ + DFT energies throughout SC iteration; their QP corrections come
from the scissor extrapolation downstream (see :mod:`gw.scissor`).

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

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from .band_partition import BandPartition, apply_band_partition
from .gw_config import ComputeMode
from .scissor import fit_scissor
from .sigma_dispatch import SigmaResult, compute_sigma_xc
from .wavefunction_bundle import (
    BandSlices, Wavefunctions, rotate_wavefunctions)


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
    quad: object             # static minimax quadrature for χ₀
    e_ref: float
    static_head_terms: object | None
    head_resolver: object
    config: object
    meta: object
    mesh_xy: Mesh
    sym: object
    wfn: object              # WFNReader (for vbm/efermi anchor + paths)
    band_slices: BandSlices
    input_dir: str
    partition: BandPartition
    e_dft_active_kn_ry: jax.Array      # (nk, nb_active) DFT energies for scissor fit
    valence_mask_active_kn: jax.Array  # (nk, nb_active) bool — for scissor val/cond split
    print_fn: Callable = print


@dataclass(frozen=True)
class SCState:
    """State carried across self-consistent iterations.

    The iteration "carry" is **just** ``H_qp_dft_active`` — the QP
    Hamiltonian on the protected active subspace
    (``slices.sigma`` of the wfn bundle), in the original DFT basis.
    Everything else (``E_qp``, ``U_dft_to_qp``, ``efermi``) is derivable
    by the next iteration's first step (``vmap(eigh)``), so we don't
    carry redundant state — that would let convergence checks read
    inconsistent (E, H) pairs if anyone forgot to keep them in sync.

    ``last_sigma_result`` is purely for the final output writer (eqp.dat,
    sigma_diag.dat, freq_debug.dat); it does not feed the next iteration.
    """

    H_qp_dft: jax.Array              # (nk, nb_active, nb_active) Ry, DFT basis
    iteration: int
    last_sigma_result: SigmaResult | None = None


# ---------------------------------------------------------------------------
# Initial state from DFT
# ---------------------------------------------------------------------------

def make_initial_state_from_dft(inputs: SCInputs) -> SCState:
    """``H_qp_dft^(0) = diag(E_DFT)`` on the active subspace.

    Iteration 1's ``eigh`` of a diagonal matrix returns ``(E_DFT, U=I)``
    so the first Σ-pipeline call uses the unrotated DFT wfns and "one
    iteration of QSGW" reduces exactly to one-shot G0W0 at E=E_DFT.
    """
    from common.load_wfns import get_enk_bandrange
    enk_dft, _ = get_enk_bandrange(
        inputs.wfn, inputs.sym,
        inputs.band_slices.sigma_range, inputs.band_slices.sigma_range,
        nspinor=inputs.meta.nspinor)
    enk_dft_ry = np.asarray(enk_dft, dtype=np.float64)
    nk, nb_active = enk_dft_ry.shape
    H0 = np.zeros((nk, nb_active, nb_active), dtype=np.complex128)
    idx = np.arange(nb_active)
    for k in range(nk):
        H0[k, idx, idx] = enk_dft_ry[k]
    rep = NamedSharding(inputs.mesh_xy, P(None, None, None))
    return SCState(
        H_qp_dft=jax.device_put(jnp.asarray(H0), rep),
        iteration=0,
    )


# ---------------------------------------------------------------------------
# Iteration map
# ---------------------------------------------------------------------------

def _make_kshard_eigh(mesh_xy: Mesh, *, eigvalsh_only: bool):
    """Return a jit'd eigh that briefly k-shards the input over the mesh
    so each device only does its slice of the per-k diagonalisations,
    then allgathers the eigenvalues (and U if requested) back to
    replicated.  Pure perf hint — the math is identical to running
    ``vmap(eigh)`` on the replicated input.

    ``mesh_xy.size`` must divide ``nk``; otherwise the resharding fails.
    """
    rep_H = NamedSharding(mesh_xy, P(None, None, None))
    rep_E = NamedSharding(mesh_xy, P(None, None))
    rep_U = NamedSharding(mesh_xy, P(None, None, None))
    k_shard_3d = NamedSharding(mesh_xy, P(('x', 'y'), None, None))

    if eigvalsh_only:
        @jax.jit
        def _f(H):
            H_k = jax.lax.with_sharding_constraint(H, k_shard_3d)
            H_h = 0.5 * (H_k + jnp.conj(jnp.swapaxes(H_k, -1, -2)))
            E = jax.vmap(jnp.linalg.eigvalsh)(H_h)
            return jax.lax.with_sharding_constraint(E, rep_E)
        return _f
    else:
        @jax.jit
        def _f(H):
            H_k = jax.lax.with_sharding_constraint(H, k_shard_3d)
            H_h = 0.5 * (H_k + jnp.conj(jnp.swapaxes(H_k, -1, -2)))
            E, U = jax.vmap(jnp.linalg.eigh)(H_h)
            E = jax.lax.with_sharding_constraint(E, rep_E)
            U = jax.lax.with_sharding_constraint(U, rep_U)
            return E, U
        return _f


# Kernel cache: one (eigh, eigvalsh) pair per mesh.  Re-used across all
# SC iterations so the JIT cost is paid once.
_KSHARD_EIGH_CACHE: dict[int, tuple] = {}


def _kshard_eigh_kernels(mesh_xy: Mesh) -> tuple:
    key = id(mesh_xy)
    pair = _KSHARD_EIGH_CACHE.get(key)
    if pair is None:
        pair = (
            _make_kshard_eigh(mesh_xy, eigvalsh_only=False),
            _make_kshard_eigh(mesh_xy, eigvalsh_only=True),
        )
        _KSHARD_EIGH_CACHE[key] = pair
    return pair


def _diagonalize_and_get_efermi(
    H: jax.Array, n_occ: int, mesh_xy: Mesh,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Hermitise + eigh + midgap E_F.  Returns (E, U, efermi_ry).

    Per-k eighs are briefly k-sharded over the device mesh so each
    device only does ``nk / mesh_size`` of them.  The midgap reduction
    runs on the gathered E (small, replicated).
    """
    eigh_kshard, _ = _kshard_eigh_kernels(mesh_xy)
    E, U = eigh_kshard(H)
    vbm = jnp.max(E[:, :n_occ])
    cbm = jnp.where(n_occ < E.shape[1], jnp.min(E[:, n_occ:]), vbm)
    efermi = 0.5 * (vbm + cbm)
    return E, U, efermi


@jax.jit
def _rotate_to_dft_basis(O_qp: jax.Array, U: jax.Array) -> jax.Array:
    """``O_DFT[m, n] = Σ_pq U[m, p] · O_QP[p, q] · U[n, q]^*`` per k."""
    return jnp.einsum('kmp,kpq,knq->kmn', U, O_qp, jnp.conj(U), optimize=True)


def gw_iteration_map(state: SCState, inputs: SCInputs) -> SCState:
    """One self-consistent QSGW step in the DFT basis.

    Pure function — no side effects on ``inputs.wfns_dft``.  All
    derived quantities (E_qp, U_qp, efermi) are recomputed each call;
    the only carried state is ``H_qp_dft`` on the active subspace.
    """
    from .w_isdf import compute_chi0, solve_w

    n_occ = int(inputs.meta.nelec)
    E_qp_ry, U_qp, efermi_ry = _diagonalize_and_get_efermi(
        state.H_qp_dft, n_occ, inputs.mesh_xy)

    # Rotate the active subspace of the DFT bundle to this iteration's QP
    # basis.  Bands outside ``slices.sigma`` keep their DFT ψ + DFT energy
    # (their QP corrections come from the scissor extrapolation downstream).
    wfns_qp = rotate_wavefunctions(
        inputs.wfns_dft, U_qp,
        enk_active_new=E_qp_ry, efermi=float(efermi_ry),
        mesh_xy=inputs.mesh_xy,
        active_slice=inputs.band_slices.sigma,
    )

    # Re-solve W using the rotated wfns (cached jits dispatch with new
    # values; XLA cache hit on iteration ≥ 2).
    if inputs.config.do_screened:
        chi0_q = compute_chi0(
            wfns_qp, inputs.quad, inputs.meta, inputs.mesh_xy,
            energy_reference=inputs.e_ref)
        W_q = solve_w(inputs.V_q, chi0_q, inputs.meta, inputs.mesh_xy,
                      solver=inputs.config.backend.screening_solver)
        del chi0_q
    else:
        W_q = inputs.V_q

    # Σ_xc dispatch — mode-orthogonal.
    sigma_result = compute_sigma_xc(
        inputs.config.compute_mode,
        wfns=wfns_qp, V_q=inputs.V_q, W_q=W_q,
        e_qp_ev=np.asarray(E_qp_ry) * RYD_TO_EV,
        static_head_terms=inputs.static_head_terms,
        head_resolver=inputs.head_resolver,
        quad=inputs.quad, e_ref=inputs.e_ref,
        config=inputs.config, meta=inputs.meta, mesh_xy=inputs.mesh_xy,
        sym=inputs.sym, wfn=inputs.wfn,
        band_slices=inputs.band_slices,
        input_dir=inputs.input_dir,
        print_fn=inputs.print_fn,
    )

    # Rotate (V_H + Σ_xc) back to DFT basis and form the *full* QSGW H
    # (as if every band were protected); the partition step below masks
    # off non-protected off-diagonals and overrides out-of-range
    # diagonals with the per-iteration scissor.
    delta_h_qp = sigma_result.v_h_kij_ry + sigma_result.sigma_xc_kij_ry
    delta_h_dft = _rotate_to_dft_basis(delta_h_qp, U_qp)
    H_qp_dft_full = inputs.kin_ion_dft + delta_h_dft
    scissor_E_qp_kn_ry = _scissor_E_qp_for_outofrange(
        H_qp_dft_full, inputs.e_dft_active_kn_ry,
        inputs.valence_mask_active_kn, inputs.partition,
    )
    H_qp_dft_new = apply_band_partition(
        H_qp_dft_full,
        protected_mask=inputs.partition.protected_mask,
        in_range_mask=inputs.partition.in_range_mask,
        scissor_E_qp_kn=scissor_E_qp_kn_ry,
    )

    return SCState(
        H_qp_dft=H_qp_dft_new,
        iteration=state.iteration + 1,
        last_sigma_result=sigma_result,
    )


# ---------------------------------------------------------------------------
# Per-iteration scissor refit for non-protected out-of-range bands
# ---------------------------------------------------------------------------

def _scissor_E_qp_for_outofrange(
    H_qp_dft_full: jax.Array,
    e_dft_kn_ry: jax.Array,
    valence_mask_kn: jax.Array,
    partition: BandPartition,
) -> jax.Array:
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
    """
    e_dft_np = np.asarray(e_dft_kn_ry, dtype=np.float64)
    in_range = np.asarray(partition.in_range_mask, dtype=bool)
    # Fast path: nothing to extrapolate.
    if bool(in_range.all()):
        return e_dft_kn_ry

    H_diag_np = np.real(np.asarray(jnp.diagonal(
        H_qp_dft_full, axis1=1, axis2=2)))
    in_range_kn = np.broadcast_to(
        in_range[None, :], e_dft_np.shape).astype(bool)
    fit = fit_scissor(
        e_dft_np * RYD_TO_EV,
        H_diag_np * RYD_TO_EV,
        valence_mask_kn=np.asarray(valence_mask_kn, dtype=bool),
        fit_mask_kn=in_range_kn,
    )
    # ΔE = (α − 1) · E + β; E_QP = E_DFT + ΔE.
    delta_ev = fit.predict(
        e_dft_np * RYD_TO_EV, np.asarray(valence_mask_kn, dtype=bool))
    return jnp.asarray((e_dft_np + delta_ev / RYD_TO_EV))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_self_consistency(
    state_init: SCState,
    inputs: SCInputs,
    *,
    max_iter: int = 1,
    tol_ev: float = 1.0e-4,
) -> tuple[SCState, list[float]]:
    """Iterate ``gw_iteration_map`` until ``max_iter`` or RMS ΔE < ``tol_ev``.

    The iteration carry holds only ``H_qp_dft``; convergence is judged
    on the **eigenvalues** of consecutive H matrices (recomputed each
    iteration via the same k-sharded eigvalsh kernel as the main map)
    so the carry never gets out of sync with a separately-tracked E.

    Returns
    -------
    state_final
        Last :class:`SCState` produced.
    rms_history
        RMS ΔE_n (eV) at each iteration ≥ 1; empty list when
        ``max_iter == 1`` (one-shot G0W0).
    """
    print_fn = inputs.print_fn
    _, eigvalsh_kshard = _kshard_eigh_kernels(inputs.mesh_xy)
    state = state_init
    rms_history: list[float] = []
    E_prev_ev = np.asarray(eigvalsh_kshard(state.H_qp_dft)) * RYD_TO_EV

    for it in range(max_iter):
        state_new = gw_iteration_map(state, inputs)
        if it == 0 and max_iter == 1:
            return state_new, rms_history
        E_new_ev = np.asarray(eigvalsh_kshard(state_new.H_qp_dft)) * RYD_TO_EV
        rms = float(np.sqrt(np.mean((E_new_ev - E_prev_ev) ** 2)))
        rms_history.append(rms)
        print_fn(f"  SC iter {state_new.iteration}: RMS ΔE = {rms:.6f} eV")
        state = state_new
        E_prev_ev = E_new_ev
        if rms < tol_ev:
            break

    return state, rms_history


__all__ = [
    "SCInputs",
    "SCState",
    "gw_iteration_map",
    "make_initial_state_from_dft",
    "run_self_consistency",
]
