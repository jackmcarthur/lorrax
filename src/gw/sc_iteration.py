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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.units import RYD_TO_EV
from .gw_config import ComputeMode
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

@jax.jit
def _diagonalize_and_get_efermi(H: jax.Array, n_occ: int) -> tuple[
    jax.Array, jax.Array, jax.Array]:
    """Hermitise + eigh; return (E, U, efermi_ry)."""
    H_herm = 0.5 * (H + jnp.conj(jnp.swapaxes(H, -1, -2)))
    E, U = jax.vmap(jnp.linalg.eigh)(H_herm)
    vbm = jnp.max(E[:, :n_occ])
    cbm = jnp.where(n_occ < E.shape[1],
                    jnp.min(E[:, n_occ:]), vbm)
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
        state.H_qp_dft, n_occ)

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

    # Rotate (V_H + Σ_xc) back to DFT basis and re-form H_qp_dft.
    delta_h_qp = sigma_result.v_h_kij_ry + sigma_result.sigma_xc_kij_ry
    delta_h_dft = _rotate_to_dft_basis(delta_h_qp, U_qp)
    H_qp_dft_new = inputs.kin_ion_dft + delta_h_dft

    return SCState(
        H_qp_dft=H_qp_dft_new,
        iteration=state.iteration + 1,
        last_sigma_result=sigma_result,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

@jax.jit
def _eigvalsh_per_k(H: jax.Array) -> jax.Array:
    H_herm = 0.5 * (H + jnp.conj(jnp.swapaxes(H, -1, -2)))
    return jax.vmap(jnp.linalg.eigvalsh)(H_herm)


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
    iteration) so the carry never gets out of sync with a separately-
    tracked E array.

    Returns
    -------
    state_final
        Last :class:`SCState` produced.
    rms_history
        RMS ΔE_n (eV) at each iteration ≥ 1; empty list when
        ``max_iter == 1`` (one-shot G0W0).
    """
    print_fn = inputs.print_fn
    state = state_init
    rms_history: list[float] = []
    E_prev_ev = np.asarray(_eigvalsh_per_k(state.H_qp_dft)) * RYD_TO_EV

    for it in range(max_iter):
        state_new = gw_iteration_map(state, inputs)
        if it == 0 and max_iter == 1:
            return state_new, rms_history
        E_new_ev = np.asarray(_eigvalsh_per_k(state_new.H_qp_dft)) * RYD_TO_EV
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
