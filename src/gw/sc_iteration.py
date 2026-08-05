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

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import barrier, device_put_process_local
from common.units import RYD_TO_EV
from .band_partition import BandPartition, apply_band_partition
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
    centroid_indices: object  # ISDF centroid set — IBZ resolve for the static W
    band_slices: BandSlices
    input_dir: str
    partition: BandPartition
    e_dft_active_kn_ry: jax.Array      # (nk, nb_active) DFT energies for scissor fit
    valence_mask_active_kn: jax.Array  # (nk, nb_active) bool — for scissor val/cond split
    print_fn: Callable = print


@dataclass(frozen=True)
class SCState:
    """State carried across self-consistent iterations.

    The iteration "carry" is **just** ``H_qp_dft`` — the QP Hamiltonian
    on the active subspace (``slices.sigma`` of the wfn bundle), in
    the original DFT basis.  Everything else (``E_qp``, ``U_dft_to_qp``,
    ``efermi``) is derivable by the next iteration's first step
    (``vmap(eigh)``), so we don't carry redundant state — that would
    let convergence checks read inconsistent (E, H) pairs if anyone
    forgot to keep them in sync.

    ``last_sigma_result`` is purely for the final output writer (eqp.dat,
    sigma_diag.dat, freq_debug.dat); it does not feed the next iteration.
    ``last_sigma_basis_U`` is the DFT→QP unitary that DEFINED the basis
    ``last_sigma_result`` was computed in (the eigh of the *previous*
    carry) — the writer must rotate Σ back to DFT with THIS U, not the
    converged U of the final carry: the two agree only at convergence,
    and using the converged U mis-rotates Σ_x/V_H whenever the loop
    stops before the fixed point (maximally so at max_iter=1, where the
    correct U is the identity).
    """

    H_qp_dft: jax.Array              # (nk, nb_active, nb_active) Ry, DFT basis
    iteration: int
    last_sigma_result: SigmaResult | None = None
    last_sigma_basis_U: jax.Array | None = None   # (nk, nb, nb) ⟨DFT_m|QP_n⟩


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
    rep = NamedSharding(inputs.mesh_xy, P(None, None, None))
    # Process-local replication (plain ``jax.device_put`` of host numpy
    # onto a multi-process sharding fires JAX's hidden ``assert_equal``
    # all-gather, P × nk × nb² × 16 B — scorecard AA.1).  ``H0`` is a
    # pure function of the DFT energies, identical on every rank;
    # ``LORRAX_CHECK_REPLICA=1`` re-arms the assertion.
    return SCState(
        H_qp_dft=device_put_process_local(H0, rep),
        iteration=0,
    )


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

    ``mesh_xy.size`` must divide ``nk``; otherwise the resharding fails.
    """
    rep_E = NamedSharding(mesh_xy, P(None, None))
    # ``u_spec`` chooses where U LANDS.  The default replicates it, which
    # is what every consumer wanted until ``gw.qsgw_density`` -- that one
    # contracts U against band-sharded psi one k at a time and wants
    # ``P(None, 'x', 'y')`` so no rank holds a full (nb, nb).  Parametrised
    # rather than copied: the eigh itself, the k-shard hint and the
    # hermitisation are identical and must not drift.
    rep_U = NamedSharding(mesh_xy,
                          P(None, None, None) if u_spec is None else u_spec)
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


# ---------------------------------------------------------------------------
# Density self-consistency: rebuild V_H from the CURRENT orbitals
# ---------------------------------------------------------------------------
#
# OFF BY DEFAULT (``config.density_self_consistent``).  With it off this
# module is byte-identical to before, which is what keeps
# tests/test_sc_oneshot_equivalence.py meaningful.

_PSI_G_CACHE: dict = {}


def _dft_psi_sphere(inputs):
    """DFT ψ(G) on the SC k-set, loaded ONCE and cached.

    The SC bundle carries ψ at ISDF CENTROIDS, which cannot reconstruct
    ρ on the FFT grid, so the density rebuild needs the G-sphere.  ψ_DFT
    is constant across iterations — only U moves — so this is one read per
    run, not one per iteration.
    """
    from jax.sharding import NamedSharding as _NS
    from common.mtxel_sweep import band_sphere_spec

    nb = int(inputs.kin_ion_dft.shape[1])
    key = (id(inputs.wfn), nb)
    hit = _PSI_G_CACHE.get(key)
    if hit is None:
        spec = band_sphere_spec()
        psi = inputs.wfn.load(bands=(0, nb), k="full_bz", sharding=spec)
        bidx = np.asarray(inputs.wfn.box_index(k="full_bz"))
        hit = (psi, bidx)
        _PSI_G_CACHE[key] = hit
    return hit


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
    from gw.efermi import fermi_level_step, step_occupations
    from gw.qsgw_density import rotate_bands, hartree_from_orbitals
    from psp.get_DFT_mtxels import spin_degeneracy_factor
    from common.mtxel_sweep import (SweepGeometry, local_potential_operator,
                                    sweep_matrix_elements)
    from psp.dft_operators import padded_gvectors

    psi_G, bidx = _dft_psi_sphere(inputs)
    nk, nb = int(psi_G.shape[0]), int(psi_G.shape[1])
    kweights = np.full(nk, 1.0 / nk)          # full BZ => uniform, no star

    E_np = np.asarray(E_qp_ry)
    e_f = fermi_level_step(E_np, kweights, float(inputs.meta.nelec))
    occ = step_occupations(E_np, e_f)

    # Rotated orbitals: needed for rho, NOT for the contraction below.
    psi_qp = rotate_bands(psi_G, U_qp, mesh=inputs.mesh_xy)
    f_spin = spin_degeneracy_factor(inputs.wfn)
    grid = tuple(int(v) for v in inputs.wfn.fft_grid)
    _rho, V_H_r = hartree_from_orbitals(
        psi_qp, occ, kweights, inputs.wfn, mesh=inputs.mesh_xy,
        box_index=bidx, fft_grid=grid,
        truncation_2d=bool(getattr(inputs.config, "sys_dim", 3) == 2),
        spin_degeneracy=f_spin,
        expected_electrons=f_spin * float(np.einsum("k,kn->", kweights, occ)),
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


def gw_iteration_map(state: SCState, inputs: SCInputs) -> SCState:
    """One self-consistent QSGW step in the DFT basis.

    Pure function — no side effects on ``inputs.wfns_dft``.  All
    derived quantities (E_qp, U_qp, efermi) are recomputed each call;
    the only carried state is ``H_qp_dft`` on the active subspace.

    Screening is mode-orthogonal: each iteration asks the configured Σ
    scheme which W's it needs (via :func:`gw.screening.screening_requests_for`),
    evaluates them in one pass (:func:`gw.screening.compute_screening`),
    and hands the resulting ``{role → W_q}`` dict to
    :func:`gw.sigma_dispatch.compute_sigma_xc`.  No ``compute_chi0``
    call lives here directly — adding a new Σ scheme that wants extra
    W frequencies is purely a screening_requests_for + compute_sigma_xc
    change.
    """
    from .screening import compute_screening, screening_requests_for

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
        # (gated by tests/test_sc_oneshot_equivalence.py).
        H_np = np.asarray(state.H_qp_dft)
        nb = H_np.shape[1]
        diag = np.diagonal(H_np, axis1=1, axis2=2)
        if not np.any(H_np - diag[:, :, None] * np.eye(nb)[None]):
            E_np = np.ascontiguousarray(diag.real)
            vbm = E_np[:, :n_occ].max()
            cbm = E_np[:, n_occ:].min() if n_occ < nb else vbm
            efermi_ry = 0.5 * (vbm + cbm)
            rep2 = NamedSharding(inputs.mesh_xy, P(None, None))
            rep3 = NamedSharding(inputs.mesh_xy, P(None, None, None))
            # Process-local replication — see the H_qp_dft note above
            # (same hidden assert_equal; same rank-invariance argument).
            E_qp_ry = device_put_process_local(E_np, rep2)
            U_qp = device_put_process_local(
                np.broadcast_to(
                    np.eye(nb, dtype=np.complex128), H_np.shape),
                rep3)
    if E_qp_ry is None:
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

    # DENSITY SELF-CONSISTENCY (opt-in).  V_H[rho_i] from THIS iteration's
    # orbitals, alongside Sigma_i and from the same U_qp, both feeding
    # H_{i+1}.  Off by default, so the one-shot equivalence gate holds.
    v_h_dft_new = None
    if bool(getattr(inputs.config, "density_self_consistent", False)):
        v_h_dft_new, e_f_ry = rebuild_hartree_dft_basis(
            inputs, U_qp, E_qp_ry)
        inputs.print_fn(
            f"    density-SC: rebuilt V_H from iteration {state.iteration} "
            f"orbitals (E_F = {e_f_ry:.6f} Ry)")

    # Per-mode screening: solve W at every frequency the Σ scheme needs.
    # XLA cache hits on iteration ≥ 2 (same shapes, new values).
    requests = screening_requests_for(
        inputs.config.compute_mode, inputs.config)
    W_by_role = compute_screening(
        wfns_qp, inputs.V_q, requests,
        quad=inputs.quad, e_ref=inputs.e_ref,
        sym=inputs.sym, centroid_indices=inputs.centroid_indices,
        config=inputs.config, meta=inputs.meta, mesh_xy=inputs.mesh_xy,
        print_fn=inputs.print_fn,
    )

    # Σ_xc dispatch — mode-orthogonal.  ``write_sigma_omega_h5=False``
    # so intermediate SC iterations don't thrash sigma_mnk.h5; the
    # converged tensor is written once after run_self_consistency
    # returns (see ``dump_sigma_omega_h5_final``).
    sigma_result = compute_sigma_xc(
        inputs.config.compute_mode,
        wfns=wfns_qp, V_q=inputs.V_q, W_by_role=W_by_role,
        e_qp_ev=np.asarray(E_qp_ry) * RYD_TO_EV,
        static_head_terms=inputs.static_head_terms,
        head_resolver=inputs.head_resolver,
        quad=inputs.quad, e_ref=inputs.e_ref,
        config=inputs.config, meta=inputs.meta, mesh_xy=inputs.mesh_xy,
        sym=inputs.sym, wfn=inputs.wfn,
        band_slices=inputs.band_slices,
        input_dir=inputs.input_dir,
        # The stored/gspace V_H lives in the DFT basis; this is the U that
        # takes it into the basis ``wfns_qp`` is expressed in.
        hartree_basis_rotation=U_qp,
        omit_v_h=v_h_dft_new is not None,
        write_sigma_omega_h5=False,
        print_fn=inputs.print_fn,
    )

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
    delta_h_dft = _rotate_to_dft_basis(delta_h_qp, U_qp)
    if v_h_dft_new is not None:
        delta_h_dft = delta_h_dft + v_h_dft_new
    H_qp_dft_full = inputs.kin_ion_dft + delta_h_dft
    scissor_E_qp_kn_ry = _scissor_E_qp_for_outofrange(
        H_qp_dft_full, inputs.e_dft_active_kn_ry,
        inputs.valence_mask_active_kn,
        inputs.partition.in_range_mask,
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
        last_sigma_basis_U=U_qp,
    )


# ---------------------------------------------------------------------------
# Per-iteration scissor refit for non-protected out-of-range bands
# ---------------------------------------------------------------------------

def _scissor_E_qp_for_outofrange(
    H_qp_dft_full: jax.Array,
    e_dft_kn_ry: jax.Array,
    valence_mask_kn: jax.Array,
    in_range_mask: jax.Array,
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
    in_range = np.asarray(in_range_mask, dtype=bool)
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
    _, eigvalsh_kshard = _kshard_eigh_kernels(inputs.mesh_xy)
    # E-history dump dir from config.sc (LORRAX_SC_DUMP_DIR env is a
    # deprecated override, applied at config construction).
    _dump_dir = inputs.config.sc.dump_dir

    # One-shot fast path: no acceleration needed.
    if max_iter == 1:
        state_new = gw_iteration_map(state_init, inputs)
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
    """Plain α-mixing fixed point.  Diagnostic / fallback path."""
    state = state_init
    rms_history: list[float] = []
    E_prev_ev = np.asarray(eigvalsh_kshard(state.H_qp_dft)) * RYD_TO_EV
    _e_history: list[np.ndarray] = [E_prev_ev.copy()]
    if mixing != 1.0:
        print_fn(f"  SC mixing α = {mixing:.3f} (linear)")

    for it in range(max_iter):
        state_new = gw_iteration_map(state, inputs)
        if mixing != 1.0:
            H_mixed = (
                mixing * state_new.H_qp_dft
                + (1.0 - mixing) * state.H_qp_dft
            )
            state_new = SCState(
                H_qp_dft=H_mixed,
                iteration=state_new.iteration,
                last_sigma_result=state_new.last_sigma_result,
                last_sigma_basis_U=state_new.last_sigma_basis_U,
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
        state = state_new
        E_prev_ev = E_new_ev
        if rms < tol_ev:
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
    flat L2-norm-of-residual on H_flat (Ry) the rCROP solver expects::

        ‖H_new − H_old‖_2 / √(nk · nb²) ≈ RMS-per-element ≈ RMS ΔE / RYD_TO_EV
    """
    from mixing.acceleration import rcrop_nojit

    H0 = state_init.H_qp_dft
    nk, nb, _ = H0.shape
    n_elem = nk * nb * nb
    print_fn(
        f"  SC rCROP: history_depth={history_depth}, "
        f"max_iter={max_iter}, tol={tol_ev:.1e} eV/band-RMS")

    # Bookkeeping for per-iteration printing + final SigmaResult capture.
    _e_history: list[np.ndarray] = [
        np.asarray(eigvalsh_kshard(H0)) * RYD_TO_EV]
    _last_sigma: list = [None]
    _last_basis_U: list = [None]
    _iter_idx = [0]
    rms_history: list[float] = []

    def residual_fn(H_flat: jnp.ndarray) -> jnp.ndarray:
        H = H_flat.reshape(H0.shape)
        # rCROP's mixing combinations don't preserve Hermitisation
        # exactly (numeric drift); re-Hermitise before feeding the
        # iteration map so eigh stays well-defined.
        H = 0.5 * (H + jnp.conj(jnp.swapaxes(H, -1, -2)))
        state_in = SCState(
            H_qp_dft=H, iteration=_iter_idx[0],
            last_sigma_result=_last_sigma[0])
        state_out = gw_iteration_map(state_in, inputs)
        _last_sigma[0] = state_out.last_sigma_result
        _last_basis_U[0] = state_out.last_sigma_basis_U
        # Track per-call eigenvalue RMS so the user sees progress in the
        # same shape the linear path prints.
        E_new = np.asarray(eigvalsh_kshard(state_out.H_qp_dft)) * RYD_TO_EV
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
        _iter_idx[0] += 1
        return (state_out.H_qp_dft - H).flatten()

    # rCROP residual tolerance: ‖f‖₂ ≤ tol_ry · √(n_elem) ⇔ per-element
    # RMS ≤ tol_ry.  Convert RMS ΔE in eV → Ry first.
    tol_ry = tol_ev / RYD_TO_EV
    tol_resid = float(np.sqrt(n_elem)) * tol_ry

    result = rcrop_nojit(
        residual_fn,
        np.asarray(H0).flatten(),
        m=history_depth,
        maxit=max_iter,
        tol=tol_resid,
        print_fn=None,  # we print our own RMS-ΔE history above
    )
    print_fn(
        f"  SC rCROP done: {result.iterations} iterations, "
        f"converged={bool(result.converged)}, "
        f"final ‖residual‖₂ = {float(result.residual_norms[-1]):.4e} Ry")

    # Final state: use the last x from rCROP (Hermitised) and the last
    # captured SigmaResult so the writer downstream has the full
    # frequency-grid Σ_c, head pieces, etc.
    H_final = jnp.asarray(result.x).reshape(H0.shape)
    H_final = 0.5 * (H_final + jnp.conj(jnp.swapaxes(H_final, -1, -2)))
    state_final = SCState(
        H_qp_dft=H_final,
        iteration=_iter_idx[0],
        last_sigma_result=_last_sigma[0],
        last_sigma_basis_U=_last_basis_U[0],
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
    os.makedirs(dump_dir, exist_ok=True)
    np.save(os.path.join(dump_dir, "e_history_kn_ev.npy"),
            np.stack(e_history, axis=0))
    print_fn(
        f"  SC dump: saved {len(e_history)} eigenvalue snapshots to "
        f"{dump_dir}/e_history_kn_ev.npy (shape (iter, k, n))"
    )


def run_sc_driver(
    wfns,
    V_q: jax.Array,
    kin_ion: jax.Array,
    *,
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
) -> tuple[SigmaResult, jax.Array, list[float]]:
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
        the basis-of-record U (``state.last_sigma_basis_U`` — the
        unitary that defined the basis the last ``compute_sigma_xc``
        call ran in; the converged U is one iteration ahead and agrees
        only at the fixed point — see test_sc_oneshot_equivalence).
        ``sigma_omega_h5_path`` points at the converged single-write
        sigma_mnk.h5; ``efermi_dft_ev`` is filled for every mode.
    sigma_total : (nk, nb, nb) Ry
        Σ_xc + V_H in the DFT basis — the eigh operand
        ``H_QP = kin_ion + sigma_total``.
    rms_history : list[float]
        RMS ΔE (eV) per iteration.
    """
    import dataclasses

    from .band_partition import BandPartition
    from .scissor import classify_bands_in_grid

    e_dft_active_kn_ry = jnp.asarray(np.asarray(enk_dft, dtype=np.float64))
    nb_active = e_dft_active_kn_ry.shape[1]
    val_mask_active = jnp.broadcast_to(
        jnp.arange(nb_active) < int(meta.nelec),
        e_dft_active_kn_ry.shape)

    # In-range mask: bands whose E_DFT lies inside [σ_ω_min, σ_ω_max]
    # at *every* k.  Bands outside the ω-grid get the per-iteration
    # scissor (otherwise their Σ_c is clamped at the grid edge → the
    # QSGW H-build feeds garbage diagonals that explode the iteration).
    efermi_ev = float(wfn.efermi) * RYD_TO_EV
    omega_min_ev = float(config.ppm.omega_min_ev) + efermi_ev
    omega_max_ev = float(config.ppm.omega_max_ev) + efermi_ev
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

    inputs = SCInputs(
        wfns_dft=wfns, V_q=V_q, kin_ion_dft=kin_ion,
        quad=quad, e_ref=e_ref,
        static_head_terms=static_head_terms,
        head_resolver=head_resolver,
        config=config, meta=meta, mesh_xy=mesh_xy,
        sym=sym, wfn=wfn, centroid_indices=centroid_indices,
        band_slices=band_slices, input_dir=input_dir,
        partition=partition,
        e_dft_active_kn_ry=e_dft_active_kn_ry,
        valence_mask_active_kn=val_mask_active,
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
    sigma_result = state_final.last_sigma_result
    print_fn(
        f"  SC done: {len(rms_history)} iterations"
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
    if config.debug.write_wfn_h5:
        dump_qp_wfn_artifacts(
            state_final, n_occ=int(meta.nelec), mesh_xy=mesh_xy,
            wfn=wfn, band_slices=band_slices, kgrid=meta.kgrid,
            output_dir=input_dir, print_fn=print_fn,
        )
    sigma_omega_h5_path = dump_sigma_omega_h5_final(
        state_final, config=config, meta=meta, mesh_xy=mesh_xy,
        input_dir=input_dir, print_fn=print_fn,
    )

    # Rotate every QP-basis SigmaResult field back to the DFT basis.
    # The Σ matrices live in the basis of the wfn bundle the last
    # ``compute_sigma_xc`` call ran in — the basis DEFINED by
    # ``state.last_sigma_basis_U``.  Downstream driver code (H build +
    # eigh, writer, freq_debug) is written for DFT-basis matrices
    # (kin_ion is DFT basis throughout).
    U = jnp.asarray(state_final.last_sigma_basis_U)
    sig_h = _rotate_to_dft_basis(sigma_result.v_h_kij_ry, U)
    sig_x = _rotate_to_dft_basis(sigma_result.sigma_x_kij_ry, U)
    sigma_xc_dft = _rotate_to_dft_basis(sigma_result.sigma_xc_kij_ry, U)
    sigma_total = sigma_xc_dft + sig_h
    sigma_result_dft = dataclasses.replace(
        sigma_result,
        v_h_kij_ry=sig_h,
        sigma_x_kij_ry=sig_x,
        sigma_xc_kij_ry=sigma_xc_dft,
        sigma_sx_kij_ry=(
            _rotate_to_dft_basis(sigma_result.sigma_sx_kij_ry, U)
            if sigma_result.sigma_sx_kij_ry is not None else None),
        sigma_coh_kij_ry=(
            _rotate_to_dft_basis(sigma_result.sigma_coh_kij_ry, U)
            if sigma_result.sigma_coh_kij_ry is not None else None),
        sigma_omega_h5_path=sigma_omega_h5_path,
        efermi_dft_ev=float(wfn.efermi) * RYD_TO_EV,
    )
    return sigma_result_dft, sigma_total, rms_history


def final_qp_eigenstates(
    state: SCState, *, n_occ: int, mesh_xy: Mesh,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diagonalise the converged ``state.H_qp_dft`` and return the QP eigenstates.

    Returned arrays are host-side numpy (not jax.Array) since the
    typical consumers (WFN_qp.h5 writer, eqp.dat tooling) operate on
    NumPy.  Use this once after :func:`run_self_consistency` to extract
    the (E_qp_ry, U_qp, efermi_ry) needed for downstream rotation +
    serialisation.

    Returns
    -------
    enk_qp_ry : (nk, nb_active) float64
    U_kmn     : (nk, nb_active, nb_active) complex128, ``U[k, m, n] = ⟨DFT_m | QP_n⟩``
    efermi_ry : float, midgap of the converged eigenvalues
    """
    E_ry, U, efermi_ry = _diagonalize_and_get_efermi(
        state.H_qp_dft, n_occ, mesh_xy)
    return (
        np.asarray(E_ry, dtype=np.float64),
        np.asarray(U, dtype=np.complex128),
        float(efermi_ry),
    )


def dump_sigma_omega_h5_final(
    state: SCState, *,
    config,
    meta,
    mesh_xy: Mesh,
    input_dir: str,
    print_fn: Callable = print,
) -> str | None:
    """Write the converged ``sigma_mnk.h5`` once after SC convergence.

    Pulls the full ω-grid Σ_c tensor from ``state.last_sigma_result``
    (which the iteration map captures from each
    :func:`compute_sigma_xc` call but does NOT write to disk during SC
    iterations — see the ``write_sigma_omega_h5=False`` flag in
    :func:`gw_iteration_map`).  Replaces ~30 redundant per-iteration
    writes with a single end-of-run write.

    Returns the on-disk path (or ``None`` for static modes that didn't
    populate a Σ_c(ω) tensor).
    """
    sigma_result = state.last_sigma_result
    if sigma_result is None or sigma_result.sigma_c_omega_kij_ry is None:
        return None
    from .ppm_pipeline import _write_sigma_omega_h5

    path = _write_sigma_omega_h5(
        sigma_result.sigma_c_omega_kij_ry,
        sig_x=sigma_result.sigma_x_kij_ry,
        sig_h=sigma_result.v_h_kij_ry,
        config=config, input_dir=input_dir,
        meta=meta, mesh_xy=mesh_xy,
    )
    print_fn(f"  Σ_c(ω) tensor: {path}")
    return path


def dump_qp_wfn_artifacts(
    state: SCState, *,
    n_occ: int,
    mesh_xy: Mesh,
    wfn,                                 # WFNReader (source of base coeffs + crystal)
    band_slices,
    kgrid,                               # (nkx, nky, nkz)
    output_dir: str,
    print_fn: Callable = print,
) -> tuple[str, str, float]:
    """Post-SC artifact dump: WFN_qp.h5 + qp_wfn_rotations.h5.

    Diagonalises the converged ``state.H_qp_dft`` once, then writes:

    * ``WFN_qp.h5`` — full BGW-format wavefunction file with active-block
      ψ rotated by ``U`` and active-block energies replaced by ``E_qp``;
      bands outside the active block keep their DFT values.
      Drop-in replacement for downstream BSE / restart paths that read
      a WFN.h5.
    * ``qp_wfn_rotations.h5`` — small companion file containing just
      ``(U, E_qp)`` for tools that prefer to apply the rotation
      themselves.

    Both files are rank-0-only writes (h5py is single-writer); a
    multihost barrier follows so the caller can rely on both files
    existing on every rank when this function returns.

    Returns ``(qp_wfn_path, qp_rotations_path, efermi_ry)``.
    """
    from file_io.qp_wfn import write_qp_rotations_h5, write_qp_wfn_h5

    enk_qp_ry, U_kmn, efermi_ry = final_qp_eigenstates(
        state, n_occ=n_occ, mesh_xy=mesh_xy)
    qp_wfn_path = os.path.join(output_dir, "WFN_qp.h5")
    qp_rot_path = os.path.join(output_dir, "qp_wfn_rotations.h5")
    if jax.process_index() == 0:
        write_qp_wfn_h5(
            qp_wfn_path, wfn=wfn,
            U_kmn=U_kmn, enk_active_qp_ry=enk_qp_ry,
            band_start=band_slices.b0, band_stop=band_slices.b3,
        )
        write_qp_rotations_h5(
            qp_rot_path,
            U_mnk=U_kmn,
            E_qp_nk=enk_qp_ry * 0.5,                       # Ry → Hartree
            band_start=band_slices.b0, band_stop=band_slices.b3,
            kpoints_crys=np.asarray(wfn.kpoints, dtype=np.float64),
            nkx=int(kgrid[0]), nky=int(kgrid[1]), nkz=int(kgrid[2]),
        )
    barrier("qp_wfn_h5_write")
    print_fn(f"  QP WFN:       {qp_wfn_path}")
    print_fn(f"  QP rotations: {qp_rot_path}")
    print_fn(f"  Final E_F (midgap, eV): {efermi_ry * RYD_TO_EV:.6f}")
    return qp_wfn_path, qp_rot_path, efermi_ry


__all__ = [
    "SCInputs",
    "SCState",
    "gw_iteration_map",
    "make_initial_state_from_dft",
    "run_self_consistency",
    "run_sc_driver",
    "final_qp_eigenstates",
    "dump_qp_wfn_artifacts",
    "dump_sigma_omega_h5_final",
]
