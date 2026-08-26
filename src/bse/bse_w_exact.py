"""Exact W_c(omega) via shifted solves with the non-TDA RPA density resolvent.

Cross-validates the GW screened Coulomb.  In the ISDF centroid (density) basis
the screened interaction obeys the Casida resolvent identity

    W(omega) - v  =  v (omega - H_RPA)^{-1} v            (omega = 0: static W0)

where H_RPA is the NON-TDA symplectic RPA (test-charge) density-response
Hamiltonian with the bare-exchange RING kernel V (the B1 dense k-summed form):

    H_RPA = [[ D + V ,   V   ],
             [  -V   , -D - V ]]

V sits in BOTH blocks — the RPA ring coupling K^A = (1/Nk)<M_t|v|M_t'>
(``build_bse_ring_matvec_full(..., screening=True)``), NOT the excitonic V_B of
Henneke Eq. 2-20.  This is the test-charge screening whose resolvent resums the
RPA bubble chi = chi0 (1 - v chi0)^{-1}; the exciton V_B kernel is a different
response and does NOT reproduce W (it overshoots the q=0 tile by ~1.8x).  The GW
W is full-RPA, not TDA — dropping the -V/Y block (TDA) fails by construction.

Convention (verified bit-for-bit against the folded static RPA and the on-disk
``W0_qmunu - V_qmunu`` q=0 tile to ~2e-9 — see the "W(0) resolvent cross-check"
section of reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md):

  * Probe column nu: g = e_nu in centroid space; the transition generator applies
    v then the pair-density vertex, f = M^dag (v e_nu)
    (``build_realspace_random_transition_generator``) — the RIGHT vertex of v X v.
  * RHS is that same f with a minus on the anti-resonant (Y) block: rhs = [f; -f]
    (density super-vertex [rho; -rho]; the ring coupling makes the excitation and
    de-excitation vertices coincide, so both blocks carry the SAME f).
  * Shifted solve x = (z - H_RPA)^{-1} rhs at z = (omega + i eta)/Ry via GMRES.
  * Readout s = x[0] + x[1] = X + Y; the density-snapshot vertex applies the pair
    density then v: w_c(mu) = v (M s) = [v chi v]_{mu,nu} = column nu of W - v.

The generator/snapshot k-SUM the pair densities, so the reconstructed tile is the
q=0 block.  H_RPA carries no q=0 head (vhead/whead are a separate rank-1 piece);
``--compare-w0`` loads head-LESS bodies on both sides (``inject_head=False``) and
compares body-to-body.
"""
from __future__ import annotations

import math
import time
import numpy as np
import h5py

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import (
    RY_TO_EV_DEFAULT, ensure_W_R, build_preconditioner_diagonal_sharded,
    _apply_shifted_matvec, _gmres_solve_core, matvec_operands,
)
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec_full,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
    create_mesh_xy_from_flags,
    make_bse_shardings,
)
from .bse_serial import compute_pair_amplitude
from common.collectives import device_put_process_local, gather_to_host
from common import rank_criterion
import common.timing as timing

jax.config.update("jax_enable_x64", True)

# Cache the compiled per-column-scan block-GMRES engine per operator structure.
# Keyed on (id(matvec), max_iter, tol, dtype) — NOT on the per-q data — so the
# finite-q W_q loop and the per-omega oracle sweep reuse ONE executable and every
# q / omega after the first is dispatch-only (see _get_block_gmres_solver).
_BLOCK_GMRES_CACHE: dict[tuple, tuple] = {}


def _build_rpa_resolvent(mesh_xy: Mesh, data: dict):
    """Assemble the RPA-screening resolvent stack for ``data``.

    Returns ``(matvec, diag_h, gen, snapshot, sh)``.  ``matvec`` is the non-TDA
    symplectic RPA density-response Hamiltonian (screening ring kernel, no W);
    ``ensure_W_R`` populates the placeholder 8th matvec argument.

    ``gen`` (zeta->pair seed) and ``snapshot`` (pair->zeta projection) are the
    two reshard boundaries of the resolvent; ``snapshot`` is built in the
    reduce-scatter mode so the projection-back assembles ``W(mu_X, nu_Y)`` tiles
    directly (see :func:`apply_screening_resolvent_block`).
    """
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    ensure_W_R(data, include_W=False, mesh_xy=mesh_xy)
    matvec = build_bse_ring_matvec_full(
        mesh_xy, nkx, nky, nkz, include_W=False, screening=True)
    diag_h = build_preconditioner_diagonal_sharded(
        data, mesh_xy, include_W=False, use_tda=False)
    gen = build_realspace_random_transition_generator(
        mesh_xy, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
    snapshot = build_density_snapshot_operator(
        mesh_xy, nkx, nky, nkz, scatter_nu_on_y=True)
    return matvec, diag_h, gen, snapshot, make_bse_shardings(mesh_xy)


#: ``(q, nkx, nky, nkz) -> flat int32 k permutation``.  Tiny (nk entries) and a
#: pure function of the grid, so memoising it costs nothing and keeps the host
#: out of the per-q path entirely.
_ROLL_INDEX_CACHE: dict = {}


def _roll_k_index(q, nkx, nky, nkz):
    """The ``+q`` roll of the C-order (nkx,nky,nkz) k-axis, AS AN INDEX.

    Returns the flat ``(nk,)`` int32 permutation ``idx`` with
    ``out[k] = arr[idx[k]] = arr[k − q]`` on the wrapped grid — the shift that
    gathers the conduction quantity at ``k − q`` into slot ``k`` and so
    reproduces the stored ``W0_qmunu[q_flat]`` tile (see
    :func:`build_finite_q_data`).

    WHY AN INDEX AND NOT A ROLL OF THE DATA.  The roll has to be compile-stable
    — a *static* device ``jnp.roll`` bakes the q-offset into the program and
    recompiles once per q, which is why this used to be done on host (numpy)
    with the rolled array uploaded as plain DATA.  But the host round trip is
    not the only way to get compile stability: hand the DEVICE a runtime index
    and the program is q-independent by construction, because q enters as an
    argument rather than as a constant.  The index is ``nk`` int32s — 36 B on
    the gnppm fixture — against the psi_c stack the host path moved.

    MEASURED (evidence/sync_audit, 2026-08-16, gnppm 399/9k fixture): the host
    path cost 24.6 ms per q at 1 process and 20.0 ms at 4, moving 2.2 MiB
    device→host plus 8.8 MiB host→device EVERY q (a ``gather_to_host`` of
    ψ_c is a cross-process ``process_allgather`` at P>1, so that traffic is on
    the network); the index path costs 5.9 ms / 9.1 ms and moves ZERO bytes
    across the host boundary.  Bit-identical on every returned slot at every q
    — a permutation is a permutation.

    The gather itself is also communication-free: every array rolled here is
    sharded on a DIFFERENT axis than k (ψ on μ over 'x'/'y', ε replicated), so
    ``take`` on axis 0 stays entirely inside each shard."""
    key = (int(q[0]) % nkx, int(q[1]) % nky, int(q[2]) % nkz, nkx, nky, nkz)
    idx = _ROLL_INDEX_CACHE.get(key)
    if idx is None:
        flat = np.arange(nkx * nky * nkz, dtype=np.int32).reshape(nkx, nky, nkz)
        idx = np.ascontiguousarray(
            np.roll(flat, shift=(key[0], key[1], key[2]),
                    axis=(0, 1, 2)).reshape(-1), dtype=np.int32)
        _ROLL_INDEX_CACHE[key] = idx
    return idx


def build_finite_q_data(data, q, mesh_xy):
    """Finite-momentum screening data for the W_q resolvent — the q generalization.

    The RPA density response at momentum ``q`` lives in the on-grid pair basis
    ``|v k, c k+q⟩``: conduction quantities are remapped on the k-axis, the
    screening tile becomes the finite-q ``V_qmunu[q_flat]``, valence/energies and
    everything else are the q=0 arrays.  Returns a shallow copy of ``data`` with
    only the conduction/V slots swapped, so the SAME
    :func:`apply_screening_resolvent_block` engine (matvec, seed, project,
    solver, sharding) runs unchanged — this GENERALIZES q=0, it does not fork it.

    Remap (``q = (qx,qy,qz)`` integer grid steps; C-order flat
    ``k = ix·nky·nkz + iy·nkz + iz``):

      * ``psi_c`` / ``eps_c`` ← rolled by ``+q`` on the (nkx,nky,nkz) k-axis
        (a device gather through :func:`_roll_k_index`) ⇒ slot ``k`` holds the
        conduction value at ``k − q``.
        The pair density is then ``M^q_cvk(μ) = Σ_s conj(ψ_c[k−q](μ)) ψ_v[k](μ)``
        and ``D^q = ε_c[k−q] − ε_v[k]``; summed over k this is exactly the GW
        producer's χ₀(q) convolution ``Σ_k Gc_k Gv*_{k+q}`` (relabel k→k−q), which
        reproduces the on-disk ``W0_qmunu[q_flat]`` tile.
      * NO umklapp Bloch phase.  The GW χ₀(q) is built by a plain periodic
        FFT-convolution over k (``w_isdf._get_chi_minimax_kernel``), which uses
        the RAW stored ψ at the wrapped index — no ``exp(-2πi G_umk·s_μ)`` factor.
        Applying the design-doc umklapp phase here breaks the match (rel_err
        0.6–3.2 vs 1e-8); the phase belongs to a DIRECT-read finite-Q BSE path,
        not to matching this FFT-convolution-produced W tile.  Verified on the
        MoS2 gnppm fixture (finite-q PHASE2_LOG section).
      * ``V_q0`` ← ``V_qmunu[q_flat]`` = ``data['V_q_full'][:,:,qx,qy,qz]`` — the
        finite-q exchange tile.  At q≠0 it KEEPS G=0 (``compute_vcoul`` zeroes
        G=0 only at q=0) and carries NO separate rank-1 head (heads are a
        q=0-only piece).
      * **THE PAIR-DENSITY VERTEX IS CONJUGATED**, and that is the whole finite-q
        correctness question — see the CONJUGATE VERTEX note on the body below.
        ``q=(0,0,0)`` is therefore no longer byte-identical to the unshifted q=0
        data (the roll is still the identity and ``V_q_full[...,0] == V_q0``, but
        ψ comes back conjugated); it is VALUE-identical, because χ₀(0) is real.

    Requires ``data`` loaded with ``load_v_full=True``.
    """
    if data.get("V_q_full") is None:
        raise ValueError(
            "build_finite_q_data needs the full V tensor; load the restart with "
            "load_bse_data_from_restart_sharded(..., load_v_full=True).")
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
    sh = make_bse_shardings(mesh_xy)
    dq = dict(data)
    dq["V_q0"] = data["V_q_full"][:, :, qx, qy, qz]
    # THE ROLL IS A DEVICE GATHER THROUGH A RUNTIME INDEX (see _roll_k_index),
    # so a different q needs no new compile AND no host round trip.  The index
    # is nk int32s, staged process-locally: plain ``device_put`` of host data
    # onto a multi-process sharding fires JAX's hidden ``assert_equal``
    # all-gather (scorecard AA.1), and LORRAX_CHECK_REPLICA=1 re-arms it.
    #
    # This replaced a gather_to_host(psi_c) -> numpy roll -> three
    # device_put_process_local round trips.  Measured 2026-08-16
    # (evidence/sync_audit): 24.6 -> 5.9 ms per q at 1 process, 20.0 -> 9.1 ms
    # at 4, and 11.0 MiB per q of host traffic -> 0.  Do not put the host roll
    # back: at P>1 the ψ_c gather is a ``process_allgather``, i.e. the whole
    # conduction stack crossed the NETWORK once per irreducible q.
    idx = device_put_process_local(
        _roll_k_index((qx, qy, qz), nkx, nky, nkz),
        NamedSharding(mesh_xy, P()))

    def _roll(arr, target):
        # axis 0 is k; every array rolled here is sharded on some OTHER axis
        # (ψ on μ, ε replicated), so this gather never leaves a shard.
        return jax.lax.with_sharding_constraint(
            jnp.take(arr, idx, axis=0), target)

    #
    # ── THE CONJUGATE VERTEX ─────────────────────────────────────────────────
    # ψ comes back CONJUGATED, on BOTH legs, and this is the finite-q fix.
    #
    # The resolvent's four pair-density vertices — the seed's decode
    # (``build_realspace_random_transition_generator``), the ring kernel's encode
    # and decode (``apply_V_ring`` / the hoisted ``M_X``), and the snapshot's
    # encode (``build_density_snapshot_operator``) — all carry ONE fixed
    # conjugation convention, ``K^x = M V M†`` with the conjugate on the ENCODE
    # leg.  That convention is NOT ours to move: it is pinned by the optical BSE
    # exchange term (``bse_serial.apply_bse_hamiltonian_single_device``, the
    # "Conjugation:" block) and gated there.  Composed through the resolvent it
    # assembles
    #
    #     X(μ,ν) = -(2/N_k) Σ_i conj(M_i(μ)) M_i(ν) / ΔE_i  =  conj(χ₀) = χ₀ᵀ ,
    #
    # whereas the GW producer's χ₀(q) — the object whose Dyson solve WROTE the
    # ``W0_qmunu[q]`` tile this path is scored against — is the other one,
    # ``χ₀(μ,ν) = -(2/N_k) Σ_i M_i(μ) conj(M_i(ν)) / ΔE_i``
    # (``w_isdf._get_chi_minimax_kernel``: ``chi_R = Gc_R · conj(Gv_R)``).
    #
    # At q=0 the two are the SAME MATRIX and nothing is visible: the k-sum runs
    # over ±k pairs whose pair densities are complex conjugates under TRS, so
    # χ₀(0) is REAL (measured ‖χ₀−χ₀ᵀ‖/‖χ₀‖ = 4.7e-11 on the MoS2 gnppm fixture).
    # At q≠0 the ±k pairing sends k→−k into the −q tile instead, χ₀(q) stays
    # Hermitian but stops being symmetric (2.9e-01 on the same fixture at
    # q=(0,1,0)), and the un-flipped chain resums χ₀(−q) against the +q Coulomb
    # tile V(q) — a hybrid that is not any stored tile in any conjugation, which
    # is exactly what the finite-q closure measured (rel_err 6.87e-01, and the
    # dense two-line model reproduces that number to five digits).
    #
    # Conjugating ψ on both legs flips all four vertices at once (each is
    # bilinear in (ψ_c, ψ_v) with exactly one conj), turning X into χ₀ itself.
    # It is EXACT — no TRS assumption, unlike "roll by −q", which reaches the
    # same place only through χ₀(−q) = conj(χ₀(q)) and would silently be wrong on
    # a TRS-broken system.  It leaves the optical BSE convention untouched, and
    # it is inert on the preconditioner diagonal (``diag(K)`` is real and
    # conjugation-invariant because V_q is Hermitian).
    # ─────────────────────────────────────────────────────────────────────────
    #
    # ψ_c_Y carries the SAME values as ψ_c_X on a different sharding (the
    # loader's ``with_sharding_constraint`` copy, bse_loading; likewise
    # enforce_trs_pair_gauge), so each is rolled on its own layout — two local
    # gathers rather than one gather plus a reshard.  The ``.get`` fallback
    # keeps a payload that never built the Y copy working; it costs one
    # reshard and never fires in tree.
    psi_c_X_q = _roll(data["psi_c_X"], sh.psi_x)
    psi_c_Y_q = _roll(data.get("psi_c_Y", data["psi_c_X"]), sh.psi_y)
    dq["psi_c_X"] = jax.lax.with_sharding_constraint(
        jnp.conj(psi_c_X_q), sh.psi_x)
    dq["psi_c_Y"] = jax.lax.with_sharding_constraint(
        jnp.conj(psi_c_Y_q), sh.psi_y)
    dq["eps_c"] = _roll(data["eps_c"], sh.eps)
    # ψ_v does not roll (the valence leg stays at k) but it DOES conjugate: a
    # vertex flip applied to one leg only is not a flip, it is a different (and
    # wrong) operator.  On device — no roll means no host round trip to reuse.
    dq["psi_v_X"] = jax.lax.with_sharding_constraint(
        jnp.conj(data["psi_v_X"]), sh.psi_x)
    dq["psi_v_Y"] = jax.lax.with_sharding_constraint(
        jnp.conj(data["psi_v_Y"]), sh.psi_y)
    # M_X/M_Y are hoisted V-term pair amplitudes (audit P3) and are pure functions
    # of ψ — the finite-q roll shifted psi_c and the vertex flip conjugated both,
    # so recompute them from the ROLLED, CONJUGATED states.  The q=0 M's
    # shallow-copied from `data` would be stale twice over.
    dq["M_X"] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(dq["psi_c_X"], dq["psi_v_X"]), sh.psi_x)
    dq["M_Y"] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(dq["psi_c_Y"], dq["psi_v_Y"]), sh.psi_y)
    # The flip is EXACT for the four density vertices (above) and WRONG for
    # the direct rung: the rung is bilinear in (c,c')/(v,v') band pairs and
    # must consume the PHYSICAL (rolled, UN-flipped) arrays — running it on
    # the conjugated ones was value-invisible at q=0 and a 3.6e-4 break of
    # W(-q) = conj(W(q)) at finite q (claim 0215).  The rung therefore gets
    # its own four operand slots (bse_ring_comm ladder_rung_slots /
    # bse_feast.ladder_matvec_operands): psi_c rolled un-conjugated, psi_v
    # the original arrays VERBATIM (aliases — zero copies).  A conj-wrap
    # compensation inside the rung appliers was tried first and refuted by
    # block-level measurement (probe_block_compare, 2026-08-16).
    dq["psi_c_W_X"] = psi_c_X_q       # rolled, NOT conjugated
    dq["psi_c_W_Y"] = psi_c_Y_q
    dq["psi_v_W_X"] = data["psi_v_X"]
    dq["psi_v_W_Y"] = data["psi_v_Y"]
    dq["vertex_flipped"] = True
    return dq


def _canonical_phase(vec_flat):
    """Deterministic phase for one band function: the largest-|entry| element
    (lowest index on exact ties — np.argmax) is made real-positive.  This is
    the 'largest-element phase-fixing' convention of the determinism contract:
    the phase depends on the FUNCTION, not on how upstream happened to phase
    it, so the same WFN yields the same phased band up to upstream jitter."""
    k = int(np.argmax(np.abs(vec_flat)))
    a = vec_flat[k]
    m = abs(a)
    return vec_flat * (np.conj(a) / m) if m > 0 else vec_flat


def _canonical_subspace_basis(Psi):
    """Span-anchored deterministic orthonormal basis of a degenerate block.

    ``Psi`` is (m, L) — m orthonormal states, L = flattened samples.  The
    returned basis depends only on span(Psi) plus fixed conventions — NOT on
    the incoming basis realization, which upstream GPU nondeterminism varies
    run to run (the registered defect: iteration counts varying, one spurious
    refusal).  Construction: project coordinate probes e_j into the span in
    FIXED index order (coefficients c_j = Psi conj at e_j — for orthonormal
    rows the projector coefficients are Psi[:, j]), Gram-Schmidt in that
    order in coefficient space (Euclidean = the band metric), skip probes
    whose residual is below 1e-6 of the block scale, then canonical-phase
    each resulting function.  No eigh/SVD anywhere: decompositions realize
    degenerate-subspace freedom chaotically; fixed-order projections do not."""
    m, L = Psi.shape
    scale = float(np.linalg.norm(Psi) / np.sqrt(m))
    cols = []
    for j in range(L):
        c = np.conj(Psi[:, j])          # coefficients of P e_j in the basis
        for u in cols:
            c = c - u * np.vdot(u, c)
        n = float(np.linalg.norm(c))
        if n > 1e-6 * scale:
            cols.append(c / n)
            if len(cols) == m:
                break
    if len(cols) != m:                   # pragma: no cover
        raise ValueError(
            "GATE trs_gauge_canonical_basis_deficient: fixed-order probes "
            f"spanned {len(cols)}/{m} of a degenerate block — numerically "
            "collapsed block. doc: bse_w_exact.enforce_trs_pair_gauge")
    V = np.stack(cols, axis=1)           # (m, m) unitary
    out = V.T @ Psi
    for r in range(m):
        out[r] = _canonical_phase(out[r])
    return out


# A stored scalar WFN block enters the TRIM identity on TWO legs:
#
#     conj(Psi_0 + dPsi) - (Psi_0 + dPsi) C
#       = conj(dPsi) - dPsi C,
#
# so a unitary exact conjugation map C gives the max-norm construction-error
# bound 2*delta.  This is about the payload and therefore independent of the
# ladder frequency.  Measured on the 66-band scalar-Si production WFN, the
# raw [61,64) block's independent little-group covariance floor is
# 8.98913428e-9; the two-leg prediction is 1.79782686e-8.  Rounding the input
# envelope to 1e-8 gives a 2e-8 bar (11.2% above that prediction).  Nested
# payload prefixes measure 7.36e-10 / 1.55e-9 / 1.327e-8 at 8/20/64 bands:
# widening adds independently constructed, increasingly high-energy blocks
# under the gate's maximum; it does not add an omega-dependent W error.
#
# This is deliberately NOT a projection or a symmetrization of a W/operator
# tile.  The raw payload identity is still measured and structural missing-
# partner failures remain O(1) (1.553e-1 relative on the edge-60 control).
_TRS_WFN_CONSTRUCTION_RTOL = 1.0e-8
_TRS_BLOCK_CONJ_RTOL = 2.0 * _TRS_WFN_CONSTRUCTION_RTOL


#: Bar on ``max|VᴴV − I|`` for a canonical TRIM-block rotation.  The rotation
#: is built by a small fixed-order Gram–Schmidt in f64, so a healthy block
#: lands at ~1e-14; 1e-8 holds six decades of margin.
#:
#: WHY IT EXISTS.  Relaxing the probe-acceptance floor from an absolute 1e-6
#: to a RELATIVE test (``rank_criterion.probe_is_independent``, floor
#: sqrt(eps)) is the repair for a valid deck being refused — but a relative
#: floor is also more permissive, and a probe accepted at the floor is a
#: direction built out of round-off.  This is the certification that makes
#: the relaxation safe, and it is affordable because m is a degenerate-block
#: size (2, 4, 6): an ``m x m`` product, priced before enabling.
#:
#: WHAT IT CERTIFIES, exactly: that the rotation is unitary, hence that the
#: canonicalized block SPANS THE SAME SUBSPACE as the input.  It does NOT
#: certify that the block is Kramers- or conj-closed — the exact gates above
#: do that, and nothing here weakens them.
_TRIM_ROTATION_UNITARITY_BAR = 1.0e-8


def _refuse_unless_rotation_is_unitary(V, *, where, kind):
    """Refuse a canonical TRIM rotation that is not numerically unitary."""
    G = V.conj().T @ V
    err = float(np.abs(G - np.eye(G.shape[0])).max())
    if err <= _TRIM_ROTATION_UNITARITY_BAR:
        return
    raise ValueError(
        f"GATE trs_gauge_canonical_rotation_not_unitary: the {kind} rotation "
        f"for {where} has max|VᴴV − I| = {err:.6e}, above the "
        f"{_TRIM_ROTATION_UNITARITY_BAR:.1e} bar.  A non-unitary rotation "
        f"does not preserve the block's span, so the canonicalized states "
        f"are not the eigenstates they replace.  The usual cause is a probe "
        f"accepted at the relative independence floor "
        f"({rank_criterion.PROBE_RTOL:.3e}) — i.e. a direction built out of "
        f"round-off, which means the block genuinely does not span. "
        "doc: bse_w_exact.enforce_trs_pair_gauge")


def _realize_trim_block(Psi, *, where="a TRIM block"):
    """Deterministic REAL basis of a conj-closed scalar block at a TRIM point.

    ``Psi`` is (mu, m) columns.  ``conj(Psi) = Psi C`` (C unitary
    complex-symmetric under exact TRS; refusal below otherwise).  Real
    vectors are built as ``r = c + C conj(c)`` (fixed by the antiunitary
    involution: ``C conj(r) = r`` using ``C conj(C) = 1``) from FIXED-ORDER
    coordinate probes, Gram-Schmidt in fixed order, sign-fixed by the
    largest element — span-anchored and decomposition-free, per the
    determinism contract (same WFN -> same canonical gauge).

    ``where`` names the k point and band block in any refusal — a gate that
    says a block was deficient without saying WHICH block is a dead end for
    the operator.

    THE ACCEPTANCE TEST IS RELATIVE, not an absolute norm floor.  See
    :func:`_kramers_canonicalize_trim_block`, which had the measured failure,
    and ``common/rank_criterion.probe_is_independent``."""
    C, _, _, _ = np.linalg.lstsq(Psi, np.conj(Psi), rcond=None)
    misclose = float(np.abs(np.conj(Psi) - Psi @ C).max())
    scale = float(np.abs(Psi).max())
    threshold = _TRS_BLOCK_CONJ_RTOL * max(scale, 1e-300)
    if misclose > threshold:
        raise ValueError(
            "GATE trs_gauge_block_not_conj_closed: a degenerate block at a "
            f"TRIM k-point is not closed under conjugation (residual "
            f"{misclose:.3e} on scale {scale:.3e}, relative "
            f"{misclose / max(scale, 1e-300):.3e} versus the two-leg "
            f"payload-construction bar {_TRS_BLOCK_CONJ_RTOL:.3e}). "
            "A degenerate partner "
            "probably sits outside the band window; widen the window or run "
            "with w_rpa. doc: bse_w_exact.enforce_trs_pair_gauge")
    C = 0.5 * (C + C.T)
    m = C.shape[0]
    cols = []
    probes = np.conj(Psi.T)              # (m, mu): probes[:, j] = Psi^dag e_j
    # Every probe's PRE-deflation norm, so the acceptance test below is
    # relative to a scale the block itself supplies.
    R_all = probes + C @ np.conj(probes)             # (m, mu)
    n0_all = np.linalg.norm(R_all, axis=0)
    probe_scale = float(n0_all.max()) if n0_all.size else 0.0
    n_rej_max = 0.0
    rel_rej_max = 0.0
    for j in range(Psi.shape[0]):
        r = R_all[:, j].copy()
        for u in cols:
            r = r - u * np.vdot(u, r)
        n = float(np.linalg.norm(r))
        if rank_criterion.probe_is_independent(
                n, float(n0_all[j]), probe_scale):
            cols.append(r / n)
            if len(cols) == m:
                break
        else:
            n_rej_max = max(n_rej_max, n)
            rel_rej_max = max(rel_rej_max,
                              n / max(float(n0_all[j]), probe_scale, 1e-300))
    if len(cols) != m:
        raise ValueError(
            "GATE trs_gauge_canonical_basis_deficient: real-structure probes "
            f"spanned {len(cols)}/{m} of {where}.  Largest REJECTED residual "
            f"norm {n_rej_max:.6e} (relative {rel_rej_max:.3e}); probe scale "
            f"{probe_scale:.6e}; relative floor "
            f"{rank_criterion.PROBE_RTOL:.3e}.  The test is RELATIVE (see "
            "common/rank_criterion.probe_is_independent), so this is a "
            "statement about the block, not about the system size. "
            "doc: bse_w_exact.enforce_trs_pair_gauge")
    V = np.stack(cols, axis=1)
    _refuse_unless_rotation_is_unitary(V, where=where, kind="real-structure")
    out = Psi @ V                        # columns are real functions
    for r in range(m):
        col = out[:, r]
        k = int(np.argmax(np.abs(col)))
        if col[k].real < 0:
            out[:, r] = -col
    return out


def _kramers_canonicalize_trim_block(Psi, R, *, where="a TRIM block"):
    """Deterministic Kramers-canonical pairing at a spinor TRIM point.

    ``Psi`` is (m, ns, mu), m EVEN.  ``phi_{2j+1} = Theta phi_{2j}``, with
    phi_{2j} chosen from FIXED-ORDER coordinate probes projected into the
    remaining subspace, canonical-phased — span-anchored, decomposition-free
    (determinism contract; the previous argmax-pivot greedy realized GPU
    input jitter as a different gauge per run).

    THE ACCEPTANCE FLOOR IS RELATIVE, AND THAT IS THE FIX.  Until 2026-08-22
    a probe was accepted on ``norm > 1e-6`` — an ABSOLUTE floor on a
    coefficient vector whose scale is ``|psi|`` at ONE sample point, i.e.
    falling like ``1/sqrt(N_mu)``.  That is a system-size-dependent refusal
    wearing a numerical constant's clothes: it passes on a fixture and
    refuses on the production deck.  MEASURED (register 2026-08-20): a
    regenerated 4x4x4 LiF spinor WFN that passes the independent
    density-symmetry audit (TRS=HOLDS, magnetization residual 1.42e-13, all
    48/48 spatial operations at max residual 6.31e-12), has a clean 70-band
    edge (1.406 meV) and clears the Theta-closure gate above, was refused
    here with ``Kramers probes spanned 0/2 of a TRIM block`` — every
    fixed-order projected probe discarded by that floor, and the message
    named neither the k point, nor the band block, nor the rejected norm.

    Both are repaired: the test routes through
    ``common/rank_criterion.probe_is_independent`` (relative to the probe's
    own norm and to the largest probe norm in the family), and ``where``
    carries the k point and band block into the refusal.  The exact
    Kramers/Theta gates above are UNCHANGED — this is a numerical-rank
    decision, not a physics one, and nothing here bypasses a physics gate.
    """
    m = Psi.shape[0]
    if m % 2:
        raise ValueError(
            "GATE trs_gauge_kramers_odd_block: a degenerate block at a TRIM "
            f"point has ODD size {m} on a spinor deck — Kramers requires even "
            "multiplicity, so either the degeneracy grouping tolerance split "
            "a pair or the deck breaks TRS. doc: bse_w_exact.enforce_trs_pair_gauge")
    F = Psi.reshape(m, -1).T                        # (ns*mu, m)
    Th = np.stack([_theta(Psi[j:j + 1], R)[0] for j in range(m)]).reshape(m, -1).T
    S, _, _, _ = np.linalg.lstsq(F, Th, rcond=None)
    misclose = float(np.abs(Th - F @ S).max())
    scale = float(np.abs(F).max())
    if misclose > 1e-8 * max(scale, 1e-300):
        raise ValueError(
            "GATE trs_gauge_block_not_theta_closed: a degenerate TRIM block "
            f"is not closed under Theta = i*sigma_y*K (residual {misclose:.3e} "
            f"on scale {scale:.3e}); a Kramers partner probably sits outside "
            "the band window. doc: bse_w_exact.enforce_trs_pair_gauge")
    cols = []
    L = F.shape[0]
    # The probe family's own scale: every probe is one row of F, so the
    # largest row norm is the natural reference and it tracks |psi| at the
    # sample points rather than an absolute constant.
    n0_all = np.linalg.norm(F, axis=1)
    probe_scale = float(n0_all.max()) if n0_all.size else 0.0
    n_rej_max = 0.0
    rel_rej_max = 0.0
    jprobe = 0
    while len(cols) < m and jprobe < L:
        j_here = jprobe
        c1 = np.conj(F[jprobe, :])                  # probe e_j's coefficients
        jprobe += 1
        for u in cols:
            c1 = c1 - u * np.vdot(u, c1)
        n = float(np.linalg.norm(c1))
        if not rank_criterion.probe_is_independent(
                n, float(n0_all[j_here]), probe_scale):
            n_rej_max = max(n_rej_max, n)
            rel_rej_max = max(
                rel_rej_max,
                n / max(float(n0_all[j_here]), probe_scale, 1e-300))
            continue
        c1 = c1 / n
        # canonical phase from the resulting FUNCTION's largest sample
        f1 = F @ c1
        a = f1[int(np.argmax(np.abs(f1)))]
        c1 = c1 * (np.conj(a) / abs(a))
        c2 = S @ np.conj(c1)
        c2 = c2 - c1 * np.vdot(c1, c2)
        c2 = c2 / np.linalg.norm(c2)
        cols += [c1, c2]
    if len(cols) != m:
        raise ValueError(
            "GATE trs_gauge_canonical_basis_deficient: Kramers probes "
            f"spanned {len(cols)}/{m} of {where} after exhausting all {L} "
            f"fixed-order coordinate probes.\n"
            f"  largest REJECTED residual norm {n_rej_max:.6e} "
            f"(relative {rel_rej_max:.3e} against the probe's own norm / "
            f"the family scale {probe_scale:.6e}); relative floor "
            f"{rank_criterion.PROBE_RTOL:.3e}.\n"
            f"  This test is RELATIVE (common/rank_criterion."
            f"probe_is_independent), so a small ABSOLUTE norm is not by "
            f"itself a reason to be here: an absolute floor is a "
            f"system-size-dependent refusal, and the one that used to live "
            f"here (1e-6) discarded every probe on a VALID fully "
            f"relativistic LiF WFN.\n"
            f"  If this fires now, the block genuinely does not span: the "
            f"Theta-closure gate above passed, so suspect a Kramers partner "
            f"outside the band window or a degeneracy grouping tolerance "
            f"that merged two blocks.\n"
            "  doc: bse_w_exact.enforce_trs_pair_gauge, "
            "docs/dev/rank_truncation_policy.md §3")
    V = np.stack(cols, axis=1)
    _refuse_unless_rotation_is_unitary(V, where=where, kind="Kramers")
    return np.tensordot(V.T, Psi, axes=(1, 0))


def _spin_rotation(nspinor):
    """The unitary part of TRS: Theta = R K.  Scalar: R = 1.  Spinor:
    R = i*sigma_y = [[0, 1], [-1, 0]] — REAL orthogonal, R^2 = -1 (Kramers)."""
    if nspinor == 1:
        return np.eye(1)
    if nspinor == 2:
        return np.array([[0.0, 1.0], [-1.0, 0.0]])
    raise ValueError(f"nspinor={nspinor}?")


def _theta(psi_k, R):
    """Theta psi for one k-slot (nb, ns, mu): R on the spin axis, conj."""
    return np.einsum("st,btm->bsm", R, np.conj(psi_k))


def _trs_fix_band_array(psi, eps, grid, *, label):
    """psi (nk, nb, ns, mu), eps (nk, nb) -> TRS/Kramers pair gauge on copies.

    For every k-pair (k, -k) the -k states are REPLACED by ``Theta psi(k)``
    (``Theta = i sigma_y K`` for spinors, plain K for scalars — any eigenbasis
    of H(-k) may be, since the assembled W is invariant under unitary mixing
    within the band window).  At TRIM points (k == -k) each degenerate block
    is rotated to real functions (scalar) or to the Kramers-canonical pairing
    (spinor).  Energies are symmetrized after an agreement check.

    Note the sigma_y itself is INVISIBLE to the ladder operator: every
    contraction it performs is a spin-singlet s-summed bilinear
    ``sum_s conj(psi^s) phi^s``, and R real-orthogonal cancels there
    identically — which is WHY the scalar derivation (w_ladder step 8)
    carries over.  R is kept so the overwritten arrays are genuine H(-k)
    eigenstates, not just operator-equivalent ones."""
    nk = psi.shape[0]
    ns = psi.shape[2]
    R = _spin_rotation(ns)
    psi = psi.copy()
    eps = eps.copy()
    idx = np.arange(nk).reshape(grid)
    coords = np.stack(np.unravel_index(np.arange(nk), grid), axis=1)
    neg = idx[tuple(((-coords) % np.array(grid)).T)]
    de = float(np.abs(eps - eps[neg]).max())
    if de > 1e-6:
        raise ValueError(
            f"GATE trs_gauge_energies_disagree: eps_{label}(k) vs "
            f"eps_{label}(-k) differ by {de:.3e} Ry — the deck's k-grid is "
            "not TRS-consistent, so the conj-pattern anti-resonant channel "
            "has no exact gauge. doc: bse_w_exact.enforce_trs_pair_gauge")
    eps = 0.5 * (eps + eps[neg])
    for k in range(nk):
        kn = int(neg[k])
        if k < kn:
            # Canonicalize the SOURCE slot first — span-anchored degenerate
            # blocks + largest-element band phases — so the Theta-overwrite
            # propagates a run-invariant gauge.  Upstream GPU nondeterminism
            # realizes degenerate subspaces differently per run (registered
            # defect: varying iteration counts, one spurious refusal); the
            # canonical gauge depends only on the spans and the fixed
            # conventions, so the same WFN yields the same output up to
            # upstream jitter magnitude, and identical input is bit-identical.
            e = eps[k]
            b0 = 0
            while b0 < e.size:
                b1 = b0 + 1
                while b1 < e.size and e[b1] - e[b1 - 1] < 1e-8:
                    b1 += 1
                if b1 - b0 > 1:
                    blk = psi[k, b0:b1].reshape(b1 - b0, -1)
                    psi[k, b0:b1] = _canonical_subspace_basis(blk).reshape(
                        psi[k, b0:b1].shape)
                else:
                    flat = _canonical_phase(psi[k, b0].reshape(-1))
                    psi[k, b0] = flat.reshape(psi[k, b0].shape)
                b0 = b1
            psi[kn] = _theta(psi[k], R)
        elif k == kn:
            e = eps[k]
            b0 = 0
            while b0 < e.size:
                b1 = b0 + 1
                while b1 < e.size and e[b1] - e[b1 - 1] < 1e-8:
                    b1 += 1
                # Name the k point and the band block in any refusal: the
                # registered LiF failure reported "0/2 of a TRIM block" and
                # an operator had no way to find which one.
                _where = (f"the {label}-manifold TRIM block at k={k} "
                          f"(grid index), bands [{b0}, {b1}) of "
                          f"{int(e.size)}")
                if ns == 1:
                    blk = psi[k, b0:b1, 0, :].T      # (mu, m)
                    psi[k, b0:b1, 0, :] = _realize_trim_block(
                        blk, where=_where).T
                else:
                    psi[k, b0:b1] = _kramers_canonicalize_trim_block(
                        psi[k, b0:b1], R, where=_where)
                b0 = b1
    return psi, eps


def enforce_trs_pair_gauge(data, mesh_xy):
    """Fix the Bloch gauge to ``psi(-k) = conj(psi(k))`` — the LADDER's need.

    Why this exists: the ladder operator's anti-resonant channel is built by
    the conj-pattern ``K^AA = conj(K^RR)`` on the SAME +q-rolled arrays (the
    step-4 hybrid row, ``w_ladder`` derivation).  That identification of the
    code's Y labels with the physical de-excitation pairs at ``-q`` is an
    antiunitary TRS fold, and it is EXACT only in the gauge
    ``psi(-k) = conj(psi(k))``.  The diagonalizer's gauge is arbitrary, so on
    a raw deck the Y channel carries wrong-gauge content — invisible at q=0
    (the two channels share one transition space there), invisible to per-q
    hermiticity (``conj(K_d)`` is Hermitian in ANY gauge), invisible to the
    RPA (the ring is a band-window-invariant dyad and D is real), and
    measured as a 1.043e-03 violation of ``W(-q) = conj(W(q))`` by the
    FIRST-PRINCIPLES dense ladder operator itself on the gnppm 2v2c fixture
    (2026-08-16; the production number on the closure fixture was 3.579e-04,
    claim 0215).  Fixing the gauge is a pure basis choice: every
    gauge-invariant object (RPA W, hermiticity, band energies, densities) is
    unchanged, which is why the RPA arm never needed it.

    SPINORS are handled by the same machinery with ``Theta = i sigma_y K``
    (Kramers pair gauge): the ladder operator's every contraction is a
    spin-singlet s-summed bilinear ``sum_s conj(psi^s) phi^s``, in which the
    real-orthogonal ``i sigma_y`` cancels IDENTICALLY — so the scalar step-8
    derivation carries over unchanged once the pair gauge is enforced, and a
    "drop the sigma_y" red twin is a NO-OP by the same theorem (sigma_y is
    observable only in eigenstate validity, which the wholesale pair
    overwrite makes moot).  TRIM points get the Kramers-canonical pairing
    ``phi_{2j+1} = Theta phi_{2j}`` per degenerate block (even multiplicity
    enforced).

    Cost: one host pass over psi_c/psi_v (same class as the per-q host roll
    in :func:`build_finite_q_data`); O(nk nb ns mu) + an eigh or Kramers
    sweep per degenerate TRIM block.  Returns a shallow copy with psi/eps/M
    slots replaced.
    """
    grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    sh = make_bse_shardings(mesh_xy)
    out = dict(data)
    psi_c, eps_c = _trs_fix_band_array(
        gather_to_host(data["psi_c_X"]), np.asarray(jax.device_get(data["eps_c"])),
        grid, label="c")
    psi_v, eps_v = _trs_fix_band_array(
        gather_to_host(data["psi_v_X"]), np.asarray(jax.device_get(data["eps_v"])),
        grid, label="v")
    out["psi_c_X"] = device_put_process_local(psi_c, sh.psi_x)
    out["psi_c_Y"] = device_put_process_local(psi_c, sh.psi_y)
    out["psi_v_X"] = device_put_process_local(psi_v, sh.psi_x)
    out["psi_v_Y"] = device_put_process_local(psi_v, sh.psi_y)
    out["eps_c"] = device_put_process_local(eps_c, sh.eps)
    out["eps_v"] = device_put_process_local(eps_v, sh.eps)
    out["M_X"] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(out["psi_c_X"], out["psi_v_X"]), sh.psi_x)
    out["M_Y"] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(out["psi_c_Y"], out["psi_v_Y"]), sh.psi_y)
    return out


def _symmetry_tables(input_file: str):
    """The ONE canonical ``SymMaps`` object for ``input_file``'s WFN.

    Carries the IBZ q-wedge (``q_irr_kgrid_int``) AND the full-BZ unfold tables
    (``irr_idx_q``, ``sym_idx_q``, ``q_irr_full_idx``) the ladder facade hands to
    ``symmetry_maps.unfold_isdf_operator``.  Single source: no BSE-side copy of
    the wedge arithmetic — :func:`_symmetry_reduced_q_list` is a projection of
    this, not a second construction."""
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from .bse_io import _parse_wfn_path
    return WfnLoader(_parse_wfn_path(input_file)).symmetry()


def _symmetry_reduced_q_list(input_file: str) -> np.ndarray:
    """Symmetry-reduced (IBZ) q-grid points as integer kgrid steps ``(n_q, 3)``,
    from the ONE canonical ``SymMaps`` table (``q_irr_kgrid_int``); no new sym
    helper.  Row 0 is Γ = (0,0,0)."""
    return np.asarray(_symmetry_tables(input_file).q_irr_kgrid_int, dtype=int)


def _get_block_gmres_solver(matvec, sh, max_iter, tol, dtype,
                            resid_relative_to: str = "b"):
    """Cached jitted per-column-scan block-GMRES engine for the screening
    resolvent — the stage-2 SOLVE of :func:`apply_screening_resolvent_block`.

    Returns a ``jax.jit`` function ``(rhs, diag_h, z, operands) -> (s_all, resids)``
    that scans the per-column-independent shifted GMRES over the probe axis.  The
    operator STRUCTURE (``matvec``, ``sh``, ``max_iter``, ``tol``) is baked into
    the compiled program; the q- and omega-dependent tensors (``rhs``, ``diag_h``,
    ``z``, and the ``matvec_operands`` tuple ``operands``) are RUNTIME ARGUMENTS.
    So it compiles ONCE per operator structure and every later q / omega is
    dispatch-only — replacing the old top-level ``lax.scan`` that closed over the
    per-q ``data`` and recompiled the ~4.8 s scan once per q.

    Cache key ``(id(matvec), max_iter, resid_relative_to, dtype)`` keeps the
    operator-identity safety: genuinely different structures (screening vs
    optical, TDA vs full) carry distinct ``matvec`` objects and so distinct
    engines.  ``tol`` is NOT in the key — it rides in as a runtime argument
    (it appears only in the ``while_loop`` predicate), so a tolerance sweep
    reuses one executable.  ``resid_relative_to`` IS in the key because it is
    a different stopping rule and so a different program; see
    :func:`bse_feast._gmres_solve_core` for what the two mean."""
    key = (id(matvec), int(max_iter), str(resid_relative_to), str(dtype))
    hit = _BLOCK_GMRES_CACHE.get(key)
    if hit is not None:
        return _bind_tol(hit[1], tol)

    @jax.jit
    def _block(rhs, diag_h, z, operands, tol_rt):
        # rhs: (2, nu, c, v, k) — scan the per-column GMRES over the probe axis nu.
        rhs_scan = jnp.moveaxis(rhs, 1, 0)  # (nu, 2, c, v, k)

        def _solve_col(carry, rhs_col):
            rhs_i = rhs_col[:, None]        # (2, 1, c, v, k) — keep the matvec batch axis
            # ``k_used`` is the while_loop's real exit index, not the budget:
            # ``_gmres_solve_core`` runs ``while k < max_iter and rel > tol``.
            # It used to be dropped on the floor here, which is why this — the
            # campaign's most-run shifted solve — was the one solver whose
            # iteration count no log could ever show.  Carried out for LOGGING.
            x, k_used = _gmres_solve_core(matvec, rhs_i, diag_h, z, operands,
                                          max_iter, tol_rt,
                                          resid_relative_to=resid_relative_to)
            r_true = rhs_i - _apply_shifted_matvec(matvec, x, z, operands)
            nrhs = jnp.linalg.norm(rhs_i)
            resid = jnp.where(nrhs == 0.0, jnp.asarray(0.0, dtype=nrhs.dtype),
                              jnp.linalg.norm(r_true) / nrhs)
            s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)  # (1, c, v, k)
            return carry, (s[0], resid, k_used)

        # unroll=1: ONE Krylov workspace alive at a time.  The workspace is
        # O(max_iter) pair-basis vectors plus an O(max_iter^2) replicated
        # Hessenberg, so unrolling the probe axis multiplies the solve's memory
        # peak by the unroll factor for no arithmetic saving — the ladder
        # matvec amortises only 1.11x over a 16-wide block (measured,
        # opt_integration 2026-08-16), i.e. the probe axis carries no width win
        # to buy the memory with.
        _, (s_all, resids, iters) = jax.lax.scan(_solve_col, None, rhs_scan,
                                                 unroll=1)
        s_all = jax.lax.with_sharding_constraint(s_all, sh.X)        # (nu, c, v, k)
        return s_all, resids, iters

    _BLOCK_GMRES_CACHE[key] = (matvec, _block)
    return _bind_tol(_block, tol)


def _bind_tol(block, tol):
    """Bind the runtime ``tol`` so callers keep the 4-argument call shape."""
    return lambda rhs, diag_h, z, operands: block(
        rhs, diag_h, z, operands, jnp.asarray(tol, dtype=jnp.float64))


def build_probe_rhs(G_zeta, data, gen, sh):
    """Stage 1 (SEED) alone: the probe block's pair-basis right-hand side.

    ``z``-INDEPENDENT — it reads only the probe block and the q-shifted payload
    — so a caller that sweeps frequencies at fixed ``(q, probe block)`` builds
    it ONCE here and hands it to :func:`apply_screening_resolvent_block` as
    ``rhs=``, instead of paying the reshard + generator dispatch again at every
    ``z``.  Extracted, not duplicated: the function below calls this one.

    Returns ``rhs`` on ``sh.X_full`` = ``(2, n_probe, c_X, v_Y, k)``.
    """
    px, py = sh.X.mesh.devices.shape
    n_probe = int(G_zeta.shape[0])
    if n_probe % py != 0:
        raise ValueError(
            f"probe block n_probe={n_probe} must be a multiple of py={py} "
            "(reduce-scatter tiles nu over y); pad the probe block with zero rows.")
    n_rmu = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    # Process-local (scorecard AA.1): the probe block is identical on every
    # rank; ``np.broadcast_to`` keeps the host operand a zero-copy view, so
    # each rank materialises only its own shard — where plain ``device_put``
    # of the materialised broadcast paid a P × n_probe·n_rmu·nk·8 B
    # assert_equal all-gather.  LORRAX_CHECK_REPLICA=1 re-arms the check.
    # REAL probe blocks only, refused rather than cast.  The stage-1 seed is
    # float64 all the way to ``gen``, and ``np.asarray(x, dtype=np.float64)``
    # on a complex array DISCARDS the imaginary part with a ComplexWarning —
    # not an error — so a caller handing this a complex probe block (a
    # Lanczos start vector, a rotated basis) would get a silently wrong tile.
    # Every probe in the tree today is a real unit column or the identity, so
    # this is a real assumption being stated, not a capability being removed.
    G0 = np.asarray(G_zeta)
    if np.iscomplexobj(G0):
        raise TypeError(
            "the probe block is complex; the screening seed is real-valued "
            "(float64 through the transition generator) and casting here "
            "would discard the imaginary part silently.  Split a complex "
            "probe into its real and imaginary blocks and solve both, or "
            "widen the generator's dtype deliberately.")
    G = np.asarray(G0, dtype=np.float64)
    # UPLOAD THE PROBE BLOCK, BROADCAST ON DEVICE.  ``np.broadcast_to`` is a
    # 0-stride VIEW on host, but ``device_put_process_local`` slices a
    # hyperslab out of it and numpy materialises the contiguous copy right
    # there — so uploading the broadcast paid n_probe·n_rmu·nk·8 B of host
    # allocation, memcpy and H2D for a tensor carrying only n_probe·n_rmu
    # numbers, once per (q, probe block).  Broadcasting on device divides that
    # by nk: 9x on the gnppm fixture, 216x on a 6x6x6 grid, and the seed is
    # rebuilt at every irreducible q.  MEASURED 2026-08-16
    # (evidence/sync_audit, chunk = full basis): H2D over a 5-q wedge
    # 54.7 -> 6.1 MiB, per-q staging 16.0 -> 15.5 ms at 1 process and
    # 23.1 -> 21.7 ms at 4.  Bit-identical — a broadcast is a copy.
    #
    # NOT hoisted out of the q loop, though the uploaded block IS
    # q-independent: holding one staged tensor per probe block across q costs
    # n_rmu·n_rmu·nk·8 B of DEVICE memory regardless of chunking, which is
    # exactly the budget ``probe_chunk`` exists to bound.  Measured, the hoist
    # buys a further 6.1 MiB of H2D per wedge over this; it is not worth
    # taking the memory knob back.
    g = device_put_process_local(
        np.ascontiguousarray(G, dtype=np.float64), sh.S_k0)
    r = jax.lax.with_sharding_constraint(
        jnp.broadcast_to(g[:, :, None], (n_probe, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    return jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)


def apply_screening_resolvent_block(G_zeta, z, data, matvec, diag_h, gen,
                                    snapshot, sh, *, max_iter, tol,
                                    return_iters: bool = False,
                                    operands_fn=matvec_operands,
                                    rhs=None,
                                    resid_relative_to: str = "b",
                                    solve_data=None,
                                    snapshot_v=None):
    """Screened-Coulomb resolvent on a block of probe columns — the ONE engine.

    Computes ``W(omega) - v`` tiles from the non-TDA RPA density-response
    resolvent ``v (z - H_RPA)^{-1} v`` for a whole block of centroid-space probe
    columns at once, and is the single shared implementation behind
    ``--compare-w0`` today and the future Lanczos-chain ``W(omega)`` model.

    Data flow (the three-stage seam the future model reuses verbatim):

      1. SEED (zeta -> pair, batched, ``gen`` shard_map).  Each probe column
         ``nu`` is a density-space vector ``g``; the transition generator applies
         ``v`` then the pair-density vertex, ``f = M^dag (v g)``, and the non-TDA
         RHS is the density super-vertex ``[f; -f]`` (both blocks carry the same
         ``f`` — the ring coupling makes excitation/de-excitation vertices
         coincide).  Output pair-basis block ``sh.X_full`` = ``(2, nu, c_X, v_Y, k)``,
         with ``nu`` a REPLICATED batch axis (the pair basis already tiles c on x
         and v on y — the probe axis has no free mesh axis and stays replicated
         through the solve; only the boundaries carry it as a shardable index).

      2. SOLVE (operator application).  Per-column-independent shifted GMRES via
         ``lax.scan`` over the probe axis — one Krylov subspace alive at a time
         (no N-way unrolled slot pile-up), bit-identical to the per-column loop
         it replaces because the GMRES norm/least-squares reductions are global
         per single column.  The scan stacks the readouts ``s = x[0] + x[1]``
         into ``sh.X`` = ``(nu, c_X, v_Y, k)`` and the per-column relative
         residuals.  Runs inside the cached, jitted :func:`_get_block_gmres_solver`
         engine with the q/omega-dependent tensors (``rhs``, ``diag_h``, ``z``,
         ``matvec_operands``) as RUNTIME ARGUMENTS, so it compiles ONCE per
         operator structure and every later q / omega is dispatch-only.

      3. PROJECT (pair -> zeta, batched, reduce-scatter ``snapshot``).  The
         density-snapshot vertex applies the pair density then ``v``,
         ``w(mu) = v (M s)``, and — because ``snapshot`` was built with
         ``scatter_nu_on_y=True`` — reduce-scatters the ``nu`` batch onto ``y``
         while completing the V_q0 contraction, emitting the tile directly as
         ``W(mu_X, nu_Y)`` = ``sh.V``.  No replicated ``(mu, nu)`` is ever formed.

    Block-Lanczos plug-in (future W(omega), do NOT build here).  The screening
    matvec (``build_bse_ring_matvec_full(screening=True)``) is symmetric under
    the symplectic metric, so a single block-Lanczos run seeded with the SAME
    stage-1 pair-basis block ``rhs`` (the v-probe seeds) tridiagonalizes the
    screening operator ONCE; every ``omega`` is then a small
    ``(T - z)^{-1}`` resolvent on the projected tridiagonal, and the moments are
    projected back to the zeta basis by the SAME stage-3 ``snapshot``.  Only the
    stage-2 middle swaps (scan-of-GMRES -> one block-Lanczos + per-omega tiny
    solves); stages 1 and 3 — the reshard boundaries — are reused unchanged.
    Keep any such model calling THIS function's seed/project so the layout stays
    single-sourced.

    Parameters
    ----------
    G_zeta : array (n_probe, n_rmu)
        Probe block in the padded centroid basis; row ``i`` is probe column
        ``nu_i`` (typically a unit column ``e_nu`` for a W column, or an identity
        block for the full basis).  ``n_probe`` must be a multiple of ``py`` (the
        reduce-scatter tiles ``nu`` over ``y``); pad with zero rows otherwise.

    Returns
    -------
    W_tile : jax.Array (n_rmu, n_probe), sharding ``sh.V`` = ``P('x', 'y')``
        Column ``i`` is ``W(omega) - v`` for probe ``nu_i`` in the padded centroid
        basis (``mu`` on x, ``nu`` on y).
    resids : jax.Array (n_probe,) float64
        Per-column relative GMRES residual of the shifted system (0 for zero-pad
        columns), so quadrature noise vs solver tolerance stay distinguishable.
    iters : jax.Array (n_probe,) int32 — ONLY when ``return_iters=True``
        Per-column GMRES iteration count actually taken: the ``lax.while_loop``
        exit index, so ``iters.max() == max_iter`` is the signal that columns
        were TRUNCATED at the cap rather than converged.  Opt-in because three
        in-tree gates and two other driver arms unpack the 2-tuple; a
        ``return_iters=False`` caller sees exactly the contract it always had.
    """
    # --- Stage 1: SEED (zeta -> pair), batched over the whole probe block. ---
    # z-independent, so a frequency sweep may hoist it (``rhs=``) — see
    # :func:`build_probe_rhs`, which is where it lives.
    if rhs is None:
        rhs = build_probe_rhs(G_zeta, data, gen, sh)      # (2, nu, c, v, k)

    # --- Stage 2: SOLVE via the cached jitted per-column-scan GMRES engine. ---
    # The engine is keyed on the operator STRUCTURE (matvec); the q/omega-dependent
    # operand arrays flow in as runtime args, so it compiles ONCE and every later
    # q / omega is dispatch-only.  z is passed as a device scalar (not a Python
    # complex) so a different omega stays a runtime arg, never a baked constant.
    solver = _get_block_gmres_solver(matvec, sh, max_iter, tol, rhs.dtype,
                                     resid_relative_to)
    # operands_fn must match the matvec's build: matvec_operands (10) for
    # every raw-payload operator, ladder_matvec_operands (14) for a
    # ladder_rung_slots build (bse_ring_comm).
    #
    # ``solve_data`` / ``snapshot_v`` are the ROUTE-A seam, and they are
    # argument substitutions rather than a second pipeline: hand the SOLVE a
    # payload whose ``V_q0`` is zero and the PROJECT the identity, and the
    # three stages return ``T = Pi v`` (the ring-lifted resolvent) instead of
    # ``v Pi v``; the caller then closes with the dense
    # ``W - v = v T (I - T)^{-1}`` Dyson.  The SEED keeps the physical ``v``
    # either way, so ``rhs``, the residual denominator and the iteration
    # counts mean the same thing on both routes.
    s_all, resids, iters = solver(rhs, diag_h,
                                  jnp.asarray(z, dtype=jnp.complex128),
                                  operands_fn(data if solve_data is None
                                              else solve_data))

    # --- Stage 3: PROJECT (pair -> zeta), reduce-scatter to W(mu_X, nu_Y). ---
    W_tile = snapshot(s_all, data["psi_c_Y"], data["psi_v_Y"],
                      data["V_q0"] if snapshot_v is None else snapshot_v)
    if return_iters:
        return W_tile, resids, iters
    return W_tile, resids


def _resolve_wc_columns(cols, z, data, matvec, diag_h, gen, snapshot, sh,
                        *, max_iter, tol, return_iters: bool = False):
    """Resolve the ``W(omega) - v`` columns listed in ``cols`` (head-less q=0
    tile, padded centroid space) via :func:`apply_screening_resolvent_block`.

    Builds the unit-column probe block for ``cols`` (zero-padded up to a multiple
    of ``py``), solves once, and returns the assembled device tile.

    Returns ``(W_tile[n_rmu, n_pad] sh.V, resids[n_pad])``, where column ``i`` of
    ``W_tile`` (``W_tile[:, i]``) is ``W - v`` for probe ``cols[i]``; the final
    ``n_pad - len(cols)`` columns are zero pad.  ``W_tile.sharding.spec`` is
    ``P('x', 'y')`` = ``(mu_X, nu_Y)``.  With ``return_iters=True`` the tuple
    gains the per-column GMRES iteration count — see
    :func:`apply_screening_resolvent_block` for why that is opt-in.
    """
    px, py = sh.X.mesh.devices.shape
    n_rmu = int(data["V_q0"].shape[0])
    cols = np.asarray(cols, dtype=int)
    n_pad = int(math.ceil(len(cols) / py) * py)
    G = np.zeros((n_pad, n_rmu), dtype=np.float64)
    for i, nu0 in enumerate(cols):
        G[i, int(nu0)] = 1.0
    return apply_screening_resolvent_block(
        G, z, data, matvec, diag_h, gen, snapshot, sh, max_iter=max_iter, tol=tol,
        return_iters=return_iters)


def _select_compare_cols(T, nlog, n_cols, seed):
    """A mix of the largest-||W0-V|| columns and random columns (logical range)."""
    col_norm = np.linalg.norm(T[:nlog, :nlog], axis=0)
    order = np.argsort(-col_norm)
    n_large = (n_cols + 1) // 2
    large = order[:n_large]
    rng = np.random.default_rng(seed)
    remaining = np.setdiff1d(np.arange(nlog), large)
    n_rand = min(n_cols - n_large, remaining.size)
    rand = (rng.choice(remaining, size=n_rand, replace=False)
            if n_rand > 0 else np.empty(0, dtype=int))
    return np.concatenate([large, rand]).astype(int), col_norm


def _parse_cols(col_str, n_mu, n_cols, seed):
    if col_str:
        cols = [int(x) for x in col_str.split(",") if x.strip() != ""]
        return np.array([c for c in cols if 0 <= c < n_mu], dtype=int)
    if n_cols is not None:
        rng = np.random.default_rng(seed)
        if n_cols >= n_mu:
            return np.arange(n_mu, dtype=int)
        return rng.choice(n_mu, size=n_cols, replace=False)
    return np.arange(n_mu, dtype=int)


def run_w_omega_chain_compare(
    data, mesh_xy, cols, *, freqs_ev, chain_len, chain_sweep,
    ry_to_ev, gmres_max_iter, gmres_tol, disk_tile=None, nlog=None,
    label="", print_fn=print,
):
    """Validate the Lanczos-chain W(omega) model against the shifted-solve oracle.

    Builds ONE block-Lanczos chain (:func:`w_omega_chain.build_w_omega_chain`) at
    ``chain_len`` for the probe ``cols``, then for every complex frequency in
    ``freqs_ev`` (each ``omega + i eta`` in eV — covers the static ``omega=0``,
    imaginary-axis ``i b``, and real-axis ``a + i eta`` cases) evaluates the chain
    at each truncated length in ``chain_sweep`` and compares to the per-omega
    oracle ``_resolve_wc_columns`` (fresh shifted block-GMRES).  At ``omega=0``
    also reports closure against the on-disk ``disk_tile`` (``W0 - V``), the GW
    ground truth.  Prints a per-omega convergence table (rel_err vs chain length)
    and a timing / omega-count break-even summary (the whole point: amortize the
    chain build over many frequencies).  Returns a results dict for the gate.

    ``cols`` are the probe density columns (``T``-largest + random mix); the chain
    block width is ``p = len(cols)``.  ``disk_tile`` / ``nlog`` are the head-less
    ``(W0 - V)`` numpy tile and its logical mu extent (omega=0 ground truth)."""
    from . import w_omega_chain as woc

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)
    cols = np.asarray(cols, dtype=int)
    p = len(cols)
    if nlog is None:
        nlog = int(data["n_rmu"])

    def _oracle(z):
        W_tile, resids = _resolve_wc_columns(
            cols, z, data, matvec, diag_h, gen, snapshot, sh,
            max_iter=gmres_max_iter, tol=gmres_tol)
        # gather_to_host, not device_get: W_tile is sh.V = P('x','y'), so at
        # P>1 its shards live on other processes and device_get RAISES.  The
        # helper's three arms cover the 1-GPU case with the same plain
        # device_get, so this costs nothing there.
        return (gather_to_host(W_tile), gather_to_host(resids)[:p])

    def _chain_eval(z, m_use=None):
        (W_tile,) = woc.eval_w_omega_chain(chain, data, snapshot, sh, z, m_use=m_use)
        jax.block_until_ready(W_tile)
        return W_tile

    # ---- build the chain ONCE (timed, warm) ----
    # Warm at the SAME chain_len that is about to be timed.  This used to warm
    # at min(2, chain_len), which was enough when every compiled thing in the
    # chain was an m-independent leaf; the chain step is now one program whose
    # (m,p,c,v,k) buffer makes m part of its signature, so a 2-step warm-up
    # compiles a DIFFERENT program and the timed build below would carry a
    # compile it claims not to.  MEASURED (Si 4x4x4, P=4, chain_len=32): 1.370 s
    # with the 2-step warm-up against 0.349 s genuinely warm.  The extra full
    # build is discarded work by design -- that is what a warm-up is.
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, chain_len)  # warm compile
    t0 = time.perf_counter()
    chain = woc.build_w_omega_chain(data, matvec, gen, sh, cols, chain_len)
    jax.block_until_ready(chain["V_stack"])
    t_build = time.perf_counter() - t0

    sweep = sorted({int(m) for m in chain_sweep if 1 <= int(m) <= chain_len})
    if chain_len not in sweep:
        sweep.append(chain_len)

    print_fn(f"\n=== W(omega) Lanczos-chain vs oracle{(' ' + label) if label else ''} ===")
    print_fn(f"probe cols={list(cols)}  block p={p}  chain_len={chain_len}  "
             f"chain build={t_build:.3f}s ({chain_len} matvecs)")
    print_fn("chain residual ||beta_m|| by length: " + ", ".join(
        f"m={m}:{woc.chain_residual_norm(chain, m):.2e}" for m in sweep))

    results = {"t_build": t_build, "chain_len": chain_len, "by_freq": {},
               "cols": cols, "p": p}

    # timing probes (warm) at a representative interior z
    z_probe = (1.0 + 0.2j) / ry_to_ev
    _chain_eval(z_probe)                                    # warm
    t_c0 = time.perf_counter(); _chain_eval(z_probe); t_chain_eval = time.perf_counter() - t_c0
    _oracle(z_probe)                                        # warm
    t_o0 = time.perf_counter(); _oracle(z_probe); t_oracle = time.perf_counter() - t_o0
    results["t_chain_eval"] = t_chain_eval
    results["t_oracle"] = t_oracle

    hdr = (f"{'freq(eV)':>16} {'m':>4} {'rel_vs_oracle':>14} "
           f"{'rel_vs_disk':>12} {'gmres_resid':>12}")
    print_fn(hdr)
    print_fn("-" * len(hdr))
    for w_ev in freqs_ev:
        w_ev = complex(w_ev)
        z = w_ev / ry_to_ev
        w_oracle, resid = _oracle(z)
        is_static = abs(w_ev) < 1e-12
        flabel = f"{w_ev.real:+.3f}{w_ev.imag:+.3f}i"
        for m in sweep:
            wc = gather_to_host(_chain_eval(z, m_use=m))
            rel_o = float(max(
                np.linalg.norm(wc[:nlog, i] - w_oracle[:nlog, i])
                / max(np.linalg.norm(w_oracle[:nlog, i]), 1e-300)
                for i in range(p)))
            if is_static and disk_tile is not None:
                rel_disk = float(max(
                    np.linalg.norm(wc[:nlog, i] - disk_tile[:nlog, int(cols[i])])
                    / max(np.linalg.norm(disk_tile[:nlog, int(cols[i])]), 1e-300)
                    for i in range(p)))
            else:
                rel_disk = np.nan
            print_fn(f"{flabel:>16} {m:4d} {rel_o:14.3e} "
                     f"{rel_disk:12.3e} {float(resid.max()):12.3e}")
            results["by_freq"].setdefault(flabel, {})[m] = {
                "rel_vs_oracle": rel_o, "rel_vs_disk": rel_disk,
                "gmres_resid": float(resid.max())}
        print_fn("-" * len(hdr))

    # amortization / break-even
    denom = t_oracle - t_chain_eval
    breakeven = (t_build / denom) if denom > 0 else float("inf")
    print_fn(f"\nTiming (warm): chain build {t_build:.3f}s | per-omega oracle "
             f"{t_oracle*1e3:.1f} ms | per-omega chain eval {t_chain_eval*1e3:.1f} ms")
    print_fn(f"omega-count break-even: chain wins after ~{breakeven:.1f} frequencies "
             f"(each extra omega is ~{t_oracle/max(t_chain_eval,1e-9):.0f}x cheaper "
             f"than a fresh oracle solve).")
    results["breakeven_omega_count"] = breakeven
    return results


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False,
        description="Exact W_c(omega) via the non-TDA RPA density resolvent")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--compare-w0", action="store_true",
                        help="Cross-check v(0-H_RPA)^-1 v against the restart's "
                             "(W0_qmunu - V_qmunu) q=0 tile.")
    parser.add_argument("--compare-wq", action="store_true",
                        help="Finite-q generalization of --compare-w0: loop over the "
                             "symmetry-reduced (IBZ) q-grid one at a time and cross-check "
                             "each W_q against its own (W0_qmunu - V_qmunu)[q_flat] tile.")
    parser.add_argument("--w-omega-chain", action="store_true",
                        help="Full-frequency W(omega) model: build ONE block-Lanczos "
                             "chain per q and evaluate W_q(omega)-v_q across a frequency "
                             "grid with NO per-omega solve; validate vs the shifted-solve "
                             "oracle (the amortized production path).")
    parser.add_argument("--chain-len", type=int, default=32,
                        help="Block-Lanczos chain length (blocks) for --w-omega-chain "
                             "(the accuracy knob: ~1e-6 on the imaginary axis, ~2e-4 "
                             "static at 32 on the MoS2 fixture; raise for tighter).")
    parser.add_argument("--chain-sweep", type=str, default=None,
                        help="Comma list of chain lengths for the convergence table "
                             "(default: 4,8,12,16,chain_len).")
    parser.add_argument("--chain-freqs-ev", type=str, default=None,
                        help="Comma list of complex frequencies (eV), e.g. "
                             "'0,2j,4j,1.5+0.2j'; default samples 0, imaginary axis, "
                             "and real axis + i*eta.")
    parser.add_argument("--chain-q", type=str, default=None,
                        help="Finite q as 'qx,qy,qz' kgrid steps for --w-omega-chain "
                             "(default: also do the smallest nonzero IBZ q).")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Valence bands (default: FULL chi0 window = n_occ).")
    parser.add_argument("--n-cond", type=int, default=None,
                        help="Conduction bands (default: FULL chi0 window).")
    parser.add_argument("--px", type=int, default=None,
                        help="mesh rows; default = the run's square startup "
                             "mesh")
    parser.add_argument("--py", type=int, default=None,
                        help="mesh columns; must equal --px, and px*py must "
                             "be the job's device count")
    parser.add_argument("--omega-ev", type=float, default=0.0,
                        help="Real frequency omega in eV (default: 0, static W0).")
    parser.add_argument("--eta-ev", type=float, default=0.0,
                        help="Imaginary broadening eta in eV (default: 0).")
    parser.add_argument("--cols", type=str, default=None,
                        help="Comma-separated mu indices to compute (0-based).")
    parser.add_argument("--n-cols", type=int, default=6,
                        help="Number of probe columns (compare-w0: largest-|W0-V| "
                             "+ random mix; else random).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmres-max-iter", type=int, default=200)
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV_DEFAULT)
    parser.add_argument("--out", type=str, default="Wc_exact.h5")
    args = parser.parse_args(argv)

    timing.reset()
    # Omitted --px/--py = the run's canonical square mesh, not 1x1; a given
    # shape must BE that mesh (bse_ring_comm.create_mesh_xy_from_flags).
    mesh_xy = create_mesh_xy_from_flags(args.px, args.py)
    args.px, args.py = tuple(int(n) for n in mesh_xy.devices.shape)
    restart_file = _find_restart_file(args.input)

    # BAND-WINDOW PARITY: the Casida pair basis must span the SAME band window the
    # GW chi0 (compute_screening) consumed = ALL occupied x ALL conduction bands
    # stored in the restart (n_val=n_occ, n_cond=nb-n_occ).  Default None -> pass a
    # large count so the loader clamps to the full window.  A smaller hand window
    # would drop transitions chi0 included and fail by construction.
    n_val = args.n_val if args.n_val is not None else 10**9
    n_cond = args.n_cond if args.n_cond is not None else 10**9

    with timing.section("w_exact.load"):
        data = load_bse_data_from_restart_sharded(
            restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
            input_file=args.input, inject_head=False,
            load_v_full=(args.compare_wq or args.w_omega_chain))

    n_rmu = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    z = (args.omega_ev + 1j * args.eta_ev) / args.ry_to_ev
    print(f"chi0 window: n_val={data['n_val']} n_cond={data['n_cond']} "
          f"(full occ x cond; matches GW compute_screening), "
          f"nk={int(data['nkx']*data['nky']*data['nkz'])}, N_mu={nlog} (padded {n_rmu})")
    print(f"omega={args.omega_ev} eV  eta={args.eta_ev} eV  z={z:.6e} Ry  "
          f"gmres(max_iter={args.gmres_max_iter}, tol={args.gmres_tol:g}); head-less bodies")

    if args.compare_wq:
        # Finite-q generalization: loop the symmetry-reduced (IBZ) q-grid one at a
        # time (NO batching across q), build the q-shifted screening data, resolve
        # W_q, and compare each against its OWN stored (W0_qmunu - V_qmunu)[q_flat]
        # tile (finite-q tiles are ~3% non-covariant under centroid permutation, so
        # NEVER validate by sym-unfolding between q's — always vs the own tile).
        # THE q LOOP LIVES IN ``w_ladder.sweep_q_wedge`` — this arm is a caller.
        # Function-level import: w_ladder imports this module at module level
        # (it reuses build_finite_q_data / apply_screening_resolvent_block), so a
        # module-level import here would be a cycle.  Same shape as the GW stage
        # helper's lazy `import bse`.
        from .w_ladder import sweep_q_wedge
        q_list = _symmetry_reduced_q_list(args.input)
        nky_, nkz_ = int(data["nky"]), int(data["nkz"])
        # The q-INDEPENDENT resolvent engine is built ONCE inside the sweep.
        # matvec / gen / snapshot depend only on (mesh, k-grid, pad sizes) — NOT
        # on q — so they are shared across every q; only the operand DATA (rolled
        # psi_c/eps_c, the V_qmunu[q] tile, the hoisted M's) and the
        # preconditioner diagonal change, and those flow as RUNTIME ARGS into the
        # single compiled block-GMRES engine.  Result: the engine compiles once
        # (first q) and every later q is dispatch-only (PHASE2_LOG "per-q
        # recompile elimination").
        print(f"\nFinite-q W_q resolvent cross-check: {len(q_list)} symmetry-reduced "
              f"q-points (IBZ), one at a time; each vs its own (W0-V)[q_flat] tile")
        print("shared resolvent engine (matvec/gen/snapshot) built once; per-q "
              "operands (rolled psi_c/eps_c, V_q tile, diag_h) are runtime args\n")
        # ``max_resid`` was headed ``max_gmres`` while being filled with
        # ``rr.max()`` — the max relative RESIDUAL, not an iteration count.
        # Anyone reading it as iterations was misled by the header.  The header
        # now says what the column holds, and ``max_it`` is the iteration count
        # itself, newly available because the engine stopped discarding it.
        hdr = (f"{'iq':>3} {'q (kgrid)':>11} {'q_flat':>6} {'max_rel_err':>12} "
               f"{'median':>11} {'max_resid':>11} {'max_it':>7} "
               f"{'build[s]':>9} {'solve[s]':>9}")
        print(hdr)
        print("-" * len(hdr))
        rel_by_q = []
        # Per-q state the two callbacks share: the disk tile and the probe
        # columns chosen from it, plus the build/solve wall clocks the table
        # prints.  ``_probe_blocks`` runs inside the sweep AFTER the q-shifted
        # payload exists, so ``build[s]`` still covers exactly what it used to.
        st = {"T": None, "cols": None, "t_build": 0.0, "t0": time.perf_counter()}

        def _build_hook(iq, q, dq):
            qx, qy, qz = q
            with timing.section("w_exact.wq_build"):
                W0 = gather_to_host(data["W_q"][:, :, qx, qy, qz])
                Vq = gather_to_host(data["V_q_full"][:, :, qx, qy, qz])
                st["T"] = W0 - Vq
                st["cols"], _ = _select_compare_cols(
                    st["T"], nlog, args.n_cols, args.seed)
            st["t_build"] = time.perf_counter() - st["t0"]
            st["t0"] = time.perf_counter()

        def _probe_blocks(iq, q):
            # Unit-column probe block for the selected columns, zero-padded up to
            # a multiple of py — the same block _resolve_wc_columns builds.
            cols = np.asarray(st["cols"], dtype=int)
            n_pad_probe = int(math.ceil(len(cols) / py_) * py_)
            G = np.zeros((n_pad_probe, n_rmu), dtype=np.float64)
            for i, nu0 in enumerate(cols):
                G[i, int(nu0)] = 1.0
            return [(0, len(cols), G)]

        def _on_result(iq, q, iz, zval, c0, n_real, W_tile, resids, gm_iters):
            qx, qy, qz = q
            q_flat = qx * nky_ * nkz_ + qy * nkz_ + qz
            jax.block_until_ready((W_tile, resids, gm_iters))
            t_solve = time.perf_counter() - st["t0"]
            cols, T = st["cols"], st["T"]
            # COMPARE: host-side rel_err vs the own (W0-V)[q_flat] tile.
            with timing.section("w_exact.wq_compare"):
                wc = gather_to_host(W_tile)
                rr = gather_to_host(resids)[:len(cols)]
                gi = gather_to_host(gm_iters)[:len(cols)]
                rels = np.asarray([
                    float(np.linalg.norm(wc[:nlog, i] - T[:nlog, int(nu0)])
                          / np.linalg.norm(T[:nlog, int(nu0)]))
                    for i, nu0 in enumerate(cols)])
            print(f"{iq:3d} {str((qx, qy, qz)):>11} {q_flat:6d} {rels.max():12.3e} "
                  f"{np.median(rels):11.3e} {rr.max():11.3e} {int(gi.max()):7d} "
                  f"{st['t_build']:9.3f} {t_solve:9.3f}")
            rel_by_q.append(rels.max())
            st["t0"] = time.perf_counter()

        py_ = int(mesh_xy.devices.shape[1])
        with timing.section("w_exact.resolve_q"):
            sweep_q_wedge(
                data, mesh_xy, q_list, [z], include_w=False,
                probe_blocks_for_q=_probe_blocks,
                gmres_tol=args.gmres_tol, gmres_max_iter=args.gmres_max_iter,
                on_result=_on_result, build_hook=_build_hook)
        print("-" * len(hdr))
        rel_by_q = np.asarray(rel_by_q)
        print(f"\nmax per-q rel_err = {rel_by_q.max():.3e}   median = "
              f"{np.median(rel_by_q):.3e}  (q=0 baseline ~2.5e-9). Closure at the GW "
              f"minimax-quadrature floor confirms W_q = v_q(0-H_RPA^q)^-1 v_q + v_q at "
              f"every symmetry-reduced q.")
        print("solve[s] column: first q carries the one-time engine compile; "
              "later q are warm dispatch (the per-q recompile is gone).")
        timing.report(print_fn=print, title="--- Timing ---")
        return

    if args.w_omega_chain:
        # Full-frequency W(omega) model: ONE block-Lanczos chain per q, then
        # evaluate across a frequency grid with no per-omega solve.  Validate
        # against the shifted-solve oracle at q=0 and the smallest nonzero IBZ q.
        if args.chain_freqs_ev:
            freqs = [complex(s) for s in args.chain_freqs_ev.split(",") if s.strip()]
        else:
            freqs = [0.0, 2j, 5j, 10j, 1.5 + 0.1j, 4.0 + 0.2j, 8.0 + 0.3j]
        if args.chain_sweep:
            sweep = [int(s) for s in args.chain_sweep.split(",") if s.strip()]
        else:
            sweep = [4, 8, 12, 16, args.chain_len]

        # Both are mesh-sharded (sh.W / sh.V), so this is gather_to_host and
        # not device_get -- see the note in run_w_omega_chain_compare.
        W0 = gather_to_host(data["W_q"][:, :, 0, 0, 0])
        V0 = gather_to_host(data["V_q0"])
        T0 = W0 - V0
        cols0, _ = _select_compare_cols(T0, nlog, args.n_cols, args.seed)
        with timing.section("w_exact.chain_q0"):
            run_w_omega_chain_compare(
                data, mesh_xy, cols0, freqs_ev=freqs, chain_len=args.chain_len,
                chain_sweep=sweep, ry_to_ev=args.ry_to_ev,
                gmres_max_iter=args.gmres_max_iter, gmres_tol=args.gmres_tol,
                disk_tile=T0, nlog=nlog, label="q=(0,0,0)")

        if args.chain_q is not None:
            q = tuple(int(x) for x in args.chain_q.split(","))
        else:
            q_list = _symmetry_reduced_q_list(args.input)
            nz = q_list[np.any(q_list != 0, axis=1)]
            q = (tuple(int(x) for x in nz[int(np.argmin((nz.astype(np.int64) ** 2).sum(axis=1)))])
                 if len(nz) else None)
        if q is not None and tuple(q) != (0, 0, 0):
            qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
            dq = build_finite_q_data(data, (qx, qy, qz), mesh_xy)
            W0q = gather_to_host(data["W_q"][:, :, qx, qy, qz])
            Vq = gather_to_host(data["V_q_full"][:, :, qx, qy, qz])
            Tq = W0q - Vq
            colsq, _ = _select_compare_cols(Tq, nlog, args.n_cols, args.seed)
            with timing.section("w_exact.chain_qfinite"):
                run_w_omega_chain_compare(
                    dq, mesh_xy, colsq, freqs_ev=freqs, chain_len=args.chain_len,
                    chain_sweep=sweep, ry_to_ev=args.ry_to_ev,
                    gmres_max_iter=args.gmres_max_iter, gmres_tol=args.gmres_tol,
                    disk_tile=Tq, nlog=nlog, label=f"q=({qx},{qy},{qz})")
        timing.report(print_fn=print, title="--- Timing ---")
        return

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)

    if args.compare_w0:
        # Head-less target: (W0_qmunu - V_qmunu) q=0 tile from the loaded bodies.
        W0 = gather_to_host(data["W_q"][:, :, 0, 0, 0])
        V0 = gather_to_host(data["V_q0"])
        T = W0 - V0
        if args.cols:
            cols = _parse_cols(args.cols, nlog, None, args.seed)
            col_norm = np.linalg.norm(T[:nlog, :nlog], axis=0)
        else:
            cols, col_norm = _select_compare_cols(T, nlog, args.n_cols, args.seed)
        print(f"\nW(0) resolvent cross-check: {len(cols)} columns "
              f"(largest-|W0-V| + random)\n")

        with timing.section("w_exact.resolve"):
            W_tile, resids, gm_iters = _resolve_wc_columns(
                cols, z, data, matvec, diag_h, gen, snapshot, sh,
                max_iter=args.gmres_max_iter, tol=args.gmres_tol,
                return_iters=True)
        # W_tile is the device (mu_X, nu_Y) tile; column i = probe cols[i].
        wc = gather_to_host(W_tile)                      # (n_rmu, n_pad)
        resids = gather_to_host(resids)                  # (n_pad,)
        gm_iters = gather_to_host(gm_iters)              # (n_pad,)

        hdr = (f"{'nu':>5} {'||(W0-V)_col||':>15} {'rel_err':>11} "
               f"{'max|Delta|':>12} {'gmres_resid':>12} {'gmres_it':>9}")
        print(hdr)
        print("-" * len(hdr))
        rel_all = []
        for i, nu0 in enumerate(cols):
            tcol = T[:nlog, int(nu0)]
            dcol = wc[:nlog, i] - tcol
            rel = float(np.linalg.norm(dcol) / np.linalg.norm(tcol))
            mx = float(np.max(np.abs(dcol)))
            rel_all.append(rel)
            print(f"{int(nu0):5d} {col_norm[int(nu0)]:15.4e} {rel:11.3e} "
                  f"{mx:12.3e} {resids[i]:12.3e} {int(gm_iters[i]):9d}")
        rel_all = np.asarray(rel_all)
        resids = resids[:len(cols)]
        gm_iters = gm_iters[:len(cols)]
        print("-" * len(hdr))
        print(f"max rel_err = {rel_all.max():.3e}   median = {np.median(rel_all):.3e}   "
              f"max gmres_resid = {resids.max():.3e}")
        # An iteration count EQUAL to the cap means truncation, not convergence.
        print(f"gmres iters: mean = {gm_iters.mean():.1f}, max = {int(gm_iters.max())} "
              f"(cap {args.gmres_max_iter}"
              + (" — TRUNCATED at the cap on at least one column"
                 if int(gm_iters.max()) >= args.gmres_max_iter else "") + ")")
        print("\nInterpretation: W0_qmunu on disk is the RPA static screened "
              "Coulomb W(0) from chi0 = chi0(iw) minimax-quadratured to w=0; the "
              "resolvent uses the EXACT 1/(e_c-e_v) static denominator, so rel_err "
              "is the GW minimax-integration noise (solver residual is orders "
              "smaller). Closure at this floor confirms W0 = v(0-H_RPA)^-1 v + v.")
    else:
        cols = _parse_cols(args.cols, nlog, args.n_cols, args.seed)
        print(f"\nComputing {len(cols)} W_c(omega) column(s) of N_mu={nlog}")
        with timing.section("w_exact.resolve"):
            W_tile, resids, gm_iters = _resolve_wc_columns(
                cols, z, data, matvec, diag_h, gen, snapshot, sh,
                max_iter=args.gmres_max_iter, tol=args.gmres_tol,
                return_iters=True)
        # Persist columns-first (n_cols, n_rmu), dropping the py zero-pad columns.
        wc = gather_to_host(W_tile)[:, :len(cols)].T     # (n_cols, n_rmu)
        resids = gather_to_host(resids)[:len(cols)]
        gm_iters = gather_to_host(gm_iters)[:len(cols)]
        with h5py.File(args.out, "w") as h5:
            h5.attrs["omega_ev"] = float(args.omega_ev)
            h5.attrs["eta_ev"] = float(args.eta_ev)
            h5.attrs["ry_to_ev"] = float(args.ry_to_ev)
            h5.attrs["gmres_max_iter"] = int(args.gmres_max_iter)
            h5.attrs["gmres_tol"] = float(args.gmres_tol)
            h5.attrs["kernel"] = "rpa_screening_nonTDA"
            h5.create_dataset("columns", data=cols.astype(np.int32))
            h5.create_dataset("gmres_resid", data=resids)
            h5.create_dataset("gmres_iters", data=gm_iters.astype(np.int32))
            h5.create_dataset("Wc", data=wc)
        print(f"Wrote {len(cols)} Wc columns (max gmres_resid={resids.max():.2e}, "
              f"max gmres_it={int(gm_iters.max())}/{args.gmres_max_iter}) "
              f"to {args.out}")

    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    raise SystemExit(main())
