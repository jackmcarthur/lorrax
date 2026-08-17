"""Ladder-corrected screened Coulomb ``W_BSE(z)`` — the test-charge density
response whose kernel carries the statically screened direct rung ``-W``.

This is the ``screening_diagrams = w_bse`` half of the GW driver's screening
fork: the SAME resolvent identity ``W(z) - v = v (z - H)^{-1} v`` that
``bse_w_exact`` uses for the RPA W, evaluated with the LADDER operator instead
of the RPA one.  It is a two-stage object by construction — the ladder kernel
``W_R`` is the RPA ``W(0)`` the ordinary path already produced — and it lives in
``src/bse/`` because the operator is the BSE ring matvec; ``gw/`` consumes only
the (mu, nu) tiles this module returns.

Scaling envelope (TASTE 8 / INVARIANTS 9) — MEASURED, not projected
---------------------------------------------------------------------------
Every claim in this section was checked against the optimized HLO on a 2x2
mesh with production PartitionSpecs on 2026-08-16 (the audit and its numbers:
``reports/screening_diagrams_wbse/evidence/hlo_audit/FINDINGS.md`` in the
sandbox); the three that did not survive are stated in their corrected form
and the defects they named are fixed in this file.

Cost per ``(q, z)`` point = (GMRES iterations) x (one ring matvec).  The
ladder matvec is NOT the RPA one at the same price: measured static collective
counts are 8 -> 24 (x3.0) for the matvec and 40 -> 88 (x2.2) for the whole
non-TDA solve, none of them on an ``N_mu^2``-class operand.  The whole
``q x z`` sweep dispatches ONE compiled program — the operator structure is
baked into the cached block-GMRES engine (``_get_block_gmres_solver``, keyed
on ``id(matvec)``) and every q/z-dependent tensor (rolled ``psi_c``/``eps_c``,
the ``V_q`` tile, the preconditioner diagonal, ``z``) flows in as a RUNTIME
argument through ``matvec_operands``.  That sentence was FALSE until
2026-08-16 by exactly one program: ``ensure_W_R`` sat inside the q loop and
``make_w_densifier`` builds a fresh ``@jax.jit`` per call, costing n_q + 1
trace+compiles per wedge.  ``W_R`` depends on ``k - k'`` and not on q, so it
is now built once, outside the loop (see :func:`sweep_q_wedge`), and the
sentence is true again — the audit's acceptance check is that a 2q x 2z sweep
shows no tracing-cache miss after the first ``(q, z)``.

MEMORY HIGH-WATER, in decreasing order, and none of it ``N_mu^2``-class per
rank:

* the SOLVE peak is the replicated ``O(max_iter^2)`` GMRES workspace (7.02 MiB
  at the production ``max_iter = 300``), which is ``N_mu``-INDEPENDENT and
  dominates both the RPA and the ladder solve;
* inside the matvec, the ``T / T_R / U_R / U_q / U`` direct-rung chain, not the
  probe block: the ladder matvec's peak is 5.3x the RPA one for that reason;
* the coupling rung (``bse_ring_comm._w_term_B`` -> ``_ring_sum_B_encode``)
  transiently holds a CONDUCTION-FULL trial buffer ``(b, c_full, v_loc, k)``,
  y-only-sharded, twice per matvec.  Pair-basis class, so the envelope holds,
  but it is not "``X_full`` sharded on (x, y)" and the docstring used to say it
  was.  Pre-existing and by design; the ladder is simply the first screening
  path to reach it;
* the pair-basis probe block ``X_full = (2, p_chunk, c, v, k)`` sharded on
  ``(x, y)``, plus ONE ``P('x','y')`` output tile.

There is no whole-``mu^2`` per-rank object anywhere on the path, INCLUDING the
assembly and validation tail — which is a statement this file had to earn
twice: the pad-stripped wedge return and then :func:`_assert_pad_block_is_zero`
each all-gathered a mu axis, and each is now a sharding-preserving form.  The
seed enters as ``sh.S = P(None,'y',None)``, the readout leaves through the
reduce-scatter snapshot as ``sh.V = P('x','y')``, and the stacked wedge is
``P(None, None, 'x', 'y')`` — the same class as the existing ``W_q``.

The envelope ``N_mu -> 1e4``, ``P -> 1e3`` is reached by chunking two axes and
nothing else: ``probe_chunk`` splits the mu^2 tile column-wise, and the q-wedge
is walked one q at a time (never batched).  ``probe_chunk`` trades DISPATCHES
for the probe block's memory — and only since 2026-08-16: while
:func:`_accumulate_columns` ran eagerly, every chunked placement all-gathered
the accumulator's nu axis (``N_mu/p_x`` x ``N_mu`` per rank, dividing by
sqrt(P) rather than P), so the knob documented as SAVING memory spent it, at
every setting except the single-chunk default.

Two-plan obligation (INVARIANTS 6): ONE code path serves both plans.  The ring
matvec at ``P = 1`` IS the local plan — the ring/ppermute reduces to a
no-op loop over a single rank and the shard_map bodies become the local
einsums — so there is no second implementation to keep in step, and
``docs/dev/large_nmu_operation.md`` cites this module for both rows.

The derivation
--------------
Notation: pair index ``I = (c, v, k)``; ``M_I(mu)`` the ISDF pair density
``sum_s conj(psi_c) psi_v``; ``D_I = eps_c - eps_v``; ``v`` the (mu, nu) Coulomb
tile; ``W`` the static screened tile.  The non-TDA operator acts on ``[X; Y]``
with the probe seed ``rhs = [f; -f]``, ``f = M v e_nu``, and the readout
``w = v M^dag (X + Y)`` — the convention ``bse_w_exact`` pins and gates.  The
RPA case is the symplectic ``H = [[A, B], [-B, -A]]``; the ladder case is NOT
(step 4).

**1. The Hartree/ring rung is a DYAD, and that forces it into all four
blocks with no conjugation.**  Write the seed as ``sigma u = [M u; -M u]`` and
the readout as ``tau [X; Y] = M^dag (X + Y)``.  For ANY irreducible operator
``H_0`` (the part of ``H`` that is not the ring term), adding the ring term as
the outer product ``sigma_v tau`` — i.e. ``H = H_0 + sigma v tau``, which
written out is exactly ``[[G, G], [-G, -G]]`` with ``G = M v M^dag`` — and
solving ``(z - H) x = sigma v e_nu`` gives, by Woodbury,

    w = v tau x = v [ Pi + Pi (v^{-1} - Pi)^{-1} Pi ] v = v Pi (1 - v Pi)^{-1} v,
    Pi(z) = tau (z - H_0)^{-1} sigma .

That is ``W(z) - v = v chi v`` with ``chi = Pi (1 - v Pi)^{-1}`` — the Dyson
resummation — for ANY ``H_0``, exactly, with no reality or symmetry assumption.
So the ring kernel is not "the same in both blocks" by analogy with the A
block: it is the outer product of the perturbation's injection and the density
readout, which is what an internal Hartree field IS.  With ``H_0 =
diag(D, -D)`` this gives ``Pi = chi_0`` and the RPA W, which is the identity
``bse_w_exact`` gates to 2.5e-9 at q=0 and at every symmetry-reduced finite q.
It also fixes the anti-resonant row's conjugation pattern: the dyad's bottom
row is ``[-G, -G]`` with NO complex conjugate.  The optical BSE's ``-B*, -A*``
row belongs to a different basis (its anti-resonant states carry the CONJUGATE
pair density; that is what makes its exchange ``M v M^T`` instead of
``M v M^dag``), and using it here breaks ``w = v chi v`` already at RPA.

**2. What is left to derive.**  Step 1 fixes the ring term in ALL FOUR blocks
(``[[G, G], [-G, -G]]``, no conjugate anywhere) and reduces the problem to
``H_0``, the irreducible part.  Writing ``H_0 = Sigma_3 (D~ + K^d)`` with
``Sigma_3 = diag(1, -1)``, ``D~ = diag(D, D)`` and ``K^d`` the direct kernel
MATRIX in the 2N basis, the free-propagator factorization
``L_0^{-1} = Sigma_3 (z - Sigma_3 D~)`` gives ``Pi = tau (z - H_0)^{-1} sigma``
= the ladder-corrected irreducible polarizability, PROVIDED ``K^d`` is the
kernel matrix in that basis.  So all four blocks of ``K^d`` must be derived; the
RPA case is ``K^d = 0``, ``H_0 = diag(D, -D)``, ``Pi = chi_0``.

Special case worth naming, because it is the RPA path's whole optimization
story: when ``H`` is the plain symplectic ``[[A, B], [-B, -A]]``, eliminating
``T = X - Y`` in favour of ``S = X + Y`` gives
``[z^2 (A-B)^{-1} - (A+B)] S = 2 f`` and hence

    W(z) - v = 2 v M^dag [ z^2 (A-B)^{-1} - (A+B) ]^{-1} M v .          (*)

RPA: ``A = D + G``, ``B = G`` ⇒ ``A - B = D``, ``A + B = D + 2G``, and (*) is
what ``w_omega_chain`` exploits.  The ladder breaks BOTH the form and
``A - B = D`` (step 4), so (*) does not apply to it at all.

**3. The ladder rung.**  The screened direct term is the SAME 4-point kernel as
the optical BSE's direct term.  Two independent matrix elements exist in the
pair basis, distinguished by whether the KET's electron/hole role assignment is
reversed — which is exactly what the de-excitation branch does:

    W_d   [I, J] = sum_{mu,nu} [conj(psi_c[k]) psi_c'[k']](mu) W(k-k')
                              [psi_v[k] conj(psi_v'[k'])](nu) / N_k
    W_d^B [I, J] = sum_{mu,nu} [conj(psi_c[k]) psi_v'[k']](mu) W(k-k')
                              [psi_v[k] conj(psi_c'[k'])](nu) / N_k

``W_d`` is ``_apply_A``'s ``W_term`` (``encode_T`` + ``apply_W_from_T``);
``W_d^B`` is ``_apply_B``'s (``encode_T_B`` + the SAME decode).  Both are the
optical operator's direct terms VERBATIM, and that is a consequence, not an
analogy: the direct rung contracts band pairs ``(c, c')`` and ``(v, v')``
through ``W`` and never touches a density vertex, so the density-vertex
convention that distinguishes this operator's exchange from the optical one
(step 1) cannot reach it.  MEASURED on the gnppm fixture: ``W_d`` is Hermitian
to 1.2e-12 and ``W_d^B`` is complex SYMMETRIC to 4.9e-13 (and 1.9e+00 away from
Hermitian) — the same structure ``test_bse_dense_reference`` asserts for the
optical blocks.

**4. The anti-resonant row is a HYBRID, and this is the result that had to be
measured.**  The remaining two blocks of ``K^d`` are fixed by ``K^d`` being a
kernel MATRIX, i.e. Hermitian: ``K^AR = (K^RA)^dag = conj(W_d^B)`` (using
``W_d^B`` symmetric) and ``K^AA = conj(K^RR) = conj(W_d)``.  Composed with the
step-1 dyad this gives

    A = D + V - W_d ,        B = V - W_d^B ,
    Y_out = -[ V X - conj(W_d^B conj(X)) ] - [ D Y + V Y - conj(W_d conj(Y)) ] ,

i.e. the RING terms of the bottom row keep the RPA's un-conjugated ``-V`` (they
are the Hartree dyad's bottom row, forced by step 1) while the DIRECT terms are
CONJUGATED (they are an ordinary two-particle kernel, so its anti-resonant
blocks are the para-Hermitian conjugates — the same structure the OPTICAL row
uses).  Equivalently: the ladder-only part of ``H`` is exactly the exchange-free
optical SHAO operator ``[[A, B], [-B*, -A*]]``, so the poles of ``Pi`` are the
exchange-free BSE's exciton energies, which is what a ladder polarizability's
poles must be.

Three candidate rows were dense-solved on the gnppm 2v2c payload at q=0 (probe
leg, JID 57052808, 2026-08-15), scored on (a) static-tile hermiticity
``max|W - W^dag|/|W|`` and (b) the ``W -> 0`` limit against the validated RPA W:

    row                                       asym       RPA limit
    -[B, A]        (naive symplectic)         2.13e-05   exact
    -[B*, A*]      (full optical conjugation) 9.24e-07   2.27e-03  REFUTED
    -[B_ring - conj(W_d^B), A_ring - conj(W_d)]  6.90e-15   exact    <- this one

The naive row's 2.13e-05 is the OPERATOR, not the solver: a dense (exact) solve
of the same operator reproduces it to every digit, and the sharded tile agrees
with the dense one to 3e-15 at GMRES tolerances from 1e-9 to 1e-14.  Fully
conjugating the row instead breaks the RPA limit, because ``K^A = M v M^dag`` is
Hermitian but NOT symmetric (measured ``|K^A - (K^A)^T|/|K^A| = 1.95`` on this
deck) — the ring term must stay un-conjugated.  This is the same class of defect
as the historical optical ``-B, -A`` bug (``_antiresonant_row`` docstring): an
un-conjugated row against a complex-SYMMETRIC coupling.  The dense oracle
(``tests/test_bse_w_ladder_dense.py``) carries both as RED TWINS.

**5. Consequence for the frequency path.**  The ladder operator is not the
symplectic ``[[A, B], [-B, -A]]`` at all (step 4), and even its ``A - B`` is
``D - W_d + W_d^B != D``.  The ``w_omega_chain`` z^2 reduction hard-assumes BOTH
(``w_omega_chain.py:39-46``: ``S = D^{1/2}(A+B)D^{1/2}``, seed scaled by
``D^{1/2}``, three-term recurrence in a Euclidean metric), so it is
STRUCTURALLY REFUSED for the ladder — see :func:`refuse_chain_path`.  A
generalized reduction through ``bse_nontda.make_ab_appliers``' product form is
the named optimization deferral.  v1 evaluates every frequency with its own
shifted block-GMRES solve.

**6. What hermiticity is, and is not, established.**  With the step-4 row,
``K^d`` is Hermitian by construction and ``W_ladder(q=0, 0)`` comes out
Hermitian to 6.9e-15 on the fixture — asserted by
``test_ladder_w_tile_is_hermitian_at_q0``.  At FINITE q nothing here has been
measured: the same construction should give ``W(q)^dag = W(q)`` and
``W(q) = conj(W(-q))``, but that is for ``gw/screening._gate_w`` to demonstrate
on the assembled tiles.  Run the gate; if it fires, that is evidence about the
operator, not a tolerance to loosen.

**7. Finite q: the vertex flip conjugates the rung, and the operator must
compensate.**  Every tile this module returns is a ``W(z) - v`` BODY,
head-less, in the GW ``chi_0(q)`` vertex convention — that is,
:func:`bse_w_exact.build_finite_q_data`'s conjugation of BOTH psi legs, whose
long correctness note (``bse_w_exact.py:199-235``) is the record of the
finite-q vertex bug (KNOWN_FAILURES:1248) that was invisible at q=0 because
``chi_0(0)`` is real.  The flip is exact for the four DENSITY vertices (each
is bilinear in ``(psi_c, psi_v)`` with exactly one conj, so conjugating both
legs flips all four at once).  The direct rung is NOT such an object: it is
bilinear in ``(c, c')`` and ``(v, v')`` band pairs, each psi appearing once
conjugated and once plain, so wholesale conjugation of psi conjugates the
rung ELEMENTWISE — ``rung_flipped(X) = conj(rung_physical(conj(X)))``,
exactly, given ``W_R`` elementwise-real (equivalent to the input W's gated
``W(-q) = conj(W(q))`` reciprocity; no other property is used).  Composed
with the step-4 hybrid row, an uncompensated flip therefore hands the
resonant row the ANTI-resonant rung and vice versa — the operator becomes a
chimera (ring resumming ``chi(q)``, rungs belonging to the folded ``-q``
channel) that is value-identical at q=0, where the static W is real
symmetric under TRS, and violates ``W(-q) = conj(W(q))`` at finite q.
MEASURED before the compensation existed: 3.579e-04 against 4.081e-11 for
the RPA operator through the identical pair of solves (claim 0215,
JID 57064957, 2026-08-15).  The compensation lives inside the rung appliers
(``bse_ring_comm`` ``_w_term_A``/``_w_term_B`` under
the physical rung operand slots (``ladder_rung_slots``), so BOTH rows see the physical rung and
the step-4 row pattern applies verbatim at every q; with it, the operator at
``-q`` is the elementwise conjugate of the operator at ``+q`` under the TRS
fold ``k -> -k`` (ring: ``M``, ``v_q`` conjugate; ``D`` invariant; rungs:
shown above), which is what makes ``W(-q) = conj(W(q))`` hold by
construction.  [AMENDED 2026-08-16: the conj-wrap compensation described
above was REFUTED at block level (probe_block_compare: ~7e-5 residual
against the dense operator despite exactly-conjugate operand arrays and
exactly-real W_R; the elementwise argument does not survive contact with
the kernels).  The mechanism that IS measured exact (1e-20, all four
blocks) is supplying the rung with the physical rolled un-flipped arrays
as four extra operands — ``ladder_rung_slots`` in ``bse_ring_comm``,
slots from ``build_finite_q_data``, threading via
``bse_feast.ladder_matvec_operands``.  No caller-side convention
declaration remains.]

**8. Finite q, second condition: the TRS pair GAUGE.**  Step 7's fold — and
step 4's ``K^AA = conj(K^RR)`` before it — identifies the code's Y labels
with the physical de-excitation pairs at ``-q`` through the antiunitary map
``psi(-k) -> conj(psi(k))``.  That is an identity of PHYSICAL states only in
the gauge ``psi(-k) = conj(psi(k))``; the diagonalizer's Bloch gauge is
arbitrary per k, so on a raw deck the conj-pattern row fills the Y channel
with wrong-gauge content.  The error is invisible at q=0 (one shared
transition space), invisible to per-q hermiticity (``conj(K_d)`` is
Hermitian in any gauge), and invisible to the RPA (ring dyad and D are
band-window/gauge invariant — which is why the RPA arm passes reciprocity
at 4e-11 on the same deck).  MEASURED 2026-08-16: the first-principles
dense ladder operator itself — no production kernels, no payload flip —
violates ``W(-q) = conj(W(q))`` at 1.043e-03 on the raw-gauge gnppm 2v2c
fixture; fixing only the rung compensation of step 7 left the production
statistic at its pre-fix 3.579e-04 (the swap transposes the tile, and the
statistic is transpose-invariant).  The cure is a pure basis choice:
``bse_w_exact.enforce_trs_pair_gauge`` overwrites ``psi(-k)`` with
``conj(psi(k))`` pairwise (legitimate because W is invariant under unitary
band mixing within each window), realizes real functions on degenerate
blocks at TRIM points, and symmetrizes the energies.  ``sweep_q_wedge``
applies it on the ``include_w=True`` branch only, so the RPA path is
bit-untouched.  SPINOR decks use the same machinery with the Kramers pair
gauge ``psi(-k) = i sigma_y conj(psi(k))``: every contraction this operator
performs is a spin-singlet s-summed bilinear, in which the real-orthogonal
``i sigma_y`` cancels identically, so the scalar derivation carries over
once the pair gauge (and Kramers-canonical TRIM pairing) is enforced —
see ``enforce_trs_pair_gauge``.

**9. The rung sign, measured (2026-08-16).**  Every gate above is blind to
the ABSOLUTE sign of the direct rung: the RPA limit, hermiticity,
reciprocity and the dense oracle (same block spec) all pass identically
under ``+W_d``.  The sign was therefore pinned by two sign-SENSITIVE
observables on physically-signed kernels (PSD ``v`` and ``W``;
``tests/test_bse_w_ladder_sign.py``, evidence ``sign_probe/`` in the
sandbox): (i) the exchange-free ladder poles sit BELOW the free e-h
continuum — attraction — and a flipped rung sits above; (ii) on the real
scalar-Si restart the production path ENHANCES screening at q=0 AND finite
q (Loewner-negative ``Wc`` delta, 4-8%), the Shishkin-Marsman-Kresse
direction.  Consequently the measured gap OPENING of the first A/B (claim
0228) is an observable of the v1 SCOPE — ladder body with the q=0
head/wings held at RPA — not of the operator sign: the full-W
gap-shrink results in the literature include the vertex in the long-range
head, which v1 deliberately defers.  Do not "fix" the sign to match the
full-W expectation; re-examine after the ladder head lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap().
# MUST run before this module's own `import jax` (same rule as bse_w_exact).
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import (ensure_W_R, build_preconditioner_diagonal_sharded,
                        ladder_matvec_operands, matvec_operands)
from .bse_io import load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec_full,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
    make_bse_shardings,
)
from .bse_w_exact import (
    _build_rpa_resolvent,
    apply_screening_resolvent_block,
    build_probe_rhs,
    build_finite_q_data,
    enforce_trs_pair_gauge,
    _symmetry_tables,
)
from common.collectives import gather_to_host

jax.config.update("jax_enable_x64", True)


def build_ladder_resolvent(mesh_xy: Mesh, data: dict, *, include_w: bool = True,
                           vertex_flipped: bool = None,
                           fuse_ladder_rung: bool = True):
    """Assemble the LADDER screening resolvent stack for ``data``.

    The :func:`bse_w_exact._build_rpa_resolvent` analog with the ladder kernel.
    Returns ``(matvec, diag_h, gen, snapshot, sh)`` — the exact 5-tuple
    :func:`bse_w_exact.apply_screening_resolvent_block` consumes, which is
    operator-agnostic: only this assembly changes between ``w_rpa`` and
    ``w_bse``.

    ``include_w=False`` DELEGATES to ``_build_rpa_resolvent``.  That is
    deliberate rather than a duplicated branch: the wiring-closure test rides
    this flag to prove the new plumbing reproduces the production RPA W, so the
    two must be the same object graph and not merely the same arguments.

    ``include_w=True`` needs ``data['W_q']`` — the converged RPA ``W(0)`` from
    the restart — because ``ensure_W_R(..., include_W=True)`` inverse-FFTs it
    into the real-space kernel the direct rung convolves.  That is the feature's
    two-stage structure, not an implementation accident.

    ``matvec`` is q-INDEPENDENT (it closes over mesh and k-grid only), so build
    it ONCE and let the per-q operands flow through ``matvec_operands``; a
    per-q rebuild silently costs one full compile per q because the block-GMRES
    cache is keyed on ``id(matvec)``.

    The LADDER matvec is built with ``ladder_rung_slots=True``: its direct
    rung consumes four extra operands — the PHYSICAL (rolled, un-flipped) psi
    arrays — which ``build_finite_q_data`` supplies and
    ``bse_feast.ladder_matvec_operands`` threads.  On a raw payload (the q=0
    identity/oracle cells) the fallback aliases the density arrays, which ARE
    physical there.  The engine call must pass
    ``operands_fn=ladder_matvec_operands`` to match (claim 0215's defect —
    the rung on conjugated arrays — is structurally unreachable this way).
    """
    if vertex_flipped is not None:
        # DEPRECATED 2026-08-16, accepted and not consulted: payload-flip
        # handling moved from a conj-wrap compensation (refuted by
        # probe_block_compare) to the physical rung operand slots
        # (ladder_rung_slots), which build_finite_q_data supplies and the
        # fallback aliases on raw payloads — both conventions are now served
        # without a caller-side declaration.  Retained one release so the
        # in-flight w_ladder_precond callers keep working; remove with them.
        pass
    if not include_w:
        return _build_rpa_resolvent(mesh_xy, data)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    ensure_W_R(data, include_W=True, mesh_xy=mesh_xy)
    # ``fuse_ladder_rung`` is a COST switch, not a physics one: it sums the two
    # rung T intermediates of each block row before the (mu, nu, s, s, k) FFT
    # chain instead of after, so the matvec runs that chain twice instead of
    # four times (bse_ring_comm._impl_core_fused).  Same operator to rounding;
    # False is the A/B arm.
    matvec = build_bse_ring_matvec_full(
        mesh_xy, nkx, nky, nkz, include_W=True, screening=True,
        ladder_rung_slots=True, fuse_ladder_rung=fuse_ladder_rung)
    diag_h = build_preconditioner_diagonal_sharded(
        data, mesh_xy, include_W=True, use_tda=False)
    gen = build_realspace_random_transition_generator(
        mesh_xy, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
    snapshot = build_density_snapshot_operator(
        mesh_xy, nkx, nky, nkz, scatter_nu_on_y=True)
    return matvec, diag_h, gen, snapshot, make_bse_shardings(mesh_xy)


def refuse_chain_path(include_w: bool) -> None:
    """Refuse the ``w_omega_chain`` z^2 reduction for the ladder operator.

    Wired-but-refused in the ``head_channel.refuse_if_unimplemented`` style: the
    path is named, the reason is the algebra, and the generalization is a named
    deferral rather than a half-build.
    """
    if not include_w:
        return
    raise NotImplementedError(
        "w_omega_chain is refused for the ladder operator (include_w=True). "
        "Its z^2 reduction assumes A - B = D EXACTLY (w_omega_chain.py:39-46 — "
        "S = D^{1/2}(A+B)D^{1/2}, seed and readout scaled by D^{1/2}), which "
        "holds for the RPA screening operator (A = D + V, B = V) and is broken "
        "by the ladder: A - B = D - W_d + W_d^B. Using the chain here would "
        "silently evaluate a DIFFERENT operator at every frequency, not an "
        "approximation of this one. The reduction generalizes with "
        "(A-B)^{1/2} in place of D^{1/2}, applied through "
        "bse_nontda.make_ab_appliers (bse_nontda.py:288) — that is the named "
        "optimization deferral and it is not half-built here. v1 evaluates "
        "each frequency with its own shifted block-GMRES solve "
        "(bse_w_exact.apply_screening_resolvent_block).")


@dataclass(frozen=True)
class WLadderWedge:
    """The ladder ``W(z) - v`` bodies on the irreducible q-wedge.

    ``wc``
        complex128 ``(nz, nq_irr, mu_pad, mu_pad)`` on the PADDED centroid
        extent, sharded ``NamedSharding(mesh_xy, P(None, None, 'x', 'y'))`` —
        the same class as the existing full-BZ ``W_q``, so no rank ever holds a
        whole mu^2 tile.  ``wc[iz, iq, mu, nu]`` is ``W_q(z) - v_q`` for
        ``z = z_list_ry[iz]``, ``q = q_irr_kgrid_int[iq]``.  Head-LESS bodies
        (the resolvent carries no q=0 head by construction; the loader runs
        ``inject_head=False``), in the GW ``chi_0(q)`` vertex convention.  The
        pad rows/columns are ZERO: the probe block is the identity in the padded
        basis and the loader's psi pad is zero, so a pad column solves to zero
        rather than to noise.

        PADDED, NOT PAD-STRIPPED — a deviation from DESIGN_2026-08-15 section 4,
        stated loudly because that document says "pad stripped".  The two halves
        of the design's own sentence are incompatible: ``n_rmu`` is a physical
        centroid count with no divisibility (399 on the gnppm fixture) and
        ``P(None, None, 'x', 'y')`` requires the last two axes to divide the
        mesh, so a stripped tile CANNOT carry the documented sharding on any
        mesh wider than 1x1.  MEASURED 2026-08-15 (JID 57064957, the
        wiring-closure gate's first 2x2 run): ``ValueError: ... global size of
        its dimension 2 should be divisible by 2, but it is equal to 399``.
        Padding wins over sharding-by-slice because the alternative — dropping
        the constraint and letting XLA pick a layout for a 399-wide axis — is a
        replicated mu^2 object per rank, the one thing the scaling doctrine
        forbids.  Nothing downstream loses: the GW consumer's ``_pad_mu``
        re-pads to exactly this extent, ``V_q`` and the unfold tables already
        live there, and ``n_rmu`` below carries the logical count.
    ``n_rmu``
        the LOGICAL centroid count, i.e. where the pad begins.  Carried so a
        consumer that needs the physical sub-tile can slice it on the host side
        of a gather instead of inferring it from a shape.
    ``z_list_ry``
        the complex frequencies in Rydberg, as passed.
    ``q_irr_kgrid_int``
        ``(nq_irr, 3)`` integer kgrid steps, row 0 = Gamma — the SAME canonical
        ``SymMaps`` table the GW producer uses.
    ``irr_idx_q`` / ``sym_idx_q`` / ``q_irr_full_idx``
        the full-BZ unfold tables from ``SymMaps``, carried so the caller can
        hand them straight to ``symmetry_maps.unfold_isdf_operator`` without a
        second symmetry pass.
    ``gmres_resid`` / ``gmres_iters``
        ``(nz, nq_irr, n_probe)`` host arrays — per-COLUMN relative residual and
        while-loop exit index.  The caller gates on these, never on a return
        code: ``iters == gmres_max_iter`` means the column was TRUNCATED at the
        cap, not converged (the rc=0 class, QUALITY_PATTERNS 7).
    ``include_w``
        which operator produced the tiles; stamp it on any artifact these enter
        (QUALITY_PATTERNS 10 — provenance).
    """

    wc: jax.Array
    n_rmu: int
    z_list_ry: np.ndarray
    q_irr_kgrid_int: np.ndarray
    irr_idx_q: np.ndarray
    sym_idx_q: np.ndarray
    q_irr_full_idx: np.ndarray
    gmres_resid: np.ndarray
    gmres_iters: np.ndarray
    include_w: bool


def _identity_probe_block(n_pad: int) -> np.ndarray:
    """Full-basis probe block: the identity in the PADDED centroid basis.

    ``n_pad`` is already a multiple of ``px*py`` (``padded_mu_extent``), hence of
    ``py`` — the divisibility the reduce-scatter snapshot requires.
    """
    return np.eye(n_pad, dtype=np.float64)


def sweep_q_wedge(
    data: dict,
    mesh_xy: Mesh,
    q_list,
    z_list_ry,
    *,
    include_w: bool,
    probe_blocks_for_q: Callable[[int, tuple], Sequence],
    gmres_tol: float,
    gmres_max_iter: int,
    on_result: Callable,
    build_hook: Optional[Callable] = None,
    route: str = "plain",
):
    """Walk the q-wedge x probe-chunk x z-list with ONE compiled engine.

    This is the single per-q loop of the screening resolvent — the one
    ``bse_w_exact.main --compare-wq`` used to carry inline and now calls.  There
    is no second loop: ``--compare-wq`` differs only in
    ``probe_blocks_for_q`` (it probes the largest-``|W0-V|`` columns of each q's
    own disk tile, one block) and in ``on_result`` (it prints a comparison row
    instead of placing a chunk into a tile).

    The engine (``matvec``, ``gen``, ``snapshot``) depends on mesh and k-grid
    ONLY, so it is built ONCE here, outside every loop; the q-dependent operands
    (rolled ``psi_c``/``eps_c``, the ``V_q`` tile, the hoisted ``M``s, the
    preconditioner diagonal) and ``z`` flow into the cached block-GMRES engine
    as runtime arguments.  The first (q, chunk, z) point carries the compile;
    every later one is dispatch.  Rebuilding the engine per q — or per chunk —
    silently pays one full compile each time, because the block-GMRES cache is
    keyed on ``id(matvec)``.

    ``probe_blocks_for_q(iq, q)`` returns a sequence of ``(c0, n_real, G)``:
    ``G`` is the ``(n_probe, n_rmu)`` probe block (``n_probe % py == 0``),
    ``n_real`` how many of its rows are live (the rest are zero pad), and ``c0``
    the caller's own offset, passed back untouched.
    ``on_result(iq, q, iz, z, c0, n_real, W_tile, resids, iters)`` gets the
    DEVICE tile still sharded ``P('x','y')``; ``build_hook(iq, q, dq)`` is
    called once per q after the q-shifted payload exists.

    ``route='lift'`` runs route A — the Hartree ring removed from the Krylov
    problem by Woodbury — and then the tile ``on_result`` receives is
    ``T = Pi v``, NOT ``W - v``: the caller must close it with
    ``w_ladder_precond.dyson_close_tile`` (``compute_wc_qwedge`` does).  The
    sweep does not close it itself because the close is once per (q, z) over
    the whole assembled tile, i.e. after the probe chunks have been placed.
    """
    # ``build_ladder_resolvent`` -> ``ensure_W_R(data, ...)`` populates
    # ``data['W_R']`` ONCE, here, OUTSIDE the q loop.  That is a correctness-
    # neutral, cost-material distinction:
    #
    #   * ``W_R = ifft_k(W_q)`` is a function of the INTERNAL momentum k - k'
    #     and of nothing else.  ``build_finite_q_data`` returns ``dict(data)``
    #     with only the conduction slots and ``V_q0`` swapped — it never
    #     touches ``W_q`` — so every q's ``W_R`` is the SAME array, and the
    #     shallow copy already carries it.
    #   * calling ``ensure_W_R`` per q was not merely recomputing it: it goes
    #     through ``bse_io.make_w_densifier``, which builds a FRESH ``@jax.jit``
    #     each call (``bse_densify.py:195``), so the tracing cache missed on
    #     "never seen function: _densify" at every q and the wedge paid n_q + 1
    #     trace+compiles.  MEASURED 2026-08-16 by the HLO/dispatch audit
    #     (``jax_explain_cache_misses``, evidence/hlo_audit/FINDINGS.md F1):
    #     4 jit__densify modules for a 2-q sweep, and exactly one cache miss,
    #     at the second q.
    #
    # This is what makes the module docstring's "the whole q x z sweep
    # dispatches ONE compiled program" true rather than aspirational.  The
    # PRECONDITIONER stays inside the loop: it reads the rolled ``eps_c`` and
    # the finite-q ``V_q0``, so it genuinely is per-q.
    if route not in ("plain", "lift"):
        raise ValueError(f"route must be 'plain' or 'lift', got {route!r}")
    # The ladder's anti-resonant channel is the conj-pattern TRS fold of the
    # -q de-excitation space, exact only in the psi(-k) = conj(psi(k)) gauge
    # — enforce it (pure basis choice; every gauge-invariant object is
    # untouched, which is why the RPA arm below skips it and stays
    # bit-identical).  Measured: without it the FIRST-PRINCIPLES dense ladder
    # operator itself breaks W(-q) = conj(W(q)) at 1.043e-03 (2026-08-16).
    if include_w:
        data = enforce_trs_pair_gauge(data, mesh_xy)
    matvec, _, gen, snapshot, sh = build_ladder_resolvent(
        mesh_xy, data, include_w=include_w)
    # ROUTE A (the Woodbury ring lift), as two constant tiles and nothing else.
    # The Hartree ring is the rank-N_mu dyad ``sigma v tau``; removing it from
    # the Krylov problem EXACTLY is (i) zero the ``V_q0`` the MATVEC reads —
    # only ``apply_V_ring_only`` reads it — and (ii) project with the IDENTITY
    # so the tile is ``T = Pi v`` and not ``v Pi v``.  The caller closes with
    # ``W - v = v T (I - T)^{-1}``.  MEASURED (opt_precond 2026-08-16, and
    # reproduced by opt_integration): 14.5 -> 11.1 GMRES iterations at z=0 and
    # a 60x SMALLER true residual at the same nominal tolerance, because with
    # the ring gone the projected residual GMRES exits on is honest.  The
    # per-iteration cost is unchanged (the ring is ~4 % of the ladder matvec).
    lift = (route == "lift")
    if lift:
        n_pad = int(data["V_q0"].shape[0])
        zero_v = jax.lax.with_sharding_constraint(
            jnp.zeros((n_pad, n_pad), dtype=data["V_q0"].dtype), sh.V)
        eye_v = jax.lax.with_sharding_constraint(
            jnp.eye(n_pad, dtype=data["V_q0"].dtype), sh.V)
    if "W_R" not in data:                              # pragma: no cover
        raise RuntimeError(
            "sweep_q_wedge: build_ladder_resolvent did not populate "
            "data['W_R'].  The per-q loop below relies on the shallow copy "
            "build_finite_q_data returns carrying it; re-adding an in-loop "
            "ensure_W_R would restore the per-q densifier retrace instead.")
    z_list_ry = np.asarray(z_list_ry, dtype=np.complex128)
    for iq, qv in enumerate(q_list):
        q = (int(qv[0]), int(qv[1]), int(qv[2]))
        dq = build_finite_q_data(data, q, mesh_xy)
        if include_w and "psi_c_W_X" not in dq:    # pragma: no cover
            raise RuntimeError(
                "sweep_q_wedge built a ladder_rung_slots engine but "
                "build_finite_q_data stopped supplying the physical rung "
                "operands (psi_*_W_*); the rung would fall back to the "
                "conjugated density arrays (claim 0215's defect class).")
        # The preconditioner reads the flipped psi arrays; the rung's diagonal
        # is real (|psi_c|^2 W |psi_v|^2 with W_R real), so the flip is inert
        # here and no compensation is needed.
        # The lift's preconditioner is the SAME builder on the ring-less
        # payload: with the ring gone the exchange contraction drops out and
        # what is left is Delta_E - W_d (or Delta_E alone at include_w=False,
        # where z - H_0 IS the operator and the solve is exact in ONE step).
        solve_dq = dict(dq)
        if lift:
            solve_dq["V_q0"] = zero_v
        diag_hq = build_preconditioner_diagonal_sharded(
            solve_dq, mesh_xy, include_W=include_w, use_tda=False)
        if build_hook is not None:
            build_hook(iq, q, dq)
        for c0, n_real, G in probe_blocks_for_q(iq, q):
            # NOT cast to float64 here: ``build_probe_rhs`` REFUSES a complex
            # probe block by name, and a cast on this line would discard its
            # imaginary part first (numpy's complex -> float is a warning, not
            # an error).  The seed is real-valued by construction — every
            # probe in the tree is a unit column or the identity — and that is
            # now an assumption the code states rather than one it enforces by
            # silently losing half the operand.
            # SEED HOIST: stage 1 reads the probe block and the q-shifted
            # payload and NOTHING z-dependent, so it is built once per
            # (q, probe block) rather than once per (q, block, z).  Three
            # dispatches per (q, z) on the RPA arm — the seed rebuilds M_X
            # locally and costs ~27 RPA-matvec-equivalents there (opt_shifts,
            # 2026-08-16) — and free on the ladder arm, where the matvec
            # dwarfs it.
            rhs = build_probe_rhs(G, dq, gen, sh)
            for iz, z in enumerate(z_list_ry):
                W_tile, resids, iters = apply_screening_resolvent_block(
                    G, complex(z), dq, matvec, diag_hq, gen, snapshot, sh,
                    max_iter=gmres_max_iter, tol=gmres_tol, return_iters=True,
                    rhs=rhs,
                    solve_data=(solve_dq if lift else None),
                    snapshot_v=(eye_v if lift else None),
                    operands_fn=(ladder_matvec_operands if include_w
                                 else matvec_operands))
                on_result(iq, q, iz, complex(z), c0, n_real,
                          W_tile, resids, iters)


def compute_wc_qwedge(
    restart_path: str,
    z_list_ry,
    mesh_xy: Mesh,
    *,
    include_w: bool = True,
    gmres_tol: float,
    gmres_max_iter: int,
    probe_chunk: Optional[int] = None,
    input_file: Optional[str] = None,
    route: str = "auto",
    allow_replicated_dyson: bool = False,
) -> WLadderWedge:
    """Ladder ``W(z) - v`` bodies for every irreducible q and every requested z.

    The facade the GW stage helper calls.  ``z_list_ry`` are complex Rydberg
    frequencies (cohsex: ``{0}``; gn_ppm: ``{0, i w_p}``; mpa: its sample plan);
    the result is a :class:`WLadderWedge`.

    ``probe_chunk`` is the memory knob: columns of the mu^2 tile solved per
    block, defaulting to the whole (padded) basis in one block.  It must be a
    multiple of ``py`` because the reduce-scatter snapshot tiles the probe axis
    over ``y``.  Chunking costs one extra dispatch per chunk and nothing else —
    the compiled engine is shared, since ``n_probe`` is the only shape that
    changes and a chunked sweep uses ONE ``n_probe`` for all chunks except a
    final short one (which is zero-padded up, not re-shaped).

    CONTRACT NOTE (deviation, stated loudly): the frozen facade in
    DESIGN_2026-08-15 section 4 takes ``restart_path`` alone.  The symmetry
    wedge cannot be derived from the restart — ``SymMaps`` needs the WFN, which
    is named by the INPUT DECK — and the sharded loader needs the same deck to
    resolve the k-grid for flat-q restart layouts.  ``input_file`` is therefore
    an ADDITIVE keyword; every call in the frozen contract's shape still type-
    checks, and omitting it raises naming exactly what is missing rather than
    guessing a deck path next to the restart.

    ``route`` — which resolvent the solve runs.

    * ``"lift"`` — route A: the Hartree ring is removed from the Krylov
      problem by Woodbury and put back with ONE dense ``N_mu`` Dyson,
      ``W - v = v T (I - T)^{-1}``.  Fewer iterations (14.5 -> 11.1 at z=0,
      measured) at unchanged per-iteration cost, and a strictly SMALLER true
      residual at the same nominal tolerance.
    * ``"plain"`` — the full operator with the in-tree diagonal
      preconditioner: the historical path, kept selectable for A/B and used
      wherever the closing Dyson cannot run.
    * ``"auto"`` (default) — ``"lift"`` on a single device, ``"plain"``
      above it, because the closing Dyson is ``jnp.linalg.solve`` and
      materialising the ``N_mu x N_mu`` tile per rank is the replication the
      scaling doctrine forbids.  ``allow_replicated_dyson=True`` takes
      ``"lift"`` anyway and accepts exactly the per-rank memory class the RPA
      Dyson's own default (``w_dyson_solver = local``, ``gw.w_isdf.solve_w``)
      already has; routing the close through ``distrib_la`` — the twin of that
      solver's ``distributed`` plan — is the registered follow-on, named here
      rather than half-built.
    """
    if input_file is None:
        raise ValueError(
            "compute_wc_qwedge needs input_file= (the COHSEX/GW input deck) in "
            "addition to restart_path: the irreducible q-wedge comes from "
            "SymMaps, which is built from the WFN named by the deck, and the "
            "sharded BSE loader resolves the k-grid from it for flat-q restart "
            "layouts. Pass the same deck the GW driver was run with.")
    px, py = mesh_xy.devices.shape
    z_list_ry = np.asarray(z_list_ry, dtype=np.complex128)
    if z_list_ry.ndim != 1 or z_list_ry.size == 0:
        raise ValueError(
            f"z_list_ry must be a non-empty 1-D list of complex Rydberg "
            f"frequencies; got shape {z_list_ry.shape}.")
    # Argument checks BEFORE the restart read: a caller that got probe_chunk
    # wrong should learn it in milliseconds, not after a multi-GB sharded load.
    if probe_chunk is not None and (int(probe_chunk) <= 0
                                    or int(probe_chunk) % py != 0):
        raise ValueError(
            f"probe_chunk={probe_chunk} must be a positive multiple of py={py} "
            "(the reduce-scatter snapshot tiles the probe axis over y).")

    # FULL chi0 band window on both legs — the pair basis must span exactly the
    # window the GW chi0 consumed, or the ladder resolvent is solving a smaller
    # problem than the W_R kernel it is fed (band-window parity, bse_w_exact.main).
    data = load_bse_data_from_restart_sharded(
        restart_path, n_val=10**9, n_cond=10**9, mesh_xy=mesh_xy,
        input_file=input_file, inject_head=False, load_v_full=True)

    sym = _symmetry_tables(input_file)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    n_pad = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    eye = _identity_probe_block(n_pad)
    chunk = n_pad if probe_chunk is None else min(int(probe_chunk), n_pad)

    # Probe block ROWS are the probe columns (apply_screening_resolvent_block
    # takes (n_probe, n_rmu) with row i the density-space probe vector), so the
    # full basis enters as the identity and a chunk as a row slice of it.  A
    # short final chunk is zero-PADDED up to `chunk` rather than reshaped, so
    # every chunk hits the same compiled engine.
    #
    # THE WALK STOPS AT ``nlog``, NOT AT ``n_pad``, and that is a correctness
    # requirement rather than a saved solve.  A probe on a PAD centroid has an
    # identically zero right-hand side (the loader zero-pads psi, so
    # ``f = M^dag (v g_pad) = 0``), and the shifted GMRES normalises by
    # ``||r0||``: 0/0 makes the whole column NaN, which then survives every
    # later step because ``0 * NaN`` is NaN, not 0.  MEASURED 2026-08-15 (JID
    # 57064957, the wiring-closure gate at 2x2): with the identity taken over
    # all 400 padded columns the assembled tile carried ``nan+nanj`` at every
    # pad entry while all 399 logical columns were correct.  The pad columns of
    # ``W - v`` are ZERO by construction — there is no such centroid — and the
    # accumulator is allocated zero, so not solving them IS the right value.
    # The same reasoning covers the zero ROWS of a short final chunk: their
    # columns are NaN inside ``W_tile`` and ``_accumulate_columns`` takes only
    # the first ``n_real``.
    blocks = []
    for c0 in range(0, nlog, chunk):
        n_real = min(chunk, nlog - c0)
        G = np.zeros((chunk, n_pad), dtype=np.float64)
        G[:n_real, :] = eye[c0:c0 + n_real, :]
        blocks.append((c0, n_real, G))

    nz, nq = int(z_list_ry.size), int(len(q_list))
    tiles = [[None] * nq for _ in range(nz)]
    resid = np.zeros((nz, nq, n_pad), dtype=np.float64)
    iters = np.zeros((nz, nq, n_pad), dtype=np.int32)

    if route not in ("auto", "plain", "lift"):
        raise ValueError(
            f"route must be 'auto', 'plain' or 'lift', got {route!r}")
    n_dev = int(np.prod(mesh_xy.devices.shape))
    if route == "auto":
        route = "lift" if (n_dev == 1 or allow_replicated_dyson) else "plain"
    if route == "lift" and n_dev > 1 and not allow_replicated_dyson:
        raise NotImplementedError(
            f"route='lift' closes with jnp.linalg.solve on the N_mu x N_mu "
            f"tile, a single-device dense solver, but this mesh is "
            f"{mesh_xy.devices.shape}.  Above one device the close must go "
            f"through the distributed solver (services/distrib_la), the twin "
            f"of gw.w_isdf.solve_w's 'distributed' plan; that routing is the "
            f"registered follow-on.  Pass allow_replicated_dyson=True to "
            f"accept the per-rank N_mu^2 tile the RPA Dyson's own DEFAULT "
            f"('local') plan already holds, or route='plain'/'auto'.")

    def _on_result(iq, q, iz, z, c0, n_real, W_tile, resids, its):
        tiles[iz][iq] = _accumulate_columns(
            tiles[iz][iq], W_tile, c0, n_real, n_pad, mesh_xy)
        resid[iz, iq, c0:c0 + n_real] = np.asarray(gather_to_host(resids))[:n_real]
        iters[iz, iq, c0:c0 + n_real] = np.asarray(gather_to_host(its))[:n_real]

    # The lift's closing Dyson needs each q's OWN ``v`` tile, which only the
    # q-shifted payload carries; ``build_hook`` is the sweep's existing seam
    # for exactly that and costs nothing on the plain route.
    v_by_q: dict = {}

    sweep_q_wedge(
        data, mesh_xy, q_list, z_list_ry, include_w=include_w,
        probe_blocks_for_q=lambda _iq, _q: blocks,
        gmres_tol=gmres_tol, gmres_max_iter=gmres_max_iter,
        on_result=_on_result, route=route,
        build_hook=((lambda iq, q, dq: v_by_q.__setitem__(iq, dq["V_q0"]))
                    if route == "lift" else None))

    if route == "lift":
        # Every accumulated tile is ``T = Pi v``; the ring goes back in here,
        # ONCE per (q, z), through the closing Dyson.  Single-sourced in
        # w_ladder_precond (route A's own module) — imported at call time
        # because that module imports this one.
        from .w_ladder_precond import dyson_close_tile
        tiles = [[dyson_close_tile(tiles[iz][iq], v_by_q[iq],
                                   allow_replicated_solve=allow_replicated_dyson)
                  for iq in range(nq)] for iz in range(nz)]

    # Stack (nz, nq, mu_pad, nu_pad) and KEEP the pad — see WLadderWedge.wc.
    # The pad rows/columns are already zero (identity probe on a zero-psi pad
    # centroid), and stripping them here would make the tile's last two axes
    # n_rmu-wide, which the documented P(None, None, 'x', 'y') sharding cannot
    # express for an n_rmu that does not divide the mesh (399 on a 2x2 —
    # measured, JID 57064957).  The logical extent travels as `n_rmu`.
    stacked = jnp.stack([jnp.stack(row, axis=0) for row in tiles], axis=0)
    wedge_sh = NamedSharding(mesh_xy, P(None, None, "x", "y"))
    wc = jax.lax.with_sharding_constraint(
        stacked.astype(jnp.complex128), wedge_sh)
    _assert_pad_block_is_zero(wc, nlog)
    return WLadderWedge(
        wc=wc,
        n_rmu=nlog,
        z_list_ry=z_list_ry,
        q_irr_kgrid_int=q_list,
        irr_idx_q=np.asarray(sym.irr_idx_q),
        sym_idx_q=np.asarray(sym.sym_idx_q),
        q_irr_full_idx=np.asarray(sym.q_irr_full_idx),
        gmres_resid=resid,
        gmres_iters=iters,
        include_w=bool(include_w),
    )


#: ``(shape, dtype, sharding, nlog) -> jitted worst-pad reduction``.  Cached
#: for the reason every other jit in this file is: ``jax.jit`` keys on function
#: identity, so building one per call re-traces every time (the F1 defect, in
#: miniature).
_PAD_CHECK_CACHE: dict = {}


def _assert_pad_block_is_zero(wc: jax.Array, nlog: int) -> None:
    """The pad rows and columns of the wedge are EXACTLY zero, checked here.

    The producer owns this guarantee (design amendment 2026-08-15): a
    consumer cannot tell "the pad is zero because there is no such centroid"
    from "the pad is zero because nothing was written there", and every
    downstream contraction multiplies the pad by a zero ζ row, where a NaN or
    an inf would NOT be inert (``0 * NaN`` is NaN).

    A MASKED REDUCTION UNDER ``jax.jit``, NOT A SLICE, and the difference is
    the whole μ²-class question.  The obvious spelling —
    ``max(|wc[..., nlog:, :]|), max(|wc[..., :, nlog:]|)`` evaluated eagerly —
    slices a ``P(None, None, 'x', 'y')`` array on the axes it is sharded on,
    and each eager slice ALL-GATHERS that axis: measured
    ``c128[nz, nq, N_mu, N_mu/p]`` per rank, ~4.0 GB/rank at N_mu = 1e4,
    p = 32, nz = 8, nq = 10, on the return path of every wedge whose n_rmu is
    not a multiple of p_x·p_y — i.e. essentially every real deck (HLO audit
    2026-08-16, FINDINGS.md F3; it is the same hazard as the
    ``stacked[:, :, :nlog, :nlog]`` return this file already lost once).
    Masking is elementwise, so GSPMD keeps the operand exactly where it is and
    the only collectives are the volume-preserving ones the reduction itself
    needs; the two scalars come back replicated.
    """
    n_pad = int(wc.shape[-1])
    if n_pad == nlog:
        return
    key = (tuple(wc.shape), str(wc.dtype), wc.sharding, int(nlog))
    fn = _PAD_CHECK_CACHE.get(key)
    if fn is None:
        sharding, n_logical = wc.sharding, int(nlog)

        def _worst(a):
            a = jax.lax.with_sharding_constraint(a, sharding)
            is_pad = jnp.arange(a.shape[-1]) >= n_logical
            mag = jnp.abs(a)
            return jnp.max(jnp.where(is_pad[:, None] | is_pad[None, :],
                                     mag, jnp.zeros((), mag.dtype)))

        fn = jax.jit(_worst)
        _PAD_CHECK_CACHE[key] = fn
    worst = float(jax.device_get(fn(wc)))
    if not (worst == 0.0):
        raise RuntimeError(
            f"w_ladder: the padded block of the returned wedge is not zero "
            f"(max |.| = {worst!r} over rows/cols {nlog}:{n_pad}).  Pad "
            f"centroids do not exist, so their W - v must be identically "
            f"zero; a NaN here is the zero-rhs GMRES signature (0/0 in the "
            f"Arnoldi normalisation) and a finite value means a pad column "
            f"was solved as if it were physical.")


#: ``(devices, n_pad, n_real, tile shape/dtype) -> jitted placement``.
_ACCUMULATE_CACHE: dict = {}


def _accumulate_columns(acc, W_tile, c0: int, n_real: int, n_pad: int,
                        mesh_xy: Mesh):
    """Place a probe chunk's ``(mu_X, probe_Y)`` tile into the (mu, nu) tile.

    ``acc`` stays sharded ``P('x','y')`` throughout and ``None`` on the first
    chunk allocates.

    THE PLACEMENT RUNS UNDER ``jax.jit`` WITH EXPLICIT SHARDINGS, and that is
    not decoration.  Eagerly, ``dynamic_update_slice`` on the nu axis of a
    ``P('x','y')`` accumulator crosses a dispatch boundary with no sharding
    context, and XLA satisfies it by ALL-GATHERING the accumulator's nu axis:
    measured ``c128[N_mu/p_x, N_mu]`` per rank per placement, which divides by
    sqrt(P) rather than by P (~50 MB/rank at N_mu = 1e4, p_x = 32) — so the
    ``probe_chunk`` knob documented as trading dispatches for MEMORY was, at
    every setting but the default single chunk, spending memory instead (HLO
    audit 2026-08-16, FINDINGS.md F2).  The identical body under ``jit`` with
    in/out shardings emits one shard-sized gather.  The single-chunk arm was
    already clean and stays bit-identical; it simply no longer has to be the
    only safe setting.

    ``c0`` IS BAKED IN AS A CONSTANT, one tiny compiled placement per chunk
    offset, and that is the trade the measurement forces.  Chunk offsets are
    shard-aligned by construction (``chunk % py == 0``), so a CONSTANT index
    lets GSPMD see the alignment and move only shard-sized pieces; a TRACED
    index hides it and XLA falls back to all-gathering the accumulator's whole
    nu axis — measured, on this same probe: ``c128[N_mu/p_x, N_mu]`` with a
    traced index against ``c128[N_mu/p_x, N_mu/p_y]`` with a constant one.  A
    program that moves nothing but data costs milliseconds to compile and is
    compiled once per offset per process; a per-placement mu-row gather costs
    bytes on every chunk of every q of every z.  ``n_real`` is static for the
    ordinary reason — it sets the update's shape.
    """
    tile_sh = NamedSharding(mesh_xy, P("x", "y"))
    if acc is None:
        acc = jax.lax.with_sharding_constraint(
            jnp.zeros((n_pad, n_pad), dtype=jnp.complex128), tile_sh)
    key = (tile_sh, int(n_pad), int(n_real), int(c0),
           tuple(W_tile.shape), str(W_tile.dtype))
    fn = _ACCUMULATE_CACHE.get(key)
    if fn is None:
        width, start = int(n_real), int(c0)

        def _place(acc_, tile_):
            # acc and the incoming chunk both carry the tile sharding (the
            # chunk's probe axis is `chunk` wide and `chunk % py == 0`).  The
            # UPDATE deliberately carries no constraint: a short final chunk's
            # width need not divide the mesh, and 15 columns on a 2x2 is a
            # legal intermediate and an illegal NamedSharding.
            acc_ = jax.lax.with_sharding_constraint(acc_, tile_sh)
            tile_ = jax.lax.with_sharding_constraint(tile_, tile_sh)
            upd = tile_[:, :width].astype(jnp.complex128)
            return jax.lax.with_sharding_constraint(
                jax.lax.dynamic_update_slice(acc_, upd, (0, start)), tile_sh)

        # No in_/out_shardings: this is called from inside another jit by the
        # HLO probe, and a nested jit carrying explicit shardings is not a
        # supported shape.  The constraints above express the same thing.
        fn = jax.jit(_place)
        _ACCUMULATE_CACHE[key] = fn
    return fn(acc, W_tile)
