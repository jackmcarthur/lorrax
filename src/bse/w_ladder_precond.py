"""Preconditioning for the ladder-W resolvent — the Hartree ring is a DYAD, and
a dyad is not something a Krylov method should be asked to resolve.

``w_ladder`` evaluates ``W(z) - v = v (z - H)^{-1} v`` for the ladder operator
with one shifted block-GMRES solve per probe column, right-preconditioned by the
assembled diagonal ``1/(z - diag H)`` (``bse_feast._gmres_solve_core``).  On the
gnppm_debug fixture that costs ~23 iterations per column.  This module supplies
two ways to spend far fewer, both of which invert the SAME piece of the operator
exactly instead of iterating on it, and neither of which edits a line of the
existing solver: they compose with it through ``matvec_operands`` and the
``gen``/``snapshot`` reshard boundaries.  A third, :func:`pair2x2_precond`, is
the in-tree diagonal plus the one term it drops (the resonant/anti-resonant
coupling at the same pair index) and costs nothing to apply.

WHAT THE ITERATIONS ARE ACTUALLY SPENT ON
-----------------------------------------
The ladder operator is (``w_ladder`` derivation, steps 1 and 4)

    H = H_0 + Sigma,        Sigma = s v p,        H_0 = Sigma_3 (D~ + K^d)

where ``K^d`` is the direct (screened-exchange) rung, and the HARTREE RING term
``Sigma`` is the outer product of the seed injection and the density readout:

    s(rho) = [ E rho ; -E rho ],   (E rho)_I = (1/sqrt(Nk)) sum_mu M_I(mu) rho(mu)
    p([X;Y]) = P(X + Y),           (P x)(nu) = (1/sqrt(Nk)) sum_I conj(M_I(nu)) x_I

with ``M_I(mu) = sum_s conj(psi_c) psi_v`` the ISDF pair density.  ``E`` and
``P`` are not analogies: they are the kernels this tree already ships.
``bse_ring_comm.build_realspace_random_transition_generator`` computes
``gen(g, V) = E(V g)`` (its ``M_X`` decode leg carries the bare vertex, its
``1/sqrt(nk)`` is the one written above), ``build_density_snapshot_operator``
computes ``snapshot(x, V) = V P(x)`` (conjugated encode leg, same
``1/sqrt(nk)``), and ``apply_V_ring`` computes ``E V P`` — the two ``1/sqrt(nk)``
compose into the ``1/Nk`` its own comment names.  So the ring coefficient
relative to the seed/readout vertices is EXACTLY ONE, not merely proportional:

    Sigma = s v p     with the SAME v the seed and the readout carry.       (R)

(R) is asserted numerically, not assumed: :func:`check_ring_dyad_identity`
applies ``matvec(x, V_q0=V) - matvec(x, V_q0=0)`` and compares it against
``s v p x`` built from ``gen``/``snapshot``, on the payload actually in use.

The ring is what makes the RPA solve hard, and — MEASURED, against the
expectation this module was built on — only about a fifth of what makes the
LADDER solve hard.  ``v`` is the Coulomb tile; the RPA part of ``H`` it
generates is positive semi-definite and LARGE (it is the whole of screening),
and ``bse_feast._gmres_solve_core``'s own comments record the consequences from
the RPA path: ``cond(H) ~ 1e8``, Arnoldi orthogonality lost by ~20 iterations
without the DGKS pass, and the normal equations unusable.  So the working
hypothesis was that the rung ``K^d`` is a small short-ranged correction —
exciton binding, not screening — and that removing the ring would leave a nearly
diagonal problem.

Half of that is right.  Everything below was measured on the gnppm_debug fixture
(MoS2 3x3, nspinor 2, 26v x 20c x 9k, N_mu = 399) over the FULL 399-column basis
at ``probe_chunk = 64``, ``gmres_tol = 1e-9``, 1 GPU, 2026-08-16; the logs and
the whole table are in the sandbox at
``reports/screening_diagrams_wbse/evidence/opt_precond/RESULTS.md``.  At q=0,
z=0, one process, all five in the same run:

    operator / preconditioner                     iters (mean/max)   ms/col-iter
    RPA (no rung), ring LIFTED                      1.0 /  1          -
    ladder, in-tree diagonal  (the baseline)       14.5 / 19         15.588
    ladder, ring LIFTED (route A)                  11.1 / 12         15.888
    ladder, exact (z-H_RPA)^-1 precond (route B)   13.4 / 14         15.729
    ladder, 2x2 pair block (:func:`pair2x2_precond`)
                                                   14.4 / 19         15.450
    ladder, LIFT + 2x2 pair block                  12.0 / 13         15.550

Removing the ring from the RPA problem removes ALL of its difficulty: the
preconditioner becomes ``(z - H_0)`` exactly and the solve converges on the
first iteration (and route A there IS the textbook ``v chi_0 (1 - v chi_0)^-1 v``
— see the sanity anchor below).  Removing it from the LADDER problem buys
14.5 -> 11.1.  The rung costs eleven of the fourteen iterations, and it is not
small: for a 2D material the screened-exchange rung is comparable to the
transition energies it sits on (MoS2's exciton binding is a sizeable fraction of
its gap), so ``|| K^d (z - H_RPA)^{-1} ||`` is O(1) and route B — an EXACT
inverse of everything except the rung — still needs 13.4 iterations.  That is the
measurement that reframes the problem: for the ladder, the object worth
preconditioning is the RUNG, and the routes here are the exact treatment of the
ring, which is the part that happens to have a closed form.

The ``ms/col-iter`` column spans 2.8% across five different preconditioners, so
EVERY difference between them is iteration count and nothing else — including
route B's, whose two vertex contractions and ``N_mu^2`` gemv per Krylov vector
cost +0.9% (they are the ring's own arithmetic, and the ladder matvec is
dominated by the rung's ``(mu, nu, s, s, k)`` intermediate).  Because a column's
wall is superlinear in its iteration count — one matvec plus an ``O(k)`` DGKS
pass and an ``O(k^2)``-shaped ``lstsq`` at step ``k`` — the 1.22x iteration cut
is worth more than 1.22x: route A measures **1.41x on 5 q_irr x z=0 and 1.44x on
the 8-point MPA z-plan**, at the same tolerance and with a maximum TRUE residual
60x SMALLER than the baseline's (1.81e-10 against 1.08e-08 — GMRES exits on the
PROJECTED residual, which tracks the true one only as well as the operator is
conditioned).

The 2x2 pair block is in this file, is correct, and IS A NULL RESULT, recorded
as one so the next person does not rebuild it.  Adding the resonant/anti-resonant
coupling ``b = (V_x - W_d^B)/nk`` to the diagonal moves the mean from 14.5 to
14.4 and the max not at all, because ``|b| << |a|`` — ``a`` carries ``ΔE`` and
``b`` does not.  Composed with the lift it HURTS (12.0 against 11.1): with the
ring gone ``b`` is ``-W_d^B/nk`` alone, and the 2x2 inverse it builds is a worse
approximation than the plain diagonal.

What is NOT here, and is the indicated next step, is a preconditioner for the
rung itself: its ``k' = k`` block is dense ``(n_c n_v) x (n_c n_v)`` per k-point
(520 x 520 on this fixture, 9 of them), carries the largest Fourier component
``W(q=0)`` of the kernel, and would factorize once per q at a few times
``10^9`` flops with an apply of ``10^7``.  Composed with route A — which has
already removed the ring, the one term that is dense in ``k`` and would spoil
the block structure — that is a well-posed and cheap construction.  It is named,
not half-built.

WHERE AN ITERATION'S TIME ACTUALLY GOES (and what preconditioning cannot reach)
-------------------------------------------------------------------------------
Measured the same day, on the same fixture, because "cut iterations" is only half
of "cut wall" and the other half turned out to be the larger half.  ONE ladder
matvec, timed against BLOCK WIDTH ``nb``, with the RPA matvec (the identical
operator minus the direct rung) alongside:

    nb        1      2      4      8     16     32     64
    ms/col   16.10   8.82   8.67   8.74   8.56  10.91  20.69
    rung share of the ladder matvec: 96-99% at every width it is measurable

Three facts, all of which constrain what this module can be worth:

1. **The rung IS the matvec** (96-99%).  The ring, the D term and the two
   density vertices are 1-3% together.  So no treatment of the RING — neither
   route here, nor anything else — can move per-iteration cost, and indeed the
   measured ``ms/col-iter`` of all five preconditioners lies in a 2.8% band.
   Conversely a future RUNG preconditioner is paid for against a very large
   denominator.
2. **Block width is worth 1.88x on the matvec and is NOT taken today.**  The
   production engine (``bse_w_exact._get_block_gmres_solver``) is a ``lax.scan``
   over probe columns at batch ONE, so it pays 16.10 ms per column where a
   matrix-RHS Arnoldi would pay 8.56.  The whole win is present already at
   ``nb = 2`` and flat to 16.
3. **The ``W_R``-sized rung buffer is the ceiling, and it is measured.**  The
   rung stages ``T -> T_R -> U_R -> U_q -> U``, each ``(nb, mu, nu, s, s, k)`` =
   ``nb`` x 87.5 MiB on this fixture (the ``W_R`` tile itself is only 21.9 MiB —
   it is the per-trial-vector chain that scales), applied four times per non-TDA
   matvec.  Past ``nb = 16`` the per-column cost REGRESSES and at ``nb = 64`` XLA
   reports ``Can't reduce memory use below 29.94GiB by rematerialization; only
   reduced to 43.77GiB``.  A blocked solver must therefore stage the block
   THROUGH the rung a few columns at a time while the ring, D, vertices and the
   Krylov algebra take the full width — and since the flat region starts at
   ``nb = 2``, a sub-chunk of 2-4 captures all of it.  At production ``N_mu`` the
   buffer grows as ``N_mu^2``, so that sub-chunk shrinks, not grows.

Route A helps blocking twice over: a block method runs every column to the WORST
column's iteration count, and the lift both lowers the count and tightens its
spread (11.1 mean / 12 max, an 8% spread, against the baseline's 14.5 / 19 =
31%).  Compounding at fixed memory: 1.22x fewer iterations x 1.88x cheaper
matvec ~ **2.3x**, less ~8% for block synchronisation.

One more per-iteration item, since it is free and it is this module's business:
``_gmres_solve_core`` re-solves a dense ``(max_iter+1) x max_iter``
``jnp.linalg.lstsq`` EVERY iteration, so its cost is set by the CAP rather than by
the iterations taken.  MEASURED at ``max_iter`` 300 / 64 / 32 with iteration
counts and residuals BIT-IDENTICAL across all three: 90.0 / 82.2 / 78.3 s
(baseline) and 69.6 / 64.5 / 61.4 s (lift).  Right-sizing the cap — the ladder has
never exceeded 23 iterations at any q or z measured — is 1.15x for a caller-side
constant.  The textbook fix (incremental Givens QR of the Hessenberg: ``O(k)`` per
iteration, residual norm free, one back-substitution at exit) is worth about that
same 15% and belongs in ``bse_feast``, where every solver in the tree gets it —
not here.

ROUTE A — LIFT (:func:`compute_wc_qwedge_lifted`)
-------------------------------------------------
Woodbury on (R), with ``R_0 = (z - H_0)^{-1}`` and ``Pi = p R_0 s`` the
ladder-corrected IRREDUCIBLE polarizability (an ``N_mu x N_mu`` object):

    (z - H_0 - s v p)^{-1} = R_0 + R_0 s v (I - Pi v)^{-1} p R_0 ,

and with the production seed ``b_nu = s(v e_nu)`` and readout ``w = v p x``,

    W(z) - v = v T (I - T)^{-1},        T := Pi v = p R_0 s v .            (A)

``T`` is what the EXISTING three-stage pipeline returns if the matvec's ``V_q0``
operand is zeroed (killing the ring, and only the ring — the rung reads ``W_R``,
the diagonal reads ``eps``) and the snapshot's ``V_q0`` is replaced by the
IDENTITY (dropping the readout's ``v`` so the tile is ``Pi v`` and not
``v Pi v``, which would need ``v^{-1}`` to close).  The seed is untouched, so the
right-hand side, the residual normalisation and therefore the meaning of the
GMRES tolerance are bit-identical to the baseline's.

The cost of the lift is one dense ``N_mu x N_mu`` solve per ``(q, z)`` — the
same object, and the same class of dense algebra, that the production RPA path
already performs (``gw.mpa.model._solve_wc``, ``gw.screening``'s Dyson).  The
per-iteration cost is UNCHANGED: the ring's flops are still executed (against a
zero tile) because ``V_q0`` is a runtime argument and nothing is folded away, so
every second the lift saves is a saved iteration.  On this fixture the ring is a
few percent of the ladder matvec anyway — the rung's ``(mu, nu, s, s, k)``
intermediate dominates — so there is nothing material to reclaim there.

Sanity anchor: at ``include_w=False`` route A is the textbook RPA Dyson.  ``H_0``
is then ``diag(D, -D)``, the zeroed-ring preconditioner diagonal IS ``(z - H_0)``
exactly, GMRES converges on iteration one, and (A) reads
``W - v = v chi_0 (1 - v chi_0)^{-1} v``.  That is not a coincidence to be
admired, it is the cheapest available proof that the algebra and the
normalisation (R) are right, and :func:`check_ring_dyad_identity` and the bench's
``--check`` arm both ride it.

ROUTE B — EXACT RPA-RESOLVENT PRECONDITIONER (:func:`rpa_dyson_precond`)
-----------------------------------------------------------------------
The same Woodbury, used the other way round: keep the FULL ladder operator in
the Krylov iteration and precondition it with ``(z - H_RPA)^{-1}``, applied
EXACTLY rather than by an inner solve or a projected chain.  ``H_RPA = D~ + ring``
is itself ``H_0^{RPA} + s v p`` with ``H_0^{RPA} = diag(D, -D)`` DIAGONAL, so

    (z - H_RPA)^{-1} r = R_D r + R_D s [ v (I - Pi_0 v)^{-1} ] p R_D r ,    (B)
    R_D = diag(1/(z-D), 1/(z+D)),      Pi_0 v = p R_D s v = chi_0(z) v .

``Pi_0 v`` is built with the same pipeline as ``T`` in route A but with ZERO
GMRES iterations (``R_D`` is a division), so the whole preconditioner costs, per
``(q, z)``: one seed + one elementwise divide + one snapshot per probe chunk, one
``N_mu^3`` inverse, and then per Krylov vector two vertex contractions and one
``N_mu^2`` gemv — i.e. exactly the arithmetic of the ring term the matvec is
already paying, which on this fixture is a few percent of a ladder matvec.

The preconditioner is z-dependent and (only) as exact as that dense inverse, and
the iteration is written FLEXIBLE (:func:`fgmres_solve_core`) so it stays
correct if it is ever made approximate.  Flexibility is nearly free here: the
in-tree ``_gmres_solve_core`` already stores the preconditioned basis ``Z`` and
reconstructs ``x = x0 + Z y`` from it, so the only difference is that the
preconditioner is a callable instead of a reciprocal.

Route B was predicted to beat route A, on the argument that ``(z - H_RPA)^{-1}``
is a STRICTLY better inverse of the ring-ful part than ``(z - D~)^{-1}`` is of
the ring-less one (the RPA spectrum obeys ``lambda^2(H_RPA) = spec(D(D+2V)) >=
D^2`` with ``V`` positive semi-definite).  MEASURED, it LOSES on both counts,
and the second one is disqualifying:

* iterations, q=0, ``tol=1e-9``, full basis: 13.4 (route B) against 11.1
  (route A) at z=0, and 10.4 against 9.0 at z=0.35i.  Being exact about the ring
  is worth less than being ring-free, because the ring is not the difficulty
  (see above) and route B still carries it inside the matvec it iterates on.
* **the dense factor ``v (I - chi_0(z) v)^{-1}`` sits INSIDE the iteration, and
  ``det(I - chi_0(z) v) = 0`` is the RPA pole condition.**  At a complex ``z``
  near an RPA excitation the preconditioner is therefore near-singular —
  precisely where an MPA sample plan puts its nodes.  MEASURED 2026-08-16: at
  ``z = 0.25 + 0.05i`` Ry route B had not returned after 11x its own ``z=0``
  wall and was killed, where route A finished the same point in 206.8 s.  Route
  A's dense Dyson is applied ONCE, after convergence, to an already-converged
  ``T``; the same near-singularity is then an accuracy question with a
  measurable condition number, not a convergence failure.

Add to that a P=1 restriction — the two vertex contractions are plain einsums
over the hoisted ``M_X`` / ``M_Y``, see the refusal in
:func:`build_rpa_dyson_preconditioner` — and route B is a measurement, not a
recommendation.  Route A is the one to wire into the facade.  Route B is kept
because it is the honest form of "precondition with the RPA resolvent", because
it is what proves the seam carries a nontrivial preconditioner at all, and
because its failure mode is the useful part of the result.

THE SEAM, AND WHY IT IS NOT LADDER-SHAPED
-----------------------------------------
Route B needs a flexible GMRES; the ladder is not the only solve in this tree
that would want one.  :func:`fgmres_solve_core` is therefore written against a
PAIR — ``(matvec, precond)`` — and nothing else:

    precond(v, z, precond_args) -> M^{-1} v            (shape/dtype/sharding preserving)
    fgmres_solve_core(matvec, b, z, operands, precond, precond_args, max_iter, tol)
        -> (x, k_iters)

with ``precond_args`` an arbitrary pytree of RUNTIME arrays, never closed over
(``jax.jit`` refuses to close over a non-addressable ``jax.Array`` at P>1 — the
defect ``bse_feast._get_feast_runner``'s docstring records having hit, and the
reason its operands became arguments).  The caching split is the in-tree one
unchanged: structure (``matvec``, ``precond``, ``max_iter``, ``tol``) is baked,
values (``b``, ``z``, ``operands``, ``precond_args``) are arguments, so one
executable serves a whole q x z sweep — or a whole FEAST quadrature contour,
whose nodes are just more ``z``.

:func:`diagonal_precond` is ``bse_feast._gmres_solve_core``'s own preconditioner
written in that signature; supplying it recovers the existing core step for step
(same ``x0 = M^{-1} b``, same DGKS pass, same ``lstsq``, same ``while_loop``
exit).  That is the point: this is a seam UNDER the shared solver, not a
ladder-private fork of it.  ``gmres_solve_sharded_jit`` and the FEAST runner can
adopt it by passing ``diagonal_precond`` and their existing ``(diag_h,)``, after
which any of them can be handed a better ``M^{-1}`` — the optical BSE's own
resolvent, a FEAST node-dependent shift, this module's (B) — without a second
Krylov implementation appearing anywhere.  What the optical/FEAST paths need to
adopt it is listed in the integration note of the campaign report; structurally
it is one delegation each, because their block layouts differ only in the
READOUT, which lives in :func:`get_block_fgmres_solver` and not in the core.

WHAT WAS TRIED AND IS NOT HERE
------------------------------
Preconditioning with the ``w_omega_chain`` z^2 reduction, as a projected
approximate inverse, was the starting sketch and is deliberately absent.  The
chain's reduced resolvent ``(z^2 - T)^{-1}`` is exact only ON the Krylov space of
the seed it was built from; applied to an arbitrary Krylov vector of the ladder
iteration it is a subspace PROJECTION, so it is singular on the complement and
needs a second-level preconditioner there — and its apply costs
``O(m p N_cvk)`` (two contractions against the whole stored chain, ~5x a matvec
at ``m=32, p=64``) against the ``O(N_mu N_cvk)`` of (B), which is EXACT.  A
cheaper approximate preconditioner that costs more than the exact one is not a
trade worth building.

SCALING ENVELOPE (TASTE 8 / INVARIANTS 9)
-----------------------------------------
Route A adds, per ``(q, z)``, exactly one object the baseline does not have: the
``N_mu x N_mu`` tile ``T`` (same class and same sharding, ``P('x','y')``, as the
``W - v`` tile the baseline already assembles) and one dense solve against it.
Peak memory is therefore the baseline's; the solve is the ``N_mu^3`` term the RPA
production path already carries per (q, z) and must go through the same
distributed dense solver at ``N_mu -> 1e4`` (``distrib_la``); doing it with
``jnp.linalg.solve`` on a gathered tile, as :func:`dyson_close_tile` does, is a
P=1 convenience and is refused above one device.  Nothing here is
``mu^2``-per-rank that was not already: the probe walk, the chunking, the
reduce-scatter snapshot and the accumulator are ``w_ladder``'s, reused rather
than re-spelled, so the F1/F2/F3 hazards of the HLO audit cannot reappear in a
second copy.

Route B adds the ``N_mu x N_mu`` dense factor ``v (I - Pi_0 v)^{-1}`` per
``(q, z)``, replicated, plus nothing per iteration.  Its einsum vertices are P=1;
the P>1 form routes them through ``gen``/``snapshot`` (which are shard_maps) and
is a named deferral, not a half-build.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap().
# MUST run before this module's own `import jax` (same rule as w_ladder).
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import (
    build_preconditioner_diagonal_sharded,
    matvec_operands,
    ladder_matvec_operands,
    _apply_shifted_matvec,
)
from .bse_io import load_bse_data_from_restart_sharded
from .bse_w_exact import (
    _get_block_gmres_solver,
    _symmetry_tables,
    build_finite_q_data,
)
from .w_ladder import WLadderWedge, build_ladder_resolvent, _accumulate_columns
from common.collectives import device_put_process_local, gather_to_host

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# The stack: the w_ladder engine plus the two density-space tiles the lift needs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecondStack:
    """``build_ladder_resolvent``'s 5-tuple plus the two constant ``(mu, nu)``
    tiles this module substitutes for ``V_q0`` at the two ends of the pipeline.

    ``eye``
        the identity on the PADDED centroid extent, on ``sh.V``.  Handed to the
        SNAPSHOT it strips the readout's ``v``, turning the returned tile from
        ``v Pi v`` into ``Pi v`` — which is what closes (A) without ``v^{-1}``.
        Its pad diagonal being 1 rather than 0 is load-bearing: ``T``'s pad rows
        and columns are identically zero (no such centroid), so ``I - T`` is the
        identity there and stays invertible.
    ``zero_v``
        the zero tile, on ``sh.V``.  Handed to the MATVEC through
        ``matvec_operands`` it removes the ring and only the ring.  Built ONCE
        and reused for every q so ``build_bse_exact_diagonal``'s identity memo
        can see the same object twice.
    """

    matvec: Callable
    gen: Callable
    snapshot: Callable
    sh: object
    eye: jax.Array
    zero_v: jax.Array
    include_w: bool


def build_precond_stack(mesh_xy: Mesh, data: dict, *, include_w: bool = True,
                        vertex_flipped: bool = False) -> PrecondStack:
    """The ladder engine (built ONCE, q-independent) plus the lift's two tiles.

    Delegates the operator assembly to :func:`w_ladder.build_ladder_resolvent`
    verbatim — same ``matvec`` object, therefore the SAME cached block-GMRES
    executable as the baseline (``_get_block_gmres_solver`` keys on
    ``id(matvec)``), which is what makes an iteration count measured here
    comparable to one measured there rather than merely similar.

    The per-q preconditioner diagonal is NOT built here; it is per-q by
    construction (it reads the rolled ``eps_c``) and route A wants it built
    against the ZEROED ring — see :func:`lifted_precond_diagonal`.

    ``include_w=False`` IS HANDED A SHALLOW COPY, and that is a correctness fix
    rather than hygiene.  ``build_ladder_resolvent`` delegates that arm to
    ``_build_rpa_resolvent``, whose ``ensure_W_R(..., include_W=False)`` sets
    ``data['W_R'] = data['W_q']`` — a shape-compatible PLACEHOLDER.  Building an
    RPA stack from the same dict a ladder stack was built from therefore
    silently replaces the real-space rung kernel with the reciprocal-space
    ``W_q`` for every ``build_finite_q_data`` copy taken afterwards, and the
    ladder then solves a different operator with no shape error to announce it.
    The copy keeps the placeholder off the caller's payload; the RPA matvec
    never reads ``W_R`` at all (``_apply_A`` drops the term at trace time), so
    nothing is lost by it.
    """
    src = data if include_w else dict(data)
    matvec, _, gen, snapshot, sh = build_ladder_resolvent(
        mesh_xy, src, include_w=include_w, vertex_flipped=vertex_flipped)
    n_pad = int(data["V_q0"].shape[0])
    dtype = data["V_q0"].dtype
    eye = jax.lax.with_sharding_constraint(jnp.eye(n_pad, dtype=dtype), sh.V)
    zero_v = jax.lax.with_sharding_constraint(
        jnp.zeros((n_pad, n_pad), dtype=dtype), sh.V)
    return PrecondStack(matvec, gen, snapshot, sh, eye, zero_v, bool(include_w))



def stack_operands(dq: dict, stack: "PrecondStack") -> tuple:
    """The operand tuple THIS stack's matvec takes.

    ``w_ladder.build_ladder_resolvent`` builds the ladder engine with
    ``ladder_rung_slots=True`` — its rung consumes four EXTRA operands (the
    rolled, un-flipped psi arrays) that ``ladder_matvec_operands`` supplies —
    while the ``include_w=False`` delegation builds the plain 11-operand RPA
    engine.  Feeding the wrong tuple is not a wrong answer but a shape refusal
    from pjit ("in_shardings ... length 15 for an args tuple of length 11"),
    which is how this module was found stale after the rung-slot landing of
    2026-08-16.  One selector, so there is one place to keep in step.
    """
    return (ladder_matvec_operands(dq) if stack.include_w
            else matvec_operands(dq))


def ringless_payload(dq: dict, stack: PrecondStack) -> dict:
    """``dict(dq)`` whose MATVEC operands carry no ring.

    Shallow copy with ``V_q0 -> 0``.  Only the ring reads ``V_q0`` inside the
    matvec (``apply_V_ring_only``); the rung reads ``W_R``, the diagonal term
    reads ``eps_c``/``eps_v``, so this removes the Hartree dyad and nothing
    else.  The seed and the readout must NOT be fed this dict — they carry the
    physical ``v`` and the ``v`` of (A) is theirs.
    """
    d0 = dict(dq)
    d0["V_q0"] = stack.zero_v
    return d0


def lifted_precond_diagonal(dq: dict, mesh_xy: Mesh, stack: PrecondStack,
                            *, include_w: bool) -> jax.Array:
    """``diag(H_0)`` for the ring-lifted operator, through the canonical builder.

    Same call the baseline makes, on the zeroed-ring payload: the exchange
    contraction drops out and what is left is ``ΔE`` (route A at
    ``include_w=False``: then this IS ``z - H_0`` exactly and the solve is a
    single iteration) or ``ΔE - W_d`` (the ladder).  No second diagonal
    implementation — the one in ``bse_davidson_helpers`` is the only one.
    """
    return build_preconditioner_diagonal_sharded(
        ringless_payload(dq, stack), mesh_xy, include_W=include_w, use_tda=False)


# ---------------------------------------------------------------------------
# (R): the ring coefficient is exactly one, checked rather than asserted
# ---------------------------------------------------------------------------

def check_ring_dyad_identity(dq: dict, stack: PrecondStack, *, seed: int = 0,
                             n_probe: Optional[int] = None) -> float:
    """Relative discrepancy of ``matvec|_V - matvec|_0 == s v p`` on this payload.

    The module docstring reads the coefficient off three kernels; this measures
    it.  A random pair-basis block ``x`` is pushed through both sides:

        left  = matvec(x, V_q0=V) - matvec(x, V_q0=0)          (the ring alone)
        right = [ E(V P(x_X + x_Y)) ; -E(V P(x_X + x_Y)) ]     (gen o snapshot)

    ``P`` is obtained from ``snapshot`` with ``V_q0 = I`` and ``E(V .)`` is
    ``gen`` itself, so ``right`` is assembled entirely from the production
    reshard boundaries.  Returns ``max|left - right| / max|left|``; anything
    above ~1e-12 means the normalisation this module's algebra rests on is not
    the one the kernels implement, and both routes are then wrong by a scalar.
    """
    sh = stack.sh
    px, py = sh.X.mesh.devices.shape
    n_pad = int(dq["V_q0"].shape[0])
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])
    nb = int(n_probe or py)
    rng = np.random.default_rng(seed)
    shape = (2, nb) + tuple(int(s) for s in
                            (dq["n_cond_pad"], dq["n_val_pad"], nk))
    x = jax.lax.with_sharding_constraint(
        jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                    dtype=jnp.complex128), sh.X_full)

    ops_v = stack_operands(dq, stack)
    ops_0 = stack_operands(ringless_payload(dq, stack), stack)
    left = stack.matvec(x, *ops_v) - stack.matvec(x, *ops_0)

    s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
    # snapshot with the identity is P; its output is (mu_X, nu_Y) with nu the
    # BATCH axis, so transpose back to (b, mu) and broadcast over k for gen.
    rho = stack.snapshot(s, dq["psi_c_Y"], dq["psi_v_Y"], stack.eye)  # (mu, b)
    rho_h = np.asarray(gather_to_host(rho)).T                          # (b, mu)
    r = device_put_process_local(
        np.broadcast_to(rho_h[:, :, None], (nb, n_pad, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        stack.gen(r, dq["psi_c_X"], dq["psi_v_X"], dq["V_q0"]), sh.X)
    right = jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0), sh.X_full)

    lh = np.asarray(gather_to_host(left))
    rh = np.asarray(gather_to_host(right))
    den = float(np.max(np.abs(lh)))
    return float(np.max(np.abs(lh - rh)) / max(den, 1e-300))


# ---------------------------------------------------------------------------
# Route A: the lift
# ---------------------------------------------------------------------------

def apply_lifted_resolvent_block(G_zeta, z, dq: dict, stack: PrecondStack,
                                 diag_h0: jax.Array, *, max_iter: int,
                                 tol: float):
    """``T = Pi v`` columns for a probe block — the three stages, ring lifted.

    Byte-for-byte ``bse_w_exact.apply_screening_resolvent_block`` except at two
    points, both of which are argument substitutions and neither of which is a
    new kernel:

      1. stage 2 solves against ``matvec_operands(ringless_payload(...))``, so
         the operator is ``z - H_0`` and not ``z - H``;
      2. stage 3 projects with the IDENTITY, so the tile is ``Pi v`` and not
         ``v Pi v``.

    Stage 1 is untouched (the seed keeps the physical ``v``), so ``rhs``, the
    residual denominator and the iteration counts mean exactly what they mean in
    the baseline.  Returns ``(T_tile[sh.V], resids, iters)``.
    """
    sh = stack.sh
    px, py = sh.X.mesh.devices.shape
    n_probe = int(G_zeta.shape[0])
    if n_probe % py != 0:
        raise ValueError(
            f"probe block n_probe={n_probe} must be a multiple of py={py} "
            "(reduce-scatter tiles nu over y); pad with zero rows.")
    n_rmu = int(dq["V_q0"].shape[0])
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])

    G = np.asarray(G_zeta, dtype=np.float64)
    r = device_put_process_local(
        np.broadcast_to(G[:, :, None], (n_probe, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        stack.gen(r, dq["psi_c_X"], dq["psi_v_X"], dq["V_q0"]), sh.X)
    rhs = jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)

    solver = _get_block_gmres_solver(stack.matvec, sh, max_iter, tol, rhs.dtype)
    s_all, resids, iters = solver(
        rhs, diag_h0, jnp.asarray(z, dtype=jnp.complex128),
        stack_operands(ringless_payload(dq, stack), stack))

    T_tile = stack.snapshot(s_all, dq["psi_c_Y"], dq["psi_v_Y"], stack.eye)
    return T_tile, resids, iters


#: ``(shape, dtype, mesh) -> jitted Dyson close``.  Cached for the reason every
#: other jit in this family is: ``jax.jit`` keys on function identity.
_DYSON_CACHE: dict = {}


def dyson_close_tile(T: jax.Array, V_q0: jax.Array, *,
                     allow_replicated_solve: bool = False) -> jax.Array:
    """``W(z) - v = v T (I - T)^{-1}`` — the lift's closing algebra.

    ``T = Pi v`` on the padded centroid extent.  The pad rows and columns of
    ``T`` are identically zero (the probe walk stops at ``n_rmu`` and the
    identity projector kills pad centroids), so ``I - T`` is block
    ``[[I - T_ll, 0], [0, I]]`` and the inverse never touches the pad — which is
    why the identity, and not the zero tile, is the right projector.

    Solved as ``(I - T)^T Y^T = T^T`` and then ``v Y``; a dense ``N_mu^3``.  This
    is the SAME dense Dyson the RPA production path performs per (q, z)
    (``gw.mpa.model._solve_wc``), not a new class of object, but ``jnp.linalg``
    is a single-device solver: above one device this must route through
    ``distrib_la``, and the caller is refused rather than silently gathering an
    ``N_mu^2`` tile per rank (the F3 hazard, in its dense-algebra form).

    ``allow_replicated_solve=True`` takes that gather DELIBERATELY, and the
    scope of the concession is exactly the RPA path's own: ``gw.w_isdf.solve_w``
    has TWO plans and its DEFAULT (``w_dyson_solver = local``) is a per-q dense
    LU in which "each rank holds whole (mu, mu) tiles for its q's".  So this
    flag makes the ladder's close the same memory class as the production RPA
    close, no better and no worse, and it is opt-in so that the choice is in
    the caller's log rather than in this function's silence.  The
    ``distrib_la`` routing — the twin of that solver's ``distributed`` plan —
    remains the registered follow-on, and is what the P -> 1e3 envelope needs.
    """
    mesh = V_q0.sharding.mesh if hasattr(V_q0.sharding, "mesh") else None
    n_dev = int(np.prod(mesh.devices.shape)) if mesh is not None else 1
    if n_dev > 1 and not allow_replicated_solve:
        raise NotImplementedError(
            "dyson_close_tile uses jnp.linalg.solve, a single-device dense "
            f"solver, but this tile lives on a {mesh.devices.shape} mesh.  The "
            "N_mu x N_mu Dyson close of the ring lift is the same dense solve "
            "the RPA path already does per (q, z) and must go through the "
            "distributed solver (services/distrib_la) at P>1; gathering the "
            "tile per rank here would be exactly the mu^2-replication the "
            "scaling doctrine forbids.  Named deferral, not a half-build.  "
            "allow_replicated_solve=True accepts the gather explicitly (the "
            "memory class of gw.w_isdf.solve_w's DEFAULT 'local' plan).")
    key = (tuple(T.shape), str(T.dtype), str(V_q0.dtype), T.sharding)
    fn = _DYSON_CACHE.get(key)
    if fn is None:
        out_sh = T.sharding

        def _close(T_, V_):
            n = T_.shape[-1]
            A = jnp.eye(n, dtype=T_.dtype) - T_
            # Y = T (I - T)^{-1}  <=>  (I - T)^T Y^T = T^T
            Y = jnp.linalg.solve(A.T, T_.T).T
            # Put the product back on the wedge tile's own sharding: at P>1
            # the dense solve above is replicated by construction, and letting
            # GSPMD pick the OUTPUT layout would leave the wedge stack's
            # with_sharding_constraint to resolve it with a second collective.
            return jax.lax.with_sharding_constraint(
                V_.astype(T_.dtype) @ Y, out_sh)

        fn = jax.jit(_close)
        _DYSON_CACHE[key] = fn
    return fn(T, V_q0)


# ---------------------------------------------------------------------------
# Route B: exact RPA-resolvent preconditioner + flexible GMRES
# ---------------------------------------------------------------------------

def build_chi0_v_tile(z, dq: dict, stack: PrecondStack, blocks, diag_rpa0):
    """``Pi_0(z) v = chi_0(z) v`` on the padded extent — ZERO Krylov iterations.

    The same three stages as :func:`apply_lifted_resolvent_block` with the SOLVE
    replaced by a division: ``H_0^{RPA} = diag(D, -D)`` is diagonal, so
    ``R_D rhs`` is ``rhs / (z - diag)`` and the readout ``p R_D s v`` needs no
    solver at all.  ``diag_rpa0`` is the non-TDA ``[D, -D]`` stack (build it with
    :func:`lifted_precond_diagonal` at ``include_w=False``).

    ``blocks`` is the probe-chunk sequence ``(c0, n_real, G)`` — the same one the
    solve walks, so the chunking hazard analysis does not fork.
    """
    sh = stack.sh
    n_pad = int(dq["V_q0"].shape[0])
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])
    zc = jnp.asarray(z, dtype=jnp.complex128)
    acc = None
    for c0, n_real, G in blocks:
        G = np.asarray(G, dtype=np.float64)
        n_probe = int(G.shape[0])
        r = device_put_process_local(
            np.broadcast_to(G[:, :, None], (n_probe, n_pad, nk)), sh.S)
        f = jax.lax.with_sharding_constraint(
            stack.gen(r, dq["psi_c_X"], dq["psi_v_X"], dq["V_q0"]), sh.X)
        rhs = jax.lax.with_sharding_constraint(
            jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)
        rd = rhs / (zc - diag_rpa0)
        s = jax.lax.with_sharding_constraint(rd[0] + rd[1], sh.X)
        tile = stack.snapshot(s, dq["psi_c_Y"], dq["psi_v_Y"], stack.eye)
        acc = _accumulate_columns(acc, tile, c0, n_real, n_pad, sh.X.mesh)
    return acc


def build_rpa_dyson_factor(chi0v: jax.Array, V_q0: jax.Array) -> jax.Array:
    """``B = v (I - chi_0 v)^{-1}`` — the preconditioner's dense middle.

    One ``N_mu^3`` factorization per ``(q, z)``, after which an apply of (B)
    costs one ``N_mu^2`` gemv.  Same single-device caveat as
    :func:`dyson_close_tile`.
    """
    n = int(chi0v.shape[-1])
    A = jnp.eye(n, dtype=chi0v.dtype) - chi0v
    return jnp.linalg.solve(A.T, V_q0.astype(chi0v.dtype).T).T


def build_rpa_dyson_preconditioner(dq: dict, stack: PrecondStack, diag_rpa0,
                                   B_dense) -> tuple:
    """The (B) preconditioner's RUNTIME-argument pytree for :func:`rpa_dyson_precond`.

    ``(diag_rpa0, M_X, M_Y, inv_sqrt_nk, B_dense)`` — everything the apply reads,
    passed as an argument rather than closed over, for the same reason
    ``matvec_operands`` is: the compiled engine is then keyed on the operator and
    the PRECONDITIONER FUNCTION only, so a whole q x z sweep reuses one
    executable and every point after the first is dispatch.

    REFUSED ABOVE ONE DEVICE, by name.  The two vertex contractions in
    :func:`rpa_dyson_precond` are plain einsums over the hoisted pair amplitudes
    because ``gen`` and ``snapshot`` are ``jax.jit``s carrying explicit in/out
    shardings and a nested jit of that shape is not supported (the same wall
    ``w_ladder._accumulate_columns`` documents).  The einsums contract ``c`` (on
    x in the trial block, full in ``M_X``) and ``v`` (on y, full in ``M_Y``),
    which GSPMD is free to satisfy by resharding the pair-amplitude tensors
    rather than the one-column trial vector.  Route A carries no such
    restriction and is the production recommendation; the P>1 form of route B is
    a shard_map of these two contractions and is a named deferral.
    """
    px, py = stack.sh.X.mesh.devices.shape
    if px * py > 1:
        raise NotImplementedError(
            "build_rpa_dyson_preconditioner is P=1 only: its two vertex "
            f"contractions are plain einsums over M_X/M_Y and this is a "
            f"{px}x{py} mesh.  They cannot call gen/snapshot instead — those "
            "are jax.jit's carrying explicit in/out shardings and nesting one "
            "inside the FGMRES program is not a supported shape — so the P>1 "
            "form is a shard_map of the two contractions, which is a named "
            "deferral.  Use the ring LIFT (compute_wc_qwedge_lifted) at P>1: "
            "it inverts the same dyad exactly and reuses the production "
            "reshard boundaries unchanged.")
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])
    return (diag_rpa0, dq["M_X"], dq["M_Y"], float(1.0 / np.sqrt(nk)), B_dense)


# ---------------------------------------------------------------------------
# THE SHARED SEAM: flexible GMRES for ANY (matvec, preconditioner) pair
# ---------------------------------------------------------------------------
#
# Everything below this line is operator-agnostic and screening-agnostic.  A
# preconditioner is a pure function ``precond(v, z, args) -> M^{-1} v`` where
# ``args`` is any pytree of RUNTIME arrays; the engine bakes ``matvec`` and
# ``precond`` (structure) and takes everything else as arguments (values).  That
# is deliberately the same split ``bse_feast._get_gmres_solver`` and
# ``bse_w_exact._get_block_gmres_solver`` already use, so the optical BSE and
# FEAST paths can adopt this without changing how they think about caching: a
# FEAST quadrature node is another ``z``, a finite q is another ``operands``.
#
# ``diagonal_precond`` below is the in-tree preconditioner expressed in this
# signature, and passing it here reproduces ``_gmres_solve_core`` step for step
# (same x0, same DGKS pass, same lstsq, same while_loop exit) — which is what
# makes this a SEAM under the existing solver rather than a fork of it.

def diagonal_precond(v, z, args):
    """``M^{-1} v = v / (z - diag)`` — the in-tree diagonal, in seam signature.

    ``args = (diag_h,)``.  The ``ndim`` fix-up is ``_gmres_solve_core``'s: the
    TDA diagonal comes back one axis short of the non-TDA trial vector.
    """
    (diag_h,) = args
    m_inv = jnp.asarray(1.0, dtype=v.dtype) / (z - diag_h)
    if m_inv.ndim == v.ndim - 1:
        m_inv = m_inv[None, ...]
    return m_inv * v


def build_pair_2x2_diagonals(dq: dict, mesh_xy: Mesh, stack: PrecondStack, *,
                             include_w: bool, ring: bool = True):
    """``(a, b)`` — the RESONANT and COUPLING diagonals at each pair index.

    The in-tree preconditioner takes the non-TDA diagonal to be
    ``[diag(A), -diag(A)]``, i.e. it treats the resonant and anti-resonant
    components of a pair index as UNCOUPLED.  They are not: the operator's
    2x2 block at pair index ``I`` is ``[[a, b], [-conj(b), -conj(a)]]`` with
    ``a = diag(A) = ΔE + (V_x - W_d)/nk`` and ``b = diag(B) = (V_x - W_d^B)/nk``
    (the anti-resonant row conjugates the DIRECT terms and leaves the ring
    alone — ``bse_ring_comm._antiresonant_row``, and ``ΔE``/``V_x`` are real, so
    the row's diagonal is the conjugate of the resonant one).  Inverting that
    2x2 exactly is elementwise arithmetic: :func:`pair2x2_precond`.

    ``b`` COSTS NO NEW CONTRACTION.  ``diag(V_ring)`` and ``diag(W_d^B)`` are the
    SAME exchange-shaped object ``sum_MN M_I(M) X_MN conj(M_I(N)) / nk`` with
    ``X = v`` and ``X = W(0)``, so both come out of the canonical builder
    (``bse_davidson_helpers.build_bse_exact_diagonal``, through
    ``build_preconditioner_diagonal_sharded``) by swapping its ``V_q0``
    argument, differenced against the ``V_q0 = 0`` call that returns ``ΔE``.
    Three extra builds of an object whose arithmetic is under a third of one
    matvec and whose program is constructed once per process.

    ``ring=False`` builds the pair block of the RING-LIFTED operator (route A):
    ``a`` and ``b`` lose their ``V_x``, leaving ``ΔE - W_d/nk`` and
    ``-W_d^B/nk``.  That is the combination worth trying — the lift removes the
    dyad, the 2x2 attacks what is left of the rung.
    """
    src = dq if ring else ringless_payload(dq, stack)
    a = build_preconditioner_diagonal_sharded(
        src, mesh_xy, include_W=include_w, use_tda=True)
    d_e = build_preconditioner_diagonal_sharded(
        ringless_payload(dq, stack), mesh_xy, include_W=False, use_tda=True)
    b = jnp.zeros_like(d_e)
    if ring:
        b = b + (build_preconditioner_diagonal_sharded(
            dq, mesh_xy, include_W=False, use_tda=True) - d_e)      # +V_x/nk
    if include_w:
        w_q0 = dq.get("_W_q0_for_precond")
        if w_q0 is None:
            w_q0 = dq["W_q"][:, :, 0, 0, 0]
        d_w = dict(dq)
        d_w["V_q0"] = w_q0
        b = b - (build_preconditioner_diagonal_sharded(
            d_w, mesh_xy, include_W=False, use_tda=True) - d_e)     # -W_d^B/nk
    return a, b


def pair2x2_precond(x, z, args):
    """Exact inverse of the pair-index 2x2 block ``[[a, b], [-b*, -a*]]``.

    ``args = (a, b)``.  With ``b = 0`` this IS :func:`diagonal_precond` on the
    non-TDA stack, which is the sense in which it is the in-tree preconditioner
    plus one term; elementwise throughout, so it emits no collective and its
    cost is a handful of flops per pair element.
    """
    a, b = args
    ac, bc = jnp.conj(a), jnp.conj(b)
    det = (z - a) * (z + ac) + b * bc
    x_x, x_y = x[0], x[1]
    return jnp.stack([((z + ac) * x_x + b * x_y) / det,
                      (-bc * x_x + (z - a) * x_y) / det], axis=0)


def rpa_dyson_precond(x, z, args):
    """``(z - H_RPA)^{-1} x`` by (B), EXACTLY, for a ``(2, b, c, v, k)`` block.

    ``R_D x``, then the density readout ``p``, then the dense
    ``v (I - chi_0 v)^{-1}``, then the injection ``s``, then ``R_D`` again.  The
    two contractions carry the SAME ``1/sqrt(nk)`` and the SAME conjugation legs
    as ``snapshot``/``gen`` (bare vertex on the decode, conjugate on the encode);
    that correspondence is what :func:`check_ring_dyad_identity` measures.
    """
    diag_rpa0, M_X, M_Y, inv_sqrt_nk, B_dense = args
    rd = x / (z - diag_rpa0)
    s = rd[0] + rd[1]                                        # (b, c, v, k)
    rho = jnp.einsum("kcvN,bcvk->bN", jnp.conj(M_Y), s) * inv_sqrt_nk
    u = jnp.einsum("MN,bN->bM", B_dense, rho)
    g = jnp.einsum("kcvM,bM->bcvk", M_X, u) * inv_sqrt_nk
    return rd + jnp.stack([g, -g], axis=0) / (z - diag_rpa0)


def fgmres_solve_core(matvec, b, z, operands, precond, precond_args,
                      max_iter, tol):
    """Flexible GMRES — ``bse_feast._gmres_solve_core`` with a callable M^{-1}.

    Structurally identical to the in-tree core, deliberately: same DGKS second
    Gram-Schmidt pass (its absence is what made the stiff screening solves
    diverge under jit), same ``lstsq`` least squares rather than the normal
    equations (``cond(H) ~ 1e8`` on these tiles), same ``while_loop`` early exit
    on the projected relative residual, same ``k_final`` returned as the real
    exit index so ``iters == max_iter`` still reads as TRUNCATED rather than
    converged.

    The one change is that ``z_k = M^{-1} v_k`` comes from ``precond`` instead
    of a reciprocal.  The in-tree core is already flexible-READY — it stores the
    preconditioned basis ``Z`` and reconstructs ``x = x0 + Z y`` from it rather
    than applying ``M^{-1}`` once at the end — so nothing about the
    reconstruction had to move; only the operator behind ``Z`` did.  That
    matters because a preconditioner that VARIES between iterations (an inner
    solve, a chain truncation, anything adaptive) makes a fixed-``M`` right-
    preconditioned GMRES silently wrong, and every preconditioner worth reaching
    for beyond a diagonal is of that kind.

    Contract for ``precond``: a pure function ``(v, z, precond_args) -> v``
    preserving shape, dtype and sharding.  ``precond_args`` is an arbitrary
    pytree of runtime arrays (never closed over — at P>1 ``jax.jit`` refuses to
    close over a non-addressable ``jax.Array``, which is the defect
    ``_get_feast_runner`` records having hit).
    """
    def _precond(v):
        return precond(v, z, precond_args)

    x0 = _precond(b)
    r0 = b - _apply_shifted_matvec(matvec, x0, z, operands).astype(b.dtype)
    beta = jnp.linalg.norm(r0)

    zero = jnp.asarray(0.0, dtype=beta.dtype)
    v0 = jnp.where(beta == zero, r0, r0 / beta)

    V = jnp.zeros((max_iter + 1,) + b.shape, dtype=b.dtype).at[0].set(v0)
    Z = jnp.zeros((max_iter,) + b.shape, dtype=b.dtype)
    H = jnp.zeros((max_iter + 1, max_iter), dtype=b.dtype)
    g = jnp.zeros((max_iter + 1,), dtype=b.dtype).at[0].set(beta)
    y = jnp.zeros((max_iter,), dtype=b.dtype)

    def cond(state):
        k, rel, *_ = state
        return jnp.logical_and(k < max_iter, rel > tol)

    def body(state):
        k, rel, V, Z, H, g, y = state
        z_k = _precond(V[k])
        Z = Z.at[k].set(z_k)
        w = _apply_shifted_matvec(matvec, z_k, z, operands).astype(b.dtype)

        def arnoldi(i, carry):
            w_local, H_local = carry
            h = jnp.vdot(V[i], w_local)
            return w_local - h * V[i], H_local.at[i, k].set(h)

        w, H = jax.lax.fori_loop(0, k + 1, arnoldi, (w, H))

        def reorth(i, carry):
            w_local, H_local = carry
            corr = jnp.vdot(V[i], w_local)
            return w_local - corr * V[i], H_local.at[i, k].add(corr)

        w, H = jax.lax.fori_loop(0, k + 1, reorth, (w, H))
        h_next = jnp.linalg.norm(w)
        H = H.at[k + 1, k].set(h_next)
        V = V.at[k + 1].set(jnp.where(h_next == 0.0, w, w / h_next))

        y = jnp.linalg.lstsq(H, g, rcond=None)[0]
        resid = jnp.linalg.norm(g - H @ y)
        rel = jnp.where(beta == zero, zero, resid / beta)
        return k + 1, rel, V, Z, H, g, y

    init = (0, jnp.asarray(jnp.inf, dtype=beta.dtype), V, Z, H, g, y)
    k_final, _, V, Z, H, g, y = jax.lax.while_loop(cond, body, init)
    return x0 + jnp.tensordot(y, Z, axes=(0, 0)), k_final


#: ``(id(matvec), id(precond), max_iter, tol, dtype) -> (matvec, precond, engine)``
#: — the FGMRES twin of ``bse_w_exact._BLOCK_GMRES_CACHE``, and for the same
#: reason: the operator and the preconditioner STRUCTURE are baked, every q/z
#: tensor is a runtime argument, so a whole sweep compiles once.  The value pins
#: both callables so neither ``id()`` can be recycled while the entry is live —
#: the hazard ``_get_feast_runner``'s docstring records.
_FGMRES_CACHE: dict = {}


def get_block_fgmres_solver(matvec, sh, precond, max_iter, tol, dtype):
    """Cached per-column-scan FGMRES engine for ANY ``(matvec, precond)`` pair.

    Returns a ``jax.jit`` ``(rhs, z, operands, precond_args) ->
    (s_all, resids, iters)`` — the drop-in twin of
    ``bse_w_exact._get_block_gmres_solver`` whose fourth argument is the
    preconditioner's pytree instead of a diagonal.  The screening-specific part
    is only the readout ``s = x_X + x_Y``; a caller with a different readout
    wants :func:`fgmres_solve_core` directly (FEAST does: it solves whole trial
    BLOCKS, not a scan over probe columns, and reads ``x`` itself).
    """
    key = (id(matvec), id(precond), int(max_iter), float(tol), str(dtype))
    hit = _FGMRES_CACHE.get(key)
    if hit is not None:
        return hit[2]

    @jax.jit
    def _block(rhs, z, operands, precond_args):
        rhs_scan = jnp.moveaxis(rhs, 1, 0)          # (nu, 2, c, v, k)

        def _solve_col(carry, rhs_col):
            rhs_i = rhs_col[:, None]                # keep the matvec batch axis
            x, k_used = fgmres_solve_core(
                matvec, rhs_i, z, operands, precond, precond_args,
                max_iter, tol)
            r_true = rhs_i - _apply_shifted_matvec(matvec, x, z, operands)
            nrhs = jnp.linalg.norm(rhs_i)
            resid = jnp.where(nrhs == 0.0, jnp.asarray(0.0, dtype=nrhs.dtype),
                              jnp.linalg.norm(r_true) / nrhs)
            s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
            return carry, (s[0], resid, k_used)

        # unroll=1: one Krylov workspace alive at a time.  The workspace is
        # O(max_iter^2) and replicated (HLO audit: 3.12 MiB at max_iter=200,
        # N_mu-independent), so unrolling the probe axis would multiply the one
        # object in this solve that is already the memory peak.
        _, (s_all, resids, iters) = jax.lax.scan(
            _solve_col, None, rhs_scan, unroll=1)
        return jax.lax.with_sharding_constraint(s_all, sh.X), resids, iters

    _FGMRES_CACHE[key] = (matvec, precond, _block)
    return _block


def apply_fgmres_resolvent_block(G_zeta, z, dq: dict, stack: PrecondStack,
                                 precond, precond_args, *, max_iter, tol,
                                 seed_v=None, snapshot_v=None):
    """``W(z) - v`` columns for a probe block, FULL operator, ``precond``-ed.

    Stages 1 and 3 are the production ones with the production ``v`` on both
    ends, so the returned tile is directly ``W(z) - v`` — no closing algebra, and
    the comparison against the oracle is on the same object the oracle returns.

    ``precond`` / ``precond_args`` are the seam's pair; passing
    :func:`rpa_dyson_precond` gives route B, passing :func:`diagonal_precond`
    with ``(diag_h,)`` gives the baseline through the flexible engine (which is
    how the two are compared without a second seed/readout path existing).

    ``seed_v`` / ``snapshot_v`` override the ``v`` tile at stage 1 / stage 3 and
    default to ``dq['V_q0']``.  They exist so ROUTE A can be driven through this
    same function: hand it the ring-less payload (whose ``V_q0`` is zero, which
    is right for the matvec and wrong for the two vertices), ``seed_v`` the
    physical ``v`` and ``snapshot_v`` the identity, and the returned tile is
    ``T = Pi v`` for :func:`dyson_close_tile` instead of ``W - v``.  One
    function, two routes, no second seed/readout path to drift.
    """
    seed_v = dq["V_q0"] if seed_v is None else seed_v
    snapshot_v = dq["V_q0"] if snapshot_v is None else snapshot_v
    sh = stack.sh
    px, py = sh.X.mesh.devices.shape
    n_probe = int(G_zeta.shape[0])
    if n_probe % py != 0:
        raise ValueError(f"n_probe={n_probe} must be a multiple of py={py}.")
    n_rmu = int(dq["V_q0"].shape[0])
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])

    G = np.asarray(G_zeta, dtype=np.float64)
    r = device_put_process_local(
        np.broadcast_to(G[:, :, None], (n_probe, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        stack.gen(r, dq["psi_c_X"], dq["psi_v_X"], seed_v), sh.X)
    rhs = jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)

    solver = get_block_fgmres_solver(stack.matvec, sh, precond, max_iter, tol,
                                     rhs.dtype)
    s_all, resids, iters = solver(
        rhs, jnp.asarray(z, dtype=jnp.complex128), stack_operands(dq, stack),
        precond_args)
    W_tile = stack.snapshot(s_all, dq["psi_c_Y"], dq["psi_v_Y"], snapshot_v)
    return W_tile, resids, iters


# ---------------------------------------------------------------------------
# The wedge facades — same contract as w_ladder.compute_wc_qwedge
# ---------------------------------------------------------------------------

def _probe_blocks(nlog: int, n_pad: int, chunk: Optional[int]):
    """``w_ladder.compute_wc_qwedge``'s probe-block walk, verbatim in shape.

    Identity rows; the walk STOPS at ``nlog`` (a pad column has an identically
    zero right-hand side and the shifted GMRES normalises by ``||r0||``, so
    solving one makes the whole column NaN); a short final chunk is zero-PADDED
    up rather than reshaped so every chunk hits one compiled engine.
    """
    eye = np.eye(n_pad, dtype=np.float64)
    width = n_pad if chunk is None else min(int(chunk), n_pad)
    out = []
    for c0 in range(0, nlog, width):
        n_real = min(width, nlog - c0)
        G = np.zeros((width, n_pad), dtype=np.float64)
        G[:n_real, :] = eye[c0:c0 + n_real, :]
        out.append((c0, n_real, G))
    return out


def _load_wedge_payload(restart_path: str, mesh_xy: Mesh, input_file: str):
    """The facade's loader call — FULL chi0 band window, head-less, full V."""
    data = load_bse_data_from_restart_sharded(
        restart_path, n_val=10**9, n_cond=10**9, mesh_xy=mesh_xy,
        input_file=input_file, inject_head=False, load_v_full=True)
    sym = _symmetry_tables(input_file)
    return data, sym


def compute_wc_qwedge_lifted(
    restart_path: str,
    z_list_ry,
    mesh_xy: Mesh,
    *,
    include_w: bool = True,
    gmres_tol: float,
    gmres_max_iter: int,
    probe_chunk: Optional[int] = None,
    input_file: Optional[str] = None,
    n_q: Optional[int] = None,
) -> WLadderWedge:
    """ROUTE A facade: ``w_ladder.compute_wc_qwedge`` with the Hartree ring lifted.

    Same signature, same returned :class:`w_ladder.WLadderWedge`, same padded
    ``P(None, None, 'x', 'y')`` wedge — a drop-in for the facade's caller, which
    is the point: the ring lift is an implementation of the same contract, not a
    different quantity.  ``n_q`` truncates the wedge for benchmarking only.

    The engine is built ONCE outside the q loop (``W_R`` depends on ``k - k'``,
    not on q — the F1 lesson), and every q/z tensor rides in as a runtime
    argument, so the sweep compiles once and dispatches thereafter.
    """
    if input_file is None:
        raise ValueError(
            "compute_wc_qwedge_lifted needs input_file= (the COHSEX/GW deck): "
            "the irreducible q-wedge comes from SymMaps, built from the WFN the "
            "deck names, and the sharded loader resolves the k-grid from it.")
    px, py = mesh_xy.devices.shape
    z_list_ry = np.asarray(z_list_ry, dtype=np.complex128)
    if z_list_ry.ndim != 1 or z_list_ry.size == 0:
        raise ValueError("z_list_ry must be a non-empty 1-D complex list.")
    if probe_chunk is not None and (int(probe_chunk) <= 0
                                    or int(probe_chunk) % py != 0):
        raise ValueError(
            f"probe_chunk={probe_chunk} must be a positive multiple of py={py}.")

    data, sym = _load_wedge_payload(restart_path, mesh_xy, input_file)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    if n_q is not None:
        q_list = q_list[:int(n_q)]
    n_pad = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    blocks = _probe_blocks(nlog, n_pad, probe_chunk)

    stack = build_precond_stack(mesh_xy, data, include_w=include_w,
                                vertex_flipped=True)

    nz, nq = int(z_list_ry.size), int(len(q_list))
    wc_rows = [[None] * nq for _ in range(nz)]
    resid = np.zeros((nz, nq, n_pad), dtype=np.float64)
    iters = np.zeros((nz, nq, n_pad), dtype=np.int32)

    for iq, qv in enumerate(q_list):
        q = (int(qv[0]), int(qv[1]), int(qv[2]))
        dq = build_finite_q_data(data, q, mesh_xy)
        diag_h0 = lifted_precond_diagonal(dq, mesh_xy, stack,
                                          include_w=include_w)
        for iz, zval in enumerate(z_list_ry):
            acc = None
            for c0, n_real, G in blocks:
                T_tile, rr, it = apply_lifted_resolvent_block(
                    G, complex(zval), dq, stack, diag_h0,
                    max_iter=gmres_max_iter, tol=gmres_tol)
                acc = _accumulate_columns(acc, T_tile, c0, n_real, n_pad,
                                          mesh_xy)
                resid[iz, iq, c0:c0 + n_real] = np.asarray(
                    gather_to_host(rr))[:n_real]
                iters[iz, iq, c0:c0 + n_real] = np.asarray(
                    gather_to_host(it))[:n_real]
            wc_rows[iz][iq] = dyson_close_tile(acc, dq["V_q0"])

    stacked = jnp.stack([jnp.stack(row, axis=0) for row in wc_rows], axis=0)
    wc = jax.lax.with_sharding_constraint(
        stacked.astype(jnp.complex128),
        NamedSharding(mesh_xy, P(None, None, "x", "y")))
    return WLadderWedge(
        wc=wc, n_rmu=nlog, z_list_ry=z_list_ry, q_irr_kgrid_int=q_list,
        irr_idx_q=np.asarray(sym.irr_idx_q),
        sym_idx_q=np.asarray(sym.sym_idx_q),
        q_irr_full_idx=np.asarray(sym.q_irr_full_idx),
        gmres_resid=resid, gmres_iters=iters, include_w=bool(include_w))
