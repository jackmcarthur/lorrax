"""Shifted-system Krylov SHARING for the ladder-W resolvent — one Arnoldi
space, every frequency.

`w_ladder` evaluates ``W(z) - v = v (z - H)^{-1} v`` one ``z`` at a time: the
q x z sweep calls ``bse_w_exact.apply_screening_resolvent_block`` once per
``(q, chunk, z)`` and each call runs a fresh preconditioned GMRES.  The MPA
ansatz asks for ~20 z per q, so that is ~20 independent Krylov spaces built
from the SAME operator ``H`` and the SAME right-hand block ``b`` — the seed
``f = M^dag (v g)`` carries no ``z`` at all.  This module is the shifted-system
half of that observation.

WHAT IS AND IS NOT SHAREABLE — the derivation, because it decides the design
---------------------------------------------------------------------------

**The Krylov space is shift-invariant.**  For any ``z``,
``K_m(zI - H, b) = span{b, (zI-H)b, ...} = span{b, Hb, ..., H^{m-1}b}
= K_m(H, b)``.  So ONE Arnoldi run on ``H`` seeded with ``b`` serves every
shift.  With ``H V_m = V_{m+1} Hbar_m`` (Hbar_m upper Hessenberg,
``(m+1) x m``) and ``E_m = [I_m; 0^T]``,

    (z I - H) V_m = V_{m+1} (z E_m - Hbar_m),

so the per-shift GMRES problem collapses to the SMALL least-squares problem
``min_y || beta e_1 - (z_j E_m - Hbar_m) y ||`` and ``x_j = V_m y_j``.  Because
``V_{m+1}`` is orthonormal, the least-squares residual IS the true residual
norm (up to the loss of orthogonality, which is why the Arnoldi below keeps
the production solver's DGKS second pass — see ``bse_feast._gmres_solve_core``,
where a single pass was measured to make the projected residual falsely tiny).
This is Frommer & Glaessner's shifted GMRES; the m matvecs are paid ONCE for
all shifts.

**The production preconditioner destroys exactly this property.**
``bse_feast._gmres_solve_core`` right-preconditions with
``M_z = diag(z - diag(H))`` and starts from ``x0 = M_z^{-1} b``.  Both are
z-DEPENDENT, so both the operator ``(zI-H) M_z^{-1}`` and the initial residual
``r_0 = b - (zI-H)x_0`` differ per shift and the spaces
``K_m((zI-H)M_z^{-1}, r_0)`` are genuinely different subspaces.  There is no
way to patch this by choosing a cleverer diagonal, and the reason is a
two-line theorem worth stating so nobody re-litigates it:

    Let T, S be any z-independent invertible preconditioners.  Then
    ``T (z I - H) S = z (TS) - T H S``.  For that to be a shifted family
    ``zeta_j I - Atilde`` for all j we need ``TS = c I``, i.e. ``T = c S^{-1}``,
    whence ``Atilde = c^{-1} S^{-1} H S`` — a SIMILARITY TRANSFORM.

So the only shift-preserving preconditioners are similarity transforms, which
move no eigenvalues; all the clustering a diagonal preconditioner buys is
unavailable to a shared space.  (The genuinely shift-preserving nonlinear
option is shift-and-invert, ``M = (sigma I - H)``, whose inverse is the thing
we are computing; a polynomial surrogate ``p(H)`` gives
``K_m(p(H)(z_j - H), p(H)b) subset K_M(H, b)`` for a larger M, so it can never
beat plain shifted GMRES at equal matvec count.  Named, not built.)

**Consequence, stated as the trade this module exists to measure.**  The
shared space is UNPRECONDITIONED, so it needs more Arnoldi steps ``m`` than the
preconditioned per-shift solve needs iterations ``k``.  The break-even is

    m  <=  nz * k

matvecs.  On the gnppm_debug ladder operator the production solve converges in
k ~ 23 (measured, claim 0218), so at nz = 8 the shared space must reach 1e-8 in
<= 184 steps and at the MPA-realistic nz = 20 in <= 460.

MEASURED, 2026-08-16, gnppm_debug ladder operator, MPA
``double_parallel_grid(n_p=4)`` in Ry, 1 GPU, ``--mode calibrate`` (per-shift
projected residual vs Arnoldi dimension, q = (0,0,0) and q = (1,0,0), two
columns each; the four curves agree to within a factor ~4):

======  ========  ==============  ==============  ==============  ===========
  m     z = 0     1.33 + 0.2i     2.67 + 0.2i     5.33 + 0.2i     Im z = 2 Ry
======  ========  ==============  ==============  ==============  ===========
   50   1.0e-02       1.7e-02         4.3e-03         4.0e-07     <= 8e-10
  100   2.8e-03       1.6e-03         1.9e-04         3.5e-12     converged
  300   1.2e-04       8.6e-08         4.9e-10       converged      converged
  600   1.4e-06       5.2e-14       converged       converged      converged
======  ========  ==============  ==============  ==============  ===========

The answer is a SPLIT, and it is a physical one rather than a property of this
deck.  The four FAR-line shifts (``Im z = 2 Ry``) and the outermost near-line
one are solved to machine precision by ~50-100 shared matvecs — ~12 matvecs
each amortised, against 25 apiece for the preconditioned solve.  The near-line
shifts are the opposite: ``z = 0`` — which every ansatz needs, cohsex, gn_ppm
and MPA alike — is still at 1.4e-06 after 600 matvecs and shows no sign of
reaching 1e-8 at any affordable dimension.  That is exactly what the algebra
predicts: those shifts sit ON the spectrum, ``(z - H)`` is nearly singular
there, and the diagonal preconditioner's power on them is the RESONANT
DENOMINATOR ``1/(z - d)`` itself.  An unpreconditioned space cannot manufacture
it and no preconditioner survives sharing.

So pure shifted GMRES LOSES on this operator — decisively at ``z = 0``, where
it does not converge at all — and the useful form is the hybrid: a SHORT shared
space that finishes the far shifts outright, then the production preconditioned
engine on what is left (:func:`solve_hybrid_block`).

THE BENCH VERDICT (5 q_irr x 64 probe columns, 1 GPU, tol 1e-8)
---------------------------------------------------------------
Matvecs per column per shift, and the TRUE relative residual each arm actually
delivered:

=========  ==============  ===============  ==============  ===============
workload   arm             matvecs/col/z    wall (s)        max true resid
=========  ==============  ===============  ==============  ===============
z = {0}    baseline               18.62       247.8           1.08e-07
           chained                20.10       255.9           9.89e-09
           hybrid m=50            69.11      1156.9           9.97e-09
MPA nz=8   baseline               12.72      1118.2           1.08e-07
           chained                13.40      1067.6           1.00e-08
           hybrid m=50            14.34      1140.5           9.97e-09
           shifted m=300 (*)      38.50       783.1           3.76e-03
=========  ==============  ===============  ==============  ===============

(*) 2 q_irr, not 5 (its wall scales exactly linearly in columns — the probe
axis is a scan — so the 5-q figure is 1958 s).  Every column exited on the loop
bound, steps min/med/max = 300/300/300, which is the TRUNCATION signature this
module's contract names: 3x the baseline's matvecs to deliver five orders LESS
accuracy.

READ THE LAST COLUMN BEFORE THE FIRST.  The baseline's numbers are for a solve
that is ~10x LOOSER than the others, and not because it was asked to be: see
"the tolerance is not the tolerance" below.  Normalising by the measured cost
of one decade on this operator (+1.48 matvecs/col, from the z={0} pair, where
`chained` IS the baseline algorithm with only the stopping norm changed), an
equal-accuracy baseline costs ~14.1 matvecs/col/z at nz=8.  Against that:
`chained` is ~5% cheaper, `hybrid` ~2% dearer, and neither is a result worth
integrating on its own.  `shifted` is not a contender at all: 3.0x the matvecs for a
result five orders worse, because ``z = 0`` never converges (see the
calibration table above and the gate).

THE TOLERANCE IS NOT THE TOLERANCE
----------------------------------
``bse_feast._gmres_solve_core`` exits on ``||r_k|| <= tol * ||r_0||`` where
``r_0 = b - (z-H) M_z^{-1} b`` is the residual AFTER the preconditioned initial
guess — not ``tol * ||b||``.  On this operator the diagonal guess makes things
worse before better: ``||r_0|| / ||b|| ~ 8-11``, so a caller asking for 1e-8
receives 1.08e-07.  Measured four independent times on this fixture: nominal
1e-9 -> 1.08e-08 (claim 0218's wedge sweep), 1e-8 -> 1.08e-07 (both bench
workloads), 1e-8 -> 8.41e-08 / 8.28e-08 (gate, q=0 / finite q), 1e-12 ->
1.07e-11 (the gate's own oracle).  Every arm in this module normalises by
``||b||`` instead (:func:`_precond_gmres_scaled`), which is why they land at
9.9e-09 for a 1e-8 request and why their matvec counts are not comparable to
the baseline's at face value.  Registered in KNOWN_LORRAX_ISSUES.

One number decides how to read every count above: the ring matvec is
WORK-bound, not launch-bound.  Measured on the same operator, cost per probe
column is 7.89 ms at batch 1 and 7.33 ms at batch 64 — a 7% amortisation over a
64x batch.  So matvec COUNT is wall time here, the honest cross-track metric
and the thing a user waits for do not come apart, and batching the probe axis
is not a hidden lever anyone is leaving on the table.

RESTARTS
--------
Not implemented, DELIBERATELY, and the reason is arithmetic rather than taste.
A restart is a memory device: it caps ``V`` at ``m+1`` basis vectors.  Here one
basis vector is ONE probe column of the pair basis, ``(2, 1, c, v, k)`` — 146 KiB
on the gnppm fixture, and ``N_mu``-INDEPENDENT (the probe axis is scanned, never
blocked, exactly as in the production engine).  At m = 400 that is 60 MiB for
the whole space, against a solve whose measured peak is already 3.1 MiB of
replicated GMRES workspace.  There is no memory pressure to relieve, so a
single long cycle is both simpler and strictly better conditioned than any
restart.

The restart variants exist and are correct — Frommer & Glaessner's collinear
restart forces every shift's restarted residual to be a multiple of the seed
shift's ``v_{m+1}``, at the cost of an augmented (m+1)x(m+1) solve per shift and
a seed-shift choice that can stall when an outlying shift's residual grows.
That machinery buys nothing at this vector size and would add the one failure
mode (a silently stalled outlying shift) this module's true-residual gate
exists to exclude.  If a future deck makes ``V`` expensive — a BLOCKED Arnoldi
over ``p_chunk`` probe columns would, at ``p_chunk`` x the storage — restart is
the lever to add, and the note belongs there and not here.

PER-SHIFT TRUE RESIDUALS ARE NOT OPTIONAL
-----------------------------------------
A shared space converges at different rates for different shifts, and the
failure signature is silent: the space is "big enough" for the easy shifts and
the outlying one exits on the loop bound with a residual nobody looked at.
:func:`solve_shifted_block` therefore returns, per (column, shift), BOTH the
projected residual and the explicitly recomputed
``|| b - (z_j - H) x_j || / ||b||`` (nz extra matvecs per column, ~4% at m=200),
and the callers gate on the latter.  ``steps == arnoldi_dim`` is the truncation
signature, the same contract ``apply_screening_resolvent_block`` states for
``iters == max_iter``.

THE SEED HOIST (independent of everything above; worth taking regardless)
------------------------------------------------------------------------
``apply_screening_resolvent_block`` is three dispatches — SEED (``gen``), SOLVE,
PROJECT (``snapshot``) — and ``w_ladder.sweep_q_wedge`` calls it once per
``(q, chunk, z)``, so a wedge pays ``3 * nq * nchunk * nz`` dispatches.  But
stage 1 reads only ``(G_zeta, psi_c_X, psi_v_X, V_q0)``: **the seed is
z-independent**, and re-running it per z recomputes a bit-identical array.
Hoisting it to per-(q, chunk) makes the count ``nq * nchunk * (2 + nz)`` —
at nz = 20, 60 dispatches per (q, chunk) become 22.  This module's block entry
points take the z-LIST and seed once, so they get the hoist by construction;
the same hoist applies to the production per-z path and is a two-line change
there (seed outside the z loop, pass ``rhs`` in).  It is orthogonal to which
solver wins.

SCALING ENVELOPE (TASTE 8 / INVARIANTS 9)
-----------------------------------------
Per (q, chunk): one seed (``sh.S`` in, ``sh.X_full`` out), one solve, nz
projections.  Solve high-water = ``(m+1)`` pair-basis vectors for ONE probe
column (the probe axis is scanned with ``unroll=1``, never blocked) + the
``nz x (m+1) x m`` replicated Hessenberg stack, which is ``N_mu``-independent.
Readout is ``nz`` tiles of ``sh.V = P('x','y')``.  No ``N_mu^2``-class
replicated object is formed at any point; the envelope is the production
engine's with the ``max_iter^2`` workspace term traded for an ``m``-long basis
whose vectors are pair-basis-class.

ADOPTION — this is a SOLVER-LAYER capability, not a ladder feature
------------------------------------------------------------------
:func:`shifted_solve_core` takes ``(matvec, operands, b, z_list)`` and nothing
else: the same 2-tuple ``bse_feast``'s GMRES core speaks, any right-hand vector
that ``matvec`` accepts, and a list of shifts.  It inspects no shape, imports
no ladder, and knows nothing about probe blocks or centroids — everything
W-specific in this file is a wrapper around it.  Two other in-tree families
solve the SAME structure and could inherit it:

* **FEAST** (``bse_feast._get_feast_runner``) is the strongest case, and it is
  already written as a shifted family: ``pole_body`` loops ``n_quad``
  quadrature nodes over the SAME Ritz vector ``x``, one preconditioned GMRES
  each, and sums ``w_j y_j``.  A shared space serves the whole contour, and
  FEAST needs the individual ``y_j`` even less than the resolvent does — pass
  ``weights=w_weights`` and the contour sum comes out of the SMALL space as
  ``V_m (sum_j w_j y_j)``, one tensordot.  Its accuracy demand is also the
  looser one (the filtered block feeds a Rayleigh-Ritz, not a W tile), which
  is exactly the regime where a shared space's slower per-shift convergence
  costs least.  What adoption takes: replace ``pole_body``'s ``fori_loop``
  with one ``shifted_solve_core`` call per Ritz vector; ``z_nodes`` are already
  a runtime array, and the FEAST cache key needs the Arnoldi dimension where
  ``max_iter`` sits today.  Its ``use_conjugate_symmetry`` arm (which halves
  the pole count by taking ``2 Re``) stays a caller-side concern.
* **The MPA z-plan** (``gw/mpa/model.py``'s per-z ``_solve_wc``) is the same
  ~20-shift family one level up, and reaches this module through the ladder
  facade's ``wc_source`` seam without further work.

The one thing an adopter MUST carry across is the true-residual gate: a shared
space under-converges outlying shifts silently, and the loop bound is not a
convergence signal.

DEEP BLOCKS — what a wide block-Arnoldi would and would not buy
---------------------------------------------------------------
Shift sharing composes with WIDTH: one Arnoldi space could be blocked over
(probe columns x shifts) so the dominant contractions become large GEMMs
instead of many thin ones.  Whether that pays is a question about the ring
matvec's SHAPES, and on this operator the shapes give a split answer.

MEASURED FIRST, because it bounds everything else.  Cost of ONE matvec per
probe column vs the matvec's batch width ``b``, for the LADDER operator and
for the same operator with the direct rung removed (``include_w=False``, i.e.
the RPA screening matvec) — the second row is the control that says which part
of the operator depth actually helps:

    b                   1       2       4       8      16      32      64
    LADDER  ms/col    9.57    9.40    9.99    9.05    9.02    8.99    9.01
    RPA     ms/col   0.458   0.104   0.208   0.065   0.068   0.058   0.056

**The RPA matvec amortises 8.1x over a 64-wide block.  The ladder matvec
amortises 6%.**  And the ladder is 21x dearer than the RPA one at b=1 (9.57 vs
0.458 ms) and 160x dearer at b=64 — so the direct rung is ~95% of a ladder
matvec, and it is precisely the 95% that depth does not touch.

This is the answer to "deep blocks to take advantage of more gemms", and it is
operator-dependent: **worth a lot for w_rpa, almost nothing for w_bse.**

The shapes say why.  Reading the einsum subscripts out of ``bse_ring_comm``
(``b`` = batch width, ``mu`` = N_mu, local extents on a px x py mesh):

* ``apply_V_ring``'s Coulomb contraction ``'MN,bNk->bMk'`` is the operator's
  only ``N_mu x N_mu`` GEMM, and at ``b = 1`` it is a **GEMV** — M=mu, K=mu,
  N=1.  This one is pure win from depth: N grows to ``b``, ``V_q0`` is an
  INPUT and does not grow, and the arithmetic intensity rises linearly.
  Measured, this is the whole RPA matvec, and it is the whole 8.1x.
* the direct rung is the opposite.  ``encode`` (``'kvsN,bcvk->bcksN'`` then
  ``'kctM,bcksN->bMNtsk'``) and ``decode`` carry ``b`` as an OUTPUT axis over
  contraction lengths ``nc`` = 20 / ``nv`` = 26 — K stays SMALL no matter how
  deep the block gets, so these are thin GEMMs at every width; and between them
  sits ``apply_W_from_T`` — an ifft, an ELEMENTWISE multiply by ``W_R``, and an
  fft, all on a ``b``-scaled buffer.  That middle is pure bandwidth with no
  GEMM in it at all, it is the ladder's dominant cost (the HLO audit measured
  the ladder matvec's peak at 5.3x the RPA one, all of it the T/T_R/U_R/U_q/U
  chain), and it is exactly linear in ``b``.  Making the rung benefit from
  depth is not a blocking question at all — it is a question about the K
  dimensions ``nc``/``nv`` and about fusing the fft/elementwise/fft triple.

So the 7% is not a tuning failure: for the LADDER operator the part that would
benefit from depth is not the part that costs.

AND DEPTH COLLIDES WITH THE STAGING BUDGET.  ``T`` is the rung's W_R-class
intermediate, and ``bse_ring_comm._ring_sum_conduction`` allocates it as
``zeros((R.shape[0], mu_local, nu_local, ns, ns, nk))`` — ``R.shape[0]`` IS
the batch width.  One ``T`` is ``mu_local x nu_local x ns^2 x nk``; a block of
width ``b`` holds ``b`` of them.  On a 10^4-centroid deck at px=py=32 that is
~0.1 GiB per unit of width, so a width-64 block would stage ~6.4 GiB of
W_R-class intermediate where the budget is "one, maybe four".

MEASURED on the bench fixture (n_mu=399, ns=2, nk=9, 1x1 mesh): one ``T`` is
**0.085 GiB**, so a width-64 block stages **5.47 GiB** of it.  The staging
budget and the width at which GEMM depth starts paying are therefore in direct
conflict, which is exactly why the budget is "one, maybe four".

The consequence is a design constraint, not a veto: **block width for the
GEMMs and block width through the rung must be different numbers.**  Widen the
probe/shift block so ``apply_V_ring``'s ``'MN,bNk->bMk'`` becomes a real GEMM,
then STREAM that block through the direct rung in sub-chunks of <= 4 so ``T``
never exceeds the staging budget.  Nothing in the shifted-Krylov algebra
objects — the Arnoldi's matvec is a black box to it — but the rung applier
would need an inner chunk loop it does not have today.

HOISTING, VERIFIED (2026-08-16, by reading the call graph)
----------------------------------------------------------
* **Pair densities are hoisted per operator, not per matvec.** ``M_X``/``M_Y``
  are payload slots threaded through ``bse_feast.matvec_operands`` as runtime
  arguments (audit P3), and ``apply_V_ring`` takes ``M_X`` as a parameter —
  the matvec never rebuilds them.  The direct rung uses no pair density at all
  (it contracts ``psi`` through ``T``), and its per-matvec work is a genuine
  function of the trial vector, so there is nothing hoistable left in it.
* **One exception, and the seed hoist covers it — and it matters most exactly
  where the ladder does not.**
  ``bse_ring_comm.build_realspace_random_transition_generator``'s inner
  ``_map`` rebuilds ``M_X`` locally (``'kcsm,kvsm->kcvm'``) instead of taking
  the hoisted slot.  That is the SEED, which the production path re-dispatches
  once per (q, chunk, z); hoisting it to per-(q, chunk) — which every entry
  point here does by construction — makes that recompute nz times rarer.
  Measured on a 64-column probe block: the seed costs 5.8-12.4 ms, which is
  **0.6 matvec-equivalents against the LADDER matvec but 27 against the RPA
  one**.  So on the w_rpa path the seed hoist alone is worth ~27 x (nz-1)
  matvecs per (q, chunk) — the single largest per-sweep saving this arm found,
  and it is free.
* **W_R is built once and held in real-space form for the whole sweep.**
  ``w_ladder.build_ladder_resolvent`` calls ``ensure_W_R(include_W=True)``
  once, ``sweep_q_wedge`` calls it outside the q loop and raises if ``W_R``
  is missing afterwards, and ``build_finite_q_data`` returns a shallow copy
  that carries the same array (it never touches ``W_q``).  This module never
  calls ``ensure_W_R``: it receives ``W_R`` inside ``operands``.

WHAT THIS MODULE IS NOT
-----------------------
It is a solver-side experiment and nothing else.  It builds no operator,
assembles no payload, walks no q-wedge and changes no convention: the caller
hands it the ``(matvec, operands)`` pair ``w_ladder.build_ladder_resolvent``
already returns and the ``gen``/``snapshot`` stages
``apply_screening_resolvent_block`` already uses, and every entry point here is
a drop-in for that function's stage 2 over a z-LIST.  There is deliberately no
second ``sweep_q_wedge`` in this file — one q walk exists, in ``w_ladder``, and
integrating any of this means giving THAT loop a z-list-aware inner call, not
standing a parallel loop beside it.

Because every arm runs the SAME ``matvec`` object on the SAME payload, the
comparison is insensitive to any concurrent change of the ladder's finite-q
convention.
"""
from __future__ import annotations

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap().
# MUST run before this module's own `import jax` (same rule as w_ladder).
from runtime import bootstrap

bootstrap()

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
from jax.sharding import NamedSharding, PartitionSpec as P

from .bse_feast import _apply_shifted_matvec, matvec_operands
from common.collectives import device_put_process_local

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Stage 1 — the SEED, hoisted out of the z loop.
# ---------------------------------------------------------------------------
def seed_probe_block(G_zeta, data, gen, sh) -> jax.Array:
    """``rhs = [f; -f]`` for a probe block — the z-INDEPENDENT stage 1.

    Byte-for-byte the seed ``apply_screening_resolvent_block`` builds (same
    ``device_put_process_local`` broadcast, same ``gen`` call, same sharding
    constraints), lifted to its own function so a z-list caller pays it ONCE.
    Returns ``sh.X_full`` = ``(2, n_probe, c, v, k)``.
    """
    px, py = sh.X.mesh.devices.shape
    n_probe = int(G_zeta.shape[0])
    if n_probe % py != 0:
        raise ValueError(
            f"probe block n_probe={n_probe} must be a multiple of py={py} "
            "(reduce-scatter tiles nu over y); pad the probe block with zero "
            "rows.")
    n_rmu = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    G = np.asarray(G_zeta, dtype=np.float64)
    r = device_put_process_local(
        np.broadcast_to(G[:, :, None], (n_probe, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    return jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)


# ---------------------------------------------------------------------------
# Stage 2 — ONE Arnoldi space, every shift.
# ---------------------------------------------------------------------------
def _shift_lsq(Hbar, beta, z_vec, k_live, m):
    """Per-shift ``y_j`` and PROJECTED residual at Arnoldi dimension ``k_live``.

    ``A_j = z_j E_m - Hbar`` with the not-yet-built columns MASKED to zero and
    then FILLED with unit subdiagonal columns ``e_{k+1}``.  Both halves matter:

    * the mask, because ``E_m``'s untouched columns are ``z_j e_k`` and NOT
      zero, so an unmasked solve would place weight on Krylov directions that
      do not exist yet;
    * the fill, because after masking those columns are identically zero and
      the least-squares problem is rank-deficient.  A unit column ``e_{k+1}``
      with ``k >= k_live`` lives in rows ``k_live+1 .. m``, which the live
      block (rows ``0..k_live``) and ``g = beta e_1`` both avoid entirely, so
      the filled columns are orthogonal to everything that matters, their
      ``y`` is exactly zero, and the residual is unchanged — while the matrix
      becomes full column rank and a QR solve is legitimate.

    QR and NOT the normal equations, for the reason
    ``bse_feast._gmres_solve_core`` records: the finite-q screening operator
    carries a large G=0 Coulomb head, ``cond(H) ~ 1e8``, and ``H^H H`` is then
    at the edge of double precision.  QR rather than the production core's
    ``lstsq`` (SVD) because this solve runs once per (column, check-block)
    rather than once per iteration of one system — measured on the bench
    fixture an SVD of the ``(m+1) x m`` Hessenberg is the single most expensive
    non-matvec op in the shared solve, and the fill above removes the only
    reason SVD was needed.  Correctness does not rest on it either way: every
    caller gates on the explicitly recomputed TRUE residual.
    """
    idx = jnp.arange(m)
    live = (idx < k_live).astype(Hbar.dtype)[None, :]
    E = jnp.eye(m + 1, m, dtype=Hbar.dtype)
    sub = jnp.eye(m + 1, m, k=-1, dtype=Hbar.dtype)
    fill = sub * (jnp.asarray(1.0, Hbar.dtype) - live)
    g = jnp.zeros((m + 1,), dtype=Hbar.dtype).at[0].set(beta.astype(Hbar.dtype))

    def one(z):
        A = z * (E * live) - Hbar + fill
        Q, R = jnp.linalg.qr(A)
        y = jsla.solve_triangular(R, Q.conj().T @ g, lower=False)
        y = y * live[0]
        return y, jnp.linalg.norm(g - A @ y)

    return jax.vmap(one)(z_vec)


def shifted_solve_core(matvec, b, z_vec, operands, m, tol, check_every,
                       basis_sh=None, weights=None):
    """ONE Arnoldi space on ``H`` seeded by ``b``; all shifts solved in it.

    THE GENERIC CAPABILITY — shape-agnostic and ladder-agnostic.  It knows
    nothing about probe blocks, W tiles, centroids or the ladder: it takes any
    ``(matvec, operands)`` pair the ``bse_feast`` solver layer already speaks,
    any single right-hand vector ``b`` that ``matvec`` accepts, and a list of
    shifts, and returns the solution of ``(z_j - H) x_j = b`` for every ``j``.
    Everything ladder-specific in this module is a wrapper around it.  See the
    module docstring's ADOPTION section for the two other in-tree families
    (FEAST's quadrature poles, the MPA z-plan) this is meant to serve.

    ``b``
        one right-hand vector in whatever layout ``matvec`` consumes — the W
        path passes a probe column ``(2, 1, c, v, k)``, a TDA caller would pass
        ``(1, c, v, k)``.  Nothing here inspects the shape.
    ``z_vec``
        ``(nz,)`` complex shifts.  ``nz`` is STATIC (it sizes the small solve).
    ``weights``
        optional ``(nz,)``.  When given, the return's ``X`` is the single
        CONTOUR SUM ``sum_j w_j x_j`` with ``b``'s own shape instead of the
        ``(nz,) + b.shape`` stack — the FEAST shape, and the reason a shared
        space is especially cheap there: the sum is
        ``V_m (sum_j w_j y_j)``, ONE tensordot for the whole contour.
    ``basis_sh``
        constrains the ``(m+1,) + b.shape`` basis: the LEADING axis is a basis
        INDEX and stays replicated, every data axis keeps the caller's
        placement.  Stated rather than left to propagation because "the basis
        is ``m+1`` vectors of the caller's class, none of them replicated" is
        the whole memory claim of the shared space.

    Returns ``(X, proj_resid, true_resid, k_used)``.  The Arnoldi is
    unpreconditioned and starts at ``x0 = 0`` — both are forced by the
    shift-invariance theorem in the module docstring, not preferences.
    """
    nz = int(z_vec.shape[0])
    beta = jnp.linalg.norm(b)
    zero = jnp.asarray(0.0, dtype=beta.dtype)
    beta_safe = jnp.where(beta == zero, jnp.asarray(1.0, beta.dtype), beta)
    v0 = jnp.where(beta == zero, b, b / beta_safe.astype(b.dtype))

    def _pin(x):
        return x if basis_sh is None else jax.lax.with_sharding_constraint(
            x, basis_sh)

    V = _pin(jnp.zeros((m + 1,) + b.shape, dtype=b.dtype).at[0].set(v0))
    Hbar = jnp.zeros((m + 1, m), dtype=b.dtype)

    def arnoldi_step(k, carry):
        V, Hbar = carry
        w = matvec(V[k], *operands).astype(b.dtype)

        # Classical Gram-Schmidt + a second (DGKS) pass.  The production core
        # records WHY the second pass is not optional here: on the stiff
        # finite-q screening operator a single pass loses orthogonality
        # catastrophically (||V^H V - I|| -> O(1) by ~20 steps), which makes
        # the projected residual falsely tiny.  For a SHARED space that is
        # worse than for a per-shift one — the projected residual is the only
        # cheap per-shift convergence signal there is.
        def cgs(i, c):
            w_, H_ = c
            h = jnp.vdot(V[i], w_)
            return w_ - h * V[i], H_.at[i, k].set(h)

        w, Hbar = jax.lax.fori_loop(0, k + 1, cgs, (w, Hbar))

        def reorth(i, c):
            w_, H_ = c
            h = jnp.vdot(V[i], w_)
            return w_ - h * V[i], H_.at[i, k].add(h)

        w, Hbar = jax.lax.fori_loop(0, k + 1, reorth, (w, Hbar))

        h_next = jnp.linalg.norm(w)
        Hbar = Hbar.at[k + 1, k].set(h_next.astype(b.dtype))
        # Happy breakdown (h_next == 0): the Krylov space is H-invariant and
        # the solution is already exact in it.  Keep the zero vector rather
        # than dividing; the masked lstsq below then never uses that column.
        h_safe = jnp.where(h_next == 0.0, jnp.asarray(1.0, h_next.dtype), h_next)
        V = V.at[k + 1].set(w / h_safe.astype(b.dtype))
        return V, Hbar

    def cond(state):
        k, rel, _, _ = state
        return jnp.logical_and(k < m, rel > tol)

    def body(state):
        k, _, V, Hbar = state
        k_next = jnp.minimum(k + check_every, m)
        V, Hbar = jax.lax.fori_loop(k, k_next, arnoldi_step, (V, Hbar))
        _, rs = _shift_lsq(Hbar, beta, z_vec, k_next, m)
        return k_next, jnp.max(rs) / beta_safe, _pin(V), Hbar

    init = (jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=beta.dtype), V, Hbar)
    k_used, _, V, Hbar = jax.lax.while_loop(cond, body, init)

    ys, rs = _shift_lsq(Hbar, beta, z_vec, k_used, m)
    proj = jnp.where(beta == zero, zero, rs / beta_safe)

    # x_j = V_m y_j.  Zero rhs -> zero solution (the pad-column contract):
    # beta == 0 makes v0 == b == 0, so the tensordot is already zero.
    Vm = V[:m]
    xs, trues, rs_vec = [], [], []
    for j in range(nz):
        x = jnp.tensordot(ys[j], Vm, axes=(0, 0))
        r_true = b - _apply_shifted_matvec(matvec, x, z_vec[j], operands)
        xs.append(x)
        rs_vec.append(r_true)
        trues.append(jnp.where(beta == zero, zero,
                               jnp.linalg.norm(r_true) / beta_safe))
    # The residual VECTORS come back too, not just their norms: a recycling
    # caller (:func:`solve_hybrid_block`) needs exactly ``r_j`` to open its
    # correction equation, and recomputing it there would pay nz matvecs for an
    # array this function already built for the gate.
    R = jnp.stack(rs_vec, axis=0)
    if weights is not None:
        # Contour sum in the SMALL space: sum_j w_j x_j = V_m (sum_j w_j y_j).
        y_sum = jnp.tensordot(jnp.asarray(weights, dtype=ys.dtype), ys,
                              axes=(0, 0))
        return (jnp.tensordot(y_sum, Vm, axes=(0, 0)), proj,
                jnp.stack(trues, axis=0), k_used, R)
    return (jnp.stack(xs, axis=0), proj, jnp.stack(trues, axis=0), k_used, R)


#: ``(id(matvec), nz, m, tol, check_every, dtype) -> jitted block engine``.
#: Same discipline as ``bse_w_exact._get_block_gmres_solver``: the operator
#: STRUCTURE is baked, every q/z-dependent tensor is a runtime arg, and the
#: cache holds a reference to ``matvec`` so its ``id()`` cannot be recycled.
_SHIFTED_BLOCK_CACHE: dict = {}


def _get_shifted_block_solver(matvec, sh, nz, m, tol, check_every, dtype):
    key = (id(matvec), int(nz), int(m), float(tol), int(check_every), str(dtype))
    hit = _SHIFTED_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit[1]

    # Basis placement: (m+1, 2, 1, c, v, k) — leading axis is the Krylov index
    # (replicated by construction), the rest is sh.X_full's placement.
    basis_sh = NamedSharding(sh.X_full.mesh, P(None, None, None, "x", "y", None))

    @jax.jit
    def _block(rhs, z_vec, operands):
        rhs_scan = jnp.moveaxis(rhs, 1, 0)          # (nu, 2, c, v, k)

        def _solve_col(carry, rhs_col):
            rhs_i = rhs_col[:, None]                # (2, 1, c, v, k)
            X, proj, true, k_used, _R = shifted_solve_core(
                matvec, rhs_i, z_vec, operands, m, tol, check_every,
                basis_sh=basis_sh)
            # Readout s = x[0] + x[1], per shift, in the production layout.
            s = tuple(
                jax.lax.with_sharding_constraint(X[j][0] + X[j][1], sh.X)[0]
                for j in range(nz))
            return carry, (s, proj, true, k_used)

        # unroll=1: one Krylov space alive at a time (the production engine's
        # invariant).  Unrolling would multiply the (m+1)-vector basis by the
        # unroll factor for no dispatch saving inside an already-jitted scan.
        _, (s_all, proj, true, ks) = jax.lax.scan(
            _solve_col, None, rhs_scan, unroll=1)
        s_all = tuple(jax.lax.with_sharding_constraint(s, sh.X) for s in s_all)
        return s_all, proj, true, ks

    _SHIFTED_BLOCK_CACHE[key] = (matvec, _block)
    return _block


def solve_shifted_block(G_zeta, z_list, data, matvec, gen, snapshot, sh, *,
                        arnoldi_dim, tol, check_every=25, rhs=None):
    """``W(z) - v`` tiles for a probe block at EVERY ``z`` from ONE Arnoldi space.

    The shifted-system analog of
    :func:`bse_w_exact.apply_screening_resolvent_block`, with the same three
    stages and the same reshard boundaries — stages 1 and 3 are that function's
    verbatim (:func:`seed_probe_block` is its stage-1 body; ``snapshot`` is its
    stage 3), only the middle changes.  It takes the whole z-LIST, so the seed
    is paid once (the hoist of the module docstring).

    Returns ``(tiles, proj, true, steps)``:

    ``tiles``
        length-``nz`` list of ``sh.V = P('x','y')`` device tiles, column ``i``
        of ``tiles[j]`` being ``W(z_j) - v`` for probe ``i`` — the same object
        the production path returns per z.
    ``proj`` / ``true``
        ``(nz, n_probe)`` — the projected (Hessenberg least-squares) and the
        explicitly recomputed ``||b - (z-H)x|| / ||b||`` relative residuals.
        GATE ON ``true``.
    ``steps``
        ``(n_probe,)`` Arnoldi dimension actually used; ``== arnoldi_dim`` is
        the TRUNCATION signature, not a success.
    """
    z_vec = jnp.asarray(np.asarray(z_list, dtype=np.complex128))
    nz = int(z_vec.shape[0])
    if arnoldi_dim % check_every != 0:
        raise ValueError(
            f"arnoldi_dim={arnoldi_dim} must be a multiple of "
            f"check_every={check_every}: the shared space is grown in blocks "
            "and the per-shift residual is evaluated at block boundaries only.")
    if rhs is None:
        rhs = seed_probe_block(G_zeta, data, gen, sh)
    solver = _get_shifted_block_solver(
        matvec, sh, nz, int(arnoldi_dim), float(tol), int(check_every),
        rhs.dtype)
    s_all, proj, true, steps = solver(rhs, z_vec, matvec_operands(data))
    tiles = [snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
             for s in s_all]
    return tiles, proj.T, true.T, steps


# ---------------------------------------------------------------------------
# Calibration — the per-shift convergence history of the shared space.
# ---------------------------------------------------------------------------
def shared_space_history(matvec, b_col, z_list, operands, *, m, stride):
    """Projected per-shift residual of the SHARED space vs Arnoldi dimension.

    ``b_col`` is one probe column ``(2, 1, c, v, k)``.  Returns
    ``(dims (n,), resid (n, nz))`` — the whole convergence curve for every
    shift from ONE run of ``m`` matvecs, which is what makes choosing
    ``arnoldi_dim`` a measurement instead of a guess.  Runs eagerly per block
    (host-side loop) because it is a one-column diagnostic, not a hot path.
    """
    z_vec = jnp.asarray(np.asarray(z_list, dtype=np.complex128))
    beta = jnp.linalg.norm(b_col)

    @jax.jit
    def _grow(V, Hbar, k0):
        def step(k, carry):
            V, Hbar = carry
            w = matvec(V[k], *operands).astype(b_col.dtype)

            def cgs(i, c):
                w_, H_ = c
                h = jnp.vdot(V[i], w_)
                return w_ - h * V[i], H_.at[i, k].set(h)

            w, Hbar = jax.lax.fori_loop(0, k + 1, cgs, (w, Hbar))

            def reorth(i, c):
                w_, H_ = c
                h = jnp.vdot(V[i], w_)
                return w_ - h * V[i], H_.at[i, k].add(h)

            w, Hbar = jax.lax.fori_loop(0, k + 1, reorth, (w, Hbar))
            hn = jnp.linalg.norm(w)
            Hbar = Hbar.at[k + 1, k].set(hn.astype(b_col.dtype))
            hs = jnp.where(hn == 0.0, jnp.asarray(1.0, hn.dtype), hn)
            V = V.at[k + 1].set(w / hs.astype(b_col.dtype))
            return V, Hbar

        return jax.lax.fori_loop(k0, k0 + stride, step, (V, Hbar))

    @jax.jit
    def _resid(Hbar, k_live):
        _, rs = _shift_lsq(Hbar, beta, z_vec, k_live, m)
        return rs / beta

    V = jnp.zeros((m + 1,) + b_col.shape, dtype=b_col.dtype).at[0].set(
        b_col / beta.astype(b_col.dtype))
    Hbar = jnp.zeros((m + 1, m), dtype=b_col.dtype)
    dims, hist = [], []
    for k0 in range(0, m, stride):
        V, Hbar = _grow(V, Hbar, k0)
        dims.append(k0 + stride)
        hist.append(np.asarray(jax.device_get(_resid(Hbar, k0 + stride))))
    return np.asarray(dims, dtype=int), np.asarray(hist)


# ---------------------------------------------------------------------------
# The production GMRES core with the TWO parameters recycling needs.
# ---------------------------------------------------------------------------
def _precond_gmres_scaled(matvec, b, diag_h, z, operands, max_iter, tol, scale):
    """``bse_feast._gmres_solve_core`` with the stopping norm supplied.

    THIS IS A TWIN OF PRODUCTION CODE AND IT SHOULD NOT SURVIVE INTEGRATION.
    It is line-for-line ``bse_feast._gmres_solve_core`` — same diagonal right
    preconditioner, same DGKS second pass, same ``lstsq`` least-squares, same
    while-loop exit — with ONE change: the convergence test divides by a
    RUNTIME ``scale`` instead of by ``||b||``.

    Why that one change cannot be worked around.  Every recycling scheme solves
    a CORRECTION equation ``(z - H) d = r`` with ``r = b - (z - H) x_prev``, and
    wants ``||r - (z-H)d|| <= tol * ||b||`` — the tolerance of the ORIGINAL
    system.  The production core can only ask for ``tol * ||r||``, so once the
    recycled guess has bought a factor ``eta = ||r||/||b||`` the correction
    solve over-converges by exactly that factor and spends back the iterations
    the recycling saved.  On the bench operator ``eta ~ 1e-2`` after a short
    shared space, i.e. two orders, i.e. ~6 of the 23 iterations — enough to
    turn a real win into no win at all.

    ``tol`` is compared inside a ``lax.while_loop`` cond, where a traced value
    is perfectly legal, so upstreaming this is a signature change and not an
    algorithm change: give ``_gmres_solve_core`` an optional ``x0`` and let its
    ``tol`` be a runtime argument, and this function and both recycling arms
    below collapse into calls to it.  Until then the twin is here, in the
    experimental module, where it cannot drift into the production path.

    ``scale = ||b||`` reproduces the production core exactly.
    """
    one = jnp.asarray(1.0, dtype=b.dtype)
    m_inv = one / (z - diag_h)
    if m_inv.ndim == b.ndim - 1:
        m_inv = m_inv[None, ...]

    x0 = m_inv * b
    r0 = b - _apply_shifted_matvec(matvec, x0, z, operands).astype(b.dtype)
    beta = jnp.linalg.norm(r0)

    zero = jnp.asarray(0.0, dtype=beta.dtype)
    scale_safe = jnp.where(scale == zero, jnp.asarray(1.0, beta.dtype), scale)
    v0 = jnp.where(beta == zero, r0, r0 / jnp.where(beta == zero, one, beta))

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
        v_k = V[k]
        z_k = m_inv * v_k
        Z = Z.at[k].set(z_k)
        w = _apply_shifted_matvec(matvec, z_k, z, operands).astype(b.dtype)

        def arnoldi(i, carry):
            w_l, H_l = carry
            h = jnp.vdot(V[i], w_l)
            return w_l - h * V[i], H_l.at[i, k].set(h)

        w, H = jax.lax.fori_loop(0, k + 1, arnoldi, (w, H))

        def reorth(i, carry):
            w_l, H_l = carry
            corr = jnp.vdot(V[i], w_l)
            return w_l - corr * V[i], H_l.at[i, k].add(corr)

        w, H = jax.lax.fori_loop(0, k + 1, reorth, (w, H))
        h_next = jnp.linalg.norm(w)
        H = H.at[k + 1, k].set(h_next)
        V = V.at[k + 1].set(jnp.where(h_next == 0.0, w, w / h_next))
        y = jnp.linalg.lstsq(H, g, rcond=None)[0]
        resid = jnp.linalg.norm(g - H @ y)
        # THE one difference from the production core: the stopping norm.
        return k + 1, resid / scale_safe, V, Z, H, g, y

    init = (0, jnp.asarray(jnp.inf, dtype=beta.dtype), V, Z, H, g, y)
    k_final, _, V, Z, H, g, y = jax.lax.while_loop(cond, body, init)
    return x0 + jnp.tensordot(y, Z, axes=(0, 0)), k_final


# ---------------------------------------------------------------------------
# Approach 2 — initial-guess chaining across neighbouring shifts.
# ---------------------------------------------------------------------------
#: The production preconditioner is KEPT, whole.  Solving ``(z_j - H) x = b``
#: from an initial guess ``x_prev`` is exactly solving
#: ``(z_j - H) d = b - (z_j - H) x_prev`` from zero and adding, so the algorithm
#: is the production one and only the right-hand side changes.  Ordering the
#: shifts by proximity is the whole idea: ``x(z_{j-1})`` is a guess for
#: ``x(z_j)`` in proportion to how close the two shifts are, and MPA's own
#: ordering (near line, then far line) jumps 1.8 Ry at the seam.
#:
#: The one thing it cannot inherit is the stopping test — see
#: :func:`_precond_gmres_scaled`, which is that engine with the stopping norm
#: made a runtime argument, and which exists ONLY because the production core
#: measures its residual against the correction rhs and would spend back every
#: iteration the guess saved.
_CHAINED_BLOCK_CACHE: dict = {}


def _get_chained_block_solver(matvec, sh, nz, max_iter, tol, dtype):
    key = (id(matvec), int(nz), int(max_iter), float(tol), str(dtype))
    hit = _CHAINED_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit[1]

    @jax.jit
    def _block(rhs, diag_h, z_vec, operands):
        rhs_scan = jnp.moveaxis(rhs, 1, 0)

        def _solve_col(carry, rhs_col):
            rhs_i = rhs_col[:, None]
            x = jnp.zeros_like(rhs_i)
            outs, resids, iters = [], [], []
            nrhs = jnp.linalg.norm(rhs_i)
            nrhs_safe = jnp.where(nrhs == 0.0, jnp.asarray(1.0, nrhs.dtype), nrhs)
            for j in range(nz):
                zj = z_vec[j]
                # j == 0 starts from x = 0, so the correction equation IS the
                # original one and the opening matvec is skipped (a STATIC
                # branch: nz is a compile-time constant).  That keeps the first
                # link bit-identical to the production solve.
                r = (rhs_i if j == 0 else
                     rhs_i - _apply_shifted_matvec(matvec, x, zj, operands))
                d, k = _precond_gmres_scaled(matvec, r, diag_h, zj, operands,
                                             max_iter, tol, nrhs)
                x = x + d
                r_true = rhs_i - _apply_shifted_matvec(matvec, x, zj, operands)
                outs.append(jax.lax.with_sharding_constraint(
                    x[0] + x[1], sh.X)[0])
                resids.append(jnp.where(nrhs == 0.0,
                                        jnp.asarray(0.0, nrhs.dtype),
                                        jnp.linalg.norm(r_true) / nrhs_safe))
                iters.append(k)
            return carry, (tuple(outs), jnp.stack(resids), jnp.stack(iters))

        _, (s_all, resids, iters) = jax.lax.scan(
            _solve_col, None, rhs_scan, unroll=1)
        s_all = tuple(jax.lax.with_sharding_constraint(s, sh.X) for s in s_all)
        return s_all, resids, iters

    _CHAINED_BLOCK_CACHE[key] = (matvec, _block)
    return _block


def solve_chained_block(G_zeta, z_list, data, matvec, diag_h, gen, snapshot, sh,
                        *, max_iter, tol, rhs=None):
    """Approach 2: per-shift PRECONDITIONED solves, chained by initial guess.

    ``z_list`` must already be ordered by proximity — the chain is only as good
    as ``x(z_{j-1})`` is a guess for ``x(z_j)``.  Keeps the production engine
    and the production preconditioner; shares only the ITERATE, not the space.
    Returns ``(tiles, true_resid (nz, n_probe), iters (nz, n_probe))``, where
    ``iters[j]`` counts the iterations of the j-th CORRECTION solve, so the
    total matvec count of a column is ``sum_j (iters[j] + 2)`` (the two extra
    are the residual evaluations that open and close each link).
    """
    z_vec = jnp.asarray(np.asarray(z_list, dtype=np.complex128))
    nz = int(z_vec.shape[0])
    if rhs is None:
        rhs = seed_probe_block(G_zeta, data, gen, sh)
    solver = _get_chained_block_solver(
        matvec, sh, nz, int(max_iter), float(tol), rhs.dtype)
    s_all, resids, iters = solver(rhs, diag_h, z_vec, matvec_operands(data))
    tiles = [snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
             for s in s_all]
    return tiles, resids.T, iters.T


# ---------------------------------------------------------------------------
# Approach 1b — SHORT shared space + preconditioned polish (the hybrid).
# ---------------------------------------------------------------------------
#: The shape of the measured convergence curve says the two approaches are
#: complementary rather than rival, and the hybrid is where that cashes out.
#:
#: On the bench operator (gnppm_debug ladder, MPA ``double_parallel_grid``
#: n_p=4) the FAR-line shifts — the ones a long way off the real axis — are
#: solved by the shared space to machine precision within ~60 matvecs, while
#: the NEAR-line ones (``z = 0`` and ``Im z = 0.2 Ry``) are still at ~1e-2 there
#: and need several hundred more.  That is not an accident of this deck: the
#: near-line shifts sit AT the spectrum, where ``(z - H)`` is nearly singular,
#: and the diagonal preconditioner's power on them is physical — ``1/(z - d)``
#: IS the resonant denominator that makes them hard.  An unpreconditioned
#: shared space cannot reproduce it, and no preconditioner can be shared (the
#: similarity theorem in the module docstring).
#:
#: So: run a SHORT shared space, which costs ``m`` matvecs and finishes the
#: easy shifts outright; then hand each unfinished shift's residual to the
#: production preconditioned engine as a correction equation.  Cost per column
#: ``m + sum_j (k_j + 2)``, with ``k_j`` collapsing to ~1 for every shift the
#: shared space already finished.
#:
#: MEASURED, AND IT DOES NOT PAY — the paragraph above is the hypothesis, and
#: the bench refuted it.  5 q_irr x 64 probe columns, nz = 8, tol 1e-8, matvecs
#: per column per shift: baseline **12.72**, hybrid at m=50 **14.34**.  The
#: hybrid's polish iteration counts came back min/med/max = **1/1/20** — the
#: shared space really does finish five of the eight shifts, exactly as the
#: calibration promised — and it still lost, because THE PRECONDITIONED SOLVE
#: FINDS THOSE SAME FIVE SHIFTS CHEAP TOO: the baseline's own per-shift counts
#: on this workload run 6/8/21, so the five shifts the shared space "saves"
#: cost the baseline about 8 matvecs apiece, ~40 in total, against the 50 + 8
#: the shared space charges to replace them.
#:
#: The reason generalises past this deck, and it is the finding that matters:
#: BOTH methods order the shifts by the same thing — distance from the
#: spectrum.  A shift far off the real axis is easy for an unpreconditioned
#: Krylov space AND easy for a diagonally preconditioned one; a shift sitting
#: on the spectrum is hard for the first and merely ordinary for the second.
#: There is no complementarity to exploit, so a shared space has no regime in
#: which it is the cheaper way to reach a given shift.  Krylov sharing pays
#: only where the per-shift solve is EXPENSIVE for a reason sharing removes —
#: an unpreconditioned or badly preconditioned family — and this one is
#: neither.
#:
#: Kept, not deleted, because it is the arm that makes that statement
#: measurable, and because it is the only correct-at-z=0 form of the shared
#: space (see the gate: pure shifted GMRES at m=300 returns 5.2e-06 relative
#: error at z = 0 while the hybrid returns 8.5e-10 at the same q).
_HYBRID_BLOCK_CACHE: dict = {}


def _get_hybrid_block_solver(matvec, sh, nz, m, check_every, max_iter, tol,
                             dtype):
    key = (id(matvec), int(nz), int(m), int(check_every), int(max_iter),
           float(tol), str(dtype))
    hit = _HYBRID_BLOCK_CACHE.get(key)
    if hit is not None:
        return hit[1]

    basis_sh = NamedSharding(sh.X_full.mesh, P(None, None, None, "x", "y", None))

    @jax.jit
    def _block(rhs, diag_h, z_vec, operands):
        rhs_scan = jnp.moveaxis(rhs, 1, 0)

        def _solve_col(carry, rhs_col):
            rhs_i = rhs_col[:, None]
            # Stage A — the shared space.  tol=0 runs it to the full m: the
            # exit test is per-shift and the polish below is what decides
            # "converged", so an early exit here would only leave the easy
            # shifts less finished than they could be for free.
            X, _proj, sh_true, _k, R = shifted_solve_core(
                matvec, rhs_i, z_vec, operands, m, 0.0, check_every,
                basis_sh=basis_sh)
            nrhs = jnp.linalg.norm(rhs_i)
            nrhs_safe = jnp.where(nrhs == 0.0, jnp.asarray(1.0, nrhs.dtype),
                                  nrhs)
            outs, resids, iters = [], [], []
            for j in range(nz):
                zj = z_vec[j]
                # Stage B — preconditioned polish on the CORRECTION equation,
                # with the stopping norm set to ||b|| so a shift the shared
                # space already finished exits at k=0 instead of grinding the
                # correction down another eight orders for nothing.
                d, k = _precond_gmres_scaled(matvec, R[j], diag_h, zj,
                                             operands, max_iter, tol, nrhs)
                x = X[j] + d
                r_true = rhs_i - _apply_shifted_matvec(matvec, x, zj, operands)
                outs.append(jax.lax.with_sharding_constraint(
                    x[0] + x[1], sh.X)[0])
                resids.append(jnp.where(nrhs == 0.0,
                                        jnp.asarray(0.0, nrhs.dtype),
                                        jnp.linalg.norm(r_true) / nrhs_safe))
                iters.append(k)
            return carry, (tuple(outs), jnp.stack(resids), jnp.stack(iters),
                           sh_true)

        _, (s_all, resids, iters, sh_true) = jax.lax.scan(
            _solve_col, None, rhs_scan, unroll=1)
        s_all = tuple(jax.lax.with_sharding_constraint(s, sh.X) for s in s_all)
        return s_all, resids, iters, sh_true

    _HYBRID_BLOCK_CACHE[key] = (matvec, _block)
    return _block


def solve_hybrid_block(G_zeta, z_list, data, matvec, diag_h, gen, snapshot, sh,
                       *, arnoldi_dim, max_iter, tol, check_every=25,
                       rhs=None):
    """Short shared Arnoldi space, then a preconditioned polish per shift.

    Returns ``(tiles, true_resid (nz, n_probe), polish_iters (nz, n_probe),
    shared_resid (nz, n_probe))``.  The matvec count of a column is
    ``arnoldi_dim + sum_j (polish_iters[j] + 2)``: the shared space, then per
    shift the polish core's own ``r_0`` evaluation, its iterations, and the
    closing true-residual check.  The correction equation's OPENING residual is
    free — :func:`shifted_solve_core` returns the residual vectors it built for
    the gate.

    ``shared_resid`` is what the shared space alone achieved, so the table can
    say WHICH shifts it finished rather than only what the pair cost.
    """
    z_vec = jnp.asarray(np.asarray(z_list, dtype=np.complex128))
    nz = int(z_vec.shape[0])
    if arnoldi_dim % check_every != 0:
        raise ValueError(
            f"arnoldi_dim={arnoldi_dim} must be a multiple of "
            f"check_every={check_every}.")
    if rhs is None:
        rhs = seed_probe_block(G_zeta, data, gen, sh)
    solver = _get_hybrid_block_solver(
        matvec, sh, nz, int(arnoldi_dim), int(check_every), int(max_iter),
        float(tol), rhs.dtype)
    s_all, resids, iters, sh_true = solver(
        rhs, diag_h, z_vec, matvec_operands(data))
    tiles = [snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
             for s in s_all]
    return tiles, resids.T, iters.T, sh_true.T
