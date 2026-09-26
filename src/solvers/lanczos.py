"""
solvers/lanczos.py — Lanczos iterative eigensolvers.

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
a callable matvec.  No physics knowledge — works for any Hermitian
eigenproblem.

Three variants:
  - simple_lanczos_eig:              Python-loop, full reorthogonalization
  - block_lanczos_eig_jit:           Block Lanczos in lax.fori_loop (any
                                     block size; block_size=1 is single-vector)
  - block_lanczos_eig_jit_converged: as above, Ritz-stability exit

The two JIT-able variants reorthogonalise by **batched classical
Gram-Schmidt, applied twice** (``cgs2``) — every overlap of a sweep in one
matrix product, so two collectives per iteration instead of ``j+1``.  It is
the only route; the basis window ``--n-reorth`` selects is independent of it —
see the route section below ``_block_alpha_stats``.

Every one of them carries the **α-Hermiticity invariant** — see the section
below ``import numpy as np``.  ``⟨q, Hq⟩`` is real for a Hermitian H, so the
imaginary part of ``α`` (which the recurrence computes and used to discard) is
a free detector for "the matvec did not return H·q".  It is always on because
it is free; it is the only invariant in this codebase that sits on COLLECTIVE
OUTPUT rather than on a construction tile.

Usage
-----
    from solvers.lanczos import block_lanczos_eig_jit
    eigenvalues, eigenvectors = block_lanczos_eig_jit(
        matvec_block, n=1000, n_eig=10, block_size=1)
"""
from __future__ import annotations

import contextlib
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


# ===========================================================================
# The Hermitian-form invariant the recurrence already computes
# ===========================================================================
#
# WHY THIS IS HERE
# ----------------
# Every Lanczos variant below forms
#
#     α_j = ⟨q_j, H q_j⟩            (scalar variants)
#     α_j = Q_jᴴ H Q_j              (block variants, (bs, bs))
#
# and then throws away the part of it that is a free integrity check.  The
# scalar variants took ``.real`` and discarded ``Im α``; the block variants
# built T and symmetrised it with ``(T + Tᴴ)/2`` (see ``_build_block_tridiag``),
# which discards ``α − αᴴ``.  For a Hermitian H both discarded quantities are
# ZERO in exact arithmetic — a Hermitian form has a real value, a Hermitian
# Gram block is Hermitian — for ANY q, converged or not, at every iteration.
#
# That makes ``Im α`` a detector for "the matvec did not return H·q".  It is
# blind to nothing that matters and it costs nothing, because the complex dot
# product that produces it is already on the critical path.
#
# The concrete motivation (2026-07-29): ``jax.lax.psum_scatter`` under
# ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`` silently returns wrong data in
# ~5 % of executions, always in output segment 0, with a plausible magnitude
# and a zero exit code (``wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md``).
# The BSE matvec issues two of them per iteration
# (``bse/bse_stack_matvec.py:126,129``).  A corrupted segment makes the
# returned vector not equal to H·q, which breaks the Hermitian form by the
# size of the error.  The archaeology found that ``check_hermitian`` runs at
# five sites in this codebase and NONE of them is downstream of a
# reduce-scatter — every invariant sat on construction tiles.  This is the one
# that sits on collective OUTPUT, and it is the reason a whole campaign of
# 1913 job logs contains no detection of a bug that was firing all along.
#
# THE QUANTITY, AND WHY IT IS check_hermitian's
# ---------------------------------------------
# The tridiagonal T built from these α's is Hermitian by construction, so
# ``(T − Tᴴ)_jj = α_j − conj(α_j) = 2i·Im α_j``.  Reporting
#
#     rel = max_j |Im α_j| / max_j |α_j|
#
# is therefore literally ``max|A − Aᴴ| / max|A|`` (up to the factor 2)
# restricted to T's diagonal — the SAME residual, scaled against the tile's own
# scale, that ``common.sanity.check_hermitian`` computes.  Both paths report
# through ``sanity.report_hermitian_residual`` so there is one verdict, one
# tolerance and one message, not two.  Normalising by ``max_j |α_j|`` rather
# than by the per-iteration ``|α_j|`` is deliberate and is check_hermitian's own
# convention: a single α passing near zero (a Krylov direction nearly orthogonal
# to Hq) must not manufacture a false positive.
#
# THE TOLERANCE — DERIVED, NOT TUNED
# ----------------------------------
# With ``‖q‖₂ = 1`` and u = 2⁻⁵³ = 1.11e-16 the unit roundoff of float64, the
# computed ``Im α`` has exactly two sources, and both scale with a CONTRACTION
# LENGTH times u:
#
#   (1) the dot product itself.  α = Σᵢ conj(qᵢ) zᵢ over n terms; the standard
#       bound is |fl(Σ) − Σ| ≤ γ_n Σ|qᵢ zᵢ| ≤ γ_n‖q‖‖z‖ = γ_n‖z‖ with
#       γ_n = nu/(1−nu) ≈ n·u (pairwise summation, which XLA/BLAS actually use,
#       reduces this to O(log n · u) — we keep the pessimistic n·u).
#
#   (2) the matvec.  z = fl(Hq) = Hq + δz with ‖δz‖ ≲ c_H·u·‖H‖, where c_H is
#       the effective accumulation depth of the matvec.  ⟨q, δz⟩ has no reason
#       to be real, so it lands in Im α at ≲ c_H·u·‖H‖.  For the BSE matvec the
#       deep chains are the two reduce-scatter contractions over μ and ν
#       (length N_mu each) plus the k-FFT (nk log nk) and the c/v einsums, so
#       c_H ≈ 2·N_mu + nk·log₂nk + n_c + n_v = O(N_mu).
#
# Adding them and dividing by scale = max_j|α_j| = θ‖H‖ (θ = O(1) once the
# Krylov space has sampled the spectrum; θ ≥ 0.1 is generous for BSE, whose H
# is gapped and positive):
#
#     rel ≲ (n + c_H)·u / θ
#
# At the LARGEST production BSE shape on this stack (N_mu = 10015, n = n_c·n_v·nk
# ≈ 4·10³):  (4·10³ + 2·10⁴)·1.11e-16 / 0.1 ≈ 2.7e-11.
#
# ``ALPHA_HERM_RTOL = 1e-9`` therefore sits ~40× above the worst-case round-off
# budget of the largest shape we run (and ~10⁶× above what a small deck
# actually measures), while the corruption it exists to catch is a RELATIVE
# perturbation of order 1e-2…1e-1 of the matvec output — 7 to 8 orders of
# magnitude above the threshold.  There is no tuning freedom in that gap.  It
# is not chosen to make a test pass; it is the round-off bound rounded up.
#
# CORRECTION (2026-08-08).  This paragraph used to end "any tolerance in
# [1e-11, 1e-4] gives the same verdict on every case we have".  A case now
# exists that the window does NOT classify uniformly: si_bse_debug, on the
# Miller-(0,0,0) head-slot rule, measures rel ~1.2e-06 -- squarely inside it.
# The claim was too strong and is retired rather than re-asserted.  The
# CONSTANT is untouched and stays untouched: with the head-slot defect fixed
# (see ``_ALPHA_CAUSE``) the same deck measures 3.165e-14 against the same
# 1e-9, i.e. 4.5 orders of headroom, so the tolerance has now been validated
# end to end rather than merely derived.
#
# COST — why this is always-on and not behind LORRAX_SANITY
# ---------------------------------------------------------
# Scalar variants: ``jnp.vdot`` already produces a complex scalar, so ``.imag``
# is the half of it that was being discarded — at worst two extra length-n
# multiply-accumulate passes, against a matvec that is orders of magnitude more
# expensive, and against the reorthogonalisation's own n_reorth dot products
# in the same loop body.  Block variants: α_j is already fully materialised
# (the recurrence subtracts ``Q_j @ α_j``), so the residual is bs² = O(10) flops
# on a tile that is already in registers.  Neither adds a collective, a device
# sync, or a full-tile pass.  Per the owner's rule, a free invariant is
# always-on: it reports through ``report_hermitian_residual(..., always=True)``,
# which bypasses the ``LORRAX_SANITY`` *cost* escape hatch while still honouring
# ``strict``.
#
# HOW THE RESIDUAL LEAVES THE TRACED REGION — and why there are two ways
# ----------------------------------------------------------------------
# The default is ONE ``jax.debug.callback`` per solve (unordered, three
# float64 scalars), which is right for the eager and small-jit callers.
#
# It is WRONG for a solve that sits inside a big, expensive jit, and this file
# used to say the callback was the only option ("it cannot be a return value:
# these solvers run inside ``bse_lanczos._full_run``'s outer jit with fixed
# ``out_shardings``").  That reasoning cost the BSE driver a 2.1 s XLA compile
# on EVERY warm run.  ``jax/_src/compiler.py::_cache_write`` refuses outright::
#
#     if host_callbacks:
#       logger.log(log_priority,
#                  "Not writing persistent cache entry for '%s' because it "
#                  "uses host callbacks (e.g. from jax.debug.print or "
#                  "breakpoint)", module_name)
#       return
#
# — a host callback is baked into the HLO module and cannot be rebound in a
# later process, so JAX will not persist ANY module that carries one.  One
# unordered three-scalar callback therefore made ``jit__full_run``, the single
# program holding the whole 200-iteration Krylov loop, permanently
# uncacheable: MEASURED on the Si 4x4x4 P=4 reference deck as
# ``cache_probes=37 hits=36 vetoed=1`` on every warm run, 2.1 s of the 17.9 s
# wall (perf/bse-warm-cache-2026-08-08).
#
# So the second way exists: :func:`alpha_herm_sink` collects the same three
# scalars as TRACED VALUES instead of emitting them, the enclosing jit returns
# them alongside its real outputs, and :func:`report_alpha_herm` runs the
# identical host-side check after the call.  Same numbers, same message, same
# ``strict`` behaviour — and the module is cacheable.  The "you cannot raise
# from inside a ``fori_loop``" half of the old note is still true and is
# exactly why the check is a post-loop reduction either way.
#
# The invariant is not weakened by the choice: a caller that opens the sink
# MUST replay it (that is what the two helpers are for), and a caller that
# does not open one still gets the callback.

ALPHA_HERM_RTOL = 1e-9

_ALPHA_FORMS = {
    "vec": ("<q,Hq>",
            "alpha_j = <q_j, H q_j> is REAL for any Hermitian H and any q_j, "
            "so a nonzero imaginary part"),
    "block": ("Q^H H Q",
              "alpha_j = Q_j^H H Q_j is HERMITIAN for any Hermitian H and any "
              "Q_j, so a nonzero antihermitian part"),
}

# WHY THIS TEXT WAS REWRITTEN (2026-08-08, alongside the head-slot fix on
# this branch).  It used to name the gloo ``psum_scatter`` corruption as
# "the known cause on this stack" and instruct the reader to re-run under
# ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` before believing any
# eigenvalue.  That text predates the diagnosis and is falsified three ways
# on the only deck that has ever fired this gate: the residual is
# bit-identical across fourteen P=4 runs (the corruption is
# non-deterministic, ~5% of executions), it is present at P=1 where there is
# no collective at all (9.365e-07), and it reproduced bit-identically on a
# single-process login-node dense probe with no reduce-scatter anywhere.
# Sending every reader to the collective is a large part of why this line
# went unread on a gate that was telling the truth.
#
# The full evidence -- the stage-by-stage reciprocity walk, the dense
# control that pins the whole non-Hermiticity on the W term, the tolerance
# derivation above, and the one-commit A/B that takes 1.155e-06 to
# 3.165e-14 -- is written up in (agent workspace, not in-tree):
#     ~/lorrax_bse_perf_2026-08-08/HERMITICITY_INVESTIGATION.md
# with the A/B logs at
#     /pscratch/sd/j/jackm/vcoul_head_0808/_reports/bse4p_{control,fixed}_400.log
# and the two frozen references the fix moves catalogued in
#     ~/lorrax_service_phase/NOTE_vcoul_head_refreeze.md
_ALPHA_CAUSE = (
    " means the matvec did not return H*q -- the operator, not the "
    "algorithm, is wrong.  WHAT WAS MEASURED: dev = max_j|Im alpha_j| "
    "(block forms: max_j max|alpha_j - alpha_j^H|) over the Krylov vectors "
    "this solve actually visited, divided by scale = max_j|alpha_j|; both "
    "are printed above, so the ratio is already normalised by the "
    "operator's own magnitude and cannot be explained away by the dynamic "
    "range of the tiles.  alpha_j is real (Hermitian) for ANY Hermitian H "
    "and ANY q_j, so the recurrence, the reorthogonalisation and the "
    "iteration count cannot manufacture this number.  KNOWN CAUSE on this "
    "stack, and the first thing to check: the mini-BZ Coulomb head used to "
    "be injected at the slot labelled Miller (0,0,0), and that label is "
    "NOT equivariant under q -> -q -- the BGW wrap sends a zone-boundary "
    "component to +1/2 rather than -1/2, so slot (0,0,0) at +q pairs with "
    "a slot carrying the BARE value at -q.  V_q therefore carried a "
    "q <-> -q reciprocity break, W inherited it, and the matvec transports "
    "it faithfully into alpha.  Measured on si_bse_debug: rel 1.155e-06 "
    "under the label rule, 3.165e-14 once the head is injected at "
    "argmin|q+G|^2 over the tied set.  If this tree's vcoul.v_qG_table "
    "still takes a per-q ``v_head_miniBZ`` table rather than "
    "``v_head_fn`` + ``head_tie_rtol``, it predates that fix and this is "
    "what you are looking at.  Other candidates, in order: any other "
    "non-Hermitian W/V tile fed to the matvec; a mis-transposed shard; "
    "and -- ONLY if the residual does not reproduce bit-for-bit at fixed "
    "process count -- the gloo reduce-scatter corruption "
    "(jax.lax.psum_scatter under JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo "
    "returns wrong data in ~5% of executions, always output segment 0; "
    "see wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md), which re-runs "
    "clean under ...=mpi.  DETERMINISM IS THE DISCRIMINATOR: an operator "
    "defect is bit-stable at fixed configuration and survives P=1; the "
    "corruption is neither."
)


def _report_alpha_herm(name: str, form: str, dev, scale, worst) -> bool:
    """Host-side half of the α-Hermiticity gate.

    Reached either from the in-jit ``jax.debug.callback`` (the default) or
    from :func:`report_alpha_herm` after an enclosing jit returned the same
    three scalars — see the header block.

    ``dev = max_j|Im α_j|`` (or ``max_j max|α_j − α_jᴴ|`` for the block
    variants), ``scale = max_j|α_j|``, ``worst`` = the iteration index where
    ``dev`` was attained, which is what a human needs to know next.
    """
    from common import sanity

    dev = float(np.asarray(dev))
    scale = float(np.asarray(scale))
    worst = int(np.asarray(worst))
    rel = dev / scale if scale > 0.0 else dev
    label, why = _ALPHA_FORMS[form]
    ok = sanity.report_hermitian_residual(
        f"{name} alpha (Hermitian form {label})", dev, scale,
        rtol=ALPHA_HERM_RTOL, always=True,
        detail=f"Worst iteration: j={worst}.  {why}{_ALPHA_CAUSE}",
    )
    if ok:
        try:
            first = jax.process_index() == 0
        except Exception:
            first = True
        if first:
            print(f"  lanczos[{name}]: alpha non-Hermitian part / "
                  f"max|alpha| = {rel:.3e} "
                  f"(tol {ALPHA_HERM_RTOL:.0e}, worst j={worst})  OK",
                  flush=True)
    return ok


# The open sink, or None.  Module-level rather than threaded through every
# solver signature because the solvers are TRACED inside the caller's jit —
# the sink is a property of the trace, not of the call.  Not thread-local:
# JAX tracing is single-threaded per process here, and a thread-local would
# silently give a second thread the callback path (i.e. an uncacheable module)
# with no way to notice.
_ALPHA_SINK: list | None = None


@contextlib.contextmanager
def alpha_herm_sink():
    """Collect the α-Hermiticity reports instead of emitting them in-jit.

    Wrap the TRACE of a jit whose module must stay persistable::

        with alpha_herm_sink() as sink:
            evs, evecs = block_lanczos_eig_jit(matvec, n, ...)
        labels, payload = split_alpha_sink(sink)
        return evs, evecs, payload          # payload joins the jit's outputs

    and then, on the host, after the jit has run::

        report_alpha_herm(labels, payload)

    ``labels`` is the static half (solver name + α form) and is known at trace
    time; ``payload`` is the traced half (three scalars per solve) and must
    travel out as a jit output.  Splitting them is what lets the static half
    be captured once at trace time while the numbers come back per call.

    Yields the raw sink list.  Nests and restores, so a caller inside another
    caller's sink does not steal its reports.
    """
    global _ALPHA_SINK
    prev = _ALPHA_SINK
    _ALPHA_SINK = sink = []
    try:
        yield sink
    finally:
        _ALPHA_SINK = prev


def split_alpha_sink(sink):
    """Split a sink into its static ``labels`` and traced ``payload`` halves."""
    labels = tuple((name, form) for name, form, _d, _s, _w in sink)
    payload = tuple((dev, scale, worst) for _n, _f, dev, scale, worst in sink)
    return labels, payload


def report_alpha_herm(labels, payload) -> bool:
    """Run the α-Hermiticity gate on scalars a jit already returned.

    The host-side counterpart of :func:`alpha_herm_sink`; identical check,
    identical message, identical ``strict`` behaviour to the in-jit callback
    path.  Returns True only if every collected solve passed.
    """
    ok = True
    for (name, form), (dev, scale, worst) in zip(labels, payload):
        ok = _report_alpha_herm(name, form, dev, scale, worst) and ok
    return ok


def _emit_alpha_herm(name: str, alpha_im, alpha_re,
                     form: str = "vec") -> None:
    """Reduce the per-iteration α residual to 3 scalars and ship them out.

    ``alpha_im`` : (n_iter,) real — |Im α_j| (scalar variants) or
    ``max|α_j − α_jᴴ|`` (block variants).  ``alpha_re`` : (n_iter,) real —
    |α_j| (or ``max|α_j|``).  Traced-safe: pure reductions plus one unordered
    ``jax.debug.callback``, no collectives, no device sync.

    With a sink open (see the header block) the three scalars are handed to
    the sink instead and NO callback is traced — which is what keeps the
    enclosing module persistable in JAX's compilation cache.
    """
    dev = jnp.max(alpha_im)
    scale = jnp.max(alpha_re)
    worst = jnp.argmax(alpha_im)
    if _ALPHA_SINK is not None:
        _ALPHA_SINK.append((name, form, dev, scale, worst))
        return
    jax.debug.callback(
        lambda d, s, w: _report_alpha_herm(name, form, d, s, w),
        dev, scale, worst)


def _block_alpha_stats(alpha_all):
    """Per-block ``(max|α − αᴴ|, max|α|)`` for a stack of (bs, bs) α blocks.

    Exactly ``check_hermitian``'s two numbers, evaluated per iteration on a
    tile that is already resident.  O(n_iter · bs²) — free.
    """
    herm = jnp.conj(jnp.swapaxes(alpha_all, -1, -2))
    dev = jnp.max(jnp.abs(alpha_all - herm), axis=(-2, -1))
    scale = jnp.max(jnp.abs(alpha_all), axis=(-2, -1))
    return dev, scale


# ===========================================================================
# Reorthogonalisation WINDOW: why the sentinel exists, and what it resolves to
# ===========================================================================
#
# The window (``n_reorth``) and the route (``cgs2``, next section) are
# independent axes and are easy to confuse.  The ROUTE decides how the overlaps
# are computed — two collectives per iteration.  The WINDOW decides WHICH basis
# vectors are in the set at all.
#
# MEASURED (stored dense Si BSE matrix, n=1024, single-vector Lanczos; error on
# the lowest 20 eigenvalues against the dense spectrum):
#
#     window     k=200    k=400    k=600    k=800    k=1000
#     2          4.3 meV  86       94       134      142
#     10         4.3      86       94       134      142
#     30         4.3      86       94       134      142
#     full       4.3      1.2      0.037    8.1e-4   4e-4
#
# Windows of 2, 10 and 30 are INDISTINGUISHABLE: the basis has already lost
# orthogonality by the time the window falls off the end, so widening it
# changes nothing.  All of them plateau at 4.3 meV and then get monotonically
# WORSE with more iterations — at k=1000, ``||Q^H Q - I|| = 0.35`` and there are
# 91 Ritz values below lambda_20 where there should be exactly 20.  That is the
# ghost-eigenvalue mechanism: a lost direction is re-discovered as a duplicate
# copy of an already-converged root.  At n=4096 the best a windowed run
# achieves is 21.3 meV.
#
# "More iterations makes it worse" is what makes a partial window unsafe as a
# DEFAULT: a caller who asks for more work and gets a worse answer has no way
# to notice from the outside.  The window is kept as an OPTION — it is a
# legitimate memory/time trade for a caller who has measured their own spectrum
# — it just must not be what you get by not choosing.
#
# Since 2026-08-08 the cost argument for a window is gone as well: under the
# default ``cgs2`` route full reorth costs TWO collectives per iteration
# regardless of window width, so ``n_reorth`` no longer buys wall time at all
# (``reorth_collective_count("cgs2", 200, 5) == reorth_collective_count("cgs2",
# 200, 200) == 400``).  It only buys arithmetic and memory traffic, which on
# every shape measured here are far inside the latency floor.
#
# THE SENTINEL, AND THE TRAP IT SETS
# ----------------------------------
# ``FULL_REORTH`` (-1) means "the whole basis", the same convention
# ``bse/bse_jax.py --n-reorth`` and ``bse/exciton_bands.py`` already used.  It
# is a SENTINEL, not a width, and two consumers below read ``n_reorth`` as a
# width without checking:
#
#   * ``_reorth_window`` builds ``idx >= j - n_reorth``.  Fed -1 that becomes
#     ``idx >= j + 1``, which intersected with ``idx <= j`` is EMPTY — full
#     reorth would silently become NO reorth.
#   * ``_announce_reorth`` prints ``n_reorth`` as the window a log reader
#     checks, so an unresolved -1 would lie in that log.
#
# Neither raises.  So ``resolve_n_reorth`` MUST run before both, on every path,
# and ``test_sentinel_is_resolved_before_both_consumers`` is the red twin that
# feeds -1 through and checks the announced window and the mask width against
# the resolved value.

#: Sentinel: reorthogonalise against the ENTIRE basis built so far.
#: Same convention as ``bse/bse_jax.py --n-reorth`` and
#: ``bse/exciton_bands.py``, so one value means one thing everywhere.
FULL_REORTH = -1


def resolve_n_reorth(n_reorth: int | None, depth: int) -> int:
    """Resolve the reorth window against the basis depth it will run to.

    ``FULL_REORTH`` (-1) and ``None`` both mean "the whole basis", expressed as
    ``depth`` — the number of iterations (single-vector) or blocks (block
    variants) the loop can reach.  Any other value passes through as a window
    width.  Centralised so the sentinel cannot mean -1 iterations in one
    variant and full reorth in another.

    Idempotent on already-resolved values, which is what lets
    ``bse/bse_jax.py`` keep its own pre-resolution without double-counting.

    MUST be called before ``_reorth_window`` and ``_announce_reorth``, and
    AFTER any Krylov-exhaustion clamp on ``max_iter`` — the depth it resolves
    against is the CLAMPED depth, not the requested one.
    """
    if n_reorth is None or int(n_reorth) < 0:
        return int(depth)
    return int(n_reorth)


# ===========================================================================
# Reorthogonalisation route — batched classical Gram-Schmidt, twice (cgs2)
# ===========================================================================
#
# Every solver below reorthogonalises the new Krylov direction against the
# stored basis.  Classical Gram-Schmidt applied TWICE ("twice is enough" —
# Giraud, Langou & Rozloznik, Computing 74:85, 2005) computes ALL overlaps of
# a pass as one matrix-vector product and applies them as one more:
#
#     h = Q^H z          # ONE all-reduce, of an (m,) vector
#     z = z - Q h        # no collective: h is replicated, Q's rows are sharded
#
# Two passes per Lanczos iteration => **2 collectives per iteration**.  A
# single classical pass loses orthogonality like ``O(u * kappa(Q))``; two give
# ``O(u)`` unconditionally, with no test, no branch and no extra collective.
#
# KEY INSIGHT: the cost of reorthogonalisation on a mesh is the collective
# COUNT, not the flops.  The per-vector modified Gram-Schmidt sweep this
# replaced (2026-08-08) issued one 16-byte all-reduce per basis vector,
# ``max_iter(max_iter+1)/2`` = 20 100 of them at 200 iterations, each ~18 us of
# latency around < 1 ns of arithmetic.  Measured on the Si 4x4x4 record deck
# at P=4 (200 iterations, full reorth): reorth all-reduces 20 100 -> 400,
# bse.eigensolve 4.209 s -> 3.431 s, max |dlambda| over 20 excitons
# 9.77e-15 eV.  CGS2 does 4x the arithmetic and is ~15x faster.
#
# DGKS (repeat only when ``||z||`` drops) is not used: its test is itself a
# norm on the sharded axis, so it costs the collective it would save and makes
# the loop body data-dependent inside a jit.
#
# The route reads no environment: ``solvers`` is L2 and must be a function of
# its arguments (``tests/test_layering.py::test_no_l2_module_reads_the_
# environment``).  ``LORRAX_LANCZOS_REORTH`` is retired and refuses by name in
# ``bse.bse_lanczos``.


def reorth_collective_count(max_iter: int) -> int:
    """Reorth-attributable all-reduces a solve issues: two per iteration."""
    return 2 * int(max_iter)


def _announce_reorth(name: str, max_iter: int, n_reorth: int) -> None:
    """One trace-time line, so a log PROVES the window its numbers used.

    Emitted at trace time (not inside the loop): a jitted solve prints it once
    per compile.
    """
    try:
        first = jax.process_index() == 0
    except Exception:
        first = True
    if first:
        n_coll = reorth_collective_count(max_iter)
        print(f"  lanczos[{name}]: reorth route=cgs2 n_reorth={n_reorth} "
              f"max_iter={max_iter} -> {n_coll} reorth all-reduces",
              flush=True)


# ---------------------------------------------------------------------------
# THE WINDOW INCLUDES THE CURRENT BLOCK Q_j
# ---------------------------------------------------------------------------
# The reorth window is ``{i : max(0, j - n_reorth) <= i <= j}``: the current
# block is projected out again after the three-term step.  The block
# recurrence subtracts the full complex ``alpha_j = Q_j^H Z``, so in exact
# arithmetic that projection is a no-op; it removes the round-off component
# left along ``Q_j``.  It costs nothing: ``cgs2`` computes all overlaps in one
# all-reduce and masks, so one more selected slot is free.
#
# History: the retired single-vector kernel subtracted only ``Re alpha`` and
# left ``i*Im alpha`` in ``z``; with an ``i < j`` window that put a
# ``4.2009e-06`` floor under the Ritz-vector orthogonality of the Si record
# deck, and widening the window to ``i <= j`` removed it (2026-08-08,
# ``RITZ_ORTHO_PROBE.md``).
#
# ``_REORTH_INCLUDE_CURRENT`` exists so
# ``tests/test_lanczos_reorth_routes.py`` can drive the narrow window
# in-process.  Production never changes it.
_REORTH_INCLUDE_CURRENT = True


def _reorth_window(j, n_slots: int, n_reorth: int):
    """Boolean ``(n_slots,)`` selector for the slots the reorth sweep visits.

    ``{i : max(0, j - n_reorth) <= i <= j}`` — the ``n_reorth`` previous basis
    vectors AND the current one, ``q_j``.  ``j`` is traced; ``n_slots`` and
    ``n_reorth`` are static.  Slots past ``j`` need no mask: the pre-allocated
    basis is exactly zero there.  Slot ``j`` is the one the pre-2026-08-08
    window dropped — see the block above for why it must be in.

    This is the SINGLE definition of the window; ``cgs2`` applies it as a mask
    on ``h``.
    """
    idx = jnp.arange(int(n_slots))
    upper = idx <= j if _REORTH_INCLUDE_CURRENT else idx < j
    return jnp.logical_and(upper, idx >= j - int(n_reorth))


def _cgs2_block(Q_all, Z, sel):
    """Two classical Gram-Schmidt passes against a block basis.

    ``Q_all`` ``(m, n, bs)`` — stored basis blocks; slots past the current
    iteration are exactly zero.  ``Z`` ``(n, bs)``.  ``sel`` ``(m,)`` bool.

    TWO collectives total, one per pass, each an ``(m, bs, bs)`` all-reduce —
    the whole triangular sweep of ``(bs, bs)`` Gram blocks in one shot.
    """
    def _pass(ZZ):
        H = jnp.einsum("inb,nc->ibc", jnp.conj(Q_all), ZZ)
        H = jnp.where(sel[:, None, None], H, jnp.zeros((), dtype=H.dtype))
        return ZZ - jnp.einsum("inb,ibc->nc", Q_all, H)
    return _pass(_pass(Z))


def _resolve_subspace_plan(plan, capacity, n_eig, *, vector_sharding=None,
                           max_block_size=None):
    """Resolve substrate policy before the recurrence is staged.

    Distributed callers supply their plan before their outer JIT. ``False``
    selects the fixed-shape reference arithmetic for numerical A/B gates.
    """
    if plan is False:
        return None
    if plan is None:
        from distrib_la import plan_subspace
        plan = plan_subspace(capacity=capacity, n_eig=n_eig,
                             vector_sharding=vector_sharding,
                             max_block_size=max_block_size)
    if plan.capacity != capacity or plan.n_eig != n_eig:
        raise ValueError(
            f"Lanczos subspace plan requires capacity={capacity}, n_eig={n_eig}")
    return plan


def _planned_ritz(plan, basis, projected, active, n_eig, structured_vectors=False):
    """Solve/reconstruct only the completed Krylov interval; no basis slice."""
    values, coefficients = plan.eigh(projected, active)
    template = jnp.zeros((n_eig, *basis.shape[1:]), basis.dtype)
    vectors, _ = plan.reconstruct(
        basis, basis, coefficients, active, template, compute_image=False)
    if not structured_vectors:
        vectors = vectors.reshape(n_eig, -1)
    norms = jnp.sqrt(jnp.sum(jnp.abs(vectors)**2, axis=tuple(range(1, vectors.ndim)), keepdims=True))
    return values, vectors / jnp.maximum(norms, 1e-15)


def _block_lanczos_step(j, Q_all, alpha_all, beta_all, *, matvec,
                        n_slots: int, n_reorth: int, bs: int,
                        subspace_plan=None, structured_vectors=False):
    """One block-Lanczos iteration, shared by both jitted block variants.

    Reads ``Q_all[j]``, writes ``alpha_all[j]``, ``beta_all[j]`` and
    ``Q_all[j + 1]``.  ``n_slots`` is the basis buffer's leading extent
    (``max_iter + 1``); ``n_slots``, ``n_reorth`` and ``bs`` are static,
    ``j`` is traced.

    ONE copy, because the fixed-iteration and convergence-driven variants ran
    byte-identical bodies, and a reorth fix applied to one of them would
    silently have missed the other.
    """
    row_basis = subspace_plan is not None
    if row_basis:
        Q_j = Q_all[j]
        Z = matvec(Q_j if structured_vectors else Q_j.reshape(bs, -1)).reshape(Q_j.shape)
        axes = tuple(range(1, Q_j.ndim))
        overlap = lambda q, z: jnp.tensordot(jnp.conj(q), z, axes=(axes, axes))
        combine = lambda q, c: jnp.tensordot(c.T, q, axes=(1, 0))
        alpha_j = overlap(Q_j, Z)
        Z = Z - combine(Q_j, alpha_j)
        prev = jnp.maximum(j - 1, 0)
        Z = jnp.where(j > 0, Z - combine(Q_all[prev], jnp.conj(beta_all[prev]).T), Z)
        first = jnp.maximum(0, j - n_reorth)
        stop = j + int(_REORTH_INCLUDE_CURRENT)
        Z = subspace_plan.orthogonalize(
            Q_all.reshape(n_slots * bs, *Q_all.shape[2:]), Z,
            (stop - first) * bs, start=first * bs)
        Q_next, beta_j = subspace_plan.qr(Z)
        return Q_all.at[j + 1].set(Q_next), alpha_all.at[j].set(alpha_j), beta_all.at[j].set(beta_j)
    Q_j = Q_all[j]  # (n, bs)
    # Block matvec over (bs, n) → (bs, n); transpose to (n, bs).
    Z = matvec(Q_j.T).T

    alpha_j = jnp.conj(Q_j).T @ Z                      # (bs, bs)
    alpha_all = alpha_all.at[j].set(alpha_j)
    Z = Z - Q_j @ alpha_j

    # Subtract Q_{j-1} · β_{j-1}^H (skip on j=0).
    Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
    beta_prev = beta_all[jnp.maximum(j - 1, 0)]
    Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

    Z = _cgs2_block(Q_all, Z, _reorth_window(j, n_slots, n_reorth))

    # QR(Z) → next block + β_j.  Write to slot j+1 (always valid with the
    # +1 buffer, 1..max_iter) — no clobber of the current Q_j.
    Q_next, beta_j = jnp.linalg.qr(Z)                  # (n, bs), (bs, bs)
    return (Q_all.at[j + 1].set(Q_next),
            alpha_all,
            beta_all.at[j].set(beta_j))


def _mask_inactive_tail(T, n_active: int):
    """Push the never-written tail of ``T`` out of the low spectrum.

    Slots past ``n_active`` are exactly zero, so ``sort()[:n_eig]`` would
    return that zero as a Ritz value below the true spectrum.  Adding a large
    constant to their diagonal moves them out of the way instead.  ``T`` is
    the block-tridiagonal, ``n_active`` the number of ROWS actually filled.
    """
    LARGE = jnp.asarray(1.0e6, dtype=T.real.dtype)
    mask = (jnp.arange(T.shape[0]) >= n_active).astype(T.real.dtype) * LARGE
    return T + jnp.diag(mask).astype(T.dtype)


def simple_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Simple Lanczos with full reorthogonalization (Python loop).

    Parameters
    ----------
    matvec : (n,) -> (n,)
        Hermitian matvec on flat vectors.
    n : int
        Vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    max_iter : int
        Maximum Lanczos iterations.
    seed : int
        Random seed for initial vector.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)

    Krylov-exhaustion clamp: the Krylov space cannot exceed the vector space,
    so ``max_iter`` is clamped to ``n``.  Running past exhaustion is not
    benign — the residual collapses, the normalisation divides by ~0, and the
    manufactured alpha/beta put Ritz values ANYWHERE, including BELOW the true
    spectrum.
    """
    max_iter = max(1, min(int(max_iter), int(n)))
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q = q + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q = q / jnp.linalg.norm(q)

    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)
    # |Im α_j| — the half of the Hermitian form this loop used to discard.
    alpha_im = jnp.zeros((max_iter,), dtype=jnp.float64)

    for j in range(max_iter):
        z = matvec(q)
        alpha_c = jnp.vdot(q, z)           # ONE dot; both halves are used.
        alpha = alpha.at[j].set(alpha_c.real)
        alpha_im = alpha_im.at[j].set(jnp.abs(alpha_c.imag))

        if j > 0:
            z = z - beta[j - 1] * Q[:, j - 1]
        z = z - alpha[j] * q

        for i in range(j + 1):
            proj = jnp.vdot(Q[:, i], z)
            z = z - proj * Q[:, i]

        beta = beta.at[j].set(jnp.linalg.norm(z))
        if beta[j] < 1e-12:
            max_iter = j + 1
            break

        q = z / beta[j]
        Q = Q.at[:, j + 1].set(q)

    _emit_alpha_herm("simple_lanczos_eig",
                     alpha_im[:max_iter], jnp.abs(alpha[:max_iter]))

    T = jnp.diag(alpha[:max_iter])
    if max_iter > 1:
        off = beta[:max_iter - 1]
        T = T + jnp.diag(off, 1) + jnp.diag(off, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]

    eigenvalues = evals_T[idx]
    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T

    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms

    return eigenvalues, eigenvectors


def _build_block_tridiag(alpha_all, beta_all, max_iter: int, bs: int,
                        *, capacity=None, active_blocks=None):
    """Build the block-tridiagonal T from per-iter (bs,bs) blocks.

    Done inside the jit by ``lax.fori_loop`` so the trace-time HLO stays
    O(1) instead of unrolling ``max_iter`` slot updates. Used by both
    the fixed-iter and convergence-driven block Lanczos paths.
    """
    T_size = bs * max_iter if capacity is None else capacity
    used = max_iter if active_blocks is None else active_blocks
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    def diagonal(j):
        block = alpha_all[j]
        return (block + block.conj().T) * 0.5 if capacity is not None else block

    def body(j, T):
        s = j * bs
        T = lax.dynamic_update_slice(T, diagonal(j), (s, s))
        # Off-diagonal beta only when j+1 < max_iter (zero alpha/beta past
        # the end keeps the slot a no-op even when j is at the boundary).
        T = lax.dynamic_update_slice(T, beta_all[j], (s + bs, s))
        T = lax.dynamic_update_slice(
            T, jnp.conj(beta_all[j]).T, (s, s + bs))
        return T

    T = lax.fori_loop(0, used - 1, body, T)
    # Final diagonal block (no off-diagonal past the end).
    s_last = (used - 1) * bs
    T = lax.dynamic_update_slice(T, diagonal(used - 1), (s_last, s_last))
    # Planned off-diagonals are already conjugate partners. Hermitize only
    # completed diagonal blocks instead of sweeping the full capacity.
    return T if capacity is not None else (T + jnp.conj(T).T) * 0.5


def block_lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
    subspace_plan=None,
    vector_shape=None,
    structured_vectors=False,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled block Lanczos using ``lax.fori_loop``.

    Block Lanczos with all state in pre-allocated arrays, so the body fits
    in ``lax.fori_loop`` and the
    caller's outer jit can fuse this with the matvec.  The matvec
    operates on a *block* of trial vectors

        matvec : (block_size, n) -> (block_size, n)

    so the BSE-style ring matvec processes ``block_size`` vectors per
    call.  That makes the per-call GEMMs ``block_size`` times larger
    (better arithmetic intensity / GPU occupancy) and reduces the host
    dispatch count by ``block_size`` for the same total Krylov
    dimension.

    The total Krylov dimension is ``block_size * max_iter``; pick
    ``max_iter`` so this is comparable to a single-vector Lanczos's
    ``max_iter``.

    Parameters
    ----------
    matvec : (block_size, n) -> (block_size, n)
        Hermitian matvec on a block of flat vectors.
    n : int
        Single-vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    block_size : int
        Vectors per Lanczos block.
    max_iter : int
        Block iterations (fixed for JIT). Total Krylov size = block_size·max_iter.
    seed : int
        Random seed for initial block.
    n_reorth : int
        Window size (in *blocks*) for partial reorthogonalisation: the
        ``n_reorth`` PREVIOUS basis blocks.  The current block ``Q_j`` is
        always projected out as well, at no collective cost -- see the route
        section, "THE WINDOW INCLUDES THE CURRENT BLOCK Q_j".
        Default ``FULL_REORTH`` (-1) = the whole basis.

    Krylov-exhaustion clamp: the Krylov space cannot exceed the vector
    space, so ``max_iter`` is clamped to ``floor(n / block_size)``.
    Running past exhaustion is not benign — the residual block collapses,
    QR of a ~zero block returns junk directions, and the manufactured
    α/β blocks put Ritz values ANYWHERE, including BELOW the true
    spectrum (measured on the 4v4c MoS2 exciton window, n=144 with a
    requested 320-dim Krylov: spurious states 60-100 meV under the dense
    ground state).  At the clamp the Krylov space spans (almost) the
    whole space and the extremal Ritz values are dense-quality.
    """
    bs = int(block_size)
    max_iter = max(1, min(int(max_iter), int(n) // bs))
    subspace_plan = _resolve_subspace_plan(
        subspace_plan, (max_iter + 1) * bs, n_eig,
        max_block_size=max(bs, n_eig))
    vector_shape = (int(n),) if vector_shape is None else tuple(vector_shape)
    if int(np.prod(vector_shape)) != int(n):
        raise ValueError('Lanczos vector_shape must contain n elements')
    n_reorth = resolve_n_reorth(n_reorth, int(max_iter))
    T_size = bs * int(max_iter)
    _announce_reorth("block_lanczos_eig_jit", int(max_iter), n_reorth)

    # Initial orthonormal block via QR of random complex Gaussian.
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0 = (subspace_plan.qr(Q0.T.reshape(bs, *vector_shape))[0]
          if subspace_plan is not None else jnp.linalg.qr(Q0)[0])

    # All Q-blocks: planned (max_iter + 1, bs, n), reference (M+1,n,bs).
    # The +1 slot holds the
    # final Q_next so the last iteration does NOT overwrite Q_{max_iter-1} (the
    # slot-overwrite bug, solver_program P1: it corrupted the last Krylov block
    # in the eigenvector reconstruction).  alpha/beta: (max_iter, bs, bs).
    row_basis = subspace_plan is not None
    basis_shape = ((int(max_iter) + 1, bs, *vector_shape) if row_basis
                   else (int(max_iter) + 1, n, bs))
    Q_all = jnp.zeros(basis_shape, dtype=jnp.complex128)
    Q_all = Q_all.at[0].set(Q0)
    alpha_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)

    def body(j, carry):
        return _block_lanczos_step(
            j, *carry, matvec=matvec,
            n_slots=int(max_iter) + 1, n_reorth=n_reorth, bs=bs,
            subspace_plan=subspace_plan, structured_vectors=structured_vectors)

    Q_all, alpha_all, beta_all = lax.fori_loop(
        0, int(max_iter), body, (Q_all, alpha_all, beta_all))

    # α-Hermiticity gate, BEFORE _build_block_tridiag's (T + Tᴴ)/2 absorbs it.
    _emit_alpha_herm("block_lanczos_eig_jit", *_block_alpha_stats(alpha_all),
                     form="block")

    # Block-tridiagonal T built inside-jit (no Python loop unroll).
    T = _build_block_tridiag(
        alpha_all, beta_all, int(max_iter), bs,
        capacity=subspace_plan.capacity if row_basis else None)
    if row_basis:
        return _planned_ritz(subspace_plan, Q_all.reshape(-1, *vector_shape), T,
                             max_iter * bs, n_eig, structured_vectors)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    # Q_all is (max_iter + 1, n, bs); the T basis is the first max_iter blocks.
    Q_full = jnp.transpose(Q_all[:int(max_iter)], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T          # (n_eig, n)
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors


def block_lanczos_eig_jit_converged(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    check_every: int = 4,
    min_iter: int | None = None,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
    subspace_plan=None,
    vector_shape=None,
    structured_vectors=False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convergence-driven block Lanczos via ``lax.while_loop``.

    Same algorithm as :func:`block_lanczos_eig_jit`, but the iteration
    count is decided by Ritz-eigenvalue stability rather than fixed:

      every ``check_every`` block-iters, build the partial T (with
      future-block α/β set to zero), eigh it, and compare the lowest
      ``n_eig`` Ritz values against the previous check.  Exit when

          max_i |λ_i - λ_i_prev| < rtol·max(|λ_i|, atol).

    The pre-allocated buffers fix the upper bound at ``max_iter`` block
    iterations (so ``max_iter * block_size`` total Krylov dimension);
    the ``while_loop`` carry includes the running iteration count and
    the previous Ritz values for comparison.

    ``n_reorth`` carries exactly the meaning
    :func:`block_lanczos_eig_jit` documents.

    Returns (eigenvalues, eigenvectors, n_iter_done) — the third value
    is the actual block iteration count where the loop exited (≤
    ``max_iter``).
    """
    bs = int(block_size)
    # Krylov-exhaustion clamp — same rationale as block_lanczos_eig_jit:
    # past floor(n/bs) blocks the residual collapses and QR manufactures
    # junk directions with arbitrary (even sub-spectrum) Ritz values.
    M = max(1, min(int(max_iter), int(n) // bs))
    subspace_plan = _resolve_subspace_plan(
        subspace_plan, (M + 1) * bs, n_eig,
        max_block_size=max(bs, n_eig))
    vector_shape = (int(n),) if vector_shape is None else tuple(vector_shape)
    if int(np.prod(vector_shape)) != int(n):
        raise ValueError('Lanczos vector_shape must contain n elements')
    T_size = bs * M
    n_reorth = resolve_n_reorth(n_reorth, M)
    _announce_reorth("block_lanczos_eig_jit_converged", M, n_reorth)
    if min_iter is None:
        min_iter = max(2 * check_every, max(1, n_eig // bs + 1))
    min_iter = int(min(min_iter, M))

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0 = (subspace_plan.qr(Q0.T.reshape(bs, *vector_shape))[0]
          if subspace_plan is not None else jnp.linalg.qr(Q0)[0])

    # +1 Krylov slot so the final block does not overwrite Q_{M-1} (P1).
    row_basis = subspace_plan is not None
    basis_shape = (M + 1, bs, *vector_shape) if row_basis else (M + 1, n, bs)
    Q_all = jnp.zeros(basis_shape, dtype=jnp.complex128)
    Q_all = Q_all.at[0].set(Q0)
    alpha_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    last_evals = jnp.full((n_eig,), jnp.inf, dtype=jnp.float64)
    converged = jnp.bool_(False)

    def step(j, Q_all, alpha_all, beta_all):
        return _block_lanczos_step(
            j, Q_all, alpha_all, beta_all, matvec=matvec,
            n_slots=M + 1, n_reorth=n_reorth, bs=bs,
            subspace_plan=subspace_plan, structured_vectors=structured_vectors)

    def cond(state):
        j, _, _, _, _, conv = state
        return jnp.logical_and(j < M, jnp.logical_not(conv))

    def body(state):
        j, Q_all, alpha_all, beta_all, last_evals, _ = state
        Q_all, alpha_all, beta_all = step(j, Q_all, alpha_all, beta_all)

        # Convergence check — only every ``check_every`` iters and after
        # ``min_iter`` warmup. ``jax.lax.cond`` keeps both branches
        # constant-cost (no Python-level branching).
        do_check = jnp.logical_and(
            (j + 1) >= min_iter,
            ((j + 1) % check_every) == 0,
        )

        def _check_branch(args):
            alpha_all, beta_all, last_evals, j_done = args
            if row_basis:
                T = _build_block_tridiag(
                    alpha_all, beta_all, M, bs,
                    capacity=subspace_plan.capacity, active_blocks=j_done + 1)
                ev, _ = subspace_plan.eigh(T, (j_done + 1) * bs)
            else:
                T = _mask_inactive_tail(
                    _build_block_tridiag(alpha_all, beta_all, M, bs),
                    (j_done + 1) * bs)
                ev = jnp.sort(jnp.linalg.eigvalsh(T))[:n_eig]
            scale = jnp.maximum(jnp.abs(ev), atol)
            delta = jnp.max(jnp.abs(ev - last_evals) / scale)
            new_conv = delta < rtol
            return ev, new_conv

        def _skip_branch(args):
            _, _, last_evals, _ = args
            return last_evals, jnp.bool_(False)

        new_evals, new_conv = lax.cond(
            do_check, _check_branch, _skip_branch,
            (alpha_all, beta_all, last_evals, j),
        )
        return (j + 1, Q_all, alpha_all, beta_all, new_evals, new_conv)

    init = (jnp.int32(0), Q_all, alpha_all, beta_all, last_evals, converged)
    j_final, Q_all, alpha_all, beta_all, _, _ = lax.while_loop(cond, body, init)

    # α-Hermiticity gate, BEFORE _build_block_tridiag's (T + Tᴴ)/2 absorbs it.
    # Blocks past ``j_final`` were never written and are exactly zero, so they
    # contribute 0 to both the deviation and the scale — no mask needed.
    _emit_alpha_herm("block_lanczos_eig_jit_converged",
                     *_block_alpha_stats(alpha_all), form="block")

    if row_basis:
        T = _build_block_tridiag(
            alpha_all, beta_all, M, bs,
            capacity=subspace_plan.capacity, active_blocks=j_final)
        values, vectors = _planned_ritz(
            subspace_plan, Q_all.reshape(-1, *vector_shape), T, j_final * bs, n_eig, structured_vectors)
        return values, vectors, j_final

    # Final eigh — same inactive-tail mask as the convergence check.
    T = _mask_inactive_tail(
        _build_block_tridiag(alpha_all, beta_all, M, bs), j_final * bs)
    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]
    Q_full = jnp.transpose(Q_all[:M], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors, j_final
