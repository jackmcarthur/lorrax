"""
solvers/lanczos.py — Lanczos iterative eigensolvers.

Finds the lowest n_eig eigenvalues of a Hermitian operator H given only
a callable matvec.  No physics knowledge — works for any Hermitian
eigenproblem.

Five variants:
  - simple_lanczos_eig:              Python-loop, full reorthogonalization
  - lanczos_eig_jit:                 lax.fori_loop, partial reorth (JIT-able)
  - block_lanczos_eig:               Block Lanczos, Python loop, shaped vectors
  - block_lanczos_eig_jit:           Block Lanczos in lax.fori_loop
  - block_lanczos_eig_jit_converged: as above, Ritz-stability exit

The three JIT-able variants reorthogonalise by **batched classical
Gram-Schmidt, applied twice** (``cgs2``) — every overlap of a sweep in one
matrix product, so two collectives per iteration instead of ``j+1``.
``LORRAX_LANCZOS_REORTH=mgs`` restores the legacy per-vector sweep for bisects
and for reproducing pre-2026-08-08 runs (the variable is read one layer up,
by ``bse.bse_lanczos.reorth_route`` — this module is L2 and takes the token).  The route changes the collective
COUNT, never the basis window ``--n-reorth`` selects — see the route section
below ``_block_alpha_stats``.

Every one of them carries the **α-Hermiticity invariant** — see the section
below ``import numpy as np``.  ``⟨q, Hq⟩`` is real for a Hermitian H, so the
imaginary part of ``α`` (which the recurrence computes and used to discard) is
a free detector for "the matvec did not return H·q".  It is always on because
it is free; it is the only invariant in this codebase that sits on COLLECTIVE
OUTPUT rather than on a construction tile.

Usage
-----
    from solvers.lanczos import lanczos_eig_jit
    eigenvalues, eigenvectors = lanczos_eig_jit(matvec, n=1000, n_eig=10)
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
            evs, evecs = lanczos_eig_jit(matvec, n, ...)
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
# The window (``n_reorth``) and the route (``LORRAX_LANCZOS_REORTH``, next
# section) are independent axes and are easy to confuse.  The ROUTE decides how
# the overlaps are computed — one collective per basis vector (``mgs``) or two
# per iteration (``cgs2``).  The WINDOW decides WHICH basis vectors are in the
# set at all.  A wrong route costs wall time; a wrong window costs eigenvalues.
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
#   * ``_announce_reorth`` calls ``mgs_trip_count(max_iter, n_reorth)``, which
#     at -1 sums ``j + 1 - max(0, j + 1) == 0`` and announces ZERO collectives.
#
# Neither raises.  Both are silent, and the second one lies in the log that a
# reader would use to check the first.  So ``resolve_n_reorth`` MUST run before
# both, on every path, and ``test_sentinel_is_resolved_before_both_consumers``
# is the red twin that feeds -1 through both routes and checks the announced
# count and the mask width against the resolved value.

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
# Reorthogonalisation route — LORRAX_LANCZOS_REORTH
# ===========================================================================
#
# THE DEFAULT IS ``cgs2`` — BATCHED.  ``mgs`` IS THE LEGACY FALLBACK.
# --------------------------------------------------------------------
# Every solver below reorthogonalises the new Krylov direction against the
# stored basis.  Until 2026-08-08 the only route was ``mgs``: a modified
# Gram-Schmidt sweep written as a ``lax.fori_loop`` over ONE basis vector per
# trip:
#
#     for i in [max(0, j - n_reorth), j):
#         z -= <q_i, z> q_i               # one dot + one axpy, per i
#
# On a mesh the flat Krylov axis is sharded, so ``<q_i, z>`` is a local dot
# followed by a ``psum`` — and that psum is a SEPARATE all-reduce of ONE
# complex128 scalar.  At full reorth (``n_reorth = max_iter``) the trip count is
# ``sum_{j=1..max_iter} j = max_iter(max_iter+1)/2``, i.e. **20 100 collectives
# for a 200-iteration solve**, each moving 16 bytes.
#
# THE COST WAS NEVER THE FLOPS, WHICH IS THE WHOLE POINT.  Measured on the Si
# 4x4x4 record deck at P=4 (2x2 mesh, xprof, 2026-08-08), per one of those
# 20 100 trips: the local dot is 512 complex128 elements = 8 192 flops = **0.845
# ns** of A100 fp64 arithmetic, and it is wrapped in **18.11 us** —
#
#     all-reduce of a ``c128[]`` SCALAR (16 bytes)   8.82 us   0 % occupancy
#     the local reduce kernel                        4.24 us
#     the axpy kernel                                1.86 us
#     the fori_loop's own counter + predicate        3.16 us
#
# — a factor of **21 000**, and the code asked for it 20 100 times: **364 ms of
# GPU kernel time** plus 20 369 D2H staging copies (35.7 ms, fully exposed), to
# do 165 MFLOP.  That is 0.005 % of peak.  It is pure LATENCY, not bandwidth:
# 16 B in 8.82 us is 0.0018 GB/s, against 45.5 GB/s achieved by the matvec's own
# 2.81 MiB all-gather in the SAME module — a 25 000x gap that no interconnect
# tuning can close, because the message is 16 bytes.
#
# THE DEFAULT ROUTE (``cgs2``)
# ----------------------------
# Classical Gram-Schmidt applied TWICE ("twice is enough" — Giraud, Langou &
# Rozloznik, Computing 74:85, 2005; the observation is Kahan's, via Parlett).
# Each pass computes ALL overlaps as one matrix-vector product and applies them
# as one more:
#
#     h = Q^H z          # ONE all-reduce, of an (m,) vector
#     z = z - Q h        # no collective: h is replicated, Q's rows are sharded
#
# Two passes per Lanczos iteration => **2 collectives per iteration**, i.e.
# ``2 * max_iter = 400`` instead of 20 100, and the payload grows from a
# 16-byte scalar to an ``(max_iter+1,)`` complex128 vector (3 216 B at
# max_iter=200) — still four orders of magnitude inside the latency floor of
# any interconnect, so the wall is governed by the collective COUNT alone.
#
# MEASURED A/B on the record deck, P=4, 200 iters, full reorth (five interleaved
# warm repeats per arm; the paired traces are cache-cold):
#
#     reorth all-reduces      20 100  ->    400     (50x)
#     GPU events in the run  152 300  -> 13 000     (11.7x)
#     D2H copies              20 369  ->     69     (295x)
#     bse.eigensolve          4.209 s -> 3.431 s    (-18.5 %, zero overlap
#                                                    across all ten legs)
#     of which Krylov exec      ~2.0 s -> ~1.23 s   (-39 %)
#     max |dlambda|, 20 excitons        9.77e-15 eV (owner gate 1e-9)
#     max|V^H V - I|          4.2009e-06 on BOTH routes (identical: a
#                             pre-existing property of this deck's output, NOT
#                             of the route — reorth off gives 1.2478e-01)
#
# It does **4x the arithmetic and moves 4x the bytes** and is ~15x faster.  Both
# costs were free; the call count never was.
#
# WHY CGS2 AND NOT ONE CLASSICAL PASS: a single classical pass loses
# orthogonality like ``O(u * kappa(Q))`` and is unusable in Lanczos.  Two passes
# give ``O(u)`` unconditionally — the same order MGS-with-full-reorth achieves —
# with no test, no branch, and no collective beyond the second psum.
#
# WHY NOT DGKS (repeat only when ``||z||`` drops by a factor eta): the repeat
# test is itself a norm, i.e. a reduction on the sharded axis, so the
# conditional costs the very collective it is trying to save, and it makes the
# loop body data-dependent inside a jit.
#
# WHY NOT BLOCKED / MATRIX-FORM MGS (panels of p vectors, MGS across panels):
# it reduces the collective count only to ``max_iter^2 / (2p)`` — still
# quadratic — and buys that with a panel-size dial nobody wants to tune.  CGS2
# is linear in ``max_iter`` and has no dial.
#
# THE TWO ROUTES PROJECT THE IDENTICAL SET.  The sweep runs
# ``fori_loop(max(0, j - n_reorth), j + 1)`` with an ``i <= j`` predicate inside
# the body, so the set of basis vectors actually projected out is
# ``{i : max(0, j - n_reorth) <= i <= j}``.  The batched route reproduces that
# set exactly, via a boolean mask on ``h`` built by the SAME ``_reorth_window``.
# Columns past ``j`` need no mask because the pre-allocated basis is exactly
# zero there.
#
# That window gained its last slot on 2026-08-08: it used to stop at ``i < j``,
# which left the current vector ``q_j`` unprojected and put a 4.2e-06 floor
# under the Ritz-vector orthogonality of every route.  See "THE WINDOW INCLUDES
# THE CURRENT VECTOR q_j" below ``_announce_reorth`` — including why it costs no
# collective on either route.  ``--n-reorth k`` therefore projects ``k + 1``
# slots: the same ``k`` previous vectors it always did, plus ``q_j``.
#
# WHY ``mgs`` IS KEPT AT ALL, since ``cgs2`` is better on every axis measured:
# it is the route every number in this codebase before 2026-08-08 was taken
# with.  Keeping it one env var away means a bisect, a bit-reproduction of an
# archived run, or a "did the reorth do this?" question costs one variable
# instead of a revert.  It is a FALLBACK, not a tuning knob — there is no deck
# on which it is the right choice.
#
# An UNKNOWN token REFUSES rather than falling back, for the reason
# ``bse.bse_stack_matvec.matvec_opts`` gives at length: a perf dial that can be
# misspelled into a silent no-op makes every A/B built on it void.  Note the
# direction that matters now that ``cgs2`` is the default — a misspelling must
# not silently hand you the SLOW route either.
#
# WHERE THE DIAL IS READ — and why NOT here
# -----------------------------------------
# ``solvers`` is **L2**: physics-agnostic mathematics, which
# ``tests/test_layering.py::test_no_l2_module_reads_the_environment`` requires
# to be a function of its ARGUMENTS.  An earlier draft of this route resolved
# ``LORRAX_LANCZOS_REORTH`` right here and that gate caught it —
# ``{'solvers.lanczos': ['<dynamic>']}`` — with the fix in its own message:
# "Pass the dial in."  So the split is:
#
#   * THIS module validates and defaults a route TOKEN.  No environment, no
#     import of ``os``.  A caller that passes nothing gets ``cgs2``.
#   * ``bse.bse_lanczos.reorth_route`` reads the environment variable and
#     hands the token down, exactly as ``bse.bse_stack_matvec.matvec_opts``
#     reads ``LORRAX_BSE_MATVEC_OPT`` one layer up from the kernels it steers.
#
# The dial's NAME appears below only inside a message string, which is text,
# not a read — and it is worth the words, because the exception this raises is
# most often triggered by someone who typed the variable.
_REORTH_KINDS = ("cgs2", "mgs")
_REORTH_DEFAULT = "cgs2"
_REORTH_LEGACY = "mgs"


def reorth_kind(route: str | None = None) -> str:
    """Validate + default a reorthogonalisation route token.  PURE.

    ``None`` or empty selects ``cgs2``.  An unrecognised value raises.  This
    function reads no environment — see the layering note above; the caller
    (``bse.bse_lanczos.reorth_route``) owns ``LORRAX_LANCZOS_REORTH``.
    """
    tok = str("" if route is None else route).strip().lower()
    if not tok:
        return _REORTH_DEFAULT
    if tok not in _REORTH_KINDS:
        raise ValueError(
            f"{route!r}: unknown reorthogonalisation route.  Valid values are "
            f"{list(_REORTH_KINDS)}, and unset/empty selects "
            f"{_REORTH_DEFAULT!r} (batched classical Gram-Schmidt, twice — "
            f"2*max_iter collectives).  {_REORTH_LEGACY!r} is the legacy "
            f"per-vector sweep, kept for bisects and for reproducing archived "
            f"runs, at max_iter(max_iter+1)/2 collectives.  Refusing rather "
            f"than silently running one route under the other's label.  "
            f"(Spelled via LORRAX_LANCZOS_REORTH on the BSE path.)")
    return tok


def mgs_trip_count(max_iter: int, n_reorth: int) -> int:
    """Exact number of MGS inner trips — hence reorth collectives — a run makes.

    ``sum_j (j + 1 - max(0, j - n_reorth))`` over ``j in [0, max_iter)``, which
    is the trip count of ``fori_loop(max(0, j - n_reorth), j + 1)``.  At full
    reorth this is ``max_iter (max_iter + 1) / 2`` = 20 100 for 200 iterations.
    """
    return sum(j + 1 - max(0, j - int(n_reorth)) for j in range(int(max_iter)))


def reorth_collective_count(kind: str, max_iter: int, n_reorth: int) -> int:
    """Reorth-attributable all-reduces a solve will issue, per route."""
    return 2 * int(max_iter) if kind == "cgs2" else mgs_trip_count(
        max_iter, n_reorth)


def _announce_reorth(name: str, kind: str, max_iter: int,
                     n_reorth: int) -> None:
    """One trace-time line, so a log PROVES which route produced its numbers.

    Emitted at trace time (not inside the loop): a jitted solve prints it once
    per compile.  Without it an A/B pair of logs is indistinguishable.
    """
    try:
        first = jax.process_index() == 0
    except Exception:
        first = True
    if first:
        n_coll = reorth_collective_count(kind, max_iter, n_reorth)
        print(f"  lanczos[{name}]: reorth route={kind} n_reorth={n_reorth} "
              f"max_iter={max_iter} -> {n_coll} reorth all-reduces",
              flush=True)


# ---------------------------------------------------------------------------
# THE WINDOW INCLUDES THE CURRENT VECTOR q_j — and why that is not optional
# ---------------------------------------------------------------------------
# The recurrence subtracts only the REAL part of the current projection:
#
#     alpha_c = <q_j, z>            # complex
#     alpha_j = alpha_c.real        # only this drives the recurrence
#     z      -= alpha_j * q_j       # ... so i*Im(alpha_c) is LEFT IN z
#
# ``alpha`` has to stay real — it is the diagonal of a real symmetric
# tridiagonal ``T`` — and ``Im alpha_c`` is exactly what the alpha-Hermiticity
# invariant above reports.  But leaving that component in ``z`` means
# ``<q_j, z> = i*Im(alpha_c)`` exactly, and ``q_{j+1} = z / beta_j``, so the
# Krylov basis acquires
#
#     |<q_j, q_{j+1}>|  ~=  |Im alpha_j| / beta_j
#
# on its first superdiagonal.  Until 2026-08-08 the reorth window stopped at
# ``i < j``, so ``q_j`` was THE ONE DIRECTION NO ROUTE REMOVED — which is why
# the defect was route-independent, and why swapping MGS for CGS2 changed the
# Ritz-vector orthogonality on the Si record deck by not one digit
# (``max|V^H V - I| = 4.2009e-06`` on both).
#
# Widening the window to ``i <= j`` closes it.  Measured on that deck (P=4,
# 200 iterations, full reorth; ``RITZ_ORTHO_PROBE.md``):
#
#     max|V^H V - I|   4.2009e-06  ->  ~1e-15    (machine precision)
#
# THE COST IS ZERO, on both routes, and this is the load-bearing part:
#
#   * ``mgs`` already runs ``fori_loop(start, j + 1)``, so the ``i == j`` trip
#     ALREADY FIRES — it computes a ``vdot``, hence a 16-byte all-reduce, and
#     then ``jnp.where`` multiplies the result by zero.  200 of the 20 100
#     collectives on a 200-iteration solve were that discarded trip.  Flipping
#     the predicate consumes a value already paid for: no new collective, no
#     new kernel, no change to ``mgs_trip_count``.
#   * ``cgs2`` computes ``h = Q^H z`` over ALL slots in one all-reduce and then
#     masks.  Selecting one more entry of an already-computed ``h`` costs
#     nothing at all.
#
# IN EXACT ARITHMETIC THIS CHANGES NOTHING.  For a Hermitian ``H``,
# ``Im alpha_c == 0`` and the ``i == j`` projection is a no-op — it is
# reorthogonalisation in the ordinary sense, removing a component that is zero
# on paper and nonzero in floating point.  What it costs on a NON-Hermitian
# operator is proportional to that operator's defect: on the Si record deck it
# moved the twenty excitons by 4.2e-10 eV while the mini-BZ Coulomb head bug
# was live, and by 2.1e-14 eV once that bug was fixed.
#
# ``_REORTH_INCLUDE_CURRENT`` exists ONLY so the gate that pins this can drive
# the pre-2026-08-08 window in-process and prove it goes red
# (``tests/test_lanczos_reorth_routes.py``).  Production never changes it, and
# both routes read the same flag so they cannot drift apart.
_REORTH_INCLUDE_CURRENT = True


def _reorth_window(j, n_slots: int, n_reorth: int):
    """Boolean ``(n_slots,)`` selector for the slots the reorth sweep visits.

    ``{i : max(0, j - n_reorth) <= i <= j}`` — the ``n_reorth`` previous basis
    vectors AND the current one, ``q_j``.  ``j`` is traced; ``n_slots`` and
    ``n_reorth`` are static.  Slots past ``j`` need no mask: the pre-allocated
    basis is exactly zero there.  Slot ``j`` is the one the pre-2026-08-08
    window dropped — see the block above for why it must be in.

    This is the SINGLE definition of the window.  ``cgs2`` applies it as a mask
    on ``h``; ``mgs``'s inner predicate is the scalar form of the same set, and
    ``test_routes_select_the_same_window`` pins them equal slot by slot.
    """
    idx = jnp.arange(int(n_slots))
    upper = idx <= j if _REORTH_INCLUDE_CURRENT else idx < j
    return jnp.logical_and(upper, idx >= j - int(n_reorth))


def _cgs2_vec(Q, z, sel):
    """Two classical Gram-Schmidt passes against a single-vector basis.

    ``Q`` ``(n, m)`` — stored basis; columns past the current iteration are
    exactly zero.  ``z`` ``(n,)``.  ``sel`` ``(m,)`` bool — the window mask.

    TWO collectives total, one per pass, each an ``(m,)`` all-reduce.  The
    second contraction is over the replicated column axis and emits none.
    """
    def _pass(zz):
        # Contracts the SHARDED row axis -> exactly one psum, of shape (m,).
        h = jnp.tensordot(jnp.conj(Q), zz, axes=([0], [0]))
        h = jnp.where(sel, h, jnp.zeros((), dtype=h.dtype))
        return zz - jnp.tensordot(Q, h, axes=([1], [0]))
    return _pass(_pass(z))


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


def block_lanczos_eig(
    matvec: Callable[[jax.Array], jax.Array],
    shape: tuple[int, ...],
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    tol: float = 1e-8,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Block Lanczos for lowest eigenvalues of a Hermitian operator.

    Parameters
    ----------
    matvec : (block_size, *shape) -> (block_size, *shape)
        Hermitian matvec operating on a block of vectors.
    shape : tuple
        Shape of a single state vector.
    n_eig : int
        Number of lowest eigenvalues to compute.
    block_size : int
        Number of vectors per Lanczos block.
    max_iter : int
        Maximum number of block iterations.
    tol : float
        Convergence tolerance on beta norm.
    seed : int
        Random seed for initial vectors.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, *shape)
    """
    n_flat = int(np.prod(shape))
    key = jax.random.PRNGKey(seed)

    k1, k2 = jax.random.split(key)
    Q0 = jax.random.normal(k1, (block_size, *shape), dtype=jnp.float64)
    Q0 = Q0 + 1j * jax.random.normal(k2, (block_size, *shape), dtype=jnp.float64)

    Q0_flat = Q0.reshape(block_size, n_flat)
    Q0_flat, _ = jnp.linalg.qr(Q0_flat.T)
    Q0_flat = Q0_flat.T

    Q_blocks = [Q0_flat]
    alpha_blocks = []
    beta_blocks = []

    Q_current = Q0_flat.reshape(block_size, *shape)

    for j in range(max_iter):
        Z = matvec(Q_current)
        Z_flat = Z.reshape(block_size, n_flat)
        Q_current_flat = Q_current.reshape(block_size, n_flat)

        alpha_j = Q_current_flat.conj() @ Z_flat.T
        alpha_blocks.append(alpha_j)

        Z_flat = Z_flat - alpha_j.T @ Q_current_flat
        if j > 0:
            Q_prev_flat = Q_blocks[-2]
            Z_flat = Z_flat - beta_blocks[-1].T @ Q_prev_flat

        for Q_old in Q_blocks:
            proj = Z_flat @ Q_old.conj().T
            Z_flat = Z_flat - proj @ Q_old

        Z_flat_T, R = jnp.linalg.qr(Z_flat.T)
        # P1: beta_j = R (NOT R.T).  With Z_col = Z_flat.T = Q R, the block
        # recurrence residual is Z = Q_{j+1} R, so the sub-diagonal block of T is
        # exactly R and the super-diagonal is R^H.  The old ``R.T`` transposed
        # both off-diagonal blocks (they came out conj(R)/R.T instead of R/R^H),
        # which the final (T+T^H)/2 masked for eigenVALUES but corrupted the
        # T->Q_all mapping used for eigenVECTORS (solver_program P1).
        beta_j = R
        beta_blocks.append(beta_j)

        beta_norm = jnp.linalg.norm(beta_j)
        if beta_norm < tol * block_size:
            print(f"Block Lanczos converged at iteration {j+1}")
            break

        Q_next_flat = Z_flat_T.T
        Q_blocks.append(Q_next_flat)
        Q_current = Q_next_flat.reshape(block_size, *shape)

    # α-Hermiticity gate.  ``alpha_j = Q_jᴴ H Q_j`` is a Hermitian Gram block
    # for any Q_j; the (T + Tᴴ)/2 below would silently absorb any violation.
    # This path is eager, so ``check_hermitian`` applies verbatim to the tiny
    # (n_blocks, bs, bs) stack — no callback needed.
    from common import sanity
    sanity.check_hermitian(
        "block_lanczos_eig alpha (Hermitian form Q^H H Q)",
        jnp.stack(alpha_blocks), rtol=ALPHA_HERM_RTOL)

    n_blocks = len(alpha_blocks)
    T_size = n_blocks * block_size
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    for i, alpha in enumerate(alpha_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        T = T.at[start:end, start:end].set(alpha)

        if i < len(beta_blocks) - 1:
            beta = beta_blocks[i]
            T = T.at[end:end + block_size, start:end].set(beta)
            T = T.at[start:end, end:end + block_size].set(beta.conj().T)

    T = (T + T.conj().T) / 2

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T.real)[:n_eig]
    eigenvalues = evals_T[idx].real

    Q_all = jnp.concatenate(Q_blocks[:n_blocks], axis=0)
    eigenvectors_flat = vecs_T[:, idx].T @ Q_all
    eigenvectors = eigenvectors_flat.reshape(n_eig, *shape)

    norms = jnp.linalg.norm(eigenvectors.reshape(n_eig, -1), axis=1, keepdims=True)
    eigenvectors = eigenvectors / norms.reshape(n_eig, *([1] * len(shape)))

    return eigenvalues, eigenvectors


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


def lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    max_iter: int = 100,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
    reorth: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled Lanczos using lax.fori_loop.

    Parameters
    ----------
    matvec : (n,) -> (n,)
        Hermitian matvec on flat vectors.
    n : int
        Vector dimension.
    n_eig : int
        Number of lowest eigenvalues to compute.
    max_iter : int
        Maximum Lanczos iterations (fixed for JIT).
    seed : int
        Random seed for initial vector.
    n_reorth : int
        Window size for partial reorthogonalization: the ``n_reorth``
        PREVIOUS basis vectors.  The current vector ``q_j`` is always
        projected out as well, at no collective cost — see the route
        section, "THE WINDOW INCLUDES THE CURRENT VECTOR q_j".
        Default ``FULL_REORTH`` (-1) = the whole basis; a finite window is a
        deliberate memory/time trade, not something you should get by not
        choosing.  See "Reorthogonalisation WINDOW" above for the measurement.
        Default ``FULL_REORTH`` (-1) = the whole basis; a finite window is a
        deliberate memory/time trade, not something you should get by not
        choosing.  See "Reorthogonalisation WINDOW" above for the measurement.
    reorth : {'cgs2', 'mgs'} or None
        Reorthogonalisation route; ``None`` reads ``LORRAX_LANCZOS_REORTH``
        and defaults to ``'cgs2'``.  See the route section above — ``'cgs2'``
        issues ``2 * max_iter`` collectives where the legacy ``'mgs'`` sweep
        issues ``max_iter (max_iter + 1) / 2``, over the SAME basis window.

    Returns
    -------
    eigenvalues : (n_eig,)
    eigenvectors : (n_eig, n)
    """
    # Krylov-exhaustion clamp, then the sentinel — in that order, because the
    # window resolves against the depth the loop can actually reach.
    max_iter = max(1, min(int(max_iter), int(n)))
    n_reorth = resolve_n_reorth(n_reorth, max_iter)
    kind = reorth_kind(reorth)
    _announce_reorth("lanczos_eig_jit", kind, max_iter, n_reorth)
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    q0 = jax.random.normal(k1, (n,), dtype=jnp.float64)
    q0 = q0 + 1j * jax.random.normal(k2, (n,), dtype=jnp.float64)
    q0 = q0 / jnp.linalg.norm(q0)

    # +1 column so the last iteration does not overwrite Q[:, max_iter-1] (P1).
    Q = jnp.zeros((n, max_iter + 1), dtype=jnp.complex128)
    Q = Q.at[:, 0].set(q0)
    alpha = jnp.zeros((max_iter,), dtype=jnp.float64)
    beta = jnp.zeros((max_iter,), dtype=jnp.float64)
    # |Im α_j| — carried alongside α, checked once after the loop.  Two extra
    # float64 slots of carry; see the module header for the cost argument.
    alpha_im = jnp.zeros((max_iter,), dtype=jnp.float64)

    def lanczos_step(j, carry):
        Q, alpha, beta, alpha_im, q_prev = carry
        z = matvec(q_prev)

        # ONE complex dot product.  ``.real`` drives the recurrence; ``.imag``
        # is the Hermitian-form residual that used to be discarded here.
        alpha_c = jnp.vdot(q_prev, z)
        alpha_j = alpha_c.real
        alpha = alpha.at[j].set(alpha_j)
        alpha_im = alpha_im.at[j].set(jnp.abs(alpha_c.imag))

        z = z - alpha_j * q_prev
        q_prev_prev = Q[:, jnp.maximum(j - 1, 0)]
        beta_prev = jnp.where(j > 0, beta[j - 1], 0.0)
        z = z - beta_prev * q_prev_prev

        if kind == "cgs2":
            # Same basis window as the sweep below, two batched passes, two
            # collectives — see the route section at the top of this module.
            z = _cgs2_vec(Q, z, _reorth_window(j, max_iter + 1, n_reorth))
        else:
            def reorth_body(i, z_acc):
                valid = i <= j if _REORTH_INCLUDE_CURRENT else i < j
                q_i = Q[:, i]
                proj = jnp.where(valid, jnp.vdot(q_i, z_acc), 0.0 + 0j)
                return z_acc - proj * q_i

            start_idx = jnp.maximum(0, j - n_reorth)
            z = lax.fori_loop(start_idx, j + 1, reorth_body, z)

        beta_j = jnp.linalg.norm(z)
        beta = beta.at[j].set(beta_j)

        q_next = z / jnp.maximum(beta_j, 1e-15)
        Q = Q.at[:, j + 1].set(q_next)

        return (Q, alpha, beta, alpha_im, q_next)

    init_carry = (Q, alpha, beta, alpha_im, q0)
    Q, alpha, beta, alpha_im, _ = lax.fori_loop(
        0, max_iter, lanczos_step, init_carry)

    _emit_alpha_herm("lanczos_eig_jit", alpha_im, jnp.abs(alpha))

    T = jnp.diag(alpha)
    off_diag = beta[:-1]
    T = T + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]

    eigenvectors = (Q[:, :max_iter] @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)

    return eigenvalues, eigenvectors


def _build_block_tridiag(alpha_all, beta_all, max_iter: int, bs: int):
    """Build the block-tridiagonal T from per-iter (bs,bs) blocks.

    Done inside the jit by ``lax.fori_loop`` so the trace-time HLO stays
    O(1) instead of unrolling ``max_iter`` slot updates. Used by both
    the fixed-iter and convergence-driven block Lanczos paths.
    """
    T_size = bs * max_iter
    T = jnp.zeros((T_size, T_size), dtype=jnp.complex128)

    def body(j, T):
        s = j * bs
        T = lax.dynamic_update_slice(T, alpha_all[j], (s, s))
        # Off-diagonal beta only when j+1 < max_iter (zero alpha/beta past
        # the end keeps the slot a no-op even when j is at the boundary).
        T = lax.dynamic_update_slice(T, beta_all[j], (s + bs, s))
        T = lax.dynamic_update_slice(
            T, jnp.conj(beta_all[j]).T, (s, s + bs))
        return T

    T = lax.fori_loop(0, max_iter - 1, body, T)
    # Final diagonal block (no off-diagonal past the end).
    s_last = (max_iter - 1) * bs
    T = lax.dynamic_update_slice(T, alpha_all[max_iter - 1], (s_last, s_last))
    return (T + jnp.conj(T).T) * 0.5


def block_lanczos_eig_jit(
    matvec: Callable[[jax.Array], jax.Array],
    n: int,
    n_eig: int = 20,
    block_size: int = 4,
    max_iter: int = 50,
    seed: int = 42,
    n_reorth: int = FULL_REORTH,
    reorth: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled block Lanczos using ``lax.fori_loop``.

    Same algorithm as :func:`block_lanczos_eig`, but all state lives in
    pre-allocated arrays so the body fits in ``lax.fori_loop`` and the
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
        section, "THE WINDOW INCLUDES THE CURRENT VECTOR q_j".
        Default ``FULL_REORTH`` (-1) = the whole basis.
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
    n_reorth = resolve_n_reorth(n_reorth, int(max_iter))
    T_size = bs * int(max_iter)
    kind = reorth_kind(reorth)
    _announce_reorth("block_lanczos_eig_jit", kind, int(max_iter), n_reorth)

    # Initial orthonormal block via QR of random complex Gaussian.
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0, _ = jnp.linalg.qr(Q0)                          # (n, bs)

    # Ring buffer of all Q-blocks: (max_iter + 1, n, bs) — the +1 slot holds the
    # final Q_next so the last iteration does NOT overwrite Q_{max_iter-1} (the
    # slot-overwrite bug, solver_program P1: it corrupted the last Krylov block
    # in the eigenvector reconstruction).  alpha/beta: (max_iter, bs, bs).
    Q_all = jnp.zeros((int(max_iter) + 1, n, bs), dtype=jnp.complex128)
    Q_all = Q_all.at[0].set(Q0)
    alpha_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((int(max_iter), bs, bs), dtype=jnp.complex128)

    def body(j, carry):
        Q_all, alpha_all, beta_all = carry
        Q_j = Q_all[j]                                 # (n, bs)
        # Block matvec over (bs, n) → (bs, n); transpose to (n, bs).
        Z = matvec(Q_j.T).T                            # (n, bs)

        alpha_j = jnp.conj(Q_j).T @ Z                  # (bs, bs)
        alpha_all = alpha_all.at[j].set(alpha_j)
        Z = Z - Q_j @ alpha_j

        # Subtract Q_{j-1} · β_{j-1}^H (skip on j=0).
        Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
        beta_prev = beta_all[jnp.maximum(j - 1, 0)]
        Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

        # Partial reorth over the last n_reorth blocks.
        if kind == "cgs2":
            Z = _cgs2_block(
                Q_all, Z, _reorth_window(j, int(max_iter) + 1, n_reorth))
        else:
            def reorth_body(i, Z_acc):
                valid = i <= j if _REORTH_INCLUDE_CURRENT else i < j
                Q_i = Q_all[i]
                proj = jnp.where(valid, jnp.conj(Q_i).T @ Z_acc, jnp.zeros((bs, bs), dtype=Z_acc.dtype))
                return Z_acc - Q_i @ proj
            start = jnp.maximum(0, j - n_reorth)
            Z = lax.fori_loop(start, j + 1, reorth_body, Z)

        # QR(Z) → next block + β_j.  Write to slot j+1 (always valid with the
        # +1 buffer, 1..max_iter) — no clobber of the current Q_j.
        Q_next, beta_j = jnp.linalg.qr(Z)              # (n, bs), (bs, bs)
        beta_all = beta_all.at[j].set(beta_j)
        Q_all = Q_all.at[j + 1].set(Q_next)
        return (Q_all, alpha_all, beta_all)

    Q_all, alpha_all, beta_all = lax.fori_loop(
        0, int(max_iter), body, (Q_all, alpha_all, beta_all))

    # α-Hermiticity gate, BEFORE _build_block_tridiag's (T + Tᴴ)/2 absorbs it.
    _emit_alpha_herm("block_lanczos_eig_jit", *_block_alpha_stats(alpha_all),
                     form="block")

    # Block-tridiagonal T built inside-jit (no Python loop unroll).
    T = _build_block_tridiag(alpha_all, beta_all, int(max_iter), bs)

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
    reorth: str | None = None,
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

    Returns (eigenvalues, eigenvectors, n_iter_done) — the third value
    is the actual block iteration count where the loop exited (≤
    ``max_iter``).
    """
    bs = int(block_size)
    # Krylov-exhaustion clamp — same rationale as block_lanczos_eig_jit:
    # past floor(n/bs) blocks the residual collapses and QR manufactures
    # junk directions with arbitrary (even sub-spectrum) Ritz values.
    M = max(1, min(int(max_iter), int(n) // bs))
    T_size = bs * M
    n_reorth = resolve_n_reorth(n_reorth, M)
    kind = reorth_kind(reorth)
    _announce_reorth("block_lanczos_eig_jit_converged", kind, M, n_reorth)
    if min_iter is None:
        min_iter = max(2 * check_every, max(1, n_eig // bs + 1))
    min_iter = int(min(min_iter, M))

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    Q0 = (jax.random.normal(k1, (n, bs), dtype=jnp.float64)
          + 1j * jax.random.normal(k2, (n, bs), dtype=jnp.float64))
    Q0, _ = jnp.linalg.qr(Q0)

    # +1 Krylov slot so the final block does not overwrite Q_{M-1} (P1).
    Q_all = jnp.zeros((M + 1, n, bs), dtype=jnp.complex128).at[0].set(Q0)
    alpha_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    beta_all = jnp.zeros((M, bs, bs), dtype=jnp.complex128)
    last_evals = jnp.full((n_eig,), jnp.inf, dtype=jnp.float64)
    converged = jnp.bool_(False)

    def step(j, Q_all, alpha_all, beta_all):
        Q_j = Q_all[j]
        Z = matvec(Q_j.T).T
        alpha_j = jnp.conj(Q_j).T @ Z
        alpha_all = alpha_all.at[j].set(alpha_j)
        Z = Z - Q_j @ alpha_j
        Q_jm1 = Q_all[jnp.maximum(j - 1, 0)]
        beta_prev = beta_all[jnp.maximum(j - 1, 0)]
        Z = jnp.where(j > 0, Z - Q_jm1 @ jnp.conj(beta_prev).T, Z)

        if kind == "cgs2":
            Z = _cgs2_block(Q_all, Z, _reorth_window(j, M + 1, n_reorth))
        else:
            def reorth_body(i, Z_acc):
                valid = i <= j if _REORTH_INCLUDE_CURRENT else i < j
                Q_i = Q_all[i]
                proj = jnp.where(
                    valid, jnp.conj(Q_i).T @ Z_acc,
                    jnp.zeros((bs, bs), dtype=Z_acc.dtype))
                return Z_acc - Q_i @ proj
            start = jnp.maximum(0, j - n_reorth)
            Z = lax.fori_loop(start, j + 1, reorth_body, Z)

        Q_next, beta_j = jnp.linalg.qr(Z)
        beta_all = beta_all.at[j].set(beta_j)
        Q_all = Q_all.at[j + 1].set(Q_next)
        return Q_all, alpha_all, beta_all

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
            T = _build_block_tridiag(alpha_all, beta_all, M, bs)
            # Mask inactive (zero) part of T by adding a large constant
            # to its diagonal — pushes inactive eigvals out of the
            # spectrum so jnp.sort()[:n_eig] picks only real Ritz vals.
            LARGE = jnp.asarray(1.0e6, dtype=T.real.dtype)
            pos = jnp.arange(M * bs)
            active = (j_done + 1) * bs                        # completed iters × bs
            mask = (pos >= active).astype(T.real.dtype) * LARGE
            T = T + jnp.diag(mask).astype(T.dtype)
            ev = jnp.linalg.eigvalsh(T)
            ev = jnp.sort(ev)[:n_eig]
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

    # Final eigh — mask inactive blocks the same way as the convergence
    # check, otherwise the zero eigvals from unfilled iters dominate
    # ``sort()[:n_eig]``.
    T = _build_block_tridiag(alpha_all, beta_all, M, bs)
    LARGE_F = jnp.asarray(1.0e6, dtype=T.real.dtype)
    pos_F = jnp.arange(T_size)
    active_F = j_final * bs
    mask_F = (pos_F >= active_F).astype(T.real.dtype) * LARGE_F
    T = T + jnp.diag(mask_F).astype(T.dtype)
    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)[:n_eig]
    eigenvalues = evals_T[idx]
    Q_full = jnp.transpose(Q_all[:M], (1, 0, 2)).reshape(n, T_size)
    eigenvectors = (Q_full @ vecs_T[:, idx]).T
    norms = jnp.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors = eigenvectors / jnp.maximum(norms, 1e-15)
    return eigenvalues, eigenvectors, j_final
