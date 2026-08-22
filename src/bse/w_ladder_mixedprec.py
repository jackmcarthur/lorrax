"""Mixed-precision twin of the ladder-W block solve: a ``complex64`` Krylov
iteration wrapped in ``complex128`` ITERATIVE REFINEMENT.

WHY THIS CAN EXIST AT ALL, in one sentence: ``bse_feast._gmres_solve_core``
takes every dtype it uses from ``b.dtype`` and from the OPERAND arrays, and the
ladder matvec (``bse_ring_comm.build_bse_ring_matvec_full``) carries no
hard-coded ``complex128`` on the production path — its one dtype-bearing
constant is ``sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=T.real.dtype))``, which
FOLLOWS the trial vector.  So the "c64 twin" is not a second implementation of
the operator: it is the SAME compiled structure traced on down-cast operands.
Nothing in this file duplicates a kernel; it casts operands, drives the shared
core, and adds the refinement loop that makes the low-precision solve deliver a
double-precision answer.

The algorithm — classical mixed-precision iterative refinement on a Krylov
solve (Buttari/Dongarra/Langou; Turner & Walker for the GMRES form):

    x_0 = 0
    for r = 1 .. R:
        rho   = || b - A_128 x_{r-1} ||          (c128 matvec; r=1 is free, x=0)
        d     = GMRES_64( A_64 d = (b - A_128 x_{r-1}) / rho , tol_lo )
        x_r   = x_{r-1} + rho * d                (c128 accumulate)

``A`` is the SHIFTED operator ``z I - H_ladder``.  The scaling by ``rho`` is not
cosmetic: the residual shrinks by ~``tol_lo`` per round, so by round 3 it is
``1e-18``-class relative to ``b``, and c64's smallest normal is ``1.2e-38`` —
without the rescale the low-precision right-hand side would walk into the
subnormal range and the inner solve would return noise.  With it, every inner
solve sees a unit-norm right-hand side no matter how converged the outer one is.

WHAT IS AND IS NOT DOWN-CAST.  Down-cast: the 14 matvec operands, the
preconditioner diagonal, the shift ``z``, and the Krylov workspace (V, Z, H, the
Givens state) — i.e. everything inside the inner GMRES.  Kept c128: the probe
SEED (``build_probe_rhs``), the outer residual/accumulate, and the stage-3
READOUT/assembly.  The refinement loop is therefore a solver-internal
approximation only: the fixed point it converges to is the c128 solution of the
c128 operator, so this is not an approximation-class change to the physics.

WHAT THE ENGINE RETURNS is deliberately the SAME 3-tuple shape
``(s_all, resids, iters)`` that ``bse_w_exact._get_block_gmres_solver`` returns,
so :func:`apply_mixed_resolvent_block` is a drop-in for
``bse_w_exact.apply_screening_resolvent_block`` at the call site
``w_ladder.sweep_q_wedge`` uses.  ``resids`` is the TRUE ``||b||``-relative
residual of the shifted system, measured in c128 — never the inner solve's
projected one, and never relative to ``||r_0||`` (the tolerance-semantics defect
``_gmres_solve_core`` documents under ``resid_relative_to``).

``low_dtype=jnp.complex128`` turns the whole thing into an A/B CONTROL: the
identical refinement engine at full precision, which prices the engine's own
overhead (the extra c128 residual matvecs) separately from the precision win.
MEASURED 2026-08-16, gnppm_debug q=0 z=0, interleaved min-of-3: the control is
6.03 s against the production engine's 6.04 s and reproduces its iteration
count, true residual and tile to every printed digit — the engine is free, and
every difference in the mixed rows below is precision.

TF32 IS A CORRECTNESS PRECONDITION OF THIS FILE, NOT A TUNING KNOB
------------------------------------------------------------------
XLA:GPU lowers ``float32`` dots at ``DEFAULT`` precision to TENSORFLOAT32 — a
10-bit mantissa, ``eps ~ 4.9e-04``.  A complex64 dot is decomposed into real
f32 dots, so the whole c64 matvec inherits it.  MEASURED on the gnppm_debug
ladder operator (``bench_w_ladder_mixedprec.py --dtype-audit``):

    operand REPRESENTATION error (round to c64, arithmetic in c128)  4.65e-08
    true c64 program, XLA default precision (TF32)                   1.90e-04
    true c64 program, jax_default_matmul_precision="highest"         3.22e-07

i.e. the c64 program is 4000x worse than its own representation floor until
the precision is pinned, and pinning it costs NOTHING at the production block
width (nb=1: 3.425 ms at DEFAULT against 3.406 ms at ``highest``).  Since the
refinement rate is exactly this forward error, TF32 turns a 2-round schedule
into a 4-round one.  **Any caller of this module must set
``jax.config.update("jax_default_matmul_precision", "highest")`` before the
first trace.**  c128 is untouched either way — f64 dots have no TF32 path — so
this knob moves only the low-precision arm.

Since 2026-08-22 ``runtime.pin_matmul_precision`` (called by
``runtime.bootstrap``, which this module calls at import) applies that pin for
the whole process, so the refusal below is a LOCAL guard on a global
precondition rather than the only thing holding it.  Both are kept: the pin is
what makes the default right, the refusal is what makes a caller that bypassed
``bootstrap`` fail at the door instead of three orders of magnitude later.

WHAT THE c64 SOLVE CAN AND CANNOT REACH.  The floor of a single c64 GMRES on
this operator is a TRUE relative residual of 3.5e-05 and a tile error of
4.9e-05 (measured; tightening the inner tolerance from 1e-05 to 1e-07 buys 4
more iterations and NOTHING else, because the projected residual keeps falling
past the point where the c64 matvec stops describing the c128 operator).  That
floor is the operator's conditioning times the 3.2e-07 forward error, so it is
DECK-DEPENDENT: measure it before trusting a round count on a new system.  Each
refinement round multiplies the residual by that floor, so 2 rounds reach
~1e-09 and 3 rounds reach ~1e-13.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp

from .bse_feast import _apply_shifted_matvec, _gmres_solve_core
from .bse_w_exact import build_probe_rhs

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# Down-casting
# --------------------------------------------------------------------------
#: c128 -> c64, f64 -> f32; everything else (integers, already-low arrays)
#: passes through untouched.  Applied to the matvec operand tuple and to the
#: preconditioner diagonal, and to NOTHING else.
_LOWER = {jnp.dtype(jnp.complex128): jnp.complex64,
          jnp.dtype(jnp.float64): jnp.float32}


def lower_dtype(dtype):
    """The complex64/float32 partner of a c128/f64 dtype (identity otherwise)."""
    return _LOWER.get(jnp.dtype(dtype), jnp.dtype(dtype))


def cast_low(x, complex_dtype=jnp.complex64):
    """Down-cast one array: complex -> ``complex_dtype``, real float -> its
    matching real part; leave integers and booleans alone.

    ``complex_dtype=jnp.complex128`` is the identity on a c128/f64 payload, which
    is what makes the c128 control arm run through the SAME code."""
    if not hasattr(x, "dtype"):
        return x
    cdt = jnp.dtype(complex_dtype)
    rdt = jnp.finfo(cdt).dtype                      # c64 -> f32, c128 -> f64
    if jnp.issubdtype(x.dtype, jnp.complexfloating):
        return x.astype(cdt) if x.dtype != cdt else x
    if jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(rdt) if x.dtype != rdt else x
    return x


def cast_operands(operands, complex_dtype=jnp.complex64):
    """Down-cast a ``matvec_operands`` / ``ladder_matvec_operands`` tuple.

    THE WHOLE "is a dtype-threaded twin needed?" QUESTION LIVES HERE.  The
    ladder matvec builder takes its dtypes from the operands, so casting the
    tuple is sufficient to get a c64 program PROVIDED nothing inside the matvec
    re-promotes.  The one real-valued pair is ``(eps_c, eps_v)``: left at f64
    they would promote the ``D`` term's product back to c128 and, through it,
    the whole trial vector — which is why this casts REAL arrays too rather
    than only the complex ones.  Verified by
    ``tests/bench/bench_w_ladder_mixedprec.py --dtype-audit``, which reads the
    jaxpr of the compiled matvec and refuses if any c128 remains."""
    return tuple(cast_low(o, complex_dtype) for o in operands)


# --------------------------------------------------------------------------
# TF32 precondition
# --------------------------------------------------------------------------
#: The f32 dot precisions that are actually fp32.  ``"high"`` is NOT among
#: them: on XLA:GPU it selects a 3-pass tf32 decomposition, better than plain
#: tf32 and still short of fp32.
_PINNED = ("highest", "float32")

#: Announced escape, for A/B measurement only (the bench's ``--precision``
#: sweep is the one legitimate caller).  Named rather than silent, per the
#: "the required layer never demotes silently" rule the FFI gate states.
_ALLOW_TF32_ENV = "LORRAX_MIXEDPREC_ALLOW_TF32"


def _refuse_unpinned_matmul_precision() -> None:
    """Refuse to build a c64 solver while f32 dots may lower to TensorFloat32.

    This is a REFUSAL and not a warning because the failure is silent and
    large: at XLA:GPU's default precision the c64 ladder matvec carries
    1.9e-04 forward error instead of 3.2e-07 (measured, module docstring), the
    GMRES floor moves with it, and a refinement schedule tuned at ``highest``
    quietly stops converging to its advertised residual.  Pinning costs nothing
    at the production block width."""
    import os
    if os.environ.get(_ALLOW_TF32_ENV, "") == "1":
        return
    prec = getattr(jax.config, "jax_default_matmul_precision", None)
    if prec is not None and str(prec).lower().split(".")[-1] in _PINNED:
        return
    raise RuntimeError(
        f"w_ladder_mixedprec: jax_default_matmul_precision is {prec!r}; the "
        f"complex64 solve requires one of {_PINNED}.  XLA:GPU lowers f32 dots "
        f"at DEFAULT precision to TensorFloat32 (10-bit mantissa), which "
        f"MEASURED 2026-08-16 costs this matvec 1.9e-04 forward error against "
        f"3.2e-07 when pinned — a 600x worse refinement rate, silently.  Call "
        f"jax.config.update('jax_default_matmul_precision', 'highest') before "
        f"the first trace; it is free at the production block width (nb=1: "
        f"3.425 -> 3.406 ms).  {_ALLOW_TF32_ENV}=1 is the announced escape for "
        f"A/B measurement.")


# --------------------------------------------------------------------------
# The refinement engine
# --------------------------------------------------------------------------
_MIXED_CACHE: dict = {}


def get_mixed_block_solver(matvec, sh, max_iter: int, rounds: int,
                           low_dtype=jnp.complex64,
                           resid_relative_to: str = "b",
                           verify: bool = True):
    """Cached jitted per-column-scan REFINED block solver.

    Signature of the returned callable:
    ``(rhs, diag_lo, z, ops_hi, ops_lo, tol_lo) -> (s_all, resids, iters)``.

    ``rounds`` and ``verify`` are STATIC (they unroll a Python loop of c128
    matvecs), so each ``(operator, max_iter, rounds, dtype)`` is one compiled
    program — the same cache discipline ``_get_block_gmres_solver`` uses, and
    for the same reason: the operand arrays are runtime arguments, so every
    ``(q, z)`` after the first is dispatch-only.

    ``iters`` is the SUM of the inner GMRES exit indices over the rounds, i.e.
    the number of LOW-precision matvecs the column actually cost.  It is not
    comparable column-for-column with the baseline's count without the
    per-matvec price of the two precisions beside it; that is what the bench
    measures.
    """
    key = (id(matvec), int(max_iter), int(rounds), str(jnp.dtype(low_dtype)),
           str(resid_relative_to), bool(verify))
    hit = _MIXED_CACHE.get(key)
    if hit is not None:
        return hit[1]
    if int(rounds) < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if jnp.dtype(low_dtype) != jnp.dtype(jnp.complex128):
        _refuse_unpinned_matmul_precision()

    cdt = jnp.dtype(low_dtype)
    n_rounds, do_verify = int(rounds), bool(verify)

    @jax.jit
    def _block(rhs, diag_lo, z, ops_hi, ops_lo, tol_lo):
        # ``tol_lo`` is a (rounds,) ARRAY: round r's inner tolerance.  A
        # per-round schedule is the whole cost story of refinement — round 1
        # must reach the c64 floor, but round 2 only has to shrink a
        # correction that is ALREADY ~1e-6 of ``b``, so a loose tolerance
        # there buys the same final residual for half the iterations.
        z_hi = jnp.asarray(z, dtype=jnp.complex128)
        z_lo = jnp.asarray(z, dtype=cdt)
        rhs_scan = jnp.moveaxis(rhs, 1, 0)            # (nu, 2, c, v, k)

        def _solve_col(carry, rhs_col):
            b = rhs_col[:, None]                      # (2, 1, c, v, k), c128
            nb = jnp.linalg.norm(b)
            x = jnp.zeros_like(b)
            res = b                                   # round 1: x = 0, r = b
            k_tot = jnp.zeros((), dtype=jnp.int32)
            for r in range(n_rounds):
                if r > 0:
                    res = b - _apply_shifted_matvec(matvec, x, z_hi, ops_hi)
                rho = jnp.linalg.norm(res)
                safe = jnp.where(rho == 0.0, jnp.ones_like(rho), rho)
                # UNIT-NORM low-precision right-hand side, always: see the
                # module docstring on subnormals.
                b_lo = (res / safe).astype(cdt)
                d, k = _gmres_solve_core(matvec, b_lo, diag_lo, z_lo, ops_lo,
                                         max_iter, tol_lo[r],
                                         resid_relative_to=resid_relative_to)
                x = x + safe.astype(jnp.complex128) * d.astype(jnp.complex128)
                k_tot = k_tot + k.astype(jnp.int32)
            if do_verify:
                r_true = b - _apply_shifted_matvec(matvec, x, z_hi, ops_hi)
            else:
                r_true = res
            resid = jnp.where(nb == 0.0, jnp.asarray(0.0, dtype=nb.dtype),
                              jnp.linalg.norm(r_true) / nb)
            s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
            return carry, (s[0], resid, k_tot)

        # unroll=1 for the reason _get_block_gmres_solver states: one Krylov
        # workspace alive at a time.
        _, (s_all, resids, iters) = jax.lax.scan(_solve_col, None, rhs_scan,
                                                 unroll=1)
        s_all = jax.lax.with_sharding_constraint(s_all, sh.X)
        return s_all, resids, iters

    _MIXED_CACHE[key] = (matvec, _block)
    return _block


def apply_mixed_resolvent_block(G_zeta, z, data, matvec, diag_h, gen, snapshot,
                                sh, *, max_iter: int,
                                tol_low,
                                operands_fn,
                                rounds: Optional[int] = None,
                                low_dtype=jnp.complex64,
                                rhs=None,
                                verify: bool = True,
                                solve_data=None,
                                snapshot_v=None):
    """Drop-in for ``bse_w_exact.apply_screening_resolvent_block`` with the
    mixed-precision refined solve in stage 2.

    Stages 1 (SEED) and 3 (PROJECT) are the PRODUCTION ones, called here, not
    copied: the seed is ``build_probe_rhs`` (c128) and the readout is the
    caller's ``snapshot``.  Only the middle changes.

    ``tol_low`` is a scalar (broadcast over ``rounds``) or a per-round
    sequence, in which case ``rounds`` defaults to its length.

    Returns ``(W_tile, resids, iters)`` — always the 3-tuple; ``resids`` is the
    TRUE c128 ``||b||``-relative residual per probe column.
    """
    tl = np.atleast_1d(np.asarray(tol_low, dtype=np.float64))
    if rounds is None:
        rounds = int(tl.size)
    if tl.size == 1:
        tl = np.repeat(tl, int(rounds))
    if tl.size != int(rounds):
        raise ValueError(
            f"tol_low has {tl.size} entries but rounds={rounds}; pass one "
            "tolerance per refinement round (or a single scalar).")
    if rhs is None:
        rhs = build_probe_rhs(G_zeta, data, gen, sh)
    src = data if solve_data is None else solve_data
    ops_hi = operands_fn(src)
    ops_lo = cast_operands(ops_hi, low_dtype)
    diag_lo = cast_low(diag_h, low_dtype)
    solver = get_mixed_block_solver(matvec, sh, max_iter, rounds,
                                    low_dtype=low_dtype, verify=verify)
    rdt = jnp.finfo(jnp.dtype(low_dtype)).dtype
    s_all, resids, iters = solver(rhs, diag_lo,
                                  jnp.asarray(z, dtype=jnp.complex128),
                                  ops_hi, ops_lo,
                                  jnp.asarray(tl, dtype=rdt))
    W_tile = snapshot(s_all, data["psi_c_Y"], data["psi_v_Y"],
                      data["V_q0"] if snapshot_v is None else snapshot_v)
    return W_tile, resids, iters


# --------------------------------------------------------------------------
# Diagnostics used by the bench (kept here so the bench stays a driver)
# --------------------------------------------------------------------------
def jaxpr_dtype_census(fn, *args) -> dict:
    """Count the dtypes of every var in ``jax.make_jaxpr(fn)(*args)``.

    The audit that answers "did operand casting actually produce a c64
    program?" without trusting the wall clock: if a single c128 array survives
    inside the traced matvec, it shows up here."""
    jaxpr = jax.make_jaxpr(fn)(*args)
    census: dict = {}

    def _walk(jx):
        for v in list(jx.invars) + list(jx.outvars) + [
                o for eq in jx.eqns for o in eq.outvars]:
            aval = getattr(v, "aval", None)
            dt = getattr(aval, "dtype", None)
            if dt is None:
                continue
            census[str(dt)] = census.get(str(dt), 0) + 1
        for eq in jx.eqns:
            for p in eq.params.values():
                sub = getattr(p, "jaxpr", None)
                if sub is not None:
                    _walk(getattr(sub, "jaxpr", sub))
                elif isinstance(p, (list, tuple)):
                    for q in p:
                        sub = getattr(q, "jaxpr", None)
                        if sub is not None:
                            _walk(getattr(sub, "jaxpr", sub))

    _walk(jaxpr.jaxpr)
    return census


def hermiticity(tile, nlog: Optional[int] = None) -> float:
    """``max|W - W^dag| / max|W|`` on the logical block of a host tile — the
    q=0 gate statistic ``w_ladder`` step 6 quotes (6.9e-15 on the fixture)."""
    a = np.asarray(tile)
    if nlog is not None:
        a = a[:nlog, :nlog]
    den = np.max(np.abs(a))
    return float(np.max(np.abs(a - a.conj().T)) / (den if den else 1.0))
