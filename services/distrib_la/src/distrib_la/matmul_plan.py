"""Planned, trace-safe cuBLASMp N,N GEMM — resolve/warm once, call inside a
hot loop.

``distrib_la.matmul`` resolves its provider and probes capability at every
call (``matmul.py:100-151,393-420``) — correct for an eager call site, but
Green's-function construction and the per-tau Sigma projector run inside
outer ``jax.jit``/``lax.scan`` kernels (``gw/ppm_tau_kernel.py:311-317,
438-490``), where redoing that resolution/probe/context-bootstrap work at
every trace is exactly the split this package's own :class:`distrib_la.Plan`
and :class:`distrib_la.polar.PolarPlan` already keep apart: an EAGER phase
(dlopen, probe, mesh geometry, communicator warmup) that runs ONCE, and a
TRACE-SAFE closure built from its result.

:func:`gemm_plan` is that split for GEMM.  It mirrors
:func:`distrib_la.polar.plan_polar_factor`/:class:`~distrib_la.polar.PolarPlan`
— the one existing precedent in this package for driving an FFI call from
inside a composed, jitted kernel: shape/mesh/backend are fixed and resolved
EAGERLY, and the returned :class:`GemmPlan` is called with only
static-shape/dtype/layout checks (safe on a tracer) plus a pre-built,
pre-compiled ``jax.jit`` executable.

Contract, deliberately narrower than :func:`distrib_la.matmul`:

* **N,N only.**  There is no ``transa``/``transb`` parameter anywhere in
  this module.  Multi-rank cuBLASMp's transpose modes are refused by
  ``distrib_la.matmul`` itself — a real P=4 gate found transpose-A returns
  a WRONG result and transpose-B can deadlock (``matmul.py:191-202``) — so
  this planned surface never offers them.  A caller with a transposed
  operand pretransposes into the complementary face layout; that is
  exactly what the two-face ``psi_nmu``/``psi_mun`` convention this
  surface exists for does, once at load time, instead of once per GEMM.
* **One leading batch, replicated — holds k.**  Every operand is rank 3 at
  ``P(None,'x','y')`` with a FIXED ``nq``, resolved into the plan at
  construction.  A spinor axis is not a second batch dimension: flatten
  ``(s, mu)``/``(s, n)`` into the contraction/output axis before calling
  (the audit's own GEMM-seam convention), or call this SAME plan ``ns``
  times in a small, statically unrolled Python loop — ``ns`` in {1,2,4} is
  a compile-time constant, so unrolling it costs no extra trace or
  compile.  Arbitrary batch ranks are out of contract, matching the
  underlying provider.  A caller that genuinely wants a bare rank-2 call
  builds a plan with ``nq=1`` and passes/unwraps a leading axis of 1
  itself — cuBLASMp's own per-slice loop makes ``nq=1`` a legitimate,
  zero-overhead special case, the same way ``distrib_la.matmul`` lifts
  rank-2 internally (``matmul.py:402-406,438``).
* **cuBLASMp only, today.**  ``lorrax_scalapack_batched_gemm`` and
  ``lorrax_slate_batched_gemm`` are claimed by ``distrib_la.loader``'s
  target table but have no C++ definition anywhere in this tree
  (``KNOWN_LORRAX_ISSUES.md``, "services/distrib_la loader vs src/ffi"
  row) — confirmed again here by `nm -D` on the pinned CUDA library, which
  exports only ``CublasMpBatchedGemmFfi``.  A request that resolves to
  either provider refuses at :func:`gemm_plan` construction, by name, using
  the SAME capability probe ``distrib_la.matmul`` already runs — this
  module adds no leniency and no second probe path.
* **Provider route only.**  ``batch_reshard`` materializes complete A, B,
  C and D on every device (``matmul.py:377-384``); the whole reason a
  caller reaches for a *planned* GEMM is a G/Sigma-sized object that must
  never be that.  :func:`gemm_plan` refuses ``backend='off'`` by name —
  the only spelling that would otherwise select the staged route.

Output liveness.  A plan built with ``beta=0`` (the default) compiles a
SECOND warmed kernel that builds its zero addend with ``jnp.zeros`` INSIDE
the same compiled program as the GEMM FFI call, so a repeated call never
pays a separate top-level ``jax.jit`` dispatch — the pattern
``distrib_la.matmul`` uses when ``C`` is omitted (``matmul.py:433-437``) —
or an allocation that has to exist as its own executable's output before
the provider's own call even starts.  Passing an existing buffer as
``out=`` skips that internal zero-fill entirely and donates the buffer's
storage to the provider instead — the shape a caller threading a scratch
accumulator through a ``lax.scan`` carry wants.  Neither path removes the
C++ handler's own requirement of a live ``C`` argument: cuBLASMp's
batched-GEMM FFI binds ``C`` as a required buffer unconditionally
(``src/ffi/cpp/cublasmp/batched_gemm_ffi.cc``, ``.Arg<AnyBuffer>() // C``),
so there is no PROVIDER-level "no C at all" mode to expose without an FFI
signature change, which is out of this module's scope.  What this module
removes is the extra Python-level allocation and the extra compiled
program, not the C++ argument — state that distinction when reporting the
memory win, rather than claiming a C-less FFI call.

Communication, not layout, is this surface's dominant cost per call at
multi-node scale — measured 2026-08-22 (``gw.greens_function_kernel``'s
module docstring carries the full writeup and numbers;
``reports/face_gemm_contiguity_2026-08-22/report.md``).  A context-free
:func:`gemm_plan` call at G-build's own shape got 4.4x SLOWER moving one
node (P4) to four nodes (P16) despite the per-rank operand shrinking 32x,
and a `nq`-widening batch experiment (folding two same-shape calls into
one) measured within noise of two separate calls at both scales — this
surface's own ``jnp.transpose`` bridging to cuBLASMp's column-major FFI
compiles to a bitcast (free), so neither is a layout problem to chase
here.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la.matmul import (_OP_CODE, _TARGETS, _mesh_shape, _zeros,
                               resolve_matmul_backend)

__all__ = ["GemmPlan", "gemm_plan"]

_SUPPORTED_DTYPES = (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128))


def _as_extent(label: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"gemm_plan: {label} must be a positive integer, got {value!r}")
    try:
        out = operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"gemm_plan: {label} must be a positive integer, "
            f"got {value!r}") from exc
    if out <= 0:
        raise ValueError(f"gemm_plan: {label} must be positive, got {out}")
    return int(out)


def _as_scalar(label: str, value) -> complex:
    try:
        return complex(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"gemm_plan: {label} must be a real or complex scalar, "
            f"got {value!r}") from exc


def _validate_dtype(dtype) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        allowed = "|".join(str(x) for x in _SUPPORTED_DTYPES)
        raise TypeError(f"gemm_plan: dtype must be {allowed}; got {dtype}")


def _same_layout(have, want: NamedSharding) -> bool:
    return (getattr(have, "spec", None) == want.spec
            and getattr(have, "mesh", None) == want.mesh)


def _build_kernel(mesh, *, px, py, nq, m, k, n, dtype, alpha, beta,
                  ctx_handle, with_c: bool) -> Callable:
    """The uncompiled shard_map+ffi_call N,N GEMM body for one exact shape.

    Mirrors ``distrib_la.matmul._cublasmp``'s column-major-transpose /
    ``ctx_handle`` / attrs convention — the only cuBLASMp GEMM FFI wrapper
    in this package — specialised to N,N and to this module's
    eager-warm-then-call lifecycle instead of ``matmul()``'s per-call
    resolve-and-cache.  The two are cross-checked numerically on real
    multi-rank CUDA (``test_distrib_la_matmul_plan_multiproc.py``) rather
    than sharing code, because ``_cublasmp`` also has to carry transa/
    transb generality this module deliberately does not.

    ``with_c=True``: ``fn(a, b, c) -> d``, ``c`` DONATED as the FFI output
    buffer (matches ``matmul()``'s own ``donate_argnums=(2,)``).
    ``with_c=False``: ``fn(a, b) -> d``, beta forced to 0 and the zero-C
    addend is built with ``jnp.zeros`` INSIDE this same traced function —
    one compiled program, not two.  See the module docstring "Output
    liveness".
    """
    attrs = dict(
        nq=nq, m=m, n=n, k=k,
        mb_a=m // px, nb_a=k // py, mb_b=k // px, nb_b=n // py,
        mb_c=m // px, nb_c=n // py,
        lld_a=m // px, lld_b=k // px, lld_c=m // px,
        transa=_OP_CODE["N"], transb=_OP_CODE["N"],
        alpha_re=float(alpha.real), alpha_im=float(alpha.imag),
        beta_re=float(beta.real) if with_c else 0.0,
        beta_im=float(beta.imag) if with_c else 0.0,
        ctx_handle=int(ctx_handle))
    out_t = jax.ShapeDtypeStruct((nq, n // py, m // px), dtype)

    if with_c:
        @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
                 out_specs=P(None, "x", "y"), check_vma=False)
        def _local(a, b, c):
            at, bt, ct = (jnp.transpose(x, (0, 2, 1)) for x in (a, b, c))
            dt = jax.ffi.ffi_call(
                _TARGETS["cublasmp"], out_t,
                input_output_aliases={2: 0})(at, bt, ct, **attrs)
            return jnp.transpose(dt, (0, 2, 1))
        return _local

    @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 2,
             out_specs=P(None, "x", "y"), check_vma=False)
    def _local(a, b):
        at, bt = (jnp.transpose(x, (0, 2, 1)) for x in (a, b))
        c0 = jnp.zeros(out_t.shape, dtype=out_t.dtype)
        dt = jax.ffi.ffi_call(
            _TARGETS["cublasmp"], out_t,
            input_output_aliases={2: 0})(at, bt, c0, **attrs)
        return jnp.transpose(dt, (0, 2, 1))
    return _local


def _check_operand(plan: "GemmPlan", label: str, x, shape: tuple[int, int, int],
                   sharding: NamedSharding) -> None:
    xshape = getattr(x, "shape", None)
    if xshape is None or tuple(int(s) for s in xshape) != shape:
        raise ValueError(
            f"gemm_plan {label}: expected shape {shape}; got "
            f"{None if xshape is None else tuple(xshape)}")
    if jnp.dtype(x.dtype) != plan.dtype:
        raise TypeError(
            f"gemm_plan {label}: expected dtype {plan.dtype}; got {x.dtype}")
    # A tracer's layout is fixed by the outer jit boundary and reinforced
    # by this plan's own in_specs; a concrete operand must already obey
    # the contract — silently device_put'ing an (nq,m,k)-class array is
    # exactly the hidden reshard distrib_la.polar refuses for the same
    # reason (polar.py:125-134).
    if not isinstance(x, jax.core.Tracer):
        have = getattr(x, "sharding", None)
        if not _same_layout(have, sharding):
            raise ValueError(
                f"gemm_plan {label}: must already be sharded "
                f"P(None,'x','y') on the plan's mesh; refusing an "
                f"implicit reshard of a {shape} array.  Got {have!r}.")


@dataclass(frozen=True)
class GemmPlan:
    """A resolved, warmed, trace-safe ``D[q] = alpha*A[q]@B[q] (+ beta*C[q])``.

    Construct with :func:`gemm_plan`; never instantiate directly.  Every
    eager step — mesh/topology refusal, provider resolution and capability
    probe, the cuBLASMp communicator, and both compiled kernels (the
    donated-``C`` form, and — when ``beta==0`` — the internal-zero form)
    — has already run and already executed once on real dummy data by the
    time :func:`gemm_plan` returns.  Calling the plan touches none of
    that: shape/dtype/layout checks that read only static metadata, then
    the pre-built executable.
    """

    mesh: Mesh
    backend: str
    m: int
    k: int
    n: int
    nq: int
    dtype: object
    alpha: complex
    beta: complex
    in_sharding_a: NamedSharding
    in_sharding_b: NamedSharding
    out_sharding: NamedSharding
    _fn_with_c: Callable
    _fn_no_c: Callable | None

    def describe(self) -> str:
        """One line for a run banner: what resolved, and to what shape."""
        px, py = _mesh_shape(self.mesh)
        return (f"gemm_plan: {self.backend} N,N on {px}x{py}, "
                f"shape (nq={self.nq}, m={self.m}, k={self.k}, n={self.n}), "
                f"dtype={self.dtype}, alpha={self.alpha}, beta={self.beta}")

    def __call__(self, A, B, C=None, *, out=None):
        """Return ``D``.  Trace-safe: usable inside nested ``jit``/``scan``.

        ``C`` (accumulate; required when this plan's ``beta != 0``) and
        ``out`` (a live buffer DONATED purely for its storage when
        ``beta == 0`` — its content is ignored) are mutually exclusive.
        With neither, and ``beta == 0``, the zero addend is built inside
        the same compiled call — see the module docstring.

        ``out=`` is refused on a ``beta != 0`` plan.  Both ``C`` and
        ``out`` reach the identical compiled ``_fn_with_c`` — the PLAN's
        own ``beta`` (fixed at construction, not chosen per call) decides
        whether that buffer's content is mathematically live.  On a
        ``beta != 0`` plan, ``out=``'s "content is ignored, pure storage"
        contract would silently be false: whatever the buffer happened to
        hold gets scaled by ``beta`` and added into the result.  Refuse
        by name rather than let a caller who read only the general
        ``out=`` framing above get a silently wrong accumulate from stale
        buffer content; use ``C=`` on such a plan, where the accumulate
        semantics are explicit at the call site.
        """
        if C is not None and out is not None:
            raise ValueError("gemm_plan: pass C or out, not both")
        if out is not None and self.beta != 0:
            raise ValueError(
                "gemm_plan: out= is only a content-ignored donation on a "
                f"beta==0 plan (this plan's beta={self.beta}).  On a "
                "beta!=0 plan out='s buffer content would be scaled by "
                "beta and added into the result -- pass C= instead, "
                "where that accumulate is explicit at the call site.")
        _check_operand(self, "A", A, (self.nq, self.m, self.k),
                      self.in_sharding_a)
        _check_operand(self, "B", B, (self.nq, self.k, self.n),
                      self.in_sharding_b)
        c_or_out = C if C is not None else out
        if c_or_out is None:
            if self.beta != 0:
                raise ValueError(
                    "gemm_plan: C is required when beta != 0 "
                    f"(this plan's beta={self.beta})")
            if self._fn_no_c is None:
                raise AssertionError(
                    "gemm_plan: internal-zero kernel was not warmed for a "
                    "beta==0 plan; this is a construction bug, not a "
                    "caller error")
            return self._fn_no_c(A, B)
        _check_operand(self, "C/out", c_or_out, (self.nq, self.m, self.n),
                      self.out_sharding)
        return self._fn_with_c(A, B, c_or_out)


def gemm_plan(
    mesh: Mesh,
    *,
    m: int,
    k: int,
    n: int,
    nq: int,
    dtype,
    backend: str = "auto",
    alpha=1.0,
    beta=0.0,
) -> GemmPlan:
    """Eagerly resolve, probe, warm and COMPILE one N,N GEMM shape, ONCE.

    Hoist this call out of every per-k/per-tau loop — G build, Sigma
    projection, Hartree.  By the time this returns, the cuBLASMp
    communicator exists and both kernel variants (donated-``C``, and —
    when ``beta==0`` — internal-zero-``C``) are compiled AND HAVE RUN ONCE
    on real dummy data, so :meth:`GemmPlan.__call__` never dlopens, never
    probes, never builds a ``jax.jit`` wrapper, and never traces for the
    first time from inside somebody else's ``lax.scan``.

    Parameters
    ----------
    mesh
        Exact 2-D ``('x','y')`` mesh, square, y-minor process order, one
        JAX process per cell — the same contract
        :func:`distrib_la.resolve_matmul_backend` enforces for an explicit
        provider request, reused here rather than re-derived.
    m, k, n
        ``D[q] = alpha * A[q] @ B[q] (+ beta * C[q])``, per q in
        ``range(nq)``.  ``A`` is ``(nq,m,k)``, ``B`` is ``(nq,k,n)``,
        ``D``/``C`` are ``(nq,m,n)``, every one at ``P(None,'x','y')``.
        ``m`` and ``n`` must tile the mesh (``m % px == 0``,
        ``n % py == 0``, matching ``matmul()``'s own output-face check);
        ``k`` must tile BOTH axes (``k % px == 0`` and ``k % py == 0``)
        because it is simultaneously A's column face (``py``) and B's row
        face (``px``).
    nq
        The one replicated leading batch — holds k-points.  A spinor axis
        does not get a second one: flatten it into m/k/n, or call this
        SAME plan ``ns`` times in a small, statically unrolled loop (see
        the module docstring).  ``nq=1`` is a legal, zero-overhead
        rank-2-equivalent plan.
    dtype
        ``float64`` or ``complex128`` — the two dtypes the cuBLASMp
        handler compiles for.
    backend
        A name from ``distrib_la.MATMUL_BACKEND_CHOICES`` other than
        ``'off'``.  ``'auto'``/``'distributed'`` resolve to the platform's
        provider exactly as ``distrib_la.matmul`` does; only
        ``cublasmp``/``cusolvermp`` have a warmed kernel in this module
        today (see the module docstring) — a resolved ``scalapack``/
        ``slate`` refuses BY NAME rather than silently falling back to a
        route this module does not implement.
    alpha, beta
        Fixed GEMM scalars, baked into the compiled kernel — cuBLASMp
        takes them as FFI attributes, not array arguments, so they cannot
        vary per call the way ``distrib_la.matmul``'s do.  ``beta=0`` (the
        default, and what every G/T/Sigma GEMM in the low_mem_bands audit
        needs) additionally compiles the internal-zero-``C`` kernel;
        ``beta != 0`` compiles only the donated-``C`` kernel, and every
        call must then supply ``C``.
    """
    px, py = _mesh_shape(mesh)
    m = _as_extent("m", m)
    k = _as_extent("k", k)
    n = _as_extent("n", n)
    nq = _as_extent("nq", nq)
    dtype = jnp.dtype(dtype)
    _validate_dtype(dtype)
    alpha_c = _as_scalar("alpha", alpha)
    beta_c = _as_scalar("beta", beta)
    if dtype.kind != "c" and (alpha_c.imag or beta_c.imag):
        raise ValueError(
            f"gemm_plan: alpha/beta must be real for real dtype {dtype}")

    requested = str(backend)
    if requested == "off":
        raise ValueError(
            "gemm_plan: backend='off' has no provider, and this planned "
            "surface never selects batch_reshard — it materializes "
            "complete A/B/C/D on every device, exactly what a planned "
            "GEMM exists to avoid for a G/Sigma-sized operand.  Use "
            "distrib_la.matmul(..., batched_route='batch_reshard') "
            "directly for that route.")
    resolved = resolve_matmul_backend(requested, mesh, batched_route="auto")
    if resolved != "cublasmp":
        raise NotImplementedError(
            f"gemm_plan: backend {requested!r} resolved to {resolved!r}, "
            "which has no warmed kernel in this module yet — only "
            "cuBLASMp is wired (see the module docstring).  "
            "distrib_la.matmul() capability-probed this provider and "
            "found it usable, so the gap is this module's, not a missing "
            f"library; land the {resolved} kernel variant here before "
            "requesting it planned, or call distrib_la.matmul() directly "
            "for this provider.")

    for label, extent, divisor in (("m", m, px), ("k", k, px),
                                   ("k", k, py), ("n", n, py)):
        if extent % divisor:
            raise ValueError(
                f"gemm_plan: {label}={extent} does not tile the "
                f"{px}x{py} mesh (needs divisor {divisor})")

    from distrib_la._cusolvermp import get_or_init_context
    ctx_handle = get_or_init_context(mesh, col_major=False)

    in_sharding_a = NamedSharding(mesh, P(None, "x", "y"))
    in_sharding_b = in_sharding_a
    out_sharding = in_sharding_a

    fn_with_c = jax.jit(
        _build_kernel(mesh, px=px, py=py, nq=nq, m=m, k=k, n=n, dtype=dtype,
                     alpha=alpha_c, beta=beta_c, ctx_handle=ctx_handle,
                     with_c=True),
        donate_argnums=(2,))
    # A REAL warmup call, not merely a trace: this is what forces the
    # cuBLASMp matmul descriptor build and workspace allocation to happen
    # now, eagerly, rather than on this plan's first use inside a caller's
    # scan.  The buffers are throwaway (c0 is DONATED away by this call).
    fn_with_c(_zeros((nq, m, k), dtype, in_sharding_a),
             _zeros((nq, k, n), dtype, in_sharding_b),
             _zeros((nq, m, n), dtype, out_sharding))

    fn_no_c = None
    if beta_c == 0:
        fn_no_c = jax.jit(_build_kernel(
            mesh, px=px, py=py, nq=nq, m=m, k=k, n=n, dtype=dtype,
            alpha=alpha_c, beta=beta_c, ctx_handle=ctx_handle,
            with_c=False))
        fn_no_c(_zeros((nq, m, k), dtype, in_sharding_a),
               _zeros((nq, k, n), dtype, in_sharding_b))

    return GemmPlan(
        mesh=mesh, backend=resolved, m=m, k=k, n=n, nq=nq, dtype=dtype,
        alpha=alpha_c, beta=beta_c,
        in_sharding_a=in_sharding_a, in_sharding_b=in_sharding_b,
        out_sharding=out_sharding, _fn_with_c=fn_with_c, _fn_no_c=fn_no_c)
