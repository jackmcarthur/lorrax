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

Composition inside a MANUAL-mode ``shard_map``
------------------------------------------------
:meth:`GemmPlan.__call__` cannot be invoked from inside somebody else's
manual-mode ``shard_map`` body.  It is not a missing feature of cuBLASMp —
it is a shape/contract mismatch, investigated and confirmed 2026-08-22
(the ζ-fit r-chunk port, ``isdf.core._z_q_face``,
``docs/architecture/zeta_fit_face_psi_cct.md``'s r-chunk section):
``__call__`` (via ``_build_kernel``) is itself a top-level ``jax.jit``
wrapping its OWN ``shard_map(mesh=..., in_specs=P(None,'x','y'))`` — it
expects GLOBALLY-sharded ``jax.Array`` operands carrying a declared
``NamedSharding`` and performs its own manual-mode entry.  A caller who is
ALREADY inside a manual-mode ``shard_map`` over the SAME mesh axes (e.g. a
streaming ``lax.scan`` kernel doing its own ``all_to_all``/``all_gather``)
holds bare, un-annotated LOCAL tiles with no global shape — re-entering
``shard_map`` over axes that are already manual there is not expressible
in JAX's shard_map model, and even if it were, ``__call__``'s shape check
would refuse the local tile's shape as not matching the declared global
one.

The FFI call itself has no such restriction.  Every wrapper this package
has for a distributed FFI operation (``_cusolvermp.py``,
``matmul_plan.py``'s own ``_build_kernel``) already follows the same
two-part shape: a ``shard_map`` that does nothing but hand the body its
LOCAL per-rank tile, and a body — the transpose/``ffi_call``/transpose
sequence below — that is pure local computation plus one collective FFI
custom-call (the cuBLASMp handler does its OWN NCCL communication across
the mesh via ``ctx_handle``, entirely inside the C++ layer; no JAX-level
collective wraps it).  That body needs nothing from ``shard_map`` except
being handed the right-shaped local array — which is exactly what a
caller's OWN manual-mode body already has in hand.  So the obstacle is the
WRAPPER, exactly as hypothesized: :meth:`GemmPlan.local_call` is the same
body, exposed directly, with no nested ``shard_map`` and no separate
``jax.jit`` of its own — call it inline from inside your own manual
``shard_map``/``lax.scan``.  Proven on real 4-rank CUDA inside a manual
``shard_map`` + ``lax.scan``: ``check_gemm_plan_manual_shard_map`` in
``services/distrib_la/tests/test_distrib_la_multiproc.py``.

Descriptor/workspace lifetime (the owner's stated concern) is a non-issue
for this route: ``batched_gemm_ffi.cc`` builds and destroys its
``cublasMpMatrixDescriptor``/``cublasMpMatmulDescriptor`` handles FRESH on
every FFI invocation (``BatchedGemmImpl``, read 2026-08-22) — the only
state persisting across calls is ``ctx`` itself (the NCCL communicator +
growable workspace buffer), addressed by the SAME ``ctx_handle`` integer
:func:`gemm_plan` already resolved once, baked as a static attr into
whichever call issues it.  A ``local_call`` invocation and an ``__call__``
invocation of the same plan therefore share the identical warmed
communicator; there is nothing further to warm for the manual route.

This does NOT make every manual-mode local contraction a GEMM
opportunity.  ``local_call`` produces a genuinely 2-D-distributed output —
``D`` at ``P(None,'x','y')``, sharded on BOTH mesh axes, matching its
inputs — because that is what SUMMA computes.  A manual kernel that needs
a mesh-axis-REPLICATED result instead (every rank holding the full
answer, not its own tile) is asking for a different communication
primitive — an all-gather/broadcast, which ``jax.lax.psum``/
``all_gather`` already provide at their own, typically lower, cost — and
wrapping a broadcast in extra GEMM FLOPs would not help it.
``isdf.core._z_q_face``'s masked-``psum('y')`` X-operand reconstruction is
exactly this second case: see that function's own docstring for the
shape mismatch that keeps it on ``psum``, not GEMM.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la.matmul import (_CUBLASMP_CACHE, _OP_CODE, _TARGETS, _mesh_shape, _zeros,
                               resolve_matmul_backend)
from distrib_la.resolve import mesh_key

__all__ = ["GemmPlan", "gemm_plan", "local_gemm_plan"]

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


def _gemm_attrs(*, px, py, nq, m, k, n, alpha, beta, ctx_handle,
                with_c: bool) -> dict:
    """The FFI attrs dict for one exact N,N shape — the SINGLE place this
    package computes it, shared by the auto-mode (``_build_kernel``) and
    manual-mode (``GemmPlan.local_call``) entry points so the two routes
    cannot drift apart on block sizes or leading dimensions."""
    return dict(
        nq=nq, m=m, n=n, k=k,
        mb_a=m // px, nb_a=k // py, mb_b=k // px, nb_b=n // py,
        mb_c=m // px, nb_c=n // py,
        lld_a=m // px, lld_b=k // px, lld_c=m // px,
        transa=_OP_CODE["N"], transb=_OP_CODE["N"],
        alpha_re=float(alpha.real), alpha_im=float(alpha.imag),
        beta_re=float(beta.real) if with_c else 0.0,
        beta_im=float(beta.imag) if with_c else 0.0,
        ctx_handle=int(ctx_handle))


def _local_gemm_call(a, b, c, *, attrs: dict, out_t, with_c: bool):
    """The bare transpose/``ffi_call``/transpose body — LOCAL per-rank
    tiles in, LOCAL per-rank tile out, no ``shard_map`` and no ``jax.jit``
    of its own.  This is the whole GEMM: everything above it (a
    ``shard_map`` in ``_build_kernel``, or nothing at all in
    :meth:`GemmPlan.local_call`) exists only to get the caller a
    correctly-shaped local tile to pass in, not to make this body correct.
    Safe to call directly from inside an ALREADY-manual ``shard_map``
    trace — see the module docstring, "Composition inside a MANUAL-mode
    shard_map".
    """
    at, bt = (jnp.transpose(x, (0, 2, 1)) for x in (a, b))
    ct = (jnp.transpose(c, (0, 2, 1)) if with_c
          else jnp.zeros(out_t.shape, dtype=out_t.dtype))
    dt = jax.ffi.ffi_call(
        _TARGETS["cublasmp"], out_t,
        input_output_aliases={2: 0})(at, bt, ct, **attrs)
    return jnp.transpose(dt, (0, 2, 1))


def _build_kernel(mesh, *, px, py, nq, m, k, n, dtype, alpha, beta,
                  ctx_handle, with_c: bool) -> Callable:
    """The uncompiled shard_map+ffi_call N,N GEMM body for one exact shape
    — the AUTO-mode entry point (``GemmPlan.__call__``): the caller hands
    in GLOBALLY-sharded operands, this ``shard_map`` extracts the local
    tile, and :func:`_local_gemm_call` does the actual work.

    Mirrors ``distrib_la.matmul._cublasmp``'s column-major-transpose /
    ``ctx_handle`` / attrs convention — the only other cuBLASMp GEMM FFI
    wrapper in this package — specialised to N,N and to this module's
    eager-warm-then-call lifecycle instead of ``matmul()``'s per-call
    resolve-and-cache.  The two are cross-checked numerically on real
    multi-rank CUDA (``check_gemm_plan_cublasmp`` in
    ``test_distrib_la_multiproc.py``) rather than sharing code, because
    ``_cublasmp`` also has to carry transa/transb generality this module
    deliberately does not.

    ``with_c=True``: ``fn(a, b, c) -> d``, ``c`` DONATED as the FFI output
    buffer (matches ``matmul()``'s own ``donate_argnums=(2,)``).
    ``with_c=False``: ``fn(a, b) -> d``, beta forced to 0 and the zero-C
    addend is built with ``jnp.zeros`` INSIDE this same traced function —
    one compiled program, not two.  See the module docstring "Output
    liveness".
    """
    key = ("planned", mesh_key(mesh), px, py, nq, m, k, n, str(dtype),
           alpha, beta, int(ctx_handle), with_c)
    if key in _CUBLASMP_CACHE:
        return _CUBLASMP_CACHE[key]
    attrs = _gemm_attrs(px=px, py=py, nq=nq, m=m, k=k, n=n, alpha=alpha,
                        beta=beta, ctx_handle=ctx_handle, with_c=with_c)
    out_t = jax.ShapeDtypeStruct((nq, n // py, m // px), dtype)

    if with_c:
        @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
                 out_specs=P(None, "x", "y"), check_vma=False)
        def _local(a, b, c):
            return _local_gemm_call(a, b, c, attrs=attrs, out_t=out_t,
                                    with_c=True)
        _CUBLASMP_CACHE[key] = _local
        return _local

    @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 2,
             out_specs=P(None, "x", "y"), check_vma=False)
    def _local(a, b):
        return _local_gemm_call(a, b, None, attrs=attrs, out_t=out_t,
                                with_c=False)
    _CUBLASMP_CACHE[key] = _local
    return _local


def _check_local_operand(plan: "GemmPlan", label: str, x,
                         shape: tuple[int, int, int]) -> None:
    """Shape/dtype check for :meth:`GemmPlan.local_call`'s bare LOCAL
    tiles.  No sharding check: a manual-mode local value carries no
    meaningful global ``NamedSharding`` to compare against — the mesh
    itself is the caller's responsibility (see ``local_call``'s
    docstring)."""
    xshape = getattr(x, "shape", None)
    if xshape is None or tuple(int(s) for s in xshape) != shape:
        raise ValueError(
            f"gemm_plan.local_call {label}: expected LOCAL tile shape "
            f"{shape} (this plan's (nq, extent/px_or_py) split — see "
            f"local_call's docstring, not the global (nq,m,k)-class "
            f"shape __call__ takes); got "
            f"{None if xshape is None else tuple(xshape)}")
    if jnp.dtype(x.dtype) != plan.dtype:
        raise TypeError(
            f"gemm_plan.local_call {label}: expected dtype {plan.dtype}; "
            f"got {x.dtype}")


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
                f"{sharding.spec} on the plan's mesh; refusing an "
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
    ctx_handle: int
    _fn_with_c: Callable = field(compare=False, hash=False)
    _fn_no_c: Callable | None = field(compare=False, hash=False)
    reduction_axis: str | None = None

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

    def local_call(self, A, B, C=None, *, out=None):
        """The SAME planned N,N GEMM as :meth:`__call__`, callable from
        INSIDE a manual-mode ``shard_map`` — the composition ``__call__``
        cannot do (module docstring, "Composition inside a MANUAL-mode
        shard_map").

        Precondition, entirely on the caller: this must be invoked from
        the body of a ``shard_map(mesh=plan.mesh, ...)`` whose manual axes
        include ``'x'`` and ``'y'`` — the exact mesh this plan was built
        on.  ``A``/``B``/``C``/``out`` are then the BARE LOCAL tiles that
        body already holds, not global ``jax.Array``s:

            A : (nq, m // px, k // py)
            B : (nq, k // px, n // py)
            C, out : (nq, m // px, n // py)

        (``px, py = plan.mesh.shape['x'], plan.mesh.shape['y']``.)  No
        ``NamedSharding``/global-shape check is possible or attempted on
        these — only shape and dtype, against the LOCAL split derived from
        the plan.  Getting the ambient mesh wrong (a different mesh
        object, a transposed axis order, more or fewer manual axes) is not
        detectable here; it was already the caller's own manual
        ``shard_map`` contract before this method existed.

        No ``jax.jit`` and no ``shard_map`` of its own: this traces as
        ordinary ops inside whatever outer ``jax.jit``/``lax.scan`` is
        already tracing the caller, exactly like every other primitive
        (``jnp.take``, ``lax.psum``, an ``io_callback``) a manual kernel
        body already calls directly.  Same C/out mutual exclusion, same
        beta-gated ``out=`` refusal, same donated-buffer semantics as
        :meth:`__call__` — see that method's docstring for the shared
        contract; only the operand SHAPES differ (local tile vs. global
        array).
        """
        if C is not None and out is not None:
            raise ValueError("gemm_plan.local_call: pass C or out, not both")
        if out is not None and self.beta != 0:
            raise ValueError(
                "gemm_plan.local_call: out= is only a content-ignored "
                f"donation on a beta==0 plan (this plan's beta={self.beta})"
                ".  Pass C= instead, where the accumulate is explicit at "
                "the call site.")
        px, py = _mesh_shape(self.mesh)
        if self.backend == "local":
            a_shape = tuple(size // (self.mesh.shape[axis] if axis else 1)
                            for size, axis in zip((self.nq, self.m, self.k), self.in_sharding_a.spec))
            b_shape = tuple(size // (self.mesh.shape[axis] if axis else 1)
                            for size, axis in zip((self.nq, self.k, self.n), self.in_sharding_b.spec))
            _check_local_operand(self, "A", A, a_shape)
            _check_local_operand(self, "B", B, b_shape)
            c = C if C is not None else out
            if c is None and self.beta != 0:
                raise ValueError("gemm_plan.local_call: C is required when beta != 0")
            if c is not None:
                c_shape = tuple(size // (self.mesh.shape[axis] if axis else 1)
                                for size, axis in zip((self.nq, self.m, self.n), self.out_sharding.spec))
                _check_local_operand(self, "C/out", c, c_shape)
            return _axis_matmul(A, B, c, alpha=self.alpha, beta=self.beta,
                                reduction_axis=self.reduction_axis)
        _check_local_operand(self, "A", A, (self.nq, self.m // px, self.k // py))
        _check_local_operand(self, "B", B, (self.nq, self.k // px, self.n // py))
        c_or_out = C if C is not None else out
        out_t = jax.ShapeDtypeStruct(
            (self.nq, self.n // py, self.m // px), self.dtype)
        if c_or_out is None:
            if self.beta != 0:
                raise ValueError(
                    "gemm_plan.local_call: C is required when beta != 0 "
                    f"(this plan's beta={self.beta})")
            attrs = _gemm_attrs(px=px, py=py, nq=self.nq, m=self.m, k=self.k,
                                n=self.n, alpha=self.alpha, beta=self.beta,
                                ctx_handle=self.ctx_handle, with_c=False)
            return _local_gemm_call(A, B, None, attrs=attrs, out_t=out_t,
                                    with_c=False)
        _check_local_operand(self, "C/out", c_or_out,
                             (self.nq, self.m // px, self.n // py))
        attrs = _gemm_attrs(px=px, py=py, nq=self.nq, m=self.m, k=self.k,
                            n=self.n, alpha=self.alpha, beta=self.beta,
                            ctx_handle=self.ctx_handle, with_c=True)
        return _local_gemm_call(A, B, c_or_out, attrs=attrs, out_t=out_t,
                                with_c=True)


def _axis_matmul(a, b, c=None, *, alpha, beta, reduction_axis=None):
    """Contract local bands or centroid tiles with the requested centroid reduction."""
    scale = alpha if a.dtype.kind == "c" else alpha.real
    result = scale * jnp.matmul(a, b)
    if reduction_axis is not None:
        result = jax.lax.psum_scatter(result, reduction_axis,
            scatter_dimension=1 if reduction_axis == "x" else 2, tiled=True)
    if beta != 0:
        scale = beta if a.dtype.kind == "c" else beta.real
        result = result + scale * c
    return result


def local_gemm_plan(mesh: Mesh, *, m: int, k: int, n: int, nq: int,
                    dtype, alpha=1.0, beta=0.0, reduction_axis=None, out_spec=None) -> GemmPlan:
    """Warm A(q,m_X,k) B(q,k,n_Y) → D(q,m_X,n_Y) with a replicated contraction axis."""
    m, k, n, nq = (_as_extent(label, value) for label, value in
                   (("m", m), ("k", k), ("n", n), ("nq", nq)))
    dtype = jnp.dtype(dtype)
    _validate_dtype(dtype)
    alpha, beta = _as_scalar("alpha", alpha), _as_scalar("beta", beta)
    if dtype.kind != "c" and (alpha.imag or beta.imag):
        raise ValueError("local_gemm_plan: alpha/beta must be real for real dtype")
    px, py = _mesh_shape(mesh)
    out_spec = P(None, "x", "y") if out_spec is None else out_spec
    if out_spec not in (P(None, "x", "y"), P(None, "x", None), P(None, None, "y")):
        raise ValueError("local_gemm_plan: output must retain its centroid axis shards")
    for extent, axis in zip((m, n), out_spec[1:]):
        if axis is not None and extent % mesh.shape[axis]:
            raise ValueError("local_gemm_plan: output extent does not tile its mesh axis")
    a_spec, b_spec = P(None, out_spec[1], None), P(None, None, out_spec[2])
    if reduction_axis is not None and out_spec != P(None, "x", "y"):
        raise ValueError("local_gemm_plan: centroid reduction requires the two-axis output")
    if reduction_axis == "y":
        a_spec, b_spec = P(None, "x", "y"), P(None, "y", None)
    elif reduction_axis == "x":
        a_spec, b_spec = P(None, None, "x"), P(None, "x", "y")
    elif reduction_axis is not None:
        raise ValueError("local_gemm_plan: reduction_axis must be x, y or None")
    if reduction_axis is not None and k % mesh.shape[reduction_axis]:
        raise ValueError("local_gemm_plan: contraction extent must tile its reduction axis")
    a_sh, b_sh, out_sh = (NamedSharding(mesh, spec)
                           for spec in (a_spec, b_spec, out_spec))
    local = partial(_axis_matmul, alpha=alpha, beta=beta, reduction_axis=reduction_axis)
    with_c = jax.jit(shard_map(local, mesh=mesh,
        in_specs=(a_spec, b_spec, out_spec), out_specs=out_spec,
        check_vma=False), donate_argnums=(2,))
    with_c(_zeros((nq, m, k), dtype, a_sh),
           _zeros((nq, k, n), dtype, b_sh),
           _zeros((nq, m, n), dtype, out_sh))
    no_c = None
    if beta == 0:
        no_c = jax.jit(shard_map(local, mesh=mesh,
            in_specs=(a_spec, b_spec), out_specs=out_spec, check_vma=False))
        no_c(_zeros((nq, m, k), dtype, a_sh), _zeros((nq, k, n), dtype, b_sh))
    return GemmPlan(mesh=mesh, backend="local", m=m, k=k, n=n, nq=nq,
                    dtype=dtype, alpha=alpha, beta=beta,
                    in_sharding_a=a_sh, in_sharding_b=b_sh, out_sharding=out_sh,
                    ctx_handle=0, _fn_with_c=with_c, _fn_no_c=no_c, reduction_axis=reduction_axis)


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
    layout="face",
    reduction_axis=None,
    out_spec=None,
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
    if layout == "axis":
        return local_gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                               alpha=alpha, beta=beta, reduction_axis=reduction_axis, out_spec=out_spec)
    if layout != "face":
        raise ValueError(f"gemm_plan: unknown psi layout {layout!r}")
    if out_spec is not None and out_spec != P(None, "x", "y"):
        raise ValueError("gemm_plan: single-axis output requires layout=axis")
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
        out_sharding=out_sharding, ctx_handle=int(ctx_handle),
        _fn_with_c=fn_with_c, _fn_no_c=fn_no_c)
