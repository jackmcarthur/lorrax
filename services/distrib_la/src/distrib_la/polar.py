"""Distributed polar factor from a Hermitian-dilation SVD.

The factorization is built from the service's planned eigh operation on
[[0, A], [A.H, 0]].  It never diagonalizes A.H @ A, which would square the
condition number.  Only the length-n singular-value vector is replicated;
all matrix-shaped work stays two-dimensionally sharded at P('x','y').
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la.plan import Plan, plan
from distrib_la.resolve import mesh_key

__all__ = ["PolarPlan", "plan_polar_factor", "polar_factor"]


_SUPPORTED_DTYPES = (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128))

# One fused executable per resolved plan/signature.  The value retains its
# plan and mesh.  Fusing the dilation, planned eigh and final distributed
# GEMM avoids a Python/JIT boundary per streamed k-point.
_KERNEL_CACHE: dict[tuple, Callable] = {}

# The one-shot door also caches plans so a streaming loop cannot accidentally
# repeat backend probing/dlopen.  Explicit planning remains the preferred
# spelling when the operation is called from another traced function.
_PLAN_CACHE: dict[tuple, "PolarPlan"] = {}


def _as_extent(n) -> int:
    """Return a positive integer matrix extent, refusing lossy coercions."""
    if isinstance(n, bool):
        raise ValueError(f"polar_factor: n must be a positive integer, got {n!r}")
    try:
        out = operator.index(n)
    except TypeError as exc:
        raise ValueError(
            f"polar_factor: n must be a positive integer, got {n!r}") from exc
    if out <= 0:
        raise ValueError(f"polar_factor: n must be positive, got {out}")
    return int(out)


def _as_rcond(rcond) -> float | None:
    """Validate the relative numerical-rank cutoff at eager plan time."""
    if rcond is None:
        return None
    if isinstance(rcond, bool):
        raise ValueError(
            f"polar_factor: rcond must be a finite non-negative real, "
            f"got {rcond!r}")
    try:
        out = float(rcond)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"polar_factor: rcond must be a finite non-negative real, "
            f"got {rcond!r}") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(
            f"polar_factor: rcond must be a finite non-negative real, "
            f"got {rcond!r}")
    return out


def _mesh_contract(mesh: Mesh, n: int) -> tuple[int, int]:
    """Validate the public P('x','y') shape contract."""
    axes = tuple(getattr(mesh, "axis_names", ()))
    if "x" not in axes or "y" not in axes:
        raise ValueError(
            "polar_factor: mesh must contain axes ('x','y'); "
            f"got {axes}")
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if n % px or n % py:
        divisor = math.lcm(px, py)
        n_pad = ((n + divisor - 1) // divisor) * divisor
        raise ValueError(
            f"polar_factor: n={n} cannot have layout P('x','y') on a "
            f"{px}x{py} mesh; n must be divisible by both mesh axes.  "
            f"Zero-pad A to at least n={n_pad}, factor the padded matrix, "
            f"then slice the logical leading block.")
    return px, py


def _same_layout(have, want: NamedSharding) -> bool:
    return (getattr(have, "spec", None) == want.spec
            and getattr(have, "mesh", None) == want.mesh)


def _validate_dtype(dtype) -> None:
    """Refuse precisions the distributed eigh backends do not share."""
    dtype = jnp.dtype(dtype)
    if dtype not in _SUPPORTED_DTYPES:
        allowed = "|".join(str(x) for x in _SUPPORTED_DTYPES)
        raise TypeError(
            f"polar_factor: A.dtype must be {allowed}; got {dtype}.  "
            "The distributed eigh backends share this precision contract.")


def _validate_operand(A, polar_plan: "PolarPlan") -> None:
    """Call-time rank, shape, dtype and concrete-layout refusal ladder."""
    shape = getattr(A, "shape", None)
    if shape is None:
        raise TypeError(
            "polar_factor: A must be a jax.Array with shape (n,n) and "
            "layout P('x','y')")
    if len(shape) != 2:
        raise ValueError(
            f"polar_factor: A must have rank 2 and shape (n,n); got {shape}")
    if shape[0] != shape[1]:
        raise ValueError(f"polar_factor: A must be square; got shape {shape}")
    if int(shape[0]) != polar_plan.n:
        raise ValueError(
            f"polar_factor: planned n={polar_plan.n}, got A.shape={shape}")

    _validate_dtype(A.dtype)

    # A tracer's layout is fixed by the outer jit boundary and reinforced by
    # this plan's inner in_shardings.  Concrete input must already obey the
    # contract: silently device_put'ing n^2 data is a hidden full-matrix move.
    if not isinstance(A, jax.core.Tracer):
        have = getattr(A, "sharding", None)
        if not _same_layout(have, polar_plan.in_sharding):
            raise ValueError(
                "polar_factor: A must already be sharded at P('x','y') "
                "on the supplied mesh; refusing an implicit n^2 reshard.  "
                f"Got {have!r}.")


@dataclass(frozen=True)
class PolarPlan:
    """A planned, trace-safe distributed polar-factor operation.

    The object owns the invariant joining the physical extent, numerical
    rank cutoff, layout, and one eagerly resolved Hermitian-eigh plan.
    Rank-deficient inputs return the canonical polar partial isometry:
    directions satisfying s <= rcond * max(s) contribute zero.  A unitary
    extension over null spaces is non-unique and is therefore not invented.
    """

    mesh: Mesh
    n: int
    requested: str
    rcond: float | None
    eigh_plan: Plan
    in_sharding: NamedSharding
    singular_value_sharding: NamedSharding

    @property
    def backend(self) -> str:
        """Concrete backend selected for the Hermitian dilation."""
        return self.eigh_plan.backend

    @property
    def is_native(self) -> bool:
        """Whether the planned Hermitian eigensolve is pure JAX."""
        return self.eigh_plan.is_native

    def describe(self) -> str:
        """One-line resolved operation description for startup banners."""
        cutoff = "dtype default" if self.rcond is None else repr(self.rcond)
        return (f"polar_factor: {self.requested!r} -> {self.backend} "
                f"(Hermitian dilation n={2 * self.n}, "
                f"A/L at P('x','y'), s replicated, rcond={cutoff})")

    def __call__(self, A):
        """Return (L, s) for one planned square matrix A.

        A and L have shape (n,n) at P('x','y').  The replicated s has shape
        (n,), is non-negative, and is sorted in NumPy SVD order (descending).
        """
        _validate_operand(A, self)
        return _kernel_for(self, jnp.dtype(A.dtype))(A)


def _kernel_for(polar_plan: PolarPlan, dtype) -> Callable:
    """Build/cache the fused dilation-eigh-polar executable."""
    dtype = jnp.dtype(dtype)
    key = (mesh_key(polar_plan.mesh), polar_plan.n, polar_plan.backend,
           polar_plan.rcond, str(dtype))
    fn = _KERNEL_CACHE.get(key)
    if fn is not None:
        return fn

    n = polar_plan.n
    tile = polar_plan.in_sharding
    replicated = polar_plan.singular_value_sharding
    eigh = polar_plan.eigh_plan
    if polar_plan.rcond is None:
        real_dtype = jnp.empty((), dtype=dtype).real.dtype
        rcond = float(n) * float(jnp.finfo(real_dtype).eps)
    else:
        rcond = polar_plan.rcond

    def _polar(A):
        A = jax.lax.with_sharding_constraint(A, tile)

        # One allocated 2n x 2n tile, rather than four concatenated operands.
        # upper is the upper-right block; upper + upper.H is the dilation.
        upper = jnp.pad(A, ((0, n), (n, 0)))
        H = upper + jnp.conj(jnp.swapaxes(upper, -1, -2))
        H = jax.lax.with_sharding_constraint(H, tile)

        evals, Q = eigh(H)

        # eigh is ascending.  Its upper half is the non-negative dilation
        # spectrum; reverse to standard descending singular-value order.
        s_ascending = jnp.maximum(evals[n:], 0)
        positive = Q[:, n:]
        root2 = jnp.asarray(math.sqrt(2.0), dtype=dtype)
        U = root2 * positive[:n]
        V = root2 * positive[n:]

        # Dilation zero modes mix the independent left/right null spaces.
        # Mask before GEMM to produce the unique partial isometry rather than
        # a backend-dependent null-space pairing.
        cutoff = (jnp.asarray(rcond, dtype=s_ascending.dtype)
                  * jnp.max(s_ascending))
        keep = (s_ascending > cutoff).astype(dtype)
        L = (U * keep[None, :]) @ jnp.conj(jnp.swapaxes(V, -1, -2))

        L = jax.lax.with_sharding_constraint(L, tile)
        # Reverse only the replicated O(n) vector.  Reversing the distributed
        # eigenvector columns would add a whole-matrix cross-device permutation
        # even though the order cancels from U @ V.H.
        s = s_ascending[::-1]
        s = jax.lax.with_sharding_constraint(s, replicated)
        return L, s

    fn = jax.jit(_polar, in_shardings=tile,
                 out_shardings=(tile, replicated))
    _KERNEL_CACHE[key] = fn
    return fn


def plan_polar_factor(
    mesh: Mesh,
    *,
    n: int,
    backend: str = "distributed",
    rcond: float | None = None,
) -> PolarPlan:
    """Eagerly resolve one reusable distributed polar-factor plan.

    Hoist this call out of streamed k-point loops.  n is the physical matrix
    extent and must be divisible by both mesh axes.  For a non-divisible
    logical band count, zero-pad rows and columns to the next common multiple,
    factor that matrix, then slice the leading logical block and singular
    values.  Thresholded null directions make this zero padding safe.

    Parameters
    ----------
    mesh
        JAX mesh containing axes ('x','y').
    n
        Positive physical square-matrix extent.
    backend
        Eigh backend request; 'distributed' by default.
    rcond
        Relative rank cutoff.  None uses n * eps(dtype).
    """
    n = _as_extent(n)
    _mesh_contract(mesh, n)
    rcond = _as_rcond(rcond)
    requested = str(backend)
    # Polar is a single distributed dilation, not a batched operation. Keep
    # it on the provider/native face route when the public batched default is
    # ``batch_reshard``.
    eig = plan("eigh", mesh, backend=backend, n=2 * n,
               batched_route="auto")
    return PolarPlan(
        mesh=mesh,
        n=n,
        requested=requested,
        rcond=rcond,
        eigh_plan=eig,
        in_sharding=NamedSharding(mesh, P("x", "y")),
        singular_value_sharding=NamedSharding(mesh, P()),
    )


def polar_factor(
    A,
    mesh: Mesh,
    *,
    backend: str = "distributed",
    rcond: float | None = None,
):
    """Return the distributed polar factor and singular values of A.

    This top-level convenience caches eager plans by mesh/shape/backend/cutoff.
    For an outer jit, eagerly call plan_polar_factor and invoke the returned
    trace-safe PolarPlan inside the traced function.

    Parameters
    ----------
    A
        Rank-2 square float64 or complex128 JAX array at P('x','y').
    mesh
        Mesh that shards A.
    backend
        Planned eigh backend; 'distributed' by default.
    rcond
        Relative numerical-rank cutoff.

    Returns
    -------
    L
        Polar factor with A's shape, dtype and P('x','y') sharding.
    s
        Replicated real singular values in descending order.
    """
    if isinstance(A, jax.core.Tracer):
        raise RuntimeError(
            "polar_factor performs eager backend planning and cannot be "
            "entered for the first time from a JAX trace.  Call "
            "plan_polar_factor(mesh, n=..., ...) eagerly, then invoke the "
            "returned trace-safe PolarPlan inside jit.")
    shape = getattr(A, "shape", None)
    if shape is None or len(shape) != 2:
        got = None if shape is None else shape
        raise ValueError(
            f"polar_factor: A must have rank 2 and shape (n,n); got {got}")
    if shape[0] != shape[1]:
        raise ValueError(f"polar_factor: A must be square; got shape {shape}")
    _validate_dtype(A.dtype)
    n = _as_extent(shape[0])
    rcond_value = _as_rcond(rcond)
    key = (mesh_key(mesh), n, str(backend), rcond_value)
    polar_plan = _PLAN_CACHE.get(key)
    if polar_plan is None:
        polar_plan = plan_polar_factor(
            mesh, n=n, backend=backend, rcond=rcond_value)
        _PLAN_CACHE[key] = polar_plan
    return polar_plan(A)
