"""The contour accumulator's door: ``A[o,q,m,n] += p[o] · c[q,m,n]``.

The CUDA handler ``lorrax_contour_accumulate`` (``cpp/response/
contour_accumulate*``) accumulates one shared contour correlation into many
outputs on each device's tile.  CUDA complex128 only, with no host arm and no
fallback: a provider without the handler refuses here (``GATE ffi-handler``)
and at startup (``ffi_loader.require_cuda_handlers``).  The response bank's
Laplace/KMS streams (``gw.w_isdf``) are its caller.
"""
from functools import partial, lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ffi.common.ffi_loader import probe_target

TARGET = "lorrax_contour_accumulate"


@lru_cache(maxsize=None)
def _require():
    """The loader registers the target from its CUDA table; refuse by name if unusable."""
    ok, why = probe_target(TARGET, "CUDA")
    if not ok:
        raise RuntimeError(
            f"GATE ffi-handler: got no usable {TARGET} ({why}); want the CUDA contour "
            "accumulator; fix: use the sealed bundle, or rebuild both legs from this tree "
            "and pin them (LORRAX_FFI_SO, LORRAX_FFI_HOST_SO).")


def contour_accumulator(mesh):
    """Build A[o,q,m,n] += projection[o] contribution[q,m,n].

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Named x/y matrix mesh. The output and q axes stay unsharded.

    Returns
    -------
    callable
        Accepts complex128 accumulator [output,q,m,n] (P(None,None,x,y)),
        correlation [q,m,n] (P(None,x,y)), and replicated [output] weights.
        Units are those of the contour owner: the weights contain the time
        quadrature and value or d/d(z_Ry**2) projection. The caller computes
        the Keldysh difference -i(A-conj(A)) once, before any output loop.
        The native handler allocates zero workspace and aliases its output
        to the accumulator. XLA may copy a non-donated input; account for
        that in compiled memory. CUDA complex128 only; no collectives.
    """
    _require()

    @partial(jax.shard_map, mesh=mesh,
             in_specs=(P(None, None, 'x', 'y'), P(None, 'x', 'y'), P()),
             out_specs=P(None, None, 'x', 'y'), check_vma=False)
    def accumulate(accumulator, contribution, projection):
        if any(a.dtype != jnp.complex128 for a in
               (accumulator, contribution, projection)):
            raise TypeError("contour accumulator requires complex128 operands")
        if (accumulator.ndim != 4 or contribution.ndim != 3
                or projection.ndim != 1
                or accumulator.shape != (projection.shape[0], *contribution.shape)
                or any(n < 1 for n in accumulator.shape)):
            raise ValueError("contour accumulator shape mismatch")
        return jax.ffi.ffi_call(
            TARGET,
            jax.ShapeDtypeStruct(accumulator.shape, accumulator.dtype),
            input_output_aliases={0: 0},
            vmap_method="sequential",
        )(accumulator, contribution, projection)

    return accumulate
