"""Local tiled accumulation of a shared contour correlation into outputs."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ffi.common.ffi_loader import get_lib


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
    """
    lib = get_lib()
    jax.ffi.register_ffi_target(
        "lorrax_contour_accumulate", jax.ffi.pycapsule(lib.ContourAccumulateFfi),
        platform="CUDA", api_version=1)

    @partial(jax.shard_map, mesh=mesh,
             in_specs=(P(None, None, 'x', 'y'), P(None, 'x', 'y'), P()),
             out_specs=P(None, None, 'x', 'y'), check_vma=False)
    def accumulate(accumulator, contribution, projection):
        if any(a.dtype != jnp.complex128 for a in
               (accumulator, contribution, projection)):
            raise TypeError("contour accumulator requires complex128 operands")
        return jax.ffi.ffi_call(
            "lorrax_contour_accumulate",
            jax.ShapeDtypeStruct(accumulator.shape, accumulator.dtype),
            input_output_aliases={0: 0},
            vmap_method="sequential",
        )(accumulator, contribution, projection)

    return accumulate
