"""The contour accumulator's entry point: ``A[o,q,m,n] += p[o] · c[q,m,n]``.

The CUDA handler ``lorrax_contour_accumulate`` (``cpp/response/
contour_accumulate*``) accumulates one shared contour correlation into many
outputs on each device's tile.  On CUDA a provider without the handler refuses
here (``GATE ffi-handler``) and at startup (``ffi_loader.require_cuda_handlers``).
On a host mesh both forms are the same sum written in XLA (no host handler is
built).  The response bank's Laplace/KMS streams (``gw.w_isdf``) and the
four-current packed carry (``gw.photon_layout``) are its callers.
"""
from functools import partial, lru_cache

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ffi.common.ffi_loader import probe_target
from ffi.gate import mesh_ffi_platform

TARGET = "lorrax_contour_accumulate"
BLOCK_TARGET = "lorrax_contour_accumulate_block"


@lru_cache(maxsize=None)
def _require(target=TARGET):
    """The loader registers the target from its CUDA table; refuse by name if unusable."""
    ok, why = probe_target(target, "CUDA")
    if not ok:
        raise RuntimeError(
            f"GATE ffi-handler: got no usable {target} ({why}); want the CUDA contour "
            "accumulator; fix: use the sealed bundle, or rebuild both legs from this tree "
            "and pin them (LORRAX_FFI_SO, LORRAX_FFI_HOST_SO).")


def _host(mesh):
    """True off CUDA, where the accumulators are XLA."""
    return mesh_ffi_platform(mesh) != "CUDA"


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
        that in compiled memory. complex128 only; no collectives. On a host
        mesh the same sum in XLA.
    """
    host = _host(mesh)
    if not host:
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
        if host:
            return accumulator + projection[:, None, None, None] * contribution[None]
        return jax.ffi.ffi_call(
            TARGET,
            jax.ShapeDtypeStruct(accumulator.shape, accumulator.dtype),
            input_output_aliases={0: 0},
            vmap_method="sequential",
        )(accumulator, contribution, projection)

    return accumulate


def contour_block_accumulate_local(accumulator, contribution, projection, valid, *, m0, n0,
                                   mesh):
    """One device tile: ``A[o, q, m0+m, n0+n] += sum_s projection[s,o] contribution[s,q,m,n]``.

    For code already inside ``shard_map``.  ``accumulator`` ``[output,q,M,N]``
    (updated in place: donate it), ``contribution`` ``[terms,q,bm,bn]``,
    ``projection`` ``[terms,output]`` complex128, ``valid`` ``[2]`` int32: the
    block's rows and columns that are not padding (the rest are not touched).
    The terms add in order with the full accumulator's rounding, so terms
    ``(a, b)`` give the bytes of two :func:`contour_accumulator` calls.
    ``mesh`` (the enclosing ``shard_map``'s) picks the CUDA handler or, on a
    host mesh, the same sum in XLA.  complex128 only; no collectives.  ``m0``/``n0``
    are Python ints (FFI attributes) or traced ints, e.g. a scanned pass's row offset (the
    handler's optional ``origin`` operand; the same sum, bit for bit).
    """
    if any(a.dtype != jnp.complex128 for a in (accumulator, contribution, projection)):
        raise TypeError("contour block accumulator requires complex128 operands")
    if _host(mesh):
        # One index dtype: a traced int32 offset beside x64 Python ints is a TypeError.
        start = tuple(jnp.asarray(v, jnp.int32) for v in (0, 0, m0, n0))
        bm, bn = contribution.shape[2:]
        block = jax.lax.dynamic_slice(accumulator, start, (*accumulator.shape[:2], bm, bn))
        new = block
        for s in range(contribution.shape[0]):
            new = new + projection[s][:, None, None, None] * contribution[s][None]
        live = ((jnp.arange(bm)[:, None] < valid[0])
                & (jnp.arange(bn)[None, :] < valid[1]))
        return jax.lax.dynamic_update_slice(accumulator, jnp.where(live, new, block), start)
    _require(BLOCK_TARGET)
    static = all(isinstance(v, (int, np.integer)) for v in (m0, n0))
    origin = () if static else (jnp.stack([jnp.asarray(m0, jnp.int32), jnp.asarray(n0, jnp.int32)]),)
    return jax.ffi.ffi_call(
        BLOCK_TARGET, jax.ShapeDtypeStruct(accumulator.shape, accumulator.dtype),
        input_output_aliases={0: 0}, vmap_method="sequential",
    )(accumulator, contribution, projection, valid.astype(jnp.int32), *origin,
      m0=np.int64(m0 if static else 0), n0=np.int64(n0 if static else 0))
