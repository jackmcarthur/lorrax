"""Opt-in shared-pole W carrier with immutable factors distributed over q.

This measurement prototype uses the existing staged face/batch exchanges.
It changes their lifetime, not the pole sum or the local GEMM contraction.
The caller explicitly selects this carrier and announces the selection.
"""
from functools import partial
import sys

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from distrib_la._batch_reshard import _batch_to_face, _face_to_batch
from distrib_la._shard_map import shard_map


def shared_w_carrier(mesh, nq):
    """Return compiled factor preparation and W(tau) synthesis functions.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        Named x/y mesh. Factors enter at P(None, 'x', 'y').
    nq : int
        Physical parent count. Only q is padded, to the mesh size.

    Returns
    -------
    prepare, synthesize : callable
        prepare(left[q,n,K], right[q,K,n]) distributes complete matrices
        over q at P(('x', 'y'), None, None), without replication. Call once
        per window, retain both outputs until its final tau has completed.
        synthesize(left, right, poles[q,K], owner[q,K], e_ref, tau) evaluates
        left diag(owner exp(-i (poles-e_ref) tau)) right, returning the
        original q/n/n face layout. Energies are Ry and tau inverse Ry.
        Poles and masks are small replicated arrays, sliced locally by q.
    """
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    if jax.process_index() == 0:
        print('ANNOUNCED OPT-IN shared W carrier: immutable factors '
              'distributed over q once per window; local pole phases; '
              'existing staged output exchanges', file=sys.stderr, flush=True)
    pad = (-nq) % (px * py)
    face = P(None, 'x', 'y')
    batch = P(('x', 'y'), None, None)

    @jax.jit
    @partial(shard_map, mesh=mesh, in_specs=(face, face),
             out_specs=(batch, batch), check_vma=False)
    def prepare(left, right):
        if left.shape[0] != nq or right.shape[0] != nq:
            raise ValueError('shared W carrier parent count changed')
        return tuple(_face_to_batch(
            jnp.pad(a, ((0, pad), (0, 0), (0, 0))), px=px, py=py)
            for a in (left, right))

    @jax.jit
    @partial(shard_map, mesh=mesh,
             in_specs=(batch, batch, P(), P(), P(), P()),
             out_specs=face, check_vma=False)
    def synthesize(left, right, poles, owner, e_ref, tau):
        first = (jax.lax.axis_index('x') * py
                 + jax.lax.axis_index('y')) * left.shape[0]
        poles = jax.lax.dynamic_slice_in_dim(
            jnp.pad(poles, ((0, pad), (0, 0))), first, left.shape[0])
        owner = jax.lax.dynamic_slice_in_dim(
            jnp.pad(owner, ((0, pad), (0, 0))), first, left.shape[0])
        phase = jnp.where(owner, jnp.exp(-1j * (poles - e_ref) * tau),
                          jnp.asarray(0.0 + 0.0j, jnp.complex128))
        w = jnp.matmul(left * phase[:, None, :], right)
        return _batch_to_face(w, px=px, py=py)[:nq]

    return prepare, synthesize
