"""Share the left child projector across the three coupled current channels."""
import inspect
from isdf import core
from gw import isdf_fitting

source = inspect.getsource(core._z_q_face_parent)
old = '''\t\t\t\t_, channels = jax.lax.scan(
\t\t\t\t\tlambda carry, vertex: (carry, channel_tail(*vertex)), 0,
\t\t\t\t\t(perms, phases, perms, phases), unroll=1)
\t\t\t\treturn channels'''
new = '''\t\t\t\toutput_pairs = (perms[:, :, None] * ns + perms[:, None, :]).reshape(3, -1)
\t\t\t\toutput_phases = (phases[:, :, None] * phases[:, None, :]).reshape(3, -1)
\t\t\t\tsources = jnp.asarray(source_pairs)
\t\t\t\tcounts = jnp.asarray(source_counts)

\t\t\t\tdef spin_channels(acc, pair):
\t\t\t\t\tleft = local_ifftn3(unfold_block(
\t\t\t\t\t\tD_l_, coefficients[pair], sources[pair], counts[pair]).reshape(
\t\t\t\t\t\t\tnkx, nky, nkz, mu_loc, r_loc), axes=(0, 1, 2), norm='forward')
\t\t\t\t\tdef channel(carry, args):
\t\t\t\t\t\tvalue, target, phase = args
\t\t\t\t\t\tright = local_ifftn3(unfold_block(
\t\t\t\t\t\t\tD_r_, coefficients[target], sources[target], counts[target]).reshape(
\t\t\t\t\t\t\t\tnkx, nky, nkz, mu_loc, r_loc), axes=(0, 1, 2), norm='forward')
\t\t\t\t\t\treturn carry, value + jnp.conj(left) * (phase * right)
\t\t\t\t\t_, result = jax.lax.scan(channel, 0,
\t\t\t\t\t\t(acc, output_pairs[:, pair], output_phases[:, pair]), unroll=1)
\t\t\t\t\treturn result, None

\t\t\t\tinitial = jnp.zeros((3, nkx, nky, nkz, mu_loc, r_loc), dtype=jnp.complex128)
\t\t\t\tresult, _ = jax.lax.scan(spin_channels, initial, jnp.arange(ns * ns), unroll=1)
\t\t\t\t_, channels = jax.lax.scan(lambda carry, value: (carry,
\t\t\t\t\tlocal_fftn3(value, axes=(0, 1, 2), norm='forward').reshape(nk, mu_loc, r_loc)),
\t\t\t\t\t0, result, unroll=1)
\t\t\t\treturn channels'''
assert source.count(old) == 1
source = source.replace(old, new)
# Separate factory cache is essential: sharing the baseline cache would test its executable twice.
namespace = dict(core.__dict__)
namespace['_pair_pipeline_sm_cache'] = {}
exec(compile(source, __file__, 'exec'), namespace)
isdf_fitting._z_q_face_parent = namespace['_z_q_face_parent']
