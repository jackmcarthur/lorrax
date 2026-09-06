"""Reuse the pinned restore callables across terms without changing their bodies."""
import inspect
from gw import w_isdf

source = inspect.getsource(w_isdf.photon_blocks_full_q)
old = '        acc = jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=spec)()'
new = '''        zero = jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=spec)
        acc = _ablation_zeros.setdefault((id(response), A, B), zero)()'''
assert source.count(old) == 1
source = source.replace(old, new)
old = '                acc = add(acc, source)'
new = '''                add = _ablation_adds.setdefault((id(response), A, B, C, D), add)
                acc = add(acc, source)'''
assert source.count(old) == 1
source = source.replace(old, new)
namespace = dict(vars(w_isdf), _ablation_zeros={}, _ablation_adds={})
exec(compile(source, __file__, 'exec'), namespace)
w_isdf.photon_blocks_full_q = namespace['photon_blocks_full_q']
# The common instrument now observes the ablated generator.
import profile_driver
