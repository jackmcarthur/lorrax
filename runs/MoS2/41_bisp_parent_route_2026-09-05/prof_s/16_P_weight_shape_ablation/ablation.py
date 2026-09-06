"""Hold occupied and complete-band weights at the same full-k shape."""
import inspect
from gw import photon_sigma
source = inspect.getsource(photon_sigma.contract_lorentz_blocks)
old = '        factor = -0.5 if term == _TERM_COH else 1.0'
assert source.count(old) == 1
source = source.replace(old, '        weights = jnp.broadcast_to(weights, (meta.nk_tot, slices.nb_full))\n'+old)
namespace = vars(photon_sigma)
exec(compile(source, __file__, 'exec'), namespace)
photon_sigma.contract_lorentz_blocks = namespace['contract_lorentz_blocks']
import profile_driver
# Use the instrumented factory in the transplanted function's globals.
