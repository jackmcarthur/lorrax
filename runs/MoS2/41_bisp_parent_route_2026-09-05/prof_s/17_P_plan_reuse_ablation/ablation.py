"""Reuse Sigma GEMM/projector plans by endpoint shapes across vertex and head variants."""
import inspect
from gw import photon_sigma
source = inspect.getsource(photon_sigma._make_photon_static_block_kernel)
start = source.index('    project = contract_bands_block_reshard(')
stop = source.index('    convolve = _make_static_convolution', start)
old = source[start:stop]
new = '''    plan_key = (id(mesh_xy), tuple(kgrid), shapes, ffi_dial_key())
    if plan_key not in _ablation_plans:
'''+''.join('    '+line+'\n' for line in old.splitlines())+'''        _ablation_plans[plan_key] = project, g_plan
    project, g_plan = _ablation_plans[plan_key]
'''
source = source[:start]+new+source[stop:]
namespace = dict(vars(photon_sigma), _ablation_plans={})
exec(compile(source, __file__, 'exec'), namespace)
photon_sigma._make_photon_static_block_kernel = namespace['_make_photon_static_block_kernel']
import profile_driver
