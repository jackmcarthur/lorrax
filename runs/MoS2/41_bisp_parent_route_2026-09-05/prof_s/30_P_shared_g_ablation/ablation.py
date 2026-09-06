"""Feed body and head convolutions from one unchanged per-block Green function."""
from gw import gw_jax
import inspect
from pathlib import Path
from gw import photon_sigma

base = Path(__file__).resolve().parent.parent
plan_patch = (base/'17_P_plan_reuse_ablation/ablation.py').read_text().split('import profile_driver')[0]
exec(compile(plan_patch, __file__, 'exec'))
# The plan ablation leaves its transplanted factory text in source.
source = source.replace('q0_only=False,', 'with_head=False,')
source = source.replace('vertex_pair, q0_only)', 'vertex_pair, with_head)')
source = source.replace('q0_only=q0_only)', 'q0_only=False)\n    head_convolve = (_make_static_convolution(mesh_xy, kgrid, nk_tot, q0_only=True)\n                     if with_head else None)')
source = source.replace('interaction, factor):', 'interaction, factor, head_interaction=None):')
old = '        return project(left.psi_nmu, jnp.take(sigma, jnp.asarray(rows), axis=0), right.psi_mun)'
new = '''        result = project(left.psi_nmu, jnp.take(sigma, jnp.asarray(rows), axis=0), right.psi_mun)
        if with_head:
            head_sigma = head_convolve(G, head_interaction, factor)
            head = project(left.psi_nmu, jnp.take(head_sigma, jnp.asarray(rows), axis=0), right.psi_mun)
            return result, head
        return result'''
assert old in source
source = source.replace(old, new)
exec(compile(source, __file__, 'exec'), namespace)
photon_sigma._make_photon_static_block_kernel = namespace['_make_photon_static_block_kernel']
source = inspect.getsource(photon_sigma.contract_lorentz_blocks)
start = source.index('        kernel = _make_photon_static_block_kernel(')
stop = source.index('        yield key, result, head', start)
replacement = '''        head_block = None
        if factors is not None:
            pairs = (factors.bare_pair,) if term == _TERM_X else factors.screened_pairs
            head_block = photon_q0_low_rank_block(pairs, response.layout, A, B, mesh_xy)
            if term == _TERM_COH:
                head_block = head_block - photon_q0_low_rank_block(
                    (factors.bare_pair,), response.layout, A, B, mesh_xy)
        kernel = _make_photon_static_block_kernel(mesh_xy, meta.kgrid, meta.nk_tot,
            left, right, vertex_pair=key, with_head=factors is not None)
        value = kernel(left.green_parent, right.green_parent, weights, interaction, factor, head_block)
        result, head = value if factors is not None else (value, None)
        jax.block_until_ready((result, head))
        del interaction, head_block
'''
source = source[:start]+replacement+source[stop:]
exec(compile(source, __file__, 'exec'), vars(photon_sigma))
restore_patch = (base/'18_P_restore_shape_ablation/ablation.py').read_text().split('import profile_driver')[0]
exec(compile(restore_patch, __file__, 'exec'))
import profile_driver
