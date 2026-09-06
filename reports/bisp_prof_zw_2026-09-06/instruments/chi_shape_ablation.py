"""Ablate per-vertex specialization while retaining the production chi body."""
import inspect
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from gw import w_isdf
from common.gamma_matrices import gamma_perm_phase

source = inspect.getsource(w_isdf._get_chi_minimax_kernel_face)
start = source.index('    if vertex_pair:\n        from common.gamma_matrices import gamma_perm_phase')
end = source.index('    @partial(jax.jit, in_shardings=_base_in', start+1)
# Replace only the vertex-specific final binding, not Green/FFT/gamma arithmetic.
end = source.index('    @partial(jax.jit, in_shardings=_base_in', source.index('        return minimax_tau_integrate_chi_vertex', start))
source = source[:start] + '''    if vertex_pair:
        vertex_shard = tuple(NamedSharding(mesh_xy, P()) for _ in range(4))
        @partial(jax.jit, in_shardings=(*_base_in, vertex_shard), out_shardings=_chi_R_shard)
        def minimax_tau_integrate_chi_vertex(
            nodes, psi_mun, psi_nmu, mask_v, mask_c, enk_full, vmax, cmin, vertex_operands,
        ):
            return _single_impl(nodes, psi_mun, psi_nmu, mask_v, mask_c,
                                enk_full, vmax, cmin, vertex_operands)
        return minimax_tau_integrate_chi_vertex

''' + source[end:]
exec(compile(source, __file__, 'exec'), w_isdf.__dict__)
original_get = w_isdf._get_chi_minimax_kernel


def get_kernel(*args, **kwargs):
    """Reuse the original cache by the identity/nonidentity vertex shape class."""
    pair = kwargs.get('vertex_pair')
    if pair is None:
        return original_get(*args, **kwargs)
    kwargs['vertex_pair'] = tuple(0 if v == 0 else 1 for v in pair)
    kernel = original_get(*args, **kwargs)
    pl, fl = gamma_perm_phase(pair[0])
    pr, fr = gamma_perm_phase(pair[1])
    operands = (pl, jax.numpy.conj(fl), pr, jax.numpy.conj(fr))
    return lambda *operands_in: kernel(*operands_in, operands)


w_isdf._get_chi_minimax_kernel = get_kernel
