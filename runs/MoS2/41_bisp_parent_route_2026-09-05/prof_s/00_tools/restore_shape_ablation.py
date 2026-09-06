"""Compile centroid transport per family and keep Lorentz mixing outside that JIT."""
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from gw import w_isdf
from gw.photon_layout import photon_block_view
from symmetry_maps import unfold_isdf_operator, mix_lorentz_blocks, bgw_integer_q_to_fractional


def restore_kernel(left, right, policy):
    sym, mesh = left.sym, left.mesh_xy
    qfrac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, policy.kgrid)
    @jax.jit
    def restore(parent):
        return unfold_isdf_operator(
            parent, irr_idx=sym.irr_idx_q, sym_idx=policy.unfold_sym_idx,
            sym_perm=left.sym_perm, L_table=left.L_table,
            right_sym_perm=right.sym_perm, right_L_table=right.L_table,
            q_irr_frac=qfrac, mesh_xy=mesh, n_sym_spatial=policy.n_sym_spatial,
            axis_local_sym_perm=left.centroid_local_perm,
            right_axis_local_sym_perm=right.centroid_local_perm)
    return restore


def photon_blocks_full_q(response, keys, *, term='W'):
    left_C, left_T = response.family_plans
    sym, mesh = left_C.sym, left_C.mesh_xy
    policy = response.qgrid_policy
    if term not in ('V', 'W', 'W-V'):
        raise ValueError(f'Unknown photon operator {term!r}.')
    restores = {}
    for A, B in keys:
        left, right = (left_T if A else left_C), (left_T if B else left_C)
        family = (bool(A), bool(B))
        if family not in restores:
            restores[family] = restore_kernel(left, right, policy)
        shape = (len(sym.irr_idx_q), left.n_centroid_packed, right.n_centroid_packed)
        spec = NamedSharding(mesh, P(None, 'x', 'y'))
        acc = jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=spec)()
        for C in ((1, 2, 3) if A else (0,)):
            for D in ((1, 2, 3) if B else (0,)):
                source = photon_block_view(response.V_packed if term == 'V' else response.W_packed,
                                           response.layout, C, D, mesh)
                if term == 'W-V':
                    source = source - photon_block_view(response.V_packed, response.layout, C, D, mesh)
                full = restores[family](source)
                mixed = mix_lorentz_blocks({(C,D): full}, sym=sym,
                    sym_idx=policy.unfold_sym_idx, mesh_xy=mesh, keys=((A,B),))
                acc = acc + mixed[A,B]
                del source, full, mixed
        yield (A,B), acc
        del acc

w_isdf.photon_blocks_full_q = photon_blocks_full_q
import profile_driver
