"""Ablate fresh restore executables without changing the service-owned action."""
from functools import partial
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np


@partial(jax.jit, donate_argnums=(0,), static_argnames=(
    'left', 'right', 'unfold_rows', 'kgrid', 'n_sym_spatial', 'vertices'))
def add_photon_block(accumulator, parent, *, left, right, unfold_rows,
                     kgrid, n_sym_spatial, vertices):
    """Accumulate one typed centroid transport and Lorentz source contribution."""
    from symmetry_maps import unfold_isdf_operator, mix_lorentz_blocks
    from symmetry_maps import bgw_integer_q_to_fractional
    A, B, C, D = vertices
    sym, mesh = left.sym, left.mesh_xy
    sym_idx = np.asarray(unfold_rows, dtype=np.int32)
    qfrac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, kgrid)
    full = unfold_isdf_operator(
        parent, irr_idx=sym.irr_idx_q, sym_idx=sym_idx,
        sym_perm=left.sym_perm, L_table=left.L_table,
        right_sym_perm=right.sym_perm, right_L_table=right.L_table,
        q_irr_frac=qfrac, mesh_xy=mesh, n_sym_spatial=n_sym_spatial,
        axis_local_sym_perm=left.centroid_local_perm,
        right_axis_local_sym_perm=right.centroid_local_perm)
    mixed = mix_lorentz_blocks(
        {(C, D): full}, sym=sym, sym_idx=sym_idx,
        mesh_xy=mesh, keys=((A, B),))
    return jax.lax.with_sharding_constraint(
        accumulator + mixed[A, B], NamedSharding(mesh, P(None, 'x', 'y')))


def photon_blocks_full_q(response, keys, *, term='W'):
    """Stream the original block order through reusable addition executables."""
    from gw.photon_layout import photon_block_view
    left_C, left_T = response.family_plans
    sym, mesh = left_C.sym, left_C.mesh_xy
    policy = response.qgrid_policy
    if term not in ('V', 'W', 'W-V'):
        raise ValueError(f'Unknown photon operator {term!r}.')
    for A, B in keys:
        left, right = (left_T if A else left_C), (left_T if B else left_C)
        shape = (len(sym.irr_idx_q), left.n_centroid_packed, right.n_centroid_packed)
        spec = NamedSharding(mesh, P(None, 'x', 'y'))
        acc = jax.jit(lambda: jnp.zeros(shape, jnp.complex128), out_shardings=spec)()
        for C in ((1, 2, 3) if A else (0,)):
            for D in ((1, 2, 3) if B else (0,)):
                source = photon_block_view(
                    response.V_packed if term == 'V' else response.W_packed,
                    response.layout, C, D, mesh)
                if term == 'W-V':
                    source = source - photon_block_view(
                        response.V_packed, response.layout, C, D, mesh)
                acc = add_photon_block(
                    acc, source, left=left, right=right,
                    unfold_rows=tuple(int(i) for i in policy.unfold_sym_idx),
                    kgrid=tuple(policy.kgrid), n_sym_spatial=policy.n_sym_spatial,
                    vertices=(A, B, C, D))
                del source
        yield (A, B), acc
        del acc
