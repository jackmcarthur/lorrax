"""Q<P grouped current contraction through canonical physical publication.

The original Breit fixture remains the sole independent radial reference.
This small seam test adds internal orbit ghosts, real public AOT pricing
and the public current-tensor reader; it is not a material accuracy test.
Run with ``python -m tests.test_current_onsite_packing_p4 --output DIR``.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json


def check_packed_current_publication(runtime,directory):
    import numpy as np
    from common.centroid_basis import PackedCentroidBasis
    from gw.isdf_augmentation import _price_current_contraction_aot
    from symmetry_maps import (QirrTables,centroid_source_map_and_wrap,
        verify_centroid_orbit_closure)
    from tests.test_atomic_breit_p4 import check_breit_group
    from tests.test_current_physical_publication_p4 import publication_fixture
    from tests.test_static_breit_body_heads import physical_head_fixture

    mesh=runtime.mesh
    if dict(mesh.shape)!={'x':2,'y':2} or runtime.process_count!=4:
        raise ValueError('This regression requires actual P4')
    points=np.asarray([[1,1,2],[1,2,3],[0,3,4],[2,0,1],[3,1,2],[3,2,3]],np.int32)
    grid=np.asarray([4,4,6],np.int32)
    operations=np.asarray([np.eye(3,dtype=np.int32),np.diag([-1,1,1])])
    translations=np.zeros((2,3));kgrid=(3,1,1)
    qfrac=np.asarray([[0,0,0],[1/3,0,0],[-1/3,0,0]])
    sym=SimpleNamespace(sym_matrices=operations,translations=translations,
        irr_idx_q=np.arange(3,dtype=np.int32),sym_idx_q=np.zeros(3,np.int32),
        q_irr_full_idx=np.arange(3,dtype=np.int32),
        q_irr_kgrid_int=np.asarray([[0,0,0],[1,0,0],[-1,0,0]],np.int32),
        trs_allowed=False,active_symmetry_rows=np.asarray([0,1],np.int32))
    basis=PackedCentroidBasis.build(points,sym,grid,mesh,coordinate_kind='fft_indices')
    literal=np.asarray([0,4,2,6,1,5],np.int32)
    assert (basis.n_logical,basis.n_packed,basis.n_canonical)==(6,8,8)
    np.testing.assert_array_equal(basis.layout.axis.canonical_to_packed,literal)
    np.testing.assert_array_equal(np.flatnonzero(~basis.active_mask),[3,7])
    perm,wraps=centroid_source_map_and_wrap(points,operations,translations,grid,
        coordinate_kind='fft_indices',extend_trs=True,validate=False)
    tables=QirrTables(irr_idx_q=sym.irr_idx_q,sym_idx_q=sym.sym_idx_q,
        q_irr_frac=qfrac,sym_perm=perm,L_table=wraps,n_sym_spatial=2)
    pperm,pwraps=basis.pack_tables(perm,wraps)
    plan=SimpleNamespace(sym=sym,n_sym_spatial=2,fft_grid=grid,
        spatial_ops=operations,translations=translations,sym_perm=pperm,L_table=pwraps)
    closure=verify_centroid_orbit_closure(points/grid,operations,tnp=translations)
    fixture,publication=publication_fixture(runtime,Path(directory),mu_basis=basis,
        operator_layout=dict(tables=tables,closure=closure,plan=plan,kgrid=kgrid))
    pricing={}

    def head_factory(**kwargs):
        return physical_head_fixture(q_rows=qfrac,kgrid=kgrid,**kwargs)

    def price(zetas,group,provider_args,K):
        state={key:provider_args[key] for key in ('radius','lm','centers_cart',
            'cell_volume','fft_points','support_radius','minimum_atom_image_distance','current_basis_rows')}
        state['deferred_rhs']=SimpleNamespace(source={},coefficients=())
        state['fixture_live_rhs']=tuple(z.local_augmentation['rhs'] for z in zetas)
        controls={key:provider_args[key] for key in ('field_tile','centroid_tile','fourier_points')}
        priced=_price_current_contraction_aot(tuple(zetas),state,K,controls)
        owners=[record['owner'] for record in priced['quotes']]
        assert 'onsite_face' in owners and 'onsite' not in owners
        pricing.update(priced)

    provider=check_breit_group(runtime,circular=False,averaged_heads=True,
        embedded_gamma=True,publication_fixture=fixture,mu_basis=basis,
        canonical_to_packed_reference=literal,price_observer=price,
        head_fixture_factory=head_factory)
    return dict(provider=provider,publication=publication,price_owner=pricing,
        q_logical=3,q_carrier=4,canonical_file_mu=6,packed_mu=8,
        canonical_to_packed_literal=literal.tolist(),internal_ghosts=[3,7],
        scope='Synthetic grouped contraction/publication/packing/pricing; no material accuracy, GW or speedup claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        result=check_packed_current_publication(runtime,output)
        if jax.process_index()==0:
            result['status']='PASS_Q_LT_P_GROUPED_CURRENT_CANONICAL_IO_AND_PRICING'
            (output/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
            print(json.dumps(result),flush=True)
        return 0
    run_main_and_finalize(main)
