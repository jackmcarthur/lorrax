"""Nonidentity packed current publication with exact typed centroid metadata.

Synthetic OWN3 arrays define packed labels; all physical and matrix host
references use the literal independent canonical-to-packed permutation.
No mu ghosts are present. This is not a source152/Qirr GW restart proof.
"""
from pathlib import Path
from types import SimpleNamespace
import json


def layout_basis(runtime, coordinate_kind):
    """Build four half-cell translation orbits with a known independent map."""
    import numpy as np
    from common.centroid_basis import PackedCentroidBasis
    if coordinate_kind=='fft_indices':
        first=np.asarray([[0,0,0],[0,1,2],[1,2,3],[1,3,4]],np.int32)
        points=np.concatenate((first,first+np.asarray([2,0,0],np.int32)))
    elif coordinate_kind=='fractional':
        first=np.asarray([[.11,.17,.23],[.19,.31,.43],[.27,.41,.59],[.33,.53,.71]],np.float64)
        points=np.concatenate((first,first+np.asarray([.5,0.,0.])))
        assert np.max(abs(points*np.asarray([4,4,6])-np.rint(points*np.asarray([4,4,6]))))>.1
    else:
        raise ValueError('unsupported fixture coordinate kind')
    # SymMaps carries raw BGW tnp=2pi*tau; the public centroid map divides it.
    sym=SimpleNamespace(sym_matrices=np.tile(np.eye(3,dtype=np.int32),(2,1,1)),
                        translations=2*np.pi*np.asarray([[0.,0.,0.],[.5,0.,0.]]))
    basis=PackedCentroidBasis.build(points,sym,(4,4,6),runtime.mesh,
                                   coordinate_kind=coordinate_kind)
    # Four equal two-point orbits assigned by the existing LPT owner.
    literal=np.asarray([0,4,2,6,1,5,3,7],np.int32)
    assert not basis.is_identity
    assert np.array_equal(basis.layout.axis.canonical_to_packed,literal)
    assert np.array_equal(basis.layout.axis.packed_to_canonical,literal)
    assert basis.n_logical==basis.n_packed==8 and basis.active_mask.all()
    return basis,literal


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        from tests.test_atomic_breit_p4 import check_breit_group
        from tests.test_current_physical_publication_p4 import publication_fixture
        root=Path(args.output);output={}
        for kind in ('fft_indices','fractional'):
            basis,literal=layout_basis(runtime,kind)
            for circular in (False,True):
                name=kind+('_circular' if circular else '_cartesian')
                fixture,evidence=publication_fixture(runtime,root/name,mu_basis=basis)
                provider=check_breit_group(runtime,circular=circular,compensated=True,
                    averaged_heads=True,embedded_gamma=True,publication_fixture=fixture,
                    mu_basis=basis,canonical_to_packed_reference=literal)
                output[name]=dict(provider=provider,publication=evidence,
                    canonical_to_packed_reference=literal.tolist(),
                    canonical_coordinates=basis.canonical_indices.tolist(),
                    coordinate_kind=kind,mu_ghosts_present=False)
        if jax.process_index()==0:
            (root/'receipt.json').write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps(output),flush=True)
        return 0
    run_main_and_finalize(main)
