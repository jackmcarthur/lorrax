"""P4 physical OWN3 publication and saved-tensor action replay.

This reuses the independently referenced static provider proof. It certifies
the collective physical payload seam, not a full production GW restart or an
AgI source152 Breit precision result. Public admission guards remain enabled.
"""
from pathlib import Path
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import json


def _synthetic_mf_source(path, facts):
    """Small complete MF header; no wavefunction or atomic field is invented."""
    import h5py
    import numpy as np
    fixture=facts['head_fixture'];q=fixture['q'];Q=len(q)
    volume=fixture['volume'];grid=np.asarray([4,4,6],np.int32)
    data={
        'versionnumber':1,'flavor':2,
        'kpoints/nspin':1,'kpoints/nspinor':4,'kpoints/nrk':Q,
        'kpoints/mnband':8,'kpoints/ngkmax':7,'kpoints/ecutwfc':2.,
        'kpoints/kgrid':np.asarray([6,6,6],np.int32),'kpoints/shift':np.zeros(3),
        'kpoints/ngk':fixture['counts'],'kpoints/ifmin':np.ones((1,Q),np.int32),
        'kpoints/ifmax':np.full((1,Q),4,np.int32),'kpoints/w':np.full(Q,1/Q),
        'kpoints/rk':q,'kpoints/el':np.zeros((1,Q,8)),'kpoints/occ':np.zeros((1,Q,8)),
        'gspace/ng':7,'gspace/ecutrho':4.,'gspace/FFTgrid':grid,
        'symmetry/ntran':1,'symmetry/cell_symmetry':1,
        'symmetry/mtrx':np.eye(3,dtype=np.int32)[None],'symmetry/tnp':np.zeros((1,3)),
        'crystal/celvol':volume,'crystal/recvol':(2*np.pi)**3/volume,
        'crystal/alat':10.,'crystal/blat':2*np.pi/10,'crystal/nat':1,
        'crystal/avec':np.eye(3),'crystal/bvec':np.eye(3),
        'crystal/adot':100*np.eye(3),'crystal/bdot':(2*np.pi/10)**2*np.eye(3),
        'crystal/atyp':np.asarray([1],np.int32),'crystal/apos':np.zeros((1,3))}
    with h5py.File(path,'w') as f:
        mf=f.create_group('mf_header')
        for name,value in data.items():mf.create_dataset(name,data=value)


def publication_fixture(runtime, directory):
    """Return plain callbacks consuming the admitted provider fixture arrays."""
    import numpy as np
    import jax
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import rank0_transaction,gather_to_host,device_put_process_local
    from file_io.restart_bundle import open_zeta,read_isdf_header
    from file_io.isdf_header import mark_zeta_done,stamp_fit_provenance
    from file_io.slab_io import SlabIO
    from gw.gw_init import _open_current_physical_writers
    result={};mesh=runtime.mesh

    @contextmanager
    def opened(zetas,facts):
        if facts['head_fixture'] is None:
            raise ValueError('publication fixture requires the canonical body/head geometry')
        directory.mkdir(parents=True,exist_ok=True)
        source=directory/'source_header.h5'
        rank0_transaction(str(source),stage='fixture_mf_source',
            write=lambda:_synthetic_mf_source(source,facts))
        # Identity μ order isolates the physical writer; q/G pads are already
        # poisoned by the independent provider proof. Packed μ transport is
        # covered by the existing centroid basis tests, not inferred here.
        points=np.asarray([[i//4,(i//2)%2,i%2] for i in range(8)],np.int32)
        meta=SimpleNamespace(fft_grid=(4,4,6),mu_basis=SimpleNamespace(
            coordinate_kind='fft_indices',canonical_indices=points))
        for mu,z in enumerate(zetas,start=1):z.path=str(directory/f'zeta_q_mu{mu}.h5')
        with ExitStack() as stack:
            writers=_open_current_physical_writers(stack,dict(enumerate(zetas,start=1)),
                wfn=SimpleNamespace(_filename=str(source)),meta=meta,zeta_cutoff=2.,mesh=mesh)
            yield writers
        # All collective handles are closed before any completion stamp.
        for z in zetas:
            assert not read_isdf_header(z.path).zeta_is_done
        def complete():
            for mu,z in enumerate(zetas,start=1):
                mark_zeta_done(z.path)
                stamp_fit_provenance(z.path,json.dumps(dict(
                    synthetic_family=True,vertex_mu_L=mu,
                    current_basis_rows=[[[float(v.real),float(v.imag)] for v in row]
                                        for row in facts['current_basis_rows']],
                    kernel='static_transverse',gamma=facts['head_fixture']['policy']['payload_sha256'])))
        rank0_transaction(zetas[0].path,stage='fixture_current_complete',write=complete)

    def verify(zetas,got,facts):
        file_error=0.;compensation_signal=0.;shell_error=0.
        for z,physical,comp in zip(zetas,facts['physical'],facts['compensation']):
            with open_zeta(z.path,mesh=mesh) as loader:
                assert loader.coordinate_kind=='fft_indices'
                assert read_isdf_header(z.path).zeta_is_done
                data=loader.read_zeta_G_slab(q_offset=0,q_count=3,mu_offset=0,mu_count=8)
                host=np.asarray(gather_to_host(data))
            file_error=max(file_error,float(np.max(abs(host-physical))))
            compensation_signal=max(compensation_signal,float(np.max(abs(host-comp))))
            shell=np.take_along_axis(host,facts['keep'][:,None],axis=2)
            shell_error=max(shell_error,float(np.max(abs(shell-np.asarray(gather_to_host(z.shell))))))
            for q,n in enumerate(z.ngk_per_q):assert not host[q,:,n:].any()
        assert file_error<2e-11 and shell_error<2e-11
        assert compensation_signal>1e-6,'fixture did not distinguish physical delta from compensation'
        host_V=tuple(np.asarray(gather_to_host(v)) for v in got)
        path=directory/'saved_current_tensors.h5'
        # Persist the COMPLETE local-augmented tensors, not a finite-G
        # contraction reconstructed from the physical zeta files.
        with SlabIO(path,mode='w',mesh=mesh) as writer:
            for (a,b),v in zip(facts['pairs'],got):
                name=f'V_{a}{b}';writer.create_dataset(name,shape=v.shape,dtype=np.complex128)
                writer.write_slab(name,v);writer.sync_writes()
        restored=[]
        with SlabIO(path,mode='r',mesh=mesh) as reader:
            for a,b in facts['pairs']:
                v=reader.read_slab(f'V_{a}{b}',shape=got[0].shape,dtype=np.complex128,
                    offset=(0,0,0),partition_spec=P(None,'x','y'))
                restored.append(np.asarray(gather_to_host(v)))
        for old,new,expected in zip(host_V,restored,facts['expected']):
            np.testing.assert_array_equal(new,old)
            np.testing.assert_allclose(new,expected,rtol=0,atol=2e-11)
        rng=np.random.default_rng(3821)
        samples=rng.normal(size=(3,3,4,8))+1j*rng.normal(size=(3,3,4,8))
        def exchange_action(tiles,*,flip_gamma2=False):
            blocks={p:t for p,t in zip(facts['pairs'],tiles)}
            for (a,b),v in tuple(blocks.items()):
                if a!=b:blocks[b,a]=v.conj().transpose(0,2,1)
            total=np.zeros((4,4),complex)
            for a in range(3):
                for b in range(3):
                    sign=(-1 if (a==1)!=(b==1) else 1) if flip_gamma2 else 1
                    total-=sign*np.einsum('qmj,qjk,qnk->mn',samples[a].conj(),blocks[a,b],samples[b])/3
            return total
        action=exchange_action(host_V);again=exchange_action(restored)
        np.testing.assert_array_equal(action,again)
        sign_signal=float(np.max(abs(action-exchange_action(restored,flip_gamma2=True))))
        assert sign_signal>1e-5,'nine-block witness missed a one-sided gamma2 sign error'
        result.update(physical_file_error=file_error,physical_shell_file_error=shell_error,
            physical_vs_compensation_signal=compensation_signal,
            saved_tensor_exact=True,nine_block_exchange_replay_exact=True,
            one_sided_gamma2_sign_signal=sign_signal,
            scope='synthetic provider physical payload and complete saved-tensor action replay; not Qirr-stamped production GW restart')
    return dict(open=opened,verify=verify),result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def main():
        import jax
        from tests.test_atomic_breit_p4 import check_breit_group
        from tests.test_mixed_zeta_contract import check_mixed_contract
        root=Path(args.output)
        output={}
        for circular in (False,True):
            name='circular' if circular else 'cartesian'
            fixture,evidence=publication_fixture(runtime,root/name)
            provider=check_breit_group(runtime,circular=circular,compensated=True,
                averaged_heads=True,embedded_gamma=True,publication_fixture=fixture)
            output[name]=dict(provider=provider,publication=evidence)
        check_mixed_contract(runtime)  # Incumbent scalar + grouped scalar refusal.
        if jax.process_index()==0:
            (root/'receipt.json').write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps(output),flush=True)
        return 0
    run_main_and_finalize(main)
