"""P4 exact-value raw-chi file-tier transport at Fe's 80/59/99 topology.

The optional offset control only runs where the owner's own tiny probe
reports sparse files. It neither waives capacity nor claims a capacity fit.
"""


def main():
    import argparse,json,os
    from pathlib import Path
    import runtime
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--offset',action='store_true')
    args=parser.parse_args()
    mesh=runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host,all_gather_processes
    from common.gpu_utils import device_budget_bytes
    from file_io.slab_io import StreamedBank
    assert mesh.size==4
    device_budget_bytes();args.out.mkdir(parents=True,exist_ok=True)
    face=NamedSharding(mesh,P(None,None,'x','y'))
    px,py=int(mesh.shape['x']),int(mesh.shape['y'])
    nout,nq,nseg=80,59,99

    def values(segment,outputs,parents,rows,cols):
        o=np.asarray(outputs)[:,None,None,None]
        q=np.asarray(parents)[None,:,None,None]
        r=np.asarray(rows)[None,None,:,None]
        c=np.asarray(cols)[None,None,None,:]
        return (segment*2**24+o*2**16+q*128+r*8+c/4
                +1j*(-(o+1)*2**12+segment*16-q/8+r/32-c/256)).astype(complex)

    r,c=2,2
    if args.offset:
        tiny=StreamedBank(mesh,root=str(args.out),label='sparse_probe',kind='file',
            n_out=1,q=1,segments=((1,1,((0,0,0,0,1,1),)),),tile=(1,1),unwritten_zero=True)
        assert tiny.fits
        sparse=not any(s.reserved for s in tiny.stores.values())
        sparse=bool(np.all(all_gather_processes(np.asarray(sparse,np.bool_))))
        tiny.release()
        if not sparse:
            result=dict(verdict='SKIP',reason='owner filesystem probe reserves full files; no sparse bypass')
            (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result,indent=2)+'\n')
            print('CHI_OFFSET SKIP',jax.process_index(),flush=True)
            return
        c=512
    segments=tuple((r,c,((0,0,s*r,0,r,c),)) for s in range(nseg))
    bank=StreamedBank(mesh,root=str(args.out),label='raw_chi',kind='file',n_out=nout,q=nq,
        segments=segments,tile=(nseg*r,c),unwritten_zero=bool(args.offset))
    assert bank.fits and bank.kind=='file'
    held=[]
    try:
        if args.offset:
            assert not any(s.reserved for s in bank.stores.values())
            outputs=(23,46,79);segment=98
            carry=device_put_process_local(values(segment,outputs,range(nq),range(px*r),range(py*c)),face)
            bank.put(segment,carry,tuple(enumerate(outputs)));del carry
            bank.commit()
            offsets=[o*bank.S+bank.starts[segment] for o in outputs]
            assert offsets[0]>2**31 and offsets[1]>2**32
            for o in outputs:
                value=next(bank.reader([(o,o+1)]))
                tile=np.asarray(value.addressable_data(0))
                dev=next(iter(value.sharding.addressable_devices_indices_map(value.shape)))
                sl=value.sharding.addressable_devices_indices_map(value.shape)[dev]
                x=(sl[-2].start or 0)//(nseg*r);y=(sl[-1].start or 0)//c
                expected=values(segment,[o],range(nq),range(x*r,(x+1)*r),range(y*c,(y+1)*c))
                assert np.all(tile[:,:,:segment*r]==0)
                np.testing.assert_array_equal(tile[:,:,segment*r:(segment+1)*r],expected)
            footprint=max(os.fstat(s.fd).st_blocks*512 for s in bank.stores.values())
            assert footprint<256*2**20
            result=dict(verdict='PASS',scope='offset correctness only; naturally sparse owner',
                logical_file_bytes=bank.nbytes,physical_file_bytes=footprint,offsets=offsets,receipt=bank.receipt())
        else:
            for segment in range(nseg):
                carry=device_put_process_local(values(segment,range(nout),range(nq),range(px*r),range(py*c)),face)
                bank.put(segment,carry,tuple((i,i) for i in range(nout)))
                del carry
            bank.commit()
            spans=[(0,2),(18,20,(0,5)),(40,42,(54,59)),(78,80,(0,1)),(0,2)]
            for span,value in zip(spans,bank.reader(spans)):
                # Keep every returned array while later reads reuse staging.
                held.append((span,value))
            for span,value in held:
                first,last=span[:2];qa,qb=(0,nq) if len(span)==2 else span[2]
                rows=np.arange(px*nseg*r)
                x=rows//(nseg*r);local=rows%(nseg*r)
                expected=values((local//r)[None,None,:,None],range(first,last),range(qa,qb),
                                x*r+local%r,range(py*c))
                np.testing.assert_array_equal(np.asarray(gather_to_host(value)),expected)
            result=dict(verdict='PASS',scope='actual raw-chi file-tier owner topology',receipt=bank.receipt(),
                held_read_arrays=len(held),all_exact=True)
    finally:
        bank.release()
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CHI_FILE_TIER PASS',jax.process_index(),flush=True)


if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
