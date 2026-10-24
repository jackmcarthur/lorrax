"""Raw value/slope records follow the producer's shared sample consumption order.

P4 standalone: mixed dense/line members, resume masks and unequal q-span
counts. An independent sample-coded oracle rejects the original reader order.
"""
def main():
    import argparse,json
    from pathlib import Path
    import runtime
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    import gw.response_bank as owner
    mesh=runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local
    from file_io.slab_io import StreamedBank
    args.out.mkdir(parents=True,exist_ok=True)
    members=[0,3,1,4,2,5];committed={1,4};nq=9
    order,fresh=owner._sample_members(members,(0,3),committed.__contains__)
    expected_samples=[3,5,0,2]
    assert [sample for _,sample in fresh]==expected_samples
    assert [members[row] for row,_ in fresh]==expected_samples
    assert owner._sample_members(members,(0,3),lambda _:True)[1]==[]
    # Dense3 has two already-separated uncommitted windows; line members read
    # the whole q bank. These literal spans are independent of the planner.
    expected_spans={3:[(0,2),(2,3),(6,8),(8,9)],5:[(1,3),(3,5)],0:[(0,9)],2:[(0,9)]}
    def spans(sample):
        if sample<3:return [None]
        windows={3:[(0,3),(6,9)],5:[(1,5)]}[sample]
        return [(a,min(hi,a+2)) for lo,hi in windows for a in range(lo,hi,2)]
    def requests(sequence):return [(2*row,2*row+2,span) for row,sample in sequence for span in spans(sample)]
    def values(sample,fields,parents):
        f=np.asarray(fields)[:,None,None,None];q=np.asarray(parents)[None,:,None,None]
        r=np.arange(4)[None,None,:,None];c=np.arange(4)[None,None,None,:]
        return (sample*2**20+f*2**16+q*128+r*4+c/4
                +1j*(-sample*2**18+f*2**12-q*8+r/16-c/128)).astype(complex)
    raw=np.concatenate([values(sample,(0,1),range(nq)) for sample in members])
    carry=device_put_process_local(raw,NamedSharding(mesh,P(None,None,'x','y')))
    bank=StreamedBank(mesh,root=str(args.out),label='mixed_members',kind='file',
        n_out=2*len(members),q=nq,segments=((2,2,((0,0,0,0,2,2),)),),tile=(2,2),unwritten_zero=True)
    assert bank.fits
    def check(sequence):
        reads=bank.reader(requests(sequence));seen=0
        try:
            for sample in expected_samples:
                for lo,hi in expected_spans[sample]:
                    got=next(reads)
                    assert got.shape==(2,hi-lo,4,4),(sample,got.shape,(lo,hi))
                    index=next(iter(got.sharding.addressable_devices_indices_map(got.shape).values()))
                    np.testing.assert_array_equal(np.asarray(got.addressable_data(0)),values(sample,(0,1),range(lo,hi))[index])
                    seen+=1
            try:next(reads)
            except StopIteration:pass
            else:raise AssertionError('unexpected raw member left')
        finally:reads.close()
        return seen
    try:
        # Original raw carry rows are preserved; the consumer order never
        # renumbers either a value/slope pair or a resumed sample's slots.
        bank.put(0,carry,[(2*row+k,2*row+k) for row,_ in sorted(fresh) for k in (0,1)])
        bank.commit();seen=check(fresh)
        rejected=False
        try:check([(row,sample) for row,sample in enumerate(members) if sample not in committed])
        except (AssertionError,ValueError):rejected=True
        assert rejected,'original-order red twin must fail the independent sample oracle'
        result=dict(verdict='PASS',fresh=fresh,consumer=order,spans_checked=seen,
            committed=sorted(committed),original_order_red_twin_rejected=rejected,receipt=bank.receipt())
    finally:bank.release()
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESPONSE_MEMBER_ORDER PASS',jax.process_index(),flush=True)

if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
