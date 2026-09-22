"""Batched charge/photon moment algebra and partial-commit orchestration."""
from runtime import initialize_communicator_stack, finalize_process
stack = initialize_communicator_stack(platform='gpu')
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host
from gw import response_bank as bank

mesh = stack.mesh
rng = np.random.default_rng(389)
nq, n = 5, 8
face = NamedSharding(mesh, P(None, 'x', 'y'))
def put(a):
    return jax.make_array_from_callback(a.shape, face, lambda ix: a[ix])
for photon in (False, True):
    h = np.broadcast_to(np.diag(np.linspace(.4,1.,n)).astype(complex),(nq,n,n)).copy()
    if photon:
        h[:,n//2:,n//2:] *= -1
    coefficients = [(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))*.03 for _ in range(4)]
    contact = np.eye(n)[None].astype(complex)*.01
    v = h if photon else h@h
    winf = np.linalg.solve(np.eye(n)[None]+v@(2*contact if photon else np.zeros_like(contact)),v)
    series = [winf]
    for power in range(1,5):
        series.append(winf@sum(coefficients[j-1]@series[power-j] for j in range(1,power+1)))
    expected = {f'M{i}':series[i+1]/2 for i in range(4)}
    if photon:
        expected['constant'] = winf-v
    fields = ('M1','M3','M0','M2','constant')[:5 if photon else 4]
    bare = tuple(put(coefficients[i]) for i in (1,3,0,2))
    for partial in (False, True):
        marked = np.zeros((nq,len(fields)),bool)
        if partial:
            marked[0] = True
            marked[1,0] = True
            marked[3:,1] = True
        header = dict(moment_written=marked)
        calls = []
        def write(path, *, q_span, **kwargs):
            lo,hi = q_span
            names = [name for name in fields if name in kwargs]
            for name in names:
                column = fields.index(name)
                assert not marked[lo:hi,column].any(), (q_span,name)
                np.testing.assert_allclose(gather_to_host(kwargs[name]),expected[name][lo:hi],rtol=1e-11,atol=1e-12)
                marked[lo:hi,column] = True
            calls.append(q_span)
            return header
        ledger = NS(live_stages=(),U_bytes_per_rank=2**30)
        meta = NS(shared_pole_capacity=ledger,mu_basis=NS(n_packed=n),
                  nk_tot=8,nspin=1,nspinor_wfnfile=2,cell_volume=2.)
        with ExitStack() as ctx:
            ctx.enter_context(patch.object(bank,'_bank_context',return_value=(header,np.arange(nq),{})))
            ctx.enter_context(patch.object(bank,'_reserve',return_value=('moments',{})))
            ctx.enter_context(patch.object(bank,'_bank_execution',return_value=lambda fn,args,name: fn(*args)))
            roots = ctx.enter_context(patch.object(bank,'_coulomb_batch',return_value=(put(h),None,[n]*nq)))
            exact = ctx.enter_context(patch.object(bank,'exact_bare_moments',return_value=(*bare,{})))
            ctx.enter_context(patch.object(bank,'_finish_receipt',side_effect=lambda r,*args:r))
            ctx.enter_context(patch('file_io.shared_pole_store.write_shared_pole_bank',side_effect=write))
            receipt=bank.compute_moment_bank(None,meta,{'linalg':'local'},mesh_xy=mesh,
                sym=NS(trs_allowed=False),bank_io=dict(path='unused',identity={},coulomb={}),
                vertex=((np.empty((0,0,n)),),) if photon else None,contact=put(contact))
        assert receipt['completion'] and marked.all()
        assert roots.call_count == exact.call_count == 1
        if not partial:
            assert calls == [(0,nq)], calls
        if stack.process_index == 0:
            print(f'PASS moment batch photon={photon} partial={partial} writes={len(calls)}',flush=True)
finalize_process()
