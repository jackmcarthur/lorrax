"""Q-local response equations: ragged batches, charge/photon and ordered moments."""
from runtime import initialize_communicator_stack, finalize_process
stack = initialize_communicator_stack(platform="gpu")
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host
from gw.response_bank import response_algebra
from types import SimpleNamespace

mesh = stack.mesh
rng = np.random.default_rng(829)
face = P(None, "x", "y")
def put(a):
    return jax.make_array_from_callback(a.shape, NamedSharding(mesh, face), lambda ix: a[ix])
def check(got, expected):
    error = np.max(abs(gather_to_host(got)-expected))
    assert error < 2e-11, error
    return error

for nq in (5, int(mesh.size)+3):
    n = 2*int(mesh.shape['x'])
    meta = SimpleNamespace(nk_tot=8, nspin=1, nspinor_wfnfile=2, cell_volume=2.)
    for photon, ordered in ((False, False), (False, True), (True, True)):
        value, slope, moments, receipt = response_algebra(meta, {'linalg':'local'},
            mesh_xy=mesh, n=n, photon=photon, ordered=ordered)
        h = np.broadcast_to(np.diag(np.linspace(.4,1.,n)).astype(complex), (nq,n,n)).copy()
        if photon: h[:,n//2:,n//2:] *= -1
        a = [(rng.normal(size=h.shape)+1j*rng.normal(size=h.shape))*.01 for _ in range(5)]
        chi, dc, a0, a1, o0 = a
        o1 = a1*.2j
        contact = np.eye(n,dtype=complex)[None]*.01
        args = (put(h), put(chi)) + ((put(contact),) if photon else ())
        wc = value(*args)
        dw = slope(put(h), wc, put(dc))
        v = h if photon else h@h
        bare = receipt['prefactor']*chi-(meta.cell_volume*contact if photon else 0)
        w = np.linalg.solve(np.eye(n)[None]-v@bare, v)
        err = max(check(wc,w-v),check(dw,w@(receipt['prefactor']*dc)@w))
        cs = (o0,a0,o1,a1) if ordered else (a0,a1)
        margs = (put(h),put(a0),put(a1)) + ((put(o0),put(o1)) if ordered else ())
        if photon: margs += (put(contact),)
        got = moments(*margs)
        winf = np.linalg.solve(np.eye(n)[None]+v@(meta.cell_volume*contact),v) if photon else v
        coefficients=[]
        for k,c in enumerate(cs):
            rhs=c@winf
            for i in range(k): rhs += cs[i]@coefficients[k-i-1]
            coefficients.append(winf@rhs)
        expected = ((winf-v,) if photon else ()) + tuple(.5*c for c in coefficients)
        err=max(err,*(check(g,e) for g,e in zip(got,expected)))
        assert len(got)==len(expected)
        if stack.process_index==0:
            print(f'PASS local response {nq=} {photon=} {ordered=} max_error={err:.3e}',flush=True)

# Exercise the ordinary GEMM service after generalizing its shared real-row helper.
from distrib_la import matmul
x=np.ones((5,n,n),complex); y=np.broadcast_to(np.eye(n),(5,n,n)).astype(complex).copy()
check(matmul(put(x),put(y),mesh=mesh,backend='off'),x@y)
if stack.process_index==0: print('PASS existing ragged GEMM service',flush=True)

# Read only the small, committed current Fe warm seed through its canonical owner.
import sys
if len(sys.argv)>1:
    from file_io.restart_bundle import read_qp_rotations_artifact
    from file_io.commit_state import assert_committed
    import h5py
    with h5py.File(sys.argv[1]) as f: assert_committed(f,path=sys.argv[1])
    seed=read_qp_rotations_artifact(sys.argv[1])
    if stack.process_index==0: print('PASS committed Fe warm seed; keys='+','.join(sorted(seed)),flush=True)
finalize_process()
