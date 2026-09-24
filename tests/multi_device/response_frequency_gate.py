"""Combined complex-time charge/photon product and scalar value/ds oracle."""
from runtime import initialize_communicator_stack, finalize_process
stack = initialize_communicator_stack(platform='gpu')
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from common.collectives import gather_to_host
from gw.w_isdf import _get_chi_fractional_contour_kernel_face
from common.wfn_layout import psi_specs
from gw.response_bank import response_algebra, response_occupation_envelope
from types import SimpleNamespace
import minimax

mesh = stack.mesh
rng = np.random.default_rng(692)
def put(a, spec=P()):
    a = np.asarray(a)
    return jax.make_array_from_callback(a.shape, NamedSharding(mesh,spec), lambda ix:a[ix])
def check(a, b):
    error = np.max(abs(gather_to_host(a)-b))
    assert error < 2e-10, error
    return error
n, nk, nb = 8, 8, 8
grid = (4, 2, 1)
energy = np.broadcast_to(np.linspace(-1.,2.,nb),(nk,nb)).copy()
f = 1/(1+np.exp(energy/.3))
f[:,:2] = 1.
f[:,6:] = 0.
refs = np.array([0.,0.])
# Shared nodes: row 0 weights A(t) at q, row 1 weights conj(A(t)) at -q.
t = np.array([.13+.27j, .41-.11j, 0.])
c = np.array([[[.3+.2j,-.7+.1j,0.], [-.2+.4j,.1-.3j,0.]],
              [[.05-.3j,.2+.6j,0.], [.4+.1j,-.15-.25j,0.]]])
coords = np.unravel_index(np.arange(nk), grid)
negative = np.ravel_multi_index(tuple((-x) % m for x, m in zip(coords, grid)), grid)
assert np.any(negative != np.arange(nk))   # the reverse rows are really -q rows
for ns in (2,4):
    bare = (rng.normal(size=(nk,ns,n,nb))+1j*rng.normal(size=(nk,ns,n,nb)))/8
    current = bare if ns == 2 else bare[:,::-1].copy()*np.array([1,1j,-1,-1j])[None,:,None,None]
    br, cr = bare.transpose(0,3,1,2), current.transpose(0,3,1,2)
    expected=np.zeros((2,nk,n,n),complex)
    def fft(a):
        return np.fft.fftn(a.reshape(grid+a.shape[1:]),axes=(0,1,2),norm='ortho').reshape(a.shape)
    def green(psi,weight,tau,ref):
        return np.einsum('kamj,kj,kbnj->kmanb',psi,weight*np.exp(-(energy-ref)*tau),psi.conj())
    for node,time in enumerate(t):
        upper=fft(green(current,1-f,time,refs[1]))
        lower=fft(green(bare,f,-time.conjugate(),refs[0]))
        transform=fft(np.einsum('kmanb,kmanb->kmn',upper,lower.conj()))
        # ordered=True returns the physical row -q; the reverse rows are then q.
        expected+=c[0,:,node,None,None,None]*transform[negative][None]
        expected+=c[1,:,node,None,None,None]*transform.conj()[None]
    for layout,bounds in [('face',None),('axis',None),('axis',((0,6),(2,8)))]:
        sn, sm = psi_specs(layout)
        args = (put(t),put(c),
            put(bare,sm) if ns==2 else (put(bare,sm),put(current,sm)),
            put(br,sn) if ns==2 else (put(br,sn),put(cr,sn)),
            put(energy),put(f),put(1-f),put(refs))
        kernel = _get_chi_fractional_contour_kernel_face(mesh,grid,2,(nk,nb,n,ns),
            selected_q=tuple(range(nk)),pair_mode='direct',ordered=True,vertex=ns==4,
            bank_carry=True,layout=layout,band_ranges=bounds)
        carry=put(np.zeros_like(expected),P(None,None,'x','y'))
        executable=kernel.lower(*args,carry).compile()
        error=check(executable(*args,carry),expected)
        assert executable.memory_analysis().alias_size_in_bytes > 0
        if stack.process_index == 0:
            print(f'PASS complex-time ns={ns} {layout=} {bounds=} max_error={error:.3e} memory={executable.memory_analysis()}',flush=True)
# A scalar fit on a metallic interval checks values and analytic ds independently.
if stack.process_index == 0:
    # One group: every member's forward (t) and reverse (conj t) rows.
    z=np.array([.7+.2j,.3+.2j,.45j])
    d=np.linspace(-.2,1.,10001)
    for rule in minimax.response_group_rules(-.2,1.,z):
        m=rule['count']
        for row,sample in enumerate(rule['members']):
            for side,(pole,times) in enumerate(((z[sample],rule['t'][:m]),(-z[sample],np.conj(rule['t'][:m])))):
                basis=np.exp(-(d[:,None]+.2)*times)
                err=np.max(abs(basis@rule['value'][row,side,:m]-1/(d-pole)))*z[sample].imag
                ds=(1 if side==0 else -1)/(2*z[sample]*(d-pole)**2)
                derr=np.max(abs(basis@rule['derivative'][row,side,:m]-ds))*z[sample].imag**3
                assert max(err,derr)<1e-8,(err,derr)
        print(f'PASS scalar group members={rule["members"]} nodes={m} value+ds<1e-8',flush=True)
    # Physical occupation products, including negative differences, against
    # exact denominators; no small occupations are removed for this check.
    eps=np.linspace(-.5,.5,32)
    occ=1/(1+np.exp(40*eps))
    beta,amp=response_occupation_envelope(eps,occ,1-occ,0.)
    delta=eps[None,:]-eps[:,None]
    weight=occ[:,None]*(1-occ)[None,:]
    assert np.all(weight <= amp*np.exp(np.minimum(beta*delta,0.))*(1+1e-14))
    z=.3+.19j
    (rule,)=minimax.response_group_rules(-1.,1.,[z],decay_rate=beta)
    m=rule['count']
    for side,(pole,t) in enumerate(((z,rule['t'][:m]),(-z,np.conj(rule['t'][:m])))):
        basis=np.exp(-delta[...,None]*t)*weight[...,None]
        error=np.max(abs(basis@rule['value'][0,side,:m]-weight/(delta-pole)))*z.imag
        exact=weight*(1 if side==0 else -1)/(2*z*(delta-pole)**2)
        slope=np.max(abs(basis@rule['derivative'][0,side,:m]-exact))*z.imag**3
        assert max(error,slope)<1e-8,(error,slope)
        assert np.all((t.real>=0)&(t.real<=beta))
        print(f'PASS weighted scalar side={side} nodes={m} value={error:.3e} ds={slope:.3e}',flush=True)
    import runpy
    from pathlib import Path
    oracle=runpy.run_path(str(Path(__file__).resolve().parents[1]/'test_response_occupation_rule.py'))
    for metal in (False,True):
        oracle['test_signed_response_matches_direct_sum'](metal)
    print('PASS full signed metallic/insulating response and ds against direct transition sums',flush=True)
multihost_utils.sync_global_devices('scalar_complete')
# Split Dyson: independent dense complex solve, signed photon contact included.
for photon in (False,True):
    meta=SimpleNamespace(nk_tot=8,nspin=1,nspinor_wfnfile=2,cell_volume=2.)
    value,slope,_,receipt=response_algebra(meta,{'linalg':'local'},mesh_xy=mesh,n=n,photon=photon)
    h=np.broadcast_to(np.diag(np.linspace(.4,1.,n)).astype(complex),(5,n,n)).copy()
    if photon:
        h[:,n//2:,n//2:]*=-1
    chi=(rng.normal(size=(5,n,n))+1j*rng.normal(size=(5,n,n)))*.02
    dc=chi*.3j
    contact=np.eye(n)[None]*.01
    face=P(None,'x','y')
    wc=value(put(h,face),put(chi,face),*((put(contact,face),) if photon else ()))
    dw=slope(put(h,face),wc,put(dc,face))
    v=h if photon else h@h
    bare=receipt['prefactor']*chi-(meta.cell_volume*contact if photon else 0)
    w=np.linalg.solve(np.eye(n)[None]-v@bare,v)
    ev=check(wc,w-v);ed=check(dw,w@(receipt['prefactor']*dc)@w)
    if stack.process_index==0:
        print(f'PASS split Dyson photon={photon} value={ev:.3e} ds={ed:.3e}',flush=True)
finalize_process()
