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
energy = np.broadcast_to(np.linspace(-1.,2.,nb),(nk,nb)).copy()
f = 1/(1+np.exp(energy/.3))
f[:,:2] = 1.
f[:,6:] = 0.
refs = np.array([0.,0.])
t = np.array([.13+.27j, .41-.11j, 0.])
c = np.array([[.3+.2j,-.7+.1j,0.], [-.2+.4j,.1-.3j,0.]])
reverse = np.array([False,True,False])
for ns in (2,4):
    bare = (rng.normal(size=(nk,ns,n,nb))+1j*rng.normal(size=(nk,ns,n,nb)))/8
    current = bare if ns == 2 else bare[:,::-1].copy()*np.array([1,1j,-1,-1j])[None,:,None,None]
    br, cr = bare.transpose(0,3,1,2), current.transpose(0,3,1,2)
    expected=np.zeros((2,nk,n,n),complex)
    def fft(a):
        return np.fft.fftn(a.reshape((2,2,2)+a.shape[1:]),axes=(0,1,2),norm='ortho').reshape(a.shape)
    def green(psi,weight,tau,ref):
        return np.einsum('kamj,kj,kbnj->kmanb',psi,weight*np.exp(-(energy-ref)*tau),psi.conj())
    for time,coef,rev in zip(t,c.T,reverse):
        tau=time.conjugate() if rev else time
        upper=fft(green(current,1-f,tau,refs[1]))
        lower=fft(green(bare,f,-tau.conjugate(),refs[0]))
        product=np.einsum('kmanb,kmanb->kmn',upper,lower.conj())
        expected+=coef[:,None,None,None]*fft(product.conj() if rev else product)[None]
    # Every q is its own negative on this 2x2x2 oracle grid.
    for layout,bounds in [('face',None),('axis',None),('axis',((0,6),(2,8)))]:
        sn, sm = psi_specs(layout)
        args = ((put(t),put(reverse)),put(c),
            put(bare,sm) if ns==2 else (put(bare,sm),put(current,sm)),
            put(br,sn) if ns==2 else (put(br,sn),put(cr,sn)),
            put(energy),put(f),put(1-f),put(refs))
        kernel = _get_chi_fractional_contour_kernel_face(mesh,(2,2,2),2,(nk,nb,n,ns),
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
    z=.7+.2j
    rule=minimax.response_frequency_rule(-.2,1.,z)
    d=np.linspace(-.2,1.,10001)
    for i,pole in enumerate((z,-z)):
        basis=np.exp(-(d[:,None]+.2)*rule['t'][i])
        err=np.max(abs(basis@rule['value'][i]-1/(d-pole)))*z.imag
        ds=(1 if i==0 else -1)/(2*z*(d-pole)**2)
        derr=np.max(abs(basis@rule['derivative'][i]-ds))*z.imag**3
        assert max(err,derr)<1e-8,(err,derr)
        print(f'PASS scalar primitive={i} nodes={rule["counts"][i]} value={err:.3e} ds={derr:.3e}',flush=True)
    # Physical occupation products, including negative differences, against
    # exact denominators; no small occupations are removed for this check.
    eps=np.linspace(-.5,.5,32)
    occ=1/(1+np.exp(40*eps))
    beta,amp=response_occupation_envelope(eps,occ,1-occ,0.)
    delta=eps[None,:]-eps[:,None]
    weight=occ[:,None]*(1-occ)[None,:]
    assert np.all(weight <= amp*np.exp(np.minimum(beta*delta,0.))*(1+1e-14))
    z=.3+.19j
    rule=minimax.response_frequency_rule(-1.,1.,z,decay_rate=beta)
    for i,pole in enumerate((z,-z)):
        t=rule['t'][i]
        basis=np.exp(-delta[...,None]*t)*weight[...,None]
        error=np.max(abs(basis@rule['value'][i]-weight/(delta-pole)))*z.imag
        exact=weight*(1 if i==0 else -1)/(2*z*(delta-pole)**2)
        slope=np.max(abs(basis@rule['derivative'][i]-exact))*z.imag**3
        assert max(error,slope)<1e-8,(error,slope)
        assert np.all((t.real>=0)&(t.real<=beta))
        print(f'PASS weighted scalar primitive={i} nodes={rule["counts"][i]} value={error:.3e} ds={slope:.3e}',flush=True)
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
