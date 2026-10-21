"""Read-only saved-operator audit; run with optional DATA_ROOT argument.

No GPU, production model mutation, clipping, or closure replacement.
"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np

BASE=Path(sys.argv[1] if len(sys.argv)>1 else '/home/jackm/mgkba_campaign_20261019/qgrid_20261020')
dag=lambda a:a.conj().swapaxes(-1,-2)
herm=lambda a:(a+dag(a))/2
width=lambda a:1j*(a-dag(a))
fn=lambda a:float(np.linalg.norm(a))
mx=lambda a:float(np.max(np.abs(a)))
with np.load(BASE/'material_Bpilot8e8_3ada_60071791/snapshot_0000.npz') as z:
    U,E,P,H,fullMH=(z[k] for k in ('U','E_ry','rho_in_basis','H_material_ry','MH_ry'))
    sourceP=z['rho_source']
with np.load(BASE/'same_w_rate_eps8.npz') as z:
    rows={b:z['retarded_'+b][0] for b in ('cond','val')}
    cols={b:z['retarded_'+b][1] for b in ('cond','val')}
    gamma,D,C=(z[k] for k in ('gamma_body_ry','injection_body_ry','C_body_ry'))
with np.load(BASE/'spectral_60080234.npz') as z:
    grid={b:z['retarded_'+b+'_ry'] for b in ('cond','val')}
    omega=z['omega_absolute_ry']
N=P.shape[-1]; filled=np.where(P[0].diagonal().real==1)[0]; empty=np.where(P[0].diagonal().real==0)[0]
Q=np.eye(N)-P
compress=lambda a,ii:a[...,ii[:,None],ii]
def report(a,b):
    c=b-(a@P+P@a)/2
    filled_e=np.linalg.eigvalsh(compress(a-b,filled))
    empty_e=np.linalg.eigvalsh(compress(b,empty))
    return dict(trace_mean_ry=float(np.trace(c,axis1=-2,axis2=-1).real.mean()),
        filled_min_ry=float(filled_e.min()),filled_worst_k=int(filled_e[:,0].argmin()),
        empty_min_ry=float(empty_e.min()),filled_bad_k=int(np.sum(filled_e[:,0]<0)),
        empty_bad_k=int(np.sum(empty_e[:,0]<0)),C_fro_ry=fn(c))

r=dict(source_commit='371efadca2df70234d406e2a8d674eefbd6fe2e0', diagnostic_only=True,
    limits='Saved active BODY, exact t0 projector, all64k row/column pairs and k0 common-frequency grid only. No model changes, clipping, roots, propagation, full guard, coherent-density material test, eta refinement, complete microscopic accuracy or positivity theorem. The PV-free comparison is not an enabled or validated replacement.',
    boundary_proof='For exact filled projector P, PCP=-P(Gamma-D)P=-P width(Rp) P; hole conversion cancels identically. For empty Q, QCQ=Q D Q. Any Hermitian commutator has zero P/P and Q/Q compression. All operators in Ry; Hartree conversion multiplies rates by positive 1/2 and cannot alter signs.',
    head_scope='Head omitted in saved replay. At this exact diagonal projector its particle filled compression and hole empty compression vanish by their Q/P factors; omission cannot change either boundary. Not true for arbitrary coherent later snapshots.',
    eta_scope='The three-level reciprocal counterexample also leaks at eta=1e-8. Reducing eta is not a general cure: unequal off-resonant PV column limits remain. Material eta dependence has not been measured; other physical channels can compensate individual negative channel minors.')
r['inputs']={}
for relative in ('spectral_60080234.npz','same_w_rate_eps8.npz','material_Bpilot8e8_3ada_60071791/snapshot_0000.npz'):
    digest=hashlib.sha256()
    with (BASE/relative).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    r['inputs'][relative]=digest.hexdigest()
r['state']=dict(U_identity_error=mx(U-np.eye(N)),projector_error=mx(P@P-P),
    rho_rotation_error=mx(sourceP-U@P@dag(U)),H_reconstruction_error_ry=mx(H-(U*E[:,None,:])@dag(U)),
    H_rho_commutator_ry=mx(H@sourceP-sourceP@H),MH_filled_commutator_ry=mx(compress(-1j*(fullMH@P-P@fullMH),filled)),
    minimum_energy_gap_ry=float(np.diff(E[0]).min()),filled_count=len(filled),empty_count=len(empty))
Wc={b:width(cols[b]) for b in cols}; Wr={b:width(rows[b]) for b in rows}
r['saved_reconstruction']=dict(gamma_error=mx(gamma-Wc['cond']-Wc['val']),D_error=mx(D-Wc['val']),
    C_error=mx(C-D+(gamma@P+P@gamma)/2))
gcol={b:np.einsum('jij->ij',v) for b,v in grid.items()}
grow={b:np.einsum('iij->ij',v) for b,v in grid.items()}
r['common_grid_reconstruction']={b:dict(column_error_ry=mx(gcol[b]-cols[b][0]),row_error_ry=mx(grow[b]-rows[b][0]),
    common_K_min_ry=float(np.linalg.eigvalsh(width(v)).min())) for b,v in grid.items()}
r['common_grid_reconstruction']['exact_energy_match']=bool(np.array_equal(omega,E[0]))
absorb={b:(Wc[b]+Wr[b])/2 for b in Wc}
pv={b:(Wc[b]-Wr[b])/2 for b in Wc}
r['raw_all_k']=report(Wc['cond']+Wc['val'],Wc['val'])
r['row_all_k']=report(Wr['cond']+Wr['val'],Wr['val'])
r['absorption_only_all_k']=report(absorb['cond']+absorb['val'],absorb['val'])
r['PV_scan']={str(t):report(absorb['cond']+absorb['val']+t*(pv['cond']+pv['val']),
    absorb['val']+t*pv['val']) for t in (-1,-.5,0,.25,.5,.75,1)}
# Independent common-frequency H/K splitting; no row/column width shortcut.
r['grid_decomposition']={}
for b,v in grid.items():
    kt,ht=width(v),herm(v)
    sa=(np.einsum('iij->ij',kt)+np.einsum('jij->ij',kt))/2
    dp=1j*(np.einsum('jij->ij',ht)-np.einsum('iij->ij',ht))
    r['grid_decomposition'][b]=dict(identity_error_ry=mx(width(gcol[b])-sa-dp),
        absorption_vs_saved_error_ry=mx(sa-absorb[b][0]),PV_vs_saved_error_ry=mx(dp-pv[b][0]))
evals,evecs=np.linalg.eigh(compress(Wc['cond'][0],filled)); v=np.zeros(N,complex);v[filled]=evecs[:,0]
r['negative_witness']=dict(eigenvalue_ry=float(evals[0]),eigenvalue_meV=float(evals[0]*13605.693122994),
    absorption_rayleigh_ry=float(np.vdot(v,absorb['cond'][0]@v).real),
    PV_rayleigh_ry=float(np.vdot(v,pv['cond'][0]@v).real),
    PCPrayleigh_ry=float(np.vdot(v,C[0]@v).real),
    significant_support=np.where(np.abs(v)>0.05)[0].tolist(),
    energy_spread_ry=float(np.sqrt(np.vdot(v,E[0]**2*v).real-np.vdot(v,E[0]*v).real**2)))
# All 2x2 occupied minors: a small witness is stronger than a black-box eigensolver alone.
a=compress(Wc['cond'][0],filled)
minor=np.real(np.diag(a))[:,None]*np.real(np.diag(a))[None,:]-np.abs(a)**2
i,j=np.unravel_index(np.argmin(minor),minor.shape)
r['two_state_witness']=dict(bands=[int(filled[i]),int(filled[j])],determinant_ry2=float(minor[i,j]),
    eigenvalues_ry=np.linalg.eigvalsh(a[np.ix_([i,j],[i,j])]).tolist(),gap_ry=float(abs(E[0,i]-E[0,j])))
# A row/column ordering of OUTER rho changes coherent/off-boundary pieces, not exact compressed signs.
rp,rh=cols['cond'],cols['val']
one=1j*(rh@Q-rp@P); literal=one+dag(one)
other=1j*(Q@rh-P@rp); opposite=other+dag(other)
Mh=herm(rp+rh)
r['outer_rho_placement']=dict(literal_vs_split_error_ry=mx(literal-(C-1j*(Mh@P-P@Mh))),
    filled_boundary_change_ry=mx(compress(literal-opposite,filled)),
    empty_boundary_change_ry=mx(compress(literal-opposite,empty)))
# Operator-valued shell map is covariant when H spectral projectors rotate too.
rng=np.random.default_rng(438); q,_=np.linalg.qr(rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)))
rt=np.einsum('ai,fab,bj->fij',q.conj(),grid['cond'],q)
projectors=np.einsum('bi,bj->bij',q.conj(),q)
covariant=np.einsum('fij,fjk->ik',rt,projectors)
target=dag(q)@gcol['cond']@q
naive=np.einsum('jij->ij',rt)
basis=dag(q)[:,filled]
r['unitary_covariance']=dict(operator_shell_error_ry=mx(covariant-target),naive_fixed_indices_error_ry=mx(naive-target),
    boundary_eigenvalue_error_ry=mx(np.linalg.eigvalsh(dag(basis)@(dag(q)@Wc['cond'][0]@q)@basis)-np.linalg.eigvalsh(a)))
# Diagonal/common-energy controls are diagnostics, not a new closure.
diag=lambda a:np.eye(N)*a.diagonal(axis1=-2,axis2=-1)[...,None,:]
r['diagonal_commuting_control']=report(diag(Wc['cond']+Wc['val']),diag(Wc['val']))
r['equal_frequency_control']=[dict(index=f,cond_min_ry=float(np.linalg.eigvalsh(compress(width(grid['cond'][f]),filled)).min()),
    val_min_ry=float(np.linalg.eigvalsh(compress(width(grid['val'][f]),empty)).min())) for f in (0,15,29,30,43)]
r['energy_degeneracy']=dict(minimum_adjacent_gap_all_k_ry=float(np.diff(E,axis=-1).min()),
    nearest_pair_k0_meV=float(np.diff(E[0]).min()*13605.693122994))
# Passive rank-one common-frequency spectrum; two fully filled destinations.
tiny_E=np.array([0.,1.]); tiny=np.ones((2,2),complex)/(tiny_E[None,:]+1j)
tinyK=width(tiny)
r['analytic_counterexample']=dict(retarded_column_real=tiny.real.tolist(),retarded_column_imag=tiny.imag.tolist(),
    filled_width_eigenvalues=np.linalg.eigvalsh(tinyK).tolist(),filled_width_determinant=float(np.linalg.det(tinyK).real),
    common_frequency_width_eigenvalues=[np.linalg.eigvalsh(width(np.ones((2,2))/(e+1j))).tolist() for e in tiny_E],
    PCP_max_eigenvalue=float(np.linalg.eigvalsh(-tinyK).max()),
    meaning='A single passive channel alone is not a closed reciprocal all-k system, so this demonstrates boundary indefiniteness, not a number-conservation counterexample.')
# Closed three-level reciprocal reference: two filled levels couple to one
# empty level through a Hermitian vertex, one passive +/-Omega boson pair.
e3=np.array([-1.,-.5,1.]); p3=np.diag([1.,1.,0.]); eta=.2; boson=1.
m3=np.array([[0.,0.,1.],[0.,0.,1.],[1.,1.,0.]])
rp3=np.zeros((3,3),complex);rh3=rp3.copy()
for b in range(3):
    for n in range(3):
        outer=np.outer(m3[:,n],m3[:,n])
        rp3[:,b]+=outer[:,b]*(1-p3[n,n])/(e3[b]-e3[n]-boson+1j*eta)
        rh3[:,b]+=outer[:,b]*p3[n,n]/(e3[b]-e3[n]+boson+1j*eta)
g3=width(rp3+rh3);d3=width(rh3);c3=d3-(g3@p3+p3@g3)/2
r['closed_reciprocal_counterexample']=dict(energies=e3.tolist(),eta=eta,boson=boson,
    trace=float(np.trace(c3).real),filled_blocking_eigenvalues=np.linalg.eigvalsh((g3-d3)[:2,:2]).tolist(),
    filled_C_eigenvalues=np.linalg.eigvalsh(c3[:2,:2]).tolist(),empty_injection=float(d3[2,2].real),
    filled_minor=float(np.linalg.det((g3-d3)[:2,:2]).real),
    scope='Literal passive reference branches; reciprocal Hermitian vertex; diagonal sharp rho commuting with H; no head, numerical quadrature, coherent rho, basis ambiguity or fitting. Arbitrary energy units. Not material fitting.')
print(json.dumps(r,indent=2))
