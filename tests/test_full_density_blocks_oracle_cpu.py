"""Independent exact FF/FA/AF/AA coherent-response identities, tiny CPU only."""
import numpy as np
import pytest

from tests._full_density_blocks_oracle import (
    pair_spectrum,pair_time,pair_frequency,commutator_moments,four_block_panels,pair_interval,
)
from tests._full_microscopic_oracle import ordered_pr,ordered_pr_time


def case(seed=319):
    rng=np.random.default_rng(seed)
    nk,nb,ns,nmu=3,4,2,3
    psi=.2*(rng.normal(size=(nk,nb,ns,nmu))+1j*rng.normal(size=(nk,nb,ns,nmu)))
    energy=np.array([[-2.1,-.31,.45,1.6],[-2.0,-.19,.51,1.8],[-2.3,-.4,.39,1.7]])
    rho=np.zeros((nk,nb,nb),complex);rho[:,0,0]=1
    for k in range(nk):
        u,_=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))
        rho[k,1:3,1:3]=u@np.diag([.19,.83])@u.conj().T
    partner=np.array([[(k-q)%nk for k in range(nk)] for q in range(nk)])
    return psi,energy,rho,partner


def test_literal_four_blocks_sum_full_coherent_time_and_frequency_all_q():
    psi,energy,rho,partners=case()
    blocks=four_block_panels(rho,slice(1,3));spin=1.7;gamma=.043
    assert np.linalg.norm(rho@np.array([np.diag(e) for e in energy])-
                          np.array([np.diag(e) for e in energy])@rho)>.1
    times=np.array([0.,.13+.21j,-.11+.29j,.03-.17j])
    sites=np.array([-.37+.23j,.19+.31j,.87+.46j])
    for partner in partners:
        got_t=np.zeros((len(times),3,3),complex);got_z=np.zeros((len(sites),3,3),complex)
        for left,right in blocks.values():
            f=pair_spectrum(psi,energy,left,right,partner,spin)
            r=pair_spectrum(psi,energy,right,left,partner,spin)
            got_t+=pair_time(f,times)-pair_time(r,times)
            got_z+=pair_frequency(f,sites,gamma)-pair_frequency(r,sites,gamma)
        np.testing.assert_allclose(got_t,ordered_pr_time(psi,energy,rho,partner,times,spin),rtol=2e-13,atol=2e-15)
        np.testing.assert_allclose(got_z,ordered_pr(psi,energy,rho,partner,sites,gamma,spin),rtol=2e-13,atol=2e-15)


def test_independent_reverse_is_minus_conjugate_time_and_q_not_stationary_shortcut():
    psi,energy,rho,partners=case()
    times=np.array([.17+.23j,-.09+.31j,.04-.19j])
    for name,(left,right) in four_block_panels(rho,slice(1,3)).items():
        forward=[pair_spectrum(psi,energy,left,right,p) for p in partners]
        reverse=[pair_spectrum(psi,energy,right,left,p) for p in partners]
        wrong=[]
        for q in range(3):
            direct=pair_time(reverse[q],-times.conjugate())
            exact=pair_time(forward[(-q)%3],times).conjugate()
            np.testing.assert_allclose(direct,exact,rtol=2e-13,atol=2e-15)
            # A spatial transpose is NOT part of the q/-q conjugation.
            if name=='FA' and q==1:
                assert np.linalg.norm(direct-exact.swapaxes(-1,-2))>1e-4
            wrong.append(np.linalg.norm(pair_time(reverse[q],times.conjugate())-exact))
        assert max(wrong)>1e-4


def test_four_block_moments_preserve_generic_nonzero_one_over_z_and_gamma_once():
    psi,energy,rho,partners=case();gamma=.071
    h=1e-4
    for p in partners:
        first=np.zeros((3,3),complex);second=first.copy()
        for left,right in four_block_panels(rho,slice(1,3)).values():
            f=pair_spectrum(psi,energy,left,right,p);r=pair_spectrum(psi,energy,right,left,p)
            x1,x2=commutator_moments(f,r,gamma);first+=x1;second+=x2
        expected1=ordered_pr_time(psi,energy,rho,p,[0.])[0]
        vals=ordered_pr_time(psi,energy,rho,p,[-2*h,-h,h,2*h])
        derivative=(vals[0]-8*vals[1]+8*vals[2]-vals[3])/(12*h)
        np.testing.assert_allclose(first,expected1,rtol=2e-13,atol=2e-15)
        np.testing.assert_allclose(second,-derivative-1j*gamma*expected1,rtol=2e-9,atol=2e-12)
        assert np.linalg.norm(first)>1e-4


def test_positive_supports_are_current_global_bounds_not_population_or_band_labels():
    _,energy,rho,_=case()
    columns=dict(FF=([3],[0]),FA=([1,2],[0]),AF=([3],[1,2]),AA=([1,2],[1,2]))
    for name,(l,r) in columns.items():
        box=pair_interval(energy,l,r)
        assert box['positive']==(name!='AA')
    # Frozen/active manifolds can overlap across k even with a locally
    # ordered band table. An explicit signed interval is then mandatory.
    crossing=energy.copy();crossing[0,0]=-.2
    box=pair_interval(crossing,[1,2],[0])
    assert box['lo']<0 and not box['positive']
    np.testing.assert_array_equal(np.diagonal(rho,axis1=-2,axis2=-1)[:,0],1.)


def test_literal_green_right_phase_and_nontrivial_complex_reference_cancellation():
    psi,energy,rho,partners=case();nk,nb,ns,nmu=psi.shape
    columns=dict(FF=([3],[0]),FA=([1,2],[0]),AF=([3],[1,2]),AA=([1,2],[1,2]))
    def green(matrix,time,reference,left_phase=False):
        out=np.zeros((nk,ns*nmu,ns*nmu),complex)
        for k in range(nk):
            for a in range(nb):
                for b in range(nb):
                    phased=energy[k,a if left_phase else b]-reference
                    out[k]+=(matrix[k,a,b]*np.exp(-phased*time)*
                        np.outer(psi[k,a].reshape(-1),psi[k,b].reshape(-1).conjugate()))
        return out.reshape(nk,ns,nmu,ns,nmu)
    times=np.array([.13+.21j,-.11+.29j,.03-.17j])
    for name,(left,right) in four_block_panels(rho,slice(1,3)).items():
        box=pair_interval(energy,*columns[name]);origin=box['lo']-.137
        if name=='FA':
            assert np.linalg.norm(green(left,times[0],0)-green(left,times[0],0,left_phase=True))>1e-4
        for q,partner in enumerate(partners):
            exact=pair_time(pair_spectrum(psi,energy,left,right,partner),times)
            for i,time in enumerate(times):
                rleft=box['left_min'] if time.real>=0 else box['left_max']
                rright=box['right_max'] if time.real>=0 else box['right_min']
                gl=green(left,time,rleft)
                gr=green(right,-time.conjugate(),rright)[partner]
                reference_product=np.einsum('ksmtn,ksmtn->mn',gl,gr.conjugate())/nk
                np.testing.assert_allclose(reference_product*np.exp(-(rleft-rright)*time),
                    exact[i],rtol=2e-13,atol=2e-15)
                np.testing.assert_allclose(reference_product*np.exp((origin-rleft+rright)*time),
                    exact[i]*np.exp(origin*time),rtol=2e-13,atol=2e-15)


def test_scalar_forward_reverse_signs_and_separate_reference_shifts():
    psi,energy,rho,partners=case();gamma=.043
    sites=np.array([-.37+.23j,.19+.31j,.87+.46j]);shifted=sites+1j*gamma
    columns=dict(FF=([3],[0]),FA=([1,2],[0]),AF=([3],[1,2]),AA=([1,2],[1,2]))
    # INDEPENDENT analytic Laplace integral, not a production planner or
    # alternative production route. Eight64-node Gauss panels suffice for
    # this tiny fixture; these rows explicitly expose BOTH signs.
    u,w=np.polynomial.legendre.leggauss(64)
    length=36/shifted.imag.min()/8
    s=np.concatenate([length*(j+(u+1)/2) for j in range(8)])
    ws=np.tile(w*length/2,8)
    for name,(left,right) in four_block_panels(rho,slice(1,3)).items():
        box=pair_interval(energy,*columns[name])
        origin=box['lo']
        values=np.stack([np.stack((1j*ws*np.exp(1j*(z-origin)*s),
                                  -1j*ws*np.exp(1j*(z+origin)*s))) for z in shifted])
        rules=[dict(t=1j*s,count=len(s),reference_ry=origin,value=values)]
        forwards=[pair_spectrum(psi,energy,left,right,p) for p in partners]
        reverses=[pair_spectrum(psi,energy,right,left,p) for p in partners]
        for q in range(3):
            got=np.zeros((len(sites),3,3),complex)
            for rule in rules:
                n=rule['count'];t=rule['t'][:n];origin=rule['reference_ry']
                # Electron and hole energy references differ and phases
                # belong to their RIGHT energy index even for coherent M.
                dl=np.where(t.real>=0,box['left_min'],box['left_max'])
                dr=np.where(t.real>=0,box['right_max'],box['right_min'])
                delta=dl-dr
                actual_f=pair_time(forwards[q],t)
                referenced_f=actual_f*np.exp(delta*t)[:,None,None]
                rule_f=referenced_f*np.exp((origin-delta)*t)[:,None,None]
                # Independently evaluate the swapped orientation: NOT a
                # population/time-stationary shortcut and NOT at +conj(t).
                actual_r=pair_time(reverses[q],-t.conjugate())
                referenced_r=actual_r*np.exp(delta*t.conjugate())[:,None,None]
                rule_r=referenced_r*np.exp((origin-delta)*t.conjugate())[:,None,None]
                for j in range(len(sites)):
                    got[j]-=np.einsum('t,tij->ij',rule['value'][j,0,:n],rule_f)
                    got[j]-=np.einsum('t,tij->ij',rule['value'][j,1,:n],rule_r)
            exact=pair_frequency(forwards[q],sites,gamma)-pair_frequency(reverses[q],sites,gamma)
            np.testing.assert_allclose(got,exact,rtol=3e-7,atol=2e-9)


def test_numerical_response_group_enlargement_is_exact_with_filled_and_empty_extras():
    psi4,e4,rho4,partners=case();rng=np.random.default_rng(712)
    psi=.2*(rng.normal(size=(3,6,2,3))+1j*rng.normal(size=(3,6,2,3)))
    psi[:,1:5]=psi4
    energy=np.column_stack(([-3.2,-3.1,-3.4],e4,[2.8,2.9,3.0]))
    rho=np.zeros((3,6,6),complex);rho[:,0,0]=1;rho[:,1:5,1:5]=rho4
    sites=np.array([-.24+.33j,.39+.27j]);gamma=.057
    # PhysicalA=2:4; expanded numericalA=1:5 additionally holds one
    # occupied and one empty spectator, with THEIR original E/rho/psi.
    for p in partners:
        expected=ordered_pr(psi,energy,rho,p,sites,gamma)
        outcomes=[]
        for group in (slice(2,4),slice(1,5)):
            value=np.zeros_like(expected);first=np.zeros((3,3),complex);second=first.copy()
            for left,right in four_block_panels(rho,group).values():
                f=pair_spectrum(psi,energy,left,right,p);r=pair_spectrum(psi,energy,right,left,p)
                value+=pair_frequency(f,sites,gamma)-pair_frequency(r,sites,gamma)
                x1,x2=commutator_moments(f,r,gamma);first+=x1;second+=x2
            np.testing.assert_allclose(value,expected,rtol=3e-13,atol=3e-15)
            outcomes.append((value,first,second))
        for old,new in zip(*outcomes):
            np.testing.assert_allclose(old,new,rtol=3e-13,atol=3e-15)
    bad=rho.copy();bad[:,0,5]=.01j;bad[:,5,0]=-.01j
    with pytest.raises(ValueError,match='all density coherences'):
        four_block_panels(bad,slice(1,5))


def test_native_paired_fft_keeps_both_independent_coherent_orientations():
    import jax
    from jax.sharding import Mesh
    from ffi import fft as fft
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream
    jax.config.update('jax_enable_x64',True)
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    psi,energy,rho,partners=case();nk,nb,ns,nmu=psi.shape
    pm=psi.transpose(0,2,3,1)
    endpoint=density_left_endpoint(mesh,pm.shape)
    times=np.array([.13+.21j,-.11+.29j,.03-.17j])
    weights=np.zeros((2,2*len(times),len(times)),complex)
    weights[0,:len(times)]=np.eye(len(times))
    weights[1,len(times):]=np.eye(len(times))
    cols=dict(FF=([3],[0]),FA=([1,2],[0]),AF=([3],[1,2]),AA=([1,2],[1,2]))
    for name,(left,right) in four_block_panels(rho,slice(1,3)).items():
        box=pair_interval(energy,*cols[name])
        bounds=tuple(box[k] for k in ('left_min','left_max','right_min','right_max'))
        with fft.xla_reference():
            stream=full_density_response_stream(mesh,(3,1,1),(nk,nb,nmu,ns),q_ids=[0,1,2],
                n_outputs=2*len(times),pair_bounds_ry=bounds,rule_reference_ry=0.)
            got=np.asarray(stream(times,weights,endpoint(pm,left),psi,endpoint(pm,right),energy))/np.sqrt(nk)
        expected=[]
        for partner in partners:
            f=pair_spectrum(psi,energy,left,right,partner)
            r=pair_spectrum(psi,energy,right,left,partner)
            expected.append(np.concatenate((pair_time(f,times),pair_time(r,-times.conjugate()))))
        np.testing.assert_allclose(got,np.stack(expected),rtol=3e-12,atol=3e-14)


def test_native_noncrossing_paired_rules_fft_match_literal_complex_frequency():
    import jax
    from jax.sharding import Mesh
    from ffi import fft as fft
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream
    from gw.response_bank import full_density_sample_rules
    jax.config.update('jax_enable_x64',True)
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    psi,energy,rho,partners=case();nk,nb,ns,nmu=psi.shape
    pm=psi.transpose(0,2,3,1);endpoint=density_left_endpoint(mesh,pm.shape)
    gamma=.043;sites=np.array([-.37+.23j,.19+.31j,.87+.46j])
    cols=dict(FF=([3],[0]),FA=([1,2],[0]),AF=([3],[1,2]))
    for name,(left,right) in four_block_panels(rho,slice(1,3)).items():
        if name=='AA':continue # signed block independently tested above
        box=pair_interval(energy,*cols[name])
        assert box['lo']>np.max(abs(sites.real))
        bounds=tuple(box[k] for k in ('left_min','left_max','right_min','right_max'))
        rules=full_density_sample_rules(energy,sites,gamma_ry=gamma,tolerance=1e-8,
            interval_ry=(box['lo'],box['hi']),paired=True)
        assert all(r['construction']=='continuum_noncrossing_laplace' for r in rules)
        got=np.zeros((nk,len(sites),nmu,nmu),complex)
        with fft.xla_reference():
            for rule in rules:
                stream=full_density_response_stream(mesh,(3,1,1),(nk,nb,nmu,ns),q_ids=[0,1,2],
                    n_outputs=len(rule['members']),pair_bounds_ry=bounds,rule_reference_ry=rule['reference_ry'])
                values=np.asarray(stream(rule['times'],rule['weights'],endpoint(pm,left),psi,
                    endpoint(pm,right),energy))/np.sqrt(nk)
                got[:,rule['members']]+=values
        expected=[]
        for partner in partners:
            f=pair_spectrum(psi,energy,left,right,partner)
            r=pair_spectrum(psi,energy,right,left,partner)
            expected.append(pair_frequency(f,sites,gamma)-pair_frequency(r,sites,gamma))
        np.testing.assert_allclose(got,np.stack(expected),rtol=2e-7,atol=2e-9)
