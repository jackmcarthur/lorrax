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


@pytest.mark.parametrize('nspin',[1,2])
def test_common_mode11_endpoint_mapping_literal_complex_time_all_q(nspin):
    """Algebra oracle only: incumbent IFFT contraction with independent LEFTs.

    This does not execute the production native kernel. It fixes its exact
    endpoint, q-sign and reference contracts before the implementation is
    consolidated. The separate transition-index oracle supplies the answers.
    """
    psi,energy,rho,partners=case(704)
    psi=psi[:,:,:nspin]
    nk=psi.shape[0]
    times=np.array([0.,.13+.21j,-.11+.29j,.03-.17j,-.09-.07j])
    identity=np.broadcast_to(np.eye(energy.shape[1]),rho.shape)

    def green_from_left(matrix,time,reference):
        # M precedes the RIGHT-band phase, including [M,E] != 0.
        left=np.einsum('kab,kasm->kbsm',matrix,psi)
        phase=np.exp(-(energy-reference)*time)
        return np.einsum('kbsm,kb,kbtn->ksmtn',left,phase,psi.conjugate())

    # Include the unsplit pair and every exact frozen/active block.
    panels=dict(full=(identity-rho,rho),
                **four_block_panels(rho,slice(1,3)))
    for name,(left,right) in panels.items():
        forward=[pair_spectrum(psi,energy,left,right,p) for p in partners]
        reverse=[pair_spectrum(psi,energy,right,left,p) for p in partners]
        for i,time in enumerate(times):
            # References deliberately differ at each node and between the
            # endpoints. They are REAL, chosen for the ACTUAL build times.
            lref=energy.min() if time.real>=0 else energy.max()
            rref=energy.max() if time.real>=0 else energy.min()
            origin=-.37+.023*i
            # Physical direct owner: lower=-s -> -conj(s),
            # upper=conj(s) -> s. Only the left faces differ.
            lower=green_from_left(right,-time.conjugate(),rref)
            upper=green_from_left(left,time,lref)
            lower_r=np.fft.ifft(lower,axis=0,norm='ortho')
            upper_r=np.fft.ifft(upper,axis=0,norm='ortho')
            # Native mode11 v(R)=conj(upper_ifft)*lower_ifft.
            v=np.einsum('Rsmtn,Rsmtn->Rmn',upper_r.conjugate(),lower_r)
            fv=np.fft.fft(v,axis=0,norm='ortho')
            delta=lref-rref
            for q in range(nk):
                ahead=fv[(-q)%nk].conjugate()/np.sqrt(nk)
                behind=fv[q]/np.sqrt(nk)
                expected_f=pair_time(forward[q],[time])[0]
                expected_r=pair_time(reverse[q],[-time.conjugate()])[0]
                np.testing.assert_allclose(
                    ahead*np.exp((origin-delta)*time),
                    expected_f*np.exp(origin*time),rtol=3e-13,atol=3e-15)
                np.testing.assert_allclose(
                    behind*np.exp((origin-delta)*time.conjugate()),
                    expected_r*np.exp(origin*time.conjugate()),rtol=3e-13,atol=3e-15)
                if name=='AA' and i==1 and q==1:
                    # These mistakes are not masked by a spin trace or a
                    # stationary/passive density fixture.
                    assert np.linalg.norm(expected_f-expected_f.T)>1e-5
                    wrong=pair_time(reverse[q],[time.conjugate()])[0]
                    assert np.linalg.norm(wrong-expected_r)>1e-5

    # Independent bilinear reduction of the full upper-endpoint commutator:
    # B(I,rho)-B(rho,I) = B(I-rho,rho)-B(rho,I-rho).
    for partner in partners:
        first=pair_time(pair_spectrum(psi,energy,identity-rho,rho,partner),times)
        second=pair_time(pair_spectrum(psi,energy,rho,identity-rho,partner),times)
        np.testing.assert_allclose(first-second,
            ordered_pr_time(psi,energy,rho,partner,times),rtol=3e-13,atol=3e-15)


def test_common_paired_moments_require_swapping_endpoint_energy_powers():
    """A single product reverses its endpoint powers as well as its endpoints."""
    psi,energy,rho,partners=case(705)
    identity=np.broadcast_to(np.eye(energy.shape[1]),rho.shape)
    upper=identity-rho
    centered=energy-.5*(energy.min()+energy.max())
    upper_e=upper*centered[:,None,:]
    rho_e=rho*centered[:,None,:]
    gamma=.041
    for partner in partners:
        def at_zero(left,right):
            return pair_time(pair_spectrum(psi,energy,left,right,partner),[0.])[0]
        forward=pair_spectrum(psi,energy,upper,rho,partner)
        reverse=pair_spectrum(psi,energy,rho,upper,partner)
        first,second=commutator_moments(forward,reverse,gamma)
        # C2 uses the SUM of the two paired orientations, not the C1
        # antisymmetric combination. The common energy origin cancels.
        first_power=(at_zero(upper_e,rho)+at_zero(rho,upper_e)
                     -at_zero(upper,rho_e)-at_zero(rho_e,upper))
        np.testing.assert_allclose(first_power-1j*gamma*first,second,
                                   rtol=3e-13,atol=3e-15)
        wrong=(at_zero(upper_e,rho)-at_zero(rho,upper_e)
               -at_zero(upper,rho_e)+at_zero(rho_e,upper))
        assert np.linalg.norm(wrong-first_power)>1e-4


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


def test_crossing_primitives_preserve_each_height_value_and_physical_square_derivative(monkeypatch):
    import minimax
    from gw.response_bank import full_density_sample_rules
    original=minimax.response_bank_rule; captured=[]
    def record(*a,**kw):
        result=original(*a,**kw);captured.append((a,kw,result));return result
    monkeypatch.setattr(minimax,'response_bank_rule',record)
    gamma=.031;sites=np.array([-.17+.19j,.23+.57j]);z=sites+1j*gamma
    interval=(-.95,1.11);center=sum(interval)/2
    rules=full_density_sample_rules(np.array([[-1.,1.]]),sites,gamma_ry=gamma,
        interval_ry=interval,paired=True,tolerance=1e-8)
    assert all(r['construction']=='continuum_crossing_real_time' for r in rules)
    d=np.linspace(*interval,137)
    got=np.zeros((2,len(z),len(d)),complex)
    for r in rules:
        phase=np.exp(-(d[None,:]-r['reference_ry'])*r['times'][:,None])
        got[0,r['members']]+=r['weights'][0]@phase
        got[1,r['members']]+=r['weights'][1]@phase.conjugate()
    expected=np.stack((1/(z[:,None]-d),-1/(z[:,None]+d)))
    assert np.max(abs(got-expected)*z.imag[None,:,None])<1e-8
    (points,_),kwargs,raw=captured[0]
    ratio=points.imag/points.imag.min();chain=abs(points/np.tile(z,2))
    assert kwargs['rel_tol']<=1e-8/max(ratio.max(),(ratio**3*chain).max())
    n=len(z);dv=raw['projection_derivative'];time=raw['t']
    derivative=np.stack((-1j*dv[:n]*(points[:n]/z)[:,None],
                         1j*dv[n:]*(points[n:]/z)[:,None]))
    phase=np.exp(-1j*time[:,None]*(d[None,:]-center))
    actual=np.stack((derivative[0]@phase,derivative[1]@phase.conjugate()))
    exact=np.stack((-1/(2*z[:,None]*(z[:,None]-d)**2),
                    1/(2*z[:,None]*(z[:,None]+d)**2)))
    assert np.max(abs(actual-exact)*z.imag[None,:,None]**3)<1e-8


def test_native_crossing_paired_rules_fft_match_literal_noncommuting_frequency():
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
    left,right=four_block_panels(rho,slice(1,3))['AA']
    box=pair_interval(energy,[1,2],[1,2])
    bounds=tuple(box[k] for k in ('left_min','left_max','right_min','right_max'))
    gamma=.031;sites=np.array([-.17+.19j,.23+.57j])
    rules=full_density_sample_rules(energy,sites,gamma_ry=gamma,tolerance=1e-8,
        interval_ry=(box['lo']-.07,box['hi']+.11),paired=True)
    got=np.zeros((nk,len(sites),nmu,nmu),complex)
    with fft.xla_reference():
        for rule in rules:
            stream=full_density_response_stream(mesh,(3,1,1),(nk,nb,nmu,ns),q_ids=[0,1,2],
                n_outputs=len(rule['members']),pair_bounds_ry=bounds,rule_reference_ry=rule['reference_ry'])
            got[:,rule['members']]+=np.asarray(stream(rule['times'],rule['weights'],
                endpoint(pm,left),psi,endpoint(pm,right),energy))/np.sqrt(nk)
    expected=[]
    for partner in partners:
        f=pair_spectrum(psi,energy,left,right,partner);r=pair_spectrum(psi,energy,right,left,partner)
        expected.append(pair_frequency(f,sites,gamma)-pair_frequency(r,sites,gamma))
    np.testing.assert_allclose(got,np.stack(expected),rtol=2e-7,atol=2e-9)


def test_paired_long_real_lag_masks_inactive_deep_states_before_exponentiation():
    import jax
    from jax.sharding import Mesh
    from ffi import fft as fft
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream
    jax.config.update('jax_enable_x64',True)
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    psi,energy,rho,partners=case();nk,nb,ns,nmu=psi.shape
    energy[:,3]=-1000. # This completely inactive column must never form exp(+1e6).
    left,right=four_block_panels(rho,slice(1,3))['FA']
    box=pair_interval(energy,[1,2],[0]);ref=box['lo']
    bounds=tuple(box[k] for k in ('left_min','left_max','right_min','right_max'))
    times=np.array([1000.+0j]);weights=np.array([[[1.]],[[0.]]])
    pm=psi.transpose(0,2,3,1);endpoint=density_left_endpoint(mesh,pm.shape)
    with fft.xla_reference():
        stream=full_density_response_stream(mesh,(3,1,1),(nk,nb,nmu,ns),q_ids=[0,1,2],
            n_outputs=1,pair_bounds_ry=bounds,rule_reference_ry=ref)
        got=np.asarray(stream(times,weights,endpoint(pm,left),psi,endpoint(pm,right),energy))/np.sqrt(nk)
    assert np.isfinite(got).all()
    expected=[]
    for partner in partners:
        gap,res=pair_spectrum(psi,energy,left,right,partner)
        live=np.any(res!=0,axis=(1,2))
        expected.append(np.einsum('i,ijk->jk',np.exp(-(gap[live]-ref)*times[0]),res[live])[None])
    np.testing.assert_allclose(got,np.stack(expected),rtol=5e-12,atol=5e-14)


def test_extra_eta_verification_keeps_all_original_ids_and_only_active_span_extensions():
    from gw.shared_pole_screening import active_eta_verification_sites
    # Cross-k frozen overlaps deliberately invalidate a strict denominator-
    # sign argument, but remain inside the global propagated active span.
    energy=np.array([[-.02,0.,.1,.12],[.78,.8,.9,.92]])
    active=slice(1,3);eta=.07
    native=np.array([.2+.3j,4.+.3j,0.+.8j,.4+.3j,.2+.3j,-.4+.3j])
    got,receipt=active_eta_verification_sites(native,energy,active,eta)
    np.testing.assert_array_equal(got[:len(native)].view(np.uint64),native.view(np.uint64))
    extra=got[len(native):];span=np.ptp(energy[:,active])
    assert receipt['native_count']==len(native) and receipt['native_ids_unchanged']
    assert receipt['extra_count']==len(extra) and receipt['active_transition_span_ry']==span
    assert set(extra)=={complex(x,eta) for x in (0.,.25*span,.5*span,.75*span,span)}
    assert np.all((extra.real>=0)&(extra.real<=span)) and np.all(extra.imag==eta)
    assert 4.+.3j in got and 4.+1j*eta not in got # native far-frequency held point survives
    assert 'not a Sigma error certificate' in receipt['scope']
    # Every possible frozen wrong-sign real resonance is covered even when
    # its band is outside the four-state collision intermediate window.
    ext=energy[:,active].reshape(-1)
    delta_occ=ext[:,None]-energy[:,0][None,:]
    delta_empty=ext[:,None]-energy[:,-1][None,:]
    assert np.any(delta_occ<0) and np.any(delta_empty>0)
    assert np.all(delta_occ[delta_occ<0]>=-span)
    assert np.all(delta_empty[delta_empty>0]<=span)


def test_extra_eta_verification_uses_current_energies_and_exact_native_deduplication():
    from gw.shared_pole_screening import active_eta_verification_sites
    eta=.125;native=np.array([0.+1j*eta,.5+1j*eta,20.+.4j])
    original=native.copy()
    energy=np.array([[-10.,0.,1.,8.]])
    first,_=active_eta_verification_sites(native,energy,slice(1,3),eta,native_fit_count=2)
    assert set(first[len(native):])=={.25+1j*eta,.75+1j*eta,1.+1j*eta}
    energy[:,2]=2.
    second,receipt=active_eta_verification_sites(native,energy,slice(1,3),eta,native_fit_count=2)
    assert receipt['active_transition_span_ry']==2.
    assert set(second[len(native):])=={1.+1j*eta,1.5+1j*eta,2.+1j*eta}
    np.testing.assert_array_equal(native,original)
    for bad_eta in (0.,-1.,np.nan):
        with pytest.raises(ValueError,match='shared_general_verification'):
            active_eta_verification_sites(native,energy,slice(1,3),bad_eta)


def test_eta_training_and_independent_holds_keep_native_roles_and_pencil_nodes_disjoint():
    from gw.shared_pole_screening import active_eta_verification_sites
    energy=np.array([[-10.,0.,1.,8.]])
    eta=.1
    # Native held midpoint CANNOT be silently promoted to training. Existing
    # fit zero is reused; existing held quarter stays held; all IDs survive.
    native=np.array([0.+.1j,.8+.3j,.5+.1j,.25+.1j,1.+.3j])
    sites,receipt=active_eta_verification_sites(native,energy,slice(1,3),eta,native_fit_count=2)
    np.testing.assert_array_equal(sites[:len(native)],native)
    fit=receipt['fit_positions'];held=receipt['held_positions']
    assert fit==[0,1,5] and held==[2,3,4,6]
    assert sites[5]==1.+.1j and sites[6]==.75+.1j
    assert any(c['role']=='fit' and c['fraction']==.5 for c in receipt['collisions'])
    assert not set(fit)&set(held) and not set(sites[fit])&set(sites[held])
    # Independently mirror exactly the builder's alternating two-sided
    # conditions: mirrors go to the opposite side, and a purely imaginary
    # support must not be duplicated into BOTH sides.
    left=[];right=[]
    for i,z in enumerate(sites[fit]):
        own,other=(right,left) if i%2 else (left,right)
        own.append(z)
        if z.real!=0:other.append(-z.conjugate())
    assert min(abs(a-b) for a in left for b in right)>0


def test_collapsed_active_span_does_not_invent_independent_eta_hold():
    from gw.shared_pole_screening import active_eta_verification_sites
    native=np.array([0.+.1j]);energy=np.array([[-1.,.3,.3,2.]])
    sites,receipt=active_eta_verification_sites(native,energy,slice(1,3),.1,native_fit_count=1)
    np.testing.assert_array_equal(sites,native)
    assert receipt['collapsed_span'] and receipt['fit_positions']==[0]
    assert receipt['held_positions']==[]
    assert len([c for c in receipt['collisions'] if c['role']=='held'])==2


def test_structural_endpoint_band_ranges_match_literal_nonstationary_all_q():
    import jax
    from jax.sharding import Mesh
    from ffi import fft as fft
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream
    jax.config.update('jax_enable_x64',True)
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    psi,energy,rho,partners=case();nk,nb,ns,nmu=psi.shape
    # Frozen exactly0/1 spectators vary with k. FF then includes an empty
    # electron endpoint at k2 and an empty hole endpoint at k1. This stresses
    # separate endpoint/per-parent ranges, not a shared global band slice.
    rho[1,0,0]=0.;rho[2,3,3]=1.
    pm=psi.transpose(0,2,3,1);endpoint=density_left_endpoint(mesh,pm.shape)
    times=np.array([.13+.21j,-.11+.29j,.03-.17j])
    weights=np.zeros((2,2*len(times),len(times)),complex)
    weights[0,:len(times)]=np.eye(len(times));weights[1,len(times):]=np.eye(len(times))
    for left,right in four_block_panels(rho,slice(1,3)).values():
        ranges=[];bounds=[]
        for panel in (left,right):
            support=np.any(panel!=0,axis=1) # M[a,b]: exact support of RIGHT phase index b
            edges=np.zeros((2,nk),np.int32)
            for k in range(nk):
                live=np.flatnonzero(support[k])
                if len(live):edges[:,k]=[live[0],live[-1]+1]
            ranges.append(edges);bounds.extend([energy[support].min(),energy[support].max()])
        expected=[]
        for partner in partners:
            f=pair_spectrum(psi,energy,left,right,partner);r=pair_spectrum(psi,energy,right,left,partner)
            expected.append(np.concatenate((pair_time(f,times),pair_time(r,-times.conjugate()))))
        # A nonzero gauge reference verifies cancellation independently of
        # zero-column skipping, for both signs of Re(t) and both orientations.
        reference=.123
        expected=np.stack(expected)*np.concatenate((np.exp(reference*times),
                                                   np.exp(reference*times.conjugate())))[None,:,None,None]
        values=[]
        with fft.xla_reference():
            for structural in (None,np.asarray(ranges)):
                stream=full_density_response_stream(mesh,(3,1,1),(nk,nb,nmu,ns),q_ids=[0,1,2],
                    n_outputs=2*len(times),pair_bounds_ry=bounds,rule_reference_ry=reference,
                    pair_band_ranges=structural)
                values.append(np.asarray(stream(times,weights,endpoint(pm,left),psi,
                    endpoint(pm,right),energy))/np.sqrt(nk))
        for value in values:
            np.testing.assert_allclose(value,expected,rtol=4e-12,atol=5e-14)
        np.testing.assert_allclose(values[1],values[0],rtol=4e-12,atol=5e-14)
