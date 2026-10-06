"""Independent physical-state carrier controls for complete native references."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.wavefunction_bundle import (BandSlices, Wavefunctions, ParentGreenCarrier,
    physical_band_mask, build_packed_parent_green_carrier)
from gw.efermi import OccupationState, assert_fixed_n
from gw.shared_pole_recipe import (active_band_mask, support_rule_line_sites,
    bind_shared_pole_census)
from gw.response_bank import response_weights, _moment_relative_energies
from gw.w_isdf import (_occupation_support_slices, occupation_support_bandwidth,
    compute_chi0_direct_fractional)
from gw.greens_function_kernel import _weighted_tau_phases
from gw.mpa.sigma import _branches


def bundle(ghost=1e100, *, valid=True):
    energy=np.asarray([[-2.,-.8,.6,1.8,ghost,-ghost],[-1.9,-.7,.5,-ghost,ghost,-ghost]])
    mask=np.asarray([[True,True,True,True,False,False],[True,True,True,False,False,False]])
    occ=np.asarray([[1.,1.,0.,0.,0.,0.],[1.,1.,0.,0.,0.,0.]])
    slices=BandSlices.from_band_edges(0,0,2,3,6,b4_logical=6)
    return Wavefunctions(enk=jnp.asarray(energy),occ=jnp.asarray(occ),slices=slices,
        valid_kn=jnp.asarray(mask) if valid else None),mask


def meta():
    return SimpleNamespace(nk_tot=2,nspin=1,nspinor=1,nspinor_wfnfile=1,
        n_rmu=2,cell_volume=10.,b_id_4_chi_user=6)


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),("x","y"))


def test_physical_weights_extrema_and_moment_powers_ignore_ghosts():
    w,mask=bundle(1e300)
    energy,f,u,reference,census=response_weights(w,meta())
    assert np.array_equal(f[~mask],np.zeros((~mask).sum()))
    assert np.array_equal(u[~mask],np.zeros((~mask).sum()))
    assert np.array_equal((f+u)[mask],np.ones(mask.sum()))
    assert reference==np.asarray(w.enk)[mask].mean()
    assert census["energy_min_ry"]==-2. and census["energy_max_ry"]==1.8
    erel=_moment_relative_energies(w,energy,reference,census)
    for power in (1,2,3):
        values=erel**power
        assert np.isfinite(values).all()
        assert np.array_equal(values[~mask],np.zeros((~mask).sum()))
    assert census["physical_bands_by_k"]==[4,3]


@pytest.mark.parametrize("ghost",[1e6,-1e6,1e300,-1e300])
def test_fixed_step_and_fd_solve_brackets_ignore_ghost_extrema(ghost):
    w,valid=bundle(ghost)
    step=OccupationState.step(w.enk,[.5,.5],2.,state_capacity=2.,valid_kn=w.valid_kn)
    assert np.array_equal(np.asarray(step.f_kn),np.asarray(w.occ))
    assert assert_fixed_n(step,[.5,.5],state_capacity=2.)==4.
    fd=OccupationState.solve_smearing(w.enk,[.5,.5],3.3,.02,state_capacity=2.,family="fd",valid_kn=w.valid_kn)
    weights=np.broadcast_to([.5,.5],(6,2)).T[valid]
    compact=OccupationState.solve_smearing(np.asarray(w.enk)[valid,None],weights/weights.sum(),
        3.3/weights.sum(),.02,state_capacity=2.,family="fd")
    assert abs(fd.mu_ry-compact.mu_ry)<1e-12
    assert np.max(np.abs(np.asarray(fd.f_kn)[valid]-np.asarray(compact.f_kn).ravel()))<1e-12
    assert np.array_equal(np.asarray(fd.f_kn)[~valid],np.zeros((~valid).sum()))


def test_active_census_and_support_histogram_ghost_invariance():
    base,mask=bundle(10.)
    changed,_=bundle(-1e100)
    for w in (base,changed):
        m=meta();bind_shared_pole_census(w,m,occupation_state=None,trs_allowed=True,state_capacity=2.,kweights=[.5,.5])
        if w is base:expected=m.shared_pole_census
        else:
            for key in ("mu_ry","gap_ev","energy_span_ry","response_transition_span_ry","active_electrons","active_bands","borderline_bands","partial_at_mu"):
                assert m.shared_pole_census[key]==expected[key]
    assert not np.any(active_band_mask(base.enk,0.,valid_kn=base.valid_kn)[4:])
    reads=np.asarray([-8.,-4.,2.,6.,12.])
    for w in (base,changed):
        sites=support_rule_line_sites(np.asarray(w.enk),0.,.25,.5,reads,6,grid=401,valid_kn=w.valid_kn)
        compact=support_rule_line_sites(np.asarray(w.enk)[mask].reshape(1,-1),0.,.25,.5,reads,6,grid=401)
        np.testing.assert_array_equal(sites,compact)


def test_fractional_support_excludes_ghost_complement():
    w,mask=bundle()
    fs,us=_occupation_support_slices(w.occ,valid_kn=w.valid_kn)
    assert fs==slice(0,2) and us==slice(2,4)
    assert occupation_support_bandwidth(w.enk,w.occ,valid_kn=w.valid_kn)==3.8


def test_green_phase_mask_suppresses_unused_overflow():
    w,mask=bundle(1e300)
    phases=_weighted_tau_phases(w.enk,1.,mask=w.band_mask(slice(0,6)),band_weight=np.where(mask,1.,0.))
    assert np.isfinite(np.asarray(phases)).all()
    assert np.array_equal(np.asarray(phases)[~mask],np.zeros((~mask).sum()))
    np.testing.assert_allclose(np.asarray(phases)[mask],np.exp(-np.asarray(w.enk)[mask]))


def test_sigma_branch_support_and_fractional_weights_are_physical():
    w,valid=bundle()
    for state in (None,OccupationState.step(w.enk,[.5,.5],2.,state_capacity=2.,valid_kn=w.valid_kn)):
        for branch in _branches(w,np.asarray([-.3,.3]),0. if state is None else state.mu_ry,state):
            assert not np.any(np.asarray(branch.base_mask_A)[~valid])
            if branch.band_weight is not None:
                assert not np.any(np.asarray(branch.band_weight)[~valid])


def test_none_and_all_valid_default_numerical_parity():
    rng=np.random.default_rng(9984)
    energy=np.asarray([[-2.,-1.,.5,1.],[-1.9,-.8,.6,1.2]])
    occ=np.asarray([[1.,1.,0.,0.],[1.,1.,0.,0.]])
    s=BandSlices.from_band_edges(0,0,2,3,4)
    a=Wavefunctions(enk=jnp.asarray(energy),occ=jnp.asarray(occ),slices=s)
    b=Wavefunctions(enk=a.enk,occ=a.occ,slices=s,valid_kn=jnp.ones_like(a.enk,dtype=bool))
    m=meta();m.b_id_4_chi_user=4
    wa=response_weights(a,m);wb=response_weights(b,m)
    for i in range(4):np.testing.assert_array_equal(wa[i],wb[i])
    np.testing.assert_array_equal(a.band_mask(slice(0,4)),b.band_mask(slice(0,4)))
    for ctor in (OccupationState.step,):
        x=ctor(energy,[.5,.5],2.,state_capacity=2.)
        y=ctor(energy,[.5,.5],2.,state_capacity=2.,valid_kn=b.valid_kn)
        assert x.mu_ry==y.mu_ry;np.testing.assert_array_equal(x.f_kn,y.f_kn)
    x=OccupationState.solve_smearing(energy,[.5,.5],4.,.02,state_capacity=2.,family="fd")
    y=OccupationState.solve_smearing(energy,[.5,.5],4.,.02,state_capacity=2.,family="fd",valid_kn=b.valid_kn)
    assert x.mu_ry==y.mu_ry;np.testing.assert_array_equal(x.f_kn,y.f_kn)


def test_validity_survives_jit_pytree_roundtrip():
    w,mask=bundle()
    roundtrip=jax.jit(lambda a:a)(w)
    np.testing.assert_array_equal(roundtrip.valid_kn,mask)
    assert np.array_equal(np.asarray(roundtrip.band_mask(slice(0,6))),mask)
    carrier=ParentGreenCarrier(jnp.zeros((2,6,1,2)),jnp.zeros((2,1,2,6)),w.enk,w.occ,plan=None,valid_kn=w.valid_kn)
    other=jax.jit(lambda a:a)(carrier)
    np.testing.assert_array_equal(other.valid_kn,mask)


def test_parent_mask_is_derived_and_ghost_coefficients_refuse(mesh):
    w,mask=bundle()
    class Plan:
        n_parent=1;nspinor=1;n_centroid_packed=2
        def parent_rows(self,a):return a[1:2]
    plan=Plan();p=jnp.ones((1,6,1,2))*mask[1:2,:,None,None]
    q=jnp.transpose(p,(0,2,3,1))
    got=build_packed_parent_green_carrier(w,p,q,plan=plan,mesh_xy=mesh)
    np.testing.assert_array_equal(got.valid_kn,mask[1:2])
    with pytest.raises(ValueError,match="invalid parent-band coefficients"):
        build_packed_parent_green_carrier(w,p.at[0,5,0,0].set(1.),q,plan=plan,mesh_xy=mesh)


@pytest.mark.parametrize("bad",[np.ones((2,6)),np.ones((2,5),bool),np.ones((1,6),bool)])
def test_bad_validity_shape_or_dtype_refuses(bad):
    w,_=bundle()
    with pytest.raises(ValueError,match="physical_band_validity"):
        Wavefunctions(enk=w.enk,occ=w.occ,slices=w.slices,valid_kn=bad)


def test_nonzero_ghost_occupation_refuses_all_physical_consumers():
    w,mask=bundle();w.occ=w.occ.at[0,5].set(1e-20)
    for action in (lambda:physical_band_mask(w),lambda:response_weights(w,meta()),
                   lambda:_occupation_support_slices(w.occ,valid_kn=w.valid_kn)):
        with pytest.raises(ValueError,match="ghost occupations"):
            action()


@pytest.mark.parametrize("ordered",[False,True])
def test_direct_chi_and_slope_against_independent_physical_lehmann_sum(mesh,ordered):
    w,valid=bundle(1e100);rng=np.random.default_rng(3321)
    # Deliberately nonzero ghost ψ tests the kernel mask independently of
    # the input owner's stronger exact-zero coefficient guard.
    psi=rng.normal(size=(2,6,1,2))+1j*rng.normal(size=(2,6,1,2))
    w.psi_nmu=jax.device_put(psi,NamedSharding(mesh,P(None,"x",None,"y")))
    w.psi_mun=jax.device_put(psi.transpose(0,2,3,1),NamedSharding(mesh,P(None,None,"x","y")))
    state=OccupationState.step(w.enk,[.5,.5],2.,state_capacity=2.,valid_kn=w.valid_kn)
    z=np.asarray([.7+.25j,1.4+.5j]);maps=np.asarray([[0,1],[1,0]])
    actual,slope=compute_chi0_direct_fractional(w,z,meta(),mesh,occupation_state=state,
        kminq_rows=maps,nb_logical=6,ordered=ordered,with_derivative=True,pair_tile=2)
    expected=np.zeros((2,2,2,2),complex);derivative=np.zeros_like(expected)
    e=np.asarray(w.enk);f=np.asarray(state.f_kn)
    for qi,mapping in enumerate(maps):
        for k,kb in enumerate(mapping):
            for a in np.flatnonzero(valid[k]):
                for b in np.flatnonzero(valid[kb]):
                    density=psi[k,a,0]*psi[kb,b,0].conj()
                    outer=np.outer(density.conj(),density) if ordered else np.outer(density,density.conj())
                    den=e[k,a]-e[kb,b]+z
                    weight=(f[k,a]-f[kb,b])/den
                    expected[:,qi]+=weight[:,None,None]*outer
                    derivative[:,qi]+=(-weight/(2*z*den))[:,None,None]*outer
    np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(slope,derivative,rtol=2e-12,atol=2e-12)
