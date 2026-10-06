"""P4 planted atomic stage versus an independent dense band-pair oracle."""
from pathlib import Path
import json
import os


def _blocked_point_sample_reference(mesh,pc,bc,npoint,g_block,fft_points):
    """Incumbent unhoisted formula, kept only as an independent test oracle."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map

    def sample(source,K,points,live):
        ng = source.shape[-1]
        width = min(g_block,ng)
        blocks = (ng+width-1)//width
        source = jnp.pad(source,((0,0),(0,0),(0,0),(0,blocks*width-ng)))
        K = jnp.pad(K,((0,0),(0,blocks*width-ng),(0,0)))
        zero = jnp.zeros((pc,bc,4,npoint),jnp.complex128)
        z0 = jnp.int32(0)
        def add(acc,step):
            c = jax.lax.dynamic_slice(source,(z0,z0,z0,step*width),(pc,bc,4,width))
            k = jax.lax.dynamic_slice(K,(z0,step*width,z0),(pc,width,3))
            phase = jnp.exp(1j*jnp.einsum('pgi,mi->pgm',k,points))
            return acc+jnp.einsum('pnsg,pgm->pnsm',c,phase),None
        partial = jax.lax.scan(add,zero,jnp.arange(blocks,dtype=jnp.int32),unroll=1)[0]
        total = jax.lax.psum(partial,('x','y'))/np.sqrt(fft_points)
        total *= live[None,None,None,:]
        bx,my = bc//mesh.shape['x'],npoint//mesh.shape['y']
        return jax.lax.dynamic_slice(total,(z0,jax.lax.axis_index('x')*bx,z0,jax.lax.axis_index('y')*my),
                                     (pc,bx,4,my))
    return jax.jit(shard_map(sample,mesh=mesh,
        in_specs=(P(None,None,None,('x','y')),P(None,('x','y'),None),P(),P()),
        out_specs=P(None,'x',None,'y'),check_vma=False))


def check_augmentation_stage(runtime, *, overlap=False, fractional=False, band_chunk=4, onsite_cross=False,
                             supplied_artifact=False, prepared_projection=False, occupied_weight=1.):
    from types import SimpleNamespace
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.bispinor_init import lift_to_4spinor
    from common.gpu_utils import set_device_budget_gb
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw import isdf_augmentation as aug
    from psp.atomic_reconstruction import log_radial_weights
    from psp.augmented_samples import atomic_projection_table, atomic_image_geometry
    from psp.augmentation_spinors import evaluate_normalized_delta
    from psp.reconstruction_overlap import atomic_delta_overlap_table,atomic_delta_gram
    from common.psi_G_store import ParentPsiG
    from symmetry_maps import q_negation_index

    mesh = runtime.mesh
    assert mesh.size == 4
    set_device_budget_gb(32.)
    rng = np.random.default_rng(20565)
    npar, nb, ng = 3, 8, 16
    grid, kgrid = (8,8,8), (3,1,1)
    lattice, reciprocal = 12*np.eye(3), 2*np.pi/12*np.eye(3)
    # Boundary-crossing atomic nodes at nonzero q are essential: wrapping
    # them silently before angular projection loses exp(+2pi i q.L).
    center = np.array([[.001,.001,.001]])
    kfrac = np.zeros((npar,3))
    kfrac[:,0] = np.arange(npar)/3
    g = np.array([[a,b,c] for a in (-1,0,1) for b in (0,1) for c in (-1,0,1)])[:ng]
    gv = np.tile(g[None],(npar,1,1))
    K = (gv+kfrac[:,None]) @ reciprocal
    pauli = (rng.normal(size=(npar,nb,2,ng))+1j*rng.normal(size=(npar,nb,2,ng)))/np.sqrt(2*ng)
    valid_g = ng-2
    pauli[...,valid_g:] = 0.
    logical_bands = 6
    pauli[:,logical_bands:] = 0.
    source = np.asarray(lift_to_4spinor(jnp.asarray(pauli),jnp.asarray(gv),jnp.asarray(kfrac),
                        jnp.asarray(reciprocal),representation='normalized_rkb'))
    if prepared_projection and not overlap:
        raise ValueError('prepared projections require the complete overlap fixture')
    mu_indices = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                           [7,0,0],[0,7,0],[0,0,7],[1,1,1]],np.int32)
    spin = np.eye(4,dtype=np.complex128)
    sym = SimpleNamespace(sym_matrices=np.eye(3,dtype=np.int32)[None],translations=np.zeros((1,3)),
        R_cart=np.eye(3)[None],
        irr_idx_k=np.arange(npar,dtype=np.int32),sym_idx_k=np.zeros(npar,dtype=np.int32),
        kirr_fullids=np.arange(npar),spinor_action=lambda rows,nspinor: np.tile(spin[None],(len(rows),1,1)),
        parent_k_domain='planted',q_irr_full_idx=np.arange(npar,dtype=np.int32),
        kvecs_asints=np.array([[0,0,0],[1,0,0],[2,0,0]],np.int32))
    mu_coordinates = mu_indices/np.asarray(grid)
    if fractional:
        mu_coordinates = mu_coordinates+np.asarray([.013,.021,-.017])
    centroid_coordinates = mu_coordinates if fractional else mu_indices
    plan = build_centroid_k_unfold_plan(sym,centroid_coordinates,grid,mesh,nspinor=4,parent_k_frac=kfrac,
        coordinate_kind='fractional' if fractional else 'fft_indices')
    radius = np.geomspace(1e-5,.7,65)
    dr = log_radial_weights(radius)
    ps_R = np.exp(-2*radius)[:,None]
    delta_R = (.2*np.exp(-20*radius)*np.maximum(1-(radius/.7)**2,0)**4)[:,None]
    data = dict(r=radius,weights_dr=dr,ps_u=radius[:,None]*ps_R,l=np.array([0]),kappa=np.array([-1]),
                delta_R=delta_R,delta_u=radius[:,None]*delta_R)
    artifact = dict(tables={47:data},identity='planted-stage',
        radial=dict(radius=[.06,.16,.34,.61],weights_dr=[.1,.2,.3,.2],support_radius=1.1),
        angular=dict(lebedev_order=5,lmax=2,orthogonality_tolerance=2e-12),
        cache=dict(momentum_max=60.,momentum_points=512,radius_max=2.,radius_points=512,
                   taper_start=.7,tail_relative_tolerance=1.),
        runtime=dict(parent_chunk=1,band_chunk=band_chunk,radial_packet=2,g_block=3))
    fitting_bands = 4 if overlap else logical_bands
    if overlap:
        artifact['overlap'] = dict(mode='full_wfn_lowdin',bands=logical_bands)
        artifact['radial'].update(interpolation_degree=3,quadrature_order=16)
    if onsite_cross:
        artifact['charge_metric'] = dict(smooth_neutral_cross='onsite')
        artifact['radial'].update(interpolation_degree=3,quadrature_order=16)
    # The fixture supplies already authenticated tables; real sidecar
    # provenance has separate source/payload negative tests. This oracle
    # targets the complete distributed stage rather than regenerating ONCV.
    original = aug.read_augmentation_manifest
    read_count = 0
    def read_fixture(directory):
        nonlocal read_count
        read_count += 1
        if supplied_artifact:
            raise AssertionError('the same authenticated artifact must not be reread')
        return artifact
    aug.read_augmentation_manifest = read_fixture
    artifact_argument = {'artifact':artifact} if supplied_artifact else {}
    meta = SimpleNamespace(nspinor=4,cell_volume=12**3,n_rtot=np.prod(grid),fft_grid=grid,
                           nk_tot=3,kgrid=kgrid,b_id_4_user=fitting_bands)
    wfn = SimpleNamespace(alat=1.,avec=lattice,blat=1.,bvec=reciprocal,
                          atom_crys=center,atom_types=np.array([47]),gvecs=lambda k: gv,
                          ngk_valid=lambda k: np.full(npar,valid_g),nbands=logical_bands)
    cfg = SimpleNamespace(paths=SimpleNamespace(atomic_reconstruction_dir='planted'))
    put = lambda a,s: device_put_process_local(np.asarray(a),NamedSharding(mesh,s))
    smooth_mu = np.einsum('pnsg,pgm->pnsm',source,
                np.exp(1j*np.einsum('pgi,mi->pgm',K,mu_coordinates @ lattice)))/np.sqrt(meta.n_rtot)
    packed_mu = plan.layout.axis.pack_host(smooth_mu,axis=3,fill_value=0.)
    parent_faces = (put(packed_mu[:,:logical_bands],P(None,'x',None,'y')),
                    put(packed_mu[:,:logical_bands].transpose(0,2,3,1),P(None,None,'x','y')))
    source_device = put(source,P(None,None,None,('x','y')))
    caches,_ = aug._normalized_caches(artifact['tables'],artifact['cache'],1.1)
    artifact['normalized_caches'] = caches
    served_D = served_B = None
    from isdf import atomic_moments
    original_binding = atomic_moments.raw_parent_moment_binding
    if overlap:
        from isdf.atomic_moments import build_served_overlap_cache,served_overlap_table
        served_cache = build_served_overlap_cache(caches[47],support_radius=1.1,
            momentum_max=float(np.linalg.norm(K,axis=-1).max())*(1+1e-12),momentum_points=1025)
        served_D = np.stack([np.einsum('isg,nsg->ni',served_overlap_table(served_cache,K[p],
            center_cart=center[0] @ lattice,cell_volume=meta.cell_volume),source[p]) for p in range(npar)])
        served_B = served_cache['B']
        fixture_binding = {'oracle':'planted-full-four-spinor-source'}
        artifact['served_moment_caches'] = {47:served_cache}
        artifact['served_moments'] = {'species_sha256':{'47':'planted-served-source'}}
        artifact['raw_parent_moments'] = dict(atom_D=[served_D[:,:logical_bands]],
            metadata={'binding':fixture_binding})
        if prepared_projection:
            coefficients_raw = np.stack([np.einsum('isg,nsg->ni',atomic_projection_table(data,K[p],
                center_cart=center[0] @ lattice,cell_volume=meta.cell_volume,
                normalized_rkb_source=True),source[p,:,:2]) for p in range(npar)])
            artifact['raw_parent_moments']['atom_C'] = [coefficients_raw[:,:logical_bands]]
            artifact['raw_parent_projection_binding'] = {'oracle':'same-normalized-source-and-duals'}
        def planted_binding(bound_wfn,**inputs):
            assert bound_wfn is wfn and inputs['physical_bands'] == logical_bands
            assert np.array_equal(inputs['k_parent_frac'],kfrac)
            assert np.array_equal(inputs['gvecs'],gv)
            assert np.array_equal(inputs['ngk_valid'],np.full(npar,valid_g))
            assert np.array_equal(inputs['centers_cart'],center @ lattice)
            assert np.array_equal(inputs['atom_types'],np.array([47]))
            assert inputs['cell_volume'] == meta.cell_volume
            assert inputs['served_cache_sha256_by_species'] == {'47':'planted-served-source'}
            assert inputs.get('projection_binding') == artifact.get('raw_parent_projection_binding')
            return fixture_binding
        # The distributed stage consumes authenticated preparation output;
        # actual WFN source/owner bindings have independent artifact tests.
        atomic_moments.raw_parent_moment_binding = planted_binding
    phase_K = K.copy();phase_K[:,valid_g:] = 0.
    K_dev,points_dev,live_dev = (put(phase_K,P(None,('x','y'),None)),
        put(mu_coordinates @ lattice,P()),put(np.ones(len(mu_coordinates)),P()))
    blocked = _blocked_point_sample_reference(mesh,npar,nb,len(mu_coordinates),3,meta.n_rtot)
    hoisted = aug._point_samples_kernel(mesh,npar,nb,len(mu_coordinates),3,meta.n_rtot)
    phase = aug._point_phase_kernel(mesh)(K_dev,points_dev)
    old_sample = np.asarray(gather_to_host(blocked(source_device,K_dev,points_dev,live_dev)))
    new_sample = np.asarray(gather_to_host(hoisted(source_device,phase,live_dev)))
    phase_error = float(np.linalg.norm(new_sample-old_sample)/np.linalg.norm(old_sample))
    assert phase_error < 2e-14,phase_error
    # Linear public-column rotation must precede the smaller DFT without
    # losing any full-window input row or exposing real source columns as
    # public ghost bands. The independent reference samples all eight rows.
    probe_factor = (rng.normal(size=(npar,nb,nb))+1j*rng.normal(size=(npar,nb,nb)))/np.sqrt(nb)
    rotated_g = aug._band_rotation_kernel(mesh,npar,'source',4,0,3)(source_device,put(probe_factor,P()))
    sampled_public = np.asarray(gather_to_host(aug._smooth_point_faces(
        rotated_g,phase_K,mu_coordinates,np.ones(len(mu_coordinates),dtype=bool),mesh,
        lattice=lattice,parent_chunk=npar,band_chunk=4,g_block=3,fft_points=meta.n_rtot)))
    sampled_then_rotated = np.einsum('pmn,pmsu->pnsu',probe_factor[...,:4],old_sample)
    sampled_then_rotated[:,3:] = 0.
    public_sample_error = float(np.linalg.norm(sampled_public-sampled_then_rotated)
                               /np.linalg.norm(sampled_then_rotated))
    assert public_sample_error < 2e-14,public_sample_error
    assert np.max(abs(sampled_public[:,3:])) == 0.,'public pad must not expose source band four'
    placement_error = 0.
    if overlap:
        from psp.augmented_samples import build_projection_radial_cache
        from psp.reconstruction_overlap import build_delta_radial_cache
        probe_K = K.copy()
        probe_K[:,valid_g:] = 0.
        maxK = float(np.linalg.norm(probe_K,axis=-1).max())*(1+1e-12)
        cache_controls = dict(momentum_max=maxK,momentum_points=1025)
        for delta_overlap,factory,cache in (
            (False,atomic_projection_table,build_projection_radial_cache(data,**cache_controls)),
            (True,atomic_delta_overlap_table,build_delta_radial_cache(data,**cache_controls))):
            geometry = dict(center_cart=center[0] @ lattice,cell_volume=meta.cell_volume)
            reference = np.asarray([factory(data,kv,normalized_rkb_source=True,radial_cache=cache,**geometry)
                                    for kv in probe_K])
            placed = aug._atomic_fourier_table(data,probe_K,geometry,mesh,
                                              radial_cache=cache,delta_overlap=delta_overlap)
            error = float(np.max(abs(gather_to_host(placed)-reference)))
            assert error < 3e-14,error
            placement_error = max(placement_error,error)
    parent_psi = ParentPsiG(source_device,None,None,kfrac,(0,logical_bands),None)
    weight_policy = {'occupied_stop':2,'occupied_weight':occupied_weight}
    original_fourier = aug._atomic_fourier_table
    if prepared_projection:
        def reject_live_projection(*args,**kwargs):
            raise AssertionError('authenticated prepared C must not rebuild the Fourier projection')
        aug._atomic_fourier_table = reject_live_projection
    try:
        faces,state = aug.prepare_augmentation(wfn=wfn,sym=sym,meta=meta,cfg=cfg,mesh_xy=mesh,
            plan=plan,centroid_indices=centroid_coordinates,parent_psi=parent_psi,
            parent_faces=None if fractional else parent_faces,
            band_range_left=(0,2),band_range_right=(1,fitting_bands),write_ibz_only=False,
            public_band_range=(0,logical_bands),charge_fit_weights=weight_policy,**artifact_argument)
        assert read_count == (0 if supplied_artifact else 1)
        if overlap:
            import pytest
            assert state['prepared_served_overlap_host_bytes_per_process'] == served_D[:,:logical_bands].nbytes
            if prepared_projection:
                assert state['atomic_projection_source'] == 'prepared_full_window_v2'
                assert state['prepared_atomic_projection_host_bytes_per_process'] == coefficients_raw[:,:logical_bands].nbytes
            assert 'monopole_rhs' not in state, 'served overlap admission must not enable monopole enrichment'
            saved_binding = artifact['raw_parent_moments']['metadata']['binding']
            artifact['raw_parent_moments']['metadata']['binding'] = {'oracle':'different-source'}
            try:
                with pytest.raises(ValueError,match='raw served-moment cache disagrees'):
                    aug.prepare_augmentation(wfn=wfn,sym=sym,meta=meta,cfg=cfg,mesh_xy=mesh,
                        plan=plan,centroid_indices=centroid_coordinates,parent_psi=parent_psi,
                        parent_faces=None if fractional else parent_faces,
                        band_range_left=(0,2),band_range_right=(1,fitting_bands),write_ibz_only=False,
                        public_band_range=(0,logical_bands),**artifact_argument)
            finally:
                artifact['raw_parent_moments']['metadata']['binding'] = saved_binding
        memory_refusal = False
        if not overlap and not fractional:
            import pytest
            from common import gpu_utils
            workspace = state['local_rhs_workspace_bytes_per_rank']
            added = workspace['total']-workspace['full_q_scalar']
            budget = state['resident_estimate_bytes_per_rank']-.5*added
            # The previous single-output ledger would admit this budget;
            # the actual live endpoint/quarter panels must now refuse it.
            assert state['resident_estimate_bytes_per_rank']-added < budget
            old_budget,old_warn = gpu_utils.device_budget_bytes,gpu_utils.warn_over_budget
            warnings = []
            gpu_utils.device_budget_bytes = lambda: budget
            gpu_utils.warn_over_budget = lambda *args: warnings.append(args)
            try:
                with pytest.raises(ValueError,match='augmentation manifest requires'):
                    aug.prepare_augmentation(wfn=wfn,sym=sym,meta=meta,cfg=cfg,mesh_xy=mesh,
                        plan=plan,centroid_indices=centroid_coordinates,parent_psi=parent_psi,
                        parent_faces=parent_faces,band_range_left=(0,2),
                        band_range_right=(1,fitting_bands),write_ibz_only=False,
                        public_band_range=(0,logical_bands),**artifact_argument)
                assert len(warnings) == 1 and warnings[0][1] > budget
                memory_refusal = True
            finally:
                gpu_utils.device_budget_bytes,gpu_utils.warn_over_budget = old_budget,old_warn
    finally:
        aug.read_augmentation_manifest = original
        aug._atomic_fourier_table = original_fourier
        atomic_moments.raw_parent_moment_binding = original_binding
    got_faces = tuple(np.asarray(gather_to_host(x)) for x in faces)
    got_rhs = np.asarray(gather_to_host(state['rhs']))
    assert np.array_equal(np.asarray(gather_to_host(source_device)),source), 'smooth source changed/donated'
    coefficients = np.stack([np.einsum('isg,nsg->ni',atomic_projection_table(data,K[p],
        center_cart=center[0] @ lattice,cell_volume=meta.cell_volume,normalized_rkb_source=True),source[p,:,:2])
        for p in range(npar)])
    if overlap:
        D,B = served_D,served_B
        S = np.einsum('pnsg,pmsg->pnm',source.conj(),source)
        cross = np.einsum('pni,pmi->pnm',D.conj(),coefficients)
        S += cross+cross.conj().transpose(0,2,1)+np.einsum('pni,ij,pmj->pnm',coefficients.conj(),B,coefficients)
        eigenvalue,eigenvector = np.linalg.eigh(S[:,:logical_bands,:logical_bands])
        A = (eigenvector*eigenvalue[:,None,:]**-.5) @ eigenvector.conj().transpose(0,2,1)
        A_full = np.tile(np.eye(nb,dtype=complex)[None],(npar,1,1))
        A_full[:,:logical_bands,:logical_bands] = A
        source = np.einsum('pmn,pmsg->pnsg',A_full,source)
        coefficients = np.einsum('pmn,pmi->pni',A_full,coefficients)
        source[:,fitting_bands:] = 0.
        coefficients[:,fitting_bands:] = 0.
        rotated_source = gather_to_host(state['parent_psi'].psi_G)
        assert np.max(abs(rotated_source-source)) < 3e-13
        assert state['parent_psi'].band_range == (0,logical_bands)
        assert state['parent_psi'].faces is None
        assert np.max(abs(rotated_source[:,fitting_bands:])) == 0., 'public face and G transport pads must be zero'
        assert np.max(abs(state['overlap_receipt']['inverse_sqrt']-A_full)) < 3e-13
        assert np.max(state['overlap_receipt']['factor_isometry_error']) < 3e-13
        assert state['overlap_receipt']['overlap_operator'] == 'actual_served_four_spinor'
        assert np.max(abs(state['overlap_receipt']['delta_overlaps'][0]-D)) < 3e-14
        assert np.array_equal(state['overlap_receipt']['atomic_delta_grams'][0],B)
        assert np.max(abs(A[:,:fitting_bands,fitting_bands:])) > .01, 'full-WFN factor must mix discarded fitting bands'
        # A finite fitting-only factor is a deliberately different model.
        e,v = np.linalg.eigh(S[:,:fitting_bands,:fitting_bands])
        partial = (v*e[:,None,:]**-.5) @ v.conj().transpose(0,2,1)
        assert np.max(abs(A[:,:fitting_bands,:fitting_bands]-partial)) > 1e-3
    scale = np.sqrt(meta.cell_volume/meta.n_rtot)
    def reconstruct(points):
        ps = np.einsum('pnsg,pgm->pnsm',source,
                      np.exp(1j*np.einsum('pgi,mi->pgm',K,points @ lattice)))/np.sqrt(meta.n_rtot)
        relative,images = atomic_image_geometry(points,center[0],lattice)
        delta = np.zeros((2,4,len(points)),np.complex128)
        mask = np.linalg.norm(relative,axis=1) <= 1.1
        delta[...,mask] = evaluate_normalized_delta(caches[47],relative[mask])*scale
        phase = np.exp(2j*np.pi*kfrac @ images.T)
        ae = ps+np.einsum('pni,ism,pm->pnsm',coefficients,delta,phase)
        return ps,ae
    _,expected_mu = reconstruct(mu_coordinates)
    expected_mu = plan.layout.axis.pack_host(expected_mu[:,:logical_bands],axis=3,fill_value=0.)
    face_error = float(np.linalg.norm(got_faces[0]-expected_mu)/np.linalg.norm(expected_mu))
    assert face_error < 2e-13,face_error
    assert np.max(abs(got_faces[1]-got_faces[0].transpose(0,2,3,1))) < 2e-14
    directions,weights,lm,Y,_ = aug._orbit_angular_quadrature(artifact['angular'],sym.R_cart)
    radii = np.asarray(artifact['radial']['radius'])
    points = (center[0]+(radii[:,None,None]*directions[None]) @ np.linalg.inv(lattice)).reshape(-1,3)
    packet_plan = aug._packet_plan(plan,points[:2*len(directions)],None)
    compress,nf,_ = aug._compress_rhs_kernel(mesh,packet_plan,1,len(lm),2,Y.conj()*weights)
    select,_ = aug._rhs_storage_kernels(mesh,np.asarray([2,0]),4,1,len(lm),4,2,nf)
    probe = (rng.normal(size=(3,plan.n_centroid_packed,packet_plan.n_centroid_packed))
             +1j*rng.normal(size=(3,plan.n_centroid_packed,packet_plan.n_centroid_packed)))
    probe = put(probe,P(None,'x','y'))
    before = np.asarray(gather_to_host(compress(select(probe))))
    after = np.asarray(gather_to_host(select(compress(probe))))
    q_selection_error = float(np.linalg.norm(before-after)/np.linalg.norm(after))
    assert q_selection_error < 2e-14,q_selection_error
    assert np.max(abs(before[2:])) == 0.
    ps,ae = reconstruct(points)
    radial_moment = np.asarray(artifact['radial']['weights_dr'])*radii**2
    if overlap or onsite_cross:
        from isdf.augmentation import radial_coulomb_metric_interpolated
        radial_moment = radial_coulomb_metric_interpolated(radii,np.array([0]),support_radius=1.1,
            interpolation_degree=3,quadrature_order=16)['moments'][0]
    physical_weights = radial_moment[:,None]*weights[None,:]
    norm_reference = np.einsum('pnsm,m->pn',abs(ae)**2-abs(ps)**2,physical_weights.reshape(-1))*meta.n_rtot/meta.cell_volume
    norm_result = np.asarray(gather_to_host(state['orbital_norm_change_estimate']))
    norm_error = float(np.max(abs(norm_reference-norm_result)))
    assert norm_error < 2e-14,norm_error
    dense = np.zeros((3,expected_mu.shape[-1],len(points)),np.complex128)
    dense_smooth = np.zeros_like(dense) if onsite_cross else None
    for q in range(3):
        for k in range(3):
            for m in range(2):
                for n in range(1,fitting_bands):
                    left = np.sum(expected_mu[k,m].conj()*expected_mu[(k+q)%3,n],axis=0)
                    right = np.sum(ae[k,m].conj()*ae[(k+q)%3,n]-ps[k,m].conj()*ps[(k+q)%3,n],axis=0)
                    pair_weight = (occupied_weight if m < 2 else 1.)*(occupied_weight if n < 2 else 1.)
                    dense[q] += pair_weight*left.conj()[:,None]*right[None]
                    if onsite_cross:
                        smooth_right = np.sum(ps[k,m].conj()*ps[(k+q)%3,n],axis=0)
                        dense_smooth[q] += pair_weight*left.conj()[:,None]*smooth_right[None]
    dense += dense[q_negation_index(kgrid)].conj()
    expected_rhs = np.einsum('qmrp,hp->qmhr',dense.reshape(3,expected_mu.shape[-1],len(radii),len(directions)),Y.conj()*weights)
    expected_rhs = expected_rhs.reshape(3,expected_mu.shape[-1],-1)
    rhs_error = float(np.linalg.norm(got_rhs[:3]-expected_rhs)/np.linalg.norm(expected_rhs))
    assert rhs_error < 3e-12,rhs_error
    assert np.max(abs(got_rhs[3])) == 0.
    smooth_rhs_error = None
    if onsite_cross:
        dense_smooth += dense_smooth[q_negation_index(kgrid)].conj()
        expected_smooth = np.einsum('qmrp,hp->qmhr',
            dense_smooth.reshape(3,expected_mu.shape[-1],len(radii),len(directions)),Y.conj()*weights)
        expected_smooth = expected_smooth.reshape(expected_rhs.shape)
        got_smooth = np.asarray(gather_to_host(state['smooth_rhs']))
        smooth_rhs_error = float(np.linalg.norm(got_smooth[:3]-expected_smooth)/np.linalg.norm(expected_smooth))
        assert smooth_rhs_error < 3e-12,smooth_rhs_error
        assert np.max(abs(got_smooth[3])) == 0.
        assert state['resident_rhs_copies'] == 4
        assert state['local_rhs_workspace_bytes_per_rank']['full_q_scalar_panels'] == 6
    else:
        assert 'smooth_rhs' not in state
        assert state['resident_rhs_copies'] == 2
        assert state['local_rhs_workspace_bytes_per_rank']['full_q_scalar_panels'] == 4
    assert np.any(points < 0.), 'fixture must cross the cell boundary'
    receipt = dict(relative_sample_error=face_error,relative_local_rhs_error=rhs_error,
        q_pad_exact=True,smooth_reciprocal_unchanged=True,multiple_global_band_tiles=True,
        unwrapped_boundary_crossing_nonzero_q=True,
        source_shape=list(source.shape),rhs_shape=list(got_rhs.shape),
        local_rhs_sharding=str(state['rhs'].sharding.spec),tail_norm=state['tail_relative_norm'])
    receipt['explicit_full_wfn_lowdin_before_public_trim'] = overlap
    receipt['fractional_mu_from_reciprocal_without_fft_faces'] = fractional
    receipt['band_chunk'] = band_chunk
    receipt['live_local_rhs_memory_budget_refused'] = memory_refusal
    receipt['q_selection_before_angular_projection_relative_error'] = q_selection_error
    receipt['hoisted_phase_vs_incumbent_blocked_dft_relative_error'] = phase_error
    receipt['post_rotation_public_dft_relative_error'] = public_sample_error
    if overlap:
        receipt['maximum_restored_served_gram_error'] = float(np.max(state['overlap_receipt']['factor_isometry_error']))
        receipt['overlap_operator'] = state['overlap_receipt']['overlap_operator']
        receipt['full_physical_bands'] = logical_bands
        receipt['fitting_bands'] = fitting_bands
        receipt['maximum_shard_native_fourier_table_error'] = placement_error
        receipt['nontrim_parent_and_dead_G_table_control'] = True
    receipt['maximum_norm_diagnostic_absolute_error'] = norm_error
    receipt['onsite_smooth_rhs_relative_error'] = smooth_rhs_error
    receipt['authenticated_artifact_reused_without_read'] = supplied_artifact
    receipt['prepared_full_window_projection_used_without_rebuild'] = prepared_projection
    receipt['occupied_endpoint_weight'] = occupied_weight
    if jax.process_index() == 0:
        print(json.dumps(receipt),flush=True)
        out = os.environ.get('AUGMENTATION_STAGE_REPORT')
        if out:
            Path(out).write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime = initialize_communicator_stack()
    def main():
        check_augmentation_stage(runtime)
        check_augmentation_stage(runtime,overlap=True)
        check_augmentation_stage(runtime,fractional=True)
        check_augmentation_stage(runtime,overlap=True,fractional=True)
        check_augmentation_stage(runtime,overlap=True,fractional=True,band_chunk=8)
        check_augmentation_stage(runtime,overlap=True,fractional=True,band_chunk=8,onsite_cross=True)
        check_augmentation_stage(runtime,overlap=True,fractional=True,band_chunk=8,onsite_cross=True,
                                 supplied_artifact=True)
        check_augmentation_stage(runtime,overlap=True,fractional=True,band_chunk=8,onsite_cross=True,
                                 supplied_artifact=True,prepared_projection=True,occupied_weight=4.)
        return 0
    run_main_and_finalize(main)
