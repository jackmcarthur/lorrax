"""Literal dense oracles for four-current signs, endpoint order and antiunitarity."""
from types import SimpleNamespace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _gamma():
    """Build I and alpha from explicit Pauli matrices, independently of production tables."""
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                      [[1, 0], [0, -1]]], complex)
    out = np.zeros((4, 4, 4), complex)
    out[0] = np.eye(4)
    out[1:, :2, 2:] = pauli
    out[1:, 2:, :2] = pauli
    return out, pauli


def _mesh():
    assert jax.default_backend() == 'cpu'
    assert jax.device_count() == 4
    return Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))


def _sym():
    """Use the actual Si WFN symmetry header without loading its wavefunctions."""
    import h5py
    from symmetry_maps import SymMaps
    path = Path(__file__).parents[1] / 'regression/si_bse_debug/WFN.h5'
    with h5py.File(path, 'r') as f:
        h = f['mf_header']
        avec = h['crystal/avec'][:]
        header = SimpleNamespace(kpoints=h['kpoints/rk'][:], kgrid=h['kpoints/kgrid'][:],
            shift=h['kpoints/shift'][:], nkpts=int(h['kpoints/nrk'][()]),
            ntran=int(h['symmetry/ntran'][()]), sym_matrices=h['symmetry/mtrx'][:],
            translations=h['symmetry/tnp'][:], avec=avec,
            atom_types=h['crystal/atyp'][:],
            atom_crys=np.einsum('ij,kj->ki', np.linalg.inv(avec).T, h['crystal/apos'][:]),
            trs_holds=True)
    return SymMaps(header)


def test_literal_gamma_lift_and_all_96_spatial_trs_rows():
    """The kinetic-balance lift intertwines polar momentum and parity-graded spin transport."""
    from common.bispinor_init import lift_to_4spinor, HALFALPHA
    from common.gamma_matrices import gamma_apply, gamma_perm_phase
    gamma, pauli = _gamma()
    rng = np.random.default_rng(106)
    psi = rng.normal(size=(1, 4, 2, 3)) + 1j*rng.normal(size=(1, 4, 2, 3))
    momentum = np.array([[[.3, -.2, .7], [1., 2., -3.], [.1, .5, .9]]])
    literal = np.concatenate((psi, HALFALPHA*np.einsum('aij,kga,kbjg->kbig',
                                                      pauli, momentum, psi)), axis=2)
    sym = _sym()
    rows = np.arange(96)
    u4, u2 = sym.spinor_action(rows, nspinor=4), sym.spinor_action(rows, nspinor=2)
    action = sym.lorentz_action(rows)
    assert np.any(np.linalg.det(action[:, 1:, 1:]) < 0)
    for mode in ('raw', 'isometric'):
        want = literal if mode == 'raw' else literal / np.sqrt(
            1 + HALFALPHA**2*np.sum(momentum**2, axis=-1))[:, None, None, :]
        got = lift_to_4spinor(jnp.asarray(psi), jnp.asarray(momentum),
                            jnp.zeros((1, 3)), jnp.eye(3), representation=mode)
        np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)
        for row in rows:
            source = psi.conj() if row >= 48 else psi
            transformed = np.einsum('ac,kbcg->kbag', u2[row], source)
            pchild = momentum @ action[row, 1:, 1:].T
            actual = lift_to_4spinor(jnp.asarray(transformed), jnp.asarray(pchild),
                jnp.zeros((1, 3)), jnp.eye(3), representation=mode)
            expected = np.einsum('ac,kbcg->kbag', u4[row],
                                want.conj() if row >= 48 else want)
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    for a in range(4):
        np.testing.assert_allclose(gamma_apply(jnp.asarray(literal), *gamma_perm_phase(a), axis=2),
            np.einsum('ac,kbcg->kbag', gamma[a], literal), atol=2e-14)
    transformed = np.einsum('qba,ibc,qcd->qiad', u4.conj(), gamma, u4)
    transformed[48:] = transformed[48:].conj()
    np.testing.assert_allclose(transformed, np.einsum('qij,jab->qiab', action, gamma), atol=3e-14)


def test_vertex_after_unfold_equals_mixed_vertices_before_unfold():
    """Complex mixed vertices obey antiunitary coefficient conjugation and Seitz phases."""
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from common.shard_map import shard_map
    sym, mesh = _sym(), _mesh()
    points = np.array(list(np.ndindex(4, 4, 4)))
    plan = build_centroid_k_unfold_plan(sym, points, (4, 4, 4), mesh, nspinor=4)
    from dataclasses import replace
    rows = np.arange(96)
    plan = replace(plan, irr_idx=np.ones(96, dtype=np.int32), sym_idx=rows,
                   spin_action_full=sym.spinor_action(rows, nspinor=4))
    rng = np.random.default_rng(107)
    parent = rng.normal(size=(plan.n_parent, 4, 4, plan.n_centroid_packed))
    parent = parent + 1j*rng.normal(size=parent.shape)
    parent[..., plan.layout.axis.packed_to_canonical < 0] = 0
    gamma, _ = _gamma()
    coeff = np.array([.3+.4j, -.7+.2j, 1.1-.5j, .9+.6j])
    spec = P(None, 'x', None, 'y')
    operand = jax.device_put(parent, NamedSharding(mesh, spec))
    direct = sum(coeff[a]*jax.jit(shard_map(
        lambda x: plan.unfold_face(x, vertex=a, spin_axis=2, mu_axis=3, mesh_axis='y'),
        mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False))(operand) for a in range(4))
    expected = np.zeros_like(np.asarray(direct))
    for k, (p, row) in enumerate(zip(plan.irr_idx, plan.sym_idx)):
        u = plan.spin_action_full[k]
        lam = sym.lorentz_action(np.array([row]))[0]
        before = np.einsum('i,ij,jab,nbm->nam', coeff, lam, gamma, parent[p])
        # For antiunitary T, T(sum c Gamma psi) requires conjugated c.
        if row >= plan.n_sym_spatial:
            before = np.einsum('i,ij,jab,nbm->nam', coeff.conj(), lam, gamma, parent[p])
        before = np.take(before, plan.sym_perm[row], axis=-1)
        phase = np.exp(2j*np.pi*(plan.L_table[row] @ plan.k_parent_frac[p]))
        before *= phase
        expected[k] = np.einsum('ab,nbm->nam', u,
            before.conj() if row >= plan.n_sym_spatial else before)
    np.testing.assert_allclose(direct, expected, rtol=3e-13, atol=3e-13)


def _cpu_algebra(monkeypatch):
    """Substitute only backend FFT/GEMM plumbing, preserving production physics contractions."""
    import common.fft_helpers as fft
    import distrib_la
    import ffi.mklfft
    def transform(mesh, grid, spec, *, norm='ortho', **kwargs):
        return lambda x: jnp.fft.fftn(x.reshape(tuple(grid)+x.shape[1:]),
            axes=(0, 1, 2), norm=norm).reshape(x.shape)
    monkeypatch.setattr(fft, 'make_flat_k_fftn', transform)
    def gemm(mesh, **kwargs):
        def contract(x, y):
            return x @ y
        contract.mesh = mesh
        return contract
    monkeypatch.setattr(distrib_la, 'gemm_plan', gemm)
    monkeypatch.setattr(ffi.mklfft, 'fused_fft_ffi_enabled', lambda: False)
    def inverse(mesh, grid, spec, *, norm='ortho', **kwargs):
        return lambda x: jnp.fft.ifftn(x.reshape(tuple(grid)+x.shape[1:]),
            axes=(0, 1, 2), norm=norm).reshape(x.shape)
    monkeypatch.setattr(fft, 'make_flat_k_ifftn', inverse)


def _chi_literal(left, right, gamma, tau, energies):
    """Sum d_I=c(k)^dagger Gamma_I v(k+q), and the reverse ordered partner."""
    nk = len(left)
    grid = (3, 3, 1)
    out = np.zeros((4, 4, nk, left.shape[-1], right.shape[-1]), complex)
    for q, qvec in enumerate(np.ndindex(grid)):
        for k, kvec in enumerate(np.ndindex(grid)):
            kp = np.ravel_multi_index((np.array(kvec)+qvec) % grid, grid)
            km = np.ravel_multi_index((np.array(kvec)-qvec) % grid, grid)
            for v in (0, 1):
                for c in (2, 3):
                    l = np.einsum('am,Iab,bm->Im', left[kp, v].conj(), gamma, left[k, c])
                    r = np.einsum('am,Iab,bm->Im', right[kp, v].conj(), gamma, right[k, c])
                    weight = np.exp(-tau*(energies[k, c]-energies[kp, v]))
                    out[:, :, q] -= weight*np.einsum('Im,Jn->IJmn', l.conj(), r)
                    l = np.einsum('am,Iab,bm->Im', left[km, v].conj(), gamma, left[k, c])
                    r = np.einsum('am,Iab,bm->Im', right[km, v].conj(), gamma, right[k, c])
                    weight = np.exp(-tau*(energies[k, c]-energies[km, v]))
                    out[:, :, q] -= weight*np.einsum('Im,Jn->IJmn', l, r.conj())
    # Three unitary FFTs leave exactly 1/sqrt(Nk), not 1/Nk, at this raw seam.
    return out / np.sqrt(nk)


def test_all_16_chi_blocks_nonzero_ct_literal_both_orientations(monkeypatch):
    """Non-TR complex wavefunctions expose q reversal, Gamma2 transposes and both orientations."""
    from gw.w_isdf import _get_chi_minimax_kernel, MinimaxNodes
    _cpu_algebra(monkeypatch)
    mesh = _mesh()
    rng = np.random.default_rng(108)
    left = (rng.normal(size=(9, 4, 4, 4))+1j*rng.normal(size=(9, 4, 4, 4)))/3
    right = (rng.normal(size=(9, 4, 4, 6))+1j*rng.normal(size=(9, 4, 4, 6)))/3
    energies = np.tile([-1.2, -.7, .4, 1.3], (9, 1))+np.arange(9)[:, None]*.013
    tau = .37
    nodes = MinimaxNodes(t=jnp.array([tau], dtype=jnp.complex128),
                         alpha=jnp.array([-np.exp(-tau*1.1)], dtype=jnp.complex128))
    gamma, _ = _gamma()
    expected = _chi_literal(left, right, gamma, tau, energies)
    assert np.max(np.abs(expected[0, 1:])) > .1
    assert np.max(np.abs(expected.imag)) > .01
    mun = jax.device_put(left.transpose(0, 2, 3, 1), NamedSharding(mesh, P(None,None,'x','y')))
    nmu = jax.device_put(right, NamedSharding(mesh, P(None,'x',None,'y')))
    for a in range(4):
        for b in range(4):
            kernel = _get_chi_minimax_kernel(mesh, (3,3,1), layout='face',
                face_shape=(9,4,4,4), right_face_shape=(9,4,6,4), vertex_pair=(a,b))
            actual = kernel(nodes, mun, nmu, jnp.asarray(energies < 0),
                jnp.asarray(energies > 0), jnp.asarray(energies), jnp.array(-.7), jnp.array(.4),
                (jnp.argmax(jnp.abs(gamma[a]),axis=1), jnp.conj(jnp.sum(gamma[a],axis=1)),
                 jnp.argmax(jnp.abs(gamma[b]),axis=1), jnp.conj(jnp.sum(gamma[b],axis=1))))
            np.testing.assert_allclose(actual, expected[a,b], rtol=3e-12, atol=3e-12,
                                       err_msg=f'Lorentz block {(a,b)}')


def _toy_plan(mesh):
    """Build a physical glide/TR group on a 3x3 k mesh through SymMaps."""
    from symmetry_maps import SymMaps
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    mirror = np.array([[0,1,0], [1,0,0], [0,0,1]])
    k = np.array(list(np.ndindex(3,3,1))) / [3,3,1]
    header = SimpleNamespace(kpoints=k[[0,1,4,5]], kgrid=np.array([3,3,1]),
        shift=np.zeros(3), nkpts=4, ntran=2,
        sym_matrices=np.array([np.eye(3,dtype=int), mirror]),
        translations=np.array([[0.,0.,0.],[np.pi,np.pi,0.]]),
        avec=np.eye(3), atom_types=np.array([1,1]),
        atom_crys=np.array([[0.,0.,0.],[.5,.5,0.]]), trs_holds=True)
    sym = SymMaps(header)
    points = np.array(list(np.ndindex(2,2,1)))
    return build_centroid_k_unfold_plan(sym, points, (2,2,1), mesh, nspinor=4,
                                       parent_k_frac=header.kpoints)


def _literal_children(parent, plan):
    """Apply explicit U4, centroid pullback, Bloch phase and antiunitary conjugation."""
    out = []
    for p, row, u in zip(plan.irr_idx, plan.sym_idx, plan.spin_action_full):
        x = np.take(parent[p], plan.sym_perm[row], axis=-1)
        x = x*np.exp(2j*np.pi*(plan.L_table[row] @ plan.k_parent_frac[p]))
        if row >= plan.n_sym_spatial:
            x = x.conj()
        out.append(np.einsum('ac,ncm->nam', u, x))
    return np.array(out)


def test_parent_sigma_all_vertices_q_convolution_and_projection(monkeypatch):
    """Production parent Sigma equals literal -sum_q Gamma_A G(k-q) Gamma_B D_AB/Nk."""
    from gw.wavefunction_bundle import ParentGreenCarrier
    from gw.photon_sigma import _make_photon_static_block_kernel
    from common.gamma_matrices import gamma_perm_phase
    _cpu_algebra(monkeypatch)
    mesh = _mesh()
    plan = _toy_plan(mesh)
    assert plan.n_parent < plan.n_full and np.any(plan.sym_idx >= plan.n_sym_spatial)
    rng = np.random.default_rng(109)
    shape = (plan.n_parent, 4, 4, plan.n_centroid_packed)
    raw = (rng.normal(size=shape)+1j*rng.normal(size=shape))/3
    raw[..., plan.layout.axis.packed_to_canonical < 0] = 0
    put = lambda x, spec: jax.device_put(x, NamedSharding(mesh,spec))
    carrier = ParentGreenCarrier(put(raw,P(None,'x',None,'y')),
        put(raw.transpose(0,2,3,1),P(None,None,'x','y')),
        jnp.zeros((plan.n_parent,4)), jnp.ones((plan.n_parent,4)), plan)
    wfns = SimpleNamespace(green_parent=carrier)
    child = _literal_children(raw,plan)
    gamma, _ = _gamma()
    interaction = rng.normal(size=(9,plan.n_centroid_packed,plan.n_centroid_packed))
    interaction = interaction+1j*rng.normal(size=interaction.shape)
    weight = np.tile([1., .7, .2, 0.],(9,1))
    green = np.einsum('knam,kn,knbv->kambv',child,weight,child.conj())
    bare = np.zeros_like(green)
    for k, kv in enumerate(np.ndindex(3,3,1)):
        for q, qv in enumerate(np.ndindex(3,3,1)):
            km = np.ravel_multi_index((np.array(kv)-qv)%[3,3,1],(3,3,1))
            bare[k] -= green[km]*interaction[q][None,:,None,:]/9
    for a in range(4):
        for b in range(4):
            sigma = np.einsum('ac,kcmdv,db->kambv',gamma[a],bare,gamma[b])
            full = np.einsum('kiam,kambv,kjbv->kij',child.conj(),sigma,child)
            kernel = _make_photon_static_block_kernel(mesh,(3,3,1),9,wfns,wfns)
            actual = kernel(carrier,carrier,jnp.asarray(weight),
                            put(interaction,P(None,'x','y')),jnp.array(1.),
                            (gamma_perm_phase(a),gamma_perm_phase(b)))
            np.testing.assert_allclose(actual,full[plan.parent_full_rows],
                                       rtol=3e-12,atol=3e-12,err_msg=f'Sigma {(a,b)}')
            coh = kernel(carrier,carrier,jnp.asarray(weight),
                        put(interaction,P(None,'x','y')),jnp.array(-.5),
                        (gamma_perm_phase(a),gamma_perm_phase(b)))
            np.testing.assert_allclose(coh,-.5*full[plan.parent_full_rows],rtol=3e-12,atol=3e-12)


def test_bare_tiles_metric_sign_and_complex_hermitian_companions(monkeypatch):
    """The TT metric is negative once, and reversing Lorentz endpoints takes the centroid dagger."""
    from gw.v_q_bispinor import _make_per_q_v_builder_for_tile, BispinorVqReader
    import gw.compute_vcoul
    from vcoul import COULOMB_GAUGE_TT_SIGN
    monkeypatch.setattr(gw.compute_vcoul,'compute_v_q_per_G',lambda *a,**k: np.array([[2.,3.]]))
    k = np.array([[[1.,2.], [2.,-1.], [3.,2.]]])
    direction = k.transpose(0,2,1)
    projector = np.eye(3)-direction[..., :,None]*direction[...,None,:]/np.sum(direction**2,axis=-1)[...,None,None]
    assert COULOMB_GAUGE_TT_SIGN == -1
    for a in (1,2,3):
        for b in (1,2,3):
            builder = _make_per_q_v_builder_for_tile(mu_L=a,nu_L=b,bvec=np.eye(3),
                cell_volume=1.,sys_dim=3,vcoul_cutoff_ry=None)
            np.testing.assert_allclose(builder(np.zeros((1,3)),k),
                -np.array([[2.,3.]])*projector[:,:,a-1,b-1],atol=2e-14)
    mesh = _mesh()
    tile = np.array([[[1+2j,3-4j],[5+6j,7-8j]]])
    reader = object.__new__(BispinorVqReader)
    reader._mesh, reader._mu_bases = mesh, None
    reader.n_q_total = 1
    reader._tile_shape = lambda *a: (1,2,2)
    reader._padded_shape_LR = lambda *a: (2,2)
    reader._io = SimpleNamespace(read_slab=lambda *a,**k: jnp.asarray(tile))
    np.testing.assert_array_equal(reader.get_tile(2,1),tile.conj().swapaxes(-1,-2))


def test_density_all_currents_signed_weights_and_time_reversal():
    """Charge and J/c use the same signed band weights, with current odd under T."""
    from psp.get_DFT_mtxels import density_components_from_psi_r
    gamma, _ = _gamma()
    rng = np.random.default_rng(110)
    psi = rng.normal(size=(4,4,2,2,1))+1j*rng.normal(size=(4,4,2,2,1))
    occ = np.array([1.,.6,-.1,0.])
    literal = np.einsum('n,na...,Iab,nb...->I...',occ,psi.conj(),gamma,psi).real
    actual = density_components_from_psi_r(jnp.asarray(psi),jnp.asarray(occ),include_dirac_current=True)
    np.testing.assert_allclose(actual,literal,rtol=3e-14,atol=3e-14)
    theta = np.kron(np.eye(2),np.array([[0,1],[-1,0]]))
    reversed_psi = np.einsum('ab,nb...->na...',theta,psi.conj())
    reversed_density = density_components_from_psi_r(jnp.asarray(reversed_psi),
        jnp.asarray(occ),include_dirac_current=True)
    np.testing.assert_allclose(reversed_density,literal*np.array([1,-1,-1,-1])[:,None,None,None],atol=3e-14)


def test_nonzero_ct_covariance_recomputed_from_spinors():
    """Every mixed response block transforms as a polar time-odd tensor, with K once."""
    from symmetry_maps import mix_lorentz_blocks
    mesh = _mesh()
    plan = _toy_plan(mesh)
    sym = plan.sym
    rng = np.random.default_rng(111)
    left = (rng.normal(size=(9,4,4,4))+1j*rng.normal(size=(9,4,4,4)))/3
    right = (rng.normal(size=(9,4,4,6))+1j*rng.normal(size=(9,4,4,6)))/3
    energy = np.tile([-1.2,-.7,.4,1.3],(9,1))
    gamma, _ = _gamma()
    original = _chi_literal(left,right,gamma,.2,energy)
    assert np.max(np.abs(original[0,1:])) > .1
    for row in range(4):
        rotation, _, anti = sym.operation_rows(np.array(row))
        full_index = np.array([np.ravel_multi_index((rotation @ np.array(k))%[3,3,1],(3,3,1))
                               for k in np.ndindex(3,3,1)])
        u = sym.spinor_action(np.array([row]),nspinor=4)[0]
        children = []
        for source in (left,right):
            x = np.einsum('ab,knbm->knam',u,source.conj() if anti else source)
            arranged = np.empty_like(x)
            arranged[full_index] = x
            children.append(arranged)
        recomputed = _chi_literal(*children,gamma,.2,energy)
        source = original.conj() if anti else original
        mixed = mix_lorentz_blocks({(a,b):jnp.asarray(source[a,b]) for a in range(4) for b in range(4)},
            sym=sym,sym_idx=np.full(9,row),mesh_xy=mesh)
        for a in range(4):
            for b in range(4):
                np.testing.assert_allclose(mixed[a,b],recomputed[a,b,full_index],rtol=3e-12,atol=3e-12)
        if row == 2:
            assert np.max(np.abs(np.asarray(mixed[0,1])-source[0,1])) > .1


def test_packed_dyson_order_prefactor_hermiticity_and_bare_limit(monkeypatch):
    """Packed W solves (I-D chi)W=D, and unscreened TT gives X=SX with zero COH."""
    import distrib_la
    from gw.photon_layout import PhotonBasisLayout, pack_photon_operator, photon_block_view
    from gw.w_isdf import solve_w
    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(4,4,mesh)
    rng = np.random.default_rng(112)
    z = (rng.normal(size=(4,16,16))+1j*rng.normal(size=(4,16,16)))/20
    d = z+z.conj().swapaxes(-1,-2)+np.diag([1.]*4+[-1.]*12)
    x = rng.normal(size=(4,16,7))+1j*rng.normal(size=(4,16,7))
    chi = -(x @ x.conj().swapaxes(-1,-2))/50
    pack = lambda v: pack_photon_operator(lambda a,b:jnp.asarray(v[:,a*4:(a+1)*4,b*4:(b+1)*4]),4,layout,mesh)
    # Exercise the production distributed A build; only the vendor LU backend is replaced.
    native = jax.jit(lambda a,b:jnp.linalg.solve(a,b),out_shardings=NamedSharding(mesh,P(None,'x','y')))
    monkeypatch.setattr(distrib_la,'plan',lambda *a,**k:SimpleNamespace(
        describe=lambda:'CPU literal LU backend',batched=native))
    meta = SimpleNamespace(nk_tot=9,nspin=1,nspinor_wfnfile=2,n_rmu=16)
    actual = solve_w(pack(d),pack(chi),meta,mesh,dyson_solver='distributed',n_rmu_logical=16)
    expected = np.linalg.solve(np.eye(16)-d @ (chi/3),d)
    for a in range(4):
        for b in range(4):
            block = photon_block_view(actual,layout,a,b,mesh)
            np.testing.assert_allclose(block,expected[:,a*4:(a+1)*4,b*4:(b+1)*4],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(expected,expected.conj().swapaxes(-1,-2),atol=3e-13)
    assert np.max(np.abs(expected[:,:4,4:])) > .01
    bare_d = np.zeros_like(d)
    bare_d[:,:4,:4],bare_d[:,4:,4:] = d[:,:4,:4],d[:,4:,4:]
    bare_chi = np.zeros_like(chi)
    bare_chi[:,:4,:4] = chi[:,:4,:4]
    bare_w = solve_w(pack(bare_d),pack(bare_chi),meta,mesh,dyson_solver='distributed',n_rmu_logical=16)
    for a in (1,2,3):
        for b in (1,2,3):
            np.testing.assert_allclose(photon_block_view(bare_w,layout,a,b,mesh),
                bare_d[:,a*4:(a+1)*4,b*4:(b+1)*4],atol=3e-13)


def test_q_star_unfold_all_blocks_ward_contact_and_daggers():
    """Scalar phase/conjugation then Lambda tensor mixing preserves every Hermitian companion."""
    from gw.photon_layout import PhotonBasisLayout,pack_photon_operator
    from gw.w_isdf import StaticPhotonResponse,photon_blocks_full_q,_subtract_static_tt_contact
    from gw.qgrid_symmetry import qgrid_trs_policy_for
    from symmetry_maps import bgw_integer_q_to_fractional
    mesh = _mesh()
    plan = _toy_plan(mesh)
    sym = plan.sym
    policy = qgrid_trs_policy_for(sym=sym,irr_idx_q=sym.irr_idx_q,sym_idx_q=sym.sym_idx_q,
        kgrid=(3,3,1),n_sym_spatial=2,context='literal physics oracle')
    nmu,nq = plan.n_centroid_packed,len(sym.q_irr_full_idx)
    layout = PhotonBasisLayout.from_centroid_extents(nmu,nmu,mesh)
    rng = np.random.default_rng(113)
    x = rng.normal(size=(nq,4*nmu,4*nmu))+1j*rng.normal(size=(nq,4*nmu,4*nmu))
    d = x+x.conj().swapaxes(-1,-2)
    packed = pack_photon_operator(lambda a,b:jnp.asarray(d[:,a*nmu:(a+1)*nmu,b*nmu:(b+1)*nmu]),nq,layout,mesh)
    response = StaticPhotonResponse(layout,packed,packed,'none','toy',qgrid_policy=policy,family_plans=(plan,plan))
    keys = [(a,b) for a in range(4) for b in range(4)]
    restored = dict(photon_blocks_full_q(response,keys,term='W'))
    qfrac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int,(3,3,1))
    for q,(p,row) in enumerate(zip(sym.irr_idx_q,policy.unfold_sym_idx)):
        perm = plan.sym_perm[row]
        phase = np.exp(2j*np.pi*(plan.L_table[row] @ qfrac[p]))
        scalar = np.diag(phase) @ np.eye(nmu)[perm]
        lam = sym.lorentz_action(np.array([row]))[0]
        action = np.kron(lam,scalar)
        # Scalar operator unfold uses the density-dual phase; verify its actual orientation.
        source = d[p].conj() if row >= 2 else d[p]
        if row >= 2:
            action = action.conj()
        expected = action @ source @ action.conj().T
        for a,b in keys:
            np.testing.assert_allclose(restored[a,b][q],expected[a*nmu:(a+1)*nmu,b*nmu:(b+1)*nmu],atol=3e-12)
    for a,b in keys:
        np.testing.assert_allclose(restored[a,b],np.asarray(restored[b,a]).conj().swapaxes(-1,-2),atol=3e-12)
    raw = rng.normal(size=(nq,nmu,nmu))+1j*rng.normal(size=(nq,nmu,nmu))
    np.testing.assert_allclose(_subtract_static_tt_contact(jnp.asarray(raw)),raw-raw[:1],atol=2e-14)


def test_parent_chi_equals_literal_full_k_for_all_16_blocks(monkeypatch):
    """Raw-parent current response equals the explicit child-wavefunction Lehmann sum."""
    from gw.w_isdf import _get_chi_minimax_kernel,MinimaxNodes
    _cpu_algebra(monkeypatch)
    mesh = _mesh()
    plan = _toy_plan(mesh)
    rng = np.random.default_rng(114)
    shape = (plan.n_parent,4,4,plan.n_centroid_packed)
    parent = (rng.normal(size=shape)+1j*rng.normal(size=shape))/3
    parent[...,plan.layout.axis.packed_to_canonical < 0] = 0
    children = _literal_children(parent,plan)
    energy = np.tile([-1.2,-.7,.4,1.3],(plan.n_parent,1))
    expected = _chi_literal(children,children,_gamma()[0],.2,energy[plan.irr_idx])
    nodes = MinimaxNodes(t=jnp.array([.2],dtype=jnp.complex128),
        alpha=jnp.array([-np.exp(-.2*1.1)],dtype=jnp.complex128))
    mun = jax.device_put(parent.transpose(0,2,3,1),NamedSharding(mesh,P(None,None,'x','y')))
    nmu = jax.device_put(parent,NamedSharding(mesh,P(None,'x',None,'y')))
    for a in range(4):
        for b in range(4):
            shape = (plan.n_parent,4,plan.n_centroid_packed,4)
            kernel = _get_chi_minimax_kernel(mesh,(3,3,1),layout='face',face_shape=shape,
                right_face_shape=shape,vertex_pair=(a,b),k_unfold_plan=(plan,plan))
            actual = kernel(nodes,mun,nmu,jnp.asarray(energy<0),jnp.asarray(energy>0),
                            jnp.asarray(energy),jnp.array(-.7),jnp.array(.4),
                            (jnp.argmax(jnp.abs(_gamma()[0][a]),axis=1), jnp.conj(jnp.sum(_gamma()[0][a],axis=1)),
                             jnp.argmax(jnp.abs(_gamma()[0][b]),axis=1), jnp.conj(jnp.sum(_gamma()[0][b],axis=1))))
            np.testing.assert_allclose(actual,expected[a,b],rtol=3e-12,atol=3e-12)


def test_full_band_unfold_matches_literal_sigma_on_symmetric_complete_toy(monkeypatch):
    """A complete occupied toy gives a covariant, non-diagonal band Sigma on every child."""
    from gw.wavefunction_bundle import ParentGreenCarrier
    from gw.photon_sigma import _make_photon_static_block_kernel
    from common.gamma_matrices import gamma_perm_phase
    from symmetry_maps import unfold_file_wedge_band_operator
    _cpu_algebra(monkeypatch)
    mesh = _mesh()
    plan = _toy_plan(mesh)
    nmu,nb = plan.n_centroid_packed,4*plan.n_centroid_packed
    rng = np.random.default_rng(115)
    raw = []
    for p in range(plan.n_parent):
        x = rng.normal(size=(nb,nb))+1j*rng.normal(size=(nb,nb))
        raw.append(np.linalg.qr(x)[0].T.reshape(nb,4,nmu))
    raw = np.array(raw)
    put = lambda x,s:jax.device_put(x,NamedSharding(mesh,s))
    carrier = ParentGreenCarrier(put(raw,P(None,'x',None,'y')),
        put(raw.transpose(0,2,3,1),P(None,None,'x','y')),
        jnp.zeros((plan.n_parent,nb)),jnp.ones((plan.n_parent,nb)),plan)
    wfns = SimpleNamespace(green_parent=carrier)
    # Symmetrize a diagonal centroid operator over all canonical permutations.
    site = np.mean(np.array([np.arange(1,nmu+1)[perm] for perm in plan.sym_perm]),axis=0)
    interaction = np.broadcast_to(np.diag(site),(9,nmu,nmu)).copy()
    child = _literal_children(raw,plan)
    parent_sigma = None
    for a,scale in ((1,-1.),(2,-1.),(3,-2.)):
        kernel = _make_photon_static_block_kernel(mesh,(3,3,1),9,wfns,wfns)
        value = kernel(carrier,carrier,jnp.ones((9,nb)),put(scale*interaction,P(None,'x','y')),jnp.array(1.),
                       (gamma_perm_phase(a),gamma_perm_phase(a)))
        parent_sigma = value if parent_sigma is None else parent_sigma+value
    actual = unfold_file_wedge_band_operator(plan.sym,parent_sigma,trs_rule='transpose')
    # G=identity on the complete occupied spin/centroid space; alpha_i^2=I.
    expected = 4*np.einsum('kiam,m,kjam->kij',child.conj(),site,child)
    np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
    assert np.max(np.abs(expected.imag)) > .01
    wrong = np.asarray(parent_sigma)[plan.irr_idx]
    assert np.max(np.abs(wrong-expected)) > .01


def test_periodic_transverse_hartree_sign_projector_and_zero_mode(monkeypatch):
    """Direct TT has the same negative metric, 8pi/G^2 Ry weight, and no exchange head."""
    import psp.dft_operators as dft
    monkeypatch.setattr(dft,'local_fftn3',lambda x,**k:jnp.fft.fftn(x,**k))
    monkeypatch.setattr(dft,'local_ifftn3',lambda x,**k:jnp.fft.ifftn(x,**k))
    grid = (3,3,3)
    rng = np.random.default_rng(116)
    current = rng.normal(size=(3,*grid))
    fourier = np.fft.fftn(current,axes=(-3,-2,-1),norm='ortho')
    expected_g = np.zeros_like(fourier)
    for index in np.ndindex(grid):
        g = np.array([np.fft.fftfreq(3)[i]*3 for i in index])
        if g @ g:
            projector = np.eye(3)-np.outer(g,g)/(g@g)
            expected_g[(slice(None),*index)] = -8*np.pi/(g@g)*(projector @ fourier[(slice(None),*index)])
    actual = dft.transverse_potential_from_current(jnp.asarray(current),jnp.eye(3),jnp.eye(3),
                                                 1.,False,tt_metric_sign=-1.)
    expected = np.fft.ifftn(expected_g,axes=(-3,-2,-1),norm='ortho').real
    np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(np.mean(actual,axis=(-3,-2,-1)),0,atol=3e-14)


def test_isdf_current_signed_normal_matrix_against_literal_pair_gram(monkeypatch):
    """The legacy Gamma2 normal matrix is minus the physical Gram; other channels are positive."""
    import isdf.core as core
    from common.gamma_matrices import gamma_perm_phase
    _cpu_algebra(monkeypatch)
    monkeypatch.setattr(core,'local_fftn3',lambda x,**k:jnp.fft.fftn(x,**k))
    monkeypatch.setattr(core,'local_ifftn3',lambda x,**k:jnp.fft.ifftn(x,**k))
    mesh = _mesh()
    plan = _toy_plan(mesh)
    rng = np.random.default_rng(117)
    shape = (plan.n_parent,4,4,plan.n_centroid_packed)
    parent = (rng.normal(size=shape)+1j*rng.normal(size=shape))/3
    child = _literal_children(parent,plan)
    gamma,_ = _gamma()
    put = lambda x,s:jax.device_put(x,NamedSharding(mesh,s))
    mun,nmu = put(parent.transpose(0,2,3,1),P(None,None,'x','y')),put(parent,P(None,'x',None,'y'))
    for a in (0,1,2,3):
        expected = np.zeros((9,plan.n_centroid_packed,plan.n_centroid_packed),complex)
        for q,qv in enumerate(np.ndindex(3,3,1)):
            for k,kv in enumerate(np.ndindex(3,3,1)):
                kp = np.ravel_multi_index((np.array(kv)+qv)%[3,3,1],(3,3,1))
                for m in (0,1):
                    for n in (2,3):
                        density = np.einsum('am,ab,bm->m',child[k,m].conj(),gamma[a],child[kp,n])
                        expected[q] += np.outer(density.conj(),density)
        actual = core._c_q_face_parent(mun,nmu,jnp.array([1.,1.,0.,0.]),jnp.array([0.,0.,1.,1.]),
            kgrid=(3,3,1),mesh_xy=mesh,gemm=lambda x,y:x@y,k_unfold_plan=plan,gamma_L=a,gamma_R=a)
        sign = -1 if a == 2 else 1
        np.testing.assert_allclose(actual,sign*expected,rtol=3e-12,atol=3e-12,err_msg=f'ISDF signed Gram gamma{a}')
        print(f'ISDF_SIGNED_GRAM channel={a} sign={sign} max_physical_difference={np.max(np.abs(actual-expected)):.16e}')


def test_signed_isdf_rhs_cancels_in_unregularized_fit():
    """Gamma2's signed normal matrix and RHS cancel before the inverse, not in a positive ridge."""
    from common.gamma_matrices import gamma_double_contract,gamma_perm_phase
    gamma,_ = _gamma()
    rng = np.random.default_rng(118)
    psi = rng.normal(size=(4,4,7))+1j*rng.normal(size=(4,4,7))
    p_l = np.einsum('nam,nbv->ambv',psi[:2].conj(),psi[:2])
    p_r = np.einsum('nam,nbv->ambv',psi[2:].conj(),psi[2:])
    for a in (1,2,3):
        pairs = np.einsum('mas,ab,nbs->mns',psi[:2].conj(),gamma[a],psi[2:]).reshape(4,7)
        gram = pairs.conj().T @ pairs
        actual = np.asarray(gamma_double_contract(jnp.asarray(p_l.conj()[None]),
            jnp.asarray(p_r[None]),*gamma_perm_phase(a),*gamma_perm_phase(a),spin_axes=(1,3)))[0]
        sign = -1 if a == 2 else 1
        np.testing.assert_allclose(actual,sign*gram,rtol=3e-13,atol=3e-13)
        # Four pair features and three centroid samples leave an overdetermined exact normal solve.
        fitted = np.linalg.solve(actual[:3,:3],actual[:3,3:])
        physical = np.linalg.solve(gram[:3,:3],gram[:3,3:])
        np.testing.assert_allclose(fitted,physical,rtol=3e-12,atol=3e-12)


def test_positive_ridge_moves_negative_gram_toward_zero():
    """The retained positive ridge anti-regularizes the signed Gamma2 normal matrix."""
    from isdf.core import _transverse_lu_math,_TRANSVERSE_LU_RIDGE
    from jax.scipy.linalg import lu_solve
    physical = np.diag([1.,6e-13]).astype(complex)
    rhs = np.array([0.,6e-13],complex)
    signed = -physical
    lu,piv = _transverse_lu_math(jnp.asarray(signed),2)
    actual = np.asarray(lu_solve((lu,piv),jnp.asarray(-rhs)))
    ridge = _TRANSVERSE_LU_RIDGE*abs(np.trace(signed))/2
    desired = np.linalg.solve(physical+ridge*np.eye(2),rhs)
    legacy = np.linalg.solve(physical-ridge*np.eye(2),rhs)
    np.testing.assert_allclose(actual,legacy,rtol=3e-13,atol=3e-13)
    assert abs(actual[1]) > 5 and abs(desired[1]) < 1
    print(f'GAMMA2_RIDGE legacy={actual[1]} positive_gram={desired[1]} ridge={ridge}')


def test_sigma_oracle_rejects_transposed_gamma2(monkeypatch):
    """A deliberate Gamma2 transpose is rejected by the mixed-vertex Sigma oracle."""
    import common.gamma_matrices as gamma
    original = gamma.gamma_apply
    def wrong(value, perm, phase, *, axis):
        return original(value, perm, jnp.conj(phase), axis=axis)
    monkeypatch.setattr(gamma, 'gamma_apply', wrong)
    with pytest.raises(AssertionError,match='Sigma'):
        test_parent_sigma_all_vertices_q_convolution_and_projection(monkeypatch)
