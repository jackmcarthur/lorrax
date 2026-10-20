"""Prepared, NOT RUN: actual P4 CUDA HF/tail/BSE software verification.

Run collectively in an existing owned allocation only, with kill-on-bad-exit
and an external timeout <=180s. No material input, SC solve or TD trajectory.
The BSE static body/head are explicitly declared toys, never RPA evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback

REVISION = 'd469c49e0e7197a384990a4fe2817f9ce1e3bafe'


def fixture(mesh):
    """Native full-parent fixture; no CPU-device factory or global host put."""
    import warnings
    import jax
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.centroid_basis import PackedCentroidBasis
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from common.meta import Meta
    from gw.wavefunction_bundle import BandSlices, ParentGreenCarrier, Wavefunctions
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.shared_pole_recipe import CapacityLedger
    from gw.shared_pole_screening import _shared_pole_tables
    from common.gpu_utils import device_budget_bytes
    from symmetry_maps import SymMaps
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(20261025)
    nk, nb, ns, mu = 4, 8, 2, 4
    grid, fftgrid = (4, 1, 1), (4, 4, 4)
    kf = np.c_[np.arange(nk)/nk, np.zeros((nk, 2))]
    centroids = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.int32)
    wfn_facts = SimpleNamespace(trs_holds=False,ntran=1,kgrid=np.asarray(grid),
        nkpts=nk,kpoints=kf,sym_matrices=np.eye(3,dtype=np.int64)[None],shift=np.zeros(3))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sym = SymMaps(wfn_facts)
    plan = build_centroid_k_unfold_plan(sym, centroids, fftgrid, mesh,
                                       nspinor=ns, parent_k_frac=kf)
    basis = PackedCentroidBasis(mesh, plan.layout, centroids, mu)
    perm = np.asarray(plan.layout.axis.pack_host(np.arange(mu)[None], axis=1))[0]
    psi = .16*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = np.tile([-.9, -.25, .6, 1.1, 1.5, 1.9, 2.3, 2.7], (nk, 1))
    energy += np.arange(nk)[:, None]*np.linspace(-.02, .03, nb)[None]
    occ = np.broadcast_to([1., 1., 0., 0., 0., 0., 0., 0.], (nk, nb)).copy()
    put = lambda a, p: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, p))
    packed = psi[..., perm]
    enk, population = put(energy, P(None, None)), put(occ, P(None, None))
    parent = ParentGreenCarrier(psi_nmu=put(packed, PSI_NMU_SPEC),
        psi_mun=put(packed.transpose(0, 2, 3, 1), PSI_MUN_SPEC),
        enk=enk, occ=population, plan=plan, layout='face')
    slices = BandSlices.from_band_edges(0, 0, 2, nb, nb, b4_logical=nb)
    wfns = Wavefunctions(enk=enk, occ=population, slices=slices, green_parent=parent, layout='face')
    meta = Meta(rank=jax.process_index(), n_proc=4, b_id_0=0, b_id_1=0, b_id_2=2, b_id_3=nb,
        b_id_4=nb, b_id_4_user=nb, fft_grid=fftgrid, cell_volume=100.,
        n_rtot=64, n_rmu=mu, n_rmu_padded=mu, npol=ns, nfreq=1, nspin=1,
        nspinor=ns, nspinor_wfnfile=ns, nkx=4, nky=1, nkz=1, nk_tot=nk)
    meta.mu_basis = basis
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh,
                                              device_budget_bytes=int(device_budget_bytes()))
    meta.shared_pole_capacity.live_stages = ()
    a = rng.normal(size=(mu, mu))
    v0 = a@a.T*.02
    V = np.broadcast_to(v0, (nk, mu, mu)).copy().astype(complex)
    Vdev = put(V[:, perm][:, :, perm], P(None, 'x', 'y'))
    prepared = dict(mesh=mesh, kgrid=grid, nk=nk, nb=nb, spin_weight=1.,
        psi=parent.projection_faces()[0], V_q=Vdev, nmu_logical=mu)
    return locals()


def padding_stage(c):
    """Exercise the real distributed scalar pad gate with seven logical bands."""
    from dataclasses import replace
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from gw.wavefunction_bundle import BandSlices
    from tdagw.snapshot_runtime import prepare_native_source
    energy, occ, psi = (c[k].copy() for k in ('energy', 'occ', 'psi'))
    energy[:, 7] = 1234.5  # Producer sentinel: must never be edge-scissored.
    occ[:, 7] = 0.
    psi[:, 7] = 0.
    packed = psi[..., c['perm']]
    parent = replace(c['wfns'].green_parent,
        psi_nmu=c['put'](packed, PSI_NMU_SPEC),
        psi_mun=c['put'](packed.transpose(0, 2, 3, 1), PSI_MUN_SPEC))
    wfns = replace(c['wfns'], green_parent=parent,
        enk=c['put'](energy, P(None, None)), occ=c['put'](occ, P(None, None)),
        slices=BandSlices.from_band_edges(0, 0, 2, 7, 8, b4_logical=7))
    prepared = prepare_native_source(dict(wfns=wfns, meta=c['meta'], V_q=c['Vdev'],
        screening_inputs=dict(kwargs=dict(sym=c['sym']))), c['mesh'])
    assert prepared['nb'] == 7 and prepared['nb_padded'] == 8
    np.testing.assert_array_equal(prepared['energy_padded'], energy)
    np.testing.assert_array_equal(prepared['occupations_padded'], occ)
    np.testing.assert_array_equal(prepared['energy'], energy[:, :7])
    return dict(logical_bands=7, padded_bands=8, exact_zero_padding_accepted=True,
        producer_energy_sentinel_preserved=True,
        scope='real_native_prepare_collective_scalar_gate_not_full_snapshot_W_fit')


def hf_stage(c):
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from tdagw.tdgw_io import HA_IN_EV
    from tdagw.snapshot_runtime import native_hf_delta_model
    from tdagw.propagate import place_window_blocks
    from tests.tdagw.test_sc_native_reference_cpu import _literal_hf
    active, mesh, rng = slice(1, 3), c['mesh'], c['rng']
    req = np.broadcast_to(np.diag([1., 0.]), (4, 2, 2)).astype(complex).copy()
    a = rng.normal(size=(1, 4, 2, 2))+1j*rng.normal(size=(1, 4, 2, 2))
    delta = .03*(a+a.conj().swapaxes(-1, -2))
    rho = place_window_blocks(req[None]+delta, mesh, 4)
    reference = c['put'](req, P(('x', 'y'), None, None))
    model = native_hf_delta_model(c['prepared'], active, reference,
        bare_head_ha=.007, hartree_q0=c['Vdev'][0])
    run = jax.jit(lambda p, r: model.apply(p, r, 0.))
    executable = run.lower(model.params, rho).compile()
    hlo = executable.as_text()
    targets = re.findall(r'custom_call_target="([^"]+)"', hlo)
    native = [x for x in targets if re.search('kconv|kdft|cufft|mathdx', x, re.I)]
    if not native:
        raise RuntimeError('GATE native_HF_FFI: compiled CUDA FFT/Fock custom call absent')
    actual = np.asarray(gather_to_host(executable(model.params, rho)))
    expected = _literal_hf(c['psi'][:, active], c['V'], delta, 1.)*RYD_TO_EV/HA_IN_EV-.007*delta
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=3e-12)
    zero = np.asarray(gather_to_host(executable(model.params,
                          place_window_blocks(req[None], mesh, 4))))
    np.testing.assert_array_equal(zero, np.zeros_like(zero))
    memory = executable.memory_analysis()
    return dict(max_error_ha=float(np.max(abs(actual-expected))),
        native_custom_call_sites=len(native), native_targets=sorted(set(native)),
        count_scope='static_compiled_HF_HLO_sites_NOT_total_GPU_launches',
        compiled_temporary_bytes=memory.temp_size_in_bytes,
        hf_zero_at_reference=True, head_applications=1)


def tail_stage(c, out):
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw import ppm_sigma as ps
    from gw.band_extrapolation import BandLadder, fit_band_extrapolation_spectral
    from common.units import RYD_TO_EV
    from tests.test_tdagw_full_microscopic_cpu import _reciprocal_banks
    from tests._full_microscopic_oracle import density_vertices, retarded_reference_pairs, branch_coefficients
    from tests.test_tdagw_dynsig_sigma import _config
    # _config owns only tiny scalar controls; set its unit constant explicitly.
    import tests.test_tdagw_dynsig_sigma as old
    old.RYD = RYD_TO_EV
    nk, mu, mesh, rng = c['nk'], c['mu'], c['mesh'], c['rng']
    banks = _reciprocal_banks(rng, nk, mu)
    width = 4*mu
    left = np.stack([np.concatenate(list(r), axis=1) for _, r in banks])[:, c['perm'], None]
    right = np.broadcast_to(np.tile(np.eye(mu,dtype=complex), (1, 4)), (nk, mu, width))[:, c['perm'], None]
    poles = np.stack([np.repeat(p, mu) for p, _ in banks])
    identity = dict(iteration_id='P4-software-toy', hamiltonian='toy-H', energies='toy-E',
        occupations='toy-rho', wavefunctions='toy-psi', centroids='all-four-toy-ports')
    path = str(out/'toy_ordered_bank.h5')
    header = write_general_shared_pole_model(path, c['put'](left, P(None, 'x', None, 'y')),
        c['put'](right, P(None, 'x', None, 'y')), c['put'](poles, P()), np.full(nk, width),
        q_span=(0, nk), meta=c['meta'], tables=_tables(c), recipe=dict(accuracy='software_test'),
        receipts=dict(identity=identity), kmax=width)
    handle = dict(path=path, identity=identity, digest=header['digest'], K=header['K'],
                  representation='ordered_complex')
    source = ps.shared_pole_window_source(handle, c['meta'], efermi_ry=0., linalg='distributed')
    rotations, rho = [], []
    for _ in range(nk):
        u = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))[0]
        v = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))[0]
        rotations.append(u);rho.append(v@np.diag([.19, .83])@v.conj().T)
    rotations, rho = np.asarray(rotations), np.asarray(rho)
    e = c['energy'][:, 1:3]*RYD_TO_EV
    ladder = BandLadder(e_dft_ev=c['energy'].T*RYD_TO_EV,
        e_weyl_ev=np.arange(3.2, 5.6, .3)*RYD_TO_EV, w_k=np.full(nk, 1/nk),
        e0_ev=0., n0=0., c_ev=1., r2=1., estar_ev=10., n_dft=8, n_target=16, fit_window=(4, 8))
    def moment(lo, hi):
        nodes, weights = ladder._terms(lo, hi)
        return np.sum(weights/(nodes-e[..., None]+2.)**3, axis=-1)
    base = np.full(e.shape, .2+.4j)
    fit = fit_band_extrapolation_spectral([4, 6, 8],
        np.stack([base, base+moment(4, 6), base+moment(4, 8)]), ladder, e_state_ev=e)
    if fit.n_failed: raise RuntimeError('toy native tail fit failed')
    config = _config(np.array([-3., 4.]))
    config.sigma.regularization_ev=.14*RYD_TO_EV
    config.sigma.quadrature_eps=2e-7
    plan = ps.window_sigma_plan(config, window_only=True)
    plan['spectral_tail']=dict(fit=fit,input_frame='DECLARED_TOY',ladder_shift_ev=0.)
    window=dict(b_lo=1,U=rotations,e=c['energy'][:,1:3],rho=rho,
                f=np.diagonal(rho,axis1=-2,axis2=-1).real)
    got=ps.window_sigma_c(c['wfns'],source,window,plan,meta=c['meta'],mesh_xy=mesh,
                         branches=True,pairing='vertex',print_fn=lambda *a,**kw:None)
    orbitals=c['psi'].copy()
    orbitals[:,1:3]=np.einsum('kmsu,kmi->kisu',orbitals[:,1:3],rotations)
    density=np.zeros((nk,8,8),complex);density[:,0,0]=1.;density[:,1:3,1:3]=rho
    totals={cut:{s:np.zeros((2,nk,2,2),complex) for s in ('cond','val')}
            for cut in ('active',4,8)}
    for q,(p,r) in enumerate(banks):
        partner=(np.arange(nk)-q)%nk
        v=density_vertices(orbitals,partner,[1,2])
        for cut in totals:
            sl=slice(1,3) if cut=='active' else slice(0,cut)
            rp,rh=retarded_reference_pairs(v[:,:,sl],c['energy'][partner,sl],
                density[partner][:,sl,sl],c['energy'][:,1:3],p,r,.14)
            totals[cut]['cond']+=rp/nk;totals[cut]['val']+=rh/nk
    ratio=moment(8,16)/moment(4,8)
    weights=np.stack((np.broadcast_to(ratio[:,:,None],(nk,2,2)),
                      np.broadcast_to(ratio[:,None,:],(nk,2,2))))
    error=0.
    for space in ('cond','val'):
        expected=totals[8][space]+(totals[8][space]-totals[4][space])*weights
        np.testing.assert_allclose(got['retarded_pair'][space],expected,rtol=1e-5,atol=3e-8)
        error=max(error,float(np.max(abs(got['retarded_pair'][space]-expected))))
    from tdagw.collision import branch_operators
    gamma,injection,_=branch_operators(got['active_branch_pair']['cond'],got['active_branch_pair']['val'])
    # These are already small HOST tables; explicitly convert only after
    # collective gather for the device branch_operators return.
    from common.collectives import gather_to_host
    gamma,injection=(np.asarray(gather_to_host(a)) for a in (gamma,injection))
    target=branch_coefficients(totals['active']['cond'],totals['active']['val'])
    for a,b in ((gamma,target['gamma']),(injection,target['injection'])):
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=3e-8)
    receipt=got['native_receipt']
    return dict(max_complex_tail_error_ry=error, counts=[4,6,8],
        active_rates_match=True,retained_matrix_counts=[0,2,4],
        window_tau_pairs=receipt['window_tau_pairs'],n_windows=receipt['n_windows'],
        native_custom_call_launch_count='not_instrumented; HF static-HLO count reported separately')


def _tables(c):
    from gw.shared_pole_screening import _shared_pole_tables
    return _shared_pole_tables(c['meta'],c['sym'],c['centroids'])


def bse_stage(c,out,root):
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from symmetry_maps import QirrOperator
    from gw.head_correction import HeadSample
    import gw.static_screening as static
    from tests.test_tdagw_dynsig_sigma import _config
    module_path=root/'tests/tdagw/cri3_full_microscopic/native_bse_probe.py'
    spec=importlib.util.spec_from_file_location('native_bse_p4_callback',module_path)
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    mu,nk,rng=c['mu'],c['nk'],c['rng']
    # Declared toy static model is the sole replacement. Real BSE solver,
    # native CUDA FFT, head injection, residual and dipole contraction run.
    W=QirrOperator.whole_zone(c['put'](np.broadcast_to(np.eye(mu)*.025,(nk,mu,mu)).copy().astype(complex),P(None,'x','y')))
    sample=HeadSample(vc0=.02+0j,wcoul0=.01+0j,source='DECLARED_TOY_NOT_RPA',omega=0j)
    original=static.build_static_screened_w
    def declared_toy_static(*a,**kw):return W,sample
    static.build_static_screened_w=declared_toy_static
    U=np.broadcast_to(np.eye(8,dtype=complex),(nk,8,8)).copy()
    a=rng.normal(size=(3,nk,8,8))+1j*rng.normal(size=(3,nk,8,8))
    source=dict(mesh=c['mesh'],meta=c['meta'],config=_config(np.array([-3.,4.])),wfns=c['wfns'],
        energy_ry=c['energy'],occupations=c['occ'],U_active=c['put'](U,P()),
        velocity_dft_cart=c['put'](a+a.conj().swapaxes(-1,-2),P()),
        V_q=QirrOperator.whole_zone(c['Vdev']),G0_canonical=np.array([.1,.2,-.1,.15],complex),
        frame=dict(reference_ready=True,frame='DECLARED_TOY_NOT_SC_CONVERGENCE'),
        W_by_role=dict(iteration_head='DECLARED_TOY',shared_pole=dict(digest='DECLARED_TOY_NO_RPA')))
    try:
        report=probe.run_bse_probe(source,out=out/'toy_bse',nval=1,ncond=1,nroots=1,maxiter=4)
    finally:static.build_static_screened_w=original
    if max(report['residual_norm_ev'])>1e-5:raise AssertionError('native BSE residual')
    return dict(residual_ev=report['residual_norm_ev'],energies_ev=report['energies_ev'],
        iterations=report['iterations'],static_screening='DECLARED_TOY_ONLY',
        optical_strength=np.sum(report['strength_xyz_bohr2_cell']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-wall-seconds',type=float,default=180.)
    args=p.parse_args()
    if not 0<args.max_wall_seconds<=180:raise ValueError('max wall must be <=180s')
    if os.environ.get('LEG_HEAD')!=REVISION:raise ValueError('source revision must be authenticated by launcher LEG_HEAD')
    started=time.monotonic()
    watchdog=threading.Timer(args.max_wall_seconds,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    # IMPORTANT: real entry bootstrap precedes EVERY NumPy/JAX/fixture import.
    import gw.gw_jax as driver
    import runtime
    import jax
    import numpy as np
    from common.collectives import rank0_transaction,all_gather_processes
    from ffi.fft import kconv_backend
    root=Path(args.source_root).resolve()
    if root not in Path(driver.__file__).resolve().parents:raise ValueError('source import origin mismatch')
    mesh=driver.RUNTIME.mesh
    if (jax.process_count()!=4 or jax.local_device_count()!=1 or tuple(mesh.devices.shape)!=(2,2)
            or any(d.platform!='gpu' for d in jax.devices()) or kconv_backend(mesh,(4,1,1))!='mathdx'):
        raise ValueError('requires actual four GPU ranks, one device each, native CUDA FFT mesh2x2')
    out=Path(args.out).resolve()
    report=dict(status='RUNNING_SOFTWARE_ONLY',material_admitted=False,SC_calculation_performed=False,
        TD_propagated_fs=0,revision=REVISION,source_root=str(root),
        archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_wall_seconds=args.max_wall_seconds,max_active_GPUh=.2,
        fixture=dict(electronic_kgrid=[4,1,1],qgrid=[4,1,1],bands=8,mu=4,spinor=2),
        ffi_route='mathdx',stages={},native_launch_count_scope='not_global_instrumented')
    def save():
        def write():
            out.mkdir(parents=True,exist_ok=True)
            temporary=out/'receipt.tmp.json'
            temporary.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
            os.replace(temporary,out/'receipt.json')
        rank0_transaction(out,stage='adapters_p4/receipt',write=write)
    def create():
        if out.exists():raise FileExistsError(str(out))
        out.mkdir(parents=True)
    rank0_transaction(out,stage='adapters_p4/new_output',write=create)
    def phase(name,fn):
        elapsed=np.asarray(all_gather_processes(np.asarray(time.monotonic()-started))).max()
        if elapsed>=args.max_wall_seconds:raise TimeoutError('P4 software wall budget')
        report['phase']=name;save();before=time.monotonic()
        value=fn();jax.block_until_ready(value)
        times=np.asarray(all_gather_processes(np.asarray(time.monotonic()-before))).reshape(-1)
        report['stages'][name]=dict(max_seconds=float(times.max()),min_seconds=float(times.min()),result=value)
        save()
        return value
    rc=0
    try:
        before=time.monotonic()
        c=fixture(mesh)
        setup=np.asarray(all_gather_processes(np.asarray(time.monotonic()-before))).reshape(-1)
        report['fixture_seconds']=dict(max=float(setup.max()),min=float(setup.min()))
        phase('native_padding_scalar_gate',lambda:padding_stage(c))
        phase('native_HF_delta',lambda:hf_stage(c))
        phase('native_ordered_Sigma_tail',lambda:tail_stage(c,out))
        phase('native_BSE_declared_toy_static',lambda:bse_stage(c,out,root))
        report['status']='PASS_SOFTWARE_ONLY_NOT_MATERIAL_OR_TD'
    except BaseException as error:
        rc=1;report.update(status='REFUSED_SOFTWARE_PROOF',error=repr(error),traceback=traceback.format_exc())
    finally:
        report['elapsed_s']=time.monotonic()-started;save();watchdog.cancel()
    runtime.finalize_process(rc)
    return rc


if __name__=='__main__':raise SystemExit(main())
