"""P4 active-band Sigma integration against frozen full-GEMM owners.

This exercises typed parent operands, the real spatial convolution/projector,
antiunitary transport and bracket additivity. Only local output shards are read.

Invoke ``run_gate(mesh, frozen_dir, output_dir)`` from a runtime-initialized
P4 runner; this is an integration gate, not a standalone pytest module.
The run-owned runner supplies the immutable full-GEMM source snapshot.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import time


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(mesh, ns, nb=32, mu=16):
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.wavefunction_bundle import Wavefunctions, ParentGreenCarrier, BandSlices
    from symmetry_maps import spinor_rotation_for_sym_row

    k = np.column_stack((np.arange(4)/4, np.zeros((4, 2))))
    parent_rows = np.asarray([0, 1, 2], np.int32)
    irr = np.asarray([0, 1, 2, 1], np.int32)
    symrows = np.asarray([0, 0, 0, 1], np.int32)
    spatial = np.eye(3, dtype=int)[None]
    def spinor_action(rows, *, nspinor):
        return spinor_rotation_for_sym_row(
            np.eye(2, dtype=complex)[None], np.asarray(rows), 1,
            nspinor=nspinor)
    sym = SimpleNamespace(sym_matrices=spatial, translations=np.zeros((1,3)),
        sym_mats_k=np.concatenate((spatial,-spatial)), irr_idx_k=irr,
        sym_idx_k=symrows, spinor_action=spinor_action, unfolded_kpts=k,
        kirr_fullids=parent_rows, nk_red=3, nk_tot=4)
    points = np.column_stack((np.arange(mu), np.zeros((mu,2),int)))
    plan = build_centroid_k_unfold_plan(sym,points,(mu,1,1),mesh,
                                       nspinor=ns,parent_k_frac=k[parent_rows])
    rng = np.random.default_rng(901+ns)
    psi = (rng.normal(size=(3,nb,ns,mu))+1j*rng.normal(size=(3,nb,ns,mu)))/np.sqrt(nb)
    psi = plan.layout.axis.pack_host(psi,axis=3)
    def put(x,spec):return device_put_process_local(np.asarray(x),NamedSharding(mesh,P(*spec)))
    energy = np.broadcast_to(np.linspace(-1,1,nb),(3,nb)).copy()
    # Exact window-boundary bands, including a degenerate threshold.
    energy[:,8:10] = -.25
    E = put(energy,(None,None));occ=put(np.ones_like(energy),(None,None))
    carrier = ParentGreenCarrier(put(psi,(None,'x',None,'y')),
        put(psi.transpose(0,2,3,1),(None,None,'x','y')), E,occ,plan)
    wfns = Wavefunctions(enk=put(energy[irr],(None,None)),
                         occ=put(np.ones_like(energy[irr]),(None,None)),slices=BandSlices.from_band_edges(0,0,0,8,nb),
                         layout='face',green_parent=carrier)
    return wfns,put,rng


def run_gate(mesh, frozen_dir, output_dir):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gw import ppm_tau_kernel as candidate
    from gw import greens_function_kernel as livegreen
    from gw.wavefunction_bundle import parent_sigma_operands,sigma_face_kernel_kwargs
    from common.collectives import barrier
    frozen_dir,output_dir = Path(frozen_dir),Path(output_dir)
    oldgreen = _load('gw._active_gate_old_green',frozen_dir/'greens_function_kernel.py')
    oldtau = _load('gw._active_gate_old_tau',frozen_dir/'ppm_tau_kernel.py')
    records=[]
    brackets=((0,3),(3,19),(19,32),(9,9))
    def discrepancy(a,b):
        error=max(float(np.max(np.abs(np.asarray(x.data)-np.asarray(y.data)),initial=0))
                  for x,y in zip(a.addressable_shards,b.addressable_shards))
        scale=max(float(np.max(np.abs(np.asarray(x.data)),initial=0)) for x in b.addressable_shards)
        assert error < 2e-11*max(scale,1), (error,scale)
        return error,scale
    for ns in (1,2):
        wfns,put,rng = _fixture(mesh,ns)
        xn,yr,xr,yn,E,_ = parent_sigma_operands(wfns)
        kwargs=sigma_face_kernel_kwargs(wfns)
        mu=wfns.green_parent.plan.n_centroid_packed
        for windows in (False,True):
            factories={}
            for mode,module in [('baseline',oldtau),('active',candidate)]:
                original=livegreen
                if mode=='baseline':sys.modules['gw.greens_function_kernel']=oldgreen
                try:
                    factories[mode]=[module._get_sigma_kij_kernel(mesh_xy=mesh,kgrid=(4,1,1),
                        brackets=b,energy_windows=windows,**kwargs) for b in (brackets,None)]
                finally:sys.modules['gw.greens_function_kernel']=original
            for weight_kind in ('bool','signed','complex'):
                weight=rng.random(E.shape)>.3
                if weight_kind!='bool':weight=weight*(rng.normal(size=E.shape))
                if weight_kind=='complex':weight=weight+1j*rng.normal(size=E.shape)
                # Different exact supports per parent, holes inside intervals,
                # and one wholly empty parent exercise per-parent trimming.
                support=np.zeros(E.shape,dtype=bool)
                support[1,3:14]=True; support[1,7]=False
                support[2,19:29]=True; support[2,23]=False
                weight=np.where(support,weight,0).astype(weight.dtype)
                selector=put(weight,())
                W=(rng.normal(size=(4,mu,mu))+1j*rng.normal(size=(4,mu,mu)))/mu
                outputs={}
                for mode in ('baseline','active'):
                    outputs[mode]=[]
                    for bracketed,fn in zip((True,False),factories[mode]):
                        # W is donated, so allocate a distinct input for every call.
                        args=(xn,yr,xr,yn,E,selector)
                        if windows:args+=tuple(put(x,()) for x in (-.25,.6))
                        args+=tuple(put(x,()) for x in (.13,.2-.31j))+(put(W,(None,'x','y')),)
                        barrier('active_sigma_before')
                        start=time.perf_counter();value=fn(*args);jax.block_until_ready(value)
                        elapsed=time.perf_counter()-start
                        outputs[mode].append(value)
                        records.append(dict(ns=ns,windows=windows,weight=weight_kind,mode=mode,
                                            bracketed=bracketed,first_call_s=elapsed))
                for i in (0,1):
                    error,scale=discrepancy(outputs['active'][i],outputs['baseline'][i])
                    records[-1].setdefault('parity',[]).append(dict(error=error,scale=scale))
                for mode in ('baseline','active'):
                    discrepancy(jnp.sum(outputs[mode][0],axis=0),outputs[mode][1])
                    assert all(np.count_nonzero(np.asarray(x.data))==0
                               for x in outputs[mode][0][3].addressable_shards)
    return records
