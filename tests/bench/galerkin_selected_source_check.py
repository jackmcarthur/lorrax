"""P4 actual-WFN parity of selected and retained Galerkin basis streams."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R = initialize_communicator_stack()
import argparse
from pathlib import Path
from dataclasses import replace
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, gather_to_host, rank0_transaction
from common.meta import Meta
from common.psi_G_store import build_psi_G_store
from common.gamma_matrices import dirac_spin_z, sigma_z
from common.wfn_transforms import get_enk_bandrange
from gw.qsgw_head import qp_frame_delta_h_dft
from file_io import WFNReader
from isdf.galerkin import (GalerkinBasis, iter_galerkin_rchunks,
    project_galerkin_spin_operator, project_lifted_galerkin_dirac_spin)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--wfn', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()

def main():
    mesh = R.mesh
    w = WFNReader(a.wfn, mesh=mesh)
    sym = w.symmetry()
    receipts = []
    # The post-SC velocity receives the authenticated QP frame in the DFT
    # state basis. Test its reconstruction against an independent dense
    # reference, with complex mixing, real WFN energies and null padding.
    dft,_=get_enk_bandrange(w,sym,(1,6),(1,6))
    dft=np.asarray(dft)
    rng=np.random.default_rng(90)
    U=np.stack([np.linalg.qr(rng.normal(size=(5,5))+
               1j*rng.normal(size=(5,5)))[0] for _ in range(len(dft))])
    qp=dft+np.linspace(-.02,.03,5)
    expected=(U*qp[:,None,:])@U.swapaxes(-1,-2).conj()
    expected-=dft[:,:,None]*np.eye(5)[None]
    with mesh:
        U_dev=device_put_process_local(np.pad(U,((0,0),(0,3),(0,3))),
                                       NamedSharding(mesh,P(None,'x','y')))
        qp_dev=device_put_process_local(qp,NamedSharding(mesh,P()))
        dft_dev=device_put_process_local(dft,NamedSharding(mesh,P()))
        delta=qp_frame_delta_h_dft(U_dev,qp_dev,dft_dev,mesh=mesh)
    delta=np.asarray(gather_to_host(delta))
    qp_delta_error=float(np.linalg.norm(delta[:,:5,:5]-expected)/np.linalg.norm(expected))
    assert qp_delta_error<5e-12,qp_delta_error
    assert np.max(np.abs(delta[:,5:]))==0
    assert np.max(np.abs(delta[:,:,5:]))==0
    for bispinor in (False, True):
        meta = Meta.from_system(w, sym, nval=1, ncond=6, nband=7,
                                n_rmu=1, bispinor=bispinor)
        ns = int(meta.nspinor)
        nk = int(meta.nk_tot)
        nb, rank, physical = 5, 8, 5
        candidates = np.arange(nk * nb)
        selected = candidates[np.linspace(0, len(candidates)-1, physical, dtype=int)]
        assert len(np.unique(selected)) == physical
        rng = np.random.default_rng(12)
        L = np.eye(rank, dtype=complex)
        L[:physical,:physical] += .07*np.tril(
            rng.normal(size=(physical,physical)) + 1j*rng.normal(size=(physical,physical)), -1)
        with mesh:
            factor = device_put_process_local(L, NamedSharding(mesh,P()))
            c = device_put_process_local(np.zeros((nk,nb,rank),complex), NamedSharding(mesh,P()))
            b = device_put_process_local(np.zeros((rank,ns,1),complex), NamedSharding(mesh,P(None,None,None)))
        basis = GalerkinBasis(ctilde=c,basis_at_nodes=b,rank_physical=physical,
            band_range=(1,6),selected_state_indices=tuple(map(int,selected)),selection_factor=factor)
        # Different r carriers, a nonzero band start and final one-band chunk.
        nrt = int(meta.n_rtot)
        cut = min(36, nrt-4)
        ranges = ((0,cut),(cut,nrt))
        with build_psi_G_store(wfn=w,mesh_xy=mesh,meta=meta,
                band_chunk_ranges=((1,5),(5,6)),band_pad_to=4,
                bispinor=bispinor) as source:
            source_calls = {'full':0,'selected':0}
            old_all = source.iter_rchunk_bandwise
            old_selected = source.iter_rows_rchunks
            def full(*args, **kwargs):
                source_calls['full'] += 1
                yield from old_all(*args,**kwargs)
            def subset(*args, **kwargs):
                source_calls['selected'] += 1
                yield from old_selected(*args,**kwargs)
            source.iter_rchunk_bandwise = full
            source.iter_rows_rchunks = subset
            def read(retain):
                values=[]
                with mesh:
                    for r0,r1,x,parts in iter_galerkin_rchunks(source,basis,meta,mesh,
                            r_chunk_ranges=ranges,retained_band_range=retain):
                        values.append(np.asarray(gather_to_host(x))[:,:,:r1-r0])
                        assert bool(parts) == (retain is not None)
                return np.concatenate(values,axis=2)
            start=time.monotonic()
            legacy=read((1,6))
            legacy_calls=source_calls['full']
            selected_basis=read(None)
            assert source_calls['full'] == legacy_calls, 'selected route transformed full k/band table'
            assert source_calls['selected'] > 0
            rel=float(np.linalg.norm(selected_basis-legacy)/np.linalg.norm(legacy))
            assert rel<5e-13,rel
            # Exact-null carrier rows must survive selection and the solve.
            assert np.max(np.abs(selected_basis[physical:]))==0
            op=rng.normal(size=(ns,ns))+1j*rng.normal(size=(ns,ns))
            op=.5*(op+op.conj().T)
            reference=np.einsum('asr,st,btr->ab',legacy.conj(),op,legacy,optimize=True)
            metric=np.einsum('asr,bsr->ab',legacy.conj(),legacy,optimize=True)
            with mesh:
                projection=project_galerkin_spin_operator(source,basis,meta,mesh,
                    spin_operator=op,q_tile_budget=4096)
            got=np.asarray(gather_to_host(projection.operator))
            norm=np.asarray(gather_to_host(projection.metric))
            error=float(np.linalg.norm(got-reference)/np.linalg.norm(reference))
            norm_error=float(np.linalg.norm(norm-metric)/np.linalg.norm(metric))
            assert error<5e-12,error
            assert norm_error<5e-12,norm_error
            assert np.max(np.abs(got-got.conj().T))<5e-12
            # Physical spin includes both large and small Dirac blocks. Use
            # an independent diagonal contraction on the actual WFN basis;
            # a Pauli-only upper block must fail when small components exist.
            physical_spin = dirac_spin_z if bispinor else 0.5*sigma_z
            spin_weights = np.tile([0.5,-0.5], ns//2)
            spin_reference=np.einsum('asr,s,bsr->ab',legacy.conj(),
                                    spin_weights,legacy,optimize=True)
            with mesh:
                if bispinor:
                    pauli_meta=replace(meta,nspinor=2,npol=1)
                    pauli_basis=replace(basis,basis_at_nodes=b[:,:2])
                    spin_projection=project_lifted_galerkin_dirac_spin(
                        source,pauli_basis,pauli_meta,mesh,component=2,q_tile_budget=4096)
                else:
                    spin_projection=project_galerkin_spin_operator(source,basis,meta,mesh,
                        spin_operator=np.asarray(physical_spin),q_tile_budget=4096)
            spin_got=np.asarray(gather_to_host(spin_projection.operator))
            # Sz can cancel in this fixture's excited-state subset. Scale
            # its absolute error by the complete positive carrier metric,
            # the same physical scale used for normalized expectations.
            spin_absolute=float(np.linalg.norm(spin_got-spin_reference))
            spin_error=spin_absolute/float(np.linalg.norm(metric))
            spin_reference_scale=float(np.linalg.norm(spin_reference)/np.linalg.norm(metric))
            assert spin_error<5e-12,spin_error
            if bispinor:
                pauli_only=np.einsum('asr,s,bsr->ab',legacy.conj(),
                                    [0.5,-0.5,0,0],legacy,optimize=True)
                assert np.linalg.norm(spin_got-pauli_only)>1e-8*np.linalg.norm(spin_reference)
            # Negative control: a changed physical operator must be detected.
            wrong=np.einsum('asr,st,btr->ab',legacy.conj(),op+.1*np.eye(ns),legacy,optimize=True)
            assert np.linalg.norm(got-wrong)>1e-4*np.linalg.norm(reference)
            row=dict(bispinor=bispinor,nspinor=ns,nk=nk,rank=rank,
                selected_states=selected.tolist(),ranges=ranges,
                basis_relative_error=rel,operator_relative_error=error,
                metric_relative_error=norm_error,source_calls=source_calls,
                physical_spin_relative_error=spin_error,
                physical_spin_absolute_error=spin_absolute,
                physical_spin_reference_metric_scale=spin_reference_scale,
                qp_frame_delta_relative_error=qp_delta_error,
                wall_seconds=time.monotonic()-start)
            receipts.append(row)
            rank0_print(json.dumps(row),flush=True)
    rank0_transaction(a.output,stage='selected basis actual WFN parity',
        write=lambda:Path(a.output).write_text(json.dumps(receipts,indent=2)+'\n'))
    w.close()
    return 0

run_main_and_finalize(main)
