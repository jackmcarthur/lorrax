"""Quantify recoverability of E4 from rotations and rounded output spectra.

This is a feasibility audit, NOT an authenticated SC restart. rCROP input H2
is affine in H0 and F(H0); H4 is affine in H0, H2 and F(H2). Saved input
rotations constrain these affine coefficients by U^dagger H U being diagonal.
"""
from pathlib import Path
import argparse
import json
import os
import numpy as np
from gw.eqp_bgw import read_bgw_eqp
from common.units import RYD_TO_EV


def offdiag(a):
    out=a.copy()
    j=np.arange(a.shape[-1])
    out[:,j,j]=0
    return out


def rotate(h,u):
    return u.conj().transpose(0,2,1)@h@u


def recover(base, directions, u):
    """Infer affine coefficients from vanishing QP-basis off-diagonals."""
    rhs=-offdiag(rotate(base,u)).reshape(-1)
    cols=np.stack([offdiag(rotate(h,u)).reshape(-1) for h in directions],axis=-1)
    a=np.concatenate([cols.real,cols.imag])
    b=np.concatenate([rhs.real,rhs.imag])
    coeff,_,_,s=np.linalg.lstsq(a,b,rcond=None)
    h=base+sum(c*d for c,d in zip(coeff,directions))
    return h,dict(coefficients=coeff.tolist(),singular_values=s.tolist(),
                  max_offdiag_ev=float(np.max(abs(offdiag(rotate(h,u))))))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    k,e0,f0,b0=read_bgw_eqp(str(args.source/'eqp0_iter0000.dat'))
    k2,_,f2,b2=read_bgw_eqp(str(args.source/'eqp0_iter0002.dat'))
    assert np.array_equal(k,k2) and b0==b2==0
    # Canonical flat 8^3 full-k ordering; use only explicitly stored file rows.
    ijk=np.rint(np.mod(k,1)*8).astype(int)%8
    assert np.max(abs(np.mod(k,1)*8-np.rint(np.mod(k,1)*8)))<1e-7
    idx=np.ravel_multi_index(ijk.T,(8,8,8))
    u={i:np.load(args.source/f'sc_history/rotation_iter{i:04d}.npy',mmap_mode='r')[idx] for i in (1,2,3,4)}
    eye=np.eye(86)[None]
    h0=e0[:,:,None]*eye
    h1=(u[1]*f0[:,None,:])@u[1].conj().transpose(0,2,1)
    h3=(u[3]*f2[:,None,:])@u[3].conj().transpose(0,2,1)
    h2,r2=recover(h0,[h1-h0],u[2])
    h4,r4=recover(h0,[h1-h0,h3-h0],u[4])
    energies=np.diagonal(rotate(h4,u[4]),axis1=1,axis2=2).real
    result=dict(jid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
                units='eV',map2=r2,map4=r4,
                approximation_only=True,
                reason='Input spectra were not persisted; source eqp spectra are rounded to 1e-9 eV. No occupation hash authentication is possible from these files alone.',
                min_input_ev=float(energies.min()),max_input_ev=float(energies.max()))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'approximate_input_map0004_NOT_AUTHENTICATED.npz',
             kpoints=k,H_input_ev=h4,E_input_ev=energies,U_input=u[4])
    (args.output/'reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
