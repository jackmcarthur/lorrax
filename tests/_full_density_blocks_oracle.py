"""Literal transition-index oracle for independent coherent ordered products.

For nonconjugated ket psi[k,b,s,mu], M multiplies on the LEFT of the
right-index energy phase: G_M=psi M exp(-E*t) psi^H. No FFT, production
contraction, quadrature fit, population pinch or reverse shortcut is used.
"""
import numpy as np


def pair_spectrum(psi, energy, left, right, partner, spin_weight=1.):
    """Return d=E_left-E_right and full-port residues of B(left,right;t)."""
    nk, nb, ns, nmu=psi.shape
    gaps=[];residues=[]
    for k in range(nk):
        kq=int(partner[k])
        for b in range(nb):
            for d in range(nb):
                amplitude=np.zeros(nmu,complex)
                vertex=np.zeros(nmu,complex)
                for mu in range(nmu):
                    for spin in range(ns):
                        vertex[mu]+=psi[k,b,spin,mu]*psi[kq,d,spin,mu].conjugate()
                        for a in range(nb):
                            for c in range(nb):
                                amplitude[mu]+=(left[k,a,b]*right[kq,c,d].conjugate()
                                    *psi[k,a,spin,mu]*psi[kq,c,spin,mu].conjugate())
                gaps.append(energy[k,b]-energy[kq,d])
                residues.append(np.outer(amplitude,vertex.conjugate())*spin_weight/nk)
    return np.asarray(gaps),np.asarray(residues)


def pair_time(spectrum,times):
    gaps,residues=spectrum
    return np.stack([sum((r*np.exp(-d*t) for d,r in zip(gaps,residues)),
                         np.zeros_like(residues[0])) for t in times])


def pair_frequency(spectrum,sites,gamma):
    gaps,residues=spectrum
    return np.stack([sum((r/(z-d+1j*gamma) for d,r in zip(gaps,residues)),
                         np.zeros_like(residues[0])) for z in sites])


def commutator_moments(forward,reverse,gamma):
    """Exact coefficients X1/z+X2/z²; gamma contributes -i gamma X1."""
    df,rf=forward;dr,rr=reverse
    first=rf.sum(axis=0)-rr.sum(axis=0)
    second=np.einsum('i,ijk->jk',df,rf)-np.einsum('i,ijk->jk',dr,rr)-1j*gamma*first
    return first,second


def four_block_panels(density,active):
    """Frozen0/1 + full active density; no density eigensystem is used."""
    nk,nb,_=density.shape
    mask=np.zeros(nb,bool);mask[active]=True
    diagonal=np.real(np.diagonal(density,axis1=-2,axis2=-1))
    outside=~mask
    if np.any((diagonal[:,outside]!=0)&(diagonal[:,outside]!=1)):
        raise ValueError('oracle requires exactly0/1 frozen spectators')
    allowed=np.outer(mask,mask)|np.eye(nb,dtype=bool)
    if np.max(abs(density[:,~allowed]),initial=0.)!=0:
        raise ValueError('oracle requires all density coherences inside the numerical group')
    occ=np.zeros_like(density);empty=np.zeros_like(density)
    for k in range(nk):
        for b in np.flatnonzero(outside):
            occ[k,b,b]=diagonal[k,b]
            empty[k,b,b]=1-diagonal[k,b]
    identity=np.broadcast_to(np.diag(mask.astype(float)),density.shape).astype(complex).copy()
    occupied=np.zeros_like(density);occupied[:,active,active]=density[:,active,active]
    unoccupied=identity-occupied
    return dict(FF=(empty,occ),FA=(unoccupied,occ),AF=(empty,occupied),AA=(unoccupied,occupied))


def pair_interval(energy,left_columns,right_columns):
    """Exact population-independent current global energy box."""
    el=energy[:,left_columns];er=energy[:,right_columns]
    lo=float(el.min()-er.max());hi=float(el.max()-er.min())
    return dict(lo=lo,hi=hi,positive=bool(lo>0),
                left_min=float(el.min()),left_max=float(el.max()),
                right_min=float(er.min()),right_max=float(er.max()))
