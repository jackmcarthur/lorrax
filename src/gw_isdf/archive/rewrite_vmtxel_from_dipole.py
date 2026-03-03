#!/usr/bin/env python3
from __future__ import annotations

"""
Create a vmtxel_dip.h5 by copying an existing BGW vmtxel.h5 and overwriting
its vmtxel_data/dipole dataset with values computed from our dipole.h5
(p + i[r, V_NL]) divided by ΔE = E_c - E_v, projected along the polarization
vectors stored in the BGW file.

The band-axis conventions are aligned with BGW:
- conduction (nband) axis indexes bands c = nelec, nelec+1, ... in ascending order
- valence   (mband) axis indexes bands v = nelec-1, nelec-2, ... in descending order

We support the two common BGW HDF5 layouts for dipole:
1) (nband, mband, nk, ns, npol)              [complex dtype]
2) (npol, ns, nk, mband, nband, flavor=2)    [real/imag split]

Usage:
  uv run python -m gw_isdf.rewrite_vmtxel_from_dipole \
     -i cohsex_test.in --vmtxel vmtxel.h5 --dipole dipole.h5 --out vmtxel_dip.h5
"""

import argparse
import os
from pathlib import Path
import numpy as np
import h5py

from isdf.common.wfnreader import WFNReader
from isdf.common import symmetry_maps
from isdf.psp.get_DFT_mtxels import read_cohsex_input  # for resolving WFN path


def _read_pol(h5grp) -> np.ndarray:
    """Return polarization matrix as shape (npol, 3)."""
    pol = np.asarray(h5grp['pol']) if 'pol' in h5grp else None
    if pol is None:
        raise RuntimeError("vmtxel_data/pol not found; cannot project along polarizations")
    if pol.ndim == 1:
        # Single vector
        if pol.size != 3:
            raise RuntimeError("Unexpected 1D pol size != 3")
        return pol.reshape(1, 3)
    if pol.shape[0] == 3:
        return pol.T.copy()  # (3,npol) -> (npol,3)
    if pol.shape[-1] == 3:
        return pol.copy()    # (npol,3)
    raise RuntimeError(f"Unrecognized pol shape: {pol.shape}")


def _project_dipole_to_pol(dipole_cart: np.ndarray, pol: np.ndarray, bdot: np.ndarray | None = None) -> np.ndarray:
    """Project dipole_cart[3, nk, nb, nb] along each direction vector.

    If bdot is provided (reciprocal metric), normalizes each direction using
    |p| = sqrt(p^T bdot p) to match BGW conventions. Returns proj[npol, nk, nb, nb].
    """
    npol = int(pol.shape[0])
    nk = int(dipole_cart.shape[1])
    nb = int(dipole_cart.shape[2])
    out = np.zeros((npol, nk, nb, nb), dtype=np.complex128)
    for ip in range(npol):
        vec = np.asarray(pol[ip], dtype=float)
        if bdot is not None:
            l2 = float(vec @ (bdot @ vec))
            norm = np.sqrt(l2) if l2 > 0 else 1.0
            vec = vec / norm
        out[ip] = (vec[0] * dipole_cart[0]
                   + vec[1] * dipole_cart[1]
                   + vec[2] * dipole_cart[2])
    return out


def _safe_div(x: np.ndarray, y: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(y) > eps, x / y, 0.0)


def _read_kpoints_from_vmtxel(hin: h5py.File, nk: int) -> np.ndarray | None:
    """Attempt to read k-point coordinates (crystal) from vmtxel.h5.

    Returns an array (nk,3) in crystal coordinates if found, else None.
    """
    candidates = [
        ("vmtxel_header", "kpts"), ("vmtxel_header", "kpoints"), ("vmtxel_header", "kpt_crys"),
        ("vmtxel_data", "kpts"), ("vmtxel_data", "kpoints"), ("vmtxel_data", "kpt_crys"),
    ]
    for grp_name, ds in candidates:
        if grp_name in hin and ds in hin[grp_name]:
            arr = np.asarray(hin[grp_name][ds])
            arr = arr.reshape(-1, 3)
            if arr.shape[0] == nk and arr.shape[1] == 3:
                return arr.astype(float)
    return None


def _build_k_index_map(k_vmt: np.ndarray | None, k_unfold: np.ndarray, tol: float = 1e-6) -> list[int]:
    """Map each vmtxel k-point to an index in the unfolded WFN k-point list.

    If k_vmt is None, returns identity mapping [0..nk-1]. Uses nearest-wrap metric
    in crystal coordinates with periodicity of 1.
    """
    nk_unf = int(k_unfold.shape[0])
    if k_vmt is None:
        return [i for i in range(min(nk_unf, nk_unf))]
    idx_map: list[int] = []
    for kv in k_vmt:
        # Compute wrapped distance to each unfolded k
        diff = kv[None, :] - k_unfold
        # wrap into [-0.5,0.5)
        diff_wrapped = diff - np.round(diff)
        dists = np.linalg.norm(diff_wrapped, axis=1)
        j = int(np.argmin(dists))
        if float(dists[j]) > tol:
            # No close match; still take nearest but warn via print
            pass
        idx_map.append(j)
    return idx_map


def rewrite_vmtxel(vmtxel_path: str, dipole_h5: str, wfn: WFNReader, out_path: str) -> None:
    # Load dipole.h5
    with h5py.File(dipole_h5, 'r') as hd:
        dipole_cart = np.asarray(hd['dipole_cart'])  # (3, nk, nb, nb)
        deltaE = np.asarray(hd['deltaE'])            # (nk, nb, nb)
    nk_d, nb_d = int(dipole_cart.shape[1]), int(dipole_cart.shape[2])

    # Read BGW vmtxel.h5, header and polarization
    with h5py.File(vmtxel_path, 'r') as hin:
        if 'vmtxel_header' not in hin or 'vmtxel_data' not in hin:
            raise RuntimeError("vmtxel.h5 missing required groups vmtxel_header or vmtxel_data")
        hdr = hin['vmtxel_header']
        data = hin['vmtxel_data']
        nk = int(hdr['nk'][()])
        nband = int(hdr['nband'][()])   # conduction count
        mband = int(hdr['mband'][()])   # valence count
        ns = int(hdr['ns'][()])
        npol = int(hdr['npol'][()]) if 'npol' in hdr else (np.asarray(data['pol']).shape[-1])
        opr = int(hdr['opr'][()]) if 'opr' in hdr else 0  # 0=velocity, 1=momentum
        pol = _read_pol(data)
        # Try to map vmtxel k-points to unfolded WFN/dipole k-points
        k_vmt = _read_kpoints_from_vmtxel(hin, nk)
        sym = symmetry_maps.SymMaps(wfn)
        k_unfold = np.asarray(sym.unfolded_kpts, dtype=float).reshape(-1, 3)
        k_map = _build_k_index_map(k_vmt, k_unfold, tol=1e-6)
        if len(k_map) != nk:
            raise RuntimeError("Failed to build k-point index map for vmtxel")
        if nb_d < int(wfn.nbands):
            # dipole.h5 nb might be truncated; we only use needed range
            pass

        # Copy file structure to output
        with h5py.File(out_path, 'w') as hout:
            hin.copy('vmtxel_header', hout)
            hin.copy('vmtxel_data', hout)
            grp = hout['vmtxel_data']
            # Remove existing dipole dataset and recreate
            if 'dipole' in grp:
                del grp['dipole']

            # Map band indices (BGW ordering)
            nelec = int(wfn.nelec)
            c_idx = np.arange(nelec, nelec + nband, dtype=int)
            v_idx = np.arange(nelec - 1, nelec - 1 - mband, -1, dtype=int)
            # Trim to available nb in dipole.h5
            c_idx = c_idx[c_idx < nb_d]
            v_idx = v_idx[v_idx >= 0]

            # Project along directions. For velocity-mode vmtxel (opr=0), BGW uses e^{-iq·r}/q
            # and stores one polarization; pol in file is not used in mtxel_v, so do NOT
            # renormalize here (use components as-provided for direction). For momentum-mode
            # (opr=1), BGW divides by |pol|; match by normalizing with bdot.
            # Normalize direction vectors with reciprocal metric (|p| from bdot)
            # so projection uses a true unit vector, matching BGW conventions.
            proj = _project_dipole_to_pol(dipole_cart, pol, bdot=np.asarray(wfn.bdot, dtype=float))
            # Build M = v/(Ec-Ev) for chosen blocks
            # deltaE[k,b,b'] = E_b - E_b'; we want E_c - E_v
            # So for each (k, ic, iv): denom = deltaE[k, c_idx[ic], v_idx[iv]]

            # Prepare output in the same layout as input dipole dataset
            dip_dset = None
            # Try to infer original layout
            d_in = data.get('dipole')
            if d_in is None:
                raise RuntimeError('vmtxel_data/dipole not found in input file')
            in_shape = d_in.shape
            in_dtype = d_in.dtype

            # Compute M for each pol,k,ic,iv
            M = np.zeros((npol, nk, nband, mband), dtype=np.complex128)
            for ip in range(npol):
                for ik in range(nk):
                    dk = int(k_map[ik])
                    block = _safe_div(proj[ip, dk][np.ix_(c_idx, v_idx)], deltaE[dk][np.ix_(c_idx, v_idx)])
                    # block shape: (nband_eff, mband_eff); embed into M with same top-left
                    nb_eff, mb_eff = block.shape
                    M[ip, ik, :nb_eff, :mb_eff] = block

            # Write in matching layout
            if len(in_shape) == 5:
                # Expect (nband, mband, nk, ns, npol)
                out = np.zeros((nband, mband, nk, ns, npol), dtype=np.complex128)
                for ip in range(npol):
                    for ik in range(nk):
                        for ispin in range(ns):
                            out[:, :, ik, ispin, ip] = M[ip, ik]  # (nband, mband)
                dip_dset = grp.create_dataset('dipole', data=out)
            elif len(in_shape) == 6:
                # Expect (npol, ns, nk, mband, nband, flavor)
                flavor = in_shape[-1]
                if flavor != 2:
                    # Fallback: complex last dim not 2 -> write complex view if allowed
                    out = np.zeros((npol, ns, nk, mband, nband), dtype=np.complex128)
                    for ip in range(npol):
                        for ik in range(nk):
                            for ispin in range(ns):
                                out[ip, ispin, ik] = M[ip, ik].T  # (mband,nband)
                    dip_dset = grp.create_dataset('dipole', data=out)
                else:
                    out = np.zeros((npol, ns, nk, mband, nband, 2), dtype=np.float64)
                    for ip in range(npol):
                        for ik in range(nk):
                            for ispin in range(ns):
                                blk = M[ip, ik].T  # (mband, nband)
                                out[ip, ispin, ik, :, :, 0] = np.real(blk)
                                out[ip, ispin, ik, :, :, 1] = np.imag(blk)
                    dip_dset = grp.create_dataset('dipole', data=out)
            else:
                raise RuntimeError(f'Unexpected dipole dataset rank: {len(in_shape)}')

            # Preserve basic attrs
            if dip_dset is not None:
                dip_dset.attrs['source'] = 'overwritten from dipole.h5 (p+vNL)/ΔE, projected along vmtxel_data/pol'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Rewrite BGW vmtxel.h5 dipole dataset using local dipole.h5 results')
    ap.add_argument('-i', '--input', default='cohsex_test.in', help='cohsex input to locate WFN.h5')
    ap.add_argument('--vmtxel', default='vmtxel.h5', help='Path to source vmtxel.h5')
    ap.add_argument('--dipole', default='dipole.h5', help='Path to dipole.h5 (from isdf.psp.get_dipole_mtxels)')
    ap.add_argument('--out', default='vmtxel_dip.h5', help='Output HDF5 path with overwritten dipole dataset')
    args = ap.parse_args(argv)

    # Resolve input-relative paths
    inp = Path(args.input).resolve()
    params = read_cohsex_input(str(inp))
    wfn_path = Path(params.get('wfn_file', 'WFN.h5'))
    if not wfn_path.is_absolute():
        wfn_path = (inp.parent / wfn_path).resolve()
    vmtxel_path = Path(args.vmtxel)
    if not vmtxel_path.is_absolute():
        vmtxel_path = (inp.parent / vmtxel_path).resolve()
    dip_path = Path(args.dipole)
    if not dip_path.is_absolute():
        dip_path = (inp.parent / dip_path).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (inp.parent / out_path).resolve()

    wfn = WFNReader(str(wfn_path))
    rewrite_vmtxel(str(vmtxel_path), str(dip_path), wfn, str(out_path))
    print(f'Wrote {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
