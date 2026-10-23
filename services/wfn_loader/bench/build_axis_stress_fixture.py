"""Write an actual, nonphysical WFN axis/I/O-layout fixture with encoded data.

This fixture tests the production HDF5/G-flat/FFT path. Its very small G sphere
and many nonorthogonal encoded bands intentionally do not model a material.
Every coefficient byte is physically written; no virtual/fill-only dataset.
"""
import argparse
import json
import os
import time
from pathlib import Path
import h5py
import numpy as np


def encoded_coeff(bands, k_rows, slots):
    """Return exact dyadic coefficient values, independently checkable by key."""
    b = np.asarray(bands, dtype=np.int64)[:, None]
    k = np.asarray(k_rows, dtype=np.int64)[None, :]
    g = np.asarray(slots, dtype=np.int64)[None, :]
    real = ((b+1)*8192+(k+1)*8+g+1).astype(np.float64)/2**29
    imag = ((b*17+k*131+g*7) % 2**20-2**19).astype(np.float64)/2**21
    return real+1j*imag


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=str(Path(__file__).resolve().parents[3] / 'tests/hsuite/fixture_na/WFN.h5'))
    p.add_argument('--out', required=True)
    p.add_argument('--nk', type=int, default=521)
    p.add_argument('--nb', type=int, default=20003)
    a = p.parse_args()
    if os.path.exists(a.out):
        raise FileExistsError(a.out)
    started = time.perf_counter()
    counts = np.resize(np.array([3,5,7], dtype=np.int32), a.nk)
    g_template = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],
                           [0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int32)
    g_all = np.concatenate([g_template[:n] for n in counts])
    k_all = np.repeat(np.arange(a.nk), counts)
    slot_all = np.concatenate([np.arange(n) for n in counts])
    tmp = a.out+'.building'
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with h5py.File(a.source, 'r') as source, h5py.File(tmp, 'w') as f:
        source.copy('mf_header', f)
        def replace(name, value):
            if name in f:
                del f[name]
            f.create_dataset(name, data=value)
        replace('mf_header/gspace/FFTgrid', np.array([8,8,8], dtype=np.int32))
        replace('mf_header/gspace/components', g_template)
        replace('mf_header/gspace/ng', np.int32(len(g_template)))
        root = 'mf_header/kpoints/'
        for key,value in dict(nrk=np.int32(a.nk), mnband=np.int32(a.nb),
                nspin=np.int32(1), nspinor=np.int32(1), ngk=counts,
                ngkmax=np.int32(max(counts)), kgrid=np.array([a.nk,1,1], dtype=np.int32),
                shift=np.zeros(3), w=np.full(a.nk, 1/a.nk),
                rk=np.column_stack((np.arange(a.nk)/a.nk, np.zeros((a.nk,2)))),
                ifmin=np.ones((1,a.nk), dtype=np.int32),
                ifmax=np.ones((1,a.nk), dtype=np.int32)).items():
            replace(root+key, value)
        energies = np.broadcast_to(np.arange(a.nb, dtype=np.float64)[None,None,:]/1024,
                                   (1,a.nk,a.nb))
        replace(root+'el', energies)
        del energies
        occ = np.zeros((1,a.nk,a.nb)); occ[:,:,0] = 1
        replace(root+'occ', occ); del occ
        replace('mf_header/symmetry/ntran', np.int32(1))
        replace('mf_header/symmetry/mtrx', np.eye(3,dtype=np.int32)[None,:,:])
        replace('mf_header/symmetry/tnp', np.zeros((1,3)))
        f.create_dataset('wfns/gvecs', data=g_all)
        coeffs = f.create_dataset('wfns/coeffs', shape=(a.nb,1,len(g_all),2), dtype=np.float64)
        for lo in range(0,a.nb,128):
            hi = min(lo+128,a.nb)
            z = encoded_coeff(np.arange(lo,hi),k_all,slot_all)
            # The occupied state is normalized and spatially constant.
            if lo == 0:
                z[0] = (slot_all == 0).astype(float)
            block = np.stack((z.real,z.imag),axis=-1)[:,None,:,:]
            coeffs[lo:hi] = block
        f.attrs['lorrax_fixture_kind'] = 'nonphysical_axis_and_IO_layout_stress'
        f.attrs['lorrax_coefficient_formula'] = 'dyadic formula in build_axis_stress_fixture.encoded_coeff; occupied band0=delta_g0'
        f.attrs['lorrax_expected_counts'] = json.dumps(dict(nk=a.nk, nb=a.nb, ngk=counts.tolist()))
        f.flush()
    os.replace(tmp,a.out)
    print(json.dumps(dict(status='PASS', path=a.out, nk=a.nk, nb=a.nb,
        ngtotal=len(g_all), coeff_bytes=a.nb*len(g_all)*16,
        file_bytes=os.path.getsize(a.out), seconds=time.perf_counter()-started)),flush=True)
