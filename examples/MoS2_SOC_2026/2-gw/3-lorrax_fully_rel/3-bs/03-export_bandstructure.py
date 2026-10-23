"""Join real DFT/QP htransform results into BerkeleyGW's eight-column layout.

The native seven-column tables use fractional k and energies relative to their
own path VBM. The exact energy_reference_eV header restores each absolute zero;
no energy is taken from the BerkeleyGW reference.
"""
from pathlib import Path
import h5py
import numpy as np


def read_native(path):
    offset = None
    for line in Path(path).read_text().splitlines():
        if line.startswith('# energy_reference_eV='):
            offset = float(line.split('=', 1)[1].split()[0])
    if offset is None or not np.isfinite(offset):
        raise ValueError(f'{path}: exact energy-reference receipt missing; use the documented LORRAX version')
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 7 or not np.isfinite(data).all():
        raise ValueError(f'{path}: expected finite seven-column native table')
    ids = data[:, :2].astype(np.int64)
    if not np.array_equal(ids, data[:, :2]) or np.any(ids < 0):
        raise ValueError(f'{path}: invalid zero-based k/band IDs')
    if len(set(map(tuple, ids))) != len(ids):
        raise ValueError(f'{path}: duplicate k/band rows')
    return data, offset


dft, dft_zero = read_native('bandstructure_dft.lorrax.dat')
qp, qp_zero = read_native('bandstructure_qp.lorrax.dat')
if dft.shape != qp.shape or not np.array_equal(dft[:, :6], qp[:, :6]):
    raise ValueError('DFT and QP k/band/path metadata differ; refusing to join them')
with h5py.File('WFN.h5', 'r') as wfn:
    crystal = wfn['mf_header/crystal']
    reciprocal_rows = np.asarray(crystal['bvec'], dtype=np.float64)
    blat = float(crystal['blat'][()])
k_cart = dft[:, 2:5] @ reciprocal_rows * blat
edf, eqp = dft[:, 6] + dft_zero, qp[:, 6] + qp_zero
output = np.column_stack((np.ones(len(dft)), dft[:, 1] + 1, k_cart, edf, eqp, eqp - edf))
# BerkeleyGW's inteqp table lists all k-points of each band consecutively.
order = np.lexsort((dft[:, 0], dft[:, 1]))
np.savetxt('bandstructure.dat', output[order],
           fmt=['%6d', '%7d', '%12.8f', '%12.8f', '%12.8f', '%15.9f', '%15.9f', '%15.9f'],
           header='spin    band          kx          ky          kz          E(MF)          E(QP)        Delta E\n'
                  '                       (Cartesian coordinates, bohr^-1)      (eV)           (eV)           (eV)')
