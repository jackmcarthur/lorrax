"""BerkeleyGW 4.0 ``bsemat.h5``: schema, header and matrix datasets.

One owner for the format facts ``bse.bsemat_export`` writes; see
docs/theory/bgw-bsemat-export.md#format. Sources: ``BSE/bsewrite.f90``
(``bse_hdf5_write``), ``Common/kernel_io.F90`` (``setup_kernel_hdf5``,
``xctinfo_to_kernel_header``), ``Common/kernel_io_hdf5_inc.f90``.

Layout (h5py order, the reverse of the Fortran declaration)::

    mats/{head,wing,body,exchange}  f8 (nk·ns, nk·ns, n2b, n2b, n1b, n1b, 2)
                                    [ikp, ik, icp, ic, ivp, iv, :] = (Re, Im)
    bse_header/versionnumber, flavor
    bse_header/params   screening icutv ecuts ecutg efermi theory nblocks storage nmat
    bse_header/bands    nvb ncb n1b n2b ns nspinor
    bse_header/kpoints  nk kpts (nk, 3) kgrid qflag exciton_Q_shift
    mf_header           the coarse WFN's header, verbatim

TDA: n1b = nvb (iv = 1 the HIGHEST valence band), n2b = ncb (ic = 1 the
lowest conduction band). ``extended_kernel``: n1b = n2b = nvb + ncb, valence
(highest first) then conduction, nblocks = 4. The matrices are dimensionless
(head) and bohr² (wing, body, exchange: v and W over 8π). Absorption multiplies
by −8π/(Ω N_k) and restores the head's screening from its own eps files.
"""
from __future__ import annotations

import h5py
import numpy as np

__all__ = ["MATRICES", "VERSION", "FLAVOR", "ICUTV_OF_SYS_DIM", "matrix_shape",
           "create_bsemat_file", "register_matrices", "read_bsemat_header",
           "read_wfn_k_list"]

#: TDA/BSE (theory = 0) order of ``bsewrite.f90:126-130``.
MATRICES = ("mats/head", "mats/wing", "mats/body", "mats/exchange")
#: ``VER_BSE_HDF5`` as the NERSC berkeleygw/4.0 build writes it.
VERSION = 2
FLAVOR = 2
#: ``TRUNC_NONE`` / ``TRUNC_SLAB`` by LORRAX ``sys_dim``.
ICUTV_OF_SYS_DIM = {3: 0, 2: 6}


def matrix_shape(nk, n1b, n2b, ns=1):
    """h5py shape of one matrix: ``(nk·ns, nk·ns, n2b, n2b, n1b, n1b, 2)``."""
    return (nk * ns, nk * ns, n2b, n2b, n1b, n1b, 2)


def read_wfn_k_list(wfn_file):
    """``(kpts (nk, 3) crystal, kgrid (3,), bdot (3, 3) bohr⁻², celvol bohr³)`` of a WFN."""
    with h5py.File(wfn_file, "r") as w:
        m = w["mf_header"]
        rk = np.asarray(m["kpoints/rk"][()], dtype=np.float64)
        if rk.shape[0] == 3 and rk.shape[1] != 3:
            rk = rk.T
        return (rk, np.asarray(m["kpoints/kgrid"][()], dtype=np.int64),
                np.asarray(m["crystal/bdot"][()], dtype=np.float64),
                float(m["crystal/celvol"][()]))


def create_bsemat_file(path, *, wfn_file, kpts, kgrid, nvb, ncb, extended, icutv,
                       ecuts, ecutg, efermi, nspinor, ns=1):
    """Write the header and empty contiguous matrix datasets (serial h5py, rank 0).

    ``mf_header`` is copied from the coarse WFN byte for byte, so absorption
    reads the k list and the G sphere it maps with. ``bse_header`` holds
    exactly the keys the NERSC berkeleygw/4.0 kernel writes (it writes no
    ``kpoints/flagbz``). Returns the matrix shape.
    """
    n1b = nvb + ncb if extended else nvb
    n2b = nvb + ncb if extended else ncb
    nk = int(np.asarray(kpts).shape[0])
    shape = matrix_shape(nk, n1b, n2b, ns)
    i4 = np.int32
    with h5py.File(wfn_file, "r") as w, h5py.File(path, "w") as f:
        w.copy("mf_header", f)
        f["bse_header/versionnumber"] = i4(VERSION)
        f["bse_header/flavor"] = i4(FLAVOR)
        p = f.create_group("bse_header/params")
        for k, v in (("screening", 0), ("icutv", icutv), ("theory", 0),
                     ("nblocks", 4 if extended else 1), ("storage", 0), ("nmat", 4)):
            p[k] = i4(v)
        for k, v in (("ecuts", ecuts), ("ecutg", ecutg), ("efermi", efermi)):
            p[k] = np.float64(v)
        b = f.create_group("bse_header/bands")
        for k, v in (("nvb", nvb), ("ncb", ncb), ("n1b", n1b), ("n2b", n2b),
                     ("ns", ns), ("nspinor", nspinor)):
            b[k] = i4(v)
        kp = f.create_group("bse_header/kpoints")
        kp["nk"] = i4(nk)
        kp["kpts"] = np.asarray(kpts, dtype=np.float64)
        kp["kgrid"] = np.asarray(kgrid, dtype=i4)
        kp["qflag"] = i4(1)
        kp["exciton_Q_shift"] = np.zeros(3)
        f.create_group("mats")
        for name in MATRICES:
            f.create_dataset(name, shape=shape, dtype=np.float64)
    return shape


def register_matrices(io, shape):
    """Register the skeleton's four matrices on a SlabIO write handle (same shape: reuse)."""
    for name in MATRICES:
        io.create_dataset(name, shape=shape, dtype=np.float64)


def read_bsemat_header(path):
    """Small header facts of a bsemat file as a dict (serial h5py, any rank)."""
    with h5py.File(path, "r") as f:
        out = {k: np.asarray(f["bse_header/" + k][()]) for k in (
            "versionnumber", "flavor", "params/screening", "params/icutv",
            "params/theory", "params/nblocks", "params/nmat", "bands/nvb",
            "bands/ncb", "bands/n1b", "bands/n2b", "bands/ns", "bands/nspinor",
            "kpoints/nk", "kpoints/kpts", "kpoints/kgrid", "kpoints/qflag")}
        out["shape"] = tuple(f[MATRICES[0]].shape)
    return out
