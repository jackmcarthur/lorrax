"""``mf_header`` — verbatim BGW-style header that travels with every
mean-field-derived HDF5 file in LORRAX.

`mf_header` is the same group tree that BerkeleyGW writes into
``WFN.h5`` (``mf_header/{versionnumber, flavor, kpoints, gspace,
symmetry, crystal}``).  This module owns the canonical read path and a
verbatim copy helper so any downstream artifact (``zeta_q.h5``,
``kin_ion.h5``, ``V_qmunu.h5``, ``sigma_omega.h5``) can carry the same
crystal / k-grid / G-grid / symmetry description as the WFN it was
built from.

``WFNReader`` reuses :func:`read_mf_header` so the on-disk schema lives
in exactly one place.

This module deliberately does NOT expose any derived quantities
(``vbm``, ``efermi``, ``kpt_starts``, …) — those are computed by the
consumer that needs them.  :class:`MfHeader` is a pure 1:1 mirror of
the on-disk group.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import h5py as h5
import numpy as np


class MfHeader(NamedTuple):
    """1:1 mirror of the BGW ``mf_header`` group.

    Attribute names match the field names ``WFNReader`` already
    exposes (``nkpts`` ↔ ``mf_header/kpoints/nrk``, ``nbands`` ↔
    ``mnband``, etc.); the original BGW dataset name is given in
    parentheses below for cross-reference.
    """

    # /mf_header/{versionnumber, flavor}
    version: int            # versionnumber
    flavor: int

    # /mf_header/kpoints/...
    nspin: int              # nspin
    nspinor: int            # nspinor
    nkpts: int              # nrk
    nbands: int             # mnband
    ngkmax: int
    ecutwfc: float
    kgrid: np.ndarray       # (3,) int
    shift: np.ndarray       # (3,) float
    ngk: np.ndarray         # (nkpts,) int
    ifmin: np.ndarray       # (nspin, nkpts) int
    ifmax: np.ndarray       # (nspin, nkpts) int
    kweights: np.ndarray    # w, (nkpts,) float
    kpoints: np.ndarray     # rk, (nkpts, 3) float — fractional
    energies: np.ndarray    # el, (nspin, nkpts, nbands) float
    occs: np.ndarray        # occ, (nspin, nkpts, nbands) float

    # /mf_header/gspace/...
    ng: int
    ecutrho: float
    fft_grid: np.ndarray    # FFTgrid, (3,) int32

    # /mf_header/symmetry/...
    ntran: int
    cell_symmetry: int
    sym_matrices: np.ndarray  # mtrx, (48, 3, 3) int
    translations: np.ndarray  # tnp, (48, 3) float — BGW convention (2π·τ_frac)

    # /mf_header/crystal/...
    cell_volume: float       # celvol
    recip_volume: float      # recvol
    alat: float
    blat: float
    nat: int
    avec: np.ndarray         # (3, 3) — lattice vectors
    bvec: np.ndarray         # (3, 3) — reciprocal lattice vectors
    adot: np.ndarray         # (3, 3) — real-space metric
    bdot: np.ndarray         # (3, 3) — reciprocal-space metric
    atom_types: np.ndarray   # atyp, (nat,) int
    atom_positions: np.ndarray  # apos, (nat, 3) — cartesian, units of alat


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_group(f: h5.File) -> MfHeader:
    """Populate :class:`MfHeader` from an already-open ``h5py.File``."""
    g = f['mf_header']
    return MfHeader(
        version=g['versionnumber'][()],
        flavor=g['flavor'][()],
        # kpoints
        nspin=g['kpoints/nspin'][()],
        nspinor=g['kpoints/nspinor'][()],
        nkpts=g['kpoints/nrk'][()],
        nbands=g['kpoints/mnband'][()],
        ngkmax=g['kpoints/ngkmax'][()],
        ecutwfc=g['kpoints/ecutwfc'][()],
        kgrid=g['kpoints/kgrid'][:],
        shift=g['kpoints/shift'][:],
        ngk=g['kpoints/ngk'][:],
        ifmin=g['kpoints/ifmin'][:],
        ifmax=g['kpoints/ifmax'][:],
        kweights=g['kpoints/w'][:],
        kpoints=g['kpoints/rk'][:],
        energies=g['kpoints/el'][:],
        occs=g['kpoints/occ'][:],
        # gspace
        ng=g['gspace/ng'][()],
        ecutrho=g['gspace/ecutrho'][()],
        fft_grid=np.asarray(g['gspace/FFTgrid'][:], dtype=np.int32),
        # symmetry
        ntran=g['symmetry/ntran'][()],
        cell_symmetry=g['symmetry/cell_symmetry'][()],
        sym_matrices=g['symmetry/mtrx'][:],
        translations=g['symmetry/tnp'][:],
        # crystal
        cell_volume=g['crystal/celvol'][()],
        recip_volume=g['crystal/recvol'][()],
        alat=g['crystal/alat'][()],
        blat=g['crystal/blat'][()],
        nat=g['crystal/nat'][()],
        avec=g['crystal/avec'][:],
        bvec=g['crystal/bvec'][:],
        adot=g['crystal/adot'][:],
        bdot=g['crystal/bdot'][:],
        atom_types=g['crystal/atyp'][:],
        atom_positions=g['crystal/apos'][:],
    )


def read_mf_header(path: str | Path) -> MfHeader:
    """Open ``path`` and return its ``mf_header`` group as :class:`MfHeader`.

    The file is closed before this function returns.  Use
    :func:`read_mf_header_from_file` if you already hold an open
    ``h5py.File`` handle (e.g. inside ``WFNReader.__init__``).
    """
    with h5.File(str(path), 'r') as f:
        return _read_group(f)


def read_mf_header_from_file(f: h5.File) -> MfHeader:
    """Same as :func:`read_mf_header` but operates on an open handle."""
    return _read_group(f)


# ---------------------------------------------------------------------------
# Copy (verbatim)
# ---------------------------------------------------------------------------

def copy_mf_header(src_path: str | Path, dst_path: str | Path,
                   *, dst_mode: str = 'a') -> None:
    """Copy ``src_path``'s ``mf_header`` group verbatim into ``dst_path``.

    Uses ``h5py.File.copy`` which preserves dataset dtypes, chunking,
    fill values, and any attributes attached to ``mf_header`` or its
    children.  If ``dst_path`` already has an ``mf_header`` group the
    copy is refused (``ValueError``) — header overwrite is destructive
    and should be done explicitly.

    Parameters
    ----------
    src_path
        Path to a file containing an ``mf_header`` group (typically
        ``WFN.h5``).
    dst_path
        Path to the destination file.  Opened with ``dst_mode``.
    dst_mode
        Mode for opening the destination — ``'a'`` (append) by
        default so this can be called against an existing artifact.
    """
    src_path = str(src_path)
    dst_path = str(dst_path)
    with h5.File(src_path, 'r') as fs, h5.File(dst_path, dst_mode) as fd:
        if 'mf_header' in fd:
            raise ValueError(
                f"copy_mf_header: {dst_path} already has an 'mf_header' "
                f"group; refusing to overwrite.")
        fs.copy('mf_header', fd, name='mf_header')


__all__ = [
    'MfHeader',
    'read_mf_header',
    'read_mf_header_from_file',
    'copy_mf_header',
]
