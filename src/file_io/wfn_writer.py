"""
file_io/wfn_writer.py — the BGW ``WFN.h5`` format: header tables and the NSCF writer.

:func:`wfn_header_tables` is the one definition of every ``mf_header/*``
dataset a LORRAX-written ``WFN.h5`` carries.  Two writers consume it:

* :class:`WFNWriter` — host-side, streaming (header first, one k at a time),
  for the NSCF producer ``psp.run_nscf``::

       writer = WFNWriter("WFN.h5", crystal, kpoints, weights, kgrid, nbands,
                          gvecs_per_k, nosym=True)
       for ik in range(nk):
           writer.write_k(ik, evals, evecs)
       writer.close()

* :func:`file_io.qp_wfn.write_qp_wfn_h5` — the collective QP ``WFN_qp.h5``
  writer: every rank writes its own G-slab through ``file_io.slab_io``.

Layout (BGW ``Common/wfn_io_hdf5.F90``): ``wfns/coeffs`` is
``(nbands, nspinor, ngktot, 2)`` float64 (re, im), the k-points
concatenated along the G axis at offsets ``cumsum(ngk)``; ``wfns/gvecs`` is
``(ngktot, 3)`` int32 on the same axis.
"""
from __future__ import annotations

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# G-space generation (QE convention)
# ---------------------------------------------------------------------------

def _build_gspace_components(crystal):
    """Charge-density G-vectors in QE convention.

    Range [-N//2+1, N//2] per axis, filtered by |G|² ≤ ecutrho,
    sorted by (round(|G|²×1e8), g1, g2, g3). Matches QE ggen.f90.
    """
    nx, ny, nz = int(crystal.fft_grid[0]), int(crystal.fft_grid[1]), int(crystal.fft_grid[2])
    gx = np.arange(-nx // 2 + 1, nx // 2 + 1)
    gy = np.arange(-ny // 2 + 1, ny // 2 + 1)
    gz = np.arange(-nz // 2 + 1, nz // 2 + 1)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.int32)
    bdot = np.asarray(crystal.bdot, dtype=float)
    G2 = np.einsum("gi,ij,gj->g", G_all.astype(float), bdot, G_all.astype(float))
    mask = G2 <= float(crystal.ecutrho)
    G_f, G2_f = G_all[mask], G2[mask]
    G2_int = np.round(G2_f * 1e8).astype(np.int64)
    order = np.lexsort((G_f[:, 2], G_f[:, 1], G_f[:, 0], G2_int))
    return G_f[order]


# ---------------------------------------------------------------------------
# Header tables — the one definition both writers publish
# ---------------------------------------------------------------------------

def wfn_header_tables(
    crystal,
    *,
    kpoints: np.ndarray,
    weights: np.ndarray,
    kgrid: tuple[int, int, int],
    nbands: int,
    ngk: np.ndarray,
    occupations: np.ndarray | None = None,
    nosym: bool = False,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    """Every ``mf_header/*`` dataset of a BGW ``WFN.h5``, by path, in creation order.

    Values are exactly what ``h5py.create_dataset(name, data=value)`` is
    handed, so a host writer and the SlabIO deferred-metadata writer land the
    same dtype and shape.  ``mf_header/kpoints/el`` is zero; the caller fills
    ``el[0]`` (only spin channel 0 is emitted).

    ``occupations``: optional ``(nk, nbands)`` or ``(nspin, nk, nbands)``
    table, stored verbatim.  Absent, the index-step table from the nominal
    electron count.  ``n_occ`` is an occupied-BAND count, a band holding
    ``2/(nspin·nspinor)`` electrons: ``num_electrons`` (WfnLoader) is the
    physical electron count, and a QE ``CrystalData`` has no such attr and its
    ``nelec`` IS that count.  A WfnLoader's ``nelec`` is already a band count
    (``max(ifmax)``), so halving it again at nspinor=1 would double-halve.
    """
    nk = int(kpoints.shape[0])
    nspin, nspinor = crystal.nspin, crystal.nspinor
    ngk = np.asarray(ngk, dtype=np.int32)
    el = np.zeros((nspin, nk, nbands), dtype=np.float64)
    occ = np.zeros((nspin, nk, nbands), dtype=np.float64)
    n_occ = int(round(float(
        getattr(crystal, "num_electrons", crystal.nelec))
        * nspin * nspinor / 2.0))
    if occupations is None:
        occ[0, :, :n_occ] = 1.0
    else:
        given = np.asarray(occupations, dtype=np.float64)
        if given.shape == (nk, nbands) and nspin == 1:
            given = given[None, ...]
        expected = (nspin, nk, nbands)
        if given.shape != expected:
            raise ValueError(
                f"WFN header occupations have shape {given.shape}, "
                f"expected {expected}.")
        if not np.all(np.isfinite(given)):
            raise ValueError("WFN header occupations must be finite.")
        occ[...] = given

    t: dict[str, object] = {}
    t["mf_header/versionnumber"] = 1
    t["mf_header/flavor"] = 2

    kp = "mf_header/kpoints/"
    t[kp + "nspin"] = nspin
    t[kp + "nspinor"] = nspinor
    t[kp + "nrk"] = nk
    t[kp + "mnband"] = nbands
    t[kp + "ngkmax"] = int(ngk.max())
    t[kp + "ecutwfc"] = float(crystal.ecutwfc)
    t[kp + "kgrid"] = np.array(kgrid, dtype=np.int32)
    t[kp + "shift"] = np.array(shift, dtype=np.float64)
    t[kp + "ngk"] = ngk
    t[kp + "w"] = weights.astype(np.float64)
    t[kp + "rk"] = kpoints.astype(np.float64)
    t[kp + "el"] = el
    t[kp + "occ"] = occ
    t[kp + "ifmin"] = np.ones((nspin, nk), dtype=np.int32)
    t[kp + "ifmax"] = np.full((nspin, nk), n_occ, dtype=np.int32)

    gspace_components = _build_gspace_components(crystal)
    gs = "mf_header/gspace/"
    t[gs + "ng"] = gspace_components.shape[0]
    t[gs + "ecutrho"] = float(crystal.ecutrho)
    t[gs + "FFTgrid"] = np.array(crystal.fft_grid, dtype=np.int32)
    t[gs + "components"] = gspace_components

    sy = "mf_header/symmetry/"
    if nosym:
        mtrx = np.zeros((48, 3, 3), dtype=np.int32)
        mtrx[0] = np.eye(3, dtype=np.int32)
        t[sy + "ntran"] = 1
        t[sy + "cell_symmetry"] = 0
        t[sy + "mtrx"] = mtrx
        t[sy + "tnp"] = np.zeros((48, 3), dtype=np.float64)
    else:
        t[sy + "ntran"] = crystal.ntran
        t[sy + "cell_symmetry"] = 0
        t[sy + "mtrx"] = crystal.sym_matrices
        t[sy + "tnp"] = crystal.translations

    apos = crystal.atom_crys @ crystal.avec
    adot = crystal.avec @ crystal.avec.T * crystal.alat ** 2
    recvol = (2.0 * np.pi) ** 3 / crystal.cell_volume
    cr = "mf_header/crystal/"
    t[cr + "celvol"] = float(crystal.cell_volume)
    t[cr + "recvol"] = float(recvol)
    t[cr + "alat"] = float(crystal.alat)
    t[cr + "blat"] = float(crystal.blat)
    t[cr + "nat"] = crystal.nat
    t[cr + "avec"] = crystal.avec.astype(np.float64)
    t[cr + "bvec"] = crystal.bvec.astype(np.float64)
    t[cr + "adot"] = adot.astype(np.float64)
    t[cr + "bdot"] = crystal.bdot.astype(np.float64)
    t[cr + "atyp"] = crystal.atom_types.astype(np.int32)
    t[cr + "apos"] = apos.astype(np.float64)
    return t


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class WFNWriter:
    """Streaming WFN.h5 writer: header first, coefficients per k-point.

    Parameters
    ----------
    path : output file path
    crystal : CrystalData
    kpoints : (nk, 3) crystal coordinates
    weights : (nk,)
    kgrid : (nkx, nky, nkz)
    nbands : number of bands
    gvecs_per_k : list of (ngk_i, 3) int arrays
    occupations : optional ``(nk, nbands)`` or ``(nspin, nk, nbands)`` table
        Stored verbatim.  When absent, retain the historical index-step
        occupations derived from the nominal electron count.
    nosym : write identity-only symmetries (QE nosym convention)
    shift : MP grid shift
    """

    def __init__(
        self,
        path: str,
        crystal,
        kpoints: np.ndarray,
        weights: np.ndarray,
        kgrid: tuple[int, int, int],
        nbands: int,
        gvecs_per_k: list[np.ndarray],
        *,
        occupations: np.ndarray | None = None,
        nosym: bool = False,
        shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.path = path
        self.nk = kpoints.shape[0]
        self.nbands = nbands
        self.ngk = np.array([g.shape[0] for g in gvecs_per_k], dtype=np.int32)
        self.ngktot = int(self.ngk.sum())

        # Cumulative offsets for each k-point's slice in the coeffs array
        self._offsets = np.zeros(self.nk + 1, dtype=np.int64)
        np.cumsum(self.ngk, out=self._offsets[1:])

        tables = wfn_header_tables(
            crystal, kpoints=kpoints, weights=weights, kgrid=kgrid,
            nbands=nbands, ngk=self.ngk, occupations=occupations,
            nosym=nosym, shift=shift)
        # Eigenvalues are filled per k and landed at close().
        self._el = tables["mf_header/kpoints/el"]

        self._f = h5py.File(path, "w")
        for name, value in tables.items():
            self._f.create_dataset(name, data=value)
        gvecs_cat = np.concatenate(gvecs_per_k, axis=0).astype(np.int32)
        self._f.create_dataset("wfns/gvecs", data=gvecs_cat)
        self._f.create_dataset("wfns/coeffs",
                               shape=(nbands, crystal.nspinor, self.ngktot, 2),
                               dtype=np.float64,
                               fillvalue=0.0)

    def write_k(self, ik: int, eigenvalues: np.ndarray,
                coeffs: np.ndarray | None = None):
        """Write one k-point's eigenvalues and (optionally) coefficients.

        Parameters
        ----------
        ik : k-point index
        eigenvalues : (nbands,) float64 — in Ry
        coeffs : (nbands, nspinor, ngk_ik) complex128, or None
        """
        self._el[0, ik] = eigenvalues

        if coeffs is not None:
            off = int(self._offsets[ik])
            ng_k = int(self.ngk[ik])
            ds = self._f["wfns/coeffs"]
            ds[:, :, off:off + ng_k, 0] = coeffs.real
            ds[:, :, off:off + ng_k, 1] = coeffs.imag

    def close(self):
        """Finalize: write eigenvalues/occupations, close file."""
        self._f["mf_header/kpoints/el"][...] = self._el
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
