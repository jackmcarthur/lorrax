"""
psp/wfn_writer.py — Write WFN.h5 files compatible with BerkeleyGW / LORRAX.

Two modes:

1. Batch (legacy): ``write_wfn_h5(...)`` — collects everything, writes at once.

2. Streaming: open → write each k as it finishes → close.
   Suitable for parallel Davidson where each GPU writes its k-points.

   ::

       writer = WFNWriter("WFN.h5", crystal, kpoints, weights, kgrid, nbands,
                           gvecs_per_k, nosym=True)
       for ik in range(nk):
           evals, evecs = davidson_k(...)   # on whichever GPU
           writer.write_k(ik, evals, evecs)
       writer.close()
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
        nosym: bool = False,
        shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.path = path
        self.nk = kpoints.shape[0]
        self.nbands = nbands
        self.nspin = crystal.nspin
        self.nspinor = crystal.nspinor
        self.ngk = np.array([g.shape[0] for g in gvecs_per_k], dtype=np.int32)
        self.ngktot = int(self.ngk.sum())

        # Cumulative offsets for each k-point's slice in the coeffs array
        self._offsets = np.zeros(self.nk + 1, dtype=np.int64)
        np.cumsum(self.ngk, out=self._offsets[1:])

        # Eigenvalues / occupations — filled per k-point
        self._el = np.zeros((self.nspin, self.nk, nbands), dtype=np.float64)
        self._occ = np.zeros((self.nspin, self.nk, nbands), dtype=np.float64)
        # nspinor=2 (SOC/bispinor): spinor bands hold 1 e- each → nelec bands.
        # nspinor=1 (spin-unpolarized): 2 e- per band → nelec//2. (nspin=2 not
        # emitted by this writer — only channel [0] is filled.)
        n_occ = int(crystal.nelec) if self.nspinor == 2 else int(crystal.nelec) // 2
        self._occ[0, :, :n_occ] = 1.0

        # Open file, write header, pre-allocate coeffs
        self._f = h5py.File(path, "w")
        self._write_header(crystal, kpoints, weights, kgrid, gvecs_per_k,
                           nosym, shift)

    def _write_header(self, crystal, kpoints, weights, kgrid, gvecs_per_k,
                      nosym, shift):
        f = self._f
        nk, nbands = self.nk, self.nbands
        nspin, nspinor = self.nspin, self.nspinor
        ngk, ngktot = self.ngk, self.ngktot
        ngkmax = int(ngk.max())

        f.create_dataset("mf_header/versionnumber", data=1)
        f.create_dataset("mf_header/flavor", data=2)

        # kpoints (el and occ written at close, once all k-points done)
        kp = "mf_header/kpoints/"
        f.create_dataset(kp + "nspin", data=nspin)
        f.create_dataset(kp + "nspinor", data=nspinor)
        f.create_dataset(kp + "nrk", data=nk)
        f.create_dataset(kp + "mnband", data=nbands)
        f.create_dataset(kp + "ngkmax", data=ngkmax)
        f.create_dataset(kp + "ecutwfc", data=float(crystal.ecutwfc))
        f.create_dataset(kp + "kgrid", data=np.array(kgrid, dtype=np.int32))
        f.create_dataset(kp + "shift", data=np.array(shift, dtype=np.float64))
        f.create_dataset(kp + "ngk", data=ngk)
        f.create_dataset(kp + "w", data=weights.astype(np.float64))
        f.create_dataset(kp + "rk", data=kpoints.astype(np.float64))
        # Placeholder — overwritten at close()
        f.create_dataset(kp + "el", data=self._el)
        f.create_dataset(kp + "occ", data=self._occ)

        # nspinor=2 (SOC/bispinor): spinor bands hold 1 e- each → nelec bands.
        # nspinor=1 (spin-unpolarized): 2 e- per band → nelec//2. (nspin=2 not
        # emitted by this writer — only channel [0] is filled.)
        n_occ = int(crystal.nelec) if self.nspinor == 2 else int(crystal.nelec) // 2
        ifmin = np.ones((nspin, nk), dtype=np.int32)
        ifmax = np.full((nspin, nk), n_occ, dtype=np.int32)
        f.create_dataset(kp + "ifmin", data=ifmin)
        f.create_dataset(kp + "ifmax", data=ifmax)

        # gspace
        gspace_components = _build_gspace_components(crystal)
        gs = "mf_header/gspace/"
        f.create_dataset(gs + "ng", data=gspace_components.shape[0])
        f.create_dataset(gs + "ecutrho", data=float(crystal.ecutrho))
        f.create_dataset(gs + "FFTgrid", data=np.array(crystal.fft_grid, dtype=np.int32))
        f.create_dataset(gs + "components", data=gspace_components)

        # symmetry
        sy = "mf_header/symmetry/"
        if nosym:
            mtrx = np.zeros((48, 3, 3), dtype=np.int32)
            mtrx[0] = np.eye(3, dtype=np.int32)
            f.create_dataset(sy + "ntran", data=1)
            f.create_dataset(sy + "cell_symmetry", data=0)
            f.create_dataset(sy + "mtrx", data=mtrx)
            f.create_dataset(sy + "tnp", data=np.zeros((48, 3), dtype=np.float64))
        else:
            f.create_dataset(sy + "ntran", data=crystal.ntran)
            f.create_dataset(sy + "cell_symmetry", data=0)
            f.create_dataset(sy + "mtrx", data=crystal.sym_matrices)
            f.create_dataset(sy + "tnp", data=crystal.translations)

        # crystal
        apos = crystal.atom_crys @ crystal.avec
        adot = crystal.avec @ crystal.avec.T * crystal.alat ** 2
        recvol = (2.0 * np.pi) ** 3 / crystal.cell_volume
        cr = "mf_header/crystal/"
        f.create_dataset(cr + "celvol", data=float(crystal.cell_volume))
        f.create_dataset(cr + "recvol", data=float(recvol))
        f.create_dataset(cr + "alat", data=float(crystal.alat))
        f.create_dataset(cr + "blat", data=float(crystal.blat))
        f.create_dataset(cr + "nat", data=crystal.nat)
        f.create_dataset(cr + "avec", data=crystal.avec.astype(np.float64))
        f.create_dataset(cr + "bvec", data=crystal.bvec.astype(np.float64))
        f.create_dataset(cr + "adot", data=adot.astype(np.float64))
        f.create_dataset(cr + "bdot", data=crystal.bdot.astype(np.float64))
        f.create_dataset(cr + "atyp", data=crystal.atom_types.astype(np.int32))
        f.create_dataset(cr + "apos", data=apos.astype(np.float64))

        # wfns — gvecs written now, coeffs pre-allocated for streaming
        gvecs_cat = np.concatenate(gvecs_per_k, axis=0).astype(np.int32)
        f.create_dataset("wfns/gvecs", data=gvecs_cat)
        f.create_dataset("wfns/coeffs",
                         shape=(nbands, nspinor, ngktot, 2),
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
        self._f["mf_header/kpoints/occ"][...] = self._occ
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Batch writer (legacy convenience wrapper)
# ---------------------------------------------------------------------------

def write_wfn_h5(
    path: str,
    crystal,
    kpoints: np.ndarray,
    weights: np.ndarray,
    kgrid: tuple[int, int, int],
    eigenvalues: np.ndarray,
    gvecs_per_k: list[np.ndarray],
    coeffs_per_k: list[np.ndarray] | None = None,
    *,
    occupations: np.ndarray | None = None,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    nosym: bool = False,
) -> None:
    """Write a complete WFN.h5 file (batch mode).

    For streaming writes, use ``WFNWriter`` directly.
    """
    nbands = eigenvalues.shape[1]
    with WFNWriter(path, crystal, kpoints, weights, kgrid, nbands,
                   gvecs_per_k, nosym=nosym, shift=shift) as w:
        for ik in range(kpoints.shape[0]):
            c = coeffs_per_k[ik] if coeffs_per_k is not None else None
            w.write_k(ik, eigenvalues[ik], c)
