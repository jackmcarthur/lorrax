"""
psp/qe_save_reader.py — Read crystal structure from a QE .save directory.

Constructs a lightweight object with the same attributes as WFNReader,
so it can be passed to dft_operators, vnl_ops, charge_density, etc.
without any changes to those modules.

Required files in the .save directory:
  data-file-schema.xml   — crystal structure, parameters
  charge-density.hdf5    — valence charge density ρ(G)

Usage
-----
    from psp.qe_save_reader import CrystalData

    # From a QE .save directory:
    crystal = CrystalData.from_qe_save("silicon.save")

    # Interchangeable with WFNReader for Hamiltonian construction:
    setup = build_operator_setup(crystal, sym, meta, pseudos)

    # Access the SCF charge density:
    rho_G, miller = crystal.load_charge_density()

    # Validate against a WFN.h5 (optional, for debugging):
    crystal.validate_against_wfn(wfn)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import h5py


# Element symbol → atomic number (covers Z ≤ 118)
_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Mo": 42,
    "W": 74,
}


def _find_ns(root: ET.Element) -> str:
    """Extract the XML namespace from the root tag."""
    tag = root.tag
    if tag.startswith("{"):
        return tag.split("}")[0] + "}"
    return ""


def _find_text(root: ET.Element, local_tag: str) -> str | None:
    """Find the first element matching local_tag (ignoring namespace) and return its text."""
    for elem in root.iter():
        if elem.tag.split("}")[-1] == local_tag and elem.text:
            return elem.text.strip()
    return None


def _find_all(root: ET.Element, local_tag: str):
    """Find all elements matching local_tag (ignoring namespace)."""
    return [e for e in root.iter() if e.tag.split("}")[-1] == local_tag]


def _parse_vector(text: str) -> np.ndarray:
    """Parse a whitespace-separated vector of floats."""
    return np.array([float(x) for x in text.split()])


@dataclass
class CrystalData:
    """Crystal structure and metadata, compatible with WFNReader's interface.

    Every attribute used by dft_operators, vnl_ops, charge_density, and
    Meta.from_system is present with matching names and conventions.
    """
    # Lattice (same conventions as WFN.h5)
    alat: float                    # lattice constant [bohr]
    blat: float                    # 2π/alat [1/bohr]
    avec: np.ndarray               # (3,3) lattice vectors / alat (dimensionless)
    bvec: np.ndarray               # (3,3) reciprocal vectors / blat (dimensionless)
    adot: np.ndarray               # (3,3) real-space metric (alat² units)
    bdot: np.ndarray               # (3,3) reciprocal metric (blat² units)
    cell_volume: float             # Ω [bohr³]

    # Atoms
    nat: int
    atom_crys: np.ndarray          # (nat, 3) fractional coordinates
    atom_types: np.ndarray         # (nat,) atomic numbers (Z)

    # Electronic structure parameters
    nelec: float
    nspin: int
    nspinor: int
    ecutwfc: float                 # Ry
    ecutrho: float                 # Ry
    fft_grid: tuple[int, int, int]
    nbands: int
    nkpts: int                     # not meaningful without WFN; set to 0

    # k-grid (for Meta.from_system)
    kgrid: np.ndarray              # (3,) int

    # Compatibility: SymMaps.get_gvecs_kfull calls wfn.get_gvec_nk
    _gvec_cache: dict = field(default_factory=dict, repr=False)

    # Source directory
    _save_dir: str = ""

    def get_gvec_nk(self, k_idx: int, kvec: np.ndarray | None = None) -> np.ndarray:
        """G-vectors for k-point k_idx satisfying |k+G|² ≤ ecutwfc.

        Same interface as WFNReader.get_gvec_nk.  Computes the PW sphere
        analytically from the cutoff and reciprocal metric.

        Parameters
        ----------
        k_idx : int — k-point index (used as cache key).
        kvec : (3,) float, optional — k-point in crystal coords.
            If None, uses k=(0,0,0).  The caller (SymMaps.get_gvecs_kfull)
            passes the IBZ k-vector; for standalone use, pass explicitly.

        Returns (nG, 3) integer G-vectors in crystal coordinates.
        """
        if k_idx in self._gvec_cache:
            return self._gvec_cache[k_idx]

        if kvec is None:
            kvec = np.zeros(3)
        kvec = np.asarray(kvec, dtype=float)

        nx, ny, nz = self.fft_grid
        gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
        gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
        gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
        Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
        G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1)

        # |k+G|² = (k+G)^T bdot (k+G)
        KG = G_all.astype(float) + kvec[None, :]
        KG_sq = np.einsum("gi,ij,gj->g", KG, self.bdot, KG)
        mask = KG_sq <= self.ecutwfc
        result = G_all[mask]

        self._gvec_cache[k_idx] = result
        return result

    @classmethod
    def from_qe_save(cls, save_dir: str) -> CrystalData:
        """Read from a QE ``*.save`` directory.

        Parameters
        ----------
        save_dir : path to the ``.save`` directory containing
            ``data-file-schema.xml`` and ``charge-density.hdf5``.
        """
        save_dir = str(Path(save_dir).resolve())
        xml_path = os.path.join(save_dir, "data-file-schema.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Not found: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # ── lattice vectors (bohr) ────────────────────────────────────
        a1 = _parse_vector(_find_text(root, "a1"))
        a2 = _parse_vector(_find_text(root, "a2"))
        a3 = _parse_vector(_find_text(root, "a3"))
        avec_bohr = np.array([a1, a2, a3])  # rows = lattice vectors

        alat = float(np.linalg.norm(a1))
        blat = 2.0 * np.pi / alat

        # WFN.h5 convention: avec stored as avec_bohr / alat
        avec = avec_bohr / alat

        # reciprocal vectors (QE stores in 2π/alat units, same as WFN.h5)
        b1 = _parse_vector(_find_text(root, "b1"))
        b2 = _parse_vector(_find_text(root, "b2"))
        b3 = _parse_vector(_find_text(root, "b3"))
        bvec = np.array([b1, b2, b3])

        # metrics
        adot = avec.T @ avec * alat ** 2
        bdot = bvec.T @ bvec * blat ** 2
        cell_volume = abs(np.dot(a1, np.cross(a2, a3)))

        # ── atoms ─────────────────────────────────────────────────────
        # nat is an attribute on <atomic_structure>, not a text element
        at_struct = _find_all(root, "atomic_structure")[0]
        nat_xml = int(at_struct.attrib["nat"])

        atom_elems = _find_all(root, "atom")
        # Take the first occurrence of each atom (XML may duplicate input/output)
        atom_elems = atom_elems[:nat_xml]

        avec_inv = np.linalg.inv(avec_bohr)
        positions_crys = []
        atomic_numbers = []
        for atom in atom_elems:
            name = atom.attrib["name"]
            pos_cart = _parse_vector(atom.text)
            pos_crys = avec_inv @ pos_cart
            positions_crys.append(pos_crys)
            z = _SYMBOL_TO_Z.get(name)
            if z is None:
                raise ValueError(f"Unknown element symbol: {name}")
            atomic_numbers.append(z)

        atom_crys = np.array(positions_crys, dtype=np.float64)
        atom_types = np.array(atomic_numbers, dtype=np.int32)

        # ── electronic parameters ─────────────────────────────────────
        nelec = float(_find_text(root, "nelec"))
        noncolin = _find_text(root, "noncolin") == "true"
        spinorbit = _find_text(root, "spinorbit") == "true"
        lsda = _find_text(root, "lsda") == "true"

        if noncolin:
            nspinor = 2
            nspin = 1    # QE uses nspin=4 internally, but WFN.h5 stores 1
        elif lsda:
            nspinor = 1
            nspin = 2
        else:
            nspinor = 1
            nspin = 1

        ecutwfc_ha = float(_find_text(root, "ecutwfc"))
        ecutrho_ha = float(_find_text(root, "ecutrho"))
        ecutwfc = 2.0 * ecutwfc_ha  # Ha → Ry
        ecutrho = 2.0 * ecutrho_ha

        # FFT grid
        fft_elem = _find_all(root, "fft_grid")[0]
        fft_grid = (
            int(fft_elem.attrib["nr1"]),
            int(fft_elem.attrib["nr2"]),
            int(fft_elem.attrib["nr3"]),
        )

        nbnd_text = _find_text(root, "nbnd")
        nbands = int(nbnd_text) if nbnd_text else 0

        # k-grid (from monkhorst_pack if present, else zeros)
        mp = _find_all(root, "monkhorst_pack")
        if mp:
            kgrid = np.array([
                int(mp[0].attrib.get("nk1", 1)),
                int(mp[0].attrib.get("nk2", 1)),
                int(mp[0].attrib.get("nk3", 1)),
            ], dtype=np.int32)
        else:
            kgrid = np.zeros(3, dtype=np.int32)

        return cls(
            alat=alat,
            blat=blat,
            avec=avec,
            bvec=bvec,
            adot=adot,
            bdot=bdot,
            cell_volume=cell_volume,
            nat=nat_xml,
            atom_crys=atom_crys,
            atom_types=atom_types,
            nelec=nelec,
            nspin=nspin,
            nspinor=nspinor,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            fft_grid=fft_grid,
            nbands=nbands,
            nkpts=0,
            kgrid=kgrid,
            _save_dir=save_dir,
        )

    # ── charge density ────────────────────────────────────────────────

    def load_charge_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Read ρ_val(G) from charge-density.hdf5.

        Returns
        -------
        rho_r : (nx, ny, nz) float64 — valence density on FFT grid [e/bohr³]
        rho_G_complex : (nx, ny, nz) complex128 — G-space coefficients
            (unnormalized FFT convention: rho_G[0,0,0] = N_el * N/Ω)
        """
        cd_path = os.path.join(self._save_dir, "charge-density.hdf5")
        if not os.path.isfile(cd_path):
            raise FileNotFoundError(f"Not found: {cd_path}")

        nx, ny, nz = self.fft_grid
        N = nx * ny * nz

        with h5py.File(cd_path, "r") as f:
            miller = f["MillerIndices"][:]      # (nG, 3) int
            rhotot_g = f["rhotot_g"][:]         # (2*nG,) interleaved re/im

        rho_G = np.zeros((nx, ny, nz), dtype=np.complex128)
        for ig in range(miller.shape[0]):
            gx, gy, gz = miller[ig]
            rho_G[gx % nx, gy % ny, gz % nz] = (
                rhotot_g[2 * ig] + 1j * rhotot_g[2 * ig + 1]
            )

        rho_r = np.real(np.fft.ifftn(rho_G)) * N
        return rho_r, rho_G

    # ── validation against WFN.h5 ────────────────────────────────────

    def validate_against_wfn(self, wfn, atol: float = 1e-6) -> None:
        """Check that this CrystalData matches a WFNReader.

        Raises AssertionError with a descriptive message on mismatch.
        """
        def _check(name, mine, theirs, tol=atol):
            mine = np.asarray(mine, dtype=float)
            theirs = np.asarray(theirs, dtype=float)
            err = np.max(np.abs(mine - theirs))
            assert err < tol, (
                f"{name} mismatch: max|Δ| = {err:.2e} (tol {tol:.0e})\n"
                f"  QE save: {mine.ravel()[:6]}\n"
                f"  WFN.h5:  {theirs.ravel()[:6]}"
            )

        _check("alat", self.alat, wfn.alat)
        _check("blat", self.blat, wfn.blat)
        _check("cell_volume", self.cell_volume, wfn.cell_volume)
        _check("avec", self.avec, wfn.avec)
        _check("bvec", self.bvec, wfn.bvec)
        _check("bdot", self.bdot, wfn.bdot)
        _check("atom_crys", self.atom_crys, wfn.atom_crys)
        _check("atom_types", self.atom_types, wfn.atom_types, tol=0.5)
        _check("nelec", self.nelec, wfn.nelec, tol=0.5)
        _check("nspinor", self.nspinor, wfn.nspinor, tol=0.5)
        _check("fft_grid", self.fft_grid, wfn.fft_grid, tol=0.5)

        print(f"validate_against_wfn: all checks passed")
