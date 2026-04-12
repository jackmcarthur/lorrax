"""
psp/qe_save_reader.py — Read crystal structure from a QE .save directory.

Constructs a lightweight object with the same attribute names as WFNReader,
so it can be passed to dft_operators, vnl_ops, charge_density, etc.

Required files in the .save directory:
  data-file-schema.xml   — crystal structure, electronic parameters
  charge-density.hdf5    — valence charge density ρ(G)

Usage
-----
    from psp.qe_save_reader import CrystalData

    crystal = CrystalData.from_qe_save("silicon.save")

    # Build Hamiltonian (duck-types as WFNReader for structure queries):
    H_k = setup_H_k_from_kvec(kvec, V_scf, vnl_setup, crystal, meta)

    # Load the SCF charge density:
    rho_r, rho_G = crystal.load_charge_density()

    # Cross-check against WFN.h5 (optional):
    crystal.validate_against_wfn(wfn)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Periodic table (Z ≤ 86, covers all common pseudopotentials)
# ---------------------------------------------------------------------------

_SYMBOL_TO_Z = {}
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W "
    "Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
for _z, _sym in enumerate(_ELEMENTS, start=1):
    _SYMBOL_TO_Z[_sym] = _z


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _text(root: ET.Element, tag: str) -> str | None:
    """First element text matching *tag* (namespace-agnostic)."""
    for e in root.iter():
        if e.tag.split("}")[-1] == tag and e.text:
            return e.text.strip()
    return None


def _all(root: ET.Element, tag: str) -> list[ET.Element]:
    """All elements matching *tag* (namespace-agnostic)."""
    return [e for e in root.iter() if e.tag.split("}")[-1] == tag]


def _vec(text: str) -> np.ndarray:
    """Parse a whitespace-separated vector of floats."""
    return np.array([float(x) for x in text.split()])


# ---------------------------------------------------------------------------
# CrystalData
# ---------------------------------------------------------------------------

@dataclass
class CrystalData:
    """Crystal structure and parameters — duck-types as WFNReader.

    Attributes match the names consumed by dft_operators, vnl_ops,
    charge_density, operator_checks, and Meta.from_system.
    """
    # ── Lattice ──
    alat: float                     # |a₁| [bohr]
    blat: float                     # 2π / alat [bohr⁻¹]
    avec: np.ndarray                # (3,3) lattice vectors / alat
    bvec: np.ndarray                # (3,3) reciprocal vectors / blat
    bdot: np.ndarray                # (3,3) reciprocal metric [bohr⁻²]
    cell_volume: float              # Ω [bohr³]

    # ── Atoms ──
    nat: int
    atom_crys: np.ndarray           # (nat, 3) crystal coordinates
    atom_types: np.ndarray          # (nat,) atomic numbers

    # ── Electronic / grid ──
    nelec: float
    nspin: int
    nspinor: int
    ecutwfc: float                  # [Ry]
    ecutrho: float                  # [Ry]
    fft_grid: tuple[int, int, int]
    nbands: int
    nkpts: int
    kgrid: np.ndarray               # (3,) int

    # ── Private ──
    _save_dir: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def from_qe_save(cls, save_dir: str) -> CrystalData:
        """Parse ``data-file-schema.xml`` from a QE ``.save`` directory."""
        save_dir = str(Path(save_dir).resolve())
        xml_path = os.path.join(save_dir, "data-file-schema.xml")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(xml_path)

        root = ET.parse(xml_path).getroot()

        # ── lattice (bohr) ──
        a1, a2, a3 = _vec(_text(root, "a1")), _vec(_text(root, "a2")), _vec(_text(root, "a3"))
        avec_bohr = np.array([a1, a2, a3])
        alat = float(np.linalg.norm(a1))
        blat = 2.0 * np.pi / alat
        avec = avec_bohr / alat
        bvec = np.array([_vec(_text(root, f"b{i}")) for i in (1, 2, 3)])
        bdot = bvec.T @ bvec * blat ** 2
        cell_volume = abs(np.dot(a1, np.cross(a2, a3)))

        # ── atoms ──
        nat = int(_all(root, "atomic_structure")[0].attrib["nat"])
        atoms = _all(root, "atom")[:nat]
        avec_inv = np.linalg.inv(avec_bohr)
        atom_crys = np.array([avec_inv @ _vec(a.text) for a in atoms])
        atom_types = np.array([_SYMBOL_TO_Z[a.attrib["name"]] for a in atoms], dtype=np.int32)

        # ── electronic ──
        nelec = float(_text(root, "nelec"))
        noncolin = _text(root, "noncolin") == "true"
        lsda = _text(root, "lsda") == "true"
        nspinor = 2 if noncolin else 1
        nspin = 2 if (lsda and not noncolin) else 1

        ecutwfc = 2.0 * float(_text(root, "ecutwfc"))   # Ha → Ry
        ecutrho = 2.0 * float(_text(root, "ecutrho"))

        fg = _all(root, "fft_grid")[0].attrib
        fft_grid = (int(fg["nr1"]), int(fg["nr2"]), int(fg["nr3"]))

        nbnd = _text(root, "nbnd")
        nbands = int(nbnd) if nbnd else 0

        mp = _all(root, "monkhorst_pack")
        kgrid = (np.array([int(mp[0].attrib.get(f"nk{i}", 1)) for i in (1, 2, 3)],
                          dtype=np.int32) if mp else np.zeros(3, dtype=np.int32))

        return cls(
            alat=alat, blat=blat, avec=avec, bvec=bvec, bdot=bdot,
            cell_volume=cell_volume, nat=nat,
            atom_crys=atom_crys, atom_types=atom_types,
            nelec=nelec, nspin=nspin, nspinor=nspinor,
            ecutwfc=ecutwfc, ecutrho=ecutrho, fft_grid=fft_grid,
            nbands=nbands, nkpts=0, kgrid=kgrid, _save_dir=save_dir,
        )

    # ------------------------------------------------------------------
    def load_charge_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Read ρ_val from ``charge-density.hdf5``.

        Returns ``(rho_r, rho_G)`` where rho_r is the real-space density
        on the FFT grid and rho_G is the G-space array (unnormalized FFT).
        """
        cd_path = os.path.join(self._save_dir, "charge-density.hdf5")
        if not os.path.isfile(cd_path):
            raise FileNotFoundError(cd_path)

        nx, ny, nz = self.fft_grid
        N = nx * ny * nz

        with h5py.File(cd_path, "r") as f:
            miller = f["MillerIndices"][:]    # (nG, 3) int
            rho_ri = f["rhotot_g"][:]         # (2*nG,) interleaved re/im

        # Scatter G-space coefficients onto FFT grid (vectorized)
        rho_G = np.zeros((nx, ny, nz), dtype=np.complex128)
        ix = miller[:, 0] % nx
        iy = miller[:, 1] % ny
        iz = miller[:, 2] % nz
        rho_G[ix, iy, iz] = rho_ri[0::2] + 1j * rho_ri[1::2]

        rho_r = np.real(np.fft.ifftn(rho_G)) * N
        return rho_r, rho_G

    # ------------------------------------------------------------------
    def validate_against_wfn(self, wfn, atol: float = 1e-6) -> None:
        """Assert all structural fields match a WFNReader."""
        def _chk(name, a, b, tol=atol):
            err = float(np.max(np.abs(np.asarray(a, dtype=float)
                                      - np.asarray(b, dtype=float))))
            assert err < tol, (
                f"{name}: max|Δ| = {err:.2e} (tol {tol:.0e})")

        _chk("alat", self.alat, wfn.alat)
        _chk("blat", self.blat, wfn.blat)
        _chk("cell_volume", self.cell_volume, wfn.cell_volume)
        _chk("avec", self.avec, wfn.avec)
        _chk("bvec", self.bvec, wfn.bvec)
        _chk("bdot", self.bdot, wfn.bdot)
        _chk("atom_crys", self.atom_crys, wfn.atom_crys)
        _chk("atom_types", self.atom_types, wfn.atom_types, tol=0.5)
        _chk("nelec", self.nelec, wfn.nelec, tol=0.5)
        _chk("nspinor", self.nspinor, wfn.nspinor, tol=0.5)
        _chk("fft_grid", self.fft_grid, wfn.fft_grid, tol=0.5)
        print("validate_against_wfn: all checks passed")
