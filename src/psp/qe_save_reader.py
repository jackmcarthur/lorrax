"""
psp/qe_save_reader.py — Read crystal structure from a QE .save directory.

Constructs a lightweight object with the same attribute names as WFNReader,
so it can be passed to dft_operators, vnl_ops, charge_density, etc.

Required files in the .save directory:
  data-file-schema.xml   — crystal structure, symmetries, parameters
  charge-density.hdf5    — valence charge density ρ(G)

Usage
-----
    from psp.qe_save_reader import CrystalData

    crystal = CrystalData.from_qe_save("silicon.save")
    kpts, weights = crystal.build_kgrid()  # Gamma-centred IBZ
    H_k = setup_H_k_from_kvec(kpts[0], V_scf, vnl_setup, crystal, meta)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Periodic table (Z ≤ 86)
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
    for e in root.iter():
        if e.tag.split("}")[-1] == tag and e.text:
            return e.text.strip()
    return None

def _all(root: ET.Element, tag: str) -> list[ET.Element]:
    return [e for e in root.iter() if e.tag.split("}")[-1] == tag]

def _vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.split()])


# ---------------------------------------------------------------------------
# CrystalData
# ---------------------------------------------------------------------------

@dataclass
class CrystalData:
    """Crystal structure, symmetries, and parameters — duck-types as WFNReader."""

    # ── Lattice ──
    alat: float
    blat: float
    avec: np.ndarray                # (3,3) lattice vectors / alat
    bvec: np.ndarray                # (3,3) reciprocal vectors / blat
    bdot: np.ndarray                # (3,3) reciprocal metric [bohr⁻²]
    cell_volume: float

    # ── Atoms ──
    nat: int
    atom_crys: np.ndarray           # (nat, 3) crystal coordinates
    atom_types: np.ndarray          # (nat,) atomic numbers

    # ── Symmetry (duck-types WFNReader for SymMaps) ──
    ntran: int                      # number of symmetry operations
    sym_matrices: np.ndarray        # (ntran, 3, 3) int — rotation matrices (crystal coords)
    translations: np.ndarray        # (ntran, 3) float — fractional translations × 2π (BGW convention)
    sym_time_rev: np.ndarray        # (ntran,) bool — True if operation includes time reversal

    # ── Electronic / grid ──
    nelec: float
    nspin: int
    nspinor: int
    ecutwfc: float                  # [Ry]
    ecutrho: float                  # [Ry]
    fft_grid: tuple[int, int, int]
    nbands: int
    nkpts: int
    kgrid: np.ndarray               # (3,) int — Monkhorst-Pack dimensions

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

        # ── lattice ──
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

        # ── symmetry operations ──
        sym_elems = _all(root, "symmetry")
        rotations, frac_trans, has_time_rev = [], [], []
        for sym_elem in sym_elems:
            children = {c.tag.split("}")[-1]: c for c in sym_elem}
            if "rotation" not in children:
                continue
            R = _vec(children["rotation"].text).reshape(3, 3)
            rotations.append(np.round(R).astype(int))
            tau = (_vec(children["fractional_translation"].text)
                   if "fractional_translation" in children else np.zeros(3))
            frac_trans.append(tau * 2.0 * np.pi)  # BGW convention
            # QE marks operations that include time reversal
            info = children.get("info")
            tr = (info is not None and info.attrib.get("time_reversal") == "true")
            has_time_rev.append(tr)

        ntran = len(rotations)
        # Pad to 48 (WFNReader convention: always 48 slots)
        sym_matrices = np.zeros((48, 3, 3), dtype=np.int32)
        translations = np.zeros((48, 3), dtype=np.float64)
        sym_time_rev = np.zeros(48, dtype=bool)
        sym_matrices[:ntran] = np.array(rotations)
        translations[:ntran] = np.array(frac_trans)
        sym_time_rev[:ntran] = np.array(has_time_rev)

        # ── electronic ──
        nelec = float(_text(root, "nelec"))
        noncolin = _text(root, "noncolin") == "true"
        lsda = _text(root, "lsda") == "true"
        nspinor = 2 if noncolin else 1
        nspin = 2 if (lsda and not noncolin) else 1

        ecutwfc = 2.0 * float(_text(root, "ecutwfc"))
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
            ntran=ntran, sym_matrices=sym_matrices, translations=translations,
            sym_time_rev=sym_time_rev,
            nelec=nelec, nspin=nspin, nspinor=nspinor,
            ecutwfc=ecutwfc, ecutrho=ecutrho, fft_grid=fft_grid,
            nbands=nbands, nkpts=0, kgrid=kgrid, _save_dir=save_dir,
        )

    # ------------------------------------------------------------------
    def build_kgrid(
        self,
        nk: np.ndarray | tuple[int, int, int] = (4, 4, 4),
        *,
        nosym: bool = False,
        noinv: bool = False,
        no_t_rev: bool = False,
        force_symmorphic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a Gamma-centred Monkhorst-Pack grid, reduced to the IBZ.

        Matches QE's ``kpoint_grid.f90`` algorithm with the same symmetry
        flags as ``pw.x``:

        Parameters
        ----------
        nk : (3,) int — grid dimensions
        nosym : bool — disable all spatial symmetries.  The grid is still
            folded by time reversal (k ↔ −k) unless noinv is also True.
        noinv : bool — disable k ↔ −k (time reversal) in grid reduction.
        no_t_rev : bool — disable symmetry operations that include time
            reversal (rotation + TR).  For non-magnetic systems this has
            no effect since all operations are purely spatial.
        force_symmorphic : bool — drop operations with nonzero fractional
            translation (glide planes, screw axes).

        Returns
        -------
        kpoints : (n_ibz, 3) float64 — IBZ k-points in crystal coordinates
        weights : (n_ibz,) float64 — weights summing to 1
        """
        nk = np.asarray(nk, dtype=int)
        sym = self.sym_matrices[:self.ntran].copy()
        tau = self.translations[:self.ntran].copy()

        t_rev = self.sym_time_rev[:self.ntran].copy()

        if nosym:
            # Identity only
            sym = sym[:1]
        else:
            keep = np.ones(len(sym), dtype=bool)

            if force_symmorphic:
                # Drop operations with nonzero fractional translation
                for i in range(len(sym)):
                    if not np.allclose(tau[i], 0.0, atol=1e-8):
                        keep[i] = False

            if no_t_rev:
                # Drop operations that are rotation + time reversal
                for i in range(len(sym)):
                    if t_rev[i]:
                        keep[i] = False

            sym = sym[keep]

        time_reversal = not noinv
        return _reduce_mp_to_ibz(nk, sym, time_reversal=time_reversal)

    # ------------------------------------------------------------------
    def load_charge_density(self) -> tuple[np.ndarray, np.ndarray]:
        """Read ρ_val from ``charge-density.hdf5``."""
        cd_path = os.path.join(self._save_dir, "charge-density.hdf5")
        if not os.path.isfile(cd_path):
            raise FileNotFoundError(cd_path)

        nx, ny, nz = self.fft_grid
        N = nx * ny * nz

        with h5py.File(cd_path, "r") as f:
            miller = f["MillerIndices"][:]
            rho_ri = f["rhotot_g"][:]

        rho_G = np.zeros((nx, ny, nz), dtype=np.complex128)
        ix, iy, iz = miller[:, 0] % nx, miller[:, 1] % ny, miller[:, 2] % nz
        rho_G[ix, iy, iz] = rho_ri[0::2] + 1j * rho_ri[1::2]

        rho_r = np.real(np.fft.ifftn(rho_G)) * N
        return rho_r, rho_G

    # ------------------------------------------------------------------
    def validate_against_wfn(self, wfn, atol: float = 1e-6) -> None:
        """Assert all structural fields match a WFNReader."""
        def _chk(name, a, b, tol=atol):
            err = float(np.max(np.abs(np.asarray(a, dtype=float)
                                      - np.asarray(b, dtype=float))))
            assert err < tol, f"{name}: max|Δ| = {err:.2e} (tol {tol:.0e})"

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
        _chk("ntran", self.ntran, wfn.ntran, tol=0.5)
        _chk("sym_matrices", self.sym_matrices[:self.ntran],
             wfn.sym_matrices[:wfn.ntran], tol=0.5)
        print("validate_against_wfn: all checks passed")


# ═══════════════════════════════════════════════════════════════════════
#  Monkhorst-Pack grid + IBZ reduction
# ═══════════════════════════════════════════════════════════════════════

def _reduce_mp_to_ibz(
    nk: np.ndarray,
    sym_matrices: np.ndarray,
    time_reversal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma-centred MP grid reduced to IBZ (reproduces QE kpoint_grid.f90).

    Algorithm:
      1. Enumerate grid with direction 3 cycling fastest (QE loop order)
      2. For each k, apply all symmetries S (and optionally -S for TR)
      3. Forward-only equivalence: mark n_equiv > nk as equivalent
      4. Extract IBZ in discovery order with accumulated weights
    """
    nk1, nk2, nk3 = int(nk[0]), int(nk[1]), int(nk[2])
    nkr = nk1 * nk2 * nk3
    nsym = sym_matrices.shape[0]
    _EPS = 1e-5

    # Phase 1: full grid in crystal coords (QE loop order)
    xkg = np.empty((nkr, 3))
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                n = k + j * nk3 + i * nk2 * nk3
                xkg[n] = [i / nk1, j / nk2, k / nk3]

    # Phase 2: forward-only equivalence marking
    equiv = np.arange(nkr, dtype=int)
    wkk = np.ones(nkr)

    for nk_idx in range(nkr):
        if equiv[nk_idx] != nk_idx:
            continue
        for ns in range(nsym):
            xkr = sym_matrices[ns] @ xkg[nk_idx]
            xkr -= np.round(xkr)

            variants = [xkr.copy()]
            if time_reversal:
                neg = -xkr.copy()
                neg -= np.round(neg)
                variants.append(neg)

            for xk_test in variants:
                xx = xk_test[0] * nk1
                yy = xk_test[1] * nk2
                zz = xk_test[2] * nk3
                if (abs(xx - round(xx)) > _EPS or
                    abs(yy - round(yy)) > _EPS or
                    abs(zz - round(zz)) > _EPS):
                    continue
                i = round(xx) % nk1
                j = round(yy) % nk2
                k = round(zz) % nk3
                n_eq = k + j * nk3 + i * nk2 * nk3
                if n_eq > nk_idx and equiv[n_eq] == n_eq:
                    equiv[n_eq] = nk_idx
                    wkk[nk_idx] += 1.0

    # Phase 3: extract IBZ in discovery order
    ibz_k, ibz_w = [], []
    for nk_idx in range(nkr):
        if equiv[nk_idx] == nk_idx:
            kpt = xkg[nk_idx] - np.round(xkg[nk_idx])
            ibz_k.append(kpt)
            ibz_w.append(wkk[nk_idx])

    kpoints = np.array(ibz_k)
    weights = np.array(ibz_w)
    weights /= weights.sum()
    return kpoints, weights


