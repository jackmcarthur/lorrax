"""
psp/qe_save_reader.py — Read crystal structure from a QE .save directory.

CrystalData duck-types as WFNReader for structure queries: bdot, bvec,
blat, alat, cell_volume, atom_crys, atom_types, nelec, nspinor, nspin,
ecutwfc, ecutrho, fft_grid, kgrid, nbands, ntran, sym_matrices, translations.

Does NOT provide wavefunction data (get_cnk, get_gvec_nk, kpoints), so
it cannot be used with SymMaps or the GW pipeline.  Designed for the
standalone DFT path: CrystalData → setup_H_k_from_kvec → Davidson.

Required files in the .save directory:
  data-file-schema.xml   — crystal structure, symmetry ops, electronic params
  charge-density.hdf5    — valence charge density ρ(G) (NLCC excluded)

Key methods:
  from_qe_save(save_dir) — parse XML, extract 48 symmetry ops with translations
  build_kgrid(nk, nosym, noinv, no_t_rev, force_symmorphic) — MP grid → IBZ
  load_charge_density()  — ρ_val(r) on FFT grid from HDF5
  validate_against_wfn(wfn) — cross-check all fields vs WFNReader

The IBZ reduction reproduces QE's kpoint_grid.f90: same enumeration order
(dir 3 fastest), forward-only equivalence, optional time reversal.
Translation convention: QE's affine shift is converted to BGW ``tnp`` as
``2π inv(mtrx) τ_qe``.  Rotation arrays match the raw WFN arrays exactly;
the transpose belongs only to the later reciprocal k/G action.

Usage
-----
    crystal = CrystalData.from_qe_save("silicon.save")
    kpts, weights = crystal.build_kgrid(nk=(6,6,6))
    V_scf = build_V_scf(V_loc, V_H, V_xc)
    H_k = setup_H_k_from_kvec(kpts[0], V_scf, vnl_setup, crystal, meta,
                                V_loc_r=V_loc, ngkmax=ngkmax)
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
    spinorbit: bool                 # QE lspinorb — NOT implied by nspinor==2
    ecutwfc: float                  # [Ry]
    ecutrho: float                  # [Ry]
    fft_grid: tuple[int, int, int]
    nbands: int
    nkpts: int
    kgrid: np.ndarray               # (3,) int — Monkhorst-Pack dimensions
    assume_isolated: str            # "none" | "2D" (from QE assume_isolated)

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
        bdot = bvec @ bvec.T * blat ** 2
        cell_volume = abs(np.dot(a1, np.cross(a2, a3)))

        # ── atoms ──
        nat = int(_all(root, "atomic_structure")[0].attrib["nat"])
        atoms = _all(root, "atom")[:nat]
        # Crystal coords: pos_bohr = avec_bohr.T @ τ_crys, so τ_crys = inv(avec_bohr.T) @ pos
        avec_inv_T = np.linalg.inv(avec_bohr.T)
        atom_crys = np.array([avec_inv_T @ _vec(a.text) for a in atoms])
        atom_types = np.array([_SYMBOL_TO_Z[a.attrib["name"]] for a in atoms], dtype=np.int32)

        # ── symmetry operations ──
        # Lazy service import keeps importing this XML/HDF5 reader itself
        # device-stack-free; the conversion is used only when parsing.
        from symmetry_maps import qe_xml_seitz_to_bgw
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
            # One owner for the QE-XML -> BGW Seitz boundary.
            frac_trans.append(qe_xml_seitz_to_bgw(
                np.round(R).astype(int), tau))
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
        # ── <spinorbit> ── QE's lspinorb, and the ONLY authoritative record
        # of it that survives the DFT run.  It is NOT implied by ``noncolin``:
        # ``noncolin=.true., lspinorb=.false.`` writes 2-component spinors
        # whose eigenvalues carry no spin-orbit at all, because QE ran
        # ``average_pp`` and collapsed the fully-relativistic pseudopotential's
        # j = ℓ±1/2 channels first.  A consumer that reads only ``nspinor``
        # cannot tell the two runs apart — and if it then builds j-resolved
        # projectors it silently produces an operator the wavefunctions never
        # saw.  ``psp.vnl_ops.resolve_soc_mode`` consumes this field.
        spinorbit = _text(root, "spinorbit") == "true"

        ecutwfc = 2.0 * float(_text(root, "ecutwfc"))
        ecutrho = 2.0 * float(_text(root, "ecutrho"))

        fg = _all(root, "fft_grid")[0].attrib
        fft_grid = (int(fg["nr1"]), int(fg["nr2"]), int(fg["nr3"]))

        nbnd = _text(root, "nbnd")
        nbands = int(nbnd) if nbnd else 0

        mp = _all(root, "monkhorst_pack")
        kgrid = (np.array([int(mp[0].attrib.get(f"nk{i}", 1)) for i in (1, 2, 3)],
                          dtype=np.int32) if mp else np.zeros(3, dtype=np.int32))

        # ── assume_isolated ──
        iso_text = _text(root, "assume_isolated")
        assume_isolated = iso_text.strip() if iso_text and iso_text.strip() else "none"
        if assume_isolated not in ("none", "2D"):
            raise ValueError(
                f"Unsupported assume_isolated='{assume_isolated}' in QE .save. "
                f"LORRAX supports: 'none', '2D'")

        return cls(
            alat=alat, blat=blat, avec=avec, bvec=bvec, bdot=bdot,
            cell_volume=cell_volume, nat=nat,
            atom_crys=atom_crys, atom_types=atom_types,
            ntran=ntran, sym_matrices=sym_matrices, translations=translations,
            sym_time_rev=sym_time_rev,
            nelec=nelec, nspin=nspin, nspinor=nspinor, spinorbit=spinorbit,
            ecutwfc=ecutwfc, ecutrho=ecutrho, fft_grid=fft_grid,
            nbands=nbands, nkpts=0, kgrid=kgrid,
            assume_isolated=assume_isolated, _save_dir=save_dir,
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
        if self.ntran == int(wfn.ntran):
            delta_tau = (
                self.translations[:self.ntran]
                - np.asarray(wfn.translations[:wfn.ntran])) / (2.0 * np.pi)
            delta_tau -= np.rint(delta_tau)
            tau_error = float(np.max(np.abs(delta_tau), initial=0.0))
            assert tau_error < atol, (
                "translations: max periodic fractional |delta| = "
                f"{tau_error:.2e} (tol {atol:.0e})")
        print("validate_against_wfn: all checks passed")


# ═══════════════════════════════════════════════════════════════════════
#  Monkhorst-Pack grid + IBZ reduction
# ═══════════════════════════════════════════════════════════════════════

def _reduce_mp_to_ibz(
    nk: np.ndarray,
    sym_matrices: np.ndarray,
    time_reversal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma-centred MP grid reduced to the IBZ, THROUGH THE SYMMETRY SERVICE.

    Returns ``(kpoints (n_ibz, 3) in [-0.5, 0.5), weights summing to 1)``,
    the pair QE's ``kpoint_grid.f90`` produces and this module's two
    consumers (``psp.run_nscf``, ``psp.kpm_dos``) want.

    WHAT THIS REPLACED.  A 70-line reimplementation of the orbit
    reduction: enumerate the grid, apply every symmetry (and ``-S`` for
    time reversal) to every k, snap to the grid with ``_EPS = 1e-5``,
    mark forward-only equivalence, accumulate weights.  That is exactly
    :func:`symmetry_maps.find_irreducible_bz_points`, which does the same
    job in INTEGER kgrid coordinates — so with no tolerance at all — and
    additionally returns the orbit map this one built and threw away.

    THE TRANSPOSE CONVENTION WAS MEASURED, NOT ASSUMED.  The old code
    applied ``S @ k`` on the raw QE/BGW rotation matrices, while
    ``SymMaps`` builds its k-space table as ``sym_matrices.transpose(0,2,1)``
    — different operations, and the docstrings alone do not settle which
    this grid wants.  Checked against the old implementation on the
    committed fixtures: on ``si_cohsex_debug`` (4x4x4, 48 ops, 8 IBZ
    points) the AS-STORED matrices reproduce it exactly and the
    transposed ones do not (k differ by 2.5e-01); on the 2-op
    ``gnppm_debug`` deck both agree, which is why the high-symmetry deck
    is the one that decides.  ``tests/test_unfold_through_the_service.py``
    carries that comparison so the convention cannot drift back.

    ``sym_matrices`` arrives already filtered by :meth:`build_kgrid` for
    ``nosym`` / ``no_t_rev`` / ``force_symmorphic``; the service function
    has no such flags and must not grow any — the filtering is a QE
    input-semantics question, not a symmetry-algebra one.
    """
    # Lazily imported: ``symmetry_maps`` pulls jax, and this module is
    # otherwise a plain QE-XML/HDF5 reader that a caller may import
    # without a device stack.
    from symmetry_maps import find_irreducible_bz_points

    kg = np.asarray(nk, dtype=int).reshape(3)
    S = np.asarray(sym_matrices, dtype=np.int32)
    # TRS enters as the augmented ``[S, -S]`` table the service documents,
    # which is what the old per-symmetry ``-S`` variant amounted to.
    sym_k = np.concatenate([S, -S]) if time_reversal else S

    # Full grid in integer kgrid coords, C-order (k3 fastest) — the same
    # enumeration QE uses and the same one the service infers its grid
    # from (it takes ``full.max(axis=0) + 1``, so every axis maximum must
    # appear, which a complete enumeration guarantees).
    full_int = np.array(
        [[i, j, k]
         for i in range(kg[0]) for j in range(kg[1]) for k in range(kg[2])],
        dtype=np.int32)

    irr_idx, _sym_idx, irr_int = find_irreducible_bz_points(
        full_int, sym_k, irr_kgrid_int=None)

    # Integer kgrid coords -> crystal fractions in [-0.5, 0.5), the range
    # QE returns and both consumers assume.
    kpoints = irr_int.astype(np.float64) / kg[None, :]
    kpoints -= np.rint(kpoints)

    # The weights ARE the orbit sizes — the quantity the old code
    # accumulated by hand and the service hands back for free.
    counts = np.bincount(irr_idx, minlength=irr_int.shape[0]).astype(np.float64)
    if int(counts.sum()) != full_int.shape[0]:
        raise RuntimeError(
            f"IBZ orbits cover {int(counts.sum())} of {full_int.shape[0]} "
            f"grid points — the reduction dropped or double-counted k.")
    return kpoints, counts / counts.sum()
