"""hBN/MoS2/hBN heterostructure (4x4 MoS2 between two 5x5 hBN) and its QE inputs.

MoS2 keeps its own lattice (a = 3.16429194 A, the MoS2_SOC_2026 tutorial value);
hBN is strained to 4 a_MoS2 / 5 = 2.53143 A (+1.1 % against 2.504 A), so the
MoS2 band edges carry no strain shift. Mo and B sit on the origin, both hBN
layers are in-plane identical and mirror images through the Mo plane: the
supercell keeps the C3 axis and sigma_h of the monolayer.

Usage: python build_inputs.py OUTDIR [--ecut 70] [--nbnd 1700] [--vacuum 15] [--kgrid 2 2 1]
(2x2x1 is the owner's memory-test grid; production needs a denser one.)
"""
import argparse
import os

import numpy as np

A_MOS2 = 3.16429194          # tutorial cell
Z_S = 0.1319282645 * 12.0    # tutorial S height (A), +-1.5831
D_SBN = 3.30                 # S plane to hBN plane (A), vdW gap
VALENCE = {"Mo": 14, "S": 6, "B": 3, "N": 5}   # SG15 ONCV (Mo with 4s4p semicore)
MASS = {"Mo": 95.94, "S": 32.065, "B": 10.811, "N": 14.007}
# FR ONCVPSP 3.3.0 set in the sandbox assets (assets/pseudopotentials/standard).
PSEUDO = {"Mo": "Mo.upf", "S": "S.upf", "B": "B.upf", "N": "N.upf"}


def structure(vacuum):
    A = 4 * A_MOS2
    z_bn = Z_S + D_SBN
    c = 2 * z_bn + vacuum
    cell = np.array([[A, 0, 0], [-A / 2, A * np.sqrt(3) / 2, 0], [0, 0, c]])
    atoms = []
    for i in range(4):
        for j in range(4):
            atoms.append(("Mo", (i / 4, j / 4, 0.0)))
            for s in (+1, -1):
                atoms.append(("S", ((i + 2 / 3) / 4, (j + 1 / 3) / 4, s * Z_S / c)))
    for s in (+1, -1):
        for i in range(5):
            for j in range(5):
                atoms.append(("B", (i / 5, j / 5, s * z_bn / c)))
                atoms.append(("N", ((i + 1 / 3) / 5, (j + 2 / 3) / 5, s * z_bn / c)))
    return cell, atoms


def pw_input(calc, cell, atoms, *, ecut, nbnd=None, kgrid=(3, 3, 1)):
    species = ["Mo", "S", "B", "N"]
    lines = ["&control", f"   prefix = 'hetero'", "   pseudo_dir = '../pseudos'",
             f"   calculation = '{calc}'", "   outdir = './'", "/", "&system",
             f"   ecutwfc = {ecut}", "   ibrav = 0", f"   nat = {len(atoms)}",
             f"   ntyp = {len(species)}", "   noncolin = .true.", "   lspinorb = .true."]
    if nbnd:
        lines.append(f"   nbnd = {nbnd}")
    lines += ["/", "&electrons", "   electron_maxstep = 200",
              f"   conv_thr = {1e-10 * len(atoms):.1e}",
              "   mixing_mode = 'local-TF'", "   mixing_beta = 0.4", "   mixing_ndim = 8",
              "   diagonalization = 'david'",
              f"   diago_david_ndim = {2 if nbnd else 4}",
              "   diago_full_acc = .true.", "/", "CELL_PARAMETERS angstrom"]
    lines += ["   " + " ".join(f"{x:.10f}" for x in v) for v in cell]
    lines.append("ATOMIC_SPECIES")
    lines += [f"   {s} {MASS[s]} '{PSEUDO[s]}'" for s in species]
    lines.append("ATOMIC_POSITIONS crystal")
    lines += [f"   {s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in atoms]
    lines += ["K_POINTS automatic", f"   {kgrid[0]} {kgrid[1]} {kgrid[2]} 0 0 0", ""]
    return "\n".join(lines)


def pw2bgw_input(nbnd, kgrid=(3, 3, 1)):
    return "\n".join([
        "&input_pw2bgw", "   prefix = 'hetero'", "   outdir = './'", "   real_or_complex = 2",
        "   wfng_flag = .true.", "   wfng_file = 'WFN'", "   rhog_flag = .true.",
        "   rhog_file = 'RHO'", "   vxc_flag = .true.", "   vxc_file = 'vxc.dat'",
        "   vxc_diag_nmin = 1", f"   vxc_diag_nmax = {nbnd}", "   wfng_kgrid = .true.",
        f"   wfng_nk1 = {kgrid[0]}", f"   wfng_nk2 = {kgrid[1]}", f"   wfng_nk3 = {kgrid[2]}",
        "   wfng_dk1 = 0.0", "   wfng_dk2 = 0.0", "   wfng_dk3 = 0.0", "/", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--ecut", type=float, default=70.0)
    ap.add_argument("--kgrid", type=int, nargs=3, default=[2, 2, 1])
    ap.add_argument("--nbnd", type=int, default=1700)
    ap.add_argument("--vacuum", type=float, default=15.0)
    a = ap.parse_args()
    cell, atoms = structure(a.vacuum)
    nel = sum(VALENCE[s] for s, _ in atoms)
    os.makedirs(f"{a.outdir}/1-scf", exist_ok=True)
    os.makedirs(f"{a.outdir}/2-wfn", exist_ok=True)
    open(f"{a.outdir}/1-scf/scf.in", "w").write(
        pw_input("scf", cell, atoms, ecut=a.ecut, kgrid=a.kgrid))
    open(f"{a.outdir}/2-wfn/wfn.in", "w").write(
        pw_input("nscf", cell, atoms, ecut=a.ecut, nbnd=a.nbnd, kgrid=a.kgrid))
    open(f"{a.outdir}/2-wfn/wfn.pp.in", "w").write(pw2bgw_input(a.nbnd, a.kgrid))
    vol = abs(np.linalg.det(cell))
    print(f"nat {len(atoms)}  electrons {nel}  occupied spinor bands {nel}  "
          f"a {cell[0, 0]:.5f} A  c {cell[2, 2]:.4f} A  V {vol:.1f} A^3 = {vol / 0.148185:.0f} bohr^3")


if __name__ == "__main__":
    main()
