"""Build ε₂(ω) from a BGW-format ``eigenvalues.dat`` at arbitrary broadening.

Both BGW (``absorption.x``) and LORRAX (``absorption_eigvecs``) write
the same per-state oscillator-strength file format:

    # neig  =  <N>
    # vol   =  <V_supercell in bohr^3>     (= V_cell × n_k)
    # nspin, nspinor = <ns>  <nspinor>
    # eig (eV)   abs(dipole)^2   Re(dipole)   Im(dipole)
    <E_1>  <|d_1|²>  <Re d_1>  <Im d_1>
    ...

This script Lorentzian-broadens any such file:

    ε₂(ω) = (16π² / V_super · n_spin · n_spinor)
            · Σ_S^{n_max} |d_S|² · L(ω - E_S, η)

so two files (e.g. BGW vs LORRAX) can be put on the same axes at any
chosen η and any truncation. Useful for matching n_eig + η when
comparing finite-Krylov LORRAX runs to BGW full-diag (``ε₂``
peak height converges with n_eig in both cases — comparison must use
the same truncation to be fair).

Used by ``BGW_COMPARE.md`` as the canonical "reproduce BGW absorption
at custom η" tool. CLI:

    python -m bse.eigvals_to_eps2 \\
        --files <bgw>/eigenvalues.dat <lorrax>/eigenvalues_b1.dat \\
        --eta-eV 0.05 --n-max 100 --out cmp.png

Or as a library:

    omegas, curves = compute_eps2_from_files([...], eta_eV=0.05, n_max=100)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np


def read_bgw_eigvals(path):
    """Return (E_eV, |d|², V_super_bohr3, n_spin, n_spinor, n_eig_in_file)."""
    text = Path(path).read_text()
    n_eig = int(re.search(r"neig\s*=\s*(\d+)", text).group(1))
    V_super = float(re.search(r"vol\s*=\s*([\d.E+\-]+)", text).group(1))
    m = re.search(r"nspin,\s*nspinor\s*=\s*(\d+)\s+(\d+)", text)
    n_spin, n_spinor = int(m.group(1)), int(m.group(2))
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    E_eV = data[:, 0]
    f_S = data[:, 1]
    return E_eV, f_S, V_super, n_spin, n_spinor, n_eig


def lorentzian(omegas, energies, weights, eta):
    delta = omegas[:, None] - energies[None, :]
    L = (eta / np.pi) / (delta ** 2 + eta ** 2)
    return L @ weights


def gaussian(omegas, energies, weights, eta):
    """Gaussian-broadened sum-of-deltas. ``eta`` is the standard deviation
    σ in the same units as ``omegas`` and ``energies``. Normalised so
    ∫G(ω) dω = 1 (peak height = 1/(σ√(2π))).
    """
    delta = omegas[:, None] - energies[None, :]
    G = np.exp(-0.5 * (delta / eta) ** 2) / (eta * np.sqrt(2 * np.pi))
    return G @ weights


def compute_eps2(E_eV, f_S, V_super, n_spin, n_spinor,
                 *, omegas_eV, eta_eV, n_max=None, kernel="lorentzian"):
    """Mirrors ``bse.absorption_eigvecs.compute_eps2``: prefactor 16π²/V
    with V the supercell volume in bohr³, broadening done in **Rydberg**
    units (the ε₂ writer + BGW's absorption.x agree on Ry-space broadening).

    Parameters
    ----------
    kernel : "lorentzian" or "gaussian". Default "lorentzian" (BGW
        default). With "gaussian", ``eta_eV`` is the Gaussian std-dev σ.
    """
    ryd2ev = 13.6056980659
    if n_max is not None:
        E_eV = E_eV[:n_max]
        f_S = f_S[:n_max]
    omegas_Ry = omegas_eV / ryd2ev
    E_Ry = E_eV / ryd2ev
    eta_Ry = eta_eV / ryd2ev
    pref = 16.0 * np.pi ** 2 / (V_super * n_spin * n_spinor)
    broaden = gaussian if kernel == "gaussian" else lorentzian
    return pref * broaden(omegas_Ry, E_Ry, f_S, eta_Ry)


def compute_eps2_from_files(paths, *, eta_eV, n_max=None,
                            omega_min_eV=0.0, omega_max_eV=8.0, n_omega=4001,
                            kernel="lorentzian"):
    omegas = np.linspace(omega_min_eV, omega_max_eV, n_omega)
    out = {}
    for p in paths:
        E, f, V, ns, nspinor, n_in = read_bgw_eigvals(p)
        n_used = min(n_max, n_in) if n_max else n_in
        eps2 = compute_eps2(E, f, V, ns, nspinor,
                            omegas_eV=omegas, eta_eV=eta_eV, n_max=n_max,
                            kernel=kernel)
        out[str(p)] = dict(eps2=eps2, n_used=n_used, V=V,
                           E=E[:n_used], f=f[:n_used])
    return omegas, out


def _main():
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n\n")[0])
    p.add_argument("--files", nargs="+", required=True,
                   help="One or more BGW-format eigenvalues.dat paths")
    p.add_argument("--eta-eV", type=float, default=0.05,
                   help="Broadening width (Lorentzian half-width η or "
                        "Gaussian std σ). Default 0.05.")
    p.add_argument("--kernel", choices=("lorentzian", "gaussian"),
                   default="lorentzian",
                   help="Broadening kernel (BGW default = lorentzian)")
    p.add_argument("--n-max", type=int, default=None,
                   help="Truncate each spectrum to first n_max states "
                        "(default: file's full neig)")
    p.add_argument("--omega-min-eV", type=float, default=0.0)
    p.add_argument("--omega-max-eV", type=float, default=8.0)
    p.add_argument("--n-omega", type=int, default=4001)
    p.add_argument("--out", default="eps2_compare.png",
                   help="Output plot path (extension .png/.pdf). "
                        "If --no-plot, write .npz instead")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot, write .npz of (omegas, eps2 per file)")
    p.add_argument("--label", nargs="+", default=None,
                   help="One label per file (default: filename)")
    args = p.parse_args()

    omegas, out = compute_eps2_from_files(
        args.files, eta_eV=args.eta_eV, n_max=args.n_max,
        omega_min_eV=args.omega_min_eV, omega_max_eV=args.omega_max_eV,
        n_omega=args.n_omega, kernel=args.kernel)

    labels = args.label or [Path(p).name for p in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("--label count must match --files count")

    if args.no_plot:
        out_path = Path(args.out).with_suffix(".npz")
        np.savez(out_path, omegas_eV=omegas,
                 **{lbl: d["eps2"] for lbl, d in zip(labels, out.values())})
        print(f"Wrote {out_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (path, d) in zip(labels, out.items()):
        ax.plot(omegas, d["eps2"],
                label=f"{label} (n={d['n_used']}, V={d['V']:.0f})", lw=1.0)
    ax.set_xlabel(r"$\omega$ (eV)")
    ax.set_ylabel(r"$\varepsilon_2(\omega)$")
    ax.set_title(f"ε₂(ω) at η = {args.eta_eV} eV"
                 + (f", first {args.n_max} eigvals" if args.n_max else ""))
    ax.set_xlim(args.omega_min_eV, args.omega_max_eV)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"Wrote {args.out}")
    for label, (path, d) in zip(labels, out.items()):
        peak = d["eps2"].max()
        peak_omega = omegas[d["eps2"].argmax()]
        print(f"  {label}: peak ε₂ = {peak:.2f} at ω = {peak_omega:.3f} eV "
              f"(n_used={d['n_used']})")


if __name__ == "__main__":
    _main()
