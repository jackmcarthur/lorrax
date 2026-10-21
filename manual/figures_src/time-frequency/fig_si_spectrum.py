"""Fig. 1.1: the single-particle spectrum of Si.

Data: data/si_dense_scalar_bands.json, the band energies (mf_header/kpoints/el,
Ry) of a complete-basis WFN of Si: scalar-relativistic, 4x4x4 k grid (8
irreducible points), 25 Ry wavefunction cutoff, all 537 bands of the basis,
written by psp.run_dense_h.  Read-only extract (h5py, header datasets only) of
the WFN_dense_scalar.h5 of a Perlmutter run (2026-09-26).
Energies are plotted from E_F, taken at midgap.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import style as S

RY = 13.605693122994
d = json.load(open(os.path.join(S.HERE, "data", "si_dense_scalar_bands.json")))
el = np.asarray(d["el_ry"])[0] * RY            # (nk, nb) eV
w = np.asarray(d["w"]).ravel()
nocc = int(np.asarray(d["ifmax"]).ravel()[0])
vbm = el[:, :nocc].max()
cbm = el[:, nocc:].min()
ef = 0.5 * (vbm + cbm)
e = el - ef
nk, nb = e.shape
print(nk, nb, "VBM", vbm, "gap", cbm - vbm, "top", e.max())
ROI = (-10.0, 10.0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.WIDTH, 4.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.06,
                                              left=0.085, right=0.985, top=0.86, bottom=0.12))
n = np.arange(1, nb + 1)
for ik in range(nk):
    ax.plot(n, e[ik], ".", ms=1.6, color=S.NAVY, alpha=0.55)
ax.axhspan(*ROI, color=S.RED, alpha=0.12, lw=0)
ax.text(nb * 0.98, ROI[1] + 4, "read region: ±10 eV about $E_F$",
        color=S.RED, ha="right", va="bottom", fontsize=9.5)
ax.set_xlim(0, nb + 5)
ax.set_ylim(-25, e.max() + 12)
ax.set_xlabel("band index n", labelpad=2)
ax.set_ylabel(r"$E_{n\mathbf{k}} - E_F$  (eV)")
ax.set_title(f"Si: every band of a 25 Ry plane-wave basis ({nb} bands, 4×4×4 k grid)",
             x=-0.1, pad=26)
ax.text(-0.1, 1.035, f"the single-particle spectrum runs to {e.max():.0f} eV ({e.max() / RY:.1f} Ry) above $E_F$",
        transform=ax.transAxes, color=S.GREY, fontsize=9.3, va="bottom")
ax.grid(True, axis="y", color=S.RULE, lw=0.6)

# density of states (k-weighted histogram of the levels)
grid = np.linspace(-25, e.max() + 12, 2000)
sig = 2.0                                         # Gaussian smoothing (eV), display only
wk = np.repeat(w, nb)
dos = (wk[None, :] * np.exp(-0.5 * ((grid[:, None] - e.ravel()[None, :]) / sig) ** 2)).sum(1)
dos /= sig * np.sqrt(2 * np.pi)
bx.fill_betweenx(grid, 0, dos, color=S.NAVY, alpha=0.25, lw=0)
bx.plot(dos, grid, color=S.NAVY, lw=1.2)
bx.set_xlim(0, dos.max() * 1.1)
bx.axhspan(*ROI, color=S.RED, alpha=0.12, lw=0)
bx.set_xlabel("states per eV", labelpad=2)
bx.set_title("density of levels (2 eV smoothing)", x=0.0, pad=26, fontsize=10.5)
bx.grid(True, axis="y", color=S.RULE, lw=0.6)
bx.tick_params(axis="y", left=False)
S.save(fig, "si-spectrum")
