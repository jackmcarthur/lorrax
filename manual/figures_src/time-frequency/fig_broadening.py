"""Fig. 1.2: what the broadening eta does to a self-energy.

MODEL, not LORRAX output.  Sigma^c_n(omega) of the Si valence-band maximum
built from the real Si band energies of fig_si_spectrum.py (537 bands, 8
irreducible k with their weights) and a model screened interaction: poles
Omega_j at 2-40 eV with weights peaked at a 16.5 eV plasmon, constant
matrix elements tapered with |E_m - E_F|.  The form is Eq. (1.2) of
docs/theory/frequency-treatments.md:
  sum_m,j B_j [ (1 - f_m)/(w - E_m - Omega_j + i eta) + f_m/(w - E_m + Omega_j - i eta) ].
eta = 0.25 eV is sigma_regularization_ev (src/gw/gw_config.py:1636).
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import style as S

RY = 13.605693122994
d = json.load(open(os.path.join(S.HERE, "data", "si_dense_scalar_bands.json")))
el = np.asarray(d["el_ry"])[0] * RY
w_k = np.asarray(d["w"]).ravel()
nocc = int(np.asarray(d["ifmax"]).ravel()[0])
vbm = el[:, :nocc].max()
ef = 0.5 * (vbm + el[:, nocc:].min())
E = (el - ef).ravel()                      # energies from E_F (midgap), eV
wk = np.repeat(w_k, el.shape[1])
occ = np.tile(np.arange(el.shape[1]) < nocc, el.shape[0])
M2 = wk / (1.0 + (E / 8.0) ** 2)           # model matrix elements, tapered away from E_F

Om = np.sort(np.random.default_rng(3).uniform(1.5, 30.0, 58))   # irregular, as in a real W
B = 1.0 + 0.5 * np.exp(-0.5 * ((Om - 16.5) / 2.0) ** 2)
B /= B.sum()

w = np.linspace(-10.0, 10.0, 6000)


def sigma(eta):
    out = np.zeros(w.size, complex)
    for o, b in zip(Om, B):
        emp = (M2 * ~occ)[None, :] / (w[:, None] - E[None, :] - o + 1j * eta)
        hol = (M2 * occ)[None, :] / (w[:, None] - E[None, :] + o - 1j * eta)
        out += b * (emp.sum(1) + hol.sum(1))
    return out


etas = [0.05, 0.25, 1.0]
sig = [sigma(x) for x in etas]
allv = np.concatenate([np.r_[s.real, s.imag] for s in sig]); lo, hi = np.percentile(allv, [0.2, 99.8]); pad = 0.06 * (hi - lo); scale = 1.0

fig, axes = plt.subplots(3, 1, figsize=(S.WIDTH, 5.4), sharex=True,
                         gridspec_kw=dict(hspace=0.12, left=0.03, right=0.985, top=0.87, bottom=0.1))
for ax, eta, s in zip(axes, etas, sig):
    ax.axhline(0, color=S.RULE, lw=0.7)
    ax.axvline(0, color=S.IDLE, lw=0.8, ls=(0, (3, 2)))
    ax.plot(w, s.imag / scale, color=S.GREY, lw=1.0, label=r"Im $\Sigma^c$")
    ax.plot(w, s.real / scale, color=S.NAVY, lw=1.2, label=r"Re $\Sigma^c$")
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    lab = f"η = {eta:g} eV" + ("   (LORRAX default)" if eta == 0.25 else "")
    ax.text(0.005, 0.06, lab, transform=ax.transAxes, va="bottom", fontsize=10,
            color=S.RED if eta == 0.25 else S.NAVY, fontweight="semibold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)
axes[0].set_title(r"A model $\Sigma^c(\omega)$ of the Si valence-band maximum at three broadenings",
                  x=0.0, pad=26)
axes[0].text(0.0, 1.04, "every pole becomes a feature of width η; same arbitrary scale in all panels; dashed line: $E_F$",
             transform=axes[0].transAxes, color=S.GREY, fontsize=9.3, va="bottom")
axes[0].legend(loc="upper right", ncol=2, fontsize=9.3, borderaxespad=0.1)
axes[-1].set_xlabel(r"$\omega - E_F$  (eV)", labelpad=2)
axes[-1].set_xlim(w[0], w[-1])
axes[-1].set_xticks(np.arange(-10, 11, 2))
S.save(fig, "broadening")
