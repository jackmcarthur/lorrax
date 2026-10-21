"""Fig. 2.1: poles of G and the exponentials they are in time.

Schematic built on the real Si band energies at Gamma (data/si_dense_scalar_bands.json,
see fig_si_spectrum.py), measured from midgap E_F.
(a) the poles of the time-ordered G in the complex frequency plane: occupied
    states at E + i*eta, empty states at E - i*eta; eta drawn exaggerated;
(b) three empty-state poles as functions of real time, Re e^{-iEt - eta t}
    with eta exaggerated to 1 eV for visibility;
(c) the same three poles at imaginary time, e^{-|E| tau}.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import style as S

RY = 13.605693122994
d = json.load(open(os.path.join(S.HERE, "data", "si_dense_scalar_bands.json")))
el = np.asarray(d["el_ry"])[0] * RY
nocc = int(np.asarray(d["ifmax"]).ravel()[0])
ef = 0.5 * (el[:, :nocc].max() + el[:, nocc:].min())
two = el[[0, 4]] - ef                          # two k points (Gamma and one more), from E_F
occ = np.unique(np.round(two[:, :nocc].ravel(), 1))
emp = np.unique(np.round(two[:, nocc:].ravel(), 1))
emp = emp[emp < 40]
ETA_DRAW = 0.45                                # exaggerated offset in the drawing

HL = {float(emp[np.argmin(np.abs(emp - v))]): c        # highlighted empty-state poles (eV)
      for v, c in ((2.0, S.NAVY), (8.0, S.BLUE), (30.0, S.RED))}

fig = plt.figure(figsize=(S.WIDTH, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.35, 1],
                      hspace=0.62, wspace=0.18, left=0.07, right=0.985, top=0.9, bottom=0.1)
ax = fig.add_subplot(gs[0, :])
bx = fig.add_subplot(gs[1, 0])
cx = fig.add_subplot(gs[1, 1])

# ---- (a) complex frequency plane
ax.axhline(0, color=S.NAVY, lw=0.9)
ax.axvspan(-10, 10, ymin=0.0, ymax=1.0, color=S.RED, alpha=0.06, lw=0)
ax.plot(occ, np.full_like(occ, ETA_DRAW), "x", ms=7, mew=1.6, color=S.SOFT, label="occupied state: pole at $E + i\\eta$")
ax.plot(emp, np.full_like(emp, -ETA_DRAW), "+", ms=8, mew=1.6, color=S.GREY, label="empty state: pole at $E - i\\eta$")
for e0, c in HL.items():
    ax.plot([e0], [-ETA_DRAW], "o", ms=8, mfc="white", mec=c, mew=2.0, zorder=5)
# Lorentzian shadow of one pole on the real axis
xs = np.linspace(-15, 40, 3000)
e0 = min(HL)
lor = 0.75 * ETA_DRAW ** 2 / ((xs - e0) ** 2 + ETA_DRAW ** 2)
ax.fill_between(xs, 0, lor, color=S.NAVY, alpha=0.18, lw=0)
ax.annotate("on the real axis: a peak of width η", xy=(e0 + 0.5, 0.5), xytext=(11, 0.78),
            color=S.SOFT, fontsize=9.3, arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
ax.annotate("", xy=(-13.3, ETA_DRAW), xytext=(-13.3, 0),
            arrowprops=dict(arrowstyle="<->", color=S.NAVY, lw=0.9, shrinkA=0, shrinkB=0))
ax.text(-13.0, ETA_DRAW / 2, "η", color=S.NAVY, fontsize=10, va="center")
ax.text(0, -1.02, "read region ±10 eV", color=S.RED, fontsize=9.3, ha="center", va="bottom")
ax.set_xlim(-15, 40)
ax.set_ylim(-1.05, 1.05)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel(r"Re $\omega - E_F$  (eV)", labelpad=2)
ax.set_title("(a)  Poles of G in the complex frequency plane (Si energies at two k points; offset η drawn enlarged)", x=-0.05)
ax.legend(loc="lower right", ncol=2, fontsize=9.0, borderaxespad=0.1, handletextpad=0.3)

# ---- (b) real time
eta = 1.0
t = np.linspace(0, 6, 5000)
for e0, c in HL.items():
    lw = 1.6 if e0 < 20 else 0.8
    bx.plot(t, np.cos(e0 * t) * np.exp(-eta * t), color=c, lw=lw, label=f"E = {e0:.1f} eV")
bx.plot(t, np.exp(-eta * t), color=S.IDLE, lw=1.0, ls=(0, (3, 2)))
bx.plot(t, -np.exp(-eta * t), color=S.IDLE, lw=1.0, ls=(0, (3, 2)))
bx.text(2.2, 0.42, r"envelope $e^{-\eta t}$: decay rate η", color=S.GREY, fontsize=9.0)
bx.axhline(0, color=S.RULE, lw=0.6)
bx.set_xlim(0, 6)
bx.set_ylim(-1.05, 1.1)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xlabel("real time t  (ħ/eV)", labelpad=2)
bx.set_title(r"(b)  Real time: Re $e^{-iEt-\eta t}$, oscillating at E", x=-0.05)
bx.legend(loc="lower right", fontsize=8.8, ncol=3, handlelength=1.4, columnspacing=0.9,
          borderaxespad=0.1)

# ---- (c) imaginary time
tau = np.linspace(0, 1.5, 2000)
for (e0, c), lev in zip(sorted(HL.items()), (0.55, 0.3, 0.1)):
    cx.plot(tau, np.exp(-e0 * tau), color=c, lw=1.6)
    k = np.argmin(np.abs(np.exp(-e0 * tau) - lev))
    cx.text(tau[k] + 0.04, lev + 0.02, f"{e0:.1f} eV", color=c, fontsize=9.0)
cx.set_xlim(0, 1.5)
cx.set_ylim(0, 1.05)
cx.set_yticks([])
cx.spines["left"].set_visible(False)
cx.set_xlabel("imaginary time τ  (ħ/eV)", labelpad=2)
cx.set_title(r"(c)  Imaginary time: $e^{-|E|\tau}$, only decay", x=-0.05)
S.save(fig, "poles")
