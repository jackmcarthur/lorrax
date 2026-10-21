"""Fig. 2.1: the poles of G and the three time-axis prescriptions.

Built on the real Si band energies at two k points (data/si_dense_scalar_bands.json, see fig_si_spectrum.py),
measured from midgap E_F. Replaces the earlier fig_poles.py (panels a-c) and fig_sampling.py (2026-10-20, owner:
plain rectangular complex plane, no peak; decaying exponentials in time; one connected figure for the prescriptions).
(a) poles of the time-ordered G: occupied states at E + i*eta, empty states at E - i*eta (eta drawn enlarged);
    the real and imaginary axes, along which the three prescriptions evaluate G.
(b) imaginary axis: e^{-|E| tau}, pure decays;
(c) broadened real axis: Re e^{-iEt - eta t}, oscillation under a decaying envelope; the time step dt resolves the
    oscillation and the span T is set by the decay of the envelope below eps;
(d) exact real axis: Re e^{-iEt}, oscillation that never decays.
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
two = el[[0, 4]] - ef
occ = np.unique(np.round(two[:, :nocc].ravel(), 1))
emp = np.unique(np.round(two[:, nocc:].ravel(), 1))
emp = emp[emp < 28]
ETA_DRAW = 1.2                                   # enlarged pole offset in the drawing (eV)

HL = {float(emp[np.argmin(np.abs(emp - v))]): c for v, c in ((2.0, S.NAVY), (8.0, S.BLUE), (20.0, S.RED))}

fig = plt.figure(figsize=(S.WIDTH, 5.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.22,
                      left=0.06, right=0.985, top=0.92, bottom=0.1)
ax = fig.add_subplot(gs[0, :])
bx, cx, dx = (fig.add_subplot(gs[1, k]) for k in range(3))

# ---- (a) complex frequency plane, ordinary rectangular axes
ax.axhline(0, color=S.NAVY, lw=1.4)
ax.axvline(0, color=S.BLUE, lw=1.4)
ax.plot(occ, np.full_like(occ, ETA_DRAW), "x", ms=7, mew=1.6, color=S.SOFT, label="occupied state, pole at $E+i\\eta$")
ax.plot(emp, np.full_like(emp, -ETA_DRAW), "+", ms=8, mew=1.6, color=S.GREY, label="empty state, pole at $E-i\\eta$")
for e0, c in HL.items():
    ax.plot([e0], [-ETA_DRAW], "o", ms=8, mfc="white", mec=c, mew=2.0, zorder=5)
ax.text(27.5, 0.35, "real axis", color=S.NAVY, fontsize=9.3, ha="right", va="bottom")
ax.text(0.5, 3.6, "imaginary axis", color=S.BLUE, fontsize=9.3, ha="left", va="center")
ax.annotate("", xy=(-14.0, ETA_DRAW), xytext=(-14.0, 0),
            arrowprops=dict(arrowstyle="<->", color=S.NAVY, lw=0.9, shrinkA=0, shrinkB=0))
ax.text(-13.6, ETA_DRAW / 2, "η", color=S.NAVY, fontsize=10, va="center")
ax.set_xlim(-15, 28)
ax.set_ylim(-4.2, 4.2)
ax.set_xlabel(r"Re $z$  (eV, from $E_F$)", labelpad=2)
ax.set_ylabel(r"Im $z$  (eV)", labelpad=2)
ax.set_title("(a)  Poles of G for Si at two k points (offset η drawn enlarged)", x=-0.02, loc="left")
ax.legend(loc="lower right", ncol=2, fontsize=8.8, borderaxespad=0.15, handletextpad=0.3)

# ---- (b) imaginary axis: pure decays
tau = np.linspace(0, 1.2, 2000)
for (e0, c), lev in zip(sorted(HL.items()), (0.55, 0.3, 0.1)):
    bx.plot(tau, np.exp(-e0 * tau), color=c, lw=1.6)
    k = np.argmin(np.abs(np.exp(-e0 * tau) - lev))
    bx.text(tau[k] + 0.03, lev + 0.03, f"{e0:.0f} eV", color=c, fontsize=8.8)
bx.set_xlim(0, 1.2)
bx.set_ylim(0, 1.05)
bx.set_xlabel(r"imaginary time $\tau$  (ħ/eV)", labelpad=2)
bx.set_title(r"(b)  Imaginary axis: $e^{-|E|\tau}$", x=-0.02, loc="left")

# ---- (c) broadened real axis: decaying oscillation, with the step and the span
eta, e0, eps = 1.0, float(sorted(HL)[1]), 0.05
T = np.log(1 / eps) / eta
t = np.linspace(0, 4.0, 4000)
cx.plot(t, np.cos(e0 * t) * np.exp(-eta * t), color=S.BLUE, lw=1.4)
cx.plot(t, np.exp(-eta * t), color=S.IDLE, lw=1.0, ls=(0, (3, 2)))
cx.plot(t, -np.exp(-eta * t), color=S.IDLE, lw=1.0, ls=(0, (3, 2)))
dt = 0.6 * np.pi / e0
ts = np.arange(0, T + 1e-9, dt)
cx.plot(ts, np.cos(e0 * ts) * np.exp(-eta * ts), "o", ms=3.2, color=S.NAVY, zorder=5)
cx.axvline(T, color=S.RED, lw=1.0, ls=(0, (4, 2)))
cx.text(T - 0.06, 0.82, r"span $T$", color=S.RED, fontsize=8.8, ha="right")
for tt in (ts[8], ts[9]):
    cx.plot([tt, tt], [-0.62, -0.42], color=S.NAVY, lw=0.8)
cx.annotate("", xy=(ts[8], -0.52), xytext=(ts[9], -0.52),
            arrowprops=dict(arrowstyle="<->", color=S.NAVY, lw=0.9, shrinkA=0, shrinkB=0))
cx.text(ts[9] + 0.08, -0.52, r"step $\Delta t$", color=S.NAVY, fontsize=8.8, va="center")
cx.text(1.0, 0.55, r"envelope $e^{-\eta t}$", color=S.GREY, fontsize=8.8)
cx.axhline(0, color=S.RULE, lw=0.6)
cx.set_xlim(0, 4.0)
cx.set_ylim(-1.05, 1.05)
cx.set_xlabel(r"real time $t$  (ħ/eV)", labelpad=2)
cx.set_title(r"(c)  Broadened real axis: $e^{-iEt-\eta t}$", x=-0.02, loc="left")

# ---- (d) exact real axis: no decay
cx_t = np.linspace(0, 4.0, 4000)
dx.plot(cx_t, np.cos(e0 * cx_t), color=S.BLUE, lw=1.4)
dx.axhline(0, color=S.RULE, lw=0.6)
dx.annotate("", xy=(4.0, 1.12), xytext=(3.0, 1.12), annotation_clip=False,
            arrowprops=dict(arrowstyle="->", color=S.GREY, lw=0.9))
dx.text(2.95, 1.12, "never decays", color=S.GREY, fontsize=8.8, ha="right", va="center")
dx.set_xlim(0, 4.0)
dx.set_ylim(-1.05, 1.3)
dx.set_xlabel(r"real time $t$  (ħ/eV)", labelpad=2)
dx.set_title(r"(d)  Exact real axis: $e^{-iEt}$", x=-0.02, loc="left")

for a in (bx, cx, dx):
    a.set_yticks([])
    a.spines["left"].set_visible(False)
S.save(fig, "prescriptions")
