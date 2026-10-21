"""Fig. 2.4: resonant and sign-definite terms of Sigma (2026-10-20, owner: explain crossing and sign-definite windows early,
with a simpler figure than the product-window map of Fig. 4.2).

Schematic in eV. Each term of the empty-state branch of Sigma^c has the denominator d = omega - (E + Omega), with E an
empty-state energy and Omega a pole of W^c. (a) Re d against the read frequency omega over a read region 0..10 eV: terms
whose E + Omega lies inside the read region cross zero there (resonant); terms above it never do (sign-definite).
(b) The two kinds in time: a resonant term on the real time axis oscillates and decays only at eta; a sign-definite term
on the rotated (imaginary) time axis is a pure decay at |Re d|.
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S

READ = (0.0, 10.0)
res = [2.5, 5.0, 8.0]              # E + Omega inside the read region
sdef = [14.0, 22.0, 35.0]          # E + Omega above it

fig = plt.figure(figsize=(S.WIDTH, 3.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.28, left=0.075, right=0.985, top=0.86, bottom=0.17)
ax = fig.add_subplot(gs[0, 0])
bx = fig.add_subplot(gs[0, 1])

# ---- (a) denominators against the read frequency
w = np.linspace(-1, 12, 400)
ax.axvspan(*READ, color=S.PANEL, lw=0)
ax.text(5.0, 7.2, "read region", color=S.SOFT, fontsize=9.0, ha="center")
ax.axhline(0, color=S.NAVY, lw=1.0)
for x in res:
    ax.plot(w, w - x, color=S.RED, lw=1.5)
    ax.plot([x], [0], "o", ms=5, mfc="white", mec=S.RED, mew=1.5, zorder=5)
for x in sdef:
    ax.plot(w, w - x, color=S.GREY, lw=1.3)
ax.text(-0.7, 3.0, "resonant:\nd crosses zero", color=S.RED, fontsize=9.0, ha="left", va="bottom")
ax.text(11.6, -14.5, "sign-definite:\nd < 0 for every read", color=S.GREY, fontsize=9.0, ha="right", va="top")
ax.set_xlim(-1, 12)
ax.set_ylim(-36, 9)
ax.set_xlabel(r"read frequency $\omega-E_F$  (eV)", labelpad=2)
ax.set_ylabel(r"Re $d=\omega-(E+\Omega)$  (eV)", labelpad=2)
ax.set_title(r"(a)  Denominators of Σ over the read region", x=-0.02, loc="left")

# ---- (b) the two kinds in time
eta = 0.6
t = np.linspace(0, 6, 3000)
bx.plot(t, np.cos(3.0 * t) * np.exp(-eta * t), color=S.RED, lw=1.5, label=r"resonant, real time: decays only at η")
bx.plot(t, np.exp(-6.0 * t), color=S.GREY, lw=1.5, label=r"sign-definite, rotated time: decays at $|$Re $d|$")
bx.axhline(0, color=S.RULE, lw=0.6)
bx.set_xlim(0, 6)
bx.set_ylim(-1.05, 1.15)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xlabel(r"time  (ħ/eV)", labelpad=2)
bx.set_title("(b)  The two kinds in time", x=-0.02, loc="left")
bx.legend(loc="upper right", fontsize=8.6, handlelength=1.6, borderaxespad=0.1)
S.save(fig, "crossing")
