"""Fig. 3.1: the time-frequency transform made concrete.

1/x = int_0^inf e^{-xt} dt  ~  sum_i w_i e^{-x t_i}  for decay rates x in [1, 100] eV,
with the N-node minimax rules of LORRAX's static-rule solver,
minimax.levelled.noncrossing_levelled (services/minimax/src/minimax/levelled.py:316),
called read-only; with x_min = 1 eV the rule on [1, R] is the physical one in eV.
(a) linear axes: 1/x, the 4-node sum, and its four weighted exponentials;
(b) the nodes and weights of the 4- and 11-node rules.
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from laplace_rules import evaluate, minimax

R = 100.0
C4 = ["#1f2a44", "#2c4f8f", "#7a86a8", "#b88a1e"]

t4, w4, e4 = minimax(R, 4)
t11, w11, e11 = minimax(R, 11)
print("4 nodes", t4, w4, e4)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.WIDTH, 4.3),
                             gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.22,
                                              left=0.07, right=0.985, top=0.78, bottom=0.13))

x = np.linspace(0.005, 12, 6000)
ax.axvspan(1, 12, color=S.PANEL, lw=0, zorder=0)
ax.plot(x, 1 / x, color=S.NAVY, lw=3.0, label=r"$1/x$", zorder=3)
order = np.argsort(t4)
for k, i in enumerate(order):
    ax.plot(x, w4[i] * np.exp(-x * t4[i]), color=C4[k], lw=1.2, zorder=2)
ax.plot(x, evaluate(t4, w4, x), color=S.RED, lw=1.6, ls=(0, (4, 2)),
        label="sum of the four terms", zorder=4)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.4)
ax.set_xlabel("decay rate x  (eV)", labelpad=2)
ax.set_ylabel(r"eV$^{-1}$")
ax.set_title("(a)  Four weighted exponentials add up to 1/x", x=-0.08, pad=8)
ax.text(1.25, 4.25, "window: x from 1 to 100 eV  →", color=S.SOFT, fontsize=9.3, va="top")
ax.annotate(r"terms $w_i\,e^{-x t_i}$", xy=(3.1, w4[order[1]] * np.exp(-3.1 * t4[order[1]])),
            xytext=(4.0, 1.2), color=S.SOFT, fontsize=9.3,
            arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
ax.annotate("below the window the sum\nflattens while 1/x diverges", xy=(0.12, 3.45),
            xytext=(3.2, 2.9), color=S.SOFT, fontsize=9.0, linespacing=1.2,
            arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
ax.legend(loc="upper right", fontsize=9.3, handlelength=2.4)

bx.vlines(t11, 0, w11, color=S.IDLE, lw=1.0)
bx.plot(t11, w11, "o", ms=4.5, color=S.IDLE, label="11 nodes")
for k, i in enumerate(order):
    bx.vlines(t4[i], 0, w4[i], color=C4[k], lw=1.8)
    bx.plot(t4[i], w4[i], "o", ms=7, color=C4[k], label="4 nodes" if k == 0 else None)
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_ylim(5e-3, 20)
bx.set_xlabel(r"time node $t_i$  (ħ/eV, log scale)", labelpad=2)
bx.set_ylabel(r"weight $w_i$  (eV$^{-1}$)")
bx.set_title("(b)  Nodes and weights", x=-0.12, pad=8)
bx.legend(loc="upper left", fontsize=9.0, handletextpad=0.3)
bx.grid(True, which="major", color=S.RULE, lw=0.6)

fig.suptitle(r"$\dfrac{1}{x} = \int_0^\infty e^{-x t}\,dt \;\approx\; \sum_{i=1}^{N} w_i\, e^{-x t_i}$"
             "      for every decay rate x in a window",
             x=0.07, ha="left", y=0.975, fontsize=12.5, color=S.NAVY)
S.save(fig, "transform")
