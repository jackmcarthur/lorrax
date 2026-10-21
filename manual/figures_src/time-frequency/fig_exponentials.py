"""Fig. 3.1: why the nodes of an exponential rule are log-spaced.

1/Delta = int_0^inf e^{-Delta t} dt = int Delta t e^{-Delta t} d(ln t) / Delta.
Per unit ln t the integrand Delta t e^{-Delta t} is one bump of fixed shape
centred at t = 1/Delta, so the transitions of a window [Delta_min, Delta_max]
fill ln R with identical bumps.

Nodes: LORRAX's static rule for 1/Delta on [1, 100] eV at the default
minimax_target_error = 1e-6 Ry^-1 (src/gw/gw_config.py:1573), solved by the
service the code calls (src/gw/minimax_screening.py:1062-1110 ->
minimax.levelled.noncrossing_levelled); 11 nodes. Uniform: the best midpoint
grid with the same 11 nodes (laplace_rules.uniform_best).
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from laplace_rules import uniform_best
from lorrax_minimax import noncrossing_levelled

RY = 13.605693122994        # eV per Ry
DMIN, DMAX = 1.0, 100.0     # eV
R = DMAX / DMIN
target = 1e-6 / RY * DMIN   # minimax_target_error (Ry^-1) scaled to [1, R] with Delta_min in eV
tau, alpha, N, err = noncrossing_levelled(R, target, 64)
tau = np.asarray(tau) / DMIN                  # 1/eV
tu, wu, eu = uniform_best(R, N)
tu = tu / DMIN
print("static rule nodes:", N, "error", err, "uniform best error", eu)

fig = plt.figure(figsize=(S.WIDTH, 4.6))
gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.55, 0.55], hspace=0.12,
                      left=0.075, right=0.985, top=0.9, bottom=0.12)
ax = fig.add_subplot(gs[0])
s1 = fig.add_subplot(gs[1], sharex=ax)
s2 = fig.add_subplot(gs[2], sharex=ax)

t = np.geomspace(1e-3, 40, 3000)
deltas = np.geomspace(DMIN, DMAX, 5)
cols = [S.NAVY, "#34416a", "#4f5d8a", "#7480a6", "#9ba4c0"][::-1]
for D, c in zip(deltas, cols):
    y = D * t * np.exp(-D * t)
    ax.plot(t, y, color=c, lw=1.7)
    ax.text(1.0 / D, 0.385, f"x = {D:g} eV" if D in (1.0, 100.0) else f"{D:.3g}",
            ha="center", va="bottom", color=c, fontsize=9.3)
ax.set_xscale("log")
ax.set_xlim(1e-3, 40)
ax.set_ylim(0, 0.53)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", labelbottom=False)
ax.set_title(r"Per unit $\ln t$, every term $e^{-x t}$ is the same bump, shifted to $t = 1/x$",
             x=-0.06)
ax.text(-0.01, 0.2, r"$x t\, e^{-x t}$", transform=ax.transAxes, ha="right",
        va="center", color=S.SOFT, fontsize=10.5)
ax.annotate("", xy=(1.0, 0.465), xytext=(0.01, 0.465),
            arrowprops=dict(arrowstyle="<->", color=S.GREY, lw=0.9, shrinkA=0, shrinkB=0))
ax.text(0.1, 0.475, r"$\ln R$: two decades of $t$ for $R = x_{\max}/x_{\min} = 100$", ha="center", va="bottom",
        color=S.GREY, fontsize=9.3)

def sci(v):
    m, e = f"{v:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


for s, nodes, col, label in ((s1, tau, S.RED, f"minimax rule: {N} nodes, error {sci(err)}"),
                             (s2, tu, S.GREY, f"best uniform grid: {N} nodes, error {sci(eu)}")):
    s.axhline(-0.4, color=S.RULE, lw=0.8)
    s.plot(nodes, np.full_like(nodes, -0.4), "o", ms=6.5, color=col, mec="white", mew=0.8, zorder=3)
    s.set_ylim(-1, 1)
    s.set_yticks([])
    for side in ("left", "bottom"):
        s.spines[side].set_visible(False)
    s.text(0.003, 0.72, label, transform=s.transAxes, ha="left", va="center",
           color=col if col != S.GREY else S.SOFT, fontsize=9.3, linespacing=1.15)
s1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax.tick_params(axis="x", which="both", labelbottom=False)
s2.spines["bottom"].set_visible(True)
s2.set_xlabel("time t  (ħ/eV, log scale)", labelpad=3)
S.save(fig, "exponentials")
