"""Fig. 4.1: decaying (sign-definite) and crossing denominator windows.

1/d = -i int_0^inf e^{i t d} dt, d = x + i eta, eta = 0.25 eV
(sigma_regularization_ev, src/gw/gw_config.py:1636), eps = 1e-4
(sigma_quadrature_eps, :1622).  Rules: LORRAX's own builder
minimax.analytic_box_rule (services/minimax/src/minimax/analytic_box.py:336),
called read-only through lorrax_minimax.py, on two boxes of an insulating
example (the windows of fig_windows.py):
  sign-definite: Re d in [-111.9, -0.96] eV  (conduction state tail)
  crossing:      Re d in [-21.0, 9.0] eV     (conduction resonant window)
Boxes are padded as gw.sigma_box_plan._box does (src/gw/sigma_box_plan.py:268).
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from lorrax_minimax import analytic_box_rule

ETA, EPS = 0.25, 1e-4


def pad_box(lo, hi):
    """gw.sigma_box_plan._box: 2 % real padding that never changes the sign topology."""
    pad = 0.02 * max(hi - lo, ETA)
    lo2 = lo - pad if lo <= 0 else max(lo - pad, 0.7 * lo)
    hi2 = hi + pad if hi >= 0 else min(hi + pad, 0.7 * hi)
    return (lo2, hi2, ETA, ETA)


sd_box = pad_box(0.6 - 110.375, 10.0 - 10.375 - 1.0)
cr_box = pad_box(0.6 - 2 * 10.375, 10.0 - 1.6)
sd = analytic_box_rule(sd_box, EPS)
cr = analytic_box_rule(cr_box, EPS)
print("sign-definite", sd_box, sd.times.size, sd.sup_error)
print("crossing", cr_box, cr.times.size, cr.sup_error)

fig = plt.figure(figsize=(S.WIDTH, 7.0))
gs = fig.add_gridspec(4, 2, height_ratios=[2.1, 0.42, 0.62, 1.9], width_ratios=[1, 1.15],
                      hspace=0.16, wspace=0.16, left=0.075, right=0.985, top=0.9, bottom=0.075)
ax = fig.add_subplot(gs[0, 0])
axs = fig.add_subplot(gs[1, 0], sharex=ax)
bx = fig.add_subplot(gs[0, 1])
bxs = fig.add_subplot(gs[1, 1], sharex=bx)
cx = fig.add_subplot(gs[3, :])

# ---- (a) sign-definite: along the rotated ray every member is a pure decay
tau = np.geomspace(1e-3, 30, 2000)
members = [-1.0, -4.0, -16.0, -110.0]
shade = ["#9ba4c0", "#7480a6", "#4f5d8a", S.NAVY]
for xm, c in zip(members, shade):
    ax.plot(tau, np.exp(xm * tau), color=c, lw=1.6)
    ax.text(1.0 / abs(xm), np.exp(-1.0) + 0.03, f"{abs(xm):g} eV", color=c, fontsize=9.0,
            ha="left", va="bottom")
ax.set_xscale("log")
ax.set_xlim(1e-3, 30)
ax.set_ylim(0, 1.12)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", which="both", labelbottom=False)
ax.set_title("(a)  Far from resonance: rotate time", x=0.0, pad=20)
ax.text(0.0, 1.02, r"$|e^{itd}|$ for $t = -i|t|$: a pure decay at rate |Re d|, Re d from −112 to −1 eV",
        transform=ax.transAxes, color=S.GREY, fontsize=9.0, va="bottom")
axs.plot(np.abs(sd.times), np.zeros(sd.times.size), "o", ms=6, color=S.RED, mec="white", mew=0.8)
axs.axhline(0, color=S.RULE, lw=0.8, zorder=0)
axs.set_ylim(-1, 1)
axs.set_yticks([])
axs.spines["left"].set_visible(False)
axs.set_xlabel("|t|  (ħ/eV, log scale)", labelpad=2)
axs.text(1.0, 0.95, f"{sd.times.size} nodes, spaced evenly in log t", transform=axs.transAxes,
         color=S.RED, fontsize=9.3, ha="right", va="bottom")

# ---- (b) crossing: on the real axis, oscillation that dies only at eta
t = np.linspace(0, 45, 6000)
for xm, c, lw in ((-21.0, S.IDLE, 0.7), (-6.0, S.GREY, 1.0), (2.0, S.NAVY, 1.5)):
    bx.plot(t, np.cos(xm * t) * np.exp(-ETA * t), color=c, lw=lw)
bx.plot(t, np.exp(-ETA * t), color=S.RED, lw=1.0, ls=(0, (3, 2)))
bx.text(6.0, 0.36, r"envelope $e^{-\eta t}$ for every member", color=S.RED, fontsize=9.0)
bx.text(44, -0.62, "members Re d = 2, −6, −21 eV", color=S.SOFT, fontsize=9.0, ha="right")
bx.axhline(0, color=S.RULE, lw=0.6)
bx.set_xlim(0, 45)
bx.set_ylim(-1.05, 1.12)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.tick_params(axis="x", labelbottom=False)
bx.set_title("(b)  Near resonance: only η damps it", x=0.0, pad=20)
bx.text(0.0, 1.02, r"Re $e^{itd}$ on the real time axis, Re d from −21 to +9 eV",
        transform=bx.transAxes, color=S.GREY, fontsize=9.0, va="bottom")
line = cr.times[np.abs(cr.times.real) > 1e-12]
bxs.plot(line.real, np.zeros(line.size), "|", ms=9, mew=0.9, color=S.RED)
bxs.axhline(0, color=S.RULE, lw=0.8, zorder=0)
bxs.set_ylim(-1, 1)
bxs.set_yticks([])
bxs.spines["left"].set_visible(False)
bxs.set_xlabel("Re t  (ħ/eV)", labelpad=2)
bxs.text(1.0, 0.95, f"{cr.times.size} nodes ({line.size} along the line), out to where the envelope falls below ε",
         transform=bxs.transAxes, color=S.RED, fontsize=9.3, ha="right", va="bottom")

# ---- (c) the crossing rule in the complex time plane
tt = cr.times
on_axis = np.abs(tt.real) <= 1e-12
cx.axhline(0, color=S.IDLE, lw=0.7)
cx.axvline(0, color=S.IDLE, lw=0.7)
cx.plot(tt.real[~on_axis], tt.imag[~on_axis], "o", ms=3.4, color=S.NAVY, zorder=3,
        label="nodes along the line: the terms near resonance")
cx.plot(tt.real[on_axis], tt.imag[on_axis], "o", ms=4.2, color=S.RED, zorder=4,
        label="nodes on the imaginary axis: the far side's decays")
cx.set_xlim(-1.5, 45)
ylo = 1.25 * tt.imag.min()
cx.set_ylim(ylo, -0.55 * ylo)
cx.set_xlabel("Re t  (ħ/eV)", labelpad=2)
cx.set_ylabel("Im t  (ħ/eV)")
cx.set_title("(c)  LORRAX nodes for the near-resonance window, complex time plane (vertical axis stretched)", x=-0.06)
cx.legend(loc="lower right", fontsize=9.0, handletextpad=0.3)
cx.annotate("the line dips below the real axis, where terms on the far side of the window decay",
            xy=(14, tt.imag[~on_axis].min()), xytext=(4, -0.42 * ylo), color=S.SOFT, fontsize=9.0,
            arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
cx.spines["left"].set_position(("data", -1.5))
S.save(fig, "sigma-kinds")
