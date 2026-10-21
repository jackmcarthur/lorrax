"""Fig. 2.4: what a uniform grid resolves.

Illustrations (not LORRAX output).
(a) Two samples per period: a cosine sampled finely enough, and too coarsely,
    where the samples lie exactly on a slower cosine (aliasing).
(b) A peak of half-width eta in frequency, sampled at two steps.
(c) The same pole in time, e^{-iEt - eta t}: the step is set by the fastest
    oscillation of the band (energies measured from mid-band), the span by the
    horizon ln(1/eps)/eta.  eps = 1e-4 is sigma_quadrature_eps (src/gw/gw_config.py:1622).
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S

EPS = 1e-4
fig = plt.figure(figsize=(S.WIDTH, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1], width_ratios=[1, 1.25], hspace=0.55,
                      wspace=0.2, left=0.035, right=0.985, top=0.86, bottom=0.08)
ax0 = fig.add_subplot(gs[0, :])
ax = fig.add_subplot(gs[1, 0])
bx = fig.add_subplot(gs[1, 1])

# ---- (a) aliasing
E = 7.0
t = np.linspace(0, 3, 3000)
dt_ok, dt_bad = 0.35, 0.55
alias = 2 * np.pi / dt_bad - E
ax0.plot(t, np.cos(E * t), color=S.NAVY, lw=1.4, label=f"signal cos(Et), period {2*np.pi/E:.2f}")
to = np.arange(0, 3.0001, dt_ok)
ax0.plot(to, np.cos(E * to), "o", ms=5, color=S.NAVY, zorder=4,
         label=f"step {dt_ok}: more than two samples per period")
tb = np.arange(0, 3.0001, dt_bad)
ax0.plot(t, np.cos(alias * t), color=S.RED, lw=1.2, ls=(0, (4, 2)),
         label="what the coarse samples see: a slower cosine")
ax0.plot(tb, np.cos(E * tb), "s", ms=7, mfc="white", mec=S.RED, mew=1.6, zorder=5,
         label=f"step {dt_bad}: fewer than two samples per period")
ax0.axhline(0, color=S.RULE, lw=0.6)
ax0.set_xlim(0, 3)
ax0.set_ylim(-1.15, 1.15)
ax0.set_yticks([])
ax0.spines["left"].set_visible(False)
ax0.set_xlabel("time t  (arbitrary units)", labelpad=2)
ax0.set_title("(a)  Two samples per period: a coarser grid mistakes a fast oscillation for a slow one",
              x=0.0, pad=44)
ax0.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=9.0,
           handlelength=2.2, columnspacing=1.6)

# ---- (b) frequency
x = np.linspace(-6, 6, 2000)
lor = lambda u: 1.0 / (u ** 2 + 1.0)
ax.plot(x, lor(x), color=S.NAVY, lw=1.6)
fine = 2 * np.pi / np.log(1 / EPS)          # 0.68 eta
xs = np.arange(-6, 6.001, fine) + 0.31
xs = xs[np.abs(xs) <= 6]
ax.plot(xs, lor(xs), "o", ms=4.6, color=S.NAVY, zorder=4)
xc = np.arange(-6, 6.001, 3.0) + 1.5
ax.plot(xc, lor(xc), "s", ms=6.5, mfc="white", mec=S.RED, mew=1.6, zorder=5)
ax.annotate("", xy=(-1, 0.5), xytext=(1, 0.5),
            arrowprops=dict(arrowstyle="<->", color=S.GREY, lw=0.9, shrinkA=0, shrinkB=0))
ax.text(0, 0.535, "2η", ha="center", va="bottom", color=S.GREY, fontsize=9.5)
ax.text(2.0, 0.93, "step ≈ 0.7η:\npeak resolved", color=S.NAVY, fontsize=9.3, va="top",
        linespacing=1.2)
ax.text(2.0, 0.60, "step 3η:\npeak missed", color=S.RED, fontsize=9.3, va="top", linespacing=1.2)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 1.08)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
ax.set_xticklabels(["−6η", "−4η", "−2η", "E", "2η", "4η", "6η"])
ax.set_xlabel("frequency ω", labelpad=2)
ax.set_title("(b)  Frequency: a peak of width η", x=0.0, pad=22)
ax.text(0.0, 1.035, r"needs a step $\lesssim \eta$", transform=ax.transAxes, color=S.GREY,
        fontsize=9.3, va="bottom")

# ---- (c) time
T = np.log(1 / EPS)                          # horizon in 1/eta
Eb = 3.0                                     # band edge from mid-band: B = 6 eta
dt = 0.9 * 2 * np.pi / (2 * Eb)
tt = np.linspace(0, 11, 4000)
env = np.exp(-tt)
bx.axvspan(T, 11, color=S.PANEL, lw=0)
bx.plot(tt, env, color=S.IDLE, lw=1.0, ls=(0, (3, 2)))
bx.plot(tt, -env, color=S.IDLE, lw=1.0, ls=(0, (3, 2)))
bx.plot(tt, env * np.cos(Eb * tt), color=S.NAVY, lw=1.4)
ts = np.arange(0, T + 1e-9, dt)
bx.plot(ts, np.exp(-ts) * np.cos(Eb * ts), "o", ms=4.0, color=S.NAVY, zorder=4)
bx.axhline(0, color=S.IDLE, lw=0.6)
bx.axvline(T, color=S.RED, lw=1.0)
bx.text(T + 0.15, 0.80, "horizon\n" r"$T \approx \ln(1/\varepsilon)/\eta$", color=S.RED, fontsize=9.3,
        va="top", linespacing=1.3)
bx.text(1.25, 0.62, r"envelope $e^{-\eta t}$", color=S.GREY, fontsize=9.3)
bx.annotate("tiny, but above " r"$\varepsilon = 10^{-4}$:" "\nstill sampled", xy=(6.6, 0.015),
            xytext=(5.0, 0.42), color=S.GREY, fontsize=9.0, linespacing=1.25,
            arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
i0 = 2
bx.annotate("", xy=(ts[i0], -0.78), xytext=(ts[i0 + 1], -0.78),
            arrowprops=dict(arrowstyle="<->", color=S.NAVY, lw=0.9, shrinkA=0, shrinkB=0))
bx.text(ts[i0 + 1] + 0.2, -0.78, r"$\Delta t \leq 2\pi/B$: two samples per period of the" "\n"
        "fastest term, energies from mid-band", color=S.NAVY, fontsize=9.0, va="center",
        linespacing=1.25)
bx.set_xlim(0, 11)
bx.set_ylim(-1.02, 1.02)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xticks([0, 2, 4, 6, 8, 10])
bx.set_xticklabels(["0", "2/η", "4/η", "6/η", "8/η", "10/η"])
bx.set_xlabel("time t", labelpad=2)
bx.set_title("(c)  Time: the same pole, oscillating and decaying", x=0.0, pad=22)
bx.text(0.0, 1.035, r"needs $T/\Delta t$ samples: the same count", transform=bx.transAxes,
        color=S.GREY, fontsize=9.3, va="bottom")
S.save(fig, "sampling")
