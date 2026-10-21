"""Fig. 1 of docs/theory/w-range-splitting.md: W(q) on a line through Gamma.

A schematic in arbitrary units, not a measurement.  A one-dimensional cut
through a bulk insulator: W(q) is a smooth short-range part plus the screened
plane-wave channels A g(K)/K^2, K = q + G, of which the one nearest the origin
(the Wigner-Seitz slot G_ws(q)) carries the 1/q^2 rise at Gamma.

What follows the code (LORRAX main 7e4999976):
  * the window w(|q|): 1 below r_a = q_ins/3, quintic smoothstep 1 - x^3(10 - 15x + 6x^2)
    to 0 at r_b = 0.9 q_ins (src/gw/lr_channels.py:248-260); here q_ins = |b|/2;
  * srlr: X = W - w LR on the coarse q (Gamma: W without its head, LR(0) = 0 in bulk),
    Fourier interpolation of X, w LR added back at each fine q
    (src/bse/bse_densify.py:488-609, 1566-1590);
  * c1: the coarse tiles interpolated with only the Gamma head held back, the head
    v/eps re-attached at fine q inside the coarse Gamma cell, with half weight on its
    edge and the projector frozen at Gamma (src/gw/head_densify.py:603-759, 839-888);
  * the Fourier interpolation: the coarse samples' Fourier series in R, the
    R = +-N_c/2 coefficient split in halves (one centroid pair; src/bse/bse_densify.py:54-105).
Simplification: the srlr model is the exact channel term here; LORRAX fits it in q.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "time-frequency"))
import style as S  # noqa: E402  (palette, fonts, rcParams)

OUT = os.path.join(S.OUT, "theory-wrange-split.png")

A = 0.04              # strength of the screened channel, A g(K)/K^2
SIG = 0.8             # decay of the channel form factor g(K) = exp(-K^2/SIG^2)
N_C, N_F = 8, 32      # coarse and fine points per reciprocal-lattice period
Q_INS = 0.5
R_A, R_B = Q_INS / 3.0, 0.9 * Q_INS
YMAX = 6.0


def g(K):
    return np.exp(-(K / SIG) ** 2)


def j0(q):            # the short-range part proper, smooth and periodic
    return 0.9 + 0.3 * np.cos(2 * np.pi * q) + 0.06 * np.cos(4 * np.pi * q)


def k_ws(q):          # q + G_ws(q), the shortest image
    return q - np.round(q)


def channel(q, n):    # the plane-wave channel q + G with G = n|b| (n relative to G_ws)
    K = k_ws(q) + n
    with np.errstate(divide="ignore"):
        return np.where(K == 0, 0.0, A * g(K) / np.where(K == 0, 1.0, K) ** 2)


def lr(q):            # the long-range channel, the WS slot only
    return channel(q, 0)


def w_true(q):        # every channel: smooth across the zone edge, 1/q^2 at Gamma
    return j0(q) + sum(channel(q, n) for n in range(-4, 5))


def window(qn):
    x = np.clip((np.abs(qn) - R_A) / (R_B - R_A), 0.0, 1.0)
    return 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def trig_interp(samples, q):
    """Fourier series of N_C equally spaced samples, R = +-N_C/2 split in halves."""
    n = len(samples)
    c = np.fft.fft(samples) / n                      # c[m] multiplies exp(2 pi i m q)
    m = np.fft.fftfreq(n, d=1.0 / n)
    out = np.zeros_like(q, dtype=complex)
    for cm, mm in zip(c, m):
        if abs(abs(mm) - n / 2) < 1e-9:
            out += cm * np.cos(2 * np.pi * (n / 2) * q)
        else:
            out += cm * np.exp(2j * np.pi * mm * q)
    return out.real


qc = np.arange(N_C) / N_C
qc = np.where(qc > 0.5, qc - 1.0, qc)               # (-1/2, 1/2]
qf = np.arange(N_F) / N_F
qf = np.where(qf > 0.5, qf - 1.0, qf)
qd = np.linspace(-0.5, 0.5, 2401)

# Coarse tiles.  Gamma carries no head (deferred by the loader); elsewhere the full W.
Wc = np.where(qc == 0, w_true(qc) - 0.0, w_true(qc))
Wc[qc == 0] = j0(0.0) + sum(channel(0.0, n) for n in range(-4, 5) if n != 0)

# srlr: subtract w LR on the coarse grid, interpolate, add w LR back on the fine grid.
X_c = Wc - window(qc) * lr(qc)
X_c[qc == 0] = Wc[qc == 0]                          # LR(0) = 0: the G = 0 slot has v = 0 there
srlr_f = trig_interp(X_c, qf) + window(qf) * lr(qf)
# c1: interpolate the coarse tiles as they are, re-attach the head inside the coarse Gamma cell.
half = 0.5 / N_C
wt = np.where(np.abs(qf) < half - 1e-12, 1.0, np.where(np.abs(np.abs(qf) - half) < 1e-12, 0.5, 0.0))
with np.errstate(divide="ignore"):
    head = np.where(qf == 0, 0.0, A * g(0.0) / np.where(qf == 0, 1.0, qf) ** 2)
c1_f = trig_interp(Wc, qf) + wt * head
truth_f = w_true(qf)

off = qd != 0
fig, (ax, bx) = plt.subplots(
    2, 1, figsize=(S.WIDTH, 7.0), sharex=True,
    gridspec_kw=dict(left=0.085, right=0.925, top=0.905, bottom=0.075, hspace=0.42,
                     height_ratios=[1.0, 1.0]))

# ---- (a) the split
for x in (-R_B, -R_A, R_A, R_B):
    ax.axvline(x, color=S.IDLE, lw=0.6, ls=":")
ax.plot(qd[off], w_true(qd[off]), color=S.NAVY, lw=1.6, label=r"$W_\mathbf{q}$, the full interaction")
ax.plot(qd[off], (window(qd) * lr(qd))[off], color=S.RED, lw=1.6,
        label=r"$w(|\mathbf{q}|)\,W^{\mathrm{LR}}_\mathbf{q}$, the windowed long-range part")
X_d = w_true(qd) - window(qd) * lr(qd)
X_d[qd == 0] = Wc[qc == 0][0]
ax.plot(qd, X_d, color=S.BLUE, lw=1.6, ls=(0, (5, 2)),
        label=r"$W^{\mathrm{SR}}_\mathbf{q}=W_\mathbf{q}-w\,W^{\mathrm{LR}}_\mathbf{q}$, what is interpolated")
ax.plot(qc, X_c, "o", ms=6.0, mfc="white", mec=S.BLUE, mew=1.3, zorder=5,
        label=r"$W^{\mathrm{SR}}$ on the coarse grid $\mathbf{q}_{\mathrm{co}}$")
ax2 = ax.twinx()
ax2.fill_between(qd, 0, window(qd), color=S.PANEL, zorder=0)
ax2.plot(qd, window(qd), color=S.GREY, lw=1.0)
ax2.set_ylim(0, 1.0 * YMAX / 2.2)
ax2.set_yticks([0, 1])
ax2.set_ylabel(r"$w(|\mathbf{q}|)$", color=S.GREY, labelpad=0)
ax2.tick_params(axis="y", colors=S.GREY)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(S.GREY)
ax2.spines["top"].set_visible(False)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set_ylim(0, YMAX)
ax.set_ylabel("W  (arbitrary units)")
for x, t, ha, dx in ((R_A, r"$r_a=q_{\mathrm{ins}}/3$", "left", 0.006), (R_B, r"$r_b=0.9\,q_{\mathrm{ins}}$", "right", -0.006)):
    ax.text(x + dx, YMAX * 0.97, t, color=S.GREY, fontsize=8.8, va="top", ha=ha)
ax.text(0.012, YMAX * 0.97, "1/q² rise", color=S.RED, fontsize=8.8, va="top")
ax.set_title("The split of W along a line through Γ", x=-0.08, pad=24)
ax.text(-0.08, 1.03, "the long-range channel is switched off inside the zone, so the remainder is smooth everywhere",
        transform=ax.transAxes, color=S.GREY, fontsize=9.0, va="bottom")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=8.6, handlelength=2.2, borderaxespad=0.2)

# ---- (b) what each route puts on the fine grid
for x in (-half, half):
    bx.axvline(x, color=S.IDLE, lw=0.6, ls=":")
bx.text(0.0, 0.12, r"coarse Γ cell $\Gamma_c$", color=S.GREY, fontsize=8.8, va="bottom", ha="center")
bx.plot(qd[off], w_true(qd[off]), color=S.NAVY, lw=1.3, label="exact W")
bx.plot(qd, trig_interp(Wc, qd), color=S.IDLE, lw=1.0, ls=(0, (3, 2)),
        label="Fourier interpolation of the coarse values")


def marks(ax_, q, y, top=0.985, **kw):
    inside = y <= YMAX
    ax_.plot(q[inside], y[inside], ls="none", **kw)
    kw2 = dict(kw)
    kw2.pop("label", None)
    kw2["marker"] = "^"
    ax_.plot(q[~inside], np.full((~inside).sum(), YMAX * top), ls="none", clip_on=False, **kw2)


sel = qf != 0
marks(bx, qf[sel], c1_f[sel], top=0.92, marker="s", ms=5.0, mfc=S.GREY, mec=S.GREY, label=r"c1 on the fine grid $\mathbf{q}_{\mathrm{fi}}$")
marks(bx, qf[sel], srlr_f[sel], marker="o", ms=5.0, mfc="white", mec=S.RED, mew=1.3,
      label=r"srlr on the fine grid $\mathbf{q}_{\mathrm{fi}}$")
bx.plot(qc[qc != 0], Wc[qc != 0], "o", ms=7.5, mfc="none", mec=S.NAVY, mew=1.0, label="coarse samples")
bx.set_ylim(0, YMAX)
bx.set_xlim(-0.5, 0.5)
bx.set_ylabel("W  (arbitrary units)")
bx.set_xlabel(r"$\mathbf{q}$ along a line through Γ  (units of $|\mathbf{b}|$; zone edge at ±0.5)", labelpad=2)
bx.set_title("Coarse to fine: the two routes", x=-0.08, pad=24)
bx.text(-0.08, 1.03,
        f"{N_C} coarse and {N_F} fine points per period; triangles mark fine values above the axis; Γ itself takes the c1 cell average on both routes",
        transform=bx.transAxes, color=S.GREY, fontsize=9.0, va="bottom")
bx.legend(loc="upper left", fontsize=8.6, handlelength=2.0, borderaxespad=0.2)

err_c1 = np.max(np.abs(c1_f[sel] - truth_f[sel]) / truth_f[sel])
err_sr = np.max(np.abs(srlr_f[sel] - truth_f[sel]) / truth_f[sel])
print(f"max relative error off Gamma: c1 {err_c1:.3f}, srlr {err_sr:.2e}")
fig.savefig(OUT, dpi=S.DPI, facecolor="white")
print("wrote", OUT)
