"""Fig. 6.1 of docs/theory/shared-pole-w.md: where the shared-pole supports go.

Si, 4x4x4 k grid, all 537 bands of the 25 Ry basis (../time-frequency/data/
si_dense_scalar_bands.json, IBZ energies replicated by their weights to the full grid).
The supports are placed by the production recipe as it stands on LORRAX origin/main
152b92d0e (src/gw/shared_pole_recipe.py):
  height        h = max(2.6 eV, 4 eta)                                  :1287
  line count    strip law, clamped to [18 - n_im, 40 - n_im - n_held]   :1086-1105
  line sites    equal quantiles of rho^(1/2), rho the crossing density  :1163-1206 (copied below)
  imaginary     geomspace(u_min, u_max, n_im), u_min = max(4 eta, E_g),
                u_max = max(16 eV, omega_pl + 3.5 eV), n_im from kappa  :1057-1069, :1289-1314
  held          line midpoints nearest 25 % and 65 %, geometric midpoints of the
                first and last imaginary pairs                          :1341-1345
Assumed deck: eta = 0.25 eV, sigma_omega_step_ev = 0.25, requested states the 4
valence and 4 lowest conduction bands at every k (reads within +-5 eV of each).
The transition density is the q-averaged density of independent transitions
E_c(k') - E_v(k), shown only to locate the poles of W; it is not used by the recipe.
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "time-frequency"))
import style as S  # noqa: E402

RY = 13.605693122994
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "docs", "assets", "figures"))


def support_rule_line_sites(energies_ev, mu_ev, eta_ev, height_ev, reads_ev, count,
                            power=0.5, grid=4001):
    """Verbatim logic of shared_pole_recipe.support_rule_line_sites (origin/main :1163-1206)."""
    levels = np.asarray(energies_ev, dtype=np.float64).ravel() - mu_ev
    evaluation, delta = np.asarray(reads_ev, dtype=np.float64).ravel(), 0.01
    pairs = np.zeros(1)
    for side in (1.0, -1.0):
        e, eps = side * evaluation, side * levels
        e, eps = e[e > 0], eps[eps > 0]
        if not e.size:
            continue
        n = int(np.rint(e.max() / delta)) + 1
        he = np.bincount(np.rint(e / delta).astype(np.int64), minlength=n)
        hs = np.bincount(np.rint(eps[eps <= e.max()] / delta).astype(np.int64), minlength=n)[:n]
        lagged = np.correlate(he, hs, 'full')[n - 1:]
        pairs = np.pad(pairs, (0, max(0, n - pairs.size))) + np.pad(lagged, (0, max(0, pairs.size - n)))
    pairs[:int(np.floor(height_ev / delta)) + 1] = 0
    lags = np.flatnonzero(pairs)
    flat = max(float(np.max(np.abs(evaluation))) if evaluation.size else 0.0, 2.0 * float(height_ev))
    reach = float(lags[-1] * delta) if lags.size else flat
    x = np.linspace(0.0, reach, grid)
    rho = (eta_ev / math.pi * (pairs[lags] / ((x[:, None] - lags * delta) ** 2 + eta_ev ** 2)).sum(axis=1)
           ) if lags.size else np.ones(grid)
    dens = rho ** power
    lo = float(height_ev)
    for _ in range(200):
        k = x >= lo - 1e-12
        c = np.concatenate([[0.0], np.cumsum(np.diff(x[k]) * 0.5 * (dens[k][1:] + dens[k][:-1]))])
        sites = np.interp(np.linspace(0.0, c[-1], int(count)), c, x[k])
        lo, previous = max(float(height_ev), float(sites[1] - sites[0])), lo
        if abs(lo - previous) < 1e-10:
            break
    return sites, x, rho


# ---- Si bands on the full 4x4x4 grid
d = json.load(open(os.path.join(HERE, "..", "time-frequency", "data", "si_dense_scalar_bands.json")))
el = np.asarray(d["el_ry"])[0] * RY
mult = np.rint(np.asarray(d["w"]) * 64).astype(int)
full = np.repeat(el, mult, axis=0)                      # [64, 537]
nocc = int(np.asarray(d["ifmax"]).ravel()[0])
vbm, cbm = el[:, :nocc].max(), el[:, nocc:].min()
mu = 0.5 * (vbm + cbm)
gap = cbm - vbm

# ---- recipe
eta = 0.25
h = max(2.6, 4 * eta)
eps = 1.0e-3
step, pad = 0.25, 5.0
req = full[:, :8] - mu
offs = step * np.arange(-int(pad / step), int(pad / step) + 1)
reads = (req.ravel()[:, None] + offs[None, :]).ravel()
volume = 10.26 ** 3 / 4                                 # bohr^3, Si fcc cell
n_e = 8.0                                               # valence electrons per cell (active bands)
omega_pl = 2.0 * math.sqrt(4 * math.pi * n_e / volume) * RY   # eV, as recipe :1289
top = omega_pl + 3.5
u_min = max(4 * eta, gap)
u_max = max(16.0, top)
kappa = top / u_min
n_im = max(2, round(math.log(16 * kappa ** 2) * math.log(4 / eps) / (2 * math.pi ** 2)))
imag = np.geomspace(u_min, u_max, n_im)
held_imag = np.sqrt(imag[[0, -2]] * imag[[1, -1]])
n_held = 2 + np.unique(held_imag).size
reach = float(np.max(np.abs(reads)))
strip = math.ceil(max(reach - h, 0.0) * math.log(4 / eps) / (math.pi * h))
n_line = max(18 - n_im, min(strip, 40 - n_im - n_held))
line, xg, rho = support_rule_line_sites(full * 1.0, mu, eta, h, reads, n_line)
mids = 0.5 * (line[:-1] + line[1:])
held_line = mids[[int(np.argmin(abs(mids - (line[0] + f * (line[-1] - line[0]))))) for f in (0.25, 0.65)]]
print(f"gap {gap:.2f} eV, omega_pl {omega_pl:.2f} eV, h {h}, reach {reach:.2f} eV, strip {strip}, "
      f"n_line {n_line}, n_im {n_im}, kappa {kappa:.1f}")
print("line", np.round(line, 2))
print("imag", np.round(imag, 2), "held", np.round(held_line, 2), np.round(held_imag, 2))

# ---- q-averaged density of independent transitions (location of the poles of W)
ev = full[:, :nocc].ravel()
ec = full[:, nocc:].ravel()
XMAX = 26.0
ec = ec[ec - vbm < XMAX + 2]
dd = (ec[None, :] - ev[:, None]).ravel()
bins = np.linspace(0, XMAX, 521)
hist, _ = np.histogram(dd, bins=bins)
xc = 0.5 * (bins[1:] + bins[:-1])
kern = np.exp(-0.5 * (np.arange(-15, 16) * (bins[1] - bins[0]) / 0.25) ** 2)
jd = np.convolve(hist, kern / kern.sum(), mode="same")
jd = jd / jd.max()

# ---- figure
fig = plt.figure(figsize=(S.WIDTH, 6.6))
gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1], hspace=0.32, left=0.075, right=0.975, top=0.93, bottom=0.085)
ax = fig.add_subplot(gs[0])
bx = fig.add_subplot(gs[1], sharex=ax)

JD_DEPTH = 2.7
ax.fill_between(xc, 0, -JD_DEPTH * jd, color=S.NAVY, alpha=0.16, lw=0)
ax.plot(xc, -JD_DEPTH * jd, color=S.NAVY, lw=0.8, alpha=0.6)
ax.text(0.5, -4.0, "density of independent transitions $E_c-E_v$, which the poles of $W^c$ follow apart from the plasmon",
        color=S.SOFT, fontsize=8.8, ha="left", va="center")
ax.axhline(0, color=S.NAVY, lw=0.9)
ax.axvline(0, color=S.NAVY, lw=0.9)
# analyticity strip of half-width h about the line
ax.axhspan(0, 2 * h, color=S.PANEL, zorder=0, lw=0)
ax.plot([0, XMAX], [h, h], color=S.RED, lw=0.9, ls=(0, (4, 3)), alpha=0.7, zorder=1)
dens = rho ** 0.5
dens = dens / dens.max()
# supports
ax.plot(line, np.full_like(line, h), "o", ms=6.2, color=S.RED, zorder=4, label="supports in the pencil")
ax.plot(held_line, np.full_like(held_line, h), "o", ms=6.2, mfc="white", mec=S.RED, mew=1.4, zorder=4,
        label="held supports (check only)")
ax.plot(np.zeros_like(imag), imag, "o", ms=6.2, color=S.RED, zorder=4, clip_on=False)
ax.plot(np.zeros_like(held_imag), held_imag, "o", ms=6.2, mfc="white", mec=S.RED, mew=1.4, zorder=4, clip_on=False)
ax.text(0.45, imag[-1], r"$iu_{\max}=i(\omega_{\mathrm{pl}}+3.5\,\mathrm{eV})$", color=S.SOFT, fontsize=9.0, va="center")
ax.annotate(r"$iu_{\min}=i\max(E_g,4\eta)$", xy=(0.15, imag[0]), xytext=(0.9, 1.0), color=S.SOFT, fontsize=9.0,
            va="center", arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.7, shrinkA=1, shrinkB=4))
ax.text(0.45, 9.6, "imaginary ladder,\nlog-spaced", color=S.RED, fontsize=9.3, va="center", linespacing=1.2)
ax.text(line[-1] + 0.4, h + 0.08, r"line $z=\omega+ih$", color=S.RED, fontsize=9.3, va="bottom")
ax.annotate("", xy=(XMAX - 0.6, h), xytext=(XMAX - 0.6, 0),
            arrowprops=dict(arrowstyle="<->", color=S.NAVY, lw=0.8, shrinkA=0, shrinkB=0))
ax.text(XMAX - 0.85, h / 2, "$h$", color=S.NAVY, fontsize=10, ha="right", va="center")
ax.text(XMAX - 0.3, 2 * h + 0.15, "strip of half-width $h$ about the line", color=S.GREY, fontsize=8.8, ha="right", va="bottom")
# the point at infinity: a short arrow out of the top-right corner
arr = FancyArrowPatch((XMAX - 3.4, 16.4), (XMAX - 0.3, 20.9), arrowstyle="-|>", mutation_scale=15,
                      color=S.NAVY, lw=1.4, zorder=5, clip_on=False)
ax.add_patch(arr)
ax.text(XMAX - 0.1, 21.0, "∞", color=S.NAVY, fontsize=15, ha="left", va="bottom", clip_on=False)
ax.text(XMAX - 3.7, 16.0, "$z\\to\\infty$: exact moments\n$2M_1/z^2+2M_3/z^4$ from band sums",
        color=S.NAVY, fontsize=9.2, ha="right", va="center", linespacing=1.25)
ax.axvline(omega_pl, ymin=0.0, ymax=1.0, color=S.IDLE, lw=0.7, ls=(0, (2, 2)), zorder=0)
ax.text(omega_pl + 0.2, 20.6, r"$\omega_{\mathrm{pl}}$", color=S.GREY, fontsize=9.3, va="bottom")
ax.set_xlim(0, XMAX)
ax.set_ylim(-4.6, 21.5)
ax.set_ylabel("Im $z$  (eV)")
ax.set_yticks([0, 5, 10, 15, 20])
ax.legend(loc="upper right", fontsize=9.0, handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.66))
ax.set_title(f"(a)  Supports for Si: {n_line} (+2 held) on the line, {n_im} (+2 held) on the imaginary axis, one at infinity", x=-0.05)
plt.setp(ax.get_xticklabels(), visible=False)

# (b) equal quantiles of g^(1/2)
bx.plot(xg, rho / rho.max(), color=S.GREY, lw=0.9, label=r"$g(\omega)$")
bx.plot(xg, dens, color=S.RED, lw=1.4, label=r"$g(\omega)^{1/2}$")
for i in range(len(line) - 1):
    sl = (xg >= line[i]) & (xg <= line[i + 1])
    bx.fill_between(xg[sl], 0, dens[sl], color=S.RED, alpha=0.16 if i % 2 == 0 else 0.07, lw=0)
for x0 in line:
    bx.axvline(x0, ymax=0.08, color=S.RED, lw=1.2)
bx.axvspan(0, h, color=S.PANEL, lw=0)
bx.text(h / 2, 0.88, "below $h$:\nno sites", color=S.GREY, fontsize=8.6, ha="center", va="top", linespacing=1.15)
bx.set_ylim(0, 1.08)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xlabel("Re $z$  (eV)", labelpad=2)
bx.legend(loc="upper right", fontsize=9.0, ncol=2, title="each scaled to unit peak", title_fontsize=8.6)
bx.set_title(r"(b)  Line sites at equal areas of $g^{1/2}$, where $g$ counts the crossings $|\omega-E_m|$ that Σ meets", x=-0.05)

path = os.path.join(OUT, "theory-sharedpole-supports.png")
fig.savefig(path, dpi=S.DPI, facecolor="white")
print("wrote", path)
