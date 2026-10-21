"""Fig. 4.3: Sigma denominator rules, error against node count.

Boxes from fig_windows.py (eta = 0.25 eV): the whole conduction crossing
branch, Re d in [-356.6, 15.6] eV, and its resonant window, [-20.7, 9.0] eV.
Error currency: the crossing box's peak-relative sup eta |Q(d) - 1/d| on
Im d = eta (gw.sigma_box_plan / minimax.analytic_box, sigma_quadrature_eps).

LORRAX: minimax.analytic_box_rule(box, eps) (services/minimax/src/minimax/
analytic_box.py:336, read-only) at a ladder of eps; its node count and its
own certified sup error.
Generic: composite Gauss-Legendre in real time on [0, T] (order 16 panels),
the rule for smooth band-limited integrands; T and the panel count chosen to
minimise the error at each N.  This is what any uniform-resolution rule needs:
the Nyquist count of section 2.
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from lorrax_minimax import analytic_box_rule

ETA = 0.25
BOXES = {"whole branch": (-356.556, 15.556), "resonant window": (-20.721, 8.971)}


def sup_err(times, weights, lo, hi, n=24000):
    x = np.linspace(lo, hi, n)
    d = x + 1j * ETA
    out = 0.0
    for k in range(0, n, 3000):
        dd = d[k:k + 3000]
        Q = np.exp(1j * np.outer(dd, times)) @ weights
        out = max(out, float(np.max(ETA * np.abs(Q - 1.0 / dd))))
    return out


def gl_rule(T, panels, order=16):
    u, v = np.polynomial.legendre.leggauss(order)
    w = T / panels
    t = np.concatenate([k * w + 0.5 * w * (u + 1) for k in range(panels)])
    h = np.concatenate([0.5 * w * v for _ in range(panels)])
    return t, -1j * h          # 1/d = -i int_0^inf e^{itd} dt


def gl_best(lo, hi, N):
    panels = max(1, N // 16)
    best = None
    for L in np.linspace(2.0, 22.0, 41):          # T = L / eta
        t, w = gl_rule(L / ETA, panels)
        e = sup_err(t, w, lo, hi, n=8000)
        if best is None or e < best[0]:
            best = (e, L)
    t, w = gl_rule(best[1] / ETA, panels)
    return panels * 16, sup_err(t, w, lo, hi)


fig, ax = plt.subplots(figsize=(S.WIDTH, 4.1),
                       gridspec_kw=dict(left=0.085, right=0.985, top=0.72, bottom=0.135))
styles = {"whole branch": "-", "resonant window": (0, (4, 2))}
for name, (lo, hi) in BOXES.items():
    pts = []
    for eps in np.geomspace(3e-2, 1e-8, 14):
        r = analytic_box_rule((lo, hi, ETA, ETA), eps)
        pts.append((r.times.size, r.sup_error))
    pts = sorted(set(pts))
    n_l, e_l = zip(*pts)
    print("lorrax", name, pts)
    ax.plot(n_l, e_l, color=S.RED, ls=styles[name], marker="o", ms=3.4, lw=1.6,
            label=f"LORRAX rule, {name}")
    gen = [gl_best(lo, hi, N) for N in (32, 64, 128, 256, 512, 1024, 2048, 4096, 6144)]
    print("generic", name, gen)
    n_g, e_g = zip(*gen)
    ax.plot(n_g, np.maximum(e_g, 1e-12), color=S.GREY, ls=styles[name], marker="o", ms=3.0,
            lw=1.4, label=f"real-time Gauss–Legendre, {name}")
ax.axhline(1e-4, color=S.SOFT, lw=0.8, ls=":")
ax.text(640, 7e-5, r"sigma_quadrature_eps = $10^{-4}$", ha="left", va="top", color=S.SOFT,
        fontsize=9.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(25, 7000)
ax.set_ylim(1e-9, 3)
ax.set_xlabel("number of time nodes", labelpad=2)
ax.set_ylabel(r"$\sup\ \eta\,|Q(d) - 1/d|$")
ax.grid(True, which="major", color=S.RULE, lw=0.6)
ax.set_title("A window near resonance: a generic rule against LORRAX's rule", x=-0.06, pad=58)
ax.text(-0.06, 1.245, "box Re d in [−357, 16] eV (one window for the branch) and [−21, 9] eV (its resonant window), η = 0.25 eV",
        transform=ax.transAxes, color=S.GREY, fontsize=9.0, va="bottom")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.22), ncol=2, fontsize=8.8,
          handlelength=2.6, columnspacing=1.6)
ax.annotate("187 nodes at the default ε", xy=(188, 4.0e-5), xytext=(175, 1.5e-2), color=S.RED, fontsize=9.0,
            arrowprops=dict(arrowstyle="-", color=S.RED, lw=0.7))
ax.annotate("≈ 4500 nodes", xy=(4500, 1.0e-4), xytext=(2600, 3e-2), color=S.SOFT, fontsize=9.0,
            ha="right", arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.7))
S.save(fig, "sigma-error")
