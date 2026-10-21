"""Fig. 3.2: a rule built for the exponential family against generic rules.

1/x on the window x in [1, 100] eV (R = 100), errors in eV^-1 (= units of 1/x_min).
(a) error against Delta at N = 11 (the LORRAX static rule's count at the
    default target, see fig_exponentials.py);
(b) sup error against N for LORRAX's levelled rule (R = 100 and 10^4) and the
    best uniform and Gauss-Legendre rules (R = 100).
Rules: laplace_rules.py. Target line: minimax_target_error = 1e-6 Ry^-1
(src/gw/gw_config.py:1573) at Delta_min = 1 eV.
"""
import numpy as np
import matplotlib.pyplot as plt

import style as S
from laplace_rules import evaluate, gauss_best, minimax, uniform_best

RY = 13.605693122994
R, N0 = 100.0, 11

fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.WIDTH, 4.3),
                             gridspec_kw=dict(width_ratios=[1, 1.1], wspace=0.25,
                                              left=0.085, right=0.985, top=0.64, bottom=0.14))

# ---- (a) error against Delta at fixed N
x = np.geomspace(0.3, 300.0, 5000)
ax.axvspan(1, R, color=S.PANEL, lw=0, zorder=0)
rules = [("best uniform grid", uniform_best(R, N0), S.GREY, "-"),
         ("best Gauss–Legendre on [0, T]", gauss_best(R, N0), S.NAVY, (0, (4, 2))),
         ("minimax rule", minimax(R, N0), S.RED, "-")]
for label, (t, w, e), col, ls in rules:
    err = np.abs(evaluate(t, w, x) - 1.0 / x)
    ax.plot(x, np.maximum(err, 1e-14), color=col, ls=ls, lw=1.5, label=label)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.3, 300)
ax.set_ylim(1e-11, 30)
ax.set_xlabel("decay rate x  (eV)", labelpad=2)
ax.set_ylabel(r"error  (eV$^{-1}$)")
ax.set_title(f"(a)  {N0} nodes each: error against the decay rate", pad=10)
ax.text(1.12, 4e-6, "window", color=S.SOFT, fontsize=9.0, va="bottom")
ax.text(2.0, 1.5e-7, "minimax: equal peaks across the window", color=S.RED, fontsize=8.8,
        va="bottom", bbox=dict(fc=S.PANEL, ec="none", pad=0.8))
ax.annotate("slow decays: span [0, T] too short", xy=(1.0, 1.1e-2), xytext=(0.33, 6.0),
            color=S.SOFT, fontsize=8.8, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
ax.annotate("fast decays: nodes too far apart", xy=(99, 1.1e-2), xytext=(280, 0.5),
            color=S.SOFT, fontsize=8.8, ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=S.IDLE, lw=0.8))
ax.grid(True, which="major", color=S.RULE, lw=0.6)

# ---- (b) error against N
Ns_mm = list(range(2, 23))
Ns_gen = [2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024]


def mm_curve(RR):
    out = []
    for n in Ns_mm:
        e = minimax(RR, n)[2]
        out.append(e)
        if e < 1e-12:
            break
    return Ns_mm[:len(out)], out


uni = [uniform_best(R, n)[2] for n in Ns_gen]
gau = [gauss_best(R, n)[2] for n in Ns_gen]
n1, e1 = mm_curve(R)
n4, e4 = mm_curve(1e4)
bx.plot(Ns_gen, uni, color=S.GREY, lw=1.5, marker="o", ms=3.2, label="best uniform grid")
bx.plot(Ns_gen, gau, color=S.NAVY, lw=1.5, ls=(0, (4, 2)), marker="o", ms=3.2,
        label="best Gauss–Legendre")
bx.plot(n1, e1, color=S.RED, lw=1.7, marker="o", ms=3.4, label=r"minimax rule, window [1, 100] eV")
bx.plot(n4, e4, color=S.RED, lw=1.2, ls=(0, (2, 1.6)), marker="o", ms=2.8, alpha=0.75,
        label=r"minimax rule, window [1, $10^4$] eV")
tgt = 1e-6 / RY
bx.axhline(tgt, color=S.SOFT, lw=0.8, ls=":")
bx.text(1024, tgt / 1.8, r"$\varepsilon_\chi$ (default)", ha="right", va="top",
        linespacing=1.2, bbox=dict(fc="white", ec="none", pad=1.0), zorder=6,
        color=S.SOFT, fontsize=8.8)
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xlim(2, 1100)
bx.set_ylim(1e-13, 1)
bx.set_xlabel("number of nodes N", labelpad=2)
bx.set_ylabel(r"max error on window  (eV$^{-1}$)", labelpad=4)
bx.set_title("(b)  Error against node count", pad=10)
handles, labels = bx.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.895), ncol=2,
           fontsize=9.3, handlelength=2.4, columnspacing=1.8)
bx.grid(True, which="major", color=S.RULE, lw=0.6)
for a in (ax, bx):
    a.title.set_x(-0.08)
print("uniform", dict(zip(Ns_gen, ["%.1e" % v for v in uni])))
print("gauss", dict(zip(Ns_gen, ["%.1e" % v for v in gau])))
print("mm R=100", dict(zip(n1, ["%.1e" % v for v in e1])))
print("mm R=1e4", dict(zip(n4, ["%.1e" % v for v in e4])))
fig.text(0.015, 0.985, r"Approximating $1/x$ on the window $x \in [1, 100]$ eV unless stated; generic rules have their step $h$ or span $T$"
         "\n" r"chosen to minimise the error at each N; dotted line: $\varepsilon_\chi$ = minimax_target_error = $10^{-6}$ Ry$^{-1}$ = $7.3\times10^{-8}$ eV$^{-1}$",
         ha="left", va="top", color=S.SOFT, fontsize=9.6)
S.save(fig, "laplace-error")
