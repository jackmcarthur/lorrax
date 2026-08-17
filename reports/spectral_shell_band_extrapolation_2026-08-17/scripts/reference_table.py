#!/usr/bin/env python3
"""Held-out reference table: the SHIPPED estimators against a MEASURED S(508).

THE CORRECTNESS GATE FOR THE PORT.  ``S(508)`` is a number BerkeleyGW
computed on the 508-band Si 50 Ry arm, not a fit, so an estimator that
predicts it from N_max < 508 is being scored on data it never saw.  This
script drives ``gw.band_extrapolation`` ITSELF -- the shipped module, not a
copy of it -- over the same rungs, the same Fermi window and the same
degeneracy-snapped band counts as the 2026-08-16 prototype
(``sandbox:reports/band_tail_exponent_50ry_2026-08-16/scripts/``), and prints
the table the module docstring quotes.

Run it from a checkout with ``src`` on PYTHONPATH; it needs only numpy + h5py
and no GPU.
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))

#: The prototype's dataset binding: parsers, the degeneracy snapper and the
#: shipped counts-of-total rule all come from the 2026-08-15/16 study rather
#: than being re-derived here, so the rungs are the SAME rungs.
PROTO = ("/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/"
         "reports/band_tail_exponent_50ry_2026-08-16/scripts")
RUN = "/pscratch/sd/j/jackm/si_bandtail50_20260816"

TARGET = 508
NMAXES = (150, 200, 250, 300, 400)


def _stub_matplotlib():
    """Register an INERT ``matplotlib`` so the prototype binding imports.

    ``arms.py`` reaches ``make_plots`` for exactly two things -- the degeneracy
    snapper and ``counts_of_total``, the SHIPPED fraction rule -- and that
    module imports ``matplotlib``/``pyplot``/``lines`` AND configures
    ``plt.rcParams`` at file scope.  matplotlib is not in the LORRAX container.

    Stubbed rather than re-derived: re-deriving the snapper would turn this
    into a check of a SECOND implementation of the rungs rather than of the
    shipped one, which is the opposite of the point.  The stub is inert (every
    attribute access, call, index and mutation returns the same do-nothing
    object) BECAUSE the import path configures rcParams -- a stub that raised
    would fail at import rather than at first plot, and there is no plot here
    to reach.  Nothing this script or test asserts depends on matplotlib, so
    inert is safe; anything that DID need a figure would silently get nothing,
    which is why this helper is used only from these two entry points.
    """
    import sys
    import types

    if "matplotlib" in sys.modules:
        return

    class _Inert:
        def __getattr__(self, name):
            return self

        def __call__(self, *a, **k):
            return self

        def __getitem__(self, k):
            return self

        def __setitem__(self, k, v):
            pass

        def update(self, *a, **k):
            pass

        def __iter__(self):
            return iter(())

    _inert = _Inert()

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            return _inert

    for nm in ("matplotlib", "matplotlib.pyplot", "matplotlib.lines",
               "matplotlib.patches", "matplotlib.colors", "matplotlib.cm",
               "matplotlib.ticker", "matplotlib.gridspec"):
        m = _Mod(nm)
        m.__path__ = []              # a package, so submodule imports resolve
        m.__spec__ = None
        sys.modules[nm] = m
    for nm in ("pyplot", "lines", "patches", "colors", "cm", "ticker",
               "gridspec"):
        object.__setattr__(sys.modules["matplotlib"], nm,
                           sys.modules[f"matplotlib.{nm}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default=PROTO)
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--proto-windows", action="store_true",
                    help="use the prototype's ladder windows (50..NB for the "
                         "Weyl fit, bands 201..400 for E*) instead of the "
                         "module defaults, to show they agree")
    args = ap.parse_args()

    sys.path.insert(0, args.proto)
    _stub_matplotlib()
    from arms import Arm                                        # noqa: E402
    from predict508 import fit_model                            # noqa: E402

    from gw.band_extrapolation import (                         # noqa: E402
        build_band_ladder, fit_band_extrapolation_spectral, SHELL_OK)
    from common.units import RYD_TO_EV                          # noqa: E402
    import h5py                                                 # noqa: E402

    A = Arm("A", f"{args.run}/armA/ch_converge.dat",
            f"{args.run}/armA/sigma_hp.log", f"{args.run}/qe/WFN.h5",
            "ecutwfc 50, nbnd 512, W cutoff 25 Ry")
    with h5py.File(f"{args.run}/qe/WFN.h5", "r") as f:
        kw = np.asarray(f["mf_header/kpoints/w"][:], float)

    # THE LADDER IS DFT-ONLY.  ``A.EL`` is mf_header/kpoints/el in Ry -- the
    # mean field, never the three Sigma values.
    kw_windows = dict(fit_window=(50, A.EL.shape[1]),
                      estar_window=(201, 400)) if args.proto_windows else {}
    ladder = build_band_ladder(
        enk_ry=A.EL, kweights=kw, n_target=TARGET, **kw_windows)
    print(f"arm A: {A.NTB} bands, {len(A.FERMI)} Fermi-window states, "
          f"k-order {A.k_check:.1e} eV")
    print(f"{ladder.describe()}")
    print(f"target = MEASURED S({TARGET}); windows = "
          f"{'prototype' if args.proto_windows else 'module default'}\n")

    print(f'{"N_max":>6} | {"band_index_only":>15} {"spectral_shell":>15} | '
          f'{"beta med":>8}{"p10":>7}{"p90":>7} {"fail":>5}')
    print(f'{"":>6} | ' + " ".join(f'{"med / max meV":>15}' for _ in range(2)))
    rows = []
    for nmax in NMAXES:
        n = A.rung(nmax)["n"]
        counts = A.counts_of_total(n, (0.80, 0.90))
        keys = A.FERMI
        Sv = np.empty((3, len(keys)), dtype=np.float64)
        truth = np.empty(len(keys), dtype=np.float64)
        for j, k in enumerate(keys):
            N_int, S, _ = A.CUR[k]
            g = {int(q): S[i] for i, q in enumerate(N_int)}
            truth[j] = g[TARGET]
            Sv[:, j] = [g[int(c)] for c in counts]

        fit = fit_band_extrapolation_spectral(counts, Sv, ladder)
        e_sp = np.abs(np.real(fit.s_inf) - truth) * 1e3
        ok = np.asarray(fit.failure) == SHELL_OK
        e_1n = np.array([abs(fit_model(counts, Sv[:, j], 1.0) - truth[j]) * 1e3
                         for j in range(len(keys))])
        b = np.asarray(fit.beta)[ok]

        def ms(v):
            return (f"{np.nanmedian(v):>6.1f} / {np.nanmax(v):>6.1f}"
                    if len(v) else "           nan")
        print(f"{n:>6} | {ms(e_1n):>15} {ms(e_sp[ok]):>15} | "
              f"{np.median(b):>8.2f}{np.percentile(b, 10):>7.2f}"
              f"{np.percentile(b, 90):>7.2f} {int((~ok).sum()):>5}")
        rows.append((n, float(np.nanmedian(e_1n)), float(np.nanmedian(e_sp[ok])),
                     float(np.median(b))))

    # THE PUBLISHED TABLE, checked rather than eyeballed.
    expect = {152: (45.8, 4.7), 204: (29.7, 14.7), 260: (17.5, 12.5),
              296: (12.6, 0.7), 396: (3.5, 0.0)}
    bad = [r for r in rows
           if r[0] in expect and (abs(round(r[1], 1) - expect[r[0]][0]) > 0.05
                                  or abs(round(r[2], 1) - expect[r[0]][1]) > 0.05)]
    print("\nagainst the published table "
          "(gw.band_extrapolation module docstring): "
          + ("MATCH to the printed digit on all 5 rungs" if not bad
             else f"MISMATCH on {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
