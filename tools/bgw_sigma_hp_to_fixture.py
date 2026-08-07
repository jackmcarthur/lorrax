#!/usr/bin/env python3
"""Turn a BerkeleyGW ``sigma_hp.log`` into the committed anchor fixture, and
diagnose a LORRAX run against it.

This is how ``tests/regression/si_cohsex_debug/bgw_sigma_hp_noavg.dat`` is
made.  It exists so the anchor is REGENERABLE from BerkeleyGW output rather
than being a magic file nobody can rebuild.  It only ever reads BGW logs —
there is deliberately no path in this tool that can fill the fixture from a
LORRAX run.

    # rebuild the committed fixture
    python3 tools/bgw_sigma_hp_to_fixture.py write \\
        <bgwrun>/sigma_hp.log <bgwrun>/sigma.inp \\
        tests/regression/si_cohsex_debug/bgw_sigma_hp_noavg.dat

    # diagnose a LORRAX run (per-k table + IBZ/full-BZ summaries)
    python3 tools/bgw_sigma_hp_to_fixture.py compare \\
        tests/regression/si_cohsex_debug/bgw_sigma_hp_noavg.dat \\
        <rundir>/eqp_si_test.dat

    # isolate bare exchange (needs sigma_freq_debug_output = true)
    python3 tools/bgw_sigma_hp_to_fixture.py comparex \\
        tests/regression/si_cohsex_debug/bgw_sigma_hp_noavg.dat \\
        <rundir>/sigma_freq_debug.dat

The committed fixture for Si 4x4x4 came from

    /pscratch/sd/j/jackm/lorrax_sandbox_pre_august/runs/Si/
        06_si_4x4x4_nosoc/D_bgw_cohsex_noavg/

``compare`` is the diagnostic; the pytest gate uses ``harness.compare_to_bgw``,
which implements the same mapping and is checked against this tool's output.

WHY THE PRIMED COLUMNS.  BerkeleyGW's 14-column sigma_hp.log block is written by
Sigma/write_result_hp.f90:88-100.  The literal argument list is:

    ... dble(ach(2,i,ispin)+achcor(i,ispin)),      <- column "CH"
        dble(asig(i,ispin)+achcor(i,ispin)),       <- column "Sig"
        ...
        dble(ach(2,i,ispin)), dble(asig(i,ispin)), <- columns "CH`", "Sig`"

`achcor` is the STATIC REMAINDER correction.  LORRAX does not compute a static
remainder, so the comparable BGW columns are the PRIMED ones:

    LORRAX sigSX   <->  BGW  X + (SX-X)
    LORRAX sigCOH  <->  BGW  CH`          (NOT CH: CH = CH` + achcor)
    LORRAX sigTOT  <->  BGW  Sig`         (= X + (SX-X) + CH`, exactly)

Usage:
    bgw_extract.py write  <sigma_hp.log> <sigma.inp> <out.dat>
    bgw_extract.py compare <fixture.dat> <lorrax_eqp.dat> [nband_limit]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

# sigma_hp.log 14-column order after the band index
COLS = ["Emf", "Eo", "X", "SXmX", "CH", "Sig", "KIH", "Eqp0", "Eqp1",
        "CHp", "Sigp", "Eqp0p", "Eqp1p", "Znk"]

_K_RE = re.compile(
    r"^\s*k =\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+ik =\s*(\d+)\s+spin =\s*(\d+)")


def parse_hp_log(path: Path):
    """-> (kfrac (nk,3), bands (nb,), data dict col -> (nk, nb) array)."""
    lines = Path(path).read_text().splitlines()
    kfrac, blocks = [], []
    i = 0
    while i < len(lines):
        m = _K_RE.match(lines[i])
        if not m:
            i += 1
            continue
        kfrac.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
        i += 1
        # skip to the numeric rows (past the blank + column-header line)
        rows = []
        while i < len(lines):
            s = lines[i].split()
            if len(s) == 15 and s[0].isdigit():
                rows.append([float(x) for x in s])
            elif rows:
                break
            i += 1
        assert rows, f"no data rows after k-block at line {i}"
        blocks.append(np.asarray(rows, dtype=np.float64))
    assert blocks, f"no k blocks parsed from {path}"
    shapes = {b.shape for b in blocks}
    assert len(shapes) == 1, f"ragged k blocks: {shapes}"
    arr = np.stack(blocks)                       # (nk, nb, 15)
    bands = arr[0, :, 0].astype(int)
    assert np.all(arr[:, :, 0].astype(int) == bands[None, :]), \
        "band index differs between k blocks"
    data = {c: arr[:, :, j + 1] for j, c in enumerate(COLS)}
    return np.asarray(kfrac), bands, data


def _inp_scalars(inp_path: Path) -> dict:
    txt = Path(inp_path).read_text()
    out = {}
    for key in ("number_bands", "band_index_min", "band_index_max",
                "screened_coulomb_cutoff", "frequency_dependence",
                "exact_static_ch", "cell_average_cutoff"):
        m = re.search(rf"^\s*{key}\s+(\S+)", txt, re.M)
        if m:
            out[key] = m.group(1)
    for flag in ("degeneracy_check_override", "use_wfn_hdf5", "use_kihdat",
                 "dont_use_vxcdat", "write_vcoul"):
        if re.search(rf"^\s*{flag}\s*$", txt, re.M):
            out[flag] = "(set)"
    return out


HEADER = """\
# BerkeleyGW COHSEX reference columns — bulk Si, 4x4x4, nosoc, 8 IBZ k-points.
#
# THIS FILE IS BERKELEYGW OUTPUT, NOT LORRAX OUTPUT.  It is the EXTERNAL anchor
# for the si_cohsex production gate.  Every number below was produced by
# BerkeleyGW Sigma and transcribed verbatim from its sigma_hp.log; nothing here
# was computed by LORRAX.  If you ever need to regenerate it, rerun BGW — do not
# fill it from a LORRAX run.
#
# PROVENANCE
#   source log   {log}
#   source input {inp}
#   log mtime    {mtime}
#   log sha256   {sha}
#   extracted    {when} by tools/bgw_sigma_hp_to_fixture.py
#
# BGW RUN CONFIGURATION (from sigma.inp)
{cfg}
#
# `cell_average_cutoff 1.0d-12` is the load-bearing one: under it BGW cell-
# averages ONLY the literal q+G=0 element and uses the point 8*pi/|q+G|^2
# everywhere else (Common/vcoul_generator.f90:101-103).  The LORRAX deck must
# therefore set `mc_average_vcoul_body = false`; with the LORRAX default (true)
# sigTOT moves 141.65 meV.  This pairing is the whole reason the fixture agrees.
#
# COLUMN CONVENTION — READ THIS BEFORE COMPARING ANYTHING
#   BGW's 14-column sigma_hp.log block (Sigma/write_result_hp.f90:88-100) writes
#       CH   = ach + achcor        Sig   = asig + achcor
#       CH`  = ach                 Sig`  = asig
#   where `achcor` is the STATIC REMAINDER correction.  LORRAX does not compute
#   a static remainder, so the comparable columns are the PRIMED ones.  The
#   mapping used by the gate, with NO offset applied to either side:
#
#       LORRAX sigSX   ==  X + SXmX
#       LORRAX sigCOH  ==  CHp                (NOT CH — they differ by ~367 meV)
#       LORRAX sigTOT  ==  Sigp               (== X + SXmX + CHp, identically)
#
#   Emf/Eo/KIH/Eqp*/Znk are carried for context and are NOT gated: Eqp0 also
#   contains kin_ion, which LORRAX does not degeneracy-symmetrise where BGW
#   does (an input-artifact difference worth ~18 meV of within-multiplet
#   spread, tracked separately).
#
# UNITS: eV throughout.  k in fractional (crystal) coordinates.
#
# Columns: ik  kx ky kz  n  {cols}
"""


def cmd_write(log_path, inp_path, out_path):
    import datetime
    import hashlib

    log_path, inp_path, out_path = map(Path, (log_path, inp_path, out_path))
    kfrac, bands, data = parse_hp_log(log_path)
    nk, nb = kfrac.shape[0], bands.size
    sha = hashlib.sha256(log_path.read_bytes()).hexdigest()
    mtime = datetime.datetime.utcfromtimestamp(
        log_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S UTC")
    cfg = _inp_scalars(inp_path)
    cfg_txt = "\n".join(f"#   {k:<26s} {v}" for k, v in cfg.items())

    # Internal consistency of the file we are about to freeze: Sig` must equal
    # X + (SX-X) + CH` to BGW's own printed precision.  If this fails the parse
    # is wrong (column drift) and the fixture must not be written.
    resid = np.abs(data["X"] + data["SXmX"] + data["CHp"] - data["Sigp"])
    assert resid.max() < 5e-9, (
        f"parsed columns are not self-consistent: max|X+SXmX+CHp-Sigp| = "
        f"{resid.max():.3e} eV — column order is wrong, refusing to write")

    lines = [HEADER.format(
        log=log_path, inp=inp_path, mtime=mtime, sha=sha,
        when=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        cfg=cfg_txt, cols="  ".join(f"{c:>14s}" for c in COLS))]
    lines.append(f"# nk = {nk}   nbands = {nb}   "
                 f"band_index {bands.min()}..{bands.max()}")
    lines.append(f"# self-check: max|X + SXmX + CHp - Sigp| = {resid.max():.3e} eV")
    for ik in range(nk):
        for j, n in enumerate(bands):
            row = "  ".join(f"{data[c][ik, j]:15.9f}" for c in COLS)
            lines.append(
                f"{ik+1:4d} {kfrac[ik,0]:11.6f} {kfrac[ik,1]:11.6f} "
                f"{kfrac[ik,2]:11.6f} {n:4d}  {row}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}  ({nk} k x {nb} bands = {nk*nb} rows)")
    print(f"self-check max|X+SXmX+CHp-Sigp| = {resid.max():.3e} eV")


def read_fixture(path: Path):
    kfrac, bands, cols = [], [], {c: [] for c in COLS}
    rows = []
    for ln in Path(path).read_text().splitlines():
        if ln.lstrip().startswith("#") or not ln.strip():
            continue
        s = ln.split()
        rows.append(s)
    nk = max(int(s[0]) for s in rows)
    bset = sorted({int(s[4]) for s in rows})
    nb = len(bset)
    kf = np.zeros((nk, 3))
    data = {c: np.zeros((nk, nb)) for c in COLS}
    bidx = {b: j for j, b in enumerate(bset)}
    for s in rows:
        ik = int(s[0]) - 1
        kf[ik] = [float(s[1]), float(s[2]), float(s[3])]
        j = bidx[int(s[4])]
        for c, v in zip(COLS, s[5:]):
            data[c][ik, j] = float(v)
    return kf, np.asarray(bset), data


_LX_RE = re.compile(
    r"n=\s*(\d+)\s+sigSX=\s*([-\d.eE+]+)\s+sigCOH=\s*([-\d.eE+]+)\s+"
    r"sigTOT=\s*([-\d.eE+]+)\s+VH=\s*([-\d.eE+]+)\s+Eo=\s*([-\d.eE+]+)")


def read_lorrax(path: Path):
    """-> dict ik -> (nb,5) array of [sigSX, sigCOH, sigTOT, VH, Eo]."""
    out, ik = {}, -1
    for ln in Path(path).read_text().splitlines():
        m = re.search(r"k-point\s+(\d+)\s*:", ln)
        if m:
            ik = int(m.group(1))
            out[ik] = []
            continue
        m = _LX_RE.search(ln)
        if m:
            out[ik].append([float(m.group(i)) for i in (2, 3, 4, 5, 6)])
    return {k: np.asarray(v) for k, v in out.items() if v}


def cmd_compare(fixture, lorrax_dat, nband_limit=None):
    kf, bands, bgw = read_fixture(Path(fixture))
    lx = read_lorrax(Path(lorrax_dat))
    nb = bands.size if nband_limit is None else min(bands.size, int(nband_limit))
    print(f"BGW fixture : {kf.shape[0]} IBZ k x {bands.size} bands  "
          f"(comparing first {nb})")
    print(f"LORRAX file : {len(lx)} k-points x "
          f"{len(next(iter(lx.values())))} bands")

    # Map each BGW IBZ k onto every LORRAX k whose mean-field Eo column matches.
    # Matching on the full Eo vector (not on k coords) is deliberate: LORRAX
    # writes the FULL BZ, its k ordering is not BGW's, and every star member of
    # an IBZ k must carry the same Sigma — so we check them all and report the
    # worst.  A star with a broken symmetry unfold shows up here as a spread.
    tot = {c: [] for c in ("sigSX", "sigCOH", "sigTOT")}
    ibz = {c: [] for c in ("sigSX", "sigCOH", "sigTOT")}
    per_k = []
    for ik in range(kf.shape[0]):
        emf = bgw["Emf"][ik, :nb]
        hits = [k for k, v in lx.items()
                if v.shape[0] >= nb and np.max(np.abs(v[:nb, 4] - emf)) < 2e-3]
        if not hits:
            best = min(((np.max(np.abs(v[:nb, 4] - emf)), k)
                        for k, v in lx.items() if v.shape[0] >= nb))
            print(f"  k{ik+1} {kf[ik]}: NO LORRAX MATCH (best Eo dev "
                  f"{best[0]*1e3:.3f} meV at k{best[1]})")
            continue
        ref = {"sigSX": bgw["X"][ik, :nb] + bgw["SXmX"][ik, :nb],
               "sigCOH": bgw["CHp"][ik, :nb],
               "sigTOT": bgw["Sigp"][ik, :nb]}
        row = {"ik": ik + 1, "k": kf[ik], "nstar": len(hits)}
        for j, name in enumerate(("sigSX", "sigCOH", "sigTOT")):
            d = np.concatenate([lx[k][:nb, j] - ref[name] for k in hits])
            tot[name].append(d)
            row[name] = (np.abs(d).mean() * 1e3, np.abs(d).max() * 1e3)
        # Star spread: do all symmetry-equivalent LORRAX k agree with EACH OTHER,
        # band by band?  Must be taken PER BAND — sigTOT ranges over several eV
        # across bands, so a spread taken over the whole (star, band) block just
        # measures the bandwidth and is meaningless.
        star = np.stack([lx[k][:nb, 2] for k in hits])       # (nstar, nb)
        row["star_spread"] = float((star.max(0) - star.min(0)).max()) * 1e3 \
            if len(hits) > 1 else 0.0
        # IBZ-only view: one representative LORRAX k per IBZ k (the lowest
        # index), which is what a "128 (k,band) pairs" comparison means.
        rep = min(hits)
        for j, name in enumerate(("sigSX", "sigCOH", "sigTOT")):
            ibz[name].append(lx[rep][:nb, j] - ref[name])
        per_k.append(row)

    print()
    print(f"{'ik':>3} {'kfrac':>26} {'star':>5} "
          f"{'sigSX MAE/max':>20} {'sigCOH MAE/max':>20} {'sigTOT MAE/max':>20}"
          f" {'starspread':>11}")
    for r in per_k:
        ks = f"[{r['k'][0]:+.3f} {r['k'][1]:+.3f} {r['k'][2]:+.3f}]"
        print(f"{r['ik']:>3} {ks:>26} {r['nstar']:>5} "
              + " ".join(f"{r[c][0]:9.4f}/{r[c][1]:9.4f}"
                         for c in ("sigSX", "sigCOH", "sigTOT"))
              + f" {r['star_spread']:11.5f}")
    for title, acc in (("IBZ ONLY — one representative LORRAX k per IBZ k "
                        f"({kf.shape[0]} k x {nb} bands)", ibz),
                       ("FULL BZ — every star member "
                        f"({sum(r['nstar'] for r in per_k)} k x {nb} bands)",
                        tot)):
        print()
        print(f"{title}   [meV]")
        for name in ("sigSX", "sigCOH", "sigTOT"):
            d = np.concatenate(acc[name]) * 1e3
            print(f"  {name:<7s} N={d.size:5d}  MAE {np.abs(d).mean():9.4f}   "
                  f"max|d| {np.abs(d).max():9.4f}   "
                  f"mean(signed) {d.mean():+9.4f}   std {d.std():8.4f}")
    worst = max(r["star_spread"] for r in per_k)
    print()
    print(f"WORST PER-BAND STAR SPREAD in LORRAX sigTOT: {worst:.5f} meV")
    print("  (symmetry-equivalent k must carry identical Sigma; a nonzero")
    print("   value here means the full-BZ Sigma is not symmetric and the")
    print("   IBZ-representative comparison above depends on which k you pick.)")


def read_freq_debug(path: Path):
    """sigma_freq_debug.dat -> (col_names, dict ik -> (nb, ncol) array)."""
    lines = Path(path).read_text().splitlines()
    names = None
    for ln in lines:
        if ln.startswith("# k") or (ln.startswith("#") and "\tn" in ln):
            names = [t.strip() for t in ln.lstrip("#").split("\t") if t.strip()]
            break
    assert names, "no column header found in sigma_freq_debug.dat"
    out, ik = {}, -1
    for ln in lines:
        if ln.startswith("k-point"):
            ik = int(ln.split()[1].rstrip(":"))
            out[ik] = []
            continue
        s = ln.split()
        if len(s) == len(names) and not ln.startswith("#"):
            try:
                out[ik].append([float(x) for x in s])
            except ValueError:
                pass
    return names, {k: np.asarray(v) for k, v in out.items() if v}


def cmd_comparex(fixture, freq_debug, nband_limit=16):
    """Compare LORRAX bare exchange (x_bare) against BGW's X column.

    This is the column that isolates v(q+G), the zeta fit and the wavefunction
    contractions from all screening: BGW's X and LORRAX's x_bare are the same
    physical object with no convention offset between them.
    """
    kf, bands, bgw = read_fixture(Path(fixture))
    names, lx = read_freq_debug(Path(freq_debug))
    ci = {n: j for j, n in enumerate(names)}
    nb = min(bands.size, int(nband_limit))
    print(f"freq-debug columns: {names}")
    allk, ibzk = [], []
    print()
    print(f"{'ik':>3} {'kfrac':>26} {'star':>5} "
          f"{'x_bare-X MAE/max (meV)':>26}")
    for ik in range(kf.shape[0]):
        emf = bgw["Emf"][ik, :nb]
        hits = [k for k, v in lx.items()
                if v.shape[0] >= nb
                and np.max(np.abs(v[:nb, ci["E_dft"]] - emf)) < 2e-3]
        if not hits:
            print(f"  k{ik+1}: NO MATCH")
            continue
        ref = bgw["X"][ik, :nb]
        d = np.concatenate([lx[k][:nb, ci["x_bare"]] - ref for k in hits])
        allk.append(d)
        ibzk.append(lx[min(hits)][:nb, ci["x_bare"]] - ref)
        ks = f"[{kf[ik,0]:+.3f} {kf[ik,1]:+.3f} {kf[ik,2]:+.3f}]"
        print(f"{ik+1:>3} {ks:>26} {len(hits):>5} "
              f"{np.abs(d).mean()*1e3:12.4f}/{np.abs(d).max()*1e3:12.4f}")
    for title, acc in (("IBZ ONLY", ibzk), ("FULL BZ", allk)):
        d = np.concatenate(acc) * 1e3
        print(f"{title:9s} x_bare vs BGW X: N={d.size:5d}  "
              f"MAE {np.abs(d).mean():8.4f}  max|d| {np.abs(d).max():8.4f}  "
              f"mean(signed) {d.mean():+8.4f}  std {d.std():7.4f}   [meV]")


if __name__ == "__main__":
    if sys.argv[1] == "write":
        cmd_write(*sys.argv[2:5])
    elif sys.argv[1] == "compare":
        cmd_compare(*sys.argv[2:])
    elif sys.argv[1] == "comparex":
        cmd_comparex(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
