"""Decompose the k-star spread of the committed Σ cubes into its causes.

WHAT THIS ANSWERS.  ``file_io.sigma_output``'s refusal block reports a
113.08 eV worst element on ``hartree_kij_ev`` when the full-BZ cube is
compared against the unfold of its own IBZ rows, and the owner's question
about that number is which of three things it is:

  (a) the non-orbit-closed 60-centroid ISDF quadrature of this fixture,
      which breaks the deck's 12-op spatial group by construction;
  (b) degenerate-manifold BAND GAUGE — ⟨m|O|n⟩ is only defined up to a
      block-diagonal unitary inside each degenerate manifold, so raw
      elements may disagree across a star while the operator does not;
  (c) a real defect in the Σ procedure.

RAW ELEMENTS CANNOT TELL THEM APART, which is why the existing numbers do
not settle it.  This script measures, for every within-star pair, four
metrics of increasing gauge-blindness:

``raw``     ‖A_i − Θ A_j‖_F / ‖A_i‖_F, the metric already reported.
``mag``     ‖ |A_i| − |A_j| ‖_F / ‖A_i‖_F.  Blind to a per-band PHASE
            (the commonest gauge freedom) and to the conjugation.
``blockF``  the matrix G[I,J] = ‖A[I,J]‖_F over degenerate manifolds
            I, J.  INVARIANT under A → U† A U for any block-diagonal
            unitary U, which is exactly the residual freedom a
            diagonaliser has inside a degenerate manifold.  This is the
            discriminator: it is blind to (b) and blind to nothing else.
``trace``   Tr A[I,I] per manifold — the same invariance, and the
            quantity ``gw.degen_average`` already treats as the physical
            one.

AND THE CONTROL, which is what actually decides the verdict.  ``kin_ion``
(T + V_loc + V_NL) is built on the exact FFT grid with NO ISDF quadrature
anywhere in its path, from the SAME wavefunctions, and is unfolded by the
SAME star map.  So:

  * kin_ion clean + Σ/V_H broken  ⇒  the ISDF quadrature, hypothesis (a).
    The symmetry map, the k ordering and the band gauge are all exonerated
    by the control, because a fault in any of them would move kin_ion too.
  * kin_ion broken as well        ⇒  the star map or the band gauge, i.e.
    hypothesis (b) or (c), and the Σ procedure is not the place to look.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python tools/sigma_star_spread_decompose.py

Read-only: it opens the committed fixtures and writes nothing.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

_DECK = os.path.join(_REPO, "tests", "regression", "cohsex_debug")
_SIGMA = os.path.join(_DECK, "sigma_mnk.h5")
_KIN_ION = os.path.join(_DECK, "kin_ion.h5")
_WFN = os.path.join(_DECK, "WFNsmall.h5")
_EQP = os.path.join(_DECK, "eqp_ref.dat")

#: Manifolds are cut on the DFT eigenvalue spectrum with BGW's own
#: TOL_Degeneracy (1e-6 Ry), converted to the eV the ``Eo`` column is in.
_TOL_DEGEN_EV = 1.0e-6 * 13.605693122994


class _Header:
    """The eleven attributes ``SymMaps`` reads, out of ``mf_header``.

    Copied in shape from ``tests/test_star_offdiag_gate.py`` so both read
    the deck the same way; this is a tool, not a second definition of the
    header contract.
    """

    def __init__(self, path, h5py):
        with h5py.File(path, "r") as f:
            g = f["mf_header"]
            avec = g["crystal/avec"][:]
            apos = g["crystal/apos"][:]
            self.kpoints = g["kpoints/rk"][:]
            self.kgrid = g["kpoints/kgrid"][:]
            self.shift = g["kpoints/shift"][:]
            self.nkpts = int(g["kpoints/nrk"][()])
            self.ntran = int(g["symmetry/ntran"][()])
            self.sym_matrices = g["symmetry/mtrx"][:]
            self.translations = g["symmetry/tnp"][:]
            self.avec = avec
            self.atom_types = g["crystal/atyp"][:]
            self.atom_crys = np.einsum("ij,kj->ki",
                                       np.linalg.inv(avec).T, apos)
            self.trs_holds = True


def star_tables(h5py):
    """``(irr_idx_k, sym_idx_k, n_sym_spatial, xor_flags)`` for the deck."""
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import SymMaps
    from symmetry_maps.maps import _star_conj_flags

    sym = SymMaps(_Header(_WFN, h5py))
    irr = np.asarray(sym.irr_idx_k, dtype=int)
    sidx = np.asarray(sym.sym_idx_k, dtype=int)
    nss = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    _, xor = _star_conj_flags(irr, sidx, nss)
    return irr, sidx, nss, np.asarray(xor, dtype=bool)


def read_eo_ev():
    """``(nk, nb)`` DFT eigenvalues in eV from ``eqp_ref.dat``'s Eo column.

    The manifolds have to be cut on the SAME spectrum the Σ matrix was
    expressed in, and this file is that spectrum written by the run that
    produced the cube — safer than re-deriving it from the WFN header,
    whose k axis is the 4-row wedge rather than the cube's 9 rows.
    """
    rows: dict[int, dict[int, float]] = {}
    kcur = None
    with open(_EQP) as fh:
        for line in fh:
            mk = re.match(r"\s*k-point\s+(\d+)\s*:", line)
            if mk:
                kcur = int(mk.group(1))
                rows.setdefault(kcur, {})
                continue
            mn = re.match(r"\s*n=(\d+)\s+.*?Eo=\s*(-?[\d.eE+]+)", line)
            if mn and kcur is not None:
                rows[kcur][int(mn.group(1))] = float(mn.group(2))
    nk = max(rows) + 1
    nb = max(max(v) for v in rows.values()) + 1
    out = np.full((nk, nb), np.nan)
    for k, bands in rows.items():
        for n, e in bands.items():
            out[k, n] = e
    if np.isnan(out).any():
        raise ValueError(f"{_EQP}: Eo table has holes; cannot cut manifolds")
    return out


def manifolds(e_ev, tol=_TOL_DEGEN_EV):
    """Contiguous degenerate groups of one k-point's spectrum, as slices."""
    groups, start = [], 0
    for n in range(1, len(e_ev)):
        if abs(e_ev[n] - e_ev[n - 1]) > tol:
            groups.append((start, n))
            start = n
    groups.append((start, len(e_ev)))
    return groups


def block_frobenius(A, groups):
    """``G[I,J] = ‖A[I,J]‖_F`` — invariant under block-diagonal U† A U.

    A degenerate manifold's basis is fixed only up to a unitary mixing
    inside it, so a genuine operator identity between two k-points shows
    up here even when no single element matches.  Conjugation leaves it
    invariant too, which is what makes it usable without first deciding
    the antiunitary class.
    """
    ng = len(groups)
    G = np.zeros((ng, ng))
    for a, (i0, i1) in enumerate(groups):
        for b, (j0, j1) in enumerate(groups):
            G[a, b] = np.linalg.norm(A[i0:i1, j0:j1])
    return G


def block_traces(A, groups):
    """``Tr A[I,I]`` per manifold — the other block-diagonal invariant."""
    return np.array([np.trace(A[i0:i1, i0:i1]) for i0, i1 in groups])


def _rel(a, b):
    den = np.linalg.norm(np.asarray(a).ravel())
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) / den)


def pair_metrics(Ai, Aj, conj, groups):
    """The four metrics for one within-star pair, most gauge-blind last."""
    B = np.conj(Aj) if conj else Aj
    off = ~np.eye(Ai.shape[0], dtype=bool)
    return {
        "raw": _rel(Ai, B),
        "diag": _rel(np.diag(Ai), np.diag(B)),
        "offd": _rel(Ai[off], B[off]),
        "mag": float(np.linalg.norm(np.abs(Ai) - np.abs(Aj))
                     / np.linalg.norm(Ai)),
        "blockF": _rel(block_frobenius(Ai, groups),
                       block_frobenius(Aj, groups)),
        "trace": _rel(block_traces(Ai, groups), block_traces(Aj, groups)),
        "worst_ev": float(np.abs(Ai - B).max()),
    }


def star_pairs(irr, sidx, nss, xor):
    """Every within-star pair, tagged ``pure-TRS`` or ``spatial``."""
    out = []
    for i in range(len(irr)):
        for j in range(i + 1, len(irr)):
            if irr[i] != irr[j]:
                continue
            kind = ("pure-TRS" if int(sidx[i]) % nss == int(sidx[j]) % nss
                    else "spatial")
            out.append((i, j, bool(xor[i]) != bool(xor[j]), kind))
    return out


def report_dataset(name, A, pairs, eo_ev, print_fn=print):
    """One table per dataset; returns the per-kind worst of each metric."""
    print_fn(f"\n  {name}   shape {A.shape}   max|A| = {np.abs(A).max():.4f}")
    print_fn(f"    {'pair':>8s} {'kind':>9s} {'conj':>5s} "
             f"{'raw':>10s} {'diag':>10s} {'offd':>10s} "
             f"{'mag':>10s} {'blockF':>10s} {'trace':>10s} {'worst eV':>10s}")
    worst: dict[str, dict[str, float]] = {}
    for i, j, conj, kind in pairs:
        groups = manifolds(eo_ev[i])
        m = pair_metrics(A[i], A[j], conj, groups)
        print_fn(f"    ({i},{j}){'':>3s} {kind:>9s} {str(conj):>5s} "
                 f"{m['raw']:10.3e} {m['diag']:10.3e} {m['offd']:10.3e} "
                 f"{m['mag']:10.3e} {m['blockF']:10.3e} {m['trace']:10.3e} "
                 f"{m['worst_ev']:10.3f}")
        acc = worst.setdefault(kind, {})
        for key, val in m.items():
            acc[key] = max(acc.get(key, 0.0), val)
    return worst


def unfold_residual(A, irr, sidx, nss):
    """``A`` against ``unfold(select(A))`` — the refusal block's own metric.

    ``select`` takes each star's first-occurrence row (the IBZ rows an
    extraction would keep) and ``unfold`` is ``star_broadcast`` under its
    DEFAULT ``trs_reference="star_row"`` predicate, which is the one
    correct for a ``star_select`` output.  Reproducing the published
    113.08 eV here is what makes the decomposition below a decomposition
    OF THAT NUMBER rather than of a differently-defined cousin.
    """
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps

    labels = sorted({int(v) for v in irr})
    rep = [int(np.where(irr == lab)[0][0]) for lab in labels]
    lab_of = np.array([labels.index(int(v)) for v in irr], dtype=np.int32)
    ids = np.arange(len(rep), dtype=np.int32)
    if A.ndim == 4:
        U = np.stack([
            np.asarray(symmetry_maps.star_broadcast(
                A[w][rep], lab_of, np.asarray(sidx), nss, irr_labels=ids,
                trs_reference="star_row"))
            for w in range(A.shape[0])])
    else:
        U = np.asarray(symmetry_maps.star_broadcast(
            A[rep], lab_of, np.asarray(sidx), nss, irr_labels=ids,
            trs_reference="star_row"))
    rel = float(np.linalg.norm((A - U).ravel()) / np.linalg.norm(A.ravel()))
    return rel, float(np.abs(A - U).max()), U, rep


def diag_star_spread(M, irr, eo_ev, degen_average=False):
    """Per-band max−min of the REAL diagonal within a star, worst overall.

    ``compare_to_bgw``'s metric.  With ``degen_average=True`` each band's
    diagonal value is first replaced by its degenerate-manifold mean —
    ``gw.degen_average``'s operation, and the ONE way to read a diagonal
    that a manifold's internal gauge cannot move.  The difference between
    the two calls is the share of the number that is gauge.
    """
    worst = 0.0
    for lab in sorted({int(v) for v in irr}):
        mem = [k for k in range(len(irr)) if int(irr[k]) == lab]
        if len(mem) < 2:
            continue
        rows = []
        for k in mem:
            d = np.real(np.diag(M[k])).copy()
            if degen_average:
                for i0, i1 in manifolds(eo_ev[k]):
                    d[i0:i1] = d[i0:i1].mean()
            rows.append(d)
        blk = np.stack(rows)
        worst = max(worst, float((blk.max(0) - blk.min(0)).max()))
    return worst


def centroid_closure(h5py):
    """``verify_centroid_orbit_closure`` on this deck's own centroid set."""
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps.orbit_syms import verify_centroid_orbit_closure

    cen = np.loadtxt(os.path.join(_DECK, "centroids_frac_60.txt"))
    with h5py.File(_WFN, "r") as f:
        g = f["mf_header"]
        mtrx = g["symmetry/mtrx"][:]
        tnp = g["symmetry/tnp"][:]
        ntran = int(g["symmetry/ntran"][()])
    return verify_centroid_orbit_closure(cen, mtrx[:ntran], tnp=tnp[:ntran])


def main():
    import h5py

    for path in (_SIGMA, _KIN_ION, _WFN, _EQP):
        if not os.path.isfile(path):
            print(f"missing fixture {path} — nothing to measure")
            return 1

    irr, sidx, nss, xor = star_tables(h5py)
    pairs = star_pairs(irr, sidx, nss, xor)
    eo_ev = read_eo_ev()

    print("=" * 78)
    print("k-star spread decomposition — tests/regression/cohsex_debug")
    print("=" * 78)
    print(f"  nk={len(irr)}  stars={len(set(irr.tolist()))}  ntran={nss}")
    print(f"  irr_idx_k = {irr.tolist()}")
    print(f"  sym_idx_k = {sidx.tolist()}")
    print(f"  xor flags = {xor.astype(int).tolist()}")
    ngroups = [len(manifolds(eo_ev[k])) for k in range(eo_ev.shape[0])]
    print(f"  degenerate manifolds per k "
          f"(of {eo_ev.shape[1]} bands): {ngroups}")

    print("\n" + "-" * 78)
    print("CONTROL — kin_ion (exact FFT grid, no ISDF quadrature anywhere)")
    print("-" * 78)
    with h5py.File(_KIN_ION, "r") as f:
        kin = f["kin_ion"][:]
    nb = eo_ev.shape[1]
    ctrl = report_dataset("kin_ion (Ry)", kin[:, :nb, :nb], pairs, eo_ev)

    print("\n" + "-" * 78)
    print("THE Σ CUBES (every one of them through the 60-centroid ISDF)")
    print("-" * 78)
    with h5py.File(_SIGMA, "r") as f:
        subject = {}
        for name in ("hartree_kij_ev", "sigma_sx_kij_ev",
                     "sigma_xc_qsgw_kij_ev"):
            subject[name] = f[name][:]
        # The dynamic cubes are measured at one ω; the star relation is a
        # per-ω statement and ω₀ is the one every static consumer uses.
        iw = int(np.argmin(np.abs(f["omega_ev"][:])))
        for name in ("sigma_c_kij_ev", "sigma_total_kij_ev"):
            subject[f"{name}[w={iw}]"] = f[name][iw]
    subj = {n: report_dataset(n, A, pairs, eo_ev) for n, A in subject.items()}

    print("\n" + "=" * 78)
    print("WITHIN-STAR PAIR SUMMARY (worst over pairs of each kind)")
    print("=" * 78)
    print(f"  {'dataset':>26s} {'kind':>9s} " + " ".join(
        f"{k:>10s}" for k in ("raw", "mag", "blockF", "trace")))
    for name, worst in [("kin_ion (CONTROL)", ctrl)] + list(subj.items()):
        for kind in ("pure-TRS", "spatial"):
            if kind not in worst:
                continue
            w = worst[kind]
            print(f"  {name:>26s} {kind:>9s} " + " ".join(
                f"{w[k]:10.3e}" for k in ("raw", "mag", "blockF", "trace")))

    # -- the published number, reproduced and then split ------------------
    print("\n" + "=" * 78)
    print("THE PUBLISHED unfold(select(A)) RESIDUAL, REPRODUCED AND SPLIT")
    print("=" * 78)
    with h5py.File(_SIGMA, "r") as f:
        full = {n: f[n][:] for n in
                ("hartree_kij_ev", "sigma_sx_kij_ev", "sigma_xc_qsgw_kij_ev",
                 "sigma_c_kij_ev", "sigma_total_kij_ev")}
    print(f"  {'dataset':>22s} {'relFrob':>11s} {'worst eV':>10s} "
          f"{'diag raw':>10s} {'diag avgd':>10s} {'gauge share':>12s}")
    for name, A in full.items():
        rel, worst, U, rep = unfold_residual(A, irr, sidx, nss)
        M = A[0] if A.ndim == 4 else A
        d_raw = diag_star_spread(M, irr, eo_ev)
        d_avg = diag_star_spread(M, irr, eo_ev, degen_average=True)
        share = (1.0 - d_avg / d_raw) if d_raw else float("nan")
        print(f"  {name:>22s} {rel:11.4e} {worst:10.2f} "
              f"{d_raw:10.2f} {d_avg:10.2f} {100 * share:11.1f}%")

    A = full["hartree_kij_ev"]
    _, _, U, _ = unfold_residual(A, irr, sidx, nss)
    k, m, n = np.unravel_index(np.abs(A - U).argmax(), A.shape)
    grp = [g for g in manifolds(eo_ev[k]) if g[0] <= m < g[1]][0]
    print(f"\n  The worst element sits at (k, m, n) = ({k}, {m}, {n}) — "
          f"{'DIAGONAL' if m == n else 'OFF-DIAGONAL'}.")
    print(f"    A = {A[k, m, n]:.4f}   unfold = {U[k, m, n]:.4f}")
    par = abs(A[rep_of(irr)[k], m, n])
    print(f"    |A| = {abs(A[k, m, n]):.4f}  vs its star parent's "
          f"|A| = {par:.4f}  "
          f"(agree to {abs(abs(A[k, m, n]) - par):.4f} eV)")
    print(f"    band {m} lies in the degenerate manifold {grp} "
          f"(multiplicity {grp[1] - grp[0]}).")

    # -- the cause, measured on the centroid set itself -------------------
    print("\n" + "=" * 78)
    print("THE CAUSE, MEASURED ON THE CENTROID SET ITSELF")
    print("=" * 78)
    v = centroid_closure(h5py)
    print(f"  centroids_frac_60.txt closed under the {v.n_sym}-op spatial "
          f"group?  {v.closed}")
    print(f"  metric: {v.metric}   tol {v.tol:g}")
    print(f"  worst residual {v.worst_residual:.4f} at op {v.worst_op}; "
          f"violating ops {list(v.violating_ops)}")
    print("  per-op worst residual:")
    for i, r in enumerate(np.asarray(v.residual_by_op)):
        print(f"    op {i:2d}  {r:.4f}"
              + ("   <- IDENTITY, exactly closed" if r == 0.0 else ""))
    return 0


def rep_of(irr):
    """``k -> the first-occurrence row of k's star`` — the select map."""
    first = {}
    out = np.empty(len(irr), dtype=int)
    for k, lab in enumerate(int(v) for v in irr):
        first.setdefault(lab, k)
        out[k] = first[lab]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
