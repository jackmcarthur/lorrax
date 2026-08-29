"""
psp/vnl_ops.py — Fast VNL operator: build once, apply many times.

All channels × atoms × betas are concatenated into a single dense
projector matrix Z of shape (total_R, nG).  E is a block-diagonal
(nspinor, nspinor, total_R, total_R) matrix.  The VNL operator is
then a single set of einsums with no Python loops.

Radial form factors use table lookup on a uniform q-grid (linear
interpolation) instead of spline evaluation. This keeps the production
path in a single JAX/table architecture and is still accurate with a
50k-point table (spacing ~1e-4).

Solid harmonics (Cartesian polynomials) replace angular Y_lm for
autodiff-safe behaviour at K=0.
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from psp.radial.build_projectors_qe import (
    build_E_blocks_full, pseudo_has_j_channels, pseudo_soc_strength_ry,
)
from psp.radial.radial_jax import differentiate_uniform_table
from psp.radial.solid_harmonics import solid_harmonics_jax as _solid_harmonics_jax


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelMeta:
    """Metadata for one (species, l) VNL channel."""
    l: int
    nbeta: int
    msize: int                      # 2l+1
    R: int                          # nbeta * msize
    tau: np.ndarray                 # (natoms, 3) crystal positions
    E: np.ndarray                   # (2, 2, R, R) D-matrix (full spin)
    beta_table_start: int           # index into flattened radial table
    natoms: int


@dataclass
class VNLSetup:
    """K-independent VNL data.  Built once from pseudopotentials.

    Radial form factors are stored as tables on a uniform q-grid.
    Per-row metadata (beta_idx, l, m, tau) is pre-flattened so the
    per-k Z assembly is a single vectorized operation — no loops.
    """
    channels: list[ChannelMeta]
    dq: float
    n_q: int
    q_max: float
    G_table: jax.Array              # (total_nbeta, n_q)
    Gp_table: jax.Array             # (total_nbeta, n_q)
    prefactor: float                # 4π/√Ω
    B: np.ndarray                   # (3,3) crystal→Cartesian
    cell_volume: float
    total_R: int
    nspinor: int
    E_super: jax.Array | None = None  # (nspinor, nspinor, total_R, total_R)
    l_max: int = 0
    # WHICH PROJECTORS THESE ARE.  True = j-resolved (spin-orbit in V_NL);
    # False = j-averaged scalar-relativistic (QE average_pp).  Resolved
    # AUTOMATICALLY (metadata via ``resolve_soc_mode``, else measured
    # against the wavefunctions by ``measure_soc_mode``) and carried so a
    # consumer can stamp it into an artifact's provenance instead of
    # guessing from ``nspinor``.
    soc: bool = True
    # HOW ``soc`` was decided, as one human-readable line for the
    # producers' report blocks (e.g. "j-AVERAGED selected by measurement —
    # multiplet consistency 1.5e-08 eV vs 4.8e-02 eV j-resolved").  The
    # production stdout filter keeps only WARNING-class legacy prints, so
    # this line is how an informational verdict reaches the report.
    soc_provenance: str = ""
    # ── Pre-flattened per-row metadata for vectorized Z assembly ──
    # Each row r of Z(total_R, nG) is: c_il * G[beta_idx[r], q] * S[l[r], m[r], G] * phase[tau[r], G]
    row_beta_idx: jax.Array | None = None  # (total_R,) int — which G_table row
    row_l: jax.Array | None = None         # (total_R,) int — angular momentum
    row_m: jax.Array | None = None         # (total_R,) int — m index into S_all[l]
    row_tau: jax.Array | None = None       # (total_R, 3) float — atom position (crystal)


@dataclass
class VNLKData:
    """Per-k-point VNL projector data.  Precomputed for fast apply."""
    Z: jax.Array                    # (total_R, nG) complex128
    E_super: jax.Array              # (nspinor, nspinor, total_R, total_R)
    nG: int
    total_R: int
    # Optional: Cartesian k-derivatives for velocity.  ``None`` IF AND ONLY
    # IF the builder was called with ``compute_dZ=False`` — a builder asked
    # for dZ always returns a ``(3, total_R, nG)`` array, including the
    # degenerate ``total_R == 0`` case (no channels), where it is the empty
    # array rather than ``None``.  Consumers may therefore branch on
    # ``compute_dZ``, never on ``dZ is None`` after asking for it.
    dZ: jax.Array | None            # (3, total_R, nG), or None iff not asked for
    # 1 on the k's physical G, 0 on the ngkmax pad — None when the G-list
    # was the k's own ragged sphere.  Set by ``build_vnl_kdata`` (the
    # SymMaps path), which ALSO zeroes Z/dZ on the pad columns, so this
    # field documents the layout rather than being a required argument to
    # anything: a kdata from that route is already inert on the pad.
    g_mask: np.ndarray | None = None   # (nG,) float64 or None


# ---------------------------------------------------------------------------
# Setup builder
# ---------------------------------------------------------------------------

RY_TO_EV = 13.605693122994

#: Two bands are "the same el manifold" within this (Ry) — the same
#: identity tolerance ``operator_checks.check_degeneracy_consistency``
#: defaults to.  Exact degeneracies on these fixtures sit at ~1e-11 Ry.
_MEASURE_EL_TOL_RY = 1e-9

#: A candidate operator whose worst within-manifold spectrum spread stays
#: below this is CONSISTENT with the wavefunctions.  Measured brackets on
#: the Si 4x4x4 spinor fixture (lspinorb=false WFN + FR Si): the correct
#: j-averaged operator sits at 1.1e-9 Ry, the wrong j-resolved one at
#: 3.5e-3 Ry — this floor is ~3 decades above the first and ~3.5 below
#: the second.
_MEASURE_SPLIT_FLOOR_RY = 1e-6

#: Probe size: bands x k-rows actually evaluated.  24 bands x 3 rows costs
#: ~0.4 s on the Si fixture (one eager ψ read + three small Z builds) —
#: measured 2026-08-28; the full k-set is never built twice.
_MEASURE_NB = 24
_MEASURE_NK = 3


def resolve_soc_mode(pseudos, wfn=None, *, nspinor,
                     caller: str = "", print_fn=print) -> bool | None:
    """Decide j-RESOLVED vs j-AVERAGED projectors from METADATA, and SAY SO.

    There is deliberately no caller/deck/CLI ``soc`` input: the resolution
    is automatic, always, from evidence (owner ruling 2026-08-28).  The
    full contract has five arms; this function owns the three that need no
    wavefunction data and returns ``None`` for the one that does:

    1. No pseudo resolves j = ℓ±1/2 (scalar-relativistic UPFs, or ℓ=0
       only) → ``False``, quietly: there is no choice to make, both paths
       build the same spin-scalar operator.
    2. ``wfn.spinorbit`` — QE's ``<spinorbit>`` from
       ``data-file-schema.xml``, present when the structure came from a QE
       ``.save`` (see ``file_io.qe_save_reader``) → honored and announced.
       That is DATA about the run, not a flag.  ``spinorbit=true`` with
       nspinor=1 RAISES (below).
    3. Undetermined + nspinor=1 → ``False``, FORCED and announced: a
       j-resolved V_NL needs 2-component spinors, so there is only one
       representable choice (= QE ``average_pp``).
    4. Undetermined + nspinor=2 → ``None``: nothing in a BerkeleyGW
       ``WFN.h5`` records lspinorb (``mf_header`` carries ``nspinor`` and
       nothing else — noncollinear does NOT imply spin-orbit), so the
       answer is MEASURED against the wavefunctions themselves by
       :func:`measure_soc_mode`; ``build_vnl_setup`` runs that measurement
       because it owns the projector tables the probe needs.
    5. (In ``measure_soc_mode``.)  Both candidate operators are evaluated
       on degenerate multiplets of the WFN; the one consistent with the
       spectrum wins, the verdict is announced with both numbers, and an
       unmeasurable or contradictory input REFUSES loudly.

    RAISES ``ValueError`` for spinorbit=true with nspinor=1: a j-resolved
    V_NL has no representation on one-component wavefunctions, and
    truncating it to the E^{↑↑} block is m-dependent and is NOT the
    scalar-relativistic operator.
    """
    tag = f"[{caller}] " if caller else ""
    j_pseudos = {el: p for el, p in (pseudos or {}).items()
                 if pseudo_has_j_channels(p)}

    # ── nothing resolves j: the question does not arise ──
    if not j_pseudos:
        return False

    declared = getattr(wfn, "spinorbit", None)
    source = "QE <spinorbit>"

    strengths = {el: pseudo_soc_strength_ry(p) for el, p in j_pseudos.items()}
    worst_ry = max(strengths.values(), default=0.0)
    worst_ev = worst_ry * RY_TO_EV

    if declared is None:
        if nspinor == 1:
            # FORCED, not assumed: one-component wavefunctions admit only
            # the j-averaged operator, so the undetermined case has a
            # unique representable resolution (= QE average_pp).  The
            # WARNING prefix is load-bearing: production output routes
            # legacy prints through preprocessing_output.legacy_print,
            # which retains only _WARNING_WORDS lines — without it this
            # announcement is silently dropped unless LORRAX_DEBUG_PRINT=1.
            print_fn(f"{tag}WARNING: V_NL j-AVERAGED (scalar-relativistic, QE "
                     f"average_pp), forced by nspinor=1.  FR pseudos "
                     f"{sorted(j_pseudos)}, discarding "
                     f"ΔD = {worst_ry:.6f} Ry = {worst_ev:.4f} eV of "
                     f"spin-orbit.")
            return False
        # ── UNDETERMINED: measure, never assume.  (Arm 4 → arm 5.) ──
        return None

    declared = bool(declared)
    if declared:
        if nspinor == 1:
            raise ValueError(
                f"{tag}spin-orbit (from {source}) is impossible with "
                f"nspinor=1: a j-resolved V_NL has no representation on "
                f"one-component wavefunctions — truncating it to the "
                f"E^{{↑↑}} block is m-dependent and is NOT the "
                f"scalar-relativistic operator.  "
                f"got: {source}=true, nspinor=1 (FR pseudos "
                f"{sorted(j_pseudos)}, ΔD = {worst_ry:.6f} Ry at stake); "
                f"want: an nspinor=2 WFN (j-resolved), or the j-averaged "
                f"operator (QE average_pp) with nspinor=1; "
                f"fix: use an nspinor=2 WFN, or a scalar-relativistic "
                f"pseudopotential.")
        print_fn(f"{tag}V_NL: j-RESOLVED (spin-orbit ON), from {source}.  "
                 f"FR pseudos {sorted(j_pseudos)}, "
                 f"ΔD = {worst_ry:.6f} Ry = {worst_ev:.4f} eV.")
        return True

    # declared False, and the pseudos DO resolve j → the averaged path
    print_fn(f"{tag}V_NL: j-AVERAGED (scalar-relativistic, QE average_pp), "
             f"from {source}.  FR pseudos {sorted(j_pseudos)}, "
             f"discarding ΔD = {worst_ry:.6f} Ry = {worst_ev:.4f} eV of "
             f"spin-orbit.")
    return False


def _row_manifolds(e_row: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Degenerate groups ``(i, j)`` inclusive in one ascending el row."""
    out, i, nb = [], 0, e_row.shape[0]
    while i < nb:
        j = i
        while j + 1 < nb and abs(e_row[j + 1] - e_row[i]) <= tol:
            j += 1
        if j > i:
            out.append((i, j))
        i = j + 1
    return out


def _spin_pairing_break_ry(en: np.ndarray, tol: float) -> float:
    """How badly ``el`` breaks exact spin degeneracy, in Ry (0.0 = paired).

    A j-averaged (lspinorb=.false., noncolin) Hamiltonian commutes with
    every spin rotation, so EVERY eigenvalue has even multiplicity at
    every k.  An odd-sized degenerate group anywhere is therefore proof
    the run was NOT average_pp — the positive lspinorb signature the
    multiplet-consistency probe alone cannot give (both candidate
    operators are (P)T-even, hence exactly scalar on protected Kramers
    doublets).  The returned magnitude is the gap to the nearest level
    that would have completed the odd group — i.e. the spin splitting
    itself.  Groups touching the TOP probed band are skipped: their
    partner may simply be truncated off the window.
    """
    worst = 0.0
    nb = en.shape[1]
    for row in en:
        i = 0
        while i < nb:
            j = i
            while j + 1 < nb and abs(row[j + 1] - row[i]) <= tol:
                j += 1
            if (j - i + 1) % 2 == 1 and j < nb - 1:
                below = row[i] - row[i - 1] if i > 0 else np.inf
                worst = max(worst, float(min(below, row[j + 1] - row[j])))
            i = j + 1
    return worst


def measure_soc_mode(wfn, setup: VNLSetup, E_super_avg, E_super_res, *,
                     delta_d_ry: float, caller: str = "",
                     print_fn=print) -> tuple[bool, str]:
    """Arm 5 of the automatic resolution: MEASURE which V_NL the WFN wants.

    The Z projector tables are soc-independent — only the small E blocks
    differ between the j-resolved and j-averaged candidates — so both
    operators are evaluated on a handful of DEGENERATE multiplets of the
    wavefunctions (≤ :data:`_MEASURE_NK` k-rows × :data:`_MEASURE_NB`
    bands; the full k-set is never built twice).  The correct operator
    leaves each degenerate subspace invariant; the wrong one splits it at
    the SOC scale.  Calibration on the Si 4x4x4 spinor fixture
    (lspinorb=false WFN + FR Si, 2026-08-28): the j-resolved candidate
    splits the probed multiplets by 4.8e-2 eV while the j-averaged one
    sits at 1.5e-8 eV — seven decades of discrimination.  The consistency
    numbers come from ``operator_checks.check_degeneracy_consistency``,
    the same gauge-invariant block-spectrum detector the kin_ion producer
    already runs post hoc.

    Decision rule, in order:

    * one candidate consistent (≤ :data:`_MEASURE_SPLIT_FLOOR_RY`), the
      other split above it → the consistent one, announced with both
      numbers;
    * both split → RuntimeError: the wavefunctions disagree with this
      pseudopotential under BOTH hypotheses (wrong WFN/UPF pairing);
    * both consistent and ΔD ≤ floor → the operators are numerically the
      same object; j-resolved, announced with the bound;
    * both consistent, ΔD material: ``el`` breaking exact spin pairing
      anywhere (see :func:`_spin_pairing_break_ry`) is proof of
      lspinorb=.true. → j-resolved; failing that, a ≥3-fold multiplet
      that FEELS the operator difference (identity shift > floor) yet is
      split by neither is the double-group signature → j-resolved;
      otherwise nothing probed discriminates → RuntimeError naming
      exactly what was measured.

    Returns ``(soc, provenance_line)``.  The caller routes
    ``provenance_line`` into the production report (the stdout filter
    would drop a non-WARNING legacy print).
    """
    tag = f"[{caller}] " if caller else ""
    from psp.dft_operators import _as_loader
    from psp.operator_checks import check_degeneracy_consistency
    try:
        loader = _as_loader(wfn)
    except Exception as exc:                              # noqa: BLE001
        raise RuntimeError(
            f"{tag}V_NL spin-orbit mode is undetermined (FR pseudopotential, "
            f"nspinor=2, no QE <spinorbit> record) and cannot be measured: "
            f"no WFN reader behind {type(wfn).__name__!r} ({exc}).  "
            f"want: a loadable WFN.h5, or a QE .save whose <spinorbit> is "
            f"authoritative.") from exc

    t0 = time.perf_counter()
    from wfn_loader import IBZRows

    en = np.asarray(loader.energies, dtype=np.float64)
    en = en[0] if en.ndim == 3 else en                     # (nk, nb) Ry
    nb_probe = min(en.shape[1], _MEASURE_NB)
    en = en[:, :nb_probe]
    kpts = np.asarray(loader.kpoints, dtype=np.float64)
    tol = _MEASURE_EL_TOL_RY
    floor = _MEASURE_SPLIT_FLOOR_RY
    ev = RY_TO_EV

    pair_break = _spin_pairing_break_ry(en, floor)

    # Rank the WFN's own rows: biggest multiplet first (a Γ 6-fold decides
    # in one block), then non-TRIM (at k ≢ −k a same-k doublet is NOT the
    # T-protected Kramers pair, so the j-resolved candidate can split it —
    # the MoS2 average_pp discriminator), then multiplet count.
    scored = []
    for r in range(en.shape[0]):
        mf = _row_manifolds(en[r], tol)
        big = max((b - a + 1 for a, b in mf), default=0)
        trim = bool(np.allclose(2 * kpts[r] - np.round(2 * kpts[r]), 0.0,
                                atol=1e-8))
        scored.append((big, not trim, len(mf), r))
    scored.sort(reverse=True)
    rows = [s[3] for s in scored[:_MEASURE_NK] if s[0] >= 2]

    def _verdict(soc: bool, why: str) -> tuple[bool, str]:
        line = (f"{'j-RESOLVED' if soc else 'j-AVERAGED'} selected by "
                f"measurement — {why}")
        print_fn(f"{tag}V_NL: {line}")
        return soc, line

    if not rows:
        if delta_d_ry <= floor:
            return _verdict(True, f"no degenerate multiplets to test, but "
                            f"ΔD = {delta_d_ry * ev:.1e} eV ≤ floor: the two "
                            f"operators are equivalent below that bound")
        if pair_break > floor:
            return _verdict(True, f"el breaks exact spin degeneracy by up to "
                            f"{pair_break * ev:.3e} eV (lspinorb signature; "
                            f"average_pp eigenvalues all have even "
                            f"multiplicity); no multiplets to block-test")
        raise RuntimeError(
            f"{tag}V_NL spin-orbit mode UNMEASURABLE: FR pseudopotential "
            f"(ΔD = {delta_d_ry * ev:.4f} eV at stake), nspinor=2, no QE "
            f"<spinorbit> record, and the WFN offers no evidence — all "
            f"{en.shape[0]} k-rows × {nb_probe} bands scanned: zero "
            f"degenerate multiplets at {tol:.0e} Ry and exact spin pairing "
            f"everywhere.  want: the QE .save (its <spinorbit> is "
            f"authoritative), or a WFN sampling a k with degenerate bands.")

    # ── evaluate BOTH candidates on the probe rows (Z shared, E swapped) ──
    kspec = IBZRows(tuple(int(r) for r in rows))
    psi = np.asarray(loader.load_process_local(bands=(0, nb_probe), k=kspec))
    gvecs = np.asarray(loader.gvecs(k=kspec))
    H = {False: [], True: []}
    for i, r in enumerate(rows):
        # ψ pad-G coefficients are zero by the loader contract, and every
        # contraction below passes through ψ, so the pad columns of Z
        # (finite by design — see _build_vnl_kdata_core) are inert.
        kdata = build_vnl_kdata_from_kvec(kpts[r], gvecs[i], setup)
        P = jnp.einsum('RG,nsG->Rsn', jnp.conj(kdata.Z),
                       jnp.asarray(psi[i]), optimize=True)
        for soc, E in ((False, E_super_avg), (True, E_super_res)):
            D = jnp.einsum('stRQ,Qtn->Rsn', E, P, optimize=True)
            H[soc].append(np.asarray(
                jnp.einsum('Rsm,Rsn->mn', jnp.conj(P), D, optimize=True)))

    silent = lambda *a, **k: None                          # noqa: E731
    res = {soc: check_degeneracy_consistency(
        np.stack(H[soc]), en[rows], el_tol_ry=tol, split_tol_ry=floor,
        label=f"V_NL probe soc={soc}", print_fn=silent) for soc in (False, True)}
    s_avg = float(res[False]["max_split_ry"])
    s_res = float(res[True]["max_split_ry"])
    n_m = int(res[False]["n_manifolds"])
    cost = time.perf_counter() - t0
    both = (f"multiplet consistency {s_avg * ev:.1e} eV j-averaged vs "
            f"{s_res * ev:.1e} eV j-resolved ({len(rows)} k, {n_m} "
            f"multiplets, {cost:.1f} s)")

    if s_avg > floor and s_res > floor:
        raise RuntimeError(
            f"{tag}V_NL REFUSED: BOTH candidate operators split el "
            f"multiplets the wavefunctions hold degenerate — {both}.  "
            f"Neither lspinorb hypothesis fits: the WFN and these "
            f"pseudopotentials do not belong together (wrong UPF set, or a "
            f"broken WFN).")
    if s_avg <= floor < s_res:
        if pair_break > floor:
            raise RuntimeError(
                f"{tag}V_NL REFUSED: contradictory evidence — el breaks "
                f"spin degeneracy by {pair_break * ev:.3e} eV (lspinorb "
                f"signature) yet the j-resolved candidate splits el "
                f"multiplets ({both}).  Check the WFN/pseudopotential "
                f"pairing.")
        return _verdict(False, both)
    if s_res <= floor < s_avg:
        return _verdict(True, both)

    # ── both candidates consistent ──
    if delta_d_ry <= floor:
        return _verdict(True, f"operators equivalent below "
                        f"ΔD = {delta_d_ry * ev:.1e} eV; {both}")
    if pair_break > floor:
        return _verdict(True, f"el breaks exact spin degeneracy by up to "
                        f"{pair_break * ev:.3e} eV (lspinorb signature); "
                        f"{both}")
    # Everything spin-paired and both candidates scalar on every probed
    # multiplet.  A ≥3-fold multiplet whose restricted operator DIFFERENCE
    # is a material identity shift (both blocks ~λ·1 with λ_res ≠ λ_avg)
    # can only be a double-group irrep — an average_pp multiplet of size
    # ≥3 with ℓ>0 character MUST be split by the j-resolved candidate.
    shift = 0.0
    for i, r in enumerate(rows):
        dM = H[True][i] - H[False][i]
        for a, b in _row_manifolds(en[r], tol):
            if b - a + 1 < 3:
                continue
            blk = dM[a:b + 1, a:b + 1]
            w = np.linalg.eigvalsh((blk + blk.conj().T) / 2.0).real
            shift = max(shift, float(np.max(np.abs(w))))
    if shift > floor:
        return _verdict(True, f"spectrum already resolves j: a ≥3-fold el "
                        f"multiplet feels the operator difference "
                        f"({shift * ev:.1e} eV identity shift) yet neither "
                        f"candidate splits it — double-group signature; "
                        f"{both}")
    raise RuntimeError(
        f"{tag}V_NL spin-orbit mode UNMEASURABLE: the candidates differ "
        f"materially (ΔD = {delta_d_ry * ev:.4f} eV) but nothing probed "
        f"discriminates — {both}; exact spin pairing everywhere and every "
        f"probed multiplet is insensitive to the difference (worst "
        f"restricted shift {shift * ev:.1e} eV ≤ {floor * ev:.1e} eV).  "
        f"want: the QE .save (its <spinorbit> is authoritative), or a WFN "
        f"sampling a high-symmetry k whose multiplets the candidates "
        f"treat differently.")


def build_vnl_setup(
    wfn, sym=None, meta=None, pseudos=None,
    n_q: int | None = None,
    nspinor: int | None = None,
    q_max: float | None = None,
    print_fn=print,
) -> VNLSetup:
    """Build k-independent VNL data: radial tables, channel metadata.

    Parameters
    ----------
    wfn : WFNReader or CrystalData (needs atom_crys, bvec, blat, cell_volume)
    sym : SymMaps, optional — used to determine q_max if not provided.
    meta : Meta, optional — used with sym for q_max scan.
    pseudos : dict — element → UPF
    n_q : int, optional — radial-table node count.  ``None`` (default)
        scales it with q_max so the grid spacing dq is ECUT-INDEPENDENT
        (target 5e-4 bohr⁻¹, floor 4000 nodes).  The historical fixed
        n_q=4000 made dq GROW with the cutoff — at 80 Ry (dq ≈ 2.3e-3)
        the linear interp of the sharply curved Fe/Zn 3d/semicore-s form
        factors cost a measured 0.0024/0.0011 meV on the KIH diagonal vs
        QE (2026-08-28 atom sweep); at dq = 5e-4 that is ~1e-4 meV, below
        the light-element agreement floor, for a ~3 s (was ~1 s) one-time
        per-run table build.  Pass an explicit n_q to override.
    q_max : float, optional — if provided, skip the k-point scan for q_max.

    j-RESOLVED vs j-AVERAGED projectors are resolved AUTOMATICALLY —
    there is deliberately no ``soc`` input anywhere: metadata first
    (:func:`resolve_soc_mode`: scalar pseudos are quiet, QE
    ``<spinorbit>`` is authoritative, nspinor=1 forces j-averaged), and
    the previously-ambiguous FR + nspinor=2 + BGW-WFN case is MEASURED
    against the wavefunctions (:func:`measure_soc_mode`) using the same
    Z tables built here — the E blocks are the only soc-dependent piece,
    so the measurement costs a few small multiplet blocks, never a
    second k-set.  The verdict lands in :attr:`VNLSetup.soc` and, as one
    human-readable line, :attr:`VNLSetup.soc_provenance`.
    """
    from psp.species import extract_species, build_atom_species_map
    from psp.radial_tables import build_all_tables

    if nspinor is None:
        nspinor = int(meta.nspinor) if meta is not None else int(wfn.nspinor)

    soc_resolved = resolve_soc_mode(
        pseudos, wfn, nspinor=int(nspinor),
        caller="build_vnl_setup", print_fn=print_fn)

    B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)
    cell_volume = float(wfn.cell_volume)
    prefactor = (4.0 * np.pi) / np.sqrt(cell_volume)

    # Determine q_max
    if q_max is None:
        if hasattr(wfn, "ecutwfc"):
            q_max = float(np.sqrt(float(wfn.ecutwfc)))
        else:
            from psp.dft_operators import padded_gvectors
            tab = padded_gvectors(wfn, k="full_bz")
            q_max = 0.0
            for ik in range(sym.nk_tot):
                G_pad, g_mask = tab.at(ik)
                kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
                K_cart = (G_pad.astype(float) + kvec[None, :]) @ B
                # Pad rows are G=(0,0,0), i.e. |k| — small, but not
                # provably below the physical maximum, so zero them
                # rather than argue about it.  q_max only ever grows the
                # radial table, so an over-estimate is harmless while an
                # under-estimate silently clips the form factors.
                qk = np.sqrt(np.sum(K_cart ** 2, axis=1)) * g_mask
                if qk.size:
                    q_max = max(q_max, float(np.max(qk)))
    q_max *= 1.01

    if n_q is None:
        # dq ecut-independent (see docstring): a fixed node count over a
        # cutoff-dependent [0, q_max] hands the COARSEST table to exactly
        # the hard-pseudo/high-ecut decks whose form factors curve most.
        n_q = max(4000, int(np.ceil(q_max / 5.0e-4)) + 1)

    # Extract species data and projector tables
    species_list = extract_species(pseudos)
    tables = build_all_tables(species_list, q_max, n_q)
    species_natoms, species_tau, _ = build_atom_species_map(wfn, species_list)
    q_grid = tables["q"]
    dq = tables["dq"]

    # ── E blocks: the ONLY soc-dependent piece of the whole setup ──
    # (Z tables, channel layout and row metadata are identical for both
    # arms, which is what makes the measured resolution below cheap.)
    # ``soc_resolved is None`` = arm 4 of resolve_soc_mode: build BOTH
    # arms and let the wavefunctions choose.  A non-j-averageable FR
    # pseudopotential (opposite-sign D_jj, QE's average_pp would NaN)
    # removes the averaged candidate: an lspinorb=.false. run cannot
    # exist with that file, so j-resolved is the only operator — forced,
    # announced.  In the DECIDED-False arms the same ValueError
    # propagates: there the averaged operator was promised (QE
    # <spinorbit>=false, or nspinor=1) and cannot be built.
    measure = soc_resolved is None
    live = [sp.element for isp, sp in enumerate(species_list)
            if int(species_natoms[isp]) > 0]

    def _species_blocks(arm: bool) -> dict:
        return {el: build_E_blocks_full(pseudos[el], soc=arm) for el in live}

    blocks_by_arm: dict[bool, dict] = {}
    forced_line = None
    if measure:
        blocks_by_arm[True] = _species_blocks(True)
        try:
            blocks_by_arm[False] = _species_blocks(False)
        except ValueError as exc:
            soc_resolved, measure = True, False
            forced_line = (f"j-RESOLVED forced — FR pseudopotential not "
                           f"j-averageable ({exc})")
            print_fn(f"[build_vnl_setup] V_NL: {forced_line}")
    else:
        blocks_by_arm[bool(soc_resolved)] = _species_blocks(bool(soc_resolved))

    # Channels carry the arm-0 blocks; the measured path swaps them for
    # the selected arm below.
    arm0 = True if measure else bool(soc_resolved)

    # Build channels and G_l/G'_l tables from species projectors
    channels: list[ChannelMeta] = []
    channel_elements: list[str] = []
    G_rows: list[np.ndarray] = []
    Gp_rows: list[np.ndarray] = []
    beta_idx = 0

    for isp, sp in enumerate(species_list):
        natoms = int(species_natoms[isp])
        if natoms == 0:
            continue
        tau = species_tau[isp, :natoms]

        E_blocks = blocks_by_arm[arm0][sp.element]

        # Group projectors by l
        per_l: dict[int, list[int]] = {}
        for ip in range(sp.n_proj):
            l = int(sp.proj_l[ip])
            per_l.setdefault(l, []).append(ip)

        for l, proj_ids in per_l.items():
            E_np = E_blocks.get(l)
            if E_np is None:
                continue
            nbeta = len(proj_ids)
            msize = 2 * l + 1
            R = nbeta * msize

            # F_l(q) from pre-built tables → G_l = F_l/q^l.
            # Gp_l = dG_l/dq is obtained ANALYTICALLY via the Bessel-recurrence
            # formula  dG_l/dq = -H_{l+1}(β; q)/q^l  (see radial_tables.py).
            # H_{l+1} is the *raw* (unscaled) deriv Hankel pre-computed by
            # build_all_tables on GPU; we apply the q^l division here.  This
            # replaces an earlier central-FD derivative which was O(dq²)
            # biased for curved G_l(q) and led to ~10% errors in velocity
            # matrix elements taken via jax.jvp through _table_interp.
            for ip in proj_ids:
                F_vals = tables["proj_tables"][isp][ip]
                H_vals = tables["deriv_tables"][isp][ip]
                if l == 0:
                    G_vals = F_vals.copy()
                    Gp_vals = -H_vals
                else:
                    G_vals = np.empty(n_q, dtype=np.float64)
                    G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                    G_vals[0] = F_vals[1] / q_grid[1] ** l
                    Gp_vals = np.zeros(n_q, dtype=np.float64)
                    Gp_vals[1:] = -H_vals[1:] / q_grid[1:] ** l
                    # q=0 limit of dG_l/dq is 0 (j_{l+1}(qr) ~ q^{l+1}, so
                    # H_{l+1}/q^l → 0 as q → 0).
                G_rows.append(G_vals)
                Gp_rows.append(Gp_vals)

            channels.append(ChannelMeta(
                l=l, nbeta=nbeta, msize=msize, R=R,
                tau=tau, E=E_np, beta_table_start=beta_idx,
                natoms=natoms,
            ))
            channel_elements.append(sp.element)
            beta_idx += nbeta

    G_table = jnp.asarray(np.stack(G_rows), dtype=jnp.float64) if G_rows else jnp.zeros((0, n_q), dtype=jnp.float64)
    Gp_table = jnp.asarray(np.stack(Gp_rows), dtype=jnp.float64) if Gp_rows else jnp.zeros((0, n_q), dtype=jnp.float64)
    total_R = sum(ch.R * ch.natoms for ch in channels)
    l_max = max((ch.l for ch in channels), default=0)

    # ── Pre-flatten per-row metadata for vectorized Z assembly ──
    # Each row of Z corresponds to one (atom, beta, m) combination.
    # Flatten ALL channels into (total_R,) index arrays.
    row_beta_idx = []
    row_l = []
    row_m = []
    row_tau = []

    for ch in channels:
        for a in range(ch.natoms):
            for ib in range(ch.nbeta):
                for im in range(ch.msize):
                    row_beta_idx.append(ch.beta_table_start + ib)
                    row_l.append(ch.l)
                    row_m.append(im)
                    row_tau.append(ch.tau[a])

    row_beta_idx_j = jnp.asarray(row_beta_idx, dtype=jnp.int32)
    row_l_j = jnp.asarray(row_l, dtype=jnp.int32)
    row_m_j = jnp.asarray(row_m, dtype=jnp.int32)
    row_tau_np = np.array(row_tau, dtype=np.float64).reshape(-1, 3) if row_tau else np.zeros((0, 3), dtype=np.float64)
    row_tau_j = jnp.asarray(row_tau_np, dtype=jnp.float64)

    # ── Pre-build E_super (k-independent block-diagonal D-matrix) ──
    def _assemble_E_super(blocks_by_el: dict) -> jax.Array:
        E_super = np.zeros((nspinor, nspinor, total_R, total_R),
                           dtype=np.complex128)
        offset = 0
        for ch, el in zip(channels, channel_elements):
            E_np = np.asarray(blocks_by_el[el][ch.l])[:nspinor, :nspinor]
            R = ch.R
            for _a in range(ch.natoms):
                E_super[:, :, offset:offset + R, offset:offset + R] = E_np
                offset += R
        return jnp.asarray(E_super, dtype=jnp.complex128)

    setup = VNLSetup(
        channels=channels,
        dq=dq, n_q=n_q, q_max=q_max,
        G_table=G_table, Gp_table=Gp_table,
        prefactor=prefactor,
        B=B, cell_volume=cell_volume,
        total_R=total_R, nspinor=nspinor,
        E_super=_assemble_E_super(blocks_by_arm[arm0]), l_max=l_max,
        soc=bool(arm0),
        row_beta_idx=row_beta_idx_j,
        row_l=row_l_j, row_m=row_m_j, row_tau=row_tau_j,
    )

    # ── say HOW the mode was decided (one line, report-block ready) ──
    j_present = [el for el in live if pseudo_has_j_channels(pseudos[el])]
    if measure:
        # Arm 5: both candidates exist and nothing declared — measure.
        delta_d = max((pseudo_soc_strength_ry(pseudos[el])
                       for el in j_present), default=0.0)
        E_avg_super = _assemble_E_super(blocks_by_arm[False])
        soc_resolved, provenance = measure_soc_mode(
            wfn, setup, E_avg_super, setup.E_super, delta_d_ry=delta_d,
            caller="build_vnl_setup", print_fn=print_fn)
        if not soc_resolved:
            setup.E_super = E_avg_super
            for ch, el in zip(channels, channel_elements):
                ch.E = np.asarray(blocks_by_arm[False][el][ch.l])
        setup.soc = bool(soc_resolved)
    elif forced_line is not None:
        provenance = forced_line
    elif not j_present:
        provenance = ("spin-scalar (scalar-relativistic pseudopotentials — "
                      "no j channels to resolve)")
    elif getattr(wfn, "spinorbit", None) is not None:
        provenance = (f"j-{'RESOLVED' if setup.soc else 'AVERAGED'}, "
                      f"from QE <spinorbit>")
    else:
        provenance = "j-AVERAGED, forced by nspinor=1 (QE average_pp)"
    setup.soc_provenance = provenance
    return setup


# ---------------------------------------------------------------------------
# Table-lookup radial evaluation
# ---------------------------------------------------------------------------

@jax.jit
def _table_interp(q, dq, table):
    """Linear interpolation on uniform grid.

    q     : (nG,) query points
    dq    : scalar grid spacing
    table : (nbeta, n_q) values

    Returns (nbeta, nG).
    """
    n_q = table.shape[1]
    idx = jnp.floor(q / dq).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_q - 2)
    t = (q - idx * dq) / dq
    t = jnp.clip(t, 0.0, 1.0)
    return (1.0 - t)[None, :] * table[:, idx] + t[None, :] * table[:, idx + 1]


@jax.custom_jvp
def _interp_with_deriv(q, dq, table, deriv_table):
    """Linear interp of ``table`` at ``q``, with a custom JVP that uses
    ``deriv_table`` as the q-tangent instead of the forward-slope of the
    linear interpolation.

    Drop-in replacement for ``_table_interp(q, dq, table)`` whenever callers
    want ``jax.jvp`` / ``jax.jacfwd`` through the form factor to produce the
    **physical** derivative (velocity operator matrix elements, k·p response,
    ∂χ/∂q in the Sternheimer solver).  The physical dG/dq is tabulated at
    setup time via the Bessel-recurrence identity (see
    ``radial_tables.projector_deriv_table``) and passed in as
    ``deriv_table`` — this avoids the O(dq²) bias that afflicts the
    forward-slope derivative of the linear interpolant.
    """
    return _table_interp(q, dq, table)


@_interp_with_deriv.defjvp
def _interp_with_deriv_jvp(primals, tangents):
    q, dq, table, deriv_table = primals
    q_dot, _, _, _ = tangents
    val = _table_interp(q, dq, table)
    slope = _table_interp(q, dq, deriv_table)
    return val, slope * q_dot


# ---------------------------------------------------------------------------
# Per-k projector construction
# ---------------------------------------------------------------------------

def build_vnl_kdata(
    k_idx: int,
    setup: VNLSetup,
    wfn, sym, meta,
    *,
    compute_dZ: bool = False,
    gvectors=None,
) -> VNLKData:
    """Build dense Z [and dZ] for one k-point (SymMaps path).

    The G-list is the loader's FIXED ``(ngkmax, 3)`` table, so
    ``_assemble_Z_jit`` — whose compile cache is keyed on ``nG`` — lowers
    ONCE for the whole k sweep instead of once per distinct ``ngk``.

    Z and dZ are returned already ZEROED on the pad columns, and the mask
    that did it is carried on :attr:`VNLKData.g_mask`.  That is what makes
    this route safe to drop into a caller that does not mask ψ: every
    contraction in ``vnl_matrix`` / ``apply_vnl`` / ``vnl_velocity_matrix``
    passes through Z or dZ at least once, so a zero there is enough.
    (Left unmasked, the pad columns would be finite — the core evaluates
    Z at K = kvec on those rows — and would contract against ψ(Γ), which
    the pad rows alias.)

    Pass ``gvectors`` (a :class:`psp.dft_operators.PaddedGVectors`) to
    reuse one table across a sweep.
    """
    from psp.dft_operators import padded_gvectors

    tab = padded_gvectors(wfn, k="full_bz") if gvectors is None else gvectors
    G_pad, g_mask = tab.at(k_idx)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    kdata = _build_vnl_kdata_core(kvec, np.asarray(G_pad, dtype=int),
                                  setup, compute_dZ=compute_dZ)
    mask_j = jnp.asarray(g_mask, dtype=kdata.Z.real.dtype)
    return VNLKData(
        Z=kdata.Z * mask_j[None, :],
        E_super=kdata.E_super,
        nG=kdata.nG,
        total_R=kdata.total_R,
        dZ=None if kdata.dZ is None else kdata.dZ * mask_j[None, None, :],
        g_mask=g_mask,
    )


def build_vnl_kdata_from_kvec(
    kvec: np.ndarray,
    Gk_int: np.ndarray,
    setup: VNLSetup,
    crystal=None,
    meta=None,
    *,
    compute_dZ: bool = False,
) -> VNLKData:
    """Build dense Z [and dZ] from explicit k-vector + G-list (no SymMaps)."""
    return _build_vnl_kdata_core(np.asarray(kvec, dtype=float),
                                  np.asarray(Gk_int, dtype=int),
                                  setup, compute_dZ=compute_dZ)


def build_vnl_kdata_traced(kvec, Gk_int, setup: VNLSetup, *,
                           compute_dZ: bool = False) -> VNLKData:
    """:func:`build_vnl_kdata_from_kvec` for a TRACED ``kvec``/``Gk_int``.

    Same body, minus the ``np.asarray`` coercion of the two per-k
    operands — which is what forbids the eager entry point inside a
    trace: ``np.asarray`` on a ``lax.scan`` carry raises
    ``TracerArrayConversionError``.  The core is already pure jax in
    both branches (``_assemble_Z_jit`` for ``compute_dZ=False``, jnp +
    ``jax.jvp`` for the derivative), so nothing else has to change.

    The caller is ``common.mtxel_sweep``'s V_NL and dipole operators,
    which build Z (and dZ) for the scan's current k.  Everything the
    setup carries is k-independent and closes over as a constant; only
    ``kvec`` and the D10 fixed-shape G table vary per iteration, so the
    body lowers ONCE for the whole sweep.
    """
    return _build_vnl_kdata_core(kvec, Gk_int, setup, compute_dZ=compute_dZ)


@functools.partial(jax.jit, static_argnames=('l_max',))
def _assemble_Z_jit(
    kvec, Gk_int,
    B, dq, G_table, Gp_table, prefactor,
    row_beta_idx, row_l, row_m, row_tau,
    *, l_max,
):
    """JIT'd body of ``_build_vnl_kdata_core`` for ``compute_dZ=False``.

    Pulled to module scope so its compile cache is shared across every
    per-k call site (run_sternheimer, run_nscf Davidson, kpm_dos,
    get_dipole_mtxels).  Without this, each k re-traces the eager
    ``K_cart``/``q``/``S_all``/``G_all``/``Z`` arithmetic and emits a
    long tail of single-primitive XLA modules.

    All inputs are JAX arrays — ``setup`` is decomposed at the call site
    so the dataclass (a Python pytree, not abstractable positionally)
    doesn't enter the jit.  ``l_max`` is the only static arg (it
    controls the unrolled solid-harmonics block).
    """
    from psp.radial.solid_harmonics import all_solid_harmonics

    K_crys = Gk_int.astype(jnp.float64) + kvec[None, :]
    K_cart = K_crys @ B
    # Regularizer: avoids 1/q divergence in autodiff.  See
    # _build_vnl_kdata_core for the full physics rationale.
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + 1e-8)
    G_all = _interp_with_deriv(q, dq, G_table, Gp_table)        # (total_nbeta, nG)
    S_all = all_solid_harmonics(K_cart, l_max=l_max)            # (l_max+1, 2*l_max+1, nG)

    G_r = G_all[row_beta_idx]                                    # (total_R, nG)
    S_r = S_all[row_l, row_m]                                    # (total_R, nG)
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ row_tau.T)).T     # (total_R, nG)
    c_il_r = prefactor * (1j) ** row_l                           # (total_R,)
    return c_il_r[:, None] * G_r * S_r * phase_r                 # (total_R, nG)


def _build_vnl_kdata_core(
    kvec: np.ndarray,
    Gk_np: np.ndarray,
    setup: VNLSetup,
    *,
    compute_dZ: bool = False,
) -> VNLKData:
    """Build dense VNL projectors Z — fully vectorized, no Python loops.

    Z[r, G] = c_il * G_beta(q) * S_lm(K) * exp(-2πi K·τ)

    All per-row metadata (beta_idx, l, m, tau) was pre-flattened at setup time.

    Tail-G handling: if ``Gk_np`` is pre-padded by the caller (e.g.
    ``setup_H_k_from_kvec`` padding to ``ngkmax``), Z is computed at
    those padded entries too — at K = kvec (no zero-G in the padded
    rows), which gives finite, in-table values for q.  The caller is
    responsible for masking Z at padded entries before any contraction.
    """
    nG = Gk_np.shape[0]

    if not compute_dZ:
        Z = _assemble_Z_jit(
            jnp.asarray(kvec, dtype=jnp.float64),
            jnp.asarray(Gk_np, dtype=jnp.int32),
            jnp.asarray(setup.B, dtype=jnp.float64),
            jnp.asarray(setup.dq, dtype=jnp.float64),
            setup.G_table, setup.Gp_table,
            jnp.asarray(setup.prefactor, dtype=jnp.float64),
            setup.row_beta_idx, setup.row_l, setup.row_m, setup.row_tau,
            l_max=int(setup.l_max),
        )
        return VNLKData(Z=Z, E_super=setup.E_super, nG=nG,
                        total_R=setup.total_R, dZ=None)

    # ── compute_dZ=True path: still eager (used by get_dipole_mtxels).
    #    TODO: jit this too once the per-channel for-loop is vectorised.
    from psp.radial.solid_harmonics import all_solid_harmonics

    K_crys = jnp.asarray(Gk_np, dtype=jnp.float64) + jnp.asarray(kvec)[None, :]
    B_j = jnp.asarray(setup.B, dtype=jnp.float64)
    K_cart = K_crys @ B_j
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + 1e-8)

    G_all = _interp_with_deriv(q, setup.dq, setup.G_table, setup.Gp_table)
    S_all = all_solid_harmonics(K_cart, l_max=setup.l_max)

    G_r = G_all[setup.row_beta_idx]
    S_r = S_all[setup.row_l, setup.row_m]
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ setup.row_tau.T)).T
    c_il_r = setup.prefactor * (1j) ** setup.row_l

    Z = c_il_r[:, None] * G_r * S_r * phase_r             # (total_R, nG)

    # dZ for velocity (optional — still uses per-channel JVP, TODO: vectorize)
    dZ_j = None
    if compute_dZ:
        K_over_q = K_cart / q[:, None]
        Gp_all = _table_interp(q, setup.dq, setup.Gp_table)
        dZ_blocks = []
        for ch in setup.channels:
            l, nbeta, R = ch.l, ch.nbeta, ch.R
            tau_j = jnp.asarray(ch.tau, dtype=jnp.float64)
            c_il = setup.prefactor * (1j) ** l

            G_bG = G_all[ch.beta_table_start:ch.beta_table_start + nbeta]
            Gp_bG = Gp_all[ch.beta_table_start:ch.beta_table_start + nbeta]
            S = S_all[l, :2*l+1]
            phase = jnp.exp(-2j * jnp.pi * (K_crys @ tau_j.T)).T

            dS = jnp.stack([
                jax.jvp(lambda K: _solid_harmonics_jax(l, K),
                        (K_cart,), (jnp.zeros((nG, 3)).at[:, j].set(1.0),))[1]
                for j in range(3)
            ], axis=0)

            Binv = jnp.linalg.inv(B_j)
            dphase = -2j * jnp.pi * (tau_j @ Binv.T)[:, :, None] * phase[:, None, :]
            drad = Gp_bG[None, :, :] * K_over_q.T[:, None, :]
            core = drad[:, :, None, :] * S[None, None, :, :] + G_bG[None, :, None, :] * dS[:, None, :, :]
            dZ_core = c_il * phase[:, None, None, None, :] * core[None, :]
            radS = G_bG[:, None, :] * S[None, :, :]
            dZ_phase = c_il * radS[None, None, :, :, :] * dphase[:, :, None, None, :]
            dZ_blocks.append((dZ_core + dZ_phase).transpose(1, 0, 2, 3, 4).reshape(3, ch.natoms * R, nG))
        # ``compute_dZ=True`` PROMISES AN ARRAY, and the empty-channel case
        # used to break that promise silently.  A setup with no channels
        # (no pseudopotentials loaded, or none covering the structure)
        # produces ``dZ_blocks == []``; returning ``None`` there handed a
        # `NoneType` to ``apply_vnl_velocity_to_ket``, which conjugates its
        # ``dZ`` argument — so the failure surfaced ~30 s later as
        # ``TypeError: conjugate requires ndarray or scalar arguments`` six
        # frames inside a jitted einsum, naming neither the deck nor the
        # missing file.  ``Z`` already degrades gracefully in that case (it
        # is the ``(0, nG)`` empty projector matrix and every contraction
        # through it is zero); ``dZ`` now degrades the SAME way, at
        # ``(3, total_R, nG)`` with ``total_R == 0``.
        #
        # THIS IS THE SECOND LINE OF DEFENCE, NOT THE FIX.  A zero V_NL
        # velocity is the arithmetically correct answer for an empty
        # projector set and the wrong ANSWER for a real deck, which is why
        # the drivers must refuse the empty set up front —
        # ``psp.operator_checks.validate_operator_inputs``, now called by
        # ``get_dipole_mtxels`` as it already was by ``gw.kin_ion_io`` and
        # ``get_DFT_mtxels``.  What this branch buys is that the kernel's
        # documented contract is true for every caller, including the ones
        # that legitimately hold no projectors.
        dZ_j = (jnp.concatenate(dZ_blocks, axis=1) if dZ_blocks
                else jnp.zeros((3, int(setup.total_R), nG), dtype=Z.dtype))

    return VNLKData(
        Z=Z, E_super=setup.E_super, nG=nG,
        total_R=setup.total_R, dZ=dZ_j,
    )


# ---------------------------------------------------------------------------
# Core operators — single JIT, no loops
# ---------------------------------------------------------------------------

@jax.jit
def apply_vnl(psi_G, Z, E_super):
    """V_NL|psi> = Z E Z† |psi>.   (nvec, nspinor, nG) → same.

    Z       : (total_R, nG)
    E_super : (nspinor, nspinor, total_R, total_R)
    """
    P = jnp.einsum('RG,vsG->Rsv', jnp.conj(Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtv->Rsv', E_super, P, optimize=True)
    return jnp.einsum('RG,Rsv->vsG', Z, D, optimize=True)


@jax.jit
def vnl_matrix(psi_G, Z, E_super):
    """V_NL matrix elements <m|V_NL|n>.   Returns (nb, nb)."""
    P = jnp.einsum('RG,nsG->Rsn', jnp.conj(Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtn->Rsn', E_super, P, optimize=True)
    return jnp.einsum('Rsm,Rsn->mn', jnp.conj(P), D, optimize=True)


@jax.jit
def apply_vnl_velocity_to_ket(psi_G, Z, dZ, E_super):
    """∂V_NL/∂K_cart^α applied to a ket on the G-sphere.

    Returns ``(3, nb, nspinor, nG)`` complex — the velocity-applied ket
    ``v_NL^α |n,k⟩``  in the SAME G-sphere layout as ``psi_G``.

    Math:  ``∂V_NL/∂k^α = (∂Z^α) E Z† + Z E (∂Z^α)†``  (k-dependence
    flows through the projectors Z(k); E_super is k-independent).  The
    apply form leaves the bra index free so callers can either contract
    against ``conj(psi_G)`` to get the q=0 matrix element (see
    :func:`vnl_velocity_matrix`) OR against a different bra at a
    different k for the finite-q matrix element used in the SOS chi
    head/wing pipeline.
    """
    P  = jnp.einsum('RG,nsG->Rsn', jnp.conj(Z),  psi_G, optimize=True)   # ⟨Z|n⟩
    D  = jnp.einsum('stRQ,Qtn->Rsn', E_super, P, optimize=True)
    dP = jnp.einsum('jRG,nsG->jRsn', jnp.conj(dZ), psi_G, optimize=True)
    dD = jnp.einsum('stRQ,jQtn->jRsn', E_super, dP, optimize=True)
    # (∂Z^j) D — first piece in the symmetrized derivative
    t1 = jnp.einsum('jRG,Rsn->jnsG', dZ, D, optimize=True)
    # Z dD — second piece
    t2 = jnp.einsum('RG,jRsn->jnsG', Z,  dD, optimize=True)
    return t1 + t2


@jax.jit
def vnl_velocity_matrix(psi_G, Z, dZ, E_super):
    """⟨m | ∂V_NL/∂K_cart^α | n⟩ matrix elements at one k.  Returns (3, nb, nb).

    Thin bra-contraction wrapper around :func:`apply_vnl_velocity_to_ket`
    so the q=0 dipole path and the finite-q SOS path share the same
    underlying velocity application — no duplicated logic.
    """
    v_ket = apply_vnl_velocity_to_ket(psi_G, Z, dZ, E_super)            # (3, nb, ns, nG)
    return jnp.einsum('msG,jnsG->jmn', jnp.conj(psi_G), v_ket,
                       optimize=True)
