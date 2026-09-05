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

import dataclasses
import functools
import time
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from psp.radial.build_projectors_qe import (
    build_E_blocks_full, pseudo_has_j_channels, pseudo_soc_strength_ry,
)
from psp.radial.solid_harmonics import solid_harmonics_jax as _solid_harmonics_jax
from common.gauss_legendre import (
    GAUSS_LEGENDRE_INTERVAL_PROVENANCE,
    gauss_legendre_interval,
)
from runtime.padding import padded_axis


# Canonical Pauli matrices (the same tables ``common.gamma_matrices`` owns),
# for the channel-specific spin-orbit term of the velocity lift.
from common.gamma_matrices import paulis as _PAULI_JNP

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
    # The j-AVERAGED (scalar-relativistic) E blocks, kept beside the selected
    # ``E_super`` so a consumer can split V_NL = V_SR + V_SO exactly:
    # ``E_SR = E_super_scalar``, ``E_SO = E_super - E_super_scalar`` (zero
    # when ``soc`` is False, since then ``E_super`` IS the averaged arm).
    # ``None`` only when a fully-relativistic pseudopotential cannot be
    # j-averaged at all (``build_E_blocks_full(soc=False)`` refused), in
    # which case no SR/SO split exists and consumers that need one refuse.
    E_super_scalar: jax.Array | None = None
    # ── Pre-flattened per-row metadata for vectorized Z assembly ──
    # Each row r of Z(total_R, nG) is: c_il * G[beta_idx[r], q] * S[l[r], m[r], G] * phase[tau[r], G]
    row_beta_idx: jax.Array | None = None  # (total_R,) int — which G_table row
    row_l: jax.Array | None = None         # (total_R,) int — angular momentum
    row_m: jax.Array | None = None         # (total_R,) int — m index into S_all[l]
    row_tau: jax.Array | None = None       # (total_R, 3) float — atom position (crystal)
    # ``(start, stop, channel_index)`` for each independently coupled atom
    # block.  Low-memory kernels traverse these blocks without ever splitting
    # a canonical ``ChannelMeta.E`` matrix or inventing a second dense D.
    coupled_row_blocks: tuple[tuple[int, int, int], ...] = ()
    # Analytic second radial derivative of the reduced projector form factor.
    # Kept beside G/Gp so every VNL derivative consumes the same radial owner.
    Gpp_table: jax.Array | None = None     # (total_nbeta, n_q)
    # Analytic third radial derivative.  This is a separately priced
    # capability for the ICL q^2 transfer jet; uniform current/contact does
    # not allocate or compile it.
    Gppp_table: jax.Array | None = None    # (total_nbeta, n_q)
    # Content identity built from the host radial/projector/E data before
    # device transfer.  Empty only on hand-built test fixtures; production
    # uniform gauge transactions refuse an empty value.
    uniform_gauge_fingerprint: str = ""


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


@dataclass(frozen=True)
class _VNLProjectorCoefficientBlock:
    r"""Private in-memory ``<beta|psi>`` operator-derivative carrier.

    It has no G axis and never crosses a public/persistence boundary. A
    durable carrier would require a canonical k/G/mask/PP/SOC/band-window
    fingerprint, which no shared artifact contract owns yet.
    """

    c: jax.Array                    # (R, spin, band)
    dc_cart: jax.Array | None       # (cart, R, spin, band)
    d2c_cart: jax.Array | None      # (cart, cart, R, spin, band)
    E: jax.Array
    d3c_cart: jax.Array | None = None


@dataclass(frozen=True)
class VNLGaugeKetDerivatives:
    """Canonical VNL current/contact action on a two-component ket block."""

    gamma_cart_ket: jax.Array | None
    lambda_cart_ket: jax.Array | None
    third_cart_ket: jax.Array | None = None
    value_ket: jax.Array | None = None


ICL_STRAIGHT_GAUGE_PATH = "icl_straight_segment_v1"


@dataclass(frozen=True)
class ICLVNLTransferJet:
    r"""Straight-segment VNL photon jet for ``bra k-q, ket k``.

    All arrays are unscaled Pauli-Hamiltonian derivatives.  With
    ``V_a = d V_NL(k) / d k_a`` and Cartesian transfer ``q``, the
    Ismail-Beigi--Chang--Louie straight Wilson segment gives

    ``Gamma_a(k,q) = V_a - q_b V_ab/2 + O(q^2)``.

    Thus the transfer gradient is not a second current construction: it is
    exactly minus one half of the incumbent uniform contact.  The uniform
    current and contact fields remain byte-for-byte the arrays returned by
    :class:`VNLGaugeKetDerivatives`.  ``d2gamma_dq2_cart_ket`` is present only
    for a setup carrying the separately priced physical third projector
    derivative; differentiating the linear table interpolant is never used as
    a substitute.
    """

    gamma0_cart_ket: jax.Array
    dgamma_dq_cart_ket: jax.Array | None
    lambda0_cart_ket: jax.Array | None
    d2gamma_dq2_cart_ket: jax.Array | None = None


@dataclass(frozen=True)
class ICLVNLFiniteTransfer:
    """Exact-ICL action and its device-side endpoint-Ward certificate."""

    gamma_cart_ket: jax.Array
    ward_residual_abs: jax.Array
    ward_residual_rel: jax.Array
    ward_reference_norm: jax.Array
    certified: jax.Array
    tolerance_abs: float
    tolerance_rel: float
    path_order: int
    vnl_path_operator_fingerprint: str


@dataclass(frozen=True)
class ICLVNLFiniteContact:
    r"""Exact straight-path ``q,-q`` VNL two-photon contact action.

    ``lambda_cart_ket`` is the unscaled Pauli-Hamiltonian vertex.  The
    separately owned kinetic contact ``2 delta_ab`` is deliberately absent.
    The Ward certificate contracts the first Cartesian photon index.
    """

    lambda_cart_ket: jax.Array
    ward_residual_abs: jax.Array
    ward_residual_rel: jax.Array
    ward_reference_norm: jax.Array
    certified: jax.Array
    tolerance_abs: float
    tolerance_rel: float
    path_order: int
    vnl_path_operator_fingerprint: str


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
    compute_contact: bool = False,
    compute_transfer_q2: bool = False,
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

    compute_contact : bool
        Opt in to the extra analytic ``G''`` radial table needed by the
        uniform nonlocal contact.  False by default so existing Hamiltonian,
        NSCF, and dipole setup pays no l+2 Bessel compilation/pass and no
        uniform-gauge content-fingerprint pass.
    compute_transfer_q2 : bool
        Opt in to the analytic ``G'''`` table and l+3 Bessel family needed by
        the ICL straight-segment second transfer derivative.  This implies
        ``compute_contact`` because the q2 jet includes the same q0 contact.
        False leaves the uniform/current compile family unchanged.
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
                kvec = np.asarray(tab.kvecs[ik], dtype=float)
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
    compute_contact = bool(compute_contact or compute_transfer_q2)
    tables = build_all_tables(
        species_list, q_max, n_q,
        second_derivatives=compute_contact,
        third_derivatives=bool(compute_transfer_q2))
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
        if bool(soc_resolved):
            # The averaged arm is also wanted beside a declared j-resolved
            # operator: it is the scalar-relativistic half of the exact
            # V_SR + V_SO split (``VNLSetup.E_super_scalar``).  A pseudo
            # that cannot be averaged simply leaves that field None.
            try:
                blocks_by_arm[False] = _species_blocks(False)
            except ValueError:
                pass

    # Channels carry the arm-0 blocks; the measured path swaps them for
    # the selected arm below.
    arm0 = True if measure else bool(soc_resolved)

    # Build channels and G_l/G'_l tables from species projectors
    channels: list[ChannelMeta] = []
    channel_elements: list[str] = []
    G_rows: list[np.ndarray] = []
    Gp_rows: list[np.ndarray] = []
    Gpp_rows: list[np.ndarray] = []
    Gppp_rows: list[np.ndarray] = []
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
                Gpp_vals = (tables["second_deriv_tables"][isp][ip]
                            if compute_contact else None)
                Gppp_vals = (tables["third_deriv_tables"][isp][ip]
                             if compute_transfer_q2 else None)
                if l == 0:
                    G_vals = F_vals.copy()
                    Gp_vals = -H_vals
                else:
                    G_vals = np.empty(n_q, dtype=np.float64)
                    G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                    G_vals[0] = tables["reduced_origins"][isp][ip]
                    Gp_vals = np.zeros(n_q, dtype=np.float64)
                    Gp_vals[1:] = -H_vals[1:] / q_grid[1:] ** l
                    # q=0 limit of dG_l/dq is 0 (j_{l+1}(qr) ~ q^{l+1}, so
                    # H_{l+1}/q^l → 0 as q → 0).
                G_rows.append(G_vals)
                Gp_rows.append(Gp_vals)
                if compute_contact:
                    Gpp_rows.append(Gpp_vals)
                if compute_transfer_q2:
                    Gppp_rows.append(Gppp_vals)

            channels.append(ChannelMeta(
                l=l, nbeta=nbeta, msize=msize, R=R,
                tau=tau, E=E_np, beta_table_start=beta_idx,
                natoms=natoms,
            ))
            channel_elements.append(sp.element)
            beta_idx += nbeta

    G_table_np = (np.stack(G_rows).astype(np.float64, copy=False)
                  if G_rows else np.zeros((0, n_q), dtype=np.float64))
    Gp_table_np = (np.stack(Gp_rows).astype(np.float64, copy=False)
                   if Gp_rows else np.zeros((0, n_q), dtype=np.float64))
    Gpp_table_np = (
        (np.stack(Gpp_rows).astype(np.float64, copy=False)
         if Gpp_rows else np.zeros((0, n_q), dtype=np.float64))
        if compute_contact else None)
    G_table = jnp.asarray(G_table_np, dtype=jnp.float64)
    Gp_table = jnp.asarray(Gp_table_np, dtype=jnp.float64)
    Gpp_table = (None if Gpp_table_np is None
                 else jnp.asarray(Gpp_table_np, dtype=jnp.float64))
    Gppp_table_np = (
        (np.stack(Gppp_rows).astype(np.float64, copy=False)
         if Gppp_rows else np.zeros((0, n_q), dtype=np.float64))
        if compute_transfer_q2 else None)
    Gppp_table = (None if Gppp_table_np is None
                  else jnp.asarray(Gppp_table_np, dtype=jnp.float64))
    total_R = sum(ch.R * ch.natoms for ch in channels)
    l_max = max((ch.l for ch in channels), default=0)

    # ── Pre-flatten per-row metadata for vectorized Z assembly ──
    # Each row of Z corresponds to one (atom, beta, m) combination.
    # Flatten ALL channels into (total_R,) index arrays.
    row_beta_idx = []
    row_l = []
    row_m = []
    row_tau = []
    coupled_row_blocks = []
    row_offset = 0

    for channel_index, ch in enumerate(channels):
        for a in range(ch.natoms):
            block_start = row_offset
            for ib in range(ch.nbeta):
                for im in range(ch.msize):
                    row_beta_idx.append(ch.beta_table_start + ib)
                    row_l.append(ch.l)
                    row_m.append(im)
                    row_tau.append(ch.tau[a])
                    row_offset += 1
            coupled_row_blocks.append(
                (block_start, row_offset, channel_index))

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

    uniform_gauge_fingerprint = ""
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
        coupled_row_blocks=tuple(coupled_row_blocks),
        Gpp_table=Gpp_table,
        Gppp_table=Gppp_table,
        uniform_gauge_fingerprint=uniform_gauge_fingerprint,
    )

    if False in blocks_by_arm:
        setup.E_super_scalar = _assemble_E_super(blocks_by_arm[False])

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
    # Computed AFTER the soc measurement above, so the identity hashes
    # the operator ACTUALLY SHIPPED (a measured j-AVERAGED demotion
    # replaces setup.E_super and setup.soc; hashing the candidate would
    # authenticate an operator no consumer ever sees).
    if compute_contact:
        # Authenticate the exact host values before device transfer.  This is
        # the single PP/SOC/radial identity consumed by the uniform current,
        # contact, and Hall transaction; ordinary VNL/dipole setup neither
        # has the G'' capability nor pays this O(table-size) pass.
        import hashlib
        from common.parallel_transport import fingerprint_update_value

        digest = hashlib.sha256()
        digest.update(b"lorrax.vnl_uniform_gauge/v1\0")

        for label, value in (
            ("B_cart", B),
            ("cell_volume", np.float64(cell_volume)),
            ("grid", np.asarray((dq, n_q, q_max), dtype=np.float64)),
            ("prefactor", np.float64(prefactor)),
            ("nspinor_soc", np.asarray(
                (int(nspinor), int(bool(setup.soc))), dtype=np.int64)),
            ("G", G_table_np),
            ("Gp", Gp_table_np),
            ("Gpp", Gpp_table_np),
            ("row_beta", np.asarray(row_beta_idx, dtype=np.int32)),
            ("row_l", np.asarray(row_l, dtype=np.int32)),
            ("row_m", np.asarray(row_m, dtype=np.int32)),
            ("row_tau", row_tau_np),
            ("coupled_blocks", np.asarray(coupled_row_blocks, dtype=np.int64)),
            ("E_super", np.asarray(setup.E_super)),
        ):
            fingerprint_update_value(digest, label, value)
        if compute_transfer_q2:
            # Preserve the incumbent contact-only identity exactly.  The
            # additional radial capability joins the identity only when it
            # can affect the finite-transfer jet.
            fingerprint_update_value(digest, "Gppp", Gppp_table_np)
        setup.uniform_gauge_fingerprint = (
            "sha256:" + digest.hexdigest())

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


@jax.custom_jvp
def _interp_with_two_derivs(q, dq, table, deriv_table, second_deriv_table):
    """Table interpolation carrying two physical derivative levels."""
    return _table_interp(q, dq, table)


@_interp_with_two_derivs.defjvp
def _interp_with_two_derivs_jvp(primals, tangents):
    q, dq, table, deriv_table, second_deriv_table = primals
    q_dot, _, _, _, _ = tangents
    value = _table_interp(q, dq, table)
    slope = _interp_with_deriv(
        q, dq, deriv_table, second_deriv_table)
    return value, slope * q_dot


def _reduced_radial_values_on_cart(K_cart, dq, table):
    """Exact-origin reduced radial values shared by H and gauge vertices."""
    q2 = jnp.sum(K_cart * K_cart, axis=1)
    q_safe = jnp.sqrt(jnp.where(q2 > 0.0, q2, 1.0))
    q = jnp.where(q2 > 0.0, q_safe, 0.0)
    return _table_interp(q, dq, table)


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
    kvec = np.asarray(tab.kvecs[int(k_idx)], dtype=float)
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
    return _build_vnl_kdata_core(
        np.asarray(kvec, dtype=float), np.asarray(Gk_int, dtype=int),
        setup, compute_dZ=compute_dZ)


def build_vnl_kdata_traced(kvec, Gk_int, setup: VNLSetup, *,
                           compute_dZ: bool = False) -> VNLKData:
    """:func:`build_vnl_kdata_from_kvec` for a TRACED ``kvec``/``Gk_int``.

    Same body, minus the ``np.asarray`` coercion of the two per-k
    operands — which is what forbids the eager entry point inside a
    trace: ``np.asarray`` on a ``lax.scan`` carry raises
    ``TracerArrayConversionError``.  The core is already pure jax in
    both branches (``_assemble_Z_jit`` for ``compute_dZ=False``,
    ``_assemble_Z_dZ_jit`` for the derivative — a jit call inside a
    trace inlines), so nothing else has to change.

    The caller is ``common.mtxel_sweep``'s V_NL and dipole operators,
    which build Z (and dZ) for the scan's current k.  Everything the
    setup carries is k-independent and closes over as a constant; only
    ``kvec`` and the D10 fixed-shape G table vary per iteration, so the
    body lowers ONCE for the whole sweep.
    """
    return _build_vnl_kdata_core(kvec, Gk_int, setup, compute_dZ=compute_dZ)


@functools.partial(jax.jit, static_argnames=('l_max', 'chan_meta'))
def _assemble_Z_dZ_jit(
    kvec, Gk_int,
    B, dq, G_table, Gp_table, prefactor,
    row_beta_idx, row_l, row_m, row_tau, chan_taus,
    *, l_max, chan_meta,
):
    """JIT'd body of ``_build_vnl_kdata_core`` for ``compute_dZ=True``.

    Returns ``(Z, dZ)`` with shapes ``(total_R, nG)`` / ``(3, total_R, nG)``.

    The per-channel Python loop is NOT vectorised — it UNROLLS at trace
    time, which is the right cost model: ``chan_meta`` is one entry per
    (species, l) channel (3 for Si, ~n_species·(l_max+1) in general —
    tens at production scale, independent of natoms), and each entry's
    static ints (l, nbeta, natoms, beta_table_start, R) shape its block.
    ``chan_taus`` (one ``(natoms, 3)`` array per channel) rides along as
    a pytree operand so atom positions stay dynamic.

    WHY THIS EXISTS: the historical eager path re-traced three
    ``jax.jvp``s through the solid harmonics per channel on EVERY k —
    measured 48.9-51.5 ms/k on the Si 4x4x4 SR deck (vs 0.7 ms/k for the
    ``compute_dZ=False`` twin), i.e. ~3.3 s of pure re-trace over a
    64-point full-BZ finite-q dipole sweep.  Jitting with the loop
    unrolled lowers ONCE per (shape, chan_meta) and replays from the
    compile cache.  Same primitives in the same order as the eager block
    it replaces — only the dispatch changed — but XLA fusion moves ~26 %
    of Z/dZ elements by <=1.4e-16 (1-ulp class; owner accepted the shift
    2026-08-28).
    """
    from psp.radial.solid_harmonics import all_solid_harmonics

    nG = Gk_int.shape[0]
    K_crys = Gk_int.astype(jnp.float64) + kvec[None, :]
    K_cart = K_crys @ B
    # Regularizer: see _assemble_Z_jit.
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + 1e-8)

    G_all = _interp_with_deriv(q, dq, G_table, Gp_table)
    S_all = all_solid_harmonics(K_cart, l_max=l_max)

    G_r = G_all[row_beta_idx]
    S_r = S_all[row_l, row_m]
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ row_tau.T)).T
    c_il_r = prefactor * (1j) ** row_l
    Z = c_il_r[:, None] * G_r * S_r * phase_r             # (total_R, nG)

    K_over_q = K_cart / q[:, None]
    Gp_all = _table_interp(q, dq, Gp_table)
    dZ_blocks = []
    for (l, nbeta, natoms, beta_table_start, R), tau_j in zip(
            chan_meta, chan_taus):
        c_il = prefactor * (1j) ** l

        G_bG = G_all[beta_table_start:beta_table_start + nbeta]
        Gp_bG = Gp_all[beta_table_start:beta_table_start + nbeta]
        S = S_all[l, :2 * l + 1]
        phase = jnp.exp(-2j * jnp.pi * (K_crys @ tau_j.T)).T

        dS = jnp.stack([
            jax.jvp(lambda K, l=l: _solid_harmonics_jax(l, K),
                    (K_cart,), (jnp.zeros((nG, 3)).at[:, j].set(1.0),))[1]
            for j in range(3)
        ], axis=0)

        Binv = jnp.linalg.inv(B)
        dphase = -2j * jnp.pi * (tau_j @ Binv.T)[:, :, None] * phase[:, None, :]
        drad = Gp_bG[None, :, :] * K_over_q.T[:, None, :]
        core = drad[:, :, None, :] * S[None, None, :, :] + G_bG[None, :, None, :] * dS[:, None, :, :]
        dZ_core = c_il * phase[:, None, None, None, :] * core[None, :]
        radS = G_bG[:, None, :] * S[None, :, :]
        dZ_phase = c_il * radS[None, None, :, :, :] * dphase[:, :, None, None, :]
        dZ_blocks.append((dZ_core + dZ_phase).transpose(1, 0, 2, 3, 4).reshape(3, natoms * R, nG))
    dZ = jnp.concatenate(dZ_blocks, axis=1)
    return Z, dZ


@functools.partial(jax.jit, static_argnames=('l_max',))
def _assemble_Z_jit(
    kvec, Gk_int,
    B, dq, G_table, prefactor,
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
    K_crys = Gk_int.astype(jnp.float64) + kvec[None, :]
    K_cart = K_crys @ B
    G_all = _reduced_radial_values_on_cart(
        K_cart, dq, G_table)                                  # (total_nbeta, nG)
    return _assemble_projector_rows(
        K_crys, K_cart, G_all, prefactor,
        row_beta_idx, row_l, row_m, row_tau, l_max=l_max)


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
            setup.G_table,
            jnp.asarray(setup.prefactor, dtype=jnp.float64),
            setup.row_beta_idx, setup.row_l, setup.row_m, setup.row_tau,
            l_max=int(setup.l_max),
        )
        return VNLKData(Z=Z, E_super=setup.E_super, nG=nG,
                        total_R=setup.total_R, dZ=None)

    # ── compute_dZ=True path: jitted at module scope (_assemble_Z_dZ_jit),
    #    same math, static per-channel metadata.  Historically this branch
    #    ran EAGER — every call re-traced three jax.jvp's through the solid
    #    harmonics per channel, ~50 ms/k measured on the Si 4x4x4 SR deck
    #    (PERFORMANCE.md item 3) — and is 1-ulp-equivalent jitted.
    #
    # ``compute_dZ=True`` PROMISES AN ARRAY, and the empty-channel case
    # used to break that promise silently.  A setup with no channels
    # (no pseudopotentials loaded, or none covering the structure)
    # produces no dZ blocks; returning ``None`` there handed a
    # `NoneType` to ``apply_vnl_velocity_to_ket``, which conjugates its
    # ``dZ`` argument — so the failure surfaced ~30 s later as
    # ``TypeError: conjugate requires ndarray or scalar arguments`` six
    # frames inside a jitted einsum, naming neither the deck nor the
    # missing file.  ``Z`` already degrades gracefully in that case (it
    # is the ``(0, nG)`` empty projector matrix and every contraction
    # through it is zero); ``dZ`` degrades the SAME way, at
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
    if not setup.channels:
        Z = _assemble_Z_jit(
            jnp.asarray(kvec, dtype=jnp.float64),
            jnp.asarray(Gk_np, dtype=jnp.int32),
            jnp.asarray(setup.B, dtype=jnp.float64),
            jnp.asarray(setup.dq, dtype=jnp.float64),
            setup.G_table,
            jnp.asarray(setup.prefactor, dtype=jnp.float64),
            setup.row_beta_idx, setup.row_l, setup.row_m, setup.row_tau,
            l_max=int(setup.l_max),
        )
        return VNLKData(Z=Z, E_super=setup.E_super, nG=nG,
                        total_R=setup.total_R,
                        dZ=jnp.zeros((3, int(setup.total_R), nG),
                                     dtype=Z.dtype))

    chan_meta = tuple(
        (int(ch.l), int(ch.nbeta), int(ch.natoms),
         int(ch.beta_table_start), int(ch.R))
        for ch in setup.channels)
    chan_taus = tuple(jnp.asarray(ch.tau, dtype=jnp.float64)
                      for ch in setup.channels)
    Z, dZ_j = _assemble_Z_dZ_jit(
        jnp.asarray(kvec, dtype=jnp.float64),
        jnp.asarray(Gk_np, dtype=jnp.int32),
        jnp.asarray(setup.B, dtype=jnp.float64),
        jnp.asarray(setup.dq, dtype=jnp.float64),
        setup.G_table, setup.Gp_table,
        jnp.asarray(setup.prefactor, dtype=jnp.float64),
        setup.row_beta_idx, setup.row_l, setup.row_m, setup.row_tau,
        chan_taus,
        l_max=int(setup.l_max), chan_meta=chan_meta,
    )

    return VNLKData(
        Z=Z, E_super=setup.E_super, nG=nG,
        total_R=setup.total_R, dZ=dZ_j,
    )


# ---------------------------------------------------------------------------
# Uniform-gauge derivatives — canonical rows, bounded private coefficients
# ---------------------------------------------------------------------------

_FINITE_Q_GATE = "EM-VERTEX-FINITE-Q-WILSON"


def require_uniform_gauge_transfer(
    q_cart_bohr_inv=(0.0, 0.0, 0.0), *, caller: str,
) -> None:
    """One fail-closed boundary for the still-unbound finite-q VNL path."""
    q = np.asarray(q_cart_bohr_inv, dtype=np.float64)
    if q.shape != (3,):
        raise ValueError(
            f"{caller}: q_cart_bohr_inv must have shape (3,), got {q.shape}")
    if not np.array_equal(q, np.zeros(3, dtype=np.float64)):
        raise NotImplementedError(
            f"GATE {_FINITE_Q_GATE}: got q_cart_bohr_inv={q.tolist()}; "
            "only exact uniform q=0 is bound. A finite-q nonlocal "
            "pseudopotential vertex requires the repository-selected "
            "Wilson-line/path prescription.")


@jax.custom_jvp
def _interp_reduced_on_cart(
    K_cart, dq, table, dtable, d2table, d3table,
):
    """Exact-origin reduced radial table with physical Cartesian JVPs."""
    return _reduced_radial_values_on_cart(K_cart, dq, table)


@_interp_reduced_on_cart.defjvp
def _interp_reduced_on_cart_jvp(primals, tangents):
    K_cart, dq, table, dtable, d2table, d3table = primals
    dK_cart, _, _, _, _, _ = tangents
    q2 = jnp.sum(K_cart * K_cart, axis=1)
    q_safe = jnp.sqrt(jnp.where(q2 > 0.0, q2, 1.0))
    q = jnp.where(q2 > 0.0, q_safe, 0.0)
    value = _reduced_radial_values_on_cart(K_cart, dq, table)
    radial_prime = _interp_with_two_derivs(
        q, dq, dtable, d2table, d3table)
    # The primal is G''.  Its custom JVP is the physical G''' table when a
    # third Cartesian derivative is requested; contact-only traces consume
    # only the primal and therefore do not create a third-derivative family.
    radial_second = _interp_with_deriv(q, dq, d2table, d3table)
    radial_prime_over_q = jnp.where(
        q2[None, :] > 0.0,
        radial_prime / q_safe[None, :],
        radial_second)
    tangent = radial_prime_over_q * jnp.sum(
        K_cart * dK_cart, axis=1)[None, :]
    return value, tangent


def _assemble_projector_rows(
    K_crys, K_cart, G_all, prefactor,
    row_beta_idx, row_l, row_m, row_tau, *, l_max,
):
    """The single flattened production projector-row spelling."""
    from psp.radial.solid_harmonics import all_solid_harmonics

    S_all = all_solid_harmonics(K_cart, l_max=l_max)
    G_r = G_all[row_beta_idx]
    S_r = S_all[row_l, row_m]
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ row_tau.T)).T
    c_il_r = prefactor * (1j) ** row_l
    return c_il_r[:, None] * G_r * S_r * phase_r


def _assemble_uniform_projector_rows(
    k_crys, G_int, setup: VNLSetup,
    row_beta_idx, row_l, row_m, row_tau,
):
    """Gauge-differentiable view of the canonical flattened row owner."""
    B = jnp.asarray(setup.B, dtype=jnp.float64)
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]
    K_cart = K_crys @ B
    G_all = _interp_reduced_on_cart(
        K_cart, jnp.asarray(setup.dq), setup.G_table, setup.Gp_table,
        setup.Gpp_table,
        (setup.Gppp_table if setup.Gppp_table is not None
         else jnp.zeros_like(setup.Gpp_table)))
    return _assemble_projector_rows(
        K_crys, K_cart, G_all, jnp.asarray(setup.prefactor),
        row_beta_idx, row_l, row_m, row_tau, l_max=int(setup.l_max))


def _projector_derivatives_cartesian_rows(
    k_crys, G_chunk, setup: VNLSetup,
    row_beta_idx, row_l, row_m, row_tau, g_mask, row_mask,
    *, derivative_order: int = 2,
):
    """Z and Cartesian derivatives through requested order zero to three."""
    if int(derivative_order) not in (0, 1, 2, 3):
        raise ValueError("derivative_order must be 0, 1, 2, or 3")
    if int(derivative_order) == 3 and setup.Gppp_table is None:
        raise ValueError(
            "GATE EM-VERTEX-VNL-GPPP-MISSING: setup.Gppp_table got: None "
            f"for derivative_order={int(derivative_order)}; want: third "
            "radial derivatives from compute_transfer_q2=True; why: the "
            "third-order VNL vertex cannot be evaluated without them.")
    B = jnp.asarray(setup.B, dtype=jnp.float64)
    Binv = jnp.linalg.inv(B)

    def z_at_cart_shift(delta_cart):
        shifted_k = k_crys + delta_cart @ Binv
        return _assemble_uniform_projector_rows(
            shifted_k, G_chunk, setup,
            row_beta_idx, row_l, row_m, row_tau)

    zero = jnp.zeros((3,), dtype=jnp.float64)
    Z = z_at_cart_shift(zero)
    mask = (
        row_mask[:, None].astype(Z.real.dtype)
        * g_mask[None, :].astype(Z.real.dtype))
    Z = Z * mask
    if int(derivative_order) == 0:
        return (Z,)
    dZ = jnp.moveaxis(jax.jacfwd(z_at_cart_shift)(zero), -1, 0)
    through_first = (
        Z,
        dZ * mask[None, :, :],
    )
    if int(derivative_order) == 1:
        return through_first
    d2_raw = jax.jacfwd(jax.jacfwd(z_at_cart_shift))(zero)
    d2Z = jnp.moveaxis(d2_raw, (-2, -1), (0, 1))
    through_second = through_first + (
        d2Z * mask[None, None, :, :],
    )
    if int(derivative_order) == 2:
        return through_second
    d3_raw = jax.jacfwd(jax.jacfwd(jax.jacfwd(z_at_cart_shift)))(zero)
    d3Z = jnp.moveaxis(d3_raw, (-3, -2, -1), (0, 1, 2))
    return through_second + (
        d3Z * mask[None, None, None, :, :],
    )


def _contract_projector_coefficients(
    psi_G, Z, dZ, d2Z, E, d3Z=None,
):
    """Contract one G tile into the private low-rank coefficient carrier."""
    return _VNLProjectorCoefficientBlock(
        c=jnp.einsum("RG,nsG->Rsn", jnp.conj(Z), psi_G, optimize=True),
        dc_cart=(None if dZ is None else jnp.einsum(
            "aRG,nsG->aRsn", jnp.conj(dZ), psi_G, optimize=True)),
        d2c_cart=(None if d2Z is None else jnp.einsum(
            "abRG,nsG->abRsn", jnp.conj(d2Z), psi_G, optimize=True)),
        E=E,
        d3c_cart=(None if d3Z is None else jnp.einsum(
            "abcRG,nsG->abcRsn", jnp.conj(d3Z), psi_G,
            optimize=True)),
    )


def _coupled_projector_coefficients(block):
    """Apply the canonical PP/SOC E block once to c/dc/d2c."""
    E = block.E
    Ec = jnp.einsum("stRQ,Qtn->Rsn", E, block.c, optimize=True)
    Edc = (None if block.dc_cart is None else jnp.einsum(
        "stRQ,aQtn->aRsn", E, block.dc_cart, optimize=True))
    Ed2c = (None if block.d2c_cart is None else jnp.einsum(
        "stRQ,abQtn->abRsn", E, block.d2c_cart, optimize=True))
    return Ec, Edc, Ed2c


def _apply_vnl_current_from_coupled(Ec, Edc, Z, dZ):
    """Re-expand one VNL current action from canonical coupled coefficients."""
    return (
        jnp.einsum("aRG,Rsn->ansG", dZ, Ec, optimize=True)
        + jnp.einsum("RG,aRsn->ansG", Z, Edc, optimize=True))


def _apply_vnl_gauge_from_coefficients(block, Z, dZ, d2Z):
    """Re-expand current/contact only where a G-space action is requested."""
    Ec, Edc, Ed2c = _coupled_projector_coefficients(block)
    if Ed2c is None:
        raise ValueError("uniform contact requires second projector coefficients")
    gamma = _apply_vnl_current_from_coupled(Ec, Edc, Z, dZ)
    contact = (
        jnp.einsum("abRG,Rsn->abnsG", d2Z, Ec, optimize=True)
        + jnp.einsum("aRG,bRsn->abnsG", dZ, Edc, optimize=True)
        + jnp.einsum("bRG,aRsn->abnsG", dZ, Edc, optimize=True)
        + jnp.einsum("RG,abRsn->abnsG", Z, Ed2c, optimize=True))
    return VNLGaugeKetDerivatives(
        gamma_cart_ket=gamma, lambda_cart_ket=contact)


def _apply_vnl_third_from_coefficients(block, Z, dZ, d2Z, d3Z):
    r"""Apply ``d3 V_NL / dk_a dk_b dk_c`` from the same coefficients."""
    if block.d3c_cart is None:
        raise ValueError("third VNL action requires d3 projector coefficients")
    Ec, Edc, Ed2c = _coupled_projector_coefficients(block)
    Ed3c = jnp.einsum(
        "stRQ,abcQtn->abcRsn", block.E, block.d3c_cart, optimize=True)
    return (
        jnp.einsum("abcRG,Rsn->abcnsG", d3Z, Ec, optimize=True)
        + jnp.einsum("abRG,cRsn->abcnsG", d2Z, Edc, optimize=True)
        + jnp.einsum("acRG,bRsn->abcnsG", d2Z, Edc, optimize=True)
        + jnp.einsum("bcRG,aRsn->abcnsG", d2Z, Edc, optimize=True)
        + jnp.einsum("aRG,bcRsn->abcnsG", dZ, Ed2c, optimize=True)
        + jnp.einsum("bRG,acRsn->abcnsG", dZ, Ed2c, optimize=True)
        + jnp.einsum("cRG,abRsn->abcnsG", dZ, Ed2c, optimize=True)
        + jnp.einsum("RG,abcRsn->abcnsG", Z, Ed3c, optimize=True)
    )


def _coupled_projector_row_blocks(setup: VNLSetup, max_rows: int):
    """Validate and return each canonical coupled block exactly once."""
    if int(max_rows) <= 0:
        raise ValueError("projector_row_chunk must be positive")
    blocks = tuple(setup.coupled_row_blocks)
    if int(setup.total_R) and not blocks:
        raise ValueError(
            "GATE EM-VERTEX-VNL-ROW-PROVENANCE: coupled_row_blocks got: "
            f"empty for total_R={int(setup.total_R)}; want: canonical "
            "coupled row provenance; why: projector rows must stay aligned "
            "with their PP/SOC coupling blocks.")
    expected = 0
    normalized = []
    for raw_block in blocks:
        if len(raw_block) != 3:
            raise ValueError(
                "GATE EM-VERTEX-VNL-ROW-PROVENANCE: coupled_row_block got: "
                f"{raw_block!r}; want: (start, stop, channel_index); why: "
                "each projector row block must name its exact coupling "
                "channel.")
        block_start, block_stop, channel_index = raw_block
        start, stop = int(block_start), int(block_stop)
        ich = int(channel_index)
        if start != expected or stop <= start:
            raise ValueError(
                "GATE EM-VERTEX-VNL-ROW-COVERAGE: coupled_row_interval got: "
                f"({start}, {stop}); want: next interval starting at "
                f"{expected} with stop > start; why: blocks must cover "
                "[0, total_R) once without gaps or overlap.")
        if ich < 0 or ich >= len(setup.channels):
            raise ValueError(
                "GATE EM-VERTEX-VNL-E-PROVENANCE: channel_index got: "
                f"{ich}; want: in [0, {len(setup.channels)}); why: each "
                "projector row block must use its authenticated ChannelMeta.E.")
        expected_width = int(setup.channels[ich].R)
        if stop - start != expected_width:
            raise ValueError(
                "GATE EM-VERTEX-VNL-E-PROVENANCE: coupled_row_width got: "
                f"{stop - start} for channel={ich}; want: "
                f"ChannelMeta.R={expected_width}; why: the coupling matrix "
                "must address exactly that channel's projector rows.")
        if stop - start > int(max_rows):
            raise ValueError(
                "GATE EM-VERTEX-VNL-ROW-CHUNK: coupled_row_width got: "
                f"{stop - start}; want: <= projector_row_chunk="
                f"{int(max_rows)}; why: a PP/SOC E coupling block cannot be "
                "split across row chunks.")
        normalized.append((start, stop, ich))
        expected = stop
    if expected != int(setup.total_R):
        raise ValueError(
            "GATE EM-VERTEX-VNL-ROW-COVERAGE: final_row_stop got: "
            f"{expected}; want: total_R={int(setup.total_R)}; why: coupled "
            "blocks must cover every canonical projector row exactly once.")

    cursor = 0
    for start, stop, _ in normalized:
        if start != cursor or stop <= start:
            raise AssertionError(
                "internal VNL row packer produced a gap/overlap/duplicate")
        cursor = stop
    if cursor != int(setup.total_R):
        raise AssertionError("internal VNL row packer lost projector rows")
    return tuple(normalized)


def _compact_channel_couplings(setup: VNLSetup, row_width: int):
    """Stack canonical ``ChannelMeta.E`` blocks at one bounded scan shape.

    There is one compact entry per channel, not one ``total_R x total_R``
    matrix and not one duplicate per atom.  ``E_super`` remains the ordinary
    Hamiltonian owner's derived dense representation; this gauge action never
    pads or copies it.
    """
    blocks = []
    for ich, channel in enumerate(setup.channels):
        E = jnp.asarray(channel.E[:2, :2], dtype=jnp.complex128)
        R = int(channel.R)
        if E.shape != (2, 2, R, R):
            raise ValueError(
                "GATE EM-VERTEX-VNL-E-PROVENANCE: ChannelMeta.E.shape got: "
                f"{E.shape} for channel={ich}; want: (2, 2, {R}, {R}); "
                "why: Pauli spin and projector-row axes must match the "
                "authenticated channel metadata.")
        blocks.append(jnp.pad(
            E, ((0, 0), (0, 0), (0, row_width - R),
                (0, row_width - R))))
    return jnp.stack(blocks, axis=0)


def _apply_vnl_derivatives_between_g_carriers(
    psi_G,
    G_source_int,
    G_target_int,
    k_crys,
    setup: VNLSetup,
    g_mask_source,
    g_mask_target,
    *,
    derivative_order: int,
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
) -> VNLGaugeKetDerivatives:
    r"""One bounded source-coefficient/target-expansion derivative owner.

    ``psi_G`` must be the explicit two-component large-component block
    ``Psi_L=(band,2,G)``. Four-component input refuses; the named bispinor
    owner must slice once before entering this Pauli pseudopotential API.

    One fixed-shape outer ``lax.scan`` traverses packed complete E blocks.
    Two inner G scans first accumulate private ``c/dc/d2c`` and then
    re-expand the action. No full-G Z/dZ/d2Z or band-square matrix exists.
    ``derivative_order`` selects value only (0), current (1),
    current/contact (2), or current/contact/third action (3). Uniform supplies the same G
    carrier twice; exact finite transfer supplies the source and unwrapped
    target carriers.  The projector, coefficient, and re-expansion spelling is
    otherwise identical.
    """
    order = int(derivative_order)
    if order not in (0, 1, 2, 3):
        raise ValueError("derivative_order must be 0, 1, 2, or 3")
    psi = jnp.asarray(psi_G)
    G_source = jnp.asarray(G_source_int, dtype=jnp.int32)
    mask_source = jnp.asarray(g_mask_source, dtype=jnp.float64)
    G_target = jnp.asarray(G_target_int, dtype=jnp.int32)
    mask_target = jnp.asarray(g_mask_target, dtype=jnp.float64)
    if psi.ndim != 3 or int(psi.shape[1]) != 2:
        raise ValueError(
            "GATE EM-VERTEX-LARGE-COMPONENTS: psi_G.shape got: "
            f"{tuple(psi.shape)}; want: explicit large components with "
            "shape (band, 2, G); why: the Pauli VNL action is defined on "
            "the two-component large block only.")
    if int(setup.nspinor) != 2:
        raise ValueError(
            "GATE EM-VERTEX-PAULI-VNL: VNLSetup.nspinor got: "
            f"{int(setup.nspinor)}; want: 2; why: this nonlocal "
            "pseudopotential vertex uses the two-component Pauli coupling.")
    if order >= 2 and setup.Gpp_table is None:
        raise ValueError(
            "GATE EM-VERTEX-VNL-GPP-MISSING: setup.Gpp_table got: None "
            f"for derivative_order={order}; want: second radial derivatives "
            "from compute_contact=True; why: the VNL contact vertex cannot "
            "be evaluated without them.")
    if order == 3 and setup.Gppp_table is None:
        raise ValueError(
            "GATE EM-VERTEX-VNL-GPPP-MISSING: setup.Gppp_table got: None "
            f"for derivative_order={order}; want: third radial derivatives "
            "from compute_transfer_q2=True; why: the third-order VNL vertex "
            "cannot be evaluated without them.")
    if (setup.row_beta_idx is None or setup.row_l is None
            or setup.row_m is None
            or setup.row_tau is None):
        missing = [
            name for name in ("row_beta_idx", "row_l", "row_m", "row_tau")
            if getattr(setup, name) is None
        ]
        raise ValueError(
            "GATE EM-VERTEX-VNL-SETUP: canonical row metadata got: "
            f"missing={missing}; want: all row_beta_idx/row_l/row_m/row_tau "
            "arrays; why: the VNL action must preserve canonical projector "
            "row provenance.")
    if (G_source.shape != (psi.shape[-1], 3)
            or mask_source.shape != (psi.shape[-1],)):
        raise ValueError(
            "paired source G/mask/Psi_L mismatch: got "
            f"G={G_source.shape}, mask={mask_source.shape}, psi={psi.shape}")
    if (G_target.ndim != 2 or int(G_target.shape[1]) != 3
            or mask_target.shape != (G_target.shape[0],)):
        raise ValueError(
            "paired target G/mask mismatch: got "
            f"G={G_target.shape}, mask={mask_target.shape}")
    if int(g_chunk) <= 0:
        raise ValueError("g_chunk must be positive")

    nband, nG_source = int(psi.shape[0]), int(psi.shape[-1])
    nG_target = int(G_target.shape[0])
    gstep = int(g_chunk)
    source_carrier = padded_axis(
        nG_source, gstep, name="source nonlocal-projector G chunk").carrier
    target_carrier = padded_axis(
        nG_target, gstep, name="target nonlocal-projector G chunk").carrier
    source_pad = source_carrier - nG_source
    target_pad = target_carrier - nG_target
    psi_pad = jnp.pad(psi, ((0, 0), (0, 0), (0, source_pad)))
    G_source_pad = jnp.pad(G_source, ((0, source_pad), (0, 0)))
    mask_source_pad = jnp.pad(mask_source, (0, source_pad))
    G_target_pad = jnp.pad(G_target, ((0, target_pad), (0, 0)))
    mask_target_pad = jnp.pad(mask_target, (0, target_pad))
    source_chunks = source_carrier // gstep
    target_chunks = target_carrier // gstep
    psi_chunks = jnp.moveaxis(
        psi_pad.reshape(nband, 2, source_chunks, gstep), 2, 0)
    G_source_chunks = G_source_pad.reshape(source_chunks, gstep, 3)
    mask_source_chunks = mask_source_pad.reshape(source_chunks, gstep)
    G_target_chunks = G_target_pad.reshape(target_chunks, gstep, 3)
    mask_target_chunks = mask_target_pad.reshape(target_chunks, gstep)

    value_zero = (jnp.zeros((nband, 2, target_carrier), dtype=psi.dtype)
                  if order == 0 else None)
    gamma_zero = (jnp.zeros(
        (3, nband, 2, target_carrier), dtype=psi.dtype)
        if order >= 1 else None)
    contact_zero = (jnp.zeros(
        (3, 3, nband, 2, target_carrier), dtype=psi.dtype)
        if order >= 2 else None)
    third_zero = (jnp.zeros(
        (3, 3, 3, nband, 2, target_carrier), dtype=psi.dtype)
        if order == 3 else None)
    row_blocks = _coupled_projector_row_blocks(
        setup, int(projector_row_chunk))
    if not row_blocks:
        return VNLGaugeKetDerivatives(
            gamma_cart_ket=(None if gamma_zero is None
                            else gamma_zero[..., :nG_target]),
            lambda_cart_ket=(None if contact_zero is None
                             else contact_zero[..., :nG_target]),
            third_cart_ket=(None if third_zero is None
                            else third_zero[..., :nG_target]),
            value_ket=(None if value_zero is None
                       else value_zero[..., :nG_target]))

    row_width = max(stop - start for start, stop, _ in row_blocks)
    row_starts = jnp.asarray(
        [start for start, _, _ in row_blocks], jnp.int32)
    row_lengths = jnp.asarray(
        [stop - start for start, stop, _ in row_blocks], jnp.int32)
    row_channels = jnp.asarray(
        [channel for _, _, channel in row_blocks], jnp.int32)
    row_beta_padded = jnp.pad(setup.row_beta_idx, (0, row_width))
    row_l_padded = jnp.pad(setup.row_l, (0, row_width))
    row_m_padded = jnp.pad(setup.row_m, (0, row_width))
    row_tau_padded = jnp.pad(setup.row_tau, ((0, row_width), (0, 0)))
    channel_E = _compact_channel_couplings(setup, row_width)

    def row_pass(total, row_spec):
        row_start, row_length, row_channel = row_spec
        row_beta = jax.lax.dynamic_slice_in_dim(
            row_beta_padded, row_start, row_width, axis=0)
        row_l = jax.lax.dynamic_slice_in_dim(
            row_l_padded, row_start, row_width, axis=0)
        row_m = jax.lax.dynamic_slice_in_dim(
            row_m_padded, row_start, row_width, axis=0)
        row_tau = jax.lax.dynamic_slice_in_dim(
            row_tau_padded, row_start, row_width, axis=0)
        row_mask = jnp.arange(row_width, dtype=jnp.int32) < row_length
        E_block = channel_E[row_channel]
        E_block = E_block * (
            row_mask[None, None, :, None]
            * row_mask[None, None, None, :]).astype(E_block.dtype)
        coeff_zero = (jnp.zeros(
            (row_width, 2, nband), dtype=psi.dtype),)
        if order >= 1:
            coeff_zero = coeff_zero + (jnp.zeros(
                (3, row_width, 2, nband), dtype=psi.dtype),)
        if order >= 2:
            coeff_zero = coeff_zero + (jnp.zeros(
                (3, 3, row_width, 2, nband), dtype=psi.dtype),)
        if order == 3:
            coeff_zero = coeff_zero + (jnp.zeros(
                (3, 3, 3, row_width, 2, nband), dtype=psi.dtype),)

        def coefficient_pass(carry, xs):
            psi_part, G_part, mask_part = xs
            derivatives = _projector_derivatives_cartesian_rows(
                k_crys, G_part, setup, row_beta, row_l, row_m, row_tau,
                mask_part, row_mask, derivative_order=order)
            Z = derivatives[0]
            dZ = derivatives[1] if order >= 1 else None
            d2Z = derivatives[2] if order >= 2 else None
            d3Z = derivatives[3] if order == 3 else None
            part = _contract_projector_coefficients(
                psi_part, Z, dZ, d2Z, E_block, d3Z=d3Z)
            updated = (carry[0] + part.c,)
            if order >= 1:
                updated = updated + (carry[1] + part.dc_cart,)
            if order >= 2:
                updated = updated + (carry[2] + part.d2c_cart,)
            if order == 3:
                updated = updated + (carry[3] + part.d3c_cart,)
            return updated, None

        coefficient_arrays, _ = jax.lax.scan(
            coefficient_pass, coeff_zero,
            (psi_chunks, G_source_chunks, mask_source_chunks), unroll=1)
        coefficients = _VNLProjectorCoefficientBlock(
            c=coefficient_arrays[0],
            dc_cart=(coefficient_arrays[1] if order >= 1 else None),
            d2c_cart=(coefficient_arrays[2] if order >= 2 else None),
            E=E_block,
            d3c_cart=(coefficient_arrays[3]
                      if order == 3 else None))

        def expansion_pass(carry, xs):
            G_part, mask_part = xs
            derivatives = _projector_derivatives_cartesian_rows(
                k_crys, G_part, setup, row_beta, row_l, row_m, row_tau,
                mask_part, row_mask, derivative_order=order)
            Z = derivatives[0]
            if order == 0:
                Ec, _, _ = _coupled_projector_coefficients(coefficients)
                outputs = (jnp.einsum(
                    "RG,Rsn->nsG", Z, Ec, optimize=True),)
            elif order == 1:
                dZ = derivatives[1]
                Ec, Edc, _ = _coupled_projector_coefficients(coefficients)
                outputs = (_apply_vnl_current_from_coupled(
                    Ec, Edc, Z, dZ),)
            else:
                dZ = derivatives[1]
                out = _apply_vnl_gauge_from_coefficients(
                    coefficients, Z, dZ, derivatives[2])
                outputs = (out.gamma_cart_ket, out.lambda_cart_ket)
            if order == 3:
                outputs = outputs + (_apply_vnl_third_from_coefficients(
                    coefficients, Z, dZ, derivatives[2], derivatives[3]),)
            return carry, outputs

        _, expanded_chunks = jax.lax.scan(
            expansion_pass, None,
            (G_target_chunks, mask_target_chunks), unroll=1)
        if order == 0:
            value_chunks = expanded_chunks[0]
            value_block = jnp.transpose(
                value_chunks, (1, 2, 0, 3)).reshape(
                    nband, 2, target_carrier)
            updated_total = (total[0] + value_block,)
        else:
            gamma_chunks = expanded_chunks[0]
            gamma_block = jnp.transpose(
                gamma_chunks, (1, 2, 3, 0, 4)).reshape(
                    3, nband, 2, target_carrier)
            updated_total = (total[0] + gamma_block,)
        if order >= 2:
            contact_chunks = expanded_chunks[1]
            contact_block = jnp.transpose(
                contact_chunks, (1, 2, 3, 4, 0, 5)).reshape(
                    3, 3, nband, 2, target_carrier)
            updated_total = updated_total + (total[1] + contact_block,)
        if order == 3:
            third_chunks = expanded_chunks[2]
            third_block = jnp.transpose(
                third_chunks, (1, 2, 3, 4, 5, 0, 6)).reshape(
                    3, 3, 3, nband, 2, target_carrier)
            updated_total = updated_total + (total[2] + third_block,)
        return updated_total, None

    initial = (value_zero if order == 0 else gamma_zero,)
    if order >= 2:
        initial = initial + (contact_zero,)
    if order == 3:
        initial = initial + (third_zero,)
    totals, _ = jax.lax.scan(
        row_pass, initial,
        (row_starts, row_lengths, row_channels), unroll=1)
    return VNLGaugeKetDerivatives(
        gamma_cart_ket=(totals[0][..., :nG_target]
                        if order >= 1 else None),
        lambda_cart_ket=(totals[1][..., :nG_target]
                         if order >= 2 else None),
        third_cart_ket=(totals[2][..., :nG_target]
                        if order == 3 else None),
        value_ket=(totals[0][..., :nG_target]
                   if order == 0 else None))


def apply_uniform_vnl_derivatives_to_ket(
    psi_G,
    G_int,
    k_crys,
    setup: VNLSetup,
    g_mask,
    *,
    q_cart_bohr_inv=(0.0, 0.0, 0.0),
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
    compute_contact: bool = True,
    compute_third: bool = False,
) -> VNLGaugeKetDerivatives:
    r"""Apply uniform VNL Gamma/Lambda through the shared bounded core.

    Uniform response supplies the same source/target G carrier.  The exact
    finite-transfer path below supplies two carriers to that same core; there
    is no second projector, coefficient, or re-expansion implementation.
    """
    require_uniform_gauge_transfer(
        q_cart_bohr_inv, caller="apply_uniform_vnl_derivatives_to_ket")
    contact = bool(compute_contact or compute_third)
    return _apply_vnl_derivatives_between_g_carriers(
        psi_G, G_int, G_int, k_crys, setup, g_mask, g_mask,
        derivative_order=(3 if bool(compute_third) else (2 if contact else 1)),
        projector_row_chunk=int(projector_row_chunk), g_chunk=int(g_chunk))


def apply_icl_vnl_transfer_jet_to_ket(
    psi_G,
    G_int,
    k_crys,
    setup: VNLSetup,
    g_mask,
    *,
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
    include_contact: bool = True,
    include_q2: bool = False,
) -> ICLVNLTransferJet:
    r"""Apply the canonical ICL straight-segment VNL jet at ``q=0``.

    ``k_crys`` is the ket momentum in the repository's transition
    orientation ``<bra,k-q|Gamma(k,q)|ket,k>``.  In the positive raw
    Hamiltonian-vertex convention used by :mod:`common.mtxel_sweep`,

    .. math::

       \Gamma_a^{\rm NL}(k,q)
       = \int_0^1 d\lambda\,\partial_a V_{\rm NL}(k-\lambda q)
       = V_{,a}(k)-\frac{q_b}{2}V_{,ab}(k)
         +\frac{q_bq_c}{6}V_{,abc}(k)+O(q^3).

    The sole bounded projector-coefficient scan already produces both
    derivatives.  This wrapper only binds their finite-transfer meaning; it
    performs no second projector contraction and creates no persistent
    coefficient, G-space, or band-square carrier.
    """
    uniform = apply_uniform_vnl_derivatives_to_ket(
        psi_G,
        G_int,
        k_crys,
        setup,
        g_mask,
        projector_row_chunk=projector_row_chunk,
        g_chunk=g_chunk,
        compute_contact=bool(include_contact),
        compute_third=bool(include_q2),
    )
    return ICLVNLTransferJet(
        gamma0_cart_ket=uniform.gamma_cart_ket,
        dgamma_dq_cart_ket=(
            None if uniform.lambda_cart_ket is None
            else -0.5 * uniform.lambda_cart_ket),
        lambda0_cart_ket=uniform.lambda_cart_ket,
        d2gamma_dq2_cart_ket=(
            uniform.third_cart_ket / 3.0
            if bool(include_q2) else None),
    )


def _icl_vnl_path_operator_fingerprint(
    setup: VNLSetup,
    *,
    schema_tag: bytes,
    path_order: int,
    path_rtol: float,
    path_atol: float,
) -> str:
    """Bind one ICL vertex kind to the shared VNL/path quadrature owner."""
    import hashlib
    from common.parallel_transport import fingerprint_update_value

    vnl_fingerprint = str(setup.uniform_gauge_fingerprint).strip()
    if len(vnl_fingerprint) != 71 or not vnl_fingerprint.startswith("sha256:"):
        raise ValueError("exact ICL requires a fingerprinted VNLSetup")
    rtol, atol = float(path_rtol), float(path_atol)
    if not (np.isfinite(rtol) and rtol > 0.0
            and np.isfinite(atol) and atol >= 0.0):
        raise ValueError("path tolerances must be finite, rtol>0 and atol>=0")
    gauss_legendre_interval(int(path_order), 0.0, 1.0)  # validates order
    digest = hashlib.sha256()
    digest.update(schema_tag + b"\0")
    for label, value in (
        ("vnl", vnl_fingerprint),
        ("path", ICL_STRAIGHT_GAUGE_PATH),
        ("quadrature", GAUSS_LEGENDRE_INTERVAL_PROVENANCE),
        ("path_order", np.int64(path_order)),
        ("path_rtol", np.float64(rtol)),
        ("path_atol", np.float64(atol)),
    ):
        fingerprint_update_value(digest, label, value)
    return "sha256:" + digest.hexdigest()


def icl_vnl_finite_transfer_operator_fingerprint(
    setup: VNLSetup, *, path_order: int, path_rtol: float, path_atol: float,
) -> str:
    """Bind the canonical one-photon VNL path integration rule."""
    return _icl_vnl_path_operator_fingerprint(
        setup,
        schema_tag=b"lorrax.icl_vnl_finite_transfer/v1",
        path_order=path_order,
        path_rtol=path_rtol,
        path_atol=path_atol,
    )


def icl_vnl_finite_contact_operator_fingerprint(
    setup: VNLSetup, *, path_order: int, path_rtol: float, path_atol: float,
) -> str:
    """Bind the canonical ``q,-q`` two-photon VNL path integration rule."""
    return _icl_vnl_path_operator_fingerprint(
        setup,
        schema_tag=b"lorrax.icl_vnl_finite_contact/v1",
        path_order=path_order,
        path_rtol=path_rtol,
        path_atol=path_atol,
    )


def compute_icl_vnl_finite_transfer_to_ket(
    psi_G,
    G_source_int,
    G_target_int,
    k_source_crys,
    k_target_crys,
    q_crys,
    G_wrap,
    setup: VNLSetup,
    g_mask_source,
    g_mask_target,
    *,
    path_order: int = 12,
    path_rtol: float = 1.0e-10,
    path_atol: float = 1.0e-12,
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
) -> ICLVNLFiniteTransfer:
    r"""Apply ``integral_0^1 dlam dV_NL(k-lam*q)/dk`` to ``psi_G``.

    The target k point and integer wrap are canonical ``common.kq_mapping``
    outputs.  Source contraction and target expansion share the bounded VNL
    core above.  The result stays on device and is certified by
    ``q.Gamma = V(k)-V(k-q)``; a transaction owner must reduce/refuse once.
    """
    psi = jnp.asarray(psi_G)
    k_source = jnp.asarray(k_source_crys, dtype=jnp.float64)
    k_target = jnp.asarray(k_target_crys, dtype=jnp.float64)
    q = jnp.asarray(q_crys, dtype=jnp.float64)
    wrap_raw = jnp.asarray(G_wrap)
    if any(x.shape != (3,) for x in (k_source, k_target, q, wrap_raw)):
        raise ValueError("k_source/k_target/q/G_wrap must all have shape (3,)")
    if not jnp.issubdtype(wrap_raw.dtype, jnp.integer):
        raise ValueError(
            f"G_wrap must be the integer output of common.kq_mapping; got "
            f"dtype={wrap_raw.dtype}")
    wrap = wrap_raw.astype(jnp.int32)
    geometry_closed = jnp.max(jnp.abs(
        (k_source - q) - (k_target + wrap))) <= 2.0e-12

    G_source = jnp.asarray(G_source_int, dtype=jnp.int32)
    G_target_unwrapped = (
        jnp.asarray(G_target_int, dtype=jnp.int32) - wrap[None, :])
    source_mask = jnp.asarray(g_mask_source, dtype=jnp.float64)
    target_mask = jnp.asarray(g_mask_target, dtype=jnp.float64)
    nG_target = int(G_target_unwrapped.shape[0])

    B = jnp.asarray(setup.B, dtype=jnp.float64)
    def masked_radius(G_values, masks, k_value):
        radius = jnp.linalg.norm((G_values + k_value) @ B, axis=-1)
        return jnp.max(jnp.where(masks > 0.0, radius, 0.0))
    carriers = ((G_source.astype(jnp.float64), source_mask),
                (G_target_unwrapped.astype(jnp.float64), target_mask))
    max_radius = jnp.max(jnp.stack([
        masked_radius(G_values, mask, k_value)
        for G_values, mask in carriers for k_value in (k_source, k_source-q)]))
    radial_covered = max_radius <= jnp.asarray(
        float(setup.q_max) * (1.0 + 16.0 * np.finfo(np.float64).eps),
        dtype=jnp.float64)

    nodes_np, weights_np = gauss_legendre_interval(
        int(path_order), 0.0, 1.0)
    nodes = jnp.asarray(nodes_np, dtype=jnp.float64)
    weights = jnp.asarray(weights_np, dtype=jnp.float64)

    nband = int(psi.shape[0])
    gamma_zero = jnp.zeros((3, nband, 2, nG_target), dtype=psi.dtype)

    def path_pass(carry, xs):
        path_node, weight = xs
        gamma_node = _apply_vnl_derivatives_between_g_carriers(
            psi, G_source, G_target_unwrapped, k_source-path_node*q, setup,
            source_mask, target_mask, derivative_order=1,
            projector_row_chunk=projector_row_chunk,
            g_chunk=g_chunk).gamma_cart_ket
        return carry + weight.astype(gamma_node.real.dtype) * gamma_node, None

    gamma, _ = jax.lax.scan(
        path_pass, gamma_zero, (nodes, weights), unroll=1)
    def endpoint(k_value):
        return _apply_vnl_derivatives_between_g_carriers(
            psi, G_source, G_target_unwrapped, k_value, setup,
            source_mask, target_mask, derivative_order=0,
            projector_row_chunk=projector_row_chunk,
            g_chunk=g_chunk).value_ket
    endpoint_delta = endpoint(k_source) - endpoint(k_source-q)
    q_cart = q @ B
    ward_delta = jnp.einsum(
        "a,ansG->nsG", q_cart, gamma, optimize=True) - endpoint_delta
    error_abs = jnp.sqrt(jnp.real(jnp.vdot(ward_delta, ward_delta)))
    reference_norm = jnp.sqrt(jnp.real(jnp.vdot(
        endpoint_delta, endpoint_delta)))
    error_rel = error_abs / jnp.maximum(
        reference_norm, jnp.asarray(np.finfo(np.float64).tiny))
    certificate_limit = (
        jnp.asarray(float(path_atol), dtype=jnp.float64)
        + jnp.asarray(float(path_rtol), dtype=jnp.float64) * reference_norm)
    certified = jnp.logical_and(
        geometry_closed,
        jnp.logical_and(radial_covered, error_abs <= certificate_limit))
    return ICLVNLFiniteTransfer(
        gamma_cart_ket=gamma,
        ward_residual_abs=error_abs,
        ward_residual_rel=error_rel,
        ward_reference_norm=reference_norm,
        certified=certified,
        tolerance_abs=float(path_atol),
        tolerance_rel=float(path_rtol),
        path_order=int(path_order),
        vnl_path_operator_fingerprint=(
            icl_vnl_finite_transfer_operator_fingerprint(
                setup, path_order=path_order, path_rtol=path_rtol,
                path_atol=path_atol)),
    )


def compute_icl_vnl_finite_contact_to_ket(
    psi_G,
    G_int,
    k_crys,
    q_crys,
    setup: VNLSetup,
    g_mask,
    *,
    path_order: int = 12,
    path_rtol: float = 1.0e-10,
    path_atol: float = 1.0e-12,
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
) -> ICLVNLFiniteContact:
    r"""Apply the exact straight-path VNL contact for photons ``q,-q``.

    For the repository orientation and positive raw Hamiltonian vertex,

    .. math::

       \Lambda^{\rm NL}_{ab}(k;q,-q)
       &= \int_0^1 ds\int_0^1 dt\,
          V^{\rm NL}_{,ab}(k-(s-t)q) \\
       &= \int_0^1 du\,(1-u)\left[
          V^{\rm NL}_{,ab}(k-uq)+V^{\rm NL}_{,ab}(k+uq)\right].

    The second form is one fixed Gauss--Legendre scan through the incumbent
    bounded projector/coefficient/re-expansion core.  It is even in ``q``
    and tends to the uniform ``lambda_raw`` without a fitted long-wave
    limit.  Contracting the first photon index gives the exact path Ward
    identity

    .. math::

       q_a\Lambda_{ab}=\int_0^1du\,[V_{,b}(k+uq)-V_{,b}(k-uq)].

    ``q_crys`` is a crystal-coordinate photon transfer while both vertex
    indices and the Ward contraction are Cartesian.  The net transfer is
    zero, so this operator maps the ket ``k`` carrier back to the same
    carrier.  The kinetic contact remains owned by
    :func:`psp.dft_operators.apply_kinetic_contact_to_ket`.
    """
    psi = jnp.asarray(psi_G)
    G_values = jnp.asarray(G_int, dtype=jnp.int32)
    k = jnp.asarray(k_crys, dtype=jnp.float64)
    q = jnp.asarray(q_crys, dtype=jnp.float64)
    mask = jnp.asarray(g_mask, dtype=jnp.float64)
    if k.shape != (3,) or q.shape != (3,):
        raise ValueError("k_crys and q_crys must both have shape (3,)")

    fingerprint = icl_vnl_finite_contact_operator_fingerprint(
        setup,
        path_order=path_order,
        path_rtol=path_rtol,
        path_atol=path_atol,
    )
    nodes_np, weights_np = gauss_legendre_interval(
        int(path_order), 0.0, 1.0)
    nodes = jnp.asarray(nodes_np, dtype=jnp.float64)
    weights = jnp.asarray(weights_np, dtype=jnp.float64)

    nband, nG = int(psi.shape[0]), int(G_values.shape[0])
    contact_zero = jnp.zeros(
        (3, 3, nband, 2, nG), dtype=psi.dtype)
    current_difference_zero = jnp.zeros(
        (3, nband, 2, nG), dtype=psi.dtype)

    def path_pass(carry, xs):
        contact, current_difference = carry
        path_node, weight = xs
        minus = _apply_vnl_derivatives_between_g_carriers(
            psi, G_values, G_values, k - path_node * q, setup, mask, mask,
            derivative_order=2,
            projector_row_chunk=projector_row_chunk,
            g_chunk=g_chunk,
        )
        plus = _apply_vnl_derivatives_between_g_carriers(
            psi, G_values, G_values, k + path_node * q, setup, mask, mask,
            derivative_order=2,
            projector_row_chunk=projector_row_chunk,
            g_chunk=g_chunk,
        )
        real_weight = weight.astype(contact.real.dtype)
        triangle_weight = real_weight * (1.0 - path_node)
        return (
            contact + triangle_weight * (
                minus.lambda_cart_ket + plus.lambda_cart_ket),
            current_difference + real_weight * (
                plus.gamma_cart_ket - minus.gamma_cart_ket),
        ), None

    (contact, current_difference), _ = jax.lax.scan(
        path_pass,
        (contact_zero, current_difference_zero),
        (nodes, weights),
        unroll=1,
    )

    B = jnp.asarray(setup.B, dtype=jnp.float64)
    q_cart = q @ B
    ward_delta = jnp.einsum(
        "a,abnsG->bnsG", q_cart, contact, optimize=True
    ) - current_difference
    error_abs = jnp.sqrt(jnp.real(jnp.vdot(ward_delta, ward_delta)))
    reference_norm = jnp.sqrt(jnp.real(jnp.vdot(
        current_difference, current_difference)))
    error_rel = error_abs / jnp.maximum(
        reference_norm, jnp.asarray(np.finfo(np.float64).tiny))

    radius_at_endpoints = jnp.stack([
        jnp.linalg.norm((G_values + k + sign * q) @ B, axis=-1)
        for sign in (-1.0, 1.0)
    ])
    max_radius = jnp.max(jnp.where(
        mask[None, :] > 0.0, radius_at_endpoints, 0.0))
    radial_covered = max_radius <= jnp.asarray(
        float(setup.q_max) * (1.0 + 16.0 * np.finfo(np.float64).eps),
        dtype=jnp.float64)
    certificate_limit = (
        jnp.asarray(float(path_atol), dtype=jnp.float64)
        + jnp.asarray(float(path_rtol), dtype=jnp.float64) * reference_norm)
    certified = jnp.logical_and(
        radial_covered, error_abs <= certificate_limit)
    return ICLVNLFiniteContact(
        lambda_cart_ket=contact,
        ward_residual_abs=error_abs,
        ward_residual_rel=error_rel,
        ward_reference_norm=reference_norm,
        certified=certified,
        tolerance_abs=float(path_atol),
        tolerance_rel=float(path_rtol),
        path_order=int(path_order),
        vnl_path_operator_fingerprint=fingerprint,
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
    coefficients = _contract_projector_coefficients(
        psi_G, Z, dZ, None, E_super)
    D, dD, _ = _coupled_projector_coefficients(coefficients)
    # (∂Z^j) D — first piece in the symmetrized derivative
    t1 = jnp.einsum('jRG,Rsn->jnsG', dZ, D, optimize=True)
    # Z dD — second piece
    t2 = jnp.einsum('RG,jRsn->jnsG', Z, dD, optimize=True)
    return t1 + t2


def spin_orbit_split_E(setup: VNLSetup):
    """``(E_SR, E_SO)`` for ``setup``: the exact V_NL = V_SR + V_SO split.

    ``E_SR`` is the j-averaged (scalar-relativistic, spin-scalar) block
    matrix and ``E_SO = E_super - E_SR`` the spin-orbit remainder.  When
    the setup runs the averaged operator (``soc`` False) the remainder is
    exactly zero.  Refuses when the pseudopotential could not be averaged,
    because then no split exists.
    """
    if not bool(setup.soc):
        return setup.E_super, jnp.zeros_like(setup.E_super)
    if setup.E_super_scalar is None:
        raise ValueError(
            "GATE vnl_spin_orbit_split_unavailable: this VNLSetup runs a "
            "j-resolved V_NL whose pseudopotential could not be j-averaged, "
            "so V_NL = V_SR + V_SO has no scalar-relativistic half to split "
            "off.  The exact per-channel velocity lift needs that split.")
    return setup.E_super_scalar, setup.E_super - setup.E_super_scalar


def nonlocal_velocity_lift(setup: VNLSetup):
    """The loader's velocity-balance hook for one projector ``setup``.

    Returns ``f(psi_2, gvecs_int, kvecs_frac, ngk_valid, channel)`` mapping
    a k-batched two-component ψ ``(n_k, nb, 2, ngkmax)`` to the
    Rydberg-velocity ket the channel-``a`` small component needs,

        sum_b sigma^b (dV_SR/dk_b psi_L)  +  sigma^a (dV_SO/dk_a psi_L)

    with shape ``(n_k, nb, 2, ngkmax)``.  ``common.bispinor_init.
    lift_to_4spinor(representation="velocity_a")`` scales it by
    ``alpha/4`` and adds it to the ``sigma.p`` small component;
    ``WfnLoader.nonlocal_velocity_lift`` is where a driver attaches it.

    WHY THE SPLIT.  The four-spinor current of channel ``a`` is
    ``psi_L^dagger sigma^a psi_S + h.c.``.  A spin-SCALAR velocity may sit
    inside the sigma sandwich: ``sigma^a sigma^b V_b + V_b sigma^b sigma^a
    = 2 V_a`` because ``[sigma, V_SR] = 0``.  The spin-orbit part does not
    commute and the sandwich would add ``i eps_abc [sigma^c, dV_SO/dk_b]``,
    which is not part of dH/dk (measured 20% of |dV_NL/dk| on MoS2, more on
    Bi).  Placing it behind ``sigma^a`` instead uses ``sigma^a sigma^a = 1``
    and returns exactly ``psi_L^dagger dV_SO/dk_a psi_L + h.c.``.  So the
    channel-``a`` carrier reproduces ``<m| 2(k+G)_a + dV_NL/dk_a |n>``, the
    Hamiltonian's velocity, with no spurious term — at the price of one
    carrier per Cartesian channel.

    Pad columns: the loader's G table carries the FFT-box sentinel beyond
    ``ngk_valid`` and ψ is zero there.  The projectors are evaluated at Γ
    on those columns (so ``|k+G|`` stays inside the radial table) and the
    result is zeroed on them, because ``(dZ) E Z† psi`` is NOT zero on a
    column where only ψ vanishes.

    Memory: one ``lax.map`` over k, so the ``(3, nb, 2, ngkmax)`` transient
    and the ``(4, total_R, ngkmax)`` projector rows exist for one k at a
    time; the returned array is the size of ψ_L.
    """
    from common.bispinor_init import sigma_dot_cartesian_kets
    if int(setup.nspinor) != 2:
        raise ValueError(
            "nonlocal_velocity_lift requires a two-component (Pauli) "
            f"VNLSetup; got nspinor={int(setup.nspinor)}")
    E_SR, E_SO = spin_orbit_split_E(setup)
    has_so = bool(setup.soc)

    def _one_k(psi_k, gvec_k, kvec_k, ngk_k, channel):
        ngkmax = int(gvec_k.shape[0])
        gmask = jnp.arange(ngkmax, dtype=jnp.int32) < ngk_k
        gvec_safe = jnp.where(gmask[:, None], gvec_k, 0)
        kdata = build_vnl_kdata_traced(
            kvec_k, gvec_safe, setup, compute_dZ=True)
        v_sr = apply_vnl_velocity_to_ket(
            psi_k, kdata.Z, kdata.dZ, E_SR)                # (3, nb, 2, ngkmax)
        ket = sigma_dot_cartesian_kets(v_sr)
        if has_so:
            a = int(channel) - 1
            v_so_a = apply_vnl_velocity_to_ket(
                psi_k, kdata.Z, kdata.dZ[a:a + 1], E_SO)[0]  # (nb, 2, ngkmax)
            ket = ket + jnp.einsum(
                "ij,bjg->big", _PAULI_JNP[a], v_so_a, optimize=True)
        return ket * gmask.astype(ket.dtype)

    @functools.partial(jax.jit, static_argnames=("channel",))
    def _batch(psi_2, gvecs_int, kvecs_frac, ngk_valid, *, channel):
        channel = int(channel)
        if channel not in (1, 2, 3):
            raise ValueError(
                f"nonlocal_velocity_lift: channel must be 1, 2 or 3; got "
                f"{channel}")
        psi_L = jnp.asarray(psi_2)[:, :, :2, :]
        return jax.lax.map(
            lambda args: _one_k(*args, channel),
            (psi_L,
             jnp.asarray(gvecs_int, dtype=jnp.int32),
             jnp.asarray(kvecs_frac, dtype=jnp.float64),
             jnp.asarray(ngk_valid, dtype=jnp.int32)))

    return _batch


def nonlocal_velocity_lift_from_pseudo_dir(
    wfn, sym, meta, pseudo_dir, *, sys_dim: int | None, caller: str,
    print_fn=print,
):
    """Load ``*.upf`` from ``pseudo_dir``, build the projector setup, and
    return :func:`nonlocal_velocity_lift` for it.  The one preflight the
    kin_ion/dipole producers run (``psp.operator_checks``) runs here too when
    the caller knows its ``sys_dim`` (the GW deck does; the centroid selector
    has no Coulomb geometry and passes ``None``, keeping only the
    missing-projector refusal), so a deck asking for velocity balance without
    projectors refuses by name instead of lifting with ``V_NL = 0``."""
    from psp.pseudos import load_pseudopotentials
    from psp.operator_checks import validate_operator_inputs
    pseudos = load_pseudopotentials(str(pseudo_dir))
    if not pseudos:
        raise ValueError(
            "GATE bispinor_current_balance_needs_pseudopotentials: "
            "bispinor_current_balance = velocity lifts the spatial-current "
            "carrier with sigma.v, v = p + dV_NL/dk, and dV_NL/dk needs the "
            "run's projectors.\n"
            f"  got:  no *.upf in {str(pseudo_dir)!r}\n"
            "  want: the deck's pseudopotentials beside the input file, or "
            "pseudo_dir = <directory>\n"
            "  doc:  docs/input_reference.md, bispinor_current_balance.")
    if sys_dim is not None:
        validate_operator_inputs(pseudos, wfn, int(sys_dim), caller=caller)
    setup = build_vnl_setup(
        wfn, sym, meta, pseudos, nspinor=int(wfn.nspinor), print_fn=print_fn)
    return NonlocalVelocityLift(
        nonlocal_velocity_lift(setup),
        provenance=f"V_NL {setup.soc_provenance}")


@dataclasses.dataclass(frozen=True)
class NonlocalVelocityLift:
    """The loader's ``nonlocal_velocity_lift`` hook plus the one line that
    says which projectors it carries, for the consumer's own report."""
    batch: object
    provenance: str

    def __call__(self, psi_2, gvecs_int, kvecs_frac, ngk_valid, *, channel):
        return self.batch(psi_2, gvecs_int, kvecs_frac, ngk_valid,
                          channel=channel)


@jax.jit
def vnl_velocity_matrix(psi_G, Z, dZ, E_super):
    """⟨m | ∂V_NL/∂K_cart^α | n⟩ matrix elements at one k.  Returns (3, nb, nb).

    At q=0 the bra and ket share the same projector coefficients, so form
    the matrix directly as ``(dP)† D + P† dD`` with ``P = Z† psi`` and
    ``D = E P``.  This is exactly the bra contraction of
    :func:`apply_vnl_velocity_to_ket`, without materializing its
    ``(3, nb, nspinor, nG)`` output.  The apply-to-ket endpoint remains the
    owner for finite-q matrix elements, where the bra is a different state.
    """
    coefficients = _contract_projector_coefficients(
        psi_G, Z, dZ, None, E_super)
    D, dD, _ = _coupled_projector_coefficients(coefficients)
    return (
        jnp.einsum(
            'jRsm,Rsn->jmn', jnp.conj(coefficients.dc_cart), D,
            optimize=True)
        + jnp.einsum(
            'Rsm,jRsn->jmn', jnp.conj(coefficients.c), dD,
            optimize=True))
