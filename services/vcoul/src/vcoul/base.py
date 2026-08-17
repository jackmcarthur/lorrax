"""Coulomb-kernel dispatcher: SysDim enum, abstract base, the v(q+G) driver.

A caller wants two things from "the Coulomb interaction" given a system
dimensionality:

1. ``v_qG(geometry, qvec, comps_qG)`` — V(q+G) on the per-q sphere, with
   q+G=0 zeroed (the head is added back as a separate rank-1 term).
2. ``q0_average(geometry, kgrid, ...)`` — the (vc0_mean, wcoul0) pair at
   q→0, typically by Monte-Carlo over the mini-BZ Voronoi cell.

Each dimension lives in its own module (:mod:`~vcoul.bulk_3d`,
:mod:`~vcoul.slab_2d`, :mod:`~vcoul.box_0d`); :func:`get_kernel` picks one
off an int or a :class:`SysDim`.  The shared mini-BZ sampler lives in
:mod:`~vcoul.minibz` so 2D and 3D don't duplicate it.

WHAT CHANGED AT THE SERVICE BOUNDARY.  These signatures used to take a
``wfn`` loader object and a lorrax ``common.Meta``, and reached into them
for ``blat``, ``bvec``, ``cell_volume``, ``bdot``, ``fft_grid``,
``nkx/nky/nkz`` and ``sys_dim``.  They take a
:class:`~vcoul.geometry.CoulombGeometry` and an explicit ``kgrid``
instead — the same numbers, named once, with ``blat * bvec`` taken in one
place rather than five.  The lorrax-side wrappers
(``gw.compute_vcoul``, ``gw.coulomb.*``, ``gw.vcoul``) do the
translation and keep their old wfn/meta-facing spellings.
"""
from __future__ import annotations

import enum
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.geometry import CoulombGeometry

__all__ = ["SysDim", "CoulombKernel", "get_kernel", "v_qG_table",
           "v_qG_single", "HeadSlotTable", "head_slot_table",
           "report_head_tensor_asymmetry"]


def report_head_tensor_asymmetry(S_cart, where: str,
                                 rtol: float = 1e-10) -> float:
    """Report transpose asymmetry before a longitudinal ``q^T S q`` use.

    Real-q quadratic forms cannot see ``S - S.T``.  The residual is therefore
    a sampling diagnostic, not a precondition on the physical transverse
    response.  Values above ``rtol`` are announced as warnings but never
    refused; the contraction sites explicitly project the longitudinal part.

    Parameters
    ----------
    S_cart : array_like, shape (3, 3)
        Cartesian q-squared response tensor.
    where : str
        Contraction site named in the diagnostic.
    rtol : float, optional
        ``||S-S.T||_F / ||S||_F`` threshold for a sampling warning.

    Returns
    -------
    float
        The reported relative transpose asymmetry.
    """
    S = np.asarray(S_cart, dtype=np.complex128)
    if S.shape != (3, 3):
        raise ValueError(f"{where}: S_cart must be (3,3); got {S.shape}")
    scale = float(np.linalg.norm(S))
    asym = float(np.linalg.norm(S - S.T))
    rel = asym / scale if scale > 0.0 else 0.0
    warning = "WARNING: " if rel > rtol else ""
    print(
        f"  {warning}S_cart transpose asymmetry [{where}]: "
        f"||S-S.T||_F/||S||_F = {rel:.6e}",
        flush=True,
    )
    return rel


class SysDim(int, enum.Enum):
    """System dimensionality for Coulomb truncation.

    Stored as int so legacy ``cohsex.in`` values (0/2/3) keep parsing
    cleanly through ``int(...)``; comparisons against literal ints
    still work (``sys_dim == 3``).
    """
    BULK_3D = 3
    SLAB_2D = 2
    BOX_0D = 0


class CoulombKernel(Protocol):
    """One implementation per dimensionality.

    All kernels return arrays with the same q+G=0-zeroed convention
    (head is injected separately via ``gw.head_correction.HeadResolver``).
    Volume-factor convention: outputs are in Rydberg with the
    BerkeleyGW factor ``v_q(G) · (1/Ω_cell)`` already applied so the
    downstream ``ζ Vc(G) ζ†`` contraction comes out in Ry directly.

    A kernel implements exactly ONE arithmetic method,
    :meth:`_v_bare_per_q` — the dimension's bare formula at one q.
    Everything dimension-independent (the q loop, the ``vcoul_cutoff_ry``
    mask, the ``argmin |q+G|`` head-slot injection, dtype and shape
    validation) lives once in :func:`v_qG_table` and is shared by all
    three.  ``v_qG`` and
    the production entry point ``gw.compute_vcoul.compute_v_q_per_G``
    are both thin wrappers over that one driver, so there is a single
    implementation of ``v(q+G)`` per dimensionality.
    """
    sys_dim: SysDim

    def _v_bare_per_q(self, qf, gvec_q, *, bvec_f, fact,
                      bdot=None, fft_grid=None):
        """Bare ``v(q+G)·fact`` and ``|q+G|²`` at ONE q.

        ``qf`` — (3,) fractional q.  ``gvec_q`` — (3, nG) float64 Miller
        indices.  Returns ``(v, denom)``, both ``(nG,)`` float64:

        * ``v``     — ``v(q+G) / Ω_cell``, already zeroed where
          ``|q+G|² < 1e-12`` (the q+G=0 slot; the head is a separate
          rank-1 term).
        * ``denom`` — ``|q+G|²`` in Ry, the operand the shared
          ``vcoul_cutoff_ry`` mask compares against.

        This stays a RAW-array method rather than a geometry-taking one:
        it is called once per q inside :func:`v_qG_table`'s loop, and the
        four things it needs are already unpacked there.
        """
        ...

    def v_qG(self, geometry: CoulombGeometry, qvec_wrapped,
             comps_qG) -> jax.Array:
        """V_q(G) on the per-q G-vector list, length nG, with q+G=0 zeroed."""
        ...

    def q0_average(
        self, geometry: CoulombGeometry, kgrid, *,
        S_cart: jnp.ndarray | None = None,
        epshead: jnp.ndarray | None = None,
        static_kappa2: jnp.ndarray | None = None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(vc0_mean, wcoul0)`` averaged over the mini-BZ Voronoi cell.

        - ``S_cart`` (preferred):  ``wcoul0 = ⟨v(q) / (1 - v(q) qᵀSq)⟩``,
          using the same sample points as ``vc0_mean`` (anisotropic).
        - ``epshead`` (fallback): historical Ismail-Beigi gamma model.
          Less accurate; kept for back-compat with older runs.
        - ``static_kappa2`` (3D metal): Thomas-Fermi
          ``wcoul0 = <8*pi/(q^2+kappa^2)>``.  This is a distinct order of
          limits from the finite-frequency ``S_cart`` model.

        ``Box0D`` ignores everything except ``geometry`` (and the
        ``epshead`` slot, unused) — V(q=G=0) is finite from the cell-box
        FFT.
        """
        ...


def get_kernel(sys_dim) -> CoulombKernel:
    """Return the :class:`CoulombKernel` for the given dimensionality.

    Accepts either :class:`SysDim` or a raw int (0/2/3).  Default is 3D
    when ``sys_dim`` is None or unset; any other value is an error.
    """
    if sys_dim is None:
        sys_dim = SysDim.BULK_3D
    try:
        sd = SysDim(int(sys_dim))
    except ValueError:
        raise ValueError(
            f"sys_dim={sys_dim!r} invalid; expected 0 (box), 2 (slab), "
            f"or 3 (bulk)."
        )
    if sd is SysDim.BULK_3D:
        from vcoul.bulk_3d import Bulk3D
        return Bulk3D()
    if sd is SysDim.SLAB_2D:
        from vcoul.slab_2d import Slab2D
        return Slab2D()
    if sd is SysDim.BOX_0D:
        from vcoul.box_0d import Box0D
        return Box0D()
    raise AssertionError(f"unhandled SysDim: {sd}")  # pragma: no cover


# ---------------------------------------------------------------------------
# The single v(q+G) driver.  Everything dimension-INDEPENDENT lives here
# exactly once; each kernel contributes only ``_v_bare_per_q``.
# ---------------------------------------------------------------------------

def v_qG_table(
    kernel,
    q_irr_frac,
    gvec_components,
    *,
    geometry: CoulombGeometry,
    vcoul_cutoff_ry: float | None = None,
    v_head_fn=None,
    head_tie_rtol: float = 1e-9,
) -> np.ndarray:
    """``v(q+G)`` at the writer's per-q WFN.h5-style G-sphere, for every q.

    Returns ``(n_q, ngkmax)`` float64.  ``gvec_components`` is
    ``(n_q, 3, ngkmax)`` Miller indices (``isdf_header/gvec_components``).

    This is the ONE place the three dimension-independent capabilities
    are implemented, in this order (the order is load-bearing — the head
    slot is injected BEFORE the cutoff mask, so a head slot outside the
    bare-Coulomb cutoff is zeroed like any other G):

    1. **``v_head_fn``** — mini-BZ head-slot injection.  When given, a
       callable ``K_cart (m, 3) -> (m,) float64`` that returns the
       mini-BZ-averaged Coulomb head at each Cartesian ``q+G``.  It is
       evaluated at every slot attaining ``argmin |q+G|²`` (see HEAD SLOT
       below) and those slots' ``v`` are replaced.  q with
       ``min |q+G|² < 1e-12`` (i.e. Γ) are SKIPPED — that head is the
       separate rank-1 Σ_X term, and ``_v_bare_per_q`` already zeroes it.
    2. **``vcoul_cutoff_ry``** — zero ``v`` wherever ``|q+G|² >`` cutoff.
       This is V_q's bare-Coulomb cutoff, which may be *smaller* than the
       ζ-sphere cutoff that built ``gvec_components``.
    3. dtype/shape: float64 ``(n_q, ngkmax)``.  Pad slots (sentinel
       Miller ``(nx/2, ny/2, nz/2)``) get whatever ``v`` is at that
       position — callers need not zero them because the V_q contract
       carries ζ̃ = 0 there.

    HEAD SLOT — why ``argmin |q+G|²``, and why ALL of the argmin.

    The rule that shipped before 2026-08-08 selected the slot whose Miller
    index is literally ``(0,0,0)``.  That label is NOT equivariant under
    q → −q.  The BGW wrap sends a boundary component to ``+1/2``, never
    ``−1/2``, so the G-list pairing between +q and −q is
    ``G ↦ −G − s`` with ``s = round(q_frac[−q] + q_frac[q])``; for
    ``s ≠ 0`` the Miller-(0,0,0) slot at +q pairs with (0,0,−1) /
    (−1,−1,0) / (−1,−1,−1) at −q, which carries the BARE value.  MEASURED
    consequence on Si 4×4×4: ``max |v(+q, i) − v(−q, pair(i))|`` sat at
    1.293e−2 with the label rule and is EXACTLY 0.0 with this one, and
    ``||V_q − conj(V_{−q})||`` sat at 6.0e−3 against an arithmetic floor
    of 1.16e−7.

    ``argmin |q+G|²`` fixes that for free, because the pairing above is
    exactly ``K ↦ −K`` in Cartesian (MEASURED: ``max |K_i(+q) +
    K_pair(i)(−q)| = 0.000e+00`` over all 64 q), and ``|K|`` is invariant
    under it.  So the argmin SET at +q maps onto the argmin SET at −q.

    But the argmin is often DEGENERATE, and then "one slot" is not well
    defined.  MEASURED over all 64 q of the Si fcc 4×4×4 fixture, stable
    to an identical histogram across rel tolerances 1e−14 … 1e−6:

        multiplicity 1 → 51 q,  multiplicity 2 → 7 q,  multiplicity 4 → 6 q

    and the smallest relative gap to the next distinct ``|q+G|²`` among
    the non-degenerate q is 7.3e−1 — these are exact symmetry
    degeneracies, not knife-edges.  Worse, 7 of those 64 q are
    SELF-PAIRED (``−q ≡ q`` as a grid index, e.g. ``q=(0,0,1/2)`` whose
    tied slots are ``G=(0,0,0)`` and ``G=(0,0,−1)``) and the pairing
    SWAPS the two tied slots.  At such a q, injecting at one slot and not
    its partner makes ``V_q ≠ conj(V_q)`` by construction: **no
    single-slot rule can be equivariant there**, whatever the tie-break.
    Every even-numbered k-grid tested (fcc/sc/bcc/hex/triclinic) has
    exactly the same 7 self-paired q; odd grids have none.

    Hence: inject at ALL slots attaining the argmin, each valued from its
    OWN Cartesian ``q+G``.  ``head_tie_rtol`` (default 1e−9) is the
    relative window on ``|q+G|²`` defining "attaining"; it sits ~5 decades
    above float64 noise on the dot product and ~6 below the smallest real
    gap measured above, and the tie count is flat across that whole range.

    Equivariance then needs one more thing from the CALLER: ``v_head_fn``
    must satisfy ``f(K) = f(−K)``.  For a Monte-Carlo mini-BZ average
    that means a CENTROSYMMETRIC δq sample set — the mini-BZ is the
    k-grid parallelepiped, not a sphere, so ``⟨v(K+δq)⟩`` and
    ``⟨v(−K+δq)⟩`` agree only when ``{δq} = {−δq}``.
    :func:`~vcoul.minibz.build_v_head_miniBZ_fn_3d` guarantees it.

    TIED-SET VALUE.  When the argmin ties, every tied slot is given the
    MEAN of ``v_head_fn`` over the tied set, not its own value.  For the
    51 q with a unique argmin the set has one member and this is a no-op.

    Two DIFFERENT symmetries have to hold at once, and only the mean
    satisfies both:

    * ``q → −q`` (the pairing, ``K ↦ −K``) — needed by the direct
      full-BZ arm.  Per-slot values already satisfy this, because
      ``f(K) = f(−K)`` and the tied set at −q is the negation of the
      tied set at +q.
    * ``K → R K`` for R in the little group — needed by the IBZ arm,
      where ``unfold_v_q`` builds ``V_{−q}`` from ``V_q`` by a point-group
      operation that PERMUTES the tied slots among themselves.  Per-slot
      values do NOT satisfy this: the tied slots have equal ``|K|`` but
      ``⟨v⟩_miniBZ`` distinguishes them, because the mini-BZ is a
      parallelepiped whose symmetry is lower than the crystal's.  A
      spherical mini-BZ would give them all the same number; the spread
      is a sampling-cell artifact, and averaging over the tied orbit is
      what removes it.

    MEASURED on armB (orbit-closed IBZ + unfold, Si 4×4×4), per-q
    ``||V_q − conj(V_{−q})||_F / ||V_q||_F``, floor 1.27e−7:

        shipped Miller-(0,0,0) label   9 q above 1e−5, worst 7.459e−3
        argmin, per-slot value         6 q above 1e−5, worst 7.709e−5
        argmin, tied-set MEAN          0 q above 1e−5, worst 2.647e−7

    The middle row is why this is a mean and not a per-slot value: it
    fixes the 7 self-paired q outright but leaves — and on four q newly
    creates — a ~1e−5 point-group asymmetry on the six multiplicity-4 q.
    """
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    gvec = np.asarray(gvec_components, dtype=np.float64)   # (n_q, 3, ngkmax)
    if gvec.ndim != 3 or gvec.shape[1] != 3:
        raise ValueError(
            f"gvec_components must be (n_q, 3, ngkmax); got {gvec.shape}")
    n_q, _, ngkmax = gvec.shape
    bvec_f = np.asarray(geometry.bvec, dtype=np.float64)
    fact = 1.0 / float(geometry.cell_volume)
    bdot = geometry.bdot
    fft_grid = geometry.fft_grid

    if v_head_fn is not None and not callable(v_head_fn):
        # The pre-2026-08-08 spelling passed a (nkx, nky, nkz) TABLE indexed
        # by the BGW-wrapped q index.  That object cannot express the fix:
        # its value belongs to q_frac @ bvec, which is not the argmin q+G on
        # 12 of Si's 64 q, and it carries no per-slot resolution at all.
        # Refuse loudly rather than silently reinstate the label rule.
        raise TypeError(
            "v_head_fn must be a callable K_cart (m, 3) -> (m,) float64. "
            "A per-q (nkx, nky, nkz) head TABLE is not accepted: the head "
            "is injected at every argmin |q+G| slot and each slot must be "
            "valued from its own Cartesian q+G. Use "
            "vcoul.build_v_head_miniBZ_fn_3d.")
    tie_rtol = float(head_tie_rtol)

    out = np.zeros((n_q, ngkmax), dtype=np.float64)
    for qi in range(n_q):
        qf = q_irr_frac[qi]
        v, denom = kernel._v_bare_per_q(
            qf, gvec[qi], bvec_f=bvec_f, fact=fact,
            bdot=bdot, fft_grid=fft_grid)
        if v_head_fn is not None:
            # ``denom`` is the kernel's own |q+G|^2 — the SAME operand the
            # cutoff mask below compares against, so slot selection and
            # truncation can never disagree about the geometry.
            d2min = float(np.min(denom))
            if d2min > 1e-12:      # Γ: head is the separate rank-1 Σ_X term
                sel = np.nonzero(
                    np.asarray(denom) <= d2min * (1.0 + tie_rtol))[0]
                # Value each selected slot from its OWN Cartesian q+G, then
                # give the whole tied set their MEAN.  Both halves are
                # load-bearing; see TIED-SET VALUE in the docstring.
                K_sel = (qf[:, None] + gvec[qi][:, sel]).T @ bvec_f  # (m, 3)
                h = np.asarray(v_head_fn(K_sel), dtype=np.float64)
                v = np.array(v, dtype=np.float64, copy=True)
                v[sel] = float(h.mean()) if h.size > 1 else h
        if vcoul_cutoff_ry is not None:
            v = np.where(denom > float(vcoul_cutoff_ry), 0.0, v)
        out[qi] = v
    return out


class HeadSlotTable(NamedTuple):
    """Which G-slots carry the Coulomb head at each q, and at what value.

    Produced by :func:`head_slot_table`, which re-runs *exactly* the slot
    selection :func:`v_qG_table` performs (same ``_v_bare_per_q``, same
    ``denom``, same argmin-plus-tie rule, same tied-set mean) and reports
    it instead of consuming it.  Nothing here changes ``v(q+G)``.

    Fields, all with a leading ``n_q`` axis matching ``q_irr_frac``:

    ``sel``    ``(n_q, k_max)`` int32 — slot indices attaining the argmin,
               right-padded by repeating ``sel[:, 0]`` so the array is
               rectangular.  Read it together with ``mask``.
    ``mask``   ``(n_q, k_max)`` float64 — 1.0 on a genuine tied slot, 0.0
               on padding, and identically 0 on every q with no head slot
               (Γ, and any q whose head slot the bare-Coulomb cutoff
               zeroes).  ``Σ_j mask[q, j] == mult[q]``.
    ``v_bare`` ``(n_q,)`` float64 — the UNAVERAGED ``v`` at the head slot,
               i.e. what ``v_qG_table`` would leave there with
               ``v_head_fn=None``.  Zero where ``mask`` is all-zero.
    ``v_avg``  ``(n_q,)`` float64 — the value ``v_qG_table`` actually
               injects: the tied-set mean of ``v_head_fn`` when one is
               given, and ``v_bare`` when it is not.
    ``mult``   ``(n_q,)`` int32 — the tie multiplicity (0 where skipped).
    ``len2``   ``(n_q,)`` float64 — ``|q+G|²`` at the head slot, the shell
               label.  Zero where ``mask`` is all-zero.  Carried because the
               BGW-parity route matches its ``vcoul`` dump to LORRAX's head
               slots by shell rather than by index order.

    The consumer of record is the head-channel Coulomb placement in
    ``gw.head_channel``: ``v_avg / v_bare`` is the per-cell mini-BZ
    enhancement ``1 + η(q)`` that BerkeleyGW applies to W's head channel
    AFTER the Dyson solve, and ``sel``/``mask`` name the ζ columns that
    span that channel.
    """
    sel: np.ndarray
    mask: np.ndarray
    v_bare: np.ndarray
    v_avg: np.ndarray
    mult: np.ndarray
    len2: np.ndarray


def head_slot_table(
    kernel,
    q_irr_frac,
    gvec_components,
    *,
    geometry: CoulombGeometry,
    vcoul_cutoff_ry: float | None = None,
    v_head_fn=None,
    head_tie_rtol: float = 1e-9,
    k_max: int | None = None,
) -> HeadSlotTable:
    """Report :func:`v_qG_table`'s head-slot selection without applying it.

    Same arguments, same arithmetic, same guards — the two functions share
    ``kernel._v_bare_per_q`` and the identical ``d2min``/``tie_rtol``
    predicate, so a slot is in ``sel`` here if and only if ``v_qG_table``
    would overwrite it there.  Splitting the two rather than returning a
    second value from ``v_qG_table`` keeps the production ``v(q+G)`` path
    byte-for-byte unchanged: this function has no caller unless a deck
    turns the head-channel placement on.

    ``k_max`` fixes the second axis of ``sel``/``mask``.  Default is the
    measured maximum multiplicity over the supplied q's (1, 2 or 4 on
    every even k-grid tested — see :func:`v_qG_table`'s HEAD SLOT note);
    pass it explicitly to pin a shape across calls.  A multiplicity above
    ``k_max`` is a refusal, not a truncation, because silently dropping a
    tied slot re-creates exactly the q → −q asymmetry the tied-set rule
    exists to remove.
    """
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    gvec = np.asarray(gvec_components, dtype=np.float64)
    if gvec.ndim != 3 or gvec.shape[1] != 3:
        raise ValueError(
            f"head_slot_table: gvec_components must be (n_q, 3, ngkmax); "
            f"got {gvec.shape}")
    n_q = gvec.shape[0]
    bvec_f = np.asarray(geometry.bvec, dtype=np.float64)
    fact = 1.0 / float(geometry.cell_volume)
    tie_rtol = float(head_tie_rtol)

    sels = []
    v_bare = np.zeros(n_q)
    v_avg = np.zeros(n_q)
    len2 = np.zeros(n_q)
    mult = np.zeros(n_q, np.int32)
    for qi in range(n_q):
        qf = q_irr_frac[qi]
        v, denom = kernel._v_bare_per_q(
            qf, gvec[qi], bvec_f=bvec_f, fact=fact,
            bdot=geometry.bdot, fft_grid=geometry.fft_grid)
        d2min = float(np.min(denom))
        if d2min <= 1e-12:
            # Γ.  ``v_qG_table`` skips it; the head there is the owner's
            # separate rank-1 term and is not this machinery's business.
            sels.append(np.zeros(0, dtype=np.int32))
            continue
        if vcoul_cutoff_ry is not None and d2min > float(vcoul_cutoff_ry):
            # The head slot is outside the bare-Coulomb cutoff, so
            # ``v_qG_table``'s cutoff mask zeroes it after injecting —
            # there is no head channel at this q.
            sels.append(np.zeros(0, dtype=np.int32))
            continue
        sel = np.nonzero(np.asarray(denom) <= d2min * (1.0 + tie_rtol))[0]
        sels.append(sel.astype(np.int32))
        mult[qi] = sel.size
        len2[qi] = d2min
        # All tied slots share |q+G|², hence share the bare value; take
        # the mean anyway so this is the same statistic as ``v_avg``.
        v_bare[qi] = float(np.mean(np.asarray(v)[sel]))
        if v_head_fn is None:
            v_avg[qi] = v_bare[qi]
        else:
            K_sel = (qf[:, None] + gvec[qi][:, sel]).T @ bvec_f
            h = np.asarray(v_head_fn(K_sel), dtype=np.float64)
            # ``h`` is (m,) even at m == 1 — v_head_fn's contract is a
            # 1-D return.  The mean over a one-element tied set is that
            # element, so this is the tied-set rule with no branch.
            v_avg[qi] = float(np.mean(h))

    k_meas = int(mult.max()) if n_q else 0
    k = int(k_max) if k_max else max(1, k_meas)
    if k_meas > k:
        raise ValueError(
            f"head_slot_table: measured argmin |q+G| tie multiplicity "
            f"{k_meas} exceeds k_max={k}. Raise k_max — truncating the "
            f"tied set would break the q -> -q equivariance the tied-set "
            f"rule guarantees (see v_qG_table's HEAD SLOT note).")

    sel_out = np.zeros((n_q, k), dtype=np.int32)
    mask_out = np.zeros((n_q, k), dtype=np.float64)
    for qi, sel in enumerate(sels):
        if sel.size:
            sel_out[qi, :sel.size] = sel
            sel_out[qi, sel.size:] = sel[0]
            mask_out[qi, :sel.size] = 1.0
    return HeadSlotTable(sel=sel_out, mask=mask_out, v_bare=v_bare,
                         v_avg=v_avg, mult=mult, len2=len2)


def v_qG_single(kernel, geometry: CoulombGeometry, qvec_wrapped,
                comps_qG) -> jax.Array:
    """Single-q :meth:`CoulombKernel.v_qG`, as complex128 ``(nG,)``.

    Thin wrapper over :func:`v_qG_table` so the Protocol method and the
    production table share one arithmetic path.  ``comps_qG`` is
    ``(nG, 3)`` here (the historical single-q spelling), transposed to
    the driver's ``(1, 3, nG)``.
    """
    comps = np.asarray(comps_qG, dtype=np.float64)
    v = v_qG_table(
        kernel,
        np.asarray(qvec_wrapped, dtype=np.float64).reshape(1, 3),
        comps.T[None, :, :],
        geometry=geometry,
    )
    return jnp.asarray(v[0], dtype=jnp.complex128)
