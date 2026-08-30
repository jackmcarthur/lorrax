"""Load-time MEASUREMENT of the symmetries a WFN file only *claims*.

Why this module exists
======================
A BerkeleyGW ``WFN.h5`` carries a symmetry block (``mtrx``/``tnp``/``ntran``)
and a k-point list with weights.  It does **not** carry a statement of
whether time-reversal symmetry (TRS) holds for the system the
wavefunctions describe, nor any independent confirmation that the stored
spatial operations really are symmetries of *these* wavefunctions.  For
years LORRAX inferred both from flags — ``ntran``, "was this a ``nosym``
deck", "QE usually TRS-folds" — and that inference chain produced a long
series of silent bugs, culminating in the ``ntran <= 1`` incident
(scorecard §Q) in which every ``nosym`` WFN read through the eager
backend had ψ(r) silently replaced by ψ*(−r): norm-, orthogonality- and
⟨T⟩-preserving, and therefore invisible to every internal consistency
check that existed.

The cure is to stop inferring and start measuring.  The occupied
wavefunctions themselves contain the answer: form the spin-resolved
charge density from them and ask the density what symmetries it has.

    1. TRS.   m(r) = Σ_k w_k Σ_occ ψ†(r) σ ψ(r).  Time reversal maps
              the occupied manifold at k to the one at −k with
              m_{−k}(r) = −m_k(r), so for every k-pair (k, −k) present
              in the file, ``w_k m_k + w_{−k} m_{−k}`` must vanish.  A
              TRS-broken (magnetic) system violates this pointwise.
    2. Spatial ops.  For each claimed ``{S|τ}`` the density built from
              the k-points that ``S`` leaves invariant must satisfy
              ρ(Sr + τ) = ρ(r).
    3. Invariants.  ∫ρ d³r = N_elec and ρ ≥ 0.

The verdict from (1) is stored on the loader as ``trs_holds`` and is
consulted by :class:`symmetry_maps.SymMaps`, which refuses to
select time-reversal rows when the density says TRS is broken —
regardless of what the flags imply.  Correctness of the unfold for
magnetic systems is thereby protected *structurally*, not by convention.

Non-circularity
===============
Everything here is computed from the **raw IBZ coefficients** exactly as
they sit in the file.  The symmetry unfold is the thing under test, so
it is never used: no ``unfold_psi``, no ``k='full_bz'``, no ``SymMaps``.
The only shared machinery is the density quadrature itself
(:func:`psp.get_DFT_mtxels.valence_density_from_kpoint`, imported, never
re-derived) and the G→FFT-box index table
(:meth:`file_io.wfn_loader.WfnLoader.box_index` with ``k='ibz'``, which
reads ``wfns/gvecs`` directly).

THE TRS ALGEBRA, WRITTEN OUT SO IT CAN BE AUDITED
=================================================
Everything below is spin-1/2 (``nspinor == 2``), BGW/LORRAX conventions.
The companion derivation for the *unfold* side lives in
``symmetry_maps.unfold_psi``; the two must agree and are
cross-checked in ``tests/test_symmetry_unfold.py``.

**(T1) The operator.**  Time reversal for spin-1/2 is the ANTIUNITARY

    Θ = i σ_y K,        K = complex conjugation,
    i σ_y = [[0, 1], [−1, 0]]   (this is ``symmetry_maps._I_SIGMA_Y``),
    Θ² = −1                      (Kramers).

**(T2) Action on a Bloch spinor, in real space and in G space.**  With
``ψ_nk(r) = Σ_G c_nk(G) e^{i(k+G)·r}`` and ``ψ = (a, b)ᵀ``:

    (Θψ_nk)(r) = i σ_y ψ*_nk(r) = (b*(r), −a*(r))ᵀ
               = Σ_G [i σ_y c*_nk(G)] e^{−i(k+G)·r}
               = Σ_G' [i σ_y c*_nk(−G')] e^{i(−k+G')·r}      (G' = −G)

so ``Θψ_nk`` is a Bloch state at **−k** whose coefficients are

    c_{Θn,−k}(G') = i σ_y  conj( c_nk(−G') ).                        (★)

Note the two separate pieces of (★): the spinor factor ``iσ_y·conj``
AND the **negation of the G list**.  Dropping the second half while
keeping the first is precisely the scorecard-§Q corruption
(``ψ(r) → ψ*(−r)``), which is why the two must always be reasoned about
together.

**(T3) What Θ does to the magnetization density.**  With
``m_ψ(r) = ψ†(r) σ ψ(r)`` and ``φ = iσ_y ψ*``:

    φ† σ_i φ = ψᵀ (iσ_y)† σ_i (iσ_y) ψ*  =  ψᵀ σ_y σ_i σ_y ψ*
             = −ψᵀ σ_i* ψ*                [σ_y σ_i σ_y = −σ_i*, all i]
             = −(ψ† σ_i ψ)*  =  −ψ† σ_i ψ  [m is real]

    ⇒  m_{Θψ}(r) = − m_ψ(r).                                        (★★)

Combining (★★) with (T2): **if the occupied manifold at k is carried by
Θ onto the occupied manifold at −k, then**

    m_{−k}(r) = − m_k(r)    pointwise in r.                       (★★★)

(★★★) is the identity this module measures, and it is the whole test.
Note what it does NOT involve: no symmetry operation, no rotation
matrix, no fractional translation τ, no umklapp vector, no spinor
rotation ``U_spinor``.  It is a statement about two real-space fields.
That is deliberate — it makes the TRS verdict independent of every
convention that has historically been got wrong here.

**(T4) Self-paired k (TRIM) and umklapp.**  When ``−k ≡ k + G₀`` for a
reciprocal-lattice vector ``G₀``, the state sets at ``−k`` and ``k`` are
the same states (a shift of G₀ only relabels the coefficient list; the
real-space Bloch function is unchanged), so (★★★) collapses to
``m_k(r) = −m_k(r)``, i.e. ``m_k ≡ 0`` at every TRIM.  This module
detects that case as ``pair[i] == i`` — the pairing is computed modulo
the k-grid in integer coordinates, so umklapp is handled by the
arithmetic and never appears explicitly.  Because the comparison is
made between real-space DENSITIES and never between coefficient lists,
no ``G₀`` bookkeeping is possible to get wrong.

**(T5) Does the band cut-off respect (★★★)?**  ``nocc = max(ifmax)``
is a band-INDEX cut.  Under TRS ``ε_{n,−k} = ε_{n,k}`` band by band, so
the same index cut selects Θ-corresponding manifolds at k and −k — the
cut is TRS-consistent even for a metal.  The one genuine hazard is a
cut that lands INSIDE a degenerate multiplet, where the diagonaliser's
arbitrary ordering among degenerate partners differs between k and −k;
the check therefore reports ``manifold_gap`` (the smallest
``ε[nocc] − ε[nocc−1]`` over the sampled k) and says so explicitly in
any failure message, so an auditor can rule that channel in or out.
Every LORRAX production deck is a gapped insulator where this cannot
arise.

NON-SYMMORPHIC OPS AND SPINORS: WHERE THEY DO AND DO NOT ENTER
==============================================================
**They do not enter the TRS arm at all** — see (T3).  This is the single
most important thing to know when auditing: a bug in the τ-phase, in
``U_spinor``, or in the umklapp vector cannot move the TRS verdict.

**They enter the spatial arm only through the point mapping.**  The
spatial test asks whether ``ρ(Sr + τ) = ρ(r)``.  Write the action of
``{S|τ}`` on a spinor Bloch state at a k in its little group:

    ψ_n{Sk}(r) = U_spinor(S) · ψ_{n'k}({S|τ}⁻¹ r) · e^{iχ}

for some unitary mixing within the degenerate/occupied manifold and
some phase χ (which carries the τ-phase and any umklapp ``e^{iG₀·r}``).
Forming ``ρ = Σ_n ψ†ψ``:

  * the phase ``e^{iχ}`` cancels — ρ is a modulus;
  * ``U_spinor`` cancels — ρ is a spinor TRACE, ``ψ†U†Uψ = ψ†ψ``;
  * the unitary mixing within the occupied manifold cancels — ρ is the
    trace of the manifold projector.

**Only the real-space point map ``r → mtrx⁻¹·r + τ/2π`` survives**, and
that is exactly what :func:`_grid_permutation` builds.  Consequences,
both worth knowing:

  * *Good*: the spatial arm is immune to the spinor and phase
    conventions, so a residual there is unambiguously a wrong
    ``mtrx``/``tnp``, not a bookkeeping slip elsewhere.  It is validated
    on ``si_cohsex_debug``, which is genuinely NON-SYMMORPHIC
    (``tnp = π`` ⇒ ``τ_frac = 1/2`` glides): all 48 ops pass at ≤3e-9.
  * *Limitation, stated plainly*: because ``U_spinor`` and the τ-phase
    cancel out of ρ, this check does **not** validate them.  It
    validates ``mtrx``, ``tnp`` and the TRS verdict.  ``U_spinor`` and
    the τ-phase are covered by ``tests/test_symmetry_unfold.py`` and by
    the §Q shape guard in ``unfold_psi``.  Extending the spatial arm to
    ``m`` (which transforms as an AXIAL vector, ``m(Sr+τ) = det(R)·R·m(r)``
    with ``R`` the CARTESIAN rotation) would test the spinor half too;
    that is deliberate future work, not an oversight.

**(S1) Why the τ commensurability guard exists.**  ``{S|τ}`` is a grid
permutation only when ``τ_frac × FFTgrid`` is integral.  QE chooses FFT
grids that satisfy this for the ops it stores, but a hand-edited or
converted file need not.  Rather than round and silently compare the
wrong points, :func:`_grid_permutation` returns ``None`` and the op is
reported as *untested* — never as *passed*.

Two design points that are easy to get wrong
============================================
**Why the k-pairing form of the TRS test and not "‖m_total‖ = 0".**
The IBZ-weighted magnetization Σ_k w_k m_k does *not* vanish for a
nonmagnetic system whenever the file's IBZ was reduced using TRS: the
weights then count both k̄'s spatial star and its time-reversed image,
but the stored ψ only supplies the k̄ half.  MoS₂ (D₃ₕ, no inversion,
strong spin-valley coupling) is exactly that case — m at the K valley is
large and opposite at K′, and a TRS-folded IBZ keeps only one of them.
Summing over ± *pairs that are both present in the file* is the
formulation that is exactly zero under TRS with no symmetrization step,
hence no circularity and no false alarm.

**Why the coverage is always adequate where it matters.**  A k-pair is
testable iff −k is also in the file's k list.  A deck whose IBZ was
reduced with TRS has poor coverage — but QE only performs TRS folding
when time reversal is *assumed*, and a genuinely magnetic system can
never produce such a mesh; its k-set is spatially reduced only, and a
spatially-reduced (or ``nosym``, full-BZ) mesh is closed under k → −k.
So the cases where TRS can actually be broken are precisely the cases
where this check has full coverage.  Coverage is reported either way.

**Why the spatial-op test restricts to each op's little group.**  The
raw IBZ-weighted ρ is *not* point-group symmetric (it is the
unsymmetrized precursor of ρ_full); testing it against a general op
would flag every legitimate op.  Symmetrizing it first would make the
test circular.  Restricting to the k-points that ``S`` fixes gives a
partial density that a true symmetry *must* leave invariant, using
nothing but raw ψ.  For a Γ-centred mesh Γ is in every op's little
group, so every op is tested.

Cost (measured, Frontera CPU, scorecard §U)
===========================================
Occupied bands only, on the ψ FFT box, with a ±-closed k-subsample
(``LORRAX_TRS_MAX_K``, default 12).  The subsample costs nothing in
rigour: each ± pair is an independent exact test, so a subset is a
*sharper* statement than the sum (no cancellation between pairs can
mask a violation).

    deck                              grid        nk    check
    cohsex_debug  (ntran=12, nk=4)    15x15x60     4    1.4 s
    gnppm_debug   (ntran=2,  nk=9)    24x24x80     9    1.2 s
    bispinor_debug(ntran=2,  nk=9)    30x30x120    9    2.7 s
    si_cohsex     (ntran=48, nk=8)    24x24x24     8    0.4 s
    MoS2 12x12    (ntran=1,  nk=144)  36x36x135   12    4.8 s

Second and later constructions of the same file are free (the verdict is
cached per (path, mtime, size) for the process; measured 0.01 s).

One caveat worth knowing: on a COLD Frontera node the first
``import psp.get_DFT_mtxels`` measures ~49 s and the first
``jnp.fft.ifftn`` ~8 s — both pure Lustre page-cache effects on the
import graph (``import jax`` itself goes 8.1 s cold → 1.5 s warm the same
way).  Warm, that import is 0.39 s.  The check's own arithmetic is the
table above; it does not do 50 s of work anywhere.

Environment
===========
``LORRAX_TRS_CHECK``      ``1`` (default, on) | ``0`` (off) | ``strict``
                          (raise instead of warn on any failure or on a
                          flags-vs-measurement disagreement).
``LORRAX_TRS_TOL``        relative tolerance for ‖m‖∞/‖ρ‖∞ (default 1e-6).
``LORRAX_TRS_SPATIAL_TOL`` relative tolerance for the ρ(Sr+τ) residual
                          (default 1e-4).
``LORRAX_TRS_MAX_K``      max k-points sampled (default 12 =
                          ``MAX_K_DEFAULT``, the measured sufficiency
                          choice — scorecard §U k-ladder; 0 = all).
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DensitySymmetryReport",
    "check_density_symmetries",
    "cached_density_symmetry_check",
    "trs_check_mode",
]

# Default tolerances.  Justification (measured, scorecard §U):
#
# ``TOL_TRS`` gates ‖Σ_pair w m‖∞ / ‖ρ‖∞.  In exact arithmetic the pair
# sum is identically zero, so the floor is set by (a) fp64 FFT round-off,
# ~1e-15 relative, and (b) the SCF/diagonalisation convergence of the
# stored ψ, which is what actually dominates: QE ``conv_thr`` of
# 1e-8…1e-10 Ry leaves residuals in the 1e-9…1e-7 band on the decks
# measured here.  1e-6 sits above that and ~4-6 orders of magnitude
# below any real magnetization (a ferromagnet gives O(1e-2…1); even a
# weak canted moment gives O(1e-3)), so the discriminating power is not
# remotely marginal.
TOL_TRS = 1.0e-6
# ``TOL_SPATIAL`` gates ‖ρ(Sr+τ) − ρ(r)‖∞ / ‖ρ‖∞.  Looser because ρ's
# grid symmetry additionally samples the SCF density error and the FFT
# box aliasing at the box edge; a *wrong* op mismatches by O(0.1…1), so
# 1e-4 is still a factor ~1e3 clear of any real failure.
TOL_SPATIAL = 1.0e-4
# Sampling cap.  See ``_select_kpoints`` for why a subset is a *sharper*
# statement than the full sum rather than an approximation.  12 is set
# from the measured MoS2 12x12 ladder (scorecard §U): the verdict and the
# residual's order of magnitude are flat from 9 k to all 144 k
# (2.5e-13 → 3.1e-14, both ~7 orders under the gate), while the cost is
# linear in k (~0.4 s/k warm on a 36x36x135 box with 26 occupied bands).
MAX_K_DEFAULT = 12
# Bound fractional diagnostics to one small band slab per sampled k; the
# exact-integer path below keeps its incumbent single occupied-window load.
FRACTIONAL_BAND_CHUNK = 16


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class DensitySymmetryReport:
    """Outcome of the load-time density symmetry measurement."""

    # --- the headline -------------------------------------------------
    trs_holds: bool
    #: How ``trs_holds`` was established:
    #: ``'measured'``      — ± k-pairs were present and ‖m‖ was measured
    #: ``'spin-degenerate'`` — nspin=1, nspinor=1: no magnetization exists
    #: ``'unmeasurable'``  — no ± k-pair in the file (TRS-folded IBZ);
    #:                       verdict defaults to True, see module docstring
    #: ``'skipped'``       — check disabled or it failed to run
    trs_basis: str
    #: ‖Σ_pair w·m‖∞ / ‖ρ‖∞, maximised over ± pairs.  None if not measured.
    m_rel: float | None
    #: ‖Σ_covered w·m‖∞ / ‖ρ‖∞ — the aggregate (weaker) form.
    m_rel_total: float | None
    #: Fraction of the total k-weight for which a ± partner was present.
    trs_coverage: float
    tol_trs: float
    #: THE FLAG SIDE of the comparison: True when the file's k list
    #: cannot generate its own ``kgrid`` using the stored SPATIAL
    #: operations alone, i.e. the mesh was reduced assuming time
    #: reversal.  ``trs_implied_by_mesh and not trs_holds`` is a
    #: flags-vs-measurement disagreement and is escalated loudly.
    trs_implied_by_mesh: bool | None

    # --- spatial table ------------------------------------------------
    spatial_ops_ok: bool
    #: (ntran,) float — ‖ρ(Sr+τ) − ρ(r)‖∞/‖ρ_s‖∞ per op; NaN = untested.
    spatial_residual: np.ndarray
    spatial_untested: tuple[int, ...]
    spatial_failed: tuple[int, ...]
    tol_spatial: float

    # --- basic invariants ---------------------------------------------
    charge: float
    charge_expected: float
    charge_rel_err: float
    rho_min_rel: float
    invariants_ok: bool
    #: min over sampled k of ``ε[nocc] − ε[nocc−1]`` in the file's energy
    #: units.  If this is ~0 the occupied-band cut splits a degenerate
    #: multiplet and the TRS residual can be spuriously nonzero — see
    #: (T5) in the module docstring.  NaN when energies are unavailable.
    manifold_gap: float

    # --- provenance / cost --------------------------------------------
    path: str
    nocc: int
    nspin: int
    nspinor: int
    n_k_used: int
    n_k_total: int
    subsampled: bool
    fft_grid: tuple[int, int, int]
    seconds: float
    #: Split of ``seconds``: HDF5 hyperslab reads vs the FFT quadrature.
    seconds_io: float = 0.0
    seconds_quad: float = 0.0
    messages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return bool(self.trs_holds and self.spatial_ops_ok
                    and self.invariants_ok)

    def summary(self) -> str:
        m = "n/a" if self.m_rel is None else f"{self.m_rel:.2e}"
        sub = f" (subsampled {self.n_k_used}/{self.n_k_total} k)" if self.subsampled \
            else f" ({self.n_k_used} k)"
        nsp = [i for i, r in enumerate(self.spatial_residual)
               if np.isfinite(r)]
        smax = (f"{np.nanmax(self.spatial_residual):.2e}"
                if nsp else "untested")
        return (
            f"[density-symmetry] {os.path.basename(self.path)}: "
            f"TRS={'HOLDS' if self.trs_holds else 'BROKEN'} "
            f"({self.trs_basis}, ‖m‖/‖ρ‖={m}, coverage={self.trs_coverage:.0%}, "
            f"mesh-implies-TRS={self.trs_implied_by_mesh}) "
            f"| spatial {len(nsp)}/{len(self.spatial_residual)} ops tested, "
            f"max resid {smax} "
            f"| ∫ρ={self.charge:.6f} (exp {self.charge_expected:.6f}, "
            f"rel {self.charge_rel_err:.1e}) "
            f"| {self.seconds:.2f}s (io {self.seconds_io:.2f} + fft "
            f"{self.seconds_quad:.2f}){sub}"
        )


def _skipped_report(path: str, reason: str) -> DensitySymmetryReport:
    return DensitySymmetryReport(
        trs_holds=True, trs_basis="skipped", m_rel=None, m_rel_total=None,
        trs_coverage=0.0, tol_trs=TOL_TRS, trs_implied_by_mesh=None,
        spatial_ops_ok=True, spatial_residual=np.zeros(0), spatial_untested=(),
        spatial_failed=(), tol_spatial=TOL_SPATIAL,
        charge=float("nan"), charge_expected=float("nan"),
        charge_rel_err=float("nan"), rho_min_rel=float("nan"),
        invariants_ok=True, manifold_gap=float("nan"),
        path=str(path), nocc=0, nspin=0, nspinor=0, n_k_used=0, n_k_total=0,
        subsampled=False, fft_grid=(0, 0, 0), seconds=0.0,
        messages=(reason,),
    )


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def trs_check_mode() -> str:
    """``'off'`` | ``'on'`` | ``'strict'`` from ``LORRAX_TRS_CHECK``."""
    raw = os.environ.get("LORRAX_TRS_CHECK", "1").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("strict", "2"):
        return "strict"
    return "on"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not a float; using {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.warn(f"{name}={raw!r} is not an int; using {default}")
        return default


def _is_reporting_process() -> bool:
    """True on the one process that should print the summary."""
    try:
        import jax
        return int(jax.process_index()) == 0
    except Exception:
        return True


# ----------------------------------------------------------------------
# k-space bookkeeping (integer grid coordinates)
# ----------------------------------------------------------------------
def _k_as_ints(kpoints: np.ndarray, kgrid: np.ndarray,
               shift: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fractional k → integer coordinates on ``kgrid``.

    Mirrors the expression :class:`SymMaps` uses in its trivial branch.
    Returns ``(kint, exact)``; ``exact`` is False when a k-point does not
    land on the grid (a shifted/unusual mesh), in which case all
    grid-integer reasoning downstream is disabled.
    """
    kgrid = np.asarray(kgrid, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.float64).reshape(3)
    shift_frac = shift / kgrid.astype(np.float64)
    wrapped = np.mod(np.asarray(kpoints, dtype=np.float64)
                     - shift_frac[None, :], 1.0)
    scaled = wrapped * kgrid[None, :]
    kint = np.mod(np.rint(scaled).astype(np.int64), kgrid[None, :])
    exact = bool(np.max(np.abs(scaled - np.rint(scaled))) < 1e-6)
    return kint, exact


def _negation_pairs(kint: np.ndarray, kgrid: np.ndarray) -> np.ndarray:
    """``pair[i] = j`` with ``k_j ≡ −k_i`` (mod the k-grid), else −1."""
    kgrid = np.asarray(kgrid, dtype=np.int64)
    neg = np.mod(-kint, kgrid[None, :])
    lookup: dict[tuple[int, int, int], int] = {}
    for i, row in enumerate(kint):
        lookup.setdefault((int(row[0]), int(row[1]), int(row[2])), i)
    return np.array(
        [lookup.get((int(r[0]), int(r[1]), int(r[2])), -1) for r in neg],
        dtype=np.int64)


def _little_groups(kint: np.ndarray, sym_mats_k: np.ndarray,
                   kgrid: np.ndarray) -> list[np.ndarray]:
    """For each spatial op, the IBZ indices it leaves invariant.

    ``sym_mats_k[s] @ k ≡ k`` (mod the k-grid).  ``sym_mats_k`` is the
    k-space action ``mtrx.T`` (SymMaps convention); it is an integer
    matrix, so it acts on the integer grid coordinates directly.
    """
    kgrid = np.asarray(kgrid, dtype=np.int64)
    out = []
    for S in np.asarray(sym_mats_k, dtype=np.int64):
        img = np.mod(kint @ S.T, kgrid[None, :])
        out.append(np.flatnonzero(np.all(img == kint, axis=1)))
    return out


def _mesh_needs_trs(kint: np.ndarray, sym_mats_k: np.ndarray,
                    kgrid: np.ndarray) -> bool:
    """Does the file's k list need time reversal to cover its own kgrid?

    This is the *flag* side of the flags-vs-measurement comparison: it
    reads only the stored k-points, weights-free, and asks whether the
    SPATIAL star of the IBZ already fills the ``kgrid``.  A ``nosym``
    full-BZ deck and a spatially-reduced deck both answer False; a
    TRS-folded IBZ answers True.
    """
    kgrid = np.asarray(kgrid, dtype=np.int64)
    seen: set[tuple[int, int, int]] = set()
    for S in np.asarray(sym_mats_k, dtype=np.int64):
        img = np.mod(kint @ S.T, kgrid[None, :])
        seen.update(map(tuple, img.tolist()))
    return len(seen) < int(np.prod(kgrid))


def _grid_permutation(mtrx: np.ndarray, tau_2pi: np.ndarray,
                      fft_grid: tuple[int, int, int]) -> np.ndarray | None:
    """Flat index map for the BGW real-space action ``r' = mtrx⁻¹·r + τ``.

    Returns ``perm`` with ``f_transformed.flat[perm[n]] = f.flat[n]``
    for a scalar field, i.e. ``perm[n]`` is the flat index of the image
    of grid point ``n``.  ``None`` when ``τ`` is not commensurate with
    the FFT grid (the op cannot be represented as a grid permutation).

    Convention source: ``symmetry_maps.centroid_source_map_and_wrap``
    and ``SymMaps.validate_atomic_symmetries`` — both use
    ``Rinv = inv(mtrx)`` and ``τ_frac = tnp / 2π``.
    """
    N = np.asarray(fft_grid, dtype=np.int64)
    Rinv = np.rint(np.linalg.inv(np.asarray(mtrx, dtype=np.float64)))
    if np.max(np.abs(Rinv - np.linalg.inv(
            np.asarray(mtrx, dtype=np.float64)))) > 1e-8:
        return None
    tau_frac = np.asarray(tau_2pi, dtype=np.float64) / (2.0 * np.pi)
    tau_grid = tau_frac * N.astype(np.float64)
    if np.max(np.abs(tau_grid - np.rint(tau_grid))) > 1e-6:
        return None
    tau_i = np.rint(tau_grid).astype(np.int64)

    ix, iy, iz = np.meshgrid(*(np.arange(int(n)) for n in N), indexing="ij")
    n_idx = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1)  # (P,3)
    img = np.mod(n_idx @ Rinv.astype(np.int64).T + tau_i[None, :], N[None, :])
    return (img[:, 0] * (N[1] * N[2]) + img[:, 1] * N[2] + img[:, 2])


def _select_kpoints(kint: np.ndarray, weights: np.ndarray,
                    pair: np.ndarray, little: list[np.ndarray],
                    max_k: int) -> np.ndarray:
    """Choose the k subset to build the density from.

    Selection rules, in priority order:
      1. every k that is the lowest-index member of some op's little
         group (so the spatial table stays fully testable);
      2. the ``−k`` partner of everything already chosen (so the TRS
         test stays exact pair-by-pair);
      3. heaviest-weight ± pairs first, until ``max_k`` is reached.
    Rule 3 is a *sufficiency* choice, not an approximation: each ± pair
    is an independent exact identity under TRS, so a subset can only be
    a sharper test than the full sum, never a weaker one.
    """
    nk = int(kint.shape[0])
    if max_k <= 0 or nk <= max_k:
        return np.arange(nk, dtype=np.int64)

    chosen: list[int] = []
    seen: set[int] = set()

    def take(i: int) -> None:
        i = int(i)
        if i not in seen:
            seen.add(i)
            chosen.append(i)

    for members in little:
        if members.size:
            take(members[0])
    for i in list(chosen):
        if pair[i] >= 0:
            take(pair[i])

    order = np.argsort(-np.asarray(weights, dtype=np.float64), kind="stable")
    for i in order:
        if len(seen) >= max_k:
            break
        take(i)
        j = int(pair[int(i)])
        if j >= 0:
            take(j)
    return np.sort(np.array(chosen, dtype=np.int64))


# ----------------------------------------------------------------------
# Raw IBZ ψ → spin-resolved density
# ----------------------------------------------------------------------
def _raw_ibz_psi_k(loader, ik: int, nb: int, *, b_lo: int = 0) -> np.ndarray:
    """Raw ``(nb-b_lo, nspinor, ngk)`` coefficients for IBZ k ``ik``.

    Deliberately reads ``wfns/coeffs`` directly rather than going through
    ``loader.load(k='ibz')``: the check must be independent of every
    build path (the unfold is what is under test), must not fire a
    collective phdf5 read at construction time, and must be able to
    touch only the sampled k-points.  ``tests/test_density_symmetry_check
    .py::test_raw_read_matches_loader_ibz_path`` pins it bit-exactly to
    ``loader.load(bands=..., k='ibz')`` so the two cannot drift.
    """
    ds = loader._file["wfns/coeffs"]
    start = int(loader._kpt_starts[int(ik)])
    ngk = int(loader.ngk[int(ik)])
    raw = ds[int(b_lo):int(nb), :, start:start + ngk, :]
    return raw[..., 0] + 1j * raw[..., 1]


def _to_box(psi_k: np.ndarray, g_index_k: np.ndarray,
            ngkmax: int) -> np.ndarray:
    """``(nb, ns, ngk)`` → ``(nb, ns, nx, ny, nz)`` via the zero-sentinel
    gather table from :meth:`WfnLoader.box_index`.

    Same semantics as ``common.wfn_transforms._box_kernel`` (append a
    zero slot so the sentinel ``ngkmax`` gathers zero) but in NumPy on a
    single k, so no ``Mesh``/``NamedSharding`` is involved — the check
    runs identically on every process of a multi-process job without
    touching the global device topology.
    """
    nb, ns, ngk = psi_k.shape
    padded = np.zeros((nb, ns, int(ngkmax) + 1), dtype=np.complex128)
    padded[:, :, :ngk] = psi_k
    return padded[:, :, g_index_k]


def _spin_resolved_density(psi_box: np.ndarray, *, nocc: int, weight: float,
                           cell_volume: float, spin_degeneracy: float,
                           band_occupations: np.ndarray | None,
                           want_transverse: bool,
                           valence_density_fn=None,
                           spin_density_fields_fn=None):
    """``(ρ_k, m_k)`` for one k, both weighted by ``weight``.

    The two-component path goes through ONE call to
    :func:`psp.get_DFT_mtxels.valence_density_from_kpoint` — the single
    source of truth for the density quadrature (ortho IFFT × √(N/Ω)) —
    requesting its local ``rho_ab = sum psi_a psi_b*`` return, then through
    :func:`psp.get_DFT_mtxels.spin_density_matrix_to_pauli_fields` for the
    Pauli contraction.  Thus charge and all three spin components share one
    IFFT, while neither normalization nor the sigma-y convention is
    reimplemented in this service.  Both callables are injectable; ``None``
    imports each lazily only when used.  See
    :func:`check_density_symmetries` for why the injection exists and why
    the default is the import.

    DERIVATION (audit me).  With the BGW Pauli convention
    ``σ_x = [[0,1],[1,0]]``, ``σ_y = [[0,−i],[i,0]]``,
    ``σ_z = [[1,0],[0,−1]]`` and ``ψ = (a, b)ᵀ``:

        ρ   = ψ†ψ      = |a|² + |b|²
        m_x = ψ†σ_x ψ  = a*b + b*a        = 2·Re(a* b)
        m_y = ψ†σ_y ψ  = −i a*b + i b*a   = 2·Im(a* b)
        m_z = ψ†σ_z ψ  = |a|² − |b|²

    ``m`` is None when the wavefunctions are scalar (``nspinor == 1``),
    where no magnetization density is representable at all.

    Four-component kinetic-balance files retain the incumbent upper-Pauli
    diagnostic below.  The new matrix API intentionally refuses them rather
    than silently choosing a relativistic spin operator; that convention is
    outside this raw-IBZ symmetry check's ownership.
    """
    if valence_density_fn is None:
        from psp.get_DFT_mtxels import valence_density_from_kpoint as _D
    else:
        _D = valence_density_fn

    def dens(arr) -> np.ndarray:
        kwargs = dict(nocc=nocc, weight=weight,
                      cell_volume=cell_volume,
                      spin_degeneracy=spin_degeneracy)
        if band_occupations is not None:
            kwargs["band_occupations"] = band_occupations
        return np.asarray(_D(arr, **kwargs), dtype=np.float64)

    ns = int(psi_box.shape[1])
    if ns == 1:
        return dens(psi_box), None

    if ns == 2:
        kwargs = dict(nocc=nocc, weight=weight,
                      cell_volume=cell_volume,
                      spin_degeneracy=spin_degeneracy,
                      return_spin_density_matrix=True)
        if band_occupations is not None:
            kwargs["band_occupations"] = band_occupations
        rho_ab = _D(psi_box, **kwargs)
        if spin_density_fields_fn is None:
            from psp.get_DFT_mtxels import (
                spin_density_matrix_to_pauli_fields as _to_fields,
            )
        else:
            _to_fields = spin_density_fields_fn
        fields = np.asarray(_to_fields(rho_ab), dtype=np.float64)
        rho = fields[0]
        m = np.array(fields[1:], copy=True)
        if not want_transverse:
            m[:2] = 0.0
        return rho, m

    # Compatibility path for four-component kinetic-balance diagnostics.
    a = psi_box[:, 0:1]
    b = psi_box[:, 1:2]
    d_a = dens(a)
    d_b = dens(b)
    rho = d_a + d_b
    m = np.zeros((3,) + rho.shape, dtype=np.float64)
    m[2] = d_a - d_b
    if want_transverse:
        m[0] = dens(a + b) - rho
        m[1] = rho - dens(a + 1j * b)
    return rho, m


# ----------------------------------------------------------------------
# The check
# ----------------------------------------------------------------------
def check_density_symmetries(
    loader,
    *,
    tol_trs: float | None = None,
    tol_spatial: float | None = None,
    max_k: int | None = None,
    nocc: int | None = None,
    want_transverse: bool = True,
    valence_density_fn=None,
    spin_degeneracy_fn=None,
    spin_density_fields_fn=None,
) -> DensitySymmetryReport:
    """Measure TRS + the spatial symmetry table against ρ built from ψ.

    ``loader`` is a :class:`file_io.wfn_loader.WfnLoader` (only its
    mf_header attributes, ``box_index(k='ibz')`` and the open HDF5
    handle are used — no ``SymMaps``, no unfold).

    THE TWO QUADRATURE PARAMETERS, and why they are not a refactor.  This
    module's non-circularity argument (module docstring, §Non-circularity)
    is quoted here unchanged, because it is the thing the parameters have
    to preserve:

        Everything here is computed from the **raw IBZ coefficients**
        exactly as they sit in the file.  The symmetry unfold is the thing
        under test, so it is never used: no ``unfold_psi``, no
        ``k='full_bz'``, no ``SymMaps``.  The only shared machinery is the
        density quadrature itself
        (:func:`psp.get_DFT_mtxels.valence_density_from_kpoint`, imported,
        never re-derived) and the G→FFT-box index table
        (:meth:`file_io.wfn_loader.WfnLoader.box_index` with ``k='ibz'``,
        which reads ``wfns/gvecs`` directly).

    "Imported, never re-derived" is the load-bearing clause, and injection
    keeps it: the caller hands in the SAME production function, so there
    is still exactly one density quadrature and one FFT-normalisation
    convention in the code base.  What injection buys is the service
    boundary — ``psp`` is an L1 physics kernel and this package must be
    importable without it (WAVE1_BRIEF ruling 4).

    ``valence_density_fn``
        ``valence_density_from_kpoint(arr, *, nocc, weight, cell_volume,
        spin_degeneracy)``.  ``None`` (the default) performs EXACTLY the
        lazy ``from psp.get_DFT_mtxels import valence_density_from_kpoint``
        this module has always performed, at the same point in the call.
    ``spin_degeneracy_fn``
        ``spin_degeneracy_factor(loader) -> float``.  Same contract, same
        default, same lazy import.
    ``spin_density_fields_fn``
        ``spin_density_matrix_to_pauli_fields(rho_ab)``.  Used only for a
        two-component spinor file and likewise lazy-imported by default.

    Ruling 4 is a COMPAT DEFAULT by construction: every call site in the
    tree passes neither, so every call site is unchanged and runs the same
    two imports it ran before.  Adoption of the explicit form at
    ``file_io/wfn_loader.py:327`` is registered to the wfn_loader
    orchestrator, not done here.
    """
    t0 = time.perf_counter()
    tol_trs = TOL_TRS if tol_trs is None else float(tol_trs)
    tol_spatial = TOL_SPATIAL if tol_spatial is None else float(tol_spatial)
    max_k = MAX_K_DEFAULT if max_k is None else int(max_k)
    messages: list[str] = []

    if spin_degeneracy_fn is None:
        from psp.get_DFT_mtxels import spin_degeneracy_factor
    else:
        spin_degeneracy_factor = spin_degeneracy_fn

    nspin = int(getattr(loader, "nspin", 1) or 1)
    nspinor = int(getattr(loader, "nspinor", 1) or 1)
    f_spin = float(spin_degeneracy_factor(loader))
    nb_file = int(loader.nbands)
    if nocc is None:
        nocc = int(loader.physical_density_band_stop)
    nocc = max(1, min(int(nocc), nb_file))
    occupation_weights = loader.physical_density_occupations(
        k="file", unit_as_none=True)
    if occupation_weights is not None:
        occupation_weights = occupation_weights[:, :nocc]

    fft_grid = tuple(int(s) for s in loader.fft_grid)
    kpoints = np.asarray(loader.kpoints, dtype=np.float64)
    weights = np.asarray(loader.kweights, dtype=np.float64)
    nk_total = int(kpoints.shape[0])
    kgrid = np.asarray(loader.kgrid, dtype=np.int64)
    shift = np.asarray(getattr(loader, "shift", (0.0, 0.0, 0.0)),
                       dtype=np.float64)
    ntran = int(getattr(loader, "ntran", 1) or 1)
    sym_matrices = np.asarray(loader.sym_matrices)[:ntran]
    translations = np.asarray(loader.translations, dtype=np.float64)[:ntran]
    # k-space action (SymMaps convention: sym_mats_k = mtrx.T).  SPATIAL
    # HALF ONLY — the TRS rows are exactly what this routine must not
    # assume.
    sym_mats_k = sym_matrices.transpose(0, 2, 1).astype(np.int64)

    kint, kint_exact = _k_as_ints(kpoints, kgrid, shift)
    if kint_exact:
        pair = _negation_pairs(kint, kgrid)
        little = _little_groups(kint, sym_mats_k, kgrid)
        trs_implied = _mesh_needs_trs(kint, sym_mats_k, kgrid)
    else:
        messages.append(
            "k-points do not land on the stored kgrid (shifted or "
            "irregular mesh): ± pairing and little groups unavailable")
        pair = np.full(nk_total, -1, dtype=np.int64)
        little = [np.zeros(0, dtype=np.int64) for _ in range(ntran)]
        trs_implied = None

    sel = _select_kpoints(kint, weights, pair, little, max_k)
    subsampled = bool(sel.size < nk_total)
    sel_set = {int(i) for i in sel}
    # Renormalise so ∫ρ = f_spin·nocc stays an exact statement about the
    # sampled sub-density (it still tests the quadrature normalisation
    # and the G→box scatter, which is what that invariant is for).
    w_sel = weights[sel] / float(np.sum(weights[sel]))

    # --- build the per-k densities ------------------------------------
    g_index = loader.box_index(k="ibz")
    ngkmax = int(loader.ngkmax)
    cell_volume = float(loader.cell_volume)
    rho_k: dict[int, np.ndarray] = {}
    m_k: dict[int, np.ndarray] = {}
    t_io = t_quad = 0.0
    for j, ik in enumerate(sel):
        if occupation_weights is not None:
            r = np.zeros(fft_grid, dtype=np.float64)
            m = None
            windows = loader.uniform_band_windows(
                0, nocc, FRACTIONAL_BAND_CHUNK)
            for b_lo, band_mask in windows:
                b_hi = b_lo + int(band_mask.size)
                t = time.perf_counter()
                psi = _raw_ibz_psi_k(
                    loader, int(ik), b_hi, b_lo=b_lo)
                box = _to_box(psi, g_index[int(ik)], ngkmax)
                t_io += time.perf_counter() - t
                t = time.perf_counter()
                r_chunk, m_chunk = _spin_resolved_density(
                    box, nocc=b_hi - b_lo, weight=float(w_sel[j]),
                    cell_volume=cell_volume, spin_degeneracy=f_spin,
                    band_occupations=(occupation_weights[
                        int(ik), b_lo:b_hi] * band_mask),
                    want_transverse=want_transverse,
                    valence_density_fn=valence_density_fn,
                    spin_density_fields_fn=spin_density_fields_fn)
                t_quad += time.perf_counter() - t
                r += r_chunk
                if m_chunk is not None:
                    if m is None:
                        m = np.zeros_like(m_chunk)
                    m += m_chunk
            rho_k[int(ik)] = r
            if m is not None:
                m_k[int(ik)] = m
            continue
        t = time.perf_counter()
        psi = _raw_ibz_psi_k(loader, int(ik), nocc)
        box = _to_box(psi, g_index[int(ik)], ngkmax)
        t_io += time.perf_counter() - t
        t = time.perf_counter()
        r, m = _spin_resolved_density(
            box, nocc=nocc, weight=float(w_sel[j]),
            cell_volume=cell_volume, spin_degeneracy=f_spin,
            band_occupations=None,
            want_transverse=want_transverse,
            valence_density_fn=valence_density_fn,
            spin_density_fields_fn=spin_density_fields_fn)
        t_quad += time.perf_counter() - t
        rho_k[int(ik)] = r
        if m is not None:
            m_k[int(ik)] = m

    rho = np.zeros(fft_grid, dtype=np.float64)
    for r in rho_k.values():
        rho += r
    rho_scale = float(np.max(np.abs(rho))) or 1.0

    # --- 3. basic invariants ------------------------------------------
    ngrid = int(np.prod(fft_grid))
    charge = float(np.sum(rho)) * cell_volume / ngrid
    if occupation_weights is None:
        charge_expected = f_spin * float(nocc)
    else:
        charge_expected = f_spin * float(np.einsum(
            "k,kb->", w_sel, occupation_weights[sel], optimize=True))
    charge_rel = abs(charge - charge_expected) / max(1.0, abs(charge_expected))
    rho_min_rel = float(np.min(rho)) / rho_scale
    invariants_ok = bool(charge_rel < 1.0e-6 and rho_min_rel > -1.0e-8)
    if not invariants_ok:
        messages.append(
            f"density invariants violated: ∫ρ={charge:.6f} vs expected "
            f"{charge_expected:.6f} (rel {charge_rel:.2e}); "
            f"min(ρ)/max(ρ)={rho_min_rel:.2e}")

    # --- (T5) is the occupied-band cut inside a degenerate multiplet? --
    # A band-INDEX cut is TRS-consistent because ε_{n,−k} = ε_{n,k} band
    # by band; the only way it can break (★★★) is if it splits a
    # degeneracy, where the diagonaliser's ordering among degenerate
    # partners is arbitrary and need not agree between k and −k.
    # Reported (not enforced) so any nonzero residual can be attributed.
    manifold_gap = float("nan")
    try:
        en = np.asarray(loader.energies)          # (nspin, nk, nb)
        if nocc < en.shape[-1]:
            gaps = en[..., nocc] - en[..., nocc - 1]
            manifold_gap = float(np.min(gaps[:, sel]))
            spread = float(np.ptp(en[..., :nocc]))
            if spread > 0 and manifold_gap < 1.0e-6 * spread:
                messages.append(
                    f"occupied-band cut at nocc={nocc} sits inside a "
                    f"degenerate multiplet (min ε[nocc]−ε[nocc−1] = "
                    f"{manifold_gap:.3e} over a {spread:.3e} band spread). "
                    f"Any TRS residual below may be an artefact of the cut "
                    f"rather than broken time reversal — see (T5).")
    except Exception:
        pass

    # --- 1. TRS --------------------------------------------------------
    if nspinor == 1 and nspin == 1:
        trs_holds, trs_basis = True, "spin-degenerate"
        m_rel = m_rel_total = None
        coverage = 1.0
        messages.append(
            "nspin=1, nspinor=1: the occupied manifold is spin-degenerate "
            "by construction, so m ≡ 0 identically and TRS cannot be "
            "broken by these wavefunctions")
    elif nspinor == 1 and nspin == 2:
        # Collinear spin-polarised WFN.  m_z = ρ_↑ − ρ_↓ and m_x = m_y ≡ 0
        # by construction.  LORRAX's reader does NOT support this layout:
        # ``wfns/coeffs`` axis 1 is treated as the SPINOR axis throughout
        # (``_eager_build``, ``to_box``, the phdf5 tables), and there is no
        # spin axis anywhere in ``WfnLoader``.  Rather than silently
        # measure whatever happens to sit on axis 1, say so.
        trs_holds, trs_basis = True, "unsupported"
        m_rel = m_rel_total = None
        coverage = 0.0
        messages.append(
            "nspin=2 (collinear spin-polarised) WFN: the magnetization is "
            "m_z = ρ_↑ − ρ_↓, but LORRAX's WFN reader has no spin axis "
            "(coeffs axis 1 is the SPINOR axis everywhere), so the two "
            "channels are not addressable and TRS cannot be measured. "
            "Verdict left permissive — but note a nspin=2 deck is the "
            "MOST likely place for TRS to actually be broken, so treat "
            "any such run as unverified.")
    else:
        covered_w = 0.0
        m_total = np.zeros((3,) + fft_grid, dtype=np.float64)
        pair_max = 0.0
        done: set[int] = set()
        for j, ik in enumerate(sel):
            ik = int(ik)
            if ik in done or ik not in m_k:
                continue
            jk = int(pair[ik])
            if jk < 0 or jk not in sel_set:
                continue
            # (★★★) of the module docstring: m_{−k}(r) = −m_k(r)
            # pointwise, so ``w·m_k + w·m_{−k}`` must vanish.  At a
            # self-paired k (a TRIM, ``−k ≡ k`` mod G) the identity
            # collapses to ``m_k ≡ 0`` — case ``jk == ik`` below.  No
            # symmetry operation, τ-phase, umklapp vector or U_spinor
            # appears anywhere in this loop; that independence is the
            # design, not an accident (see "NON-SYMMORPHIC OPS AND
            # SPINORS" in the module docstring).
            if jk != ik and not np.isclose(weights[ik], weights[jk],
                                           rtol=1e-9, atol=0.0):
                messages.append(
                    f"k {ik} and its −k partner {jk} carry unequal weights "
                    f"({weights[ik]:.6g} vs {weights[jk]:.6g}); pair excluded")
                continue
            resid = m_k[ik] if jk == ik else (m_k[ik] + m_k[jk])
            m_total += resid
            pair_max = max(pair_max, float(np.max(np.abs(resid))))
            done.add(ik)
            done.add(jk)
            covered_w += float(w_sel[j])
            if jk != ik:
                covered_w += float(w_sel[int(np.flatnonzero(sel == jk)[0])])
        coverage = covered_w
        if not done:
            trs_holds, trs_basis = True, "unmeasurable"
            m_rel = m_rel_total = None
            messages.append(
                "no ± k-pair present in the file's k list"
                + (" — consistent with an IBZ reduced using time reversal,"
                   if trs_implied else " (unexpected: the spatial star"
                   " already covers the kgrid),")
                + " which QE only does when TRS is assumed and which a "
                "TRS-broken system cannot produce; verdict defaults to "
                "TRS holds (see module docstring)")
        else:
            m_rel = pair_max / rho_scale
            m_rel_total = float(np.max(np.abs(m_total))) / rho_scale
            trs_holds = bool(m_rel < tol_trs)
            trs_basis = "measured"
            if not trs_holds and trs_implied:
                messages.append(
                    "FLAGS DISAGREE WITH THE MEASUREMENT: this file's "
                    "k-point list cannot cover its own kgrid using the "
                    "stored spatial operations alone, so its IBZ was "
                    "reduced ASSUMING time reversal — but the density "
                    "built from its own wavefunctions says time reversal "
                    "is broken. One of the two is wrong and the file "
                    "cannot be unfolded correctly either way. Regenerate "
                    "the nscf with `noinv=.true.`.")
            if not trs_holds:
                messages.append(
                    f"TIME-REVERSAL SYMMETRY IS BROKEN for these "
                    f"wavefunctions: max over ± k-pairs of "
                    f"‖w·m_k + w·m_(−k)‖∞/‖ρ‖∞ = {m_rel:.3e} > {tol_trs:.1e}. "
                    f"The occupied manifold carries a magnetization density; "
                    f"any time-reversal row in the symmetry table would "
                    f"produce a wrong ψ. "
                    f"[audit: occupied-cut gap ε[{nocc}]−ε[{nocc - 1}] = "
                    f"{manifold_gap:.3e}; if that is ~0 see (T5) in "
                    f"symmetry_maps/density_symmetry_check.py before "
                    f"believing this verdict]")

    # --- 2. spatial table ---------------------------------------------
    spatial_residual = np.full(ntran, np.nan, dtype=np.float64)
    untested: list[int] = []
    failed: list[int] = []
    for s in range(ntran):
        members = [int(i) for i in little[s] if int(i) in sel_set]
        if not members:
            untested.append(s)
            continue
        perm = _grid_permutation(sym_matrices[s], translations[s], fft_grid)
        if perm is None:
            untested.append(s)
            messages.append(
                f"op {s}: τ={translations[s] / (2 * np.pi)} is not "
                f"commensurate with the FFT grid {fft_grid} — the op is "
                f"not representable as a grid permutation, not tested")
            continue
        rho_s = np.zeros(fft_grid, dtype=np.float64)
        for i in members:
            rho_s += rho_k[i]
        scale_s = float(np.max(np.abs(rho_s)))
        if scale_s <= 0.0:
            untested.append(s)
            continue
        flat = rho_s.ravel()
        moved = np.empty_like(flat)
        moved[perm] = flat
        res = float(np.max(np.abs(moved - flat))) / scale_s
        spatial_residual[s] = res
        if res > tol_spatial:
            failed.append(s)
            messages.append(
                f"SYMMETRY OP {s} IS NOT A SYMMETRY OF THESE WAVEFUNCTIONS: "
                f"mtrx={sym_matrices[s].tolist()}, "
                f"τ/2π={np.round(translations[s] / (2 * np.pi), 6).tolist()} "
                f"gives ‖ρ(Sr+τ) − ρ(r)‖∞/‖ρ‖∞ = {res:.3e} > "
                f"{tol_spatial:.1e} (tested on {len(members)} k-point(s) of "
                f"its little group). The WFN symmetry block is wrong for "
                f"this deck.")
    if untested:
        messages.append(
            f"spatial ops not testable from the sampled k-set: "
            f"{tuple(untested)}")

    return DensitySymmetryReport(
        trs_holds=trs_holds, trs_basis=trs_basis, m_rel=m_rel,
        m_rel_total=m_rel_total, trs_coverage=float(coverage),
        tol_trs=tol_trs, trs_implied_by_mesh=trs_implied,
        spatial_ops_ok=(not failed), spatial_residual=spatial_residual,
        spatial_untested=tuple(untested), spatial_failed=tuple(failed),
        tol_spatial=tol_spatial,
        charge=charge, charge_expected=charge_expected,
        charge_rel_err=charge_rel, rho_min_rel=rho_min_rel,
        invariants_ok=invariants_ok, manifold_gap=manifold_gap,
        path=str(getattr(loader, "_path", "<wfn>")), nocc=nocc, nspin=nspin,
        nspinor=nspinor, n_k_used=int(sel.size), n_k_total=nk_total,
        subsampled=subsampled, fft_grid=fft_grid,
        seconds=time.perf_counter() - t0, seconds_io=t_io, seconds_quad=t_quad,
        messages=tuple(messages),
    )


# ----------------------------------------------------------------------
# Cached entry point used by WfnLoader
# ----------------------------------------------------------------------
#: (realpath, mtime_ns, size, nocc, tol_trs, tol_spatial, max_k) → report
_CACHE: dict[tuple, DensitySymmetryReport] = {}


def _cache_key(loader, nocc, tol_trs, tol_spatial, max_k) -> tuple:
    path = os.path.realpath(str(getattr(loader, "_path", "")))
    try:
        st = os.stat(path)
        stamp = (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        stamp = (0, 0)
    return (path, *stamp, int(nocc), float(tol_trs), float(tol_spatial),
            int(max_k))


def cached_density_symmetry_check(
    loader,
    *,
    valence_density_fn=None,
    spin_degeneracy_fn=None,
    spin_density_fields_fn=None,
) -> DensitySymmetryReport | None:
    """Run (or replay) the check for ``loader``; honours the env switches.

    Returns ``None`` when the check is disabled.  Never raises unless
    ``LORRAX_TRS_CHECK=strict``: a diagnostic that can abort a production
    run on its own bugs is worse than no diagnostic, so unexpected
    failures degrade to a warning plus the permissive (status-quo)
    verdict.

    ``valence_density_fn`` / ``spin_degeneracy_fn`` /
    ``spin_density_fields_fn`` are handed straight to
    :func:`check_density_symmetries`; see its docstring for the
    non-circularity argument they preserve.  They are deliberately NOT in
    :func:`_cache_key`: the key identifies the FILE and the tolerances,
    and two callers passing the production quadrature two ways must not
    get two cache entries for the same measurement.  A caller injecting a
    DIFFERENT quadrature is asking a different question and should call
    :func:`check_density_symmetries` directly.
    """
    mode = trs_check_mode()
    if mode == "off":
        return None

    tol_trs = _env_float("LORRAX_TRS_TOL", TOL_TRS)
    tol_spatial = _env_float("LORRAX_TRS_SPATIAL_TOL", TOL_SPATIAL)
    max_k = _env_int("LORRAX_TRS_MAX_K", MAX_K_DEFAULT)
    nocc = int(loader.physical_density_band_stop)

    key = _cache_key(loader, nocc, tol_trs, tol_spatial, max_k)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    try:
        rep = check_density_symmetries(
            loader, tol_trs=tol_trs, tol_spatial=tol_spatial, max_k=max_k,
            valence_density_fn=valence_density_fn,
            spin_degeneracy_fn=spin_degeneracy_fn,
            spin_density_fields_fn=spin_density_fields_fn)
    except Exception as exc:                      # pragma: no cover - guard
        if mode == "strict":
            raise
        warnings.warn(
            f"density symmetry check failed to run on "
            f"{getattr(loader, '_path', '<wfn>')}: {exc!r}. Proceeding with "
            f"the permissive (flags-derived) symmetry treatment. Set "
            f"LORRAX_TRS_CHECK=0 to disable this check.", RuntimeWarning)
        rep = _skipped_report(getattr(loader, "_path", "<wfn>"), repr(exc))
        _CACHE[key] = rep
        return rep

    _CACHE[key] = rep
    _announce(rep, mode)
    return rep


def _announce(rep: DensitySymmetryReport, mode: str) -> None:
    if _is_reporting_process():
        print(rep.summary(), flush=True)
    problems = [msg for msg in rep.messages
                if msg.isupper() or "BROKEN" in msg or "NOT A SYMMETRY" in msg
                or "violated" in msg]
    if not rep.ok:
        detail = "\n  ".join(rep.messages)
        text = (f"WFN density symmetry check FAILED for {rep.path}:\n  "
                f"{detail}")
        if mode == "strict":
            raise RuntimeError(text)
        print(f"\n*** {text}\n", file=sys.stderr, flush=True)
        warnings.warn(text, RuntimeWarning)
    elif problems and _is_reporting_process():
        for msg in problems:
            print(f"  [density-symmetry] {msg}", flush=True)
