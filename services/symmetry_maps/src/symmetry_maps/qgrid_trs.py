"""The q-grid time-reversal policy — one source of truth, MEASURED.

WHAT THIS MODULE OWNS
---------------------
Every decision the IBZ→full-BZ q unfold takes about *time reversal*:

1. which spatial row each full-BZ q uses (pair coherence across q/−q),
2. whether a q/−q pair may be composed through Θ at all,
3. whether the one-element Θ projector fires at a self-negative q,
4. how much anti-Θ component that projector removed (it is reported, not
   silently discarded),
5. and the measurement of the point-group covariance the unfold assumes
   of the finite ISDF ζ basis.

Three charge-channel producers consume it — bare ``V_q``
(``gw/v_q_g_flat.py``), RPA ``W_q`` (``gw/screening.py``) and ladder
``W_q`` (``gw/screening_bse.py``) — through the one adapter
``gw.qgrid_symmetry.qgrid_trs_policy_for``, which supplies rank-0 and the
once-per-run announcement.  ``tests/test_qgrid_trs_policy.py`` is the
ratchet that keeps it to one door.

WHY IT EXISTS: TRS IS MEASURED, NEVER ASSUMED
---------------------------------------------
The three producers used to compose q with −q through time reversal
*unconditionally*, and to project every self-negative q row onto its
Θ-invariant part, with no reference to whether time reversal is a symmetry
of the wavefunctions at hand.  ``density_symmetry_check`` already MEASURES
that from the occupied two-component DFT subspaces and publishes the verdict
as ``WfnLoader.trs_holds`` → ``SymMaps.trs_allowed``; nothing on the q axis
read it.

On ferromagnetic CrI3 (JID 57271494) the consequence was terminal: q and
−q are *independent* irreducible parents there, so the composition had no
relation to fabricate, and the run died in the unfold after a 685.96-GB ζ
fit had already completed.  Had the parents happened to coincide, the
composition would instead have silently overwritten one independently
solved row with the conjugate of the other — a fabricated symmetry, no
symptom, wrong screening.

So the policy has no TRS branch to guard: ``trs_measured`` is a REQUIRED
constructor argument with no default, and when it is false the returned
policy contains no time-reversal operation of any kind.  Its ``sym_idx``
map is the identity, its projector is a no-op, and it REFUSES a table that
selected a Θ row (which a ``SymMaps`` built from a magnetic WFN can never
produce, and which therefore means the tables and the verdict came from
different objects).

WHAT IS *NOT* A TRS STATEMENT, AND MUST NOT BE GATED ON THE VERDICT
-------------------------------------------------------------------
``V_{−q} = conj(V_q)`` for the charge-channel ISDF operator is NOT a
consequence of time-reversal symmetry of the wavefunctions.  The pair
densities fitted at −q, ``{u*_{n,k} u_{m,k−q}}``, are the complex
conjugates of those fitted at ``+q`` with bra and ket relabelled, for any
mean field whatsoever; ``v(|q+G|)`` is real and even in ``q+G``.  So the
reciprocity is a statement about the conjugation-equivariance of the fit,
it holds on a ferromagnet, and ``common.sanity.check_q_conjugate_reciprocity``
stays armed on a TRS-broken deck.  What changes on a magnet is only that
the relation is no longer *imposed* by the unfold: q and −q are solved
independently, so the gate becomes an independent measurement instead of
an identity — which is strictly more informative.

THE COVARIANCE ASSUMPTION, AND WHY IT IS MEASURED HERE
------------------------------------------------------
``unfold_isdf_operator`` reconstructs ``V_full[q]`` from
``V_ibz[i(q)]`` by permuting centroids and applying an umklapp phase.  For
that to be well defined the *stored parent tile* must be invariant under
the little group of its own q — an exact property of the continuum
operator and only an approximate one of a finite, possibly
ill-conditioned, ISDF fit.  It was measured at **1.240e-02** at Γ on the
Na 8×8×8 SOC c464 deck (47 of 48 ops), where the q↔−q reciprocity gate
simultaneously read 3.9e-17 because at a self-negative q that gate
degenerates to "``V_q`` is real".  :func:`little_group_covariance_residual`
is the statistic that is not blind there: it applies the unfold's OWN
formula with an op that maps the parent to itself and asks whether the
tile comes back.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "QgridTrsPolicy",
    "build_qgrid_trs_policy",
    "little_group_covariance_residual",
    "self_negative_q_mask",
    "trs_pair_coherent_unfold_sym_idx",
    "trs_project_self_negative_q_rows",
]


def self_negative_q_mask(q_full_idx, *, kgrid) -> np.ndarray:
    """``q ≡ −q`` (mod the mesh) for each row of ``q_full_idx``.

    These are the one-element orbits of the ``q → −q`` involution — the
    TRIM points of an even mesh, and Γ alone on a fully odd one.  They are
    the rows a pair composition can never touch (there is no partner) and
    the only rows the Θ projector may act on.
    """
    grid = np.asarray(tuple(int(v) for v in kgrid), dtype=np.int64)
    qidx = np.asarray(q_full_idx, dtype=np.int64).reshape(-1)
    n_full = int(np.prod(grid))
    if np.any(qidx < 0) or np.any(qidx >= n_full):
        raise ValueError(
            "GATE trs_fixed_q_projector: q_full_idx lies outside the "
            f"full k-grid extent {n_full}: {qidx.tolist()}.")
    coords = np.stack(np.unravel_index(qidx, tuple(grid)), axis=1)
    return np.all((2 * coords) % grid[None, :] == 0, axis=1)


def trs_pair_coherent_unfold_sym_idx(
    irr_idx, sym_idx, *, kgrid, q_irr_full_idx, n_sym_spatial,
):
    """Choose one spatial realization for every non-TRIM q/−q pair.

    THE MEASURED-TRS ARM ONLY.  Callers reach this through
    :class:`QgridTrsPolicy`, which refuses to call it at all when the
    density measurement says time reversal is broken.  It is public
    because the two policy arms must be separately testable, not because a
    driver may pick one.

    A centrosymmetric crystal can map both q and −q from one irreducible
    parent using two unrelated *spatial* rows.  That is mathematically
    equivalent only when the operator stored in the finite ISDF basis is
    exactly point-group covariant — the assumption
    :func:`little_group_covariance_residual` measures, and which the Na
    8×8×8 SOC c464 deck violates by 1.2e-2.  The scalar-Si production
    discriminator showed the same shape: its closed 50-band wavefunction
    subspace is covariant at 2.19e-9, while the 648-centroid fitted
    operator is not covariant closely enough to survive the
    ill-conditioned ζ solve, and the unrelated-coset choice converted that
    ordinary fit error into a forbidden 9.7e-2 TRS error.

    Keep one member's selected spatial row and make the other member use
    the same row composed with time reversal.  ``unfold_isdf_operator``
    then applies its derived conjugation rule, so q/−q reciprocity depends
    only on Θ and not on an unrelated spatial gauge.  When every full-BZ q
    row was solved independently there is no unfold relation to choose:
    return the original rows unchanged.  Irreducible representatives and
    self-negative q rows are unchanged; no value is averaged or projected.
    """
    grid = tuple(int(v) for v in kgrid)
    n_full = int(np.prod(grid))
    irr = np.asarray(irr_idx, dtype=np.int32)
    original = np.asarray(sym_idx, dtype=np.int32)
    n_spatial = int(n_sym_spatial)
    if irr.shape != (n_full,) or original.shape != (n_full,):
        raise ValueError(
            "GATE trs_pair_unfold_map: irr_idx and sym_idx must each have "
            f"the full k-grid extent {n_full}; got {irr.shape} and "
            f"{original.shape} for kgrid={grid}.")
    if (n_spatial <= 0 or np.any(original < 0)
            or np.any(original >= 2 * n_spatial)):
        lo = int(original.min()) if original.size else 0
        hi = int(original.max()) if original.size else -1
        raise ValueError(
            "GATE trs_pair_unfold_map: symmetry rows must lie in "
            f"[0, 2*n_sym_spatial) with n_sym_spatial={n_spatial}; got "
            f"range [{lo}, {hi}].")

    reps = set(int(v) for v in np.asarray(q_irr_full_idx).reshape(-1))
    out = original.copy()
    # An orbit-closed centroid set can still have no q reduction, for example
    # when the WFN exposes only the identity spatial operation.  Every q is
    # then a solved source row.  Rewiring q/−q would discard one independent
    # result and the guard below correctly rejects their distinct parents;
    # the owning policy is instead the identity map.
    if len(reps) == n_full:
        return out
    # The q-axis convention is owned by maps.q_negation_index; this policy
    # consumes the table instead of carrying another C-order spelling.
    from .maps import q_negation_index
    neg = q_negation_index(grid)
    for iq, jq_value in enumerate(neg):
        jq = int(jq_value)
        if iq >= jq:
            continue
        if int(irr[iq]) != int(irr[jq]):
            raise ValueError(
                "GATE trs_pair_unfold_map: q and -q do not share an "
                f"irreducible parent ({iq}->{int(irr[iq])}, "
                f"{jq}->{int(irr[jq])}).  A TRS composition is valid only "
                "for one solved wedge row; do not fabricate reciprocity "
                "across two independently solved rows.  On a deck whose "
                "DFT reference check says TIME REVERSAL IS BROKEN this is the "
                "expected table and the policy must not have reached here: "
                "build QgridTrsPolicy with trs_measured=False so the rows "
                "stay independent.")

        if iq in reps:
            source = iq
        elif jq in reps:
            source = jq
        elif (original[iq] < n_spatial) != (original[jq] < n_spatial):
            source = iq if original[iq] < n_spatial else jq
        else:
            source = iq
        partner = jq if source == iq else iq
        source_row = int(original[source])
        out[partner] = (source_row + n_spatial
                        if source_row < n_spatial
                        else source_row - n_spatial)
    return out


def trs_project_self_negative_q_rows(operator, q_full_idx, *, kgrid):
    """Apply the one-element Θ group projector at ``q == −q``.

    THE MEASURED-TRS ARM ONLY — see
    :meth:`QgridTrsPolicy.project_fixed_q`, which is the door and which
    also returns the residual this removes.

    Pair-coherent unfold handles every two-element q/−q orbit without
    changing a solved value.  A TRIM row is a one-element orbit, so there
    is no partner row from which to reconstruct it: the exact constraint
    is ``A_q = conj(A_q)`` in the restart convention.  A finite band sum
    or an ill-conditioned ISDF backsolve can leave a small anti-Θ
    component even when the non-TRIM pairs are exact.  Remove only that
    component with the unique group average ``(A + conj(A))/2``.

    ``operator`` remains distributed exactly as supplied; this is an
    elementwise operation and performs no host gather.  ``q_full_idx``
    names the full-grid point represented by each leading-axis row, so the
    helper works on either an irreducible wedge or the full BZ.
    """
    import jax.numpy as jnp

    fixed = self_negative_q_mask(q_full_idx, kgrid=kgrid)
    n_rows = int(fixed.size)
    if int(operator.shape[0]) != n_rows:
        raise ValueError(
            "GATE trs_fixed_q_projector: operator q extent does not match "
            f"q_full_idx ({int(operator.shape[0])} != {n_rows}).")
    mask = jnp.asarray(fixed).reshape((n_rows,) + (1,) * (operator.ndim - 1))
    projected = 0.5 * (operator + jnp.conj(operator))
    return jnp.where(mask, projected, operator)


# Compiled kernels are cached on nothing but their own identity, because
# ``jax.jit`` keys on the FUNCTION OBJECT: a fresh ``@jax.jit`` inside a
# helper recompiles on every call, which on a self-consistency loop is one
# compile per iteration.  Same reason ``common.sanity`` caches ``_herm_stats``
# and ``_negq_stats``.  Everything that varies (the mask, the permutation,
# the phase) is a traced ARGUMENT, so one module serves every call at a
# given shape.
_JIT_CACHE: dict = {}


def _anti_trs_stats_fn():
    fn = _JIT_CACHE.get("anti_trs")
    if fn is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def fn(a, mask):
            anti = jnp.where(mask, jnp.abs(a - jnp.conj(a)), 0.0)
            return jnp.stack([
                jnp.max(anti).astype(jnp.float64),
                jnp.max(jnp.abs(a)).astype(jnp.float64),
            ])

        _JIT_CACHE["anti_trs"] = fn
    return fn


def _fixed_q_anti_trs_residual(operator, fixed_mask):
    """``[max|A − conj(A)| on Θ-fixed rows, max|A|]`` — one reduction.

    The number the projector is about to delete.  TASTE's calibration rule
    is that one never repairs a downstream object to make an identity
    true; the projector is kept because at a one-element orbit it is the
    exact group average and not a repair, but the size of what it removes
    is evidence and is reported rather than discarded.
    """
    import jax.numpy as jnp

    mask = jnp.asarray(fixed_mask).reshape(
        (int(fixed_mask.size),) + (1,) * (operator.ndim - 1))
    return _anti_trs_stats_fn()(operator, mask)


def _covariance_residual_fn():
    """``(V, p, alpha, phase) -> scalar`` — the unfold's own arithmetic.

    ONE compiled module for every (parent, op): the double gather, the
    umklapp phase and the reduction to a single scalar are fused, so no
    (μ,μ) intermediate is ever materialised outside the sharded graph.
    Eager operator-by-operator arithmetic here would resolve the
    gather/transpose shardings by all-gathering the tile onto every rank —
    the trap ``common.sanity._herm_stats`` documents.
    """
    fn = _JIT_CACHE.get("covariance")
    if fn is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def fn(V, p, alpha, phase):
            row = jnp.take(V, p, axis=0)
            image = jnp.take(jnp.take(row, alpha, axis=0), alpha, axis=1)
            image = image * (phase[:, None] * jnp.conj(phase)[None, :])
            return jnp.max(jnp.abs(image - row)).astype(jnp.float64)

        _JIT_CACHE["covariance"] = fn
    return fn


# ---------------------------------------------------------------------------
# The covariance the unfold ASSUMES — measured, not assumed
# ---------------------------------------------------------------------------

_COVARIANCE_OPS_BUDGET = 64


def little_group_covariance_residual(
    V_ibz,
    *,
    q_irr_frac,
    q_irr_full_idx,
    sym_mats_k,
    sym_perm,
    L_table,
    kgrid,
    n_sym_spatial,
    parents="self_negative",
    ops_budget: int = _COVARIANCE_OPS_BUDGET,
) -> dict:
    """Does the stored parent tile survive its OWN little group?

    THE STATISTIC THE q↔−q GATE CANNOT SEE.  ``unfold_isdf_operator``
    reconstructs a full-BZ row as

        ``V[q,μ,ν] = e^{2πi q_p·(L_{s,μ}−L_{s,ν})} V_p[α_s(μ), α_s(ν)]``

    For ``s`` in the little group of ``q_p`` (``S_s q_p ≡ q_p``) that
    formula must return ``V_p`` itself, because the reconstructed point is
    the parent.  It is exact for the continuum operator and only
    approximate for a finite ISDF fit, so the residual

        ``max_s max_{μν} |phase·V_p[α_s μ, α_s ν] − V_p[μ,ν]| / max|V_p|``

    is the whole of the assumption, stated in the unfold's own arithmetic
    and needing no second convention.  On Na 8×8×8 SOC c464 it reads
    **1.240e-02 at Γ** and 2.411e-02 at H, exactly where
    ``check_q_conjugate_reciprocity`` reads 3.9e-17 and 6.4e-17.

    SENSITIVITY, stated because a null here would otherwise be quoted as
    coverage it does not have.  This sees the covariance of the STORED
    PARENT TILES only.  It is blind to an error in the centroid
    permutation tables themselves (they are the reference), to the Coulomb
    kernel (a Gram defect shows identically with ``v ≡ 1``), and to
    anything about the full-BZ rows that are not parents.

    Parameters
    ----------
    V_ibz
        ``(n_q_ibz, n_rmu, n_rmu)`` — the *pre-unfold* wedge, at whatever
        sharding the producer holds.  Nothing is gathered: every op is a
        centroid-axis double gather plus one reduction to two scalars.
    parents
        ``"self_negative"`` (default) restricts to the parents whose q is
        its own negative — the rows where the reciprocity gate is blind
        and where the defect was largest.  ``"all"`` sweeps every parent.
    ops_budget
        Upper bound on ``(parent, op)`` pairs actually evaluated.  When the
        little groups are larger than this the ops are STRIDE-sampled
        (deterministic, reproducible, and reported as
        ``n_ops_sampled``/``n_ops_available``) rather than truncated to a
        prefix, because a prefix would systematically favour the low-index
        ops and the identity is always index 0.

    Cost.  One centroid-axis double gather per evaluated pair — the SAME
    operation :func:`unfold_isdf_operator` performs once per full-BZ q.
    The default budget is therefore bounded by a fraction of the unfold
    that immediately follows it (≤64 against 512 on an 8×8×8 mesh), and
    nothing is gathered to the host: each pair reduces to one scalar.

    Returns
    -------
    dict
        ``max_rel``, ``max_abs``, ``scale``, ``worst_parent``,
        ``worst_sym``, ``n_ops_sampled``, ``n_ops_available``,
        ``n_parents``.  ``max_rel`` is ``nan`` when no ``(parent, op)``
        pair other than the identity exists — an unanswerable question is
        returned as unanswerable rather than as a pass.
    """
    import jax
    import jax.numpy as jnp

    if parents not in ("self_negative", "all"):
        raise ValueError(
            "little_group_covariance_residual: parents must be "
            f"'self_negative' or 'all'; got {parents!r}.")
    grid = np.asarray(tuple(int(v) for v in kgrid), dtype=np.int64)
    reps = np.asarray(q_irr_full_idx, dtype=np.int64).reshape(-1)
    n_spatial = int(n_sym_spatial)
    perm = np.asarray(sym_perm, dtype=np.int32)
    L_arr = np.asarray(L_table, dtype=np.float64)
    q_frac = np.asarray(q_irr_frac, dtype=np.float64)
    S = np.asarray(sym_mats_k, dtype=np.int64)[:n_spatial]

    if int(V_ibz.shape[0]) != int(reps.size):
        raise ValueError(
            "little_group_covariance_residual: V_ibz has "
            f"{int(V_ibz.shape[0])} q rows but q_irr_full_idx names "
            f"{int(reps.size)}.  Pass the PRE-unfold wedge.")
    if int(perm.shape[-1]) != int(V_ibz.shape[-1]):
        raise ValueError(
            "little_group_covariance_residual: sym_perm centroid extent "
            f"{int(perm.shape[-1])} != V_ibz extent {int(V_ibz.shape[-1])}; "
            "both carry the μ pad from _resolve_ibz_q_list.")

    coords = np.stack(np.unravel_index(reps, tuple(grid)), axis=1)
    if parents == "self_negative":
        keep = np.flatnonzero(np.all((2 * coords) % grid[None, :] == 0, axis=1))
    else:
        keep = np.arange(reps.size, dtype=np.int64)

    # Little group of each retained parent, in INTEGER mesh coordinates so
    # the ``≡ mod grid`` is exact (no float tolerance anywhere).
    pairs: list[tuple[int, int]] = []
    for p in keep:
        qc = coords[int(p)]
        images = (S @ qc) % grid[None, :]
        for s in np.flatnonzero(np.all(images == qc[None, :], axis=1)):
            if int(s) == 0:
                continue        # identity: residual is 0 by construction
            pairs.append((int(p), int(s)))
    n_available = len(pairs)
    if n_available == 0:
        return {
            "max_rel": float("nan"), "max_abs": float("nan"),
            "scale": float("nan"), "worst_parent": -1, "worst_sym": -1,
            "n_ops_sampled": 0, "n_ops_available": 0,
            "n_parents": int(keep.size),
        }
    budget = max(1, int(ops_budget))
    if n_available > budget:
        stride = int(np.ceil(n_available / budget))
        pairs = pairs[::stride][:budget]

    _residual = _covariance_residual_fn()
    scale = float(jax.device_get(jnp.max(jnp.abs(V_ibz))))
    worst_abs, worst_parent, worst_sym = -1.0, -1, -1
    for p, s in pairs:
        qL = 2.0 * np.pi * (L_arr[s] @ q_frac[p])          # (n_rmu,)
        dev = float(jax.device_get(_residual(
            V_ibz, jnp.int32(p), jnp.asarray(perm[s]),
            jnp.exp(1j * jnp.asarray(qL)))))
        if dev > worst_abs:
            worst_abs, worst_parent, worst_sym = dev, p, s
    return {
        "max_rel": (worst_abs / scale) if scale > 0.0 else worst_abs,
        "max_abs": worst_abs,
        "scale": scale,
        "worst_parent": int(worst_parent),
        "worst_sym": int(worst_sym),
        "n_ops_sampled": len(pairs),
        "n_ops_available": int(n_available),
        "n_parents": int(keep.size),
    }


# ---------------------------------------------------------------------------
# The policy object every driver consumes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QgridTrsPolicy:
    """What time reversal is allowed to do to this deck's q axis.

    Built by :func:`build_qgrid_trs_policy` (or, in the monorepo, by the
    announcing adapter ``gw.qgrid_symmetry.qgrid_trs_policy_for``).  A
    driver reads :attr:`unfold_sym_idx` and calls :meth:`project_fixed_q`;
    it takes NO time-reversal decision of its own and has no ``if trs:``
    branch left to get wrong.

    ``trs_measured`` is the verdict ``density_symmetry_check`` obtained
    from the two-component DFT reference check, arriving through
    ``SymMaps.trs_allowed``.  It has no default anywhere on this path.
    """

    trs_measured: bool
    kgrid: tuple[int, int, int]
    n_sym_spatial: int
    unfold_sym_idx: np.ndarray
    self_negative_q: np.ndarray
    n_pair_rewired: int
    context: str = ""
    _selected_sym_idx: np.ndarray = field(default=None, repr=False)

    @property
    def n_self_negative(self) -> int:
        return int(np.count_nonzero(self.self_negative_q))

    def project_fixed_q(self, operator, q_full_idx):
        """``(operator, removed)`` — the Θ projector, and what it removed.

        On a measured-TRS deck this applies the one-element group average
        at every ``q ≡ −q`` row and returns the anti-Θ residual it deleted
        so the caller can print it: a projector that silently absorbs a
        1e-2 defect is the "instrument that measures and proceeds" failure
        wearing a repair's clothes.

        On a TRS-BROKEN deck it is the identity and ``removed`` is
        ``None``.  There is no warrant for the projection there: the rows
        were solved independently and Θ is not a symmetry of this mean
        field, so ``V_q`` at a TRIM point is whatever the fit produced and
        the reciprocity gate should see it.
        """
        if not self.trs_measured:
            return operator, None
        fixed = self_negative_q_mask(q_full_idx, kgrid=self.kgrid)
        if int(operator.shape[0]) != int(fixed.size):
            raise ValueError(
                "GATE trs_fixed_q_projector: operator q extent does not "
                f"match q_full_idx ({int(operator.shape[0])} != "
                f"{int(fixed.size)}).")
        if not np.any(fixed):
            return operator, None
        import jax

        stats = _fixed_q_anti_trs_residual(operator, fixed)
        out = trs_project_self_negative_q_rows(
            operator, q_full_idx, kgrid=self.kgrid)
        dev, scale = (float(v) for v in np.asarray(jax.device_get(stats)))
        removed = (dev / scale) if scale > 0.0 else dev
        return out, removed

    def measure_covariance(self, V_ibz, **kwargs) -> dict:
        """:func:`little_group_covariance_residual` with the policy's grid.

        Independent of :attr:`trs_measured` — point-group covariance of
        the stored parent tiles is a SPATIAL property and is required by
        the unfold on a ferromagnet exactly as on a nonmagnetic deck.
        """
        kwargs.setdefault("kgrid", self.kgrid)
        kwargs.setdefault("n_sym_spatial", self.n_sym_spatial)
        return little_group_covariance_residual(V_ibz, **kwargs)

    def announcement(self) -> str:
        where = f" [{self.context}]" if self.context else ""
        if not self.trs_measured:
            return (
                f"q-grid TRS policy{where}: the DFT reference check says "
                f"TIME REVERSAL IS BROKEN for this WFN.  Every full-BZ q row "
                f"keeps the spatial parent it was solved from; no q/-q "
                f"composition and no fixed-q TRS projector is applied.  "
                f"q<->-q reciprocity of V/W is still gated — on this deck it "
                f"is an independent measurement rather than an identity of "
                f"the unfold.")
        return (
            f"q-grid TRS policy{where}: time reversal MEASURED to hold; "
            f"{int(self.n_pair_rewired)} of {int(self.unfold_sym_idx.size)} "
            f"q rows use a TRS-composed partner so q and -q share one "
            f"spatial realization, and {self.n_self_negative} self-negative "
            f"q rows carry the one-element TRS projector.")

    @property
    def announce_key(self) -> str:
        return (f"qgrid_trs_policy:{int(bool(self.trs_measured))}:"
                f"{self.kgrid}:{self.context}")


def build_qgrid_trs_policy(
    *,
    trs_measured: bool,
    irr_idx_q,
    sym_idx_q,
    q_irr_full_idx,
    kgrid,
    n_sym_spatial: int,
    context: str = "",
) -> QgridTrsPolicy:
    """Resolve the q-axis time-reversal policy for one centroid set.

    ``trs_measured`` is keyword-only and has NO DEFAULT: a caller that has
    not consulted the density measurement gets a ``TypeError`` rather than
    the historically permissive branch.  This is the same shape the tree
    already uses for ``trs_reference`` (13 explicit call sites, no
    default) and for the same reason — the wrong branch is invisible in
    the output.
    """
    grid = tuple(int(v) for v in kgrid)
    n_full = int(np.prod(grid))
    selected = np.asarray(sym_idx_q, dtype=np.int32)
    n_spatial = int(n_sym_spatial)
    if selected.shape != (n_full,):
        raise ValueError(
            "GATE trs_pair_unfold_map: sym_idx_q must have the full k-grid "
            f"extent {n_full}; got {selected.shape} for kgrid={grid}.")

    if not bool(trs_measured):
        # No guard, no branch, no projector: on a magnetic deck the policy
        # simply contains no time-reversal operation.  The one thing left
        # to check is that the TABLES agree with the VERDICT — a Θ row in
        # ``sym_idx_q`` means the wedge was reduced using a symmetry the
        # reference verdict says is absent, and the outcome is a refusal
        # rather than an unfold that conjugates a magnetic wavefunction.
        offenders = np.flatnonzero(selected >= n_spatial)
        if offenders.size:
            raise ValueError(
                "GATE trs_measured_vs_tables: the reference verdict says "
                "time reversal is BROKEN for this WFN, but the q-grid "
                f"symmetry table selects time-reversal rows at "
                f"{offenders.size} of {n_full} q "
                f"(first at q={int(offenders[0])}, row "
                f"{int(selected[offenders[0]])} >= n_sym_spatial="
                f"{n_spatial}).  A TRS-reduced q mesh for a magnetic system "
                "is not physical.  Build SymMaps from the loader that "
                "measured the density (SymMaps(wfn) reads wfn.trs_holds) so "
                "the search set excludes the Theta rows; do not pass "
                "hand-built tables beside a magnetic verdict.")
        return QgridTrsPolicy(
            trs_measured=False, kgrid=grid, n_sym_spatial=n_spatial,
            unfold_sym_idx=selected.copy(),
            self_negative_q=self_negative_q_mask(
                np.arange(n_full), kgrid=grid),
            n_pair_rewired=0, context=context,
            _selected_sym_idx=selected.copy())

    coherent = trs_pair_coherent_unfold_sym_idx(
        irr_idx_q, selected, kgrid=grid, q_irr_full_idx=q_irr_full_idx,
        n_sym_spatial=n_spatial)
    return QgridTrsPolicy(
        trs_measured=True, kgrid=grid, n_sym_spatial=n_spatial,
        unfold_sym_idx=coherent,
        self_negative_q=self_negative_q_mask(np.arange(n_full), kgrid=grid),
        n_pair_rewired=int(np.count_nonzero(coherent != selected)),
        context=context, _selected_sym_idx=selected.copy())
