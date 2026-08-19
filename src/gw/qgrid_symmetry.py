"""The q-grid symmetry decision, taken once and said out loud.

WHAT THIS MODULE IS.  One function.  It asks
``symmetry_maps.resolve_qgrid_symmetry`` whether this deck's centroid set
admits the IBZ q reduction, and — when it does not — emits ONE rank-0
announcement per run naming the failed closure and its consequence.  Every
producer of the q-axis tables in ``gw/`` goes through here; nothing else
in the monorepo calls ``centroid_source_map_and_wrap``, and
``tests/test_qgrid_symmetry_resolution.py`` is the ratchet that says so.

WHY IT EXISTS.  Before the consolidation, the closure question was asked
in three places and answered by catching the table builder's
``RuntimeError``:

* ``gw/v_q_g_flat.py:_resolve_ibz_q_list`` — caught it, set ``sym_perm =
  None``, and printed a line only when its ``verbose`` argument was true.
  ``gw/screening.py`` calls it with ``verbose=False``, so on a production
  run the W Dyson solve silently dropped from ``n_q_ibz`` blocks to
  ``n_q_full`` and no log said why.
* ``gw/v_q_bispinor.py`` — called the same helper twice and printed its
  own differently-worded line, also behind ``verbose``.
* ``gw/isdf_fitting.py`` — called the builder directly for a yes/no and
  flipped ``write_ibz_only`` off, printing unconditionally on rank 0 for
  the charge channel and RAISING for the transverse one.

Three spellings of one decision, two of them invisible in production, and
the consequence on today's 960-centroid deck is an ~8× larger restart
tensor and a 16.9 meV Σ star spread where a closed set measures 0.7.
The owner's ruling (DESIGN_symmetry_restart_followup.md, "The orbit-closure
program", item 3) is that the fallback becomes explicit and loud.  It does
not become an error: a deck whose centroids predate the orbit-aware
k-means still runs, exactly as it ran before, and now says so.

THE SEAM.  The service composes the announcement text (it holds the
residuals); this module emits it (it holds rank 0 and the once-per-run
memory).  ``symmetry_maps`` declares only jax and numpy and is tested in a
process where lorrax is not importable, so it cannot reach
``ffi.gate.announce_once`` itself — and should not want to.
"""
from __future__ import annotations

import numpy as np


def resolve_qgrid_symmetry_tables(
    *,
    sym,
    centroid_indices,
    fft_grid,
    context: str,
    translations=None,
    announce_fallback: bool = True,
):
    """Resolve the q-grid reduction for one centroid set, announcing once.

    Parameters
    ----------
    sym
        A ``symmetry_maps.SymMaps``.  ``sym.sym_matrices`` (spatial ops,
        BGW ``mtrx``) and ``sym.translations`` (BGW ``tnp`` = 2π·τ,
        already sliced to ``ntran``) are read off it.
    centroid_indices
        ``(n_rmu, 3)`` integer FFT-grid indices.  Device arrays are
        accepted; they are pulled to the host here.
    fft_grid
        ``(3,)`` int — the grid ``centroid_indices`` indexes.
    context
        Names the call site in the announcement, e.g. ``"V_q / W q-grid
        reduction"``.  The bispinor path passes the channel, because
        "the centroid set is not closed" is a different fact about the
        charge set than about the transverse one.
    translations
        Override for ``sym.translations`` (BGW ``tnp``).  Present for the
        one caller that holds the WFN rather than the SymMaps; passing
        ``None`` reads them off ``sym``.  Either way they are BGW's
        stored ``2π·τ`` and the division by 2π happens in exactly one
        place, inside the service.
    announce_fallback
        ``False`` on the ONE path whose consequence is not a fallback:
        ``gw/isdf_fitting.py``'s transverse ζ̃_T write REFUSES on a
        non-closed transverse centroid set (the V_q orchestrator assumes
        an IBZ ζ̃_T and there is nothing to degrade to), and it raises
        with its own message.  Printing "solving on the full BZ" beside a
        refusal would describe a run that is not happening.  Every other
        caller leaves this ``True``.

    Returns
    -------
    symmetry_maps.QgridSymmetryResolution
        ``.use_ibz`` is the predicate to branch on; ``.sym_perm`` /
        ``.L_table`` are the tables when it is true; ``.verdict`` carries
        the measured residuals either way.

    Notes
    -----
    The announcement is deduped on the CENTROID SET, not on the call
    site, so the V_q pass, the W Dyson solve and every self-consistency
    iteration speak once between them — while a bispinor deck's two
    genuinely different sets still get one line each.
    """
    from ffi import _services
    _services.ensure_on_path()
    from ffi.gate import announce_once
    from symmetry_maps import resolve_qgrid_symmetry

    import jax

    n_tran = int(np.asarray(sym.sym_matrices).shape[0])
    tnp = (np.asarray(sym.translations) if translations is None
           else np.asarray(translations))
    res = resolve_qgrid_symmetry(
        np.asarray(jax.device_get(centroid_indices), dtype=np.int32),
        np.asarray(sym.sym_matrices[:n_tran]),
        tnp=tnp[:n_tran],
        fft_grid=np.asarray(fft_grid, dtype=np.int32),
        extend_trs=True,
        context=context,
    )
    msg = res.announcement() if announce_fallback else None
    if msg is not None:
        # Rank-invariant fact (the centroid file is the same on every
        # rank), so scope="rank0"; keyed on the centroid hash so the
        # repeat resolves along the run are silent.
        announce_once(res.announce_key, msg, scope="rank0")
    return res


def trs_pair_coherent_unfold_sym_idx(
    irr_idx, sym_idx, *, kgrid, q_irr_full_idx, n_sym_spatial,
    trs_allowed=True,
):
    """Choose one spatial realization for every non-TRIM q/-q pair.

    A centrosymmetric crystal can map both q and -q from one irreducible
    parent using two unrelated *spatial* rows.  That is mathematically
    equivalent only when the operator stored in the finite ISDF basis is
    exactly point-group covariant.  The scalar-Si production discriminator
    showed why an unfold must not rely on that stronger condition: its
    closed 50-band wavefunction subspace is covariant at 2.19e-9, while the
    648-centroid fitted operator is not covariant closely enough to survive
    the ill-conditioned zeta solve.

    Keep one member's selected spatial row and make the other member use the
    same row composed with time reversal.  ``unfold_isdf_operator`` then
    applies its derived conjugation rule, so q/-q reciprocity depends only on
    TRS and not on an unrelated spatial gauge.  Irreducible representatives
    and self-negative q rows are unchanged; no value is averaged or
    projected.

    When the load-time density measurement says time reversal is broken,
    ``trs_allowed=False`` is the physics gate: every independently solved
    row is retained verbatim.  In particular, q and -q are then allowed to
    have different irreducible parents.  This is not a permissive fallback;
    imposing the conjugation relation in that case would change a magnetic
    operator.

    This policy was first shipped for ladder W in ``screening_bse``.  It is
    owned here because bare V, RPA W, and ladder W must use the same q-grid
    realization.
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

    if not bool(trs_allowed):
        return original.copy()

    reps = set(int(v) for v in np.asarray(q_irr_full_idx).reshape(-1))
    out = original.copy()
    coords = np.stack(np.unravel_index(np.arange(n_full), grid), axis=1)
    neg = np.ravel_multi_index(
        tuple(((-coords) % np.asarray(grid, dtype=np.int64)).T), grid)
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
                "across two independently solved rows.")

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


def trs_project_self_negative_q_rows(
    operator, q_full_idx, *, kgrid, trs_allowed=True,
):
    """Apply the one-element TRS group projector at q == -q.

    Pair-coherent unfold handles every two-element q/-q orbit without
    changing a solved value.  A TRIM row is a one-element orbit, so there is
    no partner row from which to reconstruct it: the exact constraint is
    ``A_q = conj(A_q)`` in the restart convention.  A finite band sum or an
    ill-conditioned ISDF backsolve can leave a small anti-TRS component even
    when the non-TRIM pairs are exact.  Remove only that component with the
    unique group average ``(A + conj(A))/2``.

    ``trs_allowed=False`` returns the operator unchanged after validating
    the q-axis contract.  A self-negative momentum does not make a magnetic
    operator real; only an actual time-reversal symmetry does.

    ``operator`` remains distributed exactly as supplied; this is an
    elementwise operation and performs no host gather.  ``q_full_idx`` names
    the full-grid point represented by each leading-axis row, so the helper
    works on either an irreducible wedge or the full BZ.
    """
    import jax.numpy as jnp

    grid = np.asarray(tuple(int(v) for v in kgrid), dtype=np.int64)
    qidx = np.asarray(q_full_idx, dtype=np.int64).reshape(-1)
    n_full = int(np.prod(grid))
    if np.any(qidx < 0) or np.any(qidx >= n_full):
        raise ValueError(
            "GATE trs_fixed_q_projector: q_full_idx lies outside the "
            f"full k-grid extent {n_full}: {qidx.tolist()}.")
    if int(operator.shape[0]) != int(qidx.size):
        raise ValueError(
            "GATE trs_fixed_q_projector: operator q extent does not match "
            f"q_full_idx ({int(operator.shape[0])} != {int(qidx.size)}).")
    if not bool(trs_allowed):
        return operator
    coords = np.stack(np.unravel_index(qidx, tuple(grid)), axis=1)
    fixed = np.all((2 * coords) % grid[None, :] == 0, axis=1)
    mask = jnp.asarray(fixed).reshape((qidx.size,) + (1,) * (operator.ndim - 1))
    projected = 0.5 * (operator + jnp.conj(operator))
    return jnp.where(mask, projected, operator)
