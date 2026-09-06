"""The q-grid symmetry decisions, taken once and said out loud.

WHAT THIS MODULE IS.  Two functions, one per decision, and both are
announcing adapters over ``symmetry_maps``:

* :func:`resolve_qgrid_symmetry_tables` — does this deck's centroid set
  admit the IBZ q reduction at all?
* :func:`qgrid_trs_policy_for` — what is TIME REVERSAL allowed to do to
  the q axis of this deck?  The answer comes from the load-time density
  MEASUREMENT (``SymMaps.trs_allowed``), never from an assumption, and it
  arrives as one object the driver consumes rather than a branch the
  driver takes.

Every producer of the q-axis tables in ``gw/`` goes through here; nothing
else in the monorepo calls ``centroid_source_map_and_wrap``, composes q
with −q through Θ, or applies the fixed-q Θ projector, and
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
        required_rows=np.asarray(sym.sym_idx_q),
        context=context,
    )
    msg = res.announcement() if announce_fallback else None
    if msg is not None:
        # Rank-invariant fact (the centroid file is the same on every
        # rank), so scope="rank0"; keyed on the centroid hash so the
        # repeat resolves along the run are silent.
        announce_once(res.announce_key, msg, scope="rank0")
    return res


def qgrid_trs_policy_for(
    *,
    sym,
    irr_idx_q,
    sym_idx_q,
    kgrid,
    n_sym_spatial,
    context: str,
    announce: bool = True,
):
    """The q-axis time-reversal policy for this deck, announced once.

    THE ONLY DOOR.  ``symmetry_maps.qgrid_trs`` holds the policy (it holds
    the tables and the arithmetic); this adapter holds rank 0, the
    once-per-run memory, and — the whole point — the MEASURED verdict.

    ``trs_measured`` is read off ``SymMaps.trs_allowed``, which
    ``SymMaps.__init__`` takes from ``WfnLoader.trs_holds``, which
    ``density_symmetry_check`` obtained from the occupied two-component
    DFT subspaces before antiunitary unfolding.  No caller of this function
    passes a verdict of its own, and the policy constructor has no default
    for it, so there is no path by which a driver can assume time reversal.

    THE DEFECT THIS CLOSES.  ``v_q_g_flat``, ``screening`` and
    ``screening_bse`` each composed q with −q through Θ and projected the
    self-negative rows *unconditionally*.  On ferromagnetic CrI3
    (Perlmutter JID 57271494) q and −q are independent irreducible
    parents, so the composition refused — after the 685.96-GB ζ fit had
    completed.  Where the parents had coincided it would have silently
    replaced one independently solved row by the conjugate of the other.

    Returns
    -------
    symmetry_maps.QgridTrsPolicy
        ``.unfold_sym_idx`` is the row map to hand
        ``unfold_isdf_operator``; ``.project_fixed_q(op, q_full_idx)``
        returns ``(op, removed_rel)``; ``.measure_covariance(V_ibz, ...)``
        measures the point-group assumption the unfold makes.
    """
    from ffi import _services
    _services.ensure_on_path()
    from ffi.gate import announce_once
    from symmetry_maps import build_qgrid_trs_policy

    policy = build_qgrid_trs_policy(
        trs_measured=bool(sym.trs_allowed),
        irr_idx_q=irr_idx_q,
        sym_idx_q=sym_idx_q,
        q_irr_full_idx=sym.q_irr_full_idx,
        kgrid=tuple(kgrid),
        n_sym_spatial=int(n_sym_spatial),
        active_symmetry_rows=np.asarray(
            sym.active_symmetry_rows, dtype=np.int32),
        context=str(context),
    )
    if announce:
        # Rank-invariant (the verdict and the tables are the same on every
        # rank), so scope="rank0"; keyed on the verdict + grid + context so
        # repeat resolves along a self-consistency loop are silent while a
        # genuinely different channel still gets its own line.
        announce_once(policy.announce_key, policy.announcement(),
                      scope="rank0")
    return policy
