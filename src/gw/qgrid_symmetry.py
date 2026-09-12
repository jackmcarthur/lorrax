"""The q-grid symmetry decisions, taken once and said out loud.

WHAT THIS MODULE IS.  Two decisions, expressed by announcing adapters over
``symmetry_maps``:

* :func:`resolve_qgrid_symmetry_tables` — does this deck's centroid set
  admit the IBZ q reduction at all?
* :func:`qgrid_trs_policy_for` — what is TIME REVERSAL allowed to do to
  the q axis of this deck?  The answer comes from the load-time density
  MEASUREMENT (``SymMaps.trs_allowed``), never from an assumption, and it
  arrives as one object the driver consumes rather than a branch the
  driver takes. :func:`qgrid_trs_policy_from_shared_pole_store` adapts the
  same measured reference after the model store has authenticated it.

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


_SHARED_POLE_REALIZERS = {}


def shared_pole_operator_realizer(meta, header, *, q_full_idx, mesh_xy):
    r"""Bind the versioned physical realization of packed shared-pole tiles.

    Stored factors remain the raw latent Ritz model. Physical consumers use
    ``Pi_G sum_k b_k f_k b_k^dagger`` with the complete authenticated little
    group at each selected full-grid q. Bind this adapter OUTSIDE a JIT; its
    returned callable accepts only the current operator and its SAME-TIME
    transpose. Antiunitary rows act on residue endpoints and never conjugate
    the complex scalar frequency/time coefficients.

    ``q_full_idx`` can name irreducible parents or already-unfolded children.
    Tensor readers return packed endpoints; the basis owns the conversion of
    the authenticated canonical source/wrap tables to that order. Only host
    symmetry metadata is cached, never factors, poles, W, or head samples.

    Historical raw-model stores remain readable for analysis by the generic
    store reader, but cannot silently acquire this physical realization.
    Their recipe must identify this version before a physical consumer binds.
    Constructor retained-Ritz identities and existing held/moment/passivity
    receipts concern the raw latent model; they do not certify the projected
    operator's held error or its upper passivity bound relative to raw V.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import project_little_group_operator
    from .shared_pole_recipe import shared_real_pole_v2_r1

    expected = shared_real_pole_v2_r1["operator_realization"]
    if header.get("recipe", {}).get("operator_realization") != expected:
        raise ValueError(
            "GATE shared_pole_realization: missing or unsupported operator_realization; "
            f"expected {expected!r}, rebuild this model with the current recipe")
    if (header.get("representation") != "scalar-trs-even-s"
            or header.get("nspinor") != 1
            or header.get("q_order") != "canonical-full-flat"
            or not np.array_equal(header.get("q_shift"), np.zeros(3))):
        raise ValueError("GATE shared_pole_realization: unsupported scalar q-grid representation")
    qt, operations = header["qirr"], header["operations"]
    nsp = int(qt["n_sym_spatial"])
    canonical_rows = np.arange(2*nsp, dtype=np.int32)
    if (not np.array_equal(operations["rows"], canonical_rows)
            or not np.array_equal(operations["antiunitary"], canonical_rows >= nsp)
            or not operations.get("typing_source")):
        raise ValueError("GATE shared_pole_realization: unauthenticated canonical operation typing")
    grid, qids = np.asarray(header["grid"]), np.asarray(q_full_idx)
    if (grid.shape != (3,) or grid.dtype.kind not in "iu" or np.any(grid <= 0)
            or qids.ndim != 1 or not qids.size or qids.dtype.kind not in "iu"
            or np.any(qids < 0) or np.any(qids >= np.prod(grid))
            or np.unique(qids).size != qids.size):
        raise ValueError("GATE shared_pole_realization: invalid full-grid q selection")
    basis = meta.mu_basis
    if basis.mesh_xy != mesh_xy or int(header["n_mu_logical"]) != basis.n_logical:
        raise ValueError("GATE shared_pole_realization: model and packed basis carrier differ")
    kwargs = dict(
        q_full_idx=qids,
        q_irr_frac=np.stack(np.unravel_index(qids, tuple(grid)), axis=1)/grid[None, :],
        sym_mats_k=np.asarray(operations["rotation"]),
        sym_perm=basis.layout.axis.pack_permutations_host(
            np.asarray(qt["sym_perm"], np.int32), require_local=False),
        L_table=basis.layout.axis.pack_host(np.asarray(qt["L_table"], np.int32),
                                          axis=1, fill_value=0),
        active_symmetry_rows=np.asarray(operations["authorized_rows"]),
        active_mask=np.asarray(basis.active_mask),
    )
    # Freeze host metadata so a later caller mutation cannot change a retained
    # callable while its compiled specialization still has the old tables.
    kwargs = {name: np.array(value, copy=True) for name, value in kwargs.items()}
    for value in kwargs.values():
        value.flags.writeable = False
    grid = tuple(map(int, grid))
    key = (mesh_xy, nsp, grid, expected,
           tuple((name, value.shape, value.dtype.str, value.tobytes())
                 for name, value in kwargs.items()))
    realize = _SHARED_POLE_REALIZERS.get(key)
    if realize is None:
        def realize(operator, transposed_partner):
            return project_little_group_operator(
                operator, transposed_partner=transposed_partner,
                kgrid=grid, n_sym_spatial=nsp, mesh=mesh_xy, **kwargs)
        _SHARED_POLE_REALIZERS[key] = realize
    return realize


def symmetrise_shared_pole_tiles(arrays, *, meta, header, q_span, mesh_xy):
    r"""Reynolds-project stored parent tiles onto their own little groups.

    THE CONSISTENCY THIS CLOSES.  Every Sigma consumer already realizes the
    stored model through :func:`shared_pole_operator_realizer` -- the recipe
    names that realization (``operator_realization``) and
    ``gw/mpa/sigma.py`` and ``gw/shared_pole_head.py`` apply it.  The bank
    tiles the model is FIT to, and on which the Loewner pencil's passivity
    certificate is taken, were never projected, so the fit and its
    certificate lived in a space the consumer does not use.  The residual
    little-group defect of the FITTED ISDF basis then entered the Gram
    spectrum amplified by the pencil size: measured at 1.7e-08 relative on
    ``Wc``, on ``M1`` and on ``M3`` alike at q=0 on Si (the exact bare
    moments take no quadrature, which is what places the floor in the basis
    rather than in the response rule), against a q=0 amplification of
    ~1140 -- and that product is the order of the -1e-07 Gram gate.

    EVERY TILE THAT ENTERS ONE PENCIL IS PROJECTED TOGETHER.  The confluent
    blocks use ``dWc_ds`` while the off-diagonals use ``Wc`` and the infinity
    columns use ``M1``/``M3``, so a pencil assembled from a mixture of
    projected and unprojected tiles carries an inconsistency the pencil reads
    as non-passivity.  Measured on the map-11 rCROP trial that refused three
    times byte-identically at -2.12e-07: projecting ``Wc`` alone gives
    -4.92e-08, ``dWc_ds`` alone -2.12e-07 (inert), and the two together
    -8.40e-10 -- 250.9x, and 2.1x on an accepted map.

    ``arrays`` maps name -> ``[n_q, n, n]`` or ``[n_q, n_sample, n, n]`` with
    the parent axis first; ``q_span`` is the half-open parent range those
    tiles were read with, indexing ``header['q_irr_full_idx']``.  Returns a
    dict with the same keys, shapes and shardings.  The projection is an
    average of congruences of the authenticated little group, so it preserves
    positivity and the real poles; it moves an approximate model by the size
    of its own symmetry defect, which is why the caller gates Sigma.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    lo, hi = int(q_span[0]), int(q_span[1])
    # Shape refusals first: they are free and they are the ones a caller trips.
    for name, value in arrays.items():
        if value.ndim not in (3, 4) or value.shape[-1] != value.shape[-2]:
            raise ValueError(
                f"GATE shared_pole_symmetrisation: {name} has shape "
                f"{tuple(value.shape)}; want a parent axis, an optional sample "
                "axis and a square centroid pair")
        if int(value.shape[0]) != hi - lo:
            raise ValueError(
                f"GATE shared_pole_symmetrisation: {name} carries "
                f"{value.shape[0]} parent tiles against q_span {(lo, hi)}")
    qids = np.asarray(header["q_irr_full_idx"], dtype=np.int64)[lo:hi]
    realize = shared_pole_operator_realizer(
        meta, header, q_full_idx=qids, mesh_xy=mesh_xy)

    def project(tile):
        # The same-time transposed partner is what the antiunitary rows act
        # on; it is never inferred from the values being projected.
        return realize(tile, jnp.swapaxes(tile, -2, -1))[0]

    out = {}
    for name, value in arrays.items():
        if value.ndim == 3:
            out[name] = project(value)
        else:
            out[name] = jax.lax.with_sharding_constraint(
                jnp.stack([project(value[:, i])
                           for i in range(int(value.shape[1]))], axis=1),
                NamedSharding(mesh_xy, P(None, None, "x", "y")))
    return out


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


def qgrid_trs_policy_from_shared_pole_store(header, *, announce=True):
    """Recover the measured reference policy from a validated model store.

    The shared-pole store validator binds ``scalar-trs-even-s`` to the
    measured scalar-TRS reference and authenticates its operation tables.
    Its consumer has no live SymMaps object; adapt that sealed metadata at
    the same door as live references, never infer symmetry from fitted b.
    """
    from types import SimpleNamespace

    qt = header["qirr"]
    reference = SimpleNamespace(
        trs_allowed=header["representation"] == "scalar-trs-even-s",
        q_irr_full_idx=np.asarray(header["q_irr_full_idx"]),
        active_symmetry_rows=np.asarray(header["operations"]["authorized_rows"]))
    return qgrid_trs_policy_for(
        sym=reference, irr_idx_q=np.asarray(qt["irr_idx_q"]),
        sym_idx_q=np.asarray(qt["sym_idx_q"]), kgrid=tuple(header["grid"]),
        n_sym_spatial=int(qt["n_sym_spatial"]), context="shared-pole Sigma",
        announce=announce)


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
