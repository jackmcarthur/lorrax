"""Passive shared-real-pole and general ordered-pole constructors.

The physical convention is W(s) = b (s - Lambda)^-1 b.H, s = z_Ry**2;
S_m = 2 M_(2m+1).  Thus b has units Ry**(3/2), Lambda Ry**2, and
M1/M3 are physical moments in Ry**3/Ry**5, never bare chi coefficients.

Sample matrices are consumed in bounded batches.  Only narrow direction,
output and derivative-action panels survive to the pencil assembly.  Dense
products and eigensolves of the passive constructor use resolved distrib_la
operations. The explicit general, non-Hermitian route instead gathers only
its capacity-bounded reduced frequency pencil for canonical host SciPy/LAPACK
reduction; microscopic port factors remain distributed.
"""

from __future__ import annotations

from contextlib import contextmanager
import time

import jax
from common import timing
from gw.shared_pole_capacity import ConstructorCapacity
from gw.shared_pole_directions import (_round_kernels, _sample_point, line_panel_states, port_extent,
                                       select_round_states, infinity_directions)
from gw.shared_pole_reduction import ORIENTATION_PAIR_REFUSAL


def collective_general_pencil(E, A, *, max_order, rank_tolerance=1e-10,
                             max_workspace_bytes=4*1024**3, fixed_spaces=None):
    """One canonical owner of a reduced host pencil's modal gauges.

    Every rank supplies the same gathered ``E,A[l,r]``; ONLY rank zero runs
    LAPACK. A bounded success/error record crosses first, then the exact
    uint64 bits of the five reduced complex arrays (poles, two coefficient
    maps and two held frequency spaces). No microscopic port or q stack
    crosses this handoff. Distributed faces must not mix independent host
    SVD/eigenvector gauges, even on homogeneous hardware.
    """
    from common.collectives import process_count
    if int(process_count()) == 1:
        return reduce_general_pencil(E, A, max_order=max_order, rank_tolerance=rank_tolerance,
            max_workspace_bytes=max_workspace_bytes, fixed_spaces=fixed_spaces)
    return collective_general_pencil_batch([(E,A)], max_order=max_order,
        rank_tolerance=rank_tolerance, max_workspace_bytes=max_workspace_bytes,
        fixed_spaces=[fixed_spaces])[0]


def general_pencil_batch_plan(shapes, *, max_order, max_workspace_bytes=4*1024**3,
                             processes=None, retained_host_bytes=0):
    """Price a prefix of at most P SMALL reduced pencils before host gather.

    Two disjoint phases are priced, not summed: one assigned LAPACK solve
    per process beside the other input pencils; then all returned models,
    one owner's original result and one bit-transfer buffer. The same
    numerical order bound applies at every width; capacity never cuts poles.
    ``retained_host_bytes`` accounts for the caller's existing held spaces.
    """
    from common.collectives import process_count
    world=int(process_count()) if processes is None else int(processes)
    if world<1 or int(max_order)<1 or int(retained_host_bytes)<0:
        raise ValueError('GATE shared_general_capacity: invalid batch capacity metadata')
    shapes=[tuple(map(int,s)) for s in shapes]
    if not shapes or any(len(s)!=2 or min(s)<1 for s in shapes):
        raise ValueError('GATE shared_general_capacity: nonempty reduced pencil shapes required')
    accepted=None
    for width in range(1,min(world,len(shapes))+1):
        batch=shapes[:width]
        inputs=[32*l*r for l,r in batch]
        reduce=[16*(16*l*r+12*min(l,r)**2) for l,r in batch]
        orders=[min(int(max_order),l,r) for l,r in batch]
        models=[16*k*(1+2*l+2*r) for (l,r),k in zip(batch,orders)]
        words=[16*max(l,r,1)*k for (l,r),k in zip(batch,orders)]
        control=8192*(world+1+width)
        reduction=int(retained_host_bytes)+sum(inputs)+max(x-y for x,y in zip(reduce,inputs))+control
        handoff=int(retained_host_bytes)+sum(inputs)+sum(models)+max(models)+max(words)+control
        price=max(reduction,handoff)
        if price<=int(max_workspace_bytes):
            accepted=dict(batch_size=width,processes=world,reduced_shapes=[list(s) for s in batch],
                reduction_peak_bytes=reduction,handoff_peak_bytes=handoff,
                host_workspace_bytes=price,host_workspace_bound_bytes=int(max_workspace_bytes),
                retained_host_bytes=int(retained_host_bytes),
                input_bytes=sum(inputs),model_bound_bytes=sum(models),
                full_microscopic_host_payload=False)
    if accepted is None:
        raise ValueError('GATE shared_general_capacity: even one canonical reduced pencil exceeds host bound')
    return accepted


def _share_general_pencil(model, error, *, owner_rank, price, shape):
    """Ordered canonical metadata/raw-bit handoff; no factorization or port gather."""
    import json
    import numpy as np
    from common.collectives import all_gather_processes, process_rank
    from jax.experimental.multihost_utils import broadcast_one_to_all
    rank=int(process_rank());started=time.perf_counter()
    status=None;arrays=None
    if rank==owner_rank:
        if error is not None:
            status=dict(error=f'{type(error).__name__}: {str(error)[:1000]}')
        else:
            arrays=[model[k] for k in ('poles_ry','output_map','input_map')]
            arrays += [model['spaces'][k] for k in ('right','left')]
            status=dict(error=None,shapes=[list(a.shape) for a in arrays],
                receipt=dict(model['receipt'],canonical_owner_rank=owner_rank,
                    canonical_host_workspace_bytes=price['host_workspace_bytes'],
                    canonical_batch_size=price['batch_size'],reduced_pencil_shape=list(shape)))
    payload=b'' if rank!=owner_rank else json.dumps(status,separators=(',',':')).encode('ascii')
    lengths = np.asarray(all_gather_processes(np.asarray(len(payload), np.int64))).reshape(-1)
    count = int(lengths[owner_rank])
    # Status is bounded metadata, never eigenvector payload in a control RPC.
    if not 0 < count <= 8192:
        raise ValueError("GATE shared_general_owner: invalid canonical status extent")
    record = np.zeros(count, np.uint8)
    if rank == owner_rank:
        record[:] = np.frombuffer(payload, np.uint8)
    status = json.loads(np.asarray(all_gather_processes(record))[owner_rank].tobytes())
    if status["error"] is not None:
        raise ValueError("GATE shared_general_owner: "+status["error"])
    result = []
    for i, shape in enumerate(status["shapes"]):
        array = (np.ascontiguousarray(arrays[i], dtype=np.complex128) if rank == owner_rank
                 else np.zeros(shape, np.complex128))
        # uint8 broadcast promotes under x64; uint64 preserves IEEE bits,
        # including tiny signs/complex phases and the zero-length model.
        bits = array.view(np.uint64)
        if bits.size:
            bits = np.asarray(broadcast_one_to_all(bits, is_source=rank == owner_rank))
        result.append(np.ascontiguousarray(bits).view(np.complex128).reshape(shape))
    wall=np.asarray(all_gather_processes(np.asarray(time.perf_counter()-started,np.float64))).reshape(-1)
    status['receipt']['canonical_handoff_seconds']=float(wall[owner_rank])
    return dict(zip(("poles_ry", "output_map", "input_map"), result[:3]),
                spaces=dict(right=result[3], left=result[4]), receipt=status["receipt"])


def collective_general_pencil_batch(pencils, *, max_order, rank_tolerance=1e-10,
                                   max_workspace_bytes=4*1024**3, fixed_spaces=None,
                                   retained_host_bytes=0):
    """Reduce at most P gathered frequency pencils concurrently, one owner each.

    The DRIVER gathers this admitted SMALL prefix on ALL ranks before entry.
    Each rank runs only its assigned pure-host LAPACK solve, BEFORE any
    ordered per-parent handoff. Every model/space/pole is then canonicalized
    from that parent's unique owner, with consistent all-rank failures.
    No JAX gather or full-microscopic matrix enters a rank-only branch.
    """
    from common.collectives import process_count, process_rank
    world,rank=int(process_count()),int(process_rank())
    pencils=list(pencils)
    plan=general_pencil_batch_plan([e.shape for e,a in pencils],max_order=max_order,
        max_workspace_bytes=max_workspace_bytes,processes=world,retained_host_bytes=retained_host_bytes)
    if len(pencils)>world or plan['batch_size']!=len(pencils):
        raise ValueError('GATE shared_general_capacity: caller must gather only the metadata-admitted batch prefix')
    fixed=[None]*len(pencils) if fixed_spaces is None else list(fixed_spaces)
    if len(fixed)!=len(pencils):
        raise ValueError('GATE shared_general_basis: one held space per batch parent required')
    model=None;error=None
    if rank<len(pencils):
        try:
            if world>1 and not jax.config.x64_enabled:
                raise ValueError('GATE shared_general_precision: canonical complex128 handoff requires JAX x64')
            e,a=pencils[rank]
            model=reduce_general_pencil(e,a,max_order=max_order,rank_tolerance=rank_tolerance,
                max_workspace_bytes=max_workspace_bytes,fixed_spaces=fixed[rank])
        except Exception as exc:
            error=exc
    if world==1:
        if error is not None:
            raise error
        return [model]
    results=[]
    for owner in range(len(pencils)):
        results.append(_share_general_pencil(model if rank==owner else None,
            error if rank==owner else None,owner_rank=owner,price=plan,shape=pencils[owner][0].shape))
    return results


def reduce_general_pencil(E, A, *, max_order, rank_tolerance=1e-10,
                          max_workspace_bytes=4 * 1024 ** 3, fixed_spaces=None):
    """Bounded host reduction of ONE general tangential frequency pencil.

    Only ``E,A[l,r]`` (complex128) cross this seam, never full microscopic
    response samples, q stacks, or port factors. A two-sided SVD of the
    descriptor E chooses numerical frequency spaces for this strictly proper
    finite-pole Loewner bank. A-visible directions unresolved in E are not
    promoted into nearly infinite modal states. This is numerical rank
    resolution, not a stable-pole projection: resolved growing poles remain.
    Generalized LAPACK eigenvectors produce unchanged complex poles and
    independent input/output coefficient maps. No pole or residue is clipped.
    ``max_order`` bounds realization states, not the full spatial port count.

    Returns a plain mapping with ``poles_ry[K]``, ``output_map[r,K]`` and
    ``input_map[l,K]``. Native GEMMs form full-port factors as
    ``left=C@output_map`` and ``right=B.H@input_map``, so
    ``Wc(z)=left@diag(1/(z-poles_ry))@right.H``. Workspace is conservatively
    priced before any factorization. Infinite/defective poles refuse.
    """
    import numpy as np
    import scipy.linalg as la
    started=time.perf_counter();svd_seconds=0.;eigen_started=None
    E, A = np.asarray(E, complex), np.asarray(A, complex)
    if (E.ndim != 2 or A.shape != E.shape or min(E.shape) < 1
            or not np.isfinite(E).all() or not np.isfinite(A).all()
            or int(max_order) < 1 or not 0 < rank_tolerance < 1):
        raise ValueError("GATE shared_general_pencil: finite compatible pencils required")
    nl, nr = E.shape
    # Conservative inherited bound for descriptor factors/workspace, the
    # projected descriptor and both eigenspaces. This is admission, not a
    # claim that LAPACK's hidden workspace is zero.
    price = 16 * (16 * nl * nr + 12 * min(nl, nr) ** 2)
    if price > int(max_workspace_bytes):
        raise ValueError(f"GATE shared_general_capacity: host pencil workspace {price} "
                         f"exceeds {int(max_workspace_bytes)} bytes")
    scale = max(float(np.linalg.norm(E)), float(np.linalg.norm(A)))
    if scale == 0:
        return dict(poles_ry=np.empty(0, complex), output_map=np.zeros((nr, 0), complex),
                    input_map=np.zeros((nl, 0), complex),
                    spaces=dict(right=np.zeros((nr, 0), complex), left=np.zeros((nl, 0), complex)),
                    receipt=dict(order=0, workspace_bytes=price, rank_tolerance=rank_tolerance,
                        svd_seconds=0.,eigen_seconds=0.,host_reduce_seconds=time.perf_counter()-started))
    if fixed_spaces is None:
        svd_started=time.perf_counter()
        y, singular, xh = la.svd(E / scale, full_matrices=False, check_finite=False)
        n = min(int(max_order), int(np.count_nonzero(singular > rank_tolerance * singular[0])))
        if not n:
            raise ValueError("GATE shared_general_rank: nonzero response has no resolved frequency span")
        y, x = y[:, :n], xh[:n].conj().T
        svd_seconds=time.perf_counter()-svd_started
    else:
        x, y = (np.asarray(fixed_spaces[key], complex) for key in ("right", "left"))
        n = x.shape[1]
        if (x.shape != (nr, n) or y.shape != (nl, n) or not n
                or n>min(int(max_order),nl,nr)):
            raise ValueError("GATE shared_general_basis: fixed spaces differ from current pencil")
    eigen_started=time.perf_counter()
    er, ar = y.conj().T @ E @ x, y.conj().T @ A @ x
    poles, wl, vr = la.eig(ar, er, left=True, right=True, check_finite=False)
    if not np.isfinite(poles).all():
        raise ValueError("GATE shared_general_poles: descriptor has an unresolved infinite pole")
    metric = wl.conj().T @ er @ vr
    diagonal = np.diag(metric)
    norm = np.linalg.norm(er) * np.linalg.norm(wl, axis=0) * np.linalg.norm(vr, axis=0)
    if np.any(np.abs(diagonal) <= 128 * np.finfo(float).eps * norm):
        raise ValueError("GATE shared_general_poles: defective or unresolved eigenvector normalization")
    # Generalized left/right eigenvectors are biorthogonal for distinct poles.
    # In a degenerate eigenspace use its full metric, never per-column division.
    inverse = la.solve(metric, np.eye(n, dtype=complex), check_finite=False)
    output = x @ vr
    inputs = y @ wl @ inverse.conj().T
    residual = np.linalg.norm(ar @ vr - (er @ vr) * poles[None]) / max(
        np.linalg.norm(ar @ vr), np.finfo(float).tiny)
    offdiag = metric - np.diag(diagonal)
    # Mixing distinct eigenvalues would change the transfer; only roundoff
    # off-diagonal metric is acceptable outside exactly degenerate blocks.
    commutator = inverse @ np.diag(poles) - np.diag(poles) @ inverse
    commutator_error = np.linalg.norm(commutator) / max(
        np.linalg.norm(inverse) * np.max(np.abs(poles)), np.finfo(float).tiny)
    if residual > 1e-9 or commutator_error > 1e-9:
        raise ValueError("GATE shared_general_eigenvectors: unresolved general modal reconstruction")
    order = np.lexsort((poles.imag, poles.real))
    return dict(poles_ry=poles[order], output_map=output[:, order], input_map=inputs[:, order],
                spaces=dict(right=x, left=y),
                receipt=dict(order=n, workspace_bytes=price, rank_tolerance=rank_tolerance,
                             eigen_residual=float(residual), metric_offdiagonal=float(np.linalg.norm(offdiag)),
                             modal_commutator_error=float(commutator_error),
                             max_growth_ry=float(np.max(poles.imag)),
                             svd_seconds=svd_seconds,eigen_seconds=time.perf_counter()-eigen_started,
                             host_reduce_seconds=time.perf_counter()-started,
                             numerical_spaces="two-sided-descriptor-svd",
                             construction="general-two-sided-tangential-loewner"))


def constructor_route(meta, config, recipe, *, mesh_xy, ledger, upstream, ordered,
                      odd_moments, nq):
    """Resolve local or whole-mesh parent execution for one map's bank.

    The single admission the constructor applies before its first bank read,
    also consulted by the map owner before the bank exists (bank residence).
    Returns ``(execution, receipt, column_extent, selection_faces,
    moment_fields)``; ``upstream`` names the accepted live reservations.
    The selection holds the dense fitted samples (W, dW/ds), the moments and
    every line sample's stored panels (``selection_face_count``).
    """
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_execution import constructor_execution, line_panel_count, selection_face_count
    from file_io.shared_pole_store import line_panel_geometry

    column_extent = port_extent(mesh_xy)
    moment_fields = ("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3")
    families, states = line_panel_geometry(meta, ordered=ordered)
    faces = selection_face_count(recipe, n=int(meta.n_rmu_padded), logical_n=int(meta.n_rmu),
        states=states, rows=int(meta.n_rmu_padded), dense_fields=2, moment_fields=len(moment_fields),
        column_extent=column_extent)
    execution, receipt = constructor_execution(
        meta, linalg_resolution({"linalg": config.backend.linalg}), recipe,
        mesh=mesh_xy, ledger=ledger, upstream=upstream, ordered=ordered,
        odd_moments=odd_moments, selection_faces=faces,
        sample_batch=len(recipe["fit_ids"]) - line_panel_count(recipe), parent_count=int(nq),
        column_extent=column_extent,
        ritz_budget=recipe.get("pole_budget") if ordered else None)
    return execution, receipt, column_extent, faces, moment_fields


def construct_shared_poles(bank, moments, meta, config, *, mesh_xy, output, residence=None):
    """Construct and write a current-state, bounded-batch real-pole model.

    Parameters
    ----------
    bank : mapping
        Authenticated resource descriptor: ``path``, ``identity``, ``tables``,
        ``coulomb`` (response-owner authenticated Coulomb resource). Dense
        workspace is queried from the resolved service plans. A producer
        certificate is carried as ``rule_receipt``. No dense bank is passed
        as a jit argument. The recipe is read only from meta.
    moments : mapping
        ``path`` to committed physical M1/M3 in the same scratch schema.
    meta : Meta
        Current packed centroid basis, scalar representation and the once-
        resolved ``shared_pole_recipe`` with current state identities.
    config : LorraxConfig
        Cached dense policy is read once, before any per-q plans.
    mesh_xy : Mesh
        Supplied named x/y mesh, never reconstructed by this driver.
    output : path-like
        New immutable compact model file, written by the store owner.
    residence : callable, optional
        The map owner's model residence rule, ``residence(nq, width)`` ->
        ``(ResidentSectorModel or None, receipt)``, asked once the model's
        column width is known and before it is written. A resident target
        replaces ``output`` and its stage stays reserved for the caller; the
        receipt is returned as ``model_residence``.

    Returns
    -------
    dict
        Versioned construction receipt with all gate rows, per-q diagnostics,
        capacity prices and the store header/digest. Structural failures raise
        before the affected q is written; a partial file is never finalized.
    """
    walls = {}

    @contextmanager
    def phase(name):
        """One named constructor stage; its wall adds to ``walls[name]`` (the per-map summary)."""
        started = time.monotonic()
        if getattr(meta, "shared_pole_capacity", None) is not None:
            meta.shared_pole_capacity.at_phase()
        with timing.section("spole." + name):
            yield
        walls[name] = walls.get(name, 0.0) + time.monotonic() - started

    with phase("entry"):
        import numpy as np
        from jax.sharding import NamedSharding, PartitionSpec as P
        from file_io.shared_pole_store import (
            charge_representation, validate_shared_pole_bank,
            RoundReader, read_round_samples, write_shared_pole_model,
        )
        from gw.gw_config import linalg_resolution
        from common.staged_reshard import face_to_batch_reshard
        from gw.shared_pole_local import (batch_to_face, canonical_factors, check_round, face_rows,
                                          own_extent_receipts, reduce_round, round_tables,
                                          recipe_panel_widths, recipe_infinity_width, pad_states,
                                          carrier_history)
        from gw.shared_pole_capacity import round_padding_output_bytes
        from gw.shared_pole_recipe import (
            build_construction_row, charge4_gates, construction_receipt,
            shared_real_pole_gates_v1_r3b as gates,
            shared_real_pole_gates_ordered_v1,
        )
        from common.units import RYD_TO_EV
        from gw.w_isdf import response_coulomb_powers

    with phase("setup"):
        recipe = meta.shared_pole_recipe
        # Time-reversal-broken scalar states take the ordered particle-hole route.
        ordered = not bool(bank["tables"]["sym"].trs_allowed)
        # Every charge carrier contracts to one mu x mu CC operator.
        if not charge_representation(meta):
            raise ValueError("GATE shared_pole_representation: expected an authenticated charge carrier")
        charge4 = recipe.get("charge_operator") == "four-component-spin-traced-v1"
        if (int(meta.nspinor) == 4) != charge4:
            raise ValueError("GATE shared_pole_representation: charge4 recipe and carrier disagree")
        if charge4:
            gates = charge4_gates(ordered)
        elif ordered:
            gates = shared_real_pole_gates_ordered_v1
        from gw.shared_pole_recipe import table_hash
        if recipe["gate_hash"] != table_hash(gates):
            raise ValueError("GATE shared_pole_representation: recipe gate table differs from charge carrier")
        resolution = linalg_resolution({"linalg": config.backend.linalg})
        identity = bank["identity"]
        ledger = meta.shared_pole_capacity
        upstream = ledger.live_stages
        n = int(meta.n_rmu_padded)
        # Every device byte this construction admits is priced here; the driver
        # below keeps only `budget.retained_panels` and `budget.batch_width`
        # current as it moves from selection to reduction to the model checks.
        header = validate_shared_pole_bank(bank["path"], expected_identity=identity,
                                           mesh_xy=mesh_xy, require_complete=True)
        moment_header = validate_shared_pole_bank(moments["path"], expected_identity=identity,
                                                  mesh_xy=mesh_xy)
        # The stored plan must bind the current physical points and role census.
        stored_recipe = header["recipe"]
        for name in ("recipe_hash", "gate_hash", "fit_ids", "held_ids", "role", "role_codes",
                     "distinct_id", "held", "support_pair", "census"):
            current = recipe[name]
            previous = stored_recipe[name]
            if isinstance(current, np.ndarray):
                equal = np.array_equal(current, previous)
            else:
                equal = current == previous
            if not equal:
                raise ValueError(f"GATE shared_pole_bank_state: got: stale {name}; want: current resolved recipe; why: no frozen SC inputs")
        for sample_id in (*recipe["fit_ids"], *recipe["held_ids"]):
            if _sample_point(recipe, sample_id) != _sample_point(stored_recipe, sample_id):
                raise ValueError("GATE shared_pole_bank_state: got: changed z; want: current physical sample point; why: recipe hashes alone do not bind resolved points")
        if bool(header.get("ordered", False)) != ordered:
            raise ValueError(f"GATE shared_pole_representation: got: bank ordered={bool(header.get('ordered', False))} with trs_allowed={not ordered}; want: an ordered bank exactly when time reversal is broken (both orientations, -q as its own parent); why: particle-hole pairing across parents")
        # Odd z-moments M0/M2 certify the ordered infinity block; a finite-state
        # ordered bank builds without it and records the moments NOT_MEASURED.
        odd_moments = ordered and bool(header.get("odd_moments", False))
        logical_n = int(meta.n_rmu)
        from gw.shared_pole_execution import constructor_side_upper_bound
        execution, execution_receipt, column_extent, selection_faces, moment_fields = constructor_route(
            meta, config, recipe, mesh_xy=mesh_xy, ledger=ledger, upstream=upstream,
            ordered=ordered, odd_moments=odd_moments,
            nq=int(header['bank_shape']['nq']))
        # Line supports off the imaginary axis arrive as their stored states;
        # the dense fitted samples (imaginary axis) are selected per round.
        line_lo, line_hi = (int(v) for v in header["line_panels"]["sample_span"])
        dense_fit = [int(i) for i in recipe["fit_ids"] if not line_lo <= int(i) < line_hi]
        budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh_xy,
                                     ledger=ledger, upstream=upstream,
                                     execution=execution)
        budget.ritz_budget = recipe.get("pole_budget") if ordered and execution == 'local' else None
        # A face constructor keeps the Coulomb eigensolve on the same complete
        # mesh even when the deck's default dense policy is local.
        coulomb_config = ({'linalg': 'distributed'}
                          if execution == 'face' else config)
        conservative_side = constructor_side_upper_bound(
            recipe, ordered=ordered, odd_moments=odd_moments,
            logical_n=logical_n, column_extent=column_extent)
        # A face plan waits for its room (face_eigh_room, below).
        eig = budget.eigenplan(n) if execution != 'face' else None
        receipts, factors, store_poles, store_counts, placed = {}, [], [], [], []
        receipt_entry_start = 0
        held_ids = [int(i) for i in recipe["held_ids"]]
        if not held_ids:
            raise ValueError("GATE shared_pole_held: got: no held samples; want: at least one held support; why: the model checks compare W and dW/ds there")
        held_lo, held_hi = min(held_ids), max(held_ids) + 1
        nq = int(header["bank_shape"]["nq"])
        # A local round runs one parent per rank from its sample read to its
        # sorted model. A face round runs a budget-sized batch of parents over
        # all ranks. Both take fixed-width rounds from the one schedule
        # (``parent_rounds``); a short last round repeats its last parent.
        face_batch = 1
        if execution == 'face':
            from gw.shared_pole_capacity import check_eigh_room, face_eigh_room
            from gw.shared_pole_execution import face_batch_width, face_ritz_carrier, face_round_compiled_bytes
            keep_budget = recipe.get("pole_budget")
            # The conservative round's infinity carrier and its block in the pencil side
            # (constructor_side_upper_bound), for the width's compiled admission.
            infinity_cap = column_extent(max(1, int(recipe["infinity_width"])))
            infinity_block = (2 * infinity_cap if odd_moments else 0) if ordered else infinity_cap
            sizing = dict(rows=n, side=conservative_side,
                          carrier=face_ritz_carrier(mesh_xy, keep_budget) if ordered else None)
            face_batch, execution_receipt['face_batch'] = face_batch_width(
                meta, resolution, mesh=mesh_xy, ledger=ledger, upstream=upstream,
                side=conservative_side, nq=nq,
                held=execution_receipt['retained_output_upper_bound_bytes_per_rank'],
                selection=dict(sample_batch=len(dense_fit), selection_faces=selection_faces),
                compiled=lambda width, room: face_round_compiled_bytes(
                    mesh_xy, width, rows=n, finite=conservative_side - infinity_block, infinity_width=infinity_cap,
                    infinities=1 + len(moment_fields), ordered=ordered, odd_moments=odd_moments,
                    keep_budget=keep_budget, carrier=sizing['carrier'],
                    eigh_backend=resolution.eigh_backend, room=room))
            # Every face round's reduction is priced at this size.
            budget.program_bytes = execution_receipt['face_batch']['program_bytes_per_rank']
            reduction_room = execution_receipt['face_batch'].pop('eigh_room')
            # distrib_la decides each eigh stack against the room beside the
            # admitted batch and the retained factors: whole matrices per rank
            # where they fit, else the whole mesh.
            retained_bound = execution_receipt['retained_output_upper_bound_bytes_per_rank']
            # The selection row reserves its eigh stacks as native workspace: handed back to them.
            selected = execution_receipt['face_batch']['selection']
            budget.face_room = face_eigh_room(selected, retained_bound - int(selected['workspace_bytes_per_rank']),
                                              stage="shared-pole constructor selection")
            execution_receipt['face_eigh_room_bytes_per_rank'] = dict(selection=int(budget.face_room))
            eig = budget.eigenplan(n)
        from gw.shared_pole_local import parent_rounds
        # One fixed width per route; a short last round repeats its last real parent.
        rounds = parent_rounds(header['n_q_irr'], mesh_xy.size if execution == 'local' else face_batch)
        batch_spec = P(("x", "y"))
        read_spec = batch_spec if execution == 'local' else None
        kernels = _round_kernels(mesh_xy, 'batch' if execution == 'local' else 'face')
        to_face, to_batch = batch_to_face(mesh_xy), face_to_batch_reshard(mesh_xy)
    # One read handle per bank for every round (RoundReader): the bank and
    # moments files are opened, and their datasets opened, once.
    with RoundReader(bank["path"], mesh_xy=mesh_xy) as bank_rounds, \
            RoundReader(moments["path"], mesh_xy=mesh_xy) as moment_rounds:
        for ids, real, slots in rounds:
            with phase("batch_admission"):
                budget.batch_width = len(ids)
                budget.retained_panels = tuple(factors)
                budget.plan(
                    conservative_side, phase="selection",
                    sample_batch=len(dense_fit), selection_faces=selection_faces,
                    runs_in="spole.direction_selection")
                budget.live(())
            with phase("scratch_read"):
                with moment_rounds.phase() as moment_io:
                    exact, _ = read_round_samples(moment_io, meta=meta, header=moment_header,
                                                  q_ids=ids, partition_spec=read_spec,
                                                  fields=moment_fields)
            with phase("infinity_selection"):
                width = min(logical_n, max(1, int(recipe["infinity_width"])))
                qi, round_infinity_values = infinity_directions(
                    kernels, exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"],
                    real_rows=real if execution == 'local' else None)
                infinity = (qi, *(kernels.apply(exact[name], qi)
                                  for name in moment_fields))
                del exact
            with phase("sample_batch_read"):
                budget.live(infinity)
                # The round's dense fitted samples and line panels in one
                # collective read (SlabIO.read_multi) on the rounds' one handle.
                with bank_rounds.phase() as bank_io:
                    samples, line = read_round_samples(
                        bank_io, meta=meta, header=header, q_ids=ids, partition_spec=read_spec,
                        sample_ids=dense_fit, fields=("Wc", "dWc_ds"),
                        line_family="charge", line_samples=range(line_lo, line_hi))
                # The store admits this complete bounded scratch batch before
                # allocation. Charge it while directions/actions are selected;
                # release it before admitting the dense pencil.
            with phase("direction_selection"):
                # Synthetic local slots select nothing, as a dense selection's do.
                line_states = {sid: line_panel_states(panels, np.where(np.arange(len(counts)) < real, counts, 0),
                                                      recipe, sid=sid, ordered=ordered, mesh_xy=mesh_xy)
                               for sid, (panels, counts) in line.items()}
                round_states, round_counts, round_roles = select_round_states(
                    samples, recipe, sample_ids=dense_fit, real=real, mesh_xy=mesh_xy, eigh_plan=eig,
                    column_extent=column_extent, logical_n=logical_n, ordered=ordered,
                    line_states=line_states)
                del samples, line, line_states, qi
            with phase("reduction_admission"):
                infinity_counts = [int(v.shape[-1]) for v in round_infinity_values]
                budget.retained_panels = tuple(factors)
                budget.batch_width = len(ids)
                # Every state panel is padded to its recipe carrier, so the round
                # program's inputs have one shape; the pencil extent is grow-only
                # over this model's rounds and SC maps (round_tables): discovered
                # in map 0, held from map 1. The table is host metadata, known
                # before any panel is allocated.
                widths = recipe_panel_widths(round_roles[0], round_states, recipe,
                                             column_extent=column_extent, logical_n=logical_n)
                infinity_width = recipe_infinity_width(infinity, recipe, column_extent=column_extent,
                                                       logical_n=logical_n)
                tables = round_tables(
                    round_counts, widths, [st[0] for st in round_states], infinity_counts, infinity_width,
                    column_extent=column_extent, ordered=ordered, odd_moments=odd_moments,
                    key=("scalar", logical_n, ordered, odd_moments), history=carrier_history(meta))
                side = int(tables["active"].shape[-1])
                # Resolve before either reduction program is traced; the ledger
                # warns when the route price is over the budget.
                local_eigh = budget.eigenplan(side)
                # The reservation holds the round's operands; a face round's program is priced by
                # its compiled figure where it runs (face_reduce_round). The eigh rooms below still
                # subtract the admitted (conservative) program.
                admitted = budget.program_bytes
                try:
                    budget.program_bytes = 0          # the round program: its compiled figure
                    reduction_row = budget.plan(side, phase="reduction", padding_output_bytes_per_rank=round_padding_output_bytes(
                        round_states, infinity, widths, infinity_width), runs_in="spole.gram_reduction")
                finally:
                    budget.program_bytes = admitted
                round_states, infinity = pad_states(round_states, widths, infinity, infinity_width)
            with phase("gram_reduction"):
                if execution == 'face':
                    from gw.shared_pole_execution import face_reduce_round
                    # The kept span on the budget's carrier, as the local round
                    # solves it on its Ritz carrier: the Schur and final eigh run
                    # at about 2 x budget instead of the pencil side.
                    # The room the batch width compiled this program in: a room beside the
                    # admitted row would hold the program's own stacks twice (room 0).
                    room = reduction_room
                    round_model, round_signed, vectors, round_diagnostics = face_reduce_round(
                        round_states, infinity, tables, mesh=mesh_xy,
                        budget=budget, ordered=ordered, odd_moments=odd_moments,
                        keep_budget=keep_budget, eigh_backend=resolution.eigh_backend,
                        admit=False, room=room, carrier=sizing['carrier'],
                        beside=budget.ledger.price(reduction_row))
                else:
                    round_model, round_signed, vectors, round_diagnostics = reduce_round(
                        round_states, infinity, tables, real=real, mesh_xy=mesh_xy,
                        native_eigh=local_eigh.native_fn, ordered=ordered,
                        odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"),
                        beside=budget.ledger.price(reduction_row))
                qi = infinity[0]
                # One device-to-host read of the round's replicated diagnostics and vectors.
                from gw.shared_pole_local import host_tree
                (round_reduction, round_zero, round_retained, round_permutation), (poles, active) = host_tree(
                    (tuple(round_diagnostics), tuple(vectors)))
                # rocSOLVER zheevd can return finite eigenvalues with NaN eigenvectors and info 0
                # (distrib_la.local_eigh). A local even-route round that reports it is solved once
                # more by Jacobi on the same, still-live inputs (no allocation lies between the solve
                # and this read, so the peak is the round's own); every other round is untouched.
                finite = round_reduction.get("ritz_eigh_finite")
                if execution != 'face' and finite is not None and not all(bool(v) for v in finite[:real]):
                    from distrib_la import native_eigh_jacobi
                    bad = [ids[i] for i in range(real) if not bool(finite[i])]
                    import warnings
                    warnings.warn(f"shared-pole round: local eigh returned non-finite eigenpairs at q={bad}; "
                                  "re-solved the round by Jacobi", RuntimeWarning, stacklevel=1)
                    del round_model, round_signed, vectors, round_diagnostics
                    round_model, round_signed, vectors, round_diagnostics = reduce_round(
                        round_states, infinity, tables, real=real, mesh_xy=mesh_xy,
                        native_eigh=native_eigh_jacobi, ordered=ordered,
                        odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"))
                    (round_reduction, round_zero, round_retained, round_permutation), (poles, active) = host_tree(
                        (tuple(round_diagnostics), tuple(vectors)))
                del round_states, infinity
                budget.retained_panels = tuple(factors)
            with phase("gates"):
                import numpy as np
                # Host-side context for the zero-Ritz refusal: the round's own diagnostics.
                _sp_report = lambda slot: " ".join(
                    f"{k}={np.array2string(np.asarray(d[k][slot]), precision=3, max_line_width=400)}"
                    for d in (round_zero, round_reduction) for k in (
                        "dropped_factor_weight_fraction", "dropped_count", "factor_weight", "retained_rank",
                        "infinite_weight_fraction", "positive_count", "negative_count", "paired_rank",
                        "gram_min_relative", "paired_min_relative", "gram_condition") if k in d)
                for slot, q in enumerate(ids[:real]):
                    if ordered and not round_reduction["orientation_paired"][slot]:
                        raise ValueError(ORIENTATION_PAIR_REFUSAL + f" (q={q})")
                    for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                        if not round_reduction[name][slot]:
                            raise ValueError(
                                f"GATE shared_pole_{name}: got: failed at q={q}, "
                                f"Gram min/max={round_reduction['gram_min_relative'][slot]}, "
                                f"rounding floor/max={round_reduction['gram_floor_relative'][slot]}, "
                                f"paired Schur S min/max={round_reduction['paired_min_relative'][slot] if ordered else 'n/a'}, "
                                f"metric infinity norm={round_reduction['metric_initial_infinity_norm'][slot]}, "
                                f"inverse-root residual={round_reduction['metric_inverse_root_residual_relative'][slot]}; "
                                f"want: Gram min >= -floor (normalized_gram_validity; even route: the larger of it and gram_rounding_validity) "
                                "and valid diagonal/retained metric; why: no PSD repair")
                    if not round_reduction.get("ritz_eigh_finite", [True] * (slot + 1))[slot]:
                        raise ValueError(f"GATE shared_pole_ritz_eigh_finite: got: non-finite Ritz eigenpairs at q={q}; "
                                         "want: finite eigenvalues and eigenvectors of the reduced pencil; "
                                         "why: a silent eigensolver failure, not a physical zero-Ritz weight")
                    if not round_zero["zero_policy"][slot]:
                        raise ValueError(f"GATE shared_pole_zero_ritz: got: failed at q={q} [{_sp_report(slot)}]; want: finite positive response within dropped-weight budget; why: no pole clipping")
                    # The ordered identity is on the ORIGINAL infinity directions: exact only for the
                    # full Galerkin span, projection accuracy after the keep/retention cuts. It is
                    # reported beside the full_m1/full_m3 diagnostic bands, as the TRS route reports
                    # its original-direction defects; the retained Ritz algebra is the metric gate above.
                    if not ordered and not all(value[slot] <= gates["retained_subspace_moments"]["threshold"]
                                               for value in round_retained.values()):
                        raise ValueError(f"GATE shared_pole_retained_moments: got: failed at q={q}; want: projected latent moment identity <=1e-10; why: corrected Ritz algebra")
                reductions = own_extent_receipts(round_reduction, tables["own"][:real])
            with phase("coulomb"):
                budget.batch_width = len(ids)
                model_row = budget.plan(side, phase="model", sample_batch=len(held_ids))
                budget.live((*round_model, *round_signed, qi))
                # V^-1/2 of the round's parents: one owner call per contiguous run of ids, rows in slot order.
                runs = []
                for q in sorted(ids[:real]):
                    if runs and q == runs[-1][1]:
                        runs[-1][1] += 1
                    else:
                        runs.append([q, q + 1])
                parts, coulomb = [], {}
                for lo, hi in runs:
                    coulomb_sqrt, part, receipt = response_coulomb_powers(
                        meta, coulomb_config, mesh_xy=mesh_xy, bank_io=bank,
                        q_span=(lo, hi))
                    del coulomb_sqrt
                    parts.append(part)
                    # One parent's support rank and the resource hash; the resource
                    # itself (path, q order) is the bank's, stated once.
                    coulomb.update({q: dict(sha256=receipt["coulomb_identity"].get("sha256"),
                                            support_ranks=receipt["support_ranks"][q - lo:q - lo + 1])
                                    for q in range(lo, hi)})
                inverse_sqrt = face_rows(mesh_xy, tuple(sorted(ids[:real]).index(q) for q in ids))(*parts)
                if execution == 'local':
                    inverse_sqrt = to_batch(inverse_sqrt)
                del parts
                # Callee I/O admission needs the actual arrays that survive the
                # Coulomb call, not the earlier pre-call live set.
                budget.live((*round_model, *round_signed, qi, inverse_sqrt))
            with phase("sample_batch_read"):
                # Read here, not with the fitted samples: the held panels would
                # otherwise stay resident through the reduction.
                with bank_rounds.phase() as bank_io:
                    held, _ = read_round_samples(bank_io, meta=meta, header=header, q_ids=ids,
                                                 partition_spec=read_spec,
                                                 sample_span=(held_lo, held_hi), fields=("Wc", "dWc_ds"))
                pick = kernels.take(tuple(i - held_lo for i in held_ids))
                held = tuple(pick(held[name]) for name in ("Wc", "dWc_ds"))
                budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held))
            with phase("moment_read"):
                with moment_rounds.phase() as moment_io:
                    exact, _ = read_round_samples(moment_io, meta=meta, header=moment_header, q_ids=ids,
                                                  partition_spec=read_spec, fields=("M1", "M3"))
                budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held,
                             *exact.values()))
            with phase("passivity_held"):
                passive, held_errors, reciprocity, moment_defects = check_round(
                    round_model, round_signed, inverse_sqrt, held, (exact["M1"], exact["M3"]), qi,
                    real=real, nodes=[_sample_point(recipe, i) for i in held_ids], eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                    mesh_xy=mesh_xy, eigh_plan=eig if execution == 'face' else local_eigh, ordered=ordered,
                    # A face check decides its eighs beside the model phase and its own program.
                    room=lambda price: check_eigh_room(model_row, price))
                del inverse_sqrt, held, exact
            with phase("gates"):
                for slot, q in enumerate(ids[:real]):
                    if not passive["passivity"][slot]:
                        raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
                    if not ordered and not np.all(reciprocity["passed"][slot]):
                        raise ValueError(f"GATE shared_pole_model_reciprocity: got: { {k: v[slot].tolist() for k, v in reciprocity.items()} } at q={q}; want: model preserves transpose symmetry of symmetric held data; why: conjugate-port closure must survive reduction")
            with phase("receipts"):
                price = budget.plan(side)
                counts = active.sum(axis=-1, dtype=np.int64)
                for slot, q in enumerate(ids[:real]):
                    row_of = lambda tree: {key: value[slot:slot + 1] for key, value in tree.items()}
                    row, measurements = build_construction_row(
                        (None, poles[slot:slot + 1], active[slot:slot + 1]), counts[slot:slot + 1],
                        dict(reduction=reductions[slot], zero=row_of(round_zero), passive=row_of(passive),
                             retained=row_of(round_retained),
                             moment_defects={name: row_of(value) for name, value in moment_defects.items()},
                             held=[{"sample_id": sample_id, "Wc": float(held_errors[slot, 0, i]),
                                    "dWc_ds": float(held_errors[slot, 1, i])} for i, sample_id in enumerate(held_ids)],
                             # Preserve the receipt's sample-major W,dW rows, each with one parent.
                             reciprocity={key: value[slot].T.reshape(-1, 1).tolist()
                                          for key, value in reciprocity.items()},
                             permutation=round_permutation[slot:slot + 1]),
                        span=(q, q + 1), roles=round_roles[slot], price=price, coulomb=coulomb[q],
                        native_queries=budget.native_queries, identity=identity,
                        gates=gates, nspinor=int(meta.nspinor), logical_n=logical_n,
                        ordered=ordered, odd_moments=odd_moments,
                        detail=bool(config.debug.sigma_freq_debug_output))
                    receipt = construction_receipt(
                        measurements, capacity=ledger,
                        capacity_entry_start=receipt_entry_start, ordered=ordered,
                        charge4=charge4)
                    receipt.update(identity=identity, constructor=row)
                    receipts[q] = receipt
                receipt_entry_start = len(ledger.entries)
            with phase("export_prepare"):
                # The sorted model is an active prefix: keep the round's widest K, then restore to the face.
                width = column_extent(int(counts[:real].max()))
                factors.append(face_rows(mesh_xy, tuple(range(real)), width)(round_model[0]) if execution == 'face' else
                               face_rows(mesh_xy, tuple(range(real)), width)(to_face(round_model[0])))
                # The round's sorted poles stay on the device; the writer joins them there.
                store_poles.append(((vectors[0], vectors[1]), (int(real), int(width))))
                store_counts.append(counts[:real])
                placed += list(ids[:real])
                del round_model, round_signed, qi, reductions, vectors
                ledger.live_stages = upstream
                budget.retained_panels = tuple(factors)
    with phase("writer_stack"):
        if sorted(placed) != list(range(nq)):
            raise ValueError(f"GATE shared_pole_rounds: got: parents {sorted(placed)}; want: each of {nq} parents once; why: one canonical store")
        order = np.argsort(placed)
        width = max(block.shape[-1] for block in factors)
        public_b = canonical_factors(mesh_xy, tuple(int(i) for i in order))(*factors)
        from gw.shared_pole_local import store_pole_rows
        store_poles, store_device_counts = store_pole_rows(
            mesh_xy, tuple(b for _, b in store_poles), int(width), tuple(int(i) for i in order))(
            tuple(p for (p, _), _ in store_poles), tuple(a for (_, a), _ in store_poles))
        budget.batch_width = nq
        budget.retained_panels = ()
        budget.plan(width, phase="model")
        budget.live((public_b,))
        del factors
    with phase("writer"):
        target, model_residence = output, dict(residence="file")
        if residence is not None:
            resident, model_residence = residence(nq, width)
            if resident is not None:
                target = resident
                ledger.live_stages = (*ledger.live_stages, model_residence["stage"])
        store_header = write_shared_pole_model(
            target, public_b, store_poles,
            np.concatenate(store_counts)[order], q_span=(0, nq), meta=meta, tables=bank["tables"], recipe=recipe,
            receipts={"identity": identity, "q_receipts": [receipts[q] for q in range(nq)]}, ordered=ordered,
            device_counts=store_device_counts)
        ledger.live_stages = upstream
        del public_b
    with phase("return"):
        # Stage walls summed over the rounds (host clock; a stage that only
        # dispatches lends its device time to the next stage that reads back).
        seconds = {name: round(value, 3) for name, value in walls.items()}
        seconds["rounds"] = len(rounds)
        return {"q_receipts": [receipts[q] for q in range(nq)], "model_header": store_header,
                "model": target if model_residence.get("stage") else str(output),
                "model_residence": model_residence,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED", "seconds": seconds,
                "execution": dict(mode=execution, **execution_receipt)}
