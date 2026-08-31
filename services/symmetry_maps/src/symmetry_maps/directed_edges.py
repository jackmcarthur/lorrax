"""Directed k-edge tables and the canonical band-matrix symmetry action.

The table builder is pure NumPy metadata.  It maps compact directed edges
onto a full, possibly shifted and anisotropic, uniform k-grid; the matrix
action applies the one unitary/antiunitary band-basis algebra.  No
wavefunction or object-specific policy lives here.

Expected table cost is ``O(n_k*n_sym + n_k*n_target*n_source_step)`` host
integer work.  The action never gathers a band matrix: JAX operands retain
the caller's sharding and endpoint sewing products lower on that layout.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


_EDGE_DOC = "docs/services/symmetry_maps.md#directed-band-matrix-edges"
_Q_STENCIL_DOC = "docs/services/symmetry_maps.md#q-stencil-orbits"


def _refuse(rule, got, want, fix):
    raise ValueError(
        f"directed_edge_orbit_table [{rule}]: got {got}; want {want}; "
        f"fix: {fix}; see {_EDGE_DOC}.")


def _action_refuse(rule, got, want, fix):
    raise ValueError(
        f"apply_band_matrix_symmetry [{rule}]: got {got}; want {want}; "
        f"fix: {fix}; see {_EDGE_DOC}.")


def _q_refuse(rule, got, want, fix):
    raise ValueError(
        f"q_stencil_orbit_table [{rule}]: got {got}; want {want}; "
        f"fix: {fix}; see {_Q_STENCIL_DOC}."
    )


def _integer_array(name, value, shape_tail=None):
    raw = np.asarray(value)
    if shape_tail and (raw.ndim < len(shape_tail)
                       or tuple(raw.shape[-len(shape_tail):]) != shape_tail):
        _refuse(
            "PT-EDGE-SHAPE", f"{name}.shape={raw.shape}",
            f"{name}.shape ending in {shape_tail}",
            "pass the documented pure-array table operands",
        )
    rounded = np.rint(raw)
    if not np.allclose(raw, rounded, rtol=0.0, atol=1.0e-12):
        _refuse(
            "PT-EDGE-INTEGER", f"noninteger {name}",
            f"integer-valued {name}",
            "express reciprocal operations and mesh steps in integer coordinates",
        )
    return rounded.astype(np.int64)


def _validate_steps(name, steps):
    out = _integer_array(name, steps, (3,))
    if out.ndim != 2 or out.shape[0] == 0:
        _refuse(
            "PT-EDGE-SHAPE", f"{name}.shape={out.shape}",
            f"{name}.shape=(n_{name}, 3) with n_{name}>0",
            "pass one row per directed elementary mesh step",
        )
    norms = np.sum(np.abs(out), axis=1)
    if np.any(norms != 1):
        bad = int(np.where(norms != 1)[0][0])
        _refuse(
            "PT-EDGE-NONPERMUTATION",
            f"{name}[{bad}]={out[bad].tolist()}",
            "signed elementary steps with exactly one +/-1 component",
            "store the full symmetry orbit of elementary mesh directions",
        )
    keys = [tuple(int(x) for x in row) for row in out]
    if len(set(keys)) != len(keys):
        _refuse(
            "PT-EDGE-CONFLICT", f"duplicate rows in {name}: {keys}",
            "one stored value per directed source step",
            "remove the duplicate direction so an image has one source",
        )
    return out


def _signed_closure(steps):
    rows = []
    seen = set()
    for row in np.concatenate([steps, -steps], axis=0):
        key = tuple(int(x) for x in row)
        if key not in seen:
            rows.append(row)
            seen.add(key)
    return np.asarray(rows, dtype=np.int64)


def _grid_rows(kgrid):
    axes = [np.arange(int(n), dtype=np.int64) for n in kgrid]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _flat_ids(rows, kgrid):
    rows = np.asarray(rows, dtype=np.int64) % kgrid[None, :]
    return ((rows[:, 0] * kgrid[1] + rows[:, 1]) * kgrid[2]
            + rows[:, 2]).astype(np.int64)


def _shifted_grid_permutations(kgrid, kgrid_shift, sym_mats_k):
    """Return ``perm[s, k]`` after exact shifted-grid validation."""
    full = _grid_rows(kgrid)
    coords = (full + kgrid_shift[None, :]) / kgrid[None, :]
    perms = np.empty((sym_mats_k.shape[0], full.shape[0]), dtype=np.int32)
    for isym, sym in enumerate(sym_mats_k):
        raw = (coords @ sym.T) * kgrid[None, :] - kgrid_shift[None, :]
        rounded = np.rint(raw)
        residual = float(np.max(np.abs(raw - rounded)))
        if residual > 1.0e-10:
            _refuse(
                "PT-EDGE-NONPERMUTATION",
                f"sym_mats_k[{isym}] misses shifted grid by {residual:.3e}",
                "every symmetry row to permute the shifted full k-grid exactly",
                "use a symmetry-compatible kgrid/kgrid_shift or disable this reduction",
            )
        ids = _flat_ids(rounded.astype(np.int64), kgrid)
        if np.unique(ids).size != full.shape[0]:
            _refuse(
                "PT-EDGE-NONPERMUTATION",
                f"sym_mats_k[{isym}] has {np.unique(ids).size} images for "
                f"{full.shape[0]} grid points",
                "a bijective full-grid permutation",
                "remove the singular/incommensurate symmetry row",
            )
        perms[isym] = ids.astype(np.int32)
    return perms


def _mapped_step(step, sym, kgrid, *, isym):
    fractional = np.asarray(step, dtype=np.float64) / kgrid
    raw = (sym @ fractional) * kgrid
    rounded = np.rint(raw)
    residual = float(np.max(np.abs(raw - rounded)))
    if residual > 1.0e-10:
        _refuse(
            "PT-EDGE-NONPERMUTATION",
            f"sym_mats_k[{isym}] maps step {np.asarray(step).tolist()} to "
            f"noninteger mesh displacement {raw.tolist()}",
            "a signed elementary mesh-step permutation",
            "store the full commensurate step orbit or disable this symmetry reduction",
        )
    return rounded.astype(np.int64)


def q_stencil_orbit_table(
    *,
    kgrid,
    sym_mats_k,
    irr_idx_q,
    sym_idx_q,
    seed_steps,
    n_sym_spatial,
    allow_trs,
    active_symmetry_rows=None,
):
    """Close and symmetry-reduce a compact finite-q stencil.

    This is q-only metadata. It deliberately does not act on a transition
    matrix at fixed k: a finite-q polarizability is first summed over the full
    k grid at each returned ``source_step``. The resulting scalar, vector or
    tensor response may then be unfolded to ``target_steps`` with
    :func:`apply_band_matrix_symmetry`; Cartesian wings pass
    ``SymMaps.R_cart_forward[target_sym_idx]`` as ``component_mix``.

    ``seed_steps`` need not already be closed under the crystal point group.
    The service adds their exact integer images and records which seeds
    generated each target. Points that alias modulo the mesh are refused:
    first/second-neighbour data cannot be distinguished on such a grid.
    """
    kg = _integer_array("kgrid", kgrid, (3,)).reshape(-1)
    if kg.shape != (3,) or np.any(kg <= 0):
        _q_refuse("WAV-Q-SHAPE", f"kgrid={kg.tolist()}",
                  "three positive grid extents", "pass the WFN kgrid")
    syms = _integer_array("sym_mats_k", sym_mats_k, (3, 3))
    nss = int(n_sym_spatial)
    if syms.ndim != 3 or nss <= 0 or syms.shape[0] not in (nss, 2 * nss):
        _q_refuse(
            "WAV-Q-SYM",
            f"sym_mats_k.shape={syms.shape}, n_sym_spatial={nss}",
            "the SymMaps spatial table, optionally followed by its TRS half",
            "pass SymMaps.sym_mats_k and wfn.ntran")
    if syms.shape[0] == 2 * nss and not np.array_equal(
            syms[nss:], -syms[:nss]):
        _q_refuse("WAV-Q-TRS", "antiunitary rows are not -spatial rows",
                  "the SymMaps [S, -S] layout",
                  "do not reorder SymMaps.sym_mats_k")
    if active_symmetry_rows is None:
        search_ids = np.arange(
            syms.shape[0] if bool(allow_trs) else nss, dtype=np.int32)
    else:
        search_ids = _integer_array(
            "active_symmetry_rows", active_symmetry_rows).reshape(-1)
        if (search_ids.size == 0
                or np.unique(search_ids).size != search_ids.size
                or np.any(search_ids < 0)
                or np.any(search_ids >= syms.shape[0])):
            _q_refuse(
                "WAV-Q-ACTIVE-SYM", search_ids.tolist(),
                f"a nonempty unique subset of [0,{syms.shape[0]})",
                "pass SymMaps.active_symmetry_rows")
    search = syms[search_ids]
    seeds = _integer_array("seed_steps", seed_steps, (3,))
    if seeds.ndim != 2 or seeds.shape[0] == 0:
        _q_refuse("WAV-Q-SHAPE", f"seed_steps.shape={seeds.shape}",
                  "seed_steps.shape=(n_seed,3), n_seed>0",
                  "pass the enabled first/second-neighbour directions")
    if np.any(np.all(seeds == 0, axis=1)):
        _q_refuse("WAV-Q-ZERO", "seed_steps contains q=0",
                  "nonzero neighbours only",
                  "handle Gamma through the analytic head")
    seed_keys = [tuple(int(x) for x in row) for row in seeds]
    if len(set(seed_keys)) != len(seed_keys):
        _q_refuse("WAV-Q-DUPLICATE", f"duplicate seed_steps={seed_keys}",
                  "one row per seed direction",
                  "deduplicate the caller's stencil")

    target_for_key = {}
    seed_membership = {}
    for iseed, seed in enumerate(seeds):
        for local_sym, sym in enumerate(search):
            isym = int(search_ids[local_sym])
            mapped = _mapped_step(seed, sym, kg, isym=isym)
            key = tuple(int(x) for x in mapped)
            target_for_key.setdefault(key, mapped)
            seed_membership.setdefault(key, set()).add(int(iseed))
    target_keys = sorted(target_for_key)
    targets = np.asarray(
        [target_for_key[key] for key in target_keys], dtype=np.int64)
    target_ids = _flat_ids(targets, kg)
    if np.unique(target_ids).size != target_ids.size:
        collisions = {}
        for row, iq in zip(targets, target_ids):
            collisions.setdefault(int(iq), []).append(row.tolist())
        collisions = {
            key: rows for key, rows in collisions.items() if len(rows) > 1}
        _q_refuse(
            "WAV-Q-ALIAS", collisions,
            "distinct stencil points modulo the k mesh",
            "use a denser kgrid (second neighbours require at least five "
            "points per active axis)")

    nk = int(np.prod(kg))
    irr = _integer_array("irr_idx_q", irr_idx_q).reshape(-1)
    sidx = _integer_array("sym_idx_q", sym_idx_q).reshape(-1)
    if irr.shape != (nk,) or sidx.shape != (nk,):
        _q_refuse("WAV-Q-TABLE",
                  f"irr_idx_q={irr.shape}, sym_idx_q={sidx.shape}",
                  f"both q tables to have shape ({nk},)",
                  "pass the q tables from the same SymMaps instance")
    allowed_sidx = np.isin(sidx, search_ids)
    if not np.all(allowed_sidx):
        bad = int(np.where(~allowed_sidx)[0][0])
        _q_refuse("WAV-Q-TRS", f"sym_idx_q[{bad}]={int(sidx[bad])}",
                  f"an allowed symmetry row in {search_ids.tolist()}",
                  "pass active rows from the same SymMaps instance")

    groups = irr[target_ids]
    unique_groups = []
    for group in groups:
        if int(group) not in unique_groups:
            unique_groups.append(int(group))

    # Use the exact global q-IBZ representatives already chosen by SymMaps.
    # Because target_steps is a complete orbit closure, every canonical row
    # must be in this stencil; a missing one means the tables disagree.
    source_ids = np.asarray(
        [int(np.where(irr == group)[0][0]) for group in unique_groups],
        dtype=np.int32)
    target_row_for_id = {
        int(full_id): row for row, full_id in enumerate(target_ids)}
    missing = [int(full_id) for full_id in source_ids
               if int(full_id) not in target_row_for_id]
    if missing:
        _q_refuse(
            "WAV-Q-INCOMPLETE", f"canonical q ids {missing} are outside closure",
            "every selected SymMaps q-IBZ representative in target_steps",
            "pass the full symmetry closure generated from this SymMaps table")
    source_target_row = np.asarray(
        [target_row_for_id[int(full_id)] for full_id in source_ids],
        dtype=np.int32)
    source_steps = targets[source_target_row]
    group_to_source = {group: row for row, group in enumerate(unique_groups)}
    target_source_row = np.asarray(
        [group_to_source[int(group)] for group in groups], dtype=np.int32)
    target_sym = sidx[target_ids].astype(np.int32)

    for itarget, target in enumerate(targets):
        source = source_steps[target_source_row[itarget]]
        isym = int(target_sym[itarget])
        mapped = _mapped_step(source, search[isym], kg, isym=isym)
        if not np.array_equal(np.mod(mapped, kg), np.mod(target, kg)):
            _q_refuse(
                "WAV-Q-INCOMPLETE",
                f"source={source.tolist()}, target={target.tolist()}, "
                f"sym_idx_q={isym}",
                "the canonical SymMaps q action to map source onto target",
                "rebuild the q tables and stencil from the same WFN symmetry")

    membership = np.zeros((targets.shape[0], seeds.shape[0]), dtype=np.bool_)
    for itarget, key in enumerate(target_keys):
        membership[itarget, list(seed_membership[key])] = True
    return {
        "source_steps": source_steps.astype(np.int32),
        "source_full_ids": source_ids.astype(np.int32),
        "source_target_row": source_target_row,
        "target_steps": targets.astype(np.int32),
        "target_full_ids": target_ids.astype(np.int32),
        "source_irr_idx": np.asarray(unique_groups, dtype=np.int32),
        "target_source_row": target_source_row,
        "target_sym_idx": target_sym,
        "target_antiunitary": (target_sym >= nss),
        "target_seed_mask": membership,
        "kgrid": kg.astype(np.int32),
    }


def directed_edge_orbit_table(
    *,
    kgrid,
    kgrid_shift,
    sym_mats_k,
    irr_idx_k,
    sym_idx_k,
    source_full_ids,
    source_steps,
    n_sym_spatial,
    target_steps=None,
):
    """Build the full-grid lookup for compact directed band-matrix links.

    A stored ``L[kbar, j]`` belongs to the edge
    ``source_full_ids[kbar] -> source_full_ids[kbar] + source_steps[j]``.
    For every requested full-grid edge this first uses the frozen point-map
    row ``(irr_idx_k[start], sym_idx_k[start])``.  If its direction is
    negative, it uses the endpoint's frozen point-map row and the adjoint of
    that stored forward link.  A direction is never guessed or rounded.

    Parameters
    ----------
    kgrid : array-like, shape (3,)
        Uniform k-grid extents along reduced reciprocal axes.
    kgrid_shift : array-like, shape (3,)
        Shift in mesh-index units: point ``n`` is
        ``(n + kgrid_shift) / kgrid``.
    sym_mats_k : array-like, shape (n_sym, 3, 3)
        Reciprocal-coordinate operations.  TRS rows use ``[S, -S]``.
    irr_idx_k, sym_idx_k : array-like, shape (n_k_full,)
        Canonical full-grid point tables from :class:`SymMaps`.
    source_full_ids : array-like, shape (n_source,)
        Exact full-grid row of each stored source k.  For raw WFN order this
        is ``sym.kirr_fullids``, not a star's first occurrence.
    source_steps : array-like, shape (n_source_step, 3)
        Stored elementary directions.  Link values have shape
        ``(n_source, n_source_step, ..., n_band_x, n_band_y)``.
    n_sym_spatial : int
        Number of unitary rows at the front of ``sym_mats_k``.
    target_steps : array-like, shape (n_target_step, 3), optional
        Requested directions.  Default: signed closure of ``source_steps``.

    Returns
    -------
    table : dict[str, numpy.ndarray]
        Dense fields ``source_row``, ``source_direction``, ``sym_idx``,
        ``reverse``, ``antiunitary``, ``stored_start_full``,
        ``stored_end_full``, ``source_start_full``, ``source_end_full``,
        ``source_step``, ``target_start_full`` and ``target_end_full`` are
        indexed by ``(k_full, target_direction)``.  ``source_steps`` and
        ``target_steps`` record the constant direction vocabulary.

        ``source_start/end`` are oriented after ``reverse``;
        ``stored_start/end`` name the endpoints whose sewing matrices go to
        :func:`apply_band_matrix_symmetry`.

    Notes
    -----
    The table is exact integer metadata.  Endpoint sewing has value-level,
    not bit-exact, parity because distributed GEMMs may differ at roundoff.
    The identity-sewing action has exact parity with :func:`star_broadcast`.
    """
    kg = _integer_array("kgrid", kgrid, (3,)).reshape(-1)
    if kg.shape != (3,) or np.any(kg <= 0):
        _refuse(
            "PT-EDGE-SHAPE", f"kgrid={kg.tolist()}",
            "three positive grid extents",
            "pass the WFN kgrid without folding or sorting axes",
        )
    shift = np.asarray(kgrid_shift, dtype=np.float64)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        _refuse(
            "PT-EDGE-SHAPE", f"kgrid_shift.shape={shift.shape}",
            "three finite mesh-index shifts",
            "pass the WFN shift in mesh-index units",
        )
    syms = _integer_array("sym_mats_k", sym_mats_k, (3, 3))
    if syms.ndim != 3 or syms.shape[0] == 0:
        _refuse(
            "PT-EDGE-SHAPE", f"sym_mats_k.shape={syms.shape}",
            "sym_mats_k.shape=(n_sym, 3, 3)",
            "pass the canonical reciprocal-operation table",
        )
    nss = int(n_sym_spatial)
    if nss <= 0 or syms.shape[0] not in (nss, 2 * nss):
        _refuse(
            "PT-EDGE-TRS-LAYOUT",
            f"n_sym={syms.shape[0]}, n_sym_spatial={nss}",
            "n_sym == n_sym_spatial or 2*n_sym_spatial",
            "use SymMaps.sym_mats_k and its spatial operation count",
        )
    if syms.shape[0] == 2 * nss and not np.array_equal(
            syms[nss:], -syms[:nss]):
        _refuse(
            "PT-EDGE-TRS-LAYOUT", "antiunitary rows are not -spatial rows",
            "the service [S, -S] TRS augmentation convention",
            "build sym_mats_k through SymMaps rather than reordering it",
        )

    source_steps_i = _validate_steps("source_steps", source_steps)
    target_steps_i = (_signed_closure(source_steps_i) if target_steps is None
                      else _validate_steps("target_steps", target_steps))
    signed_source = {tuple(int(x) for x in row)
                     for row in _signed_closure(source_steps_i)}
    mapped_steps = np.empty(
        (syms.shape[0], source_steps_i.shape[0], 3), dtype=np.int64)
    for isym, sym in enumerate(syms):
        for idir, step in enumerate(source_steps_i):
            mapped = _mapped_step(step, sym, kg, isym=isym)
            mapped_steps[isym, idir] = mapped
            if tuple(int(x) for x in mapped) not in signed_source:
                _refuse(
                    "PT-EDGE-NONPERMUTATION",
                    f"sym_mats_k[{isym}] maps {step.tolist()} to "
                    f"{mapped.tolist()}",
                    f"a signed permutation of source_steps={source_steps_i.tolist()}",
                    "use an elementary-step basis closed under this point group "
                    "or disable link symmetry reduction",
                )

    perms = _shifted_grid_permutations(kg, shift, syms)
    nk = int(np.prod(kg))
    irr = _integer_array("irr_idx_k", irr_idx_k).reshape(-1)
    sidx = _integer_array("sym_idx_k", sym_idx_k).reshape(-1)
    src_ids = _integer_array("source_full_ids", source_full_ids).reshape(-1)
    if irr.shape != (nk,) or sidx.shape != (nk,):
        _refuse(
            "PT-EDGE-SHAPE",
            f"irr_idx_k.shape={irr.shape}, sym_idx_k.shape={sidx.shape}, nk={nk}",
            "both point-map arrays to have shape (prod(kgrid),)",
            "pass the canonical full-grid SymMaps arrays",
        )
    if src_ids.size == 0 or np.any(src_ids < 0) or np.any(src_ids >= nk):
        _refuse(
            "PT-EDGE-SOURCE",
            f"source_full_ids={src_ids.tolist()} for nk={nk}",
            "one in-range exact full-grid id per compact source row",
            "pass SymMaps.kirr_fullids from the same WFN grid",
        )
    if np.unique(src_ids).size != src_ids.size:
        _refuse(
            "PT-EDGE-CONFLICT", f"source_full_ids={src_ids.tolist()}",
            "distinct exact source representatives",
            "remove duplicate compact source rows",
        )
    if np.any(irr < 0) or np.any(irr >= src_ids.size):
        bad = int(np.where((irr < 0) | (irr >= src_ids.size))[0][0])
        _refuse(
            "PT-EDGE-SOURCE", f"irr_idx_k[{bad}]={int(irr[bad])}",
            f"source row in [0, {src_ids.size})",
            "pair irr_idx_k with source_full_ids from the same SymMaps/WFN",
        )
    if np.any(sidx < 0) or np.any(sidx >= syms.shape[0]):
        bad = int(np.where((sidx < 0) | (sidx >= syms.shape[0]))[0][0])
        _refuse(
            "PT-EDGE-SOURCE", f"sym_idx_k[{bad}]={int(sidx[bad])}",
            f"symmetry row in [0, {syms.shape[0]})",
            "pair sym_idx_k with the same SymMaps.sym_mats_k",
        )
    canonical_images = perms[sidx, src_ids[irr]]
    if not np.array_equal(canonical_images, np.arange(nk)):
        bad = int(np.where(canonical_images != np.arange(nk))[0][0])
        _refuse(
            "PT-EDGE-SOURCE",
            f"source_full_ids[irr_idx_k[{bad}]] maps to "
            f"{int(canonical_images[bad])}, not {bad}",
            "each frozen (source row, symmetry row) to map onto its full-grid point",
            "use kirr_fullids, not the first full-BZ star occurrences",
        )

    n_target = target_steps_i.shape[0]
    shape = (nk, n_target)
    source_row = np.full(shape, -1, dtype=np.int32)
    source_direction = np.full(shape, -1, dtype=np.int32)
    sym_idx = np.full(shape, -1, dtype=np.int32)
    reverse = np.zeros(shape, dtype=np.bool_)
    stored_start = np.full(shape, -1, dtype=np.int32)
    stored_end = np.full(shape, -1, dtype=np.int32)
    target_start = np.broadcast_to(
        np.arange(nk, dtype=np.int32)[:, None], shape).copy()
    target_end = np.empty(shape, dtype=np.int32)
    source_step = np.empty(shape + (3,), dtype=np.int32)

    grid_rows = _grid_rows(kg)
    shifted_by_source = np.asarray([
        _flat_ids(grid_rows + step[None, :], kg).astype(np.int32)
        for step in source_steps_i
    ])
    shifted_by_target = np.asarray([
        _flat_ids(grid_rows + step[None, :], kg).astype(np.int32)
        for step in target_steps_i
    ])

    for target in range(nk):
        for idir, target_step in enumerate(target_steps_i):
            end = int(shifted_by_target[idir, target])
            target_end[target, idir] = end

            src_row = int(irr[target])
            isym = int(sidx[target])
            src_start = int(src_ids[src_row])
            forward = []
            for jdir in range(source_steps_i.shape[0]):
                mapped = mapped_steps[isym, jdir]
                src_end = int(shifted_by_source[jdir, src_start])
                if (np.array_equal(mapped, target_step)
                        and int(perms[isym, src_end]) == end):
                    forward.append((src_row, jdir, isym, src_start, src_end))
            if len(forward) > 1:
                _refuse(
                    "PT-EDGE-CONFLICT",
                    f"{len(forward)} start-canonical images for target edge "
                    f"({target}->{end}, step={target_step.tolist()})",
                    "exactly one source direction per target image",
                    "remove duplicate/conflicting source directions",
                )
            if forward:
                chosen = forward[0]
                use_reverse = False
            else:
                src_row = int(irr[end])
                isym = int(sidx[end])
                src_start = int(src_ids[src_row])
                backward = []
                for jdir in range(source_steps_i.shape[0]):
                    mapped = mapped_steps[isym, jdir]
                    src_end = int(shifted_by_source[jdir, src_start])
                    if (np.array_equal(mapped, -target_step)
                            and int(perms[isym, src_end]) == target):
                        backward.append((src_row, jdir, isym, src_start, src_end))
                if len(backward) > 1:
                    _refuse(
                        "PT-EDGE-CONFLICT",
                        f"{len(backward)} endpoint-canonical images for target edge "
                        f"({target}->{end}, step={target_step.tolist()})",
                        "exactly one adjoint source direction per target image",
                        "remove duplicate/conflicting source directions",
                    )
                if not backward:
                    _refuse(
                        "PT-EDGE-INCOMPLETE",
                        f"no image for target edge ({target}->{end}, "
                        f"step={target_step.tolist()})",
                        "a start-canonical forward image or endpoint-canonical adjoint image",
                        "include the missing compact source step/representative",
                    )
                chosen = backward[0]
                use_reverse = True

            src_row, jdir, isym, src_start, src_end = chosen
            source_row[target, idir] = src_row
            source_direction[target, idir] = jdir
            sym_idx[target, idir] = isym
            reverse[target, idir] = use_reverse
            stored_start[target, idir] = src_start
            stored_end[target, idir] = src_end
            source_step[target, idir] = (
                -source_steps_i[jdir] if use_reverse else source_steps_i[jdir])

    source_start = np.where(reverse, stored_end, stored_start).astype(np.int32)
    source_end = np.where(reverse, stored_start, stored_end).astype(np.int32)
    return {
        "source_row": source_row,
        "source_direction": source_direction,
        "sym_idx": sym_idx,
        "reverse": reverse,
        "antiunitary": (sym_idx >= nss),
        "stored_start_full": stored_start,
        "stored_end_full": stored_end,
        "source_start_full": source_start,
        "source_end_full": source_end,
        "source_step": source_step,
        "target_start_full": target_start,
        "target_end_full": target_end,
        "source_steps": source_steps_i.astype(np.int32),
        "target_steps": target_steps_i.astype(np.int32),
        "kgrid": kg.astype(np.int32),
        "kgrid_shift": shift.copy(),
    }


def _is_jax_value(value):
    return isinstance(value, (jax.Array, jax.core.Tracer))


def _where_flag(flag, if_true, if_false, xp):
    cond = xp.asarray(flag, dtype=bool)
    while cond.ndim < if_true.ndim:
        cond = xp.expand_dims(cond, axis=-1)
    return xp.where(cond, if_true, if_false)


def _adjoint(matrix, xp):
    return xp.swapaxes(xp.conj(matrix), -1, -2)


def apply_band_matrix_symmetry(
    matrix,
    *,
    antiunitary=False,
    reverse=False,
    sewing_start=None,
    sewing_end=None,
    component_mix=None,
    component_axis=-3,
):
    """Apply the generic endpoint-sewn symmetry action to band matrices.

    For a directed edge matrix ``M(k0, k1)``, the unitary action is
    ``M(gk0, gk1) = B_g(k0) M(k0, k1) B_g(k1)^dagger``.
    ``antiunitary`` conjugates ``M`` (never transposes a non-Hermitian link).
    ``reverse`` takes its adjoint and swaps endpoint sewings, so
    ``sewing_start/end`` always refer to the STORED link endpoints.

    Flags may be scalars or arrays broadcast over leading batch axes.  If
    supplied, ``component_mix[..., out, in]`` mixes one explicit non-band
    component axis after the band-space action.

    Parameters
    ----------
    matrix : numpy.ndarray or jax.Array
        Shape ``(..., n_band, n_band)`` when sewing, reversal, or component
        mixing is used.  With only ``antiunitary`` it may have arbitrary
        trailing shape; that identity-sewing path is shared by
        :func:`star_broadcast`.
    antiunitary, reverse : bool or array-like
        Flags broadcast over leading batch axes.
    sewing_start, sewing_end : array-like, optional
        Endpoint matrices broadcastable to ``(..., n_band, n_band)``.
    component_mix : array-like, optional
        Shape ``(n_out, n_in)`` or ``(..., n_out, n_in)``.
    component_axis : int, default=-3
        Component axis; the final two axes are band axes.

    Returns
    -------
    numpy.ndarray or jax.Array
        Transformed matrix, without a host/device conversion.
    """
    jax_operands = (matrix, antiunitary, reverse, sewing_start, sewing_end,
                    component_mix)
    xp = jnp if any(_is_jax_value(value) for value in jax_operands) else np
    out = xp.asarray(matrix)
    reverse_is_array = _is_jax_value(reverse) or np.ndim(reverse) > 0
    reverse_any = (True if _is_jax_value(reverse)
                   else bool(np.any(np.asarray(reverse))))
    if reverse_is_array and reverse_any:
        if out.ndim < 2:
            _action_refuse(
                "PT-ACTION-SHAPE", f"matrix.ndim={out.ndim}",
                "two trailing band axes for batched reverse",
                "pass (..., n_band, n_band) matrices or disable reverse",
            )
        reversed_out = _adjoint(out, xp)
        if reversed_out.shape != out.shape:
            _action_refuse(
                "PT-ACTION-SHAPE", f"band axes={out.shape[-2:]}",
                "equal global band extents for mixed reverse flags",
                "use one square manifold or split forward/reverse batches",
            )
        out = _where_flag(reverse, reversed_out, out, xp)
    elif reverse_any:
        if out.ndim < 2:
            _action_refuse(
                "PT-ACTION-SHAPE", f"matrix.ndim={out.ndim}",
                "two trailing band axes for reverse",
                "pass (..., n_left, n_right) matrices or disable reverse",
            )
        out = _adjoint(out, xp)

    anti_is_array = _is_jax_value(antiunitary) or np.ndim(antiunitary) > 0
    anti_any = (True if _is_jax_value(antiunitary)
                else bool(np.any(np.asarray(antiunitary))))
    if anti_is_array and anti_any:
        out = _where_flag(antiunitary, xp.conj(out), out, xp)
    elif anti_any:
        out = xp.conj(out)

    if (sewing_start is None) != (sewing_end is None):
        _action_refuse(
            "PT-ACTION-SEWING", "only one endpoint sewing",
            "both sewing_start and sewing_end, or neither",
            "supply the sewing at each stored-link endpoint",
        )
    if sewing_start is not None:
        if out.ndim < 2:
            _action_refuse(
                "PT-ACTION-SHAPE", f"matrix.ndim={out.ndim}",
                "two trailing band axes for endpoint sewing",
                "pass (..., n_band, n_band) matrices",
            )
        start = xp.asarray(sewing_start)
        end = xp.asarray(sewing_end)
        if reverse_is_array and reverse_any:
            if start.shape != end.shape:
                _action_refuse(
                    "PT-ACTION-SEWING",
                    f"sewing shapes {start.shape} and {end.shape}",
                    "equal endpoint-sewing shapes for mixed reverse flags",
                    "pad/distribute both endpoint sewings identically",
                )
            old_start, old_end = start, end
            start = _where_flag(reverse, old_end, old_start, xp)
            end = _where_flag(reverse, old_start, old_end, xp)
        elif reverse_any:
            start, end = end, start
        out = start @ out @ _adjoint(end, xp)

    if component_mix is not None:
        if out.ndim < 3:
            _action_refuse(
                "PT-ACTION-COMPONENT", f"matrix.ndim={out.ndim}",
                "a non-band component axis before two band axes",
                "add the explicit vector/covector component axis",
            )
        axis = int(component_axis)
        if axis < 0:
            axis += out.ndim
        if axis < 0 or axis >= out.ndim - 2:
            _action_refuse(
                "PT-ACTION-COMPONENT", f"component_axis={component_axis}",
                "an axis before the two trailing band axes",
                "pass the explicit component axis",
            )
        moved = xp.moveaxis(out, axis, -3)
        mix = xp.asarray(component_mix)
        if mix.ndim < 2 or mix.shape[-1] != moved.shape[-3]:
            _action_refuse(
                "PT-ACTION-COMPONENT",
                f"component_mix.shape={mix.shape}, component extent={moved.shape[-3]}",
                "component_mix[..., out, in] with matching input extent",
                "supply the chosen vector/covector representation matrix",
            )
        if mix.ndim == 2:
            moved = xp.einsum("oi,...iab->...oab", mix, moved)
        else:
            moved = xp.einsum("...oi,...iab->...oab", mix, moved)
        out = xp.moveaxis(moved, -3, axis)
    return out
