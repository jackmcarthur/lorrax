# Note (fractional translations): the public wfn-unfold helpers
# ``get_gvecs_kfull`` / ``get_cnk_fullzone[_batch]`` lived here until
# P5; that whole pipeline now lives inside
# ``file_io.wfn_loader.WfnLoader`` (eager + phdf5 backends, both
# applying U_spinor + τ phase + TR conjugation in one place).  This
# module retains the sym-table construction (kpoint_map, R_grid,
# unfolded_kpts, …) and the kfull-symmap / q-IBZ helpers used by the
# GW driver.
from functools import partial
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from lxkit import device_put_process_local
from ._compat import deprecated_alias
from .directed_edges import apply_band_matrix_symmetry
from ._shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def kgrid_shift_map(nkx, nky, nkz, q_off):
    """C-order fold + umklapp G for the on-grid shift ``k -> k + q_off``.

    The ONE place the ``k + q`` integer arithmetic lives (finite-momentum
    remap of on-grid conduction/valence tensors — the ``jnp.roll`` in the BSE
    W_q / exciton-Q loaders derives its offsets from here).  Pure numpy; no new
    class.  ``q_off`` is an integer grid-step vector (may be negative).

    Per-element (full-BZ k = (ix, iy, iz), C-order flat
    ``k = ix·nky·nkz + iy·nkz + iz``; ``q_off = (qx, qy, qz)``):

        jx = ix + qx ;  kpx = jx mod nkx ;  Gx = jx // nkx      (floor div)
        (same for y, z)
        kpq_index[k] = kpx·nky·nkz + kpy·nkz + kpz
        G_umk[k]     = (Gx, Gy, Gz)          integer reciprocal-lattice wrap

    So ``arr[kpq_index]`` gathers the value at ``k + q_off`` into slot ``k``,
    i.e. it equals ``jnp.roll(arr_reshaped, shift=(-qx, -qy, -qz),
    axis=(0, 1, 2)).reshape(nk, ...)`` on the C-order (nkx, nky, nkz) k-axis
    (verified by the identity ``kpq_index == roll`` in the finite-q gate).
    ``G_umk`` is the wrap count on each axis; for ``0 <= q < nk`` it is in
    ``{0, 1}``, and it drives the umklapp Bloch phase
    ``exp(-2πi G_umk · s_μ)`` at centroid fractional coords ``s_μ`` when the
    stored ψ is the cell-periodic part ``u_{n,k}`` (see the finite-q W_q
    derivation in reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md).

    Returns
    -------
    kpq_index : (nk,) int32   C-order gather index of ``k + q_off``.
    G_umk     : (nk, 3) int32 per-k reciprocal-lattice wrap.
    """
    nkx, nky, nkz = int(nkx), int(nky), int(nkz)
    qx, qy, qz = (int(v) for v in q_off)
    ix, iy, iz = np.meshgrid(
        np.arange(nkx), np.arange(nky), np.arange(nkz), indexing="ij")
    jx = ix.reshape(-1) + qx
    jy = iy.reshape(-1) + qy
    jz = iz.reshape(-1) + qz
    kpx = np.mod(jx, nkx)
    kpy = np.mod(jy, nky)
    kpz = np.mod(jz, nkz)
    Gx = jx // nkx
    Gy = jy // nky
    Gz = jz // nkz
    kpq_index = (kpx * nky * nkz + kpy * nkz + kpz).astype(np.int32)
    G_umk = np.stack([Gx, Gy, Gz], axis=1).astype(np.int32)
    return kpq_index, G_umk


def _bgw_positive_half_wrap_fractional(q_fractional):
    """Apply BGW's signed-cell wrap while retaining the ``+1/2`` tie."""
    q = np.asarray(q_fractional, dtype=np.float64)
    return np.where(q > 0.5, q - 1.0, q)


def bgw_signed_q_representative(qvec):
    r"""Return BGW's signed representative for stored fractional q rows.

    Input rows must have trailing shape ``(..., 3)`` and lie in the one-cell
    storage interval ``[-1/2, 1)`` (up to roundoff).  Components strictly
    greater than one half wrap negative; the even-grid boundary remains
    ``+1/2``.  This is the public symmetry-service door for consumers whose
    WFN contract already supplies fractional q representatives.
    """
    q = np.asarray(qvec, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 3:
        raise ValueError(
            "bgw_signed_q_representative: qvec must have trailing shape "
            f"(...,3); got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError(
            "bgw_signed_q_representative: qvec must be finite")
    tol = 32.0 * np.finfo(np.float64).eps
    if np.any(q < -0.5 - tol) or np.any(q >= 1.0 + tol):
        raise ValueError(
            "bgw_signed_q_representative: stored fractional q rows must "
            f"lie in [-1/2,1); got range [{float(np.min(q))},"
            f"{float(np.max(q))}]")
    canonical = np.mod(q, 1.0)
    return np.ascontiguousarray(
        _bgw_positive_half_wrap_fractional(canonical))


def bgw_integer_q_to_fractional(q_int_kgrid, kgrid):
    """Convert BGW integer-grid q labels to wrapped fractional vectors.

    BGW keeps the positive half-grid point on an even grid and wraps only
    labels strictly above it: ``q > kgrid/2 -> q-kgrid``.  This service owns
    that tie convention so symmetry producers and consumers cannot attach
    different fractional momenta to the same integer star table.
    """
    q = np.asarray(q_int_kgrid, dtype=np.float64)
    grid = np.asarray(kgrid, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 3:
        raise ValueError(
            "bgw_integer_q_to_fractional: q_int_kgrid must have trailing "
            f"shape (...,3); got {q.shape}.")
    if grid.shape != (3,):
        raise ValueError(
            "bgw_integer_q_to_fractional: kgrid must have shape (3,); got "
            f"{grid.shape}.")
    if not np.all(np.isfinite(q)) or not np.array_equal(q, np.rint(q)):
        raise ValueError(
            "bgw_integer_q_to_fractional: q_int_kgrid must contain finite "
            "integer labels.")
    if (not np.all(np.isfinite(grid)) or np.any(grid <= 0.0)
            or not np.array_equal(grid, np.rint(grid))):
        raise ValueError(
            "bgw_integer_q_to_fractional: kgrid must contain three finite, "
            f"positive integers; got {grid.tolist()}.")
    return _bgw_positive_half_wrap_fractional(q / grid)


def q_negation_index(kgrid):
    """C-order full-grid permutation ``index(q) -> index(-q)``.

    The q axis used by the ISDF fit, GW restart tensors, and BSE is the
    row-major flattening of ``(qx, qy, qz)``.  This service owns the
    involution so normal-equation completion, diagnostics, and downfolding
    cannot carry independent spellings of the same q convention.
    """
    raw = np.asarray(kgrid)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError(
            "q_negation_index: kgrid must contain three finite positive "
            f"integers; got shape={raw.shape}, values={raw.tolist()}.")
    grid = raw.astype(np.int64)
    if np.any(grid <= 0) or not np.array_equal(raw, grid):
        raise ValueError(
            "q_negation_index: kgrid must contain three finite positive "
            f"integers; got {raw.tolist()}.")
    coords = np.stack(
        np.unravel_index(np.arange(int(np.prod(grid))), tuple(grid)), axis=1)
    neg = (-coords) % grid[None, :]
    return np.ravel_multi_index(neg.T, tuple(grid)).astype(np.int32)


def common_uniform_grid_indices(grid_a, grid_b):
    """Aligned C-order rows shared by two unshifted uniform BZ grids.

    ``grid_a`` and ``grid_b`` describe fractional point sets
    ``(i_x/N_x, i_y/N_y, i_z/N_z)`` in ``[0, 1)^3``.  Return two ``int32``
    vectors ``(rows_a, rows_b)`` whose paired rows are the same fractional
    coordinate, ordered by the common fractional grid in C order.

    The construction is integer-only.  Along one axis, grids of lengths
    ``N_a`` and ``N_b`` share ``gcd(N_a, N_b)`` points.  Common point ``t``
    has native indices ``t*N_a/g`` and ``t*N_b/g``.  Thus 8→12 keeps the
    four-point one-dimensional intersection, not a prefix, nearest-neighbour
    match, or a fictitious eight-point nesting.

    This helper deliberately covers *unshifted* grids only.  A shifted mesh
    needs its shift in the contract and must not be silently treated as this
    point set.
    """
    def _grid(name, value):
        raw = np.asarray(value)
        if raw.shape != (3,):
            raise ValueError(
                f"common_uniform_grid_indices: {name} must have shape (3,), "
                f"got {raw.shape}.")
        rounded = np.rint(raw)
        if not np.allclose(raw, rounded, rtol=0.0, atol=0.0):
            raise ValueError(
                f"common_uniform_grid_indices: {name} must contain exact "
                f"integer extents, got {raw.tolist()}.")
        out = rounded.astype(np.int64)
        if np.any(out <= 0):
            raise ValueError(
                f"common_uniform_grid_indices: {name} extents must be "
                f"positive, got {out.tolist()}.")
        return out

    a = _grid("grid_a", grid_a)
    b = _grid("grid_b", grid_b)
    common = np.gcd(a, b)

    axes_a = [np.arange(int(g), dtype=np.int64) * int(na // g)
              for na, g in zip(a, common)]
    axes_b = [np.arange(int(g), dtype=np.int64) * int(nb // g)
              for nb, g in zip(b, common)]
    rows_a_xyz = np.stack(
        np.meshgrid(*axes_a, indexing="ij"), axis=-1).reshape(-1, 3)
    rows_b_xyz = np.stack(
        np.meshgrid(*axes_b, indexing="ij"), axis=-1).reshape(-1, 3)

    def _flat(rows, grid):
        return ((rows[:, 0] * grid[1] + rows[:, 1]) * grid[2]
                + rows[:, 2]).astype(np.int32)

    return _flat(rows_a_xyz, a), _flat(rows_b_xyz, b)


def find_irreducible_bz_points(full_kgrid_int, sym_mats_k, *, irr_kgrid_int=None):
    """For each row of ``full_kgrid_int`` (a full-BZ point in integer kgrid
    coords), find which IBZ point + ``sym_mats_k`` row maps onto it.

    Args:
        full_kgrid_int: ``(N_full, 3)`` int — full-BZ point set. The grid is
            inferred as ``full_kgrid_int.max(axis=0) + 1``.
        sym_mats_k: ``(n_sym, 3, 3)`` int — sym matrices acting on k-vectors in
            kgrid-int form. Typically TRS-augmented ``[spatial, -spatial]`` with
            ``n_sym = 2 * ntran``; ``is_trs`` is then ``sym_idx >= ntran``.
        irr_kgrid_int: optional ``(N_irr, 3)`` int — pre-specified IBZ list.
            If ``None``, the IBZ is derived as lex-smallest orbit representatives
            (q-side use). If given, IBZ is fixed (k-side use, anchored to
            ``wfn.kpoints``).

    Returns:
        irr_idx: ``(N_full,)`` int32 — IBZ row index for each full-BZ point.
        sym_idx: ``(N_full,)`` int32 — sym_mats_k row mapping IBZ → full-BZ.
        irr_kgrid_int_out: ``(N_irr, 3)`` int32 — IBZ list (echoes input if
            given, else the derived lex-min set).
    """
    full = np.asarray(full_kgrid_int, dtype=np.int64)
    Smk = np.asarray(sym_mats_k, dtype=np.int64)
    n_full = int(full.shape[0])
    n_sym = int(Smk.shape[0])
    kg = (full.max(axis=0) + 1).astype(np.int64)

    def _key(qs):
        return ((qs[..., 0] * kg[1] + qs[..., 1]) * kg[2] + qs[..., 2])

    full_keys = _key(full)

    if irr_kgrid_int is None:
        # Derive IBZ as lex-min orbit representatives over full_kgrid_int.
        images = np.einsum('sij,qj->sqi', Smk, full) % kg[None, None, :]
        image_keys = _key(images)
        best_sym = np.argmin(image_keys, axis=0)
        canon_keys = image_keys[best_sym, np.arange(n_full)]
        _, first_idx = np.unique(canon_keys, return_index=True)
        first_idx = np.sort(first_idx)
        irr_keys = canon_keys[first_idx]
        irr_out = full[first_idx].astype(np.int32)
        key_to_irr = {int(k): i for i, k in enumerate(irr_keys)}
        irr_idx = np.fromiter(
            (key_to_irr[int(k)] for k in canon_keys),
            dtype=np.int32, count=n_full,
        )
    else:
        irr_int = np.asarray(irr_kgrid_int, dtype=np.int64) % kg[None, :]
        irr_out = irr_int.astype(np.int32)
        n_irr = int(irr_int.shape[0])
        # Match existing find_symmetry_ops_simple behavior: outer iter over
        # ikbar without break ⇒ HIGHEST ikbar with any match wins, then
        # lowest sym for that ikbar. Preserves bit-equality with prior code.
        irr_images = np.einsum('sij,qj->sqi', Smk, irr_int) % kg[None, None, :]
        irr_image_keys = _key(irr_images)  # (n_sym, n_irr)
        irr_idx = np.full(n_full, -1, dtype=np.int32)
        sym_idx_pre = np.full(n_full, -1, dtype=np.int32)
        for iq in range(n_full):
            target = int(full_keys[iq])
            for ikbar in range(n_irr):
                matches = np.where(irr_image_keys[:, ikbar] == target)[0]
                if matches.size > 0:
                    irr_idx[iq] = ikbar
                    sym_idx_pre[iq] = int(matches[0])
        if np.any(irr_idx < 0):
            bad = int(np.argmin(irr_idx))
            raise RuntimeError(
                f"find_irreducible_bz_points: full point at "
                f"full_kgrid_int[{bad}]={full[bad].tolist()} has no preimage "
                "under (sym_mats_k, irr_kgrid_int).")

    # Compute sym_idx for the q-side branch (k-side has it from the inner loop).
    if irr_kgrid_int is None:
        sym_idx = np.zeros(n_full, dtype=np.int32)
        irr_images_self = np.einsum('sij,qj->sqi', Smk, full[first_idx]) % kg[None, None, :]
        irr_image_keys_self = _key(irr_images_self)
        for iq in range(n_full):
            target = int(full_keys[iq])
            ihit = int(irr_idx[iq])
            matches = np.where(irr_image_keys_self[:, ihit] == target)[0]
            if matches.size == 0:
                raise RuntimeError(
                    f"find_irreducible_bz_points: no sym maps IBZ point "
                    f"{irr_out[ihit].tolist()} to full point {full[iq].tolist()}.")
            sym_idx[iq] = int(matches[0])
    else:
        sym_idx = sym_idx_pre

    return irr_idx, sym_idx, irr_out


def map_full_kpoints_to_irreducible(
    kpoints,
    sym_mats_k,
    full_kpoints,
    *,
    tol=1.0e-6,
):
    """Map full-zone rows to stored k rows without inventing preimages.

    This is the coordinate planner used by :class:`SymMaps`.  It deliberately
    preserves the registered selection rule: the highest stored-k row with a
    match wins, then the lowest symmetry row for that stored k.  ``matched``
    is returned separately so callers can refuse incomplete WFN metadata
    before consuming the zero-initialized index arrays.

    Parameters are fractional reciprocal coordinates.  ``sym_mats_k`` is
    exactly the set of rows the caller's policy permits (spatial only, or the
    TRS-augmented table); this function never decides whether TRS is physical.
    """
    stored = np.asarray(kpoints, dtype=np.float64)
    sym = np.asarray(sym_mats_k, dtype=np.int64)
    full = np.asarray(full_kpoints, dtype=np.float64)
    if stored.ndim != 2 or stored.shape[1:] != (3,):
        raise ValueError(
            "map_full_kpoints_to_irreducible: kpoints must have shape "
            f"(nk,3), got {stored.shape}.")
    if sym.ndim != 3 or sym.shape[1:] != (3, 3) or sym.shape[0] < 1:
        raise ValueError(
            "map_full_kpoints_to_irreducible: sym_mats_k must have shape "
            f"(nsym,3,3) with nsym>0, got {sym.shape}.")
    if full.ndim != 2 or full.shape[1:] != (3,):
        raise ValueError(
            "map_full_kpoints_to_irreducible: full_kpoints must have shape "
            f"(nfull,3), got {full.shape}.")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            "map_full_kpoints_to_irreducible: tol must be finite and "
            f"positive, got {tol!r}.")

    images = np.einsum("sij,kj->ksi", sym, stored, optimize=True)
    images = np.mod(images, 1.0)
    images[images > 0.99999] = 0.0
    full_wrapped = np.mod(full, 1.0)
    full_wrapped[full_wrapped > 0.99999] = 0.0

    parent = np.zeros(full.shape[0], dtype=np.int32)
    op = np.zeros(full.shape[0], dtype=np.int32)
    matched = np.zeros(full.shape[0], dtype=bool)
    for ifull, target in enumerate(full_wrapped):
        for ikbar in range(stored.shape[0]):
            diffs = np.abs(images[ikbar] - target)
            hits = np.where(np.all(diffs < tol, axis=1))[0]
            if hits.size:
                parent[ifull] = ikbar
                op[ifull] = int(hits[0])
                matched[ifull] = True
    return parent, op, matched


@dataclass(frozen=True)
class SpatialOperatorTables:
    """Spatial WFN operations, independent of any k-mesh coverage map."""

    sym_matrices: np.ndarray
    sym_mats_k: np.ndarray
    translations: np.ndarray
    R_cart: np.ndarray
    U_spinor: np.ndarray


def build_spatial_operator_tables(wfn) -> SpatialOperatorTables:
    """Build canonical spatial/antiunitary action tables from a WFN header.

    This does not map a single k point.  The 2c reference check can therefore
    test a malformed or physically inconsistent reduced WFN and issue its TRS
    verdict before :class:`SymMaps` independently refuses incomplete coverage.
    """
    ntran = int(getattr(wfn, "ntran", 0))
    if ntran < 1:
        raise ValueError(
            f"build_spatial_operator_tables: ntran must be positive, got "
            f"{ntran}.")
    spatial = np.asarray(wfn.sym_matrices[:ntran], dtype=np.int32)
    if spatial.shape != (ntran, 3, 3):
        raise ValueError(
            "build_spatial_operator_tables: active mtrx rows must have shape "
            f"({ntran},3,3), got {spatial.shape}.")
    translations = np.asarray(
        wfn.translations[:ntran], dtype=np.float64)
    if translations.shape != (ntran, 3):
        raise ValueError(
            "build_spatial_operator_tables: active tnp rows must have shape "
            f"({ntran},3), got {translations.shape}.")
    mats_spatial = spatial.transpose(0, 2, 1).copy()
    mats_augmented = np.concatenate(
        [mats_spatial, -mats_spatial], axis=0)
    R_cart = SymMaps.syms_crystal_to_cartesian(wfn)
    U_spinor = SymMaps.get_spinor_rotations(
        wfn, R_cart[:ntran])
    return SpatialOperatorTables(
        sym_matrices=spatial,
        sym_mats_k=mats_augmented,
        translations=translations,
        R_cart=R_cart,
        U_spinor=U_spinor,
    )


def slice_q_full_to_ibz(arr_full, q_irr_full_idx, *, out_sharding=None):
    """Slice a ``(n_q_full, ...)`` array to its IBZ rows.

    The natural ``full BZ → IBZ`` companion to :func:`unfold_isdf_operator`'s
    ``IBZ → full BZ`` direction: this just picks the IBZ representative
    q-points out of a full-BZ tensor.  No centroid permute, no L-phase,
    no TRS conjugation — a pure row gather along axis 0.

    Use it whenever a q-axis quantity is built at full BZ but only the
    IBZ rows are needed for the downstream per-q step.  Two examples
    on the same shape ``(n_q, n_rmu, n_rmu)`` sharded as
    ``P(None, 'x', 'y')`` (q-axis replicated, μ on x, ν on y):

    - ``isdf_fitting.fit_zeta_to_h5``: slice C_q before ``factor_c_q``
      so Cholesky / LU runs only on the IBZ q-block, then ζ_q is solved
      and stored at IBZ; downstream V_q unfolds via
      :func:`unfold_isdf_operator`.
    - W_q = ``(1 − v_q χ_q)^{-1} v_q``: slice the Hermitian object that
      needs per-q inversion to IBZ before solve, then unfold via
      :func:`unfold_isdf_operator` for the q-axis consumers.

    Sharding contract.  The gather along axis 0 leaves the trailing
    (μ, ν) axes untouched, so XLA preserves whatever ``arr_full``
    sharding came in.  Pass ``out_sharding`` to lock in an explicit
    ``NamedSharding`` (typically ``P(None, 'x', 'y')`` for the
    Cq / V_q / χ_q quantities) — this stabilises the JIT cache key so
    repeat calls hit the same compiled module.

    Parameters
    ----------
    arr_full : jax.Array
        Shape ``(n_q_full, ...)``.
    q_irr_full_idx : np.ndarray | jax.Array
        ``(n_q_ibz,)`` int32 — full-BZ indices of the IBZ q-points.
        Sourced from :attr:`SymMaps.q_irr_full_idx`.
    out_sharding : jax.sharding.NamedSharding, optional
        If given, the output is constrained to this sharding via
        ``jax.lax.with_sharding_constraint``.

    Returns
    -------
    arr_ibz : jax.Array
        ``(n_q_ibz, ...)`` selected rows of ``arr_full``.
    """
    idx = jnp.asarray(np.asarray(q_irr_full_idx, dtype=np.int32))
    out = arr_full[idx]
    if out_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, out_sharding)
    return out


def unfold_isdf_operator(
    V_q_ibz,
    *,
    irr_idx,
    sym_idx,
    sym_perm,
    L_table,
    q_irr_frac,
    mesh_xy,
    n_sym_spatial,
    trs_rule="conj",
    right_sym_perm=None,
    right_L_table=None,
    left_logical_extent=None,
    right_logical_extent=None,
    trs_pair_q_ibz=None,
    axis_local_sym_perm=None,
    right_axis_local_sym_perm=None,
):
    """Expand ``V_q_ibz`` over the IBZ to the full BZ.

    The mapping is a centroid-axis double-gather (using the **source-map**
    ``α(μ) = sym_perm[s, μ]`` returned by ``centroid_source_map_and_wrap``)
    plus a per-centroid umklapp phase from the real-space lattice wrap.  The
    historical square form is

        V_full[q, μ', ν'] = exp(2π i q_irr · (L_{s,μ'} − L_{s,ν'}))
                            · V_ibz[i(q), α_{s}(μ'), α_{s}(ν')]

    and is exactly the ``right_* is None`` specialization of the rectangular
    response action

        X_full[q, μ_L, ν_R]
          = exp(2π i q_irr · (L^L_{s,μ_L} - L^R_{s,ν_R}))
            X_ibz[i(q), α^L_s(μ_L), α^R_s(ν_R)].

    Distinct left/right tables are required for CT/TC response tiles whose
    charge and transverse centroid bases differ.  Each padded endpoint stays
    on its native processor axis, ``P(None,'x','y')``; neither open centroid
    axis is gathered onto one rank.

    where ``i(q) = irr_idx[q]``, ``s(q) = sym_idx[q]``,
    ``q_irr = q_irr_frac[i(q)]`` is the IBZ parent q in fractional
    reciprocal coords, and ``α_s(μ) = sym_perm[s, μ]``,
    ``L_{s,μ} = L_table[s, μ]`` come from the source-map decomposition
    ``y_μ = mtrx · (x_μ − τ) = x_{α(μ)} + L_μ`` (the user-spec inverse
    form; see ``docs/SYMMETRY_COMPREHENSIVE.md`` §4 and §5).

    The phase factor is essential whenever ``S r_μ + τ`` exits the
    unit cell (i.e. ``L_μ ≠ 0``) — which happens for every non-trivial
    full-BZ q on a non-cubic / non-symmorphic system.  Skipping the
    phase produces a ~unity-relative error on umklapp q's (verified
    empirically on CrI3 30 Ry V_q dumps before this fix).

    TRS-augmented rows
    ------------------
    ``sym_idx`` values may be in ``[n_sym_spatial, 2·n_sym_spatial)`` for
    q's that fold to their IBZ parent only via time reversal.  Per-element
    derivation (``ζ_{-q,μ}(G) = ζ*_{q,μ}(-G)`` combined with ``v(|q+G|)``
    real-and-even-in-K) gives, for the scalar (charge-channel) V_q::

        V_full[TRS-q, π_s(μ), π_s(ν)] = conj(V_ibz[i(q), μ, ν])

    For Hermitian V_q the conj equals the ν↔μ transpose; we implement
    conj for clarity (and to keep the helper correct for any future
    non-Hermitian channels).  For a rectangular general operator,
    ``trs_pair_q_ibz`` is the reversed-axis partner ``(q,ν_R,μ_L)``;
    time reversal uses its transpose.  This is the exact CT↔TC pair action,
    not a shape-based transpose of CT itself.  The centroid permutation is
    unchanged under TRS (r is fixed); ``sym_perm`` rows ``[ntran:]``
    duplicate ``[:ntran]``.  Callers build ``sym_perm`` via
    ``centroid_source_map_and_wrap(..., extend_trs=True)`` and pass
    ``n_sym_spatial=ntran``.

    Parameters
    ----------
    V_q_ibz
        ``(n_q_ibz, n_rmu, n_rmu)`` complex, sharded ``P(None,'x','y')``.
    irr_idx
        ``(n_q_full,)`` int — IBZ index per full-BZ q (``sym.irr_idx_q``).
    sym_idx
        ``(n_q_full,)`` int — sym row per full-BZ q (``sym.sym_idx_q``).
        Values in ``[0, 2·n_sym_spatial)``.
    sym_perm
        ``(2·n_sym_spatial, n_rmu)`` int — centroid permutation table.
        Must cover ``max(sym_idx)``; we raise a clear error otherwise
        rather than relying on JAX's silent OOB clamp.
    L_table
        ``(2·n_sym_spatial, n_rmu, 3)`` int — per-(sym, centroid)
        integer real-space lattice wrap, from
        ``centroid_source_map_and_wrap``.  Drives the umklapp phase.
    q_irr_frac
        ``(n_q_ibz, 3)`` float — IBZ q in fractional reciprocal
        coordinates (already BGW-wrapped to the (−0.5, 0.5] convention
        if the caller is consistent).  Indexed by ``irr_idx``.
    mesh_xy
        Device mesh; the output is constrained to ``P(None,'x','y')``.
    n_sym_spatial
        ``ntran`` — count of spatial-only sym ops in ``sym_perm``'s
        first half.  Used to identify TRS-augmented rows
        (``sym_idx >= n_sym_spatial``) and apply the required ``conj``.
    trs_rule
        ``"conj"`` for Hermitian inputs (the historical default), or
        ``"pair_transpose"`` for a general complex-frequency operator.
        Time reversal transposes its centroid pair without conjugating its
        frequency dependence.
    right_sym_perm, right_L_table
        Optional right-endpoint source-map/wrap tables.  Both must be supplied
        together.  Omission preserves the historical square same-basis action.
    left_logical_extent, right_logical_extent
        Physical endpoint extents inside the padded array.  Padding is
        required to form an invariant tail under every used source map and is
        zeroed structurally in the result.  Defaults are the full stored axes.
    trs_pair_q_ibz
        Reversed-axis partner ``(n_q_ibz,n_right,n_left)`` for rectangular
        ``trs_rule='pair_transpose'``.  Square callers retain the historical
        self-transpose default.
    axis_local_sym_perm, right_axis_local_sym_perm
        Optional packed-view gather offsets with the same shapes as the
        corresponding global permutation tables.  Each value is local to
        its target X or Y shard.  Supplying these certifies an orbit-packed
        basis and replaces the two all-to-all permutation round trips with
        local gathers.  The global tables remain required and are checked
        against these offsets.  A rectangular basis supplies both local
        tables together.
    Returns
    -------
    V_q_full
        ``(n_q_full, n_left, n_right)`` complex, sharded
        ``P(None,'x','y')``.
    """
    # Trivial-IBZ short-circuit. When ntran=1 (e.g. nosym runs) the IBZ is
    # already the full BZ — irr_idx is identity, sym_idx is all zeros,
    # sym_perm is identity. The take_along_axis path below is then a
    # no-op but its sharded codegen has been observed to trip an XLA HLO
    # verifier dtype mismatch (s64 broadcast vs s32 operand on a 2×2
    # mesh), so bypass it entirely.
    idx_np = np.asarray(irr_idx)
    sym_np = np.asarray(sym_idx)
    square_defaults = (
        right_sym_perm is None and right_L_table is None
        and left_logical_extent is None and right_logical_extent is None
        and trs_pair_q_ibz is None and axis_local_sym_perm is None
        and right_axis_local_sym_perm is None)
    if (square_defaults
            and idx_np.shape[0] == int(V_q_ibz.shape[0])
            and np.array_equal(idx_np, np.arange(idx_np.shape[0]))
            and np.all(sym_np == 0)):
        return V_q_ibz

    same_basis_tables = right_sym_perm is None and right_L_table is None
    if (right_sym_perm is None) != (right_L_table is None):
        raise ValueError(
            "unfold_isdf_operator: right_sym_perm and right_L_table must "
            "be supplied together for a rectangular endpoint action")
    left_perm = np.asarray(sym_perm, dtype=np.int32)
    right_perm = (left_perm if right_sym_perm is None else
                  np.asarray(right_sym_perm, dtype=np.int32))
    left_L = np.asarray(L_table, dtype=np.float64)
    right_L = (left_L if right_L_table is None else
               np.asarray(right_L_table, dtype=np.float64))
    if right_axis_local_sym_perm is not None and axis_local_sym_perm is None:
        raise ValueError(
            "unfold_isdf_operator: right_axis_local_sym_perm requires "
            "axis_local_sym_perm")
    left_local_perm = (None if axis_local_sym_perm is None else
                       np.asarray(axis_local_sym_perm, dtype=np.int32))
    right_local_perm = (
        left_local_perm if right_axis_local_sym_perm is None
        else np.asarray(right_axis_local_sym_perm, dtype=np.int32))
    if (not same_basis_tables and left_local_perm is not None
            and right_axis_local_sym_perm is None):
        raise ValueError(
            "unfold_isdf_operator: a rectangular endpoint action must "
            "supply right_axis_local_sym_perm with axis_local_sym_perm")
    if left_perm.ndim != 2 or right_perm.ndim != 2:
        raise ValueError(
            "unfold_isdf_operator: left/right sym_perm tables must both be "
            "rank 2")
    if (left_L.ndim != 3 or left_L.shape[-1] != 3
            or right_L.ndim != 3 or right_L.shape[-1] != 3):
        raise ValueError(
            "unfold_isdf_operator: left/right L_table arrays must both have "
            "shape (n_sym_rows,n_endpoint,3)")
    if getattr(V_q_ibz, "ndim", None) != 3:
        raise ValueError(
            "unfold_isdf_operator: V_q_ibz must have rank 3 "
            "(n_q_ibz,n_left,n_right)")
    n_sym_perm = int(left_perm.shape[0])
    n_sym_perm_right = int(right_perm.shape[0])
    max_sym = int(sym_np.max()) if sym_np.size else -1
    if max_sym >= n_sym_perm or max_sym >= n_sym_perm_right:
        if same_basis_tables:
            raise ValueError(
                f"unfold_isdf_operator: sym_idx contains value {max_sym} "
                f"but sym_perm has only {n_sym_perm} rows.  Build it via "
                "``centroid_source_map_and_wrap(..., extend_trs=True)`` so "
                "it covers the TRS-augmented half of ``sym_mats_k``.")
        raise ValueError(
            f"unfold_isdf_operator: sym_idx contains value {max_sym} "
            "but the left/right sym_perm tables have only "
            f"{n_sym_perm}/{n_sym_perm_right} rows.  Build them via "
            "``centroid_source_map_and_wrap(..., extend_trs=True)`` so they "
            "cover the TRS-augmented half of ``sym_mats_k``.")
    requested_trs_rule = trs_rule
    if requested_trs_rule not in ("conj", "pair_transpose"):
        raise ValueError(
            "unfold_isdf_operator: trs_rule must be 'conj' for a Hermitian "
            "operator or 'pair_transpose' for a general operator")
    trs_used = max_sym >= int(n_sym_spatial)
    if not trs_used:
        # Keep centrosymmetric inputs on the historical compiled module.
        trs_rule = "conj"
    if (trs_used and (int(n_sym_spatial) * 2 != n_sym_perm
                     or int(n_sym_spatial) * 2 != n_sym_perm_right)):
        if same_basis_tables:
            raise ValueError(
                f"unfold_isdf_operator: sym_idx uses TRS-augmented rows "
                f"(max={max_sym} ≥ n_sym_spatial={n_sym_spatial}) but "
                f"sym_perm.shape[0]={n_sym_perm} ≠ 2·n_sym_spatial.  "
                "Build sym_perm via ``centroid_source_map_and_wrap(..., "
                "extend_trs=True)``.")
        raise ValueError(
            f"unfold_isdf_operator: sym_idx uses TRS-augmented rows "
            f"(max={max_sym} ≥ n_sym_spatial={n_sym_spatial}) but "
            "left/right sym_perm row counts are "
            f"{n_sym_perm}/{n_sym_perm_right}, not both 2·n_sym_spatial.  "
            "Build both via ``centroid_source_map_and_wrap(..., "
            "extend_trs=True)``.")

    # Forward permutation: gather V_ibz at indices sym_perm[s, μ'] — i.e.,
    # at the FORWARD image of each full-BZ centroid μ' under sym s.
    # Empirically (see ``reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py``):
    # V_full[μ', ν'] = phase(μ', ν') · V_ibz[parent, sym_perm[s, μ'],
    # sym_perm[s, ν']] closes to ISDF noise floor on all 36 q's of the
    # CrI3 6×6 30 Ry dump including order-3 ops.  The prior code used
    # inv_perm = argsort(sym_perm) (= π⁻¹) which is a no-op for
    # involutive ops (MoS2 σ_h, Si cubic) but wrong for order-3 (CrI3
    # C3) — that was the silent 4 eV gap on hex systems.
    # The μ pad is baked into the tables at construction
    # (``_resolve_ibz_q_list``: identity tail on the permutation, zero
    # tail on L).  REQUIRE an exact extent match in BOTH directions — a
    # mismatch would otherwise feed ``promise_in_bounds`` gathers with
    # out-of-range indices and clip SILENTLY (the TRS-bug failure
    # shape).  Logical/logical callers (tests, unpadded runs) match
    # trivially.
    fwd_perm = left_perm
    fwd_perm_right = right_perm
    n_left = int(V_q_ibz.shape[-2])
    n_right = int(V_q_ibz.shape[-1])
    if n_left != int(fwd_perm.shape[-1]):
        if same_basis_tables:
            raise ValueError(
                f"unfold_isdf_operator: V_q_ibz μ-extent {n_left} != "
                f"sym_perm extent {int(fwd_perm.shape[-1])}.  Tables carry "
                "the μ pad from construction (_resolve_ibz_q_list); pass V "
                "at the same (padded) extent.")
        raise ValueError(
            f"unfold_isdf_operator: V_q_ibz left extent {n_left} != "
            "left sym_perm extent "
            f"{int(fwd_perm.shape[-1])}.  Tables carry "
            "the endpoint pad from construction (_resolve_ibz_q_list); pass V "
            f"at the same (padded) extent.")
    if n_right != int(fwd_perm_right.shape[-1]):
        raise ValueError(
            f"unfold_isdf_operator: V_q_ibz right extent {n_right} != "
            f"right sym_perm extent {int(fwd_perm_right.shape[-1])}.")
    # Per-(q_full, μ) umklapp phase factor exp(2π i q_irr · L_μ).
    # ``L_table`` is shape (2·ntran, n_rmu, 3) int (any width); promote
    # to float64 then gather to (n_q_full, n_rmu, 3).  ``q_irr_frac`` is
    # (n_q_ibz, 3); we gather to (n_q_full, 3) via ``irr_idx``.  The
    # product qL = q_irr · L is (n_q_full, n_rmu), then the bilinear
    # phase is qL[μ] − qL[ν] (per-q, outer-diff).
    L_arr = left_L
    L_arr_right = right_L
    if (int(L_arr.shape[0]) != n_sym_perm
            or int(L_arr_right.shape[0]) != n_sym_perm_right):
        raise ValueError(
            "unfold_isdf_operator: each L_table must have the same symmetry "
            "row count as its endpoint sym_perm table")
    if int(L_arr.shape[1]) != n_left:
        if same_basis_tables:
            raise ValueError(
                f"unfold_isdf_operator: L_table μ-extent "
                f"{int(L_arr.shape[1])} != V_q_ibz μ-extent {n_left}; "
                "tables and V must share one (padded) extent.")
        raise ValueError(
            "unfold_isdf_operator: left L_table extent "
            f"{int(L_arr.shape[1])} != V_q_ibz left extent {n_left}.")
    if int(L_arr_right.shape[1]) != n_right:
        raise ValueError(
            "unfold_isdf_operator: right L_table extent "
            f"{int(L_arr_right.shape[1])} != V_q_ibz right extent {n_right}.")

    Px = int(mesh_xy.shape['x'])
    Py = int(mesh_xy.shape['y'])
    if left_local_perm is not None:
        if left_local_perm.shape != fwd_perm.shape:
            raise ValueError(
                "unfold_isdf_operator: axis_local_sym_perm must match "
                f"sym_perm shape {fwd_perm.shape}; got "
                f"{left_local_perm.shape}.")
        if right_local_perm.shape != fwd_perm_right.shape:
            raise ValueError(
                "unfold_isdf_operator: right_axis_local_sym_perm must match "
                f"the right sym_perm shape {fwd_perm_right.shape}; got "
                f"{right_local_perm.shape}.")
        if n_left % Px or n_right % Py:
            raise ValueError(
                "unfold_isdf_operator: axis-local maps require endpoint "
                f"extents divisible by X/Y; got {n_left}/{Px} and "
                f"{n_right}/{Py}.")
        left_chunk, right_chunk = n_left // Px, n_right // Py
        for label, global_perm, local_perm, chunk in (
                ("left", fwd_perm, left_local_perm, left_chunk),
                ("right", fwd_perm_right, right_local_perm, right_chunk)):
            target_owner = np.arange(global_perm.shape[1]) // chunk
            source_owner = global_perm // chunk
            if not np.array_equal(
                    source_owner, np.broadcast_to(
                        target_owner, source_owner.shape)):
                raise ValueError(
                    "unfold_isdf_operator: axis-local certification failed: "
                    f"the {label} global source map crosses an axis shard.")
            if not np.array_equal(local_perm, global_perm % chunk):
                raise ValueError(
                    "unfold_isdf_operator: axis-local certification failed: "
                    f"the {label} local offsets disagree with the global "
                    "source map modulo its shard extent.")

    logical_left = n_left if left_logical_extent is None else int(
        left_logical_extent)
    logical_right = n_right if right_logical_extent is None else int(
        right_logical_extent)
    if not 0 < logical_left <= n_left or not 0 < logical_right <= n_right:
        raise ValueError(
            "unfold_isdf_operator: logical left/right extents must be positive "
            f"and within padded {(n_left, n_right)}; got "
            f"{(logical_left, logical_right)}")
    for label, perm, logical, padded in (
            ("left", fwd_perm, logical_left, n_left),
            ("right", fwd_perm_right, logical_right, n_right)):
        used = perm[sym_np] if sym_np.size else perm[:0]
        if (np.any(used[:, :logical] >= logical)
                or np.any(used[:, logical:] < logical)):
            raise ValueError(
                f"unfold_isdf_operator: {label} source maps do not preserve "
                f"the logical/padded split at {logical}/{padded}; regenerate "
                "the authenticated centroid tables with an identity tail")

    pair_source = None
    if requested_trs_rule == "pair_transpose":
        if trs_pair_q_ibz is None:
            if trs_used and (n_left != n_right or not same_basis_tables):
                raise ValueError(
                    "unfold_isdf_operator: rectangular pair_transpose requires "
                    "trs_pair_q_ibz with reversed endpoint axes")
            if trs_used:
                pair_source = V_q_ibz
        else:
            expected_pair = (int(V_q_ibz.shape[0]), n_right, n_left)
            if tuple(int(v) for v in trs_pair_q_ibz.shape) != expected_pair:
                raise ValueError(
                    "unfold_isdf_operator: trs_pair_q_ibz must have reversed "
                    f"shape {expected_pair}; got {trs_pair_q_ibz.shape}")
            if trs_used:
                pair_source = trs_pair_q_ibz
    elif trs_pair_q_ibz is not None:
        raise ValueError(
            "unfold_isdf_operator: trs_pair_q_ibz is valid only with "
            "trs_rule='pair_transpose'")

    # Sym tables are baked into the jit closure as constants — XLA folds
    # them into the HLO, which is materially faster per call than
    # marshalling them as runtime args (verified empirically: runtime-
    # arg form was ~2× slower per call than closure form).  The cache
    # keys on the content of the tables (via bytes hashes) so V_q's and
    # W_q's calls with the same sym/centroid configuration share one
    # compiled module without re-baking constants.
    idx_arr = np.asarray(irr_idx, dtype=np.int32)
    sym_arr = np.asarray(sym_idx, dtype=np.int32)
    q_irr_arr = np.asarray(q_irr_frac, dtype=np.float64)
    trs_mask_arr = (sym_arr >= int(n_sym_spatial))
    fn = _get_unfold_isdf_operator_jit(
        V_q_shape=tuple(int(s) for s in V_q_ibz.shape),
        fwd_perm_arr=fwd_perm,
        fwd_perm_right_arr=fwd_perm_right,
        idx_arr=idx_arr,
        sym_arr=sym_arr,
        L_arr=L_arr,
        L_right_arr=L_arr_right,
        q_irr_arr=q_irr_arr,
        trs_mask_arr=trs_mask_arr,
        logical_left=logical_left,
        logical_right=logical_right,
        n_sym_spatial=int(n_sym_spatial),
        trs_rule=trs_rule,
        left_local_perm_arr=left_local_perm,
        right_local_perm_arr=right_local_perm,
        mesh_xy=mesh_xy)
    return fn(V_q_ibz, pair_source) if pair_source is not None else fn(V_q_ibz)


_UNFOLD_ISDF_OPERATOR_JIT_CACHE: dict = {}


def _get_unfold_isdf_operator_jit(
    *, V_q_shape, fwd_perm_arr, fwd_perm_right_arr, idx_arr, sym_arr,
    L_arr, L_right_arr, q_irr_arr, trs_mask_arr, logical_left,
    logical_right, n_sym_spatial, mesh_xy, left_local_perm_arr=None,
    right_local_perm_arr=None,
    trs_rule="conj",
):
    """Cache the inner ``_do_unfold`` jit by (shape, sym table content).

    V_q and W_q with the same sym / centroid configuration share the
    same compiled HLO (cache hit on bytes-hash of the tables).  The
    tables are baked into the jit closure as constants — runtime-arg
    form was ~2× slower per call than closure-baked due to
    per-invocation argument marshalling.
    """
    key = (
        V_q_shape,
        fwd_perm_arr.shape, fwd_perm_arr.tobytes(),
        fwd_perm_right_arr.shape, fwd_perm_right_arr.tobytes(),
        idx_arr.tobytes(),
        sym_arr.tobytes(),
        L_arr.shape, L_arr.tobytes(),
        L_right_arr.shape, L_right_arr.tobytes(),
        q_irr_arr.tobytes(),
        trs_mask_arr.tobytes(),
        int(logical_left), int(logical_right),
        int(n_sym_spatial),
        str(trs_rule),
        (None if left_local_perm_arr is None else
         (left_local_perm_arr.shape, left_local_perm_arr.tobytes())),
        (None if right_local_perm_arr is None else
         (right_local_perm_arr.shape, right_local_perm_arr.tobytes())),
        id(mesh_xy),
    )
    hit = _UNFOLD_ISDF_OPERATOR_JIT_CACHE.get(key)
    if hit is not None:
        return hit

    # Promote to jax arrays once at trace-build time.  Closure capture
    # makes these constants in the compiled HLO.
    fwd_perm_j = jnp.asarray(fwd_perm_arr)
    fwd_perm_right_j = jnp.asarray(fwd_perm_right_arr)
    idx_j = jnp.asarray(idx_arr)
    sym_j = jnp.asarray(sym_arr)
    L_j = jnp.asarray(L_arr)
    L_right_j = jnp.asarray(L_right_arr)
    q_irr_j = jnp.asarray(q_irr_arr)
    trs_mask_j = jnp.asarray(trs_mask_arr)
    left_local_perm_j = (None if left_local_perm_arr is None else
                         jnp.asarray(left_local_perm_arr))
    right_local_perm_j = (None if right_local_perm_arr is None else
                          jnp.asarray(right_local_perm_arr))
    # Memory contract: never exceed 1× single-tile per rank.  Canonical
    # ordering uses volume-preserving all_to_all redistributions.  An
    # authenticated orbit-packed order instead gathers each axis locally.
    n_left_padded = int(V_q_shape[-2])
    n_right_padded = int(V_q_shape[-1])
    Px = int(mesh_xy.shape['x'])
    Py = int(mesh_xy.shape['y'])
    if (left_local_perm_arr is None
            and (n_left_padded % (Px * Py) != 0
                 or n_right_padded % (Px * Py) != 0)):
        if n_left_padded == n_right_padded:
            raise ValueError(
                f"unfold_isdf_operator: n_rmu_padded={n_left_padded} must be "
                f"divisible by Px*Py={Px*Py} for the all_to_all "
                "redistribution.  The μ-padding in Meta should already "
                "enforce this — check that meta.n_rmu_padded is "
                "mesh-divisible.")
        raise ValueError(
            "unfold_isdf_operator: left/right padded extents "
            f"{n_left_padded}/{n_right_padded} must each be divisible by "
            f"Px*Py={Px*Py} for the all_to_all redistribution.  Endpoint "
            "padding should already enforce this; check both authenticated "
            "basis receipts.")
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    pair_transpose = (trs_rule == "pair_transpose")
    in_specs = ((P(None, 'x', 'y'), P(None, 'x', 'y')) if pair_transpose
                else P(None, 'x', 'y'))

    @partial(shard_map, mesh=mesh_xy,
             in_specs=in_specs,
             out_specs=P(None, 'x', 'y'),
             check_vma=False)
    def _kernel(*operands):
        # V_ibz_local: (n_q_ibz, μ/Px, ν/Py)
        V_ibz_local = operands[0]
        perm_left_q = fwd_perm_j[sym_j]
        perm_right_q = fwd_perm_right_j[sym_j]
        # Gather q axis (replicated → local selection via idx_j).
        if pair_transpose:
            forward_at_irr = V_ibz_local[idx_j]
            partner_at_irr = operands[1][idx_j]
            V_at_irr = jnp.where(
                trs_mask_j[:, None, None], partner_at_irr, forward_at_irr)
        else:
            V_at_irr = V_ibz_local[idx_j]

        mu_local = n_left_padded // Px
        nu_local = n_right_padded // Py
        x_idx = jax.lax.axis_index('x')
        y_idx = jax.lax.axis_index('y')

        # μ permute on 'x'.  all_to_all redistributes:
        #   split  ν (local, /Py)  → ν / (Py·Px)
        #   concat μ (/Px sharded) → full μ
        # Volume per rank: n_q · μ · ν / (Px·Py) — unchanged from 1× tile.
        # Required: ν/Py divisible by Px (ensured by the right pad).
        if left_local_perm_j is not None:
            perm_left_local_q = jax.lax.dynamic_slice_in_dim(
                left_local_perm_j[sym_j], x_idx * mu_local, mu_local,
                axis=1)
            V_perm_mu = jnp.take_along_axis(
                V_at_irr, perm_left_local_q[:, :, None], axis=1,
                mode='promise_in_bounds')
        elif Px > 1:
            V_x = jax.lax.all_to_all(
                V_at_irr, 'x', split_axis=2, concat_axis=1, tiled=True)
            # (n_q_full, μ, ν/(Px·Py))
            V_x_perm = jnp.take_along_axis(
                V_x, perm_left_q[:, :, None], axis=1,
                mode='promise_in_bounds')
            V_perm_mu = jax.lax.all_to_all(
                V_x_perm, 'x', split_axis=1, concat_axis=2, tiled=True)
            # (n_q_full, μ/Px, ν/Py)  — back to canonical
        else:
            V_perm_mu = jnp.take_along_axis(
                V_at_irr, perm_left_q[:, :, None], axis=1,
                mode='promise_in_bounds')

        # ν permute on 'y'.  Mirror trick on the 'y' axis.
        # Required: μ/Px divisible by Py (ensured by the left pad).
        if right_local_perm_j is not None:
            perm_right_local_q = jax.lax.dynamic_slice_in_dim(
                right_local_perm_j[sym_j], y_idx * nu_local, nu_local,
                axis=1)
            V_full_local = jnp.take_along_axis(
                V_perm_mu, perm_right_local_q[:, None, :], axis=2,
                mode='promise_in_bounds')
        elif Py > 1:
            V_y = jax.lax.all_to_all(
                V_perm_mu, 'y', split_axis=1, concat_axis=2, tiled=True)
            # (n_q_full, μ/(Px·Py), ν)
            V_y_perm = jnp.take_along_axis(
                V_y, perm_right_q[:, None, :], axis=2,
                mode='promise_in_bounds')
            V_full_local = jax.lax.all_to_all(
                V_y_perm, 'y', split_axis=2, concat_axis=1, tiled=True)
            # (n_q_full, μ/Px, ν/Py)  — back to canonical
        else:
            V_full_local = jnp.take_along_axis(
                V_perm_mu, perm_right_q[:, None, :], axis=2,
                mode='promise_in_bounds')

        # Umklapp phase: exp(2π i q_irr · (L_μ − L_ν)).  L_μ here
        # is L_table[s(q), μ] — wrap of centroid μ under sym op
        # s(q) (NOT permuted).  See
        # ``reports/trs_sym_audit_2026-05-14/verify_umklapp_user_math.py``.
        # Phase tables are small (~n_q · n_rmu c128 bytes); compute
        # replicated and slice this rank's μ_local / ν_local extent.
        L_left_per_q = L_j[sym_j]
        L_right_per_q = L_right_j[sym_j]
        q_per_q = q_irr_j[idx_j]                        # (n_q_full, 3)
        qL_left = jnp.einsum(
            'qi,qmi->qm', q_per_q, L_left_per_q)
        qL_right = jnp.einsum(
            'qi,qmi->qm', q_per_q, L_right_per_q)
        phase_left = jnp.exp(
            2j * jnp.pi * qL_left.astype(jnp.complex128))
        phase_right = jnp.exp(
            2j * jnp.pi * qL_right.astype(jnp.complex128))
        phase_mu = jax.lax.dynamic_slice_in_dim(
            phase_left, x_idx * mu_local, mu_local, axis=1)
        phase_nu = jax.lax.dynamic_slice_in_dim(
            phase_right, y_idx * nu_local, nu_local, axis=1)
        if pair_transpose:
            mu_phase = jnp.where(trs_mask_j[:, None],
                                 jnp.conj(phase_mu), phase_mu)
            nu_phase = jnp.where(trs_mask_j[:, None],
                                 phase_nu, jnp.conj(phase_nu))
            V_full_local = (mu_phase[:, :, None] * V_full_local
                            * nu_phase[:, None, :])
        else:
            V_full_local = (phase_mu[:, :, None] * V_full_local
                            * jnp.conj(phase_nu)[:, None, :])
            V_full_local = jnp.where(
                trs_mask_j[:, None, None],
                jnp.conj(V_full_local), V_full_local)
        if (int(logical_left) != n_left_padded
                or int(logical_right) != n_right_padded):
            global_left = x_idx * mu_local + jnp.arange(mu_local)
            global_right = y_idx * nu_local + jnp.arange(nu_local)
            valid = ((global_left[:, None] < int(logical_left))
                     & (global_right[None, :] < int(logical_right)))
            V_full_local = jnp.where(valid[None], V_full_local, 0)
        return V_full_local

    if pair_transpose:
        partner_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))

        @partial(jax.jit, in_shardings=(V_sh, partner_sh), out_shardings=V_sh)
        def _do_unfold(V_ibz, pair_ibz):
            transposed = jax.lax.with_sharding_constraint(
                jnp.swapaxes(pair_ibz, -2, -1), V_sh)
            return _kernel(V_ibz, transposed)
    else:
        @partial(jax.jit, in_shardings=V_sh, out_shardings=V_sh)
        def _do_unfold(V_ibz):
            return _kernel(V_ibz)

    _UNFOLD_ISDF_OPERATOR_JIT_CACHE[key] = _do_unfold
    return _do_unfold


def unfold_spin_centroid_operator(
    operator_ibz,
    *,
    irr_idx,
    sym_idx,
    sym_perm,
    L_table,
    k_irr_frac,
    spin_action_full,
    n_sym_spatial,
    mesh_xy,
    logical_centroid_extent=None,
    axis_local=False,
):
    r"""Unfold an open-spin centroid operator from k parents to full k.

    ``operator_ibz`` has shape ``(nk_parent,s,mu,s,nu)`` and sharding
    ``P(None,None,'x',None,'y')``.  The two endpoint pairs are merged in
    centroid-major order, transported by :func:`unfold_isdf_operator`, then
    rotated by the canonical spin representation::

        O_k[a,mu,b,nu] = U_k[a,c]
            O_parent[c,alpha(mu),d,alpha(nu)] U_k[b,d]^* .

    On an antiunitary row the parent operator is transposed in the complete
    ``(spin,centroid)`` endpoint space, not merely conjugated.  This matters
    for a real-time Green function whose band weights are complex:
    ``Theta G(t) Theta^-1`` uses ``G(t)^T``; ``conj(G(t))`` would silently
    reverse ``t``.  The underlying ``pair_transpose`` rule also owns the
    nonsymmorphic lattice-wrap phase.

    ``axis_local=True`` is accepted only when the supplied packed global
    source maps prove that every endpoint gather stays within its X/Y shard.
    The lower-level owner authenticates that claim before compiling the
    collective-free local-gather kernel.
    """
    shape = tuple(int(v) for v in operator_ibz.shape)
    if len(shape) != 5 or shape[1] != shape[3]:
        raise ValueError(
            "unfold_spin_centroid_operator: operator_ibz must have shape "
            f"(nk,s,mu,s,nu); got {shape}.")
    nk_parent, ns, n_left, _, n_right = shape
    if n_left != n_right:
        raise ValueError(
            "unfold_spin_centroid_operator: the first implementation owns "
            "one square centroid basis; mixed endpoint bases must use one "
            "authenticated rectangular plan, not truncate or pad implicitly.")
    spin = np.asarray(spin_action_full, dtype=np.complex128)
    n_full = int(np.asarray(irr_idx).shape[0])
    if spin.shape != (n_full, ns, ns):
        raise ValueError(
            "unfold_spin_centroid_operator: spin_action_full must be "
            f"({n_full},{ns},{ns}); got {spin.shape}.")
    if int(np.asarray(k_irr_frac).shape[0]) != nk_parent:
        raise ValueError(
            "unfold_spin_centroid_operator: k_irr_frac and operator_ibz "
            f"must share nk_parent={nk_parent} rows.")

    perm = np.asarray(sym_perm, dtype=np.int32)
    wraps = np.asarray(L_table)
    if perm.shape[1:] != (n_left,) or wraps.shape != perm.shape + (3,):
        raise ValueError(
            "unfold_spin_centroid_operator: sym_perm/L_table must have "
            f"shapes (n_sym,{n_left})/(n_sym,{n_left},3); got "
            f"{perm.shape}/{wraps.shape}.")

    # Merge order is (mu, spin), so a source centroid alpha(mu) expands to
    # the contiguous source rows alpha(mu)*ns + spin.  L is spin-independent.
    spin_slot = np.arange(ns, dtype=np.int32)
    perm_ms = (perm[:, :, None] * ns + spin_slot[None, None, :]).reshape(
        perm.shape[0], n_left * ns)
    wraps_ms = np.repeat(wraps, ns, axis=1)
    logical_mu = (n_left if logical_centroid_extent is None else
                  int(logical_centroid_extent))
    if not 0 < logical_mu <= n_left:
        raise ValueError(
            "unfold_spin_centroid_operator: logical_centroid_extent must be "
            f"inside (0,{n_left}]; got {logical_mu}.")

    flat_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    out_sh = NamedSharding(mesh_xy, P(None, None, 'x', None, 'y'))
    flat = jax.lax.with_sharding_constraint(
        jnp.transpose(operator_ibz, (0, 2, 1, 4, 3)).reshape(
            nk_parent, n_left * ns, n_right * ns),
        flat_sh)
    local_perm = None
    if bool(axis_local):
        px = int(mesh_xy.shape['x'])
        if (n_left * ns) % px:
            raise ValueError(
                "unfold_spin_centroid_operator: merged endpoint extent "
                f"{n_left * ns} is not divisible by X={px}.")
        local_perm = perm_ms % ((n_left * ns) // px)
    flat_full = unfold_isdf_operator(
        flat,
        irr_idx=irr_idx,
        sym_idx=sym_idx,
        sym_perm=perm_ms,
        L_table=wraps_ms,
        q_irr_frac=k_irr_frac,
        mesh_xy=mesh_xy,
        n_sym_spatial=n_sym_spatial,
        trs_rule="pair_transpose",
        left_logical_extent=logical_mu * ns,
        right_logical_extent=logical_mu * ns,
        axis_local_sym_perm=local_perm,
    )
    spatial = jnp.transpose(
        flat_full.reshape(n_full, n_left, ns, n_right, ns),
        (0, 2, 1, 4, 3))
    U = jnp.asarray(spin)
    rotated = jnp.einsum(
        'kac,kcmdn,kbd->kambn', U, spatial, jnp.conj(U), optimize=True)
    return jax.lax.with_sharding_constraint(rotated, out_sh)


def unfold_isdf_one_leg(
    zeta_ibz,
    *,
    gvec_components=None,
    source_gvec_components=None,
    sym,
    sym_idx,
    sym_perm,
    L_table,
    q_irr_frac,
    kgrid,
    mesh_xy,
    component_action,
    source_component=None,
):
    r"""Transport one ISDF Fourier leg from the q-IBZ to the full q grid.

    This is the one-leg companion to :func:`unfold_isdf_operator`.  It
    normally consumes the stored q-IBZ G-sphere rather than an
    already-selected ``G=0`` column, because under a star operation the
    full-zone literal ``G=0`` coefficient can come from a *nonzero* parent
    Miller vector::

        q_full = S_full (q_parent + G_parent).

    For target centroid ``mu`` the scalar action is

    .. math::

       z_{q,\mu}(0) = e^{-i(S K_p)\cdot t_s}
                       e^{-2\pi i q_p\cdot L_{s\mu}}
                       z_{p,\alpha_s(\mu)}(G_p),

    followed by conjugation of the complete spatial result on an
    antiunitary row.  ``alpha`` is the *source* map returned by
    :func:`centroid_source_map_and_wrap`; it is used directly, never
    inverted.  ``t_s`` is raw BGW ``tnp = 2*pi*tau``.  The tau factor is
    formed with the spatial row and conjugated with the rest of the result
    under time reversal, matching :func:`unfold_psi`.

    ``component_action='scalar'`` returns that scalar leg.  The ``'polar'``
    action streams one source Cartesian component through the canonical
    polar, time-odd :meth:`SymMaps.cartesian_action`.  Summing the three
    source components gives the full vector action while only one large zeta
    slab is resident.

    ``sym_idx`` is deliberately explicit: callers must pass the measured
    :class:`QgridTrsPolicy`'s ``unfold_sym_idx``.  This function never falls
    back to ``sym.sym_idx_q`` and therefore cannot take a second opinion on
    which antiunitary rows are legal.

    Parameters
    ----------
    zeta_ibz
        ``(n_q_ibz, n_mu, ngkmax)`` complex, sharded
        ``P(None, ('x','y'), None)``.
    gvec_components
        ``(n_q_ibz, 3, ngkmax)`` integer Miller vectors for ``zeta_ibz``.
        Required for the rank-3 literal-G=0 form.
    source_gvec_components
        Optional ``(n_q_ibz,3)`` source Miller vector when ``zeta_ibz`` is
        an already-selected rank-2 one-leg carrier.  This is used by the
        tied Coulomb-head columns: the service still owns every symmetry
        action and tau phase, while the producer owns which physical source
        column it selected.  Exactly one of this and ``gvec_components`` is
        used, selected by the rank of ``zeta_ibz``.
    sym
        :class:`SymMaps`; owns the q/star maps, reciprocal actions,
        translations and Cartesian rotations.
    sym_idx
        ``(n_q_full,)`` rows from ``QgridTrsPolicy.unfold_sym_idx``.
    sym_perm, L_table, q_irr_frac, kgrid
        The same centroid source-map, lattice-wrap and parent-q tables used
        by :func:`unfold_isdf_operator`; ``kgrid`` is explicit rather than
        invented as state on :class:`SymMaps`.
    component_action
        ``'scalar'`` or ``'polar'``.
    source_component
        Required for ``'polar'``; source Cartesian component in ``0..2``.

    Returns
    -------
    jax.Array
        Scalar: ``(n_q_full, n_mu)`` under ``P(None,'x')``.  Polar source
        contribution: ``(3, n_q_full, n_mu)`` under
        ``P(None,None,'x')``.
    """
    action = str(component_action).strip().lower()
    if action not in ("scalar", "polar"):
        raise ValueError(
            "unfold_isdf_one_leg: component_action must be 'scalar' or "
            f"'polar'; got {component_action!r}.")
    if action == "scalar":
        if source_component is not None:
            raise ValueError(
                "unfold_isdf_one_leg: source_component is only meaningful "
                "for component_action='polar'.")
        source = -1
    else:
        if source_component is None or int(source_component) not in (0, 1, 2):
            raise ValueError(
                "unfold_isdf_one_leg: polar action requires "
                "source_component in {0,1,2}.")
        source = int(source_component)

    zshape = tuple(int(v) for v in zeta_ibz.shape)
    preselected = len(zshape) == 2
    if len(zshape) not in (2, 3):
        raise ValueError(
            "unfold_isdf_one_leg: zeta_ibz must be a rank-3 G-sphere or "
            f"rank-2 preselected carrier; got {zshape}.")
    if preselected:
        if gvec_components is not None:
            raise ValueError(
                "unfold_isdf_one_leg: rank-2 preselected input takes "
                "source_gvec_components, not a G-sphere table.")
        source_gvec = np.asarray(source_gvec_components, dtype=np.int32)
        if source_gvec.shape != (zshape[0], 3):
            raise ValueError(
                "unfold_isdf_one_leg: source_gvec_components must have "
                f"shape ({zshape[0]},3); got {source_gvec.shape}.")
        gvec = None
        n_mu = zshape[1]
    else:
        if source_gvec_components is not None:
            raise ValueError(
                "unfold_isdf_one_leg: rank-3 G-sphere input determines its "
                "exact source G internally; do not pass source_gvec_components.")
        gvec = np.asarray(gvec_components, dtype=np.int32)
        if gvec.shape != (zshape[0], 3, zshape[2]):
            raise ValueError(
                "unfold_isdf_one_leg: gvec_components must have shape "
                f"({zshape[0]}, 3, {zshape[2]}); got {gvec.shape}.")
        source_gvec = None
        n_mu = zshape[1]

    q_full_int = np.asarray(sym.kvecs_asints, dtype=np.int64)
    grid = np.asarray(kgrid, dtype=np.int64).reshape(3)
    q_parent_int = np.asarray(sym.q_irr_kgrid_int, dtype=np.int64)
    n_q_full = int(np.prod(grid))
    if q_full_int.shape != (n_q_full, 3):
        raise ValueError(
            "unfold_isdf_one_leg: SymMaps.kvecs_asints disagrees with its "
            f"kgrid {tuple(grid)}; got {q_full_int.shape}.")
    idx = np.asarray(sym.irr_idx_q, dtype=np.int32)
    rows = np.asarray(sym_idx, dtype=np.int32)
    if idx.shape != (n_q_full,) or rows.shape != (n_q_full,):
        raise ValueError(
            "unfold_isdf_one_leg: SymMaps.irr_idx_q and policy sym_idx must "
            f"both have shape ({n_q_full},); got {idx.shape}, {rows.shape}.")
    if int(q_parent_int.shape[0]) != zshape[0]:
        raise ValueError(
            "unfold_isdf_one_leg: zeta q extent does not match the SymMaps "
            f"q-IBZ ({zshape[0]} != {int(q_parent_int.shape[0])}).")

    q_parent = np.asarray(q_irr_frac, dtype=np.float64)
    q_parent_expected = bgw_integer_q_to_fractional(q_parent_int, grid)
    if q_parent.shape != q_parent_expected.shape or not np.allclose(
            q_parent, q_parent_expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            "unfold_isdf_one_leg: q_irr_frac is not the BGW-wrapped "
            "SymMaps.q_irr_kgrid_int table; the zeta G labels and the star "
            "map would describe different parent momenta.")
    q_full = bgw_integer_q_to_fractional(q_full_int, grid)

    S_all = np.asarray(sym.sym_mats_k, dtype=np.int64)
    n_spatial = int(np.asarray(sym.sym_matrices).shape[0])
    if S_all.shape[0] != 2 * n_spatial:
        raise ValueError(
            "unfold_isdf_one_leg: SymMaps.sym_mats_k must contain spatial "
            "and TRS-augmented halves.")
    if np.any(rows < 0) or np.any(rows >= S_all.shape[0]):
        raise ValueError(
            "unfold_isdf_one_leg: policy sym_idx contains a row outside "
            f"[0,{S_all.shape[0]}).")

    perm = np.asarray(sym_perm, dtype=np.int32)
    wraps = np.asarray(L_table, dtype=np.float64)
    if perm.shape != (S_all.shape[0], n_mu):
        raise ValueError(
            "unfold_isdf_one_leg: sym_perm must cover every augmented row "
            f"and zeta centroid, expected {(S_all.shape[0], n_mu)}, "
            f"got {perm.shape}.")
    if wraps.shape != (S_all.shape[0], n_mu, 3):
        raise ValueError(
            "unfold_isdf_one_leg: L_table must have shape "
            f"{(S_all.shape[0], n_mu, 3)}; got {wraps.shape}.")

    # Literal target G=0 may be a nonzero parent G.  Build that exact
    # relabel once on the host from the service-owned star rows, then make
    # the device operation a pair of gathers plus phases.
    source_slot = np.empty(n_q_full, dtype=np.int32)
    tau_spatial = np.empty(n_q_full, dtype=np.complex128)
    translations = np.asarray(sym.translations, dtype=np.float64)
    for iq in range(n_q_full):
        s = int(rows[iq])
        p = int(idx[iq])
        S_full = S_all[s]
        S_inv = np.rint(np.linalg.inv(S_full)).astype(np.int64)
        if not np.array_equal(S_full @ S_inv, np.eye(3, dtype=np.int64)):
            raise ValueError(
                f"unfold_isdf_one_leg: symmetry row {s} is not unimodular.")
        if preselected:
            G_parent = source_gvec[p]
            source_slot[iq] = 0
            # A preselected carrier need not land at literal target G=0
            # (the tied Coulomb-head columns are an unordered invariant
            # set), but it must still land on an integer reciprocal label
            # at this target q.  Refuse a carrier/table mismatch rather than
            # silently attaching the transformed column to the wrong q row.
            G_target_f = S_full @ (q_parent[p] + G_parent) - q_full[iq]
            G_target = np.rint(G_target_f).astype(np.int32)
            if not np.allclose(
                    G_target_f, G_target, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "unfold_isdf_one_leg: preselected parent q+G does not "
                    f"map to an integer target G at full q {iq}: "
                    f"parent={p}, sym={s}, G_target={G_target_f.tolist()}.")
        else:
            K_parent = S_inv @ q_full[iq]
            G_parent_f = K_parent - q_parent[p]
            G_parent = np.rint(G_parent_f).astype(np.int32)
            if not np.allclose(G_parent_f, G_parent, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "unfold_isdf_one_leg: exact q/G relabel failed at full q "
                    f"{iq}: parent={p}, sym={s}, "
                    f"G_parent={G_parent_f.tolist()}.")
            if not np.allclose(S_full @ (q_parent[p] + G_parent), q_full[iq],
                               rtol=0.0, atol=1e-12):
                raise ValueError(
                    "unfold_isdf_one_leg: reconstructed parent q+G does not "
                    f"map to literal full-zone G=0 at q {iq}.")
            hits = np.flatnonzero(np.all(
                gvec[p].T == G_parent[None, :], axis=1))
            if hits.size != 1:
                raise ValueError(
                    "GATE isdf_one_leg_parent_g: literal full-zone G=0 at "
                    f"q={iq} requires parent q={p}, G={G_parent.tolist()}, "
                    f"but the stored zeta sphere contains {int(hits.size)} "
                    "exact matches.  Increase zeta_cutoff_ry if it is "
                    "missing; a one-leg coefficient cannot be reconstructed "
                    "from parent G=0 alone.")
            source_slot[iq] = int(hits[0])

        # Spatial tau phase first; the device conjugates this together with
        # zeta and the L phase on antiunitary rows.  This is the same ordering
        # as unfold_psi and makes exp[-i(S Kp).tnp] -> its conjugate there.
        s_spatial = s % n_spatial
        S_spatial = S_all[s_spatial]
        phase = tau_phase_row(
            S_spatial, translations[s_spatial],
            (q_parent[p] + G_parent)[None, :])
        tau_spatial[iq] = 1.0 if phase is None else phase[0]

    trs = rows >= n_spatial
    R_column = None
    if action == "polar":
        R_column = sym.cartesian_action(
            rows, axial=False, time_odd=True)[:, :, source]

    fn = _get_unfold_isdf_one_leg_jit(
        zeta_shape=zshape, idx=idx, rows=rows, perm=perm,
        trs=trs, preselected=preselected, action=action, mesh_xy=mesh_xy)
    slot_sh = NamedSharding(mesh_xy, P(None))
    tau_sh = NamedSharding(mesh_xy, P(None))
    q_parent_sh = NamedSharding(mesh_xy, P(None, None))
    wraps_sh = NamedSharding(mesh_xy, P(None, 'x', None))
    source_slot_dev = device_put_process_local(source_slot, slot_sh)
    tau_spatial_dev = device_put_process_local(tau_spatial, tau_sh)
    q_parent_dev = device_put_process_local(q_parent, q_parent_sh)
    wraps_dev = device_put_process_local(wraps, wraps_sh)
    if action == "scalar":
        return fn(
            zeta_ibz, source_slot_dev, tau_spatial_dev,
            q_parent_dev, wraps_dev)
    R_sh = NamedSharding(mesh_xy, P(None, None))
    return fn(
        zeta_ibz, source_slot_dev, tau_spatial_dev,
        q_parent_dev, wraps_dev,
        device_put_process_local(np.asarray(R_column), R_sh))


_UNFOLD_ISDF_ONE_LEG_JIT_CACHE: dict = {}


def _get_unfold_isdf_one_leg_jit(
    *, zeta_shape, idx, rows, perm, trs, preselected, action, mesh_xy,
):
    """Content-keyed one-leg action shared by all streamed source legs.

    Source G slots and tau phases are runtime operands on purpose: tied head
    columns differ only in those tables and must not create a separate
    compiled module per column.  The q/L tables are runtime operands for the
    same cache reason; their large ``(nq,nmu)`` phase is formed only inside
    this executable at its final sharding.
    """
    key = (
        tuple(zeta_shape), idx.tobytes(), rows.tobytes(),
        perm.shape, perm.tobytes(), trs.tobytes(), bool(preselected),
        str(action), id(mesh_xy),
    )
    hit = _UNFOLD_ISDF_ONE_LEG_JIT_CACHE.get(key)
    if hit is not None:
        return hit

    idx_j = jnp.asarray(idx)
    rows_j = jnp.asarray(rows)
    perm_j = jnp.asarray(perm)
    trs_j = jnp.asarray(trs)
    in_sh = NamedSharding(
        mesh_xy,
        P(None, 'x') if preselected else P(None, ('x', 'y'), None))
    scalar_sh = NamedSharding(mesh_xy, P(None, 'x'))
    polar_sh = NamedSharding(mesh_xy, P(None, None, 'x'))
    slot_sh = NamedSharding(mesh_xy, P(None))
    tau_sh = NamedSharding(mesh_xy, P(None))
    q_parent_sh = NamedSharding(mesh_xy, P(None, None))
    wraps_sh = NamedSharding(mesh_xy, P(None, 'x', None))

    def _spatial(
            zeta, source_slot_runtime, tau_runtime,
            q_parent_runtime, wraps_runtime):
        parent = zeta[idx_j]
        selected = (parent if preselected else jnp.take_along_axis(
            parent, source_slot_runtime[:, None, None], axis=2,
            mode='promise_in_bounds')[:, :, 0])
        gathered = jnp.take_along_axis(
            selected, perm_j[rows_j], axis=1,
            mode='promise_in_bounds')
        q_per_full = q_parent_runtime[idx_j]
        L_per_full = wraps_runtime[rows_j]
        qL = jnp.einsum('qi,qmi->qm', q_per_full, L_per_full)
        phase = tau_runtime[:, None] * jnp.exp(-2j * jnp.pi * qL)
        phase = jax.lax.with_sharding_constraint(phase, scalar_sh)
        return phase * gathered

    if action == "scalar":
        @partial(
            jax.jit,
            in_shardings=(
                in_sh, slot_sh, tau_sh, q_parent_sh, wraps_sh),
            out_shardings=scalar_sh)
        def _do_unfold(
                zeta, source_slot_runtime, tau_runtime,
                q_parent_runtime, wraps_runtime):
            spatial = _spatial(
                zeta, source_slot_runtime, tau_runtime,
                q_parent_runtime, wraps_runtime)
            return jnp.where(trs_j[:, None], jnp.conj(spatial), spatial)
    else:
        R_sh = NamedSharding(mesh_xy, P(None, None))

        @partial(jax.jit,
                 in_shardings=(
                     in_sh, slot_sh, tau_sh, q_parent_sh, wraps_sh, R_sh),
                 out_shardings=polar_sh)
        def _do_unfold(
                zeta, source_slot_runtime, tau_runtime,
                q_parent_runtime, wraps_runtime, R_column_runtime):
            spatial = _spatial(
                zeta, source_slot_runtime, tau_runtime,
                q_parent_runtime, wraps_runtime)
            scalar = jnp.where(
                trs_j[:, None], jnp.conj(spatial), spatial)
            return jnp.einsum('qi,qm->iqm', R_column_runtime, scalar)

    _UNFOLD_ISDF_ONE_LEG_JIT_CACHE[key] = _do_unfold
    return _do_unfold


def mix_channels_by_proper_rotation(
    V_tt_per_channel,
    *,
    sym,
    sym_idx,
    mesh_xy,
):
    """Mix the two Pauli-vector indices on bispinor TT tiles.

    Each index requests the canonical axial, time-odd action from ``sym``.
    The two antiunitary signs cancel while :func:`unfold_isdf_operator` owns
    the single complex conjugation.  Keeping the action typed here prevents
    callers from choosing a transpose, determinant sign, or TR convention.

    Parameters
    ----------
    V_tt_per_channel : dict[(i, j) -> jax.Array]
        Dict keyed by ``(i, j)`` with ``i, j ∈ {1, 2, 3}`` (9 entries).
        Each value is ``(n_q_full, μ, ν)`` complex128 already at full-BZ
        shape, sharded ``P(None, 'x', 'y')``.  Callers may pass the 6
        unique tiles + the 3 Hermitian-redundant tiles synthesised via
        ``conj(swapaxes(V[i,j], -1, -2))``.
    sym_idx : numpy.ndarray
        Host ``(n_q_full,)`` integer metadata from ``SymMaps.sym_idx_q``.
        Device arrays and tracers are refused before the compiled tile path.
    sym : SymMaps
        Canonical operation and representation source.
    mesh_xy : jax.sharding.Mesh
        Device mesh (used to lock the output sharding).

    Returns
    -------
    dict[(i, j) -> jax.Array]
        Same keys, same shapes, sharded ``P(None, 'x', 'y')``.
    """
    if isinstance(sym_idx, (jax.Array, jax.core.Tracer)):
        raise TypeError(
            "mix_channels_by_proper_rotation: sym_idx is host metadata; "
            "pass a NumPy array, not a JAX array or tracer.")
    sym_raw = np.asarray(sym_idx)
    if sym_raw.ndim != 1:
        raise ValueError(
            "mix_channels_by_proper_rotation: sym_idx must be rank one; "
            f"got {sym_raw.shape}.")
    R_per_q = np.asarray(sym.cartesian_action(
        sym_raw, axial=True, time_odd=True), dtype=np.float64)
    if R_per_q.shape != (sym_raw.size, 3, 3):
        raise ValueError(
            "mix_channels_by_proper_rotation: typed Cartesian actions must "
            f"have shape {(sym_raw.size, 3, 3)}; got {R_per_q.shape}.")

    # Build the 9×9 source array V_in[α, β] at full-BZ shape, contract,
    # write back into the same 6 unique slots (plus 3 redundants).
    keys_in = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    for k in keys_in:
        if k not in V_tt_per_channel:
            raise ValueError(
                f"mix_channels_by_proper_rotation: missing TT tile {k}; "
                f"caller must supply all 9 (i, j) ∈ {{1,2,3}}² "
                f"(use ``conj(swapaxes(.., -1, -2))`` to synthesise the "
                f"Hermitian-redundant entries).")
    sample = V_tt_per_channel[(1, 1)]
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    fn = _get_mix_channels_jit(
        V_shape=tuple(int(s) for s in sample.shape),
        R_per_q_arr=R_per_q,
        mesh_xy=mesh_xy,
    )
    # Stack input tiles in (α, β, q, μ, ν) layout; contract via einsum
    # and unstack into the output dict.
    V_in = jnp.stack(
        [jnp.stack([V_tt_per_channel[(a, b)] for b in (1, 2, 3)], axis=0)
         for a in (1, 2, 3)],
        axis=0,
    )                                                       # (3, 3, n_q, μ, ν)
    V_out = fn(V_in)                                        # (3, 3, n_q, μ, ν)
    return {
        (i, j): jax.lax.with_sharding_constraint(V_out[i - 1, j - 1], V_sh)
        for i in (1, 2, 3) for j in (1, 2, 3)
    }


_MIX_CHANNELS_JIT_CACHE: dict = {}


def _get_mix_channels_jit(*, V_shape, R_per_q_arr, mesh_xy):
    """Cache the inner Lorentz-mix jit by (shape, R-table content).

    Same content-keyed caching strategy as
    :func:`_get_unfold_isdf_operator_jit`: the R table is baked into the jit
    closure as a constant so XLA can fold it into the HLO.  The cache key is
    the bytes-hash of the table plus the V shape plus the mesh identity.
    """
    key = (
        V_shape,
        R_per_q_arr.shape, R_per_q_arr.tobytes(),
        id(mesh_xy),
    )
    hit = _MIX_CHANNELS_JIT_CACHE.get(key)
    if hit is not None:
        return hit

    R_per_q_j = jnp.asarray(R_per_q_arr)                    # (n_q_full, 3, 3)
    V_sh = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
    in_sh = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

    @partial(jax.jit, in_shardings=in_sh, out_shardings=V_sh)
    def _do_mix(V_in):
        # V_in: (3, 3, n_q, μ, ν); R is the forward action [q,out,in].
        return jnp.einsum(
            'qia,qjb,abqmn->ijqmn',
            R_per_q_j, R_per_q_j, V_in,
        )

    _MIX_CHANNELS_JIT_CACHE[key] = _do_mix
    return _do_mix


# i·σ_y (time-reversal spinor factor in the SOC convention T = iσ_y K).
_I_SIGMA_Y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)


def spinor_rotation_for_sym_row(U_spinor_spatial, sym_idx, n_tran, *,
                                nspinor=2):
    """Per-op spinor rotation matrix with the TRS augmentation baked in.

    Single source of the ψ-unfold spinor rule (see :func:`unfold_psi`).
    For a spatial sym row (``sym_idx < n_tran``) the spinor rotation is just
    ``U_spinor_spatial[sym_idx]``.  For a TRS-augmented row
    (``sym_idx >= n_tran``, op ``T∘{S|τ}`` with ``T = iσ_y K``) it is
    ``iσ_y · conj(U_spinor_spatial[sym_idx − n_tran])``.

    Works for a scalar ``sym_idx`` (host per-single-k path in
    :func:`unfold_psi`) or a 1-D array of ``sym_idx`` (device per-full-BZ-k
    table build in ``WfnLoader._ensure_phdf5_static``).

    NON-SOC IS A DIFFERENT REPRESENTATION, NOT A DEGENERATE CASE OF THIS
    ONE.  Everything above is the spin-1/2 (SU(2)) rule.  A scalar
    wavefunction — ``nspinor = 1``, non-SOC — carries no spinor index at
    all: a spatial op acts on it through the τ-phase and the G-relabel
    alone, and its time reversal is ``Θ = K``, plain conjugation with
    ``Θ² = +1``, NOT ``iσ_y K`` with ``Θ² = −1``.  So the effective factor
    at ``nspinor = 1`` is the 1×1 identity on BOTH row kinds, spatial and
    time-reversed, and this function returns exactly that.

    RETURNING THE 2×2 ANYWAY IS NOT A HARMLESS OVER-APPROXIMATION.  Both
    consumers contract this matrix against ψ's spinor axis, and both
    ``numpy.einsum`` and ``jax.numpy.einsum`` BROADCAST a size-1 labelled
    axis instead of raising.  A 2×2 fed a 1-component ψ therefore returns
    a 2-component ψ holding ``(U[j,0] + U[j,1]) · ψ`` — wrong in value and
    wrong in shape, and on the TRS rows it is ``iσ_y·conj(U)``'s two
    off-diagonal entries doing the summing.  That is the registered
    nspinor=1 loader defect (``tests/KNOWN_FAILURES.md``;
    ``tests/regression/hbn_cohsex_debug/README.md``), fixed here on
    2026-08-09.  ``nspinor`` defaults to 2 because that is what this
    function was before the parameter existed and what every deck in this
    tree is — it is the back-compatible spelling, not a claim that 2 is
    the safe guess when the caller does not know.

    Parameters
    ----------
    U_spinor_spatial : (n_tran, 2, 2) complex
        Spatial-only spinor rotation matrices.  Ignored (they are SU(2)
        objects with nothing to say about a scalar field) when
        ``nspinor == 1``.
    sym_idx : int or (nk,) int array
        Row(s) in the TRS-augmented row space ``[0, 2·n_tran)``.
    n_tran : int
        Count of spatial-only sym ops.
    nspinor : int, keyword-only, default 2
        Spinor components ψ actually has.  ``2`` for SOC/bispinor decks,
        ``1`` for scalar (non-SOC) wavefunctions.  Anything else is a
        raise: the 4-component bispinor lift happens downstream of the
        unfold (``WfnLoader.load(bispinor=True)``) and never reaches here.

    Returns
    -------
    (ns, ns) or (nk, ns, ns) complex
        Effective spinor rotation(s), ``ns = nspinor``.  Scalar
        ``sym_idx`` → ``(ns, ns)``.
    """
    nspinor = int(nspinor)
    if nspinor not in (1, 2):
        raise ValueError(
            f"spinor_rotation_for_sym_row: nspinor must be 1 (scalar/non-SOC) "
            f"or 2 (SOC); got {nspinor}.  The 4-component bispinor lift is "
            f"applied downstream of the unfold and does not come through "
            f"here.")
    U_spatial = np.asarray(U_spinor_spatial)
    idx = np.asarray(sym_idx)
    scalar = idx.ndim == 0
    idx1 = np.atleast_1d(idx)
    if nspinor == 1:
        # Scalar ψ: one factor, and it is 1, for spatial AND TRS rows
        # alike (Θ = K carries no iσ_y).  Shaped (1, 1) / (nk, 1, 1) so the
        # consumers' static application contracts a matching axis and is a
        # true no-op rather than a silent broadcast.
        out = np.ones((idx1.size, 1, 1), dtype=np.complex128)
        return out[0] if scalar else out
    # ``% n_tran`` folds a TRS row (idx ≥ n_tran) back to its spatial op;
    # guard the degenerate no-symmetry case (n_tran == 0, single identity
    # row) so it doesn't divide-by-zero — only idx == 0 is valid there and
    # both spellings pick spatial op 0.
    s_spatial = idx1 % (n_tran if n_tran else 1)
    is_trs = idx1 >= n_tran
    U_sp = U_spatial[s_spatial]                                    # (nk, 2, 2)
    # iσ_y · conj(U_spatial[s_spatial]), broadcast over the leading axis.
    U_trs = np.einsum('ij,kjl->kil', _I_SIGMA_Y, np.conj(U_sp))    # (nk, 2, 2)
    out = np.where(is_trs[:, None, None], U_trs, U_sp)
    return out[0] if scalar else out


def apply_spinor_rotation(U, coeff_last):
    """Apply a scalar or Pauli-spinor rotation without a general GEMM.

    ``spinor_rotation_for_sym_row`` owns the physical matrix; this function
    owns its application to wavefunction coefficients.  ``U`` has shape
    ``(..., a, c)`` and ``coeff_last`` has shape ``(..., c)``.  Their leading
    dimensions follow ordinary broadcasting, so a caller normalizes its own
    physical layout once and this service sees only the spinor-last algebra.

    The spinor extent is static and must be one or two.  Writing the Pauli
    case as two explicit two-term multiply-adds is deliberate: lowering the
    old ``einsum`` as a general K=2 cuBLAS GEMM made the production PHDF5
    unfold shape un-compilable under XLA's sharding autotuner.  This spelling
    is the exact same linear action without asking a matrix-multiply backend
    to choose a GEMM algorithm for a fixed two-component representation.

    NumPy inputs return a NumPy array; JAX arrays/tracers remain in JAX.  This
    keeps the eager host unfold host-only while giving the collective loader
    the same semantic owner inside ``jit``/``shard_map``.
    """
    u_shape = np.shape(U)
    coeff_shape = np.shape(coeff_last)
    if len(u_shape) < 2 or len(coeff_shape) < 1:
        raise ValueError(
            "apply_spinor_rotation: expected U(...,a,c) and "
            f"coeff_last(...,c); got shapes {u_shape} and {coeff_shape}")
    ns = int(coeff_shape[-1])
    if ns not in (1, 2):
        raise ValueError(
            "apply_spinor_rotation: spinor extent must be 1 (scalar/non-SOC) "
            f"or 2 (SOC); got {ns}")
    if tuple(int(x) for x in u_shape[-2:]) != (ns, ns):
        raise ValueError(
            "apply_spinor_rotation: U's final axes must match the "
            f"coefficient spinor extent; got U{u_shape[-2:]} vs ns={ns}")

    is_jax = isinstance(U, (jax.Array, jax.core.Tracer)) or isinstance(
        coeff_last, (jax.Array, jax.core.Tracer))
    xp = jnp if is_jax else np
    U_x = xp.asarray(U)
    coeff_x = xp.asarray(coeff_last)
    if ns == 1:
        return xp.expand_dims(U_x[..., 0, 0] * coeff_x[..., 0], axis=-1)
    out0 = (U_x[..., 0, 0] * coeff_x[..., 0]
            + U_x[..., 0, 1] * coeff_x[..., 1])
    out1 = (U_x[..., 1, 0] * coeff_x[..., 0]
            + U_x[..., 1, 1] * coeff_x[..., 1])
    return xp.stack((out0, out1), axis=-1)


def tau_phase_row(sym_mat_k, tau, g_kbar):
    """τ-phase ``exp(-i (S·K_parent)·τ)`` for reciprocal carriers.

    Single source of the reciprocal-space translation phase shared by
    :func:`unfold_psi` and :func:`unfold_isdf_one_leg`.
    ``sym_mat_k`` is the TRS-augmented sym matrix ``sym_mats_k[sym_idx]`` —
    for TRS rows it already carries the ``-S`` sign, so the same formula
    yields ``exp(+i (S·G_kbar)·τ) = conj(spatial-phase)`` automatically.

    Returns ``None`` when ``τ ≈ 0`` (the phase is identically 1 and callers
    skip the multiply), matching both the host and device table builds.

    Parameters
    ----------
    sym_mat_k : (3, 3) int
        ``sym_mats_k[sym_idx]`` (TRS-augmented; carries the ±S sign).
    tau : (3,) float
        Spatial fractional translation ``translations[sym_idx % n_tran]``.
    g_kbar : (ncarrier, 3) int or float
        Parent reciprocal-coordinate carriers.  Wavefunction unfolding passes
        its integer G list; an ISDF one-leg action passes q+G.

    Returns
    -------
    (ngk,) complex or None
        The per-G phase, or ``None`` when ``τ`` is (numerically) zero.
    """
    tau = np.asarray(tau, dtype=np.float64)
    if not np.any(np.abs(tau) > 1e-12):
        return None
    S = np.asarray(sym_mat_k, dtype=np.int32)
    rotated = (S @ np.asarray(g_kbar).T).T.astype(np.float64)      # (ngk, 3)
    return np.exp(-1j * (rotated @ tau))                          # (ngk,)


def tau_phase_row_jax(sym_mat_k, tau, g_kbar):
    """Device form of :func:`tau_phase_row`, including the identity case.

    This owns the same ``exp(-i (S G) . tau)`` convention but is intended
    for fusion into a larger JAX action.  Unlike the host helper it returns
    an explicit all-one row at zero translation; avoiding that allocation is
    then the compiler's job, not a Python branch on traced data.
    """
    S = jnp.asarray(sym_mat_k, dtype=jnp.int32)
    tau_j = jnp.asarray(tau, dtype=jnp.float64)
    g = jnp.asarray(g_kbar)
    # Contract the three-vector first.  Forming ``S @ g.T`` explicitly
    # would create an avoidable ``(ngk, 3)`` temporary exactly where the
    # streamed loader must remain usable for very large G spheres.
    rotated_tau = jnp.einsum('ij,i->j', S, tau_j, optimize=True)
    return jnp.exp(-1j * jnp.einsum(
        'gj,j->g', g, rotated_tau, optimize=True))


def unfold_reciprocal_carriers(sym_mat_k, g_parent, umklapp):
    """Map parent reciprocal carriers to one full-zone child.

    Implements ``G_child = S G_parent - G_umklapp`` for NumPy or JAX
    operands. This is the shared algebra used by the host G-table builder and
    device-side parent/bispinor realization.
    """
    is_jax = any(isinstance(value, (jax.Array, jax.core.Tracer)) for value in (
        sym_mat_k, g_parent, umklapp))
    xp = jnp if is_jax else np
    S = xp.asarray(sym_mat_k)
    g = xp.asarray(g_parent)
    shift = xp.asarray(umklapp)
    return xp.einsum('ij,gj->gi', S, g, optimize=True) - shift[None, :]


def unfold_psi(
    cnk_kbar,
    *,
    sym_idx,
    g_kbar,
    sym_mats_k,
    translations,
    U_spinor_spatial,
):
    """ψ at one full-BZ k from ψ at its IBZ representative ``kbar``.

    Pure-numpy / host-side. Handles spatial AND TRS-augmented sym rows;
    the bispinor TRS rule lives here (and ONLY here in PR3+).

    Math:
        For spatial sym (sym_idx < n_sym_spatial, op {S|τ}, S = sym_mats_k[sym_idx]):
            ψ_full(G_rot) = exp(-i (S·G_kbar)·τ) · U_spinor(S) · ψ_kbar(G_kbar)
            where G_rot = S·G_kbar + kg0 (umklapp; caller handles G_rot bookkeeping
            via WfnLoader.gvecs and friends — this helper only computes the
            spinor + phase factors).

        For TRS-augmented sym (sym_idx ≥ n_sym_spatial, op T∘{S|τ}, T = iσ_y K):
            ψ_full(G_rot) = (iσ_y · conj(U_spinor(S)))
                            · exp(+i (S·G_kbar)·τ)
                            · conj(ψ_kbar(G_kbar))
        Equivalently: ψ_full = iσ_y · conj(spatial-form), per the per-element
        derivation in ``reports/trs_sym_audit_2026-05-14/pr3_design.md``.

        WHY THE G-LIST IS NEGATED, AND WHY THAT IS HALF OF THE RULE.
        Θ = iσ_y K is ANTIUNITARY. Acting on ψ_nk(r) = Σ_G c(G) e^{i(k+G)·r}:

            (Θψ_nk)(r) = iσ_y ψ*_nk(r) = Σ_G [iσ_y c*(G)] e^{−i(k+G)·r}
                       = Σ_{G'} [iσ_y c*(−G')] e^{i(−k+G')·r}   (G' = −G)

            ⇒  c_{Θ,−k}(G') = iσ_y · conj( c(−G') ).                    (★)

        (★) has TWO halves: the spinor factor ``iσ_y·conj`` (applied HERE)
        and the negation of the G list (applied by the CALLER, because
        ``sym_mats_k[sym_idx] = −S`` for a TRS row, so
        ``WfnLoader.gvecs(k='full_bz')`` emits ``−S·G_kbar − kg0``).
        Applying one half without the other replaces ψ(r) by ψ*(−r) —
        norm-, orthogonality- and ⟨T⟩-preserving, hence invisible to every
        cheap check, and wrong by O(100 eV) in V_loc/V_NL. That is exactly
        the scorecard §Q bug, and it is why the length guard below is a
        hard raise rather than a warning: the ONLY thing that keeps the two
        halves in step is ``len(sym_mats_k) == 2·len(U_spinor_spatial)``.

        NON-SOC (ns = 1) IS A DIFFERENT REPRESENTATION, NOT A SPECIAL
        CASE. A scalar wavefunction has no spinor index, so the spatial
        rule loses its U:

            ψ_full(G_rot) = exp(-i (S·G_kbar)·τ) · ψ_kbar(G_kbar)

        and its time reversal is Θ = K — plain conjugation, Θ² = +1, no
        Kramers pair to protect — so the TRS rule loses its iσ_y too:

            ψ_full(G_rot) = exp(+i (S·G_kbar)·τ) · conj(ψ_kbar(G_kbar))

        The G-LIST NEGATION half of (★) is unchanged: it follows from Θ
        being antiunitary, which has nothing to do with spin. Only the
        spinor half disappears. Both lines are what this function already
        computes once ``spinor_rotation_for_sym_row`` is told ``ns = 1``
        and hands back the 1×1 identity instead of a 2×2 — see the
        service application below and that helper's own docstring for the
        defect this replaced (registered 2026-08-08, fixed 2026-08-09).

        Note that ns = 1 does NOT switch the TRS rows off. The automatic
        DFT-reference check is deliberately 2c-only, so scalar decks retain
        the historical permissive setting unless a caller supplies an
        explicit ``trs_holds`` verdict. The ns=1 TRS branch above therefore
        remains live code.

        NON-SYMMORPHIC τ UNDER TRS. ``tau_phase_row`` is fed ``S_full``
        (= −S on a TRS row), so ``exp(−i (−S·G)·τ) = exp(+i (S·G)·τ)`` —
        the conjugate of the spatial phase — which is what (★) demands
        since the whole spatial expression is conjugated. There is no
        separate τ for TRS rows and none is needed; ``translations`` is
        indexed by ``s_spatial``. Verified end-to-end on the genuinely
        non-symmorphic ``si_cohsex_debug`` deck (tnp = π ⇒ τ_frac = 1/2).

        INDEPENDENT MEASUREMENT. Whether TRS holds AT ALL for a given file
        is no longer inferred from ``ntran``/k-weights: it is measured from
        the two-component DFT reference by ``density_symmetry_check``.  The
        check compares occupied density operators using raw partners,
        spatial-only partners, or TRIM closure; it never accepts a state
        generated by this antiunitary branch as evidence.

        Implementation note: ``sym_mats_k[sym_idx]`` already encodes the
        ±S sign (TRS rows are ``-S``). Computing
        ``rotated = sym_mats_k[sym_idx] @ G_kbar`` and then
        ``exp(-i rotated·τ)`` gives ``exp(+i S·G_kbar · τ)`` automatically
        for TRS rows — no separate sign branch on the phase. Order:
        apply phase AFTER conj on the TRS branch so the conj doesn't
        invert the phase sign.

    Parameters
    ----------
    cnk_kbar : (nb, ns, ngk) complex
        IBZ ψ coefficients on the IBZ G-list. ``ns`` is the spinor axis;
        ``ns = 1`` for non-SOC, ``ns = 2`` for SOC.  ``ns`` is read off THIS
        array and handed to :func:`spinor_rotation_for_sym_row`, which is
        what makes the ``ns = 1`` spinor factor a genuine 1×1 identity (see
        NON-SOC in Math above) instead of a 2×2 broadcast against a size-1
        axis.
    sym_idx : int
        Row in ``sym_mats_k`` (length ``2·n_sym_spatial``).
    n_sym_spatial : int
        Count of spatial-only sym ops (= wfn.ntran). TRS rows are
        ``[n_sym_spatial, 2·n_sym_spatial)``.
    g_kbar : (ngk, 3) int
        IBZ G-list (ψ_kbar's G axis).
    sym_mats_k : (2·n_sym_spatial, 3, 3) int
        TRS-augmented sym matrices acting on k/q (and G).
    translations : (n_sym_spatial, 3) float
        BGW fractional translations τ_s. Length ``n_sym_spatial`` — TRS rows
        do not have a separate τ; they reuse the spatial τ with the right sign
        baked into the formula above.
    U_spinor_spatial : (n_sym_spatial, 2, 2) complex
        Spatial-only spinor rotation matrices. The TRS-row spinor is computed
        inside this helper as ``iσ_y · conj(U_spinor_spatial[s])``.

    Returns
    -------
    cnk_full : (nb, ns, ngk) complex
        ψ at the full-BZ k, returned on the IBZ G-axis (i.e. cnk_full[b, σ, g]
        corresponds to the G-vector ``sym_mats_k[sym_idx] @ g_kbar[g]`` in the
        full-k basis). The caller's G-rebuild (``WfnLoader.gvecs``) and umklapp
        handling are independent.
    """
    sym_idx = int(sym_idx)
    # ``sym_mats_k`` always has length ``2 · n_sym_spatial`` — the spatial
    # half is followed by the TRS-augmented rows (-S).
    if int(np.shape(sym_mats_k)[0]) != 2 * int(np.shape(U_spinor_spatial)[0]):
        raise ValueError(
            "unfold_psi: sym_mats_k must be TRS-augmented to length "
            f"2*n_sym_spatial; got len(sym_mats_k)="
            f"{int(np.shape(sym_mats_k)[0])} vs len(U_spinor_spatial)="
            f"{int(np.shape(U_spinor_spatial)[0])}.  A non-augmented table "
            "silently reclassifies the identity row as time reversal "
            "(returns iσ_y·conj(ψ) on an un-negated G-list).")
    n_sym_spatial = int(sym_mats_k.shape[0]) // 2
    is_trs = sym_idx >= n_sym_spatial
    s_spatial = sym_idx - n_sym_spatial if is_trs else sym_idx

    S_full = np.asarray(sym_mats_k[sym_idx], dtype=np.int32)
    cnk = np.asarray(cnk_kbar)
    g_bar = np.asarray(g_kbar)

    # τ-phase: single-sourced in ``tau_phase_row``. ``S_full`` already has
    # the ±S sign baked in for TRS rows, so the same formula yields
    # ``conj(spatial-phase)`` there. Returns ``None`` when τ ≈ 0.
    phase = tau_phase_row(S_full, translations[s_spatial], g_bar)

    if is_trs:
        # TRS rule: ψ_full = iσ_y · conj(U_s · ψ_kbar · phase_spatial)
        #         = (iσ_y · conj(U_s)) · conj(ψ_kbar) · conj(phase_spatial)
        # ``phase`` is computed via sym_mats_k[TRS row]=-S so it equals
        # conj(phase_spatial) already. Apply conj on cnk first, THEN phase
        # (else the conj would re-invert the phase sign).
        cnk = np.conj(cnk)
        if phase is not None:
            cnk = cnk * phase[None, None, :]
    else:
        if phase is not None:
            cnk = cnk * phase[None, None, :]
    # Spinor rotation with the TRS augmentation single-sourced.  ``ns`` is
    # READ OFF THE DATA rather than taken from a parameter: it is the axis
    # the static service application below contracts, so the array itself is
    # the only source that cannot disagree with what is about to be applied.
    ns = int(np.shape(cnk)[1])
    U_eff = spinor_rotation_for_sym_row(
        U_spinor_spatial, sym_idx, n_sym_spatial, nspinor=ns)

    # Spinor rotation.  For ns=1 (non-SOC) ``U_eff`` genuinely IS the 1x1
    # identity — the helper is told ``ns`` and returns it — so this
    # application is a true no-op and callers still need no special case.
    # Before
    # 2026-08-09 that sentence stood here and was FALSE: the helper
    # returned the 2x2 unconditionally, numpy broadcast the size-1 spinor
    # axis instead of raising, and a scalar WFN came back (nb, 2, ngk).
    #
    # The guard is the anti-regression, not decoration: the former einsum
    # silently broadcast this mismatch and only surfaced downstream as a
    # slab-write ValueError with no mention of spinors, which is exactly how
    # the original defect presented.
    if int(np.shape(U_eff)[-1]) != ns:
        raise ValueError(
            f"unfold_psi: spinor factor is {np.shape(U_eff)[-1]}x"
            f"{np.shape(U_eff)[-1]} but psi has ns={ns} spinor components. "
            f"einsum would BROADCAST the size-1 axis rather than raise, "
            f"returning sum_k U[j,k]*psi[n,0,l] on an ns-wrong output "
            f"shape.  See spinor_rotation_for_sym_row's nspinor argument.")
    cnk = np.moveaxis(
        apply_spinor_rotation(U_eff, np.moveaxis(cnk, 1, -1)), -1, 1)
    return cnk


class SymMaps:
    def __init__(self, wfn, *, allow_trs=None):
        """
        Initialize symmetry mappings for a given WFN file.
        class variables:
        - irr_idx_k[ik_full] = IBZ index in wfn.kpoints for each full-BZ k
        - sym_idx_k[ik_full] = sym_mats_k row mapping wfn.kpoints[irr_idx_k[ik_full]] → unfolded_kpts[ik_full]
        - irr_idx_q[iq_full] = IBZ index in q_irr_kgrid_int for each full-BZ q
        - sym_idx_q[iq_full] = sym_mats_k row mapping q_irr_kgrid_int[irr_idx_q[iq_full]] → kvecs_asints[iq_full]
        - q_irr_kgrid_int[i_irr] = IBZ q in integer kgrid coords (lex-min representatives)
        - q_irr_full_idx[i_irr] = full-BZ row index where this IBZ q lives in kvecs_asints
        U_spinor[sym_idx] is the spinor rotation matrix for the sym_idx-th symmetry operation.
        The matrices are currently 2x2 Pauli-spinor rotations; upcoming work
        will expand this to the 4-component formalism used in relativistic
        treatments.
        R_grid[sym_idx] is the corresponding list of symmetry operations in the WFN file
        u_{n,Rk,a}(G) = U_spinor_{a,b} u_{n,k,b}(Rinv G)
        
        Args:
            wfn: WFNReader instance
            allow_trs: whether time-reversal rows of ``sym_mats_k`` may be
                SELECTED when mapping the full BZ onto the IBZ.  ``None``
                (default) takes the value from ``wfn.trs_holds`` — the
                verdict that ``WfnLoader`` obtains from the occupied
                two-component DFT subspaces (``density_symmetry_check``) —
                falling back to
                ``True`` (the historical, permissive behaviour) for
                wfn-shaped objects that carry no verdict.

                When False, arbitrary global-TR partners are disabled.  An
                authenticated QE schema may still authorize individual
                antiunitary magnetic-space-group operations; without that
                receipt the search conservatively keeps only the WFN
                header's presumed-unitary half.  ``sym_mats_k`` always keeps
                its ``2·ntran`` candidate layout because wavefunction and
                nonsymmorphic-phase consumers key conjugation from the row.
        """
        # Measured-TRS gate.  ``allow_trs=None`` → consult the wfn object;
        # objects with no verdict (legacy WFNReader, hand-built stubs) get
        # the permissive default so this is a pure no-op for them.
        if allow_trs is None:
            allow_trs = getattr(wfn, 'trs_holds', None)
        self.trs_allowed = True if allow_trs is None else bool(allow_trs)

        # WFN.h5 omits QE's per-operation antiunitary bit.  WfnLoader
        # attaches a receipt only after a nearby QE schema has matched the
        # WFN's active Seitz rows and stored k coordinates.  The legacy
        # fallback remains available, but its epistemic limit is announced
        # at the exact seam where symmetry becomes executable.
        self.qe_symmetry_binding = getattr(wfn, "qe_symmetry_binding", None)
        self.qe_symmetry_diagnostic = str(getattr(
            wfn, "qe_symmetry_diagnostic",
            "the WFN-shaped input carries no QE schema receipt"))
        self.operation_typing_source = (
            "qe-schema" if self.qe_symmetry_binding is not None
            else "wfn-fallback")
        if self.qe_symmetry_binding is None:
            import warnings as _warnings
            _warnings.warn(
                "SYMMETRY PROVENANCE WARNING: no authenticated QE "
                "data-file-schema.xml is available at SymMaps "
                f"initialization ({self.qe_symmetry_diagnostic}). WFN.h5 "
                "does not record which operations are composed with time "
                "reversal. LORRAX is using the legacy all-spatial header "
                "interpretation plus the global DFT-reference TRS verdict. "
                "RESULTS WILL BE WRONG if time reversal is broken and QE "
                "used an antiunitary magnetic-space-group operation. "
                "Co-stage the WFN-generating QE *.save directory or pass "
                "WfnLoader(..., qe_schema=...).",
                RuntimeWarning)

        # get symmetry matrices from wfn file
        try:
            ntran = int(getattr(wfn, 'ntran', 1))
        except Exception:
            ntran = 1
        if ntran < 1:
            raise ValueError(
                f"SymMaps: WFN ntran must be positive, got {ntran}.")
        if self.qe_symmetry_binding is None:
            _qe_antiunitary = np.zeros(ntran, dtype=bool)
            _qe_base_rows = np.arange(ntran, dtype=np.int32)
            self.qe_permitted_pure_time_reversal = None
            self.qe_schema_path = None
            self.qe_schema_sha256 = None
        else:
            _qe_antiunitary = np.asarray(
                self.qe_symmetry_binding.antiunitary, dtype=bool)
            if _qe_antiunitary.shape != (ntran,):
                raise ValueError(
                    "SymMaps: authenticated QE operation typing has shape "
                    f"{_qe_antiunitary.shape}, expected ({ntran},).")
            _qe_base_rows = (
                np.arange(ntran, dtype=np.int32)
                + _qe_antiunitary.astype(np.int32) * ntran)
            self.qe_permitted_pure_time_reversal = bool(
                self.qe_symmetry_binding.qe_permitted_pure_time_reversal)
            self.qe_schema_path = str(self.qe_symmetry_binding.schema_path)
            self.qe_schema_sha256 = str(
                self.qe_symmetry_binding.schema_sha256)
        self.qe_operation_antiunitary = _qe_antiunitary.copy()
        self.qe_antiunitary_rows = _qe_base_rows[_qe_antiunitary].copy()
        # Candidate row layout remains [S, -S] everywhere.  This array is
        # the one authority for which candidates are physically permitted.
        self.active_symmetry_rows = (
            np.arange(2 * ntran, dtype=np.int32)
            if self.trs_allowed else _qe_base_rows.copy())
        _kgrid = np.asarray(wfn.kgrid, dtype=np.int64)
        if _kgrid.shape != (3,) or np.any(_kgrid <= 0):
            raise ValueError(
                "SymMaps: WFN kgrid must contain three positive extents; "
                f"got {_kgrid.tolist()}.")
        _nfull_declared = int(np.prod(_kgrid, dtype=np.int64))
        _nk_stored = int(getattr(wfn, "nkpts", len(wfn.kpoints)))
        if _nk_stored > _nfull_declared:
            raise ValueError(
                f"SymMaps: WFN stores {_nk_stored} k-points but its kgrid "
                f"declares only {_nfull_declared} full-BZ rows.")

        # ``ntran=1`` says only that identity is the sole stored Seitz
        # operation.  It does not say the file stores the full BZ: a
        # nonmagnetic WFN may
        # still contain a TR-reduced half mesh which needs the synthesized
        # ``-I`` row below.  Keep the fast identity path only when every
        # declared grid point is genuinely present and the active row is I.
        _identity_full_grid = (ntran == 1 and
                               _nk_stored == _nfull_declared)
        if _identity_full_grid:
            _stored_op = np.asarray(wfn.sym_matrices[0], dtype=np.int64)
            if not np.array_equal(_stored_op, np.eye(3, dtype=np.int64)):
                raise ValueError(
                    "SymMaps: a full-grid ntran=1 WFN must store identity as "
                    f"its active symmetry row; got {_stored_op.tolist()}.")
            _declared_grid = self._generate_uniform_full_kpoints(wfn)
            _, _, _full_present = map_full_kpoints_to_irreducible(
                wfn.kpoints, np.eye(3, dtype=np.int64)[None, :, :],
                _declared_grid)
            if not np.all(_full_present):
                _bad = np.where(~_full_present)[0]
                raise ValueError(
                    f"SymMaps: ntran=1 WFN has nrk=product(kgrid)="
                    f"{_nfull_declared}, but its stored k coordinates do not "
                    f"cover the declared uniform mesh; {_bad.size} rows are "
                    f"missing, first {_declared_grid[_bad[0]].tolist()}.")

        if _identity_full_grid:
            # Trivial identity-only symmetry path
            self.sym_matrices = np.eye(3, dtype=np.int32)[None, :, :]
            # ``sym_mats_k`` MUST be TRS-augmented to length ``2·ntran``
            # exactly like the general branch below: every consumer
            # (``unfold_psi``/``spinor_rotation_for_sym_row`` derive
            # ``n_sym_spatial = len(sym_mats_k)//2``, ``zeta_loader``'s q-IBZ
            # TR mapping, ``compute_vcoul``'s q lookup) assumes the spatial
            # half is followed by the ``-S`` time-reversal half.  Leaving it at
            # length 1 made ``n_sym_spatial = 1//2 = 0``, so ``unfold_psi``
            # classified the *identity* row (sym_idx=0) as a TRS row and
            # returned ``iσ_y·conj(ψ)`` on an un-negated G-list — i.e. it
            # silently replaced ψ(r) by ψ*(−r) for EVERY k of any
            # symmetry-free (``nosym``) WFN.  Norms, ⟨ψ_m|ψ_n⟩, T and
            # (because ρ is inverted too) V_H all survive that, so it only
            # shows up in the position-dependent ionic terms: V_NL
            # collapses and V_loc shifts by O(100 eV).  See scorecard §Q.
            _sym_mats_k = self.sym_matrices.transpose(0, 2, 1).copy()
            self.sym_mats_k = np.concatenate(
                [_sym_mats_k, -_sym_mats_k], axis=0)
            # Rows this instance is ALLOWED to select (see ``trs_allowed``).
            # In this branch ``sym_idx_k`` is identically zero anyway, so
            # the restriction is documentation of intent; it is still set
            # so every code path can read one attribute.
            self._sym_row_ids_search = self.active_symmetry_rows.copy()
            self._sym_mats_k_search = self.sym_mats_k[
                self._sym_row_ids_search]
            self.translations = np.zeros((1, 3), dtype=np.float64)

            # In no-symmetry case, unfolded grid equals irreducible grid
            self.unfolded_kpts = np.asarray(wfn.kpoints, dtype=float)

            # Maps: each full k maps to itself; only identity symmetry
            self.irr_idx_k = np.arange(self.unfolded_kpts.shape[0], dtype=np.int32)
            self.sym_idx_k = np.zeros(self.unfolded_kpts.shape[0], dtype=np.int32)

            self.nk_tot = int(self.unfolded_kpts.shape[0])
            self.nk_red = int(getattr(wfn, 'nkpts', self.nk_tot))

            # kirr_fullids: identity mapping
            self.kirr_fullids = np.arange(self.nk_red, dtype=np.int32)

            # Rotation matrices and spinor (identity)
            self.R_grid = np.eye(3, dtype=np.int32)[None, :, :]
            self.Rinv_grid = self.R_grid.copy()
            # Keep the same augmented-row invariant as ``sym_mats_k``:
            # the TRS row is the negative Cartesian row.  The old one-row
            # special case made ``cartesian_action(sym_idx, ...)`` fail on
            # an identity-only WFN if a measured policy selected its TRS
            # partner, despite the reciprocal table correctly having two
            # rows.
            _R_identity = self.R_grid.astype(float)
            self.R_cart = np.concatenate(
                [_R_identity, -_R_identity], axis=0)
            self.U_spinor = np.eye(2, dtype=complex)[None, :, :]
            # Build direct integer-grid lookup for the identity/no-symmetry case.
            kgrid = np.asarray(wfn.kgrid, dtype=np.int32)
            shift = np.asarray(getattr(wfn, "shift", (0.0, 0.0, 0.0)), dtype=np.float64)
            shift_frac = shift / kgrid.astype(np.float64)
            kpts_wrapped = np.mod(self.unfolded_kpts - shift_frac[None, :], 1.0)
            self.kvecs_asints = np.mod(
                np.rint(kpts_wrapped * kgrid[None, :]).astype(np.int32),
                kgrid[None, :],
            )

            lookup = -np.ones(tuple(int(x) for x in kgrid), dtype=np.int32)
            lookup[
                self.kvecs_asints[:, 0],
                self.kvecs_asints[:, 1],
                self.kvecs_asints[:, 2],
            ] = np.arange(self.unfolded_kpts.shape[0], dtype=np.int32)

            # k−q maps on the unfolded (which equals reduced) grid.
            # Use direct modular arithmetic instead of the generic O(nk^3) search path.
            kminusq_mod = np.mod(
                self.kvecs_asints[:, None, :] - self.kvecs_asints[None, :, :],
                kgrid[None, None, :],
            )
            self.kqfull_map = lookup[
                kminusq_mod[:, :, 0],
                kminusq_mod[:, :, 1],
                kminusq_mod[:, :, 2],
            ]
            self.kq_map = self.kqfull_map.copy()

            # Integer q enumerations for k' - k outside the first BZ.
            qpt_vecs = self.kvecs_asints[:, None, :] - self.kvecs_asints[None, :, :]
            self.all_unfolded_qpts, inverse = np.unique(
                qpt_vecs.reshape(-1, 3),
                axis=0,
                return_inverse=True,
            )
            self.all_unfolded_qpt_ids = inverse.reshape(
                self.kvecs_asints.shape[0],
                self.kvecs_asints.shape[0],
            ).astype(np.int32)

            # Trivial-sym q-IBZ: each full-BZ q is its own IBZ partner under identity.
            n_full = int(self.kvecs_asints.shape[0])
            self.irr_idx_q = np.arange(n_full, dtype=np.int32)
            self.sym_idx_q = np.zeros(n_full, dtype=np.int32)
            self.q_irr_full_idx = np.arange(n_full, dtype=np.int32)
            self.q_irr_kgrid_int = self.kvecs_asints.copy()
            return

        # THE G-SPACE ACTION USES ``mtrx.T``, NOT ``mtrx``.  This comment
        # used to say `G' = mtrx @ G`, which contradicts both the line
        # directly below it (``sym_mats_k = mtrx.transpose(0,2,1)``) and
        # the only place G is actually rotated
        # (``wfn_loader/loader.py:604``, ``einsum('ij,kj->ki',
        # sym_mats_k[sym_idx], k_gvecs)``).  It also contradicted the
        # correct statement in ``syms_crystal_to_cartesian`` below
        # (``G_full = mtrx.T @ G_irr = sym_mats_k @ G_irr``).
        #
        # It is corrected rather than deleted because a reader of
        # ``SymMaps.__init__`` hits this before either of those, and the
        # transposed convention it stated is the one the known-broken
        # ``tests/bench/charge_density.py:159-174`` adopted — it rotates
        # G with ``R_grid`` (= ``mtrx``) and is wrong on every
        # non-symmorphic deck.
        #
        # So: k and G both transform with ``sym_mats_k = mtrx.T``
        # (column form).  Real space uses `Rinv = inv(mtrx)`:
        # `r' = Rinv @ r + τ` (see orbit_syms.centroid_source_map_and_wrap,
        # and BerkeleyGW/Common/symmetries.f90:189 which stores mtrx as
        # invert(mtrx_inv) where mtrx_inv is the real-space rotation).
        _operator_tables = build_spatial_operator_tables(wfn)
        self.sym_matrices = _operator_tables.sym_matrices
        self.sym_mats_k = _operator_tables.sym_mats_k[:wfn.ntran].copy()
        # BGW non-symmorphic translations (``tnp``).  Carried so
        # downstream callers (orbit-aware centroid sym perm, q-IBZ
        # unfold) don't need to re-thread the WFNReader through every
        # API.  Slice to ``[:ntran]`` to match ``sym_matrices`` —
        # legacy WFN files pad ``tnp`` to length 48.
        self.translations = _operator_tables.translations
        
        # Add time-reversal symmetry (k → -k) combined with each spatial symmetry
        # This is needed because QE uses time-reversal to reduce k-points, but doesn't
        # store it as one of the ntran symmetries.
        #
        # AUDIT MAP — the four places TRS lives in LORRAX, and what each
        # one is responsible for.  Read them together; every historical
        # bug here came from changing one without the others.
        #   1. THIS LINE builds the table: rows [0, ntran) are the spatial
        #      ops S (acting on k as ``sym_mats_k[s] @ k``), rows
        #      [ntran, 2·ntran) are ``−S`` = "time reversal ∘ S".  The
        #      TRS half is a k-space sign flip ONLY; it carries no τ of
        #      its own and no separate spinor matrix.
        #   2. ``self._sym_mats_k_search`` (just below) decides which rows
        #      may be SELECTED.  This is the gate: with TRS disallowed the
        #      table keeps its 2·ntran length (``unfold_psi`` requires that
        #      shape) but ``sym_idx_k``/``sym_idx_q`` can never exceed
        #      ntran, so no ψ is ever conjugated.
        #   3. ``unfold_psi`` applies the antiunitary itself — spinor
        #      factor ``iσ_y·conj(U_spinor[s])`` and the conjugated τ
        #      phase.  Its docstring carries the (★) derivation showing
        #      that the OTHER half of Θ, the negation of the G list, is
        #      supplied by ``sym_mats_k[sym_idx] = −S`` flowing into
        #      ``WfnLoader.gvecs(k='full_bz')``.  Half of Θ without the
        #      other half = ψ(r) → ψ*(−r) = scorecard §Q.
        #   4. ``density_symmetry_check`` tests the occupied 2c reference
        #      with raw/spatial-only partners or TRIM closure.  It never
        #      uses an antiunitary row as evidence.  Its verdict arrives
        #      here as ``wfn.trs_holds`` and drives (2).
        # ``orbit_syms.centroid_source_map_and_wrap(extend_trs=True)``
        # is the real-space counterpart: TRS leaves r fixed, so its rows
        # duplicate the spatial rows rather than negating anything.
        time_reversal_syms = -self.sym_mats_k  # S @ k -> -S @ k
        self.sym_mats_k = np.concatenate([self.sym_mats_k, time_reversal_syms], axis=0)

        # ...but SELECT only the rows authorized above.  With global TRS the
        # complete pair is physical.  Without it, an authenticated receipt
        # selects each raw WFN operation from the unitary or antiunitary half;
        # the legacy fallback selects only the presumed-unitary half.  The
        # table itself always
        # keeps its ``2·ntran`` length — ``unfold_psi`` derives
        # ``n_sym_spatial = len(sym_mats_k)//2`` and hard-raises on any
        # other shape (§Q) — so the gate lives in the SEARCH set used by
        # ``create_kpoint_symmetry_map`` / ``find_symmetry_ops_simple`` /
        # ``find_irreducible_bz_points``.
        self._sym_row_ids_search = self.active_symmetry_rows.copy()
        self._sym_mats_k_search = self.sym_mats_k[
            self._sym_row_ids_search]
        if not self.trs_allowed:
            import warnings as _warnings
            if self.qe_symmetry_binding is None:
                _warnings.warn(
                    "SymMaps: the two-component DFT reference says GLOBAL "
                    "TIME-REVERSAL SYMMETRY IS BROKEN. With no authenticated QE "
                    "operation typing, only the WFN header's presumed-unitary "
                    "rows may map the full BZ; an incomplete map refuses.",
                    RuntimeWarning)
            else:
                _warnings.warn(
                    "SymMaps: the two-component DFT reference says GLOBAL "
                    "TIME-REVERSAL SYMMETRY IS BROKEN. The authenticated QE schema "
                    f"still authorizes {len(self.qe_antiunitary_rows)} "
                    "operation-specific antiunitary row(s); arbitrary "
                    "k<->-k partners remain disabled.",
                    RuntimeWarning)

        # The list of full-zone k-points.  The k_full -> k_irr PARENT MAP
        # this used to compute alongside it is gone (design decision 4):
        # it was a second, independently-ruled derivation of the same
        # relationship ``find_symmetry_ops_simple`` produces below, it was
        # published as ``self.kpoint_map``, and nothing live read it.
        self.unfolded_kpts = self.create_kpoint_symmetry_map(wfn)
        # Validate the WFN's own stored rows before a coverage failure can
        # hide a contradictory kgrid/shift behind a generic symmetry error.
        self.kirr_fullids = self._match_file_wedge_rows(wfn)

        # ``None`` for the retired parent map: the parameter stays in the
        # signature for older callers, and the method has discarded its
        # argument since the ``del kpoint_map`` at the top of its body.
        self.irr_idx_k, self.sym_idx_k = self.find_symmetry_ops_simple(
            wfn, None, self.unfolded_kpts)


        self.nk_tot = int(self.unfolded_kpts.shape[0])
        self.nk_red = int(wfn.nkpts)

        # ``kirr_fullids[i]`` is the row of ``unfolded_kpts`` that IS the
        # WFN file's irreducible k-point ``i``.  Every consumer states that
        # contract in those words — ``gw_output.write_results`` ("the full-BZ
        # index of IBZ point i", and it subsets every eqp{0,1}.dat column with
        # it), ``qp_wfn``'s own dataset note, ``eqp_bgw``'s wedge subset — and
        # the two properties they lean on are that
        # ``unfolded_kpts[kirr_fullids]`` reproduces ``wfn.kpoints`` exactly
        # and that ``sym_idx_k[kirr_fullids]`` is the identity, so a row taken
        # here is the STORED wavefunction rather than a rotated or
        # time-reversed image of it.
        #
        # THE ROW IS NOW FOUND BY MATCHING k ITSELF, not by reading the star
        # labels, and that is the whole of the 2026-08-08 fix.  What stood
        # here before was "the first full-BZ row carrying irreducible label
        # ``i``", taken out of ``irr_idx_k``, with an identity fallback
        # ``kirr_fullids[i] = i`` for any label no row carries.  Both halves
        # are unsound.
        #
        # ``irr_idx_k`` is under no obligation to use every label.  It comes
        # from ``find_symmetry_ops_simple``, whose op-selection policy is
        # register-don't-touch (survey §8.1): the inner loop carries no
        # ``break``, so a full-BZ row reachable from more than one entry of
        # ``wfn.kpoints`` is labelled with the HIGHEST such entry, and the
        # lower ones are left with no members at all.  A WFN's k-list is
        # reduced by whatever code wrote the file and not by this class, so
        # two stored entries lying in one orbit is ordinary rather than
        # exotic, and every deck where it happens has orphaned labels.  For
        # an orphaned label the fallback then wrote ``i`` — an unrelated
        # full-BZ row — and said nothing about it.
        #
        # MEASURED on the four in-tree decks at bc37b4d3, which is why this
        # is not a cosmetic change.  ``gnppm_debug`` and ``bispinor_debug``
        # (9 k, ntran 2, stored IBZ list equal to the full grid) produced
        # ``[0, 1, 1, 3, 4, 5, 3, 5, 4]`` where the answer is ``[0..8]``:
        # four rows name a k a third of a reciprocal lattice vector away
        # from the one they claim, three pairs of rows collide, and IBZ
        # k 2, 6, 7 and 8 are never emitted at all.  ``cohsex_debug``
        # produced ``[0, 1, 1, 4]`` for ``[0, 1, 2, 4]``.
        # ``si_cohsex_debug`` came out right by luck: with 48 operations its
        # eight stars are disjoint, no label is orphaned, and each star's
        # first member happens to be the stored k.  An eqp{0,1}.dat written
        # through the broken list carries duplicated and mislabelled wedge
        # rows while the full-BZ arrays behind it are perfectly correct,
        # which is the shape of defect that survives every norm, degeneracy
        # and electron-count check downstream of it.
        #
        # The match is exact rather than nearest-neighbour.  ``unfolded_kpts``
        # is the uniform grid this class generates from ``wfn.kgrid`` and
        # ``wfn.shift``, so a stored IBZ point that is not ON it means the
        # file's metadata disagrees with its own k-list; there is no useful
        # row to return in that case and it raises instead of guessing.  The
        # tolerance is ``find_symmetry_ops_simple``'s 1e-6, the same number
        # the star map two lines up was built with.
        # useful maps:
        # k (full zone) to kbar 
        # k,q (both full zone) to k-q (full zone)
        
        # Get rotation matrices and their spinor representations
        self.R_grid = np.rint(self.sym_matrices).astype(np.int32)
        self.Rinv_grid = np.rint(np.linalg.inv(self.R_grid)).astype(np.int32)

        
        # ``R_cart`` covers the full ``2·ntran``-row sym_mats_k (kept for
        # any caller that needs cartesian-frame rotations e.g. for current
        # density / transverse channels). ``U_spinor`` is restricted to the
        # SPATIAL half: the TRS-row spinor is ``iσ_y · conj(U_spinor[s])``
        # (computed inside ``unfold_psi`` and similar consumers). Before
        # 2026-05-14 this array was length 2·ntran with the TRS half
        # computed wrong by ``get_spinor_rotations(-S_spatial)``'s
        # det<0→-R flip; restricting to length ntran here makes the bug
        # unreachable by construction. See ``reports/trs_sym_audit_2026-05-14``
        # Site #6 for the per-element derivation.
        self.R_cart = _operator_tables.R_cart
        self.U_spinor = _operator_tables.U_spinor
        self.kq_map = self.get_kminusq_map(wfn, self.unfolded_kpts)
        self.kqfull_map = self.get_kminusqfull_map(wfn, self.unfolded_kpts)


        # the above kq maps are for inputting some k and some q and getting k-q in the 1BZ, but it is actually necessary to store W_q on q outside 1BZ
        # As such, the following functions are for inputting some k and some k' and getting the relevant q outside 1BZ
        kx, ky, kz = np.meshgrid(np.arange(wfn.kgrid[0]), 
                        np.arange(wfn.kgrid[1]), 
                        np.arange(wfn.kgrid[2]), 
                        indexing='ij')
        self.kvecs_asints = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1) # kpoints * kgrid (kpoints as integers)

        # Generate q-vectors using broadcasting
        qpt_vecs = self.kvecs_asints[:, None, :] - self.kvecs_asints[None, :, :]  # Automatic broadcasting

        # Find unique q-vectors (already vectorized)
        self.all_unfolded_qpts, inverse = np.unique(
            qpt_vecs.reshape(-1, 3), axis=0, return_inverse=True)
        self.all_unfolded_qpt_ids = inverse.reshape(
            len(self.kvecs_asints), len(self.kvecs_asints),
        ).astype(np.int32)

        # Eager q-IBZ reduction (was lazy in `find_irreducible_qpoints`; that
        # method is gone — all consumers read these instance attrs directly).
        # q lives on the same kgrid as k (q = k - k'), so we reuse
        # ``sym_mats_k`` (which already includes time-reversal). Note that
        # `is_trs[i_full] = sym_idx_q[i_full] >= ntran` is implicit; not stored.
        irr_idx_q, sym_idx_q_search, q_irr_kgrid_int = find_irreducible_bz_points(
            self.kvecs_asints, self._sym_mats_k_search, irr_kgrid_int=None,
        )
        self.irr_idx_q = irr_idx_q
        self.sym_idx_q = self._sym_row_ids_search[
            np.asarray(sym_idx_q_search, dtype=np.int32)]
        self.q_irr_kgrid_int = q_irr_kgrid_int
        # q_irr_full_idx[i_irr] = full-BZ row index for IBZ q i_irr.
        # Derived as first-occurrence of i_irr in irr_idx_q (ordered to match
        # the q_irr_kgrid_int row ordering).
        _, first_occ = np.unique(irr_idx_q, return_index=True)
        self.q_irr_full_idx = np.sort(first_occ).astype(np.int32)


    def get_qpt_id_from_kkp(self, kidx, kpidx):
        # meant to return the unique q idx of kp-k, so that sym.all_unfolded_qpts[qpt_id] = kp-k
        kpminkvec = self.kvecs_asints[kpidx] - self.kvecs_asints[kidx]
        return np.where(np.all(self.all_unfolded_qpts == kpminkvec, axis=1))[0][0]

        

    @staticmethod
    def _wrap_to_bz(kpts):
        """Wrap crystal-coordinate vectors into the first Brillouin zone."""
        wrapped = np.mod(np.asarray(kpts, dtype=np.float64), 1.0)
        wrapped[wrapped > 0.99999] = 0.0
        return wrapped

    @staticmethod
    def _periodic_delta(points, target):
        """Return shortest-image fractional-coordinate differences."""
        delta = np.asarray(points, dtype=np.float64) - np.asarray(target, dtype=np.float64)[None, :]
        return delta - np.round(delta)

    def _match_file_wedge_rows(self, wfn):
        """Map stored WFN k rows to the generated uniform grid or refuse."""
        rows = np.empty(int(wfn.nkpts), dtype=np.int32)
        stored = np.asarray(wfn.kpoints, dtype=np.float64)
        for kirr, kpoint in enumerate(stored):
            metric = np.max(np.abs(self._periodic_delta(
                self.unfolded_kpts, kpoint)), axis=1)
            hit = int(np.argmin(metric))
            if metric[hit] > 1.0e-6:
                raise RuntimeError(
                    f"SymMaps: irreducible k-point {kirr} {kpoint.tolist()} "
                    f"is not on the uniform "
                    f"{tuple(int(x) for x in np.asarray(wfn.kgrid))} k-grid "
                    f"that this WFN's own kgrid/shift generate; closest row "
                    f"{hit} is {self.unfolded_kpts[hit].tolist()}, off by "
                    f"{metric[hit]:.3e}. Fix the file's k-list or kgrid/shift.")
            rows[kirr] = hit
        return rows

    def _generate_uniform_full_kpoints(self, wfn):
        """Return the full uniform crystal-coordinate k-grid implied by the WFN metadata."""
        kx = np.linspace(0, 1, wfn.kgrid[0], endpoint=False)
        ky = np.linspace(0, 1, wfn.kgrid[1], endpoint=False)
        kz = np.linspace(0, 1, wfn.kgrid[2], endpoint=False)

        kx += wfn.shift[0] / wfn.kgrid[0]
        ky += wfn.shift[1] / wfn.kgrid[1]
        kz += wfn.shift[2] / wfn.kgrid[2]

        kpts_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
        return self._wrap_to_bz(np.stack([k.flatten() for k in kpts_mesh]).T)

    def create_kpoint_symmetry_map(self, wfn):
        """The full-grid k-point list.

        Returns ``full_kpoints`` — the uniform grid, wrapped to the BZ.

        THE PARENT MAP THIS USED TO RETURN IS GONE (design decision 4,
        2026-08-07).  It was a ``(n_k_full x 2*ntran x nrk)`` python triple
        loop computing, by its OWN tie-break rule (lowest sym index that
        maps ``k_full`` INTO the IBZ, plus a nearest-neighbour fallback),
        the same k_full -> k_irr relationship that
        :meth:`find_symmetry_ops_simple` computes a few lines later by the
        SHIPPING rule (highest matching ``ikbar``, then lowest sym — the
        register-don't-touch policy of survey §8.1).  3e002f2 recorded that
        the two agreed on all four in-tree fixtures, which is exactly the
        shape of a second source of truth waiting to drift.  It was
        published as ``SymMaps.kpoint_map`` and read by nothing live: the
        only readers in the tree are ``misc/archived_tests/
        get_interp_vectors.py`` (:244, :249) and ``misc/archived_tests/
        symtest.ipynb``, neither of which is collected, and the notebook
        also prints a ``kpoint_map_ibz_ids`` that has not existed for
        longer still.  The live tables come from ``find_symmetry_ops_simple``
        and are pinned bit-for-bit on all four decks by
        ``services/symmetry_maps/tests/test_symmetry_maps_deck_tables.py``.

        The method is KEPT (it is the only place the uniform grid is built,
        and it is public surface) rather than inlined.
        """
        return self._generate_uniform_full_kpoints(wfn)

    def find_symmetry_ops_simple(self, wfn, kpoint_map, full_kpts):
        del kpoint_map  # kept in signature for compatibility with older callers
        # Searches ``_sym_mats_k_search``: the full paired table when global
        # TRS holds, the schema-typed unitary/antiunitary rows when it does
        # not, or the presumed-unitary half in the receipt-free fallback.
        # The planner owns the registered highest-parent/lowest-operation
        # tie break and exposes coverage.
        irk_to_k_map, irk_sym_map_search, matched = (
            map_full_kpoints_to_irreducible(
                wfn.kpoints, self._sym_mats_k_search, full_kpts))
        irk_sym_map = self._sym_row_ids_search[
            np.asarray(irk_sym_map_search, dtype=np.int32)]

        if self.qe_symmetry_binding is not None and not np.all(matched):
            bad = np.where(~matched)[0]
            pure_tr_note = (
                " QE allowed pure k<->-k folding, but the measured global "
                "TRS verdict forbids using it."
                if (not self.trs_allowed
                    and bool(self.qe_permitted_pure_time_reversal)) else "")
            raise ValueError(
                f"SymMaps: authenticated QE symmetry operations are "
                f"incomplete for this WFN: {bad.size} of "
                f"{full_kpts.shape[0]} full-BZ k-points cannot be reached "
                f"using {len(self._sym_row_ids_search)} authorized typed "
                f"rows from {self.qe_schema_path}. First unmatched: "
                f"{full_kpts[bad[0]].tolist()}. Refusing rather than adding "
                f"an unrecorded spatial or antiunitary operation.{pure_tr_note}")

        if not self.trs_allowed and not np.all(matched):
            n_bad = int(np.count_nonzero(~matched))
            raise ValueError(
                f"SymMaps: {n_bad} of {full_kpts.shape[0]} full-BZ k-points "
                f"cannot be reached from the IBZ using the SPATIAL symmetry "
                f"operations alone, and the DFT reference check says "
                f"time-reversal symmetry is BROKEN for these wavefunctions "
                f"(occupied-subspace residual above tolerance), so the "
                f"time-reversal rows must not be used. This WFN's k-mesh was "
                f"reduced with an assumption its own wavefunctions "
                f"contradict; regenerate it with `noinv=.true.` (or fix the "
                f"magnetic ground state). Set LORRAX_TRS_CHECK=0 to restore "
                f"the old flags-only behaviour at your own risk."
            )

        if self.trs_allowed and not np.all(matched):
            bad = np.where(~matched)[0]
            raise ValueError(
                f"SymMaps: WFN symmetry data are incomplete: {bad.size} of "
                f"{full_kpts.shape[0]} full-BZ k-points cannot be reached "
                f"from the stored k rows using the permitted spatial and "
                f"time-reversal-composed operations "
                f"(ntran_search={len(self._sym_mats_k_search)}). First "
                f"unmatched: {full_kpts[bad[0]].tolist()}. Refusing instead "
                f"of substituting stored k 0 with identity; that substitution "
                f"would fabricate Gamma wavefunctions. Regenerate the WFN "
                f"with a self-consistent symmetry header or an explicit full "
                f"k-grid.")

        # Note: TRS-augmented sym indices (irk_sym_map >= ntran) are now
        # handled correctly by ``unfold_psi`` (PR3, 2026-05-14). The
        # previous warning about "non-symmorphic phases are NOT applied
        # for these k-points" is obsolete.
        return irk_to_k_map, irk_sym_map

    def validate_atomic_symmetries(self, wfn, tol=1e-6):
        """Return a list of failures for spatial symmetries acting on atoms."""
        atom_crys = np.asarray(wfn.atom_crys, dtype=np.float64)
        atom_types = np.asarray(wfn.atom_types)
        failures = []

        for sym_idx in range(int(wfn.ntran)):
            rot = np.asarray(np.linalg.inv(wfn.sym_matrices[sym_idx]), dtype=np.int32)
            tau = np.asarray(wfn.translations[sym_idx], dtype=np.float64) / (2.0 * np.pi)
            available = list(range(len(atom_crys)))

            for atom_idx, pos in enumerate(atom_crys):
                transformed = self._wrap_to_bz(rot @ pos + tau)
                same_species = [j for j in available if atom_types[j] == atom_types[atom_idx]]
                if not same_species:
                    failures.append(
                        f"sym {sym_idx}: atom {atom_idx} has no same-species candidates"
                    )
                    break

                candidates = atom_crys[same_species]
                metric = np.max(np.abs(self._periodic_delta(candidates, transformed)), axis=1)
                best = int(np.argmin(metric))
                if metric[best] > tol:
                    failures.append(
                        f"sym {sym_idx}: atom {atom_idx} maps to {transformed.tolist()} "
                        "with no unique same-species match"
                    )
                    break
                available.remove(same_species[best])

        return failures

    def validate_kgrid_unfolding(self, wfn, tol=1e-6):
        """Return a list of failures for full-grid k-point unfolding."""
        failures = []
        full_grid = self._generate_uniform_full_kpoints(wfn)
        if full_grid.shape != self.unfolded_kpts.shape:
            return [
                f"full-grid size mismatch: generated {full_grid.shape[0]} points, "
                f"SymMaps has {self.unfolded_kpts.shape[0]}"
            ]

        for ik, k_full in enumerate(full_grid):
            metric = np.max(np.abs(self._periodic_delta(self.unfolded_kpts, k_full)), axis=1)
            ik_full = int(np.argmin(metric))
            if metric[ik_full] > tol:
                failures.append(
                    f"uniform-grid point {ik} {k_full.tolist()} missing from SymMaps.unfolded_kpts"
                )
                continue

            ik_irr = int(self.irr_idx_k[ik_full])
            sym_idx = int(self.sym_idx_k[ik_full])
            sym_krep = np.asarray(self.sym_mats_k[sym_idx], dtype=np.int32)
            kg0 = self.get_umklapp_vector(wfn, ik_full, sym_idx, ik_irr, sym_krep)
            mapped = sym_krep @ np.asarray(wfn.kpoints[ik_irr], dtype=np.float64) + kg0
            if np.max(np.abs(mapped - self.unfolded_kpts[ik_full])) > tol:
                failures.append(
                    f"ik_full={ik_full}: S*k_irr + kg0 does not reproduce full k-point"
                )

        return failures

    @staticmethod
    def syms_crystal_to_cartesian(wfn):
        """Cartesian rotation matrix used as input to ``get_spinor_rotations``.

        ``get_spinor_rotations`` runs Markley's quaternion algorithm and
        requires ORTHOGONAL 3D rotation matrices. The matrix it consumes is
        the cartesian image of LORRAX's ``mtrx`` (= ``sym_matrices``) — NOT
        of ``mtrx.T`` (= ``sym_mats_k``), NOT of ``inv(mtrx)``.

        Verified against nosym ground truth: U_spinor built from this
        R_cart reproduces the nosym ψ within the degenerate-subspace
        unitary gauge.

        CAUTION FOR OTHER CONSUMERS.  Because ``mtrx`` is the inverse
        real-space rotation while ``mtrx.T`` is what acts on k and G, this
        matrix is the INVERSE of the Cartesian rotation that carries
        k_irr to S·k_irr.  ``get_spinor_rotations`` is unaffected — its
        quaternion extraction uses the transposed Shepperd form, so the two
        inversions cancel — but anything rotating a Cartesian INDEX (a
        dipole or any rank≥1 operator) must use the TRANSPOSE of this
        matrix.  Using it untransposed leaves norms, hermiticity and traces
        intact, so the error is invisible to the obvious checks.

        Conjugation formula (column form, ``r_cart = avec.T @ r_frac`` where
        ``avec[i, :]`` is the i-th real-space lattice vector):

            R_cart = avec.T @ mtrx @ inv(avec.T)
                   = inv(bvec) @ mtrx @ bvec
                   (the two are algebraically equivalent given
                   ``avec @ bvec.T = I``, hence ``inv(bvec) = avec.T``)

        This is the LORRAX convention for the "rotation in cartesian" that
        the rest of the codebase consumes: it matches the G-space action
        ``G_full = mtrx.T @ G_irr = sym_mats_k @ G_irr`` (column form) by
        the relation ``R_cart^{-T} = avec.T @ mtrx.T @ inv(avec.T) = G-side
        cartesian rotation``. For orthogonal R, ``R^{-T} = R``, so the two
        are inverses of each other but both orthogonal; the spinor SU(2)
        Markley algorithm needs the one returned here (i.e. ``mtrx``, not
        ``mtrx.T`` or ``inv(mtrx)``).

        Output covers the full TRS-augmented sym table: rows ``[:ntran]`` are
        the spatial cartesian rotations; rows ``[ntran:]`` are ``-R_spatial``
        (matches the convention that ``sym_mats_k[ntran:] = -sym_mats_k[:ntran]``;
        TRS does not change the spatial rotation, it only adds a complex-conj /
        iσ_y factor handled separately in ``unfold_psi``).

        History
        -------
        Pre-fix (2026-05-14) this used ``inv(bvec) @ sym_mats_k @ bvec``
        — wrong because ``sym_mats_k = mtrx.T`` instead of ``mtrx``. The
        two matrices ``mtrx.T`` and ``mtrx`` give different SU(2) (one is
        the adjoint of the other for orthogonal R), so U_spinor was wrong
        on every system but the error CANCELLED in Σ_X for involutive
        groups (MoS2 σ_h: mtrx = mtrx.T) and for cubic groups whose mtrx
        entries are integer-orthogonal in crystal coords. CrI3 P-3 (hex,
        non-involutive C3/S6) gave |R Rᵀ-I|∞ ≈ 3.5 → 6 eV Σ_X failure;
        Si Fd-3m (non-symmorphic) gave 160 eV failure via a different
        Σ_X-level amplification of the same wrong U_spinor.

        The original code's ``# NOT SURE IF THESE SHOULD BE SYM_MATS_K OR
        SYM_MATS TODO`` was the smoking-gun comment. Answer: SYM_MATRICES.
        """
        A_T = np.asarray(wfn.avec).T
        A_T_inv = np.linalg.inv(A_T)
        # Spatial-only mtrx (length ntran). TRS-augmented rows are -R_spatial
        # in cartesian (TRS doesn't change the spatial rotation).
        ntran = int(wfn.ntran)
        mtrx = np.asarray(wfn.sym_matrices[:ntran])
        R_spatial = np.einsum('ij,njk,kl->nil', A_T, mtrx, A_T_inv)
        R_spatial = np.around(R_spatial, decimals=10)
        # Match the existing 2·ntran-row convention for downstream consumers.
        R_full = np.concatenate([R_spatial, -R_spatial], axis=0)
        return R_full

    @property
    def q_irr_is_full_identity(self):
        """Whether the q-IBZ table is exactly the ordered full q table.

        This is stronger than equality of the two q counts.  It proves that
        every parent row is its own full-zone row, that the full-to-parent
        map is identity, and that the stored integer q rows are byte/layout
        equivalent to ``kvecs_asints``.  Storage/provenance consumers use
        this named service fact rather than rebuilding q-table logic.
        """
        q_full = np.asarray(self.kvecs_asints, dtype=np.int32)
        n_q = int(q_full.shape[0])
        identity = np.arange(n_q, dtype=np.int32)
        return bool(
            np.array_equal(
                np.asarray(self.q_irr_full_idx, dtype=np.int32), identity)
            and np.array_equal(
                np.asarray(self.irr_idx_q, dtype=np.int32), identity)
            and np.array_equal(
                np.asarray(self.q_irr_kgrid_int, dtype=np.int32), q_full))

    def operation_rows(self, rows):
        """Return reciprocal action, spatial translation and TR bit.

        Rows use the one canonical ``[unitary, antiunitary]`` layout.  The
        reciprocal matrix already contains the antiunitary minus; the
        translation always belongs to the underlying spatial Seitz row.
        """
        raw = np.asarray(rows)
        if (not np.issubdtype(raw.dtype, np.integer)
                or np.any(raw < 0)):
            raise ValueError(
                f"SymMaps.operation_rows: rows must be nonnegative integers; "
                f"got {raw!r}.")
        idx = raw.astype(np.int32, copy=False)
        n = int(np.asarray(self.sym_matrices).shape[0])
        if np.any(idx >= 2 * n):
            raise ValueError(
                f"SymMaps.operation_rows: row outside [0,{2 * n}); "
                f"got {raw!r}.")
        spatial = idx % n
        return (np.asarray(self.sym_mats_k)[idx],
                np.asarray(self.translations)[spatial], idx >= n)

    def fft_grid_pullback(self, rows, fft_grid, *, validate=True):
        """Return real-space FFT pullbacks for canonical operation rows."""
        raw = np.asarray(rows)
        if raw.ndim != 1:
            raise ValueError(
                "SymMaps.fft_grid_pullback: rows must be rank one; "
                f"got {raw.shape}.")
        _, translations, _ = self.operation_rows(raw)
        n = int(np.asarray(self.sym_matrices).shape[0])
        spatial = raw.astype(np.int32, copy=False) % n
        from .orbit_syms import fft_grid_pullback_perm
        return fft_grid_pullback_perm(
            np.asarray(self.sym_matrices)[spatial], translations, fft_grid,
            validate=validate)

    def cartesian_action(self, rows, *, axial, time_odd):
        """Forward action for a typed Cartesian index.

        ``axial`` selects ``det(R) R`` instead of the polar ``R``;
        ``time_odd`` supplies the minus on antiunitary rows.  Antiunitary
        complex conjugation is applied by the tensor action, not this real
        representation table.
        """
        if not isinstance(axial, (bool, np.bool_)):
            raise TypeError("SymMaps.cartesian_action: axial must be bool.")
        if not isinstance(time_odd, (bool, np.bool_)):
            raise TypeError("SymMaps.cartesian_action: time_odd must be bool.")
        raw = np.asarray(rows)
        _, _, antiunitary = self.operation_rows(raw)
        n = int(np.asarray(self.sym_matrices).shape[0])
        spatial = raw.astype(np.int32, copy=False) % n
        forward = np.swapaxes(
            np.asarray(self.R_cart[:n], dtype=np.float64), -1, -2)[spatial]
        if axial:
            forward = (np.linalg.det(forward)[..., None, None] * forward)
        if time_odd:
            forward = np.where(
                np.asarray(antiunitary)[..., None, None], -forward, forward)
        return forward

    def spinor_action(self, rows, *, nspinor):
        """Canonical unitary/antiunitary spinor factor for operation rows."""
        self.operation_rows(rows)  # validate against this instance
        n = int(np.asarray(self.sym_matrices).shape[0])
        return spinor_rotation_for_sym_row(
            self.U_spinor, rows, n, nspinor=nspinor)

    def reciprocal_phase(self, row, carriers):
        """Nonsymmorphic phase for reciprocal carriers under one typed row."""
        reciprocal, translation, _ = self.operation_rows(row)
        return tau_phase_row(reciprocal, translation, carriers)

    def unfold_wavefunction(self, coefficients, *, row, g_parent):
        """Apply one complete spatial/translation/TR wavefunction action."""
        self.operation_rows(row)  # validate against this instance
        return unfold_psi(
            coefficients, sym_idx=row, g_kbar=g_parent,
            sym_mats_k=self.sym_mats_k, translations=self.translations,
            U_spinor_spatial=self.U_spinor)

    @staticmethod
    def get_spinor_rotations(wfn, sym_matrices_cart):
        """
        Converts a list of rotation matrices to their spinor representations using Markley's modification
        of Shepperd's algorithm (aka quaternion representation, see Brad Barker's dissertation).

        When the wavefunction files store four-component states these routines will
        compute the corresponding 4x4 spinor rotation matrices.
        
        Parameters:
        sym_matrices (numpy.ndarray): Array of 3x3 rotation matrices with shape (nsym, 3, 3)

        Returns:
        numpy.ndarray: Array of spinor matrices with shape (nsym, 2, 2) of complex type
        """
        nsym = len(sym_matrices_cart)
        spinor_matrices = np.zeros((nsym, 2, 2), dtype=complex)
        
        # Add Pauli matrices (moved outside the loop since they're constant)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        for isym, R in enumerate(sym_matrices_cart):
            # Improper rotations (det < 0) must be converted to proper before
            # computing the SU(2) spinor matrix. The inversion part maps to
            # identity in SU(2). Matches BGW Common/spinor_symmetries.f90.
            if np.linalg.det(R) < 0:
                R = -R

            # Construct the symmetric 4x4 matrix Q
            Q = np.zeros((4, 4))
            Q[0, 0] = R[0, 0] + R[1, 1] + R[2, 2]
            Q[0, 1] = Q[1, 0] = R[1, 2] - R[2, 1]
            Q[0, 2] = Q[2, 0] = R[2, 0] - R[0, 2]
            Q[0, 3] = Q[3, 0] = R[0, 1] - R[1, 0]
            
            Q[1, 1] = R[0, 0] - R[1, 1] - R[2, 2]
            Q[1, 2] = Q[2, 1] = R[0, 1] + R[1, 0]
            Q[1, 3] = Q[3, 1] = R[0, 2] + R[2, 0]
            
            Q[2, 2] = -R[0, 0] + R[1, 1] - R[2, 2]
            Q[2, 3] = Q[3, 2] = R[1, 2] + R[2, 1]
            
            Q[3, 3] = -R[0, 0] - R[1, 1] + R[2, 2]

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(Q)
            
            # The quaternion is the eigenvector corresponding to the largest eigenvalue
            q = eigenvectors[:, np.argmax(eigenvalues)]
            q = q / np.linalg.norm(q)  # Normalize
            
            # Quaternion components
            q0, q1, q2, q3 = q
            
            # Compute the angle
            theta = 2 * np.arccos(q0)
            
            # Handle axis calculation
            sin_theta_over_2 = np.sqrt(1 - q0**2)
            if sin_theta_over_2 < 1e-8 or np.isclose(theta, 0) or np.isclose(theta, 2 * np.pi):
                theta = 0.0
                n = np.array([1.0, 0.0, 0.0])
            elif np.isclose(theta, np.pi):
                axis = np.array([q1, q2, q3])
                n = axis / np.linalg.norm(axis)
            else:
                n = np.array([q1, q2, q3]) / sin_theta_over_2
                n = n / np.linalg.norm(n)
            
            # Calculate spinor matrix components
            cos_half_theta = np.cos(theta/2)
            sin_half_theta = np.sin(theta/2)
            
            # Construct spinor matrix
            spinor = cos_half_theta * np.eye(2, dtype=complex)
            spinor -= 1j * sin_half_theta * (
                n[0] * sigma_x +
                n[1] * sigma_y +
                n[2] * sigma_z
            )
            
            spinor_matrices[isym] = spinor
        
        return spinor_matrices

    def get_kminusq_map(self, wfn, full_kpts):
        """Create mapping between k and k-q points in the full k-point grid.
        
        Args:
            wfn: WFNReader instance
            full_kpts: Array of all k-points in the full grid
            
        Returns:
            numpy.ndarray: kq_map[ik,iq] = index of k-q in full k-point grid,
                          where ik is index in full grid, iq is index in reduced grid
        """
        return self._get_kminusq_index_map(
            full_kpts, np.asarray(wfn.kpoints), name="reduced q grid")

    def get_kminusqfull_map(self, wfn, full_kpts):
        del wfn
        return self._get_kminusq_index_map(
            full_kpts, full_kpts, name="full q grid")

    @staticmethod
    def _get_kminusq_index_map(full_kpts, qpts, *, name):
        """Map ``k-q`` to the periodic full-grid row in O(Nk*Nq).

        The old implementation performed a nearest-point search over the
        complete k grid for every pair, making the full map O(Nk**3) in
        Python.  Uniform-grid coordinates are exact modulo a reciprocal
        lattice vector, so a quantized periodic-coordinate lookup gives the
        same row directly.  The quantization is much tighter than the
        historical 1e-4 acceptance tolerance.
        """
        full = np.asarray(full_kpts, dtype=np.float64)
        qarr = np.asarray(qpts, dtype=np.float64)
        if full.ndim != 2 or full.shape[1] != 3:
            raise ValueError(
                f"full_kpts must have shape (Nk, 3); got {full.shape}")
        if qarr.ndim != 2 or qarr.shape[1] != 3:
            raise ValueError(
                f"qpts must have shape (Nq, 3); got {qarr.shape}")

        key_scale = np.int64(100_000_000)

        def _keys(points):
            wrapped = np.mod(points, 1.0)
            return np.mod(
                np.rint(wrapped * key_scale).astype(np.int64), key_scale)

        full_keys = _keys(full)
        lookup = {tuple(row): i for i, row in enumerate(full_keys)}
        if len(lookup) != len(full):
            raise ValueError("full k-point grid contains periodic duplicates")

        target_keys = _keys(full[:, None, :] - qarr[None, :, :])
        flat_keys = target_keys.reshape(-1, 3)
        flat_map = np.fromiter(
            (lookup.get(tuple(row), -1) for row in flat_keys),
            dtype=np.int32,
            count=flat_keys.shape[0],
        )
        # Preserve the former 1e-4 acceptance for unusually noisy input
        # coordinates without charging every ordinary grid point for a dense
        # nearest-neighbour search.
        for bad in np.flatnonzero(flat_map < 0):
            ik, iq = np.unravel_index(int(bad), target_keys.shape[:2])
            kminusq = full[ik] - qarr[iq]
            delta = full - kminusq[None, :]
            delta -= np.round(delta)
            diffs = np.sum(np.abs(delta), axis=1)
            hit = int(np.argmin(diffs))
            min_diff = float(diffs[hit])
            if min_diff > 1e-4:
                raise ValueError(
                    f"k-q point {kminusq} from {name} not found in k-point "
                    f"grid (nearest periodic L1 distance {min_diff:.3e})")
            flat_map[bad] = hit
        return flat_map.reshape(target_keys.shape[:2])
    
    def get_umklapp_vector(self, wfn, nk, sym_idx, kbar_idx, sym_krep):
        """Return BGW's kg0 for the selected full-zone k-point.

        BGW defines the integer umklapp vector kg0 through
            k_full = S k_irred + kg0 .
        We use the same convention here so that the associated
        non-symmorphic phase matches Common/gmap.f90.
        """
        if sym_idx >= len(self.sym_matrices):
            q_full = np.asarray(sym_krep @ wfn.kpoints[kbar_idx], dtype=np.float64)
            q_inzone = q_full % 1.0
            q_inzone[q_inzone > 0.9999] = 0.0
            return (q_inzone - q_full).astype(np.int32)

        k_full = np.asarray(self.unfolded_kpts[nk], dtype=np.float64)
        skbar = np.asarray(sym_krep @ wfn.kpoints[kbar_idx], dtype=np.float64)
        kg0 = np.rint(k_full - skbar).astype(np.int32)

        if not np.allclose(skbar + kg0, k_full, atol=1e-6):
            raise ValueError(
                f"Failed to determine symmetry umklapp for nk={nk}: "
                f"k_full={k_full}, S*kbar={skbar}, kg0={kg0}"
            )
        return kg0

    def find_qpoint_index(self, q_ext, tol=1e-6):
        """Find a periodic q-point in the canonical full-grid row table.

        Args:
            q_ext: Vector of length 3 (crystal coordinates)
            tol: Tolerance for floating point comparison

        Returns:
            Index of matching q-point, or raises ValueError if not found
        """
        q = np.asarray(q_ext, dtype=np.float64)
        tolerance = float(tol)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            raise ValueError(
                "find_qpoint_index: q_ext must be a finite length-3 "
                f"crystal-coordinate vector, got shape={q.shape}, "
                f"values={q.tolist()}.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"find_qpoint_index: tol must be finite and nonnegative, "
                f"got {tol!r}.")

        # ``unfolded_kpts`` deliberately preserves the representative paired
        # with the unfolded G labels; it may therefore contain signed rows.
        # Compare on the reciprocal torus instead of wrapping only the query
        # and measuring an ordinary distance (where -4/9 and +5/9 differ by
        # one despite being the same physical point).
        metric = np.max(np.abs(self._periodic_delta(
            np.asarray(self.unfolded_kpts, dtype=np.float64), q)), axis=1)
        match = int(np.argmin(metric))
        if float(metric[match]) > tolerance:
            raise ValueError(
                "No matching q-point found within periodic tolerance "
                f"{tolerance}; nearest row {match} has max|Delta q|="
                f"{float(metric[match]):.6e}.")
        return match


# ---------------------------------------------------------------------------
# k-stars: selecting to the IBZ and broadcasting back
# ---------------------------------------------------------------------------
#
# AN INDEX MAP, NOT AN UNFOLD.  ``WfnLoader.load(k='full_bz')`` builds the
# full-BZ ψ by unfolding the IBZ ψ in G-space (rotated G-vectors, τ phase,
# spinor rotation, TRS conjugation), so band n at a star member ``Sk̄`` is
# the unfold of band n at ``k̄``:
#
#     ψ_{Sk̄,n} = 𝒰(S) ψ_{k̄,n}          at FIXED band index n
#
# An operator O commuting with S then has the same matrix at every member
# of the star, ⟨m,Sk̄|O|n,Sk̄⟩ = ⟨m,k̄|O|n,k̄⟩, and so does its eigenvector
# matrix.  Moving a BAND-INDEX object between the IBZ and the full BZ is
# therefore pure indexing: no umklapp phase, no ``sym_perm`` gather, no
# spinor rotation.  ``unfold_isdf_operator`` needs all three because
# ``V_q[μ,ν]`` is indexed by real-space centroids and a symmetry moves r; bands
# it does not move.  The two routines are not variants of each other.
#
# SPATIAL IS EQUALITY, TIME REVERSAL IS CONJUGATION.  Θ = iσ_y K is
# ANTIunitary, so for O commuting with it,
#
#     ⟨Θm,k|O|Θn,k⟩ = conj(⟨m,k|O|n,k⟩)   ⇒   O(−k) = conj(O(k))
#
# THE STAR PREDICATE IS AN XOR.  A star helper relates a member to
# another FULL-BZ ROW — the star's first member — which carries a
# ``sym_idx`` of its own, so the conjugation applies iff the two DIFFER
# in TRS-ness:
#
#     conj(member) ⟺ trs(member) XOR trs(first row of member's star)
#
# That is :func:`_star_conj_flags`, and it is the ONLY conjugation rule
# in this module's star surface: ``star_spread`` (via
# :func:`_spread_tables`), ``star_broadcast(trs_reference="star_row")``
# and both ``KStarMap`` paths all read it, and nothing re-derives it.
# Landed 3e002f2; comments elsewhere in the tree that describe the star
# helpers as using the member's own flag predate it.
#
# The member's own flag — ``sym_idx_k >= n_sym_spatial``, the convention
# ``unfold_psi`` and ``unfold_isdf_operator`` use — is the RIGHT predicate for
# exactly one flavour of operand: a raw IBZ slab, whose rows carry no
# symmetry operation and so are TRS-false by construction.  That is
# ``star_broadcast(trs_reference="ibz_slab")``, and the XOR reduces to it
# when every star's first row happens to be spatial.  The two flavours
# are named, never defaulted between; see :func:`star_broadcast`.
#
# Assuming plain equality
# instead is not a small error: on MoS₂ 4×4 (nk 16→10, 6 TRS pairs) the
# conjugation rule holds to 1.2e-16 while equality is off by 3.6e-01
# RELATIVE, on every non-singleton star, and hermiticity, the norm and the
# electron count all survive it.  Conjugation rather than transposition:
# the two agree for Hermitian O, only conjugation stays correct for a
# non-Hermitian channel.
#
# :func:`star_spread` checks the premise that full-BZ ψ really is the
# unfold.  A path that built it another way (an independent nscf, a
# re-phasing) would show a large residual, and nothing here would hold.
#
# WHICH OPERAND MOVES.  The index tables — ``irr_idx_k``, ``sym_idx_k``
# and the row tables built from them — are ``n_k`` integers and stay on
# the host.  The array operand is ``(n_k, nb, nb)`` complex128, 9.2 GB at
# nk=144/nb=2000 and four calls per SC iteration, so every helper below
# dispatches on it: a ``jax.Array`` is gathered where it already is and
# keeps its sharding, a numpy array takes the numpy path unchanged
# (``gw.scissor.k_star_weights`` and the star gate pass host arrays in).
# The index tables are baked into the jit closure as constants, as
# :func:`_get_unfold_isdf_operator_jit` bakes the centroid tables (runtime-arg
# form measured ~2× slower per call).


_STAR_JIT_CACHE: dict = {}


def _cached_star_jit(key, build):
    """One compiled module per (index table, operand aval, layout)."""
    fn = _STAR_JIT_CACHE.get(key)
    if fn is None:
        fn = build()
        _STAR_JIT_CACHE[key] = fn
    return fn


def _jit_with(fn, out_sh):
    """``jax.jit(fn)``, pinning the output sharding only when there is one.

    In jax 0.9 ``out_shardings=None`` is not the same as omitting the
    argument (``pjit._parse_jit_arguments`` keeps ``None`` as a leaf,
    distinct from ``UnspecifiedValue``), so the two cases are two calls.
    """
    return jax.jit(fn) if out_sh is None else jax.jit(fn, out_shardings=out_sh)


def _row_out_sharding(A):
    """``A``'s own sharding, when a row gather cannot invalidate it.

    A take along axis 0 leaves the trailing axes alone, so the operand's
    spec is still valid for the result PROVIDED axis 0 is replicated — it
    is for every SC operand (U and Σ are ``P(None, 'x', 'y')`` from
    ``qsgw_density.band_rotation_spec``, E is ``P(None, None)``).  If axis
    0 were mesh-sharded the new k extent need not divide that mesh axis,
    so return None and leave the layout to GSPMD.
    """
    sh = getattr(A, "sharding", None)
    if isinstance(sh, NamedSharding) and (
            len(sh.spec) == 0 or sh.spec[0] is None):
        return sh
    return None


def _scalar_out_sharding(A):
    """Replicated ``P()`` on ``A``'s mesh — for the spread's 2-vector.

    Pinned rather than inferred so the result is fully replicated on every
    rank: that is what makes the single ``np.asarray`` below legal at P>1,
    since a partially-addressable array raises "spans non-addressable
    devices".
    """
    sh = getattr(A, "sharding", None)
    if isinstance(sh, NamedSharding):
        return NamedSharding(sh.mesh, P())
    return None


def _star_row_order(irr_idx_k):
    """``(rows, labels)`` — the ONE IBZ row order both directions use.

    ``rows[j]`` is the full-BZ row :func:`star_select` keeps for star
    ``j`` — the FIRST occurrence of each star label, in full-BZ order —
    and ``labels[j] = irr_idx_k[rows[j]]``.

    The two directions agree only if first-occurrence order is ASCENDING
    IN THE LABEL: :func:`star_select` orders rows by first occurrence,
    :func:`star_broadcast` addresses ``A_irr`` by position in
    ``np.unique(irr)``.  Checked here rather than assumed, because if it
    failed ``broadcast`` would silently return a DIFFERENT star's matrix
    at every affected k.
    """
    irr = np.asarray(irr_idx_k)
    _, first = np.unique(irr, return_index=True)
    rows = np.sort(first).astype(np.int32)
    labels = np.asarray(irr[rows])
    return rows, labels


def _take_rows(A, rows):
    """``A[rows]`` along axis 0, evaluated where ``A`` already is."""
    if not isinstance(A, jax.Array):
        return np.asarray(A)[rows]
    out_sh = _row_out_sharding(A)
    key = ("take", rows.tobytes(), tuple(int(s) for s in A.shape),
           A.dtype, repr(out_sh))

    def _build():
        rows_j = jnp.asarray(rows)

        def _f(x):
            return x[rows_j]
        return _jit_with(_f, out_sh)

    return _cached_star_jit(key, _build)(A)


def _broadcast_rows(A_irr, take, trs):
    """``A_irr[take]`` with ``conj`` on the rows ``trs`` marks."""
    if not isinstance(A_irr, jax.Array):
        out = np.asarray(A_irr)[take]
        return apply_band_matrix_symmetry(out, antiunitary=trs)
    out_sh = _row_out_sharding(A_irr)
    ndim = int(A_irr.ndim)
    any_trs = bool(np.any(trs))
    key = ("bcast", take.tobytes(), trs.tobytes(),
           tuple(int(s) for s in A_irr.shape), A_irr.dtype, repr(out_sh))

    def _build():
        take_j = jnp.asarray(take)
        trs_j = jnp.asarray(np.asarray(trs).reshape((-1,) + (1,) * (ndim - 1)))

        def _f(x):
            out = x[take_j]
            if any_trs:
                out = apply_band_matrix_symmetry(out, antiunitary=trs_j)
            return out
        return _jit_with(_f, out_sh)

    return _cached_star_jit(key, _build)(A_irr)


def _star_conj_flags(irr_idx_k, sym_idx_k, n_sym_spatial):
    """``(ref_rows, conj)`` per full-BZ row: its star's reference row, and
    whether the value there must be CONJUGATED to give this row's value.

    THE REFERENCE IS A FULL-BZ ROW, NOT AN IBZ POINT.  Both directions of
    the star map address the row :func:`star_select` keeps — the first
    member of the star — and that row carries a ``sym_idx`` of its own,
    which can itself be a time-reversal row.  Θ is antiunitary, so
    ``O(−k) = conj(O(k))``; two rows that are both time-reversed images of
    the same IBZ point are therefore related to each other WITHOUT a
    conjugation, and a spatial row is related to a time-reversed reference
    WITH one.  The predicate is the XOR of the two TRS flags.

    Testing the member's flag alone — which these helpers used to do —
    inverts the rule for every star whose first member is a time-reversal
    row.  That costs nothing while every star begins on a spatial row, and
    silently conjugates (or fails to conjugate) whole stars as soon as one
    does not: the norm, hermiticity and the electron count all survive it,
    and :func:`star_spread` is the only thing that sees it.
    """
    irr = np.asarray(irr_idx_k)
    trs = np.asarray(sym_idx_k) >= int(n_sym_spatial)
    rows, labels = _star_row_order(irr)
    rep = {int(v): int(r) for v, r in zip(labels, rows)}
    ref_rows = np.array([rep[int(v)] for v in irr], dtype=np.int32)
    return ref_rows, np.asarray(trs ^ trs[ref_rows])


def _spread_tables(irr_idx_k, sym_idx_k, n_sym_spatial):
    """``(members, refs, conj)`` — :func:`star_spread`'s comparison set.

    ``members`` are the rows compared — every row after the first in its
    star, so singleton stars are absent — ``refs[i]`` is the first row of
    ``members[i]``'s star, and ``conj[i]`` says ``members[i]`` is
    time-reversed relative to it.  Rows rather than a mask, so the device
    kernel gathers only what it compares: ``2 · n_members`` tiles with
    ``n_members = n_k − n_k_irr``, not ``2 · n_k``.
    """
    ref_all, conj_all = _star_conj_flags(irr_idx_k, sym_idx_k, n_sym_spatial)
    members = np.where(ref_all != np.arange(ref_all.shape[0]))[0].astype(
        np.int32)
    return members, ref_all[members], conj_all[members]


def _star_stats(A_full, members, refs, conj):
    """``(worst, scale)``: the star residual and ``max|A|`` it is relative to.

    Both reductions are in one compiled module, so a device operand costs
    one 16-byte transfer rather than a full host readback per number.  The
    only transient is the two ``n_members``-row gathers, which inherit the
    operand's sharding.
    """
    n_mem = int(members.shape[0])
    if not isinstance(A_full, jax.Array):
        A = np.asarray(A_full)
        scale = float(np.abs(A).max())
        if n_mem == 0:
            return 0.0, scale
        bshape = (-1,) + (1,) * (A.ndim - 1)
        gathered = A[refs]
        ref_v = apply_band_matrix_symmetry(
            gathered, antiunitary=conj.reshape(bshape))
        return float(np.abs(A[members] - ref_v).max()), scale

    out_sh = _scalar_out_sharding(A_full)
    ndim = int(A_full.ndim)
    key = ("spread", members.tobytes(), refs.tobytes(),
           np.asarray(conj).tobytes(), tuple(int(s) for s in A_full.shape),
           A_full.dtype, repr(out_sh))

    def _build():
        mem_j = jnp.asarray(members)
        ref_j = jnp.asarray(refs)
        conj_j = jnp.asarray(
            np.asarray(conj).reshape((-1,) + (1,) * (ndim - 1)))

        def _f(x):
            scale = jnp.max(jnp.abs(x))
            if n_mem == 0:
                return jnp.stack([jnp.zeros_like(scale), scale])
            gathered = x[ref_j]
            ref_v = apply_band_matrix_symmetry(
                gathered, antiunitary=conj_j)
            return jnp.stack([jnp.max(jnp.abs(x[mem_j] - ref_v)), scale])
        return _jit_with(_f, out_sh)

    vals = np.asarray(_cached_star_jit(key, _build)(A_full))
    return float(vals[0]), float(vals[1])


def star_select(A_full, irr_idx_k):
    """The IBZ rows of a band-index quantity: ``A[k̄]`` for each IBZ k̄.

    ``A_full`` is ``(n_k_full, ...)``; returns ``(n_k_irr, ...)`` plus the
    star labels of those rows.  Pure index selection — the first
    occurrence of each IBZ parent in ``irr_idx_k`` — so it is exact.  A
    device operand is gathered on the device and keeps its sharding.
    """
    rows, labels = _star_row_order(irr_idx_k)
    return _take_rows(A_full, rows), labels


def star_broadcast(A_irr, irr_idx_k, sym_idx_k, n_sym_spatial,
                   irr_labels=None, *, trs_reference):
    """Spread an IBZ band-index quantity over the full BZ.

    ``A_irr`` is ``(n_k_irr, ...)``; the result is ``(n_k_full, ...)``.
    A gather plus a CONJUGATION on the rows a predicate selects.  A
    device operand never leaves the device.

    WHICH PREDICATE DEPENDS ON WHAT ``A_irr``'s ROWS ARE, and there are
    two callers in this tree that answer differently.  Θ is antiunitary,
    so the conjugation applies iff the member and the row its value is
    COPIED FROM differ in TRS-ness — and that source row is not the same
    object in the two cases:

    ``trs_reference="star_row"`` (default) — ``A_irr`` is what
        :func:`star_select` returned, so row ``j`` is a FULL-BZ row (the
        first member of star ``j``) and carries a ``sym_idx`` of its own,
        which can itself be a time-reversal row.  Predicate: the XOR of
        the two TRS flags (:func:`_star_conj_flags`).

    ``trs_reference="ibz_slab"`` — ``A_irr`` is the file's own IBZ slab,
        read verbatim with NO symmetry operation applied, so its rows are
        untransformed by construction and their TRS flag is identically
        False.  Predicate: the member's OWN flag, ``sym_idx_k >=
        n_sym_spatial``.  ``gw.kin_ion_io.broadcast_ibz_to_full_bz`` is
        this case.

    THE TWO COINCIDE ONLY WHILE EVERY STAR'S FIRST FULL-BZ ROW IS
    SPATIAL, which is a property of the op-selection policy and not of
    the physics.  MEASURED on ``tests/regression/cohsex_debug`` with the
    shipping policy, where star label 2's first row carries sym_idx 12 =
    ntran (a pure time reversal): the two predicates disagree on 6 of 9
    k-points, and using the XOR on the IBZ slab conjugates ⟨m|V_H|n⟩'s
    OFF-DIAGONALS on those rows — 183.61 eV against a V_H computed
    independently at every full-BZ k, with the DIAGONAL left exactly
    intact, so the electron count, hermiticity, the spectrum and every
    diagonal observable survive it unchanged.

    ``sym_idx_k`` and ``n_sym_spatial`` are REQUIRED, not optional with an
    equality default: a caller that omitted them would get silently wrong
    matrices on every TRS pair (measured 3.6e-01 relative, job 7889235),
    and nothing downstream would notice.

    ``trs_reference`` IS REQUIRED TOO, AND HAS NO DEFAULT.  It used to
    default to ``"star_row"``, which is right for a ``star_select``
    operand and wrong for a file slab — and the failure it produces is
    the one nothing sees: 183.61 eV on the off-diagonals with the REAL
    DIAGONAL EXACTLY INTACT, so the electron count, hermiticity, the
    spectrum, the eqp.dat V_H column and the diagonal star-spread metric
    all survive it.  A default is exactly the wrong shape for a choice
    whose wrong branch is invisible to every cheap check, so there is
    none: a caller that has not thought about which flavour its operand
    is gets a ``TypeError`` at the call site instead of a plausible wrong
    matrix.  ``KStarMap.broadcast`` does not take the argument because
    its operand can only have come from :meth:`KStarMap.select`, which
    fixes the answer to ``"star_row"`` by construction.
    """
    irr = np.asarray(irr_idx_k)
    sidx = np.asarray(sym_idx_k)
    # THE SAME ROW ORDER ``star_select`` USES, not ``np.unique``.  Deriving
    # it here rather than re-sorting is what makes the round trip exact for
    # ANY ordering: both directions address ``A_irr`` by position in one
    # array.  ``np.unique`` is ascending in the label, which coincides with
    # first-occurrence order on most k-maps and not on all of them --
    # ``tests/regression/gnppm_debug`` gives labels [0, 2, 6, 8, 7] against
    # a sorted [0, 2, 6, 7, 8], swapping the last two -- and where they
    # differ the broadcast returned another star's matrix.  (That table is
    # gnppm's own, re-derived from ``SymMaps(wfn)`` on 2026-08-07 and
    # committed as ``tests/data/star_tables_e9340d1.json``; it was
    # attributed to mos2_4x4 here until then, which is a deck this tree
    # carries no fixture for.)
    labels = (_star_row_order(irr)[1] if irr_labels is None
              else np.asarray(irr_labels))
    pos = {int(v): i for i, v in enumerate(labels)}
    take = np.array([pos[int(v)] for v in irr], dtype=np.int32)
    # The conjugation predicate follows ``trs_reference`` — see the
    # docstring.  Neither branch is a default the other can be folded
    # into: they disagree on real decks.
    if trs_reference == "star_row":
        _, conj = _star_conj_flags(irr, sidx, n_sym_spatial)
    elif trs_reference == "ibz_slab":
        conj = np.asarray(sidx) >= int(n_sym_spatial)
    else:
        raise ValueError(
            "star_broadcast: trs_reference must be 'star_row' (A_irr came "
            "from star_select, rows are full-BZ) or 'ibz_slab' (A_irr is "
            f"the untransformed IBZ slab); got {trs_reference!r}.")
    return _broadcast_rows(A_irr, take, conj)


# ─────────────────────────────────────────────────────────────────────────
# THE TWO NAMED UNFOLDS.  Two operations, ONE backend.
# ─────────────────────────────────────────────────────────────────────────
# There are two different IBZs in this tree and they are not the same size
# (docs/architecture/symmetry_register.md, "THERE ARE TWO DIFFERENT IBZs"):
#
#   file wedge  — ``wfn.kpoints``, length ``sym.nk_red``, addressed by
#                 ``kirr_fullids``.  What every .dat output is indexed by
#                 and what BerkeleyGW means by the IBZ.
#   star wedge  — what :func:`star_select` keeps, one row per orbit.
#
# MEASURED: they coincide on ``si_cohsex_debug`` (8 = 8) and diverge on
# ``cohsex_debug`` (4 vs 3) and ``gnppm_debug`` (9 vs 5).  A single
# ``unfold_ibz_to_full_bz`` would therefore be correct on the deck most
# gates run and silently wrong on the others — so there are two names, and
# the call site says which without the reader needing to know what
# ``trs_reference`` means.
#
# They are thin wrappers over ONE backend (:func:`star_broadcast`), which
# is what keeps a future spinor-rotation or bispinor upgrade a one-place
# change.  Resist making this four: a THIRD unfold means the
# parameterisation is wrong, not that another name is needed.

def star_tables_of(sym):
    """``(irr_idx_k, sym_idx_k, n_sym_spatial)`` off a live ``SymMaps``.

    ``n_sym_spatial`` is derived from ``sym_mats_k`` (always ``2·ntran``
    in both SymMaps branches) rather than read from the WFN header,
    because that is the derivation :func:`unfold_psi` uses to decide which
    rows get conjugated when it BUILDS ψ(Sk).  Reading it from the header
    instead lets the producer and the consumer of that convention drift.

    PUBLIC because a WRITER needs it too, not only the unfolds below.  A
    file that stores a wedge has to file the reconstruction tables beside
    the arrays (``kin_ion.h5``'s ``irr_idx_k``/``sym_idx_k``, and now
    ``qp_wfn_rotations.h5``'s), and the alternative to exporting this is
    every writer re-spelling the ``// 2`` — which is precisely the
    header-vs-``sym_mats_k`` drift the paragraph above exists to prevent.
    """
    return (np.asarray(sym.irr_idx_k, dtype=np.int32),
            np.asarray(sym.sym_idx_k, dtype=np.int32),
            int(np.asarray(sym.sym_mats_k).shape[0]) // 2)


#: The private spelling this function had while it was unfold-only.  Kept
#: as an alias so the in-module call sites below read unchanged.
_star_tables_of = star_tables_of


def unfold_file_wedge_to_full_bz(sym, data):
    """FILE wedge → full BZ.  ``(sym.nk_red, …)`` → ``(sym.nk_tot, …)``.

    The wedge as the WFN stores it — ``wfn.kpoints``, the k-set every
    ``.dat`` output is indexed by and what BerkeleyGW calls the IBZ.  Use
    this for anything read off disk in that indexing: ``eqp{0,1}.dat``,
    ``sigma_diag.dat``, a ``kin_ion.h5`` slab.

    Takes the ``SymMaps`` itself, not index tables: the tables are the
    service's business and a driver that holds one has already lost the
    abstraction.

    ``irr_labels`` IS THE WHOLE DIFFERENCE BETWEEN THE TWO WEDGES, and
    omitting it is what made this function a different operation from the
    production reader.  ``star_broadcast`` with no labels addresses
    ``data`` by POSITION AMONG DISTINCT STAR LABELS — right for a star
    wedge, and right here only while the WFN's k-set IS that wedge.  The
    rows of a FILE wedge are ``wfn.kpoints`` rows, so the labels are the
    identity and the gather must be ``data[irr_idx_k]`` — which is
    exactly what ``src/file_io/kin_ion.py``'s
    ``broadcast_ibz_to_full_bz`` passes.  MEASURED 2026-08-17 on a random
    operand, this function against that one: ``gnppm_debug`` (nk_red 9,
    orbits 5) 3.82e+00, ``bispinor_debug`` (9, 5) 5.18e+00,
    ``cohsex_debug`` (4, 3) 3.91e+00, and exactly 0.0 on
    ``si_cohsex_debug`` / ``si_bse_debug`` / ``hbn_cohsex_debug``, where
    ``nk_red == n_orbits``.  That partition is precisely the one a
    "31.05 / 12.44 / 8.04 Ry star-relation failure" was reported over —
    a real measurement taken with a broken instrument.
    :func:`unfold_star_wedge_to_full_bz` below is the one that WANTS the
    derived labels, and it says so by not passing any.
    """
    irr, sidx, nss = _star_tables_of(sym)
    n_rows = int(np.shape(data)[0])
    if n_rows != int(sym.nk_red):
        raise ValueError(
            f"unfold_file_wedge_to_full_bz: operand has {n_rows} rows but "
            f"the FILE wedge is sym.nk_red = {int(sym.nk_red)}.  A "
            f"{len(set(int(v) for v in irr))}-row operand is the STAR "
            f"wedge — call unfold_star_wedge_to_full_bz for that one; the "
            f"two are different functions wherever the two wedges differ.")
    return star_broadcast(data, irr, sidx, nss,
                          irr_labels=np.arange(n_rows, dtype=np.int32),
                          trs_reference="ibz_slab")


def unfold_file_wedge_polar_matrix(sym, data, *, component_axis=-3):
    """FILE-wedge polar band matrix → full BZ, on the input's backend.

    This is the canonical route for a cheap same-k observable such as
    ``<m k|v_i|n k>``.  The stored rows have no operation applied.  The
    ordinary file-wedge unfold first gathers their band matrices and applies
    antiunitary conjugation.  The target row's forward Cartesian action then
    mixes the explicit polar-vector axis.  Its antiunitary half already
    contains the time-odd minus sign, so no second velocity-parity rule is
    accepted here.

    Nonsymmorphic translation phases cancel between the equal-k bra and ket.
    Quantities with distinct endpoints must instead use the directed-edge
    service, whose endpoint sewing carries those phases.

    ``data`` has shape ``(sym.nk_red, ..., 3, nb, nb)`` by default.  A NumPy
    input stays on the host; a JAX input stays on device and keeps its band
    sharding.
    """
    out = unfold_file_wedge_to_full_bz(sym, data)
    sym_rows = np.asarray(sym.sym_idx_k, dtype=np.int32)
    rotations = np.asarray(sym.cartesian_action(
        sym_rows, axial=False, time_odd=True), dtype=np.float64)
    if sym_rows.shape != (int(sym.nk_tot),):
        raise ValueError(
            "unfold_file_wedge_polar_matrix: sym.sym_idx_k must have one "
            f"row per full k-point; got {sym_rows.shape}, expected "
            f"({int(sym.nk_tot)},).")
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or rotations.shape[0] != sym_rows.shape[0]):
        raise ValueError(
            "unfold_file_wedge_polar_matrix: invalid canonical symmetry-row "
            f"map {sym_rows.shape} for Cartesian actions {rotations.shape}.")
    return apply_band_matrix_symmetry(
        out,
        component_mix=rotations,
        component_axis=component_axis,
    )


def reduce_full_bz_to_file_wedge(sym, data):
    """full BZ → FILE wedge.  ``(sym.nk_tot, …)`` → ``(sym.nk_red, …)``.

    Selects the rows that ARE ``wfn.kpoints`` — the k-set every ``.dat``
    output is indexed by — so a writer can reduce without ever holding
    ``kirr_fullids``.  Pure row selection: no conjugation, no symmetry
    operation, nothing to get the wrong way round.

    NOT THE EXACT INVERSE of :func:`unfold_file_wedge_to_full_bz`, and the
    asymmetry is real rather than an oversight.  This keeps one row per
    STORED k; the unfold rebuilds every full-BZ k from its ORBIT PARENT.
    Where the WFN carries two k in the same orbit — ``cohsex_debug``, where
    file wedge row 1 is the time-reverse of row 2 — the round trip
    reduce→unfold replaces row 1's own stored values with ``conj`` of row
    2's.  Self-consistent, and correct if the two really are TRS partners,
    but it is not the identity and must not be assumed to be.

    There is deliberately no ``reduce_full_bz_to_star_wedge``: that is
    :func:`star_select`, which already exists and returns the labels the
    star round trip needs.
    """
    rows = np.asarray(sym.kirr_fullids, dtype=np.int32)
    return _take_rows(data, rows)


def unfold_star_wedge_to_full_bz(sym, data):
    """STAR wedge → full BZ.  ``(n_orbits, …)`` → ``(sym.nk_tot, …)``.

    The wedge :func:`star_select` produces — one row per symmetry orbit,
    each row a FULL-BZ row carrying a ``sym_idx`` of its own.  Use this
    for the round trip ``unfold_star_wedge_to_full_bz(sym,
    star_select(A_full, sym.irr_idx_k)[0])``, which is the identity.

    NOT interchangeable with :func:`unfold_file_wedge_to_full_bz`: the two
    wedges differ in LENGTH on two of the three committed decks, and
    coincide on the third.
    """
    irr, sidx, nss = _star_tables_of(sym)
    return star_broadcast(data, irr, sidx, nss, trs_reference="star_row")


def star_spread(A_full, irr_idx_k, sym_idx_k, n_sym_spatial):
    """max residual of ``A_full`` against its own star, by the right rule.

    Compares each member to its star's first member — directly when the
    two lie on the same side of time reversal, against the CONJUGATE when
    they do not (:func:`_star_conj_flags`).  Zero up
    to round-off iff the full-BZ basis really is the unfolded IBZ one and
    the operator commutes with the symmetry.

    Cheap, and the only thing that catches a gauge mismatch introduced
    upstream: hermiticity, the norm and the electron count all survive
    one.  Callers that also want the scale should use
    :meth:`KStarMap.spread_rel`, which gets both from one reduction.
    """
    return _star_stats(A_full, *_spread_tables(
        irr_idx_k, sym_idx_k, n_sym_spatial))[0]


class KStarMap:
    """IBZ ⇄ full BZ for BAND-INDEX quantities.  Pass one to the SC loop.

    Bundles the three index arrays that always travel together
    (``irr_idx_k``, ``sym_idx_k``, ``ntran``) so no call site can supply
    two of the three — omitting ``sym_idx_k`` is the failure that returns
    the wrong matrix on every time-reversed star and is invisible to
    hermiticity, the norm and the electron count (job 7889235: 3.6e-01
    relative).

    :meth:`identity` is the no-reduction map, so a driver written against
    this reads the same whether or not symmetry is in use, and the
    symmetry-off path stays byte-identical rather than becoming a
    separate branch.

    See the module note above :func:`star_select` for WHY this is an index
    map and not an unfold: the loader builds full-BZ ψ by unfolding in
    G-space, so a band index is symmetry-inert.

    The row tables are pure functions of ``irr_idx``/``sym_idx``, so they
    are built once in ``__init__`` as host int arrays and the per-call
    work is one gather.
    """

    __slots__ = ("irr_idx", "sym_idx", "n_sym_spatial", "labels",
                 "_rows", "_take", "_conj_bcast", "_members", "_refs",
                 "_conj")

    def __init__(self, irr_idx, sym_idx, n_sym_spatial, labels=None):
        self.irr_idx = np.asarray(irr_idx, dtype=np.int32)
        self.sym_idx = np.asarray(sym_idx, dtype=np.int32)
        self.n_sym_spatial = int(n_sym_spatial)
        if self.sym_idx.shape != self.irr_idx.shape:
            raise ValueError(
                f"KStarMap: irr_idx {self.irr_idx.shape} and sym_idx "
                f"{self.sym_idx.shape} must both be (n_k_full,)")
        self._rows, row_labels = _star_row_order(self.irr_idx)
        self.labels = (row_labels if labels is None
                       else np.asarray(labels, dtype=np.int32))
        pos = {int(v): i for i, v in enumerate(self.labels)}
        self._take = np.array([pos[int(v)] for v in self.irr_idx],
                              dtype=np.int32)
        # Broadcast conjugates RELATIVE to the star's kept row, so this
        # is not ``sym_idx >= n_sym_spatial`` — see ``_star_conj_flags``.
        _, self._conj_bcast = _star_conj_flags(
            self.irr_idx, self.sym_idx, self.n_sym_spatial)
        self._members, self._refs, self._conj = _spread_tables(
            self.irr_idx, self.sym_idx, self.n_sym_spatial)

    @classmethod
    def from_sym(cls, sym, n_sym_spatial) -> "KStarMap":
        """From a :class:`SymMaps` — ``irr_idx_k`` / ``sym_idx_k``."""
        return cls(sym.irr_idx_k, sym.sym_idx_k, n_sym_spatial)

    @classmethod
    def identity(cls, n_k) -> "KStarMap":
        """Every k its own star: ``select`` and ``broadcast`` are no-ops."""
        idx = np.arange(int(n_k), dtype=np.int32)
        return cls(idx, np.zeros(int(n_k), dtype=np.int32), 1, idx)

    @property
    def nk_full(self) -> int:
        return int(self.irr_idx.shape[0])

    @property
    def nk_irr(self) -> int:
        return int(self.labels.shape[0])

    @property
    def reduction(self) -> float:
        return self.nk_full / max(self.nk_irr, 1)

    @property
    def is_identity(self) -> bool:
        return self.nk_irr == self.nk_full

    def select(self, A_full):
        """``(n_k_full, …)`` → ``(n_k_irr, …)``.  Index selection.

        A ``jax.Array`` stays on the device and keeps its sharding; a
        numpy array stays on the host (``gw.scissor.k_star_weights``).
        """
        return _take_rows(A_full, self._rows)

    def broadcast(self, A_irr):
        """``(n_k_irr, …)`` → ``(n_k_full, …)``, conjugating the members
        whose TRS-ness differs from their star's kept row."""
        return _broadcast_rows(A_irr, self._take, self._conj_bcast)

    def spread(self, A_full) -> float:
        """Residual of ``A_full`` against its own stars; see :func:`star_spread`.

        Σ is built on the full BZ and then selected to the IBZ, so this is
        free evidence that the two k-sets agree — and the only check that
        catches a gauge mismatch introduced upstream.
        """
        return _star_stats(A_full, self._members, self._refs,
                           self._conj)[0]

    def spread_rel(self, A_full) -> float:
        """:meth:`spread` divided by ``max|A_full|`` — what callers print.

        One reduction and one 16-byte transfer for a device operand; the
        two-call form (``spread`` then a separate ``max``) was two full
        host readbacks.  The floor on the scale avoids a divide by zero;
        on a zero operand the residual is zero too, so the ratio is 0.
        """
        worst, scale = _star_stats(A_full, self._members, self._refs,
                                   self._conj)
        return worst / max(scale, 1e-300)

    def __repr__(self) -> str:
        return (f"KStarMap(nk_full={self.nk_full}, nk_irr={self.nk_irr}, "
                f"reduction={self.reduction:.2f}x, "
                f"n_sym_spatial={self.n_sym_spatial})")


# ─────────────────────────────────────────────────────────────────────────
# Pre-sweep spellings — MODULE-LEVEL half of the compat layer
# ─────────────────────────────────────────────────────────────────────────
# The package door binds these too (``symmetry_maps/__init__.py``).  Both
# levels are bound on purpose: a consumer that reaches this module
# directly (the wave-1 shims did, and the sibling stamp branch still
# calls the old names) must not care which door it came through.
# Retirement: see :mod:`symmetry_maps._compat`.

#: DEPRECATED — :func:`unfold_isdf_operator`.
unfold_v_q = deprecated_alias(unfold_isdf_operator, "unfold_v_q")
#: DEPRECATED — :func:`mix_channels_by_proper_rotation`.
unfold_v_q_bispinor_lorentz = deprecated_alias(
    mix_channels_by_proper_rotation, "unfold_v_q_bispinor_lorentz")
#: DEPRECATED — :func:`spinor_rotation_for_sym_row`.
trs_augment_U = deprecated_alias(
    spinor_rotation_for_sym_row, "trs_augment_U")
