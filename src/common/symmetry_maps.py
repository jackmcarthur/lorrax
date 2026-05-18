# Note (fractional translations): the public wfn-unfold helpers
# ``get_gvecs_kfull`` / ``get_cnk_fullzone[_batch]`` lived here until
# P5; that whole pipeline now lives inside
# ``file_io.wfn_loader.WfnLoader`` (eager + phdf5 backends, both
# applying U_spinor + τ phase + TR conjugation in one place).  This
# module retains the sym-table construction (kpoint_map, R_grid,
# unfolded_kpts, …) and the kfull-symmap / q-IBZ helpers used by the
# GW driver.
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from file_io import WfnLoader as WFNReader


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


def slice_q_full_to_ibz(arr_full, q_irr_full_idx, *, out_sharding=None):
    """Slice a ``(n_q_full, ...)`` array to its IBZ rows.

    The natural ``full BZ → IBZ`` companion to :func:`unfold_v_q`'s
    ``IBZ → full BZ`` direction: this just picks the IBZ representative
    q-points out of a full-BZ tensor.  No centroid permute, no L-phase,
    no TRS conjugation — a pure row gather along axis 0.

    Use it whenever a q-axis quantity is built at full BZ but only the
    IBZ rows are needed for the downstream per-q step.  Two examples
    on the same shape ``(n_q, n_rmu, n_rmu)`` sharded as
    ``P(None, 'x', 'y')`` (q-axis replicated, μ on x, ν on y):

    - ``isdf_fitting.fit_zeta_to_h5``: slice C_q before ``factor_c_q``
      so Cholesky / LU runs only on the IBZ q-block, then ζ_q is solved
      and stored at IBZ; downstream V_q unfolds via :func:`unfold_v_q`.
    - W_q = ``(1 − v_q χ_q)^{-1} v_q``: slice the Hermitian object that
      needs per-q inversion to IBZ before solve, then unfold via
      :func:`unfold_v_q` for the q-axis consumers.

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


def unfold_v_q(
    V_q_ibz,
    *,
    irr_idx,
    sym_idx,
    sym_perm,
    L_table,
    q_irr_frac,
    mesh_xy,
    n_sym_spatial,
):
    """Expand ``V_q_ibz`` over the IBZ to the full BZ.

    The mapping is a centroid-axis double-gather (using the **source-map**
    ``α(μ) = sym_perm[s, μ]`` returned by ``compute_centroid_sym_perm``)
    plus a per-centroid umklapp phase from the real-space lattice wrap:

        V_full[q, μ', ν'] = exp(2π i q_irr · (L_{s,μ'} − L_{s,ν'}))
                            · V_ibz[i(q), α_{s}(μ'), α_{s}(ν')]

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
    non-Hermitian channels).  The centroid permutation itself is
    unchanged under TRS (r is fixed); ``sym_perm`` rows ``[ntran:]``
    duplicate ``[:ntran]``.  Callers build ``sym_perm`` via
    ``compute_centroid_sym_perm(..., extend_trs=True)`` and pass
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
        ``compute_centroid_sym_perm``.  Drives the umklapp phase.
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

    Returns
    -------
    V_q_full
        ``(n_q_full, n_rmu, n_rmu)`` complex, sharded ``P(None,'x','y')``.
    """
    # Trivial-IBZ short-circuit. When ntran=1 (e.g. nosym runs) the IBZ is
    # already the full BZ — irr_idx is identity, sym_idx is all zeros,
    # sym_perm is identity. The take_along_axis path below is then a
    # no-op but its sharded codegen has been observed to trip an XLA HLO
    # verifier dtype mismatch (s64 broadcast vs s32 operand on a 2×2
    # mesh), so bypass it entirely.
    idx_np = np.asarray(irr_idx)
    sym_np = np.asarray(sym_idx)
    if (idx_np.shape[0] == int(V_q_ibz.shape[0])
            and np.array_equal(idx_np, np.arange(idx_np.shape[0]))
            and np.all(sym_np == 0)):
        return V_q_ibz

    n_sym_perm = int(np.asarray(sym_perm).shape[0])
    max_sym = int(sym_np.max()) if sym_np.size else -1
    if max_sym >= n_sym_perm:
        raise ValueError(
            f"unfold_v_q: sym_idx contains value {max_sym} but sym_perm "
            f"has only {n_sym_perm} rows.  Build sym_perm via "
            f"``compute_centroid_sym_perm(..., extend_trs=True)`` so it "
            f"covers the TRS-augmented half of ``sym_mats_k``.")
    trs_used = max_sym >= int(n_sym_spatial)
    if trs_used and int(n_sym_spatial) * 2 != n_sym_perm:
        raise ValueError(
            f"unfold_v_q: sym_idx uses TRS-augmented rows (max={max_sym} ≥ "
            f"n_sym_spatial={n_sym_spatial}) but sym_perm.shape[0]="
            f"{n_sym_perm} ≠ 2·n_sym_spatial.  Build sym_perm via "
            f"``compute_centroid_sym_perm(..., extend_trs=True)``.")

    # Forward permutation: gather V_ibz at indices sym_perm[s, μ'] — i.e.,
    # at the FORWARD image of each full-BZ centroid μ' under sym s.
    # Empirically (see ``reports/trs_sym_audit_2026-05-14/test_production_unfold_v_q.py``):
    # V_full[μ', ν'] = phase(μ', ν') · V_ibz[parent, sym_perm[s, μ'],
    # sym_perm[s, ν']] closes to ISDF noise floor on all 36 q's of the
    # CrI3 6×6 30 Ry dump including order-3 ops.  The prior code used
    # inv_perm = argsort(sym_perm) (= π⁻¹) which is a no-op for
    # involutive ops (MoS2 σ_h, Si cubic) but wrong for order-3 (CrI3
    # C3) — that was the silent 4 eV gap on hex systems.
    # Pad with identity rows for centroid-axis padding to mesh-divisible.
    fwd_perm = np.asarray(sym_perm, dtype=np.int32)
    n_rmu_logical = int(fwd_perm.shape[-1])
    n_rmu_padded = int(V_q_ibz.shape[-1])
    if n_rmu_padded > n_rmu_logical:
        pad_block = np.arange(n_rmu_logical, n_rmu_padded, dtype=np.int32)
        pad_block = np.broadcast_to(
            pad_block, (fwd_perm.shape[0], n_rmu_padded - n_rmu_logical))
        fwd_perm = np.concatenate([fwd_perm, pad_block], axis=-1)
    elif n_rmu_padded != n_rmu_logical:
        raise ValueError(
            f"unfold_v_q: V_q_ibz μ-extent {n_rmu_padded} smaller than "
            f"sym_perm logical extent {n_rmu_logical}; pad invariant "
            f"violated.")
    # Per-(q_full, μ) umklapp phase factor exp(2π i q_irr · L_μ).
    # ``L_table`` is shape (2·ntran, n_rmu, 3) int (any width); promote
    # to float64 then gather to (n_q_full, n_rmu, 3).  ``q_irr_frac`` is
    # (n_q_ibz, 3); we gather to (n_q_full, 3) via ``irr_idx``.  The
    # product qL = q_irr · L is (n_q_full, n_rmu), then the bilinear
    # phase is qL[μ] − qL[ν] (per-q, outer-diff).
    L_arr = np.asarray(L_table, dtype=np.float64)
    n_rmu_padded = int(V_q_ibz.shape[-1])
    if L_arr.shape[1] < n_rmu_padded:
        # Pad with zeros so the L axis matches the V μ-extent; padded
        # centroids have no umklapp wrap.
        pad = np.zeros(
            (L_arr.shape[0], n_rmu_padded - L_arr.shape[1], 3),
            dtype=np.float64)
        L_arr = np.concatenate([L_arr, pad], axis=1)

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
    fn = _get_unfold_v_q_jit(
        V_q_shape=tuple(int(s) for s in V_q_ibz.shape),
        fwd_perm_arr=fwd_perm,
        idx_arr=idx_arr,
        sym_arr=sym_arr,
        L_arr=L_arr,
        q_irr_arr=q_irr_arr,
        trs_mask_arr=trs_mask_arr,
        n_sym_spatial=int(n_sym_spatial),
        mesh_xy=mesh_xy)
    return fn(V_q_ibz)


_UNFOLD_V_Q_JIT_CACHE: dict = {}


def _get_unfold_v_q_jit(
    *, V_q_shape, fwd_perm_arr, idx_arr, sym_arr, L_arr, q_irr_arr,
    trs_mask_arr, n_sym_spatial, mesh_xy,
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
        idx_arr.tobytes(),
        sym_arr.tobytes(),
        L_arr.shape, L_arr.tobytes(),
        q_irr_arr.tobytes(),
        trs_mask_arr.tobytes(),
        int(n_sym_spatial),
        id(mesh_xy),
    )
    hit = _UNFOLD_V_Q_JIT_CACHE.get(key)
    if hit is not None:
        return hit

    # Promote to jax arrays once at trace-build time.  Closure capture
    # makes these constants in the compiled HLO.
    fwd_perm_j = jnp.asarray(fwd_perm_arr)
    idx_j = jnp.asarray(idx_arr)
    sym_j = jnp.asarray(sym_arr)
    L_j = jnp.asarray(L_arr)
    q_irr_j = jnp.asarray(q_irr_arr)
    trs_mask_j = jnp.asarray(trs_mask_arr)
    # Memory contract: never exceed 1× single-tile per rank.  Use
    # ``shard_map`` + ``lax.all_to_all`` to redistribute axes between
    # ranks volume-preservingly — at no point does any rank hold a
    # full μ or ν axis (which would be Px× or Py× the single-tile
    # memory).  The all_to_all calls split the OTHER big spatial axis
    # (ν during the μ-permute step, μ during the ν-permute step), so
    # this works for arbitrary Px·Py even when n_q < Px·Py.
    n_rmu_padded = int(V_q_shape[-1])
    Px = int(mesh_xy.shape['x'])
    Py = int(mesh_xy.shape['y'])
    if n_rmu_padded % (Px * Py) != 0:
        raise ValueError(
            f"unfold_v_q: n_rmu_padded={n_rmu_padded} must be divisible "
            f"by Px*Py={Px*Py} for the all_to_all redistribution.  The "
            f"μ-padding in Meta should already enforce this — check "
            f"that meta.n_rmu_padded is mesh-divisible.")
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    @partial(jax.jit, out_shardings=V_sh)
    def _do_unfold(V_ibz):
        @partial(shard_map, mesh=mesh_xy,
                 in_specs=P(None, 'x', 'y'),
                 out_specs=P(None, 'x', 'y'),
                 check_rep=False)
        def _kernel(V_ibz_local):
            # V_ibz_local: (n_q_ibz, μ/Px, ν/Py)
            perm_q = fwd_perm_j[sym_j]                      # (n_q_full, n_rmu)
            # Gather q axis (replicated → local concat via idx_j).
            V_at_irr = V_ibz_local[idx_j]                   # (n_q_full, μ/Px, ν/Py)

            # μ permute on 'x'.  all_to_all redistributes:
            #   split  ν (local, /Py)  → ν / (Py·Px)
            #   concat μ (/Px sharded) → full μ
            # Volume per rank: n_q · μ · ν / (Px·Py) — unchanged from 1× tile.
            # Required: ν/Py divisible by Px (ensured by n_rmu_padded mod (Px·Py)=0).
            if Px > 1:
                V_x = jax.lax.all_to_all(
                    V_at_irr, 'x', split_axis=2, concat_axis=1, tiled=True)
                # (n_q_full, μ, ν/(Px·Py))
                V_x_perm = jnp.take_along_axis(
                    V_x, perm_q[:, :, None], axis=1,
                    mode='promise_in_bounds')
                V_perm_mu = jax.lax.all_to_all(
                    V_x_perm, 'x', split_axis=1, concat_axis=2, tiled=True)
                # (n_q_full, μ/Px, ν/Py)  — back to canonical
            else:
                V_perm_mu = jnp.take_along_axis(
                    V_at_irr, perm_q[:, :, None], axis=1,
                    mode='promise_in_bounds')

            # ν permute on 'y'.  Mirror trick on the 'y' axis.
            # Required: μ/Px divisible by Py.
            if Py > 1:
                V_y = jax.lax.all_to_all(
                    V_perm_mu, 'y', split_axis=1, concat_axis=2, tiled=True)
                # (n_q_full, μ/(Px·Py), ν)
                V_y_perm = jnp.take_along_axis(
                    V_y, perm_q[:, None, :], axis=2,
                    mode='promise_in_bounds')
                V_full_local = jax.lax.all_to_all(
                    V_y_perm, 'y', split_axis=2, concat_axis=1, tiled=True)
                # (n_q_full, μ/Px, ν/Py)  — back to canonical
            else:
                V_full_local = jnp.take_along_axis(
                    V_perm_mu, perm_q[:, None, :], axis=2,
                    mode='promise_in_bounds')

            # Umklapp phase: exp(2π i q_irr · (L_μ − L_ν)).  L_μ here
            # is L_table[s(q), μ] — wrap of centroid μ under sym op
            # s(q) (NOT permuted).  See
            # ``reports/trs_sym_audit_2026-05-14/verify_umklapp_user_math.py``.
            # Phase tables are small (~n_q · n_rmu c128 bytes); compute
            # replicated and slice this rank's μ_local / ν_local extent.
            L_per_q = L_j[sym_j]                            # (n_q_full, n_rmu, 3)
            q_per_q = q_irr_j[idx_j]                        # (n_q_full, 3)
            qL = jnp.einsum('qi,qmi->qm', q_per_q, L_per_q) # (n_q_full, n_rmu)
            phase = jnp.exp(2j * jnp.pi * qL.astype(jnp.complex128))
            mu_local = n_rmu_padded // Px
            nu_local = n_rmu_padded // Py
            x_idx = jax.lax.axis_index('x')
            y_idx = jax.lax.axis_index('y')
            phase_mu = jax.lax.dynamic_slice_in_dim(
                phase, x_idx * mu_local, mu_local, axis=1)  # (n_q, μ/Px)
            phase_nu = jax.lax.dynamic_slice_in_dim(
                phase, y_idx * nu_local, nu_local, axis=1)  # (n_q, ν/Py)
            V_full_local = (phase_mu[:, :, None] * V_full_local
                            * jnp.conj(phase_nu)[:, None, :])

            # TRS rows: per-element rule
            # V_full[TRS-q, μ, ν] = conj(V_ibz[parent, μ, ν])
            # = V_ibz[parent, ν, μ] (by Hermiticity).
            V_full_local = jnp.where(
                trs_mask_j[:, None, None],
                jnp.conj(V_full_local), V_full_local)
            return V_full_local

        return _kernel(V_ibz)

    _UNFOLD_V_Q_JIT_CACHE[key] = _do_unfold
    return _do_unfold


def unfold_v_q_bispinor_lorentz(
    V_tt_per_channel,
    *,
    sym_idx,
    R_proper_table,
    mesh_xy,
):
    """Apply the 3-vector Lorentz mixing on the bispinor TT-block tiles.

    Operates on the (μ_L, ν_L) ∈ {1,2,3}² block of bispinor V_q tiles that
    have already been passed through :func:`unfold_v_q` (centroid
    double-permute + L-phase + TRS conj-wrap).  Implements the rule from
    ``reports/bispinor_ibz_2026-05-16/derivation.md`` §A5::

        V^{i,j}_mixed[q, μ, ν]
            = Σ_{α, β ∈ {1,2,3}} R_proper[s(q), i-1, α-1]
                                  · R_proper[s(q), j-1, β-1]
                                  · V^{α,β}_unfolded[q, μ, ν]

    where ``s(q) = sym_idx[q]`` and ``R_proper`` is the proper (det = +1)
    Cartesian rotation table on ``SymMaps``.  TRS rows reuse the spatial
    ``R_proper``: the σ-flip TRS sigma-sign on (μ_L, ν_L) ∈ {1,2,3}²
    factorises as (−1)·(−1) = +1 on every stored UNIQUE_TILE and is
    absorbed by the existing scalar ``unfold_v_q`` conj-wrap (derivation
    §A4).

    Parameters
    ----------
    V_tt_per_channel : dict[(i, j) -> jax.Array]
        Dict keyed by ``(i, j)`` with ``i, j ∈ {1, 2, 3}`` (9 entries).
        Each value is ``(n_q_full, μ, ν)`` complex128 already at full-BZ
        shape, sharded ``P(None, 'x', 'y')``.  Callers may pass the 6
        unique tiles + the 3 Hermitian-redundant tiles synthesised via
        ``conj(swapaxes(V[i,j], -1, -2))``.
    sym_idx : np.ndarray | jax.Array
        ``(n_q_full,)`` int — ``SymMaps.sym_idx_q`` (TRS-augmented).
    R_proper_table : np.ndarray
        ``(2·n_sym_spatial, 3, 3)`` float64 — ``SymMaps.R_proper``.
        Both spatial and TRS halves contain the same spatial ``R_proper``
        per the derivation.
    mesh_xy : jax.sharding.Mesh
        Device mesh (used to lock the output sharding).

    Returns
    -------
    dict[(i, j) -> jax.Array]
        Same keys, same shapes, sharded ``P(None, 'x', 'y')``.
    """
    sym_arr = np.asarray(sym_idx, dtype=np.int32)
    R_arr = np.asarray(R_proper_table, dtype=np.float64)
    if R_arr.ndim != 3 or R_arr.shape[1:] != (3, 3):
        raise ValueError(
            f"unfold_v_q_bispinor_lorentz: R_proper_table must have shape "
            f"(2·n_sym_spatial, 3, 3); got {R_arr.shape}.")
    # Per-q 3×3 mixer baked into the jit closure as a constant — same
    # caching pattern as ``unfold_v_q`` (small int + float table folded
    # into HLO).  ``R_per_q[q]`` ∈ R^{3×3} is the spatial rotation that
    # mixes the (1,2,3) Lorentz indices for full-BZ q.
    R_per_q = R_arr[sym_arr]                                # (n_q_full, 3, 3)

    # Build the 9×9 source array V_in[α, β] at full-BZ shape, contract,
    # write back into the same 6 unique slots (plus 3 redundants).
    keys_in = [(i, j) for i in (1, 2, 3) for j in (1, 2, 3)]
    for k in keys_in:
        if k not in V_tt_per_channel:
            raise ValueError(
                f"unfold_v_q_bispinor_lorentz: missing TT tile {k}; "
                f"caller must supply all 9 (i, j) ∈ {{1,2,3}}² "
                f"(use ``conj(swapaxes(.., -1, -2))`` to synthesise the "
                f"Hermitian-redundant entries).")
    sample = V_tt_per_channel[(1, 1)]
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    R_dev = jnp.asarray(R_per_q)                            # closure constant

    fn = _get_unfold_v_q_lorentz_jit(
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


_UNFOLD_V_Q_LORENTZ_JIT_CACHE: dict = {}


def _get_unfold_v_q_lorentz_jit(*, V_shape, R_per_q_arr, mesh_xy):
    """Cache the inner Lorentz-mix jit by (shape, R-table content).

    Same content-keyed caching strategy as :func:`_get_unfold_v_q_jit`:
    the R table is baked into the jit closure as a constant so XLA can
    fold it into the HLO.  The cache key is the bytes-hash of the
    table plus the V shape plus the mesh identity.
    """
    key = (
        V_shape,
        R_per_q_arr.shape, R_per_q_arr.tobytes(),
        id(mesh_xy),
    )
    hit = _UNFOLD_V_Q_LORENTZ_JIT_CACHE.get(key)
    if hit is not None:
        return hit

    R_per_q_j = jnp.asarray(R_per_q_arr)                    # (n_q_full, 3, 3)
    V_sh = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
    in_sh = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))

    @partial(jax.jit, in_shardings=in_sh, out_shardings=V_sh)
    def _do_mix(V_in):
        # V_in: (3, 3, n_q, μ, ν).  Derivation §A5 (in its own R_proper
        # convention ``R_deriv = A.T · inv(mtrx) · inv(A.T)``):
        #     V^{i,j} = Σ_{α,β} R_deriv^{i,α} · R_deriv^{j,β} · V^{α,β}.
        # The LIVE ``R_per_q`` here is ``R_LORRAX`` (``A.T · mtrx ·
        # inv(A.T)`` with the det-flip; see ``SymMaps.R_proper``
        # docstring).  For orthogonal mtrx ``R_LORRAX = R_deriv.T``
        # row-wise, so ``R_deriv^{i,α} = R_LORRAX^{α,i}`` and the
        # contraction becomes
        #     V^{i,j} = Σ R_LORRAX^{α,i}(q) · R_LORRAX^{β,j}(q) · V^{α,β}(q).
        # In einsum letters with R indexed [q, row, col]:
        return jnp.einsum(
            'qai,qbj,abqmn->ijqmn',
            R_per_q_j, R_per_q_j, V_in,
        )

    _UNFOLD_V_Q_LORENTZ_JIT_CACHE[key] = _do_mix
    return _do_mix


# i·σ_y (time-reversal spinor factor in the SOC convention T = iσ_y K).
_I_SIGMA_Y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)


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
        ``ns = 1`` for non-SOC (the spinor rotation is a no-op then), ``ns = 2``
        for SOC.
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
    n_sym_spatial = int(sym_mats_k.shape[0]) // 2
    is_trs = sym_idx >= n_sym_spatial
    s_spatial = sym_idx - n_sym_spatial if is_trs else sym_idx

    S_full = np.asarray(sym_mats_k[sym_idx], dtype=np.int32)
    cnk = np.asarray(cnk_kbar)
    g_bar = np.asarray(g_kbar)

    # τ-phase: uses sym_mats_k[sym_idx] (which already has ±S sign baked in
    # for TRS), and the spatial τ. For spatial: phase = exp(-i (S·G_kbar)·τ).
    # For TRS (sym_mats_k = -S): the same formula gives exp(+i (S·G_kbar)·τ),
    # which is the conj(spatial-phase) the TRS rule requires.
    tau = np.asarray(translations[s_spatial], dtype=np.float64)
    has_tau = bool(np.any(np.abs(tau) > 1e-12))
    if has_tau:
        rotated = (S_full @ g_bar.T).T.astype(np.float64)              # (ngk, 3)
        phase = np.exp(-1j * (rotated @ tau))                          # (ngk,)
    else:
        phase = None

    ns = int(cnk.shape[1])

    if is_trs:
        # TRS rule: ψ_full = iσ_y · conj(U_s · ψ_kbar · phase_spatial)
        #         = (iσ_y · conj(U_s)) · conj(ψ_kbar) · conj(phase_spatial)
        # We compute phase via sym_mats_k[TRS row]=-S so it equals
        # conj(phase_spatial) already. Apply conj on cnk first, THEN phase
        # (else the conj would re-invert the phase sign).
        cnk = np.conj(cnk)
        if phase is not None:
            cnk = cnk * phase[None, None, :]
        if ns == 1:
            # Non-SOC scalar: no spinor d.o.f.; TRS = complex conjugation only.
            # ``iσ_y`` only acts on a 2-component spinor — skip the einsum.
            return cnk
        U_eff = _I_SIGMA_Y @ np.conj(np.asarray(U_spinor_spatial[s_spatial]))
    else:
        if phase is not None:
            cnk = cnk * phase[None, None, :]
        if ns == 1:
            # Non-SOC scalar: spinor rotation is a no-op.
            return cnk
        U_eff = np.asarray(U_spinor_spatial[s_spatial])

    # Spinor rotation. ``U_eff`` is (ns, ns); for ns=1 we already returned above
    # so this einsum is guaranteed to be a 2×2 spinor mix.
    cnk = np.einsum("jk,nkl->njl", U_eff, cnk)
    return cnk


class SymMaps:
    def __init__(self, wfn):
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
        """
        # get symmetry matrices from wfn file
        try:
            ntran = int(getattr(wfn, 'ntran', 1))
        except Exception:
            ntran = 1
        if ntran <= 1:
            # Trivial identity-only symmetry path
            self.sym_matrices = np.eye(3, dtype=np.int32)[None, :, :]
            self.sym_mats_k = self.sym_matrices.transpose(0, 2, 1).copy()
            self.translations = np.zeros((1, 3), dtype=np.float64)

            # In no-symmetry case, unfolded grid equals irreducible grid
            self.unfolded_kpts = np.asarray(wfn.kpoints, dtype=float)
            self.kpoint_map = np.arange(self.unfolded_kpts.shape[0], dtype=np.int32)

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
            self.R_cart = self.R_grid.astype(float)
            self.U_spinor = np.eye(2, dtype=complex)[None, :, :]
            # ``R_proper`` is the proper (det = +1) Cartesian rotation used
            # by the bispinor 3-vector vertex mixing.  Identity case: a
            # single 3×3 identity is its own proper part.  See
            # ``reports/bispinor_ibz_2026-05-16/derivation.md`` §A2.
            self.R_proper = np.eye(3, dtype=np.float64)[None, :, :]

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

        # BGW convention: `mtrx` (= `sym_matrices` here) acts on G-vectors
        # in column form: `G' = mtrx @ G`. For real-space coords the
        # corresponding action uses `Rinv = inv(mtrx)`: `r' = Rinv @ r + τ`
        # (see centroid/orbit_syms.compute_centroid_sym_perm at line 285,
        # and BerkeleyGW/Common/symmetries.f90:189 which stores mtrx as
        # invert(mtrx_inv) where mtrx_inv is the real-space rotation).
        self.sym_matrices = wfn.sym_matrices[:wfn.ntran]
        self.sym_mats_k = self.sym_matrices[:wfn.ntran].transpose(0,2,1).copy()  # these apply to k-points as sym_mats_k[i] @ [kx,ky,kz]
        # BGW non-symmorphic translations (``tnp``).  Carried so
        # downstream callers (orbit-aware centroid sym perm, q-IBZ
        # unfold) don't need to re-thread the WFNReader through every
        # API.  Slice to ``[:ntran]`` to match ``sym_matrices`` —
        # legacy WFN files pad ``tnp`` to length 48.
        self.translations = np.asarray(
            wfn.translations[:wfn.ntran], dtype=np.float64)
        
        # Add time-reversal symmetry (k → -k) combined with each spatial symmetry
        # This is needed because QE uses time-reversal to reduce k-points, but doesn't
        # store it as one of the ntran symmetries
        time_reversal_syms = -self.sym_mats_k  # S @ k -> -S @ k
        self.sym_mats_k = np.concatenate([self.sym_mats_k, time_reversal_syms], axis=0)

        # get the list of full zone k-points and the map from k_full to k_irr
        self.kpoint_map, self.unfolded_kpts = self.create_kpoint_symmetry_map(wfn)
        self.kpoint_map = np.asarray(self.kpoint_map, dtype=np.int32)
        if np.any(self.kpoint_map < 0) or np.any(self.kpoint_map >= wfn.nkpts):
            raise ValueError(
                "kpoint_map contains entries outside the irreducible-k range: "
                f"[0, {wfn.nkpts})"
            )

        self.irr_idx_k, self.sym_idx_k = self.find_symmetry_ops_simple(wfn, self.kpoint_map, self.unfolded_kpts)


        self.nk_tot = int(self.unfolded_kpts.shape[0])
        self.nk_red = int(wfn.nkpts)

        # Create mapping from irreducible k-points to full BZ indices
        self.kirr_fullids = np.zeros(self.nk_red, dtype=np.int32)
        for kirr in range(self.nk_red):
            matches = np.where(self.irr_idx_k == kirr)[0]
            if matches.size == 0:
                # Fallback: identity mapping if not found
                self.kirr_fullids[kirr] = kirr
            else:
                self.kirr_fullids[kirr] = matches[0]

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
        self.R_cart = self.syms_crystal_to_cartesian(wfn)
        self.U_spinor = self.get_spinor_rotations(wfn, self.R_cart[:wfn.ntran])
        # ``R_proper[s]`` is the PROPER (det=+1) Cartesian rotation that
        # mixes the bispinor 3-vector vertices ``γ̃^{1,2,3} = (σ_x, σ_y,
        # σ_z)`` per the SO(3) image of ``U_spinor``'s SU(2) sandwich:
        #   ``U_spinor[s]† σ^i U_spinor[s] = Σ_j R_proper[s]^{j,i} σ^j``.
        # This is the SAME rotation matrix that ``get_spinor_rotations``
        # consumes (``A · mtrx · A⁻¹`` with the det-flip — see
        # ``syms_crystal_to_cartesian`` docstring), so ``R_proper`` is
        # just ``R_cart`` with the same proper-flip that
        # ``get_spinor_rotations`` applies internally at line 1069.
        # Derivation: ``reports/bispinor_ibz_2026-05-16/derivation.md``
        # §A2; mixing rule §A5
        #   ``V^{i,j}_full[q] = R_proper^{i,α}(s) · R_proper^{j,β}(s) ·
        #                       V^{α,β}_unfolded[q]``.
        #
        # NOTE: this differs from the OFFLINE fixture
        # ``reports/bispinor_ibz_2026-05-16/cri3_R_proper.npz`` by a
        # transpose on every row — the fixture follows the derivation
        # TEXT (``O = A · U · A⁻¹``, ``U = mtrx⁻¹``), while the live
        # code follows the σ-sandwich identity that LORRAX's actual
        # ``U_spinor`` satisfies (built from ``A · mtrx · A⁻¹``).  The
        # two are inverses of each other for orthogonal mtrx and pick
        # up a transpose on the (i, j) indices of the §A5 formula.
        #
        # TRS half reuses the SPATIAL R_proper (NOT ``−R_spatial``): the
        # σ-flip TRS sigma-sign on the (μ_L, ν_L) ∈ {1,2,3}² block
        # factorises as (−1)·(−1) = +1 on every stored UNIQUE_TILE and
        # is absorbed by the existing ``unfold_v_q`` conj-wrap.  See
        # derivation §A4.
        _R_spatial = np.asarray(self.R_cart[:wfn.ntran], dtype=np.float64)
        _R_proper_spatial = np.where(
            np.linalg.det(_R_spatial)[:, None, None] < 0,
            -_R_spatial, _R_spatial)
        self.R_proper = np.concatenate(
            [_R_proper_spatial, _R_proper_spatial], axis=0)
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
        self.all_unfolded_qpts = np.unique(qpt_vecs.reshape(-1, 3), axis=0)

        # Generate indices using vectorized operations
        self.all_unfolded_qpt_ids = np.zeros((len(self.kvecs_asints), len(self.kvecs_asints)), dtype=np.int32)
        # This is still a loop but operates on whole arrays at once
        for i, q in enumerate(self.all_unfolded_qpts):
            mask = (qpt_vecs == q).all(axis=2)
            self.all_unfolded_qpt_ids[mask] = i

        # Eager q-IBZ reduction (was lazy in `find_irreducible_qpoints`; that
        # method is gone — all consumers read these instance attrs directly).
        # q lives on the same kgrid as k (q = k - k'), so we reuse
        # ``sym_mats_k`` (which already includes time-reversal). Note that
        # `is_trs[i_full] = sym_idx_q[i_full] >= ntran` is implicit; not stored.
        irr_idx_q, sym_idx_q, q_irr_kgrid_int = find_irreducible_bz_points(
            self.kvecs_asints, self.sym_mats_k, irr_kgrid_int=None,
        )
        self.irr_idx_q = irr_idx_q
        self.sym_idx_q = sym_idx_q
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
        """
        Build the map from each full-grid k-point to its irreducible-k partner.
        
        Args:
            wfn (WfnReader): WFN reader object
            
        Returns:
            tuple: (kpoint_map, full_kpoints)
                - kpoint_map: Array mapping each full-grid k-point to the
                  matching irreducible-k index in ``wfn.kpoints``
                - full_kpoints: Array of all k-points in the full grid
        """
        full_kpoints = self._generate_uniform_full_kpoints(wfn)
        irr_kpts = self._wrap_to_bz(wfn.kpoints)

        # Map each full k-point to its irreducible representative.
        kpoint_map = np.zeros(len(full_kpoints), dtype=np.int32)
        unmatched_kpts = []
        
        for kfull_idx in range(len(full_kpoints)):
            k_found = False
            for sym_idx, sym_mat in enumerate(self.sym_mats_k):
                # Apply symmetry operation to k-point
                k_transformed = self._wrap_to_bz(sym_mat @ full_kpoints[kfull_idx])
                
                # Check if transformed k-point matches any k-point in wfn.kpoints
                for irk_idx, k_wrapped in enumerate(irr_kpts):
                    if np.allclose(k_transformed, k_wrapped, atol=1e-6):
                        kpoint_map[kfull_idx] = irk_idx
                        k_found = True
                        break
                
                if k_found:
                    break
            
            if not k_found:
                # Fallback: find nearest irreducible k-point and use identity
                # This handles cases where WFN symmetry data is incomplete
                kfull = full_kpoints[kfull_idx]
                dists = np.linalg.norm(self._periodic_delta(irr_kpts, kfull), axis=1)
                nearest_irr = np.argmin(dists)
                kpoint_map[kfull_idx] = nearest_irr
                unmatched_kpts.append((kfull_idx, full_kpoints[kfull_idx], nearest_irr))
        
        if unmatched_kpts:
            import warnings
            warnings.warn(f"WFN symmetry data incomplete: {len(unmatched_kpts)} k-points could not be "
                         f"mapped via stored symmetries (ntran={len(self.sym_mats_k)}). "
                         f"Using identity fallback. First unmatched: {unmatched_kpts[0][1]}")
        
        return kpoint_map, full_kpoints
        
    def find_symmetry_ops_simple(self, wfn, kpoint_map, full_kpts):
        del kpoint_map  # kept in signature for compatibility with older callers
        irk_to_k_map = np.zeros(full_kpts.shape[0], dtype=np.int32)
        irk_sym_map = np.zeros(full_kpts.shape[0], dtype=np.int32)
        ntran = len(self.sym_matrices)
        # all symmetries applied to the irr k-points: shape (nkbar, nsym, 3)
        Skbar = np.einsum('ijk,lk->lij', self.sym_mats_k, wfn.kpoints)
        Skbar = self._wrap_to_bz(Skbar)

        # find the symmetry operations that map the irr k-points to the full k-points
        for ikfull, kfull in enumerate(full_kpts):
            for ikbar in range(wfn.nkpts):
                # Compare each component within tolerance
                diffs = np.abs(Skbar[ikbar] - kfull)
                matches = np.where(np.all(diffs < 1e-6, axis=1))[0]
                if len(matches) > 0:
                    irk_to_k_map[ikfull] = ikbar
                    irk_sym_map[ikfull] = matches[0]

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
            kg0 = self._get_umklapp_vector(wfn, ik_full, sym_idx, ik_irr, sym_krep)
            mapped = sym_krep @ np.asarray(wfn.kpoints[ik_irr], dtype=np.float64) + kg0
            if np.max(np.abs(mapped - self.unfolded_kpts[ik_full])) > tol:
                failures.append(
                    f"ik_full={ik_full}: S*k_irr + kg0 does not reproduce full k-point"
                )

        return failures

    def syms_crystal_to_cartesian(self, wfn):
        """Cartesian rotation matrix used as input to ``get_spinor_rotations``.

        ``get_spinor_rotations`` runs Markley's quaternion algorithm and
        requires ORTHOGONAL 3D rotation matrices. The matrix it consumes is
        the cartesian image of LORRAX's ``mtrx`` (= ``sym_matrices``) — NOT
        of ``mtrx.T`` (= ``sym_mats_k``), NOT of ``inv(mtrx)``.

        Empirically verified against nosym ground truth on Si 4×4×4 SOC
        (Fd-3m, 48 ops): U_spinor built from this R_cart reproduces nosym ψ
        to ~3e-6 within the degenerate-subspace unitary gauge — five orders
        of magnitude tighter than the pre-fix code. See
        ``reports/trs_sym_audit_2026-05-14/algebraic_unfold_{cri3,si}.md``.

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
        mtrx = np.asarray(self.sym_matrices)
        R_spatial = np.einsum('ij,njk,kl->nil', A_T, mtrx, A_T_inv)
        R_spatial = np.around(R_spatial, decimals=10)
        # Match the existing 2·ntran-row convention for downstream consumers.
        R_full = np.concatenate([R_spatial, -R_spatial], axis=0)
        return R_full

    def get_spinor_rotations(self, wfn, sym_matrices_cart):
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
        # Initialize mapping array
        nk_full = len(full_kpts)
        nk_red = wfn.nkpts
        kq_map = np.zeros((nk_full, nk_red), dtype=np.int32)
        
        # Get reduced k-points
        reduced_kpts = np.asarray(wfn.kpoints)
        
        # For each full k-point and each reduced q-point
        for ik in range(nk_full):
            k = full_kpts[ik]
            for iq in range(nk_red):
                q = reduced_kpts[iq]

                # Calculate k-q; use periodic distance to find match
                kminusq = k - q

                # Periodic distance: min |full_kpts - kminusq - G| over G
                delta = full_kpts - kminusq[None, :]
                delta = delta - np.round(delta)  # wrap differences to [-0.5, 0.5)
                diffs = np.sum(np.abs(delta), axis=1)
                min_diff = np.min(diffs)

                if min_diff > 1e-4:
                    raise ValueError(f"k-q point {kminusq} not found in k-point grid")

                kq_idx = np.argmin(diffs)
                if kq_idx >= nk_full:
                    raise ValueError(f"Invalid k-q mapping: {kq_idx} >= {nk_full}")

                kq_map[ik, iq] = kq_idx

        return kq_map

    def get_kminusqfull_map(self, wfn, full_kpts):
        # Initialize mapping array
        nk_full = len(full_kpts)
        nk_red = wfn.nkpts
        kq_map = np.zeros((nk_full, nk_full), dtype=np.int32)

        # For each full k-point and each reduced q-point
        for ik in range(nk_full):
            k = full_kpts[ik]
            for iq in range(nk_full):
                q = full_kpts[iq]

                # Calculate k-q; use periodic distance to find match
                kminusq = k - q

                # Periodic distance: min |full_kpts - kminusq - G| over G
                delta = full_kpts - kminusq[None, :]
                delta = delta - np.round(delta)  # wrap differences to [-0.5, 0.5)
                diffs = np.sum(np.abs(delta), axis=1)
                min_diff = np.min(diffs)

                if min_diff > 1e-4:
                    raise ValueError(f"k-q point {kminusq} not found in k-point grid")

                kq_idx = np.argmin(diffs)
                if kq_idx >= nk_full:
                    raise ValueError(f"Invalid k-q mapping: {kq_idx} >= {nk_full}")

                kq_map[ik, iq] = kq_idx

        return kq_map
    
    def _get_umklapp_vector(self, wfn, nk, sym_idx, kbar_idx, sym_krep):
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
        """Find index of q-point in unfolded k-points list.

        Args:
            q_ext: Vector of length 3 (crystal coordinates)
            tol: Tolerance for floating point comparison

        Returns:
            Index of matching q-point, or raises ValueError if not found
        """
        # Get fractional part of q_ext
        q_frac = q_ext % 1.0
        diffs = jnp.abs(self.unfolded_kpts - q_frac[None, :])
        # Sum over coordinates and find minimum difference
        total_diffs = jnp.sum(diffs, axis=1)
        min_diff = jnp.min(total_diffs)

        if min_diff > tol:
            raise ValueError(f"No matching q-point found within tolerance {tol}")

        return jnp.argmin(total_diffs)
