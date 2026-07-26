"""Exact W_c(omega) via shifted solves with the non-TDA RPA density resolvent.

Cross-validates the GW screened Coulomb.  In the ISDF centroid (density) basis
the screened interaction obeys the Casida resolvent identity

    W(omega) - v  =  v (omega - H_RPA)^{-1} v            (omega = 0: static W0)

where H_RPA is the NON-TDA symplectic RPA (test-charge) density-response
Hamiltonian with the bare-exchange RING kernel V (the B1 dense k-summed form):

    H_RPA = [[ D + V ,   V   ],
             [  -V   , -D - V ]]

V sits in BOTH blocks — the RPA ring coupling K^A = (1/Nk)<M_t|v|M_t'>
(``build_bse_ring_matvec_full(..., screening=True)``), NOT the excitonic V_B of
Henneke Eq. 2-20.  This is the test-charge screening whose resolvent resums the
RPA bubble chi = chi0 (1 - v chi0)^{-1}; the exciton V_B kernel is a different
response and does NOT reproduce W (it overshoots the q=0 tile by ~1.8x).  The GW
W is full-RPA, not TDA — dropping the -V/Y block (TDA) fails by construction.

Convention (verified bit-for-bit against the folded static RPA and the on-disk
``W0_qmunu - V_qmunu`` q=0 tile to ~2e-9 — see the "W(0) resolvent cross-check"
section of reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md):

  * Probe column nu: g = e_nu in centroid space; the transition generator applies
    v then the pair-density vertex, f = M^dag (v e_nu)
    (``build_realspace_random_transition_generator``) — the RIGHT vertex of v X v.
  * RHS is that same f with a minus on the anti-resonant (Y) block: rhs = [f; -f]
    (density super-vertex [rho; -rho]; the ring coupling makes the excitation and
    de-excitation vertices coincide, so both blocks carry the SAME f).
  * Shifted solve x = (z - H_RPA)^{-1} rhs at z = (omega + i eta)/Ry via GMRES.
  * Readout s = x[0] + x[1] = X + Y; the density-snapshot vertex applies the pair
    density then v: w_c(mu) = v (M s) = [v chi v]_{mu,nu} = column nu of W - v.

The generator/snapshot k-SUM the pair densities, so the reconstructed tile is the
q=0 block.  H_RPA carries no q=0 head (vhead/whead are a separate rank-1 piece);
``--compare-w0`` loads head-LESS bodies on both sides (``inject_head=False``) and
compares body-to-body.
"""
from __future__ import annotations

import math
import time
import numpy as np
import h5py

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import (
    RY_TO_EV_DEFAULT, ensure_W_R, build_preconditioner_diagonal_sharded,
    _apply_shifted_matvec, _gmres_solve_core, matvec_operands,
)
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec_full,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
    make_bse_shardings,
)
from .bse_serial import compute_pair_amplitude
import common.timing as timing

jax.config.update("jax_enable_x64", True)

# Cache the compiled per-column-scan block-GMRES engine per operator structure.
# Keyed on (id(matvec), max_iter, tol, dtype) — NOT on the per-q data — so the
# finite-q W_q loop and the per-omega oracle sweep reuse ONE executable and every
# q / omega after the first is dispatch-only (see _get_block_gmres_solver).
_BLOCK_GMRES_CACHE: dict[tuple, tuple] = {}


def _create_mesh_xy(px: int, py: int) -> Mesh:
    devices = jax.devices()
    if px * py > len(devices):
        raise ValueError(f"Requested px*py={px*py} devices, only {len(devices)} available")
    return Mesh(np.array(devices[: px * py]).reshape(px, py), axis_names=("x", "y"))


def _build_rpa_resolvent(mesh_xy: Mesh, data: dict):
    """Assemble the RPA-screening resolvent stack for ``data``.

    Returns ``(matvec, diag_h, gen, snapshot, sh)``.  ``matvec`` is the non-TDA
    symplectic RPA density-response Hamiltonian (screening ring kernel, no W);
    ``ensure_W_R`` populates the placeholder 8th matvec argument.

    ``gen`` (zeta->pair seed) and ``snapshot`` (pair->zeta projection) are the
    two reshard boundaries of the resolvent; ``snapshot`` is built in the
    reduce-scatter mode so the projection-back assembles ``W(mu_X, nu_Y)`` tiles
    directly (see :func:`apply_screening_resolvent_block`).
    """
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    ensure_W_R(data, include_W=False)
    matvec = build_bse_ring_matvec_full(
        mesh_xy, nkx, nky, nkz, include_W=False, screening=True)
    diag_h = build_preconditioner_diagonal_sharded(
        data, mesh_xy, include_W=False, use_tda=False)
    gen = build_realspace_random_transition_generator(
        mesh_xy, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
    snapshot = build_density_snapshot_operator(
        mesh_xy, nkx, nky, nkz, scatter_nu_on_y=True)
    return matvec, diag_h, gen, snapshot, make_bse_shardings(mesh_xy)


def _roll_k_axis_host(arr_np, q, nkx, nky, nkz):
    """Host-side (numpy) roll of an on-grid tensor by ``+q`` on the C-order
    (nkx,nky,nkz) k-axis (axis 0).  ``out[k] = arr[k − q]`` on the wrapped grid,
    i.e. it gathers the conduction quantity at ``k − q`` into slot ``k`` — the
    shift that reproduces the stored ``W0_qmunu[q_flat]`` tile (see
    :func:`build_finite_q_data`).

    Done on host (numpy), NOT device ``jnp.roll``: a static device roll bakes the
    q-offset into the compiled program, so a different q recompiled the roll once
    per q.  Rolling on host (arrays are small — psi_c is a few tens of MB) yields
    the rolled array as plain DATA that ``device_put`` uploads with the target
    sharding, so no per-q compile — the whole finite-q loop stays dispatch-only
    after the first engine build."""
    tail = arr_np.shape[1:]
    a = arr_np.reshape((nkx, nky, nkz) + tail)
    a = np.roll(a, shift=(int(q[0]), int(q[1]), int(q[2])), axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def build_finite_q_data(data, q, mesh_xy):
    """Finite-momentum screening data for the W_q resolvent — the q generalization.

    The RPA density response at momentum ``q`` lives in the on-grid pair basis
    ``|v k, c k+q⟩``: conduction quantities are remapped on the k-axis, the
    screening tile becomes the finite-q ``V_qmunu[q_flat]``, valence/energies and
    everything else are the q=0 arrays.  Returns a shallow copy of ``data`` with
    only the conduction/V slots swapped, so the SAME
    :func:`apply_screening_resolvent_block` engine (matvec, seed, project,
    solver, sharding) runs unchanged — this GENERALIZES q=0, it does not fork it.

    Remap (``q = (qx,qy,qz)`` integer grid steps; C-order flat
    ``k = ix·nky·nkz + iy·nkz + iz``):

      * ``psi_c`` / ``eps_c`` ← ``jnp.roll(·, shift=+q)`` on the reshaped
        (nkx,nky,nkz) k-axis ⇒ slot ``k`` holds the conduction value at ``k − q``.
        The pair density is then ``M^q_cvk(μ) = Σ_s conj(ψ_c[k−q](μ)) ψ_v[k](μ)``
        and ``D^q = ε_c[k−q] − ε_v[k]``; summed over k this is exactly the GW
        producer's χ₀(q) convolution ``Σ_k Gc_k Gv*_{k+q}`` (relabel k→k−q), which
        reproduces the on-disk ``W0_qmunu[q_flat]`` tile.
      * NO umklapp Bloch phase.  The GW χ₀(q) is built by a plain periodic
        FFT-convolution over k (``w_isdf._get_chi_minimax_kernel``), which uses
        the RAW stored ψ at the wrapped index — no ``exp(-2πi G_umk·s_μ)`` factor.
        Applying the design-doc umklapp phase here breaks the match (rel_err
        0.6–3.2 vs 1e-8); the phase belongs to a DIRECT-read finite-Q BSE path,
        not to matching this FFT-convolution-produced W tile.  Verified on the
        MoS2 gnppm fixture (finite-q PHASE2_LOG section).
      * ``V_q0`` ← ``V_qmunu[q_flat]`` = ``data['V_q_full'][:,:,qx,qy,qz]`` — the
        finite-q exchange tile.  At q≠0 it KEEPS G=0 (``compute_vcoul`` zeroes
        G=0 only at q=0) and carries NO separate rank-1 head (heads are a
        q=0-only piece).  ``q=(0,0,0)`` returns the unshifted q=0 data
        (roll-by-0 is identity, ``V_q_full[...,0] == V_q0``).

    Requires ``data`` loaded with ``load_v_full=True``.
    """
    if data.get("V_q_full") is None:
        raise ValueError(
            "build_finite_q_data needs the full V tensor; load the restart with "
            "load_bse_data_from_restart_sharded(..., load_v_full=True).")
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
    sh = make_bse_shardings(mesh_xy)
    dq = dict(data)
    dq["V_q0"] = data["V_q_full"][:, :, qx, qy, qz]
    # Roll on host (see _roll_k_axis_host) so a different q needs no new compile;
    # device_put uploads the rolled array straight into the target sharding.
    psi_c_q = jnp.asarray(_roll_k_axis_host(
        np.asarray(jax.device_get(data["psi_c_X"])), (qx, qy, qz), nkx, nky, nkz))
    dq["psi_c_X"] = jax.device_put(psi_c_q, sh.psi_x)
    dq["psi_c_Y"] = jax.device_put(psi_c_q, sh.psi_y)
    dq["eps_c"] = jax.device_put(jnp.asarray(_roll_k_axis_host(
        np.asarray(jax.device_get(data["eps_c"])), (qx, qy, qz), nkx, nky, nkz)), sh.eps)
    # M_X/M_Y are hoisted V-term pair amplitudes (audit P3) and are pure functions
    # of psi_c — the finite-q roll shifted psi_c, so recompute them from the ROLLED
    # conduction states.  The q=0 M's shallow-copied from `data` would be stale and
    # give the wrong finite-q screening operator (M^q_cvk(μ)=Σ_s conj(ψ_c[k−q]) ψ_v[k]).
    dq["M_X"] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(dq["psi_c_X"], dq["psi_v_X"]), sh.psi_x)
    dq["M_Y"] = jax.lax.with_sharding_constraint(
        compute_pair_amplitude(dq["psi_c_Y"], dq["psi_v_Y"]), sh.psi_y)
    return dq


def _symmetry_reduced_q_list(input_file: str) -> np.ndarray:
    """Symmetry-reduced (IBZ) q-grid points as integer kgrid steps ``(n_q, 3)``,
    from the ONE canonical ``SymMaps`` table (``q_irr_kgrid_int``); no new sym
    helper.  Row 0 is Γ = (0,0,0)."""
    from file_io import WfnLoader as WFNReader
    from common.symmetry_maps import SymMaps
    from .bse_io import _parse_wfn_path
    wfn = WFNReader(_parse_wfn_path(input_file))
    sym = SymMaps(wfn)
    return np.asarray(sym.q_irr_kgrid_int, dtype=int)


def _get_block_gmres_solver(matvec, sh, max_iter, tol, dtype):
    """Cached jitted per-column-scan block-GMRES engine for the screening
    resolvent — the stage-2 SOLVE of :func:`apply_screening_resolvent_block`.

    Returns a ``jax.jit`` function ``(rhs, diag_h, z, operands) -> (s_all, resids)``
    that scans the per-column-independent shifted GMRES over the probe axis.  The
    operator STRUCTURE (``matvec``, ``sh``, ``max_iter``, ``tol``) is baked into
    the compiled program; the q- and omega-dependent tensors (``rhs``, ``diag_h``,
    ``z``, and the ``matvec_operands`` tuple ``operands``) are RUNTIME ARGUMENTS.
    So it compiles ONCE per operator structure and every later q / omega is
    dispatch-only — replacing the old top-level ``lax.scan`` that closed over the
    per-q ``data`` and recompiled the ~4.8 s scan once per q.

    Cache key ``(id(matvec), max_iter, tol, dtype)`` keeps the operator-identity
    safety: genuinely different structures (screening vs optical, TDA vs full)
    carry distinct ``matvec`` objects and so distinct engines."""
    key = (id(matvec), int(max_iter), float(tol), str(dtype))
    hit = _BLOCK_GMRES_CACHE.get(key)
    if hit is not None:
        return hit[1]

    @jax.jit
    def _block(rhs, diag_h, z, operands):
        # rhs: (2, nu, c, v, k) — scan the per-column GMRES over the probe axis nu.
        rhs_scan = jnp.moveaxis(rhs, 1, 0)  # (nu, 2, c, v, k)

        def _solve_col(carry, rhs_col):
            rhs_i = rhs_col[:, None]        # (2, 1, c, v, k) — keep the matvec batch axis
            x, _ = _gmres_solve_core(matvec, rhs_i, diag_h, z, operands,
                                     max_iter, tol)
            r_true = rhs_i - _apply_shifted_matvec(matvec, x, z, operands)
            nrhs = jnp.linalg.norm(rhs_i)
            resid = jnp.where(nrhs == 0.0, jnp.asarray(0.0, dtype=nrhs.dtype),
                              jnp.linalg.norm(r_true) / nrhs)
            s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)  # (1, c, v, k)
            return carry, (s[0], resid)

        _, (s_all, resids) = jax.lax.scan(_solve_col, None, rhs_scan)
        s_all = jax.lax.with_sharding_constraint(s_all, sh.X)        # (nu, c, v, k)
        return s_all, resids

    _BLOCK_GMRES_CACHE[key] = (matvec, _block)
    return _block


def apply_screening_resolvent_block(G_zeta, z, data, matvec, diag_h, gen,
                                    snapshot, sh, *, max_iter, tol):
    """Screened-Coulomb resolvent on a block of probe columns — the ONE engine.

    Computes ``W(omega) - v`` tiles from the non-TDA RPA density-response
    resolvent ``v (z - H_RPA)^{-1} v`` for a whole block of centroid-space probe
    columns at once, and is the single shared implementation behind
    ``--compare-w0`` today and the future Lanczos-chain ``W(omega)`` model.

    Data flow (the three-stage seam the future model reuses verbatim):

      1. SEED (zeta -> pair, batched, ``gen`` shard_map).  Each probe column
         ``nu`` is a density-space vector ``g``; the transition generator applies
         ``v`` then the pair-density vertex, ``f = M^dag (v g)``, and the non-TDA
         RHS is the density super-vertex ``[f; -f]`` (both blocks carry the same
         ``f`` — the ring coupling makes excitation/de-excitation vertices
         coincide).  Output pair-basis block ``sh.X_full`` = ``(2, nu, c_X, v_Y, k)``,
         with ``nu`` a REPLICATED batch axis (the pair basis already tiles c on x
         and v on y — the probe axis has no free mesh axis and stays replicated
         through the solve; only the boundaries carry it as a shardable index).

      2. SOLVE (operator application).  Per-column-independent shifted GMRES via
         ``lax.scan`` over the probe axis — one Krylov subspace alive at a time
         (no N-way unrolled slot pile-up), bit-identical to the per-column loop
         it replaces because the GMRES norm/least-squares reductions are global
         per single column.  The scan stacks the readouts ``s = x[0] + x[1]``
         into ``sh.X`` = ``(nu, c_X, v_Y, k)`` and the per-column relative
         residuals.  Runs inside the cached, jitted :func:`_get_block_gmres_solver`
         engine with the q/omega-dependent tensors (``rhs``, ``diag_h``, ``z``,
         ``matvec_operands``) as RUNTIME ARGUMENTS, so it compiles ONCE per
         operator structure and every later q / omega is dispatch-only.

      3. PROJECT (pair -> zeta, batched, reduce-scatter ``snapshot``).  The
         density-snapshot vertex applies the pair density then ``v``,
         ``w(mu) = v (M s)``, and — because ``snapshot`` was built with
         ``scatter_nu_on_y=True`` — reduce-scatters the ``nu`` batch onto ``y``
         while completing the V_q0 contraction, emitting the tile directly as
         ``W(mu_X, nu_Y)`` = ``sh.V``.  No replicated ``(mu, nu)`` is ever formed.

    Block-Lanczos plug-in (future W(omega), do NOT build here).  The screening
    matvec (``build_bse_ring_matvec_full(screening=True)``) is symmetric under
    the symplectic metric, so a single block-Lanczos run seeded with the SAME
    stage-1 pair-basis block ``rhs`` (the v-probe seeds) tridiagonalizes the
    screening operator ONCE; every ``omega`` is then a small
    ``(T - z)^{-1}`` resolvent on the projected tridiagonal, and the moments are
    projected back to the zeta basis by the SAME stage-3 ``snapshot``.  Only the
    stage-2 middle swaps (scan-of-GMRES -> one block-Lanczos + per-omega tiny
    solves); stages 1 and 3 — the reshard boundaries — are reused unchanged.
    Keep any such model calling THIS function's seed/project so the layout stays
    single-sourced.

    Parameters
    ----------
    G_zeta : array (n_probe, n_rmu)
        Probe block in the padded centroid basis; row ``i`` is probe column
        ``nu_i`` (typically a unit column ``e_nu`` for a W column, or an identity
        block for the full basis).  ``n_probe`` must be a multiple of ``py`` (the
        reduce-scatter tiles ``nu`` over ``y``); pad with zero rows otherwise.

    Returns
    -------
    W_tile : jax.Array (n_rmu, n_probe), sharding ``sh.V`` = ``P('x', 'y')``
        Column ``i`` is ``W(omega) - v`` for probe ``nu_i`` in the padded centroid
        basis (``mu`` on x, ``nu`` on y).
    resids : jax.Array (n_probe,) float64
        Per-column relative GMRES residual of the shifted system (0 for zero-pad
        columns), so quadrature noise vs solver tolerance stay distinguishable.
    """
    px, py = sh.X.mesh.devices.shape
    n_probe = int(G_zeta.shape[0])
    if n_probe % py != 0:
        raise ValueError(
            f"probe block n_probe={n_probe} must be a multiple of py={py} "
            "(reduce-scatter tiles nu over y); pad the probe block with zero rows.")
    n_rmu = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    # --- Stage 1: SEED (zeta -> pair), batched over the whole probe block. ---
    G = jnp.asarray(G_zeta, dtype=jnp.float64)
    r = jax.device_put(jnp.broadcast_to(G[:, :, None], (n_probe, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    rhs = jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)  # (2, nu, c, v, k)

    # --- Stage 2: SOLVE via the cached jitted per-column-scan GMRES engine. ---
    # The engine is keyed on the operator STRUCTURE (matvec); the q/omega-dependent
    # operand arrays flow in as runtime args, so it compiles ONCE and every later
    # q / omega is dispatch-only.  z is passed as a device scalar (not a Python
    # complex) so a different omega stays a runtime arg, never a baked constant.
    solver = _get_block_gmres_solver(matvec, sh, max_iter, tol, rhs.dtype)
    s_all, resids = solver(rhs, diag_h, jnp.asarray(z, dtype=jnp.complex128),
                           matvec_operands(data))

    # --- Stage 3: PROJECT (pair -> zeta), reduce-scatter to W(mu_X, nu_Y). ---
    W_tile = snapshot(s_all, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
    return W_tile, resids


def _resolve_wc_columns(cols, z, data, matvec, diag_h, gen, snapshot, sh,
                        *, max_iter, tol):
    """Resolve the ``W(omega) - v`` columns listed in ``cols`` (head-less q=0
    tile, padded centroid space) via :func:`apply_screening_resolvent_block`.

    Builds the unit-column probe block for ``cols`` (zero-padded up to a multiple
    of ``py``), solves once, and returns the assembled device tile.

    Returns ``(W_tile[n_rmu, n_pad] sh.V, resids[n_pad])``, where column ``i`` of
    ``W_tile`` (``W_tile[:, i]``) is ``W - v`` for probe ``cols[i]``; the final
    ``n_pad - len(cols)`` columns are zero pad.  ``W_tile.sharding.spec`` is
    ``P('x', 'y')`` = ``(mu_X, nu_Y)``.
    """
    px, py = sh.X.mesh.devices.shape
    n_rmu = int(data["V_q0"].shape[0])
    cols = np.asarray(cols, dtype=int)
    n_pad = int(math.ceil(len(cols) / py) * py)
    G = np.zeros((n_pad, n_rmu), dtype=np.float64)
    for i, nu0 in enumerate(cols):
        G[i, int(nu0)] = 1.0
    return apply_screening_resolvent_block(
        G, z, data, matvec, diag_h, gen, snapshot, sh, max_iter=max_iter, tol=tol)


def _select_compare_cols(T, nlog, n_cols, seed):
    """A mix of the largest-||W0-V|| columns and random columns (logical range)."""
    col_norm = np.linalg.norm(T[:nlog, :nlog], axis=0)
    order = np.argsort(-col_norm)
    n_large = (n_cols + 1) // 2
    large = order[:n_large]
    rng = np.random.default_rng(seed)
    remaining = np.setdiff1d(np.arange(nlog), large)
    n_rand = min(n_cols - n_large, remaining.size)
    rand = (rng.choice(remaining, size=n_rand, replace=False)
            if n_rand > 0 else np.empty(0, dtype=int))
    return np.concatenate([large, rand]).astype(int), col_norm


def _parse_cols(col_str, n_mu, n_cols, seed):
    if col_str:
        cols = [int(x) for x in col_str.split(",") if x.strip() != ""]
        return np.array([c for c in cols if 0 <= c < n_mu], dtype=int)
    if n_cols is not None:
        rng = np.random.default_rng(seed)
        if n_cols >= n_mu:
            return np.arange(n_mu, dtype=int)
        return rng.choice(n_mu, size=n_cols, replace=False)
    return np.arange(n_mu, dtype=int)


def run_w_omega_chain_compare(
    data, mesh_xy, cols, *, freqs_ev, chain_len, chain_sweep,
    ry_to_ev, gmres_max_iter, gmres_tol, disk_tile=None, nlog=None,
    label="", print_fn=print,
):
    """Validate the Lanczos-chain W(omega) model against the shifted-solve oracle.

    Builds ONE block-Lanczos chain (:func:`w_omega_chain.build_w_omega_chain`) at
    ``chain_len`` for the probe ``cols``, then for every complex frequency in
    ``freqs_ev`` (each ``omega + i eta`` in eV — covers the static ``omega=0``,
    imaginary-axis ``i b``, and real-axis ``a + i eta`` cases) evaluates the chain
    at each truncated length in ``chain_sweep`` and compares to the per-omega
    oracle ``_resolve_wc_columns`` (fresh shifted block-GMRES).  At ``omega=0``
    also reports closure against the on-disk ``disk_tile`` (``W0 - V``), the GW
    ground truth.  Prints a per-omega convergence table (rel_err vs chain length)
    and a timing / omega-count break-even summary (the whole point: amortize the
    chain build over many frequencies).  Returns a results dict for the gate.

    ``cols`` are the probe density columns (``T``-largest + random mix); the chain
    block width is ``p = len(cols)``.  ``disk_tile`` / ``nlog`` are the head-less
    ``(W0 - V)`` numpy tile and its logical mu extent (omega=0 ground truth)."""
    from . import w_omega_chain as woc

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)
    cols = np.asarray(cols, dtype=int)
    p = len(cols)
    if nlog is None:
        nlog = int(data["n_rmu"])

    def _oracle(z):
        W_tile, resids = _resolve_wc_columns(
            cols, z, data, matvec, diag_h, gen, snapshot, sh,
            max_iter=gmres_max_iter, tol=gmres_tol)
        return (np.asarray(jax.device_get(W_tile)),
                np.asarray(jax.device_get(resids))[:p])

    def _chain_eval(z, m_use=None):
        (W_tile,) = woc.eval_w_omega_chain(chain, data, snapshot, sh, z, m_use=m_use)
        jax.block_until_ready(W_tile)
        return W_tile

    # ---- build the chain ONCE (timed, warm) ----
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, min(2, chain_len))  # warm compile
    t0 = time.perf_counter()
    chain = woc.build_w_omega_chain(data, matvec, gen, sh, cols, chain_len)
    jax.block_until_ready(chain["V_stack"])
    t_build = time.perf_counter() - t0

    sweep = sorted({int(m) for m in chain_sweep if 1 <= int(m) <= chain_len})
    if chain_len not in sweep:
        sweep.append(chain_len)

    print_fn(f"\n=== W(omega) Lanczos-chain vs oracle{(' ' + label) if label else ''} ===")
    print_fn(f"probe cols={list(cols)}  block p={p}  chain_len={chain_len}  "
             f"chain build={t_build:.3f}s ({chain_len} matvecs)")
    print_fn("chain residual ||beta_m|| by length: " + ", ".join(
        f"m={m}:{woc.chain_residual_norm(chain, m):.2e}" for m in sweep))

    results = {"t_build": t_build, "chain_len": chain_len, "by_freq": {},
               "cols": cols, "p": p}

    # timing probes (warm) at a representative interior z
    z_probe = (1.0 + 0.2j) / ry_to_ev
    _chain_eval(z_probe)                                    # warm
    t_c0 = time.perf_counter(); _chain_eval(z_probe); t_chain_eval = time.perf_counter() - t_c0
    _oracle(z_probe)                                        # warm
    t_o0 = time.perf_counter(); _oracle(z_probe); t_oracle = time.perf_counter() - t_o0
    results["t_chain_eval"] = t_chain_eval
    results["t_oracle"] = t_oracle

    hdr = (f"{'freq(eV)':>16} {'m':>4} {'rel_vs_oracle':>14} "
           f"{'rel_vs_disk':>12} {'gmres_resid':>12}")
    print_fn(hdr)
    print_fn("-" * len(hdr))
    for w_ev in freqs_ev:
        w_ev = complex(w_ev)
        z = w_ev / ry_to_ev
        w_oracle, resid = _oracle(z)
        is_static = abs(w_ev) < 1e-12
        flabel = f"{w_ev.real:+.3f}{w_ev.imag:+.3f}i"
        for m in sweep:
            wc = np.asarray(jax.device_get(_chain_eval(z, m_use=m)))
            rel_o = float(max(
                np.linalg.norm(wc[:nlog, i] - w_oracle[:nlog, i])
                / max(np.linalg.norm(w_oracle[:nlog, i]), 1e-300)
                for i in range(p)))
            if is_static and disk_tile is not None:
                rel_disk = float(max(
                    np.linalg.norm(wc[:nlog, i] - disk_tile[:nlog, int(cols[i])])
                    / max(np.linalg.norm(disk_tile[:nlog, int(cols[i])]), 1e-300)
                    for i in range(p)))
            else:
                rel_disk = np.nan
            print_fn(f"{flabel:>16} {m:4d} {rel_o:14.3e} "
                     f"{rel_disk:12.3e} {float(resid.max()):12.3e}")
            results["by_freq"].setdefault(flabel, {})[m] = {
                "rel_vs_oracle": rel_o, "rel_vs_disk": rel_disk,
                "gmres_resid": float(resid.max())}
        print_fn("-" * len(hdr))

    # amortization / break-even
    denom = t_oracle - t_chain_eval
    breakeven = (t_build / denom) if denom > 0 else float("inf")
    print_fn(f"\nTiming (warm): chain build {t_build:.3f}s | per-omega oracle "
             f"{t_oracle*1e3:.1f} ms | per-omega chain eval {t_chain_eval*1e3:.1f} ms")
    print_fn(f"omega-count break-even: chain wins after ~{breakeven:.1f} frequencies "
             f"(each extra omega is ~{t_oracle/max(t_chain_eval,1e-9):.0f}x cheaper "
             f"than a fresh oracle solve).")
    results["breakeven_omega_count"] = breakeven
    return results


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False,
        description="Exact W_c(omega) via the non-TDA RPA density resolvent")
    parser.add_argument("-i", "--input", required=True, help="COHSEX input file")
    parser.add_argument("--compare-w0", action="store_true",
                        help="Cross-check v(0-H_RPA)^-1 v against the restart's "
                             "(W0_qmunu - V_qmunu) q=0 tile.")
    parser.add_argument("--compare-wq", action="store_true",
                        help="Finite-q generalization of --compare-w0: loop over the "
                             "symmetry-reduced (IBZ) q-grid one at a time and cross-check "
                             "each W_q against its own (W0_qmunu - V_qmunu)[q_flat] tile.")
    parser.add_argument("--w-omega-chain", action="store_true",
                        help="Full-frequency W(omega) model: build ONE block-Lanczos "
                             "chain per q and evaluate W_q(omega)-v_q across a frequency "
                             "grid with NO per-omega solve; validate vs the shifted-solve "
                             "oracle (the amortized production path).")
    parser.add_argument("--chain-len", type=int, default=32,
                        help="Block-Lanczos chain length (blocks) for --w-omega-chain "
                             "(the accuracy knob: ~1e-6 on the imaginary axis, ~2e-4 "
                             "static at 32 on the MoS2 fixture; raise for tighter).")
    parser.add_argument("--chain-sweep", type=str, default=None,
                        help="Comma list of chain lengths for the convergence table "
                             "(default: 4,8,12,16,chain_len).")
    parser.add_argument("--chain-freqs-ev", type=str, default=None,
                        help="Comma list of complex frequencies (eV), e.g. "
                             "'0,2j,4j,1.5+0.2j'; default samples 0, imaginary axis, "
                             "and real axis + i*eta.")
    parser.add_argument("--chain-q", type=str, default=None,
                        help="Finite q as 'qx,qy,qz' kgrid steps for --w-omega-chain "
                             "(default: also do the smallest nonzero IBZ q).")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Valence bands (default: FULL chi0 window = n_occ).")
    parser.add_argument("--n-cond", type=int, default=None,
                        help="Conduction bands (default: FULL chi0 window).")
    parser.add_argument("--px", type=int, default=1)
    parser.add_argument("--py", type=int, default=1)
    parser.add_argument("--omega-ev", type=float, default=0.0,
                        help="Real frequency omega in eV (default: 0, static W0).")
    parser.add_argument("--eta-ev", type=float, default=0.0,
                        help="Imaginary broadening eta in eV (default: 0).")
    parser.add_argument("--cols", type=str, default=None,
                        help="Comma-separated mu indices to compute (0-based).")
    parser.add_argument("--n-cols", type=int, default=6,
                        help="Number of probe columns (compare-w0: largest-|W0-V| "
                             "+ random mix; else random).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gmres-max-iter", type=int, default=200)
    parser.add_argument("--gmres-tol", type=float, default=1e-10)
    parser.add_argument("--ry-to-ev", type=float, default=RY_TO_EV_DEFAULT)
    parser.add_argument("--out", type=str, default="Wc_exact.h5")
    args = parser.parse_args(argv)

    timing.reset()
    mesh_xy = _create_mesh_xy(args.px, args.py)
    restart_file = _find_restart_file(args.input)

    # BAND-WINDOW PARITY: the Casida pair basis must span the SAME band window the
    # GW chi0 (compute_screening) consumed = ALL occupied x ALL conduction bands
    # stored in the restart (n_val=n_occ, n_cond=nb-n_occ).  Default None -> pass a
    # large count so the loader clamps to the full window.  A smaller hand window
    # would drop transitions chi0 included and fail by construction.
    n_val = args.n_val if args.n_val is not None else 10**9
    n_cond = args.n_cond if args.n_cond is not None else 10**9

    with timing.section("w_exact.load"):
        data = load_bse_data_from_restart_sharded(
            restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
            input_file=args.input, inject_head=False,
            load_v_full=(args.compare_wq or args.w_omega_chain))

    n_rmu = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    z = (args.omega_ev + 1j * args.eta_ev) / args.ry_to_ev
    print(f"chi0 window: n_val={data['n_val']} n_cond={data['n_cond']} "
          f"(full occ x cond; matches GW compute_screening), "
          f"nk={int(data['nkx']*data['nky']*data['nkz'])}, N_mu={nlog} (padded {n_rmu})")
    print(f"omega={args.omega_ev} eV  eta={args.eta_ev} eV  z={z:.6e} Ry  "
          f"gmres(max_iter={args.gmres_max_iter}, tol={args.gmres_tol:g}); head-less bodies")

    if args.compare_wq:
        # Finite-q generalization: loop the symmetry-reduced (IBZ) q-grid one at a
        # time (NO batching across q), build the q-shifted screening data, resolve
        # W_q, and compare each against its OWN stored (W0_qmunu - V_qmunu)[q_flat]
        # tile (finite-q tiles are ~3% non-covariant under centroid permutation, so
        # NEVER validate by sym-unfolding between q's — always vs the own tile).
        q_list = _symmetry_reduced_q_list(args.input)
        nky_, nkz_ = int(data["nky"]), int(data["nkz"])
        # Build the q-INDEPENDENT resolvent engine ONCE.  matvec / gen / snapshot
        # depend only on (mesh, k-grid, pad sizes) — NOT on q — so they are shared
        # across every q; only the operand DATA (rolled psi_c/eps_c, the
        # V_qmunu[q] tile, the hoisted M's) and the preconditioner diagonal change,
        # and those flow as RUNTIME ARGS into the single compiled block-GMRES
        # engine.  Result: the engine compiles once (first q) and every later q is
        # dispatch-only (PHASE2_LOG "per-q recompile elimination").
        matvec, _, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)
        print(f"\nFinite-q W_q resolvent cross-check: {len(q_list)} symmetry-reduced "
              f"q-points (IBZ), one at a time; each vs its own (W0-V)[q_flat] tile")
        print("shared resolvent engine (matvec/gen/snapshot) built once; per-q "
              "operands (rolled psi_c/eps_c, V_q tile, diag_h) are runtime args\n")
        hdr = (f"{'iq':>3} {'q (kgrid)':>11} {'q_flat':>6} {'max_rel_err':>12} "
               f"{'median':>11} {'max_gmres':>11} {'build[s]':>9} {'solve[s]':>9}")
        print(hdr)
        print("-" * len(hdr))
        rel_by_q = []
        for iq, qv in enumerate(q_list):
            qx, qy, qz = int(qv[0]), int(qv[1]), int(qv[2])
            q_flat = qx * nky_ * nkz_ + qy * nkz_ + qz
            # BUILD: q-shifted operands + preconditioner diagonal + probe cols.
            t_b0 = time.perf_counter()
            with timing.section("w_exact.wq_build"):
                dq = build_finite_q_data(data, (qx, qy, qz), mesh_xy)
                ensure_W_R(dq, include_W=False)
                diag_hq = build_preconditioner_diagonal_sharded(
                    dq, mesh_xy, include_W=False, use_tda=False)
                W0 = np.asarray(jax.device_get(data["W_q"][:, :, qx, qy, qz]))
                Vq = np.asarray(jax.device_get(data["V_q_full"][:, :, qx, qy, qz]))
                T = W0 - Vq
                cols, _ = _select_compare_cols(T, nlog, args.n_cols, args.seed)
                jax.block_until_ready(diag_hq)
            t_build = time.perf_counter() - t_b0
            # SOLVE (trace+compile on the FIRST q, dispatch-only afterwards).
            t_s0 = time.perf_counter()
            with timing.section("w_exact.resolve_q"):
                W_tile, resids = _resolve_wc_columns(
                    cols, z, dq, matvec, diag_hq, gen, snapshot, sh,
                    max_iter=args.gmres_max_iter, tol=args.gmres_tol)
                jax.block_until_ready((W_tile, resids))
            t_solve = time.perf_counter() - t_s0
            # COMPARE: host-side rel_err vs the own (W0-V)[q_flat] tile.
            with timing.section("w_exact.wq_compare"):
                wc = np.asarray(jax.device_get(W_tile))
                rr = np.asarray(jax.device_get(resids))[:len(cols)]
                rels = np.asarray([
                    float(np.linalg.norm(wc[:nlog, i] - T[:nlog, int(nu0)])
                          / np.linalg.norm(T[:nlog, int(nu0)]))
                    for i, nu0 in enumerate(cols)])
            print(f"{iq:3d} {str((qx, qy, qz)):>11} {q_flat:6d} {rels.max():12.3e} "
                  f"{np.median(rels):11.3e} {rr.max():11.3e} {t_build:9.3f} {t_solve:9.3f}")
            rel_by_q.append(rels.max())
        print("-" * len(hdr))
        rel_by_q = np.asarray(rel_by_q)
        print(f"\nmax per-q rel_err = {rel_by_q.max():.3e}   median = "
              f"{np.median(rel_by_q):.3e}  (q=0 baseline ~2.5e-9). Closure at the GW "
              f"minimax-quadrature floor confirms W_q = v_q(0-H_RPA^q)^-1 v_q + v_q at "
              f"every symmetry-reduced q.")
        print("solve[s] column: first q carries the one-time engine compile; "
              "later q are warm dispatch (the per-q recompile is gone).")
        timing.report(print_fn=print, title="--- Timing ---")
        return

    if args.w_omega_chain:
        # Full-frequency W(omega) model: ONE block-Lanczos chain per q, then
        # evaluate across a frequency grid with no per-omega solve.  Validate
        # against the shifted-solve oracle at q=0 and the smallest nonzero IBZ q.
        if args.chain_freqs_ev:
            freqs = [complex(s) for s in args.chain_freqs_ev.split(",") if s.strip()]
        else:
            freqs = [0.0, 2j, 5j, 10j, 1.5 + 0.1j, 4.0 + 0.2j, 8.0 + 0.3j]
        if args.chain_sweep:
            sweep = [int(s) for s in args.chain_sweep.split(",") if s.strip()]
        else:
            sweep = [4, 8, 12, 16, args.chain_len]

        W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
        V0 = np.asarray(jax.device_get(data["V_q0"]))
        T0 = W0 - V0
        cols0, _ = _select_compare_cols(T0, nlog, args.n_cols, args.seed)
        with timing.section("w_exact.chain_q0"):
            run_w_omega_chain_compare(
                data, mesh_xy, cols0, freqs_ev=freqs, chain_len=args.chain_len,
                chain_sweep=sweep, ry_to_ev=args.ry_to_ev,
                gmres_max_iter=args.gmres_max_iter, gmres_tol=args.gmres_tol,
                disk_tile=T0, nlog=nlog, label="q=(0,0,0)")

        if args.chain_q is not None:
            q = tuple(int(x) for x in args.chain_q.split(","))
        else:
            q_list = _symmetry_reduced_q_list(args.input)
            nz = q_list[np.any(q_list != 0, axis=1)]
            q = (tuple(int(x) for x in nz[int(np.argmin((nz.astype(np.int64) ** 2).sum(axis=1)))])
                 if len(nz) else None)
        if q is not None and tuple(q) != (0, 0, 0):
            qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
            dq = build_finite_q_data(data, (qx, qy, qz), mesh_xy)
            W0q = np.asarray(jax.device_get(data["W_q"][:, :, qx, qy, qz]))
            Vq = np.asarray(jax.device_get(data["V_q_full"][:, :, qx, qy, qz]))
            Tq = W0q - Vq
            colsq, _ = _select_compare_cols(Tq, nlog, args.n_cols, args.seed)
            with timing.section("w_exact.chain_qfinite"):
                run_w_omega_chain_compare(
                    dq, mesh_xy, colsq, freqs_ev=freqs, chain_len=args.chain_len,
                    chain_sweep=sweep, ry_to_ev=args.ry_to_ev,
                    gmres_max_iter=args.gmres_max_iter, gmres_tol=args.gmres_tol,
                    disk_tile=Tq, nlog=nlog, label=f"q=({qx},{qy},{qz})")
        timing.report(print_fn=print, title="--- Timing ---")
        return

    matvec, diag_h, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)

    if args.compare_w0:
        # Head-less target: (W0_qmunu - V_qmunu) q=0 tile from the loaded bodies.
        W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
        V0 = np.asarray(jax.device_get(data["V_q0"]))
        T = W0 - V0
        if args.cols:
            cols = _parse_cols(args.cols, nlog, None, args.seed)
            col_norm = np.linalg.norm(T[:nlog, :nlog], axis=0)
        else:
            cols, col_norm = _select_compare_cols(T, nlog, args.n_cols, args.seed)
        print(f"\nW(0) resolvent cross-check: {len(cols)} columns "
              f"(largest-|W0-V| + random)\n")

        with timing.section("w_exact.resolve"):
            W_tile, resids = _resolve_wc_columns(
                cols, z, data, matvec, diag_h, gen, snapshot, sh,
                max_iter=args.gmres_max_iter, tol=args.gmres_tol)
        # W_tile is the device (mu_X, nu_Y) tile; column i = probe cols[i].
        wc = np.asarray(jax.device_get(W_tile))          # (n_rmu, n_pad)
        resids = np.asarray(jax.device_get(resids))      # (n_pad,)

        hdr = f"{'nu':>5} {'||(W0-V)_col||':>15} {'rel_err':>11} {'max|Delta|':>12} {'gmres_resid':>12}"
        print(hdr)
        print("-" * len(hdr))
        rel_all = []
        for i, nu0 in enumerate(cols):
            tcol = T[:nlog, int(nu0)]
            dcol = wc[:nlog, i] - tcol
            rel = float(np.linalg.norm(dcol) / np.linalg.norm(tcol))
            mx = float(np.max(np.abs(dcol)))
            rel_all.append(rel)
            print(f"{int(nu0):5d} {col_norm[int(nu0)]:15.4e} {rel:11.3e} "
                  f"{mx:12.3e} {resids[i]:12.3e}")
        rel_all = np.asarray(rel_all)
        resids = resids[:len(cols)]
        print("-" * len(hdr))
        print(f"max rel_err = {rel_all.max():.3e}   median = {np.median(rel_all):.3e}   "
              f"max gmres_resid = {resids.max():.3e}")
        print("\nInterpretation: W0_qmunu on disk is the RPA static screened "
              "Coulomb W(0) from chi0 = chi0(iw) minimax-quadratured to w=0; the "
              "resolvent uses the EXACT 1/(e_c-e_v) static denominator, so rel_err "
              "is the GW minimax-integration noise (solver residual is orders "
              "smaller). Closure at this floor confirms W0 = v(0-H_RPA)^-1 v + v.")
    else:
        cols = _parse_cols(args.cols, nlog, args.n_cols, args.seed)
        print(f"\nComputing {len(cols)} W_c(omega) column(s) of N_mu={nlog}")
        with timing.section("w_exact.resolve"):
            W_tile, resids = _resolve_wc_columns(
                cols, z, data, matvec, diag_h, gen, snapshot, sh,
                max_iter=args.gmres_max_iter, tol=args.gmres_tol)
        # Persist columns-first (n_cols, n_rmu), dropping the py zero-pad columns.
        wc = np.asarray(jax.device_get(W_tile))[:, :len(cols)].T  # (n_cols, n_rmu)
        resids = np.asarray(jax.device_get(resids))[:len(cols)]
        with h5py.File(args.out, "w") as h5:
            h5.attrs["omega_ev"] = float(args.omega_ev)
            h5.attrs["eta_ev"] = float(args.eta_ev)
            h5.attrs["ry_to_ev"] = float(args.ry_to_ev)
            h5.attrs["gmres_max_iter"] = int(args.gmres_max_iter)
            h5.attrs["gmres_tol"] = float(args.gmres_tol)
            h5.attrs["kernel"] = "rpa_screening_nonTDA"
            h5.create_dataset("columns", data=cols.astype(np.int32))
            h5.create_dataset("gmres_resid", data=resids)
            h5.create_dataset("Wc", data=wc)
        print(f"Wrote {len(cols)} Wc columns (max gmres_resid={resids.max():.2e}) "
              f"to {args.out}")

    timing.report(print_fn=print, title="--- Timing ---")


if __name__ == "__main__":
    raise SystemExit(main())
