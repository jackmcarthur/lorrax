"""
Bispinor V^{μ_L, ν_L}_q driver — full Lorentz tensor over the 4 ζ files.

The bispinor pair density carries a 4-vector (μ_L) Lorentz index from
the Pauli-matrix decomposition of ψ†_iσ ψ_jσ′:

    n_iσ,jσ′(r) = Σ_{μ_L=0..3}  ζ_{μ_L,a}(r)  ⟨σ| τ^{μ_L} |σ′⟩  C^{a}_{ij}

where τ^0 = I and τ^{1,2,3} are the spatial Pauli matrices.  The
4-density ζ_q,μ_L(r) is fitted by the bispinor branch of
``isdf_fitting.fit_zeta_chunked_to_h5`` and persisted as four HDF5
files: ``zeta_q.h5`` (μ_L=0, scalar / time-like) and
``zeta_q_mu{1,2,3}.h5`` (μ_L=1,2,3, the spatial Lorentz components).

V^{μ_L, ν_L}_q couples the four channels via the bare Coulomb in the
Lorentz gauge:

    V^{μ_L, ν_L}_q(μ_c, ν_c) = Σ_K  conj(ζ_{μ_L,μ_c}(K))
                              · t_{μ_L, ν_L}(K) · v(|K|)
                              · ζ_{ν_L,ν_c}(K)

with K = q + G, and the (μ_L, ν_L)-dependent kernel weight

    t^{μ_L, ν_L}(K) = δ_{μ_L, 0} δ_{ν_L, 0}                    (scalar)
                   + δ_{μ_L, i} δ_{ν_L, j} (δ_{ij} − K̂_i K̂_j)   (transverse)

In Coulomb gauge the cross terms (0, i) and (i, 0) vanish identically,
so out of the 16 blocks only 10 are non-zero.  Of those 10, the kernel
is hermitian under the centroid axes:
    V^{i, j}(q; μ_c, ν_c) = V^{j, i}(q; ν_c, μ_c)*
so 7 unique kernel calls suffice; the remaining 3 are filled by
hermitian-transpose at the end.

Centroid counts
---------------
The four channels can have different centroid counts (e.g. CrI3 1800
scalar vs 1808 transverse).  Within each (μ_L, ν_L) tile both sides
are the same Lorentz channel, so the L/R centroid counts on a given
tile are equal — but channel-0 vs channel-i tiles can be sized
differently.  This driver allocates one V_acc per non-zero tile, sized
from ``n_rmu_by_channel``.

Public surface
--------------
    compute_all_V_q_lorentz_sharded(...) ->
        (V_blocks: dict[(μ_L, ν_L)] -> Array (Q, n_rmu_L, n_rmu_R)
                                         P(None, 'x', 'y'),
         G0_mu  : Array (Q, n_rmu_0)  P(None, 'x'))

The G0 head term is only populated by the (0,0) self-contraction; the
transverse channels do not contribute to the head correction (the
projector kills G=0 in Coulomb gauge).
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing


# Channel pairs for which we run a kernel call.  The remaining (i, 0)
# and (0, i) pairs are zero by Coulomb gauge; (j, i) for i < j ∈ {1,2,3}
# are filled by hermitian-transpose of (i, j) at the end.
_UNIQUE_TILES: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2), (3, 3),
    (1, 2), (1, 3), (2, 3),
)
_HERM_TILES: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((2, 1), (1, 2)),
    ((3, 1), (1, 3)),
    ((3, 2), (2, 3)),
)
_ZERO_TILES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3),
    (1, 0), (2, 0), (3, 0),
)


def _projector_weight(K_cart: jax.Array, mu_L: int, nu_L: int) -> jax.Array:
    """Lorentz transverse-projector weight on a (Q, n_G_sph, 3) K_cart.

    Returns (Q, n_G_sph) real array.

    For the scalar (0,0) tile this is just ``1`` (caller multiplies by
    v(K) outside).  For the spatial (i, j) tiles it is
    ``δ_ij − K̂_i K̂_j``.  We avoid 0/0 at K=0 by guarding |K|² with a
    tiny epsilon — at K=0 the transverse projector is undefined but
    the (1, 1)/(2, 2)/(3, 3) tiles see |K|²·sqrt = 0 (multiplied by
    v(0)=0 anyway via the Coulomb-gauge head treatment), so the value
    we return at K=0 is irrelevant.
    """
    if mu_L == 0 and nu_L == 0:
        Q, n_G, _ = K_cart.shape
        return jnp.ones((Q, n_G), dtype=jnp.float64)
    # Spatial indices are 1-based in the physics; translate to 0-based
    # axis indices on the cartesian K.
    i = mu_L - 1
    j = nu_L - 1
    K2 = jnp.sum(K_cart * K_cart, axis=-1)  # (Q, n_G_sph)
    K2_safe = jnp.where(K2 > 1e-30, K2, 1.0)
    # K̂_i K̂_j = K_i K_j / |K|²
    Khat_ij = K_cart[..., i] * K_cart[..., j] / K2_safe
    if mu_L == nu_L:
        # δ_ij − K̂_i² = 1 − K̂_i²
        return 1.0 - Khat_ij
    # i ≠ j  →  −K̂_i K̂_j
    return -Khat_ij


def compute_all_V_q_lorentz_sharded(
    *,
    zeta_io_by_channel: dict,
    coulomb_kernels,
    mesh_xy: Mesh,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    n_rmu_by_channel: dict,
    sys_dim: int = 2,
    bdot=None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff=None,
    bgw_v_grid_fn=None,
    budget_bytes=None,
    verbose: bool = True,
):
    """Compute the full bispinor V^{μ_L, ν_L}_q tensor.

    7 unique (μ_L, ν_L) tiles run through the unified V_q tile kernel
    (``v_q_tile.compute_V_q_tile``); 3 remaining non-zero tiles are
    filled by hermitian-transpose of the centroid axes; 6 tiles are
    zero by Coulomb gauge and not allocated.

    Parameters
    ----------
    zeta_io_by_channel : dict[int, SlabIO]
        Open SlabIO handles keyed by Lorentz channel (0, 1, 2, 3).
        Caller is responsible for opening/closing.  All four channels
        must be present.
    coulomb_kernels : SimpleNamespace
        Bundle from ``coulomb_kernel.make_v_munu_chunked_kernel``.  Must
        provide ``get_v_per_G_and_phase`` (for v + phase) and
        ``get_K_cart_on_sphere`` (for the projector).
    mesh_xy : Mesh
        2-D device mesh.
    kgrid, fft_grid : tuple[int, int, int]
    bvec : (3, 3) ndarray, Bohr⁻¹
    cell_volume : float, Bohr³
    n_rmu_by_channel : dict[int, int]
        Centroid count per Lorentz channel.  All four keys required.
    sys_dim : int  (2 = slab, 3 = bulk)
    bdot, mc_average_vcoul_body, bare_coulomb_cutoff, bgw_v_grid_fn :
        Plumbed through to be available if needed; the projector fix
        is applied on top of v(K), so the BGW vcoul overlay (if any)
        runs on the L=0 R=0 tile only.
    budget_bytes : float | None
        Memory budget per device (bytes).  Default 24 GB.
    verbose : bool

    Returns
    -------
    (V_blocks, G0_mu)
        V_blocks : dict[(μ_L, ν_L), jax.Array]
            10 entries (the 6 (0, i)/(i, 0) tiles are not present).
            Each Array has shape (Q, n_rmu_L, n_rmu_R) sharded
            P(None, 'x', 'y') on ``mesh_xy``.
        G0_mu : jax.Array, shape (Q, n_rmu_0), sharded P(None, 'x')
            ζ^{μ_L=0}_μ(G=0) — only the scalar channel writes G0; the
            transverse channels' projector kills G=0.
    """
    from .v_q_tile import compute_V_q_tile, _choose_v_q_chunks

    # --- Required channels present ------------------------------------------
    for ch in (0, 1, 2, 3):
        if ch not in zeta_io_by_channel:
            raise ValueError(f"v_q_lorentz: missing zeta_io for channel {ch}")
        if ch not in n_rmu_by_channel:
            raise ValueError(f"v_q_lorentz: missing n_rmu for channel {ch}")

    if coulomb_kernels.get_K_cart_on_sphere is None:
        raise ValueError(
            "v_q_lorentz: coulomb_kernels.get_K_cart_on_sphere is None — "
            "the bispinor projector requires sys_dim ∈ {2, 3}.")

    nkx, nky, nkz = kgrid
    nq_total = nkx * nky * nkz
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])

    if budget_bytes is None:
        budget_bytes = 24.0e9

    # --- v(q+G) and phase batches (vmap over Q) -----------------------------
    _vmapped_v_per_G_phase = jax.vmap(coulomb_kernels.get_v_per_G_and_phase)
    _vmapped_K_cart        = jax.vmap(coulomb_kernels.get_K_cart_on_sphere)

    def _v_per_G_real_batch(qvec_arr_jax):
        """Build v(q+G) (Q, n_G_sph) real-valued non-negative."""
        v_per_G_batch, _ = _vmapped_v_per_G_phase(qvec_arr_jax)
        # v_per_G is real ≥ 0 by construction; cast to f64 for the
        # projector multiply (which is real), and back to c128 below.
        return jnp.real(v_per_G_batch).astype(jnp.float64)

    def _phase_batch(qvec_arr_jax):
        _, phase_batch = _vmapped_v_per_G_phase(qvec_arr_jax)
        return phase_batch

    def _make_v_per_G_fn(mu_L: int, nu_L: int):
        """Build the per-tile ``v_per_G_fn`` closure.

        Returns a callable ``(qvec_batch_np) -> (Q, n_G_sph) c128``
        suitable for ``compute_V_q_tile``.
        """
        def _v_per_G_fn(qvec_batch_np):
            qvec_arr = jnp.asarray(qvec_batch_np, dtype=jnp.float64)
            v_real = _v_per_G_real_batch(qvec_arr)         # (Q, n_G_sph) f64
            if mu_L == 0 and nu_L == 0:
                # scalar Coulomb — projector ≡ 1.
                return v_real.astype(jnp.complex128)
            K_cart = _vmapped_K_cart(qvec_arr)             # (Q, n_G_sph, 3)
            t = _projector_weight(K_cart, mu_L, nu_L)      # (Q, n_G_sph) f64
            return (v_real * t).astype(jnp.complex128)
        return _v_per_G_fn

    def _phase_fn(qvec_batch_np):
        qvec_arr = jnp.asarray(qvec_batch_np, dtype=jnp.float64)
        return _phase_batch(qvec_arr)

    # --- Optional BGW v(q+G) overlay — only meaningful for the (0,0) tile.
    _sphere_idx_np = (np.asarray(coulomb_kernels.sphere_idx)
                      if coulomb_kernels.sphere_idx is not None else None)

    def _bgw_overlay(qvec_np_batch, v_per_G_batch):
        if bgw_v_grid_fn is None:
            return v_per_G_batch
        kgrid_a = np.array([nkx, nky, nkz], dtype=np.float64)
        v_np = np.asarray(v_per_G_batch).copy()
        for iq in range(qvec_np_batch.shape[0]):
            q_frac = qvec_np_batch[iq] / kgrid_a
            v_scaled_bgw = np.asarray(
                bgw_v_grid_fn(tuple(q_frac))).reshape(-1)
            if _sphere_idx_np is not None:
                v_scaled_bgw = v_scaled_bgw[_sphere_idx_np]
            v_native = np.real(v_np[iq])
            v_overlaid = np.where(
                v_scaled_bgw != 0.0,
                np.maximum(v_scaled_bgw, 0.0),
                v_native,
            ).astype(np.complex128)
            v_np[iq] = v_overlaid
        return jnp.asarray(v_np)

    # --- Per-tile chooser (one chooser per (n_rmu_L, n_rmu_R) shape) --------
    n_G_sph = int(coulomb_kernels.n_sph)

    def _choose_for(n_rmu_L: int, n_rmu_R: int) -> dict:
        # The chooser models n_rmu² for V_acc; the bispinor case has
        # n_rmu_L == n_rmu_R within every tile we run, so use either.
        n_rmu_eq = max(n_rmu_L, n_rmu_R)
        # Pass n_rtot so the chooser sizes the disk-read slab term
        # correctly (n_rtot can be 8-16× n_G_sph on systems with a
        # tight bare-Coulomb cutoff).
        n_rtot_local = int(np.prod(fft_grid))
        return _choose_v_q_chunks(
            n_rmu=n_rmu_eq, n_G=n_G_sph, n_q_total=nq_total,
            budget_bytes=budget_bytes, p_x=p_x, p_y=p_y,
            n_rtot=n_rtot_local,
        )

    # --- Allocators for V_acc + g0_acc per tile -----------------------------
    V_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh = NamedSharding(mesh_xy, P(None, 'x'))

    def _alloc_V(n_rmu_L: int, n_rmu_R: int):
        @partial(jax.jit, out_shardings=V_sh)
        def _z():
            return jnp.zeros((nq_total, n_rmu_L, n_rmu_R), dtype=jnp.complex128)
        return _z()

    def _alloc_g0(n_rmu_L: int):
        @partial(jax.jit, out_shardings=g0_sh)
        def _z():
            return jnp.zeros((nq_total, n_rmu_L), dtype=jnp.complex128)
        return _z()

    if verbose and jax.process_index() == 0:
        print(f"  V_q (bispinor): mesh={p_x}x{p_y}, "
              f"channels n_rmu={n_rmu_by_channel}, "
              f"7 unique tiles + 3 hermitian-transpose, "
              f"6 zero by Coulomb gauge")

    # FFT-redundancy optimization opportunity (deferred, see report
    # bispinor_pipeline_2026-05-04):
    # The 7 tiles do 4 + 3*2 = 10 forward FFTs of ζ-channels per q-batch
    # (4 same_zeta diag + 3 off-diag, each off-diag FFTs both inputs).
    # Caching post-FFT ζ_G across tile boundaries would save ~3 FFTs but
    # requires holding all 4 ζ_G slabs concurrently — doubles peak HBM in
    # the Case B regime where the chooser already split μ.  Skipped here.
    # Done inside a single jit'd super-kernel it's free; non-trivial
    # restructure (>200 lines).  Tracking issue: profile first.

    # --- Loop over the 7 unique tiles ---------------------------------------
    V_blocks: dict[tuple[int, int], jax.Array] = {}
    G0_mu = None

    with timing.section("compute_all_V_q_lorentz_sharded"):
        for (mu_L, nu_L) in _UNIQUE_TILES:
            zeta_L_io = zeta_io_by_channel[mu_L]
            zeta_R_io = zeta_io_by_channel[nu_L]
            n_rmu_L = int(n_rmu_by_channel[mu_L])
            n_rmu_R = int(n_rmu_by_channel[nu_L])
            same_zeta = (mu_L == nu_L)
            assert (mu_L == nu_L) or n_rmu_L == n_rmu_R, (
                f"v_q_lorentz: tile ({mu_L},{nu_L}) has unequal centroid "
                f"counts {n_rmu_L} vs {n_rmu_R}; chooser assumes equal.")
            write_g0 = (mu_L == 0 and nu_L == 0)

            choice = _choose_for(n_rmu_L, n_rmu_R)
            V_acc = _alloc_V(n_rmu_L, n_rmu_R)
            g0_acc = _alloc_g0(n_rmu_L) if write_g0 else None

            v_per_G_fn = _make_v_per_G_fn(mu_L, nu_L)
            overlay = _bgw_overlay if (mu_L == 0 and nu_L == 0) else None

            label = f"V_q_lorentz_({mu_L},{nu_L})"
            if verbose and jax.process_index() == 0:
                tag = "scalar (0,0)" if (mu_L, nu_L) == (0, 0) else (
                      "transverse diag" if mu_L == nu_L else "transverse off-diag")
                print(f"    tile ({mu_L},{nu_L}) [{tag}]: "
                      f"n_rmu={n_rmu_L}, same_zeta={same_zeta}, "
                      f"write_g0={write_g0}")

            with timing.section(label):
                V_acc, g0_acc = compute_V_q_tile(
                    coulomb_kernels=coulomb_kernels,
                    zeta_L_io=zeta_L_io,
                    zeta_R_io=zeta_R_io if not same_zeta else zeta_L_io,
                    v_per_G_fn=v_per_G_fn,
                    phase_fn=_phase_fn,
                    mesh_xy=mesh_xy,
                    kgrid=kgrid,
                    n_rmu_L=n_rmu_L,
                    n_rmu_R=n_rmu_R,
                    V_acc=V_acc,
                    g0_acc=g0_acc,
                    chooser_choice=dict(choice),
                    bgw_v_grid_overlay_fn=overlay,
                    verbose=False,  # outer driver already announced
                    timing_label=label,
                )

            V_blocks[(mu_L, nu_L)] = V_acc
            if write_g0:
                G0_mu = g0_acc

        # --- Hermitian-transpose fill ---------------------------------------
        # V^{j, i}(q; μ_c, ν_c) = V^{i, j}(q; ν_c, μ_c)*
        # Sharding swap (μ_c on x, ν_c on y) → (ν_c on x, μ_c on y) — the
        # transpose is a permutation of named axes; we re-pin to V_sh
        # afterwards (a no-op when the shapes are square, which they
        # are for the transverse off-diagonal tiles).
        for (jL, iL), (iL2, jL2) in _HERM_TILES:
            assert (iL2, jL2) in V_blocks
            src = V_blocks[(iL2, jL2)]
            herm = jnp.conj(jnp.transpose(src, (0, 2, 1)))
            herm = jax.lax.with_sharding_constraint(herm, V_sh)
            V_blocks[(jL, iL)] = herm

    if verbose and jax.process_index() == 0:
        print(f"  V_q (bispinor) done: {len(V_blocks)} non-zero tiles "
              f"(7 computed + 3 hermitian-transpose), "
              f"6 (0,i)/(i,0) tiles omitted by Coulomb gauge.")

    return V_blocks, G0_mu
