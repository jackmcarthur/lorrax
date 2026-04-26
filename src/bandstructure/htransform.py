import os
import re
import argparse
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
# Allow GPU if available (previously forced CPU)

from runtime import init_jax_distributed
init_jax_distributed()

import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg
from jax.scipy.special import erf
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

from file_io import WFNReader
from common import symmetry_maps
from common import Meta
from common.load_wfns import (
    get_enk_bandrange, load_centroids_band_chunked, iter_psi_rchunk_bandwise,
)
from common.isdf_fitting import compute_L_q_from_CCT
from common.fft_helpers import make_flat_k_ifftn


def _build_mesh_xy() -> Mesh:
    """Most-square 2D ('x','y') mesh — matches gw_jax._build_mesh idiom."""
    total = jax.process_count() * jax.local_device_count()
    gx = int(np.sqrt(total))
    while gx > 1 and total % gx != 0:
        gx -= 1
    return Mesh(np.asarray(jax.devices()).reshape(gx, total // gx), ('x', 'y'))


_accum_G_cache: dict = {}


def streaming_galerkin_solve(wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
                             band_range: tuple[int, int],
                             rtol: float = 1e-8, log_fn=None,
                             band_chunk_size: int = 64,
                             bispinor: bool = False):
    """Galerkin projection using gw_jax shared loaders.

    Single ('x','y') mesh throughout. ψ at centroids comes from
    ``load_centroids_band_chunked``; ψ at full r is streamed via
    ``iter_psi_rchunk_bandwise`` (band+r chunked, sharded on 'y'). G is built
    sharded ``P('x','y')`` and Cholesky-factored via ``compute_L_q_from_CCT``
    so the 1×1-mesh dense path and the 2D-blocked distributed path both work.

    Args:
        wfn, sym, meta: standard gw_jax handles.
        centroid_indices: (n_μ, 3) FFT-grid coordinates.
        mesh_xy: 2D ``('x','y')`` device mesh.
        band_range: ``(b_start, b_end)`` — bands included in the SVD basis.
        rtol: SVD truncation tolerance (relative to s.max()).
        band_chunk_size: bands per FFT chunk inside the loader.
        bispinor: passed through to ``load_centroids_band_chunked``.

    Returns ``(S, ctilde)`` matching the legacy contract.
    """
    import time
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = band_range
    nb = b_end - b_start
    nk = meta.nk_tot
    nspinor = meta.nspinor
    n_mu = int(centroid_indices.shape[0])

    rep = NamedSharding(mesh_xy, P())               # fully replicated
    grid_xy = NamedSharding(mesh_xy, P('x', 'y'))   # (rank, rank) face

    log_fn(
        f"  Streaming Galerkin: nk={nk}, nb={nb}, nr={meta.n_rtot}, "
        f"n_mu={n_mu}, mesh=({mesh_xy.shape['x']}x{mesh_xy.shape['y']})"
    )

    # ── 1. Load ψ at centroids (band-sharded internally on 'y') ──
    t0 = time.time()
    psi_rmu_Y, _ = load_centroids_band_chunked(
        wfn, sym, meta, centroid_indices, bispinor, mesh_xy,
        band_range=band_range, band_chunk_size=band_chunk_size,
    )
    # psi_rmu_Y: (nk, nb, ns, n_μ), sharded P(None, None, None, 'y')
    log_fn(f"  load_centroids_band_chunked: {time.time()-t0:.2f}s")

    # ── 2. SVD of A = ψ@centroids reshaped to (nk·nb, ns·n_μ) ──
    # FFI seam: today gather to replicated, run dense SVD on each device
    # (small for our sizes); swap for distributed SVD when n_μ scales up.
    t1 = time.time()
    A = psi_rmu_Y.reshape(nk * nb, nspinor * n_mu)
    A = jax.device_put(A, rep)
    del psi_rmu_Y

    @partial(jax.jit, donate_argnums=(0,), out_shardings=(rep, rep, rep))
    def _svd_replicated(A):
        result = jnp.linalg.svd(A, full_matrices=False)
        return result.U, result.S, result.Vh

    U, s, Vh = _svd_replicated(A)
    del A
    s_host = np.asarray(s)
    rank = int((s_host > s_host.max() * rtol).sum())
    U = U[:, :rank]
    s = s[:rank]
    Vh = Vh[:rank, :]                        # (rank, ns·n_μ), keep for centroid recovery
    log_fn(f"  SVD of ({nk*nb}, {nspinor*n_mu}): rank={rank}, "
           f"σ_max={float(s_host[0]):.3e}, σ_min/{rtol:.0e}={float(s_host[rank-1]):.3e} "
           f"({time.time()-t1:.2f}s)")

    coeffs = U * s[None, :]                  # (nk·nb, rank), replicated
    inv_s = (1.0 / s)[:, None]               # (rank, 1), replicated
    UH = U.conj().T                          # (rank, nk·nb), replicated
    del U

    # ── 3. G = Σ_chunks Q_chunk Q_chunk^H, streaming over r-chunks ──
    # Inside each chunk:
    #   Q_chunk = inv_s · UH @ ψ_r_chunk   (psi sharded P(...,'y') on r-axis)
    #   G_chunk = Q_chunk @ Q_chunk^H      (rank,rank), psum over 'y'
    # JIT-cached by (n_rchunk, n_bchunk, rank).
    n_rtot = meta.fft_grid[0] * meta.fft_grid[1] * meta.fft_grid[2]
    band_chunk_ranges = [
        (b_start + i * band_chunk_size, min(b_start + (i + 1) * band_chunk_size, b_end))
        for i in range((nb + band_chunk_size - 1) // band_chunk_size)
    ]

    def _make_accum_kernel(rank_, bc_size, nspinor_, mesh_, sharding_y, sharding_xy):
        key = (id(mesh_), rank_, bc_size, nspinor_)
        fn = _accum_G_cache.get(key)
        if fn is not None:
            return fn

        @partial(jax.jit, donate_argnums=(2,), out_shardings=sharding_xy)
        def _accum(UH_bc, inv_s, psi_bc, G_in):
            # UH_bc: (rank, nk·bc) replicated; inv_s: (rank,1) replicated
            # psi_bc: (nk, bc, ns, r_chunk) sharded P(None,None,None,'y')
            # Reshape psi so the contraction axis is (nk·bc) and the
            # remaining axis is (ns·r_chunk). Spinor folds with r-axis,
            # matching the (ns·n_μ) column axis of A used in the SVD.
            nkv, bcv, nsv, rcv = psi_bc.shape
            psi_flat = psi_bc.reshape(nkv * bcv, nsv * rcv)  # (nk·bc, ns·r_chunk)
            Q = inv_s * (UH_bc @ psi_flat)                   # (rank, ns·r_chunk), sharded on 'y'
            Q = jax.lax.with_sharding_constraint(Q, sharding_y)
            return G_in + (Q @ Q.conj().T)                   # (rank, rank), sharded P('x','y')

        _accum_G_cache[key] = _accum
        return _accum

    sharding_y = NamedSharding(mesh_xy, P(None, 'y'))
    G = jax.device_put(jnp.zeros((rank, rank), dtype=jnp.complex128), grid_xy)

    t2 = time.time()
    chunk_count = 0
    UH_kb = UH.reshape(rank, nk, nb)  # for band-chunk slicing
    for bc_range, psi_bc_Y in iter_psi_rchunk_bandwise(
            wfn, sym, meta, mesh_xy, band_range, 0, meta.n_rtot, bispinor,
            band_chunk_size=band_chunk_size,
            band_chunk_ranges=band_chunk_ranges):
        bc = bc_range[1] - bc_range[0]
        bc_lo = bc_range[0] - b_start
        bc_hi = bc_range[1] - b_start
        UH_bc = UH_kb[:, :, bc_lo:bc_hi].reshape(rank, nk * bc)

        accum = _make_accum_kernel(rank, bc, nspinor, mesh_xy, sharding_y, grid_xy)
        G = accum(UH_bc, inv_s, psi_bc_Y, G)
        chunk_count += 1
    jax.block_until_ready(G)
    log_fn(f"  G accumulation: {chunk_count} chunk(s), {time.time()-t2:.2f}s")

    # ── 4. Cholesky on G ──
    # FFI seam: gw_jax's compute_L_q_from_CCT (which has dual 1×1-dense /
    # 2D-blocked paths) is the natural drop-in for distributed scaling, but
    # its 1e-14·trace ridge is calibrated for ISDF's huge-trace C_q matrices
    # and biases htransform's small-trace G by ~1e-5. Use raw Cholesky for
    # numerical parity with the legacy pipeline; swap to compute_L_q_from_CCT
    # (or its FFI variant) when n_μ scales to where rank-deficiency dominates.
    t3 = time.time()
    L = jnp.linalg.cholesky(G)
    log_fn(f"  Cholesky of G: {time.time()-t3:.2f}s")

    # ── 5. ctilde = coeffs · L + ortho diagnostic + B = L⁻¹V^H ──
    # B is the rank-α basis evaluated at the centroids:
    #   ψ_nk(r_μ, s) = Σ_α ctilde[k,n,α] · B[α, s, μ]
    # Math: with A = ψ at centroids = U s V^H, the Cholesky-orthogonalised
    # interpolation vectors at r_μ are exactly L^{-1} V^H (in the (ns, n_μ)
    # column index of V^H). This is what downstream Fourier-upscaling +
    # reconstruction needs to recover ψ at any new k at the centroids.
    @jax.jit
    def _finalize(coeffs, L, Vh):
        coeffs = coeffs @ L
        ctilde = coeffs.reshape(nk, nb, rank)
        CtC = ctilde[0] @ ctilde[0].conj().T
        ortho_err = jnp.max(jnp.abs(CtC - jnp.eye(nb, dtype=ctilde.dtype)))
        # B = L⁻¹ V^H, then split (ns·n_μ) → (ns, n_μ).
        B_flat = jsp_linalg.solve_triangular(L, Vh, lower=True)  # (rank, ns·n_μ)
        B = B_flat.reshape(rank, nspinor, n_mu)
        return ctilde, ortho_err, B

    ctilde, ortho_err, B_at_mu = _finalize(coeffs, L, Vh)
    ctilde = jax.device_put(ctilde, rep)
    B_at_mu = jax.device_put(B_at_mu, rep)
    S = jnp.eye(rank, dtype=jnp.complex128)
    log_fn(f"  ctilde[0] orthogonality error: {float(ortho_err):.3e}")
    log_fn(f"  Total Galerkin: {time.time()-t0:.2f}s")

    return S, ctilde, B_at_mu


def solve_q0_galerkin(
    psi_mu: jax.Array,
    psi_r: jax.Array,
    *,
    rtol: float = 1e-8,
    log_fn=None,
):
    """Solve psi_mu @ Q = psi_r using a Galerkin projection (spin folded into μ)."""
    psi_mu = jnp.asarray(psi_mu, dtype=jnp.complex128)
    psi_r = jnp.asarray(psi_r, dtype=psi_mu.dtype)
    if psi_mu.shape[:-1] != psi_r.shape[:-1]:
        raise ValueError("psi_mu and psi_r must share all leading dimensions")

    nk, nb, nspin, n_mu = psi_mu.shape
    psi_mu_state = psi_mu.reshape(nk * nb, nspin * n_mu)
    psi_r_state = psi_r.reshape(nk * nb, -1)

    U, s, Vh = jnp.linalg.svd(psi_mu_state, full_matrices=False)
    mask = s > (s.max() * rtol)
    if not bool(mask.any()):
        raise ValueError("SVD dropped all singular values; increase rtol")
    U = U[:, mask]
    s = s[mask]
    #basis = Vh[mask].conj().T
    coeffs = U * s
    Q_reduced, *_ = jnp.linalg.lstsq(coeffs, psi_r_state, rcond=rtol)
    G = Q_reduced @ Q_reduced.conj().T
    G = G + 1e-11 * jnp.eye(G.shape[0], dtype=G.dtype)
    chol = jnp.linalg.cholesky(G)
    Q_reduced = jsp_linalg.solve_triangular(chol, Q_reduced, lower=True)
    coeffs = coeffs @ chol

    if log_fn is not None:
        residual = jnp.linalg.norm((coeffs @ Q_reduced) - psi_r_state, axis=1)
        log_fn(
            "Galerkin residual min/median/max: "
            f"{float(residual.min()):.3e} / {float(jnp.median(residual)):.3e} / {float(residual.max()):.3e}"
        )

    #Q_full = basis @ Q_reduced # do not uncomment.
    S = Q_reduced @ Q_reduced.conj().T
    ctilde = coeffs.reshape(nk, nb, coeffs.shape[1])
    return  S, ctilde


def _f_params_from_energies(enk_nb_nk: jax.Array, top_band_index: int,
                            a_band_index: int | None = None) -> tuple[float, float, float]:
    """Compute f-transform parameters from eigenvalues.

    Parameters
    ----------
    top_band_index : int
        Index of the highest band (sets epsilon0 = max eigenvalue of this band).
    a_band_index : int or None
        Index of the band whose bandwidth sets `a = 4 * bandwidth`.
        If None, defaults to top_band_index (original behavior).
        Set this to the highest band you want to keep accurately.
    """
    E_top_k = enk_nb_nk[top_band_index]
    shift = float(jnp.max(E_top_k))
    if a_band_index is None:
        a_band_index = top_band_index
    E_a_k = enk_nb_nk[a_band_index]
    gap = 4.0 * float(jnp.max(E_a_k) - jnp.min(E_a_k) + 1e-14)
    n = 3.0
    return gap, n, shift


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _fun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    f_left = y + 0.5 * a
    arg = n * (0.5 + y / a)
    term1 = a * (jnp.exp(-(n * 0.5) ** 2) - jnp.exp(-((n * (a + 2 * y)) / (2 * a)) ** 2)) / (2 * n * jnp.sqrt(jnp.pi) * erf_half)
    term2 = (a + 2 * y) * (erf_half - erf(arg)) / (4 * erf_half)
    f_mid = term1 + term2
    f = jnp.where(cond_left, f_left, 0.0)
    f = jnp.where(cond_mid, f_mid, f)
    f = jnp.where(y >= 0, 0.0, f)
    return jnp.where(f > 0, 0.0, f)


def fun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Transform function f(x) — matches Fortran fun(). Works in unshifted space."""
    return _fun_jit(float(a), float(n), float(shift), x)


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _dfun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    df = jnp.zeros_like(y)
    df = jnp.where(cond_left, 1.0, df)
    arg = n * (0.5 + y / a)
    df = jnp.where(cond_mid, 0.5 - erf(arg) / (2 * erf_half), df)
    return jnp.where(y >= 0, 0.0, df)


def dfun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Derivative of transform function — matches Fortran dfun()."""
    return _dfun_jit(float(a), float(n), float(shift), x)


def f_transform_eigs(enk_nb_nk: jax.Array,
                     a_band_index: int | None = None) -> tuple[jax.Array, float, float, float]:
    """Apply f-transform to eigenvalues. Returns (f_eps, a, n, shift)."""
    nb, _ = enk_nb_nk.shape
    a, n, shift = _f_params_from_energies(enk_nb_nk, top_band_index=nb - 1,
                                          a_band_index=a_band_index)
    f_eps = fun(a, n, shift, enk_nb_nk)
    return f_eps, a, n, shift


def build_R_grid_np(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Lattice-R grid (nk, 3) matching the IFFT-shift convention used by
    ``make_flat_k_ifftn`` — symmetric around 0, with R_i ∈ {-n/2, ..., n/2-1}."""
    def _shift(n):
        a = np.arange(n, dtype=np.float64)
        return np.where(a >= (n + 1) // 2, a - n, a)
    return np.stack(np.meshgrid(
        _shift(kgrid[0]), _shift(kgrid[1]), _shift(kgrid[2]), indexing='ij'),
        axis=-1).reshape(-1, 3)


def build_fH_R(ctilde: jax.Array, enk_sigma: jax.Array,
               kgrid_co: tuple[int, int, int], mesh_xy: Mesh,
               *, a_band_index: int | None = None,
               log_fn=None):
    """f-transformed Hamiltonian in real-space lattice representation.

    Math (htransform paper):
        fH_k  = -Σ_n |sqrt(-f(ε_n,k))|² ctilde_n,k ctilde_n,k^H
              = Σ_n f(ε_n,k) ctilde_n,k ctilde_n,k^H
        fH_R  = (1/N_k) Σ_k e^{-2πi k·R} fH_k                          # IFFT
    where f(ε) is the smooth bandwidth-bound transform from
    ``f_transform_eigs`` (≤0 for ε<shift, =0 for ε≥shift). For any q,
    fH_q = Σ_R e^{-2πi q·R} fH_R recovers the rank-α-basis Hamiltonian
    whose eigenvalues are f(ε_n,q) and whose eigenvectors are c_n,q,
    enabling both bandstructure interpolation (eigvalsh + newton_inv on
    eigvals) and wfn recovery (eigh, then ψ_n,q(r_μ) = Σ_α c_n,q[α]·B[α,s,μ]).

    Args:
        ctilde:    (nk_co, nb, rank) Galerkin coefficients in the rank-α basis,
                   replicated. Output of ``streaming_galerkin_solve``.
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index whose bandwidth sets ``a``;
                   defaults to top of the htransform window (nb-1).
        log_fn:    optional logger.

    Returns:
        fH_k:    (nk_co, rank, rank), sharded P(None, 'x', 'y').
        fH_R:    (nk_co, rank, rank), sharded P(None, 'x', 'y') (lattice-R index).
        params:  (a_f, n_f, shift) — for ``newton_inv`` on the eigvals of fH_q.
        f_eps:   (nb, nk_co) f-transformed eigenvalues, replicated.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    f_eps, a_f, n_f, shift = f_transform_eigs(enk_sigma, a_band_index=a_band_index)
    log_fn(f"  f-transform: a={a_f:.6f} Ry, n={n_f:.2f}, shift={shift:.6f} Ry"
           + (f" (a from band {a_band_index})" if a_band_index is not None else ""))

    flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    spec_3d = P(None, None, None, 'x', 'y')
    # 'backward' = 1/N normalisation in IFFT — matches Σ_R e^{-2πik·R} fH_R = fH_k.
    local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid_co, spec_3d, norm='backward')

    @partial(jax.jit, out_shardings=(flat_xy, flat_xy))
    def _build(ctilde_in, f_eps_T):
        f_eps_ki = jnp.where(f_eps_T > 0, 0.0, f_eps_T)
        bw = jnp.sqrt(jnp.clip(-f_eps_ki, 0.0, None))
        weighted = ctilde_in * bw[..., None]
        fH_k = -jnp.einsum('kim,kin->kmn', weighted, jnp.conj(weighted), optimize=True)
        fH_k = 0.5 * (fH_k + jnp.swapaxes(fH_k, -1, -2).conj())
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_R = local_ifftn(fH_k)
        return fH_k, fH_R

    fH_k, fH_R = _build(ctilde, f_eps.T)
    return fH_k, fH_R, (a_f, n_f, shift), f_eps


def newton_inv(a: float, n: float, shift: float, y: jax.Array,
               max_iter: int = 50) -> jax.Array:
    """Newton inversion — matches Fortran newton_inv(). Works in unshifted space."""
    dxmax = a / 2.0
    x0 = y + shift  # Fortran initial guess: x = y + s

    def body_fun(_, x_curr):
        res = fun(a, n, shift, x_curr) - y
        df_val = dfun(a, n, shift, x_curr)
        dx = jnp.where(jnp.abs(df_val) > 1e-14, -res / df_val, 0.0)
        dx = jnp.clip(dx, -dxmax, dxmax)
        return x_curr + dx

    return lax.fori_loop(0, max_iter, body_fun, x0)


def load_wfns_and_enk_for_sigma(wfn, sym, nval: int, ncond: int, nband: int):
    nelec = int(wfn.nelec)
    nsigmarange = (int(nelec - nval), int(nelec + ncond))
    enk_sigma, _ = get_enk_bandrange(wfn, sym, nsigmarange, nsigmarange)
    return nsigmarange, jnp.asarray(enk_sigma).transpose(1, 0)


def setup_wfn_and_sym(wfn_file: str):
    wfn = WFNReader(wfn_file)
    sym = symmetry_maps.SymMaps(wfn)
    return wfn, sym


def _clean_label(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.strip()
    if not label:
        return None
    # Map common aliases to actual Unicode Greek letters so Matplotlib displays them
    if label.lower() in {'gg', 'gamma', '\\u0393'}:
        label = 'Γ'
    elif label.lower() in {'gl', 'lambda', '\\u039b'}:
        label = 'Λ'
    elif label.lower() in {'gs', 'sigma', '\\u03a3'}:
        label = 'Σ'
    return label


def generate_kpath_from_qe_segments(params: dict, wfn) -> tuple[jnp.ndarray, np.ndarray, list[str | None]] | None:
    seginfo = params.get("kpoints_crystal_b")
    if not seginfo:
        return None
    segments = seginfo.get("segments", [])
    if len(segments) < 2:
        return None
    nodes_crys = [np.asarray(seg["k"], dtype=float) for seg in segments]
    labels = [_clean_label(seg.get("label")) for seg in segments]
    pts_crys = [nodes_crys[0]]
    node_indices = [0]
    for i in range(len(nodes_crys) - 1):
        k0 = nodes_crys[i]
        k1 = nodes_crys[i + 1]
        n = max(1, int(segments[i + 1].get("n", 1)))
        for t in range(1, n + 1):
            alpha = t / float(n)
            pts_crys.append((1.0 - alpha) * k0 + alpha * k1)
        node_indices.append(len(pts_crys) - 1)
    kpoints = np.stack(pts_crys, axis=0)
    return jnp.asarray(kpoints, dtype=jnp.float64), np.asarray(node_indices, dtype=int), labels


def _load_centroids(centroids_path: str, fft_grid: tuple[int, int, int]) -> np.ndarray:
    centroids_frac = np.loadtxt(centroids_path, ndmin=2)
    if centroids_frac.size == 0:
        raise ValueError(f"Centroids file {centroids_path} is empty")
    fft_grid = np.asarray(fft_grid, dtype=int)
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int)
    centroid_indices = np.mod(centroid_indices, fft_grid)
    return centroid_indices


def _shift_indices(n: int) -> jnp.ndarray:
    arr = jnp.arange(n, dtype=jnp.float64)
    return jnp.where(arr >= (n + 1) // 2, arr - n, arr)


def _make_logger(verbose: bool):
    return print if verbose else (lambda *_, **__: None)


def read_eqp_energies(eqp_file: str, sym, band_window: tuple[int, int]) -> jax.Array:
    """Parse EQP (or sigX) energies from a BerkeleyGW-style file.

    Supports lines like either:
      - "n=10  EDFT=  ...  EQP= -12.3456 + 0.0000i"
      - "n=10  sigX=  -12.3456 + 0.000000i  VH= ..." (x-only files)

    Returns energies shaped as (nb, nk_full) matching the requested window.
    """
    start, end = int(band_window[0]), int(band_window[1])
    nb = int(max(0, end - start))
    if nb == 0:
        raise ValueError("Empty band window requested for EQP override")

    # Precompile regexes
    k_header = re.compile(r"^\s*k-point\s+(\d+)\s*:\s*$")
    n_line = re.compile(r"n\s*=\s*(\d+)")
    eqp_val = re.compile(r"EQP\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")
    sigx_val = re.compile(r"sigX\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

    # Accumulate per-k maps from absolute band index -> energy
    energies_by_k: list[dict[int, float]] = []
    current_k: int | None = None

    with open(eqp_file, "r", encoding="utf8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m_k = k_header.match(line)
            if m_k is not None:
                # Start new k-point block
                current_k = int(m_k.group(1))
                # Ensure list is long enough
                while len(energies_by_k) <= current_k:
                    energies_by_k.append({})
                continue

            if current_k is None:
                continue

            # Parse band index and energy value
            m_n = n_line.search(line)
            if m_n is None:
                continue
            band_idx = int(m_n.group(1))

            m_eqp = eqp_val.search(line)
            val = None
            if m_eqp is not None:
                val = float(m_eqp.group(1))
            else:
                m_sigx = sigx_val.search(line)
                if m_sigx is not None:
                    val = float(m_sigx.group(1))

            if val is not None:
                energies_by_k[current_k][band_idx] = val

    if not energies_by_k:
        raise ValueError(f"No k-point blocks found in {os.path.basename(eqp_file)}")

    nk = len(energies_by_k)
    if hasattr(sym, "nk_tot"):
        expected_nk = int(sym.nk_tot)
        if nk != expected_nk:
            raise ValueError(
                f"EQP file has nk={nk}, but symmetry maps expect nk={expected_nk}"
            )
    # Build (nk, nb) by slicing the absolute band indices [start:end]
    energies_window = np.zeros((nk, nb), dtype=np.float64)
    for ik in range(nk):
        kmap = energies_by_k[ik]
        for j, b_abs in enumerate(range(start, end)):
            if b_abs not in kmap:
                raise ValueError(
                    f"Missing EQP for band {b_abs} at k-point {ik} in {os.path.basename(eqp_file)}"
                )
            energies_window[ik, j] = kmap[b_abs]

    # Return (nb, nk) to match internal conventions
    return jnp.asarray(energies_window.T, dtype=jnp.float64)


def initialize_wfns(input_path: str, params: dict, log_fn, eqp_file: str | None = None,
                    mesh_xy: Mesh | None = None):
    from file_io.centroids import load_centroids as _shared_load_centroids

    input_dir = os.path.dirname(os.path.abspath(input_path))

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    wfn_file = _resolve(params["wfn_file"])
    wfn, sym = setup_wfn_and_sym(wfn_file)
    centroid_path = _resolve(params.get("centroids_file", "centroids_frac.txt"))
    _, centroid_indices, n_rmu = _shared_load_centroids(centroid_path, tuple(int(x) for x in wfn.fft_grid))

    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    bispinor = bool(params.get("bispinor", False))
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)
    nsigmarange, enk_sigma = load_wfns_and_enk_for_sigma(wfn, sym, nval, ncond, nband)

    # Optionally override energies with EQP values from a file only if explicitly requested via CLI
    if eqp_file:
        eqp_path = _resolve(eqp_file)
        if not os.path.isfile(eqp_path):
            log_fn(f"EQP file not found: {eqp_path}")
        else:
            try:
                enk_sigma = read_eqp_energies(eqp_path, sym, nsigmarange)
                log_fn(f"Using EQP energies from {os.path.basename(eqp_path)} for band window {nsigmarange}")
            except Exception as exc:
                log_fn(f"EQP override skipped for {os.path.basename(eqp_path)}: {exc}")

    if mesh_xy is None:
        mesh_xy = _build_mesh_xy()
    band_range = (int(nsigmarange[0]), int(nsigmarange[1]))
    with mesh_xy:
        S, ctilde, B_at_mu = streaming_galerkin_solve(
            wfn, sym, meta, centroid_indices, mesh_xy, band_range,
            rtol=1e-8, log_fn=log_fn, bispinor=bispinor,
        )
    log_fn(f"Loaded wavefunctions: nk={sym.nk_tot}, nb={band_range[1]-band_range[0]}, rank={ctilde.shape[2]}")
    return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma


def initialize_kpath(wfn, params):
    info = generate_kpath_from_qe_segments(params, wfn)
    if info is None:
        return None, None, None, None, []
    kpath_frac, node_indices, node_labels = info
    bvec = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    k_cart = np.asarray(kpath_frac) @ bvec * blat * (2.0 * np.pi)
    seg_len = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    x_path = np.concatenate([[0.0], np.cumsum(seg_len)])
    gamma_positions = [int(idx) for idx, lbl in zip(node_indices, node_labels) if (lbl or '').strip() == '\\u0393']
    return kpath_frac, x_path, node_indices, node_labels, gamma_positions


def h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log_fn, mesh_xy: Mesh,
                a_band_index: int | None = None):
    nk = int(meta.nkx * meta.nky * meta.nkz)
    states = ctilde.shape[1]
    rank = ctilde.shape[2]
    kgrid = (meta.nkx, meta.nky, meta.nkz)

    rep = NamedSharding(mesh_xy, P())  # fully replicated, used for diagnostics

    coeffs = ctilde.reshape(nk, states, rank)
    fH_k, fH_R, (a_f, n_f, shift), f_eps = build_fH_R(
        coeffs, enk_sigma, kgrid, mesh_xy, a_band_index=a_band_index, log_fn=log_fn)

    # Diagnostics. Split into two small jits:
    #   _diag_stats_fast  — sharding-respecting reductions on the full fH_k
    #     (no gather needed; psum on the (x,y) face).
    #   _diag_eig_at_gamma — eigvalsh on fH_k[0] only. Pull fH_k[0:1] to
    #     replicated FIRST (7.4 MB at our scale), so the eigvalsh runs
    #     single-device locally rather than driving an all-gather of the
    #     full (rank, rank) face inside the eigvalsh module.
    @jax.jit
    def _diag_stats_fast(fH_k):
        return (jnp.min(jnp.real(fH_k)),
                jnp.max(jnp.real(fH_k)),
                jnp.max(jnp.abs(jnp.imag(fH_k))))

    @jax.jit
    def _diag_eig_at_gamma(fH_k0_rep, f_eps_col0):
        eigs0 = jnp.sort(jnp.linalg.eigvalsh(fH_k0_rep))
        f_exp0 = jnp.sort(f_eps_col0)
        eig_err = jnp.max(jnp.abs(eigs0[:f_exp0.shape[0]] - f_exp0))
        return eigs0, f_exp0, eig_err

    re_min, re_max, im_max = _diag_stats_fast(fH_k)
    log_fn("fH_k real range: [{:.3e}, {:.3e}], |imag|max={:.3e}".format(
        float(re_min), float(re_max), float(im_max)))
    fH_k0_rep = jax.device_put(fH_k[0], rep)  # (rank, rank) replicated, ~7 MB
    eigs_fHk0, f_expected_0, fH_eig_err_arr = _diag_eig_at_gamma(fH_k0_rep, f_eps[:, 0])
    fH_eig_err = float(fH_eig_err_arr)
    log_fn(f"fH(k=0) eigenvalue error vs f(eps): {fH_eig_err:.6f} Ry = {fH_eig_err * 13.6057:.3f} eV")
    log_fn(f"  f(eps) first 5: {np.array(f_expected_0[:5])}")
    log_fn(f"  fH eig first 5: {np.array(eigs_fHk0[:5])}")
    log_fn(f"  f(eps) last 5:  {np.array(f_expected_0[-5:])}")
    log_fn(f"  fH eig last 5:  {np.array(eigs_fHk0[states-5:states])}")

    # S Cholesky (rank × rank, replicated, small) — bundle the symmetrise +
    # ridge + cholesky into one jit so each line isn't a separate compile.
    @jax.jit
    def _build_S_chol(S):
        S_sym = (S + S.conj().T) * 0.5
        S_sym += 1e-10 * jnp.mean(jnp.real(jnp.diag(S_sym))) * jnp.eye(rank, dtype=S_sym.dtype)
        return jnp.linalg.cholesky(S_sym)
    S_chol = _build_S_chol(S)

    R_grid = jnp.asarray(build_R_grid_np(kgrid))

    # ── Kpath-batch processing — Regime A ────────────────────────────────
    # Replicate fH_R upfront (~240 MB at our scale), so each batch's compute
    # is fully local and the only collective in the kpath section is the
    # one-shot upfront all-gather. Per-batch eigvalsh runs ndev-parallel via
    # batch-axis sharding. Production-scale Regime B (distributed eigh per
    # batch element) is the same code with a different out_sharding + FFI
    # swap and no upfront fH_R replication.
    fH_R_rep = jax.device_put(fH_R, rep)

    # Sharding specs for batched (bs, rank, rank) → (bs, rank) eigvalsh.
    batch_mat_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    batch_eig_shard = NamedSharding(mesh_xy, P(('x', 'y'), None))

    @partial(jax.jit, out_shardings=batch_eig_shard)
    def _kpath_batch(batch_k, fH_R_rep, S_chol):
        # batch_k: (bs, 3) replicated; fH_R_rep: (nk, rank, rank) replicated;
        # S_chol: (rank, rank) replicated. All inputs replicated → einsum
        # is local, only the final reshard for eigvalsh is collective.
        phase = jnp.exp(-2j * jnp.pi * (batch_k @ R_grid.T))           # (bs, nk)
        mat = 0.5 * jnp.einsum('bk,kij->bij', phase, fH_R_rep)         # (bs, rank, rank)
        mat = mat + jnp.swapaxes(mat, 1, 2).conj()
        mat = jax.lax.with_sharding_constraint(mat, batch_mat_shard)

        def _solve_one(m):
            y = jsp_linalg.solve_triangular(S_chol, m, lower=True)
            z = jsp_linalg.solve_triangular(S_chol, y, lower=True, trans=2)
            return jnp.linalg.eigvalsh((z + z.conj().T) * 0.5)

        return jax.vmap(_solve_one)(mat)

    # Round-trip diagnostic at Γ — single-batch via einsum (replicated input).
    q0 = jnp.zeros((1, 3), dtype=jnp.float64)

    @jax.jit
    def _gamma_rt(fH_R_rep):
        phase0 = jnp.exp(-2j * jnp.pi * (q0 @ R_grid.T))
        m = 0.5 * jnp.einsum('bk,kij->bij', phase0, fH_R_rep)
        return (m + jnp.swapaxes(m, 1, 2).conj())[0]

    fH_gamma_rt = _gamma_rt(fH_R_rep)
    diffs = jnp.max(jnp.abs(fH_k - fH_gamma_rt), axis=(1, 2))
    rt_err = float(jnp.min(diffs))
    log_fn(f"FFT Γ round-trip max error: {rt_err:.3e}")

    fermi_energy = float(wfn.efermi)
    nb_keep = int(f_eps.shape[0])

    kpath_frac, x_path, node_indices, node_labels, gamma_positions = kpath_data
    energies_on_path = None
    energies_sorted = None
    path_range = None
    gamma_exact = None

    if kpath_frac is not None:
        wrapped_k = (kpath_frac + 0.5) % 1.0 - 0.5
        # Pad nq to a multiple of batch_size — every batch has the same
        # shape, _kpath_batch compiles ONCE.
        batch_size = 32
        nq = wrapped_k.shape[0]
        n_pad = (-nq) % batch_size
        if n_pad:
            wrapped_k = jnp.concatenate(
                [wrapped_k, jnp.zeros((n_pad, 3), dtype=wrapped_k.dtype)], axis=0)
        nq_padded = wrapped_k.shape[0]
        lambda_q_list = []
        for i in range(0, nq_padded, batch_size):
            batch_eigs = _kpath_batch(wrapped_k[i:i+batch_size], fH_R_rep, S_chol)
            lambda_q_list.append(batch_eigs)
            jax.block_until_ready(batch_eigs)

        # Bundle concat + slice + vmap(newton_inv) + sort into ONE jit so the
        # post-loop processing emits one compile rather than 4 (concatenate,
        # sort, gather, vmap-newton).
        @partial(jax.jit, static_argnames=('nq', 'nb_keep'))
        def _post_kpath(batches, nq, nb_keep):
            lambda_q = jnp.concatenate(batches, axis=0)[:nq]
            energies = jax.vmap(lambda row: newton_inv(a_f, n_f, shift, row.real))(lambda_q)
            energies_sorted = jnp.sort(energies, axis=1)[:, :nb_keep]
            return energies, energies_sorted

        energies_on_path, energies_sorted_jax = _post_kpath(
            tuple(lambda_q_list), int(nq), int(nb_keep))
        energies_sorted = np.asarray(energies_sorted_jax)
        # Determine Fermi energy as the maximum along path of the wfn.nelec-th band (1-based -> 0-based)
        fermi_band_idx = int(wfn.nelec) - 1
        if 0 <= fermi_band_idx < energies_sorted.shape[1]:
            fermi_energy = float(np.max(energies_sorted[:, fermi_band_idx]))
        if not gamma_positions:
            gamma_positions = [int(jnp.argmin(jnp.linalg.norm(wrapped_k, axis=1)))]
        gamma_exact = np.sort(np.asarray(enk_sigma[:, 0]))[:nb_keep]
        # Shift all reported energies so VBM is at 0
        if energies_on_path is not None:
            energies_on_path = energies_on_path - fermi_energy
        if energies_sorted is not None:
            energies_sorted = energies_sorted - fermi_energy
        if gamma_exact is not None:
            gamma_exact = gamma_exact - fermi_energy
        # Recompute path range after shift and report
        path_range = (float(energies_sorted.min()), float(energies_sorted.max()))
        log_fn(f"Path energy range: {path_range[0]:.6f} to {path_range[1]:.6f} Ry (VBM@0)")
        # Γ deltas (in mRy) remain unchanged by uniform shift
        delta = (energies_sorted[gamma_positions[0]] - gamma_exact) * 1000.0
        log_fn("Γ Δε (mRy): " + ", ".join(f"{d:+.2f}" for d in delta[:6]))
        # After shifting, the Fermi level indicator is at 0
        fermi_energy = 0.0

    return {
        "nk_total": nk,
        "nb_keep": nb_keep,
        "fermi_energy": fermi_energy,
        "energies_on_path": energies_on_path,
        "energies_sorted": energies_sorted,
        "path_range": path_range,
        "gamma_exact": gamma_exact,
        "kpath_data": (kpath_frac, x_path, node_indices, node_labels, gamma_positions),
    }


def plot_bands(result):
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = result["kpath_data"]
    energies_sorted = result["energies_sorted"]
    gamma_exact = result["gamma_exact"]
    fermi_energy = result["fermi_energy"]
    nb_keep = result["nb_keep"]

    if kpath_frac is None or energies_sorted is None:
        raise RuntimeError("Plotting requires a K_POINTS {crystal_b} path in the input file")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    for band in range(nb_keep):
        ax.plot(x_path, energies_sorted[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = 'HT Γ' if pos_idx == 0 else None
        if gamma_exact is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_sorted[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Hamiltonian-transform bands')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    plt.show() 


def write_bands_to_file(output_path: str, energies_on_path, kpath_frac, x_path):
    if energies_on_path is None or kpath_frac is None or x_path is None:
        return
    energies = np.asarray(energies_on_path)
    kpoints = np.asarray(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy\n')
        for ik in range(energies.shape[0]):
            for ib in range(energies.shape[1]):
                kx, ky, kz = kpoints[ik]
                s_coord = x_path[ik]
                fh.write(f"{ik:4d} {ib:4d} {kx: .8f} {ky: .8f} {kz: .8f} {s_coord: .8f} {energies[ik, ib]: .8f}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hamiltonian interpolation driver")
    parser.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    parser.add_argument("-wfn", "--wfn-file", default=None, help="Override WFN file (e.g. WFN_qp.h5)")
    parser.add_argument("--plot", action="store_true", help="Show interpolated band plot")
    parser.add_argument("--eqp-file", default=None, help="Path to EQP/sigX file to override DFT band energies")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic details")
    parser.add_argument("--a-band", type=int, default=None,
                        help="Band index (0-based) whose bandwidth sets 'a'. "
                             "E.g. nval+ncond_keep-1. Default: top band.")
    args = parser.parse_args(argv)
    log = _make_logger(args.verbose)

    from gw.gw_init import read_cohsex_input
    params = read_cohsex_input(args.input)
    
    # Override WFN file if provided via CLI
    if args.wfn_file is not None:
        params["wfn_file"] = args.wfn_file
        log(f"Using WFN file from CLI: {args.wfn_file}")

    wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma = initialize_wfns(
        args.input, params, log, args.eqp_file)
    kpath_data = initialize_kpath(wfn, params)
    with mesh_xy:
        result = h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log, mesh_xy,
                             a_band_index=args.a_band)

    # Optional BSE interpolation handoff: fine-k wfns at coarse centroids.
    # Driven by ``get_centroids_fi`` + ``kgrid_fi`` + ``wfn_fi_{min,max}``;
    # see ``bandstructure.bse_setup.compute_wfns_fi`` for the contract.
    if params.get("get_centroids_fi", False):
        from .bse_setup import compute_wfns_fi
        b_min = int(params["wfn_fi_min"])
        b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])
        with mesh_xy:
            psi_rmu_Y, psi_rmuT_X, lam_fi, energies_fi = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                kgrid_fi=params["kgrid_fi"],
                band_window_fi=(b_min, b_max),
                mesh_xy=mesh_xy, a_band_index=args.a_band, log_fn=log,
            )
        log(f"BSE setup: psi_rmu_Y={psi_rmu_Y.shape} P{psi_rmu_Y.sharding.spec}, "
            f"psi_rmuT_X={psi_rmuT_X.shape} P{psi_rmuT_X.sharding.spec}")

    if args.plot:
        plot_bands(result)

    output_dir = os.path.dirname(os.path.abspath(args.input))
    write_bands_to_file(
        os.path.join(output_dir, 'bandstructure.dat'),
        result['energies_sorted'],  # Use sorted & truncated to nb_keep, not raw eigenvalues
        kpath_data[0],
        kpath_data[1],
    )

    summary = f"HT complete: {result['nb_keep']} bands, nk={result['nk_total']}, fermi={result['fermi_energy']:.6f} Ry"
    if result['path_range'] is not None:
        summary += f", path range [{result['path_range'][0]:.6f}, {result['path_range'][1]:.6f}] Ry"
    print(summary)
    return 0


if __name__ == "__main__":
    main()
