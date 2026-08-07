from __future__ import annotations

"""
Direct epsilon->W consistency check for 0D (cell-box truncated) Coulomb.

This is a debug utility for validating q=0 screened interaction construction
from BerkeleyGW `eps0mat.h5` in the 0D truncation case, where vcoul(G=0) is
finite and the q=0 "head" treatment is simpler than 2D slab.

What it does
------------
1) Reads eps^{-1}_{GG'}(q=0, iw) from `eps0mat.h5`.
2) Rebuilds vcoul(G) from the in-repo 0D truncation implementation
   (`compute_vcoul_box`) and compares against vcoul stored in eps file.
3) Applies optional q=0 head fixes:
   - set v(G=0) to an override/auto value
   - optionally patch one eps^{-1} column so W_{00} is exact for the chosen
     construction
4) Constructs only:
   - W_left = diag(v) @ eps^{-1}
5) Prints per-frequency Frobenius norms, Hermiticity defects, and head/wing
   diagnostics.
6) Optionally computes diagonal Sigma matrix elements directly in G-space using
   BGW-style FFT matrix elements:
      M_{n'n}(G) = (1/Np) FFT_+[ conj(FFT_+(c_n)) * FFT_+(c_{n'}) ](G)
   and reports BGW-style decomposed terms:
      X, (SEX-X)_GN(E_n), COH_GN(E_n), (SEX-X)_static, COH_static.

Notes
-----
- This script is intentionally GG-space only; it does not project to ISDF μν.
- For sys_dim=0, `compute_q0_averages` returns vcoul0 == wcoul0 by construction.
- This script intentionally uses the left-multiplied construction only.

BGW-reproduction notes (important)
----------------------------------
For this checker, closest agreement to BerkeleyGW Sigma output is obtained when:
1) eps^{-1} matrices from HDF5 are used in transposed GG' orientation before
   screening contractions (`_to_bgw_eps_orientation`).
2) a small broadening eta (e.g. 0.5 eV) is included in all dynamic GN
   denominators for SX/CH terms (the dynamic branches only).
3) invalid GN modes are skipped in the default "main" GN columns (matching
   BGW invalid_gpp_mode=0 behavior), while optional debug columns can report
   the additional corrections from:
   - static fallback on invalid modes (BGW-like invalid_gpp_mode=3), and
   - 2 Ry fallback on invalid modes.
"""

import argparse
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from vcoul import compute_vcoul_box
from gw.gw_init import read_cohsex_input
from gw.vcoul import compute_q0_averages
from file_io import EPSReader, WFNReader, resolve_input_paths

RYD2EV = 13.6056980659


def _fro_norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, ord="fro"))


def _rel_hermiticity_defect(a: np.ndarray) -> float:
    denom = _fro_norm(a)
    if denom <= 0.0:
        return 0.0
    return _fro_norm(a - a.conj().T) / denom


def _parse_freq_list(freq_arg: str, nfreq: int) -> list[int]:
    s = (freq_arg or "").strip().lower()
    if s in ("", "all", "*"):
        return list(range(nfreq))
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        idx = int(tok)
        if idx < 0 or idx >= nfreq:
            raise ValueError(f"Frequency index out of range: {idx} (nfreq={nfreq})")
        out.append(idx)
    if not out:
        raise ValueError("No valid frequency indices parsed")
    return out


def _get_gvecs_in_eps_order(eps: EPSReader, iq: int) -> np.ndarray:
    nmtx = int(eps.nmtx[iq])
    gind = np.asarray(eps.gind_eps2rho[iq, :nmtx], dtype=np.int64)
    comps = np.asarray(eps.comps, dtype=np.int32)
    if np.max(gind) >= comps.shape[0]:
        raise ValueError(
            f"eps g-index exceeds available components: max(gind)={int(np.max(gind))}, "
            f"ncomps={comps.shape[0]}"
        )
    return np.asarray(comps[gind, :], dtype=np.int32)


def _g0_index(gvecs: np.ndarray) -> int:
    hits = np.where(np.all(gvecs == 0, axis=1))[0]
    if hits.size == 0:
        raise ValueError("Could not locate G=(0,0,0) in eps basis")
    return int(hits[0])


def _build_v_0d_from_repo(wfn: WFNReader, gvecs_eps: np.ndarray) -> np.ndarray:
    v_raw = compute_vcoul_box(
        np.asarray(wfn.bdot, dtype=np.float64),
        np.asarray(wfn.fft_grid, dtype=np.int32),
        np.asarray(gvecs_eps, dtype=np.int32),
    )
    return np.asarray(v_raw / float(wfn.cell_volume), dtype=np.complex128)


def _head_values_0d(wfn: WFNReader, v_head_override: float | None, w_head_override: float | None) -> tuple[complex, complex]:
    meta = SimpleNamespace(sys_dim=0)
    vc0, wc0 = compute_q0_averages(
        wfn=wfn,
        epshead=np.asarray(0.0, dtype=np.complex128),
        meta=meta,
    )
    v_head = complex(vc0) if v_head_override is None else complex(float(v_head_override))
    w_head = complex(wc0) if w_head_override is None else complex(float(w_head_override))
    return v_head, w_head


def _head_patch_epsinv_column(
    epsinv: np.ndarray,
    g0_idx: int,
    v_g0: complex,
    w_head: complex,
) -> np.ndarray:
    out = np.array(epsinv, copy=True)
    if abs(v_g0) <= 1.0e-20:
        raise ValueError("Cannot head-patch epsinv column: v(G=0) is ~0")
    out[:, g0_idx] = 0.0 + 0.0j
    out[g0_idx, g0_idx] = w_head / v_g0
    return out


def _fit_gn_ppm_on_epsinv(
    epsinv0: np.ndarray,
    epsinvi: np.ndarray,
    *,
    omega_p_ry: float,
    fallback_omega_ry: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit GN-PPM on the BGW Sigma convention: I_eps = (I - eps^{-1})."""
    ident = np.eye(epsinv0.shape[0], dtype=np.complex128)
    d0 = ident - epsinv0
    di = ident - epsinvi
    denom = d0 - di
    safe = np.abs(denom) > 1.0e-14
    ratio = np.zeros_like(d0.real)
    ratio[safe] = np.real(di[safe] / denom[safe])
    valid = np.isfinite(ratio) & (ratio > 0.0)

    omega = np.full_like(ratio, float(fallback_omega_ry), dtype=np.float64)
    if np.any(valid):
        omega[valid] = float(omega_p_ry) * np.sqrt(ratio[valid])
    b = -0.5 * d0 * omega
    invalid_pct = 100.0 * (1.0 - float(np.mean(valid.astype(np.float64))))
    return omega, b, valid, invalid_pct


def _to_bgw_eps_orientation(eps_mat: np.ndarray) -> np.ndarray:
    """Transpose eps matrix to BGW contraction orientation.

    In this checker, screened contractions match BGW outputs when the HDF5
    matrix is used with transposed GG' ordering.
    """
    return np.asarray(eps_mat.T, dtype=np.complex128)


def _get_band_energies_ry(wfn: WFNReader, ik: int = 0, ispin: int = 0) -> np.ndarray:
    e = np.asarray(wfn.energies)
    if e.ndim == 3:
        return np.asarray(e[int(ispin), int(ik), :], dtype=np.float64)
    if e.ndim == 2:
        return np.asarray(e[int(ik), :], dtype=np.float64)
    if e.ndim == 1:
        return np.asarray(e, dtype=np.float64)
    raise ValueError(f"Unexpected energies shape: {e.shape}")


def _g_to_fft_indices(gvecs: np.ndarray, nfft: np.ndarray) -> np.ndarray:
    idx = np.asarray(gvecs, dtype=np.int64).copy()
    for d in range(3):
        neg = idx[:, d] < 0
        idx[neg, d] += int(nfft[d])
    if np.any(idx < 0) or np.any(idx[:, 0] >= int(nfft[0])) or np.any(idx[:, 1] >= int(nfft[1])) or np.any(idx[:, 2] >= int(nfft[2])):
        raise ValueError("Mapped FFT indices out of range.")
    return idx


def _fft_plus_jax(arr: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    """BGW sign convention: FFT_+(x) = sum_j x(j) exp(+i j·p), unnormalized."""
    npts = 1
    for ax in axes:
        npts *= int(arr.shape[ax])
    return jnp.fft.ifftn(arr, axes=axes) * jnp.asarray(float(npts), dtype=jnp.float64)


def _load_bands_fft_plus_jax(
    wfn: WFNReader,
    *,
    ik: int,
    band_indices: np.ndarray,
    nfft: np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Load selected bands and return FFT_+(psi) on device plus eps-order FFT indices."""
    band_indices = np.asarray(band_indices, dtype=np.int64)
    start = int(wfn.kpt_starts[int(ik)])
    ngk = int(wfn.ngk[int(ik)])
    end = start + ngk

    coeff_raw = np.asarray(wfn.coeffs[band_indices, :, start:end, :], dtype=np.float64)
    coeff_np = coeff_raw[..., 0] + 1j * coeff_raw[..., 1]  # (nb, nspinor, ngk)
    nspinor = int(coeff_np.shape[1])
    gvec_k = np.asarray(wfn.get_gvec_nk(int(ik)), dtype=np.int64)
    if gvec_k.shape[0] != ngk:
        raise ValueError(f"ngk mismatch: gvec rows={gvec_k.shape[0]}, ngk={ngk}")
    fft_idx_np = _g_to_fft_indices(gvec_k, nfft)

    coeff = jax.device_put(jnp.asarray(coeff_np, dtype=jnp.complex128))
    fft_idx = jax.device_put(jnp.asarray(fft_idx_np, dtype=jnp.int32))
    nb = int(coeff.shape[0])
    nx, ny, nz = int(nfft[0]), int(nfft[1]), int(nfft[2])
    boxes = jnp.zeros((nb, nspinor, nx, ny, nz), dtype=jnp.complex128)

    b_idx = jnp.arange(nb, dtype=jnp.int32)[:, None, None]
    s_idx = jnp.arange(nspinor, dtype=jnp.int32)[None, :, None]
    ix = fft_idx[:, 0][None, None, :]
    iy = fft_idx[:, 1][None, None, :]
    iz = fft_idx[:, 2][None, None, :]
    boxes = boxes.at[b_idx, s_idx, ix, iy, iz].set(coeff)
    psi_fft = _fft_plus_jax(boxes, axes=(-3, -2, -1))
    return psi_fft, fft_idx


def _build_m_matrix_for_target_jax(
    psi_fft: jax.Array,
    target_local_idx: int,
    eps_fft_idx: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    """Build M_{n'n}(G) for all n' in the loaded band window at fixed target n."""
    ref = jnp.conj(jax.lax.dynamic_index_in_dim(psi_fft, target_local_idx, axis=0, keepdims=False))
    prod_r = jnp.einsum("sxyz,msxyz->mxyz", ref, psi_fft, optimize=True)  # (nb,nx,ny,nz)
    prod_g = _fft_plus_jax(prod_r, axes=(1, 2, 3))
    return scale * prod_g[:, eps_fft_idx[:, 0], eps_fft_idx[:, 1], eps_fft_idx[:, 2]]


@partial(jax.jit, static_argnames=("n_valence", "n_sum_states"))
def _sigma_terms_kernel_jit(
    psi_fft: jax.Array,
    eps_fft_idx: jax.Array,
    energies_all: jax.Array,
    energies_solve: jax.Array,
    epsinv0: jax.Array,
    v_vec: jax.Array,
    omega_fit: jax.Array,
    i_eps0: jax.Array,
    n_valence: int,
    n_sum_states: int,
    eta_ry: float,
    scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Jitted diagonal SEX/COH evaluator using BGW-style M tensors."""
    nsolve = int(energies_solve.shape[0])
    nG = int(v_vec.shape[0])
    eye = jnp.eye(nG, dtype=jnp.complex128)
    k_sx_static = epsinv0
    k_ch_static = 0.5 * (epsinv0 - eye)

    w2 = jnp.asarray(omega_fit, dtype=jnp.float64) ** 2
    r_mat = jnp.asarray(i_eps0, dtype=jnp.complex128) * w2
    wtilde = jnp.asarray(omega_fit, dtype=jnp.float64)
    v_vec = jnp.asarray(v_vec, dtype=jnp.complex128)
    scale_j = jnp.asarray(scale, dtype=jnp.float64)
    eta = jnp.asarray(1j, dtype=jnp.complex128) * jnp.asarray(eta_ry, dtype=jnp.float64)
    tiny = jnp.asarray(1.0e-18, dtype=jnp.float64)

    sx_dyn = jnp.zeros((nsolve,), dtype=jnp.complex128)
    ch_dyn = jnp.zeros((nsolve,), dtype=jnp.complex128)
    sx_static = jnp.zeros((nsolve,), dtype=jnp.complex128)
    ch_static = jnp.zeros((nsolve,), dtype=jnp.complex128)
    x_bare = jnp.zeros((nsolve,), dtype=jnp.complex128)

    def body_target(it, carry):
        sx_dyn, ch_dyn, sx_static, ch_static, x_bare = carry
        m_all_g = _build_m_matrix_for_target_jax(psi_fft, it, eps_fft_idx, scale_j)  # (n_sum, nG)
        m_all_g_conj = jnp.conj(m_all_g)
        delta_all = jnp.asarray(energies_solve[it] - energies_all, dtype=jnp.complex128)
        m_occ_g = m_all_g[: int(n_valence), :]
        vecw_occ = m_all_g_conj[: int(n_valence), :] * v_vec[None, :]
        x_t = -jnp.einsum("mg,mg->", m_occ_g, vecw_occ, optimize=True)
        rhs_sx_static_occ = jnp.einsum("gh,mh->mg", k_sx_static, vecw_occ, optimize=True)
        sx_s_t = -jnp.einsum("mg,mg->", m_occ_g, rhs_sx_static_occ, optimize=True)

        vecw_all = m_all_g_conj * v_vec[None, :]
        rhs_ch_static_all = jnp.einsum("gh,mh->mg", k_ch_static, vecw_all, optimize=True)
        ch_s_t = jnp.einsum("mg,mg->", m_all_g, rhs_ch_static_all, optimize=True)

        def body_occ(m, sx_d_t):
            vec = m_occ_g[m]
            vecw = vecw_occ[m]
            d_dyn = (delta_all[m] + eta) ** 2 - w2
            rd_dyn = jnp.where(jnp.abs(d_dyn) > tiny, r_mat / d_dyn, 0.0 + 0.0j)
            rhs_dyn = vecw + jnp.einsum("gh,h->g", rd_dyn, vecw, optimize=True)
            sx_d_t = sx_d_t - jnp.einsum("g,g->", vec, rhs_dyn, optimize=True)
            return sx_d_t

        sx_d_t = jax.lax.fori_loop(
            0,
            int(n_valence),
            body_occ,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )

        def body_all(m, ch_d_t):
            vec = m_all_g[m]
            vecw = vecw_all[m]
            d_dyn = wtilde * (delta_all[m] + eta - wtilde)
            rd_dyn = jnp.where(jnp.abs(d_dyn) > tiny, 0.5 * r_mat / d_dyn, 0.0 + 0.0j)
            rhs_dyn = jnp.einsum("gh,h->g", rd_dyn, vecw, optimize=True)
            ch_d_t = ch_d_t + jnp.einsum("g,g->", vec, rhs_dyn, optimize=True)
            return ch_d_t

        ch_d_t = jax.lax.fori_loop(
            0,
            int(n_sum_states),
            body_all,
            jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128),
        )

        sx_dyn = sx_dyn.at[it].set(sx_d_t)
        sx_static = sx_static.at[it].set(sx_s_t)
        ch_dyn = ch_dyn.at[it].set(ch_d_t)
        ch_static = ch_static.at[it].set(ch_s_t)
        x_bare = x_bare.at[it].set(x_t)
        return sx_dyn, ch_dyn, sx_static, ch_static, x_bare

    return jax.lax.fori_loop(0, nsolve, body_target, (sx_dyn, ch_dyn, sx_static, ch_static, x_bare))


def _compute_sigma_terms_gspace(
    *,
    wfn: WFNReader,
    epsinv0: np.ndarray,
    epsinvi: np.ndarray,
    v_vec: np.ndarray,
    gvecs_eps: np.ndarray,
    omega_fit: np.ndarray,
    i_eps0: np.ndarray,
    band_offset: int,
    n_valence: int,
    n_cond_solve: int,
    n_sum_states: int,
    eta_ry: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute diagonal SEX/COH terms (dynamic GN and static COHSEX) using jitted JAX kernels."""
    nsolve = int(n_valence + n_cond_solve)
    if nsolve <= 0:
        raise ValueError("n_valence + n_cond_solve must be > 0 for sigma table.")
    if n_sum_states < nsolve:
        raise ValueError("n_sum_states must be >= n_valence + n_cond_solve.")
    b0 = int(band_offset)
    bsum1 = b0 + int(n_sum_states)
    if bsum1 > int(wfn.nbands):
        raise ValueError(f"Requested bands exceed WFN range: end={bsum1}, nbands={int(wfn.nbands)}")

    band_sum = np.arange(b0, bsum1, dtype=np.int64)
    band_solve = band_sum[:nsolve]
    energies_all = _get_band_energies_ry(wfn, ik=0, ispin=0)[band_sum]
    energies_solve = energies_all[:nsolve]

    nfft = np.asarray(wfn.fft_grid, dtype=np.int64)
    scale = 1.0 / float(np.prod(nfft))
    psi_fft, fft_idx_k = _load_bands_fft_plus_jax(wfn, ik=0, band_indices=band_sum, nfft=nfft)
    eps_fft_idx_np = _g_to_fft_indices(gvecs_eps, nfft)
    eps_fft_idx = jax.device_put(jnp.asarray(eps_fft_idx_np, dtype=jnp.int32))

    nG = int(np.asarray(epsinv0).shape[0])
    if nG != int(np.asarray(v_vec).shape[0]):
        raise ValueError("v vector and eps matrix size mismatch.")
    del fft_idx_k  # only used internally to build psi_fft

    energies_all_d = jax.device_put(jnp.asarray(energies_all, dtype=jnp.float64))
    energies_solve_d = jax.device_put(jnp.asarray(energies_solve, dtype=jnp.float64))
    epsinv0_d = jax.device_put(jnp.asarray(epsinv0, dtype=jnp.complex128))
    v_vec_d = jax.device_put(jnp.asarray(v_vec, dtype=jnp.complex128))
    omega_fit_d = jax.device_put(jnp.asarray(omega_fit, dtype=jnp.float64))
    i_eps0_d = jax.device_put(jnp.asarray(i_eps0, dtype=jnp.complex128))

    sx_dyn_d, ch_dyn_d, sx_static_d, ch_static_d, x_bare_d = _sigma_terms_kernel_jit(
        psi_fft=psi_fft,
        eps_fft_idx=eps_fft_idx,
        energies_all=energies_all_d,
        energies_solve=energies_solve_d,
        epsinv0=epsinv0_d,
        v_vec=v_vec_d,
        omega_fit=omega_fit_d,
        i_eps0=i_eps0_d,
        n_valence=int(n_valence),
        n_sum_states=int(n_sum_states),
        eta_ry=float(eta_ry),
        scale=float(scale),
    )
    sx_dyn_d.block_until_ready()
    return (
        band_solve,
        energies_solve,
        np.asarray(jax.device_get(x_bare_d), dtype=np.complex128),
        np.asarray(jax.device_get(sx_dyn_d), dtype=np.complex128),
        np.asarray(jax.device_get(ch_dyn_d), dtype=np.complex128),
        np.asarray(jax.device_get(sx_static_d), dtype=np.complex128),
        np.asarray(jax.device_get(ch_static_d), dtype=np.complex128),
    )


def run(args: argparse.Namespace) -> int:
    params = read_cohsex_input(args.input)
    input_dir = args.input.rsplit("/", 1)[0] if "/" in args.input else "."
    resolve_input_paths(params, input_dir)

    sys_dim = int(params.get("sys_dim", 2))
    if sys_dim != 0:
        raise ValueError(f"This checker is for sys_dim=0 only (got sys_dim={sys_dim})")

    wfn = WFNReader(str(params["wfn_file"]))
    eps = EPSReader(args.eps0mat)

    if int(eps.matrix_type) != 0:
        print(f"Warning: eps matrix_type={int(eps.matrix_type)} (expected 0 for eps^-1)")
    iq = int(args.iq)
    nmtx = int(eps.nmtx[iq])
    gvecs = _get_gvecs_in_eps_order(eps, iq)
    g0_idx = _g0_index(gvecs)

    if args.nmtx_max and int(args.nmtx_max) > 0:
        nuse = min(nmtx, int(args.nmtx_max))
        gvecs = gvecs[:nuse]
        nmtx = nuse
        if g0_idx >= nmtx:
            raise ValueError("nmtx_max truncated out G=0 element")

    v_repo = _build_v_0d_from_repo(wfn, gvecs[:nmtx])
    v_eps = np.asarray(eps.vcoul[iq, :nmtx], dtype=np.complex128)

    v_head, w_head = _head_values_0d(wfn, args.v_head_au, args.w_head_au)
    v = np.array(v_repo, copy=True)
    v[g0_idx] = v_head

    print("")
    print("0D eps->W direct check (q=0)")
    print(f"  input:      {args.input}")
    print(f"  eps0mat:    {args.eps0mat}")
    print(f"  nmtx used:  {nmtx} (of {int(eps.nmtx[iq])})")
    print(f"  freq count: {int(eps.nfreq)}")
    print(f"  G0 index:   {g0_idx}")
    print(f"  head(v):    {v_head.real:.12f} a.u.")
    print(f"  head(W):    {w_head.real:.12f} a.u.")

    v_eps_scaled = v_eps / float(wfn.cell_volume)
    dv_scaled = v_repo - v_eps_scaled
    v_rel_scaled = np.linalg.norm(dv_scaled) / max(np.linalg.norm(v_eps_scaled), 1.0e-30)
    dv_raw = v_repo - v_eps
    v_rel_raw = np.linalg.norm(dv_raw) / max(np.linalg.norm(v_eps), 1.0e-30)
    print(
        f"  vcoul(repo vs eps/vol): abs={np.linalg.norm(dv_scaled):.6e}, rel={v_rel_scaled:.6e}"
    )
    print(f"  vcoul(repo vs eps raw): abs={np.linalg.norm(dv_raw):.6e}, rel={v_rel_raw:.6e}")

    ifreq_list = _parse_freq_list(args.ifreqs, int(eps.nfreq))
    print(f"  frequencies checked: {ifreq_list}")
    print("  eps orientation: transposed (BGW-matching)")

    ifreq0 = int(args.ifreq0)
    ifreqp = int(args.ifreqp)
    if ifreq0 < 0 or ifreq0 >= int(eps.nfreq):
        raise ValueError(f"ifreq0 out of range: {ifreq0}")
    if ifreqp < 0 or ifreqp >= int(eps.nfreq):
        raise ValueError(f"ifreqp out of range: {ifreqp}")

    epsinv0_raw = np.asarray(
        eps.get_eps_matrix(iq=iq, ifreq=ifreq0, imatrix=int(args.imatrix)),
        dtype=np.complex128,
    )[:nmtx, :nmtx]
    epsinvp_raw = np.asarray(
        eps.get_eps_matrix(iq=iq, ifreq=ifreqp, imatrix=int(args.imatrix)),
        dtype=np.complex128,
    )[:nmtx, :nmtx]
    epsinv0_full = _to_bgw_eps_orientation(epsinv0_raw)
    epsinvp_full = _to_bgw_eps_orientation(epsinvp_raw)
    epsinv0_ppm = (
        _head_patch_epsinv_column(epsinv0_full, g0_idx, v[g0_idx], w_head)
        if bool(args.patch_head_column)
        else epsinv0_full
    )
    epsinvp_ppm = (
        _head_patch_epsinv_column(epsinvp_full, g0_idx, v[g0_idx], w_head)
        if bool(args.patch_head_column)
        else epsinvp_full
    )
    freq_pair_p = np.asarray(eps.freqs[ifreqp], dtype=np.float64)
    omega_p_from_file = float(freq_pair_p[1] / 13.6056980659)
    omega_p_ry = float(args.omega_p_ry) if args.omega_p_ry is not None else omega_p_from_file
    omega_fit, b_fit, valid_fit, invalid_pct = _fit_gn_ppm_on_epsinv(
        epsinv0_ppm,
        epsinvp_ppm,
        omega_p_ry=omega_p_ry,
        fallback_omega_ry=float(args.fallback_omega_ry),
    )
    b_valid = np.where(valid_fit, b_fit, 0.0 + 0.0j)
    b_invalid = np.where(~valid_fit, b_fit, 0.0 + 0.0j)
    b_norm = _fro_norm(b_fit)
    b_valid_norm = _fro_norm(b_valid)
    b_invalid_norm = _fro_norm(b_invalid)
    print("")
    print(
        "  GN-PPM fit on (I - epsinv) [BGW Sigma convention]: "
        f"ifreq0={ifreq0}, ifreqp={ifreqp}, omega_p={omega_p_ry:.6f} Ry, "
        f"invalid={invalid_pct:.2f}%"
    )
    print(
        "    Omega stats (valid): "
        f"min={float(np.min(omega_fit[valid_fit])):.6e} Ry, "
        f"max={float(np.max(omega_fit[valid_fit])):.6e} Ry"
        if np.any(valid_fit)
        else "    Omega stats (valid): none"
    )
    print(
        "    ||B||_F totals: "
        f"all={b_norm:.6e}, valid={b_valid_norm:.6e}, invalid={b_invalid_norm:.6e}"
    )
    if b_norm > 0.0:
        print(
            "    ||B||_F fractions: "
            f"valid={b_valid_norm / b_norm:.6f}, invalid={b_invalid_norm / b_norm:.6f}"
        )
    print("  invalid-mode policy (main GN columns): skip invalid nodes (invalid_gpp_mode=0 style)")

    for ifreq in ifreq_list:
        freq_pair = np.asarray(eps.freqs[ifreq], dtype=np.float64)
        omega_re_ryd = float(freq_pair[0] / 13.6056980659)
        omega_im_ryd = float(freq_pair[1] / 13.6056980659)

        epsinv = np.asarray(
            eps.get_eps_matrix(iq=iq, ifreq=ifreq, imatrix=int(args.imatrix)),
            dtype=np.complex128,
        )
        if epsinv.shape[0] != int(eps.nmtx[iq]):
            raise ValueError("Unexpected eps matrix shape")
        epsinv = _to_bgw_eps_orientation(epsinv[:nmtx, :nmtx])

        epsinv_use = (
            _head_patch_epsinv_column(epsinv, g0_idx, v[g0_idx], w_head)
            if bool(args.patch_head_column)
            else epsinv
        )

        w_left = v[:, None] * epsinv_use
        herm = _rel_hermiticity_defect(w_left)
        norm = _fro_norm(w_left)
        head = w_left[g0_idx, g0_idx]
        head_err = abs(head - w_head)
        wing_row = np.linalg.norm(np.delete(w_left[g0_idx, :], g0_idx))
        wing_col = np.linalg.norm(np.delete(w_left[:, g0_idx], g0_idx))

        print("")
        print(
            f"  ifreq={ifreq}  omega=(re={omega_re_ryd:.6e}, im={omega_im_ryd:.6e}) Ry  "
            f"patch_head_column={bool(args.patch_head_column)}"
        )
        print(f"    epsinv herm_rel={_rel_hermiticity_defect(epsinv_use):.6e}")
        print(
            f"    W_left |W|_F={norm:.6e}  "
            f"herm_rel={herm:.6e}  "
            f"W00={head.real:.6e}{head.imag:+.2e}i  "
            f"|W00-Whead|={head_err:.3e}  "
            f"||row0_off||={wing_row:.3e}  ||col0_off||={wing_col:.3e}"
        )

    if int(args.n_sum_states) > 0:
        print("")
        print("  Computing diagonal Sigma terms from BGW-style M(G) tensors...")
        eye = np.eye(nmtx, dtype=np.complex128)
        i_eps0_full = eye - epsinv0_ppm
        i_eps0_valid = np.where(valid_fit, i_eps0_full, 0.0 + 0.0j)
        i_eps0_invalid = np.where(~valid_fit, i_eps0_full, 0.0 + 0.0j)
        gvecs_use = np.asarray(gvecs[:nmtx], dtype=np.int64)
        # Main GN output: skip invalid nodes by zeroing their residue tensor.
        band_ids, e_solve, x_bare, sx_dyn, ch_dyn, sx_static, ch_static = _compute_sigma_terms_gspace(
            wfn=wfn,
            epsinv0=epsinv0_ppm,
            epsinvi=epsinvp_ppm,
            v_vec=v,
            gvecs_eps=gvecs_use,
            omega_fit=omega_fit,
            i_eps0=i_eps0_valid,
            band_offset=int(args.band_offset),
            n_valence=int(args.n_valence),
            n_cond_solve=int(args.n_cond_solve),
            n_sum_states=int(args.n_sum_states),
            eta_ry=float(args.eta_ev) / RYD2EV,
        )
        sxm_x_dyn = sx_dyn - x_bare
        sxm_x_static = sx_static - x_bare

        if bool(args.debug_invalid_mode):
            # invalid_gpp_mode=3 style correction (static limit on invalid modes only)
            epsinv0_invalid_static = eye - i_eps0_invalid
            _, _, x_bare_s, _, _, sx_static_inv, ch_static_inv = _compute_sigma_terms_gspace(
                wfn=wfn,
                epsinv0=epsinv0_invalid_static,
                epsinvi=epsinvp_ppm,
                v_vec=v,
                gvecs_eps=gvecs_use,
                omega_fit=np.asarray(omega_fit, dtype=np.float64),
                i_eps0=np.zeros_like(i_eps0_invalid),
                band_offset=int(args.band_offset),
                n_valence=int(args.n_valence),
                n_cond_solve=int(args.n_cond_solve),
                n_sum_states=int(args.n_sum_states),
                eta_ry=float(args.eta_ev) / RYD2EV,
            )
            sxm_x_inv_static = sx_static_inv - x_bare_s

            # invalid_gpp_mode=2 style correction (Omega=2 Ry on invalid modes only)
            omega_invalid_2ry = np.where(
                valid_fit,
                np.asarray(omega_fit, dtype=np.float64),
                np.asarray(float(args.fallback_omega_ry), dtype=np.float64),
            )
            _, _, x_bare_2, sx_dyn_inv2, ch_dyn_inv2, _, _ = _compute_sigma_terms_gspace(
                wfn=wfn,
                epsinv0=epsinv0_ppm,
                epsinvi=epsinvp_ppm,
                v_vec=v,
                gvecs_eps=gvecs_use,
                omega_fit=omega_invalid_2ry,
                i_eps0=i_eps0_invalid,
                band_offset=int(args.band_offset),
                n_valence=int(args.n_valence),
                n_cond_solve=int(args.n_cond_solve),
                n_sum_states=int(args.n_sum_states),
                eta_ry=float(args.eta_ev) / RYD2EV,
            )
            sxm_x_inv_2ry = sx_dyn_inv2 - x_bare_2

            print("")
            print("  Sigma diagonal table (eV)  [BGW-style decomposition + invalid-mode debug]")
            print(
                "  n\tE_dft\tX\tSEX-X_GN(E_n)\tCOH_GN(E_n)\tSEX-X_static\tCOH_static\t"
                "d(SEX-X)_inv_static\td(COH)_inv_static\td(SEX-X)_inv_2Ry\td(COH)_inv_2Ry"
            )
            for i in range(len(band_ids)):
                print(
                    f"  {int(band_ids[i]):d}\t{float(e_solve[i] * RYD2EV):.6f}\t"
                    f"{float(np.real(x_bare[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(sxm_x_dyn[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(ch_dyn[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(sxm_x_static[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(ch_static[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(sxm_x_inv_static[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(ch_static_inv[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(sxm_x_inv_2ry[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(ch_dyn_inv2[i]) * RYD2EV):.6f}"
                )
        else:
            print("")
            print("  Sigma diagonal table (eV)  [BGW-style decomposition]")
            print("  n\tE_dft\tX\tSEX-X_GN(E_n)\tCOH_GN(E_n)\tSEX-X_static\tCOH_static")
            for i in range(len(band_ids)):
                print(
                    f"  {int(band_ids[i]):d}\t{float(e_solve[i] * RYD2EV):.6f}\t"
                    f"{float(np.real(x_bare[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(sxm_x_dyn[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(ch_dyn[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(sxm_x_static[i]) * RYD2EV):.6f}\t"
                    f"{float(np.real(ch_static[i]) * RYD2EV):.6f}"
                )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check 0D q=0 W(G,G',omega) from eps0mat and report Hermiticity diagnostics."
    )
    parser.add_argument("-i", "--input", required=True, help="cohsex/gw input file")
    parser.add_argument("--eps0mat", required=True, help="Path to eps0mat.h5")
    parser.add_argument("--iq", type=int, default=0, help="q-index in eps file (default: 0)")
    parser.add_argument("--imatrix", type=int, default=0, help="matrix index in eps file (default: 0)")
    parser.add_argument(
        "--ifreqs",
        default="all",
        help="Frequency indices to check: 'all' or comma-separated list (default: all)",
    )
    parser.add_argument(
        "--nmtx-max",
        type=int,
        default=0,
        help="Optional truncation of G-space dimension for faster debug (0 = full)",
    )
    parser.add_argument(
        "--patch-head-column",
        type=int,
        default=1,
        help="Apply epsinv[:,G0]=0 and epsinv[G0,G0]=Whead/Vhead (default: 1)",
    )
    parser.add_argument("--ifreq0", type=int, default=0, help="Static frequency index for GN fit (default: 0)")
    parser.add_argument("--ifreqp", type=int, default=1, help="Imag-axis frequency index for GN fit (default: 1)")
    parser.add_argument(
        "--omega-p-ry",
        type=float,
        default=None,
        help="Override GN omega_p in Ry (default: derive from eps frequency ifreqp)",
    )
    parser.add_argument(
        "--fallback-omega-ry",
        type=float,
        default=2.0,
        help="Fallback Omega (Ry) for invalid GN nodes (default: 2.0)",
    )
    parser.add_argument("--v-head-au", type=float, default=None, help="Override V head in a.u.")
    parser.add_argument("--w-head-au", type=float, default=None, help="Override W head in a.u.")
    parser.add_argument("--band-offset", type=int, default=0, help="Starting band index (default: 0)")
    parser.add_argument("--n-valence", type=int, default=0, help="Valence states in solve window (0 disables sigma table)")
    parser.add_argument("--n-cond-solve", type=int, default=0, help="Conduction states in solve window")
    parser.add_argument("--n-sum-states", type=int, default=0, help="Total states in explicit sums")
    parser.add_argument("--eta-ev", type=float, default=0.0, help="Small broadening in eV for dynamic denominators")
    parser.add_argument(
        "--debug-invalid-mode",
        type=int,
        default=0,
        help=(
            "Print extra invalid-mode correction columns: "
            "static-fallback and 2Ry-fallback contributions (default: 0)"
        ),
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
