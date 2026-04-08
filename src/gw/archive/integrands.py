from __future__ import annotations

import jax
import jax.numpy as jnp

from . import w_isdf_dynamic as legacy_dynamic


def _k_to_R(g_k: jax.Array, *, nkx: int, nky: int, nkz: int, flip_sign: bool) -> jax.Array:
    g_fft = g_k.reshape(nkx, nky, nkz, *g_k.shape[1:]).transpose(3, 4, 5, 6, 0, 1, 2)
    if flip_sign:
        return jnp.fft.fftn(g_fft, axes=(-3, -2, -1), norm="ortho")
    return jnp.fft.ifftn(g_fft, axes=(-3, -2, -1), norm="ortho")


def _window_views(
    psi_vTX: jax.Array,
    psi_vY: jax.Array,
    psi_cX: jax.Array,
    psi_cTY: jax.Array,
    enk_v: jax.Array,
    enk_c: jax.Array,
    *,
    val_start: jax.Array,
    cond_start: jax.Array,
    max_val_len: int,
    max_cond_len: int,
):
    psi_vTX_win = legacy_dynamic._slice_along_axis(psi_vTX, val_start, max_val_len, axis=3)
    psi_vY_win = legacy_dynamic._slice_along_axis(psi_vY, val_start, max_val_len, axis=1)
    psi_cX_win = legacy_dynamic._slice_along_axis(psi_cX, cond_start, max_cond_len, axis=1)
    psi_cTY_win = legacy_dynamic._slice_along_axis(psi_cTY, cond_start, max_cond_len, axis=3)
    enk_v_win = legacy_dynamic._slice_along_axis(enk_v, val_start, max_val_len, axis=1)
    enk_c_win = legacy_dynamic._slice_along_axis(enk_c, cond_start, max_cond_len, axis=1)
    return psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win, enk_v_win, enk_c_win


def _prep_masks(
    psi_vTX_win: jax.Array,
    psi_vY_win: jax.Array,
    psi_cX_win: jax.Array,
    psi_cTY_win: jax.Array,
    *,
    val_mask: jax.Array,
    cond_mask: jax.Array,
):
    val_mask_f = val_mask.astype(psi_vTX_win.dtype)
    cond_mask_f = cond_mask.astype(psi_cX_win.dtype)
    psi_vTX_win = psi_vTX_win * val_mask_f[:, None, None, :]
    psi_vY_win = psi_vY_win * val_mask_f[:, :, None, None]
    psi_cX_win = psi_cX_win * cond_mask_f[:, :, None, None]
    psi_cTY_win = psi_cTY_win * cond_mask_f[:, None, None, :]
    return psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win


def _chi_R_tile_standard(
    psi_vTX_win: jax.Array,
    psi_vY_win: jax.Array,
    psi_cX_win: jax.Array,
    psi_cTY_win: jax.Array,
    enk_v_win: jax.Array,
    enk_c_win: jax.Array,
    *,
    vmax: float,
    cmin: float,
    tau: jax.Array,
    gamma: float,
    val_mask: jax.Array,
    cond_mask: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    val_mask_complex = val_mask.astype(jnp.complex128)
    cond_mask_complex = cond_mask.astype(jnp.complex128)
    exp_v = jnp.exp(-gamma * tau * (vmax - enk_v_win)) * val_mask_complex
    exp_c = jnp.exp(-gamma * tau * (enk_c_win - cmin)) * cond_mask_complex
    Gv_k = jnp.einsum("ksxm,km,kmty->ksxty", jnp.conj(psi_vTX_win), exp_v, psi_vY_win, optimize=True)
    Gc_k = jnp.einsum("ksxm,km,kmty->ksxty", jnp.conj(psi_cTY_win), exp_c, psi_cX_win, optimize=True)
    Gv_R = _k_to_R(Gv_k, nkx=nkx, nky=nky, nkz=nkz, flip_sign=False)
    Gc_R = _k_to_R(Gc_k, nkx=nkx, nky=nky, nkz=nkz, flip_sign=True)
    return jnp.einsum("ambnxyz,bnamxyz->mnxyz", Gc_R, Gv_R, optimize=True)


def _chi_R_tile_reverse(
    psi_vTX_win: jax.Array,
    psi_vY_win: jax.Array,
    psi_cX_win: jax.Array,
    psi_cTY_win: jax.Array,
    enk_v_win: jax.Array,
    enk_c_win: jax.Array,
    *,
    vmin: float,
    cmax: float,
    tau: jax.Array,
    gamma: float,
    val_mask: jax.Array,
    cond_mask: jax.Array,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    val_mask_complex = val_mask.astype(jnp.complex128)
    cond_mask_complex = cond_mask.astype(jnp.complex128)
    exp_v = jnp.exp(-gamma * tau * (enk_v_win - vmin)) * val_mask_complex
    exp_c = jnp.exp(-gamma * tau * (cmax - enk_c_win)) * cond_mask_complex
    Gv_k = jnp.einsum("ksxm,km,kmty->ksxty", jnp.conj(psi_vTX_win), exp_v, psi_vY_win, optimize=True)
    Gc_k = jnp.einsum("ksxm,km,kmty->ksxty", jnp.conj(psi_cTY_win), exp_c, psi_cX_win, optimize=True)
    Gv_R = _k_to_R(Gv_k, nkx=nkx, nky=nky, nkz=nkz, flip_sign=False)
    Gc_R = _k_to_R(Gc_k, nkx=nkx, nky=nky, nkz=nkz, flip_sign=True)
    return jnp.einsum("ambnxyz,bnamxyz->mnxyz", Gc_R, Gv_R, optimize=True)


def compute_gl_standard_batch(
    *,
    psi_vTX: jax.Array,
    psi_vY: jax.Array,
    psi_cX: jax.Array,
    psi_cTY: jax.Array,
    enk_v: jax.Array,
    enk_c: jax.Array,
    val_start: jax.Array,
    cond_start: jax.Array,
    val_mask: jax.Array,
    cond_mask: jax.Array,
    max_val_len: int,
    max_cond_len: int,
    nkx: int,
    nky: int,
    nkz: int,
    vmin: float,
    vmax: float,
    cmin: float,
    cmax: float,
    tau_i: jax.Array,
    w_i: jax.Array,
    gamma: float,
    omega_values: jax.Array,
    denom_name: str,
    sign: int,
) -> jax.Array:
    """Batch GL accumulation for antiresonant and resonant-below branches."""
    del vmin, cmax
    n_omega = int(omega_values.shape[0])
    if n_omega == 0 or max_val_len == 0 or max_cond_len == 0 or int(tau_i.shape[0]) == 0:
        nrmu = psi_vTX.shape[2]
        return jnp.zeros((0, nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)

    views = _window_views(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        val_start=val_start, cond_start=cond_start,
        max_val_len=max_val_len, max_cond_len=max_cond_len,
    )
    psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win, enk_v_win, enk_c_win = views
    psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win = _prep_masks(
        psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win,
        val_mask=val_mask, cond_mask=cond_mask,
    )

    E_gap = float(cmin - vmax)
    omega_vec = jnp.asarray(omega_values, dtype=jnp.float64)
    tau_vec = jnp.asarray(tau_i, dtype=jnp.float64)
    w_vec = jnp.asarray(w_i, dtype=jnp.float64)

    def body(acc, packed):
        tau_u, w_u = packed
        chi_tau_R = _chi_R_tile_standard(
            psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win, enk_v_win, enk_c_win,
            vmax=vmax, cmin=cmin, tau=tau_u, gamma=gamma,
            val_mask=val_mask, cond_mask=cond_mask,
            nkx=nkx, nky=nky, nkz=nkz,
        )
        if denom_name == "antiresonant":
            delta = E_gap + omega_vec
        elif denom_name == "resonant_below":
            delta = E_gap - omega_vec
        else:
            raise ValueError(f"Unexpected GL standard branch {denom_name!r}")
        coeff = float(sign) * gamma * w_u * jnp.exp(-(gamma * delta - 1.0) * tau_u)
        return acc + coeff[:, None, None, None, None, None] * chi_tau_R[None, ...], None

    nrmu = psi_vTX.shape[2]
    chi_R = jnp.zeros((n_omega, nrmu, nrmu, nkx, nky, nkz), dtype=jnp.complex128)
    chi_R, _ = jax.lax.scan(body, chi_R, (tau_vec, w_vec))
    chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm="ortho")
    chi_q = chi_q.transpose(0, 3, 4, 5, 1, 2)[:, :, :, :, None, :, None, :]
    return chi_q


def compute_gl_reverse_batch(
    *,
    psi_vTX: jax.Array,
    psi_vY: jax.Array,
    psi_cX: jax.Array,
    psi_cTY: jax.Array,
    enk_v: jax.Array,
    enk_c: jax.Array,
    val_start: jax.Array,
    cond_start: jax.Array,
    val_mask: jax.Array,
    cond_mask: jax.Array,
    max_val_len: int,
    max_cond_len: int,
    nkx: int,
    nky: int,
    nkz: int,
    vmin: float,
    vmax: float,
    cmin: float,
    cmax: float,
    tau_i: jax.Array,
    w_i: jax.Array,
    gamma: float,
    omega_values: jax.Array,
) -> jax.Array:
    """Batch GL accumulation for resonant-above branch."""
    del vmax, cmin
    n_omega = int(omega_values.shape[0])
    if n_omega == 0 or max_val_len == 0 or max_cond_len == 0 or int(tau_i.shape[0]) == 0:
        nrmu = psi_vTX.shape[2]
        return jnp.zeros((0, nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)

    views = _window_views(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        val_start=val_start, cond_start=cond_start,
        max_val_len=max_val_len, max_cond_len=max_cond_len,
    )
    psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win, enk_v_win, enk_c_win = views
    psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win = _prep_masks(
        psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win,
        val_mask=val_mask, cond_mask=cond_mask,
    )

    E_bw = float(cmax - vmin)
    omega_vec = jnp.asarray(omega_values, dtype=jnp.float64)
    tau_vec = jnp.asarray(tau_i, dtype=jnp.float64)
    w_vec = jnp.asarray(w_i, dtype=jnp.float64)

    def body(acc, packed):
        tau_u, w_u = packed
        chi_tau_R = _chi_R_tile_reverse(
            psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win, enk_v_win, enk_c_win,
            vmin=vmin, cmax=cmax, tau=tau_u, gamma=gamma,
            val_mask=val_mask, cond_mask=cond_mask,
            nkx=nkx, nky=nky, nkz=nkz,
        )
        delta = omega_vec - E_bw
        coeff = gamma * w_u * jnp.exp(-(gamma * delta - 1.0) * tau_u)
        return acc + coeff[:, None, None, None, None, None] * chi_tau_R[None, ...], None

    nrmu = psi_vTX.shape[2]
    chi_R = jnp.zeros((n_omega, nrmu, nrmu, nkx, nky, nkz), dtype=jnp.complex128)
    chi_R, _ = jax.lax.scan(body, chi_R, (tau_vec, w_vec))
    chi_q = jnp.fft.fftn(chi_R, axes=(-3, -2, -1), norm="ortho")
    chi_q = chi_q.transpose(0, 3, 4, 5, 1, 2)[:, :, :, :, None, :, None, :]
    return chi_q


def compute_hgl_batch(
    *,
    psi_vTX: jax.Array,
    psi_vY: jax.Array,
    psi_cX: jax.Array,
    psi_cTY: jax.Array,
    enk_v: jax.Array,
    enk_c: jax.Array,
    val_start: jax.Array,
    cond_start: jax.Array,
    val_mask: jax.Array,
    cond_mask: jax.Array,
    max_val_len: int,
    max_cond_len: int,
    nkx: int,
    nky: int,
    nkz: int,
    tau_hgl: jax.Array,
    w_hgl: jax.Array,
    gamma: float,
    omega_values: jax.Array,
) -> jax.Array:
    """Batch HGL accumulation for crossing branch using one tau-scan."""
    n_omega = int(omega_values.shape[0])
    if n_omega == 0 or max_val_len == 0 or max_cond_len == 0 or int(tau_hgl.shape[0]) == 0:
        nrmu = psi_vTX.shape[2]
        return jnp.zeros((0, nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)

    views = _window_views(
        psi_vTX, psi_vY, psi_cX, psi_cTY, enk_v, enk_c,
        val_start=val_start, cond_start=cond_start,
        max_val_len=max_val_len, max_cond_len=max_cond_len,
    )
    psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win, enk_v_win, enk_c_win = views
    psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win = _prep_masks(
        psi_vTX_win, psi_vY_win, psi_cX_win, psi_cTY_win,
        val_mask=val_mask, cond_mask=cond_mask,
    )
    val_mask_complex = val_mask.astype(jnp.complex128)
    cond_mask_complex = cond_mask.astype(jnp.complex128)

    omega_vec = jnp.asarray(omega_values, dtype=jnp.float64)
    tau_vec = jnp.asarray(tau_hgl, dtype=jnp.float64)
    w_vec = jnp.asarray(w_hgl, dtype=jnp.float64)

    nrmu = psi_vTX.shape[2]
    chi_q0 = jnp.zeros((n_omega, nkx, nky, nkz, 1, nrmu, 1, nrmu), dtype=jnp.complex128)

    def body(acc, packed):
        tau_u, w_u = packed
        phase_v = jnp.exp(1j * gamma * tau_u * enk_v_win) * val_mask_complex
        phase_c = jnp.exp(1j * gamma * tau_u * enk_c_win) * cond_mask_complex
        Gv_k = jnp.einsum("ksxm,km,kmty->ksxty", jnp.conj(psi_vTX_win), phase_v, psi_vY_win, optimize=True)
        Gc_k = jnp.einsum("ksxm,km,kmty->ksxty", jnp.conj(psi_cTY_win), phase_c, psi_cX_win, optimize=True)
        Gv_R = _k_to_R(Gv_k, nkx=nkx, nky=nky, nkz=nkz, flip_sign=False)
        Gc_mR = _k_to_R(Gc_k, nkx=nkx, nky=nky, nkz=nkz, flip_sign=True)
        product_R = jnp.einsum("ambnxyz,bnamxyz->mnxyz", jnp.conj(Gc_mR), Gv_R, optimize=True)
        P_plus_q = jnp.fft.fftn(product_R.real, axes=(-3, -2, -1), norm="ortho")
        P_cross_q = jnp.fft.fftn(-product_R.imag, axes=(-3, -2, -1), norm="ortho")
        P_plus_q = P_plus_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
        P_cross_q = P_cross_q.transpose(2, 3, 4, 0, 1)[:, :, :, None, :, None, :]
        phase = gamma * tau_u * omega_vec
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)
        contrib = -gamma * w_u * (
            cos_phase[:, None, None, None, None, None, None, None] * P_cross_q[None, ...]
            - sin_phase[:, None, None, None, None, None, None, None] * P_plus_q[None, ...]
        )
        return acc + contrib, None

    chi_q, _ = jax.lax.scan(body, chi_q0, (tau_vec, w_vec))
    return chi_q
