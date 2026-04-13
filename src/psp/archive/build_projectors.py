from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, List
from scipy.special import spherical_jn
try:
    from scipy.special import sph_harm as _sph_harm_complex
    _HAS_SPH_HARM = True
except Exception:
    _HAS_SPH_HARM = False
from scipy.special import sph_harm_y as _sph_harm_y
from scipy.interpolate import InterpolatedUnivariateSpline


def _sphY(m: int, l: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if _HAS_SPH_HARM:
        return _sph_harm_complex(m, l, phi, theta)
    return _sph_harm_y(m, l, phi, theta)


def compute_Ylm_list(K_cart: np.ndarray, lmax: int = 2) -> List[np.ndarray]:
    """Return list over l=0..lmax of Y_lm(K_hat) arrays with shape (2l+1, nG).
    K_cart: (nG,3) Cartesian K vectors; lmax: 2 (or 3) supported.
    """
    Kx, Ky, Kz = K_cart[:, 0], K_cart[:, 1], K_cart[:, 2]
    K_norm = np.sqrt(np.maximum(0.0, Kx * Kx + Ky * Ky + Kz * Kz))
    with np.errstate(invalid='ignore', divide='ignore'):
        theta = np.where(K_norm > 0, np.arccos(np.clip(Kz / np.where(K_norm == 0, 1.0, K_norm), -1.0, 1.0)), 0.0)
        phi = np.arctan2(Ky, Kx)
    result: List[np.ndarray] = []
    for l in range(0, lmax + 1):
        vals = np.empty((2 * l + 1, K_cart.shape[0]), dtype=np.complex128)
        mi = 0
        for m in range(-l, l + 1):
            vals[mi, :] = _sphY(m, l, phi, theta)
            mi += 1
        result.append(vals)
    return result


def compute_spinor_omegas(Ylm_list: List[np.ndarray]) -> List[Dict[float, np.ndarray]]:
    """Return Ω_{ℓ j m_j}(K̂) for j=ℓ±½ as a dict keyed by j.
    Each entry Ω has shape (2, 2j+1, nG) for spinor components (↑,↓).
    """
    result: List[Dict[float, np.ndarray]] = []
    for l, Ylm in enumerate(Ylm_list):
        nG = Ylm.shape[1]
        denom = float(2 * l + 1)

        def getY(m: int) -> np.ndarray:
            if m < -l or m > l:
                return np.zeros((nG,), dtype=np.complex128)
            return Ylm[m + l, :]

        entry: Dict[float, np.ndarray] = {}

        # j = l + 1/2
        j_plus = l + 0.5
        twoj_plus = 2 * l + 1
        Omega_plus = np.zeros((2, twoj_plus + 1, nG), dtype=np.complex128)
        for idx in range(twoj_plus + 1):
            two_mj = -twoj_plus + 2 * idx
            m = (two_mj - 1) // 2
            a = np.sqrt(max(0.0, l + m + 1.0) / denom)
            b = np.sqrt(max(0.0, l - m) / denom)
            Omega_plus[0, idx, :] = a * getY(m)
            Omega_plus[1, idx, :] = b * getY(m + 1)
        entry[j_plus] = Omega_plus

        # j = l - 1/2
        if l >= 1:
            j_minus = l - 0.5
            twoj_minus = 2 * l - 1
            Omega_minus = np.zeros((2, twoj_minus + 1, nG), dtype=np.complex128)
            for idx in range(twoj_minus + 1):
                two_mj = -twoj_minus + 2 * idx
                m = (two_mj + 1) // 2
                c = np.sqrt(max(0.0, l - m + 1.0) / denom)
                d = np.sqrt(max(0.0, l + m) / denom)
                Omega_minus[0, idx, :] = c * getY(m - 1)
                Omega_minus[1, idx, :] = -d * getY(m)
            entry[j_minus] = Omega_minus

        result.append(entry)
    return result


def spherical_hankel_transform_l_np(
    l: int,
    r: np.ndarray,
    beta_r: np.ndarray,
    q: np.ndarray,
    rab: np.ndarray | None = None,
) -> np.ndarray:
    """Compute F_l(q) = ∫ r^2 beta_l(r) j_l(q r) dr (NumPy/SciPy)."""
    r = np.asarray(r, dtype=float)
    beta_r = np.asarray(beta_r, dtype=float)
    q = np.asarray(q, dtype=float)
    qr = np.multiply.outer(q, r)
    j_l = spherical_jn(l, qr)
    integrand = (r**2)[None, :] * beta_r[None, :] * j_l
    if rab is not None:
        w = np.asarray(rab, dtype=float)[None, :]
        return np.sum(integrand * w, axis=1)
    dr = r[1] - r[0]
    weights = np.ones_like(r)
    weights[0] = 0.5; weights[-1] = 0.5
    return np.sum(integrand * weights[None, :], axis=1) * dr


def compute_projector_form_factors_on_K(
    pseudo,
    K_norm: np.ndarray,
    cell_volume: float,
    q_points: int | None = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """For each PP_BETA projector, compute F_l(q) and spline to K_norm.
    Returns dict keyed by (l, beta_index)->F_l(K_norm) (nG,).
    """
    ppmesh = getattr(pseudo, 'pp_mesh', None)
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    assert ppmesh is not None and ppnl is not None, "Pseudopotential missing mesh or nonlocal section"
    r = np.asarray(ppmesh.pp_r.value, dtype=float)
    try:
        rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
    except Exception:
        rab = None
    betas = list(getattr(ppnl, 'pp_beta', []))
    qmax = float(np.max(K_norm)) if K_norm.size > 0 else 0.0
    if q_points is None:
        q_points = max(1024, 2 * len(r))
    q_grid = np.linspace(0.0, max(qmax, 1e-8), q_points)
    result: Dict[Tuple[int, int], np.ndarray] = {}
    for idx, beta in enumerate(betas, start=1):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        raw_beta = np.asarray(beta.value, dtype=float)
        beta_r = np.zeros_like(r)
        if r[0] == 0.0:
            beta_r[1:] = raw_beta[1:] / r[1:]
            beta_r[0] = beta_r[1]
        else:
            beta_r = raw_beta / r
        F_q = spherical_hankel_transform_l_np(l, r, beta_r, q_grid, rab)
        spl = InterpolatedUnivariateSpline(q_grid, F_q, k=3, ext=1)
        F_on_K = spl(K_norm)
        result[(l, idx)] = F_on_K
    return result


def expand_D_block(D_mat: np.ndarray,
                   beta_idx_arr: np.ndarray,
                   ell_idx_arr: np.ndarray,
                   j_rows: np.ndarray,
                   mj_slot_rows: np.ndarray) -> np.ndarray:
    """Expand species D_mat onto row space grouped by (ℓ,j,mj_slot).
    Returns (R,R) complex matrix.
    """
    R = beta_idx_arr.shape[0]
    D_exp = np.zeros((R, R), dtype=np.complex128)
    keys = {}
    for r, (lr, jr, mr) in enumerate(zip(ell_idx_arr, j_rows, mj_slot_rows)):
        keys.setdefault((int(lr), float(jr), int(mr)), []).append(r)
    for (_l, _j, _m), idxs in keys.items():
        idxs = np.asarray(idxs, dtype=np.int32)
        bi = beta_idx_arr[idxs]
        D_block = D_mat[np.ix_(bi, bi)]
        D_exp[np.ix_(idxs, idxs)] = D_block
    return D_exp


def build_species_Z_rows(pseudo,
                         K_norm_np: np.ndarray,
                         Omega_dicts: List[Dict[float, np.ndarray]],
                         cell_volume: float,
                         use_spinor: bool = True,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build species-level Z rows and mapping arrays.
    Returns:
      Z_base: (R, 2, nG)
      beta_idx_arr: (R,)
      ell_idx_arr: (R,)
      j_rows: (R,)
      mj_slot_rows: (R,)
      D_mat: (nproj, nproj)
    """
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    betas = list(getattr(ppnl, 'pp_beta', [])) if ppnl is not None else []
    if not betas:
        return (np.zeros((0,2,0), dtype=np.complex128),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,0), dtype=np.complex128))

    Fdict = compute_projector_form_factors_on_K(pseudo, K_norm_np, float(cell_volume))
    try:
        D_mat = np.asarray(ppnl.pp_dij.value, dtype=float)
        nproj = int(getattr(pseudo.pp_header, 'number_of_proj', len(betas)))
        D_mat = D_mat.reshape(nproj, nproj)
    except Exception:
        D_mat = None
    if D_mat is None:
        raise RuntimeError("Pseudopotential missing PP_DIJ matrix for nonlocal term")

    Z_rows: List[np.ndarray] = []
    beta_index_rows: List[int] = []
    ell_rows: List[int] = []
    j_rows: List[float] = []
    mj_slot_rows: List[int] = []

    for i, beta in enumerate(betas, start=1):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        F_on_K = Fdict.get((l, i))
        if F_on_K is None:
            continue
        j_val = float(getattr(beta, 'jjj', l + 0.5))
        Omega = Omega_dicts[l].get(j_val) if use_spinor else None
        if Omega is None:
            raise RuntimeError("Scalar-relativistic path not supported in FR projector builder")
        for idx in range(Omega.shape[1]):
            row = np.stack([
                (1j)**l * F_on_K * Omega[0, idx, :],
                (1j)**l * F_on_K * Omega[1, idx, :],
            ], axis=0)
            Z_rows.append(row)
            beta_index_rows.append(i - 1)
            ell_rows.append(l)
            j_rows.append(j_val)
            mj_slot_rows.append(idx)

    Z_base = np.asarray(Z_rows, dtype=np.complex128)
    return (Z_base,
            np.asarray(beta_index_rows, dtype=np.int32),
            np.asarray(ell_rows, dtype=np.int32),
            np.asarray(j_rows, dtype=float),
            np.asarray(mj_slot_rows, dtype=np.int32),
            np.asarray(D_mat, dtype=np.complex128))
