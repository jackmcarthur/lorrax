from __future__ import annotations

"""
Utility to construct W_{mu,nu}(q=0) from a BerkeleyGW eps^{-1} matrix:

    W_{GG'} = eps^{-1}_{GG'} · V_G'   with body-only (G=0 set to 0 and wings zeroed)
    W_{mu,nu} = sum_{GG'} zeta^*_{mu}(G) W_{GG'} zeta_{nu}(G')

This mirrors the mapping and contraction used in the legacy cohsex_isdf path
but forces the head and wings at q=0 to zero.
"""

import numpy as np

from isdf_io import EPSReader


def _fft_integer_axis(n: int) -> np.ndarray:
	if n <= 0:
		raise ValueError(f"FFT dimension must be positive, got {n}")
	if n % 2 == 0:
		half = n // 2
		positives = np.arange(0, half + 1, dtype=np.int32)
		negatives = np.arange(-half + 1, 0, dtype=np.int32)
	else:
		half = n // 2
		positives = np.arange(0, half + 1, dtype=np.int32)
		negatives = np.arange(-half, 0, dtype=np.int32)
	return np.concatenate([positives, negatives], axis=0)


def compute_Wmunu_from_eps0_body(
    wfn,
    sym,
    meta,
    zeta_q_r: np.ndarray,
    qvec_wrapped: np.ndarray,
    eps0_path: str,
) -> np.ndarray:
    """Compute body-only W_{mu,nu}(q=0) from eps^{-1}_q=0 and dense FFT data.

    Args:
        zeta_q_r: (n_mu, n_rtot) real-space zeta for this q (q=0)
        qvec_wrapped: fractional q-vector (length-3) (used for phase; here should be 0)
        eps0_path: path to eps0mat.h5 (assumed eps^{-1} at q=0)
    Returns:
        W_{mu,nu} as a complex128 NumPy array (n_mu, n_mu)
    """
    # 1) Load eps^{-1}(q=0)
    eps = EPSReader(eps0_path)
    iq_eps = 0  # q=0 only
    epsinv = eps.get_eps_matrix(iq_eps)  # (nG_eps, nG_eps)

    # 2) Build mapping from eps ordering -> dense FFT cube indices
    nx, ny, nz = map(int, meta.fft_grid)
    gx_vals = _fft_integer_axis(nx)
    gy_vals = _fft_integer_axis(ny)
    gz_vals = _fft_integer_axis(nz)
    gx_lookup = {int(v): idx for idx, v in enumerate(gx_vals)}
    gy_lookup = {int(v): idx for idx, v in enumerate(gy_vals)}
    gz_lookup = {int(v): idx for idx, v in enumerate(gz_vals)}

    eps_G_qbar_comps = np.asarray(
        eps.unfold_eps_comps(0, sym.sym_mats_k[0], np.array([0.0, 0.0, 0.0])),
        dtype=np.int32,
    )

    n_eps = int(eps.nmtx[iq_eps])
    eps_G_qbar_comps = eps_G_qbar_comps[:n_eps]

    map_eps_to_flat = np.empty((n_eps,), dtype=np.int64)
    for ie, g3 in enumerate(eps_G_qbar_comps):
        gx_idx = gx_lookup.get(int(g3[0]))
        gy_idx = gy_lookup.get(int(g3[1]))
        gz_idx = gz_lookup.get(int(g3[2]))
        if gx_idx is None or gy_idx is None or gz_idx is None:
            raise ValueError(f"G triple {tuple(int(x) for x in g3)} not found in FFT grid.")
        map_eps_to_flat[ie] = gx_idx * ny * nz + gy_idx * nz + gz_idx

    # Locate G=0 index in eps ordering via gind_eps2rho
    gind = np.asarray(eps.gind_eps2rho[iq_eps, : n_eps], dtype=np.int64)
    G0_candidates = np.where(gind == 0)[0]
    if G0_candidates.size == 0:
        raise ValueError("Could not find G=0 index in eps g-indices.")
    G0_idx = int(G0_candidates[0])

    # 3) Build W in the BGW eps-space metric:
    #    W = diag(V) · eps^{-1}   (row scaling by v(G))
    #    IMPORTANT: use v from eps file order (unscaled by 1/Ω); apply 1/Ω after μν contraction.
    v_eps = np.asarray(eps.vcoul[iq_eps, :n_eps], dtype=np.complex128)
    # Rows scaled by v(G): left-multiply by diag(V)
    W_eps = (v_eps[:, None]) * epsinv

    # 4) Zero head and wings (body-only view) at q=0
    W_eps[G0_idx, :] = 0.0
    W_eps[:, G0_idx] = 0.0

    # 5) Build zeta(G) on the dense FFT grid
    #    (reproduce the phase convention used in compute_v_munu_from_zeta)
    n_mu, n_rtot = zeta_q_r.shape
    # zeta_r: (n_mu, nx, ny, nz)
    zeta_r = zeta_q_r.reshape(n_mu, nx, ny, nz)
    # Phase for wrapped q (typically zero here)
    fx = np.arange(nx)[None, :, None, None] / float(nx)
    fy = np.arange(ny)[None, None, :, None] / float(ny)
    fz = np.arange(nz)[None, None, None, :] / float(nz)
    qx, qy, qz = [float(qvec_wrapped[i]) for i in range(3)]
    nkx, nky, nkz = float(meta.nkx), float(meta.nky), float(meta.nkz)
    phase = np.exp(-2j * np.pi * (qx / nkx * fx + qy / nky * fy + qz / nkz * fz))
    zeta_G = np.fft.fftn(zeta_r * phase, axes=(-3, -2, -1))  # (n_mu,nx,ny,nz)
    Z_flat = zeta_G.reshape(n_mu, nx * ny * nz)
    Z = Z_flat[:, map_eps_to_flat]  # (n_mu, nG_eps)

    # 6) Contract: W_{mu,nu} = Z^* · W · Z^T, then divide by Ω to match μν units
    W_munu = Z.conj() @ W_eps @ Z.T
    W_munu = W_munu / float(wfn.cell_volume)
    return W_munu.astype(np.complex128)


__all__ = ["compute_Wmunu_from_eps0_body"]
