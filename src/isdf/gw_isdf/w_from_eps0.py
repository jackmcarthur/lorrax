from __future__ import annotations

"""
Utility to construct W_{mu,nu}(q=0) from a BerkeleyGW eps^{-1} matrix:

    W_{GG'} = eps^{-1}_{GG'} · V_G'   with body-only (G=0 set to 0 and wings zeroed)
    W_{mu,nu} = sum_{GG'} zeta^*_{mu}(G) W_{GG'} zeta_{nu}(G')

This mirrors the mapping and contraction used in the legacy cohsex_isdf path
but forces the head and wings at q=0 to zero.
"""

from typing import Tuple
import numpy as np

from ..common.epsreader import EPSReader


def _map_indices_v_to_eps(
    wfn,
    eps: EPSReader,
    iqbar: int,
    sym,
    v_comps_q: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Return indices that map eps ordering -> our V/zeta ordering, and G0 index in eps order.

    Uses robust tuple-key mapping of (Gx,Gy,Gz) integer components.
    """
    # Components used by eps reader for this q (in eps ordering)
    eps_G_qbar_comps = np.asarray(
        eps.unfold_eps_comps(iqbar, sym.sym_mats_k[0], np.array([0.0, 0.0, 0.0])),
        dtype=np.int32,
    )
    n_eps = int(eps.nmtx[iqbar])
    eps_G_qbar_comps = eps_G_qbar_comps[:n_eps]

    # Build dict from our V/zeta ordering triples -> index
    v_map = {tuple(int(x) for x in v_comps_q[k]): int(k) for k in range(v_comps_q.shape[0])}
    vcoul_eps_inds = np.empty((n_eps,), dtype=np.int64)
    for ie, g3 in enumerate(eps_G_qbar_comps):
        key = (int(g3[0]), int(g3[1]), int(g3[2]))
        if key not in v_map:
            raise ValueError(f"G triple {key} from eps ordering not found in V/zeta ordering")
        vcoul_eps_inds[ie] = v_map[key]

    # Locate G=0 index in eps ordering via gind_eps2rho
    gind = np.asarray(eps.gind_eps2rho[iqbar, : n_eps], dtype=np.int64)
    G0_candidates = np.where(gind == 0)[0]
    if G0_candidates.size == 0:
        raise ValueError("Could not find G=0 index in eps g-indices.")
    G0_idx = int(G0_candidates[0])
    return vcoul_eps_inds, G0_idx


def compute_Wmunu_from_eps0_body(
    wfn,
    sym,
    meta,
    zeta_q_r: np.ndarray,
    qvec_wrapped: np.ndarray,
    vcoul_comps: np.ndarray,
    V_qfullG: np.ndarray,
    eps0_path: str,
) -> np.ndarray:
    """Compute body-only W_{mu,nu}(q=0) from eps^{-1}_q=0 and V(q=0,G) (head=0).

    Args:
        zeta_q_r: (n_mu, n_rtot) real-space zeta for this q (q=0)
        qvec_wrapped: fractional q-vector (length-3) (used for phase; here should be 0)
        vcoul_comps: (nG,3) integer components matching the V/zeta ordering
        V_qfullG: (nG,) values of Coulomb for this q with head already zeroed
        eps0_path: path to eps0mat.h5 (assumed eps^{-1} at q=0)
    Returns:
        W_{mu,nu} as a complex128 NumPy array (n_mu, n_mu)
    """
    # 1) Load eps^{-1}(q=0)
    eps = EPSReader(eps0_path)
    iq_eps = 0  # q=0 only
    epsinv = eps.get_eps_matrix(iq_eps)  # (nG_eps, nG_eps)

    # 2) Build mapping from eps ordering -> our V/zeta ordering
    vcoul_comps_np = np.asarray(vcoul_comps, dtype=np.int32)
    map_eps_to_v, G0_idx = _map_indices_v_to_eps(wfn, eps, iq_eps, sym, vcoul_comps_np)

    # 3) Reorder V to eps ordering and build W in the BGW eps-space metric:
    #    W = diag(V) · eps^{-1}   (row scaling by v(G))
    #    IMPORTANT: use v from eps file order (unscaled by 1/Ω); apply 1/Ω after μν contraction.
    n_eps = int(eps.nmtx[iq_eps])
    v_eps = np.asarray(eps.vcoul[iq_eps, :n_eps], dtype=np.complex128)
    # Rows scaled by v(G): left-multiply by diag(V)
    W_eps = (v_eps[:, None]) * epsinv

    # 4) Zero head and wings (body-only view) at q=0
    W_eps[G0_idx, :] = 0.0
    W_eps[:, G0_idx] = 0.0

    # 5) Reorder W back to V/zeta ordering
    # map eps->v; to invert, build an array inv_map such that inv_map[v_idx]=eps_idx
    inv_map = np.empty_like(map_eps_to_v)
    inv_map[map_eps_to_v] = np.arange(map_eps_to_v.size, dtype=map_eps_to_v.dtype)
    W_v = W_eps[inv_map][:, inv_map]

    # 6) Build zeta(G) in V/zeta ordering by FFT and extraction
    #    (reproduce the phase convention used in compute_v_munu_from_zeta)
    n_mu, n_rtot = zeta_q_r.shape
    nx, ny, nz = map(int, meta.fft_grid)
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
    # Extract G in V/zeta ordering
    idx = tuple([slice(None), vcoul_comps_np[:, 0], vcoul_comps_np[:, 1], vcoul_comps_np[:, 2]])
    Z = zeta_G[idx]  # (n_mu, nG)

    # 7) Contract: W_{mu,nu} = Z^* · W · Z^T, then divide by Ω to match μν units
    W_munu = Z.conj() @ W_v @ Z.T
    W_munu = W_munu / float(wfn.cell_volume)
    return W_munu.astype(np.complex128)


__all__ = ["compute_Wmunu_from_eps0_body"]
