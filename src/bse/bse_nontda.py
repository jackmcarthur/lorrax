"""Structure-preserving non-TDA (full-BSE) eigensolver — the general
lowest-eigenvalue solver for the optical coupling-block Hamiltonian, reached
(alongside TDA) through the one ``solve_bse_sharded`` dispatch.

Physics (per-element; validated to machine precision on the gnppm fixture — see
``reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md`` §"non-TDA eigensolvers").

The optical non-TDA operator (``build_bse_ring_matvec_full(screening=False)``) is
the para-Hermitian ("Casida") Hamiltonian

    H = [[ A ,  B ],
         [-B*, -A*]]                          (* = complex conjugate, elementwise)

with the resonant block ``A = D + K^x - K^d`` HERMITIAN (``A = A^H``) and the
coupling block ``B = K^x_B - K^d_B`` complex-SYMMETRIC (``B = B^T``, NOT
Hermitian — measured ||B - B^H||/||B|| ~ 1.5 on the spinor fixture).  Its
eigenvalues are real and come in +-omega pairs.  This is the physical optical
BSE (Onida-Reining-Rubio; Rohlfing-Louie).  The historical matvec computed the
naive [[A,B],[-B,-A]], whose spectrum is COMPLEX (max|Im| ~ 1e-4 Ry) — a bug that
survived because full-BSE-with-W was never value-validated; fixed in
``bse_ring_comm.build_bse_ring_matvec_full`` (screening=False).

(A +- B) actions compose from the full matvec VERBATIM (no new kernel; reuse):

    matvec([U;  U])[X-block] = A U + B U = (A + B) U
    matvec([U; -U])[X-block] = A U - B U = (A - B) U
    matvec([U;  0])[X-block] = A U
    matvec([0;  U])[X-block] = B U

because ``H[U; sU] = [A U + s B U ; -B* U - s A* U]`` and the X-block is the top row.

Structure-preserving reduction, dispatched on B's symmetry:

* B complex-symmetric (optical spinor BSE, the physical case): (A +- B) are NOT
  Hermitian, so the clean product ``omega^2 = eig((A-B)(A+B))`` does not apply.
  The correct structure-preserving object is the HERMITIAN-DEFINITE pencil

      K z = omega Sigma z,  K = [[A, B],[B*, A*]] (Hermitian),  Sigma = diag(I,-I)

  with ``K`` positive definite for a stable (real) spectrum.  Lowest positive
  omega are the extreme eigenvalues of the Hermitian ``Shat = K^{-1/2} Sigma
  K^{-1/2}`` (eigenvalue ``1/omega``).  Solved densely for small windows — the
  BGW-parity regime, where BGW itself uses dense ScaLAPACK
  (``BSE_NTDA_SOLVER_SSEIG``).  Matrix-free scale-out (FEAST-on-K / BSEPACK real
  transform) is the fine-grid follow-on (prototype validated,
  ``runs/MoS2/A_bse_nontda_2026-07-17/``).

* B Hermitian (real BSE / RPA density response): (A +- B) are Hermitian PD, so
  ``omega^2 = eig((A-B)(A+B))`` (Shao et al. LAA 488; BSEPACK product form) — the
  clean matrix-free (A+B)-metric Lanczos over the ``make_ab_appliers`` actions.
  Solved densely here for parity; used by the real-symmetric gate.

Eigenvectors: normalised (X, Y) pairs with the standard ``X^H X - Y^H Y = +1``
convention, stacked ``(n_eig, 2, nc, nv, nk)`` and wired to
``bse_io.write_eigenvectors_stream`` (use_tda=0, honest).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from common.fft_helpers import make_sharded_ifftn_3d
from .bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings

jax.config.update("jax_enable_x64", True)

# Dense materialisation is O(N^2) memory + O(1) batched matvecs; guard so a
# fine-grid request fails loudly instead of OOM-ing.  N = nc_pad * nv_pad * nk.
_DENSE_N_MAX = 4096


def _full_matvec_and_args(data, mesh_xy, sh, *, include_W):
    """Build the fixed optical non-TDA matvec + its threaded argument tuple."""
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    matvec = build_bse_ring_matvec_full(
        mesh_xy, nkx, nky, nkz, include_W=include_W, screening=False)
    if include_W:
        W_ifft = make_sharded_ifftn_3d(
            mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm="ortho")
        W_R = W_ifft(data["W_q"])
    else:
        W_R = data["W_q"]
    args = (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
            data["eps_c"], data["eps_v"], W_R, data["V_q0"],
            data["M_X"], data["M_Y"])
    return matvec, args


def make_ab_appliers(matvec_full, args, sh):
    """Return (apply_ApB, apply_AmB): matrix-free (A+-B) from the full matvec.

    ``U`` is a pair-basis block ``(b, nc, nv, nk)`` (``sh.X``); ``(A+-B)U`` is the
    X-block of the full matvec applied to ``[U; +-U]`` (module docstring).
    Reuses the production kernel verbatim; the design's matrix-free reduction
    (real BSE) and any future FEAST-on-K build on these two closures."""
    def _apply(U, sign):
        UU = jax.lax.with_sharding_constraint(
            jnp.stack([U, sign * U], axis=0).astype(jnp.complex128), sh.X_full)
        return jax.lax.with_sharding_constraint(matvec_full(UU, *args)[0], sh.X)
    return (lambda U: _apply(U, 1.0)), (lambda U: _apply(U, -1.0))


def _materialize_A_B(matvec_full, args, sh, nc, nv, nk, *, col_chunk=8):
    """Dense A, B (N x N, N = nc*nv*nk) from the full matvec, one identity slice
    at a time.  Columns are processed in chunks of ``col_chunk`` so the W-term
    T-tensor (linear in the trial-batch width) stays bounded — the gate runs on
    any 1 GPU, not only 80 GB HBM."""
    N = nc * nv * nk
    if N > _DENSE_N_MAX:
        raise ValueError(
            f"non-TDA dense window N={N} exceeds _DENSE_N_MAX={_DENSE_N_MAX}; "
            "the dense definite-pencil solver targets the BGW-parity (small) "
            "regime — fine-grid non-TDA needs matrix-free FEAST-on-K (follow-on).")
    eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk)

    def _cols(resonant):
        out = np.empty((N, N), dtype=np.complex128)     # out[flat, col]
        for c0 in range(0, N, col_chunk):
            blk = eye[c0:c0 + col_chunk]                 # (b, nc, nv, nk)
            z = np.zeros_like(blk)
            top, bot = (blk, z) if resonant else (z, blk)
            Xf = jax.lax.with_sharding_constraint(
                jnp.asarray(np.stack([top, bot], axis=0)), sh.X_full)
            cols = np.asarray(matvec_full(Xf, *args)[0]).reshape(blk.shape[0], N)
            out[:, c0:c0 + blk.shape[0]] = cols.T
        return out
    return _cols(True), _cols(False)


def _recover_xy(Z, N, n_eig):
    """Sigma-normalise Z=[X;Y] so X^H X - Y^H Y = +1; return (n_eig, 2, ...)-ready."""
    Xh, Yh = Z[:N], Z[N:]
    snorm = np.real(np.sum(np.conj(Xh) * Xh, axis=0) - np.sum(np.conj(Yh) * Yh, axis=0))
    return Z / np.sqrt(np.abs(snorm))[None, :]


def solve_nontda_definite_pencil(A, B, n_eig, *, pd_rtol=1e-10):
    """Dense structure-preserving solve of K z = omega Sigma z (host numpy).

    K = [[A,B],[B*,A*]] Hermitian; assert PD (else triplet/charge instability —
    (A-B) not positive definite => imaginary excitations; detected, not hidden).
    Returns (omega[:n_eig] ascending, Z (2N, n_eig)) with X^H X - Y^H Y = +1."""
    A = np.asarray(A); B = np.asarray(B); N = A.shape[0]
    a_herm = np.linalg.norm(A - A.conj().T) / max(np.linalg.norm(A), 1e-300)
    if a_herm > 1e-6:
        raise ValueError(f"non-TDA resonant block A is not Hermitian (rel {a_herm:.2e}).")
    K = np.block([[A, B], [B.conj(), A.conj()]])
    K = 0.5 * (K + K.conj().T)
    w, U = np.linalg.eigh(K)
    if w.min() <= pd_rtol * max(w.max(), 1e-300):
        raise ValueError(
            f"non-TDA pencil K=[[A,B],[B*,A*]] is NOT positive definite "
            f"(min eig {w.min():.3e}). Triplet/charge instability: (A-B) is not "
            "positive definite, so BSE excitation energies are imaginary. This is "
            "physical — fix the screening/window; not hidden.")
    Kmh = (U * (1.0 / np.sqrt(w))) @ U.conj().T                # K^{-1/2}
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    Shat = Kmh @ (sig[:, None] * Kmh)
    Shat = 0.5 * (Shat + Shat.conj().T)
    mu, Y = np.linalg.eigh(Shat)                               # mu = 1/omega, +/- pairs
    idx = np.where(mu > 0)[0]
    idx = idx[np.argsort(-mu[idx])][:n_eig]                    # largest 1/omega = lowest omega
    omega = 1.0 / mu[idx]
    Z = _recover_xy(Kmh @ Y[:, idx], N, n_eig)
    order = np.argsort(omega)
    return omega[order], Z[:, order]


def solve_nontda_product(A, B, n_eig):
    """Real-BSE product reduction (dense): omega^2 = eig((A-B)(A+B)); recover X,Y.

    Correct when (A +- B) are Hermitian PD (real / RPA case).  Reference for the
    matrix-free (A+B)-metric Lanczos over ``make_ab_appliers`` (follow-on)."""
    A = np.asarray(A); B = np.asarray(B); N = A.shape[0]
    ApB, AmB = A + B, A - B
    for nm, C in (("A+B", ApB), ("A-B", AmB)):
        wc = np.linalg.eigvalsh(0.5 * (C + C.conj().T))
        if wc.min() <= 0:
            raise ValueError(f"non-TDA product form: ({nm}) not positive definite "
                             f"(min eig {wc.min():.3e}); triplet instability.")
    mu, U = np.linalg.eig(AmB @ ApB)
    idx = np.argsort(mu.real)[:n_eig]
    omega = np.sqrt(np.clip(mu[idx].real, 0.0, None))
    Z = np.zeros((2 * N, n_eig), dtype=np.complex128)
    for j, i in enumerate(idx):
        u = U[:, i]
        u = u * np.sqrt(omega[j] / np.real(u.conj() @ (ApB @ u)))
        w_ = (ApB @ u) / omega[j]
        Z[:N, j], Z[N:, j] = 0.5 * (u + w_), 0.5 * (u - w_)
    order = np.argsort(omega)
    return omega[order], Z[:, order]


def solve_bse_nontda_sharded(data, mesh_xy, *, n_eig=5, include_W=True, **_ignored):
    """Non-TDA (full BSE) lowest-eigenvalue solve — one entry, dispatched on B.

    Returns ``(eigenvalues_Ry (n_eig,), eigenvectors (n_eig, 2, nc, nv, nk),
    n_iter)`` matching the TDA ``solve_bse_sharded`` tuple; the pair axis carries
    (X, Y) with ``X^H X - Y^H Y = +1``."""
    sh = make_bse_shardings(mesh_xy)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["n_cond_pad"]); nv = int(data["n_val_pad"])
    matvec, args = _full_matvec_and_args(data, mesh_xy, sh, include_W=include_W)
    with mesh_xy:
        A, B = _materialize_A_B(matvec, args, sh, nc, nv, nk)
    b_herm = np.linalg.norm(B - B.conj().T) / max(np.linalg.norm(B), 1e-300)
    if b_herm < 1e-6:
        omega, Z = solve_nontda_product(A, B, n_eig)           # real / RPA
    else:
        omega, Z = solve_nontda_definite_pencil(A, B, n_eig)   # optical spinor
    N = nc * nv * nk
    evecs = np.stack([Z[:N, :].T.reshape(n_eig, nc, nv, nk),
                      Z[N:, :].T.reshape(n_eig, nc, nv, nk)], axis=1)
    return jnp.asarray(omega), jnp.asarray(evecs), jnp.int32(0)
