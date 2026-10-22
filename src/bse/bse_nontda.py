"""Structure-preserving non-TDA (full-BSE) eigensolver — the general
lowest-eigenvalue solver for the optical coupling-block Hamiltonian, reached
(alongside TDA) through the one ``solve_bse_sharded`` dispatch.

Physics (per-element; validated to machine precision on the gnppm fixture — see
``reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md`` §"non-TDA eigensolvers").

The optical non-TDA operator (``bse_stack_matvec.build_bse_stack_matvec_full``,
two calls of the fused pair applier, the TDA matvec's kernels) is
the para-Hermitian ("Casida") Hamiltonian

    H = [[ A ,  B ],
         [-B*, -A*]]                          (* = complex conjugate, elementwise)

with the resonant block ``A = D + K^x - K^d`` HERMITIAN (``A = A^H``) and the
coupling block ``B = K^x_B - K^d_B`` complex-SYMMETRIC (``B = B^T``, NOT
Hermitian — measured ||B - B^H||/||B|| ~ 1.5 on the spinor fixture).  Its
eigenvalues are real and come in +-omega pairs.  This is the physical optical
BSE (Onida-Reining-Rubio; Rohlfing-Louie).  The historical matvec computed the
naive [[A,B],[-B,-A]], whose spectrum is COMPLEX (max|Im| ~ 1e-4 Ry) — a bug that
survived because full-BSE-with-W was never value-validated; fixed in the ring
matvec that served this operator until 2026-10-21, and kept by the pair form.

The dense blocks come from the pair applier ``pair(x, s) = A x + s B conj(x)``
on REAL identity columns ``E`` (``conj(E) = E``):

    pair(E, +1) = (A + B) E,   pair(E, -1) = (A - B) E,
    A = (pair(E, +1) + pair(E, -1)) / 2,   B = (pair(E, +1) - pair(E, -1)) / 2,

one pair call per column chunk with the signs riding the trials.  (A +- B) on a
complex block composes from the full matvec (``make_ab_appliers``):
``H[U; sU] = [A U + s B U ; -B* U - s A* U]`` and the X-block is the top row.

Structure-preserving reduction, dispatched on B's symmetry:

* B complex-symmetric (optical spinor BSE, the physical case): (A +- B) are NOT
  Hermitian, so the clean product ``omega^2 = eig((A-B)(A+B))`` does not apply.
  The correct structure-preserving object is the HERMITIAN-DEFINITE pencil

      K z = omega Sigma z,  K = [[A, B],[B*, A*]] (Hermitian),  Sigma = diag(I,-I)

  with ``K`` positive definite for a stable (real) spectrum.  Lowest positive
  omega are the extreme eigenvalues of the Hermitian ``Shat = K^{-1/2} Sigma
  K^{-1/2}`` (eigenvalue ``1/omega``).  Solved densely for small windows — the
  BGW-parity regime, where BGW itself uses dense ScaLAPACK
  (``BSE_NTDA_SOLVER_SSEIG``).

* B Hermitian (real BSE / RPA density response): (A +- B) are Hermitian PD, so
  ``omega^2 = eig((A-B)(A+B))`` (Shao et al. LAA 488; BSEPACK product form) — the
  clean matrix-free (A+B)-metric Lanczos over the ``make_ab_appliers`` actions.
  Solved densely here for parity; used by the real-symmetric gate.

Eigenvectors: normalised (X, Y) pairs with the standard ``X^H X - Y^H Y = +1``
convention, stacked ``(n_eig, 2, nc, nv, nk)`` and wired to
``bse_io.write_eigenvectors_stream`` (use_tda=0, honest).

WHAT CHANGED 2026-08-08 (this header used to describe only the dense route)
---------------------------------------------------------------------------
* THE DENSE BUILD IS 3.30x FASTER — 333.4 s -> 101.1 s, driver ~347 s -> ~115 s
  (~3.0x), both landed with control variants (``FIX_nontda_feature.md`` §0).  The
  56 GiB ring-tensor path it used to take is gone.
* A RESTART PREFLIGHT NOW GUARDS THAT BUILD.  ``check_restart_reciprocity``
  runs BEFORE the O(N^2) assembly and refuses a stale W on
  ``max|W(q) - conj(W(-q))| / max|W|`` (threshold ``recip_tol``; ``None``
  measures without refusing).  The input decides whether ``A`` can be Hermitian
  at all, so it is checked before anything expensive is built — and the refusal
  names the regeneration command rather than reporting a number.  This replaced
  a projection: a preflight tells you the artifact is stale, a projection
  quietly makes a stale artifact look well-formed.
* THE MATRIX-FREE ROUTE IS NO LONGER A PROTOTYPE.  ``solver='matrixfree'``
  runs SDY Algorithm 4 (:mod:`solvers.bse_sp_lanczos`) over the fused pair
  applier ``bse_stack_matvec.build_bse_stack_pair_matvec``, at ``O(k N)`` memory
  and a cost set by the spectrum rather than by ``N``.  It is OPT-IN and it
  REFUSES on a sharded (px*py > 1) mesh — see ``_solve_nontda_matrix_free`` and
  ``SDY_SOLVER.md`` for the ζ-sharding kernel defect behind that refusal.
  ``'dense'`` remains the default AND the oracle: it is what the matrix-free
  route is gated against, and the only route returning the complete spectrum.
  Flipping that default is an owner decision with its own A/B; nothing here
  moved it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from common.fft_helpers import make_kfft_kminor
from .bse_ring_comm import make_bse_shardings
from .bse_head import head_shifted_eps_c
from common.log import get_logger, message

log = get_logger(__name__)
_LOGGER = log     # the module logger, where a ``log`` parameter shadows it

jax.config.update("jax_enable_x64", True)

# Dense materialisation is O(N^2) memory + O(1) batched matvecs; guard so a
# fine-grid request fails loudly instead of OOM-ing.  N = nc_pad * nv_pad * nk.
_DENSE_N_MAX = 4096


def _dense_refusal(N: int) -> str:
    """The one refusal text for a dense window over ``_DENSE_N_MAX``."""
    return (f"non-TDA dense window N={N} (n_cond x n_val x nk, mesh-padded) exceeds "
            f"_DENSE_N_MAX={_DENSE_N_MAX}: the dense structure-preserving solve "
            "assembles A and B, O(N^2).  Alternatives: full_bse = false (the TDA Davidson, the "
            "default eigensolve), feast = true (FEAST windows, refused unless converged to "
            "feast_tol_mev), or a smaller number_val_bands_fine / number_cond_bands_fine.  The matrix-free SDY "
            "route (bse_nontda solver='matrixfree') has no bse_jax flag and refuses on a "
            "sharded mesh.")


def refuse_dense_window(n_cond: int, n_val: int, nk: int, mesh_xy) -> int:
    """``GATE nontda_dense_size`` before any load: N of the dense non-TDA build
    from the logical window, the k count and the mesh padding (each band axis
    rounded up to its mesh extent, as the loader pads).  A degeneracy snap can
    only widen the window, so the build's own check stays."""
    px, py = (int(n) for n in mesh_xy.devices.shape)
    N = (-(-int(n_cond) // px) * px) * (-(-int(n_val) // py) * py) * int(nk)
    if N > _DENSE_N_MAX:
        raise ValueError(_dense_refusal(N))
    return N

# --- the dense build's column chunk -----------------------------------------
# The pair applier runs its trials through a ``lax.scan`` with one T-family
# alive whatever the width (``bse_stack_matvec``), so the chunk is bounded by
# the X-class trial block (2 x chunk x N complex128 per rank, gathered whole),
# not by the W term.  The ring matvec this build used until 2026-10-21 held a
# T per trial (compiled peak ~5.3 x T per column, 900 MiB a column on Si 4^3,
# 480 centroids), which forced 3-column chunks and 342 dispatches at N = 1024;
# a fixed 256 columns is at most 16 dispatches at the N = 4096 wall.
_DENSE_COL_CHUNK = 256


# --- the restart's q <-> -q reciprocity -------------------------------------
# The dense resonant block's direct term is
#
#   K_d[(cvk),(c'v'k')] = sum_{MN,ts} conj(psi_c[k]) W_MN(k-k') psi_c[k']
#                                     psi_v[k] conj(psi_v[k'])
#
# so **A is Hermitian if and only if W_MN(q) = conj(W_MN(-q))** for every
# (M, N, q) -- equivalently, if and only if the real-space W_R is REAL.  That
# is a property of the tile the GW stage wrote and of nothing in this solver,
# so it is measurable on the loaded array before any of the O(N^2) dense build
# runs.
#
# It is worth measuring because the BSE stage does NOT compute W -- it READS it
# off the GW restart (``tmp/isdf_tensors_*.h5``).  A restart written before the
# mini-BZ Coulomb head-slot fix carries the pre-fix operator frozen into a file,
# and repointing ``LORRAX_CHECKOUT`` at a fixed source tree changes nothing at
# all for a BSE-stage run.  MEASURED on the Si 4x4x4 record deck's 2026-08-07
# restart: ``W0_qmunu`` broke reciprocity at 8.635e-04 relative, the resulting A
# missed Hermiticity by 3.05e-05, and the definite-pencil gate below refused --
# correctly, but only after the whole dense build had been paid for.
#
# The refusal therefore moves to the input, where the defect is, and names the
# GW re-run that repairs it.  It does not replace the A gate: that gate is
# unchanged and still runs.
_NONTDA_RECIP_TOL = 1e-6


def w_q_reciprocity(W_q) -> float:
    """``max|W(q) - conj(W(-q))| / max|W|`` for the loaded screening tile.

    ``W_q`` is the ``(mu, nu, nkx, nky, nkz)`` array as the solver receives it.
    The ``-q`` gather is ``index -> (-index) % n`` on each of the three k axes,
    which is a flip followed by a roll of one; those axes are the REPLICATED
    axes of ``sh.W`` (``P('x','y',None,None,None)``), so this adds no collective
    and gathers nothing.  Measured post-load, on the array the solve will
    actually use, so it is independent of whether the restart stored q on the
    full grid or on a symmetry wedge that the reader unfolded."""
    ax = (2, 3, 4)
    W_mq = jnp.roll(jnp.flip(W_q, axis=ax), shift=(1, 1, 1), axis=ax)
    scale = jnp.max(jnp.abs(W_q))
    resid = jnp.max(jnp.abs(W_q - jnp.conj(W_mq)))
    return float(resid / jnp.maximum(scale, 1e-300))


def check_restart_reciprocity(W_q, *, tol=_NONTDA_RECIP_TOL, log=None,
                              input_file=None) -> float:
    """Preflight: refuse a restart whose W tile cannot make A Hermitian.

    Returns the measured relative residual.  ``tol=None`` measures and reports
    without refusing -- the escape hatch for a system that genuinely breaks the
    conjugate-reciprocity relation (broken time reversal), where the non-TDA
    definite-pencil reduction is questionable on its own terms anyway.

    This is a PRODUCED-INPUT refusal in the sense of
    ``absorption_common.load_dipole_h5``: the thing that is wrong is upstream of
    this process, and the message carries the command that fixes it."""
    rel = w_q_reciprocity(W_q)
    if log is not None:
        log(f"  [nontda] restart q<->-q reciprocity: "
            f"max|W(q) - conj(W(-q))| / max|W| = {rel:.3e} "
            f"(tol {'off' if tol is None else format(tol, '.1e')})")
    if tol is not None and rel > tol:
        deck = Path(input_file).name if input_file else "<deck>.in"
        raise ValueError(
            f"non-TDA refuses this restart: its screened-Coulomb tile is not "
            f"conjugate-reciprocal in q "
            f"(max|W(q) - conj(W(-q))| / max|W| = {rel:.3e}, tol {tol:.1e}).\n"
            f"A = D + K^x - K^d is Hermitian IF AND ONLY IF "
            f"W_MN(q) = conj(W_MN(-q)), so this restart CANNOT produce a "
            f"Hermitian resonant block; the definite-pencil solve would refuse "
            f"it anyway, but only after the O(N^2) dense build.\n"
            f"This is a STALE ARTIFACT, not a solver bug. The BSE stage does "
            f"not compute W, it reads it from the GW restart "
            f"(tmp/isdf_tensors_*.h5), so a restart written before the mini-BZ "
            f"Coulomb head-slot fix carries the pre-fix operator frozen into "
            f"the file -- pointing LORRAX_CHECKOUT at a fixed source tree "
            f"changes nothing for a BSE-stage run.  REGENERATE it by re-running "
            f"the GW stage in the deck directory:\n"
            f"    python3 -u -m gw.gw_jax -i {deck}\n"
            f"and re-run this solve against the rewritten restart.\n"
            f"To proceed without the check, pass recip_tol=None.")
    return rel


def _full_matvec_and_args(data, mesh_xy, sh, *, include_W):
    """The fused pair applier of the optical non-TDA operator and its 9 operands.

    ``pair(X, s, *args) = A X + s B conj(X)`` (``bse_stack_matvec.
    build_bse_stack_pair_matvec``); the dense build and the matrix-free route
    both read the operator through it."""
    from .bse_stack_matvec import build_bse_stack_pair_matvec
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    pair = build_bse_stack_pair_matvec(
        mesh_xy, nkx, nky, nkz, kernel="bse" if include_W else "rpa")
    if include_W:
        W_ifft = make_kfft_kminor(
            mesh_xy, (nkx, nky, nkz), sh.W.spec, kind="ifftn", norm="ortho")
        # W_q is DONATED and the caller-side reference dropped, copied from
        # ``bse_lanczos``'s W_R build.  Same shape in and out at a real
        # top-level boundary, so XLA grants the alias and W_R becomes W_q's
        # buffer instead of a second live tile for the whole non-TDA solve
        # (56.25 MiB/rank on the Si 4x4x4 deck; 404 MB/rank at mu=10015 /
        # P=64).  In-jit peak is unchanged — the win is entirely caller-side
        # (FFT_DONATION_AUDIT.md 2.3).  Value-identical.
        #
        # NOTE FOR CALLERS: this CONSUMES ``data["W_q"]``.  That is safe for
        # every caller today — ``_full_matvec_and_args`` is called exactly
        # once per solve, and ``bse_lanczos``'s non-TDA dispatch returns
        # immediately afterwards — but a caller that wants W_q back must pass
        # its own copy.
        W_R = jax.jit(W_ifft, donate_argnums=(0,))(data["W_q"])
        data["W_q"] = None      # release the caller-side reference
    else:
        W_R = data["W_q"]
    # A block takes the exact q=0 head on its diagonal; the coupling block has
    # none (<c k|v' k> = 0), which is what a head-free W_q gives it.
    args = (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
            head_shifted_eps_c(data["eps_c"], data), data["eps_v"], W_R,
            data["V_q0"], data["M"])
    return pair, args


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


def _materialize_A_B(pair, args, sh, nc, nv, nk, *, col_chunk=_DENSE_COL_CHUNK):
    """Dense A, B (N x N, N = nc*nv*nk) from the pair applier, one chunk of
    identity columns per call.

    The identity columns ``E`` are real, so ``pair(E, +1) = (A + B) E`` and
    ``pair(E, -1) = (A - B) E`` (module docstring); one call takes the chunk
    twice, ``[E; E]`` with signs ``[+1; -1]``, and ``A = (F + G)/2``,
    ``B = (F - G)/2``.  Every operator application is used."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    N = nc * nv * nk
    if N > _DENSE_N_MAX:
        raise ValueError(_dense_refusal(N))
    # The columns come back to every process's host through one replicated
    # copy, read from that process's own buffer: an np.asarray of the
    # X-sharded result spans other processes' devices and refuses at P > 1.
    to_host = jax.jit(lambda a: a, out_shardings=NamedSharding(sh.X.mesh, P()))
    eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk)
    A_out = np.empty((N, N), dtype=np.complex128)       # [flat, col]
    B_out = np.empty((N, N), dtype=np.complex128)
    for c0 in range(0, N, int(col_chunk)):
        blk = eye[c0:c0 + int(col_chunk)]                # (b, nc, nv, nk)
        b = blk.shape[0]
        U = device_put_process_local(np.concatenate([blk, blk], axis=0), sh.X)
        s = jnp.concatenate([jnp.ones((b,)), -jnp.ones((b,))])
        FG = np.asarray(to_host(pair(U, s, *args)).addressable_data(0)).reshape(2 * b, N)
        F, G = FG[:b].T, FG[b:].T
        A_out[:, c0:c0 + b] = 0.5 * (F + G)
        B_out[:, c0:c0 + b] = 0.5 * (F - G)
    return A_out, B_out


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
        raise ValueError(
            f"non-TDA resonant block A is not Hermitian (rel {a_herm:.2e}). "
            f"A is Hermitian iff the restart's W obeys W_MN(q) = conj(W_MN(-q)); "
            f"run bse_nontda.check_restart_reciprocity(data['W_q']) on the "
            f"loaded tile to see whether the input is the cause, and if it is, "
            f"regenerate the restart with `python3 -u -m gw.gw_jax -i <deck>.in`. "
            f"The threshold is NOT the thing to move: it is a structural "
            f"property of the operator, not a convergence tolerance.")
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


# ---------------------------------------------------------------------------
# The matrix-free route — SDY Algorithm 4 over the fused pair applier
# ---------------------------------------------------------------------------
# The dense route above costs 2N half-applications and O(N^2) memory, with
# ``_DENSE_N_MAX = 4096`` a hard wall.  The matrix-free route costs a number of
# applications set by the SPECTRUM rather than by N, and O(k N) memory.  On the
# record deck (N = 1024) that is a ~2x win; at N = 1e5 the dense route does not
# exist at all, so this is the difference between having a non-TDA capability at
# production grid and not having one.
#
# It is OPT-IN and it stays opt-in until the guard below stops firing.  See
# ``SDY_SOLVER.md``: the coupling block's screened-direct term ``K^d_B`` is
# currently WRONG when the ζ axis is sharded (measured: bit-consistent at
# px·py = 1, 55% different and 69% non-symmetric at 2x2, while the resonant
# block A and the coupling EXCHANGE term are bit-consistent at both).  A
# matrix-free non-TDA solve exists to run at scale, i.e. sharded, so shipping it
# on by default would ship a wrong number on exactly the configuration it is
# for.  The refusal is not a workaround for that defect -- it is the detector
# for it, and it will go quiet by itself the day the defect is fixed.
_MF_METRIC_TOL = 1e-8

#: The matrix-free route's stop: every root's ω residual below this (meV), the
#: TDA Davidson's default; cycles are capped by ``mf_n_restarts``.
MF_TOL_MEV = 0.01


def refuse_unconverged_sdy(dg, *, n_eig, tol_mev, n_restarts) -> str:
    """``GATE nontda_converged``: the SDY solve's stop as one line; any of the
    lowest ``n_eig`` with ``resid / 2ω`` at or above ``tol_mev`` refuses before the
    caller returns eigenpairs."""
    from common.units import RYD_TO_EV
    res = np.asarray(dg["resid_omega"])[:int(n_eig)] * RYD_TO_EV * 1.0e3
    n_conv, cyc, worst = int(np.sum(res < tol_mev)), int(dg["cycles"]), float(np.max(res))
    line = (f"non-TDA matrix-free {'converged' if n_conv == int(n_eig) else 'NOT converged'}: "
            f"{n_conv}/{int(n_eig)} roots with omega residual < {tol_mev:g} meV after "
            f"{cyc}/{int(n_restarts)} restart cycles; max {worst:.3e} meV")
    if n_conv < int(n_eig):
        raise RuntimeError(
            f"GATE nontda_converged: {line}.  No eigenpairs were returned.  Fix: "
            "raise mf_n_restarts or mf_m_max, or use the dense route (solver='dense').")
    return line


def _solve_nontda_matrix_free(data, mesh_xy, sh, args, nc, nv, nk, n_eig, *,
                              m_max=140, n_keep=40, n_restarts=50,
                              tol_mev=MF_TOL_MEV, metric_tol=_MF_METRIC_TOL, log=None):
    """Lowest ``n_eig`` (omega, X, Y) with no dense assembly.

    Returns ``(omega_Ry, Z (2N, n_eig))`` in the same convention
    ``solve_nontda_definite_pencil`` returns, so the caller cannot tell the two
    apart from their outputs -- which is the point of the gate that compares
    them.
    """
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    from .bse_stack_matvec import build_bse_stack_pair_matvec
    from solvers.bse_sp_lanczos import (sdy_lanczos_eig, sdy_pair_applications,
                                        sdy_steps)
    from common.units import RYD_TO_EV

    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    pair = build_bse_stack_pair_matvec(mesh_xy, nkx, nky, nkz)
    rep = NamedSharding(mesh_xy, P())
    UV_sh = NamedSharding(mesh_xy, P(None, None, "x", "y", None))
    in_sh = (sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y, sh.eps, sh.eps,
             sh.W, sh.V, sh.M)
    _l = log if log is not None else (lambda *_a, **_k: None)

    @partial(jax.jit, in_shardings=in_sh)
    def _go(*T):
        sp = jnp.asarray(1.0)
        sm = jnp.asarray(-1.0)
        om, X, Y, dg = sdy_lanczos_eig(
            lambda Z: pair(Z, sp, *T), lambda Z: pair(Z, sm, *T),
            (nc, nv, nk), n_eig=n_eig, m_max=m_max, n_keep=n_keep,
            n_restarts=n_restarts, tol=tol_mev / (RYD_TO_EV * 1.0e3),
            sharding=UV_sh, announce=_l)
        return (jax.lax.with_sharding_constraint(om, rep),
                jax.lax.with_sharding_constraint(X, rep),
                jax.lax.with_sharding_constraint(Y, rep),
                {k: jax.lax.with_sharding_constraint(v, rep)
                 for k, v in dg.items()})

    omega, X, Y, dg = _go(*args)
    omega = np.asarray(jax.device_get(omega))
    X = np.asarray(jax.device_get(X)).reshape(n_eig, -1)
    Y = np.asarray(jax.device_get(Y)).reshape(n_eig, -1)
    dg = {k: np.asarray(jax.device_get(v)) for k, v in dg.items()}

    cycles = int(dg["cycles"])
    _l(f"  [nontda] matrix-free: {sdy_steps(m_max, n_keep, cycles)} SDY "
       f"steps, {sdy_pair_applications(m_max, n_keep, cycles)} fused pair "
       f"applications (m_max={m_max} n_keep={n_keep} "
       f"{cycles}/{n_restarts} restart cycles)")
    _l(f"  [nontda] matrix-free invariants: metric_sym {float(dg['metric_sym_err']):.3e} "
       f"orth {float(dg['orth_err']):.3e} im_uu {float(dg['im_uu']):.3e} "
       f"im_vv {float(dg['im_vv']):.3e} drift {float(dg['imag_drift']):.3e}")

    # --- the guards, in the order the derivation puts them ------------------
    # (1) definiteness.  Re(x^H F(x)) <= 0 is a CERTIFICATE that K is not
    #     positive definite: triplet/charge instability, imaginary excitation
    #     energies.  Refuse, do not clamp -- the same reading and the same
    #     refusal the dense pencil route already gives.
    if float(dg["kappa_start"]) <= 0 or float(dg["beta_sq_min"]) <= 0:
        raise ValueError(
            f"non-TDA matrix-free: the kappa metric is not positive definite "
            f"(kappa_start {float(dg['kappa_start']):.3e}, beta^2 min "
            f"{float(dg['beta_sq_min']):.3e}).  K = [[A,B],[B*,A*]] is not "
            "positive definite: triplet/charge instability, so BSE excitation "
            "energies are imaginary.  This is physical — fix the "
            "screening/window; not hidden.")
    # (2) operator integrity.  The method needs A Hermitian AND B complex
    #     SYMMETRIC, which together are exactly the statement that the kappa
    #     metric Re(x^H F(x')) is symmetric.  One number covers both, and it is
    #     measured on the operator the solve actually ran, for the price of one
    #     pair application.
    if float(dg["metric_sym_err"]) > metric_tol:
        raise ValueError(
            f"non-TDA matrix-free: the kappa metric is asymmetric at "
            f"{float(dg['metric_sym_err']):.3e} (tol {metric_tol:g}).  The "
            "method requires A = A^H and B = B^T; the metric's symmetry is "
            "exactly that pair of conditions, so this number is the operator's "
            "integrity, not the solver's convergence.  KNOWN CAUSE: the "
            "coupling block's screened-direct term K^d_B is wrong when the "
            "zeta axis is sharded (px*py > 1) — see SDY_SOLVER.md.  Re-run "
            "with one process AND one visible device (px=py=1 requires the "
            "mesh to consume the whole job: create_mesh_xy refuses a 1x1 "
            "request on a multi-device job), or use the dense route, until "
            "that is fixed.")

    # (3) convergence: restarts stop on the ω residual; a capped stop refuses.
    _l("  [nontda] " + refuse_unconverged_sdy(
        dg, n_eig=n_eig, tol_mev=tol_mev, n_restarts=n_restarts))

    Z = np.concatenate([X.T, Y.T], axis=0)          # (2N, n_eig)
    order = np.argsort(omega)
    return omega[order], Z[:, order]


def solve_bse_nontda_sharded(data, mesh_xy, *, n_eig=5, include_W=True,
                             recip_tol=_NONTDA_RECIP_TOL, solver="dense",
                             mf_m_max=140, mf_n_keep=40, mf_n_restarts=50,
                             mf_tol_mev=MF_TOL_MEV, **_ignored):
    """Non-TDA (full BSE) lowest-eigenvalue solve — one entry, dispatched on B.

    Returns ``(eigenvalues_Ry (n_eig,), eigenvectors (n_eig, 2, nc, nv, nk),
    n_iter)`` matching the TDA ``solve_bse_sharded`` tuple; the pair axis carries
    (X, Y) with ``X^H X - Y^H Y = +1``.

    ``recip_tol`` is the restart preflight's threshold on
    ``max|W(q) - conj(W(-q))| / max|W|`` (:func:`check_restart_reciprocity`);
    ``None`` measures without refusing.

    ``solver`` selects the route:

    ``'dense'`` (default)
        Assemble ``A`` and ``B`` and diagonalise on the host.  ``O(N^2)``
        memory, ``2N`` half-applications, hard-walled at
        ``_DENSE_N_MAX``.  This is the ORACLE and it is not going away: it is
        what the matrix-free route is gated against, and it is the only route
        that returns the complete spectrum or individual matrix elements.

    ``'matrixfree'``
        SDY Algorithm 4 (:mod:`solvers.bse_sp_lanczos`) over the fused pair
        applier.  Cost set by the spectrum, not by ``N``; ``O(k N)`` memory.
        OPT-IN, and it refuses on a sharded mesh today — see
        ``_solve_nontda_matrix_free`` for why and for when that stops.
        Restart cycles stop when every root's ω residual is below
        ``mf_tol_mev`` (cap ``mf_n_restarts``); a capped stop refuses
        (GATE nontda_converged)."""
    if solver not in ("dense", "matrixfree"):
        raise ValueError(
            f"solve_bse_nontda_sharded: solver must be 'dense' or "
            f"'matrixfree', got {solver!r}")
    sh = make_bse_shardings(mesh_xy)
    _log0 = log.info
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["n_cond_pad"]); nv = int(data["n_val_pad"])
    # Preflight BEFORE the dense build: the input decides whether A can be
    # Hermitian at all, and the build is O(N^2).
    # Skipped when the W term is off (include_W=False has no W to check).
    if include_W and data.get("W_q") is not None:
        check_restart_reciprocity(data["W_q"], tol=recip_tol, log=_log0,
                                  input_file=data.get("input_file"))
    pair, args = _full_matvec_and_args(data, mesh_xy, sh, include_W=include_W)
    N = nc * nv * nk
    if solver == "matrixfree":
        omega, Z = _solve_nontda_matrix_free(
            data, mesh_xy, sh, args, nc, nv, nk, n_eig,
            m_max=mf_m_max, n_keep=mf_n_keep, n_restarts=mf_n_restarts,
            tol_mev=mf_tol_mev, log=_log0)
        evecs = np.stack([Z[:N, :].T.reshape(n_eig, nc, nv, nk),
                          Z[N:, :].T.reshape(n_eig, nc, nv, nk)], axis=1)
        return jnp.asarray(omega), jnp.asarray(evecs), jnp.int32(0)
    A, B = _materialize_A_B(pair, args, sh, nc, nv, nk)
    b_herm = np.linalg.norm(B - B.conj().T) / max(np.linalg.norm(B), 1e-300)
    if b_herm < 1e-6:
        omega, Z = solve_nontda_product(A, B, n_eig)           # real / RPA
    else:
        omega, Z = solve_nontda_definite_pencil(A, B, n_eig)   # optical spinor
    evecs = np.stack([Z[:N, :].T.reshape(n_eig, nc, nv, nk),
                      Z[N:, :].T.reshape(n_eig, nc, nv, nk)], axis=1)
    return jnp.asarray(omega), jnp.asarray(evecs), jnp.int32(0)
