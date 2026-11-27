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

one pair call per column chunk with the signs riding the trials.

Structure-preserving reduction, dispatched on B's symmetry:

* B complex-symmetric (optical spinor BSE, the physical case): (A +- B) are NOT
  Hermitian, so the clean product ``omega^2 = eig((A-B)(A+B))`` does not apply.
  The correct structure-preserving object is the HERMITIAN-DEFINITE pencil

      K z = omega Sigma z,  K = [[A, B],[B*, A*]] (Hermitian),  Sigma = diag(I,-I)

  with ``K`` positive definite for a stable (real) spectrum.  Lowest positive
  omega are the extreme eigenvalues of the Hermitian ``Shat = L^-1 Sigma
  L^-H`` with ``K = L L^H`` (eigenvalue ``1/omega``; Shao et al., LAA 488,
  Algorithm 1).  Solved densely for small windows — the
  BGW-parity regime, where BGW itself uses dense ScaLAPACK
  (``BSE_NTDA_SOLVER_SSEIG``).

* B Hermitian (real BSE / RPA density response): (A +- B) are Hermitian PD, so
  ``omega^2 = eig((A-B)(A+B))`` (Shao et al. LAA 488; BSEPACK product form).
  Solved densely here for parity; used by the real-symmetric gate.

Eigenvectors: normalised (X, Y) pairs with the standard ``X^H X - Y^H Y = +1``
convention (Sigma-orthonormal inside a degenerate multiplet too), stacked
``(n_eig, 2, nc, nv, nk)`` on the solve mesh's transition axes, and wired to
``bse_io.write_eigenvectors_stream`` (use_tda=0, honest) and to the per-state
dipoles ``<0|r|S> = sum_t X_t conj(d_t) - Y_t d_t``
(``absorption_common.exciton_dipole_projections``).

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

THE DAVIDSON ROUTE (2026-11-27)
-------------------------------
``solver='davidson'`` runs the structure-preserving block Davidson
(:mod:`solvers.bse_sp_davidson`) over the same pair applier, with what the
Tamm-Dancoff Davidson has: the paired exact diagonal ``[D; -D]`` as
preconditioner, the start block on the physical transitions, soft locking,
margin roots, restarts under a priced basis capacity and a residual stop in
meV that refuses when not met (:func:`_solve_nontda_davidson`).  No assembly,
no size limit, every vector on the mesh's transition axes.  It is reached from
``bse_jax`` by ``full_bse_solver = davidson``; the dense route is still what
``full_bse = true`` runs by default.  A ``K`` that is not positive definite
refuses with the measured ``lam_min``, or, under ``on_indefinite='continue'``
(``full_bse_instability = continue``), is solved on with the unstable roots
reported and kept in :data:`LAST_DAVIDSON`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
import jax
import jax.numpy as jnp

from .bse_ring_comm import make_bse_shardings
from .bse_head import head_shifted_eps_c
# The Davidson route's residual stop (meV), margin roots and basis admission
# are the Tamm-Dancoff Davidson's.
from .bse_davidson_helpers import payload_w_r
from .bse_lanczos import DAVIDSON_MARGIN_ROOTS, DAVIDSON_TOL_MEV, _price_davidson_basis
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
            "assembles A and B, O(N^2).  Alternatives: full_bse_solver = davidson (the "
            "structure-preserving Davidson for the full BSE: no assembly, no size limit, "
            "paired (X, Y) eigenvectors), full_bse = false (the TDA Davidson, the "
            "default eigensolve), feast = true (FEAST windows, refused unless converged to "
            "feast_tol_mev), or a smaller number_val_bands_fine / number_cond_bands_fine.")


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


def operand_sets(data, mesh_xy, sh, W_R):
    """The pair applier's operands for the two rows of ``K = [[A, B], [B^H, A']]``.

    Returns ``(up, low)``.  ``up`` are the operands of
    ``bse_stack_matvec.build_bse_stack_pair_matvec``'s applier after ``(X, s)``
    (after ``(X, Yc, s)`` with a partner).  ``low`` is None at ``Q = 0``, where
    the lower row is the conjugate of the upper one.

    A FINITE PAIR MOMENTUM.  The payload's own orbitals are the resonant pair
    ``(c at k+Q, v at k)``: ``psi_c_*``, ``eps_c`` at ``k+Q``, ``psi_v_*``,
    ``eps_v`` at ``k``, ``V_q0 = V(Q)``, ``M``.  ``data["partner"]`` holds the
    antiresonant pair ``(v at k+Q, c at k)`` under the same keys: ``psi_c_X``,
    ``psi_c_Y``, ``eps_c`` at ``k`` and ``psi_v_X``, ``psi_v_Y``, ``eps_v`` at
    ``k+Q`` (``M`` is built here when absent).  Both are labelled by ``k``, so
    the antiresonant pair function ``psi_v(k+Q, r) conj(psi_c(k, r'))`` has
    the resonant one's Bloch structure and the two encodes add before ONE
    convolution.  With the orbital sets ``D`` (the payload) and ``P`` (the
    partner),

        up  = (D.c on x, P.c on y, P.v on x, D.v on y, D.eps_c, D.eps_v, W_R, V,       D.M, P.M)
        low = (P.c on x, D.c on y, D.v on x, P.v on y, P.eps_c, P.eps_v, W_R, conj(V), P.M, D.M)

        (K z)_upper       = couple(X, Y, 1, *up)
        conj((K z)_lower) = couple(conj(Y), conj(X), 1, *low)

    The second line is the projection of the same kernel onto the antiresonant
    pair functions, conjugated; it needs ``W_{mu nu}(q) = conj(W_{nu mu}(q))``
    and nothing else.  ``K`` is Hermitian when also ``W(q) = conj(W(-q))``
    (:func:`check_restart_reciprocity`) and ``V(Q)`` is Hermitian.  The lower
    right block is ``A' = conj(A(-Q))`` in this labelling and the lower left
    ``B^H``: there is no ``B = B^T`` and no ``+-`` pairing of roots unless a
    symmetry links ``Q`` and ``-Q``.

    Both conduction diagonals take the exact ``q = 0`` screened head
    (``head_shifted_eps_c``); the coupling block has none, its overlaps
    ``<c k+Q|v' k+Q>`` and ``<c' k|v k>`` vanish.  A ``head_tensor`` payload
    adds ``(D_head, M_head)`` to both sets (the same operands: the head's
    antiresonant vertex is ``conj(d)`` in both rows).

    OCCUPATION WEIGHTS.  ``data["occupation_weights"]`` (n_c, n_v, nk),
    ``sqrt(f_v - f_c)`` of the payload's transitions
    (``bse_preconditioner.occupation_weight_cv_k``), and with a partner
    ``data["partner"]["occupation_weights"]`` for the antiresonant pairs, are
    the last operands: ``(w,)`` at ``Q = 0`` (one transition set), ``(w, wb)``
    in ``up`` and ``(wb, w)`` in ``low``.  ``A = dE + w K w``, ``B = w K_B wb``,
    ``A' = dE' + wb K' wb``; zero-weight pairs are masked.
    """
    head = tuple(data.get("head_tensor") or ())
    partner = data.get("partner")
    eps_c = head_shifted_eps_c(data["eps_c"], data)
    w = data.get("occupation_weights")
    if partner is None:
        return (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
                eps_c, data["eps_v"], W_R, data["V_q0"], data["M"]) + head + (
                    () if w is None else (w,)), None
    wb = partner.get("occupation_weights")
    if (w is None) != (wb is None):
        raise ValueError(
            "GATE nontda_occupation_weights: the payload and its partner must both carry "
            "occupation_weights or neither (got: "
            f"payload {'with' if w is not None else 'without'}, partner "
            f"{'with' if wb is not None else 'without'}; want: both).  Why: the coupling block "
            "is sqrt(df) K_B sqrt(df') with the resonant pairs' weights on one side and the "
            "antiresonant pairs' on the other.  Fix: set data['partner']['occupation_weights'] "
            "from the occupations at (v at k+Q, c at k).")
    occ_up, occ_low = (() if w is None else (w, wb)), (() if w is None else (wb, w))
    from .bse_preconditioner import compute_pair_amplitude
    if partner.get("M") is None:
        partner["M"] = jax.jit(compute_pair_amplitude, out_shardings=sh.M)(
            partner["psi_c_X"], partner["psi_v_X"])
    if partner.get("V_conj") is None:
        partner["V_conj"] = jax.jit(jnp.conj, out_shardings=sh.V)(data["V_q0"])
    eps_c_low = head_shifted_eps_c(partner["eps_c"], data)
    up = (data["psi_c_X"], partner["psi_c_Y"], partner["psi_v_X"], data["psi_v_Y"],
          eps_c, data["eps_v"], W_R, data["V_q0"], data["M"], partner["M"]) + head + occ_up
    low = (partner["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], partner["psi_v_Y"],
           eps_c_low, partner["eps_v"], W_R, partner["V_conj"], partner["M"], data["M"]) + head + occ_low
    return up, low


def _operator(data, mesh_xy, sh, *, include_W):
    """``(pair, up, low)``: the fused pair applier of the payload's operator
    and its operand sets (:func:`operand_sets`).  With a partner the applier
    is the coupled one, ``couple(X, Yc, s, *up)``."""
    from .bse_stack_matvec import build_bse_stack_pair_matvec
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    pair = build_bse_stack_pair_matvec(
        mesh_xy, nkx, nky, nkz, kernel="bse" if include_W else "rpa",
        exchange_channel=data.get("exchange_channel", "singlet"),
        head_tensor=data.get("head_tensor") is not None,
        partner=data.get("partner") is not None,
        occupations=data.get("occupation_weights") is not None)
    up, low = operand_sets(data, mesh_xy, sh, payload_w_r(data, mesh_xy, sh, include_W=include_W))
    return pair, up, low


def kernel_matvec(couple):
    """``apply_K(sets, Z)``: the Hermitian ``K`` on pair rows at a finite pair
    momentum, two calls of the coupled applier (:func:`operand_sets`).

    ``Z`` (b, 2, n_c, n_v, nk) at ``P(None, None, 'x', 'y', None)`` holds
    ``[X; Y]`` per row; ``sets = (up, low)``.  Returns ``K Z`` in the same
    layout.  Complex-linear in ``Z``."""
    def apply_K(sets, Z):
        up, low = sets
        X, Y = Z[:, 0], Z[:, 1]
        one = jnp.ones((Z.shape[0],), jnp.float64)
        top = couple(X, Y, one, *up)
        bottom = jnp.conj(couple(jnp.conj(Y), jnp.conj(X), one, *low))
        return jnp.stack([top, bottom], axis=1)
    return apply_K


def _full_matvec_and_args(data, mesh_xy, sh, *, include_W):
    """The fused pair applier of the optical non-TDA operator at ``Q = 0`` and
    its operands.

    ``pair(X, s, *args) = A X + s B conj(X)`` (``bse_stack_matvec.
    build_bse_stack_pair_matvec``); the dense build and the Davidson route
    both read the operator through it.  CONSUMES ``data["W_q"]`` and leaves
    ``data["W_R"]`` (:func:`payload_w_r`)."""
    if data.get("partner") is not None:
        raise ValueError("_full_matvec_and_args serves Q = 0; a payload with a partner "
                         "(finite Q) goes through _operator")
    pair, args, _ = _operator(data, mesh_xy, sh, include_W=include_W)
    return pair, args


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


def _materialize_K(couple, up, low, sh, nc, nv, nk, *, col_chunk=_DENSE_COL_CHUNK):
    """The dense blocks ``(A, B, B_low, A_low)`` of ``K = [[A, B], [B_low, A_low]]``
    at a finite pair momentum, from the coupled applier on identity columns.

    One chunk ``E`` of real identity columns gives all four blocks' columns in
    two calls (:func:`operand_sets`): ``couple([E; 0], [0; E], 1, *up) =
    [A E; B E]`` and ``conj(couple([0; E], [E; 0], 1, *low)) = [B_low E;
    A_low E]``.  ``B_low`` is MEASURED, not set to ``B^H``: the two rows come
    from different operand sets, and :func:`solve_nontda_definite_pencil`
    refuses when they disagree."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    N = nc * nv * nk
    if N > _DENSE_N_MAX:
        raise ValueError(_dense_refusal(N))
    to_host = jax.jit(lambda a: a, out_shardings=NamedSharding(sh.X.mesh, P()))
    eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk)
    blocks = [np.empty((N, N), dtype=np.complex128) for _ in range(4)]
    for c0 in range(0, N, int(col_chunk)):
        blk = eye[c0:c0 + int(col_chunk)]
        b = blk.shape[0]
        E0 = device_put_process_local(np.concatenate([blk, np.zeros_like(blk)], axis=0), sh.X)
        Z0 = device_put_process_local(np.concatenate([np.zeros_like(blk), blk], axis=0), sh.X)
        one = jnp.ones((2 * b,))
        top = np.asarray(to_host(couple(E0, Z0, one, *up)).addressable_data(0)).reshape(2 * b, N)
        bot = np.conj(np.asarray(to_host(couple(Z0, E0, one, *low)).addressable_data(0))
                      ).reshape(2 * b, N)
        blocks[0][:, c0:c0 + b], blocks[1][:, c0:c0 + b] = top[:b].T, top[b:].T
        blocks[2][:, c0:c0 + b], blocks[3][:, c0:c0 + b] = bot[:b].T, bot[b:].T
    return tuple(blocks)


def _recover_xy(Z, N, n_eig):
    """Sigma-normalise Z=[X;Y] so X^H X - Y^H Y = +1; return (n_eig, 2, ...)-ready."""
    Xh, Yh = Z[:N], Z[N:]
    snorm = np.real(np.sum(np.conj(Xh) * Xh, axis=0) - np.sum(np.conj(Yh) * Yh, axis=0))
    return Z / np.sqrt(np.abs(snorm))[None, :]


#: Relative tolerance of the dense route's coupling-block symmetry gate, the same
#: structural 1e-6 as the Hermiticity gate on A (not a convergence tolerance).
_B_SYM_TOL = 1e-6


def refuse_asymmetric_coupling(B, *, tol=_B_SYM_TOL) -> float:
    """``GATE nontda_coupling_symmetric``: the coupling block must be complex
    symmetric, ``B = B^T`` (the pair-exchange symmetry of ``B_{cv k, c'v'k'}``).

    ``K = [[A, B], [B*, A*]]`` is Hermitian only if ``B = B^T``; the dense solves
    symmetrise ``K`` (or ``A+-B``) as round-off cleanup, which would otherwise
    replace an asymmetric ``B`` by ``(B + B^T)/2`` unseen.  Refuses on
    ``||B - B^T|| / ||B|| > tol``; returns the measured value."""
    B = np.asarray(B)
    b_sym = float(np.linalg.norm(B - B.T) / max(np.linalg.norm(B), 1e-300))
    if b_sym > tol:
        raise ValueError(
            f"GATE nontda_coupling_symmetric: non-TDA coupling block B is not "
            f"complex symmetric (got: ||B - B^T|| / ||B|| = {b_sym:.2e}; want: "
            f"<= {tol:g}).  Why: the pencil K = [[A,B],[B*,A*]] is Hermitian only "
            f"for B = B^T, and the dense solve would silently replace B by "
            f"(B + B^T)/2 and return roots of a different operator.  Fix: check "
            f"the restart's W with bse_nontda.check_restart_reciprocity(data['W_q']) "
            f"and the pair applier's B.  The threshold is a structural property "
            f"of the operator, not a convergence tolerance.")
    return b_sym


def solve_nontda_definite_pencil(A, B, n_eig, *, pd_rtol=1e-10, lower=None):
    """Dense structure-preserving solve of K z = omega Sigma z (host numpy).

    K = [[A,B],[B*,A*]] Hermitian; assert PD (else triplet/charge instability —
    (A-B) not positive definite => imaginary excitations; detected, not hidden).
    Refuses an A that is not Hermitian and a B that is not complex symmetric
    (:func:`refuse_asymmetric_coupling`) before it symmetrises K.

    ``lower = (B_low, A_low)``: the general pencil of a finite pair momentum,
    ``K = [[A, B], [B_low, A_low]]`` with the lower blocks as measured
    (:func:`_materialize_K`).  ``A`` and ``A_low`` must be Hermitian and
    ``B_low = B^H`` (``GATE nontda_lower_row``); ``B`` has no symmetry there
    and the roots returned are those of positive type, ``z^H Sigma z > 0``.

    The reduction is the Cholesky form of Shao, da Jornada, Yang, Deslippe and
    Louie, Linear Algebra Appl. 488 (2016) 148, Algorithm 1, on the complex
    pencil: ``K = L L^H``, and the Hermitian ``L^-1 Sigma L^-H`` has the
    eigenvalues ``+-1/omega`` with ``z = L^-H y / sqrt(1/omega)``.  It replaced
    the ``K^(-1/2) Sigma K^(-1/2)`` form (2026-11-27), which lost digits on
    every root once ``K`` had a small eigenvalue: on a 40-transition pencil with
    ``lambda_min(K) = 1e-8`` (``1e-9``) Ry the 39 roots above the soft one were
    wrong by 4.3e-10 (5.0e-9) Ry and are now right to 4.3e-13 (2.3e-12) Ry
    (``tests/test_bse_full_oracle_cpu.py``).  The definiteness gate is unchanged.

    Parameters
    ----------
    A, B : (N, N) complex, Ry
        Resonant block (Hermitian) and coupling block (complex symmetric),
        flat transition index, host arrays.
    n_eig : int
        Lowest positive roots returned.
    pd_rtol : float
        ``K`` counts as positive definite above ``pd_rtol`` times its largest
        eigenvalue.

    Returns
    -------
    omega : (n_eig,) float, Ry, ascending.
    Z : (2N, n_eig) complex
        ``[X; Y]`` columns with ``X^H X - Y^H Y = +1``, Sigma-orthonormal
        inside a degenerate multiplet too.
    """
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
    if lower is None:
        refuse_asymmetric_coupling(B)
        K = np.block([[A, B], [B.conj(), A.conj()]])
    else:
        B_low, A_low = (np.asarray(x) for x in lower)
        scale = max(np.linalg.norm(A), 1e-300)
        low_herm = np.linalg.norm(A_low - A_low.conj().T) / max(np.linalg.norm(A_low), 1e-300)
        adjoint = np.linalg.norm(B_low - B.conj().T) / scale
        if low_herm > 1e-6 or adjoint > 1e-6:
            raise ValueError(
                f"GATE nontda_lower_row: the lower row of K = [[A, B], [B_low, A_low]] at a finite "
                f"pair momentum does not close the Hermitian pencil (got: ||A_low - A_low^H|| / "
                f"||A_low|| = {low_herm:.2e}, ||B_low - B^H|| / ||A|| = {adjoint:.2e}; want: both "
                f"<= 1e-6).  Why: the two rows are built from the payload's and the partner's "
                f"orbitals with one kernel; they disagree when W is not conjugate-reciprocal "
                f"(check_restart_reciprocity), V(Q) is not Hermitian, or the partner orbitals are "
                f"not the payload's bands at the other momentum.  Fix: check data['partner'] "
                f"(bse_nontda.operand_sets) and the restart's W.  The threshold is structural, "
                f"not a convergence tolerance.")
        K = np.block([[A, B], [B_low, A_low]])
    K = 0.5 * (K + K.conj().T)                       # round-off cleanup, after the gates
    w = np.linalg.eigvalsh(K)                        # the definiteness gate's measurement
    if w.min() <= pd_rtol * max(w.max(), 1e-300):
        raise ValueError(
            f"non-TDA pencil K=[[A,B],[B*,A*]] is NOT positive definite "
            f"(min eig {w.min():.3e}). Triplet/charge instability: (A-B) is not "
            "positive definite, so BSE excitation energies are imaginary. This is "
            "physical — fix the screening/window; not hidden.")
    L = np.linalg.cholesky(K)                                  # K = L L^H
    sig = np.concatenate([np.ones(N), -np.ones(N)])
    T = solve_triangular(L, np.diag(sig).astype(K.dtype), lower=True)     # L^-1 Sigma
    Shat = solve_triangular(L, T.conj().T, lower=True).conj().T           # L^-1 Sigma L^-H
    Shat = 0.5 * (Shat + Shat.conj().T)
    mu, Y = np.linalg.eigh(Shat)                               # mu = 1/omega, +/- pairs
    idx = np.where(mu > 0)[0]
    idx = idx[np.argsort(-mu[idx])][:n_eig]                    # largest 1/omega = lowest omega
    omega = 1.0 / mu[idx]
    Z = _recover_xy(solve_triangular(L.conj().T, Y[:, idx], lower=False), N, n_eig)
    order = np.argsort(omega)
    return omega[order], Z[:, order]


def solve_nontda_product(A, B, n_eig):
    """Real-BSE product reduction (dense): omega^2 from (A-B)(A+B); recover X, Y.

    Correct when H is real (A and B real symmetric, so (A +- B) are real
    symmetric PD: the RPA / real-orbital case).

    The product is solved in its symmetric form,
    ``S (A-B) S t = omega^2 t`` with ``S = (A+B)^{1/2}``, by ``eigh``: the
    ``t`` are orthonormal also inside a degenerate multiplet, so
    ``u = omega^{1/2} S^{-1} t`` and ``w = S t / omega^{1/2}`` give
    ``X = (u + w)/2``, ``Y = (u - w)/2`` with
    ``X_i^H X_j - Y_i^H Y_j = delta_ij`` (a general ``eig`` of the
    non-symmetric product returns an arbitrary basis of a degenerate
    eigenspace, and each state's oscillator strength then carries the
    multiplet's cross terms).  Returns ``(omega (n_eig,), Z (2N, n_eig))`` as
    :func:`solve_nontda_definite_pencil` does."""
    A = np.asarray(A); B = np.asarray(B); N = A.shape[0]
    refuse_asymmetric_coupling(B)
    ApB, AmB = A + B, A - B
    for nm, C in (("A+B", ApB), ("A-B", AmB)):
        wc = np.linalg.eigvalsh(0.5 * (C + C.conj().T))
        if wc.min() <= 0:
            raise ValueError(f"non-TDA product form: ({nm}) not positive definite "
                             f"(min eig {wc.min():.3e}); triplet instability.")
    wp, Up = np.linalg.eigh(0.5 * (ApB + ApB.conj().T))
    S = (Up * np.sqrt(wp)) @ Up.conj().T                          # (A+B)^{1/2}
    S_inv = (Up / np.sqrt(wp)) @ Up.conj().T
    M = S @ AmB @ S
    mu, T = np.linalg.eigh(0.5 * (M + M.conj().T))               # mu = omega^2, ascending
    omega = np.sqrt(np.clip(mu[:n_eig], 0.0, None))
    T = T[:, :n_eig]
    u = (S_inv @ T) * np.sqrt(omega)[None, :]
    w_ = (S @ T) / np.sqrt(omega)[None, :]
    Z = np.concatenate([0.5 * (u + w_), 0.5 * (u - w_)], axis=0).astype(np.complex128)
    return omega, Z


def _h_is_real(A, B, *, tol=1e-6) -> bool:
    """True when ``H = [[A, B], [-B*, -A*]]`` is real to ``tol`` (relative to
    ``||A||``): the one case the product reduction holds.  A complex-Hermitian
    ``A`` with a real ``B`` (or ``B = 0``) is not that case: ``A* != A``
    couples ``Y`` through ``A*``, and the pencil is the route."""
    scale = max(np.linalg.norm(A), 1e-300)
    return (np.linalg.norm(np.imag(A)) / scale < tol
            and np.linalg.norm(np.imag(B)) / scale < tol)


# ---------------------------------------------------------------------------
# The Davidson route — structure-preserving block Davidson over the pair applier
# ---------------------------------------------------------------------------
# ``solvers.bse_sp_davidson`` on the same fused pair applier, with the pieces the
# Tamm-Dancoff Davidson uses (``bse_davidson_helpers``): the start block on the
# physical transitions, the exact diagonal, soft locking, margin roots, the
# basis admission.  It keeps every vector on the mesh's transition axes; only
# coefficient matrices cross it.  Reached from ``bse_jax`` by
# ``full_bse_solver = davidson``; the dense route stays the default.

#: Largest ``|X_i^H X_j - Y_i^H Y_j - delta_ij|`` accepted on the returned roots.
#: The solver's normalisation is exact to rounding (1e-12 with ``|Y|^2`` of 1e4),
#: so a larger value is a broken kernel, not a convergence matter.
_DAVIDSON_SIGMA_TOL = 1e-8

#: The last Davidson solve of the full BSE, for the driver's report (the role
#: ``solvers.davidson.LAST_RUN`` has for the Tamm-Dancoff route): ``line`` (the
#: stop line), ``info`` (the solver's), ``omega_ry`` (every block root, in the
#: solver's order), ``n_special`` (unstable and non-positive roots, listed
#: first), ``lam_min_ry``, and ``special_vectors``, the ``(n_special, 2, nc, nv,
#: nk)`` pairs of those roots on the mesh's transition axes (None when there
#: are none).
LAST_DAVIDSON: dict = {}


def refuse_unconverged_davidson(info, *, n_eig, tol_mev, budget) -> str:
    """``GATE nontda_davidson_converged``: the Davidson stop as one line.

    ``info`` is ``solvers.bse_sp_davidson.sp_davidson``'s.  Any stop but
    CONVERGED on the tested roots refuses before the caller returns
    eigenpairs, and so does a normalisation ``X_i^H X_j - Y_i^H Y_j`` off the
    identity by more than ``_DAVIDSON_SIGMA_TOL``.  ``budget`` is the
    iteration cap the solve ran under.
    """
    from solvers.bse_sp_davidson import CONVERGED, STALLED
    from common.units import RYD_TO_EV
    ry_mev = RYD_TO_EV * 1.0e3
    n_test = int(info["n_tested"])
    res = np.asarray(info["residuals"])[:n_test] * ry_mev
    n_conv = int(np.sum(np.asarray(info["converged"])[:n_test]))
    status = int(info["status"])
    word = ("converged" if status == CONVERGED else
            "STALLED" if status == STALLED else "NOT converged")
    worst = float(np.max(res))
    line = (f"full-BSE Davidson {word}: {n_conv}/{n_test} roots with residual < {tol_mev:g} meV "
            f"in {int(info['iterations'])}/{int(budget)} iterations, {int(info['pair_calls'])} "
            f"pair calls ({float(info['pair_calls']) / int(n_eig):.1f} per root), "
            f"{int(info['restarts'])} restarts; max residual {worst:.3e} meV; "
            f"lam_min {float(info['lam_min']) * ry_mev:.6g} meV; "
            f"|X^H X - Y^H Y - 1| {float(info['sigma_err']):.1e}")
    if status != CONVERGED:
        raise RuntimeError(
            f"GATE nontda_davidson_converged: {line} (got: status {info['status_name']}; want: "
            f"CONVERGED).  Why: an unconverged root is not an eigenpair.  No eigenpairs were "
            f"returned.  Fix: raise solver_max_iter (davidson_max_iter) or davidson_m_max, "
            f"loosen davidson_tol_mev, or use the dense route (full_bse_solver = dense) inside "
            f"its size limit.")
    if not float(info["sigma_err"]) <= _DAVIDSON_SIGMA_TOL:
        raise RuntimeError(
            f"GATE nontda_davidson_converged: {line} (got: |X_i^H X_j - Y_i^H Y_j - delta_ij| = "
            f"{float(info['sigma_err']):.3e}; want: <= {_DAVIDSON_SIGMA_TOL:g}).  Why: the "
            f"solver normalises the pairs exactly, so this is a broken subspace kernel, and "
            f"dipoles of such pairs are wrong.  No eigenpairs were returned.  Fix: run the "
            f"dense route (full_bse_solver = dense) and report the measured value.")
    return line


def _solve_nontda_davidson(pair, args, data, mesh_xy, sh, n_eig, *, include_W, w_q0,
                           davidson_tol_mev, davidson_max_iter, davidson_m_max,
                           davidson_margin, davidson_margin_tol_mev, davidson_precond,
                           davidson_eps_shift_Ry, davidson_n_random_init, on_indefinite,
                           davidson_plan_key, davidson_olsen=False, low=None):
    """Lowest ``n_eig`` (omega, X, Y) by the structure-preserving Davidson.

    ``pair`` and ``args`` are the fused pair applier and its operands
    (:func:`_operator`, the caller's ``include_W``); ``w_q0`` is
    the ``q = 0`` slice of ``W_q`` for the exact diagonal (None without W or
    with the bare diagonal).  The options are ``solve_bse_nontda_sharded``'s.

    ``low`` (the partner's operand set, :func:`operand_sets`) selects the
    finite-momentum solve: ``pair`` is then the coupled applier, the operator
    is ``K`` on pair rows (:func:`kernel_matvec`) and the solver runs its
    unpaired form.  The diagonal of ``H`` is ``[D; -D_low]`` there, ``D_low``
    the diagonal of ``A'``, the Tamm-Dancoff diagonal of the partner's set.

    Returns ``(omega_Ry (n_eig,) host, pairs (n_eig, 2, nc, nv, nk) on
    P(None, None, 'x', 'y', None), info)``; the snapshot is :data:`LAST_DAVIDSON`.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.units import RYD_TO_EV
    from solvers.bse_sp_davidson import basis_capacity, sp_davidson
    from .bse_davidson_helpers import (bse_pair_diagonal_precond, build_bse_exact_diagonal,
                                       init_bse_subspace, resolve_precond_route)

    ry_mev = RYD_TO_EV * 1.0e3
    psi_c_X, psi_v_Y, eps_c, eps_v, V_q0, M = (args[i] for i in (0, 3, 4, 5, 7, 8))
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    nc, nv = int(data["n_cond_pad"]), int(data["n_val_pad"])
    n_cond, n_val = int(data["n_cond"]), int(data["n_val"])
    channel = data.get("exchange_channel", "singlet")
    paired = low is None

    # The diagonal of H = [[A, B], [-B*, -A*]] is [D; -D], D the diagonal of A
    # (bse_davidson_helpers.bse_pair_diagonal_precond): the Tamm-Dancoff one.
    # At a finite pair momentum it is [D; -D_low].
    route = resolve_precond_route(davidson_precond, nc * nv * nk)
    diag_sh = NamedSharding(mesh_xy, P("x", "y", None))
    diag = diag_low = None
    if route == "exact":
        diag = build_bse_exact_diagonal(eps_c, eps_v, psi_c_X, psi_v_Y, w_q0, M, V_q0, nk,
                                        sharding=diag_sh, exchange_channel=channel)
        if not paired:
            diag_low = build_bse_exact_diagonal(
                low[4], low[5], low[0], low[3], w_q0, low[8], low[7], nk, sharding=diag_sh,
                exchange_channel=channel, memo=False)
    log.info("full-BSE Davidson preconditioner: " + (
        f"exact diagonal of A, dE + (V_x{' - W_d' if include_W else ''})/nk" if route == "exact"
        else "bare dE") + (", paired as [D; -D]" if paired else
                           ", with the partner's diagonal as [D; -D_low] (finite pair momentum)"))
    if davidson_olsen:
        log.info("full-BSE Davidson preconditioner: Olsen correction ON (Sigma metric)")
    # Occupation weights: the diagonals of A = dE + w K w (and A'), masked pairs parked.
    weights = data.get("occupation_weights")
    if weights is not None:
        from .bse_davidson_helpers import occupation_diagonal, occupation_start_block
        diag = occupation_diagonal(diag, eps_c, eps_v, weights, sharding=diag_sh)
        if not paired:
            diag_low = occupation_diagonal(diag_low, low[4], low[5],
                                           data["partner"]["occupation_weights"], sharding=diag_sh)
    pre = bse_pair_diagonal_precond(
        eps_c, eps_v, epsilon_shift=float(davidson_eps_shift_Ry), sharding=diag_sh, diag_H=diag,
        low=None if paired else (low[4], low[5], diag_low), olsen=bool(davidson_olsen))

    # Margin roots and the start block on the physical transitions, as the
    # Tamm-Dancoff route builds them (Gaussian rows off the masked pairs when
    # the window carries occupation weights).
    n_phys = n_cond * n_val * nk
    n_solve = n_eig + max(0, min(int(davidson_margin), n_phys // 2 - n_eig))
    if weights is not None:
        X0 = occupation_start_block(eps_c, eps_v, weights, n_rows=n_solve, n_cond=n_cond,
                                    n_val=n_val, mesh=mesh_xy, sharding=sh.X)
    else:
        X0 = init_bse_subspace(eps_c, eps_v, n_eig=n_solve, n_cond=n_cond, n_val=n_val,
                               n_random=int(davidson_n_random_init), mesh=mesh_xy, sharding=sh.X)
    cap = basis_capacity(davidson_m_max, n_solve, n_eig, nc * nv * nk, paired=paired)
    stored = (2 if paired else 4) * cap
    log.info(f"full-BSE Davidson: {n_eig} roots + {n_solve - n_eig} margin"
             + ("" if davidson_margin_tol_mev is None else
                f" (locked at {float(davidson_margin_tol_mev):g} meV)")
             + f", basis capacity {cap} {'real-form' if paired else 'pair'} vectors, storing "
               f"{stored} trial vectors (basis and images); instability: {on_indefinite}"
             + ("" if paired else "; finite pair momentum: the unpaired form, K on [X; Y]"))
    _price_davidson_basis(stored // 2, X0, mesh_xy)
    tol_ry = float(davidson_tol_mev) / ry_mev
    if paired:
        apply = lambda payload, Z, s: pair(Z, s, *payload[0])                 # noqa: E731
        operands = args
    else:
        apply_K = kernel_matvec(pair)
        apply = lambda payload, Z: apply_K(payload[0], Z)                     # noqa: E731
        operands = (args, low)
    omega, X, Y, info = sp_davidson(
        apply, X0, paired=paired, precondition_vectors=bool(davidson_olsen),
        precondition=lambda payload, RX, RY, om, X=None, Y=None: pre.apply(
            payload[1], RX, RY, om, X, Y),
        data=(operands, pre.data), n_converge=n_eig, capacity=cap, max_iter=int(davidson_max_iter),
        tol=tol_ry, lock_converged=True,
        margin_lock=(1.0 if davidson_margin_tol_mev is None else
                     max(1.0, float(davidson_margin_tol_mev) / float(davidson_tol_mev))),
        on_indefinite=on_indefinite, verbose=False,      # the stop line below is this route's
        refusal_notes={
            "definite": "The energy unit here is Ry.  In bse_jax: full_bse_instability = continue "
                        "(solve_bse_nontda_sharded: on_indefinite='continue').  No eigenpairs "
                        "were returned.",
            "operator": "In the BSE: the restart's W (bse_nontda.check_restart_reciprocity on "
                        "W_q) or the pair applier is at fault; full_bse_solver = dense names a "
                        "stale restart and assembles A and B to compare with A = A^H, B = B^T."},
        plan_key=(None if davidson_plan_key is None else (
            davidson_plan_key, mesh_xy, bool(include_W), channel, route,
            float(davidson_eps_shift_Ry), bool(davidson_olsen), paired,
            data.get("head_tensor") is not None, weights is not None,
            tuple((tuple(a.shape), str(a.dtype)) for a in args))))

    line = refuse_unconverged_davidson(info, n_eig=n_eig, tol_mev=float(davidson_tol_mev),
                                       budget=int(davidson_max_iter))
    log.info(line)
    n_special = int(info["n_unstable"]) + int(info["n_negative"])
    if n_special + n_eig > n_solve:
        raise RuntimeError(
            f"GATE nontda_davidson_block: the block of {n_solve} roots holds {n_special} unstable "
            f"or non-positive root(s) and only {n_solve - n_special} of the {n_eig} requested "
            f"positive ones (got: davidson_margin = {n_solve - n_eig}; want: >= {n_special}).  "
            f"Why: the roots returned as excitons are the positive ones above the unstable "
            f"sector.  Fix: davidson_margin >= {n_special}.")
    omega = np.asarray(omega)
    root_sh = NamedSharding(mesh_xy, P(None, None, "x", "y", None))
    lo, hi = n_special, n_special + n_eig
    pairs = jax.jit(lambda x, y: jnp.stack([x[lo:hi], y[lo:hi]], axis=1),
                    out_shardings=root_sh)(X, Y)
    special = None
    if n_special:
        special = jax.jit(lambda x, y: jnp.stack([x[:n_special], y[:n_special]], axis=1),
                          out_shardings=root_sh)(X, Y)
        for j in range(n_special):
            w = complex(omega[j])
            kind = "unstable" if j < int(info["n_unstable"]) else "non-positive"
            log.warning(
                f"full BSE: {kind} root {j + 1}: omega = {w.real * ry_mev:+.6f} "
                f"{w.imag * ry_mev:+.6f} i meV (residual "
                f"{float(info['residuals'][j]) * ry_mev:.3e} meV); the reference state is not "
                f"a minimum (lam_min = {float(info['lam_min']) * ry_mev:.6g} meV)")
    LAST_DAVIDSON.clear()
    LAST_DAVIDSON.update(line=line, info=info, omega_ry=omega, n_special=n_special,
                         lam_min_ry=float(info["lam_min"]), special_vectors=special)
    return np.real(omega[lo:hi]), pairs, info


#: The options each route of ``solve_bse_nontda_sharded`` reads, with their
#: defaults.  An option set on a route that does not read it refuses.
_ROUTE_OPTIONS = {
    "dense": {},
    "davidson": dict(davidson_tol_mev=DAVIDSON_TOL_MEV, davidson_max_iter=200,
                     davidson_m_max=None, davidson_margin=DAVIDSON_MARGIN_ROOTS,
                     davidson_margin_tol_mev=None,
                     davidson_precond="exact", davidson_eps_shift_Ry=1e-3,
                     davidson_olsen=False,
                     davidson_n_random_init=5, on_indefinite="refuse",
                     davidson_plan_key=None),
}


def _route_options(solver, options):
    """``options`` checked against :data:`_ROUTE_OPTIONS` and completed with
    the route's defaults.  Refuses an unknown solver, an unknown option, and an
    option another route reads."""
    if solver not in _ROUTE_OPTIONS:
        raise ValueError(
            f"GATE nontda_route_option: solver is not a full-BSE route (got: {solver!r}; want: one "
            f"of {', '.join(repr(s) for s in _ROUTE_OPTIONS)}).  Why: the solver picks the route "
            f"that reads the options.  Fix: pass one of the named solvers.")
    owner = {key: route for route, opts in _ROUTE_OPTIONS.items() for key in opts}
    unknown = sorted(key for key in options if key not in owner)
    if unknown:
        raise ValueError(
            f"GATE nontda_route_option: unknown option(s) {unknown} (got: "
            f"{unknown}; want: one of n_eig, include_W, recip_tol, solver, "
            f"{', '.join(sorted(owner))}).  Why: an option this "
            f"route does not read would be dropped silently and the solve would run "
            f"with a setting you did not ask for.  Fix: correct the spelling, or "
            f"drop the option.")
    foreign = sorted(key for key in options if owner[key] != solver)
    if foreign:
        raise ValueError(
            f"GATE nontda_route_option: option(s) "
            f"{', '.join(f'{key} (read by solver={owner[key]!r})' for key in foreign)} "
            f"cannot be honoured by solver={solver!r} (got: {foreign}; want: only "
            f"{sorted(_ROUTE_OPTIONS[solver]) or 'no route option'}).  Why: the solve would "
            f"run without the setting you asked for.  Fix: name the solver that reads "
            f"{'it' if len(foreign) == 1 else 'them'}, or drop "
            f"{'it' if len(foreign) == 1 else 'them'}.")
    return {**_ROUTE_OPTIONS[solver], **options}


def solve_bse_nontda_sharded(data, mesh_xy, *, n_eig=5, include_W=True,
                             recip_tol=_NONTDA_RECIP_TOL, solver="dense", **options):
    """Non-TDA (full BSE) lowest-eigenvalue solve — one entry, dispatched on B.

    Returns ``(eigenvalues_Ry (n_eig,), eigenvectors (n_eig, 2, nc, nv, nk),
    n_iter)`` matching the TDA ``solve_bse_sharded`` tuple; the pair axis carries
    (X, Y) with ``X^H X - Y^H Y = +1``, and the eigenvectors sit on the transition axes
    of ``mesh_xy`` (the TDA routes' layout).  The real product route is
    taken only when H itself is real (``_h_is_real``); everything else,
    ``B = 0`` with a complex ``A`` included, takes the definite pencil.

    ``recip_tol`` is the restart preflight's threshold on
    ``max|W(q) - conj(W(-q))| / max|W|`` (:func:`check_restart_reciprocity`);
    ``None`` measures without refusing.

    ``solver`` selects the route; ``options`` are the selected route's
    (:data:`_ROUTE_OPTIONS` holds the defaults).  An option of another route
    refuses (GATE nontda_route_option), as does an unknown one.

    ``'dense'`` (default)
        Assemble ``A`` and ``B`` and diagonalise on the host.  ``O(N^2)``
        memory, ``2N`` half-applications, hard-walled at
        ``_DENSE_N_MAX``.  This is the ORACLE and it is not going away: it is
        what the other routes are gated against, and it is the only route
        that returns the complete spectrum or individual matrix elements.
        No options.

    ``'davidson'``
        The structure-preserving block Davidson (:mod:`solvers.bse_sp_davidson`)
        over the same applier, preconditioned by the paired exact diagonal
        ``[D; -D]`` and started on the lowest physical transitions
        (``bse_davidson_helpers``).  No assembly and no size limit; two pair
        calls per open root per iteration; every vector stays on the mesh.
        ``bse_jax``: ``full_bse_solver = davidson``.  Options:
        ``davidson_tol_mev`` (0.01; every root's residual
        ``||H z - omega z|| / ||z||`` below it, else GATE
        nontda_davidson_converged), ``davidson_max_iter`` (200),
        ``davidson_m_max`` (basis capacity in real-form vectors; default
        ``20 n_eig``, storage twice that), ``davidson_margin`` (8 roots solved
        above ``n_eig`` and not returned), ``davidson_margin_tol_mev`` (their
        lock; default the tolerance), ``davidson_precond`` (``exact``, ``bare``,
        ``auto``), ``davidson_eps_shift_Ry`` (1e-3), ``davidson_olsen`` (False;
        the Olsen correction in the metric ``Sigma``), ``davidson_n_random_init``
        (5), ``davidson_plan_key`` (reuse the compiled kernels across
        same-shape payloads) and ``on_indefinite``: ``'refuse'`` (default)
        raises GATE sp_davidson_definite with the measured ``lam_min`` when
        ``K = [[A, B], [B*, A*]]`` is found not positive definite;
        ``'continue'`` solves on, logs the unstable root(s) and returns the
        ``n_eig`` positive roots above them (:data:`LAST_DAVIDSON` keeps the
        unstable ones and their vectors)."""
    opts = _route_options(solver, options)
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
    w_q0 = None
    if solver == "davidson" and include_W and opts["davidson_precond"] != "bare":
        # Before the transform below consumes W_q: the exact diagonal's slice.
        from .bse_davidson_helpers import w_q0_for_precond
        w_q0 = w_q0_for_precond(data)
    pair, args, low = _operator(data, mesh_xy, sh, include_W=include_W)
    N = nc * nv * nk
    if solver == "davidson":
        from common.collectives import replicate_to_mesh
        omega, pairs, info = _solve_nontda_davidson(
            pair, args, data, mesh_xy, sh, n_eig, include_W=include_W, w_q0=w_q0, low=low, **opts)
        return (replicate_to_mesh(np.asarray(omega), mesh_xy), pairs,
                jnp.int32(int(info["iterations"])))
    if low is not None:
        A, B, B_low, A_low = _materialize_K(pair, args, low, sh, nc, nv, nk)
        omega, Z = solve_nontda_definite_pencil(A, B, n_eig, lower=(B_low, A_low))
        evecs = np.stack([Z[:N, :].T.reshape(n_eig, nc, nv, nk),
                          Z[N:, :].T.reshape(n_eig, nc, nv, nk)], axis=1)
        return jnp.asarray(omega), _on_mesh(evecs, mesh_xy), jnp.int32(0)
    A, B = _materialize_A_B(pair, args, sh, nc, nv, nk)
    if _h_is_real(A, B):
        omega, Z = solve_nontda_product(A, B, n_eig)           # real / RPA
    else:
        omega, Z = solve_nontda_definite_pencil(A, B, n_eig)   # optical spinor
    evecs = np.stack([Z[:N, :].T.reshape(n_eig, nc, nv, nk),
                      Z[N:, :].T.reshape(n_eig, nc, nv, nk)], axis=1)
    return jnp.asarray(omega), _on_mesh(evecs, mesh_xy), jnp.int32(0)


def _on_mesh(evecs, mesh_xy):
    """The host ``(n_eig, 2, nc, nv, nk)`` pairs on the solve mesh's transition
    axes, ``P(None, None, 'x', 'y', None)``: the layout Davidson returns its
    roots on (``bse_lanczos``), so the dipole contraction
    (``absorption_common.exciton_dipoles_distributed``, which shards the dipole
    block like the eigenvectors and contracts each rank's own tile) and the
    eigenvector writer read both routes through one path.  Every process holds
    the same host array (the dense solve runs on every process) and puts its
    own tiles on its devices: no collective."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    return device_put_process_local(np.asarray(evecs, dtype=np.complex128),
                                    NamedSharding(mesh_xy, P(None, None, "x", "y", None)))
