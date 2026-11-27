"""Shared operands and the shifted GMRES resolvent solve of the sharded ISDF-BSE.

What the BSE drivers (``bse_kpm``, ``bse_w_exact``, ``w_ladder``,
``head_resolvent``, ``bse_davidson_helpers``, ``tdagw.model``) import from here:

* the operands of the ring/stack matvec: :func:`ensure_W_R` (reciprocal ``W_q`` to
  real-space ``W_R``), :func:`matvec_operands`, :func:`ladder_matvec_operands`,
  :func:`_apply_shifted_matvec` (``(z - H) x``);
* the GMRES resolvent solve ``(z - H)^{-1} b``: :func:`_gmres_solve_core`
  (diagonally right-preconditioned, while-loop exit), with the rank-k spectral
  deflation (:func:`harvest_spectral_deflation`, :func:`apply_spectral_deflation`)
  and the diagonal preconditioner :func:`build_preconditioner_diagonal_sharded`;
* :func:`estimate_spectral_bounds_sharded`: ``E_min`` and ``E_max`` of the
  operator from a short adaptive Lanczos run.

The module was ``bse_feast`` until 2026-11-27: the FEAST contour eigensolver
(``feast = true``: GMRES solves at quadrature nodes plus Rayleigh-Ritz) and
its ``python -m bse.bse_feast`` entry point were retired then; ``full_bse =
true`` with ``full_bse_solver = davidson`` solves the full BSE.
"""
from __future__ import annotations

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap()
# (env defaults + jax.distributed init + CPU fallback; all idempotent).
# MUST run before this module's own `import jax`.
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import lstsq_hessenberg_local
from .bse_ring_comm import make_bse_shardings
from .bse_stack_matvec import build_bse_stack_matvec
from .bse_preconditioner import energy_diff_cv_k
from .bse_head import head_shifted_eps_c
from .bse_preconditioner import compute_pair_amplitude
from .bse_davidson_helpers import build_bse_exact_diagonal
from common.units import RYD_TO_EV
from .bse_io import make_w_densifier
from .bse_window import restrict_to_physical_transitions

jax.config.update("jax_enable_x64", True)


#: Ry -> eV, from ``common.units``.  Name kept: ``bse_w_exact`` imports
#: ``RY_TO_EV_DEFAULT`` from here.
RY_TO_EV_DEFAULT = RYD_TO_EV


def ensure_W_R(data: dict, include_W: bool, mesh_xy: Mesh, *, consume_W_q=False) -> dict:
    """Populate ``data['W_R']`` — the 8th (real-space W) argument the ring/stack
    matvecs and the shifted-matvec chain (``_apply_shifted_matvec``) expect.

    The screened-direct term convolves the ISDF W-tile in real space, so the
    reciprocal-space ``W_q`` (μ, ν, kx, ky, kz) is inverse-FFT'd on the k-axes
    through :func:`bse_io.make_w_densifier` — the ONE sharded densifier
    (degenerate ``fine_grid == coarse`` = plain sharded ifftn), so the (μ,ν)
    sharding survives and no rank gathers an N_μ²-class tile (the eager
    ``local_ifftn3`` form this replaces was audit-P0-4-class debt, gated by
    ``tools/lint/test_fft_shardmap_context.py``).
    For the RPA density-response / kernel (``include_W=False``) the W-term is
    never built, so ``W_R`` is only a shape/sharding-correct placeholder and
    ``W_q`` itself serves.  The restart loader emits ``W_q`` but not ``W_R``, so
    every solve entry point (spectral-bound Lanczos, ``bse_w_exact``)
    must call this before invoking a matvec. Mutates and returns ``data``.

    ``consume_W_q=True`` is for a caller that owns a freshly loaded payload:
    retain only its q=0 preconditioner slice and donate W_q to the shared
    optical-solver transform. W_q is then None and W_R is reusable. The
    default borrows W_q, preserving diagnostic callers and their snapshots.
    """
    if data.get("W_q") is None:
        if data.get("W_R") is None:
            raise ValueError("ensure_W_R needs W_q or an already prepared W_R")
        return data
    if include_W:
        W_q = data["W_q"]
        if consume_W_q:
            from common.fft_helpers import get_donated_kfft_kminor
            from common.gpu_utils import shared_device_room
            from runtime.aot_memory import compiled_new_bytes
            from lxkit import memory
            if data.get("_W_q0_for_precond") is None:
                data["_W_q0_for_precond"] = W_q[:, :, 0, 0, 0]
            jax.block_until_ready((W_q, data["_W_q0_for_precond"]))
            transform = get_donated_kfft_kminor(
                mesh_xy, tuple(int(s) for s in W_q.shape[-3:]),
                make_bse_shardings(mesh_xy).W.spec, kind="ifftn", norm="ortho")
            executable = transform.lower(W_q).compile()
            stage = "BSE owned W_q to W_R"
            price = memory.priced(compiled_new_bytes(executable),
                                  "donated W transform: new buffers and provider scratch")
            room = shared_device_room(stage)
            memory.book(price, stage="bse.W_R")
            memory.admit(stage, price, int(room) if isinstance(room, (int, np.integer)) else 0)
            data["W_R"] = executable(W_q)
            data["W_q"] = None
            return data
        densify = make_w_densifier(
            mesh_xy, make_bse_shardings(mesh_xy).W.spec,
            tuple(int(s) for s in W_q.shape[-3:]), output="R")
        data["W_R"] = densify(W_q)
    else:
        data["W_R"] = data["W_q"]
    return data


def matvec_operands(data: dict, *, include_W: bool = True) -> tuple:
    """The 9 operand arrays the ring/stack matvec consumes after ``x``, in the
    matvec's positional order ``(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v,
    W_R, V_q0, M)``.

    Pulling these out of ``data`` and threading them as RUNTIME ARGUMENTS through
    the jitted solve (instead of closing over the ``data`` dict) is what lets ONE
    compiled shifted-solve engine serve every q / omega: the operator (``matvec``)
    fixes the compiled STRUCTURE, the operands are just values that change per q, so
    a finite-q loop reuses the same executable instead of recompiling once per q
    (see ``bse_w_exact`` compare-wq)."""
    return (
        data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
        (head_shifted_eps_c(data["eps_c"], data) if include_W else data["eps_c"]),
        data["eps_v"], data["W_R"],
        data["V_q0"],
        data["M"],  # hoisted V-term pair amplitude (audit P3), transition layout
    )


def ladder_matvec_operands(data: dict, *, include_W: bool = True) -> tuple:
    """Physical R/A array bundle, or the historical raw operand tuple.

    The physical bundle carries two vertices without expanding the call
    signature. Legacy rung slots are the ordinary 9 operands plus
    the rung's four PHYSICAL (rolled, un-flipped) psi arrays.  Falls back to
    the density arrays when the payload carries no rung slots — that is a raw
    payload, where the density arrays ARE physical (the q=0 identity cells)."""
    if data.get("physical_branches", False):
        keys = ("psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y",
                "psi_c_A_X", "psi_c_A_Y", "psi_v_A_X", "psi_v_A_Y",
                "eps_c", "eps_v", "eps_c_A", "eps_v_A", "M", "M_A", "W_R", "V_q0")
        operands = {key: data[key] for key in keys}
        # The internal transfer Gamma head is the exact optical diagonal:
        # -h in RR and +h in AA, with no c/v-overlap coupling contribution.
        # Keep stored DFT energies unchanged for dipole normalization.
        for key in ("eps_c", "eps_c_A"):
            operands[key] = head_shifted_eps_c(data[key], data) if include_W else data[key]
        return (operands,)
    return matvec_operands(data, include_W=include_W) + (
        data.get("psi_c_W_X", data["psi_c_X"]),
        data.get("psi_c_W_Y", data["psi_c_Y"]),
        data.get("psi_v_W_X", data["psi_v_X"]),
        data.get("psi_v_W_Y", data["psi_v_Y"]),
    )


def _apply_shifted_matvec(
    matvec,
    x: jax.Array,
    z: complex,
    operands: tuple,
) -> jax.Array:
    """``(z I - H) x`` for the shifted linear solve.  ``operands`` is the
    :func:`matvec_operands` 10-tuple (runtime args, NOT a closed-over dict)."""
    return z * x - matvec(x, *operands)


def build_preconditioner_diagonal_sharded(
    data: dict,
    mesh_xy: Mesh,
    include_W: bool = True,
    use_tda: bool = True,
    *, exchange_channel: str | None = None,
) -> jax.Array:
    """``diag(H_BSE)`` for the shifted GMRES preconditioner, from the payload.

    THIS IS AN ADAPTER, NOT A SECOND IMPLEMENTATION.  It used to be one: eight
    un-jitted einsums that assembled the same object as
    :func:`bse_davidson_helpers.build_bse_exact_diagonal`, arrived independently
    at the same ``+1/nk`` exchange / ``−1/nk`` direct normalisation, and paid
    eight separate program constructions to do it (PRECOND_BUILD_FREE.md §7.1).
    Measured against the canonical builder on the Si 4x4x4 record deck at P=4
    BEFORE the two were merged, the real parts agreed to
    ``max|Δ| = 1.11e-16 Ry = 1.5e-15 eV`` on a 0.62 Ry signal — floating-point
    associativity, the same 1.5e-15 eV the canonical builder already scores
    against the dense operator — so neither was wrong and there was nothing to
    choose between them (FIX_construction_defects.md §2).

    What the adapter keeps of the FEAST-era version, deliberately:

    * **the complex diagonal.**  The shifted GMRES solves divide by ``z − diag``
      at a complex shift, so they want the operator's antihermitian residue
      (``max|Im| = 1.13e-14 Ry`` here); Davidson takes ``real()``.  Carried by
      ``complex_out=True``, not decided here;
    * **the non-TDA stack** ``[diag, −diag]``, which is this route's shape and
      has no Davidson counterpart;
    * **``include_W=False``**, the RPA density-response route, which now drops
      the direct contraction at trace time instead of adding a zero tile.

    What it stops doing: rebuilding ``M`` from ``psi`` locally.
    Audit P3 hoisted those onto the payload precisely so nothing would, and
    ``build_finite_q_data`` maintains them per q; the two forms are the same
    ``compute_pair_amplitude`` call, which is why the agreement above is at
    round-off.
    """
    if exchange_channel is None:
        exchange_channel = data.get("exchange_channel", "singlet")
    if data.get("physical_branches", False) and not use_tda:
        anti = dict(data, M=data["M_A"], eps_c=data["eps_c_A"], eps_v=data["eps_v_A"])
        for band in ("c", "v"):
            for axis in ("X", "Y"):
                anti[f"psi_{band}_{axis}"] = jnp.conj(data[f"psi_{band}_A_{axis}"])
        dr = build_preconditioner_diagonal_sharded(data, mesh_xy, include_W, use_tda=True,
                                                  exchange_channel=exchange_channel)
        da = build_preconditioner_diagonal_sharded(anti, mesh_xy, include_W, use_tda=True,
                                                  exchange_channel=exchange_channel)
        # Swapping the physical m/n legs conjugates their Hermitian-W
        # diagonal; the ring self-contraction and excitation energy are real.
        return jax.lax.with_sharding_constraint(jnp.stack((dr, -jnp.conj(da)))[:, None],
            NamedSharding(mesh_xy, P(None, None, "x", "y", None)))
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    # Hoisted pair amplitudes (audit P3).  The fallback is for payloads
    # assembled outside the loader; it is the same contraction the loader runs.
    M = data.get("M")
    if M is None:
        M = compute_pair_amplitude(data["psi_c_X"], data["psi_v_X"])

    W_q0 = None
    if include_W:
        # Prefer the slice the driver stashes before its donated ifft consumes
        # W_q; fall back to slicing W_q here for payloads that never went
        # through that path.  Deriving it from W_R instead would make the answer
        # depend on the ifft's norm convention.
        W_q0 = data.get("_W_q0_for_precond")
        if W_q0 is None:
            W_q = data.get("W_q")
            if W_q is None:
                raise ValueError(
                    "build_preconditioner_diagonal_sharded(include_W=True) needs "
                    "the q=0 slice of W_q, but this payload has neither "
                    "'_W_q0_for_precond' nor 'W_q'.  Reload the payload.")
            W_q0 = W_q[:, :, 0, 0, 0]

    tda_sharding = NamedSharding(mesh_xy, P("x", "y", None))
    diag_h = build_bse_exact_diagonal(
        (head_shifted_eps_c(data["eps_c"], data) if include_W else data["eps_c"]), data["eps_v"],
        data["psi_c_X"], data["psi_v_Y"],
        W_q0, M, data["V_q0"], nk,
        sharding=tda_sharding, complex_out=True, memo=False,
        exchange_channel=exchange_channel,
    )
    if use_tda:
        return diag_h

    # The ladder's antiresonant row carries the conjugate diagonal:
    # H_AA = -conj(H_RR).  This distinction is invisible when ``diag_h`` is
    # real, but finite-q / band-edge truncation can leave a small legitimate
    # anti-Hermitian residue in the exact diagonal.  Keep that residue paired
    # with the operator rather than preconditioning the bottom row with the
    # resonant convention.
    diag_full = jnp.stack([diag_h, -jnp.conj(diag_h)], axis=0)[:, None, ...]
    diag_sharding = NamedSharding(mesh_xy, P(None, None, "x", "y", None))
    return jax.lax.with_sharding_constraint(diag_full, diag_sharding)


def apply_spectral_deflation(v: jax.Array, deflation) -> jax.Array:
    """``P v = v + U (T^{-1} - I) U^H v`` — the rank-k spectral preconditioner.

    ``deflation = (U, C)``: ``U`` is ``(k,) + v.shape``, orthonormal, spanning an
    approximate invariant subspace of the right-preconditioned operator
    ``A = (z - H) diag(z - D)^{-1}``, and ``C = T^{-1} - I`` with
    ``T = U^H A U``.  On an exact invariant subspace ``A P`` maps those
    eigenvalues to 1 and leaves the rest of the spectrum alone, so GMRES stops
    paying iterations for them.  ``P`` is a fixed preconditioner: any ``U``
    with ``T`` invertible gives the SAME solution to tolerance, and a poor
    ``U`` only costs iterations.  The coefficient ``U^H v`` is one
    ``(k,)`` all-reduce; the update is local.
    """
    U, C = deflation
    axes = tuple(range(v.ndim))
    c = jnp.tensordot(jnp.conj(U), v, axes=(tuple(a + 1 for a in axes), axes))
    return v + jnp.tensordot(C @ c, U, axes=(0, 0)).astype(v.dtype)


#: ``(id(matvec), n_arnoldi, shape, dtype) -> (matvec, jitted Arnoldi)``.
_ARNOLDI_CACHE: dict = {}


def _get_deflation_arnoldi(matvec, seed, *, n_arnoldi: int):
    """The shared cached harvest executable, exposed for compile-only pricing."""
    m = int(n_arnoldi)
    key = (id(matvec), m, tuple(seed.shape), str(seed.dtype))
    hit = _ARNOLDI_CACHE.get(key)
    if hit is None:
        axes = tuple(range(1, 1 + seed.ndim))

        @jax.jit
        def _arnoldi(seed_, diag_, z_, ops_):
            m_inv = 1.0 / (z_ - diag_)
            if m_inv.ndim == seed_.ndim - 1:
                m_inv = m_inv[None, ...]
            V = jnp.zeros((m + 1,) + seed_.shape, seed_.dtype)
            V = V.at[0].set(seed_ / jnp.linalg.norm(seed_))
            H = jnp.zeros((m + 1, m), seed_.dtype)

            def body(k, state):
                V, H = state
                w = _apply_shifted_matvec(matvec, m_inv * V[k], z_, ops_)
                w = w.astype(seed_.dtype)
                h = jnp.zeros((m + 1,), seed_.dtype)
                for _ in range(2):
                    hp = jnp.sum(jnp.conj(V) * w[None], axis=axes)
                    w = w - jnp.tensordot(hp, V, axes=(0, 0))
                    h = h + hp
                hn = jnp.linalg.norm(w)
                H = H.at[:, k].set(h.at[k + 1].set(hn))
                V = V.at[k + 1].set(jnp.where(hn == 0, w, w / hn))
                return V, H

            return jax.lax.fori_loop(0, m, body, (V, H))

        hit = (matvec, _arnoldi)
        _ARNOLDI_CACHE[key] = hit
    return hit[1]


def harvest_spectral_deflation(matvec, diag_h, z, operands, seed, *,
                               n_arnoldi: int, rank: int):
    """Recycled eigenvectors of the preconditioned resolvent, for every column.

    Every probe column of one ``(q, z)`` solves against the SAME operator
    ``A = (z - H) diag(z - D)^{-1}``, and on the ladder the columns all pay
    for the same few eigenvalues far from 1 (bound excitons below the
    free-pair continuum).  One ``n_arnoldi``-step Arnoldi from ``seed`` (a
    ``(2, 1, c, v, k)`` pair-basis vector) finds them once; the ``rank``
    Ritz vectors farthest from 1 become ``U`` of
    :func:`apply_spectral_deflation`, and ``T = Q^H H_m Q`` comes from the
    Arnoldi relation ``A V_m = V_{m+1} H`` with no further matvec.

    Returns ``((U, C), ritz)``: ``ritz`` are the Ritz values of ``H_m`` on the
    host (for the log).  Cost: ``n_arnoldi`` matvecs once per ``(q, z)``.
    """
    m = int(n_arnoldi)
    program = _get_deflation_arnoldi(matvec, seed, n_arnoldi=m)
    V, H = program(seed, diag_h, jnp.asarray(z, dtype=jnp.complex128), operands)
    from common.collectives import gather_to_host
    Hm = np.asarray(gather_to_host(H))[:m, :m]
    ritz, S = np.linalg.eig(Hm)
    order = np.argsort(np.abs(ritz - 1.0))[::-1][:int(rank)]
    Qk, _ = np.linalg.qr(S[:, order])
    T = Qk.conj().T @ Hm @ Qk
    C = np.linalg.inv(T) - np.eye(Qk.shape[1])
    # Host factors enter as numpy (replicated constants on every process),
    # never as a process-local jax.Array mixed into a global computation.
    U = jnp.tensordot(np.ascontiguousarray(Qk.T, dtype=np.complex128), V[:m],
                      axes=(1, 0))
    return (U, np.ascontiguousarray(C, dtype=np.complex128)), ritz


def _gmres_solve_core(matvec, b, diag_h, z, operands, max_iter, tol, *,
                      x0=None, resid_relative_to: str = "b", deflation=None):
    """Diagonally right-preconditioned GMRES with a while-loop early exit.

    Pure function of its RUNTIME args ``(b, diag_h, z, operands, tol, x0)``;
    ``matvec`` / ``max_iter`` / ``resid_relative_to`` are the static structure.
    ``operands`` is the :func:`matvec_operands` 10-tuple — threading it as an
    argument (not a closed-over ``data`` dict) is why one compiled solve serves
    every q / omega (see :func:`_get_gmres_solver`).  ``tol`` may be a traced
    scalar, so a caller may sweep it without recompiling.  Returns
    ``(x, k_iters)``.

    ``x0`` — the initial guess, RUNTIME.  ``None`` keeps the historical
    Jacobi-shaped start ``M^{-1} b``.  That start is not always an improvement
    on zero: on the ladder screening operator it was MEASURED to make
    ``||r0||`` 8-11x ``||b||`` (opt_shifts, 2026-08-16), i.e. the "initial
    guess" is an order of magnitude WORSE than starting from nothing, so a
    caller that wants ``x0 = 0`` can now ask for it.

    ``deflation`` — ``None`` (the diagonal preconditioner alone, the
    historical program) or the ``(U, C)`` pair of
    :func:`harvest_spectral_deflation`, applied inside the same right
    preconditioner by :func:`apply_spectral_deflation`.  RUNTIME: one compiled
    program per pytree shape, and every column of a ``(q, z)`` reuses it.

    ``resid_relative_to`` — the denominator of the early-exit test.

    * ``"b"`` (DEFAULT since 2026-08-16): exit on ``||r_k|| <= tol * ||b||``,
      which is what every caller in this tree writes down when it passes a
      ``tol``, and what the true-residual gates then measure.
    * ``"r0"``: the historical behaviour — ``||r_k|| <= tol * ||r_0||`` with
      ``r_0`` the residual of the PRECONDITIONED start above.  Because that
      start is the bad one, callers were delivered a residual ~10x LOOSER than
      the number they passed (measured four independent times on the ladder
      operator: nominal 1e-9, delivered true 1.08e-8).  Kept only so the change
      can be A/B'd; it is a defect, not a mode.

    THE LEAST SQUARES IS SOLVED ONCE, NOT PER ITERATION.  The early-exit test
    needs only the NORM of the GMRES residual, which the Givens rotations that
    triangularize the Hessenberg deliver for free (``|g_rot[k+1]|``) at ``O(k)``
    per iteration.  The historical spelling re-solved a dense
    ``(max_iter+1, max_iter)`` ``lstsq`` — an SVD whose cost is set by the CAP
    and not by the iterations actually taken — at EVERY iteration, measured at
    2.0 ms of a 15.6 ms column-iteration at ``max_iter=300``
    (opt_precond/RESULTS.md §9). The final ``y`` is the minimum-norm solve
    of the final ``H`` and ``g``. ROCm uses a bounded active prefix with the
    FULL carrier's singular-value cutoff; another SVD shape can change
    rounding, so the true-residual gate remains authoritative. The rotations
    are the standard stable ones, never the normal equations.
    """
    if resid_relative_to not in ("b", "r0"):
        raise ValueError(
            f"resid_relative_to must be 'b' or 'r0', got {resid_relative_to!r}")
    one = jnp.asarray(1.0, dtype=b.dtype)
    m_inv = one / (z - diag_h)
    if m_inv.ndim == b.ndim - 1:
        m_inv = m_inv[None, ...]

    x0 = m_inv * b if x0 is None else x0.astype(b.dtype)
    r0 = b - _apply_shifted_matvec(matvec, x0, z, operands).astype(b.dtype)
    beta = jnp.linalg.norm(r0)
    den = jnp.linalg.norm(b) if resid_relative_to == "b" else beta

    zero = jnp.asarray(0.0, dtype=beta.dtype)
    v0 = jnp.where(beta == zero, r0, r0 / beta)

    v_shape = (max_iter + 1,) + b.shape
    z_shape = (max_iter,) + b.shape
    V = jnp.zeros(v_shape, dtype=b.dtype).at[0].set(v0)
    Z = jnp.zeros(z_shape, dtype=b.dtype)
    H = jnp.zeros((max_iter + 1, max_iter), dtype=b.dtype)
    g = jnp.zeros((max_iter + 1,), dtype=b.dtype).at[0].set(beta)
    # ACCUMULATED Givens rotation.  Q is the product of every rotation applied
    # so far, so the residual recurrence needs no replay loop: the rotated
    # right-hand side is Q g = beta Q[:, 0] because g = beta e_0.  One
    # (max_iter+1)^2 replicated matrix — 0.65 MiB at max_iter=200, the same
    # class as the Hessenberg beside it, and it buys the removal of a
    # data-dependent inner loop (see the body).
    Q = jnp.eye(max_iter + 1, dtype=b.dtype)
    krylov_axes = tuple(range(1, 1 + b.ndim))

    def cond(state):
        k, rel, *_ = state
        return jnp.logical_and(k < max_iter, rel > tol)

    def body(state):
        k, rel, V, Z, H, g, Q = state

        v_k = V[k]
        # ``deflation`` (runtime, or None = the historical diagonal-only
        # program): the rank-k spectral correction rides the SAME right
        # preconditioner, so ``Z`` stores it and the exit ``x0 + Z y`` is
        # unchanged.  See :func:`apply_spectral_deflation`.
        z_k = m_inv * (v_k if deflation is None
                       else apply_spectral_deflation(v_k, deflation))
        Z = Z.at[k].set(z_k)
        w = _apply_shifted_matvec(matvec, z_k, z, operands).astype(b.dtype)

        # ORTHOGONALIZE WITH TWO CLASSICAL PASSES (CGS2), NOT A LOOP.
        #
        # Numerics first, because this replaced a modified Gram-Schmidt pass
        # plus a DGKS repeat and that pair was there for a measured reason:
        # stiff finite-q screening operators (bse_w_exact --compare-wq: V_q
        # carries a large G=0 Coulomb head) lose Arnoldi orthogonality
        # catastrophically under a SINGLE pass (||VᴴV−I|| → O(1) by ~20
        # iters), which makes the projected residual falsely tiny and the
        # whole solve rounding-dependent (converges in numpy, DIVERGES in
        # jit — true residual O(1)).  CGS2 — classical Gram-Schmidt applied
        # twice — is unconditionally stable to O(eps) (Giraud, Langou,
        # Rozloznik 2005), i.e. it is not a weakening of MGS+MGS but the
        # standard strengthening of it; the SECOND pass is what both spellings
        # rely on and it is kept.
        #
        # Cost, which is why the shape changed.  A dynamic-trip inner loop on
        # XLA:GPU reads its predicate back to the HOST every step: the two
        # loops here were ~2k device→host round trips per GMRES iteration, and
        # O9's kernel profile (2026-08-16) measured ~19 of them per iteration
        # with 61.6 % of the GPU idle waiting.  Each pass is now ONE reduction
        # over the whole V block.  The rows of V above k are identically zero
        # — nothing has written them yet — so their coefficients come out exact
        # zeros and no mask is needed; the arithmetic is over max_iter+1 rows
        # instead of k+1, which is a few percent of one rung matvec.
        #
        # SHARDING: the reduction is over the pair-basis axes (c on x, v on y),
        # so it emits ONE all-reduce of a (max_iter+1,) vector per pass, where
        # the loop emitted k+1 all-reduces of a scalar each — fewer AND
        # smaller collectives.  The back-projection is local (h is replicated).
        def _project(w_local):
            h_local = jnp.sum(jnp.conj(V) * w_local, axis=krylov_axes)
            return w_local - jnp.tensordot(h_local, V, axes=(0, 0)), h_local

        w, h_first = _project(w)
        w, h_corr = _project(w)
        H = H.at[:, k].set(h_first + h_corr)

        h_next = jnp.linalg.norm(w)
        H = H.at[k + 1, k].set(h_next)
        v_next = jnp.where(h_next == 0.0, w, w / h_next)
        V = V.at[k + 1].set(v_next)

        # --- residual norm by Givens, LOOP-FREE --------------------------
        # Applying the rotations already accumulated is ONE gemv against Q
        # (the running product of them), and adding the new rotation is a
        # two-row combination of Q.  The replay loop this replaces was the
        # third data-dependent inner loop in the body, i.e. k more host round
        # trips per iteration.  Unitary pair [[c, s], [−conj(s), conj(c)]]
        # with c = conj(a)/r, s = conj(bb)/r, r = hypot(|a|, |bb|) —
        # real-positive pivot, no branch on a == 0.  g = beta e_0, so the
        # rotated right-hand side is beta Q[:, 0] and the GMRES residual norm
        # is its trailing entry.
        hcol = Q @ H[:, k]
        a, bb = hcol[k], hcol[k + 1]
        r = jnp.sqrt(jnp.abs(a) ** 2 + jnp.abs(bb) ** 2).astype(b.dtype)
        safe = jnp.where(r == 0, jnp.ones_like(r), r)
        c_k = jnp.where(r == 0, jnp.ones_like(r), jnp.conj(a) / safe)
        s_k = jnp.where(r == 0, jnp.zeros_like(r), jnp.conj(bb) / safe)
        q_k, q_k1 = Q[k], Q[k + 1]
        Q = Q.at[k].set(c_k * q_k + s_k * q_k1).at[k + 1].set(
            -jnp.conj(s_k) * q_k + jnp.conj(c_k) * q_k1)
        resid = beta * jnp.abs(Q[k + 1, 0])
        rel = jnp.where(den == zero, zero, resid / den)

        return k + 1, rel, V, Z, H, g, Q

    rel0 = jnp.asarray(jnp.inf, dtype=beta.dtype)
    init = (0, rel0, V, Z, H, g, Q)
    k_final, _, V, Z, H, g, Q = jax.lax.while_loop(cond, body, init)

    # GMRES minimization min_y ‖g − H y‖ via a STABLE least-squares
    # (QR/SVD through ``lstsq``), NOT the normal equations HᴴH — those
    # square the Hessenberg condition number.  Finite-q screening tiles
    # (bse_w_exact --compare-wq) carry a large G=0 Coulomb head, so
    # cond(H) ~ 1e8 ⇒ cond(HᴴH) ~ 1e17 ≈ 1/eps, at which the normal-eq
    # solve returns a garbage y that trips the projected early-exit with a
    # huge TRUE residual (GMRES "converges" to a wrong x).  ``lstsq`` stays
    # accurate at that conditioning; head-less/q=0 tiles are well
    # conditioned so the q=0 closure is unchanged.  The trailing zero
    # columns of the padded Hessenberg get min-norm y=0. ROCm trims only
    # exact-zero padding through a static prefix bucket, retaining the full
    # carrier's eps cutoff. Other legs keep the original full-size solve.
    y = lstsq_hessenberg_local(H, g, k_final)

    x = x0 + jnp.tensordot(y, Z, axes=(0, 0))
    return x, k_final


def estimate_spectral_bounds_sharded(
    data: dict,
    mesh_xy: Mesh,
    n_lanczos: int = 20,
    n_lanczos_max: int | None = None,
    emax_rtol: float = 5e-4,
    emax_atol: float = 0.0,
    emax_patience: int = 2,
    seed: int = 0,
    include_W: bool = True,
    use_tda: bool = True,
) -> dict:
    """Estimate E_min and E_max (Ry) using diagonal gap and short Lanczos.

    Uses an adaptive Lanczos run that stops when the largest Ritz value
    (of the tridiagonal) converges to within the specified tolerances.

    In complex128, on the payload's own operands (its ``W_R`` when it has one):
    the W term's fused outer-product kernel serves complex128 only, so the
    complex64 copy this used to build refused on the GPU route
    (``kconv_outer ... operands must all be complex128``), and it held a second
    W in each precision besides.
    """
    data_b = data if data.get("W_R") is not None else ensure_W_R(dict(data), include_W, mesh_xy)

    eps_c = head_shifted_eps_c(data_b["eps_c"], data_b)
    eps_v = data_b["eps_v"]
    n_cond = int(data["n_cond"])
    n_val = int(data["n_val"])

    eps_c_use = eps_c[:, :n_cond]
    eps_v_use = eps_v[:, :n_val]
    eps_c_min = jnp.min(eps_c_use)
    eps_v_max = jnp.max(eps_v_use)
    e_min_ry = (eps_c_min - eps_v_max).real

    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n_cond_pad = int(data["n_cond_pad"])
    n_val_pad = int(data["n_val_pad"])

    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_stack_matvec(
        mesh_xy,
        data["nkx"],
        data["nky"],
        data["nkz"],
        kernel="bse" if include_W else "rpa",
        exchange_channel=data.get("exchange_channel", "singlet"),
    )

    key = jax.random.PRNGKey(seed)

    # The Lanczos start vector must have NO support on the padded block.
    # e_min above is already taken on the logical slices; e_max comes from
    # this Krylov space, and the pad block is decoupled with a diagonal at
    # ``PAD_EPS_GUARD_RY`` (~1e3 Ry) — an unmasked start would return that
    # as e_max and blow the Chebyshev half-width every KPM consumer
    # of this function scales by.  The mask is BY COUNT, from the logical
    # extents the loader put in the bundle.
    @jax.jit
    def _make_random_vector(key_in):
        k1, k2 = jax.random.split(key_in)
        q = jax.random.normal(k1, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
        q = q + 1j * jax.random.normal(k2, (1, n_cond_pad, n_val_pad, nk), dtype=jnp.float64)
        q = restrict_to_physical_transitions(q, n_cond, n_val)
        q = q / jnp.linalg.norm(q)
        return jax.lax.with_sharding_constraint(q, sh.X)

    def _tridiag_emax(alpha_seq: list[float], beta_seq: list[float]) -> float:
        t_dim = len(alpha_seq)
        T = np.zeros((t_dim, t_dim), dtype=np.float64)
        for i, alpha in enumerate(alpha_seq):
            T[i, i] = alpha
            if i < t_dim - 1:
                T[i, i + 1] = beta_seq[i]
                T[i + 1, i] = beta_seq[i]
        evals = np.linalg.eigvalsh(T)
        return float(np.max(evals))

    if n_lanczos_max is None:
        n_lanczos_max = n_lanczos

    q = _make_random_vector(key)
    q_prev = jnp.zeros_like(q)
    beta_prev = 0.0

    alphas: list[float] = []
    betas: list[float] = []
    emax_seq: list[float] = []
    converged = False
    stable = 0
    prev_emax: float | None = None

    for _ in range(n_lanczos_max):
        z = matvec(
            q,
            data_b["psi_c_X"],
            data_b["psi_c_Y"],
            data_b["psi_v_X"],
            data_b["psi_v_Y"],
            eps_c,
            eps_v,
            data_b["W_R"],
            data_b["V_q0"],
            data_b["M"],  # hoisted V-term pair amplitude (audit P3)
        )

        alpha = jnp.vdot(q, z).real
        z = z - alpha * q - beta_prev * q_prev
        beta = jnp.linalg.norm(z)

        alpha_f = float(jax.device_get(alpha))
        beta_f = float(jax.device_get(beta))
        alphas.append(alpha_f)
        betas.append(beta_f)

        if beta_f == 0.0:
            converged = True
            break

        q_prev = q
        q = z / beta
        beta_prev = beta

        if len(alphas) >= n_lanczos:
            emax = _tridiag_emax(alphas, betas)
            emax_seq.append(emax)
            if prev_emax is not None:
                delta = abs(emax - prev_emax)
                thresh = float(emax_atol) + float(emax_rtol) * abs(emax)
                if delta <= thresh:
                    stable += 1
                else:
                    stable = 0
            prev_emax = emax
            if stable >= emax_patience:
                converged = True
                break

    if not emax_seq:
        emax_seq.append(_tridiag_emax(alphas, betas))
    e_max_ry = float(emax_seq[-1])
    if not use_tda:
        e_min_ry = -abs(e_max_ry)

    # Count diagonal elements (non-interacting transitions) as a function of energy.
    delta_E = energy_diff_cv_k(eps_c_use, eps_v_use)  # (n_cond, n_val, nk)
    diag_flat = np.asarray(jax.device_get(delta_E.real)).ravel()
    if not use_tda:
        diag_flat = np.concatenate([diag_flat, -diag_flat])

    return {
        "e_min_ry": float(jax.device_get(e_min_ry)),
        "e_max_ry_raw": e_max_ry,
        "n_lanczos_used": len(alphas),
        "n_lanczos_min": int(n_lanczos),
        "n_lanczos_max": int(n_lanczos_max),
        "emax_rtol": float(emax_rtol),
        "emax_atol": float(emax_atol),
        "emax_patience": int(emax_patience),
        "emax_converged": bool(converged),
        "emax_seq": np.array(emax_seq, dtype=np.float64),
        "diag_energies_ry": diag_flat,
    }
