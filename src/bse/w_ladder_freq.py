"""Frequency-amortized ladder ``W(z)``: ONE shifted-Krylov chain per
``(q, probe block)``, then every complex ``z`` as a small host solve.

This is the optimization deferral :func:`w_ladder.refuse_chain_path` names,
built: the ``w_omega_chain`` z^2 reduction is refused for the ladder because
its algebra (``A - B = D``, symplectic row) does not hold there, and THIS
module is the reduction that holds instead.  It changes nothing about what is
computed — the gate reference remains the per-z shifted block GMRES
(:func:`bse_w_exact.apply_screening_resolvent_block`) on the SAME operator —
it only removes the per-z Krylov rebuild.

The derivation
--------------
Everything W-shaped in this file evaluates

    W(z) - v  =  v tau (z - H)^{-1} sigma v ,

with ``H`` the ladder screening operator ``w_ladder`` derives (hybrid
anti-resonant row, rung-compensated at finite q), ``sigma u = [M u; -M u]``
the seed and ``tau [X; Y] = M^dag (X + Y)`` the readout.

**1. The Krylov space of the shifted family is z-INDEPENDENT.**
``K_m(z - H, b) = span{b, (z-H)b, ...} = span{b, Hb, ..., H^{m-1} b}
= K_m(H, b)`` for every complex ``z`` — the shift only re-mixes the same
vectors.  So one block Krylov basis ``V_m`` built from the seed block
``Z_0 = sigma (v G)`` serves EVERY frequency: per ``z``, Galerkin (FOM) on
that basis gives

    x_m(z) = V_m y(z),      (z I_m - H_m) y(z) = V_m^dag Z_0 = E_1 R_0 ,

with ``H_m = V_m^dag H V_m`` the (m p x m p) block-Hessenberg projection and
``Z_0 = V_1 R_0`` the seed block QR.  The readout is stage-3 of the GMRES
engine verbatim: ``tile(z) = snapshot(sum_j S_j y_j(z))`` with
``S_j = (X + Y)``-collapse of basis block ``j`` — linearity of the snapshot
moves it outside the z loop, so the per-z cost is one ``(m p)``-sized host
solve, one small device einsum and one snapshot dispatch.  The chain build is
``m`` ladder matvecs; the oracle pays ``~iters`` matvecs PER ``z``.

**2. The exact-arithmetic residual is free.**  The block Arnoldi relation
``H V_m = V_m H_m + Q_{m+1} B_m E_m^dag`` gives, for the shifted Galerkin
solution, the TRUE residual

    (z - H) x_m(z) - Z_0  =  Q_{m+1} B_m [y(z)]_{last block} ,

whose per-column norm ``||B_m y_last(z)||_col`` is computable on the HOST from
the small matrices, per z, with no extra matvec.  Every evaluation therefore
carries its own convergence certificate — the same role the GMRES engine's
per-column residual plays, satisfying the same gate
(``screening_bse`` gates on residuals, never on a return code).

**3. Why Euclidean block Arnoldi and not a metric-exploiting Lanczos.**
The ladder operator is ``H = Sigma_3 M`` with ``Sigma_3 = diag(I, -I)`` and
``M`` HERMITIAN (read it off ``w_ladder``'s derivation: ``M = D~ + K^d +
[[G, G], [G, G]]`` — the direct kernel ``K^d`` is Hermitian by construction,
step 4 there, and the ring dyad and ``D~`` are Hermitian), so ``H`` is
self-adjoint in the indefinite ``Sigma_3`` inner product and admits a
three-term ``Sigma_3``-Lanczos whose left vectors are free
(``tau = sigma^dag Sigma_3`` — the readout IS the ``Sigma_3``-adjoint of the
seed).  That recurrence is NOT used, deliberately: full (DGKS)
reorthogonalization is mandatory in this repo for every chain
(``w_omega_chain.py`` — losing it silently corrupts the reduced model), and
with full reorthogonalization the three-term shortcut saves nothing (the
orthogonalization sweep already touches every stored block), while the
indefinite metric adds a real breakdown mode (isotropic vectors,
``<x, Sigma_3 x> = 0``) that Euclidean Arnoldi simply does not have.  Same
cost, strictly fewer failure modes; the ``Sigma_3`` structure survives as the
REASON the Hessenberg projection of this non-normal ``H`` behaves like a
Hermitian problem's (real spectrum in hyperbolic ``+/-omega`` pairs for a
positive-gap deck).

**4. What the z^2 chain could not have done — and what this one is NOT.**
``w_omega_chain`` needs ``A - B = D`` exactly and the symplectic row to
collapse 2N -> N with only ``z^2`` entering.  The ladder has neither
(``A - B = D - W_d + W_d^B``, hybrid row), so its reduction must keep the
full 2N operator — which is what the shifted-Krylov identity in step 1 does,
at the price that ``z`` and ``-z`` are no longer the same evaluation.  The
frequency sets GW actually requests (a static point, ``i omega_p``, MPA's
damped z-plan) do not use that pairing, so nothing is lost.  The converse
question (owner, 2026-08-16): the RPA chain is NOT a special case of this
one, and should not be re-expressed as one — ``w_omega_chain`` exploits two
structures this reduction deliberately does not require (the N-dim Hermitian
z^2 collapse, halving the space and pairing ``+/-z`` for free), so replacing
it would trade real structure for uniformity.  Applied to the RPA operator
this module IS valid and reduces to plain block-Arnoldi FOM — the general
fallback — at ~2x the basis memory and no ``z`` pairing.  One family, two
tiers: the z^2 chain where its algebra holds, this chain where it does not,
both on the ``_chain_step_key`` cache discipline and the same seed/snapshot
seam.  ``bse_nontda.make_ab_appliers`` / ``solvers.bse_sp_lanczos`` (the SDY
product form) were evaluated and do NOT apply here: the exchange-free ladder
part is the optical SHAO operator, but the full screening operator's
unconjugated ring dyad breaks the (A+/-B) real-structure pairing SDY needs.

**5. Integration hook — iterative refinement (the preconditioning seam).**
The Galerkin solution ``x_m(z) = V_m y(z)`` is, by construction, the best
approximation to ``(z - H)^{-1} Z_0`` in the chain's Krylov space, with a
computable residual (step 2).  Where a caller needs a tighter tolerance than
a bounded ``m`` reaches, the cheap composition is chain-as-coarse-solve: hand
``x_m(z)`` to the shifted GMRES as its INITIAL GUESS and let it polish the
remaining orders — a few matvecs per z instead of a cold ~20+.
``bse_feast._gmres_solve_core`` currently starts from zero; threading an
``x0`` through it is the one-line seam that composition needs (deliberately
NOT patched from this module — bse_feast is shared solver infrastructure and
the change belongs with its owner).

Scaling envelope (TASTE 8 / INVARIANTS 9)
-----------------------------------------
Chain build per ``(q, probe block)``: ``m`` ladder matvecs + an ``O(m^2)``
DGKS sweep of block Grams (each a mesh-reduced ``(p x p)``), one host sync per
step (the ``w_omega_chain`` loop shape).  Memory high-water DURING build: the
basis buffer ``(m, 2, p, c, v, k)`` sharded on ``(x, y)`` — the same class as
``w_omega_chain``'s ``V_stack``, pair-basis and mesh-divided, no mu^2-class
object per rank anywhere (the only mu-indexed arrays are the seed's
``(p, n_rmu, nk)`` host-broadcast view and the ``P('x','y')`` output tile).
AFTER build only the collapsed ``(m, p, c, v, k)`` snapshot stack and the
``(m p)``-sized host matrices survive.  Per z: one host ``(m p)^3`` solve
(microseconds at m p ~ few thousand), one einsum over the snapshot stack, one
reduce-scatter snapshot.  Envelope ``N_mu -> 1e4, P -> 1e3`` is reached the
same way :func:`w_ladder.compute_wc_qwedge` reaches it — chunking the probe
axis and walking the wedge one q at a time — with ``m`` bounded (32..64) and
independent of ``N_mu``.  Two-plan note: ONE code path serves both plans (the
ring matvec at ``P = 1`` is the local plan), exactly as
``docs/dev/large_nmu_operation.md`` records for the ladder family.

Dispatch discipline: the chain step is ONE cached jitted program (keyed on the
operator identity via :func:`w_omega_chain._chain_step_key`), the per-q/z
tensors ride as runtime arguments, and a whole wedge x z sweep compiles the
step ONCE — same contract, same reasoning, as ``_get_block_gmres_solver`` and
``_get_chain_step``.
"""
from __future__ import annotations

import functools
from typing import Optional

import numpy as np

from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from .w_omega_chain import _host_qr_factors, _chain_step_key, _combine_chain
from .bse_feast import matvec_operands

jax.config.update("jax_enable_x64", True)


@jax.jit
def _gram2(A: jax.Array, B: jax.Array) -> jax.Array:
    """``G[a,b] = <A_a, B_b>`` over the FULL 2N pair basis ``(2, c, v, k)``.

    ``A``/``B`` are non-TDA blocks ``(2, p, c, v, k)`` (``sh.X_full``); the
    mesh-tiled sum is completed by the XLA allreduce, so ``G`` is the
    replicated Euclidean ``(p, p)`` Gram."""
    return jnp.einsum("sacvk,sbcvk->ab", jnp.conj(A), B)


@jax.jit
def _combine2(Vblk: jax.Array, Mmat: jax.Array) -> jax.Array:
    """``out[b] = sum_a V[a] M[a, b]`` for a 2N block ``(2, p, c, v, k)``."""
    return jnp.einsum("sacvk,ab->sbcvk", Vblk, Mmat)


#: operator-identity -> (matvec, sh, init, step); same retention reasoning as
#: ``w_omega_chain._CHAIN_STEP_CACHE`` (the key names matvec/sh by id()).
_FREQ_STEP_CACHE: dict = {}


def _stack_spec(sh) -> NamedSharding:
    """Sharding of the ``(m, 2, p, c, v, k)`` basis buffer: the ``sh.X_full``
    placement (c on x, v on y) with two leading unsharded axes."""
    return NamedSharding(sh.X.mesh, P(None, None, None, "x", "y", None))


def _get_freq_chain_step(matvec, sh, *, m, p, reorth_passes, x_shape, x_dtype):
    """The compiled one-step block-Arnoldi program for this operator signature.

    Mirrors :func:`w_omega_chain._get_chain_step` (deferred block-QR device
    half, preallocated basis buffer, DGKS as a ``fori_loop`` over that buffer,
    one dispatch and one host sync per step) with two Arnoldi-specific changes:
    the recurrence is not three-term — the DGKS sweep's Gram coefficients ARE
    the reduced matrix column, so they are ACCUMULATED and returned rather than
    discarded — and the basis blocks are full 2N ``(2, p, c, v, k)`` vectors,
    because the ladder operator does not collapse to N (module docstring,
    step 4)."""
    key = ("freq",) + _chain_step_key(matvec, sh, m, p, reorth_passes,
                                      x_shape, x_dtype)
    hit = _FREQ_STEP_CACHE.get(key)
    if hit is not None:
        return hit[-2], hit[-1]

    stack_sh = _stack_spec(sh)

    @jax.jit
    def _init(Z0):
        V = jax.lax.with_sharding_constraint(
            jnp.zeros((m,) + Z0.shape, dtype=Z0.dtype), stack_sh)
        W_prev = jax.lax.with_sharding_constraint(Z0, sh.X_full)
        return V, W_prev

    # V and W_prev are DONATED: the caller rebinds both from the returns, and
    # without donation the step holds input+output copies of the (m, 2, p,
    # c, v, k) basis buffer alive across the call — the dominant term of the
    # measured compile-time working set at m=128.
    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def _step(V, W_prev, Tr_prev, j, args):
        # (1) finish the previous block-QR on device: Q_j = W_prev Tr_prev.
        Qj = jax.lax.with_sharding_constraint(
            _combine2(W_prev, Tr_prev), sh.X_full)
        # (2) park Q_j in the basis buffer (unfilled slots stay exactly zero,
        #     so the fori_loop below can run over a fixed range).
        V = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_update_index_in_dim(V, Qj, j, 0), stack_sh)
        # (3) one ladder matvec, through the production operator VERBATIM.
        Wb = jax.lax.with_sharding_constraint(
            matvec(Qj, *args), sh.X_full)

        # (4) DGKS orthogonalization against ALL stored blocks, ACCUMULATING
        #     the coefficients: after both passes, Hcol[i] = Q_i^dag (H Q_j)
        #     exactly (the second pass's corrections are part of the same
        #     projection), so Hcol IS column j of the reduced Hessenberg.  The
        #     loop runs to j+1 — zero slots would contribute zero, but the
        #     triangular range keeps the work the same as a growing list's.
        def _dgks(i, carry):
            W_cur, Hcol = carry
            Qi = jax.lax.with_sharding_constraint(
                jax.lax.dynamic_index_in_dim(V, i, axis=0, keepdims=False),
                sh.X_full)
            g = _gram2(Qi, W_cur)
            W_cur = W_cur - _combine2(Qi, g)
            Hcol = jax.lax.dynamic_update_index_in_dim(
                Hcol, jax.lax.dynamic_index_in_dim(Hcol, i, 0, keepdims=False)
                + g, i, 0)
            return (W_cur, Hcol)

        Hcol = jnp.zeros((m, p, p), dtype=Wb.dtype)
        for _ in range(int(reorth_passes)):
            Wb, Hcol = jax.lax.fori_loop(0, j + 1, _dgks, (Wb, Hcol))
        Wb = jax.lax.with_sharding_constraint(Wb, sh.X_full)
        # (5) device half of the NEXT block-QR, returned with Hcol so the
        #     caller takes ONE host sync per step.
        G = _gram2(Wb, Wb)
        return V, Wb, Hcol, G

    _FREQ_STEP_CACHE[key] = (matvec, sh, _init, _step)
    return _init, _step


def build_ladder_freq_chain(data: dict, matvec, gen, sh, G_probe,
                            chain_len: int, *, reorth_passes: int = 2) -> dict:
    """Build ONE shifted-Krylov chain for the ladder operator and this probe.

    ``G_probe`` is the ``(n_probe, n_rmu)`` probe block in the padded centroid
    basis — the SAME object :func:`bse_w_exact.apply_screening_resolvent_block`
    takes, and stage 1 (SEED) is reused verbatim so the layout stays
    single-sourced (that function's docstring makes this a requirement of any
    chain model).  ``data`` must already carry ``W_R`` and, at finite q, be a
    ``build_finite_q_data`` payload matching the matvec's declared vertex
    convention (:func:`w_ladder.build_ladder_resolvent` handles both).

    Returns a plain-dict chain:

      ``H``       : host ``(m*p, m*p)`` complex — the block-Hessenberg
                    projection ``V^dag H V`` (columns from the DGKS sweeps,
                    sub-diagonal blocks from the block QRs)
      ``B_last``  : host ``(p, p)`` — the final residual block ``B_m``
                    (feeds the per-z residual estimate, NOT part of ``H``)
      ``R0``      : host ``(p, p)`` — seed block-QR factor
      ``S_stack`` : device ``(m, p, c, v, k)`` — the ``(X + Y)``-collapsed
                    basis blocks, ready for the stage-3 snapshot
      ``seed_norm``: host ``(n_probe,)`` — per-column ``||Z_0||`` (residuals
                    are reported relative to it, matching the GMRES engine)
      ``m``, ``p``

    The evaluator may use any ``m_use <= m`` — build once at the largest
    length and read the convergence sweep off the truncations.
    """
    p = int(G_probe.shape[0])
    m = int(chain_len)
    px, py = sh.X.mesh.devices.shape
    if p % py != 0:
        raise ValueError(
            f"probe block n_probe={p} must be a multiple of py={py} "
            "(reduce-scatter tiles nu over y); pad with zero rows.")
    n_rmu = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])

    # --- Stage 1 SEED, verbatim from apply_screening_resolvent_block. ---
    G = np.asarray(G_probe, dtype=np.float64)
    r = device_put_process_local(
        np.broadcast_to(G[:, :, None], (p, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    Z0 = jax.lax.with_sharding_constraint(
        jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)

    args = matvec_operands(data)
    init, step = _get_freq_chain_step(
        matvec, sh, m=m, p=p, reorth_passes=reorth_passes,
        x_shape=Z0.shape, x_dtype=Z0.dtype)

    G0_h = np.asarray(jax.device_get(_gram2(Z0, Z0)))
    R0, Tr = _host_qr_factors(G0_h)
    seed_norm = np.linalg.norm(R0, axis=0)          # ||Z0 e_b|| column norms

    V, W_prev = init(Z0)
    Tr_prev = jnp.asarray(Tr)
    H = np.zeros((m * p, m * p), dtype=np.complex128)
    B_last = np.zeros((p, p), dtype=np.complex128)

    for j in range(m):
        V, W_prev, Hcol, Gh = step(V, W_prev, Tr_prev, np.int32(j), args)
        Hcol_h, G_h = jax.device_get((Hcol, Gh))    # ONE host sync per step
        B, Tr = _host_qr_factors(np.asarray(G_h))
        H[: m * p, j * p:(j + 1) * p] = np.asarray(Hcol_h).reshape(m * p, p)
        if j + 1 < m:
            H[(j + 1) * p:(j + 2) * p, j * p:(j + 1) * p] = B
        else:
            B_last = B
        Tr_prev = jnp.asarray(Tr)

    # Collapse X + Y once; the snapshot is linear, so the z loop never touches
    # the 2N basis again.  Same spec family as w_omega_chain's V_stack.
    S_stack = jax.lax.with_sharding_constraint(
        V.sum(axis=1), NamedSharding(sh.X.mesh, P(None, None, "x", "y", None)))
    return {"H": H, "B_last": B_last, "R0": R0, "S_stack": S_stack,
            "seed_norm": seed_norm, "m": m, "p": p}


def eval_ladder_freq_chain(chain: dict, data: dict, snapshot, sh, z,
                           *, m_use: Optional[int] = None):
    """Evaluate ``W(z) - v`` for the chain's probe block at complex ``z``.

    Per-z cost: one host ``(m_use * p)``-sized solve, one device einsum over
    the snapshot stack, one reduce-scatter snapshot.  Returns
    ``(W_tile, resid)``:

      ``W_tile`` : ``(n_rmu, n_probe_pad)`` device, ``sh.V`` = ``P('x','y')``
                   — column ``b`` is ``(W(z) - v) G_probe[b]``, identical
                   contract to the GMRES engine's stage-3 output.
      ``resid``  : host ``(n_probe,)`` float — the EXACT shifted-Galerkin
                   residual ``||B_m y_last(z)||`` per column, relative to the
                   seed column norm (module docstring, step 2).  Gate on it
                   the way the GMRES residuals are gated; a value above the
                   caller's ceiling means ``m`` was too short for this z.
    """
    m, p = int(chain["m"]), int(chain["p"])
    m_use = m if m_use is None else int(m_use)
    if not (1 <= m_use <= m):
        raise ValueError(f"m_use={m_use} out of range [1, {m}]")
    n = m_use * p
    Hm = chain["H"][:n, :n]
    B_eff = (chain["B_last"] if m_use == m
             else chain["H"][n:n + p, n - p:n])
    E = np.zeros((n, p), dtype=np.complex128)
    E[:p] = chain["R0"]
    y = np.linalg.solve(complex(z) * np.eye(n) - Hm, E)          # (n, p)
    denom = np.where(chain["seed_norm"] > 0.0, chain["seed_norm"], 1.0)
    resid = np.linalg.norm(B_eff @ y[n - p:n, :], axis=0) / denom
    C = jax.device_put(jnp.asarray(y.reshape(m_use, p, p)))

    V_use = jax.lax.with_sharding_constraint(
        chain["S_stack"][:m_use],
        NamedSharding(sh.X.mesh, P(None, None, "x", "y", None)))
    s = jax.lax.with_sharding_constraint(_combine_chain(V_use, C), sh.X)
    px, py = sh.X.mesh.devices.shape
    n_pad = int(np.ceil(p / py) * py)
    if n_pad != p:                                   # pragma: no cover
        pad = jnp.zeros((n_pad - p,) + s.shape[1:], dtype=s.dtype)
        s = jax.lax.with_sharding_constraint(
            jnp.concatenate([s, pad], axis=0), sh.X)
    W_tile = snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
    return W_tile, resid
