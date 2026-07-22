"""Full-frequency screened Coulomb ``W_q(omega)`` via ONE structure-preserving
block-Lanczos chain — the production W(omega) model.

This is the amortized companion of the per-omega shifted-solve oracle in
``bse_w_exact`` (``--compare-w0`` / ``--compare-wq``).  Where the oracle runs a
fresh block-GMRES solve for EVERY frequency, this module runs ONE block-Lanczos
chain per q and then evaluates ``W_q(omega) - v_q`` for ARBITRARY complex omega
as a tiny reduced-matrix resolvent — NO large solve per omega.  The oracle stays
the gate reference (it is the ground truth this model is validated against); this
is the object GW/BSE actually consumes across a frequency grid.

Structure-preservation (the whole point).  The RPA screening operator is the
para-Hermitian / symplectic

    H_RPA = [[ A ,  B ],[ -B , -A ]],   A = D + V,  B = V,   (screening ring K^A)

with ``D`` the diagonal transition energies ``eps_c - eps_v`` (>0) and ``V`` the
Hermitian ring kernel (``build_bse_ring_matvec_full(screening=True)``).  The
screened Coulomb column obeys (bse_w_exact docstring; PHASE2_LOG "W(0)"):

    W(z) - v = L (z I - H_RPA)^{-1} B_seed,
    B_seed_nu = [ f_nu ; -f_nu ],   f_nu = M^dag v e_nu   (= SEED / ``gen``),
    L(x)      = v M (x_X + x_Y)                            (= PROJECT / ``snapshot``).

The [[A,B],[-B,-A]] para-structure (Shao et al.; the Casida/TDDFT ``Omega^2``
reduction) collapses this 2N symplectic resolvent onto an N-dimensional SYMMETRIC
one.  In the (q = X+Y, p = X-Y) basis H acts as ``q' = (A-B) p``,
``p' = (A+B) q``; the seed ``[f;-f]`` is pure-p (q=0) and the readout reads q, so
element-by-element (derivation validated bit-exact, proto_chain.py rel 1e-15):

    W(z) - v = 2 * Phi [ z^2 I - S ]^{-1} Phi^dag,                          (*)
      S      = D^{1/2} (A+B) D^{1/2} = D^{1/2}(D + 2V)D^{1/2}   (Hermitian!),
      Phi    = v M D^{1/2}   (density<-transition),
      Phi^dag e_nu = D^{1/2} f_nu = D^{1/2} M^dag v e_nu  (the SEED, scaled).

Only ``z^2`` enters, so ONE chain serves every omega on the whole complex plane.
``S`` is Hermitian in the ordinary Euclidean inner product, so this is a genuine
symmetric block-Lanczos with a three-term recurrence and real-block-tridiagonal
reduced matrix — no indefinite-metric ghosts.  ``A-B = D`` is EXACT for the
screening operator (A=D+V, B=V), so (*) is exact, not an approximation.

Reuse (single source).  ``S`` is applied through the production matvec VERBATIM
(no new kernel, no duplicated encode/decode): for any block ``U``,
``(A+B) U = matvec([U;U])[X-block]`` because ``H[U;U] = [(A+B)U; -(A+B)U]``.  The
seed is stage-1 ``gen`` (SEED) scaled by ``D^{1/2}``; the readout is stage-3
``snapshot`` (PROJECT) of ``D^{1/2} x`` scaled by 2 — exactly the seam
``apply_screening_resolvent_block`` documents, with the middle GMRES swapped for
the chain.

Block Lanczos (per-element).  Seed block ``B0[b] = D^{1/2} f_{cols[b]}`` (b over
the ``p = len(cols)`` probe columns), block-orthonormalize ``B0 = Q_0 R_0``.
For ``j = 0 .. m-1`` (all blocks ``p x (c,v,k)``, inner product Euclidean over
``(c,v,k)``, reduced by the mesh allreduce):

    Wb        = S Q_j
    alpha_j   = Q_j^dag Wb                         (p x p, Hermitian)
    Wb        = Wb - Q_j alpha_j - Q_{j-1} beta_{j-1}^dag
    (DGKS)    Wb -= sum_{i<=j} Q_i (Q_i^dag Wb)    [full reorthogonalization]
    Q_{j+1} R = Wb  (block-QR)  =>  beta_j = R

giving the block-tridiagonal ``T`` (diag ``alpha_j``, sub ``beta_j``, super
``beta_j^dag``).  Full reorthogonalization is mandatory — the head-injected /
stiff tiles lose orthogonality catastrophically under a bare 3-term recurrence
(same lesson as the GMRES DGKS pass, bse_feast).

Evaluator (per omega, tiny).  ``z = (omega + i eta)/Ry``; with ``E = [R_0;0;..;0]``
(``mp x p``, since ``Q^dag B0`` is nonzero only in the first block):

    C(z)   = ( z^2 I - T )^{-1} E                  (mp x p, host solve — tiny)
    x(z)   = sum_j Q_j C_j(z)                       (p x (c,v,k), device einsum)
    W(z)-v = 2 * snapshot( D^{1/2} x(z) )           (tile (mu_X, nu_Y) = sh.V)

No matvec, no GMRES per omega — one small dense solve + one linear combination of
the stored chain blocks + one PROJECT.  That is the amortization: ``m`` matvecs
once, then O(1) device work per frequency.

Sharding.  Chain blocks live in the pair basis (``sh.X`` per block; the stacked
chain reuses the ``sh.X_full`` spec ``P(None,None,'x','y',None)`` for
``(m,p,c,v,k)``); the reduced matrices ``T``/``R_0`` are tiny and replicated on
host (numpy); the evaluator projects with the existing reduce-scatter ``snapshot``
so the tile lands ``(mu_X, nu_Y)`` with no replicated ``(mu,nu)``.  Block size is
the probe-block width ``p = len(cols)`` — kept small (the ring matvec is optimal
for small blocks, crossover nt~2-3; matvec efficiency audit), which also keeps
the reduced ``mp x mp`` matrix small.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# --- small block-algebra primitives (jitted; shapes constant across the chain) ---

@jax.jit
def _block_gram(A: jax.Array, B: jax.Array) -> jax.Array:
    """``G[a,b] = <A_a, B_b> = sum_{c,v,k} conj(A[a]) B[b]`` — a (p x p) matrix.

    ``A``/``B`` are pair-basis blocks ``(p, c, v, k)`` (``sh.X``); the sum over the
    mesh-tiled ``(c,v,k)`` is completed by the XLA allreduce, so ``G`` is the
    replicated Euclidean Gram of the two blocks' columns."""
    return jnp.einsum("acvk,bcvk->ab", jnp.conj(A), B)


@jax.jit
def _block_combine(Qblk: jax.Array, Mmat: jax.Array) -> jax.Array:
    """``out[b] = sum_a Q[a] M[a,b]`` — right-multiply a pair block by a (p x p).

    ``Qblk`` is ``(p, c, v, k)`` (``sh.X``), ``Mmat`` is the replicated ``(p, p)``;
    the output keeps the block layout ``(p, c, v, k)``."""
    return jnp.einsum("acvk,ab->bcvk", Qblk, Mmat)


def _block_orthonormalize(Wblk: jax.Array, sh) -> tuple[jax.Array, np.ndarray]:
    """Robust block-QR ``Wblk = Q R`` with rank-deficiency deflation.

    Columns are the ``p`` block entries; the inner product is Euclidean over the
    pair indices ``(c,v,k)``.  Uses the Gram / eigen route (``G = W^dag W`` is a
    tiny ``p x p`` matrix, so its Hermitian eigendecomposition on host is cheaper
    and more robust than a distributed tall-skinny QR):

        G = Z diag(lam) Z^dag,      (Hermitian, lam >= 0)
        R = diag(sqrt(lam)) Z^dag,  Q = W (Z diag(lam^{-1/2}))     [W = Q R]

    Near-zero ``lam`` (parallel / zero probe columns — degenerate centroids or
    zero pad columns) get a clamped inverse -> that ``Q`` column is set to 0 and
    the matching ``R`` row to 0, so the deflated directions never enter the chain.
    ``R`` is returned as host numpy (feeds the replicated reduced matrix); ``Q``
    stays on device with the block sharding ``sh.X``."""
    G = np.asarray(jax.device_get(_block_gram(Wblk, Wblk)))     # (p, p) Hermitian
    G = 0.5 * (G + G.conj().T)
    lam, Z = np.linalg.eigh(G)
    lam = lam.real
    tol = max(G.shape[0], 1) * np.finfo(np.float64).eps * (lam.max() if lam.size else 0.0)
    keep = lam > tol
    inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.where(keep, lam, 1.0)), 0.0)
    sqrt_lam = np.where(keep, np.sqrt(np.where(keep, lam, 0.0)), 0.0)
    R = (np.diag(sqrt_lam) @ Z.conj().T).astype(np.complex128)      # (p, p)
    Tr = (Z @ np.diag(inv_sqrt)).astype(np.complex128)             # (p, p) : Q = W Tr
    Q = jax.lax.with_sharding_constraint(
        _block_combine(Wblk, jnp.asarray(Tr)), sh.X)
    return Q, R


def _make_apply_S(matvec, data: dict, D_half: jax.Array, sh):
    """Return ``U -> S U`` for ``S = D^{1/2}(A+B)D^{1/2}`` reusing the matvec.

    ``(A+B) U`` is extracted from the production symplectic matvec VERBATIM:
    ``H_RPA [U;U] = [(A+B)U; -(A+B)U]`` (X=Y=U), so ``(A+B)U`` is the X-block of
    one matvec call.  ``D_half`` is the ``(1,c,v,k)`` diagonal ``sqrt(eps_c-eps_v)``
    (clamped >=0; only unphysical/pad slots clamp, and their seed weight is 0)."""
    args = (
        data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
        data["eps_c"], data["eps_v"], data["W_R"], data["V_q0"],
        data["M_X"], data["M_Y"],
    )

    def apply_S(U: jax.Array) -> jax.Array:
        u = jax.lax.with_sharding_constraint(D_half * U, sh.X)          # (p,c,v,k)
        uu = jax.lax.with_sharding_constraint(
            jnp.stack([u, u], axis=0).astype(jnp.complex128), sh.X_full)  # (2,p,c,v,k)
        w = matvec(uu, *args)[0]                                        # (A+B)u
        return jax.lax.with_sharding_constraint(D_half * w, sh.X)

    return apply_S


def _seed_block(cols, data: dict, gen, sh):
    """Symmetric seed block ``B0[b] = D^{1/2} f_{cols[b]}`` in the pair basis.

    ``f`` is the stage-1 SEED (``gen``): ``f_nu = M^dag (v e_nu)``.  Block width is
    ``p = len(cols)`` (no py-pad here — the chain runs full-rank; padding to a
    multiple of py happens only at the snapshot boundary in the evaluator)."""
    n_rmu = int(data["V_q0"].shape[0])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    cols = np.asarray(cols, dtype=int)
    p = len(cols)
    G = np.zeros((p, n_rmu), dtype=np.float64)
    for b, nu0 in enumerate(cols):
        G[b, int(nu0)] = 1.0
    r = jax.device_put(
        jnp.broadcast_to(jnp.asarray(G)[:, :, None], (p, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    return f  # (p, c, v, k)


def build_w_omega_chain(data, matvec, gen, sh, cols, chain_len,
                        *, reorth_passes=2):
    """Build ONE structure-preserving block-Lanczos chain for the probe ``cols``.

    Runs symmetric block Lanczos on ``S = D^{1/2}(D+2V)D^{1/2}`` (applied through
    the production matvec) seeded with the ``D^{1/2}``-scaled SEED block, with full
    (DGKS) reorthogonalization.  Returns a plain-dict chain object (no class):

      ``alpha`` : (m, p, p) complex  — block-tridiagonal diagonal blocks
      ``beta``  : (m, p, p) complex  — off-diagonal blocks (``beta[m-1]`` is the
                  final residual-norm block, an error estimate, NOT part of ``T``)
      ``R0``    : (p, p) complex     — seed block-QR factor (the "seed norms")
      ``V_stack``: (m, p, c, v, k) device array (``sh.X_full`` spec) — the chain
                  basis blocks ``Q_0 .. Q_{m-1}`` for the evaluator
      ``D_half``: (1, c, v, k) device — the transition ``sqrt`` diagonal
      ``cols``  : (p,) int           — the probe density columns
      ``m``, ``p``                   — chain length, block width

    The evaluator (:func:`eval_w_omega_chain`) may request any ``m_use <= m`` by
    slicing — build once at the largest length and read the convergence sweep off
    the truncations for free.
    """
    cols = np.asarray(cols, dtype=int)
    p = len(cols)

    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    # delta_E[1,c,v,k] = eps_c[k,c] - eps_v[k,v]  (same construction as the
    # matvec's apply_D_term).  A-B = D is EXACT for the screening operator.
    delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
    D_half = jax.lax.with_sharding_constraint(
        jnp.sqrt(jnp.clip(delta_E.real, 0.0, None)).astype(jnp.complex128), sh.X)

    apply_S = _make_apply_S(matvec, data, D_half, sh)

    B0 = _seed_block(cols, data, gen, sh)                       # (p,c,v,k)
    B0 = jax.lax.with_sharding_constraint(D_half * B0, sh.X)
    Q0, R0 = _block_orthonormalize(B0, sh)

    Qs = [Q0]
    alphas: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    zero_pp = jnp.zeros((p, p), dtype=jnp.complex128)
    Qprev = jax.lax.with_sharding_constraint(jnp.zeros_like(Q0), sh.X)
    beta_prev = zero_pp

    for j in range(chain_len):
        Wb = apply_S(Qs[j])
        alpha = _block_gram(Qs[j], Wb)                          # (p,p) Hermitian
        Wb = Wb - _block_combine(Qs[j], alpha) - _block_combine(Qprev, beta_prev.conj().T)
        # Full (DGKS) reorthogonalization against all stored blocks.
        for _ in range(int(reorth_passes)):
            for Qi in Qs:
                Wb = Wb - _block_combine(Qi, _block_gram(Qi, Wb))
        Wb = jax.lax.with_sharding_constraint(Wb, sh.X)
        Qn, beta = _block_orthonormalize(Wb, sh)
        alphas.append(np.asarray(jax.device_get(alpha)))
        betas.append(beta)
        beta_prev = jnp.asarray(beta)
        Qprev = Qs[j]
        Qs.append(Qn)

    m = chain_len
    V_stack = jax.lax.with_sharding_constraint(
        jnp.stack(Qs[:m], axis=0), sh.X_full)                   # (m,p,c,v,k)
    return {
        "alpha": np.stack(alphas, axis=0),                      # (m,p,p)
        "beta": np.stack(betas, axis=0),                        # (m,p,p)
        "R0": np.asarray(R0),                                   # (p,p)
        "V_stack": V_stack,
        "D_half": D_half,
        "cols": cols,
        "m": m,
        "p": p,
    }


def _reduced_T(alpha: np.ndarray, beta: np.ndarray, m_use: int) -> np.ndarray:
    """Assemble the ``(m_use*p) x (m_use*p)`` block-tridiagonal reduced matrix.

    ``T`` has diagonal blocks ``alpha_j`` and off-diagonal ``beta_j`` (sub) /
    ``beta_j^dag`` (super) for ``j = 0 .. m_use-2``; ``beta[m_use-1]`` (residual
    norm) is intentionally excluded (Galerkin/Ritz reduced operator)."""
    p = alpha.shape[1]
    T = np.zeros((m_use * p, m_use * p), dtype=np.complex128)
    for j in range(m_use):
        T[j * p:(j + 1) * p, j * p:(j + 1) * p] = alpha[j]
        if j + 1 < m_use:
            T[(j + 1) * p:(j + 2) * p, j * p:(j + 1) * p] = beta[j]
            T[j * p:(j + 1) * p, (j + 1) * p:(j + 2) * p] = beta[j].conj().T
    return T


@jax.jit
def _combine_chain(V_stack: jax.Array, C: jax.Array) -> jax.Array:
    """``x[b] = sum_{j,l} V_stack[j,l] C[j,l,b]`` — the per-omega linear combination.

    ``V_stack`` is ``(m,p,c,v,k)``, ``C`` is the replicated ``(m,p,p)`` coefficient
    tensor; the output is the pair block ``(p,c,v,k)`` (``sh.X``)."""
    return jnp.einsum("jlcvk,jlb->bcvk", V_stack, C)


def eval_w_omega_chain(chain, data, snapshot, sh, z, *, m_use=None):
    """Evaluate ``W(omega) - v`` for the chain's probe block at complex ``z``.

    ``z = (omega + i eta)/Ry`` (Rydberg).  Per (*):
    ``W(z)-v = 2 snapshot(D^{1/2} sum_j Q_j [(z^2 I - T)^{-1} E]_j)`` with
    ``E = [R0; 0; ...; 0]``.  Only ``z^2`` enters, so a single chain serves every
    ``omega``.  ``m_use`` truncates the chain (convergence sweeps); default full.

    Returns ``(W_tile, )`` with ``W_tile`` shape ``(n_rmu, p_pad)`` sharded
    ``sh.V`` = ``P('x','y')`` = ``(mu_X, nu_Y)``; column ``i`` is probe
    ``cols[i]`` (final ``p_pad - p`` columns are zero pad from the snapshot
    reduce-scatter tiling)."""
    m = int(chain["m"])
    p = int(chain["p"])
    m_use = m if m_use is None else int(m_use)
    if not (1 <= m_use <= m):
        raise ValueError(f"m_use={m_use} out of range [1, {m}]")

    T = _reduced_T(chain["alpha"], chain["beta"], m_use)         # (m_use*p, m_use*p)
    E = np.zeros((m_use * p, p), dtype=np.complex128)
    E[:p] = chain["R0"]
    z2 = complex(z) ** 2
    C = np.linalg.solve(z2 * np.eye(m_use * p) - T, E)           # (m_use*p, p)
    C = C.reshape(m_use, p, p)

    px, py = sh.X.mesh.devices.shape
    C_dev = jax.device_put(jnp.asarray(C))
    V_use = jax.lax.with_sharding_constraint(
        chain["V_stack"][:m_use], sh.X_full)
    xz = jax.lax.with_sharding_constraint(_combine_chain(V_use, C_dev), sh.X)
    # Fold the factor 2 into s BEFORE the (jitted, out_sharding=sh.V) snapshot so
    # the returned tile carries the committed (mu_X, nu_Y) = P('x','y') spec — a
    # post-snapshot scalar rescale would drop it to replicated on a 1-device mesh.
    s = jax.lax.with_sharding_constraint(2.0 * chain["D_half"] * xz, sh.X)  # (p,c,v,k)

    # Pad the probe (batch) axis to a multiple of py for the reduce-scatter
    # snapshot (nu is tiled over y); the pad columns are zero.
    n_pad = int(np.ceil(p / py) * py)
    if n_pad != p:
        pad = jnp.zeros((n_pad - p,) + s.shape[1:], dtype=s.dtype)
        s = jax.lax.with_sharding_constraint(
            jnp.concatenate([s, pad], axis=0), sh.X)
    W_tile = snapshot(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
    return (W_tile,)


def chain_residual_norm(chain, m_use=None) -> float:
    """Frobenius norm of the final off-diagonal block ``beta[m_use-1]`` — the
    block-Lanczos residual estimate (how far the ``m_use`` subspace is from
    invariant).  A convergence knob diagnostic; not used in ``T``."""
    m = int(chain["m"])
    m_use = m if m_use is None else int(m_use)
    return float(np.linalg.norm(chain["beta"][m_use - 1]))
