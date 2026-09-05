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
from jax.sharding import PartitionSpec as P

from common.collectives import device_put_process_local

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


def _host_qr_factors(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HOST block-QR factors ``(R, Tr)`` from the ``(p,p)`` Gram ``G = W^dag W``.

    This is the numerical heart of the block orthonormalization ``W = Q R``,
    and it runs on the HOST in ``numpy`` deliberately.  Columns are the ``p``
    block entries; the inner product is Euclidean over the pair indices
    ``(c,v,k)``.  The Gram / eigen route is used because ``G`` is a tiny
    ``p x p`` matrix, so its Hermitian eigendecomposition on host is cheaper and
    more robust than a distributed tall-skinny QR:

        G = Z diag(lam) Z^dag,      (Hermitian, lam >= 0)
        R = diag(sqrt(lam)) Z^dag,  Q = W (Z diag(lam^{-1/2}))     [W = Q R]

    Near-zero ``lam`` (parallel / zero probe columns — degenerate centroids or
    zero pad columns) get a clamped inverse -> that ``Q`` column is set to 0 and
    the matching ``R`` row to 0, so the deflated directions never enter the
    chain.

    WHY THIS STAYS ON HOST.  ``np.linalg.eigh`` (LAPACK ``zheevd``) and
    ``jnp.linalg.eigh`` (cuSOLVER on GPU) agree only to within rounding, and the
    eigenvectors ``Z`` they return feed ``Q`` and hence every later block of the
    chain.  Moving this decomposition onto the device is therefore a change to
    the numerical path of a physics quantity, not a dispatch-hygiene change, and
    it is deliberately NOT folded into the scan-canonicalization of the loop
    around it: the conversion in :func:`build_w_omega_chain` keeps this call on
    host so the converted chain is BIT-IDENTICAL to the loop it replaced.  The
    ``p x p`` decomposition is microseconds; what the conversion removes is the
    ~2100 XLA dispatches and 65 blocking host syncs around it.

    Returns ``(R, Tr)`` as host ``numpy`` ``(p,p)`` complex128: ``R`` feeds the
    replicated reduced matrix, ``Tr`` is the right factor that turns the block
    into its orthonormal basis, ``Q = W Tr``."""
    G = 0.5 * (G + G.conj().T)
    lam, Z = np.linalg.eigh(G)
    lam = lam.real
    tol = max(G.shape[0], 1) * np.finfo(np.float64).eps * (lam.max() if lam.size else 0.0)
    keep = lam > tol
    inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.where(keep, lam, 1.0)), 0.0)
    sqrt_lam = np.where(keep, np.sqrt(np.where(keep, lam, 0.0)), 0.0)
    R = (np.diag(sqrt_lam) @ Z.conj().T).astype(np.complex128)      # (p, p)
    Tr = (Z @ np.diag(inv_sqrt)).astype(np.complex128)              # (p, p) : Q = W Tr
    return R, Tr


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
    # Process-local (scorecard AA.1): the probe block is identical on every
    # rank; the host ``np.broadcast_to`` view lets each rank materialise
    # only its own shard, and skips plain ``device_put``'s hidden
    # P-linear assert_equal all-gather.  LORRAX_CHECK_REPLICA=1 re-arms it.
    r = device_put_process_local(
        np.broadcast_to(G[:, :, None], (p, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(
        gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    return f  # (p, c, v, k)


# --- the compiled chain step, cached per operator signature -----------------
#
# ONE Python loop trip per chain step, ONE XLA program per trip, ONE host sync
# per trip.  Everything that used to be eager glue between the jitted leaves —
# the S application, the three-term recurrence, the whole DGKS double loop, and
# the Gram of the next block-QR — lives inside :func:`_get_chain_step`'s single
# ``jax.jit``.
#
# The cache is keyed on the operator STRUCTURE (``id(matvec)``, the mesh, the
# chain geometry, the block shape/dtype) and NOT on the per-q data: ``D_half``
# and the matvec operands are RUNTIME ARGUMENTS, so a whole finite-q sweep
# reuses one executable and every q after the first is dispatch-only.  That is
# the same contract, and the same reasoning, as
# ``bse_w_exact._get_block_gmres_solver`` — see its docstring for the ~4.8 s
# per-q recompile that a top-level (uncached) construct cost there.  The cache
# boundary is load-bearing, not decoration: an uncached scan re-traces per call
# and measured 3.4x SLOWER than the Python loop it replaced
# (PATTERN_scan_canonicalization).
#
# The step is NOT a ``lax.scan`` over the 32 chain steps, and that is a
# deliberate, documented stop.  A scan body must be pure JAX, which would force
# the ``p x p`` Hermitian eigendecomposition in :func:`_host_qr_factors` onto
# the device backend and change the numerical path of a physics quantity.  The
# scan that matters is the one INSIDE the step: the DGKS reorthogonalization,
# which is where 2112 of the old form's 2306 dispatches were.
#
# WHY THIS IS NOT BIT-IDENTICAL TO THE EAGER FORM, AND WHY THAT IS ALLOWED.
# Collapsing N programs into one hands XLA a fusion opportunity the eager form
# never had.  Where a subtract feeds a dot, the fused kernel computes that
# reduction itself instead of handing a materialized buffer to a library GEMM
# with a fixed accumulation order, and it is free to contract multiply+add into
# FMA; either changes the last bit, and this file does not separate them
# because the observable is the same.  MEASURED on the MoS2 6x6 deck
# (N_mu=1496, nk=36, p=6): ONE DGKS sweep differs by exactly 1 ulp (max_rel
# 1.368e-16 = 2^-53), which 32 block-Lanczos steps amplify to max_rel 1.2e-10
# in ``V_stack`` and 6.4e-11 in ``beta``.  Measured individually, the combine,
# the combine-then-scale, the INLINED matvec and a bare Gram are each
# bit-identical; only a dot fed by a subtract is not.  A trace-time-UNROLLED
# loop drifts identically to the ``lax.fori_loop``, so the loop construct is
# innocent -- it is fusion, not iteration.
#
# An earlier revision pinned the arithmetic exactly with a
# ``lax.optimization_barrier`` at every point the eager form had an XLA program
# boundary.  It worked: the A/B came back bit-identical on every array.  It also
# cost 1.52x of the achievable speed, because those barriers block the same
# fusion that makes the DGKS sweep cheap -- MEASURED, MoS2 6x6 at P=4, warm
# chain build: 1.756 s barriered against 1.142 s not.  The owner ruled on
# 2026-08-08 that 1.2e-10 is tolerable here, so the pins are gone and the gate
# moved with them: ``tests/test_bse_w_omega_chain_scan.py`` now compares against
# the eager reference at ``rel <= 1e-9`` (roughly 8x headroom over the measured
# drift) instead of ``np.array_equal``, so a real regression is still caught.
#
# What did NOT change is the ``p x p`` eigendecomposition, which stays on HOST
# (:func:`_host_qr_factors`).  That is a different question from fusion
# reassociation -- a different LAPACK-vs-cuSOLVER algorithm applied to the
# eigenvectors that feed every later chain block -- and it has not been ruled
# on.
_CHAIN_STEP_CACHE: dict[tuple, tuple] = {}


def _chain_step_key(matvec, sh, m, p, reorth_passes, x_shape, x_dtype) -> tuple:
    """Hashable identity of a compiled chain step.

    Mesh IDENTITY (axis names, extents, device ids), not just shape: two meshes
    of the same geometry over different devices lower to different programs, and
    a shape-only key would alias them into each other's executable.  ``sh`` and
    ``matvec`` are retained alongside the entry (see :data:`_CHAIN_STEP_CACHE`
    writes) so neither ``id()`` can be recycled onto a different object while a
    cached step still names it."""
    mesh = sh.X.mesh
    return (id(matvec),
            tuple(mesh.axis_names),
            tuple(int(s) for s in mesh.shape.values()),
            mesh.devices.flat[0].platform,
            tuple(int(d.id) for d in mesh.devices.flat),
            int(m), int(p), int(reorth_passes),
            tuple(int(s) for s in x_shape), str(x_dtype))


def _get_chain_step(matvec, sh, *, m, p, reorth_passes, x_shape, x_dtype):
    """The compiled one-step block-Lanczos program for this signature.

    Signature of the returned ``jax.jit``::

        (V, W_prev, Tr_prev, Q_prev, beta_prev, j, D_half, args)
            -> (V, Wb, alpha, G, Q_j)

    ``V`` is the fixed-size ``(m,p,c,v,k)`` chain buffer (``sh.X_full``), zero in
    every slot the chain has not reached yet; ``j`` is a RUNTIME scalar, so the
    growing DGKS range is a ``lax.fori_loop`` bound rather than a new program per
    step.  ``Q_j = W_prev Tr_prev`` is the device half of the PREVIOUS step's
    block-QR, pulled into this program so a step costs one dispatch and not two.
    ``G = Wb^dag Wb`` is the device half of the NEXT one, returned so the caller
    can take a single host sync per step for ``(alpha, G)`` together.

    Returns ``(init, step)``.  ``init`` is not cosmetic: ``jax.jit`` keys its
    trace cache on argument SHARDING as well as shape, and an eagerly built
    ``with_sharding_constraint`` normalizes its spec (a trailing ``None`` is
    dropped) while a jit OUTPUT keeps the full-length one.  Seeding the loop
    with eager arrays therefore gave step j=0 a different signature from steps
    1..m-1 and the step program was traced TWICE per chain (MEASURED, and the
    reason ``test_bse_w_omega_chain_scan`` asserts exactly one).  Building the
    initial carry inside a jit of its own makes the first call's operands come
    from the same place every later call's do."""
    key = _chain_step_key(matvec, sh, m, p, reorth_passes, x_shape, x_dtype)
    hit = _CHAIN_STEP_CACHE.get(key)
    if hit is not None:
        return hit[-2], hit[-1]

    @jax.jit
    def _init(B0):
        """Initial ``(V, W_prev, Q_prev)`` carry, in the step's own shardings."""
        V = jax.lax.with_sharding_constraint(
            jnp.zeros((m,) + B0.shape, dtype=B0.dtype), sh.X_full)
        W_prev = jax.lax.with_sharding_constraint(B0, sh.X)
        Q_prev = jax.lax.with_sharding_constraint(jnp.zeros_like(B0), sh.X)
        return V, W_prev, Q_prev

    @jax.jit
    def _step(V, W_prev, Tr_prev, Q_prev, beta_prev, j, D_half, args):
        # (1) finish the previous block-QR on device: Q_j = W_prev Tr_prev.
        Qj = jax.lax.with_sharding_constraint(
            _block_combine(W_prev, Tr_prev), sh.X)
        # (2) park Q_j in the chain buffer.  DGKS below reads V[0..j], which is
        #     exactly the stored basis Q_0 .. Q_j the growing Python list held.
        V = jax.lax.with_sharding_constraint(
            jax.lax.dynamic_update_index_in_dim(V, Qj, j, 0), sh.X_full)
        # (3) Wb = S Q_j, through the production matvec VERBATIM:
        #     H_RPA [u;u] = [(A+B)u; -(A+B)u], so (A+B)u is the X-block.
        u = jax.lax.with_sharding_constraint(D_half * Qj, sh.X)
        uu = jax.lax.with_sharding_constraint(
            jnp.stack([u, u], axis=0).astype(jnp.complex128), sh.X_full)
        w = matvec(uu, *args)[0]
        Wb = jax.lax.with_sharding_constraint(D_half * w, sh.X)
        # (4) the three-term recurrence, spelled exactly as the eager form did.
        alpha = _block_gram(Qj, Wb)                             # (p,p) Hermitian
        Wb = Wb - _block_combine(Qj, alpha) - _block_combine(
            Q_prev, beta_prev.conj().T)

        # (5) full (DGKS) reorthogonalization against all stored blocks.  The
        #     unfilled slots of V are exactly zero, so the loop could run to m
        #     with the same VALUES; it runs to j+1 instead to keep the work
        #     triangular, which is what the growing Python list did.
        def _dgks(i, W_cur):
            Qi = jax.lax.with_sharding_constraint(
                jax.lax.dynamic_index_in_dim(V, i, axis=0, keepdims=False), sh.X)
            return W_cur - _block_combine(Qi, _block_gram(Qi, W_cur))

        for _ in range(int(reorth_passes)):
            Wb = jax.lax.fori_loop(0, j + 1, _dgks, Wb)
        Wb = jax.lax.with_sharding_constraint(Wb, sh.X)
        # (6) device half of the NEXT block-QR — returned with alpha so the
        #     caller takes ONE host sync per step instead of two.
        G = _block_gram(Wb, Wb)
        return V, Wb, alpha, G, Qj

    # Retain matvec and sh: the key names them by id(), and an entry that did
    # not hold them alive could be handed a recycled id (bse_w_exact:285-288).
    _CHAIN_STEP_CACHE[key] = (matvec, sh, _init, _step)
    return _init, _step


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

    Shape of the loop (BSE_CODE_SURVEY R1).  Each chain step is ONE dispatch of
    the cached program from :func:`_get_chain_step` and ONE host sync; the chain
    basis lives in a preallocated ``(m,p,c,v,k)`` buffer that the step writes
    with ``dynamic_update_index_in_dim`` instead of a growing Python list, and
    the DGKS double loop is a ``lax.fori_loop`` over that buffer inside the
    program.  It agrees with the eager form it replaced to ``rel <= 1.2e-10``
    (measured; the block comment above :data:`_CHAIN_STEP_CACHE` says where that
    comes from and what was traded for it), and the buffer is preallocated
    rather than stacked at the end because the old form held ``m+1`` separate
    blocks AND their stack alive at once, so this is a lower peak, not a new
    one."""
    cols = np.asarray(cols, dtype=int)
    p = len(cols)
    m = int(chain_len)

    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    # delta_E[1,c,v,k] = eps_c[k,c] - eps_v[k,v]  (same construction as the
    # matvec's apply_D_term).  A-B = D is EXACT for the screening operator.
    delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
    D_half = jax.lax.with_sharding_constraint(
        jnp.sqrt(jnp.clip(delta_E.real, 0.0, None)).astype(jnp.complex128), sh.X)

    # The matvec operands ride as RUNTIME ARGUMENTS of the compiled step (not
    # closed over), so the cached executable is reused across q and omega.
    args = (
        data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
        data["eps_c"], data["eps_v"], data["W_R"], data["V_q0"],
        data["M_X"], data["M_Y"],
    )

    B0 = _seed_block(cols, data, gen, sh)                       # (p,c,v,k)
    B0 = jax.lax.with_sharding_constraint(D_half * B0, sh.X)

    init, step = _get_chain_step(matvec, sh, m=m, p=p,
                                 reorth_passes=reorth_passes,
                                 x_shape=B0.shape, x_dtype=B0.dtype)

    # Seed block-QR.  Its device half (Q_0 = B0 Tr_0) is done by step j=0, so
    # the prologue is one Gram and one host sync.
    R0, Tr = _host_qr_factors(np.asarray(jax.device_get(_block_gram(B0, B0))))

    V, W_prev, Q_prev = init(B0)
    Tr_prev = jnp.asarray(Tr)
    beta_prev = jnp.zeros((p, p), dtype=jnp.complex128)
    alphas: list[np.ndarray] = []
    betas: list[np.ndarray] = []

    for j in range(m):
        V, W_prev, alpha, G, Q_prev = step(
            V, W_prev, Tr_prev, Q_prev, beta_prev,
            np.int32(j), D_half, args)
        # ONE blocking host sync per step (was two: alpha, then the Gram
        # inside _block_orthonormalize).
        alpha_h, G_h = jax.device_get((alpha, G))
        beta, Tr = _host_qr_factors(np.asarray(G_h))
        alphas.append(np.asarray(alpha_h))
        betas.append(beta)
        Tr_prev = jnp.asarray(Tr)
        beta_prev = jnp.asarray(beta)

    # V already holds Q_0 .. Q_{m-1} with the sh.X_full spec; the terminal
    # jnp.stack of the old form is gone with the list it stacked.  Q_m is never
    # formed (the old form built it and then dropped it) — only its block-QR
    # factor beta[m-1], the residual estimate, is kept, and that comes off the
    # host factorization above without touching the device.
    return {
        "alpha": np.stack(alphas, axis=0),                      # (m,p,p)
        "beta": np.stack(betas, axis=0),                        # (m,p,p)
        "R0": np.asarray(R0),                                   # (p,p)
        "V_stack": V,
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
    from runtime.padding import padded_axis
    n_pad = padded_axis(
        p, sh.X.mesh, name="BSE omega-chain probe carrier",
        spec=P("y", None, None, None), axis=0).carrier
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
