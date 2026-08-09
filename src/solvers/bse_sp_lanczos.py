"""
solvers/bse_sp_lanczos.py — structure-preserving Lanczos for the definite
Bethe-Salpeter eigenvalue problem, thick-restarted, fixed shape.

Shao, da Jornada, Lin, Yang, Deslippe & Louie, *A structure preserving Lanczos
algorithm for computing the optical absorption spectrum*, SIAM J. Matrix Anal.
Appl. **39** (2018) 683, arXiv:1611.02348 — **Algorithm 4**; with the
thick-restart adaptation and the two-coefficient reorthogonalisation of
Alvarruiz, Mellado-Pinto, Campos & Roman, *Variants of thick-restart Lanczos for
the Bethe-Salpeter eigenvalue problem*, arXiv:2503.20920 (Algorithms 1 and 2;
shipped in SLEPc >= 3.22 as the ``shao`` variant).

WHAT THIS SOLVES, AND WHY IT IS NOT A HERMITIAN SOLVE IN DISGUISE
------------------------------------------------------------------
The full (non-TDA) optical BSE Hamiltonian is

    H = [[  A ,  B  ],
         [ -B̄ , -Ā  ]]        A^H = A ,   B^T = B  (complex SYMMETRIC, not Hermitian)

which is not Hermitian and not normal.  Two structural facts rescue it.  First,
``K := diag(I,−I)·H = [[A,B],[B̄,Ā]]`` is Hermitian, and when it is positive
definite — the physical stability condition — ``H`` is self-adjoint in the
``K``-inner product, so a Lanczos recurrence exists.  Second, ``H`` maps the
real form ``V_R = {[x; x̄]}`` into ``{[f; −f̄]}`` and back, so ``H²`` preserves
``V_R`` and acts on it as the REAL-LINEAR map

    P : x ↦ G(F(x)) ,      F(x) = A x + B x̄ ,      G(v) = A v − B v̄ .

The whole ``2N``-dimensional non-Hermitian problem therefore becomes an
``N``-dimensional recurrence for ``ω²`` whose target — the lowest exciton — is
an EXTREMAL eigenvalue rather than an interior one.  That folding is done by the
algorithm, which is why this converges like a Hermitian extremal solve and not
like a two-sided one.

This module never forms a ``2N`` vector.  ``V_R`` is parameterised by ``x``
alone and the de-excitation half ``Y`` is recovered analytically at the end, so
the paired structure costs a second ``N``-basis, not a ``2N`` layout.

THE TWO INVARIANTS — THE SUBTLE PART, AND THE EASIEST THING TO GET HALF-RIGHT
------------------------------------------------------------------------------
Exact arithmetic gives BOTH

    (a)  Re( u_i^H v_j ) = δ_ij                                  (SDY Eq. 25)
    (b)  Im( u_i^H u_j ) = 0   and   Im( v_i^H v_j ) = 0

and both are consumed downstream: the strong biorthogonality that the thick
restart's oblique projector needs is exactly (a) AND (b) —

    [[V,U],[V̄,−Ū]]^H [[U,V],[Ū,−V̄]] = 2 I_{2k}

has (a) on its diagonal blocks and (b) on its off-diagonal blocks.  A
reorthogonalisation that enforces only (a) leaves (b) free to drift, degrades
SILENTLY while the residuals still look fine, and fails at the restart — which
is where (b) is spent.  So the correction carries TWO coefficient sets, one
real and one purely imaginary:

    c = Re( V^H z )        (real)         — enforces (a)
    d = i · Im( U^H z )    (imaginary)    — enforces (b)
    z ← z − U c − V d

That is Alvarruiz et al. Algorithm 1, transcribed rather than re-derived.  The
types are forced, not chosen: ``F`` is only REAL-linear (``F(αx) = αAx + ᾱBx̄``
is not ``αF(x)``), so the part of the correction that must survive passage
through ``F`` has to be real — that is ``c`` — and the imaginary part cannot
ride the same basis, so it is carried on the other one as ``d``.  **Any complex
coefficient multiplying the U basis is a bug**, and ``single_coeff_set=True``
below is the red twin that shows what it costs.

The correction is exact as a projector: with (a) and (b) holding,
``Re(V^H ž) = c − c − Im(V^H V)·Im(d) = 0`` and
``Im(U^H ž) = δ − Im(U^H U)c − Re(U^H V)δ = 0``.  And it maps ``V_R`` to
``V_R`` — the lower half of the corrected pair stays the conjugate of the upper
half — precisely because ``c`` is real and ``d`` is imaginary.

COLLECTIVE ACCOUNTING
---------------------
The basis is stored as ONE array ``UV`` of shape ``(2, M, *trailing)`` with
``UV[0] = U`` and ``UV[1] = V``.  A reorthogonalisation pass is then one
``'pm...,...->pm'`` einsum producing a single ``(2, M)`` array — **one GEMM and
one all-reduce** — with ``Re(·)`` and ``i·Im(·)`` taken locally afterwards at
zero cost.  CGS2 is two passes, so **two collectives per step**, exactly the
count ``solvers.thick_restart_lanczos`` pays for the TDA problem with a payload
twice as wide.  Not ``2m`` collectives, not four: two.

FIXED SHAPE
-----------
``m_max``, ``n_keep``, ``n_restarts`` and ``n_eig`` are Python ints, so every
slice is static and every trip count is a compile-time constant.  Three traced
bodies exist — the cold step, the restart-cycle step (the same ``_step``
function, so ONE trace), and the two ``_build_T`` branches — and their trace
counts do not depend on ``n_restarts``.  ``TRACE_COUNTS`` proves it: run with 2
restart cycles and with 20 and the numbers are identical.

THE RESTART, AND THE ONE THING THAT MUST NOT BE GOT WRONG
----------------------------------------------------------
``T_k`` is REAL symmetric positive definite (SDY Theorem 2) and its eigenvalues
are ``θ²``.  Its eigenvector matrix ``Q`` is therefore real, and the restart
rotates BOTH bases through the SAME real ``Q_r``:

    Û = U_k Q_r ,   V̂ = V_k Q_r    — exact, because F(U Q) = F(U) Q for REAL Q.

A complex ``Q`` breaks that identity and with it the companion relation
``V = F(U)``; ``T_k`` being real symmetric is what guarantees it cannot happen.
This module builds ``T`` in float64 so the guarantee is STRUCTURAL rather than
asserted, and ``q_phase`` is the red twin that makes ``Q`` complex on purpose.

WHAT IS NOT HERE
----------------
No physics.  This module is L2: it knows about a real-linear operator pair, a
metric, residuals and a restart, and nothing about excitons, decks or ζ points.
It reads no environment variable — the dials arrive as arguments from the BSE
layer, which is where they belong.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax


# ═══════════════════════════════════════════════════════════════════════
#  Retrace instrument (same contract as solvers.thick_restart_lanczos)
# ═══════════════════════════════════════════════════════════════════════
#  Incremented INSIDE traced bodies, so one increment is one TRACE of that
#  body.  The fixed-shape claim is exactly the statement that these counts do
#  not depend on ``n_restarts``.  ``compile_cache_stats()`` cannot make that
#  statement -- with the persistent cache warm it reads 0 compiles no matter
#  how many distinct programs the loop dispatches.
TRACE_COUNTS: dict = {}


def _tally(name) -> None:
    TRACE_COUNTS[name] = TRACE_COUNTS.get(name, 0) + 1


def reset_trace_counts() -> None:
    TRACE_COUNTS.clear()


# ═══════════════════════════════════════════════════════════════════════
#  Cost accounting — stated, not guessed
# ═══════════════════════════════════════════════════════════════════════

def sdy_steps(m_max: int, n_keep: int, n_restarts: int) -> int:
    """Total SDY steps a run of this shape performs.  Exact, not sampled."""
    return int(m_max) + int(n_restarts) * (int(m_max) - int(n_keep))


def sdy_pair_applications(m_max: int, n_keep: int, n_restarts: int) -> int:
    """Total applications of the fused pair applier.

    Two per step (one ``G``, one ``F``), plus ONE for the start vector's
    κ-normalisation and ONE for the metric-symmetry certificate.  Each
    application is a half-operator PAIR: it applies both ``A`` and ``B`` once,
    at a fused cost the derivation prices at 1.09 TDA-matvec units, so an SDY
    step is 2.18 TMU.
    """
    return 2 + 2 * sdy_steps(m_max, n_keep, n_restarts)


# ═══════════════════════════════════════════════════════════════════════
#  Solver
# ═══════════════════════════════════════════════════════════════════════

def sdy_lanczos_eig(
    apply_F: Callable[[jax.Array], jax.Array],
    apply_G: Callable[[jax.Array], jax.Array],
    trailing_shape: tuple[int, ...],
    *,
    n_eig: int = 20,
    m_max: int = 105,
    n_keep: int | None = None,
    n_restarts: int = 8,
    seed: int = 42,
    dtype=jnp.complex128,
    X0: jax.Array | None = None,
    sharding=None,
    single_coeff_set: bool = False,
    q_phase: float = 0.0,
    announce: Callable[[str], None] | None = None,
):
    """Lowest ``n_eig`` excitation energies of a definite BSE Hamiltonian.

    Parameters
    ----------
    apply_F, apply_G : (b, *trailing) -> (b, *trailing)
        ``F(x) = A x + B conj(x)`` and ``G(v) = A v − B conj(v)``.  Both are
        REAL-linear, not complex-linear.  Called with ``b = 1`` only; the batch
        axis is kept so the callable is the same object the TDA solvers take and
        so the caller's sharding constraint applies unchanged.
    trailing_shape : tuple[int, ...]
        Shape of ONE vector (everything after the basis axis).
    n_eig, m_max, n_keep, n_restarts
        As ``solvers.thick_restart_lanczos``.  ``m_max`` is the memory cap; the
        squared problem's Krylov dimension runs ~1.5x the Hermitian one, so a
        TDA knee at 71 slots suggests starting near 105 here.
    X0 : (k, *trailing), optional
        Start vector; only ``X0[0]`` is used.  Seeding from converged TDA
        eigenvectors is strongly preferred to a random start — the coupling
        moves the lowest state by well under a meV, so the TDA vectors have
        near-perfect overlap with the low-``ω²`` sector.
    sharding : jax.sharding.Sharding, optional
        Applied to the preallocated ``(2, M, *trailing)`` basis.  The trailing
        axes should carry the caller's production sharding; nothing here
        reshapes them and no ``shard_map`` region is opened.
    single_coeff_set : bool
        **RED TWIN.**  Drop the imaginary coefficient set ``d``, enforcing only
        invariant (a).  Produces a run that looks healthy step by step and is
        wrong after the first restart.  Never true in production.
    q_phase : float
        **RED TWIN.**  Multiply the restart rotation ``Q`` by ``exp(i·q_phase)``,
        making it complex and breaking ``F(U Q) = V Q``.  Zero in production,
        and at zero it costs nothing (the branch is resolved in Python).
    announce : callable, optional
        Called ONCE at trace time with a one-line route description.  The
        tree's ``_announce_reorth`` convention: an A/B pair of logs must be
        distinguishable from the logs alone.

    Returns
    -------
    omega : (n_eig,) float64
        Excitation energies, ascending, in the operator's units.
    X, Y : (n_eig, *trailing)
        Excitation / de-excitation amplitudes, normalised to the tree's
        ``X^H X − Y^H Y = +1`` convention EXACTLY (closed form, §4.4 of the
        derivation: the lift satisfies ``X^H X − Y^H Y = 4θ`` identically, so
        dividing by ``2√θ`` is the normalisation — no numerical rescale).
    diag : dict
        Diagnostics.  Every one of these is a gate somewhere:

        ``metric_sym_err``  **THE operator-integrity detector.**  The relative
                          asymmetry of the κ-metric,
                          ``|Re(x1^H F x2) − Re(x2^H F x1)| / |·|``, measured
                          once on two random probes.  It is ~1e-15 for a
                          correct operator, and it is the EXACT precondition
                          the method needs: ``A`` Hermitian AND ``B`` complex
                          symmetric are jointly what make the metric symmetric
                          (derivation §3.2 point ii), so one number covers
                          both.  Measured to fire at 1e-1 for a 1e-3
                          non-Hermitian perturbation of ``A`` and at 2e-2 for a
                          1e-3 non-symmetric perturbation of ``B``.  Costs one
                          pair application, paid once.
        ``alpha_im_rel``  max ``|Im(v^H x)| / |v^H x|``.  **Reported, NOT a
                          gate — and this corrects the derivation.**  §4.6 of
                          NONTDA_MATRIXFREE_DERIVATION.md lists a nonzero value
                          here as evidence of a broken operator and prescribes
                          a 1e-4 assertion.  That is right in the TDA limit and
                          WRONG with coupling: the discarded piece is
                          ``Im(x̄^T B x̄)``, the imaginary part of a
                          complex-SYMMETRIC quadratic form, which has no
                          reality theorem.  Measured on a synthetic operator
                          with ``A`` exactly Hermitian and ``B`` exactly
                          symmetric it runs 3e-3…1e-2, and it collapses to
                          3e-17 only when ``B → 0``.  A 1e-4 gate here would be
                          a permanently-red cell measuring the coupling
                          strength.  Use ``metric_sym_err`` instead.
        ``imag_drift``    max over steps of ``|Im(U^H z)|/β`` measured AFTER the
                          first CGS2 pass.  ~1e-16 when both coefficient sets
                          are maintained; O(1) when only the real one is.  This
                          is the free detector for the ``single_coeff_set`` bug.
        ``kappa_start``   ``Re(u_0^H F(u_0))``.  **A value ≤ 0 is a certificate
                          that K is not positive definite** — triplet/charge
                          instability, imaginary excitations.  The caller must
                          REFUSE on it, not clamp.
        ``beta_sq_min``   min over steps of ``Re(x^H y) = ‖x‖_κ²``.  Same
                          certificate, seen from inside the iteration.
        ``orth_err``      max ``|Re(U^H V) − I|`` on the returned Ritz block —
                          invariant (a).
        ``im_uu``, ``im_vv``  max ``|Im(U^H U)|``, ``|Im(V^H V)|`` — invariant (b).
        ``resid``         ``ρ|b_i|``, the projected residual bound of
                          arXiv:2503.20920 §5.  Free: no extra matvec.
        ``theta_sq``      the raw eigenvalues of ``T`` (these are ``ω²``).
    """
    if n_keep is None:
        n_keep = n_eig + 10
    m_max = int(m_max)
    n_keep = int(n_keep)
    n_eig = int(n_eig)
    n_restarts = int(n_restarts)
    if not (0 < n_eig <= n_keep < m_max):
        raise ValueError(
            f"sdy_lanczos_eig: need 0 < n_eig ({n_eig}) <= n_keep ({n_keep}) "
            f"< m_max ({m_max})")

    tr = tuple(int(t) for t in trailing_shape)
    M = m_max + 1                      # +1 slot for the residual vector
    rdtype = jnp.zeros((), dtype=dtype).real.dtype

    if announce is not None:
        announce(
            f"[sdy] structure-preserving Lanczos (SDY Alg. 4, shao variant): "
            f"n_eig={n_eig} m_max={m_max} n_keep={n_keep} "
            f"n_restarts={n_restarts} | reorth=cgs2 "
            f"coeff_sets={'ONE (RED TWIN)' if single_coeff_set else 'two'} "
            f"| steps={sdy_steps(m_max, n_keep, n_restarts)} "
            f"pair_applications="
            f"{sdy_pair_applications(m_max, n_keep, n_restarts)}"
            + ("  [RED TWIN: complex restart Q]" if q_phase else ""))

    # ── start vector ───────────────────────────────────────────────────
    if X0 is not None:
        u0 = jnp.asarray(X0[0], dtype=dtype)
    else:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        u0 = (jax.random.normal(k1, tr, dtype=jnp.float64)
              + 1j * jax.random.normal(k2, tr, dtype=jnp.float64)).astype(dtype)
    u0 = u0 / jnp.sqrt(jnp.sum(jnp.abs(u0) ** 2))

    def _apply(op, z):
        """One half-operator PAIR through the caller's batched applier."""
        return op(z[None, ...])[0]

    # κ-normalise: ⟨u,u⟩_κ = Re(u^H F(u)) = 1.  One pair application, paid once.
    v0 = _apply(apply_F, u0)
    kappa0 = jnp.sum(jnp.conj(u0) * v0).real
    kscale = 1.0 / jnp.sqrt(jnp.maximum(kappa0, jnp.asarray(1e-300, rdtype)))

    # The metric-symmetry certificate — the precondition of the whole method,
    # bought for one pair application.  ⟨x,x'⟩_κ = Re(x^H F(x')) is symmetric
    # IFF A is Hermitian and B is complex symmetric, so this single number
    # certifies both.  It is NOT the same thing as Im(v^H G(v)), which is
    # generically nonzero for a perfectly correct operator (see ``alpha_im_rel``).
    kp = jax.random.PRNGKey(seed + 977)
    kp1, kp2 = jax.random.split(kp)
    xprobe = (jax.random.normal(kp1, tr, dtype=jnp.float64)
              + 1j * jax.random.normal(kp2, tr, dtype=jnp.float64)).astype(dtype)
    vprobe = _apply(apply_F, xprobe)
    s_ab = jnp.sum(jnp.conj(u0) * vprobe).real         # Re(u0^H F(xprobe))
    s_ba = jnp.sum(jnp.conj(xprobe) * v0).real         # Re(xprobe^H F(u0))
    metric_sym_err = (jnp.abs(s_ab - s_ba)
                      / jnp.maximum(jnp.maximum(jnp.abs(s_ab), jnp.abs(s_ba)),
                                    jnp.asarray(1e-300, rdtype)))

    UV0 = (jnp.zeros((2, M) + tr, dtype=dtype)
           .at[0, 0].set(u0 * kscale.astype(dtype))
           .at[1, 0].set(v0 * kscale.astype(dtype)))
    if sharding is not None:
        UV0 = lax.with_sharding_constraint(UV0, sharding)

    # ── reorthogonalisation: CGS2 with the two coefficient sets ────────
    def _reorth(UV, z, sel):
        """Two classical Gram-Schmidt passes; ONE all-reduce per pass.

        Returns ``(z, drift)`` where ``drift = max|Im(U^H z)|`` measured on the
        SECOND pass, i.e. after the first correction has already been applied.
        With both coefficient sets that residue is at round-off; with only the
        real set it is the whole of the first pass's ``Im(U^H z)``, undiminished.
        """
        def _pass(zz):
            # Contracts every SHARDED trailing axis -> exactly one psum, and
            # the (2, ...) leading axis rides along for free: ONE collective
            # carries both Gram products.
            h = jnp.einsum('pm...,...->pm', jnp.conj(UV), zz, optimize=True)
            c = jnp.where(sel, h[1].real, jnp.zeros((), rdtype))   # real     (a)
            d = jnp.where(sel, h[0].imag, jnp.zeros((), rdtype))   # Im(U^H z) (b)
            dd = jnp.zeros_like(d) if single_coeff_set else d
            coef = jnp.stack([c.astype(dtype),
                              (1j * dd).astype(dtype)])            # (2, M)
            # Sums over the REPLICATED basis axes -> no collective.
            return zz - jnp.einsum('pm,pm...->...', coef, UV, optimize=True), d
        z1, _ = _pass(z)
        z2, d2 = _pass(z1)
        return z2, jnp.max(jnp.abs(d2))

    def _window(j):
        return jnp.arange(M) <= j

    # ── one SDY step: two fused pair applications ──────────────────────
    def _step(j, carry):
        _tally('sdy_step')
        UV, alpha, beta, aim, adn, imd, b2min = carry
        uj = UV[0, j]
        vj = UV[1, j]

        x = _apply(apply_G, vj)                          # G(v_j)   — pair #1
        # ONE complex dot.  .real drives the recurrence; .imag is the free
        # operator-integrity residual (A Hermitian and B symmetric make it zero
        # in exact arithmetic at every step, converged or not).
        a_c = jnp.sum(jnp.conj(vj) * x)
        alpha = alpha.at[j].set(a_c.real)
        aim = aim.at[j].set(jnp.abs(a_c.imag))
        adn = adn.at[j].set(jnp.abs(a_c))
        x = x - a_c.real.astype(dtype) * uj

        # Full reorthogonalisation subsumes the −β_{j−1} u_{j−1} term AND, after
        # a restart, the arrowhead couplings to the retained Ritz block — which
        # is why thick restart REQUIRES it rather than merely benefiting.
        x, drift = _reorth(UV, x, _window(j))
        imd = imd.at[j].set(drift)

        y = _apply(apply_F, x)                           # F(x)     — pair #2
        # β_j = ‖x‖_κ.  Real and positive whenever K ≻ 0; a negative value here
        # IS the indefiniteness surfacing, not a numerical artefact.
        b2 = jnp.sum(jnp.conj(x) * y).real
        b2min = jnp.minimum(b2min, b2)
        b = jnp.sqrt(jnp.maximum(b2, jnp.zeros((), rdtype)))
        beta = beta.at[j].set(b)
        inv = (1.0 / jnp.maximum(b, jnp.asarray(1e-300, rdtype))).astype(dtype)
        # u_{j+1} = x/β and v_{j+1} = y/β: a REAL scaling, so F(u_{j+1}) =
        # v_{j+1} holds exactly and the companion relation never drifts.
        UV = UV.at[0, j + 1].set(x * inv).at[1, j + 1].set(y * inv)
        return (UV, alpha, beta, aim, adn, imd, b2min)

    # ── the projected matrix: REAL symmetric, (m_max, m_max) ───────────
    def _build_T(alpha, beta, theta_k, s_arrow, first: bool):
        """Arrowhead + tridiagonal tail, all-static slicing, float64.

        Building this in float64 rather than complex is the structural
        guarantee that the restart rotation is real (module docstring).
        """
        _tally('build_T_first' if first else 'build_T_restart')
        T = jnp.zeros((m_max, m_max), dtype=jnp.float64)
        if first:
            T = T + jnp.diag(alpha)
            off = beta[:m_max - 1]
            return T + jnp.diag(off, 1) + jnp.diag(off, -1)
        idx = jnp.arange(n_keep)
        T = T.at[idx, idx].set(theta_k)                  # retained Ritz values
        T = T.at[:n_keep, n_keep].set(s_arrow)           # the arrow …
        T = T.at[n_keep, :n_keep].set(s_arrow)           # … symmetric: s is REAL
        tail = jnp.arange(n_keep, m_max)
        T = T.at[tail, tail].set(alpha[n_keep:m_max])
        tl = jnp.arange(n_keep, m_max - 1)
        off = beta[n_keep:m_max - 1]
        T = T.at[tl, tl + 1].set(off)
        T = T.at[tl + 1, tl].set(off)
        return T

    def _restart(UV, T, beta_last):
        """Rotate BOTH bases onto the lowest n_keep Ritz vectors, through the
        SAME real Q.  Returns (basis, retained θ², arrow couplings)."""
        _tally('restart')
        T = 0.5 * (T + T.T)
        theta, Y = jnp.linalg.eigh(T)          # float64 in, float64 out: Q REAL
        theta_k = theta[:n_keep]
        Y_k = Y[:, :n_keep]                    # (m_max, n_keep), real
        Yc = Y_k.astype(dtype)
        if q_phase:
            # RED TWIN ONLY.  A complex Q breaks F(U Q) = V Q, because F is
            # real-linear: the companion basis stops being F of the first one
            # and every subsequent α, β is measured against the wrong metric.
            Yc = Yc * jnp.exp(1j * jnp.asarray(q_phase, rdtype)).astype(dtype)
        UVk = jnp.einsum('mn,pm...->pn...', Yc, UV[:, :m_max], optimize=True)
        UV_new = (jnp.zeros_like(UV)
                  .at[:, :n_keep].set(UVk)
                  .at[:, n_keep].set(UV[:, m_max]))
        s_arrow = beta_last * Y_k[m_max - 1, :]
        return UV_new, theta_k, s_arrow

    # ── cold cycle ─────────────────────────────────────────────────────
    alpha = jnp.zeros((m_max,), dtype=jnp.float64)
    beta = jnp.zeros((m_max,), dtype=jnp.float64)
    aim = jnp.zeros((m_max,), dtype=jnp.float64)
    adn = jnp.zeros((m_max,), dtype=jnp.float64)
    imd = jnp.zeros((m_max,), dtype=jnp.float64)
    b2min = jnp.asarray(jnp.inf, dtype=jnp.float64)

    carry = lax.fori_loop(0, m_max, _step,
                          (UV0, alpha, beta, aim, adn, imd, b2min))
    UV, alpha, beta, aim, adn, imd, b2min = carry
    T = _build_T(alpha, beta, None, None, first=True)
    UV, theta_k, s_arrow = _restart(UV, T, beta[m_max - 1])

    # ── restart cycles: identical program, run n_restarts times ────────
    def _cycle(_c, carry):
        UV, theta_k, s_arrow, alpha, beta, aim, adn, imd, b2min = carry
        UV, alpha, beta, aim, adn, imd, b2min = lax.fori_loop(
            n_keep, m_max, _step, (UV, alpha, beta, aim, adn, imd, b2min))
        T = _build_T(alpha, beta, theta_k, s_arrow, first=False)
        UV, theta_k, s_arrow = _restart(UV, T, beta[m_max - 1])
        return (UV, theta_k, s_arrow, alpha, beta, aim, adn, imd, b2min)

    if n_restarts > 0:
        (UV, theta_k, s_arrow, alpha, beta, aim, adn, imd,
         b2min) = lax.fori_loop(
            0, n_restarts, _cycle,
            (UV, theta_k, s_arrow, alpha, beta, aim, adn, imd, b2min))

    # ── eigenpairs.  After the last restart UV[:, :n_keep] ARE the Ritz
    #    vectors and theta_k their values (= ω²), ascending from eigh. ──
    theta = jnp.maximum(theta_k[:n_eig], jnp.zeros((), jnp.float64))
    omega = jnp.sqrt(theta)
    p = UV[0, :n_eig]
    q = UV[1, :n_eig]
    om = omega.astype(dtype).reshape((n_eig,) + (1,) * len(tr))
    # X = θp + q,  Y = conj(θp − q),  then ÷ 2√θ.  Exact: the lift satisfies
    # X^H X − Y^H Y = 4θ·Re(p^H q) = 4θ identically, using Re(U^H V) = I.
    nrm = (0.5 / jnp.sqrt(jnp.maximum(
        omega, jnp.asarray(1e-300, jnp.float64)))).astype(dtype).reshape(
            (n_eig,) + (1,) * len(tr))
    Xout = (om * p + q) * nrm
    Yout = jnp.conj(om * p - q) * nrm

    # ── invariants on the returned block, and the free residual ────────
    U_r = UV[0, :n_keep]
    V_r = UV[1, :n_keep]
    G_uv = jnp.einsum('m...,n...->mn', jnp.conj(U_r), V_r, optimize=True)
    G_uu = jnp.einsum('m...,n...->mn', jnp.conj(U_r), U_r, optimize=True)
    G_vv = jnp.einsum('m...,n...->mn', jnp.conj(V_r), V_r, optimize=True)
    eye = jnp.eye(n_keep, dtype=jnp.float64)
    # ρ = ‖[u_{k+1}; ū_{k+1}]‖₂ = √2‖u_{k+1}‖₂; slot n_keep holds u_{k+1}.
    rho = jnp.sqrt(2.0 * jnp.sum(jnp.abs(UV[0, n_keep]) ** 2))

    diag = {
        "metric_sym_err": metric_sym_err,
        "alpha_im_rel": jnp.max(aim / jnp.maximum(adn, 1e-300)),
        "imag_drift": jnp.max(imd / jnp.maximum(beta, 1e-300)),
        "kappa_start": kappa0,
        "beta_sq_min": b2min,
        "orth_err": jnp.max(jnp.abs(G_uv.real - eye)),
        "im_uu": jnp.max(jnp.abs(G_uu.imag)),
        "im_vv": jnp.max(jnp.abs(G_vv.imag)),
        "resid": rho * jnp.abs(s_arrow[:n_eig]),
        "theta_sq": theta_k,
        "alpha": alpha,
        "beta": beta,
    }
    return omega, Xout, Yout, diag
