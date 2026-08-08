"""Gate: the block-Lanczos chain is canonical AND bit-identical to the eager form.

``w_omega_chain.build_w_omega_chain`` used to be a Python loop over chain steps
with a growing DGKS double loop inside it — 2306 XLA dispatches and 65 blocking
host syncs for a 32-step chain, measured on the MoS2 6x6 deck
(BSE_CODE_SURVEY R1).  It is now one cached program per step: 33 dispatches and
33 syncs at the same chain length.

This file is the verification floor for that conversion, and every cell here
ships the case in which it returns FALSE:

  * :func:`_eager_reference_chain` is the OLD implementation, verbatim.  The
    bit-identity cell compares against it with ``np.array_equal`` — not
    ``allclose`` — because preserving the arithmetic exactly is the whole
    reason the ``p x p`` eigendecomposition stayed on host and the reason the
    ``lax.optimization_barrier`` calls in the step exist.
  * The trace-once cell has a red twin that bakes the chain index into the
    program (the shape the conversion had to avoid) and asserts it traces once
    PER STEP instead of once.
  * The cache cell has a red twin that clears ``_CHAIN_STEP_CACHE`` and asserts
    the second build then has to build a new program — the cache boundary the
    pattern note records as load-bearing (an uncached scan measured 3.4x SLOWER
    than the loop it replaced).
  * The barrier cell has a red twin that neuters ``_ob`` to the identity and
    asserts the chain then STOPS being bit-identical, so nobody deletes a
    barrier believing it is decoration.

The operator here is synthetic (a Hermitian (c,c) contraction plus the
transition diagonal) so the cell is fast and needs no GW fixture; the physics
gate against the shifted-solve oracle is ``test_bse_w_omega_chain.py``.
"""
from functools import partial

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from bse.bse_ring_comm import make_bse_shardings
from bse import w_omega_chain as woc

# Shape: small enough to be a unit test, large enough that the (c,c) and
# (p,p) contractions are real GEMMs — the barrier red twin below is a claim
# about XLA's fusion of those dots, and it has nothing to bite on at toy size.
NC, NV, NK, NMU, P_BLK = 48, 24, 16, 32, 4
CHAIN_M = 6


# ---------------------------------------------------------------------------
# a synthetic screening operator with the production call signatures
# ---------------------------------------------------------------------------
def _fixture(mesh):
    """``(data, matvec, gen, sh)`` for a synthetic Hermitian S."""
    sh = make_bse_shardings(mesh)
    rng = np.random.default_rng(7)

    def _c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    # Hermitian (c,c) coupling -> (A+B) is Hermitian in the Euclidean inner
    # product over (c,v,k), which is what the chain's block Lanczos assumes.
    A = _c(NC, NC)
    A = 0.5 * (A + A.conj().T)
    A_d = jnp.asarray(A / (4.0 * NC))

    eps_c = jnp.asarray(rng.uniform(1.0, 3.0, size=(NK, NC)))
    eps_v = jnp.asarray(rng.uniform(-3.0, -1.0, size=(NK, NV)))
    V_q0 = jnp.asarray(np.eye(NMU))
    # The seed vertex: (p, n_rmu, nk) probe -> (p, c, v, k) pair block.
    Gmat = jnp.asarray(_c(NMU, NC * NV) / np.sqrt(NMU))

    data = {
        "eps_c": eps_c, "eps_v": eps_v, "V_q0": V_q0,
        "nkx": NK, "nky": 1, "nkz": 1,
        "psi_c_X": jnp.zeros(1), "psi_c_Y": jnp.zeros(1),
        "psi_v_X": jnp.zeros(1), "psi_v_Y": jnp.zeros(1),
        "W_R": jnp.zeros(1), "M_X": jnp.zeros(1), "M_Y": jnp.zeros(1),
        "n_rmu": NMU,
    }

    @jax.jit
    def gen(r, psi_c_X, psi_v_X, V_q0_):
        # r: (p, n_rmu, nk) -> (p, c, v, k)
        f = jnp.einsum("pmk,mn->pnk", r.astype(jnp.complex128), Gmat)
        return f.reshape(r.shape[0], NC, NV, NK).astype(jnp.complex128)

    @jax.jit
    def matvec(uu, *args):
        # uu: (2, p, c, v, k).  Returns [(A+B)U ; -(A+B)U] for U = uu[0].
        U = uu[0]
        w = jnp.einsum("pcvk,cd->pdvk", U, A_d) + 0.25 * U
        return jnp.stack([w, -w], axis=0)

    return data, matvec, gen, sh


# ---------------------------------------------------------------------------
# THE OLD IMPLEMENTATION, VERBATIM — the red twin for every cell below
# ---------------------------------------------------------------------------
def _eager_reference_chain(data, matvec, gen, sh, cols, chain_len,
                           *, reorth_passes=2, counter=None):
    """``build_w_omega_chain`` as it was before the conversion.

    Kept here, in the test, because a claim of bit-identity is only worth
    something if the thing it is identical TO is executable.  ``counter`` (a
    one-element list) is bumped once per XLA dispatch, which is what makes the
    O(m^2)-vs-O(m) cell measurable rather than asserted.
    """
    def _tick():
        if counter is not None:
            counter[0] += 1

    def _gram(A, B):
        _tick()
        return woc._block_gram(A, B)

    def _combine(Q, M):
        _tick()
        return woc._block_combine(Q, M)

    def _orth(Wblk):
        G = np.asarray(jax.device_get(_gram(Wblk, Wblk)))
        R, Tr = woc._host_qr_factors(G)
        Q = jax.lax.with_sharding_constraint(
            _combine(Wblk, jnp.asarray(Tr)), sh.X)
        return Q, R

    cols = np.asarray(cols, dtype=int)
    p = len(cols)
    eps_c, eps_v = data["eps_c"], data["eps_v"]
    delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
    D_half = jax.lax.with_sharding_constraint(
        jnp.sqrt(jnp.clip(delta_E.real, 0.0, None)).astype(jnp.complex128), sh.X)
    args = (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
            data["eps_c"], data["eps_v"], data["W_R"], data["V_q0"],
            data["M_X"], data["M_Y"])

    def apply_S(U):
        _tick()
        u = jax.lax.with_sharding_constraint(D_half * U, sh.X)
        uu = jax.lax.with_sharding_constraint(
            jnp.stack([u, u], axis=0).astype(jnp.complex128), sh.X_full)
        w = matvec(uu, *args)[0]
        return jax.lax.with_sharding_constraint(D_half * w, sh.X)

    B0 = woc._seed_block(cols, data, gen, sh)
    B0 = jax.lax.with_sharding_constraint(D_half * B0, sh.X)
    Q0, R0 = _orth(B0)

    Qs = [Q0]
    alphas, betas = [], []
    zero_pp = jnp.zeros((p, p), dtype=jnp.complex128)
    Qprev = jax.lax.with_sharding_constraint(jnp.zeros_like(Q0), sh.X)
    beta_prev = zero_pp

    for j in range(chain_len):
        Wb = apply_S(Qs[j])
        alpha = _gram(Qs[j], Wb)
        Wb = Wb - _combine(Qs[j], alpha) - _combine(Qprev, beta_prev.conj().T)
        for _ in range(int(reorth_passes)):
            for Qi in Qs:
                Wb = Wb - _combine(Qi, _gram(Qi, Wb))
        Wb = jax.lax.with_sharding_constraint(Wb, sh.X)
        Qn, beta = _orth(Wb)
        alphas.append(np.asarray(jax.device_get(alpha)))
        betas.append(beta)
        beta_prev = jnp.asarray(beta)
        Qprev = Qs[j]
        Qs.append(Qn)

    m = chain_len
    V_stack = jax.lax.with_sharding_constraint(
        jnp.stack(Qs[:m], axis=0), sh.X_full)
    return {"alpha": np.stack(alphas, axis=0), "beta": np.stack(betas, axis=0),
            "R0": np.asarray(R0), "V_stack": V_stack, "D_half": D_half,
            "cols": cols, "m": m, "p": p}


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chain_fixture():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data, matvec, gen, sh = _fixture(mesh)
    return data, matvec, gen, sh, np.arange(P_BLK, dtype=int)


def _local(x):
    """Process-local bytes of a jax.Array (P>1-safe)."""
    shards = getattr(x, "addressable_shards", None)
    if not shards:
        return np.asarray(x)
    return np.stack([np.asarray(jax.device_get(s.data)) for s in shards], axis=0)


def _cached_step():
    """The one cached step program, or an assertion failure if there is not one."""
    assert len(woc._CHAIN_STEP_CACHE) == 1, (
        f"expected exactly one cached step program, found "
        f"{len(woc._CHAIN_STEP_CACHE)}")
    return next(iter(woc._CHAIN_STEP_CACHE.values()))[-1]


def _cache_size(fn):
    """Traced signatures of a jax.jit callable, or None if this jax hides it."""
    getter = getattr(fn, "_cache_size", None)
    return int(getter()) if callable(getter) else None


# ---------------------------------------------------------------------------
# 1. BIT-IDENTITY vs the eager form.  np.array_equal, not allclose.
# ---------------------------------------------------------------------------
def test_chain_is_bit_identical_to_the_eager_reference(chain_fixture):
    data, matvec, gen, sh, cols = chain_fixture
    new = woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    ref = _eager_reference_chain(data, matvec, gen, sh, cols, CHAIN_M)

    for name in ("alpha", "beta", "R0"):
        assert np.array_equal(new[name], ref[name]), (
            f"{name} is not bit-identical to the eager reference: "
            f"max|d|={np.abs(new[name] - ref[name]).max():.3e}")
    a, b = _local(new["V_stack"]), _local(ref["V_stack"])
    assert np.array_equal(a, b), (
        f"V_stack is not bit-identical to the eager reference: "
        f"max|d|={np.abs(a - b).max():.3e}")
    assert new["V_stack"].sharding.is_equivalent_to(
        ref["V_stack"].sharding, new["V_stack"].ndim), (
        f"V_stack sharding drifted: {new['V_stack'].sharding.spec} vs "
        f"{ref['V_stack'].sharding.spec}")
    # Non-vacuous: a chain of zeros would satisfy every equality above.
    assert np.abs(new["alpha"]).max() > 1e-6
    assert np.abs(_local(new["V_stack"])).max() > 1e-6


# ---------------------------------------------------------------------------
# 2. THE BARRIER RED TWIN.  Neuter _ob and bit-identity must BREAK.
# ---------------------------------------------------------------------------
def test_removing_the_optimization_barriers_breaks_bit_identity(
        chain_fixture, monkeypatch):
    """The barriers are load-bearing, and this is the cell that proves it.

    If this ever passes — i.e. the chain stays bit-identical with the barriers
    gone — then XLA stopped fusing the dot that a subtract feeds, and the
    barriers could be dropped for a real speedup (MEASURED on the MoS2 6x6
    deck at 1 GPU: 5.34 s with barriers, 3.83 s without).  That is a decision
    to take deliberately with a fresh A/B, not a line to delete.
    """
    data, matvec, gen, sh, cols = chain_fixture
    ref = _eager_reference_chain(data, matvec, gen, sh, cols, CHAIN_M)

    monkeypatch.setattr(woc, "_ob", lambda x: x)
    monkeypatch.setattr(woc, "_CHAIN_STEP_CACHE", {})   # force a fresh trace
    unbarriered = woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)

    same = (np.array_equal(unbarriered["alpha"], ref["alpha"])
            and np.array_equal(unbarriered["beta"], ref["beta"])
            and np.array_equal(_local(unbarriered["V_stack"]),
                               _local(ref["V_stack"])))
    assert not same, (
        "the chain is bit-identical to the eager form even with the "
        "optimization barriers removed — see this test's docstring: that is a "
        "genuine change in XLA's behaviour, not a licence to delete them "
        "silently.  Re-run the A/B on a production deck and record it.")
    # ...and the damage is ulp-scale, not a broken algorithm: this is what
    # makes it a PHYSICS-ADJACENT drift rather than a bug.
    d = np.abs(unbarriered["alpha"] - ref["alpha"]).max()
    scale = np.abs(ref["alpha"]).max()
    assert d / scale < 1e-6, (
        f"unbarriered chain differs from the eager form by {d/scale:.3e} "
        f"relative — that is far more than fusion reassociation and means "
        f"something other than the barriers changed")


# ---------------------------------------------------------------------------
# 3. TRACE-ONCE, with the red twin that bakes the chain index in.
# ---------------------------------------------------------------------------
def test_chain_step_traces_once_for_the_whole_chain(chain_fixture):
    data, matvec, gen, sh, cols = chain_fixture
    woc._CHAIN_STEP_CACHE.clear()
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)

    step = _cached_step()
    n = _cache_size(step)
    if n is None:
        pytest.skip("this jax does not expose jit._cache_size()")
    assert n == 1, (
        f"the step program was traced {n} times for a {CHAIN_M}-step chain; "
        f"the chain index j must stay a RUNTIME operand, not a static one")


def test_red_twin_static_chain_index_traces_once_per_step(chain_fixture):
    """The shape the conversion had to avoid: ``j`` baked into the program.

    Same body, same buffer, same DGKS — but ``j`` static, so ``fori_loop``'s
    bound is a Python int and every step is its own signature.  This is the
    FALSE case of the cell above; if it ever stops tracing once per step the
    cell above has stopped measuring anything.
    """
    data, matvec, gen, sh, cols = chain_fixture
    _, _, _, sh_, _ = data, matvec, gen, sh, cols

    @partial(jax.jit, static_argnums=(2,))
    def _static_step(V, W, j_static: int):
        def _body(i, Wc):
            Qi = jax.lax.dynamic_index_in_dim(V, i, axis=0, keepdims=False)
            return Wc - woc._block_combine(Qi, woc._block_gram(Qi, Wc))
        return jax.lax.fori_loop(0, j_static, _body, W)

    p = len(cols)
    V = jnp.zeros((CHAIN_M, p, NC, NV, NK), dtype=jnp.complex128)
    W = jnp.ones((p, NC, NV, NK), dtype=jnp.complex128)
    for j in range(CHAIN_M):
        _static_step(V, W, j + 1)
    n = _cache_size(_static_step)
    if n is None:
        pytest.skip("this jax does not expose jit._cache_size()")
    assert n == CHAIN_M, (
        f"the static-index twin traced {n} times, not {CHAIN_M} — it is no "
        f"longer the red twin of the trace-once cell")


# ---------------------------------------------------------------------------
# 4. THE CACHE BOUNDARY, with the red twin that disables it.
# ---------------------------------------------------------------------------
def test_second_build_reuses_the_cached_step(chain_fixture):
    data, matvec, gen, sh, cols = chain_fixture
    woc._CHAIN_STEP_CACHE.clear()
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    first = _cached_step()
    n_before = _cache_size(first)

    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    assert len(woc._CHAIN_STEP_CACHE) == 1, (
        "a second build at the same signature added a second step program")
    again = _cached_step()
    assert again is first, "the second build did not reuse the cached step"
    if n_before is not None:
        assert _cache_size(first) == n_before, (
            f"the reused step was re-traced ({n_before} -> "
            f"{_cache_size(first)} signatures)")


def test_red_twin_disabling_the_cache_rebuilds_the_step(chain_fixture):
    """Clearing the cache between builds must cost a NEW program.

    This is the cell that would go red if ``_get_chain_step`` stopped caching:
    the pattern note records an uncached scan measuring 3.4x SLOWER than the
    loop it replaced, so the cache boundary is the load-bearing part of the
    conversion and not an optimisation on top of it.
    """
    data, matvec, gen, sh, cols = chain_fixture
    woc._CHAIN_STEP_CACHE.clear()
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    first = _cached_step()

    woc._CHAIN_STEP_CACHE.clear()               # the disabled-cache arm
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    second = _cached_step()
    assert second is not first, (
        "clearing _CHAIN_STEP_CACHE returned the same step object — the cache "
        "is not where the reuse comes from, so the reuse cell proves nothing")


def test_cache_key_separates_chain_length_and_block_width(chain_fixture):
    """A different chain geometry must NOT hit another geometry's program."""
    data, matvec, gen, sh, cols = chain_fixture
    woc._CHAIN_STEP_CACHE.clear()
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M + 1)
    woc.build_w_omega_chain(data, matvec, gen, sh, cols[:2], CHAIN_M)
    assert len(woc._CHAIN_STEP_CACHE) == 3, (
        f"three distinct chain geometries produced "
        f"{len(woc._CHAIN_STEP_CACHE)} cached programs")


# ---------------------------------------------------------------------------
# 5. DISPATCH COUNT: linear in m, with the eager O(m^2) form as the twin.
# ---------------------------------------------------------------------------
def test_dispatch_count_is_linear_in_chain_length(chain_fixture):
    data, matvec, gen, sh, cols = chain_fixture
    counter = [0]

    woc._CHAIN_STEP_CACHE.clear()
    orig_gram, orig_comb = woc._block_gram, woc._block_combine
    orig_get = woc._get_chain_step

    def _count(fn):
        # Trace-time calls are not dispatches: inside the step's own trace the
        # leaves are called with Tracers and inlined into ONE program.  Counting
        # them would make a cold build look like the eager form it replaced.
        def _f(*a, **k):
            if not any(isinstance(x, jax.core.Tracer) for x in a):
                counter[0] += 1
            return fn(*a, **k)
        return _f

    try:
        woc._block_gram = _count(orig_gram)
        woc._block_combine = _count(orig_comb)
        def _counted_factory(*a, **k):
            init, step = orig_get(*a, **k)
            return _count(init), _count(step)
        woc._get_chain_step = _counted_factory
        woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
        n_new = counter[0]
    finally:
        woc._block_gram, woc._block_combine = orig_gram, orig_comb
        woc._get_chain_step = orig_get

    # 1 seed Gram + 1 carry init + m step dispatches.  Everything else — the
    # matvec, the recurrence, the whole DGKS double loop — is inside the program.
    assert n_new == CHAIN_M + 2, (
        f"the converted chain issued {n_new} dispatches for a {CHAIN_M}-step "
        f"chain, expected {CHAIN_M + 2} (1 seed Gram + 1 init + 1 per step)")

    # The red twin: the eager form on the SAME inputs, counted the same way.
    ref_counter = [0]
    _eager_reference_chain(data, matvec, gen, sh, cols, CHAIN_M,
                           counter=ref_counter)
    # 2 + 6m + 2*reorth*sum_{j<m}(j+1)  -> quadratic in m.
    expected_ref = 2 + 6 * CHAIN_M + 2 * 2 * (CHAIN_M * (CHAIN_M + 1) // 2)
    assert ref_counter[0] == expected_ref, (
        f"the eager reference issued {ref_counter[0]} dispatches, expected "
        f"{expected_ref} — the red twin no longer reproduces the old form")
    assert ref_counter[0] > 5 * n_new, (
        f"the eager form ({ref_counter[0]}) is not dramatically worse than "
        f"the converted one ({n_new}); at m={CHAIN_M} it should be")


def test_host_sync_count_is_one_per_step(chain_fixture):
    """One blocking host sync per step, down from two.

    The chain cannot reach zero: the ``p x p`` eigendecomposition stays on
    host on purpose (:func:`w_omega_chain._host_qr_factors`), so each step must
    fetch its Gram.  What the conversion removed is the SECOND fetch — alpha
    now rides back with the Gram in one ``device_get``.
    """
    data, matvec, gen, sh, cols = chain_fixture
    woc._CHAIN_STEP_CACHE.clear()
    n = [0]
    orig = jax.device_get

    def _counting(x):
        n[0] += 1
        return orig(x)

    try:
        jax.device_get = _counting
        woc.build_w_omega_chain(data, matvec, gen, sh, cols, CHAIN_M)
    finally:
        jax.device_get = orig
    assert n[0] == CHAIN_M + 1, (
        f"{n[0]} host syncs for a {CHAIN_M}-step chain, expected "
        f"{CHAIN_M + 1} (1 seed + 1 per step)")


# ---------------------------------------------------------------------------
# 6. The eigendecomposition stays on HOST.  A physics guard, not a perf one.
# ---------------------------------------------------------------------------
def test_the_block_qr_eigendecomposition_stays_on_host():
    """``jnp.linalg.eigh`` must not appear in this module.

    Moving the ``p x p`` Hermitian eigendecomposition onto the device backend
    changes the eigenvectors in the last bits, and those eigenvectors feed
    every later block of the chain — so it is a change to a physics quantity's
    numerical path, not a dispatch-hygiene change, and it does not belong
    inside a performance refactor.  It may well be the right thing to do one
    day; it needs its own A/B and its own ruling.
    """
    import ast
    import inspect
    # The AST, not the text: this module's own docstrings NAME jnp.linalg.eigh
    # to explain why it is not used, and a grep cannot tell prose from a call.
    tree = ast.parse(inspect.getsource(woc))
    called = {ast.unparse(n.func) for n in ast.walk(tree)
              if isinstance(n, ast.Call)}
    device_eigh = sorted(c for c in called
                         if c.endswith("linalg.eigh") and not c.startswith("np."))
    assert not device_eigh, (
        f"w_omega_chain now calls a device eigh {device_eigh}; see this test's "
        f"docstring")
    assert "np.linalg.eigh" in called, (
        "the host eigendecomposition disappeared from w_omega_chain")
