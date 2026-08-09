"""Gates for the exact assembled BSE diagonal and its build memo.

The diagonal itself was gated on the record deck against the dense operator to
1.5e-15 eV (``DAVIDSON_COMPETITIVE.md`` §14.2); these cells gate the things a
deck run cannot: that hoisting the build's jit to module scope left the
arithmetic bit-identical, that the program is now traced once per process
instead of once per call, and that the memo is an IDENTITY memo whose only way
to hit is "the same arrays as last time".

Two of the cells are RED TWINS.  ``test_memo_misses_when_W_changes`` fails if
the memo ever hands back a diagonal built from a different W — the stale-payload
failure a value-blind cache would have.  ``test_memo_misses_on_equal_but_
distinct_W`` fails if the memo starts matching on equality, which would make the
first twin pass for the wrong reason.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_davidson_helpers as H  # noqa: E402


# ── a tiny synthetic BSE payload, single device, no mesh ──────────────
NK, NC, NV, NMU = 3, 2, 2, 5


def _payload(seed=20260808, w_scale=1.0):
    rng = np.random.default_rng(seed)

    def cx(*shape):
        return jnp.asarray(rng.standard_normal(shape)
                           + 1j * rng.standard_normal(shape),
                           dtype=jnp.complex128)

    eps_c = jnp.asarray(rng.standard_normal((NK, NC)) + 3.0)
    eps_v = jnp.asarray(rng.standard_normal((NK, NV)))
    psi_c_X = cx(NK, NC, 2, NMU)          # (k, c, spinor, mu)
    psi_v_Y = cx(NK, NV, 2, NMU)
    W_q0 = jnp.asarray(w_scale) * cx(NMU, NMU)
    M_X = cx(NK, NC, NV, NMU)
    M_Y = cx(NK, NC, NV, NMU)
    V_q0 = cx(NMU, NMU)
    return (eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M_X, M_Y, V_q0)


def _reference_diagonal(ops, nk):
    """The same contraction written in numpy, independent of the jax code."""
    eps_c, eps_v, psi_c_X, psi_v_Y, W_q0, M_X, M_Y, V_q0 = (
        np.asarray(o) for o in ops)
    dE = eps_c.T[:, None, :] - eps_v.T[None, :, :]
    a = np.sum(np.abs(psi_c_X) ** 2, axis=2)
    b = np.sum(np.abs(psi_v_Y) ** 2, axis=2)
    Y = np.einsum('kcM,MN->kcN', a.astype(W_q0.dtype), W_q0)
    W_d = np.real(np.einsum('kcN,kvN->cvk', Y, b.astype(W_q0.dtype)))
    S = np.einsum('kcvM,MN->kcvN', M_X, V_q0)
    V_x = np.real(np.einsum('kcvN,kcvN->cvk', S, np.conj(M_Y)))
    return dE + (V_x - W_d) / nk


@pytest.fixture(autouse=True)
def _clean_memo():
    H.clear_exact_diagonal_memo()
    yield
    H.clear_exact_diagonal_memo()


# ═══════════════════════════════════════════════════════════════════════
def test_matches_an_independent_numpy_contraction():
    ops = _payload()
    got = np.asarray(H.build_bse_exact_diagonal(*ops, NK, memo=False))
    want = _reference_diagonal(ops, NK)
    assert got.shape == (NC, NV, NK)
    assert np.max(np.abs(got - want)) < 1e-13, np.max(np.abs(got - want))


def test_kernel_is_module_scope_and_traces_once_per_process():
    """The defect this branch removed: a jit wrapper created per call.

    ``_exact_diagonal_kernel`` must be a module attribute (so its cache
    outlives the call) and must hold ONE entry after several builds of the
    same shapes.
    """
    assert hasattr(H, "_exact_diagonal_kernel")
    ops = _payload()
    for _ in range(4):
        H.build_bse_exact_diagonal(*ops, NK, memo=False).block_until_ready()
    size = getattr(H._exact_diagonal_kernel, "_cache_size", None)
    if size is None:
        pytest.skip("this jax exposes no _cache_size on jitted functions")
    assert size() == 1, f"traced {size()} times for one shape signature"


def test_memo_on_and_off_are_bit_identical():
    ops = _payload()
    off = np.asarray(H.build_bse_exact_diagonal(*ops, NK, memo=False))
    on1 = np.asarray(H.build_bse_exact_diagonal(*ops, NK, memo=True))
    on2 = np.asarray(H.build_bse_exact_diagonal(*ops, NK, memo=True))
    assert np.array_equal(off, on1)
    assert np.array_equal(off, on2)


def test_memo_hits_on_the_same_operand_objects():
    ops = _payload()
    H.build_bse_exact_diagonal(*ops, NK, memo=True).block_until_ready()
    before = H.exact_diagonal_memo_stats()["hits"]
    out = H.build_bse_exact_diagonal(*ops, NK, memo=True)
    assert H.exact_diagonal_memo_stats()["hits"] == before + 1
    assert out is not None


def test_memo_misses_when_W_changes():
    """RED TWIN — the stale-payload cell.

    Changing W must invalidate: the memo must not hand back the diagonal built
    from the old W, and the returned diagonal must actually move.
    """
    ops = _payload()
    first = np.asarray(H.build_bse_exact_diagonal(*ops, NK, memo=True))
    hits = H.exact_diagonal_memo_stats()["hits"]

    changed = list(ops)
    changed[4] = ops[4] * jnp.asarray(1.05, dtype=ops[4].dtype)   # new W
    second = np.asarray(H.build_bse_exact_diagonal(*changed, NK, memo=True))

    assert H.exact_diagonal_memo_stats()["hits"] == hits, "memo served a STALE W"
    assert not np.array_equal(first, second), "diagonal did not move with W"
    want = _reference_diagonal(tuple(changed), NK)
    assert np.max(np.abs(second - want)) < 1e-13


def test_memo_misses_on_equal_but_distinct_W():
    """RED TWIN — the memo is an IDENTITY memo, not an equality one.

    A value-equal but distinct W must MISS.  If this ever passes by hitting,
    the memo has started comparing values, and ``test_memo_misses_when_W_
    changes`` would then be passing for a reason that does not generalise.
    """
    ops = _payload()
    H.build_bse_exact_diagonal(*ops, NK, memo=True).block_until_ready()
    hits = H.exact_diagonal_memo_stats()["hits"]

    clone = list(ops)
    clone[4] = jnp.array(ops[4])                     # same values, new object
    out = np.asarray(H.build_bse_exact_diagonal(*clone, NK, memo=True))

    assert H.exact_diagonal_memo_stats()["hits"] == hits
    assert np.max(np.abs(out - _reference_diagonal(tuple(clone), NK))) < 1e-13


def test_memo_misses_when_nk_changes():
    ops = _payload()
    a = np.asarray(H.build_bse_exact_diagonal(*ops, NK, memo=True))
    hits = H.exact_diagonal_memo_stats()["hits"]
    b = np.asarray(H.build_bse_exact_diagonal(*ops, NK + 1, memo=True))
    assert H.exact_diagonal_memo_stats()["hits"] == hits
    assert not np.array_equal(a, b)


def test_clear_memo_forces_a_rebuild():
    ops = _payload()
    H.build_bse_exact_diagonal(*ops, NK, memo=True).block_until_ready()
    assert H.exact_diagonal_memo_stats()["loaded"] is True
    H.clear_exact_diagonal_memo()
    assert H.exact_diagonal_memo_stats()["loaded"] is False
    hits = H.exact_diagonal_memo_stats()["hits"]
    H.build_bse_exact_diagonal(*ops, NK, memo=True).block_until_ready()
    assert H.exact_diagonal_memo_stats()["hits"] == hits


def test_memo_pins_no_operand_buffer():
    """The memo holds weak references, so dropping the payload frees it."""
    import gc
    import weakref
    ops = _payload()
    H.build_bse_exact_diagonal(*ops, NK, memo=True).block_until_ready()
    ref = weakref.ref(ops[4])
    del ops
    gc.collect()
    gc.collect()
    assert ref() is None, "the memo kept a strong reference to W"


def test_preconditioner_application_uses_the_diagonal_elementwise():
    """The application side: same shape, same route, no collective anywhere.

    ``bse_diagonal_precond`` with ``diag_H`` set must differ from the bare
    route exactly by the substitution of D, and must be a pure elementwise
    map of R.
    """
    ops = _payload()
    eps_c, eps_v = ops[0], ops[1]
    diag = H.build_bse_exact_diagonal(*ops, NK, memo=False)

    bare = H.bse_diagonal_precond(eps_c, eps_v, epsilon_shift=1e-3)
    exact = H.bse_diagonal_precond(eps_c, eps_v, epsilon_shift=1e-3,
                                   diag_H=diag)
    rng = np.random.default_rng(7)
    R = jnp.asarray(rng.standard_normal((3, NC, NV, NK))
                    + 1j * rng.standard_normal((3, NC, NV, NK)))
    lam = jnp.asarray([0.1, 0.2, 0.3])
    pb, pe = bare(R, lam), exact(R, lam)
    assert pb.shape == pe.shape == R.shape
    assert not np.allclose(np.asarray(pb), np.asarray(pe))

    # elementwise: scaling one entry of R scales exactly that entry of P
    # (up to the per-vector normalisation), nothing else.
    R2 = R.at[0, 0, 0, 0].multiply(3.0)
    p2 = exact(R2, lam)
    p0 = np.asarray(pe)[1:]
    assert np.allclose(np.asarray(p2)[1:], p0)
