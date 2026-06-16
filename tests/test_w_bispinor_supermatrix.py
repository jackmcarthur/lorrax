"""Bispinor screened-W supermatrix: assembly/extraction + the milestone-A
χ⁰⁰-only reduction (W⁰⁰ screened, W^{ij} bare, W^{0i}=0).

These test the channel-block algebra of ``gw.w_bispinor`` directly (no mesh /
solve_w needed); the production path composes the same assembly + extraction
with the already-tested ``gw.w_isdf.solve_w`` per-q LU.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gw.w_bispinor import assemble_supermatrix, extract_blocks

N_C, N_T, NQ = 5, 3, 2
TRANSVERSE = (1, 2, 3)


def _rand(key, shape):
    kr, ki = jax.random.split(key)
    return (jax.random.normal(kr, shape, dtype=jnp.float64)
            + 1j * jax.random.normal(ki, shape, dtype=jnp.float64))


def _v_blocks(key):
    """The 7 unique V tiles + 3 Hermitian fills; gauge (0,i)/(i,0) omitted (zero)."""
    keys = jax.random.split(key, 7)
    sz = {0: N_C, 1: N_T, 2: N_T, 3: N_T}
    v = {}
    v[(0, 0)] = _rand(keys[0], (NQ, N_C, N_C))
    for idx, (i, j) in enumerate([(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]):
        v[(i, j)] = _rand(keys[idx + 1], (NQ, sz[i], sz[j]))
    # Hermitian companions: V[j,i] = conj(swapaxes(V[i,j])).
    for (i, j) in [(2, 1), (3, 1), (3, 2)]:
        v[(i, j)] = jnp.conj(jnp.swapaxes(v[(j, i)], -1, -2))
    return v


def test_assemble_extract_roundtrip():
    key = jax.random.PRNGKey(0)
    v = _v_blocks(key)
    # Fill the gauge-zero tiles explicitly so every block round-trips.
    sz = {0: N_C, 1: N_T, 2: N_T, 3: N_T}
    for i in TRANSVERSE:
        v[(0, i)] = jnp.zeros((NQ, N_C, N_T), jnp.complex128)
        v[(i, 0)] = jnp.zeros((NQ, N_T, N_C), jnp.complex128)
    sup = assemble_supermatrix(v, N_C, N_T)
    assert sup.shape == (NQ, N_C + 3 * N_T, N_C + 3 * N_T)
    out = extract_blocks(sup, N_C, N_T)
    for (mu, nu), block in v.items():
        assert jnp.allclose(out[(mu, nu)], block), f"roundtrip block {(mu, nu)}"


def test_chi00_only_reduces_to_analytic_A():
    """χ populated only in (0,0) ⇒ W⁰⁰ = (I−V⁰⁰·pref·χ⁰⁰)^{-1}V⁰⁰,
    W^{ij} = V^{ij} (bare), W^{0i}=W^{i0}=0."""
    key = jax.random.PRNGKey(1)
    kv, kc = jax.random.split(key)
    v = _v_blocks(kv)
    chi00 = _rand(kc, (NQ, N_C, N_C))
    pref = 0.3719  # any scalar; solve_w applies 2/(√Nk·nspin·nspinor) the same way

    v_super = assemble_supermatrix(v, N_C, N_T)
    chi_super = assemble_supermatrix({(0, 0): chi00}, N_C, N_T)

    # Per-q Dyson solve mirroring w_isdf.solve_w's solve_one: W=(I−V(pref·χ))^{-1}V.
    N = N_C + 3 * N_T
    eye = jnp.eye(N, dtype=jnp.complex128)
    A = eye[None] - jnp.einsum('qab,qbc->qac', v_super, pref * chi_super)
    w_super = jnp.linalg.solve(A, v_super)
    W = extract_blocks(w_super, N_C, N_T)

    # Charge block matches the standalone scalar screened W on V⁰⁰.
    A00 = jnp.eye(N_C, dtype=jnp.complex128)[None] - jnp.einsum(
        'qab,qbc->qac', v[(0, 0)], pref * chi00)
    W00_ref = jnp.linalg.solve(A00, v[(0, 0)])
    assert jnp.allclose(W[(0, 0)], W00_ref, atol=1e-10), "W00 != scalar screened W"

    # Transverse blocks unchanged (bare); charge-current cross blocks vanish.
    for i in TRANSVERSE:
        for j in TRANSVERSE:
            assert jnp.allclose(W[(i, j)], v[(i, j)], atol=1e-10), f"W^{i}{j} != bare V"
        assert jnp.allclose(W[(0, i)], 0.0, atol=1e-12), f"W^0{i} != 0"
        assert jnp.allclose(W[(i, 0)], 0.0, atol=1e-12), f"W^{i}0 != 0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
