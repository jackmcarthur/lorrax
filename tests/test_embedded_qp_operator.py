"""Independent finite-dimensional gates for the embedded QSGW operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gw.embedded_qp_operator import (
    apply_embedded_qp_hamiltonian,
    embedded_qp_contract_residuals,
    validate_embedded_qp_contract,
)


jax.config.update("jax_enable_x64", True)


def _fixture(seed=20260827):
    rng = np.random.default_rng(seed)
    nband, nspinor, ng, nvec = 4, 2, 7, 3
    dim = nspinor * ng
    raw_basis = (
        rng.standard_normal((dim, nband))
        + 1.0j * rng.standard_normal((dim, nband))
    )
    q_basis, _ = np.linalg.qr(raw_basis)
    basis = q_basis.T.reshape(nband, nspinor, ng)
    raw_h = (
        rng.standard_normal((nband, nband))
        + 1.0j * rng.standard_normal((nband, nband))
    )
    h_w = 0.5 * (raw_h + raw_h.conj().T)
    raw_tail = (
        rng.standard_normal((dim, dim))
        + 1.0j * rng.standard_normal((dim, dim))
    )
    h_tail = 0.5 * (raw_tail + raw_tail.conj().T)
    x = (
        rng.standard_normal((nvec, nspinor, ng))
        + 1.0j * rng.standard_normal((nvec, nspinor, ng))
    )
    return basis, h_w, h_tail, x


def _tail_apply(h_tail, x):
    flat = x.reshape(x.shape[0], -1)
    return jnp.einsum("ij,bj->bi", h_tail, flat).reshape(x.shape)


def _dense_embedded(basis, h_w, h_tail):
    rows = basis.reshape(basis.shape[0], -1)
    projector = rows.T @ rows.conj()
    complement = np.eye(rows.shape[1]) - projector
    return rows.T @ h_w @ rows.conj() + complement @ h_tail @ complement


def test_embedded_operator_matches_independent_dense_matrix():
    basis, h_w, h_tail, x = _fixture()
    actual = apply_embedded_qp_hamiltonian(
        jnp.asarray(x),
        jnp.asarray(basis),
        jnp.asarray(h_w),
        lambda q: _tail_apply(jnp.asarray(h_tail), q),
    )
    dense = _dense_embedded(basis, h_w, h_tail)
    expected = np.stack([dense @ row.reshape(-1) for row in x]).reshape(x.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_stored_block_is_exact_and_has_no_tail_leakage():
    basis, h_w, h_tail, _ = _fixture()
    coefficients = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.2j, -0.4, 0.1, 0.7j],
    ])
    x = np.einsum("bm,msG->bsG", coefficients, basis)
    expected_coefficients = np.einsum("mn,bn->bm", h_w, coefficients)
    expected = np.einsum("bm,msG->bsG", expected_coefficients, basis)
    actual = apply_embedded_qp_hamiltonian(
        jnp.asarray(x),
        jnp.asarray(basis),
        jnp.asarray(h_w),
        lambda q: _tail_apply(jnp.asarray(h_tail), q),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_complement_action_is_q_h_tail_q_and_cross_block_is_zero():
    basis, h_w, h_tail, x = _fixture()
    rows = basis.reshape(basis.shape[0], -1)
    projector = rows.T @ rows.conj()
    complement = np.eye(projector.shape[0]) - projector
    q_x = (complement @ x.reshape(x.shape[0], -1).T).T.reshape(x.shape)
    actual = apply_embedded_qp_hamiltonian(
        jnp.asarray(q_x),
        jnp.asarray(basis),
        jnp.asarray(h_w),
        lambda q: _tail_apply(jnp.asarray(h_tail), q),
    )
    expected = np.stack([
        complement @ h_tail @ complement @ row.reshape(-1) for row in q_x
    ]).reshape(q_x.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    retained_overlap = np.einsum("msG,bsG->bm", basis.conj(), actual)
    np.testing.assert_allclose(retained_overlap, 0.0, atol=2e-13)


def test_operator_is_hermitian():
    basis, h_w, h_tail, _ = _fixture()
    dense = _dense_embedded(basis, h_w, h_tail)
    np.testing.assert_allclose(dense, dense.conj().T, atol=2e-13)


def test_stored_basis_gauge_covariance():
    basis, h_w, h_tail, x = _fixture()
    rng = np.random.default_rng(93)
    raw_rotation = (
        rng.standard_normal(h_w.shape) + 1.0j * rng.standard_normal(h_w.shape)
    )
    rotation, _ = np.linalg.qr(raw_rotation)
    rotated_basis = np.einsum("ma,msG->asG", rotation, basis)
    rotated_h = rotation.conj().T @ h_w @ rotation
    apply = lambda b, h: apply_embedded_qp_hamiltonian(
        jnp.asarray(x),
        jnp.asarray(b),
        jnp.asarray(h),
        lambda q: _tail_apply(jnp.asarray(h_tail), q),
    )
    np.testing.assert_allclose(
        apply(rotated_basis, rotated_h), apply(basis, h_w),
        rtol=3e-13, atol=3e-13,
    )


def test_contract_residuals_and_refusals():
    basis, h_w, _, x = _fixture()
    orth, herm = embedded_qp_contract_residuals(
        jnp.asarray(basis), jnp.asarray(h_w))
    assert float(orth) < 1.0e-12
    assert float(herm) < 1.0e-12
    validate_embedded_qp_contract(basis, h_w, tolerance=1.0e-12)

    broken_basis = basis.copy()
    broken_basis[0] *= 1.01
    with pytest.raises(ValueError, match="not orthonormal"):
        validate_embedded_qp_contract(broken_basis, h_w)
    broken_h = h_w.copy()
    broken_h[0, 1] += 0.3j
    with pytest.raises(ValueError, match="not Hermitian"):
        validate_embedded_qp_contract(basis, broken_h)
    with pytest.raises(ValueError, match="complete spinor/G carrier"):
        apply_embedded_qp_hamiltonian(
            jnp.asarray(x[..., :-1]), jnp.asarray(basis), jnp.asarray(h_w),
            lambda q: q)
    with pytest.raises(ValueError, match="preserve the ket carrier"):
        apply_embedded_qp_hamiltonian(
            jnp.asarray(x), jnp.asarray(basis), jnp.asarray(h_w),
            lambda q: q[..., :-1])
