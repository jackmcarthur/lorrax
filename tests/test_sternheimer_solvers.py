"""Tests for solvers/projectors and solvers/sternheimer_precond.

CPU-only, deterministic (``np.random.seed``), <2 s wallclock.  These cover
correctness of the primitives in isolation.  End-to-end Sternheimer vs
band-sum validation lives in ``psp/run_sternheimer.py`` driver checks.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Sternheimer/W0 solver tooling (psp/orbital-mag program), not the GW
# pipeline — staged behind the `extra` marker (deselected by default;
# run with `-m extra`).
pytestmark = pytest.mark.extra


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _random_orthonormal_block(nv: int, nspinor: int, nG: int, seed: int) -> jax.Array:
    """Return ``(nv, nspinor, nG)`` complex orthonormal in the flat ``(nspinor*nG)`` metric."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((nv, nspinor * nG)) + 1j * rng.standard_normal((nv, nspinor * nG))
    Q, _ = np.linalg.qr(M.T)             # (nspinor*nG, nv)
    return jnp.asarray(Q.T.reshape(nv, nspinor, nG))


def _random_hermitian(n: int, seed: int) -> jax.Array:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return jnp.asarray(0.5 * (M + M.conj().T))


# ═══════════════════════════════════════════════════════════════════════
#  Projectors
# ═══════════════════════════════════════════════════════════════════════

class TestProjectors:
    nv, nspinor, nG = 4, 2, 12
    batch = 3

    def setup_method(self):
        self.U_val = _random_orthonormal_block(self.nv, self.nspinor, self.nG, seed=0)
        # Build U_extra orthogonal to U_val so P_val + P_precond is a projector
        # and P_rest = 1 - P_val - P_precond is well-defined.
        raw = _random_orthonormal_block(3, self.nspinor, self.nG, seed=1)
        coefs = jnp.einsum('msG,esG->em', jnp.conj(self.U_val), raw)
        Uext = raw - jnp.einsum('em,msG->esG', coefs, self.U_val)
        Uflat = np.asarray(Uext.reshape(3, -1))
        Q_extra, _ = np.linalg.qr(Uflat.T)
        self.U_extra = jnp.asarray(Q_extra.T.reshape(3, self.nspinor, self.nG))

        rng = np.random.default_rng(2)
        self.x = jnp.asarray(
            rng.standard_normal((self.batch, self.nspinor, self.nG))
            + 1j * rng.standard_normal((self.batch, self.nspinor, self.nG)))

    def test_P_val_is_idempotent(self):
        from solvers.projectors import make_P_val
        P = make_P_val(self.U_val)
        Px = P(self.x)
        PPx = P(Px)
        assert jnp.max(jnp.abs(Px - PPx)) < 1e-12

    def test_P_val_Hermitian(self):
        """<y, P x> = <P y, x>."""
        from solvers.projectors import make_P_val
        P = make_P_val(self.U_val)
        rng = np.random.default_rng(3)
        y = jnp.asarray(
            rng.standard_normal((self.batch, self.nspinor, self.nG))
            + 1j * rng.standard_normal((self.batch, self.nspinor, self.nG)))
        lhs = jnp.einsum('vsG,vsG->v', jnp.conj(y), P(self.x))
        rhs = jnp.einsum('vsG,vsG->v', jnp.conj(P(y)), self.x)
        assert jnp.max(jnp.abs(lhs - rhs)) < 1e-12

    def test_Q_kminq_equals_1_minus_P_val(self):
        from solvers.projectors import make_Q_kminq, make_P_val
        Q = make_Q_kminq(self.U_val)
        P = make_P_val(self.U_val)
        assert jnp.max(jnp.abs(Q(self.x) + P(self.x) - self.x)) < 1e-12

    def test_Q_annihilates_U_val(self):
        """Q_{k-q} · U_val_{k-q} = 0."""
        from solvers.projectors import make_Q_kminq
        Q = make_Q_kminq(self.U_val)
        out = Q(self.U_val)                  # (nv, nspinor, nG)
        assert jnp.max(jnp.abs(out)) < 1e-12

    def test_P_rest_novq_matches_Q(self):
        from solvers.projectors import make_P_rest, make_Q_kminq
        P_R = make_P_rest(self.U_val, U_extra=None)
        Q = make_Q_kminq(self.U_val)
        assert jnp.max(jnp.abs(P_R(self.x) - Q(self.x))) < 1e-12

    def test_P_rest_with_extra_sums_to_identity(self):
        """P_val + P_precond + P_rest = 1, when U_val ⟂ U_extra."""
        from solvers.projectors import make_P_val, make_P_precond, make_P_rest
        P_v = make_P_val(self.U_val)
        P_e = make_P_precond(self.U_extra)
        P_R = make_P_rest(self.U_val, self.U_extra)
        total = P_v(self.x) + P_e(self.x) + P_R(self.x)
        assert jnp.max(jnp.abs(total - self.x)) < 1e-11


# ═══════════════════════════════════════════════════════════════════════
#  TPA preconditioner
# ═══════════════════════════════════════════════════════════════════════

class TestTPA:
    def test_tpa_known_values(self):
        from solvers.sternheimer_precond import _tpa
        # TPA(0) = 1.  TPA(1) = 65/81 exactly.  Asymptote: TPA(x) ~ 1/(2x) for x→∞.
        assert abs(float(_tpa(jnp.asarray(0.0))) - 1.0) < 1e-15
        assert abs(float(_tpa(jnp.asarray(1.0))) - 65.0 / 81.0) < 1e-15
        # At x = 1e6:  TPA ≈ 5e-7, matching 1/(2x) to leading order.
        assert abs(float(_tpa(jnp.asarray(1e6))) * 2.0 * 1e6 - 1.0) < 1e-5

    def test_tpa_monotone(self):
        """TPA is strictly decreasing in x for x ∈ [0, 10]."""
        from solvers.sternheimer_precond import _tpa
        xs = jnp.linspace(0.0, 10.0, 200)
        ys = _tpa(xs)
        diffs = jnp.diff(ys)
        assert float(jnp.max(diffs)) < 0.0

    def test_per_band_kinetic(self):
        from solvers.sternheimer_precond import compute_per_band_kinetic
        # ψ_v[s, G] = δ_{G, v}, T_G = G (so K̄²_v = v)
        nv, nspinor, nG = 5, 2, 10
        U = jnp.zeros((nv, nspinor, nG), dtype=jnp.complex128)
        U = U.at[:, 0, :nv].set(jnp.eye(nv, nv, dtype=jnp.complex128))
        T = jnp.arange(nG, dtype=jnp.float64)
        K2 = compute_per_band_kinetic(U, T)
        np.testing.assert_allclose(np.asarray(K2), np.arange(nv, dtype=np.float64))

    def test_tpa_preconditioner_shape_and_damping(self):
        """Preconditioner weights: high-G modes damped more than low-G."""
        from solvers.sternheimer_precond import make_tpa_preconditioner
        nv, nspinor, nG = 3, 2, 20
        T = jnp.linspace(0.01, 20.0, nG)
        K2 = jnp.asarray([1.0, 5.0, 10.0])
        precond = make_tpa_preconditioner(T, K2)
        R = jnp.ones((nv, nspinor, nG), dtype=jnp.complex128)
        PR = precond(R)
        assert PR.shape == R.shape
        # For fixed v: weight at largest G < weight at smallest G
        for v in range(nv):
            w_lo = float(jnp.real(PR[v, 0, 0]))
            w_hi = float(jnp.real(PR[v, 0, -1]))
            assert w_hi < w_lo
