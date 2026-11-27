"""The full-BSE routes on an emulated 2x2 CPU mesh, against the oracle assembled from the pair applier.

Smallest payload the existing BSE tests use (synthetic time-reversal-symmetric spinor payload, k grid
2x2x2, 4v x 4c, N_mu 24, N = 128; ``tests/test_bse_full_dipoles_cpu.py``), energies in Ry:

1. The Davidson route (``solver='davidson'``) on the 2x2 mesh with ``include_W = False`` returns the
   dense route's roots for the same operator (1e-8 Ry), and a misspelled option refuses instead of being
   swallowed.
2. The oracle (``tests/bse_full_oracle.py``) assembles ``A`` and ``B`` from the fused pair applier on
   the mesh; ``A = A^H`` and ``B = B^T`` to 1e-12, equal to the production ``_materialize_A_B``, and the
   oracle's reference roots equal the dense route's.
3. No docstring of ``bse_nontda`` claims a refusal on a sharded mesh.

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_full_sharded_cpu.py -s``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec  # noqa: E402

from bse.bse_nontda import (_full_matvec_and_args, _materialize_A_B,  # noqa: E402
                            solve_bse_nontda_sharded)
from bse.bse_ring_comm import make_bse_shardings  # noqa: E402
from tests import bse_full_oracle as orc  # noqa: E402
from tests.test_bse_full_dipoles_cpu import (KGRID, NC, NK, NV, N, _mesh,  # noqa: E402,F401
                                             _operands, _payload)

N_EIG = 8
ROOT_TOL = 1e-8           # Ry, the Davidson's roots against the dense route's

def _solve(mesh, ops, solver, **kw):
    """Roots (Ry) and ``(n_eig, 2, nc, nv, nk)`` host vectors of one solve on a fresh payload.

    The payload is fresh because the solve donates ``W_q``.
    """
    omega, evecs, _ = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG,
                                               solver=solver, **kw)
    return np.asarray(omega), evecs


@pytest.fixture(scope="module")
def mesh():
    m = _mesh()
    assert tuple(m.devices.shape) == (2, 2)
    return m


@pytest.fixture(scope="module")
def ops():
    return _operands()


@pytest.fixture(scope="module")
def dense_ref(mesh, ops):
    """Dense route's roots, and the oracle's roots from independently assembled A and B."""
    omega, evecs = _solve(mesh, ops, "dense")
    sh = make_bse_shardings(mesh)
    pair, args = _full_matvec_and_args(_payload(mesh, ops), mesh, sh, include_W=True)
    A, B = _materialize_A_B(pair, args, sh, NC, NV, NK)
    return {"omega": omega, "A": A, "B": B, "ref": orc.reference_roots(A, B, n_eig=N_EIG)}


def test_no_docstring_claims_a_mesh_refusal():
    """The routes are not refused on a sharded mesh, so no text of the module says they are."""
    import inspect
    from bse import bse_nontda
    src = inspect.getsource(bse_nontda)
    flat = " ".join(src.replace("#", " ").split())
    for stale in ("refuses on a sharded", "REFUSES on a sharded", "refuses on a sharded mesh today",
                  "has no bse_jax flag and refuses"):
        assert stale not in flat, f"stale mesh-refusal claim in bse_nontda: {stale!r}"


def test_davidson_follows_include_w_and_refuses_unknown_options(mesh, ops, dense_ref):
    """``include_W = False`` reaches the pair applier on the 2x2 mesh (the Davidson's roots equal the
    dense route's, not the W ones); a misspelled option refuses instead of being swallowed."""
    omega_dv, _ = _solve(mesh, ops, "davidson", include_W=False)
    omega_dn, _ = _solve(mesh, ops, "dense", include_W=False)
    err = float(np.max(np.abs(omega_dv - omega_dn)))
    shift = float(np.max(np.abs(omega_dn - dense_ref["omega"])))
    print(f"MEASURED include_W=False 2x2: max|davidson - dense| = {err:.3e} Ry; W shifts the roots by "
          f"{shift:.3e} Ry", flush=True)
    assert shift > 1e-4, "the W term must matter here, or this test checks nothing"
    assert err <= ROOT_TOL
    with pytest.raises(ValueError, match="unknown option") as exc:
        _solve(mesh, ops, "davidson", davidson_m_maxx=56)
    assert "davidson_m_maxx" in str(exc.value)


def test_oracle_assembles_the_sharded_pair_applier(mesh, ops, dense_ref):
    """Identity columns through the 2x2 fused pair applier give ``A = A^H``, ``B = B^T``, the dense roots."""
    sh = make_bse_shardings(mesh)
    pair, args = _full_matvec_and_args(_payload(mesh, ops), mesh, sh, include_W=True)
    from lxkit import device_put_process_local

    def apply(X, s):
        """``pair(X, s)`` on the mesh: sign vector, X on the transition axes, result gathered."""
        b = X.shape[0]
        Xs = device_put_process_local(np.asarray(X, dtype=np.complex128), sh.X)
        out = pair(Xs, jnp.full((b,), float(s)), *args)
        to_host = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, PartitionSpec()))
        return np.asarray(to_host(out).addressable_data(0))

    A, B = orc.assemble_dense_ab(apply, (NC, NV, NK), chunk=32)
    orc.assert_bse_structure(A, B, tol=1e-12)
    assert np.max(np.abs(A - dense_ref["A"])) <= 1e-12 * np.max(np.abs(dense_ref["A"]))
    assert np.max(np.abs(B - dense_ref["B"])) <= 1e-12 * np.max(np.abs(dense_ref["B"]))
    # the kappa-metric symmetry Re(u^H F(x)) = Re(x^H F(u)) on the sharded applier, measured here
    rng = np.random.default_rng(5)
    u, x = (rng.standard_normal((1, NC, NV, NK)) + 1j * rng.standard_normal((1, NC, NV, NK))
            for _ in range(2))
    Fu, Fx = apply(u, 1.0).reshape(-1), apply(x, 1.0).reshape(-1)
    s_ab, s_ba = np.real(u.reshape(-1).conj() @ Fx), np.real(x.reshape(-1).conj() @ Fu)
    sym = abs(s_ab - s_ba) / max(abs(s_ab), abs(s_ba))
    print(f"MEASURED kappa-metric symmetry of the sharded pair applier (random probes): {sym:.3e}", flush=True)
    assert sym <= 1e-12
    ref = orc.reference_roots(A, B, n_eig=N_EIG)
    assert ref["definite"]
    assert np.max(np.abs(ref["omega"] - dense_ref["omega"])) <= ROOT_TOL
    assert np.max(orc.eigpair_residuals(A, B, ref["omega"], ref["X"], ref["Y"])) <= 1e-10
