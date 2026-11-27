"""The matrix-free full-BSE route (``solver='matrixfree'``) on an emulated 2x2 CPU mesh.

``bse_nontda.solve_bse_nontda_sharded`` once documented the matrix-free route as refusing on a
sharded mesh (the zeta-sharded ``K^d_B`` defect).  No code tests the mesh: the only refusal is the
run-time gate on the symmetry of the kappa metric ``Re(x^H F(x'))`` (``metric_sym_err``, tolerance
1e-8), which is ``A = A^H`` and ``B = B^T`` on the operator the solve ran.  These tests settle by
execution whether the route works sharded, on the smallest payload the existing BSE tests use
(synthetic time-reversal-symmetric spinor payload, k grid 2x2x2, 4v x 4c, N_mu 24, N = 128;
``tests/test_bse_full_dipoles_cpu.py``), energies in Ry:

1. ``solver='matrixfree'`` on the 2x2 mesh runs (no refusal) and its lowest 8 roots equal
   ``solver='dense'``'s to 1e-8 Ry, in two settings: the production defaults (``m_max = 140`` exceeds
   N = 128, so the Krylov space is the whole window) and a restarted run (``m_max = 56``,
   ``n_keep = 24``, tolerance 1e-4 meV) that has to restart.  Each returned pair is Sigma-normalised
   (``X^H X - Y^H Y = 1`` to 1e-9), solves ``H z = omega z`` of the oracle-assembled ``A`` and ``B``,
   and lies on the mesh's transition axes.  The route's own metric symmetry number is read from its
   log line and held to its 1e-8 gate; the measured value is printed (``MEASURED ...``).
2. The gate is live: the same route with the coupling block made non-symmetric (a fixed real
   antisymmetric perturbation added to ``B``) refuses with the asymmetric-kappa-metric message, so the
   passing run in 1 is not a gate that cannot fire.  The message states the measured asymmetry and
   the fix; it does not name the retired mesh cause, and no docstring of ``bse_nontda`` says the route
   refuses on a sharded mesh.
3. The oracle (``tests/bse_full_oracle.py``) assembles ``A`` and ``B`` from the fused pair applier on
   the mesh; ``A = A^H`` and ``B = B^T`` to 1e-12, equal to the production ``_materialize_A_B``, and the
   oracle's reference roots equal the dense route's.

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_full_sharded_cpu.py -s``.
"""
import logging
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec  # noqa: E402

from bse import bse_stack_matvec  # noqa: E402
from bse.bse_nontda import (_full_matvec_and_args, _materialize_A_B,  # noqa: E402
                            solve_bse_nontda_sharded)
from bse.bse_ring_comm import make_bse_shardings  # noqa: E402
from tests import bse_full_oracle as orc  # noqa: E402
from tests.test_bse_full_dipoles_cpu import (KGRID, NC, NK, NV, N, _mesh,  # noqa: E402,F401
                                             _operands, _payload)

N_EIG = 8
ROOT_TOL = 1e-8           # Ry, the matrix-free roots against the dense route's
NORM_TOL = 1e-9
METRIC_GATE = 1e-8        # bse_nontda._MF_METRIC_TOL
MF_SETTINGS = {
    "defaults": {},                                                           # m_max 140 > N
    "restarted": dict(mf_m_max=56, mf_n_keep=24, mf_n_restarts=50, mf_tol_mev=1e-4),
}


def _solve(mesh, ops, solver, **kw):
    """Roots (Ry) and ``(n_eig, 2, nc, nv, nk)`` host vectors of one solve on a fresh payload.

    The payload is fresh because the solve donates ``W_q``.
    """
    omega, evecs, _ = solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG,
                                               solver=solver, **kw)
    return np.asarray(omega), evecs


def _metric_from_log(records):
    """The ``metric_sym`` value the matrix-free route logged (``[nontda] ... metric_sym 0.000e+00 ...``)."""
    vals = [float(m.group(1)) for r in records
            if (m := re.search(r"metric_sym\s+([0-9.]+(?:e[+-]?\d+)?)", r.getMessage()))]
    assert len(vals) == 1, f"expected one metric_sym log line, got {vals}"
    return vals[0]


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


@pytest.mark.parametrize("setting", sorted(MF_SETTINGS))
def test_matrixfree_runs_sharded_and_matches_dense(mesh, ops, dense_ref, caplog, setting):
    """The route does not refuse on 2x2: roots equal the dense route's (1e-8 Ry); metric gate clean."""
    caplog.set_level(logging.INFO)
    omega, evecs = _solve(mesh, ops, "matrixfree", **MF_SETTINGS[setting])
    dense = dense_ref["omega"]
    err = float(np.max(np.abs(omega - dense)))
    metric = _metric_from_log(caplog.records)
    cyc = re.findall(r"(\d+)/(\d+) restart cycles", " ".join(r.getMessage() for r in caplog.records))
    print(f"MEASURED matrixfree[{setting}] 2x2: max|omega - dense| = {err:.3e} Ry, "
          f"metric_sym_err = {metric:.3e} (gate {METRIC_GATE:g}), restart cycles {cyc[0][0]}/{cyc[0][1]}, "
          f"lowest root {omega[0]:.10f} Ry", flush=True)
    if setting == "restarted":
        assert int(cyc[0][0]) >= 1, "the restarted setting must restart at least once"
    else:
        assert int(cyc[0][0]) == 0 and MF_SETTINGS["defaults"] == {}
    assert omega.shape == (N_EIG,) and np.all(np.diff(omega) >= 0)
    assert err <= ROOT_TOL, (setting, err)
    assert metric <= METRIC_GATE
    # the vectors: layout, Sigma norm, and the eigenproblem of the oracle-assembled blocks
    assert isinstance(evecs.sharding, NamedSharding)
    assert tuple(evecs.sharding.spec) == (None, None, "x", "y", None)
    host = np.asarray(evecs)
    X = host[:, 0].reshape(N_EIG, N).T
    Y = host[:, 1].reshape(N_EIG, N).T
    assert np.max(np.abs(orc.sigma_norms(X, Y) - 1.0)) <= NORM_TOL
    resid = orc.eigpair_residuals(dense_ref["A"], dense_ref["B"], omega, X, Y)
    # the route stops at omega residual < tol_mev (0.01 meV default); vector residuals follow
    assert np.max(resid) <= (1e-5 if setting == "defaults" else 1e-6), np.max(resid)
    assert np.max(np.abs(omega - dense_ref["ref"]["omega"])) <= ROOT_TOL


def test_metric_gate_is_live_on_the_sharded_route(mesh, ops, monkeypatch):
    """A non-symmetric coupling block makes the same sharded route refuse (not a vacuous pass)."""
    rng = np.random.default_rng(1)
    S = rng.standard_normal((N, N))
    Rasym = jnp.asarray(0.05 * (S - S.T) / np.sqrt(N))                        # real antisymmetric
    orig = bse_stack_matvec.build_bse_stack_pair_matvec

    def build_broken(*a, **kw):
        pair = orig(*a, **kw)

        def broken(X, s, *args):
            out = pair(X, s, *args)
            flat = jnp.conj(X).reshape(X.shape[0], -1) @ Rasym.T              # B -> B + R, R^T = -R
            return out + (s * flat.reshape(X.shape)).astype(out.dtype)
        return broken

    monkeypatch.setattr(bse_stack_matvec, "build_bse_stack_pair_matvec", build_broken)
    with pytest.raises(ValueError, match="kappa metric is asymmetric") as exc:
        solve_bse_nontda_sharded(_payload(mesh, ops), mesh, n_eig=N_EIG, solver="matrixfree",
                                 **MF_SETTINGS["restarted"])
    m = re.search(r"asymmetric at ([0-9.e+-]+)", str(exc.value))
    print(f"MEASURED broken-B refusal: metric_sym_err = {m.group(1)} (gate {METRIC_GATE:g})", flush=True)
    assert float(m.group(1)) > 1e-3
    msg = str(exc.value)
    for stale in ("KNOWN CAUSE", "zeta axis is sharded", "one visible device", "SDY_SOLVER"):
        assert stale not in msg, f"stale cause in the refusal: {stale!r}"
    assert "A = A^H and B = B^T" in msg and "solver='dense'" in msg


def test_no_docstring_claims_a_mesh_refusal():
    """The route is not refused on a sharded mesh (the runs above), so no text of the module says it is."""
    import inspect
    from bse import bse_nontda
    src = inspect.getsource(bse_nontda)
    flat = " ".join(src.replace("#", " ").split())
    for stale in ("refuses on a sharded", "REFUSES on a sharded", "refuses on a sharded mesh today",
                  "has no bse_jax flag and refuses"):
        assert stale not in flat, f"stale mesh-refusal claim in bse_nontda: {stale!r}"


def test_matrixfree_follows_include_w_and_refuses_unknown_options(mesh, ops, dense_ref):
    """``include_W = False`` reaches the matrix-free applier (roots equal the dense route's, not the W ones);
    a misspelled option refuses instead of being swallowed."""
    kw = dict(MF_SETTINGS["restarted"], include_W=False)
    omega_mf, _ = _solve(mesh, ops, "matrixfree", **kw)
    omega_dn, _ = _solve(mesh, ops, "dense", include_W=False)
    err = float(np.max(np.abs(omega_mf - omega_dn)))
    shift = float(np.max(np.abs(omega_dn - dense_ref["omega"])))
    print(f"MEASURED include_W=False 2x2: max|mf - dense| = {err:.3e} Ry; W shifts the roots by {shift:.3e} Ry",
          flush=True)
    assert shift > 1e-4, "the W term must matter here, or this test checks nothing"
    assert err <= ROOT_TOL
    with pytest.raises(ValueError, match="unknown option") as exc:
        _solve(mesh, ops, "matrixfree", mf_m_maxx=56)
    assert "mf_m_maxx" in str(exc.value)


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
