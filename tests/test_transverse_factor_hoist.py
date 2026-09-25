"""Bit-identity gate for the HOISTED transverse ζ factor stage (2026-08).

The transverse (bispinor mu_L=1,2,3) CCT is Hermitian indefinite; its
factor is a per-q pivoted LU with ridge.  Historically the LU was fused
into the back-solve and re-run on EVERY r-chunk; ``factor_c_q`` now runs
it ONCE per channel (``isdf.core._factor_c_q_transverse_lu``) and the
back-solve only applies ``lax.linalg.lu_solve`` (route G:
``isdf.zeta_mubatch._logical_solve('lu')``, vmapped over q per G tile).

The hoist's whole contract is the same as the charge fold's
(``test_zeta_mesh_invariance.test_qparallel_execution_is_bit_identical_to_replicated``):
it is a SCHEDULE — factor once instead of per r-chunk, scattered over
devices at P>1 — never a numerical route.  ``jnp.linalg.solve(A, b)``
lowers to ``lax.linalg.lu(A)`` + ``lax.linalg.lu_solve(lu, perm, b, 0)``
(jax ``lax_linalg._solve``), and the hoisted stage runs exactly those two
ops on exactly the ridged matrix the fused path built, so the gate below
demands EXACT bit equality of ζ against the preserved fused path (raw
CCT through the ridged ``jnp.linalg.solve``,
``isdf.core._zeta_logical_solvers(n)[0]``) across:

* CPU meshes 1x1 / 2x2 / 1x4 (``--xla_force_host_platform_device_count``),
* both factor schedules (``LORRAX_ZETA_QPARALLEL`` 0 / 1),
* multiple r-chunks against ONE factor (the reuse that motivates the
  hoist),
* a non-dividing nq (q-pad + cond-skip) and a padded mu extent
  (identity-pad re-embed + logical-extent slicing),
* a spectrum with TRS-paired near-null modes (the ridge's reason).

The moment this gate needs a tolerance, the hoist has become a numerical
route and must be re-argued.

The DISTRIBUTED plan's twin claim (pXgetrf once + pXgetrs per r-chunk ==
the fused pXgetrf+pXgetrs handler, bit-exact) needs one JAX process per
device + the host FFI .so, so it lives in the srun harness
``tools`` / job records (CLAIMS ledger), not in this single-process
suite.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_NDEV = 4


def _per_q(fn, *operands):
    """Apply a per-q logical-extent solve over the q axis (route G's
    ``jax.vmap(one)(F, Z)``), host result."""
    import numpy as np
    import jax
    return np.asarray(jax.device_get(jax.jit(jax.vmap(fn))(*operands)))


def _fused(n_log):
    """The preserved FUSED path: ridged ``jnp.linalg.solve`` on the raw CCT."""
    from isdf.core import _zeta_logical_solvers
    return _zeta_logical_solvers(n_log)[0]


def _hoisted(n_log):
    """Route G's application of the hoisted ``(LU, perm)`` factor."""
    from isdf.zeta_mubatch import _logical_solve
    one = _logical_solve('lu', n_log)
    return lambda LU, piv, Z: one((LU, piv), Z)


def _worker_hoist() -> int:
    """Child: hoisted factor+solve vs the fused path, exact equality."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q
    import isdf.core as core

    devs = jax.devices()
    if len(devs) < _NDEV:
        print(json.dumps({"skip": f"only {len(devs)} devices"}))
        return 0

    rng = np.random.default_rng(20260802)
    nq, n_log, n_pad, n_z = 6, 60, 64, 24

    # Hermitian INDEFINITE per-q logical blocks with two TRS-paired
    # near-null modes each (the transverse CCT's failure shape: gamma^i
    # (x) gamma^i has both signs of eigenvalue and band cancellations
    # push pairs of modes to ~0).
    C = np.zeros((nq, n_pad, n_pad), dtype=np.complex128)
    for q in range(nq):
        A = (rng.standard_normal((n_log, n_log))
             + 1j * rng.standard_normal((n_log, n_log)))
        H = 0.5 * (A + A.conj().T)
        lam, V = np.linalg.eigh(H)
        lam = lam - np.median(lam)          # both signs
        lam[0] = 1e-14                      # near-null pair
        lam[1] = -1e-14
        C[q, :n_log, :n_log] = (V * lam[None, :]) @ V.conj().T

    Z = np.zeros((nq, n_pad, n_z), dtype=np.complex128)
    Z[:, :n_log, :] = (rng.standard_normal((nq, n_log, n_z))
                       + 1j * rng.standard_normal((nq, n_log, n_z)))
    Z2 = np.zeros_like(Z)
    Z2[:, :n_log, :] = (rng.standard_normal((nq, n_log, n_z))
                        + 1j * rng.standard_normal((nq, n_log, n_z)))

    exact = {}
    max_abs = 0.0
    for (px, py) in [(1, 1), (2, 2), (1, 4)]:
        mesh = Mesh(np.asarray(devs[: px * py]).reshape(px, py), ('x', 'y'))
        in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
        C_dev = jax.device_put(jnp.asarray(C), in_sh)
        # The preserved FUSED path: identity-padded raw CCT (exactly what
        # the pre-hoist factor_c_q returned) + lu_piv=None.
        L_fused = core._identity_pad_block_diagonal(
            C_dev, n_rmu_logical=n_log, mesh_xy=mesh)
        for force in (('0',) if px * py == 1 else ('0', '1')):
            os.environ['LORRAX_ZETA_QPARALLEL'] = force
            core._transverse_lu_cache.clear()
            LU, piv = factor_c_q(
                C_dev, mesh, vertex_mu_L=1, n_rmu_logical=n_log,
                solver_kind='lu')
            assert piv is not None and piv.shape == (nq, n_log), piv.shape
            Z_dev = jax.device_put(jnp.asarray(Z), in_sh)
            Z2_dev = jax.device_put(jnp.asarray(Z2), in_sh)
            ref1 = _per_q(_fused(n_log), L_fused, Z_dev)
            ref2 = _per_q(_fused(n_log), L_fused, Z2_dev)
            # ONE hoisted factor, TWO r-chunks (the reuse).
            got1 = _per_q(_hoisted(n_log), LU, piv, Z_dev)
            got2 = _per_q(_hoisted(n_log), LU, piv, Z2_dev)
            tag = f"{px}x{py}_qp{force}"
            exact[tag] = bool(np.array_equal(ref1, got1)
                              and np.array_equal(ref2, got2))
            max_abs = max(max_abs,
                          float(np.max(np.abs(ref1 - got1))),
                          float(np.max(np.abs(ref2 - got2))))
            # Pad rows of zeta must be exactly zero on both paths.
            if n_pad > n_log:
                exact[tag] = exact[tag] and bool(
                    np.all(got1[:, n_log:, :] == 0.0))
    os.environ.pop('LORRAX_ZETA_QPARALLEL', None)
    print(json.dumps({"exact": exact, "max_abs": max_abs}))
    return 0


def _worker_ridge_effect() -> int:
    """Child: the hoisted factor really carries the ridge — solving the
    near-null fixture WITHOUT it would amplify to ~1/eps_machine; with it
    the solution norm stays at the 1/ridge scale.  (Sanity that the hoist
    did not silently drop the conditioning term; exactness vs the fused
    path is the other worker's job.)"""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from isdf import factor_c_q

    devs = jax.devices()
    rng = np.random.default_rng(7)
    nq, n = 2, 32
    C = np.zeros((nq, n, n), dtype=np.complex128)
    for q in range(nq):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = 0.5 * (A + A.conj().T)
        lam, V = np.linalg.eigh(H)
        lam = lam - np.median(lam)
        lam[0] = 0.0                        # EXACT null mode
        C[q] = (V * lam[None, :]) @ V.conj().T
    Z = (rng.standard_normal((nq, n, 8))
         + 1j * rng.standard_normal((nq, n, 8)))

    mesh = Mesh(np.asarray(devs[:1]).reshape(1, 1), ('x', 'y'))
    in_sh = NamedSharding(mesh, P(None, 'x', 'y'))
    C_dev = jax.device_put(jnp.asarray(C), in_sh)
    LU, piv = factor_c_q(C_dev, mesh, vertex_mu_L=1, n_rmu_logical=n,
                         solver_kind='lu')
    zeta = _per_q(_hoisted(n), LU, piv, jax.device_put(jnp.asarray(Z), in_sh))
    print(json.dumps({
        "finite": bool(np.all(np.isfinite(zeta))),
        "log10_norm": float(np.log10(np.linalg.norm(zeta))),
    }))
    return 0


def _run_worker(tag: str, timeout: int = 900):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    # This fixture deliberately contains near-null indefinite modes so it
    # can test scheduling identity at the difficult spectrum.  Keep the
    # production rank-policy diagnostic, but do not let that independent
    # policy gate prevent the fused-versus-hoisted comparison from running.
    env["LORRAX_RANK_POLICY"] = "warn"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={_NDEV}"
                        ).strip()
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), tag],
        env=env, capture_output=True, text=True, timeout=timeout)
    assert res.returncode == 0, (
        f"worker {tag} failed rc={res.returncode}\nSTDOUT:\n{res.stdout}\n"
        f"STDERR:\n{res.stderr}")
    line = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
    assert line, f"no JSON from worker.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(line[-1])


def test_hoisted_transverse_lu_is_bit_identical_to_fused():
    """factor_c_q's hoisted (LU, perm) + route G's lu_solve reproduce
    the fused per-r-chunk jnp.linalg.solve path EXACTLY — every mesh,
    both factor schedules, two r-chunks per factor."""
    out = _run_worker("worker_hoist")
    if "skip" in out:
        pytest.skip(f"hoist gate: {out['skip']}")
    bad = sorted(k for k, v in out["exact"].items() if not v)
    assert not bad, (
        f"hoisted transverse LU drifts from the fused path on {bad} "
        f"(max abs delta {out['max_abs']:.3e}); the hoist's bit-identity "
        f"contract is broken")


def test_hoisted_factor_carries_the_ridge():
    """An exact-null transverse mode stays finite through the hoisted
    factor (the 1e-12·|tr|/n ridge is baked in at factor time)."""
    out = _run_worker("worker_ridge")
    assert out["finite"], "hoisted transverse solve produced non-finite zeta"
    # 1/ridge-scale, not 1/eps-scale: allow generous headroom either way.
    assert out["log10_norm"] < 16.0, (
        f"solution norm 1e{out['log10_norm']:.1f} looks unridged")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    if tag == "worker_hoist":
        sys.exit(_worker_hoist())
    if tag == "worker_ridge":
        sys.exit(_worker_ridge_effect())
    print(f"unknown worker tag {tag!r}", file=sys.stderr)
    sys.exit(2)
