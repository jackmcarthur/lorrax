"""``exciton_bands``' path solver must stay persistable in the compile cache.

The defect (asides-audit A3, and open item 3 of ``FIX_warmcache.md``): the
per-Q block Lanczos runs inside a ``lax.scan`` inside a ``jax.jit``, and the
solver's α-Hermiticity report was a ``jax.debug.callback``.  JAX's
``_cache_write`` refuses, BY DESIGN, to write a persistent-cache entry for any
module carrying a host callback — so the single most expensive compile in the
driver (one scan over the whole Q path) was rebuilt on every warm run.

The fix is the one the warm-cache lane landed for the BSE solver
(``solvers.lanczos.alpha_herm_sink`` + ``report_alpha_herm``), applied at the
one place that differs: the sink is opened INSIDE the scan body, because that
is the trace the solver is traced in, and the three scalars per Q ride out as
scan ``ys``.

Every cell carries its red twin — the same program traced WITHOUT the sink —
so a gate cannot silently stop testing anything.  These need the FFI layer
(the driver module imports the runtime stack), so they are not part of the
laptop suite.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


# ---------------------------------------------------------------------------
# the same jaxpr walker the BSE-solver gates use (FIX_warmcache §4)
# ---------------------------------------------------------------------------
def _sub_jaxprs(x):
    out = []
    if hasattr(x, "jaxpr"):
        out.append(x.jaxpr if hasattr(x.jaxpr, "eqns") else x)
    if isinstance(x, (list, tuple)):
        for item in x:
            out.extend(_sub_jaxprs(item))
    return out


def _callback_prims(jaxpr):
    found = []
    for eqn in jaxpr.eqns:
        if "callback" in eqn.primitive.name:
            found.append(eqn.primitive.name)
        for param in eqn.params.values():
            for sub in _sub_jaxprs(param):
                found.extend(_callback_prims(sub))
    return found


# ---------------------------------------------------------------------------
# a TINY but REAL path solver: the production build_path_solver, 1x1x1 k,
# 2 conduction x 2 valence, 4 centroids, 3 Q.  Nothing is mocked -- the point
# is to gate the program the driver actually compiles.
# ---------------------------------------------------------------------------
NKX = NKY = NKZ = 1
NC = NV = 2
NMU = 4
NS = 1
NQ = 3


def _mesh(px=1, py=1):
    from bse.bse_w_exact import _create_mesh_xy
    return _create_mesh_xy(px, py)


def _operands(seed=0):
    """Operands that make the BSE Hamiltonian genuinely HERMITIAN.

    The two kernel tiles have to be Hermitized or ``H`` is not Hermitian and
    the α-Hermiticity gate refuses the operator — correctly.  (It did, on the
    first run of this file: random tiles gave dev/scale = 1.665, and the gate
    named the operator rather than the algorithm.  That refusal is itself
    evidence the invariant survived being hoisted out of the jit, but it makes
    for a useless fixture.)

    At a 1×1×1 k-grid ``R`` has a single point, so the real-space condition
    ``W_{μν}(R) = conj(W_{νμ}(−R))`` collapses to "``W_R[:, :, 0,0,0]`` is
    Hermitian"; ``V_Q`` is Hermitized per Q, which is what the production
    driver does to the interpolated tiles anyway.
    """
    rng = np.random.default_rng(seed)

    def c(*shape):
        return jnp.asarray(rng.standard_normal(shape)
                           + 1j * rng.standard_normal(shape))

    def herm_mn(a):                       # Hermitize the trailing (μ, ν) pair
        return 0.5 * (a + jnp.conj(jnp.swapaxes(a, -2, -1)))

    nk = NKX * NKY * NKZ
    # The X and Y legs are the SAME ψ under two shardings in production, so
    # they must carry the same VALUES here or H is not Hermitian either.
    psi_c = c(NQ, nk, NC, NS, NMU)
    psi_v = c(nk, NV, NS, NMU)
    V_Q = herm_mn(c(NQ, NMU, NMU))
    W_R = herm_mn(c(NMU, NMU))[:, :, None, None, None]

    return (
        psi_c, psi_c,
        jnp.asarray(rng.standard_normal((NQ, nk, NC)) + 2.0),
        V_Q,
        psi_v, psi_v,
        jnp.asarray(rng.standard_normal((nk, NV)) - 2.0),
        W_R,
    )


def _unsunk_twin(mesh):
    """THE RED TWIN: the pre-fix shape — no sink, so the callback is traced.

    Deliberately a hand-rolled copy of ``build_path_solver``'s body rather
    than a flag on the production function: a red twin that shares a switch
    with the green side goes green the moment the switch is mis-wired.
    """
    from jax import lax
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude
    from bse.bse_stack_matvec import build_bse_stack_matvec
    from solvers.lanczos import block_lanczos_eig_jit

    sh = make_bse_shardings(mesh)
    nk = NKX * NKY * NKZ
    n_flat = NC * NV * nk
    matvec = build_bse_stack_matvec(mesh, NKX, NKY, NKZ, kernel="bse")

    @jax.jit
    def solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R):
        def body(carry, xs):
            psi_c_X, psi_c_Y, eps_c, V = xs
            M_X = compute_pair_amplitude(psi_c_X, psi_v_X)
            M_Y = compute_pair_amplitude(psi_c_Y, psi_v_Y)

            def matvec_block(Vb):
                X = Vb.reshape(2, NC, NV, nk)
                X = lax.with_sharding_constraint(X, sh.X)
                HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                            eps_c, eps_v, W_R, V, M_X, M_Y)
                return HX.reshape(2, -1)

            evs, _ = block_lanczos_eig_jit(
                matvec_block, n_flat, n_eig=2, block_size=2,
                max_iter=4, n_reorth=4)
            return carry, evs[:2].real

        _, evs_all = lax.scan(body, None,
                              (psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q))
        return evs_all

    return solve_path


def _green(mesh):
    from bse.exciton_bands import build_path_solver
    return build_path_solver(mesh, NKX, NKY, NKZ, NC, NV,
                             n_eig=2, block_size=2, max_iter=4)


# ---------------------------------------------------------------------------
# 1.  no host callback under the sink; the twin has one
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_path_solver_carries_no_host_callback_p1():
    harness.skip_unless_gpu(pytest)
    mesh = _mesh()
    args = _operands()
    with mesh:
        green = _callback_prims(jax.make_jaxpr(_green(mesh))(*args).jaxpr)
        red = _callback_prims(
            jax.make_jaxpr(_unsunk_twin(mesh))(*args).jaxpr)

    assert red, ("RED TWIN DID NOT GO RED: the un-sunk scan traces no "
                 "callback primitive, so this gate is testing nothing.  Did "
                 "the alpha-Hermiticity gate get deleted rather than moved?")
    assert green == [], (
        f"build_path_solver still traces {green} under alpha_herm_sink.  JAX "
        f"will not persist any module carrying a host callback, so the whole "
        f"Q-path scan recompiles on every warm run.")


# ---------------------------------------------------------------------------
# 2.  THE cell the fix exists for: JAX's own persistence decision
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_path_solver_is_written_to_the_compile_cache_p1(tmp_path):
    """Green writes a real cache entry; the red twin writes none."""
    harness.skip_unless_gpu(pytest)
    from jax._src import compilation_cache as cc

    mesh = _mesh()
    args = _operands()
    prev_dir = jax.config.jax_compilation_cache_dir
    prev_min = jax.config.jax_persistent_cache_min_compile_time_secs

    def _entries(d):
        return sorted(p.name for p in d.iterdir() if p.is_file())

    results = {}
    try:
        with mesh:
            for name, build in (("green", _green), ("red", _unsunk_twin)):
                d = tmp_path / name
                d.mkdir()
                cc.reset_cache()
                jax.config.update("jax_compilation_cache_dir", str(d))
                jax.config.update(
                    "jax_persistent_cache_min_compile_time_secs", 0.0)
                build(mesh).lower(*args).compile()
                cc.reset_cache()
                results[name] = _entries(d)
    finally:
        cc.reset_cache()
        jax.config.update("jax_compilation_cache_dir", prev_dir)
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", prev_min)
        cc.reset_cache()

    assert results["red"] == [], (
        f"RED TWIN DID NOT GO RED: JAX wrote {results['red']} for a module "
        f"that carries a host callback.  Either jax changed _cache_write's "
        f"refusal or the twin no longer traces a callback — re-derive this "
        f"gate before trusting the green side.")
    assert results["green"], (
        "JAX wrote NO persistent-cache entry for the sunk path solver, so "
        "the whole Q-path scan will recompile on every warm run.  Look for a "
        "host callback alpha_herm_sink does not cover (jax.debug.print, "
        "io_callback, inspect_array_sharding) inside the traced region.")


# ---------------------------------------------------------------------------
# 3.  the invariant did not weaken: it still catches a non-Hermitian matvec
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_path_solver_alpha_gate_still_fires_p1(monkeypatch):
    harness.skip_unless_gpu(pytest)
    from bse.exciton_bands import _report_alpha_over_path
    monkeypatch.setenv("LORRAX_SANITY", "1")          # report, never raise

    mesh = _mesh()
    args = _operands()
    with mesh:
        solver = _green(mesh)
        _evs, alpha = solver(*args)

    labels = solver.alpha_labels
    assert len(labels) == 1, (
        f"expected exactly one collected alpha label, got {len(labels)}")
    payload = jax.device_get(alpha)
    dev, scale, worst = payload[0]
    assert np.asarray(dev).shape == (NQ,), (
        f"the alpha payload must carry ONE triple PER Q (the scan stacks it); "
        f"got shape {np.asarray(dev).shape} for {NQ} Q")

    assert _report_alpha_over_path(labels, payload) is True

    # RED TWIN of the invariant: a deliberately non-Hermitian deviation must
    # be refused.  dev/scale is the ratio the gate thresholds, so poisoning
    # dev alone is exactly the failure it exists to catch.
    bad = ((np.full(NQ, 1.0), np.asarray(scale), np.asarray(worst)),)
    assert _report_alpha_over_path(labels, bad) is False, (
        "the alpha-Hermiticity gate passed a deliberately huge non-Hermitian "
        "residual — hoisting the report out of the jit weakened the "
        "invariant")
