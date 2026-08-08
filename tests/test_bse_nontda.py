"""Non-TDA (full BSE) structure-preserving eigensolver + solver-P1 unit gates.

Small synthetic operators (host numpy / CPU jax) — no restart, no GPU — so they
run in the plain suite.  The dense-vs-solver validation on the real spinor
fixture lives in ``test_bse_dense_reference.py`` (``test_nontda_*``).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
jax.config.update("jax_enable_x64", True)


def _pos(H, k):
    ev = np.linalg.eigvals(H).real
    return np.sort(ev[ev > 1e-9])[:k]


# ---------------------------------------------------------------------------
# Structure-preserving reductions.
# ---------------------------------------------------------------------------
def test_nontda_product_real_bse():
    """Design product form: REAL BSE (A,B real symmetric, A±B PD) — the solver's
    ``omega^2 = eig((A-B)(A+B))`` reduction reproduces the dense [[A,B],[-B,-A]]
    positive spectrum with paired X²-Y²=1 eigenvectors."""
    from bse.bse_nontda import solve_nontda_product
    rng = np.random.default_rng(3); n = 30
    A = rng.standard_normal((n, n)); A = A @ A.T / n + 3.0 * np.eye(n)
    Gb = rng.standard_normal((n, n)) * 0.15; B = 0.5 * (Gb + Gb.T)
    Href = np.block([[A, B], [-B, -A]])
    ref = _pos(Href, 5)
    omega, Z = solve_nontda_product(A, B, 5)
    assert np.allclose(np.sort(omega), ref, atol=1e-8), f"{np.sort(omega)} vs {ref}"
    for j in range(5):
        X, Y = Z[:n, j], Z[n:, j]
        assert abs(float(np.real(X.conj() @ X - Y.conj() @ Y)) - 1.0) < 1e-8
        res = np.linalg.norm(Href @ Z[:, j] - omega[j] * Z[:, j]) / np.linalg.norm(Z[:, j])
        assert res < 1e-8


def test_nontda_definite_pencil_complex():
    """Physical case: complex BSE (A Hermitian, B complex-SYMMETRIC, K PD) — the
    definite-pencil solver reproduces the dense SHAO [[A,B],[-B*,-A*]] spectrum
    with X^H X - Y^H Y = +1."""
    from bse.bse_nontda import solve_nontda_definite_pencil
    rng = np.random.default_rng(5); n = 24
    Ga = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = Ga @ Ga.conj().T / n + 4.0 * np.eye(n)                 # Hermitian PD-ish
    Gb = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
    B = 0.5 * (Gb + Gb.T)                                      # complex-symmetric
    assert np.linalg.norm(B - B.T) < 1e-12 and np.linalg.norm(B - B.conj().T) > 1e-3
    Href = np.block([[A, B], [-B.conj(), -A.conj()]])
    ref = _pos(Href, 5)
    omega, Z = solve_nontda_definite_pencil(A, B, 5)
    assert np.allclose(np.sort(omega), ref, atol=1e-8), f"{np.sort(omega)} vs {ref}"
    for j in range(5):
        X, Y = Z[:n, j], Z[n:, j]
        assert abs(float(np.real(X.conj() @ X - Y.conj() @ Y)) - 1.0) < 1e-8
        res = np.linalg.norm(Href @ Z[:, j] - omega[j] * Z[:, j]) / np.linalg.norm(Z[:, j])
        assert res < 1e-8


def test_nontda_positive_definiteness_check():
    """(A-B)/K not positive definite (triplet/charge instability) must RAISE
    with a clear message, not silently return imaginary energies."""
    from bse.bse_nontda import solve_nontda_definite_pencil, solve_nontda_product
    n = 10
    A = np.diag(np.linspace(0.1, 1.0, n)).astype(np.complex128)
    B = 2.0 * np.eye(n)                                        # A-B < 0 => K indefinite
    with pytest.raises(ValueError, match="positive definite"):
        solve_nontda_definite_pencil(A, B, 3)
    with pytest.raises(ValueError, match="positive definite"):
        solve_nontda_product(A, B, 3)


# ---------------------------------------------------------------------------
# Solver P1 — block-Lanczos final-slot overwrite regression.
# ---------------------------------------------------------------------------
def test_block_lanczos_eigenvector_residual_p1():
    """P1: with the +1 Krylov slot the last block is retained (not clobbered by
    the final Q_next), so at Krylov = N block Lanczos reconstructs eigenVECTORS
    with tiny residual.  The pre-fix final-slot overwrite corrupted this block."""
    from solvers.lanczos import block_lanczos_eig_jit
    rng = np.random.default_rng(0); n = 40
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (G + G.conj().T)
    Hj = jnp.asarray(H)

    def matvec(V):                    # (bs, n) -> (bs, n)
        return (Hj @ V.T).T

    bs = 4; m = n // bs               # Krylov = bs*m = n (full space)
    ev, evec = block_lanczos_eig_jit(matvec, n, n_eig=4, block_size=bs,
                                     max_iter=m, n_reorth=m)
    ev = np.asarray(jax.device_get(ev)); evec = np.asarray(jax.device_get(evec))
    ref = np.sort(np.linalg.eigvalsh(H))[:4]
    assert np.allclose(np.sort(ev.real), ref, atol=1e-8), f"{np.sort(ev.real)} vs {ref}"
    for i in range(4):
        v = evec[i]; lam = ev[i]
        res = np.linalg.norm(H @ v - lam * v) / max(np.linalg.norm(v), 1e-30)
        assert res < 1e-6, f"state {i} eigenvector residual {res:.2e} (final-slot P1)"


def test_lanczos_jit_final_slot_shapes_p1():
    """P1: the single-vector and converged block variants also carry the +1 slot
    and return correctly-shaped, correct eigenpairs on a synthetic Hermitian op."""
    from solvers.lanczos import lanczos_eig_jit, block_lanczos_eig_jit_converged
    rng = np.random.default_rng(1); n = 30
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (G + G.conj().T); Hj = jnp.asarray(H)
    ref = np.sort(np.linalg.eigvalsh(H))[:3]

    ev, evec = lanczos_eig_jit(lambda v: Hj @ v, n, n_eig=3, max_iter=n, n_reorth=n)
    ev = np.asarray(jax.device_get(ev))
    assert evec.shape == (3, n)
    assert np.allclose(np.sort(ev.real), ref, atol=1e-8)

    def mvb(V):
        return (Hj @ V.T).T
    evb, evecb, nit = block_lanczos_eig_jit_converged(
        mvb, n, n_eig=3, block_size=3, max_iter=n // 3, rtol=1e-10, n_reorth=n)
    evb = np.asarray(jax.device_get(evb))
    assert np.allclose(np.sort(evb.real), ref, atol=1e-6)


# ---------------------------------------------------------------------------
# The α-Hermiticity gate must not make its enclosing jit uncacheable.
#
# ``jax/_src/compiler.py::_cache_write`` refuses outright:
#
#     if host_callbacks:
#       logger.log(log_priority, "Not writing persistent cache entry for '%s' "
#                  "because it uses host callbacks (e.g. from jax.debug.print "
#                  "or breakpoint)", module_name)
#       return
#
# so ONE unordered three-scalar ``jax.debug.callback`` inside
# ``bse_lanczos.solve_bse_sharded``'s ``_full_run`` — the single program that
# holds the whole Krylov loop — made that module permanently unpersistable.
# MEASURED on the Si 4x4x4 SOC P=4 reference deck: ``cache_probes=37 hits=36
# vetoed=1`` on every warm run, i.e. a 2.1 s XLA compile paid every time
# (12% of the 17.9 s wall).  ``solvers.lanczos.alpha_herm_sink`` routes the
# same three scalars out as jit OUTPUTS instead.
#
# Both cells below carry their RED TWIN — the same jit traced WITHOUT the
# sink — because a gate that cannot fail is not a gate.
# ---------------------------------------------------------------------------
def _sub_jaxprs(obj):
    """Every jaxpr reachable from an eqn param (branches, bodies, calls)."""
    out = []
    if hasattr(obj, "eqns"):                       # jax.core.Jaxpr
        out.append(obj)
    elif hasattr(obj, "jaxpr") and hasattr(obj.jaxpr, "eqns"):   # ClosedJaxpr
        out.append(obj.jaxpr)
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            out.extend(_sub_jaxprs(item))
    return out


def _callback_prims(jaxpr):
    """Names of every callback-ish primitive in ``jaxpr``, recursively."""
    found = []
    for eqn in jaxpr.eqns:
        if "callback" in eqn.primitive.name:
            found.append(eqn.primitive.name)
        for param in eqn.params.values():
            for sub in _sub_jaxprs(param):
                found.extend(_callback_prims(sub))
    return found


def _herm_op(n=24, seed=7):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return 0.5 * (G + G.conj().T)


def _krylov_pair(n=24):
    """(sunk, unsunk) — the same Krylov solve with and without the sink."""
    from solvers.lanczos import (alpha_herm_sink, lanczos_eig_jit,
                                 split_alpha_sink)
    labels_box: list = []

    def sunk(Hm):
        with alpha_herm_sink() as sink:
            evs, _ = lanczos_eig_jit(lambda v: Hm @ v, n, n_eig=3,
                                     max_iter=n, n_reorth=n)
        labels, payload = split_alpha_sink(sink)
        labels_box[:] = labels
        return evs, payload

    def unsunk(Hm):          # the RED TWIN: the pre-fix shape
        evs, _ = lanczos_eig_jit(lambda v: Hm @ v, n, n_eig=3,
                                 max_iter=n, n_reorth=n)
        return evs

    return sunk, unsunk, labels_box


def test_krylov_jit_carries_no_host_callback_p1():
    """Sunk trace has zero callback primitives; the red twin has exactly one."""
    n = 24
    Hj = jnp.asarray(_herm_op(n))
    sunk, unsunk, _labels = _krylov_pair(n)

    green = _callback_prims(jax.make_jaxpr(sunk)(Hj).jaxpr)
    red = _callback_prims(jax.make_jaxpr(unsunk)(Hj).jaxpr)

    assert red, ("RED TWIN DID NOT GO RED: the un-sunk Krylov trace shows no "
                 "callback primitive, so this gate is testing nothing.  Did "
                 "the alpha-Hermiticity gate get deleted rather than moved?")
    assert green == [], (
        f"jit wrapping a Krylov solve still traces {green} under "
        f"alpha_herm_sink.  JAX will not write a persistent-cache entry for "
        f"any module carrying a host callback, so this program recompiles on "
        f"every warm run.")


def test_krylov_jit_alpha_gate_still_fires_p1(monkeypatch):
    """Hoisting the report out of the jit did not weaken the invariant."""
    from solvers.lanczos import report_alpha_herm
    monkeypatch.setenv("LORRAX_SANITY", "1")     # report, never raise
    n = 24
    H = _herm_op(n)
    sunk, _unsunk, labels = _krylov_pair(n)

    _evs, payload = jax.jit(sunk)(jnp.asarray(H))
    assert len(labels) == 1 and len(payload) == 1, (
        f"expected exactly one collected alpha report, got "
        f"{len(labels)} labels / {len(payload)} payloads")
    assert report_alpha_herm(labels, payload) is True

    # ...and it still says NO when the operator is not Hermitian.  H + i·c·I
    # keeps every eigenvector but makes <q,Hq> carry Im = c·|q|², which is
    # exactly the failure the gate exists to catch.
    bad = jnp.asarray(H + 1j * 0.1 * np.eye(n))
    _evs_bad, payload_bad = jax.jit(sunk)(bad)
    assert report_alpha_herm(labels, payload_bad) is False, (
        "the alpha-Hermiticity gate passed a deliberately non-Hermitian "
        "operator — the invariant is no longer live")


def test_krylov_jit_is_written_to_the_compile_cache_p1(tmp_path):
    """JAX's own persistence decision, taken on a real cache directory.

    This is the cell the whole fix exists for: the sunk program gets an entry
    written, the red twin gets none.  It exercises ``_cache_write``'s refusal
    directly rather than a proxy for it.
    """
    from jax._src import compilation_cache as cc

    n = 24
    Hj = jnp.asarray(_herm_op(n))
    sunk, unsunk, _labels = _krylov_pair(n)

    prev_dir = jax.config.jax_compilation_cache_dir
    prev_min = jax.config.jax_persistent_cache_min_compile_time_secs

    def _entries(d):
        return sorted(p.name for p in d.iterdir() if p.is_file())

    try:
        for name, fn in (("green", sunk), ("red", unsunk)):
            d = tmp_path / name
            d.mkdir()
            cc.reset_cache()
            jax.config.update("jax_compilation_cache_dir", str(d))
            jax.config.update(
                "jax_persistent_cache_min_compile_time_secs", 0.0)
            jax.jit(fn).lower(Hj).compile()
            cc.reset_cache()
            if name == "green":
                green_entries = _entries(d)
            else:
                red_entries = _entries(d)
    finally:
        cc.reset_cache()
        jax.config.update("jax_compilation_cache_dir", prev_dir)
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", prev_min)
        cc.reset_cache()

    assert red_entries == [], (
        f"RED TWIN DID NOT GO RED: JAX wrote {red_entries} for a module that "
        f"carries a host callback.  Either jax changed _cache_write's refusal "
        f"or the red twin no longer traces a callback — re-derive this gate "
        f"before trusting the green side.")
    assert green_entries, (
        "JAX wrote NO persistent-cache entry for the sunk Krylov jit, so it "
        "will recompile on every warm run.  Check for a host callback that "
        "alpha_herm_sink does not cover (jax.debug.print, io_callback, "
        "jax.debug.inspect_array_sharding) inside the traced region.")
