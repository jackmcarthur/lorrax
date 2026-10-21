"""The native active-subspace provider (``lorrax_active_subspace_*``) against the XLA and
host routes on the same device.

``distrib_la.subspace.SUBSPACE_ROUTE`` sends a device leg to ``LocalSubspacePlan`` (the
leg-neutral handlers of ``src/ffi/cpp/active_subspace``: cuBLAS/cuSOLVERDn on CUDA,
rocBLAS/rocSOLVER on ROCM).  Gate (``ffi.common.families`` row ``active_subspace``, tol
1e-12 relative to max|XLA|), per case, native vs ``XlaSubspacePlan`` (the reference route)
and vs ``CpuSubspacePlan`` (host LAPACK) on the same device:

* project, reconstruct (with and without the image), Gram, CGS2 orthogonalization,
  subtract_projection (plain and fused with the next Gram) and store, for active prefixes
  and runtime [start, start+count) windows, with the inactive rows poisoned with NaN (they
  must never enter arithmetic) and one compilation per plan for every window;
* the projected eigensolve: eigenvalues vs both routes, residual ``|H c - c e|``,
  orthonormality, eigenvectors compared as rank-one projectors per (non-degenerate) column,
  zero coefficients and +inf values outside the active block, capacity 300 with active
  sizes 1..300 and a real-valued block;
* normalize (the Davidson whitening) ignoring a NaN tail;
* a complete fixed-capacity Davidson solve (``solvers.davidson_fixed``) on a synthetic
  tridiagonal operator: both routes CONVERGED, native eigenvalues vs the XLA plan's, and each
  an eigenvalue of the operator (scipy, whole spectrum);
* refusal: an invalid window raises (the native handlers refuse; they never clamp).

ROCm/CUDA: ``python -m tests.test_subspace_native_gate`` on one device prints one line per
check and ``GATE active_subspace PASS|FAIL max_rel=...``.  Under pytest without a native
device leg the cases skip.
"""
import sys

# runtime before numpy (GATE blas_spin): it sets the OpenBLAS env numpy reads once.
import runtime  # noqa: F401,E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from distrib_la.subspace import CpuSubspacePlan, XlaSubspacePlan  # noqa: E402

TOL = 1e-12
CAP, D, B = 64, 4096, 12
_REPORT = {}


def _native(cap, n_eig, max_block_size=None):
    from distrib_la.active_subspace import plan_local_subspace
    return plan_local_subspace(capacity=cap, n_eig=n_eig, max_block_size=max_block_size)


def _native_available():
    try:
        _native(8, 2)
        return True, ""
    except Exception as exc:                                   # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _rel(name, got, want, tol=TOL):
    got, want = np.asarray(got), np.asarray(want)
    scale = max(float(np.max(np.abs(want))), 1e-300)
    err = float(np.max(np.abs(got - want))) / scale if got.size else 0.0
    _REPORT[name] = max(_REPORT.get(name, 0.0), err)
    assert np.isfinite(got).all() and err <= tol, f"{name}: rel {err:.3e} > {tol:g}"


def _problem(seed=78, cap=CAP, d=D, b=B):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(d, cap)) + 1j * rng.normal(size=(d, cap))
    v = np.linalg.qr(z)[0].T.copy()
    hv = v * (np.arange(d) / d)[None, :]
    c = rng.normal(size=(cap, b)) + 1j * rng.normal(size=(cap, b))
    p = rng.normal(size=(b, d)) + 1j * rng.normal(size=(b, d))
    return v, hv, c, p


def _poison(a, start, stop):
    a = np.array(a, copy=True)
    a[:start], a[stop:] = np.nan, np.nan
    return jnp.asarray(a)


def check_prefix_ops(active):
    v, hv, c, p = _problem()
    nat, xla, host = _native(CAP, B), XlaSubspacePlan(CAP, B), CpuSubspacePlan(CAP, B)
    vv, hh, cc = (_poison(a, 0, active) for a in (v, hv, c))
    pj = jnp.asarray(p)
    h = jnp.full((CAP, CAP), jnp.nan, jnp.complex128)
    proj = np.asarray(jax.jit(nat.project)(vv, hh, active, h, 0, active))
    px = np.asarray(jax.jit(xla.project)(vv, hh, active, h, 0, active))
    _rel("project", proj[:active, :active], px[:active, :active])
    _rel("project_host", proj[:active, :active],
         np.asarray(host.project(vv, hh, active, h, 0, active))[:active, :active])
    assert np.isnan(proj[active:, active:]).all(), "project wrote outside the active block"
    pd = jnp.asarray(proj)
    e, coeff = (np.asarray(a) for a in jax.jit(nat.eigh)(pd, active))
    ex, cx = (np.asarray(a) for a in jax.jit(xla.eigh)(pd, active))
    eh, ch = (np.asarray(a) for a in host.eigh(pd, active))
    k = min(active, B)
    hm = proj[:active, :active]
    _rel("eigh_values", e[:k], ex[:k])
    _rel("eigh_values_host", e[:k], eh[:k])
    assert np.isinf(e[k:]).all() and not np.any(coeff[active:]) and not np.any(coeff[:, k:])
    scale = max(np.abs(hm).max(), 1e-300)
    _rel("eigh_residual", hm @ coeff[:active, :k] / scale, coeff[:active, :k] * e[:k] / scale, 1e-11)
    _rel("eigh_orthonormal", coeff[:active, :k].conj().T @ coeff[:active, :k], np.eye(k), 1e-11)
    gaps = np.diff(eh[:min(active, B + 1)]) if active > 1 else np.array([np.inf])
    for j in range(k):
        lo = gaps[j - 1] if j > 0 else np.inf
        hi = gaps[j] if j < len(gaps) else np.inf
        if min(lo, hi) > 1e-6 * scale:     # a non-degenerate column: compare projectors
            a, r = coeff[:active, j], ch[:active, j]
            _rel("eigh_projector", np.outer(a, a.conj()), np.outer(r, r.conj()), 1e-9)
    x, hx = jax.jit(nat.reconstruct)(vv, hh, cc, active, pj)
    xx, hxx = jax.jit(xla.reconstruct)(vv, hh, cc, active, pj)
    _rel("reconstruct", x, xx)
    _rel("reconstruct_image", hx, hxx)
    _rel("reconstruct_host", x, host.reconstruct(vv, hh, cc, active, pj)[0])
    pn = jax.jit(nat.orthogonalize)(vv, pj, active)
    _rel("orthogonalize", pn, jax.jit(xla.orthogonalize)(vv, pj, active))
    _rel("orthogonalize_host", pn, host.orthogonalize(vv, pj, active))
    g = jax.jit(nat.gram)(vv, pj, active)
    _rel("gram", g, jax.jit(xla.gram)(vv, pj, active))
    _rel("gram_host", g, host.gram(vv, pj, active))


def check_windowed_ops(start, count):
    v, hv, c, p = _problem(419)
    nat, xla, host = _native(CAP, B), XlaSubspacePlan(CAP, B), CpuSubspacePlan(CAP, B)
    vv, cc = _poison(v, start, start + count), _poison(c, start, start + count)
    pj = jnp.asarray(p)
    i32 = jnp.int32
    ortho = jax.jit(nat.orthogonalize).lower(vv, pj, i32(6), start=i32(2)).compile()
    got = ortho(vv, pj, i32(count), start=i32(start))
    _rel("w_orthogonalize", got, xla.orthogonalize(vv, pj, i32(count), start=i32(start)))
    _rel("w_orthogonalize_host", got, host.orthogonalize(vv, pj, count, start=start))
    gram = jax.jit(nat.gram).lower(vv, pj, i32(6), start=i32(2)).compile()
    got = gram(vv, pj, i32(count), start=i32(start))
    want = host.gram(vv, pj, count, start=start)
    if count:
        _rel("w_gram", got, xla.gram(vv, pj, i32(count), start=i32(start)))
        _rel("w_gram_host", got, want)
    else:
        assert not np.any(np.asarray(got))
    rec = jax.jit(nat.reconstruct, static_argnames=("compute_image",)).lower(
        vv, vv, cc, i32(6), pj, columns=i32(B), start=i32(2), compute_image=False).compile()
    got = rec(vv, vv, cc, i32(count), pj, columns=i32(5), start=i32(start))[0]
    want = host.reconstruct(vv, vv, cc, count, pj, columns=5, start=start, compute_image=False)[0]
    if count:
        _rel("w_reconstruct", got, want)
    else:
        assert not np.any(np.asarray(got))
    # subtract_projection: P -= C[start:start+count]^T V[start:start+count], optionally
    # returning the next process-local Gram panel (zero outside the window).
    coef = jnp.asarray(np.where(np.isfinite(np.asarray(cc)), np.asarray(cc), 0))
    sub = jax.jit(nat.subtract_projection, static_argnames=("next_gram",))
    got = sub(vv, pj, coef, i32(count), start=i32(start))
    vw, cw = v[start:start + count], np.asarray(coef)[start:start + count]
    want = p - (cw.T @ vw if count else 0)
    _rel("w_subtract", got, want)
    got, nxt = sub(vv, pj, coef, i32(count), start=i32(start), next_gram=True)
    _rel("w_subtract_gram_p", got, want)
    ref = np.zeros((CAP, B), complex)
    if count:
        ref[start:start + count] = vw.conj() @ want.T
        _rel("w_subtract_gram_c", nxt, ref)
    else:
        assert not np.any(np.asarray(nxt))
    stop = start + count
    if stop:
        hh, vp = _poison(hv, 0, stop), _poison(v, 0, stop)
        h0 = jnp.asarray(np.random.default_rng(1).normal(size=(CAP, CAP)) + 0j)
        proj = jax.jit(nat.project).lower(vp, hh, i32(9), h0, i32(2), i32(3)).compile()
        got = proj(vp, hh, i32(stop), h0, i32(start), i32(count))
        _rel("w_project", got, host.project(vp, hh, stop, h0, start, count))
        _rel("w_project_xla", got, xla.project(vp, hh, i32(stop), h0, i32(start), i32(count)))


def check_store():
    v, hv, c, p = _problem(7)
    nat, host = _native(CAP, B), CpuSubspacePlan(CAP, B)
    pj, hp = jnp.asarray(p), jnp.asarray(2 * p)
    for start, count in ((0, 12), (40, 5), (52, 12), (10, 0)):
        got = jax.jit(nat.store)(jnp.asarray(v), jnp.asarray(hv), pj, hp, start, count)
        want = host.store(jnp.asarray(v), jnp.asarray(hv), pj, hp, start, count)
        for g, w in zip(got, want):
            assert np.array_equal(np.asarray(g), np.asarray(w)), f"store ({start},{count})"
    _REPORT["store_bytes_equal"] = 0.0


def check_eigh_buckets():
    cap, n_eig = 300, 7
    rng = np.random.default_rng(3)
    a = rng.normal(size=(cap, cap))
    a = (a + a.T) / 2 + 0j
    nat, xla, host = _native(cap, n_eig), XlaSubspacePlan(cap, n_eig), CpuSubspacePlan(cap, n_eig)
    run = jax.jit(nat.eigh).lower(jnp.asarray(a), jnp.int32(1)).compile()
    runx = jax.jit(xla.eigh).lower(jnp.asarray(a), jnp.int32(1)).compile()
    for m in (1, 5, 64, 65, 129, 200, 256, 257, 300):
        h = a.copy()
        h[m:, :], h[:, m:] = np.nan, np.nan
        e, vec = (np.asarray(t) for t in run(jnp.asarray(h), jnp.int32(m)))
        ex, _ = runx(jnp.asarray(h), jnp.int32(m))
        eh, _ = host.eigh(jnp.asarray(h), m)
        k = min(m, n_eig)
        _rel("eigh300_values", e[:k], np.asarray(ex)[:k])
        _rel("eigh300_values_host", e[:k], np.asarray(eh)[:k])
        sc = np.abs(a[:m, :m]).max()
        _rel("eigh300_residual", a[:m, :m] @ vec[:m, :k] / sc, vec[:m, :k] * e[:k] / sc, 1e-11)
        _rel("eigh300_orthonormal", vec[:m, :k].conj().T @ vec[:m, :k], np.eye(k), 1e-11)


def check_eigh_structured():
    """Structured blocks rocSOLVER zheevd fails on unreflected (n >= 96, local_eigh docstring):
    a real Lanczos tridiagonal, a real diagonal with repeats, a scalar block, a block-diagonal."""
    cap, n_eig = 160, 6
    rng = np.random.default_rng(11)
    nat, host = _native(cap, n_eig), CpuSubspacePlan(cap, n_eig)
    run = jax.jit(nat.eigh).lower(jnp.zeros((cap, cap), jnp.complex128), jnp.int32(1)).compile()
    m = 120
    tri = np.diag(rng.normal(size=m)) + np.diag(rng.uniform(0.1, 1, m - 1), 1)
    tri = tri + np.triu(tri, 1).T
    diag = np.diag(np.repeat(rng.normal(size=m // 4), 4))
    blk = np.zeros((m, m))
    blk[:60, :60] = (lambda x: x + x.T)(rng.normal(size=(60, 60)))
    blk[60:, 60:] = (lambda x: x + x.T)(rng.normal(size=(60, 60)))
    for name, a in (("tridiagonal", tri), ("diagonal", diag), ("scalar", 2.5 * np.eye(m)),
                    ("blockdiag", blk)):
        h = np.full((cap, cap), np.nan, complex)
        h[:m, :m] = a
        e, vec = (np.asarray(t) for t in run(jnp.asarray(h), jnp.int32(m)))
        eh, _ = host.eigh(jnp.asarray(h), m)
        _rel(f"structured_{name}_values", e, np.asarray(eh)[:n_eig])
        sc = max(np.abs(a).max(), 1e-300)
        _rel(f"structured_{name}_residual", a @ vec[:m] / sc, vec[:m] * e / sc, 1e-11)
        _rel(f"structured_{name}_orthonormal", vec[:m].conj().T @ vec[:m], np.eye(n_eig), 1e-11)


def check_normalize():
    plan = _native(12, 12)
    p = jnp.eye(32, dtype=jnp.complex128)[:12]
    p = p.at[1].multiply(2.).at[2:].set(jnp.nan)
    result = np.asarray(jax.jit(plan.normalize)(p, jnp.int32(2)))
    assert np.isfinite(result).all() and not np.any(result[2:])
    _rel("normalize", result[:2].conj() @ result[:2].T, np.eye(2), 1e-13)


def check_refusal():
    v, hv, c, p = _problem()
    nat = _native(CAP, B)
    # blas handlers raise; devsize kernels (device-side descriptors) poison the output.
    try:
        out = np.asarray(jax.jit(nat.orthogonalize)(jnp.asarray(v), jnp.asarray(p),
                                                    jnp.int32(60), start=jnp.int32(8)))
    except Exception:
        return
    assert np.isnan(out).all(), "invalid window neither raised nor poisoned the output"


# The hopping keeps the diagonal preconditioner approximate: on an exactly diagonal operator
# it maps the residual back onto the Ritz vector and Davidson stops with NO_DIRECTIONS.
COUPLING = 0.05


def _davidson(route, n=6000, n_eig=4, capacity=24, seed=0):
    """One fixed-capacity Davidson solve on diag(d) with route ``route``; returns (plan, args)."""
    import distrib_la.subspace as sub
    from solvers.davidson_fixed import plan_local_davidson
    saved = dict(sub.SUBSPACE_ROUTE)
    leg = sub.native_leg().key
    sub.SUBSPACE_ROUTE[leg] = route
    try:
        def precondition(d, r, e, x):
            den = jnp.real(d)[None, :] - e[:, None]
            return r / jnp.where(jnp.abs(den) < 1e-2, 1e-2, den)
        def apply_h(d, x):   # diag(d) + COUPLING * (nearest-neighbour hopping), Hermitian
            y = x * d[None, :]
            y = y.at[:, 1:].add(COUPLING * x[:, :-1])
            return y.at[:, :-1].add(COUPLING * x[:, 1:])
        plan = plan_local_davidson(apply_h, precondition,
                                   n_eig=n_eig, capacity=capacity, vector_shape=(n,))
    finally:
        sub.SUBSPACE_ROUTE.clear()
        sub.SUBSPACE_ROUTE.update(saved)
    rng = np.random.default_rng(seed)
    # The wanted roots well separated (0.1 .. 0.1 n_eig) below a bulk in [1.5, 10).
    d = jnp.asarray(np.concatenate([0.1 * np.arange(1, n_eig + 1),
                                    rng.uniform(1.5, 10.0, n - n_eig)]) + 0j)
    x0 = jnp.asarray(rng.normal(size=(n_eig, n)) + 1j * rng.normal(size=(n_eig, n)))
    from scipy.linalg import eigh_tridiagonal
    # The whole spectrum: a converged block holds n_eig eigenvalues of the operator (Davidson
    # from a random start may converge to the bulk edge rather than the isolated lowest roots;
    # that is the solver's, identical on every route, not the provider's).
    exact = eigh_tridiagonal(np.asarray(d).real, np.full(n - 1, COUPLING), eigvals_only=True)
    return plan, (d, x0), exact


def check_davidson():
    out, status = {}, {}
    for route in ("native", "xla"):
        plan, args, exact = _davidson(route)
        vals, _, info = plan.solve(*args, 1e-8, 400)   # solve is jitted with in_shardings: positional
        out[route] = np.asarray(vals)
        status[route] = (int(info.status), int(info.iterations), int(info.matvecs))
        print(f"  davidson {route}: status {status[route][0]} iterations {status[route][1]} "
              f"matvecs {status[route][2]} (status 1 = CONVERGED)", flush=True)
    assert all(v[0] == 1 for v in status.values()), status
    _rel("davidson_values", out["native"], out["xla"], 1e-10)
    nearest = exact[np.abs(exact[None, :] - out["native"][:, None]).argmin(axis=1)]
    _rel("davidson_in_spectrum", out["native"], nearest, 1e-10)


CHECKS = ([(f"prefix active={a}", lambda a=a: check_prefix_ops(a)) for a in (1, 12, 17, 35, 64)]
          + [(f"window start={s} count={c}", lambda s=s, c=c: check_windowed_ops(s, c))
             for s, c in ((0, 0), (5, 0), (3, 5), (7, 33), (40, 24), (0, 64))]
          + [("store", check_store), ("eigh capacity 300", check_eigh_buckets),
             ("eigh structured", check_eigh_structured),
             ("normalize", check_normalize), ("refusal", check_refusal),
             ("davidson solve", check_davidson)])


def main():
    ok, why = _native_available()
    print(f"device {jax.devices()[0]} ({jax.devices()[0].platform}); native: {ok} {why}", flush=True)
    if not ok:
        print("GATE active_subspace FAIL (no native provider)")
        return 1
    failed = []
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS {name}", flush=True)
        except Exception as exc:                               # noqa: BLE001
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:400]}", flush=True)
    for k in sorted(_REPORT):
        print(f"  rel[{k}] = {_REPORT[k]:.3e}")
    worst = max(_REPORT.values()) if _REPORT else float("nan")
    print(f"GATE active_subspace {'FAIL' if failed else 'PASS'} max_rel={worst:.3e} "
          f"failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
