"""The shared-pole partner cut checks only the eigh columns it reads (CPU, 4 devices).

An ISDF basis above the pair rank (Fe V2: N_mu 1872 against a pair rank of about 1482 on
the 25^3 grid) gives the partner operands top = O O^H and perp = R R^H an exact null
cluster that is most of their spectrum. The face-route eigh check used to probe every
column, so a solver whose vectors are poor only inside that cluster refused the run
(GATE distrib_la_result_check at n = N_mu). The cut reads top's largest eigenvalue and
perp's columns above cutoff**2 * max eig(top); the check now covers exactly those
(and their boundary multiplet), and a defect on any of them still refuses.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

N, R, RQ = 48, 14, 6           # n = N_mu, columns of O, columns of Q: null cluster of N - R


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices")
    return jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _pair(rng, scale=1.0):
    """(top, perp) of the dedupe kernel: O [N, R] of decaying columns, Q [N, RQ] orthonormal."""
    o = (rng.standard_normal((N, R)) + 1j * rng.standard_normal((N, R))) * np.geomspace(1, 1e-4, R)
    q, _ = np.linalg.qr(rng.standard_normal((N, RQ)) + 1j * rng.standard_normal((N, RQ)))
    rest = o - q @ (q.conj().T @ o)
    herm = lambda a: (a + a.conj().T) / 2
    return herm(o @ o.conj().T) * scale, herm(rest @ rest.conj().T) * scale


def _stacks(b=4, seed=11):
    rng = np.random.default_rng(seed)
    pairs = [_pair(rng, 1 + 0.1 * i) for i in range(b)]
    return np.asarray([t for t, _ in pairs]), np.asarray([p for _, p in pairs])


def _extent(k):
    return max(2, -(-int(k) // 2) * 2)


def _host_counts(top, perp, cutoff, real):
    lt = np.linalg.eigvalsh(top)[:, -1]
    lp = np.linalg.eigvalsh(perp)
    return tuple(int(np.sum(lp[i] > cutoff ** 2 * lt[i])) if i < real else 0 for i in range(len(top)))


def test_the_cut_covers_every_column_read_and_leaves_out_the_null_cluster():
    from distrib_la._result_check import eigh_errors
    from distrib_la.tolerance import roundoff_tol
    from distrib_la.polar import PARTNER_CHECK_CUT
    cutoff = 1e-3
    top, perp = _stacks(1)
    top, perp = top[0], perp[0]
    lp, vp = np.linalg.eigh(perp)
    assert lp[-1] <= np.linalg.eigvalsh(top)[-1]                        # the premise, by construction
    cut = PARTNER_CHECK_CUT * cutoff ** 2
    used = lp > cut * lp[-1]
    read = lp > cutoff ** 2 * np.linalg.eigvalsh(top)[-1]
    assert read.sum() >= 1 and np.all(used[read])                       # every column read is checked
    null = np.flatnonzero(np.abs(lp) < 1e-12 * lp[-1])
    assert null.size >= N - R and not used[null].any()                  # the null cluster is not

    def errors(vectors, c):
        return tuple(float(e) for e in eigh_errors(jnp.asarray(perp), jnp.asarray(lp), jnp.asarray(vectors), cut=c))
    accept = roundoff_tol(N, perp.dtype)
    bad = vp.copy()
    bad[:, null[1:]] += 0.05 * bad[:, null[:1]]                         # non-orthonormal null cluster
    assert max(errors(bad, cut)) < accept
    assert max(errors(bad, None)) > accept
    kept = vp.copy()
    kept[:, np.flatnonzero(read)[-1]] += 0.05 * kept[:, null[0]]       # the same defect on a column read
    assert max(errors(kept, cut)) > accept


def _corrupting(native, where):
    """A local solver whose vectors are corrupted in the columns ``where(values)`` selects."""
    def eigh(a, *args, **kwargs):
        w, v = native(a, *args, **kwargs)
        sel = where(w)                                                  # [..., n] bool
        first = jnp.argmax(sel, axis=-1)                                # the first selected column
        idx = jnp.broadcast_to(first[..., None, None], v.shape[:-1] + (1,))
        anchor = jnp.take_along_axis(v, idx, axis=-1)
        later = sel & (jnp.arange(v.shape[-1]) > first[..., None])
        return w, v + 0.05 * anchor * later[..., None, :].astype(v.dtype)
    return eigh


def _run(monkeypatch, where, cutoff=1e-3, real=3):
    import distrib_la
    import distrib_la._batch_reshard as br
    import importlib
    dplan = importlib.import_module("distrib_la.plan")   # the package exports a function named plan
    import distrib_la._result_check as rc
    monkeypatch.setenv("LORRAX_CHECK_COUNTS", "1")
    for mod in (br, rc):
        monkeypatch.setattr(mod, "native_eigh", _corrupting(rc.native_eigh, where))
    br._JIT_CACHE.clear()
    dplan._reshard_stack_program.cache_clear()
    mesh = _mesh()
    top, perp = _stacks(4)
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))
    plan = distrib_la.plan("eigh", mesh, n=N, backend="off")
    q, counts = distrib_la.leading_eigenvectors_above(
        put(perp), put(top), cutoff, real=real, eigh_plan=plan, column_extent=_extent,
        multiplet_tol=1e-6)
    return top, perp, q, counts


def _null(w):
    return jnp.abs(w) < 1e-12 * jnp.max(jnp.abs(w), axis=-1, keepdims=True)


def _assert_cut(top, perp, q, counts, cutoff, real):
    assert counts == _host_counts(top, perp, cutoff, real) and max(counts) > 0
    q = np.asarray(q)
    for i in range(real):                                               # the kept span is perp's leading span
        lp, vp = np.linalg.eigh(perp[i])
        want = vp[:, ::-1][:, :counts[i]]
        got = q[i][:, :counts[i]]
        np.testing.assert_allclose(got @ got.conj().T, want @ want.conj().T, atol=1e-9)


def test_a_defective_null_cluster_is_not_checked_and_the_cut_is_unchanged(monkeypatch, capfd):
    cutoff, real = 1e-3, 3
    top, perp, q, counts = _run(monkeypatch, _null, cutoff, real)
    _assert_cut(top, perp, q, counts, cutoff, real)
    assert "failed its result check" not in capfd.readouterr().err      # no retry: the cluster is unread


def test_a_defect_on_a_column_read_is_still_checked(monkeypatch, capfd):
    """The same corruption on columns the cut reads fails the check on every attempt that
    uses the corrupted solver; the chain's gathered solve repairs it, else it refuses."""
    def leading(w):                                                     # the two largest columns
        return w >= jnp.sort(w, axis=-1)[..., -2:-1]
    cutoff, real = 1e-3, 3
    top, perp, q, counts = _run(monkeypatch, leading, cutoff, real)
    err = capfd.readouterr().err
    assert "failed its result check" in err and "attempt 1 of 2 failed" in err
    _assert_cut(top, perp, q, counts, cutoff, real)


def test_a_perp_above_top_refuses_by_name():
    import distrib_la
    mesh = _mesh()
    top, perp = _stacks(4)
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))
    plan = distrib_la.plan("eigh", mesh, n=N, backend="off")
    with pytest.raises(ValueError, match="partner_cut_check_cover"):
        distrib_la.leading_eigenvectors_above(put(perp * 1e3), put(top), 1e-3, real=3, eigh_plan=plan,
                                              column_extent=_extent, multiplet_tol=1e-6)
