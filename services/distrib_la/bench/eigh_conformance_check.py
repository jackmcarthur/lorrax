"""Distributed eigh conformance on the run's p x p CUDA mesh (P4 or P16).

Sizes n = p * m for m = n/p in {257, 389, 1144, 4576}: block sizes 1, 1, 143
and 208, so cuSOLVERMp's block-cyclic relabel runs at every size. Families,
built on the devices from fixed seeds:
  zero-row   [[I, M], [M^H, I]] on 3/4 of the rows, the rest exactly zero,
             ||M||_2 ~ 0.95 (the shared-pole joint CT metric's structure)
  small-zero-row  the same times 1e-6 (1/c-scaled transverse metrics)
  gram       B^H B with B (n/2 x n) Gaussian: half the spectrum at rounding level
  sentinel   zero-row with its zero rows replaced by the 1/n-spaced sentinels
             of distrib_la._result_check.deflate_zero_rows (a dense cluster)
  low-rank   B^H B with B (n/4 x n): a 3/4-wide cluster at rounding level, the
             shape of the PSD responses on which cuSOLVERMp returned wrong
             vectors (n 432, rank 105); also at n = 216 p
  real       .npy matrices named on the command line (dumps of failing solves)
At P4 also low-rank n 9152, seeds 4576/7 x blocks 208/176/104 (the shift's cases), kept columns judged.
Per case: bare cuSOLVERMp at its default block (reported, not judged) and
distrib_la.plan('eigh', backend='distributed').batched, which must pass:
probe residual and orthogonality within accept, and the eigenvalues equal to
host LAPACK to 1e-11 * max|w| where n <= 3000. Exit 1 on any FAIL. The gram and
low-rank families are judged as their callers use them, checked above the keep
cut (CUT, w > CUT max w; the eigenvalue comparison stays over all of them); the
full check there is reported, not judged (one attempt: the near-zero cluster's
vectors may refuse it).

    lx run --pool POOL -N 1 -G 4 -n 4 -- bash -c 'source config/perlmutter/gpu_env.sh; \
        python3 -u services/distrib_la/bench/eigh_conformance_check.py [dump.npy ...]'
    (P16: -N 4 -G 4 -n 16)
"""
import sys
from functools import partial

# runtime before numpy: its first statement sets OPENBLAS_THREAD_TIMEOUT, read once when numpy maps OpenBLAS (runtime.blas_env; GATE blas_spin).
from runtime import initialize_communicator_stack, run_main_and_finalize

import numpy as np  # noqa: E402
RUNTIME = initialize_communicator_stack()
LOCAL_SIZES = (257, 389, 1144, 4576)
LAPACK_LIMIT = 3000
CUT = 1e-8          # the shared-pole sites' keep cut (normalized_gram_keep)


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la import _cusolvermp
    from distrib_la._result_check import accept, deflate_zero_rows, eigh_errors
    from distrib_la.plan import _eigh_columns
    from common.collectives import gather_to_host

    mesh = RUNTIME.mesh
    p = int(mesh.shape["x"])
    face = NamedSharding(mesh, P("x", "y"))
    rep = NamedSharding(mesh, P())
    lead = jax.process_index() == 0
    failed = []

    def zero_row(n, seed):
        kc = n // 2
        ac, at = (3 * kc) // 4, (3 * (n - kc)) // 4
        key = jax.random.key(seed)
        m = (jax.random.normal(key, (ac, at)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (ac, at)))
        u = jnp.ones((at,), m.dtype)
        for _ in range(30):                       # power iteration for ||M||_2
            u = m.conj().T @ (m @ u)
            u = u / jnp.linalg.norm(u)
        m = m * (0.95 / jnp.sqrt(jnp.linalg.norm(m.conj().T @ (m @ u))))
        a = jnp.zeros((n, n), m.dtype)
        a = a.at[:ac, :ac].set(jnp.eye(ac, dtype=m.dtype))
        a = a.at[kc:kc + at, kc:kc + at].set(jnp.eye(at, dtype=m.dtype))
        a = a.at[:ac, kc:kc + at].set(m)
        return a.at[kc:kc + at, :ac].set(m.conj().T)

    def small_zero_row(n, seed):
        return zero_row(n, seed) * 1e-6

    def gram(n, seed):
        key = jax.random.key(seed)
        b = (jax.random.normal(key, (n // 2, n)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (n // 2, n)))
        return b.conj().T @ b / n

    def low_rank(n, seed):
        key = jax.random.key(seed)
        b = (jax.random.normal(key, (n // 4, n)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (n // 4, n)))
        return b.conj().T @ b / n

    def sentinel(n, seed):
        a = zero_row(n, seed)
        dead = jnp.all(a == 0, axis=-1)
        bound = jnp.max(jnp.sum(jnp.abs(a), axis=-1))
        s = -2 * bound * (1 + jnp.arange(n) / n)
        return a + jnp.diag(jnp.where(dead, s, 0)).astype(a.dtype)

    errors = jax.jit(eigh_errors, out_shardings=rep)
    plan_cache = {}

    def case(name, a, cut=None):
        n = int(a.shape[-1])
        bare = jax.jit(lambda x: _eigh_columns("cusolvermp", *_cusolvermp.distributed_eigh(x, mesh=mesh)))
        try:
            bare_r, bare_o = (float(x) for x in errors(a, *bare(a)))
        except ValueError:      # not a solve side: the wrapper refuses it (the checked plan pads)
            bare_r = bare_o = float("nan")
        if n not in plan_cache:
            plan_cache[n] = distrib_la.plan("eigh", mesh, n=n, backend="distributed", batched_route="auto")
        stack = jax.jit(lambda x: x[None], out_shardings=NamedSharding(mesh, P(None, "x", "y")))(a)
        w, v = jax.jit(lambda s: plan_cache[n].batched(s, cut=cut))(stack)
        r, o = (float(x) for x in jax.jit(partial(eigh_errors, cut=cut), out_shardings=rep)(a, w[0], v[0]))
        ok = r <= accept(n) and o <= accept(n)
        note = "" if cut is None else f" cut={cut:g} full_check={'pass' if np.isfinite(np.asarray(gather_to_host(jax.jit(plan_cache[n].batched)(stack)[0]))).all() else 'refused'}"
        if n <= LAPACK_LIMIT:
            whole = np.asarray(gather_to_host(a))
            ref = np.linalg.eigvalsh(whole)
            got = np.asarray(gather_to_host(w))[0]
            gap = float(np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-300))
            ok = ok and gap <= 1e-11
            note = f" lapack_gap={gap:.1e}"
        if not ok:
            failed.append(name)
        if lead:
            print(f"eigh_conformance P{p * p} {name:24s} n={n:6d} block={_cusolvermp._block_size(n, p):4d} "
                  f"bare residual={bare_r:.1e} orth={bare_o:.1e}{' BARE-WRONG' if max(bare_r, bare_o) > accept(n) else ''} | "
                  f"checked residual={r:.1e} orth={o:.1e}{note} {'PASS' if ok else 'FAIL'}", flush=True)

    for m in (216,) + LOCAL_SIZES:
        n = p * m
        for family, build in (("zero-row", zero_row), ("small-zero-row", small_zero_row),
                              ("gram", gram), ("low-rank", low_rank), ("sentinel", sentinel)):
            a = jax.jit(build, static_argnums=(0, 1), out_shardings=face)(n, m)
            case(f"{family} n/p={m}", a, CUT if family in ("gram", "low-rank") else None)
            del a
    # The always-on shift's cases (distrib_la._result_check.spectral_shift): the low-rank family at
    # n 9152 (P4), seeds 4576 and 7 x cuSOLVERMp blocks 208, 176, 104, through the plan's checked
    # eigh at a forced block. Unshifted, 4 of the 6 returned wrong kept pairs (largenmu 2026-10-19).
    # The kept columns must pass; the full check is reported (cluster vectors may refuse it).
    if p == 2:
        from types import SimpleNamespace
        from distrib_la.plan import Plan
        default = _cusolvermp.solve_layout
        safe = Plan._checked_eigh(SimpleNamespace(backend="cusolvermp"), _cusolvermp.distributed_eigh,
                                  _eigh_columns)
        for seed in (4576, 7):
            a = jax.jit(low_rank, static_argnums=(0, 1), out_shardings=face)(9152, seed)
            for block in (208, 176, 104):
                _cusolvermp.solve_layout = lambda n_, p_, b=block: (n_, b)
                name = f"low-rank 9152 s{seed} b{block}"
                try:
                    w, v, status = jax.jit(lambda x: safe(x, mesh=mesh, cut=CUT))(a)
                    r, o = (float(x) for x in jax.jit(partial(eigh_errors, cut=CUT), out_shardings=rep)(a, w, v))
                    ok = not bool(np.asarray(gather_to_host(status[0]))) and max(r, o) <= accept(9152)
                    full = "pass" if np.isfinite(np.asarray(gather_to_host(
                        jax.jit(lambda x: safe(x, mesh=mesh))(a)[0]))).all() else "refused"
                except Exception as exc:            # a refusal by name
                    r = o = float("nan"); ok = False; full = type(exc).__name__
                if not ok:
                    failed.append(name)
                if lead:
                    print(f"eigh_conformance P4 {name:24s} n=  9152 block={block:4d} shifted checked "
                          f"residual={r:.1e} orth={o:.1e} cut={CUT:g} full_check={full} {'PASS' if ok else 'FAIL'}",
                          flush=True)
            del a
        _cusolvermp.solve_layout = default
    for path in sys.argv[1:]:
        whole = np.load(path)
        if whole.shape[-1] % p:
            continue
        a = jax.make_array_from_callback(whole.shape, face, lambda i: whole[i])
        case(f"real {path.rsplit('/', 1)[-1][:16]}", a)
    if lead:
        print("eigh_conformance " + ("FAILED: " + ", ".join(failed) if failed else "PASS"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    run_main_and_finalize(main)
