"""Distributed eigh conformance on the run's p x p CUDA mesh (P4 or P16).

Sizes n = p * m for m = n/p in {257, 389, 1144, 4576}: block sizes 1, 1, 143
and 208, so cuSOLVERMp's block-cyclic relabel runs at every size. Families,
built on the devices from fixed seeds:
  zero-row   [[I, M], [M^H, I]] on 3/4 of the rows, the rest exactly zero,
             ||M||_2 ~ 0.95 (the shared-pole joint CT metric's structure)
  gram       B^H B with B (n/2 x n) Gaussian: half the spectrum at rounding level
  sentinel   zero-row with its zero rows replaced by the 1/n-spaced sentinels
             of distrib_la._result_check.deflate_zero_rows (a dense cluster)
  low-rank   B^H B with B (n/4 x n): a 3/4-wide cluster at rounding level, the
             shape of the PSD responses on which cuSOLVERMp returned wrong
             vectors (n 432, rank 105); also at n = 216 p
  real       .npy matrices named on the command line (dumps of failing solves)
Per case: bare cuSOLVERMp at its default block (reported, not judged) and
distrib_la.plan('eigh', backend='distributed').batched, which must pass:
probe residual and orthogonality within ACCEPT, and the eigenvalues equal to
host LAPACK to 1e-11 * max|w| where n <= 3000. Exit 1 on any FAIL.

    lx run --pool POOL -N 1 -G 4 -n 4 -- bash -c 'source config/perlmutter/gpu_env.sh; \
        python3 -u services/distrib_la/bench/eigh_conformance_check.py [dump.npy ...]'
    (P16: -N 4 -G 4 -n 16)
"""
import sys

import numpy as np

from runtime import initialize_communicator_stack, run_main_and_finalize
RUNTIME = initialize_communicator_stack()
LOCAL_SIZES = (257, 389, 1144, 4576)
LAPACK_LIMIT = 3000


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la import _cusolvermp
    from distrib_la._result_check import ACCEPT, deflate_zero_rows, eigh_errors
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
        s = -(2 * bound + 1) * (1 + jnp.arange(n) / n)
        return a + jnp.diag(jnp.where(dead, s, 0)).astype(a.dtype)

    errors = jax.jit(eigh_errors, out_shardings=rep)
    plan_cache = {}

    def case(name, a):
        n = int(a.shape[-1])
        bare = jax.jit(lambda x: _eigh_columns("cusolvermp", *_cusolvermp.distributed_eigh(x, mesh=mesh)))
        bare_r, bare_o = (float(x) for x in errors(a, *bare(a)))
        if n not in plan_cache:
            plan_cache[n] = distrib_la.plan("eigh", mesh, n=n, backend="distributed", batched_route="auto")
        stack = jax.jit(lambda x: x[None], out_shardings=NamedSharding(mesh, P(None, "x", "y")))(a)
        w, v = jax.jit(plan_cache[n].batched)(stack)
        r, o = (float(x) for x in errors(a, w[0], v[0]))
        ok = r <= ACCEPT and o <= ACCEPT
        note = ""
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
                  f"bare residual={bare_r:.1e} orth={bare_o:.1e}{' BARE-WRONG' if max(bare_r, bare_o) > ACCEPT else ''} | "
                  f"checked residual={r:.1e} orth={o:.1e}{note} {'PASS' if ok else 'FAIL'}", flush=True)

    for m in (216,) + LOCAL_SIZES:
        n = p * m
        for family, build in (("zero-row", zero_row), ("gram", gram), ("low-rank", low_rank),
                              ("sentinel", sentinel)):
            a = jax.jit(build, static_argnums=(0, 1), out_shardings=face)(n, m)
            case(f"{family} n/p={m}", a)
            del a
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
