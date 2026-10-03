"""P4 check: distrib_la's eigh on a matrix with an exact-zero block (2x2 CUDA mesh).

cuSOLVERMp's syevd returns wrong eigenpairs with info 0 on some Hermitian
matrices with a large exact-zero block. The matrix here has the structure of
the shared-pole joint CT metric that refused on Fe 4^3 bispinor (claim 3148):
[[I, M], [M^H, I]] on 990 + 1018 active rows of a 1152 + 1536 carrier (680
exact-zero rows at each block's tail), M random with ||M||_2 = 0.988, seed 1.
On it, bare cuSOLVERMp returned residual 2.0e-2 and orthogonality 5.5e-2 while
its min/max eigenvalue ratio, -2.7e-8, passed the constructor's Gram gate.
That bare failure depends on the matrix's last bits (the SVD-based scaling of M
rounds differently between machines), so it is reported, not judged; the
checked route's assertions hold for any matrix of this structure.

Checked (every line must say PASS; the script exits 1 otherwise):
  plan     distrib_la.plan('eigh', backend='distributed').batched: deflated
           and checked, eigenvalues equal to host LAPACK
  guard    the checked eigh around the bare library call, retrying at one
           tile per rank: the right spectrum
  refuse   the checked eigh around a solver with wrong vectors and no retry:
           a NaN result on every rank, and GATE distrib_la_result_check raised
           by the eager refusal
Reported, not checked: the bare library at its default and retry layouts.

    lx run --pool POOL -N 1 -G 4 -n 4 -- bash -c 'source config/perlmutter/gpu_env.sh; \
        python3 -u services/distrib_la/bench/eigh_zero_block_check.py'
"""
import sys

import numpy as np

from runtime import initialize_communicator_stack, run_main_and_finalize
RUNTIME = initialize_communicator_stack()


def joint_metric(seed=1, kc=1152, ac=990, kt=1536, at=1018, top=0.988):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((ac, at)) + 1j * rng.standard_normal((ac, at))
    m *= top / np.linalg.norm(m, 2)
    n = kc + kt
    a = np.zeros((n, n), complex)
    a[:ac, :ac] = np.eye(ac)
    a[kc:kc + at, kc:kc + at] = np.eye(at)
    a[:ac, kc:kc + at] = m
    a[kc:kc + at, :ac] = m.conj().T
    return a


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la import _cusolvermp
    from distrib_la._result_check import accept, checked_eigh, eigh_errors, refuse_if_poisoned
    from distrib_la.plan import _eigh_columns
    from common.collectives import gather_to_host

    mesh = RUNTIME.mesh
    if tuple(mesh.shape.values()) != (2, 2):
        raise SystemExit("eigh_zero_block_check needs a 2x2 mesh (P4)")
    a = joint_metric()
    n = a.shape[-1]
    reference = np.linalg.eigvalsh(a)
    face = NamedSharding(mesh, P("x", "y"))
    A = jax.make_array_from_callback(a.shape, face, lambda i: a[i])
    quality = jax.jit(eigh_errors, out_shardings=NamedSharding(mesh, P()))
    failed = []

    def line(name, w, v, check=None):
        values = np.asarray(gather_to_host(w))
        residual, orthogonality = (float(x) for x in quality(A, w, v))
        verdict = "" if check is None else ("PASS" if check(values, residual, orthogonality) else "FAIL")
        if verdict == "FAIL":
            failed.append(name)
        if jax.process_index() == 0:
            print(f"eigh_zero_block {name:7s} finite={bool(np.isfinite(values).all())} "
                  f"min/max={values[0] / values[-1]:+.3e} residual={residual:.2e} "
                  f"orthogonality={orthogonality:.2e} {verdict}", flush=True)
        return values

    bare = lambda x, **k: _eigh_columns("cusolvermp", *_cusolvermp.distributed_eigh(x, mesh=mesh, **k))
    retry = lambda x: bare(x, block=_cusolvermp.retry_block(n, 2))
    line("bare", *jax.jit(bare)(A))
    line("bare1", *jax.jit(retry)(A))
    plan = distrib_la.plan("eigh", mesh, n=n, backend="distributed", batched_route="auto")
    stack = NamedSharding(mesh, P(None, "x", "y"))
    w, v = jax.jit(plan.batched)(jax.make_array_from_callback((1,) + a.shape, stack, lambda i: a[None][i]))
    line("plan", w[0], v[0], lambda values, r, o: (
        r <= accept(n) and o <= accept(n) and np.max(np.abs(values - reference)) <= 1e-12))

    line("guard", *jax.jit(lambda x: checked_eigh([bare, retry], x, site="bench guard"))(A),
         lambda values, r, o: np.max(np.abs(values - reference)) <= 1e-12 and r <= accept(n) and o <= accept(n))

    def wrong(x):
        values, vectors = retry(x)
        return values, jnp.roll(vectors, 1, axis=-1)
    # Inside a jit the refused result comes back NaN on every rank; an eager
    # caller (Plan, factor/solve) then raises the GATE by name.
    w, v = jax.jit(lambda x: checked_eigh([wrong], x, site="bench refuse"))(A)
    poisoned = bool(np.isnan(np.asarray(gather_to_host(w))).all())
    try:
        refuse_if_poisoned((w, v), "eigh", n, "bench refuse")
        raised = False
    except ValueError as error:
        raised = "GATE distrib_la_result_check" in str(error)
    if not (poisoned and raised):
        failed.append("refuse")
    if jax.process_index() == 0:
        print(f"eigh_zero_block refuse  poisoned={poisoned} raised={raised} "
              f"{'PASS' if poisoned and raised else 'FAIL'}", flush=True)
    if jax.process_index() == 0:
        print("eigh_zero_block " + ("FAILED: " + ", ".join(failed) if failed else "PASS"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    run_main_and_finalize(main)
