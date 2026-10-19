# distrib_la

`distrib_la` is LORRAX's dense linear algebra over a JAX `Mesh` with axes
`('x', 'y')`: Hermitian eigensolves, Cholesky and LU, GEMM, the polar factor
and the active-subspace algebra of iterative eigensolvers. It is an
independently installable package whose runtime dependencies are `lxkit`, JAX
and NumPy; it imports nothing from LORRAX's `src/`. ScaLAPACK, SLATE,
cuSOLVERMp and cuBLASMp are not Python dependencies: `distrib_la.loader` opens
them at run time through the LORRAX native FFI pair, and without that pair the
package still imports, reports its capabilities and runs its pure-JAX routes.

```bash
cd services/distrib_la
python -m pip install -e ../lxkit -e .
python -c "import distrib_la; print(distrib_la.BATCHED_ROUTE_CHOICES)"
```

Import top-level names only. The documentation lives in the LORRAX docs:

- [The API](../../docs/services/distrib_la/api.md): layouts, plans, factor and
  solve, GEMM, the polar factor, workspace queries.
- [Backends](../../docs/services/distrib_la/backends.md): which library serves
  each operation on each platform, the `linalg` deck dial, the guard ladder,
  and how to add a backend.
- [Active subspace](../../docs/services/distrib_la/subspace.md): the
  fixed-capacity plans of Davidson and Lanczos.

`bench/` holds backend checks and benchmarks that run on a real mesh; they are
not a pytest suite.
