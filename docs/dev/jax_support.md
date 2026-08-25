# JAX support contract

LORRAX supports exactly the JAX/JAXLIB 0.9 series:

```text
jax     >= 0.9.0, < 0.10.0
jaxlib  >= 0.9.0, < 0.10.0
```

The current lock and Perlmutter `lorrax_A` module resolve both packages to
0.9.1.  Patch upgrades inside the series remain possible; a different minor
generation is a startup refusal.

Three independent surfaces enforce the same contract:

1. `pyproject.toml` constrains base, CUDA-12, CUDA-13, and development installs;
2. `tools/require_jax09.py` checks installed package metadata before a launcher's
   first JAX import;
3. `runtime.jax_support.enforce()` checks both live packages and the private
   compile-cache API before the first physics `jit`.

`tests/test_jax_support.py` proves that the package and runtime windows cannot
drift, that both JAX and JAXLIB are checked, and that 0.7/0.8/0.10 refuse.
`tests/test_require_jax09.py` provides positive and negative preflight arms.
There is no unsupported-version escape hatch.

On Perlmutter, launch from a data directory with both choices explicit:

```bash
export LX_BASE_MODULE=lorrax_A
export LORRAX_CHECKOUT=/absolute/path/to/checkout
lx run "$LORRAX_CHECKOUT/tools/require_jax09.py"
```

A source overlay must prepend the checkout's `src` and every required
`services/*/src` directory inside the compute payload.  JAX 0.9 does not by
itself certify a native FFI artifact: the CUDA major, registered handler set,
dependency closure, and source provenance remain separate launch facts.

Historical result documents may name JAX 0.5 or 0.7 because those versions
were actually used for those measurements.  They are evidence records, not
current installation or launch guidance.
