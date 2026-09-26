# JAX support contract

LORRAX supports exactly the JAX/JAXLIB 0.9 series:

```text
jax     >= 0.9.0, < 0.10.0
jaxlib  >= 0.9.0, < 0.10.0
```

The lock and the Perlmutter `lorrax_A` module resolve both packages to
0.9.1. Patch upgrades inside the series are allowed; a different minor
generation is a startup refusal.

Three independent surfaces enforce the same contract:

1. `pyproject.toml` constrains base, CUDA-12, CUDA-13, and development installs;
2. `tools/require_jax09.py` checks installed package metadata before a launcher's
   first JAX import;
3. `runtime.jax_support.enforce()`, step 5b of
   `runtime.initialize_communicator_stack` (after the first `jax.devices()`,
   before the first `jit`), checks both live packages and the arity of the
   `jax._src` privates that `common/jax_compile_cache.py` patches. A version
   string alone is not trusted: a date-stamped container build can report a
   blessed version over a different `jax._src`, and the patched functions
   would die on the first compile.

`tests/test_jax_support.py` proves that the package and runtime windows cannot
drift, that both JAX and JAXLIB are checked, and that 0.7/0.8/0.10 refuse.
`tests/test_require_jax09.py` provides positive and negative preflight arms.
There is no unsupported-version escape hatch.

On Perlmutter, `lx` selects the `lorrax_A` module by default. Launch from a
data directory with the checkout explicit:

```bash
export LORRAX_CHECKOUT=/absolute/path/to/checkout
lx run -- env PYTHONPATH="$LORRAX_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$LORRAX_CHECKOUT/tools/require_jax09.py"
```

Put the checkout's top-level `src` on the compute payload path, not only in the
outer shell. The runtime derives every first-party service root from package
metadata and refuses a mixed closure before JAX; do not duplicate a manual
`services/*/src` list. JAX 0.9 does not by itself certify a native FFI artifact:
the CUDA major, registered handler set, dependency closure, and source
provenance remain separate launch facts.
