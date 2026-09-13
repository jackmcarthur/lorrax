# JAX XC compatibility

`psp.xc` consumes sail-sg/jax_xc's generated libxc kernels. The published
`jax-xc==0.0.11` wheel eagerly imports experimental/autofd, which uses JAX
internals removed before the supported JAX 0.9 series. Until upstream fixes
that import boundary, validation requires this explicit, bounded installation
patch. It changes package initialization only; generated functional code is
unchanged. Accessing `jax_xc.experimental` still imports its own dependencies
and reports any incompatibility normally.

From the source checkout, install the recorded auxiliary requirements without
resolving or replacing the runtime's JAX/NumPy. Use the supported runtime's
Python interpreter (3.12 in the measured Perlmutter lane):

```sh
uv pip install --python /path/to/runtime/bin/python --target /explicit/site/target \
  --no-deps -r config/xc/requirements.txt
python config/xc/patch_jax_xc.py /explicit/site/target
```

The script
refuses any initializer except the exact original 0.0.11 file. Add that target
to the application import path when launching from the source checkout.
This is a locally patched dependency, not evidence that unmodified upstream
supports JAX 0.9. The patch is equally applicable to upstream's generated
`gen_repo/__init__.py`, except its current version string is 0.0.12.

Upstream source inspected: https://github.com/sail-sg/jax_xc at
60c3eadeca710fede723bd449512fd12c9e9181b. Published wheel version: 0.0.11.
The scalar-grid adapter sums unpolarized PBE exchange and correlation,
vectorizes scalar kernels, and converts Hartree/electron to Ry/electron.
It preserves the existing scalar XC model; spin-dependent magnetic XC is
outside this adapter's contract.

Focused numerical test: `tests/test_xc_pbe.py` (requires patched dependency).
Independent QE total-potential equivalence must be checked separately from
point-functional and autodiff tests.
