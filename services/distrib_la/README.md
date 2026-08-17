# distrib_la

`distrib_la` is an independently installable Python package for dense linear
algebra on a JAX `Mesh` with axes `('x', 'y')`. Its public operations are
Hermitian eigendecomposition, Cholesky factorization, LU solve, distributed
GEMM, and opaque factor/solve. Native JAX kernels are always available;
optional ScaLAPACK/PBLAS, SLATE, and cuSOLVERMp/cuBLASMp handlers are
discovered at runtime through a compatible FFI provider shared library.

The package has three runtime dependencies: `jax`, `numpy`, and the sibling
foundation package `lxkit`. It imports no LORRAX `src/` module or `common`
helper. From a LORRAX source checkout, install the two distributions with:

```bash
cd services/distrib_la
python -m pip install -e ../lxkit -e '.[test]'
python -c "import distrib_la; print(distrib_la.BATCHED_ROUTE_CHOICES)"
python -m pytest tests/test_distrib_la_shape_algebra.py \
  tests/test_distrib_la_emulated_mesh.py \
  tests/test_distrib_la_import_isolation.py \
  tests/test_distrib_la_batch_reshard.py \
  tests/test_distrib_la_matmul.py
```

That provider-free subset exercises the package boundary and native routes.
The full `python -m pytest` suite also runs FFI/ELF and machine-profile gates;
on a machine such as Perlmutter whose profile promises provider libraries,
set the documented `.so` pins and library paths first. Missing promised
capabilities are failures there, not skips.

If `h5py` happens to be installed, the FFI loader imports it in a caught,
best-effort block before `dlopen` so h5py's HDF5 symbols win the process-wide
load-order race. `h5py` is not required by `distrib_la`; its absence is
accepted and it is intentionally not a declared dependency.

An installed consumer uses only the top-level API:

```python
import jax
import numpy as np
from jax.sharding import Mesh
import distrib_la as dla

devices = np.asarray(jax.devices())
mesh = Mesh(devices.reshape(1, devices.size), ('x', 'y'))

eigh = dla.plan('eigh', mesh, backend='off', n=a_stack.shape[-1],
                batched_route='batch_reshard')
w, z = eigh.batched(a_stack)
print(eigh.describe())

# Rank-3 inputs use P(None, 'x', 'y'); rank-2 inputs use P('x', 'y').
# This provider-free spelling exchanges faces x then y, computes locally,
# and returns through the literal y-then-x inverse.
d_stack = dla.matmul(a_gemm, b_gemm, mesh=mesh, backend='off',
                     batched_route='batch_reshard')
```

For `Plan`, `batched_route='auto'` preserves backend-native batching or the
distributed scan. `batched_route='batch_reshard'` instead exchanges matrix-face shards
into complete matrices distributed over the batch axis, runs the local JAX
kernel, and applies the exact inverse exchanges to matrix outputs. This route
covers `eigh`, `cholesky`, and `solve_lu`, including leading batches not
divisible by the device count. Use it only when one complete matrix, its
output, and the native solver workspace fit on one device. It is not a
replacement for a distributed backend in the single-matrix capacity regime.

Top-level `matmul` has deliberately different default routing from `plan`.
`matmul(..., backend='auto')` selects cuBLASMp on CUDA, PBLAS
`pdgemm`/`pzgemm` on CPU, or `slate::multiply` on ROCm; `cusolvermp` is an
accepted alias for its cuBLASMp sibling. Rank-2 inputs and outputs use
`P('x','y')`, while rank-3 stacks use `P(None,'x','y')`. The explicit staged
route pads only a ragged leading batch with zero GEMM rows. It refuses matrix
or output extents that do not tile `Px` by `Py`, and each device must have
room for complete local A, B, and D matrices, C when `beta != 0`, live input
faces/exchange buffers, and GEMM workspace. No zero C is allocated or
exchanged when `beta == 0`. Provider routes require an exact y-minor 2-D
`('x','y')` process grid; cuBLASMp and SLATE require it to be square, while
PBLAS also supports rectangular grids.

No shared library is needed to import the package, inspect capabilities, or
use native routes. To grant an FFI capability, point `LORRAX_FFI_SO` (CUDA)
or `LORRAX_FFI_HOST_SO` (CPU) at a provider library that exports the handler
symbols and ABI expected by this package. The current provider is built by
the LORRAX C++ tree; an explicit missing or incompatible pin is a refusal,
never a fallback.

The canonical API, sharding contracts, route schedule, refusals, warm-up
behavior, tests, and backend limitations are documented in
[`../../docs/services/distrib_la.md`](../../docs/services/distrib_la.md).
