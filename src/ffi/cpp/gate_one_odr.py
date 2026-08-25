"""GATE 10 — a CUDA-capable process really can do host phdf5 work.

The fifth of the `gate_one_*` family (mpi / hdf5 / fftw are shell; this one
is Python because the property is only observable from inside a process that
has JAX, both `.so` files and a real HDF5 file).  Same doctrine as its
siblings: a STATIC check on the artifacts cannot see this.  `nm -D` says the
two libraries share no name of ours; only a live process says the phdf5 read
path still works when both of them are open.

WHAT IT GUARDS -- tests/KNOWN_FAILURES.md L1.  Both platform libraries are
dlopened RTLD_GLOBAL into one process and ld.so answers a name from the FIRST
object that defined it, for the whole process, including for the second
library's internal calls.  Until 2026-08-08 the two shared 25 of LORRAX's own
names, among them `lorrax_ffi::phdf5::open_ctx` -- whose `PhdfCtx` return
type has a DIFFERENT LAYOUT in the two builds.

HOW TO RUN IT (Perlmutter; needs a GPU node and BOTH pins):

    LX_BASE_MODULE=lorrax_A LORRAX_CHECKOUT=$PWD PYTHONPATH=$PWD/src \
    LORRAX_FFI_SO=<device .so> LORRAX_FFI_HOST_SO=<host .so> \
    lx run -G 1 -n 1 env JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 \
        LORRAX_FFI_SO=<device .so> LORRAX_FFI_HOST_SO=<host .so> \
        PROBE_DIR=<scratch dir> \
        bash -c 'tools/require_jax09.py && \
                 python3 -u src/ffi/cpp/gate_one_odr.py'

`JAX_PLATFORMS=cuda,cpu` is the whole setup: the process IS CUDA-capable
(`jax.default_backend() == 'gpu'`), which is what `_process_can_use_cuda()`
answers True to, so `get_lib('cpu')` pre-opens the CUDA library FIRST and
BOTH .so files end up in the process; and the SlabIO mesh is built from
`jax.devices('cpu')`, so `ffi.io._platform_for_mesh` routes the phdf5
lifecycle to the HOST library and the read `ffi_call` lowers on the cpu
platform -> `PhdfReadHostFfi`.  A host phdf5 read, executed by host-library
code, on a context minted while the CUDA library is also open.

MEASURED 2026-08-08, HDF5-200 pair, one GPU node:

    rebuilt pair (fix/ffi-odr-2026-08-08)   every check PASS, rc=0
    deployed Aug-7 pair                     rc=134, and the process dies at

        [SlabIO.close] draining 1 pending writes for gate_one_odr.h5 ...
        Fatal glibc error: tpp.c:83 (__pthread_tpp_change_priority):
          assertion failed: new_prio == -1 || ...

    -- which is `close_ctx` joining a `std::thread` and destroying a
    `std::mutex` at the offsets of the OTHER build's PhdfCtx.  It is also,
    line for line, the death KNOWN_FAILURES B2 recorded for the xdist arm.

The failing arm is what makes this a gate rather than a demonstration: run
it against a pre-2026-08-08 pair and it must abort.  Prints one `PROBE:` line
per check and exits nonzero on the first failure.
"""

import os
import sys

import numpy as np

FAILS = []


def check(name, ok, detail=""):
    print(f"PROBE: {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


import jax                                                    # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P             # noqa: E402
import h5py                                                   # noqa: E402

backend = jax.default_backend()
check("the process is CUDA-capable", backend in ("gpu", "cuda"),
      f"jax.default_backend()={backend!r}")

from ffi.common import ffi_loader                             # noqa: E402
from file_io.slab_io import SlabIO                            # noqa: E402

# Touching the cpu library is what triggers the pre-open of the CUDA one.
ffi_loader.get_lib("cpu")
order = ffi_loader.loaded_platforms_in_order()
check("both platform libraries are open, CUDA first", order == ["CUDA", "cpu"],
      f"loaded_platforms_in_order()={order}")
for plat in order:
    print(f"PROBE: info  {plat} <- {ffi_loader.loaded_lib_path(plat)}",
          flush=True)

# The host leg's C ABI carries the `_host` suffix; the loader binds it under
# the plain name.  Both must be true or the rename silently did nothing.
host = ffi_loader.get_lib("cpu")
cuda = ffi_loader.get_lib("CUDA")
check("host lib exports the leg-suffixed C ABI",
      hasattr(host, "lrx_phdf5_open_host"))
check("CUDA lib does NOT export the host-suffixed C ABI",
      not hasattr(cuda, "lrx_phdf5_open_host"))

cpu_devs = jax.devices("cpu")
check("cpu devices are available in this process", len(cpu_devs) >= 1,
      f"{len(cpu_devs)} cpu device(s)")
mesh = Mesh(np.asarray(cpu_devs[:1]).reshape(1, 1), axis_names=("x", "y"))

path = os.path.join(os.environ.get("PROBE_DIR", "."), "gate_one_odr.h5")
if os.path.exists(path):
    os.remove(path)

truth = (np.arange(6 * 8, dtype=np.float64).reshape(6, 8) * 0.25 + 0.19)

# WRITE through the host library (PhdfWriteHostFfi), CPU mesh.
with SlabIO(path, mode="w", mesh=mesh) as io:
    io.create_dataset("payload", shape=truth.shape, dtype=truth.dtype)
    io.write_slab("payload", jax.numpy.asarray(truth), offset=(0, 0))
check("the file was created", os.path.exists(path))

with h5py.File(path, "r") as f:
    on_disk = np.asarray(f["payload"])
check("host WRITE landed the right bytes",
      np.allclose(on_disk, truth),
      f"max|d|={np.max(np.abs(on_disk - truth)):.3e}")

# READ back through the host library (PhdfReadHostFfi), CPU mesh.  This is
# the handler that reads PhdfCtx's fields; a cross-layout ctx shows up as
# garbage `offset_base` / `ctx_handle is null` / an abort.
with SlabIO(path, mode="r", mesh=mesh) as io:
    got = np.asarray(io.read_slab("payload", as_numpy=True))
check("host READ returned the right bytes",
      got.shape == truth.shape and np.allclose(got, truth),
      f"shape={got.shape} max|d|={np.max(np.abs(got - truth)):.3e}"
      if got.shape == truth.shape else f"shape={got.shape}")

# A partial slab, so the offset/valid_shape path (the one whose garbage
# `offset_base=[0,0,0,4596944070643295330]` is L1's signature) is exercised.
with SlabIO(path, mode="r", mesh=mesh) as io:
    sub = np.asarray(io.read_slab("payload", shape=(3, 4), offset=(2, 3),
                                  as_numpy=True))
check("host READ of an OFFSET slab returned the right bytes",
      sub.shape == (3, 4) and np.allclose(sub, truth[2:5, 3:7]),
      f"shape={sub.shape}")

print(f"PROBE: {'ALL PASS' if not FAILS else 'FAILURES: ' + ','.join(FAILS)}",
      flush=True)
sys.exit(1 if FAILS else 0)
