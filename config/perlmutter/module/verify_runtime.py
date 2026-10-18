"""P4 identity check of a LORRAX module candidate; accept.sh runs it per rank.

It checks what the module claims, from inside a live four-rank process:
the pinned package versions, one GPU per rank and four in all, the runtime
imported from the module's git-archive snapshot (no checkout), and both FFI
legs mapped from one sealed bundle whose manifest names that snapshot's
revision and hashes the mapped files, each leg's recorded dependency closure
enforced (one mapped copy of every recorded library, the recorded one).  The
CUDA libraries are the venv's uv.lock wheels, a clone's set: no CUDA library of
the toolkit may be mapped, and libcudart.so (libfabric's name) is the wheel's.
"""

# ruff: noqa: E402 -- distributed initialization must precede JAX imports.
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path

import runtime

runtime.initialize_communicator_stack()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi.common import ffi_loader


def say(message: str) -> None:
    if jax.process_index() == 0:
        print(f"[verify_runtime] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    expect = json.loads(os.environ["LORRAX_MODULE_EXPECT"])
    got = {name: md.version(name) for name in expect}
    assert got == expect, f"version drift: got {got}, expected {expect}"
    assert jax.default_backend() == "gpu", jax.default_backend()
    assert (jax.process_count(), jax.device_count(), jax.local_device_count()) \
        == (4, 4, 1), (jax.process_count(), jax.devices(), jax.local_devices())
    say(f"PASS versions {got}; 4 ranks x 1 GPU")

    root = Path(os.environ["LORRAX_ROOT"]).resolve()
    assert Path(runtime.__file__).resolve() == root / "src/runtime/__init__.py", \
        runtime.__file__
    assert not (root / ".git").exists(), f"{root} is a checkout"
    revision = (root / "SOURCE_COMMIT").read_text().strip()
    say(f"PASS source snapshot {root} at {revision[:12]}")

    so_cuda = Path(os.environ["LORRAX_FFI_SO"]).resolve()
    so_host = Path(os.environ["LORRAX_FFI_HOST_SO"]).resolve()
    bundle = so_cuda.parent.parent
    assert so_host.parent.parent == bundle, (so_cuda, so_host)
    manifest = json.loads((bundle / "lorrax_ffi_bundle.json").read_text())
    assert manifest["source"] == {"dirty": False, "revision": revision}, \
        manifest["source"]
    for leg, so in (("CUDA", so_cuda), ("cpu", so_host)):
        assert sha256(so) == manifest["libraries"][leg]["sha256"], leg
    ffi_loader.get_lib("CUDA")
    ffi_loader.get_lib("cpu")
    maps = Path("/proc/self/maps").read_text()
    for so in (so_cuda, so_host):
        assert str(so) in maps, f"{so} is not mapped"
    sdk = Path(os.environ["CUDA_HOME"]).resolve().parents[1]
    mapped = {os.path.realpath(line.split()[-1]) for line in maps.splitlines()
              if len(line.split()) > 5 and ".so" in line.split()[-1]}
    toolkit = sorted(p for p in mapped if p.startswith(str(sdk) + "/"))
    assert not toolkit, f"toolkit CUDA libraries mapped: {toolkit}"
    site = Path(jax.__file__).resolve().parents[1]
    wheels = site / "nvidia" / "cu13" / "lib"
    for name in ("libcudart.so.13", "libcublas.so.13", "libcufft.so.12"):
        want = os.path.realpath(wheels / name)
        assert want in mapped, f"{name}: the wheel's {want} is not mapped"
    launch = [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":")
              if d and os.path.isfile(os.path.join(d, "libcudart.so"))]
    assert launch and os.path.realpath(os.path.join(launch[0], "libcudart.so")) \
        == os.path.realpath(wheels / "libcudart.so.13"), launch[:1]
    # NCCL: exactly the venv's nvidia-nccl-cu13 wheel (stack.sh NCCL_VERSION).
    nccl = sorted({os.path.realpath(line.split()[-1]) for line in maps.splitlines()
                   if "/libnccl.so" in line})
    want_nccl = os.path.realpath(site / "nvidia" / "nccl" / "lib" / "libnccl.so.2")
    assert nccl == [want_nccl], f"NCCL mapped {nccl}, want only {want_nccl}"
    assert md.version("nvidia-nccl-cu13") == expect.get("nvidia-nccl-cu13"), \
        md.version("nvidia-nccl-cu13")
    assert "libcudart.so.12" not in maps
    say(f"PASS sealed bundle {manifest['bundle_id'][:12]}: both legs mapped, "
        f"hashes match, CUDA = the uv.lock wheels, no toolkit library, "
        f"NCCL = the wheel {want_nccl}")
    # The bundle's own dependency closure was enforced when each leg opened:
    # every recorded library mapped exactly once, as the recorded file.
    from lxkit import native_provider
    assert set(manifest.get("closures", {})) == {"CUDA", "cpu"}, \
        sorted(manifest.get("closures", {}))
    ids = {leg: native_provider.native_closure_identity(leg) for leg in ("CUDA", "cpu")}
    for leg, so in (("CUDA", so_cuda), ("cpu", so_host)):
        assert ids[leg] == native_provider.expected_closure_identity(so, leg), leg
    say("PASS dependency closure enforced: "
        + ", ".join(f"{leg}={ident[:12]}" for leg, ident in sorted(ids.items())))

    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    x = jax.make_array_from_callback(
        (8, 8), NamedSharding(mesh, P("x", "y")),
        lambda index: np.ones((8, 8))[index])
    total = float(jax.jit(jnp.sum, out_shardings=NamedSharding(mesh, P()))(x))
    assert total == 64.0, total
    say("PASS 2x2 mesh reduction")
    say("MODULE IDENTITY: ALL PASSED")


if __name__ == "__main__":
    main()
