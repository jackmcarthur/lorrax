"""One-GPU check of the mathdx disk cubin cache (concurrency audit items 6 and 7).

Each case is a FRESH child process (the in-process kernel cache would hide the
disk), calling the k-leading transform on a 3x2x4 grid with an explicit
``mathdx_root`` and ``cubin_dir``, and must reproduce ``np.fft``:

1. cold: ``NVRTC built`` and one image stored;  2. warm: ``disk-cache hit``;
3. a cached image with valid framing and an ELF header that the driver
   refuses is deleted and rebuilt (``NVRTC rebuilt``), and the file is valid
   again;
4. a mathdx root whose CUTLASS ``version.h`` differs keys a NEW image;
5. a root with no CUTLASS ``version.h`` turns the disk cache off (no image).

Run: ``lx run -N 1 -G 1 -n 1 python3 -u tests/multi_device/kconv_cubin_cache_check.py``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
TAG = "[kconv-cubin-cache]"

CHILD = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[3])
from runtime import initialize_communicator_stack
initialize_communicator_stack(platform="gpu")
import jax, jax.numpy as jnp
from jax.sharding import Mesh
from ffi import fft as F
F.require_kconv(Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y")), announce=False)
x = np.random.default_rng(0).standard_normal((24, 5)) + 1j * np.random.default_rng(1).standard_normal((24, 5))
attrs = dict(nkx=np.int64(3), nky=np.int64(2), nkz=np.int64(4), scale=np.float64(1.0),
             forward=np.int64(1), mathdx_root=sys.argv[1], cubin_dir=sys.argv[2])
y = jax.ffi.ffi_call(F.KFFT_KLEAD_TARGET, jax.ShapeDtypeStruct(x.shape, jnp.complex128))(jnp.asarray(x), **attrs)
ref = np.fft.fftn(x.reshape(3, 2, 4, 5), axes=(0, 1, 2)).reshape(24, 5)
print("MAXERR", float(np.max(np.abs(np.asarray(y) - ref))), flush=True)
'''


def _fnv1a(data: bytes, h: int = 1469598103934665603) -> int:
    for c in data:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _child(root: str, cdir: str) -> str:
    env = dict(os.environ, LORRAX_DEBUG_PRINT="1")
    out = subprocess.run([sys.executable, "-c", CHILD, root, cdir, str(_ROOT / "src")],
                         capture_output=True, text=True, env=env, timeout=900)
    text = out.stdout + out.stderr
    m = re.search(r"MAXERR (\S+)", text)
    if out.returncode != 0 or not m or float(m.group(1)) > 1e-12:
        raise SystemExit(f"{TAG} FAIL child rc={out.returncode}:\n{text[-3000:]}")
    return text


def _images(cdir: str) -> list[Path]:
    return sorted(Path(cdir).glob("kconv_m3_3x2x4_*.cubin"))


def _farm(real: Path, dst: Path, version_h: str | None) -> Path:
    """A mathdx root of symlinks whose cutlass/version.h is replaced (or absent)."""
    (dst / "external" / "cutlass" / "include" / "cutlass").mkdir(parents=True)
    os.symlink(real / "include", dst / "include")
    inc = real / "external" / "cutlass" / "include"
    for e in os.scandir(inc):
        if e.name != "cutlass":
            os.symlink(e.path, dst / "external" / "cutlass" / "include" / e.name)
    for e in os.scandir(inc / "cutlass"):
        if e.name != "version.h":
            os.symlink(e.path, dst / "external" / "cutlass" / "include" / "cutlass" / e.name)
    if version_h is not None:
        (dst / "external" / "cutlass" / "include" / "cutlass" / "version.h").write_text(version_h)
    return dst


def main() -> int:
    sys.path.insert(0, str(_ROOT / "src"))
    from ffi.fft import mathdx_root
    real = Path(mathdx_root())
    work = Path(tempfile.mkdtemp(prefix="kconv_cubin_cache_"))
    try:
        cdir = str(work / "cubins")
        t = _child(str(real), cdir)
        assert "NVRTC built" in t and len(_images(cdir)) == 1, t[-2000:]
        print(f"{TAG} cold: built, 1 image", flush=True)
        t = _child(str(real), cdir)
        assert "disk-cache hit" in t, t[-2000:]
        print(f"{TAG} warm: disk-cache hit", flush=True)

        img = _images(cdir)[0]
        blob = img.read_bytes()
        head = len(b"LRXKCONV1\n") + 33
        bad = b"\x7fELF" + b"\x00" * 60 + b"not a cubin" * 16
        img.write_bytes(blob[:len(b"LRXKCONV1\n") + 16] + f"{_fnv1a(bad):016x}".encode() + b"\n" + bad)
        t = _child(str(real), cdir)
        assert "NVRTC rebuilt (cached image refused)" in t, t[-2000:]
        assert img.read_bytes()[head:head + 4] == b"\x7fELF" and len(img.read_bytes()) == len(blob)
        print(f"{TAG} refused image: deleted, rebuilt, re-stored", flush=True)

        text = (real / "external" / "cutlass" / "include" / "cutlass" / "version.h").read_text()
        farm = _farm(real, work / "root_v", text + "\n// kconv_cubin_cache_check: a different CUTLASS\n")
        t = _child(str(farm), cdir)
        assert "NVRTC built" in t and len(_images(cdir)) == 2, t[-2000:]
        print(f"{TAG} changed cutlass version.h: new key, 2 images", flush=True)

        farm0 = _farm(real, work / "root_0", None)
        t = _child(str(farm0), cdir)
        assert "disk cubin cache OFF" in t and len(_images(cdir)) == 2, t[-2000:]
        print(f"{TAG} missing version header: disk cache off", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{TAG} PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
