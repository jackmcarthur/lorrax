"""Hostile tests for the one native-provider selection policy."""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lxkit import native_provider as native
from lxkit.probe import LibraryUnusable

ABI = 3
SPECS = {
    "CUDA": {
        "so_name": "liblorrax_ffi.so",
        "env": "LORRAX_FFI_SO",
        "build_hint": "build cuda",
    },
    "cpu": {
        "so_name": "liblorrax_ffi_host.so",
        "env": "LORRAX_FFI_HOST_SO",
        "build_hint": "build host",
    },
}
ABI_SYMBOLS = {
    "CUDA": "lorrax_ffi_cuda_abi_version",
    "cpu": "lorrax_ffi_host_abi_version",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _hash(path),
    }


def _write_manifest(root: Path, *, abi: int = ABI,
                    private: tuple[Path, ...] = ()) -> Path:
    cuda = root / "lib" / "liblorrax_ffi.so"
    host = root / "lib" / "liblorrax_ffi_host.so"
    doc: dict[str, object] = {
        "schema": 1,
        "source": {"revision": "1" * 40, "dirty": False},
        "ffi_abi": abi,
        "libraries": {
            "CUDA": _entry(cuda, root),
            "cpu": _entry(host, root),
        },
        "private_libraries": [
            {**_entry(path, root), "soname": path.name} for path in private
        ],
    }
    doc["bundle_id"] = native._canonical_manifest_id(doc)
    manifest = root / native.BUNDLE_MANIFEST
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    return manifest


def _bundle(tmp_path: Path, name: str = "bundle") -> Path:
    root = tmp_path / name
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "liblorrax_ffi.so").write_bytes(b"cuda-provider")
    (root / "lib" / "liblorrax_ffi_host.so").write_bytes(b"host-provider")
    _write_manifest(root)
    return root


def _env(root: Path) -> dict[str, str]:
    return {
        "LORRAX_FFI_SO": str(root / "lib" / "liblorrax_ffi.so"),
        "LORRAX_FFI_HOST_SO": str(root / "lib" / "liblorrax_ffi_host.so"),
    }


def _locate(root: Path, platform: str = "CUDA", *, env=None) -> Path:
    return native.locate_library(
        platform, specs=SPECS,
        candidates={name: [root / "lib" / spec["so_name"]]
                    for name, spec in SPECS.items()},
        expected_abi=ABI, environ=_env(root) if env is None else env)


def test_one_manifest_binds_both_selected_legs(tmp_path):
    root = _bundle(tmp_path)
    assert _locate(root, "CUDA").samefile(root / "lib" / "liblorrax_ffi.so")
    assert _locate(root, "cpu").samefile(root / "lib" /
                                              "liblorrax_ffi_host.so")


def test_partial_sealed_override_refuses(tmp_path):
    root = _bundle(tmp_path)
    with pytest.raises(LibraryUnusable, match="partial sealed-bundle override"):
        _locate(root, env={"LORRAX_FFI_SO":
                           str(root / "lib" / "liblorrax_ffi.so")})


def test_missing_bound_leg_refuses_before_dlopen(tmp_path):
    root = _bundle(tmp_path)
    (root / "lib" / "liblorrax_ffi_host.so").unlink()
    with pytest.raises(LibraryUnusable, match="escapes or is missing"):
        _locate(root, env={})


def test_stale_bytes_refuse_before_dlopen(tmp_path):
    root = _bundle(tmp_path)
    host = root / "lib" / "liblorrax_ffi_host.so"
    host.write_bytes(host.read_bytes() + b"-rebuilt-after-seal")
    with pytest.raises(LibraryUnusable, match="byte-size mismatch"):
        _locate(root)


def test_mixed_provider_pair_refuses(tmp_path):
    first = _bundle(tmp_path, "first")
    second = _bundle(tmp_path, "second")
    env = {
        "LORRAX_FFI_SO": str(first / "lib" / "liblorrax_ffi.so"),
        "LORRAX_FFI_HOST_SO": str(second / "lib" /
                                   "liblorrax_ffi_host.so"),
    }
    with pytest.raises(LibraryUnusable, match="different bundle manifests"):
        _locate(first, env=env)


def test_wrong_leg_origin_inside_one_bundle_refuses(tmp_path):
    root = _bundle(tmp_path)
    env = {
        "LORRAX_FFI_SO": str(root / "lib" / "liblorrax_ffi_host.so"),
        "LORRAX_FFI_HOST_SO": str(root / "lib" / "liblorrax_ffi.so"),
    }
    with pytest.raises(LibraryUnusable, match="mixed native provider"):
        _locate(root, env=env)


def test_manifest_abi_refuses_before_dlopen(tmp_path, monkeypatch):
    root = _bundle(tmp_path)
    _write_manifest(root, abi=ABI - 1)
    called = False

    def _forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("dlopen reached")

    monkeypatch.setattr(ctypes, "CDLL", _forbidden)
    with pytest.raises(LibraryUnusable, match="ABI mismatch before dlopen"):
        _locate(root)
    assert called is False


def test_runtime_owned_library_cannot_enter_private_closure(tmp_path):
    root = _bundle(tmp_path)
    mpi = root / "lib" / "libmpi.so.12"
    mpi.write_bytes(b"do-not-bundle-site-mpi")
    _write_manifest(root, private=(mpi,))
    with pytest.raises(LibraryUnusable, match="runtime-owned"):
        _locate(root)


def test_symlink_escape_cannot_satisfy_a_bound_file(tmp_path):
    root = _bundle(tmp_path)
    host = root / "lib" / "liblorrax_ffi_host.so"
    outside = tmp_path / "outside.so"
    outside.write_bytes(host.read_bytes())
    host.unlink()
    host.symlink_to(outside)
    with pytest.raises(LibraryUnusable, match="escapes or is missing"):
        _locate(root)


def _compile_pair(tmp_path: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("C compiler required for the live dladdr attestation twin")
    root = tmp_path / "live"
    (root / "lib").mkdir(parents=True)
    for platform, symbol in ABI_SYMBOLS.items():
        name = SPECS[platform]["so_name"]
        source = root / f"{platform}.c"
        source.write_text(f"int {symbol}(void) {{ return {ABI}; }}\n",
                          encoding="utf-8")
        result = subprocess.run(
            [cc, "-shared", "-fPIC", "-o", str(root / "lib" / name),
             str(source)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=120)
        assert result.returncode == 0, result.stdout
    _write_manifest(root)
    return root


def test_live_abi_symbol_origin_is_the_manifest_file(tmp_path):
    root = _compile_pair(tmp_path)
    selected = _locate(root, "cpu")
    lib, actual = native.open_and_attest(
        selected, platform="cpu", expected_abi=ABI,
        abi_symbols=ABI_SYMBOLS, build_hint="build host")
    assert lib is not None
    assert actual.samefile(selected)


def test_live_wrong_origin_refuses(tmp_path, monkeypatch):
    root = _compile_pair(tmp_path)
    selected = _locate(root, "cpu")
    wrong = root / "lib" / "liblorrax_ffi.so"
    monkeypatch.setattr(native, "_actual_symbol_origin", lambda lib, sym: wrong)
    with pytest.raises(LibraryUnusable, match="origin mismatch"):
        native.open_and_attest(
            selected, platform="cpu", expected_abi=ABI,
            abi_symbols=ABI_SYMBOLS, build_hint="build host")


def test_private_closure_localizes_without_ld_library_path(tmp_path,
                                                           monkeypatch):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("C compiler required for the private-closure twin")
    root = tmp_path / "private-live"
    libdir = root / "lib"
    libdir.mkdir(parents=True)
    private_src = root / "private.c"
    private_src.write_text("int sealed_private_value(void) { return 7; }\n",
                           encoding="utf-8")
    private = libdir / "libblaspp.so.2"
    assert subprocess.run(
        [cc, "-shared", "-fPIC", "-Wl,-soname,libblaspp.so.2",
         "-o", str(private), str(private_src)], timeout=120).returncode == 0
    cuda_src = root / "cuda.c"
    cuda_src.write_text(
        f"int {ABI_SYMBOLS['CUDA']}(void) {{ return {ABI}; }}\n",
        encoding="utf-8")
    assert subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(libdir /
                                             "liblorrax_ffi.so"),
         str(cuda_src)], timeout=120).returncode == 0
    host_src = root / "host.c"
    host_src.write_text(
        "extern int sealed_private_value(void);\n"
        f"int {ABI_SYMBOLS['cpu']}(void) "
        "{ return sealed_private_value() == 7 ? 3 : -1; }\n",
        encoding="utf-8")
    assert subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(libdir /
                                             "liblorrax_ffi_host.so"),
         str(host_src), "-L", str(libdir), "-Wl,--no-as-needed",
         "-l:libblaspp.so.2"], timeout=120).returncode == 0
    _write_manifest(root, private=(private,))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    selected = _locate(root, "cpu")
    _, actual = native.open_and_attest(
        selected, platform="cpu", expected_abi=ABI,
        abi_symbols=ABI_SYMBOLS, build_hint="build host")
    assert actual.samefile(selected)
