"""The deployable native unit is an immutable, two-leg bundle."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "src" / "ffi" / "cpp" / "stage" / "seal_bundle.py"
_SPEC = importlib.util.spec_from_file_location("lorrax_seal_bundle", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SEALER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SEALER)


def test_both_build_legs_search_their_bundle_directory_first():
    """The install/build RPATH twin of exact private-provider preloading."""
    cmake = (_ROOT / "src" / "ffi" / "cpp" / "CMakeLists.txt").read_text(
        encoding="utf-8")
    assert 'INSTALL_RPATH "$ORIGIN;${CUSOLVERMP_LIBDIR}' in cmake
    assert 'INSTALL_RPATH "$ORIGIN;${_host_rpaths}"' in cmake


def _source_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    header = root / "src" / "ffi" / "cpp" / "common"
    header.mkdir(parents=True)
    (header / "lorrax_ffi_abi.h").write_text(
        "#define LORRAX_FFI_ABI_VERSION 3\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=30)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, timeout=30)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=LORRAX test",
         "-c", "user.email=lorrax-test@example.invalid", "commit", "-qm",
         "fixture"], check=True, timeout=30)
    return root


def _inputs(tmp_path: Path, source: Path) -> tuple[Path, Path]:
    cuda_dir = tmp_path / "cuda-build"
    host_dir = tmp_path / "host-build"
    cuda_dir.mkdir(parents=True)
    host_dir.mkdir(parents=True)
    cuda = cuda_dir / "liblorrax_ffi.so"
    host = host_dir / "liblorrax_ffi_host.so"
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("a C compiler is required for bundle-sealer ELF fixtures")
    for path, symbol in ((cuda, "cuda_fixture"), (host, "host_fixture")):
        src = path.with_suffix(".c")
        src.write_text(f"int {symbol}(void) {{ return 1; }}\n",
                       encoding="utf-8")
        result = subprocess.run(
            [cc, "-shared", "-fPIC", f"-Wl,-soname,{path.name}", "-o",
             str(path), str(src)], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=120)
        assert result.returncode == 0, result.stdout
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
        stdout=subprocess.PIPE, text=True, timeout=30).stdout.strip()
    for path in (cuda, host):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (path.parent / "PROVENANCE").write_text(
            f"git_rev={revision}\ngit_dirty=no\nsha256={digest}\n",
            encoding="utf-8")
    return cuda, host


def test_sealer_publishes_one_pair_manifest_and_never_overwrites(tmp_path):
    root = _source_repo(tmp_path)
    cuda, host = _inputs(tmp_path, root)
    output = tmp_path / "bundle"
    manifest = _SEALER.seal(
        root=root, cuda=cuda, host=host, output=output, private=[])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(doc["libraries"]) == {"CUDA", "cpu"}
    assert doc["source"]["dirty"] is False
    assert len(doc["source"]["revision"]) == 40
    assert doc["ffi_abi"] == 3
    assert doc["libraries"]["CUDA"]["soname"] == "liblorrax_ffi.so"
    assert doc["libraries"]["cpu"]["soname"] == "liblorrax_ffi_host.so"
    assert manifest.stat().st_mode & 0o222 == 0
    assert all(path.stat().st_mode & 0o222 == 0
               for path in (output / "lib").iterdir())
    with pytest.raises(FileExistsError, match="immutable bundle output"):
        _SEALER.seal(
            root=root, cuda=cuda, host=host, output=output, private=[])


@pytest.mark.parametrize("name", [
    "libmpi.so.12", "libhdf5_parallel_gnu.so.310", "libcudart.so.13",
    "libnccl.so.2", "libc.so.6", "libstdc++.so.6", "libgomp.so.1",
])
def test_sealer_refuses_machine_runtime_libraries(tmp_path, name):
    root = _source_repo(tmp_path)
    cuda, host = _inputs(tmp_path, root)
    private = tmp_path / name
    private.write_bytes(b"site-owned")
    with pytest.raises(RuntimeError, match="machine runtime"):
        _SEALER.seal(
            root=root, cuda=cuda, host=host,
            output=tmp_path / "must-not-exist", private=[private])
    assert not (tmp_path / "must-not-exist").exists()


def test_sealer_refuses_dirty_source_identity(tmp_path):
    root = _source_repo(tmp_path)
    cuda, host = _inputs(tmp_path, root)
    tracked = root / "src" / "ffi" / "cpp" / "common"
    (tracked / "lorrax_ffi_abi.h").write_text(
        "#define LORRAX_FFI_ABI_VERSION 4\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty source tree"):
        _SEALER.seal(
            root=root, cuda=cuda, host=host,
            output=tmp_path / "must-not-exist", private=[])


def test_sealer_refuses_an_unidentified_or_changed_provider(tmp_path):
    root = _source_repo(tmp_path)
    cuda, host = _inputs(tmp_path, root)
    (host.parent / "PROVENANCE").unlink()
    with pytest.raises(RuntimeError, match="no adjacent PROVENANCE"):
        _SEALER.seal(
            root=root, cuda=cuda, host=host,
            output=tmp_path / "no-receipt", private=[])
    assert not (tmp_path / "no-receipt").exists()

    _, host = _inputs(tmp_path / "second", root)
    host.write_bytes(host.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="changed after its build receipt"):
        _SEALER.seal(
            root=root, cuda=cuda, host=host,
            output=tmp_path / "stale-receipt", private=[])
