"""Checkout closure: an unsealed leg opens the closure its build recorded.

lxkit.native_provider._preload_checkout_closure and
src/ffi/cpp/stage/write_checkout_closure.py, with the ELF reader and dlopen
replaced by tables (no native build needed; stdlib only).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lxkit import native_provider as npv

REPO = Path(__file__).resolve().parents[1]


class _Unusable(OSError):
    pass


def _tree(tmp_path):
    """leg -> libcusolverMp.so.0 -> libcublasmp.so.0, libnccl.so.2;
    leg -> libhdf5_parallel_gnu.so.310 -> libmpi_gnu_123.so.12; leg -> libcudart."""
    stage = tmp_path / "stage" / "lib"
    mpi = tmp_path / "mpich" / "lib"
    hdf5 = tmp_path / "hdf5" / "lib"
    sdk = tmp_path / "sdk" / "lib64"           # the shell's (wrong) copies
    for d in (stage, mpi, hdf5, sdk):
        d.mkdir(parents=True)
    leg = tmp_path / "build" / "liblorrax_ffi.so"
    leg.parent.mkdir()
    files = [leg, stage / "libcusolverMp.so.0", stage / "libcublasmp.so.0",
             mpi / "libmpi_gnu_123.so.12", hdf5 / "libhdf5_parallel_gnu.so.310",
             sdk / "libcusolverMp.so.0"]
    for f in files:
        f.write_bytes(b"")
    needed = {
        leg.name: ["libcusolverMp.so.0", "libhdf5_parallel_gnu.so.310", "libcudart.so.13"],
        "libcusolverMp.so.0": ["libcublasmp.so.0", "libnccl.so.2"],
        "libcublasmp.so.0": ["libnccl.so.2"],
        "libhdf5_parallel_gnu.so.310": ["libmpi_gnu_123.so.12"],
        "libmpi_gnu_123.so.12": [],
    }
    return leg, stage, mpi, hdf5, needed


@pytest.fixture
def fake_elf(monkeypatch):
    table = {}
    monkeypatch.setattr(npv, "_elf_dynamic",
                        lambda p: {npv._DT_NEEDED: table.get(Path(p).name, [])})
    loads = []

    class _CDLL:
        def __init__(self, path, mode=0):
            loads.append(Path(path))
    monkeypatch.setattr(npv.ctypes, "CDLL", _CDLL)
    monkeypatch.setattr(npv, "_PRIVATE_HANDLES", {})
    return table, loads


def _record(leg, private, dirs, schema=1):
    leg.with_name(leg.name + npv.CHECKOUT_CLOSURE_SUFFIX).write_text(json.dumps(
        {"schema": schema, "private_libraries": [str(p) for p in private],
         "library_dirs": [str(d) for d in dirs]}))


def test_no_record_loads_nothing(tmp_path, fake_elf):
    table, loads = fake_elf
    leg, *_rest, needed = _tree(tmp_path)
    table.update(needed)
    npv._preload_checkout_closure(leg, _Unusable)
    assert loads == []


def test_record_loads_recorded_files_dependencies_first(tmp_path, fake_elf):
    table, loads = fake_elf
    leg, stage, mpi, hdf5, needed = _tree(tmp_path)
    table.update(needed)
    _record(leg, [stage / "libcublasmp.so.0", stage / "libcusolverMp.so.0"], [hdf5, mpi])
    npv._preload_checkout_closure(leg, _Unusable)
    assert loads == [stage / "libcublasmp.so.0", stage / "libcusolverMp.so.0",
                     mpi / "libmpi_gnu_123.so.12", hdf5 / "libhdf5_parallel_gnu.so.310"]
    # the device runtime and NCCL stay the environment's; once per leg
    npv._preload_checkout_closure(leg, _Unusable)
    assert len(loads) == 4


def test_record_naming_a_missing_provider_refuses(tmp_path, fake_elf):
    table, _ = fake_elf
    leg, stage, mpi, hdf5, needed = _tree(tmp_path)
    table.update(needed)
    _record(leg, [stage / "gone" / "libcusolverMp.so.0"], [mpi])
    with pytest.raises(_Unusable, match="not a file"):
        npv._preload_checkout_closure(leg, _Unusable)


def test_record_with_unknown_schema_refuses(tmp_path, fake_elf):
    leg, *_ = _tree(tmp_path)
    _record(leg, [], [], schema=99)
    with pytest.raises(_Unusable, match="schema"):
        npv._preload_checkout_closure(leg, _Unusable)


def test_record_cannot_claim_a_runtime_owned_library(tmp_path, fake_elf):
    leg, *_ = _tree(tmp_path)
    nccl = tmp_path / "libnccl.so.2"
    nccl.write_bytes(b"")
    _record(leg, [nccl], [])
    with pytest.raises(_Unusable):
        npv._preload_checkout_closure(leg, _Unusable)


def _writer():
    spec = importlib.util.spec_from_file_location(
        "write_checkout_closure", REPO / "src/ffi/cpp/stage/write_checkout_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_writer_records_private_closure_and_refuses_a_gap(tmp_path, monkeypatch):
    leg, stage, mpi, hdf5, needed = _tree(tmp_path)
    wr = _writer()
    monkeypatch.setattr(npv, "_elf_dynamic",
                        lambda p: {npv._DT_NEEDED: needed.get(Path(p).name, [])})
    assert wr.main([str(leg), "--private-dir", str(stage),
                    "--library-dirs", f"{hdf5}:{mpi}"]) == 0
    doc = json.loads(leg.with_name(leg.name + npv.CHECKOUT_CLOSURE_SUFFIX).read_text())
    assert [Path(p).name for p in doc["private_libraries"]] == [
        "libcublasmp.so.0", "libcusolverMp.so.0"]
    assert doc["library_dirs"] == [str(hdf5.resolve()), str(mpi.resolve())]
    # without the stage the private pair would come from the search path: refuse
    assert wr.main([str(leg), "--library-dirs", str(mpi)]) == 1
