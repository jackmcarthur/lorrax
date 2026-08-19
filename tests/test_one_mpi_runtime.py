"""Live-process complement to the FFI link-time one-MPI gate."""

import io

import pytest

from ffi.common import ffi_loader


def _maps(*paths):
    return "".join(
        f"7f000000-7f001000 r-xp 00000000 00:00 0 {path}\n"
        for path in paths
    )


def test_live_mpi_gate_refuses_two_runtime_objects(monkeypatch):
    text = _maps(
        "/opt/cray/pe/mpich/lib/libmpi_gnu_123.so.12.0.0",
        "/foreign/intel/libmpi.so.12",
    )
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(text))
    with pytest.raises(ffi_loader.FfiLibraryUnusable,
                       match="more than one MPI runtime"):
        ffi_loader._assert_one_mapped_mpi_runtime()


def test_live_mpi_gate_deduplicates_filename_aliases(monkeypatch):
    text = _maps(
        "/stage/libmpi_gnu_123.so.12",
        "/opt/cray/pe/mpich/lib/libmpi_gnu_123.so.12.0.0",
    )
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(text))
    monkeypatch.setattr(
        ffi_loader.os.path, "realpath",
        lambda path: "/opt/cray/pe/mpich/lib/libmpi_gnu_123.so.12.0.0")
    ffi_loader._assert_one_mapped_mpi_runtime()


def test_live_mpi_gate_ignores_adapters_and_gtl(monkeypatch):
    text = _maps(
        "/lib/libmpitrampoline.so.5",
        "/lib/libmpiwrapper.so",
        "/lib/libmpi_gtl_cuda.so.0",
        "/opt/cray/pe/mpich/lib/libmpi_gnu_123.so.12.0.0",
    )
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(text))
    ffi_loader._assert_one_mapped_mpi_runtime()
