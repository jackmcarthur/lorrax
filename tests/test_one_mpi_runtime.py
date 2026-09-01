"""Live-process complement to the FFI link-time one-MPI gate."""

import io
import os
from pathlib import Path
import subprocess

import pytest

from ffi.common import ffi_loader


_ROOT = Path(__file__).resolve().parents[1]


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


def test_live_mpi_gate_recognizes_generic_gnu_soname(monkeypatch):
    text = _maps(
        "/opt/cray/pe/mpich/9.1.0/lib/libmpi_gnu.so.12.0.0",
        "/foreign/intel/libmpi.so.12",
    )
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(text))
    with pytest.raises(ffi_loader.FfiLibraryUnusable,
                       match="more than one MPI runtime"):
        ffi_loader._assert_one_mapped_mpi_runtime()


def test_perlmutter_adapter_builder_refuses_one_mpi_gate_opt_out():
    env = os.environ.copy()
    env.update({
        "LORRAX_ROOT": str(_ROOT),
        "LORRAX_CHECKOUT": str(_ROOT),
        "LORRAX_GATE_ONE_MPI": "off",
    })
    proc = subprocess.run(
        [str(_ROOT / "config/perlmutter/build_mpiwrapper.sh"), "--fresh"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 2
    assert "cannot disable the one-MPI closure gate" in proc.stderr


def test_perlmutter_adapter_manifest_tracks_the_one_mpi_gate():
    builder = (_ROOT / "config/perlmutter/build_mpiwrapper.sh").read_text()
    prelude = (_ROOT / "config/perlmutter/cpu_mpi_env.sh").read_text()
    assert 'LORRAX_GATE_ONE_MPI=on GATE_TAG=build_mpiw.pm' in builder
    assert 'one_mpi_gate_sha256=$ONE_MPI_GATE_SHA' in builder
    assert 'src/ffi/cpp/gate_one_mpi.sh' in builder
    assert "one_mpi_gate_sha256=" in prelude
    assert "_lorrax_pm_want_one_mpi_gate_sha" in prelude
