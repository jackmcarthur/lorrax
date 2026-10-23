"""The enforced dependency closure (lxkit.native_provider), CPU only, no native build.

Build side: write_checkout_closure.py resolves a leg's NEEDED closure into a
record.  Start-up side: the record is checked (missing / stale refuses), its
files are opened by exact path, and the gate requires every recorded family to
be mapped exactly once, as the recorded file.  The ELF reader, dlopen and
/proc/self/maps are replaced by tables.  What only these tests catch is a refusal
that must fire (stale or missing record, a wrong file, a second copy of a family,
a launch library naming the wrong file, a host SLATE target in a device process).
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from lxkit import native_provider as npv

REPO = Path(__file__).resolve().parents[1]


def test_mapped_paths_deduplicates_segments_before_resolving_aliases(tmp_path, monkeypatch):
    first, second = tmp_path / "libfirst.so", tmp_path / "libsecond.so"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    alias = tmp_path / "libalias.so"
    alias.symlink_to(first)
    raw_paths = [str(first)] * 5 + [str(alias)] * 4 + [str(second)] * 3
    rows = [f"1000-2000 r--p 0000 00:00 1 {path}\n" for path in raw_paths]
    rows += [f"1000-2000 r--p 0000 00:00 1 {second} (deleted)\n",
             "1000-2000 rw-p 0000 00:00 0 [heap]\n"]
    maps_text = ["".join(rows)]
    realpath = npv.os.path.realpath
    calls = []

    def resolve(path):
        calls.append(path)
        return realpath(path)

    previous = {Path(resolve(line.rsplit(maxsplit=1)[-1]))
                for line in rows if "/" in line and not line.rstrip().endswith(" (deleted)")}
    assert len(calls) == 12
    calls.clear()
    monkeypatch.setattr(npv, "open", lambda *a, **k: io.StringIO(maps_text[0]), raising=False)
    monkeypatch.setattr(npv, "os", SimpleNamespace(path=SimpleNamespace(realpath=resolve)))
    assert npv._mapped_paths() == previous == {first, second}
    assert sorted(calls) == sorted({str(first), str(alias), str(second)})
    # The next scan resolves again, including a symlink changed in between.
    alias.unlink()
    alias.symlink_to(second)
    maps_text[0] = "".join(f"1000-2000 r--p 0000 00:00 1 {alias}\n" for _ in range(4))
    calls.clear()
    assert npv._mapped_paths() == {second}
    assert calls == [str(alias)]


class _Unusable(OSError):
    pass


def _writer():
    spec = importlib.util.spec_from_file_location(
        "write_checkout_closure", REPO / "src/ffi/cpp/stage/write_checkout_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Tree:
    """A CUDA leg's world: the stage (cuSOLVERMp 0.9.1), the venv's CUDA wheels
    (what the JAX plugin maps), the pinned MPI/HDF5, and the toolkit that ships
    a cuSOLVERMp 0.8 and a second cray-mpich under the same SONAMEs."""

    def __init__(self, root: Path):
        self.root = root
        d = {k: root / k for k in ("stage", "wheels", "mpich", "hdf5", "sdk", "mpich91", "build")}
        for p in d.values():
            p.mkdir(parents=True)
        self.d = d
        self.leg = d["build"] / "liblorrax_ffi.so"
        files = {
            self.leg: b"leg-v1",
            d["stage"] / "libcusolverMp.so.0": b"cusolvermp-0.9.1",
            d["stage"] / "libcublasmp.so.0": b"cublasmp-0.10",
            d["wheels"] / "libcublas.so.13": b"cublas-pip",
            d["wheels"] / "libnccl.so.2": b"nccl-pip",
            d["mpich"] / "libmpi_gnu_123.so.12": b"mpich-9.0.1",
            d["hdf5"] / "libhdf5_parallel_gnu.so.310": b"hdf5-1.14.3.7",
            d["sdk"] / "libcusolverMp.so.0": b"cusolvermp-0.8",
            d["sdk"] / "libcublas.so.13": b"cublas-sdk",
            d["mpich91"] / "libmpi_gnu.so.12": b"mpich-9.1.0",
        }
        for f, body in files.items():
            f.write_bytes(body)
        self.needed = {
            self.leg.name: ["libcusolverMp.so.0", "libcublas.so.13",
                            "libhdf5_parallel_gnu.so.310", "libcuda.so.1", "libc.so.6",
                            "libfabric.so.1"],
            "libcusolverMp.so.0": ["libcublasmp.so.0", "libnccl.so.2", "libcublas.so.13"],
            "libcublasmp.so.0": ["libnccl.so.2"],
            "libhdf5_parallel_gnu.so.310": ["libmpi_gnu_123.so.12", "libz.so.1"],
        }

    def elf(self, path):
        name = Path(path).name
        return {npv._DT_NEEDED: self.needed.get(name, []), npv._DT_SONAME: [name]}

    def write(self, monkeypatch, **kw):
        monkeypatch.setattr(npv, "_elf_dynamic", self.elf)
        args = [str(self.leg), "--private-dir", str(self.d["stage"]),
                "--process-dir", str(self.d["wheels"]),
                "--library-dirs", f"{self.d['hdf5']}:{self.d['mpich']}:{self.d['sdk']}"]
        wr = _writer()
        monkeypatch.setattr(wr, "plugin_dirs", lambda py, prefix: [])
        return wr.main(args + kw.get("extra", []))

    def record(self):
        return npv.read_checkout_closure(self.leg, "CUDA", unusable_cls=_Unusable)

    def good_maps(self):
        """realpath -> SONAME, as /proc/self/maps would give after a clean start."""
        maps = {e.path: e.soname for e in self.record().entries}
        maps[self.leg] = self.leg.name
        maps[Path("/usr/lib64/libc.so.6")] = "libc.so.6"
        # h5py's auditwheel HDF5: a private SONAME, not a second libhdf5
        maps[self.root / "venv/h5py.libs/libhdf5-9e18f0c6.so.320.0.0"] = "libhdf5-9e18f0c6.so.320"
        return maps


@pytest.fixture
def tree(tmp_path, monkeypatch):
    t = Tree(tmp_path)
    assert t.write(monkeypatch) == 0
    monkeypatch.setattr(npv, "_ENFORCED", {})
    monkeypatch.setattr(npv, "_PLATFORM_LEG", {})
    monkeypatch.setattr(npv, "_REVERIFIED", [False])
    monkeypatch.setattr(npv, "_PRIVATE_HANDLES", {})
    monkeypatch.setattr(npv, "_EXPECTED_CACHE", {})
    monkeypatch.setattr(npv, "_DLOPEN_ORDER", [])
    return t


# ---------------------------------------------------------------- build side
# ---------------------------------------------------------------- record checks
# ---------------------------------------------------------------- the gate
def test_gate_requires_each_recorded_family_mapped_once_as_the_recorded_file(tree, monkeypatch):
    def check(maps, rec=None):
        return npv.check_closure_mapping([rec or tree.record()], maps, unusable_cls=_Unusable)

    verified = check(tree.good_maps())
    assert verified["libcusolverMp"][0] == tree.d["stage"] / "libcusolverMp.so.0"
    assert "libhdf5-9e18f0c6" not in verified

    # the wrong file (cudatoolkit's 0.8 under the same SONAME)
    maps = tree.good_maps()
    del maps[tree.d["stage"] / "libcusolverMp.so.0"]
    maps[tree.d["sdk"] / "libcusolverMp.so.0"] = "libcusolverMp.so.0"
    with pytest.raises(_Unusable) as exc:
        check(maps)
    msg = str(exc.value)
    assert msg.startswith("GATE native-closure: got ")
    assert str(tree.d["sdk"] / "libcusolverMp.so.0") in msg            # the wrong file
    assert str(tree.d["stage"] / "libcusolverMp.so.0") in msg          # the expected one
    assert "fix:" in msg and "LD_LIBRARY_PATH cannot fix this" in msg

    # a byte-identical copy elsewhere is the recorded file
    maps = tree.good_maps()
    del maps[tree.d["stage"] / "libcublasmp.so.0"]
    copy = tree.root / "elsewhere" / "libcublasmp.so.0"
    copy.parent.mkdir()
    copy.write_bytes(b"cublasmp-0.10")
    maps[copy] = "libcublasmp.so.0"
    check(maps)

    # two copies of a family refuse, even when one of them is the right file
    maps = tree.good_maps()
    maps[tree.d["mpich91"] / "libmpi_gnu.so.12"] = "libmpi_gnu.so.12"   # site-default MPI too
    with pytest.raises(_Unusable, match=r"got 2 copies of mpi mapped: .*mpich/.*mpich91/"):
        check(maps)
    maps = tree.good_maps()
    maps[tree.d["sdk"] / "libcublas.so.13"] = "libcublas.so.13"
    with pytest.raises(_Unusable, match="2 copies of libcublas"):
        check(maps)

    # a recorded family not mapped
    maps = tree.good_maps()
    del maps[tree.d["hdf5"] / "libhdf5_parallel_gnu.so.310"]
    with pytest.raises(_Unusable, match="no libhdf5 mapped"):
        check(maps)

    # every violation is named
    maps = tree.good_maps()
    for name in ("libcusolverMp.so.0", "libcublas.so.13"):          # the HPC SDK's copies too
        maps[tree.d["sdk"] / name] = name
    with pytest.raises(_Unusable) as exc:
        check(maps)
    lines = str(exc.value).splitlines()
    assert len(lines) == 2 and all(x.startswith("GATE native-closure: got 2 copies of ") for x in lines)
    assert any("libcusolverMp" in x for x in lines) and any("libcublas " in x for x in lines)

    # an attest entry is enforced only once mapped
    rec = tree.record()
    rccl = tree.d["wheels"] / "librccl.so.1"
    rccl.write_bytes(b"rccl-pip")
    attest = npv.ClosureEntry("librccl.so.1", rccl, npv._sha256(rccl), 8, 0, "attest", "librccl")
    rec = npv.ClosureRecord(rec.origin, rec.platform, rec.leg, rec.leg_sha256,
                            rec.entries + (attest,), rec.digest)
    maps = tree.good_maps()
    verified = check(maps, rec)                                        # not yet mapped: ok
    assert "librccl" not in verified
    assert len(npv._record_identity(rec, verified)) == 64
    other = tree.d["sdk"] / "librccl.so.1"
    other.write_bytes(b"rccl-opt")
    maps[other] = "librccl.so.1"
    with pytest.raises(_Unusable, match="librccl"):
        check(maps, rec)

    # the family comes from the SONAME, not the file name
    # (libopenblas.so.0 is mapped as libopenblas_zen-r0.3.28.so on Frontier)
    assert npv.library_family("libopenblas.so.0") == "libopenblas"
    # a system pthreads OpenBLAS beside the SLATE-HIP one is a second BLAS (Frontier, measured)
    for other_name in ("libopenblas_pthreads.so.0", "libopenblas_openmp.so.0", "libopenblasp.so.0",
                       "libopenblas64_.so.0"):
        assert npv.library_family(other_name) == "libopenblas", other_name
    assert npv.library_family("libscipy_openblas64_-ff84a88b.so") != "libopenblas"
    assert npv.library_family("libmpi_gnu_123.so.12") == npv.library_family("libmpich.so.12") == "mpi"
    assert npv.library_family("libsci_gnu_mp.so.6") == npv.library_family("libsci_gnu_123_mp.so.6")
    assert npv.library_family("libsci_gnu_mpi_mp.so.6") == "libsci_mpi"
    assert npv.library_family("libhdf5_hl_parallel_gnu.so.310") == "libhdf5_hl"
    assert npv.library_family("libhdf5-9e18f0c6.so.320") != "libhdf5"

    # identity: changes with any recorded library, is cached after enforcement, and a library
    # mapped after the leg opened is still refused at the first identity call
    before = npv.expected_closure_identity(tree.leg, "CUDA")
    (tree.d["wheels"] / "libnccl.so.2").write_bytes(b"nccl-pip-2.29")
    assert tree.write(monkeypatch) == 0
    assert npv.expected_closure_identity(tree.leg, "CUDA") != before
    with pytest.raises(npv.ClosureNotEnforced):
        npv.native_closure_identity("CUDA")
    maps = tree.good_maps()
    monkeypatch.setattr(npv, "_mapped_sonames", lambda: maps)
    rec = tree.record()
    enf = npv._enforce(rec, _Unusable, "")
    npv._ENFORCED[rec.leg] = enf
    npv._PLATFORM_LEG["CUDA"] = rec.leg
    ident = npv.native_closure_identity("CUDA")
    assert ident == enf.identity == npv.expected_closure_identity(tree.leg, "CUDA")
    assert len(ident) == 64 and npv.native_closure_identity() != ident
    monkeypatch.setattr(npv, "_REVERIFIED", [False])
    maps[tree.d["mpich91"] / "libmpi_gnu.so.12"] = "libmpi_gnu.so.12"
    with pytest.raises(OSError, match="2 copies of mpi"):
        npv.native_closure_identity("CUDA")


# ---------------------------------------------------------------- two legs, one process
# ---------------------------------------------------------------- sealed bundle
# ---------------------------------------------------------------- launch libraries
