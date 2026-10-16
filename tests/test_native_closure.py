"""The enforced dependency closure (lxkit.native_provider), CPU only, no native build.

Build side: write_checkout_closure.py resolves a leg's NEEDED closure into a
record.  Start-up side: the record is checked (missing / stale refuses), its
files are opened by exact path, and the gate requires every recorded family to
be mapped exactly once, as the recorded file.  The ELF reader, dlopen and
/proc/self/maps are replaced by tables.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from lxkit import native_provider as npv

REPO = Path(__file__).resolve().parents[1]


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
    return t


# ---------------------------------------------------------------- build side
def test_record_is_the_full_closure_dependency_first(tree):
    doc = json.loads(npv.closure_record_path(tree.leg).read_text())
    got = [(lib["soname"], lib["role"], Path(lib["path"]).parent.name) for lib in doc["libraries"]]
    assert got == [
        ("libnccl.so.2", "process", "wheels"),         # cuBLASMp's dependency, first
        ("libcublasmp.so.0", "private", "stage"),      # stage, never the SDK's 0.8
        ("libcublas.so.13", "process", "wheels"),      # the JAX plugin's copy, not the SDK's
        ("libcusolverMp.so.0", "private", "stage"),
        ("libmpi_gnu_123.so.12", "pinned", "mpich"),
        ("libhdf5_parallel_gnu.so.310", "pinned", "hdf5"),
    ]
    assert {e["soname"] for e in doc["external"]} == {"libcuda.so.1", "libc.so.6",
                                                      "libfabric.so.1", "libz.so.1"}
    assert doc["schema"] == npv.CLOSURE_SCHEMA and doc["platform"] == "CUDA"
    assert doc["record_sha256"] == npv.closure_record_digest(doc)
    assert {lib["family"] for lib in doc["libraries"]} >= {"mpi", "libhdf5", "libcusolverMp"}


def test_writer_refuses_a_must_pin_library_left_to_the_run_shell(tmp_path, monkeypatch):
    t = Tree(tmp_path)
    (t.d["mpich"] / "libmpi_gnu_123.so.12").unlink()     # MPI in no recorded dir
    assert t.write(monkeypatch) == 1
    assert not npv.closure_record_path(t.leg).exists()


# ---------------------------------------------------------------- record checks
def test_missing_record_refuses(tree):
    npv.closure_record_path(tree.leg).unlink()
    with pytest.raises(_Unusable, match=r"GATE native-closure: got no closure record .*fix: rebuild"):
        tree.record()


def test_stale_record_for_other_leg_bytes_refuses(tree):
    tree.leg.write_bytes(b"leg-v2")                       # rebuilt, record not rewritten
    with pytest.raises(_Unusable, match="stale closure record .* written for a leg"):
        tree.record()


def test_stale_record_for_a_moved_or_changed_file_refuses(tree):
    rec = tree.record()
    (tree.d["stage"] / "libcublasmp.so.0").write_bytes(b"cublasmp-0.11-larger")
    with pytest.raises(_Unusable, match="stale closure record .*bytes, recorded"):
        npv.check_record_files(rec, unusable_cls=_Unusable)
    (tree.d["stage"] / "libcublasmp.so.0").unlink()
    with pytest.raises(_Unusable, match="no longer exists"):
        npv.check_record_files(rec, unusable_cls=_Unusable)


def test_same_size_rewrite_is_caught_by_hash(tree):
    rec = tree.record()
    f = tree.d["mpich"] / "libmpi_gnu_123.so.12"
    st = f.stat()
    f.write_bytes(b"mpich-9.9.9")                         # same length ...
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))   # ... new mtime
    with pytest.raises(_Unusable, match="changed content"):
        npv.check_record_files(rec, unusable_cls=_Unusable)


def test_old_schema_refuses(tree):
    side = npv.closure_record_path(tree.leg)
    doc = json.loads(side.read_text())
    doc["schema"] = 1
    side.write_text(json.dumps(doc))
    with pytest.raises(_Unusable, match="schema 1"):
        tree.record()


# ---------------------------------------------------------------- the gate
def test_gate_passes_the_recorded_closure(tree):
    verified = npv.check_closure_mapping([tree.record()], tree.good_maps(),
                                         unusable_cls=_Unusable)
    assert verified["libcusolverMp"][0] == tree.d["stage"] / "libcusolverMp.so.0"
    assert "libhdf5-9e18f0c6" not in verified


def test_gate_refuses_the_wrong_file(tree):
    maps = tree.good_maps()
    del maps[tree.d["stage"] / "libcusolverMp.so.0"]
    maps[tree.d["sdk"] / "libcusolverMp.so.0"] = "libcusolverMp.so.0"   # cudatoolkit's 0.8
    with pytest.raises(_Unusable) as exc:
        npv.check_closure_mapping([tree.record()], maps, unusable_cls=_Unusable)
    msg = str(exc.value)
    assert msg.startswith("GATE native-closure: got ")
    assert str(tree.d["sdk"] / "libcusolverMp.so.0") in msg            # the wrong file
    assert str(tree.d["stage"] / "libcusolverMp.so.0") in msg          # the expected one
    assert "fix:" in msg and "LD_LIBRARY_PATH cannot fix this" in msg


def test_gate_accepts_a_byte_identical_copy(tree):
    maps = tree.good_maps()
    del maps[tree.d["stage"] / "libcublasmp.so.0"]
    copy = tree.root / "elsewhere" / "libcublasmp.so.0"
    copy.parent.mkdir()
    copy.write_bytes(b"cublasmp-0.10")
    maps[copy] = "libcublasmp.so.0"
    npv.check_closure_mapping([tree.record()], maps, unusable_cls=_Unusable)


def test_gate_refuses_two_copies_of_a_family(tree):
    maps = tree.good_maps()
    maps[tree.d["mpich91"] / "libmpi_gnu.so.12"] = "libmpi_gnu.so.12"   # site-default MPI too
    with pytest.raises(_Unusable, match=r"got 2 copies of mpi mapped: .*mpich/.*mpich91/"):
        npv.check_closure_mapping([tree.record()], maps, unusable_cls=_Unusable)


def test_gate_refuses_two_copies_even_if_one_is_right(tree):
    maps = tree.good_maps()
    maps[tree.d["sdk"] / "libcublas.so.13"] = "libcublas.so.13"
    with pytest.raises(_Unusable, match="2 copies of libcublas"):
        npv.check_closure_mapping([tree.record()], maps, unusable_cls=_Unusable)


def test_gate_refuses_a_recorded_family_not_mapped(tree):
    maps = tree.good_maps()
    del maps[tree.d["hdf5"] / "libhdf5_parallel_gnu.so.310"]
    with pytest.raises(_Unusable, match="no libhdf5 mapped"):
        npv.check_closure_mapping([tree.record()], maps, unusable_cls=_Unusable)


def test_family_comes_from_the_soname_not_the_file_name(tree):
    """libopenblas.so.0 is mapped as libopenblas_zen-r0.3.28.so on Frontier."""
    assert npv.library_family("libopenblas.so.0") == "libopenblas"
    # a system pthreads OpenBLAS beside the SLATE-HIP one is a second BLAS (Frontier, measured)
    for other in ("libopenblas_pthreads.so.0", "libopenblas_openmp.so.0", "libopenblasp.so.0",
                  "libopenblas64_.so.0"):
        assert npv.library_family(other) == "libopenblas", other
    assert npv.library_family("libscipy_openblas64_-ff84a88b.so") != "libopenblas"
    assert npv.library_family("libmpi_gnu_123.so.12") == npv.library_family("libmpich.so.12") == "mpi"
    assert npv.library_family("libsci_gnu_mp.so.6") == npv.library_family("libsci_gnu_123_mp.so.6")
    assert npv.library_family("libsci_gnu_mpi_mp.so.6") == "libsci_mpi"
    assert npv.library_family("libhdf5_hl_parallel_gnu.so.310") == "libhdf5_hl"
    assert npv.library_family("libhdf5-9e18f0c6.so.320") != "libhdf5"


# ---------------------------------------------------------------- preload, two legs
def _host_record(tree, slate_dir: Path):
    slate_dir.mkdir()
    (slate_dir / "libslate.so.2").write_bytes(b"slate-" + slate_dir.name.encode())
    host = tree.d["build"] / "liblorrax_ffi_host.so"
    host.write_bytes(b"host-leg")
    body = {"schema": 2, "platform": "cpu", "leg_sha256": npv._sha256(host),
            "libraries": [
                {"soname": "libslate.so.2", "path": str(slate_dir / "libslate.so.2"),
                 "sha256": npv._sha256(slate_dir / "libslate.so.2"), "bytes": 10,
                 "role": "private", "family": "libslate"},
                {"soname": "libmpi_gnu_123.so.12",
                 "path": str(tree.d["mpich"] / "libmpi_gnu_123.so.12"),
                 "sha256": npv._sha256(tree.d["mpich"] / "libmpi_gnu_123.so.12"),
                 "bytes": 11, "role": "pinned", "family": "mpi"}]}
    return npv.parse_closure(body, origin=host, leg=host, platform="cpu")


def test_preload_opens_recorded_files_and_skips_what_an_open_leg_answers(tree, monkeypatch):
    loads = []
    monkeypatch.setattr(npv.ctypes, "CDLL", lambda path, mode=0: loads.append(Path(path)))
    cuda = tree.record()
    npv._preload_closure(cuda, _Unusable)
    assert loads == [e.path for e in cuda.entries]
    npv._ENFORCED[cuda.leg] = npv._Enforced(cuda, "x")
    loads.clear()
    host = _host_record(tree, tree.root / "slate_cpu")
    npv._preload_closure(host, _Unusable)
    assert loads == [tree.root / "slate_cpu" / "libslate.so.2"]      # MPI already open


def test_device_leg_slate_wins_and_any_other_disagreement_refuses(tree, monkeypatch):
    monkeypatch.setattr(npv.ctypes, "CDLL", lambda path, mode=0: None)
    rocm_slate = tree.root / "slate_hip"
    host = _host_record(tree, tree.root / "slate_cpu")
    rocm = npv.ClosureRecord(origin=tree.leg, platform="ROCM", leg=tree.leg, leg_sha256="0" * 64,
                             entries=(npv.ClosureEntry("libslate.so.2", rocm_slate / "libslate.so.2",
                                                       "1" * 64, 1, 0, "private", "libslate"),),
                             digest="d")
    want = npv._merge_expectations([rocm, host], _Unusable)
    assert want["libslate"][0].path == rocm_slate / "libslate.so.2"   # the device build answers
    mpi_other = npv.ClosureRecord(origin=tree.leg, platform="ROCM", leg=tree.leg, leg_sha256="0" * 64,
                                  entries=(npv.ClosureEntry("libmpi_gnu.so.12",
                                                            tree.d["mpich91"] / "libmpi_gnu.so.12",
                                                            "2" * 64, 1, 0, "pinned", "mpi"),),
                                  digest="e")
    with pytest.raises(_Unusable, match="one mpi for the whole process"):
        npv._merge_expectations([mpi_other, host], _Unusable)
    # host first, device second: the device build must not be shadowed -> refuse
    with pytest.raises(_Unusable, match="one libslate"):
        npv._merge_expectations([host, rocm], _Unusable)


# ---------------------------------------------------------------- identity
def test_identity_is_cached_after_enforcement_and_keys_the_closure(tree, monkeypatch):
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
    # a library mapped after the leg opened is still refused at the first identity call
    monkeypatch.setattr(npv, "_REVERIFIED", [False])
    maps[tree.d["mpich91"] / "libmpi_gnu.so.12"] = "libmpi_gnu.so.12"
    with pytest.raises(OSError, match="2 copies of mpi"):
        npv.native_closure_identity("CUDA")


def test_identity_changes_with_any_recorded_library(tree, monkeypatch):
    before = npv.expected_closure_identity(tree.leg, "CUDA")
    (tree.d["wheels"] / "libnccl.so.2").write_bytes(b"nccl-pip-2.29")
    assert tree.write(monkeypatch) == 0
    assert npv.expected_closure_identity(tree.leg, "CUDA") != before


# ---------------------------------------------------------------- sealed bundle
def _sealer():
    spec = importlib.util.spec_from_file_location(
        "seal_bundle", REPO / "src/ffi/cpp/stage/seal_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sealed_closure_points_private_entries_at_the_sealed_copies(tree):
    seal = _sealer()
    lib = tree.root / "bundle" / "lib"
    lib.mkdir(parents=True)
    copies = {}
    for name in ("libcusolverMp.so.0", "libcublasmp.so.0"):
        copies[name] = lib / name
        copies[name].write_bytes((tree.d["stage"] / name).read_bytes())
    body = seal._sealed_closure(tree.leg, "CUDA", copies, REPO, publishable=lambda p, r: True)
    priv = [x for x in body["libraries"] if x["role"] == "private"]
    assert all("path" not in x and x["bundle_path"].startswith("lib/") for x in priv)
    rec = npv.parse_closure(body, origin=lib.parent / "m.json", leg=tree.leg, platform="CUDA",
                            bundle_root=lib.parent)
    assert {e.path for e in rec.entries if e.role == "private"} == set(copies.values())
    # relocation keeps the record digest: it binds bundle-relative paths
    moved = tree.root / "moved"
    (lib.parent).rename(moved)
    rec2 = npv.parse_closure(body, origin=moved / "m.json", leg=tree.leg, platform="CUDA",
                             bundle_root=moved)
    assert rec2.digest == rec.digest
    # a sealed copy that is not the recorded file refuses
    (moved / "lib" / "libcublasmp.so.0").write_bytes(b"cublasmp-0.8")
    with pytest.raises(RuntimeError, match="not the file"):
        seal._sealed_closure(tree.leg, "CUDA", {k: moved / "lib" / k for k in copies},
                             REPO, publishable=lambda p, r: True)


def test_sealed_closure_refuses_scratch_machine_files(tree):
    seal = _sealer()
    with pytest.raises(RuntimeError, match="cannot back a published bundle"):
        seal._sealed_closure(tree.leg, "CUDA",
                             {n: tree.d["stage"] / n for n in ("libcusolverMp.so.0", "libcublasmp.so.0")},
                             REPO, publishable=lambda p, r: "wheels" not in str(p))


def test_writer_follows_the_loaders_rpath_like_ld_so(tmp_path, monkeypatch):
    """Frontier's ROCm leg: DT_RPATH -> the HIP SLATE install; libslate.so.2 has no
    search path of its own, so its BLAS++ resolves through the leg's DT_RPATH."""
    slate = tmp_path / "slate_hip" / "lib64"
    slate.mkdir(parents=True)
    leg = tmp_path / "build_rocm" / "liblorrax_ffi_rocm.so"
    leg.parent.mkdir()
    for f in (leg, slate / "libslate.so.2", slate / "libblaspp.so.2", slate / "libopenblas.so.0"):
        f.write_bytes(f.name.encode())
    table = {leg.name: {npv._DT_NEEDED: ["libslate.so.2", "libopenblas.so.0"],
                        npv._DT_RPATH: [f"$ORIGIN:{slate}"]},
             "libslate.so.2": {npv._DT_NEEDED: ["libblaspp.so.2"]},
             "libblaspp.so.2": {npv._DT_NEEDED: ["libopenblas.so.0"]}}
    elf = lambda p: table.get(Path(p).name, {})            # noqa: E731
    libs, external, refusals = npv.resolve_closure(leg, elf=elf)
    assert refusals == []
    assert [(x["soname"], x["role"]) for x in libs] == [
        ("libopenblas.so.0", "rpath"), ("libblaspp.so.2", "private"), ("libslate.so.2", "private")]
    # an object with its own DT_RUNPATH does not inherit: then BLAS++ is unresolved
    table["libslate.so.2"][npv._DT_RUNPATH] = ["/nonexistent"]
    libs, external, refusals = npv.resolve_closure(leg, elf=elf)
    assert refusals == ["libblaspp.so.2 (needed by libslate.so.2)"]


def test_an_attest_entry_is_enforced_only_once_mapped(tree):
    rec = tree.record()
    rccl = tree.d["wheels"] / "librccl.so.1"
    rccl.write_bytes(b"rccl-pip")
    attest = npv.ClosureEntry("librccl.so.1", rccl, npv._sha256(rccl), 8, 0, "attest", "librccl")
    rec = npv.ClosureRecord(rec.origin, rec.platform, rec.leg, rec.leg_sha256,
                            rec.entries + (attest,), rec.digest)
    maps = tree.good_maps()
    verified = npv.check_closure_mapping([rec], maps, unusable_cls=_Unusable)  # not yet mapped: ok
    assert "librccl" not in verified
    assert len(npv._record_identity(rec, verified)) == 64
    other = tree.d["sdk"] / "librccl.so.1"
    other.write_bytes(b"rccl-opt")
    maps[other] = "librccl.so.1"
    with pytest.raises(_Unusable, match="librccl"):
        npv.check_closure_mapping([rec], maps, unusable_cls=_Unusable)


def test_a_direct_need_resolves_through_the_legs_rpath_first(tmp_path):
    """ld.so is breadth-first: the leg's libopenblas.so.0 comes from the leg's
    RPATH (the SLATE-HIP install's copy) even though libslate's RUNPATH names the
    OpenBLAS build tree -- what Frontier maps."""
    hip, ob = tmp_path / "hip" / "lib64", tmp_path / "openblas" / "lib"
    for d in (hip, ob):
        d.mkdir(parents=True)
    leg = tmp_path / "liblorrax_ffi_rocm.so"
    for f in (leg, hip / "libslate.so.2", hip / "libopenblas.so.0", ob / "libopenblas.so.0"):
        f.write_bytes(str(f).encode())
    table = {leg.name: {npv._DT_NEEDED: ["libslate.so.2", "libopenblas.so.0"],
                        npv._DT_RPATH: [str(hip)]},
             "libslate.so.2": {npv._DT_NEEDED: ["libopenblas.so.0"], npv._DT_RUNPATH: [str(ob)]}}
    libs, _, refusals = npv.resolve_closure(leg, elf=lambda p: table.get(Path(p).name, {}))
    assert refusals == []
    assert [(x["soname"], Path(x["path"]).parent) for x in libs] == [
        ("libopenblas.so.0", hip), ("libslate.so.2", hip)]


def test_every_violation_is_named(tree):
    maps = tree.good_maps()
    for name in ("libcusolverMp.so.0", "libcublas.so.13"):          # the HPC SDK's copies too
        maps[tree.d["sdk"] / name] = name
    with pytest.raises(_Unusable) as exc:
        npv.check_closure_mapping([tree.record()], maps, unusable_cls=_Unusable)
    lines = str(exc.value).splitlines()
    assert len(lines) == 2 and all(x.startswith("GATE native-closure: got 2 copies of ") for x in lines)
    assert any("libcusolverMp" in x for x in lines) and any("libcublas " in x for x in lines)


def test_sealed_private_check_covers_the_opened_legs_only(tmp_path, monkeypatch):
    lib = tmp_path / "bundle" / "lib"
    lib.mkdir(parents=True)
    names = ("libcusolverMp.so.0", "libslate.so.2")
    for n in names:
        (lib / n).write_bytes(n.encode())
    def rec(platform, name):
        e = npv.ClosureEntry(name, lib / name, "0" * 64, 1, 0, "private", npv.library_family(name))
        return npv.ClosureRecord(lib, platform, lib / "leg", "0" * 64, (e,), platform)
    att = npv.BundleAttestation(manifest_path=lib.parent / "m.json", bundle_root=lib.parent,
                                bundle_id="b", source_revision="0" * 40, abi=6, libraries={},
                                library_hashes={}, private_libraries=tuple(lib / n for n in names),
                                closures={"CUDA": rec("CUDA", names[0]), "cpu": rec("cpu", names[1])})
    monkeypatch.setattr(npv, "_mapped_paths", lambda: {lib / names[0], Path("/usr/lib64/libc.so.6")})
    monkeypatch.setattr(npv, "_mapped_sonames", lambda: {lib / names[0]: names[0]})
    npv._attest_private_mapping(att, _Unusable, ["CUDA"])          # host SLATE not open yet: fine
    with pytest.raises(_Unusable, match="libslate.so.2"):
        npv._attest_private_mapping(att, _Unusable, ["CUDA", "cpu"])
