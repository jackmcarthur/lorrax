"""The enforced dependency closure (lxkit.native_provider), CPU only, no native build.

Build side: write_checkout_closure.py resolves a leg's NEEDED closure into a
record.  Start-up side: the record is checked (missing / stale refuses), its
files are opened by exact path, and the gate requires every recorded family to
be mapped exactly once, as the recorded file.  The ELF reader, dlopen and
/proc/self/maps are replaced by tables.  What only these tests catch is a refusal
that must fire (stale or missing record, a wrong file, a second copy of a family,
a launch library naming the wrong file).
"""
from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lxkit import native_provider as npv

REPO = Path(__file__).resolve().parents[1]


def test_rocm_acceptance_uses_exact_recorded_files_in_a_cold_process(tmp_path, monkeypatch):
    """No shell RPATH can find this MPI; production preloads must find it.

    This tiny ELF fixture tests the loader/verifier protocol, not GPU kernels.
    A conflicting live provider, stale/missing record and wrong ABI must refuse.
    """
    compiler = shutil.which("cc")
    if compiler is None or not sys.platform.startswith("linux"):
        pytest.skip("the cold ELF acceptance fixture needs Linux and a C compiler")
    providers = tmp_path / "providers"
    providers.mkdir()
    mpi = providers / "libmpi_gnu_123.so.12"
    abi = int((REPO / "src/ffi/cpp/common/lorrax_ffi_abi.h").read_text().split(
        "#define LORRAX_FFI_ABI_VERSION ")[1].split()[0])

    def build(path, body, extra=()):
        source = path.with_suffix(".c")
        source.write_text(body)
        subprocess.run([compiler, "-shared", "-fPIC", str(source),
                        "-Wl,-soname," + path.name, "-o", str(path), *extra], check=True)

    build(mpi, "int fixture_mpi(void) { return 17; }")
    build(providers / "librccl.so.1", "int fixture_rccl(void) { return 1; }")
    blas = providers / "libopenblas.so.0"
    build(blas, "int fixture_blas(void) { return 1; }")
    build(providers / "librocblas.so.5", "int fixture_gpu_blas(void) { return 1; }")
    elpa = providers / "libelpa.so.21"
    build(elpa, "extern int fixture_blas(void), fixture_gpu_blas(void); "
                "int fixture_elpa(void) { return fixture_blas()+fixture_gpu_blas(); }",
          [str(blas), str(providers / "librocblas.so.5")])
    leg = tmp_path / "liblorrax_ffi_rocm.so"
    build(leg, f'''extern int fixture_mpi(void), fixture_elpa(void);
        int lorrax_ffi_rocm_abi_version(void) {{ return {abi}; }}
        const char *lorrax_ffi_rocm_build_config(void) {{
            return "leg=rocm abi={abi} phdf5=1";
        }}
        int PhdfFixtureFfi(void) {{ return fixture_mpi()+fixture_elpa(); }}
        void ElpaFixtureFfi(void) {{}}
        void KDftFixtureFfi(void) {{}}
        ''', [str(mpi), str(elpa)])
    writer = _writer()
    monkeypatch.setattr(writer, "plugin_dirs", lambda *a: [])
    assert writer.main([str(leg), "--process-dir", str(providers)]) == 0
    env = dict(os.environ, LORRAX_ROOT=str(REPO),
               LORRAX_GATE_FFTW_PY=sys.executable,
               LORRAX_FFI_VERIFY_STRICT="1", LORRAX_FFI_EXPECT_MPI=mpi.name)
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LORRAX_FFI_VERIFY_ENV",
                "LORRAX_FFI_EXPECT_PEER_SO", "LORRAX_FFI_VERIFY",
                "LORRAX_PHDF5_STAGE", "LORRAX_GATE_ONE_HDF5"):
        env.pop(key, None)
    unresolved = subprocess.run(["ldd", str(leg)], env=env, capture_output=True, text=True)
    assert f"{mpi.name} => not found" in unresolved.stdout
    receipt = tmp_path / "live.tsv"
    probe = REPO / "scripts/verify_ffi_runtime.py"
    # The fresh child also proves that this acceptance path never imports JAX.
    code = ("import runpy,sys; m=runpy.run_path(sys.argv[1]); "
            "rc=m['main'](sys.argv[2:]); assert 'jax' not in sys.modules; sys.exit(rc)")
    command = [sys.executable, "-c", code, str(probe), str(leg),
               "--abi", str(abi), "--map-file", str(receipt)]
    good = subprocess.run(command, env=env, capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    assert f"{mpi.name}\t{mpi.resolve()}\n" in receipt.read_text()
    verify = ["bash", str(REPO / "scripts/verify_ffi_build.sh"), "--leg", "rocm", str(leg)]
    accepted = subprocess.run(verify, env=env, capture_output=True, text=True)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert "every gate ran and passed" in accepted.stdout
    assert "backend 'elpa'" in accepted.stdout
    numpy_blas = providers / "libscipy_openblas64_fixture.so"
    build(numpy_blas, "int fixture_numpy_blas64(void) { return 1; }")
    private_blas = subprocess.run(verify, env={**env, "LD_PRELOAD": str(numpy_blas)},
                                  capture_output=True, text=True)
    assert private_blas.returncode == 0, private_blas.stdout + private_blas.stderr
    # ELPA's OpenBLAS build is authenticated by its build/closure owner;
    # the removed SLATE-specific runtime gate 3d is no longer this protocol.
    wrong_mpi = subprocess.run(verify, env={**env, "LORRAX_FFI_EXPECT_MPI": "libmpi_gnu.so.12"},
                               capture_output=True, text=True)
    assert wrong_mpi.returncode != 0 and "GATE FAILED (1)" in wrong_mpi.stderr

    def refused(cmd=command, child_env=env):
        receipt.write_text("stale successful receipt")
        result = subprocess.run(cmd, env=child_env, capture_output=True, text=True)
        assert result.returncode != 0, result.stdout + result.stderr
        assert not receipt.exists()
        return result.stdout + result.stderr

    assert "ABI MISMATCH" in refused([*command[:-3], str(abi + 1), *command[-2:]])
    other = tmp_path / "other" / mpi.name
    other.parent.mkdir()
    build(other, "int fixture_mpi(void) { return 23; }")
    assert "copies of" in refused(child_env={**env, "LD_PRELOAD": str(other)})
    record = npv.closure_record_path(leg)
    saved = record.read_bytes()
    record.unlink()
    assert "closure" in refused()
    record.write_bytes(saved)
    mpi.write_bytes(mpi.read_bytes() + b"changed after build")
    assert "stale closure record" in refused()


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
def test_record_files_hashes_unknown_or_changed_times_and_keeps_known_fast_path(tmp_path):
    """A missing timestamp cannot authenticate same-size replacement bytes."""
    path = tmp_path / "librocblas.so.5"
    path.write_bytes(b"original")
    expected = npv._sha256(path)
    stamp = path.stat().st_mtime_ns
    calls = []

    def hasher(p):
        calls.append(p)
        return npv._sha256(p)

    def record(mtime):
        entry = npv.ClosureEntry(path.name, path, expected, 8, mtime, "process", "librocblas")
        return npv.ClosureRecord(tmp_path / "manifest.json", "ROCM", path,
                                 "0" * 64, (entry,), "0" * 64)

    for mtime, count in ((stamp, 0), (0, 1)):
        calls.clear()
        npv.check_record_files(record(mtime), hasher=hasher)
        assert len(calls) == count
    # An unchanged file with a new timestamp still passes by its bytes.
    os.utime(path, ns=(stamp, stamp + 1_000_000_000))
    calls.clear()
    npv.check_record_files(record(stamp), hasher=hasher)
    assert calls == [path]
    path.write_bytes(b"replaced")
    os.utime(path, ns=(stamp, stamp + 2_000_000_000))
    for mtime in (0, stamp):
        calls.clear()
        with pytest.raises(_Unusable, match="changed content since the build"):
            npv.check_record_files(record(mtime), hasher=hasher, unusable_cls=_Unusable)
        assert calls == [path]


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
    # a system pthreads OpenBLAS beside the ELPA one is a second BLAS (Frontier, measured)
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
@pytest.mark.parametrize("platform", ["CUDA", "ROCM"])
def test_sealer_device_pairs_keep_source_closure_and_pin_guards(tmp_path, monkeypatch, platform):
    """Real writer/sealer/loader protocol over an ELF metadata fixture.

    Only ELF inspection is replaced; Git source identity, hashes, closure
    rewriting, immutable publication and loader pair refusal run unchanged.
    This does not certify any GPU kernel or vendor library.
    """
    import hashlib
    from lxkit import LEGS
    spec=importlib.util.spec_from_file_location("seal_bundle",REPO/"src/ffi/cpp/stage/seal_bundle.py")
    sealer=importlib.util.module_from_spec(spec);spec.loader.exec_module(sealer)
    source=tmp_path/"source";header=source/"src/ffi/cpp/common/lorrax_ffi_abi.h"
    header.parent.mkdir(parents=True);header.write_text("#define LORRAX_FFI_ABI_VERSION 14\n")
    subprocess.run(["git","init","-q",str(source)],check=True)
    subprocess.run(["git","-C",str(source),"add","."],check=True)
    subprocess.run(["git","-C",str(source),"-c","user.name=fixture","-c","user.email=fixture@example.invalid",
                    "commit","-qm","immutable native source"],check=True)
    revision=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    runtime_dir,private_dir=tmp_path/"runtime",tmp_path/"private"
    runtime_dir.mkdir();private_dir.mkdir()
    device=tmp_path/"artifacts"/platform/LEGS[platform].so_name
    host=tmp_path/"artifacts"/"host"/LEGS["cpu"].so_name
    for path in (device,host):path.parent.mkdir(parents=True);path.write_bytes(path.name.encode())
    mpi=runtime_dir/"libmpi_fixture.so.12";mpi.write_bytes(b"one MPI")
    collective=runtime_dir/("libnccl.so.2" if platform=="CUDA" else "librccl.so.1")
    collective.write_bytes(b"one plugin collective")
    private=private_dir/("libcusolverMp.so.0" if platform=="CUDA" else "libelpa_openmp.so.19")
    private.write_bytes(b"recorded private engine")
    needed={device.name:[private.name],host.name:[mpi.name],private.name:[mpi.name]}
    monkeypatch.setattr(npv,"_elf_dynamic",lambda p:{
        npv._DT_NEEDED:needed.get(Path(p).name,[]),npv._DT_SONAME:[Path(p).name]})
    writer=_writer();monkeypatch.setattr(writer,"plugin_dirs",lambda *a:[])
    for path in (device,host):
        assert writer.main([str(path),"--private-dir",str(private_dir),
                            "--process-dir",str(runtime_dir)])==0
        (path.parent/"PROVENANCE").write_text(
            f"git_rev={revision}\ngit_dirty=no\nsha256={hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    # Pytest fixtures live under /tmp; retain the production guard against
    # one designated purgeable directory rather than disabling path checks.
    purgeable=tmp_path/"purgeable"
    monkeypatch.setattr(sealer,"_SCRATCH_ROOTS",(str(purgeable),))
    for name in ("SCRATCH","PSCRATCH","TMPDIR"):monkeypatch.delenv(name,raising=False)
    search={}
    monkeypatch.setattr(sealer,"_elf_dynamic",lambda p:(Path(p).name,tuple(needed.get(Path(p).name,[])),
        search.get(Path(p),[("RUNPATH","$ORIGIN")])))
    selected={"cuda" if platform=="CUDA" else "rocm":device}
    def seal(output,**kw):
        return sealer.seal(root=source,host=host,output=output,private=kw.pop("private",[private]),**selected,**kw)
    output=tmp_path/"release"
    assert sealer.main(["--root",str(source),"--"+platform.lower(),str(device),"--host",str(host),
                        "--output",str(output),"--private-lib",str(private)])==0
    manifest=output/sealer.MANIFEST
    att=npv._read_bundle(manifest,expected_abi=14,unusable_cls=_Unusable)
    assert set(att.libraries)==set(att.closures)=={platform,"cpu"}
    assert att.source_revision==revision and att.private_libraries==(output/"lib"/private.name,)
    assert all(not p.is_symlink() and not p.stat().st_mode&0o222 for p in [manifest,*att.libraries.values(),*att.private_libraries])
    # Real sealed records omit external mtimes. Both legs must revalidate the
    # shared runtime bytes, even at the same path and size, before loading.
    for record in att.closures.values():
        npv.check_record_files(record, unusable_cls=_Unusable)
    mpi.write_bytes(b"two MPI")
    for record in att.closures.values():
        with pytest.raises(_Unusable, match="changed content since the build"):
            npv.check_record_files(record, unusable_cls=_Unusable)
    mpi.write_bytes(b"one MPI")
    specs={name:{"env":LEGS[name].so_env,"so_name":LEGS[name].so_name,"build_hint":"fixture"}
           for name in (platform,"cpu")}
    env={LEGS[name].so_env:str(path) for name,path in att.libraries.items()}
    assert npv.locate_library(platform,specs=specs,candidates={},expected_abi=14,environ=env)==att.libraries[platform]
    candidates={name:[path] for name,path in att.libraries.items()}
    assert npv.locate_library(platform,specs=specs,candidates=candidates,expected_abi=14,environ={})==att.libraries[platform]
    for missing in (platform,"cpu"):
        with pytest.raises(_Unusable,match="partial sealed-bundle"):
            npv.locate_library(platform,specs=specs,candidates=candidates,expected_abi=14,
                               environ={k:v for k,v in env.items() if k!=LEGS[missing].so_env},unusable_cls=_Unusable)
    with pytest.raises(_Unusable,match="mixed native provider"):
        npv.locate_library(platform,specs=specs,candidates={},expected_abi=14,
                           environ={**env,LEGS['cpu'].so_env:str(host)},unusable_cls=_Unusable)
    with pytest.raises(_Unusable,match="bundle ABI mismatch"):
        npv._read_bundle(manifest,expected_abi=15,unusable_cls=_Unusable)
    with pytest.raises(FileExistsError):seal(output)
    with pytest.raises(RuntimeError,match="select exactly one"):
        sealer.seal(root=source,cuda=device,rocm=device,host=host,output=tmp_path/'bad',private=[private])
    with pytest.raises(SystemExit):
        sealer._parser().parse_args(["--cuda",str(device),"--rocm",str(device),"--host",str(host),"--output",str(tmp_path/'bad_cli')])
    search[device]=[("RUNPATH",str(purgeable))]
    with pytest.raises(RuntimeError,match="scratch or build directory"):seal(tmp_path/"bad_search")
    del search[device]
    stamp=host.parent/"PROVENANCE";original=stamp.read_text()
    stamp.write_text(original.replace(revision,"0"*40))
    with pytest.raises(RuntimeError,match="bundle source"):seal(tmp_path/"bad_source")
    stamp.write_text(original)
    with pytest.raises(RuntimeError,match="incomplete private dependency"):seal(tmp_path/"missing_private",private=[])
    wrong=tmp_path/"wrong"/private.name;wrong.parent.mkdir();wrong.write_bytes(b"different private engine")
    with pytest.raises(RuntimeError,match="not the file.*built and.*verified"):
        seal(tmp_path/"wrong_private",private=[wrong])
    assert not (tmp_path/"wrong_private").exists() and not list(tmp_path.glob('.wrong_private.tmp-*'))
    # The output is immutable, but owner tampering must still fail its hash.
    sealed=att.libraries[platform];sealed.chmod(0o644);sealed.write_bytes(b"changed");sealed.chmod(0o444)
    with pytest.raises(_Unusable,match="byte-size mismatch|SHA-256 mismatch"):
        npv._read_bundle(manifest,expected_abi=14,unusable_cls=_Unusable)

# ---------------------------------------------------------------- launch libraries
