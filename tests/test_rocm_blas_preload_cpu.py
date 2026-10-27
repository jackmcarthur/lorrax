"""Native dependency-scope preload and its ordering before communicator use."""
import ctypes
import os
import hashlib
import shutil
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def dependency(tmp_path):
    cc = shutil.which('cc')
    if cc is None:
        pytest.skip('requires a C compiler for the dependency-scope ABI control')
    (tmp_path / 'dependency.c').write_text('''
static int device = 6, failure = -1, count = 0, seen[16], gets = 0;
int hipGetDevice(int *p) { ++gets; *p = device; return 0; }
int hipSetDevice(int d) { if(d == failure) return 7; device = d; return 0; }
void rocblas_initialize(void) { seen[count++] = device; }
void fail_device(int d) { failure = d; }
int current_device(void) { return device; }
int call_count(void) { return count; }
int call_device(int i) { return seen[i]; }
''')
    (tmp_path / 'provider.c').write_text('int provider(void) { return 15; }\n')
    dep_name = 'libpreload_test_' + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12] + '.so'
    subprocess.run([cc, '-shared', '-fPIC', str(tmp_path / 'dependency.c'),
                    '-Wl,-soname,' + dep_name, '-o', str(tmp_path / dep_name)], check=True)
    provider = tmp_path / 'provider.so'
    subprocess.run([cc, '-shared', '-fPIC', str(tmp_path / 'provider.c'),
                    '-L' + str(tmp_path), '-Wl,--no-as-needed', '-l:' + dep_name,
                    '-Wl,-rpath,$ORIGIN', '-o', str(provider)], check=True)
    return ctypes.CDLL(str(provider), mode=os.RTLD_LOCAL)


def _configure(monkeypatch, dependency):
    import jax
    from ffi.common import ffi_loader
    from runtime import vendor_kernels as owner
    monkeypatch.setattr(owner, '_WARMUP', {})
    monkeypatch.setattr(owner, '_process_leg',
                        lambda: SimpleNamespace(key='ROCM', blas_host_warmup=('float64', 'complex128')))
    monkeypatch.setattr(jax, 'local_devices', lambda: [
        SimpleNamespace(local_hardware_id=i) for i in (3, 1, 3)])
    def no_gemm(*a, **kw):
        raise AssertionError('preload compiled a GEMM')
    monkeypatch.setattr(jax, 'jit', no_gemm)
    def existing_provider(leg):
        assert leg == 'ROCM'
        return dependency
    monkeypatch.setattr(ffi_loader, 'get_lib', existing_provider)
    # The provider's actual DT_NEEDED scope resolves all three native symbols;
    # a second dlopen by a bare library name would escape that attestation.
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **kw: pytest.fail('second native open'))
    return owner


def test_preload_uses_provider_dependency_scope_synchronously_and_restores_device(monkeypatch, dependency):
    owner = _configure(monkeypatch, dependency)
    result = owner.start_blas_host_warmup(say=lambda text: None)
    assert dependency.call_count() == 2
    assert result['done'] and result['error'] is None
    assert result['method'] == 'rocblas_initialize' and result['devices'] == [1, 3]
    assert result['run_s'] >= 0
    assert dependency.rocblas_initialize.argtypes == []
    assert dependency.rocblas_initialize.restype is None
    assert [dependency.call_device(i) for i in range(2)] == [1, 3]
    assert dependency.current_device() == 6
    assert owner.receipt()['blas_warmup'] == result
    assert owner.start_blas_host_warmup() == result
    assert dependency.call_count() == 2


def test_hip_selection_failure_restores_device_and_never_silently_skips_preload(monkeypatch, dependency):
    owner = _configure(monkeypatch, dependency)
    dependency.fail_device(3)
    with pytest.raises(RuntimeError, match=r'hipSetDevice\(3\) failed'):
        owner.start_blas_host_warmup()
    assert dependency.call_count() == 1 and dependency.call_device(0) == 1
    assert dependency.current_device() == 6
    assert not owner.receipt()['blas_warmup']['done']
    with pytest.raises(RuntimeError, match=r'hipSetDevice\(3\) failed'):
        owner.start_blas_host_warmup()
    assert dependency.call_count() == 1


def test_restore_failure_refuses_and_other_legs_do_not_open_vendor_libraries(monkeypatch, dependency):
    owner = _configure(monkeypatch, dependency)
    dependency.fail_device(6)
    with pytest.raises(RuntimeError, match='restoring HIP device failed'):
        owner.start_blas_host_warmup()
    assert dependency.call_count() == 2
    from ffi.common import ffi_loader
    monkeypatch.setattr(ffi_loader, 'get_lib', lambda *a: pytest.fail('unused vendor open'))
    for key in ('CUDA', 'HOST'):
        monkeypatch.setattr(owner, '_process_leg',
                            lambda key=key: SimpleNamespace(key=key, blas_host_warmup=False))
        assert owner.start_blas_host_warmup() == {}


def test_actual_startup_preloads_after_attestation_and_before_first_mesh_collective(monkeypatch, dependency):
    import jax
    import runtime
    from common import collectives, jax_compile_cache
    from ffi.common import ffi_loader
    from runtime import network_env
    owner = _configure(monkeypatch, dependency)
    monkeypatch.setattr(runtime, '_STACK', None)
    monkeypatch.setattr(runtime, '_print_rank0', lambda *a, **kw: None)
    for name in ('install_failfast_excepthook', 'bootstrap', '_enforce_supported_jax',
                 '_enforce_cpu_mpi_thread_multiple', '_enforce_gpu_placement'):
        monkeypatch.setattr(runtime, name, lambda *a, **kw: None)
    monkeypatch.setattr(jax_compile_cache, 'install_compile_agreement', lambda: None)
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(jax, 'process_index', lambda: 0)
    monkeypatch.setattr(network_env, 'network_configuration', lambda: {})
    opens = []
    def attested(leg):
        assert leg in (None, 'ROCM')
        opens.append(leg)
        return dependency
    monkeypatch.setattr(ffi_loader, 'get_lib', attested)
    class BeforeFirstCollective(Exception):
        pass
    def prepare_mesh(**kw):
        assert opens == [None, 'ROCM']
        assert owner.receipt()['blas_warmup']['done']
        assert dependency.call_count() == 2 and dependency.current_device() == 6
        raise BeforeFirstCollective()
    monkeypatch.setattr(collectives, 'prepare_mesh', prepare_mesh)
    with pytest.raises(BeforeFirstCollective):
        runtime.initialize_communicator_stack()
