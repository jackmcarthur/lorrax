"""Released-provider refusal and staging consistency; no CUDA client or kernels."""
import ast
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_native_provider_floor_requires_the_stamped_contract(tmp_path):
    """Old or unstamped libraries cannot bypass the newly required native guard."""
    from lxkit import native_provider
    from lxkit.legs import LEGS

    header = (ROOT / 'src/ffi/cpp/common/lorrax_ffi_abi.h').read_text()
    abi = int(re.search(r'^#define LORRAX_FFI_ABI_VERSION (\d+)$', header, re.MULTILINE)[1])
    assert abi >= 13
    for relative in ('src/ffi/common/ffi_loader.py',
                     'services/distrib_la/src/distrib_la/loader.py'):
        tree = ast.parse((ROOT / relative).read_text())
        mirrors = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(name, ast.Name) and name.id == 'LORRAX_FFI_ABI_VERSION'
                           for name in node.targets)]
        assert mirrors == [abi], relative
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'open_and_attest']
        assert len(calls) == 1
        assert any(kw.arg == 'strict_unstamped' and isinstance(kw.value, ast.Constant)
                   and kw.value.value is True for kw in calls[0].keywords), relative

    for platform, leg in LEGS.items():
        options = dict(expected_abi=abi, abi_symbols={platform: leg.abi_symbol},
                       build_hint='rebuild from this source', strict_unstamped=True)
        path = str(tmp_path / leg.so_name)
        old = SimpleNamespace(**{leg.abi_symbol: lambda: 12})
        with pytest.raises(OSError, match='speaks abi=12'):
            native_provider.check_abi(old, platform, path, **options)
        with pytest.raises(OSError, match='no handler-ABI stamp'):
            native_provider.check_abi(SimpleNamespace(), platform, path, **options)
        current = SimpleNamespace(**{leg.abi_symbol: lambda: abi})
        native_provider.check_abi(current, platform, path, **options)


def test_loaded_cusolvermp_release_refuses_before_grid(tmp_path):
    """Compile the actual native guard alone; old releases never reach the grid."""
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('the metadata-only native refusal check needs a C++ compiler')
    source = (ROOT / 'src/ffi/cpp/cusolvermp/context.cc').read_text()
    start = source.index('if (mp_version < 901) {')
    stop = source.index('\n    }', start) + len('\n    }')
    guard = source[start:stop]
    assert stop < source.index('cusolverMpGridMapping_t layout')
    # Only the actual checked branch is compiled. CUDA/MPI headers, native
    # libraries and clients are deliberately absent from this host fixture.
    program = tmp_path / 'guard.cc'
    program.write_text('''#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
void guard(int mp_version, int p, int q) {
    const int mp_major=mp_version/10000, mp_minor=(mp_version/100)%100, mp_patch=mp_version%100;
''' + guard + '''
}
int main(int argc, char** argv) {
    try { guard(std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3])); }
    catch (const std::runtime_error& error) { std::cout << error.what(); return 7; }
    std::cout << "GRID_REACHED";
}
''')
    executable = tmp_path / 'guard'
    subprocess.run([compiler, '-std=c++17', str(program), '-o', str(executable)], check=True)
    for p, q in ((1, 1), (2, 2), (32, 32)):
        for version in (600, 702, 800, 900, 901, 10000):
            result = subprocess.run([str(executable), str(version), str(p), str(q)],
                                    capture_output=True, text=True)
            if version < 901:
                assert result.returncode == 7 and 'GRID_REACHED' not in result.stdout
                assert 'GATE cusolvermp_2d_grid_version' in result.stdout
                assert 'released cuSOLVERMp >= 0.9.1' in result.stdout
            else:
                assert result.returncode == 0 and result.stdout == 'GRID_REACHED'


def test_cuda12_staging_uses_the_declared_released_pair():
    """Installed-wheel and legacy-stage defaults agree with the package authority."""
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    extras = project['project']['optional-dependencies']
    solver = '0.9.1.9318.post1'
    blas = '0.10.0.3695'
    for cuda in ('cuda12', 'cuda13'):
        major = cuda.removeprefix('cuda')
        assert any(f'nvidia-cusolvermp-cu{major}=={solver};' in row for row in extras[cuda])
        assert any(f'nvidia-cublasmp-cu{major}=={blas};' in row for row in extras[cuda])
    stage = (ROOT / 'config/frontera/stage_ffi_deps.sh').read_text()
    assert f'LORRAX_CUSOLVERMP_PIN:=nvidia-cusolvermp-cu12=={solver}' in stage
    assert f'LORRAX_CUBLASMP_PIN:=nvidia-cublasmp-cu12=={blas}' in stage
    pypi = (ROOT / 'src/ffi/cpp/stage/cusolvermp_stage_pypi.sh').read_text()
    assert f'CUSOLVERMP_VERSION:={solver}' in pypi
    assert 'LEGACY_HEADER=' not in pypi
    redist = (ROOT / 'src/ffi/cpp/stage/cusolvermp_stage_cublasmp_redist.sh').read_text()
    assert f'CUBLASMP_VERSION:={blas}' in redist
    assert 'LORRAX_CUSOLVERMP_STAGE:=0.9.1_cuda12.9' in redist
    site = (ROOT / 'config/perlmutter/site_config.sh').read_text()
    assert re.search(r'^LORRAX_NVHPC_SUBPATH="0\.9\.1_cuda12\.9/', site, re.MULTILINE)
    launcher = (ROOT / 'src/ffi/cpp/run_shifter.sh').read_text()
    assert 'LORRAX_NVHPC_SUBPATH:=0.9.1_cuda12.9/' in launcher
