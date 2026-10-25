"""CPU adapter provenance depends on its recipe, not unrelated CUDA defaults.

Run the actual shell checks against a tiny immutable fixture artifact. The
existing forbidden-overlay gate stops each accepted-provenance case before
Cray PMI/MPI is needed. These host checks do not load the fixture as a library.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).parents[1]
CPU_FILES = ('build_mpiwrapper.sh', 'cpu_mpi_env.sh', 'cpu_mpi_config.sh')
DEFAULTS = ('LORRAX_MPIWRAPPER_ROOT_DEFAULT', 'LORRAX_MPIWRAPPER_PREFIX_DEFAULT',
            'LORRAX_MPIWRAPPER_COMMIT_DEFAULT', 'LORRAX_MPIWRAPPER_ABI_DEFAULT',
            'LORRAX_PM_PRGENV_DEFAULT', 'LORRAX_PM_MPICH_DEFAULT', 'LORRAX_PM_CMAKE_DEFAULT')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_env():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(('LORRAX_', 'JAX_', 'SLURM_', 'MPICH_', 'MPIR_'))
            and key not in ('MPITRAMPOLINE_LIB', 'LD_PRELOAD', 'PYTHONPATH', 'BASH_ENV')}


def _fixture(tmp_path):
    checkout = tmp_path/'checkout with spaces'
    config = checkout/'config/perlmutter'
    config.mkdir(parents=True)
    for name in (*CPU_FILES, 'site_config.sh'):
        shutil.copy2(ROOT/'config/perlmutter'/name, config/name)
    gate = checkout/'src/ffi/cpp/gate_one_mpi.sh'
    gate.parent.mkdir(parents=True)
    shutil.copy2(ROOT/'src/ffi/cpp/gate_one_mpi.sh', gate)
    release = tmp_path/'immutable adapter'
    (release/'lib64').mkdir(parents=True)
    so = release/'lib64/libmpiwrapper.so'
    so.write_bytes(b'fixture artifact; never loaded as a shared library\n')
    manifest = release/'build-manifest.txt'
    manifest.write_text('\n'.join((
        'source_commit=966f4231c96153a08295fc7d0bcbd65e916a73fd',
        'mpiabi_version=2.10.0', 'source_modified=false', 'lorrax_recipe_dirty=false',
        'cray_mpich_version=9.0.1', f'artifact_sha256={_sha(so)}',
        f'builder_sha256={_sha(config/CPU_FILES[0])}',
        f'prelude_sha256={_sha(config/CPU_FILES[1])}',
        f'cpu_mpi_config_sha256={_sha(config/CPU_FILES[2])}',
        f'one_mpi_gate_sha256={_sha(gate)}'))+'\n')
    for file in (so, manifest):
        file.chmod(0o444)
    (release/'lib64').chmod(0o555)
    release.chmod(0o555)
    env = _clean_env()
    env.update(LORRAX_MPIWRAPPER_SO=str(so),
               LORRAX_MPI_INIT_FIRST='test stops after provenance before transport')
    return checkout, config, gate, so, manifest, env


def _prelude(config, env):
    return subprocess.run(['bash', '--noprofile', '--norc', '-c', '. "$1"',
                           'bash', str(config/'cpu_mpi_env.sh')],
                          env=env, text=True, capture_output=True)


def _passed_provenance(result):
    assert result.returncode == 2
    assert 'stale Frontera MPI overlay/lifecycle controls' in result.stderr, result.stderr
    assert 'provenance/toolchain/recipe check failed' not in result.stderr


def test_install_and_cpu_recipe_share_unchanged_defaults():
    expected = (str(Path.home()/'software/lorrax_mpiwrapper_cray'),
                str(Path.home()/'software/lorrax_mpiwrapper_cray/current'),
                '966f4231c96153a08295fc7d0bcbd65e916a73fd', '2.10.0',
                'PrgEnv-gnu/8.6.0', 'cray-mpich/9.0.1', 'cmake/3.30.2')
    body = '. "$1"; printf "%s\\n" ' + ' '.join(f'"${{{name}}}"' for name in DEFAULTS)
    for name in ('site_config.sh', 'cpu_mpi_config.sh'):
        result = subprocess.run(['bash', '--noprofile', '--norc', '-c', body, 'bash',
                                 str(ROOT/'config/perlmutter'/name)],
                                env=_clean_env(), text=True, capture_output=True, check=True)
        assert tuple(result.stdout.splitlines()) == expected


def test_gpu_only_site_edits_do_not_change_cpu_provenance_or_builder_input(tmp_path):
    checkout, config, gate, so, manifest, env = _fixture(tmp_path)
    _passed_provenance(_prelude(config, env))
    site = config/'site_config.sh'
    site.write_text(site.read_text().replace('0.9.1_cuda12.9/math_libs/12.9/lib64',
                                           '0.9.2_cuda12.9/math_libs/12.9/lib64')
                    + '\necho UNRELATED_SITE_CONFIG_SOURCED >&2\nreturn 93\n')
    _passed_provenance(_prelude(config, env))
    # The real builder rejects this argument before any candidate/root write;
    # a site-source dependency would instead execute the poison above.
    result = subprocess.run(['bash', str(config/'build_mpiwrapper.sh'), '--not-an-option'],
                            env=_clean_env(), text=True, capture_output=True)
    assert result.returncode == 2 and 'unknown argument' in result.stderr, result.stderr
    assert 'UNRELATED_SITE_CONFIG_SOURCED' not in result.stderr


def test_relevant_changes_and_untracked_manifests_still_refuse(tmp_path):
    for changed in ('config', 'builder', 'prelude', 'gate', 'artifact', 'abi', 'source',
                    'upstream_dirty', 'recipe_dirty', 'mpich', 'missing_cpu_hash', 'writable'):
        checkout, config, gate, so, manifest, env = _fixture(tmp_path/changed)
        files = dict(config=config/'cpu_mpi_config.sh', builder=config/'build_mpiwrapper.sh',
                     prelude=config/'cpu_mpi_env.sh', gate=gate, artifact=so)
        if changed in files:
            file = files[changed]
            file.chmod(0o644)
            file.write_bytes(file.read_bytes()+b'\n# modified recipe/artifact\n')
            if file == so:
                file.chmod(0o444)
        elif changed == 'writable':
            so.chmod(0o644)
        else:
            replacements = dict(abi=('mpiabi_version=2.10.0', 'mpiabi_version=2.9.0'),
                source=('source_commit=966f4231c96153a08295fc7d0bcbd65e916a73fd', 'source_commit=wrong'),
                upstream_dirty=('source_modified=false', 'source_modified=true'),
                recipe_dirty=('lorrax_recipe_dirty=false', 'lorrax_recipe_dirty=true'),
                mpich=('cray_mpich_version=9.0.1', 'cray_mpich_version=9.0.0'),
                missing_cpu_hash=('cpu_mpi_config_sha256=', 'site_config_sha256='))
            before, after = replacements[changed]
            manifest.chmod(0o644)
            manifest.write_text(manifest.read_text().replace(before, after))
            manifest.chmod(0o444)
        result = _prelude(config, env)
        assert result.returncode == 2, (changed, result.stderr)
        want = 'adapter release is writable' if changed == 'writable' else 'provenance/toolchain/recipe check failed'
        assert want in result.stderr, (changed, result.stderr)
        assert 'stale Frontera MPI overlay/lifecycle controls' not in result.stderr
