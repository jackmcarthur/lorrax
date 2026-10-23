"""Frontier acceptance orchestration: pinned FFTW, CPU/GPU split, no false PASS."""
import os
from pathlib import Path
import subprocess


def test_frontier_verifier_requires_fft_identity_and_propagates_failed_steps(tmp_path):
    root = tmp_path / "checkout"
    config = root / "config/frontier"
    config.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "config/frontier/verify.sh"
    script = config / "verify.sh"
    script.write_bytes(source.read_bytes())
    host = root / "build_host"
    host.mkdir()
    fftw = root / "fftw/lib"
    fftw.mkdir(parents=True)
    (fftw / "libfftw3.so").touch()
    (host / "CMakeCache.txt").write_text(
        f"LORRAX_FFTW3_LIBRARY:UNINITIALIZED={fftw}/libfftw3.so\n")
    (config / "gpu_env.sh").write_text(f'''
export LORRAX_FFI_HOST_SO='{host}/liblorrax_ffi_host.so'
export VIRTUAL_ENV='{root}/venv'
export LORRAX_OLCF_MPI_SONAME=libmpi_gnu_123.so.12
export LORRAX_OLCF_HDF5_ROOT='{root}/hdf5'
''')
    binary = tmp_path / "bin"
    binary.mkdir()
    srun = binary / "srun"
    srun.write_text('''#!/usr/bin/env bash
printf '%s\\t' "$LORRAX_FFTW3_STAGE" "$@" >> "$PROBE_LOG"
printf '\\n' >> "$PROBE_LOG"
case "$*" in *"--leg rocm"*) exit "${PROBE_ROCM_RC:-0}";; esac
''')
    srun.chmod(0o755)
    log = tmp_path / "steps.txt"
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}",
               SLURM_JOB_ID="fixture", PROBE_LOG=str(log))
    for key in ("LORRAX_FFTW3_STAGE", "LORRAX_FFI_VERIFY", "LORRAX_GATE_ONE_MPI",
                "LORRAX_GATE_ONE_HDF5", "LORRAX_GATE_ONE_FFTW"):
        env.pop(key, None)
    passed = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    steps = log.read_text().splitlines()
    assert len(steps) == 3 and all(row.startswith(str(fftw.parent) + "\t") for row in steps)
    assert "JAX_PLATFORMS=cpu" in steps[0] and "--leg\thost" in steps[0]
    assert "--leg\trocm" in steps[1]
    assert "-n4" in steps[2] and "--gpus-per-task=1" in steps[2]
    assert "JAX_PLATFORMS=rocm" in steps[2]
    assert f"{root}/tests/test_fourier_plan_xla_gate.py" in steps[2]
    log.unlink()
    failed = subprocess.run(["bash", str(script)], env={**env, "PROBE_ROCM_RC": "42"},
                            capture_output=True, text=True)
    assert failed.returncode == 42 and "PASS:" not in failed.stdout
    assert len(log.read_text().splitlines()) == 2
    log.unlink()
    (fftw / "libfftw3.so").unlink()
    refused = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert refused.returncode != 0 and not log.exists()
    assert "No compiled FFTW hint" in refused.stderr
