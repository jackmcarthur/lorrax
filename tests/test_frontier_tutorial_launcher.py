"""Actual tutorial shell command construction: native Slurm task GPU binding."""
import os
from pathlib import Path
import subprocess


def test_tutorial_frontier_gpu_binding_and_cpu_and_perlmutter_branches(tmp_path):
    root = tmp_path / "checkout"
    binary = tmp_path / "venv/bin"
    binary.mkdir(parents=True)
    # The launcher queries only the site's identity before sourcing its owner.
    python = binary / "python"
    python.write_text('''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then printf '%s\\n' "$PROBE_SITE"; else exit 91; fi
''')
    python.chmod(0o755)
    for site in ("frontier", "perlmutter"):
        config = root / "config" / site
        config.mkdir(parents=True)
        for name in ("gpu_env.sh", "cpu_mpi_env.sh"):
            (config / name).write_text("export PATH=\"$VIRTUAL_ENV/bin:$PATH\"\n")
    srun = binary / "srun"
    srun.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$@" > "$PROBE_LOG"
''')
    srun.chmod(0o755)
    log = tmp_path / "command.txt"
    script = (Path(__file__).resolve().parents[1] /
        "examples/MoS2_SOC_2026/2-gw/3-lorrax_fully_rel/lorrax_env.sh")
    env = dict(os.environ, LORRAX=str(root), VIRTUAL_ENV=str(binary.parent),
               PATH=f"{binary}:{os.environ['PATH']}", PROBE_LOG=str(log))
    for site, platform in (("frontier", "rocm"), ("frontier", "cpu"),
                           ("perlmutter", "cuda"), ("perlmutter", "cpu")):
        result = subprocess.run(["bash", "-c",
            'source "$1"; lorrax_run gw.gw_jax -i fixture.in', "probe", str(script)],
            env={**env, "PROBE_SITE": site, "JAX_PLATFORMS": platform},
            capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        args = log.read_text().splitlines()
        assert args[:4] == ["-N", "1", "-n", "4"]
        assert args[-5:] == ["-u", "-m", "gw.gw_jax", "-i", "fixture.in"]
        assert str(python) in args
        if site == "frontier":
            assert args[4:6] == ["-c", "7"]
            assert ("--gpus-per-task=1" in args) == (platform != "cpu")
            assert not any(a.startswith(("--gpu-bind", "--tres-bind")) for a in args)
            assert not any("select_gpu" in a for a in args)
        else:
            assert args[4:6] == ["-c", "16"]
            assert "--cpu-bind=threads" in args
            assert "--gpus-per-task=1" not in args
            assert ("--gpus-per-node=4" in args) == (platform != "cpu")
            assert (str(root / "src/ffi/cpp/select_gpu.sh") in args) == (platform != "cpu")
