"""The ROCm leg and its isolated ELPA child share explicit build controls."""
import os
from pathlib import Path
import subprocess


CONTROLS = ("LORRAX_OLCF_VENV", "LORRAX_OLCF_ROCM", "LORRAX_ROCM_BLAS_DIR",
            "LORRAX_ELPA_ROOT", "LORRAX_ELPA_GFX", "LORRAX_ELPA_OPENMP", "LORRAX_ELPA_MAKE_J")


def _fixture(tmp_path):
    root = tmp_path / "checkout with spaces"
    config = root / "config/frontier"
    config.mkdir(parents=True)
    source = Path(__file__).parents[1] / "config/frontier/build_ffi_rocm.sh"
    (config / source.name).write_bytes(source.read_bytes())
    venv = tmp_path / "selected venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text(":\n")
    site_profile = tmp_path / "site profile.sh"
    site_profile.write_text("export MODULEPATH=/site/modules
")
    initialization = tmp_path / "shell_init.sh"
    initialization.write_text('''module() { :; }
cmake() { printf '%s\\n' "$@" > "$PROBE_ROOT/cmake.args"; return 73; }
''')
    # Mock only the dependency builder and CMake; run the actual parent script.
    (config / "build_elpa_hip.sh").write_text('''#!/usr/bin/env bash
set -eu
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == --print-prefix ]]; then
    env -0 > "$root/query.env"
    [[ "${LORRAX_ELPA_MAKE_J:-}" != FAIL_QUERY ]] || exit 17
    echo "$root/dependency prefix"
else
    env -0 > "$root/build.env"
    [[ "${LORRAX_ELPA_MAKE_J:-}" != FAIL_BUILD ]] || exit 19
    mkdir -p "$root/dependency prefix"
    touch "$root/dependency prefix/LORRAX_ELPA_COMPLETE"
fi
''')
    env = dict(os.environ, BASH_ENV=str(initialization), PROBE_ROOT=str(root),
               LORRAX_OLCF_VENV=str(venv), LD_LIBRARY_PATH="/unwanted/library/state",
               MODULEPATH="/unwanted/modules", LOADEDMODULES="unwanted/module",
               LORRAX_OLCF_SYSTEM_PROFILE=str(site_profile),
               ROCM_PATH="/unwanted/toolkit", LORRAX_UNRELATED="must not leak")
    for name in (*CONTROLS[1:], "LORRAX_ELPA_ROCM_DIR"):
        env.pop(name, None)
    return root, config / source.name, env


def _read_env(path):
    return dict(item.split("=", 1) for item in path.read_text().split("\0") if item)


def test_elpa_query_and_build_share_only_existing_explicit_controls(tmp_path):
    for explicit in (False, True):
        case = tmp_path / str(explicit)
        root, script, env = _fixture(case)
        if explicit:
            env.update(LORRAX_OLCF_ROCM="rocm/selected", LORRAX_ROCM_BLAS_DIR="/selected BLAS",
                       LORRAX_ELPA_ROOT="/selected cache", LORRAX_ELPA_GFX="gfx90a",
                       LORRAX_ELPA_OPENMP="0", LORRAX_ELPA_MAKE_J="7")
        result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
        assert result.returncode == 73, result.stdout + result.stderr
        query, build = _read_env(root / "query.env"), _read_env(root / "build.env")
        assert query == build
        for name in CONTROLS:
            if name in env:
                assert query[name] == env[name]
            else:
                assert name not in query
        assert query["PATH"] == "/usr/bin:/bin" and query["HOME"] == env["HOME"]
        # The site MODULEPATH comes from the system profile, never the parent's.
        assert query["MODULEPATH"] == "/site/modules"
        assert not any(name in query for name in ("LD_LIBRARY_PATH", "LOADEDMODULES",
                                                  "LORRAX_OLCF_SYSTEM_PROFILE",
                                                  "ROCM_PATH", "BASH_ENV", "PROBE_ROOT",
                                                  "LORRAX_UNRELATED"))
        assert f"-DLORRAX_ELPA_ROCM_DIR={root}/dependency prefix" in (root / "cmake.args").read_text()


def test_elpa_child_failure_stops_before_native_configure(tmp_path):
    for failure in ("FAIL_QUERY", "FAIL_BUILD"):
        root, script, env = _fixture(tmp_path / failure)
        result = subprocess.run(["bash", str(script)], env={**env, "LORRAX_ELPA_MAKE_J": failure},
                                text=True, capture_output=True)
        assert result.returncode != 0 and not (root / "cmake.args").exists()
        assert (root / "query.env").is_file()
        assert (root / "build.env").exists() == (failure == "FAIL_BUILD")
