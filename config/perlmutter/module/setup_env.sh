#!/usr/bin/env bash
# ============================================================================
# Create the module's runtime venv at $LORRAX_MODULE_PREFIX/venv (or LORRAX_MODULE_VENV).
#
#   LORRAX_MODULE_PREFIX=/abs/prefix bash setup_env.sh <clean lorrax checkout>
#
# Every third-party package, the CUDA wheels included, comes from the
# checkout's uv.lock with the `cuda13` extra a clone syncs (stack.sh
# LOCK_EXTRA): the export is installed --no-deps, so the venv holds exactly
# the clone's versions (cuBLAS, cuFFT, cuSOLVER, cuSPARSE, the CUDA runtime,
# NVRTC, nvJitLink, cuDNN, NCCL, nvidia-mathdx, JAX's cuda13 plugin) and no
# resolver can move one.  No workspace member is installed: the module puts a
# source snapshot on PYTHONPATH instead.  lorrax-launch/libcudart.so links the
# wheel's CUDA runtime under the unversioned name libfabric dlopens (site.toml
# [nccl] launch_libraries); the modulefile puts that directory first on
# LD_LIBRARY_PATH.  The venv is never refreshed in place: a pin change is a
# new prefix.
# ============================================================================
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/stack.sh"

SRC="$(realpath -e "${1:?usage: setup_env.sh <clean lorrax checkout>}")"
lorrax_module_clean_rev "$SRC" >/dev/null
[[ ! -e "$VENV" ]] || lorrax_module_die "venv exists: $VENV (a pin change is a new prefix)"
UV="$(command -v uv || true)"
[[ -n "$UV" ]] || lorrax_module_die "uv is not on PATH"

mkdir -p "$PREFIX"
UV_PYTHON_INSTALL_DIR="$PREFIX/python" \
    "$UV" venv --quiet --python "$PYTHON_VERSION" --python-preference only-managed "$VENV"

REQ="$VENV/lorrax-locked-requirements.txt"
"$UV" export --quiet --project "$SRC" --frozen --no-emit-workspace --no-hashes \
    --extra "$LOCK_EXTRA" --format requirements-txt --output-file "$REQ"
echo "[setup_env] installing $(grep -c '==' "$REQ") locked packages (uv.lock, extra $LOCK_EXTRA)"
"$UV" pip install --quiet --python "$VENV/bin/python" --no-deps -r "$REQ"

SITE="$(lorrax_module_site_packages)"
for p in "$SITE/nvidia/mathdx/include/cufftdx.hpp" "$SITE/nvidia/cudnn/lib" \
         "$SITE/nvidia/nccl/lib/libnccl.so.2" "$SITE/nvidia/nccl/include/nccl.h" \
         "$SITE/nvidia/cu13/lib/libcudart.so.13" "$SITE/nvidia/cu13/lib/libcublas.so.13" \
         "$SITE/nvidia/cu13/lib/libcufft.so.12"; do
    [[ -e "$p" ]] || lorrax_module_die "the wheels did not supply $p"
done
# Every locked package installed at its locked version; the stack pins agree.
"$VENV/bin/python" - "$REQ" "$JAX_VERSION" "$NCCL_VERSION" <<'PY' \
    || lorrax_module_die "the venv does not match uv.lock"
import importlib.metadata as md, re, sys
req, jax_v, nccl_v = sys.argv[1:]
want = {}
for line in open(req):
    m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]*\])?==([^\s;\\]+)", line)
    if m and "sys_platform == 'win32'" not in line:
        want[m.group(1).lower().replace("_", "-")] = m.group(2)
bad = {}
for name, v in want.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        got = None
    if got != v:
        bad[name] = (v, got)
pins = {"jax": jax_v, "nvidia-nccl-cu13": nccl_v}
bad.update({n: (v, want.get(n)) for n, v in pins.items() if want.get(n) != v})
import jax_plugins.xla_cuda13  # noqa: F401  the wheel-CUDA plugin
print("[setup_env] " + " ".join(f"{n}={want[n]}" for n in sorted(want)
                                if n.startswith(("jax", "nvidia-"))))
if bad:
    print(f"[setup_env] REFUSED: locked (want, got): {bad}", file=sys.stderr)
    sys.exit(1)
PY
LAUNCH="$VENV/lorrax-launch"
mkdir -p "$LAUNCH"
ln -sfn "$(realpath -e "$SITE/nvidia/cu13/lib/libcudart.so.13")" "$LAUNCH/libcudart.so"
echo "[setup_env] venv: $VENV"
