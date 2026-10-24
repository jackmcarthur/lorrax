#!/usr/bin/env bash
# ============================================================================
# build_elpa_hip.sh -- ELPA (AMD GPU kernels, HIP streams, rocSOLVER, RCCL) for
# the ROCm leg on OLCF Frontier (MI250X gfx90a), plus the netlib ScaLAPACK it
# needs, both against the leg's ONE host BLAS: the sequential locking OpenBLAS
# of build_openblas_seq.sh (the leg's libopenblas.so.0).
#
#   config/frontier/build_elpa_hip.sh            # build (or verify) the prefix
#   config/frontier/build_elpa_hip.sh --print-prefix
#
# IMMUTABLE, CONTENT-KEYED PREFIX.  The install directory is
#   $ROOT/elpa-$ELPA_VERSION-$KEY
# where KEY = sha256 over (this script, both source tarball hashes, the
# OpenBLAS file hash, the ROCm module, the GPU targets, the PrgEnv/MPI pins),
# truncated to 12 hex.  A finished prefix carries LORRAX_ELPA_COMPLETE and is
# read-only; a rerun verifies its closure record and exits.  A changed input
# is a new prefix -- nothing is ever rebuilt in place.  The build runs in
# $ROOT/.stage-$KEY and is renamed into place only after every check passes.
#
# ONE ROCm RUNTIME, ONE MPI, ONE BLAS PER PROCESS:
#   - module rocm/7.14.1 == the pip rocm-sdk 7.14.1 wheels JAX maps; ELPA's
#     NEEDED librocblas.so.5 / librocsolver.so.0 / libamdhip64.so.7 /
#     librccl.so.1 bind to the objects JAX already loaded (by SONAME).
#   - PrgEnv-gnu + cray-mpich pins of ffi_mpi.sh; craype-accel-amd-gfx90a
#     UNLOADED (no libmpi_gtl_hsa): ELPA's device-to-device traffic goes over
#     RCCL (--enable-gpu-ccl=rccl), host traffic over plain MPI.
#   - cray-libsci UNLOADED: LibSci beside the leg's OpenBLAS is two BLAS.
#     ScaLAPACK is therefore netlib 2.2.2 built here against that OpenBLAS, as a
#     STATIC PIC archive linked INTO libelpa with --exclude-libs: libelpa imports
#     13 ScaLAPACK names (numroc_, p?lacpy_, p?tran[c]_, p?trmm_; generalized /
#     multiply paths only) and both legs are dlopened RTLD_GLOBAL, so a shared
#     libscalapack in the device leg would interpose on the host leg's LibSci
#     ScaLAPACK/BLACS.  Hidden inside libelpa it is visible to nobody else.
#
# Outputs in the prefix: lib/libelpa[_openmp].so.* (ScaLAPACK inside, hidden),
# lib/libopenblas*.so* (copy of the leg's), include/elpa-*/, and
# LORRAX_ELPA_CLOSURE.json: every file of the prefix's dependency closure by
# size + SHA-256 (private: ELPA/ScaLAPACK/OpenBLAS; pinned: MPI; rocm:
# resolved in the venv's ROCm wheel directories, the copies the process maps).
# ============================================================================
set -euo pipefail
ulimit -c 0
LORRAX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${LORRAX_ELPA_ROOT:-$HOME/software/elpa_builds}"
ELPA_VERSION="2026.02.001"
ELPA_SHA256="a379f27f4dbd27b2ee45017afec656d064301e97150c874649bdfd64957b75ed"
ELPA_URL="https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/$ELPA_VERSION/elpa-$ELPA_VERSION.tar.gz"
SCALAPACK_VERSION="2.2.2"
SCALAPACK_SHA256="a2f0c9180a210bf7ffe126c9cb81099cf337da1a7120ddb4cbe4894eb7b7d022"
SCALAPACK_URL="https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v$SCALAPACK_VERSION.tar.gz"
ROCM_MOD="${LORRAX_OLCF_ROCM:-rocm/7.14.1}"
GFX="${LORRAX_ELPA_GFX:-gfx90a}"
JOBS="${LORRAX_ELPA_MAKE_J:-16}"
OPENMP="${LORRAX_ELPA_OPENMP:-1}"            # host stages (band->tridi) threaded; see the report
# ELPA 2026.02 refuses --enable-gpu-ccl=rccl together with --enable-openmp: one or the other.
if [[ $OPENMP == 1 ]]; then CCL_OPT="--disable-gpu-ccl"; CCL_LIB=""; else CCL_OPT="--enable-gpu-ccl=rccl"; CCL_LIB="-lrccl"; fi
VENV="${LORRAX_OLCF_VENV:-$HOME/envs/lx-jax091-rocm7141}"
BLAS_DIR="${LORRAX_ROCM_BLAS_DIR:-$HOME/software/openblas_seq/install}"
BLAS_LIB="$BLAS_DIR/lib/libopenblas.so"

if ! type module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || source /opt/cray/pe/lmod/lmod/init/bash
fi
export LMOD_PAGER=none
# shellcheck disable=SC1091
source "$LORRAX_ROOT/config/frontier/ffi_mpi.sh"
[[ -f "$BLAS_LIB" ]] || { echo "[elpa] ERROR: no $BLAS_LIB (run config/frontier/build_openblas_seq.sh)" >&2; exit 2; }
grep -q 'USE_THREAD=0' "$BLAS_DIR/LORRAX_BUILD_FLAGS" && grep -q 'USE_LOCKING=1' "$BLAS_DIR/LORRAX_BUILD_FLAGS" || {
    echo "[elpa] ERROR: $BLAS_DIR is not the sequential LOCKING OpenBLAS" >&2; exit 2; }

sha() { sha256sum "$1" | cut -d' ' -f1; }
KEY=$( { cat "${BASH_SOURCE[0]}"; echo "$ELPA_SHA256 $SCALAPACK_SHA256 $(sha "$(readlink -f "$BLAS_LIB")")";
         echo "openmp=$OPENMP $ROCM_MOD $GFX $LORRAX_OLCF_PRGENV_MODULE $LORRAX_OLCF_MPICH_MODULE $LORRAX_OLCF_MPICH_ROOT"; } \
       | sha256sum | cut -c1-12)
PREFIX="$ROOT/elpa-$ELPA_VERSION-$KEY"
if [[ "${1:-}" == "--print-prefix" ]]; then echo "$PREFIX"; exit 0; fi

verify() {   # re-hash every recorded private file of a finished prefix
    "$VENV/bin/python" - "$1" <<'PY'
import hashlib, json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads((p / "LORRAX_ELPA_CLOSURE.json").read_text())
bad = [e["path"] for e in d["libraries"]
       if hashlib.sha256(pathlib.Path(e["path"]).read_bytes()).hexdigest() != e["sha256"]]
if bad: sys.exit(f"[elpa] closure MISMATCH: {bad}")
print(f"[elpa] closure verified: {len(d['libraries'])} files, record {d['record_sha256'][:16]}")
PY
}
if [[ -e "$PREFIX/LORRAX_ELPA_COMPLETE" ]]; then
    echo "[elpa] prefix exists (immutable): $PREFIX"; verify "$PREFIX"; exit 0; fi
[[ -e "$PREFIX" ]] && { echo "[elpa] ERROR: $PREFIX exists but is incomplete; remove it by hand" >&2; exit 2; }

module load "$LORRAX_OLCF_PRGENV_MODULE"
lorrax_olcf_pin_mpi
module load "$LORRAX_OLCF_CMAKE_MODULE" "$ROCM_MOD"
module unload craype-accel-amd-gfx90a cray-libsci 2>/dev/null || true
[[ "$CRAY_MPICH_DIR" == "$LORRAX_OLCF_MPICH_ROOT" ]] || { echo "[elpa] ERROR: MPICH=$CRAY_MPICH_DIR, not the pin" >&2; exit 2; }
: "${ROCM_PATH:?module $ROCM_MOD did not set ROCM_PATH}"
echo "[elpa] modules: $(module -t list 2>&1 | tr '\n' ' ')"
echo "[elpa] key=$KEY prefix=$PREFIX"

STAGE="$ROOT/.stage-$KEY"; SRC="$ROOT/src"
rm -rf "$STAGE"; mkdir -p "$STAGE/prefix/lib" "$SRC"
fetch() {  # url file sha
    [[ -f "$SRC/$2" ]] || curl -sSfL --max-time 600 -o "$SRC/$2" "$1"
    [[ "$(sha "$SRC/$2")" == "$3" ]] || { echo "[elpa] ERROR: $2 sha256 mismatch" >&2; exit 2; }
}
fetch "$ELPA_URL" "elpa-$ELPA_VERSION.tar.gz" "$ELPA_SHA256"
fetch "$SCALAPACK_URL" "scalapack-$SCALAPACK_VERSION.tar.gz" "$SCALAPACK_SHA256"
P="$STAGE/prefix"   # configured with the FINAL prefix; installed via DESTDIR-free staging below

# --- the leg's OpenBLAS, copied in (closure binds it by RPATH) --------------
cp -a "$BLAS_DIR"/lib/libopenblas*.so* "$P/lib/"
cp "$BLAS_DIR/LORRAX_BUILD_FLAGS" "$P/lib/libopenblas.LORRAX_BUILD_FLAGS"

# --- netlib ScaLAPACK against it --------------------------------------------
tar -C "$STAGE" -xzf "$SRC/scalapack-$SCALAPACK_VERSION.tar.gz"
cmake -S "$STAGE/scalapack-$SCALAPACK_VERSION" -B "$STAGE/scalapack-build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_TESTING=OFF \
    -DCMAKE_C_COMPILER=cc -DCMAKE_Fortran_COMPILER=ftn \
    -DCMAKE_C_FLAGS="-O2 -Wno-implicit-function-declaration" \
    -DCMAKE_Fortran_FLAGS="-O2 -fallow-argument-mismatch" \
    -DBLAS_LIBRARIES="$BLAS_LIB" -DLAPACK_LIBRARIES="$BLAS_LIB" \
    -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_INSTALL_LIBDIR=lib \
    > "$STAGE/scalapack.configure.log" 2>&1 || { tail -30 "$STAGE/scalapack.configure.log"; exit 1; }
cmake --build "$STAGE/scalapack-build" --parallel "$JOBS" > "$STAGE/scalapack.build.log" 2>&1 \
    || { tail -30 "$STAGE/scalapack.build.log"; exit 1; }
cmake --install "$STAGE/scalapack-build" > /dev/null
echo "[elpa] scalapack $SCALAPACK_VERSION built"

# --- ELPA ---------------------------------------------------------------------
tar -C "$STAGE" -xzf "$SRC/elpa-$ELPA_VERSION.tar.gz"
# --- LORRAX patch: ELPA's device allocations through a host-settable allocator --
# Every ELPA device buffer is taken by hipMallocFromC / released by hipFreeFromC
# (src/GPU/ROCm/rocmFunctions_template.h, one TU).  The patch adds an exported
#   void lorrax_elpa_set_device_allocator(alloc, free, ctx)
# and routes both wrappers through it when set, so the ROCm leg serves ELPA from
# the FFI workspace (an XLA buffer, inside the arena) -- the pool-seed
# idea.  Unset (NULL) = stock ELPA.  The free callback must keep hipFree's
# implicit device synchronization (ELPA relies on it before reuse).
"$VENV/bin/python" - "$STAGE/elpa-$ELPA_VERSION/src/GPU/ROCm/rocmFunctions_template.h" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
m = "int hipMallocFromC(intptr_t *a, size_t width_height) {\n"
f = "int hipFreeFromC(intptr_t *a) {\n"
assert s.count(m) == 1 and s.count(f) == 1, "ELPA source changed: allocator anchors not unique"
hook = (
  "/* LORRAX patch (config/frontier/build_elpa_hip.sh): host-settable device allocator. */\n"
  "typedef int (*lorrax_elpa_alloc_fn)(void **ptr, size_t bytes, void *ctx);\n"
  "typedef int (*lorrax_elpa_free_fn)(void *ptr, void *ctx);\n"
  "static lorrax_elpa_alloc_fn lorrax_elpa_alloc_ = NULL;\n"
  "static lorrax_elpa_free_fn  lorrax_elpa_free_  = NULL;\n"
  "static void *lorrax_elpa_ctx_ = NULL;\n"
  "__attribute__((visibility(\"default\"))) void lorrax_elpa_set_device_allocator(\n"
  "    lorrax_elpa_alloc_fn a, lorrax_elpa_free_fn f, void *ctx) {\n"
  "  lorrax_elpa_alloc_ = a; lorrax_elpa_free_ = f; lorrax_elpa_ctx_ = ctx;\n"
  "}\n\n")
s = s.replace(m, hook + m + "  if (lorrax_elpa_alloc_) return lorrax_elpa_alloc_((void **) a, width_height, lorrax_elpa_ctx_);\n")
s = s.replace(f, f + "  if (lorrax_elpa_free_) return lorrax_elpa_free_((void *) a, lorrax_elpa_ctx_);\n")
p.write_text(s)
print("[elpa] patched", p.name, "(lorrax_elpa_set_device_allocator)")
PY
# --- LORRAX patch 2: rocBLAS/rocSOLVER status accounting and the handle registry --
# ELPA's GPU wrappers (src/GPU/ROCm/rocmFunctions.hip, one TU with both templates)
# discard most rocBLAS statuses (the gemm wrappers assign and drop them) and only print
# rocSOLVER failures.  The ROCm leg gives every ELPA rocBLAS handle a user-owned
# workspace inside the FFI workspace (rocblas_set_workspace); a too-small workspace
# makes rocBLAS/rocSOLVER return rocblas_status_memory_error WITHOUT computing, which
# stock ELPA would ignore.  The patch wraps every BLAS_<op> call and the direct rocSOLVER
# calls (stedc, trtri, syevd) in a status counter, counts every errormessage, and
# records each rocBLAS handle ELPA creates (kind: BLAS or solver, creation order):
#   void lorrax_elpa_gpu_status(long *failures, long *degraded, int *last_status)
#   int  lorrax_elpa_rocblas_handles(void **handles, int *kinds, int cap)   (-> count)
# The handler refuses a solve during which the failure count moved.
"$VENV/bin/python" - "$STAGE/elpa-$ELPA_VERSION/src/GPU/ROCm" <<'PY'
import re, sys, pathlib
d = pathlib.Path(sys.argv[1])
p = d / "rocmFunctions.hip"; s = p.read_text()
em = '#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)\n'
assert s.count(em) == 1, "ELPA source changed: errormessage anchor"
fn = re.compile(r"^(?:rocblas|rocsolver)_(?:create_handle|destroy_handle|set_stream|set_pointer_mode|"
                r"get_pointer_mode|[sdcz](?:trsm|trmm|trmv|syrk|herk|copy|gemm|gemv|potrf|dot|dotc|dotu|scal|axpy))$")
macros = [(a, b) for a, b in re.findall(r"^#define (BLAS_\w+) ((?:rocblas|rocsolver)_\w+)\s*$", s, re.M) if fn.match(b)]
names = {a for a, _ in macros}
assert len(macros) >= 40 and {"BLAS_zgemm", "BLAS_dgemm", "BLAS_create_handle", "BLAS_ztrmm"} <= names, sorted(names)
direct = ["rocsolver_dstedc", "rocsolver_sstedc", "rocsolver_dtrtri", "rocsolver_strtri", "rocsolver_ztrtri",
          "rocsolver_ctrtri", "rocsolver_dsyevd", "rocsolver_ssyevd"]
blk = ["", "/* LORRAX patch 2 (config/frontier/build_elpa_hip.sh): status accounting + handle registry. */",
       "#ifndef HIPBLAS", "#define LORRAX_ELPA_PATCH2 1", "#include <atomic>", "#include <mutex>",
       "static std::atomic<long> lorrax_fail_{0}, lorrax_degraded_{0};",
       "static std::atomic<int> lorrax_last_{0};",
       "static std::mutex lorrax_mu_;",
       "static rocblas_handle lorrax_h_[512]; static int lorrax_k_[512]; static int lorrax_nh_ = 0;",
       "static thread_local int lorrax_kind_ = 0;   /* 1 inside rocsolverCreateFromC */",
       "static inline rocblas_status lorrax_chk_(rocblas_status st) {",
       "  if (st == rocblas_status_perf_degraded) lorrax_degraded_.fetch_add(1);",
       "  else if (st != rocblas_status_success) { lorrax_fail_.fetch_add(1); lorrax_last_.store((int) st); }",
       "  return st;", "}",
       "static rocblas_status lorrax_create_(rocblas_handle *h) {",
       "  rocblas_status st = rocblas_create_handle(h);",
       "  if (st == rocblas_status_success) { std::lock_guard<std::mutex> g(lorrax_mu_);",
       "    if (lorrax_nh_ < 512) { lorrax_h_[lorrax_nh_] = *h; lorrax_k_[lorrax_nh_] = lorrax_kind_; ++lorrax_nh_; }",
       "    else lorrax_fail_.fetch_add(1); }",
       "  return lorrax_chk_(st);", "}",
       "static rocblas_status lorrax_destroy_(rocblas_handle h) {",
       "  { std::lock_guard<std::mutex> g(lorrax_mu_);",
       "    for (int i = 0; i < lorrax_nh_; ++i) if (lorrax_h_[i] == h) {",
       "      for (int j = i + 1; j < lorrax_nh_; ++j) { lorrax_h_[j - 1] = lorrax_h_[j]; lorrax_k_[j - 1] = lorrax_k_[j]; }",
       "      --lorrax_nh_; break; } }",
       "  return lorrax_chk_(rocblas_destroy_handle(h));", "}",
       'extern "C" __attribute__((visibility("default"))) void lorrax_elpa_gpu_status(long *f, long *dg, int *last) {',
       "  if (f) *f = lorrax_fail_.load(); if (dg) *dg = lorrax_degraded_.load(); if (last) *last = lorrax_last_.load();", "}",
       'extern "C" __attribute__((visibility("default"))) int lorrax_elpa_rocblas_handles(void **hs, int *kinds, int cap) {',
       "  std::lock_guard<std::mutex> g(lorrax_mu_);",
       "  for (int i = 0; i < lorrax_nh_ && i < cap; ++i) { if (hs) hs[i] = (void *) lorrax_h_[i]; if (kinds) kinds[i] = lorrax_k_[i]; }",
       "  return lorrax_nh_;", "}",
       "#undef errormessage",
       '#define errormessage(x, ...) do { lorrax_fail_.fetch_add(1); fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)']
for a, b in macros:
    blk.append(f"#undef {a}")
    if a == "BLAS_create_handle":   blk.append(f"#define {a}(h) lorrax_create_(h)")
    elif a == "BLAS_destroy_handle": blk.append(f"#define {a}(h) lorrax_destroy_(h)")
    else:                           blk.append(f"#define {a}(...) lorrax_chk_({b}(__VA_ARGS__))")
for b in direct:
    blk.append(f"#define {b}(...) lorrax_chk_({b}(__VA_ARGS__))")
blk += ["#endif /* !HIPBLAS */", ""]
s = s.replace(em, em + "\n".join(blk) + "\n")
p.write_text(s)
q = d / "rocsolverFunctions_template.h"; t = q.read_text()
rc = "  return rocblasCreateFromC(gpusolver_handle);\n"
assert t.count(rc) == 1, "ELPA source changed: rocsolverCreateFromC anchor"
# (test/shared/GPU/ROCm/test_rocmFunctions includes this template without patch 2's block)
t = t.replace(rc, "#ifdef LORRAX_ELPA_PATCH2\n"
                  "  { lorrax_kind_ = 1; int lorrax_r_ = rocblasCreateFromC(gpusolver_handle); lorrax_kind_ = 0; return lorrax_r_; }\n"
                  "#else\n" + rc + "#endif\n")
q.write_text(t)
print("[elpa] patched", p.name, f"({len(macros)} BLAS_ wrappers, {len(direct)} rocSOLVER wrappers,",
      "lorrax_elpa_gpu_status, lorrax_elpa_rocblas_handles)")
PY
EB="$STAGE/elpa-build"; mkdir -p "$EB"
# ELPA's test helpers include the pre-5.x flat names (rocsolver.h, rocblas.h).
RINC="-I$ROCM_PATH/include -I$ROCM_PATH/include/rocsolver -I$ROCM_PATH/include/rocblas"
HIPFLAGS="--offload-arch=$GFX -O3 -std=c++17 -D__HIP_PLATFORM_AMD__ $RINC"
( cd "$EB" && "$STAGE/elpa-$ELPA_VERSION/configure" \
    --prefix="$PREFIX" \
    FC=ftn CC=cc CXX=CC HIPCC="$ROCM_PATH/bin/hipcc" \
    FCFLAGS="-O3 -march=znver3 -fallow-argument-mismatch -ffree-line-length-none" \
    CFLAGS="-O3 -march=znver3 -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include" \
    CXXFLAGS="-O3 -march=znver3 -std=c++17 -D__HIP_PLATFORM_AMD__ -I$ROCM_PATH/include" \
    HIPCCFLAGS="$HIPFLAGS" \
    CPPFLAGS="-D__HIP_PLATFORM_AMD__ $RINC" \
    LDFLAGS="-L$P/lib -L$ROCM_PATH/lib -Wl,-rpath,$PREFIX/lib" \
    LIBS="-lrocsolver -lrocblas -lamdhip64 $CCL_LIB" \
    SCALAPACK_LDFLAGS="$P/lib/libscalapack.a -L$P/lib -lopenblas -Wl,--exclude-libs,libscalapack.a" \
    SCALAPACK_FCFLAGS="-L$P/lib" \
    --with-mpi=yes $([[ $OPENMP == 1 ]] && echo --enable-openmp --enable-runtime-threading-support-checks --enable-allow-thread-limiting --without-threading-support-check-during-build || echo --disable-openmp) \
    --enable-amd-gpu-kernels --enable-gpu-streams=amd --with-rocsolver \
    --enable-hipcub $CCL_OPT \
    --disable-avx512-kernels --disable-sse-assembly-kernels \
    --disable-cpp-tests --disable-Fortran-tests --enable-c-tests \
    --disable-autotuning --disable-doxygen-doc \
    --enable-shared --disable-static \
    > "$STAGE/elpa.configure.log" 2>&1 ) || { tail -40 "$STAGE/elpa.configure.log"; exit 1; }
make -C "$EB" -j "$JOBS" > "$STAGE/elpa.build.log" 2>&1 || { grep -n -m5 -i 'error' "$STAGE/elpa.build.log"; tail -30 "$STAGE/elpa.build.log"; exit 1; }
make -C "$EB" install prefix="$P" > "$STAGE/elpa.install.log" 2>&1 || { tail -30 "$STAGE/elpa.install.log"; exit 1; }
echo "[elpa] elpa $ELPA_VERSION built"

# --- link-time checks: one ROCm, one MPI, one BLAS ---------------------------
need() { readelf -d "$1" | grep NEEDED | sed 's/.*\[//;s/\]//'; }
ELIB=$(ls "$P"/lib/libelpa*.so.* | grep -v -- -gdb | head -1)   # libelpa_openmp.so.* when OPENMP=1
echo "[elpa] $(basename "$ELIB") NEEDED: $(need "$ELIB" | tr '\n' ' ')"
fail=0
for s in librocblas.so.5 librocsolver.so.0 libamdhip64.so.7 ${CCL_LIB:+librccl.so.1} libopenblas.so.0; do
    need "$ELIB" | grep -qx "$s" || need "$ELIB" | grep -q "^${s%%.so*}" || { echo "[elpa] ERROR: libelpa lacks $s" >&2; fail=1; }; done
for f in "$ELIB"; do
    if need "$f" | grep -qE 'cudart|cublas|libmpi_gtl|libsci'; then echo "[elpa] ERROR: $(basename "$f") links CUDA/GTL/LibSci" >&2; fail=1; fi
    # the pinned cray-mpich: its C library and (ELPA is Fortran) its Fortran binding
    if need "$f" | grep -E '^libmpi' | grep -vx "$LORRAX_OLCF_MPI_SONAME" | grep -qvx "${LORRAX_OLCF_MPI_SONAME/libmpi_/libmpifort_}"; then
        echo "[elpa] ERROR: $(basename "$f") links another MPI" >&2; fail=1; fi
    if need "$f" | grep -E '^libamdhip64' | grep -qvx 'libamdhip64.so.7'; then echo "[elpa] ERROR: $(basename "$f") links another HIP" >&2; fail=1; fi
done
need "$ELIB" | grep -q '^libscalapack' && { echo "[elpa] ERROR: libelpa NEEDs a shared ScaLAPACK" >&2; fail=1; }
nm -D --defined-only "$ELIB" > "$STAGE/elpa.nm"   # (no grep -q on the pipe: pipefail + SIGPIPE)
grep -q " T lorrax_elpa_set_device_allocator$" "$STAGE/elpa.nm" || { echo "[elpa] ERROR: allocator hook not exported" >&2; fail=1; }
for s in lorrax_elpa_gpu_status lorrax_elpa_rocblas_handles; do
    grep -q " T $s$" "$STAGE/elpa.nm" || { echo "[elpa] ERROR: $s not exported (LORRAX patch 2)" >&2; fail=1; }; done
# ScaLAPACK/BLACS stay private to libelpa (see the header)
if grep -E " [TDBW] (p[sdcz][a-z0-9]+_|numroc_|descinit_|indxg2p_|[Cc]?blacs_[a-z_]*|Cdgesd2d|Csys2blacs_handle)$" "$STAGE/elpa.nm" | head -3 | grep -q .; then
    echo "[elpa] ERROR: libelpa exports ScaLAPACK/BLACS symbols:" >&2; grep -E " [TDBW] (p[sdcz][a-z0-9]+_|numroc_|blacs_)" "$STAGE/elpa.nm" | head -5 >&2; fail=1; fi
nm -D --undefined-only "$ELIB" | awk '{print $2}' > "$STAGE/elpa.und"
if grep -qE "^(p[sdcz](lacpy|tran|tranc|trmm)_|numroc_)$" "$STAGE/elpa.und"; then echo "[elpa] ERROR: libelpa still imports ScaLAPACK" >&2; fail=1; fi
[[ $fail == 0 ]] || exit 1

# --- closure record -----------------------------------------------------------
"$VENV/bin/python" - "$P" "$PREFIX" "$VENV" "$LORRAX_OLCF_RUNTIME_LIB_DIRS" "$KEY" <<'PY'
import hashlib, json, os, subprocess, sys, sysconfig, time, pathlib, glob
stage, final, venv, pinned, key = sys.argv[1:6]
st = pathlib.Path(stage).resolve()                                 # realpaths throughout
final = str(pathlib.Path(final).parent.resolve() / pathlib.Path(final).name)
def h(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def needed(p):
    out = subprocess.run(["readelf", "-d", str(p)], text=True, capture_output=True).stdout
    return [l.split("[", 1)[1].rstrip("]") for l in out.splitlines() if "(NEEDED)" in l]
site = subprocess.run([f"{venv}/bin/python", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                      text=True, capture_output=True).stdout.strip()
rocm_dirs = sorted({str(pathlib.Path(p).parent) for p in glob.glob(f"{site}/_rocm_sdk_*/**/lib*.so*", recursive=True)})
dirs = [("private", str(st / "lib"))] + [("pinned", d) for d in pinned.split(":") if d] + [("rocm", d) for d in rocm_dirs]
libs, ext, seen = [], [], set()
todo = [str(p) for p in sorted((st / "lib").glob("libelpa*.so.*")) if not p.is_symlink()]
while todo:
    f = todo.pop()
    real = os.path.realpath(f)
    if real in seen: continue
    seen.add(real)
    for n in needed(real):
        for role, d in dirs:
            c = os.path.join(d, n)
            if os.path.exists(c):
                todo.append(c); break
        else:
            ext.append(n)
    role = next((r for r, d in dirs if real.startswith(os.path.realpath(d) + "/")), "private")
    path = real.replace(str(st), final) if real.startswith(str(st)) else real
    libs.append(dict(file=os.path.basename(f), path=path, role=role, bytes=os.path.getsize(real), sha256=h(real),
                     needed=needed(real)))
doc = dict(schema="lorrax.elpa_closure/1", key=key, prefix=final, libraries=sorted(libs, key=lambda e: e["path"]),
           external=sorted(set(ext)), built_on=os.uname().nodename, built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
doc["record_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
(st / "LORRAX_ELPA_CLOSURE.json").write_text(json.dumps(doc, indent=1))
miss = [n for n in doc["external"] if n.startswith(("librocblas", "librocsolver", "libamdhip64", "librccl", "libmpi", "libopenblas", "libscalapack", "libsci"))]
if miss: sys.exit(f"[elpa] closure REFUSED: {miss} resolve from no recorded directory")
print(f"[elpa] closure: {len(libs)} files ({sum(e['role']=='private' for e in libs)} private), external: {doc['external']}")
PY
cp "${BASH_SOURCE[0]}" "$P/LORRAX_BUILD_SCRIPT.sh"
cp "$EB/config.log" "$P/elpa.config.log" 2>/dev/null || true
date -u +%FT%TZ > "$P/LORRAX_ELPA_COMPLETE"
mv "$P" "$PREFIX"
chmod -R a-w "$PREFIX"
rm -rf "$STAGE"
echo "[elpa] done: $PREFIX"
verify "$PREFIX"
# The closure in an empty environment (env -i must bind the private OpenBLAS copy through the
# library's own search path), ldd -r here, and a 64x64 1-stage eigensolve against LAPACK
# (scripts/dep_check.py elpa; SKIPPED where MPI cannot start, then verify.sh runs it).
"$VENV/bin/python" "$LORRAX_ROOT/scripts/dep_check.py" elpa "$PREFIX" \
        --blas "$PREFIX/lib/libopenblas.so.0" --solve || {
    _bad="$PREFIX.REFUSED.$(date +%s)"; mv "$PREFIX" "$_bad"
    echo "[elpa] FAILED its smoke test (above); moved aside to $_bad, nothing will use it" >&2; exit 1; }
