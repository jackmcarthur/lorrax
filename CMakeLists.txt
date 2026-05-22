cmake_minimum_required(VERSION 3.22)
project(lorrax VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# GPU-backend toggle (anticipates future ROCm support).
# Only "cuda" is wired today; passing -DLORRAX_GPU_BACKEND=rocm will error
# with a clear message until the ROCm path is implemented.
# ---------------------------------------------------------------------------
set(LORRAX_GPU_BACKEND "cuda" CACHE STRING
    "GPU backend to build the FFI against (cuda | rocm). Only cuda is supported today.")

if(NOT LORRAX_GPU_BACKEND STREQUAL "cuda")
    message(FATAL_ERROR
        "LORRAX_GPU_BACKEND=${LORRAX_GPU_BACKEND} is not yet implemented.\n"
        "Only 'cuda' is supported in this release.  ROCm/HIP support is a\n"
        "planned future refactor (cuSOLVERMp -> hipsolverMp, NCCL -> RCCL).")
endif()

# ---------------------------------------------------------------------------
# Delegate to the real FFI CMakeLists.txt.
# All build logic, dependency detection, and compiler settings live there.
# This top-level wrapper exists solely so that scikit-build-core (and plain
# cmake -S . -B build) can find a CMakeLists.txt at the repo root without
# duplicating any logic.
# ---------------------------------------------------------------------------
add_subdirectory(src/ffi/common/cpp)

# ---------------------------------------------------------------------------
# Install the built .so into the wheel tree so ffi_loader.py can find it.
#
# ffi_loader._candidate_paths() searches (in order):
#   1. $LORRAX_FFI_SO (explicit override)
#   2. src/ffi/common/cpp/build/liblorrax_ffi*.so  (in-tree build)
#   3. sys.path entries (site-packages)             <-- pip-install path
#
# We install into ffi/common/cpp/ inside the wheel so path 3 resolves
# to <site-packages>/ffi/common/cpp/liblorrax_ffi.so, matching the
# glob pattern in _candidate_paths().
# ---------------------------------------------------------------------------
install(TARGETS lorrax_ffi
        LIBRARY DESTINATION ffi/common/cpp)
