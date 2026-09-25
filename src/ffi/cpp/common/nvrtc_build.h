// nvrtc_build.h -- the CUDA leg's one NVRTC build service.
//
// Compiles an embedded CUDA source to a cubin with NVRTC, keeps the image in a disk cache keyed
// by everything that decides it, and loads it through the driver API (resolved lazily: libcuda
// is already mapped by JAX, and build/login nodes have none to link).  Every handler that builds
// kernels at run time (the nvidia-mathdx k-convolution family, the Fourier plan's fused stages)
// goes through here, so there is one compile path, one cache format and one key rule.
//
// Key (FNV-1a, in this order): the source; each embedded header's text ("\x1d" + text); each
// deciding option ("\x1f" + option); each toolchain version file ("\x1e" + basename + content);
// "nvrtc<major>.<minor>@<libnvrtc real path>"; the caller's extra key.  Include paths are not in
// the key (two installs of one version compile one image).  A version file that reads empty is
// reported in `missing`; the caller must then disable its disk cache rather than drop the file
// from the key.
//
// Disk image: "LRXKCONV1\n" + 16 hex key + 16 hex payload hash + '\n' + cubin, written to a
// unique temporary and renamed (atomic on one filesystem, so concurrent ranks cannot tear it),
// re-hashed on read; a torn, foreign or non-ELF file, or one the driver refuses, is deleted,
// recompiled once and replaced.
#pragma once

#include <cuda.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lorrax_ffi::nvrtc {

// Driver entry points resolved lazily.
struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
    bool ok = false;
    std::string err;
};
const DriverApi& driver_api();
std::string cu_err(CUresult r);

struct Program {
    const char* src = nullptr;                                   // the embedded source
    const char* name = "lrx_nvrtc.cu";                           // NVRTC program name
    std::vector<std::pair<const char*, const char*>> headers;    // embedded (include name, text)
    std::vector<std::string> defs;                               // options that decide the image
    std::vector<std::string> includes;                           // -I directories (not keyed)
    std::vector<std::string> version_files;                      // toolchain version headers
    std::string extra_key;                                       // hashed last
    const char* kernel = nullptr;                                // entry point
};

uint64_t fnv1a(std::string_view data, uint64_t h = 1469598103934665603ULL);
std::string hex16(uint64_t v);
bool exists(const std::string& path);
std::string read_file(const std::string& path);
std::string toolkit_include(std::string* why);                   // CUDA include beside libnvrtc

// An nvidia-mathdx program: the wheel at `root` (its nvidia/mathdx directory) and the CUDA
// toolkit include `cuda_inc` as -I paths; as version files `dx`'s own header
// (include/<dx>/<dx>_version.hpp), commonDx's, CUTLASS's and CCCL's; the wheel's dist-info
// name as the extra key.  Every mathdx kernel (cuFFTDx, cuBLASDx) is keyed by this one rule.
void mathdx_toolchain(const std::string& root, const std::string& cuda_inc, const char* dx,
                      Program* p);

// The program's key; `missing` lists version files that read empty.
uint64_t key(const Program& p, std::string* missing);

struct Image {
    CUfunction fn = nullptr;
    double ms = 0.0;
    bool from_disk = false, stored = false, rebuilt_bad = false;
};

// Load `cache_path` (framed with `key_hex`) when it is non-empty and valid, else compile (and
// store it under `cache_dir`); load the module and resolve `p.kernel`.  On failure *where names
// the step and *err the reason.
bool build(const Program& p, const std::string& cache_dir, const std::string& cache_path,
           const std::string& key_hex, Image* out, std::string* where, std::string* err);

}  // namespace lorrax_ffi::nvrtc
