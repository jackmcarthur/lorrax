// fft_flat_k_cuda_ffi.cc — flat-k batched 3-D cuFFT handlers, CUDA platform
// (JAX GPU backend).  The platform mirror of cpp/mklfft/fft_flat_k_ffi.cc
// (FFT-FFI workstream, 2026-07-29) — gated behind LORRAX_FFT_FFI /
// LORRAX_FFT_FFI_FUSED on the Python side (common/fft_helpers.py); the
// default XLA path is untouched.
//
// WHY IT EXISTS: same anchor as the host handler — XLA's fft custom-call
// requires the transformed axes minor-most, while the Σ τ-kernel holds its
// tiles in flat-k "dot layout" (k-major: (nk, s, μ_X, s', μ_Y)), so XLA
// MATERIALIZES THE WHOLE TILE IN A DIFFERENT LAYOUT before AND after every
// fft.  Say "layout materialisation", not "transpose": XLA does not emit a
// `transpose` opcode for this.  It emits a kLoop `fusion` whose body holds
// the transpose+copy, e.g. (after_optimizations HLO of the flat-k ifft,
// wk_REL/results/hlo/fftlayout_hlo/F_flatk_control.hlo.txt):
//
//   %transpose_copy_fusion = c128[1,312,1,312,4,4,1] fusion(%g.1),
//       kind=kLoop, calls=%fused_computation.1,
//       metadata={op_name="jit(flatk_ifft)/transpose"}
//   %fft.0 = ... fft(%transpose_copy_fusion), fft_type=IFFT
//   ROOT %bitcast_copy_fusion = c128[16,1,312,1,312] fusion(%fft.0), ...
//
// The cost is real and is the entire reason this handler exists — the tile
// is written out twice — but it is invisible to any census that keys on
// `opcode in (transpose, copy)`.  On the GPU census (job log
// wk_REL/results/logs/audit_gpu_hlo.log, jax 0.9.1, CudaDevice) the four
// flat-k cases report `transpose ops: 0  copy ops: 0  fusion ops: 1` and the
// minor-most control reports `fusion ops: 0` — same verdict line, opposite
// structure.  If you are re-measuring this, count fusions.  cuFFT's
// ADVANCED DATA LAYOUT
// (cufftPlanMany64 inembed/onembed/istride/idist) is the exact analog of the
// MKL DFTI stride descriptors: element address for FFT index (x, y, z) of
// batch b is  b·idist + ((x·inembed[1] + y)·inembed[2] + z)·istride,  so with
//     inembed = onembed = {d0, d1, d2},  istride = ostride = T,
//     idist   = odist   = 1,             batch   = T
// the plan reads the (nk, *trail) dot-layout tile exactly where it lies —
// FFT-axis element strides {d1·d2·T, d2·T, T}, batch of T transforms at
// DISTANCE 1 along the unit-stride trail (one-to-one because batch <= T; the
// same "transform along a non-minor axis" layout as the host handler).
// Consecutive CUDA threads then touch consecutive trail elements — coalesced
// global loads — so no compact-chunk staging is needed here: the host
// engine's per-thread L2 buffer was a CLX cache artifact, not part of the
// contract.
//
// Handlers (registered by ffi_loader.py under platform="CUDA", SAME target
// strings as the host table so every ffi_call site resolves per lowering
// platform — the phdf5/platform_seam.h registration split):
//   CufftFlatKCudaFfi  (target lorrax_mklfft_flat_k)
//       X (nk, *trail) c128 -> Y same shape.  One batched 3-D FFT over the
//       LEADING flat-k axis; direction + total scale are attributes computed
//       by the Python helper to match jnp.fft's norm conventions EXACTLY
//       (cuFFT scales neither direction, so the scale is applied by a tiny
//       elementwise kernel; skipped when scale == 1).
//   (The fused Sigma convolution this TU used to carry as CufftGwConvCudaFfi
//   moved to the k-convolution router's nvidia-mathdx family, 2026-09-24:
//   cpp/cufft/kconv_mathdx_cuda_ffi.cc modes 2 and 3.)
//
// DEVICE CODE WITHOUT NVCC: this TU is compiled by g++ against the CUDA
// headers — the Frontera pip toolchain ships ptxas but NO nvcc driver, and
// the CUDA .so build deliberately has no CUDA-language step (house fact,
// SPEEDUP_SCORECARD.md AE.4b).  The scale kernel therefore lives in an NVRTC
// source string compiled ONCE per process at first use, for the compute
// capability queried from the runtime device (also sidesteps the
// CMAKE_CUDA_ARCHITECTURES=80 default vs rtx-dev sm_75 mismatch), loaded
// through the driver API resolved with dlsym (libcuda is already in the
// process — JAX loaded it; we add no link-time libcuda dependency, the
// same dlsym pattern as blacs_grid.h's MKL pin).
//
// In-place: the Python wrappers alias operand 0 to the result
// (input_output_aliases={0:0}).  When XLA grants the alias the exec sees
// idata == odata and cuFFT runs in place (legal: identical advanced
// layouts); otherwise it runs out-of-place into the distinct result buffer
// (cuFFT never writes the input of an out-of-place C2C).
//
// Memory policy (scaling target: no hidden N_mu^2 allocations): plans are
// created with auto-allocation OFF and share ONE grow-only device workspace
// arena sized to the largest cufftMakePlanMany64 request.  It is cudaMalloc'd outside the XLA allocator — safe
// under the production env (ffi_env.sh: cuda_async allocator + preallocate
// false) and logged under LORRAX_DEBUG_PRINT, same class as the host
// handler's malloc'd V_R arena.  Arena growth cudaDeviceSynchronize()s
// first so no enqueued work can still reference the old allocation.
//
// Concurrency: one process mutex serializes plan-cache access, arena growth
// and ENQUEUE of each handler's work.  Device-side ordering relies on all
// calls arriving on one compute stream (XLA:GPU dispatches a module's
// custom-calls on its main stream); two concurrent streams would race on
// the shared arenas — same single-consumer assumption as the host arena,
// stated here honestly.
//
// Envelope-honesty: every extent and stride comes from the runtime buffer
// dimensions / attributes; nothing is specialized to a deck.

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include "../common/mkl_thread_pin.h"   // rank-scoped log gate only

#include <cuda.h>            // driver-API types only; entry points via dlsym
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvrtc.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::cufft_flat_k {

namespace ffi = ::xla::ffi;
using C128 = std::complex<double>;

// Native FFT detail follows the one rank-zero driver debug stream.
static bool log_enabled() {
    static const bool on = mklpin::debug_print_here();
    return on;
}

// Host-only tuning knobs that are INERT on this platform (audit P1.11).
// LORRAX_FFT_FFI_THREADS / LORRAX_FFT_FFI_CHUNK (and their deprecated
// spellings) tune the MKL DFTI handler's OpenMP chunk loop; the cuFFT
// backend has no chunk loop, so under the announce-or-refuse doctrine a
// set-but-unread knob must SAY it is inert rather than vanish.  Announced
// once per process, rank-scoped, at the first handler call.
static void announce_inert_host_knobs() {
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return;
    if (!mklpin::announce_here()) return;
    for (const char* name :
         {"LORRAX_FFT_FFI_THREADS", "LORRAX_MKLFFT_THREADS",
          "LORRAX_FFT_FFI_CHUNK", "LORRAX_MKLFFT_CHUNK"}) {
        if (std::getenv(name) != nullptr) {
            std::fprintf(stderr,
                         "[cufft_flat_k] %s is set but tunes the HOST (MKL "
                         "DFTI) flat-k handler's OpenMP chunk loop only; the "
                         "cuFFT backend has no chunk loop and the knob is "
                         "INERT on this platform (announced, not silently "
                         "dropped).\n",
                         name);
        }
    }
}

static const char* cufft_err_name(cufftResult r) {
    switch (r) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
#if CUFFT_VERSION < 12000
        // Removed from the public result enum in cuFFT 12 (CUDA 13).
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";
#endif
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "CUFFT_<unknown>";
    }
}

static ffi::Error fail(const char* where, const std::string& detail) {
    std::ostringstream os;
    os << "cufft_flat_k (cuFFT strided flat-k CUDA FFI): " << where
       << " failed — " << detail;
    return ffi::Error(ffi::ErrorCode::kInternal, os.str());
}

#define LRX_CUFFT_CHECK(expr, where)                                     \
    do {                                                                 \
        cufftResult _r = (expr);                                         \
        if (_r != CUFFT_SUCCESS) {                                       \
            return fail((where), cufft_err_name(_r));                    \
        }                                                                \
    } while (0)

#define LRX_CUDA_CHECK(expr, where)                                      \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            return fail((where), cudaGetErrorString(_e));                \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------------
//  Driver-API entry points via dlsym (no link-time libcuda: the driver is a
//  runtime-node library, absent on build/login nodes; JAX has already
//  dlopen'd it in any process that reaches these handlers).
// ---------------------------------------------------------------------------
struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*ModuleUnload)(CUmodule) = nullptr;
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    bool ok = false;
    std::string err;   // WHY resolution failed (dlerror text), for the
                       // first-use refusal — was swallowed pre-P1.9.
};

static const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        void* h = RTLD_DEFAULT;
        dlerror();  // clear any stale error state
        if (dlsym(h, "cuLaunchKernel") == nullptr) {
            h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (h == nullptr) {
                const char* e = dlerror();
                a.err = std::string("dlopen(libcuda.so.1): ") +
                        (e ? e : "(dlerror returned no detail)");
                return a;  // ok=false; err reported at first use
            }
        }
        auto need = [&](const char* name) -> void* {
            dlerror();
            void* p = dlsym(h, name);
            if (p == nullptr) {
                // COMPLETE record (2026-08-01 seam-audit leftover): the old
                // `&& a.err.empty()` guard kept only the FIRST missing
                // symbol, so a driver missing several entry points produced
                // a refusal naming one, and each fix round-tripped through
                // another failing run.  Append every miss.
                const char* e = dlerror();
                if (!a.err.empty()) a.err += "; ";
                a.err += std::string("dlsym(") + name + "): " +
                         (e ? e : "symbol not found (no dlerror detail)");
            }
            return p;
        };
        a.ModuleLoadData = reinterpret_cast<decltype(a.ModuleLoadData)>(
            need("cuModuleLoadData"));
        a.ModuleGetFunction = reinterpret_cast<decltype(a.ModuleGetFunction)>(
            need("cuModuleGetFunction"));
        a.ModuleUnload = reinterpret_cast<decltype(a.ModuleUnload)>(
            need("cuModuleUnload"));
        a.LaunchKernel = reinterpret_cast<decltype(a.LaunchKernel)>(
            need("cuLaunchKernel"));
        a.CtxGetCurrent = reinterpret_cast<decltype(a.CtxGetCurrent)>(
            need("cuCtxGetCurrent"));
        a.GetErrorString = reinterpret_cast<decltype(a.GetErrorString)>(
            need("cuGetErrorString"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.ModuleUnload &&
               a.LaunchKernel && a.CtxGetCurrent && a.GetErrorString;
        return a;
    }();
    return api;
}

static std::string cu_err(CUresult r) {
    const DriverApi& api = driver_api();
    const char* s = nullptr;
    if (api.GetErrorString && api.GetErrorString(r, &s) == CUDA_SUCCESS && s) {
        return s;
    }
    std::ostringstream os;
    os << "CUresult=" << static_cast<int>(r);
    return os.str();
}

// ---------------------------------------------------------------------------
//  The device scale kernel, NVRTC-compiled at first use (see file header for
//  why there is no nvcc).  Raw double indexing (2i / 2i+1) rather than
//  double2 keeps the source free of any header dependency under NVRTC.
//  Both use grid-stride loops so any launch size is correct.
// ---------------------------------------------------------------------------
static const char* kKernelSrc = R"__lrx__(
extern "C" __global__
void lrx_scale_c128(double* x, long long n2, double s)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n2; i += stride) x[i] *= s;
}

)__lrx__";

struct KernelPack {
    CUfunction scale_fn = nullptr;
};

// Compile + load the kernels for the CURRENT context (one per process in
// production: CUDA_VISIBLE_DEVICES pins one device per rank).  Caller holds
// the global mutex.  By the time this runs, a cufftExec has already executed
// on this thread, so the runtime has bound the device's primary context; the
// cudaFree(0) fallback covers a first-call-without-context path.
static ffi::Error get_kernels(KernelPack** out) {
    static std::map<CUcontext, KernelPack> cache;
    // Negative cache (audit P1.9): an NVRTC/module failure is deterministic
    // for a given process+context, and this function is called per FFI
    // dispatch — without the cache a persistent failure re-ran the whole
    // NVRTC compile on EVERY call before failing again.  First failure is
    // recorded; later calls refuse immediately with the recorded reason.
    static std::map<CUcontext, std::string> fail_cache;
    const DriverApi& api = driver_api();
    if (!api.ok) {
        return fail("driver-api resolve",
                    std::string("the CUDA driver entry points this handler "
                                "needs could not be resolved — ") +
                        (api.err.empty()
                             ? "no detail recorded (is a CUDA driver present "
                               "on this node?)"
                             : api.err));
    }
    CUcontext ctx = nullptr;
    api.CtxGetCurrent(&ctx);
    if (ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        api.CtxGetCurrent(&ctx);
        if (ctx == nullptr) {
            return fail("context bind", "no current CUDA context");
        }
    }
    auto it = cache.find(ctx);
    if (it != cache.end()) {
        *out = &it->second;
        return ffi::Error::Success();
    }
    auto fit = fail_cache.find(ctx);
    if (fit != fail_cache.end()) {
        return fail("kernel build (cached failure, NVRTC not re-run)",
                    fit->second);
    }
    // Every failure from here on is deterministic — record it before
    // returning so the next call refuses without re-compiling.
    auto fail_sticky = [&](const char* where,
                           const std::string& detail) -> ffi::Error {
        fail_cache.emplace(ctx, std::string(where) + " — " + detail);
        return fail(where, detail);
    };

    int dev = 0, cc_major = 0, cc_minor = 0;
    LRX_CUDA_CHECK(cudaGetDevice(&dev), "cudaGetDevice");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
                       &cc_major, cudaDevAttrComputeCapabilityMajor, dev),
                   "query compute capability (major)");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
                       &cc_minor, cudaDevAttrComputeCapabilityMinor, dev),
                   "query compute capability (minor)");

    nvrtcProgram prog = nullptr;
    nvrtcResult nr = nvrtcCreateProgram(&prog, kKernelSrc,
                                        "lrx_fft_flat_k_kernels.cu",
                                        0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS) {
        return fail_sticky("nvrtcCreateProgram", nvrtcGetErrorString(nr));
    }
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d",
                  cc_major, cc_minor);
    const char* opts[] = {arch};
    nr = nvrtcCompileProgram(prog, 1, opts);
    if (nr != NVRTC_SUCCESS) {
        size_t log_sz = 0;
        std::string log;
        if (nvrtcGetProgramLogSize(prog, &log_sz) == NVRTC_SUCCESS &&
            log_sz > 1) {
            log.resize(log_sz);
            nvrtcGetProgramLog(prog, &log[0]);
        }
        nvrtcDestroyProgram(&prog);
        return fail_sticky("nvrtcCompileProgram",
                           std::string(nvrtcGetErrorString(nr)) + " — " + log);
    }
    // Prefer a native cubin (no driver PTX-JIT: the node driver may predate
    // this toolkit's PTX ISA); fall back to PTX if cubin is unavailable.
    std::vector<char> image;
    size_t sz = 0;
    bool used_cubin = false;
    if (nvrtcGetCUBINSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        image.resize(sz);
        nr = nvrtcGetCUBIN(prog, image.data());
        used_cubin = true;
    } else if (nvrtcGetPTXSize(prog, &sz) == NVRTC_SUCCESS && sz > 0) {
        image.resize(sz);
        nr = nvrtcGetPTX(prog, image.data());
    } else {
        nr = NVRTC_ERROR_INTERNAL_ERROR;
    }
    nvrtcDestroyProgram(&prog);
    if (nr != NVRTC_SUCCESS || image.empty()) {
        return fail_sticky("nvrtc get cubin/ptx", nvrtcGetErrorString(nr));
    }

    CUmodule mod = nullptr;
    CUresult cr = api.ModuleLoadData(&mod, image.data());
    if (cr != CUDA_SUCCESS) {
        return fail_sticky("cuModuleLoadData", cu_err(cr));
    }
    KernelPack pack;
    cr = api.ModuleGetFunction(&pack.scale_fn, mod, "lrx_scale_c128");
    if (cr != CUDA_SUCCESS) {
        // Unload before failing (P1.9): the module is unreachable after
        // this return — without the unload it leaked device memory on
        // every (pre-negative-cache, per-call) retry.
        api.ModuleUnload(mod);
        return fail_sticky("cuModuleGetFunction", cu_err(cr));
    }
    if (log_enabled()) {
        std::fprintf(stderr,
                     "[cufft_flat_k] NVRTC kernels compiled for sm_%d%d "
                     "(%zu B %s, device %d)\n",
                     cc_major, cc_minor, image.size(),
                     used_cubin ? "cubin" : "ptx", dev);
    }
    auto res = cache.emplace(ctx, pack);
    *out = &res.first->second;
    return ffi::Error::Success();
}

static ffi::Error launch(CUfunction fn, cudaStream_t stream, int64_t n_iters,
                         void** args, const char* where) {
    const unsigned block = 256;
    const long long want = (n_iters + block - 1) / block;
    const unsigned grid = static_cast<unsigned>(
        std::min<long long>(std::max<long long>(want, 1), 4096));
    CUresult cr = driver_api().LaunchKernel(
        fn, grid, 1, 1, block, 1, 1, /*sharedMemBytes=*/0,
        reinterpret_cast<CUstream>(stream), args, nullptr);
    if (cr != CUDA_SUCCESS) {
        return fail(where, cu_err(cr));
    }
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Plan cache + shared arenas.  All access under g_mu (see file header for
//  the single-compute-stream assumption).
// ---------------------------------------------------------------------------
struct PlanKey {
    long long d0, d1, d2;
    long long stride;   // trail-stride multiplier T (istride == ostride)
    long long batch;    // transforms per exec (== T here)

    bool operator<(const PlanKey& o) const {
        return std::tie(d0, d1, d2, stride, batch) <
               std::tie(o.d0, o.d1, o.d2, o.stride, o.batch);
    }
};

struct PlanEntry {
    cufftHandle handle = 0;
    size_t work_bytes = 0;
};

static std::mutex g_mu;
static std::map<PlanKey, PlanEntry> g_plans;
static void* g_work = nullptr;      // shared cuFFT workspace arena
static size_t g_work_cap = 0;

// Grow-only device arena.  Synchronizes the DEVICE before releasing the old
// block so no still-enqueued work can reference it (growth is a rare,
// first-calls-only event; correctness over speed).
static ffi::Error ensure_arena(void** buf, size_t* cap, size_t want,
                               const char* what) {
    if (*cap >= want) return ffi::Error::Success();
    LRX_CUDA_CHECK(cudaDeviceSynchronize(), "arena-grow sync");
    if (*buf != nullptr) {
        LRX_CUDA_CHECK(cudaFree(*buf), "arena free");
        *buf = nullptr;
        *cap = 0;
    }
    LRX_CUDA_CHECK(cudaMalloc(buf, want), "arena cudaMalloc");
    *cap = want;
    if (log_enabled()) {
        std::fprintf(stderr, "[cufft_flat_k] %s arena -> %.1f MB\n",
                     what, want / 1e6);
    }
    return ffi::Error::Success();
}

static ffi::Error get_plan(const PlanKey& k, PlanEntry** out) {
    auto it = g_plans.find(k);
    if (it != g_plans.end()) {
        *out = &it->second;
        return ffi::Error::Success();
    }
    PlanEntry e;
    LRX_CUFFT_CHECK(cufftCreate(&e.handle), "cufftCreate");
    LRX_CUFFT_CHECK(cufftSetAutoAllocation(e.handle, 0),
                    "cufftSetAutoAllocation");
    long long n[3] = {k.d0, k.d1, k.d2};
    long long inembed[3] = {k.d0, k.d1, k.d2};
    long long onembed[3] = {k.d0, k.d1, k.d2};
    cufftResult r = cufftMakePlanMany64(
        e.handle, /*rank=*/3, n,
        inembed, /*istride=*/k.stride, /*idist=*/1,
        onembed, /*ostride=*/k.stride, /*odist=*/1,
        CUFFT_Z2Z, /*batch=*/k.batch, &e.work_bytes);
    if (r != CUFFT_SUCCESS) {
        cufftDestroy(e.handle);
        std::ostringstream os;
        os << cufft_err_name(r) << " (dims=(" << k.d0 << "," << k.d1 << ","
           << k.d2 << ") stride=" << k.stride << " batch=" << k.batch << ")";
        return fail("cufftMakePlanMany64", os.str());
    }
    if (log_enabled()) {
        std::fprintf(stderr,
                     "[cufft_flat_k] plan dims=(%lld,%lld,%lld) stride=%lld "
                     "batch=%lld workspace=%.1f MB\n",
                     k.d0, k.d1, k.d2, k.stride, k.batch,
                     e.work_bytes / 1e6);
    }
    auto res = g_plans.emplace(k, e);
    *out = &res.first->second;
    return ffi::Error::Success();
}

// One exec: bind stream + shared workspace, run Z2Z in `dir`.
static ffi::Error exec(PlanEntry* p, cudaStream_t stream, const C128* in,
                       C128* out, int dir, const char* where) {
    ffi::Error e = ensure_arena(&g_work, &g_work_cap,
                                std::max<size_t>(p->work_bytes, 1),
                                "workspace");
    if (!e.success()) return e;
    LRX_CUFFT_CHECK(cufftSetStream(p->handle, stream), "cufftSetStream");
    LRX_CUFFT_CHECK(cufftSetWorkArea(p->handle, g_work), "cufftSetWorkArea");
    // Out-of-place Z2Z never writes the input, so the const_cast is safe for
    // the XLA read-only operand; in == out is the granted-alias in-place run.
    cufftResult r = cufftExecZ2Z(
        p->handle,
        reinterpret_cast<cufftDoubleComplex*>(const_cast<C128*>(in)),
        reinterpret_cast<cufftDoubleComplex*>(out), dir);
    if (r != CUFFT_SUCCESS) return fail(where, cufft_err_name(r));
    return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
//  Handler 1: plain batched flat-k FFT/IFFT (mirror of MklFftFlatKHostFfi).
// ---------------------------------------------------------------------------
static ffi::Error FlatKDispatch(
    cudaStream_t stream,
    ffi::AnyBuffer X, ffi::Result<ffi::AnyBuffer> Y,
    int64_t nkx, int64_t nky, int64_t nkz, int64_t forward, double scale)
{
    announce_inert_host_knobs();
    if (X.element_type() != ffi::DataType::C128 ||
        Y->element_type() != ffi::DataType::C128) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cufft.flat_k: buffers must be complex128");
    }
    auto dims = X.dimensions();
    auto odims = Y->dimensions();
    if (dims.size() < 1 || dims.size() != odims.size() ||
        !std::equal(dims.begin(), dims.end(), odims.begin())) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "cufft.flat_k: output shape must equal input shape");
    }
    const int64_t nk = dims[0];
    if (nkx < 1 || nky < 1 || nkz < 1 || nk != nkx * nky * nkz) {
        std::ostringstream os;
        os << "cufft.flat_k: leading flat-k extent " << nk
           << " != nkx*nky*nkz = " << nkx << "*" << nky << "*" << nkz;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    int64_t T = 1;
    for (size_t i = 1; i < dims.size(); ++i) T *= dims[i];
    if (T == 0) return ffi::Error::Success();  // empty trail: nothing to do

    const auto* in = static_cast<const C128*>(X.untyped_data());
    auto* out = static_cast<C128*>(Y->untyped_data());
    const bool inplace = (static_cast<const void*>(in) ==
                          static_cast<const void*>(out));

    std::lock_guard<std::mutex> lock(g_mu);
    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                         "[cufft_flat_k] flat_k first call: nk=(%ld,%ld,%ld) "
                         "T=%ld fwd=%ld scale=%.6e inplace=%d\n",
                         (long)nkx, (long)nky, (long)nkz, (long)T,
                         (long)forward, scale, (int)inplace);
        }
    }
    PlanEntry* plan = nullptr;
    PlanKey key{nkx, nky, nkz, T, T};
    ffi::Error e = get_plan(key, &plan);
    if (!e.success()) return e;
    e = exec(plan, stream, in, out,
             forward != 0 ? CUFFT_FORWARD : CUFFT_INVERSE, "flat_k exec");
    if (!e.success()) return e;
    if (scale != 1.0) {
        KernelPack* kp = nullptr;
        e = get_kernels(&kp);
        if (!e.success()) return e;
        long long n2 = 2 * (long long)nk * T;
        double* xd = reinterpret_cast<double*>(out);
        void* args[] = {&xd, &n2, &scale};
        e = launch(kp->scale_fn, stream, n2, args, "flat_k scale kernel");
        if (!e.success()) return e;
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::cufft_flat_k

// SAME target string as the host handler (lorrax_mklfft_flat_k),
// DIFFERENT symbol name so both platform .so's can
// coexist under RTLD_GLOBAL — the phdf5 platform_seam.h registration split.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CufftFlatKCudaFfi,
    lorrax_ffi::cufft_flat_k::FlatKDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla::ffi::AnyBuffer>()      // X (nk, *trail) c128
        .Ret<xla::ffi::AnyBuffer>()      // Y same shape (may alias X)
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<int64_t>("forward")        // 0 = ifftn, 1 = fftn
        .Attr<double>("scale"));         // total jnp-convention norm scale
