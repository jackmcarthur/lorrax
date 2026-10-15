// rocsolver/batched_eigh_ffi.cc — the ROCM leg's batched local Hermitian eigh:
// ONE rocsolver_{z,d}{heevd,syevd}_strided_batched call for a whole [..., n, n] stack.
//
// Why: jaxlib's ROCm eigh (hipsolver_syevd_ffi) loops rocSOLVER zheevd over the batch,
// one matrix at a time (Fe line_select: c128[16,312,312], 11.2 ms per matrix, 26.4k
// kernels per call).  The strided-batched driver runs the batch in one pass of the same
// algorithm (ctypes probe, pool log stageprof_rsb: 19.0 ms vs 178.4 ms for 16 x 312,
// |dw| 1.2e-13).  distrib_la.local_eigh routes to this target by its BATCHED_EIGH_ROUTE
// row; the XLA reference (jnp.linalg.eigh) is the gate (tests/test_rocsolver_eigh_gate.py).
//
// Contract (the Python side owns layout and failure semantics):
//   in   a      [..., n, n]  c128 | f64, read as COLUMN-major per matrix (XLA hands a
//                row-major buffer: rocSOLVER sees A^T = conj(A); distrib_la undoes it)
//   out  v      [..., n, n]  same dtype: rocSOLVER's in-place eigenvectors (columns,
//                column-major), i.e. conj(V)^T read row-major
//   out  w      [..., n]     f64 ascending eigenvalues
//   out  info   [...]        s32 rocSOLVER info per matrix (0 = converged)
//   attr lower  0/1          which triangle rocSOLVER reads (column-major sense)
//   attr jacobi 0/1          the driver: {heevd,syevd} (0) or {heevdj,syevdj} (1), the route
//                row's per-dtype choice (distrib_la.local_eigh.BATCHED_EIGH_ROUTE)
//
// No library is linked: rocSOLVER is resolved from the copy this process already mapped
// (the JAX ROCm plugin's hipSOLVER pulls in librocsolver.so.0), and the rocBLAS handle
// API through that library's own dependency scope, so the process keeps one ROCm.  A
// process without it refuses by GATE rocsolver-mapped.  Workspace: queried once per
// (device, dtype, n, batch) in rocBLAS size-query mode, then handed to rocBLAS from
// XLA's scratch allocator on every call (no allocation inside rocSOLVER, stream-ordered).
#include "../common/leg.h"
#if !defined(LRX_LEG_ROCM)
#  error "rocsolver/batched_eigh_ffi.cc is ROCM-leg only (CUDA serves eigh through jaxlib's cuSOLVER)"
#endif

#include <dlfcn.h>

#include <cstdint>
#include <map>
#include <optional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>

#include "../common/ffi_helpers.h"
#include "../common/gpu_runtime.h"

namespace lorrax_ffi::rocsolver_eigh {

namespace ffi = ::xla::ffi;

// The rocBLAS / rocSOLVER C ABI this file uses, spelled here (no headers needed): handles
// are opaque pointers, enums are C ints, rocblas_int is int32 (LP64 build), strides int64.
using RbHandle = void*;
using RbStatus = int;                       // rocblas_status; 0 = rocblas_status_success
using RbInt = int32_t;
using RbStride = int64_t;
constexpr int kEvectOriginal = 211;         // rocblas_evect_original
constexpr int kFillUpper = 121;             // rocblas_fill_upper
constexpr int kFillLower = 122;             // rocblas_fill_lower

using ZheevdSb = RbStatus (*)(RbHandle, int, int, RbInt, void*, RbInt, RbStride, double*,
                              RbStride, double*, RbStride, RbInt*, RbInt);
using DsyevdSb = RbStatus (*)(RbHandle, int, int, RbInt, double*, RbInt, RbStride, double*,
                              RbStride, double*, RbStride, RbInt*, RbInt);
// The Jacobi-subproblem divide-and-conquer drivers ({z,d}{heevdj,syevdj}): no E output.
using ZheevdjSb = RbStatus (*)(RbHandle, int, int, RbInt, void*, RbInt, RbStride, double*,
                               RbStride, RbInt*, RbInt);
using DsyevdjSb = RbStatus (*)(RbHandle, int, int, RbInt, double*, RbInt, RbStride, double*,
                               RbStride, RbInt*, RbInt);

struct Api {
    RbStatus (*create_handle)(RbHandle*) = nullptr;
    RbStatus (*set_stream)(RbHandle, lrx::gpu::Stream) = nullptr;
    RbStatus (*set_workspace)(RbHandle, void*, size_t) = nullptr;
    RbStatus (*start_query)(RbHandle) = nullptr;
    RbStatus (*stop_query)(RbHandle, size_t*) = nullptr;
    ZheevdSb zheevd_sb = nullptr;
    DsyevdSb dsyevd_sb = nullptr;
    ZheevdjSb zheevdj_sb = nullptr;
    DsyevdjSb dsyevdj_sb = nullptr;
    std::string path;
    std::string err;
};

static std::string loaded_path(void* sym) {
    Dl_info info{};
    return (dladdr(sym, &info) && info.dli_fname) ? std::string(info.dli_fname) : std::string("?");
}

static const Api& api() {
    static const Api a = [] {
        Api r;
        constexpr const char* kSoname = "librocsolver.so.0";
        void* h = dlopen(kSoname, RTLD_NOW | RTLD_NOLOAD);
        if (h == nullptr) {
            r.err = std::string("GATE rocsolver-mapped: ") + kSoname + " is not mapped in this "
                    "process; want the copy the JAX ROCm plugin maps (one ROCm per process); "
                    "fix: import jax with the rocm plugin before the first call";
            return r;
        }
        auto need = [&](const char* name) {
            void* p = dlsym(h, name);       // the library, then its dependencies (rocBLAS)
            if (p == nullptr) r.err += std::string(r.err.empty() ? "" : ", ") + name;
            return p;
        };
#define LRX_BIND(field, name) r.field = reinterpret_cast<decltype(r.field)>(need(name))
        LRX_BIND(create_handle, "rocblas_create_handle");
        LRX_BIND(set_stream, "rocblas_set_stream");
        LRX_BIND(set_workspace, "rocblas_set_workspace");
        LRX_BIND(start_query, "rocblas_start_device_memory_size_query");
        LRX_BIND(stop_query, "rocblas_stop_device_memory_size_query");
        LRX_BIND(zheevd_sb, "rocsolver_zheevd_strided_batched");
        LRX_BIND(dsyevd_sb, "rocsolver_dsyevd_strided_batched");
        LRX_BIND(zheevdj_sb, "rocsolver_zheevdj_strided_batched");
        LRX_BIND(dsyevdj_sb, "rocsolver_dsyevdj_strided_batched");
#undef LRX_BIND
        if (!r.err.empty()) {
            r.err = "GATE rocsolver-symbols: dlsym found no " + r.err + " in " +
                    loaded_path(dlsym(h, "rocsolver_get_version_string"));
            return r;
        }
        r.path = loaded_path(reinterpret_cast<void*>(r.zheevd_sb));
        return r;
    }();
    return a;
}

// One rocBLAS handle per device, created on first use and kept for the process (no
// teardown-order hazard at exit); its mutex covers set_stream + set_workspace + the call.
struct DeviceHandle {
    RbHandle h = nullptr;
    std::mutex m;
    // Workspace bytes per (dtype, n, batch, fill), under m.
    std::map<std::tuple<int, int64_t, int64_t, int>, size_t> ws;
};

static ffi::Error device_handle(DeviceHandle** out) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<DeviceHandle>> handles;
    int dev = -1;
    LORRAX_GPU_CHECK(lrx::gpu::get_device(&dev));
    std::lock_guard<std::mutex> lk(mu);
    auto& slot = handles[dev];
    if (!slot) {
        auto d = std::make_unique<DeviceHandle>();
        LORRAX_LIB_CHECK(api().create_handle(&d->h), 0, "rocblas_create_handle");
        slot = std::move(d);
    }
    *out = slot.get();
    return ffi::Error::Success();
}

static ffi::Error run(lrx::gpu::Stream stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer a,
                      ffi::Result<ffi::AnyBuffer> v, ffi::Result<ffi::AnyBuffer> w,
                      ffi::Result<ffi::Buffer<ffi::S32>> info, int64_t lower, int64_t jacobi) {
    const Api& A = api();
    if (!A.err.empty()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, A.err);
    const auto dims = a.dimensions();
    const size_t rank = dims.size();
    if (rank < 2 || dims[rank - 1] != dims[rank - 2]) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-eigh-shape: want [..., n, n]");
    }
    const ffi::DataType dt = a.element_type();
    if (dt != ffi::DataType::C128 && dt != ffi::DataType::F64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-eigh-dtype: want complex128 or float64 (the route "
                          "row's dtypes); other dtypes take jnp.linalg.eigh");
    }
    if (v->element_type() != dt || v->element_count() != a.element_count() ||
        w->element_type() != ffi::DataType::F64) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-eigh-results: want v like a and f64 w");
    }
    const int64_t n = dims[rank - 1];
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < rank; ++i) batch *= dims[i];
    if (n == 0 || batch == 0) return ffi::Error::Success();
    if (n > INT32_MAX || batch > INT32_MAX) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "GATE rocsolver-eigh-int32: n and batch must fit rocblas_int");
    }
    const size_t esize = (dt == ffi::DataType::C128) ? 16 : 8;
    if (v->untyped_data() != a.untyped_data()) {
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(v->untyped_data(), a.untyped_data(),
                                                static_cast<size_t>(batch * n * n) * esize,
                                                lrx::gpu::kDeviceToDevice, stream));
    }
    // E: the tridiagonal off-diagonal rocSOLVER returns (unused), from XLA's scratch.
    std::optional<void*> e = scratch.Allocate(static_cast<size_t>(batch * n) * sizeof(double));
    if (!e) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "scratch for E");
    const int fill = lower ? kFillLower : kFillUpper;
    const RbInt ni = static_cast<RbInt>(n), bi = static_cast<RbInt>(batch);
    double* wp = static_cast<double*>(w->untyped_data());
    double* ep = static_cast<double*>(*e);
    RbInt* ip = info->typed_data();
    double* vr = static_cast<double*>(v->untyped_data());
    auto call = [&](RbHandle h) -> RbStatus {
        if (dt == ffi::DataType::C128) {
            if (jacobi) return A.zheevdj_sb(h, kEvectOriginal, fill, ni, v->untyped_data(), ni, n * n, wp, n, ip, bi);
            return A.zheevd_sb(h, kEvectOriginal, fill, ni, v->untyped_data(), ni, n * n, wp, n, ep, n, ip, bi);
        }
        if (jacobi) return A.dsyevdj_sb(h, kEvectOriginal, fill, ni, vr, ni, n * n, wp, n, ip, bi);
        return A.dsyevd_sb(h, kEvectOriginal, fill, ni, vr, ni, n * n, wp, n, ep, n, ip, bi);
    };

    DeviceHandle* dh = nullptr;
    FFI_RETURN_IF_ERROR(device_handle(&dh));
    std::lock_guard<std::mutex> lk(dh->m);
    LORRAX_LIB_CHECK(A.set_stream(dh->h, stream), 0, "rocblas_set_stream");
    // Workspace bytes per (device, dtype, n, batch): rocBLAS size-query mode runs only the
    // argument checks and the size computation (pointers must be valid, nothing executes).
    const auto key = std::make_tuple(static_cast<int>(dt) * 2 + (jacobi ? 1 : 0), n, batch, fill);
    auto it = dh->ws.find(key);
    if (it == dh->ws.end()) {
        size_t bytes = 0;
        LORRAX_LIB_CHECK(A.start_query(dh->h), 0, "rocblas_start_device_memory_size_query");
        const RbStatus q = call(dh->h);
        const RbStatus s = A.stop_query(dh->h, &bytes);
        // In query mode the solver returns rocblas_status_size_increased/unchanged (> 0 but
        // not an error) or success; only a genuine argument error (invalid_*) is fatal.
        if (s != 0) {
            std::ostringstream os;
            os << "rocblas_stop_device_memory_size_query status=" << s << " (solver query " << q << ")";
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        it = dh->ws.emplace(key, bytes).first;
    }
    void* ws = nullptr;
    if (it->second > 0) {
        std::optional<void*> p = scratch.Allocate(it->second);
        if (!p) return ffi::Error(ffi::ErrorCode::kResourceExhausted, "scratch for rocSOLVER workspace");
        ws = *p;
    }
    LORRAX_LIB_CHECK(A.set_workspace(dh->h, ws, it->second), 0, "rocblas_set_workspace");
    const RbStatus st = call(dh->h);
    if (st != 0) {
        std::ostringstream os;
        os << "rocsolver_" << (dt == ffi::DataType::C128 ? "zheevd" : "dsyevd") << (jacobi ? "j" : "")
           << "_strided_batched"
           << " status=" << st << " (n=" << n << ", batch=" << batch << ", ws=" << it->second
           << " B, " << A.path << ")";
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::rocsolver_eigh

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RocsolverBatchedEighFfi, lorrax_ffi::rocsolver_eigh::run,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()                    // a  [..., n, n]
        .Ret<xla::ffi::AnyBuffer>()                    // v  [..., n, n]
        .Ret<xla::ffi::AnyBuffer>()                    // w  [..., n] f64
        .Ret<xla::ffi::Buffer<xla::ffi::S32>>()        // info [...]
        .Attr<int64_t>("lower")                        // rocSOLVER fill (column-major sense)
        .Attr<int64_t>("jacobi"));                     // 1: {z,d}{heevdj,syevdj} (Jacobi subproblems)
