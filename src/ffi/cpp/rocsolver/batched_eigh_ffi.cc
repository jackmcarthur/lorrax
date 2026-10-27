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
//   out  arena  [query bytes] u8 vendor workspace + tridiagonal E
//   attr lower  0/1          which triangle rocSOLVER reads (column-major sense)
//   attr jacobi 0/1          the driver: {heevd,syevd} (0) or {heevdj,syevdj} (1), the route
//                row's per-dtype choice (distrib_la.local_eigh.BATCHED_EIGH_ROUTE)
//   attr tile_bytes          the workspace cap (the fixed tile): the stack runs in chunks whose
//                rocSOLVER workspace fits it, on one explicit u8 result
// No host-device synchronization: nothing on the device is read back; failed matrices are
// marked NaN on the device by distrib_la.local_eigh.
//
// No library is linked: rocSOLVER is resolved from the copy this process already mapped
// (the JAX ROCm plugin's hipSOLVER pulls in librocsolver.so.0), and the rocBLAS handle
// API through that library's own dependency scope, so the process keeps one ROCm.  A
// process without it refuses by GATE rocsolver-mapped.  Workspace: queried once per
// (device, dtype, n, batch) in rocBLAS size-query mode, then handed to rocBLAS from
// an explicit u8 FFI result on every call (no allocation inside rocSOLVER, stream-ordered),
// capped by the fixed tile (attr tile_bytes).
#include "../common/leg.h"
#if !defined(LRX_LEG_ROCM)
#  error "rocsolver/batched_eigh_ffi.cc is ROCM-leg only (CUDA serves eigh through jaxlib's cuSOLVER)"
#endif

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <climits>
#include <cstring>
#include <limits>
#include <map>
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

// The same cached shape plan sizes the explicit result and serves execution.
struct WorkspacePlan {
    int64_t chunk = 0;
    size_t vendor = 0;
    size_t e_offset = 0;
    size_t total = 0;
};

struct DeviceHandle {
    RbHandle h = nullptr;
    std::mutex m;
    std::map<std::tuple<int, int64_t, int64_t, int, int>, size_t> ws;
    std::map<std::tuple<int, int64_t, int64_t, int, int, int64_t>, WorkspacePlan> plans;
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
        LORRAX_LIB_CHECK(api().set_workspace(d->h, nullptr, 0), 0, "rocblas_set_workspace(initialize)");
        slot = std::move(d);
    }
    *out = slot.get();
    return ffi::Error::Success();
}

static ffi::Error valid_shape(int kind, int64_t n, int64_t batch, int64_t lower,
                              int64_t jacobi, int64_t tile) {
    if ((kind != 1 && kind != 3) || n < 0 || n > INT32_MAX || batch < 0 || batch > INT32_MAX ||
        (lower != 0 && lower != 1) || (jacobi != 0 && jacobi != 1) || tile <= 0)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
            "GATE rocsolver-eigh-shape: want kind1(f64)/3(c128), n/batch0..INT32_MAX, lower/jacobi0..1, tile>0");
    const size_t item = kind == 3 ? 16 : 8;
    if (n && (static_cast<uint64_t>(n) > std::numeric_limits<size_t>::max() / static_cast<uint64_t>(n) / item ||
              static_cast<uint64_t>(batch) > std::numeric_limits<size_t>::max() / (static_cast<uint64_t>(n) * n * item)))
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "rocsolver eigh matrix byte span overflow");
    return ffi::Error::Success();
}

// Query mode receives null device pointers directly, with no null-pointer
// offset arithmetic. Normal execution supplies this same driver's live buffers.
static RbStatus solve(const Api& A, RbHandle h, int kind, int64_t n, int64_t batch,
                      int fill, bool jacobi, void* v, double* w, double* e, RbInt* info) {
    const RbInt ni = static_cast<RbInt>(n), nb = static_cast<RbInt>(batch);
    if (kind == 3) {
        if (jacobi) return A.zheevdj_sb(h, kEvectOriginal, fill, ni, v, ni, n * n, w, n, info, nb);
        return A.zheevd_sb(h, kEvectOriginal, fill, ni, v, ni, n * n, w, n, e, n, info, nb);
    }
    if (jacobi) return A.dsyevdj_sb(h, kEvectOriginal, fill, ni, static_cast<double*>(v), ni, n * n, w, n, info, nb);
    return A.dsyevd_sb(h, kEvectOriginal, fill, ni, static_cast<double*>(v), ni, n * n, w, n, e, n, info, nb);
}

// The caller holds dh->m throughout planning/query/execution.
static ffi::Error workspace_locked(DeviceHandle* dh, int kind, int64_t n, int64_t batch,
                                   int fill, int jacobi, size_t* bytes) {
    const auto key = std::make_tuple(kind, n, batch, fill, jacobi);
    auto it = dh->ws.find(key);
    if (it == dh->ws.end()) {
        const Api& A = api();
        LORRAX_LIB_CHECK(A.start_query(dh->h), 0, "rocblas_start_device_memory_size_query");
        size_t need = 0;
        const RbStatus query = solve(A, dh->h, kind, n, batch, fill, jacobi, nullptr, nullptr, nullptr, nullptr);
        const RbStatus stopped = A.stop_query(dh->h, &need);
        // rocBLAS query-mode success may report size_unchanged / size_increased.
        if ((query != 0 && query != 9 && query != 10) || stopped != 0) {
            std::ostringstream os;
            os << "rocsolver eigh workspace query status=" << query
               << "; rocblas_stop_device_memory_size_query status=" << stopped;
            return ffi::Error(ffi::ErrorCode::kInternal, os.str());
        }
        it = dh->ws.emplace(key, need).first;
    }
    *bytes = it->second;
    return ffi::Error::Success();
}

static ffi::Error plan_locked(DeviceHandle* dh, int kind, int64_t n, int64_t batch,
                              int fill, int jacobi, int64_t tile, WorkspacePlan* out) {
    const auto key = std::make_tuple(kind, n, batch, fill, jacobi, tile);
    auto it = dh->plans.find(key);
    if (it == dh->plans.end()) {
        size_t one = 0;
        FFI_RETURN_IF_ERROR(workspace_locked(dh, kind, n, 1, fill, jacobi, &one));
        if (one > static_cast<size_t>(tile)) {
            std::ostringstream os;
            os << "GATE rocsolver-eigh-tile: one n=" << n << " matrix needs " << one
               << " B of rocSOLVER workspace, above the " << tile << " B tile";
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
        }
        WorkspacePlan p;
        p.chunk = one ? std::min<int64_t>(batch, static_cast<size_t>(tile) / one) : batch;
        for (;;) {
            FFI_RETURN_IF_ERROR(workspace_locked(dh, kind, n, p.chunk, fill, jacobi, &p.vendor));
            const int64_t rest = batch % p.chunk;
            if (rest) {
                size_t tail = 0;
                FFI_RETURN_IF_ERROR(workspace_locked(dh, kind, n, rest, fill, jacobi, &tail));
                p.vendor = std::max(p.vendor, tail);
            }
            if (p.vendor <= static_cast<size_t>(tile)) break;
            --p.chunk;  // the one-matrix bound above guarantees progress to a fit
        }
        constexpr size_t alignment = 256;
        const size_t e_bytes = jacobi ? 0 : static_cast<size_t>(batch) * n * sizeof(double);
        if (p.vendor > std::numeric_limits<size_t>::max() - (alignment - 1) - e_bytes)
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, "rocsolver eigh workspace byte span overflow");
        p.e_offset = (p.vendor + alignment - 1) / alignment * alignment;
        p.total = p.e_offset + e_bytes;
        it = dh->plans.emplace(key, p).first;
    }
    *out = it->second;
    return ffi::Error::Success();
}

static ffi::Error run(lrx::gpu::Stream stream, ffi::AnyBuffer a,
                      ffi::Result<ffi::AnyBuffer> v, ffi::Result<ffi::AnyBuffer> w,
                      ffi::Result<ffi::Buffer<ffi::S32>> info,
                      ffi::Result<ffi::Buffer<ffi::U8>> arena, int64_t lower,
                      int64_t jacobi, int64_t tile_bytes) {
    const Api& A = api();
    if (!A.err.empty()) return ffi::Error(ffi::ErrorCode::kFailedPrecondition, A.err);
    const auto dims = a.dimensions();
    const size_t rank = dims.size();
    if (rank < 2 || dims[rank - 1] != dims[rank - 2])
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-eigh-shape: want [..., n, n]");
    const auto dt = a.element_type();
    const int kind = dt == ffi::DataType::C128 ? 3 : dt == ffi::DataType::F64 ? 1 : -1;
    const int64_t n = dims[rank - 1];
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (dims[i] < 0 || dims[i] > INT32_MAX || batch > INT32_MAX / std::max<int64_t>(dims[i], 1))
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-eigh-int32: batch overflow");
        batch *= dims[i];
    }
    FFI_RETURN_IF_ERROR(valid_shape(kind, n, batch, lower, jacobi, tile_bytes));
    if (v->element_type() != dt || v->element_count() != a.element_count() ||
        w->element_type() != ffi::DataType::F64 || w->element_count() != batch * n ||
        info->element_count() != batch)
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-eigh-results: want v like a, f64 w and s32 info");
    if (!n || !batch) {
        if (arena->element_count() != 0)
            return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GATE rocsolver-eigh-workspace: empty solve needs zero-byte result");
        return ffi::Error::Success();
    }
    DeviceHandle* dh = nullptr;
    FFI_RETURN_IF_ERROR(device_handle(&dh));
    std::lock_guard<std::mutex> lock(dh->m);
    const int fill = lower ? kFillLower : kFillUpper;
    WorkspacePlan p;
    FFI_RETURN_IF_ERROR(plan_locked(dh, kind, n, batch, fill, jacobi, tile_bytes, &p));
    if (arena->dimensions().size() != 1 || static_cast<size_t>(arena->element_count()) != p.total) {
        std::ostringstream os;
        os << "GATE rocsolver-eigh-workspace: explicit result has " << arena->element_count()
           << " B, want " << p.total;
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, os.str());
    }
    const size_t item = kind == 3 ? 16 : 8;
    if (v->untyped_data() != a.untyped_data())
        LORRAX_GPU_CHECK(lrx::gpu::memcpy_async(v->untyped_data(), a.untyped_data(),
            static_cast<size_t>(batch) * n * n * item, lrx::gpu::kDeviceToDevice, stream));
    LORRAX_LIB_CHECK(A.set_stream(dh->h, stream), 0, "rocblas_set_stream");
    char* buffer = static_cast<char*>(arena->untyped_data());
    double* ep = jacobi ? nullptr : reinterpret_cast<double*>(buffer + p.e_offset);
    void* ws = p.vendor ? buffer : nullptr;
    LORRAX_LIB_CHECK(A.set_workspace(dh->h, ws, p.vendor), 0, "rocblas_set_workspace");
    RbStatus status = 0;
    int64_t first = 0;
    for (; first < batch; first += p.chunk) {
        const int64_t nb = std::min(p.chunk, batch - first);
        status = solve(A, dh->h, kind, n, nb, fill, jacobi,
            static_cast<char*>(v->untyped_data()) + first * static_cast<size_t>(n) * n * item,
            static_cast<double*>(w->untyped_data()) + first * n,
            jacobi ? nullptr : ep + first * n, info->typed_data() + first);
        if (status != 0) break;
    }
    const RbStatus detached = A.set_workspace(dh->h, nullptr, 0);
    if (status != 0) {
        std::ostringstream os;
        os << "rocsolver eigh status=" << status << " (n=" << n << ", matrices " << first
           << " of " << batch << ", ws=" << p.vendor << " B, " << A.path << ")";
        return ffi::Error(ffi::ErrorCode::kInternal, os.str());
    }
    LORRAX_LIB_CHECK(detached, 0, "rocblas_set_workspace(detach)");
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::rocsolver_eigh

extern "C" int lrx_rocsolver_eigh_workspace_bytes(int kind, int64_t n, int64_t batch,
                                                int lower, int jacobi, int64_t tile,
                                                uint64_t* out, char* error, size_t capacity) {
    using namespace lorrax_ffi::rocsolver_eigh;
    auto report = [&](const std::string& message) {
        if (error != nullptr && capacity) {
            const size_t len = std::min(message.size(), capacity - 1);
            std::memcpy(error, message.data(), len); error[len] = '\0';
        }
        return 1;
    };
    if (!out) return report("workspace query needs an output pointer");
    *out = 0;
    if (auto e = valid_shape(kind, n, batch, lower, jacobi, tile); e.failure()) return report(e.message());
    if (!n || !batch) return 0;
    try {
        if (!api().err.empty()) return report(api().err);
        DeviceHandle* dh = nullptr;
        if (auto e = device_handle(&dh); e.failure()) return report(e.message());
        std::lock_guard<std::mutex> lock(dh->m);
        WorkspacePlan p;
        if (auto e = plan_locked(dh, kind, n, batch, lower ? kFillLower : kFillUpper, jacobi, tile, &p); e.failure())
            return report(e.message());
        *out = p.total;
        return 0;
    } catch (const std::exception& e) { return report(e.what()); }
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RocsolverBatchedEighFfi, lorrax_ffi::rocsolver_eigh::run,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<lrx::gpu::Stream>>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::Buffer<xla::ffi::S32>>()
        .Ret<xla::ffi::Buffer<xla::ffi::U8>>()
        .Attr<int64_t>("lower")
        .Attr<int64_t>("jacobi")
        .Attr<int64_t>("tile_bytes"));
