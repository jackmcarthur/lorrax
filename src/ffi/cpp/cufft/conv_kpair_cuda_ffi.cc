// conv_kpair_cuda_ffi.cc -- ISDF CCT/ZCT post-pair convolution, CUDA leg.
//
//   U[kx,ky,kz,col,mu] = scale * FFT_k(
//       sum_ab phase_L[a] phase_R[b]
//         conj(IFFT_k(A[k,a,col,mu,b]))
//              * IFFT_k(B[k,perm_L[a],col,mu,perm_R[b]])) )
//
// The three leading transform axes are runtime attributes in the validated
// [1,24]^3 envelope.  The resident arm uses two odd-stride transform banks
// for charge and adds one accumulator bank for spin.  If one row does not fit
// the loaded device's checked opt-in SMEM ceiling, the two-stage arm uses
// XLA-owned rank-5 scratch and line-resident axis transforms.  Both arms own
// disjoint (col,mu) rows and introduce no collective.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "../common/mkl_thread_pin.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "xla/ffi/api/ffi.h"

namespace lorrax_ffi::conv_kpair {

namespace ffi = ::xla::ffi;

#if LORRAX_FFI_HAVE_PREBUILT_SM80
extern "C" {
extern const unsigned char _binary_lrx_conv_kpair_sm80_cubin_start[];
extern const unsigned char _binary_lrx_conv_kpair_sm80_cubin_end[];
}
#endif

// Policy bounds, not radix specializations or hardware limits.  Keep the
// axis envelope identical to fft.py::_CONV_KPAIR_AXIS_MAX.
static constexpr int kAxisMax = 24;
static constexpr int kSmemPreferred = 49152;
static constexpr int kEptMax = 16;
static constexpr int kEptPreferred = 4;
static constexpr int kBlockMax = 512;
static constexpr int kRowsMax = 64;

static bool log_enabled() {
    static const bool on = [] {
        return mklpin::log_here("LORRAX_CONV_KPAIR_LOG") ||
               mklpin::log_here("LORRAX_FFT_FFI_LOG");
    }();
    return on;
}

static ffi::Error fail(const char* where, const std::string& detail,
                       ffi::ErrorCode code = ffi::ErrorCode::kInternal) {
    std::ostringstream os;
    os << "conv_kpair (ISDF two-input k-convolution CUDA FFI): " << where
       << " failed -- " << detail;
    return ffi::Error(code, os.str());
}

#define LRX_CUDA_CHECK(expr, where)                                      \
    do {                                                                 \
        cudaError_t _e = (expr);                                         \
        if (_e != cudaSuccess) {                                         \
            return fail((where), cudaGetErrorString(_e));                \
        }                                                                \
    } while (0)

// INTENTIONAL NVRTC/driver-glue fork from conv_klead_cuda_ffi.cc.  Extraction
// is outside this lane.  The following protocol must stay identical: resolve
// the driver entry points lazily; bind and key the current context; cache both
// successful modules and sticky build failures per context; compile only for
// the active architecture with NVRTC's default FMA policy; prefer CUBIN then
// PTX; and check cuFuncSetAttribute for every launchable arm before recording
// the device SMEM ceiling.  Those are correctness/refusal semantics, not
// implementation convenience.
struct DriverApi {
    CUresult (*ModuleLoadData)(CUmodule*, const void*) = nullptr;
    CUresult (*ModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*ModuleUnload)(CUmodule) = nullptr;
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned,
                             unsigned, unsigned, unsigned, unsigned,
                             CUstream, void**, void**) = nullptr;
    CUresult (*CtxGetCurrent)(CUcontext*) = nullptr;
    CUresult (*GetErrorString)(CUresult, const char**) = nullptr;
    CUresult (*FuncSetAttribute)(CUfunction, int, int) = nullptr;
    bool ok = false;
    std::string err;
};

static const DriverApi& driver_api() {
    static DriverApi api = [] {
        DriverApi a;
        void* handle = RTLD_DEFAULT;
        dlerror();
        if (dlsym(handle, "cuLaunchKernel") == nullptr) {
            handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (handle == nullptr) {
                const char* e = dlerror();
                a.err = std::string("dlopen(libcuda.so.1): ") +
                        (e ? e : "(dlerror returned no detail)");
                return a;
            }
        }
        auto need = [&](const char* name) -> void* {
            dlerror();
            void* p = dlsym(handle, name);
            if (p == nullptr) {
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
        a.FuncSetAttribute = reinterpret_cast<decltype(a.FuncSetAttribute)>(
            need("cuFuncSetAttribute"));
        a.ok = a.ModuleLoadData && a.ModuleGetFunction && a.ModuleUnload &&
               a.LaunchKernel && a.CtxGetCurrent && a.GetErrorString &&
               a.FuncSetAttribute;
        return a;
    }();
    return api;
}

static std::string cu_err(CUresult result) {
    const DriverApi& api = driver_api();
    const char* text = nullptr;
    if (api.GetErrorString &&
        api.GetErrorString(result, &text) == CUDA_SUCCESS && text) {
        return text;
    }
    return "CUresult=" + std::to_string(static_cast<int>(result));
}

static const char* kKernelSrc = R"__lrx__(
struct __align__(16) lrx_c2 { double x, y; };

__device__ __forceinline__ lrx_c2 lrx_mul(lrx_c2 a, lrx_c2 b) {
    lrx_c2 z = {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x};
    return z;
}

__device__ __forceinline__ lrx_c2 lrx_phase(lrx_c2 z, int code) {
    if (code == 1) { lrx_c2 q = {-z.y, z.x}; return q; }
    if (code == 2) { lrx_c2 q = {-z.x, -z.y}; return q; }
    if (code == 3) { lrx_c2 q = {z.y, -z.x}; return q; }
    return z;
}

template <int EPT, int AXIS>
__device__ __forceinline__ void lrx_pass(
    const lrx_c2* row, const lrx_c2* tw,
    int nk, int lane, int tpr, int n0, int n1, int n2,
    const int (&kx)[EPT], const int (&ky)[EPT], const int (&kz)[EPT],
    double sign, lrx_c2 (&out)[EPT]) {
    const int len = AXIS == 0 ? n0 : (AXIS == 1 ? n1 : n2);
    const int stride = AXIS == 0 ? n1*n2 : (AXIS == 1 ? n2 : 1);
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int k = lane + e*tpr;
        if (k < nk) {
            int base, freq;
            if (AXIS == 0) {
                base = ky[e]*n2 + kz[e]; freq = kx[e];
            } else if (AXIS == 1) {
                base = kx[e]*n1*n2 + kz[e]; freq = ky[e];
            } else {
                base = (kx[e]*n1 + ky[e])*n2; freq = kz[e];
            }
            double ar = 0.0, ai = 0.0;
            int m = 0;
            for (int j = 0; j < len; ++j) {
                const lrx_c2 v = row[base + j*stride];
                const lrx_c2 w = tw[m];
                const double wi = sign*w.y;
                ar += v.x*w.x - v.y*wi;
                ai += v.x*wi + v.y*w.x;
                m += freq;
                if (m >= len) m -= len;
            }
            out[e].x = ar; out[e].y = ai;
        }
    }
}

template <int EPT>
__device__ __forceinline__ void lrx_write(
    lrx_c2* row, int nk, int lane, int tpr, const lrx_c2 (&v)[EPT]) {
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int k = lane + e*tpr;
        if (k < nk) row[k] = v[e];
    }
}

template <int EPT>
__device__ __forceinline__ void lrx_transform_resident(
    lrx_c2* row, const lrx_c2* twx, const lrx_c2* twy,
    const lrx_c2* twz, int nk, int n0, int n1, int n2, double sign) {
    const int lane = threadIdx.x;
    const int tpr = blockDim.x;
    int kx[EPT], ky[EPT], kz[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int k = lane + e*tpr;
        if (k >= nk) k = 0;
        kz[e] = k % n2; k /= n2;
        ky[e] = k % n1; kx[e] = k / n1;
    }
    lrx_c2 v[EPT];
#define LRX_AXIS(A,TW,LEN)                                             \
    if ((LEN) > 1) {                                                   \
        lrx_pass<EPT,A>(row,TW,nk,lane,tpr,n0,n1,n2,kx,ky,kz,sign,v); \
        __syncthreads();                                               \
        lrx_write<EPT>(row,nk,lane,tpr,v);                             \
        __syncthreads();                                               \
    }
    LRX_AXIS(2,twz,n2)
    LRX_AXIS(1,twy,n1)
    LRX_AXIS(0,twx,n0)
#undef LRX_AXIS
}

template <int EPT>
__device__ __forceinline__ void lrx_transform_resident_pair(
    lrx_c2* arow,lrx_c2* brow,const lrx_c2* twx,const lrx_c2* twy,
    const lrx_c2* twz,int nk,int n0,int n1,int n2,double sign) {
    const int lane=threadIdx.x,tpr=blockDim.x;
    int kx[EPT],ky[EPT],kz[EPT];
#pragma unroll
    for (int e=0;e<EPT;++e) {
        int k=lane+e*tpr;
        if (k>=nk) k=0;
        kz[e]=k%n2;k/=n2;ky[e]=k%n1;kx[e]=k/n1;
    }
    lrx_c2 av[EPT],bv[EPT];
    // The two banks have no dependency before the R-space multiply.  Sharing
    // each barrier avoids serializing identical transform schedules.
#define LRX_PAIR_AXIS(A,TW,LEN)                                      \
    if ((LEN)>1) {                                                   \
        lrx_pass<EPT,A>(arow,TW,nk,lane,tpr,n0,n1,n2,kx,ky,kz,sign,av); \
        lrx_pass<EPT,A>(brow,TW,nk,lane,tpr,n0,n1,n2,kx,ky,kz,sign,bv); \
        __syncthreads();                                             \
        lrx_write<EPT>(arow,nk,lane,tpr,av);                         \
        lrx_write<EPT>(brow,nk,lane,tpr,bv);                         \
        __syncthreads();                                             \
    }
    LRX_PAIR_AXIS(2,twz,n2)
    LRX_PAIR_AXIS(1,twy,n1)
    LRX_PAIR_AXIS(0,twx,n0)
#undef LRX_PAIR_AXIS
}

template <int EPT>
__device__ __forceinline__ void lrx_resident_body(
    const lrx_c2* ain, const lrx_c2* bin, lrx_c2* uout,
    long long rows, int ns, int nk, int n0, int n1, int n2, int sp,
    double scale, unsigned long long perm_l, unsigned long long phase_l,
    unsigned long long perm_r, unsigned long long phase_r, lrx_c2* sm) {
    const int tpr = blockDim.x;
    const int rb = blockDim.y;
    const int tid = threadIdx.y*tpr + threadIdx.x;
    const int nthr = tpr*rb;
    const long long r0 = (long long)blockIdx.x*rb;
    const int ntw = n0+n1+n2;
    const int banks = ns == 1 ? 2 : 3;
    lrx_c2* abank = sm;
    lrx_c2* bbank = abank + (long long)rb*sp;
    lrx_c2* accum = ns == 1 ? abank : bbank + (long long)rb*sp;
    lrx_c2* tw = sm + (long long)banks*rb*sp;

    // O(axis) rings serve every row and spin pair; runtime lengths cover
    // primes without a radix table or per-grid compilation.
    for (int i = tid; i < ntw; i += nthr) {
        int m, len;
        if (i < n0) { m=i; len=n0; }
        else if (i < n0+n1) { m=i-n0; len=n1; }
        else { m=i-n0-n1; len=n2; }
        double s,c; sincospi(-2.0*(double)m/(double)len,&s,&c);
        tw[i].x=c; tw[i].y=s;
    }
    __syncthreads();
    const lrx_c2* twx=tw;
    const lrx_c2* twy=tw+n0;
    const lrx_c2* twz=tw+n0+n1;

    for (int a = 0; a < ns; ++a) {
        const int ap = (perm_l >> (4*a)) & 15;
        const int pc_l = (phase_l >> (2*a)) & 3;
        for (int b = 0; b < ns; ++b) {
            const int bp = (perm_r >> (4*b)) & 15;
            const int pc = (pc_l + ((phase_r >> (2*b)) & 3)) & 3;
            const int tile = rb*nk;
            for (int i = tid; i < tile; i += nthr) {
                const int k = i/rb;
                const int j = i-k*rb;
                const long long row = r0+j;
                lrx_c2 av={0.0,0.0}, bv={0.0,0.0};
                if (row < rows) {
                    const long long base = (long long)k*ns*rows*ns;
                    av=ain[base + (long long)a*rows*ns + row*ns + b];
                    bv=bin[base + (long long)ap*rows*ns + row*ns + bp];
                }
                abank[(long long)j*sp+k]=av;
                bbank[(long long)j*sp+k]=bv;
            }
            __syncthreads();
            lrx_c2* ar=abank+(long long)threadIdx.y*sp;
            lrx_c2* br=bbank+(long long)threadIdx.y*sp;
            lrx_transform_resident_pair<EPT>(
                ar,br,twx,twy,twz,nk,n0,n1,n2,-1.0);
            for (int k=threadIdx.x;k<nk;k+=tpr) {
                const lrx_c2 ac={ar[k].x,-ar[k].y};
                const lrx_c2 term=lrx_phase(lrx_mul(ac,br[k]),pc);
                lrx_c2* dst=accum+(long long)threadIdx.y*sp+k;
                if (a==0&&b==0) *dst=term;
                else {dst->x+=term.x;dst->y+=term.y;}
            }
            __syncthreads();
        }
    }

    lrx_c2* outrow=accum+(long long)threadIdx.y*sp;
    lrx_transform_resident<EPT>(outrow,twx,twy,twz,nk,n0,n1,n2,1.0);
    for (int i = tid; i < rb*nk; i += nthr) {
        const int k=i/rb;
        const int j=i-k*rb;
        const long long row=r0+j;
        if (row < rows) {
            const lrx_c2 v=accum[(long long)j*sp+k];
            uout[(long long)k*rows+row]={v.x*scale,v.y*scale};
        }
    }
}

// Global-workspace line transforms are the coverage arm, not the small-grid
// throughput arm.  One block owns a row for the entire kernel, so every
// barrier is row-local and no grid synchronization or collective is needed.
__device__ __forceinline__ void lrx_axis_pair(
    lrx_c2* wa, lrx_c2* wb, int n0, int n1, int n2, int axis,
    double sign, lrx_c2* lines_a, lrx_c2* lines_b, const lrx_c2* tw) {
    const int lane=threadIdx.x&31;
    const int warp=threadIdx.x>>5;
    const int nwarp=blockDim.x>>5;
    const int len=axis==0?n0:(axis==1?n1:n2);
    const int nline=axis==0?n1*n2:(axis==1?n0*n2:n0*n1);
    lrx_c2* line_a=lines_a+warp*24;
    lrx_c2* line_b=lines_b+warp*24;
    // One warp owns one line at a time.  Line lengths never exceed 24, so
    // no inter-warp dependency exists and inactive lanes stay spectators.
    for (int line=warp; line<nline; line+=nwarp) {
        int base,stride;
        if (axis==0) { base=line; stride=n1*n2; }
        else if (axis==1) {
            const int x=line/n2,z=line-x*n2;
            base=x*n1*n2+z; stride=n2;
        } else { base=line*n2; stride=1; }
        if (lane < len) {
            line_a[lane]=wa[base+lane*stride];
            line_b[lane]=wb[base+lane*stride];
        }
        __syncwarp();
        lrx_c2 oa={0.0,0.0},ob={0.0,0.0};
        if (lane < len) {
            int m=0;
            for (int j=0;j<len;++j) {
                const lrx_c2 w=tw[m];
                const double wi=sign*w.y;
                const lrx_c2 va=line_a[j],vb=line_b[j];
                oa.x += va.x*w.x-va.y*wi; oa.y += va.x*wi+va.y*w.x;
                ob.x += vb.x*w.x-vb.y*wi; ob.y += vb.x*wi+vb.y*w.x;
                m += lane; if (m >= len) m -= len;
            }
        }
        __syncwarp();
        if (lane < len) {
            wa[base+lane*stride]=oa;
            wb[base+lane*stride]=ob;
        }
        __syncwarp();
    }
    // The next axis consumes every line written by this axis.
    __syncthreads();
}

__device__ __forceinline__ void lrx_axis_one(
    lrx_c2* work, int n0, int n1, int n2, int axis,
    double sign, lrx_c2* lines, const lrx_c2* tw) {
    const int lane=threadIdx.x&31;
    const int warp=threadIdx.x>>5;
    const int nwarp=blockDim.x>>5;
    const int len=axis==0?n0:(axis==1?n1:n2);
    const int nline=axis==0?n1*n2:(axis==1?n0*n2:n0*n1);
    lrx_c2* line=lines+warp*24;
    for (int li=warp;li<nline;li+=nwarp) {
        int base,stride;
        if (axis==0) { base=li; stride=n1*n2; }
        else if (axis==1) {
            const int x=li/n2,z=li-x*n2;
            base=x*n1*n2+z; stride=n2;
        } else { base=li*n2; stride=1; }
        if (lane < len) line[lane]=work[base+lane*stride];
        __syncwarp();
        lrx_c2 out={0.0,0.0};
        if (lane < len) {
            int m=0;
            for (int j=0;j<len;++j) {
                const lrx_c2 w=tw[m],v=line[j];
                const double wi=sign*w.y;
                out.x += v.x*w.x-v.y*wi; out.y += v.x*wi+v.y*w.x;
                m += lane; if (m >= len) m -= len;
            }
        }
        __syncwarp();
        if (lane < len) work[base+lane*stride]=out;
        __syncwarp();
    }
    __syncthreads();
}

extern "C" __global__ void lrx_pair_stage1(
    const lrx_c2* ain, const lrx_c2* bin, lrx_c2* wa_all,
    lrx_c2* wb_all, lrx_c2* tmp, long long rows, int ns, int nk,
    int n0, int n1, int n2, unsigned long long perm_l,
    unsigned long long phase_l, unsigned long long perm_r,
    unsigned long long phase_r) {
    extern __shared__ lrx_c2 sm[];
    const long long row=blockIdx.x;
    if (row >= rows) return;
    const int nwarp=blockDim.x>>5;
    lrx_c2* line_a=sm;
    lrx_c2* line_b=line_a+nwarp*24;
    lrx_c2* tw=line_b+nwarp*24;
    for (int i=threadIdx.x;i<n0+n1+n2;i+=blockDim.x) {
        int m,len;
        if (i<n0) {m=i;len=n0;}
        else if (i<n0+n1) {m=i-n0;len=n1;}
        else {m=i-n0-n1;len=n2;}
        double s,c;sincospi(-2.0*(double)m/(double)len,&s,&c);
        tw[i]={c,s};
    }
    __syncthreads();
    lrx_c2* wa=wa_all+row*nk;
    lrx_c2* wb=wb_all+row*nk;
    for (int a=0;a<ns;++a) {
        const int ap=(perm_l>>(4*a))&15;
        const int pcl=(phase_l>>(2*a))&3;
        for (int b=0;b<ns;++b) {
            const int bp=(perm_r>>(4*b))&15;
            const int pc=(pcl+((phase_r>>(2*b))&3))&3;
            for (int k=threadIdx.x;k<nk;k+=blockDim.x) {
                const long long base=(long long)k*ns*rows*ns;
                wa[k]=ain[base+(long long)a*rows*ns+row*ns+b];
                wb[k]=bin[base+(long long)ap*rows*ns+row*ns+bp];
            }
            __syncthreads();
            lrx_axis_pair(wa,wb,n0,n1,n2,2,-1.0,line_a,line_b,tw+n0+n1);
            lrx_axis_pair(wa,wb,n0,n1,n2,1,-1.0,line_a,line_b,tw+n0);
            lrx_axis_pair(wa,wb,n0,n1,n2,0,-1.0,line_a,line_b,tw);
            for (int k=threadIdx.x;k<nk;k+=blockDim.x) {
                lrx_c2 ac={wa[k].x,-wa[k].y};
                lrx_c2 term=lrx_phase(lrx_mul(ac,wb[k]),pc);
                lrx_c2* dst=tmp+row*nk+k;
                if (a==0 && b==0) *dst=term;
                else {dst->x+=term.x;dst->y+=term.y;}
            }
            __syncthreads();
        }
    }
}

extern "C" __global__ void lrx_pair_stage2(
    lrx_c2* tmp, lrx_c2* uout, long long rows, int nk,
    int n0, int n1, int n2, double scale) {
    extern __shared__ lrx_c2 sm[];
    const long long row=blockIdx.x;
    if (row >= rows) return;
    const int nwarp=blockDim.x>>5;
    lrx_c2* line=sm;
    lrx_c2* tw=line+nwarp*24;
    for (int i=threadIdx.x;i<n0+n1+n2;i+=blockDim.x) {
        int m,len;
        if (i<n0) {m=i;len=n0;}
        else if (i<n0+n1) {m=i-n0;len=n1;}
        else {m=i-n0-n1;len=n2;}
        double s,c;sincospi(-2.0*(double)m/(double)len,&s,&c);
        tw[i]={c,s};
    }
    __syncthreads();
    lrx_c2* work=tmp+row*nk;
    lrx_axis_one(work,n0,n1,n2,2,1.0,line,tw+n0+n1);
    lrx_axis_one(work,n0,n1,n2,1,1.0,line,tw+n0);
    lrx_axis_one(work,n0,n1,n2,0,1.0,line,tw);
    for (int k=threadIdx.x;k<nk;k+=blockDim.x) {
        const lrx_c2 v=work[k];
        uout[(long long)k*rows+row]={v.x*scale,v.y*scale};
    }
}

#define LRX_ENTRY(N)                                                    \
extern "C" __global__ __launch_bounds__(512)                          \
void lrx_pair_resident_e##N(                                           \
    const lrx_c2* a,const lrx_c2* b,lrx_c2* u,long long rows,int ns,   \
    int nk,int n0,int n1,int n2,int sp,double scale,                   \
    unsigned long long pl,unsigned long long hl,                       \
    unsigned long long pr,unsigned long long hr) {                     \
    extern __shared__ lrx_c2 sm[];                                     \
    lrx_resident_body<N>(a,b,u,rows,ns,nk,n0,n1,n2,sp,scale,           \
                         pl,hl,pr,hr,sm);                               \
}
LRX_ENTRY(1) LRX_ENTRY(2) LRX_ENTRY(3) LRX_ENTRY(4)
LRX_ENTRY(5) LRX_ENTRY(6) LRX_ENTRY(7) LRX_ENTRY(8)
LRX_ENTRY(9) LRX_ENTRY(10) LRX_ENTRY(11) LRX_ENTRY(12)
LRX_ENTRY(13) LRX_ENTRY(14) LRX_ENTRY(15) LRX_ENTRY(16)
#undef LRX_ENTRY
)__lrx__";

static std::mutex g_mu;

struct KernelArms {
    CUfunction resident[kEptMax] = {nullptr};
    CUfunction stage1 = nullptr;
    CUfunction stage2 = nullptr;
    int smem_max = 0;
};

static ffi::Error ensure_kernels(const KernelArms** out) {
    static std::map<CUcontext, KernelArms> cache;
    static std::map<CUcontext, std::string> fail_cache;
    const DriverApi& api = driver_api();
    if (!api.ok) {
        return fail("driver-api resolve", api.err.empty()
            ? "required CUDA driver entry points are unavailable" : api.err);
    }
    CUcontext ctx = nullptr;
    CUresult cr = api.CtxGetCurrent(&ctx);
    if (cr != CUDA_SUCCESS) {
        return fail("cuCtxGetCurrent", "CUresult=" +
                    std::to_string(static_cast<int>(cr)) + " (" +
                    cu_err(cr) + ")");
    }
    if (ctx == nullptr) {
        LRX_CUDA_CHECK(cudaFree(nullptr), "context bind (cudaFree(0))");
        cr = api.CtxGetCurrent(&ctx);
        if (cr != CUDA_SUCCESS || ctx == nullptr) {
            return fail("cuCtxGetCurrent after context bind",
                        cr == CUDA_SUCCESS ? "no current CUDA context"
                                           : cu_err(cr));
        }
    }
    auto it = cache.find(ctx);
    if (it != cache.end()) { *out=&it->second; return ffi::Error::Success(); }
    auto fit = fail_cache.find(ctx);
    if (fit != fail_cache.end()) {
        return fail("kernel build (cached failure, NVRTC not re-run)",fit->second);
    }
    auto fail_sticky = [&](const char* where,const std::string& detail) {
        fail_cache.emplace(ctx,std::string(where)+" -- "+detail);
        return fail(where,detail);
    };

    int dev=0,cc_major=0,cc_minor=0,smem_optin=0;
    LRX_CUDA_CHECK(cudaGetDevice(&dev),"cudaGetDevice");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
        &cc_major,cudaDevAttrComputeCapabilityMajor,dev),
        "query compute capability (major)");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
        &cc_minor,cudaDevAttrComputeCapabilityMinor,dev),
        "query compute capability (minor)");
    LRX_CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev),
        "query max dynamic shared memory per block (opt-in)");

    std::vector<char> image;
    const void* image_data = nullptr;
    size_t image_size = 0;
    bool cubin = true;
    bool used_prebuilt = false;
    double nvrtc_compile_ms = 0.0;
#if LORRAX_FFI_HAVE_PREBUILT_SM80
    if (cc_major == 8 && cc_minor == 0) {
        const auto* begin = _binary_lrx_conv_kpair_sm80_cubin_start;
        const auto* end = _binary_lrx_conv_kpair_sm80_cubin_end;
        const uintptr_t begin_addr = reinterpret_cast<uintptr_t>(begin);
        const uintptr_t end_addr = reinterpret_cast<uintptr_t>(end);
        if (end_addr <= begin_addr) {
            return fail_sticky("embedded sm_80 cubin",
                               "linker image has empty or inverted bounds");
        }
        image_data = begin;
        image_size = static_cast<size_t>(end_addr - begin_addr);
        used_prebuilt = true;
    }
#endif
    if (!used_prebuilt) {
        if (mklpin::announce_here()) {
            std::fprintf(stderr,
                "[conv_kpair] AOT_ARCH_MISS: no embedded cubin for sm_%d%d; "
                "paying runtime NVRTC compilation now\n",
                cc_major, cc_minor);
        }
        nvrtcProgram prog=nullptr;
        nvrtcResult nr=nvrtcCreateProgram(
            &prog,kKernelSrc,"lrx_conv_kpair.cu",0,nullptr,nullptr);
        if (nr != NVRTC_SUCCESS) {
            return fail_sticky("nvrtcCreateProgram",nvrtcGetErrorString(nr));
        }
        char arch[64];
        std::snprintf(arch,sizeof(arch),"--gpu-architecture=sm_%d%d",
                      cc_major,cc_minor);
        // Only architecture is explicit; value parity permits default FMA.
        const char* opts[]={arch};
        const auto nvrtc_t0 = std::chrono::steady_clock::now();
        nr=nvrtcCompileProgram(prog,1,opts);
        nvrtc_compile_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-nvrtc_t0).count();
        if (nr != NVRTC_SUCCESS) {
            size_t log_size=0; std::string log;
            if (nvrtcGetProgramLogSize(prog,&log_size)==NVRTC_SUCCESS &&
                log_size>1) {
                log.resize(log_size); nvrtcGetProgramLog(prog,&log[0]);
            }
            nvrtcDestroyProgram(&prog);
            return fail_sticky("nvrtcCompileProgram",
                std::string(nvrtcGetErrorString(nr))+" -- "+log);
        }
        nvrtcResult image_result;
        if (nvrtcGetCUBINSize(prog,&image_size)==NVRTC_SUCCESS && image_size>0) {
            image.resize(image_size);
            image_result=nvrtcGetCUBIN(prog,image.data());
        } else if (nvrtcGetPTXSize(prog,&image_size)==NVRTC_SUCCESS &&
                   image_size>0) {
            image.resize(image_size);
            image_result=nvrtcGetPTX(prog,image.data());
            cubin=false;
        } else {
            image_result=NVRTC_ERROR_INTERNAL_ERROR;
        }
        nvrtcDestroyProgram(&prog);
        if (image_result != NVRTC_SUCCESS || image.empty()) {
            return fail_sticky("nvrtc get cubin/ptx",
                               nvrtcGetErrorString(image_result));
        }
        image_data=image.data();
    }
    CUmodule module=nullptr;
    cr=api.ModuleLoadData(&module,image_data);
    if (cr != CUDA_SUCCESS) return fail_sticky("cuModuleLoadData",cu_err(cr));

    KernelArms arms;
    std::vector<std::pair<CUfunction*,std::string>> functions;
    for (int e=1;e<=kEptMax;++e) {
        char name[64]; std::snprintf(name,sizeof(name),"lrx_pair_resident_e%d",e);
        functions.emplace_back(&arms.resident[e-1],name);
    }
    functions.emplace_back(&arms.stage1,"lrx_pair_stage1");
    functions.emplace_back(&arms.stage2,"lrx_pair_stage2");
    for (auto& item:functions) {
        cr=api.ModuleGetFunction(item.first,module,item.second.c_str());
        if (cr != CUDA_SUCCESS) {
            api.ModuleUnload(module);
            return fail_sticky("cuModuleGetFunction",item.second+": "+cu_err(cr));
        }
        if (smem_optin>49152) {
            cr=api.FuncSetAttribute(
                *item.first,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem_optin);
            if (cr != CUDA_SUCCESS) {
                api.ModuleUnload(module);
                return fail_sticky(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
                    item.second+": requested "+std::to_string(smem_optin)+
                    " B: CUresult="+std::to_string(static_cast<int>(cr))+
                    " ("+cu_err(cr)+")");
            }
        }
    }
    arms.smem_max=std::max(smem_optin,49152);
    if (used_prebuilt && log_enabled()) {
        std::fprintf(stderr,
            "[conv_kpair] AOT_SM80_HIT: loaded embedded sm_80 cubin "
            "(%zu B, device %d, %d resident + 2 staged arms); NVRTC skipped; "
            "checked dynamic SMEM max %d B\n",
            image_size,dev,kEptMax,arms.smem_max);
    } else if (log_enabled()) {
        std::fprintf(stderr,
            "[conv_kpair] NVRTC compiled sm_%d%d in %.3f ms "
            "(%zu B %s, device %d, "
            "%d resident + 2 staged arms); checked dynamic SMEM max %d B\n",
            cc_major,cc_minor,nvrtc_compile_ms,image_size,
            cubin?"cubin":"ptx",dev,
            kEptMax,arms.smem_max);
    }
    auto inserted=cache.emplace(ctx,arms);
    *out=&inserted.first->second;
    return ffi::Error::Success();
}

struct LaunchCfg {
    int rb=1,tpr=1,ept=1,sp=1,smem=0;
    long long minimum_smem=0;
};

static bool plan_resident(int nk,int n0,int n1,int n2,int ns,int smem_max,
                          LaunchCfg* cfg,std::string* why) {
    const int banks=ns==1?2:3;
    const int sp=nk|1;
    const int ntw=n0+n1+n2;
    auto bytes_for=[&](long long rb) {
        return 16LL*(banks*rb*sp+ntw);
    };
    cfg->minimum_smem=bytes_for(1);
    long long budget=kSmemPreferred;
    if (bytes_for(1)>budget) budget=smem_max;
    long long rb=0;
    for (long long r=kRowsMax;r>=1;--r) {
        if (bytes_for(r)<=budget) {rb=r;break;}
    }
    if (rb<1) {
        std::ostringstream os;
        os << "resident arm needs 16*(banks=" << banks << "*rows=1*odd_stride="
           << sp << "+twiddles=" << ntw << ")=" << bytes_for(1)
           << " B, but the checked device ceiling is " << smem_max << " B";
        *why=os.str(); return false;
    }
    long long tpr=std::min<long long>(nk,
        std::max<long long>(32,(nk+kEptPreferred-1)/kEptPreferred));
    if (nk>=32) tpr=std::min<long long>(512,((tpr+31)/32)*32);
    while (tpr*rb>kBlockMax && rb>1) --rb;
    if (tpr*rb>kBlockMax) { tpr=std::min<long long>(nk,kBlockMax); rb=1; }
    const long long ept=(nk+tpr-1)/tpr;
    if (ept>kEptMax) {
        *why="resident arm ept="+std::to_string(ept)+" exceeds compiled max "+
             std::to_string(kEptMax);
        return false;
    }
    cfg->rb=static_cast<int>(rb); cfg->tpr=static_cast<int>(tpr);
    cfg->ept=static_cast<int>(ept); cfg->sp=sp;
    cfg->smem=static_cast<int>(bytes_for(rb));
    return true;
}

static bool pack_attrs(ffi::Span<const int64_t> perm,
                       ffi::Span<const int64_t> phase,int64_t ns,
                       const char* side,unsigned long long* pp,
                       unsigned long long* hp,std::string* why) {
    if (perm.size()!=static_cast<size_t>(ns) ||
        phase.size()!=static_cast<size_t>(ns)) {
        std::ostringstream os;
        os << side << " perm/phase lengths " << perm.size() << "/"
           << phase.size() << " != ns=" << ns;
        *why=os.str(); return false;
    }
    unsigned seen=0; *pp=0; *hp=0;
    for (int64_t i=0;i<ns;++i) {
        if (perm[i]<0 || perm[i]>=ns || phase[i]<0 || phase[i]>3) {
            std::ostringstream os;
            os << side << " attribute entry " << i << " has perm=" << perm[i]
               << ", phase_code=" << phase[i]
               << " (perm must be [0,ns), phase code [0,3])";
            *why=os.str(); return false;
        }
        if (seen&(1u<<perm[i])) {
            *why=std::string(side)+" perm is not one-to-one"; return false;
        }
        seen|=1u<<perm[i];
        *pp|=static_cast<unsigned long long>(perm[i])<<(4*i);
        *hp|=static_cast<unsigned long long>(phase[i])<<(2*i);
    }
    return true;
}

static bool overlaps(const void* p,size_t pn,const void* q,size_t qn) {
    const auto pb=reinterpret_cast<uintptr_t>(p);
    const auto qb=reinterpret_cast<uintptr_t>(q);
    return pb<qb+qn && qb<pb+pn;
}

static ffi::Error ConvKPairDispatch(
    cudaStream_t stream,ffi::ScratchAllocator scratch,
    ffi::AnyBuffer A,ffi::AnyBuffer B,ffi::Result<ffi::AnyBuffer> U,
    int64_t nkx,int64_t nky,int64_t nkz,double scale,int64_t requested_arm,
    ffi::Span<const int64_t> perm_l,ffi::Span<const int64_t> phase_l,
    ffi::Span<const int64_t> perm_r,ffi::Span<const int64_t> phase_r) {
    if (A.element_type()!=ffi::DataType::C128 ||
        B.element_type()!=ffi::DataType::C128 ||
        U->element_type()!=ffi::DataType::C128) {
        return fail("contract","complex128 only; use the XLA reference chain",
                    ffi::ErrorCode::kInvalidArgument);
    }
    auto ad=A.dimensions(),bd=B.dimensions(),ud=U->dimensions();
    if (ad.size()!=7 || bd.size()!=7 || ud.size()!=5) {
        std::ostringstream os;
        os << "expected ranks A=7, B=7, U=5; got " << ad.size() << ","
           << bd.size() << "," << ud.size();
        return fail("contract",os.str(),ffi::ErrorCode::kInvalidArgument);
    }
    for (size_t i=0;i<7;++i) {
        if (ad[i]!=bd[i]) return fail("contract","A/B shapes differ",
                                      ffi::ErrorCode::kInvalidArgument);
    }
    const int64_t ns=ad[3],d0=ad[4],d1=ad[5];
    const int64_t nk=nkx*nky*nkz,rows=d0*d1;
    bool shape_ok=nkx>=1&&nky>=1&&nkz>=1&&
        ad[0]==nkx&&ad[1]==nky&&ad[2]==nkz&&ad[6]==ns&&
        ud[0]==nkx&&ud[1]==nky&&ud[2]==nkz&&ud[3]==d0&&ud[4]==d1;
    if (!shape_ok || ns<1 || ns>4 || requested_arm<0 || requested_arm>2) {
        std::ostringstream os;
        os << "shape/attribute mismatch: A=(";
        for (size_t i=0;i<ad.size();++i) os << (i?",":"") << ad[i];
        os << "), U=(";
        for (size_t i=0;i<ud.size();++i) os << (i?",":"") << ud[i];
        os << "), kgrid=(" << nkx << "," << nky << "," << nkz
           << "), requested_arm=" << requested_arm;
        return fail("contract",os.str(),ffi::ErrorCode::kInvalidArgument);
    }
    if (nkx>kAxisMax || nky>kAxisMax || nkz>kAxisMax) {
        std::ostringstream os;
        os << "shape=(" << nkx << "," << nky << "," << nkz << ",ns="
           << ns << ",d0=" << d0 << ",d1=" << d1
           << ") has an axis outside [1," << kAxisMax
           << "]; native ceiling is the validated axis envelope; fallback="
              "XLA reference chain";
        return fail("named refusal",os.str(),ffi::ErrorCode::kInvalidArgument);
    }
    unsigned long long pl=0,hl=0,pr=0,hr=0; std::string attr_why;
    if (!pack_attrs(perm_l,phase_l,ns,"left",&pl,&hl,&attr_why) ||
        !pack_attrs(perm_r,phase_r,ns,"right",&pr,&hr,&attr_why)) {
        return fail("gamma attributes",attr_why,ffi::ErrorCode::kInvalidArgument);
    }
    if (rows==0) return ffi::Error::Success();
    if (rows>2147483647LL) {
        return fail("named refusal","d0*d1="+std::to_string(rows)+
                    " exceeds grid.x; fallback=XLA reference chain",
                    ffi::ErrorCode::kInvalidArgument);
    }
    const size_t a_bytes=static_cast<size_t>(nk)*ns*rows*ns*16;
    const size_t u_bytes=static_cast<size_t>(nk)*rows*16;
    const void* a_ptr=A.untyped_data(); const void* b_ptr=B.untyped_data();
    void* u_ptr=U->untyped_data();
    // U is rank-reduced and cannot safely reuse either source: all ns^2 input
    // rows remain live until their R-space sum completes.  Python declares no
    // alias; reject overlap here so a future wrapper cannot change that fact.
    if (overlaps(u_ptr,u_bytes,a_ptr,a_bytes) ||
        overlaps(u_ptr,u_bytes,b_ptr,a_bytes)) {
        return fail("alias contract","U overlaps A or B; no in-place form exists",
                    ffi::ErrorCode::kInvalidArgument);
    }

    const KernelArms* arms=nullptr; ffi::Error error=ffi::Error::Success();
    {
        std::lock_guard<std::mutex> lock(g_mu);
        error=ensure_kernels(&arms);
    }
    if (!error.success()) return error;
    LaunchCfg cfg; std::string resident_why;
    const bool resident_ok=plan_resident(
        static_cast<int>(nk),static_cast<int>(nkx),static_cast<int>(nky),
        static_cast<int>(nkz),static_cast<int>(ns),arms->smem_max,
        &cfg,&resident_why);
    const bool use_resident=resident_ok && requested_arm!=2;

    if (use_resident) {
        const auto* ap=static_cast<const double*>(a_ptr);
        const auto* bp=static_cast<const double*>(b_ptr);
        auto* up=static_cast<double*>(u_ptr);
        long long ar=rows; int ans=static_cast<int>(ns),ank=static_cast<int>(nk);
        int an0=static_cast<int>(nkx),an1=static_cast<int>(nky);
        int an2=static_cast<int>(nkz),asp=cfg.sp; double ascale=scale;
        void* args[]={(void*)&ap,(void*)&bp,(void*)&up,&ar,&ans,&ank,&an0,&an1,
                      &an2,&asp,&ascale,&pl,&hl,&pr,&hr};
        const int64_t blocks=(rows+cfg.rb-1)/cfg.rb;
        CUresult cr=driver_api().LaunchKernel(
            arms->resident[cfg.ept-1],static_cast<unsigned>(blocks),1,1,
            static_cast<unsigned>(cfg.tpr),static_cast<unsigned>(cfg.rb),1,
            static_cast<unsigned>(cfg.smem),reinterpret_cast<CUstream>(stream),
            args,nullptr);
        if (cr!=CUDA_SUCCESS) return fail("resident cuLaunchKernel",cu_err(cr));
        if (log_enabled()) {
            static std::atomic<bool> once{false};
            if (!once.exchange(true)) std::fprintf(stderr,
                "[conv_kpair] first call arm=resident shape=(%lld,%lld,%lld,"
                "ns=%lld,d0=%lld,d1=%lld) rows=%lld scale=%.9e rb=%d tpr=%d "
                "ept=%d odd_stride=%d smem=%d/%d B\n",
                (long long)nkx,(long long)nky,(long long)nkz,(long long)ns,
                (long long)d0,(long long)d1,(long long)rows,scale,cfg.rb,cfg.tpr,
                cfg.ept,cfg.sp,cfg.smem,arms->smem_max);
        }
        return ffi::Error::Success();
    }

    // Stage 1 reads both full-rank operands and writes the reduced R-space
    // temporary; stage 2 reads that temporary and writes U.  The two rank-5
    // work banks are transform workspace, allocated through XLA rather than a
    // private arena.  Their extent is O(local nk*d0*d1), never replicated.
    auto wa_opt=scratch.Allocate(u_bytes);
    auto wb_opt=scratch.Allocate(u_bytes);
    auto tmp_opt=scratch.Allocate(u_bytes);
    if (!wa_opt.has_value() || !wb_opt.has_value() || !tmp_opt.has_value()) {
        const int banks=ns==1?2:3;
        const long long resident_bytes=16LL*(banks*(nk|1)+nkx+nky+nkz);
        std::ostringstream os;
        os << "shape=(" << nkx << "," << nky << "," << nkz << ",ns="
           << ns << ",d0=" << d0 << ",d1=" << d1 << ") bytes: A+B="
           << 2ULL*a_bytes << ", U=" << u_bytes << ", staged scratch=3*"
           << u_bytes << "=" << 3ULL*u_bytes << "; resident minimum=16*("
           << banks << "*(nk|1=" << (nk|1) << ")+axes="
           << nkx+nky+nkz << ")=" << resident_bytes
           << " B, checked SMEM ceiling=" << arms->smem_max
           << " B; XLA scratch allocation refused; fallback=XLA reference chain";
        return fail("named refusal",os.str(),ffi::ErrorCode::kResourceExhausted);
    }
    const auto* ap=static_cast<const double*>(a_ptr);
    const auto* bp=static_cast<const double*>(b_ptr);
    auto* wa=static_cast<double*>(*wa_opt);
    auto* wb=static_cast<double*>(*wb_opt);
    auto* tmp=static_cast<double*>(*tmp_opt);
    auto* up=static_cast<double*>(u_ptr);
    long long ar=rows; int ans=static_cast<int>(ns),ank=static_cast<int>(nk);
    int an0=static_cast<int>(nkx),an1=static_cast<int>(nky);
    int an2=static_cast<int>(nkz); double ascale=scale;
    void* args1[]={(void*)&ap,(void*)&bp,(void*)&wa,(void*)&wb,(void*)&tmp,
                   &ar,&ans,&ank,&an0,&an1,&an2,&pl,&hl,&pr,&hr};
    const unsigned stage_threads=256;
    const unsigned stage_warps=stage_threads/32;
    const unsigned stage1_smem=16U*(2U*stage_warps*kAxisMax+nkx+nky+nkz);
    CUresult cr=driver_api().LaunchKernel(
        arms->stage1,static_cast<unsigned>(rows),1,1,stage_threads,1,1,
        stage1_smem,reinterpret_cast<CUstream>(stream),args1,nullptr);
    if (cr!=CUDA_SUCCESS) return fail("two-stage stage1 cuLaunchKernel",cu_err(cr));
    void* args2[]={(void*)&tmp,(void*)&up,&ar,&ank,&an0,&an1,&an2,&ascale};
    const unsigned stage2_smem=16U*(stage_warps*kAxisMax+nkx+nky+nkz);
    cr=driver_api().LaunchKernel(
        arms->stage2,static_cast<unsigned>(rows),1,1,stage_threads,1,1,
        stage2_smem,reinterpret_cast<CUstream>(stream),args2,nullptr);
    if (cr!=CUDA_SUCCESS) return fail("two-stage stage2 cuLaunchKernel",cu_err(cr));
    if (log_enabled()) {
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) std::fprintf(stderr,
            "[conv_kpair] first call arm=two_stage shape=(%lld,%lld,%lld,ns=%lld,"
            "d0=%lld,d1=%lld) rows=%lld scale=%.9e scratch=3*%zu B; resident: %s\n",
            (long long)nkx,(long long)nky,(long long)nkz,(long long)ns,
            (long long)d0,(long long)d1,(long long)rows,scale,u_bytes,
            resident_ok?"bypassed by plan":resident_why.c_str());
    }
    return ffi::Error::Success();
}

}  // namespace lorrax_ffi::conv_kpair

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CufftConvKPairCudaFfi,lorrax_ffi::conv_kpair::ConvKPairDispatch,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ctx<xla::ffi::ScratchAllocator>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Ret<xla::ffi::AnyBuffer>()
        .Attr<int64_t>("nkx")
        .Attr<int64_t>("nky")
        .Attr<int64_t>("nkz")
        .Attr<double>("scale")
        .Attr<int64_t>("requested_arm")
        .Attr<xla::ffi::Span<const int64_t>>("perm_l")
        .Attr<xla::ffi::Span<const int64_t>>("phase_l")
        .Attr<xla::ffi::Span<const int64_t>>("perm_r")
        .Attr<xla::ffi::Span<const int64_t>>("phase_r"));
