// kbox_stage.cuh -- the k-grid box transform stage of the k-convolution family.
//
// The family's k-LEADING operands hold element (k, col) at k*ncols + col, flat k in C order
// (kz fastest).  Every mode transforms nk-long columns over the (NX, NY, NZ) k-grid, applies an
// R-space step (Mid) and transforms back.  This header owns that stage and its launch rule; the
// modes own their Load (stage/finish), Mid and Store.  One code path per shape, chosen once when
// the door builds (kbox_plan below), from the k-grid and device attributes:
//
//   single pass  tr adjacent columns resident per block in a padded bank, cp.async straight into
//                shared memory, transform3 (z, y, x cuFFTDx thread FFTs, the family's order), Mid,
//                transform3 back, Store.  One HBM pass.
//   split (2+1)  when fewer than two columns fit at two blocks per SM: plane passes over
//                (ky, kz) on 16-column tiles, and a register x-pencil pass that fuses
//                IFFT_x . Mid . FFT_x (a spin group's pencils share a block for a group Mid).
//
// Why (F4 measurements, A100, reports/new_fft_kernel/F4_kbox_fft.md): the contiguous width per k
// (tr columns x element size) decides the load efficiency, two single-buffered blocks per SM beat
// one double-buffered block for the conv modes, and the odd (padded) z-line and row strides remove
// the shared bank conflicts of power-of-two grids.  The values are layout-independent: each line
// sees the same cuFFTDx thread FFT on the same inputs in the same axis order as the resident
// family kernel, so the single-pass transform is bitwise with it.
//
// Two halves:
//   * device code (NVRTC or nvcc, sm_80+: cp.async, no TMA or clusters), compiled when
//     __CUDACC__ or __CUDACC_RTC__ is defined.  It needs cufftdx.hpp included first.  The
//     family embeds this file into its NVRTC program as a named header.
//   * host code (the launch rule), compiled by the host compiler of the .cc.
#pragma once

namespace lrx_kbox {

// ---- geometry of a padded bank row (host and device) ---------------------------------------
struct Geometry {
    int nx, ny, nz;
    int zp() const { return nz | 1; }                        // odd z-line
    long long row() const { return (long long)nx * ny * zp(); }
    long long rs() const { return row() | 1; }                // odd row stride
    long long pr() const { return ((long long)ny * zp()) | 1; }  // padded (ky, kz) plane (split arm)
};

// ---- host: the launch rule ------------------------------------------------------------------
#if !defined(__CUDACC_RTC__)
struct Plan {
    int arm;            // 0 single pass, 1 split
    int tr;             // single: columns per block (power of two >= 2); split: plane tile columns
    int threads;        // plane/single block size; the group pencil uses group * ty threads
    int ty;             // split, group > 1: group instances per pencil block
    long long smem;     // dynamic shared memory of the single-pass or plane block, bytes
};

// group: columns a block must hold together (1, or ns*ns for a spin group); n_operands: staged
// operands per column (2 when a mode loads two Green functions); elem: 16 (c128) or 8 (c64);
// optin_smem: cudaDevAttrMaxSharedMemoryPerBlockOptin.
// Single pass: tr = the smallest power of two giving 128-byte runs per k (8 c128 columns) and a
// line for every thread in every axis pass (tr * nk / max axis >= 256), capped by two
// single-buffered blocks per SM; below two columns the split arm.  Threads: 256, or 512 for a
// convolution (transforms = 2: inverse, Mid, forward) whose axis passes have >= 384 lines.
inline Plan kbox_plan(int nx, int ny, int nz, int group, int n_operands, int elem, long long optin_smem,
                      int transforms = 1) {
    const Geometry g{nx, ny, nz};
    int threads = 256;
    const long long per_col = (long long)n_operands * group * g.rs() * elem;
    const long long lines = (long long)nx * ny * nz / (nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz));
    int want = 128 / elem;
    while ((long long)want * group * lines < threads) want *= 2;
    int cap = 1;
    while (2LL * cap * per_col <= optin_smem / 2) cap *= 2;
    const int tr = want < cap ? want : cap;
    if (transforms == 2 && (long long)tr * group * lines >= 384) threads = 512;
    if (tr >= 2) return Plan{0, tr, threads, 0, tr * per_col};
    threads = 256;
    const int tp = 16;
    return Plan{1, tp, threads, 8, (long long)n_operands * tp * g.pr() * elem};
}
#endif

}  // namespace lrx_kbox

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
namespace lrx_kbox {

template <int NX, int NY, int NZ>
struct Geo {
    static constexpr int NK = NX * NY * NZ;
    static constexpr int ZP = NZ | 1;
    static constexpr int RS = (NX * NY * ZP) | 1;
    static constexpr int PR = (NY * ZP) | 1;
    __device__ static constexpr int at(int k) { return (k / NZ) * ZP + k % NZ; }   // bank offset of flat k
    __device__ static constexpr int plane_at(int p) { return (p / NZ) * ZP + p % NZ; }  // p = ky*NZ + kz
};

// cp.async of one element (8 or 16 bytes), global -> shared; commit; wait for all.
template <int BYTES>
__device__ __forceinline__ void cp_async(void* smem, const void* gmem) {
    static_assert(BYTES == 8 || BYTES == 16, "cp.async element size");
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(s), "l"(gmem), "n"(BYTES));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

template <int N, int Arch, cufftdx::fft_direction Dir, class C>
using ThreadFFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<decltype(C::x)>() +
                           cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Dir>() +
                           cufftdx::Thread() + cufftdx::SM<Arch>());

template <int N, int Arch, cufftdx::fft_direction Dir, class C>
__device__ __forceinline__ void line_fft(C* p, int stride) {
    using F = ThreadFFT<N, Arch, Dir, C>;
    typename F::value_type v[F::storage_size];
#pragma unroll
    for (int e = 0; e < N; ++e) { v[e].x = p[e * stride].x; v[e].y = p[e * stride].y; }
    F().execute(v);
#pragma unroll
    for (int e = 0; e < N; ++e) { p[e * stride].x = v[e].x; p[e * stride].y = v[e].y; }
}

// The 3-D transform of TR resident padded rows (bank row j at bank + j*RS), axes z, y, x, all
// threads of the block, one thread FFT per line; ends with __syncthreads().  TR is the plan's
// tr, a compile-time constant of the embedded program (as the family's rows per block).
template <int NX, int NY, int NZ, int TR, int Arch, cufftdx::fft_direction Dir, class C>
__device__ void transform3(C* bank) {
    constexpr int tr = TR;
    using G = Geo<NX, NY, NZ>;
    if constexpr (NZ > 1) {
        for (int l = threadIdx.x; l < tr * NX * NY; l += blockDim.x) {
            const int j = l / (NX * NY), li = l % (NX * NY);
            line_fft<NZ, Arch, Dir>(bank + j * G::RS + li * G::ZP, 1);
        }
    }
    __syncthreads();
    if constexpr (NY > 1) {
        for (int l = threadIdx.x; l < tr * NX * NZ; l += blockDim.x) {
            const int j = l / (NX * NZ), li = l % (NX * NZ);
            line_fft<NY, Arch, Dir>(bank + j * G::RS + (li / NZ) * NY * G::ZP + li % NZ, G::ZP);
        }
    }
    __syncthreads();
    if constexpr (NX > 1) {
        for (int l = threadIdx.x; l < tr * NY * NZ; l += blockDim.x) {
            const int j = l / (NY * NZ), li = l % (NY * NZ);
            line_fft<NX, Arch, Dir>(bank + j * G::RS + G::plane_at(li), NY * G::ZP);
        }
    }
    __syncthreads();
}

// The staged tile as a direct Load writes it: v(k, j) is element k of tile column j.
template <int NX, int NY, int NZ, class C>
struct BankView {                       // single arm: k over the whole box
    C* bank;
    __device__ C& operator()(int k, int j) const {
        return bank[j * Geo<NX, NY, NZ>::RS + Geo<NX, NY, NZ>::at(k)];
    }
};
template <int NX, int NY, int NZ, class C>
struct PlaneView {                      // plane pass: k in plane kx only
    C* sm;
    int kx;
    __device__ C& operator()(int k, int j) const {
        using G = Geo<NX, NY, NZ>;
        return sm[j * G::PR + G::plane_at(k - kx * NY * NZ)];
    }
};

// Stage columns [col0, col0 + tr) into the bank.  A Load with kDirect writes the tile itself,
// block-cooperatively: ld.direct(view, k0, k1, col0, width, ncols) sets view(k, j) for k in
// [k0, k1) and j < width, zero where col0 + j >= ncols (a gathered, mixed load such as the
// Green unfold's U G U^dagger over a spin group; no cp.async).  Otherwise element (k, col)
// comes from ld.stage(k, col) (a global pointer) by cp.async, columns past ncols zero; then,
// when the Load has a finish step, bank = ld.finish(k, col, bank) in place.  Consecutive
// threads take consecutive columns.  Every Load declares kDirect and kFinish.
template <int NX, int NY, int NZ, int TR, class C, class Load>
__device__ void stage_tile(C* bank, long long col0, long long ncols, const Load& ld) {
    using G = Geo<NX, NY, NZ>;
    constexpr int tr = TR;
    if constexpr (Load::kDirect) {
        ld.direct(BankView<NX, NY, NZ, C>{bank}, 0, G::NK, col0, tr, ncols);
    } else {
        for (int i = threadIdx.x; i < tr * G::NK; i += blockDim.x) {
            const int j = i % tr, k = i / tr;
            C* dst = bank + j * G::RS + G::at(k);
            if (col0 + j < ncols) cp_async<sizeof(C)>(dst, ld.stage(k, col0 + j));
            else { dst->x = 0; dst->y = 0; }
        }
        cp_async_commit();
        cp_async_wait_all();
    }
    __syncthreads();
    if constexpr (!Load::kDirect && Load::kFinish) {
        for (int i = threadIdx.x; i < tr * G::NK; i += blockDim.x) {
            const int j = i % tr, k = i / tr;
            if (col0 + j < ncols) {
                C* e = bank + j * G::RS + G::at(k);
                *e = ld.finish(k, col0 + j, *e);
            }
        }
        __syncthreads();
    }
}

// bank = mid(k, col, bank) on the resident tile (single-pass Mid, columns independent).
template <int NX, int NY, int NZ, int TR, class C, class Mid>
__device__ void mid_tile(C* bank, long long col0, long long ncols, const Mid& mid) {
    using G = Geo<NX, NY, NZ>;
    constexpr int tr = TR;
    for (int i = threadIdx.x; i < tr * G::NK; i += blockDim.x) {
        const int j = i % tr, k = i / tr;
        if (col0 + j < ncols) {
            C* e = bank + j * G::RS + G::at(k);
            *e = mid(k, col0 + j, *e);
        }
    }
    __syncthreads();
}

// Group Mid on the resident tile: GROUP consecutive columns form one group (TR % GROUP == 0; the
// plan's tr counts groups, so a tile never splits one).  One thread per (group, k) loads the
// group's GROUP values into vals[], calls mid.group(k, g, vals) (g = the group's global index)
// which rewrites vals in place, and stores all GROUP back: a Mid that mixes a spin group, or
// one that reduces it into vals[0] (the Store then skips the other columns).
template <int NX, int NY, int NZ, int TR, int GROUP, class C, class Mid>
__device__ void mid_group_tile(C* bank, long long col0, long long ncols, const Mid& mid) {
    static_assert(TR % GROUP == 0, "a tile holds whole groups");
    using G = Geo<NX, NY, NZ>;
    constexpr int ng = TR / GROUP;
    for (int i = threadIdx.x; i < ng * G::NK; i += blockDim.x) {
        const int g = i % ng, k = i / ng;
        if (col0 + g * GROUP >= ncols) continue;
        C vals[GROUP];
#pragma unroll
        for (int q = 0; q < GROUP; ++q) vals[q] = bank[(g * GROUP + q) * G::RS + G::at(k)];
        mid.group(k, (col0 + g * GROUP) / GROUP, vals);
#pragma unroll
        for (int q = 0; q < GROUP; ++q) bank[(g * GROUP + q) * G::RS + G::at(k)] = vals[q];
    }
    __syncthreads();
}

// st.put(k, col, bank) for every stored element (the Store decides the row map and the scale).
template <int NX, int NY, int NZ, int TR, class C, class Store>
__device__ void store_tile(const C* bank, long long col0, long long ncols, const Store& st) {
    using G = Geo<NX, NY, NZ>;
    constexpr int tr = TR;
    for (int i = threadIdx.x; i < tr * G::NK; i += blockDim.x) {
        const int j = i % tr, k = i / tr;
        if (col0 + j < ncols) st.put(k, col0 + j, bank[j * G::RS + G::at(k)]);
    }
    __syncthreads();
}

// ---- split arm ------------------------------------------------------------------------------
// Plain k-leading access to the intermediate buffer between split passes.
template <class C>
struct Plain {
    static constexpr bool kDirect = false, kFinish = false;
    C* p;
    long long ncols;
    __device__ const C* stage(int k, long long col) const { return p + (long long)k * ncols + col; }
    __device__ void put(int k, long long col, C v) const { p[(long long)k * ncols + col] = v; }
};

// Plane pass: for each (kx, tile of TP columns) the (ky, kz) plane of every column in padded
// shared memory (smem: TP * PR elements), the y/z transforms, then st.put.  Load/Store as above;
// in place is allowed (a block writes only what it staged).
template <int NX, int NY, int NZ, int Arch, cufftdx::fft_direction Dir, int TP, class C, class Load, class Store>
__device__ void plane_pass(C* sm, long long ncols, const Load& ld, const Store& st) {
    using G = Geo<NX, NY, NZ>;
    const long long nct = (ncols + TP - 1) / TP;
    for (long long w = blockIdx.x; w < (long long)NX * nct; w += gridDim.x) {
        const int kx = int(w / nct);
        const long long c0 = (w % nct) * TP;
        __syncthreads();
        if constexpr (Load::kDirect) {
            ld.direct(PlaneView<NX, NY, NZ, C>{sm, kx}, kx * NY * NZ, (kx + 1) * NY * NZ, c0, TP, ncols);
        } else {
            for (int i = threadIdx.x; i < NY * NZ * TP; i += blockDim.x) {
                const int t = i % TP, p = i / TP, k = kx * NY * NZ + p;
                C* dst = sm + t * G::PR + G::plane_at(p);
                if (c0 + t < ncols) cp_async<sizeof(C)>(dst, ld.stage(k, c0 + t));
                else { dst->x = 0; dst->y = 0; }
            }
            cp_async_commit();
            cp_async_wait_all();
        }
        __syncthreads();
        if constexpr (!Load::kDirect && Load::kFinish) {
            for (int i = threadIdx.x; i < NY * NZ * TP; i += blockDim.x) {
                const int t = i % TP, p = i / TP;
                if (c0 + t < ncols) {
                    C* e = sm + t * G::PR + G::plane_at(p);
                    *e = ld.finish(kx * NY * NZ + p, c0 + t, *e);
                }
            }
            __syncthreads();
        }
        if constexpr (NZ > 1) {
            for (int l = threadIdx.x; l < TP * NY; l += blockDim.x)
                line_fft<NZ, Arch, Dir>(sm + (l / NY) * G::PR + (l % NY) * G::ZP, 1);
        }
        __syncthreads();
        if constexpr (NY > 1) {
            for (int l = threadIdx.x; l < TP * NZ; l += blockDim.x)
                line_fft<NY, Arch, Dir>(sm + (l / NZ) * G::PR + l % NZ, G::ZP);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < NY * NZ * TP; i += blockDim.x) {
            const int t = i % TP, p = i / TP;
            if (c0 + t < ncols) st.put(kx * NY * NZ + p, c0 + t, sm[t * G::PR + G::plane_at(p)]);
        }
    }
}

// Pencil pass, columns independent: per (ky, kz, col) the x-line in registers,
// inverse . mid . forward (CONV) or one transform in Dir (!CONV), in place on the intermediate.
template <int NX, int NY, int NZ, int Arch, bool CONV, cufftdx::fft_direction Dir, class C, class Mid>
__device__ void pencil_pass(C* y, long long ncols, const Mid& mid) {
    const long long total = (long long)NY * NZ * ncols;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long p = i / ncols, col = i % ncols;
        C v[NX];
#pragma unroll
        for (int kx = 0; kx < NX; ++kx) v[kx] = y[((long long)kx * NY * NZ + p) * ncols + col];
        if constexpr (CONV) {
            if constexpr (NX > 1) line_fft<NX, Arch, cufftdx::fft_direction::inverse>(v, 1);
#pragma unroll
            for (int kx = 0; kx < NX; ++kx) v[kx] = mid(int(kx * NY * NZ + p), col, v[kx]);
            if constexpr (NX > 1) line_fft<NX, Arch, cufftdx::fft_direction::forward>(v, 1);
        } else {
            if constexpr (NX > 1) line_fft<NX, Arch, Dir>(v, 1);
        }
#pragma unroll
        for (int kx = 0; kx < NX; ++kx) y[((long long)kx * NY * NZ + p) * ncols + col] = v[kx];
    }
}

// Group pencil pass (a Mid that mixes GROUP columns, e.g. the Lorentz vertex sum over a spin
// group): block work item (p = (ky,kz), TY consecutive group instances); thread (member, t).
// Each thread loads and inverse-transforms its column's x-line into shared memory, then forms
// its own member's output from the group's values at every kx, forward-transforms and stores.
//   cols.col(inst, member)            the column of a member;
//   Mid::kAux                         per-(k, instance) operand elements (the 16 V[k,x,A,y,B] of a
//                                     Lorentz block), 0 for none;
//   mid.stage_aux(saux, p, inst0, ld) block-cooperative cp.async of those operands for every kx
//                                     into saux[(kx*TY + t)*ld + e] (the header commits and waits);
//   mid.bind(member)                  a per-thread functor holding the member's vertex tables in
//                                     registers; f(grp, aux) returns the member's value from the
//                                     group's values grp[q*TY] (q < GROUP) and the operands aux[e].
// Shared memory (the caller's dynamic smem): NX*GROUP*TY + NX*TY*(kAux|1) elements; the odd
// operand stride keeps the TY instances of a warp on different banks.
// kForward = false skips the forward x transform and hands the R-space value to
// st.put(k, col, v) (k = kx*NY*NZ + p) instead of writing y: a pass that ends in R space
// (mode 11 accumulates chi_R and transforms once after the tau sum).
template <int NX, int NY, int NZ, int Arch, int GROUP, int TY, bool kForward, class C, class Cols, class Mid,
          class Store>
__device__ void pencil_group_pass(C* y, C* smem, long long ncols, long long n_inst, const Cols& cols,
                                  const Mid& mid, const Store& st) {
    constexpr int LD = Mid::kAux | 1;
    C* sg = smem;
    C* saux = smem + NX * GROUP * TY;
    const int member = threadIdx.x / TY, t = threadIdx.x % TY;
    const auto f = mid.bind(member);
    const long long nit = (n_inst + TY - 1) / TY;
    for (long long w = blockIdx.x; w < (long long)NY * NZ * nit; w += gridDim.x) {
        const long long p = w / nit, inst0 = (w % nit) * TY, inst = inst0 + t;
        const bool live = inst < n_inst;
        const long long col = live ? cols.col(inst, member) : 0;
        __syncthreads();                                  // the previous item's smem reads are done
        if constexpr (Mid::kAux > 0) {
            mid.stage_aux(saux, p, inst0, LD);
            cp_async_commit();
        }
        C v[NX];
#pragma unroll
        for (int kx = 0; kx < NX; ++kx) {
            if (live) v[kx] = y[((long long)kx * NY * NZ + p) * ncols + col];
            else { v[kx].x = 0; v[kx].y = 0; }
        }
        if constexpr (NX > 1) line_fft<NX, Arch, cufftdx::fft_direction::inverse>(v, 1);
#pragma unroll
        for (int kx = 0; kx < NX; ++kx) sg[(kx * GROUP + member) * TY + t] = v[kx];
        if constexpr (Mid::kAux > 0) cp_async_wait_all();
        __syncthreads();
#pragma unroll
        for (int kx = 0; kx < NX; ++kx) v[kx] = f(sg + kx * GROUP * TY + t, saux + (kx * TY + t) * LD);
        if constexpr (kForward && NX > 1) line_fft<NX, Arch, cufftdx::fft_direction::forward>(v, 1);
        if (live) {
#pragma unroll
            for (int kx = 0; kx < NX; ++kx) st.put(kx * NY * NZ + int(p), col, v[kx]);
        }
    }
}

// The in-place form: forward-transform and write back to y (the mode-8 vertex pencil).
template <int NX, int NY, int NZ, int Arch, int GROUP, int TY, class C, class Cols, class Mid>
__device__ void pencil_group_pass(C* y, C* smem, long long ncols, long long n_inst, const Cols& cols,
                                  const Mid& mid) {
    pencil_group_pass<NX, NY, NZ, Arch, GROUP, TY, true>(y, smem, ncols, n_inst, cols, mid, Plain<C>{y, ncols});
}

}  // namespace lrx_kbox
#endif
