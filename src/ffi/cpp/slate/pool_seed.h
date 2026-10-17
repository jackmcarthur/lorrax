// pool_seed.h — SLATE device tiles and workspace inside XLA's buffer assignment.
//
// WHY.  SLATE's Target::Devices drivers keep every device tile in a per-matrix
// memory pool (slate::Memory) that allocates with blas::device_malloc, i.e. a raw
// hipMalloc / cudaMalloc OUTSIDE the XLA arena.  insertLocalTiles(Devices) takes one
// block per local tile, and getrf / potrf / he2hb / trsm / unmqr call
// reserveDeviceWorkspace(), which tops the pool up to one block per local tile
// again.  On the ROCM leg the BFC arena is preallocated at runtime.pool_fraction
// (60.18 of 68.70 GB on an MI250X GCD), so SLATE gets the ~8.5 GB the pool policy
// leaves outside, minus the HIP context, RCCL and scratch (~1.4 GB measured): every
// distributed SLATE solve beyond ~1.6 n^2/P bytes per rank failed with
// blas::Error "out of memory, in function device_malloc", thrown from an OpenMP
// task and therefore an abort of every rank (MI250X P4: eigh c128 n=20480; pool
// logs dlascale probe1/probe2).
//
// WHAT.  The handler receives one extra XLA output buffer (a workspace the Python
// wrapper sizes with distrib_la._slate's *_workspace_bytes and never reads), and
// SEEDS a matrix's pool with blocks carved from it before any tile is inserted.
// slate::Memory::alloc then pops seeded blocks; reserveDeviceWorkspace() finds the
// capacity already there and allocates nothing.  Only when the seeded blocks run
// out does the pool fall back to device_malloc (counted, see PoolSeed::fallback).
// So SLATE's working set is part of the compiled program's memory (XLA's
// memory_analysis and the arena), not a hidden allocation beside it.
//
// HOW.  slate::Memory, slate::internal MatrixStorage and slate::BaseMatrix all
// declare `friend class Debug` (slate::Debug, defined only in SLATE's own
// auxiliary/Debug.cc, which is not an installed header).  This file defines a
// class of that name with ONE purpose — seeding — and touches only
// Memory::{block_size_, free_blocks_, capacity_} and BaseMatrix::storage_ /
// MatrixStorage::memory_ of the pinned SLATE (v2025.05.28-1, ded15290;
// config/frontier/build_slate_hip.sh).  Every member is static and inline with a
// lorrax_ prefix, so no symbol can meet one of SLATE's.  Seeded blocks are never
// in Memory::allocated_mem_, so clearDeviceBlocks() (the storage's destructor)
// drops them without freeing; the XLA buffer outlives every SLATE object because
// all of them are scoped inside the handler.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>

#include "../common/gpu_runtime.h"

#include <slate/slate.hh>

namespace slate {

class Debug {
public:
    template <typename T>
    static std::size_t lorrax_block_bytes(BaseMatrix<T>& M) {
        return M.storage_->memory_.block_size_;
    }
    template <typename T>
    static std::size_t lorrax_capacity(BaseMatrix<T>& M, int device) {
        return M.storage_->memory_.capacity_.at(device);
    }
    // Push `nblocks` blocks of the matrix's pool block size, carved from `base`,
    // onto device `device`'s free list.  Call before any tile exists (single
    // threaded: no SLATE task can be touching this pool yet).
    template <typename T>
    static void lorrax_seed(BaseMatrix<T>& M, int device, char* base, std::int64_t nblocks) {
        auto& mem = M.storage_->memory_;
        for (std::int64_t i = 0; i < nblocks; ++i)
            mem.free_blocks_.at(device).push(base + i * mem.block_size_);
        mem.capacity_.at(device) += static_cast<std::size_t>(nblocks);
    }
};

}  // namespace slate

namespace lorrax_ffi::slate {

// A bump carver over one XLA workspace buffer: hands out whole pool blocks.
struct PoolSeed {
    char*        base = nullptr;
    std::size_t  size = 0;
    std::size_t  used = 0;

    PoolSeed(void* b, std::size_t n) : base(static_cast<char*>(b)), size(n) {}

    // Seed `M`'s pool with up to `want` blocks; returns the number seeded (the
    // carve stops at the end of the buffer: the pool falls back past it).
    template <typename T>
    std::int64_t seed(::slate::BaseMatrix<T>& M, std::int64_t want, int device = 0) {
        const std::size_t bs = ::slate::Debug::lorrax_block_bytes(M);
        const std::size_t align = 256;
        std::size_t at = (used + align - 1) / align * align;
        if (bs == 0 || at >= size) return 0;
        std::int64_t fit = static_cast<std::int64_t>((size - at) / bs);
        std::int64_t n = want < fit ? want : fit;
        if (n <= 0) return 0;
        ::slate::Debug::lorrax_seed(M, device, base + at, n);
        used = at + static_cast<std::size_t>(n) * bs;
        return n;
    }
};

// Blocks the pool allocated past its seed (device_malloc fallbacks), read
// before the matrix dies.  Reported on stderr when LORRAX_SLATE_POOL_REPORT is
// set (a diagnostic, never a policy).
template <typename T>
inline std::int64_t pool_fallback(::slate::BaseMatrix<T>& M, std::int64_t seeded, int device = 0) {
    const std::int64_t cap = static_cast<std::int64_t>(::slate::Debug::lorrax_capacity(M, device));
    return cap > seeded ? cap - seeded : 0;
}

inline bool pool_report_enabled() {
    const char* v = std::getenv("LORRAX_SLATE_POOL_REPORT");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Device bytes in use (total - free, whole device) per stage, sampled every 20 ms on
// a side thread while a handler runs.  A diagnostic (LORRAX_SLATE_TIMERS /
// LORRAX_SLATE_POOL_REPORT); the XLA arena is a constant part of every reading.
struct StageMemSampler {
    std::atomic<int> stage{0};
    std::atomic<bool> stop{false};
    std::vector<std::size_t> max_used;
    std::thread th;
    int device = 0;
    explicit StageMemSampler(int nstages, bool on) : max_used(nstages, 0) {
        if (!on) { stop = true; return; }
        lrx::gpu::get_device(&device);
        th = std::thread([this] {
            lrx::gpu::set_device(device);
            while (!stop.load()) {
                std::size_t fr = 0, tot = 0;
                if (LRX_GPU_SYM(MemGetInfo)(&fr, &tot) == lrx::gpu::kSuccess) {
                    const int s = stage.load();
                    if (s >= 0 && s < (int) max_used.size() && tot - fr > max_used[s])
                        max_used[s] = tot - fr;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });
    }
    void at(int s) { stage.store(s); }
    ~StageMemSampler() { stop = true; if (th.joinable()) th.join(); }
};

}  // namespace lorrax_ffi::slate
