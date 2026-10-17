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
#include <mutex>

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
    static Memory* lorrax_memory(BaseMatrix<T>& M) {
        return &M.storage_->memory_;
    }
    // Blocks of `mem` in use on `device` (capacity minus free list).  Call only while
    // holding SLATE's `omp critical(slate_memory)` (PoolWatch), the lock under which
    // Memory::alloc / Memory::free change both.
    static std::size_t lorrax_in_use(Memory& mem, int device, std::size_t* cap) {
        const std::size_t c = mem.capacity_.at(device);
        const std::size_t f = mem.free_blocks_.at(device).size();
        *cap = c;
        return c > f ? c - f : 0;
    }
    template <typename T>
    static void lorrax_seed(BaseMatrix<T>& M, int device, char* base, std::int64_t nblocks) {
        auto& mem = M.storage_->memory_;
        for (std::int64_t i = 0; i < nblocks; ++i)
            mem.free_blocks_.at(device).push(base + i * mem.block_size_);
        mem.capacity_.at(device) += static_cast<std::size_t>(nblocks);
    }
};

}  // namespace slate

// SLATE's pool lock.  slate::Memory::alloc / ::free run under
// `#pragma omp critical(slate_memory)`; the pinned SLATE is GCC-built against libgomp,
// so that critical is GOMP_critical_name_start(&.gomp_critical_user_slate_memory),
// a symbol libslate exports (nm -D: "B .gomp_critical_user_slate_memory").  Taking
// the same mutex from PoolWatch's thread makes its reads of capacity_ / free_blocks_
// exact.  Both references are weak: a SLATE without them (another compiler or OpenMP
// runtime) leaves them null and PoolWatch reports "unavailable" instead of guessing.
extern "C" {
extern void* lorrax_slate_memory_crit __asm__(".gomp_critical_user_slate_memory")
    __attribute__((weak));
void GOMP_critical_name_start(void** pptr) __attribute__((weak));
void GOMP_critical_name_end(void** pptr) __attribute__((weak));
}

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

// High-water of each seeded pool, sampled every 1 ms under SLATE's pool lock while
// a handler runs (LORRAX_SLATE_POOL_REPORT; a diagnostic, never a policy).  "in use" =
// pool capacity minus its free list = the blocks SLATE holds at that instant (local
// tiles, remote / workspace tiles, device copies); the maximum over the call is the
// measured high-water the seeding rules in distrib_la._slate are fitted to.  A matrix
// is watched from add() until its scope object dies (declare it right after the
// matrix so it is destroyed first).  Capacity resets (clearWorkspace with nothing in
// use drops every block, seeded ones included) are counted as `drops`.
struct PoolWatch {
    struct Entry {
        ::slate::Memory* mem = nullptr;
        const char* name = "";
        std::int64_t seeded = 0, local = 0;
        std::size_t max_in_use = 0, max_cap = 0, last_cap = 0;
        std::int64_t drops = 0;
        bool live = false;
    };
    std::vector<Entry> ents;
    std::mutex mu;
    std::atomic<bool> stop{false};
    std::thread th;
    bool on = false;
    int device = 0;

    static bool available() {
        return &lorrax_slate_memory_crit != nullptr && GOMP_critical_name_start != nullptr &&
               GOMP_critical_name_end != nullptr;
    }
    explicit PoolWatch(bool enabled) : on(enabled && available()) {
        ents.reserve(16);
        if (!on) return;
        th = std::thread([this] {
            while (!stop.load()) {
                sample();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    void sample() {
        std::lock_guard<std::mutex> g(mu);
        for (auto& e : ents) {
            if (!e.live) continue;
            std::size_t cap = 0, used = 0;
            GOMP_critical_name_start(&lorrax_slate_memory_crit);
            used = ::slate::Debug::lorrax_in_use(*e.mem, device, &cap);
            GOMP_critical_name_end(&lorrax_slate_memory_crit);
            if (cap < e.last_cap) ++e.drops;
            e.last_cap = cap;
            if (used > e.max_in_use) e.max_in_use = used;
            if (cap > e.max_cap) e.max_cap = cap;
        }
    }
    template <typename T>
    int add(::slate::BaseMatrix<T>& M, const char* name, std::int64_t seeded, std::int64_t local) {
        if (!on) return -1;
        std::lock_guard<std::mutex> g(mu);
        Entry e;
        e.mem = ::slate::Debug::lorrax_memory(M);
        e.name = name; e.seeded = seeded; e.local = local; e.live = true;
        ents.push_back(e);
        return static_cast<int>(ents.size()) - 1;
    }
    void drop(int id) {
        if (id < 0) return;
        sample();                       // the last reading before the matrix dies
        std::lock_guard<std::mutex> g(mu);
        ents.at(id).live = false;
    }
    // One line per watched matrix: what was seeded vs what SLATE held at most.
    void report(const char* op, int rank, std::int64_t n, std::int64_t nb, int itemsize) {
        if (!pool_report_enabled()) return;
        if (!on) {
            std::fprintf(stderr, "[slate.pool_hw] op=%s rank=%d unavailable (SLATE pool lock "
                         "symbol not found)\n", op, rank);
            return;
        }
        std::lock_guard<std::mutex> g(mu);
        for (auto& e : ents)
            std::fprintf(stderr, "[slate.pool_hw] op=%s rank=%d n=%lld nb=%lld itemsize=%d mat=%s "
                         "local=%lld seeded=%lld max_in_use=%zu max_cap=%zu drops=%lld\n",
                         op, rank, (long long) n, (long long) nb, itemsize, e.name,
                         (long long) e.local, (long long) e.seeded, e.max_in_use, e.max_cap,
                         (long long) e.drops);
    }
    ~PoolWatch() { stop = true; if (th.joinable()) th.join(); }
};

// RAII: watch M from here until this object dies.
struct PoolWatchScope {
    PoolWatch& w;
    int id;
    template <typename T>
    PoolWatchScope(PoolWatch& w_, ::slate::BaseMatrix<T>& M, const char* name,
                   std::int64_t seeded, std::int64_t local)
        : w(w_), id(w_.add(M, name, seeded, local)) {}
    ~PoolWatchScope() { w.drop(id); }
    PoolWatchScope(const PoolWatchScope&) = delete;
    PoolWatchScope& operator=(const PoolWatchScope&) = delete;
};

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
