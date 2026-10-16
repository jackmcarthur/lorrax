// kdft_kminor_plan.h -- HOST: the k-minor plan of the kdft family (kdft_kminor_ffi.cc), modes 4/5
// on a (rows, nk) operand with k trailing, nk = NX*NY*NZ.
//
// SINGLE arm: one pass, a block owns C consecutive rows whole (kdft_kminor_kernel.h), when one
// row's tile fits a block's LDS and the single pass is the cheaper plan.
//
// SPLIT arm: the k axes split into contiguous GROUPS, as the k-leading plan
// (kdft_klead_plan.h), with the trailing group T taken by ROW passes and the others by COLUMN
// passes:
//   row pass     the k-minor kernel on the trailing group (LRX_SPLIT = O_T = nk / G_T): each
//                row's O_T contiguous runs of G_T points are rows of a (rows*O_T, G_T) operand
//   column pass  the k-leading pass kernel on the (O = rows*kob, G, I) view of a leading group
//                (kob = the k axes before it, I = the k axes after it, contiguous); the
//                convolution's K multiply rides on one column pass (LRX_MODE 4)
// Mode 5: the row pass (operand -> result), then one column pass per leading group in place.
// Mode 4: the inverse row pass (operand -> work), the inverse column passes, the convolution
// column pass, the forward column passes, and the forward row pass (work -> result, in either
// out_layout): 2*(groups-1)+1 HBM round trips, as the k-leading mode 2.  The work buffer is the
// result for out_layout 0 and a scratch buffer for out_layout 1 (its permuted store cannot run
// in place).  The single arm serves every grid it fits (measured one GCD, ~1 GB c128, pool log
// kcore_tune1: forced split vs single at 8x8x8 2.46 vs 1.31 ms (mode 5), 4.48 vs 2.06 (mode 4);
// 16x16x15 3.97 vs 3.16, 7.28 vs 4.66), except mode 4 out_layout 1 with one row per block, whose
// permuted store then writes single elements (12^3: split 6.94 vs single 9.20 ms; 16x16x15 8.21
// vs 10.16).  The split arm's grouping is the one of least summed pass cost.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "kdft_klead_plan.h"

namespace kdft {

// Row-pass launch shape, from a sweep of tile budget {8, 16, 32, 60} KiB x {256, 512, 1024}
// threads on one MI250X GCD (13 BSE tiles, modes 4 (both layouts) and 5, k-grids 8x8x1 ..
// 8x8x8): a 32 KiB budget and 512 threads is within ~20% of each tile's best and beats the
// k-leading family's 16 KiB / 512 on all 13 (mode 4 out_layout 1 at 8x8x8 6.22 -> 3.83 ms, whose
// permuted store wants long row runs).  The record is at the Python owner (ffi.fft, above
// _ROCM_KLEAD_TARGETS).  C is odd (the transposing load/store's bank rule): P|1 for the largest
// power of two P <= kRowsMax with nk*(P|1) complex in the budget -- the swept footprint -- else
// the largest odd C that fits the LDS.
constexpr int kMinorThreads = 512;
constexpr long long kMinorBudget = 32 * 1024;
constexpr int kRowsMax = 64;
// Mode 4 out_layout 1 takes the split arm when the single arm would hold fewer rows per block.
constexpr int kL1RowsMin = 2;
// The split arm's out_layout-1 store (its last row pass, LRX_OUT 1 with LRX_SPLIT > 1): a block
// owns C consecutive operand rows of one o_T and the pass takes its own rows per block
// (power-of-two runs; the first pass keeps layout 0's odd rows).  The block ORDER and C follow
// the store's k stride, M*cbytes (M = d1*d2*d3*d4): a multiple of kL1Stride takes order 0 (row
// chunks fastest, one o_T in flight) and the kLdsMax budget where that reaches kL1WideRows rows,
// else order 1 (o_T fastest) and kMinorBudget.  Measured, one MI250X GCD, mode 4, ~1 GB c128,
// ms per call, order 0 / order 1 (pool log kminfin_sweep2; layout 0 in brackets):
//   16^3  M=4096  10.56 / 12.81 (6.62)    M=3600  7.66 / 6.53 (5.84)
//   64^2  M=4096   7.73 (C=32) / 11.09 (5.95)   M=3600  7.90 / 5.22 (5.24)
//   128x64 M=4096 11.55 (C=32) / 12.85 (7.52)   M=1936 12.50 / 7.06 (7.10)
//   20^3  M=1024   3.90 / 4.16 (3.40)     12^3 M=9216 6.56 / 7.10 (4.60), M=9604 6.72 / 6.68 (4.81)
// Squares of chunks x o_T (16 or 64 chunks) and C=32 under order 1 lost on every shape.
constexpr long long kL1Stride = 4096;
constexpr int kL1WideRows = 32;

// The LDS one row-pass block takes: the padded tile, the twiddles, the per-row tables.
inline long long minor_lds_bytes(long long nk, int c, int ntw, long long cbytes) {
    return nk * c * cbytes + ntw * cbytes + c * 20LL;
}
// Rows C and the odd LDS width W (C or C + 1) of one block: P|1 for the largest power of two
// P <= kRowsMax with nk*(P|1) complex in the budget -- the swept footprint -- or less to fit the
// LDS.  Every row real (C = W = P|1) for mode 5 and out_layout 0; out_layout 1 takes C = P,
// W = P|1 (a zero pad column), so its permuted store writes aligned runs of P rows: measured
// (one GCD) 8x8x1 3.98 -> 3.11, 6x6x6 4.87 -> 4.09, 8x8x8 4.45 -> 3.99 ms, while odd rows win for
// mode 5 and layout 0 (8x8x8: 1.73 -> 1.34 and 3.36 -> 2.53 ms).
struct Rows { int c = 0, w = 0; };
inline Rows rows_per_block(long long nk, int ntw, long long cbytes, bool pow2,
                           long long budget = kMinorBudget) {
    auto fits = [&](int w) { return minor_lds_bytes(nk, w, ntw, cbytes) <= kLdsMax; };
    if (!fits(1)) return {};
    int p = kRowsMax;
    while (p > 1 && nk * (p | 1) * cbytes > budget) p /= 2;
    while (p > 1 && !fits(p | 1)) p /= 2;
    return pow2 ? Rows{p, p | 1} : Rows{p | 1, p | 1};
}
// A row pass's cost on the k-leading pass_cost scale: the measured single arm streams at
// 1.07-1.21 TB/s (one GCD) for every C >= 3 within the budget, so only the tile above the budget
// (fewer blocks per CU) costs, as pass_cost's tile term; a one-row tile also loses the
// transposing load's row parallelism.
inline double row_cost(long long nk, Rows r, long long cbytes) {
    const double tile = double(nk) * r.w * cbytes;
    const double big = tile > kMinorBudget ? std::pow(tile / kMinorBudget, 0.55) : 1.0;
    return big * (r.c == 1 ? 1.5 : 1.0);
}

struct MinorPlan {
    bool split = false;
    std::array<int, 3> grid{1, 1, 1};   // the row pass's k axes (the whole grid when single)
    long long ot = 1;                   // O_T: row-pass rows per operand row (1 when single)
    Rows rows;                          // the row pass's C, W (the first row pass, or the single arm)
    Rows last;                          // the split arm's last row pass (mode 4): == rows except
                                        // for out_layout 1, whose store takes its own
    int order = -1;                     // that store's block order (l1_order; -1: no such store)
    std::vector<Pass> cols;             // column passes in launch order (split arm)
    double cost = 0;
};

// The row-pass rows for a (sub)grid: out_layout 1 stores take aligned power-of-two row runs.
inline Rows minor_rows(const std::array<int, 3>& g, long long cbytes, bool pow2,
                       long long budget = kMinorBudget) {
    return rows_per_block(1LL * g[0] * g[1] * g[2], g[0] + g[1] + g[2], cbytes, pow2, budget);
}
// Blocks of one row pass over `rows` operand rows: rows*O_T/C, or for the split arm's
// out_layout-1 store (order 0 | 1; -1 for every other pass) one block per (row chunk, o_T).
inline long long row_blocks(long long rows, long long ot, int c, int order) {
    if (order < 0) return (rows * ot + c - 1) / c;
    return (rows + c - 1) / c * ot;
}
// The out_layout-1 store's block order for a k stride of kstride_bytes (above).
inline int l1_order(long long kstride_bytes) { return kstride_bytes % kL1Stride == 0 ? 0 : 1; }

// The plan of one call; .rows.c == 0 when no arm fits (refused, GATE kdft-kminor-tile).
// m: the operand's rows per d0 index (d1*d2*d3*d4), whose stride orders the out_layout-1 store.
inline MinorPlan plan_kminor(int nkx, int nky, int nkz, int mode, int forward, int out, long long cbytes,
                             long long m = 1) {
    const int ax[3] = {nkx, nky, nkz};
    const long long nk = 1LL * nkx * nky * nkz;
    MinorPlan best;
    best.grid = {nkx, nky, nkz};
    best.rows = minor_rows(best.grid, cbytes, out == 1);
    best.last = best.rows;
    if (best.rows.c) {
        best.cost = row_cost(nk, best.rows, cbytes);
        if (!(mode == 4 && out == 1 && best.rows.c < kL1RowsMin)) return best;   // the single arm
    }
    MinorPlan single = best;
    best = MinorPlan{};
    for (int cuts : {1, 2, 3}) {                               // bit 0: cut after x; bit 1: after y
        std::vector<std::array<int, 2>> runs;
        int a0 = 0;
        for (int a = 0; a < 3; ++a)
            if (a == 2 || (cuts >> a) & 1) { runs.push_back({a0, a + 1}); a0 = a + 1; }
        MinorPlan p;
        p.split = true;
        const auto [tlo, thi] = runs.back();
        long long gt = 1;
        for (int a = tlo; a < thi; ++a) { p.grid[a] = ax[a]; gt *= ax[a]; }
        if (gt == 1) continue;                                 // a unit trailing group: another cut
        p.ot = nk / gt;
        p.rows = minor_rows(p.grid, cbytes, false);
        if (mode == 4 && out == 1) {
            p.order = l1_order(m * cbytes);
            p.last = minor_rows(p.grid, cbytes, true);
            const Rows wide = minor_rows(p.grid, cbytes, true, kLdsMax);
            if (p.order == 0 && wide.c >= kL1WideRows) p.last = wide;
        } else {
            p.last = p.rows;
        }
        if (p.rows.c == 0 || p.last.c == 0) continue;
        std::vector<Pass> groups;
        bool ok = true;
        for (size_t i = 0; i + 1 < runs.size(); ++i) {
            Pass c;
            for (int a = runs[i][0]; a < runs[i][1]; ++a) c.g[a] = ax[a];
            if (c.g[0] * c.g[1] * c.g[2] == 1) continue;
            for (int a = 0; a < runs[i][0]; ++a) c.outer *= ax[a];          // kob
            for (int a = runs[i][1]; a < 3; ++a) c.inner_k *= ax[a];       // I
            c.w = tile_width(c.g, cbytes, &c.cost, c.inner_k);
            if (c.w == 0) { ok = false; break; }
            c.nt = tile_threads(c.g, c.w);
            groups.push_back(c);
        }
        if (!ok || groups.empty()) continue;
        const double rc = row_cost(gt, p.rows, cbytes);
        if (mode == 5) {
            for (Pass c : groups) { c.mode = 3; c.forward = forward; p.cols.push_back(c); }
            p.cost = rc;
        } else {
            Pass conv = groups[0];
            conv.mode = 4;
            Pass rg = conv;
            if (reg_tile(rg.g, cbytes, rg.inner_k, &rg)) conv = rg;
            for (size_t i = 1; i < groups.size(); ++i) { Pass c = groups[i]; c.mode = 3; c.forward = 0; p.cols.push_back(c); }
            p.cols.push_back(conv);
            for (size_t i = groups.size() - 1; i >= 1; --i) { Pass c = groups[i]; c.mode = 3; c.forward = 1; p.cols.push_back(c); }
            p.cost = rc + row_cost(gt, p.last, cbytes);
        }
        for (const Pass& c : p.cols) p.cost += c.cost;
        if (best.rows.c == 0 || p.cost < best.cost) best = p;
    }
    return best.rows.c ? best : single;
}

}  // namespace kdft
