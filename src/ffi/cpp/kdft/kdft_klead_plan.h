// kdft_klead_plan.h -- HOST: the k-leading pass plan of the kdft family (kdft_klead_ffi.cc).
//
// A k-grid (NX, NY, NZ) is split into contiguous axis GROUPS, {xyz}, {x}{yz}, {xy}{z} or
// {x}{y}{z}.  One pass transforms one group as an (O, G, I) view of the operand
// (kdft_klead_kernel.h); its LDS tile is G*W complex.  Mode 3 takes one pass per group that
// has a non-unit axis; mode 2 takes the leading groups' inverse passes, the trailing group's
// convolution pass, and the leading groups' forward passes: 2*(groups-1)+1 HBM round trips.
// The plan is the feasible grouping of least summed pass_cost (below), so a grid may take more
// passes when they stream faster (8x8x8 mode 3: two passes, 2.89 ms, against one, 3.89 ms).
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include "kdft_plan.h"

namespace kdft {

// The verified range (ffi.fft.KDFT_AXIS_MAX / KDFT_NK_MAX; gate table in kconv.md).
constexpr int kAxisMax = 128;
constexpr long long kNkMax = 10240;
// Tile sizing (kdft measured record, MI250X, 1 GB operands, ffi.fft above _ROCM_KLEAD_TARGETS):
// one pass streams at 1.16 TB/s with W >= 32 columns and a tile within kTileBudget (four blocks
// share a CU's LDS), 1.04 at W = 16, 0.78 at W = 8, and less as the tile grows past the budget
// (0.53 at G = 216, 0.44 at G = 400, both W = 8).  pass_cost() is that fit, relative to the
// best pass; the plan minimises its sum.
constexpr long long kTileBudget = 16 * 1024;
constexpr long long kLdsMax = 64 * 1024;
constexpr int kWMax = 64;
constexpr int kThreadsMax = 512;
// The fit is good to a few per cent: within that, take the wider tile and the fewer passes
// (the shapes measured): a wider W within 5% of the narrower's cost, more groups only for 3%.
constexpr double kWiderSlack = 1.05;
constexpr double kMorePassesGain = 0.97;
struct Pass {
    std::array<int, 3> g{1, 1, 1};   // the group's axes, the other axes 1
    long long outer = 1;             // O
    long long inner_k = 1;           // I / ncols
    int w = 0, nt = 0;
    int mode = 3;                    // 2: the convolution pass; 3: a transform pass
    int forward = 0;                 // a mode-3 pass's direction
    double cost = 0;                 // pass_cost of its tile
};

inline double pass_cost(int w, long long tile_bytes) {
    const double narrow = w >= 32 ? 1.0 : 1.0 + (32.0 / w - 1.0) / 6.0;
    const double big = tile_bytes > kTileBudget ? std::pow(double(tile_bytes) / kTileBudget, 0.55) : 1.0;
    return narrow * big;
}

// Whether an axis of length n has a stage above kRmax (a prime: compute-bound, kdft_core.h).
inline bool has_long_radix(int n) {
    const Plan pl = make_plan(n);
    for (int s = 0; s < pl.n; ++s)
        if (pl.r[s] > kRmax) return true;
    return false;
}

// Inner indices per block for a group (0: one column does not fit the LDS), and its cost.  A
// group with a long radix takes the widest tile that fits: its stage needs lines, not occupancy
// (97x97x1: 18.4 ms at W = 32 against 42 ms at W = 8 per mode-2 call).
inline int tile_width(const std::array<int, 3>& g, long long cbytes, double* cost) {
    const long long G = 1LL * g[0] * g[1] * g[2];
    const long long tws = g[0] + g[1] + g[2];
    const bool wide = has_long_radix(g[0]) || has_long_radix(g[1]) || has_long_radix(g[2]);
    int best = 0;
    double best_cost = 0;
    for (int w = 1; w <= kWMax; w *= 2) {
        if ((G * w + tws) * cbytes > kLdsMax) break;
        const double c = pass_cost(w, G * w * cbytes);
        if (wide || best == 0 || c <= best_cost * kWiderSlack) { best = w; best_cost = c; }
    }
    *cost = best_cost;
    return best;
}

// Threads per block: enough for the busiest stage's work items and for half the tile's points
// (the load and store keep two elements in flight per thread), in whole wavefronts.
inline int tile_threads(const std::array<int, 3>& g, int w) {
    const long long G = 1LL * g[0] * g[1] * g[2];
    long long items = G * w / 2 > 64 ? G * w / 2 : 64;
    for (int n : g)
        if (n > 1) {
            const long long it = axis_items_max(n, G, w);
            if (it > items) items = it;
        }
    const long long nt = (items + 63) / 64 * 64;
    return static_cast<int>(nt < kThreadsMax ? nt : kThreadsMax);
}

// The passes of one call, in launch order; empty when no grouping fits the LDS.
inline std::vector<Pass> plan_passes(int nkx, int nky, int nkz, int mode, int forward, long long cbytes) {
    const int ax[3] = {nkx, nky, nkz};
    // contiguous groupings as cut masks: bit 0 = cut after x, bit 1 = cut after y
    std::vector<Pass> best;
    double best_cost = -1;
    for (int cuts : {0, 1, 2, 3}) {
        std::vector<std::array<int, 2>> runs;                // [first axis, last axis + 1)
        int a0 = 0;
        for (int a = 0; a < 3; ++a)
            if (a == 2 || (cuts >> a) & 1) { runs.push_back({a0, a + 1}); a0 = a + 1; }
        std::vector<Pass> groups;
        bool ok = true;
        for (auto [lo, hi] : runs) {
            Pass p;
            for (int a = lo; a < hi; ++a) p.g[a] = ax[a];
            for (int a = 0; a < lo; ++a) p.outer *= ax[a];
            for (int a = hi; a < 3; ++a) p.inner_k *= ax[a];
            p.w = tile_width(p.g, cbytes, &p.cost);
            if (p.w == 0) { ok = false; break; }
            p.nt = tile_threads(p.g, p.w);
            groups.push_back(p);
        }
        if (!ok) continue;
        auto unit = [](const Pass& p) { return p.g[0] * p.g[1] * p.g[2] == 1; };
        std::vector<Pass> seq;
        if (mode == 2) {
            const int n = static_cast<int>(groups.size());
            for (int i = 0; i < n - 1; ++i)
                if (!unit(groups[i])) { Pass p = groups[i]; p.mode = 3; p.forward = 0; seq.push_back(p); }
            Pass mid = groups[n - 1]; mid.mode = 2; seq.push_back(mid);
            for (int i = n - 2; i >= 0; --i)
                if (!unit(groups[i])) { Pass p = groups[i]; p.mode = 3; p.forward = 1; seq.push_back(p); }
        } else {
            for (const Pass& g : groups)
                if (!unit(g)) { Pass p = g; p.mode = 3; p.forward = forward; seq.push_back(p); }
            if (seq.empty()) { Pass p = groups[0]; p.mode = 3; p.forward = forward; seq.push_back(p); }
        }
        double cost = 0;
        for (const Pass& p : seq) cost += p.cost;
        if (best_cost < 0 || cost < best_cost * kMorePassesGain) { best_cost = cost; best = seq; }
    }
    return best;
}

}  // namespace kdft
