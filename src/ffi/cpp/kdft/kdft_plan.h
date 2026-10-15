// kdft_plan.h -- the kdft line-DFT radix plan: pure constexpr C++, included by the device text
// (kdft_core.h, through the RTC's embedded headers) and by the host handler (kdft_klead_ffi.cc,
// which sizes a pass's thread count from it), so the plan has one definition.
//
// make_plan(N) factors an axis of length N into Stockham stages: it minimises
// sum(R + kStageCost) over factorisations into radices <= kRmax; a prime factor above kRmax is
// one chunked stage (kChunk conjugate output pairs per work item), scheduled first, where P = 1.
#pragma once

namespace kdft {

constexpr int kRmax = 16;        // whole-butterfly radix bound (R outputs in registers)
constexpr int kStageCost = 4;    // a stage's fixed cost (LDS round trip + 2 barriers), in FMAs/point
constexpr int kChunk = 8;        // outputs per work item of a radix above kRmax
constexpr int kMaxStages = 8;

constexpr int smallest_factor(int n) {
    for (int f = 2; f * f <= n; ++f)
        if (n % f == 0) return f;
    return n;
}
constexpr int plan_cost(int n) {
    if (n <= 1) return 0;
    int best = -1;
    for (int d = 2; d <= kRmax; ++d)
        if (n % d == 0) {
            const int c = d + kStageCost + plan_cost(n / d);
            if (best < 0 || c < best) best = c;
        }
    if (best < 0) {
        const int f = smallest_factor(n);
        best = f + kStageCost + plan_cost(n / f);
    }
    return best;
}
constexpr int plan_first(int n) {
    int best = -1, arg = n;
    for (int d = 2; d <= kRmax; ++d)
        if (n % d == 0) {
            const int c = d + kStageCost + plan_cost(n / d);
            if (best < 0 || c < best) { best = c; arg = d; }
        }
    return best < 0 ? smallest_factor(n) : arg;
}

struct Plan { int n; int r[kMaxStages]; int p[kMaxStages]; };

constexpr Plan make_plan(int N) {
    Plan pl{0, {}, {}};
    int big[kMaxStages] = {}, small[kMaxStages] = {}, nb = 0, ns = 0;
    for (int m = N; m > 1;) {
        const int r = plan_first(m);
        if (r > kRmax) big[nb++] = r; else small[ns++] = r;
        m /= r;
    }
    for (int a = 0; a < ns; ++a)                       // small radices in descending order
        for (int b = a + 1; b < ns; ++b)
            if (small[b] > small[a]) { const int t = small[a]; small[a] = small[b]; small[b] = t; }
    int prod = 1;
    for (int a = 0; a < nb + ns; ++a) {
        pl.r[a] = a < nb ? big[a] : small[a - nb];
        pl.p[a] = prod;
        prod *= pl.r[a];
    }
    pl.n = nb + ns;
    return pl;
}
// Work items of one stage: lines (G/N)*W, butterflies N/R per line, chunks per butterfly.
constexpr long long stage_items(int N, int R, long long G, int W) {
    const int chunks = R <= kRmax ? 1 : ((R - 1) / 2 + kChunk - 1) / kChunk;
    return (G / N) * W * (N / R) * chunks;
}
// The most work items of any stage of an axis of length N in a (G, W) tile.
constexpr long long axis_items_max(int N, long long G, int W) {
    const Plan pl = make_plan(N);
    long long m = 0;
    for (int s = 0; s < pl.n; ++s) {
        const long long it = stage_items(N, pl.r[s], G, W);
        if (it > m) m = it;
    }
    return m;
}

}  // namespace kdft
