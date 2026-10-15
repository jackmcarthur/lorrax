// kdft_plan.h -- the kdft line-DFT radix plan: pure constexpr C++, included by the device text
// (kdft_core.h, through the RTC's embedded headers) and by the host handler (kdft_klead_ffi.cc,
// which sizes a pass's thread count from it), so the plan has one definition.
//
// make_plan(N) factors an axis of length N into Stockham stages: it minimises the summed stage
// cost over factorisations into radices <= kRmax, a stage costing its butterfly's work per point
// (radix_cost: a power of two up to 16 is a split-radix FFT butterfly, any other radix R the direct
// R-point sum, R complex FMAs per point), kStageCost for its LDS round trip, and one twiddle
// multiply per point after the first stage.  A prime factor above kRmax is one chunked stage
// (kChunk conjugate output pairs per work item), scheduled first, where P = 1.
#pragma once

// The plan is evaluated on both sides: in device code (template arguments of kdft_core.h) and
// on the host (the handlers).  Under a device compiler (hipRTC, NVRTC, hipcc, nvcc) its functions
// are __host__ __device__, so the text needs no relaxed-constexpr or -default-device option; the
// host C++ compiler sees plain constexpr.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__) || defined(__HIPCC__) || defined(__HIPCC_RTC__)
#define LRX_HD __host__ __device__
#else
#define LRX_HD
#endif

namespace kdft {

constexpr int kRmax = 16;        // whole-butterfly radix bound (R outputs in registers)
constexpr int kStageCost = 4;    // a stage's fixed cost (LDS round trip + 2 barriers), in FMAs/point
constexpr int kChunk = 8;        // outputs per work item of a radix above kRmax
constexpr int kMaxStages = 8;
constexpr int kRegMax = 32;      // longest axis the register line DFT holds (line_reg, kdft_core.h)

LRX_HD constexpr int smallest_factor(int n) {
    for (int f = 2; f * f <= n; ++f)
        if (n % f == 0) return f;
    return n;
}
// Work per point of one radix-R butterfly, in complex FMAs: the FFT-structured radices 2, 4, 8, 16
// (kdft_core.h bfly) take log-depth add/sub trees with constant twiddles; any other radix the
// direct R-term sum.
LRX_HD constexpr int radix_cost(int r) {
    return r == 2 ? 1 : r == 4 ? 2 : r == 8 ? 4 : r == 16 ? 7 : r;
}
LRX_HD constexpr int plan_cost(int n) {
    if (n <= 1) return 0;
    int best = -1;
    for (int d = 2; d <= kRmax; ++d)
        if (n % d == 0) {
            const int c = radix_cost(d) + kStageCost + (n / d > 1 ? 1 : 0) + plan_cost(n / d);
            if (best < 0 || c < best) best = c;
        }
    if (best < 0) {
        const int f = smallest_factor(n);
        best = f + kStageCost + (n / f > 1 ? 1 : 0) + plan_cost(n / f);
    }
    return best;
}
LRX_HD constexpr int plan_first(int n) {
    int best = -1, arg = n;
    for (int d = 2; d <= kRmax; ++d)
        if (n % d == 0) {
            const int c = radix_cost(d) + kStageCost + (n / d > 1 ? 1 : 0) + plan_cost(n / d);
            if (best < 0 || c < best) { best = c; arg = d; }
        }
    return best < 0 ? smallest_factor(n) : arg;
}

struct Plan { int n; int r[kMaxStages]; int p[kMaxStages]; };

LRX_HD constexpr Plan make_plan(int N) {
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
// cos and sin of 2 pi m / n at compile time (the constant twiddles of kdft_core.h's butterflies
// and register DFT): m reduced mod n and to the nearest quadrant exactly in integers, so the
// series argument is |x| <= pi/4 and its error is a few ulp, independent of m and n.
LRX_HD constexpr double trig2pi(long long m, long long n, int want_sin) {
    m %= n;
    if (m < 0) m += n;
    const long long q = (8 * m + n) / (2 * n);                // the nearest quadrant: round(4m / n)
    const long long rem = 4 * m - q * n;                       // x = (pi/2) * rem / n, |x| <= pi/4
    const double x = 1.57079632679489661923 * double(rem) / double(n);
    const double x2 = x * x;
    double sn = x, cs = 1, ts = x, tc = 1;
    for (int k = 1; k <= 12; ++k) {
        ts *= -x2 / double((2 * k) * (2 * k + 1));
        tc *= -x2 / double((2 * k - 1) * (2 * k));
        sn += ts;
        cs += tc;
    }
    const int qq = static_cast<int>(q % 4);
    const double c = qq == 0 ? cs : qq == 1 ? -sn : qq == 2 ? -cs : sn;
    const double s = qq == 0 ? sn : qq == 1 ? cs : qq == 2 ? -sn : -cs;
    return want_sin ? s : c;
}
// Whether an axis of length n takes the register line DFT: n <= kRegMax and every radix <= kRmax.
LRX_HD constexpr bool reg_ok(int n) {
    if (n > kRegMax) return false;
    const Plan pl = make_plan(n);
    for (int s = 0; s < pl.n; ++s)
        if (pl.r[s] > kRmax) return false;
    return true;
}
// Work items of one stage: lines (G/N)*W, butterflies N/R per line, chunks per butterfly.
LRX_HD constexpr long long stage_items(int N, int R, long long G, int W) {
    const int chunks = R <= kRmax ? 1 : ((R - 1) / 2 + kChunk - 1) / kChunk;
    return (G / N) * W * (N / R) * chunks;
}
// The most work items of any stage of an axis of length N in a (G, W) tile.
LRX_HD constexpr long long axis_items_max(int N, long long G, int W) {
    const Plan pl = make_plan(N);
    long long m = 0;
    for (int s = 0; s < pl.n; ++s) {
        const long long it = stage_items(N, pl.r[s], G, W);
        if (it > m) m = it;
    }
    return m;
}

}  // namespace kdft
