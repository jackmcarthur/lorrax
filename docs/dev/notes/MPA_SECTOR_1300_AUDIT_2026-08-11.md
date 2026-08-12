# Why the MPA sector pass is 6,145 tau dispatches, not 1,336

Date: 2026-08-11

Status: theory/history audit against the accepted sector implementation and its
completed production-scale parity run. This report did not run a cluster job,
change code, or change a fit.

## Executive answer

The earlier “about 1,300” number was **not a measured prediction of the sector
algorithm that was ultimately implemented**. It was the explicit architectural
benchmark

$$
8\ \text{MPA poles}\times 167\ \text{GN tau nodes}=1{,}336,
$$

where 167 was the node count of one complete accepted GN-PPM calculation. The
agent that introduced it called it a “best-case architectural reference” and
warned that the narrow crossing core could cost more. It was a useful statement
of ambition—each MPA pole should cost on the order of one GN pole model—not a
quadrature estimate derived from the fitted pole field.

Three mismatches prevent that benchmark from describing today's count:

1. The GN count comes from small real-axis minimax/HGL rules at a nominal
   $10^{-6}$ rule target. The new sign-definite MPA rule is an optimizer-free,
   analytic-proof sinc rule over the *entire fourth-quadrant sector* at
   $10^{-8}$. Its proof and two infinite-tail truncations cost 121–140 nodes per
   sign-definite production group.
2. The shallow exact-width crossing cores are a different, harder complex
   resolvent problem. They cost 1,038 nodes. The largest individual ranks,
   179–220, are these crossing groups, not the sign-definite sector groups.
3. To establish a clean speed/BGW-parity result, the implementation deliberately
   retained the accepted $\Gamma<\xi$ compatibility route. It costs another 469
   dispatches. Removing it would change the self-energy functional; it is not a
   free optimization.

The exact source-matched census is therefore

$$
469\ \text{legacy}
+4{,}638\ \text{sign-definite sector}
+1{,}038\ \text{exact crossing core}
=6{,}145.
$$

That result is successful: the full P16 calculation completed with exit code 0
in 415 s and reproduced the old accepted output and BGW metrics. It is the
control, not a failed experiment.

At the current tolerance and with the identical functional, **1,300 is not a
credible next-step promise**. After the fixed 469-node compatibility route it
would leave only 831 nodes for 44 wide-pole groups, or 18.9 nodes per group.
Reaching that would require roughly a six- to eight-fold rank reduction across
both the sector and hard crossing families, or genuine cross-group reuse of the
same expensive $G(\tau)$ dispatches. Merely renaming or merging groups does not
do that.

The best credible near-term target is **about 1,800–2,500 actual tau-kernel
dispatches**, with a midpoint estimate of 1,969, using certified near-minimax
complex exponential sums on the *actual coupled annular wedges* and a
multi-shift/shared-node execution layout. About 1,300 remains a worthwhile
research target only if common nodes truly serve several branches or poles, or
if a separately validated change is made to pole order, tolerance, $\eta$, or
the narrow-pole functional.

## 1. What is being counted

Several quantities were called “samples,” “nodes,” or “builds” in different
parts of the discussion. They are not interchangeable.

| Quantity | Meaning | Current Si $n_p=8$ example | Is it in 6,145? |
|---|---|---:|---|
| Logical $W(z)$ fit sample | One matrix-valued screened-interaction value at a prescribed complex frequency | $2n_p=16$ | No |
| Minimax/Laplace rule node | One scalar node/weight in an HGL, composite, or exact complex-crossing approximation | 469 legacy plus 1,038 crossing nodes | Only when dispatched by $\Sigma_c$ |
| Sector sinc node | One $s_k,w_k$ term in the rotated complex-Laplace sum | 4,638 across 36 sign-definite groups | Yes |
| Window piece | A core, A-stripe, B-slab, or noncrossing piece with one scalar rule | Eight pieces in the accepted GN reference | Yes, through its nodes |
| Window group | A pole-mask plus one or more window pieces that share routing metadata | 76 census rows: 32 legacy, 44 new | Not by itself |
| Tau-kernel dispatch | One high-level evaluation at one tau node for one window: construct the selected $G$, apply the pole-resolved $W$ phase/mask, FFT/multiply/inverse-transform, and project | 6,145 | This is the cost numerator |

The $2n_p$ fit samples are logical outputs of the **MPA-$W$ construction**.
They are the inputs from which the pole field $(B_p,\Omega_p)$ is fitted. A
line-batched construction may share propagators over many such outputs, but it
must still report both logical samples and its actual tau-node sweeps. None of
that work is part of the present 6,145-node **MPA-$\Sigma$ consumption** census:
the run reads an already fitted pole store.

Likewise, a quadrature node is only a mathematical term until the production
planner executes it. The current implementation usually dispatches each
window's nodes independently. If two groups each contain a 128-node rule, the
present cost is 256 dispatches even if the two node arrays happen to have the
same length. A future “shared rule” reduces cost only if one constructed
$G(\tau)$ and associated spatial transform actually feeds both accumulations.

This distinction was already required by the August 7 theory plan:
`/home/jackm/MPA_THEORY_PLAN.md` says reports must separate $2n_p$ logical
outputs, actual tau-node dispatches normalized to a GN sweep, and downstream
transforms/solves. The 1,336 remark referred specifically to expensive
$\Sigma_c$ tau contractions, not to $W(z)$ fitting, Dyson solves, or total job
work.

## 2. How the theory and implementation evolved

### 2.1 August 7: complex-pole theory, before the 1,300 claim

The Claude transcript
`/home/jackm/.claude/projects/-mnt-c-Users-jackm/03f6f490-2dba-4b15-9713-bdf9042d2e3f.jsonl`
and its consolidated result, `/home/jackm/MPA_THEORY_PLAN.md`, established the
following points:

- The MPA fit has exactly $2n_p$ logical complex-frequency samples of $W_c$.
  Deep continuum sampling belongs to this fit stage and does not widen the
  real-frequency $\Sigma$ crossing window.
- The fitted poles are genuinely complex,
  $\Omega_p=a_p-i\Gamma_p$. For the strict complex-pole functional the
  $\Sigma$ consumer needs the full complex resolvent, not the old sine-only HGL
  object.
- On a sign-definite branch, $\Gamma_p$ is a phase in imaginary time rather
  than additional decay. The shipped real $1/y$ tables had already been found
  inaccurate by two to three orders of magnitude on the required complex
  strip, so a complex-Laplace family was required.
- The four GN branches, no-transition-tensor scaling, and one-pole memory
  discipline were non-negotiable. The plan explicitly rejected a design that
  materialized transition amplitudes or ran unnecessary pole-proportional
  spatial kernels.
- The theoretical reference configuration kept exact fitted widths and no
  broadening floor, while the GN $n_p=1$ path remained a separate compatibility
  anchor. It explicitly did not claim that HGL was the continuous
  $\Gamma\rightarrow0$ limit.

There is no 1,300-node derivation in that plan. The number arose later, after a
production cost audit.

### 2.2 August 11: the measured 52,252-node failure mode

The first exact census of the accepted pane planner was 1,148 groups and
52,252 tau dispatches:

| Pole | Old groups | Old tau dispatches |
|---:|---:|---:|
| 0 | 104 | 10,763 |
| 1 | 138 | 8,199 |
| 2 | 128 | 5,944 |
| 3 | 128 | 4,745 |
| 4 | 134 | 4,689 |
| 5 | 126 | 3,932 |
| 6 | 126 | 3,782 |
| 7 | 264 | 10,198 |
| **Total** | **1,148** | **52,252** |

The important decomposition was:

| Old route | Groups | Dispatches |
|---|---:|---:|
| Sign-definite width panes | 1,036 | 44,842 |
| Complex crossing | 80 | 6,941 |
| $\Gamma<\xi$ legacy route | 32 | 469 |

Thus 85.8% of the cost was not an unavoidable crossing singularity. It came
from replacing the coupled physical cloud $(a_i,\Gamma_i)$ by rectangular
envelopes. A pane could combine one element's $\Gamma_{\max}$ with another
element's smallest real denominator, fail the condition
$\Gamma_{\max}/x_{\min}\le1$, split again, and then rerun the entire expensive
kernel for a tiny pole mask. The mask selection was cheap; each pane's
$G$/FFT/projection pass was not.

A read-only reconstruction of the already available ratio-four binned bridge
reduced the total to 11,631 while leaving crossing unchanged. That proved most
of 52,252 was planner-induced, but it retained nonphysical width bins and was
still about seventy complete GN calculations.

These measurements and the corresponding diagnosis are preserved in
`docs/theory/2026-08-11-mpa-theory-performance-audit.md` and in the Codex theory
child transcript
`/home/jackm/.codex/sessions/2026/08/11/rollout-2026-08-11T14-09-43-019ff2a8-e9a5-7d31-a6fd-18300bf36eef.jsonl`.

### 2.3 Where 1,336 came from

In the main Codex transcript
`/home/jackm/.codex/sessions/2026/08/11/rollout-2026-08-11T11-43-50-019ff223-5914-7993-a4a0-c3a1faa9aacb.jsonl`,
the user asked whether “1,300 nodes total” meant 1,300 times the cost of a GN
calculation. The reply made the intended arithmetic explicit:

> one complete GN-PPM calculation used 167 tau nodes—84 positive half, 83
> negative; $1{,}336=8\times167$ is eight GN-sized pole passes total.

The same reply called this a clean architectural benchmark rather than a
promise, and warned that tiny $\Gamma$ could make the crossing core more
expensive. A subsequent reply said that it counted only expensive $\Sigma_c$
tau contractions and excluded constructing/fitting $W(z)$.

The longer audit gave eta-dependent planning ranges rather than a 1,336
prediction: approximately 1,500–3,000 nodes for broad $\eta=0.667$ eV,
2,500–5,000 for $\eta=0.25$ eV, and 5,000–8,000 for $\eta=0.1$ eV. It again
called $8\times167$ a best-case architecture baseline, not a lower bound.

The shorthand later hardened into a target without carrying its caveats. More
importantly, no scalar rule campaign had yet supplied ranks for the exact
production pole cloud.

### 2.4 The safeguarded sector implementation

The replacement design kept the useful part of the diagnosis—coupled pole
geometry and fixed GN-like branch topology—but introduced safeguards before
claiming a physics-preserving speedup:

1. **One coupled sector per sign-definite piece.** Widths determine the exact
   complex phase but no longer create sign-definite panes.
2. **An analytic proof, not a sampled fit.** The new sinc rule covers every
   denominator in the full fourth quadrant and exposes a rigorous error bound.
3. **Exact complex crossing cores.** Shallow crossing modes still use a
   complex-resolvent rule and geometric width bands; deep B-slabs and A-stripes
   become sign-definite sector pieces.
4. **Compatibility held fixed.** Contrary to the earlier strict-theory proposal
   to remove the $\Gamma<\xi$ substitution, the speed-first implementation
   preserved it verbatim so the comparison changed cost rather than broadening
   physics. This decision is explicit in `src/gw/mpa/sigma_pass.py:1386-1417`
   and `docs/input_reference.md:106`.
5. **Stable combined phases.** The $E_A$, pole-reference, and output-frequency
   exponent was fused in `src/gw/ppm_accumulators.py` and
   `src/gw/ppm_sigma.py`; this avoids separately enormous complex factors along
   the rotated contour.
6. **Distributed evidence made source-exact.** Commit `6a0f06c3` matched live
   centroid identity to the screening artifacts, and `bce81eb1` padded wedge
   pole fields at the reader mesh. They do not change quadrature mathematics
   or explain the node count; they make the distributed run consume the
   intended field reliably.

Commit `05384d54` is the implementation that changes the quadrature and routing.
It added `services/minimax/src/minimax/sector.py`, the sector planner in
`src/gw/mpa/sigma_pass.py`, the routing adapter in
`src/gw/mpa/sigma_routing.py`, stable phase handling, and plan/provenance
support. Commits `6a0f06c3` and `bce81eb1` are evidence-integrity fixes after
that mathematical change.

### 2.5 The measured outcome

The canonical source-exact census at
`/pscratch/sd/j/jackm/mpa_sector_0811/census/sector_all.json`, with log
`/pscratch/sd/j/jackm/mpa_sector_0811/logs/census_sector.log`, records 76 groups
and 6,145 tau dispatches. The completed P16 result and its metrics are recorded
under `/pscratch/sd/j/jackm/mpa_sector_0811/logs/full_p16.log` and
`/pscratch/sd/j/jackm/mpa_sector_0811/metrics_sector_bgw.json`. It exited 0 in
415 s and reproduced the old accepted result/BGW comparison. The direct
new-minus-old $\Sigma$ cube difference was at most $3.46\times10^{-9}$ eV, with
an RMS of $2.46\times10^{-10}$ eV; QP changes were at the micro-meV level. This
establishes functional parity for the speed-first route.

## 3. The current sector rule

### 3.1 Denominator geometry

For a sign-definite branch, write a served denominator as

$$
d=x-i\Gamma,
\qquad x>0,
\qquad \Gamma\ge0.
$$

Every such $d$ lies in the fourth quadrant. Rotating the Laplace contour by
$\theta=\pi/4$ gives the exact identity

$$
\frac{1}{d}
=e^{i\theta}\int_0^\infty
\exp\!\left[-d e^{i\theta}s\right],ds.
$$

The worst-case decay after rotation is

$$
\Re\!\left(de^{i\pi/4}\right)
=\frac{x+\Gamma}{\sqrt2}
\ge \frac{|d|}{\sqrt2}.
$$

The production planner forms $r_{\min}$ and $r_{\max}$ from the actual coupled
tuples using `hypot(x, Gamma)`; it does not pair an unrelated largest width and
smallest real part. See `src/gw/mpa/sigma_pass.py:1362-1384` and
`src/gw/mpa/sigma_routing.py:576-604`.

### 3.2 Log-sinc discretization and proof

Set $s=e^y/r_{\min}$ and apply an equally spaced trapezoidal rule in $y$:

$$
s_k=\frac{e^{kh}}{r_{\min}},
\qquad
w_k=e^{i\theta}\frac{h e^{kh}}{r_{\min}},
$$

$$
\frac{1}{d}
\approx
Q(d)=\sum_{k=k_{\min}}^{k_{\max}}
w_k\exp\!\left[-d e^{i\theta}s_k\right].
$$

Let

$$
R=\frac{r_{\max}}{r_{\min}},
\qquad c=\cos\theta=\frac{1}{\sqrt2}.
$$

For a proof margin $m$, the analytic strip has half-width
$a=\theta-m$ and constant $C=2/\sin m$. The implementation chooses

$$
h=\frac{2\pi a}{\log(1+3C/\varepsilon)}
$$

and finite endpoints so that the three relative-error terms obey

$$
E_{\rm strip}
\le \frac{C}{\exp(2\pi a/h)-1},
$$

$$
E_{\rm left}
\le
\frac{h e^{h(k_{\min}-1)}}{1-e^{-h}}R,
$$

$$
E_{\rm right}
\le
\frac{R}{c}\exp[-c e^{h k_{\max}}],
$$

with each budgeted at approximately $\varepsilon/3$. The code searches only
the proof margin $m$ for the smallest integer rank; it does not optimize sampled
denominators or fit a minimax table. The implementation is
`services/minimax/src/minimax/sector.py:20-143`.

The returned absolute-weight amplification at the least-damped edge is

$$
\kappa_0
=r_{\min}\sum_k |w_k|e^{-r_{\min}c s_k}.
$$

Its infinite-grid integral limit is $1/c=\sqrt2$, so the construction buys its
conservative rank with well-controlled weights rather than a fragile
cancellation. A lower-rank fitted rule must preserve that stability advantage.

With $L=\log(1/\varepsilon)$, this construction has approximately

$$
h=O(L^{-1}),
$$

$$
N=k_{\max}-k_{\min}+1
=O\!\left(
\frac{L}{a}
\left[L+\log R+\log(L+\log R)\right]
\right).
$$

Thus the rank is logarithmic in radial ratio $R$ only when tolerance is held
fixed; at high accuracy the conservative sinc proof also carries an
approximately quadratic dependence on $\log(1/\varepsilon)$. This is why the
docstring shorthand “rank grows with the logarithm of the radial range” is true
but incomplete as an explanation of the production ranks.

### 3.3 Why it is not one small GN-like rule

The accepted 167-node GN calculation is composed of eight small rules over a
real/HGL problem at a $10^{-6}$ target. The sector construction asks one rule to
cover a two-dimensional complex set, including both quadrant boundaries, at
$10^{-8}$, while proving both tails and the infinite trapezoid error without an
optimized node set. Even a group with a modest $R$ inherits a substantial
accuracy-driven floor from the negative-$k$ tail and strip spacing. Increasing
$R$ then extends that log grid further.

There is one numerical correction to the loose statement that “sign-definite
groups cost 125–220 nodes.” In the exact census:

- the 36 sign-definite `single`, `a_stripe`, and `b_slab` groups cost **121–140**
  nodes each;
- the eight exact crossing-core groups cost 58, 62, 179, or 220 nodes;
- the 179–220 ranks belong to the narrowest shallow crossing bands of poles 0
  and 1.

The broad observed 125–220 scale therefore combines two mechanisms. The
roughly 125–140 sector rank comes from the full-sector $10^{-8}$ sinc proof. The
179–220 hard-core rank comes from resolving a long frequency interval relative
to a small exact $\Gamma_{\min}$ in the complex crossing resolvent. Neither is
the tiny real-axis GN table that the $8\times167$ analogy implicitly assumed.

## 4. Exact accounting of all 6,145 dispatches

### 4.1 By route

| Current route | Groups | Tau dispatches | Mathematical role |
|---|---:|---:|---|
| `legacy` | 32 | 469 | Accepted $\Gamma<\xi$ compatibility bridge |
| `sector:single` | 16 | 2,036 | Two globally noncrossing branches for eight poles |
| `sector:a_stripe` | 4 | 520 | Sign-definite A-stripes of shallow crossing poles |
| `sector:b_slab` | 16 | 2,082 | Sign-definite deep-pole slabs |
| `sector:core:g0` | 4 | 798 | Harder exact-width shallow crossing bands |
| `sector:core:g1` | 4 | 240 | Softer exact-width shallow crossing bands |
| **Total** | **76** | **6,145** | |

Equivalently,

$$
N_{\rm sign-def}=2{,}036+520+2{,}082=4{,}638,
$$

$$
N_{\rm crossing}=798+240=1{,}038,
$$

$$
N_{\rm all}=4{,}638+1{,}038+469=6{,}145.
$$

The census's coarse route label calls all 5,676 non-legacy nodes “sector.” In
mathematical terms only 4,638 are sector-sinc nodes; 1,038 are exact complex
crossing-rule nodes. That naming distinction is essential when estimating a
replacement rule.

This is already an 8.50-fold reduction from 52,252. It removes 1,000 of the old
1,036 sign-definite pane groups while retaining the accepted narrow route.

### 4.2 By pole

| Pole | Legacy | New sector/crossing | Total |
|---:|---:|---:|---:|
| 0 | 152 | 1,376 | 1,528 |
| 1 | 145 | 1,263 | 1,408 |
| 2 | 31 | 513 | 544 |
| 3 | 29 | 503 | 532 |
| 4 | 28 | 504 | 532 |
| 5 | 28 | 495 | 523 |
| 6 | 28 | 492 | 520 |
| 7 | 28 | 530 | 558 |
| **Total** | **469** | **5,676** | **6,145** |

Poles 0 and 1 dominate because they contain the shallow exact-width crossing
bands. Poles 2–7 are already near 520–558 total dispatches each; their main
remaining cost is the proof-bounded sign-definite sector rank, not a return of
the pane explosion.

## 5. Was 1,300 achievable under the stated conditions?

### 5.1 As a benchmark: yes

The benchmark expressed the right architectural concern. The earlier planner
was paying 31–64 complete GN calculations per MPA pole, and pole 0 alone paid
about 64. It was reasonable to demand that one pole pass return to a bounded
multiple of one GN calculation. The sector implementation vindicates the
diagnosis by reducing the total 8.50-fold without changing the accepted answer.

### 5.2 As a quantitative prediction of this rule: no

No production radial ratios, exact crossing bandwidths, or certified complex
rule ranks were used to derive 1,336. It conflated:

- a whole-job GN node count with a per-pole MPA aspiration;
- real/HGL minimax rules with a full-complex-sector sinc proof;
- a $10^{-6}$ GN rule target with a $10^{-8}$ sector target;
- logical branch topology with independent actual dispatches;
- and, when repeated without qualification, $\Sigma$ contraction cost with
  total MPA work including $W(z)$ fitting.

The current arithmetic makes the gap concrete. With 44 independently executed
wide groups and the fixed 469-node compatibility route, a 1,336 total allows

$$
\frac{1{,}336-469}{44}=19.7
$$

nodes per wide group. A literal 1,300 allows 18.9. That is close to the average
size of one *piece* in the easier GN calculation, but it is not consistent with
uniform $10^{-8}$ complex approximation on the current domains unless the
rules or execution sharing improve radically.

A read-only re-evaluation of the current analytic sector formula on the 36
production sign-definite radial ranges gives:

| Sector tolerance | Sign-definite nodes only | Total if crossing and legacy stay fixed |
|---:|---:|---:|
| $10^{-8}$ | 4,638 | 6,145 |
| $10^{-7}$ | 3,794 | 5,301 |
| $10^{-6}$ | 3,035 | 4,542 |
| $10^{-5}$ | 2,357 | 3,864 |

Tolerance relaxation alone therefore does not recover 1,300 for this sinc
family. It also would not preserve the current numerical contract unless an
end-to-end error budget justified the new tier.

### 5.3 “Identical functional” is a real constraint

The accepted control includes the $\Gamma<\xi$ HGL compatibility substitution.
The earlier strict theory preferred retaining every positive fitted width in
the $\eta_\Sigma=0$ quasiparticle limit. That may be the cleaner future
functional, but changing it is not a dispatch optimization: it changes the
analytic self-energy near sharp poles and removes the very route responsible
for the 469 fixed nodes.

Similarly, adding a finite $\eta_\Sigma$ can make hard crossing rules cheaper by
replacing $\Gamma$ with $\Gamma+\eta_\Sigma$, but it computes a broadened
self-energy. Reducing $n_p$ changes the rational model of $W$. Loosening the
quadrature tolerance changes the numerical error contract. Any of these may be
good after validation, but none can be advertised as the same-functional
explanation of a lower count.

## 6. Levers that can actually reduce dispatches

### 6.1 Better complex-sector approximation

The sinc rule is an excellent certification oracle because it is deterministic,
stable, and proof-bounded. It is not expected to be rank-optimal. A rational or
near-minimax exponential-sum construction should approximate

$$
d^{-1},\qquad d\in
\{r e^{-i\phi}:r\in[r_{\min},r_{\max}],\ \phi\in[0,\phi_{\max}]\},
$$

with substantially fewer complex-time nodes. The production groups usually
occupy a narrower angular cloud than the full $\phi\in[0,\pi/2]$ proof domain.
Training and certifying the actual annular wedge, while retaining the sinc rule
as an independent oracle, attacks rank directly without changing the
functional.

Promising formulations include bounded-amplification complex exponential
sums, rational/Zolotarev-like approximants converted to Laplace nodes, and
double-exponential maps. The certificate must cover complex modulus uniformly,
report an amplification measure such as

$$
\kappa_0(r)=r\sum_j |w_j|e^{-r\Re(c_j)},
$$

and independently maximize residuals over the two-dimensional wedge. A low
training residual with large cancellation is not sufficient.

### 6.2 Tighter geometry and bounds

The current rule throws away three kinds of useful information:

- the actual maximum pole angle may be much smaller than $\pi/2$;
- radius and angle are correlated in the fitted cloud;
- the tail proof multiplies conservative extrema by the global radial ratio.

Using a certified polygonal or union-of-annular-wedges enclosure can widen the
analytic strip and tighten both tail bounds. Radial splitting is legitimate
only when the sum of ranks of the split rules is smaller and the additional
dispatch groups do not erase the gain. This is the opposite of the old width
pane tree: splits must be justified by total node economics, not by a
rectangular clause.

### 6.3 A simultaneous multi-shift crossing rule

The four hard `core:g0` rows repeat the same physical problem across symmetry
halves: a complex resolvent over a wide real interval and a small fitted-width
band. A vector or simultaneous multi-shift rule can target the union of those
shifts with common nodes, while each pole retains its exact $\Gamma$ in the
weights/phase. A two-band construction for `g0` and `g1` should be compared
against the current geometric bins by **sum of actual shared dispatches**, not
by number of catalog entries.

The shallow/near-real limit has an unavoidable time-bandwidth cost, so the hard
cores are unlikely to collapse to 20 nodes each at $10^{-8}$. The objective is
to share the long-time nodes and avoid paying nearly the same $G(\tau)$ build
twice.

### 6.4 Branch and pole reuse

Common-node rules matter only if execution is coupled. For a node $t_j$, the
implementation should construct each distinct A-side propagator/mask once,
stream the permitted one-pole B/residue slab, accumulate every compatible
branch or shift, and perform the spatial transforms/projections once where the
algebra permits. This preserves the no-transition-amplitude and one-pole-memory
constraints.

It is important to distinguish three claims:

| Change | Fewer groups? | Fewer actual dispatches? |
|---|---:|---:|
| Put two rules in one metadata object | Yes | No |
| Give two groups identical node arrays but run both loops | Maybe | No |
| Build $G(t_j)$ once and feed both exact accumulations | Maybe | Yes |

Cross-pole reuse is harder than within-pole branch reuse because $W_p(t)$ and
pole masks differ, and the low-memory design does not hold several pole slabs.
It should be attempted only through streamed accumulators, never by
materializing transition amplitudes or all-pole spatial kernels.

### 6.5 End-to-end error allocation

Uniform $10^{-8}$ relative error for every scalar sector is conservative
relative to meV-level quasiparticle goals. A rigorous weighted budget could
assign tolerance by residue norm, branch contribution, and sensitivity of the
Dyson root. This may reduce ranks. It must be validated against the current
$10^{-8}$ control and include cancellation/amplification; dropping “small” pole
elements or using a residue-weighted training norm alone is not a certificate.

The table in Section 5.2 shows that tolerance allocation cannot by itself make
the present sinc map a 1,300-node method. It is a complementary lever after a
better rule family exists.

### 6.6 Pole count and $W$ construction

Choosing the smallest $n_p$ that passes held-out $W_c(z)$ and quasiparticle
checks reduces both $2n_p$ fit outputs and approximately pole-proportional
$\Sigma$ work. This was part of the August 7 plan. It is a model-order decision,
not an identical-functional optimization. Likewise, line-shared construction
of the 16 $W(z)$ samples can reduce *fit-stage* propagator sweeps but does not
alter the 6,145 $\Sigma$ census. Reports must keep those savings separate.

## 7. Recommended next algorithm and count estimate

The cleanest next method is a **certified coupled-annulus, shared-node complex
quadrature**:

1. Keep the present four-branch algebra, exact fitted pole store, one-pole
   streaming, and $\Gamma<\xi$ compatibility route fixed for the first A/B.
2. For every sign-definite production group, enclose the actual coupled
   denominator cloud by the smallest simple annular wedge or small union of
   wedges. Generate a bounded-amplification near-minimax exponential sum on
   that domain. Certify it independently against the analytic sinc rule and an
   adaptive complex-domain residual maximizer.
3. Constrain symmetry-related branches to use common nodes when the union
   domain does not materially increase rank. Execute those nodes once per
   distinct A-side propagator and stream the exact pole weights into the
   appropriate accumulators.
4. Replace `core:g0/g1` only after a simultaneous multi-shift complex-resolvent
   campaign demonstrates fewer *executed* nodes than the current two-band
   rules. Preserve each element's exact $\Gamma$; bins may select a rule but
   must not round widths.
5. Keep the present sinc and exact-width crossing rules as runtime fallbacks and
   certification oracles. A difficult pole should cost more, not silently leave
   its certified domain.

This remains legible in the physics: the rational $W$ model and four causal
branches are unchanged; only the integral representation and scheduling of the
same resolvents improve. It creates no transition tensor and requires no
decorative per-element checks in the hot loop.

An explicit planning estimate at $10^{-8}$ is:

| Component | Assumed optimized ranks | Estimated dispatches |
|---|---:|---:|
| 36 sign-definite groups | 25–40 each | 900–1,440 |
| Four hard `core:g0` groups | 60–90 each | 240–360 |
| Four softer `core:g1` groups | 25–40 each | 100–160 |
| Fixed legacy compatibility route | measured | 469 |
| **Estimated total before additional sharing** | | **1,709–2,429** |

Allowing for union-domain rank inflation, imperfect shareability, and guard
nodes gives the more honest project target **1,800–2,500**. A useful midpoint
budget is

$$
36\times30 + 4\times75 + 4\times30 + 469 = 1{,}969.
$$

These are engineering estimates, not measured ranks. A 1,300 result would need
either better-than-assumed common-node reuse or an explicit model/tolerance/
functional change. It should be treated as a stretch gate, not the acceptance
criterion for the next implementation.

## 8. Failure modes that must remain visible

- **Shallow or nearly real poles.** As $\Gamma\rightarrow0$ in a crossing core,
  the required real-time horizon and time-bandwidth product grow. No contour
  slogan removes this. The present $\Gamma<\xi$ route hides the hardest limit by
  computing the accepted HGL substitute.
- **Strict QP versus broadened $\Sigma$.** The strict limit is
  $\eta_\Sigma=0$. Adding $\eta_\Sigma$ improves damping and rank but changes
  the observable. It must be named in every comparison.
- **Angular strip collapse.** Domains touching both the positive-real and
  negative-imaginary boundaries constrain contour rotation and the analytic
  strip. A cloud-adapted rule must not certify only sampled interior points.
- **Large radial ratio.** Deep poles and wide output spans can make $R$ large;
  pole 7 is a useful tail stress case even though its present count is modest.
- **Union-domain inflation.** Forcing common nodes across unrelated branches or
  poles can enlarge the domain enough that one shared rule costs more than two
  separate rules.
- **Cancellation and conditioning.** Near-minimax complex weights can achieve a
  small residual by large cancellation. Bounds on $\kappa_0$, node/weight
  perturbations, passivity/causality checks, and direct complex residuals are
  required.
- **HBM and one-pole memory.** A multi-shift execution must stream accumulators;
  it may not buy fewer dispatches by holding an all-pole or transition-sized
  object.
- **QSGW and derivatives.** Rule selection should be deterministic and smooth
  enough for repeated QSGW/phonon workflows. Contribution-dependent adaptive
  branches can introduce discontinuities unless thresholds and fallbacks are
  designed deliberately.
- **Unbalanced error budgets.** Small scalar relative error does not guarantee a
  small quasiparticle error under cancellation or a sensitive Dyson root; the
  reverse is also true. Both levels must be measured.

## 9. Staged comparison, with 6,145 as the control

The current `bce81eb1` sector result is the control throughout. No stage should
replace it until both dispatch accounting and output parity are demonstrated.

1. **Offline scalar census.** On the exact production denominator clouds,
   compare current sinc, candidate coupled-annulus minimax, and candidate
   crossing multi-shift rules. Report logical rules, groups, unique node sets,
   actual shareable dispatches, uniform complex residual, analytic/sampled
   bounds, and amplification separately.
2. **Hard-pole A/B.** Run poles 0 and 1 first because their shallow crossing
   cores determine the risk. Keep the fit store, $\Gamma<\xi$ mask, output grid,
   and branch algebra identical. Compare each denominator family against a
   direct scalar oracle before comparing the accumulated $\Sigma$ cube.
3. **Eight-pole parity.** Reproduce the same $\pm7$ eV production input and
   compare the new $\Sigma$ HDF5 to the 6,145 control, then repeat the accepted
   QP/BGW metrics. Report wall time, dispatches, transforms, and peak HBM; a
   smaller group count alone is not success.
4. **Tolerance ladder.** Only after same-tolerance parity, evaluate
   $10^{-8}$, $10^{-7}$, and $10^{-6}$ under a QP-weighted error budget. This is
   a controlled numerical-contract study, not part of the exact parity claim.
5. **Physics variants.** Test strict fitted-width $\Gamma<\xi$, finite
   $\eta_\Sigma$, and smaller $n_p$ as separately named configurations. Each
   must pass held-out $W_c(z)$, $\Sigma$, and QP gates. Do not mix them into the
   quadrature speed A/B.
6. **Fit-stage optimization.** Measure line-shared production of the 16 logical
   $W(z)$ samples independently. State its tau dispatches and transforms; do
   not add that work to or subtract it from the $\Sigma$ node census.

## 10. Evidence ledger

Primary local discussion and theory evidence:

- Original 1,336 exchange and count clarification:
  `/home/jackm/.codex/sessions/2026/08/11/rollout-2026-08-11T11-43-50-019ff223-5914-7993-a4a0-c3a1faa9aacb.jsonl`
- Full baseline diagnosis and proposed sector ranges:
  `/home/jackm/.codex/sessions/2026/08/11/rollout-2026-08-11T14-09-43-019ff2a8-e9a5-7d31-a6fd-18300bf36eef.jsonl`
- Implementation/parity continuation and exact 6,145 discussion:
  `/home/jackm/.codex/sessions/2026/08/11/rollout-2026-08-11T14-53-43-019ff2d1-2f3a-74f2-8263-fc0b37b0b814.jsonl`
- User prompt index, including the request for GN-like cost and rejection of
  transition-amplitude materialization: `/home/jackm/.codex/history.jsonl`
- Claude complex-pole theory transcript:
  `/home/jackm/.claude/projects/-mnt-c-Users-jackm/03f6f490-2dba-4b15-9713-bdf9042d2e3f.jsonl`
- Consolidated August 7 panel result: `/home/jackm/MPA_THEORY_PLAN.md`
- Historical/current method guide: `docs/mpa_method_guide.md`
- Rewritten performance diagnosis:
  `docs/theory/2026-08-11-mpa-theory-performance-audit.md`

Implementation evidence at branch head `bce81eb1`:

- `services/minimax/src/minimax/sector.py`: exact rotated-contour sinc rule and
  proof bounds.
- `src/gw/mpa/sigma_routing.py`: mapping from sector nodes to complex time.
- `src/gw/mpa/sigma_pass.py`: coupled radial bounds, fixed four-branch geometry,
  exact crossing bands, and the explicit narrow compatibility bridge.
- `src/gw/ppm_accumulators.py` and `src/gw/ppm_sigma.py`: fused complex phase.
- `src/gw/mpa/window_farm.py`: low-memory/window execution context.
- Commit `05384d54`: sector-windowed MPA-$\Sigma$ implementation.
- Commit `6a0f06c3`: live centroid/source identity correction.
- Commit `bce81eb1`: reader-mesh pole-field padding.

Measured evidence:

- `/pscratch/sd/j/jackm/mpa_sector_0811/census/sector_all.json`
- `/pscratch/sd/j/jackm/mpa_sector_0811/logs/census_sector.log`
- `/pscratch/sd/j/jackm/mpa_sector_0811/logs/full_p16.log`
- `/pscratch/sd/j/jackm/mpa_sector_0811/metrics_sector_bgw.json`

The central conclusion follows directly from those records: 1,336 was a
GN-scaled aspiration stated before the complex rule ranks were known; 6,145 is
the measured count of a deliberately conservative, proof-bounded, parity-first
implementation; and the next credible reduction requires a better complex
approximation plus real dispatch reuse, not another pane bookkeeping change.
