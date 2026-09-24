# Multipole frequency integration

`compute_mode = mpa` with `sigma_w_model = mpa` samples $\chi_0$ at $2N_p$
complex frequencies on two lines parallel to the real axis, solves Dyson at
each, and fits every $q$-wedge matrix element of $W_c=W-v$ independently with
$N_p$ complex poles (the multipole approximation of Leon *et al.*). The fitted
poles feed the common $\Sigma(\omega)$ quadrature. `compute_mode = mpa` runs;
`gw_config.UNIMPLEMENTED_MODES` is empty. The alternative body model under the
same mode, `sigma_w_model = shared_pole`, is
[the shared-pole screened interaction](shared-pole-w-model.md).

This page owns the sample geometry, the $\chi$ sample rules, the disk pipeline,
the multipole fit and the MPA-specific parts of $\Sigma$. The frequency
quadrature of $\Sigma$ is [the Σ quadrature problem](sigma-quadrature-problem.md);
finite occupations are [metallic MPA screening](metallic-mpa-screening.md);
deck defaults are the [input reference](../input_reference.md). All runtime
energies are Ry; deck keys ending in `_ev` are converted once.

## 1. One pipeline, three approximations

```text
occupied and empty bands
  -> chi0(z_j) on two complex-frequency lines          (§3, §4)
  -> Wc(z_j) = [1 - V chi0(z_j)]^-1 V - V              (§5)
  -> elementwise pole fit {Omega_p, B_p}               (§6)
  -> Sigma(omega) quadrature, one G(t) W(t) per node   (§7)
  -> q=0 head, interpolation, QSGW operator, output    (§8)
```

The three approximations are separate. The $\chi$ quadrature sets the noise in
the samples; the rational fit sets how well $N_p$ poles reproduce them; the
$\Sigma$ quadrature sets how well the fitted model is integrated. None repairs
another: more poles do not fix noisy samples, and a tighter $\Sigma$ rule does
not fix a poor fit.

## 2. The transition kernel

For a gapped system every included transition is
$\Delta_{cv\mathbf k}=E_g+A_{c\mathbf k}+B_{v\mathbf k}>0$, with $A,B\ge0$ the
distances from the band edges. For $\operatorname{Im}z>0$,

$$
K_z(\Delta)
=\frac{1}{z-\Delta}-\frac{1}{z+\Delta}
=-\frac{2\Delta}{\Delta^2-z^2}
=-2\int_0^\infty e^{izt}\sin(\Delta t)\,dt .
$$

The time integral is the common door for every complex-frequency sample: at
one time node the transition-pair sum factors into one empty-band and one
occupied-band Green function,

$$
G_c(\mathbf r,\mathbf r',t)\sim\sum_{c\mathbf k}\psi_{c\mathbf k}(\mathbf r)\psi^*_{c\mathbf k}(\mathbf r')e^{-i\epsilon_{c\mathbf k}t},
\qquad
G_v(\mathbf r,\mathbf r',-t)\sim\sum_{v\mathbf k}\psi_{v\mathbf k}(\mathbf r)\psi^*_{v\mathbf k}(\mathbf r')e^{+i\epsilon_{v\mathbf k}t},
$$

contracted in the ISDF basis. Two band sums instead of a pair sum is what keeps
the cost cubic.

### 2.1 Ordered orientations when time reversal is broken

The symmetric kernel above completes one transition carrier only under global
time reversal. That is measured, not declared: `SymMaps.trs_allowed` carries
the verdict of [the symmetry service](../services/symmetry_maps.md#contract);
no deck key asserts it. On a measured-broken-TR insulator, with the contour
kernel's native orientation

$$
F_{\mathbf q}(z)=-\sum_{vc\mathbf k}\frac{P^{\mathbf q}_{vc\mathbf k}}{z+\Delta_{vc\mathbf k}},
\qquad
\chi^0_{\mathbf q}(z)=F_{\mathbf q}(z)+\overline{F_{-\mathbf q}(-\bar z)} .
$$

$F(z)$ and $F(-\bar z)$ are separate output rows of one positive-node sweep; the
partner is a flat-$q$ negation gather and a conjugation. At a pure-imaginary
sample this keeps the anti-Hermitian, magnetization-odd channel that the even
completion would zero. The sweep also returns $\chi_q(-\bar z)$, stored as the
reflected sample that the ordered fit (§6) needs. Decks with time reversal keep
the symmetric path; metallic MPA uses explicit fractional-occupation ordered
pairs, independent of this completion. The run prints which route ran.

## 3. The double-parallel sample grid

With $\omega_m=\max\Delta$ over the screening bands, `gw.mpa.sampling` builds
$N_p$ nested fractions $0=s_0<\dots<s_{N_p-1}=1$ and two lines

$$
z_n^{(1)}=\omega_m s_n^\alpha+i\varpi_1,
\qquad
z_n^{(2)}=\omega_m s_n^\alpha+i\varpi_2,
\qquad 0<\varpi_1<\varpi_2,
$$

`mpa_varpi_near_ry`, `mpa_varpi_far_ry` (0.2 and 2 Ry, the published 0.1 and
1 Ha). The near line's first point is the exact static point $z=0$ for an
insulator and $i\varpi_0$ with $\varpi_0=2\times10^{-5}$ Ry
(`mpa_metal_origin_shift_ry`) for a metal: a stability displacement around
zero-energy intraband transitions, not a broadening. The material class is read
from the WFN occupations.

`mpa_sampling_schedule = nested` (default) is a powers-of-two ladder near the
origin, then bisection of the widest interval: raising $N_p$ adds one point per
line and moves none. `leon` is Yambo's qPPS continuation; the two agree through
$N_p=8$. $\alpha=1$ is the linear grid, $\alpha=2$ concentrates samples near
zero; `mpa_sampling_alpha` defaults to 1 for an insulator and 2 for a metal.
Neither $\alpha$ nor $N_p$ transfers between materials, meshes or broadenings:
converge $(\alpha,N_p)$ against held-out $W$ or, better, a matched
full-frequency $\Sigma$. A small backward error on the samples cannot detect an
under-resolved sampling manifold.

The grid follows the band window: more screening bands raise $\omega_m$ and
stretch both lines, while the sample count stays $2N_p$. Sample points that
collide in $x=z^2$ refuse (the fit is singular there). Changing the band window,
geometry, centroids or units invalidates stored samples and poles: readers
compare stamped identities, not shapes.

## 4. How each χ sample is evaluated

`gw.mpa.sample_plan` classifies each point by its analytic character, and the
character, not the line label, picks the rule:

| sample | target $K_z/(-2)$ | rule |
|---|---|---|
| $z=0$ | $1/\Delta$ | positive-interval Laplace minimax (`compute_chi0`) |
| $z=i\varpi$ | $\Delta/(\Delta^2+\varpi^2)$ | imaginary-axis Laplace minimax; damped line on the ordered route |
| $z=\omega$ | $\Delta/(\Delta^2-\omega^2)$ | two shifted $1/y$ minimaxes; only for $\omega>\max\Delta$, refused otherwise |
| $z=\omega+i\varpi$ | $\Delta/(\Delta^2-z^2)$ | damped real-time line rule |

The double-parallel grid has one static point, one pure-imaginary far point and
damped points elsewhere; it has no real-axis sample.

**Damped line rule** (`minimax.damped_line_rule`). For a line at height $\varpi$
the integrand beats at $F_\chi=\Delta_{\max}+\max|\operatorname{Re}z|$, which is
$2\omega_m$ on the standard grid. The rule truncates at
$t_{\max}=\log(2/\epsilon_\chi)/\varpi$ and fills $[0,t_{\max}]$ with
wavelength-sized panels of positive Gauss–Legendre weights, graded down as
$e^{-\varpi t}$ suppresses late panels. Each positive atom $(t_j,h_j)$ becomes
the two contour atoms $(+it_j,+1,+ih_j)$ and $(-it_j,-1,-ih_j)$, which together
give $-2h_je^{izt_j}\sin(\Delta t_j)$ without assuming an adjoint relation
between elements. All points of one line share $t_j,h_j$ and differ only in
scalar projections, so **one sweep of Green-function contractions produces every
sample on the line**. At fixed accuracy

$$
N_\chi=\mathcal O\!\left(\frac{F_\chi}{\varpi}\log\frac1{\epsilon_\chi}\right),
$$

linear in the transition bandwidth, with no gap precondition for $\varpi>0$.
The near line dominates; the far line is cheap. $\epsilon_\chi$ is
`minimax_target_error` and the order ceiling `minimax_max_nodes`.

## 5. Disk-bounded χ, W and fit

The frequency axis lives on disk (`file_io.mpa_store` through SlabIO), because a
few $W_q$ copies fit in memory and all $2N_p$ do not.

- **χ.** Each $z_j$ is reduced to the irreducible $q$ wedge and written
  collectively as one `P(None,'x','y')` slab of the
  $(N_z,N_{q,\rm irr},N_\mu,N_\mu)$ store, frequency leading. A per-frequency
  readiness bit is committed only after the collective close, so an allocated
  but incomplete slab is never read as zeros.
- **Dyson.** One $\chi$ slab in, $W(z_j)=[1-V\chi_0(z_j)]^{-1}V$, one $W_c$ slab
  out, both released. The local or distributed solve is the screening
  configuration's. Under `screening_diagrams = w_bse` the ladder resolvent
  (`gw.screening_bse.make_ladder_wc_source`) writes the same $W_c$ slabs in
  place of the Dyson solve; the fit, store and consumer are unchanged.
- **Fit.** All $2N_p$ frequencies are read for a bounded block of columns;
  rows stay sharded and no process holds an $N_\mu^2$ object
  (`gw.mpa.tiling`, `gw.mpa.fit_driver`). Blocks are grouped into 32-block
  checkpoint epochs; rank 0 publishes an epoch's ranges only after its writer
  closes, so a failed epoch certifies nothing and a restart skips every
  committed range.
- **Completion.** Body ranges, pole diagnostics, head identity and head
  readiness must all pass before the root COMPLETE stamp, the last mutation. A
  restart authenticates grid, $q$ table, centroid geometry, dtype, sampling
  record and WFN/charge-ζ identity, skips compatible ready slabs and refuses an
  incompatible partial store; replacing a finalized store requires
  `mpa_overwrite_completed_artifacts`. `mpa_fit_reuse_file` consumes a finalized
  fit read-only in a one-shot run.

The same bounded path runs on every QSGW iteration.

## 6. The multipole model and its fit

Each wedge element is fitted independently to

$$
W_c(z)\approx\sum_{p=1}^{N_p}\frac{2\Omega_pB_p}{z^2-\Omega_p^2}
=\sum_{p=1}^{N_p}\left[\frac{B_p}{z-\Omega_p}-\frac{B_p}{z+\Omega_p}\right],
\qquad \Omega_p=a_p-i\Gamma_p,\quad a_p>0,\ \Gamma_p\ge0,
$$

$N_p$ = `mpa_n_poles` $\in[1,16]$. $W_c$ is fitted directly, never $\chi$,
whose independently fitted poles the Coulomb factors would scramble.

**Pole identification** (`gw.mpa.pade_fit`, `mpa_pole_solver`). The default
`loewner` pencil interpolates the $2N_p$ samples in $x=z^2$ without a
Vandermonde system; the published cross-multiplied Padé solve is a Vandermonde
system in disguise and loses all digits by $N_p\approx8$–10. `companion` is
Yambo's LA construction (near/far split, $T=Y_2Y_1^{-1}$,
$(TM_1-M_2)b=Tv_1-v_2$, companion roots) and `thiele` its Padé–Thiele
recurrence, converted to a companion matrix and diagonalized by `geev`. All
three then share the same guards and residue refit, so the solver choice never
changes the physical ansatz.

**Guards, in order.** (1) reflection $b_p\to-\bar b_p$ when
$\operatorname{Re}\Omega_p^2<0$ (the published condition); (2) time order
$b_p\to\bar b_p$ when $\operatorname{Im}\Omega_p^2>0$, since a pole with
$\operatorname{Im}\Omega>0$ grows as $e^{|\operatorname{Im}\Omega|t}$ in $W(t)$;
(3) prune coincident poles; (4) prune poles outside the sampled range; (5) prune
null poles the denominator solve invented. **Any guard firing forces the
all-$2N_p$-sample complex least-squares refit of the residues** with the poles
fixed. Exact-zero residues mark pruned poles and are ignored by $\Sigma$
planning.

**Finalization** requires, per element, $\kappa\le1/r_{\rm cond}$
($r_{\rm cond}=10^{-13}$) and backward error $\le\sqrt{\epsilon_{\rm mach}}$.
These are stability guards, not an accuracy proof; the condition map is the
only full-tensor diagnostic retained (`gw.mpa.diagnostics`).

**Ordered fit** (measured-broken-TR insulator). The reflected sample gives
$W_c(-z_j)=W_c(-\bar z_j)^\dagger$ at the same element, hence
$W_{\rm even},W_{\rm odd}$. The fit above runs on $W_{\rm even}$ to fix
$\Omega_p,B_p$; one fixed-pole least-squares solve then fixes the odd residue
$D_p$ in $W_{\rm odd}(z)=\sum_p2zD_p/(z^2-\Omega_p^2)$. Conduction branches of
$\Sigma$ consume $B_p+D_p$, valence branches $B_p-D_p$
([derivation](../dev/notes/DERIVATION_gnppm_nonhermitian.md) §7–8).

Pole index is not an energy ordering and all pole fields are independent.
**More poles are not monotonically better**: a backward-stable fit at higher
$N_p$ can move QP energies more than a lower one, so $N_p$ is a model-selection
choice; raise it only with held-out $W$ and QP evidence.

## 7. Σ: branches and broadening {#mpa-sigma}

For one pole and one band energy each body-$\Sigma$ denominator is a causal
resolvent,

$$
r=x+i(\Gamma_p+\eta),
\qquad
\frac1r=-i\int_0^\infty e^{irt}\,dt,
$$

with $\eta$ = `sigma_regularization_ev`, a literal retarded broadening entered
once: $\Gamma_p$ stays in $W(t)$, and the planner multiplies every time weight
by $e^{-\eta t}$. The four branches are (ω ≥ 0, ω < 0) × (empty, occupied);
their sign topology, product windows, denominator boxes, rules
(`sigma_quadrature_eps`, `sigma_window_edge_factor`) and refusals are
[the Σ quadrature problem](sigma-quadrature-problem.md). The planner uses only
live poles ($|B_p|>0$); a nonfinite residue or a live pole with $a_p\le0$ or
$\Gamma_p<0$ refuses. Metallic weights and the Fermi-window split are
[metallic MPA screening](metallic-mpa-screening.md#5-sigma-with-finite-occupations).

**Execution.** At a time node $t_j$ MPA synthesizes

$$
W(t_j)=\sum_{p\in\mathcal W}B_p\,e^{-i(\Omega_p-E_{B,\rm ref})t_j}
$$

and calls the shared spatial kernel (`gw.ppm_tau_kernel`, common with GN/HL-PPM):
$[\mathcal F\{\mathcal F^{-1}G_k(t_j)\,\mathcal F^{-1}W_q(t_j)\}]$ projected on
bands. The spatial result $\Sigma^{(j)}_{mn\mathbf k}$ has no frequency axis;
each output frequency receives

$$
\Sigma_{mn\mathbf k}(\omega_l)\mathrel{+}=c_{jl}\,\Sigma^{(j)}_{mn\mathbf k},
\qquad
c_{jl}=p\,\alpha_j\,e^{-i(E_{\rm ref}-s\omega_l)t_j},
$$

so more output frequencies add coefficient arithmetic and output storage, never
spatial evaluations. `gw.ppm_accumulators.DeviceOmegaAccumulator` runs each
window as **one executable** with the node loop on device; its coefficient
matrix spans the complete frequency axis (zero outside the window) padded to the
plan's largest node count, so all windows share one compile. An anti-Hermitian
(one-sided) window accumulates a temporary $Z$ of the result's size and adds
$(Z-Z^\dagger)/2i$ once. The result costs $16\,n_\omega S_{\rm local}$ bytes per
rank ($S_{\rm local}$ = one rank's share of a band matrix, including $k$ and
bracket axes), distributed over all $P$ ranks; a one-sided window doubles it
while open.

**Pole batches.** Pole fields are stored on the $q$ wedge, read through SlabIO,
and unfolded on device `mpa_pole_batch_size` poles at a time (default 4,
allowed 1–8). A window touching $m_w$ resident batches runs its sweep once per
batch, so $N_{\rm eval}=\sum_wN_wm_w$. The batch is a memory choice, not a
spectral classification. Symmetry reduces storage and all non-FFT work on the
wedge; inputs are unfolded before the $k$-grid convolution.

### Pane planner (comparison control) {#pane-planner}

`LORRAX_SIGMA_PLAN=panes` replaces the box planner with a frozen pane planner
(`gw.mpa.sigma_windows.build_shared_sigma_windows`) for comparisons; it is not
the production path and is refused with the shared-pole $W$. With
$T=\max|\omega|+m_{\rm edge}\eta$ it splits each crossing branch's state × pole
product into a **crossing core** ($E_A\le T$, $a_p\le T$; positive causal
Gauss rule, $N_\times=\mathcal O(F/\gamma_{\min}\log1/\epsilon)$), an
**electronic stripe** ($E_A>T$, $a_p\le T$) and a **plasmon slab**
($a_p>T$), the latter two with rotated-contour sector rules whose rank grows as
$\log(|d|_{\max}/|d|_{\min})\log(1/\epsilon)$. Rectangular selectors keep the
band and pole sums separable; the core is deliberately overinclusive for that
reason. Its tolerances are frozen constants, and its one-sided windows are the
ones that use $Z$.

## 8. Head, output and QSGW boundary

The $q\to0$ head is a scalar $W_{c,\rm head}(z)=W_{\rm head}(z)-v_{\rm head}$
sampled on **the identical complex grid** as the body — `build_mpa_fit` refuses
otherwise ("QSGW head and MPA body must use the identical stamped z grid") —
because head and body residues are summed in one $\Sigma$. Each Dyson slab
finalizes one head sample while its total-$W$ body tile is resident, folding the
direct response and its wings through the Γ body
(`gw.head_correction.fold_cartesian_head_wings_sharded`;
[four-current heads](four-current-head-corrections.md) owns $S(\omega)$ and the
fold); only the $3\times3$ result survives. The scalar samples are fitted with
the same pole solver and guards as the body and published with it. The head
consumer (`head_correction.compute_complex_pole_head_sigma_diag`) uses the
stored complex head pole without adding $\eta$; the body applies $\eta$, so a
convergence statement covers the body quadrature at a fixed head.

After body and head, the common dynamic-$\Sigma$ finalizer
(`gw.sigma_dispatch`, `gw.dynamic_sigma`) adds the diagonal head, interpolates
the matrix-valued cube at the requested energies, writes `sigma_mnk.h5`, builds
the static Hermitian QSGW operator and applies the outside-band scissor.
Self-consistent QSGW rebuilds $\chi$, $W$, the body fit and the head on every
map (the stores are written under per-iteration names and one complete pair is
retained); no model tensor lives in `SCState`. With
`sc_head_update = parallel_transport` the fixed-DFT head is replaced each map by
the direct QP-basis head. The loop itself is [self-consistency](../self_consistency.md).

## 9. What changes the cost or the answer

- **More screening bands.** $\omega_m$ grows, both lines stretch, the near-line
  rank grows linearly with the bandwidth. Recheck $N_p$ against held-out $W$ and
  QP energies.
- **A wider Σ interval.** The crossing bandwidth grows and its rank grows
  linearly; sign-definite ranks grow logarithmically. A finer output step at
  fixed endpoints adds coefficient work and storage only.
- **A smaller η.** Changes the observable and raises the crossing cost nearly as
  $1/\eta$ when the fitted poles are narrow. Sweep it as a physical parameter;
  never compensate through $\Gamma_p$ or the line heights.
- **More poles.** Two more samples per pole, more fit work, possibly another
  resident batch, and no guarantee of a better model (§6).
- **Pole batch size.** Residency and repeated sweeps, not the approximation.
- **Metals.** The origin row leaves the sampled-rule family for an exact
  ordered-pair evaluation, and the line bandwidth comes from the occupation
  supports ([metallic MPA screening](metallic-mpa-screening.md)).

## 10. Ownership

| responsibility | owner |
|---|---|
| double-parallel points and fractions | `gw.mpa.sampling` |
| analytic sample classification and routes | `gw.mpa.sample_plan` |
| damped-line and rectangle rules | `services/minimax` (`gw.mpa.evaluator` re-exports) |
| χ sample evaluation, Dyson, head samples, fit orchestration | `gw.mpa.model` |
| multipole fit algebra and guards | `gw.mpa.pade_fit`, `gw.mpa.small_eig` |
| column-block walk and checkpoint epochs | `gw.mpa.tiling`, `gw.mpa.fit_driver` |
| sample and pole bytes | `file_io.mpa_store` through SlabIO |
| Σ box planner / pane planner | `gw.sigma_box_plan` / `gw.mpa.sigma_windows` |
| Σ executor | `gw.mpa.sigma` |
| shared $G\times W$ spatial kernel | `gw.ppm_tau_kernel` |
| accumulation into real-frequency Σ | `gw.ppm_accumulators` |
| dynamic-Σ output and QSGW finalization | `gw.sigma_dispatch`, `gw.dynamic_sigma` |

Scalar quadrature services know no bands, wavefunctions, $q$ mesh or file
format; physics planners know no HDF5; SlabIO owns large distributed bytes; the
spatial kernel receives built $G$ and $W$ tiles and holds no pole-fit policy.

## References

- D. A. Leon, C. Cardoso, T. Chiarotti, D. Varsano, E. Molinari and A. Ferretti,
  *Phys. Rev. B* **104**, 115157 (2021).
- D. A. Leon, A. Ferretti, D. Varsano, E. Molinari and C. Cardoso,
  *Phys. Rev. B* **107**, 155130 (2023).
