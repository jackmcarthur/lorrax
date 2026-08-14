# Multipole frequency integration in LORRAX

This page is the authoritative frequency-domain description of the MPA
pipeline.  It describes the mathematics, ownership, validity domains and I/O
boundaries in the order they are encountered.  `compute_mode = mpa` remains
disabled until one real-material chi/W/fixed-head/Sigma/QSGW calculation
passes end to end through the disk-bounded driver.  The scalar fit, pole
store, window planner and shared Sigma kernel are available for internal
tests without weakening that refusal.
The refusal is owned by `gw_config.UNIMPLEMENTED_MODES`; deleting its MPA row
is the final enablement gesture, not a preparatory step.

## 1. Energies and the transition spectrum

All frequency algorithms use one Fermi reference and convert stored pole
energies to Ry at the pole-store read boundary.  Occupation chooses the band
sum; it does not replace the signed energy.  Thus an empty state just below
the Fermi level or an occupied state just above it remains signed, which is
essential for future small-gap and inverted-gap support.

For an insulating reference, define

$$
E_g = E_{c,\min}-E_{v,\max},\qquad
A_{c\mathbf{k}}=E_{c\mathbf{k}}-E_{c,\min},\qquad
B_{v\mathbf{k}}=E_{v,\max}-E_{v\mathbf{k}},
$$

so a transition has

$$
\Delta_{cv\mathbf{k}}=E_g+A_{c\mathbf{k}}+B_{v\mathbf{k}}>0.
$$

The factorization is never implemented as a loop over `(v,c,mu)`.  LORRAX
forms one occupied Green function and one empty Green function at each shared
time node and contracts them in the ISDF basis.

## 2. Complex samples of chi0

The independent-particle response is sampled at a finite set of complex
frequencies `z_j`.  Schematically,

$$
G_c(\mathbf r,\mathbf r',t)
 \sim \sum_{c\mathbf k}\psi_{c\mathbf k}(\mathbf r)
 \psi^*_{c\mathbf k}(\mathbf r')e^{-iE_{c\mathbf k}t},
$$

$$
G_v(\mathbf r,\mathbf r',-t)
 \sim \sum_{v\mathbf k}\psi_{v\mathbf k}(\mathbf r)
 \psi^*_{v\mathbf k}(\mathbf r')e^{+iE_{v\mathbf k}t},
$$

and their product has spectral kernel

$$
K_z(\Delta)=-\frac{2\Delta}{\Delta^2-z^2}
            =-2\int_0^\infty e^{izt}\sin(\Delta t)\,dt.
$$

The sample plan is the double-parallel grid: a near-real line resolves sharp
structure, a farther line anchors the continuation, and their real parts use
a nested powers-of-two partition.  The lower transition bound is the actual
smallest included positive transition; the upper bound is the largest
transition formed by the requested occupied and empty band sums.  A metallic
or fractional-occupation plan needs a separate intraband treatment and is
currently refused rather than assigned an artificial gap.

There are two scalar quadrature families.

1. **Damped line.**  For `Im z = varpi > 0`, truncate at

   $$t_{\max}=\frac{\log(2/\epsilon)}{\varpi}$$

   and integrate positive real times with composite Gauss–Legendre or a
   certified sparse positive rule.  The bandwidth is
   `max |Re z| + Delta_max`; there is no `Delta_min` requirement.  This route
   remains well-defined as the band gap closes, but its node count grows with
   the time-bandwidth product.

2. **Interval exponential rules.**  Where the denominator has fixed sign,
   rotate the Laplace contour and approximate `1/x` on a bounded positive
   interval.  The cost grows approximately as
   `log(Delta_max/Delta_min) log(1/epsilon)`.  It is therefore economical for
   sign-definite cells and unsuitable when the interval crosses zero.

`services/minimax` owns these scalar rules and their sampled/certified error
statements.  The many-body kernels receive only nodes and weights.  No
wavefunction, q mesh or ISDF concept is present in the minimax service.

For the shared sine rule, each positive atom `(t_j,w_j)` is expanded exactly
into two contour nodes,

$$
(\tau,c,s)=(-it_j,-iw_j,-1),\qquad(+it_j,+iw_j,+1),
$$

whose executable sum is

$$
-\sum_{\pm}c_\pm e^{-\tau_\pm(\Delta-s_\pm z)}
=-2w_j e^{izt_j}\sin(\Delta t_j).
$$

`services/minimax.shared_sine_contour` owns this scalar identity.
`gw.w_isdf.compute_chi0_contour` carries the complex times through the same
Green-function/FFT contraction used by static chi0.  At each node LORRAX
builds the occupied and empty Green functions once, forms the response tile,
and projects every requested `z_j` through scalar weights.  The expensive
node count is therefore twice the shared sine rank, not the number of samples
times a per-sample rank.

## 3. Dyson solve, W samples and pole fit

For each sampled frequency,

$$
W(z)=\left[1-V\chi_0(z)\right]^{-1}V,
\qquad W_c(z)=W(z)-V.
$$

The matrix solve remains distributed.  One native sharded
`P(None,'x','y')` W slab is live at a time.  An offline/restart workflow may
write it through `file_io.mpa_store`, whose singleton frequency axis and
large arrays are owned by SlabIO; the small readiness bit is committed only
after collective close.  A failed write is therefore incomplete, not a
plausible zero.

The elementwise multipole model is

$$
W_c(z)\approx\sum_{p=1}^{N_p}
  \frac{2\Omega_p B_p}{z^2-\Omega_p^2},
\qquad \Omega_p=a_p-i\Gamma_p.
$$

`gw.mpa.pade_fit` owns the normalized Loewner/Padé solve,
`gw.mpa.fit_driver` owns the column walk, and `file_io.mpa_store` owns sample
and pole bytes, units, completion, and diagnostics.  Each irreducible-q
`chi(z_j)` slab is committed first.  Dyson then reads one complete chi slab,
writes one `Wc(z_j)` slab, and releases both.  The fit reads all `z_j` only
for a bounded `nu`-column block (about `N_mu/N_z` columns by default), writes
its poles, and releases the block.  No full frequency tensor exists in
memory.

The pole fit may place every pole at any positive `a_p` and nonnegative
`Gamma_p`; pole index is not a frequency band.  The executor groups poles by
their actual geometry for each window, then uses batches of four only as an
HBM schedule.  A batch boundary has no physics meaning.

## 4. Sigma denominators and causal branches

For a requested real-frequency grid `omega`, a pole contribution contains a
denominator of the schematic form

$$
d(\omega)=E_A+a_p-s\,|\omega|-i\Gamma_p,
$$

where `A` is the occupied or empty Green-function space and `s` is fixed by
the causal half.  The four branches are

| frequency half | Green-function bands | usual geometry |
|---|---|---|
| `omega >= 0` | empty | crossing |
| `omega >= 0` | occupied | sign-definite |
| `omega < 0` | empty | sign-definite |
| `omega < 0` | occupied | crossing |

This table is only the usual insulating topology.  The actual signed band
energies are authoritative.  The internal planner currently refuses a cell
whose nominally sign-definite rectangle reaches zero; public small/inverted-
gap support requires splitting that rectangle at the denominator boundary
and routing the straddling part through a crossing rule.

Let $W=\max(|\omega_{\min}|,|\omega_{\max}|)$ and let
$A_{\max}=24$ be the conditioning ceiling of the accepted HGL rule.  The
routing scale is

$$
\xi=\max\left(\eta,\frac{2W}{A_{\max}-2f_{\mathrm{edge}}}
\right).
$$

Here `eta` is the requested regularization and `f_edge` is the window-edge
factor.  The floor keeps the HGL dimensionless core bandwidth
$A_{\mathrm{core}}=2W/\xi+2f_{\mathrm{edge}}$ at or below $A_{\max}$.
Thus `xi` separates the absolutely convergent finite-width core from the
near-axis regularized core.  It is a numerical routing boundary, not a pole
fit or broadening applied to every pole.

## 5. The three Sigma quadrature methods

Every selected pole keeps its exact residue.  Except for the explicit HGL
core below, it also keeps exact `Gamma_p`.

### 5.1 Sign-definite sectors

If `Re d` has one sign over a cell, rotate the Laplace contour by an angle
that keeps `Re(c d)>0` and fit

$$
\frac1d= c\int_0^\infty e^{-c d s}\,ds
       \approx\sum_\ell w_\ell e^{-d\tau_\ell},
\qquad \tau_\ell=c s_\ell.
$$

One positive rule is fitted over the union of all actual `(Re d,Gamma)`
rectangles in a physical class.  The fit uses endpoint/log-radial training
and an independent midpoint grid; the reported bound is sampled unless the
selected minimax asset carries a continuum certificate.  The rule is valid
only over its recorded rectangles and output-frequency half.

The historical names `single`, `a_stripe` and `b_slab` describe how a
crossing rectangle is partitioned into sign-definite leftovers.  They are
not different kernels.  Compatible pole residues are summed into one W(t)
tile before the spatial convolution.

### 5.2 Finite-width crossing core

For `Gamma_p >= xi`, use the causal real-time identity

$$
\frac1d=i\int_0^\infty e^{-idt}\,dt.
$$

The common positive global-Gauss rule covers
`|Re d| <= F` and `Gamma_min <= Gamma <= Gamma_max`.  Its tail uses the
weakest damping.  Several deterministic tail-error allocations are tried,
and the smallest rule passing a disjoint dense boundary check is retained.
The natural scale remains

$$t_{\max}=\log(2/\epsilon)/\Gamma_{\min},$$

up to the selected tail allocation.  This is what permits one rule to cover
poles at several widths; a rule certified on one horizontal line alone does
not establish that claim.  The cost is set mainly by `F/Gamma_min`; adding
more strongly damped poles is usually cheap.

Both the crossing and sign-definite inputs bound the same dimensionless
relative residual, `|1-d Q(d)|`.  They nevertheless have separate error
budgets: the positive crossing error was observably much less sensitive in
the validated eight-pole calculation.  The defaults `2e-3` (crossing) and
`6.5e-4` (sector) reduced the physical census from 478 to 446 while changing
registered QP energies by at most `5e-5` meV relative to the 478-node plan.

### 5.3 Near-axis HGL core

For `Gamma_p < xi` in the crossing core, absolute real-time convergence
would require an impractically long interval.  LORRAX instead uses the
accepted regularized real-pole HGL sine functional on a bounded dimensionless
bandwidth.  Its pole phase is `Re Omega_p`.  Each sine atom is evaluated with
both signed times and the same stored residue,

$$
B_p\sin(ut)=B_p\frac{e^{iut}-e^{-iut}}{2i}.
$$

The negative-time arm is explicit: it is not inferred from a band-space
adjoint, because independently fitted matrix elements need not make an
individual pole residue Hermitian.  This preserves arbitrary complex
off-diagonal residues without pole matching.

This is the only place where fitted `Gamma_p` is intentionally discarded.
Its accepted rank/error pair is part of the implementation contract, rather
than a routine input dial: reducing rank 48 to 21 moved the tested QP levels
by about 1.56 meV, while the accepted rule was below the 0.2 meV gate.

## 6. Shared spatial execution

For each scalar node the ansatz-specific code constructs

$$
G_A(t),\qquad
W(t)=\sum_{p\in\mathcal W}B_p e^{-i(\Omega_p-E_{B,\mathrm{ref}})t},
$$

then calls the one public spatial seam

$$
G_k(t)\times W_q(t)
\longrightarrow
\operatorname{project}\!\left[
\mathcal F\{\mathcal F^{-1}G_k\,\mathcal F^{-1}W_q\}
\right]_{mn\mathbf k}.
$$

`gw.ppm_tau_kernel.get_sigma_spatial_kernel` owns the fused FFT convolution
and band projection.  GN-PPM and MPA own only their G/W synthesis.  A
different one-particle propagator, such as `dG/dQ_ph`, can therefore provide
another G tile; a two-point vertex-corrected W can provide another W tile.
A genuine three-point vertex has different tensor rank and belongs to a
separate contraction, not a flag in this kernel.

Only one G(t), W(t) and Sigma(t) tile exists per node.  The result is folded
directly into the sharded real-frequency cube; no tau history is retained.
The number of omega grid points changes fold/storage work linearly but does
not by itself multiply the expensive tau-node count.  Widening the requested
frequency interval changes the denominator rectangles and can increase
noncrossing ranks logarithmically and finite-width crossing ranks roughly as
`F/Gamma_min`, with discrete jumps when a pole moves between windows.

## 7. Head, output and QSGW boundary

The core local-field algebra is

$$
S_{\mathrm{eff}}(z)=S_0(z)
+\frac{Y(z)W_{\mathrm{body},\Gamma}(z)Z(z)}{V_{\mathrm{cell}}}.
$$

`gw.head_correction.fold_cartesian_head_wings_sharded` owns this contraction
without gathering the body tile.  The production component that rebuilds
`S_0`, `Y`, and `Z` from the current orbitals is not connected yet.  The
staged driver therefore labels and reuses the established two-point DFT
scalar head while rebuilding the MPA body.  This fixed-head approximation
omits local-field head/wing dynamics; it is not an arbitrary-frequency MPA
head.  The row-sharded fit and temporal pole consumer have passed their
four-rank integration gate; public `compute_mode = mpa` remains disabled
until one real-material run also traverses chi, W, this fixed head, the full
Sigma contraction, and the common QSGW finalizer.

After that addition, MPA must reuse the existing dynamic-Sigma finalizer:

1. add body and q→0 head;
2. interpolate the matrix-valued cube at the DFT or current QP energies;
3. write `sigma_mnk.h5` through the existing sharded output path;
4. construct the static Hermitian QSGW operator;
5. apply the existing outside-band scissor extension.

One-shot and diagonal fixed-point QP solvers can consume a finalized external
pole store.  Fully self-consistent QSGW rebuilds the body model because each
iteration changes the orbitals and transition energies.  Its bounded path is
`chi(z)` q-wedge store -> one-slab Dyson -> `Wc(z)` q-wedge store -> bounded
column fit -> q-wedge pole store.  Sigma subsequently reads and unfolds four
pole/residue slabs at a time.  The fixed-head approximation may reuse one
head fit while the body is rebuilt each iteration.

## 8. Current validation and public controls

The research reference used eight fitted poles and four-pole HBM batches.
The accepted 572-node schedule had 16 shared sweeps.  After staging and HGL
carrier cleanup it took 91.731 s wall / 72.821 s Sigma on four A100 GPUs and
agreed with its pre-optimization cube to roundoff; its maximum registered QP
difference from the 1229-node reference was 0.025891 meV.  These measurements
validate that particular frozen schedule, not every runtime-generated plan.

Common real-frequency controls already live in the Sigma section of
`docs/input_reference.md`: grid minimum, maximum and step; regularization;
window-edge factor; omega layout and accumulation.  The staged MPA keys are
pole count, insulating sample-plan choice, near/far sampling line heights,
the explicitly fixed-DFT head model, and pole batch size (hard-capped at
four).  The mode still refuses at entry while its P=4 gate is incomplete, so
none of these keys can silently select unverified work.  HGL
certification constants, panel construction details, provenance hashes and
campaign controls remain implementation data rather than deck knobs.

Module ownership is summarized in `src/gw/mpa/__init__.py`; exact input
defaults belong only in `docs/input_reference.md`, not in this theory page.
