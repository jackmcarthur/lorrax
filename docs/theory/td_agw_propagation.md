# TD-aGW propagation

`python -m tdagw.tdgw_jax -i tdgw.toml` propagates the real-time adiabatic-GW
(TD-aGW) equation of motion of Xiaoxun Gong's TDGW code in TDGW's
conventions, with TDGW's deck, header and output files. This page owns the
equation of motion as integrated, the integrators, the array layout and the
output contract. The interaction ΔΣ[ρ] of the ISDF model is derived on
[TD-aGW interaction](td_agw.md); the deck keys are in the
[input reference](../input_reference.md#td-agw). The steps from a GW run to a
drive are on [TD-aGW end to end](../how-to/td-agw-workflow.md).

## The equation of motion {#eom}

The state is the one-particle density matrix in the quasiparticle band
basis of a window of $n_b$ bands, $\rho_{k,nm}(t) = -i G^<_{k,nm}(t,t)$, on
the flat k grid of the TDGW header. TDGW propagates $G^< = i\rho$; the
factor is exact, so the RK45 error norm and the steps it selects are the same
for either. In Hartree atomic units, length gauge, electron charge $-e$:

$$
\dot\rho_k = -i\Big([H_k(t), \rho_k] + \mathbf E(t)\cdot[\mathbf r^\mathrm{intra}, \rho]_k\Big)
             - \Gamma \circ (\rho_k - \rho^\mathrm{eq}_k) \tag{1}
$$

$$
H_k(t) = \mathrm{diag}(E^\mathrm{QP}_k) + \mathbf E(t)\cdot\mathbf r^\mathrm{inter}_k
         + \Delta\Sigma_k[\rho(t)] \tag{2}
$$

(TDGW `td/driver/agw.py`, `aGWDriver.calc_rhs`). $\rho^\mathrm{eq}_k =
\mathrm{diag}(f_{kn})$; $\Gamma_{nn}$ is the depopulation rate and
$\Gamma_{n\neq m}$ the dephasing rate. The intraband term is the covariant
k-derivative commutator (Phys. Rev. B 92, 235307 (2015), Eq. 6) in TDGW's
unitary-overlap form,

$$
[\mathbf r^\mathrm{intra}, M]_k = i\sum_b w_b\,\mathbf b\,
  \big(\tilde S_{k,b} M_{k+b} \tilde S_{k,b}^\dagger - M_k\big)
  + (r^\mathrm{intra}_{k,n} - r^\mathrm{intra}_{k,m})\,M_{k,nm}, \tag{3}
$$

with $\tilde S_{k,b} = UV^\dagger$ from the SVD of the energy-masked overlap
$\langle u_{kn}|u_{k+b,m}\rangle\,e^{-(\varepsilon_{kn}-\varepsilon_{k+b,m})^2/2\sigma^2}$
($\sigma$ = `system.esmear`) and the second term only along a vacuum
direction. `lorrax.intraband = false` drops Eq. (3).

**Held collision term.** With `lorrax.collision` enabled, Eq. (1) also
carries the held collision remainder; its column-sampled one-sided MGKBA
coefficients, coherent principal-value contribution and equilibrium
counterterm are defined by [TD-aGW dynamical §1.5](td_agw_dynamical.md#damping).
The coherent contribution joins ΔΣ once. The remainder enters the right-hand
side for explicit methods and the split remainder flow for Magnus methods;
its coefficients refresh at chunk boundaries under
[the refresh contract](td_agw_dynamical.md#refresh). The static-W subtraction
in ΔΣ and the collision equilibrium counterterm are distinct operations.

**ΔΣ is injected.** The integrators see the self-energy update only as a
callable of the instantaneous state, `apply(params, rho, t) -> dSigma` (Ha,
Hermitian; the models below are linear in $\rho - \rho^\mathrm{eq}$ before
attaching a held term), `tdagw.self_energy.SelfEnergyModel`:

| `lorrax.model` | ΔΣ[ρ] | parameters |
|---|---|---|
| `none` | 0 (IP) | — |
| `tdgw_kernel` | TDGW's dense kernel, TDA ($\Delta\Sigma_{cv} = \sum K\,\rho_{c'v'}$ on the cv blocks) or full (on $\rho - \rho^\mathrm{eq}$) | `kernel.h5` (BerkeleyGW layout, TDGW `read_kernel`) |
| `isdf` | the ISDF COHSEX model of a GW restart bundle, [TD-aGW interaction](td_agw.md) | the bundle's ψ, ζ, W, V on the mesh |

The `isdf` adapter passes $\Delta\rho = \rho - \rho^\mathrm{eq}$ in the
model's layout (bands padded and sharded over the mesh, rows in the model's
k order) and converts its Ry output to Ha. Low-rank kernels
$K = U\Lambda U^\dagger$ (`lowrank_kernel_model`) serve cost studies at
large windows.

**Measurements** (TDGW `td/measurement/workers.py`), with
$\langle O\rangle = \frac{g_s}{\Omega}\sum_k w_k \mathrm{Tr}(\rho_k O_k)$ and
$g_s = 2$ for `nspinor = 1`, 1 for spinors:

- `polarization`: $\mathbf P = -\langle \mathbf r^\mathrm{inter}\rangle$, complex as in TDGW;
- `current`: $\mathbf j = -\langle \mathbf v\rangle$, $\mathbf v = i[H, \mathbf r]$ = the
  equilibrium velocity (interband $i(E_n - E_m)r_{nm}$, intraband $\partial_k E_{kn}$)
  plus $i[\Delta\Sigma, \mathbf r^\mathrm{inter}] - i[\mathbf r^\mathrm{intra}, \Delta\Sigma]$
  unless `use_v0`; `inter_only` keeps the interband parts;
- `density_matrix` / `lessG`: a band block of ρ / of $i\rho$; `Efield`: $\mathbf E(t)$.
- `carrier_density`: $(n_c, n_v)$ per unit cell, $n_c = g_s\sum_k w_k \sum_c \rho_{k,cc}$ over the
  unoccupied rows and $n_v$ the holes in the occupied ones (LORRAX);
- `exciton_projection`: $c_S(t) = \sum_{kcv} A^{S*}_{kcv}\rho_{k,cv}(t)$ on the BSE eigenvectors of
  `eigenvectors` (`bse_jax --write-eigs`, same window and k grid); $|c_S|^2/N_k$ is the
  population of exciton S per unit cell (LORRAX).

## The system from a GW bundle {#bundle}

`[lorrax] system = "bundle"` builds the header in memory from a LORRAX GW
restart bundle (`bundle_input` = its GW deck); no TDGW WFN, `vmtxel` or
finite-difference step runs. One `load_td_model` call supplies the model's
wavefunctions, the window record and, for `kgrid_fine` decks, the fine-grid
velocity, so the header and ΔΣ share one ψ gauge.

| header | from the bundle |
|---|---|
| bands | `[system] ibndmin..ibndmax` (1-based, inclusive), DFT order |
| `e_kn` | QP energies by DFT index from `eqp_file` (the window re-slice of `bse_window.apply_eqp_and_reslice_bands`), Ry → eV → Ha with TDGW's `HA_IN_EV` |
| `f_kn` | 1 for the DFT indices below `ivbm`, as TDGW |
| `kpts` | the bundle's C-order grid in [0, 1) |
| `rinter` | $r_{nm} = v_{nm}/(i(E_n - E_m))$, $v$ the bundle's DFT velocity $p + i[V_\mathrm{NL}, r]$ in the model's ψ gauge, $E$ the DFT energies of that velocity |
| `itype` | 1: no intraband commutator, as TDGW's `use_vmtxel` route |

**Velocity.** Native decks slice `dipole_file` (the bundle's `dipole.h5`) to
the window and refuse when its band energies differ from the restart's DFT
energies by more than 1e-8 Ry (`td_dipole_energies`). `kgrid_fine` decks take
Route B's window velocity in the fine $U_q$ gauge
(`densified_dipoles(full_window=True)`) with its Newton-inverted DFT
energies. With `--eqp` the fine states are QP-mixed and $v_{nm}/\Delta E$ is
not their position (the onset oscillator strength comes out 13–19 % low on
8 → 30 k); Route B then returns `r_window`, the exact
$\langle\tilde i|\hat r|\tilde j\rangle$ of the fine states (same-occupation
pairs masked at 0.05 eV upstream), and the builder takes it as given: no
division, `cut` and `cv` applied on top, `mask` only at Route B's 0.05 eV
(`td_pair_gap_route_b`).

**The denominator uses DFT energies.** $v = i[H, r]$ holds in the eigenbasis
of the Hamiltonian whose velocity it is, the DFT one; QP differences would
rescale every $r_{cv}$ by $\Delta E^\mathrm{DFT}/\Delta E^\mathrm{QP}$, a
velocity-renormalisation choice this route does not make.

**n = m and near-degenerate pairs.** $r_{nn} = 0$ and the intraband
commutator is absent, as on TDGW's `vmtxel` route; TDGW's finite-difference
route (`itype` 2) adds $\mathbf E\cdot[\mathbf r^\mathrm{intra}, \rho]$
(`TDGW_RINTRA_IN_EOM`, on by default). It vanishes at first order in the field
($[\mathbf r^\mathrm{intra}, \rho^\mathrm{eq}] = 0$ for filled bands): on TDGW's
516 finite-difference header (IP, a 10 as kick) dropping it changes P(t) by
8.8e-7 at 1e7 V/m and 7.9e-4 at 5e9 V/m, the carrier density by 3.2e-7
and 3.8e-4 (a 500x field step gives 898x and 1188x: local exponents
1.09 and 1.14 from these two points, not linear in $E_0$), and no Im ε
peak by more than 0.1 meV.
Same-occupation pairs ($vv'$, $cc'$)
make $v_{nm}/\Delta E$ singular as $\Delta E \to 0$: `pair_rule = "mask"`
(default) multiplies $r_{nm}$ by $1 - e^{-\Delta E^2/2\Delta^2}$ with
$\Delta$ = `pair_gap_ev` (default `esmear`), `cut` zeroes the pairs with
$|\Delta E| < \Delta$, `cv` keeps the c↔v blocks only (TDGW's `vmtxel` layout).
c↔v pairs are never masked; a c↔v gap at or below $\Delta$ refuses
(`td_cv_gap`). The run header prints the pairs below $\Delta$: on CrI3 516
(6×6 k, 2v2c) 2 of 36 $vv'$ and 23 of 36 $cc'$ (bands 131/132 nearly
degenerate), smallest c–v DFT gap 1.514 eV.

**Gauge receipt.** `check_eigenvectors` = a `bse_jax --write-eigs --dipole`
file on the same bundle and window: the builder forms
$\langle 0|\mathbf r|S\rangle = \sum_{kcv} A^S_{kcv}\, r_{k,vc}$ from the
header and compares it with `bse_jax`'s dipoles, so the TD drive and the BSE
oscillator strengths are the same numbers. On 516 the 16 lowest states agree
to 9.6e-15 (I+ 6.4e-15).

**Against TDGW's own header (516).** k points, bands, occupations and
`e_kn` equal those of TDGW's `vmtxel` header (energies to 0.0 Ha); the IP
Im ε_xx peaks are 2.8405 / 2.9833 / 3.0876 / 3.5446 / 3.6576 eV against
2.8406 / 2.9824 / 3.0868 / 3.5454 / 3.6578 eV. The intensities differ:
BerkeleyGW's `vmtxel` uses the momentum operator, which lacks $i[V_\mathrm{NL}, r]$,
and its $\sum_{kcv}|r_{cv}|^2$ is 10.1 % below the bundle's (per-k ratios
0.43–2.42 along x; the best global phase between the two is ≈ 1, the same WFN
gauge).

**Circular drives.** σ+ is $\mathbf E = E_0 f(t)(\cos\omega t, +\sin\omega t, 0)$
(`E_x = "E0 * f * np.cos(w * t)"`, `E_y = "E0 * f * np.sin(w * t)"`,
`w = ħω / HBAR_IN_EV_AS` in rad/as) and couples to
$I_+ = |D_x - iD_y|^2/2$, $D = \langle 0|\mathbf r|S\rangle$. On a chiral
three-band model σ± select their exciton to a population ratio below 1e-12
and the swapped twin exchanges them; the first-order population per cell is
$E_0^2 I_+ \pi\tau^2/2$ for $f = e^{-((t-t_0)/\tau)^2}$ (checked to 2e-4).
Start the pulse at $t_0 \ge 4\tau$: a Gaussian cut at $t = 0$ while still at
$e^{-9}$ of its peak leaves the counter-rotating channel at
$E_0 f(0) d/(\sqrt2(\Omega + \omega))$, which is physical for that drive.

**TDA.** `[lorrax] tda = true` applies TDGW's TDA to the ISDF ΔΣ: only
$\delta\rho_{cv}$ drives, only $\Delta\Sigma_{cv}$ and its conjugate are kept
(the projection of `tdgw_bridge --tda`).

## Integrators {#integrators}

Cost is counted in ΔΣ evaluations, which dominate at any window where the
interaction is non-trivial; an exponential costs one batched
$n_b\times n_b$ `eigh` per k and two products.

| `lorrax.method` | scheme | order | ΔΣ per step | exponentials per step | step |
|---|---|---|---|---|---|
| `dp54` | Dormand–Prince 5(4), SciPy's `RK45` | 5 (error estimate 4) | 6 (FSAL) | 0 | adaptive |
| `rk4` | classical Runge–Kutta | 4 | 4 | 0 | `lorrax.dt` |
| `etdrk4` | Krogstad's exponential RK on the diagonal $L_{nm} = -i(E_n - E_m) - \Gamma_{nm}$ | 4 | 4 | 0 (diagonal) | `lorrax.dt` |
| `magnus2` | exponential midpoint, on the whole ρ (`magnus_frame = whole_rho`, the default) or on Δρ in the interaction picture; Strang-split relaxation and intraband term | 2 | `n_corr` | `n_corr` + 1 | `lorrax.dt` dividing `dt_measure`, or `auto` |
| `cfm4` | commutator-free Magnus CFET4:2 at the Gauss nodes on Δρ in the interaction picture; relaxation and intraband term by Yoshida's triple jump | 4 | 2 `n_corr` | 4 `n_corr` + 2, three times that with Γ or the intraband term | `lorrax.dt` or `auto` |

**`dp54` reproduces TDGW's run.** The tableau, the error norm (RMS over the
logical entries, $\mathrm{atol} + \mathrm{rtol}\,\max(|y|, |y_\mathrm{new}|)$),
the controller (safety 0.9, factor in [0.2, 10], exponent −1/5, no growth
after a rejection) and the sampling are SciPy's: every sample is taken from
the dense output of the step that covers it, and steps are never truncated
at a sample, as `solve_ivp(t_eval=...)` does. The run therefore takes
TDGW's steps; the propagated quantities agree to rounding.

**`etdrk4`** integrates the band-energy phases and the relaxation exactly
and treats $-i[\mathbf E\cdot\mathbf r + \Delta\Sigma, \rho]$ (and the
intraband term) explicitly; its step is limited by the coupling, not by the
largest transition energy, and it is exact when the explicit part
vanishes. The $\varphi$ functions use a Taylor series for $|hL| < 0.5$.

**`cfm4`, and `magnus2` with `magnus_frame = interaction`, integrate the deviation in the interaction picture.**
With $x = \rho - \rho^\mathrm{eq}$, the band-energy phases are exact,
$x_{nm} \to e^{-i(E_n - E_m)\tau}x_{nm}$, and the coupling
$B(t) = \mathbf E(t)\cdot\mathbf r^\mathrm{inter} + \Delta\Sigma[\rho(t)]$, taken
in that frame, acts by conjugation with $U = e^{-ih\bar B}$ (`magnus2`: $B$ at
the midpoint; `cfm4`: two exponentials of $B$ at the Gauss nodes, CFET4:2).
The update is written on the deviation,

$$
x \to U x U^\dagger + [U - 1, \rho^\mathrm{eq}]\,U^\dagger, \qquad
U - 1 = V\big(e^{-i\tau w} - 1\big)V^\dagger, \tag{4}
$$

with $V, w$ from the `eigh` of the generator, so every term has the size of
$x$ or of $h\bar B$: rounding is relative to the response, not to the O(1)
occupations, and the weak-field floor of conjugating the whole ρ is gone
([accuracy limits](#accuracy-limits)). The spectrum of $\rho^\mathrm{eq} + x$
is conserved exactly when Γ = 0, the collision is off and the intraband term
is off. ΔΣ is frozen
over a step as the polynomial through its node values (linear through the two
Gauss nodes for `cfm4`), so everything else on the step — the corrections'
node flows, the composition below, dense output — costs exponentials, not ΔΣ
evaluations. A predictor extrapolates the node values from earlier steps,
rotated into the current frame (`lorrax.predictor`: `cubic`, through the last
two steps' four Gauss-node values for `cfm4` and quadratic through the last
three midpoints for `magnus2`; `linear`, `cfm4` only, through the last step's
two); each correction propagates $x$ to the nodes with the frozen generator
and re-evaluates ΔΣ there. Without a correction the history holds only
predicted values and the scheme does not converge, so `n_corr` < 1 refuses.
The first two (`cfm4`) or three (`magnus2`) steps, before the history
exists, take three extra corrections. The default is two corrections.

**`magnus2` on the whole ρ (the default frame).** $\rho \to e^{-ihH_m}\rho\,e^{ihH_m}$
with $H_m = \mathrm{diag}(E^\mathrm{QP}) + \mathbf E(t + h/2)\cdot\mathbf r^\mathrm{inter} +
\Delta\Sigma[(\rho_n + \rho_{n+1})/2]$, solved by `n_corr` fixed-point corrections from a
quadratic extrapolation of the last three midpoint values
(`time_integrators.magnus_midpoint_step`). ΔΣ at the average of the end states is the
implicit-midpoint structure: after the pulse $\mathrm{Tr}(H_m(\rho_{n+1} - \rho_n)) = 0$,
so the energy $E = \sum_k w_k[\mathrm{Tr}(E^\mathrm{QP}\rho_k) + \tfrac12\mathrm{Tr}(\Delta\Sigma_k\,\Delta\rho_k)]$
of a Hermitian kernel is conserved up to the corrections' residual and rounding, as are the
spectrum and $\mathrm{Tr}\,\rho^2$. The interaction-picture step splits the band energies
from the coupling and conserves E only to second order. On the TDGW fixture (Γ = 0, its pulse
×10, 400–4000 as) the largest post-pulse |ΔE| relative to the excitation energy is 2.8e-11 /
5.5e-11 at 5 / 2.5 as on the whole ρ (rounding, ∝ √N and 1/E₀: 3.1e-9 with the pulse ×1),
and 1.6e-9 / 4.0e-10 in the interaction picture, a drift falling 4.0× when h halves and
independent of the field (`tests/tdagw/test_prop_magnus_frame_cpu.py`); in TDVAL's harness
1.4e-12 against 1.76e-8. The whole ρ is the default because magnus2's truncation error, not
its rounding, sets its accuracy at production fields: on CrI3 at 100 V/m it is 1.4e-3 at 5 as
on the whole ρ and 1.3e-4 in the interaction picture (exact band phases), against a whole-ρ
floor near 1e-5 at 100 V/m that falls as 1/E₀; the structure-preserving form is kept, and the
interaction picture is the choice for a weak-field magnus2 run that needs the accuracy.

**Relaxation and the intraband term are composed.** They are not
commutators. With either active, $S_2(\tau) = R(\tau/2)\,M(\tau)\,R(\tau/2)$
— $M$ the Magnus flow, $R$ the exact relaxation around an RK4 step of the
intraband flow — is the step of `magnus2` (Strang, second order in $[M, R]$),
and `cfm4` takes Yoshida's triple jump
$S_2(\gamma_1 h)\,S_2(\gamma_0 h)\,S_2(\gamma_1 h)$, $\gamma_1 = 1/(2 - 2^{1/3})$,
$\gamma_0 = 1 - 2\gamma_1$ (Phys. Lett. A 150, 262 (1990)), adjacent $R$ pieces
merged. ΔΣ stays frozen across the three sub-steps, so the composition
triples the exponentials and leaves ΔΣ at 2 `n_corr` per step. On CrI3 (6×6 k,
2v2c, ISDF COHSEX, intraband term on, two corrections) the relative L2 error
of P(t) over 30 fs at 50 / 25 / 10 / 5 as (80 / 160 / 400 / 800 ΔΣ per fs), with
the whole-ρ `cfm4` it replaces:

| drive (reference) | `cfm4` | whole-ρ `cfm4`, Strang | `etdrk4` |
|---|---|---|---|
| 100 V/m (exact linear response) | 3.1e-5 / 1.9e-6 / 4.7e-8 / 3.0e-9: order 4.0 | 1.4e-5 / 3.4e-6 / 5.0e-6 / 6.6e-6: rounding floor | 1.0e-4 / 6.5e-6 / 1.7e-7 / — |
| 5e8 V/m (DP54, rtol 1e-11, 1 as cap) | 3.1e-5 / 1.9e-6 / 4.7e-8 / 3.0e-9: 4.0 | 1.7e-5 / 1.3e-6 / 1.6e-7 / 4.0e-8: 2.0 | — / 6.5e-6 / 1.7e-7 / — |
| 5e9 V/m (same) | 3.4e-5 / 2.1e-6 / 5.2e-8 / 3.3e-9: 4.0 | 3.3e-4 / 8.2e-5 / 1.3e-5 / 3.2e-6: 2.0 | — / 7.4e-6 / 1.9e-7 / — |
| 5e8 V/m, Γ = 0.05 eV (same) | 4.6e-5 / 2.8e-6 / 7.2e-8 / 4.5e-9: 4.0 | 1.7e-4 / 2.2e-5 / 1.5e-6 / 1.9e-7: 2.9 | — / 2.6e-6 / 6.8e-8 / 4.2e-9 |

The DP54 references agree with a run at a 2 as cap to 2.5e-13. At 5e8 V/m
this window still responds linearly (the same relative errors as at
100 V/m). With Γ ≠ 0 `cfm4`, `etdrk4` (Γ exact in its linear part) and `rk4`
are fourth order; at equal ΔΣ cost `cfm4` and `etdrk4` then agree within 10 %,
while without Γ `cfm4` is 3.5× more accurate.

**Stability of the corrected step.** For a ΔΣ linear in the deviation, each
eigenmode of the coupling map $\mathcal L x = -i\,(\Delta\Sigma[x] -
\Delta\Sigma[0]) \circ (f_m - f_n)$ (eigenvalues $\pm i\omega$ for a Hermitian
kernel, TDA or a stable full BSE) evolves alone, and a step is a linear
recurrence on $u_n$ and the history's node values $G$: the predictor
$G \leftarrow W G_\mathrm{hist}$, each correction $G \leftarrow z u_n \mathbf 1 + z A G$,
$u_{n+1} = u_n + b\cdot G$, with $z = i\omega h$, $A$ and $b$ the integrals of the
Lagrange basis on the nodes (for the Gauss pair the Gauss-collocation matrix,
$|\mathrm{eig}\,A| = 1/(2\sqrt3)$) and $W$ the extrapolation weights
(`solvers.time_integrators.magnus_amplification`). The step is stable while the
recurrence's spectral radius stays at 1 for $h\sigma \le y_\mathrm{lim}$, σ the
spectral radius of $\mathcal L$ (`magnus_stability_limit`: growth below 1e-6 per
step on the imaginary axis):

| $y_\mathrm{lim}$ | 1 correction | 2 | 3 | 4 |
|---|---|---|---|---|
| `cfm4`, cubic predictor | 0.117 | 0.730 | 1.144 | 0.515 |
| `cfm4`, linear predictor | 0.625 | 1.203 | 0.231 | 0.458 |
| `magnus2` | 0.346 | 0.126 | 0.212 | 0.819 |

The limit is not monotone in `n_corr`: a truncated fixed-point iteration is an
explicit polynomial in $z$, as RK1 and RK2 are on the imaginary axis, and the
extrapolating predictor (its weights' magnitudes sum to 50 for the cubic)
feeds the error into every step. Corrections iterated to convergence would
reach the Gauss collocation's own bound, $h\sigma \le 2\sqrt3 = 3.46$, but at
the steps of interest they converge only after 10–35 correction pairs (not
implemented). An unstable mode grows from rounding whatever the drive and
saturates at O(1), each step being unitary. The recurrence reproduces the
growth per step of the real step on a two-level TDA toy to 1e-3 (stable,
weakly and strongly unstable settings; `tests/tdagw/test_prop_stability_cpu.py`).

**The check at the start of a `magnus2` / `cfm4` run** applies two tests on the model
itself. (1) σ by 20 power iterations of $\mathcal L^2$ (40 ΔΣ evaluations in the run's
layout, `tdagw.propagate.coupling_radius`): a step with $1.1\,h\sigma > y_\mathrm{lim}$
refuses (`td_magnus_stability`). A model without interaction (IP: the map vanishes,
$\|\mathcal L^2 x\| \le (10^{-12}\,\mathrm{Ha})^2 \|x\|$, `COUPLING_ZERO_FLOOR`) has σ = 0: any
step passes, with no probe ("no coupling" in the log); a non-finite σ (a map that returns
NaN or inf) refuses by name. (2) The step probe (`tdagw.propagate.step_growth`): 48
steps of the run's own Magnus step (its predictor, corrections and composition) with the
field off, from a random coherence and an independently seeded node history, at 1.25 dt;
a growth above 2e-3 per step refuses, so an accepted step lies a measured 25 % below the
instability. `lorrax.dt = "auto"` starts at the σ limit and shrinks it ×0.85 until the probe
passes. The recurrence treats the coupling as static; in the interaction picture it is not
(several transition energies, and a kernel's resonant–antiresonant block rotates at about the
sum of two of them), so the static limit is exact for one transition only. On CrI3 2v2c
(6×6 k, ISDF COHSEX) σ = 1.644 eV after 20 iterations against 1.650 eV from the exact
spectrum gives 42.6 / 265.7 / 416.4 as for 1 / 2 / 3 corrections; the real two-correction
step is stable at 265.7 as and grows at 332 as (probe 1.074 per step, the recurrence 1.158),
so the static limit (292 as without the margin) holds there, and one correction diverges from
50 as as measured. On the 30×30 bundle (42v × 14c, N_μ 714) σ = 6.278 eV gives 69.58 as,
but a 24 fs run after TDPROP's kick grows there:

| 30×30 df714, `cfm4`, 2 corrections | 24 fs run after a kick: n_c(5 / 18 / 22 / 24 fs) | the check |
|---|---|---|
| 50 as | 9.931e-11 / 9.933e-11 / 9.932e-11 / 9.934e-11: stable | accepted (probe at 62.5 as: 1.0000018 per step) |
| 69.58 as (the σ limit) | 9.93e-11 / 2.2e-9 / 2.0e-4 / 9.4e-3: grows (1.105 per step) | refused (probe at 87.0 as: 1.930 per step) |
| 100 as, check bypassed | 1.7 / 7.2 / 7.4 / 7.5: grows, saturated | refused by the σ test |
| `dt = "auto"` | — | 50.26 as (probes 1.930 at 87.0 as, 1.195 at 73.9 as, 1.0000025 at 62.8 as) |

The edge therefore lies between 62.8 and 69.6 as. On the whole ρ (`magnus2`'s default
frame) the predictor extrapolates the lab-frame ΔΣ, the recurrence does not apply, and the
probe alone decides (`auto` starts at `dt_measure`). The probe bounds the growth only to
about 2e-3 per step (48 steps at 1.25 dt): a mode growing more slowly, which matters only
over thousands of steps, is not excluded. The field is left out of both tests: E·r is not
extrapolated. `rk4` and `etdrk4` have their own explicit limits ([choosing](#choosing)),
which are not checked.

**Fixed steps sample by dense output.** `rk4`, `etdrk4` and `cfm4` step from
t0 by `lorrax.dt` and evaluate each sample of the `dt_measure` grid inside the
step that covers it, so `dt` need not divide `dt_measure`: `rk4` and `etdrk4`
by the cubic Hermite interpolant of the step's end values and slopes taken in
the frame of the diagonal $L$ (FSAL slopes, no extra ΔΣ), `cfm4` by its own
flow over the frozen generator to the sample time (exponentials only).
`magnus2` freezes ΔΣ at one node, so a sample inside its step would be first
order; it keeps samples on steps (`td_dense_output`). On CrI3 at 100 V/m
samples between steps stay on each method's fourth-order line: `etdrk4` at
15 as 9.1e-7 (8.5e-7 from the 10 and 25 as runs), `rk4` at 15 as 3.2e-6
(3.2e-6), `cfm4` 1.2e-5 at 40 as (1.2e-5) and 4.7e-7 at 15 as (2.4e-7). The
frame removes the band-energy phases before interpolating, which is exact for
the free flow and accurate for a response near the transition frequencies; a
component far from them (the counter-rotating part of a strong off-resonant
drive) oscillates faster in that frame than in the lab frame, so there keep
`dt` well below 1/(ω_drive + ΔE_max) or let it divide `dt_measure`.

### Accuracy limits {#accuracy-limits}

**Short pulses under `dp54`.** The controller starts with a step of
`dt_measure` and measures its error against `atol` while ρ is still at
equilibrium, so a pulse shorter than the first steps is integrated with an
amplitude error the error estimate does not see. On CrI3 (6×6 k, 2v2c, a
10 as Gaussian at 100 V/m) `rtol` = 1e-8 leaves 1e-5 to 3e-4 in the amplitude
of P(t) against the exact linear response, while `max_step` = 2 as gives
2e-11; the frequencies are unaffected. TDGW selects the same steps. Set
`control.timesteps.max_step` to about a fifth of the pulse width when the
amplitude matters.

**Rounding floor of the whole-ρ Magnus steps (removed).** Conjugating the
whole ρ, whose O(1) occupations set the absolute rounding of every
exponential (ε ≈ 1.1e-16 per entry), left a relative error floor of about
(ε/|ρ_cv|)·√N_steps ∝ 1/E₀ that rose as the step shrank: on CrI3 at 100 V/m
(|ρ_cv| ≈ 1e-9) the whole-ρ `cfm4` was lowest at 25 as, 3.4e-6, and rose to
6.6e-6 at 5 as. Eq. (4) removes it: the same runs reach 3.0e-9 at 5 as.

**Strang splitting in strong fields (removed).** The intraband term grows
with the field, and its Strang splitting made the whole-ρ `cfm4` second order
in strong fields (3.2e-6 at 5 as at 5e9 V/m); the triple jump restores order
4 (3.3e-9, table above).

### Choosing a method {#choosing}

The choice for a production run, with the costs measured on the 30×30 CrI3 ISDF
model (`cfm4` at 50 as, the stiff split near 1e-4), is on
[Running TD-aGW](../how-to/td-agw-run.md#integrators); the run log that records
the gate and the cost per fs is described there too. Measured operating points
(2026-10-16). Cost is ΔΣ evaluations per
simulated fs; error is the relative L2 error of P(t) over 30 fs.

| system | DP54 | ETDRK4 | CFM4 (2 corrections) | RK4 |
|---|---|---|---|---|
| CrI3, 6×6 k, 2v2c, ISDF COHSEX, 100 V/m (exact linear response as reference) | 3.5e-4 at 46 (rtol 1e-6); 3.3e-6 at 114 (rtol 1e-8) | 1.0e-4 at 80 (50 as); 6.5e-6 at 160 (25 as); 1.7e-7 at 400 (10 as) | 3.1e-5 at 80 (50 as); 1.9e-6 at 160 (25 as); 4.7e-8 at 400 (10 as) | 3.9e-8 at 800 (5 as) |
| synthetic 4v4c, 6×6 k, transitions to 12 eV, dense TDA kernel | 2.9e-5 at 337 (rtol 1e-6) | 2.2e-5 at 160 (25 as) | 6.4e-6 at 161 (25 as; whole-ρ `cfm4`) | 1.7e-5 at 800 (5 as) |

On the CrI3 window (transitions 1.4–4 eV) DP54's steps after the pulse are
long, and it needs the fewest ΔΣ evaluations at the measured accuracies;
at a fixed step CFM4 (two corrections) is 3.5× more accurate than ETDRK4 at
the same count, and fourth order to 3.0e-9 at 5 as. The bright-peak positions
move by 0.035 meV at rtol 1e-6 and not at all below. The same runs on four
GPUs through the ISDF adapter at 5e8 V/m reproduce the CPU errors to four
digits: DP54 (rtol 1e-6) 1.92e-5 at 80.6, 0.20 s per fs; the whole-ρ CFM4 (two
corrections, 50 as) 1.72e-5 at 80.4, 0.21 s per fs; ETDRK4 (25 as) 6.53e-6 at
160, 0.31 s per fs. On the synthetic window, whose 12 eV transitions limit an
explicit step, ETDRK4 and CFM4 need 2–4× fewer evaluations than DP54. RK4
diverges beyond h·ΔE_max ≈ 2.8 (150 as at 12 eV); ETDRK4 and CFM4 with two
corrections stay usable (error < 1e-2) to about 100 as on both windows;
`magnus2` is bounded (unitary) but second order, and its corrected step is
checked like `cfm4`'s.

**Default: `dp54`**, at `rtol` 1e-6 to 1e-7, with `max_step` near a fifth of
the pulse width for short pulses: it takes TDGW's steps, needs no step choice
and is as cheap as the best fixed step on narrow windows. Use `etdrk4` or
`cfm4` (two corrections, `dt` 25–50 as, or `auto`) when the window spans a
wide energy range or a fixed ΔΣ count is wanted for a long production drive;
both are fourth order with Γ and the intraband term. `cfm4` is the more
accurate at a given step and unitary, but checks its step against the
coupling (above) and, with Γ or the intraband term, takes three times the
exponentials. On
four A100s the ISDF ΔΣ dominates the wall time (2.95 ms per call at 2v2c, 9.0
ms at 30v × 14c on 6×6 k, against 1.05 ms for one exponential and 0.27 ms per
sample at 30v × 14c), so the wall time follows the ΔΣ count. On synthetic 30v
× 14c headers with a rank-1000 stand-in ΔΣ, a DP54 step costs 2.9 / 4.9 /
23.4 ms at 64 / 144 / 900 k (one exponential 1.1 / 1.2 / 3.3 ms) and rank 0
peaks at 4.2 GB at 900 k. At 900 k the DP54 drive takes 0.69 s per fs, of
which its 153.5 stand-in ΔΣ calls per fs at 4.3 ms each are 0.66 s (ETDRK4 at
25 as: 0.73 s, of which 160 calls are 0.69 s). A production run therefore
costs about (ΔΣ calls per fs) × t_ΔΣ, with t_ΔΣ the model's own per-call
time; the rest of the drive is about 0.03–0.04 s per fs if a call costs
inside the step what it costs alone. A fixed step fixes the ΔΣ count in
advance: four per step for ETDRK4 and for CFM4 with two corrections, i.e. 80
per fs at 50 as.

With Γ or the intraband term the fourth-order `cfm4` takes three times the exponentials,
which shows where ΔΣ is cheap: on four A100s, on the synthetic 30v × 14c 30×30 header with
the rank-1000 stand-in ΔΣ (4.3 ms per call), the wall per fs of 2 fs (compiled run) is
DP54 1.29 s, RK4 (10 as) 1.89 s, ETDRK4 (25 as) 1.16 s, and CFM4 (25 as, two corrections)
2.45 s with the intraband term off, 5.17 s with Γ = 0.05 eV and 9.41 s with the intraband
term on (the whole-ρ CFM4 with Strang: 1.85 / 1.88 / 2.27 s). A run of one chunk also
writes one checkpoint of the whole carry, 0.3–0.9 s there (DP54's carry is ten states). On
the 516 window (6×6 k, 2v2c, a dense kernel) the four-GPU CFM4 errors equal the CPU ones to
every printed digit (3.0e-9 at 5 as, 0.96 s per fs; the whole-ρ CFM4 7.4e-7). With the
ISDF model the step is ΔΣ-bound (96 ms per call at 30×30, against 3.3 ms for one 44-band
exponential at 900 k), and the bundle route has no intraband term.

## Validation ladder {#validation}

Six rungs test the propagator against references that share no code with
it — closed forms, the BSE of the same kernel, SciPy's DOP853 — each through
the driver's whole path (a header on disk or a GW bundle → deck → `stages` →
`tdgw.h5`, on a 2×2 host mesh with k padded) and each with red twins that
must fail (`tests/tdagw/test_validation_*_cpu.py`, `JAX_PLATFORMS=cpu` with
four host devices; the algebra is `tools/td_response.py`). Tolerances are
fixed in advance from the scheme's order, the rounding and the size of the
next order in the drive; `pytest --junitxml` records every measured value as
a testsuite property. Weak drives run at $\pm E_0$ and are analysed through
the odd part, linear up to $O(E_0^3)$: a window without inversion and with
intra-manifold dipoles has a physical second-order response
($P^{(2)}/P^{(1)} \sim |\rho_{cv}|$) that would otherwise set the floor.

These rungs have the GW collision term off. With it enabled, Hermiticity
remains a check, but purity and the spectrum of ρ are not invariants;
particle conservation has the closure conditions of
[the collision operator](td_agw_dynamical.md#damping).

**Linear response is the full BSE.** Linearized about $\rho^\mathrm{eq}$,
Eqs. (1)–(2) for $z = (\delta\rho_{cv}, \delta\rho_{vc})$ over the pairs
$t = (k, c, v)$ read

$$
i\dot z = L z + E_b(t)\,u_b,\qquad
L = \begin{pmatrix} D + K_{XX} & K_{XY}\\ -K_{YX} & -D - K_{YY}\end{pmatrix},\qquad
u_b = (r^b_{cv}, -r^b_{vc}), \tag{V1}
$$

with $D = \mathrm{diag}(E_c - E_v)$ and $K_{XY}[t, t'] = K_{kcv,k'v'c'}$ etc. the
blocks of $\Delta\Sigma_{k,nm} = \sum K_{knm,k'n'm'}\,\delta\rho_{k',n'm'}$: the
full (non-TDA) BSE of the kernel. With $L = Z\,\mathrm{diag}(\lambda)\,Z^{-1}$
and $o_a = w_k(r^a_{vc}, r^a_{cv})$ every mode carries the residue
$R^S_{ab} = (o_a\cdot Z_S)(Z^{-1}_S\cdot u_b)$ — for a positive mode
$w_k D^S_a D^{S*}_b$ with $D^S = \langle 0|\mathbf r|S\rangle$ — and

$$
\langle r_a\rangle(t) = -i\sum_S R^S_{ab}\int_{-\infty}^{t} e^{-i\lambda_S(t-t')}E_b(t')\,dt',
\qquad
\chi_{ab}(\omega) = -\frac{g_s}{\Omega}\sum_S\frac{R^S_{ab}}{\omega - \lambda_S}, \tag{V2}
$$

$\mathbf P = -g_s\langle\mathbf r\rangle/\Omega$, in closed form for Gaussian
drives (Faddeeva functions). Three consequences are checked with no
frequency-resolution limit: P(t) itself; the projection of the propagated
state on the left eigenvectors (scaled $Z^\dagger\eta Z = 1$,
$\eta = \mathrm{diag}(1, -1)$, so $|c_S|^2/N_k$ is the exciton number per
cell), which after the pulse is the first-order amplitude rotating as
$e^{-i\Omega_S t}$; and the damped transform, exact by the convolution theorem,

$$
\mathcal F[P\,e^{-\eta t}](\omega) = \chi(\omega + i\eta)\,\mathcal F[E\,e^{-\eta t}](\omega), \tag{V3}
$$

up to the truncation $e^{-\eta T}$. Frequencies fitted to the time signal
(matrix pencil, fixed-frequency least squares) are ill-posed below the
Fourier limit $2\pi/T$ and are not used.

| rung | system, drive | reference | gate (tolerance) | measured (login node, lorrax_A) | red twins (all fail) |
|---|---|---|---|---|---|
| 1 Rabi | two-level × 5 k (1.95–2.06 eV, d = 1 bohr); cw 2e9 V/m, π pulse 3.9e9 V/m (τ 4 fs); Γ = 0 and 5/20 meV; 48 fs | exact (below); DOP853 at rtol 1e-13 for detuned and Gaussian drives | ρ to 1e-7 (order-4 schemes), 3e-5 (`magnus2`), 1e-8 (`dp54` 1e-10); observed order within 0.3 of nominal | `rk4` 1.7e-8, order 4.006; `etdrk4` 2.4e-9, 4.000; `cfm4` 1.6e-10, 4.000 (4.000 with Γ); `magnus2` 1.3e-5, 2.000; `dp54` 1.6e-9, ×98 per 100× rtol | bra/ket-swapped dipole: O(1); gap/(2 Ry) for (e/Ry)·Ry/Ha: 9.7e-6 at every step, order ≈ 0 |
| 2 TD == BSE | 3v3c spinors, 3×3 k, ISDF-form Hartree + SEX (kernel.h5); kicks (τ 50 as) and σ± pulses (τ 1.5 fs, 1e6 V/m) | (V1)–(V3) | P(t) 1e-6 `dp54`, 1e-5 `etdrk4`/`cfm4` (10 as); per state amplitude 1e-5, frequency 1 µeV; Im χ± 1e-5; σ+/σ− population ratio = I+/I− | 5.2e-8 / 4.3e-7 / 3.0e-7; 156 states 3.4e-7 and 5.6e-9 eV; 1.6e-7 / 1.2e-7; 9.1e-7 | TDA kernel: P off by 1.1e-2; full run vs TDA BSE; reversed I± |
| 3 conservation | same window, σ+ 2e9 V/m (0.13 excitons per cell), 30 fs | unitary flow, E[ρ] (below) | Tr 1e-11, Hermiticity 1e-12, purity 1e-11 (Magnus) / 1e-8 (RK); work theorem 1e-6; drift after the pulse 1e-9 (`magnus2` in `whole_rho`, the default; `interaction`: drift(2 as)/drift(1 as) in 4 ± 1 and ≤ 3.52e-8 at 1 as); equilibrium 1e-15 (RK) / 2e-12 (Magnus) | Tr ≤ 3.0e-12, Hermiticity ≤ 7.3e-14, purity 5.9e-14 (`cfm4`) to 1.9e-10 (`rk4`); work 1.3e-8 (`magnus2` 4.1e-7 at 1 as); drift ≤ 4.5e-11 (`magnus2` `interaction` 7.0e-8 / 1.76e-8 at 2 / 1 as, ratio 4.000); equilibrium 0 (RK), ≤ 6.5e-13 (Magnus) | K → (1 + 0.1i)K; dephasing (purity); E without the ½; ΔΣ of ρ instead of ρ − ρ^eq; one k shifted by 1e-10 |
| 4 f-sum | same window, IP / full / TDA; kick κ = 1e-5 | odd short-time series of P (below) | m1 1e-7, m3 1e-6, j(0+) 1e-8 | m1, m3, j(0+) ≤ 2.6e-10 (the κ²\|r\|² residue), IP, full and TDA; interacting m1/S_IP − 1 = −4.99 % | TDA run vs the full m1 (5.1e-4); interacting run vs S_IP |
| 5 helicity | Haldane honeycomb, 6×6 k (C3), ISDF-form interaction; τ 2 fs pulses on S0 (σ+, 1.4754 eV) and S1 (σ−, 2.0885 eV) | C3 selection, first-order theory | bright states pure: min(I±)/max ≤ 1e-13; N(σ∓)/N(σ±) ≤ 1e-12; N(S0) vs theory 1e-5 | 8.5e-15; 1.1e-20 and 2.5e-17; 8.2e-7 | reversed convention; (cos, −sin) as σ+; 1e-3 x/y dipole anisotropy (leak 3.4e-8) |
| 6 magneto-optics | Haldane ±φ; x and y kicks; η 0.15 eV, 70 fs | (V3), sheet and film optics (below) | χ, ε_xy, σ_xy, Kerr, Faraday 1e-5; Onsager 1e-10; σ_xy/ω² flat to 1 %; film → sheet linear in d | 8.3e-8 (χ_xx), 1.9e-7 (χ_xy, ε_xy), 2.3e-7 (σ_xy, Kerr, Faraday; maxima 897 and 11 mrad); 6e-15; 9.2e-4; 3.4e-5 / 1.8e-4 at 6.6 Å, anisotropic tensor 1.9e-5 → 1.9e-7 for d ÷ 100 | reversed table convention; Onsager without the sign; transposed σ in Kerr; circular-mode film on an anisotropic tensor |

**Rung 1.** A circular drive on a circular transition has no
counter-rotating term: with $\mathbf r_{cv} = d\,(1, -i, 0)/\sqrt2$ and
σ+ $= E_0(\cos\omega t, \sin\omega t, 0)$, $H_{cv} = g\,e^{-i\omega t}$,
$g = E_0 d/\sqrt2$, and $T = \mathrm{diag}(1, e^{-i\omega t})$ makes $H$ time
independent and commutes with $\Gamma\circ$, so ρ(t) is one exponential of an
affine generator with or without relaxation: $\rho_{cc} = (4g^2/W^2)\sin^2(Wt/2)$,
$W^2 = \delta^2 + 4g^2$, beyond the rotating-wave approximation (the σ− drive
of the same transition is the counter-rotating problem, exact too). The
observed order separates truncation, which scales with h, from a modelling
error, which does not: the energy-unit twin (2 Ry/Ha − 1 = 3.6e-7) sits 9.7e-6
from the run at every step.

**Rung 3.** $\Delta\Sigma$ is Hermitian for Hermitian $\delta\rho$, so the flow
is unitary: $\mathrm{Tr}\,\rho_k$, Hermiticity and the spectrum of ρ
($\rho^2 = \rho$ at Γ = 0) are kept. With $K_{knm,k'n'm'} = K_{k'm'n',kmn}$
($\bar V$ symmetric, $W(-q) = W(q)^T$) it is the gradient flow of

$$
E[\rho] = g_s\sum_k w_k\Big[\mathrm{Tr}(\varepsilon_k\rho_k)
  + \tfrac12\,\mathrm{Tr}\big(\Delta\Sigma_k[\delta\rho]\,\delta\rho_k\big)\Big], \tag{V4}
$$

the ½ being the Hartree and screened-exchange double counting, and
$dE/dt = \Omega\,\mathbf E\cdot\mathbf j$ with the measured current (the
$iE_b[r_b, r_a]$ term absent from TDGW's velocity drops out of $\mathbf E\cdot\mathbf j$
by antisymmetry). The Magnus rounding floor above sets the equilibrium
tolerance: $N_\mathrm{exp}\,\varepsilon$.

**Rung 4.** After $\rho(0^+) = e^{-i\kappa x}\rho^\mathrm{eq}e^{i\kappa x}$,
$P_x(t) = (g_s/\Omega)\,\kappa\,(m_1 t - m_3 t^3/6 + \dots)$ with
$m_n = \sum_S\lambda_S^n R^S_{xx}$; $m_1 = o\cdot L\cdot u$ is the window's
f-sum, for the IP window $S_\mathrm{IP} = 2\sum_k w_k\sum_{cv}(E_c - E_v)|x_{cv}|^2$
and with the kernel $S_\mathrm{IP} + o\cdot K_L\cdot u$, the double commutator of
the interaction inside the window (zero for a local interaction in a complete
basis). The full BSE obeys the sum rule of its own kernel (Thouless); the
TDA drops the coupling block's $2\,\mathrm{Re}(d^\dagger K_{XY} d^*)$.

**Rung 5.** On a k grid closed under C3, with interaction and dipoles
covariant (lattice sums truncated on a disk — a parallelogram breaks C3 at
4.5e-6), each exciton carries a C3 eigenvalue and
$\langle 0|x \mp iy|S\rangle$ vanishes unless it matches: bright states are pure
σ+ or σ−. A σ+ drive couples through $\langle S|x + iy|0\rangle$, i.e.
$I_+ = |D_x - iD_y|^2/2$, and leaves σ− states empty even where its spectrum
covers them; what leaks is the pulse's own counter-rotating component
$(x - iy)\,\tilde g(\Omega_S + \omega)$, 4e-7 at τ = 1 fs (first-order theory
reproduces it) and below 1e-20 at 2 fs.

**Rung 6.** $\chi_\pm = \chi_{xx} \pm i\chi_{xy}$ for the $(x \pm iy)$ field; from
the exciton table of an in-plane isotropic (C3, C4, C6) window
$\chi_+(z) = -(g_s/\Omega)\sum_S[I_+(S)/(z - \Omega_S) - I_-(S)/(z + \Omega_S)]$
and $\chi_-$ with $I_+ \leftrightarrow I_-$: the BSE $\varepsilon_{xy}$ of an
exciton table. The sheet conductivity is $\sigma^\mathrm{2D} = -i\omega\,\chi(\omega + i\eta)\,L_z$
at real ω — the Lorentzian-broadened convention $\varepsilon(\omega) = 1 + 4\pi\chi(\omega + i\eta)$;
continuing $-i(\omega + i\eta)\chi$ instead changes σ by O(η/ω) and spoils the
d → 0 limit below (9 % on the toy). At normal incidence a free-standing sheet
has $T = (1 + \tfrac{2\pi}{c}\sigma^\mathrm{2D})^{-1}$,
$R = -\tfrac{2\pi}{c}\sigma^\mathrm{2D}T$ on the in-plane Jones vector, and Kerr
and Faraday are $\theta + i\epsilon$ of the outgoing ellipse (lab frame, from x
toward y counter-clockwise viewed from +z; rotations compared modulo π). The
film model takes $\varepsilon_\mathrm{film} - 1 = (\varepsilon - 1)L_z/d$
(d = 6.6 Å, the interlayer spacing of bulk CrI3, as Wu et al. 2019) and the
Airy formulas on the eigenpolarizations of the in-plane $\varepsilon_\mathrm{film}$
(circular, $n_\pm = \sqrt{\varepsilon_{xx} \pm i\varepsilon_{xy}}$, only when it is
isotropic); it converges to the sheet linearly in d. The interband
length-gauge $\sigma_{xy} = -i\omega\chi_{xy}$ vanishes as ω²: the dc anomalous
Hall term would come from the intraband commutator, absent from a
velocity-built (itype 1) header. Time reversal (φ → −φ, or M → −M) flips
$\varepsilon_{xy}$ and keeps $\varepsilon_{xx}$ (Onsager).

**CrI3 516** (6×6 k, PBE+SOC spinors, FM; the bundle route with the ISDF
Hartree + SEX model, N_μ 978, and TDREF's eqp; `tools/td_validation_cri3.py`,
run `DEV/829_td_agw_validation_20261016`, 2026-10-16). The leg samples the
model's kernel column by column — Hermitian as a matrix to 1.1e-14, the
symmetry behind Eq. (V4) — and propagates ±E₀ drives with `etdrk4` at 10 as
for 60 fs on four A100s (24 000 ΔΣ per window: 168 s for four 8v8c trials,
212 s for eight 2v2c trials); rungs 2, 4–6 are then evaluated on the host.

- *8v8c* (bands 123–138, 2304 pairs, `degeneracy_mode = strict`): P(t) of
  the x and y kicks equals the full-BSE closed form to 2.8e-7 and
  χ(ω + iη) to 1.1e-7 (η 0.2 eV, e^{−ηT} 1.2e-8). The spectrum is real
  (|Im λ| ≤ 6e-15 eV), in-plane isotropic to 1.5e-9 (χ_xx − χ_yy) and 7e-9
  (χ_xy + χ_yx), and every bright state is pure σ+ or σ− to 2.9e-10.
- *2v2c* (bands 129–132, `degeneracy_mode = off`: on the QP spectrum bands
  128 and 129 are 0.548 meV apart at k = 24, inside the strict gate's 1 meV,
  although at least 48.2 meV apart in DFT, the minimum over k). A σ+ pulse
  (τ 2 fs, 1e6 V/m) on the lowest bright σ+ state (1.8087 eV, I+ 0.0152
  bohr² per cell) fills σ− states at
  1.62e-17 of the σ+ population, the counter-rotating leak of first-order
  theory (1.62e-17), and N(S+) equals first order to 8.3e-8; the σ− pulse
  is the mirror image (5.5e-17). ε_xy, σ_xy, Kerr and Faraday from TD equal
  the BSE to 7.3e-8, and χ± from the exciton table to 5e-8. At η 0.2 eV the
  free sheet's Kerr rotation peaks at 314 mrad (1.995 eV), the Faraday
  rotation at −0.52 mrad (2.215 eV); the 6.6 Å film is within 8.5e-6 of the sheet.
- *f-sum.* In the window the interaction lowers m1 below the IP sum with
  QP energies by 30 % (8v8c: 2.063 against 2.946 Ha·bohr² per cell) and 33 %
  (2v2c: 0.1121 against 0.1678). With DFT energies the IP sums are 1.507 and
  0.0807, so the QP-energy sum is the DFT one times the weighted ratio of
  transition energies (1.96, 2.08) — r is DFT's — and m1 sits at 1.37 and
  1.39 times it. Against the whole dipole set (130 valence × 30 conduction
  bands, 26.58) the 8v8c window holds 5.7 % (1.16 % of N_e = 130) and 2v2c
  0.30 % (0.062 %); the 160-band set itself holds 20 % of N_e, the rest lying
  in higher bands (the velocity p − i[r, V_NL] of a nonlocal pseudopotential
  also moves the complete-basis sum off N_e).

**Scripts outside pytest.** A script that imports jax before `runtime` runs in
float32 unless `JAX_ENABLE_X64=1` (`runtime.set_x64_on_imported_jax`). On the
synthetic window that shows as a 3e-7 error in the measured field and a P(t)
error that grows as the step shrinks (`etdrk4` 8.5e-6 at 20 as to 5.8e-5 at
2.5 as, `cfm4` 5e-2 to 1e-1 at 10 as); with x64 the same runs give 3.5e-7 at
20 as to 2.3e-9 at 5 as, order 4 down to the third-order residue.
Rung 1's order criterion catches it.

## Layout and programs {#layout}

ρ is `(ntr, nk_pad, nb, nb)` complex128: a leading trial axis (drives
propagated together, `lorrax.trials`), then the flat k axis sharded over
every device, `P(None, ('x','y'), None, None)`, padded to a multiple of the
device count with inert zero blocks (zero weight and operators; the error
norm counts the logical entries only). The static operators share the k
sharding. Each exponential is a per-device batched `eigh` inside
`shard_map`; the only cross-k access outside ΔΣ is the $k+b$ gather of
Eq. (3). The `isdf` adapter moves Δρ from this k layout to the model's band
layout, and ΔΣ back, with explicit staged all-to-alls
(`common.staged_reshard.batch_to_face_reshard` / `face_to_batch_reshard`,
one per mesh axis, built outside the traced step); an implicit sharding
constraint there made XLA replicate the whole state on every call
("Involuntary full rematerialization": an all-gather of ρ and of ΔΣ; AUDIT
B10). The adapter now adds exactly four all-to-alls to the model's own
collectives, with the same values: on the 30×30 df714 model (P4) the ΔΣ program has 4
all-to-all, 2 all-gather, 2 all-reduce and 1 reduce-scatter (96.4 ms per call) and the run
logs no rematerialization (TDPROP's runs of the same decks: 7 and 38), and the 516-bundle
ISDF run reproduces TDPROP's GPU output bitwise (20000 samples). One jitted program advances a chunk of `lorrax.chunk` samples
(default `ntcheck`) with the carry donated; the host receives each chunk's
samples and writes them with the checkpoint.

## Output {#output}

The output is TDGW 0.3.0's: `tdgw.h5` with `tdgw_header` copied from
`load_header_from` (`version`, `tgrid`, `tsteps`, `structure`,
`wfn_header`, `dipole`), `tdgw_data/time_domain/<name>/{times,data}` per
measurement and `tdgw_data/checkpoint/{it_next,tchkp,lessG_knn}`, plus a
`tdgw.toml` in TDGW's schema (the `[lorrax]` table dropped, the drive of
that trial in `[system.Efunc]`). `tdgw-pp linear` reads the directory
unchanged.

**Restart, collision off.** Every checkpoint also stores the integrator's whole
carry in the
first trial's file, `tdgw_data/checkpoint/lorrax` (`leaf_<i>` datasets at
logical k; attributes `step`, `n_leaves` and `fingerprint`, the settings the
carry is valid for): ρ with the Magnus node history, the FSAL slope or DP54's
controller state. `control.files.load_checkpoint_from` resumes from it into a
new `outdir`, copies the rows before it and continues bitwise: all five methods
reproduce the uninterrupted run's samples and final checkpoint exactly on a
2×2 host mesh (`tests/tdagw/test_prop_restart_cpu.py`), while the carry's
history zeroed (`cfm4`) or DP54's controller restarted from the state alone,
as TDGW restarts, differ at 1e-12 and 6e-10 in ρ_cv. Only `outdir`,
`load_checkpoint_from`, `lorrax.chunk` and, for fixed steps, `tf` may change
(`td_restart_settings`); a TDGW checkpoint holds the state but no carry and
refuses (`td_restart_carry`). TDGW and `tdgw-pp` read the checkpoint's three
datasets and ignore the group.

A collision-enabled run refuses checkpoint restart (`td_collision_restart`):
the checkpoint does not persist its held collision coefficients and refresh
state. The restart guarantees above apply with `lorrax.collision = off`.

## What refuses {#refusals}

A deck with neither `load_header_from` nor `[lorrax] system = "bundle"`, or
with both, or with `use_vmtxel` (`td_header_source`), patched sampling, box
geometry, `velocity_operator`, a `magnus2` step that does not divide
`dt_measure` (`td_dense_output`), `n_corr` < 1, a `linear` predictor outside
`cfm4` (`td_predictor`), a `magnus2` / `cfm4` step above the stability limit or with a
growing step probe (`td_magnus_stability`), `magnus_frame = whole_rho` outside `magnus2`
(`td_magnus_frame`), a restart into its own `outdir`
(`td_checkpoint_outdir`), from a checkpoint without a LORRAX carry
(`td_restart_carry`) or under other settings (`td_restart_settings`), and a
`dp54` step that falls below SciPy's minimum (`td_dp54_step_too_small`) refuse
with a named GATE. The
bundle route adds `td_bundle_input`, `td_ivbm`, `td_bundle_key` (bundle keys
on a header deck), `td_cv_gap`, `td_velocity_hermitian`, `td_dipole_energies`,
`td_eigenvector_kpts`, `td_degeneracy_mode` and `td_tda_model`.

**BerkeleyGW dipoles.** The dipole is whatever the header carries. For
BerkeleyGW's momentum matrix elements, run TDGW once with `use_vmtxel = true`
(it writes the header before it propagates; dipole `itype` 1, no intraband
commutator) and point `load_header_from` at that run. On CrI3 516 (6×6 k,
2v2c, BerkeleyGW's extended kernel, `vmtxel` with b1 rescaled) the propagator
then takes TDGW's 803 steps and matches its P(t) to 1.1e-15, and the Im ε_xx
peaks 1.8434 / 2.1745 / 2.3422 / 2.5989 eV lie within 0.7 meV of BerkeleyGW's
full-BSE absorption (1.8434 / 2.1752 / 2.3422 / 2.5988 eV). On that grid
TDGW's own finite-difference dipoles give different x and y spectra (peaks at
1.9673 and 2.2410 eV in x only), so a `vmtxel` header is the one to use there.
