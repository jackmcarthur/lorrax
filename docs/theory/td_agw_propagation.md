# TD-aGW propagation

`python -m tdagw.tdgw_jax -i tdgw.toml` propagates the real-time adiabatic-GW
(TD-aGW) equation of motion of Xiaoxun Gong's TDGW code in TDGW's
conventions, with TDGW's deck, header and output files. This page owns the
equation of motion as integrated, the integrators, the array layout and the
output contract. The interaction ΔΣ[ρ] of the ISDF model is derived on
[TD-aGW interaction](td_agw.md); the deck keys are in the
[input reference](../input_reference.md#td-agw).

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

**ΔΣ is injected.** The integrators see the self-energy update only as a
callable of the instantaneous state, `apply(params, rho, t) -> dSigma` (Ha,
Hermitian, linear in $\rho - \rho^\mathrm{eq}$), `tdagw.self_energy.SelfEnergyModel`:

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

## Integrators {#integrators}

Cost is counted in ΔΣ evaluations, which dominate at any window where the
interaction is non-trivial; an exponential costs one batched
$n_b\times n_b$ `eigh` per k and two products.

| `lorrax.method` | scheme | order | ΔΣ per step | exponentials per step | step |
|---|---|---|---|---|---|
| `dp54` | Dormand–Prince 5(4), SciPy's `RK45` | 5 (error estimate 4) | 6 (FSAL) | 0 | adaptive |
| `rk4` | classical Runge–Kutta | 4 | 4 | 0 | `lorrax.dt` |
| `etdrk4` | Krogstad's exponential RK on the diagonal $L_{nm} = -i(E_n - E_m) - \Gamma_{nm}$ | 4 | 4 | 0 (diagonal) | `lorrax.dt` |
| `magnus2` | exponential midpoint, self-consistent midpoint $H$ | 2 | `n_corr` | `n_corr` + 1 | `lorrax.dt` |
| `cfm4` | commutator-free Magnus CFET4:2 at the Gauss nodes | 4 | 2 `n_corr` | 4 `n_corr` + 2 | `lorrax.dt` |

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

**`magnus2` and `cfm4`** write the commutator part as
$\rho \to U\rho U^\dagger$ with $U = e^{-ih\bar H}$, so the spectrum of ρ is
conserved exactly when $\Gamma = 0$ and the intraband term is off. $H$ at a
node needs ΔΣ of ρ at that node: a predictor extrapolates ΔΣ from earlier
steps (quadratic through the last three midpoints for `magnus2`, cubic
through the last two steps' four Gauss-node values for `cfm4`), and each
correction propagates $\rho_n$ to the node(s) and re-evaluates ΔΣ there.
Each correction gains one order of $h$, so the cubic predictor plus one
correction gives order 4; without a correction the scheme is unstable, and
`n_corr` < 1 refuses. The extrapolation amplifies the predictor's error, so
one correction contracts it only while $h\,\lVert\partial\Delta\Sigma/\partial\rho\rVert$
is small: with the CrI3 COHSEX kernel (6×6 k, 2v2c) `cfm4` with one
correction diverges from 50 as, while two corrections are stable to 300 as.
The default is two. The first two (`cfm4`) or three (`magnus2`) steps,
before the history exists, take three extra corrections. Relaxation and the
intraband term are not commutators; they enter by Strang splitting
$R(h/2)\,M(h)\,R(h/2)$, with $R$ the exact relaxation around an RK4 step of
the intraband flow, so with either one active the global order is 2 with an
error constant set by the size of $[M, R]$.

### Three accuracy limits {#accuracy-limits}

**Short pulses under `dp54`.** The controller starts with a step of
`dt_measure` and measures its error against `atol` while ρ is still at
equilibrium, so a pulse shorter than the first steps is integrated with an
amplitude error the error estimate does not see. On CrI3 (6×6 k, 2v2c, a
10 as Gaussian at 100 V/m) `rtol` = 1e-8 leaves 1e-5 to 3e-4 in the amplitude
of P(t) against the exact linear response, while `max_step` = 2 as gives
2e-11; the frequencies are unaffected. TDGW selects the same steps. Set
`control.timesteps.max_step` to about a fifth of the pulse width when the
amplitude matters.

**Rounding floor of the Magnus schemes.** `magnus2` and `cfm4` conjugate the
whole ρ, whose O(1) occupations set the absolute rounding of every
exponential, while the induced coherence in weak-field linear response is
small (|ρ_cv| ~ 1e-6). Each step therefore adds a relative error of order
ε/|ρ_cv|, and the P(t) error stops falling near 1e-8 after 10⁴ steps; RK4,
ETDRK4 and DP54 add small increments and reach 1e-11. The floor is far below
spectroscopic tolerances. The cure is the interaction picture on
Δρ = ρ − ρ^eq: integrate diag(E^QP) as exact phases, take exponentials of the
small coupling only, and update
Δρ ← P (U Δρ U^† + [U − 1, ρ^eq] U^†) P^†, every term of the size of Δρ
(not implemented).

**Strang splitting in strong fields.** The intraband term is not a
commutator, so `magnus2` and `cfm4` split it off (Strang), and its splitting
error grows with the field. On CrI3 at 5e9 V/m `cfm4` (two corrections) is
second order, 3.5e-6 / 1.4e-5 / 9.0e-5 at 5 / 10 / 25 as over 10 fs, while
with the intraband term off it is fourth order, 8.8e-10 / 1.4e-8 / 5.7e-7, and
beats ETDRK4 (2.3e-6 at 25 as). ETDRK4 and DP54 carry the intraband term in
their explicit part and stay fourth/fifth order (ETDRK4 3.1e-6 at 25 as with
it on). A fourth-order composition of the Strang step (Yoshida's triple
jump, three Strang steps per step) or an exponential integrator for the
intraband flow would remove the limit (not implemented).

### Choosing a method {#choosing}

Measured operating points (2026-10-16). Cost is ΔΣ evaluations per
simulated fs; error is the relative L2 error of P(t) over 30 fs.

| system | DP54 | ETDRK4 | CFM4 (2 corrections) | RK4 |
|---|---|---|---|---|
| CrI3, 6×6 k, 2v2c, ISDF COHSEX, 100 V/m (exact linear response as reference) | 3.5e-4 at 46 (rtol 1e-6); 3.3e-6 at 114 (rtol 1e-8) | 1.0e-4 at 80 (50 as); 6.5e-6 at 160 (25 as) | 1.4e-5 at 80 (50 as); 3.4e-6 at 160 (25 as) | 3.9e-8 at 800 (5 as) |
| synthetic 4v4c, 6×6 k, transitions to 12 eV, dense TDA kernel | 2.9e-5 at 337 (rtol 1e-6) | 2.2e-5 at 160 (25 as) | 6.4e-6 at 161 (25 as) | 1.7e-5 at 800 (5 as) |

On the CrI3 window (transitions 1.4–4 eV) DP54's steps after the pulse
are long, and it needs the fewest ΔΣ evaluations at every accuracy; the
bright-peak positions move by 0.035 meV at rtol 1e-6 and not at all below.
On the synthetic window, whose 12 eV transitions limit an explicit step,
ETDRK4 and CFM4 need 2–4× fewer evaluations than DP54. RK4 diverges beyond
h·ΔE_max ≈ 2.8 (150 as at 12 eV); ETDRK4 and CFM4 with two corrections stay
usable (error < 1e-2) to about 100 as on both windows; `magnus2` is
unconditionally bounded but second order.

**Default: `dp54`**, at `rtol` 1e-6 to 1e-7, with `max_step` near a fifth of
the pulse width for short pulses: it takes TDGW's steps, needs no step
choice and is the cheapest on narrow windows. Use `etdrk4` or `cfm4` (two
corrections, `dt` 25–50 as) when the window spans a wide energy range or a
fixed step that divides `dt_measure` is wanted for a long production drive;
in strong fields with the intraband term on prefer `etdrk4`, which stays
fourth order where `cfm4` drops to second (above).
On four A100s the ISDF ΔΣ dominates the wall time (2.95 ms per call at
2v2c, 9.0 ms at 30v × 14c on 6×6 k, against 1.05 ms for one exponential and
0.27 ms per sample at 30v × 14c), so the wall time follows the ΔΣ count.

## Layout and programs {#layout}

ρ is `(ntr, nk_pad, nb, nb)` complex128: a leading trial axis (drives
propagated together, `lorrax.trials`), then the flat k axis sharded over
every device, `P(None, ('x','y'), None, None)`, padded to a multiple of the
device count with inert zero blocks (zero weight and operators; the error
norm counts the logical entries only). The static operators share the k
sharding. Each exponential is a per-device batched `eigh` inside
`shard_map`; the only cross-k access outside ΔΣ is the $k+b$ gather of
Eq. (3). One jitted program advances a chunk of `lorrax.chunk` samples
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

## What refuses {#refusals}

A deck without `load_header_from` or with `use_vmtxel`
(`td_header_source`), restart from a checkpoint (`td_checkpoint_restart`),
patched sampling, box geometry, `velocity_operator`, a fixed step that does
not divide `dt_measure`, `n_corr` < 1, and a `dp54` step that falls below
SciPy's minimum (`td_dp54_step_too_small`) refuse with a named GATE.
