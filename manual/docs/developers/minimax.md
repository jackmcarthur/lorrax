# minimax

`import minimax` The time–frequency quadrature service: short exponential sums for every frequency kernel of GW.
{: .page-summary }

Every frequency integral in GW is a sum over poles. $G$ has poles at the band energies, $W^c$ at its own pole energies $\Omega$, and χ⁰ and Σ are convolutions of the two. A convolution in frequency is a product in time, so LORRAX evaluates χ⁰ and Σ as sums over a few time nodes, $t_i$, of products of time-domain Green's functions and $W$. What makes this possible is one approximation per kernel: a reciprocal such as $1/\Delta$ or $1/d$, over the whole range of transition energies or denominators the run contains, written as $\sum_i w_i\,e^{\pm i t_i d}$ or $\sum_i w_i\,e^{-\tau_i\Delta}$. Each exponential factorizes into a function of the occupied state, a function of the empty state (or of the pole of $W$), and a function of the frequency, so each node costs one product of a $G$ and a $G$ or $W$ on the centroid basis. `minimax` builds those sums. Given the bandwidths of $G$ and $W$, the band gap, the broadening and a target accuracy, it returns the nodes $t_i$ and weights $w_i$, and certifies or measures the error. The physics and the derivations are on [Time–frequency transforms](../theory/time-frequency.md); this page is the developer's view: which variables go in, which constructor to call, what comes out, and how accurate it is.
{: .lead }

<div class="glance" markdown>

Import
:   `import minimax` (top-level names only; importing a submodule from outside the package fails the layering lint)

Depends on
:   NumPy at import; SciPy and mpmath load lazily inside the builders that need them

Devices
:   host only. Every rule is solved in float64 NumPy on the host; LORRAX's GW code converts it to complex128 JAX arrays (`gw.minimax_screening.MinimaxNodes`)

Used by
:   `gw.minimax_screening`, `gw.screening`, `gw.sigma_box_plan`, `gw.response_bank`, `gw.mpa.model`, `gw.cd.chi0_rules`, `gw.w_isdf`

</div>

## 1 From physics to a rule {: #physics }

### 1.1 The kernels {: #kernels }

Two kernels carry all of LORRAX's frequency dependence.

**The response kernel.** The independent-particle response at a complex frequency $z$ is a sum over occupied–empty pairs $(v,c)$ with transition energy $\Delta=E_c-E_v>0$,

$$
\chi^0(z)=\sum_{vc}(f_v-f_c)\,\rho_{vc}\,\rho_{vc}^*\,K_z(\Delta),\qquad K_z(\Delta)=\frac{2\Delta}{z^2-\Delta^2}. \tag{1.1}
$$

At $z=0$, $K_0(\Delta)=-2/\Delta$. With an energy reference $\mu$ between the occupied and empty states, an exponential in $\Delta$ factorizes,

$$
\frac1\Delta\approx\sum_i\alpha_i\,e^{-\tau_i\Delta},\qquad e^{-\tau\Delta}=e^{-\tau(E_c-\mu)}\,e^{-\tau(\mu-E_v)}, \tag{1.2}
$$

so each node $\tau_i$ is one product of the empty-state propagator $\sum_c\psi_c\psi_c^*e^{-\tau_i(E_c-\mu)}$ and the occupied-state propagator $\sum_v\psi_v\psi_v^*e^{-\tau_i(\mu-E_v)}$, both evaluated on the centroids. On the imaginary axis, $z=i\omega_p$, the kernel becomes $-2\Delta/(\Delta^2+\omega_p^2)$, and at a general complex $z=\omega+i\varpi$ the exponentials take complex times.

**The self-energy kernel.** The correlation self-energy is a sum over states $n$ and poles $j$ of $W^c$,

$$
\Sigma^c_n(\omega)=\sum_{n'j}\frac{M_{nn'j}}{d},\qquad d=\omega-E_{n'}\mp\Omega_j\pm i\eta, \tag{1.3}
$$

on four causal branches (state empty or occupied, $\omega$ above or below $E_F$). A rule $Q(d)=\sum_iw_i\,e^{it_id}\approx1/d$ factorizes the same way, $e^{it(\omega-E-\Omega)}=e^{it\omega}\,e^{-itE}\,e^{-it\Omega}$, so each node is one $G(t_i)$ times one $W^c(t_i)$, at complex time $t_i$. The occupied branch lies in the lower half plane and takes the complex-conjugate rule.

### 1.2 The variables {: #variables }

Every rule is fixed by a handful of numbers, all known before any $G$ or $W$ is built. LORRAX's energies are in Ry and its times in Ry⁻¹.

| Symbol | Meaning | Where LORRAX takes it from | Default |
|---|---|---|---|
| $\Delta_{\min}$ | smallest transition: the gap $E_c^{\min}-E_v^{\max}$ | band energies; floored at the occupation smearing width on a metal | — |
| $\Delta_{\max}$ | largest transition $E_c^{\max}-E_v^{\min}$ | the union of the χ⁰ and Σ band windows | — |
| $R$ | $\Delta_{\max}/\Delta_{\min}$, the dimensionless range of decay rates | derived | — |
| $\varepsilon_\chi$ | absolute error of $1/\Delta$, in Ry⁻¹ | [`minimax_target_error`](../drivers/gw/keywords.md#minimax_target_error) | $10^{-6}$ |
| $N_{\max}$ | node cap of the static and probe rules | [`minimax_max_nodes`](../drivers/gw/keywords.md#minimax_max_nodes) | 64 |
| $\mu$ | energy reference of the exponentials | [`minimax_energy_reference`](../drivers/gw/keywords.md#minimax_energy_reference) | midgap |
| $\omega_p$ | plasmon-pole probe frequency (imaginary for GN, real for HL) | `ppm_omega_p` | 2 Ry (GN) |
| $z=\omega+i\varpi$ | sample frequencies of $W$ (MPA lines, tangential supports) | the sample plan of the screening route | — |
| $\eta$ | broadening of Σ, the distance of every pole from the real axis | [`sigma_regularization_ev`](../drivers/gw/keywords.md#sigma_regularization_ev) | 0.25 eV |
| $\varepsilon_\Sigma$ | Σ rule error, peak-relative ($\eta\lvert Q-1/d\rvert$) near resonance, relative ($\lvert d\rvert\lvert Q-1/d\rvert$) elsewhere | [`sigma_quadrature_eps`](../drivers/gw/keywords.md#sigma_quadrature_eps) | $10^{-4}$ |
| $[E_{\min},E_{\max}]$ | the bandwidth of $G$ in one Σ window, measured from $E_F$ | the states of the window | — |
| $[\Omega_{\min},\Omega_{\max}]$, $[\gamma_{\min},\gamma_{\max}]$ | the bandwidth of $W^c$ in one window: real parts and dampings of its poles | the fitted poles of $W^c$ | — |
| $[\omega_{\min},\omega_{\max}]$ | output frequencies (the read region) | requested quasiparticle states plus margin | about ±10 eV |
| $\beta$ | inverse temperature, for finite-temperature χ⁰ | occupation state | — |
| bank tolerance | relative error of the shared complex-time rules of the tangential route | [`sigma_w_accuracy`](../drivers/gw/keywords.md#sigma_w_accuracy) | $10^{-8}$ (`production`) |

Two things set the node count of every rule: the **dynamic range** of the decay rates or denominators, and the **accuracy**. For a pure decay (imaginary time) the range enters only logarithmically, through $\ln R$. For a near-resonance Σ window, where the terms oscillate and decay only at rate $\eta$, the count is set by the window's narrow side over $\eta$. Neither depends on the number of bands, k points or centroids.

### 1.3 Which constructor {: #which }

| Kernel and domain | Time contour | Constructor | Error measure | LORRAX caller |
|---|---|---|---|---|
| $1/\Delta$ on $[\Delta_{\min},\Delta_{\max}]$ (static χ⁰) | imaginary, $\tau_i>0$ | `serve(family='noncrossing')` | absolute, minimax-optimal | `gw.minimax_screening.solve_laplace_minimax_interval` |
| $\Delta/(\Delta^2+\omega_p^2)$ (GN probe) | imaginary | `response_laplace_rule` at $z=i\omega_p$ | relative, continuum certificate | `gw.minimax_screening.build_imag_probe_response_rule` |
| $\Delta/(\Delta^2+\omega_p^2)$ (imaginary MPA points) | imaginary | `serve(family='noncrossing_imag')` | absolute, measured | `gw.minimax_screening.build_imag_quadrature` |
| $\Delta/(\Delta^2-\omega_p^2)$, real $\omega_p>\Delta_{\max}$ (HL probe) | imaginary, both signs | two `serve(family='noncrossing')` | absolute | `gw.minimax_screening.build_real_quadrature` |
| odd channel $\omega_p/(\Delta^2+\omega_p^2)$ (broken time reversal) | the even rule's times plus up to 16 | `augment_odd_laplace` | absolute, sampled | `gw.minimax_screening.solve_laplace_minimax_imag_interval` |
| $K_z(\Delta)$ on a line $\mathrm{Im}\,z=\varpi$ (MPA) | real, $t_i>0$, positive weights | `damped_line_rule` | relative to $1/\varpi$ | `gw.mpa.model` |
| $K_z(\Delta)$ at many complex $z$, one shared node set (tangential $W$) | complex | `ResponseGroupTree`, `response_group_rules` | relative, sampled | `gw.response_bank` |
| $K_z(\Delta)$ on a window crossing the supports (contour deformation) | imaginary, closed form | `response_imaginary_rule` | relative, sampled | `gw.cd.chi0_rules` |
| χ⁰ at bosonic Matsubara frequencies, finite $T$ | imaginary, $(0,\beta/2]$, KMS-paired | `matsubara_response_rule` | relative per target | `gw.w_isdf` |
| $1/d$ on a Σ denominator box | complex (bent contour or rotated ray) | `analytic_box_rule` | peak-relative or relative, boundary certificate | `gw.sigma_box_plan` |
| $1/d$ on one line $\mathrm{Im}\,d=\eta$ (plasmon-pole Σ, real poles) | real | `analytic_line_box_rule` | peak-relative | `gw.sigma_box_plan` |

A rule is valid only for the kernel, domain and error measure it was built for. Two rules with the same bandwidth but different kernels are not interchangeable.

### 1.4 From bandwidths to nodes {: #recipe }

**χ⁰ at $z=0$.** Given the band energies of the χ⁰ and Σ windows:

1. $\Delta_{\min}=\max(E_c^{\min}-E_v^{\max},\ \sigma_{\rm occ})$, with $\sigma_{\rm occ}$ the occupation smearing width on a metal and $10^{-12}$ otherwise. $\Delta_{\max}=E_c^{\max}-E_v^{\min}$.
2. $R=\Delta_{\max}/\Delta_{\min}$. Ask the service for the scaled rule on $[1,R]$ at error $\varepsilon_\chi\Delta_{\min}$:
   `serve(family='noncrossing', target='inverse', range_value=R, error_bound=eps*x_min, n_max=N_max)`.
3. Rescale: $\tau_i=\hat\tau_i/\Delta_{\min}$, $\alpha_i=\hat w_i/\Delta_{\min}$, physical error $=\hat\varepsilon/\Delta_{\min}$.
4. If the rule has $N_{\max}$ nodes and still misses the target, refuse (`GATE minimax_node_cap`).

`gw.minimax_screening.solve_laplace_minimax_interval(x_min, x_max, target_error=, max_nodes=)` does steps 2–4 and returns a `LaplaceMinimaxQuadrature` in Ry. The node count grows like $\ln R\cdot\ln(1/\varepsilon)$; the estimate $N\approx\ln(16R)\ln(1/\varepsilon)/\pi^2$ is within a few nodes of the levelled rule.

**A Σ window.** Given the window's states, poles and output frequencies, on the empty-state branch ($d=\omega-E-\Omega+i\eta$):

1. Real extent of the denominators, from the eight corners: $\mathrm{Re}\,d\in[\omega_{\min}-E_{\max}-\Omega_{\max},\ \omega_{\max}-E_{\min}-\Omega_{\min}]$.
2. Pad by 2 % of the width (at least $0.02\,\eta$) without moving an edge more than 30 % of the way toward zero, so a sign-definite window stays sign-definite.
3. Imaginary extent: $\mathrm{Im}\,d\in[\eta+\gamma_{\min},\ \eta+\gamma_{\max}]$.
4. `analytic_box_rule((re_lo, re_hi, im_lo, im_hi), eps_sigma)` returns a `UniformRule`. If its `sup_error` exceeds $\varepsilon_\Sigma$, the rule is uncertified and the caller must refuse.

`gw.sigma_box_plan._box_for_window` and `_box` do steps 1–3 in LORRAX. A box whose real extent contains zero takes the bent-contour rule, whose count follows the narrow side of the box (the part of $\mathrm{Re}\,d$ on the short side of zero) over $\eta$. A sign-definite box takes the sector rule, whose count grows like the logarithm of its radial extent. For comparison, a uniform time grid covering a bandwidth $B$ needs $N\approx(B/\eta)\ln(1/\varepsilon)/2\pi$ nodes ([Time–frequency transforms (2.1)](../theory/time-frequency.md#sampling-count)).

**χ⁰ on a damped line (MPA).** For samples $z=\omega+i\varpi$ on one line, `damped_line_rule(varpi, freq_max, rel_tol=)` with `freq_max` the largest angular frequency in the integrand, $\max\lvert\omega\rvert+\Delta_{\max}$ (not $\Delta_{\max}$ alone, because $e^{i\omega t}\sin\Delta t$ beats at $\omega\pm\Delta$). The dimensionless bandwidth `a_dim = freq_max/varpi` sets the cost.

### 1.5 Worked examples {: #examples }

Measured on one CPU with the service's own constructors. Energies in eV are converted to Ry for the call.

| Request | Inputs | Nodes | Achieved error | Time |
|---|---|---|---|---|
| static χ⁰, `noncrossing` | gap 2 eV, $\Delta_{\max}$ 100 eV ($R=50$), $\varepsilon_\chi=10^{-6}$ Ry⁻¹ | 9 | $5.8\times10^{-7}$ | 19 ms |
| static χ⁰, `noncrossing` | gap 0.5 eV, $\Delta_{\max}$ 100 eV ($R=200$), $10^{-6}$ | 12 | $4.9\times10^{-7}$; $\tau\in[0.074,211]$ Ry⁻¹ | 0.1 s |
| static χ⁰, `noncrossing` | same, $10^{-8}$ | 15 | $6.5\times10^{-9}$ | 35 ms |
| static χ⁰, `noncrossing` | gap 0.5 eV, $\Delta_{\max}$ 400 eV ($R=800$), $10^{-6}$ | 14 | $4.6\times10^{-7}$ | 32 ms |
| static χ⁰, `noncrossing` | gap 0.05 eV, $\Delta_{\max}$ 100 eV ($R=2000$), $10^{-6}$ | 17 | $6.7\times10^{-7}$ | 0.1 s |
| GN probe, `noncrossing_imag` | gap 0.5 eV, $\Delta_{\max}$ 100 eV, $\omega_p=2$ Ry, $10^{-6}$ | 7 | $5.4\times10^{-7}$ | 1.7 s |
| GN probe, `response_laplace_rule` | same window, $z=2i$ Ry, `rel_tol` $10^{-6}$ | 27 | certificate PASS | 0.3 s |
| Σ, crossing box | $\mathrm{Re}\,d$ from −21 to +9 eV, $\eta=0.25$ eV, $\varepsilon_\Sigma=10^{-4}$ | 110 | $4.0\times10^{-5}$ peak-relative | 0.5 s |
| Σ, crossing box | same, $\eta=0.1$ eV | 260 | — | 1.6 s |
| Σ, crossing box | same, $\eta=1$ eV | 37 | — | < 0.1 s |
| Σ, crossing box | $\mathrm{Re}\,d$ from −357 to +16 eV, $\eta=0.25$ eV | 192 | $4.4\times10^{-5}$ | 1.3 s |
| Σ, sign-definite box | $\mathrm{Re}\,d$ from −112 to −1 eV, $\eta=0.25$ eV | 8 (ray at 88°) | $8.3\times10^{-5}$ relative | < 0.1 s |
| Σ, line rule | $\mathrm{Re}\,d$ from −21 to +9 eV on $\mathrm{Im}\,d=0.25$ eV | 336 | — | — |
| MPA line, `damped_line_rule` | $\varpi=0.2$ Ry, `freq_max` $=2\times100$ eV | 519 in 14 panels, $t\le96$ Ry⁻¹ | `rel_tol` $10^{-8}$ | — |
| Matsubara | $T=300$ K ($\beta=526$ Ry⁻¹), $\Delta_{\max}=100$ eV, $k\in\{0,1,2,4,8\}$ | 30 | `rel_tol` $10^{-8}$ | 2.3 s |

What the table shows:

- **The static rule is cheap and nearly insensitive to the bandwidth.** Raising $\Delta_{\max}$ fourfold adds two nodes; each factor of 100 in accuracy adds three to four.
- **The Σ crossing rule scales as $1/\eta$ through the narrow side.** With a 9 eV narrow side the count is 37, 110 and 260 at η = 1, 0.25 and 0.1 eV, about three nodes per η of narrow side. Widening the wide side from 21 to 357 eV adds only 82 nodes. A uniform grid on the 373 eV branch at $\eta=0.25$ eV would need about 2200.
- **Windows that keep one sign are nearly free.** 8 nodes cover a sign-definite box spanning two decades.

```python
import minimax

RY = 13.605693122994                     # eV per Ry
x_min, x_max = 0.5 / RY, 100.0 / RY      # gap and largest transition, Ry
q = minimax.serve(family="noncrossing", target="inverse",
                  range_value=x_max / x_min,
                  error_bound=1e-6 * x_min,  # physical 1e-6 / Ry, scaled
                  n_max=64)
tau = q.nodes / x_min                    # Ry^-1
alpha = q.weights / x_min                # 1/x ~ sum(alpha * exp(-tau * x))
err = q.max_error / x_min                # Ry^-1

eta = 0.25 / RY
rule = minimax.analytic_box_rule((-21 / RY, 9 / RY, eta, eta), 1e-4)
print(rule.one_line())                   # box rule: 110 nodes, ray 0 deg, ...
# 1/d ~ sum(rule.weights * exp(1j * rule.times * d)) on the box
```

## 2 API reference {: #api }

All names below are top-level attributes of `minimax`. Names marked *lazy* are resolved on first access (PEP 562), so `import minimax` loads only NumPy.

### 2.1 Vocabulary {: #api-vocabulary }

| Name | What it is |
|---|---|
| `TARGETS: Mapping[str, TargetSpec]` | the target functions: `inverse` ($1/x$ on $[1,R]$), `inverse_imag` ($x/(x^2+\hat\omega^2)$), `hgl` and `fermi` (regularized sign targets on $[0,A]$), `complex_laplace` ($1/(u-i\beta)$), `damped_line` (the MPA kernel on the strip) |
| `FAMILIES: Mapping[str, FamilySpec]` | the serving families, one per cell of the sampling table (below), with `wired` saying whether `serve` can solve it |
| `CHARACTERS` | `('static', 'real', 'imag', 'strip')`: whether the sample frequency $z=\omega+i\varpi$ has $\omega$, $\varpi$ zero or not |
| `TargetSpec(name, definition, version, domain)` | frozen record; `version` bumps when the definition changes |
| `FamilySpec(name, route, target, character, range_param, wired, description)` | frozen record; `range_param` is `'R'` or `'A_dim'` |
| `families_for_character(character)` | every family serving a character, wired or not |
| `family_for_character(character) -> str` | the wired family for a character; raises `SamplingUnsupported` on `'strip'` |

| | $\omega=0$ | $\omega\ne0$ |
|---|---|---|
| $\varpi=0$ | `noncrossing` ($1/x$) | `crossing` (sine sum) |
| $\varpi\ne0$ | `noncrossing_imag` (wired), `complex_laplace` (not wired) | `damped_line` (not wired; MPA uses the `damped_*` constructors) |

### 2.2 Served rules {: #api-serve }

```python
serve(*, family, target, range_value, error_bound, n_max,
      eps_q=None, omega_hat=None) -> Quadrature
```

Solves one rule in process, in scaled units, and caches it per process (an `lru_cache` keyed on rounded inputs).

| Argument | Meaning |
|---|---|
| `family`, `target` | `('noncrossing', 'inverse')`, `('noncrossing_imag', 'inverse_imag')` or `('crossing', 'hgl' \| 'fermi')` |
| `range_value` | $R=x_{\max}/x_{\min}$ for the two Laplace families; `A_dim`, a bandwidth over the regularization width, for `crossing` |
| `error_bound` | sup-norm absolute error on the scaled interval. For a physical tolerance $\varepsilon$ on $[x_{\min},x_{\max}]$, pass $\varepsilon x_{\min}$ |
| `n_max` | node cap. If $N_{\max}$ nodes miss the bound, the $N_{\max}$ rule comes back with its larger `max_error`, and the caller decides |
| `eps_q` | `crossing` only: tail cut of the sine weight (default $10^{-3}$); the time support is $\sqrt{2\ln(1/\varepsilon_q)}$ for `hgl`, $\tfrac12\ln(1/\varepsilon_q)$ for `fermi` |
| `omega_hat` | `noncrossing_imag` only, required: $\omega_p/x_{\min}$ |

`noncrossing` is solved by `noncrossing_levelled`, the minimax-optimal (Remez) rule certified by alternation, in 20–130 ms. `noncrossing_imag` and `crossing` return the smallest rule whose error, measured on a grid, meets the bound. Each distinct request is announced once per process as a `RuntimeWarning` with its node count, error, $\sum\lvert w\rvert$, $\kappa_0$ and provenance; `reset_announcements()` clears that memory.

`solve_uncertified(...)` takes the same arguments and is what `serve` calls after validating the request.

```python
@dataclass(frozen=True)
class Quadrature:
    nodes: np.ndarray        # float64, scaled tau
    weights: np.ndarray      # float64 (complex128 for strip families)
    family: str; target: str
    range_param: str; range_value: float
    error_bound: float       # what was asked
    max_error: float         # what was measured
    kappa0: float | None     # amplification: max_u u * sum |w| exp(-tau u)
    kappa1: float | None     # reserved
    provenance: Provenance
    node_count: int          # property
```

`Provenance(source, table_hash, generator_commit, generation_backend, certified)` records where the rule came from. `source` is always `'runtime-uncertified'` and `certified` is `False`, because every rule is solved at run time; `table_hash` is a SHA-256 of the node and weight bytes, and `one_line()` is what the GW log prints. `runtime_provenance(tau, w)` builds one. `noncrossing_kappa0(tau, weights, range_value, n_eval=512)` evaluates $\kappa_0=\max_{u\in[1,R]}u\sum_l\lvert w_l\rvert e^{-\tau_lu}$, which is 1 for a positive rule and larger when weights of both signs cancel.

**Levelled rule.** `noncrossing_levelled(R, eps, N_max=64) -> (tau, weights, N, err)` returns the smallest-$N$ rule with sup error at most `eps` on $[1,R]$, or the $N_{\max}$ rule and its error. `certify_noncrossing(tau, weights, R, M=None, iters=40) -> (lower, upper)` brackets a rule's error: the alternation lower bound on the best error at that $N$, and the refined dense sup.

**Solver half** (*lazy*, SciPy): `G_hgl`, `G_fermi` (the crossing targets), `tau_max_hgl`, `tau_max_fermi`, `noncrossing_imag_grids`, `crossing_grids`, `solve_noncrossing_imag`, `solve_crossing`. These are what `serve` runs; a caller normally does not need them.

### 2.3 Σ denominator rules {: #api-box }

```python
analytic_box_rule(box, eps, *, mass_cap=83.0, relative=None) -> UniformRule   # lazy
```

The rule for $1/d$ on a box `(re_lo, re_hi, im_lo, im_hi)` with $0<$ `im_lo` $\le$ `im_hi`, in one energy unit (LORRAX passes Ry). The result satisfies $1/d\approx\sum_kw_k\,e^{it_kd}$ with complex times in the inverse unit. No node is optimized: every node is a formula of the box and `eps`, and the weights are one linear least-squares solve with a ridge of $0.05\,\varepsilon$.

- **Crossing box** (real extent contains 0): the bent contour, a line at depth $c/(m\eta)$ below the real time axis with $c=4$ and $m$ the narrow side in units of $\eta=$ `im_lo`, plus a vertical leg and two Gauss sets on the imaginary time axes. `eps` is peak-relative, $\eta\lvert Q-1/d\rvert$. A narrow side under $4\eta$ is built as $4\eta$.
- **Sign-definite box**: the sector rule, log-spaced times on a rotated ray, with the degree from a local extremal-length law. `eps` is relative, $\lvert d\rvert\lvert Q-1/d\rvert$.
- **Acceptance**: the boundary sup is at most `eps` and the term mass $\rho\sum_k\lvert w_ke^{it_kd}\rvert$ at most `mass_cap` (83 is the executor's round-off budget, $5\times10^{-6}/6\times10^{-8}$). A miss climbs a fixed ladder of six rungs. When no rung passes, the last one is returned with `sup_error > eps`, and the caller must refuse.
- `relative` overrides the currency; `None` picks relative exactly for sign-definite boxes.

The construction runs under a BLAS thread pin of $\min(16,\ \text{CPUs in the affinity mask})$, so a rule is reproducible on one machine class.

`UniformRule` fields: `times`, `weights`, `box`, `eps`, `relative`, `theta_deg` (the ray), `rank` (the count law's answer), `sup_error`, `kappa_max` (cancellation ratio $\sum\lvert w_ke^{it_kd}\rvert/\lvert Q\rvert$), `seconds`; properties `node_count` and `one_line()`.

| Helper | Returns |
|---|---|
| `crossing_nodes(box, eps, rung=0, bend=None)` | the bent-contour nodes of a crossing box (first element: the times) |
| `sector_degree(box, eps, rung=0)` | the sector rule's degree (first element) |
| `boundary_samples(box, theta_deg, horizon, eps, p=6.0)` | the boundary cloud on which a rule's error and noise attain their box maximum |
| `rule_sup_error(times, weights, d, rho=None)` | `(max rho·|Q(d)−1/d|, max kappa)` on samples `d`; `rho` defaults to `min Im d` |
| `rule_roundoff_amplification(times, weights, d, rho)` | $\max_d\rho(d)\sum_k\lvert w_ke^{it_kd}\rvert$, the worst term mass |

```python
analytic_line_box_rule(box, eps) -> UniformRule   # lazy
```

The rule for a crossing box of zero height, `im_lo == im_hi`, as on the plasmon-pole route where every pole of $W$ is real and $\mathrm{Im}\,d=\eta$ exactly. It is a composite Gauss–Legendre rule in real time with the damping $e^{-\eta t}$ removed from the weights, because the executor's $e^{itd}$ already carries it. Rectangles and sign-definite tails take `analytic_box_rule`.

### 2.4 Response rules {: #api-response }

All take frequencies and transition intervals in Ry and return times in Ry⁻¹. The sample frequencies `z_ry` must lie in the upper half plane, $\mathrm{Im}\,z>0$, and `rel_tol` in $[10^{-13},0.1)$; otherwise `ValueError`. Derivatives are with respect to $s=z^2$.

```python
response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, *, rel_tol=1e-8,
                      previous=None, domain_pad_ry=0.0, ordered=False,
                      reference_ry=None) -> dict   # lazy
```

Positive real times for samples whose real part lies outside the transition interval, from elliptic decay rates and a time-moment Ritz solve; the weights are one linear projection onto the even target $\Delta/(\Delta^2-z^2)$ and its $s$-derivative (and, with `ordered=True`, the odd target $z/(\Delta^2-z^2)$). A continuum residual bound certifies the rounded arrays; the degree grows from $\lceil\ln(16R)\ln(1/\varepsilon)/\pi^2\rceil$ until it passes, up to 64. Returns `t`, `projection_value`, `projection_derivative` (and `odd_projection_value` when ordered), `reference_ry`, `certificate`, `node_digest`, `reuse_status`, `reuse_reason`. The projections include $e^{-\mathrm{ref}\cdot t}$, and the consumer supplies $e^{-(\Delta-\mathrm{ref})t}$. `previous` is an earlier rule to reuse when its domain still contains the request.

```python
response_group_rules(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
                     decay_rate=0., pencil=None, panels=False,
                     capacity=None, max_kappa=None) -> list[dict]   # lazy
ResponseGroupTree(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
                  decay_rate=0., ...)                                # lazy
```

The tangential route samples $W$ at many complex supports and wants one node set for all of them, so that each node is one Green's-function product serving every sample. A stacked Hankel shift pencil proposes complex times shared by the forward ($z$) and reverse ($-\bar z$) poles of a group of samples, and linear projection fits $1/(\Delta-z)$ and its derivative on them. A group whose pencil fails takes a closed-form set on the imaginary time axis when it fits the capacity, and is otherwise halved, down to single samples. Each rule has `members`, `t` (padded to `RESPONSE_NODE_CAPACITY` = 384; one pencil holds `RESPONSE_RULE_CAPACITY` = 192), `value` and `derivative` of shape `[members, 2, 384]` (forward, reverse), `count`, `sampled_error`, `coefficient_mass` and `reference_ry`; zero coefficients mark unused slots. Errors are sampled, not proven. `ResponseGroupTree` exposes the same build one tree node at a time (`children`, `tried`, `attempt`, `pending`, `rules`), so that processes can evaluate attempts in parallel and `rules(attempts)` reproduces the bytes of `response_group_rules`.

`response_imaginary_rule(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, node_capacity=384, max_kappa=...)` is that closed-form set alone: times $t=is$, Gauss–Legendre in $s\in[0,\mathrm{horizon}/\eta]$, $n=\lceil(\mathrm{hi}-\mathrm{lo})/\eta\cdot\mathrm{horizon}/4\rceil$, for samples whose poles sit inside the interval. It returns `None` when $n$ exceeds the capacity or a pole fails.

`response_bank_rule(z_ry, delta_max_ry, ...)` (positive real-time Hermite panels) has no production caller. `RESPONSE_MAX_KAPPA` is the coefficient-mass gate.

### 2.5 Damped-line rules (MPA) {: #api-damped }

```python
damped_line_rule(varpi, freq_max, *, rel_tol=DEFAULT_DAMPED_REL_TOL,
                 wavelengths_per_panel=DEFAULT_WAVELENGTHS_PER_PANEL,
                 max_order=256) -> dict   # lazy
```

A composite Gauss–Legendre rule, with strictly positive weights, for the causal positive-time integral of the MPA kernel $K_z(\Delta)=-2\int_0^\infty dt\,e^{-\varpi t}e^{i\omega t}\sin\Delta t$ on the line $\mathrm{Im}\,z=\varpi$. `varpi` must be positive (at $\varpi=0$ the integral does not converge absolutely). `freq_max` is $\max\lvert\omega\rvert+\Delta_{\max}$. `rel_tol` is relative to $1/\varpi$, the kernel's peak, split half to truncation and half to the panels. Returns `t`, `h` (nodes and weights), `t_max`, `n_panels`, `orders` (per panel), `a_dim` $=$ `freq_max/varpi`, and `kappa0` $=\varpi\sum_lh_le^{-\varpi t_l}\le1$. Defaults: `DEFAULT_DAMPED_REL_TOL` $=10^{-8}$, `DEFAULT_WAVELENGTHS_PER_PANEL` $=16$ (the measured knee of cost against panel width).

`damped_rectangle_rule`, `damped_rectangle_gauss_rule` and `damped_rectangle_positive_rule(gamma_min, gamma_max, freq_max, *, rel_tol, ...)` cover a rectangle of dampings $\gamma_{\min}\le-\mathrm{Im}\,d\le\gamma_{\max}$, $\lvert\mathrm{Re}\,d\rvert\le$ `freq_max`, by panelled or global Gauss–Legendre; the `positive` variant scores both and returns the smaller passing rule.

### 2.6 Other constructors {: #api-other }

| Constructor | Kernel | Returns |
|---|---|---|
| `augment_odd_laplace(times, x_min, x_max, omega, *, tolerance, max_extra=16, grid_size=4096, candidates=48)` | odd channel $\omega/(x^2+\omega^2)$ on the even rule's times, greedily adding at most 16 | `(times, odd_weights, added_count, sampled_max_error)`; refuses if the absolute `tolerance` is missed |
| `matsubara_response_rule(beta_ry_inv, delta_max_ry, n_indices, *, rel_tol=1e-8)` | $\chi(i\nu_k)\approx\sum_l W_{kl}h(t_l)+\overline{W_{kl}}h(\beta-t_l)$, $\nu_k=2\pi k/\beta$, for any Fermi–Dirac (KMS-bounded) τ correlation | dict with `t` in $(0,\beta/2]$, `weights` $[n_\nu,L]$ complex, `nu_ry`, `certificate`, `node_digest` |
| `fit_damped_reciprocal(rectangles, *, target_error, max_rank=96, ...)` | one positive rule $1/d\approx\sum w\,e^{-dt}$ for $d=x-i\gamma$ over rectangles `(x_min, x_max, gamma_min, gamma_max)`, $x_{\min}>0$ | `DampedReciprocalFit`; error measured on an independent grid |
| `positive_reciprocal(R, tolerance, ...)` | $1/x$ on $[1,R]$, positive weights, high-precision audit | rule with `times`, `evaluate(x)` |
| `odd_reciprocal(A, tolerance, ...)` | $1/x$ on $[-A,-1]\cup[1,A]$ as a sine sum, exact-arithmetic bound | same |
| `damped_line_reciprocal(u_max, tolerance)` | $1/(u+i)$ on $\lvert u\rvert\le u_{\max}$; for $1/(x+i\eta)$ to absolute $\varepsilon$ pass `(span/eta, eta*eps)` and call `rule.rescaled(eta)` | same |

`fit_damped_reciprocal` and the three analytic reciprocal constructors have no production caller; they remain public for experiments.

### 2.7 Refusals {: #api-refusals }

All are `MinimaxRefusal`, a subclass of `RuntimeError`.

| Exception | Raised when |
|---|---|
| `UnknownTarget` | a family, target or keyword outside the vocabulary (including the retired `use_shipped`), or `noncrossing_imag` without `omega_hat` |
| `UncertifiedSolveRefused` | a declared family with no in-process solver: `complex_laplace`, `damped_line` |
| `SamplingUnsupported` | `family_for_character('strip')`, or a character outside `CHARACTERS` |

Constructors raise `ValueError` for invalid domains (non-positive heights, a box with `im_lo <= 0`, a tolerance out of range).

## 3 Invariants and gotchas {: #gotchas }

- **Scaled units in, scaled units out.** `serve` works on $[1,R]$ or $[0,A]$. The caller rescales times and weights by $1/x_{\min}$ and asks for $\varepsilon x_{\min}$, not $\varepsilon$. Asking for $\varepsilon$ directly gives a rule $1/x_{\min}$ times too loose (or too tight).
- **`serve` does not raise when the cap is hit.** It returns the $N_{\max}$ rule; compare `max_error` with `error_bound` yourself.
- **The error measure is part of the rule.** Static χ⁰ is absolute in Ry⁻¹; the GN probe and response rules are relative; Σ windows are peak-relative near resonance and relative elsewhere. An error bound in one currency says nothing about another.
- **Use the right $\Delta_{\min}$ on a metal.** The gap is zero or negative; floor it at the smearing width, because $(f_v-f_c)$ vanishes linearly below it. With $x_{\min}=10^{-12}$ the solver runs its whole ladder for minutes and fails.
- **`freq_max` of a damped line includes $\lvert\omega\rvert$.** Sizing it on $\Delta_{\max}$ alone under-resolves every sample with a real part.
- **Rules are rebuilt every run.** Nothing is stored across processes; a repeated request within one process is served from cache. Node positions can differ in the last digits between hosts because the solve goes through the local LAPACK, so compare rules by node count and error, not bytes.
- **Thread count matters for the complex-time builders.** `analytic_box_rule` and the response-group rules pin BLAS threads; a different count can pick different times.
- **Σ: one $\eta$ and one $\varepsilon_\Sigma$ per run.** A rule depends only on its box and $\varepsilon_\Sigma$, so within a run it serves every later window whose box it contains; a self-consistent run that changes either refuses.
- **The occupied branch of Σ uses the conjugate rule**, because its box lies in the lower half plane.

## 4 Where LORRAX uses it {: #uses }

| File | Use |
|---|---|
| `src/gw/minimax_screening.py` | rescales `serve` rules into Ry: the static χ⁰ rule (`solve_laplace_minimax_interval`, `build_static_quadrature`), the imaginary probe (`build_imag_quadrature`, `build_imag_probe_response_rule`), the HL real probe (`build_real_quadrature`) and the odd channel |
| `src/gw/screening.py` | chooses between the GN and HL probe builders |
| `src/gw/sigma_box_plan.py` | builds every Σ window's box and its `analytic_box_rule` or `analytic_line_box_rule`, and refuses uncertified rules |
| `src/gw/response_bank.py` | shared complex-time rules of the tangential route through `ResponseGroupTree` |
| `src/gw/cd/chi0_rules.py` | contour-deformation χ⁰ windows through `response_imaginary_rule` |
| `src/gw/mpa/model.py` | MPA line rules through `damped_line_rule` |
| `src/gw/w_isdf.py` | finite-temperature χ⁰ through `matsubara_response_rule` |
| `src/gw/minimax_config.py` | `MinimaxConfig(target_error=1e-6, max_nodes=64, energy_reference='midgap', ...)`, the settings shared by the χ⁰ consumers |

The time-domain products these rules feed are described on [Kernels](kernels.md).
