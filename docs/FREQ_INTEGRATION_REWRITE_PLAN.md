# Frequency Integration Engine: Design & Implementation Guide

Status: Implementation blueprint
Audience: developers and agents working on the GW/CTSP codebase

---

## 1. Why We're Rewriting

`w_isdf_dynamic.py` currently has four nearly-identical JIT kernels (windowed-static,
HGL, GL-omega, GL-reverse) that share ~90% of their code. Each `(max_val_len,
max_cond_len)` pair triggers a separate compilation. Only scalar-omega evaluation is
supported — no batched multi-frequency. There is no sigma(omega) path at all.

**Goal:** One engine that computes both chi(omega) and Sigma(omega), with batched
frequencies, from two inputs that have the same mathematical structure.

The rewritten code must be:

1. **Physics-explicit** — equations appear as comments next to the code that implements them.
2. **Sharding-aware** — all shardings defined in one place, not scattered through kernels.
3. **Short and non-redundant** — each formula appears exactly once.
4. **Readable as a standalone algorithm document.**

---

## 2. What We're Computing

### 2.1 chi(omega): polarizability as a product of two Green's functions

The RPA polarizability sums over valence-conduction transitions:

```
chi(omega) = Sum_{cv} A_{cv} [ 1/(omega - Delta_{cv}) - 1/(omega + Delta_{cv}) ]
```

where `Delta_{cv} = E_c - E_v > 0`. Each transition contributes two poles:
a **resonant** pole at `omega = +Delta_{cv}` and an **antiresonant** pole at
`omega = -Delta_{cv}`.

In the time domain this is equivalent to:

```
chi(omega) = int dt exp(i omega t) G_c(t) G_v(t)
```

where G_v and G_c are the valence and conduction Green's functions — sums of poles
at band energies with wavefunction residues.

### 2.2 Sigma(omega): self-energy as G times W

The screened interaction W is represented by a **plasmon decomposition** — a set of
poles `{Omega_jq}` with matrix residues `{B_jq(r_mu)}`:

```
W_q(omega; mu, nu) = Sum_j B_jq(mu) B*_jq(nu) [ 1/(omega - Omega_jq) - 1/(omega + Omega_jq) ]
```

This has **exactly the same mathematical structure** as chi: a sum of resonant and
antiresonant poles with rank-1 matrix residues. For the correlation part, use
`W_c` (not bare `v`). The self-energy channel is:

```
Sigma(omega) = int dt exp(i omega t) G(t) W_c(t)
```

and is evaluated with the same CTSP machinery (GL for fixed-sign branches, HGL in
crossing branches).

For sign bookkeeping, the two sigma channels map to:

```
Sigma^(+)(omega) ~ Sum_{p,v} (...) / (omega - E_v + Omega_p)
Sigma^(-)(omega) ~ Sum_{p,c} (...) / (omega - E_c - Omega_p)
```

which is the same as `(valence wfns, plasmons +)` and
`(conduction wfns, plasmons -)`.

### 2.3 The unifying insight: both are PoleBlock x PoleBlock

In both cases we are computing:

```
F(omega) = int dt exp(i omega t) A(t) B(t)
```

where A and B are each sums of poles with matrix residues. The only difference is
what the poles and residues represent:

| Quantity | Factor A | Factor B |
|----------|----------|----------|
| chi(omega) | G_v: poles at E_{vk}, residues psi_v psi_v† | G_c: poles at E_{ck}, residues psi_c psi_c† |
| Sigma(omega), valence channel | G_v: poles at E_{vk}, residues psi_v psi_v† | W(+): poles at +Omega_{jq}, residues B_j B_j† |
| Sigma(omega), conduction channel | G_c: poles at E_{ck}, residues psi_c psi_c† | W(-): poles at -Omega_{jq}, residues B_j B_j† |

**The engine doesn't need to know chi vs sigma.** It takes two PoleBlocks and
integrates their product. For sigma in this code path, call it twice and sum:
`(valence wfns, plasmons +)` and `(conduction wfns, plasmons -)`.

---

## 3. The CTSP Algorithm

This section explains the Complex-Time Shredded Propagator (CTSP) method from
Kim, Martyna & Ismail-Beigi (PRB 101, 035139, 2020). Full derivations are in
`docs/chi_omega_quadrature.md`.

### 3.1 The Laplace trick

The key idea: convert a fixed-sign denominator into a Laplace integral:

```
x > 0:  1/x = z_lm * integral_0^inf d_tau exp(-z_lm * x * tau)
x < 0:  1/x = -z_lm * integral_0^inf d_tau exp(-z_lm * |x| * tau)
```

where `z_lm = 1/sqrt(E_gap * E_bw)` is a scaling parameter that equalizes the
quadrature error at both edges of the energy window. This integral is approximated
by **Gauss-Laguerre (GL) quadrature**:

For each branch where `sign(x)` is constant, this becomes:

```
1/x ≈ sign(x) * z_lm * Sum_u w_u exp(-tau_u * (z_lm * |x| - 1))
```

The number of quadrature points scales as `N_tau = alpha * (0.4 - 0.3 * ln(epsq))`
where `alpha = sqrt(E_bw / E_gap)` is the condition number of the energy window.

**Why this helps:** instead of summing over all transitions directly (expensive,
singular near poles), we loop over ~10-30 quadrature points in tau. At each tau, we
build damped propagators, multiply them, and FFT. The tau loop replaces the
transition sum.

### 3.2 Denominator types: when GL works and when it doesn't

The denominator `x = omega - Delta_{cv}` can be positive, negative, or change sign
across transitions within a window. GL quadrature requires `x` to have fixed sign.

For a given energy window with gap `E_gap` and bandwidth `E_bw`:

| Denominator type | Condition | Sign of x | Quadrature |
|------------------|-----------|-----------|------------|
| **antiresonant** | `x = omega + Delta` | always positive (omega >= 0) | GL |
| **resonant_below** | `omega < E_gap` | all negative (`Delta > omega`) | GL |
| **crossing** | `E_gap <= omega <= E_bw` | mixed (some positive, some negative) | HGL |
| **resonant_above** | `omega > E_bw` | all positive (`omega > Delta`) | GL |

Each of these is a **denominator type** (`denom_type`). The engine evaluates one
denom_type at a time, with its own quadrature nodes and omega mask.

### 3.3 HGL quadrature for crossings

When `x` changes sign inside a window/frequency branch, GL fails. The
**Hermite-Gauss-Laguerre (HGL)** quadrature uses a regularized oscillatory kernel
`h(tau) = exp(-tau - tau^2/2)`:

```
F(x) = gamma * integral_0^inf sin(gamma * x * tau) * h(tau) d_tau
```

which satisfies `F(x) -> 1/x + O(1/x^5)` as `|x| -> infinity` (converges faster
than a Lorentzian's `O(1/x^3)`). The ratio 1/2 in the exponent `tau^2/2` is
special — any other ratio gives only `O(1/x^3)`.

Grid sizing for HGL is a quadratic in `gamma * E_bw` (see `chi_omega_quadrature.md`
Section 3 and Appendix D).

### 3.4 Euler identity: 2x memory savings for HGL

Instead of storing 4 real arrays (sin/cos for valence and conduction separately),
use 2 complex propagators via Euler's identity:

```
G^v_k(tau) = Sum_v exp(i * gamma * tau * E_v) psi_v psi_v†
G^c_k(tau) = Sum_c exp(i * gamma * tau * E_c) psi_c psi_c†
```

After FFT to R-space, the Hermitian product gives both tiles at once:

```
(G^c_{-R})† · G^v_R = P_plus - i * P_cross
```

where `P_plus = Re[...]` and `P_cross = -Im[...]` are the two real tiles needed
for the HGL accumulation formula:

```
chi_cross(omega) = -gamma * Sum_u w_u [ cos(gamma*tau_u*omega) * P_cross(tau_u)
                                       - sin(gamma*tau_u*omega) * P_plus(tau_u) ]
```

### 3.5 Energy windows: why we partition bands

A single energy window covering all transitions would have a huge condition number
`alpha = sqrt(E_bw / E_gap)`, requiring many quadrature points. Partitioning bands
into narrower energy windows keeps alpha bounded, so each window needs only ~10-30
GL points.

The existing windowing optimizer in `get_windows.py` (`minimize_cost_fn`) handles
this — it finds the partition that minimizes total quadrature cost across all window
pairs.

**Important:** The engine must accept windows determined externally, since chi
(G*G integration) and sigma (G*W integration) will generally need **different**
window partitions. The sigma windows must account for plasmon pole energies in
addition to band energies. The windowing code in `get_windows.py` should be extended
to handle both cases, but the integration engine itself is window-agnostic — it just
receives a list of `(PoleBlock_A, PoleBlock_B, WindowPair)` tuples.

---

## 4. What Happens at Each Quadrature Point

This is the inner loop of the engine. At each tau_u:

### 4.1 GL case (exponential damping)

```
1. Weight each pole using branch-local edge shifts (chi example):
     antiresonant / resonant_below:
       damped_v[k,n] = exp(-z_lm * tau * (E_v,max - E_v[k,n])) * mask_v[k,n]
       damped_c[k,n] = exp(-z_lm * tau * (E_c[k,n] - E_c,min)) * mask_c[k,n]
     resonant_above (reversed edges):
       damped_v[k,n] = exp(-z_lm * tau * (E_v[k,n] - E_v,min)) * mask_v[k,n]
       damped_c[k,n] = exp(-z_lm * tau * (E_c,max - E_c[k,n])) * mask_c[k,n]

2. Build propagators via einsum on the two memory layouts:
     G_A_k[k, a, mu, b, nu] = Sum_n damped_A[k,n] * psi_X[k,a,mu,n] * psi_Y[k,n,b,nu]
     G_B_k[k, a, mu, b, nu] = Sum_n damped_B[k,n] * psi_X[k,a,mu,n] * psi_Y[k,n,b,nu]

3. Reshape to (a, mu, b, nu, kx, ky, kz), FFT k -> R:
     G_A_R  = ifftn(G_A_k, axes=(-3,-2,-1), norm='ortho')
     G_B_mR = fftn(G_B_k, axes=(-3,-2,-1), norm='ortho')    [note: fftn gives -R]

4. Contract spin indices (spin trace):
     integrand_R[mu, nu, Rx, Ry, Rz] = Sum_{a,b} G_B_mR[a,mu,b,nu,...] * G_A_R[b,nu,a,mu,...]

5. FFT R -> q:
     integrand_q = fftn(integrand_R, axes=(-3,-2,-1), norm='ortho')
```

**This is the IntegrandTau** — the omega-independent matrix contribution at one
quadrature point. Omega enters only through scalar coefficients applied after tile
construction.

### 4.2 HGL case (oscillatory phases)

Same steps, except:
- Step 1 uses complex phases `exp(i * gamma * tau * E)` instead of real exponentials.
- Step 4 produces a complex product; extract `P_plus = Re[...]` and `P_cross = -Im[...]`.
- Returns two real IntegrandTau arrays instead of one.

### 4.3 `build_integrand_tau`: the central kernel

```python
build_integrand_tau(
    denom_type: DenomType,
    block_A: PoleBlock,           # "left" factor (valence for chi, G for sigma)
    block_B: PoleBlock,           # "right" factor (conduction for chi, W for sigma)
    tau_u: jax.Array,             # scalar quadrature node
    nkx: int, nky: int, nkz: int,
) -> IntegrandTau
```

This is the **only** function that builds propagators and contracts them. It doesn't
know whether it's computing chi or sigma — it just takes two PoleBlocks and produces
their contracted product at one tau point.

For GL: returns one real (mu, nu, qx, qy, qz) array.
For HGL: returns `(P_plus, P_cross)`, two real arrays of the same shape.

---

## 5. Frequency Batching and the Scan Loop

### 5.1 Why batching works

The IntegrandTau is **omega-independent**. Omega enters only through scalar
coefficients:

- GL: `coeff_GL(omega, tau) = sign * z_lm * w_u * exp(-(z_lm * Delta_ref(omega) - 1) * tau_u)`, where
  `Delta_ref = omega + E_gap` (antires), `E_gap - omega` (res_below), or `omega - E_bw` (res_above).
- HGL: `coeff_HGL(omega, tau) = -gamma * w_u` multiplied by
  `sin(gamma * omega * tau)` and `cos(gamma * omega * tau)` mixes.

So we compute the IntegrandTau once per tau point, then multiply-accumulate against
all omega values simultaneously:

```
chi_accum[omega_i] += coeff(denom_type, omega_i, tau_u) * integrand_tau
```

### 5.2 Omega masking for mixed frequency lists

Different omega values may fall in different denominator types for the same window.
Rather than sorting and reshaping, we:

1. Build a boolean `omega_mask` per denom_type during planning (on host).
2. Inside JAX, compute phases for the full omega vector.
3. Multiply by mask to zero out inactive omegas.

This gives static shapes and fewer recompiles.

### 5.3 The scan loop

Requirement: never keep all tau contributions in memory simultaneously.

```python
def scan_denom_type(denom: DenomType, block_A, block_B, omega_vec, acc):
    """Accumulate one denom_type's contribution via lax.scan."""
    mask = denom.omega_mask.astype(acc.dtype)     # (n_omega,)

    def body(acc, packed):
        tau_u, w_u = packed
        integrand = build_integrand_tau(denom, block_A, block_B, tau_u, ...)
        coeff = compute_phase_coeff(denom, omega_vec, tau_u, w_u)   # (n_omega,)
        coeff = coeff * mask
        acc = accumulate_integrand(acc, denom, coeff, integrand)
        return acc, None

    acc, _ = lax.scan(body, acc, (denom.tau, denom.w))
    return acc
```

`lax.scan` carry is the only live accumulator; each IntegrandTau is ephemeral.

### 5.4 Quadrature Sharing Policy for Batched Frequencies

For GL branches, scalar-accurate quadrature parameters depend on `omega` through
effective intervals (for example `E_gap - omega` or `omega - E_bw`). Batched
execution therefore needs an explicit policy:

1. **Bucketed quadrature (recommended):** partition omega into buckets that share
   one `(tau, w, z_lm)` per denom_type and run one scan per bucket.
2. **Conservative shared quadrature:** one grid for the whole mask using worst-case
   effective interval bounds; simpler but can over-integrate.

The planner must choose one policy explicitly; do not mix per-omega quadrature with
single-scan masked accumulation.

### 5.5 Overall loop structure

```python
# Outermost: window pairs (Python loop, fixes static shapes for JIT)
for block_A, block_B, window in window_triples:

    # Plan: classify omegas into denom_types for this window
    denom_types = plan_denom_types(omega_vec, window)

    # Inner: one scan per denom_type
    for denom in denom_types:
        acc = scan_denom_type(denom, block_A, block_B, omega_vec, acc)
```

---

## 6. Core Data Structures

### 6.1 `PoleBlock`

The single data structure for both wavefunctions and plasmon modes. Represents a set
of poles with matrix residues, stored in two memory layouts for efficient `G = R R†`
matmul:

```python
@dataclass
class PoleBlock:
    """A set of poles with matrix residues in two memory layouts.

    For wavefunctions: poles at E_{nk}, residues psi_{nk}(r_mu).
    For plasmons:      poles at Omega_{jq}, residues B_{jq}(r_mu).
    """
    psi_X: Array       # (nk, ns, nmu, npoles) — band/pole axis last, for Sum_n
    psi_Y: Array       # (nk, npoles, ns, nmu) — nmu axis last, for outer product
    energies: Array    # (nk, npoles) — pole positions
    mask: Array        # (nk, npoles) bool — True for valid (non-padded) poles
```

For **chi**: pass valence PoleBlock as A, conduction PoleBlock as B.
For **sigma (valence channel)**: pass valence-wavefunction PoleBlock as A and
plasmon PoleBlock with energies `+Omega_jq` as B.
For **sigma (conduction channel)**: pass conduction-wavefunction PoleBlock as A and
plasmon PoleBlock with energies `-Omega_jq` as B.

Plasmon PoleBlocks have `ns=1` (no spin structure on the screened interaction).

**Construction:** PoleBlocks are built once from raw `WavefunctionBundle` or
plasmon data by a single preprocessing function that handles all dynamic slicing,
padding, and masking. No later code touches raw bundles.

### 6.2 `DenomType`

Describes one denominator region for one window pair — its quadrature and which
omegas it applies to:

```python
@dataclass
class DenomType:
    name: str                   # "antiresonant", "resonant_below", "crossing",
                                # "resonant_above"
    quadrature_kind: str        # "GL" or "HGL"
    sign: int                   # +1 or -1 (overall sign of this contribution)
    E_gap_eff: float            # effective lower bound used to build this shared quadrature
    E_bw_eff: float             # effective upper bound used to build this shared quadrature
    z_lm: float                 # energy scale (= 1/sqrt(E_gap_eff * E_bw_eff))
    tau: Array                  # (n_tau,) quadrature nodes
    w: Array                    # (n_tau,) quadrature weights
    omega_mask: Array           # (n_omega,) bool — which omegas use this denom_type
    damping_kind: str           # "exponential" (GL) or "oscillatory" (HGL)
```

### 6.3 `IntegrandTau`

The omega-independent contribution at one quadrature point:

```python
@dataclass
class IntegrandTau:
    """The contracted propagator product at one tau point."""
    direct: Array | None        # (nmu, nmu, nqx, nqy, nqz) — for GL
    P_plus: Array | None        # (nmu, nmu, nqx, nqy, nqz) — for HGL
    P_cross: Array | None       # (nmu, nmu, nqx, nqy, nqz) — for HGL
```

For GL, only `direct` is populated. For HGL, only `P_plus` and `P_cross`.

### 6.4 `IntegrationPlan`

The complete recipe for one frequency_integration call:

```python
@dataclass
class IntegrationPlan:
    """All window pairs and their denom_types, ready for execution."""
    window_triples: list[tuple[PoleBlock, PoleBlock, WindowPair]]
    denom_types_per_window: list[list[DenomType]]
    omega: Array                # (n_omega,) the frequency grid
    prefactor: float            # -2/(sqrt(N_k) * nspin * nspinor) for chi
```

---

## 7. Data Flow and Module Layout

### 7.1 Data flow diagram

```
WavefunctionBundle / PlasmonBundle
        │
        ▼
┌─────────────────┐     uses     ┌──────────────────────┐
│  slicing.py     │ ◄──────────  │  get_windows.py      │
│  (PoleBlock     │              │  hgl_quadrature.py   │
│   construction) │              │  (EXISTING, don't    │
└────────┬────────┘              │   replicate)         │
         │                       └──────────────────────┘
         │ PoleBlocks
         ▼
┌─────────────────┐
│  planning.py    │  classify omegas → DenomTypes per window
└────────┬────────┘
         │ IntegrationPlan
         ▼
┌─────────────────┐     calls    ┌──────────────────────┐
│  engine.py      │ ───────────► │  integrands.py       │
│  (scan loop,    │              │  (build_integrand_tau │
│   accumulation) │              │   = the physics)     │
└────────┬────────┘              └──────────────────────┘
         │                                │
         │                       uses     │
         │                       ┌────────┘
         │                       ▼
         │               ┌──────────────────────┐
         │               │  fft_ops.py           │
         │               │  (k_to_R, R_to_q,     │
         │               │   reshape/transpose)   │
         │               └──────────────────────┘
         │
         │               ┌──────────────────────┐
         └──────────────►│  layouts.py           │
                         │  (all NamedSharding   │
                         │   specs in one place)  │
                         └──────────────────────┘
```

### 7.2 Module responsibilities

Implementation location: `src/gw_isdf/freqint/`.

| Module | One-sentence role |
|--------|-------------------|
| `src/gw_isdf/freqint/types.py` | Defines `PoleBlock`, `DenomType`, `IntegrandTau`, `IntegrationPlan` |
| `src/gw_isdf/freqint/slicing.py` | Builds PoleBlocks from raw bundles — owns all dynamic slicing and masking |
| `src/gw_isdf/freqint/planning.py` | Classifies omegas into denom_types per window, builds IntegrationPlan |
| `src/gw_isdf/freqint/integrands.py` | `build_integrand_tau` — the only function that constructs and contracts propagators |
| `src/gw_isdf/freqint/engine.py` | `frequency_integration` orchestrator — scan loop, omega batching, accumulation |
| `src/gw_isdf/freqint/fft_ops.py` | `k_to_R`, `R_to_q` — all reshape/transpose/FFT logic, shard_map switching |
| `src/gw_isdf/freqint/layouts.py` | All `NamedSharding` / `PartitionSpec` definitions for the integration pipeline |
| `src/gw_isdf/freqint/api.py` | `chi_from_bundle`, `sigma_from_bundle`, legacy wrappers for `w_isdf_dynamic.py` |

### 7.3 What NOT to replicate

The following already exist and must be reused, not reimplemented:

| Existing module | What it provides |
|-----------------|------------------|
| `get_windows.py` | `WindowPair`, `WindowInfo`, `minimize_cost_fn`, `get_window_info`, `classify_frequencies` |
| `hgl_quadrature.py` | `hgl_nodes_weights`, `n_tau_hgl` |

The new `planning.py` should call `classify_frequencies` from `get_windows.py` and
`n_tau_hgl` / `hgl_nodes_weights` from `hgl_quadrature.py`. It should **not**
contain its own GL/HGL sizing formulas.

The windowing code in `get_windows.py` will need extension to support sigma windows
(where the effective gap and bandwidth involve plasmon energies), but that extension
belongs in `get_windows.py`, not in the new integration engine.

---

## 8. Sharding Discipline

1. Build and contract propagators in `(mu_X, nu_Y)` layout — mu sharded on mesh
   axis `x`, nu on mesh axis `y`.
2. Collectives arise only from FFT lowering (automatic via `jnp.fft`) and the
   Dyson solve `(I - V chi)^{-1}` which lives **outside** this engine.
3. No explicit all-gather or all-reduce in the tau loop.
4. `layouts.py` is the single source of truth for all `PartitionSpec` definitions.

---

## 9. Implementation Sequence

### Step 1: Types and infrastructure (no math changes)

Define `PoleBlock`, `DenomType`, `IntegrandTau` in `types.py`. Extract FFT
helpers into `fft_ops.py` and sharding specs into `layouts.py`. Point existing
dynamic code at these modules to verify they work.

**Unblocks:** everything else — all later steps consume these types.

### Step 2: PoleBlock construction

Implement `slicing.py` — the single place that converts raw `WavefunctionBundle`
slices into PoleBlocks with canonical shapes, padding, and masks. Remove raw
`dynamic_slice` calls from all kernel code.

**Unblocks:** clean kernel signatures that take PoleBlocks, not raw bundles.

### Step 3: Plan-driven denom_type execution

Implement `planning.py` using existing `classify_frequencies` and quadrature
sizing from `get_windows.py` / `hgl_quadrature.py`. Replace the ad-hoc
if/elif/else chains in `w_isdf_dynamic.py` with iteration over a
`list[DenomType]`.

**Unblocks:** clean separation of "what to compute" (planning) from "how to
compute it" (kernels).

### Step 4: Unified `build_integrand_tau`

Move propagator construction into `integrands.py`. GL and HGL use the same
function with different damping (exponential vs oscillatory). This function
takes two PoleBlocks — it doesn't know chi from sigma.

**Unblocks:** sigma support (Step 6) — the kernel is already mode-agnostic.

### Step 5: Multi-omega scan accumulation

Implement the `lax.scan` loop in `engine.py` with omega masks and batched phase
multiplication. Add omega chunking if memory requires it.

**Unblocks:** batched frequency evaluation (the main performance win).

### Step 6: Sigma mode

Build plasmon PoleBlocks from the plasmon decomposition of `W_c`. Extend
`get_windows.py` to produce sigma-appropriate windows. Call the same
`build_integrand_tau` with `(valence_wfn_block, plasmon_block_plus)` and
`(conduction_wfn_block, plasmon_block_minus)`, then sum.

**Unblocks:** full GW self-energy beyond COHSEX.

### Step 7: Legacy cleanup

Make `w_isdf_dynamic.py` into thin wrappers around `api.py`. Delete the four
redundant kernel factories once parity tests pass.

---

## 10. Verification Matrix

| Category | Tests |
|----------|-------|
| **Quadrature** | GL/HGL node counts match `get_windows.py` / `hgl_quadrature.py` |
| **Denom classification** | omega_mask correct at boundaries E_gap ± delta, E_bw ± delta |
| **Boundary policy** | exact-boundary handling (`omega=E_gap`, `omega=E_bw`) follows configured tolerance and does not diverge |
| **Analytic chi** | Exact for toy 2-band model outside crossing; HGL-regularized inside |
| **Analytic sigma** | Toy system with synthetic plasmon poles and known reference |
| **Parity** | New engine matches old `w_isdf_dynamic.py` to machine precision for chi |
| **Distributed** | 1-device vs multi-device identical results |
| **Performance** | Fewer compilations, bounded peak memory, runtime/omega measured |

---

## 11. Deliverables

The rewrite is complete when:

1. `frequency_integration(block_A, block_B, windows, omega, ...)` handles both chi
   and sigma via the same code path.
2. `build_integrand_tau` is the only propagator constructor — takes two PoleBlocks,
   returns IntegrandTau.
3. Tau integration is `lax.scan`-based with one IntegrandTau in memory at a time.
4. Omega selection is mask-based inside JAX with static shapes.
5. Sigma(omega) works by two explicit channels:
   `(valence wfns, plasmons +)` and `(conduction wfns, plasmons -)`.
6. Existing public APIs still work via thin wrappers.
7. All quadrature/windowing logic delegates to `get_windows.py` and
   `hgl_quadrature.py` — no duplication.
8. Test matrix passes with no physics regressions.

---

## 12. Glossary

| Code name | Physics symbol | Definition |
|-----------|----------------|------------|
| `z_lm` | zeta_{lm} | Energy scale for window pair (l,m): `1/sqrt(E_gap * E_bw)` |
| `gamma` | gamma | HGL broadening parameter (typically = z_lm for crossing windows) |
| `epsq` | epsilon^(q) | Target fractional quadrature error |
| `alpha` | alpha | Bandwidth ratio `sqrt(E_bw / E_gap)` — controls GL point count |
| `E_gap` | E^(gap)_{lm} | Minimum transition energy in window: `E_c,min - E_v,max` |
| `E_bw` | E^(bw)_{lm} | Maximum transition energy in window: `E_c,max - E_v,min` |
| `psi_X` | — | Wavefunction in (nk, ns, nmu, nb) layout — mu sharded on mesh axis x |
| `psi_Y` | — | Wavefunction in (nk, nb, ns, nmu) layout — mu sharded on mesh axis y |
| `P_plus` | P_+ | `Re[(G^c)† G^v]` — HGL integrand component |
| `P_cross` | P_x | `-Im[(G^c)† G^v]` — HGL integrand component |
| `IntegrandTau` | chi(tau) or Sigma(tau) | The contracted propagator product at one quadrature point |
| `DenomType` | — | One denominator region (antiresonant, resonant_below, crossing, resonant_above) |
| `PoleBlock` | — | Poles + matrix residues in canonical layout (wavefunctions or plasmons) |

---

## References

- Kim, Martyna & Ismail-Beigi, PRB 101, 035139 (2020) — CTSP method
- `docs/chi_omega_quadrature.md` — Full derivation and implementation design
- `src/gw_isdf/get_windows.py` — Existing window optimization
- `src/gw_isdf/hgl_quadrature.py` — Existing HGL quadrature construction
