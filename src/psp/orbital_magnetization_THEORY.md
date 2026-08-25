# Orbital magnetization from a spinor wavefunction — modern theory, explicit dH/dk

This note documents the physics implemented in `psp/orbital_magnetization.py`:
the per-cell **orbital magnetic moment** of a spin–orbit-coupled crystal,
computed from a DFT wavefunction by the *modern theory of orbital
magnetization* in its **sum-over-states (k·p)** form, with the
Brillouin-zone derivative of the Hamiltonian taken **analytically** through
`dH/dk = 2(k+G) + dV_NL/dk`. No finite differences enter the velocity
operator.

## 0. Units

Rydberg atomic units throughout: `ħ = 1`, `2 mₑ = 1` (so `mₑ = ½`), `e² = 2`,
lengths in Bohr `a₀`, energies in Rydberg `Ry`. The velocity operator is

```
v = dH/dk = 2(k+G)_cart  +  dV_NL/dk      [Ry·Bohr]
```

i.e. the kinetic part is `2(k+G)` because a plane wave has kinetic energy
`T = |k+G|²` (Ry) and `dT/dk = 2(k+G)`; this factor of 2 is exactly what
distinguishes the *velocity* `dH/dk` from the bare *momentum* `⟨p⟩ = (k+G)`
(`v = 2p` since `mₑ = ½`). In LORRAX this is assembled from
`dft_operators.momentum_matrix_k` and `vnl_ops.vnl_velocity_matrix`.

## 1. Master formula (modern theory)

The orbital magnetization of a crystal — the orbital magnetic moment per unit
volume — is the gauge-invariant Brillouin-zone integral
[Thonhauser–Ceresoli–Vanderbilt–Resta 2005; Ceresoli et al. 2006; Xiao–Chang–Niu 2010]:

```
        e          ⌠  dᵈk
M = − ─────  Im  Σ  ⎮ ─────  f_nk  ⟨∂_k u_nk| × (H_k + ε_nk − 2μ) |∂_k u_nk⟩
       2ħ       n  ⌡ (2π)ᵈ
```

(SI; in Gaussian-cgs replace `e/2ħ → e/2ħc`). Here `u_nk` is the cell-periodic
Bloch state, `H_k = e^{−ik·r} H e^{ik·r}` the k-dependent Hamiltonian whose
eigenvalue is `ε_nk`, `f_nk` the occupation, and `μ` the chemical potential.
The **leading minus** is the electron charge `q = −e` (`e > 0`): an electron's
orbital magnetic moment is *antiparallel* to its mechanical angular momentum,
`m = −(e/2mₑ) L = −μ_B L/ħ`. The Bohr magneton is `μ_B = eħ/2mₑ > 0`.

### Local / itinerant decomposition

Writing `H + ε − 2μ = (H − ε) + 2(ε − μ)` splits `M` into the **local
circulation** (self-rotation of the wavepacket, the `H − ε` term) and the
**itinerant circulation** (center-of-mass motion, the Berry-curvature-weighted
`2(ε − μ)` term). The combined form above is gauge-invariant and is what we
evaluate; the `−2μ` term is why `M` is referenced to the chemical potential.

## 2. Sum-over-states reduction (the explicit-dH/dk method)

Insert a complete set of eigenstates into `|∂_k u_n⟩`. Off-diagonal
derivatives follow from first-order perturbation theory ("Sternheimer"):

```
⟨u_m | ∂_{k_a} u_n⟩ = ⟨u_m | ∂_{k_a} H | u_n⟩ / (ε_n − ε_m) = v^a_mn / (ε_n − ε_m),   m ≠ n
```

with `v^a_mn ≡ ⟨u_m | dH/dk_a | u_n⟩` the velocity matrix element. Because the
eigenbasis diagonalizes `H`,

```
⟨u_m | (H + ε_n − 2μ) | u_{m'}⟩ = (ε_m + ε_n − 2μ) δ_{m m'},
```

so the matrix element collapses to a single sum over `m`:

```
⟨∂_a u_n| (H+ε_n−2μ) |∂_b u_n⟩ = Σ_{m≠n} v^a_nm v^b_mn (ε_m+ε_n−2μ)/(ε_n−ε_m)²
```

(using `⟨∂_a u_n|u_m⟩ = v^a_nm/(ε_n−ε_m)` and Hermiticity `v^a_mn* = v^a_nm`).
Taking the antisymmetric (cross-product) combination `ε_{γab}` gives the final
**sum-over-states formula** for component γ:

```
        e                                v^a_nm v^b_mn (ε_m + ε_n − 2μ)
M_γ = − ── Im Σ ∫ dᵈk/(2π)ᵈ f_n Σ   ε_γab ──────────────────────────────
       2ħ      n               m≠n                  (ε_n − ε_m)²
```

The **outer sum `n` runs over occupied bands**; the **inner sum `m` runs over
*all* bands** (occupied and empty — it is a resolution of the identity).
Because the summand decays as `1/|ε_n − ε_m|` for distant bands, truncating the
`m`-sum at a finite band ceiling converges — this is the "large but truncated
set of bands" summed in practice (script flag `--nbnd`, diagnostic
`--convergence`).

## 3. Per-cell moment in Bohr magnetons (Rydberg units)

On a uniform `N_k` mesh, `∫dᵈk/(2π)ᵈ → (1/V_cell) Σ_k w_k` with `Σ_k w_k = 1`.
The per-cell moment is `m = M · V_cell`, so **V_cell cancels** and no cell
volume appears. Dividing the SI prefactor `e/2ħ` by `μ_B = eħ/2mₑ` gives
`mₑ/ħ²`, and with energy in Ry and length in Bohr,
`ħ²/(mₑ a₀²) = 2 Ry ⇒ mₑ/ħ² = 1/(2 Ry·a₀²)`. Hence, feeding the **true
velocity** `v = dH/dk` (Ry·Bohr) directly:

```
                       1
m_γ / μ_B = (−1) · ─────  Σ_k w_k  Im Σ      Σ    ε_γab v^a_nm v^b_mn (ε_m+ε_n−2μ)/(ε_n−ε_m)²
                       2             n occ  m≠n
```

> **Prefactor = −½.** The magnitude ½ is `mₑ/ħ²` in Ry·a₀² and is confirmed
> three independent ways (direct, momentum cross-check `v = 2p`, and SI
> dimensional analysis). The **sign** is the electron-charge minus
> (`m = −μ_B L/ħ`); LORRAX's production operator pair returns the velocity (not the
> momentum), so no extra factor of 2 is applied. There is **no spin-degeneracy
> factor of 2** — each 2-component spinor band is counted once.

For an out-of-plane easy-axis monolayer (CrI₃) the physical moment is `m_z`
(γ = z, ab = xy): `cross_z = v^x_nm v^y_mn − v^y_nm v^x_mn`. In code,
`v^a_nm = v[a, n, m]` (bra n, ket m) so `cross_z = v[0]·v[1]ᵀ − v[1]·v[0]ᵀ`.

## 4. Chemical potential and Chern dependence

`μ` is set midgap by default, `μ = ½(VBM + CBM)`. For a true insulator the
result is **independent of where μ sits in the gap iff the occupied-band Chern
number is zero** (`dM/dμ ∝ −C`, the anomalous-Hall relation). Monolayer CrI₃
is a quantum-anomalous-Hall *candidate*, so the `--mu-scan` diagnostic reports
`m_z` at μ = VBM, midgap, CBM to expose any `dM/dμ ≠ 0`. On a coarse mesh the
DFT indirect gap can come out slightly negative (band overlap across different
k); the moment then genuinely depends on μ and the result should be read as
under-converged in k.

## 5. Spin–orbit coupling is mandatory

Without SOC a collinear ferromagnet block-diagonalizes into two **spinless**
spin channels, each obeying spinless time-reversal `H(−k) = H(k)*`. That forces
the Berry curvature and wavepacket orbital moment to be odd in k,
`Ω_n(−k) = −Ω_n(k)`, so the BZ integral vanishes band-by-band: **M_orb ≡ 0**.
It is SOC that ties the orbital/k degrees of freedom to the time-reversal-
breaking spin order. The script therefore requires `nspinor == 2` (hard error
otherwise).

## 6. Sign relative to the spin moment, and order of magnitude

The formula contains **no spin operator** — the orbital moment's sign relative
to the spin moment is an *emergent* result, not something the prefactor sign
encodes. To make it physical and convention-robust, the script also computes
the spin moment from the same wavefunction with the *same* electron-charge
convention,

```
m_spin,z = − μ_B  Σ_k w_k Σ_{n occ} ⟨σ_z⟩_nk ,   ⟨σ_z⟩_nk = Σ_G(|c↑|² − |c↓|²),
```

which must come out `≈ ±6 μ_B` for CrI₃ (2 Cr³⁺, S = 3/2 each). Both moments
carry the same `−μ_B` gyromagnetic sign, so their **relative** orientation is
convention-independent; the script reports the orbital moment *projected onto
the spin-moment axis* (positive = parallel). The free-ion Hund's-third-rule
expectation for less-than-half-filled Cr³⁺ (3d³) is orbital *antiparallel* to
spin, but crystal field and covalency can change this in the solid, so the
computed sign is reported, not assumed. The expected magnitude for CrI₃ is
`|m_orb| ~ 0.1 μ_B` per cell, likely under-converged on a coarse k-mesh.

## 7. Validation built into the script

* **Hellmann–Feynman group velocity.** The diagonal `Re⟨n|dH/dk|n⟩` must equal
  the band group velocity `∂ε_n/∂k` (finite-differenced on the k-mesh). This
  validates the kinetic velocity magnitude/units. It is, however, *insensitive
  to the nonlocal sign*: the nonlocal velocity `dV_NL/dk` is almost purely
  off-diagonal (verified ~900× larger off-diagonal than on-diagonal for CrI₃),
  so the diagonal slope test ties between `p±vNL`.
* **Nonlocal-velocity sign (definitive).** The sign of `dV_NL/dk` is fixed by a
  direct off-diagonal finite difference of `⟨m|V_NL(k)|n⟩` (ψ held fixed):
  the analytic `compute_vnl_velocity_cart` equals `+dV_NL/dk` to ratio +1.000.
  Hence the physical velocity is **`v = p + vNL`** (the canonical production
  operator convention). The dipole driver's `p − vNL` flip is a
  BerkeleyGW-matching convention for optical matrix elements, *not* the
  physical velocity, and must not be used here — it would flip the orbital
  moment's sign (CrI₃: `+0.026` → `−0.081 μ_B`).
* **Symmetry.** `m_x, m_y ≈ 0` for an out-of-plane ferromagnet.
* **Spin moment.** `|m_spin| ≈ 6 μ_B` cross-checks the wavefunction/occupations
  and pins the reporting axis.

## Sources

1. T. Thonhauser, D. Ceresoli, D. Vanderbilt, R. Resta, *Orbital Magnetization
   in Periodic Insulators*, Phys. Rev. Lett. **95**, 137205 (2005).
   https://doi.org/10.1103/PhysRevLett.95.137205
2. D. Ceresoli, T. Thonhauser, D. Vanderbilt, R. Resta, *Orbital magnetization
   in crystalline solids…*, Phys. Rev. B **74**, 024408 (2006).
   https://doi.org/10.1103/PhysRevB.74.024408
3. J. Shi, G. Vignale, D. Xiao, Q. Niu, *Quantum Theory of Orbital
   Magnetization and Its Generalization to Interacting Systems*, Phys. Rev.
   Lett. **99**, 197202 (2007). https://doi.org/10.1103/PhysRevLett.99.197202
4. D. Xiao, M.-C. Chang, Q. Niu, *Berry phase effects on electronic
   properties*, Rev. Mod. Phys. **82**, 1959 (2010) — orbital magnetization
   section. https://doi.org/10.1103/RevModPhys.82.1959
5. M. G. Lopez, D. Vanderbilt, T. Thonhauser, I. Souza, *Wannier-based
   calculation of the orbital magnetization in crystals*, Phys. Rev. B **85**,
   014435 (2012). https://doi.org/10.1103/PhysRevB.85.014435
6. D. Vanderbilt, *Berry Phases in Electronic Structure Theory* (Cambridge,
   2018), Ch. 5–6.
