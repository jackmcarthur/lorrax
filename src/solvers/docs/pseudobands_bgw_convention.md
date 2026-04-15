# BGW Pseudobands WFN.h5 Convention

Verified from `runs/Si_pseudobands/00_si_2x2x2_60Ry/02_bgw_pseudobands_50sl/WFN_pseudo.h5`.

## Wavefunction norms

Protected bands (deterministic eigenstates):
  `|c|² = nk`  (standard BGW normalization, here nk=4)

Pseudobands (stochastic compressed bands):
  `|c|² = nk × (n_states_in_slice / n_xi)`

The spectral weight `n_states_in_slice / n_xi` (= `n_eff / k` in our notation)
is **absorbed into the wavefunction coefficients**. The GW code treats all bands
uniformly — it just sees coefficients with larger norms for pseudobands, which
naturally weights their contribution to χ⁰ and Σ correctly.

## Example (Si 2×2×2, 60 Ry cutoff)

```
Protected:  bands 0-15,  |c|² = 4.0   (nk=4, standard)
Pseudo:     bands 16-17, |c|² = 56.0  (4 × 14 states/2 xi = 4 × 7)
            bands 18-19, |c|² = 56.0
            bands 20-21, |c|² = 80.0  (4 × 20 states/2 xi = 4 × 10)
            ...
            bands 114-115, |c|² = 224.0  (4 × 56 states/2 xi = 4 × 28)
```

## Parameters used

```
nc = 8            (protected conduction bands)
nslice_c = 50     (conduction slices)
nspbps_c = 2      (pseudobands per slice)
Total: 8 val + 8 det_cond + 50×2 pseudo_cond = 116 bands
```

## LORRAX compatibility

Our `ritz_pseudobands` already absorbs `sqrt(n_eff/k)` into `Phi_out`.
The WFN writer adds the `sqrt(nk)` factor. No writer changes needed.
