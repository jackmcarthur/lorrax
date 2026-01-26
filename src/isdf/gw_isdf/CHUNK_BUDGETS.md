# Chunk Budget Cheat Sheet

This note summarizes the *exact* allocations that limit each tunable chunk.
Use it when you want to reason quickly about a single parameter without
reading the full memory-model derivation.

## Band Chunk (`band_chunk`)

Controls how many bands are FFT'd simultaneously while extracting centroids.

Peak per-device bytes:

```
M_full                                      # union of centroids (2 arrays)
+ 16 * n_k * n_r                            # phase tensor
+ 2 * 16 * n_k * (band_chunk / P) * n_s * n_r
```

*Bottleneck arrays*

- `psi_G` and `psi_r`: `(n_k, band_chunk / P, n_s, n_r)`
- `phase_spatial`: `(n_k, n_r)` (broadcast)

If this stage fails, reduce `n_r` (FFT grid), shrink the band ranges, or
increase the budget.

## X/R Chunk (`x_chunk`, `x_chunk_r`)

Controls how many contiguous r-points are processed per zeta loop.

`base = M_cent + M_L_q + cache`, where `cache` is either zero or the sum of
cached band chunks (`16 * n_k * (n_b / P) * n_s * n_r`).

Three live constraints:

1. **Pair density**

```
base
+ 16 * n_k * n_b * n_s * (x_chunk_r / p_y)                 # psi_xchunk
+ 2 * 16 * n_k * (n_rmu / p_x) * (x_chunk_r / p_y)        # P_l + P_r
```

2. **ZCT pipeline**

```
base
+ 2 * 16 * n_k * (n_rmu / p_x) * (x_chunk_r / p_y)        # P_l + P_r
+ 16 * n_q * (n_rmu / p_x) * (x_chunk_r / p_y)            # Z_q
```

3. **Solve (q_chunk = 1)**

```
base
+ 2 * 16 * n_q * n_rmu * (x_chunk_r / P)                  # Z_col + zeta
+ 16 * n_rmu^2                                            # replicated L
```

*Bottleneck arrays*

- `psi_xchunk_Y`: `(n_k, n_b, n_s, x_chunk_r/p_y)`
- `P_l`, `P_r`: `(n_k, n_rmu/p_x, x_chunk_r/p_y)`
- `Z_q`: `(n_q, n_rmu/p_x, x_chunk_r/p_y)`
- `Z_col`/`zeta`: `(n_q, n_rmu, x_chunk_r/P)`
- `L_rep`: `(n_rmu, n_rmu)` per q during the solve

If any stage exceeds the budget the solver shrinks `x_chunk` and rechecks all
three constraints to keep the GPU full without overrunning.

## Q Chunk (`q_chunk`)

Controls how many `L_q` columns are replicated at once during the solve.

After `x_chunk` is fixed, the remaining bytes are

```
available = M_budget - (base + 2 * 16 * n_q * n_rmu * (x_chunk_r / P))
```

The replicate cost is `16 * n_rmu^2` per q, so

```
q_chunk <= available / (16 * n_rmu^2)
```

*Bottleneck arrays*

- Replicated Cholesky panels (`B_q` copies of `(n_rmu, n_rmu)`)
- `Z_col`/`zeta` from the x-chunk stage

When `available < 16 * n_rmu^2`, even `q_chunk = 1` would not fit, so `x_chunk`
is reduced until one q at a time is feasible.

## μ Chunk (`mu_chunk_size` for V_q)

Defines how many μ/ν rows of ζ are Fourier transformed at once while building
`V_q`.  The working set contains the r-space block, its FFT, and one ν-block
for off-diagonal tiles:

```
per_mu_bytes ≈ 3 * 16 * n_r
available_vq = effective_budget - M_cent                      # centroids stay alive
mu_chunk <= available_vq / per_mu_bytes
```

*Bottleneck arrays*

- `ζ_μ(r)`, shape `(μ_chunk, n_r)`
- `ζ̃_μ(G)`, shape `(μ_chunk, n_r)` (after weighting)
- `ζ̃_ν(G)` for the off-diagonal contraction
- Temporary `V_block (μ_chunk, μ_chunk)` on the host

This chunk is reported by the driver so that V_q performance tuning can be
done separately from the zeta-fitting stages.

---

For deeper derivations and implementation details, see
`MEMORY_MODEL.md`.  This cheat sheet is intended for quick reference when
only one chunk knob is under consideration.
