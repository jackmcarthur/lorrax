# ISDF-BSE: Bethe-Salpeter Equation with Interpolative Separable Density Fitting

This module implements a high-performance BSE solver using ISDF for low-rank approximation of electron-hole interactions.

## Algorithm Overview

### BSE Hamiltonian (Tamm-Dancoff Approximation)

The BSE Hamiltonian in TDA for **spinors** is:

```
H_BSE = D + V - W
```

where:
- **D**: Diagonal term from QP energy differences: `D_{cvk} = ε_c(k) - ε_v(k)`
- **V**: Direct (bare Coulomb) term at q=0 (repulsive)
- **W**: Screened exchange term with k→k' momentum transfer (attractive)

**Note on spin factors**: For spin-restricted singlet excitons the textbook form is `D + 2V - W`, where the factor of 2 comes from spin summation. For spinors (spin-orbit coupled wavefunctions), we use `D + V - W` because the Coulomb interaction is spin-independent and couples to the charge density at each vertex, which is already spin-traced in the pair amplitude `M = Σ_σ ψ*_{c,σ} ψ_{v,σ}`.

### ISDF Representation

The pair density is expanded in ISDF interpolation vectors:

```
ρ_cv(r,k) = ψ*_c(r,k) ψ_v(r,k) ≈ Σ_μ ζ_μ(r) M_cv(μ,k)
```

where:
- `ζ_μ(r)`: ISDF interpolation vectors at centroids
- `M_cv(μ,k) = Σ_s ψ*_{c,s}(μ,k) ψ_{v,s}(μ,k)`: **Spin-traced** pair amplitude (scalar)

This reduces the 4-index electron-hole interaction to 2-index matrices `V_{μν}` and `W_{μν}(q)`.

### Matrix-Vector Product

For trial vector `X(c,v,k)`, the matvec `HX` is computed as:

1. **Encode**:
   - **V term**: project `X(c,v,k)` to μ-space via the **spin-traced** cv pair amplitude `M_cv`
   - **W term**: build a **2×2 spin matrix** at each ISDF point pair (μ,ν) as in Henneke (2020) eq (4-6)
   ```
   S(ν,k) = Σ_{c',v'} M(k,c',v',ν) X(c',v',k)
   ```

2. **Apply interaction**:
   - **V term** (q=0 only): `U_V(μ,k) = Σ_ν V_{μν} S(ν,k)`
   - **W term** (FFT convolution, Henneke eq (4-6)):

     Build the spin-matrix intermediate

     \[
     T_{ts}(μ,ν,k)=\sum_{c',v'} ψ_{c',t}(μ,k)\,ψ^{*}_{v',s}(ν,k)\,X(c',v',k)
     \]

     and apply the convolution in k for each \((μ,ν,t,s)\):

     ```
     T(μ,ν,t,s,R) = IFFT_k[T(μ,ν,t,s,k)]     # k → R
     U(μ,ν,t,s,R) = W(μ,ν,R) * T(μ,ν,t,s,R)  # scalar W multiplies each spin component
     U(μ,ν,t,s,k) = FFT_R[U(μ,ν,t,s,R)]      # R → k
     ```

     The BSE definition carries an overall **`1/Nk`** prefactor on the W term. We use
     unitary FFTs (`norm='ortho'`), so the FFT-based convolution carries a natural
     **`1/sqrt(Nk)`** scaling; we apply one additional **`1/sqrt(Nk)`** factor to recover
     the physical **`1/Nk`** overall normalization.

3. **Decode**:
   - **V**: project back to (c,v) using `M*` as usual
   - **W**: contract the spin matrix with the external (c,v) spinors:

     \[
     [WX](c,v,k)=\sum_{μ,ν,t,s} ψ^*_{c,t}(μ,k)\,U_{ts}(μ,ν,k)\,ψ_{v,s}(ν,k)
     \]

### Eigensolver: Lanczos Algorithm

We use the Lanczos algorithm to find the lowest exciton eigenvalues without forming the full Hamiltonian matrix.

Two implementations are provided:
- **`lanczos_eig_jit`**: Fully JIT-compiled using `lax.fori_loop` (default, faster)
- **`simple_lanczos_eig`**: Python-loop version (easier to debug)

Key features:
- Pre-allocated arrays for JIT compatibility
- Selective reorthogonalization (configurable via `n_reorth`)
- Tridiagonal solve via `jnp.linalg.eigh` (trivially fast for m×m where m~100)

---

## Sharding Strategy (Multi-GPU)

### Device Mesh

We use a 2D mesh `(X, Y)` matching the COHSEX conventions in `load_wfns.py`:

```python
mesh = Mesh(devices.reshape(Px, Py), axis_names=('x', 'y'))
```

### Array Shardings

| Array | Shape | Sharding | Memory/Device |
|-------|-------|----------|---------------|
| `X` (trial vec) | `(b, nc, nv, nk)` | `P(None, 'x', None, None)` | O(nc/Px × nv × nk) |
| `W_q` | `(n_rmu, n_rmu, nk)` | `P('x', 'y', None)` | O(n_rmu²/P) |
| `V_q0` | `(n_rmu, n_rmu)` | `P('x', 'y')` | O(n_rmu²/P) |
| `psi_c` | `(nk, nc, ns, n_rmu)` | `P(None, None, None, 'x')` | O(nk × nc × ns × n_rmu/Px) |
| `psi_v` | `(nk, nv, ns, n_rmu)` | `P(None, None, None, 'y')` | O(nk × nv × ns × n_rmu/Py) |

### Communication Pattern

Each matvec requires 3 collective operations:
1. **`psum` over X-axis**: Complete c-sum in encoding
2. **`psum` over Y-axis**: Complete ν-sum after W contraction
3. **`reduce_scatter` over X-axis**: Distribute c in decoding

This achieves O(n_rmu²/P²) memory per device for W, enabling large systems.

---

## JIT Compilation

### Static vs Dynamic Arguments

```python
@partial(jax.jit, static_argnames=("nkx", "nky", "nkz"))
def apply_bse_hamiltonian(..., nkx: int, nky: int, nkz: int):
```

- **Static**: `nkx, nky, nkz` (k-grid dimensions) - triggers recompilation if changed
- **Dynamic**: All arrays - can change without recompilation

### Caching Behavior

The matvec is JIT-compiled on first call. Subsequent calls reuse the compiled kernel.
For Lanczos, the inner matvec is called ~50-100 times, so compilation cost is amortized.

### Warm-up

The test script includes warm-up iterations to separate JIT compilation time from execution time:
```bash
uv run python -m bse_isdf.test_bse -i input.in --n-warmup 2 --n-bench 10
```

---

## Usage

### Running the Test

```bash
cd /path/to/cohsex_prod
uv run python -m bse_isdf.test_bse -i cohsex_prod.in \
  --n-val 4 --n-cond 4 \
  --n-eig 10 --max-iter 50 \
  --write-eigenvectors eigenvectors.h5
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-i, --input` | (required) | COHSEX input file (for directory context) |
| `--n-val` | 4 | Number of valence bands |
| `--n-cond` | 4 | Number of conduction bands |
| `--n-eig` | 10 | Number of exciton eigenvalues |
| `--max-iter` | 50 | Maximum Lanczos iterations |
| `--n-warmup` | 2 | JIT warm-up iterations |
| `--n-bench` | 10 | Benchmark iterations |
| `--no-jit-lanczos` | False | Use Python-loop Lanczos instead of JIT |
| `--write-eigenvectors` | None | Output HDF5 file for eigenvectors |

### Profiling

Enable JAX profiler tracing:
```bash
ISDF_JAX_PROFILE_DIR=./jax_traces uv run python -m bse_isdf.test_bse -i input.in
tensorboard --logdir=./jax_traces
```

Timing report is printed at the end of each run.

---

## Output Format

Eigenvectors are written to HDF5 following the BerkeleyGW `eigenvectors.h5` spec:

```
exciton_header/
  params/
    bse_hamiltonian_size, nevecs, ns, nc, nv, use_tda, spin_kernel
  kpoints/
    nk, kpts, nQ, exciton_Q_shifts
exciton_data/
  eigenvalues     # (n_eig,)
  eigenvectors    # (nQ, nevecs, nk, nc, nv, ns, 2) for complex
```

---

## Performance Characteristics

### Typical Timing (4 val × 4 cond × 9 k-points, 600 ISDF points, 1 GPU)

| Operation | Time |
|-----------|------|
| Load data | 2.0s |
| Matvec (JIT compile) | 1.0s |
| Matvec (per call after JIT) | 6ms |
| Lanczos (50 iters) | 2.5s |

### Scaling Considerations

For production calculations (50 val × 50 cond × 216 k-points, 2000 ISDF):
- **W_q memory**: ~7 GB → requires sharding across multiple GPUs
- **Lanczos vectors Q**: ~850 MB → fits on single GPU
- **Tridiagonal T**: ~160 KB → trivially small

The matvec cost dominates; Lanczos overhead (reorthogonalization, tridiagonal solve) is negligible.

---

## Known Limitations

1. **V as W placeholder**: Currently uses bare Coulomb V as stand-in for screened W. Real W loading from `eps0mat.h5` is TODO.

2. **No finite-Q support**: Currently Q=0 only. The infrastructure supports nQ>1 but is not yet implemented.

3. **TDA only**: Full BSE (beyond TDA) with coupling to de-excitation amplitudes is not implemented.

4. **Numerical Hermiticity**: The BSE Hamiltonian has ~1e-4 non-Hermiticity due to small imaginary parts in V_qmunu from ISDF fitting. This is a numerical artifact, not a physics error.

---

## Future TODO

- [ ] Load actual screened W from `eps0mat.h5` or computed χ₀ inversion
- [ ] Finite-Q momentum transfer for indirect excitons
- [ ] Full BSE (beyond TDA) with B matrix coupling
- [ ] Optical matrix elements and absorption spectrum calculation
- [ ] Distributed Lanczos vectors for extremely large systems
- [ ] Block Lanczos tuning for computing many eigenvalues efficiently
- [ ] Integration with COHSEX restart workflow

---

## Files

| File | Description |
|------|-------------|
| `bse_jax.py` | Core BSE matvec and Lanczos implementations |
| `test_bse.py` | Test script with timing and profiling |
| `write_eigenvectors.py` | HDF5 output in BerkeleyGW format |
| `eigenvectors.h5.spec` | Format specification for output |
| `bse_isdf_instructions.md` | Original design notes |
| `gpt5.2suggestion.md` | Alternative sharding proposals |

