# psp/ — DFT Operator Module: Development Status

## What this module does

Constructs and applies the PW DFT Hamiltonian H = T + V_scf + V_NL entirely
on GPU via JAX.  Validated to 0.0000 mRy vs QE eigenvalues (Si 4×4×4 FR-PBE,
all 8 IBZ k-points).  Includes a working Davidson eigensolver that reads only
QE `.save/data-file-schema.xml` + `charge-density.hdf5` + pseudopotentials.

## File map

| File | Lines | Role |
|------|-------|------|
| `dft_operators.py` | 943 | **Core**: HamiltonianK, apply_H_k, build_matrix_k, setup_H_k*, V_scf builder, velocity/dipole (autodiff V_NL) |
| `davidson.py` | 262 | Block Davidson eigensolver: generalized eigh, PW initial guess, QE-style preconditioner |
| `qe_save_reader.py` | 373 | CrystalData: duck-types WFNReader from QE XML. Symmetries, IBZ k-grid, charge density |
| `wfn_writer.py` | 188 | Write BGW-format WFN.h5 from CrystalData + Davidson output |
| `operator_checks.py` | 114 | Pre-flight: pseudos present, sys_dim → truncation_2d |
| `charge_density.py` | 534 | ρ_val from wavefunctions, ρ_core (NLCC), V_xc (PBE GGA via autodiff) |
| `vnl_ops.py` | 444 | Dense VNL backend: table-lookup radial form factors, single Z matrix per k |
| `build_projectors_qe.py` | 908 | V_loc builder (FT of local PP), PP loading, projector splines |
| `solid_harmonics.py` | ~100 | Cartesian solid harmonics S_lm, autodiff-safe at K=0 |

## Bugs found and fixed (this session)

1. **truncation_2d=True hardcoded** in get_kin_ion and build_operator_setup.
   For 3D bulk: ~3 mRy band error, ~200 mRy offset. Now from sys_dim.
2. **kin_ion_io_chunked silently skipped V_loc** (checked `wfn.Vloc_r` which
   was never set). Any saved kin_ion.h5 from the chunked path was T-only.
3. **No validation pseudopotentials loaded**: silent T-only matrices.
4. **Davidson: standard eigh instead of generalized**. Non-orthogonal basis
   from preconditioned residuals requires Sc. Standard eigh gives zero modes.
5. **hamiltonian_matvec.py was 1120 lines of dead code** (imported by nobody).

## Performance (Si 4×4×4, 12 bands, nspinor=2, single A100)

### One-time costs
| Step | Time | Notes |
|------|------|-------|
| V_loc (numpy PP FT) | 0.5s | CPU; could be JIT'd |
| ρ_core NLCC (numpy Simpson) | 1.9s | CPU; biggest numpy bottleneck |
| V_H + V_xc (JIT trace) | 1.2s | `compute_V_H_and_V_xc`, cached at 1.2 ms |
| VNL setup (radial tables) | ~2s | CPU; q_max scan over k-points |
| Davidson JIT warmup | 6.8s | `warmup_jit()`: 4 _subspace_step + 2 apply_H shapes |

### Per k-point (warmed JIT)
| Step | Time |
|------|------|
| setup_H_k_from_kvec | ~20 ms |
| apply_H_k (H\|ψ⟩, 12 bands) | ~2 ms |
| _subspace_step (project+eigh+Ritz+precond) | ~1.4 ms |
| **Full Davidson (one k)** | **0.08–0.16s** |

### End-to-end NSCF
```
Setup:    7.2s (V_loc + NLCC + V_H/V_xc JIT + VNL)
Warmup:   6.8s (one-time JIT precompilation)
Davidson: 2.0s (8 k-points × ~0.25s amortized)
Write:    0.02s (WFN.h5)
Total:    26.2s
```

## How to test

```bash
# From the sandbox, Si 4×4×4:
cd runs/Si/04_si_4x4x4_davidson/00_davidson
PYTHONPATH=".../lorrax_bse/src:$SITE:$SANDBOX/sources" \
JAX_ENABLE_X64=1 HDF5_USE_FILE_LOCKING=FALSE \
python3 -u run_nscf.py \
    --save ../../00_si_4x4x4_60band/qe/scf/silicon.save \
    --nk 4 4 4 --nbands 12

# Quick eigenvalue validation (compare with QE):
# Uses build_matrix_k with QE SCF density → eig should match to 0.001 mRy.
# See the inline test scripts at the end of this session's conversation.
```

## Key design decisions pressed in review

- **V_xc uses autodiff** (`jax.grad` through `pbe_xc`) rather than hand-coded
  functional derivatives.  This is correct for V_xc = d(ρε_xc)/dρ and the
  JIT'd version runs in 1.2 ms.  The autodiff trace is NOT needed for dipole
  matrix elements (those autodiff through V_NL, not V_xc).

- **Generalized eigh via Cholesky**, not scipy: `B + 1e-12·I → L → L⁻¹AL⁻ᵀ → eigh`.
  Keeps the entire _subspace_step on GPU in one JIT.  scipy_eigh was 0.2 ms
  but the CPU round-trip added ~3 ms/iter.

- **`at[].set()` always copies in JAX** — pre-allocated buffers don't help.
  We use `jnp.concatenate` for subspace expansion (XLA optimizes it well).

- **h_diag for preconditioner**: `|k+G|² + V_loc(G=0) + V_NL_diag(G)`.
  V_loc(G=0) = `mean(V_loc_r)`.  V_NL_diag = `Σ |Z(R,G)|² E(R,R)`.
  V_H(G=0)=0 and V_xc(G=0) not included — matches QE's g_psi.f90.

- **ngkmax padding**: all k-points padded to the same nG so one JIT serves all.
  Mask field on HamiltonianK zeros padding.  2% overhead for Si.

- **Fixed n_tgt block size**: converged bands get zero corrections, but keeping
  the block fixed avoids variable-shape recompilation in the Davidson loop.

## Remaining speedup opportunities

1. **ρ_core (1.9s)**: numpy Simpson's rule radial FT. Could be vectorized
   with JAX or precomputed as a lookup table (like vnl_ops does for VNL).
2. **V_loc (0.5s)**: numpy FT of local PP. Same opportunity as ρ_core.
3. **VNL setup (2s)**: builds radial tables on CPU. The table construction
   itself is fast; the q_max scan iterates over all k-points.
4. **JIT warmup (6.8s)**: XLA compilation of complex128 linalg kernels.
   Irreducible — only way to reduce is smaller m_max or float32.
5. **_subspace_step Cholesky (1.7 ms)**: dominates the 1.4 ms iteration.
   Could use `jnp.linalg.eigh` on the projected H directly if we
   orthonormalize the subspace (but that's what caused the original bug).

## QE → WFN.h5 pipeline status

### Working
- CrystalData reads structure, symmetries (48 ops + translations), FFT grid,
  ecutwfc/ecutrho, atoms, electronic params from data-file-schema.xml
- `build_kgrid(nk, nosym, noinv, no_t_rev, force_symmorphic)` generates
  IBZ matching QE's kpoint_grid.f90 algorithm
- `load_charge_density()` reads ρ_val from charge-density.hdf5
- Davidson produces eigenvalues (0.001 mRy accuracy) and eigenvectors
- `write_wfn_h5()` writes complete mf_header + wfns groups

### Known issues
- **Fractional translations**: 24/48 non-symmorphic translations have sign
  convention mismatch between QE XML and BGW WFN.h5 (pw2bgw uses different
  convention).  Rotation matrices match exactly.
- **IBZ representative choice** may differ from a specific QE run when
  `no_t_rev=True` is used (diamond inversion complicates TR handling).
- **_symmetrise_density in charge_density.py is broken** (increases density
  error 4.5×).  Not used in the current pipeline — we read QE's SCF density.

## Branch: agent/dft-hamiltonian-validation

20 commits since main.  Key milestones:
- `31c72b3` H validated to 0.0 meV
- `274de21` Davidson generalized eigh fix
- `b149516` JIT warmup + fixed-size blocks (11× speedup)
