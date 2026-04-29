# BSE comparison cookbook — LORRAX vs BGW

**For an agent producing a fair LORRAX-vs-BGW absorption comparison.**
Skipping any of the six conventions below produces silent O(1) errors.

## Six conventions you cannot get wrong

1. **Dipole operator.** BGW `use_momentum` corresponds to bare p̂
   *despite the manual implying V_NL is included*. Use LORRAX
   `psp.get_dipole_mtxels --skip-vnl` → `dipole_p_only.h5`. Default
   `dipole.h5` (with V_NL) mismatches BGW vmtxel by ~10× on Σ\|d\|².

2. **QP corrections.** Pass BGW's `eqp.dat` via `--eqp`, **not**
   LORRAX's own `eqp0.dat` (different gap, 50–200 meV off).

3. **Head injection.** Set `use_bgw_vcoul = true` and
   `bgw_vcoul_file = <bgw_eps_run>/vcoul` in `cohsex.in`, otherwise the
   q=0 head shifts eigenvalues by 10–100 meV.

4. **SOC band counting.** BGW `number_*_bands_coarse` counts SP bands;
   LORRAX `--n-val/--n-cond` historically counts Kramers pairs. For Si
   SOC: BGW 8 ↔ LORRAX 4. Post-`agent-C/bse-band-slicing-fix`,
   LORRAX `--n-val 8 --n-cond 8` *also* works (same physics).
   Pick one, don't mix within a run.

5. **n_occ resolution.** Pass `--n-occ <N>` explicitly OR rely on
   `cohsex.in[wfn_file]` → WFN.h5 `ifmax` (post-fix default). No
   silent auto-detect — loader raises if neither is available.
   For SOC, n_occ counts SP states (Si SOC: 8).

6. **Polarization, η, n_iter.** Must literally match BGW's
   `absorption.inp`. BGW polarization `1.0 0.0 0.0` ↔ LORRAX `b1`
   output. BGW `energy_resolution` ↔ LORRAX `--eta-eV`. BGW
   `number_iterations` ↔ LORRAX `--n-iter`.

## End-to-end recipe (Si SOC 8×8, validated against BGW)

Prereq: a run dir with `cohsex.in`, `centroids_frac_*.txt`, `WFN.h5`,
`tmp/isdf_tensors_*.h5` from a prior `gw_jax` pass, plus a BGW
reference dir with `eqp.dat`, `vcoul`, `absorption_eh.dat`,
`eigenvalues.dat`. See `runs/Si/04_si_4x4x4_bse/` for examples.

```bash
# 1) bare-p dipole (matches BGW use_momentum)
LORRAX_NGPU=1 lxrun python3 -u -m psp.get_dipole_mtxels \
    -i cohsex.in --skip-vnl --out dipole_p_only.h5

# 2a) Haydock route — apples-to-apples vs BGW Haydock; fast peak convergence
LORRAX_NGPU=4 lxrun python3 -u -m bse.absorption_haydock \
    -i cohsex.in --n-val 4 --n-cond 4 --n-occ 8 \
    --eqp <bgw_run>/eqp.dat --dipole dipole_p_only.h5 \
    --V-cell <V_bohr3> --n-iter 100 --eta-eV 0.15 --no-eps1 \
    --out-prefix absorption_haydock

# 2b) Eigenvector route — only if you need per-state oscillators.
#     Peak height converges SLOWLY with n_eig (n=100 ≈ 18% of full peak).
#     For fair vs-BGW comparison: use eigvals_to_eps2.py with matched n_max.
LORRAX_NGPU=4 lxrun python3 -u -m bse.bse_jax \
    -i cohsex.in --eqp <bgw_run>/eqp.dat \
    --bse --lanczos --tda --matvec-kind=simple \
    --n-val 4 --n-cond 4 --n-occ 8 --n-reorth -1 \
    --max-lanczos-iter 2400 --n-eig 100 --px 2 --py 2 --write-eigs 100
LORRAX_NGPU=1 lxrun python3 -u -m bse.absorption_eigvecs \
    --eigenvectors eigenvectors.h5 --dipole dipole_p_only.h5 \
    --n-occ 8 --V-cell <V_bohr3> --eta-eV 0.15 --no-eps1 \
    --out-prefix absorption_eigvec
```

## Comparing spectra at custom η / matched n_eig

Both BGW (`absorption.x`) and LORRAX (`absorption_eigvecs`) write a
common-format `eigenvalues.dat` (or `eigenvalues_b1.dat` per
polarisation): `# neig`, `# vol`, `# nspin, nspinor` header followed
by `(E_eV, |d|², Re d, Im d)` rows. **Use `bse.eigvals_to_eps2` to
broaden either at any η and any truncation:**

```bash
python3 -m bse.eigvals_to_eps2 \
    --files <bgw>/eigenvalues.dat <lorrax>/eigenvalues_b1.dat \
    --eta-eV 0.05 --n-max 100 \
    --label "BGW (100)" "LORRAX (100)" \
    --out cmp.png
```

Calibration: at η = 0.20 with all 500 states, the script reproduces
BGW's `absorption_eh.dat` peak (143.7) to within 0.4%. At matched
`n_max`, the LORRAX/BGW peak ratio reflects the real per-state
oscillator-strength agreement (eigenvector convergence + ISDF
compression).

## Common mistakes — every one has bitten an agent

| Mistake | Symptom | Fix |
|---|---|---|
| Default `dipole.h5` not `dipole_p_only.h5` | Σ\|d\|² off ~10×; absorption peak ~28× too small | §1 |
| LORRAX's `eqp0.dat` not BGW's `eqp.dat` | BSE eigvals 50–200 meV off | §2 |
| Missing `bgw_vcoul_file` | Eigvals shift 10–100 meV | §3 |
| Mixing SOC band conventions (`--n-val 8` + `--n-occ 4`) | Loader truncates / wrong space | §4 |
| Comparing LORRAX Lanczos eigvec n=N to BGW full-diag truncated to N | LORRAX peak ~18% of BGW at N=100, ~50% at N=400 (real, not a bug — Lanczos Ritz vectors at finite N ≠ exact lowest-N eigenvectors). **This bites every time. Use Haydock instead.** | Haydock route in §2a |
| Mismatched η | Peak heights look 30%+ off | §6 |
| Comparing raw Σ\|d\|² across codes | "10× off" results that vanish at ε₂ level | Compare ε₂(ω), not raw \|d\|² |

## Quick sanity checks (each <1 s)

```python
import h5py, numpy as np, struct
# 1. Dipole is the bare-p one
with h5py.File('dipole_p_only.h5','r') as f:
    assert bool(f.attrs.get('skip_vnl', False))
# 2. BGW vmtxel sum sanity (Si 8×8 reference: 2314.18)
raw = open('<bgw_run>/vmtxel','rb').read()
n = struct.unpack('<i', raw[:4])[0]; hdr = struct.unpack(f'<{n//4}i', raw[4:4+n])
v = np.frombuffer(raw[4+n+4+4:-4], dtype=np.complex128).reshape(*hdr[:4])[..., 0]
print(f'Σ|vmtxel|² = {np.sum(np.abs(v)**2):.2f}  (Si 8×8 ref: 2314.18)')
# 3. Haydock peak ratio (good: 0.95 < ratio < 1.05)
bgw = np.loadtxt('<bgw_run>/absorption_eh.dat', comments='#')
lor = np.loadtxt('absorption_haydock_b1_eh.dat', comments='#')
print(f'peak ratio LORRAX/BGW = {lor[:,1].max()/bgw[:,1].max():.3f}')
```

For history and project status, see [STATUS.md](STATUS.md).
