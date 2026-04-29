# BSE comparison cookbook — LORRAX vs BGW

**This document is for an agent who must produce a fair LORRAX-vs-BGW
absorption comparison and has never done one.**

Reading STATUS.md alone has bitten multiple agents (including the one
writing this) — they tried "obvious" comparisons and got 10×–π² wrong
answers on Σ\|d\|² and absorption peak heights. Every wrong answer
traced to violating one of the conventions below. **If you skip a
section here you will misread the result by O(1) factors and not
notice.** Read all six conventions before running anything.

---

## 0. What "fair comparison" means here

`absorption_eh.dat` (BGW) and the LORRAX `absorption_*_b1_eh.dat`
output should be plottable on the same axes and overlap within a few
percent at the E1 peak (~3.2–3.4 eV for Si). The validated example
lives at:

- BGW: `runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/absorption_eh.dat`
  (peak ε₂ ≈ 144 at ω = 3.23 eV, η = 0.15 eV, full diag with QP)
- LORRAX Haydock 100-iter: `runs/Si/04_si_4x4x4_bse/C_lorrax_bse_bgweqp/`
  (`eps2_8x8_haydock_compare.png` shows the overlap)

**Use Haydock as the primary comparison route.** The eigenvector route
also works but converges the peak height *slowly* with `n_eig`
(n=100 captures ~18% of full peak; Haydock at n=100 captures ~100%
because the continued fraction implicitly resums all moments).

---

## 1. The six conventions you cannot get wrong

These are independent. **Each one wrong = silent O(1) error.** Get all
six right and the spectra agree within ~3% peak.

### 1.1 Dipole operator must match BGW's `use_momentum`

| Variant | BGW | LORRAX command | Output file |
|---|---|---|---|
| Bare p̂ | `use_momentum` | `psp.get_dipole_mtxels --skip-vnl` | `dipole_p_only.h5` |
| p̂ + i[r̂, V_NL] | (not BGW's default) | `psp.get_dipole_mtxels` (default) | `dipole.h5` |

> **Empirical**: BGW's `use_momentum` keyword corresponds to **bare p̂**,
> *despite the BGW manual saying it includes the V_NL commutator*. The
> only LORRAX dipole that actually matches BGW vmtxel is the
> `--skip-vnl` one.
>
> If you compare LORRAX default `dipole.h5` (with V_NL) to BGW vmtxel
> you get a per-element magnitude scatter with median ratio ≈ 0.28 and
> a ~10× factor on Σ\|d\|². This is *not* a LORRAX dipole bug — it's a
> dipole-operator mismatch. Use `dipole_p_only.h5`.

```bash
# Always do this once before any LORRAX absorption run:
LORRAX_NGPU=1 lxrun python3 -u -m psp.get_dipole_mtxels \
    -i cohsex.in --skip-vnl --out dipole_p_only.h5
```

### 1.2 QP corrections — use BGW's eqp.dat, not LORRAX's eqp0.dat

BGW's `absorption.x` reads QP-corrected energies from `eqp.dat`
(produced by BGW's sigma run). For a fair comparison pass that *same*
file to LORRAX BSE via `--eqp`, **even when LORRAX has its own
`eqp0.dat` from a prior LORRAX-side GW run**:

```bash
--eqp /path/to/<bgw_run>/eqp.dat
```

Mixing eqp files: LORRAX with its own eqp0.dat gives a different gap
(typically 50–200 meV off from BGW's eqp.dat depending on
self-consistency / head conventions). The BSE eigenvalues differ
correspondingly.

### 1.3 Head injection (vhead, whead) must use BGW's wcoul0

LORRAX's standard `compute_vcoul` zeroes the G=G'=0 element of v(q=0).
For BSE this matters: BGW reinstates a mini-BZ-averaged 1/q² head
(`wcoul0` in BGW debug output). Match by reading the same value into
LORRAX:

```ini
# in cohsex.in
use_bgw_vcoul = true
bgw_vcoul_file = /path/to/<bgw_eps_run>/vcoul
```

For the validated Si setup the head is `vhead = 3303.748`. You will
see `BSE: q=0 head injected ... vhead=3303.748, whead[0]=...` printed
during the BSE solve.

### 1.4 Band-count convention — SOC counts differently in LORRAX vs BGW

| | BGW `number_*_bands_coarse` | LORRAX `--n-val` / `--n-cond` |
|---|---|---|
| **Non-spinor / spinless** | N single-particle bands | N single-particle bands (1:1 match) |
| **SOC (spinor)** | N single-particle bands (= N/2 Kramers pairs) | N **Kramers pairs** (= 2N single-particle bands) |

> **Concrete Si SOC example**: BGW `number_val_bands_coarse 8` and
> LORRAX `--n-val 4` describe the *same* 8 SP valence states (= 4
> Kramers pairs × 2 spinor components). The LORRAX-printed BSE problem
> dimension `4 × 4 × 64 = 1024` corresponds to the BGW 8×8×64 = 4096
> single-particle BSE space — the spinor expansion is implicit in the
> matvec.

Today after the `agent-C/bse-band-slicing-fix` commit, LORRAX `--n-val 8
--n-cond 8` *also* works (8 SP bands directly), giving matvec sig
`(m, 8, 8, 64)`. **Both forms produce equivalent physics; just don't
mix them within one run.** Pick one and stick to it.

### 1.5 n_occ (Fermi-band index) — explicit or via WFN.h5

`load_bse_data_from_restart_sharded` resolves `n_occ` in this order:

1. Explicit `--n-occ <N>` flag (or `n_occ=` kwarg) — used as-is.
2. `input_file=` (cohsex.in, with `wfn_file = WFN.h5`) → reads
   `mf_header/kpoints/ifmax` directly. Authoritative.
3. Explicit `fermi_energy=` Ry hint.

**No silent auto-detect.** If you call without `n_occ`/`input_file`
the loader raises with a clear message.

> Pre-fix bug: prior loader used `mean_enk < 0.0` as the auto-detect.
> This silently returned only deepest-semicore states for any
> pseudopotential whose VBM sat above zero (Si UPF: VBM = +4.4 eV →
> heuristic returned n_occ=2 instead of 8, BSE landed at -0.23 eV
> instead of +2.93 eV). All callers under `src/bse/` now thread
> `input_file=args.input` through to the loader.

In the SOC convention of §1.4, **n_occ counts SP states** (n_occ=8 for
Si SOC), not Kramers pairs.

### 1.6 Polarization, broadening, iteration count — must literally match

BGW's `absorption.inp` and the LORRAX BSE / absorption invocation must
agree on these scalars:

| BGW | LORRAX | Comment |
|---|---|---|
| `polarization 1.0 0.0 0.0` | `b1_eh.dat` (alpha=0 column from dipole_cart) | Cartesian x; b1 in BGW filename suffix |
| `energy_resolution 0.15` | `--eta-eV 0.15` | Lorentzian half-width η |
| `number_iterations 100` | `--n-iter 100` | Haydock iteration count must match |

Different broadenings give visually similar but quantitatively
different curves (peak heights vary by 30%+ between η=0.05 and 0.15).
Different Haydock iteration counts behave similarly.

---

## 2. End-to-end command sequence (Si SOC 8×8 example)

This is the exact recipe that produces `eps2_8x8_haydock_compare.png`
in the validated run dir. Adapt paths but **do not skip steps or change
the conventions in §1**.

### 2.1 Prerequisites — a directory containing

```
<run_dir>/
    cohsex.in              # LORRAX BSE input (see §3 for the BSE-mode flags)
    centroids_frac_*.txt    # ISDF centroids
    WFN.h5                 # QE wavefunction (the one cohsex.in's wfn_file points to)
    tmp/isdf_tensors_*.h5  # canonical restart bundle from a prior gw_jax pass
    eqp.dat                # BGW eqp.dat from sigma.x (NOT lorrax eqp0.dat)
```

You also need a BGW reference dir (`<bgw_run>/`) with `absorption_eh.dat`
to plot against.

### 2.2 Step 1 — produce the bare-p dipole

```bash
LORRAX_NGPU=1 lxrun python3 -u -m psp.get_dipole_mtxels \
    -i cohsex.in --skip-vnl --out dipole_p_only.h5
```

Sanity: `dipole_p_only.h5` has `attrs.skip_vnl = True` and the note
attribute mentions "V_NL skipped". Total Σ\|d\|² in the (n_val, n_cond)
window will match BGW's vmtxel binary (≈2314 Ry² for Si 8×8).

### 2.3 Step 2 — Haydock absorption (preferred route)

```bash
LORRAX_NGPU=4 lxrun python3 -u -m bse.absorption_haydock \
    -i cohsex.in \
    --n-val 4 --n-cond 4         # SOC: 4 Kramers pairs each → matches BGW number_*_bands_coarse 8
    --n-occ 8                     # 8 SP occupied bands
    --eqp <bgw_run>/eqp.dat       # BGW QP corrections
    --dipole dipole_p_only.h5    # bare-p dipole from §2.2
    --V-cell 270.107              # Si 4×4×4 cell volume (bohr³); read from data-file-schema.xml
    --n-iter 100                  # match BGW number_iterations
    --eta-eV 0.15                 # match BGW energy_resolution
    --no-eps1                     # skip Kramers-Kronig (for speed)
    --out-prefix absorption_haydock
```

Output: `absorption_haydock_b1_eh.dat`, `_b2_eh.dat`, `_b3_eh.dat`
(one file per Cartesian polarization). Compare `_b1_eh.dat` to BGW's
`absorption_eh.dat` — these should overlap within ~3% at the E1 peak.

### 2.4 Step 2 (alternative) — eigenvector route

Only use if you need per-state oscillator strengths. The peak height
**will not match BGW** unless n_eig is comparable to the full BSE
dimension (4096 for Si 8×8). Use Haydock for normal comparisons.

```bash
# First run BSE Lanczos to get eigenvectors.h5 (BGW-compliant format)
LORRAX_NGPU=4 lxrun python3 -u -m bse.bse_jax \
    -i cohsex.in --eqp <bgw_run>/eqp.dat \
    --bse --lanczos --tda \
    --matvec-kind=simple \
    --n-val 4 --n-cond 4 --n-occ 8 \
    --n-reorth -1                 # full reorth — essential for spinor BSE
    --max-lanczos-iter 2400 --n-eig 100 \
    --px 2 --py 2 \
    --write-eigs 100              # write eigenvectors.h5

# Then run absorption_eigvecs
LORRAX_NGPU=1 lxrun python3 -u -m bse.absorption_eigvecs \
    --eigenvectors eigenvectors.h5 \
    --dipole dipole_p_only.h5 \
    --n-occ 8 --V-cell 270.107 --eta-eV 0.15 --no-eps1 \
    --out-prefix absorption_eigvec
```

Output: `absorption_eigvec_b1_eh.dat` etc.

For a fair comparison at finite n_eig, build a *truncated* BGW
spectrum from `eigenvalues.dat` truncated to the same n_eig, with the
same η. The peaks will then align to ~0.2% (the validated 25.96 vs
26.02 datapoint).

### 2.5 Step 3 — compare the spectra

```python
# Quick comparison snippet
import numpy as np, matplotlib.pyplot as plt
bgw   = np.loadtxt('<bgw_run>/absorption_eh.dat',          comments='#')
lor_h = np.loadtxt('absorption_haydock_b1_eh.dat',         comments='#')
plt.plot(bgw[:,0],   bgw[:,1],   label='BGW')
plt.plot(lor_h[:,0], lor_h[:,1], label='LORRAX (Haydock)')
plt.xlabel('ω (eV)'); plt.ylabel('ε₂(ω)'); plt.legend()
plt.savefig('compare.png')
```

For Si 8×8 with all conventions correct: peak ε₂ ≈ 144 (BGW) vs ≈ 146
(LORRAX Haydock 100), peak position both at ω ≈ 3.2–3.3 eV.

---

## 3. cohsex.in — required entries for BSE mode

Copy from `runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/cohsex_bse.in`
as a template. Key entries:

```ini
[cohsex]
restart = true                  # use isdf_tensors_*.h5 from prior gw_jax run
centroids_file = centroids_frac_<N>.txt

nval = 8                         # SP-band counts (matches BGW number_*_bands_coarse)
ncond = 52
nband = 60

bispinor = false                 # leave false for SOC

x_only = false                    # screened (BSE-mode), not COHSEX-x
do_screened = true

use_bgw_vcoul = true
bgw_vcoul_file = /path/to/<bgw_eps_run>/vcoul

bare_coulomb_cutoff = 25.0       # match BGW screened_coulomb_cutoff
fermi_reference = midgap

wfn_file = WFN.h5                # this one is critical for n_occ resolution
```

---

## 4. Common mistakes — every one of these has bitten an agent

Avoid each by re-reading the linked section.

| Mistake | Symptom | Fix |
|---|---|---|
| Using `dipole.h5` (default) instead of `dipole_p_only.h5` | Σ\|d\|² off by ~10× vs BGW; absorption peak ~28× too small | §1.1 |
| LORRAX's own `eqp0.dat` instead of BGW's `eqp.dat` | BSE eigvals 50–200 meV off | §1.2 |
| No `bgw_vcoul_file` | BSE eigvals shifted by ~10–100 meV; head divergent | §1.3 |
| `--n-val 8` for SOC when also passing `--n-occ 4` | Loader silently truncates or errors; BSE in wrong space | §1.4 |
| No `--n-occ` and no `input_file` resolves to a valid WFN.h5 | (post-fix) loader raises ValueError; (pre-fix) silent wrong slicing | §1.5 |
| Comparing LORRAX Lanczos n=100 to BGW full diag | LORRAX peak ~18% of BGW (real, not a bug — finite-Krylov truncation) | Use Haydock instead, §2.3 |
| Mismatched η between LORRAX and BGW | Peak heights look 30%+ different | §1.6 |
| Comparing raw Σ\|d\|² across codes | "10× off" results that vanish at the ε₂ level | Compare ε₂(ω), not raw matrix-element sums |
| Reading STATUS.md "Σ\|d\|² match" as referring to dipole.h5 | Confusion when 01_lorrax_bse_vcoul/dipole.h5 doesn't match | The matching file is `dipole_p_only.h5` (`--skip-vnl`) |

---

## 5. Sanity checks after a run — verify before you trust the plot

Run all of these. Each takes < 1 s.

```python
import numpy as np, h5py, struct

# 5.1 — dipole file is the bare-p one
with h5py.File('dipole_p_only.h5','r') as f:
    assert bool(f.attrs.get('skip_vnl', False)), \
        "dipole.h5 has V_NL term — use --skip-vnl version (see §1.1)"

# 5.2 — vmtxel sum matches LORRAX dipole sum (gauge invariant)
raw = open('<bgw_run>/vmtxel','rb').read()
n_hdr = struct.unpack('<i', raw[:4])[0]
hdr = struct.unpack(f'<{n_hdr//4}i', raw[4:4+n_hdr])
data = np.frombuffer(raw[4+n_hdr+4+4:-4], dtype=np.complex128)
v_bgw = data.reshape(hdr[0], hdr[1], hdr[2], hdr[3])[..., 0]
sum_bgw = float(np.sum(np.abs(v_bgw)**2))
# LORRAX side: full ε₂ pipeline — don't compare raw |d|² here, units differ.
# Just confirm the BGW number is what STATUS.md says (~2314 for Si 8×8).
print(f'BGW Σ|vmtxel|² = {sum_bgw:.4e}  (Si 8×8 reference = 2314.18)')

# 5.3 — BSE eigvals land near BGW
bgw_eig = np.loadtxt('<bgw_run>/eigenvalues.dat', comments='#', usecols=0)
print(f'BGW lowest 5 eigvals (eV): {bgw_eig[:5]}')
# LORRAX Lanczos / Davidson should give first eigenvalue within
# ~3 meV (ISDF compression floor).

# 5.4 — Haydock peak agreement
bgw_eh = np.loadtxt('<bgw_run>/absorption_eh.dat', comments='#')
lor_eh = np.loadtxt('absorption_haydock_b1_eh.dat', comments='#')
peak_b = bgw_eh[:,1].max()
peak_l = np.interp(bgw_eh[bgw_eh[:,1].argmax(),0], lor_eh[:,0], lor_eh[:,1])
print(f'BGW peak  ε₂ = {peak_b:.2f}, LORRAX(at BGW peak ω) = {peak_l:.2f}, '
      f'ratio = {peak_l/peak_b:.3f}  (good if 0.95 < ratio < 1.05)')
```

If 5.4 fails, return to §1 and re-check every convention.

---

## 6. What this guide does NOT cover

- Non-Si systems: the conventions are universal but the head value
  (`vhead`) and band counts change. Read the BGW eqp.dat and the
  pseudopotential to establish the right `n_occ`.
- Block-Lanczos: requires `--block-size > 1` and `--n-reorth -1` for
  spinor BSE; otherwise gives ghost eigenvalues.
- Triplet vs singlet kernel comparison: BGW `spinor` flag is implicit
  in singlet via Hartree term sign. LORRAX takes `spin_kernel`-equivalent
  via `do_screened` plus other flags — see STATUS.md.
- Validating individual `(c,v,k)` matrix elements: gauge-dependent at
  the per-element level. Compare Σ-summed quantities only (Σ\|d\|² in
  the same units, manifold-summed oscillators, ε₂(ω) — all gauge-
  invariant by construction).

For the underlying decisions and history, see [STATUS.md](STATUS.md).
