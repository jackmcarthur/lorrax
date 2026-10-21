# Si with spin–orbit coupling: BerkeleyGW and LORRAX on one mean field

Bulk Si, fully relativistic: noncollinear Quantum ESPRESSO with `lspinorb`, so
every state is a two-component spinor. One DFT run feeds two GW-BSE codes:

| stage | BerkeleyGW | LORRAX |
|---|---|---|
| ε / W | `epsilon.x` (plane waves, 10 Ry) | inside `gw.gw_jax` (ISDF, 996 centroids) |
| Σ, G₀W₀ | `sigma.x`, HL plasmon pole | `gw.gw_jax`, `compute_mode = hl_ppm` |
| QP bands on a path | `inteqp.x` | `bandstructure.htransform` |
| BSE (TDA, 10×10×10) | `kernel.x` + `absorption.x` | `bse.bse_jax` |

The DFT and BerkeleyGW inputs are BerkeleyGW's own spin–orbit example,
`DFT/silicon_SOC` of
[BerkeleyGW-examples](https://github.com/BerkeleyGW/BerkeleyGW-examples)
(commit `73d0f1f`), with keys that equal their defaults removed. Cutoffs, grids
and band counts are the example's: 25 Ry, 6×6×6 k (16 irreducible), 64 bands,
60-band χ₀ and Σ sums, Σ for bands 1–28.

```text
Si_soc/
  pseudo/          Si_FR.UPF (not included, see below)
  qe/              00-kgrid (kgrid.x inputs of the k lists), 01-scf, 02-wfn ... 06-wfnq_fi
  berkeleygw/      07-epsilon, 08-sigma, 09-inteqp, 10-kernel, 11-absorption
  lorrax/          gw.in (GW and BSE), ht.in (band path)
```

## Pseudopotential

`Si_FR.UPF`: fully relativistic Si, LDA (PZ), Troullier–Martins, made with
QE's `ld1.x`. Get it from BerkeleyGW-examples,
`DFT/pseudo/ld1/Si_FR.UPF`
(<https://raw.githubusercontent.com/BerkeleyGW/BerkeleyGW-examples/master/DFT/pseudo/ld1/Si_FR.UPF>),
and put it in `pseudo/`.

## 1. Quantum ESPRESSO

```bash
cd qe/01-scf
pw.x -in scf.in > scf.out
pw2bgw.x -in pw2bgw.in > pw2bgw.out          # RHO, for sigma.x
cd ..
for d in 02-wfn 03-wfnq 04-bands 05-wfn_fi 06-wfnq_fi; do
  mkdir -p $d/silicon.save
  cp 01-scf/silicon.save/data-file-schema.xml 01-scf/silicon.save/charge-density.* $d/silicon.save/
  (cd $d && pw.x -in nscf.in > nscf.out && pw2bgw.x -in pw2bgw.in > pw2bgw.out)
done
```

| directory | writes | read by |
|---|---|---|
| `02-wfn` | `WFN` (6×6×6, 64 bands), `vxc.dat` | both codes |
| `03-wfnq` | `WFNq` (q₀-shifted) | `epsilon.x` |
| `04-bands` | `WFN_bands` (the L–Γ–X–U\|K–Γ path) | `inteqp.x` |
| `05-wfn_fi`, `06-wfnq_fi` | `WFN_fi`, `WFNq_fi` (shifted 10×10×10) | `absorption.x` |

LORRAX needs only `01-scf` and `02-wfn`. The k lists in `nscf.in` are
`kgrid.x` output for the inputs in `00-kgrid`. Run `pw2bgw.x` on at most 4 MPI
ranks for this cell (more stops with "processes with no planes"). For spinors
the stock `pw2bgw.x` (QE 7.3–7.5) writes `vxc.dat` but not the G-space `VXC`
file, so `sigma.x` reads `vxc.dat`.

## 2. BerkeleyGW

Link each step's inputs into its directory, then run the executable there
(`epsilon.cplx.x`, `sigma.cplx.x`, `inteqp.cplx.x`, `kernel.cplx.x`,
`absorption.cplx.x`, in that order):

| step | links |
|---|---|
| `07-epsilon` | `WFN`, `WFNq` |
| `08-sigma` | `WFN_inner` → `02-wfn/WFN`, `RHO`, `vxc.dat`, `eps0mat.h5`, `epsmat.h5` |
| `09-inteqp` | `WFN_co` → `02-wfn/WFN`, `eqp_co.dat` → `08-sigma/eqp1.dat`, `WFN_fi` → `04-bands/WFN_bands` |
| `10-kernel` | `WFN_co`, `eps0mat.h5`, `epsmat.h5` |
| `11-absorption` | `WFN_co`, `WFN_fi`, `WFNq_fi`, `eps0mat.h5`, `epsmat.h5`, `eqp_co.dat`, `bsemat.h5` |

## 3. LORRAX

In `lorrax/`, from the same `02-wfn` run:

```bash
wfn2hdf.x BIN ../qe/02-wfn/WFN WFN.h5
cp ../qe/02-wfn/silicon.save/data-file-schema.xml .
sed 's/&/\&amp;/g' ../pseudo/Si_FR.UPF > Si_FR.UPF   # LORRAX reads UPF as strict XML

source $LORRAX/config/perlmutter/gpu_env.sh
RUN="srun -N 1 -n 4 --gpus-per-node=4 $LORRAX/src/ffi/cpp/select_gpu.sh $LORRAX/.venv/bin/python -u -m"
$RUN centroid.kmeans_cli 1000 -i gw.in      # 1. centroids_frac_996.txt
$RUN psp.get_dipole_mtxels -i gw.in         # 2. dipole.h5
$RUN gw.kin_ion_io -i gw.in                 # 3. kin_ion.h5
$RUN gw.gw_jax -i gw.in                     # 4. eqp0.dat, eqp1.dat, qp_wfn_rotations.h5, tmp/
$RUN bse.bse_jax -i gw.in --tda --bse --n-val 8 --n-cond 8 --n-eig 16 --eqp eqp1.dat
$RUN bandstructure.htransform -i ht.in --eqp-file eqp1.dat   # bandstructure.dat, as inteqp
```

`$LORRAX` is a LORRAX clone built for Perlmutter (`config/perlmutter/`); the
launch line is the one in the Quickstart. `centroid.kmeans_cli` keeps whole
symmetry orbits, so it writes fewer points than requested: set
`centroids_file` in both decks to the table it names.

`ld1.x` writes an unescaped `&INPUT` into `<PP_INFO>`; QE accepts it, LORRAX's
XML reader refuses the file ("not well-formed"), hence the `sed`.

### What the LORRAX keys do here

`gw.in` sets only what differs from the defaults or what BerkeleyGW's inputs fix:

| key | BerkeleyGW counterpart |
|---|---|
| `nval = 8`, `ncond = 20` | `sigma.inp` `band_index_min 1`, `band_index_max 28` |
| `number_bands = 60` | `number_bands 60` in `epsilon.inp` and `sigma.inp` |
| `compute_mode = hl_ppm` | `sigma.x` default `frequency_dependence 1` |
| `ppm_omega_p = 200.0` | none: LORRAX's HL fit reads the first moment at a real probe frequency, which must lie above every χ₀ transition (3.48 Ry here); 200 Ry is the f-sum limit |
| `use_band_extrapolation = false` | BerkeleyGW truncates the band sum |
| `bse_k_grid = 10 10 10` | `absorption.x`'s `WFN_fi` grid |
| `sys_dim = 3` | no truncation |

`ht.in` asks for bands 1–24 (`ncond = 16`): htransform fits 4 guard bands above
the returned window, and those must lie inside the GW QP bands (1–28). With
`--qp-rotations qp_wfn_rotations.h5` instead of `--eqp-file`, htransform
interpolates the full QP Hamiltonian rather than the diagonal eqp1 corrections.

## Expected results

Perlmutter, one node (4 A100), BerkeleyGW 4.0 (NERSC GPU module), QE 7.5,
LORRAX `94be1f1`. Energies in eV on the 6×6×6 grid; QP energies are eqp1.

| quantity | BerkeleyGW | LORRAX | LORRAX − BGW |
|---|---|---|---|
| DFT gap, indirect / direct at Γ | 0.628 / 2.517 | 0.628 / 2.517 | 0 |
| QP gap, indirect (Γ → X) | 1.270 | 1.231 | −39 meV |
| QP gap, direct at Γ | 3.247 | 3.193 | −54 meV |
| Σx at Γ, VBM / CBM | −12.700 / −5.869 | −12.680 / −5.861 | +20 / +9 meV |
| Σc(E_DFT) at Γ, VBM / CBM | 0.949 / −3.758 | 1.052 / −3.693 | +103 / +65 meV |

On the L–Γ–X–U|K–Γ path (inteqp against `htransform --eqp-file eqp1.dat`,
each aligned to its VBM) the top valence bands agree to +10–14 meV on average
and the lowest conduction bands to −44 to −67 meV; the lowest valence band
(Γ₆ at −12 eV) is 0.18 eV higher in LORRAX. `--qp-rotations` interpolates the
eigenvalues of the full QP Hamiltonian, T + V_ion + V_H + Σ(E_DFT), without the
Z factor of eqp1, so its gaps are larger (about eqp0's).

The two codes do not compute the same Σc here, by construction:

- **The HL first moment.** BerkeleyGW's GPP takes it from the electron density
  (the f-sum rule); LORRAX's HL fit reads it from the 60-band χ₀ at
  `ppm_omega_p`.
- **The screening basis.** BerkeleyGW cuts ε⁻¹ at 10 Ry; LORRAX builds W in the
  996-point ISDF basis from the same V_q as Σx (25 Ry).
- **The q → 0 head and mini-BZ averages** follow each code's defaults.

Lowest excitons (TDA, eV):

| | BerkeleyGW | LORRAX, its own eqp1 | LORRAX, BerkeleyGW's eqp1 |
|---|---|---|---|
| S1 | 3.237 | 3.125 | 3.179 |
| S9 | 3.273 | 3.162 | 3.216 |

The fine grids differ: `absorption.x` reads the shifted `WFN_fi`, which has no
Γ point, while `bse_k_grid = 10 10 10` is unshifted and contains Γ, where the
direct gap is. On BerkeleyGW's own QP energies LORRAX's S1 lies 68 meV below
the Γ transition (3.247 eV); BerkeleyGW's lies 10 meV below it, at a k of the
shifted grid. The BerkeleyGW column is the upstream example's own absorption
output: this calculation's kernel reproduces its Frobenius norms to 6 digits,
but its `absorption.x` did not finish on one node. Full diagonalization needs
~310 GB (upstream used 20 nodes), and PRIMME (`diagonalization_primme`) did not
converge on GPU or CPU.

Wall times on the one node: BerkeleyGW epsilon 25 s, sigma 133 s, inteqp 5 s,
kernel 450 s; LORRAX centroids 26 s, dipole 30 s, kin-ion 10 s, GW 81 s,
BSE 88 s, htransform 30 s.
