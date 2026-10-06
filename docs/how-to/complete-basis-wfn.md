# A complete-basis WFN from the dense H_k

`psp.run_dense_h` rebuilds the Kohn–Sham Hamiltonian of a QE run as a dense
matrix on each k's whole plane-wave sphere and diagonalizes it completely.
Its drop-in rectangular `WFN.h5` retains the smallest native dimension at
every k. That band sum is complete only when every native dimension is
equal. `--spectrum-output PATH` additionally preserves all eigenpairs in
the source-bound ragged native reference archive; that archive is not a
production WFN replacement.

## What it builds

For each k of the source `WFN.h5`:

$$H_k(sG, s'G') = |k+G|^2\,\delta\delta' + V_\mathrm{scf}(G-G')\,\delta_{ss'} + V_\mathrm{NL},\qquad |k+G|^2 \le E_\mathrm{wfc},$$

with $V_\mathrm{scf} = V_\mathrm{loc} + V_H[\rho] + V_{xc}[\rho + \rho_\mathrm{core}]$
from the SCF density of the `.save` (`psp.scf_potential.build_dft_potentials`).
On a noncollinear magnetic run $V_{xc} = v\,\delta + B\cdot\sigma$ from ρ and m
(`psp.xc.compute_V_xc_noncollinear`), in QE's general noncollinear GGA branch,
with gradients kept on the density sphere $|G|^2 \le E_\mathrm{rho}$.

The PBE registry supplies separate exchange and correlation components to
this magnetic potential owner. Exchange gradient corrections use each
spin-density channel; correlation uses the total charge gradient. A varying
magnetization can therefore produce an exchange field even at uniform
charge. The public combined polarized PBE callable still uses the same
generated kernels. The scalar potential route is unchanged. Analytic and
directional derivative controls test this gate, while a particular native
QE density still needs its own matched operator/FFT/core check.

- **Operator.** `psp.dft_operators.dense_matrix_k` applies
  `apply_H_k_batched`, the Davidson route's operator, to the unit basis. H is
  not assembled a second time.
- **One shape.** Every k is padded to `ngkmax`. Pad rows carry a diagonal above
  the Gershgorin bound of the physical block plus 1 Ry, so the lowest
  `nspinor·ngk` eigenpairs are the physical ones. A Hermiticity skew
  max|H − H^H| above 1e-12 of the pad diagonal, or a physical eigenvector with
  weight above 1e-10 on the pad, refuses.
- **Eigensolve.** `distrib_la.plan("eigh", single_device_mesh(), backend="off")`:
  a local eigh of one whole k per rank. In round r, rank p solves k = r·P + p.
  Nothing is distributed inside one k.
- **Output.** `file_io.qp_wfn.write_complete_wfn_h5` writes
  `min_k nspinor·ngk(k)` bands through SlabIO. The file keeps the source's
  k-set, symmetry, G-lists and occupations (zero past the source's last band).
  The root carries `lorrax_complete_basis_wfn` only if all native dimensions
  equal the written extent; otherwise the crop is explicitly recorded,
  `dense_h_source_wfn` and `dense_h_qe_save`.
- **Native reference.** `--spectrum-output` writes every native eigenpair,
  paired source G lists, per-k dimensions and residual/orthogonality checks,
  source SHA bindings and finalization/commit guards. The reference reader
  retains an explicit per-k band-validity mask when constructing a padded
  numerical carrier. Ghost coefficients and occupations must be zero.
- **One calculation.** `CrystalData.validate_against_wfn` checks the `.save`
  against the WFN header (cell, atoms, electron count, spinors, FFT grid,
  symmetries), and each k's ecutwfc sphere must hold the WFN's `ngk`.

## Requirements

- **Patched `jax_xc`.** `psp.xc` imports `jax_xc` for PBE. The sealed runtime
  has no `jax_xc`, so `psp.run_dense_h` and `psp.run_nscf` need the
  patched install of [`config/xc/README.md`](../../config/xc/README.md) on the
  import path.
- **Inputs.** The QE `.save` with `charge-density.hdf5`, the UPF files the QE
  run used, and the source `WFN.h5` of the same run.

## Invoke

One rank per GPU:

```bash
python3 -u -m psp.run_dense_h --save QE.save \
    --wfn WFN.h5 -o WFN_complete.h5 --sys-dim 3
```

| flag | default | meaning |
|---|---|---|
| `--save` | required | QE `.save` directory |
| `--wfn` | required | source `WFN.h5` |
| `-o` / `--output` | required | output `WFN.h5` |
| `--sys-dim` | required | 0, 2 or 3; 2 must match QE `assume_isolated = '2D'` |
| `--pseudo-dir` | the `.save` directory | where the UPF files are |
| `--nbands` | the complete basis, `min_k nspinor·ngk` | bands written; larger refuses |
| `--memory-per-device-gb` | the detected budget ([memory model](../architecture/memory-model.md#budget)) | the budget the `memory` rule checks |
| `--nc-gga-branch` | unset | `general` or `fixed_axis`; a magnetic run must pass `general` |

## Refusals

`psp.operator_checks.validate_dense_h_inputs` runs before any heavy work. Each
refusal is a `DenseHRefusal` printed as `[dense_h:<rule>]`.

| rule | condition |
|---|---|
| `upf_missing` | a species' UPF is not in the pseudopotential directory |
| `xc_extension` | the QE run has a hybrid, vdW or +U term |
| `functional` | the QE functional is not PBE (LDA included), or a UPF was generated with another functional |
| `pseudo_type` | a UPF is not norm-conserving (USPP, PAW) |
| `magnetism` | collinear `nspin = 2`: H_k and the writer carry one spin channel |
| `nc_gga_branch` | a noncollinear magnetic run without `--nc-gga-branch general`. The `.save` does not record QE's branch. The fixed-axis branch needs QE's `ux`, which the `.save` does not keep |
| `charge_density` | no `rhotot_g` in `charge-density.hdf5`; on a magnetic run, a missing `m_x`, `m_y` or `m_z` |
| `truncation_2d` | `--sys-dim 2` without QE `assume_isolated = '2D'`, or the reverse |
| `memory` | $5 \cdot N^2 \cdot 16$ B above the device budget, $N = $ `nspinor · max_k ngk` |

A slab run with matching truncation passes the preflight; none has been measured.

## Cost

The `memory` rule prices five complex128 N×N matrices, $80 N^2$ B with
$N$ = `nspinor · max_k ngk`, the padded extent every k is solved at
(`psp.operator_checks.dense_h_bytes`). That is a price, not the peak: the
measured peaks below are 3.4–14× it. The eigh is $O(N^3)$ per k. Ranks take whole k-points, so the wall scales as
$\lceil n_k/P\rceil N^3$.

Measured on one node, 4 A100-40GB (claim 2865):

| | Si 4³ scalar | Si 4³ SOC | Fe SOC, magnetic |
|---|---|---|---|
| N per k | 537–588 | 1074–1176 | 1542–1596 |
| max \|Δε\| against QE | 2.9e-7 Ry | 4.5e-7 Ry below QE's top 8 bands | 5.7e-7 Ry |
| wall per k, first round / later | 3.1 / 0.7 s | 2.1 / 0.18 s | 3.3 / 0.1–0.5 s |
| device peak per rank | 0.38 GB | 0.38 GB | 0.78 GB |

The uniform residual (2.8e-7 Ry on Si, 5.7e-7 Ry on Fe) matches the size of
the PW92 constant difference of `psp.xc` against QE (claim 2865; the difference
itself was measured on MoS2 at Γ, claim 2229). On the Fe run, 1 − subspace
weight is 1.14e-8 at one band (k 6), which fails the 1e-8 gate by 1.14×; the
claim attributes it to QE's Davidson threshold, ethr 1e-10 (claim 2865).

A rerun of Si 4³ scalar (537 bands, 8 k): max |Δε| 2.92e-7 Ry, 1 − subspace weight 5.1e-12. A gwjax
`x_only` run on the dense Si scalar WFN (34 bands, 368 centroids) reproduces
Σ_x of the QE-WFN run within 2 µeV per band (claim 2865).

## When to use it

- **Owner ruling, 2026-09-28:** the complete basis is the default band count
  where it fits. `--nbands` defaults to it. A GW deck still sets `number_bands`
  itself ([input reference](../input_reference.md)).
- An all-band Σ_x needs an ISDF basis about as large as the plane-wave density
  basis: [ISDF exchange accuracy, out to the complete basis](../theory/isdf-exchange-accuracy.md#out-to-the-complete-basis).
- **Blocked today:** the kmeans pruner refuses a band window whose top band
  exceeds half the plane-wave basis, `0.5 · ngk_max · nspinor`
  (`centroid/pivoted_cholesky.py:1043`,
  `prune_candidates_by_pivoted_cholesky`). Pivoted-Cholesky pruning on a
  complete-basis pair set therefore refuses; `--oversample 1.0` skips pruning.
