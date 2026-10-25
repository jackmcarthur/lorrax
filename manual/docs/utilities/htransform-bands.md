# Band-structure interpolation

`bandstructure.htransform` Band energies along a path in the Brillouin zone, coloured by spin, orbital character or orbital moment.
{: .page-summary }

The standalone [H-transform utility](htransform.md) interpolates the bands of a coarse-grid calculation to a `K_POINTS {crystal_b}` path. The method, the band window and the keys shared with the fine-grid route are on the [overview](htransform.md). This page covers the path, its output and the band-character colouring.
{: .lead }

## Running {: #run }

=== "Local"
    ```bash
    uv run python -m bandstructure.htransform -i ht.in
    ```
=== "Perlmutter"
    ```bash
    srun -n 4 python -m bandstructure.htransform -i ht.in
    ```
=== "Frontera"
    ```bash
    # config/frontera/templates/gw_dev.sbatch, launch line changed to -m bandstructure.htransform
    sbatch ht.sbatch
    ```

QSGW bands: add `-wfn WFN_qp.h5` (the SC run's QP WFN) or `--qp-rotations qp_wfn_rotations.h5`. Diagonal one-shot QP energies: `--eqp-file eqp1.dat`. Relative output paths resolve beside the input file.

A minimal input file is a GW input file whose `nval` equals the number of occupied bands, plus a path:

```ini
[cohsex]
wfn_file       = WFN.h5
centroids_file = centroids_frac_400.txt
sys_dim        = 3
nval           = 4
ncond          = 8

K_POINTS {crystal_b}
3
  0.0000 0.0000 0.0000 40  #G
  0.5000 0.0000 0.0000 40  #X
  0.5000 0.5000 0.0000  1  #M
```

This path has $1+40+40=81$ points. As in Quantum ESPRESSO, each corner's count is the number of points on the segment that starts at that corner (see [`K_POINTS {crystal_b}`](#k-points)).

## Band character {: #character }

The path can be coloured by three kinds of band character, each requested with `--color` and written to `bands_<color>.png`, with the operator expectation values in `band_operators_path.npz`.

- **Spin.** `--color spin` colours each state by $\langle\boldsymbol\sigma\cdot\mathbf n\rangle$, with $\mathbf n$ the magnetization axis read from the WFN.
- **Atomic-orbital character.** `--color orbital:d` or `--color orbital:Fe:d` colours by the weight on the atomic functions of one angular momentum, optionally of one element, taken from the `PP_PSWFC` records of the `*.upf` files beside the input file.
- **Orbital moment.** `--color orbital` colours by the band-resolved orbital moment $\mathbf n\cdot\mathbf m_n$ of the modern theory, averaged over degenerate multiplets. It needs the velocity matrix of the WFN's own states (`--velocity dipole.h5`). The theory, the totals printed in the report and `--moments-grid` are on [Orbital magnetization](../theory/orbital-magnetization.md).

## Keywords {: #keywords }

The keys of the band window, the basis, the transformation and the QP energies are on the [overview](htransform.md#keywords). Defaults are read from the keyword schema (`src/gw/gw_config.py`) and the driver's argparse (`src/bandstructure/htransform.py`).

Required
{: .kw-group }

**Path and output**

### `K_POINTS {crystal_b}` {: .kw data-type="block" data-default="required" #k-points }
The $\bk_{\mathrm{fi}}$ path: a corner count, then one line per corner, `kx ky kz n  # label`, in crystal coordinates.

The segment from corner $i$ to corner $i+1$ gets the `n` of corner $i$ (default 1), as in Quantum ESPRESSO. The last corner's `n` is unused, and a path of $N$ corners has $1+\sum_{i<N}n_i$ points. With fewer than two corners no `bandstructure.dat` is written.
{: .kw-details }

Optional
{: .kw-group }

**Path and output**

### `-o`, `--output-file` {: .kw data-type="path" data-default="bandstructure.dat" }
Band table on the path.

### `--plot` {: .kw data-type="flag" data-default="off" }
Shows the band plot interactively (matplotlib). Needs a path.

On the path the returned bands are the `nval + ncond` eigenpairs with the largest weight in the Fourier-interpolated projector onto the returned bands, listed in energy order.

**Band character and moments**

### `--color` {: .kw data-type="str, repeatable" data-default="none" }
Colours the path by `spin`, `orbital` (needs `--velocity`) or `orbital:[EL:]l`; writes `bands_<color>.png` and `band_operators_path.npz`.

### `--velocity` {: .kw data-type="path" data-default="none" }
`dipole.h5` (or an SC run's `dipole_qsgw.h5`) of the WFN's own states. Refuses with `--eqp-file` or `--qp-rotations`.

### `--moments-grid` {: .kw data-type="int[3]" data-default="none" }
Interpolates $H$ and $\sigma_a$ to this uniform grid and writes `moments.txt`.

### `occ_smearing_width_ry` {: .kw data-type="float" data-default="unset → 1e-4 Ry step" }
Fermi–Dirac $k_BT$ of the moment sums only.

## Output format {: #output }

<div class="file-list" markdown>

`bandstructure.dat`
:   $E_{n\bk_{\mathrm{fi}}}$ on the path in eV, valence-band maximum on the path at 0.

`bands_<color>.png`, `band_operators_path.npz`, `moments.txt`
:   Band character and moments (`--color`, `--moments-grid`).

</div>

### bandstructure.dat {: #bandstructure-dat }

Plain text, one line per path point and band. The values shown are a format sample.

```text
# idx_k idx_b kx ky kz s energy_eV
# absolute_band_window=[0,12) fit_bands=16 guard_bands=4
   0    0  0.00000000  0.00000000  0.00000000  0.00000000 -11.94512031
   0    1  0.00000000  0.00000000  0.00000000  0.00000000  -0.00008112
 ...
```

| Column | Content |
|---|---|
| `idx_k` | path index, 0-based |
| `idx_b` | band index from the bottom of the returned window, 0-based; absolute band `b0 + idx_b` |
| `kx ky kz` | $\bk_{\mathrm{fi}}$ in crystal coordinates |
| `s` | cumulative path length |
| `energy_eV` | $E_{n\bk_{\mathrm{fi}}}$ in eV, path VBM at 0 |

## Refusals {: #refusals }

!!! refuses
    - A band-selection tie on the path wider than the multiplet tolerance.
    - `--velocity` with `--eqp-file` or `--qp-rotations`; `--color orbital` without `--velocity`; an unknown `--color`; `--color` or `--plot` without a path.
    - `--color orbital:[EL:]l` without `*.upf` files beside the input file.

The refusals of the band window, the basis and the QP energies are on the [overview](htransform.md#refusals).

