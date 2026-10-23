---
driver: gw
page: overview
---

# GW driver

`gw.gw_jax` Quasiparticle energies from the ISDF-compressed GW self-energy.
{: .page-summary }

<div class="workflow-map" data-driver="gw" data-page="overview"></div>

Computes quasiparticle energies and orbitals from the ISDF-compressed GW self-energy, one-shot or quasiparticle self-consistent (QSGW). It runs after the mean field and the [centroids](../../utilities/centroids.md), [kin-ion](../../utilities/kin-ion.md) and [dipole](../../utilities/dipole.md) utilities, and feeds [htransform](../../utilities/htransform.md), [downfold](../../utilities/downfold.md) and both BSE drivers.
{: .lead }

<div class="glance" markdown>

Invoke
:   `python -m gw.gw_jax -i gw.in`

Reads
:   input file, `WFN.h5`, `centroids_frac_<N>.txt`, `kin_ion.h5`, `dipole.h5`; under QSGW also `parallel_transport.h5`

Writes
:   `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `qp_wfn_rotations.h5`, `WFN_qp.h5`, ISDF tensor file, `gwjax.out`

Σ treatment
:   tangential model (`compute_mode = tangential`) or `gn_ppm`; others for debugging and comparison ([below](#treatment))

</div>

## What it computes {: #what }

<div class="stages" markdown>

1. **ISDF basis and Coulomb metric**
   Fit $\zeta_{\bq\mu}(\br)$ by least squares on the pair-density Gram over the fit windows, then build $V_\bq(\mu,\nu)$ on the sphere $|\bq+\bG|^2\le$ `bare_coulomb_cutoff`. A `tmp/zeta_q.h5` whose fit provenance matches is reused instead of refitted.
2. **Screening**
   $\chi^0_\bq(\br_\mu,\br_\nu;z)$ from the products $-iG^{\gtrless}G^{\lessgtr}$ on the points, at the complex frequencies the treatment requests, then the per-$\bq$ Dyson solve $W_\bq(z)=[I-V_\bq\chi^0_\bq(z)]^{-1}V_\bq$. On the production route $W^c=W-V$ becomes one set of real poles per $\bq$, shared by every matrix element ([Tangential W model](../../theory/tangential-w.md)).
3. **Self-energy**
   $\Sigma^x$ and $\Sigma^c(\omega)$, read at each state's own energy, with the $\bq\to0$ head carried as a separate scalar channel ([the q → 0 head](../../theory/q0-head.md)). On the dynamic treatments the $\Sigma^c$ band sum is extrapolated past `number_bands_sigma` ([band extrapolation](../../theory/band-extrapolation.md)).
4. **Quasiparticle bands**
   The Hermitian $\Sigma^{\mathrm{QSGW}}$ enters $H=T+V^{\mathrm{ion}}+V^{\mathrm H}+\Sigma^{\mathrm{QSGW}}$ in the DFT basis, which is diagonalized once (`one_shot_dft`) or iterated to a fixed point under Anderson mixing (`self_consistent`, [QSGW](../../theory/qsgw.md)). No QP equation is solved.

</div>

<figure class="flowchart" markdown>
[![Full-frequency GW driver](../../assets/figures/g0.png)](../../assets/figures/g0.png)
<figcaption><strong>Fig. G0.</strong> The four stages on the production route (tangential <i>W</i>), read left to right within a stage, with the input and output objects and their files on the right; the dark strip on stage 4 is the QSGW loop back to stage 2.</figcaption>
</figure>

## Running {: #run }

=== "Local"
    ```bash
    uv run python -u -m gw.gw_jax -i gw.in
    ```
=== "Perlmutter"
    ```bash
    source config/perlmutter/gpu_env.sh
    srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh .venv/bin/python -u -m gw.gw_jax -i gw.in
    ```
=== "Frontera"
    ```bash
    LORRAX_ROOT=$LORRAX LORRAX_RUN_DIR=$PWD LORRAX_INPUT=gw.in sbatch $LORRAX/config/frontera/templates/gw_dev.sbatch
    ```

Run from a compute node, one process per device. The Perlmutter line is for a clone in the current directory; the module, the Frontera build and other sites are on [Installation](../../using/installation.md#routes). The Frontera template runs `gw.gw_jax` on the CPU leg; edit its `#SBATCH` block before submitting.

A minimal production input file:

```ini
# Requires, beside the input file: WFN.h5, the centroids file (centroids), and kin_ion.h5 (kin-ion)
# and dipole.h5 (dipole), both run on this input file.
[cohsex]
wfn_file               = WFN.h5
centroids_file         = centroids_frac_1200.txt  # as centroid.kmeans_cli names it
sys_dim                = 3                        # system dimensionality: required
number_bands           = 200                      # total bands in the GW calculation, for chi0 and Sigma
number_bands_protected = 40                       # active bands for Sigma matrix elements, from the lowest valence band
compute_mode           = tangential
qp_solver              = self_consistent
sc_tol_ev              = 1e-3
```

Every other key takes its default ([keywords](keywords.md)). Choose `number_bands` for $\chi^0$, which is not extrapolated, and give $\Sigma$ at least 20 active conduction bands. A metal adds `occ_smearing_width_ry` and `fermi_reference = mp1_fixed_n` ([Metals](../../theory/metals.md)).

## Choosing a Σ treatment {: #treatment }

Two treatments are for production runs: the tangential model, full frequency, for insulators, metals and time-reversal-broken input files; and the Godby–Needs plasmon pole for insulators. The others are kept for debugging and comparison. `qp_solver` is a separate axis. Expand a row for the model, its input file lines and its limits; the derivations are on [Frequency treatments](../../theory/frequency-treatments.md).

???+ treatment production "tangential · Tangential model · full frequency"
    One set of real poles $\Omega_{j\bq}$ per irreducible $\bq$, shared by every matrix element, with one factor vector $b_{j\bq}(\mu)$ per pole ([normalisation](../../theory/tangential-w.md#model-form)). The poles come from samples of $W^c$ and $\partial_zW^c$ on a line above the real axis and an imaginary ladder, plus the exact high-frequency moments; nothing is fitted element by element. $\Sigma^c(\omega)$ is evaluated on the real axis at each state's own energy.

    $$
    W^c_{\bq}(\mu,\nu;z)=\sum_{j}b_{j\bq}(\mu)\,b^*_{j\bq}(\nu)\left[\frac{1}{z-\Omega_{j\bq}}-\frac{1}{z+\Omega_{j\bq}}\right],\qquad \Omega_{j\bq}>0
    $$

    ```ini
    compute_mode = tangential
    qp_solver    = self_consistent
    ```

    **Note** The production route for insulators, metals and time-reversal-broken input files, with `qp_solver = self_consistent`. `sigma_w_accuracy = relaxed` selects the looser recipe tier. Full page: [Tangential W model](../../theory/tangential-w.md).

??? treatment supported "gn_ppm · Godby–Needs plasmon pole · plasmon pole"
    One real pole per matrix element, fixed by $W^c$ at $z=0$ and at $z=i\omega_p$, $\omega_p=$ `ppm_omega_p` (default 2 Ry). On a time-reversal-broken input file the fit carries a second, odd residue.

    $$
    W^c_{\bq}(\mu,\nu;z)=R_{\bq}(\mu,\nu)\left[\frac{1}{z-\Omega_{\bq}(\mu,\nu)}-\frac{1}{z+\Omega_{\bq}(\mu,\nu)}\right]
    $$

    ```ini
    compute_mode = gn_ppm
    ```

    **Note** Refuses metallic occupations.

??? treatment "Other treatments: debugging and comparison"
    ??? treatment supported "x_only · Exchange only · static"
        Bare exchange, no screening: $\Sigma^c=0$ and no $\chi^0$ or $W$ is built. The QP Hamiltonian holds $\Sigma^x$ alone, so the result is Hartree–Fock on the mean-field orbitals.

        $$
        \Sigma^x_{\bk}(\br_\mu,\br_\nu)=i\sum_{\bq}G^{<}_{\bk-\bq}(\br_\mu,\br_\nu;t{=}0)\,V_{\bq}(\mu,\nu)
        $$

        ```ini
        compute_mode = x_only
        ```

        **Note** Runs on metals. The cheapest check of the ζ fit and $V_\bq$.

    ??? treatment supported "cohsex · Static COHSEX · static"
        Screened exchange plus a Coulomb hole, both from one static solve $W(0)$, evaluated in closed form at $t=0$ with no frequency axis.

        $$
        \Sigma^{\mathrm{COHSEX}}=i\,G^{<}(t{=}0)\,W(0)+\tfrac{i}{2}\big[G^{>}-G^{<}\big](t{=}0)\,W^c(0)
        $$

        ```ini
        compute_mode = cohsex
        ```

        **Note** What `compute_mode = auto` runs. Refuses metallic occupations.

    ??? treatment in-progress "hl_ppm · Hybertsen–Louie plasmon pole · plasmon pole"
        The same two-point fit with the probe on the real axis above every $\chi^0$ transition, $\omega_p>\Delta_{\max}$, so the pole carries the $f$-sum coefficient. Missing: a default probe; the GN default of 2 Ry lies inside the spectrum of most input files and refuses.

        $$
        \Omega^2_{\bq}(\mu,\nu)=-\omega_p^2\,\frac{W^c_{\bq}(\mu,\nu;\omega_p)}{W^c_{\bq}(\mu,\nu;0)-W^c_{\bq}(\mu,\nu;\omega_p)}
        $$

        ```ini
        compute_mode = hl_ppm
        ppm_omega_p  = 200.0
        ```

        **Note** Refuses metallic occupations and time-reversal-broken input files.


    ??? treatment in-progress "cd · Contour deformation · full frequency"
        The frequency integral rotated onto the imaginary axis, plus the residues of the Green's-function poles the rotation crosses, as in BerkeleyGW and Yambo, on the exact ISDF $W(z)$ with no pole model. A reference for the pole models and for comparisons with other codes; the production full-frequency route is the tangential model. Full page: [Contour deformation](../../theory/contour-deformation.md).

        $$
        \Sigma^c(E)=-\frac{1}{2\pi}\int_{-\infty}^{\infty}du\,G(E+iu)\,W^c(iu)+\sum_{\text{crossed poles}}\mathrm{Res}
        $$

        ```ini
        compute_mode                = mpa
        sigma_frequency_integration = contour_deformation
        head_correction             = off
        use_band_extrapolation      = false
        ```

        **Note** Refuses the $\bq\to0$ head, band extrapolation and `linalg = distributed` (implementation limits).

## Accuracy and convergence {: #accuracy }

The numerical controls (`sigma_quadrature_eps`, the centroid count, the pole budget of $W$) each reach about 1 meV at the settings below. The band counts, the k grid and the semicore read are systematic errors; converge them per material. Unless stated, errors are of eqp0 against a converged reference: standard deviation / maximum over the multiplet-closed states within ±10 eV of midgap / error of the smallest gap on the grid.

| Parameter | Key | Measured | To reduce it |
|---|---|---|---|
| Bands in $\chi^0$, hence $W$ | `number_bands` | Si 4³, against the complete basis (536 bands), $\Sigma^c$ band-extrapolated: 19.7 / 58 / −67 meV at 64 bands, 16.7 / 49 / −53 at 78, 7.0 / 20 / −21 at 142. Falls about as $N^{-1.2}$; mostly a rigid shift. | Raise `number_bands`. $\chi^0$ is never extrapolated ([band extrapolation](../../theory/band-extrapolation.md#trunc-scope)). |
| Bands in $\Sigma^c$ | `use_band_extrapolation` | Default on; measured accuracy in [band extrapolation §5.4](../../theory/band-extrapolation.md#ex-measured). | Keep it on. |
| QP-matrix top $b_3$ | `number_bands_protected` | Si 4³ SOC QSGW, 100 bands, direct gap against $b_3=N$: −107 meV at $b_3=16$, −89 at 24, −59 at 40, −39 at 64. | Raise $b_3$ toward `number_bands`; nothing else removes it ([QSGW §3.1](../../theory/qsgw.md#bands-qp)). |
| k grid | (WFN) | The link error $\epsilon_{\mathrm{link}}$ printed by dipole and on every QSGW map: 9 % on MoS₂ 3×3, 4.6 % on Fe 4³, 3.4 % on Si 4³. On MoS₂ 3×3 the QSGW gap is 5.167 eV without the link term and 4.483 eV with it. | Refine the k grid ([velocity operator §3.6](../../theory/velocity-operator.md#berry-valid)). |
| ISDF points | `centroids_file` | $\Sigma^x$ RMS ≤ 1 meV at $N_\mu\approx0.5\,r$ ([ISDF §4](../../theory/isdf.md#isdf-accuracy)). | Select on the Σ pair set. |
| Σ quadrature | `sigma_quadrature_eps` | Default $10^{-4}$: eqp0 within 0.89 meV of $\varepsilon=10^{-6}$ (Fe 4³ QSGW, $E_F\pm15$ eV). | Lower $\varepsilon$. |
| Pole count of $W$ | (recipe) | 0.80 meV QP RMS against contour deformation, Si ([Tangential W model §7.9](../../theory/tangential-w.md#ritz-count)). | — |
| Semicore read | (fixed, η = 5 eV) | $E_F\pm1$ eV: 3.6 / 15.9 meV on Fe 4³, 2.9 / 20.0 meV on MoS₂ 3×3 ([QSGW §3.3](../../theory/qsgw.md#bands-core)). | `sigma_omega_patches_ev` windows at a smaller η. |

!!! note "Agreement with BerkeleyGW"
    On Si with spin–orbit, matched full-frequency G₀W₀ gaps agree with BerkeleyGW contour deformation to +0.1 meV (indirect) and +1.5 meV (direct), and static COHSEX Σ to 0.34 meV mean. No metal or slab has a matched comparison. Settings and the full table: [Measured agreement](../../using/berkeleygw-users.md#agreement).

## Refusals {: #refusals }

A refusal names the key or file and the fix. The ones most runs meet are listed first; the rest are grouped below.

!!! refuses "Common"
    - No `sys_dim`. Add `sys_dim = 3` (bulk) or `2` (slab). 
    - An unknown key, or a `;` comment after a value. Delete the key; comments start with `#`. 
    - Fractional WFN occupations (a metal) with `compute_mode = cohsex`, `gn_ppm` or `hl_ppm`, without `occ_smearing_width_ry`, without `fermi_reference = mp1_fixed_n`, or with `sc_head_update = interband_commutator`. 
    - A `kin_ion.h5` that is missing, has too few bands, or was built for another `sys_dim`, spinor count, k-point count or `bispinor` setting, or in the retired format with Hartree folded in. Rerun kin-ion. Its WFN fingerprint is not compared. 
    - A missing `dipole.h5` with the head on, or one built with another velocity operator, with or without the nonlocal commutator. Under `head_correction = full`, also one from another WFN or band window; on the direct-head path that mismatch only warns, unless `LORRAX_SANITY=strict`. Rerun dipole on this input file and WFN. 
    - A centroid table whose fit windows drop an occupied band. Rerun centroids with windows that start at band 0. 
    - A `tmp/zeta_q.h5` left by a run with another centroid count or FFT grid in the same directory. Use a separate run directory or delete the file. 
    - A named `number_bands_chi` or `number_bands_sigma` that splits a degenerate multiplet or reaches the top of `WFN.h5`. Pick a legal edge from the message. 
    - A QSGW run unconverged after `sc_max_iter` maps (the per-map tables are kept), a checkpoint that continues past `sc_max_iter`, or scissored tail bands that take up occupation. Raise the budget, or widen the QP matrix. 

??? refuses "Input keys that conflict or do not apply"
    - A value outside its range or enum. The message names the accepted values. 
    - Two spellings of one setting that disagree: `nband` and `number_bands`, `number_bands` and a different `number_bands_chi` or `number_bands_sigma`, `use_band_extrapolation` and `sigma_band_extrapolation`, `do_G0` and `head_correction`, `do_screened = false` with a screened `compute_mode`. Keep one. 
    - `number_bands_protected` beside `nval` or `ncond`, or outside [occupied count, `number_bands`]. 
    - The retired key `sigma_w_model` (use `compute_mode = tangential`); `sigma_w_accuracy`, `sigma_w_support_sites_ev`, `write_poles` or `write_w` without `compute_mode = tangential`. 
    - A key of the [other W models](../../theory/multipole-w.md#keywords) under `compute_mode = tangential`, and with the head off every head-fit sampling key. Delete them. 
    - `sigma_regularization_ev` that is not finite and positive. 
    - `screening_diagrams = w_bse | w_rpa_resolvent` outside its supported treatments, on a centroid set that is not orbit-closed, or (`w_bse`) on a time-reversal-broken WFN. 
    - `sc_initial_qp_rotations_file` without `qp_solver = self_consistent`. 
    - `sc_head_update` other than `off` with `head_correction = off`; `occ_broadening > 0` outside a QSGW run with a velocity head, or beside `occ_smearing_width_ry`. 
    - `use_band_extrapolation = true` or a bracket scheme named on a run with no dynamic Σ stage; a bracket scheme named with extrapolation off. 
    - `sigma_out_of_grid = clamp | static` without both `sigma_omega_min_ev` and `sigma_omega_max_ev`. 
    - `zeta_cutoff` above the density cutoff, `bare_coulomb_cutoff` above `zeta_cutoff`, `zeta_nband` outside $[1,\max(N_\chi,N_\Sigma)]$, or a QP matrix that reaches past the ζ fit's left window. 
    - `number_bands_sigma` below twice the occupied count with band extrapolation on. Raise `number_bands`. 

??? refuses "Material class"
    - Integer WFN occupations (an insulator) with `occ_smearing_width_ry`, `fermi_reference = mp1_fixed_n` or the head-fit metal origin shift ([keys](../../theory/multipole-w.md#keywords)). Remove the key. 
    - `hl_ppm` on a time-reversal-broken WFN, or with `ppm_omega_p` at or below the largest $\chi^0$ transition. Raise `ppm_omega_p`, or use `gn_ppm`. 

??? refuses "During the run"
    - `restart = true` against a bundle written for a different centroid table of the same count, or from another WFN. Set `restart = false`. 
    - A static minimax rule that misses `minimax_target_error` within `minimax_max_nodes`. Raise the cap or the target. 
    - A QP energy inside a hole of `sigma_omega_patches_ev`. 
    - A non-finite $Z$: $\Sigma^c$ has a pole at a state's evaluation energy (`GATE z_factor_finite`). 
    - A required output missing or uncommitted at the end of the run. 

??? refuses "Retired keys and values"
    - A retired or removed key, or a retired value: `write_eqp2` and the `eqp2_*` keys (`GATE eqp2_retired`), `qp_solver = fixed_point`, `sc_accelerator = rcrop | linear`, `screening_method = ctsp`, `band_extrapolation_estimator = band_index_only`, any `sc_mixing` but `1.0`. The message names the replacement ([Retired](keywords.md#retired)). 

