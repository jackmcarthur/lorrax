# Sigma/GN-PPM Audit Status (Handoff)

Date: 2026-03-31  
Scope: dynamic GN-PPM correlation (`Sigma_cor`) validation across BGW, ISDF SoS, and GWJAX minimax.

## 1) Current Status

- CO molecular case (`tests_isdf`):
  - BGW, ISDF SoS, and GWJAX agree very well on `Sigma_cor`.
  - Current GWJAX vs BGW metric (bands with finite values, 18 states):  
    MAE `6.493e-03 eV`, max abs `1.6785e-02 eV`.
- MoS2 periodic case (`tests_isdf`):
  - trends are qualitatively right, but quantitative `Sigma_cor` mismatch remains large.
  - Current GWJAX vs BGW metric (k=0..3, bands 23..28):  
    MAE `3.682 eV`, max abs `4.815 eV`.
  - Pattern in current run:
    - valence `Sigma_cor` follows BGW trends but is too positive.
    - conduction `Sigma_cor` follows BGW trends but is too negative in magnitude.
    - per-band mismatch behaves like offset + scale error, with rough factor `~1.5x` to `~2x`.

## 2) Critical Decomposition Mapping (Do Not Mix These)

### BGW dynamic GN decomposition

- BGW reports:
  - `SX-X`: screened-exchange correction relative to bare exchange.
  - `CH`: Coulomb-hole term.
  - `Cor = (SX-X) + CH`.
- Important band-sum distinction:
  - `SX-X` is a valence/occupied-state sum.
  - `CH` is an all-state sum.

### ISDF SoS / GWJAX decomposition

- ISDF SoS and GWJAX internal dynamic correlation is naturally split as denominator branches:
  - `Sigma_cor+` and `Sigma_cor-`.
- This is **not** the same partition as BGW’s `SX-X` and `CH`.
- Therefore, branchwise comparisons (`Sigma_cor+` vs `SX-X`, `Sigma_cor-` vs `CH`) are not valid targets.
- Correct comparable quantity is total correlation:
  - `Sigma_cor_total = Sigma_cor+ + Sigma_cor-` vs BGW `Cor`.

### BGW near-pole redistribution

- BGW may reassign near-pole contribution between `SX-X` and `CH` for conditioning.
- This is another reason only the sum (`Cor`) is stable as a cross-code target.

## 3) Code Entry Points

### Production minimax pipeline (repo)

- [`src/gw/gw_jax.py`](/home/jackm/projects/lorrax/src/gw/gw_jax.py)
- [`src/gw/ppm_sigma.py`](/home/jackm/projects/lorrax/src/gw/ppm_sigma.py)
- [`src/gw/w_isdf.py`](/home/jackm/projects/lorrax/src/gw/w_isdf.py)

### ISDF SoS debug / comparison harness (tests_isdf)

- SoS generator and GN debug tables:  
  [`isdf_w_from_eps0_projection_test.py`](/home/jackm/projects/tests_isdf/qe_co/isdf_w_from_eps0_projection_test.py)
- BGW-style comparison plots/tables (`CH'`, `SX-X'`, `Cor`):  
  [`plot_sigmahp_chprime_compare_consistent.py`](/home/jackm/projects/tests_isdf/qe_co/plot_sigmahp_chprime_compare_consistent.py)
- Sweep report archive (sexcut / invalid-mode / W-source / eta experiments):  
  [`sigma_cor_sweep_report/results_summary.md`](/home/jackm/projects/tests_isdf/qe_co/sigma_cor_sweep_report/results_summary.md)

## 4) Directory Map (Comprehensive Handoff)

- CO BGW reference calc:  
  [`tests_isdf/qe_co`](/home/jackm/projects/tests_isdf/qe_co)  
  Key refs: `sigma.out`, `sigma_hp.log`, `ch_converge.dat`, `eps0mat.h5`.
- CO GWJAX/minimax run (comparison-ready):  
  [`tests_isdf/jax_co_run40`](/home/jackm/projects/tests_isdf/jax_co_run40)  
  Key refs: `eqp0_m10_10_40b.dat`, `gwjax_cor_vs_bgw.dat`, `cor_vs_bgw_gwjax_m10_10_40b.png`.
- CO base assets / earlier debug runs:  
  [`tests_isdf/jax_co`](/home/jackm/projects/tests_isdf/jax_co)
- MoS2 BGW reference calc:  
  [`tests_isdf/qe_mos2`](/home/jackm/projects/tests_isdf/qe_mos2)  
  Key refs: `sigma.out`, `sigma_hp.log`, `ch_converge.dat`, `eps0mat.h5`.
- MoS2 GWJAX/minimax run (comparison-ready):  
  [`tests_isdf/cohsex_prod_run80_compare`](/home/jackm/projects/tests_isdf/cohsex_prod_run80_compare)  
  Key refs: `eqp0_noqsym_w_m10_10_80b_compare.dat`, `gwjax_vs_bgw_sigmaout_compare.dat`, `gwjax_vs_bgw_sigmaout_compare.png`.
- MoS2 GWJAX working directory backing that run:  
  [`tests_isdf/cohsex_prod`](/home/jackm/projects/tests_isdf/cohsex_prod)

## 5) BGW Settings Used For Close Parity

For BGW-side reference runs used in current comparisons:

- `epsilon`:
  - `broadening 0.001`
- `sigma`:
  - `exact_static_ch 0`
  - `screened_coulomb_cutoff 25.0` (match PW cutoff)
  - `invalid_gpp_mode` set to explicit comparison policy
  - `frequency_dependence 3` for GN/GPP

Interpretation notes:

- If `exact_static_ch` is not used in GPP, compare with `sigma_hp.log` `CH'`/`SX-X'`-consistent quantities, not raw printed `CH` in `sigma.out`.
- In BGW GPP path, `gpp_broadening` and `gpp_sexcutoff` control conditioning/cutoff logic; `gpp_broadening` is not simply a direct `+i*eta` denominator add in the naive form.

## 6) Known Outstanding Gap

The unresolved issue is primarily periodic/k-point MoS2 in full minimax GWJAX (`Sigma_cor`) vs BGW.
CO molecular agreement indicates the core GN-PPM/SOS machinery is sound in 0D; remaining work is around periodic-path details (k-point dependent denominator handling, conditioning/cutoff parity, and related implementation differences).

## 7) Minimal Reproduction Commands

CO (good-agreement baseline):

```bash
cd /home/jackm/projects/tests_isdf/jax_co_run40
uv run python -m gw.gw_jax -i cohsex_jaxco_m10_10_40b.in
uv run python rebuild_cor_vs_bgw_gwjax_m10_10_40b.py
```

MoS2 (current mismatch baseline):

```bash
cd /home/jackm/projects/tests_isdf/cohsex_prod_run80_compare
uv run python -m gw.gw_jax -i cohsex_prod_run80_compare.in
uv run python plot_gwjax_vs_bgw_sigmaout_compare.py
```
