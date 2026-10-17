# Running TD-aGW: decks, the log, the integrator

This page is for running `tdagw.tdgw_jax` on a system you already have: three
example drives, how to read the run log `tdagw.out`, and which integrator to
pick, with costs measured on the production 30×30 CrI3 model. It restates no
equation and no key. The equation of motion and the schemes are on
[TD-aGW propagation](../theory/td_agw_propagation.md), every deck key with its
default and refusals in the [input reference](../input_reference.md#td-agw)
(generated from the parser), and the chain from a GW run to a drive in
[TD-aGW end to end](td-agw-workflow.md).

## Run {#run}

```bash
lx run --pool <POOL> -N 1 -G 4 -n 4 -- python3 -u -m tdagw.tdgw_jax -i tdgw.toml
```

One rank per GPU; the k axis is sharded over every device. In
`[control.files] outdir` the run writes `tdgw.h5` and `tdgw.toml` (one
`trial-<i>/` each with `lorrax.trials`), which `tdgw-pp` reads unchanged, and
one log, `tdagw.out`, from process 0. Stdout carries the run header, the
stability gate's verdict, one line per chunk and the summary; the log carries
all of it and more ([below](#log)).

## Three decks {#decks}

All three use the bundle route of the production runs: the system and the
ISDF ΔΣ come from one GW restart bundle, no TDGW run is needed. The common
part (paths shortened; the CrI3 monolayer window of 42 valence and 14
conduction bands, 30×30 k):

```toml
[control]
calc_type = "COHSEX"

[control.timesteps]
t0 = 0
tf = 30000              # as
dt_measure = 100        # as: the sample grid
ntcheck = 70            # samples per chunk: one checkpoint, one row of the log's table

[control.files]
outdir = "./out"

[measurements]
types = ["Efield", "polarization", "carrier_density"]

[system]
ibndmin = 89
ibndmax = 144
ivbm = 130
geometry = "slab"

[lorrax]
system = "bundle"
bundle_input = ".../k8_df6000r8_c1_e9000_42v_n112/gnppm.in"
dipole_file = ".../12_bseprod_prep/dipole.h5"
eqp_file = ".../37_bseprod_c9000_p16/eqp1.dat"
isdf_zeta = ".../55_dfold_df6000r8_w88_144/tmp/zeta_q.h5"
model = "isdf"
method = "cfm4"
dt = 50                 # as; or "auto" (the gate's largest stable step)
```

Each deck then names its drive. Fields are in V/m and the expressions take t
in as, with `np` and `HBAR_IN_EV_AS` (ħ = 658.2119569 eV·as) in scope.

**1. δ kick, linear response.** A Gaussian far shorter than ħ over the window's
largest transition excites every pole with nearly equal weight; a weak field keeps
the response linear, and `tdgw-pp linear` divides P(ω) by E(ω). Two trials give
the x and y columns of ε(ω) in one run:

```toml
[lorrax]
trials = [
  {E_x = "1e5 * np.exp(-((t - 750) / 150)**2)", E_y = 0, E_z = 0},
  {E_x = 0, E_y = "1e5 * np.exp(-((t - 750) / 150)**2)", E_z = 0},
]
```

With τ = 150 as the power spectrum of the kick has a half width of 5.2 eV, above
the 1.4–4 eV transitions of this window.

**2. Circular σ+ Gaussian pulse.** σ+ turns the field from x to y,
counter-clockwise seen from +z ([helicity
convention](../architecture/bse.md#dipoles-and-absorption)); σ− flips the sign of
E_y. This is the driven production case (5e6 V/m, ħω = 1.67043 eV, τ = 4 fs,
centred at 16 fs):

```toml
[lorrax]
trials = [
  {E_x = "5e+06 * np.exp(-((t - 16000) / 4000)**2) * np.cos((1.67043 / HBAR_IN_EV_AS) * t)",
   E_y = "5e+06 * np.exp(-((t - 16000) / 4000)**2) * np.sin((1.67043 / HBAR_IN_EV_AS) * t)",
   E_z = 0},
]
```

**3. A pulse whose bandwidth matches a broadening.** For the envelope
exp(−((t − t_c)/τ)²) the power spectrum |E(ω)|² is a Gaussian about ħω₀ with half
width at half maximum HWHM = √(2 ln 2) ħ/τ, so a pulse that spans a given
broadening (a line width, or the η of the spectrum it is compared with) has

  **τ = 1.177 ħ / HWHM**

| HWHM (eV) | τ (as) | t_c ≥ 3τ (as) |
|---|---|---|
| 0.05 | 15 500 | 46 500 |
| 0.10 | 7 750 | 23 250 |
| 0.194 | 4 000 | 12 000 |

(the last row is drive 2). Centre the pulse at t_c ≥ 3τ, where the envelope is
1.2e-4 of its peak, so the field switches on smoothly, and run to tf ≥ t_c + 3τ
plus the response time you want to sample. For a 0.1 eV HWHM at ħω₀ = 1.67 eV:

```toml
[lorrax]
trials = [
  {E_x = "5e+06 * np.exp(-((t - 23250) / 7750)**2) * np.cos((1.67043 / HBAR_IN_EV_AS) * t)",
   E_y = "5e+06 * np.exp(-((t - 23250) / 7750)**2) * np.sin((1.67043 / HBAR_IN_EV_AS) * t)",
   E_z = 0},
]
```

## Reading the log {#log}

`outdir/tdagw.out` is written as the run goes (one table row per chunk, flushed),
so `tail -f` follows a running job. Its sections, in order:

| section | what it holds | what to check |
|---|---|---|
| provenance | start time and host, git HEAD with a clean/DIRTY flag, the source closure, the FFI bundle (sealed id, revision, sha), JAX, ranks, devices, mesh | the tree and the bundle are the ones you meant |
| resolved deck | every key: value as written, `set` / `default` / `unset`, what an empty default resolved to; `not used` on a key the run does not read | no surprise in `not used` (e.g. `control.ivp.rtol` under `cfm4`) |
| system | the run header, plus the k grid, occupations, QP energy range and gap, the dipole route and its receipts | the window and the gap |
| self-energy model | model, terms, N_μ, whether ΔΣ is affine in ρ, the wall of one ΔΣ call (timed by the gate's power iteration, its compile excluded) | the per-call wall: a run costs about (ΔΣ calls per fs) × that |
| integrator and stability gate | method, step, ΔΣ calls per step and per fs, chunk; the stiff split's basis; σ, the static limit, every probe (step, growth, pass), the chosen dt and the gate's cost | `chosen dt` and the probe growth (≤ 1.002 passes) |
| progress | one row per chunk: samples, t (fs), wall (s), s/fs of the chunk, ETA (s), ΔΣ calls so far, \|P\|, n_c and \|E\| at the chunk's last sample, and the largest drift of Tr ρ and Tr ρ² | s/fs after the first row (which carries the compile); drifts near 1e-15 when Γ = 0 |
| summary | stdout's summary plus ΔΣ per fs with and without the gate, steady s/fs, the gate's share, the ΔΣ wall's share of the run, the largest drifts, the final sample | the steady s/fs is the number to plan with |

The table's columns are fixed (`tdagw.td_log.PROGRESS_COLUMNS`), so
`tdagw.td_log.parse_progress_row` reads any row back. Tr ρ and Tr ρ² are computed
on the device at every sample with no ΔΣ call; both are invariants of the exact
flow without Γ (the intraband term moves charge between k but keeps the sums), so
a drift far above rounding in a Γ = 0 run is a defect. With Γ ≠ 0, Tr ρ² decays by
design. The energy is not logged: it needs ΔΣ[ρ] at every sample. A run that a
gate refuses still closes its log: the gate's lines, the refusal's message and a
`REFUSED (… GATE <id>)` line.

## Choosing an integrator {#integrators}

The first three rows were measured on four A100s on the 30×30 CrI3 ISDF model
(N_μ = 846, 0.161 s per ΔΣ call; the σ+ deck above at 5e6 V/m over 27.9 fs; P(t)
error against a Richardson reference; 2026-10-17). On this model a ΔΣ call
outweighs everything else in a step, so the wall follows the ΔΣ count.

| want | use | ΔΣ calls per fs | wall per fs | measured on |
|---|---|---|---|---|
| P(t) error ~1e-5 (the default choice) | `cfm4`, `n_corr = 2`, cubic predictor, `dt = 50` (also its largest stable step) | 80.5 | 13.1 s | 30×30 CrI3, error 9.8e-6 |
| P(t) error ~1e-4 | `cfm4` with `stiff_modes = 10`, `stiff_krylov = 120`, `dt` ≈ 82–100 | 49.0; 44.6 with samples on step ends (dt 90.7) | 8.8 s; 8.0 s | 30×30 CrI3 |
| no step choice | `dt = "auto"`: the largest step the gate's probe passes, ≈50 as plain and ≈100 as with the split | as the rows above | | 30×30 CrI3 |
| Γ or the intraband term on, ΔΣ cheap | `etdrk4`, `dt` 25–50: Γ is exact in its linear part, while `cfm4`'s triple jump takes three times the exponentials | 160 at 25 as | 1.16 s against `cfm4`'s 5.17 s (Γ = 0.05 eV) and 9.41 s (intraband) | synthetic 30v × 14c, 30×30, 4.3 ms stand-in ΔΣ |
| TDGW's own steps, or a narrow, non-stiff window | `dp54`, the parser default (TDGW's RK45 replica) | 46 at rtol 1e-6 (error 3.5e-4) | | 6×6 CrI3 2v2c |

With the ISDF ΔΣ (0.16 s a call against 3.3 ms for an exponential) the extra
exponentials of `cfm4` with Γ cost little, so `cfm4` stays the cheaper scheme per
accuracy there. The split's gate and basis add 3.6 ΔΣ calls per fs over a 100 fs
run (232 + 132 calls). At 1e-5 the split gains nothing (80.1 against 80.5 calls per
fs): the scheme is still fourth-order `cfm4` with the same error constant, so
accuracy keeps the step at 50 as. One correction with the linear predictor is
stable only with the split and is third order with 125× the error, so it saves
nothing. Converged (Anderson) Gauss nodes are not recommended: they cost 1.8–2.0×
more ΔΣ calls at matched error (branch `feat/td-agw-implicit-nodes-2026-10-17`,
not merged). The 6×6 and synthetic operating points and the reasons behind them
are on [choosing a method](../theory/td_agw_propagation.md#choosing).
