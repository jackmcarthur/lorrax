# Running TD-aGW: decks, the log, the integrator

This page is for running `tdagw.tdgw_jax` on a system you already have: three
example drives, how to read the run log `tdagw.out`, the collision term, and
which integrator to pick, with costs measured on the production 30×30 CrI3 model. It restates no
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
| self-energy model | model, terms, N_μ, whether ΔΣ is affine in ρ, the wall of one ΔΣ call (timed by the gate's power iteration, its compile excluded) | the per-call wall: a run costs about (ΔΣ calls per fs) × that; on a cold kernel cache it includes first-use builds, and the summary's steady bound is the figure to use |
| integrator and stability gate | method, step, ΔΣ calls per step and per fs, chunk; the stiff split's basis; σ, the static limit, every probe (step, growth, pass), the chosen dt and the gate's cost | `chosen dt` and the probe growth (≤ 1.002 passes) |
| progress | one row per chunk: samples, t (fs), wall (s), s/fs of the chunk, ETA (s), ΔΣ calls so far, \|P\|, n_c and \|E\| at the chunk's last sample, and the largest drift of Tr ρ and Tr ρ² | s/fs of the rows that did not compile (the first always does; a last chunk of a new step count too); drifts at rounding level when Γ = 0 |
| summary | stdout's summary plus ΔΣ per fs with and without the gate, steady s/fs, the gate's share, the steady bound on one ΔΣ call (chunks 2–N wall over their ΔΣ calls) and the ΔΣ share of the run, the largest drifts, the final sample | the steady s/fs is the number to plan with |

From the default run of the production 846 deck shortened to 3 fs (four A100s;
`cfm4`, dt 50 as), the gate block and the progress table:

```text
  stability       : sigma = 6.208 eV (spectral radius of the dSigma commutator map, 20 power iterations, last change 0.025 %), h*sigma = 0.472, limit 0.73 (cfm4, cubic predictor, n_corr 2) -> dt <= 70.36 as; probe 1 per step at 62.5 as
  gate verdict    :
    sigma         : 6.20804 eV (spectral radius of the linearized dSigma commutator map; 20 power iterations, last change 0.025 %)
    static limit  : h sigma 1.1 <= 0.73 (cfm4, cubic predictor, n_corr 2) -> dt <= 70.36 as; this step h sigma = 0.4716
    probe         : dt 50 as -> probe step 62.5 as (1.25 dt), growth 1.000000 per step over 48 steps, field off: pass (<= 1.002); 35.87 s
    chosen dt     : 50 as (the deck's)
    gate cost     : 237 dSigma calls, 51.86 s (power iteration 41 calls 14.22 s + compile 1.42 s; 1 probe(s))

  samples              t (fs)    wall (s)       s/fs    ETA (s)    dSigma    |P| (au)  n_c (/cell)   |E| (V/m)    dTr/Tr  dPur/Pur
  --------------------------------------------------------------------------------------------------------------------------------
  0-4                  0.5000       13.54     27.079       67.7        53   1.826e-13    1.443e-20   1.240e+00   0.0e+00   0.0e+00
  5-9                  1.0000       20.33     13.587       27.2        93   4.694e-13    1.087e-19   3.236e+00   0.0e+00   0.0e+00
  10-14                1.5000       27.13     13.586       20.4       133   1.237e-12    7.193e-19   8.186e+00   0.0e+00   0.0e+00
  15-19                2.0000       33.91     13.577       13.6       173   3.078e-12    4.378e-18   2.007e+01   0.0e+00   0.0e+00
  20-24                2.5000       40.70     13.574        6.8       213   7.421e-12    2.525e-17   4.770e+01   0.0e+00   0.0e+00
  25-29                3.0000       47.49     13.574        0.0       253   1.727e-11    1.369e-16   1.099e+02   0.0e+00   0.0e+00
  --------------------------------------------------------------------------------------------------------------------------------
```

A chunk of a new step or sample count compiles its program, the first always
and often the last, which takes the steps up to tf; the summary leaves those
chunks out of its steady rate and names them (in the stiff-split run the last
chunk holds 6 steps, not 5, and runs at 16.8 s per fs against 7.59). Here only the
first compiles. Chunks 2–6 make 200 ΔΣ calls in 33.95 s over
2.5 fs (80 calls and 13.58 s per fs), so one call costs at most 0.170 s,
consistent with STEPLAB's isolated 0.161 s per call (the bound also carries the
step's non-ΔΣ work). The gate's own figure in this run, 0.347 s, held the
first-use builds of the new module's kconv kernels (a cold kernel cache); the
summary prints the steady bound beside it for that reason. The field is still in
the pulse's leading tail (≤ 110 V/m), so the drifts read 0: the populations moved
by less than one part in 1e16 of Tr ρ, the check's resolution, and a zero there
says nothing either way.

The table's columns are fixed (`tdagw.td_log.PROGRESS_COLUMNS`), so
`tdagw.td_log.parse_progress_row` reads any row back. Tr ρ and Tr ρ² are computed
on the device at every sample with no ΔΣ call; both are invariants of the exact
flow with Γ = 0 and the collision off (the intraband term moves charge between
k but keeps the sums), so a drift far above rounding there is a defect.
With relaxation or collisions, purity can change; particle conservation has
the closure conditions of the [collision theory](../theory/td_agw_dynamical.md#damping).
The energy is not logged: it needs ΔΣ[ρ] at every sample. A run that a
gate refuses still closes its log: the gate's lines, the refusal's message and a
`REFUSED (… GATE <id>)` line.

## The collision term {#collision}

For a full-density screening comparison, use `collision_basis = "instantaneous"`
and `collision_screening = "joint"`. The material Hamiltonian and coherent
self-energy converge together at each refresh, using one set of energies for
the polarization, internal lines and column shells. `joint_frozen` provides
the matched reduced equilibrium-screening control. A small exploratory setup
is:

Warning: this example explicitly selects a finite-lag statistical **model**,
not physical equilibrium/lossless GW statistics. Its screened equilibrium FDT
is known to fail in general, including as the lag decay tends to zero.
Physical collective initial statistics/adiabatic switch-on are not implemented.
Without this explicit acknowledgement, active joint modes refuse before GW
setup. Do not interpret a trajectory as validated physical thermal relaxation.

```toml
collision_basis = "instantaneous"
collision_screening = "joint"
collision_statistics = "finite_lag"
collision_hamiltonian = "reference"
collision_source = ".../04_gw_oneshot_mpa_eqp/oneshot.in"
collision = "raw"
collision_refresh_sweep = "active"
collision_in_scattering = "full"
collision_rate_intermediates = "active"
collision_internal_pairing = "vertex"
collision_refresh_fs = 0.5
collision_refresh_df = 0.0
collision_snapshot_steps = 32
collision_snapshot_tol_ev = 0.0001
collision_snapshot_mixing = 0.7
collision_snapshot_acceleration = "anderson"
collision_density_rank = 8
collision_pole_order = 48
collision_projection_order = 400
collision_pencil_mb = 128
collision_fit_samples = 192
collision_fit_real_axis = false
collision_fit_tolerance = 0.02
collision_adiabatic_ev = 0.05
collision_fit_height_ev = 0.15
```

These are explicit underconverged controls within that statistical model.
Vary finite lag decay, fit height and collision regularization separately;
numerical convergence does not cure the known statistical/FDT failure.
Compare a zero-field
trajectory before interpreting pulse decay, and inspect the density spectrum,
number balance, snapshot residual and response/noise fit errors. `subtracted`
is available as a separate imposed equilibrium-counterterm model. The
`reference` setting is an explicit DFT/QP-reference-anchored model for
comparison to the original TD-aGW. The default `hf` instead solves the absolute
Hartree/Fock plus M_H material Hamiltonian; a DFT-prepared density need not be
stationary there. Both use bare HF increments, with no static reduced-W
subtraction. See the [theory owner](../theory/td_agw_dynamical.md#damping).

Converge density rank together with rational order and support count. If a
capacity guard refuses the larger fit, explicitly raise
`collision_projection_order` (causal-projection state count, default 400) or
`collision_pencil_mb` (each block Loewner pencil, default 128 MiB) to match
available host resources. Self-inverse q crossing can increase the realized
order before projection. Dense projection cost grows cubically with state
count; pencil factorization and copies require approximately eight times the
per-pencil storage. These controls raise capacity only: the independent
response and noise checks keep the same tolerances.

If upper-half-plane fits poorly resolve boundary noise, optionally set
`collision_fit_real_axis = true`. This also samples the exact bubble at the
real parts of the complex training supports and includes those values in
the same joint fit. The declared positive finite lag decay `collision_adiabatic_ev` makes
these boundary samples finite; it is not replaced by a tiny fit offset.
The real training grid must be disjoint from the real dressed-noise check
grid, so those check values remain independent and all acceptance gates
are unchanged. This doubles training supports (roughly quadrupling pencil
storage), adds sampling cost, and may require explicit capacity increases.
It is finite rational fitting, not real-frequency integration or a claim
of converged dynamics.

The joint snapshot's stability gate probes its installed coherent Hamiltonian.
It follows paired field-free trajectories, so a physical HF quench is not
mistaken for numerical perturbation growth. The stiff-split/joint-snapshot
combination is currently refused rather than using an unvalidated basis.

This exploratory executor projects the microscopic Coulomb interaction before
fitting both polarization components. It retains separate exact response and
noise realizations on their union pole bank; the small projected self-energy
is evaluated by direct momentum sums on the host. It is not the large-grid
FFT collision executor. Density rank and transfer order must be converged
before making quantitative claims.

`[lorrax] collision = "subtracted"` adds the on-shell GW collision term of a
GN-PPM bundle ([TD-aGW dynamical §1.5](../theory/td_agw_dynamical.md#damping)):
column-sampled one-sided MGKBA coefficients from the two branches of the
bundle's own Σ sweep, held between refreshes. The coherent principal-value
contribution enters ΔΣ once; the remaining affine collision flow occupies
the Magnus remainder slot or the explicit methods' right-hand side. It
needs `system = "bundle"` on a `compute_mode = gn_ppm` run; the keys are in the
[input reference](../input_reference.md#td-agw).

```toml
[lorrax]
system = "bundle"
bundle_input = "<07>/oneshot.in"          # a GN-PPM GW deck
dipole_file = "<07>/dipole.h5"
model = "isdf"
method = "cfm4"
dt = 25.0
collision = "subtracted"                  # C[rho] - C[rho_eq]: rho_eq is a fixed point
collision_refresh_fs = 0.5                # refresh the rates every 0.5 fs ...
collision_refresh_df = 1e-3               # ... or when a population moves by 1e-3
```

`subtracted` removes the equilibrium collision counterterm defined on the
owner page. This is separate from `collision_hermitian = "dynamic"`, which
subtracts the static-W change already carried by K δρ. `full` retains that
static change too, and `none` drops the held Hermitian update. Neither the
counterterm nor a diagnostic width should be confused with the complex
one-sided branch coefficient.

At the start the driver reruns the GW deck in its own process
(`outdir/collision_source`, no restart written; 34.7 s on the 516 6×6 bundle,
warm cache) to recover the pole model. The log's `Collision refreshes` section
(after the progress table) has the t0 receipt and one line per refresh.
Check the coefficient and counterterm diagnostics there, together with the
refresh trigger, sweep wall and trace change. The numerical collision
measurements quoted below were made with the previous arithmetic-average
operator; new trajectories require validation with the one-sided operator.

Read the t0 line first. Rates of the size of the GW run's regularization width
(`sigma_regularization_ev`, 0.25 eV by default) at band edges are its tails, not
scattering: on the 516 2v2c window they shrink five-fold at 0.05 eV. Use
`raw` only to see that drift: `C[rho_eq]` is not zero, and on 516 the raw form
moves Tr ρ by 0.14 per k in 2.5 fs. A refresh costs one window Σ sweep (3.2 s on
516 at P4, the incumbent call's cost; the flow and the dissipator are below a
millisecond), so the refresh cadence, not the step, sets the wall: two
refreshes per fs made the 516 drive 8.1 s per fs (steady chunks) against 0.31 s per
fs without the term.
A collision run does not restart from a checkpoint.

### Cheaper refreshes and refreshed screening {#collision-refresh}

Only the `active` split below is currently runnable. The `refreshed` example
is retained as a record of the earlier arithmetic-operator prototype and now
refuses at `td_collision_keldysh_screening`: it lacks the independent
nonequilibrium W< required by the column-sampled operator.

```toml
collision_refresh_sweep = "active"        # sweep the window's own intermediate bands per refresh
collision_screening = "refreshed"         # historical example: currently refused
collision_screening_gamma_ev = 0.001      # principal-value width of degenerate same-side pairs
```

`collision_refresh_sweep = "active"` sweeps only the window's own intermediate
bands at a refresh and adds the frozen bands' part, swept once at t0
([TD-aGW dynamical §1.5](../theory/td_agw_dynamical.md#damping)): on the 516
2v2c window 0.95 s per refresh against 3.3 s, on held rules, so the held
ΔΣ^h does not jump when a state starts to fill (the full sweep re-plans then:
0.38 meV, where the physical ΔΣ^h is 3×10⁻³ meV). The t0 line names both
parts' τ nodes.

`collision_screening = "refreshed"` (with the split sweep) recomputes W from
the window populations at each refresh — δχ0 at ω = 0 and iω_p, the Dyson solve
with W0 in v's place, the GW run's own pole fit — and sweeps the window Σ on the
new poles, the frozen part included: 4.3–5.4 s per refresh on 516 2v2c (δχ0
1.85 s, Dyson + fit 0.40 s, frozen sweep 1.10 s, active sweep 1–2 s). The t0 line
states the fixed point (δχ0 = 0, W = W0 and the poles bitwise the run's at
ρ_eq); each refresh line adds the walls, the invalid-mode count and flips,
max|ΔΩ| and max|δW|/|W0| at both supports. `collision_frozen_bands = [n, m]`
re-sweeps only the frozen bands next to the window: the rates survive it, Re Σ
does not (§1.6).

Refreshed screening is a measured prototype, not a production setting. On the
516 drives the subtracted form keeps `C[rho_eq]` at the t0 rates, and the
change of its regularization tails with W raises the carrier density by
43–58 %; `collision_rebase = "on"` (deck key; `auto` leaves it off for
refreshed GN-PPM screening) removes it. At 3e9 V/m the held ΔΣ^h jumps by up
to 77 meV between refreshes, following the invalid-mode flips in the refresh
lines, and P(t) does not converge with the cadence. Read the flips column
before trusting a run.

### Shared-pole screening {#collision-shared-pole}

This section records the earlier arithmetic-operator prototype. The deck below
now refuses at `td_collision_keldysh_screening`; the passive retarded pole bank
cannot stand in for a joint retarded/lesser response-and-noise realization.

```toml
collision_screening = "shared_pole"       # historical example: currently refused
collision_source = "../04_gw_oneshot_mpa_eqp/oneshot.in"   # the SPM GW deck (default: bundle_input)
collision_refresh_sweep = "active"
collision_refresh_fs = 0.25
# collision_rebase = "auto"               # on for shared_pole: C[rho_eq] on W(t)
# collision_spm_support_sites_ev = "<line> | <imaginary> | <height>"   # explicit supports (V2'); default: the ladder
```

Every refresh reruns the source deck's W stage (response bank, moments,
constructor) at the instantaneous populations and re-resolves the recipe there
([TD-aGW dynamical §1.4](../theory/td_agw_dynamical.md#w)); at ρ_eq the refit is
bitwise the run's model. This remains a retarded/population snapshot of
screening. It does not supply a general nonequilibrium W< or implement the
proposed joint response/noise realization described on the theory owner page.
On 516 2v2c it costs 34 s per refresh (refit 25 s,
window Σ 5 s, rebase 4 s; [§7](../theory/td_agw_dynamical.md#cost)) and the
run stays one program. Each refresh line adds the refit walls, K, the census
(Ω_min and the residue and static weight below 0.3 and 1 eV) and the gates
that did not pass (capacity:WARN is the run's own). Explicit support sites
take eV lists; an imaginary site at 0.1 eV is refused at carrier states by
the bank's rule construction, so V2′ uses 0.25 and 0.5 eV. Measured prototype,
not a production setting.

## Choosing an integrator {#integrators}

The first three rows were measured on four A100s on the 30×30 CrI3 ISDF model
(N_μ = 846, 0.161 s per ΔΣ call; the σ+ deck above at 5e6 V/m over 27.9 fs; P(t)
error against a Richardson reference; 2026-10-17; the `auto` row over the 3 fs of
the log check, its steady chunks). On this model a ΔΣ call
outweighs everything else in a step, so the wall follows the ΔΣ count.

| want | use | ΔΣ calls per fs | wall per fs | measured on |
|---|---|---|---|---|
| P(t) error ~1e-5 (the default choice) | `cfm4`, `n_corr = 2`, cubic predictor, `dt = 50` (also its largest stable step) | 80.5 | 13.1 s | 30×30 CrI3, error 9.8e-6 |
| P(t) error ~1e-4 | `cfm4` with `stiff_modes = 10`, `stiff_krylov = 120`, `dt` ≈ 82–100 | 49.0; 44.6 with samples on step ends (dt 90.7) | 8.8 s; 8.0 s | 30×30 CrI3 |
| no step choice | `dt = "auto"`: the largest step the gate's probe passes, ≈50–55 as plain; 97.54 as with the split on the 846 deck (probes at 135 and 114.8 as fail, growth 2.13 and 1.88 per step) | 41.0 with the split at 97.54 as | 7.59 s with the split | 30×30 CrI3, 3 fs, 2026-10-17 |
| Γ or the intraband term on, ΔΣ cheap | `etdrk4`, `dt` 25–50: Γ is exact in its linear part, while `cfm4`'s triple jump takes three times the exponentials | 160 at 25 as | 1.16 s against `cfm4`'s 5.17 s (Γ = 0.05 eV) and 9.41 s (intraband) | synthetic 30v × 14c, 30×30, 4.3 ms stand-in ΔΣ |
| TDGW's own steps, or a narrow, non-stiff window | `dp54`, the parser default (TDGW's RK45 replica) | 46 at rtol 1e-6 (error 3.5e-4) | | 6×6 CrI3 2v2c |

With the ISDF ΔΣ (0.16 s a call against 3.3 ms for an exponential) the extra
exponentials of `cfm4` with Γ cost little, so `cfm4` stays the cheaper scheme per
accuracy there. The split's gate and basis add 3.6 ΔΣ calls per fs over a 100 fs
run (232 + 132 calls) at a fixed dt; with `dt = "auto"` the gate probes down from its
static limit, 3 probes on the 846 deck (629 calls, 117.55 s, plus the basis's
132 calls, 23.62 s), once per run. At 1e-5 the split gains nothing (80.1 against 80.5 calls per
fs): the scheme is still fourth-order `cfm4` with the same error constant, so
accuracy keeps the step at 50 as. One correction with the linear predictor is
stable only with the split and is third order with 125× the error, so it saves
nothing. Converged (Anderson) Gauss nodes are not recommended: they cost 1.8–2.0×
more ΔΣ calls at matched error (branch `feat/td-agw-implicit-nodes-2026-10-17`,
not merged). The 6×6 and synthetic operating points and the reasons behind them
are on [choosing a method](../theory/td_agw_propagation.md#choosing).
