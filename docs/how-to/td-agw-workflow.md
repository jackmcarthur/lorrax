# TD-aGW end to end

This page puts the real-time TD-aGW steps in order, from a GW run to a
driven density matrix, and names the module and the owner page of each step.
It restates no equation, key or refusal: the equation of motion, the
integrators and the output are on [TD-aGW propagation](../theory/td_agw_propagation.md),
the interaction ΔΣ[ρ] on [TD-aGW interaction](../theory/td_agw.md), every
deck key in the [input reference](../input_reference.md#td-agw), and each
invocation on [drivers](../drivers.md#td-agw-tdagwtdgw_jax).

## The chain

| step | runs | produces | owner |
|---|---|---|---|
| 1. GW run with a static W | `gw.gw_jax` | the restart bundle `tmp/isdf_tensors_<mu>.h5` with a ready `W0_qmunu`, its `zeta_q.h5`, `eqp1.dat` | [drivers](../drivers.md#gw-gwgw_jax) |
| 2. TDGW's input directory | data preparation | `indir/WFN.h5` (full zone, the bundle's k grid) and `indir/eqp.dat` | [below](#tdgw-inputs) |
| 3. TDGW header | TDGW `tdgw-run`, once | a TDGW output directory whose `tdgw.h5` header holds the window's energies, k grid and dipoles | [propagation](../theory/td_agw_propagation.md#refusals) |
| 4a. drive on the GPUs | `tdagw.tdgw_jax` | `tdgw.h5` + `tdgw.toml` per drive | [propagation](../theory/td_agw_propagation.md) |
| 4b. TDGW with the LORRAX ΔΣ | `tdagw.tdgw_bridge` | TDGW's own output | [drivers](../drivers.md#tdgw-kernel-and-bridge-tdagwkernel_export-tdagwtdgw_bridge) |
| 4c. a dense kernel for TDGW | `tdagw.kernel_export` | `kernel.h5` | [interaction](../theory/td_agw.md#units-normalization-and-the-tdgw-kernel-file) |
| 5. spectra | TDGW `tdgw-pp linear` | ε(ω) from P(t) | TDGW |

## 1. The bundle {#bundle}

The ISDF model reads the bundle through the BSE loader: ψ at the centroids,
ζ, V and the static W (`W0_qmunu`); the bundle's band range must contain the
TD window. A shared-pole MPA run on an identity-only k grid refuses to write
`W0_qmunu`; `compute_mode = cohsex` (static RPA W0) writes it. To propagate with BerkeleyGW's screening on the same
representation, `gw.epsmat_cli` with `mode = project` writes a copy of the
bundle whose `W0_qmunu` is BerkeleyGW's ε⁻¹ projected on ζ; `mode = export`
goes the other way ([ε⁻¹ export](../theory/bgw_epsmat_export.md)).

## 2. TDGW's inputs and header {#tdgw-inputs}

- `WFN.h5`: the bundle's k grid on the full zone (TDGW uses no symmetry). The
  model, the exporter and the bridge refuse a `WFN.h5` whose k points are not
  the full bundle grid.
- `eqp.dat`: the GW run's quasiparticle energies in TDGW's reading, one block
  per k point of that `WFN.h5`, in its order ([BerkeleyGW users](berkeleygw-users.md)
  gives the `eqp1.dat` layout). Diagonal QP energies can reorder bands inside
  the valence manifold.
- The window (`system.ibndmin`, `ibndmax`, `ivbm`) is in TDGW's 1-based DFT
  band indices. Its edges should close degenerate multiplets in the DFT and
  the QP spectrum; the BSE's `--band-degeneracy strict` (its default) on the
  same window is the check.
- The header: run TDGW once on the window. With `control.files.use_vmtxel =
  true` it carries BerkeleyGW's dipoles; on coarse grids TDGW's own
  finite-difference dipoles can break the point group (the propagation page
  measures it). Every later deck points `control.files.load_header_from` at
  that directory.

## 3. The ΔΣ route {#routes}

1. **On the GPUs** (`tdagw.tdgw_jax`): `[lorrax] model = "isdf"` with
   `isdf_input` = the GW deck of the bundle. ΔΣ is applied matrix-free at every
   evaluation; the integrator and step are `lorrax.method` / `lorrax.dt`
   ([choosing a method](../theory/td_agw_propagation.md#choosing)).
2. **Inside TDGW** (`tdagw.tdgw_bridge tdgw.toml --input <GW deck>`): TDGW's
   own propagation with its self-energy update replaced by the same operator,
   the full kernel or TDGW's TDA semantics (`--tda`). More than one rank needs
   an h5py built against parallel HDF5.
3. **As a dense kernel** (`tdagw.kernel_export`): `kernel.h5` in TDGW's
   convention, for `lorrax.model = "tdgw_kernel"` or a native TDGW run (TDA
   files only, `--tda`, at TDGW 41a5c34).

The three apply the same ΔΣ; they differ in who integrates (LORRAX's
integrators or TDGW's RK45) and in what is formed (nothing, or the dense
kernel). Other self-energy updates enter through the same contract,
`apply(params, rho, t)` ([the equation of motion](../theory/td_agw_propagation.md#eom)).

## 4. Drives {#circular-drives}

A weak short pulse along x, then y, gives the linear response; its poles are
those of the full BSE of the same kernel with no V head
([algorithm](../theory/td_agw.md#algorithm)), so `bse.bse_jax` on the same
window is the cross-check. Circular drives follow the
[helicity convention](../architecture/bse.md#helicity) (σ+ turns x → y,
counter-clockwise seen from +z); `lorrax.trials` propagates σ+ and σ−
together, one output directory each. TDGW expressions take t in as, so a
drive at ħω = 1.5777 eV has ω = 2.39695e-03 rad/as (ħ = 658.2119569 eV·as):

```toml
[control]
calc_type = "COHSEX"

[control.timesteps]
t0 = 0
tf = 60000
dt_measure = 5

[control.files]
indir = "indata"
outdir = "out_circular"
load_header_from = "header_run"

[system]            # CrI₃ monolayer: 42 valence x 14 conduction bands
ibndmin = 89
ibndmax = 144
ivbm = 130
geometry = "slab"

[measurements]
types = ["Efield", "polarization"]

[lorrax]
method = "dp54"
model = "isdf"
isdf_input = "../gw/cohsex.in"
trials = [
  {E_x = "1e7*np.exp(-((t-30000)/10000)**2)*np.cos(2.39695e-03*(t-30000))", E_y = "1e7*np.exp(-((t-30000)/10000)**2)*np.sin(2.39695e-03*(t-30000))"},
  {E_x = "1e7*np.exp(-((t-30000)/10000)**2)*np.cos(2.39695e-03*(t-30000))", E_y = "-1e7*np.exp(-((t-30000)/10000)**2)*np.sin(2.39695e-03*(t-30000))"},
]
```

Fields are in V/m. The first trial is σ+, the second σ−.
