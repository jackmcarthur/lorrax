# Quickstart

This page runs every LORRAX driver end to end on the bundled test fixture,
`tests/hsuite/fixture`: a tiny magnetic H2+ crystal (one electron,
noncollinear with spin-orbit, 9 bands, 5×5×1 k). It needs the native FFI pair
([Installation](installation/index.md); on Perlmutter,
[steps 1–2](installation/perlmutter.md)). To start from a crystal instead, first
produce a `WFN.h5` ([Preparing inputs from DFT](preprocessing.md)).

## 1. Run the bundled fixture

The fixture ships its own `WFN.h5`, pseudopotential and QE inputs. The chain
copies it into a new directory, runs every driver there (GW one-shot and 2-map
QSGW, BSE, htransform, exciton bands; the stages are listed in
[Contributing](contributing.md#the-test-suite)) and writes each driver's deck
beside its outputs:

```bash
cd /path/to/lorrax
source config/perlmutter/gpu_env.sh
srun --jobid=$JOBID -N 1 -n 1 --gpus-per-node=1 \
  .venv/bin/python -m tests.hsuite.chain --out "$SCRATCH/lorrax_quickstart_$(date +%s)"
```

This is one GPU (P1), which is enough for a smoke run; the suite below is the
P4 verdict. With `lx` (project m4598) the launch is
`lx run --pool POOL -N 1 -G 1 -n 1 -- python -m tests.hsuite.chain --out …`.

On Frontera, build the host leg with `config/frontera/build_ffi_host.sh` and
launch as [Frontera](environment/machines/frontera.md) describes.

## 2. Check the answer

Each stage is compared with the stored outputs in `tests/hsuite/reference/`.
The run ends with one line per stage and its wall time, then `hsuite PASS`, or
an `hsuite FAIL` line naming the stage, the quantity and its deviation. The
decks, reports and outputs of every driver are in `<out>/run/`.

## 3. Run the test suite

The suite is the same chain at P4 on one node.
[Contributing](contributing.md#the-test-suite) owns the command.

## Your first real calculation {#your-first-real-calculation}

The run directory holds `WFN.h5` ([Preparing inputs from DFT](preprocessing.md)),
the `*.upf` pseudopotentials of the DFT run, and one deck. The deck below is a
complete static (COHSEX) one-shot; every key it omits takes its default.

```ini
[cohsex]
wfn_file = WFN.h5
centroids_file = centroids_frac_<n>.txt   ; written by step 1
sys_dim = 3           ; 3 bulk, 2 slab (truncated Coulomb); required
nval = 4              ; valence bands in the output window
ncond = 8             ; conduction bands in the output window
number_bands = 80     ; top of the chi0 and Sigma band sums, at most the WFN's
compute_mode = cohsex
```

The chain is three preprocessing steps and GW, each as its own launch
(`srun … .venv/bin/python -u -m …`, or `lx run`), all with the same deck:

1. **Centroids:** `python3 -m centroid.kmeans_cli <N> --seed 42`. It reads
   `WFN.h5` from the working directory (there is no flag for another name),
   and writes `centroids_frac_<n>.txt`, where `n` is the count that survives
   deduplication and pruning, which can be below `N`. Set `centroids_file`
   from its `Saved centroids to …` line. Start at N = 10 × `number_bands`
   ([how many](theory/isdf-exchange-accuracy.md)).
2. **Dipoles:** `python3 -m psp.get_dipole_mtxels -i gw.in` → `dipole.h5`.
3. **Kinetic + ionic:** `python3 -m gw.kin_ion_io -i gw.in` → `kin_ion.h5`.
4. **GW:** `python3 -m gw.gw_jax -i gw.in`.

The quasiparticle energies are in `eqp0.dat` and `eqp1.dat` (BerkeleyGW
format, eV, irreducible wedge); `gwjax.out` is the run report. Every output
file is listed in [drivers](drivers.md#gw-gwgw_jax). A `dipole.h5`,
`kin_ion.h5` or restart bundle that does not match the deck or the WFN refuses
by name; rerun the step that wrote it.

From here:

- **Production QSGW.** Change the deck to the keys of
  [production QSGW](how-to/production-qsgw.md) (`compute_mode = mpa`,
  `sigma_w_model = shared_pole`, `qp_solver = self_consistent`) and rerun
  step 4. A metal also needs [the metal keys](how-to/metals.md).
- **Bands.** `python3 -m bandstructure.htransform -i ht.in --qp-rotations
  qp_wfn_rotations.h5` interpolates the QP bands along the deck's
  `K_POINTS {crystal_b}` path. Its window rules (`nval` equal to the electron
  count, guard bands) are in
  [drivers](drivers.md#htransform-bandstructurehtransform).
- **Excitons.** `python3 -m bse.bse_jax -i gw.in --lanczos --tda --bse` in
  the GW run directory. It reads the restart bundle the GW run wrote ([drivers](drivers.md#bse-bsebse_jax),
  [BSE](architecture/bse.md)).

Take key defaults from the [input reference](input_reference.md#system) and do
not copy routing or layout keys from older decks: the driver prints every
automatic choice as `[config provenance]`. For bispinor decks, the
[four-current physics scope](theory/four-current-head-corrections.md) and the
[wiring record](architecture/four_current_wiring.md) say which conditional
artifacts a material needs.

## Where to next

- [Preparing inputs from DFT](preprocessing.md): QE → `pw2bgw.x` → `WFN.h5`
- [Installation](installation/index.md) and
  [Building the FFI libraries](building_ffi.md)
- [Theory overview](theory/overview.md) and [physics](theory/physics.md)
- [Codebase](codebase.md) and [memory model](architecture/memory-model.md)
