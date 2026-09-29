# Quickstart

This page runs every LORRAX driver end to end on the bundled test fixture,
`tests/hsuite/fixture`: a tiny magnetic H2+ crystal (one electron,
noncollinear with spin-orbit, 9 bands, 5×5×1 k). It needs the native FFI pair,
which the Perlmutter `lorrax_A` module supplies
([Installation](installation/index.md)). To start from a crystal instead, first
produce a `WFN.h5` ([Preparing inputs from DFT](preprocessing.md)).

## 1. Run the bundled fixture

The fixture ships its own `WFN.h5`, pseudopotential and QE inputs. The chain
copies it into a new directory and runs kmeans → kin_ion → dipole → GN-PPM
one-shot → shared-pole QSGW (2 maps) → BSE → exciton bands there,
writing each driver's decks beside it:

```bash
cd /path/to/lorrax
lx run --pool POOL --wait 900 -N 1 -G 1 -n 1 -- \
  python -m tests.hsuite.chain --out "$SCRATCH/lorrax_quickstart_$(date +%s)"
```

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

Given a `WFN.h5` in the run directory, the chain is three preprocessing steps
and GW, each as its own `lx run`:

1. **Centroids:** `python3 -m centroid.kmeans_cli <N> --seed 42`. It reads
   `WFN.h5` from the working directory (there is no flag for another name),
   and writes `centroids_frac_<n>.txt`, where `n` is the count that survives
   deduplication and pruning, which can be below `N`. Set `centroids_file`
   from its `Saved centroids to …` line.
2. **Dipoles:** `python3 -m psp.get_dipole_mtxels -i cohsex.in` → `dipole.h5`.
3. **Kinetic + ionic:** `python3 -m gw.kin_ion_io -i cohsex.in` → `kin_ion.h5`.
4. **GW:** `python3 -m gw.gw_jax -i cohsex.in`.

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
