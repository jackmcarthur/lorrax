# Quickstart

This page runs one static-COHSEX calculation end to end on the bundled
regression fixture, `tests/regression/cohsex_debug`, on one process. It needs
the native FFI pair, which the Perlmutter `lorrax_A` module supplies
([Installation](installation/index.md)). To start from a crystal instead, first
produce a `WFN.h5` ([Preparing inputs from DFT](preprocessing.md)).

## 1. Run the bundled fixture

The fixture ships its own wavefunction (`WFNsmall.h5`), centroids
(`centroids_frac_60.txt`), `dipole.h5` and `kin_ion.h5`, so it needs no
preprocessing. It is read-only and GWJAX writes beside the deck, so copy it to
a writable directory on a filesystem the compute nodes see:

```bash
export LX_BASE_MODULE=lorrax_A
export LORRAX_CHECKOUT=/path/to/lorrax       # the copy is outside the checkout
QS=$(mktemp -d -p "$SCRATCH")
mkdir -p "$QS/tests/regression"
cp -a "$LORRAX_CHECKOUT/tests/regression/cohsex_debug" "$QS/tests/regression/"
chmod -R u+w "$QS"
cd "$QS"
lx run --pool POOL --wait 900 -N 1 -G 1 -n 1 -- \
  python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

On Frontera, build the host leg with `config/frontera/build_ffi_host.sh` and
launch as [Frontera](environment/machines/frontera.md) describes.

## 2. Check the answer

The run writes `eqp_test.dat` beside the deck and the frozen `eqp_ref.dat`.
Compare them without the first line, which is a generation timestamp:

```bash
cd "$QS/tests/regression/cohsex_debug"
diff <(sed 1d eqp_test.dat) <(sed 1d eqp_ref.dat)
```

A match prints nothing.

The bundled WFN has no co-staged QE `*.save` directory, so the run prints
`SYMMETRY PROVENANCE WARNING` and uses the global DFT time-reversal verdict.
This is expected for the fixture. Production inputs co-stage the QE `*.save`
directory that generated the WFN.

## 3. Run the test suite

`lx test` runs the two-minute core tier on a Perlmutter compute node.
[Contributing](contributing.md) owns the tiers and when each is required.

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
- [Codebase](architecture/codebase.md) and [memory model](architecture/memory-model.md)
