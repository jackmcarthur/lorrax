# Quickstart

This page runs LORRAX on the bundled test fixture, `tests/hsuite/fixture`: a
tiny magnetic H2⁻ crystal (three electrons, noncollinear with spin-orbit, 9
bands, 5×5×1 k). §2 is a first GW calculation, one step at a time. §3 runs
every driver in one command and checks the answers. §4 is the same chain on
your own crystal. LORRAX must be installed first
([Installation](installation/index.md)).

## 1. The launch line {#launch}

Every command on this page runs on a GPU compute node, never on a login node.
Get one node and work in the shell it opens:

```bash
salloc -N 1 -C gpu -G 4 -q interactive -t 1:00:00 -A <account>
```

Then set two shell variables. `LORRAX` is the source tree, which holds the
fixture. `RUN` is the launch prefix: 4 processes, one GPU each.

With the module stack (the Perlmutter default;
[using the module](installation/perlmutter-module.md#using-the-module)), where
`<prefix>` is the path the module was published to:

```bash
module use <prefix>/modulefiles && module load lorrax
LORRAX=$LORRAX_ROOT
RUN="srun -N 1 -n 4 --gpus-per-node=4 $LORRAX/src/ffi/cpp/select_gpu.sh python -u -m"
```

With a clone ([Perlmutter clone](installation/perlmutter.md), steps 1–2):

```bash
LORRAX=/path/to/lorrax
source $LORRAX/config/perlmutter/gpu_env.sh
RUN="srun -N 1 -n 4 --gpus-per-node=4 $LORRAX/src/ffi/cpp/select_gpu.sh $LORRAX/.venv/bin/python -u -m"
```

With `lx` (project m4598; it needs no `salloc`), `LORRAX` is a checkout and
`RUN="lx run -N 1 -G 4 -n 4 -- python3 -u -m"`.

On Frontera, build the host leg with `config/frontera/build_ffi_host.sh` and
launch as [Frontera](environment/machines/frontera.md) describes.

## 2. A first GW calculation {#first-calculation}

Start in an empty directory and copy the three input files:

```bash
mkdir $SCRATCH/lorrax_first && cd $SCRATCH/lorrax_first
cp $LORRAX/tests/hsuite/fixture/{WFN.h5,H.upf,data-file-schema.xml} .
```

| file | what it is |
|---|---|
| `WFN.h5` | the DFT wavefunctions, in BerkeleyGW format |
| `H.upf` | the pseudopotentials of the DFT run, one `*.upf` per species |
| `data-file-schema.xml` | Quantum ESPRESSO's record of the run, from its `.save` directory. It says which symmetry operations include time reversal. A magnetic WFN needs it. |

Write the deck, `gw.in`. This is a complete static (COHSEX) one-shot; every
key it omits takes its default.

```ini
[cohsex]
wfn_file = WFN.h5
centroids_file = centroids_frac_70.txt   # written by step 1
sys_dim = 3           # 3 bulk, 2 slab (truncated Coulomb); required
nval = 1              # valence bands in the output window
ncond = 2             # conduction bands in the output window
number_bands = 7      # top of the chi0 and Sigma band sums
compute_mode = cohsex
```

A comment starts with `#`. `number_bands` must leave at least one band of the
WFN above it and must not cut a degenerate multiplet; a deck that does either
refuses and names the fix.

Run three preprocessing steps and GW, all with the same deck:

```bash
$RUN centroid.kmeans_cli 70 --seed 42 -i gw.in   # 1. centroids
$RUN psp.get_dipole_mtxels -i gw.in              # 2. dipoles
$RUN gw.kin_ion_io -i gw.in                      # 3. kinetic + ionic
$RUN gw.gw_jax -i gw.in                          # 4. GW
```

| step | writes | time |
|---|---|---|
| 1 | `centroids_frac_<n>.txt`, `kmeans.out` | 17 s |
| 2 | `dipole.h5`, `dipole.out` | 17 s |
| 3 | `kin_ion.h5`, `kin_ion.out` | 17 s |
| 4 | `eqp0.dat`, `eqp1.dat`, `sigma_diag.dat`, `gwjax.out`, `qp_wfn_rotations.h5`, `tmp/` | 40 s |

The times are on 4 A100 GPUs with warm caches. The first run on a new
install is slower, because the GPU kernels compile once.

- **Step 1** takes the requested centroid count `N` (here 70). It reads
  `WFN.h5` from the working directory; there is no flag for another name.
  `n` is the count that survives deduplication and pruning, and it can be
  below `N`. Read the file name from the `Centroids :` line of the output
  (or `ls centroids_frac_*.txt`) and set `centroids_file` to it before step
  2. Here it is `centroids_frac_70.txt`.
- **Every file** is written to the working directory. Each `*.out` file is
  that step's report; the terminal shows the same report.
- **A step has finished** when its report ends with `LORRAX … completed.`
  A failed step prints a `REFUSED (…)` or `FAIL-FAST` line with the reason.

The quasiparticle energies are in `eqp0.dat` and `eqp1.dat` (BerkeleyGW
format, eV, irreducible wedge). Each k-point block has a header line (the
k-point in crystal coordinates and the band count), then one line per band:
spin, band, the DFT energy, the quasiparticle energy. The first block of
`eqp0.dat` from this run, which two runs reproduce to every digit:

```text
  0.000000000  0.000000000  0.000000000       5
       1       1   -6.677240544   -6.946319626
       1       2   -5.778933971   -6.297506608
       1       3   -4.885122491   -4.519243420
       1       4   -3.479877316   -1.320867637
       1       5    0.663657572    2.662949929
```

Every output file is listed in [drivers](drivers.md#gw-gwgw_jax).

The fixture is magnetic, so each report lists `TIME-REVERSAL SYMMETRY IS
BROKEN` notices under `WARNINGS`. Step 2 prints `parallel-transport artifact:
not written`: the cell has too little vacuum for that optional artifact.

## 3. Every driver, checked {#chain}

The chain copies the fixture into a new directory and runs every driver there
(GW one-shot and 2-map QSGW, BSE, htransform, exciton bands; the stages are
listed in [Contributing](contributing.md#the-test-suite)). It writes each
driver's deck beside its outputs. Run it from the source tree:

```bash
cd $LORRAX
$RUN tests.hsuite.chain --out "$SCRATCH/lorrax_quickstart_$(date +%s)"
```

It takes about 6 min at one process on cold caches. The chain writes nothing
in the source tree, so it also runs from the module's read-only tree: its
files go to `--out` and to `$SCRATCH/.cache/lorrax/hsuite`.

Each stage is compared with the stored outputs in `tests/hsuite/reference/`.
The run ends with one line per stage and its wall time, then `hsuite PASS`, or
an `hsuite FAIL` line naming the stage, the quantity and its deviation. The
decks, reports and outputs of every driver are in `<out>/run/`.

The test suite is the same chain under pytest.
[Contributing](contributing.md#the-test-suite) owns the command.

## 4. Your own crystal {#your-first-real-calculation}

1. Make `WFN.h5` from a DFT run ([Preparing inputs from DFT](preprocessing.md)).
2. In a new directory, put `WFN.h5`, the `*.upf` files of the DFT run, and
   `data-file-schema.xml` from the `.save` directory of the run that wrote
   the WFN.
3. Copy the deck of §2 and change four keys: `sys_dim`, `nval`, `ncond` and
   `number_bands`.
4. Run the four steps of §2. Start the centroid count at N = 10 ×
   `number_bands` ([how many](theory/isdf-exchange-accuracy.md)).

A `dipole.h5`, `kin_ion.h5` or restart bundle that does not match the deck or
the WFN refuses by name; rerun the step that wrote it.

From here:

- **Production QSGW.** Change the deck to the keys of
  [production QSGW](how-to/production-qsgw.md) (`compute_mode = mpa`,
  `sigma_w_model = shared_pole`, `qp_solver = self_consistent`) and rerun
  step 4. A metal also needs [the metal keys](how-to/metals.md).
- **Bands.** `$RUN bandstructure.htransform -i ht.in --qp-rotations
  qp_wfn_rotations.h5` interpolates the QP bands along the deck's
  `K_POINTS {crystal_b}` path. Its window rules (`nval` equal to the electron
  count, guard bands) are in
  [drivers](drivers.md#htransform-bandstructurehtransform).
- **Excitons.** `$RUN bse.bse_jax -i gw.in --lanczos --tda --bse` in
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
