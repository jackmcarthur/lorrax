# AMENDMENT — THE PHDF5 WFN READER LOSES ITS DATASET HANDLE, BUT ONLY WHEN DENSIFICATION IS ACTIVE (2026-08-10)

**One defect, one clean discriminating pair, and a mechanism nobody has
established.**  `bse.exciton_bands` on a densified deck dies inside the
streaming-Galerkin wavefunction read with

```
[phdf5 ERROR rank=3] phdf5 read_kchunk_union: ds_id is invalid
ValueError: INVALID_ARGUMENT: phdf5 read_kchunk_union: ds_id is invalid
```

The same binary, the same deck and the same node run the same read
successfully when densification is switched off, and run it successfully
with densification on if the wavefunction reader is forced onto its host
backend.  So the phdf5 transport is not the defect by itself and the read
shape is not the defect by itself; what fails is the two together, and this
row records that pairing rather than guessing past it.

## Where it is raised

The failure is a bad HDF5 dataset identifier handed to the FFI reader, and
it surfaces at the bottom of the ordinary window-load path:

| frame | file |
|---|---|
| `main` | `src/bse/exciton_bands.py:1175` |
| `initialize_wfns` | `src/bandstructure/htransform.py:1225` |
| `streaming_galerkin_solve` | `src/bandstructure/htransform.py:255` |
| `load_psi_gflat_padded` | `src/common/wfn_transforms.py:1803` |
| `read_slabs` | `src/file_io/_slab_io_ffi.py:2275` |
| `_reader` → `inner(handle, offset_base, count_base)` | `src/ffi/io.py:507` |

Rank 3 of 4 raises; the fail-fast excepthook exits without teardown, and
`srun` then kills the peer ranks that are blocked in the collective rank 3
will never join.  The step's exit code is therefore 137 (SIGKILL on the
peers), not the raising rank's 1 — reading the exit code alone would
misfile this as an OOM, and the `lx` epilogue says so out loud.

## The discriminating pair

All three legs below ran in the same lane, on the same tree
(`/pscratch/sd/j/jackm/triangle_0810/tree` at `53fd80ea`), against the same
downfolded bundle (`densify_exp_2026-08-10/child191/tmp`, μ 960 → 191), on
the same node in the same session, at four ranks on four GPUs.

| leg | what differs | result |
|---|---|---|
| `xb_fine888` (round 5) | deck carries `bse_k_grid = 8 8 8` over a 4×4×4 W — **densification active**; wavefunction backend left at its default, which `auto` resolves to phdf5 | **rc=137**, `ds_id is invalid` on rank 3 |
| `xb_fine888_eager` (round 6) | **byte-identical argv, workdir and deck** — the sole difference is `LORRAX_WFN_BACKEND=eager` in the leg's environment | **rc=0** in 129.2 s, and this is the run that produced the delivered `xb_fine888.dat` and `.png` |
| `xb_coarse444` (round 5) | same driver and same bundle, deck carries **no `bse_k_grid` key** — no densification; backend left at its default and again resolves to phdf5 | **rc=0** in 41.1 s |

The third row is what makes the pair a discriminator rather than an
anecdote.  The non-densified control takes the *same* transport and the
*same* read: its log prints `[WfnLoader] read backend = phdf5 (auto, 4
processes) — collective MPI-IO read through the phdf5 FFI .so` and then
`Streaming Galerkin: nk=64, nb=20, nr=13824, n_mu=960, mesh=(2x2),
band_chunk=20 (0.53 GB/chunk)`, and the densified arm prints those two lines
with **identical numbers** immediately before it dies.  Neither the backend
choice nor the hyperslab geometry distinguishes the arms.

## Mechanism: UNKNOWN

Nobody has established why the handle goes bad, and this row deliberately
does not pretend otherwise.  What can be read off the logs is one structural
difference, offered here as an untested hypothesis and labelled as such:

> In the densified arm, and only there, a **collective `H5Fclose` on
> `WFN.h5`** falls between the loader's phdf5 open and the streaming-Galerkin
> read.  `xb_fine888.log` runs `[WfnLoader] read backend = phdf5` (line 137),
> then the C1 head re-attachment, then `[SlabIO.close] … calling H5Fclose
> collectively` on `WFN.h5`, then `Streaming Galerkin: …`, then the error.
> The coarse arm goes from the same `[WfnLoader]` line straight to
> `Streaming Galerkin` and `[wfn_loader] shard_map via jax.shard_map` with
> nothing closing anything in between.  The shape of that — a borrower
> closing a file another reader still holds a dataset id into — matches the
> symptom, but **no one has tested it**: no leg has removed the close, kept
> the densification and re-read, which is the one experiment that would turn
> this paragraph into a mechanism.

Anyone picking this up should run that experiment before writing a fix.  The
cheap first probe is to log the `ds_id` and the owning file id at open and at
`read_kchunk_union` entry in both arms and diff them; if the id is the same
integer and only the file behind it has gone, the hypothesis is confirmed and
the fix is ownership, not the reader.

## The workaround, and what it costs

`LORRAX_WFN_BACKEND=eager` on the leg's environment.  The loader then does a
host `h5py` read per rank instead of the collective MPI-IO read, and the
densified run completes.  It is correctness-neutral on this deck — the
delivered fine-arm result is the eager run's — and it is a transport choice,
not a numerical one.  The cost is the read: 129.2 s for the eager fine leg
against 41.1 s for the (smaller) coarse leg on the collective path, so this
is a workaround to name in a report, not one to leave in a production deck
silently.

**No gate covers this.**  The suite has no densified `exciton_bands` cell on
the phdf5 transport, which is why a defect this reproducible reached a
delivery leg.  A cell that runs the densified path at four ranks with the
backend left at `auto` is the missing red twin; it belongs in
`tests/multi_device/`, because the failure needs four real processes and an
in-process multi-device mesh cannot express it.

## Evidence

* `/pscratch/sd/j/jackm/triangle_0810/_logs/xb_fine888.log` — the failure, rank 3, lines 287–383.
* `/pscratch/sd/j/jackm/triangle_0810/_logs/xb_fine888_eager.log` — the same leg green under `LORRAX_WFN_BACKEND=eager`.
* `/pscratch/sd/j/jackm/triangle_0810/_logs/xb_coarse444.log` — the non-densified control, phdf5, green.
* `/pscratch/sd/j/jackm/triangle_0810/round5.jsonl`, `round6.jsonl` — the two manifests, showing the argv are identical and the environment is not.
* `/pscratch/sd/j/jackm/triangle_0810/_logs/summary.json` — round 6, `rc=0`, 129.16 s.
