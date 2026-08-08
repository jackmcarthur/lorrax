# Reading wavefunctions and ζ: the slab transport

LORRAX begins every calculation by reading somebody else's file. The
mean-field wavefunctions arrive as a BerkeleyGW-format `WFN.h5`, and partway
through the run LORRAX writes and then re-reads a file of its own, `zeta_q.h5`,
holding the fitted ISDF basis. Both are large, both are read by every process at
once, and both have a padding convention that is easy to get subtly wrong. Two
services own them: `wfn_loader` for ψ(G), `zeta_loader` for ζ.

## Why this is not just `h5py.File(...)`

A production `WFN.h5` is tens of gigabytes and LORRAX runs one process per
device. The obvious implementation — every rank opens the file and reads what it
needs — has three problems that only appear at scale.

First, the read is *sharded*: the band axis is distributed over the mesh, so each
rank wants a different, ragged slice, and the slices do not line up with anything
in the file's layout. Second, doing that as N independent POSIX reads means N
processes hammering the same Lustre stripes; the filesystem serves it, badly.
Third, and least obvious, the ragged part is where the bugs live. Band counts do
not divide evenly by the number of ranks, G-vector counts differ from k-point to
k-point, and every array LORRAX builds is rectangular. Something has to decide
how much of each rank's tile is real.

The transport answer is **collective MPI-IO**: one `H5Dread` in which every rank
declares the hyperslab it wants and the library schedules the whole thing as a
single coordinated operation. LORRAX reaches that through `SlabIO`
(`src/file_io/slab_io.py`), which is the one door between a sharded `jax.Array`
and an HDF5 dataset — see [SlabIO](architecture/slab_io.md) for the transport
itself. The loaders described here are its two biggest clients.

## One read, not a loop: `read_slabs`

A wavefunction read is not one window into the file. It is *n* windows — one per
k-point in the requested set — that share a shape and differ in where they start
and how much of that shape is real. The tempting implementation is a loop over
`read_slab`. It was measured, and it loses on every deck: warm-minimum ratios of
3.58× and 3.22× at fixture scale, and 1.44× (+2.1 s per read) on a MoS₂ 12×12
400-band deck with 144 windows and 15.6 GB. About 1.4 s of that is per-call
collective `H5Dread` overhead — 144 calls at ~42.7 ms against one — and ~0.6 s is
a `jnp.stack` the union path never has to do.

What makes that decisive rather than merely unfortunate is the *axis* the cost
grows along: `n` is the number of IBZ k-points in the request, which is exactly
the axis production decks grow. So n windows became a request the door serves,
not a loop a caller writes. `SlabIO.read_slabs` takes the common slab shape, a
table of per-window offsets, a table of per-window logical extents, and a window
axis to insert in the output, and issues **one** compound-hyperslab read. The
caller owes it one precondition: the windows must be pairwise disjoint and sorted
in the file's own row-major order, because they are combined with an
`H5S_SELECT_OR` and an overlap would be a double selection.

Promoting the wavefunction loader onto this primitive deleted its hand-copied
open guard, its private counts table and every FFI import from it — 125 lines
net — and the promoted path measured bit-identical and 0.7 % faster on the
production deck. That is the shape of a good extraction: fewer concepts, no
change in the numbers.

## `WfnLoader`: one class, three ways to ask

`WfnLoader(path, mesh=…)` opens the file, reads the header and measures
time-reversal symmetry eagerly; ψ itself stays lazy. There are three ways to get
coefficients, and they are different primitives rather than flags:

* **`load(bands=…, k=…)`** returns one logical *global* array,
  `(n_k, nb_padded, ns, ngkmax)`, band axis mesh-padded and sharded. Every rank
  must request the same window; it is a collective.
* **`load_process_local(bands=…, k=…)`** returns *this* process's window only,
  single-device, with `nb = b_hi − b_lo` exactly — no mesh padding, no
  collective. Each rank may ask for something different. This is the
  k-parallel primitive.
* **`bands(b_lo, b_hi, chunk=…)`** iterates `load` in chunks.

Alongside those sit the header surface (`nkpts`, `nbands`, `nspinor`, `kgrid`,
`fft_grid`, `bvec`, and the derived `nelec`/`vbm`/`cbm`/`efermi`), the G-vector
tables, the FFT-box index tables, and two *measured* properties — `trs_holds` and
`density_symmetry` — which are facts about the coefficients rather than
inferences from a flag in the header. `symmetry_maps` is their consumer.

There are two transports underneath. `eager` is per-rank serial h5py with a host
unfold, used single-process, mesh-less, or when forced. `phdf5` is the collective
`read_slabs` path with an on-device unfold, used when there is a mesh, more than
one process, and the door probe holds. **They are byte-identical** — verified
with `np.array_equal` and no tolerance, at four processes, on hostile geometry,
on both CPU and CUDA platforms — which is the only reason it is safe to let
`LORRAX_WFN_BACKEND` select between them. Where a transport is genuinely
unavailable the loader **refuses** and names the way through rather than quietly
demoting; a deleted spelling (`phdf5_host`) refuses rather than resolving to
something else, because silently resolving a deleted name to another backend is
how an A/B ends up measuring the arm nobody asked for.

## The padding contract, and why the pad is not zero

Every array here is rectangular and the physics is ragged, so padding is
unavoidable. The contract is a **conjunction**, and both halves matter:

1. Pad slots of ψ — both the band-axis pad rows and the G-axis pad columns —
   hold **zero**. That makes them inert in any contraction.
2. The matching rows of `gvecs()` beyond `ngk_valid()` hold the **pad sentinel**:
   a Miller index at the FFT box's Nyquist corner, a cell no physical G occupies.

The first half alone would be the obvious design and it is a trap. Zero
coefficients with zero G-vectors are indistinguishable from a real
coefficient at Γ, so a consumer that forgot the validity mask would silently add
`ngkmax − ngk` copies of ψ(Γ) to its sum and produce a plausible wrong number.
With a sentinel, dropping the mask is *detectable* — and consumers do detect it,
via `refuse_padded_gvecs_without_mask`. Note carefully that detectable is not the
same as optional: `gvecs()` and `ngk_valid()` are a pair, and consuming the first
without the second is an antipattern with a name.

The single definition of the sentinel value lives in `common/gvec_fft_box.py`,
which also refuses to build a table in which a *physical* G would land on the
sentinel cell — because if that happened, "this row is the sentinel" would no
longer imply "this row is pad", and the whole scheme would quietly stop working.

The band clamp — how much of each rank's tile is real, `max(0, min(slab, logical
− offset))` per dimension — lives in exactly one place,
`_slab_io_ffi._derive_window_counts`, three definitions above the clip it
delegates to. It used to have a local copy in the loader; the copy diverged on the
band axis and wrote real file bands into pad rows on every non-divisible
geometry. A structural test greps for re-inlining.

## `zeta_q.h5`, and restarting from it

`zeta_q.h5` is written once by the ISDF fit and read many times afterwards — by
the V_q build, by the BSE interpolation, by basis projection, and by the fit-reuse
gate. `zeta_loader` owns its format and nothing else: the fit itself, the
`zeta_rcond` tier, the solver ladder are all producer-side.

Three design choices are worth knowing. **There is one layout.** Every data
method reads the G-flat `zeta_q_G` dataset and refuses anything else *by name*.
The previous arrangement had the layout dispatch in three copies, one of which
probed only the legacy dataset name and therefore passed silently on exactly the
production files it was guarding. **Agreement is checked at open, not assumed** —
header μ against dataset μ, header `ngkmax` against the dataset's G axis —
because the collective plan sizes itself from the header and the local plan from
the dataset, so a disagreement would have the two reading different extents with
no complaint. **Completeness is checked at open**: `zeta_is_done` is flipped by
the writer only after the last write drains, so a fit that died mid-write refuses
to be read (`LORRAX_ALLOW_PARTIAL_ZETA=1` overrides, for forensics only).

The header surface works with no mesh and no FFI anywhere — `mesh=None` is a
supported mode, and the package's import is jax-free until first attribute access
so that a login-node interpreter can inspect a file. With a mesh, one `SlabIO`
handle is opened and held for the loader's lifetime, which amortises the MPI
context; the measured difference between holding the handle and opening per read
is 2.3–9.4× on CPU meshes and 1.8–5.9× on CUDA.

**Restart.** A rerun does not refit ζ if it does not have to. The reuse gate
gives `True` only when the file exists, its header is readable, `zeta_is_done` is
set, the recorded fit provenance is byte-identical to this run's, the on-disk
centroid table matches, and the dataset's μ extent is what this run expects.
Anything unexpected — a missing attribute, an unreadable file, a legacy header —
means refit. That asymmetry is deliberate: every failure mode costs compute and
none costs correctness, because this cache sits in front of a multi-hour step
whose silent misuse produced a −135 eV quasiparticle gap once already.
`LORRAX_FORCE_REFIT` is the operator's escape hatch.

## Traps

* **Do not open the ζ file with raw `h5py` outside the service** for data or
  layout facts. The local plan you are about to hand-roll is
  `read_zeta_G_local`, and it works at `mesh=None`.
* **Do not make the local plan collective**, or wrap it in a `SlabIO` read "for
  consistency". It is non-collective *by contract*; making it collective turns a
  rank-0 diagnostic into a hang.
* **Do not write padded extents to disk.** Files store LOGICAL extents so they
  re-read identically at any process count.
* **Do not put `__slots__` or a strict `__setattr__` on `WfnLoader`.** The
  pseudopotential path attaches `grid_rho` to the instance at runtime and
  `gw/kin_ion_io.py` reads it back; locking the class breaks it silently.
* **Do not `jax.device_put` a numpy array with a multi-process sharding** in a
  loader path. It fires JAX's hidden `assert_equal` and therefore an allgather —
  6.45 GB per rank at 64 processes for the G-index table.
  `device_put_process_local` is the spelling.

The contracts in full are [`docs/services/wfn_loader.md`](services/wfn_loader.md)
and [`docs/services/zeta_loader.md`](services/zeta_loader.md); the transport
underneath is [SlabIO](architecture/slab_io.md).
