# Which host FFI .so is canonical

`LORRAX_FFI_HOST_SO` points at one `liblorrax_ffi_host.so`.  This directory
holds many hand-named stage dirs and, until 2026-08-02, none of them recorded
which revision it was built from.  A path is not an identity: a harness echoes
the path it *intends* from a shell variable a later line may override, and two
libs with the same filename can differ by months of source.

That cost real debugging time on 2026-08-02 — `build_host_ONE`, the lib behind
every certified b600 result, turned out to predate the tree it was being
compared against (453 exported symbols vs 475), and nothing on disk said so.

## Current

**CANONICAL: `build_host_IMPLICITPAD/`** — `fix/slab-io-audit` @ ca9bad5b
(recorded in the sibling `SRCREV`, because the lib is built from a frozen
bundle so `build_ffi_host.sh` can only stamp `git_rev=unknown +dirty`).
Point new harnesses here.  It carries, in order of when they landed:

* `d935ce7` — the empty-selection fix, so a wholly-padded rank does not
  refuse a write that was always a no-op;
* `8050e34` — the bounds test taken on the LOGICAL slab in BOTH
  `write_ffi.cc` and `read_ffi.cc`, so a refusal inside a collective fires
  on every rank or none;
* `98afe58` — `ensure_dataset` verifies an existing dataset's shape and
  dtype and refuses a change, naming both shapes (decisions.md 2026-08-04).
  This is the only C++ delta vs `build_host_SLABIO`: exported-symbol diff
  between the two is EMPTY, 475 both.

sha256 `51c69680d078bf85`, built 2026-08-04T17:14:01Z, job 7888641.
Certified by jobs 7888650 (P=4) and 7888651 (P=16): all eight padded-rank
gate cases, all four SlabIO probe cases, the 3x3 writer x reader matrix at
0.000e+00, and the write byte-identity gate.

`build_host_SLABIO/` — the audit revision (8050e34), sha `f0ecf821bcb005ae`.
Superseded by IMPLICITPAD; kept as the A/B control that attributed the
`ensure_dataset` check, and because jobs 7888525 / 7888537 name it.

`build_host_PADFIX/` — e3141e0 + `d935ce7`, sha `13a0b667`.  Superseded; it
has the empty-selection fix but NOT the logical-slab bounds test, so a caller
that overruns the dataset while some ranks are wholly padding still hangs.
Kept because jobs 7886446-7886458 and 7888470 name it.

`build_host_PADBASE/` — e3141e0 UNFIXED.  It exists solely as the one-line
control that attributed the empty-selection fix (jobs 7886450 / 7886451).
Not for production.

`build_host_ONE/` — revision unknown, unrecoverable, and missing the fix.  Kept
because the b600_p64, b600_bispinor_p64 and zeta_T_prodkappa harnesses point
at it and their logs must stay truthful about what they ran.  Do not repoint
those harnesses; they are records of completed campaigns, not live configs.

Every other `build_host_*` here is an unstamped campaign artifact.  Treat any
lib whose `PROVENANCE` is missing, or marked `BACKFILLED=yes`, as unidentified
until rebuilt.

## Going forward

`config/frontera/build_ffi_host.sh` now writes a `PROVENANCE` file beside every
`.so` it builds (git rev, dirty flag, sha256, symbol count, build host, SLATE
on/off), and refuses to be silent about an uncommitted source tree.
`ffi.common.ffi_loader.library_provenance` reads it and the runtime startup
report prints it, so **every job log now records the build it actually loaded**
rather than the path someone meant to use.

The three `PROVENANCE` files written on 2026-08-02 are reconstructions, marked
`BACKFILLED=yes`.  A reconstruction is not a record — only libs built after
this date carry a real build-time stamp.
