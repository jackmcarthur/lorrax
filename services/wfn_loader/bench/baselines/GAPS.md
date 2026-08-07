# wfn_loader baselines — what is NOT measured

A baseline directory that lists only what was measured reads as though the
measured set were the whole space. These are the holes, named, so the next
person spends their cluster time on a gap rather than re-deriving that one
exists. All three are REGISTERED for post-wave; none blocks the land.

Current coverage: `cpu2x2.json` — 7 rows, one platform (CPU), one mesh
(2×2), two decks, three windows. That is it.

## 1. GPU-platform read timings — **the biggest hole**

The L-c cells ran **green on the GPU 2×2 leg**, so the phdf5 path is known
to be CORRECT on CUDA. It has never been **timed** there. Every number in
`cpu2x2.json` is CPU milan.

Why that is defensible today, and exactly how far it goes: the same C++
read core serves both platforms and only its device-staging tail switches
(`cudaMemcpyAsync` H2D vs a host `memcpy`), and the in-tree measurement at
`read_ffi.cc:819-829` records ~1% end-to-end from async overlap — which is
why the fold RULING carries to CUDA. That argument is about the *ordering*
of the two arms, not about the absolute rate. **The MB/s figures here are
not CUDA figures and must not be quoted as such**, and CUDA is the
production path.

To close: `lx run -N 1 -G 4 -n 4 python3
services/wfn_loader/bench/bench_wfn_loader.py --mesh 2x2 --tag gpu2x2`.
The driver takes it as written; it needs a GPU allocation, not new code.

## 2. The large-deck CUDA leg

`mos2_400b` (144 IBZ windows, 15.6 GB) is the only deck at production
scale, and it has been read on CPU only. It is also the deck the fold
ruling turns on — the 1.44× row — so a CUDA run of the same deck is the
one measurement that could still surprise. It needs the deck staged
(`--wfn`, 15.6 GB, not checked in) plus a GPU allocation with enough host
memory for the cold read.

## 3. End-to-end `WfnLoader.load` on the cluster

`bench_wfn_loader.py --paths load` produces these rows and they run
locally, but **no cluster artifact records them**, so there is no row in
`cpu2x2.json`. This is a real gap in how the read numbers can be READ: a
`read_slabs` figure is only interesting as a fraction of the thing a driver
waits for, and that fraction is currently unmeasured at P>1.

Recorded so it is not mistaken for one: an earlier draft of
`docs/services/wfn_loader.md` carried a row reading "0.017–0.019 warm /
0.084 cold" for `load(k='ibz')` on gnppm (0,82). **It was dropped, because
no artifact I could read supports it.** If it turns out to be sourceable,
it belongs here as a row with a jobid; if it does not, the run above
produces the real one. Do not re-import it from the draft.

The `eager` comparison arm has the same status — the two backends are
byte-identical by contract, so the only thing such a row can show is time,
and nobody has recorded that time at P>1 on the cluster.

## Not a gap, deliberately

* **`gpu1x1` / `cpu1x1`.** The service's collective path does not exist at
  P=1 (`auto` resolves to `eager`, by design), so a 1×1 file would be a
  baseline for the eager path wearing the collective's name.
* **A threshold anywhere.** Regression detection is diffing these files
  across branches. A threshold that has to hold on a shared machine either
  gets loosened until it means nothing or fails on somebody else's
  contention.
