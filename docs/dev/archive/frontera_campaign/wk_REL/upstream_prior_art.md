# Upstream prior art for the two LORRAX defects

*Web research completed 2026-07-29. Every claim below carries a URL or a
command that can be re-run. Where a search returned nothing, that is stated
as a negative result rather than inferred.*

---

## VERDICT (read this first)

| | prior art upstream? | fixed upstream? | action |
|---|---|---|---|
| **Defect 1** — silent corruption in XLA:CPU gloo reduce-scatter | **NONE. Clean negative.** | no | **File a new issue.** It is novel. |
| **Defect 2** — `MPI_Is_thread_main` guard in `MpiCollectives::CreateCommunicators` | **One PARTIAL match**, [openxla/xla#16430](https://github.com/openxla/xla/issues/16430), closed 2024-09-03 **as completed without touching the MPI guard** | **no — guard is verbatim in `main` today** | **File a new issue** (do not just comment on #16430; it was closed by fixing the *gloo* path and its MPI half was explicitly deferred). |

Neither defect is fixed in any released jaxlib, including today's latest
(0.11.0). See §A.

**§0 records a factual premise of `UPSTREAM_gloo_psum_scatter_corruption.md`
§4.6c that was wrong — it changed both the suspect code and the repo the issue
should be filed against. That correction has now been applied to the corruption
report itself; §0 is kept as the evidence record.**

---

## 0. CORRECTION to §4.6c — **APPLIED 2026-07-29**

> **Status: done.** `UPSTREAM_gloo_psum_scatter_corruption.md` has been
> corrected against the evidence below — §1 summary, §4.6c (rewritten, with a
> method-lesson box), §4.6d (mechanism added), §4.6e (motivation restated),
> §4.6f (32-bit element-count hazard added), §6 item 2, §7 item 0, and a new
> §7b carrying the upstream-status facts from §3 of this document. The section
> is retained here as the evidence record for *why* the correction was made.

The pre-correction §4.6c asserted *"this gloo build has no reduce-scatter
algorithm at all"* and
concludes that `GlooCommunicator::ReduceScatter` is XLA's own
allgather-then-local-reduce. **Both halves are false**, and a maintainer would
find this within one minute of reading the issue.

**(1) gloo's reduce-scatter IS in the build.** `gloo/reduce_scatter.h` is a
header-only template, so it emits no `external/gloo/gloo/*.cc` path string —
which is why the `strings` census in §2/§4.6c missed it. The symbols are
present in the installed wheel:

```
$ nm -C .../jaxlib/libjax_common.so | grep -c ReduceScatterHalvingDoubling
112
0000000000cc6c5a0 t gloo::ReduceScatterHalvingDoubling<std::complex<double> >::ReduceScatterHalvingDoubling(...)
0000000000cc6dab0 t gloo::ReduceScatterHalvingDoubling<std::complex<double> >::run()
0000000000cc6e350 t gloo::ReduceScatterHalvingDoubling<std::complex<double> >::getDistributionMap(...)
```

`std::complex<double>` is exactly our dtype.

**(2) The anonymous-namespace helper §4.6c found belongs to a different
communicator.** The symbol
`xla::cpu::(anonymous)::ReduceScatter<PrimitiveType>(ReductionKind, absl::Span<const void* const> inputs, void* output, long count)`
is defined at line 198 of
[`xla/backends/cpu/collectives/in_process_communicator.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/in_process_communicator.cc)
— the **InProcessCommunicator** (single-process, multiple emulated devices).
It is not on the gloo path at all.

**What the gloo path actually does**
([`gloo_communicator.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/gloo_communicator.cc)):

```cpp
Future<> GlooCommunicator::ReduceScatter(send_buffer, recv_buffer, dtype, count, kind, exec) {
  size_t chunk_bytes = count * ByteWidth(dtype);
  std::unique_ptr<char[]> temp(new char[chunk_bytes * context_->size]);   // 253 MB for us
  std::memcpy(temp.get(), send_buffer.opaque(), chunk_bytes * context_->size);
  ... ReduceScatterHelper<std::complex<double>>(context_, kind, temp.get(), count) ...
  std::memcpy(recv_buffer.opaque(), temp.get(), chunk_bytes);             // always offset 0
}

// ReduceScatterHelper:
gloo::ReduceScatterHalvingDoubling<T> algorithm(
    context, std::vector<T*>{reinterpret_cast<T*>(buffer)},
    chunk_elems * context->size, recv_elems, reduction_function);
algorithm.run();
```

So the real suspect is **`gloo::ReduceScatterHalvingDoubling::run()`**, a
recursive-halving/distance-doubling algorithm that reduces **in place** into a
caller-owned buffer and uses an internal `recvBuf_` of the full payload
(`gloo/reduce_scatter.h:128`). Two properties of that code make it a far better
fit for our evidence than XLA's thin wrapper:

* it is **in-place** and works out of internal scratch buffers, so "foreign
  bytes of a magnitude the operands cannot produce" (§4.6d's `1.021892e+04`,
  34x the operand bound) is a *reachable* outcome — content of gloo's own
  scratch, not of our operands. That is precisely the class §4.6b/§4.6c/§4.6e
  excluded and §4.6d could not name;
* the final result lands at **offset 0** of the buffer and XLA unconditionally
  `memcpy`s from `temp.get()` — consistent with §4.2's "always segment 0".

**Consequence for filing.** The issue should still go to **jax-ml/jax or
openxla/xla** (that is where the caller lives, where the reproducer runs, and
where the gloo pin is chosen), but it must be described as *"XLA:CPU's gloo
reduce-scatter, i.e. `gloo::ReduceScatterHalvingDoubling` as driven by
`GlooCommunicator::ReduceScatter`"* — and **pytorch/gloo should be
cross-referenced**, because that algorithm is theirs and is essentially
abandoned (§C).

Also flag as a scaling hazard, not a claim about our failure:
`ReduceScatterHalvingDoubling`'s constructor takes `const int count`
(`gloo/reduce_scatter.h:115`), so the element count is 32-bit. Ours
(15,852,672) is safely inside `int`, but a complex128 reduce-scatter above
~32 GiB total would overflow it. This is a *different* mechanism from the
32-bit **byte** count that §4.6f refuted, so §4.6f does not cover it.

---

## 1. Defect 1 — candidate reports, all judged

Searches run (GitHub issues+PRs search API, open **and** closed, titles and
bodies, all three repos), with result counts:

| search | repo | hits | anything relevant |
|---|---|---|---|
| `psum_scatter gloo` | jax-ml/jax | 0 | — |
| `reduce_scatter gloo` | jax-ml/jax | 0 | — |
| `gloo corruption` | jax-ml/jax | 0 | — |
| `psum_scatter incorrect` | jax-ml/jax | 1 | unrelated PR |
| `cpu collectives wrong` | jax-ml/jax | 5 | none on CPU collectives |
| `gloo` (**every** issue/PR ever) | jax-ml/jax | **11** | **none about wrong results** |
| `gloo` (**every** issue/PR ever) | openxla/xla | **26** | all build/MacOS/plumbing |
| `gloo incorrect` | openxla/xla | 0 | — |
| `reduce-scatter incorrect` | openxla/xla | 16 | all GPU/SPMD-partitioner |
| `reduce_scatter` | pytorch/gloo | 4 | one candidate, see below |
| `incorrect` / `corruption` | pytorch/gloo | 12 / 2 | one candidate, see below |

The `gloo` sweeps are exhaustive: **11 items in jax-ml/jax and 26 in
openxla/xla is the complete lifetime population**, and I read every title.

### Candidates, with judgement

**[pytorch/gloo#303 — "output of reduce_scatter is incorrect"](https://github.com/pytorch/gloo/issues/303)**
OPEN since 2021-03-23, last activity 2021-12-02, 1 comment, no maintainer
response. Same algorithm (`ReduceScatterHalvingDoubling`).
**UNRELATED — and specifically, do NOT cite it as corroboration.** The
reporter builds `sendbuf` by pushing a pointer to each of 12 individual `int`s
(`sendbuf.push_back(&buffer_data[i])` twelve times) while passing `count = 12`.
gloo's `ptrs` argument is *N buffers of `count` elements each*, so that call
tells gloo to read 12 elements from each of 12 one-element addresses. The
garbage output is out-of-bounds reads caused by API misuse, not a gloo defect.
Our call site passes a single correctly-sized buffer. Citing #303 would weaken
our report. It is worth one sentence only as evidence that *nobody has ever
successfully reported on this code path* — the one attempt was invalid and sat
unanswered for five years.

**[pytorch/gloo#66 — "Infiniband AllReduceHalvingAndDoubling error"](https://github.com/pytorch/gloo/issues/66)** (closed, 2017) and
**[pytorch/gloo#34 — "CudaAllreduceHalvingDoubling writev error"](https://github.com/pytorch/gloo/issues/34)** (closed, 2017).
**UNRELATED.** Halving-doubling family, but both are *crashes* on transports
we do not use (IB verbs, CUDA), from 2017, and both are loud failures.

**[openxla/xla#16430](https://github.com/openxla/xla/issues/16430)** — see §2;
its *gloo* half was a rendezvous timeout, not corruption. **UNRELATED to defect 1.**

**[jax-ml/jax#39100 — "Silent wrong results on 4 GPUs: multi-output fusion aliases two in-place dynamic-update-slice outputs to the same buffer"](https://github.com/jax-ml/jax/issues/39100)**
OPEN, 2026-07-11. **UNRELATED** despite the near-identical headline. GPU-only
(explicitly "correct on CPU, any device count"), a compiler buffer-aliasing bug
in `priority-fusion`, deterministic per device count, no collectives involved.
Useful only as a *style template* — it is a recent, well-received silent-wrong-
results filing.

**[openxla/xla#42481](https://github.com/openxla/xla/issues/42481)** (closed
2026-05-12, SPMD partitioner orders dynamic-slice before all-reduce) and
**[openxla/xla#40034](https://github.com/openxla/xla/issues/40034)** (open
2026-03-28, partitioner omits an all-reduce when resharding unreduced axes).
**Both UNRELATED**, and our own data rules the whole class out: these are
compile-time partitioner faults, so they would be 100% reproducible and
backend-independent. Ours is intermittent within a single process and the
**mpi** backend runs the *same HLO* clean in 604/604 with a bit-identical
association floor (§4.6, §4.6a).

### Conclusion on defect 1

**No upstream report matches. This is a clean negative.** In eight years of
gloo history and the entire lifetime of XLA's CPU collectives, there is no
report of silent numerical corruption from `reduce_scatter` on the gloo path.
Our report is novel and should be filed.

---

## 2. Defect 2 — one partial match, closed without fixing our half

**[openxla/xla#16430 — "Segfault when using CPU collectives plus --xla_force_host_platform_device_count=2"](https://github.com/openxla/xla/issues/16430)**
Reported 2024-08-23 by @heiner, assigned @hawkinsp, **closed 2024-09-03 as
`completed`**, 8 comments.

**PARTIAL MATCH.** It quotes our error verbatim —
`"MPI: Communicator requested from a thread that is not the one MPI was
initialized from. Multiple threads/devices per process are not yet supported."`
— and it is the *only* place on GitHub where that string appears (search for
the quoted phrase across all of GitHub returns exactly 1 result: this issue).

**But it is not our bug, and it was not fixed:**

* **Different trigger.** Theirs is *many devices in one process*
  (`--xla_force_host_platform_device_count=2`). Ours is **1 device per
  process** — the guard fires because XLA:CPU's parallel `ThunkExecutor`
  issues the collective thunk from an intra-op pool worker. The guard's own
  message ("multiple threads/devices per process") describes their case, not
  ours; ours is a case the message does not anticipate.
* **It was closed by fixing gloo, not MPI.** @hawkinsp:
  *"I'll try to get the gloo support working today"* → merged
  [openxla/xla#16640 "[XLA:CPU] Allow multiple gloo communicators in the same process"](https://github.com/openxla/xla/pull/16640).
  For MPI he wrote: *"I'm not sure that's possible: I don't think you can have
  multiple ranks in the same process participate in the same collective. So to
  make it work with MPI we'd need to implement some sort of hierarchical
  collectives ... that's more work."* The MPI path was **explicitly deferred
  and never revisited**.
* **The guard is untouched today.** Fetched from `openxla/xla@main` on
  2026-07-29,
  [`xla/backends/cpu/collectives/mpi_collectives.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/mpi_collectives.cc)
  still reads exactly:

  ```cpp
  void MpiCollectives::Init() {
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);   // `provided` never read
    ...
  }
  ... CreateCommunicators(...) {
    int flag;
    MPI_Is_thread_main(&flag);
    if (!flag) { return absl::UnknownError("MPI: Communicator requested from a thread that is not "
                                           "the one MPI was initialized from. ..."); }
  ```

  `git log` on that file shows **zero commits since at least 2025-06-01**
  (`GET /repos/openxla/xla/commits?path=xla/backends/cpu/collectives/mpi_collectives.cc&since=2025-06-01`
  returns an empty list). Every claim of `jax_threadmain_alternatives.md` §0(a)
  and §0(c), read out of our local binary, is confirmed against upstream `main`.

* **The "too lax" half is also confirmed upstream.** `mpi_communicator.cc` has
  had only three commits since mid-2025 — `[xla:collectives] Migrate from
  tsl::AsyncValueRef to xla::Future` (2025-10-02), `[xla:cpu] Migrate XLA:CPU
  to se::DeviceAddress` (2025-12-06), `[XLA] NFC: Use ABSL macros` (2026-05-21)
  — all mechanical. No thread checks have been added to `AllReduce`,
  `ReduceScatter`, `AllGather`, `AllToAll`, `CollectivePermute`, `Broadcast`,
  `Send` or `Recv`.

**Action.** File a **new** issue. Reference #16430 as related-but-distinct
("same guard, different trigger; #16430's MPI half was deferred"). The draft in
`jax_threadmain_alternatives.md` §4.2 is accurate against current `main` and
can be pasted as-is; add the #16430 back-reference and the note that
`MPI_Comm_create_group` would remove the cross-rank ordering hazard.

No other prior art exists: `MPI_Is_thread_main` returns **0 hits** across
jax-ml/jax, openxla/xla, and GitHub-wide when paired with `jax`.

---

## 3. Answers to the four specific questions

### (a) Is jaxlib 0.9.1 current? Would upgrading fix either defect? — **No, and no.**

jax/jaxlib **0.9.1 was released 2026-03-02**; the current release is
**0.11.0 (2026-07-16)**, five releases later
([PyPI](https://pypi.org/pypi/jax/json): 0.9.2, 0.10.0, 0.10.1, 0.10.2, 0.11.0).

**Upgrading changes nothing for either defect.** I resolved the dependency
pins for both tags and diffed the implicated source:

| | jax 0.9.1 | jax 0.11.0 |
|---|---|---|
| XLA pin (`third_party/xla/revision.bzl`) | `3cc8846c10052cc1c32c4db87866eac4e4cdbccd` | `131bf41acb4650e4391a640c3f1859c1c86ad74b` |
| gloo pin (`third_party/gloo/workspace.bzl`) | `54cbae0d3a67fa890b4c3d9ee162b7860315e341` | **`54cbae0d…` — identical** |
| `ReduceScatterHelper` body | — | **byte-identical** |
| `GlooCommunicator::ReduceScatter` body | — | differs only by `TF_RETURN_IF_ERROR` → `RETURN_IF_ERROR` (the 2026-05-21 NFC commit); **semantically identical** |
| `MPI_Is_thread_main` guard | present | present |

The [jax CHANGELOG](https://github.com/jax-ml/jax/blob/main/CHANGELOG.md)
entries for 0.9.2, 0.10.0, 0.10.1, 0.10.2 and 0.11.0 contain **no mention** of
gloo, MPI, CPU collectives, reduce-scatter, or multi-process CPU. (0.10.0's
only distributed-related fix is a `jax.distributed.initialize()` GCE metadata
parsing bug, {jax-issue}`#36593` — unrelated.)

Upgrading is still worth doing for other reasons, but **do not present it as a
mitigation**, and a maintainer will not be able to close our issue as
"fixed in a newer version".

### (b) Have XLA's CPU collectives been rewritten or deprecated? — **No.**

`xla/backends/cpu/collectives/` has the same 25 files and the same
three-implementation structure (`gloo_*`, `mpi_*`, `in_process_*`). Commits to
the **entire directory** since 2026-01-01 number **seven**, and all are
mechanical: license headers, an ABSL-macro NFC pass, two "Automated Code
Change" commits, a test disable/re-enable pair for `cpu_cliques_test`
(2026-06-17 / 2026-06-19), and `PR #40537: [xla] Allow exchanging mutable data
via rendezvous`. **No redesign, no deprecation, no correctness work.**

Direction of travel, such as it is: gloo became the **default** in Feb 2025
([jax#26264](https://github.com/jax-ml/jax/pull/26264), *"multi-process CPU
communication works out-of-the-box"*), and the legacy alias flag was deleted in
Sept 2025 ([jax#31884](https://github.com/jax-ml/jax/pull/31884)). So the
defective path is the one every JAX user gets by default, which is worth
stating in the issue.

### (c) Is gloo documented as unsuitable for production? — **Not in those words, but close enough to quote.**

* [pytorch/gloo README](https://github.com/pytorch/gloo) carries a banner:
  > "🔵 NOTE: Gloo is considered to be feature complete and in
  > **maintenance-only mode**. For new usecases or non-bugfix changes please
  > reach out to the maintainers to discuss."

  The same README enumerates its primitives as *"a barrier, broadcast, and
  allreduce"* — **reduce-scatter is not listed**.
* **`gloo/reduce_scatter.h` has not been functionally modified since
  2018-02-09.** Its complete history is three commits: `ReduceScatter CPU
  Implementation` (2018-02-09), `Remove PATENTS clause` (2018-12-12), and
  `Applying CLANGFORMAT formatting` (2024-10-02). Eight years, zero bug fixes,
  one unanswered correctness report (§1).
* [gloo `docs/algorithms.md`](https://github.com/pytorch/gloo/blob/main/docs/algorithms.md)
  documents `reduce_scatter_halving_doubling` including a reordering phase
  ("*due to nature of recursive halving algorithm ... the blocks are not
  ordered in correct order. Enforced correct reorder by exchanging data between
  processes p and p', where p' is the bit-reverse of p*"). It states no
  production caveat.
* On the JAX side there is **no guidance at all**. `jax/_src/config.py` gives
  the whole story in three lines — `enum_values=["gloo","mpi","megascale"]`,
  help text *"Cross-process collective implementation used on CPU."* — and
  [`docs/multi_process.md`](https://github.com/jax-ml/jax/blob/main/docs/multi_process.md)
  (941 lines) mentions **neither "gloo" nor "mpi" even once** (`grep -ci`
  returns 0 for both).

So: nobody documents gloo's CPU path as unfit for production, but nobody
documents it as fit either, and its reduce-scatter is unmaintained 2018 code
reached by default from a framework whose multi-process docs never name it.
That gap is itself worth one line in the issue.

### (d) Is the main-thread clique warm-up a sanctioned pattern? — **Unsanctioned, but explicitly the direction upstream wants to go.**

There is **no public API, no documentation and no example** anywhere in jax or
XLA for pre-creating CPU communicators. Searching for it returns nothing.

However, [`xla/backends/cpu/collectives/cpu_cliques.cc`](https://github.com/openxla/xla/blob/main/xla/backends/cpu/collectives/cpu_cliques.cc)
carries this comment immediately above `AcquireCommunicator` (line 118):

```cpp
// TODO(b/380457503): Consider switching to a lockable CPU clique model similar
// to GPU cliques, and creating all communicators upfront.
absl::StatusOr<Communicator*> AcquireCommunicator(...)
```

and the surrounding comments confirm the mechanism route 1b depends on:

```cpp
// CpuClique is not thread-safe, so we wrap it in a thread-safe container as we
// create new communicators lazily and potentially from multiple threads.
// ... CPU cliques are not lockable, and we create communicators lazily when needed.
```

**Assessment.** Warming mesh-axis cliques from the main thread is an
*accident of the current lazy-creation implementation* — it works because
`AcquireCommunicator`'s process-global cache is keyed on the participating
device set alone, which is an internal detail with no stability promise. It is
**not** a sanctioned pattern. But it is aligned with an upstream TODO that
proposes exactly "creating all communicators upfront", and internal bug
b/380457503 is the right thing to cite. Two practical consequences:

1. Route 1b is safe to ship now, and safe to describe in the issue as a
   workaround — but LORRAX should keep the `TF_CPP_VMODULE=cpu_cliques=3`
   check in its belt, because if upstream ever adopts the lockable model the
   cache-key behaviour could change.
2. The issue should **ask for the sanctioned form**: either fix the guard
   (route 3), or expose an upfront-communicator-creation entry point, and cite
   b/380457503 as evidence that the maintainers already want the latter.

---

## 4. Near-misses worth watching

| item | why watch it |
|---|---|
| [openxla/xla#16430](https://github.com/openxla/xla/issues/16430) | the only other appearance of our defect-2 error string; its MPI half is an open loop a maintainer may remember |
| [pytorch/gloo#303](https://github.com/pytorch/gloo/issues/303) | invalid as a bug report, but shows the reduce-scatter path has never had a valid one |
| [jax-ml/jax#28160](https://github.com/jax-ml/jax/issues/28160) | OPEN, "CPU backend init fails with gloo error on Mac OS" — unrelated (init failure, macOS) but shows the gloo path still gets triaged |
| [jax-ml/jax#32673](https://github.com/jax-ml/jax/issues/32673) | CLOSED, `[Gloo] "Expected number of connected peer" warning since 0.8.0` — unrelated, but the most recent gloo-path activity in jax |
| [jax-ml/jax#39100](https://github.com/jax-ml/jax/issues/39100) | OPEN, unrelated (GPU fusion aliasing) but a good recent template for a silent-wrong-results filing |
| `cpu_cliques_test` disabled then re-enabled, 2026-06-17 → 2026-06-19 | the only sign of instability in CPU collectives testing this year; commits `73a33cced4` / `bab23011cb` |

---

## 5. Recommended action

1. **File defect 2 first.** It is small, provable from `main`, has a drafted
   patch, and its prior art (#16430) is closed with the MPI half openly
   deferred. Low risk of a duplicate-close, and landing it deletes both of
   LORRAX's MPI overrides.
2. **File defect 1 as a new issue against jax-ml/jax**, with a
   cross-reference to pytorch/gloo. §4.6c is **already corrected** (§0) — file it as
   *"`gloo::ReduceScatterHalvingDoubling`, as driven by XLA:CPU's
   `GlooCommunicator::ReduceScatter`, intermittently returns foreign data"*.
   The four exclusion passes (§4.6b–e), the `MALLOC_PERTURB_` result, the
   604/604 mpi control and the 34x-over-bound single event are all intact and
   are the strongest part of the report; only the "where the code lives"
   paragraph needs replacing. Lead with the fact that gloo is the **default**
   CPU collectives backend and that the failure is silent.
3. **Do not** wait for or recommend a version upgrade as a mitigation — 0.11.0
   ships byte-equivalent code (§3a).
4. State plainly in the issue that our reproducer runs, that we can run further
   experiments, and that the remaining work needs an ASAN/debug build of
   `ReduceScatterHalvingDoubling::run()` — which now has a concrete named
   target rather than "XLA's local reducer".

---

## 6. What I could not check

* **Google-internal bug b/380457503** is not publicly readable; I know only its
  title from the TODO comment.
* **jax-ml/jax Discussions** are not indexed by the issues search API. Keyword
  web searches over them surfaced nothing on gloo correctness, but that is a
  weaker negative than the issue sweep.
* I did not audit `ReduceScatterHalvingDoubling::run()` line by line — §0
  identifies it as the suspect and gives the reasons, but does not root-cause it.
