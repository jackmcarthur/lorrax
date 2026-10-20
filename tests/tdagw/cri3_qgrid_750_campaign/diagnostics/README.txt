PASSED exact-shape P16 service diagnostic — synthetic values only

Corrected job 60051836.0 completed on all 16 ranks with zero inferior exit
codes. The actual provider was cusolver_syevd_ffi. Maximum execute time
was 1.198434387 s; residual 3.344770295e-17 and orthogonality error
1.072567404e-14 both passed the unchanged tolerance 7.821654435e-11.
Backend compilation was observed on every rank: counter deltas were
[1 probe, 0 persistent hits, 1 backend compile]. No cold cache was forced.

Measured admission per rank was 1,030,043,688 compiled bytes plus
1,950,529,200 vendor-workspace bytes, totaling 2,980,572,888 bytes.
result.json preserves authenticated receipt fields and evidence hashes.
This does NOT reproduce or diagnose the original material constructor
SIGSEGV: the lost bank values and constructor batch route were not run.

reproduce_gathered_eigh_5504_p16.py is a bounded, service-only reproducer of
the recorded native gathered-eigh cache signature: one complex128 matrix
5504 x 5504, input/output-vector face P('x','y') on mesh 4 x 4, replicated f64
eigenvalues. The failed SC bank is unavailable: the values are explicitly
synthetic Hermitian values with eight exact-zero tail rows, not a replay
of a physical polarization, Gram matrix, or converged SC state.

The intentional archive pin is source a0c1882d1cc579b1670694615bb00f4b53b15b0d,
whose gathered-eigh implementation is unchanged from failed source 1134957.
The launcher must export matching LEG_HEAD and authenticated archive SHA,
and bind PYTHONPATH to that immutable source. Do not remove this pin just
because the preserved diagnostics directory is later rebased or moved.

Use only a separately authorized P16/4x4 step, one GPU per MPI rank.
Ops owns allocation and launch; this directory requests neither. Accepted
CLI used for the corrected diagnostic:
     --source-root IMMUTABLE_SOURCE --out NEW_SHARED_OUTPUT
     --max-wall-seconds 360 --device-ceiling-gib 8
Perlmutter requires --gpus-per-node=4 and select_gpu.sh, NOT
--gpus-per-task=1 (docs/environment/machines/perlmutter.md). Node-wide GPU
visibility is also required by the native intra-node collectives.
The corrected launch used an outer 390 s timeout plus collection/release
margin. The two diagnostic allocations consumed 93 + 41 = 134 reserved
seconds, or 0.595555556 GPUh at 16 GPUs. Their active steps consumed
84 + 33 = 117 seconds, or 0.52 GPUh. These accounting views overlap and
must NOT be added. Both allocations are released; no further job is
authorized by this result. Harness default is 540 s, allowed maximum 600 s.

p16_reproducer_gdb.commands is ops' reviewed batch-GDB script. It runs only
the newly launched inferior, never attaches to another user's process.
It preserves a normal inferior return code; SIGSEGV/SIGABRT stop with all
thread backtraces and exit 139. Ops verified these exit semantics against
CPU /bin/true, /bin/false and a deliberate CPU SIGSEGV before any GPU use.
The launch must retain rank-specific stdout/stderr and allow enough time
for peer backtrace collection; external timeout still bounds stuck peers.

Phases have durable BEGIN/END receipt and stdout markers: local-tile
placement; compile-or-load of the EXACT existing _gathered_program owner;
native local-eigh workspace size query; execution; original service probe
check. Every JAX/collective runs on all 16 ranks. No large matrix or vector
is gathered to a host. Compiler argument/output/temp-alias bytes plus the
existing native vendor-workspace query must fit the explicit device
ceiling, itself bounded by the actual runtime budget. No workspace waiver.

Compile-cache counter deltas are saved for every rank. No cold cache is
forced: backend compilation versus persistent hit is reported only when
the counters support that distinction. Compiled HLO records actual provider
custom-call targets. C-level crashes may bypass Python/finally; the last
durable marker and GDB/launcher verdict then control status, not absence
of a Python failure file. A pass is scoped to this synthetic known-shape
service call, not the lost bank, the whole constructor, SC, Rcut, BSE or TD.

Original failure evidence (external, deliberately not committed):
/home/jackm/mgkba_campaign_20261019/qgrid_20261020/
  sc750_q8_v2_60050061.rank*.log and wrapper.log
  cached_gathered_eigh_ifrt_metadata.json
  jit__gathered_eigh-3c5a2e70eedcbcace01279bb66b5a9bd27be9f4f68773cdbc3ef7157b96e64f9-cache
Recorded binary SHA 9860455fa2b871b29194db0ae785ff91841544b2244d47043f3a7b5359a91a8e.
Cache binaries, native libraries, core files, and large logs stay external.

First diagnostic launch 60051557 used the incorrect --gpus-per-task=1
binding. CUDA plugin initialization failed with CUDA_ERROR_NO_DEVICE on
the 12 nonzero local ranks before the matrix fixture. It did NOT test the
constructor or gathered solver. The original SC launch 60050061 used the
documented --gpus-per-node=4 contract. Corrected diagnostic 60051836 used
that node-wide contract and passed. Pre-selector CUDA_VISIBLE_DEVICES was
0,1,2,3 on all ranks; each node's local ranks selected GPUs 0,1,2,3.

Successful external evidence, under the same campaign root above:
  repro_gathered5504_a0_nodewide_v2_60051836/receipt.json
  repro_gathered5504_a0_nodewide_v2_60051836/compiled_gathered_eigh.hlo.txt
  GATHERED5504_DIAGNOSTIC_RESULT.json
  gathered5504_v2_60051836_evidence.tar.gz
Evidence archive SHA:
  d8133fa42947312ea5e6b1953384a0a9397a9381da742bda1bbd55fddfa0ee0c
The compact result does not replace the external rank logs or accounting.

Read-only failure interpretation: a 120 s faulthandler stack is a sampled
timeout stack, NOT a crash-time native backtrace. The constructor sample
shows first h_vv eigensolve route/price compilation (_stack_bytes), not
proof of gathered fallback execution. A follow-up stack-route reproducer
needs its exact batch/round metadata; do not guess those from parent count.
