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

Separate actual-material replay: 60052556 fault evidence
------------------------------------------------------
debug_sc_first_map_native.py and p16_material_gdb_v2.commands preserve the
reviewed external payload for the actual N750, electronic 8 x 8 / q8
proper-SC replay. This is NOT another synthetic gathered-eigh test and
does not inherit that test's PASS. Its first W constructor and first map
completed, but step 60052556.0 then faulted during map 1. GDB captured
SIGSEGV in CPython dump_frame -> dump_traceback -> _Py_DumpTracebackThreads
-> faulthandler_thread. This identifies the immediate faulting path, not
proof that no earlier memory corruption exists. No converged SC reference
was obtained. Consult this step's own evidence, not the synthetic result.

This replay pins original source 1134957c6e63d3d5c8ac141799e0eb17e9423233
and the actual explicit-count q8-v2 input deck, not a modified solver:
  deck SHA256
    8e6df25ebfce57a3dbbc7055cdd4f8c37def5a5e76a1c8b15d42f457bb1c00bf
  original run_sc_reference_frames.py SHA256
    31f545b6d5a58d7c8c5d053249f8967dc607191cc8d2684a8ff094117b4a3e3b
  actually launched external debug wrapper SHA256
    d734585e3bca8e68ae01cea8a811f1f63a9aebc82e00cfd47ea2f3e54535f0e5
  enhanced GDB commands SHA256
    c7cc6c95c73218242d904a94c00c626c733a7f228b7339a9898b1e79670a58b3
The wrapper also recognizes reviewed source a0 for a separately pinned
replay; that does not change which source this job actually launched.

Actual fault-time GDB evidence is in external material_sc113_60052556.rank8.log
(around line 2780; full downloaded log SHA256
15a7354d05286a470f1dd6fc42ac1f4f1d40eb0641aec6274e6990740aed9295):
  Thread 98 "python" received signal SIGSEGV
  #0 dump_frame
  #1 dump_traceback
  #2 _Py_DumpTracebackThreads
  #3 faulthandler_thread
  #4 pythread_wrapper.llvm
The first W construction had already completed (143.767 s), as had native
map 0. The immediate faulting call was a periodic Python traceback walk,
not the earlier sampled constructor stack. Removing that timer is a scoped
instrumentation correction; it does not prove every native solver/value is
correct, nor establish the cause of the older untraced 60050061 failure.

The explicit-count deck also had a separate scientific selector error:
nval=130/ncond=30 and number_bands_protected=160 resolve to the same M160,
but NOT the same semicore class. band_partition.semicore_floor uses the
lowest explicitly requested valence band in the first form (no coarse
class when all occupied bands are requested), and the gap rule in the
protected form. The explicit run reported zero coarse states and allowed
deep shifts up to 29.113 eV; the intended protected form pins 880 coarse
(k,label) entries. This was not evidence that the band-tail correction
alone caused a 29 eV shift. Do not relabel the two decks identical physics.

Every rank durably appends its own JSONL before/after the original host
pricing/compilation functions. Static metadata includes the exact op,
batch count B, matrix side n, dtype, rounds, phase and gathered flag when
those are known at that call. Hooked functions retain their original
arguments, return objects and internal caches. No new array read, device
operation, collective, solver policy, tolerance or physics is introduced
inside a trace. --log-prefix must stay OUTSIDE the original runner's fresh
--out directory. Log-file fsync and stdout are deliberate instrumentation:
these host walls are NOT a clean production performance benchmark.

The original native runner continues successful maps with unchanged
max_iter=30 and 1 meV protected-band convergence criterion. A first-map
success or saved input frame is not SC convergence. Only the original
native convergence verdict plus accepted-input and completed-driver joins
can establish a usable reference. Native crashes may bypass Python files;
use the exact last durable BEGIN and GDB fault backtrace. Enhanced GDB
captures the stopped fault thread's bt 60, registers and shared libraries
before all-thread bt 20. Repeated faulthandler timeout stacks remain
sampled timeout stacks, not fault-time native backtraces.

Safer repository copies, distinct from actual archived launch hashes
------------------------------------------------------------------
The repository run_sc_reference_frames.py defaults --stack-after to 0/off,
allows nonnegative intervals, and its wall watchdog marks/exits without a
Python frame walk. Fatal handlers and GDB remain unchanged. This is NOT
byte-identical to archived runner SHA 31f545b6...; do not substitute it for
that file while retaining a wrapper's old runner hash authentication.

Both repository debug wrappers intentionally still require that archived
31f545b6... runner. Their repository default --stack-after is now 86400,
beyond the maximum two-hour wrapper lifetime, because the original runner
requires a positive interval. Their new repository hashes therefore differ
from the actual runtime frozen wrappers. Archived/running external files
were NOT modified. Replays must record the supplied CLI interval and hashes.
Current repository helper SHA256:
  8a7c5d8c12e18cd0161b8b4d1675192c8d97ccb52494dc3d366f311777b738bf
Current repository debug_sc_first_map_native.py SHA256:
  3a63a5de92b50b822c2aaf0756d039e7300781021347c49eda360b855dfa0c2c
Current repository debug_sc_protected160_native.py SHA256:
  69e25ff95621a40cf0b5d59be097d4d8346ba4e3401f3f13161fa9d0324aa395
debug_sc_protected160_native.py additionally authenticates the corrected
protected160 q8 deck SHA 534a2c249e95ca2e8b4d35ba036fc8ed17bb32c01ddbfa20f86cfcacce197eb5;
the original explicit-nval deck erases the semicore class despite the same
resolved M160. It must not be called the same SC prescription.

Corrected protected-form run: ONGOING 60053348, not converged
---------------------------------------------------------
The new run uses source 1134957, corrected q8 deck SHA 534a2c24... and the
original authenticated 898-map-output warm guess, NOT the wrong-policy
60052556 map output. Its external wrapper hash is 28082c4e17215de683bb3ef994e4d6ed02349405455f0434a7eeb79e523c3e3e.
The launched CLI supplies --stack-after 86400, beyond the job duration;
all-rank GDB and the external wall guard remain. Actual startup confirmed
880 coarse entries. Corrected map 0 still had raw output gap -0.153337 eV,
but its actual next INPUT remained gapped at 2.484721 eV after native half
mixing; do not treat the map output as the advanced/converged state.

Progress snapshot through map 3: output gap 4.211208 eV, maximum protected
displacement 0.838951 eV, RMS 0.273604 eV. Maximum displacement is not
monotone (map 2: 0.820516 eV); all remain far above the unchanged 0.001 eV
criterion. Native Anderson SC continues without a claimed convergence or
cutoff-accuracy result. Only later authenticated completion can supersede
this explicitly partial snapshot.

Corrected inputs are preserved in ../inputs/ and header-only provenance
guard c697f567... in ../evidence/. The guard validates resolved counts and
dipole/head provenance; it is NOT SC or GPU/provider verification. The q4
deck 1712d907... is PREPARED_PENDING_CONVERGED_Q8_NOT_RUN. Its separate
seed_from_accepted_q8 path intentionally remains absent until the q8
terminal artifact joins a genuine converged accepted input and completed
driver. The companion prepared JSON lists the required checks. Never stage
an intermediate sc_seed/map output as that accepted reference.

No allocation launcher or hardcoded remote submission script is preserved
here. This directory requests no additional allocation or GPU stage.

Common coherent chi owner: NATIVE P4 SOFTWARE PROOF PASS
-----------------------------------------------------
verify_common_coherent_chi_p4.py requires an explicitly pinned source
revision (also LEG_HEAD), source-root module origins, native CUDA mathdx,
four processes/one device each and mesh2x2. Maximum runtime180s (0.2GPUh).
It forces the full current typed identity parent plan and the existing
compiled/native admission owner: full+FF/FA/AF/AA, fine8/coarse4 Fejer,
two E/rho inputs, both complex-time orientations, exact transition moments,
inactive deep-band overflow and default-vs-explicit-false bits. No reference
backend override, material/SC admission, new native build or TD trajectory.
Run with --source-root ROOT --source-revision SHA --out FRESH_DIRECTORY.
Both this file and the fixture owner verify_fejer_p4_38dbea1.py must remain
in the immutable source archive. Only agreed normal completion writes PASS;
rank-local exceptions print/rethrow without a new failure-path collective.
Host preparation tests are test_verify_common_chi_p4_prepared.py (3PASS),
including the actual typed execution config parsed by its native factory.
The first5049 P4 attempt stopped before chi/JIT at an invalid empty config
constructor; it provides no chi correctness verdict. The corrected helper
uses LorraxConfig.from_input_file(resolve_hardware=False), not a fabricated
full config.

The corrected source dbf45a48a577d5b0c4b649e4ea76af24462c2e9f passed
native P4 step 60056663.13: 39 s step, 29.654809 s receipt. All 20 paired calls,
full+FF/FA/AF/AA, two E/rho inputs, fine/coarse grids, exact moments,
overflow and compiled-execution cases passed. Maximum literal error was
5.551792708129603e-16; moment error 1.2212801772525548e-15.
Actual receipt:
/home/jackm/mgkba_campaign_20261019/qgrid_20261020/p4_commonchiDBF_60056663/receipt.json
The durable summary is ../evidence/COMMON_CHI_P4_PASS_60056663.json.
This qualifies the tiny native software case, not material/general-W,
hsuite, speedup, positivity, equilibrium stationarity or a TD trajectory.
Earlier CPU evidence and failed 5049 startup records retain their original
per-run identities; the successful run does not relabel either of them.

Proposed equilibrium Hermitian value pencil: HOST ONLY
-----------------------------------------------------
verify_equilibrium_hermitian_pencil_host.py exercises the ACTUAL small
finite_pencil_column owner, then a proof-only NumPy/SciPy positive-H
congruence. Complex gyrotropic q/-q channels have nonzero C1; shifted gamma,
all complex held values, both residue signs and non-dark zero-mu refusal are
checked. It is not a production reducer, native GPU proof, material fit or
speedup. The proposal and material sample-error/null-space caveats are in
docs/theory/shared-pole-w-model.md section6.4.

Use CPU only, with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu:
  python verify_equilibrium_hermitian_pencil_host.py --source ROOT --out FRESH.json
The output is exclusive. The focused host test is
../test_equilibrium_hermitian_pencil_host.py. The actual V2 receipt is
/home/jackm/mgkba_campaign_20261019/qgrid_20261020/PROPOSED_EQ_HERMITIAN_PENCIL_HOST_V2.json
(held complex error2.42e-14, q/-q error5.97e-15; .589s).
No production route or material source was changed by this proof.

Complex denominator-box chi quadrature: TINY NATIVE P4 PASS
---------------------------------------------------------
verify_complex_box_chi_p4.py tests the explicit value-only public minimax
rules through the typed common native mode11 stream and its actual compiled
admission. Two current E/rho inputs, full+FF/FA/AF/AA, fine8 and coarse4
Fejer outputs are compared with independent full-port transition resolvents.
The public scalar certificate and mass gates remain enabled. No supports,
virtual bands, density coherences or physical tolerances are removed.

Run on four MPI ranks / one GPU per rank / native 2x2 mesh with the usual
nodewide GPU binding and immutable source archive:
  LEG_HEAD=SHA python verify_complex_box_chi_p4.py --source-root ROOT \
    --source-revision SHA --out FRESH_DIRECTORY --max-wall-seconds 180
Native CUDA/MathDx, source origins, exact revision and FP64 are required;
the helper does not enable a reference backend. Only agreed completion
writes PASS. Rank-local failures print/rethrow without another collective.
The host preparation test in tests/test_full_density_complex_box_cpu.py
checks its independent frequency oracle and REAL typed config construction.
Actual source7aa8f92 native P4 step60065589.0 passed all20paired cases:
22.8645s fixture / 54s Slurm step, maximum full-port literal error2.14446e-9,
maximum public scalar certificate9.76432e-9 < 1e-8. The receipt is
/home/jackm/mgkba_campaign_20261019/qgrid_20261020/p4_complex7aa_60065589/receipt.json
(SHA25652427ddbf6f80f89caa74fc37ac0aff045574a6e79660d2306a5e1ca2a4e98fe).
This tiny CUDA pass is not a material timing, positivity or below60s-cycle claim.
