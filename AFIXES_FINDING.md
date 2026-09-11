# AFIXES — provisional finding, 2026-09-11

Heavy lane. **Prepared repairs; no merge clearance.** Base `810c260b`, branch
`lane/sp-afixes-2026-09-11`. The assigned SP-M2 allocation `58190627` remains
PENDING. No AFIXES job.step or numerical measurement exists yet. Syntax parsing
and `git diff --check` pass; these are not substitutes for the requested gates.

Evidence directory:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/364_afixes_20260911/`.
The detached supervisor's `supervisor.json` owns current execution state.

## Claim dispositions

| Item | Source finding and disposition | Verification still owed |
|---|---|---|
| 1a | **Fixed in candidate.** The restart metadata read preceded collective payload authentication with no error agreement. Agree serial-read errors first, then agree the membership hash (including absence), so only unanimous absence is rebuildable. Preserve an existing `SharedPoleMemberRefused` without adding its suffix again. The model validator had a second serial-HDF5/collective boundary; it now agrees metadata failures before digest entry too. | Real P4 rank-local OSError red twin; missing/refused/roundtrip contracts; all-parent production gate. |
| 1b | **Fixed in candidate.** `b` was assigned; `local` was initialized; only `shard` was potentially unbound. Removing the unsafe deletion and releasing local references by assignment handles an empty shard list. | Empty-addressable-shards digest test; real model authentication. |
| 2a | **Fixed in candidate.** Reconstruct and authenticate the stored eps/relative and all other certificate fields before compatibility filtering. A one-ULP request can reuse the original immutable stored certificate. | Stored-eps and relative corruption red twins; adjacent-ULP request hit; certifying production rules. |
| 2b | **Fixed visibility/invalidation; pruning deliberately deferred.** List the bounded directory's old `rule_*.npz` names without loading them and issue a schema-migration warning with directory, count, example, rebuild consequence and retention/removal guidance. Old files remain invalid. Automatic removal/movement would mutate potentially completed evidence runs, contrary to this brief's immutability rule; no new prune dial was introduced. | Legacy filename warning/no-open test and ordinary operator output. |
| 3 | **Measurement prepared; no production source change.** Source confirms the denser acceptance cloud. Run364/02_cloud_density uses one recorded Na B04 crossing box, NumPy, one BLAS thread, 120 seconds on both arms. It instruments completed removal-pass counts in the existing owner and reports both own-cloud and common-dense-cloud sup errors, retained nodes and planning seconds. | Execute both arms; no measured ratio, slowdown, extra-node count or budget conclusion yet. |
| 4 | **Fixed in candidate, conditional on exactness gate.** The only production `finite_pencil_column` call passes identical Q/output objects. Reuse `adjoint(O†Q)` there; preserve the general second GEMM when either panel differs. Removing a product is algebraically sound but does not establish bit-exact fitted models or a timing win. | Distinct-panel/aliased-panel contracts; exact Si/Na parents; band control, repeat and candidate timings. |
| 5a | **Fixed for the production local batch-reshard route; distributed optimization deliberately deferred.** Validate after the eigensolver's existing face-to-batch exchanges, inside the same executable. A local reduction checks the original tolerance; invalid input skips eigh and returns a NaN spectrum so the existing replicated-spectrum readback refuses on every rank. No trust-me flag or Hermitian projection. Arbitrary distributed off-diagonal tiles still require peer information, so that route keeps its safety check. | Optimized HLO on every P4 rank must show no additional all-to-all relative to the ordinary eigensolver; inspect reduction temporaries; invalid Hermitian/NaN twins; production peaks. No HLO result claimed. |
| 5b | **Fixed in candidate.** GEMM workspace resolves through the same matmul provider/route as execution, independent of the eigh plan. The service retains one vendor-query implementation. Constructor admission separately prices distributed operand-transpose staging; the query is explicitly workspace-only. | Wrong-handle/provider-route contract; real-P4 workspace and production peak gates. |
| 6a | **Fixed in candidate.** Split the existing bank writer's preparation, admitted payload write and mask/finalization owners. Ordinary callers retain pre-open validation and resumable commits. Export authenticates the source once, initializes its destination once, keeps one append handle through the copy, drains each field, then publishes masks and finalizes after close. No second bank implementation. | Exhaustive planted W/dW/M1/M3 byte parity and destination-open-count contract; production export band/bytes. No export speedup measured. |
| 6b | **Fixed in candidate.** Reject empty `wfn_file` at deck parsing for requested exports; require the loader's public `path` before screening starts. Both fresh and restart exports use that path. The private `_filename` fallback is gone. | Early-refusal test, config contracts and production outputs. |
| 6c | **Fixed in candidate.** The `_NULLABLE_INT` branch now accepts `none` uniformly for its members. No budget defaults, currencies or stopping rules changed. | Parameterized nullable-member parse test. |
| 6d | **Fixed in candidate.** Common solver messages name `solve_smearing_occupations`; invalid width names kBT for FD and BerkeleyGW half-width for MP1, in Ry. | Both-family diagnostic test. |

## Gate plan and scope

`01_contracts_p4/payload.sh` combines focused contracts, planted export/bank
checks, rank-skew fault injection, every-rank optimized HLO and the scalar cloud
experiment in one P4 dispatch. Production arms in `03_si_p4` and `04_na_p16`
reuse Run350's full-driver ANEST wrapper, Run337's measurement owner and
Run341's exact-input comparator. Each system has a fresh control, identical
control repeat, candidate off and outputs-on (W plus poles) arm. Step limits use three times
the matched historical launcher duration recorded in `arms.json`.

All Si/Na parent models must be exact; every rank's peak must not increase;
every selected rule must certify; Sigma max change must be <=2 meV.
Bit-identical Sigma is **not** required. Published band spreads remain the
reference; wall differences alone will not be called a speedup. Existing CD
receipts may be reused only after the exact-input owner passes. No Run258/CD
accuracy, storage, J/K/r, passivity value, model condition, or timing number is
claimed for this unexecuted candidate.

`src/gw/mpa/sigma.py` and production
`services/minimax/src/minimax/uniform_rule.py` are untouched. No main push,
history rewrite, pool creation, completed-run mutation, or login-node HDF5/test
execution occurred. Constructor merge overlap with ACONJ is restricted to the
finite-product shortcut and workspace/admission helper; direction selection
and its partner/covariance changes remain owned by ACONJ.

## Open items

Compute is the blocking dependency. The prepared supervisor pins SP-M2 and
stops on missing/failed artifacts, independent of launcher rc. Review its first
contract results before interpreting any later production data. Full export payload postprocessing uses the existing bounded Run350 comparator
after the final collective process in each outputs-on step; no later collective
startup follows the rank-zero comparison.

The deployed `lx status` rejects `--jid`; the runner uses its inherited job
selector and verifies the printed JID against exact `scontrol` state. Every
`lx run` still has explicit `--jid 58190627`. If the inherited selector does not
pin correctly, the supervisor refuses instead of taking another pool.
