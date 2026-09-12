# Shared-pole notation and the G/W contraction seam

The physical shared-pole factor is **lowercase `b`**:
`Wc(s) = b (s - Lambda)^-1 b†`, with `s = z_Ry²`, `Lambda = Omega_Ry²`,
and `b` in Ry^(3/2). The squared-frequency residue is `b_j b_j†`.
The time-domain weight `d = exp[-i(Omega-E_ref) tau]/(2 Omega)` remains
separate. Renaming the factor does not absorb this normalization into it.
These are the `b`, `b_X`, and `b_Y` names to use alongside the shared-pole
DESIGN references in the constructor and Sigma source.

Stored v1 models remain compatible: the dataset is still `factor`, the
normalization metadata retains its legacy `C` spelling, and the hashed
`finite_factors_poles` predicate is unchanged. That legacy `C` denotes `b`.
Changing even the predicate's prose would change `GATE_HASH` and the model's
current-map identity. Existing completed models and receipts are not rewritten.
BLAS `C` accumulators, adjoint operation codes `"C"`, and stage labels A/B/C
are unrelated and retain their names.

G and W share the weighted outer-product mathematics, and W now calls G's
implementation: `gw/mpa/sigma.py::_shared_pole_contract` routes through
`build_G(..., layout='face')` in `gw/greens_function_kernel.py`.
`distrib_la.contract_faces` still exists as a service entry point but has no
caller left in `src/`. They feed the same Sigma spatial executor.

| Aspect | G | Shared-pole W | Reason for difference |
|---|---|---|---|
| Scalar weight | Energy phase, optional energy window, band mask, linear f or 1-f | Positive sqrt(Lambda), active sorted-pole intervals, causal d | Physics and support conventions; keep separate |
| Interval representation | Energy interval (lo,hi], or band identity mask | Half-open sorted column interval [start,stop) | Different coordinates, not opposite physical boundary conventions |
| Spin | Full spin-off-diagonal output; rectangular endpoints allowed | Scalar-only gate | Current W representation restriction; G must retain its generality |
| Legacy contraction | Three-operand weighted einsum, plus identity/dense Gij branches | Explicit weighted local GEMM, optional same-time transpose partner | Implementation divergence; changing G's contraction order needs performance evidence |
| Low-memory contraction | Planned distributed GEMM on two XY-sharded psi orientations | Same one-axis row faces and local GEMM as ordinary W | Accidental storage divergence; not caused by occupations |
| Symmetry | Optional canonical parent-k operator unfold | Local parent-q operator action, or bounded factor routing for nonlocal maps; fixed-q projection | Different transport locus, shared symmetry service; W must preserve d on antiunitary children |
| Lifetime | Caller resolves face GEMM once | W stage owns bounded parent/column reads and compiled panels | Storage scheduling; callable-lifetime work belongs to ASERV |

`low_mem_bands` selects the face wavefunction carrier and refuses an explicit
`Gij`; it does not switch shared-pole factor storage at the Sigma reader.
The constructor/writer hands off `b[parent,mu,spin,K]` at
`P(None,'x',None,'y')`, and `read_shared_pole_faces` returns two arrays at
`P(None,'x',None,'y')` and `P(None,'y',None,'x')`
(`file_io/shared_pole_store.py`): both axes are tiled, mu against K, in the
two orientations the face contraction consumes. The one-axis faces
`P(None,'x',None,None)` / `P(None,'y',None,None)` this page used to describe
are not what the shipped reader returns. Poles and interval metadata are
replicated inside each bounded panel. W itself is `P(None,'x','y')`; that
output fact does not certify its input storage.
In contrast, persistent psi uses `(parent,spin,mu,band)` at
`P(None,None,'x','y')` and `(parent,band,spin,mu)` at `P(None,'x',None,'y')`.

AWB leaves every extent, layout, arithmetic operation and G source unchanged.
Full convergence is stopped at the owner's explicit no-sharding-change boundary.
The concrete next change would read two pole-sharded b orientations matching
psi, carry those orientations through canonical endpoint transport, and call
G's existing face contraction with the separately computed d. It must retain
W's paired same-time transpose and current capacity admission. That changes
input sharding and replaces the collective-free W contraction with a distributed
one; it requires an explicit layout decision before implementation. G's existing
phase, occupation, window, dense-Gij, spin, parent unfolding and planning
behavior must all remain intact.
