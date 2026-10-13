# Atomic field targets and immutable caches

`psp.augmentation_cache` owns the offline normalized-cache constructor,
binding, writer and strict loader. `tools/generate_augmentation_cache.py`
consumes these APIs from an augmentation manifest; `--served-moments`
also prepares the corresponding served overlap/Fourier tables. Prepare
raw-parent C/D afterward with `tools/generate_raw_parent_moments.py`.

The internal reconstructed current family may declare
`current_angular = {"lmax": 7}` in its augmentation manifest. This changes
only its retained harmonics and storage; the charge `angular` controls and
the shared endpoint directions, weights, radial fields and sampling scan
remain unchanged. The shared Lebedev order must cover twice the current
extent. A current preparation authenticates the sufficient atomic-channel LS/SL bound
`max(ell_large) + max(ell_small)`, with
`ell_small = ell_large - sign(kappa)`, from every represented native
channel. An insufficient inherited or explicit extent refuses. In
particular the represented Cr `ell=3, kappa=-4` channel requires `L=7`.
Smooth plane-wave times local-correction terms have unbounded angular
expansions; this channel-support requirement is not an angular or energy
convergence certificate for the total current. Native kappa must be
`ell` or `-ell-1` with the usual positive-ell restriction for the first case.
The contract and its identity survive the OWN3 fit-to-provider handoff.
This declaration does not admit compact transverse GW, slab completion,
screened currents or augmented vector Hartree; their existing guards remain.

Without `cache.target`, the target is `native_large_as_pauli`: the matched
ONCV large-component correction is treated as a Pauli precursor and lifted
with the normalized RKB graph. The existing numerical branch is unchanged.

The alternative `cache.target.kind = ae_large_preserved_free_graph` selects
an explicit offline AE-large-preserving constructor. The historical name
denotes the normalized graph with X = sigma.p/(2c), not the exact spectral
free-Dirac positive-energy graph. Its Pauli correction is R^-1 times the
windowed AE-large-minus-lifted-PS-large correction. Canonical U is applied
once. The served small component derives from the same compact large
Hermite field, including the derivative of the final taper. Native Dirac Q
is retained as a diagnostic; it is not inserted as that small component.

`cache.target` contains exactly `kind` and `species`. The `species` map is
keyed by each authenticated source UPF SHA256. Each entry contains exactly:

* `nuclear_charge` (positive integer);
* `matched_dirac_file`, `matched_dirac_sha256`;
* `pseudo_exterior_file`, `pseudo_exterior_sha256`;
* `source_upf_file`, `source_upf_sha256`;
* `dirac_window_start`, `dirac_window_stop`;
* `completion_start`, `completion_stop`.

These preparation-input paths must be absolute. Their hashes and metadata
enter the binding. The matched P/Q sidecar must preserve every original
native array and PCA coefficient exactly. The exterior PS must preserve
the same interior amplitudes, physical derivatives, angular/channel order,
training energies and source-bank identity. The nuclear charge must equal
the UPF charge. The Dirac window starts outside both declared and actual
nonzero projector support and ends at the native radius. The virtual PS
completion lies outside the native radius; it is not a larger physical
augmentation sphere. The final compact support/taper remain explicit cache
controls. No per-row normalization or new projector rank is introduced.

Runtime uses the same strict `load_normalized_cache` and common compact
evaluator for both targets. An explicit cache never triggers a hidden
build. The target, primitive inputs, controls, payload and entire owner
files are authenticated. Editing an owner creates a new provenance epoch:
regenerate normalized, served and raw-parent artifacts in new directories.
Preserve old artifacts with their immutable source snapshots. Numerical
equivalence to an older target is a separate measured proof, not an alias
of its cache identity or fitting-accuracy certificate.

## Common compact target in charge and current GW

The public manifest may contain one `compact_target_request` for the existing
full-FILE common-frame and paired-field loader. The request contains exactly
`frame_file`, `frame_sha256`, `wfn_sha256`, `carrier`, and `species_fields`.
Each species entry contains exactly `file`, `file_sha256`, and
`common_spectrum_sha256`; species keys are atomic numbers as strings.
Paths resolve relative to the manifest directory, or may be absolute.
The manifest and request must name the same carrier: `normalized_rkb` or
`pauli2embed4`. The latter loads the two physical Pauli components once and
embeds them as `[psi_Pauli; 0]` in the existing four-slot fitting source.
It uses the same common A and paired atomic spectrum, without a second
normalization or kinetic-balance lift. Its distinct charge basis stamp is
preserved through fitting and restart; it cannot serve spatial currents or
scalar head corrections.

This request requires a declared paired-field policy, the full-WFN overlap
policy, and served-monopole enrichment.
`unwindowed_U_of_compact_native_pauli` serves the original common Pauli
target; `paired_ae_large_preserved_free_graph` serves the prescribed
AE-large kinetic-balance target. Spatial currents require the latter
policy and `carrier = normalized_rkb`; the two-component embedded carrier
remains charge-only. It authenticates
the existing complete physical C/D/B/G0/A target and the paired atomic fields.
The public loader streams the currently loaded WFN file on rank zero and
broadcasts its whole-file SHA256 through the existing collective control
owner. That digest must match the declared target source; paths may differ
for byte-identical copies. The driver also joins its existing opaque WFN
fingerprint to the occupied source. Neither fitting nor
serving recalculates a second factor or normalizes the fields after U.
The source digest is checked again after coefficient preparation and before
the first fit publication; ordinary Sigma contractions do not rehash files.

Fresh preparation, resident Hartree and restart authentication consume the
same bound artifact identity. Both explicit radial arrays and log-Simpson
controls decode through the fitting stage's radial owner. The resident
Hartree record retains only its band matrix and small source/geometry
bindings; complete C/D/B/A arrays do not survive in its serving context.
A receiving matrix for an edge-only diagnostic cannot cover a wider GW
band window, even when its transport carrier is padded to that window.

The scalar public compact envelope is fixed one-shot, unsmeared,
`bispinor_gw = coulomb_only`, with `head_correction = off`. A slab uses
`sys_dim = 2` and the already implemented scalar `vcoul.Slab2D` completion.
Static and frequency-dependent scalar screening use the same corrected
samples and Coulomb tensor through their existing downstream contractions.
For spatial currents, the authenticated physical mixed slab representation
also admits `bare_transverse` and `full_shared_pole` under `sys_dim = 2`,
with the same fixed-source, unsmeared, head-off one-shot constraints.
`full_shared_pole` additionally requires `compute_mode = mpa`,
`sigma_w_model = shared_pole` and `screening_diagrams = w_rpa`.
The completed charge/current tensors and sample products enter the existing
screening and self-energy contractions. Self-consistent and reconstructed
Gamma-head routes remain outside this compact-field admission; see
[reconstructed currents](../theory/augmented-isdf.md#reconstructed-current-and-its-bare-interaction).
This API admission supplies no numerical convergence certificate; actual
source/action, Coulomb-fitting and Sigma convergence remain separate tests.

The target changes reconstructed wavefunctions and their measured full-WFN
overlap. The fitting stage must form its own full-band Lowdin factor before
cropping public bands. Original PS projection/dual semantics remain fixed;
freshly recomputed C may differ at reduction roundoff and must be reported
as numerical rather than byte equivalence when that occurs.

## Prepared periodic compensation metrics

`isdf.coulomb_fourier_cache.write_periodic_compensation_cache` and
`load_periodic_compensation_cache` serve an explicitly prepared geometry
metric through collective SlabIO/HDF. They do not build a Fourier metric
or enable a different fitting policy. The existing local spline-cache
APIs and numerical formulas are unchanged.

The geometry bundle binds reciprocal rows in inverse bohr, cell volume in
bohr cubed, ordered atom centres in bohr, the exact ordered fractional q
rows, and support radius. The moment axes are atom-major complete complex
harmonics in increasing `(l,m)` order. A different order refuses; the loader
does not permute a global tensor or infer a q symmetry action. The closed
model is the power6 compact compensation profile, bare periodic
`8*pi/(Omega*|K|^2)` Ry kernel, excluded Gamma zero mode, and
`exp(-i*K.center)` phase. Its metric has physical Ry units for unit harmonic
multipoles and remains `P(None,'x','y')` at runtime.

The bulk `lorrax.periodic_compensation_metric.v1` schema and its geometry
remain unchanged. The separate slab schema,
`lorrax.periodic_compensation_metric.slab.v1`, binds the public `vcoul.Slab2D`
kernel, Cartesian `+z` normal, truncation half-height and excluded total
Gamma zero mode. Obtain its geometry with
`periodic_compensation_geometry(geometry, sys_dim=2)` before preparation.
The public kernel validates the reciprocal orientation; the cache owner
requires in-plane q rows and a compact atomic layer whose full pair extent
is strictly below the half-height. A bulk cache cannot be relabelled as a
slab cache. Both retain the same collective dataset/layout and exact
power6 compensation profiles.

Preparation supplies a receipt path/hash, payload hash, producer-source
hashes, finite cutoffs and measured per-q refinement. The writer checks the
receipt file; the persisted evidence is self-contained and survives cache
relocation or removal of that preparation path. The preparation producer
owns the receipt-to-input numerical equality. Runtime authenticates the
complete persisted payload through an externally pinned whole-file hash
and collective commit, together with the consuming geometry/model/order.
Fixed scalar-byte metadata is bounded before allocation. Finite-cutoff
refinement is evidence at those cutoffs, not an infinite-tail bound.

`isdf.atomic_coulomb.make_periodic_compensation_action` accepts this cache,
the existing `PackedCentroidBasis`, and FFT point count. Its callable takes
moment rows and the Gram as explicit operands. Both are distributed on the
two-dimensional face; the coefficient rows retain their actual packed
centroid order. The basis active mask removes interleaved ghosts before
both products and on both output axes. Two public N,N GEMMs with
`common.collectives.transpose_xy` evaluate `conj(M) G transpose(M)` and
apply `(Nfft/Omega)^2` exactly once. This helper adds no local self term,
periodic mean or head; those remain separate physical owners.

The focused P4 test covers complex off-diagonal contractions, physical
centroid6/carrier8 and moment9/carrier10, canonical and interleaved ghosts,
NaN/large-value poison, relocation, and stale geometry/payload/commit
refusals. Evidence is in
`runs/DEV/780_augmented_isdf_20261006/positive_periodic_cache_action_v3` of
the validation sandbox (claim3837). That planted IO/action proof does not
admit an actual Fourier preparation, AgI fitting attachment or a physical
screened-QP interpretation; actual geometry-cache and full action parity
must be measured separately.

The public local-high fitting policy binds this artifact through
`charge_metric={body_metric: physical_low_local_high, moment_enrichment:
served_monopole, periodic_compensation_cache: {file: PATH, file_sha256: SHA}}`.
The file path is relative to the atomic manifest directory or absolute;
the consuming cell, ordered atoms, actual q rows, support and canonical lm
must match its metadata exactly. The positive correction metric supports
bulk3D or the explicitly bound compact-layer slab2D kernel. The separate
slab Hartree owner evaluates both smooth-neutral cross terms with
the public truncated kernel on the full FFT grid and adds no bulk Gamma
neutral-mean correction. Its operator and kernel identities are separate
from this geometry cache. Public augmented slab preparation requires the
complete common compact target binding above, together with the actual
live source/action checks. CPU mathematical controls alone do not certify
the physical reconstruction or its convergence. Legacy metric manifests
retain their existing policy.

Before allocating the global metric, the positive provider plans its same
two public GEMMs from a shape descriptor and prices the distributed Gram,
moment rows, intermediate/output and queried native workspace. The retained
action then consumes the authenticated loaded cache. A shape descriptor is
memory planning, not physical payload admission. The focused P4 oracle
checks that planned and loaded actions are bit identical, including an
interleaved packed centroid mask.

## Physical Slab three-current completion

The current source is the same authenticated full-WFN reconstruction and
band rotation as charge. Its three normal equations use the actual current
bilinears and their own factors, with unit endpoint loss; scalar charge
conditioning and occupied-band loss weights do not alter these factors.
The scalar Coulomb-only flow does not prepare current samples or local banks.

The physical Slab route automatically checks the complete smooth-current
reciprocal support against the actual q-aware body cutoff and FFT extent.
It also certifies every atom pair and periodic half-cell shifted plane,
using the actual centers, per-atom support and public Slab orientation.
For neutral two-moment `n = delta - g`, these layer conditions identify the
infinite local Slab remainder with the existing bulk local remainder.
The passed current tables must equal the public **point Slab** kernel on
all addressed slots, including finite-q heads, masks and Gamma. Averaged
heads are refused by this specialization; a head flag alone is not proof.
Provider disjoint-sphere and measured `delta(0) = g(0)` guards still apply.

The reciprocal mixed body uses the physical current `Z + delta`. Its local
bulk addition is `delta† B delta - g† B g` over the infinite local reference.
There are no additional NN/GN/NG packets in this physical specialization.
The global compensation energy above the reciprocal body cutoff remains
an explicit tail requiring an occupied-block stopping bound. The layer
certificate alone does not certify that tail, current angular convergence,
fitting error, the continuum Slab limit, or a meV-accurate self-energy.
Physical current zeta output remains `Z + delta`, and downstream GW uses
the existing current samples and Coulomb tensor interfaces.

## Charge-first current coefficient storage

Atomic current normal sources are deferred until the charge RHS is consumed
and the actual charge zeta closes. The protected corrected pseudo source and
rotated atomic coefficients retain the same physical source and frame.
The ordinary current fits create their canonical OWN3 factors first; each
subsequent radial packet uses those exact leaves and donates its result into
three final delta banks. No full raw current cloud or paired concatenation
coexists with the final banks. The separate six-bank delta/pseudo route
remains available for the established low-level comparison and bulk policy.

`CurrentCoefficientBanks` authenticates the actual fit objects, factor
leaves, source/angular identities, ordered q rows and centroid basis.
It requires complete nonoverlapping radial coverage and explicitly releases
its arrays when the current fit closes. The point/layer/kernel preflight runs
before reconstruction or fitting. The internal planner reserves actual
protected state during the PW fit and final banks during contraction, then
chooses a bounded G tile before creating the store. No input overrides it.
Sampler, solve/store and actual moment/Fourier/onsite/physical-writer AOT
checks precede final-bank allocation; the store must alias its complete
bank and accept only the quoted packet width. A compiler/lifetime estimate
is separate from a measured production peak.

The generated Coulomb tensor receipt stores the live current operator
contract and its canonical digest, binding the source WFN, full physical
frame, current point chart, q/body/basis and angular policy. This identity
contains no Hartree occupation or GW approximation. The paired Hartree
consumer authenticates its own physical occupied source against it.
Scalar charge mean conventions and default bulk kernel policies are unchanged.
