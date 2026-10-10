# Atomic field targets and immutable caches

`psp.augmentation_cache` owns the offline normalized-cache constructor,
binding, writer and strict loader. `tools/generate_augmentation_cache.py`
consumes these APIs from an augmentation manifest; `--served-moments`
also prepares the corresponding served overlap/Fourier tables. Prepare
raw-parent C/D afterward with `tools/generate_raw_parent_moments.py`.

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

## Common compact target in scalar GW

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

This request requires `field_policy = unwindowed_U_of_compact_native_pauli`,
the full-WFN overlap policy, and served-monopole enrichment. It authenticates
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

The admitted public compact envelope is fixed one-shot, unsmeared,
`bispinor_gw = coulomb_only`, with `head_correction = off`. A slab uses
`sys_dim = 2` and the already implemented scalar `vcoul.Slab2D` completion.
Static and frequency-dependent scalar screening use the same corrected
samples and Coulomb tensor through their existing downstream contractions.
Current, self-consistent and Gamma-head routes retain explicit refusals.
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
