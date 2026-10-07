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

The target changes reconstructed wavefunctions and their measured full-WFN
overlap. The fitting stage must form its own full-band Lowdin factor before
cropping public bands. Original PS projection/dual semantics remain fixed;
freshly recomputed C may differ at reduction roundoff and must be reported
as numerical rather than byte equivalence when that occurs.
