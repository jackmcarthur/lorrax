# Mesh-padded axes

`runtime.padding` is the only owner of mesh-divisibility arithmetic. Physics
code names an axis and its actual `PartitionSpec`; it does not compute a
modulus, a least common multiple, or a rounded carrier extent.

## Contract

A prefix followed by suffix padding has one
`PaddedAxis(name, logical, carrier, divisor)` receipt. The producer obtains it with `padded_axis` (or the centroid-specific
`padded_mu_axis`) and creates the carrier with `pad_to_axis`, `pad_axis`, or
`pad_square`. Padding is exact zero unless the numerical object requires an
inert diagonal sentinel, such as a BSE energy or an eigensolver matrix.

Consumers do one of three things with the same receipt:

- contract on the carrier, deriving a valid-entry mask with `axis_mask`;
- authenticate an already-carried array with `authenticate_axis` or
  `authenticate_padded_axis`;
- restore the physical shape at a public boundary with `strip_axis`.

Logical extents never come from a carrier shape. In memory they travel beside
plain JAX arrays as `PaddedAxis`; on disk `file_io.tagged_arrays` serializes the
receipt in `restart_padded_axes` and authenticates the current mesh's canonical
carrier on read. SlabIO still accepts logical dataset shapes only; its internal
transport padding is a separate implementation detail.

The canonical carrier is the smallest multiple of the divisor implied by the
spec. If one axis must satisfy several specs, the owner uses their least common
multiple. A product-sharded axis uses the product of the mesh axes in that
single spec entry. This distinction covers the regression where a logical
extent divides each mesh side but does not divide their product.
Controlled invariance tests may request an extra pad from the owner. Consumers
authenticate that explicit producer receipt, including its larger carrier,
instead of reconstructing the minimum from the divisor alone.

For suffix-padded axes, dense solves run on the logical block through
`solve_at_logical`; padding an ill-conditioned operator and solving the larger
system can change conditioning. Orbit-packed centroids are a separate case:
`PackedCentroidBasis` owns their active-slot map and complete solve extent.
A consumer may refuse only when padding is not mathematically inert,
or when a supplied logical extent exceeds its carrier. Error messages name the
axis and both extents; they do not prescribe internal helper calls to users.

## Orbit-packed runtime centroids

GW's `meta.mu_basis` owns the grouped layout and its active-slot map. Physical
centroids can lie after an interior pad, so neither `strip_axis` nor a logical
prefix mask describes that order. `basis.solve_axis` names the full packed
solve extent; `meta.n_rmu` remains the physical count. The C_q pad diagonal
uses the physical mean diagonal so artificial unit modes do not set the
rank-truncation cutoff. This does not make conditioning diagnostics a count
of physical modes; that distinction remains relevant when auditing a factor.

Files store canonical logical rows. Their suffix-padded I/O staging carrier
uses `padded_mu_axis` and may be smaller or larger than the packed carrier.
Only the basis conversion seam moves between them. The loader, q tables and
Dyson geometry consume the packed extent directly; `LORRAX_EXTRA_MU_PAD`
changes canonical staging, not a nonidentity orbit layout. Identity/no-basis
channels retain the suffix contract. Transverse metadata must not inherit a
charge basis belonging to a different centroid table.

## Owned axis families

| Family | Producer boundary | Logical receipt | Consumer boundary |
|---|---|---|---|
| loaded, χ, ζ and masked band windows | `Meta`, WFN loading, ζ fitting, and parent Sigma masks | band slices plus `PaddedAxis` | masks retain the requested interval; projection/output uses the logical slice |
| Σ band window | `sigma_band_axis` before the projected operands enter MPA/GN-PPM | `SigmaOmegaResult.band_axis` → `SigmaResult.sigma_band_axis` | QP, QSGW, SC, `eqp.dat`, and `sigma.h5` strip from that receipt |
| SC/QP protected square matrices | rotation/history producers | one receipt shared by both matrix axes | sentinel/identity pad is removed after eigensolve or at the QP boundary |
| charge and transverse centroids | `PackedCentroidBasis`, canonical loaders and `PhotonBasisLayout.channel_axes` | basis active map / solve receipt, or a suffix `PaddedAxis` | packed solves retain the carrier; suffix solves use the prefix; restart/output stores canonical logical shapes |
| q batches | screening, ζ, BSE and interpolation batch producers | local `PaddedAxis` held for the batch lifetime | the final q result is sliced to the requested q count |
| band and real-space chunks | chunk planner/stream producer | a per-chunk `PaddedAxis` | masks/slices discard only the tail of the final chunk |

## Refusal register

The former Σ refusal requiring a band window to divide both mesh axes is
retired: an 86-band window on a 4×4 mesh carries 88 bands. The former ζ band
chunk and real-space chunk divisibility refusals are likewise producer pads.
The former W/Dyson equal-shape check now authenticates the canonical product
carrier rather than accepting a merely equal, noncanonical pair.

Four refusal classes remain and are not pad opportunities:

- the runtime and BSE ring require a supported square process topology;
- the transverse indefinite LU must solve at the logical extent, so it selects
  a rank-truncated distributed route when that logical extent cannot shard;
- distributed-linear-algebra and symmetry providers authenticate that an
  already-produced carrier satisfies their collective layout; they do not
  choose a new physical carrier extent;
- cyclic ring and `ppermute` modulo expressions select neighbour ranks, not
  carrier extents.

`tests/test_padding_owner_static.py` is the executable register. It rejects a
second round-up spelling, mesh-divisor modulo, or mesh-divisibility refusal
outside the owner unless the exact topology/physics/provider exception above
is named with its reason and follow-up. It scans both `src/` and service source.
The deck doctor uses the driver's `sigma_band_axis` receipt and prints the
logical extent, carrier, divisor, and pad. An indivisible physical band window
is supported by that carrier and is not a preflight refusal.
