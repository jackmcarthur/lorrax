# Mesh-padded axes

`runtime.padding` is the only owner of mesh-divisibility arithmetic. Physics
code names an axis and its actual `PartitionSpec`; it never computes a
modulus, a least common multiple or a rounded carrier extent.

## Contract

A logical axis followed by suffix padding has one receipt,
`PaddedAxis(name, logical, carrier, divisor)`. The divisor is the product of
the mesh axes that the spec assigns to that array axis (`spec_divisor`). When
one axis must satisfy several specs, the divisor is their least common
multiple (`padded_axis(..., specs=...)`). A product-sharded entry such as
`('x','y')` divides by the product, which is not the same as dividing each
side. The carrier is the smallest multiple of the divisor. A controlled
invariance test may ask for an extra pad, and consumers then authenticate
the producer's receipt rather than rebuilding the minimum carrier.

The producer obtains the receipt (`padded_axis`, or `padded_mu_axis` for
centroids) and builds the carrier (`pad_to_axis`, `pad_axis`, `pad_square`).
Pads are exact zeros, which is inert for any operator linear or bilinear in
the axis. A diagonalization needs a sentinel instead: the BSE ε pads carry
±`PAD_EPS_GUARD_RY`, and `pad_square(pad_diagonal=...)` embeds an identity
or eigensolver sentinel. A consumer does one of three things with the same
receipt:

- contracts on the carrier with the mask from `axis_mask`;
- authenticates a carried array (`authenticate_axis`,
  `authenticate_padded_axis`);
- strips back to the logical shape at a public boundary (`strip_axis`).

A logical extent never comes from a carrier shape. In memory the receipt
travels beside the plain JAX array. On disk, `file_io.tagged_arrays`
serializes it in the `restart_padded_axes` attribute and authenticates the
current mesh's carrier on read. SlabIO datasets keep logical shapes: a write
drops pad rows past the dataset extent, and a read without a shape returns
the mesh-padded carrier, zero-filled ([SlabIO](slab_io.md)).

Dense solves on a suffix-padded axis run on the logical block
(`solve_at_logical`), because padding an ill-conditioned operator and solving
the larger system changes the round-off with the pad extent. Error messages
name the axis and both extents.

## Orbit-packed runtime centroids

GW's `meta.mu_basis` (`common.centroid_basis.PackedCentroidBasis`) owns the
in-memory centroid layout. Each shard holds whole symmetry orbits followed by
its own zero pads, so the pads are interior rather than a global suffix, and
neither `strip_axis` nor a prefix mask describes the order. `basis.solve_axis`
names the full packed solve extent (`meta.mu_solve_extent`), while
`meta.n_rmu` stays the physical count. The ζ fit makes this carrier
nonsingular with C_q's mean diagonal on the pad slots
([normal equations](zeta_fit_face_psi_cct.md#normal-equations)), and
consumers mask the pads with the basis's active-slot map.

Files store canonical logical rows. Their suffix-padded staging carrier
(`padded_mu_axis`) may be smaller or larger than the packed carrier, and only
the basis's pack and unpack seam converts between them.
`LORRAX_EXTRA_MU_PAD`, which is test-only, enlarges the canonical staging
carrier and never the packed layout. Channels without a basis keep the suffix
contract. Transverse metadata never inherits a charge basis built for a
different centroid table.

## Owned axis families

| family | producer | receipt | consumer boundary |
|---|---|---|---|
| loaded, χ, ζ and masked band windows | `Meta`, the WFN loaders, the ζ fit, parent Σ masks | band slices plus `PaddedAxis` | masks keep the requested interval; projection and output use the logical slice |
| Σ band window | `gw.ppm_sigma.sigma_band_axis`, before the projected operands enter MPA or GN-PPM | `SigmaOmegaResult.band_axis` → `SigmaResult.sigma_band_axis` | QP, QSGW, SC, `eqp.dat` and `sigma.h5` strip from it |
| SC and QP protected square matrices | the rotation and history producers | one receipt for both matrix axes | the sentinel or identity pad is removed after the eigensolve or at the QP boundary |
| charge and current centroids | `PackedCentroidBasis`, the canonical loaders, `PhotonBasisLayout.channel_axes` | the basis active map and solve receipt, or a suffix `PaddedAxis` | packed solves keep the carrier, suffix solves use the prefix, files store canonical logical shapes |
| q batches | the screening, ζ, BSE and interpolation batch producers | a local `PaddedAxis` for the batch lifetime | the final q result is sliced to the requested count |
| route-G axes | `_fit_mubatch`: stored q rows (divisor P), ζ-sphere G tiles (divisor G_tile), ψ G slots, plane groups | `PaddedAxis`; batch pad slots are −1 in `OwnerOrbitBatches.mu` | pad q rows and G slots carry v = 0 and ngk = 0 in V_q ([route G](zeta_fit_mubatch.md#refusals-and-pads)) |
| band chunks | the ψ(G) loaders and the ζ fit | a per-chunk `PaddedAxis` | masks and slices discard the tail of the final chunk |

## Refusals that are not pad opportunities

1. The runtime and the BSE ring require a supported square process topology.
2. The indefinite current-channel LU solves at the logical extent. When that
   extent does not divide the mesh axes, `linalg = local` demotes to the per-q
   replicated LU and `linalg = distributed` refuses
   ([factor and back-solve](zeta_fit_face_psi_cct.md#factor-and-back-solve)).
3. Distributed-linear-algebra and symmetry providers authenticate that a
   carrier already satisfies their collective layout. They never choose a new
   carrier extent.
4. Cyclic ring and `ppermute` modulo expressions select neighbour ranks, not
   carrier extents.

`tests/test_padding_owner_static.py` is the executable register. It scans
`src/` and the service sources and rejects any second round-up spelling,
mesh-divisor modulo or mesh-divisibility refusal outside the owner, unless the
exception is registered with its reason and follow-up. The deck doctor
(`lxkit.deck_doctor`) prints the Σ window's logical extent, carrier, divisor
and pad from `sigma_band_axis`. An indivisible physical band window is carried
by that padding, and the doctor does not refuse it at preflight.
