# Ordered photon-sector Sigma consumer

The entry `gw.mpa.sector_sigma.compute_sector_sigma` consumes the current-map
`sector-ordered-ph` manifest through the shared-pole store validator. Its
frequency planner, quadrature executor and omega accumulator are the existing
owners in `gw.mpa.sigma`. Sector construction and the signed-contact theory
belong to [the shared-pole model](../architecture/shared_pole_model.md).

The four ordered endpoint classes are CC, TT, CT and TC. CC and TT have their
own pole sets. CT and TC share the authenticated CT_C/CT_T census, with endpoint
order swapped for TC. Charge and current endpoints retain their separate
centroid bases. Gamma vertices, rectangular Green construction, FFT convolution,
and band projection use their existing owners. Fractional occupation weights
are passed unchanged by the common planner.

The hole branch conjugates both residue endpoints at minus q while retaining
the same complex time phase. Endpoint star transformations use
`symmetry_maps.unfold_endpoint_panel`, including polar time-odd Cartesian
current actions. No W, chi or G average is introduced.

The independently stored instantaneous `W_infinity-V` is contracted once with
the equal-time occupied projector through `photon_sigma.contract_lorentz_blocks`
in its exchange mode. The dispatch retains bare V exchange and bypasses the
previous static screened-current approximation. The constant carries no extra
volume factor, transverse sign or Coulomb-hole half. Photon heads remain a
separate, unsupported part of this sector handle.

## Memory and scope

One rectangular endpoint class is evaluated at a time. Factor reads use bounded
parent/pole panels. Factor faces and every large Green or interaction matrix
retain both processor axes. With centroid and pole ranks proportional to system
size, the contractions remain cubic; there is no production state/pole-pair
sum. Endpoint routing is bounded by its symmetry service cost receipt.

The constant path retains a whole all-P photon bank, then its all-P packed
replacement. Both copies and packing workspace are admitted together. Kernel
admission uses the compiler peak plus the runtime cuFFT query and distributed
GEMM workspace query; these are capacity estimates, not measured runtime peaks
or a performance comparison. Full-frequency material validation remains
separate from the focused synthetic gate.

The targeted gate is `tests/multi_device/sector_sigma_frequency_p4.py`. It
writes real endpoint stores, a bank and manifest, then calls the production
entry with unequal charge/current centroid extents, distinct sector poles,
nonreciprocal q dependence, fractional occupations and a nonzero constant. The
explicit band/q/pole oracle exists only in that harness. Its receipt must be
consulted before treating a run as passing.
