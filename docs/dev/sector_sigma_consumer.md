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

At the driver dispatch, the manifest path is selected as a handle but is not
opened as a scalar model: `compute_sector_sigma` validates it before the scalar
HDF5 consumer branch. The bare exchange owner adds charge V and, when both
the transverse carrier and bispinor V are present, transverse V once. The
sector constant is then added to the dynamic body once. Sector handles require
`head_correction = off`; the existing bispinor plus finite-temperature
occupation exception permits that setting in self-consistent maps. A material
map still requires the constructor to publish an accepted sector handle.

## Memory and scope

One rectangular endpoint class is evaluated at a time. The store reads each
sector's full parent/pole factor set once at setup; the symmetry service
unfolds it once to full q. The current map retains only those transformed
factors and replicated squared poles until that sector's frequency integration
finishes. CC/TT each read two orientations from one store; CT/TC each read
left-X and right-Y from their separate stores and compare pole arrays. The
configured `low_mem_bands` layout fixes both factor and Green placement: face
layout distributes centroid and pole axes, while axis layout replicates the
pole axis and distributes each centroid endpoint over its own axis. The face
Green's narrow band contraction uses bounded panels; W(t) uses the planned
factor GEMM with the window's exact active pole intervals. Both form one
all-P W(t) rectangle at a time. No W(t)
history or state/pole-pair sum is retained. Setup routing, resident factors, and compiled
contractions have capacity reservations with explicit sector lifetime.

The constant path retains a whole all-P photon bank, then its all-P packed
replacement. Both copies and packing workspace are admitted together. Kernel
admission uses the compiler peak plus the runtime cuFFT query and the native
workspace query for the remaining distributed GEMMs; these are capacity
estimates, not measured runtime peaks
or a performance comparison. Full-frequency material validation remains
separate from the focused synthetic gate.

The targeted gate is `tests/multi_device/sector_sigma_frequency_p4.py`. It
writes real endpoint stores, a bank and manifest, then calls the production
entry with unequal charge/current centroid extents, distinct sector poles,
nonreciprocal q dependence, fractional occupations and a nonzero constant. The
explicit band/q/pole oracle exists only in that harness. Its receipt must be
consulted before treating a run as passing.

P4 synthetic job 58497206.3 exercises both configured layouts with the same
direct oracle: maximum error is 2.046e-7 Ry in each, at a 0.008884 Ry reference
scale. The matched face-layout control 58497206.0 made 588 store face reads and
took 41.435 s for the complete consumer call; the resident route made six reads
and took 17.130 s. This is a tiny synthetic consumer measurement, including
setup and compilation, not a material speed or peak-memory claim.
