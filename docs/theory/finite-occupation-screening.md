# Finite-occupation RPA screening

**This page has been subsumed by
[Metallic MPA screening](metallic-mpa-screening.md).** That chapter is the
authoritative owner of everything this note used to state — the
`f_a(1-f_b)` weight factorization and its cancellation analysis, the
positive-time contour kernel and its `q`/`-q` orientation identity, the
`build_G_tau(band_weight=)` seam, the exact static divided-difference
kernels at `Gamma` and finite `q`, the support-derived rule bandwidths, the
staged separable rational-`f` target, and the relation of the finite-`q`
body to the `q -> 0` heads — plus the metal frequency plan, the
occupation-weighted `Sigma`, and the QSGW occupation state.

This file remains only as a stable link target: in-code comments and
docstrings (e.g. in `gw.w_isdf`, `gw.ppm_windows`, `gw.ppm_tau_kernel`,
`gw.mpa.sigma`) cite `docs/theory/finite-occupation-screening.md`. Follow
them here, then read the chapter above. Do not add content to this page.

## References

- Sesti et al., *Efficient GW calculations for metals from an accurate ab
  initio polarizability*, arXiv:2508.06930 (2025).
  https://arxiv.org/abs/2508.06930
- Kim, Martyna, and Ismail-Beigi, *Complex time, shredded propagator method
  for large-scale GW calculations*, Phys. Rev. B 101, 035139 (2020).
  https://doi.org/10.1103/PhysRevB.101.035139
- Rojas, Godby, and Needs, *Space-Time Method for Ab Initio Calculations of
  Self-Energies and Dielectric Response Functions of Solids*,
  Phys. Rev. Lett. 74, 1827 (1995).
  https://doi.org/10.1103/PhysRevLett.74.1827
