# Coulomb sphere frequencies on an FFT grid

The sparse ζ sphere is the set of physical integer G with |(q+G)B|²≤E, where B contains Cartesian reciprocal rows in bohr⁻¹ and E is in Ry. The q rows use the canonical BerkeleyGW half-grid convention. The `vcoul.bare_coulomb_sphere_rows` service owns the predicate and physical Miller labels; `common.coulomb_sphere` applies the shared sentinel padding and modulo-FFT index layout.

On an even axis, the same FFT cell samples Miller+N/2 and−N/2. The physical representative is+N/2 at q_i<0 and−N/2 otherwise. Before making this choice, the producer proves

\[h_i=\sqrt E\,\|(B^{-1})_{:i}\|_2<N_i/2.\]

Cauchy–Schwarz bounds |G_i+q_i| by h_i. With |q_i|≤1/2, every physical integer frequency lies in the chosen box and no two admitted frequencies share a cell. This argument includes skew reciprocal lattices; odd axes retain their standard FFT labels. At q_i=0 neither even-axis Nyquist sign can enter under this strict bound. Pair reversal includes the canonical q umklapp, G′=−G−q−q′. The integer symmetry action similarly includes the rotated-q umklapp.

Outside this sufficient domain, `GATE coulomb-sphere-unique-fft-image` refuses with the measured supports and a sufficient larger grid. It is a conservative implementation limit; the code does not claim every refused discrete sphere aliases. There is no automatic enlargement or new policy input. The historical fixed-box `fft_box_miller`, `bare_coulomb_sphere_mask`, and `bare_coulomb_sphere_indices` APIs retain their original ABI and values for existing explicit fixed-box callers.

The producer processes one q at a time and retains ragged physical rows plus ascending FFT slots. It never allocates a q×NFFT×3 Miller carrier. The shared padding owner still rejects a physical frequency at the sentinel cell in a row that also has padding; positive and negative Nyquist labels obey the same modulo-cell guard.

ζ provenance resolves each channel's actual stored q domain through the existing q-grid owner. Only a physical table that differs from the historical table receives `fft_sphere_convention=q_aware_unique_fft_image_v1`. Unchanged tables retain the exact historical JSON and charge identity. Changed tables invalidate older ζ and dependent tensor restarts through the existing identity comparison. The marker never selects a physics policy.

This correction changes which fitted Fourier columns are retained. A small bound on the literal smooth-wavefunction pair tail does not certify the fitted ζ tail or screened response; a fresh matched fit must measure those changes.
