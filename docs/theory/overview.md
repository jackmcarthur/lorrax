# Theory map

This page is the entry to the theory pages: the one chain every GW mode
runs, and the principles they share. It is for a reader who wants the
equations before the code; read it before any other theory page.

LORRAX computes GW quasiparticles in an interpolative separable
density-fitting (ISDF) basis. Every mode runs the same chain,

$$
\{\psi_{n\mathbf k},\epsilon_{n\mathbf k}\}
\longrightarrow \zeta_{q\mu}
\longrightarrow (V_q,\chi^0_q)
\longrightarrow W_q
\longrightarrow \Sigma_{\mathbf k}(\omega)
\longrightarrow H_{\mathrm{QP}},
$$

in which pair densities are fitted once onto \(N_\mu\) interpolation points and
every later pair sum becomes \(N_\mu\times N_\mu\) matrix algebra with the band
sum inside a GEMM. That is what makes the method cubic in system size.
[Core ISDF and GW theory](physics.md) states the shared equations and their
costs. Each arrow has one owning page; the
[register's Theory section](../index.md#theory) lists them, with what each
owns.

These pages state equations, conventions, validity domains, costs, and the few
data layouts the equations force. Deck defaults belong to the
[input reference](../input_reference.md), module ownership to the
[codebase map](../codebase.md), and binding design rulings to
[design decisions](../architecture/decisions.md).

Three principles recur:

1. Occupation selects a spectral branch and weights it; it never redefines a
   signed band energy.
2. Pair sums are replaced by separable Green-function contractions, and
   symmetry reduces only work that commutes with unfolding; every lattice FFT
   runs on full-zone data.
3. Scalar quadrature, distributed storage and spatial physics have separate
   owners. A quadrature rule knows nothing of bands or HDF5; SlabIO decides no
   physics.
