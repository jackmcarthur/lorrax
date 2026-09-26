# Theory map

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
costs; each arrow below has one owner.

| stage | question | owner |
|---|---|---|
| ζ, \(V_q\) | How are the interpolation vectors fitted and stored, and how is \(V_q\) contracted? | [G-flat ζ and V](isdf-zeta-vq.md) |
| symmetry | Which convention governs irreducible-zone work and unfolding? | [Symmetry](symmetry.md) |
| \(V_{\rm H}\) | How is the direct field built from charge and current? | [Direct Hartree field](hartree.md) |
| χ₀ quadrature | Which time rules replace static and plasmon-pole denominators, and at what node count? | [Minimax quadrature](minimax-quadrature.md) |
| χ₀ quadrature | How is the shared-pole response bank sampled on positive times? | [Compact noncrossing response](response-laplace.md) |
| \(q\to0\) | What is the long-wavelength response tensor? | [S-tensor convention](s-tensor-convention.md) |
| \(q\to0\) | Why is the exchange head direction dependent? | [LT splitting and the exchange head](lt-exchange-head.md) |
| \(q\to0\), bispinor | How do the four-current channels treat \(q\to0\), and which carry frequency? | [Four-current heads and frequency](four-current-head-corrections.md) |
| \(W\) model | What fixes the Hybertsen–Louie pole? | [HL-GPP derivation](hl-gpp-derivation.md) |
| \(W\) model | How are MPA samples taken, fitted to poles and windowed in Σ? | [Multipole frequency integration](THEORY_mpa_implementation.md) |
| \(W\) model | What is the shared-pole \(W\), and how many poles does it need? | [Shared-pole screened interaction](shared-pole-w-model.md) |
| metals | How do fractional occupations enter χ₀, the heads and Σ? | [Metallic MPA screening](metallic-mpa-screening.md) |
| Σ(ω) | How is the real-frequency denominator integrated, and what does it cost? | [The Σ(ω) quadrature problem](sigma-quadrature-problem.md) |
| \(H_{\rm QP}\) | How does the self-consistent QSGW loop run and stop? | [Self-consistency](../self_consistency.md) |

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
