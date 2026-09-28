# The Bethe–Salpeter equation

This page owns the BSE as it runs on `main`: its inputs, the screened
interaction it uses, the Hamiltonian and its matvec, the solvers, the dipoles,
the outputs and the refusals. Other owners: CLI flags and invocation,
[drivers](../drivers.md#bse-bsebse_jax); deck keys,
[input reference](../input_reference.md); the k-convolution kernels,
[the FFI layer](ffi_layout.md#kernel-operations); BerkeleyGW conventions,
`src/bse/BGW_COMPARE.md`; finite-Q exciton bands, `src/bse/EXCITON_BANDS.md`.

## Inputs

| input | producer | what the BSE reads |
|---|---|---|
| the deck (`-i`) | the GW deck | WFN path, centroid files, `nval`/`ncond`/`nband`, head settings, `bse_k_grid` |
| `isdf_tensors_<N_mu>.h5`, in the run directory or its `tmp/` | `gw.gw_jax` with `write_restart_tensors = true` (the default) | parent ψ faces at the centroids, `enk_full` (Ry), `V_qmunu`, `W0_qmunu` and its `W0_ready` flag, the head vector `G0_mu_nu`, `vhead`/`whead` |
| `WFN.h5` and the centroid table named in the deck | QE → pw2bgw; `centroid.kmeans_cli` | authentication of the bundle, and the static-screening rebuild |
| `eqp1.dat` (`--eqp`, optional) | `gw.gw_jax` | diagonal QP energies on the wedge, unfolded through the symmetry service |
| `dipole.h5` (`--dipole`) | `psp.get_dipole_mtxels` | velocity matrix elements and `band_energies` for absorption and per-state dipoles |
| `dipole_qsgw.h5` (`--dipole`) | a QSGW run with `sc_head_update = interband_commutator` | the same layout: $U^\dagger(v + [\Delta H, W])U$ with `band_energies` = $E_{QP}$ |

Exactly one `isdf_tensors_*.h5` may sit in the run directory and its `tmp/`
(`file_io.restart_bundle._find_restart_file`). Every read of the parent faces
authenticates the bundle against the deck (`restart_bundle.unfold_parent_faces`):
the WFN fingerprint, the centroid content hash, the stored band window and
$N_\mu$ (`assert_restart_window_matches`), and the parent rows.

**Band window.** The valence count comes from `--n-occ` or the Fermi level of
`enk_full` (`bse_window.resolve_n_occ`). The window is
$[n_{occ} - n_v,\; n_{occ} + n_c)$, clamped to the stored bands. An edge inside
a multiplet follows `--band-degeneracy` (`common.band_degeneracy`). Every band
of the window must lie in both legs of the ζ fit, because the direct term reads
$v\times v'$ and $c\times c'$ pair densities
(`bse_window.assert_bse_window_in_zeta_training`). $n_c$ and $n_v$ are padded to
the mesh with zero ψ and a signed ε sentinel (`PAD_EPS_GUARD_RY`), so pad
states decouple.

## The screened interaction W(ω = 0)

The direct term is always screened. Bare V is never used as W:
`restart_bundle.read_bse_payload` refuses without a stored W0 or a rebuilt one
(`GATE bse_requires_screened_w0`).

**Stored W0.** A bundle with `W0_ready = true` is read as it is. Which GW runs
store it (`gw.screening.driver_persists_w0`, `restart_static_w`):

| GW run | `W0_qmunu` |
|---|---|
| `cohsex`, `gn_ppm`, `hl_ppm` with `screening_diagrams = w_rpa` | the Dyson W(0) |
| `mpa` with `sigma_w_model = shared_pole`, scalar store, one-shot | $V + W^c(0)$ of the model (`gw.mpa.sigma.shared_pole_static_wc`) |
| the same, `qp_solver = self_consistent` | $V + W^c(0)$ of the accepted final map, evaluated once after the loop |
| `w_bse`, `w_rpa_resolvent` | the RPA W(0), written by the ladder stage before the ladder runs |
| `x_only`; `mpa` with `sigma_w_model = mpa`; bispinor `mpa` | nothing |
| shared pole on a metal, or `head_correction = full` without an ω = 0 head sample | nothing; the log says why |

**Rebuilt W0.** With `W0_ready = false`, the loader calls
`gw.static_screening.build_static_w_from_restart` (R56, `2784de492`). It reads
the WFN and centroids named in the deck, authenticates them against the bundle's
receipt, reuses the bundle's ζ and V, and solves the static Dyson equation
$W = (1 - V\chi_0)^{-1}V$ at one frequency through the response owner
(`gw.screening.compute_screening`). Insulators use the static minimax
quadrature; Fermi–Dirac occupations use the ν = 0 Matsubara response
(`w_isdf.compute_chi0_matsubara`). The q→0 head at ω = 0 comes from the head
owner (`qsgw_head.build_dft_head_response`) unless `head_correction = off`. No Σ
is evaluated, no pole is fitted, and the bundle is not modified: the rebuild
runs on every BSE call.

Measured against a stored shared-pole W0 on the same bundle (claim 2888,
P4 A100): the ten lowest exciton energies moved by at most 0.0078 meV (MoS2
3×3, 4v × 6c) and 0.00002 meV (Si 4³, 4v × 8c); ε₂ peaks are identical on a
5 meV grid. The rebuild doubled the BSE wall in those runs: 9.83 → 19.34 s on
MoS2, 10.50 → 20.31 s on Si. A metal's static producer runs (Na 8³); no metallic
exciton spectrum has been validated.

**The q = 0 head.** The loader adds the rank-one head
$(v_{head}/\Omega)\,g_0^*g_0$ to the q = 0 exchange tile, and $w_{head}$ to the
q = 0 W tile only when W is screened (`bse_head._inject_q0_head`). The
values come from the rebuilt head, else the bundle's `vhead`/`whead`; deck keys
`vhead` and `whead_0freq` override both (`bse_head._resolve_head_params`). A
rebuild on a bundle without `G0_mu_nu` refuses.

## Hamiltonian

In the transition basis $|vk \to ck\rangle$, with resonant $X$ and
antiresonant $Y$ amplitudes,

$$\begin{pmatrix} A & B \\ -B^* & -A^* \end{pmatrix}\begin{pmatrix} X \\ Y \end{pmatrix} = \Omega \begin{pmatrix} X \\ Y \end{pmatrix},\qquad
A = D + w_x V - W,\qquad B = w_x V^B - W^B,$$

with $D_{cvk} = \varepsilon_{ck} - \varepsilon_{vk}$, $V$ the bare exchange at
q = 0 and $W$ the static screened direct term at momentum transfer $k - k'$.
$w_x = 2$ for a spin-restricted scalar run (singlet) and $w_x = 1$ for spinors,
whose pair amplitude already sums both components
(`bse_preconditioner.exchange_spin_weight`, applied at every exchange encode).
The Tamm–Dancoff approximation keeps $A$; the RPA kernel drops $W$ and $W^B$.

With $\psi_{nk,s}(\mu) \equiv \psi_{nk,s}(r_\mu)$:

- pair amplitude $M_{cv}(\mu,k) = \sum_s \psi^*_{ck,s}(\mu)\,\psi_{vk,s}(\mu)$,
  built once per solve (`compute_pair_amplitude`);
- exchange, dense in $(k,k')$:
  $(VX)_{cvk} = \dfrac{w_x}{N_k} \sum_{\mu\nu} M_{cv}(\mu,k)\, V_{\mu\nu}
  \sum_{c'v'k'} M^*_{c'v'}(\nu,k')\, X_{c'v'k'}$;
- direct term through $T_{ts}(\mu,\nu,k) = \sum_{cv} \psi_{ck,t}(\mu)\,\psi^*_{vk,s}(\nu)\,X_{cvk}$:
  $$(WX)_{cvk} = \frac{1}{N_k}\sum_{\mu\nu ts} \psi^*_{ck,t}(\mu)\,\psi_{vk,s}(\nu) \sum_{k'} W_{\mu\nu}(k-k')\, T_{ts}(\mu,\nu,k').$$

The $k'$ sum is a k-grid convolution,
$U = \mathrm{FFT}_k\big(W_R \cdot \mathrm{IFFT}_k T\big)$, with $W_R$ the
screened tile already in R space. A dense $N_k \times N_k$ contraction is not
used: it is $O(N_k^2)$ against the FFT's $O(N_k \log N_k)$.

`head_minibz_average = true` adds the cell-averaged exchange head as a rank-3
term over transitions, $K^{head} = \frac{1}{N_k}\, d^*_a M_{ab} d_b$
([LT head](../theory/lt-exchange-head.md)).

`bse_k_grid` densifies the whole bundle before any solve: ψ and ε through one
htransform $f(H)$, V through `bse.vq_interp`, W by zero-padding in R
(`bse.bse_densify`).

## The matvec

`bse_stack_matvec.build_bse_stack_matvec` is the one TDA/RPA matvec every
sharded solver uses. Per trial block:

1. one all-gather of the block over `'y'` then `'x'`: each rank holds every
   trial whole, $(n_c, n_v, N_k)$;
2. a `lax.scan` over trials with no collective in the body: the W term into the
   rank's $(\mu_{loc}, \nu_{loc})$ partial of $(n_c, n_v, N_k)$;
3. one reduce-scatter back to the `X` layout.

The exchange runs outside the scan: a k-summed encode to
$(n_{trials}, N_\mu)$, one product with $V_{q0}$, a broadcast decode.

**The W term** takes the first route that serves the mesh, the k-grid and the
band counts. The choice is made at trace time and printed once (`[bse] W term:`
and `[bse] W term decode:`):

| route | door (`ffi.fft`) | CUDA target | T and U |
|---|---|---|---|
| fused outer load + decode (R45, ping-pong R54) | `make_local_kconv_klead_outer_decode` | `lorrax_mathdx_kconv_klead_outer_decode` | T formed on the load, U contracted in the store; neither reaches HBM |
| outer load, XLA decode (R44) | `make_local_kconv_klead_outer` | `lorrax_mathdx_kconv_klead_outer` | T formed on the load; U stored |
| XLA encode, k-leading convolution | `make_local_kconv_klead` | mode 2 | T and U stored, one T tile per rank |

On CUDA the outer routes form $T = \sum_K L\,R$, $K = \min(n_c, n_v)$, on the
fp64 tensor cores; on a cpu mesh the same doors compose an einsum and the plan
route. On the fused route, `LORRAX_BSE_OUTER_KSUM=fma` moves the K sum to the
FMA pipe for A/B comparison ([env registry](../dev/env_vars.md)). When each
route refuses: `ffi.fft.klead_outer_refusal` and `klead_outer_decode_refusal`
([doors](ffi_layout.md#k-convolution-router-and-the-mathdx-family)). The decode
keeps the XLA route's (t, μ)-first order on every route.

Full BSE uses `build_bse_stack_pair_matvec`, the real-linear applier
$\mathrm{pair}(X, s) = AX + s\,B\bar X$ ($s = \pm 1$ traced; Shao–da
Jornada–Yang, Algorithm 4). The $A$ and $B$ encodes share one shape and one
sharding, so their sum takes one convolution and one decode.
`bse_ring_comm.build_bse_ring_matvec_full` stays as the dense $(A, B)$ oracle of
the equality gates and as the ladder's operator for `w_bse`.

Measured on `main` 05b2e4f7 (release R54 gate; kernel claim 2874): CrI3
8×8×1 SOC, 8v × 14c, P4 A100-40GB. One Haydock step (D + V − W on three trials) takes 13.2–13.4 ms;
100 steps take 1.28 s; the solve peak is 2159 MiB per rank. The fused kernel
runs at 2.27× the flop floor.

## Solvers

| route | driver flag | module | use |
|---|---|---|---|
| FEAST | default (no `--lanczos`) | `bse.bse_feast` | contour eigensolve in KPM-sized windows |
| Lanczos, block Lanczos | `--lanczos` | `bse_lanczos.solve_bse_sharded` | spectrum shape; CGS2 reorthogonalization, full window by default |
| Davidson | `--lanczos --solver davidson` | `bse.bse_davidson_helpers` | per-state convergence |
| thick-restart Lanczos | `--lanczos --solver trlan` | `solvers.thick_restart_lanczos` | Krylov memory capped at `--trlan-m-max` |
| full BSE | `--lanczos` without `--tda` | `bse.bse_nontda` | structure-preserving non-TDA solve |
| density of states | `--kpm-dos` | `bse.bse_kpm` | KPM Chebyshev moments |
| absorption | `python -m bse.absorption_haydock` | `bse.absorption_haydock` | ε₂(ω) by continued fraction, no eigenvectors |

## Dipoles and absorption

The dipole files hold velocity matrix elements. Absorption uses the position
form $d^\alpha_{cvk} = \langle ck|\hat v_\alpha|vk\rangle / (E_c - E_v)$
(`absorption_common.slice_dipole_to_bse_window`), with the energies of the
dipole file's own `band_energies`: DFT energies for `dipole.h5`, $E_{QP}$ for
`dipole_qsgw.h5`. `--eqp` changes the transition energies of the Hamiltonian,
not the dipoles.

`absorption_haydock` seeds one Lanczos recursion per polarization; the three
seeds are one trial block of the stack matvec. It does not reorthogonalize and
runs 200 steps by default. At 200 steps both the plain and the CGS2 recursion
match the exact resolvent to ≤ 1e-5 of max ε₂ on CrI3 8×8×1 SOC (claim 2848);
at 100 steps both are truncated, about 0.5% of max ε₂ off.
$\varepsilon_2^\alpha(\omega) = \frac{16\pi^2}{\Omega N_k n_{spin} n_{spinor}}
\|d^\alpha\|^2 \left(-\mathrm{Im}\,g(\omega + i\eta)/\pi\right)$, as in
BerkeleyGW's `absh.f90`.

Per-state dipoles: `bse_jax --lanczos --tda --write-eigs N --dipole dipole.h5`
contracts each written eigenvector with the dipole
(`absorption_common.exciton_dipole_projections`, on each rank's own eigenvector
tile) and stores $\langle 0|\hat r_\alpha|S\rangle$ as `exciton_data/dipoles`,
shape (1, N, 3, 2), in bohr. `--dipole` requires `--write-eigs` and `--tda`.

## Outputs

- `bse.out`: the run report (`--report-file`).
- `eigenvectors.h5` (`--write-eigs`), rank 0, BerkeleyGW layout
  (`src/bse/eigenvectors.h5.spec`) through `bse_window.write_eigenvectors_stream`:
  `exciton_data/eigenvalues` in eV; `exciton_data/eigenvectors`
  (1, N, N_k, n_c, n_v, 1, 2); full BSE adds `eigenvectors_coupling` (Y). The
  valence axis is reversed on write (BerkeleyGW `iv = 1` is the highest valence
  band). The writer refuses to trim nonzero amplitude when the declared window
  is narrower than the solved one.
- `absorption_haydock.h5` (ε₂, ε₁, JDOS, α, β, norms) and one
  `absorption_haydock_<pol>_eh.dat` per polarization.

## W_BSE: the ladder in GW screening

`screening_diagrams = w_bse` is a GW option: Σ uses the ladder-corrected
$W(z) - v = v(z - H)^{-1}v$, with the BSE Hamiltonian's static direct rung in
$H$ (`gw.screening_bse`, `bse.w_ladder`). The stage persists the RPA W(0), then
solves the resolvent one z at a time with ψ held 2-D at 1/P, by GMRES with a
recycled spectral deflation per (q, z) (`bse_feast.harvest_spectral_deflation`).
It runs under `compute_mode = cohsex` and `gn_ppm`. Keys, supported modes and
refusals: [input reference](../input_reference.md).

## Refusals

| refusal | where | condition |
|---|---|---|
| `GATE bse_restart_ambiguous` | `restart_bundle._find_restart_file` | more than one `isdf_tensors_*.h5` in the run directory and its `tmp/` |
| `GATE bse_requires_screened_w0` | `restart_bundle.read_bse_payload` | no stored W0 and no rebuilt one |
| `GATE bse_static_w_inputs` | `static_screening.build_static_w_from_restart` | a rebuild without `-i` |
| `GATE bse_static_w_bispinor_sectors` | the same | a four-current bundle without W0: the BSE has no packed CC/CT/TC/TT handoff, and a scalar rebuild would omit the coupled sectors |
| `GATE bse_static_w_sc_state` | the same | a QSGW deck without final-map W0: the parent WFN is the DFT state |
| `GATE bse_static_w_provenance` | the same | the bundle has no WFN/centroid receipt |
| `GATE bse_static_w_head_vector` | `bse_loading.load_bse_data_from_restart_sharded` | a rebuild on a bundle without `G0_mu_nu` |
| `BseWindowOutsideZetaTrainingError` | `bse_window.assert_bse_window_in_zeta_training` | a window band outside the ζ fit legs |
| band-window degeneracy | `common.band_degeneracy.resolve_band_window` | a window edge inside a multiplet under `--band-degeneracy strict` |
| route-ignored flag | `bse_jax.parse_args` | a flag set on a route that does not read it; a retired flag |

## Limits

These are open entries in the development tracker `KNOWN_LORRAX_ISSUES.md`,
named by title there; the entries hold the mechanism and the evidence.

- Four-current BSE: only a stored charge W0 is read (the
  `bse_static_w_bispinor_sectors` refusal above).
- QSGW → BSE uses the parent DFT ψ with `--eqp`, the diagonal approximation.
  QP ψ need a restart generated from `WFN_qp.h5`.
- "MPA × W_BSE is still refused on parent-route bundles".
- "W_BSE ladder resolvent does not converge at finite q on Si 4³ SOC
  (preconditioned spectrum straddles zero)".
- "Spinor Kramers TRIM-closure gate is tighter than the WFN's own error".
- "The unchunked w_bse ladder assembly loses q=0 Hermiticity at production
  size" (set `ladder_probe_chunk = 64`).
- "Scalar-Si w_bse wedge reciprocity residual sits at ~2e-3 after TRS pair
  gauging".
- "`psp.get_dipole_mtxels`: the j-averaged (`soc=False`) velocity is all NaN".
