# The Bethe–Salpeter equation

This page owns the BSE as it runs on `main`: its inputs, the screened
interaction it uses, the Hamiltonian and its matvec, the solvers, the dipoles,
the outputs and the refusals. Other owners: CLI flags and invocation,
[drivers](../drivers.md#bse-bsebse_jax); deck keys,
[input reference](../input_reference.md); the k-convolution kernels,
[the FFI layer](ffi_layout.md#kernel-operations); BerkeleyGW conventions,
[BerkeleyGW users](../how-to/berkeleygw-users.md#bse); finite-Q exciton
bands, [exciton bands](../how-to/htransform-and-exciton-bands.md#7-exciton-bands).

## Inputs

| input | producer | what the BSE reads |
|---|---|---|
| the deck (`-i`) | the GW deck | WFN path, centroid files, `nval`/`ncond`/`nband`, head settings, `kgrid_fine` |
| `isdf_tensors_<N_mu>.h5`, in the run directory or its `tmp/` | `gw.gw_jax` with `write_restart_tensors = true` (the default) | parent ψ faces at the centroids, `enk_full` (Ry), `V_qmunu`, `W0_qmunu` and its `W0_ready` flag, the head vector `G0_mu_nu`, `vhead`/`whead`; on a static-Dyson W0, the dressed long-range channels `W0_lr` |
| `WFN.h5` and the centroid table named in the deck | QE → pw2bgw; `centroid.kmeans_cli` | authentication of the bundle, and the static-screening rebuild |
| `eqp1.dat` (`eqp_corrections`, optional) | `gw.gw_jax`, or BerkeleyGW's `sigma` | diagonal QP energies: the WFN's file wedge, unfolded through the symmetry service, or a full-zone file (a block for every full-BZ k), where every k takes its own block as BerkeleyGW reads it (`restart_bundle._eqp_on_full_bz`) |
| `qp_wfn_rotations.h5` (`qp_rotations_file`, optional; exclusive with `eqp_corrections`) | `gw.gw_jax` (`file_io.qp_wfn.write_qp_rotations_h5`) | the GW run's $H_{QP} = U\,\mathrm{diag}(E_{QP})\,U^\dagger$ on its Σ window, unfolded to the full BZ by `restart_bundle.read_qp_rotations_artifact`: the BSE runs in its eigenbasis ([QP-rotated basis](#the-qp-rotated-basis)) |
| `dipole.h5` (`dipole_file`) | `psp.get_dipole_mtxels` | velocity matrix elements and `band_energies` for absorption and per-state dipoles; under `kgrid_fine`, the deck (coarse) WFN's file, which the densifier carries to the fine grid |
| `dipole_qsgw.h5` (`dipole_file`) | a QSGW run with a velocity head ([QSGW dipoles](../self_consistency.md#qsgw-dipoles)) | the same layout: $U^\dagger v\,U$ with the head's velocity $v$ (`qsgw_head.qp_velocity`) and `band_energies` = $E_{QP}$; stamped `basis = qp` and, when the run writes `WFN_qp.h5`, bound to that WFN's fingerprint. It pairs only with a restart built from `WFN_qp.h5` |

Exactly one `isdf_tensors_*.h5` may sit in the run directory and its `tmp/`
(`file_io.restart_bundle._find_restart_file`). Every read of the parent faces
authenticates the bundle against the deck (`restart_bundle.unfold_parent_faces`):
the WFN fingerprint, the centroid content hash, the stored band window and
$N_\mu$ (`assert_restart_window_matches`), and the parent rows.

**Band window.** The occupied-band count $n_{occ}$ is `highest_occupied_band` (Lanczos
route), else the deck WFN's `ifmax` (`WfnLoader.nelec`,
`bse_window.resolve_n_occ`). The window is
$[n_{occ} - n_v,\; n_{occ} + n_c)$, clamped to the stored bands. An edge inside
a multiplet follows `band_degeneracy` (`common.band_degeneracy`). Every band
of the window must lie in both legs of the ζ fit, because the direct term reads
$v\times v'$ and $c\times c'$ pair densities
(`bse_window.assert_bse_window_in_zeta_training`). $n_c$ and $n_v$ are padded to
the mesh with zero ψ and a signed ε sentinel (`PAD_EPS_GUARD_RY`), so pad
states decouple.

### The QP-rotated basis

`qp_rotations_file` (`bse.bse_qp_rotation`, native grid only) solves the BSE in the
eigenbasis of the GW run's effective Hamiltonian on its Σ window $[q_0, q_1)$,
$H_{QP}(k) = U(k)\,\mathrm{diag}(E_{QP}(k))\,U(k)^\dagger$ with
$U_{mn}(k) = \langle m_\mathrm{DFT}k | n_\mathrm{QP}k\rangle$. The centroid states are

$$\psi'_{nk}(\mu) = \sum_{m=q_0}^{q_1-1} \psi_{mk}(\mu)\,U_{mn}(k),$$

over the complete block, which must lie in both ζ-fit legs; ε are $E_{QP}$; V and
W0 are reused, because they live in the centroid basis. `number_val_bands_fine`/`number_cond_bands_fine`
count QP states of the ascending $E_{QP}$ ladder, and the degeneracy guard runs on
that ladder. A QP state of the window may be dominated by a DFT band outside the
DFT-index window of the same counts; the log prints that leakage,
$\max_{k,n}\,(1 - \sum_{m\in[lo,hi)} |U_{mn}|^2)$, and the cross-gap mixing
$\theta = \max_k \lVert U_{VC}\rVert_F$.

The artifact unfolds a wedge-stored $U$ as $U(gp) = U(p)$ for unitary $g$ and
$U(p)^*$ for antiunitary $g$ (`symmetry_maps.star_broadcast`, `ibz_slab`). That
is the restart's gauge only if its ψ obeys $\psi_{gp} = g\,\psi_p$, so the loader
certifies it on the restart's own ψ: the centroid band matrices of 1 and of
$\sigma_z$, $S_{k,mn} = \sum_{\mu,s} \sigma_s\,\psi^*_{mks}(\mu)\,\psi_{nks}(\mu)$,
must equal $S_p$ or $S_p^*$ by the same rule to $10^{-10}$, and the artifact's own
$H_{QP}(k)$ likewise; the log prints the opposite rule's residual as the check's
sensitivity.

The diagonal of a one-shot $H_{QP}$ is $\Sigma$ at $E_\mathrm{DFT}$ without the
linearization factor $Z$ (`eqp0.dat`), not `eqp1.dat`; the like-for-like
diagonal control of a rotated run is `eqp_corrections eqp0.dat`, and the rotated minus
that control is the off-diagonal $\Sigma - v_{xc}$ alone.

## The screened interaction W(ω = 0)

The direct term is always screened. Bare V is never used as W:
`restart_bundle.read_bse_payload` refuses without a stored W0 or a rebuilt one
(`GATE bse_requires_screened_w0`). A kernel without the direct term
(`local_fields = true`) loads with `need_W = False`: no W is read, rebuilt or
kept (the matvec's W slot is a (p_x, p_y, 1, 1, 1) zero tile), the screened
q = 0 head is not taken, and a bundle without W0 loads.

**Stored W0.** A bundle with `W0_ready = true` is read as it is. Which GW runs
store it (`gw.screening.driver_persists_w0`, `restart_static_w`):

| GW run | `W0_qmunu` |
|---|---|
| `cohsex`, `gn_ppm`, `hl_ppm` with `screening_diagrams = w_rpa` | the Dyson W(0) |
| `mpa` with `sigma_w_model = tangential`, scalar store, one-shot | $V + W^c(0)$ of the model (`gw.mpa.sigma.tangential_static_wc`) |
| the same, `qp_solver = self_consistent` | $V + W^c(0)$ of the accepted final map, evaluated once after the loop |
| `w_bse`, `w_rpa_resolvent` | the RPA W(0), written by the ladder stage before the ladder runs |
| bispinor `mpa`, tangential, `bispinor_gw = bare_transverse`, one-shot or self-consistent | $V + W^c(0)$ of the four-component charge store on the charge V, the same evaluator (charge sector only) |
| bispinor `mpa`, `bispinor_gw = full_tangential` | $V + W^c_{CC}(0)$ of the CC sector (`gw.mpa.sector_sigma.sector_static_wc`) |
| `x_only`; `mpa` with `sigma_w_model = mpa` | nothing |
| tangential on a metal, or `head_correction = full` without an ω = 0 head sample | nothing; the log says why |

W0 is stored on the q parents of `V_qmunu`, with their unfold tables, whenever
the run's q axis reduces (`restart_q_storage = auto` resolves `ibz`;
[symmetry register](symmetry_register.md)); a deck whose q axis does not reduce
stores every q. GW never forms a full-q W0 for the file: the tangential
evaluators run at the parent rows and the writer stores the producer's parents
(`gw_output.persist_w0_and_head`). The BSE reader restores the full q grid on
load (`restart_bundle.read_interaction` → `symmetry_maps.unfold_isdf_operator`,
bounded q tiles), which is the layout the kernels read. Files of either layout
read.

**Rebuilt W0.** With `W0_ready = false`, the loader calls
`gw.static_screening.build_static_w_from_restart`. It reads
the WFN and centroids named in the deck, authenticates them against the bundle's
receipt, reuses the bundle's ζ and V, and solves the static Dyson equation
$W = (1 - V\chi_0)^{-1}V$ at one frequency through the response owner
(`gw.screening.compute_screening`). Insulators use the static minimax
quadrature; Fermi–Dirac occupations use the ν = 0 Matsubara response
(`w_isdf.compute_chi0_matsubara`). The q→0 head at ω = 0 comes from the head
owner (`qsgw_head.build_dft_head_response`) unless `head_correction = off`. No Σ
is evaluated, no pole is fitted, and the bundle is not modified: the rebuild
runs on every BSE call.

Measured against a stored tangential W0 on the same bundle (claim 2888,
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
`bse_jax` applies the RPA kernel, D + V, unless `spin_singlet = true` is given; `full_bse = false`
selects the Tamm–Dancoff problem, and without it every route solves the full
one.

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

`kgrid_fine` densifies the bundle before any solve (`bse.bse_densify`). ψ and
ε go through one htransform $f(H)$; with `--eqp` its energies are the
quasiparticle ones, so the fine ε are QP. W is embedded in R per centroid
pair, at the Wigner–Seitz images of minimal |R + L T + r_μ − r_ν|
(`common.ws_images`), after the srlr model (the only one since 2026-10-22;
`w_head_densify = c1` refuses) removes a fitted model of W's whole long-range part (the screened head and, in a slab, its G_z
channels) before the embedding and adds it back at every fine q, from the
dressed channels GW stored in `W0_lr`
([SR/LR densification](../theory/w-srlr-densification.md)). Both
additions go into the fine W's donated buffer: the long-range model in q
blocks of the fixed tile, the head as one fused pass. The shared W densifier
scans independent centroid-face tiles: a local inverse FFT, the same exact
pair-dependent WS embedding, then the fine FFT. It writes each completed
tile directly into the final fine W, keeping coarse input and fine output
resident with bounded FFT/WS scratch. The final five-dimensional accumulator
also avoids a whole-fine-W layout copy at the scan boundary. The
q = 0 exchange tile is k-grid invariant and is carried through unchanged,
unless `head_minibz_average = true`: then it is rebuilt through
`bse.vq_interp` with the fine grid's mini-BZ head
([LT head](../theory/lt-exchange-head.md)), and the W head's Γ-cell reference
uses the analytic sphere. Without `kgrid_fine` the optical BSE does not read
`head_minibz_average`. With `dipole_file`, the velocity goes through the same
htransform as ψ ([dipoles](#dipoles-and-absorption)).

## The matvec

`bse_stack_matvec.build_bse_stack_matvec` is the TDA matvec: Lanczos,
Davidson and thick-restart Lanczos, FEAST and KPM under `full_bse = false`, the
spectral-bound Lanczos of FEAST and KPM on either route, Haydock, and
`bse.exciton_bands`. Per trial block:

1. one all-gather of the block over `'y'` then `'x'`: each rank holds every
   trial whole, $(n_c, n_v, N_k)$;
2. a `lax.scan` over trials with no collective in the body: the W term into the
   rank's $(\mu_{loc}, \nu_{loc})$ partial of $(n_c, n_v, N_k)$;
3. one reduce-scatter back to the `X` layout.

The exchange runs outside the scan: a k-summed encode to
$(n_{trials}, N_\mu)$, one product with $V_{q0}$, a broadcast decode.

**The W term** has three routes, chosen at trace time by what the device
serves and printed once (`[bse] W term:`, `[bse] W term decode:`):

1. T is formed from its two legs, $T = \sum_K L\,R$ with $K = \min(n_c, n_v)$,
   on the convolution's load, and the decode's (t, μ) contraction runs in its
   store: neither T nor U reaches HBM (two warp groups in ping-pong).
2. T is formed on the load; U is stored and decoded by XLA.
3. XLA builds T; the k-leading convolution reads it and writes U.

The CPU plan backend and a plain XLA fallback use route 3. Their outer/decode
factories compose materialized T and U arrays, so they cannot use route 1's
whole-tile memory plan. Native CPU FFTs reduce convolution scratch while the
same block plan bounds the stored T/U carrier.

The kernels, their refusals and the `LORRAX_BSE_OUTER_KSUM` A/B switch are in
[the FFI layer](ffi_layout.md#kernel-operations). The decode keeps the
(t, μ)-first order on every route.

**Blocks.** T, U and $W_R$ are local in $(\mu, \nu)$: T is an outer product of
its two ζ legs at each k, the convolution acts on each $(t\mu, s\nu)$ column
with its own $W_R(\mu, \nu)$, and the decode sums over μ and ν. The W term is
therefore an exact sum over blocks $I_i \times J_j$ of the rank's
$(\mu_{loc}, \nu_{loc})$ tile,

$$(WX)_{cvk} = \sum_j \frac{1}{\sqrt{N_k}} \sum_{s,\,\nu \in J_j} \psi_{vk,s}(\nu)\, A^{(j)}_{cs\nu}(k),\qquad
A^{(j)}_{cs\nu}(k) = \sum_i \sum_{t,\,\mu \in I_i} \psi^*_{ck,t}(\mu)\, U^{(ij)}_{t\mu,s\nu}(k),$$

with $U^{(ij)}$ the convolution of T's $(I_i, J_j)$ block with $W_R$'s:
blocking reorders the μ and ν sums and nothing else.
`bse_stack_matvec.plan_w_blocks` sizes the blocks from the shapes alone (the
[fixed-tile rule](decisions.md#fixed-tile)). Routes 2 and 3 store U (route 3's
T is U, in place), so they take the largest square-ish block, widths multiples
of 8 or the whole axis, whose U block and $W_R$ block,
$16\,N_k\,(n_s^2 + 1)\,b_\mu b_\nu$ bytes, fit `runtime.tiles.TILE_BYTES`
(1 GiB). Route 1 stores neither (its handler bounds its scratch at U/16) and
keeps the whole tile. On every route each operand of a block's handler call
stays below $2^{31}$ elements, the outer load's 32-bit offsets. One block is
the unblocked program, bit for bit. Otherwise, inside the trial scan, a
`lax.scan` over ν blocks carries the $(c, v, k)$ partial and a nested
`lax.scan` over μ blocks carries $A^{(j)}$; each block's $W_R$ slice is laid
out in its step, and the last block of an axis is clamped onto its neighbour
with the shared rows zeroed in the legs. The slice is cut from the tile's 2-D
$(\mu_{loc}, \nu_{loc} N_k)$ view (`_w_block`): cut from the 5-D tile, XLA's GPU
layout assignment meets route 3's k-leading kernel operand by relaying out the
whole loop-invariant tile ahead of the scans, one tile-sized copy (32.4 GB per
rank at CrI3 30×30, $N_\mu$ = 3000, four ranks, where the AOT gate refused the
Davidson module at 98.5 GB; 66.1 GB from the 2-D view). No layout of the 2-D
view makes the block's transpose a bitcast, so it stays a block-sized copy. On a route-1 block the fused store
forms $A^{(ij)}$ directly. CrI3 30×30 at $N_\mu$ = 3000 on four ranks
(route 3; U alone would be 130 GB per rank per trial) plans 13 × 13 blocks of
120 × 120. The plan, the U block and the W term's per-rank price print once
(`[bse] W term blocks:`); a price over the device budget warns and runs.

Full (non-TDA) BSE runs on the same kernels. `build_bse_stack_pair_matvec` is
the real-linear applier $\mathrm{pair}(X, s) = AX + s\,B\bar X$ (Shao–da
Jornada–Yang, Algorithm 4): the B encode swaps the two ζ legs, so $T_A + s\,T_B$
goes through one convolution and one decode, and the sign rides each trial.
`build_bse_stack_matvec_full` is $H[X; Y]$ from one pair call on the $2b$
trials $z = X + \bar Y$ ($s = +1$) and $w = X - \bar Y$ ($s = -1$):
$F + G = 2(AX + BY)$ and $\overline{G - F} = 2Y_{out}$, exact. It is the
operator of FEAST and KPM with `full_bse = true`; `bse_nontda` builds the dense
$(A, B)$ (N ≤ 4096) from pair calls on real identity columns,
$A = (F + G)/2$, $B = (F - G)/2$, 256 columns a call, and its matrix-free
route runs on the same applier. Its live set, like the TDA matvec's, is one
T-family per rank whatever the trial width. Against the ring matvec it
replaced (CPU, one device, $N_k = 4^3$, 8c × 4v, $N_\mu$ = 192, $n_s$ = 2): 552
against 1063 ms for one $[X; Y]$ trial and 375 against 1195 ms a trial at four,
with XLA scratch 342 MiB flat against 869, 3467 and 6923 MiB at 1, 4 and 8
trials; $H[X; Y]$ agrees with the ring's to 2.8e-16 relative and the dense
spectra to 1.4e-14 Ry (synthetic spinor and scalar payloads and the H2⁻
restart, 1×1 and 2×2).
`bse_ring_comm.build_bse_ring_matvec_full` remains the screening operator of
`bse.w_ladder` (`w_bse`, `w_rpa_resolvent`), `bse.bse_w_exact` and
`bse.w_omega_chain`, and refuses `screening=False`.

Measured (claim 2874): CrI3
8×8×1 SOC, 8v × 14c, P4 A100-40GB. One Haydock step (D + V − W on three
trials) takes 13.2–13.4 ms; 100 steps take 1.28 s; the solve peak is 2159 MiB
per rank. The fused kernel runs at 2.27× the flop floor.

## Solvers

| route | driver flags | module | use |
|---|---|---|---|
| Davidson (default) | `full_bse = false` (no route flag; `the eigensolve` is an alias) | `solvers.davidson.davidson` with `lock_converged`, the exact diagonal preconditioner and start subspace from `bse.bse_davidson_helpers` | lowest `number_eigenvalues` roots, stopped when every residual is below `davidson_tol_mev` (0.01 meV) |
| Lanczos, block Lanczos | `full_bse = false solver = block_lanczos` | `bse_lanczos.solve_bse_sharded` | spectrum shape; fixed Krylov dimension, CGS2 reorthogonalization, full window by default |
| thick-restart Lanczos | `full_bse = false solver = trlan` | `solvers.thick_restart_lanczos` | Krylov memory capped at `trlan_m_max` |
| full BSE | no route flag, without `full_bse = false` | `bse.bse_nontda`, dense build | structure-preserving non-TDA solve, N ≤ 4096 |
| FEAST | `--feast`; `--tda` or not | `bse.bse_feast` | contour eigensolve in KPM-sized windows; refuses unless every in-window Ritz residual is below `--feast-tol-mev` (0.01 meV, `GATE feast_converged`) |
| density of states | `--kpm-dos`; `--tda` or not | `bse.bse_kpm` | KPM Chebyshev moments |
| absorption | `python -m bse.absorption_haydock`, or `haydock = true` with `number_iterations` after the eigensolve | `bse.absorption_haydock` | ε₂(ω) by continued fraction on the TDA BSE (D + V − W), no eigenvectors |

Without `full_bse = false`, `the eigensolve` ignores `solver`, `block_size` and
`lanczos_n_reorth` without a message (`bse/bse_lanczos.py:118-121`).

## Dipoles and absorption

The dipole files hold velocity matrix elements. Absorption uses the position
form $d^\alpha_{cvk} = \langle ck|\hat v_\alpha|vk\rangle / (E_c - E_v)$
(`absorption_common.slice_dipole_to_bse_window`), with the energies of the
dipole file's own `band_energies`: DFT energies for `dipole.h5`, $E_{QP}$ for
`dipole_qsgw.h5`. `eqp_corrections` changes the transition energies of the Hamiltonian,
not the dipoles.

A `kgrid_fine` bundle carries its own dipoles (`bse_densify.densified_dipoles`)
and never slices a file (`absorption_common.window_dipoles` is the one source
of both). The deck WFN's velocity, unfolded in the loader's gauge, which is
the Galerkin fit's, enters the htransform's basis and is Fourier-interpolated
as $f(H)$ is (`bandstructure.orbital.interpolate_band_operator`), then
contracted with the same fine eigenvectors $U_q$ that build the densified ψ:

$$v_B(k) = C_k^{T}\, g\, v(k)\, g\, C_k^{*},\qquad
V_R = \frac{1}{N_k}\sum_k e^{2\pi i k\cdot R}\, v_B(k),\qquad
v(q) = G_q^{-1}\, U_q^\dagger\, \mathrm{Herm}\Big[\sum_R e^{-2\pi i q\cdot R} V_R\Big]\, U_q\, G_q^{-1} .$$

The weight $g = g(\varepsilon)$ (`bse_densify.velocity_weight`) is
$1 - e^{-x^2}$, $x = (E_{\rm cut} - \varepsilon)/w$, and zero above
$E_{\rm cut}$, the lowest energy of the first band above the fit window;
$w$ is half the distance from the BSE window's top to $E_{\rm cut}$, so
$g \ge 1 - e^{-4}$ on the window, and $G_q = g$ of the window block of the
file's $f(H)$ in the fine states. When the fit window's top band meets that
band the window projector jumps where they cross and the unweighted $v$ is
not smooth in k (Si 4³→8³, 36 of 100 bands: outer-shell image 26 % against
1.9 % for $f(H)$; window-sum velocity over exact $\Delta E$ off-grid 6.1 %
rms, weighted 2.7 %).
A fit window whose top stays below every unfitted band takes $g \equiv 1$,
bit for bit the unweighted route; on-grid both are exact.

The energies are the velocity's own. On the DFT path the fine states
diagonalize the file's $f(H)$, each state's energy is its Rayleigh quotient,
Newton-inverted, and

$$E_n(q) = f^{-1}\big(U_{qn}^\dagger\,[f(H)]_{\rm file}(q)\, U_{qn}\big),\qquad
d^\alpha_{cvq} = \frac{v^\alpha_{cv}(q)}{E_c(q) - E_v(q)} .$$

Under `eqp_corrections` the fine states diagonalize the interpolated QP $f(H)$ instead,
and the default dipoles are $v/\Delta E$ of their Rayleigh quotients, as
above. A strongly band-dependent correction makes the QP $f(H)$ a different
operator off the coarse grid, so the states mix the file's bands and those
dipoles are not their position: on CrI3 8×8→12×12 against the native 12×12
they lose 1.4 % of $\sum|d|^2$ (13 % below 1.6 eV), with $\sum|v|^2$ kept to
4e-4 and the lowest transition's quotient 60 meV high. The opt-in
`dipole_position_fix` (`qp_states`; the loader passes `eqp_file is not
None and velocity_position_fix`) then takes the exact position of the fine
states from the DFT $f(H)$ eigenbasis of the window, which matches CrI3 to
0.05 %. It is exact only when the DFT interpolant is the more accurate one:
on Si 4³→8³ (a near-scissor ladder) its window block mixes valence and
conduction by 3.8 % and the fix puts the window-sum $|d|^2$ 12.1 % rms
off-grid from the explicit 8³ dipoles against 2.4 % for the default. Without the fix the log prints the
same mixing receipt (max off-block mixing, $f(H)$ off-diagonal) so a run can
see when it needs it. With an entangled fit-window top the fix receives
the weighted velocity above, so it equals the pre-weighting fix bit for bit
only for isolated tops. TD-aGW bundles opt in with `[lorrax]
dipole_position_fix = true`. The fix, per q: $F = U^\dagger [f(H)]_{\rm file} U = W\,\mathrm{diag}(\varphi)\,W^\dagger$,
$\varepsilon = f^{-1}(\varphi)$,

$$r' = \frac{(W^\dagger v W)_{mn}}{i(\varepsilon_m - \varepsilon_n)},\qquad
\langle \tilde c|\hat r|\tilde v\rangle = (W r' W^\dagger)_{cv},\qquad
d_{cvq} = i\,\langle \tilde c|\hat r|\tilde v\rangle ,$$

exact for the c↔v pairs when $W$ does not mix the window's valence and
conduction rows (the log gives that off-block mixing, the window-sum
invariance and $\sum|d|^2$ after/before). Same-occupation pairs of $r'$ follow
`tdagw.td_system.position_from_velocity`'s rule (`mask`, default:
$1 - e^{-\Delta\varepsilon^2/2\,\mathrm{gap}^2}$ at 0.05 eV; `cut`; `cv`); they
enter $d$ only through that mixing. `full_window` also returns
`r_window`, the whole window's position in the same states (zero diagonal in
the velocity's eigenbasis; on the DFT path $v/(i\Delta E)$ with the rule), so
$d = i\,r_{\rm window}[c, v]$; `d_alpha_rq` keeps the previous dipoles. The
route label ends with the corrected form. `python -m bse.dump_window_dipoles`
writes a deck's window dipoles in its run's gauge, so a saved
`eigenvectors.h5` re-contracts on CPU (`exciton_dipole_projections`; the
stored dipoles of a 112-state CrI3 30×30 run reproduce to 1.7e-15).

Dipoles and ψ share $U_q$, so no map between two sets of states, and no
gauge fit, exists, and no fine NSCF is needed. A fine WFN's own `dipole.h5`
is in that loader's gauge, which differs from the densified ψ by the random
per-(k, band) phases of an independent diagonalization; it refuses
(`GATE velocity_wfn_fingerprint`). At every fine q that is a coarse k the
fine window states span the coarse states at the window's positions in
the fH's own energy order (QP under `eqp_corrections`, which can move a band across
a window edge: MoS₂ 9×9 at Γ), and the interpolated operators reproduce
the coarse ones in them (`GATE densified_dipole_on_grid`); the
log also gives each velocity image's outer-shell locality beside $f(H)$'s,
the signature of a velocity in another gauge. Their accuracy is the
densified ψ's: the guard bands and rank that converge ψ converge the dipoles.
On MoS₂ 9×9→12×12 (DFT fH) the window dipoles match the fine NSCF's own
`dipole.h5` to 6e-6 at shared k and to 0.8 % rms off them (Frobenius norm
per k, 2 % max), and the ε₂ of the SR/LR densified BSE is 2.0 % rms of
the direct 12×12 peak.

Under `qp_rotations_file` the dipoles come from the same `window_dipoles`, after
the same k-grid and WFN-stamp checks, and the velocity is the
interband-commutator QP velocity of
[QP velocity §7](../theory/qp-velocity.md#7-the-heads-that-use-the-velocity),
$v^{QP} \approx v^\mathrm{DFT} + [\Delta H, \mathcal W]$ with
$\mathcal W_{mn} = v_{mn}/(E_m - E_n)$ on cross-gap pairs and zero within each
class. Writing $v^\mathrm{DFT} = [E^\mathrm{DFT}, \mathcal W] + v^\mathrm{same}$, the
QP-basis dipole $d = v^{QP}_{c'v'}/(E'_{c'} - E'_{v'})$ is exactly

$$d_{c'v'} = \big(U_c^\dagger\,\mathcal W\,U_v\big)_{c'v'}
 + \frac{\big(U_c^\dagger\,v^\mathrm{same}\,U_v\big)_{c'v'}}{E'_{c'} - E'_{v'}}
\qquad(\texttt{bse\_qp\_rotation.qp\_window\_dipoles}).$$

At $U = I$ the second term vanishes and the first is the `eqp_corrections` dipole, bit for
bit. It omits the explicit $\partial_k\Delta H$ of $D_k\Delta H$, an error first
order in $\theta$; the pure DFT velocity $U^\dagger v^\mathrm{DFT}U/\Delta E'$ would
also drop $[\Delta H, \mathcal W]$, which for a diagonal $\Delta H$ is the term
that keeps $r^{QP} = r^\mathrm{DFT}$. The dipole must carry $-i[r, V_\mathrm{NL}]$.

`absorption_haydock` always uses the TDA BSE kernel, D + V − W, and needs at
least two devices. It seeds one Lanczos recursion per polarization; the seeds
(`--pol`, default `xyz`; each costs one matvec per step, and `x` alone is the
spectrum of a cubic crystal) are one trial block of the stack matvec. The
recursion (`haydock_coefficients`) reuses a payload's `W_R` when the eigensolve
kept it, which is how `bse_jax haydock = true` runs it in the same process;
the standalone route transforms `W_q` once with it donated, as the eigensolve
does. It does not reorthogonalize and
runs 200 steps by default. Claim 2848 measured the solver on CrI3 8×8×1 SOC
(with bare V as W) against a 500-state sum over states plus a
deflated CGS2 tail: at 200 steps both the plain and the CGS2 recursion agree
with it to ≤ 1e-5 of max ε₂; at 100 steps both are truncated, 0.2–0.5% of
max ε₂ off. The ε₂ prefactor is $16\pi^2/(V N_k n_\mathrm{spin} n_\mathrm{spinor})$ and the
continued fraction is scaled by $\lVert d\rVert^2$, as in BerkeleyGW
(`BSE/absh.f90`, `BSE/haydock.f90`).

Per-state dipoles: `bse_jax the eigensolve full_bse = false spin_singlet = true write_eigenvectors N dipole_file
dipole.h5` contracts each written eigenvector with the dipole
(`absorption_common.exciton_dipoles_distributed`, blocked over each rank's own
eigenvector tile) and stores $\langle 0|\hat r_\alpha|S\rangle$ as
`exciton_data/dipoles`, shape (1, N, 3, 2), in bohr. Flag rules:
[drivers](../drivers.md#bse-bsebse_jax).

In-plane helicity: `helicity_table` adds each written state's circular
strengths to `bse.out` (`absorption_common.inplane_helicity`, the one place
the convention lives). σ+ is the field $E(t) = E_0(\cos\omega t, \sin\omega t,
0) = \mathrm{Re}[E_0\varepsilon_+ e^{-i\omega t}]$ with
$\varepsilon_\pm = (\hat x \pm i\hat y)/\sqrt2$: it rotates x → y,
counter-clockwise seen from +z. Absorption into $|S\rangle$ goes as
$|\langle S|\varepsilon\cdot\hat r|0\rangle|^2 = |\varepsilon\cdot D^*|^2$ with
$D = \langle 0|\hat r|S\rangle$, so

$$I_\pm = \frac{|D_x \mp i D_y|^2}{2N_k},\qquad \eta = \frac{I_+ - I_-}{I_+ + I_-},$$

per unit cell in bohr² (an exciton normalized over $N_k$ points has
$|D|^2 \propto N_k$). The stored dipoles are $D$ times the global $-i$ of the
velocity form ($d = \langle c|\hat v|v\rangle/\Delta E = i\langle c|\hat r|v\rangle$)
on the native and the `kgrid_fine` routes alike, which no modulus sees; the
conjugated spelling $\langle S|\hat r|0\rangle$ would swap $I_+$ and $I_-$. On a
C3-symmetric toy the transitions into $p_\pm = (p_x \pm i p_y)/\sqrt2$ come out
pure σ± ($\langle L_z\rangle = \pm1$), and the two routes agree to 1e-14 at
shared k (`tests/test_bse_helicity_cpu.py`). Within a degenerate multiplet only
the multiplet sums of $I_\pm$ are basis independent. A TD-aGW σ± drive uses the
same field ([TD-aGW end to end](../how-to/td-agw-workflow.md#circular-drives)).

## Outputs

- `bse.out`: the run report of the `the eigensolve` route (`--report-file`;
  FEAST and KPM refuse that flag).
- `eigenvectors.h5` (`write_eigenvectors`), rank 0, BerkeleyGW layout
  (`src/bse/eigenvectors.h5.spec`) through `bse_window.write_eigenvectors_stream`:
  `exciton_data/eigenvalues` in eV; `exciton_data/eigenvectors`
  (1, N, N_k, n_c, n_v, 1, 2); full BSE adds `eigenvectors_coupling` (Y). The
  file follows BerkeleyGW's index conventions, which differ from LORRAX's
  internal ones: the valence axis is reversed (BerkeleyGW `iv = 1` is the
  highest valence band, LORRAX `v = 0` the lowest; the conduction axis is the
  same), eigenvalues are in eV (Ry internally), and the Fortran shape
  `[scalar, ns, nv, nc, nk, N, nq]` reads through h5py as
  `(nq, N, nk, nc, nv, ns, 2)`. BerkeleyGW's `vmtxel` flat index is
  `is + (iv−1 + (ic−1 + (ik−1)·nc)·nv)·nspin`, k slowest and v fastest.
  Compare eigenvectors between codes only through gauge-invariant
  quantities ($\sum|A_{cvk}|^2$ per state, manifold-summed
  $|\langle 0|\hat r|S\rangle|^2$, ε₂): each code fixes its own phase per
  $(c, v, k)$. The writer refuses to trim
  nonzero amplitude when the declared window is narrower than the solved one.
  Under `qp_rotations_file` the amplitudes are on QP states and `exciton_header`
  carries `transition_basis = qp_rotations` and `qp_rotations_file`; `nv`/`nc`
  then do not index `dipole.h5`.
- `absorption_haydock.h5` (ε₂, ε₁, JDOS, α, β, norms; with `--no-eps1` the
  ε₁ dataset is ones) and one `absorption_haydock_<pol>_eh.dat` per
  polarization, `<pol>` = `b1`, `b2`, `b3`.

## W_BSE: the ladder in GW screening

`screening_diagrams = w_bse` is a GW option: Σ uses the ladder-corrected
$W^c(z)=C(z-H)^{-1}B$, with density vertices at both ends and a static
direct rung in $H$ (`gw.screening_bse`, `bse.w_ladder`). The stage persists the RPA W(0), then
solves the resolvent one z at a time with ψ held 2-D at 1/P, by GMRES with a
recycled spectral deflation per (q, z) (`bse_feast.harvest_spectral_deflation`).
Keys, supported modes and refusals: [input reference](../input_reference.md).
This changes the screening in $\Sigma=iGW$; it does not add a matching
self-energy vertex or close the self-consistent Hedin equations.

The direct rung scans individual trials within each probe block. Its encoded
`T(mu,nu,s,s,k)` and FFT/decode scratch are reused, so this largest temporary
does not acquire the probe-block width. The trial vectors retain their
distributed conduction/valence layout; the ring encoders move band slices
without gathering the complete transition window on every rank. Exchange
and diagonal terms remain batched; their smaller trial/centroid buffers and
the returned trial block still grow with the probe count.

### Physical resonant and antiresonant branches

At stored $+q$ use the two physical transitions

$$
R=(m=c,k+q;n=v,k),\qquad A=(m=v,k+q;n=c,k).
$$

Let $I$ run over both branches, $\eta_I=f_{n_I}-f_{m_I}$ ($+1$ on R,
$-1$ on A), $D_{II}=\epsilon_{m_I}-\epsilon_{n_I}$, and
$F_I(\mu)=\sum_s\overline{\psi_{m_I}(\mu,s)}\psi_{n_I}(\mu,s)$.
The antiresonant diagonal is negative and its orbitals differ from R.
With uniform full-zone k weights the physical density operator is

$$
H=D+\eta(K^{\rm H}-K^{\rm d}),\qquad
K^{\rm H}=\frac{g}{N_k}FV_qF^\dagger,
\qquad B=\frac{1}{\sqrt{N_k}}\eta FV_q,\qquad
C=\frac{g}{\sqrt{N_k}}V_qF^\dagger.
$$

$g$ is the shared exchange spin weight (2 for a scalar singlet, 1 for
spinors). All four Hartree blocks use these same vertices. For any two
physical pairs the positive direct matrix element is

$$
K^{\rm d}_{IJ}=\frac{1}{N_k}\sum_{\mu\nu ss'}
\overline{\psi_{m_I}(\mu,s)}\psi_{m_J}(\mu,s)
W^{\rm RPA}_{\mu\nu}(k_I-k_J;0)
\psi_{n_I}(\nu,s')\overline{\psi_{n_J}(\nu,s')}.
$$

This equation supplies R–R, R–A, A–R and A–A. The direct rung has no extra
singlet factor. Both shifted electron and shifted hole orbitals are loaded
at physical $k+q$; `build_finite_q_data` uses the existing roll-index owner
with argument $-q$ to obtain that gather, while keeping the tile $V_q$.
The construction needs neither real orbitals nor a band sewing matrix at
Kramers points. Arbitrary complex phases and rotations of degenerate
one-particle states must transform source, operator and readout together.

The rung uses the same k-convolution router and band-projection owner as
the optical/exciton kernels. One summed ket field and one k convolution
serve both physical bra decodes; the anti band owners move through the
existing transpose-peer collective. This shares the contraction machinery,
without identifying the two physical vertices or substituting the optical
solver's real-linear coordinate convention.

The unchanged folded RPA body remains an independent control. Setting
$K^{\rm d}=0$ in the explicit-branch response reproduces the Dyson RPA
response at every q. The former hybrid of folded density vertices and
optical direct blocks is not a valid ladder operator in complex orbital
gauges. Its results are historical, not a convention to reuse. The backend
identity is `physical-ra-v1`; response banks from the former convention
must be rebuilt.

### Samples, slopes and moments

Both branches remain present through the shifted solves and exact actions:

$$
W^c(z)=C(z-H)^{-1}B,\qquad
\partial_z W^c(z)=-C(z-H)^{-2}B,\qquad
\mathcal M_j=\tfrac12 CH^jB.
$$

For $s=z^2$ and $z\ne0$, the tangential slope is
$\partial_sW^c=(2z)^{-1}\partial_zW^c$. With measured time reversal,
$W^c(z)=W^c(-z)$ and $\mathcal M_0=\mathcal M_2=0$, so
$W^c(z)=2\mathcal M_1/z^2+2\mathcal M_3/z^4+\cdots$.
The bank gates the reflected solves and forbidden moments before the
even-$s$ fit. It does not average a noneven response into this model or
replace ladder slopes or moments with RPA values.

The tangential adapter samples this operator at the ordinary recipe's
complex supports, supplies its exact slopes and moments to the existing
constructor, and passes the resulting shared real poles to the unchanged
Σ executor. Canonical centroid outputs are packed through the run's one
basis owner before fitting. The identity binds the branch convention,
static-rung digest, head policy and solver tolerance/iteration cap; an RPA bank or another rung cannot
authenticate as this ladder. The elementwise `mpa` parent handoff remains
refused. Supported modes and parse refusals are owned by the
[input reference](../input_reference.md).

Hermite construction consumes values and slopes, so this path has a
stricter input-solve budget than the production GN-PPM ladder. Its body
and fresh Γ head use the same true-residual tolerance:
`0.1 * min(bank_rule_tolerance, abs(normalized_gram_validity))`, giving
`1e-9` for the production recipe and `1e-8` for the relaxed recipe, with
the existing 300-iteration cap. Every returned residual must meet that
requested tolerance; a solve at the cap is refused. The accuracy policy,
tolerance and cap authenticate the bank/model identity, so an older
looser bank cannot be reused. This allocates one decade below the existing
certificate budgets; it does not bound Gram error amplification. The
constructor still applies its unchanged positivity gate without PSD repair.
The GN-PPM ladder retains its `1e-6` tolerance.

### Macroscopic head: the same operator, restored once

The headless Γ operator already includes microscopic Hartree local fields
and the direct rung. For the velocity dipole $d=v_{cv}/\Delta=i r_{cv}$ its
matched source is $[d;\bar d]$ and readout $[d^\dagger,-d^T]$:

$$
\Xi_{ab}(z)=\frac{2}{\Omega N_k n_{\rm spin}n_{\rm spinor}}
\big[d^\dagger,-d^T\big]_a(z-H_\Gamma)^{-1}
\begin{bmatrix}d\\\bar d\end{bmatrix}_b.
$$

`head_resolvent.project_head_response` owns this readout. With both Hartree
and direct kernels disabled, the free-pair limit agrees with the direct
velocity tensor in the same units and band window. Properly matched folded
RPA coordinates remain valid;
mixing their branch sum with the physical operator does not.

Only the longitudinal symmetric part of $\Xi$ enters the scalar head.
`response_kind=micro_reducible` tells the shared head provider that its
microscopic local fields are already present. It restores the omitted
macroscopic Coulomb channel once, then performs the same 3D or slab
mini-BZ average. Applying the RPA $S+YWZ$ Schur fold again would double
count microscopic screening. No Γ head is also inserted into the body
resolvent.

The full-frequency scalar head uses fresh Γ solves at its own fit and
imaginary-moment nodes, including static and reflected nodes. Evenness,
imaginary-axis sign and solve residuals are checked before the existing
scalar fitter is used. Its convention is
$W^c_{\rm head}(z)=\sum_p2\Omega_p B_p/(z^2-\Omega_p^2)$;
the head's inverse-frequency moment $M_1^{\rm head}=\sum_pB_p/\Omega_p^2$
is distinct from the body's high-frequency $\mathcal M_1$ above.

### Self-consistency is a separate, refused extension

The implemented ladder route is single-shot. A self-consistent map would
need its own current energies, occupied/empty pair window, wavefunctions,
velocity vertices and freshly screened static rung
$W^{\rm RPA}[G_n](0)$, followed by a new ladder bank, head and poles before
evaluating that map's Σ. The resulting map also needs an identity binding
all of these objects. Reusing the initial rung while updating $G$ defines
a frozen-kernel approximation, not that self-consistent ladder map.
`w_bse_self_consistency_unimplemented` therefore remains a refusal; shared
Σ/pole infrastructure by itself does not certify this extension.

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
| `GATE dipole_basis` | `file_io.dipole.require_dipole_basis` | a dipole whose `basis` stamp (`dft` or `qp`; an unstamped file reads `dft`) differs from the basis of the WFN's ψ, e.g. `dipole_qsgw.h5` with a DFT WFN |
| `GATE dipole_k_grid` | `file_io.dipole.require_dipole_k_grid` | a dipole whose k list is not the bundle's (another grid's file) |
| `GATE velocity_wfn_fingerprint` | `file_io.dipole.require_dipole_wfn_fingerprint` (from `velocity_stamp` and `absorption_common.window_dipoles`) | a `--dipole` that is not the deck WFN's: under `kgrid_fine` e.g. the fine grid's own `dipole.h5` (read before the htransform fit); on an undensified run another WFN's file with the same k list, or a file with no `prov_wfn_sha256` |
| `GATE densified_dipoles` | `absorption_common.window_dipoles` | a `kgrid_fine` bundle whose dipoles were not built from `dipole_file` (loaded without `velocity_file`) |
| `GATE densified_velocity_bands` | `bse_densify.densified_dipoles` | a velocity file with fewer bands than the htransform window |
| `GATE densified_velocity_weight` | `bse_densify.densified_dipoles` | an unfitted band reaching the BSE window's top, or a velocity weight below 1e-3 of its maximum in the fine window |
| `GATE dipole_band_window` | `file_io.dipole.require_dipole_bands` (`absorption_common.window_dipoles` and `slice_dipole_to_bse_window`, `gw.qsgw_head.head_source`) | a `dipole.h5` whose stored bands (`band_min`/`band_max`, else `[0, nb)`) do not cover the window from band 0 |
| `GATE densified_position_gap` | `bse_densify.densified_dipoles` | `eqp_corrections` fine states whose DFT eigenbasis puts a conduction row at or below a valence row |
| `GATE densified_pair_rule` | `bse_densify.densified_dipoles` | a pair rule other than `mask`, `cut`, `cv` |
| `GATE densified_dipole_gap` | the same | a non-positive $E_c - E_v$ in the velocity's own energies |
| `GATE densified_dipole_on_grid` | the same | the on-grid identities fail: a fine q list, band window or contraction that is not the htransform's |
| `BseWindowOutsideZetaTrainingError` | `bse_window.assert_bse_window_in_zeta_training` | a window band outside the ζ fit legs |
| `GATE bse_qp_rotation_block` | `bse_qp_rotation.load_qp_rotation` | a QP block $[q_0, q_1)$ outside the ζ fit legs or the stored bands: a sliced eigenvector matrix is not unitary |
| `GATE bse_qp_window_outside_block` | `bse_qp_rotation.qp_window_from_artifact` | a window, or its occupied boundary, outside the QP block |
| `GATE bse_qp_k_order` | `bse_qp_rotation.load_qp_rotation` | a symmetry-service full-BZ list that is not the BSE's C-order grid |
| `GATE bse_qp_star_tables` | the same | a wedge-stored artifact whose unfold tables are not the WFN's |
| `GATE bse_qp_star_gauge` | the same; `bse_qp_rotation.certify_star_gauge` | an artifact $H_{QP}(k)$, or a restart centroid band matrix of 1 or $\sigma_z$, that misses its (anti)unitary star image by more than $10^{-10}$ |
| `GATE bse_qp_dipole_velocity` | `absorption_common.window_dipoles` | `qp_rotations_file dipole_file` with a momentum-only (`skip_vnl`) dipole |
| `GATE bse_qp_rotations_k_grid` | `bse_loading.load_bse_data_from_restart_sharded` | `qp_rotations_file` with `kgrid_fine` |
| `GATE eqp_star_spread` | `restart_bundle.apply_eqp_corrections` (`_eqp_on_full_bz`) | an `eqp_corrections` file whose QP shifts $E_{QP} - E_{DFT}$, averaged over each DFT multiplet, differ across a symmetry star by more than 1 meV (`EQP_STAR_SPREAD_TOL_EV`); the message names the band, its multiplet and the two k. Per-band values are applied as given |
| `GATE eqp_full_zone_count`, eqp k coverage | the same | an `eqp_corrections` file with more blocks than the full BZ; a full-zone-sized file in which some full-BZ k matches no block or two (crystal coordinates mod G to 1e-6); a file that is neither the wedge nor the full zone |
| two QP descriptions | `bse_jax.parse_args`, `file_io.qp_wfn.refuse_conflicting_qp_state_sources` | `eqp_corrections` with `qp_rotations_file`; a rotation on a QP WFN deck or on a restart from another WFN |
| band-window degeneracy | `common.band_degeneracy.resolve_band_window` | a window edge inside a multiplet under `band_degeneracy = strict` |
| route-ignored flag | `bse_jax.parse_args` | a flag set on a route that does not read it; a retired flag |
| non-TDA reciprocity preflight | `bse_nontda.check_restart_reciprocity` | `the eigensolve` without `full_bse = false`, with W on: a restart whose W fails the q ↔ −q reciprocity check, before the dense build |
| non-TDA dense size | `bse_nontda` (`_DENSE_N_MAX`) | `the eigensolve` without `full_bse = false` at N = n_c,pad · n_v,pad · N_k > 4096 |
| `GATE bse_w_block_operand` | `bse_stack_matvec.plan_w_blocks` | an 8 × 8 (μ, ν) W-term block whose outer-load handler call still has an operand of $2^{31}$ elements or more (a k grid near $8\cdot 10^6$ points at $n_s$ = 2) |
| Haydock device count | `absorption_haydock.run_haydock` | fewer than two devices |

## Limits

- A four-current bundle is read only with a stored W0: the BSE direct term has
  no packed CC/CT/TC/TT handoff (`gw/static_screening.py:86`). The stored W0
  is the charge sector: the direct term screens with $W_{CC}(0)$ and the
  exchange term uses the charge V; the CT/TC/TT blocks, screened or bare,
  never enter the BSE kernel. On `bispinor_gw = bare_transverse` it is
  V + W_c(0) of the charge store; on `bispinor_gw = full_tangential` it is
  V + W_c,CC(0) of the CC sector (`gw/mpa/sector_sigma.py:sector_static_wc`,
  the sector Σ synthesis at the ω = 0 coefficient; the CC block of
  W_∞ − V is zero because the Ward contact is TT-only). An SC run keeps its
  resident CC model past Σ until the final-map persist.
- The BSE reads ψ from the restart, which holds the deck WFN's states. `eqp_corrections`
  replaces energies only (`bse/bse_window.py:569`); `qp_rotations_file` rotates ψ
  into the QP basis on the native grid and refuses `kgrid_fine`, so a densified
  BSE with QP ψ needs a GW restart generated from `WFN_qp.h5`, with
  `dipole_qsgw.h5` as its dipole. A `WFN_qp.h5` written before 2026-09-30 from a WFN that stores both
  k and −k has broken rows; regenerate it
  ([self-consistency §8](../self_consistency.md#8-seeding-restart-and-outputs)).
- Full BSE through `the eigensolve` is dense and stops at N = 4096
  (`bse/bse_nontda.py:95`); the matrix-free solver is not reachable from the
  CLI.
- `compute_mode = mpa` with `screening_diagrams = w_bse` refuses at setup
  (`GATE parent_screening_diagrams`, `file_io/restart_bundle.py:4369`).
