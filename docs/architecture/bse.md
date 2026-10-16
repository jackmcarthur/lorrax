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
| the deck (`-i`) | the GW deck | WFN path, centroid files, `nval`/`ncond`/`nband`, head settings, `bse_k_grid` |
| `isdf_tensors_<N_mu>.h5`, in the run directory or its `tmp/` | `gw.gw_jax` with `write_restart_tensors = true` (the default) | parent ψ faces at the centroids, `enk_full` (Ry), `V_qmunu`, `W0_qmunu` and its `W0_ready` flag, the head vector `G0_mu_nu`, `vhead`/`whead`; on a static-Dyson W0, the dressed long-range channels `W0_lr` |
| `WFN.h5` and the centroid table named in the deck | QE → pw2bgw; `centroid.kmeans_cli` | authentication of the bundle, and the static-screening rebuild |
| `eqp1.dat` (`--eqp`, optional) | `gw.gw_jax` | diagonal QP energies on the wedge, unfolded through the symmetry service |
| `qp_wfn_rotations.h5` (`--qp-rotations`, optional; exclusive with `--eqp`) | `gw.gw_jax` (`file_io.qp_wfn.write_qp_rotations_h5`) | the GW run's $H_{QP} = U\,\mathrm{diag}(E_{QP})\,U^\dagger$ on its Σ window, unfolded to the full BZ by `restart_bundle.read_qp_rotations_artifact`: the BSE runs in its eigenbasis ([QP-rotated basis](#the-qp-rotated-basis)) |
| `dipole.h5` (`--dipole`) | `psp.get_dipole_mtxels` | velocity matrix elements and `band_energies` for absorption and per-state dipoles; under `bse_k_grid`, the deck (coarse) WFN's file, which the densifier carries to the fine grid |
| `dipole_qsgw.h5` (`--dipole`) | a QSGW run with a velocity head ([QSGW dipoles](../self_consistency.md#qsgw-dipoles)) | the same layout: $U^\dagger v\,U$ with the head's velocity $v$ (`qsgw_head.qp_velocity`) and `band_energies` = $E_{QP}$; stamped `basis = qp` and, when the run writes `WFN_qp.h5`, bound to that WFN's fingerprint. It pairs only with a restart built from `WFN_qp.h5` |

Exactly one `isdf_tensors_*.h5` may sit in the run directory and its `tmp/`
(`file_io.restart_bundle._find_restart_file`). Every read of the parent faces
authenticates the bundle against the deck (`restart_bundle.unfold_parent_faces`):
the WFN fingerprint, the centroid content hash, the stored band window and
$N_\mu$ (`assert_restart_window_matches`), and the parent rows.

**Band window.** The occupied-band count $n_{occ}$ is `--n-occ` (Lanczos
route), else the deck WFN's `ifmax` (`WfnLoader.nelec`,
`bse_window.resolve_n_occ`). The window is
$[n_{occ} - n_v,\; n_{occ} + n_c)$, clamped to the stored bands. An edge inside
a multiplet follows `--band-degeneracy` (`common.band_degeneracy`). Every band
of the window must lie in both legs of the ζ fit, because the direct term reads
$v\times v'$ and $c\times c'$ pair densities
(`bse_window.assert_bse_window_in_zeta_training`). $n_c$ and $n_v$ are padded to
the mesh with zero ψ and a signed ε sentinel (`PAD_EPS_GUARD_RY`), so pad
states decouple.

### The QP-rotated basis

`--qp-rotations` (`bse.bse_qp_rotation`, native grid only) solves the BSE in the
eigenbasis of the GW run's effective Hamiltonian on its Σ window $[q_0, q_1)$,
$H_{QP}(k) = U(k)\,\mathrm{diag}(E_{QP}(k))\,U(k)^\dagger$ with
$U_{mn}(k) = \langle m_\mathrm{DFT}k | n_\mathrm{QP}k\rangle$. The centroid states are

$$\psi'_{nk}(\mu) = \sum_{m=q_0}^{q_1-1} \psi_{mk}(\mu)\,U_{mn}(k),$$

over the complete block, which must lie in both ζ-fit legs; ε are $E_{QP}$; V and
W0 are reused, because they live in the centroid basis. `--n-val`/`--n-cond`
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
diagonal control of a rotated run is `--eqp eqp0.dat`, and the rotated minus
that control is the off-diagonal $\Sigma - v_{xc}$ alone.

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
| bispinor `mpa`, shared pole, `bispinor_gw = bare_transverse`, one-shot or self-consistent | $V + W^c(0)$ of the four-component charge store on the charge V, the same evaluator (charge sector only) |
| bispinor `mpa`, `bispinor_gw = full_shared_pole` | $V + W^c_{CC}(0)$ of the CC sector (`gw.mpa.sector_sigma.sector_static_wc`) |
| `x_only`; `mpa` with `sigma_w_model = mpa` | nothing |
| shared pole on a metal, or `head_correction = full` without an ω = 0 head sample | nothing; the log says why |

W0 is stored on the q parents of `V_qmunu`, with their unfold tables, whenever
the run's q axis reduces (`restart_q_storage = auto` resolves `ibz`;
[symmetry register](symmetry_register.md)); a deck whose q axis does not reduce
stores every q. GW never forms a full-q W0 for the file: the shared-pole
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
`bse_jax` applies the RPA kernel, D + V, unless `--bse` is given; `--tda`
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

`bse_k_grid` densifies the bundle before any solve (`bse.bse_densify`). ψ and
ε go through one htransform $f(H)$; with `--eqp` its energies are the
quasiparticle ones, so the fine ε are QP. W's body, with its Γ head removed, is
zero-padded in R, and the head is re-attached analytically at each fine q
(`w_head_densify = c1`). `w_head_densify = srlr` instead removes a fitted
model of W's whole long-range part (the screened head and, in a slab, its G_z
channels) before the zero-pad and adds it back at every fine q, from the
dressed channels GW stored in `W0_lr`
([SR/LR densification](../theory/w-srlr-densification.md)). The
q = 0 exchange tile is k-grid invariant and is carried through unchanged,
unless `head_minibz_average = true`: then it is rebuilt through
`bse.vq_interp` with the fine grid's mini-BZ head
([LT head](../theory/lt-exchange-head.md)), and the W head's Γ-cell reference
uses the analytic sphere. Without `bse_k_grid` the optical BSE does not read
`head_minibz_average`. With `--dipole`, the velocity goes through the same
htransform as ψ ([dipoles](#dipoles-and-absorption)).

## The matvec

`bse_stack_matvec.build_bse_stack_matvec` is the TDA matvec: Lanczos,
Davidson and thick-restart Lanczos, FEAST and KPM under `--tda`, the
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

The kernels, their refusals and the `LORRAX_BSE_OUTER_KSUM` A/B switch are in
[the FFI layer](ffi_layout.md#kernel-operations). The decode keeps the
(t, μ)-first order on every route.

Full (non-TDA) BSE runs on `bse_ring_comm.build_bse_ring_matvec_full`: it is
the operator of FEAST and KPM without `--tda`, and `bse_nontda` builds the
dense $(A, B)$ from it (N ≤ 4096). `build_bse_stack_pair_matvec`, the
real-linear applier $\mathrm{pair}(X, s) = AX + s\,B\bar X$ (Shao–da
Jornada–Yang, Algorithm 4), serves only `bse_nontda`'s matrix-free solver,
which the CLI does not select. The ring matvec is also the screening operator
of `bse.w_ladder` (`w_bse`), `bse.bse_w_exact` and `bse.w_omega_chain`.

Measured (claim 2874): CrI3
8×8×1 SOC, 8v × 14c, P4 A100-40GB. One Haydock step (D + V − W on three
trials) takes 13.2–13.4 ms; 100 steps take 1.28 s; the solve peak is 2159 MiB
per rank. The fused kernel runs at 2.27× the flop floor.

## Solvers

| route | driver flags | module | use |
|---|---|---|---|
| FEAST | no `--lanczos`; `--tda` or not | `bse.bse_feast` | contour eigensolve in KPM-sized windows |
| Lanczos, block Lanczos | `--lanczos --tda` | `bse_lanczos.solve_bse_sharded` | spectrum shape; CGS2 reorthogonalization, full window by default |
| Davidson | `--lanczos --tda --solver davidson` | `solvers.davidson.davidson`, preconditioner and start subspace from `bse.bse_davidson_helpers` | per-state convergence |
| thick-restart Lanczos | `--lanczos --tda --solver trlan` | `solvers.thick_restart_lanczos` | Krylov memory capped at `--trlan-m-max` |
| full BSE | `--lanczos` without `--tda` | `bse.bse_nontda`, dense build | structure-preserving non-TDA solve, N ≤ 4096 |
| density of states | `--kpm-dos`; `--tda` or not | `bse.bse_kpm` | KPM Chebyshev moments |
| absorption | `python -m bse.absorption_haydock` | `bse.absorption_haydock` | ε₂(ω) by continued fraction on the TDA BSE (D + V − W), no eigenvectors |

Without `--tda`, `--lanczos` ignores `--solver`, `--block-size` and
`--n-reorth` without a message (`bse/bse_lanczos.py:118-121`).

## Dipoles and absorption

The dipole files hold velocity matrix elements. Absorption uses the position
form $d^\alpha_{cvk} = \langle ck|\hat v_\alpha|vk\rangle / (E_c - E_v)$
(`absorption_common.slice_dipole_to_bse_window`), with the energies of the
dipole file's own `band_energies`: DFT energies for `dipole.h5`, $E_{QP}$ for
`dipole_qsgw.h5`. `--eqp` changes the transition energies of the Hamiltonian,
not the dipoles.

A `bse_k_grid` bundle carries its own dipoles (`bse_densify.densified_dipoles`)
and never slices a file (`absorption_common.window_dipoles` is the one source
of both). The deck WFN's velocity, unfolded in the loader's gauge, which is
the Galerkin fit's, enters the htransform's basis and is Fourier-interpolated
as $f(H)$ is (`bandstructure.orbital.interpolate_band_operator`), then
contracted with the same fine eigenvectors $U_q$ that build the densified ψ:

$$v_B(k) = C_k^{T}\, v(k)\, C_k^{*},\qquad
V_R = \frac{1}{N_k}\sum_k e^{2\pi i k\cdot R}\, v_B(k),\qquad
v(q) = U_q^\dagger\, \mathrm{Herm}\Big[\sum_R e^{-2\pi i q\cdot R} V_R\Big]\, U_q .$$

The energies are the velocity's own. On the DFT path the fine states
diagonalize the file's $f(H)$, each state's energy is its Rayleigh quotient,
Newton-inverted, and

$$E_n(q) = f^{-1}\big(U_{qn}^\dagger\,[f(H)]_{\rm file}(q)\, U_{qn}\big),\qquad
d^\alpha_{cvq} = \frac{v^\alpha_{cv}(q)}{E_c(q) - E_v(q)} .$$

Under `--eqp` the fine states diagonalize the interpolated QP $f(H)$ instead.
A band-dependent correction makes that a different operator off the coarse
grid, so the states mix the file's bands and $v/\Delta E$ of their Rayleigh
quotients is not their position: on CrI3 8×8→12×12 against the native 12×12
it loses 1.4 % of $\sum|d|^2$ (13 % below 1.6 eV), with $\sum|v|^2$ kept to
4e-4 and the lowest transition's quotient 60 meV high. The dipoles are then
the exact position of the fine states (`qp_states`, the loader passes
`eqp_file is not None`): per q, $F = U^\dagger [f(H)]_{\rm file} U = W\,\mathrm{diag}(\varphi)\,W^\dagger$,
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
the fH's own energy order (QP under `--eqp`, which can move a band across
a window edge: MoS₂ 9×9 at Γ), and the interpolated operators reproduce
the coarse ones in them (`GATE densified_dipole_on_grid`); the
log also gives each velocity image's outer-shell locality beside $f(H)$'s,
the signature of a velocity in another gauge. Their accuracy is the
densified ψ's: the guard bands and rank that converge ψ converge the dipoles.
On MoS₂ 9×9→12×12 (DFT fH) the window dipoles match the fine NSCF's own
`dipole.h5` to 6e-6 at shared k and to 0.8 % rms off them (Frobenius norm
per k, 2 % max), and the ε₂ of the SR/LR densified BSE is 2.0 % rms of
the direct 12×12 peak.

Under `--qp-rotations` the dipoles come from the same `window_dipoles`, after
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

At $U = I$ the second term vanishes and the first is the `--eqp` dipole, bit for
bit. It omits the explicit $\partial_k\Delta H$ of $D_k\Delta H$, an error first
order in $\theta$; the pure DFT velocity $U^\dagger v^\mathrm{DFT}U/\Delta E'$ would
also drop $[\Delta H, \mathcal W]$, which for a diagonal $\Delta H$ is the term
that keeps $r^{QP} = r^\mathrm{DFT}$. The dipole must carry $-i[r, V_\mathrm{NL}]$.

`absorption_haydock` always uses the TDA BSE kernel, D + V − W, and needs at
least two devices. It seeds one Lanczos recursion per polarization; the three
seeds are one trial block of the stack matvec. It does not reorthogonalize and
runs 200 steps by default. Claim 2848 measured the solver on CrI3 8×8×1 SOC
(with bare V as W) against a 500-state sum over states plus a
deflated CGS2 tail: at 200 steps both the plain and the CGS2 recursion agree
with it to ≤ 1e-5 of max ε₂; at 100 steps both are truncated, 0.2–0.5% of
max ε₂ off. The ε₂ prefactor is $16\pi^2/(V N_k n_\mathrm{spin} n_\mathrm{spinor})$ and the
continued fraction is scaled by $\lVert d\rVert^2$, as in BerkeleyGW
(`BSE/absh.f90`, `BSE/haydock.f90`).

Per-state dipoles: `bse_jax --lanczos --tda --bse --write-eigs N --dipole
dipole.h5` contracts each written eigenvector with the dipole
(`absorption_common.exciton_dipoles_distributed`, blocked over each rank's own
eigenvector tile) and stores $\langle 0|\hat r_\alpha|S\rangle$ as
`exciton_data/dipoles`, shape (1, N, 3, 2), in bohr. Flag rules:
[drivers](../drivers.md#bse-bsebse_jax).

In-plane helicity: `--helicity-table` adds each written state's circular
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
on the native and the `bse_k_grid` routes alike, which no modulus sees; the
conjugated spelling $\langle S|\hat r|0\rangle$ would swap $I_+$ and $I_-$. On a
C3-symmetric toy the transitions into $p_\pm = (p_x \pm i p_y)/\sqrt2$ come out
pure σ± ($\langle L_z\rangle = \pm1$), and the two routes agree to 1e-14 at
shared k (`tests/test_bse_helicity_cpu.py`).

## Outputs

- `bse.out`: the run report of the `--lanczos` route (`--report-file`;
  FEAST and KPM refuse that flag).
- `eigenvectors.h5` (`--write-eigs`), rank 0, BerkeleyGW layout
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
  Under `--qp-rotations` the amplitudes are on QP states and `exciton_header`
  carries `transition_basis = qp_rotations` and `qp_rotations_file`; `nv`/`nc`
  then do not index `dipole.h5`.
- `absorption_haydock.h5` (ε₂, ε₁, JDOS, α, β, norms; with `--no-eps1` the
  ε₁ dataset is ones) and one `absorption_haydock_<pol>_eh.dat` per
  polarization, `<pol>` = `b1`, `b2`, `b3`.

## W_BSE: the ladder in GW screening

`screening_diagrams = w_bse` is a GW option: Σ uses the ladder-corrected
$W(z) - v = v(z - H)^{-1}v$, with the BSE Hamiltonian's static direct rung in
$H$ (`gw.screening_bse`, `bse.w_ladder`). The stage persists the RPA W(0), then
solves the resolvent one z at a time with ψ held 2-D at 1/P, by GMRES with a
recycled spectral deflation per (q, z) (`bse_feast.harvest_spectral_deflation`).
Keys, supported modes and refusals: [input reference](../input_reference.md).

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
| `GATE velocity_wfn_fingerprint` | `file_io.dipole.require_dipole_wfn_fingerprint` (from `velocity_stamp` and `absorption_common.window_dipoles`) | a `--dipole` that is not the deck WFN's: under `bse_k_grid` e.g. the fine grid's own `dipole.h5` (read before the htransform fit); on an undensified run another WFN's file with the same k list, or a file with no `prov_wfn_sha256` |
| `GATE densified_dipoles` | `absorption_common.window_dipoles` | a `bse_k_grid` bundle whose dipoles were not built from `--dipole` (loaded without `velocity_file`) |
| `GATE densified_velocity_bands` | `bse_densify.densified_dipoles` | a velocity file with fewer bands than the htransform window |
| `GATE densified_position_gap` | `bse_densify.densified_dipoles` | `--eqp` fine states whose DFT eigenbasis puts a conduction row at or below a valence row |
| `GATE densified_pair_rule` | `bse_densify.densified_dipoles` | a pair rule other than `mask`, `cut`, `cv` |
| `GATE densified_dipole_gap` | the same | a non-positive $E_c - E_v$ in the velocity's own energies |
| `GATE densified_dipole_on_grid` | the same | the on-grid identities fail: a fine q list, band window or contraction that is not the htransform's |
| `BseWindowOutsideZetaTrainingError` | `bse_window.assert_bse_window_in_zeta_training` | a window band outside the ζ fit legs |
| `GATE bse_qp_rotation_block` | `bse_qp_rotation.load_qp_rotation` | a QP block $[q_0, q_1)$ outside the ζ fit legs or the stored bands: a sliced eigenvector matrix is not unitary |
| `GATE bse_qp_window_outside_block` | `bse_qp_rotation.qp_window_from_artifact` | a window, or its occupied boundary, outside the QP block |
| `GATE bse_qp_k_order` | `bse_qp_rotation.load_qp_rotation` | a symmetry-service full-BZ list that is not the BSE's C-order grid |
| `GATE bse_qp_star_tables` | the same | a wedge-stored artifact whose unfold tables are not the WFN's |
| `GATE bse_qp_star_gauge` | the same; `bse_qp_rotation.certify_star_gauge` | an artifact $H_{QP}(k)$, or a restart centroid band matrix of 1 or $\sigma_z$, that misses its (anti)unitary star image by more than $10^{-10}$ |
| `GATE bse_qp_dipole_velocity` | `absorption_common.window_dipoles` | `--qp-rotations --dipole` with a momentum-only (`skip_vnl`) dipole |
| `GATE bse_qp_rotations_k_grid` | `bse_loading.load_bse_data_from_restart_sharded` | `--qp-rotations` with `bse_k_grid` |
| two QP descriptions | `bse_jax.parse_args`, `file_io.qp_wfn.refuse_conflicting_qp_state_sources` | `--eqp` with `--qp-rotations`; a rotation on a QP WFN deck or on a restart from another WFN |
| band-window degeneracy | `common.band_degeneracy.resolve_band_window` | a window edge inside a multiplet under `--band-degeneracy strict` |
| route-ignored flag | `bse_jax.parse_args` | a flag set on a route that does not read it; a retired flag |
| non-TDA reciprocity preflight | `bse_nontda.check_restart_reciprocity` | `--lanczos` without `--tda`, with W on: a restart whose W fails the q ↔ −q reciprocity check, before the dense build |
| non-TDA dense size | `bse_nontda` (`_DENSE_N_MAX`) | `--lanczos` without `--tda` at N = n_c,pad · n_v,pad · N_k > 4096 |
| Haydock device count | `absorption_haydock.run_haydock` | fewer than two devices |

## Limits

- A four-current bundle is read only with a stored W0: the BSE direct term has
  no packed CC/CT/TC/TT handoff (`gw/static_screening.py:86`). The stored W0
  is the charge sector: the direct term screens with $W_{CC}(0)$ and the
  exchange term uses the charge V; the CT/TC/TT blocks, screened or bare,
  never enter the BSE kernel. On `bispinor_gw = bare_transverse` it is
  V + W_c(0) of the charge store; on `bispinor_gw = full_shared_pole` it is
  V + W_c,CC(0) of the CC sector (`gw/mpa/sector_sigma.py:sector_static_wc`,
  the sector Σ synthesis at the ω = 0 coefficient; the CC block of
  W_∞ − V is zero because the Ward contact is TT-only). An SC run keeps its
  resident CC model past Σ until the final-map persist.
- The BSE reads ψ from the restart, which holds the deck WFN's states. `--eqp`
  replaces energies only (`bse/bse_window.py:569`); `--qp-rotations` rotates ψ
  into the QP basis on the native grid and refuses `bse_k_grid`, so a densified
  BSE with QP ψ needs a GW restart generated from `WFN_qp.h5`, with
  `dipole_qsgw.h5` as its dipole. A `WFN_qp.h5` written before 2026-09-30 from a WFN that stores both
  k and −k has broken rows; regenerate it
  ([self-consistency §8](../self_consistency.md#8-seeding-restart-and-outputs)).
- Full BSE through `--lanczos` is dense and stops at N = 4096
  (`bse/bse_nontda.py:95`); the matrix-free solver is not reachable from the
  CLI.
- `compute_mode = mpa` with `screening_diagrams = w_bse` refuses at setup
  (`GATE parent_screening_diagrams`, `file_io/restart_bundle.py:4369`).
- The `w_bse` ladder refuses a degenerate TRIM block whose Kramers misclosure
  exceeds 1e-8 of its scale (`GATE trs_gauge_block_not_theta_closed`,
  `bse/bse_w_exact.py:538`).
