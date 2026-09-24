# ISDF-BSE: Hamiltonian, matvec and data contract

The BSE is solved matrix-free: $H$ is applied to trial vectors through the ISDF
factorization and never formed. How to run the drivers is in
[`docs/drivers.md`](../../../docs/drivers.md) (§bse, §exciton bands); module
status in [`STATUS.md`](../STATUS.md); BerkeleyGW comparison conventions in
[`BGW_COMPARE.md`](../BGW_COMPARE.md); finite-Q traps in
[`EXCITON_BANDS.md`](../EXCITON_BANDS.md). The other files in this folder are
design notes and paper excerpts, not contracts.

## Hamiltonian

In the transition basis $|vk \to ck\rangle$, with $X$ the resonant and $Y$ the
antiresonant amplitudes,

$$\begin{pmatrix} A & B \\ -B^* & -A^* \end{pmatrix}\begin{pmatrix} X \\ Y \end{pmatrix} = \Omega \begin{pmatrix} X \\ Y \end{pmatrix},\qquad
A = D + w_x V - W,\qquad B = w_x V^B - W^B,$$

$D_{cvk} = \varepsilon_{ck} - \varepsilon_{vk}$, $V$ the bare exchange at
$q = 0$, $W$ the static screened direct term with momentum transfer $k - k'$.
The Tamm–Dancoff approximation keeps $A$; the RPA kernel drops $W$ and $W^B$.

### Note on spin factors

$w_x = 2$ for a spin-restricted scalar run (`nspinor = 1`, singlet: $D + 2V - W$)
and $w_x = 1$ for spinors ($D + V - W$), whose pair amplitude already sums both
spinor components. `bse_preconditioner.exchange_spin_weight` owns the factor;
every exchange encode applies it (stack TDA and pair matvecs, the head term,
the ring encodes, the exact diagonal). Decodes do not.

## ISDF representation

With $\psi_{nk,s}(\mu) \equiv \psi_{nk,s}(r_\mu)$ at the $N_\mu$ centroids:

- pair amplitude $M_{cv}(\mu,k) = \sum_s \psi^*_{ck,s}(\mu)\,\psi_{vk,s}(\mu)$
  (`compute_pair_amplitude`, hoisted out of the solve);
- exchange, dense in $(k,k')$:
  $(VX)_{cvk} = \dfrac{w_x}{N_k} \sum_{\mu\nu} M_{cv}(\mu,k)\, V_{\mu\nu}
  \sum_{c'v'k'} M^*_{c'v'}(\nu,k')\, X_{c'v'k'}$;
- direct term through the spin matrix
  $T_{ts}(\mu,\nu,k) = \sum_{cv} \psi_{ck,t}(\mu)\,\psi^*_{vk,s}(\nu)\,X_{cvk}$:
  $$(WX)_{cvk} = \frac{1}{N_k}\sum_{\mu\nu ts} \psi^*_{ck,t}(\mu)\,\psi_{vk,s}(\nu) \sum_{k'} W_{\mu\nu}(k-k')\, T_{ts}(\mu,\nu,k').$$

The $k'$ sum is a convolution on the k grid, evaluated as
$U = \mathrm{fftn}_\text{ortho}\big(W_R \cdot \mathrm{ifftn}_\text{ortho}(T)\big)$
by the k-convolution router (`common.fft_helpers.make_local_kconv_kminor`:
nvidia-mathdx on CUDA, the plan route on CPU), with $W_R$ the screened tile
already in R space. The cost is $O(N_k \log N_k)$ per $(\mu,\nu,t,s)$; a dense
$N_k \times N_k$ contraction is forbidden because it inverts at large $N_k$.

The optional nonanalytic exchange head (`head_minibz_average`) is a rank-3
term over transitions,
$K^\text{head} = \frac{1}{N_k}\, d^*_a\, M_{ab}\, d_b$ with $d$ the transition
dipoles and $M_{ab} = \langle v(q)\, q_a q_b\rangle_\text{cell}$; it reuses the
exchange encode/decode with $(D_\text{head}, M_\text{head})$ in place of
$(M, V_{q0})$ ([LT head](../../../docs/theory/lt-exchange-head.md)).

## Layouts

Square mesh `('x','y')`; shardings from `bse_ring_comm.make_bse_shardings`.

| array | shape | sharding |
|---|---|---|
| trial block `X` | `(n_trials, n_c, n_v, N_k)` | `P(None, 'x', 'y', None)` |
| `psi_c_X` / `psi_v_Y` | `(N_k, n_c, n_s, N_μ)` / `(N_k, n_v, n_s, N_μ)` | μ on `'x'` / ν on `'y'` |
| `M_X` / `M_Y` | `(N_k, n_c, n_v, N_μ)` | μ on `'x'` / ν on `'y'` |
| `V_q0` | `(N_μ, N_μ)` | `P('x', 'y')` |
| `W_R` | `(N_μ, N_μ, n_kx, n_ky, n_kz)` | `P('x', 'y', None, None, None)` |
| `eps_c`, `eps_v` | `(N_k, n_c)`, `(N_k, n_v)` | replicated |

$N_\mu$ is padded to the mesh divisor (`padded_mu_extent`); $n_c$ and $n_v$ are
padded to $p_x$ and $p_y$ with zero ψ and a signed ε sentinel
(`PAD_EPS_GUARD_RY`), so pad states decouple. ψ, $X$, $V$ and $W$ are
complex128; ε is float64 in Ry.

## The trial-stack matvec (`bse_stack_matvec`)

`build_bse_stack_matvec(mesh, nkx, nky, nkz, kernel='bse'|'rpa')` is the one
TDA/RPA matvec every sharded solver uses. Per trial block:

1. one all-gather of the block over `'y'` then `'x'`, so every rank holds each
   trial whole, $(n_c, n_v, N_k)$ ($16\,n_\text{trials} n_c n_v N_k$ bytes, the
   only replicated operand);
2. `lax.scan` over trials, no collective in the body: encode
   $R = \sum_v \psi^*_v X$, $T = \sum_c \psi_c R$ into the rank's
   $(\mu_\text{loc}, \nu_\text{loc}, n_s, n_s, N_k)$ tile, convolve, decode to
   the rank's partial $(n_c, n_v, N_k)$;
3. one reduce-scatter back to the `X` layout.

Exchange runs outside the scan: a k-summed encode to $(n_\text{trials}, N_\mu)$,
one GEMM with $V_{q0}$, a broadcast decode.

Per device: one $T$ tile, $16\, n_s^2 N_\mu^2 N_k / P$ bytes, whatever the
block width (the scan bounds it); $W_R$ at $16\, N_\mu^2 N_k / P$. Per trial
vector the encode and decode cost $O(N_k\, n_c\, n_s^2 N_\mu^2 / P)$ flops
plus $O(N_k\, n_c n_v n_s N_\mu / p_y)$, and the convolution
$O(n_s^2 N_\mu^2 N_k \log N_k / P)$. The manual `shard_map` is kept because it
fixes the decode collectives to reduce-scatters on every backend;
`LORRAX_BSE_MATVEC_OPT=gspmd` runs the same scan without it as an audit route,
and any other token refuses.

Full BSE uses `build_bse_stack_pair_matvec`, the real-linear applier
$\text{pair}(X, s) = AX + s\,B\bar X$ ($s = \pm1$ traced, Shao–da Jornada–Yang
Algorithm 4). Because the $A$ and $B$ encodes produce tiles of the same shape
and sharding (one ζ set serves both legs), their sum passes through one
convolution and one decode. `bse_ring_comm.build_bse_ring_matvec_full` stays as
the dense $(A, B)$ oracle for the equality gates.

## Solvers

| route | module | use |
|---|---|---|
| Lanczos / block Lanczos | `bse_lanczos.solve_bse_sharded` | spectrum shape; full reorthogonalization by default (CGS2) |
| Davidson | `bse_davidson_helpers` | per-state convergence (oscillator strengths) |
| thick-restart Lanczos | `solvers/thick_restart_lanczos.py` | bounded Krylov memory at large dimension |
| FEAST | `bse_feast` | the driver's default without `--lanczos` |
| full BSE | `bse_nontda` | structure-preserving solve of the non-TDA problem |
| absorption | `absorption_haydock`, `absorption_eigvecs` | $\varepsilon_2(\omega)$ by continued fraction or sum over states |

## Output: `eigenvectors.h5`

Written by `bse_window.write_eigenvectors_stream`, rank 0 only, in the
BerkeleyGW layout (`../eigenvectors.h5.spec`):

- `exciton_data/eigenvalues` `(N,)` in **eV** (internal units are Ry);
- `exciton_data/eigenvectors` `(nQ=1, N, N_k, n_c, n_v, ns=1, 2)` float64,
  real/imag last; full BSE adds `eigenvectors_coupling` (Y) in the same shape;
- the **valence axis is reversed** on write (BerkeleyGW `iv = 1` is the highest
  valence band; internally `v = 0` is the lowest); conduction and k are not;
- `exciton_header/params` carries `nc`, `nv`, `ns`, `nevecs`, `use_tda`.

The writer refuses to trim nonzero amplitude when the declared window is
narrower than the solved one: pass the loader's resolved counts, not the CLI
request, because `--band-degeneracy snap` widens the window.
