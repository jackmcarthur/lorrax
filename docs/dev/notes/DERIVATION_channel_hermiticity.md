# Channel Hermiticity of $\sigma^\tau$ and legal GEMM reductions in the PPM $\Sigma_c$ projection

> **Historical derivation.** The HGL crossing channel analyzed here is
> retired. This memo remains as evidence for why that channel could not use
> the Laplace symmetry reductions; it is not a current assembly guide.

**Scope.** Derivation memo for LORRAX @ `b436e47`. Question: given Hermitian $G$ and $W$,
what symmetry do the elementwise channel objects $\sigma_R=\mathrm{Re}\,\sigma^\tau$ and
$\sigma_I=\mathrm{Im}\,\sigma^\tau$ carry in $(r_\mu, r_\nu)$, per window family, and which
GEMM reduction of the $\psi^\dagger\sigma\psi$ projection is therefore legal?
Sources: `src/gw/ppm_sigma.py`, `src/gw/ppm_windows.py`, `src/gw/ppm_tau_kernel.py`
(`_project_ri_local` lines 107–135), `src/gw/ppm_accumulators.py`
(`_project_tau_onto_omega_np`), `src/gw/greens_function_kernel.py`,
`src/gw/minimax_screening.py` (`_gn_ppm_fit_kernel`), `src/gw/screening.py` (`_gate_w`),
`src/gw/w_isdf.py`, `src/common/fft_helpers.py`, `services/symmetry_maps/`
(`maps.py`, `density_symmetry_check.py`), `manual/05_isdf/5.1, 5.4`.

No code was changed; nothing was run. Where the code's index structure defeats a clean
answer, that is stated explicitly (§3.6).

---

## 1. Exact definitions, as built by the code

### 1.1 The per-$\tau$ operand $\sigma^\tau_k(\mu,\nu)$

`_sigma_kij_kernel` (`ppm_tau_kernel.py:287–311`) builds, with `ortho`-normalized flat-k
FFTs (`common/fft_helpers.make_flat_k_{i,f}fftn`, FFT axes = the $(n_{kx},n_{ky},n_{kz})$
grid, both $\mu$ axes mesh-sharded and untouched by the FFT) and
`inv_sqrt_nk` $=-1/\sqrt{N_k}$:

$$
\sigma^\tau_k \;=\; \mathrm{FFT}_{R\to k}\!\Big[\, G_R \cdot V_R \cdot \big(-\tfrac{1}{\sqrt{N_k}}\big) \Big],
\qquad G_R = \mathrm{IFFT}_{k\to R}[G_{k'}],\quad V_R = \mathrm{IFFT}_{q\to R}[W_q].
$$

The product $G_R\cdot V_R$ is **elementwise** (Hadamard) in the composite indices: $G$ is
`(nk, s, μ_X, s', μ_Y)` and $V_R$ is broadcast as `(nk, 1, μ_X, 1, μ_Y)` — $W$ is
spin-independent. Writing $a=(s,\mu)$, $b=(s',\nu)$ and carrying the circular-convolution
identity of the ortho FFT pair exactly:

$$
\boxed{\;\sigma^\tau_k(a,b) \;=\; -\frac{1}{N_k}\sum_{q\,\in\,\text{full grid}}
G^\tau_{k\ominus q}(a,b)\; W^\tau_q(\mu,\nu)\;}
$$

where $\ominus$ is grid subtraction with periodic wrap (the k and q grids are the same
full, unshifted $\Gamma$-centered grid; $q\to -q$ is a bijection of the summation set —
this is all §2 needs, so the $k\ominus q$ vs $k\oplus q$ sign convention of `jnp.fft`
never enters the symmetry argument).

### 1.2 The $G$ factor (internal band sum)

`build_G_tau` (`greens_function_kernel.py:34–64`), called with time argument
$i\,t_{\rm node}$ so `phases` $= e^{-i\,t_{\rm node}(E - E^{\rm ref}_A)}$, gated by the
window's `mask_A`:

$$
G^\tau_{k'}(a,b) \;=\; \sum_{n\,\in\,\text{full window}} m_A(k',n)\;
\psi_{nk'}(s,\mu)\; e^{-i\,t\,\big(E_A(k',n)-E^{\rm ref}_A\big)}\;\psi^*_{nk'}(s',\nu),
$$

with $\psi$ = `psi_coh_xn` / `conj(psi_coh_yr)` (both store the *un-conjugated* $\psi$;
`build_G` applies the single conj on the $\nu$ side), $E_A = E_c-E_F$ (cond) or
$E_F-E_v$ (val) $\ge 0$, $E^{\rm ref}_A = \min$ of the masked $E_A$, and
$m_A$ = `base_mask_A` intersected with the window's $E_A \le T$ / $E_A > T$ cut.
The two window families differ **only** in the value of $t\equiv t_{\rm node}$:

- **Laplace** windows (`time_axis='imag'`): $t = -i\tau_j$, $\tau_j>0$
  $\Rightarrow$ phases $e^{-\tau_j\,\Delta E}$ **real, positive**.
- **Crossing** (HGL core) window (`time_axis='crossing_hgl'`, then $t=\tau_j/\xi$):
  $t$ **real** $\Rightarrow$ phases $e^{-i\,(\tau_j/\xi)\,\Delta E}$ **unimodular complex**.

### 1.3 The $W$ factor (PPM pole sum) and the Hermiticity status of $B_q$

`_build_W_t_q` (`ppm_tau_kernel.py:390–399`):

$$
W^\tau_q(\mu,\nu) \;=\; m_B(q,\mu,\nu)\; B_q(\mu,\nu)\;
e^{-i\,t\,\big(\Omega_q(\mu,\nu)-E^{\rm ref}_B\big)} .
$$

$B_q,\Omega_q$ come from the **elementwise** GN-PPM fit
(`minimax_screening._gn_ppm_fit_kernel:454–481`), with $W^c \equiv W-V$:

$$
\Omega_q(\mu,\nu) = \sqrt{\ \mathrm{Re}\Big[-z^2\,\frac{W^c_q(z)}{W^c_q(0)-W^c_q(z)}\Big](\mu,\nu)\ }\ \ (\text{or fallback / }0),
\qquad
B_q = -\tfrac12\, W^c_q(0)\,\Omega_q .
$$

**Claim ($B_q$ Hermitian, GN probe).** If $W_q(0)$, $W_q(z)$ and $V_q$ are Hermitian in
$(\mu,\nu)$ per $q$ and $z = i\omega_p$ is purely imaginary ($z^2$ real), then every step
is conjugation-equivariant under $(\mu,\nu)\to(\nu,\mu)$: the elementwise ratio of two
Hermitian matrices is Hermitian, $-z^2\times$Hermitian is Hermitian, so its elementwise
real part is a **real symmetric** matrix. `safe` ($|{\rm denom}|>10^{-14}$), `isfinite`,
`>0`, `mode_mask`, hence `good`, `valid`, `B_mask_raw` ($\Omega>10^{-14}$) and the
window sub-masks $\Omega\lessgtr T$ are all **symmetric** boolean matrices. Therefore

$$
\Omega_q^{\mathsf T}=\Omega_q\ (\text{real symmetric}),\qquad
B_q^\dagger = B_q\ (\text{Hermitian}),\qquad
m_B^{\mathsf T}=m_B .
$$

**Verified status of the premise in the code.** $V_q$ is Hermitian by construction and
gate-checked (`gw_init.py:878`). $W$ is produced by an LU Dyson solve
$W=(I-V\chi_0)^{-1}V$ (`w_isdf.py:213ff`) with **no Hermitization step**; Hermiticity is
*asserted*, not enforced, by `screening._gate_w` — `sanity.check_hermitian` on the $q=0$
tile only, at generous `rtol=1e-6`, and **only for imaginary-axis frequencies**
($\omega=0$ and $i\omega_p$; the comment at `screening.py:460–463` is explicit that the
real-axis HL probe is *not* Hermitian in general and is not gated). Consequently:

- **GN-PPM (default, $z=i\omega_p$):** $B_q^\dagger = B_q$ holds up to the *inherited*
  residual $\varepsilon_H \equiv \max_q \|W_q - W_q^\dagger\|_\infty / \|W_q\|_\infty$ of
  the LU solve (analytically zero; numerically $O(\kappa\,\epsilon_{\rm mach})$, gated
  only at $10^{-6}$). Every "machine precision" claim below must be read as
  "to $O(\varepsilon_H)$, measured not assumed".
- **HL-PPM (real probe $\Omega$):** the premise fails; $B_q$ is **not** Hermitian and
  $\Omega_q$ is **not** symmetric. All Laplace-family symmetry verdicts below are
  **void** on an HL run.

Pad safety: pad $\mu$ modes are born dead ($\Omega=B=0$), i.e., symmetric zeros — padding
does not perturb any symmetry statement.

### 1.4 The window families per branch and the channel split

`_iter_branches` × `_build_windows_for_branch`: `denom_can_cross = (space=="cond") XOR
neg_omega_half`.

| branch | windows | kernel type | $t_{\rm node}$ | `project` |
|---|---|---|---|---|
| cond, $+\omega$ | `core` | crossing (HGL) | $\tau_j/\xi$ (real) | `imag` |
| | `a_stripe`, `b_slab` | Laplace | $-i\tau_j$ | `full` |
| val, $+\omega$ | `single` | Laplace | $-i\tau_j$ | `full` |
| cond, $-\omega$ | `single` | Laplace | $-i\tau_j$ | `full` |
| val, $-\omega$ | `core` | crossing (HGL) | $\tau_j/\xi$ (real) | `imag` |
| | `a_stripe`, `b_slab` | Laplace | $-i\tau_j$ | `full` |

(The degenerate tiny-$\omega$ case emits a single **Laplace** window even on a crossing
branch — still `project="full"`.) `omega_sign`, `prefactor`, $\alpha_{\rm eff}$ are
**host-side scalars**; they multiply whole $(\mu,\nu)$ slices and are irrelevant to the
index symmetry.

The **channel split** (`_project_ri_local`): $\sigma_k$ is split elementwise
*before* projection, and each real channel is pushed through the same two-GEMM chain:

$$
S_R(k)\;=\;\psi_k^\dagger\,\big(\mathrm{Re}\,\sigma^\tau_k\big)\,\psi_k,\qquad
S_I(k)\;=\;\psi_k^\dagger\,\big(\mathrm{Im}\,\sigma^\tau_k\big)\,\psi_k,
$$

each a **complex** `(nk, m, n)` object (the $\psi$ are complex), reduce-scattered
$(m_X,n_Y)$ and shipped to host. The single host $\omega$-projector
(`_project_tau_onto_omega_np`) then consumes them as

- `project_code=0` (**full**, all Laplace windows):
  $\ \mathrm{contrib} = c\,(S_R + i\,S_I)$, $c = \text{pref}\cdot\alpha_{\rm eff}\cdot e^{i\,\text{sign}\,\omega t}$;
- `project_code=1` (**imag**, crossing core):
  $\ \mathrm{contrib} = \mathrm{Re}(c)\,S_I + \mathrm{Im}(c)\,S_R$ — the elementwise
  $\mathrm{Im}[c\,\sigma(\mu,\nu)]$ taken *before* projection, then projected; the two
  channels are weighted by two *independent* real $\omega$-vectors.

---

## 2. The Hadamard-product Hermiticity argument

**Lemma.** If $A^\dagger=A$ and $B^\dagger=B$ elementwise in the same index pair, then
$(A\odot B)(\nu,\mu) = A(\nu,\mu)B(\nu,\mu) = A^*(\mu,\nu)B^*(\mu,\nu) =
(A\odot B)^*(\mu,\nu)$: the Hadamard product of Hermitian matrices is Hermitian, and any
real-weighted sum of such products is Hermitian. (Contrast the ordinary matrix product,
which is *not* Hermiticity-preserving; the entire argument leans on the kernel's
elementwise structure, with $G$ at $k\ominus q$ and $W$ at $q$ evaluated at the **same**
$(\mu,\nu)$.)

### 2.1 Laplace windows — the argument holds

Real phases: $G^\tau_{k'} = \sum_n \psi_n\, w_n\, \psi_n^\dagger$ with $w_n \ge 0$ real
$\Rightarrow G^{\tau\,\dagger}_{k'} = G^\tau_{k'}$ exactly (as exploited for $\chi_0$ at
`w_isdf.py:72`). $W^\tau_q = m_B \odot B_q \odot e^{-\tau_j(\Omega_q - E^{\rm ref}_B)}$
is Hermitian $\odot$ (symmetric real) $\odot$ (symmetric bool) $=$ Hermitian, given §1.3.
By the lemma and the $q$-sum:

$$
\boxed{\ \sigma^{\tau\,\dagger}_k = \sigma^\tau_k
\quad\Longrightarrow\quad
\sigma_R^{\mathsf T} = \sigma_R\ \ (\text{symmetric}),\qquad
\sigma_I^{\mathsf T} = -\sigma_I\ \ (\text{antisymmetric}),\ }
$$

per external $k$, per $\tau$ slice, on the composite index $a=(s,\mu)$ (the spinor axes
ride along: $W$ is spin-diagonal-broadcast, $G$ Hermitian on $(a,b)$).
Accuracy: exact up to $O(\varepsilon_H)$ from $B_q$ (§1.3) plus FFT/GEMM roundoff — the
$G$ factor and all masks are Hermitian/symmetric *by construction*, so the only
symmetry-breaking input is $B_q$.

### 2.2 Crossing (HGL core) window — the argument fails

Both factors carry unimodular phases with **symmetric** (not antisymmetric) exponents:

$$
G^\tau_{k'} = \sum_n \psi_n\, e^{-i\theta_n}\,\psi_n^\dagger \ \ (\theta_n\ \text{real}),
\qquad
W^\tau_q = B_q \odot e^{-i\Phi_q},\ \ \Phi_q^{\mathsf T}=\Phi_q\ \text{real}.
$$

Neither is Hermitian: conjugate-transposing flips the *sign of the phases* but the
phases are shared symmetrically between $(\mu,\nu)$ and $(\nu,\mu)$. What survives is the
**cross-$\tau$** relation

$$
\big[G^\tau\big]^\dagger = G^{-\tau},\quad \big[W^\tau\big]^\dagger = W^{-\tau}
\ \Longrightarrow\
\boxed{\ \big[\sigma^{\tau}_k\big]^\dagger = \sigma^{-\tau}_k\ }
$$

which relates different quadrature abscissae. The HGL sin-sum grid is one-sided
($\tau_j > 0$; `solve_phase_minimax_bandwidth` targets, no $\pm$ pairing), so this buys
nothing per slice. Hence for the crossing window, **on any single $\tau$ slice**,
$\sigma_R$ and $\sigma_I$ are **neither** symmetric nor antisymmetric — generically full
matrices with $O(1)$ symmetry residual. (They are also not complex-symmetric:
$\sigma^{\mathsf T}$ would need $B^{\mathsf T} = B$, i.e., real $B$, false in the
centroid basis at general $q$.)

---

## 3. Separability: what the projection can and cannot recover

Write $X_k \equiv \psi_k^\dagger \sigma^\tau_k \psi_k = S_R + i\,S_I$ (bilinearity; this
is exactly what one *complex* GEMM chain would produce), and
$Y_k \equiv \psi_k^{\mathsf T}\, \sigma^\tau_k\, \psi_k^{*}$ (the "conjugate pair").
Since $\sigma_R,\sigma_I$ are real, $\ Y_k = S_R^* + i\,S_I^*$, hence for **any**
$\sigma$, any family:

$$
\boxed{\ S_R = \tfrac12\,(X + Y^{*}),\qquad S_I = \tfrac{1}{2i}\,(X - Y^{*}).\ }
\tag{3.1}
$$

### 3.1 Why the Toeplitz splitting of $X$ fails (Laplace case)

If $\sigma$ is Hermitian (§2.1): $\sigma_R$ real symmetric $\Rightarrow S_R^\dagger=S_R$
(Hermitian); $\sigma_I$ real antisymmetric $\Rightarrow S_I^\dagger = -S_I$
(anti-Hermitian) $\Rightarrow (iS_I)^\dagger = iS_I$. So

$$
X = \underbrace{S_R}_{\text{Hermitian}} + \underbrace{i\,S_I}_{\text{Hermitian}}
\quad\Longrightarrow\quad X^\dagger = X .
$$

The Toeplitz (Cartesian) decomposition $X = H + iK$, $H=\tfrac12(X+X^\dagger)$,
$K=\tfrac{1}{2i}(X-X^\dagger)$, returns $H = X = S_R + iS_I$, $K = 0$: it splits
Hermitian from *anti*-Hermitian, but here **both** addends are Hermitian — the
decomposition of $X$ into $(S_R,\,iS_I)$ is a sum of two elements of the *same* real
subspace and is therefore **non-unique given $X$ alone**. Degrees of freedom confirm it:
$(S_R, S_I)$ carry $n^2 + n^2$ real dof; Hermitian $X$ carries $n^2$. One complex GEMM
chain can never separate the channels. (For the crossing family $\sigma$ has no symmetry,
$X$ is a general matrix with $2n^2$ dof against $4n^2$ needed — separation fails even
harder; the Toeplitz split would recover the channels from $X$ alone **only** if
$\sigma$ were complex-symmetric, $\sigma^{\mathsf T}=\sigma$, which holds for neither
family.)

**However — and this is the punchline for the Laplace family — the separation is not
needed there.** The `project_code=0` consumer (§1.4) forms *exactly*
$c\,(S_R + iS_I) = c\,X$ and nothing else. The channel split is informationally
redundant for every `project="full"` window; only the crossing window's
`project_code=1` consumer needs $(S_R, S_I)$ separately.

### 3.2 The conjugate-pair route

$(3.1)$ recovers both channels from the pair $(X, Y)$ for **any** $\sigma$ — no symmetry
required. Cost accounting: $Y$'s chain ($\sigma\cdot\psi^*$, then $\psi^{\mathsf T}\cdot$)
is a full second GEMM chain not derivable from $X$'s intermediates
($\sigma\psi^* \ne f(\sigma\psi)$ elementwise), so per $(k,\tau)$ this route is
**GEMM-neutral** versus the current two real-channel chains: 2 chains either way. Its
value is that both chains are *plain complex* GEMMs (no real-upcast asymmetry) and that
it composes with §3.3.

### 3.3 The TRS route

**What the code actually assumes.** LORRAX does not blanket-assume TRS: it *measures* it
from the density (`symmetry_maps.check_density_symmetries`, identity $m_{-k}(r) = -m_k(r)$,
verdict stored as `wfn.trs_holds`) and only then admits TRS rows into `SymMaps`
(`symmetry_maps/maps.py`, `SymMaps.__init__`'s TRS-augmentation branch). When a full-BZ $k$ is generated from an IBZ parent by a
TRS row, `unfold_psi` constructs it as literally
$\psi_{-k} = (i\sigma_y)\,\psi_k^*$ (spinor `ns=2`) or $\psi_{-k} = \psi_k^*$ (`ns=1`),
with the non-symmorphic $\tau$ phase conjugated consistently (`unfold_psi` (★)).

Assume the deck satisfies, band-by-band at the centroids,

$$
\psi_{n,-k}(r_\mu) = \big[i\sigma_y\big]\,\psi^*_{n,k}(r_\mu),\qquad E_{n,-k}=E_{n,k}.
\tag{TRS}
$$

Then, writing $\mathsf T$ for transposition of the composite $(a,b)$ pair:

1. $G^\tau_{-k'} = \Theta\, G^{\tau\,\mathsf T}_{k'}\, \Theta^{\dagger}$ with
   $\Theta = i\sigma_y\otimes 1_\mu$ (for `ns=1`, $\Theta=1$: $G_{-k'} = G_{k'}^{\mathsf T}$),
   *with the same, un-conjugated phases* — holds for **both** families, and is
   **gauge-robust**: any unitary remixing within degenerate multiplets cancels inside
   the band sum.
2. $W$-side: under TRS, $G_R$ is real (Hermitian per $k$ + $G_{-k}=G_k^{\mathsf T}$),
   so $\chi_R$ is real, $\chi_{-q} = \chi_q^*$, $V_{-q}=V_q^*$, hence
   $W_{-q} = W_q^{*} = W_q^{\mathsf T}$, and elementwise through the fit:
   $\Omega_{-q} = \Omega_q$, $B_{-q} = B_q^{*}$, masks TRS-even
   $\Rightarrow W^\tau_{-q} = \big[W^\tau_q\big]^{\mathsf T}$ for both families
   (the phases are symmetric, so transposition conjugates only $B$).
3. Substituting $q \to -q$ (a bijection of the full grid) in §1.1, and noting
   $i\sigma_y$ is a **real** matrix so it commutes with $\mathrm{Re}/\mathrm{Im}$:

$$
\boxed{\ \sigma^\tau_{-k} = \Theta\,\big[\sigma^\tau_{k}\big]^{\mathsf T}\,\Theta^\dagger
\quad\text{(both families, every }\tau\text{)},\qquad
\sigma_{R/I}(-k) = \Theta\,\sigma_{R/I}(k)^{\mathsf T}\,\Theta^{\mathsf T}.\ }
\tag{3.2}
$$

Projected, the $\Theta$'s cancel against the TRS transform of the projector $\psi$'s:

$$
\boxed{\ S_{R}( -k) = S_{R}(k)^{\mathsf T},\qquad
S_{I}(-k) = S_{I}(k)^{\mathsf T},\qquad
X_{-k} = X_k^{\mathsf T},\qquad Y_{-k}=Y_k^{\mathsf T}.\ }
\tag{3.3}
$$

The $(m,n)$ transform is a **pure transpose — no conjugation** — of the QP band pair.

**Answer to "can the conjugated pair at $k$ be recovered from the plain pair at $-k$?"
— No.** Under (TRS), $Y_k = \psi_{-k}^\dagger\,\sigma_k\,\psi_{-k}
= \psi_{-k}^\dagger\,\sigma_{-k}^{\mathsf T}\,\psi_{-k} = Y_{-k}^{\mathsf T}$: the $X$
family and the $Y$ family each close onto **themselves** under $k\to-k$ (with the
$(m,n)\to(n,m)$ transpose) and never map onto each other. The BZ sum halves the work
*within* each family; it cannot substitute one family for the other. (Adding per-$k$
Hermiticity in the Laplace case only turns (3.3) into $X_{-k}=X_k^*$, $Y_{-k}=Y_k^*$ —
still self-referential.)

**Gauge caveat (load-bearing).** (3.2) is gauge-robust, but (3.3) is **gauge-covariant**:
if the stored deck realizes TRS only up to a band gauge,
$\psi_{m,-k} = e^{i\varphi_{mk}}\,\Theta\psi^*_{m,k}$ (per-band phases; unitary blocks on
degenerate multiplets), then $X_{-k} = D_k^{*}\, X_k^{\mathsf T}\, D_k$ with
$D_k = \mathrm{diag}(e^{i\varphi_{mk}})$ (resp. block-unitary $U$), which defeats
elementwise reuse unless $D$ is known. The code *guarantees* the literal gauge only for
TRS-**unfolded** k-points (built by `unfold_psi` as $\Theta\,\mathrm{conj}(\cdot)$); a
full-BZ WFN written directly by the DFT code carries whatever gauge its diagonalizer
chose, and `trs_holds` (a density-level test) does **not** certify band-gauge alignment.
This must be measured per deck (§4, check T3).

### 3.4 Verdicts: legal GEMM reductions per window family and channel

Let one "chain" = one `right` GEMM + one `psum_scatter(y)` + one `left` GEMM + one
`psum_scatter(x)` (the `_project_ri_local` unit). Current cost: **2 chains per
$(\tau,k\text{-batch})$** for every window (channels $R$ and $I$).

1. **Laplace windows (`single`, `a_stripe`, `b_slab`; `project="full"`; all four
   branches) — 2× reduction, unconditionally legal.** Replace the two real-channel
   chains by **one complex chain** $X = \psi^\dagger \sigma \psi$; the host consumer
   already forms only $c\,X$. This is licensed by *bilinearity alone* — it needs no
   Hermiticity, no TRS, and survives HL fits and gauge issues. It also halves the
   stacked `psum_scatter` payloads and the D2H tile bytes (today both channels ship as
   complex tiles). Value-level identical, not bit-exact (complex-GEMM association);
   gate with the repo's $10^{-12}$ parity suite.
   *Hermiticity is not what licenses this* — what Hermiticity adds for these windows is
   (i) the checkable invariants below, and (ii) optional triangle-output tricks
   ($X^\dagger = X$ admits a HERKX-style half-flop left GEMM and triangle-only D2H —
   not expressible in the current einsum/XLA stack; note only).
2. **Crossing core window (`project="imag"`; cond $+\omega$, val $-\omega$) — no per-$k$
   reduction exists.** Both channels are genuinely consumed with independent $\omega$
   weights; $\sigma^\tau$ has no per-slice symmetry (§2.2); $X$ alone under-determines
   $(S_R,S_I)$ regardless of family (§3.1); the conjugate-pair route is GEMM-neutral
   (§3.2). The only 2× is the **TRS half-BZ route**: evaluate the projection chains on a
   half-BZ (pair $k$ with $\ominus k$; self-paired points $k=\ominus k$ computed once)
   and fill the other half by $(m,n)$ transpose via (3.3). Conditional on the literal
   deck gauge (§3.3 caveat).
3. **TRS half-BZ applies to Laplace windows too** and composes with (1):
   up to **4×** on Laplace projection chains, **2×** on crossing chains, when (TRS)
   holds literally.
4. **Illegal:** any single-chain scheme for the crossing window at fixed $k$ (information
   loss — §3.1); any scheme relying on $\sigma_R/\sigma_I$ per-slice symmetry inside a
   crossing window; any Laplace-family symmetry assumption on an **HL**-probe run
   (§1.3); TRS reuse on a deck whose $\pm k$ band gauge is unverified.

### 3.5 Numerically checkable identities (per case, any $\tau$ slice)

With $\hat\varepsilon[\cdot]$ = max-abs residual normalized by $\max|\sigma^\tau_k|$
(resp. $\max|S|$):

- **(L1)** Laplace, per $k$: $\hat\varepsilon[\sigma_R - \sigma_R^{\mathsf T}] \lesssim \varepsilon_H$
  and $\hat\varepsilon[\sigma_I + \sigma_I^{\mathsf T}] \lesssim \varepsilon_H$, where
  $\varepsilon_H = \max_q \|B_q - B_q^\dagger\|_\infty/\|B_q\|_\infty$ is measured first.
  With exactly Hermitized $B$ inputs, both residuals drop to machine precision — "$=0$
  to machine precision on any $\tau$ slice" holds *conditionally on* $\varepsilon_H$,
  which the pipeline gates only at $10^{-6}$, $q=0$.
- **(L2)** Laplace, projected: $\hat\varepsilon[X - X^\dagger] \lesssim \varepsilon_H$ per $k$, and
  $\hat\varepsilon[(S_R + iS_I) - X] = O(\epsilon_{\rm mach})$ (bilinearity; the merge-legality gate).
- **(C1)** Crossing, per $k$: $\hat\varepsilon[\sigma_R - \sigma_R^{\mathsf T}] = O(1)$ — the
  *falsification* check: if this comes out small on a generic slice, the derivation in
  §2.2 is wrong (or the deck is at a degenerate point, e.g. all phases $\approx 1$).
- **(C2)** Crossing, cross-$\tau$: $\hat\varepsilon[\sigma^\tau_k{}^\dagger - \sigma^{-\tau}_k] \approx 0$
  (build the $-\tau$ slice explicitly; consistency check of §2.2).
- **(T1)** TRS, operand level (gauge-free): $\hat\varepsilon[\sigma^\tau_{\ominus k} - \Theta\,\sigma^{\tau\,\mathsf T}_{k}\Theta^\dagger] \lesssim \varepsilon_H + \varepsilon_{\rm TRS}^{(B)}$,
  with $\varepsilon_{\rm TRS}^{(B)} = \max_q\|B_{\ominus q} - B_q^*\|_\infty/\|B\|_\infty$
  measured first.
- **(T2)** TRS, projected: $\hat\varepsilon[S_{ch}(\ominus k) - S_{ch}(k)^{\mathsf T}]$ — small **iff** (T3) passes.
- **(T3)** Deck gauge: $\max_{m,k}\ \min_{\varphi}\ \|\psi_{m,\ominus k} - e^{i\varphi}\Theta\psi^*_{m,k}\|$
  band-resolved (phase-optimized per band; degenerate multiplets compared as subspace
  projectors $P_{-k} = \Theta P_k^{*}\Theta^\dagger$, plus explicit extraction of the
  block unitary $U$ if subspace-equal but element-unequal).

### 3.6 Where the index structure defeats a clean answer

- $B_q$ Hermiticity is **inherited, not enforced** (LU solve; gate $q=0$ only,
  rtol $10^{-6}$). The Laplace-channel symmetry is therefore an $O(\varepsilon_H)$
  statement whose constant must be measured per run, and any GEMM scheme that *assumes*
  exact symmetry (triangle-only transport) silently symmetrizes — a physics change at
  $O(\varepsilon_H)$, below the parity gate but not bitwise.
- The **HL probe path** shares the fit and kernel code but voids the Hermitian premise;
  nothing in `ppm_sigma` distinguishes the two downstream, so a family-level reduction
  must be gated on the probe axis at config level.
- The **$\pm k$ band gauge** of a full-BZ WFN is outside the code's control;
  `trs_holds` does not certify it (density-level test, deliberately gauge-blind), and
  degenerate multiplets can carry a nontrivial block unitary that turns the clean
  $(m,n)$ transpose of (3.3) into a congruence with an unknown $U$. For TRS-unfolded
  decks the gauge is literal by construction of `unfold_psi`.
- For SOC/bispinor decks the transform carries the (real) $i\sigma_y$ conjugation; it
  cancels in the projected identities (3.3) but must be kept in any operand-level
  ($\mu\nu$-level) reuse.

---

## 4. Cheap falsification test (sketch — not run)

Single node, `P=1` (mesh $1\times1$, CPU JAX is fine; optional `P=4` step at the end).
One script, e.g. `wk_REL/probes/check_channel_hermiticity.py`, two stages:

**Stage A — algebra falsification against the real kernel code (deck-free, seconds).**
Synthesize a small problem ($n_k = 2{\times}2{\times}1$, $n_\mu = 16$, $n_b = 8$, `ns=1`):
random $\psi$; $B_q \leftarrow \tfrac12(M_q + M_q^\dagger)$, $\Omega_q \leftarrow$
$|{\rm sym}|$ real, masks all-true. Import the *actual* building blocks —
`gw.greens_function_kernel.build_G_tau`,
`common.fft_helpers.make_flat_k_{i,f}fftn` with
`wavefunction_bundle.G_FFT7D_SPEC / V_FFT5D_SPEC`, and the 3-line
$W^\tau$ phase build mirroring `_build_W_t_q` — and assemble
$\sigma^\tau_k$ exactly as `_sigma_kij_kernel` does (`G_R * V_R * (-1/\sqrt{N_k})`,
then `_G_fftn`). Evaluate one Laplace slice ($t=-0.37i$) and one crossing slice
($t=0.29$); run checks **L1, L2, C1, C2, T1** (for T1, build the TRS-partner inputs
explicitly: $\psi_{\ominus k}:=\psi_k^*$, $B_{\ominus q}:=B_q^*$, using the same
flat-index negation map $k \mapsto (-k)\bmod N$ as `common.kq_mapping`). Then perturb
$B \mathrel{+}= 10^{-6}\,{\rm antiherm}$ and confirm L1's residual tracks
$\varepsilon_H$ linearly.

**Stage B — deck-level checks on an existing run dir (minutes).** Point at a small run
dir that has the WFN h5 (e.g. `wk_eqpO_legacy/WFNsmall.h5`-class inputs). (i) Check
$E_{n,\ominus k} = E_{n,k}$ from `enk`. (ii) Check **T3** directly in G-space on a few
$(k,\ominus k)$ pairs: $c_{\ominus k}(G) \stackrel{?}{=} \Theta\,c_k^*(-G)$ band-by-band
with per-band phase optimization and degenerate-block subspace comparison — this is the
single gate for the TRS half-BZ route on that deck. (iii) Rebuild $W(0), W(i\omega_p)$
at `P=1` via the pipeline prologue on the smallest deck (or reuse any saved $W$
artifact), run `fit_ppm`, and *measure* $\varepsilon_H$ and
$\varepsilon^{(B)}_{\rm TRS}$ — the constants that L1/T1 are conditioned on.

**Optional `P=4` parity step.** Dispatch one production `_tau_kernel` slice per family
on a $2\times2$ mesh, D2H the $(S_R, S_I)$ tiles, and verify (L2)
$S_R + iS_I = X$ against a rank-0 replicated X-route reference at $10^{-12}$ — the exact
gate the channel-merge patch would ship with.

Nothing in the test writes to the run dirs; all outputs go to stdout / a local log.

---

## 5. Verdict summary (one line per item)

1. Laplace `single` (val $+\omega$): $\sigma^\tau$ Hermitian $\Rightarrow$ $\sigma_R$ symmetric, $\sigma_I$ antisymmetric, to $O(\varepsilon_H)$; channels redundant — single-complex-chain **2× GEMM reduction legal unconditionally**.
2. Laplace `single` (cond $-\omega$): identical verdict to (1).
3. Laplace `a_stripe` (cond $+\omega$ / val $-\omega$): identical verdict to (1); masks ($E_A > T$, $\Omega \le T$) are symmetry-neutral.
4. Laplace `b_slab` (cond $+\omega$ / val $-\omega$): identical verdict to (1); $\Omega > T$ mask symmetric.
5. Crossing `core` (cond $+\omega$, val $-\omega$): $\sigma_R,\sigma_I$ **neither** symmetric nor antisymmetric per slice; only $\sigma^\tau{}^\dagger = \sigma^{-\tau}$ (one-sided grid $\Rightarrow$ unusable); both channels required; **no per-$k$ reduction**.
6. Toeplitz splitting of $\psi^\dagger\sigma\psi$: fails for Laplace ($S_R$ and $iS_I$ both Hermitian $\Rightarrow$ non-unique) and for crossing (no symmetry at all); never a legal recovery route.
7. Conjugate-pair route $(X, Y)\to(S_R,S_I)$: exact for any family, but GEMM-neutral (2 chains either way); useful only as the recovery formula under route 8.
8. TRS route: $\sigma_{\ominus k} = \Theta\sigma_k^{\mathsf T}\Theta^\dagger$ and $S_{ch}(\ominus k) = S_{ch}(k)^{\mathsf T}$ (pure $(m,n)$ transpose) — **half-BZ 2× for all families/channels**, conditional on literal deck gauge $\psi_{\ominus k} = \Theta\psi_k^*$ (guaranteed only for TRS-unfolded k-points; measure via check T3); $Y$ is never recoverable from $X$ ($X$/$Y$ orbits are disjoint under $k\to-k$).
9. Composite ceiling: Laplace 4× (merge × half-BZ), crossing 2× (half-BZ only); all claims void on an HL-probe run and degraded by $\varepsilon_H$ (Hermiticity of $B_q$ is inherited from the un-Hermitized LU Dyson solve, gated only at $10^{-6}$, $q=0$).
10. Checkable gates: L1/L2 (Laplace symmetry + merge parity), C1/C2 (crossing falsification), T1–T3 (TRS operand / projected / deck-gauge), each stated in §3.5 with its expected residual scale.
