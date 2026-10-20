# BerkeleyGW `bsemat.h5` from the ISDF tensors

`bse.bsemat_export` writes the BerkeleyGW 4.0 kernel file (`bsemat.h5`, TDA or
`extended_kernel`) from a LORRAX static-W restart. No plane-wave pair density is
formed: the kernel is assembled from the centroid pair amplitudes, the stored
μ-space tensors V and W, and one ζ column per q. This page states the format
convention it must reproduce and the identity that splits W into BerkeleyGW's
head, wing and body at every q.

## What BerkeleyGW stores {#format}

Per pair (k, k′), with q the Wigner–Seitz representative of k − k′ and
`M_nn′(G) = ⟨nk|e^{i(q+G)·r}|n′k′⟩` (spinor-summed),
`kernel.x` stores (`BSE/mtxel_kernel.f90`, `w_sum.f90`, `g_sum.f90`):

$$
\begin{aligned}
\text{head} &= M_{cc'}(G_1)\,M^*_{vv'}(G_1) &&\text{(E1a)}\\
\text{wing} &= \lambda_q\Big[\sum_{G\ne G_1}M_{cc'}(G)\,\bar W_{GG_1}M^*_{vv'}(G_1)
             +M_{cc'}(G_1)\sum_{G'\ne G_1}\bar W_{G_1G'}M^*_{vv'}(G')\Big] &&\text{(E1b)}\\
\text{body} &= \sum_{G,G'\ne G_1}M_{cc'}(G)\,\bar W_{GG'}M^*_{vv'}(G') &&\text{(E1c)}\\
\text{exchange} &= \sum_{G\ne0}M^*_{vc}(G)\,\bar v(G)\,M_{v'c'}(G)\quad(k=k'\text{ legs, }q=0) &&\text{(E1d)}
\end{aligned}
$$

$\bar W=\varepsilon^{-1}v/8\pi$ and $\bar v=v/8\pi$ (bohr², `vcoularray/(8π)`,
`kernel_main.f90`). $G_1$ is the first vector of the WFN G list, i.e. G = 0 relative to
the q that `sortbyq.f90` matches to `fullbz(…, wigner_seitz=.true.)`. That is the
shortest $q+G$; ties go to the first strict minimum of the `(i1,i2,i3)` scan in
`Common/fullbz.f90`. `screening_semiconductor` stores the head with W replaced
by 1. $\lambda_q=|q|$ without truncation and 1 with it; at q = 0 the wings are
zero (`calc_wings`). `absorption` restores the screening
(`intkernel.f90`, factor $-8\pi/(\Omega N_k)$): the head is multiplied by
$v(q)\varepsilon^{-1}_{00}(q)/8\pi$ from its own `epsdiag` (q = 0:
$w_{\rm coul0}/8\pi$). The 3D wing is multiplied by
`oneoverq`/8π = 1/|q| (`vcoul_generator.f90`: $|q|v(q)$). Body and slab wing
are taken as stored, and the exchange is scaled by $-$(the factor). The head's
screening is therefore not a kernel quantity: it lives in BerkeleyGW's eps
files and the q → 0 average.

Dataset layout (h5py order; `bsewrite.f90` `bse_hdf5_write`, `Common/kernel_io.F90`
`setup_kernel_hdf5`): `mats/{head,wing,body,exchange}` are
`f8 (nk·ns, nk·ns, n2b, n2b, n1b, n1b, 2)` with index `[ikp, ik, icp, ic, ivp, iv, re/im]`.
TDA has n1b = n_v (iv = 1 the highest valence) and n2b = n_c. `extended_kernel` has
n1b = n2b = n_v + n_c in the order (valence, then conduction).

## The split from the stored tensors {#split}

Per q let $Z_{G\mu}=\zeta_{q\mu}(G)/\sqrt\Omega$ on the full G set. The restart
stores $V=Z^\dagger vZ$ and $W=(1-V\chi)^{-1}V$, which is $Z^\dagger W_{\rm PW}Z$ for the
plane-wave interaction implied by the ISDF polarizability,

$$W_{\rm PW}=(1-vZ\chi Z^\dagger)^{-1}v=v+vZ\chi_r Z^\dagger v,\qquad
\chi_r=\chi(1-V\chi)^{-1},\qquad W-V=V\chi_rV. \tag{E2}$$

Let $e_1$ be the head slot, $z=e_1^{\rm T}Z$ (the row $z_\mu=\zeta_{q\mu}(G_1)/\sqrt\Omega$) and
$v_1=v(q+G_1)$ as V carries it. Since $z^*=Z^\dagger e_1\in\operatorname{range}V$,

$$Va=z^*,\qquad s=a^\dagger(W-V)a,\qquad r=a^\dagger(W-V), \tag{E3}$$

and neither $s$ nor $r$ depends on the null-space part of $a$, because $(W-V)$
annihilates $\ker V$. Then

$$W_{11}\equiv e_1^\dagger W_{\rm PW}e_1=v_1+v_1^2 s, \tag{E4}$$
$$y\equiv e_1^\dagger W_{\rm PW}(1-e_1e_1^\dagger)Z=v_1 r-v_1^2 s\,z, \tag{E5}$$

and the μ-space head, wing and body of $W$ in the partition $\{G_1,\text{rest}\}$ are

$$H=W_{11}\,z^\dagger z,\qquad Y=z^\dagger y+y^\dagger z,\qquad B=W-H-Y. \tag{E6}$$

E4–E6 are exact for any single slot. They need one Hermitian solve and one
product per q, one ζ column, and neither χ nor any persisted GW-time quantity.

### One eigendecomposition per wedge q {#split-ibz}

At a star member $Sq$ of the wedge point $q$ the bundle's tiles are the wedge
tiles carried by the centroid action, $V_{Sq}=MV_qM^\dagger$ and
$W_{Sq}=MW_qM^\dagger$ with $M_{\mu\mu'}=\bar\varphi_\mu\delta_{\mu',\pi(\mu)}$
(complex-conjugated on antiunitary rows; `symmetry_maps.unfold_isdf_operator`),
and the head column is the parent column carried by the one-leg action,
$z_{Sq}=c\,\varphi\odot(z_q\circ\pi)$ (`unfold_isdf_one_leg`: the same $\pi$ and
$\varphi$, $c$ the τ phase of the parent slot $q+G_p$, conjugated likewise). Hence
$V_{Sq}^+=MV_q^+M^\dagger$, the spectrum and therefore the E3 cut are the same, and

$$\bar a_{Sq}=c\,\varphi\odot(\bar a_q\circ\pi),\qquad s_{Sq}=s_q,\qquad
r_{Sq}=c\,\varphi\odot(r_q\circ\pi): \tag{E3a}$$

$\bar a$ moves exactly as $z$ does, on antiunitary rows too. The writer
(`bsemat_export.head_split_keys`) diagonalises $V$ once per wedge point that a
key needs (one batched `distrib_la` eigh over the wedge points' identity
images), solves E3 once per parent slot $(q, G_p)$ (zone-boundary ties are
separate slots), carries $\bar a$ to every key through the unfold layers that
carry $z$, and evaluates E3's residual, E4 and E5 on the key's own $V$ and $W$.
The residual gate certifies the rotation of $a$ against $V_{Sq}$; $s$ at the key
against $s$ at the parent (`bsemat_head_rotation`, relative 1e-8) certifies
$W_{Sq}$ against the same action.

## Kernel assembly {#kernel}

With the centroid pair amplitudes $C_{nn'}(\mu)=\sum_s\psi^*_{nks}(r_\mu)\psi_{n'k's}(r_\mu)$,
$M_{nn'}(q+G)=\sqrt\Omega\sum_\mu Z^*_{G\mu}C_{nn'}(\mu)$, so with
$m=\sum_\mu z^*_\mu C(\mu)$ and $n=\sum_\mu y^*_\mu C(\mu)$:

$$
\begin{aligned}
\text{head} &=\Omega\,m_{cc'}m^*_{vv'}, \\
\text{wing} &=\tfrac{\Omega}{8\pi}\lambda_q\,(m_{cc'}n^*_{vv'}+n_{cc'}m^*_{vv'}),\\
\text{body} &=\tfrac{\Omega}{8\pi}\big(C_{cc'}WC^*_{vv'}-W_{11}m_{cc'}m^*_{vv'}-m_{cc'}n^*_{vv'}-n_{cc'}m^*_{vv'}\big),\\
\text{exchange} &=\tfrac{\Omega}{8\pi}\,C^*_{vc}(k)\,\bar V_\Gamma\,C_{v'c'}(k'),
\end{aligned} \tag{E7}
$$

where $\bar V_\Gamma$ is the head-less Γ tile, the conjugation follows E1d, and the
BerkeleyGW absorption factor $-8\pi/(\Omega N_k)$ turns body and wing back into
LORRAX's $-\frac1{N_k}CWC^*$. Only $C\,W\,C^*$ is μ²-sized. It runs as two
batched GEMMs per k′ row, against $W(k-k')$ held at `P(None,'x','y')`, so one
k′ row of the file is one contiguous hyperslab. The other terms are rank-1 in
the band pairs.

## The head slot at zone-boundary ties {#ties}

Where several $q+G$ are equally short (Si L, X, W; MoS₂ K), the head is
whichever plane wave `kernel.x` chose. Its q list is `fullbz` of $[0]$ plus the
`epsmat` q points over the WFN's operations. Each point is reduced with
`k_range` to $[0,1)$, and its Wigner–Seitz representative is the first strict
minimum of the $(i_1,i_2,i_3)$ scan, computed in BerkeleyGW's own sequential
arithmetic. A pair takes the first list entry equal to $\sigma(k-k')$ modulo G,
so the head momentum is $K=\sigma q$. `--bgw-eps` gives the writer that q list,
and the head of every pair is then BerkeleyGW's: the Si 4³ fold reproduces
`kernel.x`'s printed list 64/64. A rule with a length tolerance misses 2 of the
13 tied q, because there a 10⁻¹⁷ roundoff decides. Without `--bgw-eps` the
restart's wedge stands in. Tied slots related by a symmetry of q screen equally,
so the absorption is unchanged. Only the head, wing and body partition of
those blocks moves.

## q = 0 and the gauge {#q0}

At k = k′ the head is $\Omega\,m m^*$ with $m=\sum g_0^*C\approx\delta$. The wing is
zero: `calc_wings` (BSE/mtxel_kernel.f90) zeroes the q = 0 wings under
`screening_semiconductor`. The default body (`--q0-body head-less`) is the head-less Γ
tile of W. BerkeleyGW's q0 body is the body block of its fully inverted ε at q₀,
so in 3D it also carries an O(q⁰) wing-mediated term. The eps export's stored q → 0
wings give it (its theory page, E9–E13: `head_wing_L` L (3, N_μ),
`head_wing_R` R = Lᴴ, `S_cart_head` S_eff = S_d + L W(Γ) R/Ω, `S_cart_head_direct` S_d).
Projecting its E13 with $Z^\dagger\cdots Z$, using $Z^\dagger v_bZ=V$ and
$V(1+\chi^\mu V)=W$, gives

$$W_{q_0}=W(\Gamma)+\frac{8\pi}{\Omega\,\varepsilon_M}\,a\,a^\dagger,\qquad
a=W(\Gamma)\,R\,\hat q_0,\qquad \varepsilon_M=1-8\pi\,\hat q_0^{\mathsf T}S_{\rm eff}\,\hat q_0 ,
\tag{E8}$$

independent of $|q_0|$ but not of its direction. `--q0-body dipole`
(`bsemat_algebra.q0_dipole_body_rows`) adds it along BerkeleyGW's q₀, taken from
the `eps0mat.h5` beside `--bgw-eps` or from `--q0-vector`. GATE
`bsemat_q0_wing_fold` refuses unless $\Omega\hat q_0^{\mathsf T}(S_{\rm eff}-S_{\rm d})\hat q_0
=\hat q_0^{\mathsf T}LW(\Gamma)R\hat q_0$ to 10⁻³ on the writer's own W, which certifies L's μ
order and normalisation. In a slab the term is O(q₀), so the option refuses there
(`bsemat_q0_body_dim`). `tests/test_bsemat_split_cpu.py` checks E8 against the
block inverse of a dense G-space q₀ system, and fails an unconjugated R and the
wrong sign.

BerkeleyGW combines the kernel with its own WFN coefficients. The writer therefore rotates
ψ at the centroids into the WFN file's coefficient gauge with one window unitary
per k, the polar factor of the least-squares fit of LORRAX's ψ by the file's
states evaluated at the centroids (the dipfix construction). The file's states
reach the centroids through the GW face transform (`WfnLoader.load(k="ibz")`,
the file's own rows, then `gflat_to_rmu`); the fit is batched over k on the mesh
and the block polar factors are `distrib_la`'s batched polar, the
parallel-transport route. `tests/test_bsemat_gauge_cpu.py` holds it to the
per-k host fit and fails a re-phased file, a k mismatch and an aliased grid.

## The downfold stage {#downfold}

The parent's W lives on centroids fitted to the GW band range. The file needs
only its own window, the N_b = n_v + n_c bands of the layout (TDA and
`extended_kernel` share it). `--mu-small` (an integer, `kx` = k·N_b, `auto`) runs
the downfold driver's own stages (docs/downfold.md) on that window before the
rows. The default is `off`: on Si 4³ and MoS₂ 9×9 (N_b = 16) the stage adds
20–24 meV and 14–19 meV at 7–14·N_b. At the certified ceiling (rcond 10⁻⁸;
31·N_b and 47·N_b) it still adds 1.7 and 2.5 meV, against the parent file's 0.014 and
0.245 meV from BerkeleyGW's own kernel. It reaches 0.05 meV only at
0.84 μ_L (Si, rcond 10⁻¹⁰, outside the rank policy's certified κ). At these sizes the rows
are not the cost, so the stage adds time and memory (REPORT, downfold sweep). The stages are the pair-density Gram, the orbit-closed CUR
selection of μ_S parent centroids, and the transfer
$T_q=S_{SS}(q)^+S_{SL}(q)$ at `--downfold-rcond`. The CUR fit
$C_L\approx T_q^\dagger C_S$ of the window's pair amplitudes is the statement
$Z_S=Z_LT_q^\dagger$ ($\zeta_S=\bar T\zeta_L$), so

$$V_S=Z_S^\dagger vZ_S,\qquad W_S=T_qW_LT_q^\dagger=Z_S^\dagger W_{\rm PW}Z_S,\qquad
z_S=\bar T_q z,\quad y_S=\bar T_q y,\quad W_{11}^S=W_{11}. \tag{E9}$$

The plane-wave W does not change; only the pair densities it is contracted
with do, $\rho_S=Z_SC_S$. The exact head, wing and body of the child kernel
$C_S^\dagger W_SC_S$ are therefore E6–E7 with the parent's split carried by T.
The writer splits at μ_L and transports $(z,y)$ with $\bar T_q$, $W$ with
the congruence, and $\bar V_\Gamma$ with $T_0$. The rows then contract
GEMMs of size μ_S. Re-splitting the child from $(V_S,W_S,z_S)$ alone keeps
the solve exact, since $z_S^*=Z_S^\dagger e_1\in\mathrm{range}(V_S)$. It
recovers $W_{11}$ only where $e_1\in\mathrm{span}(v^{1/2}Z_S)$, which is
head_row_S = 1; with μ_S below the number of G that is not guaranteed. The writer
runs it as a diagnostic: it gates the residual (`bsemat_downfold_head_solve`,
10⁻⁶) and reports head_row_S and $|W_{11}^S/W_{11}-1|$.
`tests/test_bsemat_split_cpu.py` checks E9 against the dense partition of
$\rho_S^\dagger W_{\rm PW}\rho_S$. `--mu-small off` builds the rows on the
parent's centroids.

## Gates {#gates}

| rule | refuses when |
|---|---|
| `bsemat_head_solve` | $\lVert Va-z^*\rVert/\lVert z^*\rVert>10^{-6}$ |
| `bsemat_head_row` | $v_1\,zV^+z^*>1$: $V-v_1z^\dagger z$ is not positive, so z is not V's head row (slot, conjugation, unfold phase) or $v_1$ is not V's value |
| `bsemat_head_screening` | $W_{11}/v_1\notin(0,1]$ |
| `bsemat_point_head` | a 3D deck with `mc_average_vcoul_body = true` (V's head slot holds a cell average at every q) |
| `bsemat_head_in_sphere` | the head slot lies outside the stored ζ sphere |
| `bsemat_full_grid`, `bsemat_k_bijection`, `bsemat_k_on_grid`, `bsemat_k_order` | WFN_co is not the explicit full grid of the restart, in any order |
| `bsemat_band_offset` | the WFN_co window energies differ from the restart's by more than 0.05 Ry |
| `bsemat_q_list`, `bsemat_head_parent`, `bsemat_head_label`, `bsemat_head_length` | a pair's head momentum is missing from the q list, off the lattice, or not a shortest $q+G$ |
| `bsemat_q0_wings_source`, `bsemat_q0_vector`, `bsemat_q0_body_dim` | `--q0-body dipole` without the bundle's stored wings, without a q₀ direction, or in a slab |
| `bsemat_downfold_head_solve` | the child's $\lVert V_Sa-z_S^*\rVert/\lVert z_S^*\rVert>10^{-6}$; the downfold driver's own refusals (μ_S above the window Gram's eigenvalue rank, a budget below the first orbit) apply unchanged |
| `bsemat_q0_wing_fold` | the stored fold $S_{\rm eff}-S_{\rm d}$ disagrees with $LW(\Gamma)R/\Omega$ on this W by more than 10⁻³ |
