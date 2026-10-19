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
zero (BerkeleyGW's semiconductor convention). The body is the head-less Γ tile
of W, so it omits the O(q⁰) wing-mediated term of BerkeleyGW's full ε⁻¹ body. That
term needs the q → 0 dipole wing vector of the eps export, and `bsemat_export`
leaves a hook for it.

BerkeleyGW combines the kernel with its own WFN coefficients. The writer therefore rotates
ψ at the centroids into the WFN file's coefficient gauge with one window unitary
per k, the polar factor of the least-squares fit of LORRAX's ψ by the file's
states evaluated at the centroids (the dipfix construction).

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
