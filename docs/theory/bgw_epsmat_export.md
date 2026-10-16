# BerkeleyGW ε⁻¹ files from the ISDF screened interaction

`gw.epsmat_cli` writes the static screened interaction of a finished GW bundle
as BerkeleyGW 4.0 `eps0mat.h5` and `epsmat.h5`, one q at a time, and projects
BerkeleyGW's files back onto the bundle's interpolation vectors. This page owns
the map between the two representations, what the map preserves, the q → 0
treatment, the file conventions and the cost. The `[epsmat]` keys are in the
[input reference](../input_reference.md#epsmat-the-epsmat-input-file); the
bundle itself is in the [BerkeleyGW how-to](../how-to/berkeleygw-users.md#outputs), §3.5.

## 1. Two representations of one screened interaction

Per q (suppressed): $Z\in\mathbb C^{N_G\times N_\mu}$ with
$Z_{G\mu}=\tilde\zeta_{q\mu}(G)=\sum_r e^{-i(q+G)\cdot r}\zeta_{q\mu}(r)$ on the
bare-Coulomb sphere $|q+G|^2<E_c$ (`bare_coulomb_cutoff`), and
$v=\operatorname{diag} v(q+G)/\Omega$ in Ry, zero at $q+G=0$
([ζ and V_q](isdf-zeta-vq.md#4-coulomb-matrix)). Pair densities, and with them
χ₀, are expanded on ζ; V and W are projected on it:

$$
V = Z^\dagger v Z, \qquad \chi_0(G,G') = \big[Z\,\chi_0^{\mu} Z^\dagger\big]_{GG'} .
\tag{E1}
$$

The plane-wave RPA with this χ₀ closes in μ space. Its reducible response is
$\chi = Z\chi^{\mu}Z^\dagger$ with $\chi^{\mu}=\chi_0^{\mu}(1-V\chi_0^{\mu})^{-1}$
(substitute into $\chi=\chi_0+\chi_0 v\chi$), so

$$
W_G = v + v Z\chi^{\mu}Z^\dagger v, \qquad \varepsilon^{-1} = 1 + v Z\chi^{\mu}Z^\dagger ,
\tag{E2}
$$

$$
W \equiv Z^\dagger W_G Z = V + V\chi^{\mu}V .
\tag{E3}
$$

(E2) is BerkeleyGW's convention $W_{GG'}=\varepsilon^{-1}_{GG'}\,v(q+G')$. At
ω = 0 a gapped χ is Hermitian, hence so is W (`GATE epsmat_static_w_hermitian`,
1e-8). Solving (E3) for $\chi^{\mu}$ and substituting into (E2):

$$
\varepsilon^{-1}_{GG'}-\delta_{GG'} = v(q+G)\,\big[Z\chi^{\mu}Z^\dagger\big]_{GG'},
\qquad \chi^{\mu} = V^{+}(W-V)\,V^{+} .
\tag{E4}
$$

(E4) is not a fit. It is the plane-wave RPA inverse dielectric matrix of the
polarizability the GW run used, recovered from the stored $V$ and $W(0)$
through the Coulomb Gram; χ₀ is never persisted. `tests/test_bgw_epsmat_cpu.py`
checks (E4) against an independent dense inversion $(1-v\chi_0)^{-1}$ in G
space to 1e-10.

The export rebuilds V from ζ and its own $v$ (`gw.compute_vcoul`) and refuses
unless it equals the bundle's V to 1e-9 (`GATE epsmat_v_reconstruction`): another
ζ normalization or Coulomb kernel would rescale ε⁻¹ − 1 by their ratio.
Measured on CrI3 6×6 (slab, $N_\mu=978$): 9.7e-15 at q = 0 and 8.3e-15 at
q = (0, 1/6, 0). A producer deck with `use_bgw_vcoul` or a non-`off`
`mc_average_placement` refuses (`GATE epsmat_coulomb_policy`).

## 2. Whitening, passivity and what is represented

$V^{+}$ is applied through a factor: $V=U\Lambda U^\dagger$ and
$B=U_k\Lambda_k^{-1/2}$ over the kept eigenvalues
$\lambda_i>\texttt{rcond}\,\lambda_{\max}$, closed over degenerate multiplets, so
that $BB^\dagger=V^{+}$ (rank-truncation site `epsmat V+`,
[policy](../architecture/rank_truncation_policy.md#site-register)):

$$
\varepsilon^{-1}-1 = v\,YKY^\dagger,\qquad Y=ZB,\qquad K=B^\dagger(W-V)B .
\tag{E5}
$$

$Q=v^{1/2}Y$ has orthonormal columns ($Q^\dagger Q=B^\dagger VB=1$, $v\ge0$), so
the symmetrized $v^{-1/2}\varepsilon^{-1}v^{1/2}-1=QKQ^\dagger$ has the nonzero
spectrum of K. A stable static response has $0<\tilde\varepsilon^{-1}\le1$,
that is $-1\prec K\preceq0$, and the export prints the range of K per q.
Measured: K ∈ [−0.885, 4e-11] at q = 0 and [−0.886, −9e-12] at q = (0, 1/6, 0);
the positive 4e-11 is round-off at κ(V) = 6.2e7.

The amplification is $\kappa(V)=\lambda_{\max}/\lambda_{\min}$: an error δ in
W − V reaches K as at most $\delta/\lambda_{\min}$. On the same bundle κ(V) is
6.2e7 at q = 0 and 1.1e6 at q = (0, 1/6, 0), full rank at the default
`rcond` = 1e-8, so nothing is discarded. Where the cut binds, the dropped
directions carry $W-V=V\chi^{\mu}V$ with weight $\lambda_i\lambda_j$ and the
μ-space round trip misses O(`rcond`) of W − V (measured 6.8e-7 and 9.1e-6 at
`rcond` = 1e-6).

What a file can hold: ε⁻¹ − 1 has rank at most $N_\mu$ per q, range in the
columns of $vZ$ and co-range in those of $Z$; outside them ε⁻¹ = 1. That is
the ISDF ansatz (E1), not a property of the export.

## 3. Restriction to BerkeleyGW's G list

The file holds $|q+G|^2<E_\varepsilon$ (`epsilon_cutoff`, BerkeleyGW's
`ecuts`) with $E_\varepsilon\le E_c$ (`GATE epsmat_ecuts_beyond_coulomb`: beyond
$E_c$ LORRAX's $v$ and ζ̃ are zero). The written block is (E5) on those rows
and columns:

$$
\varepsilon^{-1}_{GG'}-\delta_{GG'} = v(q+G)\,\big[Y_\varepsilon KY_\varepsilon^\dagger\big]_{GG'},
\qquad Y_\varepsilon=Z_\varepsilon B,\qquad |q+G|^2,|q+G'|^2<E_\varepsilon .
\tag{E6}
$$

This is the block of the full-sphere inverse, $(\varepsilon^{-1})_{\varepsilon\varepsilon}$.
BerkeleyGW's epsilon inverts the block of ε, $(\varepsilon_{\varepsilon\varepsilon})^{-1}$,
with χ₀ evaluated on the eps sphere alone. The two differ by local fields beyond
$E_\varepsilon$: a screening difference between the codes, not a
representation error, and §6 keeps the two apart.

## 4. The reverse: BerkeleyGW ε⁻¹ onto ζ

$$
W^{\text{BGW}} = V + Z_\varepsilon^\dagger\,\big(\varepsilon^{-1}_{\text{BGW}}-1\big)\,v_\varepsilon Z_\varepsilon ,
\tag{E7}
$$

with V the bundle's (full bare sphere) and ε⁻¹ read block by block. A file q
equal to a stored q plus $G_0$ reads ζ̃ at $G+G_0$. Projecting an exported
block returns, for any G list,

$$
Z_\varepsilon^\dagger v_\varepsilon Y_\varepsilon KY_\varepsilon^\dagger v_\varepsilon Z_\varepsilon
= V_\varepsilon BKB^\dagger V_\varepsilon ,\qquad V_\varepsilon=Z_\varepsilon^\dagger v_\varepsilon Z_\varepsilon ,
$$

and the export checks this identity per q against the μ-space product
(`GATE epsmat_block_projection`, 1e-8). On the full sphere $V_\varepsilon=V$ and
the round trip returns W − V on the kept span: 1.0e-13 and 1.2e-14 on CrI3 6×6.
On a smaller sphere $\|V_\varepsilon BKB^\dagger V_\varepsilon-(W-V)\|/\|W-V\|$
is the part of W that `epsilon_cutoff` leaves out; it is printed per q.

`mode = project` copies the bundle to `output_dir/tmp`, replaces `W0_qmunu` by
(E7) at every stored q and stamps `W0_source = bgw_epsmat` with the reference
path; the source is never written (`GATE epsmat_project_output`). Both
directions take $v$ at the file's own q and G list, which for q ≠ 0 is V_q's
own $v$ (the same vector $q+G$) and at Γ is $v(q_0+G')$. At Γ the
G = 0 row and column of `eps0mat.h5` are dropped: the bundle's Γ body has
v(G = 0) = 0 and LORRAX carries the head as its rank-one `whead` term, which
is kept from the bundle (§5). Both modes write `epsmat_<mode>.json` in
`output_dir`: per q the gate figures, the range of K and, for `project`, the
Hermiticity of the projected BerkeleyGW W − V (a check of the block
orientation on BerkeleyGW's own data) and $\|K_{\text{BGW}}-K\|_2/\|K\|_2$.

## 5. q → 0

`eps0mat.h5` sits at $q_0$ (`q0_vector`, or the reference file's). Its three
parts are written differently.

* **Body** (G, G′ ≠ 0): (E6) with $\chi^{\mu}$ and ζ̃ of the bundle's Γ entry,
  whose V and W carry v(G = 0) = 0, so the body is $(\varepsilon_{bb})^{-1}$,
  and rows scaled by $v(q_0+G)$, the factor BerkeleyGW's
  $W=\varepsilon^{-1}v(q_0+G')$ pairs with; W is then Hermitian, and using
  Γ's χ and ζ̃ at $q_0$ is an O($q_0$) step. In 2D (slab) the body is
  the q → 0 limit: the head–wing term of the block inverse,
  $\varepsilon_{b0}\varepsilon_{00}^{-1}\varepsilon_{0b}$, is O(q). In 3D it
  is O(1) ($v(q)\propto q^{-2}$, $\chi_{0G}\propto q$) and the bundle's body
  omits it; the file inherits that approximation.
* **Wings**: written 0. They vanish as O(q) in 2D, and BerkeleyGW's Sigma
  zeroes them at $q_0$ for semiconductor screening under every truncation
  (`fixwings`).
* **Head**: the direct (no-local-field) head from the bundle's S tensor
  ([convention](s-tensor-convention.md)),

$$
\varepsilon^{-1}_{00}(q_0) = \frac{1}{1-v(q_0)\,q_0^{\mathsf T}Sq_0},
\tag{E8}
$$

  with $v(q_0)$ the truncated $8\pi/|q_0|^2$ in Ry bohr³, without 1/Ω: the
  owner's pointwise head (`gw.head_densify`) divided by the bare $v(q_0)$.
  Dividing v by Ω weakens the head by Ω; on CrI3 6×6 that gives
  $1-\varepsilon^{-1}_{00}=2.6\times10^{-6}$ against BerkeleyGW's
  $1.6\times10^{-2}$. BerkeleyGW's head includes local fields; the bundle's `head_correction` says which head
  LORRAX had (`no_local_fields` on CrI3 6×6). Sigma rebuilds its q → 0 head
  from this $\varepsilon^{-1}_{00}(q_0)$, so (E8) is what reaches Σ.
  `q0_head = unscreened` writes 1.

## 6. Comparing with BerkeleyGW's own ε⁻¹

Two differences coexist: the screening (bands, χ₀ evaluation, local fields
beyond $E_\varepsilon$, the ISDF basis) and the representation (§2–3). They
separate in two comparisons:

1. Low-|G| blocks of ε⁻¹, with head, wings and body reported apart: the head
   and wings are the model choices of §5, the body is the response.
2. W − V of BerkeleyGW's file projected by (E7), against the bundle's. Both now
   live in the same span, so the difference is the screening alone;
   $K_{\text{BGW}}-K$ in the whitened basis is its dimensionless form.

## 7. File conventions

| item | written | LORRAX side |
|---|---|---|
| header | `versionnumber` 3, `flavor` 2 (complex); `matrix_type` 0 (ε⁻¹), `nmatrix` 1, `freq_dep` 0, `nfreq` 1 | static W(0) only |
| Coulomb | `icutv` 0 (bulk) or 6 (slab); `vcoul` = 8π/\|q+G\|² × truncation, Ry bohr³ | $v=$ `vcoul`/Ω, Ry |
| G list | `ecuts` (Ry); ascending $\|q+G\|^2$ from `bdot`, ties by BerkeleyGW's `sortrx` comparator (tolerance 1e-12, then box index); `eps0mat` by $\|G\|^2$; `gind_eps2rho`, `gind_rho2eps` 1-based into `mf_header` | `file_io.bgw_epsmat.gvector_order` |
| matrix | `mats/matrix` (nq, 1, nfreq, nmtx_max, nmtx_max, 2) f8; HDF5 block [a, b] = ε⁻¹(G_b, G_a) | `EPSReader.get_eps_matrix` returns ε⁻¹(G, G′) |
| diagonal | `mats/matrix-diagonal` (nq, nmtx_max, 2); `qpt_done` set per q | |
| Fermi level | `efermi` = (max VBM + min CBM)/2 over the WFN, Ry (`find_efermi`) | `bgw_epsmat.midgap_ry` |
| mean field | `mf_header` copied verbatim from the WFN | |

Sigma's `epscopy` recomputes the kinetic energy of each q's first G and stops
above 1e-12, and maps rows through `gind` and the file's own `mf_header`; the
copied header and the verbatim sort make both hold by construction. A static
file needs Sigma `frequency_dependence` 0 or 1.

With `bgw_reference` set, the export takes the q list, $q_0$, `ecuts` and the G
order from BerkeleyGW's files and refuses unless its own order is identical
(`GATE epsmat_bgw_gorder`). Every file q must equal a stored q up to a
reciprocal lattice vector (`GATE epsmat_q_unmatched`): a symmetry-rotated W is
not reconstructed, so a wedge bundle exports its wedge and a full-zone bundle
(`restart_q_storage = full`) every q.

## 8. Cost and distribution

Per q, with n = `nmtx`, r ≤ $N_\mu$ the kept rank and P devices on a square
√P × √P mesh (`GATE epsmat_square_mesh`):

| step | flops per device | memory per device | collectives |
|---|---|---|---|
| $V^{+}$ factor, K | $N_\mu^3$ (replicated eigh) | 16 $N_\mu^2$ B each for V, W, B, K | — |
| ζ̃ onto the eps G list | — | 16 n $N_\mu$/P B | one all-to-all |
| $Y_\varepsilon=Z_\varepsilon B$ | n $N_\mu$ r/P | 16 n r/√P B | one all-gather of the row block over `y`, one transpose-partner permute |
| block tile, (E6) | n² r/P | 16 n²/P B | — |
| write | — | — | none: each device writes its tile through SlabIO |
| projection gate | n² r/P | — | one $N_\mu\times N_\mu$ all-reduce |

No collective moves the n × n block. On a 2 × 2 mesh the compiled export
program holds exactly one all-to-all (ζ̃), one all-gather and one
collective-permute of the row block, and one all-gather of the n-vector
diagonal, which leaves replicated because every process reads it on the host.
A file costs 16 n_max² bytes per q (BerkeleyGW pads every q to `nmtx_max`).

## 9. Not exported

* A frequency axis (`freq_dep` 2 or 3): the bundle holds W(0) only.
* Symmetry-rotated q (§7) and the local-field head (§5).
