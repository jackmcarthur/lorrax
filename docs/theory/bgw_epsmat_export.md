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
the ISDF ansatz (E1), not a property of the export: for pair densities of the
fitted band window, $\rho_{ij}\approx\sum_\mu\zeta_\mu C_{ij,\mu}$, matrix
elements of the written ε⁻¹ are LORRAX's own; for any other pair the file is
LORRAX's screening projected onto span ζ. On CrI3 6×6 that span has 638–640
directions per q: V's remaining eigenvalues sit at $|\lambda|\le4\times10^{-6}$
against $\lambda_{\max}\approx3\times10^{9}$, the null space of the
rank-truncated ζ, eight decades below the kept $\lambda_{\min}\approx500$.

## 3. Restriction to BerkeleyGW's G list {#ecut}

The file holds the G with $|q+G|^2\le E_\varepsilon$ (BerkeleyGW's `ecuts`,
`gcutoff`). The full sphere $E_{\rm full}$ is the `ecuts` of `bgw_reference`
when it is set, otherwise the producer's `bare_coulomb_cutoff` (`ecutwfc`
when that is unset, gw_init's own resolution of the V sphere). With
`bare_coulomb_cutoff` set, a reference `ecuts` above it refuses
(`GATE epsmat_ecuts_beyond_coulomb`: beyond $E_c$ LORRAX's $v$ and ζ̃ are
zero). `epsmat_ecut` (Ry) selects

$$
E_\varepsilon=\begin{cases}E_{\rm full}, & \texttt{epsmat\_ecut}=0,\\
\texttt{epsmat\_ecut}, & 0<\texttt{epsmat\_ecut}\le E_{\rm full},\end{cases}
\tag{E6a}
$$

and refuses above $E_{\rm full}$ (`GATE epsmat_ecut_beyond_sphere`) or below 0
(`GATE epsmat_ecut_value`). Each q's G list is first built on the full sphere
in BerkeleyGW's order and, with a reference, checked against the reference's
own (`GATE epsmat_bgw_gorder`, §7). A smaller $E_\varepsilon$ then keeps its
leading $n_\varepsilon(q)=$ `gcutoff`$(E_\varepsilon)$ entries, so `gind_eps2rho`
of the cut file is the first $n_\varepsilon$ entries of the full file's. The
written block is (E5) on those rows and columns:

$$
\varepsilon^{-1}_{GG'}-\delta_{GG'} = v(q+G)\,\big[Y_\varepsilon KY_\varepsilon^\dagger\big]_{GG'},
\qquad Y_\varepsilon=Z_\varepsilon B,\qquad |q+G|^2,|q+G'|^2\le E_\varepsilon .
\tag{E6}
$$

$K$ and $B$ are the full-sphere ones (§2); nothing is re-inverted. The file is
therefore the leading sub-block of the full-sphere inverse,
$(\varepsilon^{-1})_{\varepsilon\varepsilon}$, entry for entry: on Si 4³
(full sphere 25 Ry, nmtx 560–588; cut 12 Ry, nmtx 178–200 and 181 at q₀) the
cut file equals the full file's leading block to 1.0 × 10⁻¹⁵.
`tests/test_bgw_epsmat_cpu.py::test_epsmat_ecut_is_the_leading_sub_block_not_a_reinversion`
holds the writer to that and fails the re-inversion on the cut G set,
$(\varepsilon_{\varepsilon\varepsilon})^{-1}$.

BerkeleyGW's epsilon at a smaller `epsilon_cutoff` inverts the block of ε,
$(\varepsilon_{\varepsilon\varepsilon})^{-1}$, with χ₀ evaluated on the eps
sphere alone. The two differ by local fields beyond $E_\varepsilon$: a
screening difference between the codes, not a representation error, and §6
keeps the two apart.

What a smaller file does downstream depends on the BerkeleyGW code that reads
it. `kernel.x` (BerkeleyGW 4.0) takes every G of the file
(`BSE/mtxel_kernel.f90`: `neps = xct%nmtxa(irq)`, `epscol(1:neps)`) and
contracts the G beyond it with the bare $v$
(`calc_direct_unscreened_contrib`). Its `screened_coulomb_cutoff` sets
`xct%neps`, which enters only `nepsmin = max(xct%neps, neps)` and never
shortens the file's list. A cut file is therefore the way to truncate the
screening seen by `kernel.x`; a full file with `screened_coulomb_cutoff` is
not. On Si 4³ (8v × 8c, 64 k) the full file with and without
`screened_coulomb_cutoff 12` gives bitwise identical kernels, and the 12 Ry cut
file moves the wing and body by 9.4 × 10⁻⁴ and 6.2 × 10⁻⁴ (relative
Frobenius) and the lowest 40 absorption eigenvalues by 0.009 meV on average,
0.078 meV at most (0.007 % of the QP gap, 1.080 eV). Sigma, unlike
`kernel.x`, truncates at its own `screened_coulomb_cutoff`.

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
is the part of W that the cut leaves out (`epsmat_ecut`, §3); it is printed per q.

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

`eps0mat.h5` sits at $q_0$ (`q0_vector`, or the reference file's). In the
{G = 0, body} partition the dielectric matrix at $q_0$ is

$$
\varepsilon=\begin{pmatrix}1-v_0\chi_{00} & -v_0\chi_{0b}\\ -v_b\chi_{b0} & 1-v_b\chi_{bb}\end{pmatrix},
\qquad v_0=v(q_0)/\Omega,\; v_b=v(q_0+G)/\Omega ,
\tag{E8a}
$$

with the body $\chi_{bb}=Z\chi_0^{\mu}Z^\dagger$ of §1 (Γ's ζ̃ and χ, rows at
$v(q_0+G)$: an O($q_0$) step) and BerkeleyGW's $W=\varepsilon^{-1}v(q_0+G')$.
The bundle's Γ body has $v(G=0)=0$, so its χ^μ gives $(\varepsilon_{bb})^{-1}$
and nothing about the head or the wings.

**The wings L.** At q → 0 the G = 0 row and column of the polarizability are
linear in q. GW forms them where it forms the head (`qsgw_head.head_wings_sharded`,
the Schur fold's operands) and stores their ω = 0 values beside `S_cart_head`
(`head_wing_L` (3, N_μ), `head_wing_R` (N_μ, 3), `S_cart_head_direct`):

$$
\chi_0(0,G)=-\sum_\mu (q\!\cdot\!L)_\mu\,\overline{\tilde\zeta_\mu(G)},\qquad
\chi_0(G,0)=-\sum_\mu \tilde\zeta_\mu(G)\,(Rq)_\mu,\qquad
\chi_{00}=\Omega\,q^{\mathsf T}S_{\rm d}\,q,
$$
$$
L_{a\mu}=\sum_{k,c,v}\overline{v^a_{cv,k}}\;\frac{4}{N_k n_s n_{\rm spinor}\,\Delta E_{cvk}^2}\;
\psi^*_{ck}(r_\mu)\psi_{vk}(r_\mu),\qquad R=L^\dagger\ (\omega=0),
\tag{E9}
$$

q Cartesian in 1/bohr, $v$ the velocity (with the V_NL commutator) that builds
$S$. Replacing the centroid pair product by the density jet $D=v/\Delta E$
gives $L=-\Omega S_{\rm d}$: the identity that fixes $g$ and the conjugation
(`tests/test_epsmat_q0_wing_cpu.py`, with a red twin). The stored head is the
fold $S_{\rm eff}=S_{\rm d}+L\,W(\Gamma)\,R/\Omega$ (`head_correction = full`,
`head_source = head_schur`), which no overall sign of L can change. The sign
of the q-linear G = 0 density against ζ̃'s Fourier convention — the minus in
the first two relations — is fixed on BerkeleyGW's own eps0mat: with the
opposite sign every wing entry of Si 4³ (centrosymmetric) and MoS₂ 9×9 (not)
is −1.0000 × BerkeleyGW's (residual 1.4e-3 on the MoS₂ row wing), and the
conjugated assignment fails (residual 1).

Write $\omega=-Rq_0$, $\tilde\omega=-q_0\!\cdot\!L$ and, with the export's Γ
factors $V$, $\chi^{\mu}=BKB^\dagger$ and $W=V+V\chi^{\mu}V$,

$$
u=(1+\chi^{\mu}V)\,\omega,\qquad \tilde u^{\mathsf T}=\tilde\omega^{\mathsf T}(1+V\chi^{\mu}).
\tag{E10}
$$

Block inversion of (E8a), using $\varepsilon_{bb}^{-1}=1+v_bZ\chi^{\mu}Z^\dagger$ and
$Z^\dagger v_bZ=V$, gives every piece from μ-space vectors:

$$
\varepsilon^{-1}_{00}=\frac1{\varepsilon_M},\qquad
\varepsilon_M=1-v_0\big(\chi_{00}+\tilde\omega^{\mathsf T}W\omega\big)
=1-v(q_0)\,q_0^{\mathsf T}S_{\rm eff}\,q_0 ,
\tag{E11}
$$
$$
\varepsilon^{-1}(G,0)=\frac{v_b(G)\,[Zu](G)}{\varepsilon_M},\qquad
\varepsilon^{-1}(0,G')=\frac{v_0}{\varepsilon_M}\,[\tilde u^{\mathsf T}Z^\dagger](G'),
\tag{E12}
$$
$$
\varepsilon^{-1}_{bb}-1=v_b\,Z\big(\chi^{\mu}+c\,u\tilde u^{\mathsf T}\big)Z^\dagger,\qquad c=\frac{v_0}{\varepsilon_M}.
\tag{E13}
$$

(E11) is the local-field head, $1/(\varepsilon_{00}-\varepsilon_{0b}\varepsilon_{bb}^{-1}\varepsilon_{b0})$;
(E13) is $\varepsilon_{bb}^{-1}+\varepsilon_{bb}^{-1}\varepsilon_{b0}\,\varepsilon^{-1}_{00}\,\varepsilon_{0b}\varepsilon_{bb}^{-1}$,
the same as $(\varepsilon_{bb}-\varepsilon_{b0}\varepsilon_{00}^{-1}\varepsilon_{0b})^{-1}$.
Orders in $q_0$: $v_0\propto q_0^{-2}$ (3D) or $q_0^{-1}$ (slab), $u,\tilde u\propto q_0$, so the
column wing is O($q_0$), the row wing O($q_0^{-1}$) in 3D and O(1) in a slab, and the body
term $c\,u\tilde u^{\mathsf T}$ is O(1) in 3D and O($q_0$) in a slab. Every step is a face
mat-vec or an elementwise face program, O($N_\mu^2/P$) per $q_0$ (`epsmat_export.q0_dipole_terms`).

`q0_wings` selects the file:

* **`dipole`** (default): head (E11), wings (E12), body (E13). GATE
  `epsmat_q0_wing_fold` re-forms the fold with the FILE's W(Γ) and refuses
  unless $\Omega q_0^{\mathsf T}(S_{\rm eff}-S_{\rm d})q_0=\tilde\omega^{\mathsf T}W\omega$
  to 1e-3 of the fold term, which certifies L's μ order and normalisation where
  they are consumed. A bundle without stored wings refuses
  (`epsmat_q0_wings_source`): rerun its GW on this branch, or set `zero`.
* **`zero`**: today's file before the wings. Wings 0; head
$$
\varepsilon^{-1}_{00}(q_0)=\frac{1}{1-v(q_0)\,q_0^{\mathsf T}Sq_0}
\tag{E8}
$$
  from the stored `S_cart_head` ($v(q_0)$ the truncated $8\pi/|q_0|^2$ in Ry bohr³,
  without 1/Ω). With a `head_schur` bundle $S=S_{\rm eff}$ and (E8) already IS the
  local-field head (E11). The body is $(\varepsilon_{bb})^{-1}$: in 2D the q → 0 limit, in 3D missing
  the O(1) term of (E13), so a `sys_dim = 3` export refuses (`GATE epsmat_q0_body_3d`)
  unless `q0_body_3d = omit_head_wing`. `q0_head = unscreened` writes 1.

Scope: L is the q → 0 object only — the eps0mat wings, the local-field head and the
3-D q0 body here, and the k = k′ (q = 0) blocks of a BSE kernel file. Head, wing and
body of W at q ≠ 0 come from the stored V, W and ζ̃ alone and need no L.

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

Per q, with n = `nmtx` and μ = $N_\mu$, each padded to a multiple of P, on P
devices in a square √P × √P mesh (`GATE epsmat_square_mesh`). Every matrix
is a `P(None,'x','y')` face: device (x, y) holds one (·/√P) × (·/√P) tile of
V, W − V, B, K, χ, ζ̃ on the eps G list (μ × n), the block (n × n), V_ε and
the projection. Only vectors (v, eigenvalues, the block diagonal) and scalars
are replicated. Products are `distrib_la.panel_matmul` (2-D SUMMA; the two
live operand panels hold at most `PANEL_BYTES` = 64 MiB per device).

| step | flops per device | memory per device | collectives |
|---|---|---|---|
| V, W − V | — | 16 μ²/P B each | none (SlabIO reads the tile) |
| $V^{+}$ factor (eigh of V), K spectrum | ≈ μ³/P | 16 μ²/P B + solver workspace | inside the solver |
| K = Bᴴ(W − V)B, χ = B K Bᴴ | 4 μ³/P | 16 μ²/P B each | SUMMA panels over `x` and `y` |
| ζ̃ onto the eps G list | — | 16 n μ/P B | one all-to-all over `y` |
| block $T_1=\bar Z^{\mathsf T}\chi^{\mathsf T}Z$, (E6) | (μ² n + μ n²)/P | 16 n²/P B | SUMMA panels |
| write | — | — | none: each device writes its tile through SlabIO |
| projection and its gate | (2 μ² n + μ n² + 2 μ³)/P | 16 μ²/P B | SUMMA panels; scalar norms |

The eigensolve is a `distrib_la` plan on the face: `distributed`
(cuSOLVERMp on CUDA, ScaLAPACK on CPU) when each device has its own process,
the production launch; `native` in one process, where the solve gathers the
16 μ² B matrix on every device (the run banner prints that figure).
`distrib_la`'s `auto` is native everywhere and is never requested. On CrI3
6×6 (μ = 980, P = 4) a face is 3.8 MB per device; at μ = 2.7 × 10⁴ it is
2.9 GB at P = 4 and 0.73 GB at P = 16, against 11.7 GB per device for one
replicated matrix. The n × n block is produced in place; the projection
streams it as one SUMMA operand in panels. A file costs 16 n_max² bytes per q
(BerkeleyGW pads every q to `nmtx_max`).

Changing the eigensolver moves the export at the conditioning of V, not at
round-off. On the CrI3 6×6 bundle the face path (cuSOLVERMp) and the
replicated path it replaced (one cuSOLVER solve per device, V's lower
triangle) write blocks that differ by ≤ 3.1 × 10⁻⁹ (Frobenius, relative to
ε⁻¹ − 1) at κ(V) = 5.7–7.6 × 10⁶ on the kept span (κ ε = 1.3–1.7 × 10⁻⁹), with
the same kept rank at every q. The projection, which solves nothing, agrees
to 1.5 × 10⁻¹⁴.

## 9. Not exported

* A frequency axis (`freq_dep` 2 or 3): the bundle holds W(0) only.
* Symmetry-rotated q (§7).
