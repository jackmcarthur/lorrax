# On-site Breit operator: design, conventions and the sidecar generator

*Owner page of the on-site four-spinor Breit term, `onsite_breit = true`.
The input keys are listed in the [input reference](../input_reference.md); the
DFT-level driver `gw.onsite_breit_cli` is described in
[drivers](../drivers.md#on-site-breit-gwonsite_breit_cli). The code cites this
page by section number: §2 conventions, §3 double counting, §4 drivers, §6
files, §7 generator, §8 what is not included.*

Code: `src/psp/onsite_breit.py` (physics library: sidecar reader, shell algebra,
projector, kernels), `src/psp/onsite_breit_sidecar.py` (the sidecar generator),
`gw.hartree.onsite_breit_field` (the one assembly for every caller),
`src/file_io/onsite_breit_output.py` (debug file). Tests:
`tests/test_onsite_breit.py`, `tests/test_onsite_breit_field_cpu.py`,
`tests/test_onsite_breit_sc_seam_cpu.py`, `tests/test_onsite_breit_sidecar.py`,
`tests/test_onsite_breit_hartree_pairing_cpu.py`.

## 1. What is added {#what}

$$
H = T + V_\mathrm{loc} + V_\mathrm{NL} + V_H[\rho] + \alpha\,\mathbf A[\mathbf J] + H_\mathrm{OS}[D] + \Sigma_{xc}
$$

$H_\mathrm{OS}[D]$ is the Hartree–Fock mean field of the transverse (Breit)
two-body interaction between the four-spinors of one open $(n\,l)$ shell per
atom, built from the on-site density matrix $D$ of the current orbitals with
**all-electron** radial functions. Each shell function is lifted to a
four-spinor exactly as a bispinor run lifts its orbitals
(`common.bispinor_init`, the raw kinetic-balance lift), and the interaction
$-\boldsymbol\alpha_1\cdot\boldsymbol\alpha_2/r_{12}$ acts between the lifted
spinors. No Pauli sub-operator is formed: spin–other-orbit, orbit–orbit and
spin–spin are all parts of the one current–current integral, and none is
added or dropped by hand.

The smooth transverse direct field $\alpha\,\mathbf A[\mathbf J]$ of a
bispinor run is the same interaction on pseudo-orbitals, which carry only part
of the one-centre transverse integrals: every such integral of a $3d$ shell
lies inside the pseudization sphere, where norm conservation does not fix the
radial distribution of the current. Because the on-site operator uses the same
carrier and the same kernel, its pseudo part $h_\mathrm{PS}$ is the one-centre
part of what the smooth field computes, and $h_\mathrm{AE}-h_\mathrm{PS}$ is a
PAW-style completion to the all-electron value. In a run without the smooth
field the on-site term supplies the whole one-centre term.

The frozen-core Gaunt term is not part of this operator: it is carried by the
UPF itself (a j-resolved $\Delta D$ in PP_DIJ), and a j-averaged use of such a
file refuses (`GATE gaunt_upf_j_averaged`, `tests/test_gaunt_upf_gate.py`).

## 2. Operator and conventions {#conventions}

Hartree atomic units inside the tensors, $\alpha = 1/c$ =
`common.bispinor_init.ALPHA_FS`; Rydberg for every potential and band matrix
the code returns ($\times 2$); meV only in the log
($10^3\,$`common.units.RYD_TO_EV` per Ry).

**Shell basis.** $|a\rangle = \tilde\phi_{I\mu}(\mathbf r)\otimes|\sigma\rangle$
on atom $I$: complex $Y_{lm}$ (Condon–Shortley), $m=-l\dots l$, pure spinors,

$$
a = \sigma\,(2l+1) + (m+l),
$$

$\sigma = 0$ the first spinor component of the WFN ($s_z = +\tfrac12$ along
Cartesian $z$ of the cell). Every on-site matrix that is logged or written is
in this basis.

**Projector** (input key `onsite_breit_projector`; one type, `ortho-atomic`).
The j-averaged PP_PSWFC functions of the UPFs,
$\chi_l = [(l+1)\chi_{l+1/2} + l\,\chi_{l-1/2}]/(2l+1)$
(`psp.hubbard_ops.j_averaged_atomic_wfcs`), Löwdin-orthogonalised at each $k$
over all atomic functions of all atoms (QE ortho-atomic, the DFT+U projector:
`build_atwfc_setup`, `lowdin_rows_with_derivative`). The rows are held in QE's
real $Y_{lm}$; the shell basis is reached per spin block by
$P^c = W^\dagger P^r$ with $Y^c_\mu = \sum_i W_{i\mu} Y^r_i$, $W$ fitted
numerically from `psp.radial.solid_harmonics` (no ordering or phase
assumed). A projector type is an entry of `psp.onsite_breit._PROJECTORS`;
everything after the projection table is projector-agnostic.

$$
\begin{aligned}
P_{kIan} &= \langle\tilde\phi_{Ia}(k)|\psi_{nk}\rangle &&\text{projection table, full BZ}\\
D_{I,ab} &= \textstyle\sum_k w_k \sum_i f_{ik}\,(PU)_{ai}\,(PU)^*_{bi} &&\text{on-site density matrix } (U=1 \text{ at DFT level})\\
h_I &= h^\mathrm{dir}[D_I] + h^\mathrm x[D_I] &&\text{combined as in §3}\\
H_{\mathrm{OS},k,nn'} &= \textstyle\sum_I\sum_{ab} P^*_{kIan}\,h_{I,ab}\,P_{kIbn'} &&\text{band matrix (Ry)}
\end{aligned}
$$

$U_{k,ni} = \langle \mathrm{DFT}_n|\mathrm{QP}_i\rangle$ is the map's rotation
inside the QP window, $f$ its occupations, $w_k = 1/N_k$ on the full BZ.
$D_{ab} = \langle a|\hat\rho|b\rangle$; QE's `ns` is its transpose. $P$ is
computed from the two-component DFT $\psi(\mathbf G)$ unfolded by the loader
to the full BZ, so $D$ is the plain zone sum and needs no second symmetry
action.

**Lifted spinors and the two-body tensor.** With one radial function $u = rR$
per shell (the j-average, §7), $\phi_p = (u/r)\,Y_{lm}\,\chi_\sigma$ in the
shell basis above, the lifted four-spinor and the transition current are

$$
\Psi_p = \begin{pmatrix}\phi_p\\[2pt] \tfrac{\alpha}{2}\,\boldsymbol\sigma\cdot\mathbf p\,\phi_p\end{pmatrix},\qquad
\mathbf j_{pr}(\mathbf r) = \Psi_p^\dagger(\mathbf r)\,\boldsymbol\alpha\,\Psi_r(\mathbf r),
$$

$\boldsymbol\alpha$ the Dirac matrices. The two-body tensor is

$$
G_{pqrs} = \Big\langle p(1)\,q(2)\Big|\,-\frac{\boldsymbol\alpha_1\cdot\boldsymbol\alpha_2}{r_{12}}\Big|r(1)\,s(2)\Big\rangle
= -\iint \frac{\mathbf j_{pr}(\mathbf r_1)\cdot\mathbf j_{qs}(\mathbf r_2)}{r_{12}}\,d^3r_1\,d^3r_2 ,
$$

written to and read from the sidecar without the $\alpha^2$ that the two
currents carry. For one radial function every $\mathbf j_{pr}$ is
divergence-free, so the Gaunt kernel and the transverse (Breit) kernel give
the same $G$. It has the pair symmetry $G_{pqrs} = G_{qpsr}$ and is Hermitian,
$G_{pqrs} = G^*_{rspq}$.

**Mean field.**

$$
h^\mathrm{dir}[p,r] = \alpha^2\sum_{qs} G_{pqrs}\,D_{sq},\qquad
h^\mathrm{x}[p,r] = -\alpha^2\sum_{qs} G_{pqsr}\,D_{sq}
$$

(Hartree; `psp.onsite_breit.meanfield_kernels` precontracts them). These are
the only two pieces; the potential pieces `dir_ae`, `dir_ps`, `x_ae`, `x_ps`
of §3 are $h^\mathrm{dir}$ and $h^\mathrm{x}$ of the AE and the PS radial
function.

**Separation.** With $R = u/r$ the current separates as
$\mathbf j_{pr} = \alpha\,[R R'\,\mathbf F_{pr}(\Omega) + (R^2/r)\,\mathbf H_{pr}(\Omega)]$,
where, with $\Phi_p = Y_{lm}\chi_\sigma$, $A = (\boldsymbol\sigma\cdot\hat{\mathbf r})\Phi$
and $B = (\boldsymbol\sigma\cdot\nabla_\Omega)\Phi$,

$$
\mathbf F_{pr} = \tfrac{i}{2}\big[A_p^\dagger\boldsymbol\sigma\Phi_r - \Phi_p^\dagger\boldsymbol\sigma A_r\big],\qquad
\mathbf H_{pr} = \tfrac{i}{2}\big[B_p^\dagger\boldsymbol\sigma\Phi_r - \Phi_p^\dagger\boldsymbol\sigma B_r\big].
$$

The two radial shapes are $f_1 = u u' - u^2/r$ (with $\mathbf F$) and
$f_2 = u^2/r$ (with $\mathbf H$). Their Slater integrals
$W_L[a,b] = \iint f_a(r_1) f_b(r_2)\, r_<^L/r_>^{L+1}\,dr_1\,dr_2$ lose every
derivative of $u$ under integration by parts:

$$
W_L[1,1] = -\tfrac{(L-1)(L+2)}{2}\,N_L + \tfrac{2L+1}{4}\,K,\qquad
W_L[1,2] = W_L[2,1] = -\tfrac32\,N_L,\qquad
W_L[2,2] = 2\,N_L,
$$

$$
N_L = \iint_{s<r} \frac{u^2(r)\,u^2(s)\,s^{L-1}}{r^{L+2}}\,ds\,dr\quad (L = 0\dots 2l+1),\qquad
K = \int_0^\infty \frac{u^4}{r^2}\,dr ,
$$

and

$$
G_{pqrs} = -\sum_{L=0}^{2l+1} \frac{4\pi}{2L+1} \sum_{M,c,t,t'} W_L[t,t']\;
a^{t,c}_{pr,LM}\, b^{t',c}_{qs,LM},\qquad
a^{t,c}_{pr,LM} = \int X^{t,c}_{pr}\,Y_{LM}\,d\Omega,\quad
b^{t',c}_{qs,LM} = \int X^{t',c}_{qs}\,Y^*_{LM}\,d\Omega,
$$

$X = (\mathbf F, \mathbf H)$, the angular integrals exact on a Gauss–Legendre
$\times$ uniform-$\varphi$ rule above the degree $4l+2$
(`psp.onsite_breit.shell_breit_tensor`). A shell's whole tensor is fixed by
the $2l+3$ radial numbers $N_0,\dots,N_{2l+1}, K$ (bohr$^{-3}$); $N_1 = I^0$,
Marvin's

$$
I^0 = \int_0^\infty\! dr_2\, u^2(r_2)\, r_2^{-3} \int_0^{r_2}\! dr_1\, u^2(r_1).
$$

**Limits** (unit tests, `tests/test_onsite_breit.py`,
`tests/test_onsite_breit_sidecar.py`): a closed shell $D = 1$ gives
$h^\mathrm{dir} = 0$ and $h^\mathrm x = a\,\mathbb 1 + c\,\mathbf l\cdot\mathbf s$
with $c > 0$ (the exchange of a closed shell raises $\xi$; $c$ equals the
spin–other-orbit value of the same radial function, the orbit–orbit exchange
adds only the constant $a$); a spherical collinear shell gives an
$\mathbf l\cdot\hat{\mathbf m}$ coefficient of $h^\mathrm{dir}$ equal to
$b = -\tfrac{\alpha^2}{2}(N_\uparrow - N_\downarrow)\,I^0$, $b<0$ (the direct
term lowers $\xi_\mathrm{eff} = \xi + 2\sigma b$ for majority carriers); one
electron gives $h^\mathrm{dir}_{aa} + h^\mathrm{x}_{aa} = 0$;
$h[\Theta D\Theta^{-1}] = \Theta h[D]\Theta^{-1}$;
$h[MDM^\dagger] = M h[D] M^\dagger$ for $M = U_\mathrm{spin}\otimes D^l(R)$.

**The spin–other-orbit operator is contained exactly.** Write
$h = h_0\otimes\mathbb 1 + \sum_c h_c\otimes\sigma_c$ and let
$g = -\alpha^2\,\boldsymbol\Lambda(1;2)\cdot\mathbf s(2)$,
$\boldsymbol\Lambda(1;2) = \mathbf r_{12}\times\mathbf p_1/r_{12}^3$, be the
Breit–Pauli spin–other-orbit operator, with direct pieces
$d_1[p,r] = \sum_{qs} g_{pqrs}D_{sq}$ (orbit of the test electron in the
shell's spin field) and $d_2[p,r] = \sum_{qs} g_{qpsr}D_{sq}$ (spin of the test
electron in the shell's orbital current). For a pure orbital source
$D = \mathbb 1\otimes D_0$ the spin-dependent part $h_c$ of $h^\mathrm{dir}$ is
that of $d_2$; for a pure spin source $D = \sum_c \sigma_c\otimes D_c$ the
spin-independent part $h_0$ of $h^\mathrm{dir}$ is that of $d_1$ (both to
$10^{-13}$ relative on the shipped inputs). The rest of $h^\mathrm{dir}$, the
spin-independent response to the orbital source and the spin-dependent
response to the spin source, is orbit–orbit and spin–spin. The
spin–other-orbit tensor is built independently by the generator for this
check only (§7); no run reads it.

## 3. Double counting {#double-counting}

Two input keys, both required when `onsite_breit = true`, no defaults.

| `onsite_breit_double_counting` | direct part of $h_I$ | requires | meaning |
|---|---|---|---|
| `onsite_only` | $h^\mathrm{dir}_\mathrm{AE}[D]$ | `bispinor = false` | no smooth four-current term in the run; the on-site model is the whole direct term |
| `paw` | $h^\mathrm{dir}_\mathrm{AE}[D] - h^\mathrm{dir}_\mathrm{PS}[D]$ | `bispinor = true` | smooth $\alpha\,\mathbf A$ (all space, pseudo) plus the on-site AE − PS completion |

| `onsite_breit_exchange` | exchange part of $h_I$ | use |
|---|---|---|
| `full` | $h^\mathrm x_\mathrm{AE}[D]$ | no transverse exchange on pseudo-orbitals anywhere in $H$: DFT level, scalar GW |
| `paw` | $h^\mathrm x_\mathrm{AE}[D] - h^\mathrm x_\mathrm{PS}[D]$ | $\Sigma$ carries the bare transverse (TT) exchange on pseudo-orbitals: bispinor GW with `bispinor_transverse_exchange = true` |
| `off` | $0$ | direct term only |

The wrong pairing refuses (`GATE onsite_breit_keys`, one owner:
`psp.onsite_breit.resolve_onsite_keys`): `onsite_only` with `bispinor = true`
counts the direct term twice; `paw` with `bispinor = false` subtracts a pseudo
term that is not there; `exchange = paw` where $\Sigma$ has no TT exchange
(the DFT-level driver, or `bispinor_transverse_exchange = false`) and
`exchange = full` where it has one refuse likewise. A `paw` key needs the
sidecar's `ps` block.

**Not validated:** that the on-site part of $\Sigma$'s TT exchange equals
$h^\mathrm x_\mathrm{PS}$. Only the direct-term premise was checked (the
on-site PS model against the run's own smooth field). `exchange = paw` is
wired on that assumption and the log says so.

## 4. Where it enters each driver {#drivers}

| driver | change |
|---|---|
| `gw.kin_ion_io`, `psp.get_dipole_mtxels` | none. The velocity has no $i[\mathbf r, H_\mathrm{OS}]$ term (§8). |
| `gw.onsite_breit_cli` (DFT level) | $D$ from the DFT occupations, $h$, $H_\mathrm{OS}$ by piece; under `paw` the smooth $\alpha\,\mathbf A$ from `gw.hartree.direct_field_matrices`; applied once ($\mathrm{eigh}$ of $\mathrm{diag}(E_\mathrm{DFT}) + \Delta H$ per $k$), $D$ re-measured; writes `onsite_breit.h5`. |
| `gw.gw_jax`, one shot | `sigma_dispatch.onsite_breit_into_transverse`: after the live direct field, $H_T \leftarrow H_T + H_\mathrm{OS}[D_\mathrm{DFT}]$ (static); on a scalar input file the slot holds $H_\mathrm{OS}$ alone. |
| `gw.gw_jax`, self-consistent | `sc_iteration.rebuild_hartree_dft_basis`: after the transverse Hartree rebuild, $D$ from this map's $(U, f)$, $H_\mathrm{OS}$, $H_T \leftarrow H_T + H_\mathrm{OS}$. `kin_ion_dft` stays pristine, so nothing is counted twice. Requires `density_self_consistent = true`. |

One function owns the assembly for all three callers
(`gw.hartree.onsite_breit_field`). In the outputs the on-site matrix rides
the transverse direct-field slot (`h_transverse_kij`,
`hartree_transverse_kij_ev`), stamped, and is written separately in the debug
file.

**Self-consistency.** Map $i$: $H_i\to(E,U,f)\to\rho_i,\mathbf J_i,D_i$ from
the same orbitals, $H_{i+1} = \dots + H_\mathrm{OS}[D_i]$. Straight feedback,
no mixing, like $V_H$. $D$ is restricted to the QP window like $\rho$. The
radial integrals and the projector are fixed; only $D$ moves. There is no
energy functional, so no total energy.

## 5. Input keys {#keys}

`onsite_breit`, `onsite_breit_files`, `onsite_breit_double_counting`,
`onsite_breit_exchange`, `onsite_breit_projector`, `onsite_breit_pseudo_dir`:
types, defaults and refusals in the [input reference](../input_reference.md).
`onsite_breit_files` names one sidecar per species that carries the term
(input-file-relative); `onsite_breit_pseudo_dir` is where the UPFs with
PP_PSWFC are, one for every species of the WFN (the Löwdin runs over all
atoms). Debug output rides `sigma_lorentz_debug_output`.

## 6. Files {#files}

### The sidecar `<El>.onsite_breit.json` {#sidecar}

Format `lorrax.onsite_breit_sidecar/v2`, one per species, written by
`python -m psp.onsite_breit_sidecar` (§7). JSON; every float is written with
Python `repr` (an exact float64 round trip). No tensor is stored: the reader
rebuilds $G$ from the radial numbers of each block (§2).

| field | content |
|---|---|
| `format` | `"lorrax.onsite_breit_sidecar/v2"` |
| `element` | element symbol (from the UPF's PP_HEADER) |
| `conventions` | text: basis, lift, $G$, the radial numbers and the $W_L$ formulas, units, mean-field formulas (§2) |
| `shells` | list of shell objects (the generator writes one) |
| `provenance` | `generator`, `date`, `code_sha256`, `source_upf` {`path`, `md5`, `sha256`}, `ae_source`, `quadrature`, `checks` {`ae`, `ps`: the §7 check values}; `null_gate` in a null sidecar. |

A `lorrax.onsite_breit_sidecar/v1` file (the earlier spin–other-orbit tensor
$\Lambda$, fields `shape`, `lam_re`, `lam_im`) refuses at load
(`GATE onsite_breit_sidecar`): it holds the spin–other-orbit tensor only and
not the radial numbers of the four-spinor operator. Regenerate it with
`python -m psp.onsite_breit_sidecar` from the same inputs (UPF, shell label,
all-electron radial function).

Shell object:

| field | content |
|---|---|
| `label`, `l` | PP_CHI label (`"3D"`; matched case-insensitively) and $l$ ($1\le l\le 3$) |
| `projector.type` | `"ortho-atomic"`; `"none"` only in the operator-algebra test fixtures |
| `projector.function`, `.construction` | the j-average formula; which PP_CHI entered with which weight |
| `projector.mesh_size`, `.r`, `.chi` | the UPF's PP_R mesh (bohr) and the **plain** j-average $\chi_l = rR$ of `psp.hubbard_ops.j_averaged_atomic_wfcs` (no sign alignment, no renormalisation) |
| `ae`, `ps` | tensor blocks; `ps` may be `null` (then only `onsite_only` / `full` / `off`) |

Tensor block (`ae`, `ps`):

| field | content |
|---|---|
| `reference` | what the radial function is (configuration, source) |
| `N` | $[N_0, \dots, N_{2l+1}]$ of the normalised radial function (§2), bohr$^{-3}$ |
| `K` | $\int u^4/r^2\,dr$, bohr$^{-3}$ |
| `I0`, `r3` | diagnostics: Marvin $I^0 = N_1$, $\langle r^{-3}\rangle$ (bohr$^{-3}$) |
| `b1_mev_per_muB` | $-\tfrac{\alpha^2}{2} I^0$ in meV: the $\mathbf l\cdot\hat{\mathbf m}$ coefficient of $h^\mathrm{dir}$ per unit $N_\uparrow - N_\downarrow$ of a spherical collinear shell |
| `x_closed_ls_mev`, `x_closed_shift_mev` | $c$ and $a$ of $h^\mathrm x = a\,\mathbb 1 + c\,\mathbf l\cdot\mathbf s$ at $D = 1$ (meV) |

**Load-time checks** (`GATE onsite_breit_sidecar`, `psp.onsite_breit.read_sidecar`
and `build_onsite_setup`): the format string (v1 refuses, above);
$1 \le l \le 3$; per block the length of `N` ($2l+2$), $N_L > 0$, $K > 0$ and
`I0` $= N_1$ (the file's integrity); the closed-shell identity of the tensor
rebuilt from `N`, `K` ($\max|h^\mathrm{dir}| \le 10^{-10}\max|h^\mathrm x|$
and $h^\mathrm x = a\,\mathbb 1 + c\,\mathbf l\cdot\mathbf s$ to $10^{-10}$
relative with $c>0$: it holds for any radial numbers, so it tests the angular
construction of the code that reads the file); `ae` present, `ps` present
when a key says `paw`; the element is a species of the WFN; the projector type
equals the input key; and `projector.chi` equals the j-average of the run's
UPF on the same mesh to $10^{-8}\max|\chi|$ (a Gaunt-corrected UPF keeps
PP_PSWFC and passes). The sidecar's SHA-256 is stamped into every output.

**The null sidecar** (`--null`): the `ae` block is a copy of the `ps` block,
so $h_\mathrm{AE} - h_\mathrm{PS} = 0$ exactly in every `paw` combination. A
run with it must reproduce the run without the term bit for bit; it gates
the wiring, the projector and the outputs before any physics is read.

### Debug output and log {#debug}

`onsite_breit.h5` (CLI) and `onsite_breit_iterNNNN.h5` (self-consistent
loop, beside `sigma_lorentz_iterNNNN.h5`), format
`lorrax.onsite_breit_debug/v1`: the matrix that entered $H$, the shell-basis
$D$ and $h$ by piece, moments, shell-atom labels and provenance attributes.
The layout is the docstring of `src/file_io/onsite_breit_output.py`.

The log (rank 0, each build) gives per atom and shell $\mathrm{Tr}\,D$,
$2\langle\mathbf S\rangle$, $\langle\mathbf L\rangle$ (Cartesian),
$\max|D - D^\dagger|$, the $\mathbf L\cdot\hat{\mathbf m}$,
$\mathbf S\cdot\hat{\mathbf m}$, $(\mathbf L\cdot\hat{\mathbf m})(\mathbf S\cdot\hat{\mathbf m})$
and $\mathbf l\cdot\mathbf s$ coefficients of $h$ by piece in meV, and
$\max|H_\mathrm{OS}|$.

## 7. The sidecar generator {#generator}

```
python -m psp.onsite_breit_sidecar --upf Cr.upf --shell 3D --ae Cr_3d_ae.dat [--ae-reference "3d5 4s1 ..."] [-o Cr.onsite_breit.json]
python -m psp.onsite_breit_sidecar --upf Cr.upf --shell 3D --ae-oncvpsp Cr.out --ae-n 3
python -m psp.onsite_breit_sidecar --upf Cr.upf --shell 3D --null        # writes Cr.null.onsite_breit.json
```

**Inputs.** The fully relativistic norm-conserving UPF the run will use (it
must carry PP_SPIN_ORB), the PP_CHI label of the shell, and one all-electron
radial function of that shell:

- `--ae FILE`: a two-column table, $r$ (bohr) and $u(r) = rR(r)$, any
  normalisation and sign, `#` comments. Any monotone smooth grid; the
  function is continued exponentially past the last point if it has not
  decayed there ($|u(r_\mathrm{end})| \ge 10^{-7}\max|u|$; decay constant from a
  log-linear fit over the last bohr, out to 40 bohr).
- `--ae-oncvpsp FILE --ae-n N`: the output file of the fully relativistic
  ONCVPSP run that generated the UPF. For each valence state of the
  reference configuration it prints a block headed
  `n= 3,  l= 2  kap=-3, all-electron wave function, pseudo w-f` with lines
  `&  tag  r  u_AE  u_PS` ($u$ the large component). The two $\kappa$ blocks
  of $(n, l)$ ($\kappa = -(l+1)$: $j = l+\tfrac12$; $\kappa = l$:
  $j = l-\tfrac12$) are read and j-averaged as below. The plot stops at a few
  bohr with six decimals; for the Fe $3d$ shell (537 points, $r\le 5.96$ bohr)
  the spin–other-orbit tensor from the plot blocks differed from the one from
  the full-grid Dirac–Kohn–Sham function by $7\times10^{-7}$ relative
  (measured with the v1 generator).
  To write a table by hand from the same blocks, average the two columns as
  below; `--ae-oncvpsp` does exactly that.

The all-electron function is the j-average of the Dirac large components,
$u = [(l+1)\,P_{l+1/2} + l\,P_{l-1/2}]/(2l+1)$, with $P_{l-1/2}$ first given
positive overlap with $P_{l+1/2}$. Its configuration matters more than any
numerical setting: the reference atom of the pseudopotential generation is
the default, and an ionic configuration can change $I^0$ by a factor of two
for an early $3d$ metal ($\mathrm{Cr}^{3+}$ against $3d^5 4s^1$). State it
with `--ae-reference`.

The pseudo function is the same j-average of the UPF's two PP_CHI partners
(the pair `j_averaged_atomic_wfcs` uses), sign-aligned and normalised;
`projector.chi` is the plain average of the same pair, written unchanged for
the authentication of §6.

**The radial numbers.** No basis fit: `radial_breit` integrates the
normalised radial function directly,

$$
N_L = \int_0^\infty dr\, \frac{u^2(r)}{r^{L+2}} \int_0^r ds\, u^2(s)\, s^{L-1}\quad (L = 0\dots 2l+1),\qquad
K = \int_0^\infty \frac{u^4}{r^2}\,dr,
$$

the inner integral running. These are all the four-spinor tensor needs (§2):
the derivative $u'$ in $f_1 = u u' - u^2/r$ is removed by integration by
parts, so no derivative of the tabulated $u$ enters, and a test against a
brute-force double integral with the analytic $u'$ of
$u = r^{l+1}e^{-ar}$ agrees to $2.4\times10^{-9}$ for every $L$, $l = 1,2,3$
(the floor of the brute-force reference; $3\times10^{-10}$ for $l\le 2$); on
the tabulated Cl $3p$ AE function, with a spline $u'$, to $5\times10^{-9}$.

**The check tensor (spin–other-orbit, independent check only).** The
generator also builds the Breit–Pauli spin–other-orbit tensor
$\Lambda^c_{a a_2 b d} = \langle a(1)\,a_2(2)|(\mathbf r_{12}\times\mathbf p_1)_c/r_{12}^3|b(1)\,d(2)\rangle$
of the same radial function by a separate route (`shell_tensor`, with its
mean-field pieces `meanfield_pieces`), to test the four-spinor tensor against
it before a file is written (§2, "contained exactly"). It is not written to
the sidecar and no run reads it. Its construction:
acting on electron 1, $\mathbf r_{12}\times\mathbf p_1/r_{12}^3 =
i\,(\nabla_1 V)\times\nabla_1$ with
$V(\mathbf r_1) = \int \rho_{a_2 d}(\mathbf r_2)/r_{12}$ and
$\rho_{a_2 d} = \phi^*_{a_2}\phi_d$, $\phi_m = (u/r)\,Y_{lm}$. Expanding
$Y^*_{a_2} Y_d = \sum_{LM} c^{LM}_{a_2 d}\, Y_{LM}$ (even $L \le 2l$),
$V = \sum_{LM} c^{LM}_{a_2 d}\, v_L(r)\, Y_{LM}$ with

$$
v_L(r) = \frac{4\pi}{2L+1}\Big[r^{-(L+1)} P_L(r) + r^L Q_L(r)\Big],\qquad
v_L'(r) = \frac{4\pi}{2L+1}\Big[-(L+1)\,r^{-(L+2)} P_L(r) + L\,r^{L-1} Q_L(r)\Big],
$$

$P_L(r) = \int_0^r u^2 s^L\,ds$, $Q_L(r) = \int_r^\infty u^2 s^{-(L+1)}\,ds$.
With $\nabla = \hat{\mathbf r}\,\partial_r + r^{-1}\nabla_\Omega$ and
$\nabla_\Omega Y = -i\,\hat{\mathbf r}\times(\mathbf L Y)$,

$$
\Lambda^c_{a a_2 b d} = i \sum_{LM} c^{LM}_{a_2 d}\Big(A_L\, X^{1,c}_{a,LM,b} + B_L\, X^{2,c}_{a,LM,b} + C_L\, X^{3,c}_{a,LM,b}\Big),
$$

$$
A_L = \int_0^\infty \frac{u^2\, v_L'}{r}\,dr,\qquad
C_L = \int_0^\infty \frac{u^2\, v_L}{r^2}\,dr,\qquad
B_L = \int_0^\infty r R R'\, v_L\,dr = -\tfrac12\,(A_L + C_L),
$$

$$
X^{1,c} = \int d\Omega\; Y^*_a Y_{LM}\,\big(i L_c Y_b\big),\qquad
X^{2,c} = \int d\Omega\; Y^*_a Y_b\,\big(-i L_c Y_{LM}\big),\qquad
X^{3,c} = \int d\Omega\; Y^*_a\,\big(\nabla_\Omega Y_{LM}\times\nabla_\Omega Y_b\big)_c .
$$

$B_L$ is integrated by parts, and the angular integrals are exact
(Gauss–Legendre in $\cos\theta$ times a uniform $\varphi$ rule, above the
polynomial degree $4l+2$). The $L=0$ term gives $A_0 = -4\pi I^0$, so the
$\mathbf l\cdot\hat{\mathbf m}$ coefficient of $d_1$ equals
$-\tfrac{\alpha^2}{2} I^0$ to rounding.

**Quadrature.** Every radial integral, single or running, is a not-a-knot
cubic spline of $f\,dr/di$ in the grid index $i$, with $dr/di$ from a cubic
spline of $r(i)$: fourth order in the spacing on the uniform UPF mesh and on
a logarithmic all-electron mesh alike. For $u = r^3 e^{-r}$ a logarithmic grid
(2001 points) and a uniform grid (6001 points) give the same $N_L$ to
$3.1\times10^{-9}$ (worst at $N_5$; $2.0\times10^{-10}$ and
$1.2\times10^{-11}$ on twice and four times refined grids) and the same $K$ to
$2\times10^{-11}$.

**Checks before writing** (`verify_breit`; the generator refuses on failure):
pair symmetry $G_{pqrs} = G_{qpsr}$ and Hermiticity $G_{pqrs} = G^*_{rspq}$ to
$10^{-10}\max|G|$; at $D = 1$, $\max|h^\mathrm{dir}| \le 10^{-10}\max|h^\mathrm x|$
and $h^\mathrm x = a\,\mathbb 1 + c\,\mathbf l\cdot\mathbf s$ to $10^{-10}$ with
$c > 0$; the two source-resolved identities against the check tensor of the
same radial function (§2: spin-dependent part of $h^\mathrm{dir}$ for
$D = \mathbb 1\otimes D_0$ against $d_2$, spin-independent part for
$D = \sum_c\sigma_c\otimes D_c$ against $d_1$, random Hermitian $D_0$, $D_c$)
to $10^{-9}$; for $D = \mathrm{diag}(0.9, 0.3)$ per spin, the
$\mathbf l\cdot\hat{\mathbf m}$ coefficient of $h^\mathrm{dir}$ equal to
$-\tfrac{\alpha^2}{2}(N_\uparrow - N_\downarrow) I^0$ to $10^{-8}$. The
written file is then re-read with `read_sidecar`. On the shipped inputs the
residuals are $\le 10^{-13}$; a $3d$ function used as an $l = 3$ shell
reaches $3\times10^{-10}$ in the source-resolved identities (the two routes
integrate different radial integrands).

**Numbers** (normalised radial functions; closed-shell coefficients in meV):

| shell | $N_0 \dots N_{2l+1}$ (bohr$^{-3}$) | $K$ | $a$ | $c$ |
|---|---|---|---|---|
| Ni $3d$ PS (`Ni.ref.upf`) | 0.625797, 0.443372, 0.336881, 0.268826, 0.222299, 0.188805 | 1.149832 | 8.68796 | 0.75376 |
| Ni $3d$ AE (ONCVPSP `Ni.out`, $n=3$) | 1.193398, 0.751206, 0.532875, 0.407787, 0.328305, 0.273907 | 1.576887 | 13.25660 | 1.25736 |
| Fe $3d$ AE (`Fe_3d_ae.dat`) | 0.832273, 0.522864, 0.370525, 0.283391, 0.228082, 0.190252 | 1.094696 | 9.21554 | 0.87498 |
| Cr $3d$ AE (`Cr_3d_ae.dat`) | 0.434874, 0.268997, 0.189204, 0.144144, 0.115754, 0.096425 | 0.552890 | 4.69970 | 0.44947 |
| Cl $3p$ AE (`Cl_3p_ae.dat`) | 0.208555, 0.112515, 0.078682, 0.060943 | 0.250473 | 0.87050 | 0.16304 |

The Fe, Cr and Cl values are pinned in `tests/test_onsite_breit_sidecar.py`;
the test data are rebuilt by `tests/data/onsite_breit/make_v2_fixtures.py`.

**The check tensor against the earlier sidecars** (`tests/data/onsite_breit/v1/`,
made with a Gaussian fit of each radial function and analytic two-electron
integrals; same radial inputs, `tests/test_onsite_breit_sidecar.py`). The
difference is the fit error of the files:

| shell | $\max|\Delta\Lambda|/\max|\Lambda|$, AE | PS |
|---|---|---|
| Fe $3d$ | $2.7\times10^{-8}$ | $2.7\times10^{-4}$ (fit error of $u$ $1.6\times10^{-3}$ near $r_c$) |
| Cr $3d$ | $1.5\times10^{-6}$ | $1.7\times10^{-7}$ |
| Cl $3p$ | $4.3\times10^{-9}$ | $1.3\times10^{-6}$ |

The mean-field pieces of the v1 tensors agree with the post-processing code
`onsite_ops` (`v1/reference_values.json`) to $10^{-10}$. The projector
($r$, $\chi$) is reproduced bit for bit. The scalar `I0` of the earlier files
carries the trapezoid error of their quadrature (up to $6\times10^{-5}$
relative).

## 8. Not included {#not-included}

Stated in the log where it matters. Spin–other-orbit, orbit–orbit and
spin–spin are all inside the one current–current integral (§2); none of them
is missing. What is missing:

- **One radial function per shell, j-averaged.** The $j = l\pm\tfrac12$
  partners are averaged into one $u$ (§7) for the AE and for the PS function
  alike, so the spin–orbit splitting of the radial functions is absent from
  the on-site operator; and one function of one configuration per shell (the
  PAW basis would hold two partial waves per channel).
- **The all-electron small component is a lift, not the Dirac small
  component.** $\Psi_p$ carries $\tfrac{\alpha}{2}\boldsymbol\sigma\cdot\mathbf p\,\phi_p$
  of the AE *large* component (the kinetic-balance lift the plane-wave code
  uses). The Dirac small component of the all-electron atom differs from it
  near the nucleus, where the potential is not small against $2c^2$.
- $i[\mathbf r, H_\mathrm{OS}]$ in the velocity (dipole, head, orbital moment).
  `hubbard_ops.hubbard_velocity_ket` is the machinery; not wired.
- Exchange with other shells: only the shell's own direct and exchange terms
  are built (no exchange with other valence shells or with the core, beyond
  the frozen-core term of the UPF); inter-site terms beyond the smooth field;
  screening of the on-site exchange.
- The `exchange = paw` premise (that the on-site part of $\Sigma$'s TT
  exchange equals $h^\mathrm x_\mathrm{PS}$) is not validated (§3).
- The frozen-core Gaunt term (carried by the UPF, §1).
- An energy functional and the double-counting energy.
- Collinear (nspinor = 1) WFNs and ultrasoft/PAW pseudopotentials: refused.
- A production-scale projection pass: $P$ holds the full-BZ sphere once; a
  wedge pass with a symmetry action on shell matrices is not built.
