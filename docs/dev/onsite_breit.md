# On-site Breit operator: design, conventions and the sidecar generator

*Owner page of the on-site Breit (spin–other-orbit) term, `onsite_breit = true`.
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
`tests/test_onsite_breit_sc_seam_cpu.py`, `tests/test_onsite_breit_sidecar.py`.

## 1. What is added {#what}

$$
H = T + V_\mathrm{loc} + V_\mathrm{NL} + V_H[\rho] + \alpha\,\mathbf A[\mathbf J] + H_\mathrm{OS}[D] + \Sigma_{xc}
$$

$H_\mathrm{OS}[D]$ is the Hartree–Fock mean field of the Breit–Pauli
spin–other-orbit operator inside one open $(n\,l)$ shell per atom, built from
the on-site density matrix $D$ of the current orbitals with **all-electron**
radial integrals. The smooth transverse direct field $\alpha\,\mathbf A[\mathbf J]$
of a bispinor run is computed from pseudo-orbitals, which carry only part of
the one-centre transverse integrals: every such integral of a $3d$ shell lies
inside the pseudization sphere, where norm conservation does not fix the
radial distribution of the current. The on-site term restores the
all-electron value of that one-centre part (a PAW-style completion,
$h_\mathrm{AE}-h_\mathrm{PS}$), or supplies the whole term in a run without
the smooth field.

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

**Mean field.** The sidecar tensor (§6) is

$$
\Lambda^c_{a a_2 b d} = \langle a(1)\,a_2(2)|\Lambda^c|b(1)\,d(2)\rangle,\qquad
\boldsymbol\Lambda(1;2) = \frac{\mathbf r_{12}\times\mathbf p_1}{r_{12}^3},
$$

spatial indices only, $c = x,y,z$. With the spin–orbital tensor

$$
g_{pqrs} = -\alpha^2 \sum_c \Lambda^c_{a_p a_q a_r a_s}\,\delta_{\sigma_p\sigma_r}\,(\sigma^c/2)_{\sigma_q\sigma_s}
= \langle p(1)\,q(2)|g|r(1)\,s(2)\rangle,\qquad H = \sum_{pqrs} g_{pqrs}\,a^\dagger_p a^\dagger_q a_s a_r,
$$

the four pieces are

$$
\begin{aligned}
d_1[p,r] &= \textstyle\sum_{qs} g_{pqrs} D_{sq} &&\text{orbit of the test electron in the shell's spin field}\\
d_2[p,r] &= \textstyle\sum_{qs} g_{qpsr} D_{sq} &&\text{spin of the test electron in the shell's orbital current}\\
x_1[p,r] &= -\textstyle\sum_{qs} g_{pqsr} D_{sq},\qquad x_2[p,r] = -\sum_{qs} g_{qprs} D_{sq} &&\text{bare exchange}
\end{aligned}
$$

and $h^\mathrm{dir} = d_1 + d_2$, $h^\mathrm x = x_1 + x_2$.
$g = -\alpha^2\,\boldsymbol\Lambda(1;2)\cdot\mathbf s(2)$ is the coupling
$\alpha\,\mathbf A\cdot\mathbf p$ of electron 1 to the vector potential of the
spin moment $\boldsymbol\mu = -\alpha\,\mathbf s$ ($g=2$) of electron 2.

**Limits** (unit tests, `tests/test_onsite_breit.py`): a closed shell $D = 1$
gives $d_1 = d_2 = 0$ and $h^\mathrm x = c\,\mathbf l\cdot\mathbf s$ with
$c > 0$ (the exchange of a closed shell raises $\xi$); a spherical collinear
shell gives $d_1 = b\,\mathbf l\cdot\hat{\mathbf m}$ with
$b = -\tfrac{\alpha^2}{2}(N_\uparrow - N_\downarrow)\,I^0$, $b<0$ (the direct
term lowers $\xi_\mathrm{eff} = \xi + 2\sigma b$ for majority carriers); one
electron gives $h_{aa} = 0$; $h[\Theta D\Theta^{-1}] = \Theta h[D]\Theta^{-1}$;
$h[MDM^\dagger] = M h[D] M^\dagger$ for $M = U_\mathrm{spin}\otimes D^l(R)$.
Here

$$
I^0 = \int_0^\infty\! dr_2\, u^2(r_2)\, r_2^{-3} \int_0^{r_2}\! dr_1\, u^2(r_1),\qquad u = rR(r)
$$

is Marvin's $I^0$ of the shell.

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

Format `lorrax.onsite_breit_sidecar/v1`, one per species, written by
`python -m psp.onsite_breit_sidecar` (§7). JSON; every float is written with
Python `repr` (an exact float64 round trip).

| field | content |
|---|---|
| `format` | `"lorrax.onsite_breit_sidecar/v1"` |
| `element` | element symbol (from the UPF's PP_HEADER) |
| `conventions` | text: basis, index order of $\Lambda$, storage, units, mean-field formulas (§2) |
| `shells` | list of shell objects (the generator writes one) |
| `provenance` | `generator`, `date`, `code_sha256`, `source_upf` {`path`, `md5`, `sha256`}, `ae_source`, `quadrature`, `checks` {`ae`, `ps`: the §7 check values}; `null_gate` in a null sidecar. Files written by the earlier generator carry `tensor_npz` and `fit` instead of `quadrature`. |

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
| `shape` | $[3, n, n, n, n]$, $n = 2l+1$ |
| `lam_re`, `lam_im` | flat lists, C order over `shape`, of $\mathrm{Re}/\mathrm{Im}\,\Lambda^c_{a a_2 b d}$ (§2), Hartree a.u., **without** $\alpha^2$; rebuild with `(np.array(lam_re) + 1j*np.array(lam_im)).reshape(shape)` |
| `I0`, `r3`, `I2` | Marvin $I^0$, $\langle r^{-3}\rangle$, Marvin $I^2$ of the normalised radial function (bohr$^{-3}$) |
| `b1_mev_per_muB` | $-\tfrac{\alpha^2}{2} I^0$ in meV: the $\mathbf l\cdot\hat{\mathbf m}$ coefficient of $d_1$ per unit $N_\uparrow - N_\downarrow$ |
| `x_closed_mev` | $c$ of $x_1 + x_2 = c\,\mathbf l\cdot\mathbf s$ at $D = 1$ (meV) |

**Load-time checks** (`GATE onsite_breit_sidecar`, `psp.onsite_breit.read_sidecar`
and `build_onsite_setup`): the format string; $0 \le l \le 3$; tensor shapes;
`ae` present, `ps` present when a key says `paw`; the closed-shell identity of
every tensor ($d_1 = d_2 = 0$ and $h^\mathrm x \propto \mathbf l\cdot\mathbf s$
with $c>0$, to $10^{-10}$ relative: a corrupted tensor or another index order
fails it); the element is a species of the WFN; the projector type equals the
input key; and `projector.chi` equals the j-average of the run's UPF on the
same mesh to $10^{-8}\max|\chi|$ (a Gaunt-corrected UPF keeps PP_PSWFC and
passes). The sidecar's SHA-256 is stamped into every output.

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
  the tensor from the plot blocks differs from the one from the full-grid
  Dirac–Kohn–Sham function by $7\times10^{-7}$ relative (measured).
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

**The tensor.** No basis fit: $\Lambda$ is built from the radial function
directly. Acting on electron 1, $\mathbf r_{12}\times\mathbf p_1/r_{12}^3 =
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

$B_L$ is integrated by parts, so no derivative of the tabulated $u$ enters,
and $\Lambda$ is Hermitian as a two-body operator by construction. The angular
integrals are exact (Gauss–Legendre in $\cos\theta$ times a uniform $\varphi$
rule, above the polynomial degree $4l+2$). The $L=0$ term gives
$A_0 = -4\pi I^0$, so the $\mathbf l\cdot\hat{\mathbf m}$ coefficient of $d_1$
equals $-\tfrac{\alpha^2}{2} I^0$ to rounding.

**Quadrature.** Every radial integral, single or running, is a not-a-knot
cubic spline of $f\,dr/di$ in the grid index $i$, with $dr/di$ from a cubic
spline of $r(i)$: fourth order in the spacing on the uniform UPF mesh and on
a logarithmic all-electron mesh alike. On the native grids the tensor is
converged to $9\times10^{-9}$ (Fe AE, ONCVPSP log mesh, 1229 points) and
$5\times10^{-9}$ (Cr PS, UPF mesh) relative to an eightfold (fourfold)
refinement.

**Checks before writing** (the generator refuses on failure): at $D = 1$,
$\max|d_1|, \max|d_2| \le 10^{-10}\max|x|$ and
$\|x_1 + x_2 - c\,\mathbf l\cdot\mathbf s\| \le 10^{-10}\|x_1 + x_2\|$ with
$c > 0$; for a random Hermitian $D$, $d_1$, $d_2$, $x_1 + x_2$ and their sum
Hermitian to $10^{-10}$; for $D = \mathrm{diag}(0.9, 0.3)$ per spin,
$d_1 \propto L_z$ to $10^{-10}$ and $b_1$ from the tensor equal to
$-\tfrac{\alpha^2}{2}I^0$ to $10^{-8}$. The written file is then re-read with
`read_sidecar`. On the shipped inputs the residuals are $\sim10^{-15}$.

**Agreement with the earlier sidecars** (`tests/data/onsite_breit/`, made
with a Gaussian fit of each radial function and analytic two-electron
integrals; same radial inputs, `tests/test_onsite_breit_sidecar.py`). Fed
the *fitted* functions themselves, the generator reproduces the files'
analytic integrals to $\le 3\times10^{-12}$ relative ($9\times10^{-9}$ for the
27-primitive Fe PS fit, whose evaluation cancels digits); against the files
with the original radial functions, the difference is the fit error of the
files:

| shell | $\max|\Delta\Lambda|/\max|\Lambda|$, AE | PS |
|---|---|---|
| Fe $3d$ | $2.7\times10^{-8}$ | $2.7\times10^{-4}$ (fit error of $u$ $1.6\times10^{-3}$ near $r_c$) |
| Cr $3d$ | $1.5\times10^{-6}$ | $1.7\times10^{-7}$ |
| Cl $3p$ | $4.3\times10^{-9}$ | $1.3\times10^{-6}$ |

The projector ($r$, $\chi$) is reproduced bit for bit. The scalar `I0` and
`I2` of the earlier files carry the trapezoid error of their quadrature (up
to $6\times10^{-5}$ and $2.3\times10^{-4}$ relative); `x_closed_mev` agrees to
$2.6\times10^{-4}$ (Fe PS, the fit) and $\le 1.7\times10^{-6}$ elsewhere,
`b1_mev_per_muB` to $\le 6\times10^{-5}$ (the $I^0$ quadrature, plus
$3.6\times10^{-7}$ from the meV constant: the earlier files used QE's Rydberg,
13.605693122994 eV).

## 8. Not included {#not-included}

Stated in the log where it matters.

- $i[\mathbf r, H_\mathrm{OS}]$ in the velocity (dipole, head, orbital moment).
  `hubbard_ops.hubbard_velocity_ket` is the machinery; not wired.
- On-site spin–spin and orbit–orbit (AE − PS) corrections; exchange with
  other valence shells; inter-site terms beyond the smooth field; screening of
  the on-site exchange.
- Two partial waves per channel: the sidecar holds one AE and one PS radial
  function per shell, of one configuration (the PAW basis would hold two).
- The frozen-core Gaunt term (carried by the UPF, §1).
- An energy functional and the double-counting energy.
- Collinear (nspinor = 1) WFNs and ultrasoft/PAW pseudopotentials: refused.
- A production-scale projection pass: $P$ holds the full-BZ sphere once; a
  wedge pass with a symmetry action on shell matrices is not built.
