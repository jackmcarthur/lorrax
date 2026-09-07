# BISP-PHYS: first-principles convention audit

Branch **audit/bisp-phys-2026-09-06**, unmerged; audited source **b8e036a8f00fcbb9092cc711ab53438c129576fb**. The task's initial base designation takes precedence over its older ade4fc66 boilerplate. The moving orchestrator and fixed-main e1559a07 worktrees were read-only references. No production code was changed. This report and sixteen executable oracles are the deliverable.

**Result:** the tested lift, antiunitary vertex action, all sixteen finite-q response and self-energy contractions, packed Dyson ordering, complex block unfolding, band projection, and periodic transverse Hartree signs agree with literal formulas. The final CPU leg collected **38 tests, all passed, no skips**. This does not certify a complete relativistic response or a production GW deck. A signed α_y ISDF normal-matrix convention was exposed: it cancels in the unregularized fit, but makes the retained positive ridge act toward zero. Its production energy impact remains unmeasured.

## Evidence and scope

All new numerical evidence is under `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/114_bisp_phys_codex_2026-09-06/`. Every row below used pool **57966610**; the step identifiers and results were read from disk. Each directory contains a manifest, runner, rank wrapper, test-source snapshot, pytest log, XML and launcher receipt. Claims live in the sandbox `CLAIMS.md` and `claims/NNNN.md`; they explicitly identify this branch as unmerged.

| Directory | Step | Observed result | Claim |
|---|---|---|---|
| `01_literal_cpu` | `lx-Xg0-014724-1425097-5291` | 2 passed, 1 failed: draft oracle used opposite χ endpoint/q convention | 1008 |
| `02_literal_cpu` | `lx-Xg0-015032-1444842-1536` | 6 passed | 1010 |
| `03_covariance_cpu` | `lx-Xg0-015304-1463517-3112` | 9 passed | 1013 |
| `04_complete_chain_cpu` | `lx-Xg0-015531-1490807-7407` | 12 passed | 1015 |
| `05_current_gram_cpu` | `lx-Xg0-015728-1505383-3346` | 1 failed, 12 deselected: α_y matrix is minus physical Gram | 1016 |
| `06_signed_fit_cpu` | `lx-Xg0-020028-1529740-4586` | 16 passed; sign cancellation and ridge consequence isolated | 1018 |
| `07_final_cpu` | `lx-Xg0-020155-1543792-5653` | **38 passed in 51.14 s**, launcher exit 0 in 58 s | **1026** |
| `08_delivery_cpu` | `lx-Xg0-021525-1651505-9666` | **38 passed in 48.34 s**, launcher exit 0 in 54 s; final gate path | **1041** |

Final artifacts: `08_delivery_cpu/{pytest.log,results.xml,attempt1.log,oracle.py.snapshot}`. The run08 snapshot is byte-identical to the delivered gate. This explicit gate stays outside ordinary pytest discovery because it requires a four-device CPU invocation. The run07 snapshot predates only the final gate-path relocation (the Si fixture path changes accordingly); the numerical code is unchanged. The delivered gate is [tests/multi_device/bispinor_physics_oracles.py](../../tests/multi_device/bispinor_physics_oracles.py). The final leg includes its sixteen tests plus `test_photon_head_sign_oracle.py`, `test_photon_chi_vertices.py`, and the symmetry service's `test_bispinor_actions.py`.

These are CPU algebra tests with four emulated devices in a 2×2 named mesh, x64 enabled. Backend FFT/GEMM and vendor LU are replaced by native CPU operations; production vertex, carrier, contraction, packing, distributed Dyson matrix construction and unfold logic are exercised. They do not test the replaced vendor backends, GPU memory behavior, or bit identity with fixed-main. CPU cells are exempt from the P=4 GPU verification rule. No GPU leg or new physical-deck result is claimed.

The synthetic group is an actual `SymMaps` construction: identity, x↔y improper reflection with half-cell translation (1/2,1/2,0), and their TR partners; a two-site crystal, 3×3×1 k grid, four parents, nine children and 2×2×1 centroid grid. It has nontrivial centroid permutations, Bloch phases, determinant and antiunitary actions. A separate oracle uses all 96 spatial/TR rows from the actual Si WFN header. The header lacks an authenticated QE schema receipt, producing the expected provenance warnings: these tests check the supplied mathematical rows, not the historical magnetic classification of the WFN generator.

Absolute/relative tolerances of 2–3×10^-14 for small matrix identities, 3×10^-13 for dense solves and density/Hartree, and 3×10^-12 for sums, FFTs and band projections allow x64 summation/reassociation roundoff. No approximation-error budget or fitted tolerance is used. “Machine precision” below means this floating-point bound, not bitwise equality.

## Equations and endpoint conventions

All line references in this report are at **b8e036a8**, not the moving orchestrator tip. Write Γ⁰=I₄ and Γⁱ=αⁱ=[[0,σᵢ],[σᵢ,0]]. The matrices are Hermitian; Γ² is imaginary and antisymmetric under plain transpose, while Γ⁰, Γ¹ and Γ³ are real symmetric. That difference makes α_y the decisive transpose witness. The metric sign lives in D, not in Γ.

### Lift, spatial action, TR and vertices

`common/bispinor_init.py:237` lifts a two-component reciprocal coefficient u(p), p=(G+k) in inverse Bohr, to

    Ψ(p) = r(p) [u(p); h (σ·p)u(p)],  h=α/2=0.00364867628215,
    r=1 (raw), or r=(1+h²p²)^(-1/2) (isometric).

This is the declared kinetic-balance/no-pair representation, not a solution including the negative-energy complement. Independent Pauli matrices verify both lifts.

For a spatial row S, the four-component action is U₄=diag(U₂,det(S)U₂). For an antiunitary row, the code constructs the two-component matrix iσ_y conj(U₂) and applies K to the wavefunction; the same TR action acts on both large and small blocks. The determinant remains on the small block (`symmetry_maps/maps.py:1913`). In the service's coordinate convention R_cart is transposed to give the forward polar action (`:3090`); Λ=diag(1,R_forward) for spatial rows, diag(1,−R_forward) for time-odd antiunitary current rows (`:3115`). One must not insert an additional axial determinant for current.

The identities checked for every Si row are

    U₄† Γᴵ U₄ = Σⱼ Λᴵⱼ Γʲ                         (unitary),
    conj(U₄† Γᴵ U₄) = Σⱼ Λᴵⱼ Γʲ                   (antiunitary).

The conjugation in the second line is essential: the input is KΨ. The scalar charge is TR even; the three current channels I=1,2,3 are polar and TR odd.

For centroid pullback π_s and lattice return L_s(μ), the actual local transport is

    Ψ_child(μ) = U₄ T_s [exp(+2πi k_parent·L_s(μ)) Ψ_parent(π_s μ)],

where T_s is identity or conjugation (`maps.py:1051`). K conjugates the phase as well. `unfold_psi` in reciprocal space (`:2061`) uses exp[−i(SG)·tnp], tnp=2πτ; its cell-periodic gauge may differ by a band-independent Seitz phase, as handled by the existing sphere/face gate. The literal local tests do not independently exercise that reciprocal-sphere loader.

The density feature is Mᴵ_mn(μ)=Ψ_m(μ)†ΓᴵΨ_n(μ). Thus

    Γᴵ T_s(Ψ) = T_s(Σⱼ Λᴵⱼ Γʲ Ψ)

with coefficients conjugated when a complex linear combination is carried through an antiunitary map. `centroid_k_unfold.py:121` applies the vertex **after** transport. Tests compare this with mixed vertices before transport, using complex coefficients and all 96 rows. A transpose on Γ² is rejected by the negative-control Σ test.

### ISDF normal equations: signed convention

For left/right band windows, define the physical positive normal matrix

    Qᴵ_μν(q)=Σ_kmn w_m w_n conj(Mᴵ_mn(k,k+q;μ)) Mᴵ_mn(k,k+q;ν).

The parent's C builder (`isdf/core.py:1490`, phase preparation `:1549`, contraction `:1593`) instead uses Γ on both open-spin factors, without conjugating one Γ. `gamma_double_contract` implements exactly its documented contraction, Γ_L[a,A]Γ_R[b,B] P_L*[a,b]P_R[A,B]; it does not silently conjugate a vertex. For equal Hermitian Dirac vertices this gives

    Cᴵ = sᴵ Qᴵ,  s=(+1,+1,−1,+1).

The Z tail (`core.py:3280`, `:3298`, coupled vertices `:3309`) has the same phase product. At the algebraic contraction seam Zᴵ=sᴵ Z_positive, so

    (sQ)^(-1)(sZ)=Q^(-1)Z.

The normal-matrix oracle exercises the actual parent C kernel; the RHS oracle exercises the actual `gamma_double_contract` on independently formed two-point tensors. A full host-store/FFT `_z_q_face_parent` literal oracle is **not** included. This distinction matters before changing either sign. See finding F1 below for regularization.

### All sixteen χ blocks and both orientations

Let tᴵ(k,q;μ)=v_(k+q)(μ)†Γᴵ c_k(μ), with occupied v and empty c, and a(k,q)=exp[−τ(ε_ck−ε_v,k+q)]. At the raw face-kernel seam the implemented static response is

    χᴵᴶ_μν(q;τ) = −1/√Nₖ Σ_kvc {
        a(k,q) conj(tᴵ(k,q;μ)) tᴶ(k,q;ν)
      + a(k,−q) tᴵ(k,−q;μ) conj(tᴶ(k,−q;ν)) }.

Equivalently the first transition dᴵ=c_k†Γᴵv_(k+q) contributes dᴵ_μ conj(dᴶ_ν). The reverse is a separate ordered particle-hole orientation, not “multiply the forward result by two.” This explicitly states which endpoint and which q are conjugated. All sixteen Γᴵ/Γᴶ pairs, including rectangular left/right centroid families, match this formula on complex, non-TR data with CT amplitude >0.1 and imaginary amplitude >0.01.

At `w_isdf.py:661–666` both supplied monomial vertex phases are conjugated for the already-conjugated Green operand; the endpoint helper handles its own row phase. Reversing the BA contraction swaps spin/centroid endpoints and takes the corresponding dagger. `_complete_static_vertex_orientations` (`:98`) realizes A_R+conj(A_R) with those ordered endpoint conventions. `compute_no_pair_dirac_current_block` (`:1815`) supplies the negative quadrature coefficient, without an extra factor two, and selects q-IBZ at `:1910`.

The raw 1/√Nₖ comes from three orthonormal k FFTs. The downstream `_w_solve_pref_scalar` supplies 2/(√Nₖ nspin nspinor_wfnfile), so for a two-component WFN and nspin=1 the combined normalization is 1/Nₖ. Nₖ is the full k count, not the number of stored q parents. The dense Dyson oracle verifies the extra 1/3 for Nₖ=9. Volume is in D; it is not inserted again in this raw χ seam.

Run01's draft oracle chose the opposite transition/q convention and differed by 0.93337471 on CC. Writing both ordered transitions explicitly resolved that oracle error. No production sign was changed to make it pass.

### Ward proxy, bare interaction, packing and Dyson

`_subtract_static_tt_contact` (`w_isdf.py:1918`) performs, for every TT block,

    Π_corrected(q_parent)=Π_raw(q_parent)−Π_raw(0),  Π_corrected(0)=0.

It is applied in q-IBZ before star restoration. The zero slice is assumed to be Γ. The numerical subtraction and later block transport are tested separately; an invariant Γ contact is required for full covariance. The complete little-group/Γ response construction is a BISP-HEAD obligation, not established by subtracting any arbitrary matrix. This is a declared Ward proxy, not a derivation of the missing contact or negative-energy response.

For K=G+q away from the separately treated singular head,

    D⁰⁰(G,q)=v(G,q)/Ω,
    D⁰ⁱ=Dⁱ⁰=0,
    Dⁱʲ(G,q)=−[v(G,q)/Ω](δᵢⱼ−KᵢKⱼ/|K|²).

`v_q_bispinor.py:270` builds per-G tiles and uses `COULOMB_GAUGE_TT_SIGN=-1` once (`:335`, applied near `:384`). The numerical test supplies a scalar v and independently constructs all nine transverse projectors: it verifies the transverse tensor/sign, not the scalar Coulomb quadrature. The centroid tile is

    Dᴵᴶ_μν=Σ_G conj(ζᴵ_Gμ) Dᴵᴶ(G,q) ζᴶ_Gν.

The seven unique stored blocks are CC and upper-triangle TT; the reverse tile is Dᴶᴵ=(Dᴵᴶ)†, including centroid-axis swap **and** conjugation (`BispinorVqReader.get_tile`, `:815`, reverse at `:831`). Rectangular C/T extents do not change this rule. The new reader test uses a deliberately complex, non-Hermitian stored off-diagonal tile.

`photon_layout.py` interleaves C⊕T₁⊕T₂⊕T₃ over mesh tiles; it is a permutation of the block matrix, not a change of Lorentz signs. For physical χ including the normalization above,

    A=I−Dχ,  W=A^(-1)D=D(I−χD)^(-1).

The order Dχ in `w_isdf.solve_w:1548` / distributed A construction `:1345` is verified with noncommuting complex Hermitian D and χ. A is generally not Hermitian; W is Hermitian under these static assumptions. Testing only commuting/diagonal matrices would miss the order error. Packing/unpacking all sixteen blocks and Wᴶᴵ=(Wᴵᴶ)† are checked.

### Star restoration and band-space self-energy

For a scalar centroid operator define F_s=diag(exp[+2πi q_parent·L_s])P_s. With K_s denoting optional elementwise conjugation,

    Bᴬᴮ(q_child)=Σ_CD Λᴬ_C Λᴮ_D F'_s K_s[Bᶜᴰ(q_parent)] F'_s†,
    F'_s=F_s (unitary), conj(F_s) (antiunitary).

`photon_blocks_full_q` (`w_isdf.py:2003`) owns scalar permutation, phase and antiunitary conjugation. `mix_lorentz_blocks` (`maps.py:1867`) then applies the real Λ⊗Λ factors and does **not** conjugate again. This preserves off-diagonal Hermitian companions. The q-star test exercises all sixteen complex blocks and nontrivial glide phases; the independent χ covariance test recomputes response from transformed spinors with O(1) CT. The latter isolates spin/k covariance without centroid Seitz phases; together the tests cover these two aspects.

For G_f(k;aμ,bν)=Σ_n f_nk Ψ_nk(aμ) conj(Ψ_nk(bν)), define

    C_f[B](k)=−1/Nₖ Σ_q Σ_AB Γᴬ G_f(k−q) Γᴮ Bᴬᴮ(q).

Centroid B multiplies the corresponding μν entry; spin multiplication is only through Γ. In `_build_G_face` (`greens_function_kernel.py:139`, conjugation `:176`) the right wavefunction is conjugated. Applying Γ to that unconjugated face produces ΓᴬGΓᴮ†=ΓᴬGΓᴮ, because Γ is Hermitian. It does **not** request Γᴮᵀ at input. `_make_static_convolution` (`cohsex_sigma.py:233`) supplies the minus and 1/Nₖ; the block kernel is `photon_sigma.py:111`.

The static terms are

    Σ_X=C_occ[D],
    Σ_SX=C_occ[W],
    Σ_COH=−(1/2) C_sum[W−D] = +(1/2Nₖ) Σ_qAB ΓᴬG_sumΓᴮ(W−D)ᴬᴮ.

`contract_lorentz_blocks` (`photon_sigma.py:161`, selection `:171`, factor `:182`) chooses V, W, W−V respectively; X/SX use occupied weights and COH the sigma-sum band window. The −0.5 is **in addition to** the minus already inside C. The literal kernel test evaluates all sixteen vertices with noninteger weights, arbitrary complex interactions, and both factors 1 and −0.5. It does not call the whole wrapper with independent fractional-occupation/window selection; that wiring is source-audited rather than independently proved here.

Bare transverse exchange is Σᴮ=C_occ[D_TT]. Since D_TT itself is negative, no further “Lorentz minus” should be added to its contraction. On the incumbent static route, `cohsex_sigma.py:747–748` adds this same Σᴮ to reported X and SX. The physical static total is SX+COH (`sigma_dispatch.py:1336`), so **this is not double counting**. In the unscreened packed limit, W_TT=D_TT and W−D has zero TT: Σ_X^TT=Σ_SX^TT=Σᴮ, Σ_COH^TT=0. The actual packed solve verifies this limit; the assembly conclusion follows from the cited callers. On dynamic incumbent routes only X gets bare Σᴮ (`cohsex_sigma.py:825`), while scalar dynamical correlation remains separate. The packed dynamic branch may add static current SX+COH into its X column (`sigma_dispatch.py:1190–1204`); this is a reporting convention, not a second addition of bare X. There are **15** non-CC Lorentz pairs, not 12.

Projection is Σ_ij(k_parent)=Σ_aμbν conj(Ψ_i(aμ))Σ_aμ,bν Ψ_j(bν). The kernel's result at each parent equals a literal full-child k−q sum and this projection for every Γ pair. Final full-k band restoration uses `unfold_file_wedge_band_operator(...,trs_rule='transpose')` (`maps.py:3872`): in the transported band gauge, antiunitary static Hermitian operators satisfy Σ_child=Σ_parent*=Σ_parentᵀ. This applies after summing the covariant Lorentz operator, not independently to arbitrary isolated AB blocks. A complete occupied spin/centroid toy with invariant interaction and complex off-diagonal band matrices verifies every child; omitting the transpose differs by >0.01. A realistic partially occupied screened parent-vs-fixed-main driver comparison remains unperformed.

### Density, transverse Hartree and head completion

`psp/get_DFT_mtxels.py:165` forms ρᴵ(r)=Σ_kn w_kn Ψ_kn†ΓᴵΨ_kn with the same signed weights in every channel; current extraction is at `:263`. Charge is TR even and current TR odd. The literal oracle includes a negative occupation increment, so discarding or taking absolute values of weights fails. Representation-dependent overall volume/k normalization belongs to callers; not every density caller is independently rederived here.

For the periodic transverse direct field, with code input J=j/c,

    A_i(G)=−(8π/|G|²)(δ_ij−G_iG_j/|G|²)J_j(G), G≠0; A(0)=0.

The 8π is the Rydberg Poisson convention. `psp/dft_operators.py:174` applies this tensor, with the negative metric passed by `gw/kin_ion_io.py:893`. Its matrix element is α·A; there is no exchange minus. Independent NumPy Fourier sums verify sign, projector and zero mode. `qsgw_density.rho_from_wfns:361` routes scalar and typed polar-current symmetry separately; full SCF feedback/finite slab Hartree is not covered by this periodic test.

The head response is explicitly incomplete as a relativistic model (`static_gauge_response.py` module contract): current implementation includes charge q², charge wings, and Hall CT q¹, but omits TT q², CT q², current wings, contact and negative-energy complement. In `head_correction.static_hall_linear_response:162`, H_a,0i=−i ε_bai σ_H,b and H_a,i0=conj(H_a,0i). The energy-ordered velocity convention is P=−ΔD. The existing literal head-sign suite was rerun in the final leg and checks the declared signs.

`complete_static_slab_photon_q0:1343` uses S_eff=S+YWΓZ/Ω, the qH+qqS head response, and samplewise Dyson inversion before cubature averaging. Its Γ rank-four bare update is conj(g0)⊗<D>g0/Ω (`:1526`). Screened left factors contain (WZ)ᵀ and right factors YW (`:1536–1556`); the explicit transpose on WZ is not an invitation to conjugate it again. The existing head-sign/moment tests support these local formulas, but share parts of the Coulomb sampling infrastructure. BISP-HEAD should own little-group completeness, anisotropic missing terms and generator/cubature provenance. No completed BISP-HEAD report was available at the checked report locations during this audit; this section is the report handoff, not a claim of coordination already performed.

## Findings and unresolved obligations

### F1 — signed α_y ISDF Gram changes the meaning of positive regularization

**Sites:** `src/isdf/core.py:1549–1593` (C); `:3280–3315` (Z phase product); positive ridge at `:5148`, distributed preparation `:5464–5468`, fused/local preparations `:6712`, `:6835`. These are b8e036a8 locations.

**Equation and witness:** C²=−Q² instead of Q². In run05, the actual parent-k normal matrix differs from the physical Gram by max **28.129017934227004**, relative **2**. Run07 repeats that value with the strengthened glide fixture; channels 0,1,3 differ only by approximately 5.34e−15, 8.89e−15, 7.11e−15. The common C/Z sign cancels without regularization. For δ=10^-12|tr C|/n, however,

    (−Q+δI)^(-1)(−Z)=(Q−δI)^(-1)Z,

rather than the positive-Gram regularization (Q+δI)^(-1)Z. The actual `_transverse_lu_math` primitive on Q=diag(1,6e−13), Z=(0,6e−13), gives **6.000000000018**, while the positive-Gram ridge gives **0.5454545454543968**, δ=5.000000000003e−13.

This refines a hazard already documented in `core.py:5131–5144`, where positive ridge on negative modes is acknowledged. The public fit certification may reject this near-null example; the primitive oracle does **not** prove that a harmful fit passes `_certify_transverse_ridge`. No Si/MoS2 QP shift is measured or inferred. The sign is a convention, not by itself a wrong self-energy. The negative-ridge consequence is the actionable limitation.

**Fix disposition:** no production fix in this lane. A change must treat C and Z together, including coupled/legacy paths and certification, or define a sign/spectrum-aware regularizer. Flipping only C would break a currently cancelling pair. The shortest responsible next step is an independent full Z oracle, then a certified-fit counterexample and one combined P4 deck comparison if the public path admits it. Registered in sandbox `KNOWN_LORRAX_ISSUES.md` under this unmerged branch. The new tests deliberately record the signed convention and cancellation; they are not a claim that positive ridge is physically preferred.

### F2 — existing CT covariance evidence was almost null

Inherited probes, not new jobs: `runs/Si/100_bisp_parent_route_2026-09-05/{34_chi_covariance_p4,35_chi_raw_covariance_p4,36_chi_vertex_covariance_p4}`. Their receipts are respectively `lx-Xg4-184754-1475466-2642`, `lx-Xg4-184953-1485609-6817`, `lx-Xg4-185143-1494585-3809`, each exit0. The extracted per-block receipts are in [inherited_chi_probes.json](inherited_chi_probes.json).

There are 28 antiunitary q rows, but CT norms are only about **6.06e−17 to 8.10e−17**, with relative errors near 2. That does not test the polar/time-odd CT rule. The raw TT probes have relative errors about 5e−10; the earlier corrected TT probe had about 0.995 error, consistent with the already addressed contact-placement defect. These measurements are not evidence of a new TT defect at the audit base.

**Closure:** new complex non-TR response data produce CT >0.1; all16 χ blocks and their recomputed covariance pass, including improper/TR sign and conjugation. The q-star test separately carries nontrivial complex centroid phases. This closes the local algebra blind spot, not a realistic broken-inversion production-deck validation.

### F3 — model and route claims must remain bounded

The Ward proxy is not the full Ward identity, finite static head coverage is not gauge completeness, and a static Hermitian band unfold is not a proof for an arbitrary dynamic non-Hermitian operator. The assembled static Σᴮ double entry is consistent; no fix is indicated. Documentation describing all non-CC packed blocks as “12” should say **15** (three CT, three TC, nine TT); no interface rename or refactor is required. The historical task's long `sigma_x_bispinor` equation docstring has already been replaced by a shared-contraction caller at this base, so the actual minus must be read at the convolution and D builder.

Unproved sites are recorded in the register below. No failed or unrun link is promoted to a production correctness claim.

## Gate blindness and recommended minimal additions

[gate_blindness.md](gate_blindness.md) classifies the direct bispinor gates and supporting files; [gate_census.csv](gate_census.csv) enumerates test/function names and base-line locations. The census intentionally includes broad supporting tests, and does not imply every such test was independently rederived or rerun. The highest-value additions are the sixteen new oracles: explicit α_y, all16 blocks, O(1) complex CT, actual improper/TR group, glide phases, noncommuting Dyson, complex Hermitian companions and band-transpose red control. Remaining minimal additions are a literal full Z/store path; public ridge-certification adversary; independent all-window X/SX/COH wrapper assembly; and one realistic screened parent-vs-full-k P4 comparison. BISP-HEAD's group-complete head response is a separate obligation.

## Convention register

PROVED means equality at the explicitly tested seam/fixture within the tolerance above. FLAGGED means source-derived, model-limited, or missing an independent numerical witness. Oracle names below omit the `test_` prefix and refer to the new module unless marked existing.

| Convention | Site at b8e036a8 | Status | Oracle / remaining scope |
|---|---|---|---|
| Raw/isometric hσ·p lift; positive h | bispinor_init:237 | PROVED | literal_gamma_lift_and_all_96_spatial_trs_rows |
| Γ² Hermitian but Γ²ᵀ=−Γ²; other Γ transpose signs | gamma_matrices | PROVED | literal_gamma_lift_and_all_96_spatial_trs_rows; independent Pauli |
| det(S) on small block | maps:1913 | PROVED | all96 row oracle |
| iσ_y K on both blocks; conjugated U†ΓU for TR | maps:1913 | PROVED | all96 row oracle |
| Current polar/time-odd; charge even | maps:3090,3115 | PROVED | all96 rows; nonzero_ct_covariance_recomputed_from_spinors |
| Local +k·L phase and K acting on phase | maps:1051 | PROVED | vertex_after_unfold_equals_mixed_vertices_before_unfold; parent χ/Σ glide |
| Reciprocal exp(−iSG·tnp), sphere/local gauge | maps:2061 | FLAGGED | existing parent-face gate only; no new sphere oracle |
| Vertex after unfold and complex coefficient conjugation | centroid_k_unfold:121 | PROVED | vertex_after_unfold_equals_mixed_vertices_before_unfold |
| Pair ψ†Γψ and signed C²=−Q² | core:1593 | PROVED (signed) | isdf_current_signed_normal_matrix_against_literal_pair_gram |
| Same C/Z sign cancels | gamma_double_contract; core:3280 | PROVED at algebra seam | signed_isdf_rhs_cancels_in_unregularized_fit |
| Full Z host-store/FFT sign cancellation | core:2985,3280–3315 | FLAGGED | end-to-end Z oracle missing |
| Positive ridge becomes Q−δI for Γ² | core:5148 | PROVED limitation | positive_ridge_moves_negative_gram_toward_zero; public certification/deck effect unpriced |
| χ left/right conjugation, Γ not Γᵀ, k±q order | w_isdf:416,661–666 | PROVED | all_16_chi_blocks_nonzero_ct_literal_both_orientations |
| Reverse χ ordered orientation, not 2×forward | w_isdf:98,1815 | PROVED | same non-TR χ oracle |
| Raw1/√Nₖ and downstream spin/file prefactor | w_isdf raw kernel, solve_w | PROVED for nspin1/file2 | χ + packed_dyson_order_prefactor_hermiticity_and_bare_limit; other caller conventions source-only |
| Parent χ equals explicit full children | w_isdf parent face kernel | PROVED on toy | parent_chi_equals_literal_full_k_for_all_16_blocks |
| TT Π(q)−Π(0), exact zero at Γ | w_isdf:1918 | PROVED subtraction | q_star_unfold_all_blocks_ward_contact_and_daggers |
| Γ little-group completeness of contact | w_isdf contact placement; head_correction:1343 | FLAGGED | BISP-HEAD obligation |
| Single negative TT metric and transverse projector | v_q_bispinor:270,335 | PROVED | bare_tiles_metric_sign_and_complex_hermitian_companions |
| Scalar v/Ω and full ζ†vζ tile pipeline | v_q_bispinor builder/accumulator | FLAGGED whole pipeline | per-G tensor proved; scalar quadrature/full tile accumulation shared by existing references |
| Reverse stored tile = conjugate centroid transpose | v_q_bispinor:815,831 | PROVED | complex reader companion oracle |
| Packed matrix permutation and I−Dχ order | photon_layout; w_isdf:1345,1548 | PROVED | noncommuting packed Dyson oracle; CPU LU backend |
| W_AB†=W_BA, including complex CT | solve_w; photon_blocks_full_q | PROVED | packed Dyson and q-star oracles |
| Scalar unfold conjugates once; Λ⊗Λ does not conjugate | w_isdf:2003; maps:1867 | PROVED | q_star_unfold_all_blocks_ward_contact_and_daggers |
| Nonzero CT/TC polar/TR covariance | maps:1867,3115 | PROVED | nonzero_ct_covariance_recomputed_from_spinors |
| G=ψ f ψ†; right Γ not Γᵀ | greens_function_kernel:176 | PROVED | parent_sigma_all_vertices_q_convolution_and_projection; transpose red twin |
| Σ convolution minus, k−q and1/Nₖ | cohsex_sigma:233 | PROVED | all16 parent Sigma oracle |
| COH factor −0.5 multiplies already-negative convolution | photon_sigma:182 | PROVED at kernel seam | all16 Sigma factor oracle |
| Wrapper V/W/W−V and occupied/sum windows | photon_sigma:161–182 | FLAGGED numerical assembly | source-consistent; independent complete wrapper test missing |
| Bare TT W=D gives X=SX, COH=0 | solve_w; cohsex_sigma:747–748 | PROVED limit + source assembly | packed bare-limit oracle; physical static total SX+COH |
| Dynamic current correction reported in X | sigma_dispatch:1190–1204 | FLAGGED dynamic route | source audit only, no dynamic driver oracle |
| Parent-band bra conjugated, ket unconjugated | photon_sigma:111 | PROVED | all16 parent Sigma literal projection |
| Final antiunitary band transpose after sector sum | maps:3872; photon_sigma:235 | PROVED static toy | full_band_unfold_matches_literal_sigma_on_symmetric_complete_toy |
| Screened production parent/full-k self-energy | complete driver | FLAGGED | no new fixed-main comparison |
| Density signed weights, real ψ†Γψ, TR signs | get_DFT_mtxels:165,263 | PROVED local density | density_all_currents_signed_weights_and_time_reversal |
| Density caller Ω/k factors, complete SCF feedback | qsgw_density:361 and callers | FLAGGED | local density proof does not cover every caller |
| Periodic direct TT −8πP_T/G², G0=0 | dft_operators:174; kin_ion_io:893 | PROVED | periodic_transverse_hartree_sign_projector_and_zero_mode |
| Hall −iεσ and CT/TC conjugate pairing | head_correction:162 | PROVED declared model | existing photon_head_sign_oracle, rerun final leg |
| Head S+YWZ/Ω, WZ transpose and YW orientation | head_correction:1343,1526–1556 | PROVED local declared formulas | existing head-sign/moment suite; shared sampler limitation |
| Full Ward/gauge completeness, missing current head terms | static_gauge_response module | FLAGGED model limitation | not supplied by the no-pair paramagnetic approximation |
