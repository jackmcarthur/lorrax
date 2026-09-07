| Lane | Weight | Branch | Baseline | Evidence root |
|---|---|---|---|---|
| BISP-RESTART-COMPAT | heavy: audit and end-to-end compatibility | fix/restart-consumers-2026-09-07, unmerged | 891047f4e6e8639b28d6da61375c3deecfae3259 | /pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/122_bisp_restart_compat_codex_2026-09-07 |

| Pre-edit consumer census (2026-09-07) | Bundle datasets / ancillary input | Reader and source | Assumptions / defect |
|---|---|---|---|
| Census evidence | All requested search terms, including writers and non-read mentions | `census.txt`: worktree `git grep -n` over src and services at baseline | Written before source edits; broad matches are discovery evidence, not all reads |
| gw.gw_jax restart via gw_init | psi_parent_y, psi_parent_y_mun, psi_parent_k_rows; transverse siblings; legacy psi_full_y and mun; V_qmunu, S_qmunu, V0_noG0_munu, G0_mu_nu, enk_full, kgrid, band windows, n_rmu logical extents, head scalars; dynamic PPM/MPA stores and rules; zeta_q_G/isdf_header | file_io.tagged_arrays.read_restart_state_from_h5; file_io ISDF/restart and dynamic stores; ZetaLoader; symmetry_maps q wedge | Parent k rows authenticated; ns 1/2/4 shape; canonical file to packed runtime via centroid basis; both band layouts requested as SlabIO specs; q-IBZ unfolded by service |
| BSE bse_jax, feast, kpm, pseudopoles, w_exact, w_ladder, w_ladder_precond, davidson_absorption, ring subset | psi_parent_y or psi_full_y; V_qmunu/W0_qmunu and optional nohead siblings; enk_full; G0_mu_nu, vhead, whead, cell_volume, kgrid, band windows, centroid and QP source stamps | bse_io exports bse_loading.load_bse_data_from_restart_sharded / _load_ring_subset | Full-k result via shared parent action; q-IBZ via shared wedge store; ns shape; canonical centroids; layout independent. Charge-family compatibility with htransform requires runtime proof |
| BSE absorption_haydock | Above plus enk_full privately; dipole_cart/deltaE | bse_loading for tensors; private h5py at absorption_haydock:174, absorption_common:208 for dipole | Defect: private energy/dipole reads; full-k energies/dipoles, no packed axes |
| BSE bse_window energy correction | enk_full | private h5py bse_window:642 | Defect: route energy read through shared backend; full-k ladder |
| exciton_bands | BSE datasets above; enk_full privately for eqp; whead/W0_ready privately for head; dipole provenance attrs | bse_loading plus private accesses exciton_bands:450,1393,1648 | Defect: private metadata reads; htransform carrier currently may be 4-spinor while source WFN is SOC 2-spinor |
| exciton_bands VQ path | psi_parent_y or psi_full_y, psi_parent_k_rows, kgrid, enk_full, V_qmunu, W0_qmunu; zeta_q_G, G vectors, ngk, FFT grid, centroid coordinates and solve identity | vq_interp.load_zeta_coarse:482 private restart handle; bse_loading parent unfold and q store; ZetaLoader for zeta | Defect: private restart dataset access remains. Canonical host psi cache, full-q lazy tiles or shared q unfold. Interp needs full-q zeta; pure refit accepts q-IBZ zeta metadata |
| bandstructure/bse_setup | No direct HDF5 bundle access; accepts B_at_mu, interpolated energies and arrays | htransform setup + bse_loading caller | Spin is B_at_mu shape; reconstructed psi must share charge family with BSE inputs |
| bandstructure/htransform | qp_wfn_rotations: U_mnk, E_qp_nk_rydberg, band_range, kgrid, kpoints_crys, source provenance; centroid text; WFN | file_io.qp_wfn.read_qp_rotations_artifact; WfnLoader, file_io.centroids | No private HDF5; authenticates source k/window, canonical rotation order; no restart V/W or packed centroid dependency |
| gw.downfold_run | psi_parent_y/legacy psi_full_y; V_qmunu/W0_qmunu and nohead siblings; enk_full; G0_mu_nu, kgrid, band windows, heads/S_cart_head, logical extents, centroid digest, q tables; zeta_q_G/header | tagged_arrays load_restart_state_from_h5/read_munu_tensor_from_h5; bse_loading parent action; ZetaLoader; private _read_geometry:178 and q-storage probe:268 | Defect: private geometry and storage metadata reads; shared tensors full-k/full-q, canonical order, ns shape; reads axis layout regardless of writer |
| postprocess.rotate_wfn_to_qp | U_mnk, E_qp_nk_rydberg, band_range, kgrid, kpoints_crys, provenance, kirr_to_kfull | file_io.qp_wfn.read_qp_rotations_artifact except private kirr_to_kfull read:48 | Defect: remaining private mapping read; canonical k rows; no centroid/q layout assumptions |
| gw.kin_ion_io | WFN/PSP input; writes kin_ion.h5 | WfnLoader; producer HDF5 writer | Upstream producer, no gwjax restart consumption; ns from representation |
| psp.get_dipole_mtxels | WFN/PSP input; writes dipole.h5 | WfnLoader; producer HDF5 writer, provenance probe | Upstream producer; no gwjax restart consumption |
| centroid kmeans/centroids | WFN/optional preprocessing data; writes centroid text | WfnLoader; file_io.centroids | Upstream producer; canonical FFT-grid centroid order; no gwjax restart consumption |
| psp.finite_q_head_interp | No executable HDF5 reads; docstring describes V/W/zeta and dipole finite_q | Pure array helpers; caller owns loading | Not a bundle consumer or CLI; finite-q inputs supplied as arrays |
| GW restart additional private reads | Parent dataset presence, centroid root attrs, photon_g0_vectors shapes | gw_init:3104,3142,3376 | Defects: metadata bypasses shared reader despite tensor transport using file_io |
| W_BSE handoff | W0_qmunu/V_qmunu readiness; psi_full_y presence; enk_full | screening_bse:500 | Defect: stale mandatory psi_full_y preflight rejects post-landing parent-only bundles before BSE loader |
| Photon static restart | kgrid, n_rmu_C/T, n_q_total, v_qmunu_format, data_ready, unique-tile inventory, q tables | v_q_bispinor:666 h5py metadata; tensors via SlabIO and symmetry_maps | Defect: private metadata validation in GW module; q-IBZ family tables otherwise service-owned |
| Shared head helper used by BSE/GW | S_cart_head and dipole matrices/provenance | head_correction:1326,335; qsgw_head:2778 | Defect: private HDF5 metadata/payload reads; no band-layout dependence |
| GW eqp_bgw postprocessing | sigma eigenvalue reference attr; QP energy identity/mapping; WFN energies | eqp_bgw:960,968,1041 plus canonical file_io sigma/rotation readers | Defect: ancillary metadata bypass; reduced-k physics arrays already use file_io owner |
| Other BSE h5py matches | exciton eigenvectors/eigenvalues, pseudopole chains and sweep outputs | absorption_common / pseudopoles readers and writers | These are BSE-produced outputs, not gwjax bundle datasets; outside restart-format census |

| Driver | Bundle (ns; writer layout; q storage) | Verdict | Deviation / cause | Evidence path (relative to evidence root) |
|---|---|---|---|---|
| Baseline fresh GW | scalar (1; true/false; auto→IBZ), SOC supplied deck (4; true; auto→IBZ), MoS2 (4; true/false; auto→IBZ) | PASS | All five built from pristine 891047f4 at P4; supplied SOC deck is bispinor, not ns2 | `00_audit/fresh.lx.log`, `01_scalar_true` through `05_mos2_false`, `fresh.rank0.log` |
| Additional fresh GW | recorded Si fixture (2; true; auto→IBZ) | PASS | Genuine two-spinor coverage; COHSEX fixture, separate from supplied GN deck | `06_soc_ns2_fixture/fresh_full_retry.rank0.log` |
| GW same-layout restart | scalar (1; true→true and false→false; IBZ) | PASS | eqp0/eqp1 identical in all64 printed rows | `10_scalar_tt`, `13_scalar_ff`: `restart_final.rank0.log`, `eqp*.compare_final.txt` |
| GW cross-layout restart | scalar (1; true→false; IBZ) | RED, registered | eqp0 max0.051 μeV; eqp1 max0.122 μeV; exact tolerance0 | `11_scalar_tf/eqp*.compare_final.txt` |
| GW cross-layout restart | scalar (1; false→true; IBZ) | RED, registered | eqp0 max0.128 μeV; eqp1 max0.174 μeV; exact tolerance0 | `12_scalar_ft/eqp*.compare_final.txt` |
| Dynamic GW restart | SOC GN (4; true→true; IBZ) | PASS | Both EQPs identical; mpa and sigma_quadrature_rules stores carried into restart | `14_soc_tt/restart_final.rank0.log`, `eqp*.compare_final.txt`, `tmp/mpa`, `tmp/sigma_quadrature_rules` |
| GW same/cross-layout restart | MoS2 (4; true→true, true→false, false→true, false→false; IBZ) | PASS | eqp0/eqp1 identical in all210 printed rows for all four arms | `15_mos2_tt` through `18_mos2_ff`: `restart_final.rank0.log`, `eqp*.compare_final.txt` |
| Fresh + dynamic GW restart | additional SOC GN (2; true; IBZ) | PASS | Both EQPs identical in all256 printed rows after moving dynamic store readers | `07_soc_ns2_gn/{fresh,restart}.rank0.log`, `eqp*.compare.txt` |
| BSE, downfold, htransform | additional SOC GN (2; true; IBZ) | PASS execution | All complete on the new two-spinor GN bundle; htransform includes the complete QP block | `07_soc_ns2_gn/{bse,downfold,htransform}.rank0.log` |
| Exciton ongrid | additional SOC GN (2; true; IBZ) | PASS | Production ongrid calculation completes | `07_soc_ns2_gn/ongrid.rank0.log`, `ongrid.out` |
| Exciton default (interp) | additional SOC GN (2; true; IBZ) | RED, registered | ζ is8 stored q versus64 full q; existing interpolation requires full-BZ ζ and separately supports only slab geometry | `07_soc_ns2_gn/exciton.rank0.log` |
| Dynamic store/pipeline CPU tests | four emulated CPU devices, G0 | PASS | 112 passed after retargeting instrumentation to the actual reader owner | `00_audit/dynamic_cpu_v2.lx.log` |
| Shared reader P4 | All six fresh bundles (1/2/4; both read layouts; IBZ) | PASS | V, energies and both raw parent faces in every present family: max absolute difference0 between layouts | `00_audit/reader_layout_parity.json`, `reader_layout.rank0.log` |
| BSE ring Lanczos | scalar (1; true/false; IBZ) | PASS execution/layout parity | Lowest20 spectra identical between layouts; no same-deck recorded external spectrum was supplied for the 8-band GN fresh deck | `01_scalar_true/bse_debug.rank0.log`, `04_scalar_false/bse_layout_comparison.json` |
| BSE ring Lanczos | supplied SOC GN (4; true; IBZ) | PASS execution | Charge-family ns4 kernel completes; recorded receipt uses a different band/energy deck, so no same-deck reference claim | `02_soc_true/bse_debug.rank0.log` |
| BSE recorded reference | Si fixture (2; true; IBZ) | PASS | Lowest20 frozen max1e-8 eV ≤1e-6 eV; BGW mean6.521563 meV ≤10, max9.839810 meV ≤25 | `06_soc_ns2_fixture/bse_comparison.json`, `bse_debug.rank0.log`; parser `tests/test_bse_bgw_regression.py` |
| BSE, exciton default/ongrid, downfold | MoS2 (4; true/false; IBZ) | RED, registered | Completed packed-photon producer leaves W0_ready=false; screened charge payload unavailable | `03_mos2_true`, `05_mos2_false`: `{bse_retry,exciton_retry,ongrid_retry,downfold_final}.rank0.log`; `KNOWN_LORRAX_ISSUES.md` packed-photon row |
| Exciton default/ongrid | scalar (1; true/false; IBZ) | RED, bounded and registered | Rank147 saturates search160; rank multiplier40 clears search then six zero fH slots expose seed guard-band limit; explicit refit guards do not widen seed | `01_scalar_true`, `04_scalar_false`: `{exciton_retry,ongrid_retry,exciton_rank40,ongrid_rank40,exciton_guard4,ongrid_guard4}.rank0.log` |
| Exciton default/ongrid | supplied SOC (4; true; IBZ) | RED, registered | Shared reader returns true charge ns4; htransform source carrier has ns2; no two-spinor charge family exists in this model | `02_soc_true/{exciton_retry,ongrid_retry}.rank0.log` |
| Htransform + QP rotations | scalar (1; true; IBZ) | RED, registered | QRCP budget diagnostic reaches existing corrected-interior guard: corrected block ends at8, same as requested output endpoint8 | `01_scalar_true/{htransform_retry,htransform_fit}.rank0.log` |
| Htransform + QP rotations | supplied SOC (4; true; IBZ) | PASS execution; reference unavailable | Initial fitted[0,20) omitted part of QP[0,32); ncond20 plus four guards includes full block and completes. Run74 reference is a separate DFT-only deck | `02_soc_true/{htransform_retry,htransform_fit}.rank0.log`, `htransform_fit.dat` |
| Downfold CLI | scalar (1; true/false; IBZ), supplied SOC (4; true; IBZ) | PASS | Production CLI completes; children now write canonical parent-face pairs. `python -m gw.downfold_run` is a library import, excluded from execution evidence | `01_scalar_true`, `02_soc_true`, `04_scalar_false`: `downfold_final.rank0.log`, `child_v3/tmp` |
| W_BSE historical decks | scalar COHSEX/GN (1; true; auto→IBZ requested) | RED, registered | Historical left-fit window ends12 and cuts a zero-gap multiplet. Clean ncond4 diagnostic reaches blanket parent-screening preflight refusal | `21_wbse_cohsex`, `22_wbse_gn`: `{wbse,clean_window}.rank0.log` |
| kmeans producer → GW | scalar WFN | PASS | Requested192, orbit/block selection delivered168; GW explicitly consumes generated168-centroid file | `20_scalar_producers/kmeans.rank0.log`, `producer_gw_retry.rank0.log` |
| dipole producer → GW | scalar WFN | PASS | Full VNL producer completes; generated dipole staged with consumer deck | `20_scalar_producers/dipole.rank0.log`, `dipole.out`, `producer_gw_retry.rank0.log` |
| kin_ion producer → GW | scalar WFN | PASS | Eight-band kinetic/ionic producer completes and GW consumes its output | `20_scalar_producers/kin_ion.rank0.log`, `producer_gw_retry.rank0.log` |
| Postprocess QP WFN | scalar, SOC supplied, MoS2, ns2 fixture (1/2/4; canonical k-irr rotations) | PASS | Authenticated QP WFN written for each source; CPU preprocessing, no GPU algorithm invoked | `{01_scalar_true,02_soc_true,03_mos2_true,06_soc_ns2_fixture}/postprocess.log`, `WFN_qp_proof.h5`; `00_audit/postprocess.lx.log` |
| finite_q_head_interp | array helper, all ns/layouts | N/A | No bundle read or CLI; consumes caller-supplied arrays | `census.txt`, `src/psp/finite_q_head_interp.py` |
| Focused CPU reader tests | four emulated CPU devices, G0 | PASS | 77 passed; one real-GN execution test deselected because it is separately exercised at P4 | `00_audit/readers_cpu_final.lx.log` |
| Default CPU core | four emulated CPU devices, G0 | RED, registered environment | 66 passed,7 skipped,4 native-provider failures; absent liblorrax_ffi_host.so | `00_audit/core_cpu_v3.lx.log` |

| Post-consolidation consumer | Reader | Returned meaning / transport owner |
|---|---|---|
| GW restart, BSE, exciton/VQ, downfold | `file_io.restart_bundle` | Canonical raw families or selected full-k charge faces; SlabIO + one symmetry-service action; q-IBZ restoration shared |
| htransform, eqp_bgw, postprocess | `file_io.restart_bundle` | Authenticated QP rotations, energies, source stamps and k-irr mapping; one EQP assembly receipt reader |
| GW/BSE head, dipole consumers | `file_io.restart_bundle` | Cartesian dipole payload, metadata and selected parent-band windows |
| Photon static readers | `file_io.restart_bundle` | Existing BispinorVqReader moved intact; family tile validation and q tables; writer retains format constants |
| ζ consumers | `file_io.restart_bundle` | `open_zeta` opens the one zeta_loader service; moved VQ tile adapter owns no independent HDF5 handle |
| Upstream centroids/WFN/PSP | N/A | Producers or source inputs, not gwjax bundle consumers; existing canonical source services |
| Dynamic pole stores | `file_io.restart_bundle` | W-slab/header/column, pole/head, resume and fit readers moved out of mpa_store; writer and format helpers remain there. Quadrature rules retain their existing non-HDF5 service |

| Deletion ledger module | Lines removed in batch | Private or duplicate reader deleted |
|---|---:|---|
| `src/bse/bse_loading.py` | 1243 | Serial transport, old full-k/full-q branches, parent action moved to shared reader; retained only array assembly |
| `src/bse/vq_interp.py` | 218 | Private restart handle and zeta tile adapter moved |
| `src/gw/downfold_run.py` | 158 | Geometry, q-table probe and private wavefunction restoration |
| `src/file_io/tagged_arrays.py` | 992 | All reader functions and full-k writer branches |
| `src/file_io/qp_wfn.py` | 148 | All QP payload/stamp readers |
| `src/file_io/kin_ion.py` | 309 | Star-table, matrix and provenance readers |
| `src/file_io/sigma_output.py` | 320 | EQP receipt and evaluation/reference readers |
| `src/gw/v_q_bispinor.py` | 213 | BispinorVqReader moved intact |
| `src/bse/absorption_common.py` | 25 | Dipole reader |
| `src/gw/eqp_bgw.py` | 41 | Private QP/evaluation energy reader |
| `src/postprocess/rotate_wfn_to_qp.py` | 73 | Private k-irr map reader |

| `src/file_io/mpa_store.py` | 1294 | Public W, fit, head and pole readers/classes moved to shared module; writer calls the same reader |

| Shared reader public surface | Returns |
|---|---|
| `read_metadata`, `_find_restart_file`, `require_screened_bundle` | Small semantic metadata; unique canonical path; authenticated readiness |
| `read_restart_state_from_h5`, `load_restart_state_from_h5` | Named raw-parent state with requested face/axis sharding; no legacy full-k tuple slots |
| `unfold_parent_faces`, `read_wavefunctions` | Canonical full-k ψ for a family and consuming contiguous band set; true spin extent |
| `read_interaction`, `read_bse_payload`, `read_coarse_interactions` | Semantic bare/screened q tensors and BSE/VQ payloads; shared q symmetry restoration |
| `read_downfold_geometry`, `read_interaction_orbits`, `read_downfold_inputs` | Authenticated small facts, producer orbit tables, canonical downfold inputs |
| `read_qp_rotations_artifact`, `read_qp_rotations_full_bz`, `read_kirr_to_kfull` | Canonical rotations/energies and validated k indexing |
| `read_qp_wfn_stamp`, `read_qp_state_source_provenance` | Source identity and persisted QP provenance |
| `read_eqp_assembly_receipt`, `read_eval_energies`, `read_omega_reference` | One canonical EQP assembly, evaluation-energy coverage and omega reference |
| `read_star_map`, `read_full_bz_dataset`, `load_kin_ion_submatrix`, `read_kin_ion_provenance`, `validate_kin_ion_against_run` | Stored k-table restoration and kinetic/ionic provenance/window reads |
| `BispinorVqReader`, `read_photon_charge`, `read_photon_gamma` | Family photon tiles and canonical head factors |
| `load_dipole_h5`, `read_dipole_metadata`, `read_dipole_parent_window` | Dipole arrays, small provenance, selected parent velocity blocks |
| `open_zeta` | Canonical zeta service handle, whose local/collective transports remain single-owner |
| `read_w_header`, `read_w_slab[_collective]`, `read_w_tables`, `read_w_columns[_collective]`, `open_w_column_reader`, `WColumnReader` | Dynamic sampled-W headers, q slabs and bounded frequency-column tiles |
| `read_head_fit[_collective]`, `read_fit_block`, `read_fit_tensors`, `read_fit_io_receipt`, `read_fit_unfold_tables`, `read_poles`, `open_pole_reader`, `PoleReader` | Dynamic head and pole payloads with the existing bounded collective handle lifetime |
| `validate_fit_store[_for_resume]`, `read_occupation_stamps` | Resume admission, completed-fit ledger and occupation identity |
| Policy and band-window validators | Existing provenance/refusal contracts, now beside payload reads |

| Batch state | Evidence / limitation |
|---|---|
| Foundation batch 76da8f9f pushed; claim1436 | Gates above distinguish baseline runs, current reader proof, follow-up diagnostics and independent physical restrictions; no all-green claim |

| Dynamic batch | Evidence |
|---|---|
| Ready for push | `07_soc_ns2_gn`: fresh/restart exact EQP; `00_audit/dynamic_cpu_v2.lx.log`:112 passed |
