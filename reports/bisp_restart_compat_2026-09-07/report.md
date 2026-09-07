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
| GW same-layout restart | scalar (1; true→true and false→false; IBZ) | PASS | eqp0/eqp1 identical in all64 printed rows | `10_scalar_tt`, `13_scalar_ff`: `layout_handoff.rank0.log`, `eqp*.layout_handoff_compare.txt` |
| GW cross-layout restart | scalar (1; true→false; IBZ) | RED, registered | eqp0 max0.051 μeV; eqp1 max0.122 μeV; exact tolerance0 | `11_scalar_tf/eqp*.layout_handoff_compare.txt` |
| GW cross-layout restart | scalar (1; false→true; IBZ) | RED, registered | eqp0 max0.128 μeV; eqp1 max0.174 μeV; exact tolerance0 | `12_scalar_ft/eqp*.layout_handoff_compare.txt` |
| Dynamic GW restart | SOC GN (4; true→true; IBZ) | PASS | Both EQPs identical; mpa and sigma_quadrature_rules stores carried into restart | `14_soc_tt/layout_handoff.rank0.log`, `eqp*.layout_handoff_compare.txt`, `tmp/mpa`, `tmp/sigma_quadrature_rules` |
| GW same/cross-layout restart | MoS2 (4; true→true, true→false, false→true, false→false; IBZ) | PASS | eqp0/eqp1 identical in all210 printed rows for all four arms | `15_mos2_tt` through `18_mos2_ff`: `layout_handoff.rank0.log`, `eqp*.layout_handoff_compare.txt` |
| Fresh + dynamic GW restart | additional SOC GN (2; true; IBZ) | PASS | Both EQPs identical in all256 printed rows after moving dynamic store readers | `07_soc_ns2_gn/{fresh,restart}.rank0.log`, `eqp*.compare.txt` |
| BSE, downfold, htransform | additional SOC GN (2; true; IBZ) | PASS execution; BSE/htransform reference obligation RED, registered | All complete on the new two-spinor GN bundle; htransform includes the complete QP block | `07_soc_ns2_gn/{bse,downfold,htransform}.rank0.log` |
| Exciton ongrid | additional SOC GN (2; true; IBZ) | PASS | Production ongrid calculation completes | `07_soc_ns2_gn/ongrid.rank0.log`, `ongrid.out` |
| Exciton default (interp) | additional SOC GN (2; true; IBZ) | RED, registered | ζ is8 stored q versus64 full q; existing interpolation requires full-BZ ζ and separately supports only slab geometry | `07_soc_ns2_gn/exciton.rank0.log` |
| Dynamic store/pipeline CPU tests | four emulated CPU devices, G0 | PASS | 112 passed after retargeting instrumentation to the actual reader owner | `00_audit/dynamic_cpu_v2.lx.log` |
| Shared reader P4 | All six fresh bundles (1/2/4; both read layouts; IBZ) | PASS | V, energies and both raw parent faces in every present family: max absolute difference0 between layouts | `00_audit/reader_layout_parity.json`, `reader_layout.rank0.log` |
| BSE ring Lanczos | scalar (1; true/false; IBZ) | PASS execution/layout parity; reference obligation RED, registered | Lowest20 spectra identical between layouts; no same-deck recorded external spectrum was supplied for the 8-band GN fresh deck | `01_scalar_true/bse_debug.rank0.log`, `04_scalar_false/bse_layout_comparison.json` |
| BSE ring Lanczos | supplied SOC GN (4; true; IBZ) | PASS execution; reference obligation RED, registered | Charge-family ns4 kernel completes; recorded receipt uses a different band/energy deck, so no same-deck reference claim | `02_soc_true/bse_debug.rank0.log` |
| BSE recorded reference | Si fixture (2; true; IBZ) | PASS | Lowest20 frozen max1e-8 eV ≤1e-6 eV; BGW mean6.521563 meV ≤10, max9.839810 meV ≤25 | `06_soc_ns2_fixture/bse_comparison.json`, `bse_debug.rank0.log`; parser `tests/test_bse_bgw_regression.py` |
| BSE, exciton default/ongrid, downfold | MoS2 (4; true/false; IBZ) | RED, registered | Completed packed-photon producer leaves W0_ready=false; screened charge payload unavailable | `03_mos2_true`, `05_mos2_false`: `{bse_retry,exciton_retry,ongrid_retry,downfold_final}.rank0.log`; `KNOWN_LORRAX_ISSUES.md` packed-photon row |
| Exciton default/ongrid | scalar (1; true/false; IBZ) | RED, bounded and registered | Rank147 saturates search160; rank multiplier40 clears search then six zero fH slots expose seed guard-band limit; explicit refit guards do not widen seed | `01_scalar_true`, `04_scalar_false`: `{exciton_retry,ongrid_retry,exciton_rank40,ongrid_rank40,exciton_guard4,ongrid_guard4}.rank0.log` |
| Exciton default/ongrid | supplied SOC (4; true; IBZ) | RED, registered | Shared reader returns true charge ns4; htransform source carrier has ns2; no two-spinor charge family exists in this model | `02_soc_true/{exciton_retry,ongrid_retry}.rank0.log` |
| Htransform + QP rotations | scalar (1; true; IBZ) | RED, registered | QRCP budget diagnostic reaches existing corrected-interior guard: corrected block ends at8, same as requested output endpoint8 | `01_scalar_true/{htransform_retry,htransform_fit}.rank0.log` |
| Htransform + QP rotations | supplied SOC (4; true; IBZ) | PASS execution; reference obligation RED, registered | Initial fitted[0,20) omitted part of QP[0,32); ncond20 plus four guards includes full block and completes. Run74 reference is a separate DFT-only deck | `02_soc_true/{htransform_retry,htransform_fit}.rank0.log`, `htransform_fit.dat` |
| Downfold CLI | scalar (1; true/false; IBZ), supplied SOC (4; true; IBZ) | PASS | Production CLI completes; children now write canonical parent-face pairs. `python -m gw.downfold_run` is a library import, excluded from execution evidence | `01_scalar_true`, `02_soc_true`, `04_scalar_false`: `downfold_final.rank0.log`, `child_v3/tmp` |
| W_BSE historical decks | scalar COHSEX/GN (1; true; auto→IBZ requested) | RED, registered | Historical left-fit window ends12 and cuts a zero-gap multiplet. Clean ncond4 controls on untouched baseline and the final branch both refuse parent_screening_diagrams; the interim ladder diagnostics are superseded by the baseline comparison below | `21_wbse_cohsex`, `22_wbse_gn`: `{wbse,clean_window}.rank0.log` |
| kmeans producer → GW | scalar WFN | PASS | Requested192, orbit/block selection delivered168; GW explicitly consumes generated168-centroid file | `20_scalar_producers/kmeans.rank0.log`, `producer_gw_retry.rank0.log` |
| dipole producer → GW | scalar WFN | PASS | Full VNL producer completes; generated dipole staged with consumer deck | `20_scalar_producers/dipole.rank0.log`, `dipole.out`, `producer_gw_retry.rank0.log` |
| kin_ion producer → GW | scalar WFN | PASS | Eight-band kinetic/ionic producer completes and GW consumes its output | `20_scalar_producers/kin_ion.rank0.log`, `producer_gw_retry.rank0.log` |
| Postprocess QP WFN | scalar, SOC supplied, MoS2, ns2 fixture (1/2/4; canonical k-irr rotations) | PASS | Authenticated QP WFN written for each source; CPU preprocessing, no GPU algorithm invoked | `{01_scalar_true,02_soc_true,03_mos2_true,06_soc_ns2_fixture}/postprocess.log`, `WFN_qp_proof.h5`; `00_audit/postprocess.lx.log` |
| finite_q_head_interp | array helper, all ns/layouts | N/A | No bundle read or CLI; consumes caller-supplied arrays | `census.txt`, `src/psp/finite_q_head_interp.py` |
| Focused CPU reader tests | four emulated CPU devices, G0 | PASS | 77 passed; one real-GN execution test deselected because it is separately exercised at P4 | `00_audit/readers_cpu_final.lx.log` |
| Default CPU core | four emulated CPU devices, G0 | RED, registered environment | 66 passed,7 skipped,4 native-provider failures; absent liblorrax_ffi_host.so | `00_audit/core_cpu_closeout.lx.log` |

| Post-consolidation consumer | Reader | Returned meaning / transport owner |
|---|---|---|
| GW restart, BSE, exciton/VQ, downfold | `file_io.restart_bundle` | Canonical raw families or selected full-k charge faces; SlabIO + one symmetry-service action; q-IBZ restoration shared |
| htransform, eqp_bgw, postprocess | `file_io.restart_bundle` | Authenticated QP rotations, energies, source stamps and k-irr mapping; one EQP assembly receipt reader |
| GW/BSE head, dipole consumers | `file_io.restart_bundle` | Cartesian dipole payload, metadata and selected parent-band windows |
| Photon static readers | `file_io.restart_bundle` | Existing BispinorVqReader moved intact; family tile validation and q tables; writer retains format constants |
| ζ consumers | `file_io.restart_bundle` | `open_zeta` opens the one zeta_loader service; moved VQ tile adapter owns no independent HDF5 handle |
| Upstream centroids/WFN/PSP | N/A | Producers or source inputs, not gwjax bundle consumers; existing canonical source services |
| Dynamic pole stores | `file_io.restart_bundle` | W-slab/header/column, pole/head, resume and fit readers moved out of mpa_store; writer and format helpers remain there. Quadrature rules retain their existing non-HDF5 service |

| Deletion ledger module (b05e586a versus891047f4; source/service code) | Lines removed | Lines added | Reader deletion / disposition |
|---|---:|---:|---|
| `services/zeta_loader/src/zeta_loader/loader.py` | 2 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bandstructure/htransform.py` | 101 | 3 | EQP parser moved; canonical rotation and energy readers used |
| `src/bse/absorption_common.py` | 25 | 0 | Dipole payload reader moved; BSE-output reader is outside bundle scope |
| `src/bse/absorption_eigvecs.py` | 13 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/absorption_haydock.py` | 13 | 6 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/bse_feast.py` | 2 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/bse_io.py` | 22 | 2 | Three-tuple EQP façade deleted |
| `src/bse/bse_jax.py` | 1 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/bse_kpm.py` | 2 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/bse_loading.py` | 1245 | 56 | Parent unfold, serial transport and legacy full-file branches removed; array assembly retained |
| `src/bse/bse_pseudopoles.py` | 1 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/bse_ring_comm.py` | 1 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/bse_w_exact.py` | 12 | 5 | Private scalar/two-spinor action deleted; canonical symmetry service used |
| `src/bse/bse_window.py` | 106 | 3 | EQP correction reader moved |
| `src/bse/davidson_absorption.py` | 8 | 4 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/exciton_bands.py` | 21 | 15 | Call sites/imports use the sole reader; no independent reader retained |
| `src/bse/vq_interp.py` | 447 | 17 | Private restart handle, VQ payload reader and ζ tile adapter moved |
| `src/bse/w_ladder.py` | 2 | 2 | Call sites/imports use the sole reader; no independent reader retained |
| `src/file_io/__init__.py` | 8 | 12 | Call sites/imports use the sole reader; no independent reader retained |
| `src/file_io/isdf_header.py` | 50 | 6 | ISDF decoder and reader moved; dataclass, binder and writer retained |
| `src/file_io/kin_ion.py` | 309 | 0 | Star-table, matrix and provenance readers moved |
| `src/file_io/mpa_store.py` | 1216 | 11 | Dynamic W, head, fit and pole readers/handle classes moved; writers retained |
| `src/file_io/qp_wfn.py` | 148 | 7 | Rotation, stamp and provenance readers moved |
| `src/file_io/restart_bundle.py` | 0 | 4500 | One actual implementation module; moved functions, shared admission and semantic accessors |
| `src/file_io/sigma_output.py` | 320 | 10 | EQP assembly/evaluation/reference readers moved |
| `src/file_io/tagged_arrays.py` | 992 | 37 | All readers moved; full-k writer compatibility deleted |
| `src/gw/downfold.py` | 1 | 1 | Stale full-k dataset name removed from band-window explanation |
| `src/gw/downfold_config.py` | 2 | 2 | Stale full-k dataset name removed from band-window explanation |
| `src/gw/downfold_run.py` | 165 | 15 | Geometry, q-storage probe and payload readers moved |
| `src/gw/eqp_bgw.py` | 116 | 8 | EQP text parser moved; private QP/evaluation reads replaced |
| `src/gw/gw_init.py` | 48 | 16 | Private bundle probes replaced; reader layout consumed; obsolete W_BSE refusal removed |
| `src/gw/head_correction.py` | 15 | 14 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/mpa/fit_driver.py` | 8 | 10 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/mpa/model.py` | 5 | 7 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/mpa/sigma.py` | 1 | 5 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/qsgw_head.py` | 10 | 8 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/sc_iteration.py` | 1 | 5 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/screening_bse.py` | 65 | 7 | Private handoff probe replaced; central canonical-to-packed accessor used |
| `src/gw/sigma_dispatch.py` | 1 | 3 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/sigma_x_bispinor.py` | 1 | 1 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/v_q_bispinor.py` | 213 | 0 | BispinorVqReader moved intact |
| `src/gw/v_q_g_flat.py` | 1 | 1 | Call sites/imports use the sole reader; no independent reader retained |
| `src/gw/w_isdf.py` | 2 | 4 | Call sites/imports use the sole reader; no independent reader retained |
| `src/postprocess/rotate_wfn_to_qp.py` | 73 | 5 | Private k-irr mapping reader deleted |
| `src/psp/get_dipole_mtxels.py` | 176 | 2 | Provenance reader moved; producer retained |
| Total, 45 source/service modules; excludes tests | 5971 | 4824 | Net -1147 lines; counts include moved implementations and call-site edits, not a claim that every deleted line was a private read |

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
| `read_bgw_eqp`, `read_eqp_energies`, `apply_eqp_corrections` | One text parser; authenticated wedge energies and BSE correction application, with service-owned k unfolding |
| `read_kin_ion_full_bz` | Canonical full-k kinetic/ionic matrix for the consuming symmetry map |
| `read_isdf_header`, `read_isdf_header_from_file` | The shared ISDF header decoder used by ζ service and bundle consumers |
| `read_vq_payload`, `check_dipole_provenance` | VQ wavefunction/ζ metadata payload and source-identity validation |
| `require_parent_screening_consumer` | Baseline admission for GW screening consumers; unported non-RPA diagrams refuse before parent loading |
| Policy and band-window validators | Existing provenance/refusal contracts, now beside payload reads |

| Batch state | Evidence / limitation |
|---|---|
| Foundation batch 76da8f9f pushed; claim1436 | Gates above distinguish baseline runs, current reader proof, follow-up diagnostics and independent physical restrictions; no all-green claim |

| Dynamic batch | Evidence |
|---|---|
| 64313881 pushed; claim1437 | `07_soc_ns2_gn`: fresh/restart exact EQP; `00_audit/dynamic_cpu_v2.lx.log`:112 passed |


| Metadata seam | Verdict | Evidence |
|---|---|---|
| VQ metadata and dipole provenance | Actual readers moved to `file_io.restart_bundle`; removed 232 VQ and 178 PSP lines, including their private provenance handle | `read_vq_payload`, `check_dipole_provenance`; central family and q-storage decisions |
| P4 provenance/scope/parent velocity tests | 24 passed on each rank | `00_audit/metadata_tests.rank0.log`; JID58016040 |
| CPU counterpart | 12 passed,12 blocked by missing host FFI during producer-module import | `00_audit/metadata_cpu.lx.log`; P4 counterpart passes |
| GW full-head restart / frozen BSE | Both complete with moved provenance reader | `06_soc_ns2_fixture/metadata_{restart,bse}.rank0.log` |
| Historical DFT htransform control | Completes; RED reference equality: max0.0564528864 meV, RMS0.0375181558 meV over4×16 bands | `23_htransform_reference/comparison.json`; existing stamped parser from `tools/compare_bgw_inteqp_htransform.py`. Retired CLI/deck options translated; this DFT-only path reads no restart. Difference is outside bundle restoration; exact algorithmic cause unassigned |
| ns2 GN pure-refit diagnostic, finite on-grid Q | RED:0.66200 meV against0.01 meV certificate; tolerance unchanged | `07_soc_ns2_gn/metadata_refit_finite.rank0.log`; 8v8c, four guards, refit-window=bse |


| EQP consolidation seam | Verdict | Evidence |
|---|---|---|
| `read_bgw_eqp`, `read_eqp_energies`, `apply_eqp_corrections`, `read_kin_ion_full_bz` | One actual reader module; removed BSE three-tuple façade and htransform/GW parser implementations | `file_io.restart_bundle`; energies retain their existing units and window contracts |
| BSE with eqp1.dat | PASS execution on ns2 GN q-IBZ bundle | `07_soc_ns2_gn/bse_eqp_shared.rank0.log` |
| htransform with eqp1.dat | PASS execution on same bundle | `07_soc_ns2_gn/htransform_eqp_shared.rank0.log`, `bands_eqp.dat` |
| CPU parser and header selection | 52 passed:19 EQP plus33 header/import checks | `00_audit/eqp_headers_cpu_final.lx.log`; two native-I/O cells excluded and separately recorded |
| P4 broad unit selection | 61 passed;10 tests require fully addressable local arrays and fail on distributed four-process arrays | `00_audit/shared_tests.rank0.log`; these unit fixtures are not production P4 drivers. Production BSE/htransform above complete |
| Metadata batch | 8a274048 pushed; claim1438 | P4 metadata tests and full-head restart/BSE |


| ζ header consolidation | Verdict | Evidence |
|---|---|---|
| `read_isdf_header`, `read_isdf_header_from_file` | Actual metadata reader and decoder moved from isdf_header into restart_bundle; zeta service uses that same decoder | 33 CPU header/import checks, within the52-pass selection |
| Dynamic ns2 GN restart after move | PASS:256/256 rows exact for eqp0 and eqp1 | `07_soc_ns2_gn/header_restart.rank0.log`, `eqp{0,1}.header_compare.txt` |
| Canonical four-spinor writer/readback | PASS: both charge/current orientations exact, four max errors0 | `08_parent_contract/parent_contract.rank0.log`, `restart_parity.json`; synthetic writer now supplies complete current band/energy/grid receipts |
| Downfolded child → BSE | PASS: child96-centroid bundle read by the same canonical parent reader and solved | `25_child_reader/bse_child.rank0.log`; source `07_soc_ns2_gn/child/tmp` |
| Final CPU core |66 passed,7 skipped, same4 registered host-provider failures | `00_audit/core_cpu_closeout.lx.log` |
| EQP batch |0056ba63 pushed; claim1439 | BSE and htransform eqp1 legs complete |
| Remaining direct reads audit |No private gwjax-bundle HDF5 or SlabIO reader remains in the audited drivers | `post_census.txt`; qsgw_head's two SlabIO reads are upstream parallel_transport.h5 preprocessing, not gwjax output. compute_vcoul's auxiliary symmetry read and eqp_bgw's source-energy read are WFN inputs. BSE pseudopole/eigenvector readers consume BSE outputs |


| Exact BSE spin-family seam | Verdict | Evidence |
|---|---|---|
| Private scalar/two-spinor `_spin_rotation` | Deleted; call site uses `symmetry_maps.spinor_rotation_for_sym_row` for the family's true extent | No new spin-specific branch; the existing symmetry service supplies scalar/Pauli/Dirac actions |
| CPU family-span and TRS refusal controls |3 passed (ns1,2,4) | `00_audit/trs_family_cpu.lx.log` |
| Exact BSE, scalar true |PASS: two Wc columns, max residual7.40e-11 ≤1e-10;9/200 iterations | `01_scalar_true/exact_trs.rank0.log` |
| Exact BSE, ns2 GN and ns4 supplied GN |PASS: two Wc columns each, max residual7.28e-11 ≤1e-10;9/200 iterations | `07_soc_ns2_gn/exact_trs_complete.rank0.log`, `02_soc_true/exact_trs_complete.rank0.log`; full8v8c windows avoid the trial4-conduction-band multiplet cut |
| ζ header batch |80f2b2f6 pushed; claim1440 | Exact dynamic restart and canonical downfold-child BSE |


| Superseded W_BSE diagnostic (before baseline-admission follow-up) | Verdict | Evidence |
|---|---|---|
| Interim W_BSE ordering change | WITHDRAWN: untouched baseline refuses this route before any ladder solve. Baseline admission restored and the unported ordering change removed | Historical diagnostics below are not evidence that those physics failures pre-existed on baseline |
| Scalar COHSEX, clean4-conduction-band diagnostic |PASS execution; RED covariance remains | `21_wbse_cohsex/shared_order_fixed.rank0.log`: GMRES1.00e-6,201/300; covariance0.1230 vs1e-5 |
| Ordering defect size |Max eqp0/1 correction225.665854486 eV; old covariance0.6837 | `21_wbse_cohsex/eqp*.order_fix_compare.txt`; compares the two diagnostic runs, not a historical physics reference |
| Scalar GN, clean window |RED after shared handoff:400/512 states refuse spectral-shell extrapolation, counts14/18/20 | `22_wbse_gn/shared_order_fixed.rank2.log`; static/probe covariance0.1230/0.01103 also fails1e-5 |
| Extra diagnostic cancellations |Early7-minute quiet run cancelled; later trace showed the ladder solve was active, and COHSEX completes in246 s of resolvent work | `00_audit/wbse_v2.lx.log`, `wbse_trace.lx.log`, `21_wbse_cohsex/shared_parent_bounded.rank0.log`. Timeout/stack-dump teardown noise is excluded from correctness evidence |
| TRS batch |0a7d2d0a pushed; claim1442 | Exact BSE covers ns1/2/4 |


| Final GW layout handoff | Verdict | Evidence |
|---|---|---|
| Charge/current carrier layout | Reader returns the requested layout; GW uses that value without branching on the band layout again | `src/gw/gw_init.py`; raw tensor transport remains in `file_io.restart_bundle` |
| Nine-arm P4 repeat after handoff | All36 rank exits0; same-layout scalar, dynamic SOC and all MoS2 arms retain exact printed EQPs | `00_audit/layout_handoff.lx.log`; `10_scalar_tt` through `18_mos2_ff`: `layout_handoff.rank{0,1,2,3}.rc`, `eqp*.layout_handoff_compare.txt` |
| Scalar cross-layout bound | RED remains0.051/0.122 μeV and0.128/0.174 μeV, tolerance0 | Same comparison files; same-file reader parity is exact |
| Final default CPU core |66 passed,7 skipped,4 registered native-provider failures | `00_audit/core_cpu_closeout.lx.log`; no native library rebuilt |
| W_BSE batch |d8edf9f7 pushed; claim1443 | Execution and remaining scientific failures distinguished above |


| Closeout | Result |
|---|---|
| Consolidation |45 source/service modules;5971 lines removed,4824 added, net1147 removed. Reader implementations moved into `file_io.restart_bundle`; writer and canonical transport/symmetry services retain their existing ownership |
| Obsolete full-k reader contracts |No `require_full_k_psi`, `psi_full_y` compatibility arm or `_unfold_bse_parent_faces` remains in src/services; old static bundles are refused centrally: regenerate with gwjax at main ≥ 891047f4 |
| GW layout batch |a63037cf pushed; claim1444; branch fix/restart-consumers-2026-09-07, unmerged |
| Completion criterion |Every requested execution/reference obligation is green, N/A for non-consumers, or explicitly RED with an invocation and measured boundary in `KNOWN_LORRAX_ISSUES.md`. This is not an all-green compatibility certification |
| Compute |One allocation58016040; P4 production legs, CPU tests with G0/four emulated devices; no native rebuild. Allocation released after closeout |


| Baseline follow-up instrument | Contract / evidence |
|---|---|
| Untouched baseline | Detached `891047f4` at `tmp/worktrees/wt_restart_baseline_891047f4`; tracked and untracked status clean |
| Matched runs | JID58021406, P4, one pool/node, BFC@0.85, cache off, same native provider; six deck pairs byte-identical in `30_tip_controls/deck_identity.json`; same argv in `controls.sh` |
| Numerical extraction | Existing `_parse_eigenvalues` from `tests/test_bse_bgw_regression.py` and stamped `read_htransform`/`metrics` from sandbox `tools/compare_bgw_inteqp_htransform.py`; all values in `30_tip_controls/tip_comparison.json` |
| Full-precision refit witness | `capture_refit.py` records the existing certificate's input arrays, then calls it unchanged. Both original module invocations also reproduce the refusal. Witness teardown noise follows the expected certificate failure and is excluded |
| W_BSE interpretation | Baseline has no GMRES/covariance/extrapolation result for these decks: both stop earlier. Removing its admission guard was a behavior regression, not evidence of pre-existing solver failures. Restore the guard centrally and withdraw the unported ordering change |

| RED-row tip control | Untouched891047f4 | Branch before admission correction | Final baseline/branch difference | Verdict | Evidence under `30_tip_controls` |
|---|---|---|---|---|---|
| W_BSE scalar COHSEX, clean4c | `parent_screening_diagrams` refusal before GMRES; residual/iterations/covariance unavailable | GMRES1e-6,201/300; covariance0.1230 vs1e-5; run completes | Final branch refuses with the identical gate/message on all4 ranks; solver physics remains unported | regression — admission fixed | `base/21_wbse_cohsex/control.rank0.log`; `branch/21_wbse_cohsex/{control,admission_fixed}.rank0.log`; original detailed solver receipt `../21_wbse_cohsex/shared_order_fixed.rank0.log` |
| W_BSE scalar GN, clean window | Same early refusal; no states reach shell extrapolation |400/512 refusals at14/18/20; static/probe covariance0.1230/0.01103 | Final branch has the same baseline refusal on all4 ranks. Baseline did NOT refuse the same400 states: it never computed them | regression — admission fixed | `base/22_wbse_gn/control.rank0.log`; `branch/22_wbse_gn/control.rank2.log`, `admission_fixed.rank0.log`; original detailed covariance receipt `../22_wbse_gn/shared_order_fixed.rank0.log` |
| ns2 GN finite-Q pure refit |0.6620023038539945 meV vs0.01 |0.6620023038539945 meV vs0.01 |Both refit/stored eigenvalue arrays max absolute/relative difference0; certificate relative difference0 ≤1e-6 | pre-existing | `{base,branch}/07_soc_ns2_gn/{control,refit_witness}.rank0.log`, `refit_certificate_inputs.json`; `tip_comparison.json` |
| Htransform DFT vs run74 b5a58202 |Max0.05645288638657168 meV; RMS0.03751815584079322 |Same max/RMS |Tip max absolute/relative difference0 over4×16 values ≤1e-6 | pre-existing — reference drift | `{base,branch}/23_htransform_reference/{control.rank0.log,bandstructure.dat}`; `tip_comparison.json` |
| Scalar8-band GN BSE lowest20 |2.48309532…2.79925695 eV |Same20 values |Max absolute0 eV; max relative0 ≤1e-6 | pre-existing — no tip regression | `{base,branch}/01_scalar_true/control.rank0.log`; all20 values in `tip_comparison.json` |
| Supplied SOC ns4 GN BSE lowest20 |2.45655385…2.57529601 eV |Same20 values |Max absolute0 eV; max relative0 ≤1e-6 | pre-existing — no tip regression | `{base,branch}/02_soc_true/control.rank0.log`; all20 values in `tip_comparison.json` |
