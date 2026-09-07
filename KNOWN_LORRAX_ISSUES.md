# BISP post-landing review — main 0e16bdb6

| Finding | Location on main 0e16bdb6 | Status |
|---|---|---|
| 1 | `src/bse/vq_interp.py:484`, `src/gw/downfold_run.py:179` | Reader port gated on fresh SOC/scalar P4 bundles; both downfolds complete. Full exciton gate blocked separately (below). |
| 13 | `src/gw/w_isdf.py:1276` | Deferred: eager restore of nine photon sources plus nine mixed outputs holds 18 full-q tiles; scan sources into requested outputs to hold 10. |
| 14 | `src/gw/cohsex_sigma.py:329` | Deferred: no-parent-plan axis projection uses raw faces and inserts two hidden all-to-all reshards; production carries a parent plan. |
| 15 | `src/ffi/cpp/cufft/conv_kpair_cuda_ffi.cc:194` | Deferred: parent row decode, table lookups and phase trigonometry repeat inside spin-pair loops; hoist after device-library rebuild and GPU oracle (+47% measured CCT current channel). |
| Gate blocker | `src/bandstructure/htransform.py` → `src/isdf/galerkin.py` | Fresh SOC parent bundle: exciton htransform hits spin axis 2 vs 4 before interpolation; scalar deck saturates QRCP search (rank 147 > 90% of 160). Default interpolation also has independent 3D/slab and full-q ζ scope guards. Evidence `01_restart_readers/{soc,scalar}/exciton.rank0.log`. |
| CPU fixture blocker | `src/bse/bse_loading.py:262` | Three legacy padding tests construct a 1x1 mesh while four CPU devices are visible; `_get_local_mesh_coords` indexes missing devices. Initial targeted run 70 passed/4 failed; geometry regression fixed, remaining three unrelated mesh failures excluded from 68-pass downfold/scope rerun. |
