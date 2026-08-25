"""I/O module for reading and writing various file formats.

This module contains:
- wfn_loader: Reading BerkeleyGW WFN.h5 wavefunction files
- wfn_writer: Writing BGW-compatible WFN.h5 from LORRAX NSCF
- qe_save_reader: Reading QE .save directories (crystal + charge density)
- epsreader: Reading epsilon/eps0mat.h5 dielectric files
- tagged_arrays: Reading/writing ISDF tagged arrays (restart files)
- qp_wfn: QP rotation matrices and eigenvalue I/O
- centroids: Centroid file loading
- kin_ion: Kinetic + ionic Hamiltonian I/O
"""

# THE TWO LOADERS ARE SERVICES NOW, and this package reaches their doors
# like any other consumer.  ``file_io/wfn_loader.py`` and
# ``file_io/zeta_loader.py`` were transitional re-export shims and were
# deleted by the phase-wide cleanup commit (wave-1 ruling 2); these two
# lines used to be relative imports of them.
#
# THE RE-EXPORTS STAY, and they are not a new shim.  ``from file_io import
# EPSReader, resolve_input_paths`` is how this package has always been
# spelled, and a caller wanting a loader alongside those is asking
# ``file_io`` for a name it can legitimately provide.  What changed is
# where the object comes from: the door, once, with no second module
# object in between.
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                            # noqa: E402
from zeta_loader import ZetaLoader                          # noqa: E402

# Back-compat alias: every active caller uses ``WFNReader(path)`` as a
# pure-metadata accessor today.  ``WfnLoader`` covers that surface 1:1
# (mf_header attributes mirrored verbatim; ``_filename`` exposed for
# callers that thread it).
WFNReader = WfnLoader
from .qe_save_reader import CrystalData
from .wfn_writer import WFNWriter
from .epsreader import EPSReader
from .tagged_arrays import (
    COULOMB_POLICY_DATASET,
    COULOMB_POLICY_KEYS,
    DOWNFOLD_PROVENANCE_GROUP,
    read_downfold_provenance,
    assert_restart_window_matches,
    compare_coulomb_policy,
    coulomb_policy_from_config,
    describe_coulomb_policy_match,
    describe_coulomb_policy_stamp,
    format_coulomb_policy,
    parse_coulomb_policy,
    read_coulomb_policy_from_h5,
    write_restart_state_to_h5,
    write_w0_qmunu_to_h5,
    write_head_scalars_to_h5,
    read_restart_state_from_h5,
    read_munu_tensor_from_h5,
    load_restart_state_from_h5,
)
from .sigma_output import (
    QSGW_PLOT_DATASETS,
    SIGMA_CUBE_DATASETS,
    SIGMA_K_AXIS,
    SPREAD_ATTR_PREFIX,
    append_eqp_assembly_receipt_h5,
    append_qsgw_datasets_h5,
    compact_star_tables,
    extract_and_stamp_k_irr,
    k_irr_rows_for,
    read_eqp_assembly_receipt,
    read_eval_energies,
    sigma_star_spread_stats,
    star_select_k_irr,
    write_sigma_to_file,
    write_eqp_g0w0,
    write_sigma_omega_h5,
    write_sigma_freq_debug_table,
)
from .qp_wfn import write_qp_rotations_h5
from .kin_ion import (
    HARTREE_DATASET,
    HARTREE_SOURCES,
    IRR_IDX_DATASET,
    K_STORAGE_ATTR,
    K_STORAGE_FULL,
    K_STORAGE_IBZ,
    K_STORAGE_VERSION,
    K_STORAGE_VERSION_ATTR,
    N_SYM_SPATIAL_ATTR,
    SYM_IDX_DATASET,
    broadcast_ibz_to_full_bz,
    read_full_bz_dataset,
    read_star_map,
    load_kin_ion_submatrix,
    load_hartree_submatrix,
    kin_ion_has_hartree,
    kin_ion_hartree_source,
    resolve_hartree_source,
    read_kin_ion_provenance,
    validate_kin_ion_against_run,
)
from .centroids import load_centroids
from .paths import resolve_input_paths
from .read_bgw_vcoul import read_bgw_vcoul, fill_v_grid_for_q, BGWVcoulTable
