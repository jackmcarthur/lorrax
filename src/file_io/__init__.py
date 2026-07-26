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

from .wfn_loader import WfnLoader
from .zeta_loader import ZetaLoader
# Back-compat alias: every active caller uses ``WFNReader(path)`` as a
# pure-metadata accessor today.  ``WfnLoader`` covers that surface 1:1
# (mf_header attributes mirrored verbatim; ``_filename`` exposed for
# callers that thread it).  This shim lets the existing import sites
# stay; a follow-up commit will sweep them to ``WfnLoader`` directly.
WFNReader = WfnLoader
from .qe_save_reader import CrystalData
from .wfn_writer import WFNWriter
from .epsreader import EPSReader
from .tagged_arrays import (
    assert_restart_window_matches,
    write_restart_state_to_h5,
    write_w0_qmunu_to_h5,
    write_head_scalars_to_h5,
    read_restart_state_from_h5,
    load_restart_state_from_h5,
    save_restart_state_per_proc,
)
from .sigma_output import (
    write_sigma_to_file,
    write_eqp_g0w0,
    write_sigma_omega_h5,
    write_chunked_complex_dataset_h5,
    write_sigma_freq_debug_table,
    copy_sigma_kij_h5_to_omega_h5,
)
from .qp_wfn import write_qp_rotations_h5
from .kin_ion import (
    load_kin_ion_submatrix,
    kin_ion_has_hartree,
    read_kin_ion_provenance,
    validate_kin_ion_against_run,
)
from .centroids import load_centroids
from .paths import resolve_input_paths
from .read_bgw_vcoul import read_bgw_vcoul, fill_v_grid_for_q, BGWVcoulTable
