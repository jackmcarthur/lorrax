"""Deprecated shim — the ψ(G)-loading helpers moved to
``common.wfn_transforms`` (they compose ``WfnLoader.load`` with the FFT-box /
r-chunk / centroid transforms that already live there).

This module re-exports them for one release so any straggling import keeps
working; new code should import from ``common.wfn_transforms`` directly.
Scheduled for deletion once all consumers are repointed.
"""
from .wfn_transforms import (  # noqa: F401
    get_enk_bandrange,
    load_kpoint_fftbox,
    read_Gvecs_to_devices,
    iter_psi_rchunk_bandwise,
    load_centroids_band_chunked,
)
