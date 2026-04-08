from __future__ import annotations

from .engine import execute_chi_from_bundle


def chi_from_bundle(
    wf_bundle,
    windows,
    omega,
    meta,
    mesh_xy,
    *,
    policy: str = "shared_conservative",
    return_plan: bool = False,
):
    """Public chi API for the archived separate freqint pipeline."""
    return execute_chi_from_bundle(
        wf_bundle=wf_bundle,
        windows=windows,
        omega=omega,
        meta=meta,
        mesh_xy=mesh_xy,
        policy=policy,
        return_plan=return_plan,
    )


def sigma_from_bundle(*args, **kwargs):
    """Placeholder for sigma channel implementation."""
    raise NotImplementedError(
        "sigma_from_bundle is not implemented yet in the freqint rewrite."
    )


def frequency_integration(
    *,
    mode: str,
    wf_bundle,
    windows,
    omega,
    meta,
    mesh_xy,
    policy: str = "shared_conservative",
    return_plan: bool = False,
):
    """Unified entrypoint for the archived chi/sigma freqint modes."""
    if mode == "chi":
        return chi_from_bundle(
            wf_bundle=wf_bundle,
            windows=windows,
            omega=omega,
            meta=meta,
            mesh_xy=mesh_xy,
            policy=policy,
            return_plan=return_plan,
        )
    if mode == "sigma":
        return sigma_from_bundle(
            wf_bundle=wf_bundle,
            windows=windows,
            omega=omega,
            meta=meta,
            mesh_xy=mesh_xy,
            policy=policy,
            return_plan=return_plan,
        )
    raise ValueError(f"Unsupported mode={mode!r}. Expected 'chi' or 'sigma'.")
