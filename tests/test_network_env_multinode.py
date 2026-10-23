"""Multi-node decisions of ``runtime.network_env`` (planner and start-up application).

Plain Python on any host: no GPU, no jax, no MPI.  Every case is a simulated
launch environment; the file-system and NIC probes are stubbed (``is_file``,
``cxi``), so the result does not depend on the machine the test runs on.  The
only site facts used are the repository's own ``config/<site>/site.toml``.
``tests/conftest.py`` restores ``os.environ`` and thaws the environment freeze
after every test.
"""
import os

import pytest

from runtime import network_env as ne

FRONTIER = ne.site_profile({"LORRAX_SITE_PROFILE": "frontier"})
PERLMUTTER = ne.site_profile({"LORRAX_SITE_PROFILE": "perlmutter"})
FRONTIER_ENV = FRONTIER["rccl"]["env"]
PERLMUTTER_ENV = PERLMUTTER["nccl"]["env"]

SITE = {"frontier": {"LMOD_SYSTEM_NAME": "frontier"},
        "perlmutter": {"NERSC_HOST": "perlmutter"},
        "none": {}}
LEG = {"rocm": {"JAX_PLATFORMS": "rocm,cpu"},
       "cuda": {"JAX_PLATFORMS": "cuda,cpu"},
       "cpu": {"JAX_PLATFORMS": "cpu"}}
STEP = {"1node": {"SLURM_STEP_NUM_NODES": "1"},
        # no_vni is Perlmutter's step network; Frontier's profile names none.
        "2node": {"SLURM_STEP_NUM_NODES": "2", "SLURM_NETWORK": "no_vni"},
        "unknown": {}}
TRANSPORT = {"none": {},
             "net": {"NCCL_NET": "AWS Libfabric"},
             "plugin": {"NCCL_NET_PLUGIN": "/opt/site/lib/libnet-plugin.so"}}
#: Frontier's launch_libraries (libhsa-runtime64.so) resolve through this path; is_file is stubbed.
LAUNCH = {"LD_LIBRARY_PATH": "/shim"}
def env_of(site, leg, step, transport, **extra):
    return {**SITE[site], **LEG[leg], **STEP[step], **TRANSPORT[transport], **LAUNCH, **extra}


def plan(env, *, is_file=lambda path: True, cxi=False):
    return ne.plan_network_environment(env, is_file=is_file, cxi=cxi)


# --- the decision matrix -----------------------------------------------------
# (site, leg, step, transport) -> (policy, applied, mpi_before_collectives)
# applied: "site" = every key of the site's section env; "fabric" = those keys except the
# transport choice (NCCL_NET, NCCL_NET_PLUGIN), which the caller made; {} = nothing.
# mpi: True/False = recorded; None = not recorded (no profile section applies).
MATRIX = [
    # Frontier, ROCm: the profile on a multi-node step; its start-up requirements under any transport.
    ("frontier", "rocm", "2node", "none", "frontier profile", "site", True),
    ("frontier", "rocm", "2node", "net", "explicit transport", "fabric", True),
    ("frontier", "rocm", "2node", "plugin", "explicit transport", "fabric", True),
    ("frontier", "rocm", "1node", "none", "unchanged", {}, None),
    ("frontier", "rocm", "1node", "plugin", "explicit transport", {}, None),
    ("frontier", "rocm", "unknown", "none", "unchanged", {}, None),
    # Frontier's profile has no [nccl]: a CUDA leg there gets no profile.
    ("frontier", "cuda", "2node", "none", "unchanged", {}, None),
    ("frontier", "cpu", "2node", "none", "no device collectives", {}, None),
    # Perlmutter, CUDA: the profile on a multi-node step; no MPI-first requirement.
    ("perlmutter", "cuda", "2node", "none", "perlmutter profile", "site", False),
    ("perlmutter", "cuda", "2node", "net", "explicit transport", "fabric", False),
    ("perlmutter", "cuda", "2node", "plugin", "explicit transport", "fabric", False),
    ("perlmutter", "cuda", "1node", "none", "unchanged", {}, None),
    ("perlmutter", "cuda", "unknown", "none", "unchanged", {}, None),
    ("perlmutter", "rocm", "2node", "none", "unchanged", {}, None),
    ("perlmutter", "cpu", "2node", "none", "no device collectives", {}, None),
    # No profile and no CXI NICs: the library's own policy.
    ("none", "cuda", "2node", "none", "unchanged", {}, None),
    ("none", "rocm", "2node", "none", "unchanged", {}, None),
    ("none", "cuda", "2node", "net", "explicit transport", {}, None),
]


@pytest.mark.parametrize("site, leg, step, transport, policy, applied, mpi", MATRIX,
                         ids=["-".join(row[:4]) for row in MATRIX])
def test_decision_matrix(site, leg, step, transport, policy, applied, mpi):
    result = plan(env_of(site, leg, step, transport))
    assert result["policy"] == policy
    site_env = {"frontier": FRONTIER_ENV, "perlmutter": PERLMUTTER_ENV}.get(site, {})
    want = (site_env if applied == "site" else
            {k: v for k, v in site_env.items() if k not in ("NCCL_NET", "NCCL_NET_PLUGIN")}
            if applied == "fabric" else applied)
    assert result["applied"] == want
    assert result.get("mpi_before_collectives") == mpi
    if leg != "cpu":
        assert result["library"] == {"rocm": "RCCL", "cuda": "NCCL"}[leg]


# --- topology: the STEP, never the allocation --------------------------------

# --- refusals, caller exports, site selection --------------------------------

@pytest.mark.parametrize("transport", TRANSPORT)
def test_step_network_required_with_any_transport(transport):
    env = env_of("perlmutter", "cuda", "2node", transport)
    del env["SLURM_NETWORK"]
    with pytest.raises(RuntimeError, match="MPI/OFI endpoint initialization"):
        plan(env)
    # The requirement is a token, not a substring, and may be combined with
    # other Slurm network options. Explicit transports obey the same rule.
    env["SLURM_NETWORK"] = "no_vni_extra"
    with pytest.raises(RuntimeError, match="before srun"):
        plan(env)
    env["SLURM_NETWORK"] = "no_vni,job_vni"
    assert plan(env)["multi_node"] is True

# --- start-up application ----------------------------------------------------

@pytest.fixture
def fresh(monkeypatch):
    """A clean process environment for configure_gpu_network, host probes stubbed;
    ``setup(**env)`` may be called again to start a scenario from scratch."""
    real = ne.plan_network_environment
    monkeypatch.setattr(ne, "plan_network_environment",
                        lambda env, **kw: real(env, is_file=lambda path: True, cxi=False, **kw))
    said = []

    def setup(**env):
        for name in list(os.environ):
            if name.startswith(("NCCL_", "FI_", "SLURM_", "OMPI_", "HSA_FORCE")) or name in (
                    "LMOD_SYSTEM_NAME", "NERSC_HOST", "LORRAX_SITE_PROFILE", "JAX_PLATFORMS"):
                monkeypatch.delenv(name)
        monkeypatch.setattr(ne, "_configuration", None)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return said
    return setup

