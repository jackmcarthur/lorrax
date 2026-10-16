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

from runtime import env_guard
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
# applied: "site" = every key of the site's section env; {} = nothing.
# mpi: True/False = recorded; None = not recorded (no profile section applies).
MATRIX = [
    # Frontier, ROCm: the profile on a multi-node step; its start-up requirements under any transport.
    ("frontier", "rocm", "2node", "none", "frontier profile", "site", True),
    ("frontier", "rocm", "2node", "net", "explicit transport", {}, True),
    ("frontier", "rocm", "2node", "plugin", "explicit transport", {}, True),
    ("frontier", "rocm", "1node", "none", "unchanged", {}, None),
    ("frontier", "rocm", "1node", "plugin", "explicit transport", {}, None),
    ("frontier", "rocm", "unknown", "none", "unchanged", {}, None),
    # Frontier's profile has no [nccl]: a CUDA leg there gets no profile.
    ("frontier", "cuda", "2node", "none", "unchanged", {}, None),
    ("frontier", "cpu", "2node", "none", "no device collectives", {}, None),
    # Perlmutter, CUDA: the profile on a multi-node step; no MPI-first requirement.
    ("perlmutter", "cuda", "2node", "none", "perlmutter profile", "site", False),
    ("perlmutter", "cuda", "2node", "net", "explicit transport", {}, False),
    ("perlmutter", "cuda", "2node", "plugin", "explicit transport", {}, False),
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
    want = {"frontier": FRONTIER_ENV, "perlmutter": PERLMUTTER_ENV}[site] if applied == "site" else applied
    assert result["applied"] == want
    assert result.get("mpi_before_collectives") == mpi
    if leg != "cpu":
        assert result["library"] == {"rocm": "RCCL", "cuda": "NCCL"}[leg]


@pytest.mark.parametrize("leg, plugin_file", [("cuda", "libnccl-net.so"), ("rocm", "librccl-net.so")])
def test_unprofiled_multi_node_warns_with_the_legs_plugin_name(leg, plugin_file):
    result = plan(env_of("none", leg, "2node", "none"))
    assert len(result["warnings"]) == 1 and plugin_file in result["warnings"][0]


@pytest.mark.parametrize("leg", ["cuda", "rocm"])
def test_unprofiled_slingshot_applies_only_the_generic_cxi_settings(leg):
    result = plan(env_of("none", leg, "2node", "none"), cxi=True)
    assert result["policy"] == "generic Slingshot" and result["applied"] == ne._CXI_DEFAULTS
    assert result.get("mpi_before_collectives") is None


# --- refusals ----------------------------------------------------------------

@pytest.mark.parametrize("site, leg", [("frontier", "rocm"), ("perlmutter", "cuda")])
def test_a_profiled_multi_node_socket_transport_refuses(site, leg):
    with pytest.raises(RuntimeError, match="NCCL_NET=Socket is set"):
        plan(env_of(site, leg, "2node", "none", NCCL_NET="Socket"))
    # One node: Socket is the caller's business.
    assert plan(env_of(site, leg, "1node", "none", NCCL_NET="Socket"))["policy"] == "explicit transport"


@pytest.mark.parametrize("network, refuses", [("", True), ("vni", True), ("no_vni", False),
                                              ("exclusive, no_vni", False)])
def test_perlmutter_step_network_is_read_as_a_comma_list(network, refuses):
    env = env_of("perlmutter", "cuda", "2node", "none", SLURM_NETWORK=network)
    if refuses:
        with pytest.raises(RuntimeError, match="SLURM_NETWORK=no_vni"):
            plan(env)
    else:
        assert plan(env)["policy"] == "perlmutter profile"


def test_frontier_names_no_step_network():
    env = env_of("frontier", "rocm", "2node", "none")
    del env["SLURM_NETWORK"]
    assert plan(env)["policy"] == "frontier profile"


@pytest.mark.parametrize("site, leg", [("frontier", "rocm"), ("perlmutter", "cuda")])
def test_a_missing_site_plugin_refuses_by_path(site, leg):
    plugin = {"frontier": FRONTIER_ENV, "perlmutter": PERLMUTTER_ENV}[site]["NCCL_NET_PLUGIN"]
    with pytest.raises(RuntimeError, match="cannot find the site OFI plugin " + plugin):
        plan(env_of(site, leg, "2node", "none"), is_file=lambda path: path != plugin)


@pytest.mark.parametrize("transport", ["none", "net", "plugin"])
def test_frontier_launch_libraries_refuse_under_any_transport(transport):
    missing_hsa = lambda path: not path.endswith("libhsa-runtime64.so")
    with pytest.raises(RuntimeError, match="GATE launch_libraries.*libhsa-runtime64.so.*rocm_env.sh"):
        plan(env_of("frontier", "rocm", "2node", transport), is_file=missing_hsa)
    # Not on one node, and never on Perlmutter (no launch_libraries there).
    plan(env_of("frontier", "rocm", "1node", transport), is_file=missing_hsa)
    plan(env_of("perlmutter", "cuda", "2node", transport), is_file=missing_hsa)


def test_launch_libraries_need_the_name_on_the_launch_path():
    env = env_of("frontier", "rocm", "2node", "none", LD_LIBRARY_PATH="")
    with pytest.raises(RuntimeError, match="GATE launch_libraries"):
        plan(env)


# --- site selection ----------------------------------------------------------

def test_lorrax_site_profile_takes_precedence():
    env = env_of("frontier", "cuda", "2node", "none", LORRAX_SITE_PROFILE="perlmutter")
    assert plan(env)["policy"] == "perlmutter profile"


def test_an_unknown_named_site_refuses_but_an_unknown_machine_name_is_ignored():
    with pytest.raises(RuntimeError, match="names no site profile"):
        plan(env_of("none", "cuda", "2node", "none", LORRAX_SITE_PROFILE="nowhere"))
    assert plan(env_of("none", "cuda", "2node", "none", NERSC_HOST="nowhere"))["policy"] == "unchanged"


# --- topology: the STEP, never the allocation --------------------------------

@pytest.mark.parametrize("env, want", [
    ({"SLURM_STEP_NUM_NODES": "2"}, (2, True, "Slurm step")),
    ({"SLURM_STEP_NUM_NODES": "1"}, (1, False, "Slurm step")),
    ({"SLURM_STEP_NODELIST": "nid[001000-001003]"}, (4, True, "Slurm step")),
    ({"SLURM_STEP_NODELIST": "frontier[07208,07210]"}, (2, True, "Slurm step")),
    ({"SLURM_STEP_NODELIST": "nid001000,nid001000"}, (1, False, "Slurm step")),
    ({"SLURM_STEP_NUM_NODES": "0", "SLURM_STEP_NODELIST": "a[1-2]"}, (2, True, "Slurm step")),
    ({"SLURM_STEP_NUM_NODES": "x", "SLURM_STEP_NODELIST": "a1"}, (1, False, "Slurm step")),
    ({"OMPI_COMM_WORLD_SIZE": "8", "OMPI_COMM_WORLD_LOCAL_SIZE": "8"}, (1, False, "Open MPI local group")),
    ({"OMPI_COMM_WORLD_SIZE": "8", "OMPI_COMM_WORLD_LOCAL_SIZE": "4"}, (None, True, "Open MPI local group")),
    # The enclosing allocation is not evidence about this step.
    ({"SLURM_NNODES": "4", "SLURM_NODELIST": "nid[1-4]", "SLURM_JOB_NUM_NODES": "4"},
     (None, None, "unknown")),
    ({}, (None, None, "unknown")),
])
def test_launch_topology(env, want):
    assert ne.launch_topology(env) == want


def test_a_multi_node_step_of_unknown_size_still_takes_the_profile():
    env = {**SITE["frontier"], **LEG["rocm"], **LAUNCH,
           "OMPI_COMM_WORLD_SIZE": "16", "OMPI_COMM_WORLD_LOCAL_SIZE": "8"}
    result = plan(env)
    assert result["nodes"] is None and result["multi_node"] is True
    assert result["policy"] == "frontier profile" and result["mpi_before_collectives"] is True


def test_an_unknown_topology_writes_nothing_even_on_a_profiled_site():
    for site, leg in (("frontier", "rocm"), ("perlmutter", "cuda")):
        result = plan(env_of(site, leg, "unknown", "none"))
        assert result["policy"] == "unchanged" and result["applied"] == {}


# --- caller exports win ------------------------------------------------------

@pytest.mark.parametrize("site, leg", [("frontier", "rocm"), ("perlmutter", "cuda")])
def test_every_caller_export_wins_over_the_site_default(site, leg):
    defaults = {"frontier": FRONTIER_ENV, "perlmutter": PERLMUTTER_ENV}[site]
    for name in defaults:
        if name in ("NCCL_NET", "NCCL_NET_PLUGIN"):
            continue                   # these make the step an explicit transport (matrix above)
        result = plan(env_of(site, leg, "2node", "none", **{name: "caller"}))
        assert name not in result["applied"]
        assert set(result["applied"]) == set(defaults) - {name}


# --- start-up application ----------------------------------------------------

@pytest.fixture
def fresh(monkeypatch):
    """A clean process environment for configure_gpu_network, host probes stubbed."""
    for name in list(os.environ):
        if name.startswith(("NCCL_", "FI_", "SLURM_", "OMPI_", "HSA_FORCE")) or name in (
                "LMOD_SYSTEM_NAME", "NERSC_HOST", "LORRAX_SITE_PROFILE", "JAX_PLATFORMS"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(ne, "_configuration", None)
    real = ne.plan_network_environment
    monkeypatch.setattr(ne, "plan_network_environment",
                        lambda env, **kw: real(env, is_file=lambda path: True, cxi=False, **kw))
    said = []

    def setup(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return said
    return setup


def test_configure_applies_exactly_the_decision_once(fresh):
    said = fresh(**env_of("frontier", "rocm", "2node", "none"))
    first = ne.configure_gpu_network(say=said.append)
    assert first["policy"] == "frontier profile" and first["mpi_before_collectives"] is True
    assert {name: os.environ[name] for name in FRONTIER_ENV} == FRONTIER_ENV
    assert any("frontier profile" in line and "set NCCL_NET" in line for line in said)
    # Process-lifetime decision: a second call changes nothing and repeats the record.
    os.environ["NCCL_CROSS_NIC"] = "caller-after"
    assert ne.configure_gpu_network(say=said.append) == first
    assert os.environ["NCCL_CROSS_NIC"] == "caller-after"
    assert ne.network_configuration() == first


def test_configure_writes_nothing_for_a_single_node_step(fresh):
    said = fresh(**env_of("frontier", "rocm", "1node", "none"))
    before = dict(os.environ)
    decision = ne.configure_gpu_network(say=said.append)
    assert decision["policy"] == "unchanged" and dict(os.environ) == before


def test_configure_must_precede_the_environment_freeze(fresh):
    """The order the runtime keeps (set_default_env, step 1, before step 7b):
    after the freeze, the multi-node writes refuse by name; one node writes nothing."""
    fresh(**env_of("frontier", "rocm", "2node", "none"))
    env_guard.freeze_process_environment("test")
    try:
        # os.environ.update reaches putenv through _collections_abc: the writer is network_env.
        with pytest.raises(RuntimeError, match="GATE env-frozen.*'NCCL_NET'.*network_env.py"):
            ne.configure_gpu_network(say=lambda line: None)
    finally:
        env_guard.thaw_for_tests()


def test_a_single_node_configure_is_safe_after_the_freeze(fresh):
    fresh(**env_of("perlmutter", "cuda", "1node", "none"))
    env_guard.freeze_process_environment("test")
    try:
        assert ne.configure_gpu_network(say=lambda line: None)["applied"] == {}
    finally:
        env_guard.thaw_for_tests()


@pytest.mark.parametrize("write", ["item", "update", "setdefault"])
def test_the_freeze_attributes_every_mapping_write_to_its_caller(write, monkeypatch):
    """GATE env-frozen judges the writer, not the stdlib frames between it and putenv."""
    seen = []
    monkeypatch.setattr(env_guard, "_own", lambda path: seen.append(path) or False)
    monkeypatch.delenv("LORRAX_TEST_FREEZE_KEY", raising=False)
    env_guard.freeze_process_environment("test")
    try:
        {"item": lambda: os.environ.__setitem__("LORRAX_TEST_FREEZE_KEY", "1"),
         "update": lambda: os.environ.update(LORRAX_TEST_FREEZE_KEY="1"),
         "setdefault": lambda: os.environ.setdefault("LORRAX_TEST_FREEZE_KEY", "1")}[write]()
    finally:
        env_guard.thaw_for_tests()
        os.environ.pop("LORRAX_TEST_FREEZE_KEY", None)
    assert seen and seen[0].endswith("test_network_env_multinode.py")
