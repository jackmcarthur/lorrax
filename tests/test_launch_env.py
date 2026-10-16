"""Launcher-agnostic world resolution, the coordinator address and the site network profile.

Plain Python on a login node: every case is an environment simulation.  Only
the coordinator refusal imports jax, and it fails before any network call.
"""
import os
import subprocess
import sys

import pytest

import runtime
from runtime import network_env as ne

#: Every variable either resolver reads; each test starts from none of them.
_LAUNCH_VARS = (*runtime._WORLD_SIZE_ENVS, *runtime._WORLD_RANK_ENVS,
                "JAX_COORDINATOR_ADDRESS", "SLURM_STEP_NODELIST", "SLURM_JOB_ID",
                "SLURM_STEP_ID", "SLURM_NODELIST", "SLURM_STEP_NUM_NODES",
                "OMPI_COMM_WORLD_LOCAL_SIZE", "OMPI_MCA_orte_hnp_uri", "SLURM_LOCALID")

LAUNCHERS = {
    # srun on Perlmutter: Cray PMI and Slurm agree.
    "srun": dict(SLURM_NTASKS="8", SLURM_PROCID="5", PMI_SIZE="8", PMI_RANK="5"),
    # hydra inside Slurm (Frontera's ibrun): one Slurm task per node, so the
    # ranks of node 1 all inherit SLURM_PROCID=1.
    "ibrun": dict(SLURM_NTASKS="2", SLURM_PROCID="1", PMI_SIZE="8", PMI_RANK="6",
                  MV2_COMM_WORLD_SIZE="8", MV2_COMM_WORLD_RANK="6"),
    "ompi": dict(OMPI_COMM_WORLD_SIZE="4", OMPI_COMM_WORLD_RANK="3"),
    "pmi": dict(PMI_SIZE="4", PMI_RANK="2"),
    "jax": dict(JAX_PROCESS_COUNT="2", JAX_PROCESS_INDEX="1", SLURM_NTASKS="4",
                SLURM_PROCID="3"),
    "none": {},
}
WORLDS = {"srun": (8, 5), "ibrun": (8, 6), "ompi": (4, 3), "pmi": (4, 2),
          "jax": (2, 1), "none": (1, 0)}


@pytest.fixture
def launch(monkeypatch):
    for name in _LAUNCH_VARS:
        monkeypatch.delenv(name, raising=False)

    def set_env(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    return set_env


@pytest.mark.parametrize("launcher", sorted(LAUNCHERS))
def test_world_per_launcher(launch, launcher):
    launch(**LAUNCHERS[launcher])
    assert (runtime._resolve_proc_count(), runtime._resolve_proc_id()) == WORLDS[launcher]
    runtime._refuse_split_launch()


def test_split_launch_refuses_by_name(launch):
    launch(JAX_PROCESS_COUNT="1", SLURM_NTASKS="4", SLURM_PROCID="2")
    with pytest.raises(RuntimeError, match="GATE split_launch") as err:
        runtime._refuse_split_launch()
    assert "SLURM_NTASKS=4" in str(err.value) and "Unset JAX_PROCESS_COUNT" in str(err.value)
    launch(JAX_PROCESS_COUNT="4")
    runtime._refuse_split_launch()


def test_single_process_does_not_touch_jax_or_the_environment(launch):
    before = dict(os.environ)
    runtime.init_jax_distributed()
    assert dict(os.environ) == before
    assert "single-process" in runtime._DISTRIBUTED_FORM[0]


@pytest.mark.parametrize("expr, hosts", [
    ("nid001000", ["nid001000"]),
    ("nid[001000-001003,001010]",
     ["nid001000", "nid001001", "nid001002", "nid001003", "nid001010"]),
    ("nid[001000-001001],nid002000,login[07,09-10]",
     ["nid001000", "nid001001", "nid002000", "login07", "login09", "login10"]),
    ("rack[1-2]-n[08-09]", ["rack1-n08", "rack1-n09", "rack2-n08", "rack2-n09"]),
    ("c1,c2", ["c1", "c2"]),
])
def test_expand_nodelist(expr, hosts):
    assert ne.expand_nodelist(expr) == hosts


def test_coordinator_is_the_steps_first_host_on_a_job_and_step_port(launch):
    launch(SLURM_STEP_NODELIST="nid[001002-001003,001010]", SLURM_JOB_ID="59753236",
           SLURM_STEP_ID="3", SLURM_NODELIST="nid[001000-001010]")
    host, port = runtime._resolve_coordinator_address().split(":")
    assert host == "nid001002" and 61440 <= int(port) <= 65535
    ports = set()
    for job, step in [("59753236", "3"), ("59753236", "4"), ("59753237", "3")]:
        launch(SLURM_JOB_ID=job, SLURM_STEP_ID=step)
        ports.add(runtime._resolve_coordinator_address())
    assert len(ports) == 3
    launch(JAX_COORDINATOR_ADDRESS="10.0.0.1:1234")
    assert runtime._resolve_coordinator_address() == "10.0.0.1:1234"


def test_no_step_means_no_local_host_fallback(launch):
    launch(SLURM_NODELIST="nid[001000-001001]", SLURM_JOB_ID="1", PMI_SIZE="2", PMI_RANK="1")
    assert runtime._resolve_coordinator_address() is None


def test_no_coordinator_refuses_naming_the_remedy(launch):
    # A fresh process: jax.distributed.initialize refuses once a backend exists.
    env = {**os.environ, "PMI_SIZE": "2", "PMI_RANK": "1", "JAX_PLATFORMS": "cpu"}
    child = subprocess.run([sys.executable, "-c", "import runtime; runtime.init_jax_distributed()"],
                           env=env, capture_output=True, text=True, timeout=120)
    assert child.returncode != 0 and "GATE coordinator_address" in child.stderr
    assert "JAX_COORDINATOR_ADDRESS" in child.stderr and "PMI_SIZE" in child.stderr


# --- site network profile ----------------------------------------------------

PERLMUTTER_DEFAULTS = {
    "NCCL_NET": "AWS Libfabric",
    "NCCL_NET_PLUGIN": "/global/common/software/nersc9/nccl/2.29.2-cu13/plugin/lib/libnccl-net.so",
    "NCCL_NET_GDR_LEVEL": "PHB", "FI_CXI_DISABLE_HOST_REGISTER": "1",
    "FI_CXI_RDZV_THRESHOLD": "0", "NCCL_CROSS_NIC": "2", "NCCL_SOCKET_IFNAME": "hsn",
}
TWO_NODES = dict(SLURM_STEP_NODELIST="nid[001000-001001]", SLURM_NETWORK="no_vni")


def plan(env, **kw):
    return ne.plan_network_environment(env, is_file=lambda path: True, **kw)


def test_perlmutter_profile_applies_the_site_module_settings():
    result = plan({"NERSC_HOST": "perlmutter", **TWO_NODES})
    assert result["nodes"] == 2 and result["policy"] == "perlmutter profile"
    assert result["applied"] == PERLMUTTER_DEFAULTS and not result["warnings"]
    kept = plan({"LORRAX_SITE_PROFILE": "perlmutter", "NCCL_CROSS_NIC": "1", **TWO_NODES})
    assert "NCCL_CROSS_NIC" not in kept["applied"]
    # The base module exports LORRAX_SITE as a site-packages path; it selects nothing.
    module = plan({"LORRAX_SITE": "/x/lib/python3.12/site-packages", "NERSC_HOST": "perlmutter",
                   **TWO_NODES})
    assert module["policy"] == "perlmutter profile"


@pytest.mark.parametrize("env, cause", [
    ({"SLURM_STEP_NUM_NODES": "2"}, "SLURM_NETWORK=no_vni"),
    ({**TWO_NODES, "NCCL_NET": "Socket"}, "NCCL_NET=Socket"),
])
def test_perlmutter_profile_refuses_a_sockets_step(env, cause):
    with pytest.raises(RuntimeError, match=cause):
        plan({"NERSC_HOST": "perlmutter", **env})


def test_perlmutter_profile_refuses_a_missing_plugin():
    with pytest.raises(RuntimeError, match="cannot find the site OFI plugin"):
        ne.plan_network_environment({"NERSC_HOST": "perlmutter", **TWO_NODES},
                                    is_file=lambda path: False)


@pytest.mark.parametrize("env, policy", [
    ({"NERSC_HOST": "perlmutter", "SLURM_STEP_NUM_NODES": "1"}, "unchanged"),
    ({"NERSC_HOST": "perlmutter", "NCCL_NET": "AWS Libfabric", **TWO_NODES},
     "explicit transport"),
    ({"NERSC_HOST": "perlmutter", "JAX_PLATFORMS": "cpu", **TWO_NODES}, "no device collectives"),
])
def test_profile_leaves_other_steps_alone(env, policy):
    result = plan(env)
    assert result["policy"] == policy and result["applied"] == {}


def test_unprofiled_slingshot_gets_generic_cxi_settings_and_warnings():
    result = plan({"NERSC_HOST": "elsewhere", **TWO_NODES}, cxi=True)
    assert result["policy"] == "generic Slingshot"
    assert result["applied"] == ne._CXI_DEFAULTS
    assert len(result["warnings"]) == 2 and "elsewhere" in result["warnings"][0]
    plain = plan(dict(TWO_NODES), cxi=False)
    assert plain["applied"] == {} and len(plain["warnings"]) == 1
    assert "neither NCCL_NET nor a net plugin" in plain["warnings"][0]


def test_an_unknown_explicit_site_refuses():
    with pytest.raises(RuntimeError, match="LORRAX_SITE_PROFILE='nowhere'"):
        plan({"LORRAX_SITE_PROFILE": "nowhere", **TWO_NODES})


def test_perlmutter_profile_carries_the_comm_model_row():
    row = ne.site_profile({"NERSC_HOST": "perlmutter"})["comm_model"]
    assert row["beta_Bps"] == 19.7e9 and row["alpha0_s"] == 31.6e-6


# --- Frontier (ROCm, RCCL) ----------------------------------------------------

FRONTIER = dict(LMOD_SYSTEM_NAME="frontier", JAX_PLATFORMS="rocm,cpu",
                SLURM_STEP_NODELIST="frontier[07208,07210]")


def test_frontier_profile_applies_the_rccl_module_settings():
    result = plan({**FRONTIER, "LD_LIBRARY_PATH": "/shim"})
    assert result["policy"] == "frontier profile" and result["library"] == "RCCL"
    applied = result["applied"]
    assert applied["NCCL_NET"] == "AWS Libfabric" and applied["NCCL_NET_GDR_LEVEL"] == "PHB"
    assert applied["NCCL_NET_PLUGIN"].endswith("/rocm/7.13.0/lib/librccl-net.so")
    assert applied["FI_CXI_RDZV_THRESHOLD"] == "0" and applied["FI_MR_CACHE_MONITOR"] == "kdreg2"
    assert result["mpi_before_collectives"] is True
    assert plan({"NERSC_HOST": "perlmutter", **TWO_NODES})["mpi_before_collectives"] is False
    kept = plan({**FRONTIER, "LD_LIBRARY_PATH": "/shim", "FI_MR_CACHE_MONITOR": "memhooks"})
    assert "FI_MR_CACHE_MONITOR" not in kept["applied"]


def test_frontier_profile_refuses_a_launch_without_the_hsa_name():
    with pytest.raises(RuntimeError, match="GATE launch_libraries.*libhsa-runtime64.so.*rocm_env.sh"):
        ne.plan_network_environment({**FRONTIER, "LD_LIBRARY_PATH": "/a:/b"},
                                    is_file=lambda path: not path.endswith("libhsa-runtime64.so"))


def test_frontier_explicit_transport_keeps_the_site_start_up_requirements():
    # rccl-net-plugin exports the transport: nothing is filled in, but MPI must
    # still come before RCCL and the HSA name must still be on the launch path.
    module = {**FRONTIER, "LD_LIBRARY_PATH": "/shim",
              "NCCL_NET_PLUGIN": "/sw/frontier/rccl-plugins/x/librccl-net.so"}
    result = plan(module)
    assert result["policy"] == "explicit transport" and result["applied"] == {}
    assert result["mpi_before_collectives"] is True
    with pytest.raises(RuntimeError, match="GATE launch_libraries.*libhsa-runtime64.so"):
        ne.plan_network_environment({**module, "NCCL_NET": "AWS Libfabric"},
                                    is_file=lambda path: not path.endswith("libhsa-runtime64.so"))
    single = plan({**module, "SLURM_STEP_NODELIST": "frontier07208"})
    assert single["policy"] == "explicit transport" and "mpi_before_collectives" not in single


def test_perlmutter_explicit_transport_is_unchanged():
    result = plan({"NERSC_HOST": "perlmutter", "NCCL_NET": "AWS Libfabric",
                   "SLURM_STEP_NODELIST": "nid[001000-001001]"})
    assert result["policy"] == "explicit transport" and result["applied"] == {}
    assert result["mpi_before_collectives"] is False


def test_rocm_single_node_and_cpu_keep_the_library_policy():
    one = plan({**FRONTIER, "SLURM_STEP_NODELIST": "frontier07208"})
    assert one["policy"] == "unchanged" and one["applied"] == {}
    cpu = plan({**FRONTIER, "JAX_PLATFORMS": "cpu"})
    assert cpu["policy"] == "no device collectives"


def test_the_frontier_site_rule_sets_scratch(tmp_path):
    import runtime
    work = tmp_path / "mat280"
    work.mkdir()
    env = {"LMOD_SYSTEM_NAME": "frontier", "MEMBERWORK": str(tmp_path), "SLURM_JOB_ACCOUNT": "mat280"}
    assert runtime.site_scratch(env) == (str(work), "frontier profile")
    path, why = runtime.site_scratch({k: v for k, v in env.items() if k != "SLURM_JOB_ACCOUNT"})
    assert path is None and "does not expand" in why
    assert runtime.site_scratch({**env, "SCRATCH": "/s"}) == (None, "SCRATCH is set")


def test_the_profiler_gate_refuses_rocprofv3_above_one_process(monkeypatch):
    import runtime
    monkeypatch.setenv("JAX_PLATFORMS", "rocm,cpu")
    # The world size is the first of runtime's launcher variables that is set; inside an
    # srun step the step's own variables would answer, so clear them all first.
    for name in runtime._WORLD_SIZE_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "4")
    env = {"LD_PRELOAD": "/x/librocprofiler-sdk-tool.so"}
    with pytest.raises(RuntimeError, match="GATE profiler.*rocprofv3"):
        runtime.check_profiler_ranks(environ=env, maps_text="")
    runtime.check_profiler_ranks(environ={}, maps_text="")          # not profiled
    monkeypatch.setenv("SLURM_NTASKS", "1")
    runtime.check_profiler_ranks(environ=env, maps_text="")         # one process
