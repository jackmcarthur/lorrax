"""``RuntimeStack.reshape`` must update the facts and the report it swaps under.

Before 2026-08-28, ``reshape`` swapped ``self.mesh`` and left ``self.facts``
and ``self.report`` describing the startup mesh.  Every consumer that reads
the stack instead of the mesh — ``common.scientific_output.architecture_lines``
(the "Processor mesh : gx x gy" line every production driver prints) and any
test or probe grepping ``RUNTIME.report`` — then stated a shape the run was
not using.  Measured on the pre-fix tree in a 4-device CPU child: after
``reshape(2, 2)`` from a 4x1 start, ``facts["mesh_shape"]`` was still
``(4, 1)`` and the report still said "mesh 4x1".

The child process follows the house wide-process pattern
(tests/test_bse_gather_and_mesh.py): the parent suite must not be widened, so
the four emulated CPU devices exist only in the child.  The child builds the
stack directly rather than through ``initialize_communicator_stack`` — this
is a question about ``reshape``'s bookkeeping, and the full entry point would
drag in distributed bring-up that has nothing to say about it.
"""
import os
import subprocess
import sys


_WIDE_CHILD = r"""
import numpy as np
import jax
from jax.sharding import Mesh
import runtime
from common.scientific_output import architecture_lines

devices = jax.devices()
print("DEVICES", len(devices))
# A 4x1 start so that reshape(2, 2) is a real swap, not the no-op arm.
mesh = Mesh(np.asarray(devices).reshape(4, 1), ("x", "y"))
facts = runtime.collect_startup_facts(mesh)
report = runtime.format_production_startup_report(facts)
stack = runtime.RuntimeStack(
    mesh=mesh, platform=facts["backend"], device_kind=facts["device_kind"],
    n_devices=facts["n_devices"], n_local_devices=facts["n_local_devices"],
    process_index=facts["process_index"],
    process_count=facts["process_count"], facts=facts, report=report)
print("START_FACTS_SHAPE", stack.facts["mesh_shape"])
print("START_REPORT_SAYS_4X1", any("mesh 4x1" in l for l in stack.report))

stack.reshape(2, 2, print_fn=lambda *a, **k: None)

print("MESH_SHAPE", stack.mesh_shape)
print("FACTS_SHAPE", stack.facts["mesh_shape"])
print("FACTS_AXES", stack.facts["mesh_axes"])
print("FACTS_HAS_LINALG", "linalg" in stack.facts)
print("REPORT_SAYS_2X2", any("mesh 2x2" in l for l in stack.report))
print("REPORT_SAYS_4X1", any("mesh 4x1" in l for l in stack.report))
print("ARCH_SAYS_2X2",
      any("Processor mesh : 2 x 2" in l for l in architecture_lines(stack)))
"""


def _run_wide_child(n_devices: int = 4):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    # The child must import what this process imports; under ``lx test`` the
    # source root arrives on sys.path, not necessarily in PYTHONPATH.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run([sys.executable, "-c", _WIDE_CHILD],
                          capture_output=True, text=True, env=env,
                          timeout=900)
    assert proc.returncode == 0, (
        f"the widened child died (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}")
    facts = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isupper():
            facts[parts[0]] = parts[1].strip()
    return facts, proc.stdout


def test_reshape_updates_the_facts_and_the_report():
    facts, out = _run_wide_child(4)
    assert facts.get("DEVICES") == "4", (
        f"the child did not widen — reshape(2, 2) needs 4 devices, so the "
        f"test would refuse rather than measure.  Child said:\n{out}")
    # Controls: the startup records really did say 4x1, so the assertions
    # below are capable of returning False.
    assert facts.get("START_FACTS_SHAPE") == "(4, 1)", out
    assert facts.get("START_REPORT_SAYS_4X1") == "True", out
    # The swap itself.
    assert facts.get("MESH_SHAPE") == "(2, 2)", out
    # The defect: these three were stale before the fix
    # (facts (4, 1), report "mesh 4x1", architecture line "4 x 1").
    assert facts.get("FACTS_SHAPE") == "(2, 2)", (
        f"facts['mesh_shape'] still names the startup mesh after reshape.  "
        f"Child said:\n{out}")
    assert facts.get("REPORT_SAYS_2X2") == "True" and \
        facts.get("REPORT_SAYS_4X1") == "False", (
        f"the report still describes the startup mesh after reshape.  "
        f"Child said:\n{out}")
    assert facts.get("ARCH_SAYS_2X2") == "True", (
        f"scientific_output.architecture_lines read a stale shape.  "
        f"Child said:\n{out}")
    # The other mesh-derived facts entries were refreshed, not dropped.
    assert facts.get("FACTS_AXES") == "('x', 'y')", out
    assert facts.get("FACTS_HAS_LINALG") == "True", out
