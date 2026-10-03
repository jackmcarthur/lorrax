"""Lockstep eigh-stack route decision across processes, on host CPU (2 processes x 2 devices).

Each rank is handed a different compiled size for the same candidate
(``_stack_bytes`` is replaced, so nothing is compiled): rank 0 fits the
first candidate, rank 1 does not. The decision must be the same on both
ranks, both must walk the same candidates in the same order, and neither may
hang. A second stack exercises the shape-bound rejection (no candidate is
sized); a third makes rank 1's sizing raise at every candidate, which must
reject them all on both ranks without a hang. Run with no arguments; it starts its two ranks itself and exits 1 on
any disagreement or a timeout.

    python3 services/distrib_la/bench/stack_route_lockstep_check.py
"""
import json
import os
import socket
import subprocess
import sys

TIMEOUT_S = 180
ROOM = 10_000


def rank_main(rank, port):
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    import importlib
    import jax
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)
    jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=2, process_id=rank)
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    plan_mod = importlib.import_module("distrib_la.plan")

    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    calls = []

    def injected(op, mesh_, nb, n, dtype, rounds, site, phase):
        calls.append((int(nb), int(rounds)))
        if nb == 7 and rank == 1:           # the sizing compile fails on one rank only
            raise RuntimeError("injected compile failure")
        if nb == 7:
            return ROOM // 4, 0.0
        if rounds == 1:                     # two whole matrices per rank
            return (ROOM // 2 if rank == 0 else ROOM + 1), 0.0
        return ROOM // 4, 0.0               # one whole matrix per rank, two rounds
    plan_mod._stack_bytes = injected

    p = plan_mod.Plan(op="eigh", requested="distributed", backend="scalapack", mesh=mesh, n=4,
                      in_sharding=NamedSharding(mesh, P("x", "y")),
                      batch_in_sharding=NamedSharding(mesh, P(None, "x", "y")),
                      requested_batched_route="auto", budget_bytes=ROOM)
    decided = p.stack_route((5, 4, 4), np.complex128)
    # Shape bound: 2 n^2 x 16 B per whole matrix exceeds the room, no candidate sized.
    rejected = p.stack_route((5, 32, 32), np.complex128)
    # One rank's sizing raises at every candidate: both ranks reject them all.
    failed = p.stack_route((7, 4, 4), np.complex128)
    print("RESULT " + json.dumps(dict(rank=rank, route=decided.route, per_rank=decided.per_rank,
                                      rounds=decided.rounds, bytes=decided.program_bytes, calls=calls,
                                      rejected=rejected.route, rejected_bytes=rejected.program_bytes,
                                      failed=failed.route, failed_bytes=failed.program_bytes)),
          flush=True)
    jax.distributed.shutdown()


def main():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    procs = [subprocess.Popen([sys.executable, __file__, str(r), str(port)], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) for r in (0, 1)]
    results, failed = [], []
    for r, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            print(f"stack_route_lockstep FAILED: rank {r} hung past {TIMEOUT_S} s", flush=True)
            return 1
        lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
        if proc.returncode or not lines:
            print(out[-3000:])
            failed.append(f"rank {r} rc={proc.returncode}")
            continue
        results.append(json.loads(lines[-1][7:]))
    for row in results:
        print("stack_route_lockstep", json.dumps(row), flush=True)
    if len(results) == 2:
        keys = ("route", "per_rank", "rounds", "bytes", "calls", "rejected", "rejected_bytes",
                "failed", "failed_bytes")
        if any(results[0][k] != results[1][k] for k in keys):
            failed.append("ranks disagree")
        if results[0]["route"] != "batch_reshard" or results[0]["rounds"] != 2 \
                or results[0]["bytes"] != ROOM // 4:
            failed.append("wrong decision: want batch_reshard, 1 per rank, 2 rounds at the agreed size")
        if results[0]["rejected"] == "batch_reshard" or results[0]["rejected_bytes"] is not None:
            failed.append("shape bound did not reject without a size")
        if results[0]["failed"] == "batch_reshard":
            failed.append("a one-rank sizing failure did not reject on every rank")
    print("stack_route_lockstep " + ("FAILED: " + "; ".join(failed) if failed else "PASS"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 3:
        rank_main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        sys.exit(main())
