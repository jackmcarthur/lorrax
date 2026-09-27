"""``sweep_q_wedge(z_outer=True)``: one z at a time, handed off per z.

CPU, no device work: the engine, payload and solve are replaced by recorders,
so this pins the ORDER of the walk and the per-z hand-off, not the numbers
(the numbers are the facade's own gates, and the 2-z wedge equality is
measured on Si 4^3 SOC, claim 2861).
"""
import pytest


def _fake_engine(monkeypatch, wl, calls):
    def build(mesh, data, include_w):
        data["W_R"] = 0
        return "mv", None, "gen", "snap", "sh"

    monkeypatch.setattr(wl, "enforce_trs_pair_gauge", lambda d, m: d)
    monkeypatch.setattr(wl, "build_ladder_resolvent", build)
    monkeypatch.setattr(wl, "build_finite_q_data",
                        lambda d, q, m: dict(d, q=q))
    monkeypatch.setattr(wl, "build_preconditioner_diagonal_sharded",
                        lambda d, m, include_W, use_tda: "diag")
    monkeypatch.setattr(wl, "build_probe_rhs", lambda G, d, gen, sh: "rhs")

    def solve(G, z, dq, *a, **k):
        calls.append(("solve", dq["q"], complex(z)))
        return "W", "resid", "iters"

    monkeypatch.setattr(wl, "apply_screening_resolvent_block", solve)


@pytest.mark.parametrize("z_outer", [True, False])
def test_two_z_walk_order_and_per_z_handoff(monkeypatch, z_outer):
    pytest.importorskip("jax")
    from bse import w_ladder as wl

    calls = []
    _fake_engine(monkeypatch, wl, calls)
    q_list = [(0, 0, 0), (1, 0, 0)]
    z_list = [0.0, 0.5j]
    wl.sweep_q_wedge(
        {}, None, q_list, z_list, include_w=False,
        probe_blocks_for_q=lambda iq, q: [(0, 1, "G")],
        gmres_tol=1e-6, gmres_max_iter=300, deflation_rank=0,
        on_result=lambda iq, q, iz, z, *a: calls.append(("result", iq, iz)),
        z_outer=z_outer,
        on_z_done=lambda iz, z: calls.append(("done", iz)))
    results = [c for c in calls if c[0] in ("result", "done")]
    if z_outer:
        assert results == [("result", 0, 0), ("result", 1, 0), ("done", 0),
                           ("result", 0, 1), ("result", 1, 1), ("done", 1)]
    else:
        assert results == [("result", 0, 0), ("result", 0, 1),
                           ("result", 1, 0), ("result", 1, 1)]
    solved = [(c[1], c[2]) for c in calls if c[0] == "solve"]
    assert sorted(solved, key=str) == sorted(
        [(q, complex(z)) for q in q_list for z in z_list], key=str)
