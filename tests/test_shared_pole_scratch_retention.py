"""SC shared-pole scratch keeps one map generation (MPA's retention rule).

Each SC map screens into ``tmp/mpa/sc_NNNN_shared_pole/``; a file-tier bank
there is 0.87 TB on Fe 8^3, so without retention the scratch grows linearly in
maps.  Only exact managed generation names may go; the current map's stays, and the
photon static reference beside the generations (run-lifetime) is never touched.
"""
from gw import shared_pole_screening


def test_shared_pole_scratch_retains_only_the_current_map(tmp_path, monkeypatch):
    root = tmp_path / "mpa"
    for label in ("sc_0000", "sc_0001", "sc_0002", "sc_0007"):
        generation = root / f"{label}_shared_pole"
        generation.mkdir(parents=True)
        (generation / "bank.h5").write_bytes(b"x")
        (generation / "bank_receipt.json").write_text("{}")
    unrelated = [root / "oneshot_shared_pole", root / "sc_0000_shared_pole.h5",
                 root / "mpa_fit_sc_0000.h5",
                 root / "sc_0000_shared_pole_photon_static_reference.h5"]
    unrelated[0].mkdir()
    for path in unrelated[1:]:
        path.touch()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "keep.h5").touch()
    (root / "sc_0003_shared_pole").symlink_to(outside, target_is_directory=True)
    barriers = []
    monkeypatch.setattr("common.collectives.process_rank", lambda: 0)
    monkeypatch.setattr("common.collectives.barrier",
                        lambda label, print_fn=print: barriers.append(label))

    removed = shared_pole_screening.retain_iteration_scratch(
        root, "sc_0002", print_fn=lambda *_: None)

    assert removed == ("sc_0000_shared_pole", "sc_0001_shared_pole",
                       "sc_0003_shared_pole", "sc_0007_shared_pole")
    assert (root / "sc_0002_shared_pole" / "bank.h5").exists()
    assert all(path.exists() for path in unrelated)
    assert (outside / "keep.h5").exists(), "a symlinked generation is unlinked, never followed"
    assert barriers == ["shared_pole.scratch.retain.sc_0002"]

