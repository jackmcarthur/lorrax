"""SC shared-pole scratch keeps one map generation (MPA's retention rule).

Each SC map screens into ``tmp/mpa/sc_NNNN_shared_pole/``; a file-tier bank
there is 0.87 TB on Fe 8^3, so without retention the scratch grows linearly in
maps.  Only exact managed generation names may go; the current map's stays.
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
                 root / "mpa_fit_sc_0000.h5"]
    unrelated[0].mkdir()
    unrelated[1].touch()
    unrelated[2].touch()
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


def test_the_run_wide_photon_reference_pins_its_generation(tmp_path, monkeypatch):
    """Map 0 holds the photon static reference every later map freezes."""
    root = tmp_path / "mpa"
    for label in ("sc_0000", "sc_0001", "sc_0002"):
        (root / f"{label}_shared_pole").mkdir(parents=True)
    reference = root / "sc_0000_shared_pole" / "photon_static_reference.h5"
    reference.touch()
    monkeypatch.setattr("common.collectives.process_rank", lambda: 0)
    monkeypatch.setattr("common.collectives.barrier", lambda *a, **k: None)

    removed = shared_pole_screening.retain_iteration_scratch(
        root, "sc_0002", pinned=(str(reference), str(tmp_path / "not_managed.h5")),
        print_fn=lambda *_: None)

    assert removed == ("sc_0001_shared_pole",)
    assert reference.exists()
