"""The release cubin store feeds the per-user cache (``ffi.fft.seed_cubin_cache``); CPU, no devices.

A store image the cache lacks is linked in; a cache file already present is kept; a
dangling link (a retired release's store) is replaced; a non-image file is ignored; a
missing store links nothing.  A torn or foreign image is the native reader's case
(``common/nvrtc_build.h``: key and hash checked, rebuilt into the cache).
"""
import os


def test_seed_links_missing_keeps_present_replaces_dangling(tmp_path):
    from ffi.fft import seed_cubin_cache
    store, cache = tmp_path / "store", tmp_path / "cache"
    store.mkdir()
    for name in ("kconv_m3_2x1x1_ns1_sm80_a.cubin", "kconv_m7_8x8x8_ns2_sm80_b.cubin",
                 "plan_pair_c.cubin"):
        (store / name).write_bytes(b"store")
    (store / "README").write_text("not an image")
    cache.mkdir()
    (cache / "kconv_m7_8x8x8_ns2_sm80_b.cubin").write_bytes(b"user")
    os.symlink(tmp_path / "retired" / "plan_pair_c.cubin", cache / "plan_pair_c.cubin")

    assert seed_cubin_cache(str(store), str(cache)) == 2
    assert (cache / "kconv_m3_2x1x1_ns1_sm80_a.cubin").read_bytes() == b"store"
    assert (cache / "plan_pair_c.cubin").read_bytes() == b"store"
    assert (cache / "kconv_m7_8x8x8_ns2_sm80_b.cubin").read_bytes() == b"user"
    assert sorted(os.listdir(cache)) == sorted(n for n in os.listdir(store) if n.endswith(".cubin"))


def test_seed_without_store(tmp_path):
    from ffi.fft import seed_cubin_cache
    assert seed_cubin_cache(str(tmp_path / "none"), str(tmp_path / "cache")) == 0
