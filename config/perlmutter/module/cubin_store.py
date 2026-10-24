"""Keep native cache images after module acceptance in the existing read-only store.

Only asset copying happens here. The runtime's native reader continues to
check each image's source/toolchain/device key and payload hash on use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(store, revision, bundle_id):
    store = Path(store)
    record = json.loads((store / "manifest.json").read_text())
    if (record["source_revision"], record["bundle_id"]) != (revision, bundle_id):
        raise ValueError("cubin store is not from this accepted source/bundle")
    names = [item["name"] for item in record["images"]]
    if not names or sorted(names) != sorted(p.name for p in store.glob("*.cubin")):
        raise ValueError("cubin store image inventory differs from its receipt")
    for image in record["images"]:
        name = image["name"]
        if Path(name).name != name:
            raise ValueError("cubin store image must have a flat filename")
        path = store / name
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != image["bytes"]
                or digest(path) != image["sha256"]):
            raise ValueError(f"cubin store image differs: {name}")
    return record


def capture(receipt, source):
    """Copy regular, self-contained images after the P4 acceptance passes."""
    source = Path(source)
    record = json.loads(Path(receipt).read_text())
    if (source / "SOURCE_COMMIT").read_text().strip() != record["source_revision"]:
        raise ValueError("cubin cache receipt is not from this source")
    cache = Path(record["cache"])
    images = []
    temporary = Path(tempfile.mkdtemp(prefix=".cubin_store.", dir=source))
    try:
        for path in sorted(cache.glob("*.cubin")):
            target = temporary / path.name
            shutil.copyfile(path, target)  # follow a cache link; ship its bytes
            images.append(dict(name=path.name, bytes=target.stat().st_size,
                               sha256=digest(target)))
        record["images"] = images
        (temporary / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        verify(temporary, record["source_revision"], record["bundle_id"])
        destination = source / "cubin_store"
        if destination.exists():
            old = verify(destination, record["source_revision"], record["bundle_id"])
            if old != record:
                raise ValueError("candidate already has a different cubin store")
        else:
            temporary.rename(destination)
        return len(images)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def copy(store, destination, revision, bundle_id):
    """Verify and copy accepted images into the fresh publication archive."""
    record = verify(store, revision, bundle_id)
    destination = Path(destination)
    destination.mkdir()
    for image in record["images"]:
        shutil.copyfile(Path(store) / image["name"], destination / image["name"])
    shutil.copyfile(Path(store) / "manifest.json", destination / "manifest.json")
    verify(destination, revision, bundle_id)
    return len(record["images"])


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        count = capture(*sys.argv[2:])
    elif sys.argv[1] == "copy":
        count = copy(*sys.argv[2:])
    else:
        raise SystemExit("usage: cubin_store.py capture RECEIPT SOURCE | copy STORE DEST REV BUNDLE")
    print(f"[cubin-store] {count} images packaged")
