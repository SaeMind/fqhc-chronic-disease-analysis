"""Acquire the exact ledger snapshots. Reject changed downloads."""

import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def matches(path, source):
    if not path.is_file() or path.stat().st_size != source["bytes"]:
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == source["sha256"]


def main():
    ledger = json.loads((ROOT / "evidence" / "sources.json").read_text())
    directory = ROOT / "data" / "public"
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in ledger.items():
        target = directory / name
        if matches(target, source):
            print(f"Verified {name}")
            continue
        with TemporaryDirectory(dir=directory) as temporary:
            pending = Path(temporary) / name
            with urlopen(source["url"], timeout=90) as response, pending.open("wb") as file:
                shutil.copyfileobj(response, file)
            if not matches(pending, source):
                raise SystemExit(f"G1 · {name} differs from the pinned snapshot.")
            pending.replace(target)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
