"""Restore verified model assets from a local submission directory (no downloads)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-package", type=Path, required=True)
    parser.add_argument("--include-demo", action="store_true")
    args = parser.parse_args()
    source = args.from_package.resolve()
    if not (source / "models").is_dir() and (source / "final/models").is_dir():
        source = source / "final"
    manifest = json.loads((ROOT / "models/model_manifest.json").read_text("utf-8"))
    pending: list[tuple[Path, Path]] = []
    for relative, expected in manifest["files"].items():
        original = source / relative
        destination = ROOT / relative
        if not original.is_file():
            raise FileNotFoundError(f"Missing source asset: {original}")
        if sha256(original) != expected["sha256"]:
            raise ValueError(f"SHA256 mismatch: {original}")
        if destination.exists():
            if sha256(destination) != expected["sha256"]:
                raise FileExistsError(f"Refusing to overwrite a different model: {destination}")
        else:
            pending.append((original, destination))
    if args.include_demo:
        labels = json.loads((ROOT / "demo/labels.json").read_text("utf-8"))
        for filename in labels:
            if Path(filename).name != filename or "\\" in filename:
                raise ValueError(f"Unsafe video name: {filename}")
            original = source / "demo/source" / filename
            destination = ROOT / "demo/source" / filename
            if not original.is_file():
                raise FileNotFoundError(original)
            if destination.exists():
                if sha256(destination) != sha256(original):
                    raise FileExistsError(f"Refusing to overwrite a different clip: {destination}")
            else:
                pending.append((original, destination))
    # Validate every input before copying any file.
    for original, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        print(f"Restored {destination.relative_to(ROOT).as_posix()}")
    print(f"Validated {len(manifest['files'])} model hashes; copied {len(pending)} assets.")


if __name__ == "__main__":
    main()
