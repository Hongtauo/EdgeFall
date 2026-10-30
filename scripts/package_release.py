"""Create a source archive; explicitly opt in to independently licensed local assets."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    ".github", "Application", "annotations", "configs", "demo", "docs", "edgefall_st",
    "external_eval", "models", "reports", "scripts", "setup", "tests", "tools",
}
ROOT_FILES = {
    ".gitignore", ".gitattributes", "README.md", "README.en.md", "CONTRIBUTING.md",
    "LICENSE", "NOTICE", "CITATION.cff", "pyproject.toml", "infer.py", "run_demo.bat", "run_demo.sh",
}
EXCLUDED_PARTS = {"node_modules", ".venv", ".venv-live", ".runtime", "__pycache__", ".pytest_cache", ".git", "evidence_private"}
ASSET_SUFFIXES = {".pt", ".onnx", ".engine", ".safetensors", ".mp4", ".avi", ".mov", ".mkv"}
MAX_BYTES = 200_000_000


def included(path: Path, include_assets: bool) -> bool:
    relative = path.relative_to(ROOT)
    if path.is_symlink() or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.parts[0] not in TOP_LEVEL and relative.as_posix() not in ROOT_FILES:
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    if path.suffix.lower() in {".pyc", ".pyo", ".zip", ".npy", ".npz", ".pkl"}:
        return False
    return include_assets or path.suffix.lower() not in ASSET_SUFFIXES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    label = "local-assets" if args.include_local_assets else "source"
    output = (args.output or ROOT / "dist" / f"EdgeFall-{label}.zip").resolve()
    if output.exists():
        raise FileExistsError(f"Archive exists; choose a new --output: {output}")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p, args.include_local_assets))
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                archive.write(path, (Path("EdgeFall") / path.relative_to(ROOT)).as_posix())
        if output.stat().st_size >= MAX_BYTES:
            raise ValueError("Archive exceeds the 200,000,000-byte release limit")
        with zipfile.ZipFile(output) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt archive entry: {bad}")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"{output}\nfiles={len(files)} bytes={output.stat().st_size} sha256={digest}")
        if args.include_local_assets:
            print("Local assets retain their own terms. This archive is not cleared for public redistribution.")
    except BaseException:
        output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
