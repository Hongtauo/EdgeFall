"""Check portable source integrity, published metrics and documentation links."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

from package_release import ASSET_SUFFIXES, ROOT, included


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", action="store_true", help="Also require an approved root LICENSE")
    args = parser.parse_args()
    problems: list[str] = []
    count = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or not included(path, False):
            continue
        count += 1
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8-sig"))
            elif path.suffix == ".svg":
                ET.parse(path)
        except (SyntaxError, ValueError, ET.ParseError) as exc:
            problems.append(f"{path.relative_to(ROOT)}: {exc}")
    # Legacy experiment records retain their original relative artifact references.
    documents = [ROOT / "README.md", ROOT / "README.en.md", ROOT / "CONTRIBUTING.md"]
    documents.extend((ROOT / "docs").glob("*.md"))
    documents.extend([ROOT / "models/README.md", ROOT / "demo/source/README.md"])
    for path in documents:
        if not path.is_file():
            problems.append(f"Missing document: {path.relative_to(ROOT)}")
            continue
        for destination in re.findall(r"\]\(([^)]+)\)", path.read_text("utf-8")):
            destination = unquote(destination.strip("<> ").split("#")[0])
            if not destination or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", destination):
                continue
            if not (path.parent / destination).exists():
                problems.append(f"Broken link in {path.relative_to(ROOT)}: {destination}")
    summary = json.loads((ROOT / "reports/benchmark_summary.json").read_text("utf-8"))
    for metric in ("ap", "p90", "p95", "mean_p"):
        value = sum(d[metric] for d in summary["datasets"]) / len(summary["datasets"])
        if abs(value - summary["macro"][metric]) > 1e-12:
            problems.append(f"Inconsistent macro {metric}")
    if summary["parameters"]["total"] != 12_212_714:
        problems.append("Unexpected final model parameter count")
    labels = json.loads((ROOT / "demo/labels.json").read_text("utf-8"))
    if len(labels) != 18:
        problems.append("Demo playlist should contain 18 labels")
    if args.public and not (ROOT / "LICENSE").is_file():
        problems.append("Public source release requires a root LICENSE file")
    if (ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "-z"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout.decode("utf-8").split("\0")
        for filename in filter(None, tracked):
            if Path(filename).suffix.lower() in ASSET_SUFFIXES:
                problems.append(f"Binary asset staged in Git: {filename}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"Release integrity passed: {count} source files; metric arithmetic and document links verified.")
    if not args.public:
        print("This check does not approve licensing or public asset redistribution.")


if __name__ == "__main__":
    main()
