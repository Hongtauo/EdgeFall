"""Validate and launch the manifest-backed HN3-B-R training recipe.

Dry-run uses only the Python standard library and never starts training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CACHE_KEYS = (
    "train_cache", "val_cache", "extra_train_cache", "extra_val_cache",
    "train_overlay_cache",
)
FILE_KEYS = (
    "train_candidates", "hard_negative_manifest", "hard_positive_manifest",
    "split_manifest", "geometry_checkpoint", "resume",
)


def cli_arguments(arguments: dict[str, Any]) -> list[str]:
    """Map configuration destinations to train_fygr.build_parser options."""
    result: list[str] = []
    for key, value in arguments.items():
        option = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            result.append(option if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            # All list fields in this recipe are argparse action='append'.
            for item in value:
                if isinstance(item, (dict, list, bool)) or item is None:
                    raise ValueError(f"Invalid repeated option {key}: {item!r}")
                result.extend((option, str(item)))
        elif isinstance(value, (str, int, float)):
            result.extend((option, str(value)))
        else:
            raise ValueError(f"Invalid argument value for {key}: {value!r}")
    return result


def resolve_input(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def missing_inputs(arguments: dict[str, Any], root: Path) -> list[str]:
    """Check all input files plus the index and manifest of each cache."""
    missing: list[str] = []
    for key in CACHE_KEYS:
        values = arguments.get(key, [])
        values = values if isinstance(values, list) else [values]
        for value in values:
            if not isinstance(value, str) or not value:
                missing.append(f"{key}: expected a nonempty cache path")
                continue
            cache = resolve_input(value, root)
            if not cache.is_dir():
                missing.append(f"{key}: {cache}")
                continue
            for filename in ("manifest.json", "index.jsonl"):
                if not (cache / filename).is_file():
                    missing.append(f"{key}: {cache / filename}")
    for key in FILE_KEYS:
        value = arguments.get(key)
        if value is not None and not resolve_input(str(value), root).is_file():
            missing.append(f"{key}: {resolve_input(str(value), root)}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hn3br_train.json")
    parser.add_argument("--dry-run", action="store_true", help="Print command and missing inputs; do not train")
    parser.add_argument("--validate-parser", action="store_true", help="Also import the actual training parser; needs model dependencies")
    options = parser.parse_args()
    config = json.loads(options.config.read_text(encoding="utf-8"))
    arguments = config.get("arguments")
    if not isinstance(arguments, dict) or not arguments:
        parser.error("config must contain a nonempty 'arguments' object")
    required = ("train_cache", "val_cache", "geometry_checkpoint", "split_manifest")
    for key in required:
        if not isinstance(arguments.get(key), str) or not arguments[key]:
            parser.error(f"required configuration argument: {key}")
    command_arguments = cli_arguments(arguments)
    missing = missing_inputs(arguments, ROOT)
    print(json.dumps({
        "mode": "dry-run" if options.dry_run else "train",
        "root": str(ROOT),
        "config": str(options.config.resolve()),
        "source_run_id": config.get("source_run_id"),
        "command": shlex.join([sys.executable, "-m", "edgefall_st.training.train_fygr", *command_arguments]),
        "missing_inputs": missing,
        "ready": not missing,
    }, ensure_ascii=False, indent=2))
    if options.dry_run and not options.validate_parser:
        return 0
    if not options.dry_run and missing:
        print("Training not started: prepare all missing inputs listed above.", file=sys.stderr)
        return 2
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from edgefall_st.training.train_fygr import build_parser, run_training

    parsed = build_parser().parse_args(command_arguments)
    if options.dry_run:
        print("train_fygr.build_parser: accepted all configured options; training not started.")
        return 0
    os.chdir(ROOT)
    run_training(parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
