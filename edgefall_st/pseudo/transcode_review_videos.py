"""Transcode review videos to browser-compatible H.264 MP4 files."""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _dependency_path(package_root: Path) -> str:
    directories = sorted(
        path
        for path in package_root.glob("*/Library/bin")
        if path.is_dir()
    )
    return os.pathsep.join(str(path) for path in directories)


def _transcode(
    source: Path,
    *,
    output_root: Path,
    ffmpeg: Path,
    environment: dict[str, str],
    encoder: str,
    overwrite: bool,
) -> str:
    destination = output_root / source.name
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return "skipped"
    temporary = output_root / f"{source.stem}.tmp.mp4"
    temporary.unlink(missing_ok=True)
    encoder_arguments = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "25", "-b:v", "0"]
        if encoder == "h264_nvenc"
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "25"]
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        *encoder_arguments,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        creationflags=creation_flags,
    )
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        message = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Could not transcode {source}: {message}")
    if not temporary.exists() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg produced an empty file for {source}")
    temporary.replace(destination)
    return "converted"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "artifacts/reviews/omnifall_train_multi_person_yolo26s/videos"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/reviews/omnifall_train_multi_person_yolo26s/videos_h264"
        ),
    )
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--conda-package-root", type=Path)
    parser.add_argument(
        "--encoder",
        choices=("h264_nvenc", "libx264"),
        default="h264_nvenc",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, int]:
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    source_root = args.source_root.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root.resolve(strict=True)
    if source_root == output_root:
        raise ValueError("source-root and output-root must be different")

    environment = os.environ.copy()
    if args.conda_package_root is not None:
        package_root = args.conda_package_root.resolve(strict=True)
        dependencies = _dependency_path(package_root)
        if dependencies:
            environment["PATH"] = (
                dependencies + os.pathsep + environment.get("PATH", "")
            )

    sources = sorted(source_root.glob("*.mp4"))
    counts = {"total": len(sources), "converted": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _transcode,
                source,
                output_root=output_root,
                ffmpeg=ffmpeg,
                environment=environment,
                encoder=args.encoder,
                overwrite=args.overwrite,
            ): source
            for source in sources
        }
        completed = 0
        for future in as_completed(futures):
            status = future.result()
            counts[status] += 1
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"transcoded={completed}/{len(futures)} "
                    f"converted={counts['converted']} skipped={counts['skipped']}",
                    flush=True,
                )
    return counts


def main() -> None:
    counts = run(build_parser().parse_args())
    print(counts, flush=True)


if __name__ == "__main__":
    main()
