"""Benchmark EdgeFall-ST neural-network latency and CUDA memory."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from edgefall_st.models import EdgeFallST


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("frame", "stream"),
        default="stream",
    )
    return parser


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * quantile)))
    return ordered[index]


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    device = torch.device("cuda")
    model = EdgeFallST(input_size=(args.input_height, args.input_width))
    model = model.to(device).eval()
    image = torch.randn(
        (1, 3, args.input_height, args.input_width),
        device=device,
    )
    state = model.create_stream_state()

    def forward() -> None:
        if args.mode == "frame":
            model.forward_frame(image)
        else:
            model.forward_stream(image, state)

    with torch.inference_mode():
        for _ in range(args.warmup):
            forward()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        timings_ms: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "device": torch.cuda.get_device_name(0),
        "mode": args.mode,
        "input_size": [args.input_height, args.input_width],
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": model.parameter_count,
        "fp32_parameter_mb": round(model.parameter_count * 4 / 1024**2, 3),
        "mean_ms": round(statistics.fmean(timings_ms), 3),
        "median_ms": round(statistics.median(timings_ms), 3),
        "p95_ms": round(percentile(timings_ms, 0.95), 3),
        "minimum_ms": round(min(timings_ms), 3),
        "maximum_ms": round(max(timings_ms), 3),
        "peak_cuda_memory_mb": round(
            torch.cuda.max_memory_allocated() / 1024**2,
            3,
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
