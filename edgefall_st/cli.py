"""Inspect EdgeFall-ST structure and run a synthetic forward pass."""

from __future__ import annotations

import argparse
import json

import torch

from edgefall_st.models import EdgeFallST


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = EdgeFallST(input_size=(args.input_height, args.input_width)).to(device)
    model.eval()
    image = torch.zeros(
        (1, 3, args.input_height, args.input_width),
        device=device,
    )
    with torch.inference_mode():
        outputs = model(image)
    summary = {
        "device": device,
        "parameters": model.parameter_count,
        "parameters_million": round(model.parameter_count / 1_000_000, 4),
        "input_shape": list(image.shape),
        "output_shapes": {
            name: list(value.shape) for name, value in outputs.items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
