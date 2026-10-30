"""Verify standalone model files, dependencies, and model construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import lap
import numpy as np
import torch
import torchvision
import ultralytics
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edgefall_st.runtime.fygr_online import OnlineFygrRuntime


EXPECTED_HASHES = {
    "models/yolo26s.pt": (
        "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
    ),
    "models/geometry_v33g_inference.pt": (
        "cfe72559c61b5808fbc0e6634b35e4ff9ee34deec8f6ddcf590dfbf84e8a6242"
    ),
    "models/hn3br_epoch008_inference.pt": (
        "85558b2384e4f363953188f560a2f6838b5b2b5243541e0043dae0f07ffce54e"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    hashes = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {relative}: {actual}")
        hashes[relative] = actual
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    gpu = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        architecture_list = torch.cuda.get_arch_list()
        if capability == (7, 0) and not any(
            architecture in {"sm_70", "compute_70"}
            for architecture in architecture_list
        ):
            raise RuntimeError(
                "This PyTorch CUDA wheel does not contain Volta sm_70 "
                "kernels. Install setup/requirements-v100-cu118.txt."
            )
        gpu = {
            "name": properties.name,
            "compute_capability": list(capability),
            "total_memory_gb": round(properties.total_memory / (1024**3), 2),
            "torch_cuda_runtime": torch.version.cuda,
            "torch_cuda_arch_list": architecture_list,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    runtime = OnlineFygrRuntime(
        checkpoint_path=ROOT / "models" / "hn3br_epoch008_inference.pt",
        weights_path=ROOT / "models" / "yolo26s.pt",
        device=device,
        amp=device.type == "cuda",
        max_people=2,
    )
    try:
        if gpu is not None:
            gpu["selected_amp_dtype"] = str(runtime.amp_dtype).removeprefix("torch.")
        report = {
            "status": "ok",
            "source_package": str(Path(__file__).resolve()),
            "device": str(device),
            "gpu": gpu,
            "versions": {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "PyYAML": yaml.__version__,
                "ultralytics": ultralytics.__version__,
                "lap": getattr(lap, "__version__", "unknown"),
            },
            "hashes": hashes,
            "parameters": {
                "detector": runtime.detector_parameter_count,
                "confirmer": runtime.confirmer_parameter_count,
                "total": (
                    runtime.detector_parameter_count
                    + runtime.confirmer_parameter_count
                ),
            },
        }
    finally:
        runtime.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
