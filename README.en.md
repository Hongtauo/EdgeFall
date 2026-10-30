# EdgeFall

**Understanding falls as a process, with ordinary cameras.**

[中文首页](README.md) · [Method](docs/method.md) · [Training](docs/training.md) · [Benchmarks](docs/benchmarks.md)

![Architecture overview](docs/assets/overview.svg)

EdgeFall combines tracked bounding-box geometry, person ROI features and global scene features for causal video fall detection.
A shared YOLO26s front end feeds a lightweight geometry-guided visual confirmer. The repository includes training, evaluation, streaming inference and a Vue / FastAPI demonstration application.

The locked model is **HN3-B-R Seed-1 Epoch 8**, with **12,212,714 parameters** and a 3.2-second causal history at 10 sampled FPS.
Three-domain cached-feature tests report macro AP **0.9226** and macro Mean-P **0.8262**.
An independent online UR Fall protocol detects **30/30 falls**, with **19/40 ADL false positives**; mean algorithm latency is **44.34 ms** on an RTX 4070 Ti SUPER, excluding decoding, with one scored person.
The default runtime supports two scored people. These latency protocols are distinct.

```bash
conda create -n EdgeFall-ST python=3.11 -y
conda activate EdgeFall-ST
python -m pip install -e ".[inference,demo,dev]"
# Restore separately managed weights before running inference.
python tools/verify_install.py --device auto
python infer.py --input /path/to/video.mp4
```

Third-party videos and model binaries are excluded from Git. See [model assets](models/README.md), [datasets](docs/datasets.md) and [deployment](docs/deployment.md).
No public weight Release is currently configured. Training requires independently obtained datasets and regenerated feature caches, not just the inference weights.

Source licensing is pending the owner's decision. Existing third-party licensing continues to apply.
NPU deployment, ONNX / TensorRT equivalence and field reliability are future validation milestones, not completed capabilities.

See the [Chinese project documentation](README.md) for detailed methods, reproducibility, evidence, product demonstration and milestones.
