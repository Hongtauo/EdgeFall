# 环境、部署与演示

以下命令从仓库根目录执行，推荐 Python 3.11，现有 Conda 环境名为 `EdgeFall-ST`。

## 本地模型推理

```bash
conda create -n EdgeFall-ST python=3.11 -y
conda activate EdgeFall-ST
# 先按显卡驱动安装匹配的 PyTorch/TorchVision，再安装本项目。
python -m pip install -e ".[inference,demo,dev]"
python scripts/import_assets.py --from-package /path/to/submission/final
python tools/verify_install.py --device auto
python infer.py --input /path/to/video.mp4 --device auto
```

`setup/requirements.txt` 是保留的通用依赖快照；其中 torch 2.6 / torchvision 0.21
与历史本机 torch 2.13+cu130 / torchvision 0.28 不同。为精确复查历史数值应记录实际环境。
机器无 CUDA 时可用 CPU 功能验证，不能因此声称满足实时目标。

NVIDIA V100 需要包含 Volta `sm_70` 的 wheel：

```bash
python -m pip install -r setup/requirements-v100-cu118.txt
python -m pip install -e ".[inference,demo,dev]" --no-deps
python -m pip install -r Application/backend/requirements-live.txt
python tools/verify_install.py --device cuda
```

V100 不支持原生 BF16，运行器自动选择 FP16。现有证据为代码兼容审计，本机测试 GPU
是 RTX 4070 Ti SUPER；没有 V100 实机时延报告。

## Python 与批量接口

```bash
python infer.py --input /path/to/videos --output-dir results/batch --device cuda
python infer.py --input /path/to/video.mp4 --device cpu --no-amp
python infer.py --help
```

输入支持 `.mp4`、`.avi`、`.mov`、`.mkv`、`.m4v`；目录默认递归搜索。
输出目录必须不存在。每条人物轨迹独立构造历史、输出风险分数，当前人物最大值形成画面风险，
视频所有采样时刻最大值形成视频级风险。默认最多确认 2 人；人数增加时应重新测量延时。
参数与逐帧 API 示例见[接口说明](api_zh.md)。

## Web App

```bash
python -m pip install -e ".[inference,demo]"
python scripts/import_assets.py --from-package /path/to/submission/final --include-demo
cd Application/frontend
npm ci
npm run build
cd ../backend
```

Linux / macOS 运行真实模型：

```bash
EDGEFALL_ENGINE=live python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows 在仓库根目录：

```powershell
$env:EDGEFALL_ENGINE = "live"
cd Application\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

也可运行 `Application/start_demo.ps1 -Mode live -PythonPath <已激活环境的python路径>`，
该脚本会安装缺失的前端依赖并构建前端。页面位于 `http://127.0.0.1:8000`。

| 模式 | 实际行为 | 资源依赖 |
|---|---|---|
| `live` | 对采样帧运行 YOLO、ByteTrack、HN3-B-R | 模型 + 视频 + 推理依赖 |
| `replay` | 按视频时间读取保存的预测 | 视频 + `demo/inference/frames.csv` |
| `auto` | 尝试 live，失败时显示原因并回退 replay | 与以上两种模式一致 |

正式演示推荐显式选择模式。回放不用于测量实时推理性能，界面已明确标识。
服务默认监听本地；公开服务还需鉴权、HTTPS、负载管理和运维监控。

## ONNX、TensorRT 和目标 NPU

当前仓库验证的是 PyTorch 在线流程。尚未提供已验收的端到端 ONNX / TensorRT engine
或海思 / 瑞芯微 NPU 产物，不能把“模型参数少”写成“已通过 NPU 验收”。

建议按以下边界实施并逐项比较输出：

1. 导出 YOLO 检测与 P3/P4/P5 特征输出，固定层与通道契约。
2. 导出以历史几何、视觉特征和有效掩码为输入的确认器。
3. 主机侧继续维护视频解码、采样、ByteTrack、ROI/池化与因果历史状态；逐项核对可下沉算子。
4. 对同一视频比较 PyTorch 与导出后逐帧、逐人物分数和报警时刻，再进行 FP16 / INT8 回归。
5. 在目标设备测量 1080P 解码到报警的端到端延迟、峰值内存、长时稳定性与多人吞吐。

单文件权重打包只改变交付形式，不自动完成跟踪、动态人数、历史缓存等逻辑的图导出。
这条路线是后续工程计划，未被包装为本次已完成能力。
