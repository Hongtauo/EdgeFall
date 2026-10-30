# EdgeFall-ST 模型加载与评测接口说明

本文面向使用自有数据集的评测人员，说明模型文件、加载方法、输入输出契约，以及未提供
评测协议时建议采用的默认口径。环境安装与演示命令见[部署说明](deployment.md)。

## 1. 模型怎么组成

交付模型为 **EdgeFall-ST HN3-B-R Seed-1 Epoch 8**，是一个面向多人视频的因果时序
跌倒检测模型。它不是单独的分类网络，而是由人体检测与跟踪、几何时序、全局视觉、
人物 ROI 视觉和跨模态融合共同组成。

```text
视频帧
  │
  ├─ YOLO26s 人体检测 + ByteTrack 多人跟踪
  │        │
  │        ├─ 人体框序列 ───────────────→ 冻结 V3.3-G 几何时序分支 ─┐
  │        ├─ YOLO P4/P5 全局特征 ─────→ 全局视觉时序分支 ──────────┤
  │        └─ YOLO P3/P4 人物 ROI 特征 → ROI 视觉时序分支 ──────────┤
  │                                                               │
  └──────────────── 几何条件时序交叉注意力融合 ←───────────────────┘
                                  │
                         每个人物的跌倒状态分数
                                  │ 人物间最大值
                             时刻级 score
                                  │ 视频内最大值
                           视频级 video_score
                                  │ 阈值 0.7094
                           is_fall：0 或 1
```

### 1.1 人体检测与多人跟踪

- `yolo26s.pt` 检测类别为 person，并同时提供 P3/P4/P5 中间视觉特征；
- ByteTrack 在连续帧之间维护人物轨迹 ID；
- 默认最多对检测置信度、人物面积和画面位置综合优先级最高的 2 条人物轨迹分别推理；
- 每个人物独立生成风险分数，再使用最大值聚合为画面级风险。

### 1.2 三个时序分支

| 分支 | 输入 | 作用 |
|---|---|---|
| 几何时序分支 | 人体框的中心、宽高、面积、宽高比、底边位置及运动变化 | 捕捉失衡、快速下落和倒地形态 |
| 全局视觉分支 | YOLO P4/P5 全局特征，拼接后为 768 维 | 利用地面、家具和整体场景上下文 |
| 人物 ROI 分支 | YOLO P3/P4 特征上的人物区域，输出 `384×3×3` | 识别人姿、局部外观和倒地状态 |

模型按 10 FPS 维护历史。几何分支使用 15 个历史偏移，覆盖当前时刻至约 3.2 秒前；
视觉分支使用其中 8 个稀疏时刻，降低计算量。原始画面内部 letterbox 到 `544×960`
提取视觉特征，人体框同步映射到几何分支的 `288×512` 坐标系。

### 1.3 跨模态融合与分类

- 冻结的 V3.3-G 几何分支首先形成几何事件状态；
- 几何时序作为查询，通过 4 头时序交叉注意力检索全局和 ROI 视觉证据；
- 融合隐层维度为 128，前馈维度为 256；
- 事件头输出 `falling`（正在跌倒）与 `fallen`（已经倒地）状态，转换为 `[0,1]` 风险分数；
- 人物间采用 max-MIL 聚合，视频内采用全部采样时刻的最大值聚合。

模型总参数量为 `12,212,714`：YOLO 检测与特征网络 `10,009,784`，融合确认器（含冻结
几何分支）`2,202,930`。模型只使用视觉信息，不依赖音频、深度图或穿戴设备。

## 2. 输入是什么

评测方可以使用以下两种输入形式：

### 2.1 视频文件或视频目录

- 单个 `.mp4`、`.avi`、`.mov`、`.mkv`、`.m4v` 视频；
- 包含上述视频的目录，程序默认递归扫描；
- 视频内容为 RGB 画面，不需要人体框、姿态、光流或预提取特征；
- 不需要把真实标签传给模型，标签仅用于评测方在推理完成后计算指标；
- 原始分辨率和源视频帧率可以不同，程序会保持宽高比并自动采样到 10 FPS。

命令行示例：

```powershell
python infer.py --input <单个视频或视频目录> --device cuda
```

### 2.2 在线图像帧

直接调用 Python 接口时，每次输入一帧：

```text
类型：NumPy ndarray
形状：[H, W, 3]
颜色：OpenCV BGR 顺序
数据类型：uint8
数值范围：0–255
时间顺序：连续、不可打乱
输入频率：10 FPS
```

模型内部会完成人体检测、ByteTrack 跟踪、全局/人物 ROI 特征提取和时序跌倒判断。
调用方不应直接向融合模型传入单张 Tensor，也不需要自行裁剪人物区域。

## 3. 输出是什么

模型核心输出是取值 `[0, 1]` 的跌倒风险分数：

| 输出层级 | 核心字段 | 含义 |
|---|---|---|
| 视频级 | `video_score` | 视频内全部 10 FPS 采样时刻跌倒分数的最大值 |
| 视频级 | `is_fall` | `video_score >= 0.7094` 为 `1`，否则为 `0` |
| 视频级 | `first_alarm_s` | 第一次达到报警阈值的视频时间，未报警时为空 |
| 时刻级 | `score` | 当前采样时刻的融合跌倒概率 |
| 时刻级 | `geometry_score` | 当前时刻的几何时序分支概率 |
| 人物级 | `people` | 每个人的轨迹 ID、人物框及跌倒状态分数 |
| 性能 | `algorithm_ms` | 当前采样时刻的模型算法总耗时 |

命令行推理会生成三个文件：

```text
results/<时间戳>/
├── predictions.csv   # 每个视频一行：最终分数与是否跌倒
├── frames.csv        # 每个采样时刻一行：逐时刻、逐人物结果
└── summary.json      # 模型、设备、权重校验、参数量和时延摘要
```

最简二分类结果可以只读取 `predictions.csv` 中的以下三列：

```text
path,video_score,is_fall
```

其中 `path` 用于关联评测方的真实标签，`video_score` 用于计算 ROC-AUC、PR-AUC 等
阈值无关指标，`is_fall` 用于计算准确率、精确率、召回率和 F1。

## 4. 最简评测入口

模型不要求特定数据集目录结构，也不从文件名推断标签。将一个视频或视频目录传给
`infer.py` 即可：

```powershell
cd <解压目录>\final
Application\.venv-live\Scripts\Activate.ps1

python -u infer.py `
  --input <评测视频目录> `
  --output-dir results\judge_eval `
  --device cuda
```

输入目录默认递归扫描。支持 `.mp4`、`.avi`、`.mov`、`.mkv`、`.m4v`，结果写入：

- `predictions.csv`：视频级分数和二分类结果；
- `frames.csv`：10 FPS 采样时刻级、人物级分数和时延；
- `summary.json`：模型、权重、设备、参数量和整体时延摘要。

`--output-dir` 必须是尚不存在的目录。评测标签由评测方根据 `path` 字段与其标注表关联，
推理过程不读取真实标签。

## 5. 模型文件与加载关系

以下三个文件必须共同保留在 `models/`：

| 文件 | 用途 |
|---|---|
| `yolo26s.pt` | 人体检测、特征提取与 ByteTrack 输入 |
| `geometry_v33g_inference.pt` | 冻结的几何时序分支 |
| `hn3br_epoch008_inference.pt` | HN3-B-R 融合确认模型与特征契约 |

推荐从 `hn3br_epoch008_inference.pt` 创建 `OnlineFygrRuntime`。运行时会读取融合权重中的
特征契约，自动定位同目录的几何权重，并校验 YOLO 权重 SHA256；权重组合不匹配时会
直接报错，不能任意替换其中一个文件。

### 5.1 整视频 Python 接口

以下代码与 `infer.py` 使用同一条无缓存在线推理链路：

```python
from pathlib import Path

import torch

from edgefall_st.runtime.fygr_online import OnlineFygrRuntime, run_video

# 从 final 目录执行接入脚本。
root = Path.cwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

runtime = OnlineFygrRuntime(
    checkpoint_path=root / "models" / "hn3br_epoch008_inference.pt",
    weights_path=root / "models" / "yolo26s.pt",
    device=device,
    amp=device.type == "cuda",
    confidence=0.10,
    iou=0.30,
    tracker="bytetrack.yaml",
    quantize=16,
    max_actor_missed=2,
    max_people=2,
)

try:
    result, frame_predictions = run_video(
        runtime,
        Path("sample.mp4"),
        path="sample.mp4",
        label=None,
        alarm_threshold=0.7094,
        warmup_samples=10,
    )
finally:
    runtime.close()

print(result.video_score)
print(int(result.video_score >= 0.7094))
```

`run_video` 会读取源视频帧率，确定性采样到 10 FPS，在每个新视频开始前重置跟踪器和
时序状态，并以全部采样时刻分数的最大值作为 `video_score`。

### 5.2 在线逐帧接口

摄像头、RTSP 或其他解码器可直接调用 `process_frame`：

```python
runtime.reset_stream()

# frame_bgr 是 OpenCV 格式的 uint8 三通道图像。
# 调用方需按 10 FPS 提供连续、按时间排序的采样帧。
for sample_index, frame_bgr in enumerate(sampled_frames):
    prediction = runtime.process_frame(
        frame_bgr,
        sample_index=sample_index,
    )
    is_alarm = prediction.score >= 0.7094
```

接入逐帧接口时必须遵守：

1. `sample_index` 从 `0` 开始连续递增；
2. 输入顺序必须与时间顺序一致，不能打乱帧；
3. 每路独立视频流使用独立运行时，或在切换视频/摄像头时调用 `reset_stream()`；
4. 输入帧应预先按 10 FPS 采样，不要把 25/30 FPS 的每一帧都直接传入；
5. 推理结束调用 `close()` 释放检测器钩子和相关资源。

## 6. 详细输入契约

| 项目 | 要求 |
|---|---|
| 输入内容 | 仅视频画面，不依赖音频、深度图或穿戴设备 |
| 文件输入 | `.mp4`、`.avi`、`.mov`、`.mkv`、`.m4v` |
| 帧输入 | NumPy 数组，形状 `[H, W, 3]`，OpenCV BGR 顺序 |
| 数据类型 | 推荐 `uint8`，数值范围 `0–255` |
| 分辨率 | 原始分辨率不限，运行时按权重契约自动 letterbox |
| 有效采样率 | 固定 10 FPS |
| 时间方向 | 因果推理，只使用当前及历史帧 |
| 场景边界 | 每个独立视频开始前必须重置时序状态 |
| 多人 | 默认最多对 2 条有效人物轨迹分别推理 |

RGB 视频由 OpenCV 解码后直接输入。单通道红外或灰度数据应先复制/转换为三通道 BGR；
虽然接口可以接收三通道红外画面，但跨成像域精度需要由评测方单独验证。

## 7. 详细输出契约

所有风险分数均在 `[0, 1]`。分数越大，模型判定为跌倒事件的置信程度越高。

### 7.1 视频级输出 `predictions.csv`

| 字段 | 类型 | 含义 |
|---|---|---|
| `path` | string | 输入视频绝对路径，可用于关联标签 |
| `label` | int/空 | `infer.py` 不读取标签，因此默认留空 |
| `source_fps` | float | 源视频帧率 |
| `source_frames` | int | 源视频总帧数 |
| `sampled_frames` | int | 实际处理的 10 FPS 采样帧数 |
| `video_score` | float | 全部采样时刻 `score` 的最大值 |
| `first_alarm_s` | float/空 | 首次达到阈值的时间；未报警时为空 |
| `decode_ms` | float | 视频解码累计耗时 |
| `wall_s` | float | 整个视频的墙钟运行时间 |
| `is_fall` | int | 达到 `--threshold` 时为 `1`，否则为 `0`；默认阈值为 `0.7094` |

### 7.2 时刻级输出 `frames.csv`

| 字段 | 含义 |
|---|---|
| `path` | 所属视频路径 |
| `sample_index` | 10 FPS 采样序号，从 0 开始 |
| `timestamp_s` | 采样时间，等于 `sample_index / 10` |
| `score` | 当前时刻的融合跌倒分数 |
| `geometry_score` | 几何时序分支分数 |
| `global_attention` | 全局场景特征的注意力权重占比；不可用时为空 |
| `roi_attention` | 人物 ROI 特征的注意力权重占比；不可用时为空 |
| `actor_track_id` | 产生画面级最大风险的人物轨迹 ID |
| `actor_box` | 主人物框 `[x1, y1, x2, y2]`，为源图像像素坐标 |
| `people` | JSON 数组，包含每个人物的 ID、框和各类分数 |
| `detector_ms` | 人体检测耗时 |
| `tracking_ms` | 跟踪与人物选择耗时 |
| `feature_pool_ms` | 全局/ROI 特征池化耗时 |
| `confirmer_ms` | 时序确认模型耗时 |
| `algorithm_ms` | 上述模型算法端到端耗时 |
| `timed` | 是否计入汇总时延；预热样本为 `false` |

`people` 中每个人物包含：

- `track_id`：ByteTrack 轨迹 ID；
- `bbox`：源图像像素坐标 `[x1, y1, x2, y2]`；
- `score`：人物级融合跌倒分数；
- `falling_score`：正在跌倒状态分数；
- `fallen_score`：已经倒地状态分数；
- `geometry_score`：人物级几何分支分数。

### 7.3 运行摘要 `summary.json`

摘要记录输入数量、阳性预测数、固定阈值、模型和 YOLO 权重路径及 SHA256、设备信息、
参数量、算法平均/P95 时延、算法 FPS 与 CUDA 峰值显存，可用于评测结果追溯。

## 8. 未提供评测协议时的建议口径

以下是保证结果可复现的建议，并非评测方预先规定的官方协议。如评测方之后提供正式协议，
应以正式协议为准，并在报告中说明差异。

模型推理与指标计算是分开的：

```text
私有测试视频 ──→ infer.py ──→ predictions.csv（path、video_score、is_fall）
                                      │
私有真实标签表 ───────────────────────┘ 按视频路径或唯一 ID 关联
                                      │
                                      └─→ 计算分类与排序指标
```

建议真实标签表至少包含两列：

```text
path,label
dataset/fall/fall_001.mp4,1
dataset/adl/walk_001.mp4,0
```

其中 `1` 表示跌倒，`0` 表示非跌倒。评测脚本按相对路径或稳定的视频 ID 将标签表与
`predictions.csv` 一一关联；不要只按文件名关联，以免不同子目录存在同名视频。

1. **样本粒度**：一个视频文件作为一个独立样本，每个视频开始前清空历史状态。
2. **标签定义**：包含真实人体意外跌倒过程或倒地状态的视频记为 `1`；日常活动记为 `0`。
3. **固定推理设置**：10 FPS、`confidence=0.10`、`iou=0.30`、`max_people=2`。
4. **视频聚合**：`video_score = max(frame score)`，不增加额外平滑、投票或后处理。
5. **分类阈值**：固定使用交付阈值 `0.7094`，即 `video_score >= 0.7094` 判为跌倒。
6. **禁止测试集调参**：不得根据评测集标签重新选择阈值、采样率或后处理参数。
7. **建议指标**：至少报告视频级 TP、FP、TN、FN、准确率、精确率、召回率/敏感度、
   F1 和特异度；类别明显不平衡时同时报告 PR-AUC 与 ROC-AUC。
8. **时延统计**：第一条视频可将前 10 个采样时刻标为预热，不计入平均/P95 时延，
   但这些时刻仍必须参与报警分数计算。

若标注是事件区间而不是视频标签，建议额外报告首次报警相对事件起点的时间差和每小时误报
次数。事件容忍窗口、重复报警合并规则必须由评测方明确记录，不能在看到测试结果后选择。

## 9. 接入注意事项

- 不要直接对 `hn3br_epoch008_inference.pt` 调用单帧 `forward`；它依赖 YOLO 特征、人物框、
  多尺度历史序列和冻结几何分支，应通过 `OnlineFygrRuntime` 加载。
- 不要在连续处理两个视频时沿用上一视频的跟踪和历史状态。
- 不要把源视频 FPS 当作模型有效 FPS；高层 `run_video` 会自动采样，低层接口需自行采样。
- 不要用 `geometry_score` 代替最终 `score`，分类应使用融合分数。
- 不要把人物检测置信度 `confidence` 当作跌倒报警阈值。
- `video_score` 是视频最大风险，不是所有帧分数的平均值。
- CPU 与 GPU 应产生相同语义的输出，但浮点精度和时延可能存在轻微差异。

在评测脚本无法直接调用命令行时，优先复用第 5.1 节的 `run_video` 接口，避免自行重写
采样、跟踪、时序重置和视频级聚合逻辑。
