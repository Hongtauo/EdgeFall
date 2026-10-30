# EdgeFall-ST 模型结构说明

本文档描述 `edgefall_st/models` 下的模型结构设计。项目当前保留两条可对照的训练路线：

1. `unified`：一次前向联合完成行人检测、跌倒阶段、起跌时刻、跨帧关联和动作识别。
2. `bbox-classifier`：直接读取预处理阶段选出的主动作人物 bbox，只训练跌倒事件语义，不训练人体定位。

`bbox-classifier` 是语义上限基线。它用于先回答“在人物位置已知时，模型能否区分跌倒与躺下、坐下等动作”，不能单独作为最终无外部 bbox 输入的部署方案。

## 1. 目录与文件总览

| 文件               | 职责                             | 主要类                                                 |
| ------------------ | -------------------------------- | ------------------------------------------------------ |
| `edgefall_st.py` | 顶层网络组装与流式状态           | `EdgeFallST`、`StreamState`                        |
| `bbox_classifier.py` | 基于已有主人物框的时序语义基线 | `BBoxFallClassifier` |
| `backbone.py`    | 二维空间特征提取（多尺度金字塔） | `LiteBackbone`                                       |
| `cmtf.py`        | 多尺度因果时序融合               | `CausalTemporalFusion`、`MultiScaleTemporalFusion` |
| `neck.py`        | 轻量 PAN-FPN 跨尺度融合          | `LitePANFPN`                                         |
| `head.py`        | 统一无锚框多任务检测头           | `UnifiedAnchorFreeHead`                              |
| `layers.py`      | NPU 友好基础算子                 | `ConvBNAct`、`InvertedResidual`、`FusionBlock`   |

## BBox 语义基线

`BBoxFallClassifier` 对每个稀疏历史帧执行：

```text
已有主人物 bbox
  -> 保持长宽比并扩展上下文
  -> ROI crop
  -> LiteBackbone
  -> RGB 特征 + bbox 几何轨迹 + 显式速度/加速度
  -> 短期与长期双分支因果 TCN
  -> fall / stage / onset
```

其中 `fall` 是直接优化的事件二分类主任务，目标由同目录 `events.csv` 的
`[start_s, end_s)` 区间决定；`stage` 仍学习当前姿态的
`normal/falling/fallen`，两者不再把无可观察跌倒过程的静态 `fallen` 混为一谈。
`stage` 和 `onset` 是辅助任务。16 类 `action`
以低权重参与训练，用于区分 `lie_down/lying/sit_down/stand_up` 等困难负动作。
`fall` 与 `stage` 目标一致的样本上默认增加概率一致性损失；fallen-only
负事件样本不计算该项，避免相互冲突。
该模式没有 box、objectness、NMS、目标分配和跟踪损失。

默认 `dual-tcn` 先按照有效采样间隔把视觉变化和 bbox 轨迹换算为速度，
再按“最早历史到当前”执行 dilation 为 1/2/4 的深度可分离因果
Temporal Conv1D。短期分支只读取 offset 不超过 4 的历史并服务于 onset；
长期分支读取完整历史并与短期状态一起服务于 fall/stage。`attention`
选项保留用于复现旧检查点，新结构需要重新训练。

`v32` 是独立的 V3.2 识别增强路径：

- P3/P4/P5 分别保留 `2x2` 空间布局，不再立即压成全局均值；
- 短分支使用 0-0.7 秒连续 10 FPS 特征，长分支使用 0-3.2 秒规则 2.5 FPS
  特征，避免在同一 TCN 内混用非均匀时间间隔；
- 姿态、短期运动和长短期差异分别编码，再融合为 fall/stage/action 表征；
- 增加受控下降辅助头，只在训练中约束 `sit_down/lie_down/kneel_down/
  squat_down` 的共享动态特征；
- 视频块默认使用 top-2 均值参与 MIL/ranking，使多个高分负窗口共同反向传播。

V3.2 仍然只读取既有人物 bbox，不包含场景分支、检测头或状态机。V3、V3.1
checkpoint 的旧模型路径和参数形状保持不变。

### V3.3 Geometry-first Gated Residual

V3.3 不修改 V3.2，而是增加两个可独立训练和比较的路径：

- `v33-g`：只读取 bbox 序列，完全不实例化或执行 RGB backbone；
- `v33-gr`：以同一几何网络为主判，RGB 只能通过不确定性门控添加残差。

几何分支从归一化 bbox 提取中心、宽高、面积、对数高宽比和底边位置，短期
与长期 TCN 分别编码轨迹、速度和加速度。每个时间尺度还计算均值、标准差、
最小值、最大值、范围、时间斜率、端点变化、最大速度和速度峰值时刻。这些统计
均由固定张量算子生成，不引入测试集阈值或不可部署的 tsfresh 特征。

V3.3 将事件状态拆成两个独立 logit：

```text
falling_event_logit  ─┐
                      ├─ 概率 OR ─> fall_logit
fallen_event_logit   ─┘
```

这里的 `fallen_event_logit` 表示“已观察到跌倒后进入倒地阶段”，不同于 3 类
stage 头中的一般 `fallen` 姿态。没有可观察跌倒过程的 fallen-only 视频仍可学习
`stage=fallen`，但其两个事件 logit 都接受负监督，避免把静态倒地姿态直接报警。

`v33-gr` 的融合为：

```text
final_logits = geometry_logits + gate * visual_residual
gate = geometry_uncertainty * sigmoid(geometry_gate)
```

门控偏置默认初始化为 `-2.0`，视觉残差头权重初始化为零，因此训练开始时
`v33-gr` 与几何判断完全一致。门控只读取停止梯度的几何状态，并乘以几何事件
不确定性；训练时默认以 0.3 概率丢弃整条视觉残差。几何 stage/event 头在
`v33-gr` 中还有独立辅助损失，保证 RGB 无法通过早期融合覆盖几何轨迹。
视觉确认分支在主干前把已有的 `128x128` 人物 crop 以区域平均降采样到
`64x64`（可由 `--classifier-visual-input-size` 调整）；这不会改变缓存数据格式，
只减少训练和端侧确认分支的卷积开销。
视觉时序只编码 8 个稀疏锚点：短期 `(0,2,4,7)` 与长期
`(0,8,16,24,32)`；几何分支仍使用完整 15 点时间网格。稀疏视觉路径只承担
歧义确认，不改变几何主判的时间覆盖。

V3.3 的采样和视频目标同时调整：正事件视频在 falling/fallen 候选块之间轮转，
不再在整轮训练中重复一个 fall 锚点；正片段对事件窗口取均值以约束持续覆盖，
负片段仍取最大值惩罚偶发高分。OmniFall 验证集继续负责 checkpoint 选择，外部
数据集只在模型锁定后用于跨域审计。

先训练几何基线：

```powershell
python -u -m edgefall_st.training.train `
  --training-mode bbox-classifier `
  --classifier-temporal-model v33-g `
  --train-windows artifacts\omnifall_v2\windows_train.jsonl `
  --val-windows artifacts\omnifall_v2\windows_val.jsonl `
  --train-actor-crops-root artifacts\actor_crops_v1\train\crops `
  --val-actor-crops-root artifacts\actor_crops_v1\val\crops `
  --epochs 15 `
  --batch-size 72 `
  --windows-per-video 12 `
  --positive-videos-per-batch 2 `
  --workers 4 `
  --freeze-backbone-epochs 0 `
  --device cuda
```

完成 `v33-g` 的 OmniFall 验证后，再独立训练门控视觉残差：

```powershell
python -u -m edgefall_st.training.train `
  --training-mode bbox-classifier `
  --classifier-temporal-model v33-gr `
  --classifier-geometry-init-checkpoint artifacts\checkpoints\<v33-g-run>\best.pt `
  --classifier-freeze-geometry `
  --classifier-geometry-stage-loss-weight 0 `
  --classifier-geometry-fall-loss-weight 0 `
  --classifier-control-loss-weight 0 `
  --classifier-visual-input-size 64 `
  --train-windows artifacts\omnifall_v2\windows_train.jsonl `
  --val-windows artifacts\omnifall_v2\windows_val.jsonl `
  --train-actor-crops-root artifacts\actor_crops_v1\train\crops `
  --val-actor-crops-root artifacts\actor_crops_v1\val\crops `
  --epochs 15 `
  --batch-size 480 `
  --windows-per-video 12 `
  --positive-videos-per-batch 13 `
  --workers 4 `
  --freeze-backbone-epochs 0 `
  --learning-rate 0.001 `
  --save-every 1 `
  --log-interval 20 `
  --device cuda
```

GR 训练从已经锁定的 G 几何主判开始，只优化视觉主干、视觉 TCN、门控和残差
头。`batch-size=480` 对应每批 40 个视频，`positive-videos-per-batch=13`
保持约三分之一正事件视频；较大的学习率用于补偿物理 batch 增大后的更新次数下降。

V3.3-G 的完整结构、训练协议、模型规模、OmniFall/CAUCAFall 结果和部署边界见：

```text
note/V3.3-G模型设计与实验结论_20260726_113926.md
note/FallVideoDataset_V3.3-G实验结果_20260726_131816.md
```

训练 sampler 默认让每批包含 1 个带可观察 fall 窗口的事件视频和若干负视频，
其自然比例接近当前训练集分布。V3 的 BBox criterion 对组内 fall logit
执行 max pooling；V3.2 默认执行 top-2 mean pooling。两者都增加视频片段级
focal loss 与正负片段 ranking loss。正视频
中完全不包含事件区间的普通片段不参与视频级损失，仍接受窗口级监督；
无跌倒事件的视频则直接约束其最高窗口分数。该目标与最终视频最大分数评估对齐，
且仅影响训练，不增加部署参数或推理算子。

启动一个短训练：

```powershell
python -u -m edgefall_st.training.train `
  --training-mode bbox-classifier `
  --epochs 20 `
  --batch-size 72 `
  --windows-per-video 12 `
  --positive-videos-per-batch 1 `
  --workers 4 `
  --freeze-backbone-epochs 0 `
  --backbone-lr-factor 1.0
```

该模式按验证集视频级 `(P90 + P95) / 2` 保存 `best.pt`。这些 bbox 来自
YOLO 伪标注及主动作人物选择，因此所得结果表示“给定现有 bbox 的语义判别能力”。
视频真值来自 `events.csv`，而不是由 `fallen` 窗口反推。
训练增强在时间维保持一致：整批共享水平翻转和光度参数，bbox 轨迹共享相对
平移/缩放扰动，并随机截短部分历史。它们用于缓解第8轮后出现的验证过拟合，
不进入导出模型。

整体数据流：

```
Image
  │
  ▼
LiteBackbone ──► P3 / P4 / P5  (strides 8 / 16 / 32)
  │
  ▼
MultiScaleTemporalFusion ──► 因果时序融合（P3 / P4 / P5 各自独立）
  │
  ▼
LitePANFPN ──► 跨尺度双向融合后的 P3 / P4 / P5
  │
  ▼
UnifiedAnchorFreeHead ──► boxes / objectness / stage / onset / embedding / action
```

## 2. 顶层网络 `EdgeFallST`

### 2.1 组成

`EdgeFallST`（`edgefall_st.py`）按以下顺序串联五个组件：

1. `LiteBackbone`：输出 3 个尺度的特征图。
2. `MultiScaleTemporalFusion`：对每个尺度做因果时序融合。
3. `LitePANFPN`：跨尺度双向融合。
4. `UnifiedAnchorFreeHead`：多任务预测。

### 2.2 稀疏非均匀历史采样（核心设计）

不同尺度使用不同的时间偏移 `history_offsets`（默认值 `edgefall_st.py:16-20`）：

```python
DEFAULT_HISTORY_OFFSETS = {
    "p3": (0, 1, 2, 4),                    # 近距、密集、短期细粒度
    "p4": (0, 1, 2, 4, 8, 16, 24, 31),     # 完整中程
    "p5": (0, 1, 2, 4, 8, 16, 24, 31),     # 完整中程
}
```

说明：

- `offset = 0` 恒为当前帧；其余为过去帧偏移（帧数）。
- `p3` 只采样 4 帧，关注短期瞬时姿态；`p4 / p5` 各采样 8 帧，最远覆盖 31 帧前，用于捕捉完整跌倒过程。
- 这是一种**多尺度时间金字塔**：浅层感受野小、采密；深层感受野大、采疏，以较低算力覆盖较长时间窗。
- `GLOBAL_FRAME_OFFSETS` 是所有尺度 offset 的去重并集（默认 8 个唯一时刻），用于一次性输入训练时的全部帧。

### 2.3 三种 forward 模式

`EdgeFallST.forward` 根据输入维度自动分派：

| 方法                | 输入                               | 用途                                   | 时序状态                    |
| ------------------- | ---------------------------------- | -------------------------------------- | --------------------------- |
| `forward_frame`   | `[B, 3, H, W]`                   | 单帧无记忆推理（冷启动 / 调试）        | 历史置零，仅当前帧有效      |
| `forward_history` | `[B, K, 3, H, W]`                | 训练，K =`global_frame_offsets` 长度 | 由`valid_mask` 标注有效帧 |
| `forward_stream`  | `[B, 3, H, W]` + `StreamState` | 流式推理                               | 复用历史 backbone 特征      |

训练与推理**严格对齐**：训练时以稀疏历史帧为输入，推理时通过 `StreamState` 维护同等偏移的历史特征，二者进入同一时序融合模块。

## 3. `LiteBackbone`（`backbone.py`）

MobileNetV2 风格的轻量二维主干。默认 5 个 stage 宽度 `(16, 32, 64, 128, 192)`，经若干 `InvertedResidual` 下采样得到三个尺度：

```
stem (s=2) → stage1 (s=2) → stage2 (P3, s=8)  → stage3 (P4, s=16) → stage4 (P5, s=32)
```

输出 `{"p3": c2, "p4": c3, "p5": c4}` 的 feature dict。

特点：

- 仅使用标准 2D 算子（1×1 / depthwise 3×3 / BN / SiLU），无 3D 卷积与注意力，对 NPU 量化友好。
- 不同尺度通道数递增，符合「分辨率越低、通道越多」的惯例。

## 4. `MultiScaleTemporalFusion`（`cmtf.py`）

### 4.1 `CausalTemporalFusion`

对单个尺度的 `[B, K, C, H, W]` 序列做因果融合，步骤：

1. **压缩**：`C → C/4`（下限 8）通道压缩，降低融合算力。
2. **生成两路信号并拼接**（总通道 = `C/4 * (2*K - 1)`）：
   - 所有 K 帧的压缩特征；
   - 当前帧与各历史帧的**差分** `(current - past)`，共 `K-1` 路。
3. **1×1 融合 + InvertedResidual**：把拼接特征压回 `C` 通道。
4. **残差加回当前帧**：`features[:, 0] + fuse(...)`。

关键性质：

- **因果性**：只用过去帧预测当前帧，无未来信息泄露。
- **差分通道直接编码运动量**，对跌倒这类强时序事件尤为有效。
- **`valid_mask`** 处理序列头部历史不足的帧：不足 offset 时填零并将该 slot 的 mask 置 0，融合前以 mask 屏蔽零填充。

### 4.2 `MultiScaleTemporalFusion`

对 P3 / P4 / P5 各自独立实例化一个 `CausalTemporalFusion`（分别用各尺度的 `history_offsets`），输入 `sequences` 与 `masks` 两套 dict，输出融合后的特征 dict。

## 5. `LitePANFPN`（`neck.py`）

轻量 PAN-FPN，输入融合后特征 dict，输出双向增强的 `(P3_out, P4_out, P5_out)`：

- **横向连接**：把各尺度原始通道对齐到统一 `output_channels`（默认 96）。
- **自顶向下**：`P5` 上采样 → 与 `P4` 拼接 → `FusionBlock`；再上采样 → 与 `P3` 拼接 → `FusionBlock`。
- **自底向上**：`P3_out` 下采样 → 与 `P4` 拼接 → `FusionBlock`；`P4_out` 下采样 → 与 `P5` 拼接 → `FusionBlock`。

`FusionBlock`（`layers.py`）= 1×1 投影 + `InvertedResidual`，保持轻量。

## 6. `UnifiedAnchorFreeHead`（`head.py`）

无锚框（FCOS / NanoDet 风格）检测头，对三个尺度的特征各跑一个共享 stem（两层 3×3 `ConvBNAct`），再用 1×1 卷积输出 7 路：

| 输出                  | 通道                | 含义                                                                          |
| --------------------- | ------------------- | ----------------------------------------------------------------------------- |
| `boxes`             | 4                   | 到四边距离 → softplus → 归一化`[x1,y1,x2,y2]`（按 `input_size` 归一化） |
| `box_distances`     | 4                   | 原始四边距离（softplus 后），用于损失                                         |
| `objectness_logits` | 1                   | 是否为行人                                                                    |
| `stage_logits`      | `num_stages`=3    | 跌倒阶段（如站立 / 下落 / 倒地）                                              |
| `onset_logits`      | 1                   | 跌倒起始时刻分支                                                              |
| `track_embeddings`  | `embedding_dim`=8 | L2 归一化嵌入，用于跨帧关联（替代大体积 ReID）                                |
| `action_logits`     | `num_actions`=16  | 动作识别分支                                                                  |

每个尺度各网格点展平后沿网格维度拼接，`strides` 张量同时拼接以供解码/损失区分尺度。框解码参见 `_decode_boxes`：以网格中心为锚点，softplus 距离解码后按 stride 与输入尺寸归一化到 `[0, 1]`。

## 7. 流式状态 `StreamState`（`edgefall_st.py`）

流式推理的历史管理：

- 每个 scale 维护一个 `deque` 环形缓冲，长度上限 `max_history = max(global_frame_offsets)`。
- `update()`：把当前 backbone 特征 `detach()` 后入队，溢出则丢弃最旧帧。`detach()` 切断梯度，保证流式推理只更新当前帧。
- `sequence()`：按 `history_offsets` 取帧；不足时填零并返回对应 `valid_mask=0`。
- `forward_stream()` 流程：`backbone(当前帧) → 按 offset 组序列与 mask → _predict → state.update(当前)`。

**关键性能保证**：流式推理时 backbone 每帧只跑一次，历史帧特征从缓冲复用，推理成本 ≈ 单帧 backbone + 轻量融合，与 31 帧历史长度无关，这是边缘部署可行的核心。

## 8. 基础算子（`layers.py`）

| 算子                 | 说明                                                                                                |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `ConvBNAct`        | Conv2d(bias=False) + BatchNorm2d + 激活（relu / silu），padding = kernel_size // 2                  |
| `InvertedResidual` | MobileNetV2 倒残差：1×1 扩张 → depthwise 3×3 → 1×1 压缩；stride=1 且通道不变时残差；末尾再激活 |
| `FusionBlock`      | 1×1 投影 +`InvertedResidual`，用于 PAN-FPN 融合                                                  |

全部为标准 2D 算子，无 3D 卷积、无 attention，便于 NPU 算子覆盖与量化。

## 9. 默认配置小结

| 项               | 默认值                              |
| ---------------- | ----------------------------------- |
| 输入尺寸         | `(288, 512)`                      |
| Backbone 通道    | `(16, 32, 64, 128, 192)`          |
| Neck 通道        | `96`                              |
| 检测头 stem 通道 | `96`（=neck 输出）                |
| 嵌入维度         | `8`                               |
| 跌倒阶段数       | `3`                               |
| 动作类别数       | `16`                              |
| 历史偏移         | P3: 4 帧；P4/P5: 8 帧（最远 31 帧） |
| 激活函数         | `silu`                            |
| 检测尺度 strides | `(8, 16, 32)`                     |

## 10. 信息流时序图（训练 vs 流式）

训练 `forward_history`：

```
[B,K,3,H,W] ──reshape──> [B*K,3,H,W] ──backbone──> {p3,p4,p5}:[B*K,C,h,w]
   ──reshape回[B,K,...]──> 按各尺度 history_offsets 选 slot + slice valid_mask
   ──MultiScaleTemporalFusion──> 融合后 {p3,p4,p5}
   ──LitePANFPN──> 双向融合
   ──UnifiedAnchorFreeHead──> 7 路预测
```

流式 `forward_stream`：

```
[1,3,H,W] ──backbone──> current {p3,p4,p5}
   ──StreamState.sequence(offsets)──> 各尺度序列 + mask（历史来自 deque）
   ──MultiScaleTemporalFusion──> LitePANFPN──> UnifiedAnchorFreeHead
   ──StreamState.update(current)──> 历史 deque 追加当前帧（detach）
```

两者共享同一 `_predict`（融合 + neck + head），保证训练—推理行为一致。
