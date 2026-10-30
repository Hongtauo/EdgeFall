# 训练与复现

本项目的公开基线是 **HN3-B-R Seed-1 Epoch 8**：冻结 YOLO26s 与几何时序分支，只训练 Global/ROI 视觉时序编码器、几何条件跨模态注意力和预测头。发布目录包含训练源码、最终配置和历史参数证据；原始数据、冻结特征缓存、人工修订 overlay 缓存和困难样本清单需要另外准备。

训练路径与部署路径不同。训练读取离线冻结特征以提高吞吐；`infer.py` 直接读取视频，在线执行检测、跟踪、特征提取与确认。完整训练不是运行提交包示例视频。

## 1. 固定实验身份

| 项目 | 锁定值 |
|---|---|
| 原训练 run | `20260801_195609_932737` |
| 最终选中 checkpoint | 该 run 的 `epoch_008.pt` |
| 随机种子 | `20260726` |
| 原参数证据 | [`reports/training_parameters.yaml`](../reports/training_parameters.yaml) |
| 便携配置 | [`configs/hn3br_train.json`](../configs/hn3br_train.json) |
| 融合模式 | `cross-attention-summary`，4 头，128 维，FFN 256 |
| 最终报警阈值 | `0.7094`，三个验证域 pooled video F1 选取 |

原 run 的 `best.pt` 使用历史缓存标签口径选择，不能替代 `epoch_008.pt`。最终 epoch 是使用锁定视频 manifest 的验证指标重新选择的。测试集和 UR Fall 不用于选 epoch 或调阈值。

配置由原 `arguments` 忠实转录，路径改为 `/`；唯一模型路径替换是将原几何构造文件 `artifacts/checkpoints/20260725_220646_947213/best.pt` 改为 `models/geometry_v33g_inference.pt`。后者是精简推理权重，保留几何网络构造参数与模型状态，文件哈希不同，未包含原优化器状态。使用它启动的是同结构的训练重跑，不是原训练文件的逐字节恢复。

## 2. 环境与启动检查

```bash
conda activate EdgeFall-ST
python -m pip install -e ".[dev,inference]"
python scripts/train_hn3br.py --dry-run
```

`--dry-run` 仅使用标准库，输出完整命令和所有缺失输入，不加载权重、不训练。数据准备完毕后可验证真实训练解析器：

```bash
python scripts/train_hn3br.py --dry-run --validate-parser
```

实际训练命令如下，缺少任意输入会在训练前退出：

```bash
python scripts/train_hn3br.py
```

配置中的相对路径均以仓库根目录解析，可复制配置并通过 `--config /path/to/your_config.json` 调整缓存位置。保留原配置作为审计依据。显存不足时可减小 batch，但这属于新的训练设置，应单独报告。V100 需使用兼容 `sm_70` 的 PyTorch，并把 `amp_dtype` 改为 `float16`；这也不同于原 BF16 训练环境。

## 3. 数据准备顺序

完整命令见[数据指南](datasets.md)。依次准备：

1. 原始 OmniFall of-syn、Fall Video Dataset、CAUCAFall 与固定视频划分；
2. YOLO26s 人体框、ByteTrack 轨迹与主人物选择；
3. YOLO 全局/ROI 冻结特征缓存；
4. OmniFall 多人审核决定、base-cache selection 与修正人物 overlay 缓存；
5. 仅训练划分上的困难负样本和困难正样本清单。

原基础 OmniFall train cache 包含 8,400 个视频，最终多域 manifest 的 train 只列 6,000 个。训练主缓存与额外域缓存的筛选语义不同，最终主缓存没有自动按多域 manifest 缩到 6,000 个；reviewed base selection 使用了其中 8,384 个，并加入修正人物 overlay。经逐视频 ID 核查，这些 train-only 视频与 val/test 没有交集。[数据指南](datasets.md)完整记录数量差异；重新训练时应保存实际参与训练的 ID，不能只报告 manifest 行数。

必须使用同一套 `models/yolo26s.pt`。缓存契约包含权重 SHA256、金字塔层、通道数、ROI 设置、历史偏移和输入尺寸；评测会拒绝不匹配的契约。部分历史 CLI 默认指向 YOLO26x，应显式传入 YOLO26s 路径及输出目录。

几何历史偏移为 `0,1,2,3,4,5,6,7,8,12,16,20,24,28,32`，视觉历史偏移为 `0,2,4,7,8,16,24,32`，均以 10 FPS 采样步计。最长回看 3.2 秒。全局特征 768 维，ROI 特征 `384×3×3`，视觉输入 `544×960`，几何坐标参考 `288×512`，ROI context scale 为 1.25。

## 4. 训练设置和监督

| 设置 | 值 |
|---|---:|
| epochs / selected epoch | 15 / 8 |
| batch windows / windows per video | 144 / 12 |
| 正视频 / 其中困难正视频 | 4 / 2 |
| 语义困难负视频 / 分数困难负视频 | 2 / 1 |
| 其余普通负视频 | 5 |
| AdamW learning rate / weight decay | `6e-4` / `1e-4` |
| warmup epochs / minimum LR ratio | 2 / 0.05 |
| gradient clip / history truncation probability | 10 / 0.2 |
| dropout / workers | 0.1 / 4 |
| AMP | BF16 |

每批共 12 个视频。2 个困难正视频包含在 4 个正视频中。冻结几何分支包含在确认器的 2,202,930 参数内，可训练参数为 1,678,107；YOLO26s 不参与反向传播。

损失权重为 stage 1.0、fall focal 2.0、event state 0.5、onset 0.5、16 类 action 0.05、control 0.25、consistency 0.25、video MIL 0.5、video ranking 0.25、Global/ROI/weak-video auxiliary 各 0.5。Focal alpha/gamma 为 0.75/2.0。`hard_video_ranking_loss_weight=0`；常规 video-ranking 项仍为 0.25，二者不是同一项。

OmniFall 提供强帧级监督，真实数据中的视频级标签通过 MIL 与辅助损失参与，不伪造帧级 onset。最终困难负清单为 4,459 个候选，困难正清单为 327 个视频；类别计数和采样权重可在历史参数报告中核对。

## 5. 困难样本清单

最终清单来自分阶段训练：V3.6 基线挖掘 HN1 负样本；HN1 挖掘低分正视频；HN3-A 刷新负样本；HN3-B-R 在相同采样方案上加入 reviewed actor overlay。所有挖掘仅读取 train split。

```bash
python -m edgefall_st.features.mine_hard_negatives \
  --checkpoint /path/to/hn3a_checkpoint.pt \
  --train-cache artifacts/frozen_features/yolo26s/train \
  --extra-train-cache artifacts/frozen_features/v35_md_yolo26s/fall_video_dataset/train \
  --extra-train-cache artifacts/frozen_features/cauca60_yolo26s/caucafall/train \
  --split-manifest artifacts/multidomain_cauca60/split_manifest.csv \
  --output artifacts/hard_negatives/v36_hn3b_yolo26s/train.jsonl \
  --windows-per-video 12 --semantic-chunks-per-video 2 \
  --score-chunks-per-video 2 --score-threshold 0.25 \
  --batch-size 512 --workers 4 --device cuda

python -m edgefall_st.features.mine_hard_positives \
  --checkpoint /path/to/hn1_checkpoint.pt \
  --train-cache artifacts/frozen_features/yolo26s/train \
  --extra-train-cache artifacts/frozen_features/v35_md_yolo26s/fall_video_dataset/train \
  --extra-train-cache artifacts/frozen_features/cauca60_yolo26s/caucafall/train \
  --split-manifest artifacts/multidomain_cauca60/split_manifest.csv \
  --output artifacts/hard_positives/v36_hn3a_yolo26s/train.jsonl \
  --bottom-fraction 0.2 --windows-per-video 12 \
  --batch-size 512 --workers 4 --device cuda
```

上述命令展示生成接口，前置 HN1/HN3-A 检查点并未随公开仓库提供。直接用最终权重重新挖掘会得到另一套清单，不能称为精确复现本次训练。要重现已报告结果，需要原困难清单、reviewed selection、兼容缓存及其来源检查；没有这些产物时，可以按此流程训练新的可追溯基线并报告差异。

## 6. 选模、测试与阈值

训练后先对所有 epoch 用三个验证域选模，避免直接依赖训练阶段的历史 `best.pt`：

```bash
python -m edgefall_st.evaluation.select_fygr_checkpoint \
  --checkpoint-dir artifacts/checkpoints/YOUR_RUN \
  --cache omnifall=artifacts/frozen_features/yolo26s/val \
  --cache fall_video_dataset=artifacts/frozen_features/v35_md_yolo26s/fall_video_dataset/val \
  --cache caucafall=artifacts/frozen_features/v35_md_yolo26s/caucafall/val \
  --video-manifest artifacts/multidomain_cauca60/split_manifest.csv \
  --output-dir results/model_selection --batch-size 144 --workers 4 --device cuda

python -m edgefall_st.evaluation.fygr \
  --checkpoint artifacts/checkpoints/YOUR_RUN/epoch_008.pt \
  --cache artifacts/frozen_features/yolo26s/test \
  --video-manifest artifacts/multidomain_cauca60/split_manifest.csv \
  --domain omnifall --output-dir results/omnifall_test \
  --batch-size 144 --workers 4 --device cuda
```

`epoch_008.pt` 在第二条中仅表示历史发布模型的选择；新训练应替换为自己验证选中的 epoch。其它域替换 `--cache`、`--domain` 与输出目录。阈值标定入口 `python -m edgefall_st.evaluation.video_threshold --help` 支持多次传入 `--validation-predictions DOMAIN=CSV`，测试预测使用独立的 `--test-predictions DOMAIN=CSV`。P90/P95 是 PR 曲线排序指标，不能作为固定报警阈值。

## 7. 代码范围与复现边界

`models/fygr.py`、`training/train_fygr.py`、`features/*` 和 `evaluation/fygr.py` 是最终训练主路径；`training/train.py` 和部分旧模型模块作为公共基础依赖保留。历史 concat、temporal-token、person-MIL 等模式保留了接口，它们不代表发布权重采用这些结构。V3.7—V3.11 的实验结果不并入最终模型指标。

仓库包含推理代码和最终训练配置，原始数据、特征缓存与部分历史训练产物需另外准备。
具体清单见[数据指南](datasets.md)。
