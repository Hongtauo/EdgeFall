# 数据集与数据准备

EdgeFall 使用 OmniFall of-syn、Fall Video Dataset 和 CAUCAFall 进行多域训练/验证/测试，UR Fall Detection 作为最终方案锁定后的在线外部测试。仓库不包含这些原始视频、全量标签、冻结特征缓存或训练用人工修订 overlay；请从作者来源获取，并遵守各数据来源的使用与再分发条件。

## 1. 数据来源与本项目协议

| 数据 | 作者来源 | 在本项目中的作用 |
|---|---|---|
| OmniFall | [作者数据页](https://huggingface.co/datasets/simplexsigil2/omnifall)、[作者代码](https://github.com/simplexsigil/omnifall-experiments) | 使用本地 `of-syn` 版本，提供 16 类动作和阶段监督 |
| Fall Video Dataset | [原发布页](https://www.kaggle.com/datasets/payutch/fall-video-dataset) | 真实视频，主要提供视频级跌倒/非跌倒标签 |
| CAUCAFall | [作者 Mendeley Data 页面](https://data.mendeley.com/datasets/7w7fccy7ky/5) | 100 个受控居家动作视频，补充真实人物与视角 |
| UR Fall Detection | [作者数据页](https://www.fenix.ur.edu.pl/~mkepski/ds/uf.html) | 30 段跌倒 + 40 段 ADL 的 RGB 序列，仅外部测试 |

本项目的本地 `of-syn` 清单与作者当前网页的其它 OmniFall 配置、分类数或默认划分不能混用。下面的数值以本项目固定 manifest 为准，不是对当前上游数据集所有配置的统计。

| 数据域 | train | val | test | 划分说明 |
|---|---:|---:|---:|---|
| OmniFall of-syn（最终多域 manifest） | 6,000 | 1,200 | 2,400 | 原完整 train cache 含 8,400 个 train-only 视频，见下文 |
| Fall Video Dataset | 1,048 | 699 | 5,241 | 以项目固定 manifest 的视频 ID 为准 |
| CAUCAFall | 60 | 20 | 20 | 以 manifest 的 subject/group 字段为准 |
| UR Fall Detection | 0 | 0 | 70 | 完全外部评测 |

原 `omnifall_v2` 基础划分为 train/val/test=`8400/1200/2400`。后续多域 manifest 仅登记其中 6,000 个 train 视频；实际训练主缓存仍沿用 8,400 个 train-only 视频。逐 ID 核查显示：train cache 与 val/test 交集均为 0，其余 2,400 个缓存视频没有列在最终多域 manifest 中。原 reviewed base selection 为 8,384 个视频（5,988 个在最终 train manifest 中，2,396 个未列入）；还会加入修正人物 overlay。不能把 manifest train=6000 当成训练实际只使用6000，或把缓存8400误写成最终 manifest 的数量。这个历史差异需要在任何重新训练中保留记录。

网页数据版本可能更新。下载后保存来源版本、文件清单和 SHA256；不能只用同名文件夹来证明和已报告实验数据一致。URFD 作者页面明确将其用于非商业学术用途；公开演示原视频时应核对实际使用范围，并保留来源与许可声明。

## 2. 推荐本地结构

```text
Dataset/
├── omnifall/of-syn/             # of-syn.csv、动作 label map 与视频子目录
├── fall-video-dataset/
│   ├── Fall/
│   └── No_Fall/
├── caucafall/Dataset CAUCAFall/CAUCAFall/
│   └── Subject.*/               # 动作 .avi、PNG 和原帧级标签
└── ur-fall-detection-dataset/   # 作者发布的 camera-0 RGB 序列
artifacts/
├── omnifall_v2/                 # split_manifest.csv、windows_*.jsonl
├── multidomain_cauca60/         # 最终多域 split_manifest.csv
├── pseudo_labels/              # train/val/test_yolo26s_nms
├── actor_targets_yolo26s/
├── frozen_features/
├── manual_actor_overrides/
├── hard_negatives/
└── hard_positives/
```

数据目录保持本地，不提交 Git。Windows 路径可改为任意位置；manifest 的 `path` 是稳定逻辑 ID，而 `video_path` 是解码用本地路径，迁移机器时必须重定位后者。

## 3. 生成 OmniFall 窗口与人物轨迹

以下为 Bash 命令，从仓库根目录执行；PowerShell 使用一行命令或把行尾 `\` 改为反引号。

```bash
python -m edgefall_st.data.prepare \
  --dataset-root Dataset/omnifall/of-syn \
  --output-dir artifacts/omnifall_v2 \
  --effective-fps 10 --window-length 32 --stride 4 \
  --boundary-fps 30 --seed 20260717
```

该步骤校验视频与标注 inventory，写出 split、动作/阶段窗口和事件清单。它只接受本项目解析器支持的 `of-syn` CSV 格式。不要以此命令自动生成的一个新划分，替换已有实验的固定 manifest 后仍沿用旧结果。

使用 YOLO26s 对每个基础 split 的全部视频生成跟踪结果；`--sample-size` 必须等于该 split 的视频数量，不能填 0。首次可以先用少量样本验证环境，正式缓存必须使用完整 split。

```bash
for split in train val test; do
  case "$split" in train) count=8400;; val) count=1200;; test) count=2400;; esac
  python -m edgefall_st.pseudo.pilot \
    --dataset-root Dataset/omnifall/of-syn \
    --split-manifest artifacts/omnifall_v2/split_manifest.csv \
    --split "$split" --sample-size "$count" \
    --output-dir "artifacts/pseudo_labels/${split}_yolo26s_nms" \
    --model models/yolo26s.pt --imgsz 544 960 \
    --confidence 0.10 --iou 0.30 --tracker bytetrack.yaml \
    --quantize 16 --device 0
done

python -m edgefall_st.pseudo.actor_targets \
  --split all --pseudo-root artifacts/pseudo_labels \
  --pseudo-suffix yolo26s_nms \
  --output-root artifacts/actor_targets_yolo26s --workers 8
```

`pseudo.pilot` 发现既有 `pilot_manifest.csv` 时会复用它。若以前只跑了 160 个样本，换成完整数据时使用新的输出目录，或明确用 `--reselect` 重建清单并核对数量。

## 4. 提取冻结全局/ROI 特征

```bash
for split in train val test; do
  python -m edgefall_st.features.extract \
    --split "$split" --dataset-root Dataset/omnifall/of-syn \
    --windows "artifacts/omnifall_v2/windows_${split}.jsonl" \
    --tracks-root "artifacts/pseudo_labels/${split}_yolo26s_nms/tracks" \
    --actor-targets-root "artifacts/actor_targets_yolo26s/${split}/targets" \
    --output-root "artifacts/frozen_features/yolo26s/${split}" \
    --weights models/yolo26s.pt --input-height 544 --input-width 960 \
    --geometry-input-height 288 --geometry-input-width 512 \
    --global-levels p4 p5 --roi-levels p3 p4 \
    --roi-output-size 3 --roi-context-scale 1.25 \
    --effective-fps 10 --device cuda
done
```

输出目录必须不存在。缓存包含 `manifest.json`、`index.jsonl` 与逐视频 `.npz`，不是推理时必须下载的额外模型。不要只复制 index 而遗漏视频数组。

## 5. 多域清单、轨迹与缓存

```bash
python -m edgefall_st.data.multidomain \
  --omnifall-manifest artifacts/omnifall_v2/split_manifest.csv \
  --fall-root Dataset/fall-video-dataset \
  --cauca-root "Dataset/caucafall/Dataset CAUCAFall/CAUCAFall" \
  --omnifall-train-count 6000 --fall-train-count 1048 \
  --fall-val-count 699 --cauca-train-count 60 \
  --seed 20260727 --output artifacts/multidomain_cauca60/split_manifest.csv
```

这会创建一个新确定性划分。精确重现实验必须使用原固定视频 ID 列表和原随机种子；此命令是重新准备数据的接口，不证明新下载的视频版本与原版本相同。

```bash
for domain in fall_video_dataset caucafall; do
  python -m edgefall_st.pseudo.multidomain \
    --manifest artifacts/multidomain_cauca60/split_manifest.csv \
    --domain "$domain" --splits train val test \
    --output-root "artifacts/external_pseudo_labels/${domain}/yolo26s_nms" \
    --model models/yolo26s.pt --imgsz 544 960 \
    --confidence 0.10 --iou 0.30 --tracker bytetrack.yaml \
    --quantize 16 --device 0
done
```

将 `features.extract_multidomain` 的 `--boxes-root` 指向上一步相应 domain 输出目录。为了保持训练配置的历史路径，FVD 各 split 写入 `v35_md_yolo26s/fall_video_dataset`；CAUCA train 写入 `cauca60_yolo26s/caucafall`，val/test 写入 `v35_md_yolo26s/caucafall`。例如：

```bash
python -m edgefall_st.features.extract_multidomain \
  --manifest artifacts/multidomain_cauca60/split_manifest.csv \
  --domain caucafall --split train \
  --boxes-root artifacts/external_pseudo_labels/caucafall/yolo26s_nms \
  --output-root artifacts/frozen_features/cauca60_yolo26s/caucafall/train \
  --contract-cache artifacts/frozen_features/yolo26s/train \
  --weights models/yolo26s.pt \
  --cauca-root "Dataset/caucafall/Dataset CAUCAFall/CAUCAFall" \
  --device cuda
```

FVD 没有可靠细粒度 ADL 标签时只保留视频级监督。它的锁定 test 清单有 8 个视频没有有效窗口，最终端到端视频协议将其计为 0 分，不能在统计前删除。

## 6. 多人审核修订与训练清单

最终模型在 OmniFall train 上使用人工主人物修订。原审核文件 `annotations/omnifall/train_multi_person_decisions.csv` 以及其生成的 selection/overlay 缓存未随本仓库发布。代码支持在获得相应决定后重建：

```bash
python -m edgefall_st.data.manual_actor_overrides \
  --decisions annotations/omnifall/train_multi_person_decisions.csv \
  --tracks-root artifacts/pseudo_labels/train_yolo26s_nms/tracks \
  --windows artifacts/omnifall_v2/windows_train.jsonl \
  --base-cache-index artifacts/frozen_features/yolo26s/train/index.jsonl \
  --output-root artifacts/manual_actor_overrides/omnifall_train_v2
```

此工具输出 `targets/`、`windows.jsonl` 和 `base_cache_selection.jsonl`，随后为修订人物单独提取 overlay：

```bash
python -m edgefall_st.features.extract \
  --split train --dataset-root Dataset/omnifall/of-syn \
  --windows artifacts/manual_actor_overrides/omnifall_train_v2/windows.jsonl \
  --tracks-root artifacts/pseudo_labels/train_yolo26s_nms/tracks \
  --actor-targets-root artifacts/manual_actor_overrides/omnifall_train_v2/targets \
  --output-root artifacts/frozen_features/yolo26s_manual_primary_v2/train \
  --weights models/yolo26s.pt --input-height 544 --input-width 960 \
  --geometry-input-height 288 --geometry-input-width 512 \
  --roi-output-size 3 --roi-context-scale 1.25 --effective-fps 10 --device cuda
```

核对输出 summary 中更正和排除的视频数量。selection 的 `cache_index` 与基础缓存索引顺序绑定，重排缓存后不能复用旧 selection。

最终训练需要训练用困难正负清单与 reviewed overlay。清单生成命令、依赖的历史 HN1/HN3-A 检查点及训练参数见[训练指南](training.md)。当缺少这些产物时，`scripts/train_hn3br.py --dry-run` 会列出缺项；仓库不声称已经具备下载即完整重训练的全部数据。

## 7. 外部 UR Fall 在线测试

解压作者的 camera-0 RGB 序列后，运行 `python -m external_eval.ur_fall_detection_dataset.evaluate_online --help` 确认发现的视频序列。最终协议使用 30 FPS 源帧率、10 FPS 有效采样、固定 `0.7094` 阈值，不读取离线 bbox 或 YOLO 特征缓存。

```bash
python -m external_eval.ur_fall_detection_dataset.evaluate_online \
  --dataset-root Dataset/ur-fall-detection-dataset \
  --checkpoint models/hn3br_epoch008_inference.pt \
  --weights models/yolo26s.pt --source-fps 30 \
  --alarm-threshold 0.7094 \
  --threshold-source "three-domain validation F1; URFD excluded" \
  --confidence 0.10 --iou 0.30 --tracker bytetrack.yaml \
  --quantize 16 --max-actor-missed 2 --max-people 2 \
  --warmup-samples 10 --device cuda --output-dir results/urfd_online
```

保存解码总时长、算法分模块时延和逐视频预测，并将漏报/误报与最终锁定模型的记录核对。不同硬件的速度与公开历史数字不要求逐位一致。真实家庭长时间运行还需要新的现场数据协议，短片段测试不能替代长期误报率评估。
