# Fall Video Dataset外部评估

本目录与OmniFall训练管线隔离，只用于评估冻结的V3检查点。数据集不会参与训练、模型选择、阈值选择或超参数调优。

## 评估定义

- 标签来自`Fall`和`No_Fall`目录，仅评估视频级二分类。
- 使用CSV中的17点姿态坐标生成主人物BBox。
- 固定关键点置信度阈值为0.25，每帧至少需要4个有效点。
- 与训练预处理一致，先将原始帧letterbox到288 x 512，再变换BBox并执行ROI Align。
- 使用检查点保存的10 FPS有效帧率和历史偏移。
- 每个视频的分数为所有有效窗口`sigmoid(fall_logits)`的最大值。
- 固定阈值指标使用0.5，不在该测试集上搜索阈值。
- 同时报告可评估子集，以及将无法评估视频记为0分的保守全集结果。

这套协议评估的是给定关键点定位条件下的V3语义分类能力，不代表包含人体检测器的完整端到端性能。

## 数据审计

```powershell
python -m external_eval.fall_video_dataset.audit `
  --dataset-root Dataset\fall-video-dataset `
  --output artifacts\external_evaluation\fall_video_dataset\audit.json
```

## 小规模冒烟测试

```powershell
python -u -m external_eval.fall_video_dataset.evaluate_v3 `
  --checkpoint artifacts\checkpoints\20260722_224348_455822\best.pt `
  --limit-per-class 2 `
  --run-name smoke
```

## 完整锁定测试

```powershell
python -u -m external_eval.fall_video_dataset.evaluate_v3 `
  --checkpoint artifacts\checkpoints\20260722_224348_455822\best.pt `
  --run-name v3_epoch10_locked
```

输出包括：

- `parameters.json`：固定评估参数和检查点信息
- `predictions.csv`：逐视频分数、处理状态和数据质量字段
- `summary.json`：AP、P90、P95、固定阈值混淆矩阵及覆盖率

## 固定协议结果分析

```powershell
python -m external_eval.fall_video_dataset.analyze_results `
  --predictions artifacts\external_evaluation\fall_video_dataset\v3_epoch10_locked\predictions.csv `
  --output artifacts\external_evaluation\fall_video_dataset\v3_epoch10_locked\error_analysis.json
```

分析器只读取已生成的分数，不调整阈值或重新运行模型。它输出正负类分数分布、来源文件名前缀、帧对齐子集、同名正负片段排序以及高置信度错误样本。
