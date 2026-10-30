# CAUCAFall 作者框外部评测

该目录只用于冻结模型的跨域评测，不接入训练数据管线。评测直接读取作者提供的
逐帧 PNG 和 YOLO 格式 `class_id x_center y_center width height` 标注，其中
`0=nofall`、`1=fall`，框表示当前帧主人物。

V3：

```powershell
python -u -m external_eval.caucafall.evaluate `
  --checkpoint artifacts\checkpoints\20260722_224348_455822\best.pt `
  --run-name v3_epoch10_author_bbox_locked `
  --device cuda
```

V3.2：

```powershell
python -u -m external_eval.caucafall.evaluate `
  --checkpoint artifacts\checkpoints\20260725_105157_173554\best.pt `
  --run-name v32_epoch11_author_bbox_locked `
  --device cuda
```

输出位于 `artifacts/external_evaluation/caucafall/<run-name>/`：

- `parameters.json`：冻结协议和检查点参数；
- `predictions.csv`：逐视频结果；
- `frame_predictions.jsonl`：有效采样帧结果；
- `summary.json`：视频级、帧级、动作、光照和受试者分层指标。

视频分数固定为有效采样帧 `sigmoid(fall_logits)` 的最大值，固定阈值为 0.5。
全量评测不使用 CAUCAFall 选择模型、阈值或状态机参数。
