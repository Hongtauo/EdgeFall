# Fall Video Dataset 的 YOLO 伪框评测

这套代码与关键点框评测分开存放，但共享完全相同的图像预处理、人物裁剪和 V3 推理函数。类别标签不参与 YOLO 检测、ByteTrack 或主人物路径选择。

## 生成伪框

小规模随机试验：

```powershell
python -u -m external_eval.fall_video_dataset.generate_yolo_boxes `
  --output-root artifacts\external_pseudo_labels\fall_video_dataset\yolo26x_nms_sample256 `
  --limit-per-class 128 `
  --video-batch-size 32 `
  --decode-workers 8 `
  --actor-workers 8
```

全量处理时去掉 `--limit-per-class`，并使用新的输出目录：

```powershell
python -u -m external_eval.fall_video_dataset.generate_yolo_boxes `
  --output-root artifacts\external_pseudo_labels\fall_video_dataset\yolo26x_nms `
  --video-batch-size 32 `
  --decode-workers 8 `
  --actor-workers 8
```

默认参数与 OF-Syn 伪框一致：YOLO26x、输入 544 x 960、置信度 0.10、IoU 0.30、ByteTrack、多视频 batch 32、每个视频独立跟踪。

## 评测冻结模型

```powershell
python -u -m external_eval.fall_video_dataset.evaluate_v3_yolo `
  --boxes-root artifacts\external_pseudo_labels\fall_video_dataset\yolo26x_nms_sample256 `
  --checkpoint artifacts\checkpoints\20260722_224348_455822\best.pt `
  --run-name v3_epoch10_yolo_sample256
```

## 与关键点框配对比较

```powershell
python -m external_eval.fall_video_dataset.compare_bbox_sources `
  --reference artifacts\external_evaluation\fall_video_dataset\v3_epoch10_locked\predictions.csv `
  --candidate artifacts\external_evaluation\fall_video_dataset\v3_epoch10_yolo_sample256\predictions.csv `
  --output artifacts\external_evaluation\fall_video_dataset\v3_epoch10_yolo_sample256\paired_bbox_comparison.json `
  --bootstrap-samples 2000
```

比较脚本只使用两边都能评估的相同视频，并报告配对 bootstrap 置信区间和固定阈值下的 McNemar 精确检验。
