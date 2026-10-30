# 模型资产

本地发布候选目录包含三个锁定权重；Git 默认不跟踪 `.pt` 文件。模型 Release 尚未发布，
仓库不会在运行时静默下载未核实的权重。当前持有原提交包的开发者可本地恢复：

```bash
python scripts/import_assets.py --from-package /path/to/submission/final
python tools/verify_install.py --device auto
```

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `yolo26s.pt` | 20422725 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| `geometry_v33g_inference.pt` | 2127083 | `cfe72559c61b5808fbc0e6634b35e4ff9ee34deec8f6ddcf590dfbf84e8a6242` |
| `hn3br_epoch008_inference.pt` | 8898520 | `85558b2384e4f363953188f560a2f6838b5b2b5243541e0043dae0f07ffce54e` |

`model_manifest.json` 是机器可读的配置与权重清单。确认器会相对目录寻找几何权重，三个文件应放在一起。
部署脚本自动处理内部模块加载关系，用户仅调用 `infer.py`。

`.pt` 是模型文件格式，不表示完整视频处理、跟踪和时序状态可以直接变成一张 ONNX 图。
后续导出计划见[部署说明](../docs/deployment.md)。权重许可与第三方来源见[第三方说明](../docs/third_party.md)。
