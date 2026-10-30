# 演示视频

原始视频不进入 Git 历史。本地目录可放入已有提交包中的 18 个视频，文件名见
[`../labels.json`](../labels.json) 与 App 的 `DEMO_CLIPS` 配置。

```bash
python scripts/import_assets.py --from-package /path/to/submission/final --include-demo
```

该命令只从你已有的本地包复制文件，不授予数据集视频的再分发或商业使用权限。
也可使用自有且获授权的视频直接运行 `python infer.py --input your_video.mp4`。
Web App 播放列表目前使用固定样本，替换视频时需同步 `DEMO_CLIPS`；回放模式还需重新生成对应预测。

公开项目演示建议使用自制或获书面授权的素材。特别是 UR Fall 数据的非商业学术条款，
需要在商业展示前单独确认。详细来源见[数据说明](../../docs/datasets.md)。
