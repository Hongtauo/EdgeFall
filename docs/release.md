# 发布与维护

源码、配置和文档在 `main` 分支维护。模型权重、数据集视频和训练缓存单独管理。

## 当前发布状态

- 仓库：`https://github.com/Hongtauo/EdgeFall`，首次源码提交于 2026-10-30。
- 源码采用 [GNU AGPL-3.0](../LICENSE)，版权与第三方声明见 [NOTICE](../NOTICE)。
- 模型权重和演示视频尚未发布下载。

## Git 与资产边界

Git 跟踪源码、配置、文档、模型元数据和实验结果 JSON。
`.gitignore` 排除权重、原视频、数据集、训练缓存、虚拟环境和依赖目录。
本地模型和视频不随源码 ZIP 打包。

```bash
python scripts/check_release.py --public
python scripts/package_release.py
```

源码压缩包默认写入 `dist/EdgeFall-source.zip`，按白名单收录，并强制检查大小小于
200,000,000 字节。仅为了本地离线交付已有资产时可显式添加 `--include-local-assets`；
这不意味着视频或模型已获公开再分发许可。

## 发布检查

1. 确认许可证和第三方材料的发布范围。
2. 运行 `python scripts/check_release.py --public`，检查源码、文档与许可文件。
3. 运行相关单元测试和前端构建。有模型权重时，补充视频推理检查。
4. 检查暂存文件，排除权重、视频、缓存和本地凭据。
5. 在 Release 中记录版本、环境、验证结果及下载文件的 SHA256。

## 演示素材

[Web App](showcase.md) 支持实时推理和结果回放，界面显示当前模式。
数据集视频的使用范围和署名要求见[第三方说明](third_party.md)。

Web App 提供源码和许可证链接。部署修改后的版本时，应将源码链接指向实际运行版本的对应源码，并保留适用的版权、许可与无担保声明。
