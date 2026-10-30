# 第三方来源与公开边界

代码、训练权重、原始视频和衍生演示素材应分别核对来源。仓库中的实验结果不自动授予原始数据再分发或商业展示权限。以下来源核查日期为 **2026-10-30**。

EdgeFall 源码采用 [GNU AGPL-3.0](../LICENSE)，项目版权声明见 [NOTICE](../NOTICE)。以下第三方材料继续遵循原有许可。

## 数据集

| 名称 | 一级来源 | 本项目用途 | 已核实的公开条件 |
|---|---|---|---|
| OmniFall | [项目主页](https://dsch.ai/omnifall/) / [数据集仓库](https://huggingface.co/datasets/simplexsigil2/omnifall) | of-syn 合成训练、验证、测试 | 需按具体子集的数据卡及上游素材条款获取；不以统一项目名称推断所有素材具有同一许可 |
| Fall Video Dataset | [Kaggle 原始发布页](https://www.kaggle.com/datasets/payutch/fall-video-dataset) | 真实视频的训练、验证、测试 | 本次页面未返回可读许可文本；不将第三方论文声称的许可当作原作者授权 |
| CAUCAFall v4 | [Mendeley DOI 10.17632/7w7fccy7ky.4](https://data.mendeley.com/datasets/7w7fccy7ky/4) | 训练与跨主体测试 | 发布页注明 **CC BY 4.0**；使用时保留作者、数据集、版本和许可归属 |
| UR Fall Detection Dataset | [University of Rzeszow 官方页](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | 锁定后的在线外部评测 | 发布页注明 **CC BY-NC-SA 4.0**、非商业学术用途；商业用途需联系原作者 |

UR Fall 官方联系方式见原页。公开使用视频时需确认原始条款允许对应的用途，并保留必要的署名和许可声明。

本项目的数据划分是本实验协议，不代表各数据集作者发布的唯一标准排行榜协议。尤其是 FVD 的本地划分与其他论文的划分可能不同，跨论文数字不能直接比较。

## 演示截图

首页和演示说明中的 `docs/assets/demo-fall-alert.png` 为 EdgeFall Web App 的运行截图。
其中的视频画面来自 [CAUCAFall v4](https://data.mendeley.com/datasets/7w7fccy7ky/4)，
作者为 Jose Camilo Eraso、Elena Muñoz、Mariela Muñoz 和 Jesus Pinto，
原数据按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。
截图在原始画面上叠加了人物框、分数和报警信息，并包含 EdgeFall 的界面。

## 检测器与依赖

共享视觉前端依赖 [Ultralytics](https://github.com/ultralytics/ultralytics) 与其 YOLO26 权重生态。许可应参照 [Ultralytics 官方许可说明](https://www.ultralytics.com/license) 和实际使用版本的 LICENSE。源码开源策略、商业部署和权重再分发应分别核对；本页不把本项目代码的许可扩展到第三方权重。

PyTorch、TorchVision、OpenCV、NumPy、ByteTrack 相关实现、FastAPI、Vue 等依赖保留各自许可。请在产品发布时依据锁定依赖版本生成完整软件物料清单并保留必要通知。

## 本地素材与公开仓库

原参赛包内含来自多个数据集的演示短片；它们并非 EdgeFall 自有视频。公开仓库默认采用源码、实验摘要和获取指南；第三方数据、视频和权重只有在许可与来源确认后才加入公开 Release。内部路径及样本清单用于复现定位，不构成许可证明。

模型设计图若使用合成或手绘响应，应标为结构示意；模型输出截图和视频应明确标出实际模型、阈值、输入来源，以及 live 在线推理还是 replay 结果回放。

## 引用建议

使用本项目时引用 EdgeFall 仓库及版本；使用数据集时另行引用数据集作者和其原始论文。CAUCAFall 数据作者为 Jose Camilo Eraso、Elena Muñoz、Mariela Muñoz、Jesus Pinto；UR Fall 官方页要求引用 Kwolek 与 Kepski 2014 年关于嵌入式跌倒检测的工作。完整引用格式以各来源页为准。
