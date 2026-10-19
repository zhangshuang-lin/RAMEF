# RAMEF脚本与预测复算包

本包用于整理与论文 **Rotation-aware measurement evidence fusion for cross-speed five-shot bearing fault diagnosis** 对应的代码、预测账本与复现说明。完整中文说明见 `README.md`，英文说明见 `README_en.md`。

这是供作者自行上传GitHub的轻量包，不包含第三方原始振动、特征缓存、模型检查点、拟合后的分类器、论文Word/PDF或文献PDF。当前本地整理不等于已经完成公开发布。

## 先验证，再复算

在仓库根目录运行。以下步骤只需Python 3.10及以上版本的标准库：

```text
python scripts/verify_bundle.py
python -m unittest discover -s tests -v
python scripts/replay_paper_results.py --output-dir runs/replay01
```

复算脚本读取 `evidence/pu/predictions.csv` 与 `evidence/mcc5/predictions.csv` 等包内账本，重新计算指标并核对参考结果，不读取波形、不训练网络、不选择模型，也不重新推理目标query。输出使用新的工作目录，不覆盖封存证据。

## 目录与用途

- `scripts/`：完整性校验、预测复算和MCC5便携入口。
- `code/`：历史实验入口及其本地Python依赖；文件定位适配不改变原始ID。
- `evidence/`：PU和MCC5预测、参考结果及相关元数据。
- `figures/`：选定图源脚本和可公开的数值输入。
- `docs/`：脚本用途、论文映射、依赖和来源说明。
- `tests/`：仅用标准库的结构与复算测试。

具体运行前提见 `docs/SCRIPT_INDEX.md`，论文证据对应关系见 `docs/MANUSCRIPT_MAP.md`。

## 指标口径

PU按 **171条唯一measurement各计一票**。中心切片、U2-Hz-DM8与锁定RAMEF-F的Accuracy为88.89%、97.66%和98.83%，Macro-F1为88.16%、97.55%和98.78%。240-entry账本仅保留为历史来源，不应当作240条独立measurement。新五种子RAMEF-F的Accuracy为94.04% ± 1.91%，Macro-F1为93.76% ± 2.00%。PU属于历史上已访问的开发队列；measurement隔离不等于轴承个体隔离，block不计为shot。

MCC5原主实验包含42条源训练运行、42条独立源验证运行、15条目标support和27条固定query（NRF/IR/OR = 13/7/7）。五个种子共享27条query，并非五个独立外部队列。完整63维RAMEF的原主结果为Accuracy 96.30% ± 2.62%、Macro-F1 96.23% ± 2.81%。该证据是本地预先冻结协议的运行级机理压力测试，不是第三方平台注册，也不是轴承个体级确认。

NRF表示没有内圈或外圈滚道故障，不表示整机完全健康。容量/维度及2×2等后续对照是目标评估后的补充分析，紧凑融合对照的99.26%不能替代原主结果96.30%。query-free稳定性结论限定于已注册15条support的局部邻域。

## MCC5新工作目录运行

先从官方提供方自行取得数据并遵守其使用条件。按数据加载器要求提供目录，使用新工作目录：

```text
python scripts/mcc5_pipeline.py prepare --data-root "MCC5数据目录" --work-dir runs/mcc5_fresh
python scripts/mcc5_pipeline.py train --data-root "MCC5数据目录" --work-dir runs/mcc5_fresh
python scripts/mcc5_pipeline.py evaluate --data-root "MCC5数据目录" --work-dir runs/mcc5_fresh
```

训练依赖与纯预测复算不同，需安装包内说明列出的PyTorch、NumPy、SciPy、scikit-learn等。原GPU运行版本不保证在其他硬件上得到逐位一致的检查点。评估锁不得删除后重跑；整理本包不表示已经完成全新机器上的原始数据端到端训练验证。

历史PU脚本保留了阶段缓存前提，本包尚未实现全新的便携式RAW-PU端到端训练入口。图源脚本也并非全部能从此仓库一键重建：部分依赖原 `outputs/staged/` 布局，图1使用的真实波形资产不分发。请区分“预测复算可执行”“需自行取得数据的建模流程”与“供追溯的历史源码”。

## 上传前

依次完成 `UPLOAD_CHECKLIST.md`。许可证、作者信息、GitHub URL和归档DOI待作者确认，本包不替作者选择MIT等授权，也不虚构发布地址。SHA-256用于核对文件内容，不构成第三方预注册时间戳。
