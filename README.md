# RAMEF：跨转速五样本轴承故障诊断

**Rotation-aware measurement evidence fusion for cross-speed five-shot bearing fault diagnosis**

[English documentation](README_en.md) · [公开仓库](https://github.com/zhangshuang-lin/RAMEF)

本地发布包版本：`2026.10.23-submission-sync`。对应主稿、补充材料与图源身份见[版本映射](docs/MANUSCRIPT_VERSION.json)。该包尚未被声明为新的线上release或DOI归档。

本项目提供RAMEF的模型实现、实验脚本、固定数据划分、预测记录和指标复算工具。RAMEF以一条完整物理测量记录（measurement）为决策单位，通过分布式观测、载波与软阶次证据融合，以及目标support校准，完成跨转速小样本诊断。

## 1. 方法概述

RAMEF的处理流程如下：

```text
完整振动记录
    ↓
八个分布式、不重叠block
    ├── U2-Hz：提取并聚合共振载波表示
    └── 物理路径：计算并汇总几何约束软阶次证据
    ↓
measurement级载波与63维物理描述符融合
    ↓
源特征与每类5条目标support拟合标准化器、正则化线性分类头
    ↓
冻结特征提取器与校准头，对query逐记录诊断
```

- **分布式观测**：从同一记录的不同位置获取八个block，扩展观测覆盖；八个block仍只消耗一个measurement标签，不是八个shot。
- **U2-Hz载波路径**：采用嵌套U结构编码Hz域时频表示，聚合不同位置的载波响应。
- **63维物理路径**：在固定频带内提取包络证据，围绕轴承故障阶次的理论位置计算软频率邻域证据，并汇总跨block统计量；该路径不含可训练参数。
- **目标support校准**：冻结编码器后，用源特征和少量带标签目标support拟合标准化器与多项逻辑回归分类头。query不参与拟合、参数选择或测试时更新。

本文及当前文档统一使用RAMEF。历史脚本、预测CSV或日志中的RAMEF-F是同一完整63维实现的旧别名，保留原字段以维持历史记录及哈希可核对。

## 2. 快速开始

在仓库根目录运行以下命令。文件校验、单元测试和预测指标复算只需Python 3.10及以上版本的标准库，不需要GPU、原始数据或模型权重。

```bash
# 核对文件SHA-256和本地依赖清单
python scripts/verify_bundle.py

# 运行入口保护、去重及指标计算测试
python -m unittest discover -s tests -v

# 根据冻结预测复算论文指标
python scripts/replay_paper_results.py --output-dir runs/replay01
```

`runs/replay01`必须是尚不存在的目录。复算脚本会生成：

| 输出文件 | 内容 |
|---|---|
| `pu_metrics.csv` | 按唯一measurement计票的PU各组指标与混淆矩阵 |
| `mcc5_seed_metrics.csv` | MCC5各方法、各种子的运行级指标 |
| `seed_summary.csv` | 各种子指标的均值与样本标准差 |
| `summary.json` | 关键结果核对、输入哈希及统计口径 |

以上步骤只重新计算已有预测的指标，不重新训练网络或推理目标query。

## 3. 仓库结构

```text
.
├── code/                      # 模型、物理描述符及实验入口的本地依赖闭包
├── scripts/                   # 完整性校验、指标复算和MCC5便携运行入口
├── configs/                   # 当前有效参数及模型初始化配置
├── protocols/                 # PU、MCC5、SQ和CWRU的协议与划分清单
├── evidence/                  # 用于统一指标复算的预测记录
├── reference/                 # 冻结统计、图表数值输入及历史结果快照
├── figures/                   # 与论文图对应的绘图源码和索引
├── docs/                      # 脚本用途、论文映射、模型资产及来源说明
├── tests/                     # 不执行训练或推理的单元测试
├── requirements-model.txt     # 建模依赖；PyTorch需单独配置
├── requirements-figures.txt   # 绘图依赖
├── MANIFEST.sha256            # 文件完整性清单
├── LICENSE_STATUS.md          # 许可证确认事项
└── THIRD_PARTY_NOTICES.md      # 第三方来源与授权说明
```

详细入口及其执行前提见[脚本索引](docs/SCRIPT_INDEX.md)，论文章节与代码对应关系见[论文映射](docs/MANUSCRIPT_MAP.md)。

## 4. 数据集与实验任务

### 4.1 PU跨转速任务

- 三个1500 rpm源工况D0、DM、DF，迁移至900 rpm目标工况DS。
- 三分类：正常（N）、内圈故障（IR）和外圈故障（OR）。
- 源训练与验证分别为816和204条measurement；编码器冻结后，1020条源记录与15条目标support用于最终校准。
- 目标support每类5条，support与query的measurement身份不重叠。
- 主评分对171条不同query measurement各计一票：N、IR、OR分别为55、62、54条。

原登记账本的240个条目存在重复measurement身份，复算时会检查重复预测的一致性，再按唯一身份计票。PU query在项目早期开发中曾被访问，因此PU结果定位为锁定开发队列结果，而非未触碰的外部确认。

### 4.2 MCC5滚道位置识别任务

- 3000-rpm标称运行用于源训练，共42条。
- 2000-rpm标称运行用于源验证，共42条。
- 1000-rpm标称目标域包含15条support与27条query，query的NRF、IR、OR数量分别为13、7、7条。
- 原最终标准化器和分类头仅使用42条源训练运行与15条support；42条源验证运行不进入该拟合。
- NRF表示不存在内圈或外圈滚道故障，可包含其他电机故障，不等同于整机完全健康。

MCC5使用本地预先冻结协议的运行级机理压力测试。CSV运行身份互斥不等同于轴承个体隔离。五个编码器种子共享同一组27条query，不能被视为五个独立目标队列。

### 4.3 CWRU与SQ补充任务

CWRU提供原生十分类跨负载实验，SQ提供原生三分类变速实验。相关源码及协议用于补充天花板与可迁移性检查，不承担RAMEF独立正增益的主张。历史入口的缓存和路径前提见脚本索引。

## 5. 主要参考结果

### PU：锁定开发队列，171条不同measurement

| 方法 | Accuracy（%） | Macro-F1（%） |
|---|---:|---:|
| Centre-crop U2-Hz | 88.89 | 88.16 |
| U2-Hz-DM8 | 97.66 | 97.55 |
| RAMEF | 98.83 | 98.78 |

另一次预先固定种子的五种子RAMEF实验得到Accuracy **94.04% ± 1.91%**、Macro-F1 **93.76% ± 2.00%**。锁定单种子结果与五种子均值应分开解释。

### MCC5：原本地预先指定比较，27条固定query运行

| 方法 | Accuracy（%） | Macro-F1（%） |
|---|---:|---:|
| U2-Hz-LM8 | 67.41 ± 4.06 | 60.25 ± 7.18 |
| U2-Hz-DM8 | 79.26 ± 6.20 | 75.17 ± 7.11 |
| RAMEF | 96.30 ± 2.62 | 96.23 ± 2.81 |

“±”表示五个编码器种子上的样本标准差，反映固定队列上的优化波动，不是新轴承或新设备总体的置信区间。后续容量、维度和同物理证据对照保持其冻结后的补充分析身份，不替换原主结果。

## 6. MCC5原始数据运行

### 6.1 安装建模依赖

建模环境与标准库预测复算环境不同。原实验记录使用Python 3.13.13、PyTorch 2.12.0+cu126、NumPy 2.4.4、SciPy 1.17.1及scikit-learn 1.8.0。

```bash
python -m pip install -r requirements-model.txt
```

上述文件不安装PyTorch。请根据操作系统和GPU环境单独配置PyTorch；原运行版本记录在`configs/recorded_runtime.json`中，不代表代码包提供对应安装包或下载镜像。

### 6.2 准备数据并分阶段执行

从数据提供方自行取得原始MCC5数据。`--data-root`应指向包含`MCC5-THU Motor_speed_circulation`子目录的父目录。下面的`PATH_TO_MCC5_DATA`需替换为实际目录。

```bash
# 可先检查参数及将执行的命令，不读取波形
python scripts/mcc5_pipeline.py prepare --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh --dry-run

# 准备源训练、源验证及support缓存
python scripts/mcc5_pipeline.py prepare --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh

# 源域训练和检查点选择
python scripts/mcc5_pipeline.py train --work-dir runs/mcc5_fresh

# 冻结后执行独立阶段的目标评价
python scripts/mcc5_pipeline.py evaluate --data-root "PATH_TO_MCC5_DATA" --work-dir runs/mcc5_fresh
```

使用新的工作目录，不得指向`evidence/`、`reference/`或原历史档案。评价入口会保留永久锁；不要删除锁后在同一工作目录重复评价。重新训练属于计算复现，不构成新的独立目标队列，其他硬件上的训练也不保证检查点逐位一致。

## 7. PU训练、统计与绘图说明

- **PU原始训练**：历史训练入口及依赖已保留，但部分步骤依赖原阶段缓存和未随包分发的模型资产。本包已验证的是预测重放，不承诺从全新机器一键完成原始PU数据到结果的全过程。
- **PU训练观测**：源训练使用中心crop；冻结后的分布式特征提取使用八个block。不要将二者混写为八块均参与源训练。
- **配对bootstrap**：如需重放，将`reference/pu_paired_bootstrap/`复制到新的临时目录，在副本中执行`python replay.py`。该统计脚本需要NumPy，并会写入派生结果；不要在封存参考目录中直接执行。
- **绘图**：安装`requirements-figures.txt`中的依赖，并查看[图源映射](figures/README.md)。部分图源仍依赖历史阶段布局；图1使用的真实源波形不在代码包内。图7复用冻结坐标，不重新拟合t-SNE。

## 8. 数据字段与命名

| 字段或名称 | 含义 |
|---|---|
| `measurement_id` | PU物理测量记录身份；重复条目在评分时合并 |
| `run_id` | MCC5 CSV运行身份 |
| `cohort` | 区分锁定结果、种子实验或冻结后对照，不能混为一个实验 |
| `seed` | 编码器运行标识；`locked`是锁定事件标签，不是数值随机种子 |
| `method` | 该组预测对应的方法或决策规则 |
| `true` / `prediction` | 真实类别及预测类别 |
| PU类别0/1/2 | N / IR / OR |
| MCC5类别0/1/2 | NRF / IR / OR |
| U2-Hz-K1 | 单block载波观测 |
| U2-Hz-LM8 | 局部八block载波观测 |
| U2-Hz-DM8 | 分布式八block载波观测 |

不同数据集使用各自的观测坐标和原生源域训练设置。SQ历史日志中的Single、Local-M8和Distributed-M8分别对应K1、LM8和DM8；这些别名不表示各数据集使用相同的绝对切片位置。

## 9. 包含与不包含的文件

本包包含代码、协议、预测记录、必要的数值参考输入、测试和文件哈希，不包含第三方原始振动、波形缓存、完整特征缓存、模型检查点、拟合后的分类器、论文稿件或文献PDF。

原模型资产的文件名与SHA-256可在[模型资产清单](docs/EXTERNAL_MODEL_ASSETS.csv)中核对；当前清单不提供下载地址，也不表示这些资产已经公开发布。




## 10. 当前版本与引用

仓库：https://github.com/zhangshuang-lin/RAMEF 。引用未发表稿的方式见[CITATION.md](CITATION.md)。现行论文标题保持不变；尚未采纳的候选标题不写入本包。

图1现行源码为v16及v15依赖；[当前成图](figures/current/Figure_01.svg)可直接查看。原始波形片段不随包分发，重绘前提见[图源说明](figures/README.md)。

本包使用`.gitattributes`保留精确字节；不要批量规范化历史协议、预测或评估锁的换行。代码使用、修改及再分发授权仍见[LICENSE_STATUS.md](LICENSE_STATUS.md)，本次未擅自授予软件许可证。

更新内容见[CHANGELOG.md](CHANGELOG.md)，上传步骤见[UPLOAD_INSTRUCTIONS.md](UPLOAD_INSTRUCTIONS.md)。上传后需从GitHub再下载一次运行校验；本地测试不等于已发布新版本。
