# 华为昇腾 BACH 病例分类加速项目

<div align="center">

面向华为 ICT 大赛创新赛赛题三的 `before / after` 双模型加速交付仓库

<br/>

![Contest](https://img.shields.io/badge/Contest-Huawei%20ICT%20Competition-C70039?style=for-the-badge&logo=huawei&logoColor=white)
![Task](https://img.shields.io/badge/Task-WSI%20Multi--label%20Classification-0A66C2?style=for-the-badge&logo=googledocs&logoColor=white)
![After](https://img.shields.io/badge/Formal%20After-mixed%20OM%20attn__score__path__norm1-2DA44E?style=for-the-badge)
![Speedup](https://img.shields.io/badge/Speedup-6.2441x-F59E0B?style=for-the-badge)
![ExactMatch](https://img.shields.io/badge/Proxy%20ExactMatch-1.0-6F42C1?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deliverable-238636?style=for-the-badge)

[![分析总表](https://img.shields.io/badge/分析总表-analyse.md-0969DA?style=flat-square)](./analyse.md)
[![过程文档](https://img.shields.io/badge/过程文档-README--process.md-8250DF?style=flat-square)](./README-process.md)
[![部署说明](https://img.shields.io/badge/部署说明-模型部署步骤说明-1F6FEB?style=flat-square)](./最终提交材料/模型部署步骤说明/模型部署步骤说明.md)
[![最小运行包](https://img.shields.io/badge/最小运行包-data--set--minimal.tar.gz-2DA44E?style=flat-square)](./data-set-minimal.tar.gz)

</div>

## 项目简介

本仓库不是单一训练脚本仓，而是一个**比赛收口仓**：当前可直接运行的核心代码集中在 [`src/A3/`](./src/A3/)，A1/A2/A4 的实验结论、测速、对齐和提交材料主要沉淀在 [`logs/A3_output/`](./logs/A3_output/) 与各类报告中。

补充说明：

- 原始训练/测试数据目录（如 `data/BACH/...`）不随当前仓库一并上传，README 中这类路径只用于说明历史数据来源
- 可直接迁移运行的完整目录结构，已经打包在 [`data-set-minimal.tar.gz`](./data-set-minimal.tar.gz) 和最终交付目录中

当前正式交付口径已经统一为：

- `before`：`PyTorch eager`
- `after`：`mixed OM(attn_score_path_norm1) + ACL sync + prefetch + persistent_workers + pin_memory + manifest cache`
- 主任务：`WSI 多标签分类`
- 默认统一入口：[`src/A3/scripts/run_submission_infer.py`](./src/A3/scripts/run_submission_infer.py)

如果你只想知道“现在交什么、怎么跑、数据怎么看”，看这份 README 即可。  
如果你想追溯完整过程、历史实验和每轮收口依据，请看 [`README-process.md`](./README-process.md)。

## 结果总览

### 当前正式 before / after

| 方案 | 编码器后端 | 总耗时(s) | WSI/s | tiles/s | 本地代理 exact_match | 本地代理 macro_f1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| before | PyTorch eager | 157.7303 | 0.063399 | 14.5185 | 1.0000 | 0.933333 |
| after 主提交 | mixed OM `attn_score_path_norm1` | 25.2606 | 0.395874 | 90.6551 | 1.0000 | 0.933333 |
| after 回退 | origin OM | 27.6070 | 0.362226 | 82.9499 | 1.0000 | 0.933333 |

### 关键结论

- 当前正式 `after` 相对 `before` 加速 `6.2441x`
- 总耗时从 `157.7303s` 降到 `25.2606s`
- 本地代理评测 `exact_match = 1.0`、`macro_f1 = 0.933333`
- `mixed OM` 与 `origin OM` 在官方 `10` 张无标签测试 `WSI` 上最终 `pred_labels` 为 `10/10` 完全一致
- `mixed OM` 已经不只是 patch 级候选，而是完成了端到端测速、代理评测和提交闭环的正式主线

### 当前备选链

| 方案 | 定位 | 总耗时(s) | 结论 |
| --- | --- | ---: | --- |
| `origin OM` | 保守回退链 | 27.6070 | 精度口径稳定，但速度略慢于 mixed OM |
| `5%` 结构化剪枝 `OM(origin)` | 轻量候选链 | 29.4291 | 精度达标，但仍比 mixed OM 慢 `16.50%`，暂不替代正式 after |

## 快速开始

### 1. 安装依赖

```bash
python3 -m pip install -r src/A3/requirements.txt
```

说明：

- `after` 的 `.om` 推理需要目标机已正确安装 `CANN` 运行时，且 `python3 -c "import acl"` 可执行
- `before` 若走昇腾 `PyTorch/NPU` 路径，需要环境中存在与当前环境匹配的 `torch_npu`
- 如目标机缺少系统级 `OpenSlide`，建议直接使用最小运行包；该包内置 `_vendor/`，可提供 `tiffslide` 回退读图能力

### 2. 直接运行统一入口

#### before

```bash
python3 src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant before \
  --backend pytorch \
  --input_dir ./input \
  --out_dir ./output/before \
  --save_features
```

#### after

```bash
python3 src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant after \
  --backend om \
  --input_dir ./input \
  --out_dir ./output/after \
  --save_features
```

### 3. 输入输出约定

- 输入目录：`input/`
- 支持格式：`.svs`、`.tif`、`.tiff`
- 输出目录默认为：
  - `before`：`output/before/`
  - `after`：`output/after/`

每次运行后会生成：

- `features/`
- `manifests/`
- `predictions/slide_predictions.csv`
- `reports/run_summary.json`
- `reports/per_slide_timing.csv`

## 当前正式口径

### 正式提交链

- `before`
  - 统一任务、统一切块、统一阈值、统一聚合逻辑
  - 只保留 `PyTorch eager` 作为基线后端
- `after`
  - 与 `before` 完全同任务、同输入输出、同阈值、同聚合逻辑
  - 只替换编码器执行后端和工程化运行方式
  - 当前正式工件已切到 `mixed OM(attn_score_path_norm1)`

### 当前主线对应文件

- 正式 checkpoint：
  - [`logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`](./logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt)
- 正式阈值：
  - [`logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`](./logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json)
- 正式 mixed OM：
  - [`logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om`](./logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om)
- 正式 mixed OM 元信息：
  - [`logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json`](./logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json)
- 正式 after 实测结果：
  - [`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json`](./logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json)
- 当前默认工件选择逻辑：
  - [`src/A3/bach_mil/runtime/submission_defaults.py`](./src/A3/bach_mil/runtime/submission_defaults.py)

### 主线结论的事实来源

- 结果汇总：[`analyse.md`](./analyse.md)
- 过程总文档：[`README-process.md`](./README-process.md)
- mixed OM 正式测速目录：[`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/`](./logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/)
- mixed OM 代理评测目录：[`logs/A3_output/submission_closure/proxy_eval/wsi_mixed_om_attn_score_path_norm1_cv/`](./logs/A3_output/submission_closure/proxy_eval/wsi_mixed_om_attn_score_path_norm1_cv/)

## 仓库怎么读

### 如果你是第一次看这个仓库

1. 先看本 README，确定当前正式交付口径
2. 再看 [`analyse.md`](./analyse.md)，快速了解 A1~A4 的关键数据
3. 如需追溯历史过程，再看 [`README-process.md`](./README-process.md)
4. 如需迁移到服务器运行，看 [`模型部署步骤说明.md`](./最终提交材料/模型部署步骤说明/模型部署步骤说明.md)

### 当前目录分工

```text
.
├── src/
│   └── A3/                         # 当前可直接运行的主线代码
├── logs/
│   ├── A3_output/                  # A3/A4 主线权重、测速、对齐、报告
│   └── sweep_v3/                   # A2 端到端吞吐 sweep 日志
├── 最终提交材料/          # 最终交付材料汇总目录
│   ├── 模型部署步骤说明/
│   ├── 思路和过程报告/
│   └── 最小可运行模型/
├── analyse.md                     # A1~A4 关键数据总表
├── README-process.md              # 过程型长文档
├── data-set-minimal.tar.gz        # 根目录最小运行包
└── 赛题三具体评测要求（必看）.md     # 赛题约束理解
```

## A1 ~ A4 阶段摘要

<details>
<summary><strong>A1 纯推理基线</strong></summary>

`A1` 只测模型前向，不包含 WSI 读图与切块。

| 精度 | 最优 Batch | 吞吐(patch/s) | 延迟(ms/patch) |
| --- | ---: | ---: | ---: |
| FP32 | 48 | 427.6829 | 2.3382 |
| FP16(AMP) | 96 | 972.8688 | 1.0279 |

- 纯前向最优甜点值是 `bs=96 + FP16(AMP)`
- 相对 FP32 纯前向加速 `2.2747x`
- 这组结果说明了 NPU 编码器本身有明显的 batch 甜点值

</details>

<details>
<summary><strong>A2 Photos 端到端链路</strong></summary>

`A2` 用 `Photos` 任务验证“端到端吞吐不只取决于 NPU 前向”。

| 轮次 | 代表配置 | overall tile/s | steady tile/s |
| --- | --- | ---: | ---: |
| baseline | `w6_pf4_bs96` 类基线 | 407.0671 | 432.2236 |
| 第一轮最优 | `w16_pf4_bs96` | 425.7986 | 471.8586 |
| v3 最优 | `w4_pf6_bs96` | 440.46 | 571.21 |

- `A2` 的主要收益来自 CPU 侧读图、预处理、预取和 batch 组织优化
- 这也是为什么后续 `A3` 不会机械地只盯住单次前向速度

</details>

<details>
<summary><strong>A3 WSI 正式提交闭环</strong></summary>

`A3` 是本项目真正的正式主线，核心是：

- 用 `A01~A10 + XML` 构建 WSI 训练与代理评测口径
- 训练 1 个正式 WSI patch encoder
- 固定阈值和聚合逻辑
- 做 `before / after` 统一入口、统一输出、统一测速和统一代理评测

当前最重要文件：

- tile 清单来源：`data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv`（历史训练数据路径，不随当前仓库上传）
- 正式 encoder：[`logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`](./logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt)
- 正式阈值：[`logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`](./logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json)
- 统一入口：[`src/A3/scripts/run_submission_infer.py`](./src/A3/scripts/run_submission_infer.py)

</details>

<details>
<summary><strong>A4 高级优化与备选路线</strong></summary>

`A4` 主要负责离线化、混合精度、剪枝、蒸馏、外部验证等高级优化收口。

已完成并能诚实写进结果页的内容：

- `ONNX` 导出与对齐
- `OM(origin)` 导出与对齐
- `mixed OM(attn_score_path_norm1)` 稳定性收口
- mixed OM 官方整目录测速
- mixed OM 本地代理评测
- `5%` 结构化剪枝的导出、对齐、官方测速与代理复核
- `BRACS 20` 外部多标签验证

仍属于预研或未形成正式提交产物的内容：

- `AIPP`
- `INT8 / PTQ / QAT`
- 当前蒸馏 student 主线替代

</details>

## 交付材料

### 最终交付目录

[`最终提交材料/`](./最终提交材料/)

其中包含：

| 材料 | 位置 | 说明 |
| --- | --- | --- |
| 部署说明 | [`模型部署步骤说明/`](./最终提交材料/模型部署步骤说明/) | 服务器迁移、依赖安装、输入输出约定 |
| 过程报告 | [`思路和过程报告/`](./最终提交材料/思路和过程报告/) | 用于答辩、追溯和补充说明 |
| 最小运行包 | [`最小可运行模型/`](./最终提交材料/最小可运行模型/) | 最终打包交付镜像 |
| 最终 PDF / PPT / DOCX | `华为ICT最终提交材料-我们能拿奖团队 推理加速设计方案.*` | 最终汇报材料 |

### 最小运行包

当前仓库中有两份同类最小运行包：

- 根目录镜像包：[`data-set-minimal.tar.gz`](./data-set-minimal.tar.gz)
- 交付目录镜像包：[`最终提交材料/最小可运行模型/data-set-minimal.tar.gz`](./最终提交材料/最小可运行模型/data-set-minimal.tar.gz)

包内核心结构为：

```text
data-set/
├── input/
├── output/
├── src/A3/
├── logs/A3_output/
├── _vendor/
├── RUN_ME_FIRST.md
├── run_wsi_before.sh
└── run_wsi_after.sh
```

最小包默认模型：

- `before`：`PyTorch checkpoint`
- `after`：`mixed OM(attn_score_path_norm1)`

## 口径说明与常见问题

### 1. 官方 TestDataset 能不能算真实 ACC？

不能。

- 官方 `TestDataset` 无标签
- 它只能用于：
  - 推理速度测试
  - 最终预测结果输出
  - 特征导出

### 2. README 里的 `exact_match = 1.0` 指什么？

它指的是 **本地代理评测口径**，不是官方无标签测试集真实精度。

- 数据来源是 `A01~A10 + XML`
- 指标来源是 slide 级多标签代理评测
- 当前 `before / after / mixed OM / origin OM` 在这个代理口径下都保持 `exact_match = 1.0`

### 3. 为什么 A1 最优 batch 是 `96`，但离线模型工件大多叫 `bs64`？

因为两者回答的是**不同问题**：

- `A1 bs=96` 结论来自“纯编码器前向 benchmark”
- `A3/A4 bs64` 是离线导出、对齐、统一入口和端到端交付收口时固定下来的工件 shape
- 端到端真实瓶颈不只在 NPU 前向，还包含 WSI 扫描、切块、预处理、缓存与数据组织

也就是说：

- `bs=96` 证明了纯前向甜点值
- `bs=64` 则是当前正式交付链实际收口下来的可复现工件规格

### 4. 为什么有些旧报告还写的是 `origin OM`？

因为仓库里保留了完整的过程留痕。

较早阶段的文件，例如：

- [`logs/A3_output/submission_closure/reports/05_before_after速度与精度对比报告.md`](./logs/A3_output/submission_closure/reports/05_before_after速度与精度对比报告.md)
- [`logs/A3_output/submission_closure/reports/06_最终提交材料汇总报告.md`](./logs/A3_output/submission_closure/reports/06_最终提交材料汇总报告.md)
- [`logs/A3_output/submission_closure/optimization_rounds/wsi/rounds_summary.csv`](./logs/A3_output/submission_closure/optimization_rounds/wsi/rounds_summary.csv)

记录的是 `mixed OM` 收口前的阶段状态。

当前正式口径请以以下文件为准：

- [`src/A3/bach_mil/runtime/submission_defaults.py`](./src/A3/bach_mil/runtime/submission_defaults.py)
- [`analyse.md`](./analyse.md)
- [`README-process.md`](./README-process.md) 末尾更新章节
- [`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json`](./logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json)

### 5. `Photos after = 0.945 ACC` 能直接拿来当 before/after 精度对比吗？

不能直接这么写。

- 这组结果只能作为“离线后端可运行的补充代理结果”
- 它不是严格配对的 `before / after` 正式精度口径
- 当前正式对外精度说明应优先使用 `WSI` 本地代理评测结果

## 推荐阅读顺序

- 结果总表：[`analyse.md`](./analyse.md)
- 过程长文：[`README-process.md`](./README-process.md)
- 最终部署说明：[`模型部署步骤说明.md`](./最终提交材料/模型部署步骤说明/模型部署步骤说明.md)
- 交付版过程镜像：[`思路和过程报告/README.md`](./最终提交材料/思路和过程报告/README.md)
- 赛题约束理解：[`赛题三具体评测要求（必看）.md`](<./赛题三具体评测要求（必看）.md>)

## 结论

截至当前仓库状态，项目已经从“实验堆积”收口为一条清晰的正式交付链：

- `before` 是 `PyTorch eager`
- `after` 是 `mixed OM(attn_score_path_norm1)`
- 正式主任务是 `WSI 多标签分类`
- 正式测速、代理评测、最小运行包、部署说明和答辩材料都已形成闭环

如果后续只做答辩、展示、部署或迁移，优先围绕这条主线展开，不再把早期 `origin OM`、旧版 `Photos` 指标和阶段性草稿混在首页口径里。
