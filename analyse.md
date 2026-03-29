# A1-A4 全流程数据与优化分析总表

本文把当前仓库 `logs/` 中最重要的 A1~A4 数据统一整理到一处，便于你写答辩材料、迁移新服务器，以及判断现在该用哪条 before/after 方案。

## 1. 当前推荐交付

### 1.1 当前正式 before / after

- `before`
  - 任务：`WSI 多标签分类`
  - 后端：`PyTorch eager`
  - 目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/`
- `after` 主提交
  - 后端：`mixed OM(attn_score_path_norm1) + ACL sync`
  - 目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/`
- `after` 保守回退
  - 后端：`origin OM`
  - 目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/`

### 1.2 当前轻量化候选

- `5%` 结构化剪枝模型：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/models/prune_ratio_0p05_v3_long_e10/best_pruned.pt`
- 当前结论：
  - 已完成 `ONNX/OM` 导出、`OM` 对齐、官方整目录测速与本地代理复核
  - 官方整目录：`29.4291s`，`77.8141 tiles/s`，`0.339799 WSI/s`
  - 本地代理：`exact_match = 1.0`，`macro_f1 = 0.933333`
  - 但仍比当前正式 `mixed OM` 慢 `16.50%`，所以保留为备选，不替代正式 after

## 2. A1 纯推理性能基线

来源日志：

- `logs/A1_fp32__img224__bs16-512__iters120_warm30_r2.tsv`
- `logs/A1_fp16__img224__bs16-512__iters120_warm30_r2.tsv`

关键结论：

| 精度 | 最优 Batch | 吞吐(patch/s) | 延迟(ms/patch) |
| --- | ---: | ---: | ---: |
| FP32 | 48 | 427.6829 | 2.3382 |
| FP16(AMP) | 96 | 972.8688 | 1.0279 |

- FP16(AMP) 相对 FP32 的纯前向加速：`2.2747x`
- A1 结论：NPU 单模型推理甜点值明确存在，`bs=96 + FP16(AMP)` 是后续 A2/A3 工程优化的重要参考

## 3. A2 Photos 端到端链路

来源日志：

- `logs/A2_final_baseline.json`
- `logs/A2_sweep_summary.tsv`
- `logs/sweep_v3/A2_sweep_v3_20260305_230042.tsv`

### 3.1 A2 基线

| 指标 | 数值 |
| --- | ---: |
| 总 tiles | 5400 |
| 图片数 | 100 |
| overall tile/s | 407.0671 |
| steady tile/s | 432.2236 |
| overall img/s | 7.5383 |
| steady img/s | 8.0041 |

### 3.2 A2 多轮优化结果

| 轮次 | 代表配置 | overall tile/s | steady tile/s | 说明 |
| --- | --- | ---: | ---: | --- |
| baseline | `w6_pf4_bs96` 类基线 | 407.0671 | 432.2236 | 当前 A2 最初完整链路 |
| v2 最优 | `w16_pf4_bs96` | 425.7986 | 471.8586 | 第一轮 workers/prefetch/global-mix 优化 |
| v3 最优 | `w4_pf6_bs96` | 440.46 | 571.21 | 当前 A2 最优稳态吞吐 |

### 3.3 A2 结论

- 相比 baseline，v3 最优：
  - `overall tile/s` 提升约 `8.20%`
  - `steady tile/s` 提升约 `32.16%`
- A2 真正瓶颈主要在：
  - CPU 侧解码
  - tile 生成
  - 预处理与预取
- A2 的意义是证明：
  - 不是只做模型前向优化就够
  - 端到端链路必须同时优化 I/O、预处理和 batch 组织

## 4. A3 WSI 任务闭环

### 4.1 当前正式 WSI 训练链

- tile 清单：
  - `data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv`
- 正式 encoder：
  - `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- 正式阈值：
  - `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- 正式聚合：
  - `TileAgg topk16`

### 4.2 WSI before / after 全流程速度

来源日志：

- `logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/run_summary.json`
- `logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/run/reports/run_summary.json`
- `logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json`

| 方案 | 总耗时(s) | 平均每张 slide(s) | WSI/s | tiles/s |
| --- | ---: | ---: | ---: | ---: |
| before PyTorch | 157.7303 | 15.7730 | 0.063399 | 14.5185 |
| after origin OM | 27.6070 | 2.7607 | 0.362226 | 82.9499 |
| after mixed OM | 25.2606 | 2.5261 | 0.395874 | 90.6551 |

### 4.3 WSI 速度结论

- origin OM 相对 before：
  - 加速比：`5.7134x`
  - 总耗时下降：`82.4973%`
- mixed OM 相对 before：
  - 加速比：`6.2441x`
  - 总耗时下降：`83.9850%`
- mixed OM 相对 origin OM：
  - 额外加速：`1.0929x`
  - 总耗时再下降：`8.4995%`

### 4.4 WSI 本地代理评测

来源：

- before：`logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv`
- origin OM：`logs/A3_output/submission_closure/proxy_eval/wsi_after_om_origin_nw8_cv/cv_summary_mean_std.csv`
- mixed OM：`logs/A3_output/submission_closure/proxy_eval/wsi_mixed_om_attn_score_path_norm1_cv/cv_summary_mean_std.csv`

| 方案 | exact_match | macro_f1 | sample_f1 |
| --- | ---: | ---: | ---: |
| before | 1.0000 | 0.933333 | 1.0000 |
| after origin OM | 1.0000 | 0.933333 | 1.0000 |
| after mixed OM | 1.0000 | 0.933333 | 1.0000 |

结论：

- mixed OM 与 origin OM 在本地代理口径下没有精度损失
- mixed OM 可以正式替代 origin OM 成为当前推荐 after

### 4.5 官方无标签测试集说明

- 目录：`data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI`
- 这个目录只能做：
  - 真正的推理速度测试
  - 最终预测结果输出
  - 特征文件导出
- 不能做真实 ACC：
  - 因为没有标签

## 5. A4 部署优化与调优

### 5.1 ONNX / OM 离线化

来源报告：

- `logs/A3_output/submission_closure/reports/03_ONNX导出与对齐报告.md`
- `logs/A3_output/submission_closure/reports/04_OM导出与对齐报告.md`

关键结果：

| 工件 | 关键指标 |
| --- | --- |
| ONNX | `feature_cosine_mean = 1.0`，`prob_max_abs_diff = 8.64e-07` |
| origin OM | `feature_cosine_mean = 0.9999753`，`prob_max_abs_diff = 0.0043112` |

### 5.2 mixed OM 收口

来源：

- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_precision收口报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_om端到端补充报告.md`

关键结果：

| 项目 | 数值 |
| --- | ---: |
| 最优稳定 profile | `attn_score_path_norm1` |
| patch 级 bench | `514.8922 patch/s` |
| real tile `feature_cosine_mean` | `0.9999683` |
| real tile `prob_max_abs_diff` | `0.0079028` |
| 官方 10 张测试 WSI `pred_labels` vs origin OM | `10/10` 一致 |
| 本地代理 exact_match | `1.0` |
| 本地代理 macro_f1 | `0.933333` |

结论：

- mixed OM 已经从“patch 级候选”升级成“可正式提交的 after 主线”

### 5.3 ACL Python 包装层

来源：

- `logs/A3_output/submission_closure/optimization_stepwise/02_acl_python_runtime_optimized/analysis/acl_runtime_summary.csv`

| host_io_mode | patch/s |
| --- | ---: |
| legacy | 299.1713 |
| buffer_reuse | 292.1168 |

结论：

- 当前 Python 层 `buffer_reuse` 没有带来收益，反而比 `legacy` 慢约 `2.36%`
- 如果后续还要继续冲 OM 纯前向极限，优先考虑更低层实现，而不是继续在当前 Python 包装上堆逻辑

### 5.4 蒸馏

来源：

- `logs/A3_output/submission_closure/optimization_stepwise/04_teacher_student_distill/analysis/04_蒸馏长时训练报告.md`

最优结果：

- 最优 epoch：`11`
- `acc = 0.7681564`
- `macro_f1 = 0.4017274`
- `ovr_auc = 0.8548201`

结论：

- 当前蒸馏 student 精度明显不足，不适合主提交

### 5.5 结构化剪枝

来源：

- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝长时训练报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝5pct补充报告.md`

| 剪枝率 | 最优 epoch | acc | macro_f1 | 相对原始大模型结论 |
| --- | ---: | ---: | ---: | --- |
| 10% | 6 | 0.9078212 | 0.8289096 | `macro_f1` 下降约 `1.240` 个百分点 |
| 5% | 5 | 0.9189944 | 0.8391020 | `acc` +`0.279` 个百分点，`macro_f1` -`0.221` 个百分点 |

结论：

- `5%` 剪枝是当前最值得继续部署化的轻量模型候选

### 5.6 AIPP / 量化

当前状态：

- AIPP：预研中
- INT8/PTQ/QAT：预研中

原因：

- 仓库内还没有能直接复现并提交的正式产物、完整命令和整链路对齐结果
- 当前不能诚实写成“已完成”

## 6. BRACS 20 张外部多标签验证补充

为了验证当前 before / after 在外部数据上的表现，额外下载并跑通了 `20` 张带 `ROI` 分布标注的 `BRACS` 测试 `WSI`。

### 6.1 真值口径

真值来自：

- `BRACS/subset20_test_balanced/meta/BRACS.xlsx`
- 工作表：`WSI_with_RoI_Distribution`

折叠为当前任务 4 类多标签：

- `Normal = N > 0`
- `Benign = PB > 0 or UDH > 0`
- `InSitu = FEA > 0 or ADH > 0 or DCIS > 0`
- `Invasive = IC > 0`

这个口径是“外部代理验证口径”，不是 BRACS 官方原生四分类协议。

### 6.2 20 张 BRACS before / after 速度

来源：

- `BRACS/subset20_test_balanced/runs/bracs20_before_pytorch_v1/reports/run_summary.json`
- `BRACS/subset20_test_balanced/runs/bracs20_after_mixedom_v1/reports/run_summary.json`

| 方案 | 总耗时(s) | WSI/s | tiles/s |
| --- | ---: | ---: | ---: |
| before | 483.1228 | 0.041397 | 19.2022 |
| after | 440.4347 | 0.045410 | 21.0633 |

结论：

- after 相对 before：
  - 加速比：`1.0969x`
  - 总耗时下降：`8.8359%`
  - `tiles/s` 提升：`9.6923%`

### 6.3 默认阈值结果

默认阈值：

- `Benign = 0.1`
- `InSitu = 0.1`
- `Invasive = 0.5`

来源：

- `BRACS/subset20_test_balanced/eval_runs/bracs20_roi_eval_v1/bracs_roi_multilabel_eval_summary.json`

| 方案 | exact_match | macro_f1 | sample_f1 |
| --- | ---: | ---: | ---: |
| before | 0.30 | 0.6882663 | 0.7178571 |
| after | 0.30 | 0.6882663 | 0.7178571 |

说明：

- `before/after` 默认逐张 `pred_labels` 完全一致
- 当前主要错误模式是：
  - 纯 `InSitu` 常被多报成 `Benign;InSitu`
  - 纯 `Benign` 常被多报成 `Benign;InSitu;Invasive`
  - `Normal;Benign` 混合样本常被压成单个 `Benign`

### 6.4 阈值搜索后结果

在同一批 `20` 张上，对 `Benign / InSitu / Invasive` 三阈值做网格搜索，目标优先最大化 `exact_match`，再用 `macro_f1` 打破平局。

最优阈值：

- before：
  - `Benign = 0.311959922314`
  - `InSitu = 0.396706029773`
  - `Invasive = 0.736946821213`
- after：
  - `Benign = 0.331659242511`
  - `InSitu = 0.396754652262`
  - `Invasive = 0.847445487976`

| 方案 | exact_match | macro_f1 | sample_f1 | micro_f1 |
| --- | ---: | ---: | ---: | ---: |
| before tuned | 0.45 | 0.8550549 | 0.7845238 | 0.8000 |
| after tuned | 0.45 | 0.8550549 | 0.7845238 | 0.8000 |

结论：

- 仅靠阈值调节：
  - `exact_match` 可从 `0.30` 提升到 `0.45`
  - `macro_f1` 可从 `0.6883` 提升到 `0.8551`
- 但该 tuned 阈值是在同一批样本上直接搜索得到：
  - 适合“分析参考”
  - 不适合直接替换正式提交阈值

### 6.5 BRACS 20 的当前价值

- 证明了当前 `after` 在外部数据上仍然没有相对 `before` 掉点
- 也暴露出当前默认阈值在跨数据集场景下偏保守/偏黏连的问题
- 后续如果想继续冲“加权评分”，最值得继续的方向有两个：
  - 用外部数据拆分出“小验证集 / 小测试集”再定阈值
  - 做更有针对性的后处理规则，缓解 `Benign / InSitu` 粘连和 `Normal` 丢失

## 7. 关键日志与文件索引

### 7.1 正式主线

- 正式 encoder：
  - `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- 正式阈值：
  - `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- 推荐 after mixed OM：
  - `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om`
- 推荐 after mixed OM meta：
  - `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json`
- 回退 origin OM：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.om`

### 7.2 关键报告

- `logs/A3_output/submission_closure/reports/03_ONNX导出与对齐报告.md`
- `logs/A3_output/submission_closure/reports/04_OM导出与对齐报告.md`
- `logs/A3_output/submission_closure/reports/05_before_after速度与精度对比报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_precision收口报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_om端到端补充报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/04_teacher_student_distill/analysis/04_蒸馏长时训练报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝长时训练报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝5pct补充报告.md`

## 8. 当前最终结论

- 当前正式推荐 after 已经从 `origin OM` 升级到 `mixed OM(attn_score_path_norm1)`
- 它在官方无标签测试 WSI 整目录上比 origin OM 再快约 `8.50%`
- 它在本地 WSI 代理评测上与原正式线 `exact_match / macro_f1` 完全一致
- 它在外部 `BRACS 20` 张多标签验证上与 before 的逐张预测也完全一致
- BRACS 默认阈值下 `exact_match = 0.30`，经外部分析阈值搜索可提升到 `0.45`
- 当前最值得继续推进的下一条部署化路线是：
  - `5%` 剪枝模型继续导出 `ONNX/OM`

## 9. 迁移包与运行入口

这一节已经按当前最新实际状态更新，不再使用旧的 `/tmp/data-set` 和 `/data/demo/data-set.tar`。

### 9.1 当前已经生成的最小迁移包

已准备好的最小迁移包位置：

- 未压缩目录：
  - `/home/ma-user/work/uni_run/release_minimal_a3_wsi_before_after_20260327/data-set`
- 压缩包：
  - `/data/demo/a3_wsi_before_after_minimal_20260327.tar.gz`

当前压缩包大小：

- 约 `1.7G`

推荐新服务器工作目录：

- `/home/modellite/workspace/data-set`

### 9.2 这次迁移的目标

这次迁移包的目标不是“把整个比赛仓库全量搬走”，而是只带走当前最小可运行、可复现、可直接提交的 `WSI before/after` 主线：

- `before`
  - 同任务、同输入输出口径
  - `PyTorch eager` 推理
- `after`
  - 同任务、同输入输出口径
  - 当前最优可用 `mixed OM(attn_score_path_norm1)` 推理

也就是说，这个迁移包只关注一件事：

- 在新服务器上，把待测 `WSI` 放进 `input/`
- 一条命令跑出：
  - `features`
  - `manifests`
  - `slide_predictions.csv`
  - `run_summary.json`

### 9.3 当前最小迁移树

```text
/home/modellite/workspace/data-set/
├── RUN_ME_FIRST.md
├── run_wsi_before.sh
├── run_wsi_after.sh
├── src/
│   └── A3/
├── _vendor/
├── logs/
│   └── A3_output/
│       ├── C_phase/
│       │   └── stage1p5_uni_large_L1_s448_mt40_random_v1/
│       │       └── best.pt
│       ├── E_phase/
│       │   └── tileagg_thresholds_L1_s448_uniStage1p5_v1.json
│       └── submission_closure/
│           └── optimization_stepwise/
│               └── 01_mixed_float16_keep_dtype_modify_mixlist/
│                   └── artifacts/
│                       └── refined_attn_score_path_norm1/
│                           ├── wsi_attn_score_path_norm1.om
│                           ├── wsi_attn_score_path_norm1.meta.json
│                           ├── compile_summary.json
│                           └── bench_summary.json
├── input/
└── output/
```

### 9.4 为什么 `requirements.txt` 说“基本完整”

当前运行用的是：

- `src/A3/requirements.txt`

它已经覆盖了这次最小迁移链所需的大部分 Python 依赖，包括：

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`
- `Pillow`
- `PyYAML`
- `tqdm`
- `safetensors`
- `onnx`
- `onnxruntime`
- `openslide-python`
- `tiffslide`

但是这里要明确区分两类依赖：

- `pip` 能装的 Python 包
- 不能只靠 `pip` 解决的系统或框架依赖

真正还需要目标机自己具备的有：

- `acl`
  - 用于 `after` 的 `OM/ACL` 推理
- `torch_npu`
  - 如果你希望 `before` 在 NPU 上走 `PyTorch/NPU`
- 系统级 `OpenSlide` 共享库
  - 用于 `openslide-python`

### 9.5 为什么 `_vendor/` 必须一起带

我已经在当前环境做过轻量自检：

- `torch / timm / onnxruntime / acl` 可以导入
- `openslide / tiffslide` 没有直接装进当前 Python 环境

所以这次迁移包里必须一并携带：

- `_vendor/`

作用是：

- 当目标机没有装好 `openslide` 时
- 代码会优先尝试 `_vendor` 里的 `tiffslide` 回退

这能显著减少“环境装不全导致 WSI 根本打不开”的风险。

### 9.6 包里实际携带了哪些关键文件

#### before 主模型

- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`

这是当前正式 `WSI encoder` 的 PyTorch checkpoint。

#### 阈值文件

- `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`

这是 `TileAgg topk16` 主线对应的正式阈值。

#### after 主模型

- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json`

这是当前正式推荐的 `after`。

它已经不是旧的 `origin OM`，而是当前数值稳定、速度更优的 `mixed OM(attn_score_path_norm1)`。

#### 统一入口

- `src/A3/scripts/run_submission_infer.py`

这是现在统一的推理总入口，支持：

- `--task {wsi,photos}`
- `--variant {before,after}`
- `--input_dir`
- `--out_dir`
- `--save_features`

### 9.7 新服务器如何解压和摆放

推荐做法：

1. 把压缩包上传到：
   - `/home/modellite/workspace/`
2. 在新服务器执行：

```bash
cd /home/modellite/workspace
tar -xzf a3_wsi_before_after_minimal_20260327.tar.gz
```

解压后应直接得到：

- `/home/modellite/workspace/data-set`

### 9.8 新服务器依赖安装

不需要升级 Python，不需要换成新版本，只用现有 `python3` 即可。

推荐命令：

```bash
cd /home/modellite/workspace/data-set
python3 -m pip install -r src/A3/requirements.txt
```

### 9.9 新服务器环境自检

建议先跑下面两条检查命令。

基础推理依赖检查：

```bash
cd /home/modellite/workspace/data-set
python3 -c "import torch, timm, onnxruntime, acl; print('基础推理依赖正常')"
```

WSI 回退读图检查：

```bash
cd /home/modellite/workspace/data-set
python3 -c "import sys; sys.path.append('_vendor'); import tiffslide; print('WSI 回退读图正常')"
```

如果第一条失败：

- `after` 暂时不能跑
- 说明目标机的 `ACL/CANN` 环境没就绪

如果第二条失败：

- 说明 `_vendor/` 不完整
- 或目标机 Python / 二进制兼容性不一致

### 9.10 输入目录怎么放

把待测 `WSI` 文件放进：

- `/home/modellite/workspace/data-set/input`

支持后缀：

- `.svs`
- `.tif`
- `.tiff`

不要求你提前知道尺寸，脚本会自动：

- 打开整张 WSI
- 扫描组织区域
- 根据当前固定参数切 tile
- 提取特征
- 做 slide 级聚合

当前固定参数仍然是：

- `level = 1`
- `tile_size = 224`
- `step = 448`
- `min_tissue = 0.4`
- `agg = topk_mean_prob`
- `topk = 16`

### 9.11 输出目录会生成什么

`before` 输出目录：

- `/home/modellite/workspace/data-set/output/before`

`after` 输出目录：

- `/home/modellite/workspace/data-set/output/after`

每次 WSI 运行后会生成：

- `features/`
  - 每张 slide 对应一个 `.pt` 特征文件
- `manifests/`
  - 每张 slide 的 tile 清单
- `predictions/slide_predictions.csv`
  - 每张 slide 的最终预测标签与概率
- `reports/run_summary.json`
  - 总耗时、WSI/s、tiles/s、总 tiles、平均每张 slide tiles 数
- `reports/per_slide_timing.csv`
  - 每张 slide 的耗时统计

### 9.12 一键运行方式

#### before

```bash
cd /home/modellite/workspace/data-set
bash run_wsi_before.sh
```

#### after

```bash
cd /home/modellite/workspace/data-set
bash run_wsi_after.sh
```

### 9.13 对应的直接命令

#### before

```bash
cd /home/modellite/workspace/data-set
python3 src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant before \
  --backend pytorch \
  --input_dir input \
  --out_dir output/before \
  --save_features
```

#### after

```bash
cd /home/modellite/workspace/data-set
python3 src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant after \
  --backend om \
  --input_dir input \
  --out_dir output/after \
  --save_features
```

### 9.14 `before` 和 `after` 的真正区别

这次迁移包里，二者的任务定义完全一致：

- 输入目录一致
- 输出字段一致
- WSI 切 tile 参数一致
- 阈值文件一致
- slide 聚合逻辑一致

唯一核心区别是：

- `before`
  - 编码器前向走 `PyTorch`
- `after`
  - 编码器前向走 `OM/ACL`

所以这是一组符合赛题要求的“同任务 before / after 双模型”。

### 9.15 如果 after 跑不起来，怎么排查

优先排查这四件事：

1. `python3 -c "import acl"` 是否成功
2. `wsi_attn_score_path_norm1.om` 是否存在
3. `wsi_attn_score_path_norm1.meta.json` 是否存在
4. `_vendor/` 是否一起带上

如果 `acl` 导入失败：

- 先不要怀疑模型
- 先修复目标机的 `CANN/ACL` 运行环境

如果 WSI 打不开：

- 先检查 `_vendor/`
- 再检查目标机是否缺少 OpenSlide 动态库

### 9.16 当前最小迁移包的边界

这次迁移包只保证：

- 当前 `before` 可跑
- 当前 `after` 可跑
- 能输出特征、预测、运行统计

它没有额外携带：

- 全量比赛训练数据
- 全部历史实验日志
- 所有候选 `OM`
- 全部蒸馏/剪枝/量化中间产物

这样做是为了：

- 控制包体积
- 减少迁移复杂度
- 优先确保“当前最优正式方案”先在新机器稳定复现

## 10. 当前迁移结论

- 当前最小迁移包已经实际生成完成
- 当前正式迁移包是：
  - `/data/demo/a3_wsi_before_after_minimal_20260327.tar.gz`
- 当前正式未压缩目录是：
  - `/home/ma-user/work/uni_run/release_minimal_a3_wsi_before_after_20260327/data-set`
- 当前推荐新服务器目标路径是：
  - `/home/modellite/workspace/data-set`
- 当前 requirements 对 Python 侧依赖已经基本完整
- 不能只靠 requirements 解决的部分是：
  - `acl`
  - `torch_npu`
  - 系统级 OpenSlide
- 当前 `_vendor/` 必须一起迁移
- 当前最推荐执行顺序是：
  - 先解压迁移包
  - 再安装 requirements
  - 再做环境自检
  - 最后跑 `before/after`
