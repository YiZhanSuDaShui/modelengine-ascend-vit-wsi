# 华为昇腾 BACH 病例分类加速项目

本仓库用于华为 ICT 大赛创新赛赛题三的最终收口与交付，当前已经把项目冻结为一套可复现、可解释、可部署的 `before / after` 双模型方案。

如果你只想知道“现在到底交什么、怎么部署、怎么跑”，优先看本文。  
如果你想看全过程、历史实验、训练与评测关系、为什么会这样设计，请继续看 [README-process.md](/home/ma-user/work/uni_run/README-process.md)。

## 0. 2026-03-27 最新口径

本节优先级高于下文旧收口描述；若后文仍出现 `origin OM` 作为正式 after，请以这里为准。

### 0.1 当前推荐交付

- 主任务：`WSI 多标签分类`
- `before`：
  - `PyTorch eager`
  - 同一 `checkpoint / 切块参数 / 阈值 / 聚合逻辑 / 输出结构`
  - 对应速度基线目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/`
- `after` 主提交：
  - `mixed OM(attn_score_path_norm1) + ACL sync + prefetch + persistent_workers + pin_memory + manifest cache`
  - 默认 after 工件已切到该 mixed OM
  - 对应整链路实测目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/`
- `after` 保守回退：
  - `origin OM`
  - 对应目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/`
- 轻量化候选：
  - `5%` 结构化剪枝模型
  - 当前最优权重：`logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/models/prune_ratio_0p05_v3_long_e10/best_pruned.pt`
  - 已完成 `ONNX/OM` 导出、`OM` 对齐、`A01~A10` 代理复核、官方无标签整目录测速
  - 官方整目录实测：`29.4291s / 77.8141 tiles/s / 0.339799 WSI/s`
  - 本地代理：`exact_match = 1.0`，`macro_f1 = 0.933333`
  - 结论：精度达标，但仍比当前正式 `mixed OM` 慢 `16.50%`，暂不替代正式 after

### 0.2 当前最关键数据

#### A1 纯推理基线

- FP32 最优：`bs=48`，`427.6829 patch/s`，`2.3382 ms/patch`
- FP16(AMP) 最优：`bs=96`，`972.8688 patch/s`，`1.0279 ms/patch`
- FP16 相对 FP32 纯前向加速：`2.2747x`

#### A2 Photos 端到端链路

- 基线：`overall_tile_s = 407.0671`，`steady_tile_s = 432.2236`
- 第一轮最优 sweep：`w16_pf4_bs96`，`steady_tile_s = 471.8586`
- v3 最优 sweep：`w4_pf6_bs96`，`overall_tile_s = 440.46`，`steady_tile_s = 571.21`
- 结论：A2 的主要收益来自 CPU 侧读图/预处理/预取优化，而不是单纯继续堆 NPU 前向

#### A3 WSI 正式 before/after

| 口径 | 总耗时(s) | WSI/s | tiles/s | 本地代理 exact_match | 本地代理 macro_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| before PyTorch | 157.7303 | 0.063399 | 14.5185 | 1.0000 | 0.933333 |
| after origin OM | 27.6070 | 0.362226 | 82.9499 | 1.0000 | 0.933333 |
| after mixed OM | 25.2606 | 0.395874 | 90.6551 | 1.0000 | 0.933333 |

- mixed OM 相对 before：
  - 加速比：`6.2441x`
  - 总耗时下降：`83.9850%`
- mixed OM 相对 origin OM：
  - 额外加速：`1.0929x`
  - 总耗时再下降：`8.4995%`
- mixed OM 与 origin OM 在官方 10 张无标签测试 WSI 上最终 `pred_labels`：`10/10` 完全一致

#### A4 高级优化现状

- ONNX：
  - `feature_cosine_mean = 1.0`
  - `prob_max_abs_diff = 8.64e-07`
- origin OM：
  - `feature_cosine_mean = 0.9999753`
  - `prob_max_abs_diff = 0.0043112`
- mixed OM `attn_score_path_norm1`：
  - patch 级 bench：`514.8922 patch/s`
  - 真实 tile 对齐：`feature_cosine_mean = 0.9999683`
  - WSI 本地代理：`exact_match = 1.0`，`macro_f1 = 0.933333`
- 5% 结构化剪枝：
  - 最优 epoch：`5`
  - `acc = 0.9189944`
  - `macro_f1 = 0.8391020`
  - `ONNX` 对齐：`feature_cosine_mean = 1.0`
  - `OM(origin)` 对齐：`feature_cosine_mean = 0.9999768`，`prob_max_abs_diff = 0.0061094`
  - 官方整目录：`29.4291s`，`77.8141 tiles/s`
  - 本地代理：`exact_match = 1.0`，`macro_f1 = 0.933333`
  - 相对原始大模型：
    - `acc` 提升约 `0.279` 个百分点
    - `macro_f1` 下降约 `0.221` 个百分点
  - 相对当前正式 `mixed OM`：
    - 官方整目录总耗时慢 `16.50%`
    - 官方 10 张无标签测试 WSI 最终 `pred_labels` 仅 `7/10` 与 mixed 一致
  - 当前不替代正式 `mixed OM`，保留为“轻量化但未获最快速度”的备选链
- 10% 结构化剪枝：
  - `macro_f1` 下降约 `1.240` 个百分点，略高于 `<1%` 目标
- 蒸馏 student：
  - `acc = 0.7681564`
  - `macro_f1 = 0.4017274`
  - 当前不适合主提交
- AIPP / INT8 量化：
  - 仍属于预研
  - 当前仓库内没有可直接提交的正式 AIPP 下沉链和 PTQ/QAT 最终产物

#### 外部 BRACS 20 张多标签验证

- 真值来源：
  - `BRACS Summary File.xlsx + ROI 类型折叠`
  - 多标签口径：
    - `Normal = N`
    - `Benign = PB/UDH`
    - `InSitu = FEA/ADH/DCIS`
    - `Invasive = IC`
- 默认正式阈值：
  - before/after 都是 `exact_match = 0.30`
  - before/after 都是 `macro_f1 = 0.6883`
- 在同一批 `20` 张外部 WSI 上做阈值搜索后：
  - before/after 都可到 `exact_match = 0.45`
  - before/after 都可到 `macro_f1 = 0.8551`
- 速度：
  - before：`19.2022 tiles/s`
  - after：`21.0633 tiles/s`
  - after 额外提速：`9.69%`
- 说明：
  - tuned 阈值只作为外部分析参考
  - 不直接替换正式提交阈值 JSON

### 0.3 一条命令怎么跑

#### after

当前默认 after 已经切到推荐 mixed OM，所以直接执行：

```bash
python src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant after \
  --backend om \
  --input_dir input \
  --out_dir output_after \
  --save_features
```

#### before

```bash
python src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant before \
  --backend pytorch \
  --input_dir input \
  --out_dir output_before \
  --save_features
```

#### 输入输出约定

- 输入目录支持：`.svs / .tif / .tiff`
- 输出结构固定为：
  - `out_dir/features/`
  - `out_dir/manifests/`
  - `out_dir/predictions/slide_predictions.csv`
  - `out_dir/reports/run_summary.json`

### 0.4 该看哪些说明文件

- 汇总分析：`analyse.md`
- 长过程记录：`README-process.md`
- 赛题约束理解：`赛题三具体评测要求（必看）.md`

### 0.5 已准备好的迁移包

- 最小可运行目录：
  - `/home/ma-user/work/uni_run/tmp/data-set`
- 已打包压缩文件：
  - `/data/demo/data-set.tar`
- 目标机推荐落点：
  - `/home/modellite/workspace/data-set`
- 目录内已包含：
  - `README_DEPLOY.md`
  - `install_requirements.sh`
  - `env_check.sh`
  - `run_before.sh`
  - `run_after.sh`

## 1. 最终结论

### 1.1 当前正式交付口径

- 主任务：`WSI 多标签分类`
- `before`：
  - 同一任务
  - 同一输入输出口径
  - 同一切块参数
  - 同一阈值文件
  - 同一聚合逻辑
  - 编码器执行后端为 `PyTorch eager`
- `after`：
  - 与 `before` 完全同任务、同输入输出
  - 只替换编码器执行后端，不改变切块、阈值、聚合、输出 schema
  - 当前正式 after 选型为：
    - `mixed OM(attn_score_path_norm1) + ACL sync + prefetch + persistent_workers + pin_memory + manifest cache`
    - 对应轮次目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/`
  - 保守回退 after：
    - `OM(origin)`
    - 对应目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/`
  - 轻量化候选 after：
    - `5%` 结构化剪枝 `OM(origin)`
    - 对应目录：`logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/official_runs/prune_ratio_0p05_v3_long_e10/`

### 1.2 当前最重要的结果

- WSI 正式 after 全流程速度：
  - `90.655053 tiles/s`
  - `0.395874 WSI/s`
  - 结果文件：`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/run/reports/run_summary.json`
- WSI 本地代理评测：
  - `before exact_match = 1.0`
  - `after exact_match = 1.0`
  - `before macro_f1 = 0.933333`
  - `after macro_f1 = 0.933333`
  - 精度损失：`0.0` 个百分点
- 5% 结构化剪枝候选：
  - 官方整目录：`29.429120s`，`77.814085 tiles/s`，`0.339799 WSI/s`
  - 相对 before：`5.3597x` 加速，总耗时下降 `81.3421%`
  - 相对 mixed OM：总耗时慢 `16.50%`
  - 本地代理：`exact_match = 1.0`，`macro_f1 = 0.933333`，精度变化 `0.0` 个百分点
  - 官方 10 张无标签测试 WSI 最终 `pred_labels` 与 mixed OM：`7/10` 一致
- WSI 离线模型对齐：
  - ONNX：`feature_cosine_mean = 1.0`
  - OM(origin)：`feature_cosine_mean = 0.999975323677063`
  - prune5 ONNX：`feature_cosine_mean = 1.0`
  - prune5 OM(origin)：`feature_cosine_mean = 0.9999767541885376`
  - 当前正式 after 仍以 `mixed OM` 为主，`origin OM` 与 prune5 OM 作为回退/候选

### 1.3 需要特别记住的口径

- 官方 `TestDataset` 无标签，只能做：
  - 真正的推理速度测试
  - 最终预测结果输出
  - 特征文件导出
- 官方 `TestDataset` 不能做真实 ACC。
- `thumbnails` 只能人工观察，不能当真值。
- 当前 `Photos after = 0.945 ACC` 这一组结果只能作为“离线后端可运行的补充代理结果”，不能直接当成严格配对的 before/after 精度对比结论。

## 2. 数据、训练、评测三条线

## 2.1 数据角色

- `Photos`
  - 400 张带四分类标签的 `.tif`
  - 主要用于训练和验证 patch encoder
- `WSI A01~A10 + XML`
  - 有标注
  - 用于生成 tile 级训练清单与 slide 级多标签
  - 用于本地代理评测
- `官方 TestDataset`
  - 无标签
  - 只用于测速和导出最终预测

## 2.2 当前正式 WSI 训练链路

当前正式 WSI 主线不是“5 个 encoder 训练后再融合成 1 个”，而是：

1. 用 `A01~A10 + XML` 生成 tile 清单  
   文件：`data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv`
2. 用同一批 tile 做 `random_tile` 切分  
   口径：`val_ratio = 0.1`
3. 训练 1 个 WSI patch encoder  
   目录：`logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/`
4. 在 `epoch 1~5` 中按 `macro_f1` 选最优  
   当前最佳是 `epoch 4`
5. 保存为 1 个正式 checkpoint  
   文件：`logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
6. 用这 1 个 encoder 去提 WSI 特征
7. 再在 slide 层做 5 折代理评测、搜阈值、算 exact_match / macro_f1

这里最容易混淆的一点是：

- `epoch`：同一个模型训练了第几轮
- `fold`：数据怎么切训练/验证
- `slide`：一张整张病理大图样本

当前正式 WSI encoder 是：

- `1 次训练`
- `5 个 epoch 里选 1 个最优状态`
- 最后只保留 `1 个 best.pt`

不是：

- `5 折训练 5 个 encoder`
- 再把 5 个 encoder 合并成 1 个

## 2.3 为什么 WSI 5 折结果能是 1.0

当前 WSI `exact_match = 1.0` 的含义是：

- 在本地代理评测口径下
- 用 `A01~A10` 的 slide 级标签做 5 折阈值搜索与验证
- before 和 after 的最终 slide 标签 10/10 一致

它不等于“官方外部测试集真实 100% ACC”，原因是：

- 官方测试集没有标签
- 当前 WSI encoder 训练时，正式 checkpoint 用到的 tile 清单来自 `A01~A10`
- 因此这组结果只能被诚实写成：
  - `本地代理评测结果`
  - 不能写成 `官方测试集真实 ACC`

更详细的解释见 [README-process.md](/home/ma-user/work/uni_run/README-process.md) 末尾新增补充章节。

## 3. 当前正式模型与关键文件

### 3.1 WSI 主线

- 正式 encoder checkpoint：
  - `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- 正式阈值：
  - `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- ONNX：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64.onnx`
- 正式 OM：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.om`
- OM Meta：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.meta.json`

### 3.2 Photos 补充链路

- 严格 before 5 折代理评测：
  - `logs/A3_output/submission_closure/proxy_eval/photos_before_pytorch_strict_cv/`
- 补充型 after 代理评测：
  - `logs/A3_output/submission_closure/proxy_eval/photos_after_om_origin/`

### 3.3 不采用的离线工件

- `wsi_patch_encoder_bs64.om`
  - 旧 `fp16 OM`
  - 速度高，但数值出现 `NaN`
  - 不作为正式提交件
- `mixed_float16 OM`
  - 只保留编译尝试，不作为正式提交件

## 4. 部署迁移说明

## 4.1 最小可运行迁移树

推荐的最小迁移目录如下：

```text
<你的根目录>/
├── README.md
├── README-process.md
├── 赛题三具体评测要求（必看）.md
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
│           └── offline_models/
│               └── wsi/
│                   ├── wsi_patch_encoder_bs64_origin.om
│                   ├── wsi_patch_encoder_bs64_origin.meta.json
│                   ├── wsi_patch_encoder_bs64.onnx
│                   └── wsi_patch_encoder_bs64.meta.json
├── input/
├── output_before/
└── output_after/
```

补充说明：

- 根目录名字可以不是 `data-set`。
- 可以叫：
  - `/home/modellite/workspace/dataset`
  - `/home/modellite/workspace/data-set`
  - 任何别的名字
- 关键不是目录名，而是内部相对结构要保持一致。

## 4.2 当前已经同步到 `/data/demo` 的内容

当前挂载目录内已经准备好一份最小运行包：

- `/home/ma-user/work/uni_run/tmp/data-set/`
- `/data/demo/data-set/`
- `/data/demo/data-set.tar`

如果你直接在 `/data/demo/data-set` 下跑，不需要自己重新拼目录。

## 4.3 一键安装依赖

进入部署根目录后执行：

```bash
cd /home/modellite/workspace/data-set
python3 -m pip install -r src/A3/requirements.txt
```

## 4.4 环境自检

### before 自检

```bash
python3 -c "import torch; print('torch ok')"
python3 -c "import torch_npu; print('torch_npu ok')"
```

### after 自检

```bash
python3 -c "import acl; print('acl ok')"
```

### WSI 读图自检

```bash
python3 -c "import openslide; print('openslide ok')"
```

说明：

- `before` 依赖 `torch_npu`
- `after` 依赖 `acl`
- `openslide` 如果缺失，代码会尽量走 `_vendor + tiffslide` 回退

## 4.5 输入文件格式

当前 WSI 统一入口已经支持：

- `.svs`
- `.tif`
- `.tiff`
- 大小写混合后缀也支持，例如：
  - `.SVS`
  - `.TIF`
  - `.TIFF`

也就是说，你现在可以把待测 WSI 放到：

```text
<你的根目录>/input/
├── sample01.svs
├── sample02.tif
└── sample03.tiff
```

脚本会自动扫描这些文件。

## 4.6 一键运行命令

建议先设置根目录变量：

```bash
export ROOT=/home/modellite/workspace/data-set
cd $ROOT
```

### before

```bash
python3 $ROOT/src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant before \
  --backend pytorch \
  --input_dir $ROOT/input \
  --out_dir $ROOT/output_before \
  --save_features
```

### after

```bash
python3 $ROOT/src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant after \
  --backend om \
  --input_dir $ROOT/input \
  --out_dir $ROOT/output_after \
  --save_features
```

如果你的根目录不是 `data-set`，只改 `ROOT` 就行。

## 4.7 输出结构

### WSI 输出

```text
output_before/ 或 output_after/
├── features/
│   └── *.pt
├── manifests/
│   └── *_manifest.csv
├── predictions/
│   └── slide_predictions.csv
└── reports/
    ├── per_slide_timing.csv
    └── run_summary.json
```

其中：

- `features/*.pt`
  - 每张 WSI 的 tile 特征与 tile 概率
- `manifests/*_manifest.csv`
  - 每张 WSI 的切块清单
- `predictions/slide_predictions.csv`
  - 每张 WSI 的最终标签
- `reports/run_summary.json`
  - 总耗时、平均耗时、WSI/s、tiles/s、总 tiles 等统计

## 5. 当前实测结果

## 5.1 WSI 正式 before / after

### before

- 结果目录：
  - `logs/A3_output/submission_closure/official_runs/wsi_before_unified/`
- 关键速度：
  - `151.361294 s`
  - `0.066067 WSI/s`

### after

- 正式选型：
  - `logs/A3_output/submission_closure/optimization_rounds/wsi/04_after_om_acl_sync/`
- 关键速度：
  - `27.607040 s`
  - `0.362226 WSI/s`
  - `82.949856 tiles/s`
- 相对 before：
  - `5.713409x` 加速
  - 耗时下降 `82.497316%`

## 5.2 WSI 本地代理评测

- before：
  - `logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv`
- after：
  - `logs/A3_output/submission_closure/proxy_eval/wsi_after_om_origin_nw8_cv/cv_summary_mean_std.csv`

结果：

- `exact_match = 1.0 -> 1.0`
- `macro_f1 = 0.933333 -> 0.933333`
- 精度损失：`0.0` 个百分点

## 5.3 Photos 代理评测

### 严格 before

- 目录：
  - `logs/A3_output/submission_closure/proxy_eval/photos_before_pytorch_strict_cv/`
- 结果：
  - `ACC = 0.92`
  - `macro_f1 = 0.919118`
  - `ovr_auc = 0.990958`

### 补充型 after

- 目录：
  - `logs/A3_output/submission_closure/proxy_eval/photos_after_om_origin/`
- 结果：
  - `ACC = 0.945`
  - `macro_f1 = 0.944514`
  - `ovr_auc = 0.996375`

注意：

- 这组 `0.945` 不是严格的每折一一配对 after 结果
- 它只能说明：
  - `Photos OM` 路径能跑
  - 结果可用
- 不能直接拿来当“加速后精度提高了 2.5 个百分点”的正式答辩结论

## 5.4 离线模型对齐

### WSI ONNX

- 文件：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_onnx_parity_summary.json`
- 结果：
  - `feature_cosine_mean = 1.0`
  - `logit_cosine_mean = 1.0`
  - `prob_max_abs_diff = 8.642673492431641e-07`

### WSI OM(origin)

- 文件：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_om_parity_summary_origin.json`
- 结果：
  - `feature_cosine_mean = 0.999975323677063`
  - `logit_cosine_mean = 0.9999200105667114`
  - `prob_max_abs_diff = 0.004311233758926392`

### Photos OM(origin)

- 文件：
  - `logs/A3_output/submission_closure/offline_models/photos/photos_om_parity_summary_origin.json`
- 结果：
  - `feature_cosine_mean = 0.9999548196792603`
  - `logit_cosine_mean = 0.9997665882110596`
  - `prob_max_abs_diff = 0.005356550216674805`

## 6. 纯推理数字与全流程数字不要混写

这是答辩时最容易说乱的一点。

### 6.1 纯前向 patch/s

- A1 `PyTorch FP16` 纯 backbone 基线：
  - `972.868762 patch/s`
- 当前正式可用 `OM(origin)` 纯前向：
  - 约 `171.34 patch/s`
- 旧 `fp16 OM` 纯前向：
  - 约 `618.48 patch/s`
  - 但结果出现 `NaN`，不能正式使用

### 6.2 WSI 全流程速度

当前正式提交更应该强调的是：

- `WSI/s`
- `tiles/s`
- `总耗时`

因为赛题评测不是只看 encoder 空跑，还包含：

- WSI 读图
- 切块
- 过滤组织
- 数据搬运
- 编码器推理
- slide 级聚合
- 结果落盘

所以：

- `972.87 patch/s` 不能直接和 `82.95 tiles/s` 做一一等价比较
- 两者量纲和链路范围不同

## 7. 常见问题

### 7.1 输入图片放哪

放在：

- `<你的根目录>/input/`

WSI 支持：

- `svs`
- `tif`
- `tiff`

### 7.2 output 在哪

取决于你命令里的 `--out_dir`。

如果你按推荐命令跑：

- before 输出在：
  - `<你的根目录>/output_before/`
- after 输出在：
  - `<你的根目录>/output_after/`

### 7.3 根目录名字不一样会不会有影响

没有本质影响。

只要下面这些相对结构还在：

- `src/A3`
- `logs/A3_output`
- `_vendor`
- `input`

就可以。

### 7.4 tile、patch、ViT patch size 到底什么关系

当前工程里：

- 一张 WSI 裁出的 `tile`
- 送进模型的一张 `patch` 输入

这两件事在工程口径里基本可以看成同一个 `224 x 224` 图块。

而 `ViT patch16` 里的 `16`

- 是模型内部把 `224 x 224` 再切成更小 token patch 的网络结构参数
- 不是你在 WSI 外部切块时用的 tile 大小

### 7.5 为什么官方测试集没有 ACC

因为官方 `TestDataset` 没有标签。  
所以只能输出：

- 速度
- 预测结果
- 特征文件

真实 ACC 由组委会统一环境跑完后得到。

### 7.6 为什么 thumbnails 不能当答案

因为它只是缩略图，不是官方标签文件。  
最多用来人工观察，不能拿来复核 ACC。

## 8. 最终建议同步目录

如果你要把当前正式方案同步到另一台昇腾服务器，至少带上：

- `src/A3/`
- `_vendor/`
- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/`
- `tmp/data-set/README_DEPLOY.md`
- `tmp/data-set/install_requirements.sh`
- `tmp/data-set/env_check.sh`
- `tmp/data-set/run_before.sh`
- `tmp/data-set/run_after.sh`

如果直接使用已经准备好的挂载目录，则优先使用：

- `/data/demo/data-set/`

## 9. BRACS 外部多标签验证补充

为了避免只看 BACH `A01~A10` 的本地代理结果过于乐观，额外下载并验证了 `20` 张带真实 `ROI` 分布标注的 `BRACS` 测试 WSI。

外部验证口径：

- 真值来源：`BRACS.xlsx` 的 `WSI_with_RoI_Distribution`
- 4 类折叠规则：
  - `Normal = N > 0`
  - `Benign = PB > 0 or UDH > 0`
  - `InSitu = FEA > 0 or ADH > 0 or DCIS > 0`
  - `Invasive = IC > 0`
- 指标：
  - 多标签 `exact_match`
  - `macro_f1`
  - `sample_f1`

关键结果：

| 口径 | before | after |
| --- | ---: | ---: |
| 默认阈值 exact_match | 0.30 | 0.30 |
| 默认阈值 macro_f1 | 0.6883 | 0.6883 |
| tuned 阈值 exact_match | 0.45 | 0.45 |
| tuned 阈值 macro_f1 | 0.8551 | 0.8551 |

速度结果：

| 方案 | 总耗时(s) | WSI/s | tiles/s |
| --- | ---: | ---: | ---: |
| before | 483.1228 | 0.0414 | 19.2022 |
| after | 440.4347 | 0.0454 | 21.0633 |

结论：

- `after` 相比 `before` 在这 `20` 张外部 WSI 上快约 `9.69%`
- 默认阈值下，`before/after` 的逐张 `pred_labels` 完全一致
- 仅靠阈值搜索即可把 `exact_match` 从 `0.30` 提升到 `0.45`
- 但该阈值搜索是在同一批 `20` 张上完成的，属于外部分析参考，不直接替换正式提交阈值

可复查目录：

- `BRACS/subset20_test_balanced/eval_runs/bracs20_roi_eval_v1/`
- 其中包含：
  - `bracs_truth_multilabel.csv`
  - `bracs_roi_multilabel_eval.csv`
  - `bracs_roi_multilabel_eval_summary.json`
  - `bracs_roi_multilabel_eval_report.md`
  - `before_tuned_thresholds.json`
  - `after_tuned_thresholds.json`

## 10. 最终建议同步目录

如果你要把当前正式方案同步到另一台昇腾服务器，至少带上：

- `src/A3/`
- `_vendor/`
- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/`
- `tmp/data-set/README_DEPLOY.md`
- `tmp/data-set/install_requirements.sh`
- `tmp/data-set/env_check.sh`
- `tmp/data-set/run_before.sh`
- `tmp/data-set/run_after.sh`

如果直接使用已经准备好的迁移包，则优先使用：

- 解包源：
  - `/data/demo/data-set.tar`
- 当前工作区内的未压缩目录：
  - `/home/ma-user/work/uni_run/tmp/data-set/`

简短总结：

- 当前正式提交件已经冻结为 `before(PyTorch eager)` 与 `after(mixed OM attn_score_path_norm1)`。
- 当前正式 after 在官方 10 张无标签测试 WSI 上达到 `90.6551 tiles/s`、`0.395874 WSI/s`，相对 before 提速 `6.2441x`，本地代理精度损失 `0.0` 个百分点。
- 在外部 `BRACS 20` 张多标签验证上，after 相对 before 继续保持 `0` 精度差，同时速度提升约 `9.69%`。
- 现在已经支持 `svs / tif / tiff` 输入，并且部署根目录名字可以自由更换，只要内部结构不乱即可。
