# BACH WSI 多标签 / 4分类基线工程

这是一个**可直接改造成你当前赛题三方案**的 PyTorch 基线工程，围绕你目前已经确认的路线组织：

1. **Stage 1：Photos 4分类 patch/field 编码器训练**
2. **Stage 1.5：可选加入 A01~A10.xml 监督切出的 WSI tile 做域对齐**
3. **Stage 2：从 WSI 提取 tile 特征**
4. **Stage 3：Attention / CLAM-style MIL 做 slide-level 多标签聚合**
5. **Stage 4：测试集 Photos / WSI 推理与导出**

> 重要提醒：
>
> - 这个工程**主线支持你想做的 WSI 多标签 MIL**。
> - 同时保留一个**官方评测口径兜底**：Photos 4分类单标签 softmax。
> - `Normal` 在 slide-level 多标签里往往语义不稳定，因为一张病理切片几乎总会包含正常组织。若训练后发现 `Normal` 几乎总是阳性，建议把 WSI slide-level 标签改成 `{Benign, InSitu, Invasive}` 三标签，并把 `Normal` 作为“所有病变标签都不触发时”的 fallback。

## 环境依赖补充

除了 `src/A3/requirements.txt` 里的 Python 依赖，这套工程在实际运行时还依赖下面两项环境能力：

- `torch_npu`：如果你要在华为 Ascend NPU 上运行 A1/A2 脚本，或让 A3 自动选择 `npu:0`，需要安装与当前 CANN/PyTorch 版本匹配的 `torch_npu`。
- 系统级 OpenSlide 动态库：`openslide-python` 只是 Python 绑定，本机还必须提供 OpenSlide 共享库（例如 `libopenslide.so`），否则 WSI 相关脚本无法打开 `.svs`。

## 0. 实测最佳 Photos 配置（2026-03-14）

这是目前在本项目里已经实际跑通、并且 **5 折平均 ACC 超过 0.90** 的配置：

- backbone：`UNI-Large` 本地权重
- 权重文件：`assets/ckpts/UNI/pytorch_model.bin`
- 模型骨架：`vit_large_patch16_224`
- 关键结构参数：
  - `--backbone_pool token`
  - `--backbone_init_values 1.0`
- 训练输入：`224x224`
- 验证方式：`patch_agg` 多尺度聚合
  - `crop_sizes = [512, 1024, 1536]`
  - `stride = 256`
  - `topk_per_size = 8`
  - `agg = topk_mean_logit`
  - `logit_topk = 12`

对应 5 折实测结果：

- 输出目录：`logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1`
- 汇总文件：`logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1/cv_summary_mean_std.csv`
- 平均 `acc = 0.9200`
- 平均 `macro_f1 = 0.9191`
- 平均 `ovr_auc = 0.9910`

5 折逐折结果：

- fold0：`acc = 0.9250`
- fold1：`acc = 0.8750`
- fold2：`acc = 0.9250`
- fold3：`acc = 0.9625`
- fold4：`acc = 0.9125`

更细的验证分析文件：

- 混淆矩阵：`logs/A3_output/reports/stage1_uni_large_cv5_ms512_1024_1536_confusion_matrix.csv`
- 分类报告：`logs/A3_output/reports/stage1_uni_large_cv5_ms512_1024_1536_classification_report.txt`
- 结构化汇总：`logs/A3_output/reports/stage1_uni_large_cv5_ms512_1024_1536_summary.json`

为什么这一版比前面的 `ViT-Base + safetensors` 明显更强：

- 这次不是“随便把本地权重塞进 ViT”，而是把 `UNI-Large` 的真实结构对齐了
- 关键修正是：
  - `global_pool` 从之前错误的 `avg` 改成了 `token`
  - `init_values` 显式设为 `1.0`
- 对齐后，`assets/ckpts/UNI/pytorch_model.bin` 可以做到 `missing=0 / unexpected=0`
- 也就是说，这次加载的是“真正完整匹配的 UNI-Large”，不是近似替代

完整复现命令：

```bash
python src/A3/scripts/run_stage1_cv.py \
  --data_root data/BACH \
  --split_csv data/BACH/derived/split/photos_folds.csv \
  --out_root logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1 \
  --n_folds 5 \
  --model_name vit_large_patch16_224 \
  --backbone_pool token \
  --backbone_init_values 1.0 \
  --input_size 224 \
  --epochs 3 \
  --batch_size 8 \
  --num_workers 4 \
  --lr 3e-4 \
  --encoder_lr 1e-5 \
  --weight_decay 1e-4 \
  --freeze_encoder_epochs 1 \
  --seed 3407 \
  --init_backbone_weights assets/ckpts/UNI/pytorch_model.bin \
  --eval_mode patch_agg \
  --eval_patch_crop_sizes 512 1024 1536 \
  --eval_patch_stride 256 \
  --eval_patch_topk_per_size 8 \
  --eval_patch_min_tissue 0.4 \
  --eval_patch_working_max_side 1536 \
  --eval_patch_batch_size 32 \
  --eval_patch_agg topk_mean_logit \
  --eval_patch_logit_topk 12
```

如果后面继续做 WSI，这套 Stage1 也是当前最推荐的 encoder 起点：

- 模型：`vit_large_patch16_224`
- 特征维度：`1024`
- 额外参数：
  - `--backbone_pool token`
  - `--backbone_init_values 1.0`

也就是说，后续 `extract_bag_features.py`、`infer_photos.py`、`train_wsi_tile_stage1p5.py`、`eval_wsi_tiles_ckpt.py` 这些脚本，已经可以沿用同一套 UNI-Large 结构参数。

## 0.1 实测最佳 WSI 生产线（2026-03-14）

在继续把 D/E/F 跑完后，当前项目里最推荐的 WSI 交付线不是“多尺度一定更强”，而是下面这条更稳、更快的配置：

- Stage1：`UNI-Large`
- Stage1.5：`level=1, step=448, min_tissue=0.4`
- 聚合：`TileAgg(topk_mean_prob, topk=16)`
- 标签口径：`Benign / InSitu / Invasive` 做阈值，`Normal` 走 fallback

为什么选它：

- `L1 sparse` 的 slide-level 5 折结果已经和 multi-scale 持平
- 但它只需要一套特征提取，不需要再跑 `level=2`
- 所以最终测试时更快，结构也更简单

对应结果：

- TileAgg 5 折汇总：`logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv`
- mean `macro_f1 = 0.9333`
- mean `micro_f1 = 1.0000`
- mean `sample_f1 = 1.0000`
- mean `exact_match = 1.0000`

对比文件：

- 方法对比表：`logs/A3_output/reports/wsi_method_comparison.csv`
- 最终总览：`logs/A3_output/reports/final_wsi_pipeline_summary.json`

### 0.1.1 训练侧特征

- 训练 WSI 特征目录：`logs/A3_output/D_phase/wsi_train_features_L1_s448_uniStage1p5_v1`
- 测试 WSI 特征目录：`logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1`
- 测试侧统计：
  - `10` 张 WSI
  - `2294` 个有效 tile
  - 总耗时 `328s`
  - 平均每张 `32.8s`
- 统计文件：`logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1/extract_summary.json`

### 0.1.2 阈值与最终输出

- 阈值文件：`logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- 阈值：
  - `Benign = 0.1`
  - `InSitu = 0.1`
  - `Invasive = 0.5`
- 最终 CSV：`logs/A3_output/G_phase/tileagg_test_wsi_pred_L1_s448_uniStage1p5_topk16_v1.csv`

### 0.1.3 直接复现命令

1. 提测试 WSI 特征

```bash
python src/A3/scripts/extract_bag_features.py \
  --mode scan \
  --slide_dir data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI \
  --ckpt logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt \
  --out_dir logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1 \
  --model_name vit_large_patch16_224 \
  --backbone_pool token \
  --backbone_init_values 1.0 \
  --batch_size 64 --num_workers 4 \
  --level 1 --tile_size 224 --step 448 --min_tissue 0.4
```

2. 校准阈值

```bash
python src/A3/scripts/calibrate_wsi_tileagg_thresholds.py \
  --feature_dir logs/A3_output/D_phase/wsi_train_features_L1_s448_uniStage1p5_v1 \
  --bag_csv data/BACH/derived/split/wsi_train_bags_mt40.csv \
  --out_json logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json \
  --label_names Benign InSitu Invasive \
  --agg topk_mean_prob --topk 16 \
  --all_positive_threshold 0.5 \
  --all_negative_threshold 1.0
```

3. 生成最终多标签结果

```bash
python src/A3/scripts/infer_wsi_tileagg.py \
  --feature_dir logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1 \
  --out_csv logs/A3_output/G_phase/tileagg_test_wsi_pred_L1_s448_uniStage1p5_topk16_v1.csv \
  --label_names Normal Benign InSitu Invasive \
  --agg topk_mean_prob --topk 16 \
  --thresholds_json logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json \
  --default_threshold 1.0 \
  --normal_fallback
```

### 0.1.4 MIL 实验线与热图

MIL 这次也补跑了单尺度 5 折：

- 目录：`logs/A3_output/E_phase/mil_cv_L1_s448_uniStage1p5_v1`
- mean `macro_f1 = 0.9333`
- 但因为 `A01~A10` 的 slide 标签里 `Invasive` 全正、`Normal` 全负，MIL 更适合作为实验线和可解释性线，不作为最终交付主线。

可解释性热图示例：

- `logs/A3_output/F_phase/attention_A01_fold0_L1s448_uniStage1p5_v1/attn_Benign.png`
- `logs/A3_output/F_phase/attention_A01_fold0_L1s448_uniStage1p5_v1/attn_InSitu.png`
- `logs/A3_output/F_phase/attention_A01_fold0_L1s448_uniStage1p5_v1/attn_Invasive.png`
- 元信息：`logs/A3_output/F_phase/attention_A01_fold0_L1s448_uniStage1p5_v1/slide_attention_meta.json`

---

## 一、推荐目录

你的数据目录应保持如下结构：

```text
data/BACH/
├── derived/
│   ├── cache/
│   ├── split/
│   └── tiles/
├── ICIAR2018_BACH_Challenge/
│   ├── Photos/
│   └── WSI/
└── ICIAR2018_BACH_Challenge_TestDataset/
    ├── Photos/
    └── WSI/
```

---

## 二、建议训练顺序

### 第 0 步：生成 Photos 交叉验证划分

```bash
python scripts/prepare_photo_splits.py \
  --data_root data/BACH \
  --out_dir data/BACH/derived/split \
  --n_splits 5 \
  --seed 3407
```

### 第 1 步：训练 Photos 4分类 patch/field 编码器

```bash
python scripts/train_patch_stage1.py \
  --data_root data/BACH \
  --split_csv data/BACH/derived/split/photos_folds.csv \
  --fold 0 \
  --out_dir outputs/stage1_fold0 \
  --model_name vit_base_patch16_224 \
  --input_size 224 \
  --epochs 30 \
  --batch_size 24 \
  --lr 3e-4 \
  --encoder_lr 1e-5 \
  --freeze_encoder_epochs 2
```

### 第 1.5 步：从 A01~A10.xml 构建监督 tile manifest（可选但推荐）

```bash
python scripts/build_wsi_manifest.py \
  --data_root data/BACH \
  --wsi_dir data/BACH/ICIAR2018_BACH_Challenge/WSI \
  --out_csv data/BACH/derived/wsi_train_tiles.csv \
  --out_bag_csv data/BACH/derived/wsi_train_bags.csv \
  --level 1 \
  --tile_size 224 \
  --step 224 \
  --min_tissue 0.40
```

### 第 1.6 步：把监督 tile 混到 Stage 1 再训一轮（可选）

```bash
python scripts/train_patch_stage1.py \
  --data_root data/BACH \
  --split_csv data/BACH/derived/split/photos_folds.csv \
  --fold 0 \
  --out_dir outputs/stage1_plus_wsi_fold0 \
  --init_ckpt outputs/stage1_fold0/best.pt \
  --extra_tiles_csv data/BACH/derived/wsi_train_tiles.csv \
  --model_name vit_base_patch16_224 \
  --input_size 224 \
  --epochs 10 \
  --batch_size 24 \
  --lr 1e-4 \
  --encoder_lr 5e-6
```

### 第 2 步：从训练 A01~A10 WSI 提特征

```bash
python scripts/extract_bag_features.py \
  --mode manifest \
  --manifest_csv data/BACH/derived/wsi_train_tiles.csv \
  --ckpt outputs/stage1_plus_wsi_fold0/best.pt \
  --out_dir outputs/wsi_train_features \
  --model_name vit_base_patch16_224 \
  --batch_size 64
```

### 第 3 步：训练 slide-level 多标签 MIL

```bash
python scripts/train_mil_stage2.py \
  --bag_csv data/BACH/derived/wsi_train_bags.csv \
  --feature_dir outputs/wsi_train_features \
  --out_dir outputs/mil_fold0 \
  --num_folds 5 \
  --fold 0 \
  --feature_dim 768 \
  --max_instances 1024 \
  --epochs 80 \
  --lr 1e-4
```

### 第 4 步：从测试 WSI 自动切 patch 并提特征

```bash
python scripts/extract_bag_features.py \
  --mode scan \
  --slide_dir data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI \
  --ckpt outputs/stage1_plus_wsi_fold0/best.pt \
  --out_dir outputs/wsi_test_features \
  --model_name vit_base_patch16_224 \
  --batch_size 64 \
  --level 1 \
  --tile_size 224 \
  --step 224 \
  --min_tissue 0.40
```

### 第 5 步：WSI 多标签推理

```bash
python scripts/infer_wsi.py \
  --feature_dir outputs/wsi_test_features \
  --mil_ckpt outputs/mil_fold0/best.pt \
  --thresholds_json outputs/mil_fold0/thresholds.json \
  --out_csv outputs/mil_fold0/test_wsi_predictions.csv \
  --feature_dim 768
```

### 第 6 步：Photos 测试集 4分类推理

```bash
python scripts/infer_photos.py \
  --data_root data/BACH \
  --ckpt outputs/stage1_plus_wsi_fold0/best.pt \
  --out_csv outputs/stage1_plus_wsi_fold0/test_photos_predictions.csv \
  --model_name vit_base_patch16_224 \
  --input_size 224
```

---

## 三、最重要的建模建议

### 1. Photos 不是废数据，而是你最稳的强监督来源

- 400 张 tiff 是单标签、干净、平衡的 4分类数据。
- 这批数据最适合先把 encoder 训练稳。
- Stage 1 的目标不是直接解决 WSI，而是学到**可迁移到 WSI tile 的病理形态特征**。

### 2. A01~A10.xml 最值钱的作用不是交分割结果，而是：

- 生成监督 tile
- 帮你验证 patch 尺度是否合适
- 给 MIL 提供 slide-level 多标签真值
- 做 attention 热图和 patch 判别分析

### 3. `01.svs ~ 20.svs` 第一版可以先不用

第一版以跑通闭环为主：
- Photos → patch encoder
- A01~A10.xml → 监督 tile / 多标签 bag
- Test WSI → 推理

第二版再考虑：
- 用 `01~20.svs` 做 stain statistics、无监督自训练、难负样本挖掘、伪标签

### 4. WSI 多标签里 `Normal` 要谨慎

如果你发现：
- 几乎所有 slide 都含正常组织
- `Normal` 标签几乎总是 1

那就改成：
- slide-level 只做 `{Benign, InSitu, Invasive}` 三标签
- `Normal = 所有病变分数都低于阈值`

工程里已经预留了这种改法，只要改 `label_names` 即可。

---

## 四、建议超参数

### Stage 1（Photos）

- input size: 224
- optimizer: AdamW
- head lr: 3e-4
- encoder lr: 1e-5
- epochs: 30~40
- augment:
  - RandomResizedCrop
  - HorizontalFlip
  - VerticalFlip
  - ColorJitter
  - RandomRotation(90)

### Stage 2（MIL）

- feature dim: 768（若你换成 UNI 1024，就把配置改为 1024）
- max instances per bag: 512~1024
- dropout: 0.25
- lr: 1e-4
- loss: BCEWithLogitsLoss
- threshold: 验证集按每类 F1 搜索

---

## 五、把 timm backbone 换成 UNI

默认代码使用 `timm.create_model(model_name, pretrained=True, num_classes=0)`。

如果你已有 UNI 权重，可以改 `bach_mil/models/encoder.py`：

1. 替换 `build_backbone()`
2. 保持 `forward_features()` 返回 `[B, D]`
3. 在命令里把 `--feature_dim` 改成 `1024`

---

## 六、导出与后续加速

等精度稳定后再做：

1. patch encoder 导出 ONNX/OM
2. WSI 推理链路做 batching / pipeline overlap
3. AIPP / INT8 单独开分支评估
4. 最终把 Photos 4分类 + WSI 多标签 / slide 推理整合到一套入口

---

## 七、已知需要你根据本地情况调整的地方

1. **XML 标签名映射**：`build_wsi_manifest.py` 里默认通过关键词匹配标签名，如果你的 XML 用的不是 `Normal/Benign/InSitu/Invasive` 这些字符串，需要改映射表。
2. **OpenSlide 依赖**：Linux 环境一般要先装系统库。
3. **提特征倍率**：不同扫描倍率下，`level` 需要自己试。通常 level=1/2 比 level=0 更稳。
4. **Normal 多标签语义**：建议实际训练后看验证集再决定保留还是 fallback。

---

## 八、输出文件

训练和推理后会产出：

- `best.pt`：Stage 1 / Stage 2 最优权重
- `metrics.json`：训练日志与验证指标
- `thresholds.json`：WSI 每类阈值
- `test_photos_predictions.csv`
- `test_wsi_predictions.csv`

---

## 九、你现在最推荐的里程碑

### 里程碑 1（2~3 天）
- 跑通 Photos 4分类
- 跑通 A01~A10.xml → tile manifest
- 跑通训练 A01~A10 WSI 特征提取

### 里程碑 2（2~3 天）
- 跑通 WSI 多标签 MIL
- 画出 attention 热图
- 确定是否保留 `Normal` 作为 slide-level 标签

### 里程碑 3
- 测试集 WSI 推理
- 阈值搜索
- 导出最终 CSV
