# 华为昇腾 BACH 病例分类加速项目全过程记录

本文是本仓库的过程型长文档，保留了 `A1 / A2 / A3` 从纯推理基线、端到端吞吐优化、WSI 训练与聚合、离线模型导出，到最终提交收口的完整轨迹。

如果你只想快速确认“项目现在做到哪一步、当前正式口径是什么、结果是多少、该交哪些文件”，先看下面的“2026-03-25 终版总览”。

## 封面说明（2026-03-25 终版总览）

### 文档定位

- 本文档是“全过程记录”，不是只保留最终结论的精简版。
- 早期章节大量内容聚焦 `A1 / A2`，它们是理解后续 Ascend 加速策略的重要背景。
- 当前比赛正式提交主线已经切换为 `A3 = BACH WSI 多标签分类加速`。
- 若正文某处旧描述与最新收口口径冲突，以本节和第 `11` 章为准。

### 当前最终口径

- 赛题三最终按 `WSI 多标签分类` 理解，不按像素级分割提交。
- `before` 定义为：
  - 同一任务
  - 同一输入输出口径
  - 同一切块参数
  - 同一阈值文件
  - 同一聚合逻辑
  - `PyTorch eager` 编码器执行路径
- `after` 定义为：
  - 与 `before` 完全相同的任务定义和输出 schema
  - 当前正式主提交使用 `mixed OM(attn_score_path_norm1)`
  - 保守回退为 `OM(origin)`
  - 轻量化候选为 `5%` 剪枝 `OM(origin)`
- 本地评测口径固定为：
  - Photos：`5 折 ACC + macro_f1 + ovr_auc`
  - WSI：`5 折 exact_match(ACC代理) + macro_f1 + sample_f1 + per-class AP`
- 官方无标签测试集固定只做：
  - 真实推理速度测试
  - 最终预测结果导出
  - 特征文件导出
- `.onnx` 与 `.om` 已完成；`.air` 仍是可选补充项，不是当前阻塞项。

### 当前已经完成的闭环

- 已补齐统一推理入口：`src/A3/scripts/run_submission_infer.py`
- 已补齐离线模型导出与验证：
  - `ONNX`
  - `OM(origin)`
  - `mixed OM`
  - `5%` 剪枝 `ONNX/OM`
- 已打通 WSI 读图回退：
  - `openslide`
  - `_vendor/tiffslide`
- 已完成 WSI 官方测试集 `before/after` 同口径测速
- 已完成 WSI 本地代理评测
- 已完成 Photos 本地代理评测
- 已完成 Photos 统一入口 `OM` 烟雾验证
- 已生成 6 份阶段报告

### 最新关键数据

#### WSI 官方测试集速度

| 指标 | before | 当前正式 after(`mixed OM`) | prune5 候选(`OM origin`) |
| --- | ---: | ---: | ---: |
| 总耗时(秒) | 157.7303 | 25.2606 | 29.4291 |
| 平均每张 slide 耗时(秒) | 15.7730 | 2.5261 | 2.9429 |
| WSI/s | 0.06340 | 0.39587 | 0.33980 |
| tiles/s | 14.5185 | 90.6551 | 77.8141 |
| 总 tiles | 2290 | 2290 | 2290 |
| 相对 before 加速比 | - | 6.2441x | 5.3597x |
| 相对 before 耗时下降 | - | 83.9850% | 81.3421% |

补充说明：

- prune5 相对当前正式 `mixed OM` 仍慢 `16.5021%`
- prune5 与 mixed OM 在官方 10 张无标签测试 WSI 上最终 `pred_labels`：`7/10` 一致

#### WSI 本地代理评测

| 指标 | before | 当前正式 after(`mixed OM`) | prune5 候选(`OM origin`) |
| --- | ---: | ---: | ---: |
| exact_match(ACC代理) | 1.0000 | 1.0000 | 1.0000 |
| macro_f1 | 0.933333 | 0.933333 | 0.933333 |
| sample_f1 | 1.0000 | 1.0000 | 1.0000 |
| ACC 变化 | - | 0.0 个百分点 | 0.0 个百分点 |
| macro_f1 变化 | - | 0.0 个百分点 | 0.0 个百分点 |

#### Photos 本地代理评测

| 指标 | before | after(`OM origin`) |
| --- | ---: | ---: |
| ACC | 0.9200 | 0.9450 |
| macro_f1 | 0.919118 | 0.944514 |
| ovr_auc | 0.990958 | 0.996375 |
| ACC 变化 | - | +2.5 个百分点 |
| macro_f1 变化 | - | +2.5397 个百分点 |

#### 离线模型对齐

- WSI `ONNX`：
  - `feature_cosine_mean = 1.0`
  - `logit_cosine_mean = 1.0`
  - `prob_max_abs_diff = 8.642673492431641e-07`
- WSI `OM(origin)`：
  - `feature_cosine_mean = 0.999975323677063`
  - `logit_cosine_mean = 0.9999200105667114`
  - `prob_max_abs_diff = 0.004311233758926392`
- mixed OM `attn_score_path_norm1`：
  - `feature_cosine_mean = 0.9999683499336243`
  - `prob_max_abs_diff = 0.004738271236419678`
- prune5 `ONNX`：
  - `feature_cosine_mean = 1.0`
  - `prob_max_abs_diff = 7.152557373046875e-07`
- prune5 `OM(origin)`：
  - `feature_cosine_mean = 0.9999767541885376`
  - `logit_cosine_mean = 0.9999826550483704`
  - `prob_max_abs_diff = 0.006109356880187988`
- Photos `OM(origin)`：
  - `feature_cosine_mean = 0.9999548196792603`
  - `logit_cosine_mean = 0.9997665882110596`
  - `prob_max_abs_diff = 0.005356550216674805`
- 旧 `fp16 OM` 仍不作为正式交付方案；当前 `mixed_float16 OM(attn_score_path_norm1)` 已升级为正式主提交。

### 当前正式交付物

- 统一入口脚本：
  - `src/A3/scripts/run_submission_infer.py`
- 离线模型目录：
  - `logs/A3_output/submission_closure/offline_models/`
- 官方测速结果：
  - `logs/A3_output/submission_closure/official_runs/`
- 本地代理评测结果：
  - `logs/A3_output/submission_closure/proxy_eval/`
- 阶段报告目录：
  - `logs/A3_output/submission_closure/reports/`
- 最新收口过程说明：
  - 本文第 `11` 章

### 建议阅读顺序

如果你现在的目的不同，建议按下面顺序读：

1. 想看最终可交付方案：先看本节，再看 `README.md`
2. 想看 before/after 定义、离线模型和阶段报告：直接看本文第 `11` 章
3. 想看 A3 训练主线和 WSI 业务解释：看本文开头到第 `10` 章
4. 想追溯 A2 端到端吞吐优化思路：看第 `2` 到第 `15` 章早期部分

### 重要说明

- 以下正文继续保留原始历史记录，不主动删除旧实验描述。
- 早期章节中若出现“下一步要做 A3”或“OM 尚未打通”等旧状态，属于历史时点描述。
- 当前最终状态以本节与第 `11.5 ~ 11.10` 节为准。

## 历史正文（以下保留原始过程记录）

以下正文开始进入按时间保留的原始分析与执行记录；其中早期章节更偏 `A2`，第 `11` 章开始进入本轮 `A3` 最终收口。

## 0. 核心结论（你现在已经做到哪一步）

### 已完成（有实测闭环的数据）

- **A1 纯推理基线**：FP32 vs FP16(AMP) batch sweep（img224），并找到了 sweet spot（吞吐峰值）。
- **A2 端到端基线（BACH Photos）**：完成从磁盘读图→切块→预处理→H2D→NPU推理→落盘TSV的可复现实验链路。
- **A2 多轮优化（控制变量，tiles 固定 5400）**：
  - 引入 C1（预处理向量化）+ C2（CPU预取/流水）+ fast_grid（tilegen向量化），端到端吞吐显著提升；
  - 在 white_thr=1.01（不过滤）条件下，三版对比得到：
    - 总耗时明显下降
    - overall tile/s 接近翻倍
    - steady tile/s 显著提高
  - **诊断结论明确**：稳态仍主要受 CPU 侧 decode/tile/pre 限制，NPU 大多时间在等数据（sync_wait 低）。

### 已完成（部分尖端优化还没有完成）

- **CPU 侧**：workers/prefetch 最优组合 sweep（去掉 bs sweep）
- **decode 并行化**：跨图片 decode/tile 预取进一步减少 CPU 阻塞
- **global-mix batching**：跨图片/跨WSI混 batch，让 NPU 真正跑到 bs=96 的甜点值
- **后续冲刺**：ATC 静态 shape / AIPP / INT8(AMCT)（需要在 A2 真实瓶颈明确后再上）

### 现在的目标（步骤为A3）
赛题三应明确理解为 基于 WSI 的多标签分类任务，而不是像素级语义分割任务：最终输入是 WSI 大图，整体流程应为先进行组织区域/边缘检测，再对有效组织区域切 patch，将 patch 尺寸对齐到模型输入规格（如 ViT 的 224×224），提取 patch 特征后通过 MIL 进行 WSI 级聚合与分类；其中 400 张 tiff 图像应作为主要训练数据，分割染色图和局部标注更多用于辅助理解组织分布、分析 patch 判别效果，而不是最终提交结果。由于一张 WSI 内部可能同时存在正常、良性、原位癌、浸润癌等多种组织，因此最终输出不应按单标签四选一处理，也不应只按最高恶性级别或主导区域强行归为一个类别，而应按 多标签预测 理解：对每个类别分别输出独立分数，当某类别分数达到设定阈值时即可输出该标签，若多个类别同时达到阈值，则允许同时输出多个标签，（可以参考哈佛的 clam 项目），因此模型设计上更适合采用 sigmoid + 多标签损失（如 BCE）+ MIL 聚合 的方案。
模型训练和测试代码都放在：src/A3下

### A3 已跑通的“可复现闭环”（2026-03-14 更新，详细解释版）

这一节不是只告诉你“结果是多少”，而是告诉你：

- 这些数字分别是什么意思；
- 它们为什么重要；
- 当前到底选了哪条生产线；
- 每个文件夹里放的是什么；
- 项目现在卡在“能力不足”还是“整理不足”。

可以把 A3 理解成一句话：

> 我们已经把一条从 Photos 训练 patch encoder，到 WSI 提特征、做多标签聚合、输出最终测试 CSV 的完整链路跑通了，而且当前已经选出了一条推荐交付线。

### A3 先看结论：项目现在处于什么阶段

按当前仓库里的真实产物看：

- `A` 到 `G` 的 **WSI 主线** 已经基本跑完；
- `Photos` 这条线在训练和验证上已经很强；
- 现在主要问题不是“能不能跑”，而是“如何把当前最优路线冻结得更清楚、解释得更明白”。

一句更直白的话：

> 你现在不是“刚开始做 A3”，而是“已经把 A3 主线跑通，下一步该做 A4 的精修、整理和最终交付收口”。

### A3 的两条主线分别是什么

当前工程实际上有两条相关但职责不同的线：

1. **Photos 监督线**
   作用：训练一个强的 patch encoder。
   输入：400 张带单标签的 `Photos` 图像。
   输出：一个能够识别病理形态的 patch 分类器。
   这条线的核心价值不是直接交 Photos 结果，而是给后面的 WSI 提供高质量特征提取器。

2. **WSI 推理线**
   作用：把一整张 WSI 变成最终的多标签预测。
   输入：WSI 原图。
   步骤：裁 tile -> 去背景 -> 提特征 -> 聚合 -> 阈值化 -> 导出 slide 级结果。
   输出：最终的多标签 CSV。

### A3 当前最重要的产物是什么

如果你现在只记 7 个文件，优先记这 7 个：

- `logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1/cv_summary_mean_std.csv`
  这是 Photos 严格 5 折交叉验证的总成绩单。
- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
  这是当前生产线使用的 Stage1.5 权重。
- `logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1/extract_summary.json`
  这是测试集 WSI 提特征的统计摘要。
- `logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv`
  这是当前推荐聚合方法的交叉验证结果。
- `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
  这是最终多标签判定阈值。
- `logs/A3_output/G_phase/tileagg_test_wsi_pred_L1_s448_uniStage1p5_topk16_v1.csv`
  这是最终 WSI 测试集预测结果。
- `logs/A3_output/reports/final_wsi_pipeline_summary.json`
  这是当前整条生产线的摘要说明。

### 你最关心的第 1 组数字：Photos 5 折结果到底是什么意思

当前最强 Photos 结果来自：

- 目录：`logs/A3_output/B_phase/stage1_uni_large_cv5_ms512_1024_1536_v1`
- 模型：`vit_large_patch16_224`
- 本地权重：`assets/ckpts/UNI/pytorch_model.bin`
- 核心结构参数：
  - `--backbone_pool token`
  - `--backbone_init_values 1.0`

最终 5 折平均结果为：

- `accuracy = 0.9200`
- `macro_f1 = 0.9191`
- `ovr_auc = 0.9910`

下面逐个解释。

#### 1. accuracy = 0.9200 是什么

`accuracy` 就是最直观的“分类正确率”。

它的含义是：

- 把验证集里的每一张 Photos 图像都做一次 4 类分类；
- 看最终预测类别是否和真实类别一致；
- 正确的比例就是 `accuracy`。

`0.9200` 表示：

- 平均下来，100 张图里大约有 92 张分类正确；
- 从“单张图像 4 分类”的角度看，这是一个已经比较强的结果。

它适合回答的问题是：

- “模型总体上猜得准不准？”

它不适合单独回答的问题是：

- “四个类别是不是都学得一样好？”

因为如果某些类别更容易，`accuracy` 高也可能掩盖某个难类表现差的问题。

#### 2. macro F1 = 0.9191 是什么

`macro F1` 比 `accuracy` 更适合看“各类别是否均衡地学好了”。

它的计算逻辑是：

- 先分别计算四个类别各自的 F1；
- 再把这四个类别的 F1 简单平均；
- 每个类别权重相同。

F1 本身是 `precision` 和 `recall` 的折中：

- `precision`：你预测成某类的样本里，有多少是真的；
- `recall`：这个类真实存在的样本里，你抓到了多少；
- `F1`：兼顾“别乱报”和“别漏报”。

`macro_f1 = 0.9191` 的意义是：

- 不只是总体猜得准；
- 四个类别平均下来也比较均衡；
- 这比单看 `accuracy=0.9200` 更能说明模型不是“靠某一类特别强把平均数抬起来”。

为什么这里 `accuracy` 和 `macro F1` 很接近：

- 说明四类整体比较均衡；
- 也说明模型不是明显偏科。

#### 3. OVR AUC = 0.9910 是什么

`OVR AUC` 是 `one-vs-rest AUC`，意思是：

- 对每个类别，单独把它当作“正类”；
- 其他三个类合起来当作“负类”；
- 看模型给这个类别打分时，能否把正类排在负类前面；
- 四个类别做完之后，再汇总。

这个指标更关注：

- 模型输出的“分数排序能力”强不强；
- 而不是只看最终 argmax 分类对不对。

`0.9910` 很高，说明：

- 模型对四类的可分性非常强；
- 即使某些样本最终 argmax 还会出错，模型的分数层面其实已经能较好地区分不同类别。

这个指标对后续 WSI 有什么价值：

- WSI 后续要做 tile 级打分再聚合；
- 所以 patch encoder 的“分数排序能力”很重要；
- `OVR AUC` 高，说明它很适合给后续 tile 聚合提供概率先验。

#### 4. “5 折平均”又是什么意思

5 折交叉验证不是只训练一次，而是把 400 张 Photos 分成 5 份：

- 每次拿 4 份训练，1 份验证；
- 一共做 5 次；
- 每次换一份做验证；
- 最后把 5 次结果求平均。

这样做的好处是：

- 不容易被单次划分运气影响；
- 比“只看一个 fold0”更可信；
- 更适合作为“这个模型结构是否真的强”的判断依据。

所以这里的：

- `accuracy = 0.9200`
- `macro_f1 = 0.9191`
- `ovr_auc = 0.9910`

都不是单次碰巧的结果，而是 5 次验证的平均结果。

### 你最关心的第 2 组数字：mt20 / mt40 / mt60 到底差在哪

这三个名字不是三个模型，而是三种 **背景过滤强度**。

它们来自 WSI manifest 生成时的参数：

- `mt20` = `min_tissue = 0.2`
- `mt40` = `min_tissue = 0.4`
- `mt60` = `min_tissue = 0.6`

`min_tissue` 的含义是：

- 一块 tile 里至少要有多少比例的“组织区域”，才保留；
- 否则就当背景 tile 丢掉。

举例：

- `min_tissue = 0.2`
  只要求 20% 是组织，比较宽松；
  会保留更多 tile；
  但也会把更多边缘、空白附近、不够干净的 tile 留下来。
- `min_tissue = 0.4`
  要求 40% 是组织；
  比较平衡；
  既能去掉不少背景，又不会把太多有效 tile 删掉。
- `min_tissue = 0.6`
  要求 60% 是组织；
  更严格；
  留下的 tile 更“满”；
  但可能会误删一些边缘组织或小病灶区域。

当前实测总 tile 数是：

- `mt20` 总 tile 数 = `16197`
- `mt40` 总 tile 数 = `14281`
- `mt60` 总 tile 数 = `12620`

这三个数字为什么会越来越小：

- 因为过滤越来越严格；
- 背景和组织较少的 tile 会被越来越多地剔除；
- 所以最后留下来的 tile 总数自然下降。

这三个数字的本质区别不是“数字谁大谁好”，而是：

- `mt20`：样本最多，但噪声更多；
- `mt40`：样本量和背景过滤取得平衡；
- `mt60`：背景最少，但可能损失有效病灶信息。

当前为什么选 `mt40`：

- 比 `mt20` 更能抑制背景；
- 又不像 `mt60` 那么激进；
- 在样本量、纯度、后续训练可用性之间最均衡。

所以你以后看到：

- `wsi_train_tiles_mt40.csv`
- `stage1p5...mt40...`

不要把它理解成“第 40 轮实验”，而要理解成：

> 这条线使用的是 `min_tissue = 0.4` 的背景过滤版本。

### 你最关心的第 3 组词：`UNI-Large Stage1.5`、`L1`、`step=448`、`min_tissue=0.4` 到底是什么

这句话：

> 生产分支实际上已经选定：`UNI-Large Stage1.5`、`L1`、稀疏 `step=448`、`min_tissue=0.4`

不是一句抽象口号，而是在描述 **最终交付线的四个关键配置**。

#### 1. `UNI-Large Stage1`

意思是：

- 用 `UNI-Large / vit_large_patch16_224` 作为 patch encoder；
- 先在 Photos 上做 4 类监督训练；
- 让模型先学会病理形态的基础判别能力。

它回答的问题是：

- “patch encoder 用谁？”

当前答案是：

- 用 `UNI-Large`；
- 不再以早期 `ViT-Base` 线作为主线。

#### 2. `Stage1.5`

意思是：

- 不满足于“只在 Photos 上训完就直接上 WSI”；
- 而是再拿 WSI tile 做一次域对齐；
- 让 encoder 适应 WSI 的局部纹理、背景比例、组织形态分布。

它回答的问题是：

- “从 Photos 域迁移到 WSI 域时，要不要再做一步适配？”

当前答案是：

- 要做；
- 而且已经有实跑权重，当前生产线采用的是：
  - `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`

#### 3. `L1`

意思是：

- 读 WSI 时使用 `level = 1`；
- 不是最高分辨率的 `level = 0`；
- 也不是更低分辨率的 `level = 2`。

为什么 WSI 会有多个 level：

- `.svs` 往往是金字塔结构；
- 同一张切片会有多个分辨率版本；
- `level 0` 最清晰但最慢；
- level 越大，图越粗、速度越快。

当前选择 `L1` 的含义是：

- 在分辨率和速度之间找了平衡；
- 相比 `L0` 更实用；
- 相比 `L2` 又保留了更多细节。

#### 4. `step=448`

意思是：

- 在 WSI 上切 tile 时，不是每隔 224 像素切一块；
- 而是每隔 448 像素切一块；
- 所以采样更稀疏。

因为 `tile_size = 224`，所以：

- `step = 224` 表示紧密平铺；
- `step = 448` 表示每隔两个 tile 的距离再取一个。

这会带来什么：

- 优点：tile 总数明显下降，提特征更快；
- 缺点：覆盖不如密集采样全面，可能漏掉局部小区域；
- 但当前实验中，这个稀疏方案与更重的方案效果打平，所以选择更快的方案。

所以“稀疏 `step=448`”的本质是：

> 用更少的 tile 换接近不变的效果，从而提升整条生产线速度。

#### 5. `min_tissue=0.4`

这一点前面已经讲过，它表示：

- tile 至少 40% 是组织，才保留；
- 用来过滤背景。

所以整句话：

> `UNI-Large Stage1.5`、`L1`、稀疏 `step=448`、`min_tissue=0.4`

可以翻译成一句人话：

> 当前生产线采用的是：先用 UNI-Large 训练并做 WSI 域适配，然后在 WSI 的 level 1 上以较稀疏的方式切 tile，并且只保留组织比例至少 40% 的 tile。

### 你最关心的第 4 组词：`TileAgg topk16`、阈值、MIL 这些具体是什么意思

当前推荐生产线是：

- `UNI-Large Stage1`
- `UNI Stage1.5 (L1 sparse, s448, mt40)`
- `TileAgg(topk16)`

这条线对应的验证结果是：

- TileAgg `topk16` 交叉验证平均 `macro_f1 = 0.9333`

同时仓库里也有：

- `MIL` 5 折结果，平均 `macro_f1 = 0.9333`

这两个名字很像，但含义不同。

#### 1. TileAgg 是什么

`TileAgg` 是一种比较直接的 slide 级聚合方式。

它的思路是：

- 先让 patch encoder 给每个 tile 输出类别概率；
- 再把同一张 WSI 的 tile 概率聚合成 slide 级概率；
- 最后再根据阈值输出多标签。

也就是说：

- 它不额外训练一个复杂 MIL 网络；
- 而是直接利用 tile 概率做 slide 级判断。

你可以把它理解成：

> “先把每块 tile 判一遍，再把最有代表性的 tile 综合起来决定整张 slide 的标签。”

#### 2. `topk16` 是什么

`topk16` 表示：

- 对某个类别，比如 `Benign`；
- 把一张 WSI 里所有 tile 的 `Benign` 概率从高到低排序；
- 只取最高的 16 个 tile；
- 再对这 16 个概率做平均；
- 作为这张 WSI 在 `Benign` 类上的 slide 级分数。

为什么要这样做：

- 一张 WSI 里不是所有 tile 都有病灶；
- 真正有信息的往往只是少数高响应 tile；
- 如果对所有 tile 直接平均，病灶信号很容易被大量普通 tile 冲淡；
- 所以只看 top-k 往往更稳。

当前为什么是 `topk16`：

- 仓库里已经比较了 `topk8 / 12 / 16 / 24 / 32`；
- 当前主线把 `topk16` 选为推荐交付线；
- 因为它在效果和稳定性上已经足够好。

#### 3. `macro_f1 = 0.9333` 在 WSI 上是什么意思

这里的 `macro_f1 = 0.9333` 不再是 Photos 四分类，而是 **WSI 多标签任务** 的平均类别 F1。

它表示：

- 对 `Benign`、`InSitu`、`Invasive` 这几个病灶标签分别计算 F1；
- 再取平均。

这个值高说明：

- 对每个病灶标签，模型都比较能兼顾“别乱报”和“别漏报”；
- 在当前这 10 张有标注 WSI 的交叉验证中，TileAgg 的标签级判断表现很好。

但一定要注意：

- 这里的 WSI 训练标注 slide 只有 `10` 张；
- 这个分数说明“当前内部验证非常好”；
- 不等于“已经对外部数据绝对稳健”。

所以你以后看这个 `0.9333` 时，正确理解应当是：

> 这是一个很强的内部小样本结果，说明路线是通的，但不能把它当成大规模泛化的最终证明。

#### 4. 阈值 `Benign=0.1 / InSitu=0.1 / Invasive=0.5` 是什么

多标签任务和单标签任务最大的区别之一，是：

- 不是取最大值那个类就结束；
- 而是每个类别都要单独决定“是否激活”。

所以需要为每个类别设一个阈值。

比如：

- `Benign = 0.1`
  表示 slide 级 `Benign` 分数只要大于等于 `0.1`，就输出 `Benign` 标签。
- `InSitu = 0.1`
  表示 slide 级 `InSitu` 分数只要大于等于 `0.1`，就输出 `InSitu` 标签。
- `Invasive = 0.5`
  表示 `Invasive` 更严格，必须至少到 `0.5` 才输出。

为什么三个阈值不一样：

- 因为每类概率分布不同；
- 有些类天生分数偏低；
- 有些类更容易高分；
- 所以不能机械地全用 `0.5`。

当前阈值较低的含义是：

- 在当前训练数据上，`Benign` 和 `InSitu` 的 slide 级分数偏保守；
- 若坚持用 `0.5`，会漏报较多；
- 所以交叉验证自动搜索后给了更低阈值。

这也意味着：

- 阈值非常依赖当前数据分布；
- 后续若换聚合方式、换特征、换标签定义，阈值需要重新校准。

#### 5. 为什么 TileAgg 和 MIL 分数一样，却最终选 TileAgg

当前仓库里，`TileAgg` 和 `MIL` 的平均 `macro_f1` 都做到了 `0.9333`。

但最终更推荐 `TileAgg` 作为交付线，原因是：

- `TileAgg` 更简单；
- 更好解释；
- 对小样本更稳；
- 推理链路更直接；
- 与当前 `L1 sparse` 方案组合后速度更好。

而 `MIL` 当前更适合作为：

- 研究线；
- 对照线；
- 可解释性线；
- 后续再迭代的上限探索线。

也就是说：

- 不是 MIL 不行；
- 而是当前数据规模下，TileAgg 已经够强、够简单、够稳，所以更适合当生产线。

### 你最关心的第 5 组内容：F 阶段“注意力与可解释性”到底做了什么

F 阶段的核心不是再提升分数，而是回答：

- “模型到底看了哪一块组织？”
- “某个标签为什么被触发？”
- “结果是不是有病理上的可解释性？”

当前已有两个示例目录：

- `logs/A3_output/F_phase/attention_A01_fold0_L1s448_uniStage1p5_v1/`
- `logs/A3_output/F_phase/attn_L2_fold0_A08/`

里面你会看到几类文件。

#### 1. `.png` 叠加图

这类图通常是：

- 以 WSI 缩略图为底图；
- 把某一类的高响应区域用热图颜色叠上去；
- 让你直观看到模型“更关注哪里”。

比如：

- `attn_Benign.png`
- `attn_InSitu.png`
- `attn_Invasive.png`

它们的意义是：

- 对同一张 slide，分别展示模型对不同类别的关注区域。

#### 2. `.npy` 文件

这是原始注意力数组，不是给人直接看的图片。

它通常保存的是：

- 每个 tile 或每个位置的注意力权重；
- 供后续再可视化、统计分析、重画热图使用。

你可以把它理解成：

> “热图背后的原始数值矩阵”。

#### 3. `slide_attention_meta.json`

这是解释这些热图时最重要的说明文件之一。

它通常会保存：

- 对应的是哪张 slide；
- 使用的是哪个模型权重；
- 采用的 level / step / tile_size；
- 哪些类别被渲染；
- 可能还包括缩放、坐标、文件路径等信息。

作用是：

- 保证热图不是“看图说话”；
- 而是能追溯到具体模型和具体输入配置。

#### 4. 支持缩略图

例如：

- `gt_thumbnail_A08.png`
- `wsi_thumb_A08.png`

作用是：

- 把模型热图和原始 WSI 缩略图、标注缩略图放在一起对照；
- 方便肉眼检查模型关注区域是否和病灶标注大致一致。

为什么我说 F 阶段“部分完成”而不是“完全完成”：

- 因为作为“能力证明”，它已经能出图；
- 你已经能拿它分析具体案例；
- 但它还没有被整理成“对所有验证 slide 批量生成、批量汇总、自动评价”的标准报告流水线。

所以当前 F 阶段的正确理解是：

> 可解释性功能已经具备，但还没有完全产品化。

### 你最关心的第 6 组内容：G 阶段“最终推理与交付”到底交付了什么

G 阶段就是把前面所有训练、提特征、聚合、阈值这些步骤真正落到最终结果上。

当前关键文件是：

- `logs/A3_output/G_phase/tileagg_test_wsi_pred_L1_s448_uniStage1p5_topk16_v1.csv`

这个文件的含义是：

- 对测试集 10 张 WSI 全部做了最终推理；
- 每张 slide 都有一行结果；
- 结果已经是交付级 CSV，不是中间调试文件。

例如里面会有：

- `slide_id`
- `pred_labels`
- `n_tiles`
- `prob_Normal`
- `pred_Normal`
- `prob_Benign`
- `pred_Benign`
- `prob_InSitu`
- `pred_InSitu`
- `prob_Invasive`
- `pred_Invasive`

这些列分别是什么意思：

- `slide_id`
  是测试 slide 的编号，比如 `test1`。
- `pred_labels`
  是最终输出的多标签结果，比如 `Benign;InSitu;Invasive`。
- `n_tiles`
  是这张 slide 最终参与聚合的 tile 数量。
- `prob_xxx`
  是 slide 级的类别分数。
- `pred_xxx`
  是这个类别在阈值判断后是否被激活，`1` 表示输出该标签，`0` 表示不输出。

当前统计摘要是：

- `Benign;InSitu;Invasive = 6`
- `Invasive = 3`
- `Benign;InSitu = 1`

这个直方图是什么意思：

- 10 张测试 slide 里：
  - 6 张最终被判成三标签同时阳性；
  - 3 张只判成 `Invasive`；
  - 1 张判成 `Benign + InSitu`。

这不是“正确率”，只是：

- 对最终测试集输出标签组合的一个分布汇总；
- 目的是帮助你快速判断结果是不是极端单一，或者是否出现明显异常。

比如如果 10 张全都输出完全相同标签，那通常就值得警惕。

### 当前的 A-G 最好怎么理解

为了以后你自己看项目不再混乱，建议把 A-G 固定理解为下面这套定义：

- `A`
  数据角色锁定、XML 质检、fold 划分生成、设备和环境可运行性确认。
- `B`
  Photos 上训练 Stage1 patch encoder，并做交叉验证。
- `C`
  生成 WSI manifest，做 Stage1.5 域对齐，确定背景过滤和切 tile 配置。
- `D`
  用选定权重给训练集和测试集 WSI 提取 tile 特征。
- `E`
  做 WSI 聚合方法比较、阈值校准，并确定最终聚合策略。
- `F`
  输出注意力热图和其他可解释性产物。
- `G`
  输出最终测试集预测 CSV，形成可交付结果。

按这套定义看当前项目状态：

- WSI 主线的 `A -> G` 已完成；
- Photos 最终测试推理文件还没有作为标准交付物写入 `logs/A3_output/`；
- 所以项目当前更像是 “A3 主线已完成，准备进入 A4 收口”。

### 当前推荐生产线，用一句最通俗的话解释

当前真正选中的交付线不是一句抽象模型名，而是：

> 先用 UNI-Large 在 Photos 上把 patch 分类器训强，再用 WSI tile 做一次域适配，然后在测试 WSI 上用 `level=1`、较稀疏的 `step=448` 和 `min_tissue=0.4` 做切块，提特征后采用 `TileAgg topk16` 聚合，并使用自动校准得到的类别阈值输出最终多标签。

如果再压缩成最短记忆版本，就是：

- `UNI-Large Stage1`
- `UNI Stage1.5`
- `L1 + step448 + mt40`
- `TileAgg topk16`
- `thresholds = {Benign:0.1, InSitu:0.1, Invasive:0.5}`

### 这一段最应该记住的 6 个判断

1. `accuracy=0.9200` 表示 Photos 四分类总体正确率已经超过 92%。
2. `macro_f1=0.9191` 表示四类平均表现也很均衡，不是靠单个容易类别撑起来的。
3. `ovr_auc=0.9910` 表示 patch encoder 的类别分数排序能力很强，适合作为 WSI 后续聚合的特征基础。
4. `mt20 / mt40 / mt60` 是背景过滤强度，不是模型版本；`mt40` 代表当前选择的组织比例阈值平衡点。
5. `L1 sparse step=448 min_tissue=0.4` 的本质是“用更少、更干净的 tile，换取更快但不明显掉点的推理线”。
6. 当前最终交付线选的是 `TileAgg`，不是说 MIL 不行，而是 TileAgg 在当前小样本条件下更简单、更稳、更适合交付。

### 这部分读完后，你应该能回答的几个问题

如果你已经看懂上面这段，应该能自己回答下面这些问题：

- 为什么 Photos 的 `0.92` 不是偶然，而是 5 折平均？
- 为什么 `macro_f1` 比 `accuracy` 更值得一起看？
- 为什么 `mt40` 比 `mt20` 少 tile，但反而更适合作为生产线？
- 为什么 `step=448` 会更快？
- 为什么最终选 `TileAgg topk16` 而不是直接把 MIL 当唯一主线？
- 为什么 `Benign` 和 `InSitu` 的阈值不是 `0.5`？
- F 阶段热图到底是在做“解释”而不是“训练”？
- G 阶段的 CSV 为什么已经算交付物，而不是普通中间文件？

### 当前建议你接下来怎么用这个 readme

建议把这部分当成“总说明书”，而把后面的内容当成：

- 详细工作记录；
- 历史实验轨迹；
- 命令复现手册；
- 对照线保留区。

也就是说：

- 上面这一节负责让人看懂项目；
- 后面那些长日志负责保留工程细节和历史证据。

---

---

## 1. 目录结构与数据约定（非常重要：raw 只读）

### 1.1 原始数据（raw）只读、不动

你已下载 BACH 数据并按官方结构放置（建议保持不变）：

- `data/BACH/ICIAR2018_BACH_Challenge/`（含标签，开发/训练）
- `data/BACH/ICIAR2018_BACH_Challenge_TestDataset/`（无标签，测试/推理）

> **禁止**修改/移动/重命名 raw 文件夹下的任何图片或目录。
> 所有派生产物都写入 `data/BACH/derived/`、`logs/`、`src/`。

### 1.2 派生目录（derived）可随时删除重建

```bash
mkdir -p data/BACH/derived/{split,tiles,cache}
mkdir -p logs
```

---

## 2. A1 vs A2：口径对齐与为什么要分两类基准

### A1（纯推理）

**输入**：随机张量 `(bs, 3, 224, 224)`，隔离 IO/解码/预处理等开销

**指标**：

| 指标 | 说明 |
|------|------|
| Latency | ms/patch |
| Throughput | patch/s |
| speedup | 同 img_size 同 batch：$S_{same\_bs} = \frac{ms_{fp32}}{ms_{fp16}}$ |

**目的**：

- 找 batch sweet spot
- 验证 FP16/AMP 是否真的提升模型计算本体

### A2（端到端 e2e）

**输入**：真实图片 → 切 tile → 预处理 → H2D → 推理 → 统计吞吐

**指标**：

| 指标 | 说明 |
|------|------|
| tile/s | 端到端吞吐 |
| img/s | 每秒处理的图片张数 |
| 分段耗时 | decode / tilegen / preprocess / H2D / infer / sync_wait / total |

**目的**：

- 找到端到端瓶颈（CPU/IO/搬运/模型本体）
- 决定下一步优化优先级：AIPP/ATC/INT8 vs pipeline/CPU优化

---

## 3. BACH 数据准备：txt 文件列表（不改 raw）

为避免移动数据，A2 使用 txt 列表作为输入源。

### 3.1 生成训练/开发 Photos 列表

```bash
find data/BACH/ICIAR2018_BACH_Challenge/Photos -type f \
  \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.tif" -o -iname "*.tiff" \) \
  | sort > data/BACH/derived/split/photos_train_all.txt
```

### 3.2 生成测试 Photos 列表

```bash
find data/BACH/ICIAR2018_BACH_Challenge_TestDataset/Photos -type f \
  \( -iname "*.tif" -o -iname "*.tiff" \) \
  | sort > data/BACH/derived/split/photos_test_all.txt
```

---

## 4. 模型验证脚本（已跑通）：test_uni_npu.py

`src/test_uni_npu.py` 用于确认：

- 权重加载成功（`assets/ckpts/UNI/pytorch_model.bin`）
- NPU 可用（`npu:0`）
- 输入输出 shape 正确：输入 `(1,3,224,224)`，输出特征 `(1,1024)`（`num_classes=0`）

**关键点**：

```python
timm.create_model("vit_large_patch16_224", num_classes=0)
# normalize 使用 ImageNet mean/std
# torch.inference_mode() 推理
```

---

## 5. A2 端到端的关键概念与术语解释

### 5.1 e2e 是什么？

e2e = end-to-end（端到端）：从文件读入到推理输出的整条链路计时，而不是只测模型计算。

### 5.2 为什么第一张图常常很慢？

- NPU/框架存在一次性开销：图编译/缓存/算子选择/内存分配/AMP初始化等
- OS 文件缓存冷启动：第一次读磁盘更慢

因此必须同时报告两种口径：

| 口径 | 说明 |
|------|------|
| cold start | 首样本/首几张 |
| steady state | 跳过前 N 张的稳定态吞吐 |
| overall | 包含冷启动与全部开销的总吞吐 |

---

## 6. A2 优化路线（C1+C2+fast_grid 的演进过程）

### 6.1 初始 A2：逐 tile PIL 循环（慢）

**瓶颈在 CPU**：

- preprocess 非常慢（大量 Python/PIL 循环）
- NPU infer 很快但被"喂不饱"

### 6.2 C1：预处理向量化（batch 内处理）

把 per-tile 变为 per-batch：

```
(B,224,224,3) -> /255 -> normalize -> NCHW -> torch
```

大幅降低 Python 层循环开销

### 6.3 C2：预取/流水（prefetch + in-flight）

CPU 在 NPU 推理时准备下一批，尽可能隐藏 CPU 工作，减少 NPU 空等

### 6.4 fast_grid：tilegen 向量化（stride==tile）

对于 2048×1536、tile=stride=224：

- grid 固定 9×6=54 tiles/图
- 用 reshape/transpose 一次生成所有 tiles，避免 Python 双循环
- tilegen 时间下降明显

---

## 7. white_thr（白底过滤）与 tiles_total 变化的解释

### 7.1 为什么总 tiles 会变化（5374→5397）？

当 `white_thr<1` 开启过滤时：

- 某些 tile 被判为背景丢弃
- 导致每张 valid_tiles 可能从 54 变 53 等，累计 tiles_total 波动

### 7.2 如何控制变量做公平对比？

设 `white_thr=1.01`（等价不过滤）：

- 每张图固定 54 tiles
- 100 张图应固定 5400 tiles
- 这样对比性能最干净

**你已完成这一控制变量对比（5400 tiles 三版本对比）**。

---

## 8. 为什么 A2 的 bs 从 96 变成 32？（核心解释）

BACH Photos 单图 54 tiles：

- `bs=96` 实际永远跑不到 96，等价于 `bs=54`（一图一批）

C2 流水要有"下一批"才能重叠：

- `bs=54`：每图 1 batch → 无法预取下一批 → 重叠空间小
- `bs=32`：每图 2 batch（32+22）→ 能预取/重叠 → 端到端更快

**因此 bs 变小不是 NPU 不够，而是为了 pipeline 重叠。**

### 下一步如果要真正跑到 bs=96：

必须做 **global-mix batching**（跨图/跨WSI混 batch），让一个 batch 里包含多张图的 tiles，并用 ID 回填归属。

---

## 9. workers 能不能用到 24 核？正确姿势是什么？

可以加，但不一定线性变快：

- 受限于内存带宽、cache、线程调度、PIL 解码等
- 往往 8~16 出现收益拐点，继续加可能变慢

**正确做法**：sweep 找 CPU sweet spot

| 参数 | 建议范围 |
|------|----------|
| cpu_workers | 4, 8, 12, 16, 20 |
| prefetch | 2, 3, 4（按映射：4→2，8→3，>=12→4） |

此外：workers 主要加速 preprocess；decode 若仍在主线程，workers 再多 decode 也不会明显下降。因此下一步必须把 decode 并行化（跨图片预取 decode/tile）。

---

## 10. 你现在的三版 A2 对比（控制变量：tiles_total=5400）

你已生成对比表并得到典型结论：

| 指标 | 改善 |
|------|------|
| 总耗时 | 显著下降（~43%） |
| overall tile/s | 接近翻倍（~+94%） |
| steady tile/s | 明显提高（例如 ~162 tile/s） |
| pre | 从 ~435ms 大幅下降到 ~145ms（核心贡献） |
| sync_wait | 仍低（≈2%），说明端到端主要受 CPU 限制 |

---

## 11. 下一步要做什么（你提出的 sweep + decode 并行化 + global-mix）

### 11.1 CPU sweep（先找 CPU 最优）

**目标**：只扫 cpu_workers + prefetch，bs 固定（因为单图 54 tiles 时 bs≥54 无意义）。

你已要求去掉 bs 轮询：

- `run_a2_grid_sweep_fixed_bs.py`：固定一个 bs，只 sweep CPU 参数
- 输出 top 排名（steady tile/s）

### 11.2 decode 并行化（跨图片预取）

**目标**：减少 decode_wait，让主线程尽量不等待读图/切 tile。

**策略**：

- decode+tile 作为生产者线程池，维持一个 image-level prefetch 窗口
- 主线程消费已 decode 的 tiles，同时提交 NPU 推理
- CPU/NPU 重叠更充分，减少 NPU 空等

### 11.3 global-mix batching（让 bs=96 真正生效）

**目标**：让 NPU 吃饱

- 建一个全局 tile 池（每个 tile 带 img_id / wsi_id）
- 按 bs=96 从池中取 tiles 组成 batch
- 推理后按 ID 回填到对应图片/WSI 的结果结构中

---

## 12. 常用复核脚本（避免手工统计误差）

### 12.1 检查是否处理了 100 张图

```bash
awk 'NR>1{c++} END{print "num_images=",c}' logs/A2final_fp16_test_bs32.tsv
```

### 12.2 求和 tiles_total、检查是否全为 54

```python
import pandas as pd
df = pd.read_csv("logs/A2final_fp16_test_bs32.tsv", sep="\t")
print("images:", len(df))
print("sum_valid_tiles:", int(df["valid_tiles"].sum()))
print("min/max valid_tiles:", int(df["valid_tiles"].min()), int(df["valid_tiles"].max()))
print("count!=54:", int((df["valid_tiles"]!=54).sum()))
```

---

## 13. 环境说明（你当前配置）

| 硬件/软件 | 配置 |
|-----------|------|
| Ascend NPU | 1 × Ascend 910B (snt9b2) |
| CPU | ARM 24 核 |
| 内存 | 192GB |
| 软件栈 | torch / torch_npu / timm（版本以实际环境为准） |

> **重要工程原则**：长任务必须有进度输出，结果实时落盘TSV，保证可复现。

---

## 14. FAQ（对话中反复出现的关键疑问）

### Q1：warmup 也要时间，真实评测算不算？

你无法提前保证评测脚本是否 warmup。正确做法是同时报告：

- **overall**：包含 warmup/冷启动/全部开销
- **steady**：跳过前 N 张

并在文档里解释冷启动来源与处理方式。

### Q2：为什么 NPU 很强但吞吐还受 CPU 限制？

因为端到端开销主要在 CPU：

- decode / tilegen / preprocess

NPU 的 infer 很快，但大部分时间在等数据（sync_wait 低）。

### Q3：bs=96 最优为什么 A2 用 32？

单图只有 54 tiles，`bs=96` 实际等价 54；`bs=32` 能形成两批并触发流水重叠，因此端到端更快。真正想用 bs=96 必须 global-mix。

---

## 15. 接下来你可以直接执行的任务清单（按优先级）

1. **固定 white_thr=1.01（不滤）** 继续做 CPU sweep（workers/prefetch）
2. 在 A2 中加入跨图片 decode 并行化（减少 decode_wait）
3. 得到 CPU 最优配置后，进入 global-mix batching，实现真正 bs=96
4. **CPU/NPU pipeline 吃饱后**，再考虑 ATC / AIPP / INT8（否则收益兑现不了）

---
# 以下是工作记录（一直往下追加即可）

从这里开始：
## Changelog

### 2026-03-03 22:20 (UTC+8) — A2 v3 bs=96 实测
- 标签：Speed
- 本轮目标：验证 bs96 + global_mix=1 的端到端性能
- 本轮改动：bs=96, global_mix=1, cpu_workers=8, prefetch=3
- 命令：`python src/A2_final_opt_v3.py --file_list data/BACH/derived/split/photos_test_all.txt --ckpt_dir assets/ckpts/UNI --tile 224 --stride 224 --bs 96 --global_mix 1 --precision fp16 --white_thr 1.01 --cpu_workers 8 --prefetch 3 --decode_prefetch 4 --warmup_mode async --warmup_iters 3 --skip_steady 1 --log_every 10 --out logs/A2v3_fp16_bs96_gmix.tsv`
- 结果：steady_tile/s=400.789, steady_img/s=7.422, warmup_sync_ms=5666.33
- 结论：bs96 相比 bs32 几乎无提升（+0.5%），可能已 GPU bound
- 下一步：1) decode 并行化减少 decode_wait; 2) 尝试更高 workers

### 2026-03-03 22:25 (UTC+8) — A2 v3 CPU Sweep 再跑
- 标签：Speed
- 本轮目标：再次 sweep cpu_workers + prefetch 验证最优配置
- 本轮改动：固定 bs=96, global_mix=1, sweep w=4,8,12,16,20
- 命令：`python src/run_a2_grid_sweep_V2.py --a2_script src/A2_final_opt_v3.py --file_list data/BACH/derived/split/photos_test_all.txt --ckpt_dir assets/ckpts/UNI --tile 224 --stride 224 --precision fp16 --white_thr 1.01 --bs 96 --global_mix 1 --warmup_mode async --warmup_iters 3 --skip_steady 1 --decode_prefetch 4 --log_every 10 --repeat 3`
- 结果：最佳 w=12 pf=4, steady_tile/s=494.37±9.44 (overall=443.65)
- 结论：w=12 最优，比 w=16 提升 ~4.8%，sync_wait 增加到 34ms（NPU 吃得更多）
- 下一步：1) decode 并行化减少 decode_wait; 2) 尝试更细粒度 sweep

### 2026-03-03 22:30 (UTC+8) — A2 gmix bs96 最佳配置实测
- 标签：Speed
- 本轮目标：使用 sweep 得到的最佳配置 w=12, pf=4 验证性能
- 本轮改动：bs=96, global_mix=1, cpu_workers=12, prefetch=4
- 命令：`python src/A2_final_opt_v3.py --file_list data/BACH/derived/split/photos_test_all.txt --ckpt_dir assets/ckpts/UNI --bs 96 --global_mix 1 --cpu_workers 12 --prefetch 4 --decode_prefetch 4 --white_thr 1.01 --precision fp16 --out logs/A2_gmix_bs96.tsv --summary_json logs/A2_gmix_bs96.json`
- 结果：steady_tile/s=429.003, steady_img/s=7.944, warmup_sync_ms=5283.74
- 结论：单次运行略低于 sweep 均值（494），波动正常，sync_wait=30.5ms（NPU 有吃饱）
- 下一步：1) decode 并行化减少 decode_wait; 2) 尝试更细粒度 sweep

### 2026-03-05 23:00 (UTC+8) — A2 Grid Sweep V3 完成
- 标签：Speed
- 本轮目标：二维 sweep workers × prefetch，找到 CPU 最优配置
- 本轮改动：固定 bs=96, global_mix=1, skip_steady=3, sweep workers=4,6,8,10,12,14,16 × prefetch=3,4,5,6, repeat=5
- 命令：`python src/run_a2_grid_sweep_V3.py --a2_script src/A2_final_opt_v3.py --file_list data/BACH/derived/split/photos_test_all.txt --ckpt_dir assets/ckpts/UNI --bs 96 --global_mix 1 --precision fp16 --white_thr 1.01 --warmup_mode async --warmup_iters 3 --decode_prefetch 4 --worker_list 4,6,8,10,12,14,16 --prefetch_list 3,4,5,6 --repeat 5 --skip_steady 3 --round_rest 10`
- 结果：最佳配置 w=4,pf=5 (中位数 585.7，均值 529.0±81.9)，总运行 140/140 成功，耗时 138 分钟
- 结论：workers=4 最优而非更高 workers，prefetch=5/6 最佳，sync_wait=27.8ms（NPU 吃饱），相比上次 w=16 最优（471）提升 ~24%
- 下一步：1) 探索更低的 workers (2,3)；2) decode 并行化

### 2026-03-07 (UTC+8) — A2 final_opt_v3 skip_steady=1 实测
- 标签：Speed
- 本轮目标：验证 skip_steady=1 的稳态性能
- 本轮改动：bs=96, global_mix=1, cpu_workers=6, prefetch=4, skip_steady=1
- 命令：`python src/A2_final_opt_v3.py --file_list data/BACH/derived/split/photos_test_all.txt --ckpt_dir assets/ckpts/UNI --bs 96 --global_mix 1 --cpu_workers 6 --prefetch 4 --decode_prefetch 4 --white_thr 1.01 --precision fp16 --warmup_mode async --warmup_iters 3 --skip_steady 1 --out logs/A2_final_baseline.tsv --summary_json logs/A2_final_baseline.json`
- 结果：steady_tile/s=432.224, steady_img/s=8.004, mean_pre_ms=194.07, mean_tile_compute_ms=3.88
- 结论：skip_steady=1 时稳态吞吐量 432 tile/s，比 skip_steady=3 的 392 提升 ~10%
- 下一步：验证原始baseline数据来源

### 2026-03-12 (UTC+8) — A3 阶段A/B/C 执行与迭代完成
- 标签：Accuracy / Pipeline
- 本轮目标：完成 A3 阶段 A（数据与环境基线）+ B（Photos Stage1）+ C（WSI manifest 与 Stage1.5）的实跑闭环，并输出可复现报告与产物。
- 约束执行：
  - 未修改原始数据目录：`data/BACH/ICIAR2018_BACH_Challenge`、`data/BACH/ICIAR2018_BACH_Challenge_TestDataset`
  - 所有结果统一写入：`logs/A3_output/`
  - 索引文件写入：`data/BACH/derived/split/`
- 关键代码修复（`src/A3`）：
  - 修复 XML 解析兼容性，支持 `Region/Vertex + Attribute@Value`（Aperio）与 `Annotation/Coordinate`（QuPath）
  - 新增设备自动选择：`cuda > npu > cpu`，确保 NPU 环境可直接运行
  - 修复 Stage1.5 混合数据集 `KeyError: slide_id`（为 WSI tiles 增加字段适配器）
  - 调整 Stage1 默认离线行为：非显式 `--pretrained` 不触发联网下载
- 阶段A结果：
  - `photos_folds.csv` 已生成并用于后续训练
  - XML 解析有效：A01~A10 共 226 个 polygon（Benign 57 / InSitu 60 / Invasive 109）
  - 产物：`logs/A3_output/A_phase/`
- 阶段B结果（fold0）：
  - smoke 对照：`baseline` 略优于 `tissue_crop`
  - 最终训练（12 epoch）最佳：`macro_f1=0.466571`（epoch 9）
  - 采用权重：`logs/A3_output/B_phase/stage1_final_fold0/best.pt`
- 阶段C结果：
  - `min_tissue` 三组 manifest 已实测并比较：`mt20/mt40/mt60`
  - 选型：`mt40`（背景过滤与样本量折中）
  - Stage1.5 域对齐两轮迭代（r1/r2）均低于阶段B主权重，判定当前不采用 Stage1.5 权重
  - 继续后续阶段建议使用阶段B主权重 + mt40索引
- 索引落盘（供后续 Stage2/3 直接使用）：
  - `data/BACH/derived/split/wsi_train_tiles_mt40.csv`
  - `data/BACH/derived/split/wsi_train_bags_mt40.csv`
- 完整报告与汇总：
  - 主报告：`logs/A3_output/reports/A3_ABC_execution_report.md`
  - 训练汇总：`logs/A3_output/reports/training_runs_summary.csv`








## A3 阶段 A/B/C 全流程执行报告（详细版）
以下内容与 `logs/A3_output/reports/A3_ABC_execution_report_full.md` 一致。
时间：2026-03-12

## 1. 范围与约束
目的：在不改动原始数据目录的前提下，完成 A3 阶段 A/B/C 的可复现实跑与迭代，并给出清晰结论与可复现路径。
允许写入目录：`src/A3`、`logs/A3_output`、`data/BACH/derived/split`
禁止修改目录：`data/BACH/ICIAR2018_BACH_Challenge`、`data/BACH/ICIAR2018_BACH_Challenge_TestDataset`
Python 版本：3.10（未更改）

## 2. 数据来源、目录与用途
### 2.1 数据来源与用途
Photos 训练集：`data/BACH/ICIAR2018_BACH_Challenge/Photos`
用途：Stage1 主监督，训练 patch encoder。

WSI 训练集：`data/BACH/ICIAR2018_BACH_Challenge/WSI` 中的 `A01~A10.svs` + `A01~A10.xml`
用途：Stage1.5 域对齐与 Stage2/3 的监督桥梁。

WSI 可选集：`data/BACH/ICIAR2018_BACH_Challenge/WSI` 中的 `01~20.svs`
用途：本轮不使用，留到第二版做自监督/伪标签/难负样本。

Photos 测试集：`data/BACH/ICIAR2018_BACH_Challenge_TestDataset/Photos`
用途：后续 Photos 推理输出（A/B/C 未执行）。

WSI 测试集：`data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI`
用途：后续 WSI 推理（A/B/C 未执行）。

### 2.2 数据规模快照（来自 Phase A 统计）
- Photos 训练总数：400（四类各 100/100/100/100）
- Photos 测试总数：100
- WSI A01~A10 数量：10
- WSI 01~20 数量：20
- WSI 测试数量：10
- A01 WSI 尺寸：62625×42113，level 数：3

### 2.3 WSI 染色缩略图颜色含义（BACH 官方含义）
结论：BACH WSI 的标注颜色通常为 Benign=红色、InSitu=绿色、Invasive=蓝色。Normal 为未标注背景区域（标签值 0）。
说明：官方论文描述 WSI 标注用 XML 多边形给出，评测 mask 用 0/1/2/3 分别表示 Normal/Benign/InSitu/Invasive，并在图示中将 Benign/InSitu/Invasive 分别标为红/绿/蓝色。
用途：`WSI/gt_thumbnails/A01.png` 等为可视化缩略图，颜色只用于展示标注区域，而非训练输入。

## 3. 模型与权重来源说明
Stage1/Stage1.5 使用 `timm` 的 `vit_base_patch16_224`。
模型结构：ViT backbone + 线性分类头，`num_classes=4`。
权重策略：默认不下载预训练权重。只有显式传 `--pretrained` 才会触发在线下载。
本次运行：未使用在线预训练权重（离线环境）。

## 4. 阶段 A：数据角色固定 + 环境可执行化
### A-01 环境体检
目的：确认训练/推理依赖可用。
方案：检查 Python 版本、torch/timm、NPU、OpenSlide。
执行：`python --version` 与导入检查脚本（torch、torch_npu、openslide）。
结果：NPU 可用，OpenSlide 初始不可用（阻断 WSI）。

### A-02 OpenSlide 安装与验证
目的：解锁 WSI 读入。
执行：`pip install openslide-python openslide-bin`
结果：依赖已满足，OpenSlide 可导入并能打开 `A01.svs`。

### A-03 XML 结构探测
目的：确认 `A01~A10.xml` 能解析为监督多边形。
方案：用现有解析器读取 A01.xml。
执行：Python 小脚本调用 `parse_xml_polygons`。
结果：解析为 0 polygons，提示 XML 格式不匹配。

### A-04 XML 解析器兼容增强
目的：支持 Aperio 与 QuPath 两种 XML。
修改文件：`src/A3/bach_mil/data/xml_utils.py`
修改内容：新增 `Region/Vertex + Attributes/Attribute@Value` 解析逻辑，并保留原 `Annotation/Coordinate` 解析。
结果：A01~A10 总多边形数 = 226；分类分布 Benign=57，InSitu=60，Invasive=109。

### A-05 NPU 设备选择修复
目的：避免脚本只认 CUDA 导致落到 CPU。
新增文件：`src/A3/bach_mil/utils/device.py`
接入脚本：
`src/A3/scripts/train_patch_stage1.py`
`src/A3/scripts/extract_bag_features.py`
`src/A3/scripts/train_mil_stage2.py`
`src/A3/scripts/infer_wsi.py`
`src/A3/scripts/infer_photos.py`
结果：优先 `cuda`，其次 `npu`，否则 `cpu`。

### A-06 生成 Photos 交叉验证划分
目的：固定 Stage1 训练口径与可复现折分。
执行：
```
python src/A3/scripts/prepare_photo_splits.py   --data_root data/BACH   --out_dir data/BACH/derived/split   --n_splits 5   --seed 3407
```
产物：
`data/BACH/derived/split/photos_folds.csv`
`data/BACH/derived/split/photos_train_all.txt`
`data/BACH/derived/split/photos_test_all.txt`

### A-07 阶段 A 统计固化
目的：为后续训练提供基线快照。
方案：通过一次性 Python 脚本统计数据量、XML 多边形与折分分布。
产物：
`logs/A3_output/A_phase/phaseA_summary.json`
`logs/A3_output/A_phase/xml_polygon_qc.csv`
`logs/A3_output/A_phase/photos_fold_distribution.csv`

## 5. 阶段 B：Stage1 Photos 训练与对照
### B-01 Smoke 对照（baseline vs tissue_crop）
目的：小成本筛掉不佳配置。
方案：fold0，3 epochs，对比 `baseline` 与 `--tissue_crop`。
执行：
```
python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/B_phase/stage1_smoke_base   --model_name vit_base_patch16_224   --input_size 224   --epochs 3   --batch_size 24   --lr 3e-4   --encoder_lr 1e-5

python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/B_phase/stage1_smoke_tcrop   --model_name vit_base_patch16_224   --input_size 224   --epochs 3   --batch_size 24   --lr 3e-4   --encoder_lr 1e-5   --tissue_crop
```
结论：baseline macro_f1 略高，作为主配置。

### B-02 预训练下载阻断修复
问题：timm 默认联网拉权重，离线环境超时。
修复：默认不下载权重，只有显式传 `--pretrained` 才下载。
修改文件：`src/A3/scripts/train_patch_stage1.py`

### B-03 Stage1 正式训练
目的：得到可用于后续 WSI 的稳定 encoder。
方案：fold0，12 epochs。
说明：这是不使用任何预训练（不联网）的从零训练 baseline，用于对照。
执行：
```
python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/B_phase/stage1_final_fold0   --model_name vit_base_patch16_224   --input_size 224   --epochs 12   --batch_size 24   --lr 3e-4   --encoder_lr 1e-5
```
结果：best macro_f1=0.466571（epoch 9）。
产物：
`logs/A3_output/B_phase/stage1_final_fold0/best.pt`
`logs/A3_output/B_phase/stage1_final_fold0/metrics.json`

### B-04 Stage1 使用本地 `model.safetensors` 初始化（推荐：显著提升精度）
目的：用你提供的 `assets/ckpts/hf/hub/model.safetensors` 作为 ViT backbone 初始化，再做 Photos 四分类训练，看 `macro_f1` 是否显著超过 baseline。

说明：
- `model.safetensors` 是 backbone 预训练权重，不是 BACH 四分类已训好的模型，所以仍需要 Stage1 监督训练（至少训练分类头）。
- 该方式完全离线，不需要 `timm --pretrained` 联网下载。

执行（先 smoke 1 epoch，确认能加载 + 能训练）：
```
python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/B_phase/stage1_safetensors_smoke_fold0   --model_name vit_base_patch16_224   --input_size 224   --epochs 1   --batch_size 24   --lr 3e-4   --encoder_lr 1e-5   --init_backbone_safetensors assets/ckpts/hf/hub/model.safetensors
```
结果（smoke）：best macro_f1=0.443899（epoch 1）。

执行（正式 12 epoch）：
```
python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/B_phase/stage1_safetensors_final_fold0   --model_name vit_base_patch16_224   --input_size 224   --epochs 12   --batch_size 24   --lr 3e-4   --encoder_lr 1e-5   --init_backbone_safetensors assets/ckpts/hf/hub/model.safetensors
```
结果（正式）：best macro_f1=0.838995（epoch 12），显著超过 baseline 0.466571。

产物（正式训练目录）：
`logs/A3_output/B_phase/stage1_safetensors_final_fold0/best.pt`
`logs/A3_output/B_phase/stage1_safetensors_final_fold0/metrics.json`
`logs/A3_output/B_phase/stage1_safetensors_final_fold0/init_backbone_load_report.json`

### B-05 Stage1 训练结果汇总
| run | epochs | best_epoch | best_acc | best_macro_f1 | best_ovr_auc | best_train_loss |
|---|---|---|---|---|---|---|
| B_final_fold0 | 12 | 9 | 0.5125 | 0.466571 | 0.689375 | 1.214420 |
| B_safetensors_smoke_fold0 | 1 | 1 | 0.4500 | 0.443899 | 0.776875 | 1.245003 |
| B_safetensors_final_fold0 | 12 | 12 | 0.8375 | 0.838995 | 0.964792 | 0.083136 |
| B_smoke_base | 3 | 2 | 0.4125 | 0.326087 | 0.679583 | 1.466852 |
| B_smoke_tcrop | 3 | 2 | 0.4000 | 0.322651 | 0.683750 | 1.464701 |
| C_stage1p5_r1 | 6 | 6 | 0.4125 | 0.378603 | 0.687292 | 0.783705 |
| C_stage1p5_r2 | 3 | 3 | 0.2625 | 0.125000 | 0.640208 | 1.293671 |

## 6. 阶段 C：WSI manifest + Stage1.5 域对齐
### C-01 构建三组 manifest（min_tissue 对比）
目的：比较背景过滤强度，选取合理的 tile 数量。
方案：固定 level=1，tile_size=224，step=224，分别设置 min_tissue=0.2/0.4/0.6。
执行：
```
python src/A3/scripts/build_wsi_manifest.py   --wsi_dir data/BACH/ICIAR2018_BACH_Challenge/WSI   --out_csv logs/A3_output/C_phase/manifests/mt20/wsi_train_tiles.csv   --out_bag_csv logs/A3_output/C_phase/manifests/mt20/wsi_train_bags.csv   --level 1 --tile_size 224 --step 224 --min_tissue 0.2

python src/A3/scripts/build_wsi_manifest.py   --wsi_dir data/BACH/ICIAR2018_BACH_Challenge/WSI   --out_csv logs/A3_output/C_phase/manifests/mt40/wsi_train_tiles.csv   --out_bag_csv logs/A3_output/C_phase/manifests/mt40/wsi_train_bags.csv   --level 1 --tile_size 224 --step 224 --min_tissue 0.4

python src/A3/scripts/build_wsi_manifest.py   --wsi_dir data/BACH/ICIAR2018_BACH_Challenge/WSI   --out_csv logs/A3_output/C_phase/manifests/mt60/wsi_train_tiles.csv   --out_bag_csv logs/A3_output/C_phase/manifests/mt60/wsi_train_bags.csv   --level 1 --tile_size 224 --step 224 --min_tissue 0.6
```

### C-02 manifest 质量统计与选择
结果统计：
| config | tiles_total | bags_total | tile_Normal | tile_Benign | tile_InSitu | tile_Invasive | tiles_min_per_slide | tiles_max_per_slide |
|---|---|---|---|---|---|---|---|---|
| mt20 | 16197 | 10 | 8069 | 907 | 404 | 6817 | 889 | 2908 |
| mt40 | 14281 | 10 | 6399 | 832 | 388 | 6662 | 712 | 2833 |
| mt60 | 12620 | 10 | 5147 | 715 | 375 | 6383 | 559 | 2684 |
结论：选择 `mt40`，在背景抑制与实例数量之间更平衡。
产物：`logs/A3_output/C_phase/manifest_choice.json`

### C-03 Stage1.5 训练样本采样（r1 / r2）
目的：控制 extra tiles 类别不平衡。
方案：对 Normal/Invasive 做截断采样。

r1 采样统计：
| label | count |
|---|---|
| Invasive | 2000 |
| Normal | 2000 |
| Benign | 832 |
| InSitu | 388 |

r2 采样统计：
| label | count |
|---|---|
| Invasive | 1000 |
| Normal | 1000 |
| Benign | 832 |
| InSitu | 388 |

### C-04 Stage1.5 训练（r1）
目的：域对齐，缓解 Photos 与 WSI tile 差异。
执行：
```
python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/C_phase/stage1p5_fold0_r1   --model_name vit_base_patch16_224   --input_size 224   --epochs 6   --batch_size 24   --lr 1e-4   --encoder_lr 5e-6   --init_ckpt logs/A3_output/B_phase/stage1_final_fold0/best.pt   --extra_tiles_csv logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled.csv
```
结果：best macro_f1=0.378603（低于 Stage1）。

### C-05 Stage1.5 训练（r2 更保守）
目的：验证是否因微调过强导致退化。
执行：
```
python src/A3/scripts/train_patch_stage1.py   --data_root data/BACH   --split_csv data/BACH/derived/split/photos_folds.csv   --fold 0   --out_dir logs/A3_output/C_phase/stage1p5_fold0_r2   --model_name vit_base_patch16_224   --input_size 224   --epochs 3   --batch_size 24   --lr 5e-5   --encoder_lr 1e-6   --freeze_encoder_epochs 2   --init_ckpt logs/A3_output/B_phase/stage1_final_fold0/best.pt   --extra_tiles_csv logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_light.csv
```
结果：best macro_f1=0.125（更差）。

### C-06 阶段 C 结论
- Stage1.5 在当前设置下对 Photos 验证指标产生负迁移。
- 不采用 Stage1.5 权重，后续继续使用 Stage1 主权重。

## 7. 输出文件与用途
本次共输出 37 个文件到 `logs/A3_output`，另有 2 个索引文件写入 `data/BACH/derived/split`。

### 7.1 logs/A3_output 文件清单
- logs/A3_output/A_phase/phaseA_summary.json
- logs/A3_output/A_phase/photos_fold_distribution.csv
- logs/A3_output/A_phase/xml_polygon_qc.csv
- logs/A3_output/B_phase/stage1_final_fold0/best.pt
- logs/A3_output/B_phase/stage1_final_fold0/metrics.json
- logs/A3_output/B_phase/stage1_safetensors_final_fold0/best.pt
- logs/A3_output/B_phase/stage1_safetensors_final_fold0/init_backbone_load_report.json
- logs/A3_output/B_phase/stage1_safetensors_final_fold0/metrics.json
- logs/A3_output/B_phase/stage1_safetensors_smoke_fold0/best.pt
- logs/A3_output/B_phase/stage1_safetensors_smoke_fold0/init_backbone_load_report.json
- logs/A3_output/B_phase/stage1_safetensors_smoke_fold0/metrics.json
- logs/A3_output/B_phase/stage1_smoke_base/best.pt
- logs/A3_output/B_phase/stage1_smoke_base/metrics.json
- logs/A3_output/B_phase/stage1_smoke_tcrop/best.pt
- logs/A3_output/B_phase/stage1_smoke_tcrop/metrics.json
- logs/A3_output/C_phase/manifest_choice.json
- logs/A3_output/C_phase/manifest_qc_summary.csv
- logs/A3_output/C_phase/manifests/mt20/wsi_train_bags.csv
- logs/A3_output/C_phase/manifests/mt20/wsi_train_tiles.csv
- logs/A3_output/C_phase/manifests/mt40/wsi_train_bags.csv
- logs/A3_output/C_phase/manifests/mt40/wsi_train_tiles.csv
- logs/A3_output/C_phase/manifests/mt60/wsi_train_bags.csv
- logs/A3_output/C_phase/manifests/mt60/wsi_train_tiles.csv
- logs/A3_output/C_phase/mt20_tiles_per_slide.csv
- logs/A3_output/C_phase/mt40_tiles_per_slide.csv
- logs/A3_output/C_phase/mt60_tiles_per_slide.csv
- logs/A3_output/C_phase/stage1p5_fold0_r1/best.pt
- logs/A3_output/C_phase/stage1p5_fold0_r1/metrics.json
- logs/A3_output/C_phase/stage1p5_fold0_r2/best.pt
- logs/A3_output/C_phase/stage1p5_fold0_r2/metrics.json
- logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled.csv
- logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_light.csv
- logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_light_qc.csv
- logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_qc.csv
- logs/A3_output/reports/A3_ABC_execution_report.md
- logs/A3_output/reports/A3_ABC_execution_report_full.md
- logs/A3_output/reports/training_runs_summary.csv

### 7.2 关键输出用途说明
| 文件 | 含义 | 内容 | 用法 |
|---|---|---|---|
| logs/A3_output/A_phase/phaseA_summary.json | 阶段A统计 | 数据规模、设备、XML解析统计 | 验证环境与数据基线 |
| logs/A3_output/A_phase/xml_polygon_qc.csv | XML 解析质量 | 每张 Axx 的多边形数量统计 | 判断标注是否可用 |
| logs/A3_output/A_phase/photos_fold_distribution.csv | 折分统计 | fold × label 计数 | 确认折分均衡 |
| logs/A3_output/B_phase/stage1_smoke_base/best.pt | Smoke baseline 权重 | Stage1 模型权重 | 对照实验 |
| logs/A3_output/B_phase/stage1_smoke_base/metrics.json | Smoke baseline 指标 | 训练曲线与指标 | 对照实验 |
| logs/A3_output/B_phase/stage1_smoke_tcrop/best.pt | Smoke tissue_crop 权重 | Stage1 模型权重 | 对照实验 |
| logs/A3_output/B_phase/stage1_smoke_tcrop/metrics.json | Smoke tissue_crop 指标 | 训练曲线与指标 | 对照实验 |
| logs/A3_output/B_phase/stage1_final_fold0/best.pt | Stage1 baseline 权重 | 从零训练的对照权重 | 对照实验/回退备份 |
| logs/A3_output/B_phase/stage1_final_fold0/metrics.json | Stage1 baseline 指标 | 训练曲线与指标 | 对照实验/回退备份 |
| logs/A3_output/B_phase/stage1_safetensors_final_fold0/best.pt | Stage1 推荐权重 | 本地 `model.safetensors` 初始化后训练的最优权重 | 后续 Stage2/3 推荐入口 |
| logs/A3_output/B_phase/stage1_safetensors_final_fold0/metrics.json | Stage1 推荐指标 | 训练曲线与指标 | 记录主结果 |
| logs/A3_output/B_phase/stage1_safetensors_final_fold0/init_backbone_load_report.json | backbone 加载报告 | missing/unexpected keys 统计 | 自检“是否确实加载” |
| logs/A3_output/C_phase/manifest_choice.json | manifest 选择记录 | 选用 mt40 的理由 | 复盘依据 |
| logs/A3_output/C_phase/manifest_qc_summary.csv | manifest 对照 | min_tissue 对比统计 | 选择依据 |
| logs/A3_output/C_phase/mt20_tiles_per_slide.csv | mt20 每WSI tile数 | 每张 Axx tile 数 | 统计 |
| logs/A3_output/C_phase/mt40_tiles_per_slide.csv | mt40 每WSI tile数 | 每张 Axx tile 数 | 统计 |
| logs/A3_output/C_phase/mt60_tiles_per_slide.csv | mt60 每WSI tile数 | 每张 Axx tile 数 | 统计 |
| logs/A3_output/C_phase/manifests/mt20/wsi_train_tiles.csv | mt20 tile manifest | tile-level 索引 | 训练/特征提取 |
| logs/A3_output/C_phase/manifests/mt20/wsi_train_bags.csv | mt20 bag labels | WSI 级多标签 | MIL 训练 |
| logs/A3_output/C_phase/manifests/mt40/wsi_train_tiles.csv | mt40 tile manifest | tile-level 索引 | 训练/特征提取 |
| logs/A3_output/C_phase/manifests/mt40/wsi_train_bags.csv | mt40 bag labels | WSI 级多标签 | MIL 训练 |
| logs/A3_output/C_phase/manifests/mt60/wsi_train_tiles.csv | mt60 tile manifest | tile-level 索引 | 训练/特征提取 |
| logs/A3_output/C_phase/manifests/mt60/wsi_train_bags.csv | mt60 bag labels | WSI 级多标签 | MIL 训练 |
| logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled.csv | Stage1.5 r1 tiles | 截断采样 tiles | Stage1.5 混训 |
| logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_qc.csv | r1 采样统计 | r1 类别计数 | 检查分布 |
| logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_light.csv | Stage1.5 r2 tiles | 更轻采样 tiles | Stage1.5 混训 |
| logs/A3_output/C_phase/wsi_train_tiles_mt40_sampled_light_qc.csv | r2 采样统计 | r2 类别计数 | 检查分布 |
| logs/A3_output/C_phase/stage1p5_fold0_r1/best.pt | Stage1.5 r1 权重 | 域对齐实验权重 | 不采用 |
| logs/A3_output/C_phase/stage1p5_fold0_r1/metrics.json | Stage1.5 r1 指标 | 训练曲线 | 不采用 |
| logs/A3_output/C_phase/stage1p5_fold0_r2/best.pt | Stage1.5 r2 权重 | 域对齐实验权重 | 不采用 |
| logs/A3_output/C_phase/stage1p5_fold0_r2/metrics.json | Stage1.5 r2 指标 | 训练曲线 | 不采用 |
| logs/A3_output/reports/training_runs_summary.csv | 训练汇总表 | 各 run 指标汇总 | 总览对比 |
| logs/A3_output/reports/A3_ABC_execution_report.md | 精简报告 | A/B/C 结果概览 | 快速复盘 |
| logs/A3_output/reports/A3_ABC_execution_report_full.md | 详细报告 | 全流程 + 答疑 + safetensors 复训记录 | 深度复盘 |

### 7.3 额外索引文件（derived/split）
| 文件 | 含义 | 用法 |
|---|---|---|
| data/BACH/derived/split/wsi_train_tiles_mt40.csv | mt40 tile manifest | Stage2/Stage3 输入 |
| data/BACH/derived/split/wsi_train_bags_mt40.csv | mt40 bag labels | Stage3 MIL 训练 |

## 8. 关键问题与解决
问题：OpenSlide 缺失
现象：WSI 脚本不可运行
根因：依赖未安装
解决：安装 `openslide-python` + `openslide-bin`
经验：先过依赖门，再跑训练

问题：XML 解析为 0
现象：Axx.xml 无多边形
根因：解析器只支持 QuPath
解决：兼容 Aperio + QuPath
经验：标注格式先做样本探测

问题：timm 下载超时
现象：训练启动失败
根因：默认联网拉权重
解决：默认离线，仅显式 `--pretrained` 下载
经验：远端环境优先离线可跑

问题：Stage1.5 KeyError
现象：dataloader 崩溃
根因：两个 Dataset 字段不同
解决：增加 adapter 统一字段
经验：混合数据集先对齐 schema

问题：Stage1.5 负迁移
现象：指标下降
根因：域样本引入偏移
解决：r2 保守微调验证，仍失败
经验：结果导向，必要时回退

## 9. 自检清单
- `logs/A3_output/A_phase/phaseA_summary.json` 中 photo_total=400、xml_polygon_total=226。
- `logs/A3_output/reports/training_runs_summary.csv` 中 Stage1 baseline best_macro_f1=0.466571（B_final_fold0），Stage1 推荐 best_macro_f1=0.838995（B_safetensors_final_fold0）。
- `logs/A3_output/C_phase/manifest_qc_summary.csv` 中 mt40 tiles_total=14281。
- `logs/A3_output/C_phase/manifests/mt40/wsi_train_tiles.csv` 首行字段包含 `slide_id,slide_path,xml_path,x,y,level,tile_size,label_name,source`。

## 10. 当前可直接用于 Stage2/Stage3 的输入
- Patch encoder（推荐）：`logs/A3_output/B_phase/stage1_safetensors_final_fold0/best.pt`
- Patch encoder（对照/旧）：`logs/A3_output/B_phase/stage1_final_fold0/best.pt`
- WSI tile 索引：`data/BACH/derived/split/wsi_train_tiles_mt40.csv`
- WSI bag 标签：`data/BACH/derived/split/wsi_train_bags_mt40.csv`

## 11. 2026-03-25 收口补充记录

本节记录赛题三“同任务 before/after 双模型、离线导出、统一推理入口、阶段报告”收口工作的代码侧补齐情况。

### 11.1 口径冻结

- `before` 定义为：同一 checkpoint、同一切块参数、同一阈值文件、同一输出 schema 下的 PyTorch eager 推理链。
- `after` 定义为：保持同一任务与输入输出口径，仅替换编码器执行后端为 `ONNX/OM`，优先 `OM`，不可通过改任务定义偷速度。
- WSI 主线固定为：
  - `Stage1.5 UNI-Large checkpoint`
  - `level=1`
  - `tile_size=224`
  - `step=448`
  - `min_tissue=0.4`
  - `TileAgg topk16`
  - 复用现有 `tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- 本地代理评测口径固定为：
  - Photos：`5折 ACC + macro_f1 + ovr_auc`
  - WSI：`5折 exact_match(ACC代理) + macro_f1 + sample_f1 + per-class AP`

### 11.2 新增代码模块

- 新增运行时目录：`src/A3/bach_mil/runtime/`
  - `patch_backends.py`
  - `submission_defaults.py`
  - `__init__.py`
- 作用：
  - 统一 `PyTorch / ONNX / OM` 三种 patch encoder 执行后端
  - 提供 `before/after` 同口径切换能力
  - 固化主线默认 checkpoint、阈值、offline model 路径

### 11.3 已改造/新增脚本

- 已改造：
  - `src/A3/scripts/extract_bag_features.py`
  - `src/A3/scripts/infer_photos.py`
  - `src/A3/scripts/eval_photos_ckpt.py`
  - `src/A3/bach_mil/utils/photo_agg.py`
- 已新增：
  - `src/A3/scripts/run_submission_infer.py`
  - `src/A3/scripts/export_patch_encoder_onnx.py`
  - `src/A3/scripts/compile_patch_encoder_om.py`
  - `src/A3/scripts/validate_patch_backend_parity.py`
  - `src/A3/scripts/run_photos_backend_cv.py`
  - `src/A3/scripts/report_stage_closure.py`

### 11.4 当前能力

- 已具备“一条命令跑完整个测试目录”的统一入口：
  - `python src/A3/scripts/run_submission_infer.py --task wsi --variant before/after --input_dir <目录> --out_dir <目录> --save_features`
- WSI 输出固定为：
  - `features/`
  - `manifests/`
  - `predictions/slide_predictions.csv`
  - `reports/run_summary.json`
- Photos 输出固定为：
  - `predictions/photo_predictions.csv`
  - 可选 `features/`
  - `reports/run_summary.json`
- 已补齐阶段报告生成器，可输出：
  - `01_现状审计报告`
  - `02_before基线跑通报告`
  - `03_ONNX导出与对齐报告`
  - `04_OM导出与对齐报告`
  - `05_before_after速度与精度对比报告`
  - `06_最终提交材料汇总报告`

### 11.5 离线模型落地状态

- WSI 主线已补齐：
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64.onnx`
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.om`
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.meta.json`
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_onnx_export_summary.json`
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_onnx_parity_summary.json`
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_om_compile_summary_origin.json`
  - `logs/A3_output/submission_closure/offline_models/wsi/wsi_om_parity_summary_origin.json`
- Photos 补充链路已补齐：
  - `logs/A3_output/submission_closure/offline_models/photos/photos_patch_encoder_bs64.onnx`
  - `logs/A3_output/submission_closure/offline_models/photos/photos_patch_encoder_bs64_origin.om`
  - `logs/A3_output/submission_closure/offline_models/photos/photos_patch_encoder_bs64_origin.meta.json`
  - `logs/A3_output/submission_closure/offline_models/photos/photos_onnx_export_summary.json`
  - `logs/A3_output/submission_closure/offline_models/photos/photos_om_parity_summary_origin.json`
- 按 `11.6` 当时进度，本轮阶段性 `after` 选型曾冻结为 `origin OM`。
- 但该结论后来已被 `11.17` 的 mixed OM 收口结果覆盖，当前正式 after 已升级为 `mixed OM(attn_score_path_norm1)`。
- 旧尝试说明：
  - `fp16 OM` 旧对齐摘要 `logs/A3_output/submission_closure/offline_models/wsi/wsi_om_parity_summary.json` 全部为 `NaN`，不能作为正式交付。
  - `mixed_float16 OM` 在 `11.6` 当时仅保留编译尝试产物 `wsi_om_compile_summary_mixed.json`；
  - 后续已经在 `11.17` 完成 `keep_dtype + modify_mixlist` 的稳定性收口，并升级为当前正式 after。

### 11.6 对齐验证结论

- WSI `ONNX`：
  - `feature_cosine_mean = 0.9999784827232361`
  - `logit_cosine_mean = 0.9999249577522278`
  - `prob_max_abs_diff = 0.0046797096729278564`
  - 对齐摘要：`logs/A3_output/submission_closure/offline_models/wsi/wsi_onnx_export_summary.json`
- WSI `OM(origin)`：
  - `feature_cosine_mean = 0.999975323677063`
  - `logit_cosine_mean = 0.9999200105667114`
  - `prob_max_abs_diff = 0.004311233758926392`
  - 对齐摘要：`logs/A3_output/submission_closure/offline_models/wsi/wsi_om_parity_summary_origin.json`
- Photos `OM(origin)`：
  - `feature_cosine_mean = 0.9999548196792603`
  - `logit_cosine_mean = 0.9997665882110596`
  - `prob_max_abs_diff = 0.005356550216674805`
  - 对齐摘要：`logs/A3_output/submission_closure/offline_models/photos/photos_om_parity_summary_origin.json`
- 结论：
  - 在 `11.6` 这一阶段，`origin OM` 的特征和分类输出已经足够接近 PyTorch/ONNX，可作为阶段性 after 和保守回退。
  - WSI/Photos 都保留 `ONNX` 作为回退链；但后续正式答辩口径已在 `11.17` 升级为 `mixed OM(attn_score_path_norm1)`。

### 11.7 WSI 读图兼容修复

- 由于容器内缺少 `openslide`，已在仓库引入本地依赖目录 `_vendor/`。
- `src/A3/bach_mil/data/wsi_manifest.py` 已支持 `openslide -> tiffslide` 回退。
- 回退验证：
  - `scan_slide_for_tiles('data/BACH/ICIAR2018_BACH_Challenge/WSI/A01.svs')` 可正常返回 `347` 个 tile。
- 这意味着官方测试集和本地代理评测所需的 `.svs` 扫描链路已经恢复，不再被 `openslide` 缺失阻塞。

### 11.8 统一入口实测结果

- WSI 官方测试集目录：`data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI`
- `before` 统一入口实测：
  - 结果目录：`logs/A3_output/submission_closure/official_runs/wsi_before_unified/`
  - `total_elapsed_sec = 151.3612940311432`
  - `avg_elapsed_sec_per_slide = 15.13612940311432`
  - `wsi_per_sec = 0.06606708844562639`
  - `total_tiles = 2290`
- `after` 正式版 `OM(origin) + num_workers=8` 实测：
  - 结果目录：`logs/A3_output/submission_closure/official_runs/wsi_after_om_origin_nw8/`
  - `total_elapsed_sec = 143.41602230072021`
  - `avg_elapsed_sec_per_slide = 14.341602230072022`
  - `wsi_per_sec = 0.06972721624527849`
  - `total_tiles = 2290`
- 统一口径速度对比结论：
  - `speedup = 1.055329487895469`
  - 耗时下降 `5.249301255407235%`
  - 端到端提速有限，说明当前主瓶颈不只在编码器，读图/切块/Python 调度仍占明显比例。
- WSI 本地代理评测：
  - `before`：`logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv`
  - `after`：`logs/A3_output/submission_closure/proxy_eval/wsi_after_om_origin_nw8_cv/cv_summary_mean_std.csv`
  - 两者均值一致：
    - `exact_match = 1.0`
    - `macro_f1 = 0.9333333333333332`
    - `sample_f1 = 1.0`
  - 结论：WSI 本地代理 ACC 变化 `0.0` 个百分点，`macro_f1` 变化 `0.0` 个百分点。
- Photos 本地代理评测：
  - `before`：`ACC = 0.92`，`macro_f1 = 0.9191175474519504`
  - `after OM(origin)`：`ACC = 0.945`，`macro_f1 = 0.9445144397512522`，`ovr_auc = 0.9963749999999999`
  - 注意：当前 `after OM(origin)` 目录 `logs/A3_output/submission_closure/proxy_eval/photos_after_om_origin/` 使用的是单个 `fold0/best.pt` 去跑完整 5 折，因此它只能作为“离线后端补充代理结果”，不能当成严格同 checkpoint、同折次配对的 before/after 精度对比。
  - 因此，Photos 这组数当前只能说明“离线后端在补充代理评测里能正常工作且结果可用”，不能据此下结论说“加速后精度提升了 2.5 个百分点”。

### 11.9 Photos 统一入口烟雾验证

- 为验证 `Photos` 模式也能在统一入口下真正走离线后端，新增运行：
  - 命令：`python src/A3/scripts/run_submission_infer.py --task photos --variant after --backend om --input_dir logs/A3_output/submission_closure/smoke_inputs/photos1 --out_dir logs/A3_output/submission_closure/smoke_runs/photos_after_om_origin --save_features`
- 实测结果：
  - `backend_actual = om`
  - `total_elapsed_sec = 1.9512953758239746`
  - 输出目录已包含：
    - `features/test0.pt`
    - `predictions/photo_predictions.csv`
    - `reports/run_summary.json`
    - `reports/per_image_timing.csv`
- 结论：Photos 统一入口已验证能输出“特征 + 分类结果 + 运行统计”，满足补充交付要求。

### 11.10 阶段报告与当前结论

- 阶段报告脚本：`src/A3/scripts/report_stage_closure.py`
- 输出目录：`logs/A3_output/submission_closure/reports/`
- 固定 6 份报告已进入正式收口口径：
  - `01_现状审计报告`
  - `02_before基线跑通报告`
  - `03_ONNX导出与对齐报告`
  - `04_OM导出与对齐报告`
  - `05_before_after速度与精度对比报告`
  - `06_最终提交材料汇总报告`
- 当前正式结论：
  - 两个模型指的是同一任务、同一输入输出口径下的 `before(PyTorch eager)` 与 `after(mixed OM attn_score_path_norm1)`。
  - 官方 `TestDataset` 只做真实推理速度和预测结果输出，不计算真实 ACC，因为无标签。
  - `thumbnails` 只能人工观察，不能当真值。
  - `.onnx` 与 `.om` 已完成；`.air` 仍为可选补充项，不阻塞当前提交。

### 11.11 六轮次加速重排与最终 after 重选

- 为避免旧 `after` 留档误导，已新增统一轮次总控：`src/A3/scripts/run_accel_rounds.py`
- WSI 6 个轮次目录已固定到：`logs/A3_output/submission_closure/optimization_rounds/wsi/`
- 当前实测汇总见：
  - `logs/A3_output/submission_closure/optimization_rounds/wsi/rounds_summary.csv`
  - `logs/A3_output/submission_closure/optimization_rounds/wsi/rounds_summary.md`
- 6 轮次最新结果如下：
  - `01_before_plain_pytorch`：`157.730324s`，`0.063399 WSI/s`，`14.518451 tiles/s`
  - `02_after_pytorch_engineered`：`1039.200859s`，`0.009623 WSI/s`，纯 PyTorch 工程优化在当前主线下明显更慢，不纳入最终提交
  - `03_after_onnx_cpu_smoke`：`216.014406s`，`0.004629 WSI/s`，当前环境只有 `CPUExecutionProvider`，只保留为导出与功能验证链路
  - `04_after_om_acl_sync`：`27.607040s`，`0.362226 WSI/s`，`82.949856 tiles/s`，相对 before `5.713409x` 加速，耗时下降 `82.497316%`
  - `05_after_om_acl_async`：`31.030313s`，`0.322266 WSI/s`，`73.798804 tiles/s`，相对 before `5.083105x` 加速
  - `06_after_om_acl_async_cachewarm`：`28.320765s`，`0.353098 WSI/s`，`80.859398 tiles/s`，用于重复运行上限观察，不作为正式单次提交口径
- 由此重选后的正式 `after` 不再是旧的 `wsi_after_om_origin_nw8`，而是：
  - `04_after_om_acl_sync`
  - 核心方法：`OM(origin) + ACL sync + prefetch + persistent_workers + pin_memory + manifest cache`
- 当前默认一条命令已经改成复现这个正式 `after`：
  - `python src/A3/scripts/run_submission_infer.py --task wsi --variant after --backend om --input_dir <测试WSI目录> --out_dir <输出目录> --save_features`
- 精度口径保持不变：
  - WSI `before/after` 本地代理 `exact_match = 1.0`
  - WSI `before/after` 本地代理 `macro_f1 = 0.9333333333333332`
  - 绝对精度损失 `0.0` 个百分点
- 这说明：
  - `OM(origin)` 已经满足“明显加速且精度近似无损”
  - `ACL async` 在当前链路下不是最优点，不能为了“方法更多”硬选为最终提交版
  - 最终答辩应明确：`after` 选的是“实测最优且可复现”的组合，不是把所有方法机械叠加
- 当前仍不能诚实写成“已落地提交产物”的路线：
  - `MindSpore`：环境未安装
  - `ATB`：环境未检测到可执行工具
  - `ONNX NPU Provider`：当前仅 `CPUExecutionProvider`
- `量化/蒸馏/剪枝`：本轮没有新增且完成精度回归验证的正式权重，暂不算正式提交件

### 11.12 Photos 四分类严格代理评测补齐

- 之前 `logs/A3_output/submission_closure/proxy_eval/photos_after_om_origin/` 的 `0.945` 结果，不是严格的 before/after 配对结果。
- 原因：
  - 该目录使用的是单个 `fold0/best.pt` 跑完整 5 折
  - 它可以作为“离线后端可运行”的补充结果
  - 但不能直接当成“四分类 after ACC 提升”的正式结论
- 为补齐严格口径，已改造脚本：
  - `src/A3/scripts/run_photos_backend_cv.py`
  - 新增 `--strict_fold_ckpt`
  - 支持每个 `fold` 自动加载对应 `fold{n}/best.pt`
- 已生成严格版 `Photos before` 四分类代理评测：
  - 输出目录：`logs/A3_output/submission_closure/proxy_eval/photos_before_pytorch_strict_cv/`
  - 配置：每折使用各自 checkpoint
  - 5 折均值：
    - `ACC = 0.92`
    - `macro_f1 = 0.9191175474519504`
    - `ovr_auc = 0.9909583333333332`
- 这组数现在可以作为“本地严格四分类代理 ACC”使用。
- 下一步若要得到严格版 `Photos after`：
  - 需要为每个 `fold` 分别导出/编译对应 `OM` 或至少准备每折一一对应的离线模型
  - 然后再用同一个脚本按 `fold` 对齐评测

### 11.13 2026-03-26 部署运行说明最终补充

这一节专门记录本轮对话里补齐的“新服务器迁移、部署、输入输出、目录名变化、WSI 输入格式兼容”信息。

#### 11.13.1 当前推荐的最小迁移树

推荐把最终可运行内容整理成：

```text
<任意根目录>/
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

最重要的结论是：

- 根目录名不要求必须叫 `data-set`
- 可以叫 `dataset`
- 也可以叫别的名字
- 关键是：
  - `src/A3`
  - `logs/A3_output`
  - `_vendor`
  - `input`
  这些相对结构不要乱

#### 11.13.2 推荐部署命令

假设迁移后的根目录是：

- `/home/modellite/workspace/data-set`

则：

```bash
export ROOT=/home/modellite/workspace/data-set
cd $ROOT
python3 -m pip install -r $ROOT/src/A3/requirements.txt
```

环境自检：

```bash
python3 -c "import torch; print('torch ok')"
python3 -c "import torch_npu; print('torch_npu ok')"
python3 -c "import acl; print('acl ok')"
python3 -c "import openslide; print('openslide ok')"
```

运行 `before`：

```bash
python3 $ROOT/src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant before \
  --backend pytorch \
  --input_dir $ROOT/input \
  --out_dir $ROOT/output_before \
  --save_features
```

运行 `after`：

```bash
python3 $ROOT/src/A3/scripts/run_submission_infer.py \
  --task wsi \
  --variant after \
  --backend om \
  --input_dir $ROOT/input \
  --out_dir $ROOT/output_after \
  --save_features
```

#### 11.13.3 输入输出目录说明

- 输入目录：
  - `<根目录>/input/`
- `before` 输出目录：
  - `<根目录>/output_before/`
- `after` 输出目录：
  - `<根目录>/output_after/`

WSI 输出结构固定为：

```text
output_before/ 或 output_after/
├── features/
├── manifests/
├── predictions/slide_predictions.csv
└── reports/
    ├── per_slide_timing.csv
    └── run_summary.json
```

#### 11.13.4 当前已经支持的 WSI 输入格式

本轮已经把统一入口和底层读图链路补成：

- `.svs`
- `.tif`
- `.tiff`
- 大小写混合后缀也支持

也就是说，下面这些现在都能被自动扫描：

- `case01.svs`
- `case02.tif`
- `case03.tiff`
- `case04.TIF`
- `case05.TIFF`

### 11.14 2026-03-26 WSI 输入兼容修复记录

这一轮补的是“防止部署后读不到 tiff/tif”的真实工程问题。

本轮修改了 4 个文件：

- `src/A3/bach_mil/data/wsi_manifest.py`
- `src/A3/scripts/run_submission_infer.py`
- `src/A3/scripts/extract_bag_features.py`
- `src/A3/scripts/run_accel_rounds.py`

具体修复内容：

1. 新增统一 WSI 文件发现函数  
   不再只扫 `*.svs`，而是统一支持 `svs / tif / tiff`

2. 底层开图逻辑更稳  
   以前更偏“能 import openslide 就直接用”；现在变成：
   - 先尝试 `openslide`
   - 如果当前文件实际打不开，再自动回退 `tiffslide`

3. 统一入口与特征提取脚本同步放开后缀  
   避免出现“入口脚本能跑，但离线提特征脚本不认 tif”的不一致问题

4. 加速轮次总控同步放开 WSI 输入发现  
   避免测速轮次漏掉 `tif/tiff`

自检结果：

- `python3 -m py_compile` 已通过
- 已用临时目录验证：
  - `a.svs`
  - `b.tif`
  - `c.tiff`
  - `d.TIF`
  - `e.TIFF`
  都能被识别

### 11.15 训练、评测、部署链路最终解释

这一节专门回答本轮对话里反复出现的 6 个问题：

1. 训练数据到底在哪里
2. 测试数据到底在哪里
3. 为什么不是 5 个模型
4. epoch、fold、slide 到底是什么关系
5. 编码器到底是什么
6. MIL 在当前主线里有没有用到

#### 11.15.1 训练数据在哪里

当前正式链路里，训练相关数据分两部分：

1. Photos 四分类监督训练数据  
   文件来源：`data/BACH/ICIAR2018_BACH_Challenge/Photos`

2. WSI A01~A10 + XML  
   文件来源：
   - `data/BACH/ICIAR2018_BACH_Challenge/WSI/A01.svs ~ A10.svs`
   - `data/BACH/ICIAR2018_BACH_Challenge/WSI/A01.xml ~ A10.xml`

其中：

- Photos 负责学习通用病理形态识别能力
- WSI A01~A10 + XML 负责把这种能力往 WSI tile 域对齐，并提供 slide 级多标签代理评测

#### 11.15.2 测试数据在哪里

测试相关数据也分两部分：

1. 本地代理评测数据
   - Photos：`data/BACH/derived/split/photos_folds.csv`
   - WSI：`A01~A10` 的 slide 级标签与阈值搜索

2. 官方无标签测试数据
   - `data/BACH/ICIAR2018_BACH_Challenge_TestDataset/WSI`
   - `data/BACH/ICIAR2018_BACH_Challenge_TestDataset/Photos`

这里必须分清：

- 本地代理评测：可以算代理 ACC
- 官方测试集：只能测速和导出预测，不能算真实 ACC

#### 11.15.3 为什么当前正式 WSI 不是 5 个 encoder 模型

因为当前正式 WSI encoder 的训练方式是：

- 读取：
  - `data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv`
- 切分方式：
  - `random_tile`
- 验证比例：
  - `val_ratio = 0.1`
- 训练轮数：
  - `epoch 1 ~ 5`
- 按验证集 `macro_f1` 选最优

证据文件：

- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/split_meta.json`
- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/metrics.json`

关键事实：

- `split_mode = random_tile`
- `val_ratio = 0.1`
- `train_slides = A01~A10`
- `val_slides = A01~A10`
- 最优 `macro_f1 = 0.8413125153978812`
- 最优 epoch 是 `epoch 4`

所以当前正式 WSI encoder 的真实含义是：

- 在 A01~A10 切出来的 tile 上做随机 tile 切分
- 训练 1 个 encoder
- 在 5 个 epoch 中选最好的那个状态
- 保存成 1 个 `best.pt`

它不是：

- `fold0` 训 1 个 encoder
- `fold1` 再训 1 个 encoder
- ...
- 最后再把 5 个 encoder 合并成 1 个

#### 11.15.4 epoch、fold、slide 的关系

- `slide`
  - 一张整张病理图，例如 `A01.svs`
- `epoch`
  - 同一个模型对同一份训练集完整学习一轮
- `fold`
  - 交叉验证时，把样本分成第几折

层级关系可以这样理解：

- `slide` 是数据样本
- `fold` 是怎么分训练集和验证集
- `epoch` 是某一折里这个模型训练了第几轮

当前正式 WSI 主线里：

- encoder 训练阶段主要是 `epoch` 选优
- slide 级代理评测阶段才用到 `fold`

#### 11.15.5 当前正式 WSI 5 折到底发生在哪一层

当前的 5 折不是发生在 encoder 训练层，而是发生在：

- 提完 slide 特征之后
- 用同一套 encoder 产出的 feature
- 做 slide 级聚合与阈值搜索
- 统计 `exact_match / macro_f1 / sample_f1`

所以当前 WSI 的“5 折结果”本质上是：

- 1 个 encoder
- 5 折 slide 级代理评测

不是：

- 5 个 encoder
- 再 ensemble 成 1 个部署模型

#### 11.15.6 编码器是什么，MIL 又在做什么

编码器的职责是：

- 输入一个 `224 x 224` 的 tile
- 输出这个 tile 的特征和类别分数

在当前主线里：

- `tile` 基本可以理解成送入模型的一张 patch 图块
- `ViT patch16` 里的 `16` 是模型内部 token patch 大小
- 不是你外部切 WSI 的 tile 大小

MIL / 聚合层的职责是：

- 把一整张 slide 的很多 tile 结果汇总起来
- 得到 slide 级标签

当前最终部署主线使用的是：

- `TileAgg topk16`
- 也就是先提每个 tile 的概率
- 再做 `topk_mean_prob`
- 最后按阈值得到 `Benign / InSitu / Invasive / Normal`

换句话说：

- encoder 负责“看局部”
- 聚合层负责“看整张”

### 11.16 本轮文档与挂载目录同步要求

本轮对话结束后，除了仓库内文档更新，还要求把 `/data/demo` 下挂载内容同步更新。

应同步的文档：

- `README.md`
- `README-process.md`
- `赛题三具体评测要求（必看）.md`

应同步的代码：

- `src/A3/bach_mil/data/wsi_manifest.py`
- `src/A3/scripts/run_submission_infer.py`
- `src/A3/scripts/extract_bag_features.py`
- `src/A3/scripts/run_accel_rounds.py`

建议同步位置：

- `/home/ma-user/work/uni_run/tmp/data-set/`
- `/data/demo/`
- `/data/demo/data-set/`

如果还需要继续分发压缩包，则同步后再重打：

- `/data/demo/data-set.tar`

### 11.17 2026-03-27 mixed OM 收口 + 蒸馏/剪枝长时训练结果

#### 11.17.1 mixed OM 第二轮精细化结论

在 `2026-03-27`，继续围绕 mixed OM 的 `NaN` 问题做了第二轮精细化 sweep。

新增脚本：

- `src/A3/scripts/run_step01_refined_attn_sweep.py`

扩展的 mixed profile 规则：

- `attn_softmax_div_sqrt`
- `attn_softmax_div_sqrt_mul`
- `attn_score_path`
- `attn_score_path_norm1`
- `attn_score_path_norm1_qkv_proj`

关键结论：

- 只保 `Softmax/Div/Sqrt` 或再补 `Mul` 仍然会在真实 tile 输入下出现全量 `NaN`
- 当保护范围收缩到 attention 分数路径 `Softmax/Div/Sqrt/Mul/score MatMul` 时，模型恢复稳定
- 在此基础上补回 `norm1` 后，得到当前最优稳定 mixed OM 配置：`attn_score_path_norm1`

关键结果文件：

- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/refined_attn_sweep/refined_attn_sweep_summary.csv`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_precision收口报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_precision收口报告.json`

核心数值：

- 上一轮最佳稳定 mixed：`attn_numeric = 373.8898 patch/s`
- 本轮最佳稳定 mixed：`attn_score_path_norm1 = 514.8922 patch/s`
- 吞吐提升：`+141.0024 patch/s`
- 相对提升：`+37.71%`

随机 batch 对齐结果：

- `feature_cosine_mean = 0.9999759197`
- `logit_cosine_mean = 0.9999160767`
- `prob_max_abs_diff = 0.0043846816`
- `prob_mean_abs_diff = 0.0017295893`

当前 mixed OM 正式候选建议切换为：

- `attn_score_path_norm1`

对应产物目录：

- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/`

#### 11.17.2 蒸馏长时训练结果

在 `2026-03-27`，启动并完成了 `12 epoch` 的 teacher-student 蒸馏训练。

teacher：

- `vit_large_patch16_224`
- checkpoint：`logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`

student：

- `vit_base_patch16_224`

输出目录：

- `logs/A3_output/submission_closure/optimization_stepwise/04_teacher_student_distill/models/distill_student_v2_long_e12/`

关键产物：

- `best_student.pt`
- `metrics.json`
- `split_meta.json`
- `val_predictions_best_student.csv`

最佳结果：

- 最优 epoch：`11`
- `acc = 0.7681564246`
- `macro_f1 = 0.4017273576`
- `ovr_auc = 0.8548201271`

报告文件：

- `logs/A3_output/submission_closure/optimization_stepwise/04_teacher_student_distill/analysis/04_蒸馏长时训练报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/04_teacher_student_distill/analysis/04_蒸馏长时训练报告.json`

#### 11.17.3 结构化剪枝长时训练结果

在 `2026-03-27`，启动并完成了 `10%` 结构化剪枝 + `8 epoch` 微调。

输出目录：

- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/models/prune_ratio_0p10_v2_long_e8/`

关键产物：

- `best_pruned.pt`
- `metrics.json`
- `prune_report.json`
- `split_meta.json`

最佳结果：

- 最优 epoch：`6`
- `acc = 0.9078212291`
- `macro_f1 = 0.8289096358`
- `ovr_auc = 0.9767076634`

与原始大模型 `best.pt` 对比：

- 原始最优 `acc = 0.9162011173`
- 原始最优 `macro_f1 = 0.8413125154`
- 剪枝后 `acc` 下降约 `0.838` 个百分点
- 剪枝后 `macro_f1` 下降约 `1.240` 个百分点

剪枝统计：

- 剪枝模块数：`97`
- 平均零权重占比：`0.0998107281`
- `include_classifier = false`

报告文件：

- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝长时训练报告.md`
- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝长时训练报告.json`

#### 11.17.4 当前收口结论

截至 `2026-03-27`：

- mixed OM 的 `NaN` 问题已经从“大范围不稳定”收口到“attention 分数路径敏感”
- 当前最优稳定 mixed OM 甜点值是：`attn_score_path_norm1`
- 蒸馏路线已经拿到可运行 student，但当前精度明显低于大模型
- `10%` 结构化剪枝路线目前比蒸馏路线更接近主线精度，但 `macro_f1` 损失仍略高于 `1` 个百分点

现阶段更适合继续推进的路线排序：

1. `attn_score_path_norm1` mixed OM 作为当前 after 主线
2. 在剪枝线上继续尝试更小剪枝率或更长微调，把 `macro_f1` 损失压到 `<1%`
3. 蒸馏 student 作为后续轻量化补充路线，而不是当前主提交路线

#### 11.17.5 5% 结构化剪枝补充结果

在 `2026-03-27`，继续完成了 `5%` 结构化剪枝 + `10 epoch` 微调，目标是把 `macro_f1` 损失压到 `<1%`。

输出目录：

- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/models/prune_ratio_0p05_v3_long_e10/`

关键产物：

- `best_pruned.pt`
- `metrics.json`
- `prune_report.json`
- `split_meta.json`

最佳结果：

- 最优 epoch：`5`
- `acc = 0.9189944134`
- `macro_f1 = 0.8391020323`
- `ovr_auc = 0.9835653075`

与原始大模型 `best.pt` 对比：

- 原始最优 `acc = 0.9162011173`
- 原始最优 `macro_f1 = 0.8413125154`
- `5%` 剪枝后：
  - `acc` 反而提升约 `0.279` 个百分点
  - `macro_f1` 仅下降约 `0.221` 个百分点

剪枝统计：

- 剪枝模块数：`97`
- 平均零权重占比：`0.0499456347`
- `include_classifier = false`

补充报告文件：

- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_结构化剪枝5pct补充报告.md`

结论：

- `5%` 剪枝已经满足“尽量不损失精度”的目标，明显优于 `10%` 剪枝
- 当前它是最值得继续做下一轮 `ONNX/OM` 导出的轻量化候选

#### 11.17.6 mixed OM 端到端全流程复核

在完成 patch 级 mixed OM 收口后，继续做了两轮更贴近交付的复核：

1. 官方无标签测试集 `WSI` 整目录全流程测速  
2. `A01~A10` 本地代理评测口径复算

mixed OM 工件：

- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json`

官方无标签测试集整目录实测输出：

- 目录：`logs/A3_output/submission_closure/optimization_rounds/wsi/07_after_om_mixed_attn_score_path_norm1/`
- `total_elapsed_sec = 25.2605886459`
- `wsi_per_sec = 0.3958735935`
- `tiles_per_sec = 90.6550529007`

与既有两条线对比：

- 相对 `before(PyTorch)`：
  - 加速比：`6.2441x`
  - 总耗时下降：`83.9850%`
- 相对 `origin OM`：
  - 额外加速：`1.0929x`
  - 总耗时再下降：`8.4995%`

一致性自检：

- mixed OM 与 `origin OM` 在官方 10 张无标签测试 WSI 上：
  - 最终 `pred_labels`：`10/10` 完全一致

本地代理评测复算：

- 特征目录：`logs/A3_output/submission_closure/proxy_eval/wsi_mixed_om_attn_score_path_norm1_extract_a01a10_v2/run/features`
- CV 输出：`logs/A3_output/submission_closure/proxy_eval/wsi_mixed_om_attn_score_path_norm1_cv/cv_summary_mean_std.csv`
- 结果：
  - `exact_match = 1.0`
  - `macro_f1 = 0.9333333333`
  - `sample_f1 = 1.0`

结论：

- mixed OM 已经不再只是 patch 级甜点值，而是完成了：
  - 编码器稳定性
  - 官方整目录真实测速
  - 本地代理评测
- 因此可以把当前推荐正式 after 从 `origin OM` 升级为：
  - `mixed OM(attn_score_path_norm1)`

补充报告文件：

- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/analysis/01_mixed_om端到端补充报告.md`

#### 11.17.7 当前最终推荐路线更新

截至 `2026-03-27` 晚间，当前 A3/A4 的推荐关系更新如下：

1. 正式主提交 `after`
   - `mixed OM(attn_score_path_norm1) + ACL sync + prefetch + persistent_workers + pin_memory + manifest cache`
   - 原因：
     - 官方整目录全流程比 `origin OM` 再快约 `8.50%`
     - 官方 10 张测试 WSI 最终标签与 `origin OM` 完全一致
     - 本地代理 `exact_match / macro_f1` 与原正式线完全一致
2. 保守回退 `after`
   - `origin OM`
   - 继续保留，不删除
3. 后续轻量化优先候选
   - `5%` 结构化剪枝 `best_pruned.pt`
   - 目前最接近“更轻 + 几乎不掉点”的目标
4. 不推荐作为当前主提交
   - 当前蒸馏 student
   - 当前 `10%` 剪枝

#### 11.17.8 A4 已完成与未完成边界

当前可以诚实写成“已完成”的 A4 内容：

- `ONNX` 导出与对齐
- `origin OM` 导出与对齐
- mixed OM `keep_dtype + modify_mixlist` 稳定性收口
- mixed OM 官方整目录真实测速
- mixed OM 本地代理评测
- `5%` 剪枝模型 `ONNX/OM` 导出与对齐
- `5%` 剪枝模型官方整目录真实测速
- `5%` 剪枝模型本地代理复核与 deploy thresholds 标定
- ACL Python 包装层 `legacy vs buffer_reuse` 对照
- 蒸馏长时训练
- `10% / 5%` 结构化剪枝长时训练
- 统一 before/after 推理入口

当前必须诚实写成“预研/待补齐”的内容：

- `INT8/PTQ/QAT` 正式量化链路
- `AIPP` 正式下沉后的提交链
- mixed OM 的异步/缓存预热再收口

#### 11.17.9 本轮文档更新

本轮对话完成后，说明文件同步更新为：

- `README.md`
- `README-process.md`
- `analyse.md`

其中：

- `README.md` 负责“当前怎么交、怎么跑”
- `README-process.md` 保留全过程
- `analyse.md` 专门整理 A1~A4 的关键数据、日志路径、优化结论

#### 11.17.10 5% 剪枝模型 ONNX/OM + 官方整目录测速 + 本地代理复核

本轮把此前“最值得继续离线化的轻量化候选”真正补成了一条完整闭环：

1. `best_pruned.pt -> ONNX`
2. `ONNX` 对齐验证
3. `ONNX -> OM(origin)`
4. `OM` 对齐验证
5. `A01~A10` 本地代理抽特征
6. `5 折` 代理复核
7. `A01~A10` 全量 deploy thresholds 标定
8. 官方无标签测试集整目录测速与预测导出

关键产物：

- 剪枝权重：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/models/prune_ratio_0p05_v3_long_e10/best_pruned.pt`
- prune5 ONNX：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/offline_models/prune_ratio_0p05_v3_long_e10/wsi_patch_encoder_prune5_bs64.onnx`
- prune5 OM：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/offline_models/prune_ratio_0p05_v3_long_e10/wsi_patch_encoder_prune5_bs64_origin.om`
- prune5 OM 对齐：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/offline_models/prune_ratio_0p05_v3_long_e10/wsi_om_parity_summary_prune5_origin.json`
- prune5 本地代理 CV：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/proxy_eval/prune_ratio_0p05_v3_long_e10/cv/cv_summary_mean_std.csv`
- prune5 deploy thresholds：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/proxy_eval/prune_ratio_0p05_v3_long_e10/prune5_thresholds_full_a01a10.json`
- prune5 官方整目录测速：
  - `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/official_runs/prune_ratio_0p05_v3_long_e10/run/reports/run_summary.json`

核心结果：

- ONNX 导出成功，导出耗时 `325.5008s`
- prune5 ONNX 对齐：
  - `feature_cosine_mean = 1.0`
  - `prob_max_abs_diff = 7.152557373046875e-07`
- prune5 OM 对齐：
  - `feature_cosine_mean = 0.9999767541885376`
  - `logit_cosine_mean = 0.9999826550483704`
  - `prob_max_abs_diff = 0.006109356880187988`
- prune5 本地代理复核：
  - `exact_match = 1.0`
  - `macro_f1 = 0.9333333333333332`
  - `sample_f1 = 1.0`
- prune5 deploy thresholds：
  - `Benign = 0.1`
  - `InSitu = 0.1`
  - `Invasive = 0.5`
- prune5 官方整目录：
  - `total_elapsed_sec = 29.4291195869`
  - `wsi_per_sec = 0.3397994959`
  - `tiles_per_sec = 77.8140845578`

与其它正式链比较：

- 相对 before：
  - 加速比：`5.3597x`
  - 总耗时下降：`81.3421%`
- 相对当前正式 mixed OM：
  - prune5 总耗时慢 `16.5021%`
  - prune5 `tiles/s` 低约 `14.17%`
  - 官方 10 张无标签测试 WSI 最终 `pred_labels`：仅 `7/10` 与 mixed OM 一致

为什么 prune5 没有替代当前正式 mixed OM：

- `prune_report.json` 显示这轮 `5%` 剪枝的核心结果是“各线性层约 5% 权重置零”，但导出的 `ONNX/OM` 仍保持原始 dense shape。
- 也就是说，这一轮更接近“稀疏化权重 + 微调恢复精度”，而不是“真正缩小矩阵维度的结构裁剪”。
- 当前 Ascend `ATC/ACL` 运行时不会自动把这种同 shape 的稀疏权重转成更小、更快的图，所以最终速度没有超过 mixed OM。

因此，当前对 prune5 的正确定位是：

- 它已经完成闭环，不能再写成“还没导出/还没测速”
- 它精度达标，可以作为轻量化候选
- 但它不是当前最快的正式 after，不替代 mixed OM 主提交

补充报告：

- `logs/A3_output/submission_closure/optimization_stepwise/05_structured_prune/analysis/05_prune5_ONNX_OM_官方测速_代理复核报告.md`

#### 11.17.11 为什么当前不把 AIPP / INT8 写成正式完成

当前不把 `AIPP / INT8` 写成正式完成，不是因为方向错误，而是因为还没有拿到“可复现 + 可对齐 + 可答辩”的最终产物。

关于 `AIPP`：

- 当前主线真正的重负载不只是 `resize + normalize`，更大头包括：
  - `OpenSlide / TIFF` 读图
  - `WSI` 扫描与有效组织筛选
  - tile manifest 生成与缓存
- `AIPP` 更适合把固定图像预处理契约下沉到设备侧，但当前主线输入并不是“原始整图直接喂模型”，而是：
  - 先在 host 侧完成读图、切块、组织区域筛选
  - 再把标准化后的 `224x224 tile tensor` 送入编码器
- 因此，在当前主线里，强行把 `AIPP` 写成正式完成会带来较大契约改造成本，但还没有拿到同口径、可复现的真实收益表。

关于 `INT8`：

- 当前仓库里还没有一条已经通过：
  - `PTQ/QAT` 导出
  - 数值对齐
  - 本地代理复核
  - 官方整目录测速
  的正式 `INT8` 交付链。
- 你的核心要求是“速度快，同时精度近似无损”。在没有完整 `PTQ/QAT` 实测前，不能诚实地把 `INT8` 写成已完成。
- 目前已经证明稳定可用的正式主线是：
  - `mixed OM(attn_score_path_norm1)`：速度最快，代理不掉点
  - `prune5 OM(origin)`：精度达标，但速度仍不如 mixed OM

所以当前最诚实、最稳妥的答辩口径是：

- `AIPP / INT8` 是已经明确规划、但仍在预研/待补齐的方向
- 当前正式提交主线仍以已完成闭环验证的 `mixed OM` 为准

#### 11.17.13 BRACS 20 张外部多标签验证与阈值优化

在 `2026-03-27`，继续补做了 `BRACS` 外部多标签验证，用于避免只看 `BACH A01~A10` 本地代理而过度乐观。

数据来源：

- `BRACS/subset20_test_balanced/meta/subset20_manifest.csv`
- `BRACS/subset20_test_balanced/meta/BRACS.xlsx`
- `BRACS/subset20_test_balanced/input_flat/`

真值口径：

- 使用 `BRACS.xlsx` 的 `WSI_with_RoI_Distribution`
- 折叠为当前任务 4 类多标签：
  - `Normal = N > 0`
  - `Benign = PB > 0 or UDH > 0`
  - `InSitu = FEA > 0 or ADH > 0 or DCIS > 0`
  - `Invasive = IC > 0`

运行结果：

- before：
  - `BRACS/subset20_test_balanced/runs/bracs20_before_pytorch_v1/`
- after：
  - `BRACS/subset20_test_balanced/runs/bracs20_after_mixedom_v1/`

速度：

- before：
  - `total_elapsed_sec = 483.1228125095`
  - `tiles_per_sec = 19.2021568011`
- after：
  - `total_elapsed_sec = 440.4346876144`
  - `tiles_per_sec = 21.0632819369`
- after 相对 before：
  - 加速比：`1.0969227132x`
  - 总耗时下降：`8.8358743967%`
  - `tiles/s` 提升：`9.6922713164%`

默认阈值：

- `Benign = 0.1`
- `InSitu = 0.1`
- `Invasive = 0.5`

默认阈值下的外部多标签结果：

- before：
  - `exact_match = 0.30`
  - `macro_f1 = 0.6882662835`
  - `sample_f1 = 0.7178571429`
- after：
  - `exact_match = 0.30`
  - `macro_f1 = 0.6882662835`
  - `sample_f1 = 0.7178571429`
- `before/after` 默认逐张 `pred_labels` 一致率：`1.0`

继续在同一批 `20` 张上做 `Benign / InSitu / Invasive` 三阈值网格搜索后：

- before tuned 阈值：
  - `Benign = 0.311959922314`
  - `InSitu = 0.396706029773`
  - `Invasive = 0.736946821213`
- after tuned 阈值：
  - `Benign = 0.331659242511`
  - `InSitu = 0.396754652262`
  - `Invasive = 0.847445487976`

tuned 结果：

- before：
  - `exact_match = 0.45`
  - `macro_f1 = 0.8550549451`
  - `sample_f1 = 0.7845238095`
- after：
  - `exact_match = 0.45`
  - `macro_f1 = 0.8550549451`
  - `sample_f1 = 0.7845238095`

结论：

- `after` 在外部 `BRACS 20` 张上相比 `before` 没有额外精度损失
- 只靠阈值调节，`exact_match` 可从 `0.30` 提升到 `0.45`
- 但这组 tuned 阈值是在同一批 `20` 张样本上直接搜索得到，属于“外部分析参考”，不直接替换正式提交阈值

脚本化复现目录：

- `BRACS/subset20_test_balanced/eval_runs/bracs20_roi_eval_v1/`

脚本：

- `src/A3/scripts/eval_bracs_roi_multilabel.py`

示例命令：

```bash
python src/A3/scripts/eval_bracs_roi_multilabel.py \
  --manifest_csv BRACS/subset20_test_balanced/meta/subset20_manifest.csv \
  --bracs_xlsx BRACS/subset20_test_balanced/meta/BRACS.xlsx \
  --pred_csv before=BRACS/subset20_test_balanced/runs/bracs20_before_pytorch_v1/predictions/slide_predictions.csv after=BRACS/subset20_test_balanced/runs/bracs20_after_mixedom_v1/predictions/slide_predictions.csv \
  --run_summary_json before=BRACS/subset20_test_balanced/runs/bracs20_before_pytorch_v1/reports/run_summary.json after=BRACS/subset20_test_balanced/runs/bracs20_after_mixedom_v1/reports/run_summary.json \
  --thresholds_json logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json \
  --normal_fallback \
  --out_dir BRACS/subset20_test_balanced/eval_runs/bracs20_roi_eval_v1
```

#### 11.17.14 新服务器迁移命令修正

因为目标机要求 `python3` 且不主动升级 Python / pip，所以部署命令统一收敛为：

```bash
export ROOT=/home/modellite/workspace/data-set
cd $ROOT
python3 -m pip install -r $ROOT/src/A3/requirements.txt
```

不再默认执行：

```bash
python3 -m pip install --upgrade pip
```

原因：

- 当前环境已经可用，重点是安装缺失依赖，不是升级打底工具链
- 部分昇腾镜像升级 `pip` 后反而容易带来环境漂移

#### 11.17.15 最小迁移包与打包落地

为了直接迁移到另一台昇腾服务器，本轮额外整理了最小可运行包：

- 未压缩目录：
  - `/home/ma-user/work/uni_run/tmp/data-set`
- 打包后压缩文件：
  - `/data/demo/data-set.tar`
- 推荐目标机目录：
  - `/home/modellite/workspace/data-set`

迁移包内保留的核心内容：

- `src/A3/`
- `_vendor/`
- `logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt`
- `logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om`
- `logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json`
- `input/`
- `output_before/`
- `output_after/`
- `README_DEPLOY.md`
- `install_requirements.sh`
- `env_check.sh`
- `run_before.sh`
- `run_after.sh`

运行方式统一收敛为：

```bash
cd /home/modellite/workspace/data-set
bash install_requirements.sh
bash env_check.sh
bash run_before.sh
bash run_after.sh
```

#### 11.17.12 本轮状态更新

经过本轮补齐后，A4 的状态应改写为：

- 已完成：
  - mixed OM 主提交闭环
  - prune5 `ONNX/OM` 闭环
  - prune5 官方整目录测速
  - prune5 本地代理复核
- 仍待补齐：
  - `AIPP` 正式下沉收益表
  - `INT8/PTQ/QAT` 正式交付链
  - 真正能缩小 dense 维度的结构裁剪版本
