# 赛题三：从“工程已完成”到“精度优化”的详细落地方案

## 一、总策略

你现在已经完成了：

- A1 纯推理基准
- A2 端到端测速
- sweep 找 CPU/工程最优配置

接下来重点不再是“跑快”，而是：

1. **先把 patch encoder 训练稳**
2. **再把 WSI 的多标签 MIL 跑稳**
3. **最后再回到部署和导出**

---

## 二、数据怎么用

### 1. Photos（400 张）——主监督源

用途：
- 训练 patch encoder
- 做 4分类主监督
- 作为最稳定的精度基线

不建议：
- 直接忽略
- 只把它当“辅助数据”

### 2. A01~A10.svs + xml —— WSI 监督桥梁

用途：
- 生成监督 tile
- 建 slide-level 多标签标签
- 训练 MIL
- 做 attention / 热图可解释性

### 3. 01~20.svs —— 第二阶段再考虑

用途（第二版再上）：
- 自监督 / 伪标签
- stain 统计
- 难负样本
- 半监督

### 4. Test WSI —— 最终推理

用途：
- 端到端 WSI 推理
- 最终 CSV 导出

---

## 三、推荐训练顺序

### Stage 1：Photos encoder 4分类

目标：
- 学到能迁移到 WSI tile 的病理形态表征

### Stage 1.5：A01~A10 监督 tile 再适配

目标：
- 让 encoder 适应 WSI 域

### Stage 2：提 WSI tile 特征

目标：
- 为 MIL 准备 bag features

### Stage 3：多标签 MIL

目标：
- slide-level 输出多标签分数

### Stage 4：阈值校准 + 导出

目标：
- 决定每一类的最终阈值
- 输出最终 test CSV

---

## 四、关键判断

### `Normal` 要不要保留为 slide-level 标签？

建议训练后看验证集：

- 如果 `Normal` 几乎总是阳性：改成 fallback
- 如果 `Normal` 与病变区域确实可分：保留四标签

### `01~20.svs` 要不要第一版就用？

不建议。

第一版先闭环：
- Photos → patch classifier
- A01~A10 → supervised bag
- Test WSI → infer

---

## 五、你最该看的 4 个指标

### Photos
- ACC
- macro F1
- 每类召回率
- OVR AUC

### WSI
- 每类 AUROC
- 每类 F1
- mAP
- exact match / sample F1

---

## 六、实验优先级

### P0
- 跑通端到端闭环
- 有可复现 best checkpoint

### P1
- 加入 A01~A10 监督 tile
- 调 level / tile_size / step

### P2
- 加 class-wise attention
- 阈值搜索
- TTA / multi-crop

### P3
- 半监督 / 伪标签 / 01~20.svs
- ONNX/OM/INT8

