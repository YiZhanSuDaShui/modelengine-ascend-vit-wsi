# BACH 数据集目录结构

```
data/BACH/
├── ICIAR2018_BACH_Challenge/           # 训练/开发集（有标签）
│   ├── Photos/                         # 显微镜图像 (400 images)
│   │   ├── Benign/                     # 良性 (100 images)
│   │   ├── InSitu/                     # 原位癌 (100 images)
│   │   ├── Invasive/                   # 浸润癌 (100 images)
│   │   └── Normal/                     # 正常 (100 images)
│   ├── WSI/                            # 整张幻灯片图像
│   │   ├── thumbnails/                 # 缩略图
│   │   └── gt_thumbnails/              # Ground truth 缩略图
│   └── Photos/microscopy_ground_truth.csv  # 标签文件
│
├── ICIAR2018_BACH_Challenge_TestDataset/  # 测试集（无标签）
│   ├── Photos/                         # 显微镜图像 (100 images)
│   │   └── test*.tif (0-99)
│   └── WSI/                            # 整张幻灯片图像
│       └── thumbnails/                 # 缩略图
│
└── derived/                            # 派生产物（可删除重建）
    ├── split/
    │   ├── photos_train_all.txt        # 训练图片列表
    │   └── photos_test_all.txt         # 测试图片列表 (100 images)
    ├── tiles/                          # 切块缓存
    └── cache/                          # 其他缓存
```

## 数据统计

| 类别 | 数量 |
|------|------|
| 训练 Photos (Benign) | 100 |
| 训练 Photos (InSitu) | 100 |
| 训练 Photos (Invasive) | 100 |
| 训练 Photos (Normal) | 100 |
| 测试 Photos | 100 |
| **总计** | **500** |

## 关键文件

- `derived/split/photos_test_all.txt` - 100 张测试图片路径
- `derived/split/photos_train_all.txt` - 训练图片列表
- `ICIAR2018_BACH_Challenge/Photos/microscopy_ground_truth.csv` - 训练集标签
