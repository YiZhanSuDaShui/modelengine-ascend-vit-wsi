#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================
# 逐行注释版：A2_final_opt_v3.py（global-mix batching 端到端测速）
# 说明：本文件由原脚本自动加注释生成；注释以“上一行 # ...”形式给出，
#      避免在代码行末追加注释导致续行符/括号续行等语法风险。
# =============================================================


"""
A2_final_opt_v3.py
==================
A2 端到端（IO/解码/切块/预处理/H2D/NPU推理/落盘TSV）性能评测脚本。

相比旧版“单图口径（一张图内分 batch）”，本版本新增：
✅ global-mix batching（跨图片混 batch），让 bs=96 真正吃满（适配 BACH Photos 单图仅 ~54 tiles 的场景）。

核心思想：
- decode+tile 仍以“图片”为单位并行预取（decode_workers + decode_prefetch）
- 推理 batch 以“全局 tile 池”为单位从多张图混合抽取，凑满 bs
- 推理完成后，按 tile 的 img_id 回填到各自图片的统计器里
- 输出仍保持 per-image TSV（便于复现与排查），summary_json 仍输出 overall/steady tile/s

用法示例：
  python A2_final_opt_v2.py \
    --file_list data/BACH/derived/split/photos_test_all.txt \
    --ckpt_dir assets/ckpts/UNI \
    --bs 96 --global_mix 1 --cpu_workers 12 --prefetch 4 \
    --white_thr 1.01 --precision fp16 \
    --out logs/A2_gmix.tsv --summary_json logs/A2_gmix.json

说明：
- 默认 global_mix=1（可用 --global_mix 0 退回旧的单图口径流水）
- 为了保持可解释的“sync_wait_ms”（NPU 等待时间）口径：
  * 仍采用“最多 1 个 inflight”的 NPU 提交方式：提交下一批前先 synchronize 等上一批完成
  * sync_wait_ms 分摊计入“上一批”包含的图片（更符合语义：我们等的是上一批算完）
"""

# 导入模块：import os
import os
# 导入模块：import time
import time
# 导入模块：import csv
import csv
# 导入模块：import json
import json
# 导入模块：import argparse
import argparse
# 导入依赖：from collections import deque
from collections import deque
# 导入依赖：from dataclasses import dataclass
from dataclasses import dataclass
# 导入依赖：from typing import Deque, List, Tuple, Optional, Dict
from typing import Deque, List, Tuple, Optional, Dict

# 导入模块：import numpy as np
import numpy as np
# 导入依赖：from PIL import Image
from PIL import Image

# 导入模块：import torch
import torch
# 导入模块：import torch_npu
import torch_npu
# 导入模块：import timm
import timm


# ----------------------------
# 0) 计时 & NPU 同步
# ----------------------------

# 定义函数：now
def now() -> float:
    """高精度计时（秒）"""
    # 返回结果
    return time.perf_counter()


# 定义函数：sync_npu
def sync_npu() -> None:
    """
    NPU 异步：forward 返回不代表算完。
    精准计时必须在关键点 synchronize。
    """
    # 语句：torch_npu.npu.synchronize()
    torch_npu.npu.synchronize()


# ----------------------------
# 1) 构建 UNI backbone（与你 test_uni_npu.py 保持一致）
# ----------------------------

# 定义函数：build_uni_backbone
def build_uni_backbone(ckpt_dir: str,
                       # 语句：device: str = "npu:0",
                       device: str = "npu:0",
                       # 语句：img_size: int = 224) -> torch.nn.Module:
                       img_size: int = 224) -> torch.nn.Module:
    """
    ViT-L/16 224，num_classes=0，只输出特征。
    """
    # 赋值/初始化：ckpt_path
    ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    # 条件判断：if not os.path.exists(ckpt_path):
    if not os.path.exists(ckpt_path):
        # 抛出异常
        raise FileNotFoundError(f"[UNI] missing checkpoint: {ckpt_path}")

    # 赋值/初始化：dev
    dev = torch.device(device)
    # 语句：torch.npu.set_device(dev)
    torch.npu.set_device(dev)

    # 赋值/初始化：model
    model = timm.create_model(
        # 语句："vit_large_patch16_224",
        "vit_large_patch16_224",
        # 赋值/初始化：img_size
        img_size=img_size,
        # 赋值/初始化：patch_size
        patch_size=16,
        # 赋值/初始化：init_values
        init_values=1e-5,
        # 赋值/初始化：num_classes
        num_classes=0,
        # 赋值/初始化：dynamic_img_size
        dynamic_img_size=True,
    # 语句：)
    )

    # 赋值/初始化：state
    state = torch.load(ckpt_path, map_location="cpu")
    # 语句：model.load_state_dict(state, strict=True)
    model.load_state_dict(state, strict=True)

    # 语句：model.eval().to(dev)
    model.eval().to(dev)
    # 返回结果
    return model


# ----------------------------
# 2) fast_grid：stride==tile 时最快切块（避免 Python for 循环）
# ----------------------------

# 定义函数：extract_tiles_fast_grid
def extract_tiles_fast_grid(img_np: np.ndarray,
                            # 语句：tile: int,
                            tile: int,
                            # 语句：stride: int) -> Tuple[np.ndarray, int]:
                            stride: int) -> Tuple[np.ndarray, int]:
    """
    stride==tile 时，用 reshape/transpose 一次生成 tiles。
    返回：
    - tiles_uint8: (N, tile, tile, 3)
    - total_tiles_possible: N（未过滤前）
    """
    # 断言：assert stride == tile, "fast_grid 仅支持 stride==tile"
    assert stride == tile, "fast_grid 仅支持 stride==tile"

    # 赋值/初始化：H, W, C
    H, W, C = img_np.shape
    # 条件判断：if C != 3:
    if C != 3:
        # 抛出异常
        raise ValueError(f"expect RGB image with 3 channels, got shape={img_np.shape}")

    # 赋值/初始化：nx
    nx = W // tile
    # 赋值/初始化：ny
    ny = H // tile
    # 条件判断：if nx <= 0 or ny <= 0:
    if nx <= 0 or ny <= 0:
        # 返回结果
        return np.zeros((0, tile, tile, 3), dtype=np.uint8), 0

    # 赋值/初始化：H_use
    H_use = ny * tile
    # 赋值/初始化：W_use
    W_use = nx * tile
    # 赋值/初始化：crop
    crop = img_np[:H_use, :W_use, :]

    # 赋值/初始化：tiles
    tiles = crop.reshape(ny, tile, nx, tile, 3).transpose(0, 2, 1, 3, 4)
    # 赋值/初始化：tiles
    tiles = tiles.reshape(ny * nx, tile, tile, 3)

    # 返回结果
    return tiles, ny * nx


# 定义函数：tissue_mask_uint8
def tissue_mask_uint8(tiles_uint8: np.ndarray,
                      # 语句：white_thr: float = 0.8,
                      white_thr: float = 0.8,
                      # 语句：gray_high: float = 0.9) -> np.ndarray:
                      gray_high: float = 0.9) -> np.ndarray:
    """
    极简白底过滤：
    - gray = mean(RGB)/255
    - white_ratio = (gray > gray_high).mean()
    - white_ratio < white_thr => 保留
    """
    # 赋值/初始化：gray
    gray = tiles_uint8.mean(axis=3) / 255.0
    # 赋值/初始化：white_ratio
    white_ratio = (gray > gray_high).mean(axis=(1, 2))
    # 赋值/初始化：keep
    keep = white_ratio < white_thr
    # 返回结果
    return keep


# ----------------------------
# 3) 向量化 preprocess：uint8 tiles -> NCHW tensor
# ----------------------------

# 定义函数：preprocess_tiles_uint8_vectorized
def preprocess_tiles_uint8_vectorized(tiles_uint8: np.ndarray,
                                      # 语句：out_fp16: bool = True,
                                      out_fp16: bool = True,
                                      # 语句：mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                                      mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                                      # 语句：std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                                      std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                                      # 语句：) -> torch.Tensor:
                                      ) -> torch.Tensor:
    """
    (B,H,W,3) uint8 -> torch tensor (B,3,H,W)
    1) /255 -> [0,1]
    2) (x-mean)/std
    3) NHWC->NCHW
    4) 可选转 fp16（减少 H2D 传输）
    """
    # 赋值/初始化：x
    x = tiles_uint8.astype(np.float32) / 255.0

    # 赋值/初始化：m
    m = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    # 赋值/初始化：s
    s = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    # 赋值/初始化：x
    x = (x - m) / s

    # 赋值/初始化：x
    x = np.transpose(x, (0, 3, 1, 2))
    # 赋值/初始化：x_cpu
    x_cpu = torch.from_numpy(x)

    # 条件判断：if out_fp16:
    if out_fp16:
        # 赋值/初始化：x_cpu
        x_cpu = x_cpu.half()

    # 返回结果
    return x_cpu.contiguous()


# ----------------------------
# 4) decode + tile（并行 worker）
# ----------------------------

# 装饰器：@dataclass
@dataclass
# 定义类/数据结构：DecodeTileResult
class DecodeTileResult:
    # 语句：file: str
    file: str
    # 语句：W: int
    W: int
    # 语句：H: int
    H: int
    # 语句：mode: str
    mode: str
    # 语句：total_tiles_possible: int
    total_tiles_possible: int
    # 语句：tiles_uint8: np.ndarray
    tiles_uint8: np.ndarray
    # 语句：decode_compute_ms: float
    decode_compute_ms: float
    # 语句：tile_compute_ms: float
    tile_compute_ms: float


# 定义函数：decode_and_tile_worker
def decode_and_tile_worker(path: str,
                           # 语句：tile: int,
                           tile: int,
                           # 语句：stride: int,
                           stride: int,
                           # 语句：white_thr: float) -> DecodeTileResult:
                           white_thr: float) -> DecodeTileResult:
    """
    线程池 worker：
    - decode (PIL)
    - tilegen (fast_grid)
    - 可选白底过滤（white_thr<1.0 才做；你现在 1.01 直接跳过）
    """
    # 赋值/初始化：t0
    t0 = now()

    # 上下文管理（自动关闭资源）：with Image.open(path) as im:
    with Image.open(path) as im:
        # 赋值/初始化：im
        im = im.convert("RGB")
        # 赋值/初始化：img_np
        img_np = np.array(im)

    # 赋值/初始化：t1
    t1 = now()

    # 条件判断：if stride == tile:
    if stride == tile:
        # 赋值/初始化：mode
        mode = "fast_grid"
        # 赋值/初始化：tiles_uint8, total
        tiles_uint8, total = extract_tiles_fast_grid(img_np, tile=tile, stride=stride)
        # 条件判断：if total > 0 and white_thr < 1.0:
        if total > 0 and white_thr < 1.0:
            # 赋值/初始化：keep
            keep = tissue_mask_uint8(tiles_uint8, white_thr=white_thr)
            # 赋值/初始化：tiles_uint8
            tiles_uint8 = tiles_uint8[keep]
    # 条件分支：else
    else:
        # 赋值/初始化：mode
        mode = "fallback"
        # 赋值/初始化：H, W, _
        H, W, _ = img_np.shape
        # 赋值/初始化：coords
        coords = []
        # 循环：for y in range(0, H - tile + 1, stride):
        for y in range(0, H - tile + 1, stride):
            # 循环：for x in range(0, W - tile + 1, stride):
            for x in range(0, W - tile + 1, stride):
                # 语句：coords.append((x, y))
                coords.append((x, y))
        # 赋值/初始化：total
        total = len(coords)

        # 赋值/初始化：tiles_list
        tiles_list = []
        # 循环：for (x, y) in coords:
        for (x, y) in coords:
            # 语句：tiles_list.append(img_np[y:y + tile, x:x + tile])
            tiles_list.append(img_np[y:y + tile, x:x + tile])
        # 赋值/初始化：tiles_uint8
        tiles_uint8 = np.stack(tiles_list, axis=0).astype(np.uint8) if total > 0 else \
            # 语句：np.zeros((0, tile, tile, 3), dtype=np.uint8)
            np.zeros((0, tile, tile, 3), dtype=np.uint8)

        # 条件判断：if total > 0 and white_thr < 1.0:
        if total > 0 and white_thr < 1.0:
            # 赋值/初始化：keep
            keep = tissue_mask_uint8(tiles_uint8, white_thr=white_thr)
            # 赋值/初始化：tiles_uint8
            tiles_uint8 = tiles_uint8[keep]

    # 赋值/初始化：t2
    t2 = now()

    # 赋值/初始化：H0, W0, _
    H0, W0, _ = img_np.shape
    # 返回结果
    return DecodeTileResult(
        # 赋值/初始化：file
        file=os.path.basename(path),
        # 赋值/初始化：W
        W=W0,
        # 赋值/初始化：H
        H=H0,
        # 赋值/初始化：mode
        mode=mode,
        # 赋值/初始化：total_tiles_possible
        total_tiles_possible=total,
        # 赋值/初始化：tiles_uint8
        tiles_uint8=tiles_uint8,
        # 赋值/初始化：decode_compute_ms
        decode_compute_ms=(t1 - t0) * 1000.0,
        # 赋值/初始化：tile_compute_ms
        tile_compute_ms=(t2 - t1) * 1000.0,
    # 语句：)
    )


# ----------------------------
# 5) global-mix 数据结构
# ----------------------------

# 装饰器：@dataclass(frozen=True)
@dataclass(frozen=True)
# 定义类/数据结构：TileRef
class TileRef:
    # 语句：img_idx: int
    img_idx: int
    # 语句：tile_idx: int
    tile_idx: int


# 装饰器：@dataclass
@dataclass
# 定义类/数据结构：ImageState
class ImageState:
    # 语句：idx: int
    idx: int
    # 语句：file: str = ""
    file: str = ""
    # 语句：W: int = 0
    W: int = 0
    # 语句：H: int = 0
    H: int = 0
    # 语句：mode: str = ""
    mode: str = ""
    # 语句：total_tiles_possible: int = 0
    total_tiles_possible: int = 0
    # 语句：valid_tiles: int = 0
    valid_tiles: int = 0

    # decode/tile
    # 语句：decode_compute_ms: float = 0.0
    decode_compute_ms: float = 0.0
    # 语句：tile_compute_ms: float = 0.0
    tile_compute_ms: float = 0.0
    # 语句：decode_wait_ms: float = 0.0
    decode_wait_ms: float = 0.0

    # 分摊统计（按 tile 数比例分摊到 image）
    # 语句：pre_ms: float = 0.0
    pre_ms: float = 0.0
    # 语句：submit_ms: float = 0.0
    submit_ms: float = 0.0
    # 语句：sync_wait_ms: float = 0.0
    sync_wait_ms: float = 0.0

    # global-mix 计数
    # 语句：processed_tiles: int = 0
    processed_tiles: int = 0
    # 语句：num_batches: int = 0
    num_batches: int = 0

    # 资源（保持 tiles 数组引用，直到该图片处理完成）
    # 语句：tiles_uint8: Optional[np.ndarray] = None
    tiles_uint8: Optional[np.ndarray] = None

    # 语句：done: bool = False
    done: bool = False


# 定义函数：preprocess_mixed_batch
def preprocess_mixed_batch(tile_refs: List[TileRef],
                           # 语句：image_states: List[ImageState],
                           image_states: List[ImageState],
                           # 语句：out_fp16: bool) -> Tuple[torch.Tensor, float]:
                           out_fp16: bool) -> Tuple[torch.Tensor, float]:
    """
    在线程池里：
    - 根据 tile_refs 把 tile 拼成 (B,H,W,3) uint8
    - 向量化 preprocess -> torch tensor
    返回：
    - x_cpu
    - pre_ms
    """
    # 赋值/初始化：t0
    t0 = now()
    # 赋值/初始化：tiles
    tiles = []
    # 循环：for ref in tile_refs:
    for ref in tile_refs:
        # 赋值/初始化：arr
        arr = image_states[ref.img_idx].tiles_uint8
        # 条件判断：if arr is None:
        if arr is None:
            # 抛出异常
            raise RuntimeError(f"tiles_uint8 already freed for img_idx={ref.img_idx}")
        # 语句：tiles.append(arr[ref.tile_idx])
        tiles.append(arr[ref.tile_idx])
    # 赋值/初始化：batch_uint8
    batch_uint8 = np.stack(tiles, axis=0) if tiles else np.zeros((0, 224, 224, 3), dtype=np.uint8)

    # 赋值/初始化：x_cpu
    x_cpu = preprocess_tiles_uint8_vectorized(batch_uint8, out_fp16=out_fp16)
    # 赋值/初始化：t1
    t1 = now()
    # 返回结果
    return x_cpu, (t1 - t0) * 1000.0


# ----------------------------
# 6) 单图口径（旧逻辑）保持不变：便于回退对比
# ----------------------------

# 定义函数：run_per_image
def run_per_image(args: argparse.Namespace,
                  # 语句：files: List[str],
                  files: List[str],
                  # 语句：model: torch.nn.Module,
                  model: torch.nn.Module,
                  # 语句：device: str,
                  device: str,
                  # 语句：use_amp: bool,
                  use_amp: bool,
                  # 语句：warmup_pending: bool,
                  warmup_pending: bool,
                  # 语句：warmup_sync_ms: float) -> Dict:
                  warmup_sync_ms: float) -> Dict:
    """
    旧版：逐图消费 decode->图内 batch 推理。
    返回 summary dict（result 部分），并且会落盘 per-image TSV。
    """
    # 语句：os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 赋值/初始化：header
    header = [
        # 语句："file", "W", "H", "mode",
        "file", "W", "H", "mode",
        # 语句："total_tiles_possible", "valid_tiles", "num_batches",
        "total_tiles_possible", "valid_tiles", "num_batches",
        # 语句："t_decode_compute_ms", "t_tile_compute_ms", "t_decode_wait_ms",
        "t_decode_compute_ms", "t_tile_compute_ms", "t_decode_wait_ms",
        # 语句："t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        "t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        # 语句："t_total_ms",
        "t_total_ms",
        # 语句："tile_per_s", "img_per_s",
        "tile_per_s", "img_per_s",
        # 语句："avg_ms_per_tile_total", "pre_ms_per_tile",
        "avg_ms_per_tile_total", "pre_ms_per_tile",
        # 语句："note"
        "note"
    # 语句：]
    ]

    # 赋值/初始化：total_valid_tiles
    total_valid_tiles = 0
    # 赋值/初始化：total_images
    total_images = 0
    # 赋值/初始化：steady_valid_tiles
    steady_valid_tiles = 0
    # 赋值/初始化：steady_images
    steady_images = 0
    # 赋值/初始化：steady_time_s
    steady_time_s = 0.0

    # 赋值/初始化：sum_decode_compute_ms
    sum_decode_compute_ms = 0.0
    # 赋值/初始化：sum_tile_compute_ms
    sum_tile_compute_ms = 0.0
    # 赋值/初始化：sum_decode_wait_ms
    sum_decode_wait_ms = 0.0
    # 赋值/初始化：sum_pre_ms
    sum_pre_ms = 0.0
    # 赋值/初始化：sum_submit_ms
    sum_submit_ms = 0.0
    # 赋值/初始化：sum_sync_wait_ms
    sum_sync_wait_ms = 0.0
    # 赋值/初始化：sum_total_ms
    sum_total_ms = 0.0

    # 语句：cold_first_total_ms: Optional[float] = None
    cold_first_total_ms: Optional[float] = None

    # 导入依赖：from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    # 赋值/初始化：decode_prefetch
    decode_prefetch = max(1, min(args.decode_prefetch, len(files)))

    # 赋值/初始化：t_all0
    t_all0 = now()

    # 上下文管理（自动关闭资源）：with ThreadPoolExecutor(max_workers=args._decode_workers) as decode_ex, \
    with ThreadPoolExecutor(max_workers=args._decode_workers) as decode_ex, \
            # 语句：ThreadPoolExecutor(max_workers=args._pre_workers) as pre_ex, \
            ThreadPoolExecutor(max_workers=args._pre_workers) as pre_ex, \
            # 语句：open(args.out, "w", newline="", encoding="utf-8") as f_out:
            open(args.out, "w", newline="", encoding="utf-8") as f_out:

        # 赋值/初始化：writer
        writer = csv.writer(f_out, delimiter="\t")
        # 语句：writer.writerow(header)
        writer.writerow(header)
        # 语句：f_out.flush()
        f_out.flush()

        # 语句：decode_q: Deque = deque()
        decode_q: Deque = deque()
        # 赋值/初始化：next_decode_idx
        next_decode_idx = 0

        # 循环：for _ in range(decode_prefetch):
        for _ in range(decode_prefetch):
            # 赋值/初始化：path
            path = files[next_decode_idx]
            # 赋值/初始化：fut
            fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
            # 语句：decode_q.append(fut)
            decode_q.append(fut)
            # 语句：next_decode_idx += 1
            next_decode_idx += 1

        # 循环：for img_i in range(len(files)):
        for img_i in range(len(files)):
            # 赋值/初始化：img_start
            img_start = now()

            # 赋值/初始化：w0
            w0 = now()
            # 语句：dec_res: DecodeTileResult = decode_q.popleft().result()
            dec_res: DecodeTileResult = decode_q.popleft().result()
            # 赋值/初始化：w1
            w1 = now()
            # 赋值/初始化：decode_wait_ms
            decode_wait_ms = (w1 - w0) * 1000.0

            # 条件判断：if next_decode_idx < len(files):
            if next_decode_idx < len(files):
                # 赋值/初始化：path
                path = files[next_decode_idx]
                # 赋值/初始化：fut
                fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
                # 语句：decode_q.append(fut)
                decode_q.append(fut)
                # 语句：next_decode_idx += 1
                next_decode_idx += 1

            # 赋值/初始化：tiles_uint8
            tiles_uint8 = dec_res.tiles_uint8
            # 赋值/初始化：valid_tiles
            valid_tiles = int(tiles_uint8.shape[0])
            # 赋值/初始化：total_tiles_possible
            total_tiles_possible = int(dec_res.total_tiles_possible)

            # 赋值/初始化：bs
            bs = max(1, args.bs)
            # 赋值/初始化：num_batches
            num_batches = int((valid_tiles + bs - 1) // bs) if valid_tiles > 0 else 0

            # 赋值/初始化：t_pre_ms
            t_pre_ms = 0.0
            # 语句：pre_q: Deque = deque()
            pre_q: Deque = deque()

            # 定义函数：submit_pre_batch
            def submit_pre_batch(start: int, end: int):
                # 赋值/初始化：sub
                sub = tiles_uint8[start:end]
                # 赋值/初始化：t0
                t0 = now()
                # 赋值/初始化：x_cpu
                x_cpu = preprocess_tiles_uint8_vectorized(sub, out_fp16=args.cpu_out_fp16)
                # 赋值/初始化：t1
                t1 = now()
                # 返回结果
                return x_cpu, (t1 - t0) * 1000.0

            # 赋值/初始化：next_b
            next_b = 0
            # 赋值/初始化：max_prefetch
            max_prefetch = max(1, args.prefetch)

            # 循环：while next_b < num_batches and len(pre_q) < max_prefetch:
            while next_b < num_batches and len(pre_q) < max_prefetch:
                # 赋值/初始化：b_start
                b_start = next_b * bs
                # 赋值/初始化：b_end
                b_end = min(valid_tiles, (next_b + 1) * bs)
                # 赋值/初始化：fut
                fut = pre_ex.submit(submit_pre_batch, b_start, b_end)
                # 语句：pre_q.append(fut)
                pre_q.append(fut)
                # 语句：next_b += 1
                next_b += 1

            # 赋值/初始化：inflight
            inflight = False
            # 赋值/初始化：t_submit_ms
            t_submit_ms = 0.0
            # 赋值/初始化：t_sync_wait_ms
            t_sync_wait_ms = 0.0
            # 语句：note_parts: List[str] = []
            note_parts: List[str] = []
            # 条件判断：if args.global_mix:
            if args.global_mix:
                # 语句：note_parts.append("global_mix=1")
                note_parts.append("global_mix=1")

            # 循环：for _ in range(num_batches):
            for _ in range(num_batches):
                # 赋值/初始化：fut
                fut = pre_q.popleft()
                # 赋值/初始化：x_cpu, pre_ms
                x_cpu, pre_ms = fut.result()
                # 语句：t_pre_ms += pre_ms
                t_pre_ms += pre_ms

                # 循环：while next_b < num_batches and len(pre_q) < max_prefetch:
                while next_b < num_batches and len(pre_q) < max_prefetch:
                    # 赋值/初始化：b_start
                    b_start = next_b * bs
                    # 赋值/初始化：b_end
                    b_end = min(valid_tiles, (next_b + 1) * bs)
                    # 赋值/初始化：fut2
                    fut2 = pre_ex.submit(submit_pre_batch, b_start, b_end)
                    # 语句：pre_q.append(fut2)
                    pre_q.append(fut2)
                    # 语句：next_b += 1
                    next_b += 1

                # 条件判断：if inflight:
                if inflight:
                    # 赋值/初始化：sw0
                    sw0 = now()
                    # 语句：sync_npu()
                    sync_npu()
                    # 赋值/初始化：sw1
                    sw1 = now()
                    # 语句：t_sync_wait_ms += (sw1 - sw0) * 1000.0
                    t_sync_wait_ms += (sw1 - sw0) * 1000.0
                    # 赋值/初始化：inflight
                    inflight = False

                # 条件判断：if warmup_pending:
                if warmup_pending:
                    # 赋值/初始化：sw0
                    sw0 = now()
                    # 语句：sync_npu()
                    sync_npu()
                    # 赋值/初始化：sw1
                    sw1 = now()
                    # 赋值/初始化：warmup_sync_ms
                    warmup_sync_ms = (sw1 - sw0) * 1000.0
                    # 赋值/初始化：warmup_pending
                    warmup_pending = False
                    # 语句：note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")
                    note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")

                # 赋值/初始化：sb0
                sb0 = now()
                # 赋值/初始化：x
                x = x_cpu.to(device, non_blocking=True)
                # 条件判断：if use_amp:
                if use_amp:
                    # 上下文管理（自动关闭资源）：with torch.npu.amp.autocast():
                    with torch.npu.amp.autocast():
                        # 赋值/初始化：_
                        _ = model(x)
                # 条件分支：else
                else:
                    # 赋值/初始化：_
                    _ = model(x)
                # 赋值/初始化：sb1
                sb1 = now()
                # 语句：t_submit_ms += (sb1 - sb0) * 1000.0
                t_submit_ms += (sb1 - sb0) * 1000.0
                # 赋值/初始化：inflight
                inflight = True

            # 条件判断：if inflight:
            if inflight:
                # 赋值/初始化：sw0
                sw0 = now()
                # 语句：sync_npu()
                sync_npu()
                # 赋值/初始化：sw1
                sw1 = now()
                # 语句：t_sync_wait_ms += (sw1 - sw0) * 1000.0
                t_sync_wait_ms += (sw1 - sw0) * 1000.0
                # 赋值/初始化：inflight
                inflight = False

            # 赋值/初始化：img_end
            img_end = now()
            # 赋值/初始化：t_total_ms
            t_total_ms = (img_end - img_start) * 1000.0

            # 赋值/初始化：tile_per_s
            tile_per_s = (valid_tiles / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            # 赋值/初始化：img_per_s
            img_per_s = (1.0 / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            # 赋值/初始化：avg_ms_per_tile_total
            avg_ms_per_tile_total = (t_total_ms / valid_tiles) if valid_tiles > 0 else 0.0
            # 赋值/初始化：pre_ms_per_tile
            pre_ms_per_tile = (t_pre_ms / valid_tiles) if valid_tiles > 0 else 0.0

            # 赋值/初始化：note
            note = ";".join(note_parts)

            # 语句：writer.writerow([
            writer.writerow([
                # 语句：dec_res.file, dec_res.W, dec_res.H, dec_res.mode,
                dec_res.file, dec_res.W, dec_res.H, dec_res.mode,
                # 语句：total_tiles_possible, valid_tiles, num_batches,
                total_tiles_possible, valid_tiles, num_batches,
                # 语句：f"{dec_res.decode_compute_ms:.3f}", f"{dec_res.tile_compute_ms:.3f}", f"{decode…
                f"{dec_res.decode_compute_ms:.3f}", f"{dec_res.tile_compute_ms:.3f}", f"{decode_wait_ms:.3f}",
                # 语句：f"{t_pre_ms:.3f}", f"{t_submit_ms:.3f}", f"{t_sync_wait_ms:.3f}",
                f"{t_pre_ms:.3f}", f"{t_submit_ms:.3f}", f"{t_sync_wait_ms:.3f}",
                # 语句：f"{t_total_ms:.3f}",
                f"{t_total_ms:.3f}",
                # 语句：f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                # 语句：f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                # 语句：note
                note
            # 语句：])
            ])
            # 语句：f_out.flush()
            f_out.flush()

            # 条件判断：if (img_i + 1) % args.log_every == 0 or (img_i + 1) == len(files):
            if (img_i + 1) % args.log_every == 0 or (img_i + 1) == len(files):
                # 打印进度/汇总信息
                print(
                    # 语句：f"[A2_v2] [{img_i+1}/{len(files)}] {dec_res.file} "
                    f"[A2_v2] [{img_i+1}/{len(files)}] {dec_res.file} "
                    # 语句：f"tiles={valid_tiles} total_ms={t_total_ms:.1f} tile/s={tile_per_s:.1f} "
                    f"tiles={valid_tiles} total_ms={t_total_ms:.1f} tile/s={tile_per_s:.1f} "
                    # 语句：f"decode_wait_ms={decode_wait_ms:.1f} pre_ms={t_pre_ms:.1f} "
                    f"decode_wait_ms={decode_wait_ms:.1f} pre_ms={t_pre_ms:.1f} "
                    # 语句：f"submit_ms={t_submit_ms:.1f} sync_wait_ms={t_sync_wait_ms:.1f}"
                    f"submit_ms={t_submit_ms:.1f} sync_wait_ms={t_sync_wait_ms:.1f}"
                # 语句：)
                )

            # 语句：total_valid_tiles += valid_tiles
            total_valid_tiles += valid_tiles
            # 语句：total_images += 1
            total_images += 1

            # 语句：sum_decode_compute_ms += dec_res.decode_compute_ms
            sum_decode_compute_ms += dec_res.decode_compute_ms
            # 语句：sum_tile_compute_ms += dec_res.tile_compute_ms
            sum_tile_compute_ms += dec_res.tile_compute_ms
            # 语句：sum_decode_wait_ms += decode_wait_ms
            sum_decode_wait_ms += decode_wait_ms
            # 语句：sum_pre_ms += t_pre_ms
            sum_pre_ms += t_pre_ms
            # 语句：sum_submit_ms += t_submit_ms
            sum_submit_ms += t_submit_ms
            # 语句：sum_sync_wait_ms += t_sync_wait_ms
            sum_sync_wait_ms += t_sync_wait_ms
            # 语句：sum_total_ms += t_total_ms
            sum_total_ms += t_total_ms

            # 条件判断：if cold_first_total_ms is None:
            if cold_first_total_ms is None:
                # 赋值/初始化：cold_first_total_ms
                cold_first_total_ms = t_total_ms

            # 条件判断：if img_i >= args.skip_steady:
            if img_i >= args.skip_steady:
                # 语句：steady_valid_tiles += valid_tiles
                steady_valid_tiles += valid_tiles
                # 语句：steady_images += 1
                steady_images += 1
                # 语句：steady_time_s += (t_total_ms / 1000.0)
                steady_time_s += (t_total_ms / 1000.0)

    # 赋值/初始化：t_all1
    t_all1 = now()
    # 赋值/初始化：total_time_s
    total_time_s = (t_all1 - t_all0)

    # 赋值/初始化：overall_tile_s
    overall_tile_s = total_valid_tiles / total_time_s
    # 赋值/初始化：overall_img_s
    overall_img_s = total_images / total_time_s

    # 赋值/初始化：steady_tile_s
    steady_tile_s = (steady_valid_tiles / steady_time_s) if steady_time_s > 0 else 0.0
    # 赋值/初始化：steady_img_s
    steady_img_s = (steady_images / steady_time_s) if steady_time_s > 0 else 0.0

    # 定义函数：safe_div
    def safe_div(a: float, b: int) -> float:
        # 返回结果
        return a / b if b > 0 else 0.0

    # 赋值/初始化：mean_decode_compute_ms
    mean_decode_compute_ms = safe_div(sum_decode_compute_ms, total_images)
    # 赋值/初始化：mean_tile_compute_ms
    mean_tile_compute_ms = safe_div(sum_tile_compute_ms, total_images)
    # 赋值/初始化：mean_decode_wait_ms
    mean_decode_wait_ms = safe_div(sum_decode_wait_ms, total_images)
    # 赋值/初始化：mean_pre_ms
    mean_pre_ms = safe_div(sum_pre_ms, total_images)
    # 赋值/初始化：mean_submit_ms
    mean_submit_ms = safe_div(sum_submit_ms, total_images)
    # 赋值/初始化：mean_sync_wait_ms
    mean_sync_wait_ms = safe_div(sum_sync_wait_ms, total_images)
    # 赋值/初始化：mean_total_ms
    mean_total_ms = safe_div(sum_total_ms, total_images)

    # 返回结果
    return {
        # 语句："tiles_total": total_valid_tiles,
        "tiles_total": total_valid_tiles,
        # 语句："num_images": total_images,
        "num_images": total_images,
        # 语句："total_time_s": total_time_s,
        "total_time_s": total_time_s,
        # 语句："overall_tile_s": overall_tile_s,
        "overall_tile_s": overall_tile_s,
        # 语句："overall_img_s": overall_img_s,
        "overall_img_s": overall_img_s,
        # 语句："steady_tile_s": steady_tile_s,
        "steady_tile_s": steady_tile_s,
        # 语句："steady_img_s": steady_img_s,
        "steady_img_s": steady_img_s,
        # 语句："cold_first_ms": float(cold_first_total_ms or 0.0),
        "cold_first_ms": float(cold_first_total_ms or 0.0),
        # 语句："warmup_sync_ms": warmup_sync_ms,
        "warmup_sync_ms": warmup_sync_ms,
        # 语句："mean_decode_compute_ms": mean_decode_compute_ms,
        "mean_decode_compute_ms": mean_decode_compute_ms,
        # 语句："mean_tile_compute_ms": mean_tile_compute_ms,
        "mean_tile_compute_ms": mean_tile_compute_ms,
        # 语句："mean_decode_wait_ms": mean_decode_wait_ms,
        "mean_decode_wait_ms": mean_decode_wait_ms,
        # 语句："mean_pre_ms": mean_pre_ms,
        "mean_pre_ms": mean_pre_ms,
        # 语句："mean_submit_ms": mean_submit_ms,
        "mean_submit_ms": mean_submit_ms,
        # 语句："mean_sync_wait_ms": mean_sync_wait_ms,
        "mean_sync_wait_ms": mean_sync_wait_ms,
        # 语句："mean_total_ms": mean_total_ms,
        "mean_total_ms": mean_total_ms,
    # 语句：}
    }


# ----------------------------
# 7) global-mix 主流程
# ----------------------------

# 定义函数：run_global_mix
def run_global_mix(args: argparse.Namespace,
                   # 语句：files: List[str],
                   files: List[str],
                   # 语句：model: torch.nn.Module,
                   model: torch.nn.Module,
                   # 语句：device: str,
                   device: str,
                   # 语句：use_amp: bool,
                   use_amp: bool,
                   # 语句：warmup_pending: bool,
                   warmup_pending: bool,
                   # 语句：warmup_sync_ms: float) -> Dict:
                   warmup_sync_ms: float) -> Dict:
    """
    global-mix batching：
    - decode+tile：按图片并行预取
    - 推理 batch：全局 tile 池混合凑满 bs
    仍输出 per-image TSV 与 summary。
    """
    # 语句：os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 赋值/初始化：header
    header = [
        # 语句："file", "W", "H", "mode",
        "file", "W", "H", "mode",
        # 语句："total_tiles_possible", "valid_tiles", "num_batches",
        "total_tiles_possible", "valid_tiles", "num_batches",
        # 语句："t_decode_compute_ms", "t_tile_compute_ms", "t_decode_wait_ms",
        "t_decode_compute_ms", "t_tile_compute_ms", "t_decode_wait_ms",
        # 语句："t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        "t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        # 语句："t_total_ms",
        "t_total_ms",
        # 语句："tile_per_s", "img_per_s",
        "tile_per_s", "img_per_s",
        # 语句："avg_ms_per_tile_total", "pre_ms_per_tile",
        "avg_ms_per_tile_total", "pre_ms_per_tile",
        # 语句："note"
        "note"
    # 语句：]
    ]

    # 赋值/初始化：n_img
    n_img = len(files)
    # 语句：image_states: List[ImageState] = [ImageState(idx=i) for i in range(n_img)]
    image_states: List[ImageState] = [ImageState(idx=i) for i in range(n_img)]

    # steady（wall-clock）统计
    # 语句：first_tile_time: List[Optional[float]] = [None] * n_img
    first_tile_time: List[Optional[float]] = [None] * n_img
    # 语句：last_tile_done_time: List[Optional[float]] = [None] * n_img
    last_tile_done_time: List[Optional[float]] = [None] * n_img

    # 导入依赖：from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    # 赋值/初始化：decode_prefetch
    decode_prefetch = max(1, min(args.decode_prefetch, n_img))
    # 赋值/初始化：bs
    bs = max(1, args.bs)
    # 赋值/初始化：max_prefetch_batches
    max_prefetch_batches = max(1, args.prefetch)

    # 语句：tile_pool: Deque[TileRef] = deque()
    tile_pool: Deque[TileRef] = deque()
    # 语句：pre_q: Deque[Tuple] = deque() # (future, tile_refs)
    pre_q: Deque[Tuple] = deque()  # (future, tile_refs)

    # 语句：decode_q: Deque = deque()
    decode_q: Deque = deque()
    # 赋值/初始化：next_submit_idx
    next_submit_idx = 0
    # 赋值/初始化：next_consume_idx
    next_consume_idx = 0

    # 赋值/初始化：next_write_idx
    next_write_idx = 0
    # 赋值/初始化：written_images
    written_images = 0

    # 定义函数：submit_decode
    def submit_decode(i: int):
        # 赋值/初始化：path
        path = files[i]
        # 赋值/初始化：fut
        fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
        # 语句：decode_q.append(fut)
        decode_q.append(fut)

    # 定义函数：consume_one_decode
    def consume_one_decode() -> None:
        # 语句：nonlocal next_submit_idx, next_consume_idx
        nonlocal next_submit_idx, next_consume_idx
        # 条件判断：if next_consume_idx >= n_img:
        if next_consume_idx >= n_img:
            # 返回结果
            return

        # 赋值/初始化：w0
        w0 = now()
        # 语句：dec_res: DecodeTileResult = decode_q.popleft().result()
        dec_res: DecodeTileResult = decode_q.popleft().result()
        # 赋值/初始化：w1
        w1 = now()
        # 赋值/初始化：decode_wait_ms
        decode_wait_ms = (w1 - w0) * 1000.0

        # 赋值/初始化：st
        st = image_states[next_consume_idx]
        # 语句：st.file = dec_res.file
        st.file = dec_res.file
        # 语句：st.W = dec_res.W
        st.W = dec_res.W
        # 语句：st.H = dec_res.H
        st.H = dec_res.H
        # 语句：st.mode = dec_res.mode
        st.mode = dec_res.mode
        # 语句：st.total_tiles_possible = int(dec_res.total_tiles_possible)
        st.total_tiles_possible = int(dec_res.total_tiles_possible)
        # 语句：st.decode_compute_ms = float(dec_res.decode_compute_ms)
        st.decode_compute_ms = float(dec_res.decode_compute_ms)
        # 语句：st.tile_compute_ms = float(dec_res.tile_compute_ms)
        st.tile_compute_ms = float(dec_res.tile_compute_ms)
        # 语句：st.decode_wait_ms = float(decode_wait_ms)
        st.decode_wait_ms = float(decode_wait_ms)
        # 语句：st.tiles_uint8 = dec_res.tiles_uint8
        st.tiles_uint8 = dec_res.tiles_uint8

        # 赋值/初始化：valid_tiles
        valid_tiles = int(dec_res.tiles_uint8.shape[0])
        # 语句：st.valid_tiles = valid_tiles
        st.valid_tiles = valid_tiles

        # 条件判断：if first_tile_time[st.idx] is None:
        if first_tile_time[st.idx] is None:
            # 语句：first_tile_time[st.idx] = now()
            first_tile_time[st.idx] = now()

        # 条件判断：if valid_tiles <= 0:
        if valid_tiles <= 0:
            # 语句：st.done = True
            st.done = True
            # 语句：last_tile_done_time[st.idx] = now()
            last_tile_done_time[st.idx] = now()
            # 语句：st.tiles_uint8 = None
            st.tiles_uint8 = None
        # 条件分支：else
        else:
            # 循环：for ti in range(valid_tiles):
            for ti in range(valid_tiles):
                # 语句：tile_pool.append(TileRef(img_idx=st.idx, tile_idx=ti))
                tile_pool.append(TileRef(img_idx=st.idx, tile_idx=ti))

        # 语句：next_consume_idx += 1
        next_consume_idx += 1

        # 条件判断：if next_submit_idx < n_img:
        if next_submit_idx < n_img:
            # 语句：submit_decode(next_submit_idx)
            submit_decode(next_submit_idx)
            # 语句：next_submit_idx += 1
            next_submit_idx += 1

    # 定义函数：try_flush_done_rows
    def try_flush_done_rows(writer, f_out):
        # 语句：nonlocal next_write_idx, written_images, warmup_sync_ms
        nonlocal next_write_idx, written_images, warmup_sync_ms
        # 循环：while next_write_idx < n_img and image_states[next_write_idx].done:
        while next_write_idx < n_img and image_states[next_write_idx].done:
            # 赋值/初始化：st
            st = image_states[next_write_idx]

            # 赋值/初始化：t_total_ms
            t_total_ms = (
                # 语句：st.decode_compute_ms + st.tile_compute_ms + st.decode_wait_ms +
                st.decode_compute_ms + st.tile_compute_ms + st.decode_wait_ms +
                # 语句：st.pre_ms + st.submit_ms + st.sync_wait_ms
                st.pre_ms + st.submit_ms + st.sync_wait_ms
            # 语句：)
            )

            # 赋值/初始化：tile_per_s
            tile_per_s = (st.valid_tiles / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            # 赋值/初始化：img_per_s
            img_per_s = (1.0 / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            # 赋值/初始化：avg_ms_per_tile_total
            avg_ms_per_tile_total = (t_total_ms / st.valid_tiles) if st.valid_tiles > 0 else 0.0
            # 赋值/初始化：pre_ms_per_tile
            pre_ms_per_tile = (st.pre_ms / st.valid_tiles) if st.valid_tiles > 0 else 0.0

            # 赋值/初始化：note_parts
            note_parts = ["global_mix=1"]
            # 条件判断：if st.idx == 0 and warmup_sync_ms > 0:
            if st.idx == 0 and warmup_sync_ms > 0:
                # 语句：note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")
                note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")
            # 赋值/初始化：note
            note = ";".join(note_parts)

            # 语句：writer.writerow([
            writer.writerow([
                # 语句：st.file, st.W, st.H, st.mode,
                st.file, st.W, st.H, st.mode,
                # 语句：st.total_tiles_possible, st.valid_tiles, st.num_batches,
                st.total_tiles_possible, st.valid_tiles, st.num_batches,
                # 语句：f"{st.decode_compute_ms:.3f}", f"{st.tile_compute_ms:.3f}", f"{st.decode_wait_m…
                f"{st.decode_compute_ms:.3f}", f"{st.tile_compute_ms:.3f}", f"{st.decode_wait_ms:.3f}",
                # 语句：f"{st.pre_ms:.3f}", f"{st.submit_ms:.3f}", f"{st.sync_wait_ms:.3f}",
                f"{st.pre_ms:.3f}", f"{st.submit_ms:.3f}", f"{st.sync_wait_ms:.3f}",
                # 语句：f"{t_total_ms:.3f}",
                f"{t_total_ms:.3f}",
                # 语句：f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                # 语句：f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                # 语句：note
                note
            # 语句：])
            ])
            # 语句：f_out.flush()
            f_out.flush()

            # 语句：written_images += 1
            written_images += 1
            # 条件判断：if (written_images % args.log_every) == 0 or written_images == n_img:
            if (written_images % args.log_every) == 0 or written_images == n_img:
                # 打印进度/汇总信息
                print(f"[A2_v2][GMIX] wrote {written_images}/{n_img} images, "
                      # 语句：f"tile_pool={len(tile_pool)} pre_q={len(pre_q)}")
                      f"tile_pool={len(tile_pool)} pre_q={len(pre_q)}")

            # 语句：next_write_idx += 1
            next_write_idx += 1

    # 定义函数：distribute_time
    def distribute_time(tile_refs: List[TileRef], ms: float, field: str) -> None:
        # 条件判断：if not tile_refs:
        if not tile_refs:
            # 返回结果
            return
        # 赋值/初始化：B
        B = len(tile_refs)
        # 语句：counts: Dict[int, int] = {}
        counts: Dict[int, int] = {}
        # 循环：for ref in tile_refs:
        for ref in tile_refs:
            # 语句：counts[ref.img_idx] = counts.get(ref.img_idx, 0) + 1
            counts[ref.img_idx] = counts.get(ref.img_idx, 0) + 1
        # 循环：for img_i, c in counts.items():
        for img_i, c in counts.items():
            # 赋值/初始化：st
            st = image_states[img_i]
            # 赋值/初始化：share
            share = ms * (c / B)
            # 条件判断：if field == "pre_ms":
            if field == "pre_ms":
                # 语句：st.pre_ms += share
                st.pre_ms += share
            # 条件分支：elif field == "submit_ms":
            elif field == "submit_ms":
                # 语句：st.submit_ms += share
                st.submit_ms += share
            # 条件分支：elif field == "sync_wait_ms":
            elif field == "sync_wait_ms":
                # 语句：st.sync_wait_ms += share
                st.sync_wait_ms += share
            # 条件分支：elif field == "num_batches":
            elif field == "num_batches":
                # 语句：st.num_batches += 1
                st.num_batches += 1
            # 条件分支：else
            else:
                # 抛出异常
                raise ValueError(field)

    # 定义函数：mark_batch_done
    def mark_batch_done(tile_refs: List[TileRef]) -> None:
        # 条件判断：if not tile_refs:
        if not tile_refs:
            # 返回结果
            return
        # 语句：counts: Dict[int, int] = {}
        counts: Dict[int, int] = {}
        # 循环：for ref in tile_refs:
        for ref in tile_refs:
            # 语句：counts[ref.img_idx] = counts.get(ref.img_idx, 0) + 1
            counts[ref.img_idx] = counts.get(ref.img_idx, 0) + 1
        # 赋值/初始化：t_done
        t_done = now()
        # 循环：for img_i, c in counts.items():
        for img_i, c in counts.items():
            # 赋值/初始化：st
            st = image_states[img_i]
            # 语句：st.processed_tiles += c
            st.processed_tiles += c
            # 条件判断：if st.valid_tiles > 0 and st.processed_tiles >= st.valid_tiles and not st.done:
            if st.valid_tiles > 0 and st.processed_tiles >= st.valid_tiles and not st.done:
                # 语句：st.done = True
                st.done = True
                # 语句：last_tile_done_time[img_i] = t_done
                last_tile_done_time[img_i] = t_done
                # 语句：st.tiles_uint8 = None
                st.tiles_uint8 = None

    # 赋值/初始化：t_all0
    t_all0 = now()
    # 语句：inflight_refs: Optional[List[TileRef]] = None
    inflight_refs: Optional[List[TileRef]] = None

    # 上下文管理（自动关闭资源）：with ThreadPoolExecutor(max_workers=args._decode_workers) as decode_ex, \
    with ThreadPoolExecutor(max_workers=args._decode_workers) as decode_ex, \
            # 语句：ThreadPoolExecutor(max_workers=args._pre_workers) as pre_ex, \
            ThreadPoolExecutor(max_workers=args._pre_workers) as pre_ex, \
            # 语句：open(args.out, "w", newline="", encoding="utf-8") as f_out:
            open(args.out, "w", newline="", encoding="utf-8") as f_out:

        # 赋值/初始化：writer
        writer = csv.writer(f_out, delimiter="\t")
        # 语句：writer.writerow(header)
        writer.writerow(header)
        # 语句：f_out.flush()
        f_out.flush()

        # 赋值/初始化：next_submit_idx
        next_submit_idx = 0
        # 循环：while next_submit_idx < decode_prefetch:
        while next_submit_idx < decode_prefetch:
            # 语句：submit_decode(next_submit_idx)
            submit_decode(next_submit_idx)
            # 语句：next_submit_idx += 1
            next_submit_idx += 1

        # 循环：while True:
        while True:
            # 异常处理：try
            try_flush_done_rows(writer, f_out)

            # 填满 preprocess prefetch 队列
            # 循环：while len(pre_q) < max_prefetch_batches:
            while len(pre_q) < max_prefetch_batches:
                # 循环：while len(tile_pool) < bs and next_consume_idx < n_img:
                while len(tile_pool) < bs and next_consume_idx < n_img:
                    # 语句：consume_one_decode()
                    consume_one_decode()
                    # 异常处理：try
                    try_flush_done_rows(writer, f_out)

                # 条件判断：if len(tile_pool) >= bs:
                if len(tile_pool) >= bs:
                    # 赋值/初始化：batch_size
                    batch_size = bs
                # 条件分支：else
                else:
                    # 条件判断：if next_consume_idx >= n_img and len(tile_pool) > 0:
                    if next_consume_idx >= n_img and len(tile_pool) > 0:
                        # 赋值/初始化：batch_size
                        batch_size = len(tile_pool)
                    # 条件分支：else
                    else:
                        # 语句：break
                        break

                # 赋值/初始化：tile_refs
                tile_refs = [tile_pool.popleft() for _ in range(batch_size)]
                # 语句：distribute_time(tile_refs, 0.0, "num_batches")
                distribute_time(tile_refs, 0.0, "num_batches")

                # 赋值/初始化：fut
                fut = pre_ex.submit(preprocess_mixed_batch, tile_refs, image_states, args.cpu_out_fp16)
                # 语句：pre_q.append((fut, tile_refs))
                pre_q.append((fut, tile_refs))

            # 条件判断：if pre_q:
            if pre_q:
                # 赋值/初始化：fut, tile_refs
                fut, tile_refs = pre_q.popleft()
                # 赋值/初始化：x_cpu, pre_ms
                x_cpu, pre_ms = fut.result()
                # 语句：distribute_time(tile_refs, pre_ms, "pre_ms")
                distribute_time(tile_refs, pre_ms, "pre_ms")

                # 提交下一批前：先等上一批 inflight 完成，并把 sync_wait 计入上一批
                # 条件判断：if inflight_refs is not None:
                if inflight_refs is not None:
                    # 赋值/初始化：sw0
                    sw0 = now()
                    # 语句：sync_npu()
                    sync_npu()
                    # 赋值/初始化：sw1
                    sw1 = now()
                    # 赋值/初始化：sync_wait_ms
                    sync_wait_ms = (sw1 - sw0) * 1000.0
                    # 语句：distribute_time(inflight_refs, sync_wait_ms, "sync_wait_ms")
                    distribute_time(inflight_refs, sync_wait_ms, "sync_wait_ms")
                    # 语句：mark_batch_done(inflight_refs)
                    mark_batch_done(inflight_refs)
                    # 赋值/初始化：inflight_refs
                    inflight_refs = None
                    # 异常处理：try
                    try_flush_done_rows(writer, f_out)

                # 条件判断：if warmup_pending:
                if warmup_pending:
                    # 赋值/初始化：sw0
                    sw0 = now()
                    # 语句：sync_npu()
                    sync_npu()
                    # 赋值/初始化：sw1
                    sw1 = now()
                    # 赋值/初始化：warmup_sync_ms
                    warmup_sync_ms = (sw1 - sw0) * 1000.0
                    # 赋值/初始化：warmup_pending
                    warmup_pending = False

                # 赋值/初始化：sb0
                sb0 = now()
                # 赋值/初始化：x
                x = x_cpu.to(device, non_blocking=True)
                # 条件判断：if use_amp:
                if use_amp:
                    # 上下文管理（自动关闭资源）：with torch.npu.amp.autocast():
                    with torch.npu.amp.autocast():
                        # 赋值/初始化：_
                        _ = model(x)
                # 条件分支：else
                else:
                    # 赋值/初始化：_
                    _ = model(x)
                # 赋值/初始化：sb1
                sb1 = now()
                # 赋值/初始化：submit_ms
                submit_ms = (sb1 - sb0) * 1000.0
                # 语句：distribute_time(tile_refs, submit_ms, "submit_ms")
                distribute_time(tile_refs, submit_ms, "submit_ms")

                # 赋值/初始化：inflight_refs
                inflight_refs = tile_refs
                # 语句：continue
                continue

            # 条件判断：if next_consume_idx < n_img:
            if next_consume_idx < n_img:
                # 语句：consume_one_decode()
                consume_one_decode()
                # 语句：continue
                continue

            # 条件判断：if inflight_refs is not None:
            if inflight_refs is not None:
                # 赋值/初始化：sw0
                sw0 = now()
                # 语句：sync_npu()
                sync_npu()
                # 赋值/初始化：sw1
                sw1 = now()
                # 赋值/初始化：sync_wait_ms
                sync_wait_ms = (sw1 - sw0) * 1000.0
                # 语句：distribute_time(inflight_refs, sync_wait_ms, "sync_wait_ms")
                distribute_time(inflight_refs, sync_wait_ms, "sync_wait_ms")
                # 语句：mark_batch_done(inflight_refs)
                mark_batch_done(inflight_refs)
                # 赋值/初始化：inflight_refs
                inflight_refs = None
                # 异常处理：try
                try_flush_done_rows(writer, f_out)
                # 语句：continue
                continue

            # 条件判断：if written_images >= n_img and not tile_pool and not pre_q and inflight_refs is…
            if written_images >= n_img and not tile_pool and not pre_q and inflight_refs is None:
                # 语句：break
                break

        # 异常处理：try
        try_flush_done_rows(writer, f_out)

    # 赋值/初始化：t_all1
    t_all1 = now()
    # 赋值/初始化：total_time_s
    total_time_s = (t_all1 - t_all0)

    # 赋值/初始化：total_valid_tiles
    total_valid_tiles = sum(st.valid_tiles for st in image_states)
    # 赋值/初始化：total_images
    total_images = n_img

    # 赋值/初始化：overall_tile_s
    overall_tile_s = (total_valid_tiles / total_time_s) if total_time_s > 0 else 0.0
    # 赋值/初始化：overall_img_s
    overall_img_s = (total_images / total_time_s) if total_time_s > 0 else 0.0

    # 语句：steady_ids = [i for i in range(n_img) if i >= args.skip_steady]
    steady_ids = [i for i in range(n_img) if i >= args.skip_steady]
    # 赋值/初始化：steady_valid_tiles
    steady_valid_tiles = sum(image_states[i].valid_tiles for i in steady_ids)

    # 赋值/初始化：t0_candidates
    t0_candidates = [first_tile_time[i] for i in steady_ids if first_tile_time[i] is not None]
    # 赋值/初始化：t1_candidates
    t1_candidates = [last_tile_done_time[i] for i in steady_ids if last_tile_done_time[i] is not None]

    # 条件判断：if t0_candidates and t1_candidates:
    if t0_candidates and t1_candidates:
        # 赋值/初始化：steady_time_s
        steady_time_s = max(t1_candidates) - min(t0_candidates)
    # 条件分支：else
    else:
        # 赋值/初始化：steady_time_s
        steady_time_s = 0.0

    # 赋值/初始化：steady_tile_s
    steady_tile_s = (steady_valid_tiles / steady_time_s) if steady_time_s > 0 else 0.0
    # 赋值/初始化：steady_img_s
    steady_img_s = ((len(steady_ids)) / steady_time_s) if steady_time_s > 0 else 0.0

    # 定义函数：safe_div
    def safe_div(a: float, b: int) -> float:
        # 返回结果
        return a / b if b > 0 else 0.0

    # 赋值/初始化：sum_decode_compute_ms
    sum_decode_compute_ms = sum(st.decode_compute_ms for st in image_states)
    # 赋值/初始化：sum_tile_compute_ms
    sum_tile_compute_ms = sum(st.tile_compute_ms for st in image_states)
    # 赋值/初始化：sum_decode_wait_ms
    sum_decode_wait_ms = sum(st.decode_wait_ms for st in image_states)
    # 赋值/初始化：sum_pre_ms
    sum_pre_ms = sum(st.pre_ms for st in image_states)
    # 赋值/初始化：sum_submit_ms
    sum_submit_ms = sum(st.submit_ms for st in image_states)
    # 赋值/初始化：sum_sync_wait_ms
    sum_sync_wait_ms = sum(st.sync_wait_ms for st in image_states)
    # 赋值/初始化：sum_total_ms
    sum_total_ms = sum(
        # 语句：st.decode_compute_ms + st.tile_compute_ms + st.decode_wait_ms +
        st.decode_compute_ms + st.tile_compute_ms + st.decode_wait_ms +
        # 语句：st.pre_ms + st.submit_ms + st.sync_wait_ms
        st.pre_ms + st.submit_ms + st.sync_wait_ms
        # 循环：for st in image_states
        for st in image_states
    # 语句：)
    )

    # 赋值/初始化：mean_decode_compute_ms
    mean_decode_compute_ms = safe_div(sum_decode_compute_ms, total_images)
    # 赋值/初始化：mean_tile_compute_ms
    mean_tile_compute_ms = safe_div(sum_tile_compute_ms, total_images)
    # 赋值/初始化：mean_decode_wait_ms
    mean_decode_wait_ms = safe_div(sum_decode_wait_ms, total_images)
    # 赋值/初始化：mean_pre_ms
    mean_pre_ms = safe_div(sum_pre_ms, total_images)
    # 赋值/初始化：mean_submit_ms
    mean_submit_ms = safe_div(sum_submit_ms, total_images)
    # 赋值/初始化：mean_sync_wait_ms
    mean_sync_wait_ms = safe_div(sum_sync_wait_ms, total_images)
    # 赋值/初始化：mean_total_ms
    mean_total_ms = safe_div(sum_total_ms, total_images)

    # 赋值/初始化：cold_first_ms
    cold_first_ms = (
        # 语句：image_states[0].decode_compute_ms + image_states[0].tile_compute_ms + image_sta…
        image_states[0].decode_compute_ms + image_states[0].tile_compute_ms + image_states[0].decode_wait_ms +
        # 语句：image_states[0].pre_ms + image_states[0].submit_ms + image_states[0].sync_wait_…
        image_states[0].pre_ms + image_states[0].submit_ms + image_states[0].sync_wait_ms
    # 语句：) if image_states else 0.0
    ) if image_states else 0.0

    # 返回结果
    return {
        # 语句："tiles_total": int(total_valid_tiles),
        "tiles_total": int(total_valid_tiles),
        # 语句："num_images": int(total_images),
        "num_images": int(total_images),
        # 语句："total_time_s": float(total_time_s),
        "total_time_s": float(total_time_s),
        # 语句："overall_tile_s": float(overall_tile_s),
        "overall_tile_s": float(overall_tile_s),
        # 语句："overall_img_s": float(overall_img_s),
        "overall_img_s": float(overall_img_s),
        # 语句："steady_tile_s": float(steady_tile_s),
        "steady_tile_s": float(steady_tile_s),
        # 语句："steady_img_s": float(steady_img_s),
        "steady_img_s": float(steady_img_s),
        # 语句："cold_first_ms": float(cold_first_ms),
        "cold_first_ms": float(cold_first_ms),
        # 语句："warmup_sync_ms": float(warmup_sync_ms),
        "warmup_sync_ms": float(warmup_sync_ms),
        # 语句："mean_decode_compute_ms": float(mean_decode_compute_ms),
        "mean_decode_compute_ms": float(mean_decode_compute_ms),
        # 语句："mean_tile_compute_ms": float(mean_tile_compute_ms),
        "mean_tile_compute_ms": float(mean_tile_compute_ms),
        # 语句："mean_decode_wait_ms": float(mean_decode_wait_ms),
        "mean_decode_wait_ms": float(mean_decode_wait_ms),
        # 语句："mean_pre_ms": float(mean_pre_ms),
        "mean_pre_ms": float(mean_pre_ms),
        # 语句："mean_submit_ms": float(mean_submit_ms),
        "mean_submit_ms": float(mean_submit_ms),
        # 语句："mean_sync_wait_ms": float(mean_sync_wait_ms),
        "mean_sync_wait_ms": float(mean_sync_wait_ms),
        # 语句："mean_total_ms": float(mean_total_ms),
        "mean_total_ms": float(mean_total_ms),
    # 语句：}
    }


# ----------------------------
# 8) main
# ----------------------------

# 定义函数：main
def main() -> None:
    # 赋值/初始化：ap
    ap = argparse.ArgumentParser()

    # 语句：ap.add_argument("--file_list", type=str, required=True)
    ap.add_argument("--file_list", type=str, required=True)
    # 语句：ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")
    ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")

    # 语句：ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--tile", type=int, default=224)
    # 语句：ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)

    # 语句：ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--bs", type=int, default=96)

    # global mix 默认开启
    # 语句：ap.add_argument("--global_mix", type=int, default=1, choices=[0, 1],
    ap.add_argument("--global_mix", type=int, default=1, choices=[0, 1],
                    # 赋值/初始化：help
                    help="1=global-mix batching（跨图混 batch 凑满 bs），0=旧的单图口径")

    # 语句：ap.add_argument("--white_thr", type=float, default=1.01)
    ap.add_argument("--white_thr", type=float, default=1.01)

    # 语句：ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp1…
    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    # 语句：ap.add_argument("--cpu_out_fp16", action="store_true")
    ap.add_argument("--cpu_out_fp16", action="store_true")
    # 语句：ap.set_defaults(cpu_out_fp16=True)
    ap.set_defaults(cpu_out_fp16=True)

    # 语句：ap.add_argument("--cpu_workers", type=int, default=8)
    ap.add_argument("--cpu_workers", type=int, default=8)

    # 语句：ap.add_argument("--decode_workers", type=int, default=-1,
    ap.add_argument("--decode_workers", type=int, default=-1,
                    # 赋值/初始化：help
                    help="-1=自动分配，否则手动指定 decode 线程数")
    # 语句：ap.add_argument("--decode_prefetch", type=int, default=4,
    ap.add_argument("--decode_prefetch", type=int, default=4,
                    # 赋值/初始化：help
                    help="预取多少张图做 decode+tile（窗口越大越占内存）")

    # 语句：ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--prefetch", type=int, default=3)

    # 语句：ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], d…
    ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], default="async")
    # 语句：ap.add_argument("--warmup_iters", type=int, default=3)
    ap.add_argument("--warmup_iters", type=int, default=3)

    # 语句：ap.add_argument("--skip_steady", type=int, default=1)
    ap.add_argument("--skip_steady", type=int, default=1)
    # 语句：ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--max_images", type=int, default=-1)

    # 语句：ap.add_argument("--out", type=str, default="logs/A2_final_v2.tsv")
    ap.add_argument("--out", type=str, default="logs/A2_final_v2.tsv")
    # 语句：ap.add_argument("--summary_json", type=str, default="")
    ap.add_argument("--summary_json", type=str, default="")
    # 语句：ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=10)

    # 赋值/初始化：args
    args = ap.parse_args()

    # 上下文管理（自动关闭资源）：with open(args.file_list, "r", encoding="utf-8") as f:
    with open(args.file_list, "r", encoding="utf-8") as f:
        # 赋值/初始化：files
        files = [line.strip() for line in f if line.strip()]

    # 条件判断：if args.max_images > 0:
    if args.max_images > 0:
        # 赋值/初始化：files
        files = files[:args.max_images]

    # 条件判断：if not files:
    if not files:
        # 抛出异常
        raise ValueError("file_list is empty")

    # 条件判断：if args.decode_workers is None or args.decode_workers < 0:
    if args.decode_workers is None or args.decode_workers < 0:
        # 赋值/初始化：decode_workers
        decode_workers = max(1, min(4, args.cpu_workers // 4))
    # 条件分支：else
    else:
        # 赋值/初始化：decode_workers
        decode_workers = max(1, args.decode_workers)

    # 赋值/初始化：pre_workers
    pre_workers = max(1, args.cpu_workers - decode_workers)

    # 语句：args._decode_workers = decode_workers
    args._decode_workers = decode_workers
    # 语句：args._pre_workers = pre_workers
    args._pre_workers = pre_workers

    # 赋值/初始化：device
    device = "npu:0"
    # 赋值/初始化：model
    model = build_uni_backbone(args.ckpt_dir, device=device, img_size=args.tile)

    # 语句：use_amp = (args.precision == "fp16")
    use_amp = (args.precision == "fp16")

    # 赋值/初始化：warmup_pending
    warmup_pending = False
    # 赋值/初始化：warmup_sync_ms
    warmup_sync_ms = 0.0

    # 赋值/初始化：warm_bs
    warm_bs = min(args.bs, 64)
    # 赋值/初始化：x_warm
    x_warm = torch.randn(warm_bs, 3, args.tile, args.tile, device=device)

    # 条件判断：if args.warmup_mode == "sync":
    if args.warmup_mode == "sync":
        # 循环：for _ in range(args.warmup_iters):
        for _ in range(args.warmup_iters):
            # 条件判断：if use_amp:
            if use_amp:
                # 上下文管理（自动关闭资源）：with torch.npu.amp.autocast():
                with torch.npu.amp.autocast():
                    # 赋值/初始化：_
                    _ = model(x_warm)
            # 条件分支：else
            else:
                # 赋值/初始化：_
                _ = model(x_warm)
        # 语句：sync_npu()
        sync_npu()
        # 赋值/初始化：warmup_pending
        warmup_pending = False

    # 条件分支：elif args.warmup_mode == "async":
    elif args.warmup_mode == "async":
        # 循环：for _ in range(args.warmup_iters):
        for _ in range(args.warmup_iters):
            # 条件判断：if use_amp:
            if use_amp:
                # 上下文管理（自动关闭资源）：with torch.npu.amp.autocast():
                with torch.npu.amp.autocast():
                    # 赋值/初始化：_
                    _ = model(x_warm)
            # 条件分支：else
            else:
                # 赋值/初始化：_
                _ = model(x_warm)
        # 赋值/初始化：warmup_pending
        warmup_pending = True
    # 条件分支：else
    else:
        # 赋值/初始化：warmup_pending
        warmup_pending = False

    # 条件判断：if args.global_mix == 1:
    if args.global_mix == 1:
        # 赋值/初始化：result
        result = run_global_mix(args, files, model, device, use_amp, warmup_pending, warmup_sync_ms)
    # 条件分支：else
    else:
        # 赋值/初始化：result
        result = run_per_image(args, files, model, device, use_amp, warmup_pending, warmup_sync_ms)

    # 打印进度/汇总信息
    print("\n================ [A2_v2 SUMMARY] ================")
    # 打印进度/汇总信息
    print(f"file_list         : {args.file_list}")
    # 打印进度/汇总信息
    print(f"num_images        : {result['num_images']}")
    # 打印进度/汇总信息
    print(f"tile/stride       : {args.tile}/{args.stride}")
    # 打印进度/汇总信息
    print(f"white_thr         : {args.white_thr}")
    # 打印进度/汇总信息
    print(f"precision         : {args.precision} (cpu_out_fp16={args.cpu_out_fp16})")
    # 打印进度/汇总信息
    print(f"global_mix        : {args.global_mix}")
    # 打印进度/汇总信息
    print(f"bs                : {args.bs}")
    # 打印进度/汇总信息
    print(f"cpu_workers       : {args.cpu_workers} (decode={decode_workers}, pre={pre_workers})")
    # 打印进度/汇总信息
    print(f"decode_prefetch   : {min(args.decode_prefetch, len(files))}")
    # 打印进度/汇总信息
    print(f"prefetch(batches) : {args.prefetch}")
    # 打印进度/汇总信息
    print(f"warmup_mode       : {args.warmup_mode} (iters={args.warmup_iters})")
    # 打印进度/汇总信息
    print(f"skip_steady       : {args.skip_steady}")
    # 打印进度/汇总信息
    print("-----------------------------------------------")
    # 打印进度/汇总信息
    print(f"tiles_total       : {result['tiles_total']}")
    # 打印进度/汇总信息
    print(f"total_time_s      : {result['total_time_s']:.3f}")
    # 打印进度/汇总信息
    print(f"overall_tile/s    : {result['overall_tile_s']:.3f}")
    # 打印进度/汇总信息
    print(f"steady_tile/s     : {result['steady_tile_s']:.3f}")
    # 打印进度/汇总信息
    print(f"overall_img/s     : {result['overall_img_s']:.3f}")
    # 打印进度/汇总信息
    print(f"steady_img/s      : {result['steady_img_s']:.3f}")
    # 打印进度/汇总信息
    print("-----------------------------------------------")
    # 打印进度/汇总信息
    print(f"cold_first_ms     : {result['cold_first_ms']:.2f}")
    # 条件判断：if result.get("warmup_sync_ms", 0.0) > 0:
    if result.get("warmup_sync_ms", 0.0) > 0:
        # 打印进度/汇总信息
        print(f"warmup_sync_ms    : {result['warmup_sync_ms']:.2f}")
    # 打印进度/汇总信息
    print("-----------------------------------------------")
    # 打印进度/汇总信息
    print(f"mean_decode_compute_ms : {result['mean_decode_compute_ms']:.2f}")
    # 打印进度/汇总信息
    print(f"mean_tile_compute_ms   : {result['mean_tile_compute_ms']:.2f}")
    # 打印进度/汇总信息
    print(f"mean_decode_wait_ms    : {result['mean_decode_wait_ms']:.2f}")
    # 打印进度/汇总信息
    print(f"mean_pre_ms            : {result['mean_pre_ms']:.2f}")
    # 打印进度/汇总信息
    print(f"mean_submit_ms         : {result['mean_submit_ms']:.2f}")
    # 打印进度/汇总信息
    print(f"mean_sync_wait_ms      : {result['mean_sync_wait_ms']:.2f}")
    # 打印进度/汇总信息
    print(f"mean_total_ms          : {result['mean_total_ms']:.2f}")
    # 打印进度/汇总信息
    print("================================================\n")

    # 条件判断：if args.summary_json:
    if args.summary_json:
        # 赋值/初始化：summary
        summary = {
            # 语句："config": {
            "config": {
                # 语句："file_list": args.file_list,
                "file_list": args.file_list,
                # 语句："ckpt_dir": args.ckpt_dir,
                "ckpt_dir": args.ckpt_dir,
                # 语句："tile": args.tile,
                "tile": args.tile,
                # 语句："stride": args.stride,
                "stride": args.stride,
                # 语句："bs": args.bs,
                "bs": args.bs,
                # 语句："global_mix": args.global_mix,
                "global_mix": args.global_mix,
                # 语句："white_thr": args.white_thr,
                "white_thr": args.white_thr,
                # 语句："precision": args.precision,
                "precision": args.precision,
                # 语句："cpu_out_fp16": args.cpu_out_fp16,
                "cpu_out_fp16": args.cpu_out_fp16,
                # 语句："cpu_workers": args.cpu_workers,
                "cpu_workers": args.cpu_workers,
                # 语句："decode_workers": decode_workers,
                "decode_workers": decode_workers,
                # 语句："pre_workers": pre_workers,
                "pre_workers": pre_workers,
                # 语句："decode_prefetch": min(args.decode_prefetch, len(files)),
                "decode_prefetch": min(args.decode_prefetch, len(files)),
                # 语句："prefetch": args.prefetch,
                "prefetch": args.prefetch,
                # 语句："warmup_mode": args.warmup_mode,
                "warmup_mode": args.warmup_mode,
                # 语句："warmup_iters": args.warmup_iters,
                "warmup_iters": args.warmup_iters,
                # 语句："skip_steady": args.skip_steady,
                "skip_steady": args.skip_steady,
                # 语句："max_images": args.max_images,
                "max_images": args.max_images,
                # 语句："out_tsv": args.out,
                "out_tsv": args.out,
            # 语句：},
            },
            # 语句："result": result
            "result": result
        # 语句：}
        }
        # 语句：os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        # 上下文管理（自动关闭资源）：with open(args.summary_json, "w", encoding="utf-8") as f:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            # 语句：json.dump(summary, f, ensure_ascii=False, indent=2)
            json.dump(summary, f, ensure_ascii=False, indent=2)
        # 打印进度/汇总信息
        print(f"[A2_v2] wrote summary_json -> {args.summary_json}")


# 条件判断：if __name__ == "__main__":
if __name__ == "__main__":
    # 语句：main()
    main()
