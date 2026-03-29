#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

import os
import time
import csv
import json
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch_npu
import timm


# ----------------------------
# 0) 计时 & NPU 同步
# ----------------------------

def now() -> float:
    """高精度计时（秒）"""
    return time.perf_counter()


def sync_npu() -> None:
    """
    NPU 异步：forward 返回不代表算完。
    精准计时必须在关键点 synchronize。
    """
    torch_npu.npu.synchronize()


# ----------------------------
# 1) 构建 UNI backbone（与你 test_uni_npu.py 保持一致）
# ----------------------------

def build_uni_backbone(ckpt_dir: str,
                       device: str = "npu:0",
                       img_size: int = 224) -> torch.nn.Module:
    """
    ViT-L/16 224，num_classes=0，只输出特征。
    """
    ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[UNI] missing checkpoint: {ckpt_path}")

    dev = torch.device(device)
    torch.npu.set_device(dev)

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=img_size,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model.eval().to(dev)
    return model


# ----------------------------
# 2) fast_grid：stride==tile 时最快切块（避免 Python for 循环）
# ----------------------------

def extract_tiles_fast_grid(img_np: np.ndarray,
                            tile: int,
                            stride: int) -> Tuple[np.ndarray, int]:
    """
    stride==tile 时，用 reshape/transpose 一次生成 tiles。
    返回：
    - tiles_uint8: (N, tile, tile, 3)
    - total_tiles_possible: N（未过滤前）
    """
    assert stride == tile, "fast_grid 仅支持 stride==tile"

    H, W, C = img_np.shape
    if C != 3:
        raise ValueError(f"expect RGB image with 3 channels, got shape={img_np.shape}")

    nx = W // tile
    ny = H // tile
    if nx <= 0 or ny <= 0:
        return np.zeros((0, tile, tile, 3), dtype=np.uint8), 0

    H_use = ny * tile
    W_use = nx * tile
    crop = img_np[:H_use, :W_use, :]

    tiles = crop.reshape(ny, tile, nx, tile, 3).transpose(0, 2, 1, 3, 4)
    tiles = tiles.reshape(ny * nx, tile, tile, 3)

    return tiles, ny * nx


def tissue_mask_uint8(tiles_uint8: np.ndarray,
                      white_thr: float = 0.8,
                      gray_high: float = 0.9) -> np.ndarray:
    """
    极简白底过滤：
    - gray = mean(RGB)/255
    - white_ratio = (gray > gray_high).mean()
    - white_ratio < white_thr => 保留
    """
    gray = tiles_uint8.mean(axis=3) / 255.0
    white_ratio = (gray > gray_high).mean(axis=(1, 2))
    keep = white_ratio < white_thr
    return keep


# ----------------------------
# 3) 向量化 preprocess：uint8 tiles -> NCHW tensor
# ----------------------------

def preprocess_tiles_uint8_vectorized(tiles_uint8: np.ndarray,
                                      out_fp16: bool = True,
                                      mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                                      std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
                                      ) -> torch.Tensor:
    """
    (B,H,W,3) uint8 -> torch tensor (B,3,H,W)
    1) /255 -> [0,1]
    2) (x-mean)/std
    3) NHWC->NCHW
    4) 可选转 fp16（减少 H2D 传输）
    """
    x = tiles_uint8.astype(np.float32) / 255.0

    m = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    s = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - m) / s

    x = np.transpose(x, (0, 3, 1, 2))
    x_cpu = torch.from_numpy(x)

    if out_fp16:
        x_cpu = x_cpu.half()

    return x_cpu.contiguous()


# ----------------------------
# 4) decode + tile（并行 worker）
# ----------------------------

@dataclass
class DecodeTileResult:
    file: str
    W: int
    H: int
    mode: str
    total_tiles_possible: int
    tiles_uint8: np.ndarray
    decode_compute_ms: float
    tile_compute_ms: float


def decode_and_tile_worker(path: str,
                           tile: int,
                           stride: int,
                           white_thr: float) -> DecodeTileResult:
    """
    线程池 worker：
    - decode (PIL)
    - tilegen (fast_grid)
    - 可选白底过滤（white_thr<1.0 才做；你现在 1.01 直接跳过）
    """
    t0 = now()

    with Image.open(path) as im:
        im = im.convert("RGB")
        img_np = np.array(im)

    t1 = now()

    if stride == tile:
        mode = "fast_grid"
        tiles_uint8, total = extract_tiles_fast_grid(img_np, tile=tile, stride=stride)
        if total > 0 and white_thr < 1.0:
            keep = tissue_mask_uint8(tiles_uint8, white_thr=white_thr)
            tiles_uint8 = tiles_uint8[keep]
    else:
        mode = "fallback"
        H, W, _ = img_np.shape
        coords = []
        for y in range(0, H - tile + 1, stride):
            for x in range(0, W - tile + 1, stride):
                coords.append((x, y))
        total = len(coords)

        tiles_list = []
        for (x, y) in coords:
            tiles_list.append(img_np[y:y + tile, x:x + tile])
        tiles_uint8 = np.stack(tiles_list, axis=0).astype(np.uint8) if total > 0 else \
            np.zeros((0, tile, tile, 3), dtype=np.uint8)

        if total > 0 and white_thr < 1.0:
            keep = tissue_mask_uint8(tiles_uint8, white_thr=white_thr)
            tiles_uint8 = tiles_uint8[keep]

    t2 = now()

    H0, W0, _ = img_np.shape
    return DecodeTileResult(
        file=os.path.basename(path),
        W=W0,
        H=H0,
        mode=mode,
        total_tiles_possible=total,
        tiles_uint8=tiles_uint8,
        decode_compute_ms=(t1 - t0) * 1000.0,
        tile_compute_ms=(t2 - t1) * 1000.0,
    )


# ----------------------------
# 5) global-mix 数据结构
# ----------------------------

@dataclass(frozen=True)
class TileRef:
    img_idx: int
    tile_idx: int


@dataclass
class ImageState:
    idx: int
    file: str = ""
    W: int = 0
    H: int = 0
    mode: str = ""
    total_tiles_possible: int = 0
    valid_tiles: int = 0

    # decode/tile
    decode_compute_ms: float = 0.0
    tile_compute_ms: float = 0.0
    decode_wait_ms: float = 0.0

    # 分摊统计（按 tile 数比例分摊到 image）
    pre_ms: float = 0.0
    submit_ms: float = 0.0
    sync_wait_ms: float = 0.0

    # global-mix 计数
    processed_tiles: int = 0
    num_batches: int = 0

    # 资源（保持 tiles 数组引用，直到该图片处理完成）
    tiles_uint8: Optional[np.ndarray] = None

    done: bool = False


def preprocess_mixed_batch(tile_refs: List[TileRef],
                           image_states: List[ImageState],
                           out_fp16: bool) -> Tuple[torch.Tensor, float]:
    """
    在线程池里：
    - 根据 tile_refs 把 tile 拼成 (B,H,W,3) uint8
    - 向量化 preprocess -> torch tensor
    返回：
    - x_cpu
    - pre_ms
    """
    t0 = now()
    tiles = []
    for ref in tile_refs:
        arr = image_states[ref.img_idx].tiles_uint8
        if arr is None:
            raise RuntimeError(f"tiles_uint8 already freed for img_idx={ref.img_idx}")
        tiles.append(arr[ref.tile_idx])
    batch_uint8 = np.stack(tiles, axis=0) if tiles else np.zeros((0, 224, 224, 3), dtype=np.uint8)

    x_cpu = preprocess_tiles_uint8_vectorized(batch_uint8, out_fp16=out_fp16)
    t1 = now()
    return x_cpu, (t1 - t0) * 1000.0


# ----------------------------
# 6) 单图口径（旧逻辑）保持不变：便于回退对比
# ----------------------------

def run_per_image(args: argparse.Namespace,
                  files: List[str],
                  model: torch.nn.Module,
                  device: str,
                  use_amp: bool,
                  warmup_pending: bool,
                  warmup_sync_ms: float) -> Dict:
    """
    旧版：逐图消费 decode->图内 batch 推理。
    返回 summary dict（result 部分），并且会落盘 per-image TSV。
    """
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    header = [
        "file", "W", "H", "mode",
        "total_tiles_possible", "valid_tiles", "num_batches",
        "t_decode_compute_ms", "t_tile_compute_ms", "t_decode_wait_ms",
        "t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        "t_total_ms",
        "tile_per_s", "img_per_s",
        "avg_ms_per_tile_total", "pre_ms_per_tile",
        "note"
    ]

    total_valid_tiles = 0
    total_images = 0
    steady_valid_tiles = 0
    steady_images = 0
    steady_time_s = 0.0

    sum_decode_compute_ms = 0.0
    sum_tile_compute_ms = 0.0
    sum_decode_wait_ms = 0.0
    sum_pre_ms = 0.0
    sum_submit_ms = 0.0
    sum_sync_wait_ms = 0.0
    sum_total_ms = 0.0

    cold_first_total_ms: Optional[float] = None

    from concurrent.futures import ThreadPoolExecutor

    decode_prefetch = max(1, min(args.decode_prefetch, len(files)))

    t_all0 = now()

    with ThreadPoolExecutor(max_workers=args._decode_workers) as decode_ex, \
            ThreadPoolExecutor(max_workers=args._pre_workers) as pre_ex, \
            open(args.out, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(header)
        f_out.flush()

        decode_q: Deque = deque()
        next_decode_idx = 0

        for _ in range(decode_prefetch):
            path = files[next_decode_idx]
            fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
            decode_q.append(fut)
            next_decode_idx += 1

        for img_i in range(len(files)):
            img_start = now()

            w0 = now()
            dec_res: DecodeTileResult = decode_q.popleft().result()
            w1 = now()
            decode_wait_ms = (w1 - w0) * 1000.0

            if next_decode_idx < len(files):
                path = files[next_decode_idx]
                fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
                decode_q.append(fut)
                next_decode_idx += 1

            tiles_uint8 = dec_res.tiles_uint8
            valid_tiles = int(tiles_uint8.shape[0])
            total_tiles_possible = int(dec_res.total_tiles_possible)

            bs = max(1, args.bs)
            num_batches = int((valid_tiles + bs - 1) // bs) if valid_tiles > 0 else 0

            t_pre_ms = 0.0
            pre_q: Deque = deque()

            def submit_pre_batch(start: int, end: int):
                sub = tiles_uint8[start:end]
                t0 = now()
                x_cpu = preprocess_tiles_uint8_vectorized(sub, out_fp16=args.cpu_out_fp16)
                t1 = now()
                return x_cpu, (t1 - t0) * 1000.0

            next_b = 0
            max_prefetch = max(1, args.prefetch)

            while next_b < num_batches and len(pre_q) < max_prefetch:
                b_start = next_b * bs
                b_end = min(valid_tiles, (next_b + 1) * bs)
                fut = pre_ex.submit(submit_pre_batch, b_start, b_end)
                pre_q.append(fut)
                next_b += 1

            inflight = False
            t_submit_ms = 0.0
            t_sync_wait_ms = 0.0
            note_parts: List[str] = []
            if args.global_mix:
                note_parts.append("global_mix=1")

            for _ in range(num_batches):
                fut = pre_q.popleft()
                x_cpu, pre_ms = fut.result()
                t_pre_ms += pre_ms

                while next_b < num_batches and len(pre_q) < max_prefetch:
                    b_start = next_b * bs
                    b_end = min(valid_tiles, (next_b + 1) * bs)
                    fut2 = pre_ex.submit(submit_pre_batch, b_start, b_end)
                    pre_q.append(fut2)
                    next_b += 1

                if inflight:
                    sw0 = now()
                    sync_npu()
                    sw1 = now()
                    t_sync_wait_ms += (sw1 - sw0) * 1000.0
                    inflight = False

                if warmup_pending:
                    sw0 = now()
                    sync_npu()
                    sw1 = now()
                    warmup_sync_ms = (sw1 - sw0) * 1000.0
                    warmup_pending = False
                    note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")

                sb0 = now()
                x = x_cpu.to(device, non_blocking=True)
                if use_amp:
                    with torch.npu.amp.autocast():
                        _ = model(x)
                else:
                    _ = model(x)
                sb1 = now()
                t_submit_ms += (sb1 - sb0) * 1000.0
                inflight = True

            if inflight:
                sw0 = now()
                sync_npu()
                sw1 = now()
                t_sync_wait_ms += (sw1 - sw0) * 1000.0
                inflight = False

            img_end = now()
            t_total_ms = (img_end - img_start) * 1000.0

            tile_per_s = (valid_tiles / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            img_per_s = (1.0 / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            avg_ms_per_tile_total = (t_total_ms / valid_tiles) if valid_tiles > 0 else 0.0
            pre_ms_per_tile = (t_pre_ms / valid_tiles) if valid_tiles > 0 else 0.0

            note = ";".join(note_parts)

            writer.writerow([
                dec_res.file, dec_res.W, dec_res.H, dec_res.mode,
                total_tiles_possible, valid_tiles, num_batches,
                f"{dec_res.decode_compute_ms:.3f}", f"{dec_res.tile_compute_ms:.3f}", f"{decode_wait_ms:.3f}",
                f"{t_pre_ms:.3f}", f"{t_submit_ms:.3f}", f"{t_sync_wait_ms:.3f}",
                f"{t_total_ms:.3f}",
                f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                note
            ])
            f_out.flush()

            if (img_i + 1) % args.log_every == 0 or (img_i + 1) == len(files):
                print(
                    f"[A2_v2] [{img_i+1}/{len(files)}] {dec_res.file} "
                    f"tiles={valid_tiles} total_ms={t_total_ms:.1f} tile/s={tile_per_s:.1f} "
                    f"decode_wait_ms={decode_wait_ms:.1f} pre_ms={t_pre_ms:.1f} "
                    f"submit_ms={t_submit_ms:.1f} sync_wait_ms={t_sync_wait_ms:.1f}"
                )

            total_valid_tiles += valid_tiles
            total_images += 1

            sum_decode_compute_ms += dec_res.decode_compute_ms
            sum_tile_compute_ms += dec_res.tile_compute_ms
            sum_decode_wait_ms += decode_wait_ms
            sum_pre_ms += t_pre_ms
            sum_submit_ms += t_submit_ms
            sum_sync_wait_ms += t_sync_wait_ms
            sum_total_ms += t_total_ms

            if cold_first_total_ms is None:
                cold_first_total_ms = t_total_ms

            if img_i >= args.skip_steady:
                steady_valid_tiles += valid_tiles
                steady_images += 1
                steady_time_s += (t_total_ms / 1000.0)

    t_all1 = now()
    total_time_s = (t_all1 - t_all0)

    overall_tile_s = total_valid_tiles / total_time_s
    overall_img_s = total_images / total_time_s

    steady_tile_s = (steady_valid_tiles / steady_time_s) if steady_time_s > 0 else 0.0
    steady_img_s = (steady_images / steady_time_s) if steady_time_s > 0 else 0.0

    def safe_div(a: float, b: int) -> float:
        return a / b if b > 0 else 0.0

    mean_decode_compute_ms = safe_div(sum_decode_compute_ms, total_images)
    mean_tile_compute_ms = safe_div(sum_tile_compute_ms, total_images)
    mean_decode_wait_ms = safe_div(sum_decode_wait_ms, total_images)
    mean_pre_ms = safe_div(sum_pre_ms, total_images)
    mean_submit_ms = safe_div(sum_submit_ms, total_images)
    mean_sync_wait_ms = safe_div(sum_sync_wait_ms, total_images)
    mean_total_ms = safe_div(sum_total_ms, total_images)

    return {
        "tiles_total": total_valid_tiles,
        "num_images": total_images,
        "total_time_s": total_time_s,
        "overall_tile_s": overall_tile_s,
        "overall_img_s": overall_img_s,
        "steady_tile_s": steady_tile_s,
        "steady_img_s": steady_img_s,
        "cold_first_ms": float(cold_first_total_ms or 0.0),
        "warmup_sync_ms": warmup_sync_ms,
        "mean_decode_compute_ms": mean_decode_compute_ms,
        "mean_tile_compute_ms": mean_tile_compute_ms,
        "mean_decode_wait_ms": mean_decode_wait_ms,
        "mean_pre_ms": mean_pre_ms,
        "mean_submit_ms": mean_submit_ms,
        "mean_sync_wait_ms": mean_sync_wait_ms,
        "mean_total_ms": mean_total_ms,
    }


# ----------------------------
# 7) global-mix 主流程
# ----------------------------

def run_global_mix(args: argparse.Namespace,
                   files: List[str],
                   model: torch.nn.Module,
                   device: str,
                   use_amp: bool,
                   warmup_pending: bool,
                   warmup_sync_ms: float) -> Dict:
    """
    global-mix batching：
    - decode+tile：按图片并行预取
    - 推理 batch：全局 tile 池混合凑满 bs
    仍输出 per-image TSV 与 summary。
    """
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    header = [
        "file", "W", "H", "mode",
        "total_tiles_possible", "valid_tiles", "num_batches",
        "t_decode_compute_ms", "t_tile_compute_ms", "t_decode_wait_ms",
        "t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        "t_total_ms",
        "tile_per_s", "img_per_s",
        "avg_ms_per_tile_total", "pre_ms_per_tile",
        "note"
    ]

    n_img = len(files)
    image_states: List[ImageState] = [ImageState(idx=i) for i in range(n_img)]

    # steady（wall-clock）统计
    first_tile_time: List[Optional[float]] = [None] * n_img
    last_tile_done_time: List[Optional[float]] = [None] * n_img

    from concurrent.futures import ThreadPoolExecutor

    decode_prefetch = max(1, min(args.decode_prefetch, n_img))
    bs = max(1, args.bs)
    max_prefetch_batches = max(1, args.prefetch)

    tile_pool: Deque[TileRef] = deque()
    pre_q: Deque[Tuple] = deque()  # (future, tile_refs)

    decode_q: Deque = deque()
    next_submit_idx = 0
    next_consume_idx = 0

    next_write_idx = 0
    written_images = 0

    def submit_decode(i: int):
        path = files[i]
        fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
        decode_q.append(fut)

    def consume_one_decode() -> None:
        nonlocal next_submit_idx, next_consume_idx
        if next_consume_idx >= n_img:
            return

        w0 = now()
        dec_res: DecodeTileResult = decode_q.popleft().result()
        w1 = now()
        decode_wait_ms = (w1 - w0) * 1000.0

        st = image_states[next_consume_idx]
        st.file = dec_res.file
        st.W = dec_res.W
        st.H = dec_res.H
        st.mode = dec_res.mode
        st.total_tiles_possible = int(dec_res.total_tiles_possible)
        st.decode_compute_ms = float(dec_res.decode_compute_ms)
        st.tile_compute_ms = float(dec_res.tile_compute_ms)
        st.decode_wait_ms = float(decode_wait_ms)
        st.tiles_uint8 = dec_res.tiles_uint8

        valid_tiles = int(dec_res.tiles_uint8.shape[0])
        st.valid_tiles = valid_tiles

        if first_tile_time[st.idx] is None:
            first_tile_time[st.idx] = now()

        if valid_tiles <= 0:
            st.done = True
            last_tile_done_time[st.idx] = now()
            st.tiles_uint8 = None
        else:
            for ti in range(valid_tiles):
                tile_pool.append(TileRef(img_idx=st.idx, tile_idx=ti))

        next_consume_idx += 1

        if next_submit_idx < n_img:
            submit_decode(next_submit_idx)
            next_submit_idx += 1

    def try_flush_done_rows(writer, f_out):
        nonlocal next_write_idx, written_images, warmup_sync_ms
        while next_write_idx < n_img and image_states[next_write_idx].done:
            st = image_states[next_write_idx]

            t_total_ms = (
                st.decode_compute_ms + st.tile_compute_ms + st.decode_wait_ms +
                st.pre_ms + st.submit_ms + st.sync_wait_ms
            )

            tile_per_s = (st.valid_tiles / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            img_per_s = (1.0 / (t_total_ms / 1000.0)) if t_total_ms > 0 else 0.0
            avg_ms_per_tile_total = (t_total_ms / st.valid_tiles) if st.valid_tiles > 0 else 0.0
            pre_ms_per_tile = (st.pre_ms / st.valid_tiles) if st.valid_tiles > 0 else 0.0

            note_parts = ["global_mix=1"]
            if st.idx == 0 and warmup_sync_ms > 0:
                note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")
            note = ";".join(note_parts)

            writer.writerow([
                st.file, st.W, st.H, st.mode,
                st.total_tiles_possible, st.valid_tiles, st.num_batches,
                f"{st.decode_compute_ms:.3f}", f"{st.tile_compute_ms:.3f}", f"{st.decode_wait_ms:.3f}",
                f"{st.pre_ms:.3f}", f"{st.submit_ms:.3f}", f"{st.sync_wait_ms:.3f}",
                f"{t_total_ms:.3f}",
                f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                note
            ])
            f_out.flush()

            written_images += 1
            if (written_images % args.log_every) == 0 or written_images == n_img:
                print(f"[A2_v2][GMIX] wrote {written_images}/{n_img} images, "
                      f"tile_pool={len(tile_pool)} pre_q={len(pre_q)}")

            next_write_idx += 1

    def distribute_time(tile_refs: List[TileRef], ms: float, field: str) -> None:
        if not tile_refs:
            return
        B = len(tile_refs)
        counts: Dict[int, int] = {}
        for ref in tile_refs:
            counts[ref.img_idx] = counts.get(ref.img_idx, 0) + 1
        for img_i, c in counts.items():
            st = image_states[img_i]
            share = ms * (c / B)
            if field == "pre_ms":
                st.pre_ms += share
            elif field == "submit_ms":
                st.submit_ms += share
            elif field == "sync_wait_ms":
                st.sync_wait_ms += share
            elif field == "num_batches":
                st.num_batches += 1
            else:
                raise ValueError(field)

    def mark_batch_done(tile_refs: List[TileRef]) -> None:
        if not tile_refs:
            return
        counts: Dict[int, int] = {}
        for ref in tile_refs:
            counts[ref.img_idx] = counts.get(ref.img_idx, 0) + 1
        t_done = now()
        for img_i, c in counts.items():
            st = image_states[img_i]
            st.processed_tiles += c
            if st.valid_tiles > 0 and st.processed_tiles >= st.valid_tiles and not st.done:
                st.done = True
                last_tile_done_time[img_i] = t_done
                st.tiles_uint8 = None

    t_all0 = now()
    inflight_refs: Optional[List[TileRef]] = None

    with ThreadPoolExecutor(max_workers=args._decode_workers) as decode_ex, \
            ThreadPoolExecutor(max_workers=args._pre_workers) as pre_ex, \
            open(args.out, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(header)
        f_out.flush()

        next_submit_idx = 0
        while next_submit_idx < decode_prefetch:
            submit_decode(next_submit_idx)
            next_submit_idx += 1

        while True:
            try_flush_done_rows(writer, f_out)

            # 填满 preprocess prefetch 队列
            while len(pre_q) < max_prefetch_batches:
                while len(tile_pool) < bs and next_consume_idx < n_img:
                    consume_one_decode()
                    try_flush_done_rows(writer, f_out)

                if len(tile_pool) >= bs:
                    batch_size = bs
                else:
                    if next_consume_idx >= n_img and len(tile_pool) > 0:
                        batch_size = len(tile_pool)
                    else:
                        break

                tile_refs = [tile_pool.popleft() for _ in range(batch_size)]
                distribute_time(tile_refs, 0.0, "num_batches")

                fut = pre_ex.submit(preprocess_mixed_batch, tile_refs, image_states, args.cpu_out_fp16)
                pre_q.append((fut, tile_refs))

            if pre_q:
                fut, tile_refs = pre_q.popleft()
                x_cpu, pre_ms = fut.result()
                distribute_time(tile_refs, pre_ms, "pre_ms")

                # 提交下一批前：先等上一批 inflight 完成，并把 sync_wait 计入上一批
                if inflight_refs is not None:
                    sw0 = now()
                    sync_npu()
                    sw1 = now()
                    sync_wait_ms = (sw1 - sw0) * 1000.0
                    distribute_time(inflight_refs, sync_wait_ms, "sync_wait_ms")
                    mark_batch_done(inflight_refs)
                    inflight_refs = None
                    try_flush_done_rows(writer, f_out)

                if warmup_pending:
                    sw0 = now()
                    sync_npu()
                    sw1 = now()
                    warmup_sync_ms = (sw1 - sw0) * 1000.0
                    warmup_pending = False

                sb0 = now()
                x = x_cpu.to(device, non_blocking=True)
                if use_amp:
                    with torch.npu.amp.autocast():
                        _ = model(x)
                else:
                    _ = model(x)
                sb1 = now()
                submit_ms = (sb1 - sb0) * 1000.0
                distribute_time(tile_refs, submit_ms, "submit_ms")

                inflight_refs = tile_refs
                continue

            if next_consume_idx < n_img:
                consume_one_decode()
                continue

            if inflight_refs is not None:
                sw0 = now()
                sync_npu()
                sw1 = now()
                sync_wait_ms = (sw1 - sw0) * 1000.0
                distribute_time(inflight_refs, sync_wait_ms, "sync_wait_ms")
                mark_batch_done(inflight_refs)
                inflight_refs = None
                try_flush_done_rows(writer, f_out)
                continue

            if written_images >= n_img and not tile_pool and not pre_q and inflight_refs is None:
                break

        try_flush_done_rows(writer, f_out)

    t_all1 = now()
    total_time_s = (t_all1 - t_all0)

    total_valid_tiles = sum(st.valid_tiles for st in image_states)
    total_images = n_img

    overall_tile_s = (total_valid_tiles / total_time_s) if total_time_s > 0 else 0.0
    overall_img_s = (total_images / total_time_s) if total_time_s > 0 else 0.0

    steady_ids = [i for i in range(n_img) if i >= args.skip_steady]
    steady_valid_tiles = sum(image_states[i].valid_tiles for i in steady_ids)

    t0_candidates = [first_tile_time[i] for i in steady_ids if first_tile_time[i] is not None]
    t1_candidates = [last_tile_done_time[i] for i in steady_ids if last_tile_done_time[i] is not None]

    if t0_candidates and t1_candidates:
        steady_time_s = max(t1_candidates) - min(t0_candidates)
    else:
        steady_time_s = 0.0

    steady_tile_s = (steady_valid_tiles / steady_time_s) if steady_time_s > 0 else 0.0
    steady_img_s = ((len(steady_ids)) / steady_time_s) if steady_time_s > 0 else 0.0

    def safe_div(a: float, b: int) -> float:
        return a / b if b > 0 else 0.0

    sum_decode_compute_ms = sum(st.decode_compute_ms for st in image_states)
    sum_tile_compute_ms = sum(st.tile_compute_ms for st in image_states)
    sum_decode_wait_ms = sum(st.decode_wait_ms for st in image_states)
    sum_pre_ms = sum(st.pre_ms for st in image_states)
    sum_submit_ms = sum(st.submit_ms for st in image_states)
    sum_sync_wait_ms = sum(st.sync_wait_ms for st in image_states)
    sum_total_ms = sum(
        st.decode_compute_ms + st.tile_compute_ms + st.decode_wait_ms +
        st.pre_ms + st.submit_ms + st.sync_wait_ms
        for st in image_states
    )

    mean_decode_compute_ms = safe_div(sum_decode_compute_ms, total_images)
    mean_tile_compute_ms = safe_div(sum_tile_compute_ms, total_images)
    mean_decode_wait_ms = safe_div(sum_decode_wait_ms, total_images)
    mean_pre_ms = safe_div(sum_pre_ms, total_images)
    mean_submit_ms = safe_div(sum_submit_ms, total_images)
    mean_sync_wait_ms = safe_div(sum_sync_wait_ms, total_images)
    mean_total_ms = safe_div(sum_total_ms, total_images)

    cold_first_ms = (
        image_states[0].decode_compute_ms + image_states[0].tile_compute_ms + image_states[0].decode_wait_ms +
        image_states[0].pre_ms + image_states[0].submit_ms + image_states[0].sync_wait_ms
    ) if image_states else 0.0

    return {
        "tiles_total": int(total_valid_tiles),
        "num_images": int(total_images),
        "total_time_s": float(total_time_s),
        "overall_tile_s": float(overall_tile_s),
        "overall_img_s": float(overall_img_s),
        "steady_tile_s": float(steady_tile_s),
        "steady_img_s": float(steady_img_s),
        "cold_first_ms": float(cold_first_ms),
        "warmup_sync_ms": float(warmup_sync_ms),
        "mean_decode_compute_ms": float(mean_decode_compute_ms),
        "mean_tile_compute_ms": float(mean_tile_compute_ms),
        "mean_decode_wait_ms": float(mean_decode_wait_ms),
        "mean_pre_ms": float(mean_pre_ms),
        "mean_submit_ms": float(mean_submit_ms),
        "mean_sync_wait_ms": float(mean_sync_wait_ms),
        "mean_total_ms": float(mean_total_ms),
    }


# ----------------------------
# 8) main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--file_list", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")

    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)

    ap.add_argument("--bs", type=int, default=96)

    # global mix 默认开启
    ap.add_argument("--global_mix", type=int, default=1, choices=[0, 1],
                    help="1=global-mix batching（跨图混 batch 凑满 bs），0=旧的单图口径")

    ap.add_argument("--white_thr", type=float, default=1.01)

    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--cpu_out_fp16", action="store_true")
    ap.set_defaults(cpu_out_fp16=True)

    ap.add_argument("--cpu_workers", type=int, default=8)

    ap.add_argument("--decode_workers", type=int, default=-1,
                    help="-1=自动分配，否则手动指定 decode 线程数")
    ap.add_argument("--decode_prefetch", type=int, default=4,
                    help="预取多少张图做 decode+tile（窗口越大越占内存）")

    ap.add_argument("--prefetch", type=int, default=3)

    ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], default="async")
    ap.add_argument("--warmup_iters", type=int, default=3)

    ap.add_argument("--skip_steady", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)

    ap.add_argument("--out", type=str, default="logs/A2_final_v2.tsv")
    ap.add_argument("--summary_json", type=str, default="")
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()

    with open(args.file_list, "r", encoding="utf-8") as f:
        files = [line.strip() for line in f if line.strip()]

    if args.max_images > 0:
        files = files[:args.max_images]

    if not files:
        raise ValueError("file_list is empty")

    if args.decode_workers is None or args.decode_workers < 0:
        decode_workers = max(1, min(4, args.cpu_workers // 4))
    else:
        decode_workers = max(1, args.decode_workers)

    pre_workers = max(1, args.cpu_workers - decode_workers)

    args._decode_workers = decode_workers
    args._pre_workers = pre_workers

    device = "npu:0"
    model = build_uni_backbone(args.ckpt_dir, device=device, img_size=args.tile)

    use_amp = (args.precision == "fp16")

    warmup_pending = False
    warmup_sync_ms = 0.0

    warm_bs = min(args.bs, 64)
    x_warm = torch.randn(warm_bs, 3, args.tile, args.tile, device=device)

    if args.warmup_mode == "sync":
        for _ in range(args.warmup_iters):
            if use_amp:
                with torch.npu.amp.autocast():
                    _ = model(x_warm)
            else:
                _ = model(x_warm)
        sync_npu()
        warmup_pending = False

    elif args.warmup_mode == "async":
        for _ in range(args.warmup_iters):
            if use_amp:
                with torch.npu.amp.autocast():
                    _ = model(x_warm)
            else:
                _ = model(x_warm)
        warmup_pending = True
    else:
        warmup_pending = False

    # ====== 优化：利用 warmup 时间将所有图片预读到 OS page cache ======
    import threading
    def _preload_files(file_list):
        for fpath in file_list:
            try:
                with open(fpath, "rb") as f:
                    f.read()  # 仅读入内存不保留，触发系统缓冲
            except Exception:
                pass

    if warmup_pending or args.warmup_mode == "sync":
        preload_thread = threading.Thread(target=_preload_files, args=(files,), daemon=True)
        preload_thread.start()
        print(f"[预热] 已启动后台线程预读 {len(files)} 张图片到 OS Cache...")

    if args.global_mix == 1:
        result = run_global_mix(args, files, model, device, use_amp, warmup_pending, warmup_sync_ms)
    else:
        result = run_per_image(args, files, model, device, use_amp, warmup_pending, warmup_sync_ms)

    print("\n================ [A2_v2 SUMMARY] ================")
    print(f"file_list         : {args.file_list}")
    print(f"num_images        : {result['num_images']}")
    print(f"tile/stride       : {args.tile}/{args.stride}")
    print(f"white_thr         : {args.white_thr}")
    print(f"precision         : {args.precision} (cpu_out_fp16={args.cpu_out_fp16})")
    print(f"global_mix        : {args.global_mix}")
    print(f"bs                : {args.bs}")
    print(f"cpu_workers       : {args.cpu_workers} (decode={decode_workers}, pre={pre_workers})")
    print(f"decode_prefetch   : {min(args.decode_prefetch, len(files))}")
    print(f"prefetch(batches) : {args.prefetch}")
    print(f"warmup_mode       : {args.warmup_mode} (iters={args.warmup_iters})")
    print(f"skip_steady       : {args.skip_steady}")
    print("-----------------------------------------------")
    print(f"tiles_total       : {result['tiles_total']}")
    print(f"total_time_s      : {result['total_time_s']:.3f}")
    print(f"overall_tile/s    : {result['overall_tile_s']:.3f}")
    print(f"steady_tile/s     : {result['steady_tile_s']:.3f}")
    print(f"overall_img/s     : {result['overall_img_s']:.3f}")
    print(f"steady_img/s      : {result['steady_img_s']:.3f}")
    print("-----------------------------------------------")
    print(f"cold_first_ms     : {result['cold_first_ms']:.2f}")
    if result.get("warmup_sync_ms", 0.0) > 0:
        print(f"warmup_sync_ms    : {result['warmup_sync_ms']:.2f}")
    print("-----------------------------------------------")
    print(f"mean_decode_compute_ms : {result['mean_decode_compute_ms']:.2f}")
    print(f"mean_tile_compute_ms   : {result['mean_tile_compute_ms']:.2f}")
    print(f"mean_decode_wait_ms    : {result['mean_decode_wait_ms']:.2f}")
    print(f"mean_pre_ms            : {result['mean_pre_ms']:.2f}")
    print(f"mean_submit_ms         : {result['mean_submit_ms']:.2f}")
    print(f"mean_sync_wait_ms      : {result['mean_sync_wait_ms']:.2f}")
    print(f"mean_total_ms          : {result['mean_total_ms']:.2f}")
    print("================================================\n")

    if args.summary_json:
        summary = {
            "config": {
                "file_list": args.file_list,
                "ckpt_dir": args.ckpt_dir,
                "tile": args.tile,
                "stride": args.stride,
                "bs": args.bs,
                "global_mix": args.global_mix,
                "white_thr": args.white_thr,
                "precision": args.precision,
                "cpu_out_fp16": args.cpu_out_fp16,
                "cpu_workers": args.cpu_workers,
                "decode_workers": decode_workers,
                "pre_workers": pre_workers,
                "decode_prefetch": min(args.decode_prefetch, len(files)),
                "prefetch": args.prefetch,
                "warmup_mode": args.warmup_mode,
                "warmup_iters": args.warmup_iters,
                "skip_steady": args.skip_steady,
                "max_images": args.max_images,
                "out_tsv": args.out,
            },
            "result": result
        }
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[A2_v2] wrote summary_json -> {args.summary_json}")


if __name__ == "__main__":
    main()