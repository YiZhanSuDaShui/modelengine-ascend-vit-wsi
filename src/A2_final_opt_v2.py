#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A2_final_opt_v2.py
==================
端到端 A2（含 IO/预处理/搬运/推理）性能评测与优化版本。

本脚本定位：
- 单图口径（per-image）：每次只处理一张图的 tiles，并在该图内部按 batch 切分推理；
- 通过：
  1) decode + tile 并行化（线程池）
  2) batch preprocess 预取（线程池）
  3) CPU/NPU 重叠（prefetch + NPU 异步执行）
  4) warmup async（把 warmup 的等待“藏”到 CPU 工作中）
  来尽量减少 NPU 空等。

注意：
- 你现在固定 tiles_total=5400 的对比口径，是为了控制变量做 A2 端到端对比。
- 本脚本仍是“单图口径”，因此在 BACH Photos（单图最多 54 tiles）上：
  即使 bs=96，effective_bs 也只能到 54（这对你当前“找 CPU 最优 workers/prefetch”仍然有意义）。
"""

import os
import time
import csv
import json
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional

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

    # 设置 NPU 设备
    dev = torch.device(device)
    torch.npu.set_device(dev)

    # 创建模型
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=img_size,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    # 加载权重
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    # eval + 放到 NPU
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

    # 裁掉边缘不能整除的部分
    H_use = ny * tile
    W_use = nx * tile
    crop = img_np[:H_use, :W_use, :]

    # (ny, tile, nx, tile, 3) -> (ny, nx, tile, tile, 3) -> (N, tile, tile, 3)
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

    # decode
    with Image.open(path) as im:
        im = im.convert("RGB")
        img_np = np.array(im)

    t1 = now()

    # tilegen
    if stride == tile:
        mode = "fast_grid"
        tiles_uint8, total = extract_tiles_fast_grid(img_np, tile=tile, stride=stride)

        # white_thr>=1.0 -> 不过滤，直接跳过 mask 计算（省时间）
        if total > 0 and white_thr < 1.0:
            keep = tissue_mask_uint8(tiles_uint8, white_thr=white_thr)
            tiles_uint8 = tiles_uint8[keep]
    else:
        # fallback（慢）：为了功能完整保留
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
        tiles_uint8 = np.stack(tiles_list, axis=0).astype(np.uint8)

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
# 5) A2 主流程：逐图计时 + CPU/NPU 流水
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    # 输入
    ap.add_argument("--file_list", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")

    # tile / batch
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)

    # ✅ 默认 bs=96（你已确定 A1 sweet spot=96；这里用于固定口径做 CPU sweep）
    ap.add_argument("--bs", type=int, default=96)

    # 白底过滤（你目前固定 1.01 = 不过滤）
    ap.add_argument("--white_thr", type=float, default=1.01)

    # 精度
    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--cpu_out_fp16", action="store_true")
    ap.set_defaults(cpu_out_fp16=True)

    # CPU 线程预算（decode + preprocess 总和）
    ap.add_argument("--cpu_workers", type=int, default=8)

    # decode 并行参数
    ap.add_argument("--decode_workers", type=int, default=-1,
                    help="-1=自动分配，否则手动指定 decode 线程数")
    ap.add_argument("--decode_prefetch", type=int, default=4,
                    help="预取多少张图做 decode+tile（窗口越大越占内存）")

    # preprocess 预取（每张图内提前提交多少个 batch 的 preprocess）
    ap.add_argument("--prefetch", type=int, default=3)

    # warmup
    ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], default="async")
    ap.add_argument("--warmup_iters", type=int, default=3)

    # steady 统计
    ap.add_argument("--skip_steady", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)

    # 输出
    ap.add_argument("--out", type=str, default="logs/A2_final_v2.tsv")
    ap.add_argument("--summary_json", type=str, default="")
    ap.add_argument("--log_every", type=int, default=10)

    args = ap.parse_args()

    # 读 file list
    with open(args.file_list, "r", encoding="utf-8") as f:
        files = [ln.strip() for ln in f.readlines() if ln.strip()]

    if args.max_images > 0:
        files = files[:args.max_images]

    if not files:
        raise ValueError("file_list is empty")

    # 自动分配 decode/pre 线程
    if args.decode_workers is None or args.decode_workers < 0:
        # decode 给约 25% 预算，上限 4（经验值，减少线程争用）
        decode_workers = max(1, min(4, args.cpu_workers // 4))
    else:
        decode_workers = max(1, args.decode_workers)

    pre_workers = max(1, args.cpu_workers - decode_workers)

    # 构建模型
    device = "npu:0"
    model = build_uni_backbone(args.ckpt_dir, device=device, img_size=args.tile)

    # 是否 AMP
    use_amp = (args.precision == "fp16")

    # warmup（sync / async / none）
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
        # 先把 warmup 丢进 NPU 队列，但不等待；CPU 同时开始 decode/tile/pre
        for _ in range(args.warmup_iters):
            if use_amp:
                with torch.npu.amp.autocast():
                    _ = model(x_warm)
            else:
                _ = model(x_warm)
        warmup_pending = True

    else:
        warmup_pending = False

    # 输出 TSV 准备
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

    # summary 累加器
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

    with ThreadPoolExecutor(max_workers=decode_workers) as decode_ex, \
         ThreadPoolExecutor(max_workers=pre_workers) as pre_ex, \
         open(args.out, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(header)
        f_out.flush()

        # 先提交 decode+tile 预取任务（窗口）
        decode_q: Deque = deque()
        next_decode_idx = 0

        for _ in range(decode_prefetch):
            path = files[next_decode_idx]
            fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
            decode_q.append(fut)
            next_decode_idx += 1

        # 主循环：逐图消费 decode_q（不会一次性把全部图片都 decode 完）
        for img_i in range(len(files)):
            img_start = now()

            # 取出下一张（可能会等待）
            w0 = now()
            dec_res: DecodeTileResult = decode_q.popleft().result()
            w1 = now()
            decode_wait_ms = (w1 - w0) * 1000.0

            # 补充下一张 decode 任务，维持窗口
            if next_decode_idx < len(files):
                path = files[next_decode_idx]
                fut = decode_ex.submit(decode_and_tile_worker, path, args.tile, args.stride, args.white_thr)
                decode_q.append(fut)
                next_decode_idx += 1

            tiles_uint8 = dec_res.tiles_uint8
            valid_tiles = int(tiles_uint8.shape[0])
            total_tiles_possible = int(dec_res.total_tiles_possible)

            # 这行就是你说的“effective bs”：
            # 单图口径下，实际每批最多就是 min(bs, valid_tiles)
            bs = max(1, args.bs)
            num_batches = int((valid_tiles + bs - 1) // bs)

            # --- preprocess 预取队列（同一张图内不同 batch） ---
            t_pre_ms = 0.0
            pre_q: Deque = deque()

            def submit_pre_batch(start: int, end: int):
                """
                在线程池里做 preprocess（向量化），返回：
                - x_cpu：CPU tensor
                - pre_ms：本 batch preprocess 耗时
                """
                sub = tiles_uint8[start:end]

                t0 = now()
                x_cpu = preprocess_tiles_uint8_vectorized(sub, out_fp16=args.cpu_out_fp16)
                t1 = now()

                return x_cpu, (t1 - t0) * 1000.0, (end - start)

            next_b = 0
            max_prefetch = max(1, args.prefetch)

            # 先提交 prefetch 个 batch
            while next_b < num_batches and len(pre_q) < max_prefetch:
                b_start = next_b * bs
                b_end = min(valid_tiles, (next_b + 1) * bs)
                fut = pre_ex.submit(submit_pre_batch, b_start, b_end)
                pre_q.append(fut)
                next_b += 1

            # --- NPU 推理流水：最多 1 个 inflight ---
            inflight = False
            t_submit_ms = 0.0
            t_sync_wait_ms = 0.0
            note_parts: List[str] = []

            for _ in range(num_batches):
                # 取一个已完成 preprocess 的 batch
                fut = pre_q.popleft()
                x_cpu, pre_ms, _ = fut.result()
                t_pre_ms += pre_ms

                # 继续补足 prefetch
                while next_b < num_batches and len(pre_q) < max_prefetch:
                    b_start = next_b * bs
                    b_end = min(valid_tiles, (next_b + 1) * bs)
                    fut2 = pre_ex.submit(submit_pre_batch, b_start, b_end)
                    pre_q.append(fut2)
                    next_b += 1

                # 等上一批 NPU 完成（让计时可解释）
                if inflight:
                    sw0 = now()
                    sync_npu()
                    sw1 = now()
                    t_sync_wait_ms += (sw1 - sw0) * 1000.0
                    inflight = False

                # warmup async：第一次真正提交 batch 前再 sync 一次
                # -> warmup 的等待被 CPU decode/tile/pre 吃掉，首图 cold 会更小
                if warmup_pending:
                    sw0 = now()
                    sync_npu()
                    sw1 = now()
                    warmup_sync_ms = (sw1 - sw0) * 1000.0
                    warmup_pending = False
                    note_parts.append(f"warmup_sync_ms={warmup_sync_ms:.2f}")

                # H2D + forward（不 sync）
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

            # 收尾 sync：确保最后一批完成
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

            # 写 per-image TSV（实时落盘）
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

            # 控制台进度（满足“运行中必须有中间输出”的硬要求）
            if (img_i + 1) % args.log_every == 0 or (img_i + 1) == len(files):
                print(
                    f"[A2_v2] [{img_i+1}/{len(files)}] {dec_res.file} "
                    f"tiles={valid_tiles} total_ms={t_total_ms:.1f} tile/s={tile_per_s:.1f} "
                    f"decode_wait_ms={decode_wait_ms:.1f} pre_ms={t_pre_ms:.1f} "
                    f"submit_ms={t_submit_ms:.1f} sync_wait_ms={t_sync_wait_ms:.1f}"
                )

            # summary 累加
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

            # steady：跳过前 skip_steady 张
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

    # 控制台 summary
    print("\n================ [A2_v2 SUMMARY] ================")
    print(f"file_list         : {args.file_list}")
    print(f"num_images        : {total_images}")
    print(f"tile/stride       : {args.tile}/{args.stride}")
    print(f"white_thr         : {args.white_thr}")
    print(f"precision         : {args.precision} (cpu_out_fp16={args.cpu_out_fp16})")
    print(f"bs                : {args.bs} (单图口径 effective_bs<=valid_tiles)")
    print(f"cpu_workers       : {args.cpu_workers} (decode={decode_workers}, pre={pre_workers})")
    print(f"decode_prefetch   : {min(args.decode_prefetch, len(files))}")
    print(f"prefetch(batches) : {args.prefetch}")
    print(f"warmup_mode/iters : {args.warmup_mode}/{args.warmup_iters} (warmup_sync_ms={warmup_sync_ms:.2f})")
    print(f"out_tsv           : {args.out}")
    print("-------------------------------------------------")
    print(f"tiles_total       : {total_valid_tiles}")
    print(f"total_time_s      : {total_time_s:.4f}")
    print(f"overall tile/s    : {overall_tile_s:.2f}")
    print(f"overall img/s     : {overall_img_s:.4f}")
    if args.skip_steady > 0:
        print(f"steady(skip {args.skip_steady} imgs) tile/s : {steady_tile_s:.2f}")
        print(f"steady(skip {args.skip_steady} imgs) img/s  : {steady_img_s:.4f}")
    if cold_first_total_ms is not None:
        print(f"cold_first_ms     : {cold_first_total_ms:.2f}")
    print("-------------------------------------------------")
    print("mean per-image stage (ms):")
    print(f"  decode_compute_ms : {mean_decode_compute_ms:.2f}")
    print(f"  tile_compute_ms   : {mean_tile_compute_ms:.2f}")
    print(f"  decode_wait_ms    : {mean_decode_wait_ms:.2f}   (越小越说明 decode/tile 被隐藏得越好)")
    print(f"  pre_ms            : {mean_pre_ms:.2f}")
    print(f"  submit_ms         : {mean_submit_ms:.2f}")
    print(f"  sync_wait_ms      : {mean_sync_wait_ms:.2f}   (越小越说明 NPU 在等 CPU)")
    print(f"  total_ms          : {mean_total_ms:.2f}")
    print("=================================================\n")

    # summary JSON（给 sweep 脚本读取）
    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        summary = {
            "config": {
                "file_list": args.file_list,
                "tile": args.tile,
                "stride": args.stride,
                "bs": args.bs,
                "white_thr": args.white_thr,
                "precision": args.precision,
                "cpu_out_fp16": bool(args.cpu_out_fp16),
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
            "result": {
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
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[A2_v2] wrote summary_json -> {args.summary_json}")


if __name__ == "__main__":
    main()