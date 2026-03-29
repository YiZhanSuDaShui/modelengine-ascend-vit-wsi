import os
import time
import csv
import argparse
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import torch
import torch_npu
import timm


def now() -> float:
    return time.perf_counter()

def sync():
    torch_npu.npu.synchronize()


# -----------------------------
# UNI: 复用 test_uni_npu.py
# -----------------------------
def build_uni_model(ckpt_dir: str, device: torch.device, img_size: int = 224) -> torch.nn.Module:
    ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    assert os.path.exists(ckpt_path), f"missing: {ckpt_path}"

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
    model.eval().to(device)
    return model


# -----------------------------
# 白底过滤（向量化友好）
# -----------------------------
def tissue_mask_uint8(tiles_uint8: np.ndarray, white_ratio_thr: float) -> np.ndarray:
    """
    tiles_uint8: (N,H,W,3) uint8
    使用 min(channel) > 240 判断“白像素”（比 mean 更快、无需 float）
    返回：keep mask (N,)
    """
    # (N,H,W)
    min_ch = tiles_uint8.min(axis=-1)
    white_ratio = (min_ch > 240).mean(axis=(1, 2))
    keep = white_ratio < white_ratio_thr
    return keep


def chunk_indices(n: int, bs: int) -> List[slice]:
    return [slice(i, min(i + bs, n)) for i in range(0, n, bs)]


# -----------------------------
# FAST PATH: stride==tile 的网格切块向量化
# -----------------------------
def extract_tiles_fast_grid(img_np: np.ndarray, tile: int, stride: int) -> Tuple[np.ndarray, int, int, int]:
    """
    仅在 stride == tile 时使用：
    返回 tiles_uint8 shape (N,tile,tile,3)，以及 total_tiles_possible, nx, ny
    注意：只取到能完整覆盖 tile 的区域（右边/下边剩余像素丢弃），与原公式一致：
      nx = (W-tile)//stride + 1
      ny = (H-tile)//stride + 1
      W_use = tile + (nx-1)*stride
      H_use = tile + (ny-1)*stride
    """
    H, W, _ = img_np.shape
    ny = max(0, (H - tile) // stride + 1)
    nx = max(0, (W - tile) // stride + 1)
    total = nx * ny

    if total == 0:
        return np.empty((0, tile, tile, 3), dtype=np.uint8), 0, nx, ny

    H_use = tile + (ny - 1) * stride
    W_use = tile + (nx - 1) * stride
    cropped = img_np[:H_use, :W_use, :]  # (H_use, W_use, 3)

    # reshape -> (ny, tile, nx, tile, 3) -> transpose -> (ny, nx, tile, tile, 3) -> flatten
    tiles = cropped.reshape(ny, tile, nx, tile, 3).transpose(0, 2, 1, 3, 4).reshape(total, tile, tile, 3)
    return tiles, total, nx, ny


# -----------------------------
# Fallback: 任意 stride 的坐标生成（保底）
# -----------------------------
def gen_coords_fallback(img_np: np.ndarray, tile: int, stride: int, white_ratio_thr: float) -> Tuple[List[Tuple[int, int]], int]:
    H, W, _ = img_np.shape
    ny = max(0, (H - tile) // stride + 1)
    nx = max(0, (W - tile) // stride + 1)
    total = nx * ny
    coords = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            patch = img_np[y:y+tile, x:x+tile]
            # 快速白底：min(channel) > 240
            min_ch = patch.min(axis=-1)
            white_ratio = (min_ch > 240).mean()
            if white_ratio < white_ratio_thr:
                coords.append((x, y))
    return coords, total


# -----------------------------
# C1: 向量化预处理（按 batch），输入 tiles_uint8（避免 PIL crop）
# -----------------------------
def preprocess_tiles_uint8_vectorized(
    tiles_uint8: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    out_dtype: np.dtype,
    pin_memory: bool,
) -> Tuple[torch.Tensor, float]:
    """
    tiles_uint8: (B,tile,tile,3) uint8
    输出：x_cpu (B,3,tile,tile) CPU tensor
    """
    t0 = now()
    arr = tiles_uint8.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    x_cpu = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

    if pin_memory:
        try:
            x_cpu = x_cpu.pin_memory()
        except Exception:
            pass

    if out_dtype == np.float16:
        x_cpu = x_cpu.half()

    t1 = now()
    return x_cpu, (t1 - t0) * 1000.0


# -----------------------------
# 主函数
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_list", required=True)
    ap.add_argument("--ckpt_dir", default="./assets/ckpts/UNI")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--white_thr", type=float, default=1.01, help="白底过滤阈值，越小越严格，>1.0 则不过滤")
    ap.add_argument("--cpu_workers", type=int, default=8)
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--cpu_out_fp16", action="store_true")
    ap.add_argument("--warmup_iters", type=int, default=3)
    ap.add_argument("--warmup_mode", choices=["none", "sync", "async"], default="async",
                    help="none:不warmup；sync:先warmup+sync再开始；async:enqueue warmup，第一次用NPU前sync（会记到第一张）")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--skip_steady", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert args.tile % 16 == 0, "tile 必须能被 16 整除（ViT patch_size=16）"

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    with open(args.file_list, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    if args.max_images > 0:
        files = files[:args.max_images]
    assert files, "file_list 为空"

    model = build_uni_model(args.ckpt_dir, device=device, img_size=args.tile)
    use_amp = (args.precision == "fp16")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    out_dtype = np.float16 if args.cpu_out_fp16 else np.float32

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # warmup
    warmup_submit_ms = 0.0
    warmup_sync_ms = 0.0
    warmup_pending = False

    if args.warmup_mode != "none" and args.warmup_iters > 0:
        x_warm = torch.randn(args.bs, 3, args.tile, args.tile, device=device)
        t0 = now()
        for _ in range(args.warmup_iters):
            if use_amp:
                with torch.npu.amp.autocast():
                    _ = model(x_warm)
            else:
                _ = model(x_warm)
        t1 = now()
        warmup_submit_ms = (t1 - t0) * 1000.0

        if args.warmup_mode == "sync":
            s0 = now()
            sync()
            s1 = now()
            warmup_sync_ms = (s1 - s0) * 1000.0
        elif args.warmup_mode == "async":
            warmup_pending = True

    print("\n=== A2 FINAL START ===")
    print(f"device={device}, precision={args.precision}, tile={args.tile}, stride={args.stride}, bs={args.bs}")
    print(f"cpu_workers={args.cpu_workers}, prefetch={args.prefetch}, pin_memory={args.pin_memory}, cpu_out_fp16={args.cpu_out_fp16}")
    print(f"warmup_mode={args.warmup_mode}, warmup_submit_ms={warmup_submit_ms:.2f}, warmup_sync_ms={warmup_sync_ms:.2f}\n")

    header = [
        "file", "W", "H",
        "mode", "total_tiles_possible", "valid_tiles", "num_batches",
        "t_decode_ms", "t_tile_ms", "t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        "t_total_ms", "tile_per_s", "img_per_s",
        "avg_ms_per_tile_total", "pre_ms_per_tile",
        "note"
    ]

    global_t0 = now()
    global_tiles = 0
    global_images = 0

    steady_tiles = 0
    steady_time_s = 0.0
    cold_first_total_ms: Optional[float] = None

    with open(args.out, "w", newline="") as f_out:
        w = csv.writer(f_out, delimiter="\t")
        w.writerow(header)

        with ThreadPoolExecutor(max_workers=args.cpu_workers) as ex:
            for idx, path in enumerate(files, 1):
                note = ""
                img_t0 = now()

                # decode
                d0 = now()
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)  # uint8
                d1 = now()
                H, W, _ = img_np.shape
                t_decode_ms = (d1 - d0) * 1000.0

                # warmup async：第一次真正用 NPU 前 sync
                if warmup_pending:
                    s0 = now()
                    sync()
                    s1 = now()
                    warmup_sync_ms = (s1 - s0) * 1000.0
                    warmup_pending = False
                    note = f"warmup_sync_ms={warmup_sync_ms:.2f}"

                # tilegen + tiles extraction
                t_tile_ms = 0.0
                mode = "fallback"
                tiles_all_uint8 = None
                total_tiles_possible = 0

                g0 = now()
                if args.stride == args.tile:
                    mode = "fast_grid"
                    tiles_all_uint8, total_tiles_possible, _, _ = extract_tiles_fast_grid(img_np, args.tile, args.stride)
                    if total_tiles_possible > 0:
                        keep = tissue_mask_uint8(tiles_all_uint8, args.white_thr)
                        tiles_all_uint8 = tiles_all_uint8[keep]
                else:
                    # fallback：只得到 coords，然后逐 batch slice 成 tiles
                    coords, total_tiles_possible = gen_coords_fallback(img_np, args.tile, args.stride, args.white_thr)
                    # 这里为了复用向量化预处理，仍然先按 batch 把 tiles 取出来（下面会做）
                    tiles_all_uint8 = coords  # 暂存 coords
                g1 = now()
                t_tile_ms = (g1 - g0) * 1000.0

                # valid tiles
                if mode == "fast_grid":
                    valid_tiles = int(tiles_all_uint8.shape[0])
                    batches_slices = chunk_indices(valid_tiles, args.bs)
                else:
                    coords = tiles_all_uint8
                    valid_tiles = len(coords)
                    batches_slices = [coords[i:i+args.bs] for i in range(0, valid_tiles, args.bs)]

                if valid_tiles == 0:
                    img_t1 = now()
                    t_total_ms = (img_t1 - img_t0) * 1000.0
                    w.writerow([
                        os.path.basename(path), W, H,
                        mode, total_tiles_possible, 0, 0,
                        f"{t_decode_ms:.3f}", f"{t_tile_ms:.3f}", f"{0.0:.3f}", f"{0.0:.3f}", f"{0.0:.3f}",
                        f"{t_total_ms:.3f}", f"{0.0:.3f}", f"{(1.0/(t_total_ms/1000.0)):.6f}",
                        f"{0.0:.6f}", f"{0.0:.6f}",
                        "all background"
                    ])
                    f_out.flush()
                    global_images += 1
                    if idx == 1:
                        cold_first_total_ms = t_total_ms
                    continue

                num_batches = len(batches_slices)

                # 预取任务：返回 (x_cpu, pre_ms)
                def submit_pre_task(batch_obj):
                    if mode == "fast_grid":
                        # batch_obj is slice
                        sub_tiles = tiles_all_uint8[batch_obj]  # (B,224,224,3) uint8
                        return preprocess_tiles_uint8_vectorized(sub_tiles, mean, std, out_dtype, args.pin_memory)
                    else:
                        # batch_obj is list of coords
                        buf = np.empty((len(batch_obj), args.tile, args.tile, 3), dtype=np.uint8)
                        for ii, (x, y) in enumerate(batch_obj):
                            buf[ii] = img_np[y:y+args.tile, x:x+args.tile]
                        return preprocess_tiles_uint8_vectorized(buf, mean, std, out_dtype, args.pin_memory)

                # prefetch window
                futures = []
                next_k = 0
                while next_k < num_batches and len(futures) < args.prefetch:
                    fut = ex.submit(submit_pre_task, batches_slices[next_k])
                    futures.append(fut)
                    next_k += 1

                t_pre_ms = 0.0
                t_submit_ms = 0.0
                t_sync_wait_ms = 0.0
                inflight = False

                for bidx in range(num_batches):
                    x_cpu, pre_ms = futures.pop(0).result()
                    t_pre_ms += pre_ms

                    # 等上一批完成（最多1个inflight，避免显存增长）
                    if inflight:
                        s0 = now()
                        sync()
                        s1 = now()
                        t_sync_wait_ms += (s1 - s0) * 1000.0

                    # submit：H2D + enqueue forward（不在这里 sync）
                    sub0 = now()
                    x = x_cpu.to(device, non_blocking=True)
                    if use_amp:
                        with torch.npu.amp.autocast():
                            _ = model(x)
                    else:
                        _ = model(x)
                    sub1 = now()
                    t_submit_ms += (sub1 - sub0) * 1000.0
                    inflight = True

                    # 补预取
                    if next_k < num_batches:
                        fut = ex.submit(submit_pre_task, batches_slices[next_k])
                        futures.append(fut)
                        next_k += 1

                # 最后一批 sync
                if inflight:
                    s0 = now()
                    sync()
                    s1 = now()
                    t_sync_wait_ms += (s1 - s0) * 1000.0

                img_t1 = now()
                t_total_ms = (img_t1 - img_t0) * 1000.0

                tile_per_s = valid_tiles / (t_total_ms / 1000.0)
                img_per_s = 1.0 / (t_total_ms / 1000.0)
                avg_ms_per_tile_total = t_total_ms / valid_tiles
                pre_ms_per_tile = t_pre_ms / valid_tiles

                w.writerow([
                    os.path.basename(path), W, H,
                    mode, total_tiles_possible, valid_tiles, num_batches,
                    f"{t_decode_ms:.3f}", f"{t_tile_ms:.3f}", f"{t_pre_ms:.3f}", f"{t_submit_ms:.3f}", f"{t_sync_wait_ms:.3f}",
                    f"{t_total_ms:.3f}", f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                    f"{avg_ms_per_tile_total:.6f}", f"{pre_ms_per_tile:.6f}",
                    note
                ])
                f_out.flush()

                if idx == 1 or idx % args.log_every == 0:
                    share_sync = 100.0 * t_sync_wait_ms / t_total_ms
                    print(f"[A2] {idx}/{len(files)} {os.path.basename(path)} mode={mode} "
                          f"tiles={valid_tiles}/{total_tiles_possible} batches={num_batches} "
                          f"total={t_total_ms/1000:.3f}s tile/s={tile_per_s:.1f} "
                          f"sync_wait={t_sync_wait_ms:.1f}ms({share_sync:.0f}%) "
                          f"decode={t_decode_ms:.1f} tilegen={t_tile_ms:.1f} pre={t_pre_ms:.1f} submit={t_submit_ms:.1f} {note}",
                          flush=True)

                global_tiles += valid_tiles
                global_images += 1
                if idx == 1:
                    cold_first_total_ms = t_total_ms
                if idx > args.skip_steady:
                    steady_tiles += valid_tiles
                    steady_time_s += (t_total_ms / 1000.0)

    global_t1 = now()
    total_time_s = global_t1 - global_t0
    overall_tile_s = global_tiles / total_time_s if total_time_s > 0 else 0.0
    overall_img_s = global_images / total_time_s if total_time_s > 0 else 0.0

    steady_tile_s = steady_tiles / steady_time_s if steady_time_s > 0 else 0.0
    steady_img_s = (global_images - args.skip_steady) / steady_time_s if steady_time_s > 0 and global_images > args.skip_steady else 0.0

    print("\n=== A2 FINAL SUMMARY ===")
    print(f"output_tsv: {args.out}")
    print(f"images={global_images}, tiles={global_tiles}, total_time={total_time_s:.3f}s")
    print(f"overall: tile/s={overall_tile_s:.2f}, img/s={overall_img_s:.4f}  (包含warmup/冷启动/所有开销)")
    print(f"steady(skip_first={args.skip_steady}): tile/s={steady_tile_s:.2f}, img/s={steady_img_s:.4f}")
    if cold_first_total_ms is not None:
        print(f"cold_first_image_total_ms={cold_first_total_ms:.2f}")
    print(f"warmup: mode={args.warmup_mode}, submit_ms={warmup_submit_ms:.2f}, sync_ms={warmup_sync_ms:.2f}")
    print("判读：sync_wait占比长期接近0% => CPU仍是瓶颈；sync_wait明显变大 => NPU成为瓶颈，此时再上ATC/INT8最划算。")
    print("========================\n")


if __name__ == "__main__":
    main()