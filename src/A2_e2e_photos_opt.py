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


# -----------------------------
# 0) 计时与同步
# -----------------------------
def now() -> float:
    return time.perf_counter()

def sync():
    # NPU 异步执行，不 sync 会导致计时“虚快”
    torch_npu.npu.synchronize()


# -----------------------------
# 1) 复用你 test_uni_npu.py 的 UNI 加载方式
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
# 2) A2: 前景过滤 + 生成 tile 坐标
# -----------------------------
def is_tissue_from_np(patch_rgb_uint8: np.ndarray, white_ratio_thr: float = 0.8) -> bool:
    """
    极简白底过滤：
      - 把灰度 > 240 的像素当作白色
      - 白色比例过高则视为背景
    """
    gray = patch_rgb_uint8.mean(axis=2)  # 0..255
    white_ratio = (gray > 240).mean()
    return white_ratio < white_ratio_thr

def gen_coords(img_np: np.ndarray, tile: int, stride: int, white_ratio_thr: float) -> Tuple[List[Tuple[int, int]], int]:
    """
    返回：
      coords_valid: 通过过滤的 (x,y) 坐标列表
      total_tiles_possible: 不过滤时理论 tile 总数（用于统计有效率）
    """
    H, W, _ = img_np.shape
    ny = max(0, (H - tile) // stride + 1)
    nx = max(0, (W - tile) // stride + 1)
    total_tiles_possible = ny * nx

    coords = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            patch = img_np[y:y+tile, x:x+tile]
            if is_tissue_from_np(patch, white_ratio_thr):
                coords.append((x, y))

    return coords, total_tiles_possible


# -----------------------------
# 3) C1：向量化预处理（按 batch 处理，不一次性全 stack 500 tiles）
# -----------------------------
def preprocess_batch_vectorized(
    img_np: np.ndarray,
    coords: List[Tuple[int, int]],
    tile: int,
    mean: np.ndarray,
    std: np.ndarray,
    out_dtype: np.dtype = np.float32,
    pin_memory: bool = False,
) -> Tuple[torch.Tensor, float]:
    """
    输入：
      - img_np: (H,W,3) uint8
      - coords: batch 内的 (x,y) 列表
    输出：
      - x_cpu: (B,3,tile,tile) CPU tensor
      - t_pre_ms: 本 batch 预处理耗时（毫秒）

    向量化要点：
      - 先把 B 个 tile 拼成 (B,tile,tile,3)（uint8）
      - 一次性 /255、一次性 normalize（广播）
      - 一次性转成 torch 并 permute 成 (B,3,H,W)
    """
    t0 = now()

    B = len(coords)
    # 预分配 uint8 buffer（避免 Python list + np.stack 的额外拷贝）
    buf = np.empty((B, tile, tile, 3), dtype=np.uint8)
    for i, (x, y) in enumerate(coords):
        buf[i] = img_np[y:y+tile, x:x+tile]

    arr = buf.astype(np.float32) / 255.0
    arr = (arr - mean) / std  # 广播：mean/std shape=(1,1,1,3)

    # 转 torch，变为 (B,3,H,W)
    x_cpu = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

    # 可选：pin_memory（有时能加速 H2D，但也会占用更多内存）
    if pin_memory:
        try:
            x_cpu = x_cpu.pin_memory()
        except Exception:
            pass

    # 输出 dtype（通常保持 float32；如果你只做速度线，也可尝试 float16）
    if out_dtype == np.float16:
        x_cpu = x_cpu.half()

    t1 = now()
    return x_cpu, (t1 - t0) * 1000.0


# -----------------------------
# 4) C2：CPU 预取并行（线程池） + NPU 单卡流水（1个 in-flight）
# -----------------------------
def chunk_list(coords: List[Tuple[int, int]], bs: int) -> List[List[Tuple[int, int]]]:
    return [coords[i:i+bs] for i in range(0, len(coords), bs)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_list", required=True, help="txt，每行一张图片路径")
    ap.add_argument("--ckpt_dir", default="./assets/ckpts/UNI")
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32, help="A2 的 batch（建议扫 16/32/54/64）")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--white_thr", type=float, default=1.01, help="白底过滤阈值，越小越严格，>1.0 则不过滤")
    ap.add_argument("--cpu_workers", type=int, default=8, help="CPU 线程数（你24核ARM建议 8~16）")
    ap.add_argument("--prefetch", type=int, default=3, help="预取窗口（同时在CPU准备的batch数）")
    ap.add_argument("--pin_memory", action="store_true", help="CPU tensor pin_memory（可选试验）")
    ap.add_argument("--cpu_out_fp16", action="store_true", help="CPU预处理输出fp16（速度线可试）")
    ap.add_argument("--warmup_iters", type=int, default=3, help="程序开头 warmup 次数")
    ap.add_argument("--warmup_async", action="store_true",
                    help="异步 warmup：先enqueue warmup，不立刻sync，让它被CPU decode/pre覆盖")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--skip_steady", type=int, default=1, help="稳态统计跳过前N张（默认跳过第1张）")
    ap.add_argument("--out", required=True, help="输出TSV路径")
    args = ap.parse_args()

    assert args.tile % 16 == 0, "tile 必须能被 16 整除（ViT patch_size=16）"

    # 设备
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    # 读取文件列表
    with open(args.file_list, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    if args.max_images > 0:
        files = files[:args.max_images]
    assert len(files) > 0, "file_list 为空"

    # 模型
    model = build_uni_model(args.ckpt_dir, device=device, img_size=args.tile)
    use_amp = (args.precision == "fp16")

    # 预处理常量（numpy广播用）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    out_dtype = np.float16 if args.cpu_out_fp16 else np.float32

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # -----------------------------
    # Warmup：程序开头把 NPU 跑起来
    # -----------------------------
    warmup_submit_ms = 0.0
    warmup_sync_ms = 0.0
    warmup_pending = False

    if args.warmup_iters > 0:
        x_warm = torch.randn(args.bs, 3, args.tile, args.tile, device=device)
        t_w0 = now()
        for _ in range(args.warmup_iters):
            if use_amp:
                with torch.npu.amp.autocast():
                    _ = model(x_warm)
            else:
                _ = model(x_warm)
        t_w1 = now()
        warmup_submit_ms = (t_w1 - t_w0) * 1000.0

        if args.warmup_async:
            # 不立即 sync，让 warmup 在后台跑，尽量被 CPU decode/pre 覆盖
            warmup_pending = True
        else:
            t_s0 = now()
            sync()
            t_s1 = now()
            warmup_sync_ms = (t_s1 - t_s0) * 1000.0
            warmup_pending = False

    print("\n=== A2 OPT START ===")
    print(f"device={device}, precision={args.precision}, tile={args.tile}, stride={args.stride}, bs={args.bs}")
    print(f"cpu_workers={args.cpu_workers}, prefetch={args.prefetch}, pin_memory={args.pin_memory}, cpu_out_fp16={args.cpu_out_fp16}")
    print(f"warmup: iters={args.warmup_iters}, async={args.warmup_async}, submit_ms={warmup_submit_ms:.2f}, sync_ms={warmup_sync_ms:.2f}\n")

    # TSV 表头（更详尽）
    header = [
        "file", "W", "H",
        "total_tiles_possible", "valid_tiles", "num_batches",
        "t_decode_ms", "t_tile_ms",
        "t_pre_ms", "t_submit_ms", "t_sync_wait_ms",
        "t_total_ms", "tile_per_s", "img_per_s",
        "avg_ms_per_tile_total",
        "pre_ms_per_tile",
        "note"
    ]

    # 全局统计
    global_t0 = now()
    global_tiles = 0
    global_images = 0

    steady_tiles = 0
    steady_time_s = 0.0

    cold_first_total_ms: Optional[float] = None

    with open(args.out, "w", newline="") as f_out:
        w = csv.writer(f_out, delimiter="\t")
        w.writerow(header)

        # 线程池：用于预处理 batch（C2）
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as ex:
            for idx, path in enumerate(files, 1):
                note = ""
                img_t0 = now()

                # ---------- decode ----------
                d0 = now()
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)  # uint8
                d1 = now()
                H, W, _ = img_np.shape
                t_decode_ms = (d1 - d0) * 1000.0

                # ---------- tile coords ----------
                g0 = now()
                coords, total_tiles_possible = gen_coords(
                    img_np, tile=args.tile, stride=args.stride, white_ratio_thr=args.white_thr
                )
                g1 = now()
                t_tile_ms = (g1 - g0) * 1000.0
                valid_tiles = len(coords)

                # 若 warmup_pending：在第一张图第一次要用 NPU 前 sync（把 warmup 完成）
                if warmup_pending:
                    s0 = now()
                    sync()
                    s1 = now()
                    warmup_sync_ms = (s1 - s0) * 1000.0
                    warmup_pending = False
                    note = f"warmup_sync_ms={warmup_sync_ms:.2f}"

                if valid_tiles == 0:
                    # 全背景，仍落盘
                    img_t1 = now()
                    t_total_ms = (img_t1 - img_t0) * 1000.0
                    w.writerow([
                        os.path.basename(path), W, H,
                        total_tiles_possible, 0, 0,
                        f"{t_decode_ms:.3f}", f"{t_tile_ms:.3f}",
                        f"{0.0:.3f}", f"{0.0:.3f}", f"{0.0:.3f}",
                        f"{t_total_ms:.3f}", f"{0.0:.3f}", f"{(1.0/(t_total_ms/1000.0)):.6f}",
                        f"{0.0:.6f}",
                        f"{0.0:.6f}",
                        "all background"
                    ])
                    f_out.flush()

                    global_images += 1
                    if idx == 1:
                        cold_first_total_ms = t_total_ms
                    continue

                # 切分 batch
                batches = chunk_list(coords, args.bs)
                num_batches = len(batches)

                # 预取：先提交前 prefetch 个 batch 的预处理任务
                # future 返回 (x_cpu, pre_ms)
                futures = []
                next_k = 0
                while next_k < num_batches and len(futures) < args.prefetch:
                    sub = batches[next_k]
                    fut = ex.submit(preprocess_batch_vectorized, img_np, sub, args.tile, mean, std, out_dtype, args.pin_memory)
                    futures.append(fut)
                    next_k += 1

                t_pre_ms = 0.0
                t_submit_ms = 0.0
                t_sync_wait_ms = 0.0
                inflight = False

                # 逐 batch：取一个预处理结果 ->（若有inflight先sync）-> submit到NPU -> 再补充预取
                for bidx in range(num_batches):
                    # 取最早的 future（FIFO）
                    x_cpu, pre_ms = futures.pop(0).result()
                    t_pre_ms += pre_ms

                    # 在提交当前 batch 前，先等待上一批 NPU 完成（保证最多1个inflight，避免显存增长）
                    if inflight:
                        s0 = now()
                        sync()
                        s1 = now()
                        t_sync_wait_ms += (s1 - s0) * 1000.0

                    # submit：H2D + model（不在这里 sync）
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

                    # 继续补充预取窗口
                    if next_k < num_batches:
                        sub_coords = batches[next_k]
                        fut = ex.submit(preprocess_batch_vectorized, img_np, sub_coords, args.tile, mean, std, out_dtype, args.pin_memory)
                        futures.append(fut)
                        next_k += 1

                # 最后一批提交后，需要 sync 确保本图计算完成
                if inflight:
                    s0 = now()
                    sync()
                    s1 = now()
                    t_sync_wait_ms += (s1 - s0) * 1000.0
                    inflight = False

                img_t1 = now()
                t_total_ms = (img_t1 - img_t0) * 1000.0

                tile_per_s = valid_tiles / (t_total_ms / 1000.0)
                img_per_s = 1.0 / (t_total_ms / 1000.0)
                avg_ms_per_tile_total = t_total_ms / valid_tiles
                pre_ms_per_tile = t_pre_ms / valid_tiles

                # 写 TSV
                w.writerow([
                    os.path.basename(path), W, H,
                    total_tiles_possible, valid_tiles, num_batches,
                    f"{t_decode_ms:.3f}", f"{t_tile_ms:.3f}",
                    f"{t_pre_ms:.3f}", f"{t_submit_ms:.3f}", f"{t_sync_wait_ms:.3f}",
                    f"{t_total_ms:.3f}", f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                    f"{avg_ms_per_tile_total:.6f}",
                    f"{pre_ms_per_tile:.6f}",
                    note
                ])
                f_out.flush()

                # 进度输出（更详尽）
                if idx == 1 or idx % args.log_every == 0:
                    # 简单占比（不严格，因为有重叠，但够诊断）
                    share_pre = 100.0 * t_pre_ms / t_total_ms
                    share_sync = 100.0 * t_sync_wait_ms / t_total_ms
                    print(f"[A2] {idx}/{len(files)} {os.path.basename(path)} "
                          f"tiles={valid_tiles}/{total_tiles_possible} batches={num_batches} "
                          f"total={t_total_ms/1000:.3f}s tile/s={tile_per_s:.1f} "
                          f"pre={t_pre_ms:.1f}ms({share_pre:.0f}%) sync_wait={t_sync_wait_ms:.1f}ms({share_sync:.0f}%) "
                          f"decode={t_decode_ms:.1f}ms tilegen={t_tile_ms:.1f}ms submit={t_submit_ms:.1f}ms {note}",
                          flush=True)

                # 全局统计
                global_tiles += valid_tiles
                global_images += 1
                if idx == 1:
                    cold_first_total_ms = t_total_ms

                # 稳态统计（跳过前 N 张）
                if idx > args.skip_steady:
                    steady_tiles += valid_tiles
                    steady_time_s += (t_total_ms / 1000.0)

    global_t1 = now()
    total_time_s = global_t1 - global_t0
    overall_tile_s = global_tiles / total_time_s if total_time_s > 0 else 0.0
    overall_img_s = global_images / total_time_s if total_time_s > 0 else 0.0

    steady_tile_s = steady_tiles / steady_time_s if steady_time_s > 0 else 0.0
    steady_img_s = (global_images - args.skip_steady) / steady_time_s if steady_time_s > 0 and global_images > args.skip_steady else 0.0

    print("\n=== A2 OPT SUMMARY ===")
    print(f"output_tsv: {args.out}")
    print(f"images={global_images}, tiles={global_tiles}, total_time={total_time_s:.3f}s")
    print(f"overall: tile/s={overall_tile_s:.2f}, img/s={overall_img_s:.4f}  (包含所有开销)")
    print(f"steady(skip_first={args.skip_steady}): tile/s={steady_tile_s:.2f}, img/s={steady_img_s:.4f}")
    if cold_first_total_ms is not None:
        print(f"cold_first_image_total_ms={cold_first_total_ms:.2f}")
    print(f"warmup: submit_ms={warmup_submit_ms:.2f}, sync_ms={warmup_sync_ms:.2f}, async={args.warmup_async}")
    print("提示：如果 sync_wait 占比接近 0%，说明仍是 CPU 瓶颈；sync_wait 变大才说明 NPU 成为瓶颈。")
    print("======================\n")


if __name__ == "__main__":
    main()