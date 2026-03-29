# src/A2_e2e_photos.py
import os
import time
import csv
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch_npu
import timm
from torchvision import transforms


# ==============================
# 1) 工具函数：计时 & 同步
# ==============================
def now() -> float:
    return time.perf_counter()


def sync():
    """
    NPU/加速器通常是异步执行：
    - 你调用 model(x) 后，Python 可能立刻返回，但设备上还在算
    - 如果不 synchronize()，计时会“虚快”
    所以我们在 H2D 后、infer 后都 sync，保证计时准确
    """
    torch_npu.npu.synchronize()


# ==============================
# 2) 模型加载：完全复用 test_uni_npu.py 的做法
# ==============================
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


# ==============================
# 3) 前景过滤：极简“白底过滤”
# ==============================
def is_tissue_from_np(patch_rgb_uint8: np.ndarray, white_ratio_thr: float = 0.8) -> bool:
    """
    patch_rgb_uint8: (H, W, 3) uint8
    逻辑：如果“很白”的像素比例太高 -> 判为背景 -> 丢弃
    white_ratio_thr 越小：越严格（保留更少 tile）
    """
    gray = patch_rgb_uint8.mean(axis=2)  # 0..255
    white_ratio = (gray > 240).mean()    # 240~255 视为白
    return white_ratio < white_ratio_thr


def gen_coords(img_np: np.ndarray, tile: int, stride: int, white_ratio_thr: float) -> Tuple[List[Tuple[int, int]], int]:
    """
    在整张图上滑窗生成 tile 坐标 (x, y)，并做前景过滤。
    返回：
      - coords_valid: 通过前景过滤的坐标列表
      - total_tiles: 不做过滤时理论 tile 总数（用于统计有效率）
    """
    H, W, _ = img_np.shape
    ny = max(0, (H - tile) // stride + 1)
    nx = max(0, (W - tile) // stride + 1)
    total_tiles = ny * nx

    coords_valid = []
    for y in range(0, H - tile + 1, stride):
        for x in range(0, W - tile + 1, stride):
            patch = img_np[y:y + tile, x:x + tile]
            if is_tissue_from_np(patch, white_ratio_thr=white_ratio_thr):
                coords_valid.append((x, y))

    return coords_valid, total_tiles


# ==============================
# 4) 预处理：tile 已是 224，不做 Resize/Crop（只做 ToTensor+Normalize）
# ==============================
def make_tile_transform():
    """
    与 test_uni_npu.py 的 Normalize(mean/std) 一致。
    A2 里 tile 已经裁成 224x224，所以不再做 Resize(256)+CenterCrop(224)，避免额外 CPU 开销干扰端到端瓶颈判断。
    """
    return transforms.Compose([
        transforms.ToTensor(),  # [0,1], CHW
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def preprocess_batch(img_pil: Image.Image, coords: List[Tuple[int, int]], tile: int, tfm) -> torch.Tensor:
    """
    输入：PIL 大图 + 坐标列表
    输出：CPU Tensor (B, 3, tile, tile) float32
    """
    batch = []
    for (x, y) in coords:
        patch = img_pil.crop((x, y, x + tile, y + tile))  # PIL patch
        batch.append(tfm(patch))                           # CPU tensor (3,tile,tile)
    return torch.stack(batch, dim=0)                       # (B,3,tile,tile)


# ==============================
# 5) 主流程：A2 端到端 benchmark（逐图、逐段计时、落盘 TSV）
# ==============================
@torch.inference_mode()
def run_a2(args):
    # 设备设置
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    # ViT-L/16 的关键约束：输入高宽必须能被 16 整除
    assert args.tile % 16 == 0, "tile 必须能被 16 整除（ViT patch_size=16）"

    # 读文件列表
    with open(args.file_list, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    if args.max_images > 0:
        files = files[:args.max_images]
    assert len(files) > 0, f"empty file list: {args.file_list}"

    # 创建输出目录
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.err_log:
        os.makedirs(os.path.dirname(args.err_log), exist_ok=True)

    # 加载模型（复用 test_uni_npu.py）
    model = build_uni_model(args.ckpt_dir, device=device, img_size=args.tile)
    tfm = make_tile_transform()
    use_amp = (args.precision == "fp16")

    # 写 TSV 表头
    header = [
        "file", "W", "H",
        "total_tiles_possible", "valid_tiles",
        "t_decode_ms", "t_tile_ms", "t_pre_ms", "t_h2d_ms", "t_infer_ms", "t_total_ms",
        "tile_per_s", "img_per_s",
        "avg_ms_per_tile_total"
    ]

    # 统计汇总（便于你最后看平均占比）
    sum_decode = sum_tile = sum_pre = sum_h2d = sum_inf = sum_total = 0.0
    sum_valid_tiles = 0
    ok_images = 0
    bad_images = 0

    with open(args.out, "w", newline="") as f_out:
        w = csv.writer(f_out, delimiter="\t")
        w.writerow(header)

        err_f = open(args.err_log, "w") if args.err_log else None

        for i, path in enumerate(files, 1):
            try:
                t0 = now()

                # -------- 1) decode：读图 ----------
                a0 = now()
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)  # uint8，用于快速前景过滤
                a1 = now()

                H, W, _ = img_np.shape

                # -------- 2) tile：生成 coords + 前景过滤 ----------
                b0 = now()
                coords, total_tiles = gen_coords(
                    img_np, tile=args.tile, stride=args.stride, white_ratio_thr=args.white_thr
                )
                b1 = now()

                # -------- 3) batch loop：preprocess -> H2D -> infer ----------
                t_pre = t_h2d = t_inf = 0.0
                n_tiles = 0

                # 若一张图全是背景，valid_tiles=0：仍记录一行（便于你看到过滤过严/数据问题）
                for j in range(0, len(coords), args.bs):
                    sub = coords[j:j + args.bs]
                    if not sub:
                        continue

                    # 3.1 preprocess（CPU）：crop tile + ToTensor + Normalize + stack
                    c0 = now()
                    x_cpu = preprocess_batch(img, sub, tile=args.tile, tfm=tfm)  # CPU float32
                    c1 = now()

                    # 3.2 H2D：CPU -> NPU（计时要 sync）
                    d0 = now()
                    x = x_cpu.to(device, non_blocking=True)
                    sync()
                    d1 = now()

                    # 3.3 infer：NPU forward（计时要 sync）
                    e0 = now()
                    if use_amp:
                        with torch.npu.amp.autocast():
                            _ = model(x)
                    else:
                        _ = model(x)
                    sync()
                    e1 = now()

                    n_tiles += x_cpu.shape[0]
                    t_pre += (c1 - c0) * 1000
                    t_h2d += (d1 - d0) * 1000
                    t_inf += (e1 - e0) * 1000

                    # 可选：更细粒度日志（新手调试用）
                    if args.verbose_batch:
                        print(f"    [batch] tiles={x_cpu.shape[0]} pre={((c1-c0)*1000):.2f}ms "
                              f"h2d={((d1-d0)*1000):.2f}ms infer={((e1-e0)*1000):.2f}ms")

                t1 = now()

                # -------- 汇总本图计时 ----------
                t_decode = (a1 - a0) * 1000
                t_tile   = (b1 - b0) * 1000
                t_total  = (t1 - t0) * 1000

                tile_per_s = (n_tiles / (t_total / 1000.0)) if t_total > 0 else 0.0
                img_per_s  = (1.0 / (t_total / 1000.0)) if t_total > 0 else 0.0
                avg_ms_per_tile_total = (t_total / n_tiles) if n_tiles > 0 else 0.0

                # 写 TSV（逐图一行）
                w.writerow([
                    os.path.basename(path), W, H,
                    total_tiles, n_tiles,
                    f"{t_decode:.3f}", f"{t_tile:.3f}", f"{t_pre:.3f}", f"{t_h2d:.3f}", f"{t_inf:.3f}", f"{t_total:.3f}",
                    f"{tile_per_s:.3f}", f"{img_per_s:.6f}",
                    f"{avg_ms_per_tile_total:.6f}"
                ])
                f_out.flush()

                # 汇总统计
                sum_decode += t_decode
                sum_tile += t_tile
                sum_pre += t_pre
                sum_h2d += t_h2d
                sum_inf += t_inf
                sum_total += t_total
                sum_valid_tiles += n_tiles
                ok_images += 1

                if i == 1 or i % args.log_every == 0:
                    print(f"[A2] {i}/{len(files)} file={os.path.basename(path)} "
                          f"valid_tiles={n_tiles}/{total_tiles} total={t_total/1000:.3f}s "
                          f"tile/s={tile_per_s:.1f} out={args.out}",
                          flush=True)

            except Exception as e:
                bad_images += 1
                if err_f:
                    err_f.write(f"{path}\t{repr(e)}\n")
                    err_f.flush()
                else:
                    print(f"[A2][ERROR] {path} -> {e}")

        if err_f:
            err_f.close()

    # 最后输出一个“平均占比”总结（让你一眼看瓶颈）
    if ok_images > 0:
        def pct(x): return 100.0 * x / sum_total if sum_total > 0 else 0.0
        print("\n=== A2 Summary (avg over images) ===")
        print(f"images_ok={ok_images}, images_bad={bad_images}")
        print(f"avg_total_ms={sum_total/ok_images:.2f} ms")
        print(f"avg_valid_tiles={sum_valid_tiles/ok_images:.2f} tiles/image")
        print(f"avg tile/s (global)={(sum_valid_tiles/(sum_total/1000.0)) if sum_total>0 else 0.0:.2f}")
        print("time share (approx):")
        print(f"  decode: {pct(sum_decode):.1f}%")
        print(f"  tile  : {pct(sum_tile):.1f}%")
        print(f"  pre   : {pct(sum_pre):.1f}%")
        print(f"  h2d   : {pct(sum_h2d):.1f}%")
        print(f"  infer : {pct(sum_inf):.1f}%")
        print("===================================\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file_list", required=True, help="txt，每行一个图片路径（不移动 raw 数据）")
    ap.add_argument("--ckpt_dir", default="./assets/ckpts/UNI", help="包含 pytorch_model.bin 的目录")
    ap.add_argument("--tile", type=int, default=224, help="tile大小（必须能被16整除）")
    ap.add_argument("--stride", type=int, default=224, help="切块步长")
    ap.add_argument("--bs", type=int, default=96, help="batch size（端到端可能与A1不同）")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--white_thr", type=float, default=1.01, help="白底过滤阈值，越小越严格，>1.0 则不过滤")
    ap.add_argument("--max_images", type=int, default=0, help=">0则只跑前N张（冒烟测试）")
    ap.add_argument("--log_every", type=int, default=10, help="每N张打印一次进度")
    ap.add_argument("--verbose_batch", action="store_true", help="打印每个batch的耗时（新手调试用）")
    ap.add_argument("--out", required=True, help="输出TSV路径")
    ap.add_argument("--err_log", default="logs/A2_errors.txt", help="坏图/异常记录（可设为空禁用）")
    args = ap.parse_args()
    if args.err_log == "":
        args.err_log = None
    return args


if __name__ == "__main__":
    args = parse_args()
    run_a2(args)