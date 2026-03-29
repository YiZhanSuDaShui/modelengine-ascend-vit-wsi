#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_a2_grid_sweep.py
====================
只 sweep CPU 参数（cpu_workers + 对应 prefetch），并固定 bs=96。

你现在的目的非常明确：
- bs 你已经确定为 96（A1 sweet spot）
- 当前阶段只想找到 A2 端到端里：CPU pipeline 的最优点（workers/prefetch）
- 下一步再改输入逻辑做 global-mix batching（跨图凑满 bs=96）

因此本脚本去掉 bs 轮询：
- cpu_workers: 4, 8, 12, 16, 20
- prefetch_map: 4->2, 8->3, 12->4, 16->4, 20->4
- bs 固定 96（可通过 --bs 修改，但默认就是 96）

输出：
- logs/sweep_xxx/A2_sweep_summary_时间戳.tsv
- 每个配置单独的 per-image TSV + summary JSON（便于回溯）
"""

import os
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any


def main() -> None:
    ap = argparse.ArgumentParser()

    # 基本输入/输出
    ap.add_argument("--a2_script", type=str, default="src/A2_final_opt_v2.py")
    ap.add_argument("--file_list", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")
    ap.add_argument("--out_dir", type=str, default="logs/sweep")

    # 固定口径（你当前的控制变量）
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--white_thr", type=float, default=1.01)

    # ✅ 固定 bs=96（不再轮询）
    ap.add_argument("--bs", type=int, default=96)

    # warmup / steady
    ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], default="async")
    ap.add_argument("--warmup_iters", type=int, default=3)
    ap.add_argument("--skip_steady", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)  # 想快速试可以先设 20/30

    # sweep 维度：只扫 workers（prefetch 用 map 自动配）
    ap.add_argument("--worker_list", type=str, default="4,8,12,16,20")
    ap.add_argument("--prefetch_map", type=str, default="4:2,8:3,12:4,16:4,20:4")

    # 运行控制
    ap.add_argument("--decode_prefetch", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")

    args = ap.parse_args()

    # 解析 workers 列表
    worker_list = [int(x) for x in args.worker_list.split(",") if x.strip()]

    # 解析 prefetch_map
    prefetch_map: Dict[int, int] = {}
    for item in args.prefetch_map.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        prefetch_map[int(k)] = int(v)

    os.makedirs(args.out_dir, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_tsv = os.path.join(args.out_dir, f"A2_sweep_summary_{run_tag}.tsv")

    header = [
        "cpu_workers", "prefetch", "bs",
        "decode_workers", "pre_workers",
        "tiles_total", "num_images",
        "total_time_s",
        "overall_tile_s", "steady_tile_s",
        "overall_img_s", "steady_img_s",
        "cold_first_ms",
        "mean_decode_compute_ms", "mean_tile_compute_ms", "mean_decode_wait_ms",
        "mean_pre_ms", "mean_submit_ms", "mean_sync_wait_ms", "mean_total_ms",
        "out_tsv", "summary_json"
    ]

    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")

    print(f"[SWEEP] will write summary -> {summary_tsv}")
    print(f"[SWEEP] fixed bs={args.bs} (no bs loop)")

    all_rows: List[Dict[str, Any]] = []
    total_runs = len(worker_list)
    run_idx = 0

    for w in worker_list:
        run_idx += 1

        # 取该 workers 对应的 prefetch（若没写就给个默认 3）
        pf = prefetch_map.get(w, 3)

        run_name = f"w{w}_pf{pf}_bs{args.bs}_tile{args.tile}_s{args.stride}_{args.precision}"
        out_tsv = os.path.join(args.out_dir, f"{run_name}.tsv")
        out_json = os.path.join(args.out_dir, f"{run_name}.json")

        # 组装命令：调用 A2 脚本跑一遍
        cmd = [
            "python", args.a2_script,
            "--file_list", args.file_list,
            "--ckpt_dir", args.ckpt_dir,
            "--tile", str(args.tile),
            "--stride", str(args.stride),
            "--bs", str(args.bs),
            "--white_thr", str(args.white_thr),
            "--precision", args.precision,
            "--cpu_workers", str(w),
            "--prefetch", str(pf),
            "--decode_prefetch", str(args.decode_prefetch),
            "--warmup_mode", args.warmup_mode,
            "--warmup_iters", str(args.warmup_iters),
            "--skip_steady", str(args.skip_steady),
            "--max_images", str(args.max_images),
            "--log_every", str(args.log_every),
            "--out", out_tsv,
            "--summary_json", out_json,
        ]

        print(f"\n[SWEEP] [{run_idx}/{total_runs}] {run_name}")
        print("[SWEEP] cmd:", " ".join(cmd))

        if args.dry_run:
            continue

        ret = subprocess.run(cmd)

        if ret.returncode != 0:
            print(f"[SWEEP][ERROR] run failed (returncode={ret.returncode}): {run_name}")
            if args.stop_on_error:
                raise RuntimeError(f"sweep stopped due to error: {run_name}")
            else:
                continue

        if not os.path.exists(out_json):
            print(f"[SWEEP][WARN] missing summary_json: {out_json}")
            continue

        # 读 summary json
        with open(out_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        cfg = summary["config"]
        res = summary["result"]

        row = {
            "cpu_workers": cfg["cpu_workers"],
            "prefetch": cfg["prefetch"],
            "bs": cfg["bs"],
            "decode_workers": cfg["decode_workers"],
            "pre_workers": cfg["pre_workers"],
            "tiles_total": res["tiles_total"],
            "num_images": res["num_images"],
            "total_time_s": res["total_time_s"],
            "overall_tile_s": res["overall_tile_s"],
            "steady_tile_s": res["steady_tile_s"],
            "overall_img_s": res["overall_img_s"],
            "steady_img_s": res["steady_img_s"],
            "cold_first_ms": res["cold_first_ms"],
            "mean_decode_compute_ms": res["mean_decode_compute_ms"],
            "mean_tile_compute_ms": res["mean_tile_compute_ms"],
            "mean_decode_wait_ms": res["mean_decode_wait_ms"],
            "mean_pre_ms": res["mean_pre_ms"],
            "mean_submit_ms": res["mean_submit_ms"],
            "mean_sync_wait_ms": res["mean_sync_wait_ms"],
            "mean_total_ms": res["mean_total_ms"],
            "out_tsv": cfg["out_tsv"],
            "summary_json": out_json,
        }

        all_rows.append(row)

        # 追加写入总表
        with open(summary_tsv, "a", encoding="utf-8") as f:
            f.write("\t".join([
                str(row["cpu_workers"]),
                str(row["prefetch"]),
                str(row["bs"]),
                str(row["decode_workers"]),
                str(row["pre_workers"]),
                str(row["tiles_total"]),
                str(row["num_images"]),
                f'{row["total_time_s"]:.6f}',
                f'{row["overall_tile_s"]:.6f}',
                f'{row["steady_tile_s"]:.6f}',
                f'{row["overall_img_s"]:.6f}',
                f'{row["steady_img_s"]:.6f}',
                f'{row["cold_first_ms"]:.3f}',
                f'{row["mean_decode_compute_ms"]:.3f}',
                f'{row["mean_tile_compute_ms"]:.3f}',
                f'{row["mean_decode_wait_ms"]:.3f}',
                f'{row["mean_pre_ms"]:.3f}',
                f'{row["mean_submit_ms"]:.3f}',
                f'{row["mean_sync_wait_ms"]:.3f}',
                f'{row["mean_total_ms"]:.3f}',
                row["out_tsv"],
                row["summary_json"],
            ]) + "\n")

        print(f"[SWEEP] done: steady_tile/s={row['steady_tile_s']:.2f}, overall_tile/s={row['overall_tile_s']:.2f}")

    if not all_rows:
        print("[SWEEP] no successful runs.")
        return

    # 按 steady_tile/s 排名
    all_rows_sorted = sorted(all_rows, key=lambda r: r["steady_tile_s"], reverse=True)

    print("\n================ [SWEEP RANK by steady_tile/s] ================")
    for i, r in enumerate(all_rows_sorted, 1):
        print(f"[{i:02d}] steady_tile/s={r['steady_tile_s']:.2f} "
              f"(overall={r['overall_tile_s']:.2f}) "
              f"w={r['cpu_workers']} pf={r['prefetch']} bs={r['bs']} "
              f"decode_w={r['decode_workers']} pre_w={r['pre_workers']} "
              f"mean_decode_wait_ms={r['mean_decode_wait_ms']:.1f} "
              f"mean_sync_wait_ms={r['mean_sync_wait_ms']:.1f} "
              f"out={os.path.basename(r['out_tsv'])}")
    print("==============================================================\n")
    print(f"[SWEEP] summary saved at: {summary_tsv}")


if __name__ == "__main__":
    main()