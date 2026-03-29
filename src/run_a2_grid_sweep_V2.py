#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_a2_grid_sweep.py
====================
只 sweep CPU 参数（cpu_workers + 对应 prefetch），并固定 bs=96。
（同时支持 global-mix batching）

新增：
### 多次运行取平均（快速修复）
- 在 sweep 脚本中添加 --repeat 参数
- 对每个配置重复运行 N 次（默认 3 次）
- 计算 steady_tile/s 的均值与标准差（并同时给 overall_tile/s 的均值/标准差，方便参考）
- 排名按 steady_tile/s_mean 降序

输出：
- logs/sweep_xxx/A2_sweep_summary_时间戳.tsv
- 每个配置每次运行单独的 per-image TSV + summary JSON（便于回溯）
"""

import os
import json
import argparse
import subprocess
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def main() -> None:
    ap = argparse.ArgumentParser()

    # 基本输入/输出
    ap.add_argument("--a2_script", type=str, default="src/A2_final_opt_v2.py")
    ap.add_argument("--file_list", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")
    ap.add_argument("--out_dir", type=str, default="logs/sweep")

    # 固定口径（控制变量）
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--white_thr", type=float, default=1.01)

    # ✅ 固定 bs=96（默认）
    ap.add_argument("--bs", type=int, default=96)

    # ✅ global-mix 开关（默认开启）
    ap.add_argument("--global_mix", type=int, default=1, choices=[0, 1])

    # warmup / steady
    ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], default="async")
    ap.add_argument("--warmup_iters", type=int, default=3)
    ap.add_argument("--skip_steady", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)

    # sweep 维度：只扫 workers（prefetch 用 map 自动配）
    ap.add_argument("--worker_list", type=str, default="4,8,12,16,20")
    ap.add_argument("--prefetch_map", type=str, default="4:2,8:3,12:4,16:4,20:4")

    # 运行控制
    ap.add_argument("--decode_prefetch", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3,
                    help="每个配置重复运行 N 次，输出 steady_tile/s 的均值与标准差（默认 3）")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")

    args = ap.parse_args()

    worker_list = [int(x) for x in args.worker_list.split(",") if x.strip()]

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
        "cpu_workers", "prefetch", "bs", "global_mix",
        "decode_workers", "pre_workers",
        "repeat", "success_runs",
        "tiles_total", "num_images",
        "total_time_s_mean", "total_time_s_std",
        "overall_tile_s_mean", "overall_tile_s_std",
        "steady_tile_s_mean", "steady_tile_s_std",
        "overall_img_s_mean", "overall_img_s_std",
        "steady_img_s_mean", "steady_img_s_std",
        "mean_decode_compute_ms_mean",
        "mean_tile_compute_ms_mean",
        "mean_decode_wait_ms_mean",
        "mean_pre_ms_mean",
        "mean_submit_ms_mean",
        "mean_sync_wait_ms_mean",
        "mean_total_ms_mean",
        "best_out_tsv", "best_summary_json"
    ]

    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")

    print(f"[SWEEP] will write summary -> {summary_tsv}")
    print(f"[SWEEP] fixed bs={args.bs} (no bs loop), global_mix={args.global_mix}")
    print(f"[SWEEP] repeat per config = {args.repeat}")

    all_rows: List[Dict[str, Any]] = []
    total_runs = len(worker_list)
    run_idx = 0

    for w in worker_list:
        run_idx += 1
        pf = prefetch_map.get(w, 3)

        base_name = f"w{w}_pf{pf}_bs{args.bs}_gm{args.global_mix}_tile{args.tile}_s{args.stride}_{args.precision}"
        print(f"\n[SWEEP] [{run_idx}/{total_runs}] {base_name}")

        if args.dry_run:
            continue

        run_results: List[Dict[str, Any]] = []
        best_run: Optional[Dict[str, Any]] = None

        for r in range(max(1, args.repeat)):
            run_name = f"{base_name}_r{r+1}"
            out_tsv = os.path.join(args.out_dir, f"{run_name}.tsv")
            out_json = os.path.join(args.out_dir, f"{run_name}.json")

            cmd = [
                "python", args.a2_script,
                "--file_list", args.file_list,
                "--ckpt_dir", args.ckpt_dir,
                "--tile", str(args.tile),
                "--stride", str(args.stride),
                "--bs", str(args.bs),
                "--global_mix", str(args.global_mix),
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

            print(f"[SWEEP]   ({r+1}/{args.repeat}) cmd:", " ".join(cmd))
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

            with open(out_json, "r", encoding="utf-8") as f:
                summary = json.load(f)

            cfg = summary["config"]
            res = summary["result"]

            one = {
                "cpu_workers": cfg["cpu_workers"],
                "prefetch": cfg["prefetch"],
                "bs": cfg["bs"],
                "global_mix": cfg.get("global_mix", args.global_mix),
                "decode_workers": cfg["decode_workers"],
                "pre_workers": cfg["pre_workers"],
                "tiles_total": res["tiles_total"],
                "num_images": res["num_images"],
                "total_time_s": res["total_time_s"],
                "overall_tile_s": res["overall_tile_s"],
                "steady_tile_s": res["steady_tile_s"],
                "overall_img_s": res["overall_img_s"],
                "steady_img_s": res["steady_img_s"],
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
            run_results.append(one)

            if best_run is None or one["steady_tile_s"] > best_run["steady_tile_s"]:
                best_run = one

            print(f"[SWEEP]   ({r+1}/{args.repeat}) done: "
                  f"steady_tile/s={one['steady_tile_s']:.2f}, overall_tile/s={one['overall_tile_s']:.2f}")

        if not run_results:
            print(f"[SWEEP][WARN] no successful runs for config: {base_name}")
            continue

        steady_list = [x["steady_tile_s"] for x in run_results]
        overall_list = [x["overall_tile_s"] for x in run_results]
        total_time_list = [x["total_time_s"] for x in run_results]
        overall_img_list = [x["overall_img_s"] for x in run_results]
        steady_img_list = [x["steady_img_s"] for x in run_results]

        steady_mean, steady_std = mean_std(steady_list)
        overall_mean, overall_std = mean_std(overall_list)
        total_mean, total_std = mean_std(total_time_list)
        oimg_mean, oimg_std = mean_std(overall_img_list)
        simg_mean, simg_std = mean_std(steady_img_list)

        stage_keys = [
            "mean_decode_compute_ms",
            "mean_tile_compute_ms",
            "mean_decode_wait_ms",
            "mean_pre_ms",
            "mean_submit_ms",
            "mean_sync_wait_ms",
            "mean_total_ms",
        ]
        stage_means = {k: statistics.mean([x[k] for x in run_results]) for k in stage_keys}

        assert best_run is not None

        row = {
            "cpu_workers": w,
            "prefetch": pf,
            "bs": args.bs,
            "global_mix": args.global_mix,
            "decode_workers": best_run["decode_workers"],
            "pre_workers": best_run["pre_workers"],
            "repeat": args.repeat,
            "success_runs": len(run_results),
            "tiles_total": best_run["tiles_total"],
            "num_images": best_run["num_images"],
            "total_time_s_mean": total_mean,
            "total_time_s_std": total_std,
            "overall_tile_s_mean": overall_mean,
            "overall_tile_s_std": overall_std,
            "steady_tile_s_mean": steady_mean,
            "steady_tile_s_std": steady_std,
            "overall_img_s_mean": oimg_mean,
            "overall_img_s_std": oimg_std,
            "steady_img_s_mean": simg_mean,
            "steady_img_s_std": simg_std,
            **{k + "_mean": stage_means[k] for k in stage_keys},
            "best_out_tsv": best_run["out_tsv"],
            "best_summary_json": best_run["summary_json"],
        }

        all_rows.append(row)

        with open(summary_tsv, "a", encoding="utf-8") as f:
            f.write("\t".join([
                str(row["cpu_workers"]),
                str(row["prefetch"]),
                str(row["bs"]),
                str(row["global_mix"]),
                str(row["decode_workers"]),
                str(row["pre_workers"]),
                str(row["repeat"]),
                str(row["success_runs"]),
                str(row["tiles_total"]),
                str(row["num_images"]),
                f"{row['total_time_s_mean']:.6f}",
                f"{row['total_time_s_std']:.6f}",
                f"{row['overall_tile_s_mean']:.6f}",
                f"{row['overall_tile_s_std']:.6f}",
                f"{row['steady_tile_s_mean']:.6f}",
                f"{row['steady_tile_s_std']:.6f}",
                f"{row['overall_img_s_mean']:.6f}",
                f"{row['overall_img_s_std']:.6f}",
                f"{row['steady_img_s_mean']:.6f}",
                f"{row['steady_img_s_std']:.6f}",
                f"{row['mean_decode_compute_ms_mean']:.3f}",
                f"{row['mean_tile_compute_ms_mean']:.3f}",
                f"{row['mean_decode_wait_ms_mean']:.3f}",
                f"{row['mean_pre_ms_mean']:.3f}",
                f"{row['mean_submit_ms_mean']:.3f}",
                f"{row['mean_sync_wait_ms_mean']:.3f}",
                f"{row['mean_total_ms_mean']:.3f}",
                row["best_out_tsv"],
                row["best_summary_json"],
            ]) + "\n")

        print(f"[SWEEP] aggregated: steady_tile/s={steady_mean:.2f}±{steady_std:.2f} "
              f"(overall={overall_mean:.2f}±{overall_std:.2f})")

    if not all_rows:
        print("[SWEEP] no successful runs.")
        return

    all_rows_sorted = sorted(all_rows, key=lambda r: r["steady_tile_s_mean"], reverse=True)

    print("\n================ [SWEEP RANK by steady_tile/s_mean] ================")
    for i, r in enumerate(all_rows_sorted, 1):
        print(f"[{i:02d}] steady_tile/s={r['steady_tile_s_mean']:.2f}±{r['steady_tile_s_std']:.2f} "
              f"(overall={r['overall_tile_s_mean']:.2f}±{r['overall_tile_s_std']:.2f}) "
              f"w={r['cpu_workers']} pf={r['prefetch']} bs={r['bs']} gm={r['global_mix']} "
              f"decode_w={r['decode_workers']} pre_w={r['pre_workers']} "
              f"mean_decode_wait_ms={r['mean_decode_wait_ms_mean']:.1f} "
              f"mean_sync_wait_ms={r['mean_sync_wait_ms_mean']:.1f} "
              f"best_out={os.path.basename(r['best_out_tsv'])}")
    print("====================================================================\n")
    print(f"[SWEEP] summary saved at: {summary_tsv}")


if __name__ == "__main__":
    main()