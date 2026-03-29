#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_a2_grid_sweep_V3.py
========================
最终验证版 sweep：二维扫描 cpu_workers × prefetch，固定 bs=96。

核心特性：
✅ 轮次制（Round-Robin）：每轮跑完所有配置，再进入下一轮
   → 避免某个配置连续吃到系统好/坏状态
✅ 每轮之间短暂休息（默认 10 秒），让系统调度恢复平稳
✅ 用中位数排名（抗离群值）
✅ skip_steady=3（减少启动抖动）
✅ 完整统计输出（median / mean / std / min / max）

运行顺序示意：
  Round 1: (w=2,pf=2) → (w=2,pf=3) → ... → (w=18,pf=6) → 休息
  Round 2: (w=2,pf=2) → (w=2,pf=3) → ... → (w=18,pf=6) → 休息
  ...
  Round 5: (w=2,pf=2) → (w=2,pf=3) → ... → (w=18,pf=6)

用法示例：
  python run_a2_grid_sweep_V3.py \
    --a2_script src/A2_final_opt_v3.py \
    --file_list data/BACH/derived/split/photos_test_all.txt \
    --ckpt_dir assets/ckpts/UNI \
    --bs 96 --global_mix 1 \
    --precision fp16 --white_thr 1.01 \
    --warmup_mode async --warmup_iters 3 \
    --decode_prefetch 4 \
    --worker_list 2,4,6,8,10,12,14,16,18 \
    --prefetch_list 2,3,4,5,6 \
    --repeat 5 --skip_steady 3 \
    --round_rest 10
"""

import os
import json
import time
import argparse
import subprocess
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


# ==================== 统计工具 ====================

def robust_stats(xs: List[float]) -> Dict[str, float]:
    """计算全套统计量：median, mean, std, min, max"""
    if not xs:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    if len(xs) == 1:
        v = float(xs[0])
        return {"median": v, "mean": v, "std": 0.0, "min": v, "max": v, "n": 1}
    return {
        "median": float(statistics.median(xs)),
        "mean": float(statistics.mean(xs)),
        "std": float(statistics.stdev(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "n": len(xs),
    }


# ==================== 主函数 ====================

def main() -> None:
    ap = argparse.ArgumentParser(description="A2 最终验证 sweep：轮次制 Round-Robin")

    # 基本输入/输出
    ap.add_argument("--a2_script", type=str, default="src/A2_final_opt_v3.py")
    ap.add_argument("--file_list", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="assets/ckpts/UNI")
    ap.add_argument("--out_dir", type=str, default="logs/sweep_v3")

    # 固定口径
    ap.add_argument("--tile", type=int, default=224)
    ap.add_argument("--stride", type=int, default=224)
    ap.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--white_thr", type=float, default=1.01)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--global_mix", type=int, default=1, choices=[0, 1])

    # warmup
    ap.add_argument("--warmup_mode", type=str, choices=["none", "sync", "async"], default="async")
    ap.add_argument("--warmup_iters", type=int, default=3)
    ap.add_argument("--skip_steady", type=int, default=3)
    ap.add_argument("--max_images", type=int, default=-1)

    # 二维 sweep
    ap.add_argument("--worker_list", type=str, default="2,4,6,8,10,12,14,16,18",
                    help="要扫的 cpu_workers 列表")
    ap.add_argument("--prefetch_list", type=str, default="2,3,4,5,6",
                    help="要扫的 prefetch 列表")

    # 运行控制
    ap.add_argument("--decode_prefetch", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=5,
                    help="总轮数（每轮跑完所有配置）")
    ap.add_argument("--round_rest", type=int, default=10,
                    help="每轮之间的休息秒数（让系统恢复平稳）")
    ap.add_argument("--run_gap", type=int, default=3,
                    help="每次运行后等待秒数（让 NPU driver 回收内存，防止 OOM）")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")

    args = ap.parse_args()

    worker_list = [int(x) for x in args.worker_list.split(",") if x.strip()]
    prefetch_list = [int(x) for x in args.prefetch_list.split(",") if x.strip()]
    configs = [(w, pf) for w in worker_list for pf in prefetch_list]
    total_configs = len(configs)
    total_runs = total_configs * args.repeat

    os.makedirs(args.out_dir, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_tsv = os.path.join(args.out_dir, f"A2_sweep_v3_{run_tag}.tsv")

    # ==================== 打印 Sweep 计划 ====================
    print("=" * 70)
    print(f"  🔬 SWEEP V3 — 轮次制 Round-Robin")
    print("=" * 70)
    print(f"  配置数     : {total_configs} (workers={worker_list} × prefetch={prefetch_list})")
    print(f"  总轮数     : {args.repeat}")
    print(f"  总运行次数 : {total_runs}")
    print(f"  轮间休息   : {args.round_rest}s")
    print(f"  运行间隔   : {args.run_gap}s（NPU 内存回收）")
    print(f"  固定参数   : bs={args.bs}, global_mix={args.global_mix}, skip_steady={args.skip_steady}")

    # 预估时间
    est_per_run = 14  # 秒/次（含 warmup）
    est_rest = (args.repeat - 1) * args.round_rest
    est_total_min = (total_runs * est_per_run + est_rest) / 60
    print(f"  预计耗时   : ~{est_total_min:.0f} 分钟")
    print(f"  输出文件   : {summary_tsv}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] 仅展示计划，不执行。")
        for r in range(args.repeat):
            print(f"\n  Round {r+1}/{args.repeat}:")
            for idx, (w, pf) in enumerate(configs, 1):
                print(f"    [{idx:>3}/{total_configs}] w={w:>2} pf={pf}")
        return

    # ==================== 收集结果的容器 ====================
    # key = (w, pf), value = list of result dicts
    results_by_config: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    global_run_idx = 0
    sweep_start = time.time()

    # ==================== 轮次制执行 ====================
    for round_idx in range(args.repeat):
        round_start = time.time()
        print(f"\n{'━' * 70}")
        print(f"  📋 Round {round_idx + 1}/{args.repeat}  "
              f"(已完成 {global_run_idx}/{total_runs} 次运行)")
        print(f"{'━' * 70}")

        for config_idx, (w, pf) in enumerate(configs, 1):
            global_run_idx += 1
            run_name = f"w{w}_pf{pf}_bs{args.bs}_r{round_idx + 1}"
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

            # 进度条 + 配置信息
            elapsed = time.time() - sweep_start
            eta_per_run = elapsed / global_run_idx if global_run_idx > 1 else est_per_run
            eta_remain = eta_per_run * (total_runs - global_run_idx)
            eta_min = eta_remain / 60

            print(f"  [{global_run_idx:>3}/{total_runs}] R{round_idx+1} "
                  f"w={w:>2} pf={pf} ", end="", flush=True)

            ret = subprocess.run(cmd, capture_output=True, text=True)

            if ret.returncode != 0:
                print(f"❌ (ETA ~{eta_min:.0f}min)")
                if ret.stderr:
                    for line in ret.stderr.strip().split("\n")[-2:]:
                        print(f"       stderr: {line}")
                if args.stop_on_error:
                    raise RuntimeError(f"因错误停止: {run_name}")
                # OOM 后额外等待，确保 NPU 内存彻底释放
                if "out of memory" in (ret.stderr or "").lower():
                    oom_wait = args.run_gap * 2
                    print(f"       💤 OOM 后等待 {oom_wait}s 回收 NPU 内存 ...")
                    time.sleep(oom_wait)
                continue

            if not os.path.exists(out_json):
                print(f"⚠️ 缺少 json (ETA ~{eta_min:.0f}min)")
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
                "mean_decode_compute_ms": res["mean_decode_compute_ms"],
                "mean_tile_compute_ms": res["mean_tile_compute_ms"],
                "mean_decode_wait_ms": res["mean_decode_wait_ms"],
                "mean_pre_ms": res["mean_pre_ms"],
                "mean_submit_ms": res["mean_submit_ms"],
                "mean_sync_wait_ms": res["mean_sync_wait_ms"],
                "mean_total_ms": res["mean_total_ms"],
                "out_tsv": out_tsv,
                "summary_json": out_json,
                "round": round_idx + 1,
            }

            results_by_config[(w, pf)].append(one)
            print(f"✓ steady={one['steady_tile_s']:>6.1f}  (ETA ~{eta_min:.0f}min)")

            # 每次运行后短暂等待，让 NPU driver 回收内存
            if args.run_gap > 0:
                time.sleep(args.run_gap)

        # 轮间统计
        round_time = time.time() - round_start
        print(f"\n  ⏱️  Round {round_idx + 1} 完成，耗时 {round_time:.0f}s")

        # 轮间休息（最后一轮不休息）
        if round_idx < args.repeat - 1 and args.round_rest > 0:
            print(f"  💤 休息 {args.round_rest}s ...")
            time.sleep(args.round_rest)

    # ==================== 聚合统计 ====================
    print(f"\n{'━' * 70}")
    print(f"  📊 聚合 {len(results_by_config)} 个配置的统计结果 ...")
    print(f"{'━' * 70}")

    all_rows: List[Dict[str, Any]] = []

    for (w, pf), runs in results_by_config.items():
        if not runs:
            continue

        steady_stats = robust_stats([x["steady_tile_s"] for x in runs])
        overall_stats = robust_stats([x["overall_tile_s"] for x in runs])
        total_time_stats = robust_stats([x["total_time_s"] for x in runs])

        stage_keys = [
            "mean_decode_compute_ms", "mean_tile_compute_ms",
            "mean_decode_wait_ms", "mean_pre_ms",
            "mean_submit_ms", "mean_sync_wait_ms", "mean_total_ms",
        ]
        stage_means = {k: statistics.mean([x[k] for x in runs]) for k in stage_keys}

        # 找该配置的最佳单次运行
        best_run = max(runs, key=lambda x: x["steady_tile_s"])

        row = {
            "cpu_workers": w,
            "prefetch": pf,
            "bs": args.bs,
            "global_mix": args.global_mix,
            "decode_workers": best_run["decode_workers"],
            "pre_workers": best_run["pre_workers"],
            "repeat": args.repeat,
            "success_runs": len(runs),
            "tiles_total": best_run["tiles_total"],
            "num_images": best_run["num_images"],
            # 核心指标
            "steady_median": steady_stats["median"],
            "steady_mean": steady_stats["mean"],
            "steady_std": steady_stats["std"],
            "steady_min": steady_stats["min"],
            "steady_max": steady_stats["max"],
            "overall_median": overall_stats["median"],
            "overall_mean": overall_stats["mean"],
            "overall_std": overall_stats["std"],
            "total_time_median": total_time_stats["median"],
            # 分段耗时
            **stage_means,
            # 追溯
            "best_out_tsv": best_run["out_tsv"],
            "best_summary_json": best_run["summary_json"],
        }
        all_rows.append(row)

    # ==================== 写入 TSV ====================
    header = [
        "cpu_workers", "prefetch", "bs", "global_mix",
        "decode_workers", "pre_workers",
        "repeat", "success_runs", "tiles_total", "num_images",
        "steady_tile_s_median", "steady_tile_s_mean", "steady_tile_s_std",
        "steady_tile_s_min", "steady_tile_s_max",
        "overall_tile_s_median", "overall_tile_s_mean", "overall_tile_s_std",
        "total_time_s_median",
        "mean_decode_compute_ms", "mean_tile_compute_ms",
        "mean_decode_wait_ms", "mean_pre_ms",
        "mean_submit_ms", "mean_sync_wait_ms", "mean_total_ms",
        "best_out_tsv", "best_summary_json",
    ]

    with open(summary_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in all_rows:
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
                f"{row['steady_median']:.2f}",
                f"{row['steady_mean']:.2f}",
                f"{row['steady_std']:.2f}",
                f"{row['steady_min']:.2f}",
                f"{row['steady_max']:.2f}",
                f"{row['overall_median']:.2f}",
                f"{row['overall_mean']:.2f}",
                f"{row['overall_std']:.2f}",
                f"{row['total_time_median']:.3f}",
                f"{row['mean_decode_compute_ms']:.3f}",
                f"{row['mean_tile_compute_ms']:.3f}",
                f"{row['mean_decode_wait_ms']:.3f}",
                f"{row['mean_pre_ms']:.3f}",
                f"{row['mean_submit_ms']:.3f}",
                f"{row['mean_sync_wait_ms']:.3f}",
                f"{row['mean_total_ms']:.3f}",
                row["best_out_tsv"],
                row["best_summary_json"],
            ]) + "\n")

    # ==================== 最终排名 ====================
    all_rows_sorted = sorted(all_rows, key=lambda r: r["steady_median"], reverse=True)

    total_elapsed = time.time() - sweep_start
    print(f"\n{'=' * 80}")
    print(f"  🏆  SWEEP V3 最终排名（按 steady_tile/s 中位数降序）")
    print(f"  总耗时: {total_elapsed/60:.1f} 分钟 | 成功运行: {sum(r['success_runs'] for r in all_rows)}/{total_runs}")
    print(f"{'=' * 80}")
    print(f"{'排名':>4} {'配置':<16} {'中位数':>8} {'均值±标准差':>16} "
          f"{'范围':>18} {'sync_wait':>10} {'pre_ms':>8}")
    print("-" * 80)

    # 显示所有配置（但重点标记前 10 名）
    for i, r in enumerate(all_rows_sorted, 1):
        config_str = f"w={r['cpu_workers']:>2} pf={r['prefetch']}"
        median_str = f"{r['steady_median']:.1f}"
        mean_std_str = f"{r['steady_mean']:.1f}±{r['steady_std']:.1f}"
        range_str = f"[{r['steady_min']:.0f}, {r['steady_max']:.0f}]"
        sync_str = f"{r['mean_sync_wait_ms']:.1f}ms"
        pre_str = f"{r['mean_pre_ms']:.1f}ms"

        if i == 1:
            marker = "🥇"
        elif i == 2:
            marker = "🥈"
        elif i == 3:
            marker = "🥉"
        elif i <= 10:
            marker = "  "
        else:
            marker = "  "

        print(f"{marker}{i:>2} {config_str:<16} {median_str:>8} {mean_std_str:>16} "
              f"{range_str:>18} {sync_str:>10} {pre_str:>8}")

        # 前 10 名之后用分隔线
        if i == 10 and len(all_rows_sorted) > 10:
            print(f"   {'─' * 74}")

    print("=" * 80)

    # 推荐
    best = all_rows_sorted[0]
    print(f"\n✅ 推荐配置: cpu_workers={best['cpu_workers']}, prefetch={best['prefetch']}, "
          f"bs={best['bs']}, global_mix={best['global_mix']}")
    print(f"   steady_tile/s 中位数: {best['steady_median']:.2f}")
    print(f"   decode_workers={best['decode_workers']}, pre_workers={best['pre_workers']}")

    # 差距提示
    if len(all_rows_sorted) >= 2:
        gap = abs(all_rows_sorted[0]["steady_median"] - all_rows_sorted[1]["steady_median"])
        pct = gap / all_rows_sorted[0]["steady_median"] * 100 if all_rows_sorted[0]["steady_median"] > 0 else 0
        second = all_rows_sorted[1]
        if pct < 5:
            print(f"\n⚠️  注意：第一名与第二名差距仅 {pct:.1f}%"
                  f"（w={second['cpu_workers']} pf={second['prefetch']}），"
                  f"可视为同一档次。选更稳定的（std 更小的）即可。")
        else:
            print(f"\n🎯 第一名领先第二名 {pct:.1f}%，配置优势明确。")

    print(f"\n📁 详细数据: {summary_tsv}")
    print()


if __name__ == "__main__":
    main()
