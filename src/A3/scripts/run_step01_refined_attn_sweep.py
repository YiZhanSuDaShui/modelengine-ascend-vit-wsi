from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.submission_defaults import get_default_backend_artifacts  # noqa: E402
from bach_mil.utils.io import load_json, save_json  # noqa: E402
from bach_mil.utils.optimization_steps import ensure_step_layout, write_markdown_report  # noqa: E402


REFINED_PROFILES = [
    'attn_softmax_div_sqrt',
    'attn_softmax_div_sqrt_mul',
    'attn_score_path',
    'attn_score_path_norm1',
    'attn_score_path_norm1_qkv_proj',
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT.parents[1]), capture_output=True, text=True)


def _load_existing_baseline(step_layout: dict) -> list[dict]:
    rows: list[dict] = []
    sweep_json = step_layout['analysis'] / 'targeted_sweep' / 'targeted_mixed_sweep_summary.json'
    if not sweep_json.exists():
        return rows
    payload = load_json(sweep_json)
    for row in payload.get('profiles', []):
        rows.append(
            {
                'profile': f"baseline::{row.get('profile', 'unknown')}",
                'compile_status': row.get('compile_status'),
                'compile_elapsed_sec': row.get('compile_elapsed_sec'),
                'compile_returncode': row.get('compile_returncode'),
                'real_features_finite_ratio': row.get('real_features_finite_ratio'),
                'real_logits_finite_ratio': row.get('real_logits_finite_ratio'),
                'real_probs_finite_ratio': row.get('real_probs_finite_ratio'),
                'bench_patches_per_sec': row.get('bench_patches_per_sec'),
                'bench_sec_per_patch': row.get('bench_sec_per_patch'),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=3)
    parser.add_argument('--bench_steps', type=int, default=10)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    step_layout = ensure_step_layout('step01_mixed_keepdtype')
    sweep_root = step_layout['analysis'] / 'refined_attn_sweep'
    sweep_root.mkdir(parents=True, exist_ok=True)
    artifacts = get_default_backend_artifacts(args.task)
    onnx_path = Path(artifacts['onnx'])
    meta_json = Path(artifacts['meta_json'])

    rows = _load_existing_baseline(step_layout)
    refined_rows = []
    for profile in REFINED_PROFILES:
        config_dir = step_layout['configs'] / f'refined_{profile}'
        artifact_dir = step_layout['artifacts'] / f'refined_{profile}'
        config_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        build_cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'build_mixed_precision_mixlist.py'),
            '--onnx_path',
            str(onnx_path),
            '--out_dir',
            str(config_dir),
            '--profile',
            profile,
        ]
        _run(build_cmd)

        om_path = artifact_dir / f'wsi_{profile}.om'
        compile_json = artifact_dir / 'compile_summary.json'
        compile_cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'compile_patch_encoder_om.py'),
            '--task',
            'wsi',
            '--onnx_path',
            str(onnx_path),
            '--meta_json',
            str(meta_json),
            '--out_om',
            str(om_path),
            '--report_json',
            str(compile_json),
            '--precision_mode_v2',
            'mixed_float16',
            '--modify_mixlist',
            str(config_dir / f'modify_mixlist_{profile}.json'),
            '--keep_dtype',
            str(config_dir / f'keep_dtype_{profile}.txt'),
        ]
        compile_proc = _run(compile_cmd)
        (artifact_dir / 'compile.stdout.txt').write_text(compile_proc.stdout, encoding='utf-8')
        (artifact_dir / 'compile.stderr.txt').write_text(compile_proc.stderr, encoding='utf-8')

        row = {'profile': profile}
        if compile_json.exists():
            row.update({f'compile_{k}': v for k, v in load_json(compile_json).items() if k in {'status', 'elapsed_sec', 'returncode'}})
        if om_path.exists():
            diag_json = artifact_dir / 'diag_real.json'
            bench_json = artifact_dir / 'bench_summary.json'
            diag_cmd = [
                sys.executable,
                str(ROOT / 'scripts' / 'diagnose_patch_backend_nan.py'),
                '--task',
                'wsi',
                '--backend',
                'om',
                '--sample_source',
                'real_wsi_tiles',
                '--om_path',
                str(om_path),
                '--meta_json',
                str(om_path.with_suffix('.meta.json')),
                '--report_json',
                str(diag_json),
                '--device_id',
                str(args.device_id),
            ]
            bench_cmd = [
                sys.executable,
                str(ROOT / 'scripts' / 'bench_patch_backend.py'),
                '--task',
                'wsi',
                '--backend',
                'om',
                '--om_path',
                str(om_path),
                '--meta_json',
                str(om_path.with_suffix('.meta.json')),
                '--report_json',
                str(bench_json),
                '--batch_size',
                str(args.batch_size),
                '--warmup_steps',
                str(args.warmup_steps),
                '--bench_steps',
                str(args.bench_steps),
                '--device_id',
                str(args.device_id),
            ]
            diag_proc = _run(diag_cmd)
            bench_proc = _run(bench_cmd)
            (artifact_dir / 'diag.stdout.txt').write_text(diag_proc.stdout, encoding='utf-8')
            (artifact_dir / 'diag.stderr.txt').write_text(diag_proc.stderr, encoding='utf-8')
            (artifact_dir / 'bench.stdout.txt').write_text(bench_proc.stdout, encoding='utf-8')
            (artifact_dir / 'bench.stderr.txt').write_text(bench_proc.stderr, encoding='utf-8')
            if diag_json.exists():
                diag = load_json(diag_json)
                row['real_features_finite_ratio'] = diag['features_stats']['finite_ratio']
                row['real_logits_finite_ratio'] = diag['logits_stats']['finite_ratio']
                row['real_probs_finite_ratio'] = diag['probs_stats']['finite_ratio']
            if bench_json.exists():
                bench = load_json(bench_json)
                row['bench_patches_per_sec'] = bench['patches_per_sec']
                row['bench_sec_per_patch'] = bench['sec_per_patch']
        refined_rows.append(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = sweep_root / 'refined_attn_sweep_summary.csv'
    df.to_csv(out_csv, index=False)

    refined_df = pd.DataFrame(refined_rows)
    stable_df = refined_df[refined_df.get('real_probs_finite_ratio', 0).fillna(0) >= 1.0] if not refined_df.empty and 'real_probs_finite_ratio' in refined_df.columns else pd.DataFrame()
    best_refined = None
    if not stable_df.empty:
        best_refined = stable_df.sort_values('bench_patches_per_sec', ascending=False).iloc[0].to_dict()

    baseline_best = None
    baseline_df = df[df['profile'].astype(str).str.startswith('baseline::')].copy() if not df.empty else pd.DataFrame()
    if not baseline_df.empty and 'real_probs_finite_ratio' in baseline_df.columns:
        baseline_df = baseline_df[baseline_df['real_probs_finite_ratio'].fillna(0) >= 1.0]
        if not baseline_df.empty:
            baseline_best = baseline_df.sort_values('bench_patches_per_sec', ascending=False).iloc[0].to_dict()

    summary = {
        'profiles': rows,
        'best_refined_profile': best_refined,
        'best_baseline_profile': baseline_best,
        'summary_csv': str(out_csv),
    }
    out_json = sweep_root / 'refined_attn_sweep_summary.json'
    out_md = sweep_root / 'refined_attn_sweep_report.md'
    save_json(summary, out_json)

    summary_lines = [
        f'- 汇总 CSV：`{out_csv}`',
        f"- 最佳 refined profile：`{best_refined['profile']}`" if best_refined is not None else '- 最佳 refined profile：`暂无`',
    ]
    if best_refined is not None and baseline_best is not None:
        delta = float(best_refined['bench_patches_per_sec']) - float(baseline_best['bench_patches_per_sec'])
        pct = delta / max(float(baseline_best['bench_patches_per_sec']), 1e-12) * 100.0
        summary_lines.append(
            f"- 相对上一轮最佳稳定 profile：`{baseline_best['profile']}`，吞吐变化 `{delta:.2f} patch/s`（`{pct:.2f}%`）"
        )

    write_markdown_report(
        out_md,
        title='Step01 Refined Attention Sweep 报告',
        summary_lines=summary_lines,
        sections=[
            (
                '说明',
                [
                    '- 本轮不再保护整条 `/attn/`，而是只保护 attention 分数计算链上的 Softmax/Div/Sqrt/Mul/score MatMul，并逐步补回 norm1 与 qkv/proj。',
                    '- 目标是在不重新引入 NaN 的前提下，尽量缩小 keep_dtype 范围，继续把稳定 mixed OM 的吞吐往上推。',
                ],
            ),
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
