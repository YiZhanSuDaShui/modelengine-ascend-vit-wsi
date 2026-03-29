from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.submission_defaults import get_default_backend_artifacts  # noqa: E402
from bach_mil.utils.io import load_json, save_json  # noqa: E402
from bach_mil.utils.optimization_steps import build_step_manifest, ensure_step_layout, write_markdown_report  # noqa: E402


VARIANTS = [
    {
        'name': 'mixed_plain',
        'profile': 'plain',
        'precision_mode_v2': 'mixed_float16',
        'use_keep_dtype': False,
        'use_modify_mixlist': False,
    },
    {
        'name': 'mixed_keep_ln_softmax',
        'profile': 'ln_softmax',
        'precision_mode_v2': 'mixed_float16',
        'use_keep_dtype': True,
        'use_modify_mixlist': True,
    },
    {
        'name': 'mixed_keep_ln_softmax_head_mlp',
        'profile': 'ln_softmax_head_mlp',
        'precision_mode_v2': 'mixed_float16',
        'use_keep_dtype': True,
        'use_modify_mixlist': True,
    },
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT.parents[1]), capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--bench_steps', type=int, default=30)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--skip_compile', action='store_true')
    parser.add_argument('--skip_parity', action='store_true')
    parser.add_argument('--skip_bench', action='store_true')
    args = parser.parse_args()

    layout = ensure_step_layout('step01_mixed_keepdtype')
    manifest = build_step_manifest('step01_mixed_keepdtype', task=args.task)
    artifacts = get_default_backend_artifacts(args.task)
    onnx_path = Path(artifacts['onnx'])

    rows: list[dict] = []
    for variant in VARIANTS:
        variant_dir = layout['artifacts'] / variant['name']
        variant_dir.mkdir(parents=True, exist_ok=True)
        mix_dir = layout['configs'] / variant['name']
        mix_dir.mkdir(parents=True, exist_ok=True)

        build_cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'build_mixed_precision_mixlist.py'),
            '--onnx_path',
            str(onnx_path),
            '--out_dir',
            str(mix_dir),
            '--profile',
            str(variant['profile']),
        ]
        build_proc = _run(build_cmd)
        (variant_dir / 'build_mixlist.stdout.txt').write_text(build_proc.stdout, encoding='utf-8')
        (variant_dir / 'build_mixlist.stderr.txt').write_text(build_proc.stderr, encoding='utf-8')

        om_path = variant_dir / f'{args.task}_{variant["name"]}.om'
        compile_json = variant_dir / 'compile_summary.json'
        parity_json = variant_dir / 'parity_summary.json'
        bench_json = variant_dir / 'bench_summary.json'

        compile_status = 'skipped'
        if not args.skip_compile:
            compile_cmd = [
                sys.executable,
                str(ROOT / 'scripts' / 'compile_patch_encoder_om.py'),
                '--task',
                str(args.task),
                '--onnx_path',
                str(onnx_path),
                '--meta_json',
                str(artifacts['meta_json']),
                '--out_om',
                str(om_path),
                '--report_json',
                str(compile_json),
                '--precision_mode_v2',
                str(variant['precision_mode_v2']),
            ]
            if variant['use_keep_dtype']:
                compile_cmd.extend(['--keep_dtype', str(mix_dir / f'keep_dtype_{variant["profile"]}.txt')])
            if variant['use_modify_mixlist']:
                compile_cmd.extend(['--modify_mixlist', str(mix_dir / f'modify_mixlist_{variant["profile"]}.json')])
            proc = _run(compile_cmd)
            (variant_dir / 'compile.stdout.txt').write_text(proc.stdout, encoding='utf-8')
            (variant_dir / 'compile.stderr.txt').write_text(proc.stderr, encoding='utf-8')
            compile_status = 'ok' if proc.returncode == 0 and om_path.exists() else 'failed'

        parity_status = 'skipped'
        if not args.skip_parity and om_path.exists():
            parity_cmd = [
                sys.executable,
                str(ROOT / 'scripts' / 'validate_patch_backend_parity.py'),
                '--task',
                str(args.task),
                '--after_backend',
                'om',
                '--om_path',
                str(om_path),
                '--meta_json',
                str(om_path.with_suffix('.meta.json')),
                '--report_json',
                str(parity_json),
                '--device_id',
                str(args.device_id),
                '--om_host_io_mode',
                'legacy',
                '--om_output_mode',
                'both',
            ]
            proc = _run(parity_cmd)
            (variant_dir / 'parity.stdout.txt').write_text(proc.stdout, encoding='utf-8')
            (variant_dir / 'parity.stderr.txt').write_text(proc.stderr, encoding='utf-8')
            parity_status = 'ok' if proc.returncode == 0 and parity_json.exists() else 'failed'

        bench_status = 'skipped'
        if not args.skip_bench and om_path.exists():
            bench_cmd = [
                sys.executable,
                str(ROOT / 'scripts' / 'bench_patch_backend.py'),
                '--task',
                str(args.task),
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
                '--om_host_io_mode',
                'legacy',
                '--om_output_mode',
                'both',
            ]
            proc = _run(bench_cmd)
            (variant_dir / 'bench.stdout.txt').write_text(proc.stdout, encoding='utf-8')
            (variant_dir / 'bench.stderr.txt').write_text(proc.stderr, encoding='utf-8')
            bench_status = 'ok' if proc.returncode == 0 and bench_json.exists() else 'failed'

        row = {
            'variant': variant['name'],
            'profile': variant['profile'],
            'compile_status': compile_status,
            'parity_status': parity_status,
            'bench_status': bench_status,
            'om_path': str(om_path),
            'keep_dtype_path': str(mix_dir / f'keep_dtype_{variant["profile"]}.txt'),
            'modify_mixlist_path': str(mix_dir / f'modify_mixlist_{variant["profile"]}.json'),
        }
        if compile_json.exists():
            row.update({f'compile_{k}': v for k, v in load_json(compile_json).items() if k in {'status', 'elapsed_sec', 'precision_mode_v2'}})
        if parity_json.exists():
            row.update({f'parity_{k}': v for k, v in load_json(parity_json).items() if k in {'feature_cosine_mean', 'logit_cosine_mean', 'prob_max_abs_diff', 'prob_mean_abs_diff'}})
        if bench_json.exists():
            row.update({f'bench_{k}': v for k, v in load_json(bench_json).items() if k in {'patches_per_sec', 'sec_per_patch', 'elapsed_sec'}})
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_csv = layout['analysis'] / 'mixed_precision_summary.csv'
    df.to_csv(summary_csv, index=False)
    best_variant = None
    if not df.empty and 'bench_patches_per_sec' in df.columns:
        stable_df = df[df.compile_status.eq('ok')]
        if 'parity_prob_max_abs_diff' in stable_df.columns:
            stable_df = stable_df[stable_df.parity_prob_max_abs_diff.fillna(1e9) < 0.2]
        if not stable_df.empty:
            best_variant = stable_df.sort_values('bench_patches_per_sec', ascending=False).iloc[0].to_dict()

    summary_json = {
        'step': 'step01_mixed_keepdtype',
        'manifest': manifest,
        'summary_csv': str(summary_csv),
        'best_variant': best_variant,
        'variants': rows,
    }
    save_json(summary_json, layout['reports'] / 'step01_summary.json')
    write_markdown_report(
        layout['reports'] / '01_mixed_precision报告.md',
        title='Step01 Mixed Precision 报告',
        summary_lines=[
            f'- 任务：`{args.task}`',
            f'- ONNX：`{onnx_path}`',
            f'- 汇总 CSV：`{summary_csv}`',
            f'- 最优候选：`{best_variant["variant"]}`' if best_variant is not None else '- 最优候选：`暂无`',
        ],
        sections=[
            (
                '阶段说明',
                [
                    '- 本阶段只改 OM 编译精度策略，不改 checkpoint、切块参数、阈值和聚合逻辑。',
                    '- 当前测试轮次包含 plain mixed、保 LayerNorm/Softmax、再额外保 head+MLP 关键节点三类。 ',
                ],
            ),
            (
                '自检',
                [
                    f'- 编译轮次数：`{len(rows)}`',
                    f'- 成功编译轮次数：`{int(df.compile_status.eq("ok").sum()) if not df.empty else 0}`',
                    f'- 生成汇总：`{summary_csv}`',
                ],
            ),
        ],
    )
    print(summary_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
