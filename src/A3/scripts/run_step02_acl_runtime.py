from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.patch_backends import create_patch_backend  # noqa: E402
from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import load_json, save_json  # noqa: E402
from bach_mil.utils.optimization_steps import build_step_manifest, ensure_step_layout, write_markdown_report  # noqa: E402


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(ROOT.parents[1]), capture_output=True, text=True)


def _compare_runtime_modes(task: str, om_path: Path, meta_json: Path, *, batch_size: int, seed: int, device_id: int) -> dict:
    defaults = get_task_defaults(task)
    sample = np.random.default_rng(seed).standard_normal((batch_size, 3, int(defaults['input_size']), int(defaults['input_size'])), dtype=np.float32)
    sample_tensor = torch.from_numpy(sample)
    outputs = {}
    for mode in ['legacy', 'buffer_reuse']:
        backend = create_patch_backend(
            backend='om',
            ckpt_path=defaults['ckpt'],
            model_name=defaults['model_name'],
            backbone_pool=defaults['backbone_pool'],
            backbone_init_values=defaults['backbone_init_values'],
            default_classes=defaults['labels'],
            input_size=int(defaults['input_size']),
            om_path=om_path,
            meta_json=meta_json,
            device_id=device_id,
            om_execution_mode='sync',
            om_host_io_mode=mode,
            om_output_mode='both',
        )
        try:
            outputs[mode] = backend.predict_batch(sample_tensor)
        finally:
            backend.close()
    return {
        'feature_max_abs_diff': float(np.max(np.abs(outputs['legacy'].features - outputs['buffer_reuse'].features))),
        'feature_mean_abs_diff': float(np.mean(np.abs(outputs['legacy'].features - outputs['buffer_reuse'].features))),
        'logit_max_abs_diff': float(np.max(np.abs(outputs['legacy'].logits - outputs['buffer_reuse'].logits))),
        'logit_mean_abs_diff': float(np.mean(np.abs(outputs['legacy'].logits - outputs['buffer_reuse'].logits))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    parser.add_argument('--om_path', type=str, default=None)
    parser.add_argument('--meta_json', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--bench_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    layout = ensure_step_layout('step02_acl_runtime')
    manifest = build_step_manifest('step02_acl_runtime', task=args.task)
    artifacts = get_default_backend_artifacts(args.task)
    om_path = Path(args.om_path) if args.om_path is not None else Path(artifacts['om'])
    if args.meta_json is not None:
        meta_json = Path(args.meta_json)
    else:
        meta_json = Path(artifacts.get('om_meta_json', artifacts['meta_json']))

    rows: list[dict] = []
    for host_io_mode in ['legacy', 'buffer_reuse']:
        bench_json = layout['benchmarks'] / f'bench_{host_io_mode}.json'
        cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'bench_patch_backend.py'),
            '--task',
            str(args.task),
            '--backend',
            'om',
            '--om_path',
            str(om_path),
            '--meta_json',
            str(meta_json),
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
            '--om_execution_mode',
            'sync',
            '--om_host_io_mode',
            host_io_mode,
            '--om_output_mode',
            'both',
        ]
        proc = _run(cmd)
        (layout['logs'] / f'bench_{host_io_mode}.stdout.txt').write_text(proc.stdout, encoding='utf-8')
        (layout['logs'] / f'bench_{host_io_mode}.stderr.txt').write_text(proc.stderr, encoding='utf-8')
        row = {'host_io_mode': host_io_mode, 'status': 'ok' if proc.returncode == 0 and bench_json.exists() else 'failed'}
        if bench_json.exists():
            row.update(load_json(bench_json))
        rows.append(row)

    parity = _compare_runtime_modes(
        args.task,
        om_path,
        meta_json,
        batch_size=args.batch_size,
        seed=args.seed,
        device_id=args.device_id,
    )
    df = pd.DataFrame(rows)
    summary_csv = layout['analysis'] / 'acl_runtime_summary.csv'
    df.to_csv(summary_csv, index=False)
    speedup = None
    if not df.empty and {'host_io_mode', 'patches_per_sec'}.issubset(df.columns):
        speed_map = dict(zip(df.host_io_mode, df.patches_per_sec))
        if speed_map.get('legacy') and speed_map.get('buffer_reuse'):
            speedup = float(speed_map['buffer_reuse'] / speed_map['legacy'])

    summary = {
        'step': 'step02_acl_runtime',
        'manifest': manifest,
        'om_path': str(om_path),
        'meta_json': str(meta_json),
        'summary_csv': str(summary_csv),
        'parity': parity,
        'speedup_buffer_reuse_vs_legacy': speedup,
        'rows': rows,
    }
    save_json(summary, layout['reports'] / 'step02_summary.json')
    write_markdown_report(
        layout['reports'] / '02_acl_runtime报告.md',
        title='Step02 ACL Runtime 报告',
        summary_lines=[
            f'- OM：`{om_path}`',
            f'- 汇总 CSV：`{summary_csv}`',
            f'- buffer_reuse 相对 legacy 速度倍率：`{speedup}`' if speedup is not None else '- buffer_reuse 相对 legacy 速度倍率：`暂无`',
        ],
        sections=[
            (
                '阶段说明',
                [
                    '- 本阶段不改 OM 模型文件，只优化 Python ACL 包装层的数据搬运路径。',
                    '- 对比 legacy `arr.tobytes()/ptr_to_bytes()` 路径 与 buffer_reuse 复用 host buffer 路径。',
                ],
            ),
            (
                '数值一致性',
                [
                    f"- feature_max_abs_diff: `{parity['feature_max_abs_diff']}`",
                    f"- logit_max_abs_diff: `{parity['logit_max_abs_diff']}`",
                ],
            ),
        ],
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
