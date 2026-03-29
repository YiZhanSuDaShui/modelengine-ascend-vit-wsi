from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
REPO_ROOT = ROOT.parents[1]

from bach_mil.runtime.submission_defaults import DEFAULT_REPORT_ROOT, get_acceleration_rounds, get_task_defaults  # noqa: E402
from bach_mil.data.wsi_manifest import list_wsi_paths  # noqa: E402
from bach_mil.utils.io import ensure_dir, load_json, save_json  # noqa: E402


def _task_patterns(task: str) -> tuple[str, ...]:
    if task == 'wsi':
        return ('*.svs', '*.tif', '*.tiff', '*.SVS', '*.TIF', '*.TIFF')
    return ('*.tif', '*.png', '*.jpg', '*.jpeg')


def _list_inputs(task: str, input_dir: Path) -> list[Path]:
    if task == 'wsi':
        return list_wsi_paths(input_dir)
    paths: list[Path] = []
    for pattern in _task_patterns(task):
        paths.extend(sorted(input_dir.glob(pattern)))
    return sorted({p.resolve(): p for p in paths}.values(), key=lambda p: p.name)


def _prepare_smoke_input(task: str, src_input_dir: Path, smoke_dir: Path, max_items: int) -> Path:
    ensure_dir(smoke_dir)
    items = _list_inputs(task, src_input_dir)[: max(1, int(max_items))]
    if not items:
        raise FileNotFoundError(f'no inputs found in {src_input_dir}')
    for src in items:
        dst = smoke_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return smoke_dir


def _env_facts() -> dict:
    facts = {
        'mindspore_installed': bool(importlib.util.find_spec('mindspore')),
        'acl_installed': bool(importlib.util.find_spec('acl')),
        'onnxruntime_installed': bool(importlib.util.find_spec('onnxruntime')),
        'torch_npu_installed': bool(importlib.util.find_spec('torch_npu')),
        'atc_exists': bool(shutil.which('atc')),
        'atb_exists': bool(shutil.which('atb')),
        'msame_exists': bool(shutil.which('msame')),
        'benchmark_exists': bool(shutil.which('benchmark')),
    }
    try:
        import onnxruntime as ort  # type: ignore

        facts['onnxruntime_providers'] = list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover
        facts['onnxruntime_providers'] = []
        facts['onnxruntime_provider_error'] = repr(exc)
    return facts


def _load_proxy_metrics(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == '.csv':
        df = pd.read_csv(p)
        if 'index' in df.columns:
            sub = df[df['index'] == 'mean']
            if not sub.empty:
                row = sub.iloc[0].to_dict()
                row['source'] = str(p)
                return row
        row = df.iloc[0].to_dict()
        row['source'] = str(p)
        return row
    if p.suffix == '.json':
        obj = load_json(p)
        if isinstance(obj, dict) and 'mean_std' in obj and obj['mean_std']:
            row = dict(obj['mean_std'][0])
            row['source'] = str(p)
            return row
        return {'source': str(p), 'payload': obj}
    return {'source': str(p)}


def _speed_fields(task: str) -> tuple[str, str]:
    if task == 'wsi':
        return 'wsi_per_sec', 'tiles_per_sec'
    return 'images_per_sec', 'crops_per_sec'


def _time_drop_pct(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return float((before - after) / before * 100.0)


def _speedup(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or abs(after) < 1e-12:
        return None
    return float(before / after)


def _summary_to_row(task: str, round_cfg: dict, summary: dict | None, before_summary: dict | None, proxy_metrics: dict | None) -> dict:
    per_item_key, per_patch_key = _speed_fields(task)
    before_elapsed = before_summary.get('total_elapsed_sec') if before_summary else None
    after_elapsed = summary.get('total_elapsed_sec') if summary else None
    per_patch_value = None
    runtime_profile = None
    if summary:
        per_patch_value = summary.get(per_patch_key)
        runtime_profile = summary.get('runtime_profile')
        if per_patch_value is None and after_elapsed:
            if task == 'wsi' and summary.get('total_tiles') is not None:
                per_patch_value = float(summary['total_tiles'] / max(after_elapsed, 1e-12))
            if task == 'photos' and summary.get('total_crops') is not None:
                per_patch_value = float(summary['total_crops'] / max(after_elapsed, 1e-12))
        if runtime_profile is None:
            runtime_profile = 'before_baseline' if round_cfg['variant'] == 'before' else 'after_optimized'
    row = {
        'round_name': round_cfg['name'],
        'backend': round_cfg['backend'],
        'variant': round_cfg['variant'],
        'execution_scope': round_cfg['execution_scope'],
        'description': round_cfg['description'],
        'methods_enabled': ' / '.join(round_cfg['methods_enabled']),
        'status': summary.get('status', 'ok') if summary else 'missing',
        'backend_actual': summary.get('backend_actual') if summary else None,
        'runtime_profile': runtime_profile,
        'total_elapsed_sec': after_elapsed,
        per_item_key: summary.get(per_item_key) if summary else None,
        per_patch_key: per_patch_value,
        'speedup_vs_before': _speedup(before_elapsed, after_elapsed),
        'time_drop_pct_vs_before': _time_drop_pct(before_elapsed, after_elapsed),
        'proxy_exact_match': proxy_metrics.get('exact_match') if proxy_metrics else None,
        'proxy_acc': proxy_metrics.get('acc') if proxy_metrics else None,
        'proxy_macro_f1': proxy_metrics.get('macro_f1') if proxy_metrics else None,
        'run_summary_path': summary.get('_path') if summary else None,
        'proxy_metrics_path': proxy_metrics.get('source') if proxy_metrics else None,
    }
    return row


def _write_round_manifest(round_dir: Path, payload: dict) -> None:
    save_json(payload, round_dir / 'round_manifest.json')
    md = [
        f"# {payload['round_name']}",
        '',
        f"- 任务：`{payload['task']}`",
        f"- 轮次说明：`{payload['description']}`",
        f"- 后端：`{payload['backend']}`",
        f"- 执行范围：`{payload['execution_scope']}`",
        f"- 方法：`{' / '.join(payload['methods_enabled'])}`",
        '',
        "## 模型引用",
    ]
    for k, v in payload.get('model_refs', {}).items():
        md.append(f"- {k}: `{v}`")
    md.extend([
        '',
        '## 运行配置',
    ])
    for k, v in payload.get('overrides', {}).items():
        md.append(f"- {k}: `{v}`")
    if payload.get('proxy_metrics_ref'):
        md.extend(['', '## 代理评测引用', f"- `{payload['proxy_metrics_ref']}`"])
    (round_dir / 'round_manifest.md').write_text('\n'.join(md).strip() + '\n', encoding='utf-8')


def _build_command(round_cfg: dict, input_dir: Path, out_dir: Path, save_features: bool) -> list[str]:
    script = ROOT / 'scripts' / 'run_submission_infer.py'
    cmd = [
        sys.executable,
        str(script),
        '--task',
        str(round_cfg['task']),
        '--variant',
        str(round_cfg['variant']),
        '--backend',
        str(round_cfg['backend']),
        '--input_dir',
        str(input_dir),
        '--out_dir',
        str(out_dir),
    ]
    if save_features:
        cmd.append('--save_features')
    for k, v in round_cfg.get('model_refs', {}).items():
        if k == 'thresholds_json':
            cmd.extend(['--thresholds_json', str(v)])
        elif k == 'ckpt':
            cmd.extend(['--ckpt', str(v)])
        elif k == 'onnx':
            cmd.extend(['--onnx_path', str(v)])
        elif k == 'om':
            cmd.extend(['--om_path', str(v)])
        elif k == 'meta_json':
            cmd.extend(['--meta_json', str(v)])
        elif k == 'om_meta_json':
            cmd.extend(['--meta_json', str(v)])
    for k, v in round_cfg.get('overrides', {}).items():
        if v is None:
            continue
        cmd.extend([f'--{k}', str(v)])
    return cmd


def _load_existing_summary(round_cfg: dict, round_dir: Path) -> dict | None:
    existing_run_dir = round_cfg.get('existing_run_dir')
    if not existing_run_dir:
        return None
    existing_run_dir = Path(existing_run_dir)
    summary_path = existing_run_dir / 'reports' / 'run_summary.json'
    if not summary_path.exists():
        return None
    link_path = round_dir / 'run_reuse'
    if not link_path.exists():
        try:
            os.symlink(existing_run_dir, link_path)
        except OSError:
            pass
    summary = load_json(summary_path)
    summary['_path'] = str(summary_path)
    summary['_reused_from'] = str(existing_run_dir)
    return summary


def _run_round(round_cfg: dict, input_dir: Path, round_dir: Path, save_features: bool, skip_existing: bool) -> dict | None:
    run_dir = ensure_dir(round_dir / 'run')
    summary_path = run_dir / 'reports' / 'run_summary.json'
    if skip_existing and summary_path.exists():
        summary = load_json(summary_path)
        summary['_path'] = str(summary_path)
        return summary
    existing_summary = _load_existing_summary(round_cfg, round_dir)
    if existing_summary is not None:
        return existing_summary

    actual_input_dir = input_dir
    if round_cfg.get('execution_scope') == 'smoke':
        actual_input_dir = _prepare_smoke_input(round_cfg['task'], input_dir, round_dir / 'smoke_input', int(round_cfg.get('smoke_items', 1)))

    cmd = _build_command(round_cfg, actual_input_dir, run_dir, save_features=save_features)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    (round_dir / 'command.txt').write_text(' '.join(cmd) + '\n', encoding='utf-8')
    (round_dir / 'stdout.log').write_text(proc.stdout or '', encoding='utf-8')
    (round_dir / 'stderr.log').write_text(proc.stderr or '', encoding='utf-8')
    if proc.returncode != 0:
        save_json(
            {
                'round_name': round_cfg['name'],
                'status': 'failed',
                'returncode': int(proc.returncode),
                'cmd': cmd,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
            },
            round_dir / 'run_failed.json',
        )
        return None
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    summary['_path'] = str(summary_path)
    return summary


def _write_summary_reports(task: str, out_root: Path, env: dict, rows: list[dict]) -> None:
    ensure_dir(out_root)
    df = pd.DataFrame(rows)
    csv_path = out_root / 'rounds_summary.csv'
    json_path = out_root / 'rounds_summary.json'
    md_path = out_root / 'rounds_summary.md'
    df.to_csv(csv_path, index=False)
    save_json({'task': task, 'environment': env, 'rounds': rows}, json_path)

    per_item_key, per_patch_key = _speed_fields(task)
    lines = [
        f'# {task.upper()} 加速轮次汇总',
        '',
        '## 环境事实',
        f"- MindSpore 已安装：`{env['mindspore_installed']}`",
        f"- ACL 已安装：`{env['acl_installed']}`",
        f"- ONNXRuntime 已安装：`{env['onnxruntime_installed']}`",
        f"- torch_npu 已安装：`{env['torch_npu_installed']}`",
        f"- ONNXRuntime providers：`{env.get('onnxruntime_providers', [])}`",
        f"- atc 可用：`{env['atc_exists']}`",
        f"- atb 可用：`{env['atb_exists']}`",
        f"- msame 可用：`{env['msame_exists']}`",
        '',
        '## 轮次结果',
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['round_name']}",
                f"- backend: `{row['backend_actual'] or row['backend']}`",
                f"- 说明: `{row['description']}`",
                f"- total_elapsed_sec: `{row['total_elapsed_sec']}`",
                f"- {per_item_key}: `{row[per_item_key]}`",
                f"- {per_patch_key}: `{row[per_patch_key]}`",
                f"- speedup_vs_before: `{row['speedup_vs_before']}`",
                f"- time_drop_pct_vs_before: `{row['time_drop_pct_vs_before']}`",
                f"- proxy_exact_match: `{row['proxy_exact_match']}`",
                f"- proxy_acc: `{row['proxy_acc']}`",
                f"- proxy_macro_f1: `{row['proxy_macro_f1']}`",
                '',
            ]
        )
    lines.extend(
        [
            '## 未形成正式可提交产物的加速项',
            '- MindSpore：当前环境未安装，不能诚实声明已完成。',
            '- ATB：当前环境未检测到 `atb` 可执行工具，不能诚实声明已完成。',
            '- ONNX NPU Provider：当前只有 `CPUExecutionProvider`，因此 ONNX 在本环境只能作为导出与功能验证链路，不是正式 Ascend 速度路线。',
            '- 量化/蒸馏/剪枝：属于会改变权重或训练链路的优化，本轮若没有新增可验证权重与精度报告，不能直接算作已完成提交产物。',
        ]
    )
    md_path.write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['wsi', 'photos'])
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--out_root', type=str, default=None)
    parser.add_argument('--round_names', type=str, nargs='*', default=None)
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--save_features', action='store_true')
    args = parser.parse_args()

    defaults = get_task_defaults(args.task)
    input_dir = Path(args.input_dir) if args.input_dir is not None else Path(defaults['official_test_input'])
    out_root = Path(args.out_root) if args.out_root is not None else DEFAULT_REPORT_ROOT / 'optimization_rounds' / args.task
    rounds = get_acceleration_rounds(args.task)
    if args.round_names:
        wanted = set(args.round_names)
        rounds = [x for x in rounds if x['name'] in wanted]

    env = _env_facts()
    before_summary = None
    rows: list[dict] = []

    for round_cfg in rounds:
        round_dir = ensure_dir(out_root / round_cfg['name'])
        manifest = {
            'round_name': round_cfg['name'],
            'task': round_cfg['task'],
            'variant': round_cfg['variant'],
            'backend': round_cfg['backend'],
            'execution_scope': round_cfg['execution_scope'],
            'description': round_cfg['description'],
            'methods_enabled': round_cfg['methods_enabled'],
            'overrides': round_cfg.get('overrides', {}),
            'model_refs': round_cfg.get('model_refs', {}),
            'proxy_metrics_ref': round_cfg.get('proxy_metrics_csv'),
            'environment': env,
        }
        _write_round_manifest(round_dir, manifest)
        summary = _run_round(round_cfg, input_dir, round_dir, save_features=args.save_features, skip_existing=args.skip_existing)
        if summary is not None and before_summary is None and round_cfg['name'].startswith('01_'):
            before_summary = summary
        proxy_metrics = _load_proxy_metrics(round_cfg.get('proxy_metrics_csv'))
        row = _summary_to_row(args.task, round_cfg, summary, before_summary, proxy_metrics)
        save_json({'round': manifest, 'summary': summary, 'proxy_metrics': proxy_metrics}, round_dir / 'round_result.json')
        rows.append(row)

    _write_summary_reports(args.task, out_root, env, rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
