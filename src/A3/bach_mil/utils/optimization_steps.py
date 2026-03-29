from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Iterable

from ..runtime.submission_defaults import DEFAULT_REPORT_ROOT, get_default_backend_artifacts, get_task_defaults
from .io import ensure_dir, save_json

DEFAULT_OPTIMIZATION_ROOT = DEFAULT_REPORT_ROOT / 'optimization_stepwise'

STEP_SPECS = {
    'step00_audit': '00_现状审计与环境检查',
    'step01_mixed_keepdtype': '01_mixed_float16_keep_dtype_modify_mixlist',
    'step02_acl_runtime': '02_acl_python_runtime_optimized',
    'step03_quant': '03_quant_ptq_qat',
    'step04_distill': '04_teacher_student_distill',
    'step05_prune': '05_structured_prune',
    'step06_aipp': '06_aipp_sink',
    'step07_final': '07_final_self_check_and_report',
}


def ensure_step_layout(step_key: str) -> dict[str, Path]:
    if step_key not in STEP_SPECS:
        raise KeyError(f'unknown step_key={step_key}')
    root = ensure_dir(DEFAULT_OPTIMIZATION_ROOT / STEP_SPECS[step_key])
    layout = {
        'root': root,
        'configs': ensure_dir(root / 'configs'),
        'artifacts': ensure_dir(root / 'artifacts'),
        'models': ensure_dir(root / 'models'),
        'benchmarks': ensure_dir(root / 'benchmarks'),
        'analysis': ensure_dir(root / 'analysis'),
        'reports': ensure_dir(root / 'reports'),
        'logs': ensure_dir(root / 'logs'),
    }
    return layout


def ensure_all_step_layouts() -> dict[str, dict[str, Path]]:
    return {step_key: ensure_step_layout(step_key) for step_key in STEP_SPECS}


def collect_env_facts() -> dict:
    mods = ['acl', 'torch_npu', 'onnxruntime', 'mindspore', 'amct_pytorch', 'amct_onnx']
    facts = {name: bool(importlib.util.find_spec(name)) for name in mods}
    for binary in ['atc', 'benchmark', 'msame', 'atb']:
        facts[binary] = bool(shutil.which(binary))
    return facts


def build_step_manifest(step_key: str, *, task: str = 'wsi') -> dict:
    defaults = get_task_defaults(task)
    artifacts = get_default_backend_artifacts(task)
    layout = ensure_step_layout(step_key)
    manifest = {
        'step_key': step_key,
        'step_name': STEP_SPECS[step_key],
        'task': task,
        'layout': {k: str(v) for k, v in layout.items()},
        'defaults': {k: str(v) if isinstance(v, Path) else v for k, v in defaults.items()},
        'artifacts': {k: str(v) if isinstance(v, Path) else v for k, v in artifacts.items()},
        'env_facts': collect_env_facts(),
    }
    save_json(manifest, layout['root'] / 'step_manifest.json')
    return manifest


def write_markdown_report(
    path: str | Path,
    *,
    title: str,
    summary_lines: Iterable[str],
    sections: Iterable[tuple[str, Iterable[str]]] | None = None,
) -> None:
    lines = [f'# {title}', '']
    for line in summary_lines:
        lines.append(str(line))
    if sections is not None:
        for section_title, body in sections:
            lines.extend(['', f'## {section_title}'])
            for line in body:
                lines.append(str(line))
    Path(path).write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')
