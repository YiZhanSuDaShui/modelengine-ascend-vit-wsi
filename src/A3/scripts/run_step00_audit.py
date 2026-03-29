from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.submission_defaults import get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import save_json  # noqa: E402
from bach_mil.utils.optimization_steps import build_step_manifest, ensure_step_layout, write_markdown_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    args = parser.parse_args()

    layout = ensure_step_layout('step00_audit')
    manifest = build_step_manifest('step00_audit', task=args.task)
    defaults = get_task_defaults(args.task)
    artifacts = get_default_backend_artifacts(args.task)
    summary = {
        'task': args.task,
        'manifest': manifest,
        'defaults': {k: str(v) if isinstance(v, Path) else v for k, v in defaults.items()},
        'artifacts': {k: str(v) if isinstance(v, Path) else v for k, v in artifacts.items()},
    }
    save_json(summary, layout['reports'] / 'step00_summary.json')
    write_markdown_report(
        layout['reports'] / '00_现状审计报告.md',
        title='Step00 现状审计报告',
        summary_lines=[
            f"- task：`{args.task}`",
            f"- ckpt：`{defaults['ckpt']}`",
            f"- onnx：`{artifacts['onnx']}`",
            f"- om：`{artifacts['om']}`",
        ],
        sections=[
            (
                '环境',
                [f"- {k}: `{v}`" for k, v in manifest['env_facts'].items()],
            ),
        ],
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
