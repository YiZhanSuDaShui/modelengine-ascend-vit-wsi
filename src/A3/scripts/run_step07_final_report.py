from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import load_json, save_json  # noqa: E402
from bach_mil.utils.optimization_steps import DEFAULT_OPTIMIZATION_ROOT, STEP_SPECS, ensure_step_layout, write_markdown_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi')
    args = parser.parse_args()

    layout = ensure_step_layout('step07_final')
    collected = {}
    for step_key, step_name in STEP_SPECS.items():
        if step_key == 'step07_final':
            continue
        step_root = DEFAULT_OPTIMIZATION_ROOT / step_name
        summary_candidates = sorted(step_root.glob('reports/step*_summary.json'))
        if summary_candidates:
            collected[step_key] = load_json(summary_candidates[0])

    final_summary = {
        'task': args.task,
        'optimization_root': str(DEFAULT_OPTIMIZATION_ROOT),
        'collected_steps': collected,
    }
    save_json(final_summary, layout['reports'] / 'step07_final_summary.json')
    write_markdown_report(
        layout['reports'] / '07_最终汇总报告.md',
        title='Step07 最终汇总报告',
        summary_lines=[
            f'- 优化根目录：`{DEFAULT_OPTIMIZATION_ROOT}`',
            f'- 已汇总步骤数：`{len(collected)}`',
        ],
        sections=[
            (
                '步骤清单',
                [f"- {step_key}: `{STEP_SPECS[step_key]}`" for step_key in collected],
            ),
        ],
    )
    print(final_summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
