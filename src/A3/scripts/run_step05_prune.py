from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.submission_defaults import get_task_defaults  # noqa: E402
from bach_mil.utils.io import save_json  # noqa: E402
from bach_mil.utils.optimization_steps import build_step_manifest, ensure_step_layout, write_markdown_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi'])
    args = parser.parse_args()

    layout = ensure_step_layout('step05_prune')
    manifest = build_step_manifest('step05_prune', task=args.task)
    defaults = get_task_defaults(args.task)
    plan = {
        'init_ckpt': str(defaults['ckpt']),
        'prune_ratio': 0.1,
        'epochs': 5,
        'target_layers': ['patch_embed.proj', 'attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
        'exclude_default': ['LayerNorm', 'thresholds', '聚合层'],
    }
    command = (
        'python src/A3/scripts/prune_wsi_tile_stage1p5.py '
        '--manifest_csv data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv '
        f'--init_ckpt {defaults["ckpt"]} '
        f'--out_dir {layout["models"] / "prune_ratio_0p10_v1"} '
        '--split_mode random_tile --val_ratio 0.1 '
        '--model_name vit_large_patch16_224 --backbone_pool token --backbone_init_values 1.0 '
        '--prune_ratio 0.1 --epochs 5 --batch_size 64'
    )
    save_json(plan, layout['configs'] / 'prune_plan.json')
    (layout['configs'] / 'prune_command.sh').write_text(command + '\n', encoding='utf-8')
    summary = {
        'step': 'step05_prune',
        'manifest': manifest,
        'plan': plan,
        'command': command,
        'train_script': str(ROOT / 'scripts' / 'prune_wsi_tile_stage1p5.py'),
    }
    save_json(summary, layout['reports'] / 'step05_summary.json')
    write_markdown_report(
        layout['reports'] / '05_prune报告.md',
        title='Step05 Structured Prune 报告',
        summary_lines=[
            f"- init_ckpt：`{defaults['ckpt']}`",
            f"- 剪枝脚本：`{ROOT / 'scripts' / 'prune_wsi_tile_stage1p5.py'}`",
            f"- 命令模板：`{layout['configs'] / 'prune_command.sh'}`",
        ],
        sections=[
            (
                '说明',
                [
                    '- 当前脚本采用结构化剪枝，优先剪 `qkv/proj/fc1/fc2/patch_embed.proj` 的输出通道。',
                    '- LayerNorm、阈值聚合、slide 后处理不在默认剪枝范围内，避免直接破坏数值稳定性。 ',
                ],
            ),
        ],
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
