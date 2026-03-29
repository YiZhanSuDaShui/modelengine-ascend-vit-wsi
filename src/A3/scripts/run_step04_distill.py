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

    layout = ensure_step_layout('step04_distill')
    manifest = build_step_manifest('step04_distill', task=args.task)
    defaults = get_task_defaults(args.task)
    plan = {
        'teacher_ckpt': str(defaults['ckpt']),
        'student_model_name': 'vit_base_patch16_224',
        'student_backbone_pool': 'token',
        'student_backbone_init_values': 1.0,
        'temperature': 2.0,
        'kd_alpha': 0.5,
        'epochs': 10,
        'batch_size': 64,
    }
    command = (
        'python src/A3/scripts/train_wsi_tile_stage1p5_distill.py '
        f'--manifest_csv data/BACH/derived/split/wsi_train_tiles_L1_s448_mt40.csv '
        f'--teacher_ckpt {defaults["ckpt"]} '
        f'--out_dir {layout["models"] / "distill_student_v1"} '
        '--split_mode random_tile --val_ratio 0.1 '
        '--teacher_model_name vit_large_patch16_224 --student_model_name vit_base_patch16_224 '
        '--teacher_backbone_pool token --student_backbone_pool token '
        '--teacher_backbone_init_values 1.0 --student_backbone_init_values 1.0 '
        '--epochs 10 --batch_size 64 --save_val_predictions'
    )
    save_json(plan, layout['configs'] / 'distill_plan.json')
    (layout['configs'] / 'distill_command.sh').write_text(command + '\n', encoding='utf-8')
    summary = {
        'step': 'step04_distill',
        'manifest': manifest,
        'plan': plan,
        'command': command,
        'train_script': str(ROOT / 'scripts' / 'train_wsi_tile_stage1p5_distill.py'),
    }
    save_json(summary, layout['reports'] / 'step04_summary.json')
    write_markdown_report(
        layout['reports'] / '04_distill报告.md',
        title='Step04 Distillation 报告',
        summary_lines=[
            f"- teacher_ckpt：`{defaults['ckpt']}`",
            f"- student 目标：`vit_base_patch16_224`",
            f"- 训练脚本：`{ROOT / 'scripts' / 'train_wsi_tile_stage1p5_distill.py'}`",
            f"- 命令模板：`{layout['configs'] / 'distill_command.sh'}`",
        ],
        sections=[
            (
                '说明',
                [
                    '- 本阶段已经补齐可运行蒸馏训练脚本，采用 CE + KL(KD) 组合损失。',
                    '- 目标是把 teacher 的判别能力迁移到更轻的 student，为后续 ONNX/OM/剪枝提供更轻的候选模型。 ',
                ],
            ),
        ],
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
