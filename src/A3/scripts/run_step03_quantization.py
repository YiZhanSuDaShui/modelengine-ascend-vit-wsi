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

    layout = ensure_step_layout('step03_quant')
    manifest = build_step_manifest('step03_quant', task=args.task)
    defaults = get_task_defaults(args.task)
    artifacts = get_default_backend_artifacts(args.task)

    ptq_cfg = {
        'task': args.task,
        'method': 'ptq_int8',
        'teacher_ckpt': str(defaults['ckpt']),
        'onnx_path': str(artifacts['onnx']),
        'calibration_batches': 32,
        'batch_size': 64,
        'preferred_toolchain': 'amct_onnx',
        'fallback_note': '当前环境未检测到 amct_onnx/amct_pytorch 时，仅落盘配置与命令模板，不能诚实标记为已完成量化产物。',
    }
    qat_cfg = {
        'task': args.task,
        'method': 'qat_int8',
        'teacher_ckpt': str(defaults['ckpt']),
        'epochs': 5,
        'batch_size': 64,
        'preferred_toolchain': 'amct_pytorch',
        'fallback_note': 'ViT 主线 QAT 需要专用量化工具链；当前仓库先保留配置与训练入口占位。 ',
    }
    save_json(ptq_cfg, layout['configs'] / 'ptq_plan.json')
    save_json(qat_cfg, layout['configs'] / 'qat_plan.json')
    command_templates = {
        'ptq_prepare': f'python src/A3/scripts/run_step03_quantization.py --task {args.task}',
        'ptq_compile_after_tool_ready': 'python src/A3/scripts/compile_patch_encoder_om.py --task wsi --precision_mode origin',
        'qat_train_placeholder': 'python src/A3/scripts/train_wsi_tile_stage1p5.py --manifest_csv <manifest> --out_dir <out>',
    }
    save_json(command_templates, layout['configs'] / 'command_templates.json')
    summary = {
        'step': 'step03_quant',
        'manifest': manifest,
        'ptq_plan': ptq_cfg,
        'qat_plan': qat_cfg,
        'command_templates': command_templates,
        'status': 'blocked_by_missing_amct' if (not manifest['env_facts']['amct_onnx'] and not manifest['env_facts']['amct_pytorch']) else 'toolchain_available',
    }
    save_json(summary, layout['reports'] / 'step03_summary.json')
    write_markdown_report(
        layout['reports'] / '03_quant报告.md',
        title='Step03 Quantization 报告',
        summary_lines=[
            f"- 量化工具可用性：`amct_onnx={manifest['env_facts']['amct_onnx']}`，`amct_pytorch={manifest['env_facts']['amct_pytorch']}`",
            f"- 当前状态：`{summary['status']}`",
            f"- PTQ 配置：`{layout['configs'] / 'ptq_plan.json'}`",
            f"- QAT 配置：`{layout['configs'] / 'qat_plan.json'}`",
        ],
        sections=[
            (
                '结论',
                [
                    '- 当前环境缺少正式 Ascend INT8 量化工具链时，不能诚实产出真正可提交的 INT8/OM 结果。',
                    '- 本阶段先把 PTQ/QAT 的配置、命令模板、产物目录和阻塞原因固定下来，后续只需在工具链就绪环境执行。 ',
                ],
            ),
        ],
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
