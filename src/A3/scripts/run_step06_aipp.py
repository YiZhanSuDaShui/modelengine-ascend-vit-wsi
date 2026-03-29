from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.runtime.submission_defaults import get_default_backend_artifacts  # noqa: E402
from bach_mil.utils.io import save_json  # noqa: E402
from bach_mil.utils.optimization_steps import build_step_manifest, ensure_step_layout, write_markdown_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='wsi', choices=['wsi', 'photos'])
    args = parser.parse_args()

    layout = ensure_step_layout('step06_aipp')
    manifest = build_step_manifest('step06_aipp', task=args.task)
    artifacts = get_default_backend_artifacts(args.task)
    aipp_cfg = layout['configs'] / 'aipp_static.cfg'
    aipp_cmd = (
        f'python src/A3/scripts/build_aipp_config.py --out_cfg {aipp_cfg} && '
        'python src/A3/scripts/compile_patch_encoder_om.py '
        f'--task {args.task} --onnx_path {artifacts["onnx"]} --meta_json {artifacts["meta_json"]} '
        f'--out_om {layout["models"] / f"{args.task}_aipp.om"} '
        f'--report_json {layout["reports"] / "aipp_compile_summary.json"} '
        f'--insert_op_conf {aipp_cfg} --precision_mode_v2 origin'
    )
    (layout['configs'] / 'aipp_command.sh').write_text(aipp_cmd + '\n', encoding='utf-8')
    summary = {
        'step': 'step06_aipp',
        'manifest': manifest,
        'aipp_cfg': str(aipp_cfg),
        'command': aipp_cmd,
        'status': 'needs_input_contract_change',
        'note': '当前 ONNX 输入是归一化后的 float32 NCHW tensor。若要真正发挥 AIPP，需要把输入契约改成原始图像/uint8，再重新导出并编译。当前先落配置和编译命令模板。',
    }
    save_json(summary, layout['reports'] / 'step06_summary.json')
    write_markdown_report(
        layout['reports'] / '06_aipp报告.md',
        title='Step06 AIPP 报告',
        summary_lines=[
            f"- AIPP 配置模板：`{aipp_cfg}`",
            f"- 命令模板：`{layout['configs'] / 'aipp_command.sh'}`",
            f"- 当前状态：`{summary['status']}`",
        ],
        sections=[
            (
                '说明',
                [
                    '- 当前主线 ONNX 输入不是原始图像，而是 host 侧已经完成 resize + normalize 的 float32 tensor。',
                    '- 因此 AIPP 真正下沉前，需要先调整导出与推理入口的输入契约；本阶段先保留 cfg 与 ATC 编译命令模板。 ',
                ],
            ),
        ],
    )
    print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
