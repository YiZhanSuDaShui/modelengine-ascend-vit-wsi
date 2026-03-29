from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import stat


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _copy_file(src: Path, dst: Path, *, optional: bool = False) -> bool:
    if not src.exists():
        if optional:
            return False
        raise FileNotFoundError(f'缺少文件：{src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_dir(src: Path, dst: Path, *, optional: bool = False) -> bool:
    if not src.exists():
        if optional:
            return False
        raise FileNotFoundError(f'缺少目录：{src}')
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def _write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest_root', type=str, required=True)
    parser.add_argument('--bundle_name', type=str, default='runtime_bundle_20260327')
    args = parser.parse_args()

    dest_root = Path(args.dest_root)
    bundle_root = dest_root / args.bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []

    docs = [
        'README.md',
        'README-process.md',
        'analyse.md',
        '赛题三具体评测要求（必看）.md',
    ]
    for rel in docs:
        if _copy_file(PROJECT_ROOT / rel, bundle_root / rel, optional=True):
            copied.append({'type': 'file', 'path': rel})

    if _copy_dir(PROJECT_ROOT / 'src' / 'A3', bundle_root / 'src' / 'A3'):
        copied.append({'type': 'dir', 'path': 'src/A3'})
    if _copy_dir(PROJECT_ROOT / '_vendor', bundle_root / '_vendor', optional=True):
        copied.append({'type': 'dir', 'path': '_vendor'})

    artifacts = [
        ('logs/A3_output/C_phase/stage1p5_uni_large_L1_s448_mt40_random_v1/best.pt', False),
        ('logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json', False),
        ('logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64.onnx', True),
        ('logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64.meta.json', True),
        ('logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.om', True),
        ('logs/A3_output/submission_closure/offline_models/wsi/wsi_patch_encoder_bs64_origin.meta.json', True),
        ('logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.om', False),
        ('logs/A3_output/submission_closure/optimization_stepwise/01_mixed_float16_keep_dtype_modify_mixlist/artifacts/refined_attn_score_path_norm1/wsi_attn_score_path_norm1.meta.json', False),
        ('BRACS/subset20_test_balanced/meta/bracs20_roi_multilabel_eval.csv', True),
        ('BRACS/subset20_test_balanced/meta/bracs20_roi_multilabel_eval_summary.json', True),
        ('BRACS/subset20_test_balanced/meta/bracs20_roi_multilabel_eval_report.md', True),
    ]
    for rel, optional in artifacts:
        if _copy_file(PROJECT_ROOT / rel, bundle_root / rel, optional=optional):
            copied.append({'type': 'file', 'path': rel})

    experimental_dir = bundle_root / 'logs' / 'A3_output' / 'submission_closure' / 'external_eval'
    experimental_dir.mkdir(parents=True, exist_ok=True)
    tuned_before = PROJECT_ROOT / 'BRACS/subset20_test_balanced/meta/bracs20_roi_multilabel_eval_summary.json'
    if tuned_before.exists():
        data = json.loads(tuned_before.read_text(encoding='utf-8'))
        if 'before' in data and 'tuned_thresholds' in data['before']:
            out = experimental_dir / 'bracs20_tuned_thresholds_before.json'
            out.write_text(json.dumps(data['before']['tuned_thresholds'], ensure_ascii=False, indent=2), encoding='utf-8')
            copied.append({'type': 'file', 'path': str(out.relative_to(bundle_root))})
        if 'after' in data and 'tuned_thresholds' in data['after']:
            out = experimental_dir / 'bracs20_tuned_thresholds_after.json'
            out.write_text(json.dumps(data['after']['tuned_thresholds'], ensure_ascii=False, indent=2), encoding='utf-8')
            copied.append({'type': 'file', 'path': str(out.relative_to(bundle_root))})

    for rel in ['input', 'output_before', 'output_after']:
        (bundle_root / rel).mkdir(parents=True, exist_ok=True)

    run_common = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-$ROOT/input}"
OUT_DIR="${2:-%OUT_DIR%}"
THR_JSON="${3:-%THR_JSON%}"
python3 "$ROOT/src/A3/scripts/run_submission_infer.py" \\
  --task wsi \\
  --variant %VARIANT% \\
  --backend %BACKEND% \\
  --input_dir "$INPUT_DIR" \\
  --out_dir "$OUT_DIR" \\
  --thresholds_json "$THR_JSON" \\
  --save_features
"""
    default_thr = '$ROOT/logs/A3_output/E_phase/tileagg_thresholds_L1_s448_uniStage1p5_v1.json'
    _write_text(
        bundle_root / 'run_before.sh',
        run_common.replace('%OUT_DIR%', '$ROOT/output_before').replace('%THR_JSON%', default_thr).replace('%VARIANT%', 'before').replace('%BACKEND%', 'pytorch'),
        executable=True,
    )
    _write_text(
        bundle_root / 'run_after.sh',
        run_common.replace('%OUT_DIR%', '$ROOT/output_after').replace('%THR_JSON%', default_thr).replace('%VARIANT%', 'after').replace('%BACKEND%', 'om'),
        executable=True,
    )
    _write_text(
        bundle_root / 'run_after_bracs20_tuned_experimental.sh',
        run_common.replace('%OUT_DIR%', '$ROOT/output_after').replace('%THR_JSON%', '$ROOT/logs/A3_output/submission_closure/external_eval/bracs20_tuned_thresholds_after.json').replace('%VARIANT%', 'after').replace('%BACKEND%', 'om'),
        executable=True,
    )
    _write_text(
        bundle_root / 'install_requirements.sh',
        """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m pip install -r "$ROOT/src/A3/requirements.txt"
""",
        executable=True,
    )

    _write_text(
        bundle_root / 'README_DEPLOY.md',
        """# 迁移包使用说明

## 1. 依赖安装

```bash
cd <本目录>
./install_requirements.sh
```

说明：

- `before` 还需要目标机已正确安装 `torch_npu`
- `after` 还需要目标机已正确安装 `acl`
- 读 WSI 时优先走 `openslide`，缺失时会尝试 `_vendor + tiffslide` 回退

## 2. 输入目录

把待测 WSI 放到：

```text
input/
```

支持：

- `.svs`
- `.tif`
- `.tiff`

## 3. 一键运行

### before

```bash
./run_before.sh
```

### after

```bash
./run_after.sh
```

### after 外部实验阈值版本

```bash
./run_after_bracs20_tuned_experimental.sh
```

## 4. 输出目录

- `output_before/`
- `output_after/`

WSI 输出结构固定为：

```text
features/
manifests/
predictions/slide_predictions.csv
reports/run_summary.json
```
""",
    )

    manifest = {
        'bundle_root': str(bundle_root),
        'copied_items': copied,
    }
    _write_text(bundle_root / 'bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    print(f'bundle ready -> {bundle_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
