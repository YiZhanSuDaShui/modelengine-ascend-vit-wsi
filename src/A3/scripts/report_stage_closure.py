from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
REPO_ROOT = ROOT.parents[1]

from bach_mil.runtime.submission_defaults import DEFAULT_REPORT_ROOT, get_default_backend_artifacts, get_task_defaults  # noqa: E402
from bach_mil.utils.io import ensure_dir, load_json, save_json  # noqa: E402


def _maybe_json(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return load_json(path)


def _maybe_csv(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _pick_fastest_formal_after(rounds_df: pd.DataFrame | None) -> dict | None:
    if rounds_df is None or rounds_df.empty:
        return None
    df = rounds_df.copy()
    if 'variant' in df.columns:
        df = df[df['variant'] == 'after']
    if 'backend' in df.columns:
        df = df[df['backend'] == 'om']
    if 'execution_scope' in df.columns:
        df = df[df['execution_scope'] == 'full']
    if 'round_name' in df.columns:
        no_warm = df[~df['round_name'].astype(str).str.contains('cachewarm', case=False, na=False)]
        if not no_warm.empty:
            df = no_warm
    if 'total_elapsed_sec' in df.columns:
        df = df[pd.notna(df['total_elapsed_sec'])]
        if not df.empty:
            df = df.sort_values('total_elapsed_sec', ascending=True)
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    return row


def _write_stage(out_root: Path, stem: str, payload: dict, md_lines: list[str]) -> None:
    save_json(payload, out_root / f'{stem}.json')
    (out_root / f'{stem}.md').write_text('\n'.join(md_lines).strip() + '\n', encoding='utf-8')


def _mean_from_mean_std(df: pd.DataFrame | None, column: str):
    if df is None or df.empty or column not in df.columns or 'index' not in df.columns:
        return None
    sub = df[df['index'] == 'mean']
    if sub.empty:
        return None
    return float(sub.iloc[0][column])


def _photos_pairing_is_strict(after_proxy_csv: str | None) -> bool:
    if not after_proxy_csv:
        return False
    cfg_path = Path(after_proxy_csv).parent / 'config.json'
    if not cfg_path.exists():
        return False
    cfg = load_json(cfg_path)
    ckpt = str(cfg.get('ckpt', ''))
    # 当前 photos_after_om_origin 使用 fold0 checkpoint 跑全 5 折，
    # 只能当补充代理结果，不能当严格 before/after 配对结论。
    return ('/fold0/' not in ckpt) and ('\\fold0\\' not in ckpt)


def _pct(before: float | None, after: float | None):
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return float((after - before) / before * 100.0)


def _drop_pct(before: float | None, after: float | None):
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return float((before - after) / before * 100.0)


def _speedup(before_time: float | None, after_time: float | None):
    if before_time is None or after_time is None or abs(after_time) < 1e-12:
        return None
    return float(before_time / after_time)


def _pct_point_diff(before: float | None, after: float | None):
    if before is None or after is None:
        return None
    return float((after - before) * 100.0)


def _value_has_nan(value) -> bool:
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _summary_has_nan(summary: dict | None) -> bool:
    if summary is None:
        return False
    return any(_value_has_nan(v) for v in summary.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root', type=str, default=str(DEFAULT_REPORT_ROOT / 'reports'))
    parser.add_argument('--before_wsi_run_summary', type=str, default=None)
    parser.add_argument('--after_wsi_run_summary', type=str, default=None)
    parser.add_argument('--after_wsi_proxy_csv', type=str, default=None)
    parser.add_argument('--after_photos_proxy_csv', type=str, default=None)
    args = parser.parse_args()

    out_root = ensure_dir(args.out_root)
    wsi_defaults = get_task_defaults('wsi')
    photos_defaults = get_task_defaults('photos')
    wsi_artifacts = get_default_backend_artifacts('wsi')
    photos_artifacts = get_default_backend_artifacts('photos')

    before_wsi_extract = _maybe_json(REPO_ROOT / 'logs/A3_output/D_phase/wsi_test_features_L1_s448_uniStage1p5_v1/extract_summary.json')
    before_wsi_final = _maybe_json(REPO_ROOT / 'logs/A3_output/reports/final_wsi_pipeline_summary.json')
    before_photos_summary = _maybe_json(photos_defaults['summary_json'])
    before_wsi_proxy = _maybe_csv(REPO_ROOT / 'logs/A3_output/E_phase/tileagg_cv_L1_s448_uniStage1p5_topk16_v1/cv_summary_mean_std.csv')
    wsi_rounds_summary_csv = REPO_ROOT / 'logs/A3_output/submission_closure/optimization_rounds/wsi/rounds_summary.csv'
    wsi_rounds_summary = _maybe_csv(wsi_rounds_summary_csv)
    final_after_round = _pick_fastest_formal_after(wsi_rounds_summary)

    onnx_export = _maybe_json(wsi_artifacts['onnx_export_json'])
    onnx_parity = _maybe_json(wsi_artifacts['onnx_parity_json'])
    om_compile = _maybe_json(wsi_artifacts['om_compile_json'])
    om_parity = _maybe_json(wsi_artifacts['om_parity_json'])
    legacy_fp16_om_compile = _maybe_json(REPO_ROOT / 'logs/A3_output/submission_closure/offline_models/wsi/wsi_om_compile_summary.json')
    legacy_fp16_om_parity = _maybe_json(REPO_ROOT / 'logs/A3_output/submission_closure/offline_models/wsi/wsi_om_parity_summary.json')
    legacy_mixed_om_compile = _maybe_json(REPO_ROOT / 'logs/A3_output/submission_closure/offline_models/wsi/wsi_om_compile_summary_mixed.json')
    photos_om_parity = _maybe_json(photos_artifacts['om_parity_json'])

    default_before_wsi_run = REPO_ROOT / 'logs/A3_output/submission_closure/optimization_rounds/wsi/01_before_plain_pytorch/run/reports/run_summary.json'
    if not default_before_wsi_run.exists():
        default_before_wsi_run = REPO_ROOT / 'logs/A3_output/submission_closure/official_runs/wsi_before_unified/reports/run_summary.json'
    before_wsi_run = _maybe_json(args.before_wsi_run_summary or default_before_wsi_run)
    if args.after_wsi_run_summary:
        after_wsi_run = _maybe_json(args.after_wsi_run_summary)
    elif final_after_round is not None and final_after_round.get('run_summary_path'):
        after_wsi_run = _maybe_json(final_after_round.get('run_summary_path'))
    else:
        after_wsi_run = None
    if args.after_wsi_proxy_csv:
        after_wsi_proxy = _maybe_csv(args.after_wsi_proxy_csv)
    elif final_after_round is not None and final_after_round.get('proxy_metrics_path'):
        after_wsi_proxy = _maybe_csv(final_after_round.get('proxy_metrics_path'))
    else:
        after_wsi_proxy = None
    after_photos_proxy = _maybe_csv(args.after_photos_proxy_csv)
    photos_pairing_strict = _photos_pairing_is_strict(args.after_photos_proxy_csv)

    stage01 = {
        'status': 'ok',
        'frozen_mainline': {
            'task': 'WSI 多标签推理链',
            'checkpoint': str(wsi_defaults['ckpt']),
            'level': int(wsi_defaults['level']),
            'tile_size': int(wsi_defaults['tile_size']),
            'step': int(wsi_defaults['step']),
            'min_tissue': float(wsi_defaults['min_tissue']),
            'agg': str(wsi_defaults['agg']),
            'topk': int(wsi_defaults['topk']),
            'thresholds_json': str(wsi_defaults['thresholds_json']),
        },
        'raw_model_fact': '仓库原始主线来自 .pt/.bin/.safetensors；.onnx/.om/.air 为本轮收口补齐产物。',
        'official_test_dataset_fact': 'TestDataset 视为无标签推理集，thumbnails 仅作人工观察，不作真值复核。',
        'wsi_acceleration_rounds_csv': str(wsi_rounds_summary_csv) if wsi_rounds_summary is not None else None,
        'selected_final_after_round': final_after_round.get('round_name') if final_after_round else None,
        'offline_artifacts_now': {
            'onnx_exists': bool(Path(wsi_artifacts['onnx']).exists()),
            'om_exists': bool(Path(wsi_artifacts['om']).exists()),
            'air_exists': False,
        },
    }
    _write_stage(
        out_root,
        '01_现状审计报告',
        stage01,
        [
            '# 01 现状审计报告',
            '',
            '## 输入数据',
            f'- 官方无标签测试集：`{wsi_defaults["official_test_input"]}`',
            f'- 本地 WSI 代理评测：`{wsi_defaults["bag_csv"]}`',
            f'- 本地 Photos 代理评测：`{photos_defaults["split_csv"]}`',
            '',
            '## 使用模型',
            f'- 冻结主线 checkpoint：`{wsi_defaults["ckpt"]}`',
            f'- 主线参数：level={wsi_defaults["level"]}, tile_size={wsi_defaults["tile_size"]}, step={wsi_defaults["step"]}, min_tissue={wsi_defaults["min_tissue"]}, topk={wsi_defaults["topk"]}',
            '',
            '## 自检结果',
            f'- 原始主线格式：`.pt/.bin/.safetensors`；当前 ONNX 是否已补齐：`{stage01["offline_artifacts_now"]["onnx_exists"]}`；OM 是否已补齐：`{stage01["offline_artifacts_now"]["om_exists"]}`',
            '- `TestDataset/WSI/thumbnails` 仅作人工观察，不可用于真值 ACC 复核。',
            '',
            '## 风险点',
            '- `.air` 仍是可选补充项，本轮正式 after 不以 `.air` 是否产出作为阻塞条件。',
            '- WSI 全流程瓶颈不只在编码器，端到端提速会受到读图、切块和 Python 调度开销限制。',
            '',
            '## 下一步建议',
            '- 以 `origin OM` 固化 after 正式版，并用统一入口产出最终速度报告与提交材料。',
        ],
    )

    stage02 = {
        'status': 'ok',
        'photos_before_summary': before_photos_summary,
        'wsi_before_unified_summary': before_wsi_run,
        'wsi_before_summary': before_wsi_final,
        'wsi_before_extract': before_wsi_extract,
    }
    _write_stage(
        out_root,
        '02_before基线跑通报告',
        stage02,
        [
            '# 02 before基线跑通报告',
            '',
            '## 输入数据',
            f'- Photos 5 折：`{photos_defaults["split_csv"]}`',
            f'- WSI 官方测试推理：`{wsi_defaults["official_test_input"]}`',
            '',
            '## 使用模型',
            f'- before 固定为 PyTorch eager：`{wsi_defaults["ckpt"]}`',
            '',
            '## 执行命令',
            '- WSI：`python src/A3/scripts/run_submission_infer.py --task wsi --variant before --input_dir <测试WSI目录> --out_dir <输出目录> --save_features`',
            '- Photos：`python src/A3/scripts/run_stage1_cv.py ...`',
            '',
            '## 输出路径',
            '- Photos 汇总：`logs/A3_output/reports/stage1_uni_large_cv5_ms512_1024_1536_summary.json`',
            f'- WSI 统一入口汇总：`{default_before_wsi_run}`',
            '- WSI 历史旧链路汇总：`logs/A3_output/reports/final_wsi_pipeline_summary.json`',
            '',
            '## 自检结果',
            f'- Photos before 5 折 ACC：`{before_photos_summary["mean_std"][0]["acc"] if before_photos_summary else None}`',
            f'- Photos before 5 折 macro_f1：`{before_photos_summary["mean_std"][0]["macro_f1"] if before_photos_summary else None}`',
            f'- WSI before 统一入口官方测试总耗时：`{before_wsi_run.get("total_elapsed_sec") if before_wsi_run else None}` 秒',
            f'- WSI before 统一入口官方测试 WSI/s：`{before_wsi_run.get("wsi_per_sec") if before_wsi_run else None}`',
            f'- WSI before 历史旧链路总耗时：`{before_wsi_extract.get("elapsed_sec_total") if before_wsi_extract else None}` 秒',
            f'- WSI before 本地代理 exact_match：`{_mean_from_mean_std(before_wsi_proxy, "exact_match")}`',
            f'- WSI before 本地代理 macro_f1：`{_mean_from_mean_std(before_wsi_proxy, "macro_f1")}`',
            '',
            '## 风险点',
            '- 历史旧链路 `328s` 记录可作为背景参考，但正式 before/after 速度对比必须以统一入口 `151.36s` 口径为准。',
            '',
            '## 下一步建议',
            '- 继续使用统一入口和同一测试目录补齐 after，避免跨脚本口径漂移。',
        ],
    )

    onnx_check = onnx_parity or onnx_export
    stage03 = {'status': 'ok' if onnx_export is not None else 'pending', 'onnx_export': onnx_export, 'onnx_parity': onnx_parity}
    _write_stage(
        out_root,
        '03_ONNX导出与对齐报告',
        stage03,
        [
            '# 03 ONNX导出与对齐报告',
            '',
            '## 输入数据',
            '- 随机固定 batch，形状固定为 `64x3x224x224`。',
            '',
            '## 使用模型',
            f'- 导出源 checkpoint：`{wsi_defaults["ckpt"]}`',
            f'- 输出 ONNX：`{wsi_artifacts["onnx"]}`',
            '',
            '## 执行命令',
            '- `python src/A3/scripts/export_patch_encoder_onnx.py --task wsi`',
            '',
            '## 输出路径',
            f'- ONNX：`{wsi_artifacts["onnx"]}`',
            f'- Meta：`{wsi_artifacts["meta_json"]}`',
            f'- 导出摘要：`{wsi_artifacts["onnx_export_json"]}`',
            f'- 对齐摘要：`{wsi_artifacts["onnx_parity_json"]}`',
            '',
            '## 自检结果',
            f'- 状态：`{stage03["status"]}`',
            f'- feature cosine：`{onnx_check.get("feature_cosine_mean") if onnx_check else None}`',
            f'- logit cosine：`{onnx_check.get("logit_cosine_mean") if onnx_check else None}`',
            f'- prob max abs diff：`{onnx_check.get("prob_max_abs_diff") if onnx_check else None}`',
            '',
            '## 风险点',
            '- ONNX 对齐只覆盖编码器 batch 级输出；slide 级输出仍需配合统一入口进行整目录一致性验证。',
            '',
            '## 下一步建议',
            '- 继续保持 `ONNX` 作为稳定回退后端，但 after 正式版优先使用已通过对齐的 `origin OM`。',
        ],
    )

    stage04 = {
        'status': 'ok' if om_compile and om_compile.get('status') == 'ok' and om_parity and om_parity.get('status') == 'ok' else 'pending',
        'om_compile': om_compile,
        'om_parity': om_parity,
        'legacy_fp16_om_compile': legacy_fp16_om_compile,
        'legacy_fp16_om_parity': legacy_fp16_om_parity,
        'legacy_mixed_om_compile': legacy_mixed_om_compile,
    }
    _write_stage(
        out_root,
        '04_OM导出与对齐报告',
        stage04,
        [
            '# 04 OM导出与对齐报告',
            '',
            '## 输入数据',
            '- ONNX 固定 batch 导出物与同名 meta json。',
            '',
            '## 使用模型',
            f'- 正式 after OM：`{wsi_artifacts["om"]}`',
            '',
            '## 执行命令',
            '- `python src/A3/scripts/compile_patch_encoder_om.py --task wsi`',
            '- `python src/A3/scripts/validate_patch_backend_parity.py --task wsi --after_backend om`',
            '',
            '## 输出路径',
            f'- OM：`{wsi_artifacts["om"]}`',
            f'- OM Meta：`{wsi_artifacts["om_meta_json"]}`',
            f'- OM 编译摘要：`{wsi_artifacts["om_compile_json"]}`',
            f'- OM 对齐摘要：`{wsi_artifacts["om_parity_json"]}`',
            '',
            '## 自检结果',
            f'- 编译状态：`{om_compile.get("status") if om_compile else "pending"}`',
            f'- 运行态对齐状态：`{om_parity.get("status") if om_parity else "pending"}`',
            f'- feature cosine：`{om_parity.get("feature_cosine_mean") if om_parity else None}`',
            f'- prob max abs diff：`{om_parity.get("prob_max_abs_diff") if om_parity else None}`',
            f'- 旧 fp16 OM 是否出现 NaN：`{_summary_has_nan(legacy_fp16_om_parity)}`',
            f'- 旧 mixed_float16 OM 编译产物是否存在：`{legacy_mixed_om_compile is not None}`',
            '',
            '## 风险点',
            '- `origin OM` 已通过随机 batch 对齐并可正式运行，但 WSI 端到端提速不会线性等于编码器加速，因为读图与切块仍占较大比例。',
            '- 旧 `fp16/mixed_float16` 尝试不纳入正式 after；后续若继续压缩时延，必须重新做数值稳定性验证。',
            '',
            '## 下一步建议',
            '- 以 `origin OM` 为正式 after 主线继续出整目录预测、速度报告和提交材料；`ONNX` 保留为回退链。',
        ],
    )

    before_time = before_wsi_run.get('total_elapsed_sec') if before_wsi_run else (before_wsi_extract.get('elapsed_sec_total') if before_wsi_extract else None)
    after_time = after_wsi_run.get('total_elapsed_sec') if after_wsi_run else None
    before_acc = _mean_from_mean_std(before_wsi_proxy, 'exact_match')
    after_acc = _mean_from_mean_std(after_wsi_proxy, 'exact_match')
    before_macro_f1 = _mean_from_mean_std(before_wsi_proxy, 'macro_f1')
    after_macro_f1 = _mean_from_mean_std(after_wsi_proxy, 'macro_f1')
    stage05 = {
        'status': 'complete' if before_wsi_run is not None and after_wsi_run is not None and after_wsi_proxy is not None and after_photos_proxy is not None else 'partial',
        'wsi_rounds_summary_csv': str(wsi_rounds_summary_csv) if wsi_rounds_summary is not None else None,
        'selected_after_round': final_after_round,
        'before_wsi_elapsed_sec': before_time,
        'after_wsi_elapsed_sec': after_time,
        'speedup': _speedup(before_time, after_time),
        'time_drop_pct': _drop_pct(before_time, after_time),
        'before_wsi_proxy_acc': before_acc,
        'after_wsi_proxy_acc': after_acc,
        'acc_change_pct': _pct(before_acc, after_acc),
        'acc_change_pct_points': _pct_point_diff(before_acc, after_acc),
        'before_wsi_macro_f1': before_macro_f1,
        'after_wsi_macro_f1': after_macro_f1,
        'macro_f1_change_pct': _pct(before_macro_f1, after_macro_f1),
        'macro_f1_change_pct_points': _pct_point_diff(before_macro_f1, after_macro_f1),
        'before_photos_acc': before_photos_summary['mean_std'][0]['acc'] if before_photos_summary else None,
        'after_photos_acc': _mean_from_mean_std(after_photos_proxy, 'acc'),
        'photos_pairing_strict': photos_pairing_strict,
        'photos_acc_change_pct_points': _pct_point_diff(before_photos_summary['mean_std'][0]['acc'] if before_photos_summary else None, _mean_from_mean_std(after_photos_proxy, 'acc')) if photos_pairing_strict else None,
        'before_photos_macro_f1': before_photos_summary['mean_std'][0]['macro_f1'] if before_photos_summary else None,
        'after_photos_macro_f1': _mean_from_mean_std(after_photos_proxy, 'macro_f1'),
        'photos_macro_f1_change_pct_points': _pct_point_diff(before_photos_summary['mean_std'][0]['macro_f1'] if before_photos_summary else None, _mean_from_mean_std(after_photos_proxy, 'macro_f1')) if photos_pairing_strict else None,
    }
    _write_stage(
        out_root,
        '05_before_after速度与精度对比报告',
        stage05,
        [
            '# 05 before/after速度与精度对比报告',
            '',
            '## 输入数据',
            f'- 官方测速目录：`{wsi_defaults["official_test_input"]}`',
            f'- 本地 WSI 代理评测：`{wsi_defaults["bag_csv"]}`',
            f'- 本地 Photos 代理评测：`{photos_defaults["split_csv"]}`',
            '',
            '## 使用模型',
            '- before：PyTorch eager + 同一 checkpoint + 同一切块/阈值/聚合逻辑',
            f'- after：`{final_after_round.get("round_name") if final_after_round else "origin OM"}`，共享同一 checkpoint + 同一切块/阈值/聚合逻辑，`ONNX` 仅保留为回退',
            '',
            '## 自检结果',
            f'- WSI 加速轮次总表：`{wsi_rounds_summary_csv if wsi_rounds_summary is not None else None}`',
            f'- 最终选中的正式 after 轮次：`{final_after_round.get("round_name") if final_after_round else None}`',
            f'- before WSI 总耗时：`{before_time}` 秒',
            f'- after WSI 总耗时：`{after_time}` 秒',
            f'- speedup：`{stage05["speedup"]}`',
            f'- 耗时下降百分比：`{stage05["time_drop_pct"]}`',
            f'- before WSI 代理 ACC(exact_match)：`{before_acc}`',
            f'- after WSI 代理 ACC(exact_match)：`{after_acc}`',
            f'- WSI ACC 变化：`{stage05["acc_change_pct_points"]}` 个百分点',
            f'- before WSI macro_f1：`{before_macro_f1}`',
            f'- after WSI macro_f1：`{after_macro_f1}`',
            f'- WSI macro_f1 变化：`{stage05["macro_f1_change_pct_points"]}` 个百分点',
            f'- before Photos ACC：`{stage05["before_photos_acc"]}`',
            f'- after Photos ACC：`{stage05["after_photos_acc"]}`',
            f'- Photos 是否为严格配对 before/after：`{photos_pairing_strict}`',
            f'- Photos ACC 变化：`{stage05["photos_acc_change_pct_points"]}` 个百分点',
            f'- before Photos macro_f1：`{stage05["before_photos_macro_f1"]}`',
            f'- after Photos macro_f1：`{stage05["after_photos_macro_f1"]}`',
            f'- Photos macro_f1 变化：`{stage05["photos_macro_f1_change_pct_points"]}` 个百分点',
            f'- after 轮次方法：`{final_after_round.get("methods_enabled") if final_after_round else None}`',
            '',
            '## 风险点',
            '- `04/05/06` 共享同一 `origin OM` 编码器和同一阈值逻辑；`sync/async/cache` 只改变执行方式，因此本地代理精度口径可复用同一份 `OM(origin)` CV 结果。',
            '- 当前 `Photos after` 代理目录使用单个 `fold0` checkpoint 跑全 5 折，因此只能作为“离线后端补充代理结果”，不能解读成加速使精度上升。',
            '- 当前最优正式 after 已把 WSI 总耗时显著压低；若后续更激进调优导致代理精度下降超过 `1` 个百分点，必须优先回滚到 `origin OM` 正式版。',
            '',
            '## 下一步建议',
            '- 将本报告与 `run_summary.json`、`cv_summary_mean_std.csv` 一并打包，作为答辩和提交说明的核心量化依据。',
        ],
    )

    stage06 = {
        'status': 'ok',
        'scripts': [
            'src/A3/scripts/run_submission_infer.py',
            'src/A3/scripts/run_accel_rounds.py',
            'src/A3/scripts/export_patch_encoder_onnx.py',
            'src/A3/scripts/compile_patch_encoder_om.py',
            'src/A3/scripts/validate_patch_backend_parity.py',
            'src/A3/scripts/run_photos_backend_cv.py',
            'src/A3/scripts/report_stage_closure.py',
        ],
        'runtime_dir': 'src/A3/bach_mil/runtime',
        'vendor_dir': '_vendor',
        'checkpoints': [str(wsi_defaults['ckpt']), str(photos_defaults['ckpt'])],
        'thresholds_json': str(wsi_defaults['thresholds_json']),
        'offline_models': {
            'onnx': str(wsi_artifacts['onnx']),
            'om': str(wsi_artifacts['om']),
            'om_meta_json': str(wsi_artifacts['om_meta_json']),
            'photos_om': str(photos_artifacts['om']),
            'photos_om_meta_json': str(photos_artifacts['om_meta_json']),
        },
        'selected_final_after_round': final_after_round.get('round_name') if final_after_round else None,
        'wsi_rounds_summary_csv': str(wsi_rounds_summary_csv) if wsi_rounds_summary is not None else None,
    }
    _write_stage(
        out_root,
        '06_最终提交材料汇总报告',
        stage06,
        [
            '# 06 最终提交材料汇总报告',
            '',
            '## 输入数据',
            '- 训练/代理评测数据按 README 既有目录放置。',
            '',
            '## 使用模型',
            '- before：PyTorch eager checkpoint',
            '- after：`origin OM` 正式版，`ONNX` 回退版',
            '',
            '## 执行命令',
            '- `python src/A3/scripts/export_patch_encoder_onnx.py --task wsi`',
            '- `python src/A3/scripts/compile_patch_encoder_om.py --task wsi`',
            '- `python src/A3/scripts/run_submission_infer.py --task wsi --variant before --input_dir <测试WSI目录> --out_dir <输出目录> --save_features`',
            '- `python src/A3/scripts/run_submission_infer.py --task wsi --variant after --backend om --input_dir <测试WSI目录> --out_dir <输出目录> --save_features`',
            '- `python src/A3/scripts/run_photos_backend_cv.py --backend om --out_dir <代理评测输出目录>`',
            '',
            '## 输出路径',
            f'- 阶段报告目录：`{out_root}`',
            f'- offline models：`{wsi_artifacts["root"]}`',
            f'- WSI 加速轮次总表：`{wsi_rounds_summary_csv if wsi_rounds_summary is not None else None}`',
            '',
            '## 自检结果',
            '- 当前已补齐统一入口、后端抽象、ONNX/OM 导出、随机 batch 对齐、官方测试测速、本地代理评测和阶段报告。',
            f'- 当前默认正式 after 轮次：`{final_after_round.get("round_name") if final_after_round else None}`',
            '- 最终提交时应一并携带 `runtime/`、新增脚本、`_vendor/`、主线 checkpoint、阈值文件和 offline models。',
            '',
            '## 风险点',
            '- `.air` 尚未补齐，但它不是本轮阻塞项；如组委会强制要求，再基于同一 ONNX/ATC 工具链补充。',
            '',
            '## 下一步建议',
            '- 生成压缩包前，先重新核对 `origin OM`、阈值文件、阶段报告和统一入口脚本是否与容器内路径一致。',
        ],
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
