from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import ensure_dir, load_json, save_json  # noqa: E402


LABELS = ['Normal', 'Benign', 'InSitu', 'Invasive']
ACTIVE_LABELS = ['Benign', 'InSitu', 'Invasive']
ROI_COLUMNS = ['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']


@dataclass
class EvalResult:
    name: str
    default_thresholds: dict[str, float]
    default_metrics: dict
    tuned_thresholds: dict[str, float]
    tuned_metrics: dict
    searched_combos: int
    same_default_tuned: bool
    details: pd.DataFrame


def _parse_named_path(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if '=' not in item:
            raise SystemExit(f'参数格式错误，期望 name=path，收到：{item}')
        name, path = item.split('=', 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise SystemExit(f'参数格式错误，期望 name=path，收到：{item}')
        out[name] = Path(path)
    return out


def _load_truth_from_bracs(manifest_csv: Path, bracs_xlsx: Path) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    manifest_df = pd.read_csv(manifest_csv)
    if 'stem' not in manifest_df.columns:
        raise SystemExit(f'manifest_csv 缺少 stem 列：{manifest_csv}')

    roi_df = pd.read_excel(bracs_xlsx, sheet_name='WSI_with_RoI_Distribution', header=0)
    roi_df.columns = ['WSI Filename', 'WSI label', 'Set', 'Total RoI number', 'N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']
    if str(roi_df.iloc[0, 0]).strip() == 'WSI Filename':
        roi_df = roi_df.iloc[1:].copy()
    for col in ROI_COLUMNS:
        roi_df[col] = roi_df[col].replace('-', 0).fillna(0).astype(int)

    need = manifest_df['stem'].astype(str).tolist()
    roi_df = roi_df[roi_df['WSI Filename'].astype(str).isin(need)].copy()
    if len(roi_df) != len(need):
        missing = sorted(set(need) - set(roi_df['WSI Filename'].astype(str).tolist()))
        raise SystemExit(f'BRACS.xlsx 缺少以下 WSI：{missing}')

    truth_rows = []
    y_true = []
    for slide_id in need:
        row = roi_df[roi_df['WSI Filename'].astype(str) == str(slide_id)].iloc[0]
        truth = {
            'Normal': int(int(row['N']) > 0),
            'Benign': int(int(row['PB']) > 0 or int(row['UDH']) > 0),
            'InSitu': int(int(row['FEA']) > 0 or int(row['ADH']) > 0 or int(row['DCIS']) > 0),
            'Invasive': int(int(row['IC']) > 0),
        }
        truth_rows.append({
            'slide_id': str(slide_id),
            'wsi_label': str(row['WSI label']),
            'roi_N': int(row['N']),
            'roi_PB': int(row['PB']),
            'roi_UDH': int(row['UDH']),
            'roi_FEA': int(row['FEA']),
            'roi_ADH': int(row['ADH']),
            'roi_DCIS': int(row['DCIS']),
            'roi_IC': int(row['IC']),
            'true_labels': ';'.join([name for name in LABELS if truth[name] == 1]),
        })
        y_true.append([truth[name] for name in LABELS])

    truth_df = pd.DataFrame(truth_rows)
    return need, np.asarray(y_true, dtype=np.int64), truth_df


def _labels_from_binary(y_bin: np.ndarray) -> list[str]:
    out = []
    for row in y_bin:
        names = [LABELS[i] for i, v in enumerate(row) if int(v) == 1]
        out.append(';'.join(names))
    return out


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'exact_match': float(np.all(y_true == y_pred, axis=1).mean()),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'sample_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'per_class_f1': {
            LABELS[i]: float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
            for i in range(len(LABELS))
        },
    }


def _make_pred_from_thresholds(probs: np.ndarray, thresholds: dict[str, float], normal_fallback: bool) -> np.ndarray:
    y_pred = np.zeros((probs.shape[0], len(LABELS)), dtype=np.int64)
    for j, name in enumerate(ACTIVE_LABELS):
        y_pred[:, j + 1] = (probs[:, j] >= float(thresholds[name])).astype(np.int64)
    if normal_fallback:
        empty = (y_pred[:, 1:].sum(axis=1) == 0)
        y_pred[empty, 0] = 1
    return y_pred


def _search_best_thresholds(y_true: np.ndarray, probs: np.ndarray, normal_fallback: bool) -> tuple[dict[str, float], np.ndarray, int]:
    candidates: list[list[float]] = []
    for j in range(probs.shape[1]):
        vals = np.unique(probs[:, j])
        mids = [0.0]
        for i, v in enumerate(vals):
            mids.append(float(v))
            if i < len(vals) - 1:
                mids.append(float((v + vals[i + 1]) / 2.0))
        mids.append(1.0)
        candidates.append(sorted(set(round(x, 12) for x in mids)))

    best_exact = -1.0
    best_macro = -1.0
    best_thr: dict[str, float] | None = None
    best_pred: np.ndarray | None = None
    combos = 0

    for tb in candidates[0]:
        bmask = (probs[:, 0] >= tb).astype(np.int64)
        for ti in candidates[1]:
            imask = (probs[:, 1] >= ti).astype(np.int64)
            bi = bmask + imask
            for tv in candidates[2]:
                vmask = (probs[:, 2] >= tv).astype(np.int64)
                y_pred = np.zeros((probs.shape[0], len(LABELS)), dtype=np.int64)
                y_pred[:, 1] = bmask
                y_pred[:, 2] = imask
                y_pred[:, 3] = vmask
                if normal_fallback:
                    empty = ((bi + vmask) == 0)
                    y_pred[empty, 0] = 1

                exact = float(np.all(y_true == y_pred, axis=1).mean())
                if exact < best_exact:
                    combos += 1
                    continue
                macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
                if exact > best_exact or macro > best_macro:
                    best_exact = exact
                    best_macro = macro
                    best_thr = {
                        'Benign': float(tb),
                        'InSitu': float(ti),
                        'Invasive': float(tv),
                    }
                    best_pred = y_pred.copy()
                combos += 1

    if best_thr is None or best_pred is None:
        raise RuntimeError('阈值搜索失败，没有找到可用结果')
    return best_thr, best_pred, combos


def _load_summary_map(items: list[str]) -> dict[str, dict]:
    if not items:
        return {}
    out = {}
    for name, path in _parse_named_path(items).items():
        out[name] = load_json(path)
    return out


def _evaluate_one(
    *,
    name: str,
    pred_csv: Path,
    slide_ids: list[str],
    y_true: np.ndarray,
    truth_df: pd.DataFrame,
    default_thresholds: dict[str, float],
    normal_fallback: bool,
) -> EvalResult:
    pred_df = pd.read_csv(pred_csv)
    if 'slide_id' not in pred_df.columns:
        raise SystemExit(f'预测 CSV 缺少 slide_id 列：{pred_csv}')
    for name_i in ACTIVE_LABELS:
        col = f'prob_{name_i}'
        if col not in pred_df.columns:
            raise SystemExit(f'预测 CSV 缺少概率列 {col}：{pred_csv}')

    pred_df = pred_df.set_index('slide_id').loc[slide_ids].reset_index()
    probs = np.stack([pred_df[f'prob_{name_i}'].to_numpy(dtype=np.float64) for name_i in ACTIVE_LABELS], axis=1)

    default_pred = _make_pred_from_thresholds(probs, default_thresholds, normal_fallback=normal_fallback)
    default_metrics = _evaluate(y_true, default_pred)

    tuned_thresholds, tuned_pred, combos = _search_best_thresholds(y_true, probs, normal_fallback=normal_fallback)
    tuned_metrics = _evaluate(y_true, tuned_pred)

    detail_df = truth_df[['slide_id', 'true_labels']].copy()
    detail_df[f'{name}_default_pred'] = _labels_from_binary(default_pred)
    detail_df[f'{name}_tuned_pred'] = _labels_from_binary(tuned_pred)
    detail_df[f'{name}_default_exact'] = (default_pred == y_true).all(axis=1).astype(int)
    detail_df[f'{name}_tuned_exact'] = (tuned_pred == y_true).all(axis=1).astype(int)

    return EvalResult(
        name=name,
        default_thresholds=dict(default_thresholds),
        default_metrics=default_metrics,
        tuned_thresholds=tuned_thresholds,
        tuned_metrics=tuned_metrics,
        searched_combos=combos,
        same_default_tuned=bool(np.array_equal(default_pred, tuned_pred)),
        details=detail_df,
    )


def _render_markdown(
    *,
    out_path: Path,
    summary: dict,
    merged_df: pd.DataFrame,
) -> None:
    speed_section = []
    speed_summary = summary.get('speed_summary')
    if speed_summary is not None:
        speed_section = [
            '## 2. 速度结果',
            '',
            f"- before 总耗时：`{speed_summary['before_total_elapsed_sec']:.4f} s`",
            f"- after 总耗时：`{speed_summary['after_total_elapsed_sec']:.4f} s`",
            f"- 总体加速比：`{speed_summary['speedup_x']:.4f}x`",
            f"- 总耗时下降：`{speed_summary['time_drop_pct']:.4f}%`",
            f"- before 吞吐：`{speed_summary['before_tiles_per_sec']:.4f} tiles/s`",
            f"- after 吞吐：`{speed_summary['after_tiles_per_sec']:.4f} tiles/s`",
            f"- tiles/s 提升：`{speed_summary['tiles_gain_pct']:.4f}%`",
            '',
        ]

    lines = [
        '# BRACS ROI 多标签验证报告',
        '',
        '## 1. 评测口径',
        '',
        '- 真值来自 `BRACS.xlsx` 的 `WSI_with_RoI_Distribution` 工作表',
        '- 4 类折叠规则：',
        '  - `Normal = N > 0`',
        '  - `Benign = PB > 0 or UDH > 0`',
        '  - `InSitu = FEA > 0 or ADH > 0 or DCIS > 0`',
        '  - `Invasive = IC > 0`',
        '- 该口径用于跨数据集代理验证，不是 BRACS 官方原生四分类协议',
        '',
    ]
    lines.extend(speed_section)
    lines.extend([
        '## 3. 默认阈值结果',
        '',
        f"- before exact_match：`{summary['before']['default_metrics']['exact_match']:.4f}`",
        f"- after exact_match：`{summary['after']['default_metrics']['exact_match']:.4f}`",
        f"- before macro_f1：`{summary['before']['default_metrics']['macro_f1']:.4f}`",
        f"- after macro_f1：`{summary['after']['default_metrics']['macro_f1']:.4f}`",
        f"- before/after 默认预测一致率：`{summary['consistency']['default_pred_same_rate']:.4f}`",
        '',
        '## 4. 阈值搜索结果',
        '',
        f"- before tuned exact_match：`{summary['before']['tuned_metrics']['exact_match']:.4f}`",
        f"- after tuned exact_match：`{summary['after']['tuned_metrics']['exact_match']:.4f}`",
        f"- before tuned macro_f1：`{summary['before']['tuned_metrics']['macro_f1']:.4f}`",
        f"- after tuned macro_f1：`{summary['after']['tuned_metrics']['macro_f1']:.4f}`",
        f"- before tuned 阈值：`{summary['before']['tuned_thresholds']}`",
        f"- after tuned 阈值：`{summary['after']['tuned_thresholds']}`",
        '',
        '## 5. 逐张结果',
        '',
        merged_df.to_markdown(index=False),
        '',
        '## 6. 说明',
        '',
        '- 如果 tuned 阈值是在同一批外部样本上直接搜索得到的，那么它更适合做“分析参考”，不建议直接替换正式提交阈值。',
        '- 如果后续继续下载新的 BRACS 子集，推荐把外部样本拆成“小验证集 / 小测试集”，先用验证集选阈值，再用测试集看泛化。',
    ])
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_csv', type=str, required=True)
    parser.add_argument('--bracs_xlsx', type=str, required=True)
    parser.add_argument('--pred_csv', nargs='+', required=True, help='格式：before=path after=path')
    parser.add_argument('--run_summary_json', nargs='*', default=None, help='格式：before=path after=path')
    parser.add_argument('--thresholds_json', type=str, default=None)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--default_threshold', type=float, default=1.0)
    parser.add_argument('--normal_fallback', action='store_true')
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    pred_map = _parse_named_path(args.pred_csv)
    summary_map = _load_summary_map(args.run_summary_json or [])

    default_thresholds = {name: float(args.default_threshold) for name in ACTIVE_LABELS}
    if args.thresholds_json is not None and Path(args.thresholds_json).exists():
        thr_json = load_json(args.thresholds_json)
        for name in ACTIVE_LABELS:
            if name in thr_json:
                default_thresholds[name] = float(thr_json[name])

    slide_ids, y_true, truth_df = _load_truth_from_bracs(Path(args.manifest_csv), Path(args.bracs_xlsx))
    truth_df.to_csv(out_dir / 'bracs_truth_multilabel.csv', index=False)

    results: dict[str, EvalResult] = {}
    merged_df = truth_df[['slide_id', 'true_labels']].copy()
    for name, pred_csv in pred_map.items():
        res = _evaluate_one(
            name=name,
            pred_csv=pred_csv,
            slide_ids=slide_ids,
            y_true=y_true,
            truth_df=truth_df,
            default_thresholds=default_thresholds,
            normal_fallback=bool(args.normal_fallback),
        )
        results[name] = res
        merged_df = merged_df.merge(res.details, on=['slide_id', 'true_labels'], how='left')
        save_json(res.tuned_thresholds, out_dir / f'{name}_tuned_thresholds.json')

    default_same_rate = None
    tuned_same_rate = None
    names = list(results.keys())
    if len(names) >= 2:
        a = names[0]
        b = names[1]
        default_same_rate = float((merged_df[f'{a}_default_pred'] == merged_df[f'{b}_default_pred']).mean())
        tuned_same_rate = float((merged_df[f'{a}_tuned_pred'] == merged_df[f'{b}_tuned_pred']).mean())

    speed_summary = None
    if 'before' in summary_map and 'after' in summary_map:
        before = summary_map['before']
        after = summary_map['after']
        speed_summary = {
            'before_total_elapsed_sec': float(before['total_elapsed_sec']),
            'after_total_elapsed_sec': float(after['total_elapsed_sec']),
            'before_tiles_per_sec': float(before['tiles_per_sec']),
            'after_tiles_per_sec': float(after['tiles_per_sec']),
            'speedup_x': float(before['total_elapsed_sec'] / after['total_elapsed_sec']),
            'time_drop_pct': float((before['total_elapsed_sec'] - after['total_elapsed_sec']) / before['total_elapsed_sec'] * 100.0),
            'tiles_gain_pct': float((after['tiles_per_sec'] - before['tiles_per_sec']) / before['tiles_per_sec'] * 100.0),
        }

    merged_df.to_csv(out_dir / 'bracs_roi_multilabel_eval.csv', index=False)

    summary = {
        name: {
            'default_thresholds': res.default_thresholds,
            'default_metrics': res.default_metrics,
            'tuned_thresholds': res.tuned_thresholds,
            'tuned_metrics': res.tuned_metrics,
            'searched_combos': res.searched_combos,
            'same_default_tuned': res.same_default_tuned,
        }
        for name, res in results.items()
    }
    summary['truth_mapping'] = {
        'Normal': 'N > 0',
        'Benign': 'PB > 0 or UDH > 0',
        'InSitu': 'FEA > 0 or ADH > 0 or DCIS > 0',
        'Invasive': 'IC > 0',
    }
    summary['default_thresholds_source'] = str(args.thresholds_json) if args.thresholds_json is not None else None
    summary['consistency'] = {
        'default_pred_same_rate': default_same_rate,
        'tuned_pred_same_rate': tuned_same_rate,
    }
    summary['speed_summary'] = speed_summary
    save_json(summary, out_dir / 'bracs_roi_multilabel_eval_summary.json')

    _render_markdown(
        out_path=out_dir / 'bracs_roi_multilabel_eval_report.md',
        summary=summary,
        merged_df=merged_df,
    )

    print(f'saved -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
