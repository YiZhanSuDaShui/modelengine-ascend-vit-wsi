from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.io import ensure_dir, save_json  # noqa: E402
from bach_mil.utils.metrics import multilabel_metrics, search_thresholds  # noqa: E402


def _align_probs(tile_probs: torch.Tensor, prob_classes: list[str], label_names: list[str]) -> torch.Tensor:
    if list(prob_classes) == list(label_names):
        return tile_probs
    idx = []
    for name in label_names:
        if name not in prob_classes:
            raise ValueError(f'missing class={name} in prob_classes={prob_classes}')
        idx.append(int(prob_classes.index(name)))
    return tile_probs[:, idx]


def _agg_probs(tile_probs: torch.Tensor, agg: str, topk: int) -> torch.Tensor:
    if tile_probs.ndim != 2:
        raise ValueError(f'tile_probs must be [N, C], got {tuple(tile_probs.shape)}')
    if tile_probs.shape[0] == 0:
        raise ValueError('tile_probs is empty')
    if agg == 'mean_prob':
        return tile_probs.mean(dim=0)
    if agg == 'max_prob':
        return tile_probs.max(dim=0).values
    if agg == 'topk_mean_prob':
        k = int(max(1, min(int(topk), int(tile_probs.shape[0]))))
        return torch.topk(tile_probs, k=k, dim=0).values.mean(dim=0)
    raise ValueError(f'unknown agg={agg}')


def _load_slide_probs(
    *,
    feature_dirs: list[Path],
    label_names: list[str],
    agg: str,
    topk: int,
) -> dict[str, np.ndarray]:
    slide_to_probs: dict[str, np.ndarray] = {}
    for feat_dir in feature_dirs:
        cur: dict[str, np.ndarray] = {}
        for pt_path in sorted(feat_dir.glob('*.pt')):
            blob = torch.load(pt_path, map_location='cpu')
            tile_probs = blob.get('tile_probs')
            if tile_probs is None:
                raise RuntimeError(f'{pt_path} missing tile_probs')
            tile_probs = tile_probs.float()
            prob_classes = blob.get('classes')
            if not prob_classes:
                raise RuntimeError(f'{pt_path} missing classes')
            aligned = _align_probs(tile_probs, list(prob_classes), label_names)
            cur[pt_path.stem] = _agg_probs(aligned, agg=str(agg), topk=int(topk)).cpu().numpy().astype(np.float32)
        if not slide_to_probs:
            slide_to_probs = cur
        else:
            if set(slide_to_probs.keys()) != set(cur.keys()):
                raise RuntimeError(
                    f'feature dirs have different slide sets: {feature_dirs[0]} vs {feat_dir}'
                )
            for sid in slide_to_probs.keys():
                slide_to_probs[sid] = slide_to_probs[sid] + cur[sid]
    if not slide_to_probs:
        raise RuntimeError(f'no .pt files found in {feature_dirs}')
    scale = float(len(feature_dirs))
    for sid in list(slide_to_probs.keys()):
        slide_to_probs[sid] = slide_to_probs[sid] / scale
    return slide_to_probs


def _thresholds_with_degenerate_guard(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    label_names: list[str],
    all_positive_threshold: float,
    all_negative_threshold: float,
) -> np.ndarray:
    thr = search_thresholds(y_true, y_prob)
    for i, _name in enumerate(label_names):
        s = int(y_true[:, i].sum())
        if s == 0:
            thr[i] = float(all_negative_threshold)
        elif s == int(len(y_true)):
            thr[i] = float(all_positive_threshold)
    return thr


def _pred_df(
    *,
    slide_ids: list[str],
    label_names: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    y_pred = (y_prob >= thresholds[None, :]).astype(np.int64)
    rows = []
    for i, sid in enumerate(slide_ids):
        row = {'slide_id': str(sid)}
        for ci, name in enumerate(label_names):
            row[f'true_{name}'] = int(y_true[i, ci])
            row[f'prob_{name}'] = float(y_prob[i, ci])
            row[f'pred_{name}'] = int(y_pred[i, ci])
        row['pred_labels'] = ';'.join([name for ci, name in enumerate(label_names) if int(y_pred[i, ci]) == 1])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_csv', type=str, required=True)
    parser.add_argument('--feature_dirs', nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=['Benign', 'InSitu', 'Invasive'])
    parser.add_argument('--agg', type=str, default='topk_mean_prob', choices=['mean_prob', 'max_prob', 'topk_mean_prob'])
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--all_positive_threshold', type=float, default=0.5)
    parser.add_argument('--all_negative_threshold', type=float, default=1.0)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    label_names = list(args.label_names)
    feature_dirs = [Path(p) for p in args.feature_dirs]

    bag_df = pd.read_csv(args.bag_csv).copy()
    for name in label_names:
        col = f'label_{name}'
        if col not in bag_df.columns:
            raise SystemExit(f'bag_csv missing column {col}')

    slide_to_probs = _load_slide_probs(
        feature_dirs=feature_dirs,
        label_names=label_names,
        agg=str(args.agg),
        topk=int(args.topk),
    )
    bag_df = bag_df[bag_df.slide_id.astype(str).isin(slide_to_probs.keys())].reset_index(drop=True)
    if bag_df.empty:
        raise SystemExit('no overlapping slide_ids between bag_csv and feature_dirs')

    n_splits = min(int(args.num_folds), len(bag_df))
    if n_splits < 2:
        raise SystemExit('need at least 2 slides for CV')

    rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(args.seed))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(bag_df)):
        tr_df = bag_df.iloc[tr_idx].reset_index(drop=True)
        va_df = bag_df.iloc[va_idx].reset_index(drop=True)

        train_slide_ids = tr_df.slide_id.astype(str).tolist()
        val_slide_ids = va_df.slide_id.astype(str).tolist()

        y_prob_tr = np.stack([slide_to_probs[sid] for sid in train_slide_ids], axis=0).astype(np.float32)
        y_true_tr = np.stack(
            [tr_df[f'label_{name}'].astype(np.int64).to_numpy() for name in label_names],
            axis=1,
        )
        thresholds = _thresholds_with_degenerate_guard(
            y_true=y_true_tr,
            y_prob=y_prob_tr,
            label_names=label_names,
            all_positive_threshold=float(args.all_positive_threshold),
            all_negative_threshold=float(args.all_negative_threshold),
        )

        y_prob_va = np.stack([slide_to_probs[sid] for sid in val_slide_ids], axis=0).astype(np.float32)
        y_true_va = np.stack(
            [va_df[f'label_{name}'].astype(np.int64).to_numpy() for name in label_names],
            axis=1,
        )
        metrics = multilabel_metrics(y_true_va, y_prob_va, thresholds=thresholds)
        row = {
            'fold': int(fold),
            'macro_f1': float(metrics['macro_f1']),
            'micro_f1': float(metrics['micro_f1']),
            'sample_f1': float(metrics['sample_f1']),
            'exact_match': float(metrics['exact_match']),
            'macro_ap': float(metrics['macro_ap']),
        }
        for ci, name in enumerate(label_names):
            row[f'auc_{name}'] = float(metrics['per_class_auc'][ci])
            row[f'ap_{name}'] = float(metrics['per_class_ap'][ci])
            row[f'threshold_{name}'] = float(thresholds[ci])
        rows.append(row)

        fold_dir = out_dir / f'fold{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json({k: float(v) for k, v in zip(label_names, thresholds.tolist())}, fold_dir / 'thresholds.json')
        _pred_df(
            slide_ids=val_slide_ids,
            label_names=label_names,
            y_true=y_true_va,
            y_prob=y_prob_va,
            thresholds=thresholds,
        ).to_csv(fold_dir / 'val_predictions.csv', index=False)

        pd.DataFrame(rows).to_csv(out_dir / 'cv_summary_partial.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'cv_summary.csv', index=False)
    agg_cols = [c for c in df.columns if c != 'fold']
    agg = df[agg_cols].agg(['mean', 'std']).reset_index()
    agg.to_csv(out_dir / 'cv_summary_mean_std.csv', index=False)

    meta = {
        'bag_csv': str(args.bag_csv),
        'feature_dirs': [str(p) for p in feature_dirs],
        'label_names': label_names,
        'agg': str(args.agg),
        'topk': int(args.topk),
        'num_folds': int(n_splits),
        'seed': int(args.seed),
    }
    save_json(meta, out_dir / 'config.json')
    print(df)
    print(agg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
