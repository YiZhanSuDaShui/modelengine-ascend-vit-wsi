from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from bach_mil.utils.metrics import multilabel_metrics, search_thresholds  # noqa: E402


def _run_one_fold(
    *,
    script_path: Path,
    bag_csv: str,
    feature_dir: str,
    out_dir: Path,
    label_names: list[str],
    num_folds: int,
    fold: int,
    feature_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_instances: int,
    instance_sampling: str,
    seed: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script_path),
        '--bag_csv',
        str(bag_csv),
        '--feature_dir',
        str(feature_dir),
        '--out_dir',
        str(out_dir),
        '--num_folds',
        str(num_folds),
        '--fold',
        str(fold),
        '--feature_dim',
        str(feature_dim),
        '--hidden_dim',
        str(hidden_dim),
        '--dropout',
        str(dropout),
        '--epochs',
        str(epochs),
        '--lr',
        str(lr),
        '--weight_decay',
        str(weight_decay),
        '--max_instances',
        str(max_instances),
        '--instance_sampling',
        str(instance_sampling),
        '--seed',
        str(seed),
        '--save_val_predictions',
        '--label_names',
        *label_names,
    ]

    log_path = out_dir / 'train.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(cmd) + '\n')
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Fold {fold} failed. See log: {log_path}')


def _read_val_preds(p: Path, label_names: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(p)
    slide_ids = df.slide_id.astype(str).tolist()
    y_true = np.stack([df[f'true_{k}'].astype(np.int64).to_numpy() for k in label_names], axis=1)
    y_prob = np.stack([df[f'prob_{k}'].astype(np.float32).to_numpy() for k in label_names], axis=1)
    return slide_ids, y_true, y_prob


def _write_ensemble_preds(out_csv: Path, slide_ids: list[str], label_names: list[str], y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray):
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
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_csv', type=str, required=True)
    parser.add_argument('--feature_dir_a', type=str, required=True)
    parser.add_argument('--feature_dir_b', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=['Benign', 'InSitu', 'Invasive'])
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_instances', type=int, default=1024)
    parser.add_argument('--instance_sampling', type=str, default='hybrid', choices=['random', 'tile_prior', 'hybrid'])
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    label_names = list(args.label_names)

    train_script = Path(__file__).resolve().parent / 'train_mil_stage2.py'

    rows = []
    for fold in range(int(args.num_folds)):
        a_dir = out_root / f'a_fold{fold}'
        b_dir = out_root / f'b_fold{fold}'
        _run_one_fold(
            script_path=train_script,
            bag_csv=args.bag_csv,
            feature_dir=args.feature_dir_a,
            out_dir=a_dir,
            label_names=label_names,
            num_folds=int(args.num_folds),
            fold=fold,
            feature_dim=int(args.feature_dim),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            max_instances=int(args.max_instances),
            instance_sampling=str(args.instance_sampling),
            seed=int(args.seed),
        )
        _run_one_fold(
            script_path=train_script,
            bag_csv=args.bag_csv,
            feature_dir=args.feature_dir_b,
            out_dir=b_dir,
            label_names=label_names,
            num_folds=int(args.num_folds),
            fold=fold,
            feature_dim=int(args.feature_dim),
            hidden_dim=int(args.hidden_dim),
            dropout=float(args.dropout),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            max_instances=int(args.max_instances),
            instance_sampling=str(args.instance_sampling),
            seed=int(args.seed),
        )

        # Ensemble (avg probs on the same val fold)
        a_val = a_dir / 'val_predictions_best.csv'
        b_val = b_dir / 'val_predictions_best.csv'
        slide_a, y_true_a, y_prob_a = _read_val_preds(a_val, label_names)
        slide_b, y_true_b, y_prob_b = _read_val_preds(b_val, label_names)
        if slide_a != slide_b:
            raise RuntimeError('Val slide_id order mismatch between scales; cannot ensemble safely.')
        if not np.array_equal(y_true_a, y_true_b):
            raise RuntimeError('Val y_true mismatch between scales; check bag_csv / splits.')
        slide_ids = slide_a
        y_true = y_true_a
        y_prob_avg = 0.5 * (y_prob_a + y_prob_b)

        thr_opt = search_thresholds(y_true, y_prob_avg)
        m_opt = multilabel_metrics(y_true, y_prob_avg, thresholds=thr_opt)
        thr_05 = np.full((len(label_names),), 0.5, dtype=np.float32)
        m_05 = multilabel_metrics(y_true, y_prob_avg, thresholds=thr_05)

        ens_dir = out_root / f'ensemble_fold{fold}'
        ens_dir.mkdir(parents=True, exist_ok=True)
        with open(ens_dir / 'thresholds_opt.json', 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in zip(label_names, thr_opt.tolist())}, f, indent=2)
        _write_ensemble_preds(ens_dir / 'val_predictions_opt.csv', slide_ids, label_names, y_true, y_prob_avg, thr_opt)
        _write_ensemble_preds(ens_dir / 'val_predictions_05.csv', slide_ids, label_names, y_true, y_prob_avg, thr_05)

        ckpt_a = torch.load(a_dir / 'best.pt', map_location='cpu')
        ckpt_b = torch.load(b_dir / 'best.pt', map_location='cpu')
        m_a = ckpt_a.get('metrics', {})
        m_b = ckpt_b.get('metrics', {})
        row = {
            'fold': int(fold),
            'a_macro_f1': float(m_a.get('macro_f1', float('nan'))),
            'a_exact_match': float(m_a.get('exact_match', float('nan'))),
            'b_macro_f1': float(m_b.get('macro_f1', float('nan'))),
            'b_exact_match': float(m_b.get('exact_match', float('nan'))),
            'ensemble_macro_f1_opt': float(m_opt['macro_f1']),
            'ensemble_exact_match_opt': float(m_opt['exact_match']),
            'ensemble_macro_f1_05': float(m_05['macro_f1']),
            'ensemble_exact_match_05': float(m_05['exact_match']),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_root / 'ensemble_cv_summary_partial.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'ensemble_cv_summary.csv', index=False)
    agg_cols = [
        'a_macro_f1',
        'a_exact_match',
        'b_macro_f1',
        'b_exact_match',
        'ensemble_macro_f1_opt',
        'ensemble_exact_match_opt',
        'ensemble_macro_f1_05',
        'ensemble_exact_match_05',
    ]
    agg = df[agg_cols].agg(['mean', 'std']).reset_index()
    agg.to_csv(out_root / 'ensemble_cv_summary_mean_std.csv', index=False)
    print(df)
    print(agg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
