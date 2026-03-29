from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def _read_best_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / 'metrics.json'
    with open(metrics_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    history = d.get('history', [])
    best = max(history, key=lambda r: float(r.get('macro_f1', -1)))
    keys = ['epoch', 'macro_f1', 'micro_f1', 'sample_f1', 'exact_match', 'macro_ap', 'train_loss']
    return {k: best.get(k) for k in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_csv', type=str, required=True)
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--label_names', nargs='+', default=['Benign', 'InSitu', 'Invasive'])
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--feature_dim', type=int, default=1024)
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

    script = Path(__file__).resolve().parent / 'train_mil_stage2.py'
    rows = []
    for fold in range(int(args.num_folds)):
        run_dir = out_root / f'fold{fold}'
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            '--bag_csv',
            str(args.bag_csv),
            '--feature_dir',
            str(args.feature_dir),
            '--out_dir',
            str(run_dir),
            '--num_folds',
            str(args.num_folds),
            '--fold',
            str(fold),
            '--feature_dim',
            str(args.feature_dim),
            '--hidden_dim',
            str(args.hidden_dim),
            '--dropout',
            str(args.dropout),
            '--epochs',
            str(args.epochs),
            '--lr',
            str(args.lr),
            '--weight_decay',
            str(args.weight_decay),
            '--max_instances',
            str(args.max_instances),
            '--instance_sampling',
            str(args.instance_sampling),
            '--seed',
            str(args.seed),
            '--save_val_predictions',
            '--label_names',
            *list(args.label_names),
        ]
        log_path = run_dir / 'train.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(cmd) + '\n')
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'Fold {fold} failed. See log: {log_path}')

        best = _read_best_metrics(run_dir)
        row = {'fold': fold, **best}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_root / 'cv_summary_partial.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / 'cv_summary.csv', index=False)
    agg_cols = [c for c in df.columns if c != 'fold']
    agg = df[agg_cols].agg(['mean', 'std']).reset_index()
    agg.to_csv(out_root / 'cv_summary_mean_std.csv', index=False)
    print(df)
    print(agg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
